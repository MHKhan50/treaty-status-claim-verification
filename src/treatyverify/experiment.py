"""End-to-end experiment: build the corpus, run CPU baselines and agents, GPU verifiers, and summarise."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .agents import LookupAgent, ProvenanceAuditor, combine
from .baselines import fit_predict
from .corpus import LABELS, document_text, histories, make_claims, split_by_country, split_by_treaty
from .metrics import report, retrieval_metrics, risk_coverage
from .retrieval import EvidenceRetriever
from .sources import load_panel

SPLITS = ("country", "treaty")


def build(raw_dir, out_dir) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    hist = histories(load_panel(raw_dir))
    hist["text"] = hist.apply(document_text, axis=1)
    claims = make_claims(hist)
    claims["split_country"] = split_by_country(claims)
    claims["split_treaty"] = split_by_treaty(claims, "TPNW", claims["split_country"])
    hist.assign(spans=hist["spans"].map(json.dumps)).to_csv(out / "documents.csv", index=False)
    claims.to_csv(out / "claims.csv", index=False)
    summary = {"documents": int(len(hist)), "claims": int(len(claims)), "countries": int(claims["code"].nunique()),
               "labels": claims["label"].value_counts().to_dict(), "kinds": claims["kind"].value_counts().to_dict(),
               "split_country": claims["split_country"].value_counts().to_dict(),
               "split_treaty": claims["split_treaty"].value_counts().to_dict()}
    (out / "corpus_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load(out_dir):
    out = Path(out_dir)
    hist = pd.read_csv(out / "documents.csv")
    hist["spans"] = hist["spans"].map(lambda s: [tuple(x) for x in json.loads(s)])
    claims = pd.read_csv(out / "claims.csv", keep_default_na=False)
    return hist, claims


def _retrieve(hist, claims):
    r = EvidenceRetriever(hist["doc_id"], hist["text"])
    ranked, _ = r.rank(claims["claim"], top_k=5)
    text = dict(zip(hist["doc_id"], hist["text"]))
    return ranked, [text[ids[0]] for ids in ranked], text


def run_cpu(data_dir, out_dir, n_boot=1000) -> dict:
    hist, claims = load(data_dir)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ranked, top1, text = _retrieve(hist, claims)
    claims = claims.assign(top_ids=[";".join(r) for r in ranked], top1_text=top1,
                           gold_text=[text.get(g, "") for g in claims["gold_doc_id"]])
    res = {"retrieval_tfidf": retrieval_metrics(claims["gold_doc_id"], ranked)}
    lookup = LookupAgent(hist)
    claims["lookup"] = [lookup.verify(c)["label"] for c in claims["claim"]]
    preds = claims[["claim_id", "label", "code", "kind", "split_country", "split_treaty", "lookup"]].copy()
    for split in SPLITS:
        col = f"split_{split}"
        tr, va, te = (claims[claims[col] == s] for s in ("train", "val", "test"))
        res[split] = {"test_claims": int(len(te))}
        for kind, ev in (("majority", None), ("claim_lr", None), ("pair_lr", "top1_text"), ("pair_mlp", "top1_text"),
                         ("pair_lr_gold", "gold_text")):
            k = kind.replace("_gold", "")
            p, info = fit_predict(k, tr, va, te, *(x[ev] if ev else None for x in (tr, va, te)))
            res[split][kind] = {**report(te["label"], p, te["code"], n_boot), **info}
            preds.loc[te.index, f"{split}_{kind}"] = p
        res[split]["lookup_agent"] = report(te["label"], te["lookup"], te["code"], n_boot)
        res[split]["by_kind_lookup_acc"] = te.assign(ok=te["lookup"] == te["label"]).groupby("kind")["ok"].mean().round(4).to_dict()
    claims.to_csv(out / "claims_with_retrieval.csv", index=False)
    preds.to_csv(out / "predictions_cpu.csv", index=False)
    (out / "results_cpu.json").write_text(json.dumps(res, indent=2))
    return res


def run_gpu(data_dir, out_dir, seeds=(0, 1, 2), llm="Qwen/Qwen2.5-1.5B-Instruct", llm_claims=600, epochs=3,
            n_boot=1000) -> dict:
    from .agents import LLMVerifier
    from .neural import train_cross_encoder, zero_shot_nli
    from .retrieval import DenseRetriever
    hist, _ = load(data_dir)
    out = Path(out_dir)
    claims = pd.read_csv(out / "claims_with_retrieval.csv", keep_default_na=False)
    res = {}
    dense = DenseRetriever(hist["doc_id"], hist["text"])
    ranked_dense, _ = dense.rank(claims["claim"], top_k=5)
    res["retrieval_dense_e5_small"] = retrieval_metrics(claims["gold_doc_id"], ranked_dense)
    probs_store = {}
    for split in SPLITS:
        col = f"split_{split}"
        tr, va, te = (claims[claims[col] == s] for s in ("train", "val", "test"))
        res[split] = {"cross_encoder": {"runs": []}}
        all_p = []
        for seed in seeds:
            p, info = train_cross_encoder(tr, va, te, tr["top1_text"], va["top1_text"], te["top1_text"], seed=seed,
                                          epochs=epochs)
            pred = np.array(LABELS)[p.argmax(1)]
            res[split]["cross_encoder"]["runs"].append({**info, **report(te["label"], pred, te["code"], 200)})
            all_p.append(p)
        mean_p = np.mean(all_p, axis=0)
        probs_store[split] = mean_p
        ens = np.array(LABELS)[mean_p.argmax(1)]
        f1s = [r["macro_f1"] for r in res[split]["cross_encoder"]["runs"]]
        res[split]["cross_encoder"].update({"macro_f1_mean": float(np.mean(f1s)), "macro_f1_sd": float(np.std(f1s, ddof=1)),
                                            "ensemble": report(te["label"], ens, te["code"], n_boot),
                                            "risk_coverage": risk_coverage(te["label"], ens, mean_p.max(1))})
        nli = zero_shot_nli(te, te["top1_text"])
        res[split]["zero_shot_nli"] = report(te["label"], np.array(LABELS)[nli.argmax(1)], te["code"], n_boot)
        np.save(out / f"cross_encoder_probs_{split}.npy", mean_p)
    # Multi-agent pipeline on a stratified sample of the country-split test claims.
    te = claims[claims["split_country"] == "test"].copy()
    te["ce"] = np.array(LABELS)[probs_store["country"].argmax(1)]
    sample = pd.concat([g.sample(min(len(g), llm_claims // 3), random_state=0) for _, g in te.groupby("label")])
    text = dict(zip(hist["doc_id"], hist["text"]))
    recs = [[(d, text[d]) for d in ids.split(";")[:3]] for ids in sample["top_ids"]]
    verifier = LLMVerifier(llm)
    raw = verifier.verify_batch(list(sample["claim"]), recs)
    auditor = ProvenanceAuditor(hist)
    audited = [auditor.audit(c, v, [d for d, _ in r]) for c, v, r in zip(sample["claim"], raw, recs)]
    sample["llm_raw"] = [v["label"] for v in raw]
    sample["llm_audited"] = [v["label"] for v in audited]
    sample["audit"] = [v["audit"] for v in audited]
    sample["final"] = combine(sample["llm_audited"], sample["ce"])
    kept = sample[sample["final"] != "refer"]
    res["agent_pipeline"] = {
        "llm": llm, "claims": int(len(sample)),
        "llm_raw": report(sample["label"], sample["llm_raw"].replace("invalid", "insufficient_evidence"), sample["code"], n_boot),
        "llm_audited": report(sample["label"], sample["llm_audited"], sample["code"], n_boot),
        "cross_encoder_same_claims": report(sample["label"], sample["ce"], sample["code"], n_boot),
        "lookup_same_claims": report(sample["label"], sample["lookup"], sample["code"], n_boot),
        "audit_counts": sample["audit"].value_counts().to_dict(),
        "invalid_outputs": int((sample["llm_raw"] == "invalid").sum()),
        "agreement_coverage": float(len(kept) / len(sample)),
        "agreement_accuracy": float((kept["final"] == kept["label"]).mean()) if len(kept) else None,
        "agreement_report": report(kept["label"], kept["final"], kept["code"], n_boot) if len(kept) else None}
    sample.to_csv(out / "agent_pipeline_predictions.csv", index=False)
    (out / "agent_traces.jsonl").write_text("\n".join(json.dumps({"claim_id": i, **v}) for i, v in zip(sample["claim_id"], audited)))
    (out / "results_gpu.json").write_text(json.dumps(res, indent=2))
    return res


def agent_sample(claims, llm_claims=600):
    """The stratified sample of country-held-out test claims used by the GPU agent pipeline (same seed)."""
    te = claims[claims["split_country"] == "test"].copy()
    return pd.concat([g.sample(min(len(g), llm_claims // 3), random_state=0) for _, g in te.groupby("label")])


def ollama_info(host, models) -> dict:
    """Ollama version and model digests, so the run can be identified later."""
    import urllib.request
    tags = json.loads(urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=10).read()).get("models", [])
    digests = {t["name"]: t.get("digest", "") for t in tags}
    missing = [m for m in models if m not in digests]
    if missing:
        raise SystemExit("Pull these models first: " + "; ".join("ollama pull " + m for m in missing))
    version = json.loads(urllib.request.urlopen(host.rstrip("/") + "/api/version", timeout=10).read()).get("version", "")
    return {"ollama_version": version, "models": {m: digests[m] for m in models}}


def run_local(data_dir, out_dir, models, llm_claims=600, n_boot=1000, host="http://localhost:11434", ask=None,
              meta=None) -> dict:
    """Agent pipeline with local models through Ollama, on the same 600 claims as the GPU run.

    Needs ``treatyverify cpu`` first (claims_with_retrieval.csv, predictions_cpu.csv). Every reply is appended to
    local_llm_runs.jsonl at once and finished (model, claim) pairs are skipped, so the run can be stopped and
    restarted. The second verifier for the agreement rule is the claim + record TF-IDF MLP from the CPU run, because
    the cross-encoder needs a GPU.
    """
    from .agents import OllamaVerifier
    hist, _ = load(data_dir)
    out = Path(out_dir)
    claims = pd.read_csv(out / "claims_with_retrieval.csv", keep_default_na=False)
    cpu = pd.read_csv(out / "predictions_cpu.csv", keep_default_na=False).set_index("claim_id")
    sample = agent_sample(claims, llm_claims)
    sample["mlp"] = cpu.loc[sample["claim_id"], "country_pair_mlp"].values
    text = dict(zip(hist["doc_id"], hist["text"]))
    recs = {cid: [(d, text[d]) for d in ids.split(";")[:3]] for cid, ids in zip(sample["claim_id"], sample["top_ids"])}
    runs_path = out / "local_llm_runs.jsonl"
    done = set()
    if runs_path.exists():
        for line in runs_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["model"], r["claim_id"]))
            except (ValueError, KeyError):
                continue
    if meta is None and ask is None:
        meta = ollama_info(host, models)
    if meta is not None:
        (out / "local_llm_metadata.json").write_text(json.dumps({**meta, "claims": int(len(sample)), "temperature": 0,
                                                                 "seed": 42, "num_predict": 128}, indent=2))
    todo = [(m, cid, c) for m in models for cid, c in zip(sample["claim_id"], sample["claim"]) if (m, cid) not in done]
    with runs_path.open("a", encoding="utf-8") as f:
        for i, (m, cid, c) in enumerate(todo, 1):
            v = OllamaVerifier(m, host, ask).verify(c, recs[cid])
            f.write(json.dumps({"model": m, "claim_id": cid, **v}) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print("[{}/{}] {}".format(i, len(todo), m), flush=True)
    runs = {}
    for line in runs_path.read_text().splitlines():
        try:
            r = json.loads(line)
            runs[(r["model"], r["claim_id"])] = r
        except (ValueError, KeyError):
            continue
    from .agents import ProvenanceAuditor
    auditor = ProvenanceAuditor(hist)
    res = {"claims": int(len(sample)), "second_verifier": "country_pair_mlp",
           "mlp_same_claims": report(sample["label"], sample["mlp"], sample["code"], n_boot),
           "lookup_same_claims": report(sample["label"], sample["lookup"], sample["code"], n_boot), "models": {}}
    rows = []
    for m in models:
        raw = [runs[(m, cid)] for cid in sample["claim_id"]]
        audited = [auditor.audit(c, v, [d for d, _ in recs[cid]]) for c, v, cid in zip(sample["claim"], raw, sample["claim_id"])]
        s = sample[["claim_id", "label", "code", "kind", "mlp"]].copy()
        s["model"] = m
        s["llm_raw"] = [v["label"] for v in raw]
        s["llm_audited"] = [v["label"] for v in audited]
        s["audit"] = [v["audit"] for v in audited]
        s["final"] = combine(s["llm_audited"], s["mlp"])
        kept = s[s["final"] != "refer"]
        res["models"][m] = {
            "llm_raw": report(s["label"], s["llm_raw"].replace("invalid", "insufficient_evidence"), s["code"], n_boot),
            "llm_audited": report(s["label"], s["llm_audited"], s["code"], n_boot),
            "audit_counts": s["audit"].value_counts().to_dict(),
            "invalid_outputs": int((s["llm_raw"] == "invalid").sum()),
            "raw_accuracy_by_kind": s.assign(ok=s["llm_raw"] == s["label"]).groupby("kind")["ok"].mean().round(4).to_dict(),
            "agreement_coverage": float(len(kept) / len(s)),
            "agreement_accuracy": float((kept["final"] == kept["label"]).mean()) if len(kept) else None}
        rows.append(s)
    pd.concat(rows).to_csv(out / "local_llm_predictions.csv", index=False)
    (out / "results_local_llm.json").write_text(json.dumps(res, indent=2))
    return res
