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
