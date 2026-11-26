import json

import numpy as np
import pandas as pd
import pytest

from treatyverify.agents import LookupAgent, ProvenanceAuditor, combine, parse_claim, parse_json
from treatyverify.corpus import LABELS, histories, make_claims, split_by_country, split_by_treaty
from treatyverify.experiment import build, load, run_cpu
from treatyverify.metrics import retrieval_metrics
from treatyverify.sources import TREATIES, download, load_panel

COUNTRIES = ["Pakistan", "India", "Niger", "Nigeria", "Guinea", "Equatorial Guinea", "France", "Brazil", "Chile",
             "Kenya", "Ghana", "Peru", "Japan", "Norway", "Egypt", "Mexico", "Spain", "Italy", "Iran", "Iraq",
             "Oman", "Mali", "Chad", "Laos", "Fiji", "Togo", "Benin", "Nepal", "Qatar", "Yemen"]


@pytest.fixture
def raw(tmp_path):
    rng = np.random.default_rng(0)
    for key, info in TREATIES.items():
        rows = []
        for i, c in enumerate(COUNTRIES):
            sign = info["opened"] + int(rng.integers(0, 6))
            ratify = sign + int(rng.integers(1, 8))
            for year in range(info["opened"], 2026):
                status = "Not signed" if year < sign else ("Signed" if year < ratify else "Approved")
                if i % 7 == 0:
                    status = "Not signed"
                rows.append({"Entity": c, "Code": f"C{i:02d}", "Year": year, info["name"]: status})
        rows.append({"Entity": "World", "Code": "OWID_WRL", "Year": 2020, info["name"]: "Signed"})
        pd.DataFrame(rows).to_csv(tmp_path / f"{key}.csv", index=False)
    return tmp_path


def test_panel_and_manifest(raw):
    manifest = download(raw)
    assert set(manifest["files"]) == set(TREATIES)
    panel = load_panel(raw)
    assert "World" not in set(panel["country"])
    assert set(panel["status"]) <= {"Not signed", "Signed", "Approved"}


def test_claims_balanced_and_labelled_correctly(raw):
    hist = histories(load_panel(raw))
    claims = make_claims(hist)
    shares = claims["label"].value_counts(normalize=True)
    assert set(shares.index) == set(LABELS) and shares.min() > 0.2
    agent = LookupAgent(hist)
    agree = np.mean([agent.verify(c)["label"] == l for c, l in zip(claims["claim"], claims["label"])])
    assert agree == 1.0


def test_splits_do_not_leak(raw):
    claims = make_claims(histories(load_panel(raw)))
    sc = split_by_country(claims)
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        assert not set(claims.loc[sc == a, "code"]) & set(claims.loc[sc == b, "code"])
    st = split_by_treaty(claims, "TPNW", sc)
    assert set(claims.loc[st == "test", "treaty"]) == {"TPNW"}
    assert "TPNW" not in set(claims.loc[st != "test", "treaty"])


def test_parse_claim_longest_country():
    c = parse_claim("By 1999, Equatorial Guinea had signed but not ratified the CTBT.", ["Guinea", "Equatorial Guinea"])
    assert c == {"treaty": "CTBT", "country": "Equatorial Guinea", "year": 1999, "kind": "status", "stated": "Signed"}
    d = parse_claim("Niger signed the Treaty on the Non-Proliferation of Nuclear Weapons on 3 May 1970.", ["Niger", "Nigeria"])
    assert d["kind"] == "day" and d["country"] == "Niger" and d["treaty"] == "NPT"


def test_auditor_and_json(raw):
    hist = histories(load_panel(raw))
    aud = ProvenanceAuditor(hist)
    row = hist.iloc[1]
    claim = f"By {row.last_year + 3}, {row.country} had ratified or acceded to the {row.treaty}."
    v = aud.audit(claim, {"label": "supported", "doc_id": row.doc_id, "reason": ""}, [row.doc_id])
    assert v["label"] == "insufficient_evidence" and v["audit"] == "year_outside_record"
    v = aud.audit(claim, {"label": "supported", "doc_id": row.doc_id, "reason": ""}, [])
    assert v["audit"] == "uncited_or_unretrieved_record"
    assert parse_json('ok {"label": "Supported", "doc_id": "NPT:C01"}')["label"] == "supported"
    assert parse_json("no json")["label"] == "invalid"
    assert combine(["supported", "contradicted"], ["supported", "supported"]) == ["supported", "refer"]


def test_end_to_end_cpu(raw, tmp_path):
    summary = build(raw, tmp_path / "proc")
    assert summary["claims"] > 300
    res = run_cpu(tmp_path / "proc", tmp_path / "reports", n_boot=50)
    assert res["country"]["lookup_agent"]["macro_f1"] == 1.0
    assert res["retrieval_tfidf"]["recall@5"] > 0.8
    assert 0 <= res["country"]["claim_lr"]["macro_f1"] <= 1
    hist, claims = load(tmp_path / "proc")
    assert isinstance(hist["spans"].iloc[0][0], tuple)


def test_retrieval_metrics():
    m = retrieval_metrics(["a", "b", ""], [["a", "x"], ["x", "b"], ["y"]])
    assert m["recall@1"] == 0.5 and m["mrr"] == 0.75 and m["n"] == 2


def test_local_llm_pipeline_resumes(raw, tmp_path):
    from treatyverify.experiment import run_local
    build(raw, tmp_path / "proc")
    run_cpu(tmp_path / "proc", tmp_path / "reports", n_boot=20)
    calls = []

    def ask(prompt):
        calls.append(prompt)
        first_id = prompt.split("[", 2)[1].split("]")[0]
        return json.dumps({"label": "supported", "doc_id": first_id, "reason": "fake"})

    meta = {"ollama_version": "test", "models": {"fake:1b": "x"}}
    res = run_local(tmp_path / "proc", tmp_path / "reports", ["fake:1b"], llm_claims=30, n_boot=20, ask=ask, meta=meta)
    n = res["claims"]
    assert len(calls) == n and res["models"]["fake:1b"]["invalid_outputs"] == 0
    assert 0 <= res["models"]["fake:1b"]["agreement_coverage"] <= 1
    run_local(tmp_path / "proc", tmp_path / "reports", ["fake:1b"], llm_claims=30, n_boot=20, ask=ask, meta=meta)
    assert len(calls) == n  # finished claims are skipped on a second run
    assert (tmp_path / "reports" / "local_llm_predictions.csv").exists()
