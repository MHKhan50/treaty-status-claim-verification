"""Verification agents.

* ``LookupAgent``: parses the claim and checks the cited record with code (a structured tool; reference upper bound).
* ``LLMVerifier``: a small instruction-tuned LLM reads the retrieved records and returns a label with a cited record.
* ``ProvenanceAuditor``: checks that the cited record was retrieved, matches the claim's country and treaty, and covers
  the claim's year; otherwise the verdict becomes ``insufficient_evidence``.
* ``combine``: the final verdict needs the LLM verifier (after audit) and a second verifier to agree; disagreements are
  referred to a human (abstention).
"""
from __future__ import annotations

import json
import re

from .corpus import OTHER_TREATIES, event_year, status_at
from .sources import TREATIES

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
STATED = {"had not signed": "Not signed", "had signed but not ratified": "Signed",
          "had ratified or acceded to": "Approved", "had withdrawn from": "Withdrawn"}


def parse_claim(claim: str, countries) -> dict:
    """Treaty, country, year and assertion of a claim, or None fields when absent."""
    out = {"treaty": None, "country": None, "year": None, "kind": None, "stated": None}
    if any(name in claim for name in OTHER_TREATIES):
        out["treaty"] = "OTHER"
    for key, info in TREATIES.items():
        if info["name"] in claim or re.search(rf"\b{key}\b", claim):
            out["treaty"] = key
    for c in sorted(countries, key=len, reverse=True):
        if c in claim:
            out["country"] = c
            break
    years = re.findall(r"\b(1[89]\d\d|20\d\d)\b", claim)
    out["year"] = int(years[-1]) if years else None
    if re.search(rf"\bon \d{{1,2}} ({MONTHS}) \d{{4}}", claim):
        out["kind"] = "day"
        out["stated"] = "Signed" if " signed the " in claim else "Approved"
        return out
    for phrase, status in STATED.items():
        if phrase in claim:
            out["kind"], out["stated"] = "status", status
            return out
    if "signed the" in claim or "became a signatory" in claim:
        out["kind"], out["stated"] = "event", "Signed"
    elif "ratified or acceded" in claim or "became a party" in claim:
        out["kind"], out["stated"] = "event", "Approved"
    return out


class LookupAgent:
    def __init__(self, hist):
        self.docs = {(r.treaty, r.country): r for r in hist.itertuples()}
        self.countries = sorted(hist["country"].unique())

    def verify(self, claim: str) -> dict:
        c = parse_claim(claim, self.countries)
        doc = self.docs.get((c["treaty"], c["country"]))
        if doc is None or c["year"] is None or c["kind"] is None:
            return {"label": "insufficient_evidence", "doc_id": "", "reason": "no record for this treaty and country"}
        if not doc.first_year <= c["year"] <= doc.last_year:
            return {"label": "insufficient_evidence", "doc_id": doc.doc_id, "reason": "year outside the record"}
        if c["kind"] == "status":
            ok = status_at(doc.spans, c["year"]) == c["stated"]
        else:
            ok = event_year(doc.spans, c["stated"]) == c["year"]
            if c["kind"] == "day":
                return {"label": "insufficient_evidence" if ok else "contradicted", "doc_id": doc.doc_id,
                        "reason": "records are yearly" if ok else "different year"}
        return {"label": "supported" if ok else "contradicted", "doc_id": doc.doc_id, "reason": "record lookup"}


PROMPT = """You verify claims about nuclear arms-control treaties using only the records below.
Each record gives one country's yearly status under one treaty ("ratified or acceded" means a party).

Records:
{records}

Claim: {claim}

Answer "supported" if the records show the claim is true, "contradicted" if they show it is false, and
"insufficient_evidence" if no record covers the claim's treaty, country or year, or the claim is more precise
than the records (for example an exact day). Reply with JSON only:
{{"label": "supported|contradicted|insufficient_evidence", "doc_id": "<record id you used or empty>", "reason": "<short>"}}"""


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    label = str(d.get("label", "")).strip().lower()
    if label not in ("supported", "contradicted", "insufficient_evidence"):
        label = "invalid"
    return {"label": label, "doc_id": str(d.get("doc_id", "")).strip(), "reason": str(d.get("reason", ""))[:200]}


class LLMVerifier:
    """Greedy decoding with a Hugging Face chat model (default Qwen2.5-1.5B-Instruct)."""

    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device).eval()
        self.device = device

    def verify_batch(self, claims, records_list, batch_size=8):
        out = []
        for i in range(0, len(claims), batch_size):
            prompts = [self.tok.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(records="\n".join(f"[{d}] {t}" for d, t in recs), claim=c)}],
                tokenize=False, add_generation_prompt=True) for c, recs in zip(claims[i:i + batch_size], records_list[i:i + batch_size])]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=96, do_sample=False, pad_token_id=self.tok.eos_token_id)
            texts = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            out.extend({**parse_json(t), "raw": t[:400]} for t in texts)
        return out


class ProvenanceAuditor:
    def __init__(self, hist):
        self.by_id = {r.doc_id: r for r in hist.itertuples()}
        self.countries = sorted(hist["country"].unique())

    def audit(self, claim: str, verdict: dict, retrieved_ids) -> dict:
        v = dict(verdict)
        v["audit"] = "ok"
        if v["label"] == "invalid":
            v["label"], v["audit"] = "insufficient_evidence", "invalid_output"
            return v
        if v["label"] == "insufficient_evidence":
            return v
        doc = self.by_id.get(v["doc_id"])
        c = parse_claim(claim, self.countries)
        if doc is None or v["doc_id"] not in retrieved_ids:
            v["label"], v["audit"] = "insufficient_evidence", "uncited_or_unretrieved_record"
        elif doc.country != c["country"] or doc.treaty != c["treaty"]:
            v["label"], v["audit"] = "insufficient_evidence", "record_does_not_match_claim"
        elif c["year"] is None or not doc.first_year <= c["year"] <= doc.last_year:
            v["label"], v["audit"] = "insufficient_evidence", "year_outside_record"
        return v


def combine(primary, secondary):
    """Agreement rule: keep the verdict when both verifiers agree, otherwise refer (abstain)."""
    return [p if p == s else "refer" for p, s in zip(primary, secondary)]
