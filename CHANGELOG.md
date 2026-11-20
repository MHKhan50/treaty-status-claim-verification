# Changelog

## 0.3.0 — 2026-11-20

- Real evidence corpus: yearly treaty-status records for PTBT, NPT, CTBT and TPNW (UNODA via Our World in Data),
  downloaded with a SHA-256 manifest; one evidence document per country and treaty.
- Claims generated from the records with label-independent templates, including insufficient-evidence cases (years
  after the records, exact days, treaties not in the corpus); country-held-out and treaty-held-out splits.
- Verifiers: majority, claim-only and claim + record TF-IDF baselines, fine-tuned DeBERTa-v3-small cross-encoder
  (3 seeds), zero-shot NLI, a lookup agent, and a multi-agent pipeline (retriever, LLM verifier, provenance auditor,
  agreement-based abstention).
- Metrics with country-level bootstrap intervals, risk-coverage and retrieval Recall@k / MRR; Kaggle notebook.
- The synthetic demo data of 0.1–0.2 is removed.

## 0.2.0 — 2026-11-20

- Prototype on a fictional fixture with TF-IDF classifiers and a retrieval-assisted agent.
