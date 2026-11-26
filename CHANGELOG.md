# Changelog

## Unreleased

- `treatyverify local`: the agent pipeline with local models through Ollama (temperature 0, seed 42), on the same 600 country-held-out test claims as the GPU run; resumable, with model digests recorded.

## 0.4.1 — 2026-11-21

- Renamed from WarheadVerify-AgentBench to **TreatyVerify** (repository `treaty-status-claim-verification`, package and command `treatyverify`, app *Treaty Claim Check*) so that the name describes what the project does: checking public claims against public treaty-status records. Code, data and results are unchanged. Version 0.4.0 is archived as doi:10.5281/zenodo.24429804.

## 0.4.0 — 2026-11-20

- App: **Treaty Claim Check**, a static Hugging Face Space (`space/`): voice or text claims in English, Urdu or Turkish, verdicts from the live treaty record with year-by-year status and source, paragraph checking, and a world map with a timeline; parser and verdict tests in `space/test_core.mjs`; upload script `scripts/upload_space.py`.
- Results of the v0.3.0 Kaggle run in `results/kaggle-2026-11-20/` and in the README.

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
