# WarheadVerify-AgentBench

**Verifying public nuclear arms-control claims against treaty-status records: lexical baselines, a fine-tuned cross-encoder, zero-shot NLI and a multi-agent LLM pipeline with provenance checks and abstention.**

Version `0.3.0` · Python `3.10+`

> **Scope.** Public treaty participation only (signature, ratification, accession, withdrawal). No information on weapons, materials, delivery, deployment or targeting. Labels describe whether the published records support a claim; they are not legal opinions or compliance findings.

## Task

Given a claim such as *"By 2005, Brazil had ratified or acceded to the Comprehensive Nuclear-Test-Ban Treaty"*, a verifier returns **supported**, **contradicted** or **insufficient_evidence** and cites the record it used.

## Data

- **Records:** yearly status of every country under four treaties (Partial Test-Ban Treaty, NPT, CTBT, TPNW): *not signed*, *signed*, *ratified or acceded*, *withdrawn*. Source: UNODA Treaties Database as processed by Our World in Data (CC BY 4.0). Downloaded at run time with a manifest (URL, time, rows, SHA-256); not redistributed.
- **Evidence documents:** one per country and treaty, the status history in plain text.
- **Claims** (generated from the records with fixed seeds; the same sentence forms are used for supported and contradicted claims):
  - *status at a year* and *year of signature or ratification* → supported or contradicted;
  - *a year after the records end*, *an exact day* (records are yearly) and *treaties not in the corpus* → insufficient evidence.
- **Splits:** (1) by country, 70/15/15, so test countries are unseen; (2) by treaty, TPNW held out for testing.
- The labelling is checked in the tests: a parser that looks up each claim's record reproduces every label.

## Verifiers

| Verifier | What it sees |
|---|---|
| Majority class | nothing |
| Claim only, TF-IDF logistic regression | the claim (checks for wording shortcuts) |
| Claim + record, TF-IDF LR and MLP | claim and the top TF-IDF record (or the gold record) |
| Cross-encoder, DeBERTa-v3-small, 3 seeds | claim and the top record, fine-tuned |
| Zero-shot NLI, DeBERTa-v3-small | record as premise, claim as hypothesis |
| Lookup agent | parses the claim and checks the record with code (reference) |
| **Multi-agent pipeline** | retriever (top 3) → LLM verifier (Qwen2.5-1.5B-Instruct, JSON with cited record) → provenance auditor (cited record retrieved, matches country and treaty, covers the year) → agreement with the cross-encoder, otherwise *refer* |

Metrics: macro-F1 with 95% intervals from 1,000 country-level bootstrap resamples, balanced accuracy, per-class F1, accuracy by claim type, risk–coverage of the cross-encoder, Recall@k and MRR for TF-IDF and e5-small retrieval, audit counts and the agreement rule's coverage and accuracy.

## Results

Results will be added from the Kaggle run (`notebooks/kaggle_warheadverify.ipynb`).

## Run

```bash
pip install -e ".[dev]"            # CPU parts and tests
pytest -q
warheadverify download             # data/raw + manifest.json
warheadverify build                # data/processed: documents.csv, claims.csv
warheadverify cpu                  # retrieval, lexical baselines, lookup agent
pip install -e ".[gpu]"
warheadverify gpu                  # cross-encoder (3 seeds), zero-shot NLI, dense retrieval, LLM agent pipeline
warheadverify summary              # reports/results.md
```

On Kaggle: import `notebooks/kaggle_warheadverify.ipynb` from this repository (GPU T4, Internet on) and run all; about 1 hour.

## Limitations

- Claims are generated from templates; real-world claims are more varied. The benchmark isolates reading and reasoning over records.
- Yearly records: the month and day of an action are not available, which is why day-level claims are labelled insufficient.
- Our World in Data's status categories simplify UNODA actions (for example, Russia's CTBT de-ratification is shown as *signed* from 2023).

## Citation

See `CITATION.cff`. Please also cite the UNODA Treaties Database and Our World in Data.
