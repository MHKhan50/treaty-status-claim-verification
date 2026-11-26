# TreatyVerify: Treaty-Status Claim Verification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24429803.svg)](https://doi.org/10.5281/zenodo.24429803) · [App](https://huggingface.co/spaces/MHkhan50/treaty-claim-check)

**Verifying public claims about arms-control treaty status against official records: lexical baselines, a fine-tuned cross-encoder, zero-shot NLI and a multi-agent LLM pipeline with provenance checks and abstention.**

Version `0.4.1` · Python `3.10+`

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

## App: Treaty Claim Check

[**Open the app**](https://huggingface.co/spaces/MHkhan50/treaty-claim-check): say or type a claim in English, Urdu or Turkish, for example *"Pakistan has signed the CTBT"* or *"Kya Iran NPT ka rukan hai?"*, and get a verdict from the treaty record with the year-by-year status, the source and a world map with a timeline from 1963. Paste a paragraph and every sentence that names a country and a treaty is checked.

The app follows the design this benchmark supports: language understanding only extracts *which country, which treaty, which action*; the verdict comes from the record. It runs in the browser (`space/index.html`), downloads the record live from Our World in Data, and needs no server. Parser and verdict tests: `node space/test_core.mjs`.

## Results (v0.3.0 Kaggle run, 20 November 2026)

Macro-F1 on the test sets with 95% country-level bootstrap intervals. Corpus: 788 evidence documents, 4,964 claims, 197 countries. Full outputs: `results/kaggle-2026-11-20/`.

| Verifier | Country-held-out test | Treaty-held-out test (TPNW) |
|---|---|---|
| Majority class | 0.181 [0.169, 0.191] | 0.166 [0.156, 0.176] |
| Claim only, TF-IDF LR | 0.733 [0.700, 0.766] | 0.583 [0.558, 0.610] |
| Claim + retrieved record, TF-IDF LR | 0.725 [0.691, 0.759] | 0.587 [0.566, 0.611] |
| Claim + retrieved record, TF-IDF MLP | 0.828 [0.788, 0.867] | 0.623 [0.596, 0.651] |
| Claim + gold record, TF-IDF LR | 0.722 [0.690, 0.756] | 0.570 [0.543, 0.599] |
| Cross-encoder (DeBERTa-v3-small), 3 seeds, mean ± SD | 0.910 ± 0.004 | 0.836 ± 0.139 |
| Zero-shot NLI (DeBERTa-v3-small) | 0.413 [0.372, 0.451] | 0.466 [0.440, 0.491] |
| Lookup agent (parser + record check) | 1.000 | 1.000 |

Retrieval Recall@1: TF-IDF 0.838, e5-small 0.944.

Agent pipeline on 600 country-held-out test claims: the LLM verifier (Qwen2.5-1.5B-Instruct) scores 0.400 [0.351, 0.449]; after the provenance audit 0.167, because 548 of 600 answers cited a record that was not retrieved or did not match. *Note (26 November 2026): some models copy the brackets of the record id shown in the prompt (`[NPT:PAK]`), which the audit used to count as an uncited record. Record ids are now stripped of brackets before the audit; this run will be rescored from its saved traces.* The cross-encoder scores 0.907 on the same claims; requiring the LLM and the cross-encoder to agree keeps 33% of claims at accuracy 1.000.

Reading these numbers: claims are generated from templates, so the lookup agent's 1.000 shows that a parser plus the record solves this version of the task, not that the problem is solved; the treaty-held-out cross-encoder varies strongly across seeds (SD 0.139). Version 1.0 adds real-world claims and larger models.

## Run

```bash
pip install -e ".[dev]"            # CPU parts and tests
pytest -q
treatyverify download             # data/raw + manifest.json
treatyverify build                # data/processed: documents.csv, claims.csv
treatyverify cpu                  # retrieval, lexical baselines, lookup agent
pip install -e ".[gpu]"
treatyverify gpu                  # cross-encoder (3 seeds), zero-shot NLI, dense retrieval, LLM agent pipeline
treatyverify summary              # reports/results.md
```

On Kaggle: import `notebooks/kaggle_treatyverify.ipynb` from this repository (GPU T4, Internet on) and run all; about 1 hour.

**Larger local models (Mac, Ollama).** The agent pipeline can also run with local models through [Ollama](https://ollama.com), on the same 600 test claims as the GPU run. It needs the CPU step first; replies are appended to `reports/local_llm_runs.jsonl` as they arrive and finished claims are skipped, so the run can be stopped and restarted. Because the cross-encoder needs a GPU, the second verifier for the agreement rule here is the claim + record TF-IDF MLP.

```bash
treatyverify local --models qwen2.5:3b llama3.2:3b mistral:7b qwen2.5:7b llama3.1:8b
```

## Limitations

- Claims are generated from templates; real-world claims are more varied. The benchmark isolates reading and reasoning over records.
- Yearly records: the month and day of an action are not available, which is why day-level claims are labelled insufficient.
- Our World in Data's status categories simplify UNODA actions (for example, Russia's CTBT de-ratification is shown as *signed* from 2023).

## Citation

```bibtex
@software{hussain2026treatyverify,
  author    = {Hussain, Muzammal},
  title     = {{TreatyVerify}: verification of public arms-control treaty-status claims},
  year      = {2026},
  version   = {0.4.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24429804},
  url       = {https://doi.org/10.5281/zenodo.24429804}
}
```

Please also cite the UNODA Treaties Database and Our World in Data.
