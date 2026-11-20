# WarheadVerify-AgentBench v0.2.0 (Research Development)

CPU-first benchmark for evidence-based verification of public nuclear arms-control and disarmament claims.

**Safety scope:** public policy, treaty documentation, transparency and disarmament only. No warhead design, explosive performance, delivery systems, targeting, deployment or operational guidance.

## Included
- TF-IDF + Logistic Regression baseline.
- TF-IDF + MLP neural baseline.
- TF-IDF cosine-similarity evidence retrieval.
- Deterministic retrieval-assisted verification agent with explicit abstention.
- CLI, tests, methodology, data governance and Kaggle CPU experiment script.

The included CSV is a **fictional synthetic software fixture**. Treaty names and clauses are invented. It is only for testing code execution; it is not a factual dataset and its scores are not research findings.

## macOS / VS Code setup

Open this folder in VS Code, then run these commands in its Terminal:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
```

Train and evaluate the demo:

```bash
warheadverify train --data data/synthetic_demo.csv --model-dir artifacts --model-type mlp
warheadverify evaluate --data data/synthetic_demo.csv --model-dir artifacts --model-type mlp
```

Try the retrieval-assisted agent:

```bash
warheadverify verify --claim "Under fictional Accord A, Article 1 requires annual public reporting." --evidence "Fictional Accord A, Article 1: Each participating party shall publish an annual implementation report." --model-dir artifacts
```

## Kaggle CPU

Upload the extracted project folder as a Kaggle Dataset, select CPU, then:

```bash
pip install -e .
python scripts/kaggle_cpu_experiment.py
```

The current experiment uses two classifiers (Logistic Regression and MLP), five random seeds (11, 23, 37, 51, 79), and three-fold document-grouped GroupKFold evaluation using document_id. It generates 30 experiment records and checks that training and test document groups do not overlap. The included synthetic fixture is for software smoke testing only; its scores are not research findings.

## Dataset schema
`claim_id`, `document_id`, `claim`, `label`, `evidence_text`, `source_name`, `source_url`, `is_synthetic`.

Labels: `supported`, `contradicted`, `insufficient_evidence`.

For a real corpus, curate dated public primary documents, retain exact document/page/section provenance and short evidence spans, use at least two independent annotators, adjudicate disagreements, and split by source document or treaty family to reduce leakage. Absence of public evidence is not proof that a claim is false.

## Suggested experiments
1. Logistic Regression vs MLP on identical text features.
2. Claim-only classification vs retrieval-assisted workflow.
3. Evidence sufficiency threshold and abstention analysis.
4. Document-held-out and treaty-family-held-out robustness.
5. Ablations: remove retrieval, vary top-k, remove metadata.

Report macro-F1, balanced accuracy, per-class precision/recall, confusion matrix, calibration, and retrieval Recall@k / MRR where gold evidence IDs exist.

## Reputable starting sources
These are candidate sources for future corpus construction, not sources for the synthetic labels:
- SIPRI World Nuclear Forces: https://www.sipri.org/research/armament-and-disarmament/nuclear-weapons/world-nuclear-forces
- Federation of American Scientists: https://fas.org/initiative/status-world-nuclear-forces/
- UN disarmament documentation: https://research.un.org/en/docs/disarmament
- UN Treaty Collection: https://treaties.un.org/

Verify the document version, publication date and relevant section directly at the source.

Version 0.2.0 is a research-development milestone. The current dataset is synthetic and no empirical real-world benchmark results are claimed.
