# Methodology

## 1. Research question

Can a lightweight neural text classifier, combined with transparent
evidence retrieval and explicit abstention, improve the triage of
public arms-control claims?

## 2. Task definition

Input:
- A short arms-control claim.
- Documentary passages potentially relevant to the claim.

Output:
- `supported`
- `contradicted`
- `insufficient_evidence`

Labels describe the relationship between a claim and the supplied
documentary evidence. They are not legal opinions, treaty-compliance
determinations, or independent verification of geopolitical events.

## 3. System architecture

The benchmark includes three modelling components:

- **TF-IDF + Logistic Regression:** classical machine-learning baseline.
- **TF-IDF + MLP:** lightweight neural classification baseline.
- **Retrieval-assisted workflow:** cosine-similarity evidence ranking,
  classifier predictions, and threshold-based abstention.

Retrieval similarity is a ranking signal, not proof of semantic
entailment or contradiction.

## 4. Dataset and provenance

The included `synthetic_demo.csv` dataset contains fictional claims
and fictional documentary passages.

Each example has:
- A unique `claim_id`.
- A `document_id` identifying its source document.
- A claim and its assigned label.
- An evidence passage.
- Source metadata.
- A synthetic-data indicator.

The fixture contains 18 examples across 12 unique document groups.

It is intended exclusively for software testing and experimental
pipeline validation. It does not represent real treaties, states,
nuclear stockpiles, or arms-control events.

## 5. Evaluation design

### 5.1 Document-grouped cross-validation

The experiment uses three-fold `GroupKFold` cross-validation.

The grouping variable is `document_id`. All examples associated
with one document remain in the same fold.

This prevents claims from the same source document appearing in
both training and test partitions within a fold.

The implementation explicitly checks that the training and test
document sets are disjoint. An experiment raises an error if
document overlap is detected.

### 5.2 Models and random seeds

Two classifiers are evaluated:

- Logistic Regression.
- Multi-layer Perceptron (MLP).

The experiment uses five random seeds:

`11, 23, 37, 51, 79`

Both models use the same document-grouped folds and seed list.

With two models, five seeds, and three folds, the experiment
produces 30 result records.

### 5.3 Reported metrics

The experiment reports:

- Macro-F1.
- Balanced accuracy.
- Training and test example counts.
- Training and test document counts.
- Document overlap.

For a substantive evaluation, additional reporting should include
per-class precision and recall, confusion matrices, and probability
calibration analysis.

Retrieval should be evaluated separately using Recall@k and
Mean Reciprocal Rank (MRR) when gold evidence identifiers are
available.

## 6. Synthetic experiment status

The current experiment is a pipeline smoke test.

The generated report is marked:

`synthetic_demo_only`

The observed scores must not be interpreted as real-world
claim-verification performance, evidence quality, or model
generalisation.

The small number of fictional document groups also limits the
statistical strength of cross-validation estimates.

## 7. Limitations and future work

A substantive research benchmark requires:

1. A larger corpus of authentic public documents.
2. Traceable source URLs and publication metadata.
3. Expert-reviewed claim and evidence annotations.
4. Document-level train, validation, and test separation.
5. Gold evidence identifiers for retrieval evaluation.
6. Per-class performance and calibration analysis.
7. Repeated experiments and uncertainty estimates.
8. Explicit evaluation of abstention and evidence sufficiency.

Future work should investigate whether retrieval-assisted
verification improves evidence-grounded claim triage over
classification-only baselines.

## 8. Research scope and responsible use

The project is limited to public documentary analysis for
arms-control and disarmament research.

It does not provide weapon design, explosive performance,
delivery-system, targeting, deployment, or tactical guidance.

Outputs should be treated as research assistance requiring
independent source verification and human review.
