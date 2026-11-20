# Data Governance and Annotation Protocol

## 1. Data sources

Use public, legally accessible documentary sources.

Prioritise:
- Primary treaty texts.
- United Nations depositary records.
- Official government publications.
- Official international organisation documentation.
- Reputable research institutions for supplementary context.

Every source must have a stable URL or a documented reason why
a stable URL is unavailable.

## 2. Source provenance

For every authentic source, record:

- Document identifier (`document_id`).
- Document title.
- Issuing organisation.
- Publication date, where available.
- Version or revision identifier, where available.
- Original source URL.
- Date accessed.
- Relevant section, paragraph, or page.
- Exact evidence passage used for annotation.

Preserve the distinction between the original source and any
derived claim or annotation.

## 3. Claim and evidence structure

Each annotated example should include:

- A unique `claim_id`.
- A `document_id` linking it to its source document.
- The factual claim being assessed.
- The documentary evidence passage.
- The assigned relation label.
- Source metadata.
- A synthetic-data indicator.

A single source document may support multiple examples.
All examples originating from the same document must retain
the same `document_id`.

## 4. Annotation labels

Use the following labels:

### supported

The supplied documentary passage directly supports the
substantive content of the claim.

### contradicted

The supplied documentary passage directly conflicts with
the substantive content of the claim.

### insufficient_evidence

The supplied passage does not provide enough information
to establish support or contradiction.

Absence of evidence must not automatically be labelled
as contradiction.

Labels describe the relationship to the supplied evidence.
They do not establish legal compliance, state intent, or
independent geopolitical truth.

## 5. Annotation quality assurance

For a substantive research corpus:

1. Use at least two independent annotators.
2. Provide a shared annotation guide.
3. Record individual annotations before adjudication.
4. Document disagreements and their resolution.
5. Report inter-annotator agreement where appropriate.
6. Preserve adjudication decisions and annotation versions.

The current synthetic fixture is not an expert-annotated
real-world corpus.

## 6. Synthetic data policy

The included `synthetic_demo.csv` contains fictional claims
and fictional evidence passages.

It exists exclusively for:
- Unit and integration testing.
- CLI verification.
- Pipeline smoke testing.
- Demonstrating expected data structure.

Synthetic examples must not be presented as authentic treaty
evidence or used to support substantive claims about states,
nuclear forces, treaties, or arms-control outcomes.

Synthetic and authentic datasets must remain distinguishable
through explicit metadata and separate dataset documentation.

## 7. Dataset splitting and leakage prevention

Use `document_id` as the grouping variable for evaluation.

Examples from one source document must not appear across
training and test partitions within the same evaluation fold.

The benchmark currently uses three-fold GroupKFold evaluation
for its synthetic smoke test.

For a substantive corpus, define and document the training,
validation, and test strategy before model evaluation.

Any duplicate, revised, or derived document records should
be reviewed for potential cross-partition leakage.

## 8. Copyright and licensing

Respect source copyright, licensing terms, and website
conditions.

Where redistribution of full documents is not permitted:
- Store source metadata.
- Store stable links to original documents.
- Retain only appropriately limited evidence excerpts.
- Document applicable reuse restrictions.

Do not assume that publicly accessible content is
automatically unrestricted for redistribution.

## 9. Uncertainty and responsible interpretation

Do not infer state intent, classified capability, operational
readiness, or undisclosed activity from incomplete public
information.

Record uncertainty explicitly.

A lack of public evidence is not proof that a claim is false.

All model outputs require independent source verification
and appropriate human review.

## 10. Dataset versioning and corrections

Maintain:
- Dataset version identifiers.
- Source and annotation change records.
- Correction history.
- Stable example and document identifiers.
- Reproducible dataset preparation procedures.

Corrections should be traceable without silently changing
previously reported experimental datasets.

## 11. Research scope

This project supports public documentary analysis for
arms-control and disarmament research.

It excludes weapon design, explosive performance,
delivery-system engineering, targeting, deployment,
and tactical or operational guidance.
