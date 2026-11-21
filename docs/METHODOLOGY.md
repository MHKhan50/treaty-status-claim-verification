# Methodology

## Question

How well do lexical models, a fine-tuned cross-encoder, a zero-shot NLI model and a small LLM agent verify claims about
arms-control treaty participation against official status records, and does a provenance check with an
agreement rule make the agent's verdicts safer to use?

## Records and documents

The four status tables (PTBT, NPT, CTBT, TPNW) give, for every country and year, one of *Not signed*, *Signed*,
*Approved* (ratified, acceded or succeeded) and *Withdrawn*, following UNODA's actions as processed by Our World in Data.
`histories` turns each country's series into status spans, and `document_text` writes them as one evidence document
(for example `2000-2004: signed, not ratified. 2005-2025: ratified or acceded.`).

## Claims and labels

`make_claims` draws six claims per document with a fixed seed:

| Kind | Example | Label |
|---|---|---|
| status | By 2002, Benin had ratified or acceded to the CTBT. | supported if the record shows that status in that year, else contradicted |
| event | Benin became a party to the CTBT in 2010. | supported if that is the first year of the status, else contradicted (wrong years are shifted by 1–5) |
| future | As of 2029, Benin had signed but not ratified the CTBT. | insufficient: the records end earlier |
| day_precision | Chile ratified or acceded to the CTBT on 6 September 1999. | insufficient when the year is right: records are yearly |
| uncovered_treaty | By 1995, Qatar had ratified or acceded to the Treaty of Bangkok. | insufficient: no record in the corpus |

Supported and contradicted claims share the same sentence forms, so wording alone cannot reveal the label; the
claim-only baseline measures what remains (for example, base rates by treaty). A record-lookup parser reproduces every
label (tested), which confirms the labels follow from the records.

## Splits

- **Country-held-out:** countries are shuffled with a fixed seed and assigned 70/15/15 to train, validation and test.
- **Treaty-held-out:** all TPNW claims are test; the other treaties' claims about training and validation countries are
  used for training and validation (test-country claims about other treaties go to training).

## Verifiers and agent

See the README table. The cross-encoder reads the claim with the top TF-IDF record (as a deployed system would), is
trained for up to three epochs and scored at the epoch with the best validation macro-F1. The multi-agent pipeline runs
on a label-stratified sample of the country-held-out test claims: TF-IDF retrieval of three records, an LLM verifier
that must answer in JSON and cite a record, a provenance auditor that turns any verdict whose cited record was not
retrieved, does not match the claim's country and treaty, or does not cover the claim's year into
*insufficient_evidence*, and an agreement rule that keeps the verdict only when the cross-encoder agrees.

## Evaluation

Macro-F1 (primary), balanced accuracy and per-class F1; 95% percentile intervals from 1,000 bootstrap resamples of
countries. Accuracy by claim kind, risk–coverage for the cross-encoder ensemble, Recall@1/3/5 and MRR for retrieval,
audit counts, invalid LLM outputs, and the agreement rule's coverage and accuracy.
