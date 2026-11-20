| Verifier | Country-held-out test | Treaty-held-out test (TPNW) |
|---|---|---|
| Majority class | 0.181 [0.169, 0.191] | 0.166 [0.156, 0.176] |
| Claim only, TF-IDF LR | 0.733 [0.700, 0.766] | 0.583 [0.558, 0.610] |
| Claim + retrieved record, TF-IDF LR | 0.725 [0.691, 0.759] | 0.587 [0.566, 0.611] |
| Claim + retrieved record, TF-IDF MLP | 0.828 [0.788, 0.867] | 0.623 [0.596, 0.651] |
| Claim + gold record, TF-IDF LR | 0.722 [0.690, 0.756] | 0.570 [0.543, 0.599] |
| Cross-encoder (DeBERTa-v3-small), 3 seeds, mean ± SD | 0.910 ± 0.004 | 0.836 ± 0.139 |
| Zero-shot NLI (DeBERTa-v3-small) | 0.413 [0.372, 0.451] | 0.466 [0.440, 0.491] |
| Lookup agent (parser + record check) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

Retrieval (TF-IDF): recall@1 0.838, recall@3 0.977, recall@5 0.998, mrr 0.906
Retrieval (e5-small): recall@1 0.944, recall@3 0.997, recall@5 0.999, mrr 0.970

Agent pipeline (Qwen/Qwen2.5-1.5B-Instruct, 600 country-held-out test claims):
- LLM verifier raw: 0.400 [0.351, 0.449]; after provenance audit: 0.167 [0.152, 0.180]
- Cross-encoder on the same claims: 0.907 [0.885, 0.928]; lookup agent: 1.000 [1.000, 1.000]
- Audit: {'uncited_or_unretrieved_record': 548, 'ok': 52}; invalid outputs: 0
- Agreement rule: coverage 0.333, accuracy on kept claims 1.000
