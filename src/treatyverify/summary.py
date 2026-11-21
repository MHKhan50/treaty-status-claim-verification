"""Write results.md and the figure from results_cpu.json and results_gpu.json."""
from __future__ import annotations

import json
from pathlib import Path

ROWS = (("majority", "Majority class"), ("claim_lr", "Claim only, TF-IDF LR"), ("pair_lr", "Claim + retrieved record, TF-IDF LR"),
        ("pair_mlp", "Claim + retrieved record, TF-IDF MLP"), ("pair_lr_gold", "Claim + gold record, TF-IDF LR"))


def _fmt(r):
    return f"{r['macro_f1']:.3f} [{r['macro_f1_ci95'][0]:.3f}, {r['macro_f1_ci95'][1]:.3f}]"


def write(out_dir) -> str:
    out = Path(out_dir)
    cpu = json.loads((out / "results_cpu.json").read_text())
    gpu = json.loads((out / "results_gpu.json").read_text()) if (out / "results_gpu.json").exists() else {}
    lines = ["| Verifier | Country-held-out test | Treaty-held-out test (TPNW) |", "|---|---|---|"]
    for key, name in ROWS:
        lines.append(f"| {name} | {_fmt(cpu['country'][key])} | {_fmt(cpu['treaty'][key])} |")
    if gpu:
        ce = [gpu[s]["cross_encoder"] for s in ("country", "treaty")]
        lines.append("| Cross-encoder (DeBERTa-v3-small), 3 seeds, mean ± SD | "
                     + " | ".join(f"{c['macro_f1_mean']:.3f} ± {c['macro_f1_sd']:.3f}" for c in ce) + " |")
        lines.append("| Zero-shot NLI (DeBERTa-v3-small) | " + " | ".join(_fmt(gpu[s]["zero_shot_nli"]) for s in ("country", "treaty")) + " |")
    lines.append(f"| Lookup agent (parser + record check) | {_fmt(cpu['country']['lookup_agent'])} | {_fmt(cpu['treaty']['lookup_agent'])} |")
    lines.append("")
    lines.append("Retrieval (TF-IDF): " + ", ".join(f"{k} {v:.3f}" for k, v in cpu["retrieval_tfidf"].items() if k != "n"))
    if gpu:
        lines.append("Retrieval (e5-small): " + ", ".join(f"{k} {v:.3f}" for k, v in gpu["retrieval_dense_e5_small"].items() if k != "n"))
        a = gpu["agent_pipeline"]
        lines += ["", f"Agent pipeline ({a['llm']}, {a['claims']} country-held-out test claims):",
                  f"- LLM verifier raw: {_fmt(a['llm_raw'])}; after provenance audit: {_fmt(a['llm_audited'])}",
                  f"- Cross-encoder on the same claims: {_fmt(a['cross_encoder_same_claims'])}; lookup agent: {_fmt(a['lookup_same_claims'])}",
                  f"- Audit: {a['audit_counts']}; invalid outputs: {a['invalid_outputs']}",
                  f"- Agreement rule: coverage {a['agreement_coverage']:.3f}, accuracy on kept claims {a['agreement_accuracy']:.3f}"]
    text = "\n".join(lines) + "\n"
    (out / "results.md").write_text(text)
    return text
