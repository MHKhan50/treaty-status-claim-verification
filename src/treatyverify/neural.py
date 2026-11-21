"""Neural verifiers on (claim, retrieved evidence): a fine-tuned cross-encoder and a zero-shot NLI model."""
from __future__ import annotations

import time

import numpy as np

from .corpus import LABELS
from .metrics import macro_f1


def train_cross_encoder(train, val, test, ev_train, ev_val, ev_test, model_name="microsoft/deberta-v3-small",
                        seed=0, epochs=3, lr=3e-5, batch_size=32, max_len=256):
    """Fine-tune and return test probabilities (columns in LABELS order), the best epoch and minutes."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    # Recent transformers load weights in their stored dtype (fp16 for some checkpoints); train in fp32 with AMP.
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(LABELS)).float().to(dev)
    lab = {l: i for i, l in enumerate(LABELS)}

    def enc(df, ev):
        return tok(list(df["claim"]), list(ev), truncation="only_second", max_length=max_len, padding="max_length",
                   return_tensors="pt")

    tr, va, te = enc(train, ev_train), enc(val, ev_val), enc(test, ev_test)
    y_tr = torch.tensor([lab[l] for l in train["label"]])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(y_tr) / batch_size))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    def probs(e):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(e["input_ids"]), 128):
                batch = {k: v[i:i + 128].to(dev) for k, v in e.items()}
                with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                    out.append(torch.softmax(model(**batch).logits.float(), -1).cpu().numpy())
        return np.concatenate(out)

    start, best = time.time(), (-1, None, 0)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(y_tr))
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            batch = {k: v[idx].to(dev) for k, v in tr.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                loss = torch.nn.functional.cross_entropy(model(**batch).logits.float(), y_tr[idx].to(dev))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        p_val = probs(va)
        score = macro_f1(val["label"], np.array(LABELS)[p_val.argmax(1)])
        print(f"cross-encoder seed={seed} epoch={epoch} val_macro_f1={score:.4f}", flush=True)
        if score > best[0]:
            best = (score, probs(te), epoch)
    return best[1], {"model": model_name, "seed": seed, "best_epoch": best[2], "val_macro_f1": float(best[0]),
                     "minutes": round((time.time() - start) / 60, 2)}


def zero_shot_nli(test, ev_test, model_name="cross-encoder/nli-deberta-v3-small", batch_size=64):
    """Map entailment / contradiction / neutral to supported / contradicted / insufficient_evidence."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(dev).eval()
    names = {i: n.lower() for i, n in model.config.id2label.items()}
    order = [next(i for i, n in names.items() if n.startswith(k)) for k in ("entail", "contradict", "neutral")]
    out = []
    with torch.no_grad():
        for i in range(0, len(test), batch_size):
            # NLI convention: premise = evidence, hypothesis = claim.
            e = tok(list(ev_test[i:i + batch_size]), list(test["claim"].iloc[i:i + batch_size]), truncation="only_first",
                    max_length=256, padding=True, return_tensors="pt").to(dev)
            out.append(torch.softmax(model(**e).logits.float(), -1)[:, order].cpu().numpy())
    return np.concatenate(out)
