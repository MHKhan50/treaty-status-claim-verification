"""Command line: download, build, cpu, gpu, summary."""
from __future__ import annotations

import argparse
import json


def main(argv=None):
    p = argparse.ArgumentParser(prog="treatyverify")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("download", "build", "cpu", "gpu", "summary"):
        s = sub.add_parser(name)
        s.add_argument("--raw-dir", default="data/raw")
        s.add_argument("--data-dir", default="data/processed")
        s.add_argument("--out", default="reports")
    sub.choices["gpu"].add_argument("--seeds", default="0,1,2")
    sub.choices["gpu"].add_argument("--epochs", type=int, default=3)
    sub.choices["gpu"].add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    sub.choices["gpu"].add_argument("--llm-claims", type=int, default=600)
    sub.choices["cpu"].add_argument("--bootstrap", type=int, default=1000)
    a = p.parse_args(argv)
    if a.cmd == "download":
        from .sources import download
        print(json.dumps(download(a.raw_dir), indent=2))
    elif a.cmd == "build":
        from .experiment import build
        print(json.dumps(build(a.raw_dir, a.data_dir), indent=2))
    elif a.cmd == "cpu":
        from .experiment import run_cpu
        r = run_cpu(a.data_dir, a.out, a.bootstrap)
        print(json.dumps({k: (v if k.startswith("retrieval") else {m: x.get("macro_f1") for m, x in v.items() if isinstance(x, dict)})
                          for k, v in r.items()}, indent=2))
    elif a.cmd == "gpu":
        from .experiment import run_gpu
        run_gpu(a.data_dir, a.out, tuple(int(s) for s in a.seeds.split(",")), a.llm, a.llm_claims, a.epochs)
    elif a.cmd == "summary":
        from .summary import write
        print(write(a.out))


if __name__ == "__main__":
    main()
