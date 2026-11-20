"""Public treaty-status data: UNODA Treaties Database as published by Our World in Data (CC BY 4.0)."""
from __future__ import annotations

import hashlib
import json
import socket
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

OWID = "https://ourworldindata.org/grapher/{slug}.csv?v=1&csvType=full&useColumnShortNames=false"
TREATIES = {
    "PTBT": {"slug": "partial-nuclear-test-ban-treaty",
             "name": "Partial Nuclear Test-Ban Treaty", "opened": 1963},
    "NPT": {"slug": "nuclear-non-proliferation-treaty",
            "name": "Treaty on the Non-Proliferation of Nuclear Weapons", "opened": 1968},
    "CTBT": {"slug": "comprehensive-nuclear-test-ban-treaty",
             "name": "Comprehensive Nuclear-Test-Ban Treaty", "opened": 1996},
    "TPNW": {"slug": "prohibition-of-nuclear-weapons",
             "name": "Treaty on the Prohibition of Nuclear Weapons", "opened": 2017},
}
STATUSES = ("Not signed", "Signed", "Approved", "Withdrawn")


def download(raw_dir: str | Path, retries: int = 5) -> dict:
    """Download the four status tables and write a manifest with URL, retrieval time, rows and SHA-256."""
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    socket.setdefaulttimeout(60)
    manifest = {"source": "UNODA Treaties Database via Our World in Data (CC BY 4.0)", "files": {}}
    for key, info in TREATIES.items():
        url = OWID.format(slug=info["slug"])
        path = raw / f"{key}.csv"
        if not path.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (warheadverify-agentbench)"})
            for attempt in range(retries):
                try:
                    with urllib.request.urlopen(req) as resp:
                        path.write_bytes(resp.read())
                    break
                except Exception as err:  # noqa: BLE001
                    print(f"{key}: attempt {attempt + 1} failed: {err!r}", flush=True)
                    if attempt == retries - 1:
                        raise
                    time.sleep(2 ** attempt * 3)
        data = path.read_bytes()
        manifest["files"][key] = {"url": url, "sha256": hashlib.sha256(data).hexdigest(),
                                  "rows": len(pd.read_csv(path)),
                                  "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_panel(raw_dir: str | Path) -> pd.DataFrame:
    """Country-year statuses for all treaties: columns treaty, country, code, year, status."""
    frames = []
    for key in TREATIES:
        df = pd.read_csv(Path(raw_dir) / f"{key}.csv")
        status_col = [c for c in df.columns if c not in ("Entity", "Code", "Year", "entity", "code", "year")][0]
        df = df.rename(columns={"Entity": "country", "Code": "code", "Year": "year", status_col: "status"})
        df = df[df["code"].notna() & ~df["code"].astype(str).str.startswith("OWID")]
        df = df[df["status"].isin(STATUSES)]
        df["treaty"] = key
        frames.append(df[["treaty", "country", "code", "year", "status"]])
    panel = pd.concat(frames, ignore_index=True)
    panel["year"] = panel["year"].astype(int)
    return panel.sort_values(["treaty", "country", "year"]).reset_index(drop=True)
