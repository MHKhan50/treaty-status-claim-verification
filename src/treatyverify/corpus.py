"""Evidence documents and labelled claims built from yearly treaty-status records.

Each evidence document is the status history of one country under one treaty. Claims are generated from the
records with label-independent templates: the same sentence forms are used for supported and contradicted claims,
so the label can only be read from the evidence, not from the wording.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sources import TREATIES

LABELS = ("supported", "contradicted", "insufficient_evidence")
PHRASE = {"Not signed": "had not signed", "Signed": "had signed but not ratified",
          "Approved": "had ratified or acceded to", "Withdrawn": "had withdrawn from"}
STATUS_TEMPLATES = ("By {year}, {country} {phrase} the {treaty}.",
                    "In {year}, {country} {phrase} the {treaty} ({abbr}).",
                    "As of {year}, {country} {phrase} the {abbr}.")
EVENT_TEMPLATES = {"Signed": ("{country} signed the {treaty} in {year}.", "{country} became a signatory of the {abbr} in {year}."),
                   "Approved": ("{country} ratified or acceded to the {treaty} in {year}.",
                                "{country} became a party to the {abbr} in {year}.")}
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December")
OTHER_TREATIES = ("Treaty of Tlatelolco", "Treaty of Pelindaba", "Treaty of Rarotonga", "Treaty of Bangkok",
                  "Treaty on a Nuclear-Weapon-Free Zone in Central Asia")


def histories(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (treaty, country): spans of each status as a list of (status, first_year, last_year)."""
    rows = []
    for (treaty, country), g in panel.groupby(["treaty", "country"], sort=True):
        g = g.sort_values("year")
        spans, start, prev = [], None, None
        for year, status in zip(g["year"], g["status"]):
            if status != prev:
                if prev is not None:
                    spans.append((prev, start, last))
                start, prev = year, status
            last = year
        spans.append((prev, start, last))
        rows.append({"doc_id": f"{treaty}:{g['code'].iloc[0]}", "treaty": treaty, "country": country,
                     "code": g["code"].iloc[0], "first_year": int(g["year"].min()), "last_year": int(g["year"].max()),
                     "spans": spans})
    return pd.DataFrame(rows)


def document_text(row) -> str:
    info = TREATIES[row["treaty"]]
    parts = [f"{info['name']} ({row['treaty']}), status of {row['country']} recorded yearly from "
             f"{row['first_year']} to {row['last_year']} (UNODA Treaties Database)."]
    words = {"Not signed": "not signed", "Signed": "signed, not ratified", "Approved": "ratified or acceded",
             "Withdrawn": "withdrawn"}
    for status, a, b in row["spans"]:
        parts.append(f"{a}-{b}: {words[status]}." if a != b else f"{a}: {words[status]}.")
    return " ".join(parts)


def status_at(spans, year):
    for status, a, b in spans:
        if a <= year <= b:
            return status
    return None


def event_year(spans, status):
    for s, a, _ in spans:
        if s == status:
            return a
    return None


def make_claims(hist: pd.DataFrame, per_doc: int = 6, seed: int = 7) -> pd.DataFrame:
    """Balanced supported / contradicted / insufficient claims; every claim keeps its gold document id."""
    rng = np.random.default_rng(seed)
    out = []

    def add(row, claim, label, kind, year):
        out.append({"claim": claim, "label": label, "kind": kind, "year": int(year), "gold_doc_id": row["doc_id"],
                    "treaty": row["treaty"], "country": row["country"], "code": row["code"]})

    for _, row in hist.iterrows():
        info = TREATIES[row["treaty"]]
        fill = {"country": row["country"], "treaty": info["name"], "abbr": row["treaty"]}
        statuses = sorted({s for s, _, _ in row["spans"]})
        for _ in range(per_doc):
            kind = rng.choice(["status", "event", "insufficient"], p=[0.45, 0.22, 0.33])
            if kind == "event":
                events = [s for s in ("Signed", "Approved") if event_year(row["spans"], s) is not None
                          and event_year(row["spans"], s) > row["first_year"]]
                kind = "status" if not events else kind
            if kind == "status":
                year = int(rng.integers(row["first_year"], row["last_year"] + 1))
                true = status_at(row["spans"], year)
                supported = rng.random() < 0.5
                options = [s for s in ("Not signed", "Signed", "Approved") if s != true]
                stated = true if supported else rng.choice(options)
                t = STATUS_TEMPLATES[rng.integers(len(STATUS_TEMPLATES))]
                add(row, t.format(year=year, phrase=PHRASE[stated], **fill),
                    "supported" if supported else "contradicted", "status", year)
            elif kind == "event":
                ev = events[rng.integers(len(events))]
                year = event_year(row["spans"], ev)
                supported = rng.random() < 0.5
                if not supported:
                    shift = int(rng.choice([-5, -3, -2, -1, 1, 2, 3, 5]))
                    year = int(np.clip(year + shift, row["first_year"], row["last_year"]))
                    supported = year == event_year(row["spans"], ev)
                t = EVENT_TEMPLATES[ev][rng.integers(2)]
                add(row, t.format(year=year, **fill), "supported" if supported else "contradicted", "event", year)
            else:
                events = [s for s in ("Signed", "Approved") if event_year(row["spans"], s) is not None
                          and event_year(row["spans"], s) > row["first_year"]]
                if events and rng.random() < 0.5:
                    # Day-level claim with the right year: yearly records can neither confirm nor refute the day.
                    ev = events[rng.integers(len(events))]
                    year = event_year(row["spans"], ev)
                    day = f"{int(rng.integers(1, 29))} {MONTHS[rng.integers(12)]} {year}"
                    verb = "signed" if ev == "Signed" else "ratified or acceded to"
                    add(row, f"{row['country']} {verb} the {info['name']} on {day}.", "insufficient_evidence",
                        "day_precision", year)
                else:
                    # A year after the records end.
                    year = int(row["last_year"] + rng.integers(2, 9))
                    stated = rng.choice(["Signed", "Approved"])
                    t = STATUS_TEMPLATES[rng.integers(len(STATUS_TEMPLATES))]
                    add(row, t.format(year=year, phrase=PHRASE[stated], **fill), "insufficient_evidence", "future", year)
    claims = pd.DataFrame(out)
    # Claims about nuclear-weapon-free-zone treaties that are not in the corpus: no document can support them.
    extra = []
    countries = hist.drop_duplicates("country")[["country", "code"]].to_numpy()
    for i in range(len(claims) // 20):
        country, code = countries[rng.integers(len(countries))]
        name = OTHER_TREATIES[rng.integers(len(OTHER_TREATIES))]
        year = int(rng.integers(1970, 2024))
        extra.append({"claim": f"By {year}, {country} had ratified or acceded to the {name}.",
                      "label": "insufficient_evidence", "kind": "uncovered_treaty", "year": year,
                      "gold_doc_id": "", "treaty": "OTHER", "country": country, "code": code})
    claims = pd.concat([claims, pd.DataFrame(extra)], ignore_index=True)
    claims.insert(0, "claim_id", [f"C{i:05d}" for i in range(len(claims))])
    return claims


def split_by_country(claims: pd.DataFrame, seed: int = 11, frac=(0.70, 0.15, 0.15)) -> pd.Series:
    """Every claim about a country falls in one partition, so test countries are unseen in training."""
    codes = np.array(sorted(claims["code"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(codes)
    n_train, n_val = int(len(codes) * frac[0]), int(len(codes) * frac[1])
    part = {c: "train" for c in codes[:n_train]}
    part.update({c: "val" for c in codes[n_train:n_train + n_val]})
    part.update({c: "test" for c in codes[n_train + n_val:]})
    return claims["code"].map(part)


def split_by_treaty(claims: pd.DataFrame, test_treaty: str = "TPNW", country_split: pd.Series | None = None) -> pd.Series:
    """Train and validate on the other treaties; test on claims about one held-out treaty (and unseen countries)."""
    country_split = country_split if country_split is not None else split_by_country(claims)
    out = country_split.where(claims["treaty"] != test_treaty, "test")
    out = out.where(~((claims["treaty"] != test_treaty) & (country_split == "test")), "train")
    return out
