# Data

Nothing is redistributed here. `warheadverify download` fetches four yearly treaty-status tables (PTBT, NPT, CTBT, TPNW)
from Our World in Data, which processes the UNODA Treaties Database (CC BY 4.0), into `data/raw/` and writes
`manifest.json` with each URL, retrieval time, row count and SHA-256. `warheadverify build` writes the evidence documents
and claims to `data/processed/`.

Cite the original source: United Nations Office for Disarmament Affairs, Treaties Database, with major processing by
Our World in Data (Herre, Rosado and Roser, "Nuclear Weapons", Our World in Data).
