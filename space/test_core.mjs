// Run with: node space/test_core.mjs  (checks the claim parser and verdicts on a small synthetic record)
import {readFileSync, writeFileSync, mkdtempSync} from "node:fs";
import {tmpdir} from "node:os";
import {join, dirname} from "node:path";
import {fileURLToPath} from "node:url";
import {createRequire} from "node:module";
const here = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(here, "index.html"), "utf8");
const core = html.match(/<script id="core">([\s\S]*?)<\/script>/)[1];
const file = join(mkdtempSync(join(tmpdir(), "core-")), "core.cjs");
writeFileSync(file, core);
const C = createRequire(import.meta.url)(file);
const mk = rows => "Entity,Code,Year,Status of the treaty\n" + rows.map(r => r.join(",")).join("\n");
const yrs = (name, code, list) => list.map(([y, s]) => [name, code, y, s]);
const P = C.buildPanel({
  NPT: mk([...yrs("Pakistan", "PAK", [[2024, "Not signed"]]), ...yrs("Iran", "IRN", [[1968, "Signed"], [1970, "Approved"], [2024, "Approved"]]),
           ...yrs("North Korea", "PRK", [[1985, "Approved"], [2003, "Withdrawn"], [2024, "Withdrawn"]]), ...yrs("India", "IND", [[2024, "Not signed"]]),
           ...yrs("World", "OWID_WRL", [[2024, "Approved"]])]),
  CTBT: mk([...yrs("Pakistan", "PAK", [[2024, "Not signed"]]), ...yrs("United States", "USA", [[1996, "Signed"], [2024, "Signed"]])]),
  PTBT: mk([...yrs("Pakistan", "PAK", [[1963, "Signed"], [1988, "Approved"], [2024, "Approved"]])]),
  TPNW: mk([...yrs("Turkey", "TUR", [[2024, "Not signed"]]), ...yrs("Austria", "AUT", [[2017, "Signed"], [2018, "Approved"], [2024, "Approved"]])]),
});
const names = new Set(); for (const t in P) for (const c in P[t]) names.add(c);
const I = C.countryIndex(names);
const cases = [
  ["Pakistan has signed the CTBT.", "contradicted"], ["Pakistan has not signed the CTBT.", "supported"],
  ["North Korea withdrew from the NPT.", "supported"], ["India is a party to the NPT.", "contradicted"],
  ["Has the United States ratified the CTBT?", "contradicted"], ["Has the US signed the CTBT?", "supported"],
  ["Kya Iran NPT ka rukan hai?", "supported"], ["Türkiye TPNW'yi imzaladı mı?", "contradicted"],
  ["Türkiye TPNW'yi imzalamadı.", "supported"], ["پاکستان نے این پی ٹی پر دستخط نہیں کیے۔", "supported"],
  ["ایران این پی ٹی کا رکن ہے", "supported"], ["Iran ratified the NPT in 1969.", "contradicted"],
  ["Austria ratified the ban treaty", "supported"], ["Pakistan joined the partial test ban treaty", "supported"],
  ["France signed the NPT", "unknown"], ["Pakistan is great", "unknown"],
];
let failed = 0;
for (const [text, expected] of cases) {
  const got = C.verdict(C.parseClaim(text, I), P).label;
  if (got !== expected) { failed++; console.log(`FAIL ${text}: expected ${expected}, got ${got}`); }
}
console.log(`${cases.length - failed}/${cases.length} passed`);
process.exit(failed ? 1 : 0);
