""" composed-charge lever — reusable RT probe + charged-backlog re-derivation .

`probe(smiles_list) -> dict` is the interface every later workstream imports:

    {smiles: {"name": str | None, "rt": "full"|"block1"|"wrong"|"abstain"|"unparseable"}}

    full = OPSIN(name) -> InChIKey equals the input's InChIKey (constitution +
                  stereo + charge all match) -- ONLY when both keys are genuine
                  full-length (14+ char) InChIKeys; never a falsy/empty-vs-falsy match.
    block1 = only the first 14-char InChIKey block matches (right constitution,
                  stereo/charge miss) -- again, only when both keys are genuine.
    wrong = a name was emitted, OPSIN parsed it, but InChIKey block-1 differs --
                  OR the emitted name's key is genuine while the INPUT's key is falsy
                  (fail-closed: nothing genuine to compare against, so never call it
                  a match).
    abstain = no name emitted (Orthonym declined, or the output is a failure
                  sentinel per ``errors.is_failure_name``).
    unparseable = a name was emitted but its round-trip produced no genuine InChIKey:
                  either OPSIN itself could not parse the name, RDKit could not parse
                  OPSIN's output SMILES, or RDKit parsed it but ``MolToInchiKey``
                  returned '' (its documented behaviour for e.g. a wildcard/dummy-atom
                  structure -- NOT an exception, so ``is None`` alone does not catch
                  it). All three are an unusable round-trip, so all three fold into
                  this one bucket per the 5-way spec.

    ⚠ Falsy-key guard (fixed 2026-08-22 review round 1): ``Chem.MolToInchiKey`` can
    return ``''`` for a structure it cannot key (e.g. any wildcard/dummy-atom SMILES)
    without raising. Comparing ``'' == ''`` would misclassify two unrelated
    wildcard-bearing structures as a "full" match -- exactly the false-positive this
    oracle exists to catch. The compare below never treats an empty/falsy key as
    equal to anything; ``full``/``block1`` require BOTH sides to be genuine keys.

BEST-EFFORT tier config (grounded in the brief — do not change):
    Orthonym(general_fallback=True, general_fallback_unverified=True,
              allow_aromatic_general=True)

Reuses the OPSIN idiom of `internal notes` (per-molecule
SIGALRM hang guard; ``opsin_parse``, which prefers the in-process JVM bridge over a
subprocess launch per name) — adapted, not reinvented. The whole probe call reserves
exactly one JVM slot via ``orthonym.jvm_budget.jvm_slots(1)`` per the environment rules;
never call ``probe`` from inside another ``jvm_slots`` block or a multiprocessing
worker.

Run standalone to re-derive the charged backlog against the grounded corpus (a dev split +
first-500 of chebi_5000.csv + first-500 of pubchem_2000.csv):

    .venv/bin/python internal notes

writes `internal notes` (abstain/wrong/unparseable/block1
rows only — RT-full rows are not backlog) and prints per-bucket counts, where bucket
follows the design doc's A/B/C split:

    A = scope-gap (rt == "abstain")
    B = name-quality (rt in {"wrong", "unparseable"})
    C = stereo (rt == "block1")
"""
from __future__ import annotations

import csv
import json
import signal
from pathlib import Path
from typing import Dict, List, Optional

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

REPO = Path(__file__).resolve().parents[2]

PER_MOL_TIMEOUT = 30  # seconds — matches charged_remeasure_be.py's giant-hang guard

BE_KWARGS = dict(general_fallback=True, general_fallback_unverified=True,
                  allow_aromatic_general=True)


class _TimeOut(Exception):
    pass


def _alarm(sig, frm):
    raise _TimeOut()


def _inchikey(smiles: Optional[str]) -> Optional[str]:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def probe(smiles_list: List[str]) -> Dict[str, dict]:
    """Best-effort-tier name + OPSIN round-trip every SMILES in ``smiles_list``.

    Returns ``{smiles: {"name": str|None, "rt": <class>}}`` per the module docstring.
    Reserves one JVM slot (``jvm_slots(1, purpose="-probe")``) for the whole call.
    """
    from orthonym import Orthonym
    from orthonym.validation.opsin_roundtrip import opsin_parse
    from orthonym.errors import is_failure_name
    from orthonym.jvm_budget import jvm_slots

    signal.signal(signal.SIGALRM, _alarm)
    namer = Orthonym(**BE_KWARGS)
    out: Dict[str, dict] = {}
    with jvm_slots(1, purpose="v34-probe"):
        for smi in smiles_list:
            want = _inchikey(smi)
            name = None
            signal.alarm(PER_MOL_TIMEOUT)
            try:
                name = namer.name(smi)
            except _TimeOut:
                name = None
            except Exception:
                name = None
            finally:
                signal.alarm(0)

            if not name or is_failure_name(name):
                out[smi] = {"name": None, "rt": "abstain"}
                continue

            got = opsin_parse(name)
            have = _inchikey(got) if got else None

            # Fail-closed compare -- see the module docstring's "Falsy-key guard"
            # note. Never let an empty/falsy key on EITHER side reach an
            # equality or prefix comparison; that is how a wildcard/dummy-atom
            # structure (MolToInchiKey -> '', no exception) could otherwise be
            # misclassified as a "full" round-trip match against another
            # unrelated wildcard-bearing structure.
            if not have:
                out[smi] = {"name": name, "rt": "unparseable"}
            elif not want:
                out[smi] = {"name": name, "rt": "wrong"}
            elif have == want:
                out[smi] = {"name": name, "rt": "full"}
            elif len(have) >= 14 and len(want) >= 14 and have[:14] == want[:14]:
                out[smi] = {"name": name, "rt": "block1"}
            else:
                out[smi] = {"name": name, "rt": "wrong"}
    return out


# --------------------------------------------------------------------------
# Standalone backlog re-derivation (Step 4 of the brief).
# --------------------------------------------------------------------------

def load_dev500() -> List[str]:
    d = json.load(open(REPO / "eval" / "splits" / "dev500.json"))
    return list(d["smiles"])[:500]


def load_csv_smiles(path: str, n: int) -> List[str]:
    out: List[str] = []
    seen = set()
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) < 2:
                continue
            smi = row[1].strip()
            if not smi or smi in seen:
                continue
            if Chem.MolFromSmiles(smi) is None:
                continue
            seen.add(smi)
            out.append(smi)
            if len(out) >= n:
                break
    return out


def _is_charged(smi: str) -> bool:
    """``perception.ions.get_ion_sites`` returns a genuine (non-internal) ionic centre."""
    from orthonym.perception.ions import get_ion_sites
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    sites = get_ion_sites(mol, exclude_internal=True)
    return bool(sites["cations"] or sites["anions"])


def main() -> None:
    corpora = {
        "dev500": load_dev500(),
        "chebi500": load_csv_smiles(str(REPO / "benchmarks" / "chebi_5000.csv"), 500),
        "pubchem500": load_csv_smiles(str(REPO / "benchmarks" / "pubchem_2000.csv"), 500),
    }
    total = sum(len(v) for v in corpora.values())

    charged_rows: List[tuple] = []  # (split, smiles)
    seen_smi = set()
    for split, smis in corpora.items():
        for smi in smis:
            if smi in seen_smi:
                continue
            if _is_charged(smi):
                seen_smi.add(smi)
                charged_rows.append((split, smi))
    print(f"corpus: {total} molecules ({', '.join(f'{k}={len(v)}' for k, v in corpora.items())}); "
          f"charged (get_ion_sites non-empty, non-internal): {len(charged_rows)}", flush=True)

    results = probe([smi for _, smi in charged_rows])

    backlog = []
    counts = {"A": 0, "B": 0, "C": 0}
    n_full = 0
    for split, smi in charged_rows:
        r = results[smi]
        rt = r["rt"]
        if rt == "full":
            n_full += 1
            continue
        if rt == "abstain":
            bucket = "A"
        elif rt in ("wrong", "unparseable"):
            bucket = "B"
        else:  # "block1"
            bucket = "C"
        counts[bucket] += 1
        backlog.append({
            "smiles": smi,
            "split": split,
            "name": r["name"],
            "rt": rt,
            "bucket": bucket,
        })

    out_path = REPO / ".planning" / "audit-v33" / "v34_charged_backlog.json"
    json.dump(
        {
            "corpus": {k: len(v) for k, v in corpora.items()},
            "n_charged": len(charged_rows),
            "n_full": n_full,
            "n_backlog": len(backlog),
            "counts": counts,
            "backlog": backlog,
        },
        open(out_path, "w"),
        indent=2,
    )
    print(f"backlog written: {out_path} (n={len(backlog)})", flush=True)
    print(f"A(scope-gap/abstain)={counts['A']}  B(name-quality/wrong+unparseable)={counts['B']}  "
          f"C(stereo/block1)={counts['C']}  full(not backlog)={n_full}", flush=True)


if __name__ == "__main__":
    main()
