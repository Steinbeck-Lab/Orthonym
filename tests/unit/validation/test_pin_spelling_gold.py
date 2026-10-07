"""The spelling checks on the gate's PIN gold rows: no expected PIN fails, except the explained rows.

A check that failed a gold row's ``expected_pin`` would make the default tier decline a gate
target or protect row. The rows are those of ``benchmarks/the gold set/packs/*.json``,
``gold_pins.json`` and ``among_rings_gold.json``. Names of the exact-match natural-product list
(``NAME_EXACT_NP_PARENTS``) are not read at the call site (they keep the label of their path,
user decision 2026-09-30), so they are left out here too.
"""
import json
from pathlib import Path

from rdkit import Chem, RDLogger

from orthonym.data.natural_products import NAME_EXACT_NP_PARENTS
from orthonym.validation.pin_spelling import check_pin_spelling

RDLogger.DisableLog("rdApp.*")
ORACLE = Path(__file__).resolve().parents[3] / "benchmarks/pin_oracle"

#: (file, def_id, expected_pin) -> why the check fails it
EXPLAINED = {
    # 'pyrrolizidine' is not a retained name retains 'pyrrolizine', with indicated
    # hydrogen,:11628); the default tier already declines this row (it is listed in
    # default_tier_non_pin_rows.json and judged at best-effort)
    ("rings_numbering.json", "P13B-A-S57-17", "2-methylpyrrolizidine"): "P-14.7.1, a non-PIN row",
    # (:24088) "Phane names are preferred IUPAC names rather than ring assembly names
    # when seven or more rings or ring systems are present": a correct ring assembly name that is
    # not the PIN (the phane name, which OPSIN 2.9.0 cannot read, is); the default tier declines
    # this protect row (listed in default_tier_non_pin_rows.json) and judges it at best-effort
    ("characteristic_groups.json", "W2E-P2IP-P2",
     "1,1':4',1'':4'',1''':4''',1'''':4'''',1''''':4''''',1''''''-septiphenyl"):
        "P-52.2.5.1, a non-PIN row",
    # a parent of the natural-product route (four pyrroles on a chain of seven nodes): the
    # call site does not read the names of the routes (namer._SPELLING_CHECK_SKIPPED_CLASSES;
    #:50943, no PINs in Chapter, so the gate target keeps its label
    ("np_parents.json", "P14-NP-BILINE", "21H-biline"): "P-52.2.5.1, a P-10 route name the call site does not read",
}


def _gold_rows():
    files = sorted((ORACLE / "packs").glob("*.json")) + [ORACLE / "gold_pins.json", ORACLE / "among_rings_gold.json"]
    for f in files:
        data = json.loads(f.read_text())
        for r in data["rows"] if isinstance(data, dict) else data:
            if r.get("smiles") and r.get("expected_pin"):
                yield f.name, r.get("def_id"), r["smiles"], r["expected_pin"]


def test_no_gold_pin_fails_a_spelling_check():
    fired = {}
    n = 0
    for fname, def_id, smiles, name in _gold_rows():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or name in NAME_EXACT_NP_PARENTS:
            continue
        n += 1
        fails = check_pin_spelling(mol, name, strict=True)
        if fails:
            fired[(fname, def_id, name)] = [(f.rule, f.detail) for f in fails]
    assert n > 1500, n
    assert {k: v for k, v in fired.items() if k not in EXPLAINED} == {}
    assert set(EXPLAINED) <= set(fired)


def test_the_explained_gold_row_is_a_default_tier_non_pin_row():
    rows = json.loads((ORACLE / "default_tier_non_pin_rows.json").read_text())
    ids = {r.get("def_id") for r in (rows["rows"] if isinstance(rows, dict) else rows)}
    assert "P13B-A-S57-17" in ids
