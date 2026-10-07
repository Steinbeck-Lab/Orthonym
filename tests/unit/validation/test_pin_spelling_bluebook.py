"""The spelling checks on the Blue Book's own PIN rows: every disagreement is explained.

The 3,053 rows of ``benchmarks/bb_conformance/bb_conformance_report.json`` whose note marks a PIN.
The measure's ``expected_pin`` column lost the braces of 38 names in text extraction; for those
rows the spelling printed on the book line is taken from ``data/bb_pin_brace_spellings.json``
(the line and both spellings; ``N-{[({[(2-oxoethyl)amino]sulfanyl}methyl)amino]oxy}...`` at
the Blue Book against the column's ``N-[([(2-oxoethyl)amino]sulfanylmethyl)amino]oxy...``).
A check flags exactly the rows of ``EXPLAINED``, each with its reason; a new flag fails the test
until it is explained (or the check corrected).
"""
import json
import re
from pathlib import Path

from rdkit import Chem, RDLogger

from orthonym.validation.pin_spelling import check_pin_spelling
from tests.unit.validation._book_explained import LISTING_ROWS, OWN_RULE_ROWS

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "benchmarks/bb_conformance/bb_conformance_report.json"
BRACES = Path(__file__).parent / "data/bb_pin_brace_spellings.json"

#: (book line, PIN as printed, rule) -> why the check and the row disagree: the rows both book
#: corpora share (``_book_explained``), and the row of this corpus only
EXPLAINED = {
    **LISTING_ROWS,
    **OWN_RULE_ROWS,
    (11509, "isochromene,and xanthene", "P-14.7.1"): "two names extracted into one row",
}


def _book_rows():
    recs = json.loads(REPORT.read_text())["records"]
    braces = {(b["def_id"], b["line"], b["expected"]): b["book"] for b in json.loads(BRACES.read_text())}
    rows = []
    for r in recs:
        kind = re.sub(r".*\((.*)\).*", r"\1", r["note"])
        m = re.search(r":(\d+)", r["note"])
        if not kind.startswith("pin") or not m:
            continue
        line = int(m.group(1))
        name = braces.get((r["def_id"], line, r["expected_pin"]))
        if name is None:
            name = re.sub(r"(?<=[,\-λ]) +", "", r["expected_pin"])
        rows.append((line, r["smiles"], name))
    return rows


def test_every_disagreement_with_a_book_pin_is_explained():
    rows = _book_rows()
    assert len(rows) == 3053, len(rows)
    flagged = {}
    for line, smiles, name in rows:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        for f in check_pin_spelling(mol, name, strict=True):
            flagged[(line, name, f.rule)] = f.detail
    unexplained = {k: v for k, v in flagged.items() if k not in EXPLAINED}
    assert not unexplained, unexplained
    missing = set(EXPLAINED) - set(flagged)
    assert not missing, missing


def test_the_brace_spellings_are_the_column_with_its_braces():
    for b in json.loads(BRACES.read_text()):
        assert re.sub(r"[\s{}]", "", b["book"]) == re.sub(r"[\s{}]", "", b["expected"]), b
        assert "{" in b["book"], b
