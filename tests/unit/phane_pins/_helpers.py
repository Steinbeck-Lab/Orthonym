import csv
from pathlib import Path
from typing import List, Optional, Tuple

from rdkit import Chem

from orthonym.validation.spelling.cyclophane import is_phane_name
from tests.support.default_tier import default_tier_row
from tests.support.opsin_marks import needs_phane_opsin  # noqa: F401 (re-exported to the tests)
from tests.support.rt_assert import assert_full_rt, name_best_effort  # noqa: F401

DATA = Path(__file__).parent / "data"


def _read(fname: str) -> List[dict]:
    with open(DATA / fname, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def gold(kind: Optional[str] = None, *, replaced: Optional[bool] = None,
         unsaturated: Optional[bool] = None) -> List[Tuple[str, str, str]]:
    """(bb_line, name, smiles) of bb_phane_gold.tsv, filtered."""
    out = []
    for r in _read("bb_phane_gold.tsv"):
        if kind is not None and r["kind"] != kind:
            continue
        if replaced is not None and (r["replaced"] == "1") != replaced:
            continue
        if unsaturated is not None and (r["unsaturated"] == "1") != unsaturated:
            continue
        out.append((r["bb_line"], r["name"], r["smiles"]))
    return out


def substituted(cls: str) -> List[Tuple[str, str, str]]:
    return [(r["bb_line"], r["name"], r["smiles"])
            for r in _read("bb_phane_substituted.tsv") if r["class"] == cls]


def mol(smiles: str) -> Chem.Mol:
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, smiles
    return m




def phane_row(smiles: str) -> dict:
    """Default-tier row; asserts a phane name that round-trips to the full input."""
    row = default_tier_row(smiles)
    assert is_phane_name(row["name"]), row
    assert_full_rt(row["name"], smiles)
    return row
