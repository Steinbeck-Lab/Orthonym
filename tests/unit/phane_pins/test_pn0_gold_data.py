"""The Blue Book phane PINs used as gold rows by the phane tests.

The names are the book's, each printed with "(PIN" in and examples; the structure of a row is the
reading of the name by OPSIN 9f30df5cc22c, which is the check of the extraction and not the authority.
"""
from collections import Counter

import pytest
from rdkit import Chem

from tests.support.rt_assert import _full_inchikey, _independent_parse

from ._helpers import _read, gold, mol, needs_phane_opsin, substituted

# (1) and (2): a cyclic phane skeleton is a cycle of nodes, a linear one a path of nodes
# (edges minus nodes of the node graph is 0 and -1)
GOLD_COUNTS = {
    ("cycle", False, False): 19, ("cycle", False, True): 4, ("cycle", True, False): 6,
    ("path", False, False): 32, ("path", False, True): 6, ("path", True, False): 17,
}
SUBSTITUTED_COUNTS = {"prefix": 4, "suffix": 1, "compound_prefix": 3, "stereo": 2, "indicated_hydrogen": 3,
                      "hydro": 3, "skeleton": 4, "phane_as_prefix": 1}


def test_gold_smiles_parse():
    for _, _, smiles in gold() + [r for cls in SUBSTITUTED_COUNTS for r in substituted(cls)]:
        assert Chem.MolFromSmiles(smiles) is not None, smiles


def test_gold_names_are_unique():
    rows = _read("bb_phane_gold.tsv") + _read("bb_phane_substituted.tsv")
    names = [r["name"] for r in rows]
    assert len(names) == len(set(names))
    canonical = [Chem.MolToSmiles(mol(r["smiles"])) for r in rows]
    assert len(canonical) == len(set(canonical))


def test_gold_counts():
    rows = _read("bb_phane_gold.tsv")
    assert len(rows) == 84
    got = Counter((r["kind"], r["replaced"] == "1", r["unsaturated"] == "1") for r in rows)
    assert dict(got) == GOLD_COUNTS
    assert sum(n for (kind, _, _), n in GOLD_COUNTS.items() if kind == "cycle") == 29
    assert sum(n for (kind, _, _), n in GOLD_COUNTS.items() if kind == "path") == 55


def test_substituted_counts():
    got = Counter(r["class"] for r in _read("bb_phane_substituted.tsv"))
    assert dict(got) == SUBSTITUTED_COUNTS


def test_no_gold_row_is_both_replaced_and_unsaturated():
    assert not [r for r in _read("bb_phane_gold.tsv") if r["replaced"] == "1" and r["unsaturated"] == "1"]


@needs_phane_opsin
@pytest.mark.parametrize("line,name,smiles", gold() + [r for cls in SUBSTITUTED_COUNTS for r in substituted(cls)],
                         ids=lambda v: str(v)[:30])
def test_gold_smiles_read_back_from_the_names(line, name, smiles):
    got = _independent_parse(name, smiles)
    assert got and _full_inchikey(got) == _full_inchikey(smiles), (name, got)
