"""One name for one molecule: a triazolyl ring whose two flanking substituents tie on the locant set.

The 4H-1,2,4-triazol-4-yl of maraviroc has its two substituents on C3 and C5, so the locant set {3,5} is the same either
way round and the first cited prefix in alphanumerical order takes the lower locant: "NUMBERING" (the Blue Book),
criterion (g) "the prefix cited first in alphanumerical order": '3-methyl-5-(propan-2-yl)', never '5-methyl-3-(propan-2-yl)'.
A lead of the roadmap saw '5-methyl-3' once in a loaded run; it was not reproduced (a dev split best-effort twice with 12
workers, 0 differences; 6 random orders of 120 rows in one process; 60 random spellings of the row; three PYTHONHASHSEEDs),
and this test keeps the equivalence that a spelling or an order of the input must not change the name (principle 4).
"""
import random

import pytest
from rdkit import Chem

from orthonym.cli import _tiered_namer

ROW = "Cc1nnc(C(C)C)n1C1CC2CCC(C1)N2CC[C@H](NC(=O)C1CCC(F)(F)CC1)c1ccc(O)cc1"
FRAGMENT = "Cc1nnc(C(C)C)n1C1CCCCC1"


def _spellings(smiles, n, seed):
    mol = Chem.MolFromSmiles(smiles)
    rnd = random.Random(seed)
    out = {smiles}
    for _ in range(n):
        out.add(Chem.MolToSmiles(mol, doRandom=True, canonical=False))
        rnd.random()
    return sorted(out)


@pytest.mark.unit
@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_every_spelling_of_the_triazolyl_ring_gets_the_one_name(tier):
    namer = _tiered_namer("pin", tier)
    names = {namer.name_tiered(s)["name"] for s in _spellings(FRAGMENT, 12, 7)}
    assert len(names) == 1, names


@pytest.mark.unit
@pytest.mark.opsin_gate
def test_every_spelling_of_the_row_gets_the_one_name():
    namer = _tiered_namer("pin", "best-effort")
    names = {namer.name_tiered(s)["name"] for s in _spellings(ROW, 4, 11)}
    assert len(names) == 1, names
    assert "3-methyl-5-(propan-2-yl)-4H-1,2,4-triazol-4-yl" in names.pop()
