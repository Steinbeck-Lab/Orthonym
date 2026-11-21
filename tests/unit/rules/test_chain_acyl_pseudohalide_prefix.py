"""An acyl pseudohalide at the end of a chain under a senior group: 'oxo' plus the pseudohalide prefix.

 'Acyl halides and pseudohalides as substituent groups' (the Blue Book): "(3) at the end of an
acyclic carbon chain by a prefix denoting the halide or pseudohalide group and the prefix 'oxo'... method (3)
generates preferred IUPAC names for acyclic carbon chains"; '(3) 4-bromo-3,4-dioxobutanoic acid (PIN)'. On a ring,
method (1): '2-carbonocyanidoyl-5-methylbenzoyl chloride (PIN)' (:31543). The chain took the cyano carbon of an
acyl cyanide as a chain member and the chain producer cited the group again as '(azidocarbonyl)', a different
molecule ('4-(azidocarbonyl)butanoic acid'), so these had no PIN (perception.chains, rules.polyfunctional).
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("N#CC(=O)CCC(=O)O", "4-cyano-4-oxobutanoic acid"),
    ("N#CC(=O)CCC(=O)OC", "methyl 4-cyano-4-oxobutanoate"),
    ("N#CC(=O)C(C)CC(=O)O", "4-cyano-3-methyl-4-oxobutanoic acid"),
    ("[N-]=[N+]=NC(=O)CCC(=O)O", "4-azido-4-oxobutanoic acid"),
    ("O=C=NC(=O)CCC(=O)O", "4-isocyanato-4-oxobutanoic acid"),
    ("S=C=NC(=O)CCC(=O)O", "4-isothiocyanato-4-oxobutanoic acid"),
    ("N#CC(=O)c1ccc(C(=O)O)cc1", "4-carbonocyanidoylbenzoic acid"),
    # an acyl cyanide on a branch: the branch is named whole, also in the chain comparison's sort keys
    ("N#CC(=O)CC(CCC)C(=O)O", "2-(2-cyano-2-oxoethyl)pentanoic acid"),
    ("CC1=C(CCCC/C=C(/CC(=O)C#N)C(=O)O)C(=O)OC1=O",
     "(2Z)-2-(2-cyano-2-oxoethyl)-7-(4-methyl-2,5-dioxo-2,5-dihydrofuran-3-yl)hept-2-enoic acid"),
    # unchanged: the pseudohalide is the principal group, or a plain nitrile
    ("CCCC(=O)C#N", "butanoyl cyanide"),
    ("N#CC(=O)CCC(N)=O", "4-amino-4-oxobutanoyl cyanide"),
    ("N#CCCC(=O)O", "3-cyanopropanoic acid"),
])
def test_chain_acyl_pseudohalide_pins(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)
