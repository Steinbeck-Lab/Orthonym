"""A ring sulfur acid group beside a ketone: the acid is the suffix and the ketone an acyl prefix.

 (seniority of classes): sulfonic and sulfinic acids, their halides and amides rank above ketones, so the ketone
is cited as an acyl prefix, 'acetyl'): '4-acetylbenzene-1-sulfonyl chloride'. The benzene producer let
the acyl prefix in beside a carbon-attached suffix only, so these fell to the general engine
('1-acetyl-4-sulfobenzene', no PIN) (rules.benzene._ring_bears_senior_suffix).
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("CC(=O)c1ccc(S(=O)(=O)O)cc1", "4-acetylbenzene-1-sulfonic acid"),
    ("CC(=O)c1cccc(S(=O)(=O)O)c1", "3-acetylbenzene-1-sulfonic acid"),
    ("CC(=O)c1ccccc1S(=O)(=O)O", "2-acetylbenzene-1-sulfonic acid"),
    ("CC(=O)c1ccc(S(=O)(=O)Cl)cc1", "4-acetylbenzene-1-sulfonyl chloride"),
    ("CC(=O)c1ccc(S(=O)Cl)cc1", "4-acetylbenzene-1-sulfinyl chloride"),
    ("CC(=O)c1ccc(S(=O)(=O)F)cc1", "4-acetylbenzene-1-sulfonyl fluoride"),
    ("CC(=O)c1ccc(S(N)(=O)=O)cc1", "4-acetylbenzene-1-sulfonamide"),
    ("CC(=O)c1ccc(S(=O)(=O)NC)cc1", "4-acetyl-N-methylbenzene-1-sulfonamide"),
    ("CC(=O)c1ccc(S(=O)O)cc1", "4-acetylbenzene-1-sulfinic acid"),
    ("CCC(=O)c1ccc(S(=O)(=O)O)cc1", "4-propanoylbenzene-1-sulfonic acid"),
    ("O=C(c1ccccc1)c1ccc(S(=O)(=O)O)cc1", "4-benzoylbenzene-1-sulfonic acid"),
    ("CC(=O)c1ccc(C)c(S(=O)(=O)Cl)c1", "5-acetyl-2-methylbenzene-1-sulfonyl chloride"),
    # unchanged: no senior suffix, the ketone is the principal group
    ("CC(=O)c1ccc(C)cc1", "1-(4-methylphenyl)ethan-1-one"),
])
def test_ketone_is_an_acyl_prefix_beside_a_sulfur_acid(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)
