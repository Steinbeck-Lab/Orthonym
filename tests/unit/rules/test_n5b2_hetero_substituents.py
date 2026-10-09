"""A fused ring system with ring heteroatoms as a '-yl' prefix (lane N5b step 2).

 (the Blue Book): free-valence locants are "as low as is consistent with any
established numbering of the parent hydride"; (:23710) gives the ring system its fusion
name, so the prefix is never a von Baeyer one. A ring nitrogen with indicated or added hydrogen
(or a hydro prefix) could stand for either tautomer in the InChI, so ``_substituent_name`` reads
the prefix back through OPSIN with its free valence capped by a hydrogen atom and requires the
input's tautomer (``_same_tautomer_as_radical``); before this lane every such prefix declined."""
import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import build_fused_substituent
from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate

# (acid, prefix name): the von Baeyer prefix each had is in the comment
PREFIXES = [
    ("OC(=O)CC1CCC2NCCC2C1", "(octahydro-1H-indol-5-yl)acetic acid"),            # (7-azabicyclo[4.3.0]nonan-3-yl)
    ("OC(=O)CC1CCCC2NCCC12", "(octahydro-1H-indol-4-yl)acetic acid"),
    ("OC(=O)CC1CCc2c(C1)[nH]c1ccccc21", "(2,3,4,9-tetrahydro-1H-carbazol-2-yl)acetic acid"),
    ("OC(=O)CC1CCC2OCCC2C1", "(octahydro-1-benzofuran-5-yl)acetic acid"),
    ("OC(=O)CC1COC2CCCCC2O1", "(octahydro-1,4-benzodioxin-2-yl)acetic acid"),
    ("OC(=O)CC1CC2CCOC2O1", "(hexahydrofuro[2,3-b]furan-2-yl)acetic acid"),
    ("OC(=O)CC1=CC2CCOC2OC1", "(2,3,3a,7a-tetrahydro-6H-furo[2,3-b]pyran-5-yl)acetic acid"),
    ("CC(=O)OC1CCC2OCCC2C1", "octahydro-1-benzofuran-5-yl acetate"),
]


@pytest.mark.parametrize("smiles,expected", PREFIXES)
def test_the_ring_is_named_by_fusion_nomenclature_as_a_prefix(smiles, expected):
    row = default_tier_row(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row
    assert name_best_effort(smiles)["name"] == expected
    assert_full_rt(expected, smiles)


@pytest.mark.parametrize("smiles,attach,expected", [
    ("c1ccc2[nH]ncc2c1", 6, "1H-indazol-3-yl"),
    ("c1ccc2c[nH]nc2c1", 4, "2H-indazol-3-yl"),            # the 2H tautomer keeps its 2H
    ("C1CCC2NCCC2C1", 0, "octahydro-1H-indol-5-yl"),
    ("c1ccc2[nH]ccc2c1", 6, "1H-indol-3-yl"),
])
def test_the_prefix_keeps_the_tautomer_of_the_input(smiles, attach, expected):
    assert build_fused_substituent(Chem.MolFromSmiles(smiles), attach) == expected


def test_a_prefix_with_a_stereocentre_still_declines():
    """build_fused_substituent declines on any chiral tag; the stereo prefix keeps its older name."""
    assert build_fused_substituent(Chem.MolFromSmiles("C1C[C@H]2CCOC2CC1"), 0) is None


# 1H- and 2H-indazole share one standard InChI; the fixed-H layer tells them apart
_INDAZOLE_1H = "c1ccc2[nH]ncc2c1"
_INDAZOLE_2H = "c1ccc2n[nH]cc2c1"


@pytest.mark.parametrize("smiles,name,expected", [
    (_INDAZOLE_1H, "1H-indazol-3-yl", True),
    (_INDAZOLE_2H, "2H-indazol-3-yl", True),
    (_INDAZOLE_1H, "2H-indazol-3-yl", False),       # a wrong tautomer declines
    (_INDAZOLE_2H, "1H-indazol-3-yl", False),
    (_INDAZOLE_1H, "garbage-yl", False),            # OPSIN cannot read it
])
def test_the_wrong_tautomer_of_a_prefix_declines(smiles, name, expected):
    from orthonym.rules.bridged_fused_pin import _same_tautomer_as_radical
    assert _same_tautomer_as_radical(Chem.MolFromSmiles(smiles), name) is expected
