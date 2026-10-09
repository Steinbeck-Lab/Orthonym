"""Every tier names an all-carbon fused ring system by fusion nomenclature (lane N5b step 1).

 (the Blue Book): "Fusion nomenclature gives preferred IUPAC names only to
compounds having at least two rings of at least five or more members. [...] When fusion names are
not allowed, unsaturated von Baeyer ring system names are preferred IUPAC names", so for this
class a von Baeyer descriptor is never the preferred name. The default tier ships the name as a
certified PIN and best-effort gives the same name; both read back to the input's full InChIKey."""
import pytest
from rdkit import Chem

from orthonym.rules.fused_rings import _name_saturated_fused_carbocyclic
from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate

# (smiles, name) -- the von Baeyer name each had is in the comment
ROUTED = [
    ("OC1CCC2CCCC2C1", "octahydro-1H-inden-5-ol"),                       # bicyclo[4.3.0]nonan-3-ol
    ("O=C1CCC2CCCCC2C1", "octahydronaphthalen-2(1H)-one"),               # bicyclo[4.4.0]decan-3-one
    ("O=C1CCC2CCCC2C1", "octahydro-5H-inden-5-one"),                     # bicyclo[4.3.0]nonan-3-one
    ("OC1CCC2CC=CC2C1", "3a,4,5,6,7,7a-hexahydro-1H-inden-5-ol"),        # bicyclo[4.3.0]non-8-en-3-ol
    ("c1ccc2c(c1)CC1CCCC21", "1,2,3,3a,8,8a-hexahydrocyclopenta[a]indene"),   # tricyclo[6.4.0.0^2,6]...
    ("C1CCC2C(C1)CCC1C2CCC2CCCC12", "hexadecahydro-1H-cyclopenta[a]phenanthrene"),
    ("CC12CCCCC1CCC1C2CCC2(C)C(O)CCC12",
     "10,13-dimethylhexadecahydro-1H-cyclopenta[a]phenanthren-17-ol"),
    ("BrC12C=CC=CC1C=CC=C2", "4a-bromo-4a,8a-dihydronaphthalene"),       # the Blue Book
    ("O[C@H]1CC[C@H]2CCC[C@@H]2C1", "(3aR,5S,7aR)-octahydro-1H-inden-5-ol"),
    ("C1CCC2CCCC3CCCC1C23", "dodecahydro-1H-phenalene"),                 # tricyclo[7.3.1.0^5,13]tridecane
    ("C1CCC2CCCCCC2C1", "decahydro-1H-benzo[7]annulene"),
    ("CC(C)C1CCC2CCCC2C1", "5-(propan-2-yl)octahydro-1H-indene"),
    ("C1CCC(CC1)C1CCC2CCCC2C1", "5-cyclohexyloctahydro-1H-indene"),
]


@pytest.mark.parametrize("smiles,expected", ROUTED)
def test_both_tiers_give_the_fusion_name(smiles, expected):
    row = default_tier_row(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified" and row["is_pin"] is True, row
    assert name_best_effort(smiles)["name"] == expected
    assert_full_rt(expected, smiles)


# names right before this lane; they stay byte-identical, and the PIN tier still ships them
UNCHANGED = [
    ("OC1CCC2CCCCC2C1", "decahydronaphthalen-2-ol"),
    ("CC1CCC2CCCCC2C1", "2-methyldecahydronaphthalene"),
    ("C1CCC2CCCC2C1", "octahydro-1H-indene"),
    ("Oc1ccc2CCCCc2c1", "5,6,7,8-tetrahydronaphthalen-2-ol"),
    ("O=C1CCCc2ccccc12", "3,4-dihydronaphthalen-1(2H)-one"),
    ("C1=CC2CCCCC2C1", "3a,4,5,6,7,7a-hexahydro-1H-indene"),
    ("C1CCC2CCC3CCCCC3C2C1", "tetradecahydrophenanthrene"),
    ("C1=CCCC2CCCCC12", "1,2,3,4,4a,5,6,8a-octahydronaphthalene"),
]


@pytest.mark.parametrize("smiles,expected", UNCHANGED)
def test_names_that_were_right_stay_byte_identical(smiles, expected):
    assert default_tier_row(smiles)["name"] == expected
    assert name_best_effort(smiles)["name"] == expected


VON_BAEYER = [
    ("C1CCC2CCC2C1", "bicyclo[4.2.0]octane"),         # one ring of five or more:
    ("C1CC2CCC1C2", "bicyclo[2.2.1]heptane"),         # bridged
    ("OC1CCC2(CC1)CCCC2", "spiro[4.5]decan-8-ol"),    # spiro
]


@pytest.mark.parametrize("smiles,expected", VON_BAEYER)
def test_systems_outside_the_class_keep_their_names(smiles, expected):
    assert name_best_effort(smiles)["name"] == expected


def test_the_saturated_table_names_no_six_seven_system_as_heptalene():
    # heptalene is the 7/7 system; the 6/7 system is benzo[7]annulene
    assert _name_saturated_fused_carbocyclic(Chem.MolFromSmiles("C1CCC2CCCCCC2C1")) is None


def test_a_decorated_hydroazulene_has_a_round_tripping_fusion_name():
    # was (1R,3S,5S,6R,7S)-3-(hydroxymethyl)-6,9,9-trimethyl-2-methylidenebicyclo[5.3.0]decane-1,5-diol
    smiles = "C=C1[C@@H](CO)C[C@H](O)[C@H](C)[C@@H]2CC(C)(C)C[C@]12O"
    name = name_best_effort(smiles)["name"]
    assert name == ("(3aR,5S,7S,8R,8aS)-5-(hydroxymethyl)-2,2,8-trimethyl-4-methylidene"
                    "octahydroazulene-3a,7(1H)-diol")
    assert_full_rt(name, smiles)
