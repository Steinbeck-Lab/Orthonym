"""A name built on a Chapter semisystematic name is not a PIN.

 INTRODUCTION (the Blue Book): "Preferred IUPAC names (PINs) are not identified for the compounds in this
Chapter" carbohydrates, amino acids, cyclitols, / nucleosides and nucleotides). The
name is unchanged; its label is lowered, the PIN tier declines it and the best-effort tier ships it.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.validation.spelling.checks_natural_products import chapter_p10_name_is_no_pin

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("name", [
    "D-glucose", "β-D-glucopyranose", "4-(β-D-glucopyranosyloxy)benzaldehyde",
    "methyl β-D-glucopyranosiduronic acid", "L-alanine", "myo-inositol",
    "adenosine 5'-(tetrahydrogen triphosphate)", "N-glycylglycine",
])
def test_a_chapter_p10_name_is_flagged(name):
    assert getattr(chapter_p10_name_is_no_pin(None, name), "rule", None) == "P-100", name


@pytest.mark.parametrize("name", [
    "propane-1,2,3-triol",                         # glycerol's PIN
    "(2S)-2-aminopropanoic acid",
    "benzenetelluronic acid",                      # an oxoacid of tellurium, not a uronic acid
    "4-methylbenzene-1-sulfonyl chloride",         # 'tosyl' is not a glycosyl
    "cyclohexane-1,2,3,4,5,6-hexol",
])
def test_a_systematic_name_is_not_flagged(name):
    assert chapter_p10_name_is_no_pin(None, name) is None


@pytest.mark.opsin_gate
def test_a_retained_sugar_name_is_not_labelled_a_pin():
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(
        "O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO")
    assert row["name"] == "D-glucose" and row["tier"] == "systematic_verified"
    assert Orthonym().name_tiered("O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO")["limit_code"] == "NO_VERIFIED_PIN"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O", "myo-inositol"),
    ("O[C@@H]1[C@@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O", "scyllo-inositol"),
])
def test_an_inositol_ships_at_best_effort_and_the_pin_tier_declines(smiles, expected):
    # OPSIN has no grammar for inositol names; the exact InChIKey of the meso stereoisomer is the licence
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert row["name"] == expected and row["is_pin"] is False
    assert Orthonym().name_tiered(smiles)["limit_code"] == "NO_VERIFIED_PIN"


def test_the_inositol_licence_is_the_structure_s_own_name():
    from orthonym.rules.inositols import is_inositol_name_of
    myo = "O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O"
    assert is_inositol_name_of(myo, "myo-inositol")
    assert not is_inositol_name_of(myo, "scyllo-inositol")             # another stereoisomer's name
    assert not is_inositol_name_of(Chem.MolToSmiles(Chem.MolFromSmiles("OC1CCCCC1")), "myo-inositol")
