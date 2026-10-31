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


def test_the_word_list_of_the_check_is_the_words_of_the_amino_acid_tables():
    # the check keeps a copy of the table words so that it reads no naming code; the copy may not drift
    from orthonym.data import amino_acids as aa
    from orthonym.validation.spelling.checks_natural_products import AMINO_ACID_TABLE_WORDS
    words = set()
    for table in (aa.STANDARD_AMINO_ACIDS, aa.NON_STANDARD_AMINO_ACIDS, aa.GENERAL_ONLY_AMINO_ACIDS):
        for value in table.values():
            if any(ch.isdigit() for ch in value):
                continue
            word = aa._RETAINED_NAME_DESCRIPTOR.sub("", value.strip(), count=1).lower()
            words.add(word.split(" acid")[0] if word.endswith(" acid") else word.split(" ")[0])
    assert words == AMINO_ACID_TABLE_WORDS


@pytest.mark.parametrize("name", [
    # E-AA (2026-10-31): names of the amino-acid tables and their ion forms, beyond Table 10.4,:50943)
    "pidolate", "L-alaninium chloride", "lysinium(1+)", "glycinium", "homocystine", "cysteamine", "sodium glutamate",
])
def test_a_name_of_the_amino_acid_tables_is_flagged(name):
    assert getattr(chapter_p10_name_is_no_pin(None, name), "rule", None) == "P-100", name


@pytest.mark.parametrize("name", [
    "aminoacetic acid", "methyl aminoacetate", "2-aminoethane-1-thiol", "2-aminoethane-1-sulfonic acid",
    "2-(1H-imidazol-5-yl)ethan-1-amine",
])
def test_a_systematic_amino_acid_name_is_not_flagged(name):
    assert chapter_p10_name_is_no_pin(None, name) is None
