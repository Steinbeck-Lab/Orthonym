"""A descriptor block stands before the word that holds its unit (program 20, 2026-11-08; SureChEMBL rows with a
block fronted to an ester's or a salt's first word, 66 of them labelled pin_verified).

 "Naming of stereoisomers" (the Blue Book): "When they relate to substituent groups, they are cited at
the front of the corresponding prefix. They are preceded by a numerical or letter locant to describe the position of
the stereogenic unit when such locants are present". '(2Z)-ethyl anilino(hydroxyimino)acetate' put the block on
'ethyl', which has no C-2; the parent's block belongs before the acid word when the unit is in the acid.
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.opsin_gate

_WIDE = dict(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)


@pytest.mark.parametrize("smiles, name", [
    ("CCOC(=O)/C(=N/O)Nc1ccccc1", "ethyl (2Z)-anilino(hydroxyimino)acetate"),
    ("C[C@H](N)C(=O)[O-].[Na+]", "sodium (2S)-2-aminopropanoate"),
    ("CC[C@H](C)C(=O)OCC", "ethyl (2S)-2-methylbutanoate"),
    # the units on the alcohol side: the block stays in front of the alcohol word
    ("CC(=O)O[C@@H](c1cc(O)cc(O)c1)C(CC)[C@H](C)Cl", "(1R,3S)-3-chloro-1-(3,5-dihydroxyphenyl)-2-ethylbutyl acetate"),
    ("CC(=O)OC[C@@H](N)C(C)C", "(2S)-2-amino-3-methylbutyl acetate"),
    # a unit on each side: each block before its own word
    ("C[C@@H](O)C(=O)O[C@@H](C)CC", "(2S)-butan-2-yl (2R)-2-hydroxypropanoate"),
    # one word, and an acyl halide: the front is the parent's word
    ("C[C@H](N)C(=O)O", "(2S)-2-aminopropanoic acid"),
    ("C/C=C/C(=O)Cl", "(2E)-but-2-enoyl chloride"),
])
def test_the_block_stands_before_the_word_of_its_unit(smiles, name):
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")


@pytest.mark.parametrize("smiles, name", [
    ("COC(C[S@@](=O)C1=CC=CC=C1)=O", "(R)-methyl (benzenesulfinyl)acetate"),
    ("CN([P@](OC)(=O)Cl)C", "(R)-methyl N,N-dimethylphosphoramidochloridate"),
])
def test_a_lone_hetero_centre_fronted_off_its_word_is_below_the_pin(smiles, name):
    # the fronted block is a valid name that OPSIN reads back, never the PIN: the wider tiers keep it, the PIN tier
    # declines (it was labelled pin_verified)
    assert Orthonym().name_tiered(smiles)["tier"] == "abstain"
    row = Orthonym(**_WIDE).name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "systematic_verified")


def test_a_lone_hetero_centre_in_a_one_word_name_keeps_its_pin():
    # (the Blue Book) '(S)-(methanesulfinyl)benzene (PIN)'
    row = Orthonym().name_tiered("C[S@](=O)c1ccccc1")
    assert (row["name"], row["tier"]) == ("(S)-(methanesulfinyl)benzene", "pin_verified")


@pytest.mark.parametrize("smiles", ["C[P@](=O)([O-])OCC.[Na+]", "C[P@](=O)([O-])Oc1ccccc1.[NH4+]"])
def test_a_salt_with_a_lone_hetero_centre_keeps_its_wider_tier_name(smiles):
    # review 2026-11-08: a salt is not judged by the fronted-block rule (its producer would drop a name labelled below
    # the PIN): best-effort never goes down
    assert Orthonym(**_WIDE).name_tiered(smiles)["name"]
