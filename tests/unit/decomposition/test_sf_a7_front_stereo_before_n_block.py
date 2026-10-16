"""SF lane A item 7: the descriptor block of the parent stays at the front of the complete name.

 "NAMING OF STEREOISOMERS" (the Blue Book): the descriptors of the parent "are placed at the
front of the complete name when related to the parent structure"; (:4826); the Blue Book's own
'(1S,3R)-3-amino-N-(3-amino-3-iminopropyl)cyclopentane-1-carboxamide (PIN)' (:48077) has the block in
front of the N-substituent. The amide, sulfonamide and N-acyl assemblers of decomposition/fragment_assembly
wrote 'N-<group>' in front of the name of the other fragment, so a fragment name that opens with its block
shipped 'N-methyl(2S)-2-...butanamide'; the block now goes first. The assembler of the amine names
(composer._assemble_amine_name) wrote its prefixes in front of a base name that opens with the block
('...-N-propyl(2S,3R)-butan-1-amine'); it uses composition_primitives.prefixes_after_front_stereo.

The N-group itself is not moved among the other prefixes: the label of a name whose N-prefix then stands
out of the alphanumerical order,:3436) is lowered as before (``n_prefix_breaks_citation_order``,
checked against the name without its block). Changed asserted values: none; the rows below are new.
Machine-confirmed: OPSIN 2.9.0 reads every name back to the full InChIKey of its structure, and the tests
fail on the tree before the change (an A/B check). Unverified by the book: the position of the N-group among
the other prefixes of d1914-like names ('N,3,3-trimethyl' would join the two methyls,, which this
change does not decide.
"""
import pytest

from orthonym.assembly.composition_primitives import prefixes_after_front_stereo
from orthonym.decomposition import fragment_assembly as fa
from orthonym.namer import Orthonym

pytestmark = pytest.mark.unit

BEST_EFFORT = dict(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)


@pytest.mark.parametrize("name, expected", [
    ("(2S)-2-chloro", ("(2S)-", "2-chloro")),
    ("(1R,2S)-1-methyl", ("(1R,2S)-", "1-methyl")),
    ("(E)-but-2-enamide", ("(E)-", "but-2-enamide")),
    ("rel-(1R,2S)-x", ("rel-(1R,2S)-", "x")),
    ("2-chloro", ("", "2-chloro")),
    ("(2-chloroethyl)amine", ("", "(2-chloroethyl)amine")),     # an enclosed prefix is no block
])
def test_split_front_stereo(name, expected):
    assert fa.split_front_stereo(name) == expected


def test_prefixes_after_front_stereo():
    assert prefixes_after_front_stereo("2,3-dimethyl-N-propyl", "(2S,3R)-butan-1-amine") == \
        "(2S,3R)-2,3-dimethyl-N-propylbutan-1-amine"
    assert prefixes_after_front_stereo("2,3-dimethyl-N-propyl", "butan-1-amine") == \
        "2,3-dimethyl-N-propylbutan-1-amine"
    assert prefixes_after_front_stereo("", "(2S)-butan-2-ol") == "(2S)-butan-2-ol"


@pytest.mark.parametrize("smiles, expected", [
    # the amide split of an N-methyl amide of a chiral acid (PB final evals, d1914)
    ("CNC(=O)[C@@H](NC(=O)[C@H](CC(C)C)[C@H](O)C(=O)NO)C(C)(C)C",
     "(2S)-N-methyl-2-{(2R)-2-[(1S)-1-hydroxy-2-(hydroxyamino)-2-oxoethyl]-4-methylpentanamido}"
     "-3,3-dimethylbutanamide"),
    # the acyl on a ring nitrogen, the Z double bond of the ring name
    ("C/C=C1/C(O)=C(C(=O)CCCCCCC)C(=O)N1C(C)=O",
     "(5Z)-N-acetyl-5-ethylidene-4-hydroxy-3-octanoyl-2-oxo-2,5-dihydro-1H-pyrrole"),
    # an N-aryl carboxamide of a fused heterocycle (sample, p41005272)
    ("CCCC1=NN=C2N1N[C@@H]([C@H](S2)C(=O)NC3=CC=CC=C3)C4=CC=C(C=C4)C",
     "(6R,7S)-N-phenyl-6-(4-methylphenyl)-3-propyl-6,7-dihydro-5H-[1,2,4]triazolo[3,4-b][1,3,4]"
     "thiadiazine-7-carboxamide"),
])
def test_amide_assembler_cites_the_parent_block_first(smiles, expected):
    assert Orthonym(**BEST_EFFORT).name(smiles) == expected


def test_amide_assemblers_called_directly():
    assert fa._assemble_amide({"acid": "(2S)-2-chlorobutanoic acid", "amine": "methanamine"}, "pin",
                              {"acid": "CC[C@H](Cl)C(=O)O", "amine": "CN"}) == \
        "(2S)-N-methyl-2-chlorobutanamide"
    assert fa._assemble_sulfonamide({"acid": "(2S)-2-chlorobutane-1-sulfonic acid", "amine": "methanamine"},
                                    "pin") == "(2S)-N-methyl-2-chlorobutane-1-sulfonamide"


def test_a_name_without_a_block_is_unchanged():
    assert fa._assemble_amide({"acid": "2-chlorobutanoic acid", "amine": "methanamine"}, "pin",
                              {"acid": "CCC(Cl)C(=O)O", "amine": "CN"}) == "N-methyl-2-chlorobutanamide"


def test_amine_assembler_cites_the_parent_block_first():
    """The explicit-hydrogen drawing of (the default tier shipped '...-N-propyl(2S,3R)-butan-1-amine')."""
    smiles = ("[H]c1c(F)c([H])c(C([H])([H])[C@@]([H])(C([H])([H])[H])[C@]([H])(C([H])([H])[H])"
              "C([H])([H])N([H])C([H])([H])C([H])([H])C([H])([H])[H])c([H])c1F")
    assert Orthonym().name(smiles) == "(2S,3R)-4-(3,5-difluorophenyl)-2,3-dimethyl-N-propylbutan-1-amine"


def test_the_label_still_follows_the_citation_order_of_the_name_without_its_block():
    """'N-methyl' cited before '2-{...hydroxy...}' is out of the alphanumerical order whatever the
    block in front: the check reads the name without it."""
    rest = "2-{(2R)-2-[(1S)-1-hydroxy-2-oxoethyl]pentanamido}-3,3-dimethylbutanamide"
    assert fa.n_prefix_breaks_citation_order(["methyl"], rest) is True
    assert fa.n_prefix_breaks_citation_order(["methyl"], "(2S)-" + rest) is True       # the block alone does
    assert fa.n_prefix_breaks_citation_order(["chloro"], "2-methylbutanamide") is False
