import random

import pytest
from rdkit import Chem

from orthonym.rules import phane_pin

from ._helpers import mol, needs_phane_opsin, substituted

SPELLINGS = 30


def _name(smiles):
    m = mol(smiles)
    return phane_pin.general_phane_name(m) or phane_pin.linear_phane_name(Chem.Mol(m))


def _respelled(smiles, seed):
    m = mol(smiles)
    order = list(range(m.GetNumAtoms()))
    random.Random(seed).shuffle(order)
    return Chem.MolFromSmiles(Chem.MolToSmiles(Chem.RenumberAtoms(m, order), canonical=False))


# (:49684): a cyclic phane cites the descriptors of its double bonds from eight nodes on; a linear one always
NINE_NODES_E = "C1/C=C/c2cccc(c2)CCCc2cccc(c2)C1"
NINE_NODES_Z = "C1/C=C\\c2cccc(c2)CCCc2cccc(c2)C1"
SIX_NODES_SPECIFIED = "C1=C/C2=CC=C(C=C2)/C=C\\C2=CC=C1C=C2"
SIX_NODES_PLAIN = "C1=Cc2ccc(cc2)C=Cc2ccc1cc2"
ALCOHOL_R = "O[C@@H]1CCc2cccc(c2)CCCc2cccc(c2)C1"
ALCOHOL_S = "O[C@H]1CCc2cccc(c2)CCCc2cccc(c2)C1"
# (:44639): the descriptor of a centre inside a substituent is cited in the prefix of the substituent
BROMOETHYL_R = "C[C@@H](Br)C1=C2C=CC(=C1)CCC1=CC=C(C=C1)CC2"
# the book's example 2 (:50787): the printed name and the name of the rule differ, the structure is one (below)
EX2_SMILES = ("O=C(O)[C@@H]1CC[C@H](Oc2ccc(/C=C/O/C=C/c3ccc(O[C@@H]4CC[C@H](C(=O)O)C[C@@H]4O/C=C/c4ccccc4)cc3)cc2)"
              "[C@H](O/C=C/c2ccccc2)C1")
EX2_TAIL = ("-4,6,10,14,16-pentaoxa-1,19(1),7,13(1,4)-tetrabenzena-5,15(1,2)-dicyclohexananonadecaphane-2,8,11,17-tetraene-"
            "5^5,15^4-dicarboxylic acid")
EX2_BOOK = "(2E,5^1S,5^2R,5^5S,8E,11E,15^1S,15^2R,15^4R,17E)" + EX2_TAIL
EX2_RULE = "(2E,5^1R,5^2S,5^5R,8E,11E,15^1R,15^2S,15^4S,17E)" + EX2_TAIL


@pytest.mark.parametrize("line,name,smiles", substituted("stereo"), ids=lambda v: str(v)[:30])
def test_stereo_rows(line, name, smiles):
    #,: descriptors with composite locants at the front of the name, in the order of the locants
    assert _name(smiles) == name


@pytest.mark.parametrize("smiles,name", [
    (ALCOHOL_R, "(7R)-1,5(1,3)-dibenzenacyclononaphan-7-ol"),
    (ALCOHOL_S, "(7S)-1,5(1,3)-dibenzenacyclononaphan-7-ol"),
    (NINE_NODES_E, "(6E)-1,5(1,3)-dibenzenacyclononaphan-6-ene"),
    (NINE_NODES_Z, "(6Z)-1,5(1,3)-dibenzenacyclononaphan-6-ene"),
    (BROMOETHYL_R, "1^2-[(1R)-1-bromoethyl]-1,4(1,4)-dibenzenacyclohexaphane"),
    (SIX_NODES_PLAIN, "1,4(1,4)-dibenzenacyclohexaphane-2,5-diene"),
])
def test_derived_stereo_names(smiles, name):
    # rule-derived (no book row): the descriptor of a centre or of a double bond with its locant; the double bonds
    # of a cyclic skeleton of fewer than eight nodes are the exception of
    assert _name(smiles) == name


def test_a_specified_double_bond_of_a_small_cyclic_skeleton_is_not_named():
    # six nodes: the descriptors are omitted by, which OPSIN does not read as the configuration
    assert _name(SIX_NODES_SPECIFIED) is None


def test_a_stereogenic_double_bond_to_a_substituent_is_not_named():
    # the ylidene bond joins a skeleton atom and a substituent: no locant in the name of the skeleton
    assert _name("C/C=C1\\CCc2cccc(c2)CCCc2cccc(c2)C1") is None


@pytest.mark.parametrize("smiles", [
    ALCOHOL_R, NINE_NODES_E, BROMOETHYL_R,
    "C[C@@H]1[C@@H]2CCC(C1)CCC1=CC=C(C=C1)CC2",
    EX2_SMILES,
], ids=lambda v: v[:24])
def test_the_name_does_not_depend_on_the_order_of_the_atoms(smiles):
    names = {_name_of(_respelled(smiles, seed)) for seed in range(SPELLINGS)}
    assert len(names) == 1 and None not in names, names


def _name_of(m):
    return phane_pin.general_phane_name(m) or phane_pin.linear_phane_name(m)


def test_the_book_example_that_the_rule_numbers_the_other_way_is_named_by_the_rule():
    # the example 2 of:50787 numbers the meso structure from the end that gives '5^1S'; (j) (:3346) gives
    # the lower locant to R before S ('(2R,4S)-2,4-difluoropentane (PIN)',:3383), so the name begins '(2E,5^1R,...'. The two names are the two numberings of one structure (OPSIN
    # test below); the example 3 (:50791), where the Z bond decides, is a gold row.
    assert _name(EX2_SMILES) == EX2_RULE


@needs_phane_opsin
def test_the_two_numberings_of_the_meso_example_name_one_structure():
    from orthonym.jvm_budget import jvm_slots
    from orthonym.validation.opsin_roundtrip import opsin_parse
    with jvm_slots(1, purpose="pn4b"):
        book, rule = opsin_parse(EX2_BOOK), opsin_parse(EX2_RULE)
    canon = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in (book, rule)]
    assert canon[0] == canon[1] == Chem.MolToSmiles(mol(EX2_SMILES))


@needs_phane_opsin
@pytest.mark.parametrize("line,name,smiles", substituted("stereo"), ids=lambda v: str(v)[:30])
def test_stereo_rows_read_back_to_the_isomeric_smiles(line, name, smiles):
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="pn4b"):
        assert phane_pin.verified_or_none(name, smiles) == name
        assert phane_pin._isomeric_read_back(name, smiles)


@needs_phane_opsin
def test_a_name_with_the_other_configuration_is_voided():
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="pn4b"):
        assert phane_pin.verified_or_none("(7S)-1,5(1,3)-dibenzenacyclononaphan-7-ol", ALCOHOL_R) is None
        assert phane_pin.verified_or_none("(6Z)-1,5(1,3)-dibenzenacyclononaphan-6-ene", NINE_NODES_E) is None


# (g) (:3307) stands before (j) (:3346), and (j) before the citation order of the prefixes after the first (K10 to K12):
# two numberings that tie on the prefix cited first, and one of them gives R at the lower locant while the other cites a
# later prefix at the lower locant
G_BEFORE_J = "C[C@H]1C[C@@H](Cl)c2cccc(c2)C[C@@H](C)C[C@H](Br)c2cccc(c2)C1"
J_BEFORE_LATER = "c1ccccc1[C@H](Cl)c1ccc(cc1)[C@H](Br)c1ccc(cc1)[C@H](F)c1ccccc1"


def test_the_prefix_cited_first_decides_before_the_descriptors():
    # 'bromo' takes the lower locant (2) although the other numbering gives R to the lowest stereogenic locant
    assert _name(G_BEFORE_J) == "(2S,4R,7R,9R)-2-bromo-7-chloro-4,9-dimethyl-1,6(1,3)-dibenzenacyclodecaphane"


def test_the_preferred_descriptor_decides_before_the_prefixes_cited_after_the_first():
    # 'bromo' is at 4 in both numberings; 'chloro' at 2 would be cited first by the order of citation, but the
    # numbering with R at locant 2 is the one of (j): '(2R,4R,6S)-4-bromo-6-chloro-2-fluoro', not
    # '(2S,4R,6R)-4-bromo-2-chloro-6-fluoro'
    assert _name(J_BEFORE_LATER) == "(2R,4R,6S)-4-bromo-6-chloro-2-fluoro-1,7(1),3,5(1,4)-tetrabenzenaheptaphane"


@needs_phane_opsin
@pytest.mark.parametrize("smiles", [G_BEFORE_J, J_BEFORE_LATER], ids=["g_before_j", "j_before_later"])
def test_the_numbering_rule_names_read_back(smiles):
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="pn4b"):
        name = _name(smiles)
        assert phane_pin.verified_or_none(name, smiles) == name


@pytest.mark.parametrize("smiles", [G_BEFORE_J, J_BEFORE_LATER], ids=["g_before_j", "j_before_later"])
def test_the_numbering_rule_names_do_not_depend_on_the_order_of_the_atoms(smiles):
    assert len({_name_of(_respelled(smiles, seed)) for seed in range(SPELLINGS)}) == 1


# (j) (:3346) for the descriptors inside the prefixes: '1-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]benzene (PIN)' (:3399) gives the
# prefix with R the lower locant, and Z is preferred to E in the same way; the prefix cited first takes the lower locant, so the
# names of two prefixes that differ only in their descriptors are cited Z before E and R before S
EZ_PREFIXES = "C/C=C/C1Cc2ccc(cc2)C(/C=C\\C)Cc2ccc1cc2"
EZ_PREFIXES_SWAPPED = "C/C=C\\C1Cc2ccc(cc2)C(/C=C/C)Cc2ccc1cc2"
EZ_PREFIXES_NAME = "2-[(1Z)-prop-1-en-1-yl]-5-[(1E)-prop-1-en-1-yl]-1,4(1,4)-dibenzenacyclohexaphane"
RS_PREFIXES = "C[C@H](Cl)C1Cc2ccc(cc2)C([C@@H](C)Cl)Cc2ccc1cc2"
RS_PREFIXES_NAME = "2-[(1R)-1-chloroethyl]-5-[(1S)-1-chloroethyl]-1,4(1,4)-dibenzenacyclohexaphane"


@pytest.mark.parametrize("smiles,name", [(EZ_PREFIXES, EZ_PREFIXES_NAME), (EZ_PREFIXES_SWAPPED, EZ_PREFIXES_NAME),
                                         (RS_PREFIXES, RS_PREFIXES_NAME)], ids=["EZ", "EZ_swapped", "RS"])
def test_the_prefix_with_the_preferred_descriptor_takes_the_lower_locant(smiles, name):
    assert _name(smiles) == name


def test_the_descriptors_of_the_prefixes_are_ranked_by_the_cip_rule_not_by_the_string():
    from orthonym.rules.phane_pin.numbering import citation_key
    from orthonym.rules.phane_pin.stereo import prefix_codes
    assert citation_key("(1Z)-prop-1-en-1-yl") < citation_key("(1E)-prop-1-en-1-yl")     # the string order is E, Z
    assert citation_key("(1R)-1-chloroethyl") < citation_key("(1S)-1-chloroethyl")
    assert citation_key("bromo") < citation_key("(1Z)-prop-1-en-1-yl")                    # descriptors are no part of the alphabetical order
    assert prefix_codes("(1E)-prop-1-en-1-yl") == ["E"] and prefix_codes("(1R,2S)-1,2-dichloropropyl") == ["R", "S"]
    assert prefix_codes("prop-1-en-1-yl") == [] and prefix_codes("(1RS)-1-chloroethyl") == []


@pytest.mark.parametrize("smiles", [EZ_PREFIXES, RS_PREFIXES], ids=["EZ", "RS"])
def test_the_names_with_descriptors_in_prefixes_do_not_depend_on_the_order_of_the_atoms(smiles):
    assert len({_name_of(_respelled(smiles, seed)) for seed in range(SPELLINGS)}) == 1


@needs_phane_opsin
@pytest.mark.parametrize("smiles,name", [(EZ_PREFIXES, EZ_PREFIXES_NAME), (RS_PREFIXES, RS_PREFIXES_NAME)], ids=["EZ", "RS"])
def test_the_names_with_descriptors_in_prefixes_read_back(smiles, name):
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="pn4b"):
        assert phane_pin.verified_or_none(name, smiles) == name
        assert phane_pin._isomeric_read_back(name, smiles)
