"""Phane names with double and triple bonds in the bridges, and the read-back of a phane name.

The producer (``rules.phane.build_phane_pin``) used to name every monocycle of benzene amplificants and
bridge atoms as if the bridges were saturated, so a stilbene or tolane bridge gave the name of the
saturated phane, and the ``phane`` grammar-gap carve-out shipped the name unread (OPSIN 2.9.0 has no phane
grammar): 40 of the 66 names of the CO audit sample (2026-10-17) were the name of another molecule.

The rules:

* Phane parent hydrides, Double bonds in amplificants and in simplified phane skeletons
  (the Blue Book): "The presence of one or more double or triple bonds in an otherwise saturated
  phane parent hydride, except in amplificants with Hantzsch-Widman names, is denoted by changing the final
  letter 'e' of the phane parent hydride name to 'ene' or 'yne', with appropriate multiplying prefixes to
  indicate the multiplicity of each kind of unsaturated site" (:16793); "Low locants are allocated for
  double or triple bonds in accordance with the fixed numbering of the phane parent hydride" (:16795);
  '1,9(1,3)-dibenzenacyclohexadecaphan-2-ene (PIN)' (:16821).
* Phane structures with both double and triple bonds (:16823): "Double and triple bonds in a
  phane structure are described by the method of. Low locants are allocated to double and
  triple bonds first when considered together as a set in ascending order and, if a choice is still needed,
  to double bonds" (:16825); '1,7(1,3)-dibenzenacyclotridecaphan-4-en-2-yne (PIN)' (:16829),
  '1,7(1,3)-dibenzenacyclotridecaphan-2-en-5-yne (PIN)' (:16833).
* (:16489): "In names, the ending 'ene' always precedes 'yne', with elision of the final letter
  'e' in 'ene'"; (:16497): "The multiplying prefixes 'di', 'tri', etc., are placed before endings
  denoting unsaturation".
* Specification of double bonds (:49684): "The omission of stereodescriptors 'E' and 'Z' to denote
  the configuration of double bonds in cyclophanes is related to the total number of nodes in the system;
  for up to seven nodes, no descriptor is necessary. When more than seven nodes are present, all
  descriptors are must be cited" (:49686); '1,4(1,4)-dibenzenacyclohexaphane-2,5-diene (PIN)' (:49692).

The read-back (``validation.phane_read_back``) shares no code with the producer: it parses the name, builds
the molecule it states and compares the FULL InChIKey with the input.
"""
import ast
import re
from pathlib import Path

import pytest
from rdkit import Chem

import orthonym.rules.phane as phane
from tests.support.opsin_marks import release_opsin_only
from orthonym.rules.phane import build_phane_pin
from orthonym.validation import phane_read_back as reader
from orthonym.validation.phane_read_back import phane_name_shape, phane_read_back

# (the Blue Book's printed PIN, a SMILES drawn from the book's structure, the line of the example)
BLUE_BOOK = [
    # [2.2]paracyclophane-1,9-diene: two CH=CH bridges between two para rings
    ("1,4(1,4)-dibenzenacyclohexaphane-2,5-diene", "C1=Cc2ccc(cc2)C=Cc2ccc1cc2", ":49692"),
    # two meta rings 16 nodes apart, one CH=CH at the first bridge atoms (7 + 7 bridge atoms)
    ("1,9(1,3)-dibenzenacyclohexadecaphan-2-ene", "C1=CCCCCCc2cccc(c2)CCCCCCCc2cccc1c2", ":16821"),
    # 5 + 6 bridge atoms: C#C-CH=CH-CH2 and (CH2)6
    ("1,7(1,3)-dibenzenacyclotridecaphan-4-en-2-yne", "C1#CC=CCc2cccc(c2)CCCCCCc2cccc1c2", ":16829"),
    # CH=CH-CH2-C#C and (CH2)6: the double bond takes the lower locant when the set is the same
    ("1,7(1,3)-dibenzenacyclotridecaphan-2-en-5-yne", "C1=CCC#Cc2cccc(c2)CCCCCCc2cccc1c2", ":16833"),
]
# the saturated phanes keep their names
SATURATED = [
    ("1,4(1,4)-dibenzenacyclohexaphane", "C1CC2=CC=C(CCC3=CC=C1C=C3)C=C2"),
    ("1,4(1,3)-dibenzenacyclohexaphane", "C1CC2=CC(=CC=C2)CCC3=CC=CC1=C3"),
    ("1,6(1,4)-dibenzenacycloundecaphane", "C1CCC2=CC=C(CCCCC3=CC=C(CC1)C=C3)C=C2"),
]
# rows of the audit sample whose saturated name was the name of another molecule: (smiles, the name now)
UNSATURATED = [
    ("C1CC2=CC=C(C=C2)C#CC3=CC=C1C=C3", "1,4(1,4)-dibenzenacyclohexaphan-2-yne"),
    ("C1=CC2=CC=C1C#CC3=CC=C(C=C3)C#C2", "1,4(1,4)-dibenzenacyclohexaphane-2,5-diyne"),
    ("C1CC2=CC(=CC=C2)C=CC3=CC=CC1=C3", "1,4(1,3)-dibenzenacyclohexaphan-2-ene"),
    ("C1=CC2=CC(=C1)C#CC3=CC(=CC=C3)C#CC4=CC=CC(=C4)C#C2", "1,4,7(1,3)-tribenzenacyclononaphane-2,5,8-triyne"),
    # RDKit flags the macrocycle of the metacyclophane-diene aromatic: the bridge bonds are no 'double' to it
    ("C1=CC2=CC(=C1)C=CC3=CC=CC(=C3)C=C2", "1,4(1,3)-dibenzenacyclohexaphane-2,5-diene"),
    # an ene-diyne bridge twice: the CH=CH between two C#C is flagged aromatic beside its two triple bonds
    ("C1=CC2=CC(=C1)C#C/C=C/C#CC3=CC=CC(=C3)C#C/C=C/C#C2",
     "1,8(1,3)-dibenzenacyclotetradecaphane-4,11-dien-2,6,9,13-tetrayne"),
]
# up to seven nodes the double bond is the cis one the ring allows: a Z input is named without a descriptor
CIS_SMALL = "C1CC2=CC=C(C=C2)/C=C\\C3=CC=C1C=C3"
TRANS_SMALL = "C1CC2=CC=C(C=C2)/C=C/C3=CC=C1C=C3"
# eight nodes: 'all descriptors must be cited'
EIGHT_Z = "C1CCCc2ccc(cc2)/C=C\\c2ccc1cc2"
EIGHT_E = "C1CCCc2ccc(cc2)/C=C/c2ccc1cc2"
EIGHT_OPEN = "C1CCCc2ccc(cc2)C=Cc2ccc1cc2"
CUMULENE = "C1=CC2=CC=C1C=C=CC3=CC=C(C=C3)C=C=C2"


@pytest.mark.parametrize("name,smiles,line", BLUE_BOOK)
def test_the_producer_gives_the_name_the_blue_book_prints(name, smiles, line):
    assert build_phane_pin(Chem.MolFromSmiles(smiles)) == name, line


@pytest.mark.parametrize("name,smiles,line", BLUE_BOOK)
def test_the_read_back_reads_the_blue_book_name_to_the_drawn_structure(name, smiles, line):
    assert phane_name_shape(name) and phane_read_back(name, smiles), line


@pytest.mark.parametrize("name,smiles", SATURATED)
def test_a_saturated_phane_keeps_its_name(name, smiles):
    assert build_phane_pin(Chem.MolFromSmiles(smiles)) == name
    assert phane_read_back(name, smiles)


@pytest.mark.parametrize("smiles,name", UNSATURATED)
def test_a_bridge_with_a_multiple_bond_names_the_ene_and_yne(smiles, name):
    assert build_phane_pin(Chem.MolFromSmiles(smiles)) == name
    assert phane_read_back(name, smiles)


@pytest.mark.parametrize("smiles,name", UNSATURATED)
def test_the_saturated_name_is_the_name_of_another_molecule(smiles, name):
    """The read-back refuses the name the producer used to give: it states the saturated bridges."""
    saturated = re.sub(r"(phan)(?:e-|-)[\d,a-z-]*(?:ene|yne)$", r"phane", name)
    assert saturated != name and phane_name_shape(saturated)
    assert not phane_read_back(saturated, smiles)


@pytest.mark.parametrize("name,smiles", [
    ("1,4(1,4)-dibenzenacyclooctaphan-3-ene", EIGHT_OPEN),         # the double bond moved by one locant
    ("1,4(1,3)-dibenzenacyclooctaphan-2-ene", EIGHT_OPEN),         # the attachment of the rings changed
    ("1,4(1,4)-dibenzenacyclooctaphan-2-yne", EIGHT_OPEN),         # a triple bond for the double bond
    ("1,5(1,4)-dibenzenacyclooctaphan-2-ene", EIGHT_OPEN),         # the second ring moved
    ("1,4(1,4)-dibenzenacyclononaphan-2-ene", EIGHT_OPEN),         # one more node
    ("1,4(1,4)-dibenzenacyclooctaphane", EIGHT_OPEN),              # the double bond left out
])
def test_a_name_of_another_structure_is_not_read_back(name, smiles):
    """Mutations of a name that reads back (see above): each states another molecule."""
    assert phane_read_back("1,4(1,4)-dibenzenacyclooctaphan-2-ene", smiles)
    assert not phane_read_back(name, smiles)


def test_the_name_does_not_depend_on_the_order_of_the_atoms():
    """Two double bonds at the locants 2 and 6, one of them defined: the numbering that gives the 'Z'
    the lower locant is chosen NUMBERING (j), the Blue Book: "the lower locant is assigned to CIP
    stereodescriptors Z, R, M, and r [...] that are preferred to E, S, P, and s"), whichever atom the SMILES
    starts at."""
    import random
    mol = Chem.MolFromSmiles("c12C/C=C\\c3ccc(CC=Cc(cc2)cc1)cc3")
    names = set()
    rng = random.Random(20261017)
    for _ in range(40):
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        again = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False))
        names.add(build_phane_pin(again))
    assert names == {"(2Z)-1,5(1,4)-dibenzenacyclooctaphane-2,6-diene"}, names


def test_the_lowest_locants_go_to_the_multiple_bonds_as_a_set():
    """ (:16825): the set {2,4} of the first numbering is lower than {3,5} of the second."""
    name = build_phane_pin(Chem.MolFromSmiles("C1#CC=CCc2cccc(c2)CCCCCCc2cccc1c2"))
    assert name.endswith("phan-4-en-2-yne")


def test_up_to_seven_nodes_a_cis_double_bond_has_no_descriptor():
    name = build_phane_pin(Chem.MolFromSmiles(CIS_SMALL))
    assert name == "1,4(1,4)-dibenzenacyclohexaphan-2-ene"
    assert phane_read_back(name, CIS_SMALL)
    assert phane_read_back(name, "C1CC2=CC=C(C=C2)C=CC3=CC=C1C=C3")  # an input that leaves the bond open


def test_up_to_seven_nodes_a_trans_double_bond_has_no_name():
    """The name would state the cis bond the ring allows; the input defines the other one."""
    assert build_phane_pin(Chem.MolFromSmiles(TRANS_SMALL)) is None
    assert not phane_read_back("1,4(1,4)-dibenzenacyclohexaphan-2-ene", TRANS_SMALL)


@pytest.mark.parametrize("smiles,name", [
    (EIGHT_Z, "(2Z)-1,4(1,4)-dibenzenacyclooctaphan-2-ene"),
    (EIGHT_E, "(2E)-1,4(1,4)-dibenzenacyclooctaphan-2-ene"),
    (EIGHT_OPEN, "1,4(1,4)-dibenzenacyclooctaphan-2-ene"),
])
def test_more_than_seven_nodes_cite_the_descriptor_the_input_defines(smiles, name):
    """ (:49686): 'When more than seven nodes are present, all descriptors are must be cited'."""
    assert build_phane_pin(Chem.MolFromSmiles(smiles)) == name
    assert phane_read_back(name, smiles)


def test_a_descriptor_the_input_does_not_define_is_no_read_back():
    assert not phane_read_back("(2Z)-1,4(1,4)-dibenzenacyclooctaphan-2-ene", EIGHT_E)
    assert not phane_read_back("(2E)-1,4(1,4)-dibenzenacyclooctaphan-2-ene", EIGHT_Z)
    assert not phane_read_back("(2Z)-1,4(1,4)-dibenzenacyclooctaphan-2-ene", EIGHT_OPEN)


def test_a_cumulated_bridge_has_no_name():
    """Two double bonds on one bridge atom: the 'ene' endings state two separate double bonds."""
    assert build_phane_pin(Chem.MolFromSmiles(CUMULENE)) is None


@pytest.mark.parametrize("smiles", [
    "C1[13CH2]C2=CC=C(CCC3=CC=C1C=C3)C=C2",     # an isotope the name does not state
    "C1C[CH+]C2=CC=C(CCC3=CC=C1C=C3)C=C2",      # a charge
])
def test_an_atom_the_name_does_not_state_is_declined(smiles):
    assert build_phane_pin(Chem.MolFromSmiles(smiles)) is None


def test_the_ene_ending_follows_the_spelling_of_the_book():
    """The 'e' of 'phane' goes before a vowel only (:16821,:49692); 'ene' loses its 'e' before 'yne'."""
    for ok in ("1,4(1,4)-dibenzenacyclohexaphan-2-ene", "1,4(1,4)-dibenzenacyclohexaphane-2,5-diene",
               "1,7(1,3)-dibenzenacyclotridecaphan-4-en-2-yne", "1,4(1,4)-dibenzenacyclohexaphan-2-yne"):
        assert reader._parse(ok) is not None, ok
    for bad in ("1,4(1,4)-dibenzenacyclohexaphane-2-ene", "1,4(1,4)-dibenzenacyclohexaphan-2,5-diene",
                "1,7(1,3)-dibenzenacyclotridecaphan-4-ene-2-yne", "1,7(1,3)-dibenzenacyclotridecaphan-2-yn-4-ene",
                "1,4(1,4)-dibenzenacyclohexaphan-2,5-ene"):
        assert reader._parse(bad) is None, bad


def test_the_numeral_words_of_the_read_back_agree_with_the_producer():
    """The read-back has its own table of the multiplying affixes; it must say what the producer's
    affix of the node count says for every size a phane can have (4 to 99 nodes)."""
    for n in range(4, 100):
        word = phane._phane_node_multiplier(n)
        assert reader._WORDS.get(word) == n, (n, word)


def test_the_read_back_shares_no_code_with_the_producer():
    tree = ast.parse(Path(reader.__file__).read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(("." * node.level) + (node.module or ""))
        elif isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
    assert not [m for m in imported if "phane" in m or m.startswith("..") or m.startswith("orthonym")], imported


@pytest.mark.parametrize("name", [
    "1,4(1,4)-dibenzenacyclohexaphane", "(2Z)-1,4(1,4)-dibenzenacyclooctaphan-2-ene",
    "1,4,7(1,3)-tribenzenacyclononaphane-2,5,8-triyne", "1,7(1,3)-dibenzenacyclotridecaphan-4-en-2-yne",
])
def test_the_shape_the_gate_reads_is_the_shape_of_a_phane_name(name):
    assert phane_name_shape(name)


@pytest.mark.parametrize("name", ["ethanol", "cyclohexane", "1,4-dimethylbenzene", "propane-1,2,3-triyl triacetate", None])
def test_no_other_name_has_the_shape(name):
    assert not phane_name_shape(name)


# ---------------------------------------------------------------------------------------------------------
# the validity gate: a phane name ships only when the structure it states is the input
# ---------------------------------------------------------------------------------------------------------

def test_the_carve_out_ships_a_name_only_when_the_read_back_agrees():
    from orthonym.namer import _grammar_gap_carveout
    saturated = "C1CC2=CC=C(CCC3=CC=C1C=C3)C=C2"
    alkyne = "C1CC2=CC=C(C=C2)C#CC3=CC=C1C=C3"
    assert _grammar_gap_carveout("1,4(1,4)-dibenzenacyclohexaphane", saturated) == "phane"
    assert _grammar_gap_carveout("1,4(1,4)-dibenzenacyclohexaphane", alkyne) is None
    assert _grammar_gap_carveout("1,4(1,4)-dibenzenacyclohexaphan-2-yne", alkyne) == "phane"
    assert _grammar_gap_carveout("1,4(1,4)-dibenzenacyclohexaphane", None) is None


@pytest.mark.opsin_gate
@release_opsin_only
@pytest.mark.parametrize("smiles,name", [
    ("C1CC2=CC=C(C=C2)C#CC3=CC=C1C=C3", "1,4(1,4)-dibenzenacyclohexaphan-2-yne"),
    ("C1=Cc2ccc(cc2)C=Cc2ccc1cc2", "1,4(1,4)-dibenzenacyclohexaphane-2,5-diene"),
    (EIGHT_Z, "(2Z)-1,4(1,4)-dibenzenacyclooctaphan-2-ene"),
    ("C1CC2=CC=C(CCC3=CC=C1C=C3)C=C2", "1,4(1,4)-dibenzenacyclohexaphane"),
])
def test_the_default_tier_gives_the_name_that_reads_back(smiles, name):
    from tests.support.default_tier import default_tier_row
    row = default_tier_row(smiles)
    assert row["name"] == name and row["gate_outcome"] == "carveout:phane", row


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [TRANS_SMALL, pytest.param(CUMULENE, marks=release_opsin_only)])
def test_the_default_tier_declines_a_phane_it_cannot_name(smiles):
    """The producer names neither (a trans bond up to seven nodes, a cumulated bridge): the strict path has
    no name, and no phane name of another molecule stands in."""
    from orthonym.errors import is_failure_name
    from tests.support.default_tier import default_tier_row
    row = default_tier_row(smiles)
    assert row["tier"] == "abstain" and is_failure_name(row["name"]) and not phane_name_shape(row["name"]), row


@release_opsin_only
@pytest.mark.opsin_gate
def test_a_name_of_another_molecule_is_declined_at_the_default_tier(monkeypatch):
    """The producer that names the alkyne as the saturated phane (the old defect) is not shipped."""
    from tests.support.default_tier import default_tier_row
    monkeypatch.setattr(phane, "build_phane_pin", lambda mol: "1,4(1,4)-dibenzenacyclohexaphane")
    row = default_tier_row("C1CC2=CC=C(C=C2)C#CC3=CC=C1C=C3")
    assert row["name"] != "1,4(1,4)-dibenzenacyclohexaphane" and row["tier"] == "abstain", row


# ---------------------------------------------------------------------------------------------------------
# best-effort: a phane name the default tier gives is the last resort when the general engine cannot name
# the input; it never replaces a name OPSIN reads (namer.py, the natural-product / phane carve-out docstring:
# "A phane name never replaces a name OPSIN reads back: the general engine's von Baeyer name of a phane
# stands."). UN lane U1: these four rows were last-resort rows only because the general engine's von Baeyer
# name spelled its ene/yne count as digits (more than 20: 'tetracosaene'), which OPSIN cannot read; with the
# composed multiplier the general engine names them, so best-effort now gives that name (a full read-back by a
# fresh OPSIN call) and the default tier keeps the phane name. UNVERIFIED which of the two names is preferred
# (the Blue Book is silent on the best-effort tier); what is confirmed is that each name describes the input
# (phane_read_back for the phane name, OPSIN for the other) and that the pre-fix code gave the phane name at
# both tiers.
# ---------------------------------------------------------------------------------------------------------
# the four rows of the CO audit sample the default tier named and best-effort left unnamed (PubChem CIDs)
LAST_RESORT_ROWS = [
    ("1,3,5,7,9,11,13,15(1,3)-octabenzenacyclohexadecaphane",                                       # CID 15871804
     "C1C2=CC(=CC=C2)CC3=CC(=CC=C3)CC4=CC(=CC=C4)CC5=CC(=CC=C5)CC6=CC=CC(=C6)CC7=CC=CC(=C7)CC8=CC=CC(=C8)CC9=CC=CC1=C9"),
    ("1,4,7,10,13,16,19,22,25(1,3)-nonabenzenacycloheptacosaphane",                                  # CID 57357798
     "C1CC2=CC(=CC=C2)CCC3=CC(=CC=C3)CCC4=CC(=CC=C4)CCC5=CC(=CC=C5)CCC6=CC=CC(=C6)CCC7=CC=CC(=C7)CCC8=CC=CC(=C8)CCC9=CC=CC(=C9)CCC2=CC=CC1=C2"),
    ("1,4,7,10,13,16,19(1,4)-heptabenzenacyclohenicosaphane-2,5,8,11,14,17,20-heptayne",             # CID 15505942
     "C1=CC2=CC=C1C#CC3=CC=C(C=C3)C#CC4=CC=C(C=C4)C#CC5=CC=C(C=C5)C#CC6=CC=C(C=C6)C#CC7=CC=C(C=C7)C#CC8=CC=C(C=C8)C#C2"),
    ("(2Z,5Z,8E,11Z,14Z,17E)-1,4,7,10,13,16(1,4)-hexabenzenacyclooctadecaphane-2,5,8,11,14,17-hexaene",   # CID 11262018
     "C1=CC/2=CC=C1/C=C/C3=CC=C(/C=C\\C4=CC=C(/C=C\\C5=CC=C(/C=C/C6=CC=C(/C=C\\C7=CC=C(/C=C2)C=C7)C=C6)C=C5)C=C4)C=C3"),
]


@pytest.mark.opsin_gate
@release_opsin_only
@pytest.mark.parametrize("name,smiles", LAST_RESORT_ROWS)
def test_default_tier_names_the_phane_rows_with_the_phane_name(name, smiles):
    from tests.support.default_tier import default_tier_row
    assert phane_read_back(name, smiles)
    row = default_tier_row(smiles)
    assert row["name"] == name and row["gate_outcome"] == "carveout:phane", row


@pytest.mark.opsin_gate
@release_opsin_only
@pytest.mark.parametrize("name,smiles", LAST_RESORT_ROWS)
def test_best_effort_gives_the_phane_rows_the_general_engine_name_opsin_reads(name, smiles):
    from tests.support.rt_assert import assert_full_rt, name_best_effort
    row = name_best_effort(smiles)
    assert row["name"] != name and row["source"] == "general_engine", row
    assert row["gate_outcome"] != "carveout:phane", row
    assert_full_rt(row["name"], smiles, what="best-effort: ")


@pytest.mark.opsin_gate
@release_opsin_only
@pytest.mark.parametrize("tier", ["valid", "complete"])
def test_the_valid_and_complete_tiers_still_read_back_every_name_with_opsin(tier):
    """The phane builder is an exception of best-effort only (README: 'every name must pass a full-InChIKey OPSIN
    round trip' at valid and complete)."""
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    row = Orthonym(**_emit_tier_flags(tier)).name_tiered(LAST_RESORT_ROWS[0][1])
    assert row["tier"] == "abstain", row


@release_opsin_only
@pytest.mark.opsin_gate
def test_a_phane_name_never_replaces_a_name_opsin_reads_back_at_best_effort():
    """A one-line pass of the gate would have given every phane its phane name at best-effort and replaced 59 of
    the 65 names OPSIN reads (von Baeyer, systematic_verified) by names no OPSIN reads: the last resort turns an
    abstention into a name and changes no other."""
    from tests.support.rt_assert import assert_full_rt, name_best_effort
    smiles = "C1CC2=CC=C(CCC3=CC=C1C=C3)C=C2"
    row = name_best_effort(smiles)
    assert row["name"].startswith("tricyclo[") and row["tier"] == "systematic_verified", row
    assert row["verified"] == "opsin", row
    assert_full_rt(row["name"], smiles)
