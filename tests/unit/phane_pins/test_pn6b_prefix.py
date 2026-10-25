import random

import pytest
from rdkit import Chem

from orthonym.rules.phane_pin import prefix
from tests.support.opsin_marks import release_opsin_only

from ._helpers import default_tier_row, mol, name_best_effort, needs_phane_opsin, substituted

SPELLINGS = 30
PHANE = "2,4,6-trioxa-1,7(1),3,5(1,4)-tetrabenzenaheptaphan-1^4-yl"
#: the chain of four benzene rings and three oxygen atoms (:23959) on the aryl group of a parent of another class
CHAIN = "c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1"
SILANE = "C[Si](C)(C)" + CHAIN
#: the rows of the reader's 'phane prefix' tests: the parent of another class cites more of the principal group
#: or has a senior atom, and the PIN cites the phane in a prefix
ROWS = [
    ("OC(=O)C" + CHAIN, f"[{PHANE}]acetic acid"),
    ("COC(=O)N" + CHAIN, f"methyl [{PHANE}]carbamate"),
    ("NC(=O)N" + CHAIN, f"[{PHANE}]urea"),
    ("CC(=O)" + CHAIN, f"1-[{PHANE}]ethan-1-one"),
    ("CC(=S)" + CHAIN, f"1-[{PHANE}]ethane-1-thione"),
    ("CC(=[Se])" + CHAIN, f"1-[{PHANE}]ethane-1-selone"),
    ("Nc1nc(Nc2ccc(Oc3ccc(Oc4ccc(Oc5ccccc5)cc4)cc3)cc2)ccn1", f"N4-[{PHANE}]pyrimidine-2,4-diamine"),
    ("OC(=O)CC(C(=O)O)c1ccc(Oc2ccc(Oc3ccc(Oc4ccc(C(=O)O)cc4)cc3)cc2)cc1",
     f"2-[7^4-carboxy-{PHANE}]butanedioic acid"),
    (SILANE, f"trimethyl[{PHANE}]silane"),               #:19325 shows this prefix of a silane
]
IDS = ["acetic_acid", "carbamate", "urea", "ethanone", "ethanethione", "ethaneselone", "pyrimidine_diamine",
       "butanedioic_acid", "silane"]


def _respelled(smiles, seed):
    m = mol(smiles)
    order = list(range(m.GetNumAtoms()))
    random.Random(seed).shuffle(order)
    return Chem.MolToSmiles(Chem.RenumberAtoms(m, order), canonical=False)


@needs_phane_opsin
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", ROWS, ids=IDS)
def test_the_parent_of_another_class_cites_the_phane_prefix(smiles, name):
    # (:16148), (:18875), (:18917): the parent stays the parent, the phane is its prefix
    for row in (default_tier_row(smiles), name_best_effort(smiles)):
        assert row["name"] == name and row["tier"] == "pin_verified" and row["spelling_failures"] == [], row


@needs_phane_opsin
@pytest.mark.opsin_gate
@pytest.mark.parametrize("line,name,smiles", substituted("phane_as_prefix"), ids=lambda v: str(v)[:30])
def test_the_book_row_of_a_phane_prefix(line, name, smiles):
    # (:16148), (:18917): the silane is the parent, the phane its prefix; the pyran amplificant is
    # numbered by its parent name first,:14967), so the indicated hydrogen is '1^2H' and the attachment '6'
    for row in (default_tier_row(smiles), name_best_effort(smiles)):
        assert row["name"] == name and row["tier"] == "pin_verified" and row["spelling_failures"] == [], row


@needs_phane_opsin
@pytest.mark.opsin_gate
def test_a_phane_prefix_with_a_stereogenic_atom_cites_the_descriptor_in_the_prefix():
    # (:44639): the descriptor of a substituent group is cited at the front of its prefix; the whole name
    # reads back to the full InChIKey of the structure
    for smiles, code in (("C[Si](C)(C)c1ccc([C@H](Cl)c2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1", "R"),
                         ("C[Si](C)(C)c1ccc([C@@H](Cl)c2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1", "S")):
        row = default_tier_row(smiles)
        assert row["name"] == (f"[({6}{code})-6-chloro-2,4-dioxa-1,7(1),3,5(1,4)-tetrabenzenaheptaphan-7^4-yl]"
                               "tri(methyl)silane"), row
        assert row["tier"] == "pin_verified", row


@needs_phane_opsin
def test_the_prefix_of_the_book_group_is_built_from_the_fragment():
    m = mol(SILANE)
    atoms = range(4, m.GetNumAtoms())
    assert prefix.phane_prefix(m, atoms, 4) == PHANE
    assert prefix.phane_prefix(m, atoms, 1) == PHANE        # the parent-side atom, as some namers pass it


@needs_phane_opsin
def test_the_prefix_of_one_fragment_is_built_once_for_a_molecule(monkeypatch):
    # the substituent namers ask for one fragment of one molecule many times (108 calls for 4 fragments on a porphyrin of
    # the sample); each build perceives every chain of the fragment, so the answer is kept per molecule, fragment and atom
    m = mol(SILANE)
    atoms = list(range(4, m.GetNumAtoms()))
    built = []
    real = prefix._name
    monkeypatch.setattr(prefix, "_name", lambda *args: built.append(1) or real(*args))
    assert prefix.phane_prefix(m, atoms, 4) == PHANE
    assert prefix.phane_prefix(m, atoms, 4) == PHANE and len(built) == 1
    assert prefix.phane_prefix(m, set(atoms), 4) == PHANE and len(built) == 1       # the same fragment, as a set
    assert prefix.phane_prefix(Chem.Mol(m), atoms, 4) == PHANE and len(built) == 2   # another molecule is not reused
    assert prefix.phane_prefix(m, atoms[:-6], 4) is None and len(built) == 2        # another fragment: no name, not built


@needs_phane_opsin
def test_a_fragment_without_the_whole_chain_has_no_phane_prefix():
    m = mol(SILANE)
    assert prefix.phane_prefix(m, range(4, m.GetNumAtoms() - 6), 4) is None     # a ring and an oxygen atom short


@needs_phane_opsin
def test_a_phane_that_is_the_parent_is_not_a_prefix():
    # the acid on the end ring makes the phane the parent: no group of it is a prefix
    m = mol("OC(=O)" + CHAIN)
    assert prefix.phane_prefix(m, range(3, m.GetNumAtoms()), 3) is None


@release_opsin_only
def test_there_is_no_prefix_with_an_opsin_that_reads_no_phane_name():
    m = mol(SILANE)
    assert prefix.phane_prefix(m, range(4, m.GetNumAtoms()), 4) is None


@needs_phane_opsin
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [SILANE, ROWS[7][0]], ids=["silane", "butanedioic_acid"])
def test_the_prefix_does_not_depend_on_the_order_of_the_atoms(smiles):
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    from orthonym.jvm_budget import jvm_slots
    with jvm_slots(1, purpose="pn6b"):
        namer = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
        names = {namer.name_tiered(_respelled(smiles, seed))["name"] for seed in range(SPELLINGS)}
    assert len(names) == 1 and PHANE in next(iter(names)), names


TWO_ACIDS = "OC(=O)CCc1ccc(Cc2ccc(Cc3ccc(Cc4ccc(CCC(O)=O)cc4)cc3)cc2)cc1"
TWO_AMINES = "NCCc1ccc(Cc2ccc(Cc3ccc(Cc4ccc(CCN)cc4)cc3)cc2)cc1"


@needs_phane_opsin
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [TWO_ACIDS, TWO_AMINES], ids=["acids", "amines"])
def test_two_identical_parents_get_no_phane_prefix(smiles):
    # (:23178): the PIN is the multiplicative name with the phane as its central group ('3,3'-[...phane-1^4,7^4-diyl]
    # dipropanoic acid'), which is not built here; a name with the phane in the prefix of one parent is no PIN, so no prefix is
    # built and the default tier declines (the best-effort tier names the structure without a phane word)
    assert prefix._whole_structure_cites_a_prefix(mol(smiles)) is False
    assert prefix._whole_structure_cites_a_prefix(mol("OC(=O)CCc1ccc(Cc2ccc(Cc3ccc(Cc4ccccc4)cc3)cc2)cc1")) is True
    row = default_tier_row(smiles)
    assert row["tier"] == "abstain" and [f["rule"] for f in row["spelling_failures"]] == ["P-52.2.5.1"], row
    assert "phan" not in name_best_effort(smiles)["name"]
