"""The input's stereo composed with each descriptor at the front of the part it relates to (rules/prefix_stereo.py).

 NAMING OF STEREOISOMERS (the Blue Book,:44643): "When they relate to substituent groups, they are cited
at the front of the corresponding prefix. They are preceded by a numerical or letter locant to describe the position of
the stereogenic unit when such locants are present".
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.rules.prefix_stereo import _places, _units, _with_block


def test_places_innermost_part_first_then_the_front():
    name = "4-[(2-aminophenyl)methyl]pyridazine"
    assert [name[p:p + 4] for p in _places(name)] == ["2-am", "(2-a", "4-[("]


@pytest.fixture(scope="module")
def best_effort():
    return Orthonym(style="pin", **_emit_tier_flags("best-effort"))


@pytest.mark.opsin_gate
def test_a_substituent_double_bond_is_cited_at_the_front_of_its_prefix(best_effort):
    # CID147930797: every descriptor at the front of the name ('(1Z)-...') is read on the parent and does not read back
    row = best_effort.name_tiered("C1=CC(=CN=C1)C(=N/C(=C(/C2=CC=C(C=C2)Br)\\N)/C3=CC=C(C=C3)Br)[NH-]")
    assert row["name"] == "({[(1Z)-2-amino-1,2-bis(4-bromophenyl)ethen-1-yl]imino}(pyridin-3-yl)methyl)azanide"
    assert row["tier"] == "systematic_verified"


@pytest.mark.opsin_gate
def test_a_bare_descriptor_beside_locanted_ones_is_not_offered(best_effort):
    # CID10929611: '(R,S,1R,24S)-...' reads back only by the order in which OPSIN gives the bare descriptors to the
    # centres, so the molecule keeps its abstention
    row = best_effort.name_tiered(
        "C1C[C@H]2C3=NC4=CC5=C(C=C4N=C3[C@@H]1C6=NC(=C(N=C26)C#N)C#N)N=C7[C@@H]8CC[C@H](C7=N5)"
        "C9=NC(=C(N=C89)C#N)C#N")
    assert row["tier"] == "abstain"


def test_a_block_inside_a_part_raises_the_marks_around_it():
    # (BB 7446) nesting order; a descriptor block nests, BB 7478)
    name = "4-(pent-2-en-1-yl)pyridine"
    assert _with_block(name, 3, "2E") == "4-[(2E)-pent-2-en-1-yl]pyridine"


def test_a_metal_centre_that_is_not_tetrahedral_is_not_placed():
    from rdkit import Chem
    from orthonym.perception.stereo import assign_stereochemistry
    mol = Chem.MolFromSmiles("C[C@H](N)C(=O)O.Cl[Pt@SP1](Cl)(N)N")
    assign_stereochemistry(mol)
    assert _units(mol) is None
