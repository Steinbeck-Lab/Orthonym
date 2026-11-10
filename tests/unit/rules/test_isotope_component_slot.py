"""A nuclide on a later component of a multi-part name that opens with a locant (program 22, 2026-11-10).

 (the Blue Book): the name of an isotopically substituted compound is formed by "inserting the nuclide
symbol(s)... before the part of the compound that is isotopically substituted", and when that part "includes a
preceding locant, a hyphen is inserted" ('(15N)-1H-indole (PIN)',:43790). The descriptor had no slot in front of the
'1H-' of 'ethanol—1H-imidazole (1/1)', so the labelled adduct got no name although '(2-2H)-1H-imidazole' alone was
named (``rules.isotope_slots.COMPONENT_LOCANT_START_RE``).
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module")
def best_effort():
    return Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)


@pytest.mark.parametrize("smiles, name", [
    ("CC(=O)O.[2H]C1=NC=CN1", "acetic acid—(2-2H)-1H-imidazole (1/1)"),
    ("C(C)OC(C(C(CC)=O)=[N+]=[N-])=O.N1N=CC=2C1=NC(=NC2C#N)[2H]",
     "ethyl 2-diazo-3-oxopentanoate—(6-2H)-1H-pyrazolo[3,4-d]pyrimidine-4-carbonitrile (1/1)"),
])
def test_label_on_a_later_component_that_opens_with_a_locant(best_effort, smiles, name):
    assert best_effort.name_tiered(smiles)["name"] == name


def test_slot_is_the_digit_after_the_component_separator():
    from orthonym.rules.isotopes import _insertion_offsets
    skeleton = "ethanol—1H-imidazole (1/1)"
    assert skeleton.index("1H") in _insertion_offsets(skeleton)
    assert skeleton.index("(1/1)") + 1 not in _insertion_offsets(skeleton)   # the ratio opens with '(', not a digit


def test_an_unlabelled_component_keeps_its_unmodified_name(best_effort):
    """ (the Blue Book) requires every locant of the compound whose label needs one; the other
    component of the adduct keeps its own name ('ethanol', not 'ethan-1-ol')."""
    row = best_effort.name_tiered("CCO.[2H]c1nc[nH]c1")
    assert row["name"].startswith("ethanol—") and "(4-2H)" in row["name"], row
