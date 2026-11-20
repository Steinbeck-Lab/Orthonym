"""A carboxamido prefix built on a ring parent hydride is enclosed in marks.

 (a prefix that includes the name of a parent hydride is enclosed, '4-(cyclohexanecarbonyl)benzene-1-
carbothioic acid (PIN)', the Blue Book) with "Substituents of the types -NH-CO-R and -NH-SO2-R"
(:32991, method (1) "generates preferred IUPAC names": '3-(methanesulfonamido)propanoic acid (PIN)'). The shared rule
``naming_utils.needs_brackets`` knew 'carbonyl' on a ring stem but not 'carboxamido', so
'4-cyclohexanecarboxamidobutanoic acid' shipped below the PIN.
"""
import pytest

from orthonym.assembly.naming_utils import needs_brackets
from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt


@pytest.mark.parametrize("prefix,marked", [
    ("cyclohexanecarboxamido", True), ("cyclopropanecarboxamido", True), ("pyridine-3-carboxamido", True),
    ("benzamido", False), ("acetamido", False),
])
def test_needs_brackets(prefix, marked):
    assert needs_brackets(prefix) is marked


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("OC(=O)CCCNC(=O)C1CCCCC1", "4-(cyclohexanecarboxamido)butanoic acid"),
    ("OC(=O)c1ccc(NC(=O)C2CCCCC2)cc1", "4-(cyclohexanecarboxamido)benzoic acid"),
])
def test_ring_carboxamido_acid_pins(smiles, name):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)
