"""Isotope locants on the nitrogens of a hydrazide or a hydrazine parent (2026-11-09, found in the hydrazine review).

A hydrazide cites its nitrogens N and N', 'N'-methylacetohydrazide'), not the numbered N1, N2 of the di- and
polyamides, the Blue Book); OPSIN reads both, so the read-back could not choose and '(N1-2H)acetohydrazide'
shipped as a PIN. The parent hydride hydrazine "is numbered by numerical locants, 1 and 2, not N and N'",
:38507). The isotopic descriptor sits before the name with its locant,:43824, '(N-2H1)acetamide (PIN)').
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name", [
    ("CC(=O)N([2H])N", "(N-2H)acetohydrazide"),
    ("CC(=O)NN([2H])C", "N'-methyl(N'-2H)acetohydrazide"),
    ("CNN([2H])C(=O)O", "2-methyl(1-2H)hydrazine-1-carboxylic acid"),
    ("CN([2H])NC(=O)O", "2-methyl(2-2H)hydrazine-1-carboxylic acid"),
    # unchanged: the amide and the diamine
    ("CC(=O)N[2H]", "(N-2H1)acetamide"),
    ("NCCN[2H]", "(N1-2H1)ethane-1,2-diamine"),
])
def test_hydrazine_nitrogens_take_the_parents_locants(smiles, name):
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")
