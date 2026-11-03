"""A halogen -uide name in a salt or with an isotope descriptor reads back through its twin at every tier.

The method (1) name 'diphenyliodanuide' has no OPSIN grammar; its method (2) twin 'diphenyl-λ3-iodanide' does
(``validation.uide_twin``). The twin now keeps the cation words of a salt ('sodium diphenyliodanuide') and the isotope
placement search and the exit check of a wider tier read an isotope-labelled -uide name through it.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.validation.uide_twin import uide_twin


@pytest.mark.parametrize("name, twin", [
    ("diphenyliodanuide", "diphenyl-λ3-iodanide"),
    ("sodium diphenyliodanuide", "sodium diphenyl-λ3-iodanide"),
    ("(13C)methyl(phenyl)iodanuide", "(13C)methyl(phenyl)-λ3-iodanide"),
], ids=lambda v: v)
def test_the_twin(name, twin):
    assert uide_twin(name) == twin


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, name", [
    ("[Na+].c1ccc(cc1)[I-]c1ccccc1", "sodium diphenyliodanuide"),
    ("c1ccc(cc1)[I-][13CH3]", "(13C)methyl(phenyl)iodanuide"),
    ("[Na+].c1ccc(cc1)[I-][13CH3]", "sodium (13C)methyl(phenyl)iodanuide"),
], ids=lambda v: v)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_embedded_uide_names(smiles, name, tier):
    row = Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)
    assert row["name"] == name, row
