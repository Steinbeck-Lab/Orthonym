"""The chalcogen cyanic acids and their anions, and the arsanium and tellanium cations.

 "Cyanic acid" (the Blue Book): "Cyanic acid is the retained name for NC-OH"; 'thiocyanic acid (PIN)'
(:30987); the prefixes 'thiocyanato', 'selenocyanato', 'tellurocyanato' (:30981); an '-ic acid' gives the '-ate' anion
.: a substituted mononuclear parent hydride cation is the hydride name + 'ium', as
'chlorotri(methyl)phosphanium (PIN)'.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name", [
    ("N#CS", "thiocyanic acid"),
    ("N#C[SeH]", "selenocyanic acid"),
    ("N#C[TeH]", "tellurocyanic acid"),
    ("N#C[O-]", "cyanate"),
    ("N#C[S-]", "thiocyanate"),
    ("N#C[Se-]", "selenocyanate"),
    ("N#C[Te-]", "tellurocyanate"),
    ("[Na+].N#C[S-]", "sodium thiocyanate"),
    ("[K+].N#C[Se-]", "potassium selenocyanate"),
    ("c1ccc([As+](c2ccccc2)(c2ccccc2)c2ccccc2)cc1", "tetraphenylarsanium"),
    ("[Cl-].c1ccc([As+](c2ccccc2)(c2ccccc2)c2ccccc2)cc1", "tetraphenylarsanium chloride"),
    ("C[As+](C)(C)Cl", "chlorotri(methyl)arsanium"),
    ("C[Te+](C)C", "trimethyltellanium"),
], ids=lambda v: v)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_names(smiles, name, tier):
    row = Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified", row
