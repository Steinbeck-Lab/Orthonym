"""
Tests for stereo-free steroid SMILES -- expected behavior documentation.

These compounds have no stereo annotations in their input SMILES.
Orthonym correctly omits stereodescriptors (cannot invent stereo).
RT mismatch with OPSIN is expected: OPSIN adds default steroid stereo
when parsing retained names like "stigmast-5-en-3,7-diol".

: Closed as expected behavior, not a bug.
"""
import pytest
from orthonym import name_compound


STIGMASTADIENEDIOL = "CCC(CCC(C)C1CCC2C3C(O)C=C4CC(O)CCC4(C)C3CCC12C)C(C)C"
ERGOSTADIENOL = "C=C(C)C(C)CCC(C)C1CCC2C3=CCC4CC(O)CCC4(C)C3CCC21C"

# The two raw checks below passed on names that dropped atoms: '2-methyl-3-[5-
# (propan-2-yl)heptan-2-yl]-1-tridecylcyclopentanediol' (OPSIN cannot parse it)
# and '5-henicosyl-4-methylcyclohexan-1-ol' (OPSIN: a different molecule), one
# ring of the steroid named as a monocycle. bdd69a673 stopped that,
# the Blue Book). A stereoparent name (stigmastane, ergostane) implies the
# absolute configuration,:51047), so it cannot name a stereo-free
# input either; the PIN is a hydro-cyclopenta[a]phenanthrene name the PIN tier
# does not build. Production (gate on) abstained at the PIN tier at 4e0e5c29b
# too. Strict: 'ol' may only come back through a real, complete name.
_STEROID_XFAIL = pytest.mark.xfail(strict=True, reason=(
    "PIN tier abstains: needs the hydro-cyclopenta[a]phenanthrene PIN for a "
    "stereo-free steroid (P-101.2.6); the old names dropped atoms (bdd69a673) "
    "-- TODO in TRIAGE.md 'Suite fix -- j1-regressions'"))


@pytest.mark.integration
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [STIGMASTADIENEDIOL, ERGOSTADIENOL])
def test_stereo_free_steroid_tier_contract(smiles):
    """What ships (gate on): the PIN tier fails closed or ships an RT-exact name;
    best-effort names it RT-exact, invents no stereodescriptor, and keeps the
    hydroxy groups as the '-ol' suffix."""
    from tests.support.rt_assert import assert_tier_contract
    _pin, be = assert_tier_contract(smiles)
    assert "(R)" not in be and "(S)" not in be and "R," not in be and "S," not in be, be
    assert be.endswith("ol"), be


@pytest.mark.integration
class TestStereoFreeSteroids:
    @_STEROID_XFAIL
    def test_stigmastadienediol_no_stereo_in_input(self):
        """Stereo-free stigmasterol derivative: no @ in SMILES -> no stereo in name."""
        smiles = "CCC(CCC(C)C1CCC2C3C(O)C=C4CC(O)CCC4(C)C3CCC12C)C(C)C"
        name = name_compound(smiles)
        assert name is not None
        # Name should NOT contain stereodescriptors (no stereo in input)
        assert "(R)" not in name and "(S)" not in name, f"Unexpected stereo in: {name}"
        # Should contain steroid-related naming
        assert "stigmast" in name.lower() or "ol" in name.lower()

    @_STEROID_XFAIL
    def test_ergostadienol_no_stereo_in_input(self):
        """Stereo-free ergosterol derivative: no @ in SMILES -> no stereo in name."""
        smiles = "C=C(C)C(C)CCC(C)C1CCC2C3=CCC4CC(O)CCC4(C)C3CCC21C"
        name = name_compound(smiles)
        assert name is not None
        assert "(R)" not in name and "(S)" not in name, f"Unexpected stereo in: {name}"
        assert "ergost" in name.lower() or "ol" in name.lower()
