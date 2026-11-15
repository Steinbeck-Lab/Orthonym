"""a phase BBR-.6-caveats — retained-name-first charged routing .

route_charged (the 169.6 chokepoint) neutralized-then-renamed sanctioned retained
charged species, producing OPSIN-unparseable systematic forms the gate then
suppressed to 'unknown' (or a wrong retained form). 169.7 consults the charged
retained-name lookup FIRST //; there is no rule forcing systematic
re-derivation over a retained name). Recovers the documented 169.6 regressions.
All target names are OPSIN-RT-verified.
"""
import pytest


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("[SH3+]", "sulfanium"),        # PIN (was 'sulfonium'; earlier -> unknown)
    ("N[O-]", "aminoxide"),                                             # was -> unknown
    ("O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F", "bistriflimide"),        # was -> triflimidic acid
])
def test_charged_retained_recovered(namer, smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)[O-]", "acetate"),                  # carboxylate (deferred path) unchanged
    ("C[S+](C)C", "trimethylsulfanium"),        # PIN (was 'trimethylsulfonium')
    ("[NH4+]", "azanium"),                       # PIN (was 'ammonium')
])
def test_existing_charged_unchanged(namer, smiles, expected):
    assert namer.name(smiles) == expected


@pytest.mark.unit
def test_retained_recovered_round_trip():
    from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
    for smi, name in [("[SH3+]", "sulfanium"), ("N[O-]", "aminoxide")]:
        assert opsin_roundtrip_check(smi, name)["passed"]


@pytest.mark.opsin_gate
def test_bistriflimide_is_no_pin():
    """'bistriflimide' is not in the recommendations (0 hits in the Blue Book); the systematic name of the anion is
    bis(trifluoromethanesulfonyl)azanide: the name ships, below the PIN."""
    from tests.support.rt_assert import name_best_effort
    for smiles in ("O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F", "[Li+].O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F"):
        r = name_best_effort(smiles)
        assert "bistriflimide" in r["name"] and not r["is_pin"] and r["tier"] == "systematic_verified", r
