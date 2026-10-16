"""tests/support/rt_assert.py is the independent oracle behind ~15 test files: its
round-trip check must not be radical-blind and must not reuse the engine's own
OPSIN cache (fix a performance pass, wp1-zero-wrong, F6).

The full standard InChIKey encodes neither radical electrons nor bond order, so a
key-only comparison accepts a radical reading of a closed-shell molecule and the
reverse (OPSIN 2.9.0 -r: 'ethane-1,2-diyl' -> [CH2][CH2], same key as C=C;
'but-3-en-2-yl' -> C=C[CH]C, same key as [CH2]C=CC; 'tristannan-2-yl' ->
[SnH3][SnH][SnH3], same key as [SnH2][SnH2][SnH3]).
"""
import pytest

from orthonym.assembly.memo import pop_scope, push_scope
from tests.support import rt_assert


@pytest.mark.parametrize("name,smiles", [
    ("ethane-1,2-diyl", "C=C"),
    ("ethene", "[CH2][CH2]"),
    ("but-3-en-2-yl", "[CH2]C=CC"),
    ("tristannan-2-yl", "[SnH2][SnH2][SnH3]"),
])
def test_radical_key_twins_are_not_rt_exact(name, smiles):
    assert not rt_assert.name_is_rt_exact(name, smiles)


@pytest.mark.parametrize("name,smiles", [
    ("ethanol", "CCO"),
    ("ethane-1,2-diyl", "[CH2][CH2]"),
    ("but-2-en-1-yl", "[CH2]C=CC"),
    ("tristannan-1-yl", "[SnH2][SnH2][SnH3]"),
    ("tin(II) dichloride", "[Sn+2].[Cl-].[Cl-]"),
])
def test_right_names_stay_rt_exact(name, smiles):
    assert rt_assert.name_is_rt_exact(name, smiles)


def test_parse_does_not_use_the_engine_oracle(monkeypatch):
    # A poisoned engine oracle / memo must not change the helper's answer.
    import orthonym.namer as nm
    monkeypatch.setattr(nm, "_validity_gate_name_to_smiles", lambda *a, **k: "CCO")
    assert not rt_assert.name_is_rt_exact("propan-1-ol", "CCO")
    assert rt_assert.name_is_rt_exact("propan-1-ol", "CCCO")
    scope = push_scope()
    try:
        assert rt_assert._independent_parse("ethanol") in ("CCO", "C(C)O")
    finally:
        pop_scope(scope)


# ---- SL lane M2: ``-r`` only for an input that has radical electrons ----------------------------
#
# "GENERAL METHODOLOGY" (the Blue Book): "Radicals are named by modifying a parent
# hydride name to signal the subtraction or addition of one or more hydrogen atoms"; "The suffixes
# 'yl' (-H.), 'ylidene' (-2H.), 'ylidyne' (-3H.) denote the removal of hydrogen atoms". A name with a
# free valence is a name of a radical or a substituent group. OPSIN's ``-r`` reads such a name and
# writes the structure without marking the radical: 'methoxyphosphoryl' (CH3O-PO<, two unfilled
# valences) reads back as COP=O, the closed-shell structure of its atoms, so the helper used to
# accept it as the name of COP=O. ``orthonym.validation.read_back_policy`` is the engine's rule
# (SF lane E); the helper reads as the engine does.

def test_a_closed_shell_input_is_read_back_without_radicals():
    # no input structure: the legacy reading, the name read for its own sake
    assert rt_assert._independent_parse("methoxyphosphoryl") == "COP=O"
    # the closed-shell input COP=O: OPSIN reads nothing
    assert rt_assert._independent_parse("methoxyphosphoryl", "COP=O") is None
    assert not rt_assert.name_is_rt_exact("methoxyphosphoryl", "COP=O")
    with pytest.raises(AssertionError):
        rt_assert.assert_full_rt("methoxyphosphoryl", "COP=O")


def test_a_radical_input_is_still_read_back_with_radicals():
    assert rt_assert._independent_parse("ethyl", "C[CH2]") == "[CH2]C"
    assert rt_assert._independent_parse("ethyl", "CC") is None          # the closed-shell ethane
    assert rt_assert.name_is_rt_exact("ethane-1,2-diyl", "[CH2][CH2]")
    assert rt_assert.name_is_rt_exact("pentan-3-yl", "CC[CH]CC")


def test_the_policy_is_the_engines():
    """The helper asks ``read_back_policy``; it keeps no rule of its own."""
    from orthonym.validation.read_back_policy import smiles_has_radical_electrons
    for smiles in ("CCO", "C[CH2]", "[CH2][CH2]", "COP=O", "[Sn+2].[Cl-].[Cl-]"):
        radical = smiles_has_radical_electrons(smiles)
        assert (rt_assert._independent_parse("ethyl", smiles) is not None) is radical
