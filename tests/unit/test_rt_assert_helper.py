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
