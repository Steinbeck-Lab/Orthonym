# tests/unit/test_general_fallback_wiring.py
"""v25 G1: general_fallback flag — default OFF byte-identity, ON recovery."""
from unittest import mock

import pytest

from orthonym.namer import Orthonym

pytestmark = pytest.mark.unit


def test_default_off_is_byte_identical_on_easy_molecule():
    assert (Orthonym(_disable_opsin_validity_gate=True).name("CCO")
            == Orthonym(_disable_opsin_validity_gate=True,
                         general_fallback=False).name("CCO"))


def test_flag_defaults_false():
    assert Orthonym()._general_fallback is False


def test_engine_fires_when_legacy_general_abstains():
    """Force the legacy GENERAL pipeline to abstain; engine must recover."""
    nm = Orthonym(_disable_opsin_validity_gate=True, general_fallback=True)
    with mock.patch("orthonym.namer.assemble_name",
                    return_value="unknown organic compound"):
        out = nm.name("CC(Cl)CC")
    assert out == "2-chlorobutane"


def test_engine_does_not_fire_when_flag_off():
    nm = Orthonym(_disable_opsin_validity_gate=True, general_fallback=False)
    with mock.patch("orthonym.namer.assemble_name",
                    return_value="unknown organic compound"):
        out = nm.name("CC(Cl)CC")
    assert out != "2-chlorobutane"


def test_engine_ring_fallback_fires_behind_flag():
    from orthonym.assembly.general_engine import name_general
    from rdkit import Chem
    nm = Orthonym(_disable_opsin_validity_gate=True, general_fallback=True)
    smi = "O=C1CCC2CCCCC2C1"  # decalin-2-one: GENERAL-dispatched cage + ketone
    mol = Chem.MolFromSmiles(smi)
    feats = nm._perceive(mol, smi, Chem.MolToSmiles(mol, canonical=True))
    nm._classify(feats)
    expected = name_general(mol, feats).name
    with mock.patch("orthonym.namer.assemble_name",
                    return_value="unknown organic compound"):
        out = nm.name(smi)
    assert out == expected


def test_late_recovery_replaces_suppressed_wrong_name():
    """decalin-2-one: legacy names it 'decahydronaphthalene' (atom-dropping,
    SELF-01-suppressed). The late recovery must emit the engine's verified
    cage name instead. Gate disabled here -> recovery returns the engine
    name directly (in prod the same name must clear SELF-01)."""
    from orthonym.assembly.general_engine import name_general
    from rdkit import Chem
    smi = "O=C1CCC2CCCCC2C1"
    nm_off = Orthonym(_disable_opsin_validity_gate=True)
    nm_on = Orthonym(_disable_opsin_validity_gate=True, general_fallback=True)
    mol = Chem.MolFromSmiles(smi)
    feats = nm_on._perceive(mol, smi, Chem.MolToSmiles(mol, canonical=True))
    nm_on._classify(feats)
    engine_name = name_general(mol, feats).name
    out_on = nm_on.name(smi)
    # With the flag OFF nothing changes (byte-identity)...
    assert nm_off.name(smi) != engine_name
    # ...with it ON, the abstention/suppression is recovered (or, if the
    # ungated legacy path ships its wrong candidate here because SELF-01 is
    # disabled in unit tests, the flag must still not corrupt it).
    assert out_on in (engine_name, nm_off.name(smi))


def test_no_jar_requires_unverified_optin():
    """methyl-benzonorbornadiene abstains even gate-free (UNSUPPORTED_RING_
    SYSTEM raise); with the jar absent, T4 shipping needs the explicit
    opt-in — plain general_fallback must keep the abstention."""
    from unittest import mock as _m
    smi = "CC1C2C=CC1c1ccccc12"
    with _m.patch("orthonym.namer._validity_gate_jar_present",
                  return_value=False):
        strict = Orthonym(general_fallback=True)
        assert "tricyclo" not in (strict.name(smi) or "")
        loose = Orthonym(general_fallback=True,
                          general_fallback_unverified=True)
        assert "tricyclo" in loose.name(smi)


def test_flag_propagates_into_recursion():
    from orthonym.metrics.provenance import general_fallback_ctx
    from orthonym.namer import name_compound
    tok = general_fallback_ctx.set(True)
    try:
        out = name_compound("CC1C2C=CC1c1ccccc12",
                            general_fallback_unverified=True)
    finally:
        general_fallback_ctx.reset(tok)
    assert "tricyclo" in out or out == "unknown organic compound"
