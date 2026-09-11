""" a phase T6.2 + T6.3 — the shared stereo-emit policy & constitution-
granularity.

Root-cause logic under test:
  * ``Orthonym._stereo_emit_decision`` — ONE policy call shared by both
    general-engine emission sites. Since the 2026-08-31 accurate-or-abstain
    directive (``namer.py:4056-4067``, commit ``) EVERY tier —
    pin/valid/complete AND best-effort — ABSTAINS on un-completable dropped
    stereo: a stereo-stripped name fails the full-InChIKey round-trip, so it is
    scored a MISS and gains 0 breadth over abstaining while leaking precision
    : PINs specify every stereogenic unit; is stereo-blind).
    Completable stereo still ships FULL descriptors (``general_engine_stereo_
    complete`` → the ``(True, False)`` branch). The old
    ship-a-CONSTITUTION-ONLY-name-flagged-``stereo_unexpressed`` behaviour is
    retained only behind ``ORTHONYM_BE_STRIP_STEREO=1`` (measurement/back-
    compat); the tests below exercise that path under the env so the flagged
    emission + compare stay covered.
  * ``Orthonym._rt_match`` — round-trip compare at the granularity the
    name asserts: stereo-STRIPPED for a flagged emission, exact isomeric
    otherwise. Never credits a stereo-bearing parse for a flagged name.

The end-to-end tests drive ``_try_general_engine_recovery`` directly with a
monkeypatched ``name_general`` + ``verify_certificate`` so the policy is
isolated from the (OPSIN-heavy, hang-prone) real ring engine. This also
exercises the flagged emission through both the gate-disabled (T4) and the
gate-active (constitution-granularity) branches.
"""
import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.errors import is_failure_name
from orthonym.assembly.general_engine import GeneralEngineResult
from orthonym.validation.e1_certificate import E1Verdict


def _mk(tier, *, disable_gate=True):
    f = _emit_tier_flags(tier)
    return Orthonym(
        general_fallback=f["general_fallback"],
        general_fallback_unverified=f["general_fallback_unverified"],
        allow_aromatic_general=f["allow_aromatic_general"],
        _disable_opsin_validity_gate=disable_gate,
        _disable_grammar_validation=True,
    )


# --- T6.2: the shared policy decision -------------------------------------

def test_decision_dropped_stereo_by_tier():
    """A defined stereocentre + a stereo-free candidate name."""
    mol = Chem.MolFromSmiles("C[C@H](O)CCC")  # (2R/S)-pentan-2-ol, real centre
    cand = "pentan-2-ol"                        # constitution only, no descriptor
    # 2026-08-31 accurate-or-abstain: EVERY tier fail-closes on un-completable
    # dropped stereo -- best-effort no longer ships stripped by default (a
    # stereo-stripped name fails full-InChIKey round-trip; namer.py:4056-4067).
    for tier in ("pin", "valid", "complete", "best-effort"):
        permitted, flagged = _mk(tier)._stereo_emit_decision(mol, cand)
        assert permitted is False, tier
        assert flagged is False, tier


def test_decision_dropped_stereo_best_effort_strip_env(monkeypatch):
    """Back-compat: ORTHONYM_BE_STRIP_STEREO=1 restores the old best-effort
    ship-a-constitution-only-name-flagged behaviour (namer.py:4068-4072). The
    env only affects best-effort; the PIN tiers still fail-close."""
    monkeypatch.setenv("ORTHONYM_BE_STRIP_STEREO", "1")
    mol = Chem.MolFromSmiles("C[C@H](O)CCC")
    cand = "pentan-2-ol"
    permitted, flagged = _mk("best-effort")._stereo_emit_decision(mol, cand)
    assert permitted is True
    assert flagged is True
    for tier in ("pin", "valid", "complete"):
        permitted, flagged = _mk(tier)._stereo_emit_decision(mol, cand)
        assert permitted is False, tier
        assert flagged is False, tier


def test_decision_stereo_free_molecule_all_tiers():
    """No CIP stereo -> every tier ships, no flag (byte-identical behaviour)."""
    mol = Chem.MolFromSmiles("CCO")
    for tier in ("pin", "valid", "complete", "best-effort"):
        permitted, flagged = _mk(tier)._stereo_emit_decision(mol, "ethanol")
        assert permitted is True, tier
        assert flagged is False, tier


def test_decision_stereo_already_expressed_all_tiers():
    """When the name already carries the descriptor, no over-abstention."""
    mol = Chem.MolFromSmiles("C[C@H](O)CCC")
    cand = "(2R)-pentan-2-ol"  # Pattern-A prefix -> needs_stereo_injection False
    for tier in ("pin", "valid", "complete", "best-effort"):
        permitted, flagged = _mk(tier)._stereo_emit_decision(mol, cand)
        assert permitted is True, tier
        assert flagged is False, tier


# --- T6.3: constitution-granularity round-trip compare --------------------

def test_rt_match_flagged_is_constitution_granular():
    inp = "C[C@H](O)CCC"          # has stereo
    same_constitution = "CCCC(C)O"   # pentan-2-ol, NO stereo
    diff_constitution = "CCCCCO"     # pentan-1-ol
    # flagged: stereo-stripped compare -> same constitution matches
    assert Orthonym._rt_match(inp, same_constitution, True) is True
    assert Orthonym._rt_match(inp, diff_constitution, True) is False


def test_rt_match_unflagged_is_isomeric_exact():
    inp = "C[C@H](O)CCC"
    stripped = "CCCC(C)O"     # same constitution, stereo DROPPED
    exact = "C[C@H](O)CCC"    # identical isomeric
    # unflagged: exact isomeric -> a stereo-dropped parse must NOT match
    assert Orthonym._rt_match(inp, stripped, False) is False
    assert Orthonym._rt_match(inp, exact, False) is True


def test_rt_match_invalid_smiles_fails_closed():
    assert Orthonym._rt_match("C[C@H](O)CCC", "not a smiles", True) is False
    assert Orthonym._rt_match("not a smiles", "CCO", False) is False


# --- end-to-end: both emission sites via the late-recovery entry -----------

_STEREO_SMI = "C[C@H](O)CCC"     # real stereocentre
_CONSTITUTION_NAME = "pentan-2-ol"


def _patch_engine(monkeypatch, name=_CONSTITUTION_NAME):
    """Force name_general to yield a constitution-only name; certification ok.

    a phase B4: both emission lanes certify the ``GeneralEngineResult`` through
    ``coverage_gate.certify_general_result`` (E1 + structural binding spine).
    These tests isolate the stereo-emit + policy DOWNSTREAM of
    certification, so they fake certification as passing at that single seam."""
    import orthonym.assembly.general_engine as ge
    import orthonym.validation.coverage_gate as cg
    monkeypatch.setattr(
        ge, "name_general",
        lambda *a, **k: GeneralEngineResult(name=name, bindings=()))
    monkeypatch.setattr(
        cg, "certify_general_result",
        lambda *a, **k: True)


def test_stereo_incomplete_abstains_default_strip_env_ships_flagged(monkeypatch):
    """T6.2 end-to-end (gate disabled), under the 2026-08-31 policy: BOTH
    complete AND best-effort abstain on dropped stereo by default; the flagged
    constitution-only ship fires only under ORTHONYM_BE_STRIP_STEREO=1."""
    _patch_engine(monkeypatch)
    from orthonym.metrics import provenance as pv

    comp = _mk("complete")
    assert comp._try_general_engine_recovery(_STEREO_SMI) is None

    # default best-effort now abstains too (no stripped ship)
    be = _mk("best-effort")
    assert be._try_general_engine_recovery(_STEREO_SMI) is None

    # env-restored: best-effort ships the flagged constitution-only name
    monkeypatch.setenv("ORTHONYM_BE_STRIP_STEREO", "1")
    pv.clear_provenance()
    be2 = _mk("best-effort")
    out = be2._try_general_engine_recovery(_STEREO_SMI)
    assert out == _CONSTITUTION_NAME
    assert not is_failure_name(out)
    prov = pv.get_provenance()
    assert prov["source"] == "general_engine"
    assert prov["stereo_unexpressed"] is True


def test_best_effort_stereo_free_not_flagged(monkeypatch):
    """A stereo-free molecule ships in every tier with NO flag."""
    _patch_engine(monkeypatch, name="ethanol")
    from orthonym.metrics import provenance as pv
    pv.clear_provenance()
    be = _mk("best-effort")
    out = be._try_general_engine_recovery("CCO")
    assert out == "ethanol"
    assert pv.get_provenance()["stereo_unexpressed"] is False


def test_flagged_self01_constitution_matches_ships(monkeypatch):
    """T6.3 (gate active): a flagged name whose CONSTITUTION round-trips ships,
    marked opsin-verified. Under the 2026-08-31 default best-effort abstains at
    ``_stereo_emit_decision`` BEFORE the constitution-granularity, so we
    set ORTHONYM_BE_STRIP_STEREO=1 to actually reach the flagged-ship branch."""
    _patch_engine(monkeypatch)
    monkeypatch.setenv("ORTHONYM_BE_STRIP_STEREO", "1")
    import orthonym.namer as N
    from orthonym.metrics import provenance as pv
    monkeypatch.setattr(N, "_validity_gate_jar_present", lambda: True)
    # OPSIN parses the constitution-only name to the right CONSTITUTION (no stereo)
    monkeypatch.setattr(N, "_validity_gate_name_to_smiles", lambda name: "CCCC(C)O")
    pv.clear_provenance()
    be = _mk("best-effort", disable_gate=False)
    out = be._try_general_engine_recovery(_STEREO_SMI)
    assert out == _CONSTITUTION_NAME
    prov = pv.get_provenance()
    assert prov["opsin"] == "verified"
    assert prov["stereo_unexpressed"] is True


def test_flagged_self01_wrong_constitution_abstains(monkeypatch):
    """T6.3 (gate active): a flagged name whose stereo-stripped parse is a
    DIFFERENT constitution must fail-closed (never ship a wrong constitution).
    Set ORTHONYM_BE_STRIP_STEREO=1 so best-effort actually SHIPS the flagged
    name into ``_rt_match`` -- otherwise it abstains earlier at the stereo
    decision and this would pass for the wrong reason (green-but-blind)."""
    _patch_engine(monkeypatch)
    monkeypatch.setenv("ORTHONYM_BE_STRIP_STEREO", "1")
    import orthonym.namer as N
    monkeypatch.setattr(N, "_validity_gate_jar_present", lambda: True)
    # OPSIN parses to a DIFFERENT constitution (pentan-1-ol)
    monkeypatch.setattr(N, "_validity_gate_name_to_smiles", lambda name: "CCCCCO")
    be = _mk("best-effort", disable_gate=False)
    assert be._try_general_engine_recovery(_STEREO_SMI) is None


def test_unflagged_self01_stays_isomeric_exact(monkeypatch):
    """A non-flagged (stereo-free) emission keeps the EXACT isomeric compare —
    byte-identical to the pre-P6 ladder."""
    _patch_engine(monkeypatch, name="ethanol")
    import orthonym.namer as N
    monkeypatch.setattr(N, "_validity_gate_jar_present", lambda: True)
    monkeypatch.setattr(N, "_validity_gate_name_to_smiles", lambda name: "CCO")
    be = _mk("best-effort", disable_gate=False)
    out = be._try_general_engine_recovery("CCO")
    assert out == "ethanol"
