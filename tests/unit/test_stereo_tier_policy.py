"""v27 Phase 6 T6.2 + T6.3 — the shared stereo-emit policy & constitution-
granularity SELF-01.

Root-cause logic under test:
  * ``Orthonym._stereo_emit_decision`` — ONE policy call shared by both
    general-engine emission sites. complete/valid/pin ABSTAIN on dropped stereo
    (P-91.2.1: PINs specify every stereogenic unit; SELF-01 is stereo-blind);
    best-effort ships a CONSTITUTION-ONLY name flagged ``stereo_unexpressed``
    (P-91.2.2 sanctions omission for exactly the polycyclic classes it emits).
  * ``Orthonym._rt_match`` — SELF-01 round-trip compare at the granularity the
    name asserts: stereo-STRIPPED for a flagged emission, exact isomeric
    otherwise. Never credits a stereo-bearing parse for a flagged name.

The end-to-end tests drive ``_try_general_engine_recovery`` directly with a
monkeypatched ``name_general`` + ``verify_certificate`` so the policy is
isolated from the (OPSIN-heavy, hang-prone) real ring engine. This also
exercises the flagged emission through both the gate-disabled (T4) and the
gate-active (constitution-granularity SELF-01) branches.
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
    # complete/valid/pin abstain (fail-closed)
    for tier in ("pin", "valid", "complete"):
        permitted, flagged = _mk(tier)._stereo_emit_decision(mol, cand)
        assert permitted is False, tier
        assert flagged is False, tier
    # best-effort ships flagged (constitution-only)
    permitted, flagged = _mk("best-effort")._stereo_emit_decision(mol, cand)
    assert permitted is True
    assert flagged is True


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
    """Force name_general to yield a constitution-only name; E1 always ok."""
    import orthonym.assembly.general_engine as ge
    import orthonym.validation.e1_certificate as e1
    monkeypatch.setattr(
        ge, "name_general",
        lambda *a, **k: GeneralEngineResult(name=name, bindings=()))
    monkeypatch.setattr(
        e1, "verify_certificate",
        lambda *a, **k: E1Verdict(ok=True, reason="test"))


def test_complete_abstains_best_effort_flags(monkeypatch):
    """T6.2 end-to-end (gate disabled): complete abstains on dropped stereo,
    best-effort ships the flagged constitution-only name."""
    _patch_engine(monkeypatch)
    from orthonym.metrics import provenance as pv

    comp = _mk("complete")
    assert comp._try_general_engine_recovery(_STEREO_SMI) is None

    pv.clear_provenance()
    be = _mk("best-effort")
    out = be._try_general_engine_recovery(_STEREO_SMI)
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
    marked opsin-verified."""
    _patch_engine(monkeypatch)
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
    DIFFERENT constitution must fail-closed (never ship a wrong constitution)."""
    _patch_engine(monkeypatch)
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
