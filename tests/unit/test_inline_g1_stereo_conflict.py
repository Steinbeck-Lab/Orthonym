""" no-abstain Phase A fix-round-1, Finding 4 (HIGH, a review adversarial
0-wrong review): the INLINE G1 emission lane (`namer.py::_name_impl`, the
`if self._general_fallback and (not name or is_failure_name(name)):` block)
consumed `_stereo_emit_decision`'s flag but never checked a flagged emission
against `_rt_match` before shipping -- it relied entirely on whatever the
caller does with `_name_impl`'s return value downstream.

Reproduced by calling `Orthonym._name_impl` DIRECTLY (bypassing `name`'s
outer retry-cascade / late-recovery layer, which for SOME molecules happens
to independently re-derive and correctly gate the same candidate via a
DIFFERENT mechanism -- see the note below): for input
`C[C@@H](O)[C@@H](N)C` (2 real stereocentres), a stubbed engine result of
`(3R)-3-aminobutan-2-ol` (omits the butan-2-ol centre, asserts the WRONG
configuration at the amino centre -- confirmed via full InChIKey: input
`...-IUYQGCFVSA-N` vs the name's parse-back `...-SYPWQXSBSA-N`, same
skeleton block `FERWBXLFSBWTDE`) was SHIPPED VERBATIM by `_name_impl` at
HEAD before this fix (`scripts/an A/B check` confirmed: HEAD ships the
conflict name unchanged; the fix abstains). This is a genuine wrong-
stereoisomer ship with zero verification at the point `_name_impl` returns.

⚠ Note on why testing at the full `.name` level does NOT discriminate for
THIS specific molecule (a real, if narrower, finding in its own right,
documented so nobody re-derives it from scratch): `.name` layers a retry-
cascade on top of `_name_impl` -- when the outer `_final_opsin_validity_gate`
 suppresses a result to the descriptive fallback, `is_failure_name`
triggers `_try_general_engine_recovery` (the LATE-RECOVERY lane, Task A's own
domain, already `_rt_match`-gated) to re-derive a candidate FRESH. For this
molecule that lane happens to independently reach the SAME right answer via
its OWN (correct) `_rt_match` check, which is why `.name` end-to-end looks
safe regardless of this fix. That is a coincidence of THIS reproduction, not
an architectural guarantee for every molecule the inline-G1 lane might
mishandle -- `_name_impl` itself must not depend on a downstream lane to
clean up after it, which is exactly what this fix (checking `_rt_match`
INSIDE the inline-G1 block, before ever setting `name = _eng.name`) ensures.

The fix routes a FLAGGED (best-effort, `_stereo_emit_decision`'s
`(True, True)`) emission through the SAME `_rt_match` superset gate the
late-recovery lane already uses (proven omission-safe / conflict-rejecting,
a review's Q1, `a temp dir/probe_rtmatch_semantics.py`) before it may ship,
without touching the unflagged (already stereo-complete) path at all.
"""
from unittest import mock

import pytest

from orthonym.namer import Orthonym
from orthonym.assembly.general_engine import GeneralEngineResult

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

# 2 real stereocentres (confirmed via a temp dir/probe_finding4_g1_stereo.py).
SMILES_2_CENTRES = "C[C@@H](O)[C@@H](N)C"

# Omits the butan-2-ol centre entirely and asserts the WRONG (negated) value
# for the amino centre -- genuine partial-omission + partial-CONFLICT. Full
# InChIKey of its OPSIN parse-back differs from the input's (same skeleton).
CONFLICT_NAME = "(3R)-3-aminobutan-2-ol"

# Omits the butan-2-ol centre but asserts the CORRECT value for the amino
# centre -- a genuine, SAFE partial omission (must still ship: /
# sanction citing fewer descriptors than the input defines as a valid,
# less-specific best-effort degrade -- never gate a true omission).
SAFE_OMISSION_NAME = "(3S)-3-aminobutan-2-ol"


def _run_inline_g1_isolated(monkeypatch, stub_name: str) -> str:
    """Call `_name_impl` directly -- isolates the inline-G1 lane from
    `name`'s outer retry-cascade / late-recovery layer (see module
    docstring for why that layer would otherwise mask this specific hole)."""
    import orthonym.assembly.general_engine as ge_mod
    import orthonym.validation.coverage_gate as cg_mod

    def _fake_name_general(mol, features, **kwargs):
        return GeneralEngineResult(name=stub_name, bindings=(),
                                    stereo_atom_to_locant={})

    monkeypatch.setattr(ge_mod, "name_general", _fake_name_general)
    monkeypatch.setattr(cg_mod, "certify_general_result",
                        lambda *a, **k: True)
    with mock.patch("orthonym.namer.assemble_name",
                    return_value="unknown organic compound"):
        nm = Orthonym(general_fallback=True, general_fallback_unverified=True)
        return nm._name_impl(SMILES_2_CENTRES)


def test_stereo_conflict_does_not_ship_from_inline_g1(monkeypatch):
    """a review Finding 4's reproduction, isolated at `_name_impl`: a stubbed
    engine result that OMITS one stereocentre and CONFLICTS on the other
    must NOT ship straight out of the inline-G1 lane."""
    out = _run_inline_g1_isolated(monkeypatch, CONFLICT_NAME)
    assert out != CONFLICT_NAME, (
        f"stereo-CONFLICT name shipped via the inline-G1 lane unverified: "
        f"{out!r}")


def test_safe_partial_omission_still_ships_from_inline_g1(monkeypatch):
    """Regression guard: a genuine, CORRECT partial-omission name (the
    documented, intentional best-effort degrade) must still ship straight
    out of the inline-G1 lane -- the fix tightens CONFLICT, not OMISSION."""
    out = _run_inline_g1_isolated(monkeypatch, SAFE_OMISSION_NAME)
    assert out == SAFE_OMISSION_NAME, (
        f"a safe partial-omission name was wrongly rejected: {out!r}")


def test_unflagged_stereo_complete_emission_unaffected(monkeypatch):
    """Regression guard: an emission that already expresses EVERY defined
    stereo element (`_stereo_emit_decision` returns `(True, False)` --
    unflagged) never even reaches the new check -- byte-identical to before
    this fix."""
    import orthonym.assembly.general_engine as ge_mod
    import orthonym.validation.coverage_gate as cg_mod

    # both centres correctly cited (verified against the input's real CIP
    # assignment -- the ONLY one of the 4 R/S combinations whose OPSIN
    # parse-back's full InChIKey matches the input's) -- complete AND correct.
    FULL_NAME = "(2R,3S)-3-aminobutan-2-ol"

    def _fake_name_general(mol, features, **kwargs):
        return GeneralEngineResult(name=FULL_NAME, bindings=(),
                                    stereo_atom_to_locant={})

    monkeypatch.setattr(ge_mod, "name_general", _fake_name_general)
    monkeypatch.setattr(cg_mod, "certify_general_result",
                        lambda *a, **k: True)
    with mock.patch("orthonym.namer.assemble_name",
                    return_value="unknown organic compound"):
        nm = Orthonym(general_fallback=True, general_fallback_unverified=True)
        out = nm._name_impl(SMILES_2_CENTRES)
    assert out == FULL_NAME, (
        f"a fully-stereo-complete (unflagged) emission was unexpectedly "
        f"altered: {out!r}")
