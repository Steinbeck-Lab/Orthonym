"""v29 Phase 1: the binding-proof flag is off by default, audit changes no
name, and the audit path is actually EXERCISED (not merely 'no crash')."""
from unittest import mock

import pytest

from orthonym.namer import Orthonym
from orthonym.validation import proof_ledger as pl

pytestmark = pytest.mark.unit

# Byte-identity probes: a deliberately MIXED set. Most are named by the PIN
# path and never touch the engine -- that is the point, since audit mode must
# be inert for them too.
PROBES = ["CCO", "CC(=O)O", "Cc1ccccc1", "CCCCCCCCCCCC(=O)O",
          "FC(F)(F)SCCNC1CC1", "ClCCOc1ccccc1", "CC1CCC2CCCCC2C1"]

# Molecules MEASURED to route through the general engine under best-effort
# flags, i.e. ones for which a spine is genuinely recorded. The brief's
# original PROBES list did NOT reach the engine for any of its five entries
# (four are named by the PIN path, the fifth abstains without the engine), so
# the provenance check below would have asserted on an empty set forever.
#
# NOTE: conftest disables the OPSIN validity gate for the whole unit suite
# (``_disable_opsin_validity_gate_for_tests``), which changes WHICH molecules
# reach the engine: two of these three are named by a legacy handler here and
# only reach the engine in production, where SELF-01 suppresses that emission.
ENGINE_PROBES = ["ClCCOc1ccccc1", "CC1CCC2CCCCC2C1", "CC1C2C=CC1c1ccccc12"]
# The one verified to record under THIS harness (gate disabled). Used where a
# test needs a recorded spine as a precondition rather than as a sweep.
ENGINE_PROBE = "CC1C2C=CC1c1ccccc12"


def _best_effort(**kw):
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True, **kw)


def test_flag_defaults_to_off():
    assert Orthonym()._binding_proof == "off"


def test_invalid_flag_value_raises():
    with pytest.raises(ValueError):
        Orthonym(binding_proof="yes-please")


def test_audit_mode_does_not_change_any_name():
    off = _best_effort()
    audit = _best_effort(binding_proof="audit")
    for smi in PROBES:
        assert off.name(smi) == audit.name(smi), smi


def test_audit_path_is_exercised_provenance_check():
    """A proof must actually be recorded+finalized for at least one probe --
    otherwise this phase would 'pass' while running no new code at all."""
    namer = _best_effort(binding_proof="audit")
    seen = []
    for smi in ENGINE_PROBES:
        pl.clear_ledger()
        namer.name(smi)
        rec = pl.get_proof()
        if rec["stage"] is not None:
            seen.append((smi, rec))
    assert seen, "no molecule exercised the binding-proof audit path"
    assert all(r["final_name"] for _, r in seen)
    # Recorded is not the same as VERIFIED: assert a verdict was actually
    # computed, so a finalize that silently no-ops cannot pass this test.
    assert all(r["ok"] is not None for _, r in seen)


def test_off_mode_records_nothing():
    """The mirror of the provenance check: with the flag off, the same
    molecules that DO record under audit must record nothing at all."""
    namer = _best_effort()
    for smi in ENGINE_PROBES:
        pl.clear_ledger()
        namer.name(smi)
        assert pl.get_proof()["stage"] is None, smi


def test_ledger_is_reset_per_molecule():
    """Added by mutation check M11: dropping clear_ledger() from the
    per-molecule reset block broke no test, yet a molecule that records
    nothing would then finalize the PREVIOUS molecule's spine against ITS
    name -- inventing findings that never happened."""
    namer = _best_effort(binding_proof="audit")
    pl.clear_ledger()
    namer.name(ENGINE_PROBE)             # engine path: records a spine
    assert pl.get_proof()["stage"] is not None, "precondition: a spine recorded"

    namer.name("CCO")                    # PIN path: records nothing
    rec = pl.get_proof()
    assert rec["stage"] is None, f"stale record survived into a new molecule: {rec}"
    assert rec["ok"] is None, rec


def test_a_stale_spine_cannot_abstain_the_next_molecule_under_enforce():
    """The damage M11 would do, stated as behaviour: under enforce, a spine
    left over from the previous molecule would fail against this molecule's
    name and abstain a correctly-named compound. 0-wrong is not enough -- the
    audit must not manufacture abstentions either."""
    namer = _best_effort(binding_proof="enforce")
    pl.clear_ledger()
    namer.name(ENGINE_PROBE)
    assert namer.name("CCO") == "ethanol"


def test_inline_general_engine_site_records():
    """Both general-engine record sites must be hooked. The late-recovery site
    is the one real molecules reach naturally; the INLINE site is reached by
    forcing the legacy GENERAL pipeline to abstain, which is how the existing
    suite exercises it (tests/unit/test_general_fallback_wiring.py)."""
    namer = Orthonym(_disable_opsin_validity_gate=True, general_fallback=True,
                      binding_proof="audit")
    pl.clear_ledger()
    with mock.patch("orthonym.namer.assemble_name",
                    return_value="unknown organic compound"):
        out = namer.name("CC(Cl)CC")
    assert out == "2-chlorobutane"
    rec = pl.get_proof()
    assert rec["stage"] == "general_engine", rec
    assert rec["final_name"] == "2-chlorobutane"


def test_early_limit_exit_finalizes_too():
    """name() has TWO exits. The early ``return _rec`` after a recovered
    OrthonymLimitError bypasses even _apply_trivial_fallback, so a finalize
    hooked only at the main exit would leave that whole path unaudited."""
    from orthonym.namer import OrthonymLimitError
    from rdkit import Chem
    from orthonym.validation import binding_spine as bs

    namer = _best_effort(binding_proof="audit")
    mol = Chem.MolFromSmiles("CCO")
    spine = bs.BindingSpine(roots=(
        bs.SpineBinding(token="eth", kind=bs.BindingKind.PARENT,
                        atom_ids=frozenset([0, 1])),
        bs.SpineBinding(token="ol", kind=bs.BindingKind.SUFFIX,
                        atom_ids=frozenset([2])),
    ))

    def _boom(_smiles):
        raise OrthonymLimitError("UNSUPPORTED_RING_SYSTEM", "forced")

    def _recover(_smiles):
        pl.record_spine(mol, spine, stage="general_engine",
                        name_at_record="ethan-1-ol")
        return "ethan-1-ol"

    pl.clear_ledger()
    with mock.patch.object(type(namer), "_name_impl",
                           staticmethod(_boom)), \
         mock.patch.object(type(namer), "_try_general_engine_recovery",
                           staticmethod(_recover)):
        out = namer.name("CCO")
    assert out == "ethan-1-ol"
    rec = pl.get_proof()
    assert rec["final_name"] == "ethan-1-ol", (
        "the early limit exit did not finalize the ledger")
    assert rec["ok"] is True, rec


def test_a_suppressed_emission_is_not_proved_against_the_abstention_sentinel():
    """A recorded spine whose emission was SUPPRESSED downstream must not be
    asserted against 'unknown organic compound'. The proof answers "does the
    shipped name spell the graph?"; when nothing shipped, the question does
    not arise, and answering it anyway reports TOKEN_ABSENT for every token.

    Measured on 250 corpus rows before this guard: 9 of 15 recorded spines
    finalized against an abstention sentinel, i.e. 60% of the Task 7 census
    would have been this artefact rather than real proof-coverage gaps."""
    from rdkit import Chem
    from orthonym.namer import is_failure_name
    from orthonym.validation import binding_spine as bs

    namer = _best_effort(binding_proof="audit")
    mol = Chem.MolFromSmiles("CCO")
    spine = bs.BindingSpine(roots=(
        bs.SpineBinding(token="eth", kind=bs.BindingKind.PARENT,
                        atom_ids=frozenset([0, 1])),
        bs.SpineBinding(token="ol", kind=bs.BindingKind.SUFFIX,
                        atom_ids=frozenset([2])),
    ))

    def _recover(_smiles):
        pl.record_spine(mol, spine, stage="general_engine_recovery",
                        name_at_record="ethan-1-ol")
        return None          # a downstream gate suppressed the emission

    pl.clear_ledger()
    with mock.patch.object(type(namer), "_name_impl",
                           staticmethod(lambda _s: "unknown organic compound")), \
         mock.patch.object(type(namer), "_try_general_engine_recovery",
                           staticmethod(_recover)), \
         mock.patch("orthonym.assembly.t4_coverage.name_t4_complete",
                    return_value=None):
        # v33 Phase 0 L3-1: `_try_general_engine_recovery` is no longer the
        # ONLY rescue path for a suppressed emission -- `_finish` now ALSO
        # tries the systematic floor directly (`name_t4_complete`) whenever
        # the primary is a failure-name sentinel, independent of this test's
        # `_try_general_engine_recovery` mock. "CCO" is trivially
        # T4-nameable, so without also patching the floor producer this test
        # would (correctly, by L3-1's design) get a REAL rescued name here
        # instead of the abstention its scenario is built to force -- that is
        # the intended new capability, not a bug, but it defeats this test's
        # specific job of isolating the PROOF-LEDGER's suppressed-emission
        # guard. Patched to None so the abstention this test needs as a
        # precondition still holds.
        out = namer.name("CCO")

    assert is_failure_name(out)
    rec = pl.get_proof()
    assert rec["stage"] == "general_engine_recovery", "precondition: recorded"
    assert rec["ok"] is None, (
        "a suppressed emission must not be proved against the abstention "
        f"sentinel: {rec}")


def test_a_retained_name_swap_after_the_certificate_is_reported(monkeypatch):
    """The EXPECTED true positive, asserted so it cannot be silently lost.

    _retained_structural_preference can replace the whole name AFTER
    verify_certificate passed, so the recorded bindings then describe a string
    that is no longer shipped and the re-anchor must say so (TOKEN_ABSENT).
    This is a real gap in today's proof coverage -- Phase 3 produces bindings
    for the substituted name -- and it is deliberately NOT suppressed.

    It also pins the abstention guard's SCOPE: that guard skips only the
    failure sentinel, so a swap to a genuine name must still be proved."""
    # The swap consults the INSTANCE attribute (conftest only sets a
    # module-level gate flag), so it must be disabled here explicitly.
    namer = _best_effort(binding_proof="audit",
                         _disable_opsin_validity_gate=True)
    monkeypatch.setattr(type(namer), "_retained_structural_preference",
                        staticmethod(lambda _mol: "anthracene"))
    pl.clear_ledger()
    out = namer.name(ENGINE_PROBE)

    assert out == "anthracene", "precondition: the retained swap took effect"
    rec = pl.get_proof()
    assert rec["final_name"] == "anthracene"
    assert rec["name_at_record"] != "anthracene", (
        "precondition: the certificate was computed on a different string")
    assert rec["ok"] is False, rec
    from orthonym.validation import binding_spine as bs
    assert bs.TOKEN_ABSENT in rec["codes"], rec


def test_enforce_mode_abstains_on_a_failed_proof(monkeypatch):
    from orthonym.validation import binding_spine as bs
    namer = _best_effort(binding_proof="enforce")
    bad = bs.SpineProof(ok=False, findings=(
        bs.Finding(bs.ATOM_UNBOUND, "forced", "error"),), stats={})
    monkeypatch.setattr("orthonym.validation.proof_ledger.finalize",
                        lambda *a, **k: bad)
    from orthonym.namer import is_failure_name
    assert is_failure_name(namer.name("CCO"))


def test_audit_mode_does_not_abstain_on_the_same_failed_proof(monkeypatch):
    """The differential that makes the enforce test mean something: under
    audit the identical failing proof must leave the name untouched. Without
    this, an _apply_binding_proof that abstained in BOTH modes would pass."""
    from orthonym.validation import binding_spine as bs
    namer = _best_effort(binding_proof="audit")
    bad = bs.SpineProof(ok=False, findings=(
        bs.Finding(bs.ATOM_UNBOUND, "forced", "error"),), stats={})
    monkeypatch.setattr("orthonym.validation.proof_ledger.finalize",
                        lambda *a, **k: bad)
    assert namer.name("CCO") == "ethanol"
