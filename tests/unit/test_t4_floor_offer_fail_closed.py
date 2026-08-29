"""v33 no-abstain Phase A fix-round-1, Finding 1 (HIGH, FABLE adversarial 0-wrong
review): the `t4_floor` offer used to fail-open on `gate_outcome=not_run`.

Mechanism: `resolve_gate_outcome` returns `not_run` whenever no gate call was
recorded anywhere in the whole `name()` call (`metrics/provenance.py:222-223`).
Several real exits -- the limit-path/exception-path exits, the wildcard exit,
the isotope-decorator-failure exit -- return through `_finish` WITHOUT ever
calling `_final_opsin_validity_gate`, so `not_run` is reachable on real
inputs (measured: 8/41 real dev500 best-effort abstainers). `not_run` used to
be treated exactly like `carveout:*`/`gate_disabled` (a "no check is possible"
outcome) inside `_offer_rt_ok`, so a `t4_floor` offer with NO recorded
verdict at all could win `select_rt_passing` with ZERO verification.

FABLE proved this end-to-end (): stubbing
ONLY the T4 producer (`t4_coverage.name_t4_complete`) to return `"ethanol"`
for the real `not_run` abstainer `CC(=O)C1=C(C)S[C@@H](C)CC1=O`, the genuine
`_finish`/`_maybe_append_t4_floor_offer`/`_offer_rt_ok`/`select_rt_passing`
chain shipped `'ethanol'` -- a wrong-molecule name -- with winning offer
`source=t4_floor`, `gate_outcome=not_run`, and no gate having fired.

This test reproduces that exact scenario as a real pytest test (the negative
case: a wrong-molecule floor offer must NOT win) and its positive sibling (a
floor offer that GENUINELY verifies must still win, unaffected by the fix).
"""
import pytest

from orthonym.namer import Orthonym
from orthonym.errors import is_failure_name

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

# Measured real dev500 best-effort abstainer whose whole name() call ends
# with gate_outcome=not_run (task-A-fixround1-findings.md Finding 1).
NOT_RUN_ABSTAINER = "CC(=O)C1=C(C)S[C@@H](C)CC1=O"


def _best_effort_namer() -> Orthonym:
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                      allow_aromatic_general=True)


def test_wrong_molecule_stub_does_not_ship_fable_reproduction(monkeypatch):
    """FABLE's exact white-box mechanism proof, as a real test: stubbing ONLY
    the T4 producer to return a wrong-but-real name for a genuine not_run
    abstainer must NOT ship it. This is the reproduction FIRST, on the
    fixed code -- if this test is red, the hole is still open."""
    import orthonym.assembly.t4_coverage as t4c
    from orthonym.metrics import provenance as pv

    WRONG = "ethanol"  # a real, OPSIN-parseable name -- and NOT this molecule
    monkeypatch.setattr(t4c, "name_t4_complete", lambda mol, feats: WRONG)

    pv.clear_provenance()
    nm = _best_effort_namer()
    out = nm.name(NOT_RUN_ABSTAINER)

    assert out != WRONG, (
        f"wrong-molecule name shipped via the t4_floor fail-open hole: {out!r}")
    sel = getattr(nm, "_last_selected_offer", None)
    assert getattr(sel, "name", None) != WRONG
    assert getattr(sel, "source", None) != "t4_floor", (
        "the stubbed (wrong) t4_floor offer must not have won at all")
    # CQ5 Task A (2026-08-29): this molecule is no longer a bare abstainer. The
    # best-effort RT-failure fall-through (`_try_besteffort_clean_general_fallthrough`)
    # now offers the general engine's whole-graph systematic candidate computed in
    # a clean context, from source=general_engine -- a name that OPSIN-round-trips
    # to the input at the FULL InChIKey (incl. the 6S stereocentre), verified
    # independently of the namer (RDKit InChIKey compare). The EXACT spelling is
    # not asserted (this test is name-agnostic) and it depends on parent selection
    # in the recovery, which the `name_t4_complete` stub above perturbs: as this
    # test runs (T4 stubbed) it ships
    # `1-[(6S)-2,6-dimethyl-4-oxo-1-thiacyclohex-2-en-3-yl]ethan-1-one`, while
    # PLAIN best-effort (no stub) ships the ring-parent form
    # `(6S)-2,6-di(methan-1-yl)-4-oxo-3-(1-oxoethan-1-yl)-1-thiacyclohex-2-ene` --
    # both verified full-InChIKey RT-correct 2026-08-29. So the honest outcome is
    # now a CORRECT name, not silence -- but the FABLE 0-wrong invariant is
    # unchanged and is what this test guards: the stubbed WRONG t4_floor offer must
    # never win, and whatever DOES ship must round-trip. Encoding 0-wrong directly
    # is strictly stronger than the old "must abstain" (which rested on the
    # now-refuted premise that no verified candidate existed).
    if out is not None and not is_failure_name(out):
        from orthonym.validation.opsin_roundtrip import opsin_parse
        from rdkit import Chem
        osmi = opsin_parse(out)
        assert osmi is not None, f"shipped name does not OPSIN-parse: {out!r}"
        assert (Chem.MolToInchiKey(Chem.MolFromSmiles(osmi))
                == Chem.MolToInchiKey(Chem.MolFromSmiles(NOT_RUN_ABSTAINER))), (
            f"shipped a non-round-tripping (wrong-molecule) name: {out!r}")


def test_verified_floor_offer_still_wins_when_never_gated(monkeypatch):
    """Positive control (predicate-level, mirrors test_rt_over_pool.py's own
    style for this function): a `not_run`/never-gated floor candidate that
    DOES genuinely verify (real full-InChIKey match) must still win -- the
    fix denies-by-default on NO verdict, not on every never-gated offer."""
    import orthonym.namer as namer_mod

    monkeypatch.setattr(
        namer_mod, "_self01_lookup",
        lambda name: (None, True, "not_run: no self01, reanchor skipped"))
    monkeypatch.setattr(namer_mod, "_validity_gate_jar_present", lambda: True)

    from orthonym.validation import reconstruct as recon_mod
    monkeypatch.setattr(
        recon_mod, "verify_or_none",
        lambda name, input_smiles, name_facts=None: name)

    assert namer_mod._offer_rt_ok("ethanol", "CCO") is True


def test_never_gated_offer_denies_when_verify_or_none_cannot_confirm(monkeypatch):
    """The core Finding-1 fix, at the predicate: a `not_run` candidate that
    verify_or_none CANNOT confirm must lose (never fail-open)."""
    import orthonym.namer as namer_mod
    from orthonym.validation import reconstruct as recon_mod

    monkeypatch.setattr(
        namer_mod, "_self01_lookup",
        lambda name: (None, True, "not_run: no self01, reanchor skipped"))
    monkeypatch.setattr(namer_mod, "_validity_gate_jar_present", lambda: True)
    monkeypatch.setattr(
        recon_mod, "verify_or_none",
        lambda name, input_smiles, name_facts=None: None)

    assert namer_mod._offer_rt_ok("ethanol", "CCO") is False


def test_unavailable_skip_reanchor_denies_when_jar_present_finding2(monkeypatch):
    """Finding 2: the `unavailable`-resolved skip_reanchor branch gets the
    SAME deny-by-default treatment as `not_run` -- a transient OPSIN hiccup
    for a never-gated string must not let it win just because it is labelled
    'unavailable' rather than 'not_run'."""
    import orthonym.namer as namer_mod
    from orthonym.validation import reconstruct as recon_mod

    monkeypatch.setattr(
        namer_mod, "_self01_lookup",
        lambda name: (None, True, "unavailable: no self01, reanchor skipped"))
    monkeypatch.setattr(namer_mod, "_validity_gate_jar_present", lambda: True)
    monkeypatch.setattr(
        recon_mod, "verify_or_none",
        lambda name, input_smiles, name_facts=None: None)

    assert namer_mod._offer_rt_ok("some-name", "CCO") is False


def test_carveout_and_gate_disabled_skip_reanchor_unaffected(monkeypatch):
    """Regression guard: a genuine `carveout:*` or `gate_disabled` outcome
    keeps its advisory fail-open UNCHANGED -- neither ever calls
    `_validity_gate_jar_present` (proven via a raising stub, matching the
    existing `test_skip_reanchor_carveout_fails_open` contract in
    test_rt_over_pool.py) nor `verify_or_none`."""
    import orthonym.namer as namer_mod
    from orthonym.validation import reconstruct as recon_mod

    def _boom(*a, **k):
        raise AssertionError("must not be called for a by-design skip")

    monkeypatch.setattr(namer_mod, "_validity_gate_jar_present", _boom)
    monkeypatch.setattr(recon_mod, "verify_or_none", _boom)

    monkeypatch.setattr(
        namer_mod, "_self01_lookup",
        lambda name: (None, True, "carveout:thioperoxol: no self01"))
    assert namer_mod._offer_rt_ok("methane-SO-thioperoxol", "CS(=O)O") is True

    monkeypatch.setattr(
        namer_mod, "_self01_lookup",
        lambda name: (None, True, "gate_disabled: no self01, reanchor skipped"))
    assert namer_mod._offer_rt_ok("some-name", "CCO") is True


def test_never_gated_offer_fails_open_when_jar_genuinely_absent(monkeypatch):
    """The advisory fail-open MUST remain when OPSIN is genuinely absent --
    the fix only denies-by-default when a real check is actually possible."""
    import orthonym.namer as namer_mod
    from orthonym.validation import reconstruct as recon_mod

    def _boom(*a, **k):
        raise AssertionError("must not call verify_or_none with no jar")

    monkeypatch.setattr(
        namer_mod, "_self01_lookup",
        lambda name: (None, True, "not_run: no self01, reanchor skipped"))
    monkeypatch.setattr(namer_mod, "_validity_gate_jar_present", lambda: False)
    monkeypatch.setattr(recon_mod, "verify_or_none", _boom)

    assert namer_mod._offer_rt_ok("some-name", "CCO") is True
