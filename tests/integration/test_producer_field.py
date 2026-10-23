"""The ``producer`` of the ``name_tiered`` row: the code that built the name that ships.

The pairs of ``tests/support/producer_pairs.py`` run the shipped path end to end. The tests at the
end force the call sites that no corpus row reached when the seams were wired, so that each seam
is exercised by a test that fails when the wrap is missing or reads the wrong molecule.
"""
import sys

import pytest
from rdkit import Chem

import orthonym.namer as namer
from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.errors import is_failure_name
from orthonym.metrics import producer_trail as pt
from tests.support.producer_pairs import HEROIN, LABELLED_ESTER, PAIRS, PREFIX_ORDER_ROW

# The suite switches the OPSIN validity gate off unless a test asks for it; the producers of the
# shipped configuration are the ones with the gate on.
pytestmark = [pytest.mark.integration, pytest.mark.opsin_gate]

FAILURE = "unknown organic compound"


def row(smiles, tier="pin"):
    flags = {} if tier == "pin" else _emit_tier_flags(tier)
    return Orthonym(style="pin", **flags).name_tiered(smiles)


@pytest.mark.parametrize(
    "smiles,tier,expected", PAIRS,
    ids=[f"{expected}-{i}" for i, (_s, _t, expected) in enumerate(PAIRS)])
def test_producer_of_the_shipped_name(smiles, tier, expected):
    r = row(smiles, tier)
    assert r["producer"] == expected, r


# a NO_VERIFIED_PIN decline, an UNNAMEABLE decline, and a natural-product name the default tier withholds
@pytest.mark.parametrize("smiles", [LABELLED_ESTER, "CC(=O)OCl", HEROIN])
def test_producer_is_null_when_nothing_ships(smiles):
    r = row(smiles, "pin")
    assert r["tier"] == "abstain" and r["producer"] is None, r


# ---------------------------------------------------------------------------
# Forced call sites, one test each. A trace over 8,700 namings (a dev split, the gold rows, Blue Book
# rows, pubchem_2000 and chebi_5000 rows) found no row whose shipped name came from the last-resort
# decomposition of name, from the three quality gates of _name_impl, or from the perf-budget T4
# rescue; one row ships from _try_t4_rescue, which the forced test pins beside the pair.
# ---------------------------------------------------------------------------

def _best_effort_engine():
    return Orthonym(style="pin", **_emit_tier_flags("best-effort"))


def _decomposing_stub(monkeypatch, callers):
    """``try_decompose`` that names ``ethanol`` for every caller but the routing table's shim, which
    it declines so that the cascade goes on to the site under test; ``callers`` collects who called."""
    import orthonym.decomposition as decomposition

    def stub(mol, style="pin"):
        who = sys._getframe(1).f_code.co_name
        callers.append(who)
        return None if who == "_handle_decomposition_pre_general" else "ethanol"

    monkeypatch.setattr(decomposition, "try_decompose", stub)


def test_the_t4_rescue_reports_what_it_built(monkeypatch):
    monkeypatch.setattr("orthonym.assembly.t4_coverage.name_t4_complete", lambda mol, feats: "ethanol")
    token = pt.open_trail("CCO")
    try:
        out = _best_effort_engine()._try_t4_rescue(Chem.MolFromSmiles("CCO"), None, "CCO")
        assert out is not None and out[0] == "ethanol", out
        assert pt.lookup("ethanol") == "engine:t4"
    finally:
        pt.close_trail(token)


def test_the_perf_budget_t4_rescue_reports_what_it_built(monkeypatch):
    monkeypatch.setattr("orthonym.assembly.t4_coverage.name_t4_complete", lambda mol, feats: "ethanol")
    token = pt.open_trail("CCO")
    try:
        assert _best_effort_engine()._try_perf_budget_t4_rescue("CCO") == "ethanol"
        assert pt.lookup("ethanol") == "engine:t4"
    finally:
        pt.close_trail(token)


def test_the_last_resort_decomposition_of_name_reports_what_it_built(monkeypatch):
    # the main path ships a failure, so name asks the decomposition engine before abstaining
    callers = []
    _decomposing_stub(monkeypatch, callers)
    monkeypatch.setattr(Orthonym, "_name_impl", lambda self, *a, **k: FAILURE)
    r = row("CCO")
    assert callers and set(callers) == {"run"}, callers     # _FailureRescues.run, in name
    assert r["name"] == "ethanol" and r["producer"] == "decomposition", r


@pytest.mark.parametrize("gate", ["garbled", "low_confidence", "low_atom_coverage"])
def test_the_quality_gates_of_name_impl_report_what_they_built(monkeypatch, gate):
    # a name the composer built is rejected by one of the three quality gates of _name_impl, which
    # then asks the decomposition engine; each gate is reached alone
    smiles = "N=C(N)NC=O"
    callers = []
    _decomposing_stub(monkeypatch, callers)
    monkeypatch.setattr("orthonym.namer.assemble_name",
                        lambda features, style="pin": "cycloane" if gate == "garbled" else "propane")
    if gate != "garbled":
        conf = {"confidence": 0.1, "handler": "chain"} if gate == "low_confidence" else {
            "confidence": 0.9, "handler": "chain", "factors": {"atom_coverage": 0.1}}
        monkeypatch.setattr("orthonym.assembly.coverage_scoring.retrieve_confidence", lambda: conf)
        monkeypatch.setattr("orthonym.decomposition.bond_cleavage.find_cleavable_bonds", lambda mol: [1])
    token = pt.open_trail(smiles)
    try:
        out = Orthonym(style="pin")._name_impl(smiles)
        assert callers == ["_handle_decomposition_pre_general", "_name_impl"], callers
        assert out == "ethanol"
        assert pt.lookup("ethanol") == "decomposition"
        assert pt.lookup("cycloane" if gate == "garbled" else "propane") == "composer"
    finally:
        pt.close_trail(token)


def test_the_decomposition_rescue_of_name_carries_the_label_through_the_gate(monkeypatch):
    # name sends the decomposition engine's name through the validity gate, which can rewrite it
    # (a recomposed stereo descriptor, a grammar carve-out): the rewritten name keeps the producer
    callers = []
    _decomposing_stub(monkeypatch, callers)
    monkeypatch.setattr(Orthonym, "_name_impl", lambda self, *a, **k: FAILURE)
    real_gate = namer._final_opsin_validity_gate
    seen = []

    def gate(name, *a, **k):
        out = "ethyl alcohol" if name == "ethanol" else real_gate(name, *a, **k)
        seen.append((name, out))
        return out

    monkeypatch.setattr(namer, "_final_opsin_validity_gate", gate)
    r = row("CCO")
    assert ("ethanol", "ethyl alcohol") in seen, seen        # the rescue's gate rewrote the name
    assert r["name"] == "ethyl alcohol" and r["producer"] == "decomposition", r


# ---------------------------------------------------------------------------
# The late paths: the rewriting steps, the tables, and the withdrawn name that comes back.
# ---------------------------------------------------------------------------

CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
# another spelling of ethanol that OPSIN reads back to CCO: a step that returns it has rewritten
# the name, and the best-effort tier ships it
SPELLING = "ethyl alcohol"


@pytest.mark.parametrize("smiles", ["CCO", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1", LABELLED_ESTER, "CC(=O)OCl"])
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_named_rows_have_a_producer(smiles, tier):
    # the last two are declines at the default tier and named at best-effort: both sides of the converse
    r = row(smiles, tier)
    assert (r["producer"] is None) == (r["tier"] == "abstain" or r["name"] is None), r


# One forced-path test per step that turns a string into another: the step is stubbed to return
# another spelling of ethanol, and the row's producer is the one of the string it was given.
_STEP_STUBS = {
    "stereo_check": (namer, "_final_stereo_check", lambda mol, name, *a, **k: SPELLING),
    "grammar_check": (namer, "_final_grammar_check", lambda name, *a, **k: SPELLING),
    "validity_gate": (namer, "_final_opsin_validity_gate", lambda name, *a, **k: SPELLING),
    "retry_cascade": (Orthonym, "_retry_cascade_on_gate_rejection", lambda self, smiles, pre, gated: SPELLING),
    "exit_round_trip_check": (Orthonym, "_exit_round_trip_check", lambda self, name, smiles: SPELLING),
    "prefix_order": (Orthonym, "_maybe_prefix_order_fallback", lambda self, name, smiles: SPELLING),
    "offer_selection": (Orthonym, "_select_rt_passing_offer_name", lambda self, current, smiles: SPELLING),
}


@pytest.mark.parametrize("step", sorted(_STEP_STUBS))
def test_carry(monkeypatch, step):
    owner, attr, stub = _STEP_STUBS[step]
    monkeypatch.setattr(owner, attr, stub)
    r = row("CCO", "best-effort")
    assert r["name"] == SPELLING, r                     # the stub ran and its string shipped
    assert r["producer"] == "outer:retained_name", r    # the producer of 'ethanol', the string it was given


def test_carry_never_labels_a_failure_name(monkeypatch):
    # a step that withdraws a name leaves the failure label unrecorded, whatever it was given
    monkeypatch.setattr(namer, "_final_stereo_check", lambda mol, name, *a, **k: FAILURE)
    token = pt.open_trail("CCO")
    try:
        _best_effort_engine().name("CCO")
        recorded = pt._TRAIL.get().by_name
        assert recorded and not [n for n in recorded if is_failure_name(n)], recorded
    finally:
        pt.close_trail(token)


def test_the_retry_after_a_rejected_candidate_carries_the_label(monkeypatch):
    # gate_retry applies the three steps to every later candidate: each one passes the label on
    from orthonym.gate_retry import gate_candidate
    monkeypatch.setattr(namer, "_final_stereo_check", lambda mol, name, *a, **k: name + " (1)")
    monkeypatch.setattr(namer, "_final_grammar_check", lambda name, *a, **k: name + " (2)")
    monkeypatch.setattr(namer, "_final_opsin_validity_gate", lambda name, *a, **k: name + " (3)")
    token = pt.open_trail("CCO")
    try:
        pt.noted("outer:retained_name", "ethanol", Chem.MolFromSmiles("CCO"))
        out = gate_candidate(_best_effort_engine(), "CCO", "ethanol")
        assert out == "ethanol (1) (2) (3)"
        assert [pt.lookup(n) for n in ("ethanol (1)", "ethanol (1) (2)", out)] == ["outer:retained_name"] * 3
    finally:
        pt.close_trail(token)


def _compose_decision(monkeypatch, cand, composed, labelled):
    """``Orthonym._stereo_emit_decision`` for ``CCO`` with every check that precedes the composition
    stubbed, so that the step is reached without OPSIN; the composition returns ``composed``.
    ``labelled``: whether a seam returned ``cand`` before."""
    import orthonym.rules.stereochemistry as stereochemistry
    monkeypatch.setattr(stereochemistry, "general_engine_stereo_complete", lambda mol, name: False)
    monkeypatch.setattr(namer, "_validity_gate_jar_present", lambda: True)
    monkeypatch.setattr(namer, "_validity_gate_name_to_smiles", lambda name: "CC")
    monkeypatch.setattr(namer, "_self_consistency_skeleton", lambda smiles: "SKELETON")
    monkeypatch.setattr(namer, "_self_consistency_full_key", lambda smiles: "KEY")
    monkeypatch.setattr(namer, "_STEREO_OMISSION_RECLAIM", True)
    monkeypatch.setattr(Orthonym, "_rt_match", staticmethod(lambda a, b, flagged: False))
    monkeypatch.setattr(namer, "_try_compose_input_stereo", lambda name, smiles, key: composed)
    mol = Chem.MolFromSmiles("CCO")
    token = pt.open_trail("CCO")
    try:
        if labelled:
            pt.noted("engine:t4", cand, mol)
        out = _best_effort_engine()._stereo_emit_decision(mol, cand, "CCO")
        return out, pt.lookup(cand), pt.lookup(composed) if composed else None
    finally:
        pt.close_trail(token)


def test_the_stereo_composition_of_the_emit_decision_carries_the_label(monkeypatch):
    # the general engine and omit the descriptor of an exocyclic double bond between two rings; the
    # emit decision composes it onto their string, and the composed string is the one that ships
    out, was, now = _compose_decision(monkeypatch, "oxolane", "(2E)-oxolane", labelled=True)
    assert out == (True, False, "(2E)-oxolane"), out
    assert (was, now) == ("engine:t4", "engine:t4")


def test_the_stereo_composition_of_the_emit_decision_invents_no_label(monkeypatch):
    # a candidate no seam returned has no label to pass on; a composition that fails leaves the record alone
    out, was, now = _compose_decision(monkeypatch, "oxolane", "(2E)-oxolane", labelled=False)
    assert out == (True, False, "(2E)-oxolane") and (was, now) == (None, None), (out, was, now)
    out, was, now = _compose_decision(monkeypatch, "oxolane", None, labelled=True)
    assert out == (False, False, None) and (was, now) == ("engine:t4", None), (out, was, now)


def test_the_relative_stereo_offer_names_the_producer_of_its_source():
    # (the Blue Book): at best-effort the name with lowercase r/s that a producer built
    # is converted into the cis/trans name (rules/relative_stereo.py), a new string that offers itself
    # in place of an abstention; the string has the producer of the name it was made from
    from orthonym.rules import relative_stereo as rs
    smiles, strict = "O[C@H]1CC[C@H](O)CC1", "(1r,4r)-cyclohexane-1,4-diol"
    mol = Chem.MolFromSmiles(smiles)
    engine = _best_effort_engine()
    for labelled in (True, False):
        rs.clear_withdrawn_names()
        rs.note_withdrawn_name(strict, smiles, smiles)
        engine._offers = []
        token = pt.open_trail(smiles)
        try:
            if labelled:
                pt.noted("inner:general_acyclic", strict, mol)
            engine._maybe_append_relative_stereo_offer(mol, smiles, FAILURE, primary_is_failure=True)
            assert [o.name for o in engine._offers] == ["trans-cyclohexane-1,4-diol"], engine._offers
            assert pt.lookup("trans-cyclohexane-1,4-diol") == ("inner:general_acyclic" if labelled else None)
        finally:
            pt.close_trail(token)
            rs.clear_withdrawn_names()


def test_restore_withdrawn_keeps_the_producer(monkeypatch):
    # The isotope decorator names the labelled ester; OPSIN reads the name as ambiguous and the gate
    # withdraws it; the last step of name gives the withdrawn name back. The name is the one the
    # decorator returned, which was recorded before the gate withdrew it.
    calls = []
    real = namer.restore_withdrawn

    def spy(name, smiles, *a, **k):
        out = real(name, smiles, *a, **k)
        calls.append((name, out))
        return out

    monkeypatch.setattr(namer, "restore_withdrawn", spy)
    r = row(LABELLED_ESTER, "best-effort")
    name_in, name_out = calls[-1]
    # the path was exercised: a failure name went in and the withdrawn name came out
    assert is_failure_name(name_in) and not is_failure_name(name_out), calls
    assert r["name"] == name_out, (r["name"], calls)
    assert r["producer"] == "isotope", r


def test_the_prefix_order_floor_reports_t4():
    # the name that _maybe_prefix_order_fallback ships is built by the floor in another order of the
    # prefixes, which no other seam covers (the step is only ever given a failure name)
    r = row(PREFIX_ORDER_ROW, "best-effort")
    assert r["prefix_order_fallback"] is True and r["source"] == "t4_floor", r
    assert r["producer"] == "engine:t4", r


# ---------------------------------------------------------------------------
# The retained-name tables of the last resort, forced: no molecule of the corpora reaches them
# with the validity gate on (the metal-complex list is covered by the pairs).
# ---------------------------------------------------------------------------

def test_table_trivial_fallback_reports_the_table_name(monkeypatch):
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(CAFFEINE))
    monkeypatch.setattr("orthonym.data.get_unverified_retained_name",
                        lambda smiles: "forcedname" if smiles == canonical else None)
    engine = Orthonym(style="pin")
    token = pt.open_trail(CAFFEINE)
    try:
        assert engine._apply_trivial_fallback(FAILURE, CAFFEINE) == "forcedname"
        assert pt.lookup("forcedname") == "table:trivial"
        assert pt.lookup(FAILURE) is None
        assert engine._apply_trivial_fallback("ethanol", CAFFEINE) == "ethanol"   # a name is left alone
        assert pt.lookup("ethanol") is None
    finally:
        pt.close_trail(token)
    # the table name is of the input: with the record of another molecule it is not recorded
    token = pt.open_trail("CCO")
    try:
        assert engine._apply_trivial_fallback(FAILURE, CAFFEINE) == "forcedname"
        assert pt.lookup("forcedname") is None
    finally:
        pt.close_trail(token)


def test_table_trivial_one_atom_ion_salt(monkeypatch):
    # the salt branch of name: when the producers leave a failure name, the table's name of the
    # salt replaces it. The stub reaches the branch (the class that names the salt would otherwise
    # return the same string first, and keep it); it does not depend on how the salt is named.
    monkeypatch.setattr(Orthonym, "_name_impl", lambda self, *a, **k: FAILURE)
    r = row("[Li+].[Li+].[Li+].[N-3]", "best-effort")
    assert r["name"] == "lithium nitride" and r["source"] == "trivial_retained", r
    assert r["producer"] == "table:trivial", r


def test_table_np_parent_exact_list_name(monkeypatch):
    # every producer declines, so name ends at the exact-match list of natural-product parents
    import orthonym.assembly.general_engine as general_engine
    import orthonym.assembly.t4_coverage as t4_coverage
    import orthonym.decomposition as decomposition
    monkeypatch.setattr(Orthonym, "_name_impl", lambda self, *a, **k: FAILURE)
    for attr in ("_try_general_engine_recovery", "_try_retained_fused_upgrade",
                 "_try_besteffort_clean_general_fallthrough", "_try_perf_budget_t4_rescue",
                 "_try_demote_senior_group_rescue", "_try_alternate_parent_rescue",
                 "_name_multifragment_complete", "_try_t4_rescue"):
        monkeypatch.setattr(Orthonym, attr, lambda self, *a, **k: None)
    monkeypatch.setattr(decomposition, "try_decompose", lambda *a, **k: None)
    monkeypatch.setattr(t4_coverage, "name_t4_complete", lambda *a, **k: None)
    monkeypatch.setattr(general_engine, "name_general", lambda *a, **k: None)
    monkeypatch.setattr(namer, "_np_exact_list_name", lambda smiles: "ethanol")
    r = row("CCO", "best-effort")
    # a list name records source trivial_retained since the label decision (lead, 2026-10-17;
    # the Blue Book "Preferred IUPAC names (PINs) are not identified for the compounds in this Chapter")
    assert r["name"] == "ethanol" and r["source"] == "trivial_retained", r
    assert r["producer"] == "table:np_parent", r


def test_table_retained_structural_preference(monkeypatch):
    # the recovery lane of name prefers the retained name of a molecule that is exactly a retained
    # structure to the engine's own name, when it round-trips. The main path is stubbed to decline,
    # so that the engine names ethanol 'ethan-1-ol' and the table's 'ethanol' replaces it.
    monkeypatch.setattr(Orthonym, "_name_impl", lambda self, *a, **k: FAILURE)
    token = pt.open_trail("CCO")
    try:
        assert _best_effort_engine().name("CCO") == "ethanol"
        recorded = pt._TRAIL.get().by_name
        assert recorded == {"ethan-1-ol": "engine:general", "ethanol": "table:retained"}, recorded
    finally:
        pt.close_trail(token)
    r = row("CCO", "best-effort")
    assert r["name"] == "ethanol" and r["producer"] == "table:retained", r
