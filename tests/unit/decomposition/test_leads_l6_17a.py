"""Leads program L6, item 17a: the charged route is asked once per ion per outermost name.

Several naming paths ask ``route_charged`` for the same ion of one molecule
(``salts.name_salt``, ``ions.name_anion``, ``dispatch_table._handle_anion_small``,
``composer.assemble_ion_name``, ``namer._retry_cascade_on_gate_rejection``), and a refusal
costs the whole re-entry each time. The 108-heavy-atom sulfate anion of the ChEBI glycolipid
below was routed eight times at 48-107 analysis calls each: 483 of the 500 the molecule gets,
so the hang budget ran out and the molecule, which was named at 8a27e3df2, was not named.
``charged_router._memoised_route`` keeps the value of the first ask (a refusal included) for
the outermost ``name``; an exception keeps nothing.

No naming rule changes: every name is still OPSIN-gated and read back below. The route
serves (the Blue Book) and (the Blue Book).

Mutation check (run once, recorded in internal notes): replacing the
``@_memoised_route`` decoration of ``route_charged`` by the identity makes the memo tests and
the ``row 2`` test below fail.
"""
import pytest
from rdkit import Chem

import orthonym.rules.charged_router as cr
from orthonym.assembly import fragment_naming as fn
from orthonym.assembly import memo
from tests.support.rt_assert import name_best_effort, name_is_rt_exact

# ChEBI idx 38605 (the glycolipid sulfate without the fucose): named, but with 426 of the 500
# analysis calls spent in the route.
GLYCOLIPID_NO_FUCOSE = (
    "CCCCCCCCCCCCCCC(CCCCCCCCCCCCCC)CO[C@@H]1O[C@H](CO)[C@@H](O[C@@H]2O[C@H](CO)[C@H](O)"
    "[C@H](O[C@@H]3O[C@H](COS(=O)(=O)[O-])[C@@H](O[C@@H]4O[C@H](CO)[C@H](O)[C@H](O[C@]56C"
    "[C@H](O)[C@@H](NC5=O)[C@H]([C@H](O)[C@H](O)CO)O6)[C@H]4O)[C@H](O)[C@H]3NC(C)=O)"
    "[C@H]2O)[C@H](O)[C@H]1O.[Na+]")
# ChEBI idx 49789 (the same glycolipid with the fucose): lost, 483 of 500 spent in the route.
GLYCOLIPID_FUCOSE = (
    "CCCCCCCCCCCCCCC(CCCCCCCCCCCCCC)CO[C@@H]1O[C@H](CO)[C@@H](O[C@@H]2O[C@H](CO)[C@H](O)"
    "[C@H](O[C@@H]3O[C@H](COS(=O)(=O)[O-])[C@@H](O[C@@H]4O[C@H](CO)[C@H](O)[C@H](O[C@]56C"
    "[C@H](O)[C@@H](NC5=O)[C@H]([C@H](O)[C@H](O)CO)O6)[C@H]4O)[C@H](O[C@@H]4O[C@@H](C)"
    "[C@@H](O)[C@@H](O)[C@@H]4O)[C@H]3NC(C)=O)[C@H]2O)[C@H](O)[C@H]1O.[Na+]")
# The reason the route has no heavy-atom bound: a 55-carbon alkoxide whose PIN is
# the -olate. Restoring a 50-heavy-atom bound turns it into '1-oxidopentapentacontane'.
PENTAPENTACONTANOLATE = "[O-]" + "C" * 55


# ---------------------------------------------------------------------------
# The memo itself: ask once, keep a returned value, never keep an exception
# ---------------------------------------------------------------------------

@pytest.fixture
def scope():
    """An open memo scope (the one ``Orthonym.name`` opens) around the body."""
    token = memo.push_scope()
    try:
        yield
    finally:
        memo.pop_scope(token)


def _mol(smiles="CC(=O)[O-]"):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return mol


def test_a_repeated_ask_of_the_same_ion_runs_the_route_once(scope):
    calls = []

    def body(mol, style="pin"):
        calls.append(style)
        return ""

    ask = cr._memoised_route(body)
    ion = _mol()
    assert [ask(ion, "pin") for _ in range(5)] == [""] * 5
    assert calls == ["pin"]


def test_a_returned_name_is_kept_as_well_as_a_refusal(scope):
    calls = []

    def body(mol, style="pin"):
        calls.append(1)
        return "acetate"

    ask = cr._memoised_route(body)
    assert ask(_mol(), "pin") == ask(_mol(), "pin") == "acetate"
    assert len(calls) == 1


@pytest.mark.parametrize("exc", [fn.PerfBudgetExceeded, fn.OptionalNamingCapExceeded, ValueError])
def test_an_exception_is_never_kept(scope, exc):
    """The item-36 hazard: a memo must not keep the value of an except branch. The hang
    budget (``PerfBudgetExceeded``) and the cap of an optional naming
    (``OptionalNamingCapExceeded``) are BaseExceptions that unwind past the route."""
    calls = []

    def body(mol, style="pin"):
        calls.append(1)
        if len(calls) == 1:
            raise exc()
        return "acetate"

    ask = cr._memoised_route(body)
    with pytest.raises(exc):
        ask(_mol(), "pin")
    # the ask that raised kept nothing: the next ask runs the route again...
    assert ask(_mol(), "pin") == "acetate"
    #... and only a returned value is kept
    assert ask(_mol(), "pin") == "acetate"
    assert len(calls) == 2


def test_the_key_separates_what_the_re_entry_reads(scope):
    """Style, the breadth contexts, the route's own nesting depth and the ion (isotope,
    stereo, radical) each get their own entry."""
    from orthonym.metrics.provenance import best_effort_ctx
    calls = []

    def body(mol, style="pin"):
        calls.append(1)
        return ""

    ask = cr._memoised_route(body)
    ask(_mol(), "pin")
    ask(_mol(), "systematic")
    token = best_effort_ctx.set(True)
    try:
        ask(_mol(), "pin")
    finally:
        best_effort_ctx.reset(token)
    depth = getattr(cr._route_reentry, "depth", 0)
    cr._route_reentry.depth = depth + 1
    try:
        ask(_mol(), "pin")
    finally:
        cr._route_reentry.depth = depth
    ask(_mol("CC(=O)[O-]"), "pin")                      # a repeat of the first: a hit
    ask(_mol("[2H]C([2H])([2H])C(=O)[O-]"), "pin")      # an isotope-labelled ion: its own
    assert len(calls) == 5


def test_no_scope_no_memo():
    """Outside a name scope every ask runs the route (fail-open, as before)."""
    calls = []

    def body(mol, style="pin"):
        calls.append(1)
        return ""

    ask = cr._memoised_route(body)
    for _ in range(3):
        ask(_mol(), "pin")
    assert len(calls) == 3


def test_a_hit_spends_no_analysis_budget(scope):
    """A hit does no work, so it charges none: the repeat is the same deterministic work
    whose outcome is known. Charging its cold cost per hit is what ran the budget out."""
    fn.enter_name_scope()
    try:
        def body(mol, style="pin"):
            fn.spend_analysis_call(60)
            return ""

        ask = cr._memoised_route(body)
        before = fn._fragment_guard.analysis_budget
        ask(_mol(), "pin")
        after_first = fn._fragment_guard.analysis_budget
        for _ in range(6):
            ask(_mol(), "pin")
        assert before - after_first == 60
        assert fn._fragment_guard.analysis_budget == after_first
    finally:
        fn.exit_name_scope()


def test_a_hit_replays_the_provenance_the_first_ask_wrote(scope):
    """A refusal hit is put back to the caller's producer record exactly as a fresh refusal
    is (the decorators of ``route_charged``); a name-scoped record the fresh ask made
    (a non-PIN fragment) comes with every hit."""
    from orthonym.metrics import provenance as pv
    pv.clear_provenance()

    def body(mol, style="pin"):
        pv.record_non_pin_fragment("trimethylazaniumyl")
        return "named"

    ask = cr._memoised_route(body)
    assert ask(_mol(), "pin") == "named"
    assert "trimethylazaniumyl" in pv.get_provenance()["non_pin_fragments"]
    pv.clear_provenance()
    assert "trimethylazaniumyl" not in pv.get_provenance()["non_pin_fragments"]
    assert ask(_mol(), "pin") == "named"          # a hit: the body does not run again
    assert "trimethylazaniumyl" in pv.get_provenance()["non_pin_fragments"]
    pv.clear_provenance()


def test_route_charged_re_enters_the_pipeline_once_for_a_repeated_ask(scope, monkeypatch):
    """The real route: the re-entry of the whole pipeline on the neutral form (``_reenter``)
    runs for the first ask of an ion only."""
    seen = []
    real = cr._reenter

    def counting(neutral_smi, style):
        seen.append(neutral_smi)
        return real(neutral_smi, style)

    monkeypatch.setattr(cr, "_reenter", counting)
    ion = _mol("CCCCCCCC[O-]")
    first = cr.route_charged(ion, "pin")
    n_first = len(seen)
    assert n_first >= 1
    assert cr.route_charged(ion, "pin") == first
    assert cr.route_charged(_mol("CCCCCCCC[O-]"), "pin") == first
    assert len(seen) == n_first


# ---------------------------------------------------------------------------
# The molecules (production: the OPSIN validity gate on; read back independently)
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
@pytest.mark.slow
@pytest.mark.parametrize("smiles", [GLYCOLIPID_NO_FUCOSE, GLYCOLIPID_FUCOSE],
                         ids=["chebi-38605", "chebi-49789"])
def test_the_glycolipid_sulfates_are_named_at_best_effort(smiles):
    """ChEBI 49789 abstained (UNSUPPORTED_ELEMENT, 13-17 s) since e4d8fd0b1 and had a
    systematic_verified name at 8a27e3df2; 38605 was named with 74 of 500 calls to spare."""
    res = name_best_effort(smiles)
    name = res.get("name")
    assert name and res.get("source") != "abstain", res
    assert res.get("tier") == "systematic_verified", res
    assert name.startswith("sodium ") and "sulfonatooxy" in name, name
    assert name_is_rt_exact(name, smiles), name


@pytest.mark.opsin_gate
def test_the_long_alkoxide_stays_the_olate():
    """The 50-heavy-atom bound the route used to carry turned this into
    '1-oxidopentapentacontane'; the memo bounds the repeated work and keeps the -olate."""
    res = name_best_effort(PENTAPENTACONTANOLATE)
    assert res.get("name") == "pentapentacontan-1-olate", res
    assert res.get("tier") == "pin_verified", res
    assert name_is_rt_exact(res["name"], PENTAPENTACONTANOLATE)
