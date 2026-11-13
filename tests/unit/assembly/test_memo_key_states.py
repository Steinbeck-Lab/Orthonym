"""A memo of the naming call serves a stored value only in the state that computed it.

``ORTHONYM_MEMO=verify`` (every memo hit computed again and compared) found values that one key served across two
states that compute them differently: a fragment named inside the N-acyl float refusal scope of the peptide handler's
attempt and outside it, where its naming reached the float gate (the tier of a peptidoglycan fragment then depended on which scope named the fragment first),
an ester offer and a fragment seniority score at two tiers, a von Baeyer main ring with and without the main-bridge
preference, a substituent fragment with and without the located-branch fallback, and a nested naming whose own SMILES
was, or was not, on the visited set. Each test counts the computations: a second state is a miss.
"""
import pytest
from rdkit import Chem

from orthonym.assembly import fragment_naming as fn
from orthonym.assembly import memo

pytestmark = pytest.mark.unit


@pytest.fixture
def scope():
    if memo._MODE != "on":
        pytest.skip("counts memo hits; ORTHONYM_MEMO is not 'on'")
    token = memo.push_scope()
    saved = {k: getattr(fn._fragment_guard, k, None) for k in ("cache", "visited", "cycle_hits", "fragment_requests")}
    fn._fragment_guard.cache, fn._fragment_guard.visited, fn._fragment_guard.cycle_hits = {}, set(), []
    fn._fragment_guard.fragment_requests = {}
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(fn._fragment_guard, k, v)
        memo.pop_scope(token)


def test_a_fragment_named_in_the_nacyl_refusal_scope_is_cached_apart(scope, monkeypatch):
    from orthonym import namer
    from orthonym.decomposition import engine
    calls = []

    def fake(smiles, style="pin"):
        calls.append(smiles)
        if smiles.endswith("N"):
            memo.note_nacyl_gate()      # this naming reaches the N-acyl float gate
        return "a name"

    monkeypatch.setattr(namer, "name_compound", fake)

    def name_in_both_scopes(smiles):
        fn.name_fragment_recursively(smiles)
        fn.name_fragment_recursively(smiles)
        token = engine._NACYL_FLOAT_REFUSALS.set([])
        try:
            fn.name_fragment_recursively(smiles)
        finally:
            engine._NACYL_FLOAT_REFUSALS.reset(token)

    name_in_both_scopes("CCCCCCCC(C)(CC)CCCN")       # reached the gate: named again inside the scope
    assert calls == ["CCCCCCCC(C)(CC)CCCN"] * 2
    name_in_both_scopes("CCCCCCCC(C)(CC)CCCO")       # did not: the name is the same in both scopes
    assert calls[2:] == ["CCCCCCCC(C)(CC)CCCO"]


def test_the_ester_offer_and_the_fragment_seniority_are_keyed_on_the_tier(scope, monkeypatch):
    from orthonym.decomposition import fragment_ranker
    from orthonym.metrics.provenance import best_effort_ctx
    from orthonym.rules import esters, seniority
    offers, scores = [], []
    monkeypatch.setattr(esters, "name_ester", lambda mol, match: offers.append(match) or "ethyl acetate")
    monkeypatch.setattr(fragment_ranker, "_score_fragment_seniority_impl",
                        lambda smi: scores.append(smi) or (0, 0, 0, 0, 0))
    mol = Chem.MolFromSmiles("CCOC(C)=O")
    for _ in range(2):
        seniority._acyclic_ester_offered(mol, [(3, 4, 2)])
        fragment_ranker.score_fragment_seniority("CCOC(C)=O")
    assert (len(offers), len(scores)) == (1, 1)
    token = best_effort_ctx.set(True)
    try:
        seniority._acyclic_ester_offered(mol, [(3, 4, 2)])
        fragment_ranker.score_fragment_seniority("CCOC(C)=O")
    finally:
        best_effort_ctx.reset(token)
    assert (len(offers), len(scores)) == (2, 2)


def test_the_main_ring_search_is_keyed_on_the_main_bridge_preference(scope, monkeypatch):
    from orthonym.perception.rings import find_ring_bridgeheads
    from orthonym.rules.polycyclic import VonBaeyerAnalyzer
    calls = []

    def impl(self, mol, ring_atoms, bridgeheads, return_candidates=False):
        calls.append(getattr(self, "_allow_multiatom_bridge", True))
        return [0, 1, 2, 3, 4, 5], (2, 5), []

    monkeypatch.setattr(VonBaeyerAnalyzer, "_find_main_ring_impl", impl)
    mol = Chem.MolFromSmiles("C1CC2CCC1C2")
    ring = set(range(mol.GetNumAtoms()))
    heads = find_ring_bridgeheads(mol, ring)
    analyzer = VonBaeyerAnalyzer()
    analyzer._find_main_ring(mol, ring, heads)
    analyzer._find_main_ring(mol, ring, heads)
    analyzer._allow_multiatom_bridge = False
    analyzer._find_main_ring(mol, ring, heads)
    assert calls == [True, False]


def test_the_substituent_fragment_key_carries_the_branch_fallback():
    from orthonym.assembly import substituent_naming as sn
    mol = Chem.MolFromSmiles("CCCC(C)CC")
    plain = sn._nsf_memo_key(mol, [3, 4, 5, 6], 3, [0, 1, 2])
    token = sn.BRANCH_FALLBACK.set(lambda mol, atoms, root: None)
    try:
        with_fallback = sn._nsf_memo_key(mol, [3, 4, 5, 6], 3, [0, 1, 2])
    finally:
        sn.BRANCH_FALLBACK.reset(token)
    assert plain != with_fallback


def test_a_nested_naming_is_keyed_on_its_own_smiles_and_the_cycles_it_met(scope):
    from orthonym.assembly.nested_memo import cached_nested_naming
    calls = []

    def fresh(meets=None, asks=None):
        def run():
            calls.append(1)
            if asks:            # the naming asks for a fragment (``name_fragment_recursively``)
                fn._fragment_guard.fragment_requests[asks] = fn._fragment_guard.fragment_requests.get(asks, 0) + 1
            if meets:
                fn._note_cycle_hit(meets)
            return f"name {len(calls)}"
        return run

    # a naming that asks for its own SMILES is kept apart for the state where that SMILES is up the stack
    key = ("CCO",)
    assert cached_nested_naming(key, "CCO", fresh(asks="CCO")) == "name 1"
    assert cached_nested_naming(key, "CCO", fresh(asks="CCO")) == "name 1"           # a hit
    fn._fragment_guard.visited = {"CCO"}
    assert cached_nested_naming(key, "CCO", fresh(asks="CCO")) == "name 2"
    # one that never asks for it is the same in both states
    fn._fragment_guard.visited = set()
    assert cached_nested_naming(("CCCl",), "CCCl", fresh()) == "name 3"
    fn._fragment_guard.visited = {"CCCl"}
    assert cached_nested_naming(("CCCl",), "CCCl", fresh()) == "name 3"
    # a naming that met 'CCN' on the visited set is served only where 'CCN' is on it again
    key2 = ("CCC",)
    fn._fragment_guard.visited = {"CCN"}
    assert cached_nested_naming(key2, "CCC", fresh("CCN")) == "name 4"
    fn._fragment_guard.visited = set()
    assert cached_nested_naming(key2, "CCC", fresh()) == "name 5"
    fn._fragment_guard.visited = {"CCN", "CCS"}
    fn._fragment_guard.cycle_hits = []
    assert cached_nested_naming(key2, "CCC", fresh()) == "name 4"                  # a hit, and its cycle hit logged again
    assert fn._fragment_guard.cycle_hits == [frozenset({"CCN"})]


def test_a_memo_hit_counts_the_float_gate_its_computation_reached(scope):
    def gated():
        memo.note_nacyl_gate()
        return "gated"

    assert memo.cache_or_compute("test_gate", 1, gated) == "gated"
    mark = memo.nacyl_gate_mark()
    assert memo.cache_or_compute("test_gate", 1, gated) == "gated"          # a hit
    assert memo.nacyl_gate_mark() == mark + 1
    assert memo.cache_or_compute("test_gate", 2, lambda: "free") == "free"
    mark = memo.nacyl_gate_mark()
    assert memo.cache_or_compute("test_gate", 2, gated) == "free"           # a hit on an entry that reached no gate
    assert memo.nacyl_gate_mark() == mark
