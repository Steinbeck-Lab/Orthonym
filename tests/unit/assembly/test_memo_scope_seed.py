"""The junior-parent namings of ``spelling_offer`` start from the nested-call names the earlier namings
made (``memo.recording_scopes`` / ``shareable_entries`` / ``seeded_scopes``), and the acid fragments of the
lipid producers are memoised for the naming call (``lipids._name_fragment``)."""
import pytest

from orthonym.assembly import memo, spelling_offer
from orthonym.assembly.nested_memo import ambient_naming_state, cached_nested_call
from orthonym.metrics import provenance as pv


def _scope_with(entries):
    token = memo.push_scope()
    try:
        for namespace, key, value in entries:
            memo.cache_or_compute(namespace, key, lambda v=value: v)
        scope = memo._cache_var.get()
    finally:
        memo.pop_scope(token)
    return scope


def test_the_recorder_collects_each_top_level_scope_in_order():
    with memo.recording_scopes() as seen:
        first = memo.push_scope()
        inner = memo.push_scope()            # a nested open shares the scope and is not recorded again
        assert inner is None
        memo.pop_scope(first)
        second = memo.push_scope()
        memo.pop_scope(second)
    assert len(seen) == 2 and seen[0] is not seen[1]
    assert memo._recorders_var.get() == ()   # no recorder is left behind


def test_recorders_nest_so_an_inner_naming_is_seen_by_the_outer_one():
    with memo.recording_scopes() as outer:
        with memo.recording_scopes() as inner:
            token = memo.push_scope()
            memo.pop_scope(token)
    assert len(outer) == len(inner) == 1


def test_only_the_nested_call_namespaces_are_given_on():
    scope = _scope_with([("name_compound_nested", ("CCO",), "ethanol"),
                         ("lipids_fragment_name", ("CC(=O)O",), "acetic acid"),
                         ("fg_detect", ("CCO",), "not shared"),
                         ("substituent_fragment_name", (1, 2), "not shared either")])
    shared = memo.shareable_entries([scope])
    namespaces = {key[0] for key in shared if isinstance(key, tuple)}
    assert namespaces == {"name_compound_nested", "lipids_fragment_name"}


def test_a_seeded_scope_starts_with_a_copy_and_never_writes_back():
    scope = _scope_with([("name_compound_nested", ("CCO",), "ethanol")])
    seed = memo.shareable_entries([scope])
    before = dict(seed)
    with memo.seeded_scopes(seed):
        token = memo.push_scope()
        try:
            calls = []
            assert memo.cache_or_compute("name_compound_nested", ("CCO",), lambda: calls.append(1)) == "ethanol"
            assert calls == []                                    # served from the seed
            memo.cache_or_compute("name_compound_nested", ("CCN",), lambda: "ethanamine")
            memo.side_put("name_compound_nested", ("CCN",), ("meta",))
        finally:
            memo.pop_scope(token)
    assert seed == before                                          # the seed itself is untouched
    assert memo._SIDE_KEY in seed and ("name_compound_nested", ("CCN",)) not in seed


def test_a_hit_from_the_seed_replays_what_the_stored_call_wrote(monkeypatch):
    """The side record of a nested call (provenance and budget units) travels with its value."""
    first = memo.push_scope()
    pv.clear_provenance()
    try:
        cached_nested_call("name_compound_nested", ("k",),
                           lambda: (pv.record_source("general_engine", "verified"), "name")[1])
        scope = memo._cache_var.get()
    finally:
        memo.pop_scope(first)
    seed = memo.shareable_entries([scope])
    with memo.seeded_scopes(seed):
        token = memo.push_scope()
        pv.clear_provenance()
        try:
            calls = []
            assert cached_nested_call("name_compound_nested", ("k",), lambda: calls.append(1) or "other") == "name"
            assert calls == [] and pv.get_provenance()["source"] == "general_engine"
        finally:
            memo.pop_scope(token)
            pv.clear_provenance()


def test_the_offer_hands_each_junior_naming_the_names_of_the_namings_before_it(monkeypatch):
    from rdkit import Chem
    from orthonym import namer as namer_module
    smiles = "COCCNCC=Cc1ccccc1"
    seen = []

    class FakeNamer:
        _general_fallback_unverified = True
        _forced_parent_rank = 0
        _top_parent_pool_size = 3
        style = "pin"

    class Inner:
        def __init__(self, **kw):
            self.rank = kw["_forced_parent_rank"]

        def name_tiered(self, smi):
            token = memo.push_scope()                     # the naming opens its own scope
            try:
                seen.append((self.rank, memo.cache_or_compute("name_compound_nested", ("CCO",), lambda: "computed"),
                             memo.cache_or_compute("name_compound_nested", (f"rank{self.rank}",), lambda: "own")))
            finally:
                memo.pop_scope(token)
            return {"name": "x", "tier": "abstain"}

    monkeypatch.setattr(namer_module, "Orthonym", Inner)
    spelling_offer._AMINE_CHAIN_PARENT.set(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))
    monkeypatch.setattr("orthonym.validation.pin_spelling.check_pin_spelling",
                        lambda mol, name, **kw: [type("F", (), {"rule": "P-14.5", "detail": "stub"})()])
    first_pass = _scope_with([("name_compound_nested", ("CCO",), "from the first naming")])
    row = {"name": "n", "tier": "systematic_verified", "verified": "opsin", "spelling_failures": []}
    spelling_offer.prefer_clean_name(FakeNamer(), smiles, row, [first_pass])
    assert [s[:2] for s in seen] == [(1, "from the first naming"), (2, "from the first naming")]


def test_the_ambient_state_separates_the_tier_flags_of_a_call():
    from orthonym.metrics.provenance import general_fallback_ctx
    plain = ambient_naming_state()
    token = general_fallback_ctx.set(not general_fallback_ctx.get())
    try:
        assert ambient_naming_state() != plain
    finally:
        general_fallback_ctx.reset(token)
    assert ambient_naming_state() == plain


def test_an_acid_fragment_is_named_once_per_naming_and_state(monkeypatch):
    import orthonym
    from orthonym.rules import lipids
    names = []

    def fake_name_compound(smiles, *a, **k):
        names.append(smiles)
        return "butanedioic acid"

    monkeypatch.setattr(orthonym, "name_compound", fake_name_compound)
    token = memo.push_scope()
    pv.clear_provenance()
    try:
        assert lipids._name_fragment("O=C(O)CCC(=O)O") == "butanedioic acid"
        assert lipids._name_fragment("O=C(O)CCC(=O)O") == "butanedioic acid"
        assert names == ["O=C(O)CCC(=O)O"]                    # the second ask is a hit
        from orthonym.metrics.provenance import best_effort_ctx
        flag = best_effort_ctx.set(not best_effort_ctx.get())
        try:
            lipids._name_fragment("O=C(O)CCC(=O)O")
        finally:
            best_effort_ctx.reset(flag)
        assert len(names) == 2                                # another tier state is another key
    finally:
        pv.clear_provenance()
        memo.pop_scope(token)
    lipids._name_fragment("O=C(O)CCC(=O)O")                  # outside a scope nothing is cached
    lipids._name_fragment("O=C(O)CCC(=O)O")
    assert len(names) == 4


def test_a_naming_that_ran_out_of_a_hang_budget_hands_on_nothing(monkeypatch):
    from orthonym.assembly import fragment_naming as fn
    got = []
    monkeypatch.setattr(spelling_offer, "prefer_clean_name",
                        lambda namer, smi, row, passes=(): got.append(list(passes)) or row)
    monkeypatch.setattr("orthonym.assembly.default_tier_offer.offer_default_tier_row", lambda namer, smi, row: row)

    def ran_out(smi):
        token = memo.push_scope()
        memo.pop_scope(token)
        fn._count_hang_budget_trip()
        return {"name": "x", "tier": "abstain"}

    def clean(smi):
        token = memo.push_scope()
        memo.pop_scope(token)
        return {"name": "x", "tier": "abstain"}

    spelling_offer.name_tiered_with_offer(None, "CCO", ran_out)
    spelling_offer.name_tiered_with_offer(None, "CCO", clean)
    assert got[0] == [] and len(got[1]) == 1


def test_a_junior_naming_that_ran_out_of_a_hang_budget_adds_nothing_to_the_seed(monkeypatch):
    """The seed grows with each junior naming that ran clean; the entries of one that tripped a hang budget
    are cut short and the next junior must not start with them (the first-naming check is the test above)."""
    from rdkit import Chem
    from orthonym import namer as namer_module
    from orthonym.assembly import fragment_naming as fn
    smiles = "COCCNCC=Cc1ccccc1"
    seen = []

    class FakeNamer:
        _general_fallback_unverified = True
        _forced_parent_rank = 0
        _top_parent_pool_size = 4
        style = "pin"

    class Inner:
        def __init__(self, **kw):
            self.rank = kw["_forced_parent_rank"]

        def name_tiered(self, smi):
            token = memo.push_scope()
            try:
                scope = memo._cache_var.get()               # read what the scope starts with, store nothing
                inherited = [scope.get(memo._ck("name_compound_nested", (f"rank{r}",))) for r in (1, 2)]
                memo.cache_or_compute("name_compound_nested", (f"rank{self.rank}",), lambda: f"made by {self.rank}")
                seen.append((self.rank, inherited))
            finally:
                memo.pop_scope(token)
            if self.rank == 1:
                fn._count_hang_budget_trip()              # the first junior ran out of a hang budget
            return {"name": "x", "tier": "abstain"}

    monkeypatch.setattr(namer_module, "Orthonym", Inner)
    spelling_offer._AMINE_CHAIN_PARENT.set(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))
    monkeypatch.setattr("orthonym.validation.pin_spelling.check_pin_spelling",
                        lambda mol, name, **kw: [type("F", (), {"rule": "P-14.5", "detail": "stub"})()])
    row = {"name": "n", "tier": "systematic_verified", "verified": "opsin", "spelling_failures": []}
    spelling_offer.prefer_clean_name(FakeNamer(), smiles, row, [])
    by_rank = dict(seen)
    assert by_rank[1] == [None, None]                    # the first junior starts with nothing
    assert by_rank[2] == [None, None]                    # it tripped a budget: rank 2 got none of its entries
    assert by_rank[3] == [None, "made by 2"]             # the clean rank 2 is handed on to rank 3


#: heavy-atom carbamate and sulfolane amine rows whose best-effort names are built from a chain parent with a
#: spelling failure, so the offer names them again with a junior parent (1 and 2 junior namings). 2026-10-26 (PINB
#: lane): the sulfolane row is the 2-oxo one: 'CCN(CC)CCCC(C)NC1CCS(=O)(=O)C1' (3-{[5-(diethylamino)pentan-2-yl]amino}-
#: 1,1-dioxothiolane) is named '3-{[5-(diethylamino)pentan-2-yl]amino}-1λ6-thiolane-1,1-dione' by the ring heterone
#: producer now, the Blue Book: the ring -SO2- is the suffix 'dione' of the lambda heterocycle, senior to
#: the amine of,:18158), a name with no spelling failure, so the offer has no junior naming to make; the
#: ring ketone keeps the '1,1-dioxo' prefix (a ketone beside the heterone is not a case of it) and the row reaches the
#: junior namings (1 and 2) as the replaced row did
OFFERED_ROWS = ["CN(C)CCCNC(=O)OCC1(CCCCC1)COC(=O)NC2CCCCC2", "CCN(CC)CCCC(C)NC1CCS(=O)(=O)C1=O"]


@pytest.mark.parametrize("smiles", OFFERED_ROWS)
def test_the_seed_changes_neither_a_junior_row_nor_the_final_row(smiles, monkeypatch):
    from orthonym import namer as namer_module
    from orthonym.cli import _emit_tier_flags

    def named(seeded):
        juniors = []
        real = namer_module.Orthonym.name_tiered

        def spy(self, smi):
            row = real(self, smi)
            if getattr(self, "_forced_parent_rank", 0):
                juniors.append((self._forced_parent_rank, dict(row)))
            return row

        with monkeypatch.context() as m:
            m.setattr(namer_module.Orthonym, "name_tiered", spy)
            if not seeded:           # nothing is handed on: every junior naming starts with an empty scope
                m.setattr(memo, "shareable_entries", lambda scopes, into=None: {} if into is None else into)
            final = namer_module.Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
        return juniors, dict(final)

    with_seed, without_seed = named(True), named(False)
    assert with_seed[0], "the row no longer reaches the junior-parent namings; pick another row"
    assert with_seed == without_seed


def test_a_hit_of_the_acid_fragment_memo_charges_no_hang_budget(monkeypatch):
    """``replay_budgets=False`` of ``lipids._name_fragment`` is a choice, not a default: a hit spends no
    perf, analysis or work unit, as the repeat it stands for does in a warm fragment memo. The unmemoised
    repeats of CID 58399696 still charge the work budget (222 of 1,686 repeats, 4,203 of its 9,905 units),
    so with the memo the row runs 57 s base to 102 s lane with the same name (the budget no longer
    runs out; the work budget of base is 6,274 units against 6,000). Replaying the COLD cost of the stored
    call on every hit, the nested-call default, overshoots the budget about 190 times (measured over the
    same row), which is why it is not used here."""
    import orthonym
    from orthonym.assembly import fragment_naming as fn
    from orthonym.rules import lipids

    def fake_name_compound(smiles, *a, **k):
        fn.spend_perf_work(250)
        fn.spend_analysis_call(3)
        for _ in range(7):
            fn.spend_fragment_work()
        return "butanedioic acid"

    monkeypatch.setattr(orthonym, "name_compound", fake_name_compound)
    token = memo.push_scope()
    pv.clear_provenance()
    fn._fragment_guard.perf_budget = 10_000
    fn._fragment_guard.analysis_budget = 100
    fn._fragment_guard.work_budget = 100
    try:
        lipids._name_fragment("O=C(O)CCC(=O)O")
        spent = (10_000 - fn._fragment_guard.perf_budget, 100 - fn._fragment_guard.analysis_budget,
                 100 - fn._fragment_guard.work_budget)
        assert spent == (250, 3, 7)                                  # the stored call charged its own work
        for _ in range(5):
            lipids._name_fragment("O=C(O)CCC(=O)O")                  # five hits
        assert (10_000 - fn._fragment_guard.perf_budget, 100 - fn._fragment_guard.analysis_budget,
                100 - fn._fragment_guard.work_budget) == spent       #... charge nothing
    finally:
        fn._fragment_guard.perf_budget = fn._fragment_guard.analysis_budget = fn._fragment_guard.work_budget = None
        pv.clear_provenance()
        memo.pop_scope(token)
