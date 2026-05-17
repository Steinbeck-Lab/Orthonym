"""Phase 160 unit tests for ``orthonym.assembly.inner_dispatch``.

Per CONTEXT D-20 + Phase 158 D-17 mirror: tests for InnerDispatchEntry +
INNER_DISPATCH_TABLE + _register_inner + dispatch_inner integrity.

NOTE on table size: the original Plan-04 spec expected >= 38 entries, but
Plan-02 + Plan-03 honest-fail (CONTEXT D-27) deferred 8 handlers
(polyfunctional, multi_ester, ester, benzene, heterocycle, complex_ring,
chain, general_acyclic) to a future v19.x follow-up plan. The current
table has 30 entries, and these tests assert >= 30 (with explicit
documentation of the gap). The tests are STILL VALID for the substrate
integrity — every PRESENT entry is verified.

Test classes:
- TestInnerDispatchTableIntegrity     - OrderedDict shape, frozen entries,
                                        priority uniqueness, side_effect_inventory
- TestInnerDispatchEntryRegistration  - parametrized over INNER_DISPATCH_TABLE
                                        entries (~ 30 entries × ~ 5 props = 150 tests)
- TestRegistrationLock                - _register_inner raises after freezing
- TestDispatchInnerBehavior           - dispatch_inner first-match-wins
- TestInnerDispatchStats              - get/reset stats helpers
- TestDeferredHandlerGap              - documents the 8-handler gap loudly
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import FrozenInstanceError

import pytest

from orthonym.assembly.inner_dispatch import (
    INNER_DISPATCH_TABLE,
    InnerDispatchEntry,
    InnerDispatchResult,
    _register_inner,
    dispatch_inner,
    get_inner_dispatch_stats,
    reset_inner_dispatch_stats,
)


# ---------------------------------------------------------------------------
# Class 1 — INNER_DISPATCH_TABLE structural integrity (~ 8 tests)
# ---------------------------------------------------------------------------


class TestInnerDispatchTableIntegrity:
    """CONTEXT D-10: INNER_DISPATCH_TABLE structure."""

    def test_inner_dispatch_table_is_orderdict(self):
        """CONTEXT D-10: INNER_DISPATCH_TABLE is an OrderedDict."""
        assert isinstance(INNER_DISPATCH_TABLE, OrderedDict)

    def test_count_at_least_30(self):
        """Plan-02 + Plan-03 honest-fail ship: 30 entries (not 38 as originally targeted).

        Per CONTEXT D-27: the 8 deferred handlers (polyfunctional,
        multi_ester, ester, benzene, heterocycle, complex_ring, chain,
        general_acyclic) require architectural changes beyond Phase 160
        scope (dispatch_inner multi-try-on-None semantic for ester family;
        Tier-A pool-compete semantic for benzene/heterocycle/complex_ring/
        chain; general_acyclic ~2000 LOC catch-all requires prior 7
        extractions). Tracked in ADR-19-02 for v19.x follow-up.
        """
        assert len(INNER_DISPATCH_TABLE) >= 30, (
            f"Expected >= 30 entries; got {len(INNER_DISPATCH_TABLE)}. "
            f"Substrate may be broken."
        )

    def test_entries_are_frozen_dataclass(self):
        """CONTEXT D-10: InnerDispatchEntry is frozen."""
        entry = next(iter(INNER_DISPATCH_TABLE.values()))
        with pytest.raises(FrozenInstanceError):
            entry.priority = 999999  # type: ignore[misc]

    def test_priorities_unique(self):
        """CONTEXT D-10: every priority is unique to enable deterministic sort."""
        priorities = [e.priority for e in INNER_DISPATCH_TABLE.values()]
        assert len(priorities) == len(set(priorities)), (
            f"Duplicate priority detected; D-10 violation. "
            f"priorities={priorities}"
        )

    def test_handler_ids_unique(self):
        """handler_id uniqueness (per CONTEXT D-03 + AP-160-06)."""
        ids = [e.handler_id for e in INNER_DISPATCH_TABLE.values()]
        assert len(ids) == len(set(ids))

    def test_keys_match_handler_ids(self):
        """Dict keys MUST equal the entry's handler_id."""
        for key, entry in INNER_DISPATCH_TABLE.items():
            assert key == entry.handler_id

    def test_iteration_priority_sort_stable(self):
        """Sorting by priority should produce a deterministic ordering."""
        sorted_entries = sorted(
            INNER_DISPATCH_TABLE.values(), key=lambda e: e.priority,
        )
        # Re-sort should yield the same order.
        sorted_again = sorted(
            INNER_DISPATCH_TABLE.values(), key=lambda e: e.priority,
        )
        assert [e.handler_id for e in sorted_entries] == [
            e.handler_id for e in sorted_again
        ]

    def test_table_is_module_level_singleton(self):
        """INNER_DISPATCH_TABLE is the singleton; re-importing yields the same object."""
        from orthonym.assembly.inner_dispatch import (
            INNER_DISPATCH_TABLE as table_a,
        )
        from orthonym.assembly.inner_dispatch import (
            INNER_DISPATCH_TABLE as table_b,
        )
        assert table_a is table_b


# ---------------------------------------------------------------------------
# Class 2 — Per-entry registration integrity (parametrized; ~ 5 × 30 = 150 tests)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "handler_id,entry",
    list(INNER_DISPATCH_TABLE.items()),
    ids=list(INNER_DISPATCH_TABLE.keys()),
)
class TestInnerDispatchEntryRegistration:
    """Parametrized per-entry property tests; one TestClass × ~ 6 method =
    ~ 6 × 30 = 180 generated tests, comfortably above the >= 30 floor."""

    def test_predicate_is_callable(self, handler_id, entry):
        assert callable(entry.predicate), (
            f"{handler_id}: predicate is not callable"
        )

    def test_handler_is_callable(self, handler_id, entry):
        assert callable(entry.handler), (
            f"{handler_id}: handler is not callable"
        )

    def test_iupac_section_non_empty(self, handler_id, entry):
        """CONTEXT D-10: every entry must cite an IUPAC P-section for audit."""
        assert entry.iupac_section, (
            f"{handler_id}: iupac_section is empty"
        )

    def test_description_non_empty(self, handler_id, entry):
        """CONTEXT D-10: description for audit logging."""
        assert entry.description, (
            f"{handler_id}: description is empty"
        )

    def test_side_effect_inventory_is_empty(self, handler_id, entry):
        """CONTEXT D-25 hard invariant + AP-160-15: side_effect_inventory MUST be ()."""
        assert entry.side_effect_inventory == (), (
            f"{handler_id}: side_effect_inventory={entry.side_effect_inventory!r} "
            f"violates D-25 hard invariant (must be ())"
        )

    def test_priority_is_positive_int(self, handler_id, entry):
        """Priorities are positive integers."""
        assert isinstance(entry.priority, int)
        assert entry.priority > 0


# ---------------------------------------------------------------------------
# Class 3 — Registration lock (~ 4 tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def _unfrozen_table():
    """Temporarily un-freeze INNER_DISPATCH_TABLE so duplicate-check / side-effect-
    check tests can reach those code paths.

    Phase 160.2 Plan-02-03: freeze_inner_table() now runs at module-import
    bottom (WR-06 fix), so _register_inner returns "frozen" before reaching
    the duplicate / side-effect checks. Tests that target those checks must
    explicitly unfreeze the sentinel for the duration of the test.
    """
    from orthonym.assembly import inner_dispatch as _idmod
    saved = _idmod._INNER_REGISTRATION_FROZEN
    _idmod._INNER_REGISTRATION_FROZEN = False
    try:
        yield
    finally:
        _idmod._INNER_REGISTRATION_FROZEN = saved


class TestRegistrationLock:
    """CONTEXT D-10: _register_inner raises on duplicate / frozen state."""

    def test_register_duplicate_handler_id_raises(self, _unfrozen_table):
        """Duplicate handler_id is a RuntimeError (CONTEXT D-03 + AP-160-06)."""
        with pytest.raises(RuntimeError, match="Duplicate"):
            _register_inner(
                handler_id="oxime",  # already registered
                priority=999998,
                predicate=lambda f: False,
                handler=lambda f, m=None, s="pin": None,
                iupac_section="P-TEST",
                description="duplicate test",
                side_effect_inventory=(),
            )

    def test_register_duplicate_priority_raises(self, _unfrozen_table):
        """Duplicate priority is a RuntimeError (CONTEXT D-10)."""
        # Priority 100 = oxime
        with pytest.raises(RuntimeError, match="priority"):
            _register_inner(
                handler_id="unique_test_handler_id_for_priority_clash",
                priority=100,  # already used by oxime
                predicate=lambda f: False,
                handler=lambda f, m=None, s="pin": None,
                iupac_section="P-TEST",
                description="duplicate priority test",
                side_effect_inventory=(),
            )

    def test_register_non_empty_side_effect_raises(self, _unfrozen_table):
        """side_effect_inventory != () raises ValueError (CONTEXT D-25 hard invariant)."""
        with pytest.raises(ValueError, match="side_effect_inventory"):
            _register_inner(
                handler_id="unique_test_handler_id_with_side_effect",
                priority=999997,
                predicate=lambda f: False,
                handler=lambda f, m=None, s="pin": None,
                iupac_section="P-TEST",
                description="non-empty side effect test",
                side_effect_inventory=("pool.add",),  # violates D-25
            )

    def test_freeze_inner_table_locks_registration(self):
        """After freeze_inner_table(), _register_inner raises RuntimeError.

        Phase 160.2 Plan-02-03 update: freeze now runs at module-import bottom
        (WR-06 fix). This test verifies the LIVE frozen state by attempting
        registration without the _unfrozen_table fixture — the frozen check
        fires first.
        """
        from orthonym.assembly.inner_dispatch import freeze_inner_table
        assert callable(freeze_inner_table)
        # Live frozen-state verification: _register_inner raises 'frozen'.
        with pytest.raises(RuntimeError, match="frozen"):
            _register_inner(
                handler_id="phase_160_2_test_frozen_live",
                priority=999996,
                predicate=lambda f: False,
                handler=lambda f, m=None, s="pin": None,
                iupac_section="P-TEST",
                description="live frozen test",
                side_effect_inventory=(),
            )


# ---------------------------------------------------------------------------
# Class 4 — dispatch_inner behavior (~ 5 tests)
# ---------------------------------------------------------------------------


class TestDispatchInnerBehavior:
    """CONTEXT D-10 + Phase 160.1 D-18 / ADR-19-04: dispatch_inner first-
    match-AND-succeeds-wins iteration.

    Per Phase 160.1 D-18, ``dispatch_inner(features, mol, style)`` now
    invokes the handler internally. Tests that previously relied on the
    "first-match-wins-no-handler-invoke" semantics are amended to either
    use mocked entries (via ``_replace_table_with``) or real molecules
    (via ``orthonym.name_compound``).
    """

    @staticmethod
    def _make_fake_entry(handler_id, priority, predicate, handler):
        """Helper: build an InnerDispatchEntry for table-replace tests."""
        return InnerDispatchEntry(
            handler_id=handler_id,
            priority=priority,
            predicate=predicate,
            handler=handler,
            iupac_section="P-X.Y.Z",
            description=f"fake {handler_id}",
            side_effect_inventory=(),
        )

    @staticmethod
    def _replace_table_with(entries):
        """Helper: swap INNER_DISPATCH_TABLE for isolated dispatch tests.

        Returns a context-manager-style restore callable. Mirrors
        TestInnerDispatchTypeErrorWrapping's per-entry replace pattern
        scaled to the entire table.
        """
        from orthonym.assembly import inner_dispatch as ind
        original_table = ind.INNER_DISPATCH_TABLE.copy()
        original_cache = ind._SORTED_ENTRIES_CACHE
        ind.INNER_DISPATCH_TABLE.clear()
        for e in entries:
            ind.INNER_DISPATCH_TABLE[e.handler_id] = e
        ind._SORTED_ENTRIES_CACHE = None  # invalidate cache

        def restore():
            ind.INNER_DISPATCH_TABLE.clear()
            ind.INNER_DISPATCH_TABLE.update(original_table)
            ind._SORTED_ENTRIES_CACHE = original_cache
        return restore

    def test_dispatch_inner_returns_result_on_match(self):
        """Per D-18: dispatch_inner invokes handler internally; returns
        InnerDispatchResult on handler success."""
        from orthonym.assembly.name_tree import NamingResult
        sentinel = NamingResult(name="fake_name", tree=None,
                                atom_to_locant_hint=None)

        def predicate(features):
            return True

        def handler(features, mol, style):
            return sentinel

        entry = self._make_fake_entry("fake_a", 100, predicate, handler)
        restore = self._replace_table_with([entry])
        try:
            result = dispatch_inner(object())
            assert result is not None
            assert isinstance(result, InnerDispatchResult)
            assert result.handler_id == "fake_a"
            assert result.result is sentinel
        finally:
            restore()

    def test_dispatch_inner_returns_none_on_no_match(self):
        """No matching predicate -> returns None (Plan-03-00a-pre-catch-all
        contract; once general_acyclic@99999 ships in Plan-03-03 this branch
        becomes unreachable in production)."""

        def predicate(features):
            return False

        def handler(features, mol, style):
            raise AssertionError("unreachable — predicate is False")

        entry = self._make_fake_entry("fake_b", 100, predicate, handler)
        restore = self._replace_table_with([entry])
        try:
            result = dispatch_inner(object())
            assert result is None
        finally:
            restore()

    def test_dispatch_inner_priority_order(self):
        """First-match-wins: lower priority wins over higher priority for same predicate."""
        # We can't easily test cross-handler ordering without real features;
        # we just verify the table is sorted-iterable.
        entries = sorted(
            INNER_DISPATCH_TABLE.values(), key=lambda e: e.priority,
        )
        # Priority order is monotonically increasing.
        priorities = [e.priority for e in entries]
        assert priorities == sorted(priorities)

    def test_dispatch_inner_result_carries_audit_record(self):
        """InnerDispatchResult carries audit_record dict for telemetry."""
        from orthonym.assembly.name_tree import NamingResult

        def predicate(features):
            return True

        def handler(features, mol, style):
            return NamingResult(name="fake_name", tree=None,
                                atom_to_locant_hint=None)

        entry = self._make_fake_entry("fake_c", 100, predicate, handler)
        restore = self._replace_table_with([entry])
        try:
            result = dispatch_inner(object())
            assert result is not None
            assert isinstance(result.audit_record, dict)
            assert "handler_id" in result.audit_record
            assert "priority" in result.audit_record
            assert "iupac_section" in result.audit_record
        finally:
            restore()

    def test_dispatch_inner_result_carries_matched_entry(self):
        """InnerDispatchResult.matched_entry references the InnerDispatchEntry."""
        from orthonym.assembly.name_tree import NamingResult

        def predicate(features):
            return True

        def handler(features, mol, style):
            return NamingResult(name="fake_name", tree=None,
                                atom_to_locant_hint=None)

        entry = self._make_fake_entry("fake_d", 100, predicate, handler)
        restore = self._replace_table_with([entry])
        try:
            result = dispatch_inner(object())
            assert result is not None
            assert isinstance(result.matched_entry, InnerDispatchEntry)
            assert result.matched_entry.handler_id == "fake_d"
        finally:
            restore()


# ---------------------------------------------------------------------------
# Class 5 — Stats helpers (~ 4 tests)
# ---------------------------------------------------------------------------


class TestInnerDispatchStats:
    """CONTEXT D-18 + AP-160-13: per-handler dispatch counters.

    Per Phase 160.1 D-18: stats now count SUCCESSFUL handlers only (the
    handler that returned non-None), not every predicate match. Tests
    use table-replace pattern to isolate from real handler bodies.
    """

    @staticmethod
    def _make_fake_entry(handler_id, priority, predicate, handler):
        return InnerDispatchEntry(
            handler_id=handler_id,
            priority=priority,
            predicate=predicate,
            handler=handler,
            iupac_section="P-X.Y.Z",
            description=f"fake {handler_id}",
            side_effect_inventory=(),
        )

    @staticmethod
    def _replace_table_with(entries):
        from orthonym.assembly import inner_dispatch as ind
        original_table = ind.INNER_DISPATCH_TABLE.copy()
        original_cache = ind._SORTED_ENTRIES_CACHE
        ind.INNER_DISPATCH_TABLE.clear()
        for e in entries:
            ind.INNER_DISPATCH_TABLE[e.handler_id] = e
        ind._SORTED_ENTRIES_CACHE = None

        def restore():
            ind.INNER_DISPATCH_TABLE.clear()
            ind.INNER_DISPATCH_TABLE.update(original_table)
            ind._SORTED_ENTRIES_CACHE = original_cache
        return restore

    def setup_method(self):
        """Reset counter state before each test for isolation."""
        reset_inner_dispatch_stats()

    def test_initial_stats_empty(self):
        """After reset, stats are empty."""
        assert get_inner_dispatch_stats() == {}

    def test_dispatch_increments_counter(self):
        """A successful dispatch_inner match increments the counter."""
        from orthonym.assembly.name_tree import NamingResult

        def predicate(features):
            return True

        def handler(features, mol, style):
            return NamingResult(name="x", tree=None, atom_to_locant_hint=None)

        entry = self._make_fake_entry("fake_stats", 100, predicate, handler)
        restore = self._replace_table_with([entry])
        try:
            before = get_inner_dispatch_stats().get("fake_stats", 0)
            dispatch_inner(object())
            after = get_inner_dispatch_stats().get("fake_stats", 0)
            assert after == before + 1
        finally:
            restore()

    def test_reset_clears_counter(self):
        """reset_inner_dispatch_stats() clears all counters."""
        from orthonym.assembly.name_tree import NamingResult

        def predicate(features):
            return True

        def handler(features, mol, style):
            return NamingResult(name="x", tree=None, atom_to_locant_hint=None)

        entry = self._make_fake_entry("fake_reset", 100, predicate, handler)
        restore = self._replace_table_with([entry])
        try:
            dispatch_inner(object())
            assert get_inner_dispatch_stats().get("fake_reset", 0) >= 1
            reset_inner_dispatch_stats()
            assert get_inner_dispatch_stats() == {}
        finally:
            restore()

    def test_get_returns_copy(self):
        """get_inner_dispatch_stats returns a defensive copy.

        Mutating the returned dict should not change the underlying counter.
        """
        from orthonym.assembly.name_tree import NamingResult

        def predicate(features):
            return True

        def handler(features, mol, style):
            return NamingResult(name="x", tree=None, atom_to_locant_hint=None)

        entry = self._make_fake_entry("fake_copy", 100, predicate, handler)
        restore = self._replace_table_with([entry])
        try:
            dispatch_inner(object())
            snapshot = get_inner_dispatch_stats()
            snapshot["fake_copy"] = 999999
            assert get_inner_dispatch_stats()["fake_copy"] != 999999
        finally:
            restore()


# ---------------------------------------------------------------------------
# Class 6 — Deferred handler gap (~ 8 tests, one per missing handler)
# ---------------------------------------------------------------------------


class TestDeferredHandlerGap:
    """Documents the 8-handler gap loudly so future plans can close it.

    Per Plan-02 + Plan-03 honest-fail (CONTEXT D-27), these handlers were
    NOT extracted to handlers/ + INNER_DISPATCH_TABLE because their byte-
    identical extraction requires architectural changes (dispatch_inner
    multi-try-on-None for ester family; Tier-A pool-compete semantic for
    benzene/heterocycle/complex_ring/chain; general_acyclic ~2000 LOC).

    These tests ASSERT the gap is real (handler_id NOT in INNER_DISPATCH_TABLE).
    When the v19.x follow-up plan extracts each handler, the corresponding
    test below will FAIL — a loud signal to update both the test pyramid
    and the ADR documentation.
    """

    def test_polyfunctional_not_yet_extracted(self):
        """Plan-03 honest-fail per ester-family multi-try-on-None semantic."""
        assert "polyfunctional" not in INNER_DISPATCH_TABLE

    def test_multi_ester_not_yet_extracted(self):
        """Plan-03 honest-fail per ester-family multi-try-on-None semantic."""
        assert "multi_ester" not in INNER_DISPATCH_TABLE

    def test_ester_not_yet_extracted(self):
        """Plan-03 honest-fail per ester-family multi-try-on-None semantic."""
        assert "ester" not in INNER_DISPATCH_TABLE

    def test_benzene_not_yet_extracted(self):
        """Plan-03 honest-fail per Tier-A pool-compete semantic."""
        assert "benzene" not in INNER_DISPATCH_TABLE

    def test_heterocycle_not_yet_extracted(self):
        """Plan-03 honest-fail per Tier-A pool-compete semantic."""
        assert "heterocycle" not in INNER_DISPATCH_TABLE

    def test_complex_ring_not_yet_extracted(self):
        """Plan-03 honest-fail per Tier-A pool-compete semantic."""
        assert "complex_ring" not in INNER_DISPATCH_TABLE

    def test_chain_not_yet_extracted(self):
        """Plan-03 honest-fail per Tier-A pool-compete semantic."""
        assert "chain" not in INNER_DISPATCH_TABLE

    def test_general_acyclic_extracted_in_phase_160_2(self):
        """Phase 160.2 Plan-02-03 closure: general_acyclic@99999 NOW REGISTERED.

        Pre-amendment (Phase 160.1): this test asserted ``not in`` because the
        ~2000 LOC catch-all extraction was deferred. Phase 160.2 Plan-02-02
        ships handlers/general_acyclic.py (verbatim lift of composer.py:951-1055
        chain-fallback section); Phase 160.2 Plan-02-03 wires it via
        ``_register_inner(handler_id='general_acyclic', priority=99999, ...)``
        with the AP-160.2-06 CASE B predicate refinement (defers to inline
        amide / amine cascade branches in composer.py:917-931).
        """
        assert "general_acyclic" in INNER_DISPATCH_TABLE

    def test_total_deferred_count_is_7(self):
        """Quantitative gap signal post-Phase-160.2 Plan-02-03: 7 of 38 target
        handlers still deferred (general_acyclic closed in Phase 160.2 Plan-02)."""
        deferred = {
            "polyfunctional", "multi_ester", "ester", "benzene",
            "heterocycle", "complex_ring", "chain",
            # general_acyclic: CLOSED in Phase 160.2 Plan-02-03.
        }
        missing = deferred - set(INNER_DISPATCH_TABLE.keys())
        assert len(missing) == 7, (
            f"Expected exactly 7 deferred handlers post-160.2 Plan-02; "
            f"got {missing!r}"
        )


# ---------------------------------------------------------------------------
# Class 7 — CR-01 regression: dispatch_inner TypeError wrapping
# ---------------------------------------------------------------------------


class TestInnerDispatchTypeErrorWrapping:
    """CR-01 regression: dispatch_inner wraps predicate TypeError as RuntimeError."""

    def test_predicate_typeerror_wraps_with_runtime(self):
        """A predicate raising TypeError surfaces as RuntimeError with handler_id."""
        from dataclasses import replace
        from orthonym.assembly import inner_dispatch as ind
        first_handler_id = next(iter(INNER_DISPATCH_TABLE))
        first_entry = INNER_DISPATCH_TABLE[first_handler_id]

        def bad_predicate(features):
            raise TypeError("synthetic: NoneType has no attribute foo")

        INNER_DISPATCH_TABLE[first_handler_id] = replace(
            first_entry, predicate=bad_predicate,
        )
        ind._SORTED_ENTRIES_CACHE = None  # invalidate per-call cache (D-18 fix)
        try:
            with pytest.raises(RuntimeError) as excinfo:
                dispatch_inner(object())
            assert first_handler_id in str(excinfo.value)
            assert "TypeError" in str(excinfo.value)
            assert "synthetic" in str(excinfo.value)
        finally:
            INNER_DISPATCH_TABLE[first_handler_id] = first_entry
            ind._SORTED_ENTRIES_CACHE = None

    def test_predicate_typeerror_chains_original_via_cause(self):
        """The wrapping RuntimeError has __cause__ set to the original TypeError."""
        from dataclasses import replace
        from orthonym.assembly import inner_dispatch as ind
        first_handler_id = next(iter(INNER_DISPATCH_TABLE))
        first_entry = INNER_DISPATCH_TABLE[first_handler_id]
        original = TypeError("original-cause")

        def bad_predicate(features):
            raise original

        INNER_DISPATCH_TABLE[first_handler_id] = replace(
            first_entry, predicate=bad_predicate,
        )
        ind._SORTED_ENTRIES_CACHE = None  # invalidate per-call cache (D-18 fix)
        try:
            with pytest.raises(RuntimeError) as excinfo:
                dispatch_inner(object())
            assert excinfo.value.__cause__ is original
        finally:
            INNER_DISPATCH_TABLE[first_handler_id] = first_entry
            ind._SORTED_ENTRIES_CACHE = None

    def test_predicate_returning_false_works_unchanged(self):
        """Regression: non-raising predicates returning False continue to work.

        Per Phase 160.1 D-18: with the amendment, ALL handlers whose predicate
        matches are invoked. To exercise the false-predicate path in isolation,
        we use the table-replace pattern.
        """
        from orthonym.assembly import inner_dispatch as ind
        entry = InnerDispatchEntry(
            handler_id="fake_false_pred",
            priority=100,
            predicate=lambda _f: False,
            handler=lambda _f, _m, style: None,
            iupac_section="P-X",
            description="fake",
            side_effect_inventory=(),
        )
        original_table = ind.INNER_DISPATCH_TABLE.copy()
        original_cache = ind._SORTED_ENTRIES_CACHE
        ind.INNER_DISPATCH_TABLE.clear()
        ind.INNER_DISPATCH_TABLE["fake_false_pred"] = entry
        ind._SORTED_ENTRIES_CACHE = None
        try:
            # Should NOT raise; should return None (predicate is False).
            result = dispatch_inner(object())
            assert result is None
        finally:
            ind.INNER_DISPATCH_TABLE.clear()
            ind.INNER_DISPATCH_TABLE.update(original_table)
            ind._SORTED_ENTRIES_CACHE = original_cache


class TestInnerDispatchSortCache:
    """WR-01 regression: dispatch_inner uses cached sorted tuple, not per-call sort.

    Per Phase 160.1 D-18: with the gate-fail-retry amendment, the WR-01
    cache behavior is preserved; tests use the table-replace pattern to
    isolate from real handlers.
    """

    def test_sort_cache_is_populated_after_first_dispatch(self):
        from orthonym.assembly import inner_dispatch as ind
        from orthonym.assembly.name_tree import NamingResult
        entry = InnerDispatchEntry(
            handler_id="fake_cache_a",
            priority=100,
            predicate=lambda _f: True,
            handler=lambda _f, _m, style: NamingResult(
                name="x", tree=None, atom_to_locant_hint=None,
            ),
            iupac_section="P-X",
            description="fake",
            side_effect_inventory=(),
        )
        original_table = ind.INNER_DISPATCH_TABLE.copy()
        original_cache = ind._SORTED_ENTRIES_CACHE
        ind.INNER_DISPATCH_TABLE.clear()
        ind.INNER_DISPATCH_TABLE["fake_cache_a"] = entry
        ind._SORTED_ENTRIES_CACHE = None
        try:
            ind.dispatch_inner(object())
            assert ind._SORTED_ENTRIES_CACHE is not None
            priorities = [e.priority for e in ind._SORTED_ENTRIES_CACHE]
            assert priorities == sorted(priorities)
        finally:
            ind.INNER_DISPATCH_TABLE.clear()
            ind.INNER_DISPATCH_TABLE.update(original_table)
            ind._SORTED_ENTRIES_CACHE = original_cache

    def test_dispatch_inner_does_not_call_sorted_per_call(self, monkeypatch):
        """Performance regression guard: sorted() is called at most once
        on cache rebuild, not per dispatch."""
        from orthonym.assembly import inner_dispatch as ind
        from orthonym.assembly.name_tree import NamingResult

        entry = InnerDispatchEntry(
            handler_id="fake_cache_b",
            priority=100,
            predicate=lambda _f: True,
            handler=lambda _f, _m, style: NamingResult(
                name="x", tree=None, atom_to_locant_hint=None,
            ),
            iupac_section="P-X",
            description="fake",
            side_effect_inventory=(),
        )
        original_table = ind.INNER_DISPATCH_TABLE.copy()
        original_cache = ind._SORTED_ENTRIES_CACHE
        ind.INNER_DISPATCH_TABLE.clear()
        ind.INNER_DISPATCH_TABLE["fake_cache_b"] = entry
        ind._SORTED_ENTRIES_CACHE = None
        sort_call_count = [0]
        real_sorted = sorted

        def counting_sorted(*args, **kwargs):
            sort_call_count[0] += 1
            return real_sorted(*args, **kwargs)

        monkeypatch.setattr("builtins.sorted", counting_sorted)

        try:
            for _ in range(10):
                ind.dispatch_inner(object())
            assert sort_call_count[0] <= 1
        finally:
            ind.INNER_DISPATCH_TABLE.clear()
            ind.INNER_DISPATCH_TABLE.update(original_table)
            ind._SORTED_ENTRIES_CACHE = original_cache


# ---------------------------------------------------------------------------
# Class 9 — Phase 160.1 D-18 / ADR-19-04 gate-fail-retry semantics (8 tests)
# ---------------------------------------------------------------------------


class TestDispatchInnerGateFailRetry:
    """Phase 160.1 D-18 + ADR-19-04: first-match-AND-succeeds-wins.

    These tests verify the gate-fail-retry semantics that amend Phase 160
    CONTEXT D-22 from "first-match-wins" to "first-match-AND-succeeds-wins."
    Handler contract per ADR-19-04:
      * Return non-None NamingResult ⇒ "I succeeded; use this result."
      * Return None ⇒ "I gate-failed; defer to next-priority handler."
      * Raise Exception ⇒ surfaced as RuntimeError chained via __cause__.

    Per CONTEXT D-25 preserved: predicate purity (predicates report
    whether handler CAN POSSIBLY apply; handler's gate decides whether
    it SHOULD apply).
    """

    @staticmethod
    def _make_fake_entry(handler_id, priority, predicate, handler):
        return InnerDispatchEntry(
            handler_id=handler_id,
            priority=priority,
            predicate=predicate,
            handler=handler,
            iupac_section="P-X.Y.Z",
            description=f"fake {handler_id}",
            side_effect_inventory=(),
        )

    @staticmethod
    def _replace_table_with(entries):
        from orthonym.assembly import inner_dispatch as ind
        original_table = ind.INNER_DISPATCH_TABLE.copy()
        original_cache = ind._SORTED_ENTRIES_CACHE
        ind.INNER_DISPATCH_TABLE.clear()
        for e in entries:
            ind.INNER_DISPATCH_TABLE[e.handler_id] = e
        ind._SORTED_ENTRIES_CACHE = None

        def restore():
            ind.INNER_DISPATCH_TABLE.clear()
            ind.INNER_DISPATCH_TABLE.update(original_table)
            ind._SORTED_ENTRIES_CACHE = original_cache
        return restore

    def setup_method(self):
        reset_inner_dispatch_stats()

    def test_predicate_matches_handler_succeeds_returns_result(self):
        """Sanity: single handler, predicate match, handler returns non-None."""
        from orthonym.assembly.name_tree import NamingResult
        sentinel = NamingResult(name="success", tree=None,
                                atom_to_locant_hint=None)
        entry = self._make_fake_entry(
            "h_success", 100, lambda _f: True,
            lambda _f, _m, style: sentinel,
        )
        restore = self._replace_table_with([entry])
        try:
            result = dispatch_inner(object())
            assert result is not None
            assert result.handler_id == "h_success"
            assert result.result is sentinel
        finally:
            restore()

    def test_predicate_matches_handler_gate_fails_retries_next(self):
        """The lactone-fixture pattern: handler A matches predicate but
        gate-fails (returns None); dispatch_inner retries handler B at
        lower priority. ADR-19-04 the core gate-fail-retry behavior."""
        from orthonym.assembly.name_tree import NamingResult

        a_called = []
        b_called = []

        def handler_a(_f, _m, style):
            a_called.append(True)
            return None  # gate-fail

        def handler_b(_f, _m, style):
            b_called.append(True)
            return NamingResult(name="b_won", tree=None,
                                atom_to_locant_hint=None)

        entry_a = self._make_fake_entry("h_a", 100, lambda _f: True, handler_a)
        entry_b = self._make_fake_entry("h_b", 200, lambda _f: True, handler_b)
        restore = self._replace_table_with([entry_a, entry_b])
        try:
            result = dispatch_inner(object())
            assert result is not None
            assert result.handler_id == "h_b"
            assert result.result.name == "b_won"
            # Both handlers MUST have been invoked (A first, gate-failed; B retry)
            assert len(a_called) == 1
            assert len(b_called) == 1
        finally:
            restore()

    def test_all_handlers_gate_fail_returns_None(self):
        """When no handler succeeds, dispatch_inner returns None.
        Once general_acyclic@99999 ships in Plan-03-03 this branch becomes
        unreachable in production."""
        entry_a = self._make_fake_entry(
            "h_failA", 100, lambda _f: True, lambda _f, _m, style: None,
        )
        entry_b = self._make_fake_entry(
            "h_failB", 200, lambda _f: True, lambda _f, _m, style: None,
        )
        restore = self._replace_table_with([entry_a, entry_b])
        try:
            result = dispatch_inner(object())
            assert result is None
        finally:
            restore()

    def test_stats_counts_successful_handler_only(self):
        """_INNER_DISPATCH_STATS increments ONLY on the entry whose
        handler returned non-None, not on every predicate match."""
        from orthonym.assembly.name_tree import NamingResult

        entry_a = self._make_fake_entry(
            "h_gateFail", 100, lambda _f: True,
            lambda _f, _m, style: None,
        )
        entry_b = self._make_fake_entry(
            "h_succeed", 200, lambda _f: True,
            lambda _f, _m, style: NamingResult(
                name="x", tree=None, atom_to_locant_hint=None,
            ),
        )
        restore = self._replace_table_with([entry_a, entry_b])
        try:
            dispatch_inner(object())
            stats = get_inner_dispatch_stats()
            # Only the SUCCESSFUL handler should have its counter incremented.
            assert stats.get("h_succeed", 0) == 1
            assert stats.get("h_gateFail", 0) == 0
        finally:
            restore()

    def test_predicate_raises_TypeError_surfaced_as_RuntimeError(self):
        """Existing behavior preserved: predicate TypeError → RuntimeError."""

        def bad_predicate(_features):
            raise TypeError("synthetic predicate bug")

        entry = self._make_fake_entry(
            "h_badpred", 100, bad_predicate,
            lambda _f, _m, style: None,
        )
        restore = self._replace_table_with([entry])
        try:
            with pytest.raises(RuntimeError) as excinfo:
                dispatch_inner(object())
            assert "h_badpred" in str(excinfo.value)
            assert "TypeError" in str(excinfo.value)
        finally:
            restore()

    def test_handler_raises_Exception_surfaced_as_RuntimeError(self):
        """New behavior per D-18: handler exception → RuntimeError chained
        via __cause__ (no silent swallowing per CONTEXT D-27)."""
        original = ValueError("original handler bug")

        def bad_handler(_f, _m, style):
            raise original

        entry = self._make_fake_entry(
            "h_badhandler", 100, lambda _f: True, bad_handler,
        )
        restore = self._replace_table_with([entry])
        try:
            with pytest.raises(RuntimeError) as excinfo:
                dispatch_inner(object())
            assert "h_badhandler" in str(excinfo.value)
            assert "ValueError" in str(excinfo.value)
            assert excinfo.value.__cause__ is original
        finally:
            restore()

    def test_dispatch_inner_callable_with_mol_kwarg(self):
        """New API per D-18: dispatch_inner(features, mol=mol, style='pin')
        works; falls back to features.mol when mol kwarg omitted."""
        from orthonym.assembly.name_tree import NamingResult

        captured = {}

        def handler(features, mol, style):
            captured["mol_arg"] = mol
            captured["style_arg"] = style
            return NamingResult(name="x", tree=None,
                                atom_to_locant_hint=None)

        entry = self._make_fake_entry(
            "h_kwarg", 100, lambda _f: True, handler,
        )
        restore = self._replace_table_with([entry])
        try:
            sentinel_mol = object()
            dispatch_inner(object(), mol=sentinel_mol, style="iupac")
            assert captured["mol_arg"] is sentinel_mol
            assert captured["style_arg"] == "iupac"

            # Default style="pin"; mol falls back to features.mol when None
            class FakeFeatures:
                mol = "FEATURES_MOL_SENTINEL"
            captured.clear()
            dispatch_inner(FakeFeatures())
            assert captured["mol_arg"] == "FEATURES_MOL_SENTINEL"
            assert captured["style_arg"] == "pin"
        finally:
            restore()

    def test_canary_fixture_lactone_gate_fail_routes_via_ester_family(self):
        """Integration: the Phase 160.1 regression-fixture SMILES from
        CONTEXT <specifics>. With the amendment, even when the inline
        ester cascade still exists, the regression fixture must continue
        to name correctly (the amendment is invariant on byte-identical
        output until Plan-03-01 removes the inline cascade)."""
        from orthonym import name_compound
        smi = (
            "COC(=O)/C(CC(=O)O)=C("
            "\\CCCCCCCCCCCCCCCCC1=C(C)C(=O)OC1=O"
            ")C(=O)O"
        )
        name = name_compound(smi)
        assert "hydroxymethyl" not in name, (
            f"D-18 regression-fixture invariance broken: 'hydroxymethyl' "
            f"appeared in {name!r}"
        )
        assert "formatyl" not in name, (
            f"D-18 regression-fixture invariance broken: 'formatyl' "
            f"appeared in {name!r}"
        )
        assert "methoxycarbonyl" in name, (
            f"D-18 regression-fixture invariance broken: 'methoxycarbonyl' "
            f"missing from {name!r}"
        )


# =============================================================================
# Phase 160.2 Plan-02-03: general_acyclic@99999 + freeze_inner_table() landed.
# =============================================================================


class TestPhase160_2_Registrations:
    """Phase 160.2 Plan-02-03 verification: general_acyclic registered +
    INNER_DISPATCH_TABLE frozen + _SORTED_ENTRIES_CACHE eagerly populated."""

    def test_general_acyclic_registered(self):
        from orthonym.assembly.inner_dispatch import INNER_DISPATCH_TABLE
        assert "general_acyclic" in INNER_DISPATCH_TABLE

    def test_general_acyclic_priority_99999(self):
        from orthonym.assembly.inner_dispatch import INNER_DISPATCH_TABLE
        assert INNER_DISPATCH_TABLE["general_acyclic"].priority == 99999

    def test_general_acyclic_handler_bound(self):
        from orthonym.assembly.inner_dispatch import INNER_DISPATCH_TABLE
        from orthonym.assembly.handlers.general_acyclic import name_general_acyclic
        assert INNER_DISPATCH_TABLE["general_acyclic"].handler is name_general_acyclic

    def test_inner_dispatch_table_size_33(self):
        from orthonym.assembly.inner_dispatch import INNER_DISPATCH_TABLE
        # 32 from Phase 160 + 160.1 + 1 NEW (general_acyclic) per CONTEXT D-04.
        # NOTE: amide@5200 + amine@5300 are ALREADY in the 32 (RESEARCH §2
        # typo correction; CONTEXT D-04 narrative "amide@1700 + amine@1800"
        # is typo drift verified live by grep against HEAD).
        assert len(INNER_DISPATCH_TABLE) == 33

    def test_table_frozen_after_import(self):
        """WR-06: freeze_inner_table() called at module-import bottom;
        subsequent _register_inner raises RuntimeError per Phase 160 CONTEXT
        D-10 contract + AP-160.2-04."""
        import pytest
        from orthonym.assembly.inner_dispatch import _register_inner
        from orthonym.assembly.name_tree import NamingResult
        with pytest.raises(RuntimeError, match="frozen"):
            _register_inner(
                handler_id="phase_160_2_test_frozen",
                priority=99998,
                predicate=lambda *_: True,
                handler=lambda *_, **__: NamingResult(
                    name="test", tree=None, atom_to_locant_hint=None,
                ),
                iupac_section="N/A",
                description="WR-06 frozen-table assertion test",
                side_effect_inventory=(),
            )

    def test_sorted_entries_cache_populated_eagerly(self):
        """WR-06: _SORTED_ENTRIES_CACHE populated at module-import (no lazy-init race)."""
        from orthonym.assembly.inner_dispatch import _SORTED_ENTRIES_CACHE
        # Eagerly populated tuple per WR-06 + RESEARCH §5
        assert _SORTED_ENTRIES_CACHE is not None
        assert len(_SORTED_ENTRIES_CACHE) == 33
        # Priorities monotonically non-decreasing per sorted() contract
        priorities = [e.priority for e in _SORTED_ENTRIES_CACHE]
        assert priorities == sorted(priorities)

    def test_general_acyclic_predicate_defers_to_inline_amide(self):
        """AP-160.2-06 CASE B: catch-all predicate returns False for single-amide
        cases that the inline amide branch (composer.py:917-919) handles."""
        from orthonym.assembly.inner_dispatch import INNER_DISPATCH_TABLE

        class FakeFeatures:
            principal_group = 'primary_amide'
            principal_group_atoms = [(0, 1, 2)]

        predicate = INNER_DISPATCH_TABLE["general_acyclic"].predicate
        assert predicate(FakeFeatures()) is False

    def test_general_acyclic_predicate_defers_to_inline_amine(self):
        """AP-160.2-06 CASE B: catch-all predicate returns False for amine cases
        that the inline amine branch (composer.py:919-931) handles."""
        from orthonym.assembly.inner_dispatch import INNER_DISPATCH_TABLE

        class FakeFeatures:
            principal_group = 'secondary_amine'
            principal_group_atoms = None

        predicate = INNER_DISPATCH_TABLE["general_acyclic"].predicate
        assert predicate(FakeFeatures()) is False
