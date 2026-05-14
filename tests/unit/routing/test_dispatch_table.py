"""Phase 158 unit tests for ``orthonym.routing.dispatch_table``.

Per CONTEXT D-17 LOCKED test pyramid floor: ≥ 30 dispatch-table integrity tests.
Per CONTEXT D-26 hard invariant: every entry's ``side_effect_inventory`` MUST
be ``()`` — verified parametrically for every member of ``StoutClass``.
Per CONTEXT D-27 + AP-6: no module-globals; every test creates the resources
it touches.  Per CONTEXT D-15 + RESEARCH § 5.2: dispatch is no-retry; this
file does not exercise perf (see ``tests/benchmarks/`` for D-15 HARD gate).
Per CONTEXT D-29 + AP-17: NO ``@pytest.mark.xfail`` markers — every test must
pass green.  Per CONTEXT D-08 + audit § 1: GENERAL @ priority 99999 is the
explicit catch-all (``lambda *_: True``); no silent fallthrough is possible
by construction.

Test classes per concern (mirrors ``tests/unit/validation/test_opsin_grammar.py``
class-per-concern pattern):

- ``TestStoutClassEnum`` — StoutClass StrEnum shape + member presence.
- ``TestDispatchTableIntegrity`` — OrderedDict shape, frozen dataclass,
  priority uniqueness + ordering, sort-stability.
- ``TestStoutClassRegistration`` — parametrized over ``list(StoutClass)``:
  every member is registered with callable predicate + handler, non-empty
  IUPAC section, tier ∈ {1, 2}, side_effect_inventory == () (D-26).
- ``TestGeneralCatchAll`` — D-08 catch-all behavior: predicate always
  returns True; accepts arbitrary kwargs; handler is ``_handle_general``.
- ``TestRegistrationLock`` — D-05 post-import freeze sentinel.

All tests run in < 10 seconds total.  Parametrize expansion over ~ 19 StoutClass
members yields ~ 6 × 19 = ~ 114 generated tests in TestStoutClassRegistration,
putting the file well above the ≥ 30 floor.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import FrozenInstanceError

import pytest

from orthonym.routing.dispatch_table import (
    DISPATCH_TABLE,
    ClassDispatchEntry,
    StoutClass,
    _register_dispatch,
)


# ---------------------------------------------------------------------------
# Class 1 — StoutClass enum shape (~ 4 tests)
# ---------------------------------------------------------------------------


class TestStoutClassEnum:
    """158-AUDIT-CFR.md § 1 + CONTEXT D-11: StoutClass StrEnum shape."""

    def test_stoutclass_is_str_enum(self):
        """CONTEXT D-11: StoutClass values must be JSON-serializable as strings."""
        assert issubclass(StoutClass, str)
        assert isinstance(StoutClass.SALT, str)
        # str() yields just the value (mirrors stdlib StrEnum behavior)
        assert str(StoutClass.GENERAL) == "general"

    def test_stoutclass_has_general_member(self):
        """CONTEXT D-08 + audit § 1 row 18: GENERAL is the catch-all."""
        assert hasattr(StoutClass, "GENERAL")
        assert StoutClass.GENERAL.value == "general"

    def test_stoutclass_has_decomposition_pre_general(self):
        """158-AUDIT-CFR.md § 1 row 17: DECOMPOSITION_PRE_GENERAL exists."""
        assert hasattr(StoutClass, "DECOMPOSITION_PRE_GENERAL")
        assert StoutClass.DECOMPOSITION_PRE_GENERAL.value == "decomposition_pre_general"

    def test_stoutclass_values_are_lowercase_snakecase(self):
        """CONTEXT D-11: enum values are JSON-friendly lowercase snake_case."""
        for c in StoutClass:
            value = c.value
            assert value == value.lower(), f"{c.name} value {value!r} is not lowercase"
            assert " " not in value, f"{c.name} value {value!r} contains spaces"
            # Snake_case: only [a-z0-9_]
            assert all(ch.islower() or ch.isdigit() or ch == "_" for ch in value), (
                f"{c.name} value {value!r} is not snake_case"
            )


# ---------------------------------------------------------------------------
# Class 2 — DISPATCH_TABLE structural integrity (~ 8 tests)
# ---------------------------------------------------------------------------


class TestDispatchTableIntegrity:
    """158-AUDIT-CFR.md § 1 + CONTEXT D-05 + D-06: DISPATCH_TABLE structure."""

    def test_dispatch_table_is_orderdict(self):
        """CONTEXT D-05: DISPATCH_TABLE is an ``OrderedDict`` for explicit ordering."""
        assert isinstance(DISPATCH_TABLE, OrderedDict)

    def test_dispatch_table_count_matches_stoutclass_count(self):
        """CFR-02 enumeration completeness: every StoutClass has a dispatch row.

        No orphan enum members; no orphan registrations.
        """
        assert len(list(StoutClass)) == len(DISPATCH_TABLE), (
            f"len(StoutClass)={len(list(StoutClass))} but "
            f"len(DISPATCH_TABLE)={len(DISPATCH_TABLE)}; CFR-02 violation."
        )

    def test_dispatch_entries_are_frozen(self):
        """CONTEXT D-05: ClassDispatchEntry is a frozen dataclass."""
        entry = next(iter(DISPATCH_TABLE.values()))
        with pytest.raises(FrozenInstanceError):
            entry.priority = 99  # type: ignore[misc]

    def test_priorities_are_unique(self):
        """CONTEXT D-06: every priority is unique to enable deterministic sort."""
        priorities = [e.priority for e in DISPATCH_TABLE.values()]
        assert len(priorities) == len(set(priorities)), (
            f"Duplicate priority detected; D-06 violation. priorities={priorities}"
        )

    def test_priorities_are_spaced_in_hundreds(self):
        """CONTEXT D-06: non-GENERAL priorities are spaced in 100s for v19 insertability.

        Exception: GENERAL @ 99999 + DECOMPOSITION_PRE_GENERAL @ 99000 sit far
        beyond the dense outer-cascade region (100-1600); spacing rule applies
        within the dense region only.
        """
        outer = sorted(
            e.priority for e in DISPATCH_TABLE.values() if e.priority < 10000
        )
        for prev, curr in zip(outer, outer[1:]):
            assert curr - prev >= 100, (
                f"Priority spacing < 100 between {prev} and {curr}; D-06 violation."
            )

    def test_general_is_priority_99999(self):
        """CONTEXT D-08 + audit § 1 row 18: GENERAL catch-all @ priority 99999."""
        assert DISPATCH_TABLE[StoutClass.GENERAL].priority == 99999

    def test_general_is_last_when_sorted_by_priority(self):
        """CFR-02: GENERAL is the explicit terminal catch-all in priority order."""
        sorted_entries = sorted(DISPATCH_TABLE.values(), key=lambda e: e.priority)
        assert sorted_entries[-1].class_id == StoutClass.GENERAL

    def test_iteration_order_matches_priority_order(self):
        """CONTEXT D-05 + D-06: insertion order in ``_register_dispatch`` call sequence
        is priority-ascending — verifiable by comparing iteration order to sorted order.
        """
        iter_priorities = [e.priority for e in DISPATCH_TABLE.values()]
        assert iter_priorities == sorted(iter_priorities), (
            f"DISPATCH_TABLE iteration order {iter_priorities} is not "
            f"priority-ascending; D-05 + D-06 violation."
        )


# ---------------------------------------------------------------------------
# Class 3 — Per-StoutClass registration integrity (parametrized)
#
# Each parametrize generates one test per StoutClass member, so the 6 methods
# below expand to ~ 6 × 19 = ~ 114 generated tests (well above the ≥ 30 floor).
# ---------------------------------------------------------------------------


class TestStoutClassRegistration:
    """158-AUDIT-CFR.md § 1: every StoutClass row registered with valid contents."""

    @pytest.mark.parametrize("class_id", list(StoutClass), ids=lambda c: c.name)
    def test_every_stoutclass_has_dispatch_entry(self, class_id):
        """158-AUDIT-CFR.md § 1: every StoutClass member is in DISPATCH_TABLE (CFR-02)."""
        assert class_id in DISPATCH_TABLE

    @pytest.mark.parametrize("class_id", list(StoutClass), ids=lambda c: c.name)
    def test_predicates_are_callable(self, class_id):
        """158-AUDIT-CFR.md § 1 column 'predicate_helpers': predicate is a callable."""
        entry = DISPATCH_TABLE[class_id]
        assert callable(entry.predicate), (
            f"{class_id.name}: predicate {entry.predicate!r} is not callable"
        )

    @pytest.mark.parametrize("class_id", list(StoutClass), ids=lambda c: c.name)
    def test_handlers_are_callable(self, class_id):
        """158-AUDIT-CFR.md § 1 column 'handler_function': handler is a callable."""
        entry = DISPATCH_TABLE[class_id]
        assert callable(entry.handler), (
            f"{class_id.name}: handler {entry.handler!r} is not callable"
        )

    @pytest.mark.parametrize("class_id", list(StoutClass), ids=lambda c: c.name)
    def test_iupac_section_is_non_empty(self, class_id):
        """158-AUDIT-CFR.md § 1 column 'iupac_section': non-empty traceability cite."""
        entry = DISPATCH_TABLE[class_id]
        assert isinstance(entry.iupac_section, str)
        assert entry.iupac_section.strip(), (
            f"{class_id.name}: iupac_section is empty"
        )

    @pytest.mark.parametrize("class_id", list(StoutClass), ids=lambda c: c.name)
    def test_tier_values_are_1_or_2(self, class_id):
        """CONTEXT D-07: tier ∈ {1, 2}; Tier-1 is mol-only, Tier-2 is features-required."""
        entry = DISPATCH_TABLE[class_id]
        assert entry.tier in (1, 2), (
            f"{class_id.name}: tier={entry.tier!r}; CONTEXT D-07 requires tier ∈ {{1, 2}}."
        )

    @pytest.mark.parametrize("class_id", list(StoutClass), ids=lambda c: c.name)
    def test_side_effect_inventory_is_empty(self, class_id):
        """CONTEXT D-26 HARD INVARIANT: every entry's side_effect_inventory MUST be ().

        Per audit § 4 AP-21 + § 2 predicate-purity proofs: predicates MUST be pure
        — no mol mutation, no MolecularFeatures mutation, no module-global state,
        no thread-local state.  The locked spec encodes this as an empty
        ``side_effect_inventory`` tuple per row.
        """
        entry = DISPATCH_TABLE[class_id]
        assert entry.side_effect_inventory == (), (
            f"{class_id.name}: side_effect_inventory={entry.side_effect_inventory!r}; "
            f"CONTEXT D-26 hard invariant violation. Every predicate factory MUST "
            f"be pure (no mol/features/module-global mutation per AP-21)."
        )


# ---------------------------------------------------------------------------
# Class 4 — GENERAL catch-all (~ 3 tests)
# ---------------------------------------------------------------------------


class TestGeneralCatchAll:
    """CONTEXT D-08 + audit § 1 row 18: GENERAL is the explicit catch-all."""

    def test_general_predicate_always_true(self):
        """CONTEXT D-08 + CFR-02: GENERAL predicate is ``lambda *_: True``.

        AP-1 prevention: no silent fallthrough is possible by construction.
        """
        entry = DISPATCH_TABLE[StoutClass.GENERAL]
        # Call with arbitrary args; must return True regardless.
        assert entry.predicate(None, "", "", None) is True
        assert entry.predicate(None, "X", "Y", "Z") is True

    def test_general_predicate_accepts_arbitrary_kwargs(self):
        """RL-7 kwarg threading: predicate signature must absorb dispatcher kwargs."""
        entry = DISPATCH_TABLE[StoutClass.GENERAL]
        # _skip_decomposition + _style are the two kwargs the dispatcher threads.
        assert entry.predicate(
            None, "", "", None, _skip_decomposition=True, _style="pin"
        ) is True
        assert entry.predicate(
            None, "", "", None, _skip_decomposition=False, _style="systematic"
        ) is True

    def test_general_handler_is_handle_general(self):
        """158-AUDIT-CFR.md § 1 row 18 traceability: handler is ``_handle_general``."""
        entry = DISPATCH_TABLE[StoutClass.GENERAL]
        assert entry.handler.__name__ == "_handle_general"


# ---------------------------------------------------------------------------
# Class 5 — Registration freeze sentinel (~ 2 tests)
# ---------------------------------------------------------------------------


class TestRegistrationLock:
    """CONTEXT D-05: DISPATCH_TABLE is frozen post-import."""

    def test_register_dispatch_raises_post_frozen(self):
        """D-05: ``_register_dispatch`` raises RuntimeError after the sentinel sets."""
        with pytest.raises(RuntimeError, match="frozen"):
            _register_dispatch(
                class_id=StoutClass.GENERAL,  # any class_id; freeze check fires first
                priority=42424,
                tier=1,
                predicate=lambda *a, **kw: True,
                handler=lambda *a, **kw: None,
                iupac_section="test",
                description="test",
            )

    def test_register_dispatch_raises_on_duplicate_priority_post_frozen(self):
        """D-05 + D-06: registering a colliding priority on a frozen table raises.

        The freeze check fires first; the duplicate-priority check is exercised by
        construction during module import (verified indirectly via
        ``test_priorities_are_unique`` in TestDispatchTableIntegrity).  This test
        confirms the registration helper still refuses post-freeze.
        """
        existing_priority = DISPATCH_TABLE[StoutClass.SALT].priority
        with pytest.raises(RuntimeError):
            _register_dispatch(
                class_id=StoutClass.GENERAL,
                priority=existing_priority,
                tier=1,
                predicate=lambda *a, **kw: True,
                handler=lambda *a, **kw: None,
                iupac_section="test",
                description="test",
            )
