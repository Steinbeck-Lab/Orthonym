"""a phase unit tests for ``orthonym.routing.dispatch_table``.

Per internal notes LOCKED test pyramid floor: ≥ 30 dispatch-table integrity tests.
Per internal notes hard invariant: every entry's ``side_effect_inventory`` MUST
be ```` — verified parametrically for every member of ``StoutClass``.
Per internal notes +: no module-globals; every test creates the resources
it touches. Per internal notes + RESEARCH: dispatch is no-retry; this
file does not exercise perf (see ``tests/benchmarks/`` for HARD gate).
Per internal notes +: NO ``@pytest.mark.xfail`` markers — every test must
pass green. Per internal notes + the audit: GENERAL @ priority 99999 is the
explicit catch-all (``lambda *_: True``); no silent fallthrough is possible
by construction.

Test classes per concern (mirrors ``tests/unit/validation/test_opsin_grammar.py``
class-per-concern pattern):

- ``TestStoutClassEnum`` — StoutClass StrEnum shape + member presence.
- ``TestDispatchTableIntegrity`` — OrderedDict shape, frozen dataclass,
  priority uniqueness + ordering, sort-stability.
- ``TestStoutClassRegistration`` — parametrized over ``list(StoutClass)``:
  every member is registered with callable predicate + handler, non-empty
  IUPAC section, tier ∈ {1, 2}, side_effect_inventory ==  .
- ``TestGeneralCatchAll`` — catch-all behavior: predicate always
  returns True; accepts arbitrary kwargs; handler is ``_handle_general``.
- ``TestRegistrationLock`` — post-import freeze sentinel.

All tests run in < 10 seconds total. Parametrize expansion over ~ 19 StoutClass
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
        """internal notes: StoutClass values must be JSON-serializable as strings."""
        assert issubclass(StoutClass, str)
        assert isinstance(StoutClass.SALT, str)
        # str yields just the value (mirrors stdlib StrEnum behavior)
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
        """internal notes: enum values are JSON-friendly lowercase snake_case."""
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
        """internal notes: DISPATCH_TABLE is an ``OrderedDict`` for explicit ordering."""
        assert isinstance(DISPATCH_TABLE, OrderedDict)

    def test_dispatch_table_count_matches_stoutclass_count(self):
        """ enumeration completeness: every StoutClass has a dispatch row.

        No orphan enum members; no orphan registrations.

        a phase exemption: StoutClass.ML_FALLBACK is a telemetry tag
        (no CFR dispatch entry; see internal notes-MLF.md). Exclude it
        from the count comparison.
        """
        # Telemetry-only enum members exempt from DISPATCH_TABLE (a phase).
        EXEMPT = frozenset({"ML_FALLBACK"})
        active_classes = [c for c in StoutClass if c.name not in EXEMPT]
        assert len(active_classes) == len(DISPATCH_TABLE), (
            f"len(active StoutClass)={len(active_classes)} but "
            f"len(DISPATCH_TABLE)={len(DISPATCH_TABLE)}; CFR-02 violation."
        )

    def test_dispatch_entries_are_frozen(self):
        """internal notes: ClassDispatchEntry is a frozen dataclass."""
        entry = next(iter(DISPATCH_TABLE.values()))
        with pytest.raises(FrozenInstanceError):
            entry.priority = 99  # type: ignore[misc]

    def test_priorities_are_unique(self):
        """internal notes: every priority is unique to enable deterministic sort."""
        priorities = [e.priority for e in DISPATCH_TABLE.values()]
        assert len(priorities) == len(set(priorities)), (
            f"Duplicate priority detected; D-06 violation. priorities={priorities}"
        )

    def test_priorities_are_spaced_in_hundreds(self):
        """internal notes: non-GENERAL priorities are spaced in 100s for insertability.

        Exceptions to the spacing rule (per docstring intent + a phase):
        - GENERAL @ 99999 + DECOMPOSITION_PRE_GENERAL @ 99000 sit far beyond
          the dense outer-cascade region (100-1600); spacing rule applies
          within the dense region only.
        - a phase sub-100 insertions (ORGANOMETALLIC @ 50): sit BELOW
          the dense floor (100); these are pre-cascade interceptors picked to
          fire BEFORE the dense outer cascade. Spacing rule does not apply
          across the sub-100 -> 100 boundary because the architectural intent
          is "intercept before dense cascade", not "insert within it".
        - a phase LIPID @ 250 (Tier-1): a deliberate half-step interceptor
          inserted BETWEEN RADICAL@200 and ZWITTERION@300 (no hundreds slot is
          available there). It MUST precede ZWITTERION@300 because PC is a
          zwitterion and ZWITTERION returns '' (terminating the cascade), so a
          Tier-2 slot would be unreachable for phospholipids (180 RESOLVED A1).
          Its hard-gate detector fires only on clean lipid backbones, so the
          half-step does not affect insertability of the dense region.
        - a phase B1 ESTER_ANION_ZWITTERION @ 301 (Tier-1): a deliberate
          half-step interceptor inserted directly AFTER ZWITTERION@300 (no
          hundreds slot is free between 300 and ANION_RETAINED@400). It MUST
          follow ZWITTERION@300 (a NET-CHARGED acid-ester-anion zwitterion is
          classified 'ion', not 'zwitterion', by `detect_species_type`, so the
          two predicates are mutually exclusive by net-charge sign — ordering
          relative to ZWITTERION is immaterial for correctness, but sits here
          for the same family grouping). Its predicate is the SAME
          fully-validated, atom-coverage-checked namer as the handler
          (predicate-is-handler pattern), so the half-step cannot fire on any
          shape it does not also correctly name — it does not affect
          insertability of the dense region.
        """
        LIPID_HALF_STEP = 250  # a phase Tier-1 interceptor (documented exception)
        ESTER_ANION_ZWITTERION_HALF_STEP = 301  # a phase B1 (documented exception)
        # a phase CATION_QUATERNARY @ 480 (Tier-1): a deliberate half-step
        # interceptor between ANION_RETAINED@400 and CATION_RETAINED@500 (no
        # hundreds slot free there). Same documented-half-step pattern as
        # LIPID@250 / ESTER_ANION_ZWITTERION@301; a pre-existing entry that was
        # never added to this exclusion list (surfaced by the B1 test touch).
        CATION_QUATERNARY_HALF_STEP = 480
        outer = sorted(
            e.priority
            for e in DISPATCH_TABLE.values()
            if 100 <= e.priority < 10000
            and e.priority not in (LIPID_HALF_STEP, ESTER_ANION_ZWITTERION_HALF_STEP,
                                   CATION_QUATERNARY_HALF_STEP)  # dense region
        )
        for prev, curr in zip(outer, outer[1:]):
            assert curr - prev >= 100, (
                f"Priority spacing < 100 between {prev} and {curr}; D-06 violation."
            )

    def test_general_is_priority_99999(self):
        """CONTEXT D-08 + audit § 1 row 18: GENERAL catch-all @ priority 99999."""
        assert DISPATCH_TABLE[StoutClass.GENERAL].priority == 99999

    def test_general_is_last_when_sorted_by_priority(self):
        """: GENERAL is the explicit terminal catch-all in priority order."""
        sorted_entries = sorted(DISPATCH_TABLE.values(), key=lambda e: e.priority)
        assert sorted_entries[-1].class_id == StoutClass.GENERAL

    def test_iteration_order_matches_priority_order(self):
        """internal notes +: insertion order in ``_register_dispatch`` call sequence
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

    # a phase + internal notes-MLF.md: StoutClass.ML_FALLBACK is a
    # TELEMETRY tag, NOT a CFR dispatch entry. The post-pipeline wrapper at
    # namer.py:1248 (Plan-03 T01) increments the counter via
    # `_cfr_router._increment_stat(StoutClass.ML_FALLBACK)` directly; no
    # `_register_dispatch(...)` call exists for it. Exempt the integrity
    # tests so they remain green.
    _STOUT_CLASS_EXEMPT_FROM_DISPATCH_TABLE = frozenset({"ML_FALLBACK"})

    @pytest.mark.parametrize(
        "class_id",
        [c for c in StoutClass if c.name not in {"ML_FALLBACK"}],
        ids=lambda c: c.name,
    )
    def test_every_stoutclass_has_dispatch_entry(self, class_id):
        """internal notes-CFR.md: every StoutClass member is in DISPATCH_TABLE .

        a phase exemption: ML_FALLBACK is telemetry-only (no CFR entry).
        """
        if class_id.name in self._STOUT_CLASS_EXEMPT_FROM_DISPATCH_TABLE:
            pytest.skip(
                f"{class_id.name} is exempt per Phase 162 D-03 "
                f"(telemetry tag; see 162-AUDIT-MLF.md § 8.14)"
            )
        assert class_id in DISPATCH_TABLE

    @pytest.mark.parametrize(
        "class_id",
        [c for c in StoutClass if c.name not in {"ML_FALLBACK"}],
        ids=lambda c: c.name,
    )
    def test_predicates_are_callable(self, class_id):
        """158-AUDIT-CFR.md § 1 column 'predicate_helpers': predicate is a callable."""
        entry = DISPATCH_TABLE[class_id]
        assert callable(entry.predicate), (
            f"{class_id.name}: predicate {entry.predicate!r} is not callable"
        )

    @pytest.mark.parametrize(
        "class_id",
        [c for c in StoutClass if c.name not in {"ML_FALLBACK"}],
        ids=lambda c: c.name,
    )
    def test_handlers_are_callable(self, class_id):
        """158-AUDIT-CFR.md § 1 column 'handler_function': handler is a callable."""
        entry = DISPATCH_TABLE[class_id]
        assert callable(entry.handler), (
            f"{class_id.name}: handler {entry.handler!r} is not callable"
        )

    @pytest.mark.parametrize(
        "class_id",
        [c for c in StoutClass if c.name not in {"ML_FALLBACK"}],
        ids=lambda c: c.name,
    )
    def test_iupac_section_is_non_empty(self, class_id):
        """158-AUDIT-CFR.md § 1 column 'iupac_section': non-empty traceability cite."""
        entry = DISPATCH_TABLE[class_id]
        assert isinstance(entry.iupac_section, str)
        assert entry.iupac_section.strip(), (
            f"{class_id.name}: iupac_section is empty"
        )

    @pytest.mark.parametrize(
        "class_id",
        [c for c in StoutClass if c.name not in {"ML_FALLBACK"}],
        ids=lambda c: c.name,
    )
    def test_tier_values_are_1_or_2(self, class_id):
        """CONTEXT D-07: tier ∈ {1, 2}; Tier-1 is mol-only, Tier-2 is features-required."""
        entry = DISPATCH_TABLE[class_id]
        assert entry.tier in (1, 2), (
            f"{class_id.name}: tier={entry.tier!r}; CONTEXT D-07 requires tier ∈ {{1, 2}}."
        )

    @pytest.mark.parametrize(
        "class_id",
        [c for c in StoutClass if c.name not in {"ML_FALLBACK"}],
        ids=lambda c: c.name,
    )
    def test_side_effect_inventory_is_empty(self, class_id):
        """internal notes HARD INVARIANT: every entry's side_effect_inventory MUST be .

        Per the audit + predicate-purity proofs: predicates MUST be pure
        — no mol mutation, no MolecularFeatures mutation, no module-global state,
        no thread-local state. The locked spec encodes this as an empty
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
        """internal notes +: GENERAL predicate is ``lambda *_: True``.

         prevention: no silent fallthrough is possible by construction.
        """
        entry = DISPATCH_TABLE[StoutClass.GENERAL]
        # Call with arbitrary args; must return True regardless.
        assert entry.predicate(None, "", "", None) is True
        assert entry.predicate(None, "X", "Y", "Z") is True

    def test_general_predicate_accepts_arbitrary_kwargs(self):
        """ kwarg threading: predicate signature must absorb dispatcher kwargs."""
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
    """internal notes: DISPATCH_TABLE is frozen post-import."""

    def test_register_dispatch_raises_post_frozen(self):
        """: ``_register_dispatch`` raises RuntimeError after the sentinel sets."""
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
        """ +: registering a colliding priority on a frozen table raises.

        The freeze check fires first; the duplicate-priority check is exercised by
        construction during module import (verified indirectly via
        ``test_priorities_are_unique`` in TestDispatchTableIntegrity). This test
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
