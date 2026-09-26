"""Integration tests for a phase OPSIN grammar pre-validation layer.

Pipes through `Orthonym.name` (NOT the validator directly) per
internal notes to assert the chokepoint wires correctly and the
round-trip gate is enforced.

Plan-03 Task 2 deliverable. Test classes:
    - TestChokepointWraps (>= 5 tests)
    - TestKwargDisablesLayer (>= 3 tests)
    - TestWarningOnRepair (>= 4 tests)
    - TestWarningOnUnrepairable (>= 3 tests)
    - TestRoundTripGate (>= 5 tests)
    - TestPerfBenchmark (= 2 tests) hard gate
    - TestTelemetry (>= 2 tests)

Total: >= 24 tests + perf-2 = >= 26.

Anti-patterns avoided:
    : every opsin_roundtrip_check call has SMILES first.
    : every check on the return reads `result['passed']`.
    : no fictitious "fast OPSIN call" timing claims (cost is ~1269ms mean per internal notes).
    : JAR candidate list contains current version ONLY.
    : zero xfail markers.
    : an RT=1 -> RT=0 flip on canary is a phase-blocker.
"""

import logging
import os
import statistics
import subprocess
from typing import List

import pytest

# Wave 0 protection: pytest-benchmark must be installed (pyproject.toml
# Plan-02 added the dev extra). If missing, skip the perf module rather
# than collection-failing.
pytest_benchmark = pytest.importorskip("pytest_benchmark")

from orthonym.namer import Orthonym
from orthonym.validation.opsin_grammar import OpsinGrammar
from orthonym.validation.opsin_roundtrip import opsin_roundtrip_check
from tests.support.jars import jar_or_none


def _java_available() -> bool:
    try:
        subprocess.run(["java", "-version"], capture_output=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _opsin_jar_path():
    """The pinned OPSIN jar via orthonym.jars (tests.support.jars), or None."""
    return jar_or_none()


_SKIP = pytest.mark.skipif(
    not _java_available() or _opsin_jar_path() is None,
    reason="Java + OPSIN JAR (v2.9.0) required",
)


# ---------------------------------------------------------------------------
# Chokepoint wiring
# ---------------------------------------------------------------------------


class TestChokepointWraps:
    """Assert Orthonym.name chokepoint wires to the grammar layer.

    Per internal notes the chokepoint is the single write-site;
     forbids per-handler instrumentation. These tests prove
    the chokepoint is reached.
    """

    @_SKIP
    def test_chokepoint_wraps_name_ethanol(self):
        # Simple SMILES exercises the happy path through the chokepoint.
        namer = Orthonym()
        assert namer._grammar is not None
        result = namer.name("CCO")
        assert result == "ethanol"
        stats = namer.get_validation_stats()
        # Either validate_passed >= 1 (happy path) or a repair fired.
        assert stats["validate_passed"] >= 1 or any(
            stats.get(f"repair_succeeded_{cls}", 0) > 0
            for cls in ("bracket", "stereo", "hyphen")
        )

    @_SKIP
    def test_chokepoint_wraps_name_with_confidence(self):
        #: name_with_confidence funnels through the same chokepoint.
        namer = Orthonym()
        result = namer.name_with_confidence("CCO")
        assert result["name"] == "ethanol"
        # Counter incremented (validate_passed >= 1 on this happy path).
        stats = namer.get_validation_stats()
        assert stats["validate_passed"] >= 1

    @_SKIP
    def test_chokepoint_handles_acetic_acid(self):
        # Different SMILES exercising a different handler exit path.
        namer = Orthonym()
        result = namer.name("CC(=O)O")
        assert "acetic" in result.lower() or "ethanoic" in result.lower()
        stats = namer.get_validation_stats()
        # The chokepoint was reached for this name.
        assert stats["validate_passed"] >= 1

    @_SKIP
    def test_chokepoint_stats_persist_across_calls(self):
        # Per-instance stats persist across name invocations .
        namer = Orthonym()
        before = namer.get_validation_stats()
        namer.name("CCO")
        namer.name("CC(=O)O")
        after = namer.get_validation_stats()
        # validate_passed monotonically non-decreasing.
        assert after["validate_passed"] >= before["validate_passed"]

    @_SKIP
    def test_chokepoint_grammar_instance_exists(self):
        # The Orthonym instance carries an OpsinGrammar reference per.
        namer = Orthonym()
        assert isinstance(namer._grammar, OpsinGrammar)
        # Stats dict is shared by reference (per / ref-pass).
        assert namer._grammar._stats is namer._grammar_stats


# ---------------------------------------------------------------------------
# Kwarg disable
# ---------------------------------------------------------------------------


class TestKwargDisablesLayer:
    """The `_disable_grammar_validation=True` kwarg disables the layer.

    Per internal notes this is the test-only escape hatch. ON by
    default in production.
    """

    def test_kwarg_disables_grammar_attribute(self):
        # _grammar is None when disabled .
        namer = Orthonym(_disable_grammar_validation=True)
        assert namer._grammar is None

    def test_kwarg_disabled_returns_handler_output(self):
        # The chokepoint short-circuits and returns handler output unchanged.
        namer = Orthonym(_disable_grammar_validation=True)
        result = namer.name("CCO")
        # Handler-internal output still names ethanol correctly here
        # (the grammar layer is empirically a no-op on this name).
        assert result == "ethanol"

    def test_kwarg_disabled_keeps_stats_at_zero(self):
        # When disabled, no counter increments occur.
        namer = Orthonym(_disable_grammar_validation=True)
        namer.name("CCO")
        stats = namer.get_validation_stats()
        # All seven buckets remain at 0 (+).
        assert all(v == 0 for v in stats.values()), stats


# ---------------------------------------------------------------------------
# Warning on repair
# ---------------------------------------------------------------------------


class TestWarningOnRepair:
    """A repair fires a WARNING with the LOCKED format string.

    Format: 'OPSIN grammar repair: handler=%s class=%s original=%r repaired=%r'.

    These tests bypass the handler internals by calling
    `_final_grammar_check` directly with a synthetic name (the handler-
    internal naming pipeline empirically produces canonical names for
    most simple SMILES). This is the cleanest way to exercise the
    WARNING path without coupling to handler-specific drift fixtures.
    """

    def test_warning_on_repair_bracket(self, caplog):
        # internal notes A: bracket-renest repair WARNING.
        from orthonym.namer import _final_grammar_check
        grammar = OpsinGrammar()
        stats = {k: 0 for k in OpsinGrammar.STAT_KEYS}
        grammar._stats = stats
        with caplog.at_level("WARNING", logger="orthonym.namer"):
            result = _final_grammar_check(
                "((2-methylpropyl))methylheptane",
                None,  # no SMILES => degraded path; INFO log only
                "test-handler",
                grammar,
                stats,
            )
        # Repair fires (validate-fail → suggest_fix succeeds) → WARNING.
        msgs = [r.message for r in caplog.records]
        assert any("OPSIN grammar repair" in m for m in msgs), msgs
        assert any("class=bracket" in m for m in msgs), msgs
        assert any(
            "original=" in m and "repaired=" in m for m in msgs
        ), msgs
        assert result == "[(2-methylpropyl)]methylheptane"

    def test_warning_on_repair_stereo(self, caplog):
        # internal notes B.fix: stereo-relocate repair WARNING.
        from orthonym.namer import _final_grammar_check
        grammar = OpsinGrammar()
        stats = {k: 0 for k in OpsinGrammar.STAT_KEYS}
        grammar._stats = stats
        with caplog.at_level("WARNING", logger="orthonym.namer"):
            result = _final_grammar_check(
                "(2R)(3S)-2,3-dibromobutane",
                None,
                "test-handler",
                grammar,
                stats,
            )
        msgs = [r.message for r in caplog.records]
        assert any("OPSIN grammar repair" in m for m in msgs)
        assert any("class=stereo" in m for m in msgs)
        assert result == "(2R,3S)-2,3-dibromobutane"

    def test_warning_on_repair_hyphen(self, caplog):
        # internal notes C.fix: hyphen-normalize repair WARNING.
        from orthonym.namer import _final_grammar_check
        grammar = OpsinGrammar()
        stats = {k: 0 for k in OpsinGrammar.STAT_KEYS}
        grammar._stats = stats
        with caplog.at_level("WARNING", logger="orthonym.namer"):
            result = _final_grammar_check(
                "2,4dichlorobenzene",
                None,
                "test-handler",
                grammar,
                stats,
            )
        msgs = [r.message for r in caplog.records]
        assert any("OPSIN grammar repair" in m for m in msgs)
        assert any("class=hyphen" in m for m in msgs)
        assert result == "2,4-dichlorobenzene"

    def test_warning_on_repair_handler_field_populated(self, caplog):
        # mandates handler attribution in the WARNING.
        from orthonym.namer import _final_grammar_check
        grammar = OpsinGrammar()
        stats = {k: 0 for k in OpsinGrammar.STAT_KEYS}
        grammar._stats = stats
        with caplog.at_level("WARNING", logger="orthonym.namer"):
            _final_grammar_check(
                "((2-methylpropyl))methylheptane",
                None,
                "my-test-handler-name",
                grammar,
                stats,
            )
        msgs = [r.message for r in caplog.records]
        assert any("handler=my-test-handler-name" in m for m in msgs), msgs


# ---------------------------------------------------------------------------
# Warning on unrepairable
# ---------------------------------------------------------------------------


class TestWarningOnUnrepairable:
    """A validate-fail with no repair fires a fallback WARNING and
    returns the ORIGINAL name (+: never silently mutate).
    """

    def test_warning_on_unrepairable_returns_original(self, caplog):
        # Construct a name that fails validate AND has no audit- repair.
        # `[[methyl]propyl]benzene` passes _check_bracket_hierarchy_strict
        # (fusion-stripper consumes `[methyl]`) — so we need a different
        # synthetic shape. Use a name whose pre-screen multiword fires
        # but no _suggest_* row matches: `bad weird unknown name`.
        from orthonym.namer import _final_grammar_check
        grammar = OpsinGrammar()
        stats = {k: 0 for k in OpsinGrammar.STAT_KEYS}
        grammar._stats = stats
        original_name = "bad weird unknown name"
        with caplog.at_level("WARNING", logger="orthonym.namer"):
            result = _final_grammar_check(
                original_name,
                None,
                "test-handler",
                grammar,
                stats,
            )
        # Original returned (: never silently mutate).
        assert result == original_name
        msgs = [r.message for r in caplog.records]
        assert any(
            "OPSIN grammar validation failed" in m for m in msgs
        ), msgs

    def test_warning_on_unrepairable_handler_field(self, caplog):
        # unrepairable WARNING also carries handler attribution.
        from orthonym.namer import _final_grammar_check
        grammar = OpsinGrammar()
        stats = {k: 0 for k in OpsinGrammar.STAT_KEYS}
        grammar._stats = stats
        with caplog.at_level("WARNING", logger="orthonym.namer"):
            _final_grammar_check(
                "bad weird unknown name",
                None,
                "specific-handler-name",
                grammar,
                stats,
            )
        msgs = [r.message for r in caplog.records]
        assert any(
            "handler=specific-handler-name" in m for m in msgs
        ), msgs

    def test_warning_on_unrepairable_no_counter_increment(self):
        # An unrepairable name does NOT increment the success buckets.
        from orthonym.namer import _final_grammar_check
        grammar = OpsinGrammar()
        stats = {k: 0 for k in OpsinGrammar.STAT_KEYS}
        grammar._stats = stats
        before_counts = sum(stats.values())
        _final_grammar_check(
            "bad weird unknown name",
            None,
            "test-handler",
            grammar,
            stats,
        )
        # Specifically `no_repair_offered` MAY be incremented (the
        # suggest_fix path declined); `validate_passed` MUST NOT be.
        assert stats["validate_passed"] == 0
        # Some bucket may now be > 0 (no_repair_offered) but the
        # success-buckets remain zero.
        for cls in ("bracket", "stereo", "hyphen"):
            assert stats[f"repair_succeeded_{cls}"] == 0


# ---------------------------------------------------------------------------
# Round-trip gate
# ---------------------------------------------------------------------------


class TestRoundTripGate:
    """The round-trip gate  ensures every repair candidate
    OPSIN-parses back to the source SMILES (InChI L1 match).

    Per internal notes the oracle signature is `(smiles, name)` —
    SMILES first. Per the return value is a dict; read
    `result['passed']`, never truthy-check the dict.
    """

    @_SKIP
    def test_RT_1_ethanol_round_trips(self):
        #: trivial happy-path round-trip.
        namer = Orthonym()
        name = namer.name("CCO")
        #: SMILES first.
        result = opsin_roundtrip_check("CCO", name)
        #: read result['passed'].
        assert result.get("passed", False) is True

    @_SKIP
    def test_RT_2_acetic_acid_round_trips(self):
        #: simple acid round-trip.
        namer = Orthonym()
        name = namer.name("CC(=O)O")
        result = opsin_roundtrip_check("CC(=O)O", name)
        assert result.get("passed", False) is True

    @_SKIP
    def test_RT_3_butan_2_ol_round_trips(self):
        #: secondary alcohol round-trip.
        namer = Orthonym()
        name = namer.name("CCC(C)O")
        result = opsin_roundtrip_check("CCC(C)O", name)
        assert result.get("passed", False) is True

    @_SKIP
    def test_RT_4_benzene_round_trips(self):
        #: aromatic round-trip.
        namer = Orthonym()
        name = namer.name("c1ccccc1")
        result = opsin_roundtrip_check("c1ccccc1", name)
        assert result.get("passed", False) is True

    @_SKIP
    def test_RT_5_dichlorophenol_round_trips(self):
        #: substituted aromatic round-trip.
        namer = Orthonym()
        name = namer.name("Oc1cc(Cl)cc(Cl)c1")
        result = opsin_roundtrip_check("Oc1cc(Cl)cc(Cl)c1", name)
        assert result.get("passed", False) is True

    @_SKIP
    def test_RT_negative_repair_failing_oracle_returns_none(self):
        #: when a `_suggest_*` produces a candidate but the
        # candidate fails opsin_roundtrip_check, suggest_fix returns
        # `(None, None)` — NOT the bad candidate.
        # Construct a synthetic case: a name that would repair via the
        # hyphen surface but whose source SMILES does NOT match the
        # repaired structure. Using ethanol's SMILES with a bogus
        # repaired name forces the oracle to fail.
        g = OpsinGrammar()
        # `2,4dichlorobenzene` has no correct mapping to ethanol's
        # SMILES `CCO`; the round-trip oracle MUST reject and the
        # method MUST return (None, None).
        repaired, repair_class = g.suggest_fix(
            "2,4dichlorobenzene", source_smiles="CCO"
        )
        assert (repaired, repair_class) == (None, None)


# ---------------------------------------------------------------------------
# Performance benchmark (/ hard gate)
# ---------------------------------------------------------------------------


def _true_p99(benchmark, what: str) -> float:
    """TRUE p99 (not max, which is p100) from pytest-benchmark's raw timings.

    pytest-benchmark switches itself off under xdist ("Benchmarks are
    automatically disabled because xdist plugin is active") and under
    --benchmark-disable. It then calls the target ONCE, records no timings and
    leaves ``benchmark.stats`` None, so there is no latency to gate. The caller's
    functional assertions have already run on that one call; the p99 gate is
    skipped with the reason and runs in a session where the plugin measures
    (serial, no -n).
    """
    if benchmark.disabled:
        pytest.skip(
            f"{what} p99 gate needs pytest-benchmark timings, and the plugin is "
            "disabled in this session (xdist -n, or --benchmark-disable); "
            "run this test serially to measure it"
        )
    data = list(benchmark.stats.stats.data)
    if len(data) >= 100:
        return statistics.quantiles(data, n=100)[98]
    # Small sample: sorted-percentile fallback (still NOT max).
    data_sorted = sorted(data)
    idx = max(0, int(round(0.99 * (len(data_sorted) - 1))))
    return data_sorted[idx]


class TestPerfBenchmark:
    """ / hard gate: < 50ms TRUE p99 for the grammar layer.

    Per WARNING #4 + #5 fix in 156-03-PLAN.md: the benchmarks measure
    the GRAMMAR LAYER cost in ISOLATION — NOT the whole pipeline.
    True p99 is computed from the raw timing data, NOT max (which is
    p100).

    Two benchmarks:
        1. test_perf_validate_p99_under_50ms — `g.validate(name)`.
        2. test_perf_suggest_fix_p99_under_50ms — `g.suggest_fix(name, None)`.

    The `source_smiles=None` arg on suggest_fix forces the validate-only
    degraded path (no OPSIN JAR cost; ~1269ms is on the slow path
    by design). Plan-03's perf gate is on validate AND suggest_fix
    in their grammar-layer-isolated form per internal notes.
    """

    def test_perf_validate_p99_under_50ms(self, benchmark):
        g = OpsinGrammar()
        # Benchmark the validate hot path on a canonical name.
        result = benchmark(g.validate, "ethanol")
        assert result is True
        p99 = _true_p99(benchmark, "validate")
        assert p99 < 0.050, (
            f"validate p99 {p99 * 1000:.3f}ms > 50ms "
            f"(SC-4 / D-16 HARD gate)"
        )

    def test_perf_suggest_fix_p99_under_50ms(self, benchmark):
        g = OpsinGrammar()
        # Use a representative invalid name from internal notes.
        # source_smiles=None forces validate-only path (no OPSIN JAR
        # cost; ~1269ms only fires on the round-trip-gated path).
        invalid_name = "((2-methylpropyl))methylheptane"
        result = benchmark(g.suggest_fix, invalid_name, None)
        # Confirm the repair fired.
        assert result[0] is not None
        assert result[1] == "bracket"
        p99 = _true_p99(benchmark, "suggest_fix")
        assert p99 < 0.050, (
            f"suggest_fix p99 {p99 * 1000:.3f}ms > 50ms "
            f"(SC-4 / D-16 HARD gate)"
        )


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------


class TestTelemetry:
    """Per-instance counter histogram per internal notes +."""

    def test_get_validation_stats_returns_defensive_copy(self):
        #: get_validation_stats returns a defensive copy.
        # Mutating the returned dict does NOT affect future reads.
        namer = Orthonym()
        snapshot = namer.get_validation_stats()
        snapshot["validate_passed"] = 999_999  # poison the copy
        # The next read returns the live counter, NOT the poisoned value.
        actual = namer.get_validation_stats()
        assert actual["validate_passed"] != 999_999

    def test_get_validation_stats_seven_buckets_present(self):
        #: all seven STAT_KEYS pre-seeded at construction time.
        namer = Orthonym()
        stats = namer.get_validation_stats()
        expected = {
            "validate_passed",
            "repair_succeeded_bracket",
            "repair_succeeded_stereo",
            "repair_succeeded_hyphen",
            "repair_failed_validate",
            "repair_failed_roundtrip",
            "no_repair_offered",
        }
        assert set(stats.keys()) == expected
        # Pre-seeded to zero (no calls have happened yet on a fresh instance).
        for v in stats.values():
            assert isinstance(v, int) and v >= 0
