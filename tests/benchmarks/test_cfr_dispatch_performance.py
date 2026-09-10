"""a phase CFR dispatch performance benchmark (HARD gate).

Per CONTEXT: p99 < 1ms per dispatch call (HARD gate; phase fails if violated).
Per RESEARCH § 5.2 interpretation (b) + Pitfall 8: measure DISPATCH OVERHEAD ONLY
on cheap-path Tier-1 dispatch (retained-name lookup). DO NOT measure end-to-end
``name_compound()`` — that exceeds 1ms regardless of CFR (because
``name_multiplicative`` SMARTS or ``name_natural_product`` scaffold-detection
dominates per RESEARCH § 5.1).

Median target: < 100µs per CONTEXT (informational; not a HARD gate).

The benchmark uses ``pytest-benchmark`` for percentile distribution recording.
Per pytest-benchmark 5.x API: ``benchmark.stats.stats.<key>`` is the verified
percentile accessor (with a graceful fallback for ``benchmark.stats["stats"]``
in case of API variation).
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from orthonym.routing.dispatcher import ClassFirstRouter


@pytest.mark.benchmark(group="cfr_dispatch")
def test_dispatch_p99_lt_1ms_cheap_path(benchmark):
    """ HARD gate: dispatch p99 < 1ms on cheap-path Tier-1 (retained-name).

    Measurement scope per RESEARCH § 5.2 interpretation (b):
    - Pure dispatch overhead = predicate iteration + first-match return.
    - Cheap-path = Tier-1 retained-name (CCO → ethanol; <1µs predicate cost).
    - Excluded: SMARTS-heavy predicates (name_multiplicative,
      name_natural_product) which exceed 1ms regardless of CFR (Pitfall 8);
      those are part of the existing cost amortized in
      scripts/benchmark_performance.py.

    Module-load cost (StoutClass construction + DISPATCH_TABLE registration)
    is amortized by the test fixture; per-call cost is what's measured.

    Per CONTEXT honest-fail-on-data: if p99 exceeds the HARD gate,
    investigate cheap-path predicate cost. NO band-aid relaxation of the
    1ms threshold per AP-17 + AP-18 + AP-19.
    """
    router = ClassFirstRouter()
    # Pre-build the cheap-path mol — Chem.MolFromSmiles cost is OUT of the
    # measurement scope (CFR predicate runs on the prebuilt mol).
    mol = Chem.MolFromSmiles("CCO")
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

    # Run benchmark; pytest-benchmark records percentiles natively.
    # Pass _style="pin" so the RETAINED_NAME predicate is the first match
    # (priority 1300) — matches the production code path for retained names.
    result = benchmark(
        router.dispatch, mol, "CCO", canonical_smiles, None,
        _style="pin",
    )

    # Smoke check: result is a ClassDispatchResult
    assert result.class_id is not None

    # HARD gate (a phase acceptance): p99 < 1ms (1.0e-3 seconds).
    # pytest-benchmark 5.x: benchmark.stats.stats.<percentile> is the API.
    stats_obj = benchmark.stats.stats
    p99_seconds = getattr(stats_obj, "p99", None)
    if p99_seconds is None:
        # Fallback for older/different pytest-benchmark APIs
        try:
            p99_seconds = benchmark.stats["stats"]["p99"]
        except (KeyError, TypeError):
            # Last resort: use max as a conservative p99 proxy (1 call → max ≈ p99)
            p99_seconds = getattr(stats_obj, "max", None)
    assert p99_seconds is not None, (
        f"pytest-benchmark stats API did not expose p99/max; stats={stats_obj!r}"
    )
    assert p99_seconds < 0.001, (
        f"D-15 HARD GATE VIOLATION: dispatch p99 = {p99_seconds * 1000:.3f}ms "
        f"(target < 1.000ms).  Investigate cheap-path predicate cost; "
        f"NO band-aid relaxation per CONTEXT D-29."
    )

    # median target (informational; not a HARD gate): < 100µs.
    median_seconds = getattr(stats_obj, "median", None)
    if median_seconds is None:
        try:
            median_seconds = benchmark.stats["stats"]["median"]
        except (KeyError, TypeError):
            median_seconds = getattr(stats_obj, "mean", None)

    # Print for the verification report (Task 158-03-10 § 3)
    if median_seconds is not None:
        print(
            f"\nCFR dispatch median: {median_seconds * 1e6:.1f}us "
            f"(informational target < 100us)"
        )
    print(
        f"CFR dispatch p99:    {p99_seconds * 1000:.3f}ms "
        f"(HARD gate < 1.000ms) PASS"
    )
