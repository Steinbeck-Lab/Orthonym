"""a phase feature flag tests (,).

Verifies env-var-controlled FACTOR_WEIGHTS dispatch:

- ORTHONYM_USE_V18_WEIGHTS env var, read at module-import time
- Default 'false' → V17 weights active (byte-identical 145.2 baseline)
- 'true' (case-insensitive, whitespace-tolerant) → V18 weights active
- FACTOR_WEIGHTS_V17 has 5 keys (no multiple_bond_count)
- FACTOR_WEIGHTS_V18 has 6 keys, with multiple_bond_count APPENDED LAST
  per IEEE 754 byte-identical invariant

**Test strategy note:** Some tests exercise the env-var parsing by running
a fresh Python subprocess with the env var set (via ``sys.executable``
invocation). Using a subprocess avoids reloading ``coverage_scoring`` in
the same interpreter, which would invalidate the ``CandidateName`` class
identity and cause downstream tests' ``isinstance(...)`` checks to fail
(per Python dataclass semantics, a reloaded class is NOT the same class
object as the originally-imported reference). Tests that only inspect
module-level dict shape don't need subprocess isolation and can be run
in-process.

Source: https://iupac.qmul.ac.uk/BlueBook/P4.html
"""

import subprocess
import sys
import textwrap

import pytest


# ---------------------------------------------------------------------------
# Subprocess helper — runs a short Python snippet under a specific env-var
# setup, returns the stripped stdout. Exit-code 0 is required.
# ---------------------------------------------------------------------------
def _run_with_env(env_overrides: dict, snippet: str) -> str:
    """Run ``snippet`` in a fresh interpreter with env vars overridden.

    Returns stripped stdout. Raises AssertionError on non-zero exit so
    test failures include the subprocess's stderr context.
    """
    import os as _os
    env = _os.environ.copy()
    # Always start from a clean slate for the relevant keys.
    env.pop("ORTHONYM_USE_V18_WEIGHTS", None)
    env.pop("ORTHONYM_SELECTION_MODE", None)
    env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        env=env, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, (
        f"subprocess failed (rc={proc.returncode})\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    return proc.stdout.strip()


class TestEnvVarDefault:
    """D-07: default ORTHONYM_USE_V18_WEIGHTS=false → V17 active."""

    def test_default_is_v17(self):
        """No env var set → FACTOR_WEIGHTS is FACTOR_WEIGHTS_V17."""
        out = _run_with_env({}, """
            import orthonym.assembly.coverage_scoring as cs
            print(cs.FACTOR_WEIGHTS is cs.FACTOR_WEIGHTS_V17)
        """)
        assert out == "True"

    def test_explicit_false_is_v17(self):
        """ORTHONYM_USE_V18_WEIGHTS=false → FACTOR_WEIGHTS is V17."""
        out = _run_with_env({"ORTHONYM_USE_V18_WEIGHTS": "false"}, """
            import orthonym.assembly.coverage_scoring as cs
            print(cs.FACTOR_WEIGHTS is cs.FACTOR_WEIGHTS_V17)
        """)
        assert out == "True"

    def test_explicit_true_is_v18(self):
        """ORTHONYM_USE_V18_WEIGHTS=true → FACTOR_WEIGHTS is V18."""
        out = _run_with_env({"ORTHONYM_USE_V18_WEIGHTS": "true"}, """
            import orthonym.assembly.coverage_scoring as cs
            print(cs.FACTOR_WEIGHTS is cs.FACTOR_WEIGHTS_V18)
        """)
        assert out == "True"

    def test_case_insensitive(self):
        """Uppercase 'TRUE' should still select V18 (case-insensitive)."""
        out = _run_with_env({"ORTHONYM_USE_V18_WEIGHTS": "TRUE"}, """
            import orthonym.assembly.coverage_scoring as cs
            print(cs.FACTOR_WEIGHTS is cs.FACTOR_WEIGHTS_V18)
        """)
        assert out == "True"

    def test_whitespace_tolerant(self):
        """Surrounding whitespace should be stripped by.strip.lower."""
        out = _run_with_env({"ORTHONYM_USE_V18_WEIGHTS": "  true  "}, """
            import orthonym.assembly.coverage_scoring as cs
            print(cs.FACTOR_WEIGHTS is cs.FACTOR_WEIGHTS_V18)
        """)
        assert out == "True"

    def test_nonsense_value_defaults_to_v17(self):
        """Any non-'true' value (e.g. 'yes', '1', 'on') must remain V17."""
        out = _run_with_env({"ORTHONYM_USE_V18_WEIGHTS": "yes"}, """
            import orthonym.assembly.coverage_scoring as cs
            print(cs.FACTOR_WEIGHTS is cs.FACTOR_WEIGHTS_V17)
        """)
        assert out == "True"


class TestV17V18DictShape:
    """ invariants on dict structure — no env-var manipulation needed.

    These tests inspect the module-level dict constants directly; no reload
    required, so they run safely in-process.
    """

    def test_v17_has_5_keys(self):
        """V17 dict must have exactly 5 keys (no multiple_bond_count)."""
        from orthonym.assembly.coverage_scoring import FACTOR_WEIGHTS_V17
        assert len(FACTOR_WEIGHTS_V17) == 5
        expected = {
            'ratio', 'atom_coverage', 'fg_recognition',
            'substituent_completeness', 'parent_correctness',
        }
        assert set(FACTOR_WEIGHTS_V17.keys()) == expected

    def test_v18_has_6_keys(self):
        """V18 dict must have exactly 6 keys (V17 keys + multiple_bond_count)."""
        from orthonym.assembly.coverage_scoring import (
            FACTOR_WEIGHTS_V17, FACTOR_WEIGHTS_V18,
        )
        assert len(FACTOR_WEIGHTS_V18) == 6
        assert set(FACTOR_WEIGHTS_V18.keys()) == (
            set(FACTOR_WEIGHTS_V17.keys()) | {'multiple_bond_count'}
        )

    def test_v18_multiple_bond_count_is_last(self):
        """ IEEE 754 invariant: multiple_bond_count must be APPENDED LAST.

        Python 3.7+ dict iteration is insertion-order-deterministic.
        compute_confidence iterates FACTOR_WEIGHTS keys in insertion order.
        The new factor must be LAST so the prior 5 terms accumulate first —
        byte-identical preservation when V18 weights mimic V17 by setting
        weights for the first 5 keys equal.
        """
        from orthonym.assembly.coverage_scoring import FACTOR_WEIGHTS_V18
        assert list(FACTOR_WEIGHTS_V18.keys())[-1] == 'multiple_bond_count'

    def test_v17_ratio_is_zero(self):
        """a phase -a.1: ratio is permanently demoted to 0.0 in V17."""
        from orthonym.assembly.coverage_scoring import FACTOR_WEIGHTS_V17
        assert FACTOR_WEIGHTS_V17['ratio'] == 0.0

    def test_v18_ratio_is_zero(self):
        """Permanent demotion carries forward to V18 (-a.1)."""
        from orthonym.assembly.coverage_scoring import FACTOR_WEIGHTS_V18
        assert FACTOR_WEIGHTS_V18['ratio'] == 0.0

    def test_v18_parent_correctness_calibrated(self):
        """V18 parent_correctness holds the a phase boundary-extended value.

        History:
          - Plan 03 (a phase): PLACEHOLDER 0.35 shipped initially.
          - Plan 04 (a phase): grid search dropped to 0.0 per
            (ParentCorrectnessScorer short-circuits to 0.5 with no reference).
          - a phase: locked weight at 0.00 pending a phase /.
          - **a phase (2026-04-24, per STATE.md "a phase | PC=0.35
            INTERIOR"):** calibration boundary extended; PC=0.35 chosen as
            an interior-of-grid value; a phase cumulative G3 G2 HARD CLEAN
            preserved.

        Live invariant locked at 0.35 since a phase boundary extension.
        a phase cleanup rebaselines test expectation from the stale 0.0
        (a phase era) to the post-148.2 production value 0.35 — zero
        code change to `coverage_scoring.py`; only the test assertion updated
        to track the shipped weight.
        """
        from orthonym.assembly.coverage_scoring import FACTOR_WEIGHTS_V18
        assert FACTOR_WEIGHTS_V18['parent_correctness'] == 0.35

    def test_v18_multiple_bond_count_placeholder_above_zero(self):
        """Plan 03 placeholder for multiple_bond_count weight is > 0.

        Plan 06 drops the factor entirely if grid search converges to 0 per
        . Until then, the placeholder must be non-zero so the factor
        actually contributes to V18 weighted sums.
        """
        from orthonym.assembly.coverage_scoring import FACTOR_WEIGHTS_V18
        assert FACTOR_WEIGHTS_V18['multiple_bond_count'] > 0.0


class TestRollbackOneLiner:
    """: rollback one-liner must produce sensible output.

    Runs under a subprocess so the V17-mode interpreter state doesn't
    contaminate other tests in the suite.
    """

    def test_v18_off_byte_identical_intent(self):
        """ORTHONYM_USE_V18_WEIGHTS=false + SELECTION_MODE=first_applicable preserves V17.

        Smoke test asserting that with the documented rollback env-var pair
        set, a simple molecule (ethanol) still names correctly. This proves
        the rollback path is functional end-to-end, not just at the dict
        binding layer.
        """
        out = _run_with_env(
            {
                "ORTHONYM_USE_V18_WEIGHTS": "false",
                "ORTHONYM_SELECTION_MODE": "first_applicable",
            },
            """
            from orthonym import name_compound
            print(name_compound("CCO"))
            """,
        )
        assert out == "ethanol"
