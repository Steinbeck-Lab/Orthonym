"""a phase Plan 01 Task 2 — Unit tests for the pre-flight diagnostic.

Tests the three pure-Python helpers in
``scripts/phase146_factor_distributions.py``:

* ``decide_grid_reduction`` — KEEP/DROP decision logic (rules).
* ``_aggregate_per_handler`` — group-by + quantile fallback for low n.
* ``_compute_multiple_bond_count_normalized`` — parent-only bond density
  with substituent-exclusion semantics (Blue Book.

The script lives in the project ``scripts/`` directory (not on the package
import path), so we add it to ``sys.path`` and use ``importlib`` to load.
"""
import importlib
import sys
from pathlib import Path

import pytest
from rdkit import Chem

# Add scripts/ to sys.path so we can import the script as a module
_SCRIPTS_DIR = Path(__file__).parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
fd = importlib.import_module("phase146_factor_distributions")


# ============================================================================
# TestDecideGridReduction — KEEP/DROP decision logic
# ============================================================================

class TestDecideGridReduction:
    """Validate the KEEP/DROP decision rules per internal notes / RESEARCH

    Rules:
      * KEEP iff at least one handler has n >= MIN_HANDLER_N AND
        std >= THRESHOLD_LOW_VARIANCE.
      * DROP otherwise (insufficient data treated as drop — err on fewer
        grid configs).
    """

    def _row(self, handler, factor, n=200, mean=0.5, std=0.1):
        """Build a summary row in the format produced by _aggregate_per_handler."""
        return {
            "handler": handler,
            "factor": factor,
            "n": n,
            "mean": mean,
            "std": std,
            "p10": mean - std,
            "p50": mean,
            "p90": mean + std,
            "p95": mean + 1.5 * std,
        }

    def test_keep_when_high_variance(self):
        """One handler with std=0.10 (>=0.05) and n=200 (>=100) → KEEP."""
        summary = [
            # The high-variance row that should flip atom_coverage to KEEP
            self._row("chain", "atom_coverage", n=200, std=0.10),
            # Other handlers below threshold — irrelevant given the one above.
            self._row("ring", "atom_coverage", n=150, std=0.02),
            self._row("polyfunctional", "atom_coverage", n=120, std=0.01),
            # Other factors absent — those will be DROP for lack of data.
        ]
        decisions = fd.decide_grid_reduction(summary)
        assert decisions["atom_coverage"] == "KEEP", (
            f"Expected KEEP for atom_coverage when chain has std=0.10; "
            f"got {decisions}"
        )

    def test_drop_when_all_handlers_low_variance(self):
        """All handlers with n>=100 have std<0.05 → DROP for that factor."""
        summary = [
            self._row("chain", "parent_correctness", n=200, std=0.02),
            self._row("ring", "parent_correctness", n=150, std=0.04),
            self._row("benzene", "parent_correctness", n=300, std=0.01),
        ]
        decisions = fd.decide_grid_reduction(summary)
        assert decisions["parent_correctness"] == "DROP", (
            f"Expected DROP for parent_correctness when all handlers have "
            f"std<0.05; got {decisions}"
        )

    def test_drop_when_no_handlers_meet_min_n(self):
        """All handlers have n<MIN_HANDLER_N → DROP (insufficient data)."""
        summary = [
            self._row("chain", "fg_recognition", n=50, std=0.10),
            self._row("ring", "fg_recognition", n=99, std=0.20),
        ]
        decisions = fd.decide_grid_reduction(summary)
        assert decisions["fg_recognition"] == "DROP", (
            f"Expected DROP for fg_recognition when no handler reaches "
            f"MIN_HANDLER_N=100; got {decisions}"
        )


# ============================================================================
# TestAggregatePerHandler — group-by + quantile fallback
# ============================================================================

class TestAggregatePerHandler:
    """Validate _aggregate_per_handler grouping + statistic computation."""

    def test_groups_by_handler_and_factor(self):
        """Two rows for (chain, ratio) → one summary entry with n=2 and
        correct mean/std.
        """
        rows = [
            {"handler": "chain", "ratio": 0.5, "atom_coverage": 0.7},
            {"handler": "chain", "ratio": 0.7, "atom_coverage": 0.9},
        ]
        # 'ratio' is NOT in FACTORS_TO_MEASURE (-a.1) so it's skipped;
        # use atom_coverage which IS measured.
        summary = fd._aggregate_per_handler(rows)
        # Filter to (chain, atom_coverage) entries
        chain_ac = [r for r in summary if r["handler"] == "chain" and r["factor"] == "atom_coverage"]
        assert len(chain_ac) == 1, f"Expected 1 entry for (chain, atom_coverage); got {len(chain_ac)}"
        entry = chain_ac[0]
        assert entry["n"] == 2
        assert entry["mean"] == pytest.approx(0.8, abs=1e-9)
        # Sample std of [0.7, 0.9] = 0.1414...
        assert entry["std"] == pytest.approx(0.14142135623730953, abs=1e-9)

    def test_quantile_fallback_for_small_n(self):
        """n<10 → p10=min, p90=max (no IndexError from quantiles)."""
        rows = [
            {"handler": "chain", "atom_coverage": v}
            for v in [0.1, 0.3, 0.5, 0.7, 0.9]
        ]
        summary = fd._aggregate_per_handler(rows)
        entry = next(
            r for r in summary
            if r["handler"] == "chain" and r["factor"] == "atom_coverage"
        )
        assert entry["n"] == 5
        # n<10 → fallback path returns min/max
        assert entry["p10"] == 0.1
        assert entry["p90"] == 0.9
        # p50 is the median regardless of n
        assert entry["p50"] == 0.5
        # n<20 → p95 falls back to max
        assert entry["p95"] == 0.9

    def test_empty_input(self):
        """Empty input → empty summary (no exception)."""
        assert fd._aggregate_per_handler([]) == []


# ============================================================================
# TestComputeMultipleBondCountNormalized — parent-only bond density
# ============================================================================

class TestComputeMultipleBondCountNormalized:
    """Validate _compute_multiple_bond_count_normalized.

    Blue Book reads "ring system or chain" — i.e., the parent
    skeleton. Substituent multiple bonds (e.g., a nitrile C#N attached to
    a chain) do NOT count when their endpoints are outside the parent
    atom set.
    """

    def test_methane(self):
        """Methane: 1 atom, 0 multiple bonds → 0.0."""
        mol = Chem.MolFromSmiles("C")
        assert mol is not None
        assert fd._compute_multiple_bond_count_normalized(mol, {0}) == 0.0

    def test_propene(self):
        """Propene C=CC with parent={0,1,2}: 1 double bond / 3 atoms ≈ 0.333."""
        mol = Chem.MolFromSmiles("C=CC")
        assert mol is not None
        result = fd._compute_multiple_bond_count_normalized(mol, {0, 1, 2})
        assert result == pytest.approx(1 / 3, abs=1e-9)

    def test_nitrile_substituent_excluded(self):
        """CC#N with parent={0,1} (the chain CC, NOT the nitrile N at idx 2):
        the C#N triple bond has one endpoint (atom 2) OUTSIDE the parent
        set, so it must NOT be counted → 0.0.
        """
        mol = Chem.MolFromSmiles("CC#N")
        assert mol is not None
        # Verify our atom-index assumption
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        assert atoms == ["C", "C", "N"], f"Unexpected atoms: {atoms}"
        # Parent is the 2-carbon chain only; nitrile N is a substituent atom.
        result = fd._compute_multiple_bond_count_normalized(mol, {0, 1})
        assert result == 0.0, (
            f"Triple bond C#N has endpoint outside parent set; "
            f"must not be counted (got {result})"
        )

    def test_none_parent_returns_zero(self):
        """parent_atom_indices=None → 0.0 (defensive guard)."""
        mol = Chem.MolFromSmiles("C=C")
        assert fd._compute_multiple_bond_count_normalized(mol, None) == 0.0
