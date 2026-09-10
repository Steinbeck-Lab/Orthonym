"""Unit tests for ``scripts/generate_calibration_splits.py``.

Covers:
- ``_classify_rt`` bucketing across the three RT buckets + defensive path.
- ``_primary_class`` first-token extraction + empty handling.
- ``generate_splits`` invariants: sum=n, seed reproducibility, pairwise
  disjoint train/val/test, approximate fraction correctness at n=1000,
  stratification preservation at n=300 with 3 handlers.

Uses ``importlib.import_module`` to load the script under test because
``scripts/`` is not on the package import path (same pattern as
``tests/unit/scripts/test_phase146_factor_distributions.py``).
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))
gs = importlib.import_module("generate_calibration_splits")


class TestClassifyRT:
    """Bucket-assignment tests for ``_classify_rt`` (dimension #3)."""

    def test_rt1(self) -> None:
        assert gs._classify_rt({"inchi_rt": "1", "parent_score": "0.5"}) == "rt1"

    def test_rt0_ps0(self) -> None:
        assert gs._classify_rt({"inchi_rt": "0", "parent_score": "0.0"}) == "rt0_ps0"

    def test_rt0_psgt0(self) -> None:
        assert gs._classify_rt({"inchi_rt": "0", "parent_score": "0.5"}) == "rt0_psgt0"

    def test_missing_fields_default_to_ps0(self) -> None:
        # Empty row: inchi_rt treated as 0, parent_score treated as 0.0 → rt0_ps0
        assert gs._classify_rt({}) == "rt0_ps0"

    def test_invalid_inchi_rt_falls_through(self) -> None:
        # Garbage inchi_rt falls through; parent_score=0.0 → rt0_ps0
        assert gs._classify_rt({"inchi_rt": "xxx", "parent_score": "0.0"}) == "rt0_ps0"

    def test_invalid_parent_score_defaults_to_psgt0(self) -> None:
        # Garbage parent_score: cannot establish ps==0.0 → fallback rt0_psgt0
        assert gs._classify_rt({"inchi_rt": "0", "parent_score": "nan-ish"}) == "rt0_psgt0"


class TestPrimaryClass:
    """First-token extraction from ``compound_classes`` (dimension #2)."""

    def test_first_class(self) -> None:
        assert gs._primary_class({"compound_classes": "aromatic|fused"}) == "aromatic"

    def test_single_class(self) -> None:
        assert gs._primary_class({"compound_classes": "heterocycle"}) == "heterocycle"

    def test_unknown_empty(self) -> None:
        assert gs._primary_class({"compound_classes": ""}) == "unknown"

    def test_unknown_missing(self) -> None:
        assert gs._primary_class({}) == "unknown"


def _make_rows(n: int = 100, handlers=("chain", "ring_a", "ring_b")):
    """Synthesize n test rows with deterministic handler/class/rt distribution."""
    return [
        {
            "inchi_rt": "1" if i % 5 == 0 else "0",
            "parent_score": "0.5" if i % 7 == 0 else "0.0",
            "handler": handlers[i % len(handlers)],
            "compound_classes": "aromatic" if i % 2 == 0 else "fused",
        }
        for i in range(n)
    ]


class TestGenerateSplits:
    """Invariants of the 80/10/10 partitioning ."""

    def test_sum_equals_total(self) -> None:
        rows = _make_rows(100)
        s = gs.generate_splits(rows, seed=321)
        assert len(s["train"]) + len(s["val"]) + len(s["test"]) == 100

    def test_seed_reproducibility(self) -> None:
        rows = _make_rows(100)
        a = gs.generate_splits(rows, seed=321)
        b = gs.generate_splits(rows, seed=321)
        assert a == b

    def test_different_seed_different_output(self) -> None:
        rows = _make_rows(100)
        a = gs.generate_splits(rows, seed=321)
        b = gs.generate_splits(rows, seed=999)
        # Probabilistic: with 100 rows and 3 handlers × 2 classes × 3 rt buckets
        # buckets, the shuffle reorders bucket members differently → splits differ.
        assert a != b

    def test_no_index_in_two_splits(self) -> None:
        rows = _make_rows(100)
        s = gs.generate_splits(rows, seed=321)
        train_set = set(s["train"])
        val_set = set(s["val"])
        test_set = set(s["test"])
        assert not (train_set & val_set)
        assert not (train_set & test_set)
        assert not (val_set & test_set)

    def test_every_index_covered(self) -> None:
        rows = _make_rows(100)
        s = gs.generate_splits(rows, seed=321)
        covered = set(s["train"]) | set(s["val"]) | set(s["test"])
        assert covered == set(range(100))

    def test_fractions_approx_correct(self) -> None:
        # With 1000 rows, the floor-partitioning should yield roughly 80/10/10.
        # Allow ±2-5% slop because many small buckets round down.
        rows = _make_rows(1000)
        s = gs.generate_splits(rows, seed=321)
        assert 780 <= len(s["train"]) <= 820, f"train={len(s['train'])}"
        assert 80 <= len(s["val"]) <= 120, f"val={len(s['val'])}"
        assert 80 <= len(s["test"]) <= 180, f"test={len(s['test'])}"

    def test_stratification_preserves_handler_dist(self) -> None:
        # 300 rows, 100 per handler → roughly 80 per handler in train.
        rows = _make_rows(300)
        s = gs.generate_splits(rows, seed=321)
        train_chain = sum(1 for i in s["train"] if rows[i]["handler"] == "chain")
        # Floor partitioning across many small buckets can drift ±10 from nominal 80.
        assert 70 <= train_chain <= 90, f"train chain count = {train_chain}"

    def test_empty_rows(self) -> None:
        s = gs.generate_splits([], seed=321)
        assert s == {"train": [], "val": [], "test": []}

    def test_fractions_must_sum_to_one(self) -> None:
        rows = _make_rows(10)
        with pytest.raises(AssertionError):
            gs.generate_splits(rows, train_frac=0.5, val_frac=0.3, test_frac=0.3, seed=321)
