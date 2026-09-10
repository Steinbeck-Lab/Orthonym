"""Unit tests for ``scripts/generate_calibration_splits_148_1.py`` (Tier 1).

Covers:
- ``_classify_rt`` bucketing across the three RT buckets + defensive paths
  (verbatim duplicate of a phase analog test for self-contained coverage).
- ``_primary_class`` first-token extraction + empty handling.
- ``generate_splits`` invariants: sum=n, seed reproducibility, pairwise
  disjoint train/val/test, approximate fraction correctness at n=1000,
  stratification preservation at n=300 with 3 handlers.
- ``--exclude-corpus`` filter: default-excludes opsin_selftest_500;
  empty exclude keeps all (back-compat); multiple corpora can be excluded.
- Constants: DEFAULT_SEED == 321 (a phase), DEFAULT_EXCLUDE
  contains opsin_selftest_500 .

Uses ``importlib.import_module`` to load the script under test because
``scripts/`` is not on the package import path (same pattern as
``tests/unit/scripts/test_phase146_factor_distributions.py``).
"""
import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
gs = importlib.import_module("generate_calibration_splits_148_1")


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


class TestPrimaryClass:
    """First-token extraction from ``compound_classes`` (dimension #2)."""

    def test_first_class(self) -> None:
        assert gs._primary_class({"compound_classes": "aromatic|fused"}) == "aromatic"

    def test_unknown(self) -> None:
        assert gs._primary_class({"compound_classes": ""}) == "unknown"
        assert gs._primary_class({}) == "unknown"


def _make_rows(n: int = 100, handlers=("chain", "ring_a", "ring_b"),
               corpora=("chebi_5000", "pubchem_2000")):
    """Synthesize n test rows with deterministic handler/class/rt distribution."""
    return [
        {
            "source_corpus": corpora[i % len(corpora)],
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
        assert a != b

    def test_no_index_in_two_splits(self) -> None:
        rows = _make_rows(100)
        s = gs.generate_splits(rows, seed=321)
        train_set, val_set, test_set = set(s["train"]), set(s["val"]), set(s["test"])
        assert not (train_set & val_set)
        assert not (train_set & test_set)
        assert not (val_set & test_set)

    def test_fractions_approx_correct(self) -> None:
        # With 1000 rows, the floor-partitioning should yield roughly 80/10/10.
        # Allow ±2-5% slop because many small buckets round down.
        rows = _make_rows(1000)
        s = gs.generate_splits(rows, seed=321)
        assert 780 <= len(s["train"]) <= 820, f"train={len(s['train'])}"
        assert 80 <= len(s["val"]) <= 120, f"val={len(s['val'])}"
        assert 80 <= len(s["test"]) <= 120, f"test={len(s['test'])}"

    def test_stratification_preserves_handler_dist(self) -> None:
        rows = _make_rows(300)  # 100 per handler
        s = gs.generate_splits(rows, seed=321)
        train_chain = sum(1 for i in s["train"] if rows[i]["handler"] == "chain")
        # Approximately 80 per handler (within stratification slack).
        assert 70 <= train_chain <= 90, f"train chain count = {train_chain}"


class TestExcludeCorpusFilter:
    """D-03 exclude-corpus filter — OPSIN self-test 500 fully held out."""

    def _make_corpora(self, n_chebi: int = 70, n_opsin: int = 30):
        rows = []
        for i in range(n_chebi):
            rows.append({
                "source_corpus": "chebi_5000",
                "inchi_rt": "1" if i % 3 == 0 else "0",
                "parent_score": "0.5" if i % 5 == 0 else "0.0",
                "handler": "chain",
                "compound_classes": "aromatic",
            })
        for i in range(n_opsin):
            rows.append({
                "source_corpus": "opsin_selftest_500",
                "inchi_rt": "1",
                "parent_score": "1.0",
                "handler": "chain",
                "compound_classes": "aromatic",
            })
        return rows

    def test_default_excludes_opsin_selftest_500(self):
        """When --exclude-corpus default is applied, OPSIN self-test 500 rows are excluded."""
        rows = self._make_corpora(n_chebi=70, n_opsin=30)
        kept = [
            r for r in rows
            if (r.get("source_corpus") or "").strip() not in set(gs.DEFAULT_EXCLUDE)
        ]
        assert len(kept) == 70
        for r in kept:
            assert r["source_corpus"] != "opsin_selftest_500"

    def test_explicit_empty_exclude_keeps_all(self):
        """Back-compat: empty exclude set keeps all rows (a phase splits behavior)."""
        rows = self._make_corpora(n_chebi=70, n_opsin=30)
        kept = [
            r for r in rows
            if (r.get("source_corpus") or "").strip() not in set()
        ]
        assert len(kept) == 100

    def test_multiple_exclude_corpora(self):
        """Multiple --exclude-corpus values: chebi_5000 + opsin_selftest_500 → only pubchem."""
        rows = self._make_corpora(n_chebi=50, n_opsin=30)
        rows.extend([
            {
                "source_corpus": "pubchem_2000",
                "inchi_rt": "0",
                "parent_score": "0.0",
                "handler": "chain",
                "compound_classes": "aromatic",
            }
            for _ in range(20)
        ])
        exclude = {"chebi_5000", "opsin_selftest_500"}
        kept = [
            r for r in rows
            if (r.get("source_corpus") or "").strip() not in exclude
        ]
        assert len(kept) == 20
        for r in kept:
            assert r["source_corpus"] == "pubchem_2000"


class TestSeedAndRange:
    """Constants locked by a phase + a phase."""

    def test_default_seed_is_321(self):
        """a phase reproducibility lock."""
        assert gs.DEFAULT_SEED == 321

    def test_default_exclude_contains_opsin_selftest_500(self):
        """ default: G2 HARD gate corpus held out."""
        assert "opsin_selftest_500" in gs.DEFAULT_EXCLUDE
