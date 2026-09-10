"""a phase.6 CHOKE-03 instrument self-check.

Asserts the *correctness* invariants of ``scripts/audit_defect_distance.py``
(the cohort *usefulness* judgement for Phase-171 targeting stays MANUAL per
169.6-VALIDATION.md -- this test does not judge whether the cohort is a good
targeting feed, only that the instrument computes its defect vectors honestly):

  1. popcount(defect_vector) == defect_distance for every emitted row.
  2. distance == 0 iff inchi_rt == 1 (the correctness invariant).
  3. The distance == 1 cohort is non-empty AND each entry has exactly one set bit.
  4. A/B/C reconciliation: the unparseable-bit count equals the CSV
     opsin_error-nonempty count over the processed rows; the live taxonomy on
     the FULL file reproduces A==1946, RT==2068 -- the script does NOT hardcode
     the brief's stale 1475.
  5. The JSON carries the heuristic_caveat and the structured_vs_aggregate
     coverage (the coarse/None-tree fallback is reported, not hidden).

Uses ``importlib.import_module`` to load the script under test (the established
``tests/unit/scripts`` pattern). Runs on a small ``--subset`` for speed.

RESEARCH Pitfall 6 / A3: the instrument is a HEURISTIC -- it must NEVER store a
"fix N -> +N RT" claim; this test confirms the heuristic caveat is shipped and
the low-confidence bits (4, 5) are declared.
"""
import csv as _csv
import importlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

add = importlib.import_module("audit_defect_distance")

_CSV = (
    PROJECT_ROOT
    / "benchmarks"
    / "169.5_post_review"
    / "benchmark_multi_corpus_results.csv"
)
# Small subset keeps the per-node OPSIN-gated scoring pass fast in CI while
# still exercising structured + aggregate-fallback rows and (empirically) at
# least one distance==1 compound.
_SUBSET = 120

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# One shared run of the instrument on a subset (records + report + out paths).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def instrument_run(tmp_path_factory):
    if not _CSV.exists():
        pytest.skip(f"benchmark CSV missing: {_CSV}")
    out_dir = tmp_path_factory.mktemp("defect_distance")
    out_json = out_dir / "defect_distance.json"
    out_csv = out_dir / "defect_distance.csv"
    report, json_path, sidecar_path = add.run(
        [
            "--csv", str(_CSV),
            "--subset", str(_SUBSET),
            "--out", str(out_json),
            "--sidecar", str(out_csv),
        ]
    )
    assert json_path == out_json
    assert sidecar_path == out_csv
    loaded = json.loads(out_json.read_text())
    return {
        "report": report,
        "loaded": loaded,
        "json_path": out_json,
        "sidecar_path": out_csv,
    }


def _first_n_rows(n):
    with open(_CSV, encoding="utf-8") as f:
        return list(_csv.DictReader(f))[:n]


# ---------------------------------------------------------------------------
# Pure-helper unit tests (no corpus needed)
# ---------------------------------------------------------------------------
class TestPopcountAndBits:
    def test_popcount_matches_bin_count(self):
        for v in (0, 1, 2, 3, 7, 0b1010101, 0b1111111):
            assert add._popcount(v) == bin(v).count("1")

    def test_set_bits_enumerates_indices(self):
        assert list(add._set_bits(0)) == []
        assert list(add._set_bits(1)) == [0]
        assert list(add._set_bits(0b101)) == [0, 2]
        # all seven defect bits set
        assert list(add._set_bits(0b1111111)) == [0, 1, 2, 3, 4, 5, 6]

    def test_seven_bit_layout(self):
        # The bit layout is exactly seven named defect classes.
        assert set(add._BIT_NAMES) == {0, 1, 2, 3, 4, 5, 6}
        # The two declared low-confidence bits are wrong-substituent-name + suffix.
        assert add._LOW_CONFIDENCE_BITS == {
            add.BIT_WRONG_SUBST_NAME,
            add.BIT_WRONG_SUFFIX,
        }


class TestSuffixHeuristic:
    def test_missing_principal_suffix_flagged(self):
        # reference declares an acid; generated name drops it -> suffix defect.
        assert add.suffix_defect("propylbenzene", "benzoic acid", "")

    def test_matching_suffix_not_flagged(self):
        assert not add.suffix_defect("ethanol", "ethanol", "")

    def test_unsupported_category_flagged(self):
        assert add.suffix_defect("anything", "anything", "not_supported")


# ---------------------------------------------------------------------------
# Live-taxonomy reconciliation on the FULL file (the stale-1475 guard)
# ---------------------------------------------------------------------------
class TestLiveTaxonomy:
    def test_full_file_A_and_RT_are_live_not_stale(self):
        if not _CSV.exists():
            pytest.skip("benchmark CSV missing")
        with open(_CSV, encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        tax = add.compute_taxonomy(rows)
        # The instrument's expected anchors equal the LIVE counts...
        assert add._EXPECT_A == 1946
        assert add._EXPECT_RT == 2068
        assert tax["A"] == 1946
        assert tax["RT"] == 2068
        #... and explicitly NOT the brief's stale A=1475.
        assert tax["A"] != 1475
        # taxonomy closure
        assert tax["A"] + tax["B"] + tax["C"] + tax["RT"] == tax["total"] == 7500

    def test_full_file_invariants_assert_clean(self):
        # assert_invariants on the full file must not raise (full-file branch:
        # exact A/RT/closure). Build minimal RT-flag records to satisfy the
        # popcount/RT branch cheaply.
        if not _CSV.exists():
            pytest.skip("benchmark CSV missing")
        with open(_CSV, encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        full_tax = add.compute_taxonomy(rows)
        # empty record list is fine for the taxonomy branch; subset=None forces
        # the strict A==1946 / RT==2068 / closure assertions.
        add.assert_invariants([], full_tax, subset=None)


# ---------------------------------------------------------------------------
# End-to-end instrument invariants (run on a subset)
# ---------------------------------------------------------------------------
class TestInstrumentInvariants:
    def test_required_json_keys_present(self, instrument_run):
        d = instrument_run["loaded"]
        for key in (
            "histogram",
            "distance_1_cohort",
            "heuristic_caveat",
            "structured_vs_aggregate",
            "per_bit_totals",
            "bit_layout",
        ):
            assert key in d, f"missing JSON key: {key}"

    def test_heuristic_caveat_present_and_no_fix_n_claim(self, instrument_run):
        d = instrument_run["loaded"]
        caveat = d["heuristic_caveat"].lower()
        assert "heuristic" in caveat
        # The caveat NEGATES the fix-N -> +N-RT reading (it must say so explicitly).
        assert "fix n -> +n rt" in caveat
        assert "break" in caveat  # "...interactions break any 'fix N -> +N RT' reading"
        # The anti-pattern phrase appears ONLY inside the heuristic_caveat -- no
        # other field (histogram, cohort, per-bit totals) stores it as a claim.
        other = dict(d)
        other.pop("heuristic_caveat", None)
        assert "+n rt" not in json.dumps(other).lower()
        # low-confidence bits are declared.
        assert "wrong-substituent-name" in d["low_confidence_bits"]
        assert "wrong/missing-suffix" in d["low_confidence_bits"]

    def test_popcount_equals_distance_for_every_row(self, instrument_run):
        # Re-derive the records from the sidecar CSV and check popcount==distance.
        sidecar = instrument_run["sidecar_path"]
        with open(sidecar, encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        assert rows, "sidecar CSV is empty"
        for r in rows:
            vec = int(r["defect_vector"])
            dist = int(r["defect_distance"])
            assert add._popcount(vec) == dist, r

    def test_distance_zero_iff_rt(self, instrument_run):
        # The histogram's distance==0 bucket must equal the RT rows in the
        # processed subset (distance==0 iff inchi_rt==1).
        d = instrument_run["loaded"]
        processed = _first_n_rows(_SUBSET)

        def truthy(v):
            return (v or "").strip().lower() in ("true", "1", "yes")

        rt_rows = sum(1 for r in processed if truthy(r.get("inchi_rt")))
        assert int(d["histogram"]["0"]) == rt_rows
        # And no non-RT row landed at distance 0: every processed row is either
        # RT (dist 0) or has dist >= 1, so dist-0 count == RT count exactly.
        sidecar = instrument_run["sidecar_path"]
        with open(sidecar, encoding="utf-8") as f:
            sc = list(_csv.DictReader(f))
        zero_dist = sum(1 for r in sc if int(r["defect_distance"]) == 0)
        assert zero_dist == rt_rows

    def test_distance_1_cohort_nonempty_and_single_bit(self, instrument_run):
        d = instrument_run["loaded"]
        cohort = d["distance_1_cohort"]
        assert len(cohort) >= 1, "distance==1 cohort is empty on the subset"
        assert d["distance_1_cohort_size"] == len(cohort)
        for entry in cohort:
            # exactly one set bit, and it matches the recorded defect_bit.
            bit = entry["defect_bit"]
            assert 0 <= bit <= 6
            # defect_class matches the bit-name table.
            assert entry["defect_class"] == add._BIT_NAMES[bit]
        # Every distance==1 row in the sidecar has popcount 1.
        sidecar = instrument_run["sidecar_path"]
        with open(sidecar, encoding="utf-8") as f:
            sc = list(_csv.DictReader(f))
        for r in sc:
            if int(r["defect_distance"]) == 1:
                assert add._popcount(int(r["defect_vector"])) == 1

    def test_unparseable_bit_reconciles_to_csv(self, instrument_run):
        # A/B/C reconciliation: the unparseable bit total equals the CSV
        # opsin_error-nonempty count over the SAME processed rows.
        d = instrument_run["loaded"]
        processed = _first_n_rows(_SUBSET)
        csv_err = sum(
            1 for r in processed if (r.get("opsin_error") or "").strip()
        )
        assert d["per_bit_totals"]["unparseable"] == csv_err

    def test_structured_vs_aggregate_reported(self, instrument_run):
        # The coarse/None-tree fallback MUST be reported (not hidden).
        d = instrument_run["loaded"]
        sva = d["structured_vs_aggregate"]
        assert "per_node" in sva and "aggregate_fallback" in sva
        # per_node + aggregate_fallback + rt_skipped == rows processed.
        assert (
            sva["per_node"] + sva["aggregate_fallback"] + sva["rt_rows_skipped"]
            == d["rows_processed"]
        )
        # The aggregate-reason breakdown is present whenever there is fallback.
        if sva["aggregate_fallback"]:
            assert sva["aggregate_reasons"]

    def test_taxonomy_full_file_carried_in_report(self, instrument_run):
        # Even on a subset run the full-file taxonomy (A=1946, RT=2068) is
        # carried, proving the live read (not the stale 1475).
        d = instrument_run["loaded"]
        assert d["taxonomy_full_file"]["A"] == 1946
        assert d["taxonomy_full_file"]["RT"] == 2068
