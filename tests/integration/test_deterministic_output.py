"""a phase.2 -b: determinism regression suite.

Runs name_compound(smiles) under 5 different PYTHONHASHSEED values via
subprocess (PYTHONHASHSEED is consumed at Python startup — we cannot
change it inside a running pytest process). Asserts byte-identical
output across all 5 seeds per molecule.

WHY: Commit f32206d3 fixed a set() iteration non-determinism bug in
_descriptive_fallback that forced correcting 3 baseline CSV cells
. a phase's grid search is ~4.7M invocations (3,125 configs
x 500 mol x 3 corpora) — each latent non-det amplifies catastrophically.
This suite surfaces such bugs BEFORE 146 ships.

CORPUS SIZE: EXACTLY 100 test cases (3 pinned multi-metal regression
anchors + 97 stratified samples). See `sample_determinism_corpus()` for
the exact arithmetic: the return slice is
`sampled_unique[:max(0, TOTAL_SAMPLE - len(pinned_smiles))]` so the
total is always `len(pinned_smiles) + (TOTAL_SAMPLE - len(pinned_smiles))
= TOTAL_SAMPLE = 100`. The earlier a phase.2 draft documented "103"
and returned 100 — REVIEWS.md Plan 02 HIGH #1 flagged this off-by-3;
the fix unifies docstring + code + CI comment + must_haves on 100.

SAMPLING (-b.3 + RESEARCH.md Q2):
  97 stratified molecules across (compound-class-tier x size-bucket),
  chosen uniform-within-stratum with random.Random(321). Matches a phase precedent for reproducibility. The stratum-targets table
  (STRATUM_TARGETS) aims for 100 sampled rows; `sample_determinism_corpus`
  then slices to 97 so the three pinned anchors fit inside a 100-slot
  budget without double-counting.

REGRESSION ANCHORS (-b + RESEARCH.md Q3):
  3 multi-metal SMILES from commit ed353408 that the pre-f32206d3 bug
  mis-identified under varying hash seeds. Pinned, never sampled out,
  always occupy the first 3 slots of _CORPUS.

CORPUS-SIZE INVARIANT (REVIEWS.md Plan 02 MEDIUM #5):
  Module-level `assert len(_CORPUS) >= 95` runs at test collection. If
  a stratum is severely under-filled (e.g., a future corpus edit drops
  charged-large below 2 and multiple other strata go thin), the
  assertion fires at collection time rather than silently reducing
  coverage. Below-95 triggers investigation; above-95-and-below-100 is
  a known minor under-fill (log in docstring for traceability).

RUNTIME (RESEARCH.md Q5): ~10-12 min for 500 subprocess invocations at
~1.2s each (Python import overhead dominates). Marked @pytest.mark.slow
so local `pytest -m "not slow"` skips it; CI runs it via the
`determinism_subset` job in.github/workflows/benchmark_subset.yml
(added by Task 2 of this plan).

FIX-METHODOLOGY (CLAUDE.md): if a molecule fails non-determinism, fix
at source (find the set() / dict.keys() / random iteration and make it
deterministic). NEVER exclude the molecule from the sample — that is
exactly the band-aid the a phase grid search would pay for later.
"""

from __future__ import annotations

import csv
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import pytest

# ---------------------------------------------------------------------------
# Constants (-b locked)
# ---------------------------------------------------------------------------

BASELINE_CSV = (
    Path(__file__).parents[2]
    / ".planning"
    / "phases"
    / "145-multi-corpus-benchmark-foundation"
    / "baseline_v17_all_corpora.csv"
)

SAMPLE_SEED = 321  # matches a phase precedent

# -b.2 — five exact PYTHONHASHSEED values
HASH_SEEDS: Tuple[str, ...] = ("0", "1", "42", "1234567", "99999999")

# -b.3 — exactly 100 test cases (pinned + sampled = 100 total).
# If len(PINNED_ANCHORS) ever changes, the sampled count auto-compensates
# via the slice `sampled_unique[:max(0, TOTAL_SAMPLE - len(pinned_smiles))]`.
TOTAL_SAMPLE = 100

# RESEARCH.md Q3 — 3 multi-metal regression anchors (commit ed353408).
#
# MAINTENANCE NOTE (REVIEWS.md Plan 02 LOW #9):
# These expected strings are the verified runtime output of
# `name_compound(smi)` on main @ the commit that landed this file.
# If the output format of name_compound changes intentionally for
# these SMILES (e.g., new suffix on "(not supported)" strings,
# different metal-priority tie-breaker, punctuation change), the
# expected strings below MUST be updated IN THE SAME COMMIT as the
# output-format change. Drift here is a signal, not noise — do not
# update the expected strings to match the new output without also
# verifying the change is intentional (CLAUDE.md root-cause rule).
#
# Each MUST produce its documented "(not supported)" output on all 5 seeds.
PINNED_ANCHORS: List[Tuple[str, str]] = [
    (
        "[Mn+2].[Co+2].[Cu+]",
        "manganese compound (not supported)",
    ),  # PubChem 19425167 — primary anchor (shortest SMILES, smallest blast radius)
    (
        "[Li+].[Li+].CC(C)(C)[NH2+][N-][Si](C)(C)C."
        "CC(C)(C)[NH2+][N-][Si](C)(C)C.C[In](C)C.C[In](C)C",
        "lithium compound (not supported)",
    ),  # PubChem 6394214 — cycled lithium/indium pre-fix
    (
        "[C-]#N.[C-]#N.[C-]#N.[C-]#N.[C-]#N.[N-]=O.[Na+].[Na+].[Na+].[Na+].[Fe+3]",
        "sodium compound (not supported)",
    ),  # PubChem 45469 — cycled sodium/iron pre-fix
]

# Per-cell targets (must sum to 100 — see RESEARCH.md Q2 matrix).
# sample_determinism_corpus() slices the resulting pool to
# `TOTAL_SAMPLE - len(PINNED_ANCHORS) = 97` entries before prepending
# the 3 pinned anchors, so the final _CORPUS always has exactly 100.
STRATUM_TARGETS = {
    ("ring-aromatic", "small"): 10,
    ("ring-aromatic", "medium"): 10,
    ("ring-aromatic", "large"): 5,
    ("fused", "small"): 5,
    ("fused", "medium"): 5,
    ("fused", "large"): 5,
    ("polyfunc", "small"): 10,
    ("polyfunc", "medium"): 10,
    ("charged", "small"): 10,
    ("charged", "medium"): 3,
    ("charged", "large"): 2,
    ("acyclic", "small"): 10,
    ("acyclic", "medium"): 10,
    ("acyclic", "large"): 5,
}
assert sum(STRATUM_TARGETS.values()) == 100, \
    "STRATUM_TARGETS must sum to 100"


# ---------------------------------------------------------------------------
# Stratification helpers
# ---------------------------------------------------------------------------

def _size_bucket(ha: int) -> str:
    return "small" if ha <= 15 else "medium" if ha <= 30 else "large"


def _stratum_key(row: dict) -> Tuple[str, str]:
    classes = set(row.get("compound_classes", "").split(","))
    size = _size_bucket(int(row.get("heavy_atoms") or 0))
    if "salt" in classes or "charged" in classes:
        return ("charged", size)
    if "fused-ring" in classes:
        return ("fused", size)
    if "aromatic" in classes or "heterocycle" in classes:
        return ("ring-aromatic", size)
    if "polyfunctional" in classes:
        return ("polyfunc", size)
    return ("acyclic", size)


def sample_determinism_corpus() -> List[str]:
    """Return EXACTLY 100 SMILES (3 pinned + 97 sampled).

    Deterministic: uses random.Random(SAMPLE_SEED) and sorts the baseline
    CSV rows by corpus_row_id before sampling so two calls return the
    same list.

    Off-by-3 fix (REVIEWS.md Plan 02 HIGH #1): the return slice is
    `sampled_unique[:max(0, TOTAL_SAMPLE - len(pinned_smiles))]` so the
    total is always `len(pinned_smiles) + (TOTAL_SAMPLE - len(pinned_smiles))
    = TOTAL_SAMPLE = 100`. Earlier draft sliced to `[:TOTAL_SAMPLE]`
    which silently capped the total at 100 but undercounted pinned
    contribution by 3.
    """
    if not BASELINE_CSV.exists():
        pytest.skip(
            f"Baseline CSV not found at {BASELINE_CSV}. "
            "This test requires the Phase 145 baseline to be committed."
        )
    rng = random.Random(SAMPLE_SEED)
    with BASELINE_CSV.open() as f:
        # Deviation (Rule 3 — blocking fix): RESEARCH.md Q2 assumed corpus_row_id
        # was an integer but the live baseline CSV stores prefixed string IDs
        # (e.g., "CHEBI:100247", "PubChem 4242"). Casting to int() raises
        # ValueError. Use stable lexicographic sort on the compound key
        # (source_corpus, corpus_row_id) — still deterministic, still reproducible
        # across runs, still independent of filesystem ordering.
        rows = sorted(
            csv.DictReader(f),
            key=lambda r: (
                r.get("source_corpus", ""),
                r.get("corpus_row_id", ""),
            ),
        )
    strata: dict = {}
    for row in rows:
        strata.setdefault(_stratum_key(row), []).append(row["smiles"])
    sampled: List[str] = []
    for key, n in STRATUM_TARGETS.items():
        pool = strata.get(key, [])
        if not pool:
            continue  # under-filled stratum — surfaced by len(_CORPUS) assert
        sampled.extend(rng.sample(pool, min(n, len(pool))))
    # Pinned anchors come FIRST so test IDs sort them to the top of the
    # report (easiest to read when a regression lands).
    pinned_smiles = [s for s, _ in PINNED_ANCHORS]
    # Deduplicate: if any pinned SMILES also appears in the sample, drop
    # the sampled one (keep the pinned version with expected name).
    pinned_set = set(pinned_smiles)
    sampled_unique = [s for s in sampled if s not in pinned_set]
    # SLICE so `pinned + sampled = TOTAL_SAMPLE` exactly, NOT `pinned + 100`.
    # `max(0,...)` guards against TOTAL_SAMPLE < len(pinned_smiles) (unreachable
    # today but keeps the invariant safe under future edits).
    sampled_budget = max(0, TOTAL_SAMPLE - len(pinned_smiles))
    return pinned_smiles + sampled_unique[:sampled_budget]


# ---------------------------------------------------------------------------
# Subprocess harness
# ---------------------------------------------------------------------------

_NAME_SNIPPET = (
    "import sys;"
    "from orthonym.namer import name_compound;"
    "smi = sys.argv[1];"
    "print(name_compound(smi))"
)


def _name_under_seed(smiles: str, seed: str, timeout: float = 60.0) -> str:
    """Invoke name_compound in a fresh subprocess with PYTHONHASHSEED=seed.

    Timeout propagation (REVIEWS.md Plan 02 HIGH #2):
      subprocess.TimeoutExpired is caught and converted to pytest.fail
      with a clear message so a stuck subprocess shows up as a named
      failure in the pytest report rather than a bare RuntimeError.
    """
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = seed
    # Strip PYTHONDONTWRITEBYTECODE/other vars? No — inherit full env so
    # uv-managed venv still works; only override PYTHONHASHSEED.
    try:
        result = subprocess.run(
            [sys.executable, "-c", _NAME_SNIPPET, smiles],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"name_compound timed out after {timeout}s for "
            f"SMILES={smiles!r} seed={seed!r}. This may indicate a "
            f"naming-loop hang or extremely slow fused-ring path — "
            f"investigate at source, do NOT increase the timeout as "
            f"a band-aid (CLAUDE.md root-cause rule)."
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"name_compound subprocess failed for SMILES={smiles!r} "
            f"seed={seed!r}: stderr={result.stderr!r}"
        )
    # strip exactly one trailing newline from print()
    return result.stdout.rstrip("\n")


# ---------------------------------------------------------------------------
# Tests (-b.4 — @pytest.mark.slow)
# ---------------------------------------------------------------------------

# Build the parametrize list ONCE at module load (not on every test).
_CORPUS = sample_determinism_corpus()

# Corpus-size invariant (REVIEWS.md Plan 02 MEDIUM #5).
# Expected EXACTLY 100 in normal operation; the `>= 95` floor is a
# fail-fast guard against silent stratum under-filling rather than an
# upper bound. Running with < 95 test cases is a collection-time error.
assert len(_CORPUS) >= 95, (
    f"test_deterministic_output corpus under-filled: got {len(_CORPUS)} "
    f"entries, expected exactly 100 (3 pinned + 97 sampled) — typical "
    f"floor is >= 95. Investigate stratum under-filling (RESEARCH.md Q2) "
    f"before running this suite. Do NOT lower this floor — fix the "
    f"baseline CSV or STRATUM_TARGETS at source."
)


@pytest.mark.slow
@pytest.mark.parametrize("smiles", _CORPUS, ids=lambda s: s[:40])
def test_determinism_across_hash_seeds(smiles: str) -> None:
    """name_compound(smiles) must produce the same output under all 5 seeds."""
    outputs = [_name_under_seed(smiles, seed) for seed in HASH_SEEDS]
    reference = outputs[0]
    for i, out in enumerate(outputs[1:], start=1):
        assert out == reference, (
            f"Non-determinism detected for SMILES={smiles!r}:\n"
            f"  seed={HASH_SEEDS[0]!r} -> {reference!r}\n"
            f"  seed={HASH_SEEDS[i]!r} -> {out!r}\n"
            f"Fix at source (find the set() / dict iteration and make "
            f"it deterministic). Do NOT exclude this molecule — that is "
            f"the band-aid CLAUDE.md root-cause rule prohibits."
        )


@pytest.mark.slow
@pytest.mark.parametrize("smiles,expected", PINNED_ANCHORS,
                         ids=[s[:40] for s, _ in PINNED_ANCHORS])
def test_multi_metal_regression_anchor(smiles: str, expected: str) -> None:
    """Pinned anchors must produce the exact post-f32206d3 output under all 5 seeds."""
    for seed in HASH_SEEDS:
        got = _name_under_seed(smiles, seed)
        assert got == expected, (
            f"Multi-metal regression anchor broke for SMILES={smiles!r} "
            f"seed={seed!r}: got {got!r}, expected {expected!r}. "
            f"This likely means f32206d3 was reverted or a new non-det "
            f"bug was introduced in the metal-selection path."
        )
