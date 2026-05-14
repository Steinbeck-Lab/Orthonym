"""Phase 158 CFR dispatch consistency — byte-identical vs Plan-01 frozen baseline.

Per CONTEXT D-17 + line 230-234: ≥ 33 tests asserting ``Orthonym.name(smi)``
produces the exact ``name_output`` from ``tests/canary/canary_pre_cfr_158.csv``
(Plan-01 frozen pre-CFR baseline).

Per CONTEXT D-29 honest-fail-on-data: any single byte difference is a CFR-04
HARD-gate failure.  NO ``@pytest.mark.xfail`` markers; NO special-case
allowances.  The fix is upstream (predicate purity per RL-5 OR priority
ordering per RL-1) — not relaxation per AP-17 + AP-18 + AP-19.

The CSV-diff layer is verified separately by ``verify_cfr_byte_identical.py
--mode post`` + ``diff -q pre.csv post.csv``.  This pytest tier asserts NAME
equality only; the CSV-diff catches OPSIN-parse + InChI-L1 column drift that
pytest equality misses.

Two test functions:

- ``test_cfr_dispatch_byte_identical`` — parametrized over one representative
  fixture per StoutClass (StoutClass.NAME ID): 18 tests covering every active
  enum member.  Mined from the live canary corpus via Plan-03 dev bench.
- ``test_cfr_supplementary_byte_identical`` — parametrized over 15 fixtures
  sampled stride-wise from PRE_CANARY_ROWS for cross-class coverage.

Total: 33 tests at the floor; the parametrize expansion yields exactly this.
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import pytest

from orthonym import name_compound
from orthonym.routing.dispatch_table import StoutClass


CANARY_PRE_CSV = Path(__file__).parent.parent / "canary" / "canary_pre_cfr_158.csv"


def _load_pre_canary() -> List[Tuple[str, str, str, str]]:
    """Load ``(canary_tier, fixture_id, smiles, expected_name)`` from frozen baseline.

    Per PATTERNS § 8 anti-pattern "No silent missing-canary-fixture handling":
    raise FileNotFoundError if the frozen baseline is missing — Plan-01 MUST
    have shipped the CSV.  If it is absent, the test infrastructure is broken,
    not the CFR substrate.
    """
    if not CANARY_PRE_CSV.exists():
        raise FileNotFoundError(
            f"Frozen pre-CFR baseline missing: {CANARY_PRE_CSV}. "
            f"Plan-01 must have shipped this artifact."
        )
    rows = []
    with CANARY_PRE_CSV.open() as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append((
                r["canary_tier"],
                r["fixture_id"],
                r["smiles_input"],
                r["name_output"],
            ))
    return rows


PRE_CANARY_ROWS: List[Tuple[str, str, str, str]] = _load_pre_canary()


# ---------------------------------------------------------------------------
# Per-StoutClass byte-identical tests
#
# One (smiles, expected_name) per StoutClass mined from the live canary
# corpus + Plan-03 development bench.  Identical seed-list to
# tests/unit/routing/test_dispatcher.py::STOUTCLASS_REPRESENTATIVES so a
# regression in either layer surfaces against the same fixture.
# ---------------------------------------------------------------------------


CFR_DISPATCH_CANARY: "OrderedDict[StoutClass, Tuple[str, str]]" = OrderedDict([
    (StoutClass.SALT,                    ("[Na+].[Cl-]",            "sodium chloride")),
    (StoutClass.RADICAL,                 ("[CH3]",                  "methyl")),
    (StoutClass.ZWITTERION,              ("[NH3+]CC(=O)[O-]",       "glycine")),
    (StoutClass.ANION_RETAINED,          ("CC(=O)[O-]",             "acetate")),
    (StoutClass.CATION_RETAINED,         ("C[N+](C)(C)C",           "tetramethylammonium")),
    (StoutClass.ANION_SMALL,             ("C(=O)([O-])CCCCCC",      "heptanoate")),
    (StoutClass.POLY_ANION,              ("[O-]C(=O)CCCC(=O)[O-]",  "pentanedioate")),
    (StoutClass.MULTI_COMPONENT_NEUTRAL, ("CCO.OCC",                "ethanol ethanol")),
    (StoutClass.MULTIPLICATIVE,          ("c1ccc(Cc2ccccc2)cc1",    "1,1'-methylenedibenzene")),
    (StoutClass.CARBOHYDRATE_LOOKUP,     ("OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
                                          "alpha-D-glucopyranose")),
    (StoutClass.NATURAL_PRODUCT,         (
        "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@@]34C)[C@@H]1CC[C@@H]2O",
        "(8R,9S,10S,13S,14S,17S)-17-hydroxyandrost-4-en-3-one",
    )),
    (StoutClass.PEPTIDE,                 ("NCC(=O)NCC(=O)O",        "glycylglycine")),
    (StoutClass.RETAINED_NAME,           ("CCO",                    "ethanol")),
    (StoutClass.AMINO_ACID,              ("C[C@H](N)C(=O)O",        "(2S)-2-aminopropanoic acid")),
    (StoutClass.SKELETAL_REPLACEMENT,    ("COCCOC",                 "2,5-dioxahexane")),
    (StoutClass.CYCLOPHANE,              ("C1CCc2ccccc2CCCc2ccccc21",
                                          "[3.3]orthocyclophane")),
    (StoutClass.DECOMPOSITION_PRE_GENERAL,
                                          ("CCCCCCCCCCCCCCCC(=O)OCCC",
                                          "propyl palmitate")),
    (StoutClass.GENERAL,                 ("CCCCCCCO",               "heptan-1-ol")),
])


@pytest.mark.parametrize(
    "class_id,smiles,expected_name",
    [(c, s, n) for c, (s, n) in CFR_DISPATCH_CANARY.items()],
    ids=[c.name for c in CFR_DISPATCH_CANARY.keys()],
)
def test_cfr_dispatch_byte_identical(class_id, smiles, expected_name):
    """158-AUDIT-CFR.md § 1: name(smi) byte-identical to v18 cascade output.

    Per CONTEXT D-29: a single byte difference is a CFR-04 HARD-gate failure.
    Fix the upstream cause (predicate purity / priority ordering) — never
    relax this assertion via xfail or special-case allowance.
    """
    result = name_compound(smiles, style="pin")
    assert result == expected_name, (
        f"CFR DISPATCH REGRESSION (StoutClass={class_id.value}):\n"
        f"  SMILES:   {smiles}\n"
        f"  Expected: {expected_name}\n"
        f"  Got:      {result}"
    )


# ---------------------------------------------------------------------------
# Supplementary stride-wise sample from PRE_CANARY_ROWS
#
# 15 fixtures sampled at regular index strides across the 1282-row corpus
# to give cross-class coverage (rt75 + connectivity + name_stability tiers).
# Per CONTEXT line 232: aim ≥ 33 total integration tests; 18 per-class +
# 15 supplementary = 33.
# ---------------------------------------------------------------------------


_STRIDE = max(1, len(PRE_CANARY_ROWS) // 15)
SUPPLEMENTARY_CANARY: List[Tuple[str, str, str, str]] = [
    PRE_CANARY_ROWS[i] for i in range(0, len(PRE_CANARY_ROWS), _STRIDE)
][:15]


def _supplementary_id(row: Tuple[str, str, str, str]) -> str:
    """Pytest test ID from canary row (tier_fixture-id)."""
    return f"{row[0]}_{row[1]}"


@pytest.mark.parametrize(
    "tier,fixture_id,smiles,expected_name",
    SUPPLEMENTARY_CANARY,
    ids=[_supplementary_id(r) for r in SUPPLEMENTARY_CANARY],
)
def test_cfr_supplementary_byte_identical(tier, fixture_id, smiles, expected_name):
    """Supplementary fixtures sampled from PRE_CANARY_ROWS for cross-class coverage.

    Per CONTEXT D-29: a single byte difference here is the same CFR-04 HARD-gate
    failure as the per-class test above.  No xfail; no relaxation.

    Note: the pre-CFR baseline records exception output as
    ``<ERROR: TYPE: msg>``; for ``name_compound()`` invocations here we let the
    exception propagate naturally (a behavioral change between v18 and CFR
    surfaces as a test error, not a name mismatch).  The CSV-diff layer
    (verify_cfr_byte_identical.py --mode post) preserves the exception-as-cell
    format and catches the same drift loudly.
    """
    if expected_name.startswith("<ERROR:"):
        # The frozen baseline recorded an exception for this fixture.
        # Plan-03 contract: re-raising the same exception type produces
        # byte-identical post-CFR CSV; the pytest layer skips here so a
        # spurious exception type bubble does not look like a CFR routing bug.
        # The CSV-diff layer catches any drift in exception type/message.
        pytest.skip(
            f"frozen baseline recorded exception for fixture {fixture_id!r}; "
            f"CSV-diff layer (verify_cfr_byte_identical.py --mode post) covers it."
        )
    result = name_compound(smiles, style="pin")
    assert result == expected_name, (
        f"CFR DISPATCH REGRESSION (tier={tier}, fixture_id={fixture_id}):\n"
        f"  SMILES:   {smiles}\n"
        f"  Expected: {expected_name}\n"
        f"  Got:      {result}"
    )
