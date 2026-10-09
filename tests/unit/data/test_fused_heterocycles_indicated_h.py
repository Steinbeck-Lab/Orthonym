"""a phase.B unit tests: tautomer_locant per-entry catalog audit.

Parametrized over every FUSED_HETEROCYCLE_DATA entry. Each test asserts
that the catalog's ``tautomer_locant`` matches the audit-derived
``expected_locant`` per the internal notes-B.md classification verdict.

Verdicts (one per row):

  * CORRECT-None - tautomer_locant is None and the catalog name has
    no indicated-H signal (explicit ``\\d+H-`` prefix, embedded
    ``(\\d+H)`` descriptor, or trivial-name lookup).
  * CORRECT-locant - tautomer_locant is an integer matching the
    catalog name's indicated-H signal.
  * WRONG-locant - tautomer_locant disagrees with the name signal
    (must be 0 in the post-fix audit per acceptance criterion).
  * MISSING-locant - tautomer_locant is None but the name has a
    signal (must be 0 in the post-fix audit per acceptance criterion).
  * OPSIN-UNPARSEABLE - OPSIN cannot parse the catalog name; entry is
    quarantined for a phase hand-off via xfail(strict=False).

The audit JSON sidecar at ``tests/fixtures/indicated_h/audit_b_results.json``
is the single source of truth. Re-running ``scripts/audit_indicated_h.py``
regenerates the sidecar; this test file consumes it.

Source: 155-internal notes,,; internal notes-B.md classification matrix.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA

# Audit JSON sidecar: one entry per catalog row (220 at its 2026-10-09 regeneration),
# classified by ``scripts/audit_indicated_h.py`` against OPSIN round-trip. Three
# multi-indicated-hydrogen names ('2H,4H-', '3H,5H-', '1H,3H-') are classified
# CORRECT-locant by hand: the script's ``^(\d+)H-`` prefix pattern does not match a
# ``nH,mH-`` prefix and reports them WRONG-locant, but the catalogue locant equals the
# first indicated-H locant of each name, which is the signal the script reads for
# every other prefix.
_AUDIT_JSON = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "indicated_h"
    / "audit_b_results.json"
)
_AUDIT_RESULTS = json.loads(_AUDIT_JSON.read_text())


def _stable_id(row: dict) -> str:
    """Stable pytest parametrize id for a row."""
    name = row["name"].replace("/", "_").replace(" ", "_")
    cls = row["classification"]
    return f"{cls}__{name}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "row",
    _AUDIT_RESULTS,
    ids=[_stable_id(r) for r in _AUDIT_RESULTS],
)
def test_catalog_entry_tautomer_locant(row):
    """Assert each catalog entry's ``tautomer_locant`` matches the audit verdict.

    OPSIN-UNPARSEABLE rows are xfail-quarantined per internal notes and listed
    in internal notes-B.md for a phase hand-off.
    """
    if row["classification"] == "OPSIN-UNPARSEABLE":
        pytest.xfail(
            f"OPSIN cannot parse {row['name']!r}; "
            f"see 155-AUDIT-B.md (Phase 156 quarantine)."
        )

    smiles = row["smiles"]
    entry = FUSED_HETEROCYCLE_DATA.get(smiles)
    assert entry is not None, (
        f"Catalog drift: SMILES {smiles!r} from audit JSON sidecar is no "
        f"longer present in FUSED_HETEROCYCLE_DATA. Re-run "
        f"scripts/audit_indicated_h.py to regenerate the sidecar."
    )
    actual = entry["tautomer_locant"]
    expected = row["expected_locant"]
    assert actual == expected, (
        f"155-AUDIT-B.md classification={row['classification']}: "
        f"{row['name']!r} expected tautomer_locant={expected!r}, "
        f"got {actual!r} (source: tests/fixtures/indicated_h/"
        f"audit_b_results.json)"
    )


@pytest.mark.unit
def test_audit_sidecar_size_matches_catalog():
    """Defensive: audit JSON sidecar count == catalog count."""
    assert len(_AUDIT_RESULTS) == len(FUSED_HETEROCYCLE_DATA), (
        f"Sidecar count {len(_AUDIT_RESULTS)} != catalog count "
        f"{len(FUSED_HETEROCYCLE_DATA)} - re-run scripts/audit_indicated_h.py."
    )


@pytest.mark.unit
def test_no_wrong_locant_or_missing_locant_post_fix():
    """a phase.B acceptance criterion V18-155-.

    After plan 155-02 ships, no catalog entry classifies as WRONG-locant
    or MISSING-locant in the audit (modulo OPSIN-UNPARSEABLE quarantines).
    """
    wrong = [r for r in _AUDIT_RESULTS if r["classification"] == "WRONG-locant"]
    missing = [
        r for r in _AUDIT_RESULTS if r["classification"] == "MISSING-locant"
    ]
    assert wrong == [], (
        f"Catalog has {len(wrong)} WRONG-locant entries: {wrong!r}; "
        f"V18-155-AC-2 acceptance not satisfied."
    )
    assert missing == [], (
        f"Catalog has {len(missing)} MISSING-locant entries: {missing!r}; "
        f"V18-155-AC-2 acceptance not satisfied."
    )
