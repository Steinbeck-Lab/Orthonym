"""a phase.C unit tests: per-entry seniority catalog audit.

Parametrized over every entry in
``src/orthonym/data/fusion_components.py`` ``MONOCYCLIC_COMPONENTS``;
each entry's ``seniority`` value is asserted to fall within the
 Jan 2022 errata expected tier range emitted by the audit
script ``scripts/audit_benzo_fusion.py``.

Tier ranges (STRICT — see 155-internal notes):

  * 40-49: 6-membered N-heterocycles
  * 50-59: 5-membered N-heterocycles
  * 60-69: 5-membered N+O heterocycles
  * 70-79: O-heterocycles
  * 80-89: S-heterocycles + N+S
  * 200: carbocycles

The fixture file ``tests/fixtures/benzo_fusion/audit_c_results.json`` is
the machine-readable mirror of ``internal notes-C.md``; if the catalog grows
or shifts, regenerate via:

    python scripts/audit_benzo_fusion.py \\
        --out internal notes \\
        --json tests/fixtures/benzo_fusion/audit_c_results.json

Source: 155-internal notes; internal notes-C.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orthonym.data.fusion_components import MONOCYCLIC_COMPONENTS


_AUDIT_TIERS_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "benzo_fusion"
    / "audit_c_results.json"
)
_AUDIT_TIERS = json.loads(_AUDIT_TIERS_PATH.read_text())


@pytest.mark.unit
@pytest.mark.parametrize(
    "row",
    _AUDIT_TIERS,
    ids=[r["name"] for r in _AUDIT_TIERS],
)
def test_component_seniority_in_tier(row):
    """Every catalog entry's seniority falls within its tier."""
    if row["classification"] == "NO-EXPECTED-TIER":
        pytest.skip(
            f"No P-25.2.2.4 expected tier defined for {row['name']!r} — "
            f"add an entry to scripts/audit_benzo_fusion.py:EXPECTED_TIERS "
            f"with a Blue Book citation if a tier becomes appropriate."
        )
    actual = MONOCYCLIC_COMPONENTS[row["name"]]["seniority"]
    assert row["expected_lo"] <= actual <= row["expected_hi"], (
        f"155-AUDIT-C.md classification={row['classification']!r}: "
        f"{row['name']!r} expected seniority in "
        f"[{row['expected_lo']}, {row['expected_hi']}], got {actual}"
    )


@pytest.mark.unit
def test_no_wrong_tier_post_fix():
    """V18-155- acceptance gate: zero WRONG-tier entries post-fix.

    Cross-references internal notes-C.md Classification Summary; this test
    pre-existing as a sentinel to detect future regressions or
    catalog growth that introduces band drift.
    """
    wrong = [
        r for r in _AUDIT_TIERS if r["classification"] == "WRONG-tier"
    ]
    assert wrong == [], (
        f"Phase 155.C D-11 invariant violated: {len(wrong)} WRONG-tier "
        f"entries remain in the audit JSON sidecar. Re-run "
        f"scripts/audit_benzo_fusion.py and either correct the catalog "
        f"or update the EXPECTED_TIERS map with a Blue Book citation. "
        f"Offending entries: {[r['name'] for r in wrong]}"
    )


@pytest.mark.unit
def test_audit_sidecar_is_complete():
    """Every MONOCYCLIC_COMPONENTS key has a corresponding audit row."""
    sidecar_names = {row["name"] for row in _AUDIT_TIERS}
    catalog_names = set(MONOCYCLIC_COMPONENTS.keys())
    missing = catalog_names - sidecar_names
    extra = sidecar_names - catalog_names
    assert not missing and not extra, (
        f"Audit sidecar drift: missing={missing!r} extra={extra!r}. "
        f"Re-run scripts/audit_benzo_fusion.py."
    )
