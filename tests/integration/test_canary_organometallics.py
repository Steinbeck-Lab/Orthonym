"""Phase 161 integration canary: ORGM 53-fixture per-tier validation.

ORGM-04 acceptance: 50/53 Tier-A on the in-Phase-161-scope subset
(53 total minus 3 Phase-161.1-deferred fixtures: T4-07, T4-15, T4-22).
v23 Phase 10 resolved T4-12 (ethenyllithium); it is now in-scope/passing.

Per CONTEXT D-09 + project memory rule #4 (no band-aids): the 3 deferred
fixtures remain in the canary and produce HONEST FAILING tests per
honest-fail-on-data. They are NOT @pytest.mark.xfail-masked — VERIFICATION.md
§ 3 documents each disposition.

Audit amendments applied at Plan-04 (documented in VERIFICATION.md):
- T4-13: PIN updated to 'trimethylphenylsilane' (IUPAC P-29.2 alphabetic)
- T4-14: PIN/SYS updated to 'butyllithium' (SMILES is n-butyl, not sec-butyl)
"""
import csv
from pathlib import Path

import pytest

from orthonym import name_compound

_CANARY_CSV = Path(__file__).parent.parent / "canary" / "canary_organometallics.csv"


def _load_orgm_canary():
    """Load 53-fixture canary from FROZEN CSV per CONTEXT D-10.

    Per CONTEXT D-29 + Phase 161 Plan-04: the CSV is loaded as-is; the 4
    Phase-161.1-deferred fixtures (T4-07, T4-12, T4-15, T4-22) are NOT
    filtered out — they produce HONEST FAILING tests documented in
    161-VERIFICATION.md.
    """
    fixtures = []
    with _CANARY_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fixtures.append(row)
    return fixtures


ORGM_CANARY = _load_orgm_canary()
_ORGM_IDS = [row['id'] for row in ORGM_CANARY]

# Phase-161.1 backlog (3 fixtures): honest failures documented in 161-VERIFICATION.md § 3
# These fixtures remain in the canary; the tests fail; the failures are scope-deferred per CONTEXT D-01.
# v23 Phase 10 RESOLVED ORG-T4-12 ([Li]C=C): the σ-unsaturated ligand recogniser now
# emits 'ethenyl' (root-cause fix in _ligand_name_from_atoms). Its CSV expected_name_pin
# was also corrected from the mislabeled 'vinyllithium' to the true PIN 'ethenyllithium'
# (P-31.1.4.3.4 — vinyl is retained, general-nomenclature only).
_PHASE_161_1_BACKLOG = frozenset({'ORG-T4-07', 'ORG-T4-15', 'ORG-T4-22'})


@pytest.mark.integration
@pytest.mark.parametrize("row", ORGM_CANARY, ids=_ORGM_IDS)
def test_canary_organometallic_pin(row):
    """ORGM-04 Tier-A name-string equality per fixture (style='pin').

    Per Plan-04 acceptance reformulation: 49/49 in-scope fixtures must pass.
    The 4 Phase-161.1-deferred fixtures (T4-07/12/15/22) honestly fail;
    failures are scope-deferred per CONTEXT D-01.
    """
    smiles = row['smiles']
    expected = row['expected_name_pin']
    result = name_compound(smiles, style='pin')
    if row['id'] in _PHASE_161_1_BACKLOG and result != expected:
        # Honest fail — Phase-161.1-deferred fixture; per VERIFICATION.md § 3
        # the test surface is preserved (no xfail) so the fail is visible.
        pytest.fail(
            f"ORGM CANARY (PIN) — Phase-161.1-deferred fixture {row['id']} "
            f"({smiles}): expected={expected!r}, got={result!r}. "
            f"See 161-VERIFICATION.md § 3 for backlog disposition."
        )
    assert result == expected, (
        f"ORGM CANARY REGRESSION (PIN): {row['id']} ({smiles})\n"
        f"  Expected: {expected}\n"
        f"  Got: {result}"
    )


@pytest.mark.integration
@pytest.mark.parametrize("row", ORGM_CANARY, ids=_ORGM_IDS)
def test_canary_organometallic_systematic(row):
    """ORGM-04 Tier-A name-string equality per fixture (style='systematic')."""
    smiles = row['smiles']
    expected = row['expected_name_systematic']
    result = name_compound(smiles, style='systematic')
    if row['id'] in _PHASE_161_1_BACKLOG and result != expected:
        pytest.fail(
            f"ORGM CANARY (SYSTEMATIC) — Phase-161.1-deferred fixture {row['id']} "
            f"({smiles}): expected={expected!r}, got={result!r}. "
            f"See 161-VERIFICATION.md § 3 for backlog disposition."
        )
    assert result == expected, (
        f"ORGM CANARY REGRESSION (SYSTEMATIC): {row['id']} ({smiles})\n"
        f"  Expected: {expected}\n"
        f"  Got: {result}"
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "row",
    [r for r in ORGM_CANARY if r['tier_tag'] == 'tier1'],
    ids=lambda r: r['id'],
)
def test_tier1_metallocenes(row):
    """Tier-1 sub-canary per VALIDATION row 161-03-01. All 9 fixtures pass."""
    smiles = row['smiles']
    result_pin = name_compound(smiles, style='pin')
    assert result_pin == row['expected_name_pin']


@pytest.mark.integration
@pytest.mark.parametrize(
    "row",
    [r for r in ORGM_CANARY if r['tier_tag'] == 'tier2'],
    ids=lambda r: r['id'],
)
def test_tier2_carbonyls(row):
    """Tier-2 sub-canary. All 8 fixtures pass."""
    smiles = row['smiles']
    result_pin = name_compound(smiles, style='pin')
    assert result_pin == row['expected_name_pin']


@pytest.mark.integration
@pytest.mark.parametrize(
    "row",
    [r for r in ORGM_CANARY if r['tier_tag'] == 'tier3'],
    ids=lambda r: r['id'],
)
def test_tier3_sigma_bonded(row):
    """Tier-3 sub-canary. All 11 fixtures pass."""
    smiles = row['smiles']
    result_pin = name_compound(smiles, style='pin')
    assert result_pin == row['expected_name_pin']


@pytest.mark.integration
@pytest.mark.parametrize(
    "row",
    [r for r in ORGM_CANARY if r['tier_tag'] == 'tier4'],
    ids=lambda r: r['id'],
)
def test_tier4_eta_bonded(row):
    """Tier-4 sub-canary. 21/25 fixtures pass; 4 are Phase-161.1 backlog."""
    smiles = row['smiles']
    result_pin = name_compound(smiles, style='pin')
    if row['id'] in _PHASE_161_1_BACKLOG and result_pin != row['expected_name_pin']:
        pytest.fail(
            f"Tier-4 backlog fixture {row['id']}: expected={row['expected_name_pin']!r}, "
            f"got={result_pin!r}. See 161-VERIFICATION.md § 3."
        )
    assert result_pin == row['expected_name_pin']


@pytest.mark.integration
def test_canary_in_scope_count():
    """Audit invariant: exactly 50 in-scope fixtures + 3 Phase-161.1 backlog = 53 total.

    v23 Phase 10 moved ORG-T4-12 (ethenyllithium) from backlog → in-scope
    (the σ-unsaturated ligand recogniser fix resolved it): 49→50 in-scope, 4→3 backlog.
    """
    in_scope = [r for r in ORGM_CANARY if r['id'] not in _PHASE_161_1_BACKLOG]
    backlog = [r for r in ORGM_CANARY if r['id'] in _PHASE_161_1_BACKLOG]
    assert len(in_scope) == 50, f"Expected 50 in-scope; got {len(in_scope)}"
    assert len(backlog) == 3, f"Expected 3 backlog; got {len(backlog)}"
    assert len(ORGM_CANARY) == 53


@pytest.mark.integration
def test_canary_csv_schema_locked():
    """Audit amendment safety: CSV has 53 rows + 12-column schema preserved."""
    with _CANARY_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    assert len(rows) == 54, f"Expected 54 lines (1 header + 53 data); got {len(rows)}"
    header = rows[0]
    expected_columns = {
        'id', 'smiles', 'canonical_smiles',
        'expected_name_pin', 'expected_name_systematic',
        'tier_tag', 'ligand_class', 'hapticity_n',
        'opsin_parse_pin_status', 'opsin_parse_systematic_status',
        'inchi_rt_status', 'iupac_section_cite',
    }
    assert set(header) == expected_columns, (
        f"Schema drift: header={header!r}, expected={expected_columns!r}"
    )
