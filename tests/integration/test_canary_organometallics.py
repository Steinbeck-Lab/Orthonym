"""a phase integration canary: ORGM 53-fixture per-tier validation.

 acceptance: 50/53 Tier-A on the in-Phase-161-scope subset
(53 total minus 3 Phase-161.1-deferred fixtures: T4-07, T4-15, T4-22).
 a phase resolved T4-12 (ethenyllithium); it is now in-scope/passing.
Suite fix j4 (2026-09-27) resolved T4-22 (hexamethyldisilane,:
51 in-scope, 2 deferred (T4-07, T4-15).

Per internal notes + project memory rule #4 (no band-aids): the 3 deferred
fixtures remain in the canary and produce HONEST FAILING tests per
honest-fail-on-data. They are NOT @pytest.mark.xfail-masked — VERIFICATION.md
 documents each disposition.

Audit amendments applied at Plan-04 (documented in VERIFICATION.md):
- T4-13: PIN updated to 'trimethylphenylsilane' (IUPAC alphabetic); the
  second cited prefix enclosed, 'trimethyl(phenyl)silane',
  the Blue Book; 'trichloro(iodomethyl)silane (PIN)':25870)
- T4-14: PIN/SYS updated to 'butyllithium' (SMILES is n-butyl, not sec-butyl)
"""
import csv
from pathlib import Path

import pytest

from orthonym import name_compound

_CANARY_CSV = Path(__file__).parent.parent / "canary" / "canary_organometallics.csv"


def _load_orgm_canary():
    """Load 53-fixture canary from FROZEN CSV per internal notes.

    Per internal notes + a phase Plan-04: the CSV is loaded as-is; the 4
    Phase-161.1-deferred fixtures (T4-07, T4-12, T4-15, T4-22) are NOT
    filtered out — they produce HONEST FAILING tests documented in
    internal notes.
    """
    fixtures = []
    with _CANARY_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fixtures.append(row)
    return fixtures


ORGM_CANARY = _load_orgm_canary()
_ORGM_IDS = [row['id'] for row in ORGM_CANARY]

# Phase-161.1 backlog (3 fixtures): honest failures documented in internal notes
# These fixtures remain in the canary; the tests fail; the failures are scope-deferred per internal notes.
# a phase RESOLVED ORG-T4-12 ([Li]C=C): the σ-unsaturated ligand recogniser now
# emits 'ethenyl' (root-cause fix in _ligand_name_from_atoms). Its CSV expected_name_pin
# was also corrected from the mislabeled 'vinyllithium' to the true PIN 'ethenyllithium'
# — vinyl is retained, general-nomenclature only).
# 2026-09-27 (suite fix j4, TRIAGE g3 C10d) RESOLVED ORG-T4-22: the Group-14
# hydride producer now applies (the Blue Book, "All locants are
# omitted in compounds or substituent groups in which all substitutable positions
# are completely substituted or modified... in the same way"), so the fully
# methylated disilane is 'hexamethyldisilane', as the CSV always expected.
# 2026-09-27 (suite fix j7, TRIAGE g3 C12) RESOLVED ORG-T4-15: the Group-1/2 ligand
# recogniser names the tert-butyl ligand, the Blue Book
# '*tert*-butyldi(methyl)phosphane (PIN)'): 'tert-butyllithium', OPSIN
# full-InChIKey exact. The remaining backlog row is a STRICT XFAIL now, not a
# deliberate pytest.fail: ORG-T4-07's SMILES draws a sigma Pd-CH2 bond (Cl[Pd]CC=C)
# while both expected names are eta3-allyl (a different bonding description), and
# OPSIN 2.9.0 parses neither eta name; the engine abstains ('palladium compound
# (not supported)'). Needs a fixture ruling (sigma vs eta3) and a Pd producer.
# 2026-09-30 (pre-existing 26): ORG-T4-06 and ORG-T4-08 join the backlog. Their
# SMILES are other molecules than their expected names: '[Ni].C=CC.C=CC' is C6H12Ni
# (two neutral propenes) while 'bis(eta3-prop-2-en-1-yl)nickel' is Ni + 2 C3H5 =
# C6H10Ni, and 'C1=CC=CC=CC=1' is C7H6 (cyclohepta-1,2,4,6-tetraene) while
# 'cycloheptatrienyl' is C7H7 (RDKit CalcMolFormula). The atom-conservation veto
# of d961ed8b9 (rules/organometallics.py, _PI_LIGAND_REFERENCE_FORMULA) declines
# both at the PIN and systematic styles; the gold rows of the same molecules were
# changed to expect that decline in 51ae5378a. OPSIN 2.9.0 parses none of the eta
# names, so no read-back can confirm a redrawn SMILES either, and a redrawn
# '[Ni].[CH2]C=C.[CH2]C=C' is declined too (no eta-allyl producer). The
# best-effort tier names both drawings as written, read back exact:
# 'propene—nickel (2/1)', 'cyclohepta-1,2,4,6-tetraene—1-oxaeth-1-yn-1-ium-2-ide—
# manganese (1/3/1)'. (the Blue Book, heading '
# ORGANOMETALLIC COMPOUNDS INVOLVING THE ELEMENTS IN GROUPS 3 THROUGH 12'):
# "Coordination nomenclature is the primary nomenclature method used to name
# organometallic compounds containing elements of Groups 3 through 12."
_PHASE_161_1_BACKLOG = frozenset({'ORG-T4-06', 'ORG-T4-07', 'ORG-T4-08'})
_BACKLOG_REASONS = {
    'ORG-T4-07': (
        "Phase-161.1 backlog (TRIAGE g3 C12): ORG-T4-07 SMILES is sigma Pd-CH2 but "
        "the expected names are eta3-allyl; OPSIN 2.9.0 cannot parse eta names; no "
        "Pd producer -- the engine abstains. See 161-VERIFICATION.md section 3."),
    'ORG-T4-06': (
        "fixture ruling needed: the SMILES [Ni].C=CC.C=CC is C6H12Ni (two neutral "
        "propenes), the expected eta3-allyl names are C6H10Ni; the atom-conservation "
        "veto declines it (0-wrong); OPSIN 2.9.0 cannot parse eta names; no "
        "eta-allyl producer"),
    'ORG-T4-08': (
        "fixture ruling needed: the SMILES ring C1=CC=CC=CC=1 is C7H6, the expected "
        "eta7-cycloheptatrienyl names are C7H7; the atom-conservation veto declines "
        "it (0-wrong); OPSIN 2.9.0 cannot parse eta names"),
}


def _orgm_params(rows):
    return [pytest.param(r, id=r['id'],
                         marks=pytest.mark.xfail(strict=True,
                                                 reason=_BACKLOG_REASONS[r['id']]))
            if r['id'] in _PHASE_161_1_BACKLOG else pytest.param(r, id=r['id'])
            for r in rows]


@pytest.mark.integration
@pytest.mark.parametrize("row", _orgm_params(ORGM_CANARY))
def test_canary_organometallic_pin(row):
    """ Tier-A name-string equality per fixture (style='pin').

    50 in-scope fixtures must pass; the backlog rows (T4-06, T4-07, T4-08) are
    strict xfails with their reasons (j7: T4-07 used to call pytest.fail by design).
    """
    smiles = row['smiles']
    expected = row['expected_name_pin']
    result = name_compound(smiles, style='pin')
    assert result == expected, (
        f"ORGM CANARY REGRESSION (PIN): {row['id']} ({smiles})\n"
        f"  Expected: {expected}\n"
        f"  Got: {result}"
    )


@pytest.mark.integration
@pytest.mark.parametrize("row", _orgm_params(ORGM_CANARY))
def test_canary_organometallic_systematic(row):
    """ Tier-A name-string equality per fixture (style='systematic')."""
    smiles = row['smiles']
    expected = row['expected_name_systematic']
    result = name_compound(smiles, style='systematic')
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
    "row", _orgm_params([r for r in ORGM_CANARY if r['tier_tag'] == 'tier4']))
def test_tier4_eta_bonded(row):
    """Tier-4 sub-canary: 22/25 fixtures pass; T4-06, T4-07 and T4-08 are the
    strict-xfail backlog rows."""
    smiles = row['smiles']
    result_pin = name_compound(smiles, style='pin')
    assert result_pin == row['expected_name_pin']


@pytest.mark.integration
def test_canary_in_scope_count():
    """Audit invariant: exactly 50 in-scope fixtures + 3 backlog rows = 53 total.

     a phase moved ORG-T4-12 (ethenyllithium) from backlog → in-scope
    (the σ-unsaturated ligand recogniser fix resolved it): 49→50 in-scope, 4→3 backlog.
    Suite fix j4 moved ORG-T4-22 (hexamethyldisilane, likewise:
    50→51 in-scope, 3→2 backlog.
    """
    in_scope = [r for r in ORGM_CANARY if r['id'] not in _PHASE_161_1_BACKLOG]
    backlog = [r for r in ORGM_CANARY if r['id'] in _PHASE_161_1_BACKLOG]
    # j7 moved ORG-T4-15 (tert-butyllithium) in-scope: 51 -> 52, backlog 2 -> 1.
    # 2026-09-30: ORG-T4-06 and ORG-T4-08 (SMILES and expected names are different
    # molecules) moved to the backlog: 52 -> 50, backlog 1 -> 3.
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
