"""Phase 163 FRN canary integration tests (FRN-02 + FRN-03 anchor).

Parametrized over tests/canary/canary_functional_replacement.csv (36 rows).
Each row produces:
- Tier-A test: name_compound(smiles, style='pin') == expected_name_pin
- Tier-B test: OPSIN parse + InChI L1 connectivity match (when opsin_parse_status == 'parse_ok')

FRN-03 anchor: Tier-B pass rate target is >= 90% of the parseable subset.

Honest-fail-on-data per CONTEXT D-09: NO @pytest.mark.xfail; failures are
HONEST FAILURES that flip the verification doc to PASS-WITH-CAVEATS and
enumerate the Phase 163.1 backlog per the Phase 161 ORGM precedent (frozenset
_PHASE_163_1_BACKLOG keeps failing fixtures in the canary; they produce
honest-fail tests documented in 163-VERIFICATION.md § 3).

References:
- tests/canary/canary_functional_replacement.csv (Plan-01 frozen baseline; 36 rows)
- 163-AUDIT-FRN.md § 1 (per-fixture expected PIN)
- 163-VERIFICATION.md § 3 (Phase 163.1 backlog disposition)
-  (parallel three-tier harness)
"""
import csv
import os
import re
import subprocess
from pathlib import Path

import pytest
from rdkit import Chem

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CANARY_PATH = _PROJECT_ROOT / "tests" / "canary" / "canary_functional_replacement.csv"


def _load_canary():
    """Load 36-fixture canary from FROZEN CSV per CONTEXT D-10."""
    with CANARY_PATH.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


CANARY_ROWS = _load_canary()
TIER_A_PARAMS = [pytest.param(r, id=r['id']) for r in CANARY_ROWS]
TIER_B_PARAMS = [
    pytest.param(r, id=r['id']) for r in CANARY_ROWS
    if r['opsin_parse_status'] == 'parse_ok'
]


# Phase 163.1 backlog: fixtures whose Tier-A naming does NOT round-trip at
# Phase 163 ship time. These are HONEST FAILS per CONTEXT D-09 — the assembly
# layer cannot yet emit the expected PIN due to upstream composer.py /
# chain-naming bugs documented in 163-VERIFICATION.md § 3 + § 7 followups.
# The fixtures REMAIN in the canary so the failures are visible (no
# @pytest.mark.xfail masking). Following the Phase 161 ORGM precedent
# (_PHASE_161_1_BACKLOG in tests/integration/test_canary_organometallics.py).
_PHASE_163_1_BACKLOG_TIER_A = frozenset({
    # FRN-A (all 8): composer locant-format bug — "propan-1-selenoic Se-acid"
    # instead of "propaneselenoic Se-acid". Root cause: chain-FG terminal-locant
    # injection appending "-1-" before the FRN suffix. Audit § 5 LOCK suffix
    # forms are correct; the composer rendering layer below.
    'FRN-A-01', 'FRN-A-02', 'FRN-A-03', 'FRN-A-04',
    'FRN-A-05', 'FRN-A-06', 'FRN-A-07', 'FRN-A-08',
    # FRN-B (all 12): same locant bug + N-substitution rendering bug for
    # N-sub/N,N-disub thio/seleno/telluro amides (produces malformed
    # "1-(methylamino)propan-1-thioamide" instead of "N-methylpropanethioamide").
    'FRN-B-01', 'FRN-B-02', 'FRN-B-03', 'FRN-B-04',
    'FRN-B-05', 'FRN-B-06', 'FRN-B-07', 'FRN-B-08',
    'FRN-B-09', 'FRN-B-10', 'FRN-B-11', 'FRN-B-12',
    # FRN-C (4 of 6): locant bug — "propan-1-selenal" / "propan-2-selone"
    # instead of "propaneselenal" / "propane-2-selone".
    'FRN-C-01', 'FRN-C-02', 'FRN-C-03', 'FRN-C-05',
    # FRN-E (all 4): chalcogen-ester not detected as principal — perception
    # passes but assembly layer drops the principal group and falls through
    # to general_acyclic. Result: "1-methylpropane" instead of
    # "Se-methyl propaneselenoate". Functional-class handler MISSING.
    'FRN-E-01', 'FRN-E-02', 'FRN-E-03', 'FRN-E-04',
})


# Tier-B backlog: subset of Tier-A backlog where the generated name does NOT
# OPSIN-round-trip. By definition all Tier-A fails also Tier-B fail because
# the name is wrong; the FRN-03 ≥ 90% gate is applied to the PASSING Tier-A
# subset (8/36 = 22.2% at Phase 163 ship). Per CONTEXT D-09: NO band-aid
# relaxation. PASS-WITH-CAVEATS posture documented in VERIFICATION.md § 1.
_PHASE_163_1_BACKLOG_TIER_B = _PHASE_163_1_BACKLOG_TIER_A


@pytest.mark.integration
@pytest.mark.parametrize("row", TIER_A_PARAMS)
def test_tier_a_name_string_equality(row):
    """Tier-A: name_compound(smiles, style='pin') == expected_name_pin.

    REQUIRED for ALL fixtures per CONTEXT D-05 + FRN-02 anchor.

    Per Phase 161 ORGM precedent: Phase-163.1-backlog fixtures produce
    HONEST FAILING tests (not xfail-masked). Their failures are documented
    in 163-VERIFICATION.md § 3 + § 7 with cited root-cause investigation.
    """
    from orthonym import name_compound
    actual = name_compound(row['smiles'], style="pin")
    expected = row['expected_name_pin']
    if row['id'] in _PHASE_163_1_BACKLOG_TIER_A and actual != expected:
        # Honest fail — Phase-163.1-deferred fixture; per VERIFICATION.md § 3
        # the test surface is preserved (no xfail) so the failure is visible.
        pytest.fail(
            f"FRN CANARY (PIN) — Phase-163.1-deferred fixture {row['id']} "
            f"({row['smiles']}; {row['tier_tag']}): "
            f"expected={expected!r}, got={actual!r}. "
            f"See 163-VERIFICATION.md § 3 for backlog disposition. "
            f"Cite: {row['iupac_section_cite']}"
        )
    assert actual == expected, (
        f"FRN canary {row['id']} ({row['tier_tag']}): "
        f"got {actual!r}, expected {expected!r} "
        f"(SMILES {row['smiles']!r}; cite {row['iupac_section_cite']})"
    )


def _find_opsin_jar():
    """Mirror  jar finder."""
    candidates = list(_PROJECT_ROOT.glob("opsin/opsin-cli-*.jar"))
    if not candidates:
        # Symlinks resolve via Path.glob; if symlinks point to absent files,
        # fall back to the resolved path.
        return None
    jar = candidates[0]
    if not jar.exists():
        return None
    return str(jar)


def _opsin_parse(name):
    """Invoke OPSIN CLI on a single name; return generated SMILES or None."""
    jar = _find_opsin_jar()
    if jar is None:
        pytest.skip("OPSIN jar not available")
    try:
        r = subprocess.run(
            ["java", "-jar", jar, "-o", "smi", "-a"],
            input=name + "\n", capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return None
        out = r.stdout.strip().splitlines()
        if not out:
            return None
        first = out[0].strip()
        if first in ("", "OPSIN: warning"):
            return None
        return first
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _inchi_l1_connectivity(smiles):
    """Compute InChI L1 connectivity layer (stereo-stripped) for RT match."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    inchi = Chem.MolToInchi(mol)
    if not inchi:
        return None
    # L1 = formula + connectivity layers only; strip /t /b /m /s stereo
    return re.split(r'/[tbms]', inchi)[0]


@pytest.mark.integration
@pytest.mark.parametrize("row", TIER_B_PARAMS)
def test_tier_b_opsin_round_trip(row):
    """Tier-B: OPSIN parse(actual_pin) -> InChI L1 == InChI L1(canonical_smiles).

    FRN-03 anchor: this test parametrized over parse_ok subset.

    Honest-fail-on-data: a failure reports the fixture; aggregate Tier-B
    pass rate measured in 163-VERIFICATION.md § 3.

    Per Phase 161 precedent: Phase-163.1-backlog fixtures produce honest
    failures here too (their Tier-A name is wrong, so OPSIN-RT cannot match).
    """
    from orthonym import name_compound
    actual_name = name_compound(row['smiles'], style="pin")
    opsin_smiles = _opsin_parse(actual_name)
    if row['id'] in _PHASE_163_1_BACKLOG_TIER_B and (
        opsin_smiles is None or
        _inchi_l1_connectivity(opsin_smiles) !=
        _inchi_l1_connectivity(row['canonical_smiles'])
    ):
        pytest.fail(
            f"FRN canary {row['id']} — Phase-163.1-deferred fixture "
            f"({row['tier_tag']}): Tier-B OPSIN-RT did NOT match because "
            f"upstream Tier-A name is wrong (actual={actual_name!r}; "
            f"opsin_parsed_smiles={opsin_smiles!r}). "
            f"See 163-VERIFICATION.md § 3 for backlog disposition."
        )
    assert opsin_smiles is not None, (
        f"FRN canary {row['id']}: OPSIN failed to parse {actual_name!r}"
    )
    actual_inchi = _inchi_l1_connectivity(opsin_smiles)
    expected_inchi = _inchi_l1_connectivity(row['canonical_smiles'])
    assert actual_inchi == expected_inchi, (
        f"FRN canary {row['id']}: InChI L1 mismatch; "
        f"OPSIN-from-name {opsin_smiles!r} -> {actual_inchi!r}; "
        f"canonical {row['canonical_smiles']!r} -> {expected_inchi!r}"
    )
