"""Tier-B recovery test — Phase 156 SC-3 acceptance gate.

Plan-03 Task 3 deliverable. Parametrizes over the audit-locked
`tier_b_v18_pre_156.json` fixture (Plan-01 deliverable) and asserts
the data-driven SC-3 acceptance bar:

    N_target = min(40, floor(M_format_fixable * 0.85))

per `156-AUDIT.md` § 3. With M_format_fixable = 5 the target is 4.

Anti-pattern hygiene:
    AP-15: batched OPSIN call (single JVM startup) — `parse_batch_with_opsin`
           helper from . Per-row
           `subprocess.run` is forbidden (would cost ~1269ms × N).
    AP-16: JAR candidate list contains v2.9.0 ONLY.
    AP-17: pytest.skip with classification reason for non-format-fixable
           rows; NEVER xfail.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orthonym.namer import Orthonym


_FIXTURE_PATH = (
    Path(__file__).parent.parent.parent
    / "" / "phases" / "156-opsin-grammar-pre-validation"
    / "tier_b_v18_pre_156.json"
)


def _java_available() -> bool:
    try:
        subprocess.run(["java", "-version"], capture_output=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _opsin_jar_path():
    # AP-16: v2.9.0 ONLY.
    candidates = [
        "opsin-cli-2.9.0-jar-with-dependencies.jar",
        "opsin/opsin-cli-2.9.0-jar-with-dependencies.jar",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _load_tier_b():
    if not _FIXTURE_PATH.exists():
        pytest.skip(f"Tier-B baseline fixture missing: {_FIXTURE_PATH}")
    with open(_FIXTURE_PATH) as f:
        return json.load(f)


_SKIP = pytest.mark.skipif(
    not _java_available() or _opsin_jar_path() is None,
    reason="Java + OPSIN JAR (v2.9.0) required",
)


def _parse_batch_with_opsin_helper(names, jar):
    """Local re-import of .

    Per 156-PATTERNS.md line 575 + AP-15 prevention: ONE JVM startup, NOT N.
    """
    # scripts/ is not on the package path; add it once for this test module.
    scripts_dir = Path(__file__).parent.parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from benchmark_chebi500 import parse_batch_with_opsin
    return parse_batch_with_opsin(names, jar, timeout=120.0)


# ---------------------------------------------------------------------------
# Module-scoped fixture: name + batch-OPSIN-parse all rows ONCE.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tier_b_with_156():
    """Compute v18-with-156 outputs ONCE per test module run.

    Returns the rows list with two added keys per row:
        - `v18_name_with_156`: Orthonym().name(smiles) (grammar layer ON)
        - `v18_opsin_score_with_156`: 1 if OPSIN parses + RDKit accepts, else 0
    """
    if not _java_available() or _opsin_jar_path() is None:
        pytest.skip("Java + OPSIN JAR required")

    rows = _load_tier_b()
    namer = Orthonym()

    # Phase 1: Name every row through the grammar-layer-ON pipeline.
    for r in rows:
        try:
            r["v18_name_with_156"] = namer.name(r["smiles"])
        except Exception as e:
            r["v18_name_with_156"] = ""
            r["v18_naming_error"] = str(e)

    # Phase 2: Batch-OPSIN-parse all unique non-empty names in ONE JVM call
    # per AP-15 prevention. Per 156-AUDIT.md CF-1 the empirical OPSIN
    # call cost is 1269ms mean; per-row subprocess.run would cost ~89s
    # for 70 rows. Batched stdin/stdout streaming amortizes to ~17ms/name.
    unique_names = list({
        r["v18_name_with_156"] for r in rows if r.get("v18_name_with_156")
    })
    jar = _opsin_jar_path()
    rt_map, _err_map = _parse_batch_with_opsin_helper(unique_names, jar)

    # InChI L1 score per row. The canonical helper lives at
    # :inchi_l1_match` — try-import; if the
    # script path is unavailable, use a stdlib RDKit-direct fallback.
    try:
        from benchmark_7metric import inchi_l1_match  # type: ignore
    except ImportError:
        from rdkit import Chem
        from rdkit.Chem.inchi import MolToInchi

        def inchi_l1_match(smi_a, smi_b):  # type: ignore[no-redef]
            mol_a = Chem.MolFromSmiles(smi_a) if smi_a else None
            mol_b = Chem.MolFromSmiles(smi_b) if smi_b else None
            if mol_a is None or mol_b is None:
                return 0
            try:
                ia = MolToInchi(mol_a)
                ib = MolToInchi(mol_b)
            except Exception:
                return 0
            if not ia or not ib:
                return 0
            # Layer 1 = chemical formula + connectivity (strip stereo).
            la = ia.split("/s")[0].split("/t")[0].split("/m")[0]
            lb = ib.split("/s")[0].split("/t")[0].split("/m")[0]
            return 1 if la == lb else 0

    for r in rows:
        n = r.get("v18_name_with_156")
        rt_smi = rt_map.get(n) if n else None
        if rt_smi:
            r["v18_opsin_score_with_156"] = (
                inchi_l1_match(r["smiles"], rt_smi) or 0
            )
        else:
            r["v18_opsin_score_with_156"] = 0

    return rows


# ---------------------------------------------------------------------------
# SC-3 aggregate acceptance gate
# ---------------------------------------------------------------------------


@_SKIP
def test_tier_b_recovery_aggregate(tier_b_with_156):
    """SC-3 data-driven acceptance per 156-AUDIT.md § 3.

    N_target = min(40, floor(M_format_fixable * 0.85))
    With M_format_fixable = 5, N_target = 4.
    """
    rows = tier_b_with_156
    ff = [r for r in rows if r.get("v18_classification") == "format-fixable"]
    m_ff = len(ff)
    n_recovered = sum(
        1 for r in ff if r.get("v18_opsin_score_with_156") == 1
    )
    n_target = min(40, math.floor(m_ff * 0.85))
    assert n_recovered >= n_target, (
        f"SC-3 acceptance: recovered {n_recovered} of {m_ff} format-fixable; "
        f"target was {n_target}. See 156-VERIFICATION.md per-Tier-B "
        f"disposition table for full diagnostic data."
    )


# ---------------------------------------------------------------------------
# Per-row parametrized test (documentation-quality; pytest.skip for
# non-format-fixable rows per AP-17 — no xfail).
# ---------------------------------------------------------------------------


def _row_id(r):
    smi = r.get("smiles", "?")
    return smi[:30]


# Using `_load_tier_b()` at collection time is acceptable here: this
# only loads the audit-locked JSON; the Orthonym naming + OPSIN parsing
# happens inside the module-scoped fixture, NOT at collection.
_ALL_ROWS = _load_tier_b() if _FIXTURE_PATH.exists() else []


@_SKIP
@pytest.mark.parametrize("row", _ALL_ROWS, ids=_row_id)
def test_tier_b_per_row(row, tier_b_with_156):
    """Per-row diagnostic test.

    Skip with classification reason for non-format-fixable rows per
    156-PATTERNS.md line 577 + AP-17 (NEVER xfail).
    """
    classification = row.get("v18_classification", "?")
    if classification != "format-fixable":
        pytest.skip(
            f"Not format-fixable: {classification} — "
            f"{row.get('v18_classification_reason', '')}"
        )
    # Find this row in the with-156 fixture and assert it gained
    # opsin_score=1.
    matches = [r for r in tier_b_with_156 if r["smiles"] == row["smiles"]]
    assert matches, f"Row not found in with-156 fixture: {row['smiles'][:40]!r}"
    matched = matches[0]
    assert matched.get("v18_opsin_score_with_156") == 1, (
        f"Tier-B compound {row['smiles'][:40]!r} did not recover; "
        f"v18_name_with_156={matched.get('v18_name_with_156')!r}"
    )
