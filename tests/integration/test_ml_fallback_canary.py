"""ML-fires canary assertion per MLF-02 + MLF-06 ml-attaches branch.

For >=30 fixtures from ``canary_ml_fallback.csv`` (compounds where the
rule-based pipeline fails or degrades enough to trigger ML attachment):

* ``Orthonym(allow_ml_fallback=True).name_with_confidence(smi)`` returns
  a dict
* ``result['ml_fallback_used']`` is ``True``
* ``result['ml_model_version'] == STOUT_MODEL_SHA256`` (pinned SHA per
  MLF-05 reproducibility-pin)
* ``result['name']`` is the ML-produced name (may differ from the rule-based
  degraded output)

If STOUT is not installed OR the R-02 sentinel SHA is in place: every test
gracefully skips via ``pytest.skip`` with a helpful diagnostic message.

This contract asserts that **the wrapper attaches** and that **the SHA-pin
holds** — NOT that the ML output is any particular name (STOUT may emit
different but valid names; equality assertions would be brittle).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import pytest

from orthonym import Orthonym
from orthonym.data.ml_model_pin import STOUT_MODEL_SHA256


CANARY_CSV = (
    Path(__file__).resolve().parent.parent / "canary" / "canary_ml_fallback.csv"
)


def _load_canary() -> List[Dict[str, str]]:
    """Load the >=30 ML-test canary."""
    with CANARY_CSV.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _stout_available() -> bool:
    """Returns ``True`` iff the SHA pin is real (not the R-02 sentinel)
    AND the STOUT package is importable. Used by the test gate.
    """
    if STOUT_MODEL_SHA256.startswith("BLOCKED_ON_R02"):
        return False
    try:
        import STOUT  # noqa: F401, WPS433
    except ImportError:
        return False
    return True


CANARY_ROWS = _load_canary()
CANARY_IDS = [f"ml-{i:02d}" for i in range(len(CANARY_ROWS))]


@pytest.mark.integration
@pytest.mark.skipif(
    not _stout_available(),
    reason=(
        "STOUT not installed OR R-02 sentinel SHA in ml_model_pin.py "
        "(see 162-AUDIT-MLF.md § 1.3)"
    ),
)
@pytest.mark.parametrize("row", CANARY_ROWS, ids=CANARY_IDS)
def test_ml_fallback_canary_fires(row: Dict[str, str]) -> None:
    """ML-test canary: wrapper fires + MLF-02 annotations populated correctly.

    Per 162-AUDIT-MLF.md § 3: every row's rule_based_output is degraded
    (P1..P6 catches); enabling --allow-ml-fallback should produce a
    non-None ``ml_result.name`` (or, at minimum, set
    ``ml_fallback_used=True`` with the pinned SHA — the wrapper attaches
    even when STOUT inference fails).
    """
    namer = Orthonym(allow_ml_fallback=True)
    result = namer.name_with_confidence(row["canonical_smiles"])

    assert isinstance(result, dict), (
        f"name_with_confidence must return dict; got {type(result)!r}"
    )
    assert "ml_fallback_used" in result, (
        f"MLF-02 contract violation: missing 'ml_fallback_used' key; "
        f"result={result!r}"
    )
    assert "ml_model_version" in result, (
        f"MLF-02 contract violation: missing 'ml_model_version' key; "
        f"result={result!r}"
    )

    # The ML fallback MUST fire on these fixtures (audit § 3 guarantees
    # they are rule-based-degraded).
    assert result["ml_fallback_used"] is True, (
        f"ML-test canary fixture failed to trigger ML fallback.\n"
        f"  SMILES:           {row['canonical_smiles']!r}\n"
        f"  Rule-based output: {row['rule_based_output']!r}\n"
        f"\n"
        f"Root cause: the quality-gate predicate `is_name_quality_inadequate` "
        f"did not flag this output as degraded — P1..P6 missed it.\n"
        f"Fix: extend the predicate via audit-amendment commit (NOT a "
        f"fixture skip per memory rule 4 + 162-AUDIT-MLF.md § 2 lock + "
        f"project memory feedback_user_preferences.md rules 1+4+5+6)."
    )

    # SHA-pin contract per MLF-05 + audit § 6: when ML attaches, the
    # model version recorded MUST match the pinned SHA-256 manifest hash.
    assert result["ml_model_version"] == STOUT_MODEL_SHA256, (
        f"MLF-05 SHA-pin mismatch:\n"
        f"  Observed: {result['ml_model_version']!r}\n"
        f"  Pinned:   {STOUT_MODEL_SHA256!r}\n"
        f"\n"
        f"This indicates model artifact drift. Bump STOUT_MODEL_SHA256 in "
        f"src/orthonym/data/ml_model_pin.py + re-run the MLF-04 dual-config "
        f"measurement per Phase 164 audit cadence (see 162-AUDIT-MLF.md § 6.4 "
        f"amendment commit protocol)."
    )
