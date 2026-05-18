"""MLF-06 byte-identical contract assertion (Phase 162).

For >=100 stratified rule-based-success fixtures from
``canary_ml_rule_based_success.csv``:

    Orthonym(allow_ml_fallback=True).name(smi) == Orthonym(allow_ml_fallback=False).name(smi)

The contract is **byte-identical**: enabling the ML fallback MUST NOT change
the output of any compound that the rule-based pipeline already handles.

Per 162-AUDIT-MLF.md § 4 + RESEARCH R-08:
* Tier 1 + Tier 2 fixtures (OPSIN-parseable expected names) use the default
  ``opsin_parse_required=True`` — the Pattern 5 OPSIN gate is the right
  signal for these
* Tier 3 fixtures (OPSIN-unparseable expected names) use
  ``opsin_parse_required=False`` — these fixtures HAVE a rule-based handler
  but the handler emits names that OPSIN cannot parse (grammar gap, NOT
  pipeline degradation); P5 would fire incorrectly and attach ML

Failure mode: any violation indicates a quality-gate predicate fired
inappropriately on a rule-based-success output. The fix is upstream in
``src/orthonym/ml_fallback/quality_gate.py`` (audit-amendment commit per
CONTEXT D-07), NOT a threshold relaxation or fixture skip per the contributor guide
memory rule 4 + project memory feedback_user_preferences.md rules 1+4+5+6.

Inherits the Phase 145.1 byte-identical methodology + Phase 161 D-29
honest-fail-on-data convention.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import pytest

from orthonym import Orthonym


CANARY_CSV = (
    Path(__file__).resolve().parent.parent / "canary" / "canary_ml_rule_based_success.csv"
)


def _load_canary() -> List[Dict[str, str]]:
    """Load the >=100 stratified rule-based-success canary."""
    with CANARY_CSV.open(newline="") as fh:
        return list(csv.DictReader(fh))


CANARY_ROWS = _load_canary()
CANARY_IDS = [
    f"{r['stout_class']}-{r['source_fixture_id']}"
    for r in CANARY_ROWS
]


@pytest.mark.integration
@pytest.mark.parametrize("row", CANARY_ROWS, ids=CANARY_IDS)
def test_ml_never_overrides_rule_based(row: Dict[str, str]) -> None:
    """MLF-06: ML triggers ONLY for compounds with NO rule-based handler.

    Asserts byte-identical preservation:
        Orthonym(allow_ml_fallback=True).name(smi) ==
        Orthonym(allow_ml_fallback=False).name(smi)

    Per 162-AUDIT-MLF.md § 4.3 + RESEARCH R-08: Tier 3 fixtures use
    ``opsin_parse_required=False`` because their rule-based-success names
    are OPSIN-unparseable by design (grammar gap, not pipeline degradation).
    """
    # Tier-3 conditional: P5 OPSIN-parse gate must be bypassed for Tier 3
    # fixtures (RESEARCH R-08; audit § 4.3).
    tier = row.get("tier", "Tier 1")
    opsin_required = "Tier 3" not in tier

    n_off = Orthonym(
        allow_ml_fallback=False,
        opsin_parse_required=opsin_required,
    )
    n_on = Orthonym(
        allow_ml_fallback=True,
        opsin_parse_required=opsin_required,
    )

    name_off = n_off.name(row["smiles"])
    name_on = n_on.name(row["smiles"])

    assert name_off == name_on, (
        f"MLF-06 VIOLATION on {row['source_fixture_id']} "
        f"({row['stout_class']}, {tier}):\n"
        f"  ML-OFF: {name_off!r}\n"
        f"  ML-ON:  {name_on!r}\n"
        f"\n"
        f"Root cause: the quality-gate predicate "
        f"`is_name_quality_inadequate` fired on a rule-based-success output, "
        f"attaching ML. The fix is upstream in "
        f"`src/orthonym/ml_fallback/quality_gate.py` (audit-amendment commit "
        f"per CONTEXT D-07), NOT a threshold relaxation or canary exception "
        f"per the contributor guide memory rule 4 + 162-AUDIT-MLF.md § 2 lock + project "
        f"memory feedback_user_preferences.md rules 1+4+5+6."
    )
