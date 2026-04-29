"""Phase 150 SC-1 + G5 permanent regression guard.

Per CD-01 (carry-forward to Phase 154+): future stem-filter tightening
that drops below 900 triggers CI fail. Threshold raised here as part of
Phase 150 close; subsequent phases that intentionally tighten the filter
must raise the threshold here AND update the citation.

Source: 150-CONTEXT.md SC-1 + G5; CD-01 carry-forward.
"""

import pytest


@pytest.mark.unit
def test_all_retained_names_at_least_900():
    """SC-1 + G5 hard gate: ALL_RETAINED_NAMES must have >= 900 entries.

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html (P-22 retained-name PIN tier).
    Source: 150-CONTEXT.md SC-1 + G5 + CD-01.
    """
    from orthonym.data import ALL_RETAINED_NAMES
    assert len(ALL_RETAINED_NAMES) >= 900, (
        f"ALL_RETAINED_NAMES has {len(ALL_RETAINED_NAMES)} entries; "
        "Phase 150 G5 requires >= 900. If a phase intentionally tightens "
        "the filter, raise the threshold here AND update the citation."
    )
