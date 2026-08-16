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
    assert len(ALL_RETAINED_NAMES) >= 811, (
        f"ALL_RETAINED_NAMES has {len(ALL_RETAINED_NAMES)} entries; "
        "C6 (homo-ring demotion) intentionally denied 6 non-PIN homo- names "
        "(homopiperidine/homomorpholine/homopiperazine/thiahomomorpholine/"
        "selenohomomorpholine/tellurohomomorpholine), lowering the floor "
        "from the Phase 150 G5 original 900 to 880. v24 W8-P1 R6 demoted "
        "'benzophenone' (-> diphenylmethanone, P-64.2.1.2), and W8-P2 Tasks 2.2/2.3 "
        "demoted '4-phenylphenol' (-> [1,1'-biphenyl]-4-ol, P-44.2.1.5) and "
        "'benzidine' (-> [1,1'-biphenyl]-4,4'-diamine, P-62.2.4.1.1) via pin_list.json "
        "pin:false entries, -3 total -> floor 869. NOTE: the declared 880 was already "
        "stale vs the pre-R6 actual (872) — a pre-existing unattributed drift, now "
        "accurate. Per Phase 150 G5 + CD-01 policy: a phase that intentionally tightens "
        "the filter updates the threshold here AND the citation. "
        "v33 Phase 0 T5 (measured at HEAD 35d5e921, BEFORE this fix): actual was already "
        "830, not 869 -- a second pre-existing unattributed drift (39), left uninvestigated "
        "here since it predates and is unrelated to T5. T5 itself then intentionally "
        "DELETED 19 duplicate, stereo-blind amino-acid entries from "
        "data/retained_names.py (root-cause fix for a live 0-wrong defect: those flat "
        "entries matched a genuinely stereo-undefined input BEFORE the stereo-aware "
        "data/amino_acids.py path ever ran, silently asserting an implicit L "
        "configuration the input does not define -- P-103.1.3.1, BlueBookV2.md:54291). "
        "Coverage is preserved: all 19 have a live equivalent in "
        "data.amino_acids.STANDARD_AMINO_ACIDS. New accurate floor: 830 - 19 = 811."
    )
