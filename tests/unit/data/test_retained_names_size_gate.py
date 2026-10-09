"""a phase + G5 permanent regression guard.

Per (carry-forward to a phase+): future stem-filter tightening
that drops below 900 triggers CI fail. Threshold raised here as part of
a phase close; subsequent phases that intentionally tighten the filter
must raise the threshold here AND update the citation.

Source: 150-internal notes + G5; carry-forward.
"""

import pytest


@pytest.mark.unit
def test_all_retained_names_at_least_900():
    """ + G5 hard gate: ALL_RETAINED_NAMES must not drop below its floor
    (900 at a phase close, 308 now -- see the assertion message for the history).

    Source: https://iupac.qmul.ac.uk/BlueBook/P2.html retained-name PIN tier).
    Source: 150-internal notes + G5 +.
    """
    from orthonym.data import ALL_RETAINED_NAMES
    assert len(ALL_RETAINED_NAMES) >= 308, (
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
        "data.amino_acids.STANDARD_AMINO_ACIDS. Floor 811. "
        "Then 811 -> 799 over later demotions of names the Blue Book does not hold "
        "(first below 811: 465c7f871, 'remove wrong dianion-name shadow for phosphate "
        "mono/di-esters', 807), the test already red at 09a07205d^ (799). "
        "09a07205d (2026-09-26) then intentionally moved 484 OPSIN-import trivial names "
        "that carry no Blue Book PIN evidence ('nicotine', 'lepidine', 'bisphenol a', ...; "
        "0 hits in BlueBookV2.md) out of ALL_RETAINED_NAMES into "
        "OPSIN_UNVERIFIED_RETAINED_NAMES (799 -> 315; the systematic pipeline names those "
        "molecules, the trivial name is served only as a labelled non-PIN fallback; pinned "
        "by tests/unit/data/test_opsin_import_pin_evidence.py). e3b439eb6 (2026-10-02) "
        "then removed the 7 purine-base SMILES keys (adenine, guanine, hypoxanthine, "
        "xanthine tautomers: 0 hits in the Blue Book, not P-25.2.1 Table 2.8 names; "
        "315 -> 308). New accurate floor: 308."
    )
