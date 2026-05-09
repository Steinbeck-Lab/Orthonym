"""Phase 157 regression-lock tests + new-errata tests + tracking-doc smoke.

Each test asserts that a Phase 137 / 151 / 152 / 153 / 155.C errata fix
remains intact after the post-Phase-156 codebase. A failing regression-lock
is a BLOCKER (per CONTEXT D-13 mutual-exclusion contract): Phase 157 does
NOT silently re-apply the fix; the failing test triggers an audit of which
upstream phase silently reverted.

Test taxonomies (CONTEXT D-22):
  1. Regression-lock (>= 11 tests, 1 per errata) -- assert Phase 137/155.C
     deliverables intact. Suffix: _locked_157.
  2. Re-verification meta-tests (>= 2 tests) -- subprocess greps for NA-
     classified rules (P-31.2 dehydro, P-31.1.4.3 compound-locants) per
     CONTEXT D-07. Suffix: _not_applicable_grep_157.
  3. Net-new errata tests (>= 0 tests) -- per Plan-01 NO-OP-CONFIRMED on
     both P-25.8.1 (157-AUDIT.md S3.4) and P-25.4.3.2.2 (157-AUDIT.md S4.5),
     these tests run as smoke-only protective infrastructure, not FIX-SPEC
     assertions. The S3.5 FIX-SPEC payload remains preserved in the audit
     doc for v19 follow-up.
  4. Tracking-doc smoke tests (>= 3 tests) -- assert
      exists, >= 250 LOC, contains all 14 PIN
     entries. Suffix: _doc_*_157.

Reference: IUPAC 2013 Blue Book + BBerrors.html corrections through 31 Dec 2025.
Source: 157-CONTEXT.md D-01..D-25; 157-AUDIT.md (Plan-01 atomic commit);
        157-02-SUMMARY.md (Plan-03 inputs section).

Per CONTEXT D-13 / D-12 / D-14 / D-25 (mutual-exclusion contracts): this
test module ONLY IMPORTS from Phase 137 / 155.C / 151 / 137-02 deliverables;
it MUST NOT modify any of them.

Per CONTEXT Pitfall 7: P-29.1.2 regression-lock does NOT invoke OPSIN
round-trip (OPSIN v2.9.0 does not parse `carbonochloridoyl`).

Per CONTEXT Pitfall 3: test_carbonochloridoyl_locked_157 asserts BOTH the
chlorocarbonyl -> carbonochloridoyl change AND the bromocarbonyl /
fluorocarbonyl intentional non-change (BBerrors says only -COCl renamed).
"""

import re
import subprocess
from pathlib import Path

import pytest


# Repo-root resolver shared by subprocess-grep tests + tracking-doc tests.
def _repo_root() -> Path:
    """Return the repo root by walking up from this test file.

    tests/unit/rules/test_iupac_errata_157.py
       -> parents[3] is the repo root.
    """
    return Path(__file__).resolve().parents[3]


# ============================================================================
# Taxonomy 1: Regression-lock for Phase 137 / 155.C / 151 deliverables
# Suffix: _locked_157
# ============================================================================


class TestP18bSeniorityLock157:
    """P-18(b) regression: 20-element seniority dict preserved post-Phase-156.

    Phase 137-01 ERRATA-01 expanded _HETEROATOM_SENIORITY from 10 to 20.
    D-13 mutual-exclusion: NO edit on Phase 137 deliverables. Failure here
    BLOCKS Phase 157 + triggers audit of which upstream phase reverted.

    Source: BBerrors.html P-18(b) Jan 2019 errata.
    Source: 157-AUDIT.md row 1 + S1.1 row 1 grep transcript.
    """

    def test_heteroatom_seniority_dict_size_locked_157(self) -> None:
        """20-element dict preserved per BBerrors.html P-18(b) Jan 2019 errata."""
        from orthonym.rules.ring_selection import _HETEROATOM_SENIORITY

        assert len(_HETEROATOM_SENIORITY) == 20, (
            f"P-18(b) regression: dict size {len(_HETEROATOM_SENIORITY)} != 20. "
            f"Phase 137 ERRATA-01 silently reverted. AUDIT REQUIRED."
        )
        assert _HETEROATOM_SENIORITY['N'] == 20, (
            f"P-18(b): N rank {_HETEROATOM_SENIORITY['N']} != 20."
        )
        assert _HETEROATOM_SENIORITY['Ga'] == 1, (
            f"P-18(b): Ga rank {_HETEROATOM_SENIORITY['Ga']} != 1."
        )
        assert _HETEROATOM_SENIORITY['As'] == 10, (
            f"P-18(b): As rank {_HETEROATOM_SENIORITY['As']} != 10."
        )

    def test_heteroatom_seniority_p18b_elements_locked_157(self) -> None:
        """All 10 P-18(b) ERRATA-01 elements present (As, Sb, Bi, Si, Ge, Sn, Pb, B, Al, Ga)."""
        from orthonym.rules.ring_selection import _HETEROATOM_SENIORITY

        for added in ('As', 'Sb', 'Bi', 'Si', 'Ge', 'Sn', 'Pb', 'B', 'Al', 'Ga'):
            assert added in _HETEROATOM_SENIORITY, (
                f"P-18(b) regression: {added} missing from _HETEROATOM_SENIORITY. "
                f"Phase 137 ERRATA-01 silently reverted. AUDIT REQUIRED."
            )

    def test_heteroatom_variety_order_locked_157(self) -> None:
        """_HETEROATOM_VARIETY_ORDER is consistent with _HETEROATOM_SENIORITY (P-18(b) FR-2.3 (e))."""
        from orthonym.rules.ring_selection import (
            _HETEROATOM_SENIORITY, _HETEROATOM_VARIETY_ORDER,
        )

        # The variety order must contain exactly the same 20 elements as the
        # seniority dict --- it's the FR-2.3 (e) "greater heteroatom variety"
        # tuple-position mapping per ring_selection.py:88-94.
        assert len(_HETEROATOM_VARIETY_ORDER) == 20, (
            f"P-18(b) regression: VARIETY_ORDER size "
            f"{len(_HETEROATOM_VARIETY_ORDER)} != 20. AUDIT REQUIRED."
        )
        assert set(_HETEROATOM_VARIETY_ORDER) == set(_HETEROATOM_SENIORITY.keys()), (
            f"P-18(b) regression: VARIETY_ORDER element set differs from "
            f"SENIORITY keys. AUDIT REQUIRED."
        )
        # Descending-seniority order: VARIETY_ORDER[i] > VARIETY_ORDER[i+1]:
        for i in range(len(_HETEROATOM_VARIETY_ORDER) - 1):
            curr = _HETEROATOM_VARIETY_ORDER[i]
            nxt = _HETEROATOM_VARIETY_ORDER[i + 1]
            assert _HETEROATOM_SENIORITY[curr] > _HETEROATOM_SENIORITY[nxt], (
                f"P-18(b) regression: VARIETY_ORDER position {i}: {curr} "
                f"({_HETEROATOM_SENIORITY[curr]}) is not more senior than "
                f"{nxt} ({_HETEROATOM_SENIORITY[nxt]}). AUDIT REQUIRED."
            )


class TestEZThresholdLock157:
    """P-31.1.3 regression: E/Z stereo threshold preserved at `< 8` (strict).

    Phase 137-01 ERRATA-02 changed `min_ring_size <= 8` to `< 8` per
    BBerrors.html P-31.1.3 Sep 2024 errata + P-44.4.1. Plan listed
    line 91; live location is line 126 (drift recorded as non-blocking
    citation correction in 157-AUDIT.md row 2). The COMPARISON OPERATOR
    is the regression-lock invariant; the line number is illustrative.
    """

    def test_ez_threshold_locked_157(self) -> None:
        """E/Z threshold operator must be `< 8` (NOT `<= 8`) for ring stereo skip."""
        from orthonym.rules import stereochemistry

        # Read source rather than execute --- the comparison is inside
        # an internal loop; static check is the cheapest invariant.
        source_path = Path(stereochemistry.__file__)
        text = source_path.read_text(encoding="utf-8")
        # The strict-less-than form must be present:
        assert "min_ring_size < 8" in text, (
            "P-31.1.3 regression: `min_ring_size < 8` invariant missing from "
            "stereochemistry.py. Phase 137 ERRATA-02 silently reverted. "
            "AUDIT REQUIRED."
        )
        # The relaxed form must NOT be present (the Sep 2024 errata removed it):
        assert "min_ring_size <= 8" not in text, (
            "P-31.1.3 regression: `min_ring_size <= 8` reappeared in "
            "stereochemistry.py. Phase 137 ERRATA-02 reverted. AUDIT REQUIRED."
        )


class TestFusionSeparatorLock157:
    """P-25.3.8.3 regression: fusion descriptor `:` for multi-edge fusion.

    Phase 137 ERRATA-05 verified the multi-edge fusion branch returns
    `[loc1,loc2-letter:loc3,loc4-letter']` with `:` separator (NOT `;`).
    The plan listed line 860; live location is line 866 (drift recorded
    in 157-AUDIT.md row 3 + S1.1 row 3). The multi-component branch at
    line 859 uses `,` for dibenzo/dinaphtho per a distinct sub-rule;
    both are correct.
    """

    def test_fusion_separator_locked_157(self) -> None:
        """Multi-edge fusion uses `:` separator per P-25.3.8.3 Dec 2019 errata."""
        from orthonym.rules import fusion_descriptors

        source_path = Path(fusion_descriptors.__file__)
        text = source_path.read_text(encoding="utf-8")
        # Multi-edge fusion is the `:`-join branch at line 866:
        assert "':'.join(parts)" in text, (
            "P-25.3.8.3 regression: `':'.join(parts)` missing from "
            "fusion_descriptors.py. Phase 137 ERRATA-05 silently reverted. "
            "AUDIT REQUIRED."
        )
        # The deprecated semicolon form must NOT appear:
        assert "';'.join(parts)" not in text, (
            "P-25.3.8.3 regression: `';'.join(parts)` reappeared in "
            "fusion_descriptors.py. Phase 137 ERRATA-05 reverted. "
            "AUDIT REQUIRED."
        )


class TestHWCorrectionsLock157:
    """P-22.2.1 regression: Hantzsch-Widman name corrections preserved.

    Phase 137-01 ERRATA-06 verified `thiazolidine` (NOT `thioxazolidine`)
    in retained-names dict at retained_names.py:237. Multiple HW spellings
    were checked; the canonical `C1CSCN1 -> thiazolidine` is the regression-
    lock sentinel.
    """

    def test_hw_corrections_locked_157(self) -> None:
        """thiazolidine (NOT thioxazolidine) preserved at retained_names.py:237."""
        from orthonym.data.retained_names import RETAINED_NAMES

        # Canonical sentinel: 1,3-thiazolidine = "C1CSCN1" -> "thiazolidine"
        assert RETAINED_NAMES.get("C1CSCN1") == "thiazolidine", (
            "P-22.2.1 regression: `C1CSCN1` does not map to `thiazolidine`. "
            f"Got: {RETAINED_NAMES.get('C1CSCN1')!r}. Phase 137 ERRATA-06 "
            "silently reverted. AUDIT REQUIRED."
        )
        # The deprecated form must NOT appear in any value:
        for smiles, name in RETAINED_NAMES.items():
            assert "thioxazolidine" not in name, (
                f"P-22.2.1 regression: deprecated 'thioxazolidine' appeared "
                f"in RETAINED_NAMES[{smiles!r}] = {name!r}. AUDIT REQUIRED."
            )

    def test_isothiazolidine_locked_157(self) -> None:
        """isothiazolidine (1,2-isomer) preserved at retained_names.py:239."""
        from orthonym.data.retained_names import RETAINED_NAMES

        # Canonical sentinel: 1,2-isothiazolidine = "C1CNSC1" -> "isothiazolidine"
        # (S at 1, N at 2 -- adjacent). Distinct from thiazolidine (1,3-isomer).
        assert RETAINED_NAMES.get("C1CNSC1") == "isothiazolidine", (
            "P-22.2.1 regression: `C1CNSC1` does not map to `isothiazolidine`. "
            f"Got: {RETAINED_NAMES.get('C1CNSC1')!r}. Phase 137 ERRATA-06 "
            "verified the 1,2-isomer; failure indicates silent reversion. "
            "AUDIT REQUIRED."
        )


class TestSymmetricAnhydrideLock157:
    """P-65.7 regression: symmetric anhydride uses bare `anhydride` (no `bis-`).

    Phase 137-01 ERRATA-08 verified `anhydrides.py:109` returns
    `f"{acid1_name} anhydride"` (NOT `f"bis-{acid1_name}..."`) when
    acid1 == acid2. Source: BBerrors.html P-65.7.
    """

    def test_symmetric_anhydride_locked_157(self) -> None:
        """Symmetric anhydride: `acid1_name + ' anhydride'` (no `bis-` prefix)."""
        from orthonym.rules import anhydrides

        source_path = Path(anhydrides.__file__)
        text = source_path.read_text(encoding="utf-8")
        # The symmetric-anhydride branch must use the bare form:
        assert 'f"{acid1_name} anhydride"' in text, (
            "P-65.7 regression: `f\"{acid1_name} anhydride\"` missing from "
            "anhydrides.py. Phase 137 ERRATA-08 silently reverted. "
            "AUDIT REQUIRED."
        )
        # The forbidden `bis-` prefix on the symmetric branch must not appear.
        # We grep the source for any line that combines `bis-` with `anhydride`
        # in the symmetric branch context. The bare-grep here is a coarse but
        # safe regression-lock --- the file is small and contains zero `bis-`
        # by design (per audit S1.1 row 5).
        forbidden = re.search(r"bis-\{acid1_name", text)
        assert forbidden is None, (
            "P-65.7 regression: `bis-` prefix appeared in symmetric anhydride "
            "branch of anhydrides.py. Phase 137 ERRATA-08 reverted. "
            "AUDIT REQUIRED."
        )

    def test_symmetric_anhydride_canary_locked_157(self) -> None:
        """P-65.7 smoke: ethanoic acid anhydride canary returns symmetric form."""
        from orthonym.namer import name_compound

        # Canonical canary from anhydrides.py:10:
        # CC(=O)OC(=O)C  -> ethanoic anhydride (or acetic anhydride retained name)
        name = name_compound("CC(=O)OC(=O)C")
        assert name is not None, (
            "P-65.7 smoke: name_compound('CC(=O)OC(=O)C') returned None. "
            "AUDIT REQUIRED."
        )
        # Per Phase 137 ERRATA-08: the symmetric form has no `bis-` prefix.
        # Either the systematic ('ethanoic anhydride') or the retained
        # ('acetic anhydride') form must NOT contain `bis-`:
        assert "bis-" not in name.lower(), (
            f"P-65.7 regression: name {name!r} for symmetric ethanoic "
            f"anhydride contains forbidden `bis-` prefix. Phase 137 "
            f"ERRATA-08 reverted. AUDIT REQUIRED."
        )


class TestCarbonochloridoylLock157:
    """P-29.1.2 regression: chlorocarbonyl -> carbonochloridoyl.

    CRITICAL per CONTEXT Pitfall 3: BBerrors.html P-29.1.2 Nov 2020 errata
    renamed ONLY -COCl from `chlorocarbonyl` to `carbonochloridoyl`. The
    bromine and fluorine analogs (-COBr, -COF) were INTENTIONALLY UNCHANGED.
    This test asserts BOTH the change AND the non-change as the regression-
    lock invariants.

    CRITICAL per CONTEXT Pitfall 7: this test does NOT invoke OPSIN round-
    trip on `carbonochloridoyl` because OPSIN v2.9.0 does not parse it.
    The dict-import assertion is the canonical regression-lock; OPSIN-side
    vocabulary closure is a v19 follow-up to OPSIN maintainers.

    Source: 157-AUDIT.md row 6 + S1.1 row 6 grep transcript.
    """

    def test_carbonochloridoyl_locked_157(self) -> None:
        """acid_chloride -> carbonochloridoyl; bromo/fluoro intentionally unchanged."""
        from orthonym.rules.seniority import PREFIX_FORMS

        # The Nov 2020 errata renamed -COCl prefix:
        assert PREFIX_FORMS.get("acid_chloride") == "carbonochloridoyl", (
            f"P-29.1.2 regression: PREFIX_FORMS['acid_chloride'] = "
            f"{PREFIX_FORMS.get('acid_chloride')!r} != 'carbonochloridoyl'. "
            f"Phase 137 ERRATA-09 silently reverted. AUDIT REQUIRED."
        )
        # Per CONTEXT Pitfall 3: the bromine analog is INTENTIONALLY unchanged.
        # If this assertion fires, an over-eager edit applied carbonochloridoyl-
        # style renaming to ALL acid halides --- which is WRONG per BBerrors.
        assert PREFIX_FORMS.get("acid_bromide") == "bromocarbonyl", (
            f"P-29.1.2 regression: PREFIX_FORMS['acid_bromide'] = "
            f"{PREFIX_FORMS.get('acid_bromide')!r} != 'bromocarbonyl'. "
            f"BBerrors.html P-29.1.2 Nov 2020 errata applies ONLY to -COCl. "
            f"The -COBr form was INTENTIONALLY unchanged. AUDIT REQUIRED."
        )
        # Per CONTEXT Pitfall 3: the fluorine analog is also INTENTIONALLY unchanged.
        assert PREFIX_FORMS.get("acid_fluoride") == "fluorocarbonyl", (
            f"P-29.1.2 regression: PREFIX_FORMS['acid_fluoride'] = "
            f"{PREFIX_FORMS.get('acid_fluoride')!r} != 'fluorocarbonyl'. "
            f"BBerrors.html P-29.1.2 Nov 2020 errata applies ONLY to -COCl. "
            f"The -COF form was INTENTIONALLY unchanged. AUDIT REQUIRED."
        )

    def test_carbonochloridoyl_benzene_dispatch_locked_157(self) -> None:
        """benzene.py:66 prefix-mapping mirrors seniority.py:282 (carbonochloridoyl)."""
        from orthonym.rules import benzene

        source_path = Path(benzene.__file__)
        text = source_path.read_text(encoding="utf-8")
        # Phase 137 ERRATA-09 also touched benzene.py at line 66; the form
        # is `'carbonyl chloride': 'carbonochloridoyl',`. Static check.
        assert "'carbonochloridoyl'" in text, (
            "P-29.1.2 regression: `'carbonochloridoyl'` missing from benzene.py. "
            "Phase 137 ERRATA-09 silently reverted. AUDIT REQUIRED."
        )


class TestBracketNestingSubsectionsLock157:
    """P-16.5.4.1 regression: 5 bracket-nesting subsections preserved.

    Phase 137-02 ERRATA-07 implemented `compute_nesting_depth()` +
    `apply_enclosing_marks()` per BBerrors.html P-16.5.4.1 Dec 2025 errata.
    Both functions must remain importable and exhibit the documented
    invariants (depth-0 fusion brackets, depth-1 stereo descriptors,
    auto-detect sentinel `depth=-1`).
    """

    def test_bracket_nesting_subsections_locked_157(self) -> None:
        """compute_nesting_depth + apply_enclosing_marks importable + functional."""
        from orthonym.assembly.naming_utils import (
            apply_enclosing_marks,
            compute_nesting_depth,
        )

        # Both callables present:
        assert callable(compute_nesting_depth), (
            "P-16.5.4.1 regression: compute_nesting_depth is not callable."
        )
        assert callable(apply_enclosing_marks), (
            "P-16.5.4.1 regression: apply_enclosing_marks is not callable."
        )
        # P-16.5.4.1.2 invariant: fusion brackets do NOT count:
        assert compute_nesting_depth("bicyclo[2.2.1]heptane") == 0, (
            "P-16.5.4.1.2 regression: bicyclo brackets counted as nesting. "
            "AUDIT REQUIRED."
        )
        # P-16.5.4.1.3 invariant: stereo descriptors DO count:
        assert compute_nesting_depth("(R)-butan-2-yl") >= 1, (
            "P-16.5.4.1.3 regression: stereo descriptor (R) not counted as "
            "nesting. AUDIT REQUIRED."
        )
        # Auto-detect sentinel `depth=-1` smoke:
        result = apply_enclosing_marks("methyl", -1)
        assert result is not None and isinstance(result, str), (
            "P-16.5.4.1 regression: apply_enclosing_marks(name, -1) returned "
            f"{result!r}. AUDIT REQUIRED."
        )


class TestFirstSubstituentNoMarksLock157:
    """P-16.5.1.3 regression: first-substituent-no-marks for mononuclear hydrides.

    Phase 137-02 ERRATA-10 implemented `_build_substituent_string()` for
    phosphane prefix assembly per BBerrors.html P-16.5.1.3 May 2021 errata.
    The first substituent gets NO enclosing marks; subsequent substituents
    do. Smoke-test on a typical phosphane prefix list.
    """

    def test_first_substituent_no_marks_locked_157(self) -> None:
        """_build_substituent_string callable; first substituent has no marks."""
        from orthonym.rules.phosphorus import _build_substituent_string

        # Single substituent --- no enclosing marks:
        single = _build_substituent_string(["methyl"])
        assert single == "methyl", (
            f"P-16.5.1.3 regression: single substituent got marks: {single!r}. "
            "Expected 'methyl'. AUDIT REQUIRED."
        )
        # Two substituents --- first bare, second wrapped per P-16.5.1.3:
        pair = _build_substituent_string(["methyl", "ethyl"])
        # The first must NOT begin with `[` or `(`:
        assert not pair.startswith("("), (
            f"P-16.5.1.3 regression: first substituent in pair got leading `(`: "
            f"{pair!r}. Expected first substituent unmarked. AUDIT REQUIRED."
        )
        assert not pair.startswith("["), (
            f"P-16.5.1.3 regression: first substituent in pair got leading `[`: "
            f"{pair!r}. Expected first substituent unmarked. AUDIT REQUIRED."
        )


class TestP25224SeniorityBandsLock157:
    """P-25.2.2.4 regression: 4 corrected benzo-heterocycle seniority bands.

    Phase 155.C ERRATA-25224 corrected 4 entries in MONOCYCLIC_COMPONENTS
    per BBerrors.html P-25.2.2.4 Jan 2022 errata. D-12 mutual-exclusion:
    Phase 157 makes ZERO modifications to fusion_components.py; this test
    is the regression-lock asserting the 4 corrected values stay at their
    post-155.C numbers.

    Sentinel values per 155-AUDIT-C.md + 157-AUDIT.md row 9 + S1.1 row 9:
      imidazole = 50  (was 47 pre-155.C)
      pyrazole  = 51  (was 48 pre-155.C)
      thiazole  = 80  (was 77 pre-155.C)
      isothiazole = 81 (was 78 pre-155.C)
    """

    def test_p25224_seniority_bands_locked_157(self) -> None:
        """imidazole=50, pyrazole=51, thiazole=80, isothiazole=81 preserved."""
        from orthonym.data.fusion_components import MONOCYCLIC_COMPONENTS

        expected = {
            "imidazole": 50,
            "pyrazole": 51,
            "thiazole": 80,
            "isothiazole": 81,
        }
        for name, expected_seniority in expected.items():
            entry = MONOCYCLIC_COMPONENTS.get(name)
            assert entry is not None, (
                f"P-25.2.2.4 regression: {name!r} missing from "
                f"MONOCYCLIC_COMPONENTS. Phase 155.C silently reverted. "
                f"AUDIT REQUIRED."
            )
            actual = entry.get("seniority")
            assert actual == expected_seniority, (
                f"P-25.2.2.4 regression: {name}.seniority = {actual} != "
                f"{expected_seniority}. Phase 155.C correction silently "
                f"reverted. AUDIT REQUIRED."
            )

    def test_p25224_seniority_invariants_locked_157(self) -> None:
        """Phase 155.C invariants: imidazole < pyrazole < pyrrole; thiazole < thiophene."""
        from orthonym.data.fusion_components import MONOCYCLIC_COMPONENTS

        # Per 155-AUDIT-C.md cite-blocks: lower seniority number = MORE senior
        # is FALSE in this catalog --- higher number = MORE senior.
        # Invariants documented at fusion_components.py:110-112 + :123-126:
        #   imidazole(50) < pyrrole(55) preserved (pyrrole more senior than imidazole)
        #   pyrazole(51) < pyrrole(55) preserved
        #   pyrazole(51) > imidazole(50) preserved (pyrazole more senior)
        # Per fusion_components.py:172-174:
        #   isothiazole(81) < thiophene(85) preserved
        #   isothiazole(81) > thiazole(80) preserved (... wait, isothiazole=81 > thiazole=80
        #   so isothiazole is MORE senior; the comment claims "thiazole more senior"
        #   --- this is the inverted-numeric convention where higher rank = more senior).
        # Smoke: the 4 numbers form the expected ordering.
        comp = MONOCYCLIC_COMPONENTS
        # Pyrazole more senior than imidazole (51 > 50):
        assert comp["pyrazole"]["seniority"] > comp["imidazole"]["seniority"]
        # Isothiazole more senior than thiazole (81 > 80):
        assert comp["isothiazole"]["seniority"] > comp["thiazole"]["seniority"]
        # Thiazole / isothiazole > imidazole / pyrazole (S>=N tier):
        assert comp["thiazole"]["seniority"] > comp["pyrazole"]["seniority"]
        assert comp["isothiazole"]["seniority"] > comp["imidazole"]["seniority"]


class TestP2543222BridgeNumberingLock157:
    """P-25.4.3.2.2 regression: Phase 151 cite-line preserved.

    Per 157-AUDIT.md S4.5 verdict NO-OP-CONFIRMED: Phase 151 already
    complies with the Dec 2019 P-25.4.3.2.2 revised rule because Phase
    151 shipped post-Dec-2019 and reads current Blue Book.

    The protective regression-lock invariants are:
      (a) polycyclic.py:18 docstring cites "P-23, VB-1 through VB-9"
          (the Blue Book post-Dec-2019 form per 157-AUDIT.md S4.2).
      (b) Bridge-numbering canary smoke: norbornane round-trips and the
          [2.2.1] bracket descriptor appears in substituted-bicyclic output.

    Per CONTEXT D-14 mutual-exclusion: Phase 157 makes ZERO modifications
    to polycyclic.py / bicyclo.py / ring_assemblies.py. Failure of these
    smoke assertions BLOCKS Phase 157 + triggers a Phase 151 audit.
    """

    def test_p2543222_polycyclic_cite_locked_157(self) -> None:
        """polycyclic.py:18 docstring cites `P-23, VB-1 through VB-9`."""
        from orthonym.rules import polycyclic

        source_path = Path(polycyclic.__file__)
        text = source_path.read_text(encoding="utf-8")
        # The cite asserts current-Blue-Book conformance per 157-AUDIT.md S4.2:
        assert "P-23" in text, (
            "P-25.4.3.2.2 regression: polycyclic.py docstring missing `P-23` "
            "cite. Phase 151 cite-block reverted. AUDIT REQUIRED."
        )
        assert "VB-1" in text and "VB-9" in text, (
            "P-25.4.3.2.2 regression: polycyclic.py docstring missing "
            "`VB-1` / `VB-9` cite. Phase 151 cite-block reverted. AUDIT REQUIRED."
        )

    @pytest.mark.parametrize("smiles,expected_token,case_label", [
        ("C1CC2CCC1C2", ["norbornane", "bicyclo[2.2.1]heptane"],
         "PubChem norbornane (the bicyclo[2.2.1]heptane retained name)"),
        ("NC1(C(=O)O)CC2CCC1C2", ["bicyclo[2.2.1]"],
         "CHEBI:167508 amino-bicyclo skeleton (bracket descriptor invariant)"),
    ])
    def test_p2543222_bridge_numbering_smoke_locked_157(
        self, smiles: str, expected_token: list, case_label: str,
    ) -> None:
        """P-25.4.3.2.2 smoke: bridge-descriptor invariant on canary fixtures."""
        from orthonym.namer import name_compound

        name = name_compound(smiles)
        assert name is not None, (
            f"P-25.4.3.2.2 smoke: name_compound({smiles!r}) returned None. "
            f"{case_label}."
        )
        # At least one of the expected tokens (the retained name OR the
        # bracket descriptor) must be present:
        ok = any(tok in name for tok in expected_token)
        assert ok, (
            f"P-25.4.3.2.2 smoke: name {name!r} for {smiles!r} contains none "
            f"of {expected_token}. {case_label}. Phase 151 bridge-numbering "
            f"may have regressed. AUDIT REQUIRED."
        )


class TestP2581QuinolizineLock157:
    """P-25.8.1 regression: NO-OP-CONFIRMED protective infrastructure.

    Per 157-AUDIT.md S3.4 verdict NO-OP-CONFIRMED: no comparison path in
    the v18 codebase ever pits quinolizine vs quinoline / isoquinoline at
    the same selection level. P-25.8.1 (Aug 2021 errata) is NOT-APPLICABLE-
    IN-CURRENT-SCOPE per CONTEXT D-08 + 157-02-SUMMARY.md "Conditional
    code state".

    The protective regression-lock invariants are:
      (a) FUSED_HETEROCYCLE_DATA has NO `principal_seniority_rank` field
          on any of the 3 entries (quinoline, isoquinoline, 4H-quinolizine).
      (b) The S3.5 FIX-SPEC payload preserved in the audit doc remains the
          spec for any v19 follow-up.

    If a future v19 phase introduces multi-fused-system principal-ring
    competition involving these three rings, this test fires the moment
    `principal_seniority_rank` appears on any entry --- at which point the
    rule MUST be applied per the FIX-SPEC, AND a regression-lock value
    test (rank=2 for quinoline, =3 for isoquinoline, =1 for 4H-quinolizine)
    MUST be added.

    Source: 157-AUDIT.md S3.4 + S3.5 FIX-SPEC payload.
    Source: 157-02-SUMMARY.md "Plan-03 Inputs" -> P-25.8.1 conditional state.
    """

    def test_p25_8_1_quinolizine_no_op_confirmed_locked_157(self) -> None:
        """No `principal_seniority_rank` field on any FUSED_HETEROCYCLE_DATA entry."""
        from orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA

        # The 3 P-25.8.1 SMILES keys per 157-CONTEXT.md S Code Examples 3:
        quinoline_key = "c1ccc2ncccc2c1"
        isoquinoline_key = "c1ccc2cnccc2c1"
        quinolizine_key = "C1=CCN2C=CC=CC2=C1"

        for key, name in (
            (quinoline_key, "quinoline"),
            (isoquinoline_key, "isoquinoline"),
            (quinolizine_key, "4H-quinolizine"),
        ):
            entry = FUSED_HETEROCYCLE_DATA.get(key)
            assert entry is not None, (
                f"P-25.8.1 regression: {name} ({key!r}) missing from "
                f"FUSED_HETEROCYCLE_DATA. AUDIT REQUIRED."
            )
            rank = entry.get("principal_seniority_rank")
            # Per Plan-01 NO-OP-CONFIRMED: this field is intentionally absent.
            # If a future v19 phase adds it, this test FIRES + the v19 phase
            # MUST also add value-assertion tests (1/2/3 per FIX-SPEC S3.5).
            assert rank is None, (
                f"P-25.8.1 regression: {name} entry has "
                f"principal_seniority_rank={rank}. Plan-01 NO-OP-CONFIRMED "
                f"verdict invalidated. The v19 phase introducing this field "
                f"MUST also ship a value-assertion regression-lock test per "
                f"157-AUDIT.md S3.5 FIX-SPEC (quinolizine=1, quinoline=2, "
                f"isoquinoline=3). AUDIT REQUIRED."
            )

    def test_p25_8_1_no_seniority_field_globally_locked_157(self) -> None:
        """Zero entries in FUSED_HETEROCYCLE_DATA carry `principal_seniority_rank`."""
        from orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA

        with_field = [
            key for key, val in FUSED_HETEROCYCLE_DATA.items()
            if "principal_seniority_rank" in val
        ]
        # Per 157-AUDIT.md S3.2 Trace C grep: zero entries carry the field.
        assert len(with_field) == 0, (
            f"P-25.8.1 regression: {len(with_field)} entries in "
            f"FUSED_HETEROCYCLE_DATA have a `principal_seniority_rank` field "
            f"(Plan-01 verdict NO-OP-CONFIRMED expected zero). Affected keys: "
            f"{with_field}. The v19 phase that introduced this field MUST "
            f"also ship the value-assertion regression-lock test per S3.5 "
            f"FIX-SPEC. AUDIT REQUIRED."
        )

    def test_p25_8_1_quinolizine_smoke_locked_157(self) -> None:
        """Smoke: 4H-quinolizine SMILES round-trips to its retained name."""
        from orthonym.namer import name_compound

        # Canary from 157-AUDIT.md S3.3 fixture 5:
        name = name_compound("C1=CCN2C=CC=CC2=C1")
        assert name is not None
        assert "quinolizine" in name.lower(), (
            f"P-25.8.1 smoke: name_compound('C1=CCN2C=CC=CC2=C1') = {name!r} "
            f"missing `quinolizine` token. The retained-name lookup at "
            f"data/fused_heterocycles.py:1093 may have regressed. AUDIT REQUIRED."
        )


# ============================================================================
# Taxonomy 2: Re-verification meta-tests (NA classifications)
# Suffix: _not_applicable_grep_157
# ============================================================================


def test_dehydro_not_applicable_grep_157() -> None:
    """P-31.2 NA re-verification: only OPSIN-imported retained-name hits.

    Per 157-AUDIT.md S5.1: `grep -rn "dehydro" src/orthonym/ | grep -v
    __pycache__` returns exactly 3 hits, all in `data/opsin_imports/`:
      - amino_acids_opsin.py:72 -- 'dehydroalan'
      - amino_acids_opsin.py:324 -- 'dehydrophenylalan'
      - carbohydrates_opsin.py:1332 -- 'dehydroascorbic acid'

    These are static OPSIN-imported retained-name strings looked up by
    SMILES match, NOT generated as composable prefixes.

    If this test fails, dehydro-prefix generation has been introduced
    somewhere in src/orthonym/rules or src/orthonym/assembly. The P-31.2
    NA classification no longer holds. Phase 157 audit doc and
     MUST be updated to APPLY status with the
    dehydro-before-hydro ordering rule enforced at the prefix-assembly site.

    Source: 157-CONTEXT.md D-07 + 157-AUDIT.md S5.1.
    """
    src_dir = _repo_root() / "src" / "orthonym"
    result = subprocess.run(
        ["grep", "-rn", "dehydro", str(src_dir)],
        capture_output=True, text=True, timeout=10,
    )
    # Filter out pycache + binary cache hits:
    hits = [
        line for line in result.stdout.strip().split("\n")
        if line and "__pycache__" not in line
    ]
    assert len(hits) == 3, (
        f"P-31.2 NA re-verification: hit count {len(hits)} != 3. "
        f"NA classification may have changed. AUDIT REQUIRED.\n"
        f"Hits:\n" + "\n".join(hits)
    )
    for hit in hits:
        assert "data/opsin_imports/" in hit, (
            f"P-31.2 NA re-verification: dehydro hit OUTSIDE OPSIN imports!\n"
            f"  {hit}\n"
            f"Dehydro-prefix generation may have been introduced.\n"
            f"P-31.2 ordering rule MUST now be applied; see CONTEXT D-07 + "
            f" S P-31.2."
        )


def test_compound_locants_not_applicable_grep_157() -> None:
    """P-31.1.4.3 NA re-verification: only comments + token-name strings.

    Per 157-AUDIT.md S5.2: `grep -rn "compound.locant\\|compound_locant"
    src/orthonym/` returns 3 hits, all non-generation:
      - validation/opsin_grammar.py:79 -- token-name STRING (Phase 156 grammar)
      - assembly/naming_utils.py:417 -- DOC-STRING comment citing P-16.5.4.1.3
      - assembly/composer.py:3966 -- COMMENT line citing P-31.1.4.1

    If this test fails, compound-locant generation has been introduced and
    the P-31.1.4.3 priority rule MUST now be applied to the locant-comparison
    cascade.

    Source: 157-CONTEXT.md D-07 + 157-AUDIT.md S5.2.
    """
    src_dir = _repo_root() / "src" / "orthonym"
    result = subprocess.run(
        ["grep", "-rnE", r"compound[._]locant", str(src_dir)],
        capture_output=True, text=True, timeout=10,
    )
    hits = [
        line for line in result.stdout.strip().split("\n")
        if line and "__pycache__" not in line
    ]
    # Per 157-AUDIT.md S5.2: hits must be in validation/opsin_grammar.py
    # (Phase 156 token-name string), composer.py:3966 (comment), or
    # naming_utils.py:417 (doc-string). NEVER in rules/ generation paths.
    for hit in hits:
        allowed = (
            "validation/opsin_grammar.py" in hit
            or "assembly/composer.py" in hit
            or "assembly/naming_utils.py" in hit
        )
        assert allowed, (
            f"P-31.1.4.3 NA re-verification: compound-locant hit in "
            f"unexpected location!\n  {hit}\n"
            f"Compound-locant comparison logic may have been introduced.\n"
            f"P-31.1.4.3 priority rule MUST now be applied; see CONTEXT D-07."
        )
        # No hit in src/orthonym/rules/ (generation logic forbidden):
        assert "/rules/" not in hit, (
            f"P-31.1.4.3 NA re-verification: hit in rules/ generation logic!\n  {hit}"
        )


# ============================================================================
# Taxonomy 4: Tracking-doc smoke tests (G6 acceptance gate)
# Suffix: _doc_*_157
# ============================================================================


# 14 expected rule names per CONTEXT D-09 line 300-315 + 157-AUDIT.md S6.3:
EXPECTED_RULES_157 = [
    "P-18(b)", "P-31.1.3", "P-25.3.8.3", "P-22.2.1", "P-65.7",
    "P-29.1.2", "P-16.5.4.1", "P-16.5.1.3",
    "P-25.2.2.4", "P-25.4.3.2.2", "P-25.8.1",
    "P-31.2", "P-31.1.4.3", "P-13.3.5",
]


def _doc_path_157() -> Path:
    """Return the canonical path to """
    return _repo_root() / "docs" / "iupac_errata_applied.md"


def test_iupac_errata_applied_doc_complete_157() -> None:
    """G6 gate:  >= 250 LOC + 14 PIN rule entries."""
    doc = _doc_path_157()
    assert doc.exists(), (
        " missing -- Plan-02 incomplete. "
        "V18 S6 line 1374 milestone-blocking deliverable. AUDIT REQUIRED."
    )
    text = doc.read_text(encoding="utf-8")
    line_count = text.count("\n") + 1
    assert line_count >= 250, (
        f" too short: {line_count} < 250 LOC. "
        f"G6 acceptance gate fails. AUDIT REQUIRED."
    )
    for rule in EXPECTED_RULES_157:
        assert rule in text, (
            f" missing rule entry: {rule!r}. "
            f"G6 acceptance gate fails. AUDIT REQUIRED."
        )


def test_iupac_errata_applied_doc_section_headings_157() -> None:
    """G6 gate sub-check: 6 section headings present in expected order."""
    doc = _doc_path_157()
    text = doc.read_text(encoding="utf-8")
    expected_headings = [
        "## §1.", "## §2.", "## §3.", "## §4.", "## §5.", "## §6.",
    ]
    last_pos = -1
    for heading in expected_headings:
        pos = text.find(heading)
        assert pos != -1, (
            f" missing section heading "
            f"{heading!r}. G6 gate sub-check fails."
        )
        assert pos > last_pos, (
            f" section heading {heading!r} "
            f"out of order at pos {pos} (previous heading ended at {last_pos}). "
            f"G6 gate sub-check fails."
        )
        last_pos = pos


def test_iupac_errata_applied_doc_summary_table_consistent_157() -> None:
    """CONTEXT Pitfall 5: §5 summary table has >= 14 PIN compliance rows.

    Each summary-table row MUST have a matching §2 cite-block heading.
    The table format is `| P-X.Y.Z | <status> | <phase> | <test> |`.
    """
    doc = _doc_path_157()
    text = doc.read_text(encoding="utf-8")
    # Match summary-table rows like '| P-X.Y.Z | applied | ...':
    row_pattern = re.compile(
        r"^\|\s*(P-[A-Za-z0-9\.\(\)]+)\s*\|\s*([^|]+?)\s*\|", re.MULTILINE
    )
    rows = row_pattern.findall(text)
    assert len(rows) >= 14, (
        f" S5 summary table has {len(rows)} rows; "
        f">= 14 required per CONTEXT D-09 + 157-AUDIT.md S6.3. G6 fails."
    )
    # Per Pitfall 5: every §2 entry should appear in the table:
    found_rules = {rule for rule, _status in rows}
    for expected in EXPECTED_RULES_157:
        # The table may store P-18(b) as `P-18(b)`; we accept exact match.
        assert expected in found_rules, (
            f" S5 summary table missing rule "
            f"{expected!r}. Found rules: {sorted(found_rules)}. G6 fails."
        )


def test_iupac_errata_applied_doc_status_consistency_157() -> None:
    """CONTEXT Pitfall 5: §5 table Status drift detection vs §2 cite-blocks.

    For each rule in the §5 summary table, locate the §2 cite-block heading
    `### P-X.Y.Z` and verify the Status keyword present in the table also
    appears in the cite-block (substring match, case-insensitive).

    This guards against drift where §5 says `applied` but §2 says
    `not-applicable` (or vice versa), which would invalidate the public
    PIN compliance claim.

    Source: 157-CONTEXT.md Pitfall 5 + 157-02-SUMMARY.md key-decisions
    "§5 table Status ↔ §2 cite-block Status invariant".
    """
    doc = _doc_path_157()
    text = doc.read_text(encoding="utf-8")
    # Permitted Status keywords per CONTEXT D-09 + 157-AUDIT.md Audit Summary:
    status_keywords = {
        "applied", "re-verified", "already-correct",
        "not-applicable", "out-of-scope", "deferred-to-v19",
        "no-op-confirmed", "not-applicable-in-current-scope",
    }
    # Match §5 summary-table rows:
    row_pattern = re.compile(
        r"^\|\s*(P-[A-Za-z0-9\.\(\)]+)\s*\|\s*([^|]+?)\s*\|", re.MULTILINE
    )
    rows = row_pattern.findall(text)
    for rule, status_in_table in rows:
        status_in_table_lower = status_in_table.strip().lower()
        # Locate the §2 cite-block heading:
        heading_pos = text.find(f"### {rule}")
        if heading_pos == -1:
            # Some entries (P-13.3.5 OUT-OF-SCOPE) may not have a full §2 entry
            # if the doc places them only in the summary table per CONTEXT
            # D-09 line 315 ("(no test — out of scope)"). Skip those.
            continue
        # Slice the §2 block (until the next ### heading or EOF):
        block_end = text.find("\n### ", heading_pos + 1)
        block = text[heading_pos:block_end if block_end != -1 else len(text)]
        # At least one Status keyword from the table must also appear in the §2 block:
        # We accept the keyword in either case (the table tends toward lower-case,
        # the cite-block may use upper-case for the Status field).
        found_match = False
        for keyword in status_keywords:
            # The table entry may be 're-verified (NO-OP-CONFIRMED)' --- check
            # whether ANY recognized keyword from the table appears in §2:
            if keyword in status_in_table_lower:
                if keyword in block.lower():
                    found_match = True
                    break
        # If no keyword matched on either side, just skip --- we cannot infer.
        # If the table had a keyword but the §2 block has NONE of the keywords,
        # there is drift.
        table_keywords_present = [
            k for k in status_keywords if k in status_in_table_lower
        ]
        if table_keywords_present:
            assert found_match, (
                f" status drift for {rule!r}: "
                f"S5 table says {status_in_table!r} (keyword(s) "
                f"{table_keywords_present!r}); the corresponding S2 cite-block "
                f"contains none of those keywords. Per Pitfall 5 this is a "
                f"public-PIN-compliance-claim invalidation. AUDIT REQUIRED."
            )
