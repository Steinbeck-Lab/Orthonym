"""Phase 163 Tier FRN-A..E seniority entries unit tests.

Asserts the 14 new SENIORITY_ORDER + SUFFIX_FORMS + PREFIX_FORMS entries
shipped in Plan-03 commit 163-03-01 are correctly placed AND that LOCKED
PIN suffix forms match AUDIT § 5 decisions.

Test pyramid per CONTEXT D-12 + RESEARCH §8.2:
- Section A: SUFFIX_FORMS entries (chain_terminal + ring_attached forms)
- Section B: PREFIX_FORMS entries (non-principal-group prefix names)
- Section C: SENIORITY_ORDER position invariants
- Section D: no regression on existing thio* entries

Total: >= 20 tests.

References:
- src/orthonym/rules/seniority.py SENIORITY_ORDER + SUFFIX_FORMS + PREFIX_FORMS
- 163-AUDIT-FRN.md § 5 position map + LOCKED PIN suffix forms
"""
import pytest

from orthonym.rules.seniority import SENIORITY_ORDER, SUFFIX_FORMS, PREFIX_FORMS


@pytest.mark.unit
class TestFRNSuffixForms:
    """SUFFIX_FORMS verification per AUDIT § 5 LOCKS (8 tests)."""

    def test_selenoic_Se_acid_suffix(self):
        """AUDIT § 5.2: selenoic Se-acid + carboselenoic Se-acid."""
        assert SUFFIX_FORMS['selenoic_Se_acid'] == (
            'selenoic Se-acid', 'carboselenoic Se-acid'
        )

    def test_selenoic_O_acid_suffix(self):
        """AUDIT § 5.2: selenoic O-acid + carboselenoic O-acid."""
        assert SUFFIX_FORMS['selenoic_O_acid'] == (
            'selenoic O-acid', 'carboselenoic O-acid'
        )

    def test_diselenoic_acid_suffix(self):
        """AUDIT § 5.2: diselenoic acid + carbodiselenoic acid."""
        assert SUFFIX_FORMS['diselenoic_acid'] == (
            'diselenoic acid', 'carbodiselenoic acid'
        )

    def test_telluroic_Te_acid_suffix(self):
        """AUDIT § 5.2: telluroic Te-acid + carbotelluroic Te-acid."""
        assert SUFFIX_FORMS['telluroic_Te_acid'] == (
            'telluroic Te-acid', 'carbotelluroic Te-acid'
        )

    def test_selenoamide_suffix(self):
        """AUDIT § 5.2 + P-66.6.3: selenoamide + carboselenoamide."""
        assert SUFFIX_FORMS['selenoamide'] == ('selenoamide', 'carboselenoamide')

    def test_telluroamide_suffix(self):
        """AUDIT § 5.2 + P-66.6.3: telluroamide + carbotelluroamide."""
        assert SUFFIX_FORMS['telluroamide'] == ('telluroamide', 'carbotelluroamide')

    def test_selenoaldehyde_suffix_selenal_short_form(self):
        """AUDIT § 5.2 LOCK: PIN is -selenal short form (parallel to -thial).

        v22 C-T2 (V-2): the added-carbon form is 'carboselenaldehyde' (Blue Book
        Table 28, BB ~line 18827; parallel to the 'carbothialdehyde' above) — the
        prior 'carboselenoaldehyde' (extra 'o') was a data typo mirroring the
        acid/amide infix form. Assertion updated to the corrected value.
        """
        assert SUFFIX_FORMS['selenoaldehyde'] == ('selenal', 'carboselenaldehyde')

    def test_selenoketone_suffix_form_only(self):
        """AUDIT § 5.2 LOCK: PIN is -selone suffix (dialkyl-word form fails OPSIN)."""
        assert SUFFIX_FORMS['selenoketone'] == ('selone', 'selone')


@pytest.mark.unit
class TestFRNFunctionalClassSuffixesAreNone:
    """Functional-class entries have SUFFIX_FORMS = None (handler-emitted; 3 tests)."""

    def test_iminoester_suffix_is_none(self):
        """AUDIT § 5.2: iminoester emitted via handlers/imidate.py — SUFFIX_FORMS None."""
        assert SUFFIX_FORMS.get('iminoester') is None

    def test_selenoester_suffix_is_none(self):
        """AUDIT § 5.2: selenoester is functional-class form per CONTEXT D-03."""
        assert SUFFIX_FORMS.get('selenoester') is None

    def test_telluroester_suffix_is_none(self):
        """AUDIT § 5.2: telluroester is functional-class form per CONTEXT D-03."""
        assert SUFFIX_FORMS.get('telluroester') is None


@pytest.mark.unit
class TestFRNPrefixForms:
    """PREFIX_FORMS verification per AUDIT § 5.3 (6 tests)."""

    def test_selenoic_Se_acid_prefix(self):
        """AUDIT § 5.3 (RESEARCH §4.3): -C(=O)SeH as prefix → selanylcarbonyl."""
        assert PREFIX_FORMS['selenoic_Se_acid'] == 'selanylcarbonyl'

    def test_diselenoic_acid_prefix(self):
        """AUDIT § 5.3 (parallel to dithiocarboxy): -C(=Se)SeH → diselenocarboxy."""
        assert PREFIX_FORMS['diselenoic_acid'] == 'diselenocarboxy'

    def test_thioamide_prefix(self):
        """AUDIT § 5.3: -C(=S)N(H,R) → carbamothioyl (P-66.1.4.1.1)."""
        assert PREFIX_FORMS['thioamide'] == 'carbamothioyl'

    def test_selenoamide_prefix(self):
        """AUDIT § 5.3: -C(=Se)N(H,R) → carbamoselenoyl (parallel)."""
        assert PREFIX_FORMS['selenoamide'] == 'carbamoselenoyl'

    def test_selenoaldehyde_prefix_selenoxo(self):
        """AUDIT § 5.3: =Se on aldehyde C as prefix (parallel to =S → thioxo at line 316)."""
        assert PREFIX_FORMS['selenoaldehyde'] == 'selenoxo'

    def test_selenoketone_prefix_selanylidene(self):
        """AUDIT § 5.3: =Se as non-principal prefix (parallel to sulfanylidene at line 237)."""
        assert PREFIX_FORMS['selenoketone'] == 'selanylidene'


@pytest.mark.unit
class TestFRNSeniorityOrderPositions:
    """SENIORITY_ORDER position invariants per AUDIT § 5.1 (6 tests)."""

    def test_thioamide_position_between_amide_and_sulfonamide(self):
        """AUDIT § 5.1 (P-66.1.4.1.1): thioamide ranks BELOW tertiary_amide, ABOVE primary_sulfonamide."""
        idx_amide = SENIORITY_ORDER.index('tertiary_amide')
        idx_thio = SENIORITY_ORDER.index('thioamide')
        idx_sulf = SENIORITY_ORDER.index('primary_sulfonamide')
        assert idx_amide < idx_thio < idx_sulf, (
            f"position invariant violated: tertiary_amide={idx_amide} "
            f"thioamide={idx_thio} primary_sulfonamide={idx_sulf}"
        )

    def test_selenoamide_after_thioamide(self):
        """AUDIT § 5.1: chalcogen seniority Se follows S."""
        assert SENIORITY_ORDER.index('thioamide') < SENIORITY_ORDER.index('selenoamide')

    def test_telluroamide_after_selenoamide(self):
        """AUDIT § 5.1: chalcogen seniority Te follows Se."""
        assert SENIORITY_ORDER.index('selenoamide') < SENIORITY_ORDER.index('telluroamide')

    def test_selenoic_Se_acid_after_dithioic_acid(self):
        """AUDIT § 5.1 (P-65.3): selenoic acids rank BELOW thio acids."""
        assert SENIORITY_ORDER.index('dithioic_acid') < \
            SENIORITY_ORDER.index('selenoic_Se_acid')

    def test_selenoaldehyde_after_thioaldehyde(self):
        """AUDIT § 5.1 (P-66.6.3): selenoaldehyde ranks BELOW thioaldehyde."""
        assert SENIORITY_ORDER.index('thioaldehyde') < \
            SENIORITY_ORDER.index('selenoaldehyde')

    def test_selenoketone_after_thioketone(self):
        """AUDIT § 5.1 (P-66.6.3): selenoketone ranks BELOW thioketone."""
        assert SENIORITY_ORDER.index('thioketone') < \
            SENIORITY_ORDER.index('selenoketone')


@pytest.mark.unit
class TestFRNNoRegressionOnExistingEntries:
    """No regression: existing thio* entries unchanged (5 tests)."""

    def test_thioic_S_acid_still_present(self):
        """No regression: v18 thioic_S_acid SENIORITY + SUFFIX_FORMS unchanged."""
        assert 'thioic_S_acid' in SENIORITY_ORDER
        assert SUFFIX_FORMS.get('thioic_S_acid') == (
            'thioic S-acid', 'carbothioic S-acid'
        )

    def test_thioaldehyde_thial_pin_unchanged(self):
        """No regression: thioaldehyde -thial PIN preserved."""
        assert SUFFIX_FORMS.get('thioaldehyde') == ('thial', 'carbothialdehyde')

    def test_thioketone_thione_pin_unchanged(self):
        """No regression: thioketone -thione PIN preserved."""
        assert SUFFIX_FORMS.get('thioketone') == ('thione', 'thione')

    def test_sulfanylcarbonyl_prefix_unchanged(self):
        """No regression: thioic_S_acid prefix sulfanylcarbonyl preserved."""
        assert PREFIX_FORMS.get('thioic_S_acid') == 'sulfanylcarbonyl'

    def test_thioxo_prefix_unchanged(self):
        """No regression: existing =S non-principal prefix preserved (line 316)."""
        assert PREFIX_FORMS.get('thioaldehyde') == 'thioxo'


@pytest.mark.unit
class TestFRNCount:
    """All 14+ new FRN entries present in SENIORITY_ORDER + SUFFIX_FORMS + PREFIX_FORMS."""

    def test_all_substitutive_FRN_entries_in_SENIORITY_ORDER(self):
        """AUDIT § 5.1: all 14 substitutive + 2 functional-class entries present."""
        expected_substitutive = {
            'selenoic_Se_acid', 'selenoic_O_acid', 'diselenoic_acid',
            'telluroic_Te_acid', 'telluroic_O_acid', 'ditelluroic_acid',
            'thioamide', 'selenoamide', 'telluroamide',
            'selenoaldehyde', 'telluroaldehyde', 'selenoketone', 'telluroketone',
            'iminoester', 'selenoester', 'telluroester',
        }
        for key in expected_substitutive:
            assert key in SENIORITY_ORDER, f"missing SENIORITY_ORDER entry: {key}"

    def test_substitutive_FRN_entries_have_SUFFIX_FORMS(self):
        """AUDIT § 5.2: substitutive FRN classes (FRN-A/B/C) have SUFFIX_FORMS."""
        substitutive_with_suffix = {
            'selenoic_Se_acid', 'selenoic_O_acid', 'diselenoic_acid',
            'telluroic_Te_acid', 'telluroic_O_acid', 'ditelluroic_acid',
            'thioamide', 'selenoamide', 'telluroamide',
            'selenoaldehyde', 'telluroaldehyde', 'selenoketone', 'telluroketone',
        }
        for key in substitutive_with_suffix:
            value = SUFFIX_FORMS.get(key)
            assert value is not None, f"missing SUFFIX_FORMS entry: {key}"
            assert isinstance(value, tuple) and len(value) == 2, (
                f"SUFFIX_FORMS[{key}] must be 2-tuple, got {value!r}"
            )
