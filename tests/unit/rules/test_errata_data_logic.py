"""
Tests for IUPAC Blue Book errata corrections applied in Phase 137.

Covers:
- ERRATA-01: Expanded heteroatom seniority tables (20 elements)
- ERRATA-06: Hantzsch-Widman name corrections
- ERRATA-09: carbonochloridoyl prefix (replaces chlorocarbonyl)

Reference: IUPAC 2013 Blue Book + BBerrors.html corrections through Dec 2025
"""

import inspect
import pytest


# ============================================================================
# ERRATA-01: Heteroatom seniority expansion (P-18(b) + P-44.2.1)
# ============================================================================


class TestErrataHeteroatomSeniority:
    """ERRATA-01: _HETEROATOM_SENIORITY must include all 20 elements."""

    def test_seniority_dict_has_20_entries(self):
        """_HETEROATOM_SENIORITY must have exactly 20 entries."""
        from orthonym.rules.ring_selection import _HETEROATOM_SENIORITY

        assert len(_HETEROATOM_SENIORITY) == 20, (
            f"Expected 20 entries, got {len(_HETEROATOM_SENIORITY)}. "
            f"Keys: {sorted(_HETEROATOM_SENIORITY.keys())}"
        )

    def test_seniority_dict_contains_all_elements(self):
        """All 20 elements must be present as keys."""
        from orthonym.rules.ring_selection import _HETEROATOM_SENIORITY

        expected = {
            'N', 'F', 'Cl', 'Br', 'I', 'O', 'S', 'Se', 'Te',
            'P', 'As', 'Sb', 'Bi', 'Si', 'Ge', 'Sn', 'Pb', 'B', 'Al', 'Ga'
        }
        assert set(_HETEROATOM_SENIORITY.keys()) == expected

    def test_seniority_ordering_p18b(self):
        """P-18(b) ordering: N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga."""
        from orthonym.rules.ring_selection import _HETEROATOM_SENIORITY

        p18b_order = ['N', 'P', 'As', 'Sb', 'Bi', 'Si', 'Ge', 'Sn', 'Pb', 'B', 'Al', 'Ga']
        for i in range(len(p18b_order) - 1):
            senior = p18b_order[i]
            junior = p18b_order[i + 1]
            assert _HETEROATOM_SENIORITY[senior] > _HETEROATOM_SENIORITY[junior], (
                f"{senior} ({_HETEROATOM_SENIORITY[senior]}) should be more senior than "
                f"{junior} ({_HETEROATOM_SENIORITY[junior]})"
            )

    def test_seniority_halogens_preserved(self):
        """Halogens must remain in seniority dict for P-44.2.1 ring comparison."""
        from orthonym.rules.ring_selection import _HETEROATOM_SENIORITY

        for halogen in ['F', 'Cl', 'Br', 'I']:
            assert halogen in _HETEROATOM_SENIORITY, f"{halogen} must be in seniority dict"

    def test_variety_order_has_20_elements(self):
        """_HETEROATOM_VARIETY_ORDER must have exactly 20 elements."""
        from orthonym.rules.ring_selection import _HETEROATOM_VARIETY_ORDER

        assert len(_HETEROATOM_VARIETY_ORDER) == 20, (
            f"Expected 20 elements, got {len(_HETEROATOM_VARIETY_ORDER)}"
        )

    def test_variety_order_matches_seniority_keys(self):
        """_HETEROATOM_VARIETY_ORDER elements must match _HETEROATOM_SENIORITY keys."""
        from orthonym.rules.ring_selection import (
            _HETEROATOM_SENIORITY, _HETEROATOM_VARIETY_ORDER
        )

        assert set(_HETEROATOM_VARIETY_ORDER) == set(_HETEROATOM_SENIORITY.keys())

    def test_variety_order_in_seniority_order(self):
        """_HETEROATOM_VARIETY_ORDER must be sorted by descending seniority."""
        from orthonym.rules.ring_selection import (
            _HETEROATOM_SENIORITY, _HETEROATOM_VARIETY_ORDER
        )

        for i in range(len(_HETEROATOM_VARIETY_ORDER) - 1):
            curr = _HETEROATOM_VARIETY_ORDER[i]
            next_ = _HETEROATOM_VARIETY_ORDER[i + 1]
            assert _HETEROATOM_SENIORITY[curr] > _HETEROATOM_SENIORITY[next_], (
                f"Position {i}: {curr} ({_HETEROATOM_SENIORITY[curr]}) must be more "
                f"senior than {next_} ({_HETEROATOM_SENIORITY[next_]})"
            )


class TestErrataHWPriority:
    """ERRATA-01: HETEROATOM_PRIORITY must include Al and Ga."""

    def test_hw_priority_includes_al_ga(self):
        """HETEROATOM_PRIORITY must include Al and Ga entries."""
        from orthonym.data.hw_heteroatoms import HETEROATOM_PRIORITY

        assert 'Al' in HETEROATOM_PRIORITY, "Al must be in HETEROATOM_PRIORITY"
        assert 'Ga' in HETEROATOM_PRIORITY, "Ga must be in HETEROATOM_PRIORITY"

    def test_hw_priority_ordering_preserved(self):
        """HW ordering: O>S>Se>Te>N>P>As>Sb>Bi>Si>Ge>Sn>Pb>B>Al>Ga (NOT P-18(b) order)."""
        from orthonym.data.hw_heteroatoms import HETEROATOM_PRIORITY

        hw_order = ['O', 'S', 'Se', 'Te', 'N', 'P', 'As', 'Sb', 'Bi',
                     'Si', 'Ge', 'Sn', 'Pb', 'B', 'Al', 'Ga']
        for i in range(len(hw_order) - 1):
            higher = hw_order[i]
            lower = hw_order[i + 1]
            assert HETEROATOM_PRIORITY[higher] < HETEROATOM_PRIORITY[lower], (
                f"{higher} (priority {HETEROATOM_PRIORITY[higher]}) should have higher "
                f"priority (lower number) than {lower} (priority {HETEROATOM_PRIORITY[lower]})"
            )


# ============================================================================
# ERRATA-06: Hantzsch-Widman name corrections
# ============================================================================


class TestErrataHWNames:
    """ERRATA-06: Verify HW names are correct (no erroneous forms)."""

    def test_thiazolidine_in_retained_names(self):
        """retained_names.py must contain 'thiazolidine' (not 'thioxazolidine')."""
        from orthonym.data.retained_names import RETAINED_NAMES

        # Check thiazolidine present in values
        values = list(RETAINED_NAMES.values()) if isinstance(RETAINED_NAMES, dict) else []
        assert 'thiazolidine' in values, "thiazolidine must be in RETAINED_NAMES values"

    def test_isothiazolidine_in_retained_names(self):
        """retained_names.py must contain 'isothiazolidine' (not 'isothioxazolidine')."""
        from orthonym.data.retained_names import RETAINED_NAMES

        values = list(RETAINED_NAMES.values()) if isinstance(RETAINED_NAMES, dict) else []
        assert 'isothiazolidine' in values, "isothiazolidine must be in RETAINED_NAMES values"

    def test_no_thioxazol_forms_in_hw_stems(self):
        """hw_stems.py must not contain erroneous 'thioxazol' forms."""
        import orthonym.data.hw_stems as hw_stems_mod

        source = inspect.getsource(hw_stems_mod)
        for bad_form in ['thioxazol', 'selenoxazol', 'telluroxazol']:
            assert bad_form not in source, (
                f"Erroneous form '{bad_form}' found in hw_stems.py"
            )

    def test_no_thioxazol_forms_in_retained_names(self):
        """retained_names.py must not contain erroneous HW forms."""
        import orthonym.data.retained_names as rn_mod

        source = inspect.getsource(rn_mod)
        for bad_form in ['thioxazol', 'selenoxazol', 'telluroxazol']:
            assert bad_form not in source, (
                f"Erroneous form '{bad_form}' found in retained_names.py"
            )


# ============================================================================
# ERRATA-09: carbonochloridoyl prefix (replaces chlorocarbonyl)
# ============================================================================


class TestErrataCarbonochloridoyl:
    """ERRATA-09: acid_chloride non-principal prefix must be 'carbonochloridoyl'."""

    def test_prefix_forms_acid_chloride(self):
        """PREFIX_FORMS['acid_chloride'] must be 'carbonochloridoyl'."""
        from orthonym.rules.seniority import PREFIX_FORMS

        assert PREFIX_FORMS["acid_chloride"] == "carbonochloridoyl", (
            f"Expected 'carbonochloridoyl', got '{PREFIX_FORMS['acid_chloride']}'"
        )

    def test_benzene_suffix_to_prefix_carbonyl_chloride(self):
        """_SUFFIX_TO_PREFIX['carbonyl chloride'] must be 'carbonochloridoyl'."""
        from orthonym.rules.benzene import _SUFFIX_TO_PREFIX

        assert _SUFFIX_TO_PREFIX["carbonyl chloride"] == "carbonochloridoyl", (
            f"Expected 'carbonochloridoyl', got '{_SUFFIX_TO_PREFIX['carbonyl chloride']}'"
        )

    def test_no_chlorocarbonyl_functional_value_in_seniority(self):
        """'chlorocarbonyl' must not appear as a functional value in PREFIX_FORMS."""
        from orthonym.rules.seniority import PREFIX_FORMS

        for key, value in PREFIX_FORMS.items():
            assert value != "chlorocarbonyl", (
                f"PREFIX_FORMS['{key}'] still has 'chlorocarbonyl' -- must be updated"
            )

    def test_no_chlorocarbonyl_functional_value_in_benzene(self):
        """'chlorocarbonyl' must not appear as a value in _SUFFIX_TO_PREFIX."""
        from orthonym.rules.benzene import _SUFFIX_TO_PREFIX

        for key, value in _SUFFIX_TO_PREFIX.items():
            assert value != "chlorocarbonyl", (
                f"_SUFFIX_TO_PREFIX['{key}'] still has 'chlorocarbonyl' -- must be updated"
            )

    def test_acid_bromide_unchanged(self):
        """PREFIX_FORMS['acid_bromide'] must remain 'bromocarbonyl' (NOT changed by errata)."""
        from orthonym.rules.seniority import PREFIX_FORMS

        assert PREFIX_FORMS["acid_bromide"] == "bromocarbonyl", (
            f"acid_bromide should remain 'bromocarbonyl', got '{PREFIX_FORMS['acid_bromide']}'"
        )

    def test_acid_fluoride_unchanged(self):
        """PREFIX_FORMS['acid_fluoride'] must remain 'fluorocarbonyl' (NOT changed by errata)."""
        from orthonym.rules.seniority import PREFIX_FORMS

        assert PREFIX_FORMS["acid_fluoride"] == "fluorocarbonyl", (
            f"acid_fluoride should remain 'fluorocarbonyl', got '{PREFIX_FORMS['acid_fluoride']}'"
        )
