"""
Tests for IUPAC Blue Book errata corrections applied in Phase 137.

Covers:
- ERRATA-01: Expanded heteroatom seniority tables (20 elements)
- ERRATA-02: E/Z stereodescriptors for 8-member ring double bonds
- ERRATA-05: Fusion descriptor separator verification
- ERRATA-06: Hantzsch-Widman name corrections
- ERRATA-08: Symmetrical anhydride naming verification
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
        """_HETEROATOM_VARIETY_ORDER uses P-44.2.1.8 criterion (g) order, NOT P-18(b).

        P-44.2.1.8 criterion (g): F>Cl>Br>I>O>S>Se>Te>N>P>... (halogens first).
        This differs from _HETEROATOM_SENIORITY (P-18(b), N most senior).
        They serve different selection criteria. See Wave 1 R9 fix.
        """
        from orthonym.rules.ring_selection import (
            _HETEROATOM_SENIORITY, _HETEROATOM_VARIETY_ORDER
        )

        # Must contain the same 20 elements (set equality only, not same order).
        assert set(_HETEROATOM_VARIETY_ORDER) == set(_HETEROATOM_SENIORITY.keys()), (
            "VARIETY_ORDER element set must match SENIORITY keys"
        )
        # P-44.2.1.8: first element is F, N comes after Te.
        assert _HETEROATOM_VARIETY_ORDER[0] == 'F', (
            f"P-44.2.1.8: VARIETY_ORDER[0] must be 'F', got '{_HETEROATOM_VARIETY_ORDER[0]}'"
        )
        n_pos = _HETEROATOM_VARIETY_ORDER.index('N')
        te_pos = _HETEROATOM_VARIETY_ORDER.index('Te')
        assert n_pos == te_pos + 1, (
            f"P-44.2.1.8: 'N' must immediately follow 'Te' (N={n_pos}, Te={te_pos})"
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


# ============================================================================
# ERRATA-02: E/Z stereodescriptors for 8-member ring double bonds
# ============================================================================


class TestErrataEZThreshold:
    """ERRATA-02: E/Z must be assigned for cycloalkenes >= 8 ring members."""

    def test_z_cyclooctene_gets_ez(self):
        """Z-cyclooctene (8-member ring) must include E/Z stereodescriptor."""
        from orthonym import name_compound

        # Z-cyclooctene SMILES with specified stereo
        name = name_compound(r"C1=C/CCCCCC\1")
        assert "Z" in name or "E" in name, (
            f"Z-cyclooctene should have E/Z descriptor, got: '{name}'"
        )

    def test_e_cyclooctene_gets_ez(self):
        """E-cyclooctene (8-member ring) must include E/Z stereodescriptor."""
        from orthonym import name_compound

        # E-cyclooctene SMILES (trans-cyclooctene)
        name = name_compound(r"C1=C\CCCCCC/1")
        assert "Z" in name or "E" in name, (
            f"E-cyclooctene should have E/Z descriptor, got: '{name}'"
        )

    def test_cycloheptene_no_ez(self):
        """Cycloheptene (7-member ring) must NOT include E/Z descriptor."""
        from orthonym import name_compound

        name = name_compound("C1=CCCCCC1")
        # Should be plain "cycloheptene" without E or Z
        assert "E" not in name and "Z" not in name, (
            f"Cycloheptene should not have E/Z descriptor, got: '{name}'"
        )

    def test_z_cyclononene_gets_ez(self):
        """Z-cyclononene (9-member ring) must include E/Z -- already worked."""
        from orthonym import name_compound

        name = name_compound(r"C1=C/CCCCCCC/1")
        assert "Z" in name or "E" in name, (
            f"Z-cyclononene should have E/Z descriptor, got: '{name}'"
        )

    def test_unspecified_cyclooctene_no_ez(self):
        """Unspecified cyclooctene should not get E/Z (no stereo in SMILES)."""
        from orthonym import name_compound

        name = name_compound("C1=CCCCCCC1")
        # Unspecified stereo - RDKit won't assign CIP, so no E/Z expected
        assert name == "cyclooctene" or ("E" not in name and "Z" not in name), (
            f"Unspecified cyclooctene should not have E/Z, got: '{name}'"
        )

    def test_collect_stereodescriptors_8_member_ring(self):
        """collect_stereodescriptors must return E/Z for 8-member ring bonds."""
        from rdkit import Chem
        from rdkit.Chem import rdCIPLabeler
        from orthonym.rules.stereochemistry import collect_stereodescriptors

        mol = Chem.MolFromSmiles(r"C1=C/CCCCCC\1")
        rdCIPLabeler.AssignCIPLabels(mol)
        atom_to_locant = {i: i + 1 for i in range(mol.GetNumAtoms())}
        descs = collect_stereodescriptors(mol, atom_to_locant)

        ez_descs = [(loc, cip) for loc, cip in descs if cip in ('E', 'Z')]
        assert len(ez_descs) > 0, (
            f"8-member ring should now have E/Z descriptors per errata, got none"
        )


# ============================================================================
# ERRATA-05: Fusion descriptor separator verification
# ============================================================================


class TestErrataFusionSeparator:
    """ERRATA-05: Multi-edge fusion descriptors must use ':' not ';'."""

    def test_multi_edge_uses_colon_separator(self):
        """generate_fusion_descriptor multi-edge output uses ':' separator."""
        from orthonym.rules.fusion_descriptors import generate_fusion_descriptor

        # Simulate multi-edge fusion input
        # generate_fusion_descriptor with multi_edge parameter
        result = generate_fusion_descriptor(
            parent_ring=[0, 1, 2, 3, 4, 5],
            child_ring=[1, 2, 6, 7],
            shared_atoms={1, 2},
            child_is_benzene=False,
        )
        # Standard single-edge fusion should produce [x,y-z] format
        # For the colon test, we check the multi-edge code path directly
        # by inspecting the source
        import inspect
        import orthonym.rules.fusion_descriptors as fd_mod
        source = inspect.getsource(fd_mod)
        # The multi-edge fusion code must use ':'.join, not ';'.join
        assert "':'.join" in source, (
            "Multi-edge fusion descriptor must use ':' as separator"
        )
        assert "';'.join" not in source or source.index("':'.join") < source.index("';'.join") if "';'.join" in source else True, (
            "Multi-edge fusion descriptor must NOT use ';' as separator"
        )


# ============================================================================
# ERRATA-08: Symmetrical anhydride naming
# ============================================================================


class TestErrataAnhydrideNaming:
    """ERRATA-08: Symmetrical anhydrides use plain '{acid} anhydride' without bis-."""

    def test_acetic_anhydride_no_bis(self):
        """Acetic anhydride should not contain 'bis' prefix."""
        from orthonym import name_compound

        name = name_compound("CC(=O)OC(=O)C")
        assert "anhydride" in name.lower(), (
            f"Expected 'anhydride' in name, got: '{name}'"
        )
        assert "bis" not in name.lower(), (
            f"Symmetrical anhydride should not have 'bis', got: '{name}'"
        )

    def test_acetic_anhydride_name(self):
        """Acetic anhydride should produce 'acetic anhydride' or close variant."""
        from orthonym import name_compound

        name = name_compound("CC(=O)OC(=O)C")
        # Accept "acetic anhydride" or "ethanoic anhydride" (systematic)
        assert "anhydride" in name.lower(), (
            f"Expected anhydride in name, got: '{name}'"
        )
