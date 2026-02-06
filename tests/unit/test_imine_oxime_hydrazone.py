"""Tests for imine suffix naming, oxime and hydrazone functional class naming.

Phase 21 Plan 01: Missing Functional Groups (imines, oximes, hydrazones).
"""

import pytest
from orthonym import name_compound


class TestImineSuffixNaming:
    """Imine uses suffix naming: chain/ring + locant + 'imine'."""

    def test_simple_imine(self):
        """CC=N -> ethan-1-imine"""
        result = name_compound("CC=N")
        assert "imine" in result
        # Should be "ethan-1-imine" (suffix naming)
        assert result == "ethan-1-imine"

    def test_internal_imine(self):
        """CC(=N)C -> propan-2-imine"""
        result = name_compound("CC(=N)C")
        assert "imine" in result
        assert result == "propan-2-imine"

    def test_longer_chain_imine(self):
        """CCCCCC=N -> hexan-1-imine"""
        result = name_compound("CCCCCC=N")
        assert "imine" in result
        assert "hexan" in result


class TestOximeFunctionalClassNaming:
    """Oxime uses functional class naming: '[parent carbonyl name] oxime'."""

    def test_ketone_oxime(self):
        """CC(=NO)C -> 'acetone oxime' or 'propan-2-one oxime'."""
        result = name_compound("CC(=NO)C")
        assert "oxime" in result
        # Accept either retained or systematic form of the parent
        assert result in ("acetone oxime", "propan-2-one oxime")

    def test_aldehyde_oxime(self):
        """CC=NO -> 'acetaldehyde oxime' or 'ethanal oxime'."""
        result = name_compound("CC=NO")
        assert "oxime" in result
        assert result in ("acetaldehyde oxime", "ethanal oxime")

    def test_cyclic_oxime(self):
        """C1(=NO)CCCCC1 -> '[cyclohexanone variant] oxime'."""
        result = name_compound("C1(=NO)CCCCC1")
        assert "oxime" in result
        # Cyclic ketone parent -- accept any valid cyclohexanone form
        assert "cyclohex" in result

    def test_oxime_contains_parent_name(self):
        """Verify the parent carbonyl name appears before ' oxime'."""
        result = name_compound("CC(=NO)C")
        parts = result.rsplit(" ", 1)
        assert len(parts) == 2
        assert parts[1] == "oxime"
        # Parent name should be a valid carbonyl name
        assert len(parts[0]) > 0


class TestHydrazoneFunctionalClassNaming:
    """Hydrazone uses functional class naming: '[parent carbonyl name] hydrazone'."""

    def test_ketone_hydrazone(self):
        """CC(=NN)C -> 'acetone hydrazone' or 'propan-2-one hydrazone'."""
        result = name_compound("CC(=NN)C")
        assert "hydrazone" in result
        assert result in ("acetone hydrazone", "propan-2-one hydrazone")

    def test_aldehyde_hydrazone(self):
        """CC=NN -> 'acetaldehyde hydrazone' or 'ethanal hydrazone'."""
        result = name_compound("CC=NN")
        assert "hydrazone" in result
        assert result in ("acetaldehyde hydrazone", "ethanal hydrazone")

    def test_cyclic_hydrazone(self):
        """C1(=NN)CCCCC1 -> '[cyclohexanone variant] hydrazone'."""
        result = name_compound("C1(=NN)CCCCC1")
        assert "hydrazone" in result
        assert "cyclohex" in result

    def test_hydrazone_contains_parent_name(self):
        """Verify the parent carbonyl name appears before ' hydrazone'."""
        result = name_compound("CC(=NN)C")
        parts = result.rsplit(" ", 1)
        assert len(parts) == 2
        assert parts[1] == "hydrazone"
        assert len(parts[0]) > 0


class TestFGDetectionNewGroups:
    """Verify SMARTS detection for all 7 new FGs + collision prevention."""

    def test_isocyanate_detection(self):
        from orthonym.perception.functional_groups import detect_functional_groups
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CN=C=O")
        fgs = detect_functional_groups(mol)
        assert "isocyanate" in fgs

    def test_isothiocyanate_detection(self):
        from orthonym.perception.functional_groups import detect_functional_groups
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CN=C=S")
        fgs = detect_functional_groups(mol)
        assert "isothiocyanate" in fgs

    def test_urea_detection_no_amide_collision(self):
        from orthonym.perception.functional_groups import detect_functional_groups
        from rdkit import Chem
        mol = Chem.MolFromSmiles("NC(=O)N")
        fgs = detect_functional_groups(mol)
        assert "urea" in fgs
        assert "primary_amide" not in fgs

    def test_guanidine_detection_no_imine_collision(self):
        from orthonym.perception.functional_groups import detect_functional_groups
        from rdkit import Chem
        mol = Chem.MolFromSmiles("NC(=N)N")
        fgs = detect_functional_groups(mol)
        assert "guanidine" in fgs
        assert "imine" not in fgs

    def test_carbamate_detection_no_ester_collision(self):
        from orthonym.perception.functional_groups import detect_functional_groups
        from rdkit import Chem
        mol = Chem.MolFromSmiles("NC(=O)OC")
        fgs = detect_functional_groups(mol)
        assert "carbamate" in fgs
        assert "ester" not in fgs

    def test_boronic_acid_detection(self):
        from orthonym.perception.functional_groups import detect_functional_groups
        from rdkit import Chem
        mol = Chem.MolFromSmiles("OB(O)c1ccccc1")
        fgs = detect_functional_groups(mol)
        assert "boronic_acid" in fgs

    def test_n_oxide_aromatic_detection(self):
        from orthonym.perception.functional_groups import detect_functional_groups
        from rdkit import Chem
        mol = Chem.MolFromSmiles("[O-][n+]1ccccc1")
        fgs = detect_functional_groups(mol)
        assert "n_oxide_aromatic" in fgs

    def test_n_oxide_aliphatic_detection(self):
        from orthonym.perception.functional_groups import detect_functional_groups
        from rdkit import Chem
        mol = Chem.MolFromSmiles("C[N+](C)(C)[O-]")
        fgs = detect_functional_groups(mol)
        assert "n_oxide_aliphatic" in fgs


class TestRetainedNames:
    """Verify retained names for urea and guanidine."""

    def test_urea_retained_name(self):
        assert name_compound("NC(=O)N") == "urea"

    def test_guanidine_retained_name(self):
        assert name_compound("NC(=N)N") == "guanidine"


class TestSeniorityData:
    """Verify seniority, prefix, and suffix data for new FGs."""

    def test_boronic_acid_in_seniority(self):
        from orthonym.rules.seniority import SENIORITY_ORDER
        assert "boronic_acid" in SENIORITY_ORDER
        # Should be after phosphinic_acid and before anhydride
        pi = SENIORITY_ORDER.index("phosphinic_acid")
        bi = SENIORITY_ORDER.index("boronic_acid")
        ai = SENIORITY_ORDER.index("anhydride")
        assert pi < bi < ai

    def test_oxime_suffix_form(self):
        from orthonym.rules.seniority import SUFFIX_FORMS
        assert "oxime" in SUFFIX_FORMS
        assert SUFFIX_FORMS["oxime"] == ("oxime", "oxime")

    def test_hydrazone_suffix_form(self):
        from orthonym.rules.seniority import SUFFIX_FORMS
        assert "hydrazone" in SUFFIX_FORMS

    def test_new_prefix_forms_exist(self):
        from orthonym.rules.seniority import PREFIX_FORMS
        expected = {
            "isocyanate": "isocyanato",
            "isothiocyanate": "isothiocyanato",
            "urea": "carbamoylamino",
            "guanidine": "guanidino",
            "boronic_acid": "dihydroxyboranyl",
            "hydrazone": "hydrazinylidene",
        }
        for fg, prefix in expected.items():
            assert PREFIX_FORMS.get(fg) == prefix, f"{fg}: expected {prefix}, got {PREFIX_FORMS.get(fg)}"
