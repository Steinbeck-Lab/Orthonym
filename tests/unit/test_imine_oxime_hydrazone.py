"""Tests for imine suffix naming, oxime and hydrazone functional class naming.

a phase Plan 01: Missing Functional Groups (imines, oximes, hydrazones).
"""

import pytest
from orthonym import name_compound


class TestImineSuffixNaming:
    """Imine uses suffix naming: chain/ring + locant + 'imine'."""

    def test_simple_imine(self):
        """CC=N -> ethanimine (C2 -> locant elided,."""
        result = name_compound("CC=N")
        assert "imine" in result
        assert result == "ethanimine"

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
    """Oxime PINs are formed SUBSTITUTIVELY as N-hydroxy derivatives of imines
    (f) /, NOT by functional-class nomenclature (the
    retired '[carbonyl] oxime'). The substitutive form has been the shipped PIN
    since Wave2 T2a; these assertions were reconciled to it in W3-P15."""

    def test_ketone_oxime(self):
        """CC(=NO)C -> 'N-hydroxypropan-2-imine' (PIN)."""
        result = name_compound("CC(=NO)C")
        assert result == "N-hydroxypropan-2-imine"

    def test_aldehyde_oxime(self):
        """CC=NO -> 'N-hydroxyethanimine' (PIN)."""
        result = name_compound("CC=NO")
        assert result == "N-hydroxyethanimine"

    def test_cyclic_oxime(self):
        """C1(=NO)CCCCC1 -> 'N-hydroxycyclohexan-1-imine' (PIN)."""
        result = name_compound("C1(=NO)CCCCC1")
        assert "imine" in result
        assert "cyclohex" in result

    def test_oxime_contains_parent_name(self):
        """The substitutive oxime PIN carries the 'N-hydroxy' prefix + imine parent."""
        result = name_compound("CC(=NO)C")
        assert result.startswith("N-hydroxy")
        assert result.endswith("imine")


class TestOximeEZStereoPreservation:
    """Test that C=N E/Z stereodescriptor is preserved in oxime names."""

    def test_oxime_with_cn_stereo(self):
        """Oxime with C=N stereo should include both C=N and C=C stereo descriptors."""
        result = name_compound("C=C/C(C)=C/CC(C)(C)/C(=N\\O)C(C)C")
        assert "oxime" in result
        assert "Z" in result, f"Expected Z descriptor in '{result}'"
        assert "E" in result, f"Expected E descriptor in '{result}'"
        # Should have both 3Z and 6E
        assert "3Z" in result, f"Expected 3Z in '{result}'"
        assert "6E" in result, f"Expected 6E in '{result}'"

    def test_simple_oxime_no_spurious_stereo(self):
        """Simple oxime without C=N stereo should NOT get stereo added
        (substitutive PIN 'N-hydroxypropan-2-imine')."""
        result = name_compound("CC(=NO)C")
        assert "imine" in result
        # Should NOT contain E or Z
        assert "E" not in result, f"Unexpected E in simple oxime '{result}'"
        assert "Z" not in result, f"Unexpected Z in simple oxime '{result}'"

    def test_aldehyde_oxime_no_spurious_stereo(self):
        """Aldehyde oxime without stereo should not get stereo added
        (substitutive PIN 'N-hydroxyethanimine')."""
        result = name_compound("CC=NO")
        assert "imine" in result
        assert "E" not in result
        assert "Z" not in result


class TestHydrazoneFunctionalClassNaming:
    """Hydrazone PINs are formed SUBSTITUTIVELY as 'ylidene' derivatives of
    hydrazine method (1) = PIN), NOT by functional-class
    nomenclature (the retired '[carbonyl] hydrazone'). Reconciled to the
    substitutive PIN in W3-P15 (idx 1927)."""

    def test_ketone_hydrazone(self):
        """CC(=NN)C -> '(propan-2-ylidene)hydrazine' (PIN)."""
        result = name_compound("CC(=NN)C")
        assert result == "(propan-2-ylidene)hydrazine"

    def test_aldehyde_hydrazone(self):
        """CC=NN -> 'ethylidenehydrazine' (PIN)."""
        result = name_compound("CC=NN")
        assert result == "ethylidenehydrazine"

    def test_cyclic_hydrazone(self):
        """C1(=NN)CCCCC1 -> 'cyclohexylidenehydrazine' (PIN)."""
        result = name_compound("C1(=NN)CCCCC1")
        assert "hydrazine" in result
        assert "cyclohex" in result

    def test_hydrazone_contains_parent_name(self):
        """The substitutive hydrazone PIN is an 'ylidene' derivative of hydrazine."""
        result = name_compound("CC(=NN)C")
        assert result.endswith("hydrazine")
        assert "ylidene" in result


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
            # (the Blue Book) carbamimidoylamino (preferred prefix)
            "guanidine": "carbamimidoylamino",
            "boronic_acid": "borono",  # preselected prefix)
            "hydrazone": "hydrazinylidene",
        }
        for fg, prefix in expected.items():
            assert PREFIX_FORMS.get(fg) == prefix, f"{fg}: expected {prefix}, got {PREFIX_FORMS.get(fg)}"
