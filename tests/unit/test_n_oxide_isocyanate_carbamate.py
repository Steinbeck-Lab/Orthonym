"""Tests for N-oxide, isocyanate, isothiocyanate, and carbamate naming.

Phase 21, Plan 02: Functional class naming for these 4 FG types.
All use functional class nomenclature (two-word names like "methyl isocyanate").

FG-04: N-oxides ("pyridine 1-oxide", "trimethylamine N-oxide")
FG-05: Isocyanates ("methyl isocyanate")
FG-06: Isothiocyanates ("phenyl isothiocyanate")
FG-09: Carbamates ("ethyl carbamate", "ethyl N-methylcarbamate")
"""

import pytest
from orthonym import name_compound


# ============================================================================
# N-oxide tests (FG-04)
# ============================================================================

class TestAromaticNOxide:
    """Aromatic N-oxides: heterocycle N-oxide."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        ("[O-][n+]1ccccc1", "pyridine 1-oxide"),
        ("Cc1cc[n+]([O-])cc1", "4-methylpyridine 1-oxide"),
    ])
    def test_aromatic_n_oxide(self, smiles, expected):
        result = name_compound(smiles)
        assert result == expected, f"Expected '{expected}', got '{result}'"


class TestAliphaticNOxide:
    """Aliphatic N-oxides: amine N-oxide."""

    @pytest.mark.unit
    def test_trimethylamine_n_oxide(self):
        result = name_compound("C[N+](C)(C)[O-]")
        assert "N-oxide" in result, f"Expected 'N-oxide' in '{result}'"
        assert "trimethylamine" in result, f"Expected 'trimethylamine' in '{result}'"


class TestNOxideRouting:
    """N-oxides must NOT be routed through ion naming."""

    @pytest.mark.unit
    def test_pyridine_n_oxide_not_ion(self):
        """Pyridine N-oxide should NOT produce ion-style name."""
        result = name_compound("[O-][n+]1ccccc1")
        assert "-ium" not in result, f"N-oxide should not have -ium suffix: '{result}'"
        assert "-olate" not in result, f"N-oxide should not have -olate suffix: '{result}'"
        assert "oxide" in result

    @pytest.mark.unit
    def test_trimethylamine_n_oxide_not_ion(self):
        """Trimethylamine N-oxide should NOT produce ion-style name."""
        result = name_compound("C[N+](C)(C)[O-]")
        assert "-ium" not in result
        assert "N-oxide" in result

    @pytest.mark.unit
    def test_n_oxide_species_type_neutral(self):
        """N-oxide molecules should be classified as 'neutral' species."""
        from orthonym.perception.ions import detect_species_type
        from rdkit import Chem

        mol = Chem.MolFromSmiles("[O-][n+]1ccccc1")
        assert detect_species_type(mol) == "neutral"

        mol = Chem.MolFromSmiles("C[N+](C)(C)[O-]")
        assert detect_species_type(mol) == "neutral"


# ============================================================================
# Isocyanate tests (FG-05)
# ============================================================================

class TestIsocyanate:
    """Isocyanates: R-N=C=O -> 'R isocyanate'."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        ("CN=C=O", "methyl isocyanate"),
        ("CCN=C=O", "ethyl isocyanate"),
        ("CCCN=C=O", "propyl isocyanate"),
        ("CCCCN=C=O", "butyl isocyanate"),
        ("c1ccc(cc1)N=C=O", "phenyl isocyanate"),
    ])
    def test_isocyanate_functional_class(self, smiles, expected):
        result = name_compound(smiles)
        assert result == expected, f"Expected '{expected}', got '{result}'"

    @pytest.mark.unit
    def test_isocyanate_not_nitrile(self):
        """Isocyanate should NOT be named as a nitrile."""
        result = name_compound("CN=C=O")
        assert "nitrile" not in result
        assert "isocyanate" in result

    @pytest.mark.unit
    def test_isocyanate_not_amide(self):
        """Isocyanate should NOT be named as an amide."""
        result = name_compound("CN=C=O")
        assert "amide" not in result
        assert "isocyanate" in result


# ============================================================================
# Isothiocyanate tests (FG-06)
# ============================================================================

class TestIsothiocyanate:
    """Isothiocyanates: R-N=C=S -> 'R isothiocyanate'."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        ("CN=C=S", "methyl isothiocyanate"),
        ("CCN=C=S", "ethyl isothiocyanate"),
        ("c1ccc(cc1)N=C=S", "phenyl isothiocyanate"),
    ])
    def test_isothiocyanate_functional_class(self, smiles, expected):
        result = name_compound(smiles)
        assert result == expected, f"Expected '{expected}', got '{result}'"


# ============================================================================
# Carbamate tests (FG-09)
# ============================================================================

class TestCarbamate:
    """Carbamates: N-C(=O)-O-R -> 'R carbamate' (functional class)."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        ("NC(=O)OC", "methyl carbamate"),
        ("NC(=O)OCC", "ethyl carbamate"),
        ("NC(=O)OC(C)(C)C", "tert-butyl carbamate"),
    ])
    def test_carbamate_simple(self, smiles, expected):
        result = name_compound(smiles)
        assert result == expected, f"Expected '{expected}', got '{result}'"

    @pytest.mark.unit
    def test_carbamate_n_methyl(self):
        """N-methylcarbamate: CNC(=O)OCC -> ethyl N-methylcarbamate."""
        result = name_compound("CNC(=O)OCC")
        assert result == "ethyl N-methylcarbamate", f"Got '{result}'"

    @pytest.mark.unit
    def test_carbamate_not_ester(self):
        """Carbamate should NOT be named as a generic ester."""
        result = name_compound("NC(=O)OCC")
        assert "carbamate" in result
        assert "formate" not in result.lower()
        assert "alkanoate" not in result.lower()

    @pytest.mark.unit
    def test_carbamate_not_amide(self):
        """Carbamate should NOT be named as an amide."""
        result = name_compound("NC(=O)OCC")
        assert "carbamate" in result
        assert "amide" not in result


# ============================================================================
# Prefix form tests (when another FG has higher seniority)
# ============================================================================

class TestPrefixForms:
    """When isocyanate/isothiocyanate is subordinate, use prefix form."""

    @pytest.mark.unit
    @pytest.mark.xfail(reason="Polyfunctional handler does not yet assemble isocyanato prefix on benzene")
    def test_isocyanato_prefix_on_acid(self):
        """Isocyanato prefix when carboxylic acid is principal."""
        result = name_compound("OC(=O)c1ccc(N=C=O)cc1")
        assert "isocyanato" in result, f"Expected 'isocyanato' in '{result}'"

    @pytest.mark.unit
    @pytest.mark.xfail(reason="Polyfunctional handler does not yet assemble isothiocyanato prefix on benzene")
    def test_isothiocyanato_prefix_on_acid(self):
        """Isothiocyanato prefix when carboxylic acid is principal."""
        result = name_compound("OC(=O)c1ccc(N=C=S)cc1")
        assert "isothiocyanato" in result, f"Expected 'isothiocyanato' in '{result}'"
