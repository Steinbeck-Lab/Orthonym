"""Tests for N-oxide, isocyanate, isothiocyanate, and carbamate naming.

a phase, Plan 02: Functional class naming for these 4 FG types.
All use functional class nomenclature (two-word names like "methyl isocyanate").

: N-oxides ("pyridine 1-oxide", "trimethylamine N-oxide")
: Isocyanates ("methyl isocyanate")
: Isothiocyanates ("phenyl isothiocyanate")
: Carbamates ("ethyl carbamate", "ethyl N-methylcarbamate")
"""

import pytest
from orthonym import name_compound


# ============================================================================
# N-oxide tests
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
        # Phase B (DD1 Fix 4 / H5): trimethylamine is general-nomenclature;
        # removing the retained PIN-headline entry makes the N-oxide use the
        # substitutive amine PIN -> 'N,N-dimethylmethanamine N-oxide'.
        assert "N,N-dimethylmethanamine" in result, (
            f"Expected systematic amine PIN in '{result}'"
        )


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
# Isocyanate tests
# ============================================================================

class TestIsocyanate:
    """Isocyanates. Wave2: the PIN is the SUBSTITUTIVE
    isocyanato prefix on the parent hydride (BB VERBATIM
    'isocyanatocyclohexane (PIN) cyclohexyl isocyanate'); the functional-
    class 'R isocyanate' remains for --trivial and (interim) for AROMATIC
    attachment (the benzene FG table cannot emit isocyanatobenzene yet)."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        ("CN=C=O", "isocyanatomethane"),
        ("CCN=C=O", "isocyanatoethane"),
        ("CCCN=C=O", "1-isocyanatopropane"),
        ("CCCCN=C=O", "1-isocyanatobutane"),
        ("c1ccc(cc1)N=C=O", "phenyl isocyanate"),  # aryl: functional class kept
    ])
    def test_isocyanate_pin_forms(self, smiles, expected):
        result = name_compound(smiles)
        assert result == expected, f"Expected '{expected}', got '{result}'"

    @pytest.mark.unit
    def test_isocyanate_not_nitrile(self):
        """Isocyanate should NOT be named as a nitrile."""
        result = name_compound("CN=C=O")
        assert "nitrile" not in result
        assert "isocyanato" in result

    @pytest.mark.unit
    def test_isocyanate_not_amide(self):
        """Isocyanate should NOT be named as an amide."""
        result = name_compound("CN=C=O")
        assert "amide" not in result
        assert "isocyanato" in result


# ============================================================================
# Isothiocyanate tests
# ============================================================================

class TestIsothiocyanate:
    """Isothiocyanates. Wave2: substitutive isothiocyanato PIN
    (parallel to isocyanato); aryl keeps functional class interim."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected", [
        ("CN=C=S", "isothiocyanatomethane"),
        ("CCN=C=S", "isothiocyanatoethane"),
        ("c1ccc(cc1)N=C=S", "phenyl isothiocyanate"),  # aryl: kept
    ])
    def test_isothiocyanate_pin_forms(self, smiles, expected):
        result = name_compound(smiles)
        assert result == expected, f"Expected '{expected}', got '{result}'"


# ============================================================================
# Carbamate tests
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
