"""Heterocycle handler accuracy test suite from benchmark failures.

Phase 142-02: Tests heterocycle retained name routing, fused heterocycle
identification, and basic heterocycle naming accuracy. Derived from
D-03 diagnostic analysis of 500-compound benchmark failures.

Tests cover:
- Nucleobase retained names (xanthine, adenine, guanine, hypoxanthine, uracil)
- Fused heterocycle identification (acridine, carbazole, phenanthridine)
- Basic heterocycle regression (pyridine, pyrrole, furan, thiophene)
- Fused bicyclic heterocycles (quinoline, isoquinoline, benzofuran, etc.)
- Xanthine derivative systematic naming (caffeine, theophylline, theobromine)
"""

import pytest
from orthonym.namer import name_compound


# =============================================================================
# Heterocycle accuracy test cases: (SMILES, expected_name)
# Each entry validated against IUPAC 2013 retained names and OPSIN round-trip
# =============================================================================
HETEROCYCLE_ACCURACY_CASES = [
    # --- Nucleobase retained names (key benchmark failures) ---
    ("O=c1[nH]c(=O)c2[nH]cnc2[nH]1", "xanthine"),
    ("Nc1ncnc2[nH]cnc12", "adenine"),
    ("O=c1[nH]cnc2[nH]cnc12", "hypoxanthine"),
    ("O=c1cc[nH]c(=O)[nH]1", "uracil"),

    # --- Guanine (both tautomeric input forms) ---
    ("Nc1nc2[nH]cnc2c(=O)[nH]1", "guanine"),
    ("Nc1nc(=O)c2[nH]cnc2[nH]1", "guanine"),

    # --- Purine (IUPAC retained name with indicated H) ---
    ("c1ncnc2[nH]cnc12", "9H-purine"),

    # --- Orotic acid (pyrimidine derivative, IUPAC retained name) ---
    ("OC(=O)c1cc(=O)[nH]c(=O)[nH]1", "orotic acid"),

    # --- Xanthine derivatives (systematic naming, NOT retained names) ---
    ("Cn1c(=O)c2c(ncn2C)n(C)c1=O", "1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione"),

    # --- Tricyclic fused heterocycles ---
    ("c1ccc2nc3ccccc3cc2c1", "acridine"),
    ("c1ccc2c(c1)[nH]c1ccccc12", "9H-carbazole"),
    ("c1ccc2c(c1)cnc1ccccc12", "phenanthridine"),

    # --- Basic monocyclic heterocycles (regression checks) ---
    ("c1ccncc1", "pyridine"),
    ("c1cc[nH]c1", "pyrrole"),
    ("c1ccoc1", "furan"),
    ("c1ccsc1", "thiophene"),
    ("c1c[nH]cn1", "imidazole"),
    ("c1cncnc1", "pyrimidine"),
    ("c1cnccn1", "pyrazine"),

    # --- Fused bicyclic heterocycles (regression checks) ---
    ("c1ccc2ncccc2c1", "quinoline"),
    ("c1ccc2cnccc2c1", "isoquinoline"),
    ("c1ccc2[nH]ccc2c1", "1H-indole"),
    ("c1ccc2[nH]cnc2c1", "1H-benzimidazole"),
    ("c1ccc2occc2c1", "benzofuran"),
    ("c1ccc2sccc2c1", "benzothiophene"),
    ("c1ccc2ncncc2c1", "quinazoline"),
]


@pytest.mark.parametrize("smiles,expected", HETEROCYCLE_ACCURACY_CASES,
                         ids=[f"{exp}" for _, exp in HETEROCYCLE_ACCURACY_CASES])
def test_heterocycle_accuracy(smiles, expected):
    """Each heterocycle SMILES must produce the expected IUPAC name."""
    result = name_compound(smiles)
    assert result == expected, (
        f"Expected '{expected}' for SMILES '{smiles}', got '{result}'"
    )


class TestXanthineRetainedNamePriority:
    """Xanthine (unsubstituted) must return retained name, not systematic."""

    def test_xanthine_not_systematic(self):
        """Xanthine must return 'xanthine', not '3,7-dihydro-1H-purine-2,6-dione'."""
        result = name_compound("O=c1[nH]c(=O)c2[nH]cnc2[nH]1")
        assert result == "xanthine", f"Expected 'xanthine', got '{result}'"
        assert "purine" not in result, (
            f"Systematic name leaked through: '{result}'"
        )

    def test_caffeine_systematic(self):
        """Caffeine returns systematic name (not a retained IUPAC name)."""
        result = name_compound("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        assert "trimethyl" in result, f"Expected trimethyl in '{result}'"
        assert "purine" in result, f"Expected purine in '{result}'"


class TestGuanineTautomers:
    """Both guanine tautomer inputs must return 'guanine'."""

    def test_guanine_tautomer_1(self):
        """Guanine tautomer from retained_names.py entry."""
        result = name_compound("Nc1nc2[nH]cnc2c(=O)[nH]1")
        assert result == "guanine", f"Expected 'guanine', got '{result}'"

    def test_guanine_tautomer_2(self):
        """Guanine alternate tautomer (common input form)."""
        result = name_compound("Nc1nc(=O)c2[nH]cnc2[nH]1")
        assert result == "guanine", f"Expected 'guanine', got '{result}'"


class TestFusedHeterocycleIdentification:
    """Fused heterocycles must be correctly identified by their ring system."""

    def test_acridine_not_phenanthridine(self):
        """Acridine (N at center) must not be confused with phenanthridine (N at 5)."""
        acridine = name_compound("c1ccc2nc3ccccc3cc2c1")
        phenanthridine = name_compound("c1ccc2c(c1)cnc1ccccc12")
        assert acridine == "acridine", f"Expected 'acridine', got '{acridine}'"
        assert phenanthridine == "phenanthridine", (
            f"Expected 'phenanthridine', got '{phenanthridine}'"
        )

    def test_carbazole_with_indicated_h(self):
        """Carbazole must include 9H- indicated hydrogen."""
        result = name_compound("c1ccc2c(c1)[nH]c1ccccc12")
        assert result == "9H-carbazole", f"Expected '9H-carbazole', got '{result}'"
