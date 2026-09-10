"""
End-to-end integration tests for non-principal FG prefix emission.

Verifies that the full naming pipeline (perception -> classification -> assembly)
correctly emits subordinate prefixes for non-principal functional groups.
"""
import pytest
from orthonym import name_compound


class TestPrefixEmissionE2E:
    """End-to-end naming tests for non-principal functional group prefixes."""

    def test_glutamine_full_name(self):
        """Glutamine should be named with carbamoyl prefix and amino prefix.

        Expected: 2-amino-4-carbamoylbutanoic acid (or 2-amino-5-carbamoylpentanoic acid
        depending on chain length computation, but MUST contain carbamoyl).
        """
        name = name_compound("NC(=O)CCCC(N)C(=O)O")
        assert "carbamoyl" in name, f"Expected 'carbamoyl' in '{name}'"
        assert "amino" in name, f"Expected 'amino' in '{name}'"
        assert "oic acid" in name, f"Expected 'oic acid' suffix in '{name}'"

    def test_cyanopropanoic_acid_full(self):
        """3-cyanopropanoic acid end-to-end."""
        name = name_compound("N#CCC(=O)O")
        assert name == "3-cyanopropanoic acid", f"Expected '3-cyanopropanoic acid', got '{name}'"

    def test_formylbenzoic_acid_full(self):
        """4-formylbenzoic acid end-to-end."""
        name = name_compound("O=Cc1ccc(C(=O)O)cc1")
        assert name == "4-formylbenzoic acid", f"Expected '4-formylbenzoic acid', got '{name}'"

    def test_mixed_acid_amide(self):
        """Compound with both acid (principal) and amide (subordinate).

        5-amino-5-carbamoylpentanoic acid or similar.
        """
        # NC(=O)CCCC(N)C(=O)O is glutamine
        name = name_compound("NC(=O)CCCC(N)C(=O)O")
        # Must have carbamoyl for non-principal amide
        assert "carbamoyl" in name
        # Must have amino for non-principal amine
        assert "amino" in name
        # Acid is the principal group suffix
        assert "acid" in name

    def test_acid_nitrile_compound(self):
        """Compound with acid (principal) and nitrile (subordinate).

        N#CCCC(=O)O -> 4-cyanobutanoic acid.
        """
        name = name_compound("N#CCCC(=O)O")
        assert name == "4-cyanobutanoic acid", f"Expected '4-cyanobutanoic acid', got '{name}'"

    def test_amide_as_principal_not_affected(self):
        """When amide IS the principal group, no carbamoyl prefix should appear."""
        name = name_compound("CCC(=O)N")
        assert "carbamoyl" not in name, f"Unexpected 'carbamoyl' in '{name}'"
        assert "propanamide" == name, f"Expected 'propanamide', got '{name}'"
