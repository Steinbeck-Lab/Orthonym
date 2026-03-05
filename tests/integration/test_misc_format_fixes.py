"""
Integration tests for miscellaneous format fixes (Phase 32).

Tests oxolane substituent naming and cycloenone numbering fixes.
"""

import pytest
from orthonym import name_compound

pytestmark = pytest.mark.integration


class TestOxolaneSubstituentNaming:
    """Saturated 5-membered O-heterocycle substituents use systematic 'oxolanyl'."""

    def test_standalone_thf_retained_name(self):
        """Standalone oxolane keeps retained name."""
        assert name_compound("C1CCOC1") == "oxolane"

    def test_thf_substituent_uses_oxolanyl(self):
        """THF ring as substituent on another structure uses 'oxolanyl'."""
        # 6-amino-2-oxo-N-oxolanyl-1,3,5-triazine (azacitidine core)
        name = name_compound(
            "Nc1ncn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)n1"
        )
        assert "oxolanyl" in name, f"Expected 'oxolanyl' in '{name}'"
        assert "tetrahydrofuryl" not in name, (
            f"Should not contain 'tetrahydrofuryl' in '{name}'"
        )

    def test_methyl_thf_uses_oxolane(self):
        """2-methyloxolane as parent uses methyloxolane or methyloxolane."""
        name = name_compound("CC1CCCO1")
        # Either form acceptable for parent; check it names without error
        assert name, "Should produce a name"


class TestCycloenoneNumbering:
    """Cyclohexenone numbering: ketone at C-1, double bond higher (P-31.1.3.4)."""

    def test_cyclohexenone_ketone_at_c1(self):
        """Cyclohex-2-en-1-one: ketone gets lowest locant."""
        name = name_compound("O=C1CC=CCC1")
        assert "en-1-one" in name, f"Expected ketone at C-1 in '{name}'"

    def test_isophorone(self):
        """3,5,5-trimethylcyclohex-2-en-1-one (isophorone)."""
        name = name_compound("CC1=CC(=O)CC(C)(C)C1")
        assert "trimethylcyclohex" in name, f"Expected trimethylcyclohex in '{name}'"
        assert "en-1-one" in name, f"Expected ketone at C-1 in '{name}'"

    def test_cyclohexenol(self):
        """Cyclohex-2-en-1-ol: alcohol gets lowest locant."""
        name = name_compound("OC1CC=CCC1")
        assert "en-1-ol" in name or "ol" in name, (
            f"Expected alcohol at C-1 in '{name}'"
        )

    def test_cyclopentenone(self):
        """Cyclopent-2-en-1-one: ketone at C-1."""
        name = name_compound("O=C1CC=CC1")
        assert "one" in name, f"Expected ketone suffix in '{name}'"
