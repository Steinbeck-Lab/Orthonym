"""TDD tests for natural product ester decoration naming.

Tests that steroid esters (e.g., testosterone acetate) produce names
containing the acylate fragment ('acetate'), using functional class format
('parent-yl acylate') per IUPAC.

RED phase: ester tests should FAIL initially (NP naming currently drops esters).
Regression tests should PASS (bare steroid naming must not break).
"""

import pytest
from orthonym import name_compound


# ---------------------------------------------------------------------------
# Core steroid ester tests (expected to FAIL in RED phase)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSteroidEsterNaming:
    """Tests for steroid ester naming via functional class format."""

    def test_testosterone_acetate_contains_acetate(self):
        """Testosterone acetate: ester fragment 'acetate' must appear in name."""
        # testosterone 17-acetate
        smiles = "CC(=O)O[C@H]1CC[C@@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@@H]3CC[C@]12C"
        name = name_compound(smiles)
        assert name is not None, "Should produce a name"
        assert "acetate" in name.lower(), (
            f"Name '{name}' must contain 'acetate' (ester fragment preserved)"
        )

    def test_testosterone_acetate_contains_androst(self):
        """Testosterone acetate: steroid scaffold 'androst' must appear in name."""
        smiles = "CC(=O)O[C@H]1CC[C@@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@@H]3CC[C@]12C"
        name = name_compound(smiles)
        assert name is not None
        assert "androst" in name.lower(), (
            f"Name '{name}' must contain 'androst' (steroid scaffold preserved)"
        )

    def test_testosterone_acetate_contains_oxo_or_one(self):
        """Testosterone acetate: ketone decoration must be preserved."""
        smiles = "CC(=O)O[C@H]1CC[C@@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@@H]3CC[C@]12C"
        name = name_compound(smiles)
        assert name is not None
        # In functional class format, ketone becomes 'oxo' prefix or '-one' suffix
        has_oxo = "oxo" in name.lower()
        has_one = "-one" in name.lower() or name.lower().endswith("one")
        assert has_oxo or has_one, (
            f"Name '{name}' must contain 'oxo' or '-one' (ketone decoration preserved)"
        )

    def test_testosterone_acetate_functional_class_format(self):
        """Testosterone acetate uses functional class format: parent-yl acylate."""
        smiles = "CC(=O)O[C@H]1CC[C@@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@@H]3CC[C@]12C"
        name = name_compound(smiles)
        assert name is not None
        # Functional class format: "...yl acetate"
        assert "-yl " in name.lower() or "-yl\t" in name.lower(), (
            f"Name '{name}' should use functional class format with '-yl' "
            f"followed by acylate name"
        )

    def test_testosterone_propanoate(self):
        """Testosterone propanoate: name must contain 'propanoate'."""
        # Propanoyl ester at C17 instead of acetyl
        smiles = "CCC(=O)O[C@H]1CC[C@@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@@H]3CC[C@]12C"
        name = name_compound(smiles)
        assert name is not None, "Should produce a name"
        assert "propanoate" in name.lower() or "propionate" in name.lower(), (
            f"Name '{name}' must contain 'propanoate' (C3 acid fragment)"
        )


# ---------------------------------------------------------------------------
# Regression tests (MUST pass even in RED phase)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSteroidNamingRegression:
    """Regression tests: existing NP naming must not break."""

    def test_bare_androstane_still_works(self):
        """Bare androstane scaffold: no ester, no crash."""
        smiles = "C[C@@]12CCC[C@H]1[C@@H]1CCC3CCCC[C@]3(C)[C@H]1CC2"
        name = name_compound(smiles)
        assert name is not None
        assert "androstane" in name.lower(), (
            f"Bare androstane should produce name containing 'androstane', got '{name}'"
        )

    def test_testosterone_hydroxyl_naming_intact(self):
        """Testosterone (OH at 17, not ester): hydroxy/oxo decorations preserved."""
        smiles = "O[C@@H]1CC[C@@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@@H]3CC[C@]12C"
        name = name_compound(smiles)
        assert name is not None
        # Should have both hydroxy and oxo/one
        name_lower = name.lower()
        assert "androst" in name_lower, f"Expected 'androst' in '{name}'"
        has_hydroxy = "hydroxy" in name_lower
        has_ol = "-ol" in name_lower or name_lower.endswith("ol")
        assert has_hydroxy or has_ol, (
            f"Testosterone should have 'hydroxy' or '-ol', got '{name}'"
        )

    def test_cholesterol_exact_retained_name(self):
        """Cholesterol: exact retained name must be preserved."""
        smiles = "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"
        name = name_compound(smiles)
        assert name is not None
        assert name.lower() == "cholesterol", (
            f"Cholesterol should return exact retained name, got '{name}'"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNPEsterEdgeCases:
    """Edge case tests for NP ester decoration."""

    def test_non_steroid_np_with_ester_no_crash(self):
        """Non-steroid NP with ester: should not crash, falls back gracefully."""
        # Camphor is a terpenoid, not a steroid - it has an exact retained name
        # Let's use a non-NP ester to verify no interference
        # Simple cyclohexyl acetate - not an NP scaffold
        smiles = "CC(=O)OC1CCCCC1"
        name = name_compound(smiles)
        # Should produce some name (not crash), and NOT go through NP ester path
        assert name is not None, "Non-steroid ester must not crash"
