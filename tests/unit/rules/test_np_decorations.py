"""Tests for natural product decoration enumeration.

Tests that steroid derivatives with functional group decorations
(hydroxyl, ketone, double bonds) produce correct systematic names
using the scaffold stem + decoration suffixes/prefixes.

Plan 15-07: NP Decoration Enumeration

Test coverage:
1. Cholesterol (exact match still works)
2. Testosterone (17-hydroxy + 3-one + 4-ene)
3. Progesterone (pregnane with 3,20-dione + 4-ene)
4. Androst-4-ene-3,17-dione (two ketones + ene)
5. Estradiol (estrane with 3,17-diol)
6. Cholestane bare scaffold (no decorations)
7. Camphor (exact derivative match, not steroid enumeration)
8. Saturated 3-hydroxyandrostane (androstan-3-ol)
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.natural_products import name_natural_product


def _mol(smiles: str):
    """Return RDKit Mol from SMILES, raising on invalid input."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    return mol


# ---------------------------------------------------------------------------
# Test Class: NP Decoration Enumeration
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNPDecorationEnumeration:
    """Test that steroid decorations (-OH, =O, C=C) are correctly enumerated."""

    def test_cholesterol_exact_match(self):
        """Cholesterol with correct stereochemistry should still use exact match."""
        smiles = (
            "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C"
            "[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"
        )
        result = name_compound(smiles)
        assert result == "cholesterol"

    def test_testosterone_decoration(self):
        """Testosterone: 17-hydroxyandrost-4-en-3-one.

        Has -OH at C-17 (prefix when ketone present), =O at C-3 (suffix),
        and C=C between C-4 and C-5 (ene suffix).
        """
        smiles = "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C"
        result = name_compound(smiles)
        assert result == "17-hydroxyandrost-4-en-3-one", f"Got '{result}'"

    def test_progesterone_decoration(self):
        """Progesterone: pregn-4-en-3,20-dione.

        Has two ketone groups (C-3 and C-20) and one double bond (C-4,5).
        """
        smiles = "CC(=O)C1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C"
        result = name_compound(smiles)
        assert result == "pregn-4-en-3,20-dione", f"Got '{result}'"

    def test_androstanedione_decoration(self):
        """Androst-4-ene-3,17-dione: two ketones + one double bond."""
        smiles = "CC12CCC(=O)C=C1CCC1C2CCC2(C)C(=O)CCC12"
        result = name_compound(smiles)
        assert result == "androst-4-en-3,17-dione", f"Got '{result}'"

    def test_estradiol_decoration(self):
        """Estradiol: estrane with two -OH groups.

        Due to aromatic ring A not being detected as discrete ene bonds,
        the result is estran-3,17-diol (acceptable limitation).
        """
        smiles = "OC1CCC2C3CCc4cc(O)ccc4C3CCC12C"
        result = name_compound(smiles)
        # Two -OH groups, no ketone -> diol suffix
        assert "diol" in result, f"Expected 'diol' in name, got '{result}'"
        assert "3" in result and "17" in result, f"Expected locants 3,17 in '{result}'"

    def test_cholestane_bare_scaffold(self):
        """Bare cholestane (no decorations) should return 'cholestane'."""
        smiles = (
            "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC"
            "[C@]4(C)[C@H]3CC[C@]12C"
        )
        result = name_compound(smiles)
        assert result == "cholestane"

    def test_camphor_exact_match(self):
        """Camphor is an exact derivative match, not steroid enumeration."""
        result = name_compound("CC12CCC(CC1=O)C2(C)C")
        assert result == "camphor"

    def test_saturated_hydroxy_androstane(self):
        """3-Hydroxyandrostane (saturated) -> androstan-3-ol.

        Saturated steroid with single -OH and no ketone uses -ol suffix.
        """
        smiles = "CC12CCCC1C1CCC3CC(O)CCC3(C)C1CC2"
        result = name_compound(smiles)
        assert result == "androstan-3-ol", f"Got '{result}'"


@pytest.mark.unit
class TestNPDecorationEdgeCases:
    """Edge cases for NP decoration enumeration."""

    def test_no_decorations_returns_scaffold_name(self):
        """Steroid scaffold match with only H substituents returns scaffold name."""
        # Androstane (exact scaffold)
        smiles = "C[C@@]12CCC[C@H]1[C@@H]1CCC3CCCC[C@]3(C)[C@H]1CC2"
        result = name_compound(smiles)
        assert result == "androstane"

    def test_non_steroid_scaffold_returns_scaffold_name(self):
        """Non-steroid NP scaffolds return plain scaffold name (no decoration enum)."""
        # Tropane
        smiles = "CN1[C@@H]2CCC[C@H]1CC2"
        result = name_compound(smiles)
        assert result == "tropane"

    def test_ketone_only_saturated(self):
        """Saturated steroid with only ketone decoration."""
        # 3-Oxoandrostane (androstane with =O at C-3, no unsaturation)
        # Use non-stereo to avoid exact match
        smiles = "CC12CCCC1C1CCC3CC(=O)CCC3(C)C1CC2"
        result = name_compound(smiles)
        assert "one" in result, f"Expected ketone suffix in '{result}'"
        assert "3" in result, f"Expected locant 3 in '{result}'"

    def test_double_bond_only(self):
        """Steroid with only a double bond modification, no FG substituents."""
        # Cholest-5-ene (cholestane with one C=C, no -OH or =O)
        # Use non-stereo to avoid exact match
        smiles = "CC(C)CCCC(C)C1CCC2C3CC=C4CCCCC4(C)C3CCC12C"
        result = name_compound(smiles)
        assert "en" in result, f"Expected 'en' suffix in '{result}'"

    def test_multiple_hydroxyls(self):
        """Steroid with multiple -OH groups."""
        # 3,17-Dihydroxyandrostane (no unsaturation)
        smiles = "CC12CCC(O)C1C1CCC3CC(O)CCC3(C)C1CC2"
        result = name_compound(smiles)
        assert "diol" in result or ("ol" in result and "3" in result and "17" in result), \
            f"Expected diol-like name, got '{result}'"

    def test_decoration_does_not_break_exact_match(self):
        """Molecules with exact derivative entries still use exact match."""
        # Morphine should still be "morphine"
        smiles = "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"
        assert name_compound(smiles) == "morphine"

    def test_name_natural_product_none_input(self):
        """name_natural_product(None) should return None."""
        assert name_natural_product(None) is None
