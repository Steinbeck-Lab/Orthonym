"""Unit tests for natural product modification prefix detection and formatting.

Tests the detect_np_modifications() and format_np_modification_prefix() functions
that implement IUPAC nor-, homo-, seco- modification prefixes for natural product
scaffolds per P-10/P-31.1.3 and steroid nomenclature rules 3S-7, 3S-8.

Test classes:
- TestDetectNpModifications: Detection of nor-/homo-/seco- structural changes
- TestFormatNpModificationPrefix: Prefix string formatting per IUPAC
- TestNameNaturalProductWithModifications: Integration with naming pipeline
"""

import pytest
from rdkit import Chem

from orthonym.rules.natural_products import (
    detect_np_modifications,
    format_np_modification_prefix,
    name_natural_product,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _mol(smiles: str):
    """Return RDKit Mol from SMILES, raising on invalid input."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    return mol


# ---------------------------------------------------------------------------
# SMILES constants
# ---------------------------------------------------------------------------

# Bare scaffolds
ANDROSTANE_SMILES = (
    "C[C@@]12CCC[C@H]1[C@@H]1CCC3CCCC[C@]3(C)[C@H]1CC2"
)
ESTRANE_SMILES = (
    "C[C@@]12CCC[C@H]1[C@@H]1CCC3CCCC[C@@H]3[C@H]1CC2"
)
GONANE_SMILES = (
    "C1CC[C@H]2C(C1)CC[C@H]1[C@@H]3CCC[C@H]3CC[C@@H]12"
)
CHOLESTANE_SMILES = (
    "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C"
)

# Exact derivatives (should NOT be affected by modification detection)
CHOLESTEROL_SMILES = (
    "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C"
    "[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C"
)
MORPHINE_SMILES = (
    "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5"
)


# ===========================================================================
# Test Class 1: detect_np_modifications()
# ===========================================================================

@pytest.mark.unit
class TestDetectNpModifications:
    """Test detection of nor-/homo-/seco- modifications on NP scaffolds."""

    def test_bare_androstane_no_modifications(self):
        """Bare androstane scaffold has no modifications (has both C-18, C-19)."""
        mol = _mol(ANDROSTANE_SMILES)
        from orthonym.perception.natural_products import detect_natural_product
        from orthonym.rules.natural_products import _build_target_to_iupac

        scaffold_info = detect_natural_product(mol)
        assert scaffold_info is not None
        assert scaffold_info["scaffold_name"] == "androstane"
        numbering = _build_target_to_iupac(scaffold_info)
        assert numbering is not None

        modifications = detect_np_modifications(mol, scaffold_info, numbering)
        assert modifications == []

    def test_estrane_no_nor_detected(self):
        """Estrane scaffold's numbering lacks C-19 so no nor- is detected.

        Estrane is structurally 19-norandrostane but has its own retained name.
        Since estrane's numbering map doesn't include locant 19, the nor-
        detection correctly skips it -- the scaffold name "estrane" is used.
        """
        mol = _mol(ESTRANE_SMILES)
        from orthonym.perception.natural_products import detect_natural_product
        from orthonym.rules.natural_products import _build_target_to_iupac

        scaffold_info = detect_natural_product(mol)
        assert scaffold_info is not None
        assert scaffold_info["scaffold_name"] == "estrane"
        numbering = _build_target_to_iupac(scaffold_info)
        assert numbering is not None

        modifications = detect_np_modifications(mol, scaffold_info, numbering)
        # Estrane scaffold numbering lacks C-19, so no nor- detected
        assert modifications == []

    def test_gonane_no_nor_detected(self):
        """Gonane scaffold's numbering lacks C-18 and C-19 so no nor- detected.

        Gonane is the bare tetracyclic steroid with its own name. Its numbering
        map doesn't include locants 18 or 19, so no modification is detected.
        """
        mol = _mol(GONANE_SMILES)
        from orthonym.perception.natural_products import detect_natural_product
        from orthonym.rules.natural_products import _build_target_to_iupac

        scaffold_info = detect_natural_product(mol)
        assert scaffold_info is not None
        assert scaffold_info["scaffold_name"] == "gonane"
        numbering = _build_target_to_iupac(scaffold_info)
        assert numbering is not None

        modifications = detect_np_modifications(mol, scaffold_info, numbering)
        # Gonane numbering lacks both C-18 and C-19
        assert modifications == []

    def test_cholestane_no_modifications(self):
        """Cholestane has both angular methyls -- no nor modifications."""
        mol = _mol(CHOLESTANE_SMILES)
        from orthonym.perception.natural_products import detect_natural_product
        from orthonym.rules.natural_products import _build_target_to_iupac

        scaffold_info = detect_natural_product(mol)
        assert scaffold_info is not None
        numbering = _build_target_to_iupac(scaffold_info)
        assert numbering is not None

        modifications = detect_np_modifications(mol, scaffold_info, numbering)
        assert modifications == []

    def test_androstane_has_both_angular_methyls(self):
        """Androstane molecule has both C-18 and C-19 -- no nor modification."""
        mol = _mol(ANDROSTANE_SMILES)
        from orthonym.perception.natural_products import detect_natural_product
        from orthonym.rules.natural_products import _build_target_to_iupac

        scaffold_info = detect_natural_product(mol)
        assert scaffold_info is not None
        numbering = _build_target_to_iupac(scaffold_info)
        assert numbering is not None

        # Androstane numbering includes locants 18 and 19
        locants = set(numbering.values())
        assert 18 in locants
        assert 19 in locants

        # Both methyls present -> no modifications
        modifications = detect_np_modifications(mol, scaffold_info, numbering)
        assert modifications == []

    def test_format_and_pipeline_integration(self):
        """Integration: format_np_modification_prefix produces correct prefix for
        nor-modification and _assemble_np_name inserts it before the stem."""
        from orthonym.rules.natural_products import _assemble_np_name, format_np_modification_prefix

        modifications = [
            {"prefix": "nor", "locants": [19], "ring_letter": None}
        ]
        mod_prefix = format_np_modification_prefix(modifications)
        assert mod_prefix == "19-nor"

        # Test with _assemble_np_name
        result = _assemble_np_name(
            stem="androst",
            scaffold_name="androstane",
            hydroxyls=[],
            ketones=[3],
            unsaturation={"ene": [4], "yne": []},
            modification_prefix=mod_prefix,
        )
        assert "19-norandrost" in result

    def test_non_np_molecule_returns_none_from_pipeline(self):
        """Molecules that don't match any NP scaffold return None."""
        mol = _mol("CCO")  # ethanol
        result = name_natural_product(mol)
        assert result is None

    def test_benzene_returns_none_from_pipeline(self):
        """Benzene is not a natural product scaffold."""
        mol = _mol("c1ccccc1")
        result = name_natural_product(mol)
        assert result is None


# ===========================================================================
# Test Class 2: format_np_modification_prefix()
# ===========================================================================

@pytest.mark.unit
class TestFormatNpModificationPrefix:
    """Test IUPAC formatting of modification prefixes."""

    def test_single_nor_format(self):
        """Single nor at position 19 -> '19-nor'."""
        modifications = [
            {"prefix": "nor", "locants": [19], "ring_letter": None}
        ]
        result = format_np_modification_prefix(modifications)
        assert result == "19-nor"

    def test_dinor_format(self):
        """Two nor positions -> '18,19-dinor'."""
        modifications = [
            {"prefix": "nor", "locants": [18, 19], "ring_letter": None}
        ]
        result = format_np_modification_prefix(modifications)
        assert result == "18,19-dinor"

    def test_homo_with_ring_letter(self):
        """Homo with ring letter D -> 'D-homo'."""
        modifications = [
            {"prefix": "homo", "locants": [], "ring_letter": "D"}
        ]
        result = format_np_modification_prefix(modifications)
        assert result == "D-homo"

    def test_seco_format(self):
        """Seco at positions 9,10 -> '9,10-seco'."""
        modifications = [
            {"prefix": "seco", "locants": [9, 10], "ring_letter": None}
        ]
        result = format_np_modification_prefix(modifications)
        assert result == "9,10-seco"

    def test_multiple_modifications_alphabetical_order(self):
        """Multiple modifications sorted alphabetically: homo < nor < seco."""
        modifications = [
            {"prefix": "seco", "locants": [9, 10], "ring_letter": None},
            {"prefix": "nor", "locants": [19], "ring_letter": None},
            {"prefix": "homo", "locants": [], "ring_letter": "D"},
        ]
        result = format_np_modification_prefix(modifications)
        # homo before nor before seco, all directly concatenated
        assert result == "D-homo-19-nor-9,10-seco"

    def test_empty_modifications(self):
        """No modifications -> empty string."""
        result = format_np_modification_prefix([])
        assert result == ""

    def test_trinor_format(self):
        """Three nor positions -> 'A,B,C-trinor'."""
        modifications = [
            {"prefix": "nor", "locants": [2, 18, 19], "ring_letter": None}
        ]
        result = format_np_modification_prefix(modifications)
        assert result == "2,18,19-trinor"

    def test_seco_locants_sorted(self):
        """Seco locants always in ascending order."""
        modifications = [
            {"prefix": "seco", "locants": [10, 9], "ring_letter": None}
        ]
        result = format_np_modification_prefix(modifications)
        assert result == "9,10-seco"

    def test_homo_with_locants_no_ring_letter(self):
        """Homo with locant but no ring letter -> '{locant}-homo'."""
        modifications = [
            {"prefix": "homo", "locants": [17], "ring_letter": None}
        ]
        result = format_np_modification_prefix(modifications)
        assert result == "17-homo"


# ===========================================================================
# Test Class 3: Integration with name_natural_product()
# ===========================================================================

@pytest.mark.unit
class TestNameNaturalProductWithModifications:
    """Integration test: modification prefixes in full naming pipeline."""

    def test_estrane_retains_scaffold_name(self):
        """Estrane has its own scaffold name -- no modification prefix added.

        Estrane = 19-norandrostane structurally, but since estrane has its own
        IUPAC retained name and numbering map (without C-19), it is named
        as 'estrane' without a modification prefix.
        """
        mol = _mol(ESTRANE_SMILES)
        result = name_natural_product(mol)
        assert result is not None
        assert result == "estrane"

    def test_gonane_retains_scaffold_name(self):
        """Gonane has its own scaffold name -- no modification prefix added."""
        mol = _mol(GONANE_SMILES)
        result = name_natural_product(mol)
        assert result is not None
        assert result == "gonane"

    def test_androstane_no_modification_prefix(self):
        """Androstane has both angular methyls -- no modification prefix."""
        mol = _mol(ANDROSTANE_SMILES)
        result = name_natural_product(mol)
        assert result is not None
        assert "nor" not in result, f"Unexpected 'nor' in '{result}'"
        assert result == "androstane"

    def test_cholestane_no_modification_prefix(self):
        """Cholestane has both angular methyls -- no modification prefix."""
        mol = _mol(CHOLESTANE_SMILES)
        result = name_natural_product(mol)
        assert result is not None
        assert "nor" not in result, f"Unexpected 'nor' in '{result}'"

    def test_exact_derivative_cholesterol_unaffected(self):
        """Exact derivatives bypass modification detection entirely."""
        mol = _mol(CHOLESTEROL_SMILES)
        result = name_natural_product(mol)
        assert result == "cholesterol"

    def test_exact_derivative_morphine_unaffected(self):
        """Morphine exact derivative unaffected by modification detection."""
        mol = _mol(MORPHINE_SMILES)
        result = name_natural_product(mol)
        assert result == "morphine"

    def test_non_np_ethanol_returns_none(self):
        """Non-NP molecules still return None from name_natural_product."""
        mol = _mol("CCO")
        result = name_natural_product(mol)
        assert result is None

    def test_non_np_benzene_returns_none(self):
        """Benzene is not a natural product scaffold."""
        mol = _mol("c1ccccc1")
        result = name_natural_product(mol)
        assert result is None


# ===========================================================================
# Test Class 4: Full pipeline integration via name_compound()
# ===========================================================================

@pytest.mark.unit
class TestNameNaturalProductIntegration:
    """End-to-end tests via name_compound() for NP modification prefixes."""

    def test_name_compound_estrane_retains_name(self):
        """name_compound() on estrane scaffold produces 'estrane' (retained name)."""
        from orthonym import name_compound
        mol = _mol(ESTRANE_SMILES)
        result = name_compound(Chem.MolToSmiles(mol))
        assert result is not None
        assert result == "estrane"

    def test_name_compound_cholesterol_unchanged(self):
        """name_compound() on cholesterol still returns 'cholesterol'."""
        from orthonym import name_compound
        result = name_compound(CHOLESTEROL_SMILES)
        assert result == "cholesterol"

    def test_modification_prefix_before_stem_in_assembled_name(self):
        """Modification prefix appears before scaffold stem in assembled name."""
        from orthonym.rules.natural_products import _assemble_np_name
        # Test _assemble_np_name directly with modification_prefix
        result = _assemble_np_name(
            stem="androst",
            scaffold_name="androstane",
            hydroxyls=[],
            ketones=[3],
            unsaturation={"ene": [4], "yne": []},
            stereo_prefix="",
            modification_prefix="19-nor",
        )
        # The modification prefix should be before the stem
        assert "19-nor" in result
        # Should produce something like "19-norandrost-4-en-3-one"
        assert result.startswith("19-nor") or "19-norandrost" in result

    def test_assemble_np_name_no_modification(self):
        """_assemble_np_name without modification_prefix works unchanged."""
        from orthonym.rules.natural_products import _assemble_np_name
        result = _assemble_np_name(
            stem="androst",
            scaffold_name="androstane",
            hydroxyls=[],
            ketones=[3],
            unsaturation={"ene": [4], "yne": []},
            stereo_prefix="",
        )
        # Without modification prefix, name starts with stem
        assert result.startswith("androst")
        assert "19-nor" not in result
