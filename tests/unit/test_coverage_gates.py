"""Tests for coverage estimation utilities and NP scaffold coverage gates.

Tests coverage_utils.py functions:
- estimate_parent_coverage: atom-index-based coverage ratio
- estimate_name_coverage_heuristic: name-length-based coverage heuristic

Tests NP scaffold gating:
- Large molecule + small NP scaffold -> name_natural_product returns None
- NP scaffold covering most of molecule -> returns scaffold name
- Small molecule -> always returns name regardless of coverage

Tests retained name exact-match (NAM-01):
- Exact benzene SMILES -> "benzene"
- Toluene SMILES -> NOT bare "benzene"
- Large benzene-containing molecule -> NOT bare "benzene"
"""

import pytest
from rdkit import Chem

from orthonym.assembly.coverage_utils import (
    estimate_parent_coverage,
    estimate_name_coverage_heuristic,
)


# ---------------------------------------------------------------------------
# estimate_parent_coverage tests
# ---------------------------------------------------------------------------

class TestEstimateParentCoverage:
    """Tests for estimate_parent_coverage()."""

    def test_full_coverage_benzene(self):
        """Benzene (6 atoms) with all 6 atoms in parent -> 1.0."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert mol is not None
        parent_atoms = set(range(mol.GetNumHeavyAtoms()))  # all 6
        result = estimate_parent_coverage(mol, parent_atoms)
        assert result == pytest.approx(1.0)

    def test_partial_coverage_ring_in_large_mol(self):
        """Benzene ring (6 atoms) in a 20-atom mol -> 0.3."""
        mol = Chem.MolFromSmiles("c1ccc(CCCCCCCCCCCCCC)cc1")
        assert mol is not None
        total_heavy = mol.GetNumHeavyAtoms()
        # Use first 6 atoms as parent (the ring)
        parent_atoms = set(range(6))
        result = estimate_parent_coverage(mol, parent_atoms)
        assert result == pytest.approx(6.0 / total_heavy, abs=0.01)

    def test_empty_parent_set(self):
        """Empty parent set in a 10-atom mol -> 0.0."""
        mol = Chem.MolFromSmiles("CCCCCCCCCC")  # decane
        assert mol is not None
        result = estimate_parent_coverage(mol, set())
        assert result == pytest.approx(0.0)

    def test_full_molecule_coverage(self):
        """All atoms in parent -> 1.0."""
        mol = Chem.MolFromSmiles("CCCCC")  # pentane, 5 heavy atoms
        assert mol is not None
        parent_atoms = set(range(mol.GetNumHeavyAtoms()))
        result = estimate_parent_coverage(mol, parent_atoms)
        assert result == pytest.approx(1.0)

    def test_zero_heavy_atoms(self):
        """Molecule with zero heavy atoms -> 1.0 (edge case)."""
        # This is degenerate but we handle it gracefully
        mol = Chem.MolFromSmiles("[H][H]")
        if mol is not None and mol.GetNumHeavyAtoms() == 0:
            result = estimate_parent_coverage(mol, set())
            assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# estimate_name_coverage_heuristic tests
# ---------------------------------------------------------------------------

class TestEstimateNameCoverageHeuristic:
    """Tests for estimate_name_coverage_heuristic()."""

    def test_naphthalene_in_large_mol(self):
        """'naphthalene' (12 chars) for a 30-atom mol -> ~0.27."""
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1CCCCCCCCCCCCCCCCCCCC")
        assert mol is not None
        total = mol.GetNumHeavyAtoms()
        result = estimate_name_coverage_heuristic("naphthalene", mol)
        expected = min(len("naphthalene") / total / 1.5, 1.0)
        assert result == pytest.approx(expected, abs=0.05)

    def test_long_name_for_medium_mol(self):
        """Long name for moderate molecule -> higher coverage."""
        # Use a real steroid-like SMILES
        mol = Chem.MolFromSmiles("C1CCC2C(C1)CCC1C2CCC2(C)C1CCC2=O")
        assert mol is not None
        name = "3-hydroxycholest-4-en-17-one"
        result = estimate_name_coverage_heuristic(name, mol)
        # Should be moderate to high
        assert result > 0.3

    def test_short_name_small_mol(self):
        """'ethanol' (7 chars) for a 3-atom mol -> 1.0 (capped)."""
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        result = estimate_name_coverage_heuristic("ethanol", mol)
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# NP scaffold coverage gate tests (behavioral -- test via name_natural_product)
# ---------------------------------------------------------------------------

class TestNPScaffoldCoverageGate:
    """Tests that NP scaffold coverage gates reject bare names for oversized molecules."""

    def test_large_molecule_small_np_scaffold_returns_none(self):
        """Tropane (9 atoms) in a 25-atom molecule (coverage 0.36) -> None.

        CN1C2CCCC1CC2 is detected as tropane scaffold (9 heavy atoms).
        Adding a C16 chain yields 25 heavy atoms, coverage = 9/25 = 0.36.
        The coverage gate should return None instead of bare 'tropane'.
        """
        from orthonym.rules.natural_products import name_natural_product

        # Tropane core + C16 chain (25 heavy atoms, 9 matched)
        smiles = "CN1C2CCCC1CC2CCCCCCCCCCCCCCCC"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, "Could not parse tropane+chain SMILES"

        total_heavy = mol.GetNumHeavyAtoms()
        assert total_heavy > 10, f"Expected >10 heavy atoms, got {total_heavy}"

        result = name_natural_product(mol)
        # Coverage gate should reject bare scaffold name
        assert result is None, (
            f"Expected None for tropane in {total_heavy}-atom molecule, got '{result}'"
        )

    def test_np_scaffold_covering_most_of_molecule_returns_name(self):
        """Pure tropane scaffold (coverage = 1.0) -> returns 'tropane'."""
        from orthonym.rules.natural_products import name_natural_product

        # Pure tropane scaffold (9 heavy atoms, 9 matched)
        smiles = "CN1C2CCCC1CC2"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None

        result = name_natural_product(mol)
        assert result == "tropane", f"Expected 'tropane', got '{result}'"

    def test_small_molecule_never_gated(self):
        """Tropane (9 heavy atoms, <= 10) should never be gated even at low coverage."""
        from orthonym.rules.natural_products import name_natural_product

        # Pure tropane (9 heavy atoms) -- small molecule, should always return name
        smiles = "CN1C2CCCC1CC2"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None

        total_heavy = mol.GetNumHeavyAtoms()
        assert total_heavy <= 10, f"Expected <=10 heavy, got {total_heavy}"

        result = name_natural_product(mol)
        # Small molecule should return the name regardless of coverage
        assert result == "tropane", (
            f"Small molecule tropane ({total_heavy} atoms) should not be gated, got '{result}'"
        )


# ---------------------------------------------------------------------------
# Retained name exact-match tests (NAM-01 validation)
# ---------------------------------------------------------------------------

class TestRetainedNameExactMatch:
    """Tests that retained name lookup is exact SMILES match, not substructure."""

    def test_retained_name_exact_match_benzene(self):
        """Exact benzene SMILES -> 'benzene'."""
        from orthonym.namer import name_compound

        result = name_compound("c1ccccc1")
        assert result == "benzene"

    def test_retained_name_no_substructure_leak_toluene(self):
        """Toluene should NOT return bare 'benzene'."""
        from orthonym.namer import name_compound

        result = name_compound("Cc1ccccc1")
        assert result != "benzene", (
            f"Toluene returned bare 'benzene' -- retained name leak"
        )
        # Should be "toluene" or "methylbenzene"
        assert "benz" in result.lower() or "toluen" in result.lower()

    def test_retained_name_no_substructure_leak_large(self):
        """Benzene + decyl chain should NOT return bare 'benzene'."""
        from orthonym.namer import name_compound

        smiles = "c1ccc(CCCCCCCCCC)cc1"  # benzene + decyl chain, ~16 heavy atoms
        result = name_compound(smiles)
        assert result != "benzene", (
            f"Large molecule returned bare 'benzene' -- retained name leak"
        )
