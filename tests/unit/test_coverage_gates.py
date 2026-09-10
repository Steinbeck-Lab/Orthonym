"""Tests for coverage estimation utilities, NP scaffold gates, composer gates, and quality gates.

Tests coverage_utils.py functions:
- estimate_parent_coverage: atom-index-based coverage ratio

Task Z2 (2026-08-02) removed the tests for ``estimate_name_coverage_heuristic``
along with the function itself: it had zero callers and measured a name's
character count. Its tests did not protect anything -- they PINNED the defect,
one of them asserting outright that "longer names should give higher coverage".
See internal notes

Tests NP scaffold gating:
- Large molecule + small NP scaffold -> name_natural_product returns None
- NP scaffold covering most of molecule -> returns scaffold name
- Small molecule -> always returns name regardless of coverage

Tests composer coverage gates (benzene, heterocycle, complex ring):
- Small molecules pass through gates unconditionally
- Large molecules with bare ring names are rejected (fall through)
- Large molecules with complete names pass through

Tests decomposition quality gate:
- Tightened threshold (heavy_atoms // 2) rejects short names for large molecules
- Small molecules with short names still pass
- Fragment context uses appropriate thresholds

Tests coverage heuristic edge cases:
- Empty name -> 0.0
- Zero heavy atoms -> 1.0

Tests retained name exact-match (NAM-01):
- Exact benzene SMILES -> "benzene"
- Toluene SMILES -> NOT bare "benzene"
- Large benzene-containing molecule -> NOT bare "benzene"
"""

import pytest
from rdkit import Chem

from orthonym.assembly.coverage_utils import estimate_parent_coverage


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
# NP scaffold coverage gate tests (behavioral -- test via name_natural_product)
# ---------------------------------------------------------------------------

class TestNPScaffoldCoverageGate:
    """Tests that NP scaffold coverage gates reject bare names for oversized molecules."""

    def test_large_molecule_small_np_scaffold_with_numbering_names_substituents(self):
        """Tropane (9 atoms) in a 25-atom molecule with numbering map -> names substituents.

        CN1C2CCCC1CC2 is detected as tropane scaffold (9 heavy atoms).
        Adding a C16 chain yields 25 heavy atoms, coverage = 9/25 = 0.36.
        With tropane numbering map (a phase-02), the NP pipeline can enumerate
        substituents instead of falling through the coverage gate.
        """
        from orthonym.rules.natural_products import name_natural_product

        # Tropane core + C16 chain (25 heavy atoms, 9 matched)
        smiles = "CN1C2CCCC1CC2CCCCCCCCCCCCCCCC"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, "Could not parse tropane+chain SMILES"

        total_heavy = mol.GetNumHeavyAtoms()
        assert total_heavy > 10, f"Expected >10 heavy atoms, got {total_heavy}"

        result = name_natural_product(mol)
        # With numbering map, tropane names substituents instead of hitting coverage gate
        assert result is not None, "Tropane with numbering map should name substituents"
        assert "tropan" in result.lower(), f"Expected 'tropan' in: {result}"

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


# ---------------------------------------------------------------------------
# Benzene coverage gate tests (composer)
# ---------------------------------------------------------------------------

class TestBenzeneCoverageGate:
    """Tests that benzene handler's coverage gate works correctly."""

    def test_benzene_coverage_gate_small_molecule(self):
        """Small benzene derivative (8 heavy atoms) should NOT be gated.

        Ethylbenzene has 8 heavy atoms, well below the 10-atom guard.
        The benzene handler should return a benzene-based name.
        """
        from orthonym import name_compound

        smiles = "c1ccc(CC)cc1"  # ethylbenzene, 8 heavy atoms
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert mol.GetNumHeavyAtoms() == 8

        result = name_compound(smiles)
        assert result is not None and result != "unknown"
        # Should contain 'benzene' or 'ethylbenzene' - a benzene-based name
        assert "benz" in result.lower() or "ethyl" in result.lower(), (
            f"Small benzene molecule should produce benzene-based name, got '{result}'"
        )

    def test_benzene_coverage_gate_large_molecule_adequate_name(self):
        """Large benzene derivative with complete substituent naming should pass gate.

        Icosylbenzene has 26 heavy atoms and the benzene handler produces
        'icosylbenzene' which has adequate length (>= 0.4 * 26 = 10.4 chars).
        """
        from orthonym import name_compound

        smiles = "c1ccc(CCCCCCCCCCCCCCCCCCCC)cc1"  # benzene + C20 chain
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        total_heavy = mol.GetNumHeavyAtoms()
        assert total_heavy >= 25, f"Expected >=25 heavy atoms, got {total_heavy}"

        result = name_compound(smiles)
        assert result is not None and result != "unknown"
        # The name should be adequate length (not just "benzene")
        assert result != "benzene", (
            f"Large molecule should not return bare 'benzene', got '{result}'"
        )
        # Name length should reflect molecule size
        assert len(result) >= total_heavy * 0.4, (
            f"Name '{result}' ({len(result)} chars) too short for {total_heavy} atoms"
        )


# ---------------------------------------------------------------------------
# Heterocycle coverage gate tests (composer)
# ---------------------------------------------------------------------------

class TestHeterocycleCoverageGate:
    """Tests that heterocycle handler's coverage gate works correctly."""

    def test_heterocycle_gate_small(self):
        """Simple pyridine (6 heavy atoms) should not be gated."""
        from orthonym import name_compound

        smiles = "c1ccncc1"  # pyridine, 6 heavy atoms
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert mol.GetNumHeavyAtoms() == 6

        result = name_compound(smiles)
        assert result == "pyridine", f"Expected 'pyridine', got '{result}'"

    def test_heterocycle_gate_large_adequate_name(self):
        """Large heterocycle derivative with substituent naming should pass.

        3-nonadecylpyridine: pyridine (6 atoms) + C19 chain = 25 heavy atoms.
        The heterocycle handler should generate an adequate name with locant
        and substituent prefix.
        """
        from orthonym import name_compound

        smiles = "c1ccncc1CCCCCCCCCCCCCCCCCCC"  # pyridine + C19 chain
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        total_heavy = mol.GetNumHeavyAtoms()
        assert total_heavy >= 20, f"Expected >=20 heavy atoms, got {total_heavy}"

        result = name_compound(smiles)
        assert result is not None and result != "unknown"
        # Should not be bare "pyridine"
        assert result != "pyridine", (
            f"Large molecule should not return bare 'pyridine', got '{result}'"
        )
        # Name should have locant and substituent info
        assert len(result) > len("pyridine"), (
            f"Name should be longer than bare 'pyridine', got '{result}'"
        )


# ---------------------------------------------------------------------------
# Decomposition quality gate tests
# ---------------------------------------------------------------------------

class TestQualityGate:
    """Tests for _name_quality_is_acceptable in decomposition engine."""

    def test_quality_gate_tightened(self):
        """'naphthalene' for a 30-atom molecule should return False.

        With the tightened threshold (heavy_atoms // 2), 'naphthalene' (11 chars)
        is less than 30 // 2 = 15, so it should be flagged as unacceptable.
        """
        from orthonym.decomposition.engine import _name_quality_is_acceptable

        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1CCCCCCCCCCCCCCCCCCCC")
        assert mol is not None
        total_heavy = mol.GetNumHeavyAtoms()
        assert total_heavy >= 30, f"Expected >=30 heavy atoms, got {total_heavy}"

        result = _name_quality_is_acceptable("naphthalene", mol)
        assert result is False, (
            f"'naphthalene' for {total_heavy}-atom mol should be rejected"
        )

    def test_quality_gate_small_molecule(self):
        """Short names for small molecules (heavy_atoms <= 15) still pass.

        'decane' (6 chars) for a 10-atom molecule should be acceptable
        because the molecule is too small to trigger any threshold.
        """
        from orthonym.decomposition.engine import _name_quality_is_acceptable

        mol = Chem.MolFromSmiles("CCCCCCCCCC")  # decane, 10 heavy atoms
        assert mol is not None
        assert mol.GetNumHeavyAtoms() == 10

        result = _name_quality_is_acceptable("decane", mol)
        assert result is True, (
            "'decane' for 10-atom molecule should be acceptable"
        )

    def test_quality_gate_adequate_name_large_mol(self):
        """A name with locants and hyphens for a large molecule should pass.

        Names with digits, hyphens, and adequate length pass the gate.
        """
        from orthonym.decomposition.engine import _name_quality_is_acceptable

        # 20-atom molecule with adequate name
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCCCCCC")  # icosane
        assert mol is not None
        assert mol.GetNumHeavyAtoms() == 20

        # A name with digits and hyphens that is long enough
        result = _name_quality_is_acceptable("3-methylnonadecane", mol)
        assert result is True, (
            "'3-methylnonadecane' for 20-atom mol should be acceptable"
        )

    def test_quality_gate_bare_name_large_mol(self):
        """Bare ring name (no digits/hyphens) for 25-atom mol -> rejected.

        'naphthalene' has no digits or hyphens but molecule is > 20 atoms.
        """
        from orthonym.decomposition.engine import _name_quality_is_acceptable

        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1CCCCCCCCCCCCCCC")
        assert mol is not None
        total_heavy = mol.GetNumHeavyAtoms()
        assert total_heavy > 20

        result = _name_quality_is_acceptable("naphthalene", mol)
        assert result is False, (
            f"Bare 'naphthalene' for {total_heavy}-atom mol should be rejected"
        )

    def test_quality_gate_none_name(self):
        """None name -> False (always triggers decomposition)."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable

        mol = Chem.MolFromSmiles("CCO")
        result = _name_quality_is_acceptable(None, mol)
        assert result is False

    def test_quality_gate_unknown_name(self):
        """'unknown' name -> False (always triggers decomposition)."""
        from orthonym.decomposition.engine import _name_quality_is_acceptable

        mol = Chem.MolFromSmiles("CCO")
        result = _name_quality_is_acceptable("unknown", mol)
        assert result is False

    def test_quality_gate_ratio_check_very_large_mol(self):
        """For >25 atom molecule, ratio < 0.45 triggers rejection.

        A 30-atom molecule with a 12-char name (ratio 0.40) should fail.
        """
        from orthonym.decomposition.engine import _name_quality_is_acceptable

        # 30-atom molecule
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
        assert mol is not None
        total_heavy = mol.GetNumHeavyAtoms()
        assert total_heavy == 30

        # Task Z3: the probe used to be 'triacontane', which is this molecule's
        # own correct systematic name (C30 unbranched alkane) -- so the test was
        # asserting that a correct name must be rejected, on a character count.
        # 'decane' has the same shape for the guard (6 chars, ratio 0.20 < 0.45,
        # no digits, no hyphens) but describes only 10 of the 30 atoms, so the
        # rejection it pins is a true positive.
        result = _name_quality_is_acceptable("decane", mol)
        assert result is False, (
            f"'decane' (ratio {len('decane')/total_heavy:.2f}) for "
            f"{total_heavy}-atom mol should be rejected"
        )
        # The molecule's own correct name is NOT rejected.
        assert _name_quality_is_acceptable("triacontane", mol) is True


# ---------------------------------------------------------------------------
# Coverage heuristic edge cases -- REMOVED (Task Z2, 2026-08-02)
# ---------------------------------------------------------------------------
# Three tests for estimate_name_coverage_heuristic stood here and went with the
# function. They are worth recording as a pattern rather than just deleting:
# all three passed, and all three asserted the defect. The clearest was
# `test_heuristic_monotonic_with_name_length` -- "Longer names should give
# higher coverage" -- which is precisely the anti-correlation that makes the
# quantity useless: a name that INVENTS atoms scores higher than a correct one.
# Green tests over a meaningless quantity are not a safety net.
