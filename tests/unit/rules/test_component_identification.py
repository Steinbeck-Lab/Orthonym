"""
Unit tests for monocyclic component identification in fusion naming.

Tests that _identify_ring_name() correctly distinguishes:
- Carbocyclic rings (benzene, cyclopentadiene)
- Single-heteroatom rings (furan, thiophene, pyrrole, pyridine)
- 2-heteroatom 5-membered positional isomers (imidazole/pyrazole, oxazole/isoxazole, thiazole/isothiazole)
- 2-heteroatom 6-membered positional isomers (pyrimidine/pyrazine/pyridazine)
- Parent/child seniority (heterocyclic > carbocyclic, N > O > S)
"""

import pytest
from rdkit import Chem

from src.orthonym.rules.fusion_descriptors import (
    _identify_ring_name,
    identify_parent_and_child,
)
from src.orthonym.data.fusion_components import (
    MONOCYCLIC_COMPONENTS,
    get_component_by_pattern,
    get_component_seniority,
    get_component_prefix,
)


# ============================================================================
# Helper to extract ring atoms from a simple monocyclic molecule
# ============================================================================

def _get_ring(smiles):
    """Get the first ring atom list from a SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    ri = mol.GetRingInfo()
    rings = ri.AtomRings()
    assert len(rings) >= 1, f"No rings in {smiles}"
    return mol, list(rings[0])


def _get_two_rings(smiles):
    """Get mol and two ring sets from a fused bicyclic SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    ri = mol.GetRingInfo()
    rings = ri.AtomRings()
    assert len(rings) >= 2, f"Need 2+ rings in {smiles}, got {len(rings)}"
    return mol, set(rings[0]), set(rings[1])


# ============================================================================
# Registry sanity checks
# ============================================================================

class TestComponentRegistry:
    """Tests for MONOCYCLIC_COMPONENTS registry."""

    def test_registry_has_minimum_entries(self):
        """Registry should have at least 16 monocyclic components."""
        assert len(MONOCYCLIC_COMPONENTS) >= 16

    def test_all_entries_have_required_keys(self):
        """Every entry must have prefix, ring_size, heteroatoms, hetero_positions, seniority, aromatic."""
        required = {'prefix', 'ring_size', 'heteroatoms', 'hetero_positions', 'seniority', 'aromatic'}
        for name, props in MONOCYCLIC_COMPONENTS.items():
            for key in required:
                assert key in props, f"Missing '{key}' in {name}"

    def test_seniority_ordering(self):
        """N-heterocycles should be more senior (lower value) than O/S/carbocyclic."""
        assert get_component_seniority('pyrimidine') < get_component_seniority('furan')
        assert get_component_seniority('pyridine') < get_component_seniority('thiophene')
        assert get_component_seniority('furan') < get_component_seniority('benzene')
        assert get_component_seniority('thiophene') < get_component_seniority('benzene')

    def test_unknown_component_seniority(self):
        """Unknown component should return high seniority value (999)."""
        assert get_component_seniority('unknownring') == 999

    def test_component_prefix_lookup(self):
        """Prefix forms should be correct."""
        assert get_component_prefix('benzene') == 'benzo'
        assert get_component_prefix('furan') == 'furo'
        assert get_component_prefix('pyrrole') == 'pyrrolo'
        assert get_component_prefix('pyridine') == 'pyrido'
        assert get_component_prefix('pyrimidine') == 'pyrimido'


# ============================================================================
# Carbocyclic identification
# ============================================================================

class TestCarbocyclicIdentification:
    """Tests for carbocyclic ring identification."""

    def test_benzene_identified(self):
        """6-membered aromatic all-carbon ring -> 'benzene'."""
        mol, ring = _get_ring('c1ccccc1')
        assert _identify_ring_name(mol, ring) == 'benzene'

    def test_cyclopentadiene_identified(self):
        """5-membered aromatic all-carbon ring -> 'cyclopentadiene'."""
        # cyclopentadienyl anion (aromatic)
        mol = Chem.MolFromSmiles('[cH-]1cccc1')
        if mol is not None:
            ri = mol.GetRingInfo()
            rings = ri.AtomRings()
            if rings:
                name = _identify_ring_name(mol, list(rings[0]))
                assert name == 'cyclopentadiene'


# ============================================================================
# Single-heteroatom heterocycle identification
# ============================================================================

class TestSingleHeteroatomIdentification:
    """Tests for single-heteroatom heterocyclic ring identification."""

    def test_furan_identified(self):
        """5-ring with 1 O -> 'furan'."""
        mol, ring = _get_ring('c1ccoc1')
        assert _identify_ring_name(mol, ring) == 'furan'

    def test_thiophene_identified(self):
        """5-ring with 1 S -> 'thiophene'."""
        mol, ring = _get_ring('c1ccsc1')
        assert _identify_ring_name(mol, ring) == 'thiophene'

    def test_pyrrole_identified(self):
        """5-ring with 1 N -> 'pyrrole'."""
        mol, ring = _get_ring('c1cc[nH]c1')
        assert _identify_ring_name(mol, ring) == 'pyrrole'

    def test_pyridine_identified(self):
        """6-ring with 1 N -> 'pyridine'."""
        mol, ring = _get_ring('c1ccncc1')
        assert _identify_ring_name(mol, ring) == 'pyridine'


# ============================================================================
# 6-membered 2N positional isomers (CRITICAL BUG FIX)
# ============================================================================

class TestDiazineIsomers:
    """Tests for pyrimidine vs pyrazine vs pyridazine (all 2N 6-membered)."""

    def test_pyrimidine_identified(self):
        """1,3-diazine -> 'pyrimidine' (N separated by 1 C)."""
        mol, ring = _get_ring('c1ccncn1')
        assert _identify_ring_name(mol, ring) == 'pyrimidine'

    def test_pyrazine_identified(self):
        """1,4-diazine -> 'pyrazine' (N separated by 2 C)."""
        mol, ring = _get_ring('c1cnccn1')
        assert _identify_ring_name(mol, ring) == 'pyrazine'

    def test_pyridazine_identified(self):
        """1,2-diazine -> 'pyridazine' (adjacent N)."""
        mol, ring = _get_ring('c1ccnnc1')
        assert _identify_ring_name(mol, ring) == 'pyridazine'

    def test_all_three_diazines_distinct(self):
        """All three 2N 6-membered isomers must produce different names."""
        names = set()
        for smiles in ['c1ccncn1', 'c1cnccn1', 'c1ccnnc1']:
            mol, ring = _get_ring(smiles)
            names.add(_identify_ring_name(mol, ring))
        assert len(names) == 3, f"Expected 3 distinct names, got {names}"
        assert names == {'pyrimidine', 'pyrazine', 'pyridazine'}


# ============================================================================
# 5-membered 2N positional isomers (CRITICAL BUG FIX)
# ============================================================================

class TestDiazoleIsomers:
    """Tests for imidazole vs pyrazole (both 2N 5-membered)."""

    def test_imidazole_identified(self):
        """1,3-diazole -> 'imidazole' (N separated by 1 C)."""
        mol, ring = _get_ring('c1cnc[nH]1')
        assert _identify_ring_name(mol, ring) == 'imidazole'

    def test_pyrazole_identified(self):
        """1,2-diazole -> 'pyrazole' (adjacent N)."""
        mol, ring = _get_ring('c1cc[nH]n1')
        assert _identify_ring_name(mol, ring) == 'pyrazole'

    def test_imidazole_pyrazole_distinct(self):
        """Imidazole and pyrazole must produce different names."""
        mol_im, ring_im = _get_ring('c1cnc[nH]1')
        mol_pz, ring_pz = _get_ring('c1cc[nH]n1')
        name_im = _identify_ring_name(mol_im, ring_im)
        name_pz = _identify_ring_name(mol_pz, ring_pz)
        assert name_im != name_pz
        assert name_im == 'imidazole'
        assert name_pz == 'pyrazole'


# ============================================================================
# 5-membered N+O positional isomers
# ============================================================================

class TestOxazoleIsomers:
    """Tests for oxazole vs isoxazole (N+O 5-membered)."""

    def test_oxazole_identified(self):
        """O and N separated by 1 C -> 'oxazole'."""
        mol, ring = _get_ring('c1cocn1')
        assert _identify_ring_name(mol, ring) == 'oxazole'

    def test_isoxazole_identified(self):
        """O and N adjacent -> 'isoxazole'."""
        mol, ring = _get_ring('c1ccno1')
        assert _identify_ring_name(mol, ring) == 'isoxazole'

    def test_oxazole_isoxazole_distinct(self):
        """Oxazole and isoxazole must produce different names."""
        mol_ox, ring_ox = _get_ring('c1cocn1')
        mol_ix, ring_ix = _get_ring('c1ccno1')
        assert _identify_ring_name(mol_ox, ring_ox) != _identify_ring_name(mol_ix, ring_ix)


# ============================================================================
# 5-membered N+S positional isomers
# ============================================================================

class TestThiazoleIsomers:
    """Tests for thiazole vs isothiazole (N+S 5-membered)."""

    def test_thiazole_identified(self):
        """S and N separated by 1 C -> 'thiazole'."""
        mol, ring = _get_ring('c1cscn1')
        assert _identify_ring_name(mol, ring) == 'thiazole'

    def test_isothiazole_identified(self):
        """S and N adjacent -> 'isothiazole'."""
        mol, ring = _get_ring('c1ccsn1')
        assert _identify_ring_name(mol, ring) == 'isothiazole'

    def test_thiazole_isothiazole_distinct(self):
        """Thiazole and isothiazole must produce different names."""
        mol_tz, ring_tz = _get_ring('c1cscn1')
        mol_it, ring_it = _get_ring('c1ccsn1')
        assert _identify_ring_name(mol_tz, ring_tz) != _identify_ring_name(mol_it, ring_it)


# ============================================================================
# Parent/child seniority (identify_parent_and_child)
# ============================================================================

class TestParentChildSeniority:
    """Tests for IUPAC seniority-based parent/child selection."""

    def test_hetero_over_carbo(self):
        """Heterocyclic ring should be parent over carbocyclic (pyridine > benzene)."""
        mol, r0, r1 = _get_two_rings('c1ccc2ncccc2c1')  # quinoline
        parent, child, _, _ = identify_parent_and_child(mol, r0, r1)
        assert parent == 'pyridine', f"Expected pyridine as parent, got {parent}"
        assert child == 'benzene', f"Expected benzene as child, got {child}"

    def test_n_over_o(self):
        """N-heterocycle should be parent over O-heterocycle."""
        # Build a fused system with pyridine + furan if possible
        # Use c1cc2ccoc2nc1 -- might not be standard; test with simulated rings
        # Alternative: just test seniority values directly
        assert get_component_seniority('pyridine') < get_component_seniority('furan')

    def test_n_over_s(self):
        """N-heterocycle should be parent over S-heterocycle."""
        assert get_component_seniority('pyridine') < get_component_seniority('thiophene')

    def test_o_over_s(self):
        """O-heterocycle should be parent over S-heterocycle."""
        assert get_component_seniority('furan') < get_component_seniority('thiophene')

    def test_larger_ring_tiebreaker(self):
        """Among same heteroatom type, larger ring should be parent."""
        # pyridine (6, seniority 49) vs pyrrole (5, seniority 55)
        # pyridine is more senior, should be parent
        assert get_component_seniority('pyridine') < get_component_seniority('pyrrole')

    def test_benzofuran_parent_is_furan(self):
        """In benzofuran, furan (O-heterocycle) should be parent over benzene."""
        mol, r0, r1 = _get_two_rings('c1ccc2occc2c1')
        parent, child, _, _ = identify_parent_and_child(mol, r0, r1)
        assert parent == 'furan', f"Expected furan as parent, got {parent}"
        assert child == 'benzene', f"Expected benzene as child, got {child}"

    def test_benzothiophene_parent_is_thiophene(self):
        """In benzothiophene, thiophene (S-heterocycle) should be parent over benzene."""
        mol, r0, r1 = _get_two_rings('c1ccc2sccc2c1')
        parent, child, _, _ = identify_parent_and_child(mol, r0, r1)
        assert parent == 'thiophene', f"Expected thiophene as parent, got {parent}"
        assert child == 'benzene', f"Expected benzene as child, got {child}"


# ============================================================================
# Registry pattern matching
# ============================================================================

class TestPatternMatching:
    """Tests for get_component_by_pattern() lookup."""

    def test_benzene_pattern(self):
        """6-membered no-heteroatom -> benzene."""
        assert get_component_by_pattern(6, []) == 'benzene'

    def test_pyridine_pattern(self):
        """6-membered 1N -> pyridine."""
        assert get_component_by_pattern(6, ['N']) == 'pyridine'

    def test_furan_pattern(self):
        """5-membered 1O -> furan."""
        assert get_component_by_pattern(5, ['O']) == 'furan'

    def test_pyrimidine_pattern_gap1(self):
        """6-membered 2N gap=1 -> pyrimidine."""
        assert get_component_by_pattern(6, ['N', 'N'], hetero_gap=1) == 'pyrimidine'

    def test_pyrazine_pattern_gap2(self):
        """6-membered 2N gap=2 -> pyrazine."""
        assert get_component_by_pattern(6, ['N', 'N'], hetero_gap=2) == 'pyrazine'

    def test_pyridazine_pattern_gap0(self):
        """6-membered 2N gap=0 -> pyridazine."""
        assert get_component_by_pattern(6, ['N', 'N'], hetero_gap=0) == 'pyridazine'

    def test_imidazole_pattern_gap1(self):
        """5-membered 2N gap=1 -> imidazole."""
        assert get_component_by_pattern(5, ['N', 'N'], hetero_gap=1) == 'imidazole'

    def test_pyrazole_pattern_gap0(self):
        """5-membered 2N gap=0 -> pyrazole."""
        assert get_component_by_pattern(5, ['N', 'N'], hetero_gap=0) == 'pyrazole'

    def test_no_match_returns_none(self):
        """Unknown pattern should return None."""
        assert get_component_by_pattern(8, ['N', 'N', 'N']) is None
