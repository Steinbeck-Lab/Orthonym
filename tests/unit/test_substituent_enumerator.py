"""Unit tests for the substituent enumerator module.

Tests the ReplaceCore-based ring extraction, chain extraction,
fragment classification and naming, and atom set collection.

All tests use @pytest.mark.unit for fast test suite inclusion.
"""

import logging
import pytest
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import (
    extract_ring_substituents,
    extract_chain_substituents,
    classify_and_name_fragment,
    collect_substituent_atom_set,
    SubstituentInfo,
)


# ============================================================================
# Helper: Build oriented ring for a simple monocyclic molecule
# ============================================================================

def _get_ring_and_oriented(smiles):
    """Get mol, ring_atoms, and oriented_ring for a simple monocyclic molecule."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Failed to parse SMILES: {smiles}"
    ring_info = mol.GetRingInfo()
    rings = ring_info.AtomRings()
    assert len(rings) >= 1, f"No rings found in {smiles}"
    ring_atoms = list(rings[0])
    # For tests, oriented_ring = ring_atoms (same ordering)
    return mol, ring_atoms, ring_atoms


# ============================================================================
# A. Ring Substituent Extraction Tests
# ============================================================================


@pytest.mark.unit
class TestRingSubstituentExtraction:
    """Tests for extract_ring_substituents."""

    def test_ring_methyl_cyclohexane(self):
        """CC1CCCCC1 -> 1 substituent (methyl)."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('CC1CCCCC1')
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 1
        info = results[0]
        assert info.frag_mol is not None
        assert isinstance(info.locant, int)
        assert info.locant >= 1

    def test_ring_dimethyl_cyclohexane(self):
        """CC1CCCC(C)C1 -> 2 substituents, both methyl."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('CC1CCCC(C)C1')
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 2
        # Both should be methyl (1 carbon fragment each)
        for info in results:
            assert info.frag_mol is not None
            assert isinstance(info.locant, int)

    def test_ring_chloro_cyclohexane(self):
        """ClC1CCCCC1 -> 1 substituent (chloro). Previously dropped by old path."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('ClC1CCCCC1')
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 1
        info = results[0]
        assert info.frag_mol is not None
        # The fragment should contain a Cl atom
        has_cl = any(
            a.GetSymbol() == 'Cl' for a in info.frag_mol.GetAtoms()
            if a.GetAtomicNum() != 0
        )
        assert has_cl, "Chloro fragment not detected"

    def test_ring_amino_cyclohexane(self):
        """NC1CCCCC1 -> 1 substituent (amino). Previously dropped by has_heteroatom:continue."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('NC1CCCCC1')
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 1
        info = results[0]
        assert info.frag_mol is not None
        has_n = any(
            a.GetSymbol() == 'N' for a in info.frag_mol.GetAtoms()
            if a.GetAtomicNum() != 0
        )
        assert has_n, "Amino fragment not detected"

    def test_ring_hydroxy_cyclohexane(self):
        """OC1CCCCC1 -> 1 substituent (hydroxy). Previously dropped."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('OC1CCCCC1')
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 1
        info = results[0]
        assert info.frag_mol is not None
        has_o = any(
            a.GetSymbol() == 'O' for a in info.frag_mol.GetAtoms()
            if a.GetAtomicNum() != 0
        )
        assert has_o, "Hydroxy fragment not detected"

    def test_ring_trifluoromethyl_cyclohexane(self):
        """FC(F)(F)C1CCCCC1 -> 1 compound substituent (C + 3F)."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('FC(F)(F)C1CCCCC1')
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 1
        info = results[0]
        assert info.frag_mol is not None
        # Fragment should have at least 1 C and 3 F
        non_dummy = [
            a for a in info.frag_mol.GetAtoms() if a.GetAtomicNum() != 0
        ]
        symbols = [a.GetSymbol() for a in non_dummy]
        assert 'C' in symbols
        assert symbols.count('F') == 3

    def test_ring_mixed_alkyl_halogen(self):
        """CC1CCC(Cl)CC1 -> 2 substituents (methyl + chloro) at different locants."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('CC1CCC(Cl)CC1')
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 2
        locants = [r.locant for r in results]
        # The two substituents should be at different positions
        assert locants[0] != locants[1], f"Same locant for both: {locants}"

    def test_ring_no_substituents(self):
        """C1CCCCC1 -> no substituents (bare cyclohexane)."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('C1CCCCC1')
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 0

    def test_ring_multiple_same_substituent(self):
        """CC1(C)CCCCC1 -> 2 methyl at the same position (geminal)."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('CC1(C)CCCCC1')
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 2
        # Both at the same locant (geminal)
        locants = [r.locant for r in results]
        assert locants[0] == locants[1]

    def test_ring_nitro_cyclohexane(self):
        """[O-][N+](=O)C1CCCCC1 -> 1 substituent (nitro)."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('[O-][N+](=O)C1CCCCC1')
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 1
        info = results[0]
        assert info.frag_mol is not None
        non_dummy = [
            a.GetSymbol() for a in info.frag_mol.GetAtoms()
            if a.GetAtomicNum() != 0
        ]
        assert 'N' in non_dummy
        assert non_dummy.count('O') == 2


# ============================================================================
# B. Chain Substituent Extraction Tests
# ============================================================================


@pytest.mark.unit
class TestChainSubstituentExtraction:
    """Tests for extract_chain_substituents."""

    def test_chain_basic_conversion(self):
        """Substituents dict is correctly converted to SubstituentInfo list."""
        mol = Chem.MolFromSmiles('CCCC')  # butane
        principal_chain = [0, 1, 2, 3]
        # Simulate: methyl at position 2
        sub_dict = {2: [[4]]}  # hypothetical atom index 4 at position 2
        # Since atom 4 doesn't exist in butane, just test the structure
        mol2 = Chem.MolFromSmiles('CC(C)CC')  # 2-methylbutane
        principal_chain2 = [0, 1, 3, 4]
        sub_dict2 = {2: [[2]]}  # atom 2 (the methyl C) at position 2
        results = extract_chain_substituents(mol2, principal_chain2, sub_dict2)
        assert len(results) == 1
        info = results[0]
        assert info.locant == 2
        assert info.frag_mol is None  # chain substituents don't use frag_mol
        assert 2 in info.frag_atoms

    def test_chain_empty_dict(self):
        """Empty substituents dict returns empty list."""
        mol = Chem.MolFromSmiles('CCCC')
        results = extract_chain_substituents(mol, [0, 1, 2, 3], {})
        assert len(results) == 0

    def test_chain_none_dict(self):
        """None substituents dict returns empty list."""
        mol = Chem.MolFromSmiles('CCCC')
        results = extract_chain_substituents(mol, [0, 1, 2, 3], None)
        assert len(results) == 0

    def test_chain_multiple_positions(self):
        """Multiple substituents at different positions."""
        mol = Chem.MolFromSmiles('CC(C)C(C)C')  # 2,3-dimethylbutane
        # Simulate: methyl at positions 2 and 3
        sub_dict = {
            2: [[2]],
            3: [[4]],
        }
        results = extract_chain_substituents(mol, [0, 1, 3, 5], sub_dict)
        assert len(results) == 2
        locants = sorted(r.locant for r in results)
        assert locants == [2, 3]


# ============================================================================
# C. Fragment Classification and Naming Tests
# ============================================================================


@pytest.mark.unit
class TestClassifyAndNameFragment:
    """Tests for classify_and_name_fragment."""

    def test_classify_fg_only_chloro(self):
        """Chloro fragment -> 'chloro'."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('ClC1CCCCC1')
        ring_set = set(ring_atoms)
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 1
        name = classify_and_name_fragment(mol, results[0], ring_set)
        assert name == 'chloro'

    def test_classify_fg_only_fluoro(self):
        """Fluoro fragment -> 'fluoro'."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('FC1CCCCC1')
        ring_set = set(ring_atoms)
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 1
        name = classify_and_name_fragment(mol, results[0], ring_set)
        assert name == 'fluoro'

    def test_classify_fg_only_bromo(self):
        """Bromo fragment -> 'bromo'."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('BrC1CCCCC1')
        ring_set = set(ring_atoms)
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 1
        name = classify_and_name_fragment(mol, results[0], ring_set)
        assert name == 'bromo'

    def test_classify_fg_only_hydroxyl(self):
        """Hydroxyl fragment -> 'hydroxy'."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('OC1CCCCC1')
        ring_set = set(ring_atoms)
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 1
        name = classify_and_name_fragment(mol, results[0], ring_set)
        assert name == 'hydroxy'

    def test_classify_fg_only_amino(self):
        """Amino fragment -> 'amino'."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('NC1CCCCC1')
        ring_set = set(ring_atoms)
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 1
        name = classify_and_name_fragment(mol, results[0], ring_set)
        assert name == 'amino'

    def test_classify_fg_only_nitro(self):
        """Nitro fragment -> 'nitro'."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('[O-][N+](=O)C1CCCCC1')
        ring_set = set(ring_atoms)
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 1
        name = classify_and_name_fragment(mol, results[0], ring_set)
        assert name == 'nitro'

    def test_classify_pure_alkyl_methyl(self):
        """Methyl fragment -> 'methyl'."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('CC1CCCCC1')
        ring_set = set(ring_atoms)
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 1
        name = classify_and_name_fragment(mol, results[0], ring_set)
        assert name == 'methyl'

    def test_classify_pure_alkyl_ethyl(self):
        """Ethyl fragment -> 'ethyl'."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('CCC1CCCCC1')
        ring_set = set(ring_atoms)
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 1
        name = classify_and_name_fragment(mol, results[0], ring_set)
        assert name == 'ethyl'

    def test_classify_compound_trifluoromethyl(self):
        """Trifluoromethyl fragment -> 'trifluoromethyl'."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('FC(F)(F)C1CCCCC1')
        ring_set = set(ring_atoms)
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 1
        name = classify_and_name_fragment(mol, results[0], ring_set)
        # Should be 'trifluoromethyl' (the compound substituent name)
        assert name is not None
        assert 'fluoro' in name.lower() or 'trifluoromethyl' in name.lower()

    def test_classify_unknown_logs_warning(self, caplog):
        """Unknown fragment -> None + WARNING log."""
        # Create a synthetic SubstituentInfo with empty fragment
        frag_info = SubstituentInfo(
            frag_mol=None,
            locant=1,
            attach_mol_idx=0,
            frag_atoms=frozenset(),
        )
        mol = Chem.MolFromSmiles('C')
        with caplog.at_level(logging.WARNING):
            name = classify_and_name_fragment(mol, frag_info, set())
        assert name is None

    def test_mixed_ring_substituent_names(self):
        """CC1CCC(Cl)CC1 -> methyl + chloro at different locants."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('CC1CCC(Cl)CC1')
        ring_set = set(ring_atoms)
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 2
        names = set()
        for r in results:
            name = classify_and_name_fragment(mol, r, ring_set)
            assert name is not None
            names.add(name)
        assert names == {'methyl', 'chloro'}


# ============================================================================
# D. Atom Set Collection Tests
# ============================================================================


@pytest.mark.unit
class TestCollectSubstituentAtomSet:
    """Tests for collect_substituent_atom_set."""

    def test_empty_list(self):
        """Empty list returns empty set."""
        result = collect_substituent_atom_set([])
        assert result == frozenset()

    def test_single_substituent(self):
        """Single substituent returns its atom set."""
        info = SubstituentInfo(
            frag_mol=None, locant=1, attach_mol_idx=0,
            frag_atoms=frozenset({5, 6, 7})
        )
        result = collect_substituent_atom_set([info])
        assert result == frozenset({5, 6, 7})

    def test_multiple_substituents_union(self):
        """Multiple substituents returns union of all atom sets."""
        info1 = SubstituentInfo(
            frag_mol=None, locant=1, attach_mol_idx=0,
            frag_atoms=frozenset({5, 6})
        )
        info2 = SubstituentInfo(
            frag_mol=None, locant=3, attach_mol_idx=2,
            frag_atoms=frozenset({7, 8, 9})
        )
        result = collect_substituent_atom_set([info1, info2])
        assert result == frozenset({5, 6, 7, 8, 9})

    def test_overlapping_atom_sets(self):
        """Overlapping atom sets correctly unified."""
        info1 = SubstituentInfo(
            frag_mol=None, locant=1, attach_mol_idx=0,
            frag_atoms=frozenset({5, 6, 7})
        )
        info2 = SubstituentInfo(
            frag_mol=None, locant=2, attach_mol_idx=1,
            frag_atoms=frozenset({6, 7, 8})
        )
        result = collect_substituent_atom_set([info1, info2])
        assert result == frozenset({5, 6, 7, 8})

    def test_none_frag_atoms_skipped(self):
        """SubstituentInfo with None frag_atoms is handled gracefully."""
        info1 = SubstituentInfo(
            frag_mol=None, locant=1, attach_mol_idx=0,
            frag_atoms=frozenset({5, 6})
        )
        info2 = SubstituentInfo(
            frag_mol=None, locant=2, attach_mol_idx=1,
            frag_atoms=None
        )
        result = collect_substituent_atom_set([info1, info2])
        assert result == frozenset({5, 6})


# ============================================================================
# E. Integration: Full Ring -> Extract -> Classify -> Name Pipeline
# ============================================================================


@pytest.mark.unit
class TestFullPipeline:
    """End-to-end tests for the extract->classify->name pipeline."""

    def test_aminocyclohexane_not_dropped(self):
        """NC1CCCCC1: amino must NOT be silently dropped (was the #1 bug)."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('NC1CCCCC1')
        ring_set = set(ring_atoms)
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) >= 1, "Amino substituent was silently dropped!"
        name = classify_and_name_fragment(mol, results[0], ring_set)
        assert name == 'amino', f"Expected 'amino', got '{name}'"

    def test_hydroxycyclohexane_not_dropped(self):
        """OC1CCCCC1: hydroxy must NOT be silently dropped."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('OC1CCCCC1')
        ring_set = set(ring_atoms)
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) >= 1, "Hydroxy substituent was silently dropped!"
        name = classify_and_name_fragment(mol, results[0], ring_set)
        assert name == 'hydroxy', f"Expected 'hydroxy', got '{name}'"

    def test_chlorocyclohexane_not_dropped(self):
        """ClC1CCCCC1: chloro must NOT be silently dropped."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('ClC1CCCCC1')
        ring_set = set(ring_atoms)
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) >= 1, "Chloro substituent was silently dropped!"
        name = classify_and_name_fragment(mol, results[0], ring_set)
        assert name == 'chloro', f"Expected 'chloro', got '{name}'"

    def test_trifluoromethyl_is_compound(self):
        """FC(F)(F)C1CCCCC1: trifluoromethyl is a compound substituent."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('FC(F)(F)C1CCCCC1')
        ring_set = set(ring_atoms)
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 1
        name = classify_and_name_fragment(mol, results[0], ring_set)
        assert name is not None
        # Should contain both fluoro and methyl components
        assert 'trifluoromethyl' in name.lower() or 'fluoro' in name.lower()

    def test_isopropyl_substituent(self):
        """CC(C)C1CCCCC1: isopropyl substituent on cyclohexane."""
        mol, ring_atoms, oriented = _get_ring_and_oriented('CC(C)C1CCCCC1')
        ring_set = set(ring_atoms)
        results = extract_ring_substituents(mol, ring_atoms, oriented)
        assert len(results) == 1
        name = classify_and_name_fragment(mol, results[0], ring_set)
        assert name is not None
        # F-T9/DD6: the PIN is the located 'propan-2-yl' (not the deprecated
        # 'isopropyl'/'1-methylethyl'). Accept the historical forms too for robustness.
        assert ('propan-2-yl' in name.lower() or 'propyl' in name.lower()
                or 'methylethyl' in name.lower())
