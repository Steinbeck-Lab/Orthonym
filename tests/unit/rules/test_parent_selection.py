"""
Unit tests for parent selection logic (IUPAC P-44.1).

Tests the select_parent() function which determines whether the ring or
chain should be the parent structure based on principal group location.
"""

import pytest
from rdkit import Chem

from orthonym.rules.parent_selection import (
    ParentSelectionResult,
    select_parent,
    is_principal_group_on_chain,
    is_principal_group_on_ring,
)
from orthonym.perception.rings import get_ring_systems
from orthonym.perception.functional_groups import detect_functional_groups
from orthonym.rules.seniority import get_principal_group


class TestIsPrincipalGroupOnChain:
    """Tests for is_principal_group_on_chain()."""

    def test_phenylbutanoic_acid_pg_on_chain(self):
        """Carboxylic acid on butyl chain attached to benzene."""
        mol = Chem.MolFromSmiles('c1ccc(CCCC(=O)O)cc1')
        # Chain atoms: 4 carbons of butyl chain (indices 6-9 in SMILES)
        # Find the chain atoms excluding ring
        ring_atoms = set(range(6))  # benzene atoms 0-5
        # Butyl chain carbons are 6, 7, 8, 9
        chain_atoms = [6, 7, 8, 9]

        fg = detect_functional_groups(mol)
        _, pg_atoms = get_principal_group(mol, fg)

        result = is_principal_group_on_chain(mol, chain_atoms, pg_atoms)
        assert result is True, "COOH should be on the butyl chain"

    def test_benzoic_acid_pg_not_on_chain(self):
        """Carboxylic acid directly on benzene has no chain."""
        mol = Chem.MolFromSmiles('c1ccc(C(=O)O)cc1')
        # There's no meaningful chain - only COOH attached to benzene
        chain_atoms = []  # Empty chain

        fg = detect_functional_groups(mol)
        _, pg_atoms = get_principal_group(mol, fg)

        result = is_principal_group_on_chain(mol, chain_atoms, pg_atoms)
        assert result is False, "COOH directly on ring has no chain"

    def test_phenylacetic_acid_pg_on_short_chain(self):
        """Carboxylic acid on 2-carbon chain attached to benzene."""
        mol = Chem.MolFromSmiles('c1ccc(CC(=O)O)cc1')
        ring_atoms = set(range(6))  # benzene atoms 0-5
        # Find chain atoms (CH2-COOH)
        chain_atoms = [6, 7]  # CH2 and C(=O)

        fg = detect_functional_groups(mol)
        _, pg_atoms = get_principal_group(mol, fg)

        result = is_principal_group_on_chain(mol, chain_atoms, pg_atoms)
        assert result is True, "COOH should be on the short chain"

    def test_alcohol_on_chain(self):
        """Alcohol group on chain attached to benzene."""
        mol = Chem.MolFromSmiles('c1ccc(CCCO)cc1')
        chain_atoms = [6, 7, 8]  # propyl chain

        fg = detect_functional_groups(mol)
        _, pg_atoms = get_principal_group(mol, fg)

        result = is_principal_group_on_chain(mol, chain_atoms, pg_atoms)
        assert result is True, "OH should be on the propyl chain"


class TestIsPrincipalGroupOnRing:
    """Tests for is_principal_group_on_ring()."""

    def test_benzoic_acid_pg_on_ring(self):
        """Carboxylic acid where carbonyl C is bonded to ring."""
        mol = Chem.MolFromSmiles('c1ccc(C(=O)O)cc1')
        ring_atoms = set(range(6))  # benzene atoms 0-5

        fg = detect_functional_groups(mol)
        _, pg_atoms = get_principal_group(mol, fg)

        result = is_principal_group_on_ring(mol, ring_atoms, pg_atoms)
        assert result is True, "COOH carbonyl C is bonded to benzene"

    def test_phenylbutanoic_acid_pg_not_on_ring(self):
        """Carboxylic acid on chain - not directly on ring."""
        mol = Chem.MolFromSmiles('c1ccc(CCCC(=O)O)cc1')
        ring_atoms = set(range(6))  # benzene atoms 0-5

        fg = detect_functional_groups(mol)
        _, pg_atoms = get_principal_group(mol, fg)

        result = is_principal_group_on_ring(mol, ring_atoms, pg_atoms)
        assert result is False, "COOH is on chain, not directly on ring"

    def test_phenol_pg_on_ring(self):
        """Phenol - hydroxyl attached to benzene carbon."""
        mol = Chem.MolFromSmiles('c1ccc(O)cc1')
        ring_atoms = set(range(6))  # benzene atoms 0-5

        fg = detect_functional_groups(mol)
        _, pg_atoms = get_principal_group(mol, fg)

        result = is_principal_group_on_ring(mol, ring_atoms, pg_atoms)
        assert result is True, "OH is on benzene carbon"


class TestSelectParent:
    """Tests for select_parent() main function."""

    def test_phenylbutanoic_acid_chain_is_parent(self):
        """c1ccc(CCCC(=O)O)cc1 -> chain is parent (4-phenylbutanoic acid)."""
        smiles = 'c1ccc(CCCC(=O)O)cc1'
        mol = Chem.MolFromSmiles(smiles)

        # Get ring systems
        ring_systems = get_ring_systems(mol)
        ring_atoms = set()
        for ring in ring_systems:
            ring_atoms.update(ring)

        # Get functional groups and principal group
        fg = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fg)

        # Find chain atoms (excluding ring)
        # The butyl chain is atoms 6, 7, 8, 9 (C-C-C-C(=O)O)
        chain_atoms = [6, 7, 8, 9]

        result = select_parent(
            mol=mol,
            ring_systems=ring_systems,
            principal_chain=chain_atoms,
            principal_group=pg_name,
            principal_group_atoms=pg_atoms
        )

        assert result.parent_type == 'chain', f"Expected 'chain' but got '{result.parent_type}'"
        assert len(result.parent_atoms) == 4, "Chain should have 4 carbons"
        assert len(result.substituent_rings) == 1, "Benzene should be a substituent"

    def test_benzoic_acid_ring_is_parent(self):
        """c1ccc(C(=O)O)cc1 -> ring is parent (benzoic acid)."""
        smiles = 'c1ccc(C(=O)O)cc1'
        mol = Chem.MolFromSmiles(smiles)

        ring_systems = get_ring_systems(mol)
        fg = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fg)

        # Chain is just the COOH group - only 1 carbon
        chain_atoms = [6]  # C of COOH

        result = select_parent(
            mol=mol,
            ring_systems=ring_systems,
            principal_chain=chain_atoms,
            principal_group=pg_name,
            principal_group_atoms=pg_atoms
        )

        assert result.parent_type == 'ring', f"Expected 'ring' but got '{result.parent_type}'"

    def test_cyclohexylbutanoic_acid_chain_is_parent(self):
        """C1CCCCC1CCCC(=O)O -> chain is parent (4-cyclohexylbutanoic acid)."""
        smiles = 'C1CCCCC1CCCC(=O)O'
        mol = Chem.MolFromSmiles(smiles)

        ring_systems = get_ring_systems(mol)
        ring_atoms = set()
        for ring in ring_systems:
            ring_atoms.update(ring)

        fg = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fg)

        # Find chain atoms (atoms 6-9 are the butyl chain)
        chain_atoms = [6, 7, 8, 9]

        result = select_parent(
            mol=mol,
            ring_systems=ring_systems,
            principal_chain=chain_atoms,
            principal_group=pg_name,
            principal_group_atoms=pg_atoms
        )

        assert result.parent_type == 'chain', "Chain should be parent for cyclohexylbutanoic acid"

    def test_empty_chain_ring_is_parent(self):
        """When no chain is provided, ring should be parent."""
        smiles = 'c1ccc(C(=O)O)cc1'
        mol = Chem.MolFromSmiles(smiles)

        ring_systems = get_ring_systems(mol)
        fg = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fg)

        result = select_parent(
            mol=mol,
            ring_systems=ring_systems,
            principal_chain=[],
            principal_group=pg_name,
            principal_group_atoms=pg_atoms
        )

        assert result.parent_type == 'ring', "Ring should be parent when no chain"

    def test_phenylpropanol_chain_is_parent(self):
        """c1ccc(CCCO)cc1 -> chain is parent (3-phenylpropan-1-ol)."""
        smiles = 'c1ccc(CCCO)cc1'
        mol = Chem.MolFromSmiles(smiles)

        ring_systems = get_ring_systems(mol)
        ring_atoms = set()
        for ring in ring_systems:
            ring_atoms.update(ring)

        fg = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fg)

        # Propyl chain is atoms 6, 7, 8
        chain_atoms = [6, 7, 8]

        result = select_parent(
            mol=mol,
            ring_systems=ring_systems,
            principal_chain=chain_atoms,
            principal_group=pg_name,
            principal_group_atoms=pg_atoms
        )

        assert result.parent_type == 'chain', "Chain should be parent for phenylpropanol"


class TestResultDataclass:
    """Tests for ParentSelectionResult dataclass."""

    def test_result_has_all_fields(self):
        """ParentSelectionResult should have required fields."""
        result = ParentSelectionResult(
            parent_type='chain',
            parent_atoms=[0, 1, 2, 3],
            substituent_rings=[(4, 5, 6, 7, 8, 9)],
            reasoning="Test reasoning"
        )

        assert result.parent_type == 'chain'
        assert result.parent_atoms == [0, 1, 2, 3]
        assert result.substituent_rings == [(4, 5, 6, 7, 8, 9)]
        assert result.reasoning == "Test reasoning"


class TestEdgeCases:
    """Edge case tests for parent selection."""

    def test_no_functional_group(self):
        """Molecule with no functional group."""
        smiles = 'c1ccc(CCCC)cc1'  # butylbenzene
        mol = Chem.MolFromSmiles(smiles)

        ring_systems = get_ring_systems(mol)
        fg = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fg)

        chain_atoms = [6, 7, 8, 9]  # butyl chain

        result = select_parent(
            mol=mol,
            ring_systems=ring_systems,
            principal_chain=chain_atoms,
            principal_group=pg_name,
            principal_group_atoms=pg_atoms
        )

        # For hydrocarbons without FG, ring has seniority (IUPAC P-44.1.2.2)
        assert result.parent_type == 'ring', "Ring has seniority for hydrocarbons"

    def test_single_carbon_chain(self):
        """Chain with only 1 carbon should favor ring."""
        smiles = 'c1ccc(C(=O)O)cc1'  # benzoic acid
        mol = Chem.MolFromSmiles(smiles)

        ring_systems = get_ring_systems(mol)
        fg = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fg)

        result = select_parent(
            mol=mol,
            ring_systems=ring_systems,
            principal_chain=[6],  # Just the COOH carbon
            principal_group=pg_name,
            principal_group_atoms=pg_atoms
        )

        assert result.parent_type == 'ring', "Single carbon chain -> ring is parent"

    def test_two_carbon_chain_with_fg(self):
        """2-carbon chain with FG should be chain parent."""
        smiles = 'c1ccc(CC(=O)O)cc1'  # phenylacetic acid
        mol = Chem.MolFromSmiles(smiles)

        ring_systems = get_ring_systems(mol)
        fg = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fg)

        chain_atoms = [6, 7]  # CH2-COOH

        result = select_parent(
            mol=mol,
            ring_systems=ring_systems,
            principal_chain=chain_atoms,
            principal_group=pg_name,
            principal_group_atoms=pg_atoms
        )

        assert result.parent_type == 'chain', "2-carbon chain with FG -> chain is parent"
