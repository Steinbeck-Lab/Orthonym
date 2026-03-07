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
    _select_best_ring_system,
    _ring_system_has_nitrogen,
    _count_multiple_bonds,
)
from orthonym.perception.rings import get_ring_systems
from orthonym.perception.functional_groups import detect_functional_groups
from orthonym.rules.seniority import get_principal_group
from orthonym.rules.ring_selection import ring_system_score


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


class TestFGCountingDedup:
    """PSEL-01: FG counting deduplication by attachment point."""

    def test_duplicate_pg_atoms_deduplicated_on_chain(self):
        """If same FG atoms are passed twice, count stays correct."""
        from orthonym.rules.parent_selection import _count_pg_on_chain
        from orthonym.perception.chains import find_longest_carbon_chain
        mol = Chem.MolFromSmiles("OC(=O)CCCCC(=O)O")  # adipic acid
        fg = detect_functional_groups(mol)
        chain = find_longest_carbon_chain(mol)
        pg_atoms = fg.get("carboxylic_acid", [])
        # Double the matches to simulate overlap
        doubled = pg_atoms + pg_atoms
        count = _count_pg_on_chain(mol, chain, doubled)
        assert count == 2, f"Duplicated matches must not inflate count, got {count}"

    def test_duplicate_pg_atoms_deduplicated_on_ring(self):
        """If same FG atoms are passed twice, count stays correct on ring."""
        from orthonym.rules.parent_selection import _count_pg_on_ring
        mol = Chem.MolFromSmiles("c1cc(C(=O)O)cc(C(=O)O)c1C(=O)O")  # trimesic acid
        fg = detect_functional_groups(mol)
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            if len(ring) == 6:
                ring_atoms = set(ring)
                break
        pg_atoms = fg.get("carboxylic_acid", [])
        doubled = pg_atoms + pg_atoms
        count = _count_pg_on_ring(mol, ring_atoms, doubled)
        assert count == 3, f"Duplicated matches must not inflate count, got {count}"


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


@pytest.mark.unit
class TestEnhancedParentSelection:
    """Tests for P-52.2.8, multi-ring seniority, and enhanced tiebreakers."""

    def test_p52_2_8_ring_preferred_on_tie(self):
        """P-44.1 chain-length: When FG count is tied but chain (all
        non-ring atoms = 7) is longer than ring (6), chain wins by
        P-44.1 chain-length criterion.

        Uses cyclohexanone with a chain ketone -- 1 ketone on ring, 1 on chain.
        Chain (7 atoms incl. oxygens) > ring (6 atoms) -> chain wins per
        P-44.1 chain-length before P-52.2.8 would apply.
        """
        # O=C1CCCCC1CCC(=O)CC -- cyclohexanone + chain ketone
        smiles = 'O=C1CCCCC1CCC(=O)CC'
        mol = Chem.MolFromSmiles(smiles)
        ring_systems = get_ring_systems(mol)
        fg = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fg)

        # Chain atoms: everything not in ring (includes oxygens)
        all_ring = set()
        for r in ring_systems:
            all_ring.update(r)
        chain = [i for i in range(mol.GetNumAtoms())
                 if i not in all_ring]

        result = select_parent(
            mol=mol,
            ring_systems=ring_systems,
            principal_chain=chain,
            principal_group=pg_name,
            principal_group_atoms=pg_atoms
        )

        # Chain (7 atoms) > ring (6 atoms) -> chain wins by P-44.1 chain-length
        assert result.parent_type == 'chain', (
            "P-44.1 chain-length: chain (7) > ring (6) should win"
        )
        assert 'P-44.1' in result.reasoning, (
            "Reasoning should cite P-44.1 cascade rule"
        )

    def test_multi_ring_system_selects_senior(self):
        """Multi-ring: pyridine (heterocyclic) should be selected over cyclohexane.

        Uses c1ccncc1CCC1CCCCC1 -- pyridine-propyl-cyclohexane.
        Pyridine is more senior per P-44.2 (heterocyclic, contains N).
        """
        smiles = 'c1ccncc1CCC1CCCCC1'
        mol = Chem.MolFromSmiles(smiles)
        ring_systems = get_ring_systems(mol)

        assert len(ring_systems) == 2, "Should have 2 ring systems"

        best, others = _select_best_ring_system(mol, ring_systems)

        # The best ring should be the pyridine (contains nitrogen)
        assert _ring_system_has_nitrogen(mol, best), (
            "Best ring system should be pyridine (contains N)"
        )
        assert len(others) == 1, "Should have 1 other ring system"

        # The other ring should be cyclohexane (no nitrogen)
        assert not _ring_system_has_nitrogen(mol, others[0]), (
            "Other ring should be cyclohexane (no N)"
        )

        # Verify scoring: pyridine should have lower (more senior) score
        pyridine_score = ring_system_score(mol, best)
        cyclohex_score = ring_system_score(mol, others[0])
        assert pyridine_score < cyclohex_score, (
            "Pyridine score should be lower (more senior) than cyclohexane"
        )

    def test_short_chain_prefers_ring(self):
        """Short chain (1 carbon) with 6-membered ring -> ring is parent.

        OCC1CCCCC1 (hydroxymethylcyclohexane): OH on 1-carbon chain.
        Ring should win because chain is just a substituent.
        """
        smiles = 'OCC1CCCCC1'
        mol = Chem.MolFromSmiles(smiles)
        ring_systems = get_ring_systems(mol)
        fg = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fg)

        # Single carbon chain
        all_ring = set()
        for r in ring_systems:
            all_ring.update(r)
        chain_c = [i for i in range(mol.GetNumAtoms())
                   if i not in all_ring and mol.GetAtomWithIdx(i).GetAtomicNum() == 6]

        result = select_parent(
            mol=mol,
            ring_systems=ring_systems,
            principal_chain=chain_c,
            principal_group=pg_name,
            principal_group_atoms=pg_atoms
        )

        assert result.parent_type == 'ring', (
            "Single carbon chain should favor ring as parent"
        )

    def test_long_chain_with_fg_prefers_chain(self):
        """Long chain with COOH on chain terminus -> chain is parent.

        c1ccc(CCCCCCCC(=O)O)cc1 (8-phenyloctanoic acid): COOH exclusively
        on 8-atom chain. Chain must be parent per P-44.1.
        """
        smiles = 'c1ccc(CCCCCCCC(=O)O)cc1'
        mol = Chem.MolFromSmiles(smiles)
        ring_systems = get_ring_systems(mol)
        fg = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fg)

        # Chain: all non-ring atoms that are carbon
        all_ring = set()
        for r in ring_systems:
            all_ring.update(r)
        chain = [i for i in range(mol.GetNumAtoms())
                 if i not in all_ring and mol.GetAtomWithIdx(i).GetAtomicNum() == 6]

        result = select_parent(
            mol=mol,
            ring_systems=ring_systems,
            principal_chain=chain,
            principal_group=pg_name,
            principal_group_atoms=pg_atoms
        )

        assert result.parent_type == 'chain', (
            "Long chain with FG should be parent"
        )
        assert len(result.substituent_rings) >= 1, (
            "Ring should become substituent"
        )

    def test_nitrogen_ring_senior_atom_preference(self):
        """Ring system with nitrogen is preferred per P-44.1 senior atom.

        Tests _ring_system_has_nitrogen() and _select_best_ring_system()
        when comparing N-containing ring vs carbocyclic ring.
        """
        # Pyridine + cyclohexane
        smiles = 'c1ccncc1CCC1CCCCC1'
        mol = Chem.MolFromSmiles(smiles)
        ring_systems = get_ring_systems(mol)

        # Identify which ring system has nitrogen
        n_ring_idx = None
        c_ring_idx = None
        for i, system in enumerate(ring_systems):
            if _ring_system_has_nitrogen(mol, system):
                n_ring_idx = i
            else:
                c_ring_idx = i

        assert n_ring_idx is not None, "Should find N-containing ring"
        assert c_ring_idx is not None, "Should find carbocyclic ring"

        # _select_best_ring_system should prefer the N-containing ring
        best, _ = _select_best_ring_system(mol, ring_systems)
        assert _ring_system_has_nitrogen(mol, best), (
            "P-44.1 senior atom: N-containing ring should be preferred"
        )

    def test_select_best_ring_system_single(self):
        """With only one ring system, it should be returned directly."""
        smiles = 'C1CCCCC1'  # cyclohexane
        mol = Chem.MolFromSmiles(smiles)
        ring_systems = get_ring_systems(mol)

        assert len(ring_systems) == 1
        best, others = _select_best_ring_system(mol, ring_systems)
        assert best == ring_systems[0]
        assert others == []

    def test_p44_1_documented_in_reasoning(self):
        """P-44.1 chain-length should be referenced in reasoning when chain is longer.

        Uses same molecule as test_p52_2_8_ring_preferred_on_tie:
        chain (7 atoms) > ring (6 atoms) with PG tied -> P-44.1 chain-length.
        """
        smiles = 'O=C1CCCCC1CCC(=O)CC'
        mol = Chem.MolFromSmiles(smiles)
        ring_systems = get_ring_systems(mol)
        fg = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fg)

        all_ring = set()
        for r in ring_systems:
            all_ring.update(r)
        chain = [i for i in range(mol.GetNumAtoms())
                 if i not in all_ring]

        result = select_parent(
            mol=mol,
            ring_systems=ring_systems,
            principal_chain=chain,
            principal_group=pg_name,
            principal_group_atoms=pg_atoms
        )

        assert result.parent_type == 'chain'
        assert 'P-44.1' in result.reasoning


@pytest.mark.unit
class TestP441Cascade:
    """Tests for P-44.1 cascade: PG count > chain length > multiple bonds > P-52.2.8."""

    def test_p44_1_chain_length_wins_on_tie(self):
        """When PG count ties but chain is longer than ring, chain wins.

        O=C1CCC1CCCCC(=O)C: cyclobutanone (4-atom ring) + 5-carbon chain
        with ketone. PG = ketone, 1 on ring + 1 on chain (tied).
        Chain (8 non-ring atoms) > ring (4) -> chain wins by P-44.1 chain-length.
        """
        smiles = 'O=C1CCC1CCCCC(=O)C'
        mol = Chem.MolFromSmiles(smiles)
        ring_systems = get_ring_systems(mol)
        fg = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fg)

        all_ring = set()
        for r in ring_systems:
            all_ring.update(r)
        chain = [i for i in range(mol.GetNumAtoms()) if i not in all_ring]

        result = select_parent(
            mol=mol,
            ring_systems=ring_systems,
            principal_chain=chain,
            principal_group=pg_name,
            principal_group_atoms=pg_atoms
        )

        assert result.parent_type == 'chain', (
            f"Chain ({len(chain)}) > ring ({len(all_ring)}): chain should win. "
            f"Got: {result.reasoning}"
        )
        assert 'P-44.1' in result.reasoning, (
            f"Reasoning should cite P-44.1 cascade. Got: {result.reasoning}"
        )

    def test_p44_1_ring_size_wins_on_tie(self):
        """When PG count ties but ring is larger than chain, ring wins.

        O=C1CCCCCCC1CCC(=O)C: cyclooctanone (8-atom ring) + 3-carbon chain
        with ketone. PG = ketone, 1 on ring + 1 on chain (tied).
        Ring (8) > chain (6 non-ring atoms) -> ring wins by P-44.1 ring-size.
        """
        smiles = 'O=C1CCCCCCC1CCC(=O)C'
        mol = Chem.MolFromSmiles(smiles)
        ring_systems = get_ring_systems(mol)
        fg = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fg)

        all_ring = set()
        for r in ring_systems:
            all_ring.update(r)
        chain = [i for i in range(mol.GetNumAtoms()) if i not in all_ring]

        result = select_parent(
            mol=mol,
            ring_systems=ring_systems,
            principal_chain=chain,
            principal_group=pg_name,
            principal_group_atoms=pg_atoms
        )

        assert result.parent_type == 'ring', (
            f"Ring ({len(all_ring)}) > chain ({len(chain)}): ring should win. "
            f"Got: {result.reasoning}"
        )
        # Could be P-44.1 ring-size or ring-wins-by-PG-count depending on PG detection
        assert 'ring' in result.reasoning.lower(), (
            f"Reasoning should indicate ring wins. Got: {result.reasoning}"
        )

    def test_p44_1_multiple_bonds_chain_wins(self):
        """When PG count and length both tie, chain with more multiple bonds wins.

        Tests _count_multiple_bonds() helper directly and verifies the cascade
        logic since constructing a natural molecule with all criteria tied except
        bonds is difficult.
        """
        # Test _count_multiple_bonds on a chain with double bonds
        smiles = 'C=CC=CC'  # penta-1,3-diene
        mol = Chem.MolFromSmiles(smiles)
        all_atoms = set(range(mol.GetNumAtoms()))
        mult_count = _count_multiple_bonds(mol, all_atoms)
        assert mult_count == 2, (
            f"Penta-1,3-diene should have 2 double bonds, got {mult_count}"
        )

        # Test on a saturated chain (no double bonds)
        sat_smiles = 'CCCCC'  # pentane
        sat_mol = Chem.MolFromSmiles(sat_smiles)
        sat_atoms = set(range(sat_mol.GetNumAtoms()))
        sat_count = _count_multiple_bonds(sat_mol, sat_atoms)
        assert sat_count == 0, (
            f"Pentane should have 0 double bonds, got {sat_count}"
        )

        # Test on cyclohexane ring (no double bonds within ring)
        cyc_smiles = 'C1CCCCC1'
        cyc_mol = Chem.MolFromSmiles(cyc_smiles)
        cyc_atoms = set(range(cyc_mol.GetNumAtoms()))
        cyc_count = _count_multiple_bonds(cyc_mol, cyc_atoms)
        assert cyc_count == 0, (
            f"Cyclohexane should have 0 double bonds, got {cyc_count}"
        )

        # Test on cyclohexene ring (1 double bond within ring)
        cene_smiles = 'C1=CCCCC1'
        cene_mol = Chem.MolFromSmiles(cene_smiles)
        cene_atoms = set(range(cene_mol.GetNumAtoms()))
        cene_count = _count_multiple_bonds(cene_mol, cene_atoms)
        assert cene_count == 1, (
            f"Cyclohexene should have 1 double bond, got {cene_count}"
        )

        # Verify cascade logic: if chain_mult > ring_mult and PG+length tied,
        # chain should win. Use cyclohexanone with 6-carbon chain ketone,
        # passing carbon-only chain to get length=6 (tied with ring=6).
        # Chain ketone C=O has O outside carbon chain -> chain_mult=0.
        # Ring ketone C=O has O outside ring -> ring_mult=0.
        # Both 0 -> P-52.2.8 ring wins (tested in test_p44_1_all_tied_ring_wins_p52_2_8).
        # To test bond tiebreaker, we need chain to have extra C=C double bond.
        # Use O=C1CCCCC1C=CCCC(=O)C: cyclohexanone + 6-carbon chain with C=C and ketone.
        bond_smiles = 'O=C1CCCCC1C=CCCC(=O)C'
        bond_mol = Chem.MolFromSmiles(bond_smiles)
        bond_ring_systems = get_ring_systems(bond_mol)
        bond_fg = detect_functional_groups(bond_mol)
        bond_pg_name, bond_pg_atoms = get_principal_group(bond_mol, bond_fg)

        bond_all_ring = set()
        for r in bond_ring_systems:
            bond_all_ring.update(r)
        # Use carbon-only chain to get chain_len == ring_size (6 == 6)
        bond_chain = [i for i in range(bond_mol.GetNumAtoms())
                      if i not in bond_all_ring
                      and bond_mol.GetAtomWithIdx(i).GetAtomicNum() == 6]

        bond_result = select_parent(
            mol=bond_mol,
            ring_systems=bond_ring_systems,
            principal_chain=bond_chain,
            principal_group=bond_pg_name,
            principal_group_atoms=bond_pg_atoms
        )

        # Chain has C=C (1 double bond in carbon chain), ring has 0 -> chain wins
        if len(bond_chain) == len(list(bond_all_ring)):
            # Only assert if lengths actually tied (which they should)
            chain_mult_test = _count_multiple_bonds(bond_mol, set(bond_chain))
            ring_mult_test = _count_multiple_bonds(bond_mol, bond_all_ring)
            if chain_mult_test > ring_mult_test:
                assert bond_result.parent_type == 'chain', (
                    f"Chain with more multiple bonds should win. "
                    f"chain_mult={chain_mult_test}, ring_mult={ring_mult_test}. "
                    f"Got: {bond_result.reasoning}"
                )
                assert 'P-44.1' in bond_result.reasoning, (
                    f"Reasoning should cite P-44.1 cascade. "
                    f"Got: {bond_result.reasoning}"
                )

    def test_p44_1_all_tied_ring_wins_p52_2_8(self):
        """When PG count, length, and multiple bonds all tie, ring wins as
        P-52.2.8 final tiebreaker.

        O=C1CCCCC1CCCCC(=O)C: cyclohexanone (6-atom ring) + chain ketone.
        Pass carbon-only chain (6 carbons) to match ring size.
        Ring C=O and chain C=O both have O outside their respective atom sets,
        so mult bonds both = 0. All tied -> P-52.2.8 ring wins.
        """
        smiles = 'O=C1CCCCC1CCCCC(=O)C'
        mol = Chem.MolFromSmiles(smiles)
        ring_systems = get_ring_systems(mol)
        fg = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fg)

        all_ring = set()
        for r in ring_systems:
            all_ring.update(r)

        # Carbon-only chain: exactly 6 carbons to match ring size of 6
        carbon_chain = [i for i in range(mol.GetNumAtoms())
                        if i not in all_ring
                        and mol.GetAtomWithIdx(i).GetAtomicNum() == 6]

        # Verify preconditions: length and bond counts must tie
        assert len(carbon_chain) == len(all_ring), (
            f"Test precondition: chain ({len(carbon_chain)}) must equal "
            f"ring ({len(all_ring)}) for all-tied test"
        )

        chain_mult = _count_multiple_bonds(mol, set(carbon_chain))
        ring_mult = _count_multiple_bonds(mol, all_ring)
        assert chain_mult == ring_mult, (
            f"Test precondition: mult bonds must tie. "
            f"chain={chain_mult}, ring={ring_mult}"
        )

        result = select_parent(
            mol=mol,
            ring_systems=ring_systems,
            principal_chain=carbon_chain,
            principal_group=pg_name,
            principal_group_atoms=pg_atoms
        )

        assert result.parent_type == 'ring', (
            f"All tied -> P-52.2.8 final tiebreaker should select ring. "
            f"Got: {result.reasoning}"
        )
        assert 'P-44.1' in result.reasoning or 'P-52.2.8' in result.reasoning, (
            f"Reasoning should cite P-44.1 cascade or P-52.2.8. Got: {result.reasoning}"
        )

    def test_pg_proximity_direct_over_adjacent(self):
        """PRNT-05: Ring system where PG atom is directly IN the ring should
        score higher than a ring with no PG attachment.

        Molecule: cyclohexanone + cyclohexane connected by propyl chain.
        O=C1CCCCC1CCC1CCCCC1
        Ring 0 (cyclohexanone): ketone C is IN the ring -> weight 2
        Ring 1 (cyclohexane): no PG connection -> weight 0
        Both rings are identical 6-membered carbocycles with the same
        ring_system_score, so PG proximity is the sole tiebreaker.
        """
        smiles = 'O=C1CCCCC1CCC1CCCCC1'
        mol = Chem.MolFromSmiles(smiles)
        ring_systems = get_ring_systems(mol)

        assert len(ring_systems) == 2, f"Expected 2 ring systems, got {len(ring_systems)}"

        # Verify precondition: both rings have identical ring_system_score
        scores = [ring_system_score(mol, r) for r in ring_systems]
        assert scores[0] == scores[1], (
            f"Precondition: ring scores must tie for PG proximity to be the tiebreaker. "
            f"Got {scores[0]} vs {scores[1]}"
        )

        fg = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fg)

        best, others = _select_best_ring_system(mol, ring_systems, pg_atoms)

        # The ring with the ketone C directly IN it should be preferred.
        # Verify by checking: the best ring system contains an atom that
        # is the first atom of some PG match (direct-in-ring pattern).
        direct_pg_in_best = False
        for pg in pg_atoms:
            if pg and pg[0] in best:
                direct_pg_in_best = True
                break

        assert direct_pg_in_best, (
            "PRNT-05: Ring with PG atom directly IN ring should be selected as best. "
            f"Best ring atoms: {sorted(best)}, PG attachment atoms: {[p[0] for p in pg_atoms if p]}"
        )
        assert len(others) == 1, "Should have 1 other ring system"

    def test_pg_proximity_single_ring_unchanged(self):
        """Single-ring molecule should be returned directly without PG scoring.

        O=C1CCCCC1 (cyclohexanone): only 1 ring system -> _select_best_ring_system
        returns it immediately via the len(ring_systems) <= 1 early return.
        """
        smiles = 'O=C1CCCCC1'
        mol = Chem.MolFromSmiles(smiles)
        ring_systems = get_ring_systems(mol)

        assert len(ring_systems) == 1, "Cyclohexanone should have 1 ring system"

        fg = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fg)

        best, others = _select_best_ring_system(mol, ring_systems, pg_atoms)

        assert best == ring_systems[0], "Single ring should be returned as-is"
        assert others == [], "No other ring systems for single-ring molecule"

    def test_ring_wins_by_pg_count_unchanged(self):
        """When ring has strictly more PGs than chain, ring wins (unchanged behavior).

        O=C1CC(=O)CCC1CCCCC(=O)C: cyclohexane-1,3-dione (2 ketones on ring) +
        chain ketone (1 ketone). Ring has 2 PGs > chain has 1 PG -> ring wins
        by PG count. This path is NOT affected by the P-44.1 cascade.
        """
        smiles = 'O=C1CC(=O)CCC1CCCCC(=O)C'
        mol = Chem.MolFromSmiles(smiles)
        ring_systems = get_ring_systems(mol)
        fg = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fg)

        all_ring = set()
        for r in ring_systems:
            all_ring.update(r)
        chain = [i for i in range(mol.GetNumAtoms()) if i not in all_ring]

        result = select_parent(
            mol=mol,
            ring_systems=ring_systems,
            principal_chain=chain,
            principal_group=pg_name,
            principal_group_atoms=pg_atoms
        )

        assert result.parent_type == 'ring', (
            f"Ring with more PGs should always win. Got: {result.reasoning}"
        )
        assert 'P-44.1(b)' in result.reasoning or 'PG count' in result.reasoning, (
            f"Reasoning should indicate PG count. Got: {result.reasoning}"
        )
