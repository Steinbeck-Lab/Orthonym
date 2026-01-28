"""
Unit tests for locant assignment utilities.

Tests the core locant infrastructure:
- build_atom_to_locant: atom index -> locant mapping
- compare_locant_sets: first-point-of-difference comparison
- orient_chain: IUPAC 2013 chain orientation criteria
- get_functional_group_locants: FG atom -> chain locant resolution

All tests use @pytest.mark.unit marker for fast execution.
"""

import pytest
from rdkit import Chem

from orthonym.rules.locants import (
    build_atom_to_locant,
    compare_locant_sets,
    orient_chain,
    get_functional_group_locants,
)
from orthonym.perception.functional_groups import detect_functional_groups
from orthonym.perception.chains import find_principal_chain
from orthonym.rules.seniority import get_principal_group


# ============================================================================
# build_atom_to_locant tests
# ============================================================================


@pytest.mark.unit
class TestBuildAtomToLocant:
    """Tests for build_atom_to_locant."""

    def test_empty_chain(self):
        """Empty chain returns empty dict."""
        assert build_atom_to_locant([]) == {}

    def test_single_atom(self):
        """Single atom chain maps to locant 1."""
        assert build_atom_to_locant([0]) == {0: 1}

    def test_multi_atom_chain(self):
        """Multi-atom chain maps correctly to 1-indexed locants."""
        result = build_atom_to_locant([5, 3, 1, 0])
        assert result == {5: 1, 3: 2, 1: 3, 0: 4}

    def test_sequential_atoms(self):
        """Sequential atom indices map correctly."""
        result = build_atom_to_locant([0, 1, 2, 3])
        assert result == {0: 1, 1: 2, 2: 3, 3: 4}

    def test_reversed_atoms(self):
        """Reversed atom order produces reversed locants."""
        result = build_atom_to_locant([3, 2, 1, 0])
        assert result == {3: 1, 2: 2, 1: 3, 0: 4}

    def test_locants_are_1_indexed(self):
        """Locants start at 1, not 0."""
        result = build_atom_to_locant([7])
        assert result[7] == 1

    def test_all_atoms_mapped(self):
        """Every atom in the chain gets a locant."""
        chain = [10, 20, 30, 40, 50]
        result = build_atom_to_locant(chain)
        assert len(result) == 5
        for atom_idx in chain:
            assert atom_idx in result

    def test_locants_are_consecutive(self):
        """Locants are 1, 2, 3, ... regardless of atom index gaps."""
        chain = [100, 5, 42, 0]
        result = build_atom_to_locant(chain)
        assert list(result.values()) == [1, 2, 3, 4]


# ============================================================================
# compare_locant_sets tests
# ============================================================================


@pytest.mark.unit
class TestCompareLocantSets:
    """Tests for compare_locant_sets (first-point-of-difference rule)."""

    def test_first_element_wins_a(self):
        """Set A wins at first position: 2 < 3."""
        assert compare_locant_sets([2, 3, 5], [3, 4, 6]) == -1

    def test_first_element_wins_b(self):
        """Set B wins at second position: 3 < 4."""
        assert compare_locant_sets([2, 4, 5], [2, 3, 5]) == 1

    def test_equal_sets(self):
        """Identical sets return 0."""
        assert compare_locant_sets([2, 3], [2, 3]) == 0

    def test_difference_at_first_position(self):
        """Set A wins at position 1: 1 < 2."""
        assert compare_locant_sets([1, 3, 5, 7], [2, 3, 5, 7]) == -1

    def test_difference_at_last_position(self):
        """Set B wins at last position: 7 < 8."""
        assert compare_locant_sets([2, 3, 5, 8], [2, 3, 5, 7]) == 1

    def test_single_element_sets(self):
        """Single-element comparison."""
        assert compare_locant_sets([1], [2]) == -1
        assert compare_locant_sets([5], [3]) == 1
        assert compare_locant_sets([4], [4]) == 0

    def test_unsorted_input(self):
        """Inputs are sorted internally before comparison."""
        # [5, 2, 3] sorted = [2, 3, 5]
        # [6, 4, 3] sorted = [3, 4, 6]
        assert compare_locant_sets([5, 2, 3], [6, 4, 3]) == -1

    def test_shorter_set_wins_on_tie(self):
        """When all compared elements equal, shorter set wins."""
        assert compare_locant_sets([2, 3], [2, 3, 5]) == -1

    def test_longer_set_loses_on_tie(self):
        """Longer set loses when prefix matches shorter set."""
        assert compare_locant_sets([2, 3, 5], [2, 3]) == 1

    def test_not_sum_of_locants(self):
        """Verify first-point-of-difference, NOT sum-of-locants.

        Sum of [1, 4, 6] = 11, sum of [2, 3, 5] = 10.
        Sum-of-locants would prefer B (10 < 11), but first-point-of-
        difference prefers A (1 < 2).
        """
        assert compare_locant_sets([1, 4, 6], [2, 3, 5]) == -1

    def test_empty_sets(self):
        """Two empty sets are equal."""
        assert compare_locant_sets([], []) == 0

    def test_empty_vs_nonempty(self):
        """Empty set wins over non-empty (fewer locants)."""
        assert compare_locant_sets([], [1]) == -1
        assert compare_locant_sets([1], []) == 1


# ============================================================================
# orient_chain tests (real molecule tests)
# ============================================================================


@pytest.mark.unit
class TestOrientChain:
    """Tests for orient_chain using real molecules."""

    def _get_chain_and_orient(self, smiles):
        """Helper: get principal chain and orient it."""
        mol = Chem.MolFromSmiles(smiles)
        fgs = detect_functional_groups(mol)
        pg_name, pg_atoms = get_principal_group(mol, fgs)

        chain = find_principal_chain(mol, fgs, pg_name)

        pg_atom_set = set()
        if pg_name and pg_name in fgs:
            for match in fgs[pg_name]:
                pg_atom_set.update(match)

        # Detect C=C and C#C bonds
        double_bonds = []
        triple_bonds = []
        for bond in mol.GetBonds():
            bt = bond.GetBondType()
            a_sym = bond.GetBeginAtom().GetSymbol()
            b_sym = bond.GetEndAtom().GetSymbol()
            if a_sym == "C" and b_sym == "C":
                if bt == Chem.BondType.DOUBLE:
                    double_bonds.append(
                        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                    )
                elif bt == Chem.BondType.TRIPLE:
                    triple_bonds.append(
                        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                    )

        oriented = orient_chain(
            chain=chain,
            mol=mol,
            principal_group_atoms=pg_atom_set,
            double_bonds=double_bonds,
            triple_bonds=triple_bonds,
        )
        locant_map = build_atom_to_locant(oriented)
        return mol, oriented, locant_map, fgs, pg_name

    def test_2_methylbutane(self):
        """2-methylbutane: methyl at position 2, not 3.

        CC(C)CC -> principal chain has 4 carbons; the methyl branch
        must receive locant 2 (not 3).
        """
        mol, oriented, locant_map, fgs, pg = self._get_chain_and_orient("CC(C)CC")

        # Chain should be 4 carbons long
        assert len(oriented) == 4

        # Find the methyl substituent position
        chain_set = set(oriented)
        branch_atom = None
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "C" and atom.GetIdx() not in chain_set:
                # This is the branch carbon; find where it attaches
                for nbr in atom.GetNeighbors():
                    if nbr.GetIdx() in chain_set:
                        branch_atom = nbr.GetIdx()
                        break

        assert branch_atom is not None, "Should find branch attachment point"
        branch_locant = locant_map[branch_atom]
        assert branch_locant == 2, f"Methyl should be at locant 2, got {branch_locant}"

    def test_propan_1_ol_orientation(self):
        """propan-1-ol (CCCO): OH-bearing carbon at locant 1.

        The chain must be oriented so the OH-bearing carbon gets
        locant 1 (lowest locant for principal group).
        """
        mol, oriented, locant_map, fgs, pg = self._get_chain_and_orient("CCCO")

        assert pg == "primary_alcohol"
        assert len(oriented) == 3

        # Find the carbon bearing OH
        alcohol_matches = fgs.get("primary_alcohol", [])
        assert len(alcohol_matches) > 0

        fg_locants = get_functional_group_locants(
            oriented, alcohol_matches, locant_map, mol=mol
        )
        assert fg_locants == [1], f"OH carbon should be locant 1, got {fg_locants}"

    def test_butan_2_one_orientation(self):
        """butan-2-one (CCC(=O)C): carbonyl carbon at locant 2.

        The chain must be oriented so the ketone C gets locant 2
        (lowest possible for the principal group).
        """
        mol, oriented, locant_map, fgs, pg = self._get_chain_and_orient("CCC(=O)C")

        assert pg == "ketone"
        assert len(oriented) == 4

        ketone_matches = fgs.get("ketone", [])
        assert len(ketone_matches) > 0

        fg_locants = get_functional_group_locants(
            oriented, ketone_matches, locant_map, mol=mol
        )
        assert 2 in fg_locants, f"Ketone should be at locant 2, got {fg_locants}"

    def test_single_atom_chain(self):
        """Single-atom chain (methane): orient_chain returns as-is."""
        mol = Chem.MolFromSmiles("C")
        chain = [0]
        result = orient_chain(
            chain=chain,
            mol=mol,
            principal_group_atoms=set(),
            double_bonds=[],
            triple_bonds=[],
        )
        assert result == [0]

    def test_symmetric_chain_no_fg(self):
        """Symmetric chain with no FG (butane): either direction is valid."""
        mol = Chem.MolFromSmiles("CCCC")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, None)

        result = orient_chain(
            chain=chain,
            mol=mol,
            principal_group_atoms=set(),
            double_bonds=[],
            triple_bonds=[],
        )
        # Should return a valid 4-carbon chain
        assert len(result) == 4


# ============================================================================
# get_functional_group_locants tests
# ============================================================================


@pytest.mark.unit
class TestGetFunctionalGroupLocants:
    """Tests for get_functional_group_locants."""

    def test_empty_matches(self):
        """No FG matches returns empty list."""
        result = get_functional_group_locants(
            chain=[0, 1, 2],
            fg_atom_tuples=[],
            atom_to_locant={0: 1, 1: 2, 2: 3},
        )
        assert result == []

    def test_alcohol_locant(self):
        """Alcohol: carbon bearing OH is the locant.

        propan-1-ol: the primary_alcohol SMARTS [OX2H][CX4H2]
        matches (O, C). The carbon should provide locant 1.
        """
        mol = Chem.MolFromSmiles("CCCO")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, "primary_alcohol")

        pg_atom_set = set()
        for match in fgs.get("primary_alcohol", []):
            pg_atom_set.update(match)

        oriented = orient_chain(
            chain=chain,
            mol=mol,
            principal_group_atoms=pg_atom_set,
            double_bonds=[],
            triple_bonds=[],
        )
        locant_map = build_atom_to_locant(oriented)

        fg_locants = get_functional_group_locants(
            oriented,
            fgs.get("primary_alcohol", []),
            locant_map,
            mol=mol,
        )
        assert fg_locants == [1]

    def test_ketone_locant(self):
        """Ketone: carbonyl carbon is the locant (not a neighbor).

        butan-2-one: SMARTS [#6][CX3](=O)[#6] matches 4 atoms.
        Only the CX3 carbon is the locant-defining atom.
        """
        mol = Chem.MolFromSmiles("CCC(=O)C")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, "ketone")

        pg_atom_set = set()
        for match in fgs.get("ketone", []):
            pg_atom_set.update(match)

        oriented = orient_chain(
            chain=chain,
            mol=mol,
            principal_group_atoms=pg_atom_set,
            double_bonds=[],
            triple_bonds=[],
        )
        locant_map = build_atom_to_locant(oriented)

        fg_locants = get_functional_group_locants(
            oriented,
            fgs.get("ketone", []),
            locant_map,
            mol=mol,
        )
        assert 2 in fg_locants, f"Ketone should be at locant 2, got {fg_locants}"

    def test_no_fg_atoms_on_chain(self):
        """FG atoms not on chain returns empty list."""
        # Simulate: chain is [0, 1, 2], but FG is at atoms [5, 6]
        result = get_functional_group_locants(
            chain=[0, 1, 2],
            fg_atom_tuples=[(5, 6)],
            atom_to_locant={0: 1, 1: 2, 2: 3},
        )
        assert result == []

    def test_multiple_fg_instances(self):
        """Multiple FG instances return sorted locant list."""
        # pentane-2,4-diol: two OH groups
        mol = Chem.MolFromSmiles("CC(O)CC(O)C")
        fgs = detect_functional_groups(mol)

        # Get alcohol matches
        alcohol_key = None
        for key in ["secondary_alcohol", "primary_alcohol", "tertiary_alcohol"]:
            if key in fgs and fgs[key]:
                alcohol_key = key
                break

        if alcohol_key:
            chain = find_principal_chain(mol, fgs, alcohol_key)
            pg_atoms = set()
            for match in fgs[alcohol_key]:
                pg_atoms.update(match)

            oriented = orient_chain(
                chain=chain,
                mol=mol,
                principal_group_atoms=pg_atoms,
                double_bonds=[],
                triple_bonds=[],
            )
            locant_map = build_atom_to_locant(oriented)

            fg_locants = get_functional_group_locants(
                oriented,
                fgs[alcohol_key],
                locant_map,
                mol=mol,
            )
            # Should have 2 locants, both sorted
            assert len(fg_locants) == 2
            assert fg_locants == sorted(fg_locants)

    def test_carboxylic_acid_locant(self):
        """Carboxylic acid: carboxyl carbon should be locant 1 (terminal).

        propanoic acid (CCC(=O)O): the COOH carbon is at the chain end.
        """
        mol = Chem.MolFromSmiles("CCC(=O)O")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, "carboxylic_acid")

        pg_atom_set = set()
        for match in fgs.get("carboxylic_acid", []):
            pg_atom_set.update(match)

        oriented = orient_chain(
            chain=chain,
            mol=mol,
            principal_group_atoms=pg_atom_set,
            double_bonds=[],
            triple_bonds=[],
        )
        locant_map = build_atom_to_locant(oriented)

        fg_locants = get_functional_group_locants(
            oriented,
            fgs.get("carboxylic_acid", []),
            locant_map,
            mol=mol,
        )
        assert fg_locants == [1], f"COOH should be at locant 1, got {fg_locants}"


# ============================================================================
# Integration: MolecularFeatures.atom_to_locant populated
# ============================================================================


@pytest.mark.unit
class TestNamerAtomToLocant:
    """Verify that Orthonym._classify populates atom_to_locant."""

    def test_atom_to_locant_populated_for_acyclic(self):
        """atom_to_locant is populated for acyclic molecules."""
        from orthonym.namer import Orthonym

        namer = Orthonym()
        mol = Chem.MolFromSmiles("CCCC")
        canonical = Chem.MolToSmiles(mol, canonical=True)
        features = namer._perceive(mol, "CCCC", canonical)
        namer._classify(features)

        assert len(features.atom_to_locant) > 0
        assert len(features.atom_to_locant) == len(features.principal_chain)

    def test_atom_to_locant_empty_for_no_chain(self):
        """atom_to_locant stays empty when no principal chain found."""
        from orthonym.namer import MolecularFeatures

        features = MolecularFeatures(mol=None)
        assert features.atom_to_locant == {}

    def test_atom_to_locant_consistent_with_chain(self):
        """atom_to_locant keys match principal_chain atoms."""
        from orthonym.namer import Orthonym

        namer = Orthonym()
        mol = Chem.MolFromSmiles("CC(C)CC")
        canonical = Chem.MolToSmiles(mol, canonical=True)
        features = namer._perceive(mol, "CC(C)CC", canonical)
        namer._classify(features)

        chain_set = set(features.principal_chain)
        locant_keys = set(features.atom_to_locant.keys())
        assert chain_set == locant_keys

    def test_locant_values_are_consecutive(self):
        """Locant values are 1, 2, 3, ..., n."""
        from orthonym.namer import Orthonym

        namer = Orthonym()
        mol = Chem.MolFromSmiles("CCCCC")
        canonical = Chem.MolToSmiles(mol, canonical=True)
        features = namer._perceive(mol, "CCCCC", canonical)
        namer._classify(features)

        values = sorted(features.atom_to_locant.values())
        expected = list(range(1, len(features.principal_chain) + 1))
        assert values == expected
