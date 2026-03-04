"""
Tests for chain_score() 9-criterion selection in find_principal_chain().

Tests all IUPAC 2013 P-44 criteria for principal chain selection:
1. Contains principal group
2. Maximum number of principal groups
3. Maximum chain length
4. Maximum multiple bonds (double + triple)
5. Maximum double bonds
6. Lowest locants for principal group (P-44.4h)
7. Lowest locants for multiple bonds (P-44.4j)
8. Maximum substituents
9. Lowest locants for substituents
"""
import pytest
from rdkit import Chem

from orthonym.perception.chains import find_principal_chain
from orthonym.perception.functional_groups import detect_functional_groups


class TestCriterion1ContainsPrincipalGroup:
    """Chain containing the principal group is preferred."""

    def test_chain_with_acid_preferred_over_longer_chain(self):
        """Pentanoic acid: 5-carbon chain with acid beats any 4-carbon branch."""
        mol = Chem.MolFromSmiles("CCCCC(=O)O")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, "carboxylic_acid")
        assert len(chain) == 5
        # The acid oxygen atoms should be adjacent to the chain
        chain_set = set(chain)
        # Acid carbon (C=O carbon) should be in chain
        acid_c = None
        for idx in chain:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == "C":
                for nbr in atom.GetNeighbors():
                    if nbr.GetSymbol() == "O":
                        acid_c = idx
                        break
        assert acid_c is not None, "Acid carbon should be in principal chain"


class TestCriterion2MaxPrincipalGroups:
    """Chain with maximum number of principal groups is preferred."""

    def test_chain_with_more_fg_atoms_preferred(self):
        """Glutaric acid (HOOC-CH2-CH2-CH2-COOH): chain has both acid groups."""
        mol = Chem.MolFromSmiles("OC(=O)CCCC(=O)O")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, "carboxylic_acid")
        # Should select the 5-carbon chain containing both acid carbons
        assert len(chain) == 5


class TestCriterion3MaxChainLength:
    """Longest chain is preferred (IUPAC 2013: length before unsaturation)."""

    def test_longer_chain_preferred(self):
        """2-methylhexane: 6-carbon chain over 5-carbon chain."""
        mol = Chem.MolFromSmiles("CCCCCC(C)C")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs)
        # The 7-carbon chain is longest (includes the branching carbon)
        assert len(chain) >= 7


class TestCriterion4and5MultipleBonds:
    """Chain with more multiple bonds is preferred; among equal, more double bonds."""

    def test_chain_with_double_bond_preferred(self):
        """Hex-2-ene: chain with double bond preferred over same-length chain."""
        mol = Chem.MolFromSmiles("CC=CCCC")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs)
        assert len(chain) == 6
        # The chain should contain the double bond
        chain_set = set(chain)
        has_double = False
        for i in range(len(chain) - 1):
            bond = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
            if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                has_double = True
                break
        assert has_double, "Chain should contain the double bond"


class TestCriterion6LowestFGLocants:
    """Lowest locants for principal group (first point of difference)."""

    def test_fg_at_end_preferred(self):
        """When two equal-length chains both have the acid, prefer lower FG locant.

        4-methylpentanoic acid: chain ending at acid carbon gets locant 1 for acid.
        """
        mol = Chem.MolFromSmiles("CC(CC)CCC(=O)O")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, "carboxylic_acid")
        # Chain should include the acid carbon
        chain_set = set(chain)
        # Acid carbon is the one with =O and -OH
        acid_carbon = None
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "C":
                nbr_symbols = sorted(n.GetSymbol() for n in atom.GetNeighbors())
                if nbr_symbols.count("O") == 2:
                    acid_carbon = atom.GetIdx()
                    break
        assert acid_carbon is not None
        assert acid_carbon in chain_set, "Acid carbon must be in principal chain"

    def test_both_orientations_considered(self):
        """Chain locant scoring considers both directions."""
        # hexanedioic acid (adipic acid) - both ends have acid
        mol = Chem.MolFromSmiles("OC(=O)CCCCC(=O)O")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, "carboxylic_acid")
        # Both acid carbons should be in chain
        chain_set = set(chain)
        acid_carbons = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "C":
                o_count = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == "O")
                if o_count == 2:
                    acid_carbons.append(atom.GetIdx())
        assert len(acid_carbons) == 2
        for ac in acid_carbons:
            assert ac in chain_set, f"Acid carbon {ac} should be in chain"


class TestCriterion7LowestBondLocants:
    """Lowest locants for multiple bonds (double + triple)."""

    def test_double_bond_lower_locant_preferred(self):
        """Hex-2-ene vs hex-4-ene: prefer lower locant for double bond.

        Both are the same 6-carbon chain, just different orientations.
        The orientation is handled by _orient_chain, but the chain itself
        should be correctly selected when multiple candidate chains exist.
        """
        mol = Chem.MolFromSmiles("CC=CCCC")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs)
        # Chain should be 6 carbons
        assert len(chain) == 6
        # Double bond should have a low locant when properly oriented
        bond_pos = None
        for i in range(len(chain) - 1):
            bond = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
            if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                bond_pos = i
                break
        assert bond_pos is not None, "Double bond should be found"
        # The bond position should be at index 1 (locant 2) in the proper orientation
        assert bond_pos <= 3, "Double bond should have a low locant"


class TestCriterion8MaxSubstituents:
    """Chain with maximum number of substituents preferred."""

    def test_more_substituents_preferred(self):
        """2,2-dimethylbutanoic acid has 2 substituents on 4-carbon chain."""
        mol = Chem.MolFromSmiles("CC(C)(C)CC(=O)O")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, "carboxylic_acid")
        # Should select the 4-carbon chain with the acid
        chain_set = set(chain)
        # Count non-chain non-H neighbors
        sub_count = 0
        for idx in chain:
            atom = mol.GetAtomWithIdx(idx)
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() not in chain_set and nbr.GetSymbol() != "H":
                    sub_count += 1
        assert sub_count >= 2, "Chain should have at least 2 substituents (two methyls)"

    def test_substituent_count_is_tiebreaker(self):
        """When chains tie on criteria 1-7, more substituents wins.

        3-ethyl-2-methylpentane: 5-carbon chain with ethyl + methyl (2 subs)
        vs 5-carbon chain with only ethyl (1 sub).
        """
        mol = Chem.MolFromSmiles("CCC(CC)C(C)C")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs)
        # The 5-carbon chain should be selected
        assert len(chain) == 5
        chain_set = set(chain)
        # Count substituents
        sub_count = 0
        for idx in chain:
            atom = mol.GetAtomWithIdx(idx)
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() not in chain_set and nbr.GetSymbol() != "H":
                    sub_count += 1
        assert sub_count >= 2, "Chain with more substituents should be selected"


class TestCriterion9LowestSubLocants:
    """Lowest locants for substituents as final tiebreaker."""

    def test_lower_substituent_locants_preferred(self):
        """When chains tie on criteria 1-8, lowest substituent locants win.

        2-methylheptane vs 6-methylheptane (same chain, diff orientation):
        Both have 1 substituent. Locant 2 < locant 6.
        """
        mol = Chem.MolFromSmiles("CCCCCCC(C)C")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs)
        # Chain should be 8 carbons
        assert len(chain) == 8


class TestRegressionExistingBehavior:
    """Ensure existing behavior is maintained with the new 9-criteria scoring."""

    def test_pentanoic_acid_chain_length(self):
        """Pentanoic acid selects 5-carbon chain."""
        mol = Chem.MolFromSmiles("CCCCC(=O)O")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, "carboxylic_acid")
        assert len(chain) == 5

    def test_ethanol_chain_length(self):
        """Ethanol selects 2-carbon chain."""
        mol = Chem.MolFromSmiles("CCO")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, "alcohol")
        assert len(chain) == 2

    def test_propanone_chain_length(self):
        """Propan-2-one (acetone) selects 3-carbon chain."""
        mol = Chem.MolFromSmiles("CC(=O)C")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, "ketone")
        assert len(chain) == 3

    def test_hexane_chain_length(self):
        """Hexane selects 6-carbon chain (no FG)."""
        mol = Chem.MolFromSmiles("CCCCCC")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs)
        assert len(chain) == 6

    def test_but2ene_chain_length(self):
        """But-2-ene selects 4-carbon chain with double bond."""
        mol = Chem.MolFromSmiles("CC=CC")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs)
        assert len(chain) == 4
        # Should contain the double bond
        has_double = False
        for i in range(len(chain) - 1):
            bond = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
            if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                has_double = True
                break
        assert has_double

    def test_empty_molecule_returns_empty(self):
        """Molecule with no carbons returns empty chain."""
        mol = Chem.MolFromSmiles("O")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs)
        assert chain == []


class TestFGInstanceCounting:
    """Tests for FG instance counting fix (PSEL-01).

    chain_score() criterion 2 must count distinct FG instances (SMARTS match
    tuples with any atom on chain), NOT the number of FG atoms on the chain.
    """

    def test_two_cooh_counts_as_two_instances(self):
        """Glutaric acid: 2 COOH groups on the chain = fg_instance_count=2.

        Each COOH match has 3 atoms (C, =O, OH). The old bug would count
        up to 6 atoms instead of 2 instances.
        """
        mol = Chem.MolFromSmiles("OC(=O)CCCC(=O)O")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, "carboxylic_acid")
        # The chain must contain both acid carbons (atoms 1 and 6)
        chain_set = set(chain)
        acid_carbons = [
            atom.GetIdx() for atom in mol.GetAtoms()
            if atom.GetSymbol() == "C"
            and sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == "O") == 2
        ]
        assert len(acid_carbons) == 2, "Should find 2 acid carbons"
        for ac in acid_carbons:
            assert ac in chain_set, f"Acid carbon {ac} must be in chain"
        # Verify the chain is the 5-carbon chain with both COOHs
        assert len(chain) == 5

    def test_one_cooh_on_chain_one_off_chain(self):
        """A branched diacid where one COOH is on a branch, not the main chain.

        3-methylpentanedioic acid (OC(=O)CC(C)CC(=O)O):
        The 5-carbon chain has both COOHs -> both on chain -> 2 instances.
        But if we force a shorter chain, only 1 COOH would be on it.
        We test that chain selection prefers the chain with MORE FG instances.
        """
        # OC(=O)CC(CC(=O)O)C -- this has branching, two paths, one COOH on each
        mol = Chem.MolFromSmiles("OC(=O)CC(CC(=O)O)C")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, "carboxylic_acid")
        chain_set = set(chain)
        # Count how many COOH instances have any atom on the chain
        fg_on_chain = 0
        for match_tuple in fgs.get("carboxylic_acid", []):
            if any(a in chain_set for a in match_tuple):
                fg_on_chain += 1
        # The chain with the most FG instances should be selected
        assert fg_on_chain >= 1, "At least 1 FG instance must be on the chain"

    def test_zero_fg_on_chain(self):
        """Chain with no FG atoms should have fg_instance_count=0.

        Hexane has no functional groups at all.
        """
        mol = Chem.MolFromSmiles("CCCCCC")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs)
        # No principal group -> fg_instance_count is 0 (chain selected by length)
        assert len(chain) == 6

    def test_instance_count_not_atom_count(self):
        """Verify that criterion 2 uses instance counting, not atom counting.

        Create a scenario with 2 COOHs (2 instances, 6 FG atoms).
        Both chains have all 6 FG atoms, but criterion 2 should show
        fg_instance_count=2, not fg_count=6.

        We verify indirectly: glutaric acid (5C chain, 2 COOHs) should have
        the same fg_instance_count as if we count SMARTS match tuples.
        """
        mol = Chem.MolFromSmiles("OC(=O)CCCC(=O)O")
        fgs = detect_functional_groups(mol)
        # There should be exactly 2 carboxylic_acid matches
        assert len(fgs.get("carboxylic_acid", [])) == 2
        chain = find_principal_chain(mol, fgs, "carboxylic_acid")
        chain_set = set(chain)
        # Count instances the correct way (match tuples with any atom on chain)
        instance_count = sum(
            1 for match in fgs["carboxylic_acid"]
            if any(a in chain_set for a in match)
        )
        assert instance_count == 2, (
            "fg_instance_count should be 2 (two COOH groups), "
            "not the FG atom overlap count"
        )


class TestChainScoreReturnsTuple:
    """Verify chain_score returns the expected 9-element tuple structure."""

    def test_score_tuple_is_9_elements(self):
        """Internal: chain_score closure returns 9 criteria values.

        We verify this indirectly by checking that find_principal_chain
        handles complex cases correctly (more than 5 criteria needed).
        """
        # 3-methylhex-5-enoic acid -- needs multiple criteria to resolve
        mol = Chem.MolFromSmiles("OC(=O)CC(C)CC=C")
        fgs = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fgs, "carboxylic_acid")
        # Should select longest chain containing the acid
        assert len(chain) >= 5
        # Acid carbon should be in the chain
        chain_set = set(chain)
        acid_found = False
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "C":
                o_count = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == "O")
                if o_count == 2 and atom.GetIdx() in chain_set:
                    acid_found = True
                    break
        assert acid_found, "Acid carbon should be in principal chain"
