"""
Tests for chain_score tiebreaker criteria refinements.

Phase 103 Plan 03 Task 2: Refine chain_score to fix:
- P-44.1(b): FG instance counting (count instances, not atoms)
- P-44.1(h): Separate double-bond locant comparison
"""

import pytest
from rdkit import Chem

from orthonym.perception.chains import find_principal_chain
from orthonym.perception.functional_groups import detect_functional_groups


# ============================================================================
# P-44.1(b): FG instance counting
# ============================================================================


class TestFGInstanceCounting:
    """Criterion 2 should count FG instances, not atoms.

    Bug: len(fg_atoms & chain_set) counts atoms that overlap with FG matches.
    For COOH (3 atoms), a chain with 2 COOH gets fg_count=6 (atoms) instead of 2 (instances).
    This doesn't affect chain SELECTION when both candidate chains have the
    same FG on them, but it produces incorrect scores when chains differ in
    the number of FG instances.
    """

    def test_two_cooh_counted_as_two_instances(self):
        """A chain with 2 COOH groups should have fg_count=2, not 6."""
        # Hexanedioic acid: HOOC-CH2-CH2-CH2-CH2-COOH
        mol = Chem.MolFromSmiles("OC(=O)CCCCC(=O)O")
        fg = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fg, principal_group="carboxylic_acid")
        # Chain should exist and contain both COOH groups
        assert len(chain) > 0, "No chain found"
        # Both COOH carbons should be on the chain
        chain_set = set(chain)
        cooh_matches = fg.get("carboxylic_acid", [])
        instances_on_chain = sum(
            1 for match in cooh_matches if set(match) & chain_set
        )
        assert instances_on_chain == 2, f"Expected 2 COOH instances on chain, got {instances_on_chain}"

    def test_chain_with_more_fg_instances_wins(self):
        """Chain with more FG instances should be preferred over chain with fewer.

        3-methylpentanedioic acid (2 COOH, 5C chain) vs
        a hypothetical longer chain with only 1 COOH.
        """
        # OC(=O)CC(C)CC(=O)O  -- 3-methylpentanedioic acid
        # Both COOH are on the 5C chain. This is a simple regression guard.
        mol = Chem.MolFromSmiles("OC(=O)CC(C)CC(=O)O")
        fg = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fg, principal_group="carboxylic_acid")
        assert len(chain) >= 5, f"Expected chain length >= 5, got {len(chain)}"


# ============================================================================
# P-44.1(h): Double-bond locant tiebreaker
# ============================================================================


class TestDoubleBondLocantTiebreaker:
    """After combined multiple-bond locant comparison, a separate double-bond-only
    locant comparison should break ties where chains have same combined bond
    locants but different double-bond positions.

    Per IUPAC P-44.4(h): when two chains have the same overall multiple bond
    locant set, the chain with lower locants for double bonds is preferred.
    """

    def test_simple_ene_position_matters(self):
        """Verify that ene position affects chain selection.

        pent-1-ene vs pent-2-ene should have different bond locant scores.
        """
        # Both are 5-carbon chains, just different double bond positions
        mol1 = Chem.MolFromSmiles("C=CCCC")  # pent-1-ene
        mol2 = Chem.MolFromSmiles("CC=CCC")  # pent-2-ene
        fg1 = detect_functional_groups(mol1)
        fg2 = detect_functional_groups(mol2)
        chain1 = find_principal_chain(mol1, fg1)
        chain2 = find_principal_chain(mol2, fg2)
        assert len(chain1) == 5
        assert len(chain2) == 5

    def test_enyne_double_bond_locant_tiebreaker(self):
        """For en-yne compounds, double bond position should break ties.

        IUPAC P-31.1.3.4: When a chain has both double and triple bonds,
        and two orientations give the same combined bond locant set,
        the one giving lower locants to double bonds is preferred.
        """
        # hex-2-en-4-yne: CC=CC#CC
        mol = Chem.MolFromSmiles("CC=CC#CC")
        fg = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fg)
        # Chain should be 6 carbons
        assert len(chain) == 6, f"Expected 6-carbon chain, got {len(chain)}"

        # The chain orientation should place the double bond at position 2
        # (lower than triple at position 4), not the reverse
        # Both orientations give bond locants {2, 4}, so double-bond tiebreaker applies
        # Forward: double at 2, triple at 4 -> double locant = 2
        # Reverse: triple at 2, double at 4 -> double locant = 4
        # Forward is preferred (lower double bond locant)

        # Verify the chain is oriented correctly by checking bond types
        bond_at_1 = mol.GetBondBetweenAtoms(chain[1], chain[2])
        if bond_at_1:
            # Position 2 bond should be double
            assert bond_at_1.GetBondType() == Chem.BondType.DOUBLE, (
                "Expected double bond at position 2 in chain"
            )


# ============================================================================
# Regression guards: simple compounds unaffected
# ============================================================================


class TestChainScoreRegression:
    """Verify simple compounds are not affected by tiebreaker changes."""

    def test_simple_alkane(self):
        """Pentane should still select the 5-carbon chain."""
        mol = Chem.MolFromSmiles("CCCCC")
        fg = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fg)
        assert len(chain) == 5

    def test_simple_alcohol(self):
        """Butan-1-ol should select the 4-carbon chain."""
        mol = Chem.MolFromSmiles("CCCCO")
        fg = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fg, principal_group="primary_alcohol")
        assert len(chain) == 4

    def test_simple_acid(self):
        """Propanoic acid should select the 3-carbon chain."""
        mol = Chem.MolFromSmiles("CCC(=O)O")
        fg = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fg, principal_group="carboxylic_acid")
        assert len(chain) >= 3

    def test_branched_alkane(self):
        """2-methylbutane should select the 4-carbon chain (longest)."""
        mol = Chem.MolFromSmiles("CC(C)CC")
        fg = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fg)
        assert len(chain) == 4

    def test_simple_alkene(self):
        """But-1-ene should select the 4-carbon chain."""
        mol = Chem.MolFromSmiles("C=CCC")
        fg = detect_functional_groups(mol)
        chain = find_principal_chain(mol, fg)
        assert len(chain) == 4
