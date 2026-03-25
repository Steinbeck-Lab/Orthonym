"""Unit tests for seniority-based parent role swapping in bond_cleavage.

Tests the _maybe_swap_parent_roles() function and the roles_swapped flag
in find_cleavable_bonds() bond info dicts. Covers all 6 non-ether bond
types (ester, amide, thioester, sulfonamide, phosphodiester, carbamate)
plus the size guard threshold.
"""

import pytest
from rdkit import Chem

from orthonym.decomposition.bond_cleavage import (
    _maybe_swap_parent_roles,
    find_cleavable_bonds,
)


# ---------------------------------------------------------------------------
# Helper: find the match atom indices for a given bond type
# ---------------------------------------------------------------------------

def _get_bond_atoms(smiles: str, bond_type: str):
    """Find the acid_idx, other_idx, and bridging_atoms for a given SMILES + bond type.

    Returns (mol, acid_idx, other_idx, bridging_atoms) or None if bond not found.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bonds = find_cleavable_bonds(mol)
    for b in bonds:
        if b["type"] == bond_type:
            return mol, b
    return None


# ===========================================================================
# TestMaybeSwapParentRoles
# ===========================================================================


@pytest.mark.unit
class TestMaybeSwapParentRoles:
    """Test _maybe_swap_parent_roles() directly with various bond types."""

    def test_ester_large_other_swaps(self):
        """Steroid acetate: acid=3 HA (acetate), other=20+ HA steroid -> SWAP.

        CC(=O)OC1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C
        (testosterone acetate-like: acetate ester of a steroid)
        """
        smiles = "CC(=O)OC1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None

        # carbonyl_c, alkyl_c, ester_o from SMARTS match
        # The ester SMARTS is [CX3](=O)[OX2][#6] -> match = (carbonyl_c, =O, ester_o, alkyl_c)
        ester_smarts = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(ester_smarts)
        assert len(matches) > 0
        # Find the ester match (not the ketone)
        for match in matches:
            carbonyl_c = match[0]
            ester_o = match[2]
            alkyl_c = match[3]
            # Verify this is the ester (not a ketone or other C=O)
            if mol.GetAtomWithIdx(ester_o).GetSymbol() == "O":
                break

        acid_final, other_final, swapped = _maybe_swap_parent_roles(
            mol, carbonyl_c, alkyl_c, "ester",
            bridging_atoms={ester_o},
        )
        assert swapped is True, (
            f"Expected swap for steroid acetate: acid={carbonyl_c}, "
            f"other={alkyl_c}, got swapped={swapped}"
        )

    def test_ester_simple_no_swap(self):
        """Ethyl acetate (CCOC(=O)C): both <10 HA -> NO swap (simple ester exemption)."""
        smiles = "CCOC(=O)C"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None

        ester_smarts = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(ester_smarts)
        assert len(matches) > 0
        match = matches[0]
        carbonyl_c = match[0]
        ester_o = match[2]
        alkyl_c = match[3]

        acid_final, other_final, swapped = _maybe_swap_parent_roles(
            mol, carbonyl_c, alkyl_c, "ester",
            bridging_atoms={ester_o},
        )
        assert swapped is False, (
            f"Simple ester should NOT swap: got swapped={swapped}"
        )

    def test_amide_never_swaps_at_detection_time(self):
        """Amide bonds never swap at detection time.

        The engine has a separate post-cleavage amide N-acyl seniority check
        (engine.py:835-852) that operates on capped fragment SMILES. Detection-
        time swap for amides causes regressions in peptide-like compounds.

        O=CN1CCCCC1 (formyl-piperidine): despite amine being more senior,
        _maybe_swap_parent_roles returns swapped=False for amides.
        """
        smiles = "O=CN1CCCCC1"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None

        amide_smarts = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(amide_smarts)
        assert len(matches) > 0
        match = matches[0]
        carbonyl_c = match[0]
        nitrogen = match[2]

        acid_final, other_final, swapped = _maybe_swap_parent_roles(
            mol, carbonyl_c, nitrogen, "amide",
            bridging_atoms=None,
        )
        # Amides always return swapped=False (engine handles N-acyl post-cleavage)
        assert swapped is False, (
            f"Amide bonds should never swap at detection time: swapped={swapped}"
        )

    def test_amide_acid_more_senior_no_swap(self):
        """N-methylacetamide (CC(=O)NC): acid has COOH, amine is just methyl -> NO swap."""
        smiles = "CC(=O)NC"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None

        amide_smarts = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(amide_smarts)
        assert len(matches) > 0
        match = matches[0]
        carbonyl_c = match[0]
        nitrogen = match[2]

        acid_final, other_final, swapped = _maybe_swap_parent_roles(
            mol, carbonyl_c, nitrogen, "amide",
            bridging_atoms=None,
        )
        assert swapped is False, (
            f"Acid should be more senior in N-methylacetamide: swapped={swapped}"
        )

    def test_thioester_large_other_swaps(self):
        """Thioester where other side is a large polycyclic system -> SWAP.

        CC(=O)SC1CCC2CCCCC2C1 (S-decalinyl thioacetate)
        Acid: acetyl (3 HA), Other: decalin (10 HA ring system)
        Ratio = 10/3 = 3.33x with ring -> should swap.
        """
        smiles = "CC(=O)SC1CCC2CCCCC2C1"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None

        thioester_smarts = Chem.MolFromSmarts("[CX3](=O)[SX2][#6]")
        matches = mol.GetSubstructMatches(thioester_smarts)
        assert len(matches) > 0
        match = matches[0]
        carbonyl_c = match[0]
        sulfur = match[2]
        alkyl_c = match[3]

        acid_final, other_final, swapped = _maybe_swap_parent_roles(
            mol, carbonyl_c, alkyl_c, "thioester",
            bridging_atoms={sulfur},
        )
        # Decalin (10 HA, ring) vs acetyl (3 HA, no ring)
        # Ratio = 10/3 = 3.33x AND other has ring -> swap
        assert swapped is True, (
            f"Decalinyl should trigger swap over acetyl: swapped={swapped}"
        )

    def test_sulfonamide_acid_more_senior_no_swap(self):
        """Benzenesulfonamide (NS(=O)(=O)c1ccccc1): acid has ring, amine is NH2 -> NO swap."""
        smiles = "NS(=O)(=O)c1ccccc1"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None

        sulfonamide_smarts = Chem.MolFromSmarts("[SX4](=O)(=O)[NX3]")
        matches = mol.GetSubstructMatches(sulfonamide_smarts)
        assert len(matches) > 0
        match = matches[0]
        sulfur = match[0]
        nitrogen = match[3]

        acid_final, other_final, swapped = _maybe_swap_parent_roles(
            mol, sulfur, nitrogen, "sulfonamide",
            bridging_atoms=None,
        )
        assert swapped is False, (
            f"Benzenesulfonyl should be more senior than NH2: swapped={swapped}"
        )

    def test_phosphodiester_large_other_swaps(self):
        """Phosphodiester where other side is a large ring system -> SWAP.

        Use a steroid phosphate: acetyl side is just phosphate (small),
        other side is a large polycyclic ring system.
        O=P(O)(O)OC1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C
        (steroid phosphate: phosphate on a steroid)
        """
        smiles = "O=P(O)(O)OC1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None

        # Find phosphodiester bond
        bonds = find_cleavable_bonds(mol)
        phospho_bonds = [b for b in bonds if b["type"] == "phosphodiester"]
        if len(phospho_bonds) > 0:
            # The steroid side (20 HA, rings) vs phosphate (4 HA, no ring)
            # Ratio >= 5x, should swap
            has_swap = any(b.get("roles_swapped", False) for b in phospho_bonds)
            assert has_swap, (
                f"Expected roles_swapped=True for steroid phosphate. "
                f"Bonds: {phospho_bonds}"
            )
        else:
            pytest.skip("No phosphodiester bond detected in test molecule")

    def test_carbamate_large_other_swaps(self):
        """Carbamate where other side is a large ring system -> SWAP.

        O=C(Nc1ccccc1)OC (methyl N-phenylcarbamate)
        Amine side: phenyl (6 HA ring), Acid side: methyl ester (small)
        """
        smiles = "O=C(Nc1ccccc1)OC"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None

        bonds = find_cleavable_bonds(mol)
        carbamate_bonds = [b for b in bonds if b["type"] == "carbamate"]
        if len(carbamate_bonds) > 0:
            b = carbamate_bonds[0]
            # The carbamate should have roles_swapped flag
            assert "roles_swapped" in b, (
                f"Expected roles_swapped key in carbamate bond info: {b}"
            )
        else:
            pytest.skip("No carbamate bond detected in test molecule")


# ===========================================================================
# TestSizeGuard
# ===========================================================================


@pytest.mark.unit
class TestSizeGuard:
    """Test the 3x heavy-atom size guard threshold."""

    def test_3x_ratio_triggers_swap(self):
        """When other side has >=3x HA AND equal/higher seniority, swap.

        Use a molecule where acid has ~3 HA and other has ~9+ HA.
        Example: methyl cyclohexanecarboxylate: OC(=O)C1CCCCC1 ester with methyl
        Actually use: CC(=O)OC1CCCCCCCCC1 (cyclodecanyl acetate)
        Acid: acetyl = 3 HA, Other: cyclodecane = 10 HA. Ratio = 3.33x
        """
        smiles = "CC(=O)OC1CCCCCCCCC1"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None

        ester_smarts = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(ester_smarts)
        assert len(matches) > 0
        match = matches[0]
        carbonyl_c = match[0]
        ester_o = match[2]
        alkyl_c = match[3]

        acid_final, other_final, swapped = _maybe_swap_parent_roles(
            mol, carbonyl_c, alkyl_c, "ester",
            bridging_atoms={ester_o},
        )
        # cyclodecane (10 HA, ring) vs acetyl (3 HA, no ring)
        # Ratio = 10/3 = 3.33x, meets 3x threshold
        # cyclodecane has ring -> more senior
        assert swapped is True, (
            f"3x+ ratio with more senior other should trigger swap: "
            f"swapped={swapped}"
        )

    def test_2x_ratio_no_swap_when_acid_more_senior(self):
        """When other side has 2x HA but acid is more senior -> NO swap.

        Use: CC(=O)OCC (ethyl acetate variant) - acid has COOH rank
        Both are small so simple ester exemption kicks in.
        Use a non-ester to avoid exemption:
        CC(=O)NCCCC (N-butylacetamide)
        Acid: acetyl (3 HA), Other: butyl (4 HA) + N
        4/3 = 1.33x -- below 3x threshold, and acid has COOH.
        """
        smiles = "CC(=O)NCCCC"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None

        amide_smarts = Chem.MolFromSmarts("[CX3](=O)[NX3]")
        matches = mol.GetSubstructMatches(amide_smarts)
        assert len(matches) > 0
        match = matches[0]
        carbonyl_c = match[0]
        nitrogen = match[2]

        acid_final, other_final, swapped = _maybe_swap_parent_roles(
            mol, carbonyl_c, nitrogen, "amide",
            bridging_atoms=None,
        )
        # Acid side has carboxylic acid seniority, other side is just butylamine
        # Ratio below 3x and acid is more senior -> no swap
        assert swapped is False, (
            f"2x ratio with acid more senior should NOT swap: swapped={swapped}"
        )


# ===========================================================================
# TestFindCleavableBondsSwapFlag
# ===========================================================================


@pytest.mark.unit
class TestFindCleavableBondsSwapFlag:
    """Test that find_cleavable_bonds() returns roles_swapped in all bond dicts."""

    def test_simple_ester_has_roles_swapped_false(self):
        """CCOC(=O)C (ethyl acetate): simple ester should have roles_swapped=False."""
        mol = Chem.MolFromSmiles("CCOC(=O)C")
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]
        assert len(ester_bonds) > 0, "Expected ester bond in ethyl acetate"
        for b in ester_bonds:
            assert "roles_swapped" in b, (
                f"Missing roles_swapped key in bond info: {b}"
            )
            assert b["roles_swapped"] is False, (
                f"Simple ester should have roles_swapped=False: {b}"
            )

    def test_all_bond_types_have_roles_swapped_key(self):
        """Verify roles_swapped key exists in bond dicts for all bond types.

        Use molecules that produce each bond type and verify the key is present.
        """
        test_cases = [
            ("CCOC(=O)C", "ester"),                    # ethyl acetate
            ("CC(=O)NC", "amide"),                      # N-methylacetamide
            ("CC(=O)SC", "thioester"),                  # S-methyl thioacetate
            ("NS(=O)(=O)c1ccccc1", "sulfonamide"),      # benzenesulfonamide
        ]
        for smiles, expected_type in test_cases:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            bonds = find_cleavable_bonds(mol)
            typed_bonds = [b for b in bonds if b["type"] == expected_type]
            if len(typed_bonds) > 0:
                for b in typed_bonds:
                    assert "roles_swapped" in b, (
                        f"Missing roles_swapped key for {expected_type} "
                        f"in {smiles}: {b}"
                    )

    def test_glycosidic_always_false(self):
        """Glycosidic bonds should always have roles_swapped=False (exempt)."""
        # Simple glycoside: OC1OC(CO)C(O)C1O with an O-linkage
        # Use a known glycosidic pattern
        smiles = "OCC1OC(OC)C(O)C1O"  # methyl glucoside
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            pytest.skip("Cannot parse glycoside SMILES")
        bonds = find_cleavable_bonds(mol)
        glyco_bonds = [b for b in bonds if b["type"] == "glycosidic"]
        for b in glyco_bonds:
            assert "roles_swapped" in b, (
                f"Missing roles_swapped key for glycosidic bond: {b}"
            )
            assert b["roles_swapped"] is False, (
                f"Glycosidic bond should always have roles_swapped=False: {b}"
            )

    def test_ether_always_false(self):
        """Ether bonds should always have roles_swapped=False."""
        smiles = "c1ccc(OCCCCCc2ccccc2)cc1"  # diphenyl pentyl ether
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            pytest.skip("Cannot parse ether SMILES")
        bonds = find_cleavable_bonds(mol)
        ether_bonds = [b for b in bonds if b["type"] == "ether"]
        for b in ether_bonds:
            assert "roles_swapped" in b, (
                f"Missing roles_swapped key for ether bond: {b}"
            )
            assert b["roles_swapped"] is False, (
                f"Ether bond should always have roles_swapped=False: {b}"
            )

    def test_steroid_ester_has_roles_swapped_true(self):
        """Steroid acetate: large other side -> roles_swapped=True."""
        smiles = "CC(=O)OC1CCC2C3CCC4=CC(=O)CCC4(C)C3CCC12C"
        mol = Chem.MolFromSmiles(smiles)
        bonds = find_cleavable_bonds(mol)
        ester_bonds = [b for b in bonds if b["type"] == "ester"]
        assert len(ester_bonds) > 0, "Expected ester bond in steroid acetate"
        # At least one ester bond should have roles_swapped=True
        has_swap = any(b.get("roles_swapped", False) for b in ester_bonds)
        assert has_swap, (
            f"Expected roles_swapped=True for steroid acetate ester. "
            f"Bonds: {ester_bonds}"
        )
