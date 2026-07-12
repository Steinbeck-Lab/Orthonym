"""Tests for ester orientation guard (ASML-16).

ASML-16: When both carbonyl C and ester O are on the principal chain,
_get_alkoxycarbonyl_prefix should return None (ester is backbone, not substituent).
"""

import pytest
from rdkit import Chem


# ============================================================================
# ASML-16: Both-ends-on-chain guard returns None
# ============================================================================

@pytest.mark.unit
class TestEsterOrientationGuard:
    """ASML-16: Both-ends-on-chain guard returns None."""

    def test_both_ends_on_chain_returns_none(self):
        """When carbonyl C and ester O are both on principal chain, return None.

        For O=COCCCC:
          idx 0 = O (carbonyl oxygen)
          idx 1 = C (carbonyl carbon)
          idx 2 = O (ester oxygen)
          idx 3,4,5,6 = C chain

        ester_atoms = (1, 0, 2, 3)  -- (carbonyl_C, carbonyl_O, ester_O, alkyl_C)
        principal_chain includes both C(1) and O(2).
        """
        from orthonym.rules.polyfunctional import _get_alkoxycarbonyl_prefix

        mol = Chem.MolFromSmiles("O=COCCCC")
        assert mol is not None
        # ester_atoms: (carbonyl_C, carbonyl_O, ester_O, alkyl_C)
        ester_atoms = (1, 0, 2, 3)
        # Both carbonyl C(1) and ester O(2) on principal chain
        principal_chain = [1, 2, 3, 4, 5]
        result = _get_alkoxycarbonyl_prefix(mol, ester_atoms, principal_chain)
        assert result is None, f"Expected None when both ends on chain, got: {result}"

    def test_normal_ester_c_on_chain_o_off(self):
        """In-chain ester carbonyl -> None per P-65.6.3.3.5 (was the pre-2026 alkoxycarbonyl case).

        Ethyl propanoate: CCC(=O)OCC
        The carbonyl C is on the principal chain but the ester O is not.
        This is the standard alkoxycarbonyl case.
        """
        from orthonym.rules.polyfunctional import _get_alkoxycarbonyl_prefix

        mol = Chem.MolFromSmiles("CCC(=O)OCC")
        assert mol is not None
        # Find ester atoms via SMARTS
        pat = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
        matches = mol.GetSubstructMatches(pat)
        assert len(matches) > 0
        ester_atoms = matches[0]
        # Principal chain includes the carbonyl carbon but NOT the ester O
        carbonyl_c = ester_atoms[0]
        principal_chain = [0, 1, carbonyl_c]  # chain through C-C-C(=O)
        result = _get_alkoxycarbonyl_prefix(mol, ester_atoms, principal_chain)
        # W2F-P2 (P-65.6.3.3.5 method (1)): a CHAIN-MEMBER ester carbonyl is
        # never R-oxycarbonyl; it decomposes to oxo + alkoxy. This test was
        # previously assertion-free; it now pins the None contract.
        assert result is None, f"in-chain ester C must return None, got {result!r}"
