"""Tests for branched acid ester acyloxy prefix naming (IUPAC P-65.6.3.2.2).

Ensures that esters with branched acid fragments use the principal chain
length (not total carbon count) for the acid stem, and include branch
substituent prefixes in the acyloxy name.
"""
import pytest
from rdkit import Chem

from orthonym.rules.esters import name_ester_as_prefix
from orthonym.perception.functional_groups import detect_functional_groups


def _get_acyloxy_prefix(smiles: str) -> str:
    """Helper: get acyloxy prefix from first ester in molecule."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    fgs = detect_functional_groups(mol)
    esters = fgs.get("ester", [])
    assert esters, f"No ester found in {smiles}"
    return name_ester_as_prefix(mol, esters[0])


class TestBranchedAcidAcyloxy:
    """IUPAC P-65.6.3.2.2: branched acid -> branched acyloxy prefix."""

    def test_isobutyrate_uses_principal_chain(self):
        """Isobutyric acid: 3C principal chain + 1 methyl branch.

        CC(C)C(=O)O... has 4 total carbons but principal chain = 3C,
        so the acid stem should be "propanoic" (not "butanoic").
        """
        # Phenyl isobutyrate: CC(C)C(=O)Oc1ccccc1
        result = _get_acyloxy_prefix("CC(C)C(=O)Oc1ccccc1")
        assert result is not None
        # Should contain "propanoyl" (3C chain), not "butanoyl" (4C total)
        assert "propanoyl" in result, f"Expected propanoyl in {result}"
        assert "methyl" in result, f"Expected methyl branch in {result}"

    def test_simple_acetate_unchanged(self):
        """Acetic acid: 2C, no branch -> 'acetyloxy' (trivial)."""
        result = _get_acyloxy_prefix("CC(=O)Oc1ccccc1")
        assert result == "acetyloxy"

    def test_simple_propanoate_unchanged(self):
        """Propanoic acid: 3C, no branch -> 'propanoyloxy'."""
        result = _get_acyloxy_prefix("CCC(=O)Oc1ccccc1")
        assert result == "propanoyloxy"

    def test_linear_butanoate_unchanged(self):
        """Butanoic acid: 4C linear, no branch -> 'butanoyloxy'."""
        result = _get_acyloxy_prefix("CCCC(=O)Oc1ccccc1")
        assert result == "butanoyloxy"

    def test_trivial_fatty_acid_preserved(self):
        """Palmitic acid ester: 16C linear -> 'palmitoyloxy' (trivial name).

        The branched naming path should NOT override trivial acid names.
        """
        result = _get_acyloxy_prefix("CCCCCCCCCCCCCCCC(=O)OC")
        assert result is not None
        assert "palmitoyloxy" in result, f"Expected palmitoyloxy in {result}"

    def test_isovalerate_branched(self):
        """Isovaleric acid: 3-methylbutanoic acid (4C chain + 1 branch).

        CC(C)CC(=O)O... has 5 total C but principal chain = 4C.
        """
        result = _get_acyloxy_prefix("CC(C)CC(=O)OC")
        assert result is not None
        assert "butanoyl" in result, f"Expected butanoyl (4C chain) in {result}"
