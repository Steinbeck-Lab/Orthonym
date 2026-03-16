"""Tests for DATA-02: broadened secondary/tertiary amine SMARTS.

The new SMARTS patterns use [#6] instead of [CX4] to allow aromatic carbon
neighbors (e.g., N-methylaniline), while excluding amides and guanidines
via !$([NX3][CX3]=O) and !$([NX3][CX3]=[NX2]).
"""
import pytest
from rdkit import Chem
from orthonym.perception.functional_groups import detect_functional_groups


def _detect(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    return detect_functional_groups(mol)


class TestAmineSmartsFix:
    """Verify broadened amine SMARTS detect N-aryl amines without false positives."""

    def test_n_methylaniline_secondary(self):
        """N-methylaniline (CNc1ccccc1) should be detected as secondary_amine."""
        fgs = _detect("CNc1ccccc1")
        assert "secondary_amine" in fgs

    def test_nn_dimethylaniline_tertiary(self):
        """N,N-dimethylaniline (CN(C)c1ccccc1) should be detected as tertiary_amine."""
        fgs = _detect("CN(C)c1ccccc1")
        assert "tertiary_amine" in fgs

    def test_n_methylacetamide_not_amine(self):
        """N-methylacetamide (CNC(C)=O) should NOT be detected as secondary_amine (amide exclusion)."""
        fgs = _detect("CNC(C)=O")
        assert "secondary_amine" not in fgs

    def test_diethylamine_still_works(self):
        """Diethylamine (CCNCC) should still be detected as secondary_amine (sp3 regression check)."""
        fgs = _detect("CCNCC")
        assert "secondary_amine" in fgs

    def test_triethylamine_still_works(self):
        """Triethylamine (CCN(CC)CC) should still be detected as tertiary_amine (sp3 regression check)."""
        fgs = _detect("CCN(CC)CC")
        assert "tertiary_amine" in fgs

    def test_aniline_not_secondary(self):
        """Aniline (Nc1ccccc1) should be aromatic_amine, NOT secondary_amine."""
        fgs = _detect("Nc1ccccc1")
        assert "secondary_amine" not in fgs
        assert "aromatic_amine" in fgs

    def test_guanidine_not_amine(self):
        """Guanidine (NC(=N)N) should NOT be detected as secondary_amine (guanidine exclusion)."""
        fgs = _detect("NC(=N)N")
        assert "secondary_amine" not in fgs
