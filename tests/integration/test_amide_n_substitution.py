"""
Tests for N-substituted amide naming patterns (Phase 29, Plan 01).

Covers:
  AM-01: (phenylamino) format in general substituent path
  AM-02: Simple amide N-substitution regression guard
  AM-03: Peptide aromatic residue format verification
  Regression guard: ensures (anilino) never appears in output
"""

import pytest
from orthonym import name_compound


# ---------------------------------------------------------------------------
# Section 1: N-phenyl amide naming (AM-01 verification)
# ---------------------------------------------------------------------------


class TestNPhenylAmideNaming:
    """Verify (phenylamino) format is produced in general substituent path."""

    @pytest.mark.integration
    def test_n_phenylacetamide_via_amide_path(self):
        """Simple amide path: CC(=O)Nc1ccccc1 -> N-phenylacetamide."""
        result = name_compound("CC(=O)Nc1ccccc1")
        assert result == "N-phenylacetamide"

    @pytest.mark.integration
    def test_phenylamino_in_pentanedioic_acid(self):
        """General substituent path should produce (phenylamino), not (anilino)."""
        smiles = "NC(CCC(=O)NC(CSC(CC=O)c1ccccc1O)C(=O)NCC(=O)O)C(=O)O"
        result = name_compound(smiles)
        assert "phenylamino" in result
        assert "anilino" not in result

    @pytest.mark.integration
    def test_sulfonamide_no_anilino(self):
        """Sulfonamide compound should not produce (anilino) in name."""
        result = name_compound("CC(=O)Nc1ccc(S(=O)(=O)NC(C)=O)cc1")
        assert "anilino" not in result


# ---------------------------------------------------------------------------
# Section 2: Simple amide N-substituent naming (AM-02 regression guard)
# ---------------------------------------------------------------------------


class TestSimpleAmideNSubstitution:
    """Verify existing amide N-substitution is not broken."""

    @pytest.mark.integration
    def test_n_phenylacetamide_exact(self):
        """N-phenylacetamide exact match."""
        result = name_compound("CC(=O)Nc1ccccc1")
        assert result == "N-phenylacetamide"

    @pytest.mark.integration
    def test_n_ethylacetamide(self):
        """N-ethylacetamide contains N-ethyl."""
        result = name_compound("CC(=O)NCC")
        assert "N-ethyl" in result

    @pytest.mark.integration
    def test_nn_dimethylformamide(self):
        """N,N-dimethylformamide contains dimethyl."""
        result = name_compound("CN(C)C=O")
        assert "dimethyl" in result

    @pytest.mark.integration
    def test_n_benzylacetamide(self):
        """N-benzylacetamide contains benzyl."""
        result = name_compound("CC(=O)NCc1ccccc1")
        assert "benzyl" in result


# ---------------------------------------------------------------------------
# Section 3: Peptide aromatic residue format (AM-03 verification)
# ---------------------------------------------------------------------------


class TestPeptideAromaticResidue:
    """Verify peptide compounds don't produce (anilino)."""

    @pytest.mark.integration
    def test_phenylalanine_naming(self):
        """Phenylalanine should produce a name without (anilino)."""
        result = name_compound("NC(Cc1ccccc1)C(=O)O")
        assert result is not None
        assert "anilino" not in result

    @pytest.mark.integration
    def test_benchmark_idx_216_no_anilino(self):
        """Benchmark idx=216: sulfonamide compound should not have anilino."""
        result = name_compound("CC(=O)Nc1ccc(S(=O)(=O)NC(C)=O)cc1")
        assert "anilino" not in result

    @pytest.mark.integration
    def test_benchmark_idx_270_no_anilino(self):
        """Benchmark idx=270: methoxybenzene amide should not have anilino."""
        result = name_compound("COCc1ccc(O)c(NC(C)=O)c1")
        assert "anilino" not in result


# ---------------------------------------------------------------------------
# Section 4: Guard against anilino regression
# ---------------------------------------------------------------------------


class TestAnilinoRegressionGuard:
    """Parametrized test ensuring (anilino) never appears in output."""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "smiles,test_id",
        [
            (
                "NC(CCC(=O)NC(CSC(CC=O)c1ccccc1O)C(=O)NCC(=O)O)C(=O)O",
                "pentanedioic-phenylamino",
            ),
            (
                "CC(=O)Nc1ccc(S(=O)(=O)NC(C)=O)cc1",
                "sulfonamide-phenyl",
            ),
            (
                "COCc1ccc(O)c(NC(C)=O)c1",
                "methoxybenzene-amide",
            ),
            (
                "CC(=O)Nc1ccccc1",
                "n-phenylacetamide",
            ),
            (
                "NC(Cc1ccccc1)C(=O)O",
                "phenylalanine",
            ),
            (
                "CCCCCC(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H]1C(=O)N[C@@H]"
                "(CCCN=C(N)N)C(=O)N[C@H]2CC[C@@H](O)N(C2=O)[C@@H]"
                "([C@@H](C)CC)C(=O)N(C)[C@@H](CC(=O)c2ccccc2N)C(=O)"
                "N[C@@H](C(C)C)C(=O)O[C@@H]1C",
                "complex-peptide-ci021",
            ),
        ],
        ids=lambda x: x if isinstance(x, str) and "-" in x else "",
    )
    def test_no_anilino_in_output(self, smiles, test_id):
        """No compound should ever produce (anilino) in its IUPAC name."""
        result = name_compound(smiles)
        assert result is not None, f"name_compound returned None for {test_id}"
        assert "anilino" not in result, (
            f"REGRESSION: {test_id} still produces (anilino): {result}"
        )
