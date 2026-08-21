"""Tests for amino acid multi-group naming format (NP-03 / 30-02).

Covers:
- Simple amino acids (regression guards for trivial name lookup)
- Multi-amine format (the core NP-03 fix: no "2-diamino" malformation)
- Geminal amine cases (two NH2 on same carbon)
- Diacid amino acids (two COOH groups)
- Peptide guard (peptides should NOT be named as simple amino acids)
- Format validation: locants must precede multiplier correctly
"""
import pytest
from orthonym import name_compound


class TestSimpleAminoAcidRegression:
    """Regression guards: simple amino acids must stay correct."""

    @pytest.mark.integration
    def test_alanine_systematic(self):
        """Alanine (UNDEFINED stereo): 2-aminopropanoic acid.

        v33 Phase-1 stereo honesty (748bf56d): the retained name 'alanine'
        implies the L enantiomer (P-101.2.6 + P-103.1.3.1), so an
        undefined-stereo input honestly declines it and emits the systematic
        name. OPSIN round-trip verified."""
        result = name_compound("NC(C)C(=O)O")
        assert result == "2-aminopropanoic acid"

    @pytest.mark.integration
    def test_glycine(self):
        """Glycine: simplest amino acid."""
        assert name_compound("NCC(=O)O") == "glycine"

    @pytest.mark.integration
    def test_ornithine(self):
        """Ornithine (UNDEFINED stereo): 2,5-diaminopentanoic acid.

        Same stereo-honesty rule as test_alanine_systematic: bare 'ornithine'
        implies L, so undefined stereo declines to the systematic name.
        OPSIN round-trip verified."""
        result = name_compound("NCCCC(N)C(=O)O")
        assert result == "2,5-diaminopentanoic acid"

    @pytest.mark.integration
    def test_lysine(self):
        """Lysine (UNDEFINED stereo): 2,6-diaminohexanoic acid.

        Same stereo-honesty rule as test_alanine_systematic: bare 'lysine'
        implies L, so undefined stereo declines to the systematic name.
        OPSIN round-trip verified."""
        result = name_compound("NCCCCC(N)C(=O)O")
        assert result == "2,6-diaminohexanoic acid"


class TestMultiAmineFormat:
    """Core NP-03 fix: multi-amine amino acids must have correct locant format."""

    @pytest.mark.integration
    def test_no_2_diamino_malformation(self):
        """NCCC(N)(CC(=O)O)C(=O)O must NOT produce '2-diamino' malformation."""
        result = name_compound("NCCC(N)(CC(=O)O)C(=O)O")
        # Must not have a single locant with 'di' multiplier
        assert "2-diamino" not in result or "2,2-diamino" in result, (
            f"Malformed '2-diamino' found in: {result}"
        )

    @pytest.mark.integration
    def test_aminoethyl_substituent_named(self):
        """NCCC(N)(CC(=O)O)C(=O)O should name the aminoethyl branch correctly."""
        result = name_compound("NCCC(N)(CC(=O)O)C(=O)O")
        # The aminoethyl branch should appear as (2-aminoethyl) or similar
        assert "amino" in result, f"Missing 'amino' in: {result}"
        # Should have butanedioic acid as the base (4-carbon diacid)
        assert "butanedioic acid" in result, f"Missing 'butanedioic acid' in: {result}"

    @pytest.mark.integration
    def test_geminal_diamine_ethanoic(self):
        """NC(N)C(=O)O: geminal diamine -> 2,2-diaminoethanoic acid."""
        result = name_compound("NC(N)C(=O)O")
        assert result == "2,2-diaminoethanoic acid"

    @pytest.mark.integration
    def test_geminal_diamine_diacid(self):
        """NC(CC(=O)O)(C(=O)O)N: geminal diamine diacid -> 2,2-diaminobutanedioic acid."""
        result = name_compound("NC(CC(=O)O)(C(=O)O)N")
        assert result == "2,2-diaminobutanedioic acid"

    @pytest.mark.integration
    def test_separated_diamine_butanoic(self):
        """NCC(N)CC(=O)O: separated diamine -> 3,4-diaminobutanoic acid."""
        result = name_compound("NCC(N)CC(=O)O")
        assert result == "3,4-diaminobutanoic acid"

    @pytest.mark.integration
    def test_multi_amine_diacid(self):
        """NC(CC(N)C(=O)O)C(=O)O: diamine diacid -> 2,4-diaminopentanedioic acid."""
        result = name_compound("NC(CC(N)C(=O)O)C(=O)O")
        assert result == "2,4-diaminopentanedioic acid"


class TestPeptideGuard:
    """Peptides must NOT be named as simple amino acids."""

    @pytest.mark.integration
    def test_dipeptide_not_simple_amino_acid(self):
        """Dipeptide NC(C)C(=O)NC(CC)C(=O)O should not be flattened."""
        result = name_compound("NC(C)C(=O)NC(CC)C(=O)O")
        # Should NOT be a simple amino acid name like "2-aminoXXXanoic acid"
        assert result != "2-aminopentanoic acid"
        # Peptides go through the general pipeline or peptide handler
        assert result is not None
        assert len(result) > 0


class TestFormatValidation:
    """Validate that multi-amine outputs have correct locant-multiplier format."""

    @pytest.mark.integration
    def test_locants_match_multiplier_count(self):
        """When 'di' appears, there must be exactly 2 locants before it."""
        result = name_compound("NCC(N)CC(=O)O")
        # Expected: "3,4-diaminobutanoic acid"
        # "3,4-di" has 2 locants matching "di"
        if "diamino" in result:
            # Extract locants before "diamino"
            idx = result.index("diamino")
            prefix = result[:idx]
            # Count comma-separated locants in the last segment
            parts = prefix.rsplit("-", 1)
            if len(parts) == 2:
                locant_str = parts[0].split("-")[-1]
                locant_count = len(locant_str.split(","))
                assert locant_count == 2, (
                    f"Expected 2 locants for 'di', got {locant_count} in: {result}"
                )

    @pytest.mark.integration
    def test_no_single_locant_with_di_multiplier(self):
        """No compound should have pattern 'N-diamino' where N is a single locant."""
        import re
        test_cases = [
            "NCCC(N)(CC(=O)O)C(=O)O",
            "NC(N)C(=O)O",
            "NCC(N)CC(=O)O",
        ]
        pattern = re.compile(r"(\d)-diamino")
        for smiles in test_cases:
            result = name_compound(smiles)
            match = pattern.search(result)
            if match:
                # Only valid if the match is actually "N,N-diamino" (geminal)
                full_locant = result[:match.end()].rsplit("-", 1)[0]
                if "," not in full_locant.split("-")[-1]:
                    pytest.fail(
                        f"Single-locant 'diamino' in {smiles} -> {result}"
                    )
