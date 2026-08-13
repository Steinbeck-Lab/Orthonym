"""Tests for AMAC-01: Dicarboxylic amino acid bailout to polyfunctional pipeline.

Verifies that amino acids with 2+ carboxylic acid groups (aspartic acid,
glutamic acid) bail out of the specialized amino acid handler and produce
correct "dioic acid" names via the polyfunctional pipeline.
"""
import pytest
from orthonym import name_compound


@pytest.mark.unit
class TestDicarboxylicAminoAcidBailout:
    """AMAC-01: Dicarboxylic amino acids should produce 'dioic' names."""

    def test_aspartic_acid_nonstereo(self):
        """Non-stereo aspartic acid should still use retained name."""
        result = name_compound("NC(CC(=O)O)C(=O)O")
        assert result == "aspartic acid"

    def test_glutamic_acid_nonstereo(self):
        """Non-stereo glutamic acid should still use retained name."""
        result = name_compound("NC(CCC(=O)O)C(=O)O")
        assert result == "glutamic acid"

    def test_aspartic_acid_stereo(self):
        """WSD-07 (Phase 175): a stereo-tagged free aspartic acid resolves to the
        retained PIN 'aspartic acid' (OPSIN-round-trip-verified), not the old
        systematic 'aminobutanedioic acid' bailout (standard AAs use the retained
        name with the configurational descriptor, P-103.1.1.1)."""
        result = name_compound("N[C@@H](CC(=O)O)C(=O)O")
        assert result == "aspartic acid", f"Expected 'aspartic acid', got: {result}"

    def test_glutamic_acid_stereo(self):
        """WSD-07 (Phase 175): a stereo-tagged free glutamic acid resolves to the
        retained PIN 'glutamic acid' (OPSIN-round-trip-verified), not the old
        systematic 'aminopentanedioic acid' bailout."""
        result = name_compound("N[C@@H](CCC(=O)O)C(=O)O")
        assert result == "glutamic acid", f"Expected 'glutamic acid', got: {result}"

    def test_mono_cooh_amino_acid_unchanged(self):
        """Mono-COOH amino acid should NOT bail out -- regression guard.
        After 141-02 expansion, this compound is recognized as 'butyrine'
        (OPSIN simpleGroup entry), which is the correct trivial name."""
        result = name_compound("NC(CC)C(=O)O")
        assert result in ("2-aminobutanoic acid", "butyrine"), (
            f"Expected systematic or trivial name, got: {result}"
        )

    def test_alanine_unchanged(self):
        """Alanine (1 COOH) should still produce 'alanine' -- regression guard."""
        result = name_compound("NC(C)C(=O)O")
        assert result == "alanine"


@pytest.mark.unit
def test_n_carboxymethyl_aspartic_counts_only_on_chain_acids():
    """v30: a COOH inside a SUBSTITUENT must not inflate the parent acid-suffix
    multiplicity. N-(carboxymethyl)aspartic acid's third COOH is the carboxymethyl
    group, so the butanedioic parent stays 'dioic' -- it was mis-built as
    'butanetrioic' (three -oic on a two-acid parent) and suppressed as unparseable.
    """
    assert name_compound("OC(=O)CNC(CC(=O)O)C(=O)O") == \
        "2-[(carboxymethyl)amino]butanedioic acid"
    # a GENUINE tricarboxylic acid (all three C on/attached to the chain) is
    # unaffected -- still the carboxylic-acid suffix form, not a broken 'trioic'.
    assert name_compound("OC(=O)C(CC(=O)O)CC(=O)O") == \
        "propane-1,2,3-tricarboxylic acid"
