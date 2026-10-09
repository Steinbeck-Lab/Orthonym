"""Tests for: Dicarboxylic amino acid bailout to polyfunctional pipeline.

Verifies that amino acids with 2+ carboxylic acid groups (aspartic acid,
glutamic acid) bail out of the specialized amino acid handler and produce
correct "dioic acid" names via the polyfunctional pipeline.
"""
import pytest
from orthonym import name_compound


@pytest.mark.unit
class TestDicarboxylicAminoAcidBailout:
    """: Dicarboxylic amino acids should produce 'dioic' names."""

    def test_aspartic_acid_nonstereo(self):
        """Non-stereo (alpha-carbon CHI_UNSPECIFIED) aspartic acid now DEFERS to
        the systematic name rather than the retained name, which asserts a
        defined (L) configuration this input lacks.

         a phase (change-asserted-value, was `== "aspartic acid"`): see
        `test_amino_acids.py`'s module docstring for the full /
        InChIKey / mutation-test evidence -- identical argument, same input.
        """
        result = name_compound("NC(CC(=O)O)C(=O)O")
        assert result == "aminobutanedioic acid"

    def test_glutamic_acid_nonstereo(self):
        """Non-stereo glutamic acid now DEFERS to the systematic name (T5, see
        `test_aspartic_acid_nonstereo`'s docstring)."""
        result = name_compound("NC(CCC(=O)O)C(=O)O")
        assert result == "2-aminopentanedioic acid"

    def test_aspartic_acid_stereo(self):
        """-07 (a phase): a stereo-tagged free aspartic acid resolves to the
        retained name carrying its configurational descriptor, 'L-aspartic acid'
        (OPSIN-round-trip-verified), not the old systematic 'aminobutanedioic acid'
        bailout. The descriptor is cited since a6cadc255 ("a free amino acid keeps its L": the
        namer used to drop the 'L-' that it kept for 'D-' and so emitted a bare name
        for either form of a defined-configuration input): "The stereodescriptors 'D' and
        'L'" (the Blue Book) -- "The absolute configuration at the α-carbon
        atom of the α-amino carboxylic acids is designated by the stereodescriptor
        'D' or 'L'" -- and the Blue Book itself writes 'L-glutamic acid' inside a
        name (:54506)."""
        result = name_compound("N[C@@H](CC(=O)O)C(=O)O")
        assert result == "L-aspartic acid", f"Expected 'L-aspartic acid', got: {result}"

    def test_glutamic_acid_stereo(self):
        """-07 (a phase): a stereo-tagged free glutamic acid resolves to the
        retained name with its configurational descriptor, 'L-glutamic acid'
        (OPSIN-round-trip-verified;, see `test_aspartic_acid_stereo`),
        not the old systematic 'aminopentanedioic acid' bailout."""
        result = name_compound("N[C@@H](CCC(=O)O)C(=O)O")
        assert result == "L-glutamic acid", f"Expected 'L-glutamic acid', got: {result}"

    def test_mono_cooh_amino_acid_unchanged(self):
        """Mono-COOH amino acid should NOT bail out -- regression guard.
        After 141-02 expansion, this compound is recognized as 'butyrine'
        (OPSIN simpleGroup entry), which is the correct trivial name."""
        result = name_compound("NC(CC)C(=O)O")
        assert result in ("2-aminobutanoic acid", "butyrine"), (
            f"Expected systematic or trivial name, got: {result}"
        )

    def test_alanine_unchanged(self):
        """Alanine (1 COOH), non-stereo -- T5: now DEFERS to the systematic name
        (see `test_aspartic_acid_nonstereo`'s docstring); the mono-COOH bailout
        guard itself is unaffected (still not misrouted to the dicarboxylic path)."""
        result = name_compound("NC(C)C(=O)O")
        assert result == "2-aminopropanoic acid"


@pytest.mark.unit
def test_n_carboxymethyl_aspartic_counts_only_on_chain_acids():
    """: a COOH inside a SUBSTITUENT must not inflate the parent acid-suffix
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
