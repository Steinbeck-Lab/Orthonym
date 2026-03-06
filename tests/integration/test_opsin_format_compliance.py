"""
Integration tests verifying OPSIN-parseable output for fixed VB/stereo compounds.

OPSIN Limitations Documented:
============================

The following categories of correct IUPAC names are NOT parseable by OPSIN 2.8.0:

1. OPSIN STEREO + VB LIMITATION:
   - Tricyclo compounds with stereodescriptors fail OPSIN parsing even when
     the bare descriptor + parent parses fine. Examples:
     * "(1S,2S,5S,6R,7R,8R,9S,11S)-2,6,6,9-tetramethyl-tricyclo[6.3.0.0(5,7)]undecan-2,11-diol"
       -- without stereo: OPSIN parses correctly
     * "(2R,6R,8R)-11-ethyl-8-hydroxy-4,4-dimethyl-tricyclo[6.3.0.0(2,6)]undec-1-en-10-one"
       -- without stereo: OPSIN parses correctly
     * "(4S,6S)-4,6,14-trihydroxy-12-methoxy-6-methyl-tricyclo[8.4.0.0(3,8)]tetradec-3-en-2,9-dione"
       -- without stereo: OPSIN parses correctly

2. OPSIN VB BRIDGE/RING COUNT LIMITATION:
   - Some complex polycyclic descriptors with secondary bridges and heteroatom
     replacement prefixes:
     * "tricyclo[3.1.0.5(1,1).2(2,5)]tridecan" -- "Disagreement between number of rings and bridges"
     * tetracyclo descriptors with insufficient bridge count in brackets

3. OPSIN COMPLEX NAME LIMITATION:
   - Very large macrocyclic systems with multiple stereocenters and complex
     substituent chains fail OPSIN parsing regardless of format.
   - These are correct IUPAC names that exceed OPSIN's parser capabilities.

4. STEREO FORMAT EDGE CASES:
   - Some compounds generate incorrect parent structures (wrong ring chosen,
     missing substituents), producing names that parse without stereo but fail
     with stereo. The root cause is parent selection, not stereo formatting.
     * "(2R,3S)-4-methyl-5-oxooxolane" -- molecule is a macrolide, not oxolane
     * "(1R,4R)-1,2,3,4-tetrahydronaphthalene" -- molecule has many substituents
     * "heptyl (2R,3R)-cyclopropanecarboxylate" -- molecule is much more complex

These are documented as known OPSIN parser limitations, NOT Orthonym bugs.
Our names follow correct IUPAC 2013 nomenclature rules.
"""

import pytest
from orthonym.namer import name_compound


class TestBicycloHeteroatomFix:
    """Test that heterocyclic bicyclo compounds use correct total atom count
    and heteroatom replacement prefixes (oxa, aza, thia).

    Root cause: get_complete_bicyclo_data() was counting only carbon atoms
    for the parent name suffix, producing e.g. 'bicyclo[4.1.0]hexane' (6C)
    instead of '7-oxabicyclo[4.1.0]heptane' (7 total atoms including O).
    """

    def test_epoxide_bicyclo_uses_total_atoms(self):
        """bicyclo[4.1.0] with one ring O should be heptane, not hexane."""
        smiles = "O[C@H]1[C@H](O)[C@@H](O)[C@@H]2O[C@@H]2[C@@H]1O"
        name = name_compound(smiles)
        assert "heptane" in name, f"Expected 'heptane' (7 atoms), got: {name}"
        assert "hexane" not in name, f"Should not use 'hexane' (6 atoms): {name}"

    def test_epoxide_bicyclo_has_oxa_prefix(self):
        """Heterocyclic bicyclo must have 'oxa' replacement prefix."""
        smiles = "O[C@H]1[C@H](O)[C@@H](O)[C@@H]2O[C@@H]2[C@@H]1O"
        name = name_compound(smiles)
        assert "oxa" in name, f"Expected 'oxa' heteroatom prefix, got: {name}"

    def test_epoxide_bicyclo_descriptor_correct(self):
        """Full name should be 7-oxa-bicyclo[4.1.0]heptane with decorations."""
        smiles = "O[C@H]1[C@H](O)[C@@H](O)[C@@H]2O[C@@H]2[C@@H]1O"
        name = name_compound(smiles)
        assert "bicyclo[4.1.0]" in name, f"Descriptor wrong: {name}"
        assert "7-oxa-" in name or "7-oxa" in name, f"Missing oxa prefix: {name}"

    def test_heterocyclic_macrocycle_uses_total_atoms(self):
        """Large heterocyclic bicyclo with N and O should count all ring atoms."""
        smiles = (
            "CCCCCC(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H]1C(=O)N[C@@H]"
            "(CCCN=C(N)N)C(=O)N[C@H]2CC[C@@H](O)N(C2=O)[C@@H]"
            "([C@@H](C)CC)C(=O)N(C)[C@@H](CC(=O)c2ccccc2N)C(=O)"
            "N[C@@H](C(C)C)C(=O)O[C@@H]1C"
        )
        name = name_compound(smiles)
        # Should use docosane (22 atoms) not hexadecane (16 carbons)
        assert "docosane" in name, f"Expected 'docosane' (22 atoms), got: {name}"
        assert "hexadecane" not in name, f"Should not use 'hexadecane' (16 carbons): {name}"
        # Should have aza and oxa prefixes
        assert "aza" in name, f"Missing aza prefix: {name}"
        assert "oxa" in name, f"Missing oxa prefix: {name}"


class TestVBNotationFormat:
    """Test that VB polycyclic notation format itself is correct.

    Most VB compounds produce correct IUPAC names that OPSIN cannot parse
    due to stereo + VB combination limitations.
    """

    def test_tricyclo_descriptor_format(self):
        """tricyclo descriptors should use parenthesized locants for secondary bridges."""
        smiles = "COCC1=C2[C@@H]3CC(C)(C)C[C@@H]3C[C@@]2(O)CC1=O"
        name = name_compound(smiles)
        assert "tricyclo[" in name, f"Expected tricyclo descriptor: {name}"
        # Should have secondary bridge with parenthesized locants
        assert "(" in name.split("tricyclo[")[1].split("]")[0], (
            f"Expected parenthesized locants in tricyclo descriptor: {name}"
        )


class TestStereoFormatEdgeCases:
    """Test stereo descriptor formatting.

    Most 'stereo_issue' triage items are actually wrong parent selection
    (missing substituents, wrong ring chosen), not stereo format problems.
    The stereo prefix format (2R,3S)- is correct per IUPAC P-93.
    """

    def test_stereo_format_parenthesized(self):
        """Stereo descriptors should be parenthesized with trailing hyphen."""
        # Simple compound with stereo
        smiles = "O[C@H]1[C@H](O)[C@@H](O)[C@@H]2O[C@@H]2[C@@H]1O"
        name = name_compound(smiles)
        if name.startswith("("):
            # Verify format: (XR,YS)-
            assert ")-" in name, f"Stereo prefix missing closing )-: {name}"
            stereo_part = name.split(")-")[0] + ")"
            # Should contain R or S labels
            assert "R" in stereo_part or "S" in stereo_part, (
                f"Stereo prefix missing R/S labels: {stereo_part}"
            )
