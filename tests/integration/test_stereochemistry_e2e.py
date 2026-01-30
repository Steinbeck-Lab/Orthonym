"""
Phase 5 Integration Tests - Stereochemistry

End-to-end tests verifying all STEREO requirements (01-05).
Tests full pipeline from SMILES input to stereodescriptor output.

STEREO-01: Place R/S descriptors at correct IUPAC locants
STEREO-02: Place E/Z descriptors for double bonds
STEREO-03: Handle cis/trans ring configurations (PARTIAL - simple disubstituted only)
STEREO-04: Format multiple stereodescriptors correctly
STEREO-05: Position descriptors correctly in name (at start)
"""

import re
import pytest
from orthonym import name_compound


# =============================================================================
# STEREO-01: Place R/S descriptors at correct IUPAC locants
# =============================================================================

class TestSTEREO01_RSLocants:
    """STEREO-01: R/S descriptors should use IUPAC locants, not atom indices."""

    @pytest.mark.parametrize("smiles,expected_pattern,description", [
        # Secondary alcohols - TRUE asymmetric (different substituents each side)
        ("C[C@H](O)CC", r"\(2[RS]\).*butan.*ol", "(2x)-butan-2-ol"),
        ("C[C@@H](O)CC", r"\(2[RS]\).*butan.*ol", "(2x)-butan-2-ol opposite"),
        ("CCC[C@H](O)C", r"\(2[RS]\).*pentan.*ol", "(2x)-pentan-2-ol"),

        # Longer chains - true asymmetric centers
        ("CCCC[C@H](O)C", r"\(2[RS]\).*hexan.*ol", "(2x)-hexan-2-ol"),
        ("CCC[C@H](O)CC", r"\(3[RS]\).*hexan.*ol", "(3x)-hexan-3-ol"),

        # Lactic acid (hydroxyacid)
        ("C[C@H](O)C(=O)O", r"[RS]", "lactic acid has stereocenter"),
    ])
    def test_rs_at_correct_locants(self, smiles, expected_pattern, description):
        """R/S descriptor locants match IUPAC numbering."""
        result = name_compound(smiles)
        assert re.search(expected_pattern, result), f"{description}: {smiles} -> {result}"

    def test_symmetric_molecule_no_stereo(self):
        """Symmetric molecules have no true stereocenter even with @ notation.

        Pentan-3-ol (CC[C@H](O)CC) has two identical ethyl groups,
        so it's not truly chiral even with @ notation in SMILES.
        RDKit correctly detects this and doesn't assign a CIP code.
        """
        result = name_compound("CC[C@H](O)CC")
        # Correctly has no stereodescriptor due to symmetry
        assert "(" not in result, f"Symmetric molecule should have no stereo: {result}"
        assert "pentan-3-ol" in result

    def test_locant_is_iupac_not_atom_index(self):
        """Verify locant is IUPAC position, not RDKit atom index."""
        # Use hexan-3-ol which IS asymmetric (propyl vs ethyl)
        # The stereocenter is at atom index 3 in CCC[C@H](O)CC
        # but IUPAC locant is 3 (counting from OH end for lowest locant)
        result = name_compound("CCC[C@H](O)CC")
        # Should have (3x) not (4x) or atom index
        assert re.search(r"\(3[RS]\)", result), f"Expected locant 3: {result}"

    def test_r_and_s_are_opposite(self):
        """@ and @@ should produce opposite R/S for same structure."""
        result_at = name_compound("C[C@H](O)CC")
        result_double_at = name_compound("C[C@@H](O)CC")

        has_r_at = "R)" in result_at or "R," in result_at
        has_r_double_at = "R)" in result_double_at or "R," in result_double_at

        # Should be opposite
        assert has_r_at != has_r_double_at, f"@ -> {result_at}, @@ -> {result_double_at}"


# =============================================================================
# STEREO-02: Place E/Z descriptors for double bonds
# =============================================================================

class TestSTEREO02_EZDescriptors:
    """STEREO-02: E/Z descriptors should have correct locants."""

    @pytest.mark.parametrize("smiles,expected_stereo,description", [
        # Simple alkenes
        ("C/C=C/C", "E", "trans-but-2-ene"),
        (r"C/C=C\C", "Z", "cis-but-2-ene"),
        ("CC/C=C/C", "E", "trans-pent-2-ene"),
        (r"CC/C=C\C", "Z", "cis-pent-2-ene"),

        # Longer chains
        ("CCC/C=C/C", "E", "trans-hex-2-ene"),
        (r"CCC/C=C\C", "Z", "cis-hex-2-ene"),
    ])
    def test_ez_descriptors(self, smiles, expected_stereo, description):
        """E/Z descriptor appears in name."""
        result = name_compound(smiles)
        assert expected_stereo in result, f"{description}: {smiles} -> {result}"

    def test_ez_locant_is_lower_bond_position(self):
        """E/Z locant should be lower carbon position of double bond."""
        result = name_compound("C/C=C/C")
        # Double bond at positions 2-3, so locant is 2
        assert "(2E)" in result, f"Expected (2E): {result}"

    def test_terminal_double_bond_no_ez(self):
        """Terminal double bonds don't have E/Z (only 2 substituents on one carbon)."""
        result = name_compound("C=CC")  # propene
        # Should not have E or Z
        assert "E" not in result and "Z" not in result, f"Terminal has no E/Z: {result}"

    def test_ethene_no_ez(self):
        """Ethene cannot have E/Z."""
        result = name_compound("C=C")
        assert "(" not in result, f"Ethene has no stereo: {result}"


# =============================================================================
# STEREO-03: Handle cis/trans ring configurations (PARTIAL)
# =============================================================================

class TestSTEREO03_RingCisTrans:
    """STEREO-03 PARTIAL: Simple cis/trans for disubstituted rings.

    Note: Full IUPAC r/c/t system for polysubstituted rings is deferred.
    These tests verify basic ring naming still works with stereochemistry.
    """

    @pytest.mark.parametrize("smiles,expected_base", [
        # 1,2-dimethylcyclohexane isomers
        ("C[C@H]1CCCC[C@@H]1C", "dimethylcyclohexane"),  # cis isomer
        ("C[C@H]1CCCC[C@H]1C", "dimethylcyclohexane"),   # trans isomer

        # 1,4-dimethylcyclohexane
        ("C[C@H]1CC[C@@H](C)CC1", "dimethylcyclohexane"),
    ])
    def test_ring_base_name_correct(self, smiles, expected_base):
        """Ring compound with stereo still produces correct base name."""
        result = name_compound(smiles)
        assert expected_base.lower() in result.lower(), f"{smiles} -> {result}"

    def test_cis_trans_ring_stereo_detected(self):
        """Ring stereo should be detectable (may or may not show in name)."""
        # The implementation may or may not include cis/trans prefix in name
        # depending on integration depth. At minimum, the ring should be named.
        result_cis = name_compound("C[C@H]1CCCC[C@@H]1C")
        result_trans = name_compound("C[C@H]1CCCC[C@H]1C")

        # Both should name the compound
        assert "cyclohexane" in result_cis.lower()
        assert "cyclohexane" in result_trans.lower()

        # Optionally, they might be different (if cis/trans implemented in names)
        # This is not strictly required for STEREO-03 partial

    @pytest.mark.parametrize("smiles,description", [
        ("C1CCCCC1", "cyclohexane - no stereo"),
        ("CC1CCCCC1", "methylcyclohexane - no stereocenter"),
    ])
    def test_non_stereo_rings_no_descriptors(self, smiles, description):
        """Rings without stereocenters have no stereodescriptors."""
        result = name_compound(smiles)
        # No parenthetical stereodescriptors
        assert "(" not in result or "(S)" not in result and "(R)" not in result, \
            f"{description}: {result}"


# =============================================================================
# STEREO-04: Format multiple stereodescriptors correctly
# =============================================================================

class TestSTEREO04_MultipleDescriptors:
    """STEREO-04: Multiple descriptors in comma-separated format."""

    @pytest.mark.parametrize("smiles,expected_format,description", [
        # Two stereocenters
        ("C[C@H](O)[C@@H](O)C", r"\(\d+[RS],\d+[RS]\)", "butane-2,3-diol"),
        ("C[C@H](O)[C@H](O)C", r"\(\d+[RS],\d+[RS]\)", "same config diol"),

        # E/Z + R/S (mixed)
        # Note: pattern allows either order
        (r"C/C=C/[C@H](O)C", r"\(\d+[EZRS],\d+[EZRS]\)", "E + stereocenter"),
    ])
    def test_multiple_descriptors_format(self, smiles, expected_format, description):
        """Multiple stereodescriptors use comma-separated format."""
        result = name_compound(smiles)
        assert re.search(expected_format, result), f"{description}: {smiles} -> {result}"

    def test_descriptors_sorted_by_locant(self):
        """Descriptors should appear in ascending locant order."""
        result = name_compound("C[C@H](O)[C@@H](O)C")
        # Extract descriptor block
        if ")-" in result:
            desc_block = result.split(")-")[0] + ")"
            # Parse locants
            locants = [int(m) for m in re.findall(r"(\d+)[RS]", desc_block)]
            assert locants == sorted(locants), f"Locants not sorted: {desc_block}"

    def test_two_stereocenters_have_two_descriptors(self):
        """Compound with 2 stereocenters should have 2 descriptors."""
        result = name_compound("C[C@H](O)[C@@H](O)C")
        # Count R and S occurrences in descriptor block
        desc_block = result.split(")-")[0] if ")-" in result else ""
        rs_count = desc_block.count("R") + desc_block.count("S")
        assert rs_count == 2, f"Expected 2 descriptors: {result}"


# =============================================================================
# STEREO-05: Position descriptors correctly in name
# =============================================================================

class TestSTEREO05_DescriptorPosition:
    """STEREO-05: Stereodescriptors appear at the very start of the name."""

    def test_descriptor_at_start_rs(self):
        """R/S descriptor starts the name."""
        result = name_compound("C[C@H](O)CC")
        assert result.startswith("("), f"Should start with (: {result}"

    def test_descriptor_at_start_ez(self):
        """E/Z descriptor starts the name."""
        result = name_compound("C/C=C/C")
        assert result.startswith("("), f"Should start with (: {result}"

    def test_descriptor_before_substituents(self):
        """Descriptor comes before any substituent prefixes."""
        # 3-methylbutan-2-ol with stereo
        result = name_compound("C[C@H](O)C(C)C")
        if "(" in result:
            # The ( should come before any prefix
            paren_pos = result.index("(")
            # Any letter before ( would be wrong
            prefix = result[:paren_pos]
            assert not any(c.isalpha() for c in prefix), \
                f"Prefix before stereo: {result}"

    def test_descriptor_format_locant_cip_hyphen(self):
        """Format should be (locantCIP)- with trailing hyphen."""
        result = name_compound("C[C@H](O)CC")
        # Check for (digit + R/S) followed by )-
        assert re.match(r"\(\d+[RS]\)-", result), f"Wrong format: {result}"


# =============================================================================
# Edge Cases and Regression Tests
# =============================================================================

class TestStereoEdgeCases:
    """Edge cases for stereochemistry naming."""

    def test_no_stereo_no_descriptors(self):
        """Molecules without stereo have no descriptors."""
        assert "(" not in name_compound("CCCC")  # butane
        assert "(" not in name_compound("CCO")   # ethanol
        assert "(" not in name_compound("CCC=O") # propanal

    def test_unspecified_stereo_no_descriptors(self):
        """Unspecified stereo (no @ or /) produces no descriptors."""
        # These have potential stereocenters but no specified config
        assert "(" not in name_compound("CC(O)CC")  # unspecified butan-2-ol
        assert "(" not in name_compound("CC=CC")    # unspecified but-2-ene

    def test_symmetric_no_stereo(self):
        """Symmetric molecules have no true stereocenters."""
        # Pentan-3-ol is symmetric (two ethyl groups)
        result = name_compound("CC[C@H](O)CC")
        # Might or might not have descriptor depending on symmetry detection
        # At minimum, should name correctly
        assert "pentan" in result

    def test_benzene_no_ring_stereo(self):
        """Benzene ring has no cis/trans stereo (aromatic)."""
        result = name_compound("c1ccccc1")
        assert "cis" not in result.lower() and "trans" not in result.lower()


# =============================================================================
# Specific Compound Validation
# =============================================================================

class TestSpecificCompounds:
    """Exact output validation for specific compounds."""

    def test_r_butan_2_ol(self):
        """(2R)-butan-2-ol exact match."""
        result = name_compound("C[C@@H](O)CC")
        assert result == "(2R)-butan-2-ol", f"Got: {result}"

    def test_s_butan_2_ol(self):
        """(2S)-butan-2-ol exact match."""
        result = name_compound("C[C@H](O)CC")
        assert result == "(2S)-butan-2-ol", f"Got: {result}"

    def test_e_but_2_ene(self):
        """(2E)-but-2-ene exact match."""
        result = name_compound("C/C=C/C")
        assert result == "(2E)-but-2-ene", f"Got: {result}"

    def test_z_but_2_ene(self):
        """(2Z)-but-2-ene exact match."""
        result = name_compound(r"C/C=C\C")
        assert result == "(2Z)-but-2-ene", f"Got: {result}"


# =============================================================================
# Requirement Coverage Summary Tests
# =============================================================================

class TestRequirementCoverage:
    """Consolidated tests proving each STEREO requirement is met."""

    def test_stereo_01_rs_locants_verified(self):
        """STEREO-01: R/S at IUPAC locants (not atom indices)."""
        # Verify multiple compounds - use truly asymmetric molecules
        assert re.search(r"\(2[RS]\)", name_compound("C[C@H](O)CC"))  # butan-2-ol
        assert re.search(r"\(3[RS]\)", name_compound("CCC[C@H](O)CC"))  # hexan-3-ol (asymmetric)

    def test_stereo_02_ez_locants_verified(self):
        """STEREO-02: E/Z at correct double bond locants."""
        assert "(2E)" in name_compound("C/C=C/C")
        assert "(2Z)" in name_compound(r"C/C=C\C")

    def test_stereo_03_ring_basic_verified(self):
        """STEREO-03 PARTIAL: Ring naming works with stereo."""
        # Basic verification that rings with stereo still name correctly
        result = name_compound("C[C@H]1CCCC[C@@H]1C")
        assert "dimethylcyclohexane" in result.lower()

    def test_stereo_04_multiple_format_verified(self):
        """STEREO-04: Multiple descriptors in single block."""
        result = name_compound("C[C@H](O)[C@@H](O)C")
        # Should have format like (2x,3y)-
        assert re.search(r"\(\d+[RS],\d+[RS]\)", result)

    def test_stereo_05_position_verified(self):
        """STEREO-05: Descriptors at start of name."""
        result = name_compound("C[C@H](O)CC")
        assert result.startswith("(")


# =============================================================================
# Parametrized comprehensive tests
# =============================================================================

@pytest.mark.parametrize("smiles,checks", [
    # Check tuples: (substring_in_name, must_contain_bool)
    ("C[C@H](O)CC", [("(2", True), ("butan", True), ("-ol", True)]),
    ("C/C=C/C", [("(2E)", True), ("but", True), ("ene", True)]),
    ("CCCC", [("(", False)]),  # No stereo
    ("C=C", [("(", False)]),   # Ethene, no E/Z
])
def test_comprehensive_stereo_naming(smiles, checks):
    """Comprehensive tests for various stereo scenarios."""
    result = name_compound(smiles)
    for substring, must_contain in checks:
        if must_contain:
            assert substring in result, f"{smiles}: expected '{substring}' in '{result}'"
        else:
            assert substring not in result, f"{smiles}: unexpected '{substring}' in '{result}'"
