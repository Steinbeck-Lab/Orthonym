"""
Comprehensive End-to-End Integration Tests for a phase: Complex Ring Systems.

Validates all a phase requirements (COMPLEX-01 through COMPLEX-06 + VALID):
- COMPLEX-01: Bicyclo compound naming (bicyclo[x.y.z] format)
- COMPLEX-02: Spiro compound naming (spiro[a.b] format)
- COMPLEX-03: Ortho-fused bicyclics (decalin, tetralin, naphthalene)
- COMPLEX-04: Fusion descriptors (systematic fusion name generation)
- COMPLEX-05: Heterocyclic fused systems (indole, quinoline, carbazole)
- COMPLEX-06: Complex PAHs (pyrene, perylene, coronene, etc.)

This test file provides comprehensive E2E coverage for a phase Final Validation.

Reference: IUPAC 2013 Blue Book, Sections,,,
"""

import pytest
from orthonym import name_compound


# =============================================================================
# COMPLEX-01: Bicyclo Compound Naming
# =============================================================================

class TestCOMPLEX01_Bicyclo_Hydrocarbons:
    """COMPLEX-01: Bicyclo hydrocarbon systematic naming."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        # Bicyclo[3.2.1]octane
        ("C1CC2CCCC1C2", "bicyclo[3.2.1]octane"),
        # Bicyclo[4.2.1]nonane (pinned here since the pass-or-xfail test_accuracy test was removed)
        ("C1CC2CCCCC1C2", "bicyclo[4.2.1]nonane"),
    ])
    def test_bicyclo_systematic_naming(self, smiles, expected):
        """Test systematic bicyclo naming with various bridge sizes."""
        assert name_compound(smiles) == expected

    @pytest.mark.integration
    def test_bicyclo_descriptor_format(self):
        """Verify bicyclo descriptor format [x.y.z] is correct."""
        result = name_compound("C1CC2CCC1CC2")  # bicyclo[2.2.2]octane
        assert "bicyclo[" in result
        assert "]" in result
        # Should have two periods in descriptor
        desc_start = result.index("[")
        desc_end = result.index("]")
        descriptor = result[desc_start:desc_end+1]
        assert descriptor.count(".") == 2

    @pytest.mark.integration
    def test_bicyclo_numbers_ordered_correctly(self):
        """Verify bicyclo numbers are in decreasing order (largest bridge first)."""
        result = name_compound("C1CC2CCCC1C2")  # bicyclo[3.2.1]octane
        # Extract the numbers from descriptor
        desc_start = result.index("[")
        desc_end = result.index("]")
        descriptor = result[desc_start+1:desc_end]
        numbers = [int(n) for n in descriptor.split(".")]
        assert numbers == sorted(numbers, reverse=True), "Bridge numbers should be in decreasing order"


# =============================================================================
# COMPLEX-02: Spiro Compound Naming
# =============================================================================

class TestCOMPLEX02_Spiro_Hydrocarbons:
    """COMPLEX-02: Spiro hydrocarbon systematic naming."""

    @pytest.mark.integration
    def test_spiro_descriptor_format(self):
        """Verify spiro descriptor format [a.b] is correct."""
        result = name_compound("C1CCC2(CC1)CCCCC2")
        assert "spiro[" in result
        assert "]" in result
        # Extract descriptor
        desc_start = result.index("[")
        desc_end = result.index("]")
        descriptor = result[desc_start:desc_end+1]
        assert descriptor.count(".") == 1  # Single period in spiro descriptor


# =============================================================================
# COMPLEX-04: Fusion Descriptors
# =============================================================================

class TestCOMPLEX04_FusionDescriptors:
    """COMPLEX-04: Fusion descriptor generation for systematic names."""

    @pytest.mark.integration
    def test_fusion_prefix_generation(self):
        """Test that fusion prefixes are generated correctly."""
        from orthonym.rules.fusion_descriptors import get_fusion_prefix

        assert get_fusion_prefix('benzene') == 'benzo'
        assert get_fusion_prefix('naphthalene') == 'naphtho'
        assert get_fusion_prefix('furan') == 'furo'
        assert get_fusion_prefix('pyrrole') == 'pyrrolo'
        assert get_fusion_prefix('pyridine') == 'pyrido'

    @pytest.mark.integration
    def test_fusion_edge_calculation(self):
        """Test fusion edge letter assignment."""
        from orthonym.rules.fusion_descriptors import get_fusion_letter

        # Ring atoms [0,1,2,3,4,5], shared edge between 0-1 = edge 'a'
        assert get_fusion_letter([0, 1, 2, 3, 4, 5], (0, 1)) == 'a'
        # Edge between 1-2 = edge 'b'
        assert get_fusion_letter([0, 1, 2, 3, 4, 5], (1, 2)) == 'b'

    @pytest.mark.integration
    def test_fusion_descriptor_format(self):
        """Test fusion descriptor format [num,num-letter]."""
        from orthonym.rules.fusion_descriptors import generate_fusion_descriptor

        # Use ring indices where shared atoms exist in both rings
        parent = [0, 1, 2, 3, 4, 5]  # 6-membered
        child = [0, 1, 6, 7, 8]      # 5-membered sharing atoms 0,1 with parent
        shared = {0, 1}              # Shared edge

        descriptor = generate_fusion_descriptor(parent, child, shared)
        # Should be in format [num,num-letter]
        assert '[' in descriptor
        assert ']' in descriptor
        assert '-' in descriptor


class TestCOMPLEX04_SystematicFusionNames:
    """Systematic fusion name building."""

    @pytest.mark.integration
    def test_benzo_fusion_prefix_no_elision(self):
        """IUPAC 2013: 'o' in benzo not elided before vowels."""
        from orthonym.rules.fusion_descriptors import build_systematic_fusion_name

        # benzo + anthracene = benzoanthracene (NOT benzanthracene)
        result = build_systematic_fusion_name('anthracene', 'benzene', '[a]')
        assert result == 'benzo[a]anthracene'
        assert 'benzanthracene' not in result


# =============================================================================
# COMPLEX-05: Heterocyclic Fused Systems
# =============================================================================


class TestCOMPLEX05_SubstitutedFusedHeterocycles:
    """Substituted fused heterocycle naming - known limitations."""

    # Task 12 fix a performance pass (wp6-tests): the non-strict xfail on the whole test was
    # stale for 2-methylindole (it XPASSed) and hid a live defect for
    # 4-methylquinoline, which was named 'lepidine' at pin_verified: a trivial name
    # imported from the OPSIN resource tables with is_pin False
    # (data/opsin_imports/aryl_groups.py), returned by the retained-name lookup
    # (routing/dispatch_table._handle_retained_name -> ALL_RETAINED_NAMES), 0 Blue
    # Book hits. Quinoline is a retained PIN parent, so the PIN is
    # '4-methylquinoline' (OPSIN 2.9.0 full-InChIKey exact, as is 'lepidine').
    # wp7: FIXED -- an OPSIN-import name enters the PIN lookup only with Blue
    # Book PIN evidence (data._OPSIN_IMPORT_PIN_EVIDENCE); the strict xfail is gone.
    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        ("Cc1cc2ccccc2[nH]1", "2-methyl-1H-indole"),
        ("Cc1ccnc2ccccc12", "4-methylquinoline"),
    ])
    def test_substituted_fused_heterocycles(self, smiles, expected):
        """Substituted fused heterocycle naming."""
        assert name_compound(smiles) == expected


# =============================================================================
# COMPLEX-06: Complex PAHs (Polycyclic Aromatic Hydrocarbons)
# =============================================================================


class TestCOMPLEX06_PentacyclicAndHigher:
    """Pentacyclic and larger PAH naming."""

    @pytest.mark.integration
    @pytest.mark.parametrize("smiles,expected", [
        # Perylene - peri-fused pentacyclic: OPSIN 2.9.0's `perylene`, '(5) perylene
        # (PIN)' (the Blue Book); the SMILES here before, c1ccc2cccc3cc4c(c1)cc1cccc4c1c23,
        # is another C20 skeleton with six rings
        ("c1cc2cccc3c4cccc5cccc(c(c1)c23)c54", "perylene"),
        # Coronene - hexagonal 7-ring
        ("c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61", "coronene"),
    ])
    def test_larger_pah_retained_names(self, smiles, expected):
        """Test larger PAH retained names."""
        assert name_compound(smiles) == expected


class TestCOMPLEX06_PAHLookup:
    """PAH data lookup and verification."""

    @pytest.mark.integration
    def test_pah_smiles_lookup_works(self):
        """Test PAH lookup by SMILES."""
        from orthonym.data.polycyclic_data import get_polycyclic_by_smiles

        result = get_polycyclic_by_smiles('c1ccc2ccccc2c1')
        assert result is not None
        assert result['name'] == 'naphthalene'


# =============================================================================
# a phase-5 Regression Tests
# =============================================================================


class TestPhase5Regression:
    """Regression tests for a phase (Stereochemistry)."""

    @pytest.mark.integration
    def test_stereochemistry_preserved(self):
        """Stereochemistry naming should be preserved."""
        # R-butan-2-ol
        result = name_compound("C[C@H](O)CC")
        assert "R" in result or "S" in result  # Has stereodescriptor

        # E-but-2-ene
        result = name_compound("C/C=C/C")
        assert "E" in result or "Z" in result  # Has E/Z descriptor


# =============================================================================
# Compound Classification Tests
# =============================================================================

class TestCompoundClassification:
    """Test compound classification for validation."""

    @pytest.mark.integration
    def test_bicyclo_is_complex_ring(self):
        """Bicyclo should be classified as complex ring system."""
        from orthonym.assembly.composer import _is_complex_ring_system, _classify_complex_ring
        from rdkit import Chem

        mol = Chem.MolFromSmiles("C1CC2CCC1C2")
        assert _is_complex_ring_system(mol) == True
        assert _classify_complex_ring(mol) == "bicyclo"

    @pytest.mark.integration
    def test_spiro_is_complex_ring(self):
        """Spiro should be classified as complex ring system."""
        from orthonym.assembly.composer import _is_complex_ring_system, _classify_complex_ring
        from rdkit import Chem

        mol = Chem.MolFromSmiles("C1CCC2(CC1)CCCC2")
        assert _is_complex_ring_system(mol) == True
        assert _classify_complex_ring(mol) == "spiro"

    @pytest.mark.integration
    def test_monocyclic_not_complex(self):
        """Simple monocyclic should NOT be classified as complex."""
        from orthonym.assembly.composer import _is_complex_ring_system
        from rdkit import Chem

        mol = Chem.MolFromSmiles("C1CCCCC1")  # cyclohexane
        assert _is_complex_ring_system(mol) == False


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================

class TestKnownLimitations:
    """Document known limitations (expected failures)."""

    @pytest.mark.integration
    def test_dispiro_not_supported(self):
        """Dispiro systems are named (the old 'not supported' marker was stale).

        Task 12 fix a performance pass (wp6-tests): the non-strict xfail XPASSed. The engine
        names the system 'dispiro[2.1.3^5.2^3]decane' (pin_verified), which OPSIN
        2.9.0 parses to the input's full InChIKey, so this asserts the round
        trip. The Blue Book spelling carries the superscripts:
        (the Blue Book) "Each time a spiro atom is reached for the second
        time its locant... is cited as a superscript number", e.g.
        'dispiro[3.2.3^7.2^4]dodecane (PIN)' (:9981); the next test asserts it.
        The test id is kept."""
        from tests.support.jars import jar_or_skip
        from tests.support.rt_assert import assert_full_rt
        jar_or_skip()
        result = name_compound("C1CC2(C1)CC3(CC2)CC3")
        assert "dispiro" in result
        assert_full_rt(result, "C1CC2(C1)CC3(CC2)CC3")

    @pytest.mark.integration
    def test_dispiro_descriptor_carries_superscripts(self):
        """ (the Blue Book): "Each time a spiro atom is reached for
        the second time its locant, which has already been assigned, is cited as
        a superscript number to the number of the preceding linking atoms";
        'dispiro[3.2.3^7.2^4]dodecane (PIN)' (:9981)."""
        assert name_compound("C1CC2(C1)CC3(CC2)CC3") == "dispiro[2.1.3^5.2^3]decane"
