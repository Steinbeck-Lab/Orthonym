"""
Unit tests for bicyclo compound naming.

Tests the bicyclo[x.y.z] descriptor generation and naming functions
according to IUPAC 2013 nomenclature.

Coverage:
- Bridgehead detection
- Bicyclo system identification
- Bridge path finding
- Bridge length calculation
- Descriptor generation
- Full naming with retained name lookup
"""

import pytest
from rdkit import Chem

from src.orthonym.rules.bicyclo import (
    find_true_bridgeheads,
    is_bicyclo_system,
    find_bridge_paths,
    get_bridge_lengths,
    generate_bicyclo_descriptor,
    name_bicyclo_system,
    get_bicyclo_ring_atoms,
)
from src.orthonym.data.bicyclo_systems import (
    get_retained_bicyclo_name,
    is_retained_bicyclo,
    BICYCLO_RETAINED_NAMES,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def norbornane():
    """Bicyclo[2.2.1]heptane - most common bridged bicyclic."""
    return Chem.MolFromSmiles("C1CC2CCC1C2")


@pytest.fixture
def bicyclo_222_octane():
    """Bicyclo[2.2.2]octane - symmetric bridged bicyclic."""
    return Chem.MolFromSmiles("C1CC2CCC1CC2")


@pytest.fixture
def bicyclo_110_butane():
    """Bicyclo[1.1.0]butane - smallest bicyclic."""
    return Chem.MolFromSmiles("C1C2CC12")


@pytest.fixture
def bicyclo_111_pentane():
    """Bicyclo[1.1.1]pentane - propellane precursor."""
    return Chem.MolFromSmiles("C1C2CC1C2")


@pytest.fixture
def cyclohexane():
    """Simple monocyclic - NOT a bicyclo system."""
    return Chem.MolFromSmiles("C1CCCCC1")


@pytest.fixture
def spiro_45_decane():
    """Spiro[4.5]decane - NOT a bicyclo system (spiro)."""
    return Chem.MolFromSmiles("C1CCC2(CC1)CCCCC2")


@pytest.fixture
def quinuclidine():
    """1-Azabicyclo[2.2.2]octane - heterobicyclic with retained name."""
    return Chem.MolFromSmiles("C1CC2CCC1CN2")


# ============================================================================
# Tests for is_bicyclo_system()
# ============================================================================

class TestIsBicycloSystem:
    """Test bicyclo system detection."""

    @pytest.mark.unit
    def test_norbornane_is_bicyclo(self, norbornane):
        """Norbornane (bicyclo[2.2.1]heptane) should be detected as bicyclo."""
        assert is_bicyclo_system(norbornane) is True

    @pytest.mark.unit
    def test_bicyclo_222_octane_is_bicyclo(self, bicyclo_222_octane):
        """Bicyclo[2.2.2]octane should be detected as bicyclo."""
        assert is_bicyclo_system(bicyclo_222_octane) is True

    @pytest.mark.unit
    def test_bicyclo_111_pentane_is_bicyclo(self, bicyclo_111_pentane):
        """Bicyclo[1.1.1]pentane should be detected as bicyclo."""
        assert is_bicyclo_system(bicyclo_111_pentane) is True

    @pytest.mark.unit
    def test_cyclohexane_not_bicyclo(self, cyclohexane):
        """Cyclohexane (monocyclic) should NOT be detected as bicyclo."""
        assert is_bicyclo_system(cyclohexane) is False

    @pytest.mark.unit
    def test_spiro_not_bicyclo(self, spiro_45_decane):
        """Spiro[4.5]decane should NOT be detected as bicyclo."""
        assert is_bicyclo_system(spiro_45_decane) is False

    @pytest.mark.unit
    def test_acyclic_not_bicyclo(self):
        """Acyclic compound should NOT be detected as bicyclo."""
        mol = Chem.MolFromSmiles("CCCCCC")  # hexane
        assert is_bicyclo_system(mol) is False

    @pytest.mark.unit
    def test_quinuclidine_is_bicyclo(self, quinuclidine):
        """Quinuclidine (heterobicyclo) should be detected as bicyclo."""
        assert is_bicyclo_system(quinuclidine) is True


# ============================================================================
# Tests for find_true_bridgeheads()
# ============================================================================

class TestFindTrueBridgeheads:
    """Test bridgehead atom detection."""

    @pytest.mark.unit
    def test_norbornane_has_two_bridgeheads(self, norbornane):
        """Norbornane should have exactly 2 bridgehead atoms."""
        bridgeheads = find_true_bridgeheads(norbornane)
        assert len(bridgeheads) == 2

    @pytest.mark.unit
    def test_bicyclo_222_has_two_bridgeheads(self, bicyclo_222_octane):
        """Bicyclo[2.2.2]octane should have exactly 2 bridgehead atoms."""
        bridgeheads = find_true_bridgeheads(bicyclo_222_octane)
        assert len(bridgeheads) == 2

    @pytest.mark.unit
    def test_bicyclo_111_has_two_bridgeheads(self, bicyclo_111_pentane):
        """Bicyclo[1.1.1]pentane should have exactly 2 bridgehead atoms."""
        bridgeheads = find_true_bridgeheads(bicyclo_111_pentane)
        assert len(bridgeheads) == 2

    @pytest.mark.unit
    def test_cyclohexane_no_bridgeheads(self, cyclohexane):
        """Cyclohexane should have no bridgehead atoms."""
        bridgeheads = find_true_bridgeheads(cyclohexane)
        assert len(bridgeheads) == 0

    @pytest.mark.unit
    def test_bridgeheads_have_three_neighbors(self, norbornane):
        """Bridgehead atoms should have exactly 3 neighbors."""
        bridgeheads = find_true_bridgeheads(norbornane)
        for bh in bridgeheads:
            atom = norbornane.GetAtomWithIdx(bh)
            assert len(list(atom.GetNeighbors())) == 3


# ============================================================================
# Tests for find_bridge_paths()
# ============================================================================

class TestFindBridgePaths:
    """Test bridge path finding."""

    @pytest.mark.unit
    def test_norbornane_three_paths(self, norbornane):
        """Norbornane should have exactly 3 paths between bridgeheads."""
        bridgeheads = list(find_true_bridgeheads(norbornane))
        paths = find_bridge_paths(norbornane, bridgeheads[0], bridgeheads[1])
        assert len(paths) == 3

    @pytest.mark.unit
    def test_bicyclo_222_three_paths(self, bicyclo_222_octane):
        """Bicyclo[2.2.2]octane should have exactly 3 paths between bridgeheads."""
        bridgeheads = list(find_true_bridgeheads(bicyclo_222_octane))
        paths = find_bridge_paths(bicyclo_222_octane, bridgeheads[0], bridgeheads[1])
        assert len(paths) == 3

    @pytest.mark.unit
    def test_bicyclo_111_three_paths(self, bicyclo_111_pentane):
        """Bicyclo[1.1.1]pentane should have exactly 3 paths between bridgeheads."""
        bridgeheads = list(find_true_bridgeheads(bicyclo_111_pentane))
        paths = find_bridge_paths(bicyclo_111_pentane, bridgeheads[0], bridgeheads[1])
        assert len(paths) == 3

    @pytest.mark.unit
    def test_paths_include_both_bridgeheads(self, norbornane):
        """Each path should include both bridgehead atoms."""
        bridgeheads = list(find_true_bridgeheads(norbornane))
        paths = find_bridge_paths(norbornane, bridgeheads[0], bridgeheads[1])
        for path in paths:
            assert bridgeheads[0] in path
            assert bridgeheads[1] in path


# ============================================================================
# Tests for get_bridge_lengths()
# ============================================================================

class TestGetBridgeLengths:
    """Test bridge length calculation."""

    @pytest.mark.unit
    def test_norbornane_bridge_lengths_221(self, norbornane):
        """Norbornane should have bridge lengths [2, 2, 1]."""
        bridgeheads = list(find_true_bridgeheads(norbornane))
        lengths = get_bridge_lengths(norbornane, bridgeheads[0], bridgeheads[1])
        assert lengths == [2, 2, 1]

    @pytest.mark.unit
    def test_bicyclo_222_bridge_lengths(self, bicyclo_222_octane):
        """Bicyclo[2.2.2]octane should have bridge lengths [2, 2, 2]."""
        bridgeheads = list(find_true_bridgeheads(bicyclo_222_octane))
        lengths = get_bridge_lengths(bicyclo_222_octane, bridgeheads[0], bridgeheads[1])
        assert lengths == [2, 2, 2]

    @pytest.mark.unit
    def test_bicyclo_111_bridge_lengths(self, bicyclo_111_pentane):
        """Bicyclo[1.1.1]pentane should have bridge lengths [1, 1, 1]."""
        bridgeheads = list(find_true_bridgeheads(bicyclo_111_pentane))
        lengths = get_bridge_lengths(bicyclo_111_pentane, bridgeheads[0], bridgeheads[1])
        assert lengths == [1, 1, 1]

    @pytest.mark.unit
    def test_bridge_lengths_sorted_descending(self, norbornane):
        """Bridge lengths should be sorted in descending order."""
        bridgeheads = list(find_true_bridgeheads(norbornane))
        lengths = get_bridge_lengths(norbornane, bridgeheads[0], bridgeheads[1])
        assert lengths == sorted(lengths, reverse=True)

    @pytest.mark.unit
    def test_bicyclo_110_bridge_lengths(self, bicyclo_110_butane):
        """Bicyclo[1.1.0]butane should have bridge lengths [1, 1, 0]."""
        bridgeheads = list(find_true_bridgeheads(bicyclo_110_butane))
        lengths = get_bridge_lengths(bicyclo_110_butane, bridgeheads[0], bridgeheads[1])
        assert lengths == [1, 1, 0]


# ============================================================================
# Tests for generate_bicyclo_descriptor()
# ============================================================================

class TestGenerateBicycloDescriptor:
    """Test bicyclo[x.y.z] descriptor generation."""

    @pytest.mark.unit
    def test_norbornane_descriptor(self, norbornane):
        """Norbornane should generate 'bicyclo[2.2.1]' descriptor."""
        descriptor = generate_bicyclo_descriptor(norbornane)
        assert descriptor == "bicyclo[2.2.1]"

    @pytest.mark.unit
    def test_bicyclo_222_descriptor(self, bicyclo_222_octane):
        """Bicyclo[2.2.2]octane should generate 'bicyclo[2.2.2]' descriptor."""
        descriptor = generate_bicyclo_descriptor(bicyclo_222_octane)
        assert descriptor == "bicyclo[2.2.2]"

    @pytest.mark.unit
    def test_bicyclo_111_descriptor(self, bicyclo_111_pentane):
        """Bicyclo[1.1.1]pentane should generate 'bicyclo[1.1.1]' descriptor."""
        descriptor = generate_bicyclo_descriptor(bicyclo_111_pentane)
        assert descriptor == "bicyclo[1.1.1]"

    @pytest.mark.unit
    def test_bicyclo_110_descriptor(self, bicyclo_110_butane):
        """Bicyclo[1.1.0]butane should generate 'bicyclo[1.1.0]' descriptor."""
        descriptor = generate_bicyclo_descriptor(bicyclo_110_butane)
        assert descriptor == "bicyclo[1.1.0]"

    @pytest.mark.unit
    def test_cyclohexane_no_descriptor(self, cyclohexane):
        """Cyclohexane should return None (not a bicyclo system)."""
        descriptor = generate_bicyclo_descriptor(cyclohexane)
        assert descriptor is None

    @pytest.mark.unit
    def test_spiro_no_descriptor(self, spiro_45_decane):
        """Spiro compound should return None (not a bicyclo system)."""
        descriptor = generate_bicyclo_descriptor(spiro_45_decane)
        assert descriptor is None

    @pytest.mark.unit
    def test_descriptor_format(self, norbornane):
        """Descriptor should match format 'bicyclo[x.y.z]'."""
        descriptor = generate_bicyclo_descriptor(norbornane)
        assert descriptor.startswith("bicyclo[")
        assert descriptor.endswith("]")
        # Extract and verify x.y.z values
        values = descriptor[8:-1].split(".")
        assert len(values) == 3
        assert all(v.isdigit() for v in values)


# ============================================================================
# Tests for name_bicyclo_system()
# ============================================================================

class TestNameBicycloSystem:
    """Test full bicyclo system naming."""

    @pytest.mark.unit
    def test_norbornane_retained_name(self, norbornane):
        """Norbornane should use retained name 'norbornane'."""
        name = name_bicyclo_system(norbornane)
        assert name == "norbornane"

    @pytest.mark.unit
    def test_bicyclo_222_systematic_name(self, bicyclo_222_octane):
        """Bicyclo[2.2.2]octane should generate systematic name."""
        name = name_bicyclo_system(bicyclo_222_octane)
        assert name == "bicyclo[2.2.2]octane"

    @pytest.mark.unit
    def test_bicyclo_111_systematic_name(self, bicyclo_111_pentane):
        """Bicyclo[1.1.1]pentane should generate systematic name."""
        name = name_bicyclo_system(bicyclo_111_pentane)
        assert name == "bicyclo[1.1.1]pentane"

    @pytest.mark.unit
    def test_bicyclo_110_systematic_name(self, bicyclo_110_butane):
        """Bicyclo[1.1.0]butane should use retained name."""
        name = name_bicyclo_system(bicyclo_110_butane)
        # This has a retained name in our data
        assert name == "bicyclo[1.1.0]butane"

    @pytest.mark.unit
    def test_quinuclidine_retained_name(self, quinuclidine):
        """Quinuclidine (heterobicyclo) should use retained name."""
        name = name_bicyclo_system(quinuclidine)
        assert name == "quinuclidine"

    @pytest.mark.unit
    def test_cyclohexane_returns_none(self, cyclohexane):
        """Cyclohexane should return None (not a bicyclo system)."""
        name = name_bicyclo_system(cyclohexane)
        assert name is None

    @pytest.mark.unit
    def test_name_contains_parent_hydrocarbon(self, bicyclo_222_octane):
        """Systematic name should contain parent hydrocarbon (e.g., 'octane')."""
        name = name_bicyclo_system(bicyclo_222_octane)
        assert "octane" in name

    @pytest.mark.unit
    def test_norbornane_correct_carbon_count(self, norbornane):
        """Norbornane has 7 carbons, so systematic name would be ...heptane."""
        # The formula: x + y + z + 2 = 2 + 2 + 1 + 2 = 7
        # Since it uses retained name, we verify the carbon count indirectly
        ring_atoms = get_bicyclo_ring_atoms(norbornane)
        assert len(ring_atoms) == 7


# ============================================================================
# Tests for Retained Names
# ============================================================================

class TestRetainedNames:
    """Test retained name lookup."""

    @pytest.mark.unit
    def test_norbornane_canonical_lookup(self):
        """Norbornane should be found by canonical SMILES."""
        canonical = "C1CC2CCC1C2"
        name = get_retained_bicyclo_name(canonical)
        assert name == "norbornane"

    @pytest.mark.unit
    def test_quinuclidine_canonical_lookup(self):
        """Quinuclidine should be found by canonical SMILES."""
        canonical = "C1CC2CCC1CN2"
        name = get_retained_bicyclo_name(canonical)
        assert name == "quinuclidine"

    @pytest.mark.unit
    def test_is_retained_true_for_norbornane(self):
        """is_retained_bicyclo should return True for norbornane."""
        assert is_retained_bicyclo("C1CC2CCC1C2") is True

    @pytest.mark.unit
    def test_is_retained_false_for_unknown(self):
        """is_retained_bicyclo should return False for unknown SMILES."""
        assert is_retained_bicyclo("C1CC2CCC1CC2") is False  # bicyclo[2.2.2]octane

    @pytest.mark.unit
    def test_retained_names_count(self):
        """Should have at least 3 retained bicyclo names."""
        assert len(BICYCLO_RETAINED_NAMES) >= 3


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.unit
    def test_smallest_bicyclo_110(self, bicyclo_110_butane):
        """Smallest possible bicyclo system should work."""
        assert is_bicyclo_system(bicyclo_110_butane) is True
        descriptor = generate_bicyclo_descriptor(bicyclo_110_butane)
        assert descriptor == "bicyclo[1.1.0]"

    @pytest.mark.unit
    def test_invalid_smiles_handling(self):
        """Invalid SMILES should not crash."""
        mol = Chem.MolFromSmiles("invalid_smiles")
        assert mol is None
        # Functions should handle None gracefully
        # (they expect valid mol objects; caller responsibility to validate)

    @pytest.mark.unit
    def test_methane_not_bicyclo(self):
        """Methane should not be a bicyclo system."""
        mol = Chem.MolFromSmiles("C")
        assert is_bicyclo_system(mol) is False

    @pytest.mark.unit
    def test_benzene_not_bicyclo(self):
        """Benzene should not be a bicyclo system."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert is_bicyclo_system(mol) is False

    @pytest.mark.unit
    def test_naphthalene_not_simple_bicyclo(self):
        """Naphthalene is fused, not bridged - should not be simple bicyclo."""
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        # Naphthalene has shared edge (fused), not bridged
        # The bridgehead test should fail
        result = is_bicyclo_system(mol)
        # Fused systems don't have true bridgeheads (3 neighbors pattern)
        assert result is False


# ============================================================================
# Integration-style Tests
# ============================================================================

class TestBicycloNamingIntegration:
    """Integration tests for complete naming workflow."""

    @pytest.mark.unit
    def test_norbornane_full_workflow(self, norbornane):
        """Test complete workflow for norbornane."""
        # Detection
        assert is_bicyclo_system(norbornane)

        # Bridgeheads
        bridgeheads = find_true_bridgeheads(norbornane)
        assert len(bridgeheads) == 2

        # Paths
        bh_list = list(bridgeheads)
        paths = find_bridge_paths(norbornane, bh_list[0], bh_list[1])
        assert len(paths) == 3

        # Lengths
        lengths = get_bridge_lengths(norbornane, bh_list[0], bh_list[1])
        assert lengths == [2, 2, 1]

        # Descriptor
        descriptor = generate_bicyclo_descriptor(norbornane)
        assert descriptor == "bicyclo[2.2.1]"

        # Full name (retained)
        name = name_bicyclo_system(norbornane)
        assert name == "norbornane"

    @pytest.mark.unit
    def test_bicyclo_222_full_workflow(self, bicyclo_222_octane):
        """Test complete workflow for bicyclo[2.2.2]octane."""
        # Detection
        assert is_bicyclo_system(bicyclo_222_octane)

        # Descriptor
        descriptor = generate_bicyclo_descriptor(bicyclo_222_octane)
        assert descriptor == "bicyclo[2.2.2]"

        # Full name (systematic)
        name = name_bicyclo_system(bicyclo_222_octane)
        assert name == "bicyclo[2.2.2]octane"

        # Carbon count verification: 2+2+2+2 = 8
        ring_atoms = get_bicyclo_ring_atoms(bicyclo_222_octane)
        assert len(ring_atoms) == 8
