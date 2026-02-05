"""
Unit tests for polycyclic bridged system detection and classification.

Tests the polycyclic_bridged module functions:
- Ring count calculation
- System classification (bicyclo/tricyclo/etc.)
- Bridgehead detection
- Bridge path finding

Coverage of IUPAC 2013 P-23.3 (polycyclic ring systems).
"""

import pytest
from rdkit import Chem

from src.orthonym.rules.polycyclic_bridged import (
    get_ring_count,
    count_cuts_to_open,
    classify_bridged_system,
    is_tricyclo_system,
    is_tetracyclo_system,
    is_pentacyclo_or_higher,
    find_all_bridgeheads,
    get_ring_atoms,
    find_all_bridge_paths,
    find_main_ring,
    analyze_polycyclic_system,
    get_ring_heteroatoms,
    get_heteroatom_prefix,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def norbornane():
    """Bicyclo[2.2.1]heptane - classic bicyclic."""
    return Chem.MolFromSmiles("C1CC2CCC1C2")


@pytest.fixture
def adamantane():
    """Tricyclo[3.3.1.13,7]decane - diamond lattice fragment."""
    return Chem.MolFromSmiles("C1C2CC3CC1CC(C2)C3")


@pytest.fixture
def cyclohexane():
    """Simple monocyclic - NOT polycyclic."""
    return Chem.MolFromSmiles("C1CCCCC1")


@pytest.fixture
def bicyclo_222_octane():
    """Bicyclo[2.2.2]octane - symmetric bicyclic."""
    return Chem.MolFromSmiles("C1CC2CCC1CC2")


# ============================================================================
# Tests for get_ring_count()
# ============================================================================

class TestGetRingCount:
    """Test ring count calculation using cycle rank."""

    @pytest.mark.unit
    def test_cyclohexane_one_ring(self, cyclohexane):
        """Cyclohexane has 1 ring."""
        assert get_ring_count(cyclohexane) == 1

    @pytest.mark.unit
    def test_norbornane_two_rings(self, norbornane):
        """Norbornane (bicyclo[2.2.1]heptane) has 2 rings."""
        assert get_ring_count(norbornane) == 2

    @pytest.mark.unit
    def test_adamantane_three_rings(self, adamantane):
        """Adamantane (tricyclo) has 3 rings."""
        assert get_ring_count(adamantane) == 3

    @pytest.mark.unit
    def test_bicyclo_222_two_rings(self, bicyclo_222_octane):
        """Bicyclo[2.2.2]octane has 2 rings."""
        assert get_ring_count(bicyclo_222_octane) == 2

    @pytest.mark.unit
    def test_hexane_no_rings(self):
        """Hexane (acyclic) has 0 rings."""
        mol = Chem.MolFromSmiles("CCCCCC")
        assert get_ring_count(mol) == 0


# ============================================================================
# Tests for classify_bridged_system()
# ============================================================================

class TestClassifyBridgedSystem:
    """Test polycyclic system classification."""

    @pytest.mark.unit
    def test_cyclohexane_not_bridged(self, cyclohexane):
        """Cyclohexane is not a bridged system."""
        assert classify_bridged_system(cyclohexane) is None

    @pytest.mark.unit
    def test_norbornane_is_bicyclo(self, norbornane):
        """Norbornane classifies as bicyclo."""
        assert classify_bridged_system(norbornane) == "bicyclo"

    @pytest.mark.unit
    def test_adamantane_is_tricyclo(self, adamantane):
        """Adamantane classifies as tricyclo."""
        assert classify_bridged_system(adamantane) == "tricyclo"

    @pytest.mark.unit
    def test_bicyclo_222_is_bicyclo(self, bicyclo_222_octane):
        """Bicyclo[2.2.2]octane classifies as bicyclo."""
        assert classify_bridged_system(bicyclo_222_octane) == "bicyclo"


# ============================================================================
# Tests for is_tricyclo_system()
# ============================================================================

class TestIsTricycloSystem:
    """Test tricyclo system detection."""

    @pytest.mark.unit
    def test_adamantane_is_tricyclo(self, adamantane):
        """Adamantane is a tricyclo system."""
        assert is_tricyclo_system(adamantane) is True

    @pytest.mark.unit
    def test_norbornane_not_tricyclo(self, norbornane):
        """Norbornane is not tricyclo (it's bicyclo)."""
        assert is_tricyclo_system(norbornane) is False

    @pytest.mark.unit
    def test_cyclohexane_not_tricyclo(self, cyclohexane):
        """Cyclohexane is not tricyclo."""
        assert is_tricyclo_system(cyclohexane) is False


# ============================================================================
# Tests for find_all_bridgeheads()
# ============================================================================

class TestFindAllBridgeheads:
    """Test bridgehead detection for polycyclic systems."""

    @pytest.mark.unit
    def test_norbornane_two_bridgeheads(self, norbornane):
        """Norbornane has exactly 2 bridgeheads."""
        bridgeheads = find_all_bridgeheads(norbornane)
        assert len(bridgeheads) == 2

    @pytest.mark.unit
    def test_adamantane_four_bridgeheads(self, adamantane):
        """Adamantane has 4 bridgehead atoms."""
        bridgeheads = find_all_bridgeheads(adamantane)
        assert len(bridgeheads) == 4

    @pytest.mark.unit
    def test_cyclohexane_no_bridgeheads(self, cyclohexane):
        """Cyclohexane has no bridgeheads."""
        bridgeheads = find_all_bridgeheads(cyclohexane)
        assert len(bridgeheads) == 0

    @pytest.mark.unit
    def test_bicyclo_222_two_bridgeheads(self, bicyclo_222_octane):
        """Bicyclo[2.2.2]octane has 2 bridgeheads."""
        bridgeheads = find_all_bridgeheads(bicyclo_222_octane)
        assert len(bridgeheads) == 2


# ============================================================================
# Tests for analyze_polycyclic_system()
# ============================================================================

class TestAnalyzePolycyclicSystem:
    """Test complete polycyclic system analysis."""

    @pytest.mark.unit
    def test_norbornane_analysis(self, norbornane):
        """Norbornane should be fully analyzed."""
        info = analyze_polycyclic_system(norbornane)
        assert info is not None
        assert info.system_type == "bicyclo"
        assert info.ring_count == 2
        assert len(info.bridgeheads) == 2

    @pytest.mark.unit
    def test_adamantane_analysis(self, adamantane):
        """Adamantane should be fully analyzed."""
        info = analyze_polycyclic_system(adamantane)
        assert info is not None
        assert info.system_type == "tricyclo"
        assert info.ring_count == 3
        assert len(info.bridgeheads) == 4

    @pytest.mark.unit
    def test_cyclohexane_no_analysis(self, cyclohexane):
        """Cyclohexane should return None."""
        info = analyze_polycyclic_system(cyclohexane)
        assert info is None


# ============================================================================
# Tests for Heteroatom Detection
# ============================================================================

class TestHeteroatomDetection:
    """Test heteroatom detection in ring systems."""

    @pytest.mark.unit
    def test_oxygen_prefix(self):
        """Oxygen should give 'oxa' prefix."""
        assert get_heteroatom_prefix("O") == "oxa"

    @pytest.mark.unit
    def test_nitrogen_prefix(self):
        """Nitrogen should give 'aza' prefix."""
        assert get_heteroatom_prefix("N") == "aza"

    @pytest.mark.unit
    def test_sulfur_prefix(self):
        """Sulfur should give 'thia' prefix."""
        assert get_heteroatom_prefix("S") == "thia"

    @pytest.mark.unit
    def test_quinuclidine_heteroatoms(self):
        """Quinuclidine has nitrogen in ring."""
        mol = Chem.MolFromSmiles("C1CC2CCC1CN2")  # quinuclidine
        ring_atoms = get_ring_atoms(mol)
        heteroatoms = get_ring_heteroatoms(mol, ring_atoms)
        # Should find one nitrogen
        assert len(heteroatoms) == 1
        assert heteroatoms[0][1] == "N"


# ============================================================================
# Integration Tests
# ============================================================================

class TestPolycyclicIntegration:
    """Integration tests for polycyclic detection workflow."""

    @pytest.mark.unit
    def test_adamantane_full_workflow(self, adamantane):
        """Test complete workflow for adamantane."""
        # Classification
        classification = classify_bridged_system(adamantane)
        assert classification == "tricyclo"

        # Bridgeheads
        bridgeheads = find_all_bridgeheads(adamantane)
        assert len(bridgeheads) == 4

        # Ring count
        ring_count = get_ring_count(adamantane)
        assert ring_count == 3

        # Full analysis
        info = analyze_polycyclic_system(adamantane)
        assert info is not None
        assert info.system_type == "tricyclo"

    @pytest.mark.unit
    def test_norbornane_full_workflow(self, norbornane):
        """Test complete workflow for norbornane."""
        # Classification
        classification = classify_bridged_system(norbornane)
        assert classification == "bicyclo"

        # Bridgeheads
        bridgeheads = find_all_bridgeheads(norbornane)
        assert len(bridgeheads) == 2

        # Ring count
        ring_count = get_ring_count(norbornane)
        assert ring_count == 2


# ============================================================================
# Tests for Zero-Length Secondary Bridges (Plan 16-08)
# ============================================================================

from src.orthonym.rules.polycyclic import VonBaeyerAnalyzer, generate_polycyclic_name


def _get_ring_atoms(mol):
    """Helper to get ring atoms from a molecule."""
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)
    return ring_atoms


class TestZeroLengthBridges:
    """Test zero-length secondary bridge detection for highly symmetric polycyclics."""

    @pytest.mark.unit
    def test_cubane_has_correct_bridge_count(self):
        """Cubane must have exactly 6 bridge lengths (ring_count + 1 = 5 + 1 = 6).

        Three of the 6 bridge lengths should be zero (zero-length secondary bridges).
        """
        mol = Chem.MolFromSmiles("C12C3C4C1C5C3C4C25")
        ring_atoms = _get_ring_atoms(mol)

        analyzer = VonBaeyerAnalyzer()
        desc = analyzer.analyze(mol, ring_atoms)

        # ring_count + 1 = 6 total bridge lengths
        assert len(desc.bridge_lengths) == 6, (
            f"Expected 6 bridge lengths for cubane (pentacyclo), got {len(desc.bridge_lengths)}"
        )

        # Exactly 3 zero-length bridges
        zero_count = sum(1 for bl in desc.bridge_lengths if bl == 0)
        assert zero_count == 3, (
            f"Expected 3 zero-length bridges for cubane, got {zero_count}"
        )

    @pytest.mark.unit
    def test_cubane_descriptor_format(self):
        """Cubane descriptor must contain 'pentacyclo[' and zero-length bridge entries.

        The OPSIN-compatible format uses inline superscript locants:
        0locant_low,locant_high (e.g., 02,6 for zero-length bridge between 2 and 6).
        """
        mol = Chem.MolFromSmiles("C12C3C4C1C5C3C4C25")
        ring_atoms = _get_ring_atoms(mol)

        analyzer = VonBaeyerAnalyzer()
        desc = analyzer.analyze(mol, ring_atoms)

        # Must be pentacyclo
        assert desc.descriptor_string.startswith("pentacyclo["), (
            f"Expected pentacyclo prefix, got: {desc.descriptor_string}"
        )

        # Must contain zero-length bridge entries with inline superscript locants
        # OPSIN-compatible format: 0locant_low,locant_high
        import re
        zero_bridge_pattern = r"0\d+,\d+"
        matches = re.findall(zero_bridge_pattern, desc.descriptor_string)
        assert len(matches) == 3, (
            f"Expected 3 zero-length bridge entries in descriptor, found {len(matches)}: "
            f"{desc.descriptor_string}"
        )

    @pytest.mark.unit
    def test_cubane_verification_formula(self):
        """Cubane: sum(bridge_lengths) + 2 must equal total ring atoms (8)."""
        mol = Chem.MolFromSmiles("C12C3C4C1C5C3C4C25")
        ring_atoms = _get_ring_atoms(mol)

        analyzer = VonBaeyerAnalyzer()
        desc = analyzer.analyze(mol, ring_atoms)

        # Verification formula: sum(bridge_lengths) + 2 == total_ring_atoms
        computed = sum(desc.bridge_lengths) + 2
        assert computed == 8, (
            f"Verification failed: sum({desc.bridge_lengths}) + 2 = {computed}, expected 8"
        )
        assert computed == desc.total_atoms

    @pytest.mark.unit
    def test_adamantane_unchanged(self):
        """Adamantane must still produce tricyclo[3.3.1.13,7]decane.

        Regression guard: zero-length bridge detection must not alter systems
        that already had correct secondary bridge detection via unassigned atoms.
        """
        mol = Chem.MolFromSmiles("C1C2CC3CC1CC(C2)C3")

        name = generate_polycyclic_name(mol)
        assert name is not None
        assert "tricyclo[3.3.1.13,7]" in name, (
            f"Expected tricyclo[3.3.1.13,7] in adamantane name, got: {name}"
        )
        assert name.endswith("decane"), (
            f"Expected name to end with 'decane', got: {name}"
        )

    @pytest.mark.unit
    def test_secondary_bridges_from_unassigned_still_work(self):
        """Adamantane secondary bridge via unassigned atoms must still work.

        Adamantane has unassigned atoms forming a secondary bridge of length 1.
        This ensures the Phase 1 (unassigned-atom) processing path is preserved.
        """
        mol = Chem.MolFromSmiles("C1C2CC3CC1CC(C2)C3")
        ring_atoms = _get_ring_atoms(mol)

        analyzer = VonBaeyerAnalyzer()
        desc = analyzer.analyze(mol, ring_atoms)

        # Adamantane: ring_count=3, so (ring_count + 1) = 4 bridge lengths
        assert len(desc.bridge_lengths) == 4, (
            f"Expected 4 bridge lengths for adamantane, got {len(desc.bridge_lengths)}"
        )
        # Verification formula
        assert sum(desc.bridge_lengths) + 2 == desc.total_atoms, (
            f"Verification failed: sum({desc.bridge_lengths}) + 2 != {desc.total_atoms}"
        )
        # The secondary bridge has length 1 (from unassigned atom)
        secondary_bridges = [bi for bi in desc.bridge_info_list if bi.is_secondary]
        assert len(secondary_bridges) == 1
        assert secondary_bridges[0].length == 1

    @pytest.mark.unit
    def test_verification_formula_pentacyclic(self):
        """Pentacyclic lactone: sum(bridge_lengths) + 2 must equal total ring atoms."""
        smiles = "C=C1C[C@]23C[C@@]1(O)CC[C@H]2[C@@]12CC[C@H](O)[C@@](C)(C(=O)O1)[C@H]2[C@@H]3C(=O)O"
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, "Failed to parse pentacyclic lactone SMILES"

        ring_atoms = _get_ring_atoms(mol)
        analyzer = VonBaeyerAnalyzer()

        ring_count = analyzer._get_ring_count(mol, ring_atoms)
        assert ring_count == 5, f"Expected pentacyclic (5 rings), got {ring_count}"

        bridgeheads = analyzer._find_all_bridgeheads(mol, ring_atoms)
        assert len(bridgeheads) >= 2, "Need at least 2 bridgeheads"

        desc = analyzer.analyze(mol, ring_atoms)

        # Verification formula must hold
        computed = sum(desc.bridge_lengths) + 2
        assert computed == desc.total_atoms, (
            f"Verification failed: sum({desc.bridge_lengths}) + 2 = {computed}, "
            f"expected {desc.total_atoms}"
        )
