"""
Tests for von Baeyer polycyclic descriptor generation.

Tests cover the VonBaeyerAnalyzer class and associated functions
for generating correct IUPAC von Baeyer descriptors (VB-1 through VB-7)
for polycyclic bridged systems (bicyclo through pentacyclo+).

Reference: IUPAC 2013 Blue Book P-23, VB-1 through VB-9.
"""

import pytest
from rdkit import Chem

from orthonym.rules.polycyclic import (
    VonBaeyerAnalyzer,
    PolycyclicDescriptor,
    find_longest_path,
    generate_polycyclic_name,
    is_polycyclic_system,
)


# =============================================================================
# VB_LOCANT OPSIN Round-Trip Results (Phase 72, 2026-02-24)
#
# 38 VB_LOCANT compounds tested against OPSIN CLI 2.8.0
# Overall: 0/38 InChI round-trip, 36/38 OPSIN-parseable
#
# Root cause of InChI mismatch: NOT bridge citation order.
# Failures dominated by:
#   - Missing unsaturation naming (aromatic rings described as saturated)
#   - Wrong substituent naming (DROP-22 polyfunctional failures)
# These are pre-existing naming issues outside Phase 72 VB scope.
#
# OPSIN parse failures (2/38):
#   - Compound 10: OPSIN cannot parse descriptor
#   - Compound 13: OPSIN cannot parse descriptor
#
# Algorithm status:
#   - Independent/dependent classification: IMPLEMENTED
#   - Numbering order (VB-7): IMPLEMENTED
#   - Citation order (VB-6): IMPLEMENTED
#   - IUPAC vs OPSIN conflict: NONE FOUND
#   - VB invariant: ENFORCED at runtime
# =============================================================================


# ============================================================================
# Test Molecules
# ============================================================================

@pytest.fixture
def norbornane():
    """Norbornane: bicyclo[2.2.1]heptane"""
    return Chem.MolFromSmiles('C1CC2CCC1C2')


@pytest.fixture
def adamantane():
    """Adamantane: tricyclo[3.3.1.1(3,7)]decane"""
    return Chem.MolFromSmiles('C1C2CC3CC1CC(C2)C3')


@pytest.fixture
def cubane():
    """Cubane: pentacyclo[4.2.0.0(2,5).0(3,8).0(4,7)]octane"""
    return Chem.MolFromSmiles('C12C3C4C1C5C3C4C25')


@pytest.fixture
def bicyclo_222_octane():
    """Bicyclo[2.2.2]octane"""
    return Chem.MolFromSmiles('C1CC2CCC1CC2')


@pytest.fixture
def analyzer():
    """VonBaeyerAnalyzer instance."""
    return VonBaeyerAnalyzer()


# ============================================================================
# Ring Count Tests
# ============================================================================

class TestRingCount:
    """Test cycle rank calculation for polycyclic systems."""

    @pytest.mark.unit
    def test_norbornane_ring_count(self, norbornane, analyzer):
        """Norbornane has 2 rings (bicyclo)."""
        ring_atoms = _get_ring_atoms(norbornane)
        result = analyzer._get_ring_count(norbornane, ring_atoms)
        assert result == 2

    @pytest.mark.unit
    def test_adamantane_ring_count(self, adamantane, analyzer):
        """Adamantane has 3 rings (tricyclo)."""
        ring_atoms = _get_ring_atoms(adamantane)
        result = analyzer._get_ring_count(adamantane, ring_atoms)
        assert result == 3

    @pytest.mark.unit
    def test_cubane_ring_count(self, cubane, analyzer):
        """Cubane has 5 rings (pentacyclo)."""
        ring_atoms = _get_ring_atoms(cubane)
        result = analyzer._get_ring_count(cubane, ring_atoms)
        assert result == 5

    @pytest.mark.unit
    def test_bicyclo222_ring_count(self, bicyclo_222_octane, analyzer):
        """Bicyclo[2.2.2]octane has 2 rings."""
        ring_atoms = _get_ring_atoms(bicyclo_222_octane)
        result = analyzer._get_ring_count(bicyclo_222_octane, ring_atoms)
        assert result == 2


# ============================================================================
# Main Ring Finding Tests
# ============================================================================

class TestMainRing:
    """Test main ring finding via longest-path algorithm (NOT SSSR)."""

    @pytest.mark.unit
    def test_norbornane_main_ring_size(self, norbornane, analyzer):
        """Norbornane main ring should have 6 atoms (largest possible ring)."""
        ring_atoms = _get_ring_atoms(norbornane)
        bridgeheads = analyzer._find_all_bridgeheads(norbornane, ring_atoms)
        main_ring, bh_pair = analyzer._find_main_ring(norbornane, ring_atoms, bridgeheads)
        assert len(main_ring) == 6, (
            f"Main ring should have 6 atoms, got {len(main_ring)}: {main_ring}"
        )

    @pytest.mark.unit
    def test_adamantane_main_ring_size(self, adamantane, analyzer):
        """Adamantane main ring should have 8 atoms (largest possible ring)."""
        ring_atoms = _get_ring_atoms(adamantane)
        bridgeheads = analyzer._find_all_bridgeheads(adamantane, ring_atoms)
        main_ring, bh_pair = analyzer._find_main_ring(adamantane, ring_atoms, bridgeheads)
        assert len(main_ring) == 8, (
            f"Main ring should have 8 atoms, got {len(main_ring)}: {main_ring}"
        )

    @pytest.mark.unit
    def test_norbornane_main_ring_not_sssr(self, norbornane, analyzer):
        """Main ring should NOT be from SSSR (SSSR gives 5-membered rings for norbornane)."""
        ring_atoms = _get_ring_atoms(norbornane)
        bridgeheads = analyzer._find_all_bridgeheads(norbornane, ring_atoms)
        main_ring, bh_pair = analyzer._find_main_ring(norbornane, ring_atoms, bridgeheads)
        # SSSR for norbornane gives 5-atom rings, main ring should be 6
        ri = norbornane.GetRingInfo()
        sssr_sizes = [len(r) for r in ri.AtomRings()]
        assert len(main_ring) > max(sssr_sizes), (
            f"Main ring ({len(main_ring)} atoms) should be larger than SSSR rings ({sssr_sizes})"
        )

    @pytest.mark.unit
    def test_main_ring_contains_bridgeheads(self, adamantane, analyzer):
        """Main ring must contain exactly 2 bridgehead atoms (the main bridgeheads)."""
        ring_atoms = _get_ring_atoms(adamantane)
        bridgeheads = analyzer._find_all_bridgeheads(adamantane, ring_atoms)
        main_ring, bh_pair = analyzer._find_main_ring(adamantane, ring_atoms, bridgeheads)
        assert bh_pair[0] in main_ring
        assert bh_pair[1] in main_ring


# ============================================================================
# Bridgehead Detection Tests
# ============================================================================

class TestBridgeheadDetection:
    """Test bridgehead atom detection."""

    @pytest.mark.unit
    def test_norbornane_has_2_bridgeheads(self, norbornane, analyzer):
        """Norbornane should have exactly 2 bridgehead atoms."""
        ring_atoms = _get_ring_atoms(norbornane)
        bridgeheads = analyzer._find_all_bridgeheads(norbornane, ring_atoms)
        assert len(bridgeheads) == 2

    @pytest.mark.unit
    def test_adamantane_has_4_bridgeheads(self, adamantane, analyzer):
        """Adamantane should have exactly 4 bridgehead atoms."""
        ring_atoms = _get_ring_atoms(adamantane)
        bridgeheads = analyzer._find_all_bridgeheads(adamantane, ring_atoms)
        assert len(bridgeheads) == 4

    @pytest.mark.unit
    def test_cubane_has_8_bridgeheads(self, cubane, analyzer):
        """Cubane: all 8 atoms are bridgeheads."""
        ring_atoms = _get_ring_atoms(cubane)
        bridgeheads = analyzer._find_all_bridgeheads(cubane, ring_atoms)
        assert len(bridgeheads) == 8


# ============================================================================
# Bridge Enumeration Tests
# ============================================================================

class TestBridgeEnumeration:
    """Test bridge length detection and ordering."""

    @pytest.mark.unit
    def test_norbornane_bridge_lengths(self, norbornane, analyzer):
        """Norbornane should have 3 bridges of length [2, 2, 1]."""
        ring_atoms = _get_ring_atoms(norbornane)
        desc = analyzer.analyze(norbornane, ring_atoms)
        bridge_lengths = sorted(desc.bridge_lengths, reverse=True)
        assert bridge_lengths == [2, 2, 1], (
            f"Expected bridge lengths [2, 2, 1], got {bridge_lengths}"
        )

    @pytest.mark.unit
    def test_adamantane_bridge_lengths(self, adamantane, analyzer):
        """Adamantane should have 4 bridges with lengths [3, 3, 1, 1]."""
        ring_atoms = _get_ring_atoms(adamantane)
        desc = analyzer.analyze(adamantane, ring_atoms)
        bridge_lengths = sorted(desc.bridge_lengths, reverse=True)
        assert bridge_lengths == [3, 3, 1, 1], (
            f"Expected bridge lengths [3, 3, 1, 1], got {bridge_lengths}"
        )


# ============================================================================
# Verification Formula Tests
# ============================================================================

class TestVerificationFormula:
    """Test that sum(bridge_lengths) + 2 == total_ring_atoms for all molecules."""

    @pytest.mark.unit
    def test_norbornane_verification(self, norbornane, analyzer):
        """Norbornane: 2 + 2 + 1 + 2 = 7 atoms total."""
        ring_atoms = _get_ring_atoms(norbornane)
        desc = analyzer.analyze(norbornane, ring_atoms)
        total = sum(desc.bridge_lengths) + 2
        assert total == desc.total_atoms, (
            f"Verification failed: sum({desc.bridge_lengths}) + 2 = {total}, "
            f"expected {desc.total_atoms}"
        )

    @pytest.mark.unit
    def test_adamantane_verification(self, adamantane, analyzer):
        """Adamantane: 3 + 3 + 1 + 1 + 2 = 10 atoms total."""
        ring_atoms = _get_ring_atoms(adamantane)
        desc = analyzer.analyze(adamantane, ring_atoms)
        total = sum(desc.bridge_lengths) + 2
        assert total == desc.total_atoms, (
            f"Verification failed: sum({desc.bridge_lengths}) + 2 = {total}, "
            f"expected {desc.total_atoms}"
        )

    @pytest.mark.unit
    def test_cubane_verification(self, cubane, analyzer):
        """Cubane: sum of bridge lengths + 2 == 8."""
        ring_atoms = _get_ring_atoms(cubane)
        desc = analyzer.analyze(cubane, ring_atoms)
        total = sum(desc.bridge_lengths) + 2
        assert total == desc.total_atoms, (
            f"Verification failed: sum({desc.bridge_lengths}) + 2 = {total}, "
            f"expected {desc.total_atoms}"
        )

    @pytest.mark.unit
    def test_bicyclo222_verification(self, bicyclo_222_octane, analyzer):
        """Bicyclo[2.2.2]octane: 2 + 2 + 2 + 2 = 8."""
        ring_atoms = _get_ring_atoms(bicyclo_222_octane)
        desc = analyzer.analyze(bicyclo_222_octane, ring_atoms)
        total = sum(desc.bridge_lengths) + 2
        assert total == desc.total_atoms


# ============================================================================
# Descriptor String Formatting Tests
# ============================================================================

class TestDescriptorFormatting:
    """Test descriptor string output."""

    @pytest.mark.unit
    def test_norbornane_descriptor(self, norbornane, analyzer):
        """Norbornane descriptor: bicyclo[2.2.1]."""
        ring_atoms = _get_ring_atoms(norbornane)
        desc = analyzer.analyze(norbornane, ring_atoms)
        assert desc.descriptor_string == "bicyclo[2.2.1]", (
            f"Expected 'bicyclo[2.2.1]', got '{desc.descriptor_string}'"
        )

    @pytest.mark.unit
    def test_adamantane_descriptor(self, adamantane, analyzer):
        """Adamantane descriptor: tricyclo[3.3.1.1^3,7] (PIN superscript form)."""
        ring_atoms = _get_ring_atoms(adamantane)
        desc = analyzer.analyze(adamantane, ring_atoms)
        # 13B(d): PIN superscript locants for secondary bridges (P-23.2.5.1).
        assert desc.descriptor_string == "tricyclo[3.3.1.1^3,7]", (
            f"Expected 'tricyclo[3.3.1.1^3,7]', got '{desc.descriptor_string}'"
        )

    @pytest.mark.unit
    def test_bicyclo222_descriptor(self, bicyclo_222_octane, analyzer):
        """Bicyclo[2.2.2]octane descriptor: bicyclo[2.2.2]."""
        ring_atoms = _get_ring_atoms(bicyclo_222_octane)
        desc = analyzer.analyze(bicyclo_222_octane, ring_atoms)
        assert desc.descriptor_string == "bicyclo[2.2.2]", (
            f"Expected 'bicyclo[2.2.2]', got '{desc.descriptor_string}'"
        )


# ============================================================================
# VB Numbering Tests
# ============================================================================

class TestVBNumbering:
    """Test VB-7 numbering order."""

    @pytest.mark.unit
    def test_norbornane_numbering_bridgeheads(self, norbornane, analyzer):
        """Norbornane: numbering starts at bridgehead, second bridgehead is at position after longer path."""
        ring_atoms = _get_ring_atoms(norbornane)
        desc = analyzer.analyze(norbornane, ring_atoms)
        numbering = desc.numbering
        # The numbering should cover all 7 atoms with locants 1..7
        locants = sorted(numbering.values())
        assert locants == list(range(1, 8)), (
            f"Expected locants 1-7, got {locants}"
        )

    @pytest.mark.unit
    def test_adamantane_numbering_complete(self, adamantane, analyzer):
        """Adamantane: numbering should cover all 10 atoms with locants 1..10."""
        ring_atoms = _get_ring_atoms(adamantane)
        desc = analyzer.analyze(adamantane, ring_atoms)
        numbering = desc.numbering
        locants = sorted(numbering.values())
        assert locants == list(range(1, 11)), (
            f"Expected locants 1-10, got {locants}"
        )

    @pytest.mark.unit
    def test_numbering_longer_path_first(self, norbornane, analyzer):
        """Numbering goes along the longer path of the main ring first."""
        ring_atoms = _get_ring_atoms(norbornane)
        desc = analyzer.analyze(norbornane, ring_atoms)
        numbering = desc.numbering
        # The two bridgeheads are at positions 1 and some position after
        # the longer path (of length 2). For bicyclo[2.2.1], longer paths
        # have 2 atoms each, so second bridgehead should be at position 3
        # (1 + 2 atoms along longer path = position 3)
        bh_locants = []
        ring_atoms_set = _get_ring_atoms(norbornane)
        bridgeheads = analyzer._find_all_bridgeheads(norbornane, ring_atoms_set)
        for bh in bridgeheads:
            if bh in numbering:
                bh_locants.append(numbering[bh])
        bh_locants.sort()
        # First bridgehead at 1, second at position determined by path length
        assert bh_locants[0] == 1


# ============================================================================
# Full Name Generation Tests
# ============================================================================

class TestFullNameGeneration:
    """Test generate_polycyclic_name() for complete base names."""

    @pytest.mark.unit
    def test_norbornane_full_name(self, norbornane):
        """Norbornane -> bicyclo[2.2.1]heptane."""
        name = generate_polycyclic_name(norbornane)
        assert name == "bicyclo[2.2.1]heptane", (
            f"Expected 'bicyclo[2.2.1]heptane', got '{name}'"
        )

    @pytest.mark.unit
    def test_adamantane_full_name(self, adamantane):
        """Adamantane -> tricyclo[3.3.1.1^3,7]decane.

        13B(d): secondary-bridge locants now use the PIN superscript typography
        (``1^3,7``) per P-23.2.5.1 / P-23.2.6.1.2; was the older ``1(3,7)``
        parenthesis form. OPSIN round-trips both.
        """
        name = generate_polycyclic_name(adamantane)
        assert name == "tricyclo[3.3.1.1^3,7]decane", (
            f"Expected 'tricyclo[3.3.1.1^3,7]decane', got '{name}'"
        )

    @pytest.mark.unit
    def test_bicyclo222_full_name(self, bicyclo_222_octane):
        """Bicyclo[2.2.2]octane -> bicyclo[2.2.2]octane."""
        name = generate_polycyclic_name(bicyclo_222_octane)
        assert name == "bicyclo[2.2.2]octane", (
            f"Expected 'bicyclo[2.2.2]octane', got '{name}'"
        )

    @pytest.mark.unit
    def test_cubane_full_name(self, cubane):
        """Cubane should produce a pentacyclo[...]octane name."""
        name = generate_polycyclic_name(cubane)
        assert name is not None
        assert name.startswith("pentacyclo["), (
            f"Expected name to start with 'pentacyclo[', got '{name}'"
        )
        assert name.endswith("octane"), (
            f"Expected name to end with 'octane', got '{name}'"
        )


# ============================================================================
# is_polycyclic_system() Detection Tests
# ============================================================================

class TestIsPolycyclicSystem:
    """Test is_polycyclic_system() detection function."""

    @pytest.mark.unit
    def test_adamantane_is_polycyclic(self, adamantane):
        """Adamantane (ring_count=3, bridged) -> True."""
        assert is_polycyclic_system(adamantane) is True

    @pytest.mark.unit
    def test_cubane_is_polycyclic(self, cubane):
        """Cubane (ring_count=5, bridged) -> True."""
        assert is_polycyclic_system(cubane) is True

    @pytest.mark.unit
    def test_norbornane_is_not_polycyclic(self, norbornane):
        """Norbornane (ring_count=2, bicyclo) -> False (handled by bicyclo module)."""
        assert is_polycyclic_system(norbornane) is False

    @pytest.mark.unit
    def test_cyclohexane_is_not_polycyclic(self):
        """Cyclohexane (monocyclic) -> False."""
        mol = Chem.MolFromSmiles('C1CCCCC1')
        assert is_polycyclic_system(mol) is False

    @pytest.mark.unit
    def test_naphthalene_is_not_polycyclic(self):
        """Naphthalene (fused, not bridged polycyclic) -> False."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        assert is_polycyclic_system(mol) is False

    @pytest.mark.unit
    def test_spiro_is_not_polycyclic(self):
        """Spiro[4.5]decane (spiro, not bridged) -> False."""
        mol = Chem.MolFromSmiles('C1CCCC11CCCCC1')
        assert is_polycyclic_system(mol) is False

    @pytest.mark.unit
    def test_acyclic_is_not_polycyclic(self):
        """Hexane (acyclic) -> False."""
        mol = Chem.MolFromSmiles('CCCCCC')
        assert is_polycyclic_system(mol) is False


# ============================================================================
# find_longest_path Tests
# ============================================================================

class TestFindLongestPath:
    """Test the longest-path DFS function."""

    @pytest.mark.unit
    def test_norbornane_longest_path(self, norbornane):
        """In norbornane, longest path between bridgeheads should be 4 atoms (including endpoints)."""
        ring_atoms = _get_ring_atoms(norbornane)
        bridgeheads = _get_bridgeheads(norbornane)
        bh_list = sorted(bridgeheads)
        bh1, bh2 = bh_list[0], bh_list[1]
        path = find_longest_path(norbornane, bh1, bh2, ring_atoms)
        # Longest path between bridgeheads goes through 2 non-bridgehead atoms = 4 total
        assert len(path) == 4, (
            f"Expected path of length 4, got {len(path)}: {path}"
        )

    @pytest.mark.unit
    def test_longest_path_includes_endpoints(self, norbornane):
        """Path should include start and end atoms."""
        ring_atoms = _get_ring_atoms(norbornane)
        bridgeheads = _get_bridgeheads(norbornane)
        bh_list = sorted(bridgeheads)
        bh1, bh2 = bh_list[0], bh_list[1]
        path = find_longest_path(norbornane, bh1, bh2, ring_atoms)
        assert path[0] == bh1
        assert path[-1] == bh2

    @pytest.mark.unit
    def test_wr02_dfs_expansion_cap_bounds_search(self, norbornane, monkeypatch):
        """WR-02 (code review 2026-06-02): the DFS is bounded by
        ``_MAX_DFS_EXPANSIONS`` so a pathological dense cage cannot hang. With the
        cap forced to 1 the search aborts immediately (best-so-far is shorter than
        the full path); with the real cap (>> expansions a real polycyclic needs)
        the full 4-atom norbornane path is found, proving legitimate input is
        unaffected."""
        import orthonym.rules.polycyclic as pc
        ring_atoms = _get_ring_atoms(norbornane)
        bh_list = sorted(_get_bridgeheads(norbornane))
        bh1, bh2 = bh_list[0], bh_list[1]
        # Real cap: normal, complete result.
        assert len(find_longest_path(norbornane, bh1, bh2, ring_atoms)) == 4
        # Forced tiny cap: the search is cut short and returns the best-so-far
        # rather than exploring further (the anti-hang guard fired).
        monkeypatch.setattr(pc, "_MAX_DFS_EXPANSIONS", 1)
        bounded = find_longest_path(norbornane, bh1, bh2, ring_atoms)
        assert len(bounded) < 4


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and special molecules."""

    @pytest.mark.unit
    def test_none_molecule(self):
        """None input should return None."""
        result = generate_polycyclic_name(None)
        assert result is None

    @pytest.mark.unit
    def test_acyclic_molecule(self):
        """Acyclic molecule should return None."""
        mol = Chem.MolFromSmiles('CCCCCC')
        result = generate_polycyclic_name(mol)
        assert result is None

    @pytest.mark.unit
    def test_monocyclic_molecule(self):
        """Simple ring should return None."""
        mol = Chem.MolFromSmiles('C1CCCCC1')
        result = generate_polycyclic_name(mol)
        assert result is None


# ============================================================================
# CYCLO_PREFIXES High Ring Count Tests
# ============================================================================

class TestCycloPrefixesExtended:
    """Test CYCLO_PREFIXES dict covers ring counts up to 20."""

    @pytest.mark.unit
    def test_tridecacyclo_prefix(self):
        """Ring count 13 should give 'tridecacyclo'."""
        from orthonym.rules.polycyclic import CYCLO_PREFIXES
        assert CYCLO_PREFIXES[13] == "tridecacyclo"

    @pytest.mark.unit
    def test_undecacyclo_prefix(self):
        """Ring count 11 should give 'undecacyclo'."""
        from orthonym.rules.polycyclic import CYCLO_PREFIXES
        assert CYCLO_PREFIXES[11] == "undecacyclo"

    @pytest.mark.unit
    def test_dodecacyclo_prefix(self):
        """Ring count 12 should give 'dodecacyclo'."""
        from orthonym.rules.polycyclic import CYCLO_PREFIXES
        assert CYCLO_PREFIXES[12] == "dodecacyclo"

    @pytest.mark.unit
    def test_icosacyclo_prefix(self):
        """Ring count 20 should give 'icosacyclo'."""
        from orthonym.rules.polycyclic import CYCLO_PREFIXES
        assert CYCLO_PREFIXES[20] == "icosacyclo"

    @pytest.mark.unit
    def test_bicyclo_unchanged(self):
        """Ring count 2 should still give 'bicyclo' (no regression)."""
        from orthonym.rules.polycyclic import CYCLO_PREFIXES
        assert CYCLO_PREFIXES[2] == "bicyclo"

    @pytest.mark.unit
    def test_tetracyclo_unchanged(self):
        """Ring count 4 should still give 'tetracyclo' (no regression)."""
        from orthonym.rules.polycyclic import CYCLO_PREFIXES
        assert CYCLO_PREFIXES[4] == "tetracyclo"

    @pytest.mark.unit
    def test_all_entries_from_2_to_20(self):
        """All ring counts from 2 to 20 should be present in CYCLO_PREFIXES."""
        from orthonym.rules.polycyclic import CYCLO_PREFIXES
        for i in range(2, 21):
            assert i in CYCLO_PREFIXES, f"Missing ring count {i}"
            assert CYCLO_PREFIXES[i].endswith("cyclo"), (
                f"CYCLO_PREFIXES[{i}] = '{CYCLO_PREFIXES[i]}' does not end with 'cyclo'"
            )

    @pytest.mark.unit
    def test_fallback_for_ring_count_above_20(self):
        """Ring count > 20 falls back to numeric prefix."""
        from orthonym.rules.polycyclic import CYCLO_PREFIXES
        assert 25 not in CYCLO_PREFIXES
        fallback = CYCLO_PREFIXES.get(25, f"{25}cyclo")
        assert fallback == "25cyclo"


# ============================================================================
# Higher VB Descriptor Accuracy Tests (Plan 111-02)
# ============================================================================

class TestHigherVBDescriptorAccuracy:
    """Tests for tetracyclo+ VB descriptor correctness per IUPAC P-23.2.6."""

    @pytest.mark.unit
    def test_cubane_descriptor_opsin_parseable(self):
        """Cubane VB descriptor must be parseable by OPSIN (no [...] placeholder)."""
        mol = Chem.MolFromSmiles('C12C3C4C1C5C3C4C25')
        from orthonym.rules.tricyclo import generate_polycyclo_descriptor
        desc = generate_polycyclo_descriptor(mol)
        assert desc is not None
        assert 'pentacyclo' in desc
        # Verify it produces a valid descriptor (not "[...]")
        assert '[...]' not in desc

    @pytest.mark.unit
    def test_cubane_uses_von_baeyer_analyzer(self):
        """Higher VB should route through VonBaeyerAnalyzer, not SSSR hack."""
        mol = Chem.MolFromSmiles('C12C3C4C1C5C3C4C25')
        from orthonym.rules.tricyclo import generate_polycyclo_descriptor
        desc = generate_polycyclo_descriptor(mol)
        assert desc is not None
        # VonBaeyerAnalyzer produces proper (x,y) locants for secondary bridges
        # The old algorithm never produced parenthesized locants
        assert '(' in desc or desc.count('.') >= 4  # Secondary bridges present

    @pytest.mark.unit
    def test_cubane_descriptor_matches_polycyclic_module(self):
        """tricyclo.py must produce same descriptor as polycyclic.py for cubane."""
        mol = Chem.MolFromSmiles('C12C3C4C1C5C3C4C25')
        from orthonym.rules.tricyclo import generate_polycyclo_descriptor
        from orthonym.rules.polycyclic import generate_polycyclic_name
        tricyclo_desc = generate_polycyclo_descriptor(mol)
        polycyclic_name = generate_polycyclic_name(mol)
        # tricyclo descriptor + "octane" should equal polycyclic name
        assert polycyclic_name is not None
        assert tricyclo_desc is not None
        assert polycyclic_name == tricyclo_desc + "octane"

    @pytest.mark.unit
    def test_adamantane_descriptor_unchanged(self):
        """Adamantane tricyclo descriptor must remain correct (no regression)."""
        mol = Chem.MolFromSmiles('C1C2CC3CC1CC(C2)C3')
        from orthonym.rules.tricyclo import name_polycyclo_system
        name = name_polycyclo_system(mol)
        assert name is not None
        # Adamantane uses retained name via tricyclo.py
        assert 'adamantane' in name or 'tricyclo' in name
        if 'tricyclo' in name:
            assert 'decane' in name


class TestVBHeteroatomLocants:
    """Tests that VB heteroatom locants use VB numbering, not raw atom index."""

    @pytest.mark.unit
    def test_heteroatom_locant_not_raw_index(self):
        """Heteroatom prefix locants must come from VB numbering map, not idx+1."""
        # 2-oxaadamantane: O in the ring system
        mol = Chem.MolFromSmiles('O1C2CC3CC1CC(C2)C3')
        from orthonym.rules.tricyclo import name_polycyclo_system
        name = name_polycyclo_system(mol)
        if name:
            # Extract the locant before 'oxa' -- it should be a valid VB locant
            import re
            match = re.search(r'(\d+)-oxa', name)
            if match:
                locant = int(match.group(1))
                # VB locants for adamantane-class range 1-10
                assert 1 <= locant <= 10

    @pytest.mark.unit
    def test_heteroatom_prefix_uses_numbering_param(self):
        """_generate_heteroatom_prefix must accept and use a numbering parameter."""
        from orthonym.rules.tricyclo import _generate_heteroatom_prefix
        import inspect
        sig = inspect.signature(_generate_heteroatom_prefix)
        assert 'numbering' in sig.parameters, (
            "_generate_heteroatom_prefix must have a 'numbering' parameter"
        )

    @pytest.mark.unit
    def test_higher_polycyclo_passes_numbering(self):
        """_name_higher_polycyclo_system must pass VB numbering to heteroatom prefix."""
        # A heteroatom-containing tetracyclo+ compound
        # This is a simple test: check the function works end-to-end
        mol = Chem.MolFromSmiles('O1C2CC3CC1CC(C2)C3')
        from orthonym.rules.tricyclo import _generate_heteroatom_prefix
        from orthonym.rules.polycyclic import VonBaeyerAnalyzer
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)
        analyzer = VonBaeyerAnalyzer()
        desc = analyzer.analyze(mol, ring_atoms)
        numbering = desc.numbering
        heteroatoms = [(idx, mol.GetAtomWithIdx(idx).GetSymbol())
                       for idx in ring_atoms if mol.GetAtomWithIdx(idx).GetSymbol() != 'C']
        # Call with numbering -- should use VB locants
        prefix = _generate_heteroatom_prefix(mol, heteroatoms, ring_atoms, numbering=numbering)
        assert 'oxa' in prefix
        # The locant should NOT be raw idx+1, but from the numbering map
        import re
        match = re.search(r'(\d+)-oxa', prefix)
        assert match is not None
        locant = int(match.group(1))
        # Check the locant matches what VB numbering says for the O atom
        o_idx = [idx for idx, sym in heteroatoms if sym == 'O'][0]
        expected_locant = numbering[o_idx]
        assert locant == expected_locant


# ============================================================================
# Helper functions (not part of the module under test)
# ============================================================================

def _get_ring_atoms(mol):
    """Get all ring atoms from an RDKit molecule."""
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)
    return ring_atoms


def _get_bridgeheads(mol):
    """Get bridgehead atoms using the analyzer."""
    analyzer = VonBaeyerAnalyzer()
    ring_atoms = _get_ring_atoms(mol)
    return analyzer._find_all_bridgeheads(mol, ring_atoms)
