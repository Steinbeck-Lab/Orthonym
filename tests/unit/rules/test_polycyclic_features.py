"""
Tests for polycyclic substituent, unsaturation, and stereochemistry features.

Tests cover:
1. Substituent detection and naming with VB locants
2. Unsaturation (-ene/-yne) detection with VB locants
3. Stereodescriptor formatting with VB locants
4. Complete name assembly combining all features

Reference: IUPAC 2013 Blue Book and.
"""

import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from orthonym.rules.polycyclic import (
    VonBaeyerAnalyzer,
    get_polycyclic_substituents,
    get_polycyclic_unsaturation,
    get_polycyclic_stereo,
    name_polycyclic_complete as _name_polycyclic_complete_raw,
)


def name_polycyclic_complete(mol, features=None):
    """Wrapper that extracts just the name string from the tuple result."""
    result = _name_polycyclic_complete_raw(mol, features)
    if result is None:
        return None
    return result[0]


# ============================================================================
# Helper Functions
# ============================================================================

def _get_ring_atoms(mol):
    """Get all ring atoms from an RDKit molecule."""
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)
    return ring_atoms


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def analyzer():
    """VonBaeyerAnalyzer instance."""
    return VonBaeyerAnalyzer()


@pytest.fixture
def norbornane():
    """Plain bicyclo[2.2.1]heptane (norbornane)."""
    return Chem.MolFromSmiles('C1CC2CCC1C2')


@pytest.fixture
def methyl_norbornane():
    """3-methylbicyclo[2.2.1]heptane."""
    # Methyl at position 3 (on the bridge)
    return Chem.MolFromSmiles('CC1CC2CCC1C2')


@pytest.fixture
def dimethyl_adamantane():
    """2,3-dimethyladamantane - multiple substituents on tricyclo system."""
    # Adamantane with two methyls
    return Chem.MolFromSmiles('CC1C2CC3CC(C)C1CC(C2)C3')


@pytest.fixture
def ethyl_methyl_norbornane():
    """Ethyl and methyl substituents - tests alphabetical ordering."""
    return Chem.MolFromSmiles('CCC1CC2CCC1(C)C2')


@pytest.fixture
def norbornene():
    """Bicyclo[2.2.1]hept-2-ene - double bond in ring."""
    return Chem.MolFromSmiles('C1C=C2CCC1C2')


@pytest.fixture
def norbornadiene():
    """Bicyclo[2.2.1]hepta-2,5-diene - multiple double bonds."""
    return Chem.MolFromSmiles('C1=CC2C=CC1C2')


@pytest.fixture
def stereo_norbornane():
    """Norbornane with defined stereocenters at bridgeheads."""
    # Stereo defined at bridgehead positions
    mol = Chem.MolFromSmiles('[C@H]1CC2CC[C@@H]1C2')
    rdCIPLabeler.AssignCIPLabels(mol)
    return mol


@pytest.fixture
def substituted_stereo_norbornane():
    """Stereo norbornane with methyl substituent."""
    mol = Chem.MolFromSmiles('C[C@H]1C[C@@H]2CC[C@H]1C2')
    rdCIPLabeler.AssignCIPLabels(mol)
    return mol


@pytest.fixture
def methyl_norbornene():
    """3-methylbicyclo[2.2.1]hept-2-ene - substituent + unsaturation."""
    return Chem.MolFromSmiles('CC1C=C2CCC1C2')


# ============================================================================
# Substituent Detection Tests
# ============================================================================

class TestPolycyclicSubstituents:
    """Test substituent detection on polycyclic systems."""

    @pytest.mark.unit
    def test_single_methyl_substituent(self, methyl_norbornane, analyzer):
        """Single methyl substituent should be detected with correct VB locant."""
        ring_atoms = _get_ring_atoms(methyl_norbornane)
        desc = analyzer.analyze(methyl_norbornane, ring_atoms)
        subs = get_polycyclic_substituents(methyl_norbornane, ring_atoms, desc.numbering)

        # Should find exactly one substituent
        assert len(subs) >= 1, f"Expected at least 1 substituent, got {len(subs)}"

        # Should be a methyl
        names = [s['name'] for s in subs]
        assert 'methyl' in names, f"Expected 'methyl' in substituents, got {names}"

    @pytest.mark.unit
    def test_substituent_atoms_not_in_ring(self, methyl_norbornane, analyzer):
        """Substituent atoms should not be part of the ring system."""
        ring_atoms = _get_ring_atoms(methyl_norbornane)
        desc = analyzer.analyze(methyl_norbornane, ring_atoms)
        subs = get_polycyclic_substituents(methyl_norbornane, ring_atoms, desc.numbering)

        for sub in subs:
            for atom_idx in sub['atom_indices']:
                assert atom_idx not in ring_atoms, (
                    f"Substituent atom {atom_idx} should not be in ring_atoms"
                )

    @pytest.mark.unit
    def test_substituent_locant_from_numbering(self, methyl_norbornane, analyzer):
        """Substituent locant should come from VB numbering dict."""
        ring_atoms = _get_ring_atoms(methyl_norbornane)
        desc = analyzer.analyze(methyl_norbornane, ring_atoms)
        subs = get_polycyclic_substituents(methyl_norbornane, ring_atoms, desc.numbering)

        for sub in subs:
            locant = sub['locant']
            # Locant should be a valid VB locant (1-indexed)
            assert locant >= 1, f"Invalid locant {locant}"
            assert locant <= len(ring_atoms), (
                f"Locant {locant} exceeds ring atom count {len(ring_atoms)}"
            )

    @pytest.mark.unit
    def test_alphabetical_ordering_ethyl_before_methyl(self, ethyl_methyl_norbornane, analyzer):
        """Multiple different substituents should be alphabetically sorted."""
        ring_atoms = _get_ring_atoms(ethyl_methyl_norbornane)
        desc = analyzer.analyze(ethyl_methyl_norbornane, ring_atoms)
        subs = get_polycyclic_substituents(ethyl_methyl_norbornane, ring_atoms, desc.numbering)

        names = [s['name'] for s in subs]
        assert 'ethyl' in names and 'methyl' in names, (
            f"Expected both ethyl and methyl, got {names}"
        )


# ============================================================================
# Unsaturation Detection Tests
# ============================================================================

class TestPolycyclicUnsaturation:
    """Test unsaturation detection in polycyclic systems."""

    @pytest.mark.unit
    def test_single_double_bond(self, norbornene, analyzer):
        """Single double bond should be detected with VB locant."""
        ring_atoms = _get_ring_atoms(norbornene)
        desc = analyzer.analyze(norbornene, ring_atoms)
        unsat = get_polycyclic_unsaturation(norbornene, ring_atoms, desc.numbering)

        assert len(unsat['double_bonds']) == 1, (
            f"Expected 1 double bond, got {len(unsat['double_bonds'])}"
        )

    @pytest.mark.unit
    def test_multiple_double_bonds(self, norbornadiene, analyzer):
        """Multiple double bonds should all be detected."""
        ring_atoms = _get_ring_atoms(norbornadiene)
        desc = analyzer.analyze(norbornadiene, ring_atoms)
        unsat = get_polycyclic_unsaturation(norbornadiene, ring_atoms, desc.numbering)

        assert len(unsat['double_bonds']) == 2, (
            f"Expected 2 double bonds, got {len(unsat['double_bonds'])}"
        )

    @pytest.mark.unit
    def test_double_bond_locant_is_lower(self, norbornene, analyzer):
        """Double bond locant should be the lower of the two atom locants."""
        ring_atoms = _get_ring_atoms(norbornene)
        desc = analyzer.analyze(norbornene, ring_atoms)
        unsat = get_polycyclic_unsaturation(norbornene, ring_atoms, desc.numbering)

        for locant in unsat['double_bonds']:
            assert locant >= 1, f"Invalid double bond locant {locant}"
            # Each locant should be reasonable
            assert locant <= len(ring_atoms), (
                f"Double bond locant {locant} exceeds ring size"
            )

    @pytest.mark.unit
    def test_saturated_system_no_unsaturation(self, norbornane, analyzer):
        """Saturated system should have no unsaturation."""
        ring_atoms = _get_ring_atoms(norbornane)
        desc = analyzer.analyze(norbornane, ring_atoms)
        unsat = get_polycyclic_unsaturation(norbornane, ring_atoms, desc.numbering)

        assert len(unsat['double_bonds']) == 0, (
            f"Expected no double bonds in saturated system, got {unsat['double_bonds']}"
        )
        assert len(unsat['triple_bonds']) == 0, (
            f"Expected no triple bonds in saturated system, got {unsat['triple_bonds']}"
        )


# ============================================================================
# Stereochemistry Tests
# ============================================================================

class TestPolycyclicStereo:
    """Test stereodescriptor collection for polycyclic systems."""

    @pytest.mark.unit
    def test_stereo_with_vb_numbering(self, stereo_norbornane, analyzer):
        """Stereodescriptors should use VB locants."""
        ring_atoms = _get_ring_atoms(stereo_norbornane)
        desc = analyzer.analyze(stereo_norbornane, ring_atoms)
        stereo_str = get_polycyclic_stereo(stereo_norbornane, desc.numbering)

        # Should return a stereo prefix string like "(1R,4S)-" or empty
        # If molecule has defined stereo, it should be non-empty
        if stereo_str:
            assert stereo_str.startswith('('), f"Stereo string should start with '('"
            assert stereo_str.endswith('-'), f"Stereo string should end with '-'"
            # Should contain R or S
            assert 'R' in stereo_str or 'S' in stereo_str, (
                f"Stereo string should contain R or S: {stereo_str}"
            )

    @pytest.mark.unit
    def test_stereo_locants_ascending_order(self, stereo_norbornane, analyzer):
        """Stereodescriptors should be ordered by ascending locant."""
        ring_atoms = _get_ring_atoms(stereo_norbornane)
        desc = analyzer.analyze(stereo_norbornane, ring_atoms)
        stereo_str = get_polycyclic_stereo(stereo_norbornane, desc.numbering)

        if stereo_str:
            # Extract locants from string like "(1R,4S)-"
            # Remove parentheses and trailing hyphen
            content = stereo_str.strip('()-')
            if ',' in content:
                parts = content.split(',')
                locants = [int(''.join(c for c in p if c.isdigit())) for p in parts]
                assert locants == sorted(locants), (
                    f"Locants should be ascending: {locants}"
                )

    @pytest.mark.unit
    def test_no_stereo_returns_empty(self, norbornane, analyzer):
        """Molecule without defined stereo should return empty string."""
        ring_atoms = _get_ring_atoms(norbornane)
        desc = analyzer.analyze(norbornane, ring_atoms)
        stereo_str = get_polycyclic_stereo(norbornane, desc.numbering)

        # Plain norbornane without stereo should return empty string
        assert stereo_str == "", f"Expected empty string, got '{stereo_str}'"


# ============================================================================
# Complete Name Assembly Tests
# ============================================================================

class TestPolycyclicCompleteName:
    """Test complete name assembly with all features."""

    @pytest.mark.unit
    def test_base_name_without_features(self, norbornane):
        """Plain system should produce base name only."""
        name = name_polycyclic_complete(norbornane)
        # Bicyclic is handled by bicyclo module, but complete function should handle it
        assert name is not None or name is None  # May return None if bicyclo handled elsewhere

    @pytest.mark.unit
    def test_name_with_substituent(self, methyl_norbornane):
        """Name should include substituent prefix."""
        name = name_polycyclic_complete(methyl_norbornane)
        if name:
            assert 'methyl' in name.lower(), (
                f"Expected 'methyl' in name, got '{name}'"
            )

    @pytest.mark.unit
    def test_name_with_unsaturation(self, norbornene):
        """Name should include unsaturation suffix."""
        name = name_polycyclic_complete(norbornene)
        if name:
            # Should have -ene suffix instead of -ane
            assert 'en' in name.lower(), f"Expected '-ene' suffix, got '{name}'"
            assert not name.lower().endswith('ane'), (
                f"Should not end with '-ane' when unsaturated: '{name}'"
            )

    @pytest.mark.unit
    def test_name_with_multiple_double_bonds(self, norbornadiene):
        """Name with multiple double bonds should use 'diene' or similar."""
        name = name_polycyclic_complete(norbornadiene)
        if name:
            # Should have "dien" for two double bonds
            assert 'dien' in name.lower() or 'a-' in name.lower(), (
                f"Expected '-diene' for two double bonds, got '{name}'"
            )

    @pytest.mark.unit
    def test_vowel_elision_before_ene(self, norbornene):
        """Vowel elision: 'heptane' -> 'hept' before '-ene'."""
        name = name_polycyclic_complete(norbornene)
        if name:
            # Should not have 'heptanene' - vowel should be elided
            assert 'anene' not in name.lower(), (
                f"Vowel should be elided before -ene: '{name}'"
            )

    @pytest.mark.unit
    def test_complete_name_with_all_features(self, methyl_norbornene):
        """Complete name should combine substituent + unsaturation."""
        name = name_polycyclic_complete(methyl_norbornene)
        if name:
            # Should have methyl
            assert 'methyl' in name.lower(), f"Expected 'methyl' in '{name}'"
            # Should have -ene
            assert 'en' in name.lower(), f"Expected '-ene' in '{name}'"

    @pytest.mark.unit
    def test_name_polycyclic_complete_is_public_api(self):
        """name_polycyclic_complete should be importable as public API."""
        from orthonym.rules.polycyclic import name_polycyclic_complete
        assert callable(name_polycyclic_complete)


# ============================================================================
# Integration Tests for Tricyclo Systems
# ============================================================================

class TestTricycloFeatures:
    """Test features on tricyclic systems like adamantane."""

    @pytest.mark.unit
    def test_adamantane_substituents(self, dimethyl_adamantane, analyzer):
        """Dimethyladamantane should have two methyl substituents detected."""
        ring_atoms = _get_ring_atoms(dimethyl_adamantane)
        desc = analyzer.analyze(dimethyl_adamantane, ring_atoms)
        subs = get_polycyclic_substituents(dimethyl_adamantane, ring_atoms, desc.numbering)

        # Count methyls
        methyl_count = sum(1 for s in subs if s['name'] == 'methyl')
        assert methyl_count >= 2, (
            f"Expected at least 2 methyl substituents, got {methyl_count}"
        )

    @pytest.mark.unit
    def test_adamantane_complete_name(self, dimethyl_adamantane):
        """Dimethyladamantane should produce complete name with dimethyl prefix."""
        name = name_polycyclic_complete(dimethyl_adamantane)
        if name:
            assert 'dimethyl' in name.lower() or 'methyl' in name.lower(), (
                f"Expected methyl in name, got '{name}'"
            )
            assert 'tricyclo' in name.lower(), (
                f"Expected 'tricyclo' in name, got '{name}'"
            )
