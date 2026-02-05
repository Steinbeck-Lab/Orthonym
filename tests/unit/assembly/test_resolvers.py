"""
Unit tests for orthonym.assembly.resolvers module.

Tests resolver functions independently:
- resolve_parent: parent type classification
- resolve_suffix: suffix text, locants, and multiplier resolution
- apply_ion_suffix_modification: ion suffix transforms
"""

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym, MolecularFeatures
from orthonym.assembly.resolvers import (
    ParentInfo,
    SuffixInfo,
    resolve_parent,
    resolve_suffix,
    apply_ion_suffix_modification,
)


# ============================================================================
# Helper to create features from SMILES
# ============================================================================

def _make_features(smiles: str):
    """Create MolecularFeatures from SMILES using the full namer pipeline."""
    namer = Orthonym()
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    canonical = Chem.MolToSmiles(mol, canonical=True)
    features = namer._perceive(mol, smiles, canonical)
    namer._classify(features)
    return features, mol


# ============================================================================
# TestResolveParent
# ============================================================================

@pytest.mark.unit
class TestResolveParent:
    """Tests for resolve_parent() parent type classification."""

    def test_simple_chain_butane(self):
        """Butane: parent_type='chain', atom_count=4."""
        features, mol = _make_features("CCCC")
        pi = resolve_parent(features, mol)
        assert pi.parent_type == "chain"
        assert pi.atom_count == 4
        assert pi.parent_label == "but"
        assert pi.is_named_ring is False

    def test_benzene(self):
        """Benzene: parent_type='benzene', atom_count=6, is_named_ring=True."""
        features, mol = _make_features("c1ccccc1")
        pi = resolve_parent(features, mol)
        assert pi.parent_type == "benzene"
        assert pi.atom_count == 6
        assert pi.parent_label == "benzene"
        assert pi.is_named_ring is True

    def test_cyclohexane(self):
        """Cyclohexane: parent_type='ring', atom_count=6, is_named_ring=False."""
        features, mol = _make_features("C1CCCCC1")
        pi = resolve_parent(features, mol)
        assert pi.parent_type == "ring"
        assert pi.atom_count == 6
        assert pi.parent_label == "cyclohex"
        assert pi.is_named_ring is False

    def test_pyridine(self):
        """Pyridine: parent_type='ring' (heterocyclic), atom_count=6."""
        features, mol = _make_features("c1ccncc1")
        pi = resolve_parent(features, mol)
        assert pi.parent_type == "ring"
        assert pi.atom_count == 6
        assert pi.is_named_ring is False

    def test_naphthalene(self):
        """Naphthalene: parent_type='polycyclic_aromatic', is_named_ring=True."""
        features, mol = _make_features("c1ccc2ccccc2c1")
        pi = resolve_parent(features, mol)
        assert pi.parent_type == "polycyclic_aromatic"
        assert pi.parent_label == "naphthalene"
        assert pi.atom_count == 10
        assert pi.is_named_ring is True

    def test_propanol(self):
        """Propan-1-ol: parent_type='chain', atom_count=3."""
        features, mol = _make_features("CCCO")
        pi = resolve_parent(features, mol)
        assert pi.parent_type == "chain"
        assert pi.atom_count == 3
        assert pi.parent_label == "prop"

    def test_methane_single_atom(self):
        """Methane: parent_type='chain', atom_count=1."""
        features, mol = _make_features("C")
        pi = resolve_parent(features, mol)
        assert pi.parent_type == "chain"
        assert pi.atom_count == 1
        assert pi.parent_label == "meth"
        assert pi.is_named_ring is False

    def test_cyclohexanone(self):
        """Cyclohexanone: parent_type='ring', atom_count=6."""
        features, mol = _make_features("O=C1CCCCC1")
        pi = resolve_parent(features, mol)
        assert pi.parent_type == "ring"
        assert pi.atom_count == 6

    def test_bicyclo_system(self):
        """Norbornane: parent_type='complex_ring'."""
        features, mol = _make_features("C1CC2CC1CC2")
        pi = resolve_parent(features, mol)
        assert pi.parent_type in ("complex_ring", "fused_heterocycle")
        assert pi.atom_count > 0

    def test_atom_to_locant_populated_for_chain(self):
        """Chain compounds should have non-empty atom_to_locant."""
        features, mol = _make_features("CCCCC")  # pentane
        pi = resolve_parent(features, mol)
        assert pi.parent_type == "chain"
        assert len(pi.atom_to_locant) > 0

    def test_atom_to_locant_populated_for_ring(self):
        """Ring compounds should have non-empty atom_to_locant (when oriented)."""
        features, mol = _make_features("C1CCCCC1")  # cyclohexane
        pi = resolve_parent(features, mol)
        # Cyclohexane should have oriented ring -> locant map
        assert pi.parent_type == "ring"
        assert len(pi.atom_to_locant) > 0

    def test_cyclopentane(self):
        """Cyclopentane: parent_type='ring', atom_count=5."""
        features, mol = _make_features("C1CCCC1")
        pi = resolve_parent(features, mol)
        assert pi.parent_type == "ring"
        assert pi.atom_count == 5
        assert pi.parent_label == "cyclopent"

    def test_ethane(self):
        """Ethane: parent_type='chain', atom_count=2."""
        features, mol = _make_features("CC")
        pi = resolve_parent(features, mol)
        assert pi.parent_type == "chain"
        assert pi.atom_count == 2
        assert pi.parent_label == "eth"

    def test_indole_fused_heterocycle(self):
        """Indole: parent_type should be complex_ring or fused_heterocycle."""
        features, mol = _make_features("c1ccc2[nH]ccc2c1")
        pi = resolve_parent(features, mol)
        # Indole is a complex ring system (fused heterocycle)
        assert pi.parent_type in ("complex_ring", "fused_heterocycle")
        assert pi.is_named_ring is True
        assert pi.atom_count > 0


# ============================================================================
# TestResolveSuffix
# ============================================================================

@pytest.mark.unit
class TestResolveSuffix:
    """Tests for resolve_suffix() suffix resolution."""

    def test_alcohol_suffix(self):
        """Propan-1-ol: suffix text contains 'ol', count=1."""
        features, mol = _make_features("CCCO")
        pi = resolve_parent(features, mol)
        si = resolve_suffix(features, pi)
        assert "ol" in si.text
        assert si.count >= 1
        assert si.is_terminal is False

    def test_diol_suffix(self):
        """Ethane-1,2-diol: suffix text contains 'ol', count=2."""
        features, mol = _make_features("OCCO")
        pi = resolve_parent(features, mol)
        si = resolve_suffix(features, pi)
        assert "ol" in si.text
        assert si.count == 2

    def test_carboxylic_acid_suffix(self):
        """Propanoic acid: suffix text contains 'oic acid', is_terminal=True."""
        features, mol = _make_features("CCC(=O)O")
        pi = resolve_parent(features, mol)
        si = resolve_suffix(features, pi)
        assert "oic acid" in si.text
        assert si.is_terminal is True

    def test_ketone_suffix(self):
        """Propan-2-one (acetone): suffix text contains 'one'."""
        features, mol = _make_features("CC(=O)C")
        pi = resolve_parent(features, mol)
        si = resolve_suffix(features, pi)
        assert "one" in si.text
        assert si.is_terminal is False

    def test_aldehyde_suffix(self):
        """Propanal: suffix text contains 'al', is_terminal=True."""
        features, mol = _make_features("CCC=O")
        pi = resolve_parent(features, mol)
        si = resolve_suffix(features, pi)
        assert "al" in si.text
        assert si.is_terminal is True

    def test_no_principal_group(self):
        """Butane: no suffix (no principal group)."""
        features, mol = _make_features("CCCC")
        pi = resolve_parent(features, mol)
        si = resolve_suffix(features, pi)
        assert si.text == ""
        assert si.count == 0

    def test_suffix_locants_within_parent_capacity(self):
        """Suffix locant values should not exceed parent atom count."""
        features, mol = _make_features("CCCO")  # propanol
        pi = resolve_parent(features, mol)
        si = resolve_suffix(features, pi)
        for loc in si.locants:
            assert loc <= pi.atom_count

    def test_amine_suffix(self):
        """Propan-1-amine: suffix text contains 'amine'."""
        features, mol = _make_features("CCCN")
        pi = resolve_parent(features, mol)
        si = resolve_suffix(features, pi)
        assert "amine" in si.text
        assert si.count >= 1

    def test_nitrile_suffix(self):
        """Propanenitrile: suffix is_terminal=True."""
        features, mol = _make_features("CCC#N")
        pi = resolve_parent(features, mol)
        si = resolve_suffix(features, pi)
        assert si.is_terminal is True

    def test_thiol_suffix(self):
        """Ethanethiol: suffix text contains 'thiol'."""
        features, mol = _make_features("CCS")
        pi = resolve_parent(features, mol)
        si = resolve_suffix(features, pi)
        assert "thiol" in si.text


# ============================================================================
# TestApplyIonSuffixModification
# ============================================================================

@pytest.mark.unit
class TestApplyIonSuffixModification:
    """Tests for apply_ion_suffix_modification() ion suffix transforms."""

    def _make_mock_features(self, species_type='neutral', total_charge=0):
        """Create a minimal mock features object."""
        features = MolecularFeatures(mol=None)
        features.species_type = species_type
        features.total_charge = total_charge
        return features

    def test_anion_alcohol_to_olate(self):
        """Anion on alcohol: '-ol' -> '-olate'."""
        si = SuffixInfo(text="ol", locants=[1], count=1, is_terminal=False)
        features = self._make_mock_features(species_type='ion', total_charge=-1)
        result = apply_ion_suffix_modification(si, features)
        assert result.text == "olate"
        assert result.locants == [1]

    def test_anion_acid_to_oate(self):
        """Anion on acid: '-oic acid' -> '-oate'."""
        si = SuffixInfo(text="oic acid", locants=[], count=1, is_terminal=True)
        features = self._make_mock_features(species_type='ion', total_charge=-1)
        result = apply_ion_suffix_modification(si, features)
        assert result.text == "oate"

    def test_anion_no_suffix_to_ide(self):
        """Anion with no suffix: '' -> '-ide'."""
        si = SuffixInfo(text="", locants=[], count=0, is_terminal=False)
        features = self._make_mock_features(species_type='ion', total_charge=-1)
        result = apply_ion_suffix_modification(si, features)
        assert result.text == "ide"

    def test_cation_amine_to_aminium(self):
        """Cation on amine: '-amine' -> '-aminium'."""
        si = SuffixInfo(text="amine", locants=[1], count=1, is_terminal=False)
        features = self._make_mock_features(species_type='ion', total_charge=1)
        result = apply_ion_suffix_modification(si, features)
        assert result.text == "aminium"

    def test_cation_no_suffix_to_ium(self):
        """Cation with no suffix: '' -> '-ium'."""
        si = SuffixInfo(text="", locants=[], count=0, is_terminal=False)
        features = self._make_mock_features(species_type='ion', total_charge=1)
        result = apply_ion_suffix_modification(si, features)
        assert result.text == "ium"

    def test_neutral_unchanged(self):
        """Neutral species: suffix unchanged."""
        si = SuffixInfo(text="ol", locants=[1], count=1, is_terminal=False)
        features = self._make_mock_features(species_type='neutral', total_charge=0)
        result = apply_ion_suffix_modification(si, features)
        assert result.text == "ol"
        assert result.locants == [1]

    def test_salt_unchanged(self):
        """Salt species: suffix unchanged (salts use functional class naming)."""
        si = SuffixInfo(text="ol", locants=[1], count=1, is_terminal=False)
        features = self._make_mock_features(species_type='salt', total_charge=0)
        result = apply_ion_suffix_modification(si, features)
        assert result.text == "ol"

    def test_zwitterion_unchanged(self):
        """Zwitterion species: suffix unchanged."""
        si = SuffixInfo(text="oic acid", locants=[], count=1, is_terminal=True)
        features = self._make_mock_features(species_type='zwitterion', total_charge=0)
        result = apply_ion_suffix_modification(si, features)
        assert result.text == "oic acid"

    def test_radical_unchanged(self):
        """Radical species: suffix unchanged (radicals handled separately)."""
        si = SuffixInfo(text="ol", locants=[1], count=1, is_terminal=False)
        features = self._make_mock_features(species_type='radical', total_charge=0)
        result = apply_ion_suffix_modification(si, features)
        assert result.text == "ol"

    def test_anion_thiol_to_thiolate(self):
        """Anion on thiol: '-thiol' -> '-thiolate'."""
        si = SuffixInfo(text="thiol", locants=[1], count=1, is_terminal=False)
        features = self._make_mock_features(species_type='ion', total_charge=-1)
        result = apply_ion_suffix_modification(si, features)
        assert result.text == "thiolate"

    def test_anion_carboxylic_acid_ring_to_carboxylate(self):
        """Anion on ring carboxylic acid: '-carboxylic acid' -> '-carboxylate'."""
        si = SuffixInfo(text="carboxylic acid", locants=[], count=1, is_terminal=True)
        features = self._make_mock_features(species_type='ion', total_charge=-1)
        result = apply_ion_suffix_modification(si, features)
        assert result.text == "carboxylate"


# ============================================================================
# TestParentInfoDataclass
# ============================================================================

@pytest.mark.unit
class TestParentInfoDataclass:
    """Tests for ParentInfo dataclass defaults and construction."""

    def test_default_atom_to_locant(self):
        """Default atom_to_locant should be empty dict."""
        pi = ParentInfo(parent_label="meth", parent_type="chain", atom_count=1)
        assert pi.atom_to_locant == {}

    def test_default_is_named_ring(self):
        """Default is_named_ring should be False."""
        pi = ParentInfo(parent_label="cyclohex", parent_type="ring", atom_count=6)
        assert pi.is_named_ring is False

    def test_full_construction(self):
        """Full construction with all fields."""
        pi = ParentInfo(
            parent_label="benzene",
            parent_type="benzene",
            atom_count=6,
            atom_to_locant={0: 1, 1: 2, 2: 3},
            is_named_ring=True,
        )
        assert pi.parent_label == "benzene"
        assert pi.parent_type == "benzene"
        assert pi.atom_count == 6
        assert pi.is_named_ring is True
        assert len(pi.atom_to_locant) == 3


# ============================================================================
# TestSuffixInfoDataclass
# ============================================================================

@pytest.mark.unit
class TestSuffixInfoDataclass:
    """Tests for SuffixInfo dataclass defaults and construction."""

    def test_default_construction(self):
        """Default SuffixInfo has empty text and zero count."""
        si = SuffixInfo()
        assert si.text == ""
        assert si.locants == []
        assert si.count == 0
        assert si.is_terminal is False

    def test_full_construction(self):
        """Full construction with all fields."""
        si = SuffixInfo(text="ol", locants=[1, 3], count=2, is_terminal=False)
        assert si.text == "ol"
        assert si.locants == [1, 3]
        assert si.count == 2
        assert si.is_terminal is False
