"""
Tests for stereo locant plumbing fixes (Phase 124-01).

Validates:
- _generate_stereodescriptors accepts atom_to_locant_override parameter
- _inject_stereo_if_missing accepts atom_to_locant parameter and forwards it
- get_stereodescriptor_string delegates to production pipeline (no idx+1 fallback)
"""

import pytest
from types import SimpleNamespace
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler


# ---- Helpers ----

def _make_features(smiles, atom_to_locant=None, heterocycle_atom_to_locant=None,
                   oriented_ring=None):
    """Create a minimal MolecularFeatures-like object for testing."""
    mol = Chem.MolFromSmiles(smiles)
    rdCIPLabeler.AssignCIPLabels(mol)

    stereocenters = []
    for atom in mol.GetAtoms():
        if atom.HasProp('_CIPCode'):
            stereocenters.append((atom.GetIdx(), atom.GetProp('_CIPCode')))

    double_bond_stereo = []
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE and bond.HasProp('_CIPCode'):
            cip = bond.GetProp('_CIPCode')
            if cip in ('E', 'Z'):
                double_bond_stereo.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), cip))

    features = SimpleNamespace(
        mol=mol,
        stereocenters=stereocenters,
        double_bond_stereo=double_bond_stereo,
        atom_to_locant=atom_to_locant,
        heterocycle_atom_to_locant=heterocycle_atom_to_locant,
        oriented_ring=oriented_ring,
    )
    return features


# ---- Test _generate_stereodescriptors with atom_to_locant_override ----

class TestGenerateStereodescriptorsOverride:
    """Test that _generate_stereodescriptors accepts and uses atom_to_locant_override."""

    def test_override_map_used_when_provided(self):
        """Test 1: Override map takes priority over features.atom_to_locant."""
        from src.orthonym.assembly.composer import _generate_stereodescriptors

        # C[C@H](O)CC -- atom 1 is the stereocenter
        # With features.atom_to_locant = {0:1, 1:2, 3:3, 4:4}, stereocenter at locant 2
        features = _make_features(
            'C[C@H](O)CC',
            atom_to_locant={0: 1, 1: 2, 3: 3, 4: 4}
        )

        # Override with reversed numbering: atom 1 -> locant 3
        override = {0: 4, 1: 3, 3: 2, 4: 1}
        result = _generate_stereodescriptors(features, atom_to_locant_override=override)

        assert result is not None
        assert '3' in result.text, f"Expected locant 3 in stereo text, got {result.text}"
        # Should NOT contain locant 2 (that would be from features.atom_to_locant)
        # The stereo letter should be present
        assert 'S' in result.text or 'R' in result.text

    def test_fallback_to_features_when_no_override(self):
        """Test 2: Without override, falls back to features.atom_to_locant (backward compat)."""
        from src.orthonym.assembly.composer import _generate_stereodescriptors

        features = _make_features(
            'C[C@H](O)CC',
            atom_to_locant={0: 1, 1: 2, 3: 3, 4: 4}
        )

        # No override -- should use features.atom_to_locant, stereocenter at locant 2
        result = _generate_stereodescriptors(features, atom_to_locant_override=None)

        assert result is not None
        assert '2' in result.text, f"Expected locant 2 in stereo text, got {result.text}"


# ---- Test _inject_stereo_if_missing with atom_to_locant ----

class TestInjectStereoIfMissing:
    """Test that _inject_stereo_if_missing accepts and forwards atom_to_locant."""

    def test_forwards_override_to_generate(self):
        """Test 3: atom_to_locant is forwarded to _generate_stereodescriptors."""
        from src.orthonym.assembly.composer import _inject_stereo_if_missing

        features = _make_features(
            'C[C@H](O)CC',
            atom_to_locant={0: 1, 1: 2, 3: 3, 4: 4}
        )

        # Pass override that maps stereocenter to locant 3
        override = {0: 4, 1: 3, 3: 2, 4: 1}
        result = _inject_stereo_if_missing(features, 'butan-3-ol', atom_to_locant=override)

        # Should prepend (3S)- or (3R)- to the name
        assert result.startswith('(3'), f"Expected stereo prefix with locant 3, got {result}"
        assert 'butan-3-ol' in result

    def test_backward_compat_no_third_arg(self):
        """Test 4: Calling with only (features, name) works identically to before."""
        from src.orthonym.assembly.composer import _inject_stereo_if_missing

        features = _make_features(
            'C[C@H](O)CC',
            atom_to_locant={0: 1, 1: 2, 3: 3, 4: 4}
        )

        # No third argument -- should use features.atom_to_locant
        result = _inject_stereo_if_missing(features, 'butan-2-ol')

        # Should prepend (2S)- or (2R)-
        assert result.startswith('(2'), f"Expected stereo prefix with locant 2, got {result}"
        assert 'butan-2-ol' in result


# ---- Test get_stereodescriptor_string delegation ----

class TestGetStereodescriptorStringDelegation:
    """Test that get_stereodescriptor_string delegates to production pipeline."""

    def test_with_explicit_locant_map(self):
        """Test 5: With locant_map, uses collect_stereodescriptors correctly."""
        from src.orthonym.perception.stereo import get_stereodescriptor_string

        mol = Chem.MolFromSmiles('C[C@H](O)CC')
        rdCIPLabeler.AssignCIPLabels(mol)

        locant_map = {0: 3, 1: 2, 3: 1, 4: 4}
        result = get_stereodescriptor_string(mol, locant_map=locant_map)

        # Stereocenter is atom 1, mapped to locant 2
        assert '(2' in result, f"Expected locant 2 in result, got {result}"
        assert result.endswith('-'), f"Expected trailing hyphen, got {result}"

    def test_with_none_locant_map_uses_identity(self):
        """Test 6: With locant_map=None, uses identity mapping (idx+1)."""
        from src.orthonym.perception.stereo import get_stereodescriptor_string

        mol = Chem.MolFromSmiles('C[C@H](O)CC')
        rdCIPLabeler.AssignCIPLabels(mol)

        result = get_stereodescriptor_string(mol, locant_map=None)

        # Stereocenter is atom 1, identity map -> locant 2
        assert '(2' in result, f"Expected locant 2 in result, got {result}"

    def test_ez_backward_compat(self):
        """Test 7: E/Z output matches existing test case (2E)- for but-2-ene."""
        from src.orthonym.perception.stereo import get_stereodescriptor_string

        mol = Chem.MolFromSmiles('C/C=C/C')
        rdCIPLabeler.AssignCIPLabels(mol)

        locant_map = {0: 1, 1: 2, 2: 3, 3: 4}
        result = get_stereodescriptor_string(mol, locant_map)

        assert result == '(2E)-', f"Expected (2E)- but got {result}"
