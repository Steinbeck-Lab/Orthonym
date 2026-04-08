"""Tests for is_cyclic mutation removal and chain_is_parent architecture (Phase 139 ARCH-01).

Verifies that:
1. is_cyclic=False is never set in namer.py
2. Ring data is fully populated even when chain is parent
3. Chain orientation runs for chain-is-parent molecules
4. Assembly layer routes correctly based on chain_is_parent
5. No regressions for pure ring or pure chain molecules
"""

import os

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym, MolecularFeatures


@pytest.fixture
def namer():
    """Create an Orthonym instance."""
    return Orthonym()


def _classify(namer, smiles):
    """Run perception + classification on a SMILES and return features."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    mol = Chem.AddHs(mol)
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    features = namer._perceive(mol, smiles, canonical)
    namer._classify(features)
    return features


class TestIsCyclicMutationRemoved:
    """Verify the is_cyclic = False mutation is deleted from namer.py."""

    def test_no_is_cyclic_false_in_namer(self):
        """Grep for 'is_cyclic = False' in namer.py -- must return zero hits."""
        namer_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'src', 'orthonym', 'namer.py'
        )
        namer_path = os.path.normpath(namer_path)
        with open(namer_path, 'r') as f:
            content = f.read()
        # Count actual assignment lines (not comments or strings)
        count = 0
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            if 'features.is_cyclic = False' in stripped:
                count += 1
        assert count == 0, f"Found {count} instances of 'features.is_cyclic = False' in namer.py"


class TestChainIsParentRingData:
    """Verify ring data is fully populated when chain is parent."""

    def test_phenylbutanoic_acid_is_cyclic_true(self, namer):
        """For phenylbutanoic acid, is_cyclic must be True (ring is present)."""
        features = _classify(namer, 'c1ccccc1CCC(=O)O')
        assert features.is_cyclic is True, (
            f"is_cyclic should be True for cyclic molecule with chain parent, got {features.is_cyclic}"
        )

    def test_phenylbutanoic_acid_chain_is_parent(self, namer):
        """For phenylbutanoic acid, chain_is_parent must be True."""
        features = _classify(namer, 'c1ccccc1CCC(=O)O')
        assert features.chain_is_parent is True, (
            f"chain_is_parent should be True, got {features.chain_is_parent}"
        )

    def test_phenylbutanoic_acid_principal_ring_populated(self, namer):
        """For phenylbutanoic acid, principal_ring must be populated (ring data preserved)."""
        features = _classify(namer, 'c1ccccc1CCC(=O)O')
        # principal_ring should be a tuple of ring atom indices
        assert features.principal_ring is not None, (
            "principal_ring should be populated when chain is parent for a cyclic molecule"
        )


class TestChainOrientationForChainIsParent:
    """Verify chain orientation runs for chain-is-parent molecules."""

    def test_phenylbutanoic_acid_atom_to_locant(self, namer):
        """For phenylbutanoic acid, atom_to_locant must be populated (chain orientation ran)."""
        features = _classify(namer, 'c1ccccc1CCC(=O)O')
        assert features.atom_to_locant, (
            f"atom_to_locant should be populated for chain-is-parent molecule, got {features.atom_to_locant}"
        )


class TestPureRingRegression:
    """Verify pure ring molecules are unaffected by the change."""

    def test_cyclohexane_is_cyclic_true(self, namer):
        """Cyclohexane must have is_cyclic=True."""
        features = _classify(namer, 'C1CCCCC1')
        assert features.is_cyclic is True

    def test_cyclohexane_chain_is_parent_false(self, namer):
        """Cyclohexane must have chain_is_parent=False."""
        features = _classify(namer, 'C1CCCCC1')
        assert features.chain_is_parent is False

    def test_cyclohexanol_is_cyclic_true(self, namer):
        """Cyclohexanol must have is_cyclic=True."""
        features = _classify(namer, 'OC1CCCCC1')
        assert features.is_cyclic is True


class TestPureChainRegression:
    """Verify pure chain molecules are unaffected."""

    def test_hexane_is_cyclic_false(self, namer):
        """Hexane must have is_cyclic=False."""
        features = _classify(namer, 'CCCCCC')
        assert features.is_cyclic is False

    def test_hexanol_is_cyclic_false(self, namer):
        """Hexanol must have is_cyclic=False."""
        features = _classify(namer, 'CCCCCCO')
        assert features.is_cyclic is False


class TestAssemblyRouting:
    """Verify assembly routes chain-is-parent molecules to chain naming."""

    def test_phenylbutanoic_acid_chain_suffix(self):
        """Phenylbutanoic acid must produce a chain-type suffix ('oic acid'), not ring-based."""
        from orthonym import name_compound
        name = name_compound('c1ccccc1CCC(=O)O')
        assert name is not None, "name_compound returned None"
        # Must contain "oic acid" (chain naming) not "carboxylic acid" only
        # The chain is 4 carbons (butanoic) with suffix "oic acid"
        assert 'butanoic acid' in name.lower() or 'oic acid' in name.lower(), (
            f"Expected chain-type suffix 'oic acid' in name, got: {name}"
        )
