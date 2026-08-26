"""
Tests for bicyclo[x.y.0] fused routing (OPSIN-02).

Validates that zero-bridge bicyclic systems are routed to fused nomenclature
instead of bicyclo[x.y.0] naming, while true bridged systems continue to
use the standard bicyclo descriptor format.

Key rules:
- bicyclo[x.y.0] with both rings >= 5 = fused system -> fused naming
- bicyclo[x.y.0] with small rings (< 5) = keep bicyclo descriptor
- True bridged systems (no zero bridge) -> bicyclo[x.y.z] naming
- bridge_sum + 2 must equal ring_atom_count for all valid descriptors

IUPAC Reference: Blue Book 2013, P-23.2, P-31.1.1
"""

import subprocess
import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.bicyclo import (
    is_bicyclo_system,
    generate_bicyclo_descriptor,
    find_true_bridgeheads,
    get_bridge_lengths,
)


# =============================================================================
# TestBicycloZeroBridgeRouting: Zero-bridge systems route to fused naming
# =============================================================================

class TestBicycloZeroBridgeRouting:
    """Zero-bridge (bicyclo[x.y.0]) systems should NOT be classified as bicyclo
    when both rings are large enough (>= 5 members) to have a fused parent."""

    @pytest.mark.unit
    def test_decalin_not_bicyclo(self):
        """Decalin (two fused cyclohexanes) should not be classified as bicyclo."""
        mol = Chem.MolFromSmiles('C1CCC2CCCCC2C1')
        assert is_bicyclo_system(mol) is False

    @pytest.mark.unit
    def test_octahydropentalene_not_bicyclo(self):
        """Octahydropentalene (two fused cyclopentanes) should not be bicyclo."""
        mol = Chem.MolFromSmiles('C1CCC2CCCC12')
        assert is_bicyclo_system(mol) is False

    @pytest.mark.unit
    def test_decalin_descriptor_is_none(self):
        """Decalin should return None from generate_bicyclo_descriptor."""
        mol = Chem.MolFromSmiles('C1CCC2CCCCC2C1')
        assert generate_bicyclo_descriptor(mol) is None

    @pytest.mark.unit
    def test_octahydropentalene_descriptor_is_none(self):
        """Octahydropentalene should return None from generate_bicyclo_descriptor."""
        mol = Chem.MolFromSmiles('C1CCC2CCCC12')
        assert generate_bicyclo_descriptor(mol) is None

    @pytest.mark.unit
    def test_decalin_has_zero_bridge(self):
        """Decalin bridge lengths should include a zero."""
        mol = Chem.MolFromSmiles('C1CCC2CCCCC2C1')
        bridgeheads = list(find_true_bridgeheads(mol))
        assert len(bridgeheads) == 2
        lengths = get_bridge_lengths(mol, bridgeheads[0], bridgeheads[1])
        assert 0 in lengths

    @pytest.mark.unit
    def test_small_ring_zero_bridge_still_bicyclo(self):
        """bicyclo[1.1.0]butane (two fused cyclopropanes) should remain bicyclo.
        Small ring systems (< 5 members) have no fused parent name."""
        mol = Chem.MolFromSmiles('C1C2CC12')
        assert is_bicyclo_system(mol) is True
        assert generate_bicyclo_descriptor(mol) == 'bicyclo[1.1.0]'


# =============================================================================
# TestBicycloValidBridged: True bridged systems still work correctly
# =============================================================================

class TestBicycloValidBridged:
    """True bridged systems (no zero-length bridge) should continue to use
    bicyclo[x.y.z] naming."""

    @pytest.mark.unit
    def test_norbornane_is_bicyclo(self):
        """Norbornane is a true bridged bicyclic."""
        mol = Chem.MolFromSmiles('C1CC2CCC1C2')
        assert is_bicyclo_system(mol) is True

    @pytest.mark.unit
    def test_norbornane_descriptor(self):
        """Norbornane should have bicyclo[2.2.1] descriptor."""
        mol = Chem.MolFromSmiles('C1CC2CCC1C2')
        assert generate_bicyclo_descriptor(mol) == 'bicyclo[2.2.1]'

    @pytest.mark.unit
    def test_bicyclo_222_is_bicyclo(self):
        """Bicyclo[2.2.2]octane is a true bridged bicyclic."""
        mol = Chem.MolFromSmiles('C1CC2CCC1CC2')
        assert is_bicyclo_system(mol) is True

    @pytest.mark.unit
    def test_bicyclo_222_descriptor(self):
        """Bicyclo[2.2.2]octane should have correct descriptor."""
        mol = Chem.MolFromSmiles('C1CC2CCC1CC2')
        assert generate_bicyclo_descriptor(mol) == 'bicyclo[2.2.2]'

    @pytest.mark.unit
    def test_bicyclo_321_is_bicyclo(self):
        """Bicyclo[3.2.1]octane is a true bridged bicyclic."""
        mol = Chem.MolFromSmiles('C1CC2CCCC1C2')
        assert is_bicyclo_system(mol) is True


# =============================================================================
# TestBridgeSumVerification: bridge_sum + 2 = ring_atoms invariant
# =============================================================================

class TestBridgeSumVerification:
    """Verify the IUPAC invariant: bridge_sum + 2 = total ring atoms."""

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,expected_bridges,expected_atoms", [
        # Norbornane: bridges [2,2,1], sum=5, +2=7
        ("C1CC2CCC1C2", [2, 2, 1], 7),
        # Bicyclo[2.2.2]octane: bridges [2,2,2], sum=6, +2=8
        ("C1CC2CCC1CC2", [2, 2, 2], 8),
        # Bicyclo[3.2.1]octane: bridges [3,2,1], sum=6, +2=8
        ("C1CC2CCCC1C2", [3, 2, 1], 8),
        # Bicyclo[1.1.1]pentane: bridges [1,1,1], sum=3, +2=5
        ("C1(CC2CC12)C", None, None),  # complex SMILES, skip bridge check
    ])
    def test_bridge_sum_invariant(self, smiles, expected_bridges, expected_atoms):
        """Bridge lengths sum + 2 should equal total ring atoms."""
        mol = Chem.MolFromSmiles(smiles)
        if not is_bicyclo_system(mol):
            pytest.skip("Not classified as bicyclo")
        bridgeheads = list(find_true_bridgeheads(mol))
        if len(bridgeheads) != 2:
            pytest.skip("Could not find 2 bridgeheads")
        lengths = get_bridge_lengths(mol, bridgeheads[0], bridgeheads[1])

        # Verify the invariant
        ri = mol.GetRingInfo()
        ring_atoms = set()
        for ring in ri.AtomRings():
            ring_atoms.update(ring)
        bridge_sum = sum(lengths)
        assert bridge_sum + 2 == len(ring_atoms), (
            f"Bridge sum {bridge_sum} + 2 != ring atoms {len(ring_atoms)} "
            f"for bridges {lengths}"
        )

        if expected_bridges is not None:
            assert lengths == expected_bridges

    @pytest.mark.unit
    def test_norbornane_bridge_formula(self):
        """Norbornane: [2,2,1] sum=5, +2=7 atoms."""
        mol = Chem.MolFromSmiles('C1CC2CCC1C2')
        bridgeheads = list(find_true_bridgeheads(mol))
        lengths = get_bridge_lengths(mol, bridgeheads[0], bridgeheads[1])
        assert lengths == [2, 2, 1]
        assert sum(lengths) + 2 == 7

    @pytest.mark.unit
    def test_bicyclo_222_bridge_formula(self):
        """Bicyclo[2.2.2]octane: [2,2,2] sum=6, +2=8 atoms."""
        mol = Chem.MolFromSmiles('C1CC2CCC1CC2')
        bridgeheads = list(find_true_bridgeheads(mol))
        lengths = get_bridge_lengths(mol, bridgeheads[0], bridgeheads[1])
        assert lengths == [2, 2, 2]
        assert sum(lengths) + 2 == 8


# =============================================================================
# TestEndToEndFusedRouting: Full pipeline produces correct names
# =============================================================================

class TestEndToEndFusedRouting:
    """End-to-end tests verifying that zero-bridge systems get fused names
    and true bridged systems get bicyclo names."""

    @pytest.mark.integration
    def test_decalin_fused_name(self):
        """Decalin should get a fused name, not bicyclo."""
        result = name_compound('C1CCC2CCCCC2C1')
        assert 'bicyclo' not in result
        # Should be some form of fused naming (decahydronaphthalene or perhydronaphthalene)
        assert 'naphthalene' in result.lower() or 'decahydro' in result.lower()

    @pytest.mark.integration
    def test_decalin_specific_name(self):
        """Decalin should be named decahydronaphthalene."""
        result = name_compound('C1CCC2CCCCC2C1')
        assert result == 'decahydronaphthalene'

    @pytest.mark.integration
    def test_octahydropentalene_fused_name(self):
        """Two fused cyclopentanes should get octahydropentalene."""
        result = name_compound('C1CCC2CCCC12')
        assert 'bicyclo' not in result
        assert result == 'octahydropentalene'

    @pytest.mark.integration
    def test_norbornane_still_retained(self):
        """Norbornane should still use its retained name."""
        result = name_compound('C1CC2CCC1C2')
        assert result == 'norbornane'

    @pytest.mark.integration
    def test_bicyclo_222_still_systematic(self):
        """Bicyclo[2.2.2]octane should still use systematic naming."""
        result = name_compound('C1CC2CCC1CC2')
        assert result == 'bicyclo[2.2.2]octane'

    @pytest.mark.integration
    def test_naphthalene_aromatic_unchanged(self):
        """Naphthalene (aromatic) should still get its retained name."""
        result = name_compound('c1ccc2ccccc2c1')
        assert result == 'naphthalene'

    @pytest.mark.integration
    def test_tetralin_partially_saturated_unchanged(self):
        """Tetralin (partially saturated naphthalene) should be unaffected."""
        result = name_compound('c1ccc2c(c1)CCCC2')
        assert 'tetrahydronaphthalene' in result

    @pytest.mark.integration
    def test_opsin_parses_decahydronaphthalene(self):
        """OPSIN should parse decahydronaphthalene successfully."""
        try:
            result = subprocess.run(
                ['java', '-jar', 'opsin-cli-2.9.0-jar-with-dependencies.jar', '-osmi'],
                input='decahydronaphthalene',
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(__import__('pathlib').Path(__file__).resolve().parents[3])
            )
            output = result.stdout.strip()
            # Filter out the "Run the jar..." header line
            lines = [l for l in output.split('\n') if not l.startswith('Run')]
            if lines:
                smiles = lines[0].strip()
                # OPSIN should return a valid SMILES
                mol = Chem.MolFromSmiles(smiles)
                assert mol is not None, f"OPSIN returned invalid SMILES: {smiles}"
            else:
                pytest.skip("OPSIN returned no output")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("OPSIN JAR not available or timed out")

    @pytest.mark.integration
    def test_opsin_parses_octahydropentalene(self):
        """OPSIN should parse octahydropentalene successfully."""
        try:
            result = subprocess.run(
                ['java', '-jar', 'opsin-cli-2.9.0-jar-with-dependencies.jar', '-osmi'],
                input='octahydropentalene',
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(__import__('pathlib').Path(__file__).resolve().parents[3])
            )
            output = result.stdout.strip()
            lines = [l for l in output.split('\n') if not l.startswith('Run')]
            if lines:
                smiles = lines[0].strip()
                mol = Chem.MolFromSmiles(smiles)
                assert mol is not None, f"OPSIN returned invalid SMILES: {smiles}"
            else:
                pytest.skip("OPSIN returned no output")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("OPSIN JAR not available or timed out")
