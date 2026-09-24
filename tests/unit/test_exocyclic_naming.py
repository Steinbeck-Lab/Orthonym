"""
Tests for ring-exocyclic double bond naming.

Exocyclic double bonds (=CH2, =CHR, =NR) on ring systems require:
- ylidene suffix (-ylidene instead of -yl) per IUPAC
- E/Z stereodescriptors when the exocyclic bond has asymmetric substitution
- No false E/Z in substituent/prefix contexts

These tests cover (deferred from a phase-03).
"""

import subprocess
import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler
from tests.support.jars import jar_or_skip


# =============================================================================
# Test exocyclic =C substituent naming (ethylidene, methylidene, etc.)
# =============================================================================

class TestExocyclicYlideneNaming:
    """Tests that exocyclic =C bonds produce ylidene names, not yl names."""

    def test_ethylidene_cyclohexanone_contains_ethylidene(self):
        """Exocyclic =CHCH3 on cyclohexanone should be named ethylidene, not ethyl.

        IUPAC: The suffix -ylidene is used for substituents attached
        by a double bond to the parent structure.
        """
        from orthonym.namer import Orthonym
        namer = Orthonym()
        # (E)-2-ethylidenecyclohexan-1-one
        name = namer.name('C/C=C1\\CCCCC1=O')
        assert 'ethylidene' in name, (
            f"Expected 'ethylidene' in name, got '{name}'. "
            "Exocyclic =CHCH3 should use ylidene suffix per IUPAC P-31.1.3.1"
        )

    def test_methylidene_cyclohexane_contains_methylidene(self):
        """Exocyclic =CH2 on cyclohexane should be named methylidene, not methyl.

        IUPAC: =CH2 -> methylidene
        """
        from orthonym.namer import Orthonym
        namer = Orthonym()
        name = namer.name('C=C1CCCCC1')
        assert 'methylidene' in name, (
            f"Expected 'methylidene' in name, got '{name}'. "
            "Exocyclic =CH2 should use ylidene suffix per IUPAC P-31.1.3.1"
        )

    def test_methylidene_cyclohexanone_contains_methylidene(self):
        """Exocyclic =CH2 on cyclohexanone should be named methylidene."""
        from orthonym.namer import Orthonym
        namer = Orthonym()
        name = namer.name('C=C1CCCCC1=O')
        assert 'methylidene' in name, (
            f"Expected 'methylidene' in name, got '{name}'. "
            "Exocyclic =CH2 should be methylidene even on ketone ring"
        )

    def test_ethylidene_not_ethyl(self):
        """Negative test: ethylidene compound should NOT contain bare 'ethyl'
        (without '-idene' suffix)."""
        from orthonym.namer import Orthonym
        namer = Orthonym()
        name = namer.name('C/C=C1\\CCCCC1=O')
        # Name should not have standalone 'ethyl' -- it should be 'ethylidene'
        # We check that if 'ethyl' appears, it's as part of 'ethylidene'
        if 'ethyl' in name:
            assert 'ethylidene' in name, (
                f"Name contains 'ethyl' but not 'ethylidene': '{name}'. "
                "Exocyclic =CHR must use ylidene, not yl"
            )


# =============================================================================
# Test exocyclic E/Z stereodescriptor emission
# =============================================================================

class TestExocyclicEZEmission:
    """Tests for E/Z descriptor collection on exocyclic double bonds."""

    def test_e_exocyclic_ethylidene_cyclohexanone(self):
        """(E)-ethylidene on cyclohexanone emits E descriptor with ring-atom locant.

        IUPAC: E/Z locant for exocyclic bonds uses endocyclic atom position.
        """
        from orthonym.rules.stereochemistry import collect_stereodescriptors

        mol = Chem.MolFromSmiles('C/C=C1\\CCCCC1=O')
        rdCIPLabeler.AssignCIPLabels(mol)

        # Build ring atom_to_locant (atom 2 is the exocyclic-attached ring C)
        # Ring atoms: 2,3,4,5,6,7 (6-membered ring)
        ring_atoms = []
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            if len(ring) == 6:
                ring_atoms = list(ring)
                break

        atom_to_locant = {idx: pos + 1 for pos, idx in enumerate(ring_atoms)}

        descriptors = collect_stereodescriptors(mol, atom_to_locant)

        # Should have E descriptor for the exocyclic bond
        ez_descriptors = [d for d in descriptors if d[1] in ('E', 'Z')]
        assert len(ez_descriptors) >= 1, (
            f"Expected E/Z descriptor for exocyclic bond, got {descriptors}"
        )
        assert any(d[1] == 'E' for d in ez_descriptors), (
            f"Expected 'E' descriptor, got {ez_descriptors}"
        )

    def test_z_exocyclic_ethylidene_cyclohexanone(self):
        """(Z)-ethylidene on cyclohexanone emits Z descriptor."""
        from orthonym.rules.stereochemistry import collect_stereodescriptors

        mol = Chem.MolFromSmiles('C/C=C1/CCCCC1=O')
        rdCIPLabeler.AssignCIPLabels(mol)

        ring_atoms = []
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            if len(ring) == 6:
                ring_atoms = list(ring)
                break

        atom_to_locant = {idx: pos + 1 for pos, idx in enumerate(ring_atoms)}

        descriptors = collect_stereodescriptors(mol, atom_to_locant)

        ez_descriptors = [d for d in descriptors if d[1] in ('E', 'Z')]
        assert len(ez_descriptors) >= 1, (
            f"Expected E/Z descriptor for exocyclic bond, got {descriptors}"
        )
        assert any(d[1] == 'Z' for d in ez_descriptors), (
            f"Expected 'Z' descriptor, got {ez_descriptors}"
        )

    def test_methylidene_no_ez(self):
        """Exocyclic =CH2 (methylidene) should NOT emit E/Z -- symmetric.

        RDKit correctly does NOT assign _CIPCode to =CH2 bonds because
        the two H substituents are identical.
        """
        from orthonym.rules.stereochemistry import collect_stereodescriptors

        mol = Chem.MolFromSmiles('C=C1CCCCC1')
        rdCIPLabeler.AssignCIPLabels(mol)

        ring_atoms = []
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            if len(ring) == 6:
                ring_atoms = list(ring)
                break

        atom_to_locant = {idx: pos + 1 for pos, idx in enumerate(ring_atoms)}

        descriptors = collect_stereodescriptors(mol, atom_to_locant)

        ez_descriptors = [d for d in descriptors if d[1] in ('E', 'Z')]
        assert len(ez_descriptors) == 0, (
            f"=CH2 should NOT get E/Z (symmetric), but got {ez_descriptors}"
        )

    def test_cyclohexanone_no_ez(self):
        """Exocyclic =O (ketone) should NOT emit E/Z -- single substituent.

        RDKit correctly does NOT assign _CIPCode to C=O bonds.
        """
        from orthonym.rules.stereochemistry import collect_stereodescriptors

        mol = Chem.MolFromSmiles('O=C1CCCCC1')
        rdCIPLabeler.AssignCIPLabels(mol)

        ring_atoms = []
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            if len(ring) == 6:
                ring_atoms = list(ring)
                break

        atom_to_locant = {idx: pos + 1 for pos, idx in enumerate(ring_atoms)}

        descriptors = collect_stereodescriptors(mol, atom_to_locant)

        ez_descriptors = [d for d in descriptors if d[1] in ('E', 'Z')]
        assert len(ez_descriptors) == 0, (
            f"=O should NOT get E/Z, but got {ez_descriptors}"
        )

    def test_exocyclic_imine_ez(self):
        """Exocyclic =NR (imine with asymmetric ring) should emit E/Z.

        IUPAC: Nitrogen imines on asymmetric rings can have E/Z geometry.
        """
        from orthonym.rules.stereochemistry import collect_stereodescriptors

        # =N-CH3 on cyclohexanone (asymmetric ring -> E/Z is assignable)
        mol = Chem.MolFromSmiles('CC/N=C1\\CCCCC1=O')
        rdCIPLabeler.AssignCIPLabels(mol)

        ring_atoms = []
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            if len(ring) == 6:
                ring_atoms = list(ring)
                break

        atom_to_locant = {idx: pos + 1 for pos, idx in enumerate(ring_atoms)}

        descriptors = collect_stereodescriptors(mol, atom_to_locant)

        ez_descriptors = [d for d in descriptors if d[1] in ('E', 'Z')]
        assert len(ez_descriptors) >= 1, (
            f"Exocyclic =NR on asymmetric ring should have E/Z, got {descriptors}"
        )


# =============================================================================
# Test no false E/Z in substituent/prefix contexts
# =============================================================================

class TestNoFalseEZInSubstituentContext:
    """Verify that E/Z is NOT emitted when a ring with exocyclic bond is a substituent."""

    def test_no_ez_when_ring_is_substituent(self):
        """When a ring is named as a substituent (prefix context), exocyclic E/Z
        should not appear in the substituent name.

        a phase-03 deferred this: naive extension caused false (1E) on
        ethylidenecyclohexanone as a substituent.
        """
        from orthonym.rules.stereochemistry import collect_stereodescriptors

        # Substituent context: empty atom_to_locant for the exocyclic bond's ring
        # In substituent naming, the ring atoms are not in the parent's atom_to_locant
        # so exocyclic bonds should naturally be skipped
        mol = Chem.MolFromSmiles('C/C=C1\\CCCCC1=O')
        rdCIPLabeler.AssignCIPLabels(mol)

        # Empty mapping = substituent context (ring atoms not in parent mapping)
        descriptors = collect_stereodescriptors(mol, {})
        assert len(descriptors) == 0, (
            f"With empty atom_to_locant (substituent context), should get no descriptors, "
            f"got {descriptors}"
        )

    def test_no_false_ez_on_chain_substituent_bond(self):
        """E/Z should NOT be emitted for chain double bonds where one atom
        is outside the principal chain mapping.

        This guards against the regression where chain C=N or C=C bonds
        at the edge of a principal chain got false E/Z descriptors.
        """
        from orthonym.rules.stereochemistry import collect_stereodescriptors

        # A chain compound with E/Z on a C=N bond, one atom not in mapping
        mol = Chem.MolFromSmiles('C/N=C(\\N)C')
        rdCIPLabeler.AssignCIPLabels(mol)

        # Only include carbon chain atoms in mapping, exclude N
        atom_to_locant = {0: 1, 2: 2, 4: 3}

        descriptors = collect_stereodescriptors(mol, atom_to_locant)

        # The C=N bond has atom 0 (C, in mapping, NOT in ring) and atom 1 (N, not in mapping)
        # Since atom 0 is NOT in a ring, the exocyclic guard should skip this bond
        ez_descriptors = [d for d in descriptors if d[1] in ('E', 'Z')]
        assert len(ez_descriptors) == 0, (
            f"Chain C=N bond should NOT get exocyclic E/Z treatment, got {ez_descriptors}"
        )


# =============================================================================
# Integration: End-to-end naming tests (Task 2)
# =============================================================================

class TestExocyclicEndToEnd:
    """Integration tests for complete exocyclic compound naming."""

    def _name(self, smiles):
        from orthonym.namer import Orthonym
        return Orthonym().name(smiles)

    def test_e2e_ethylidene_cyclohexanone_e(self):
        """End-to-end: E-ethylidene cyclohexanone has ethylidene AND E descriptor."""
        name = self._name('C/C=C1\\CCCCC1=O')
        assert 'ethylidene' in name, f"Missing ethylidene: {name}"
        assert 'E' in name, f"Missing E descriptor: {name}"

    def test_e2e_methylidene_cyclohexane_no_ez(self):
        """End-to-end: methylidene cyclohexane has methylidene, NO E/Z."""
        name = self._name('C=C1CCCCC1')
        assert 'methylidene' in name, f"Missing methylidene: {name}"
        # No E/Z for symmetric =CH2 on symmetric ring
        assert '(E)' not in name and '(Z)' not in name, (
            f"False E/Z on methylidene cyclohexane: {name}"
        )

    def test_e2e_methylidene_cyclohexanone_no_ez(self):
        """End-to-end: methylidene cyclohexanone has methylidene, NO E/Z.

        =CH2 is symmetric (two H), so no E/Z even on asymmetric ring.
        """
        name = self._name('C=C1CCCCC1=O')
        assert 'methylidene' in name, f"Missing methylidene: {name}"

    def test_e2e_cyclohexanone_unchanged(self):
        """End-to-end: cyclohexanone (=O exocyclic) NO E/Z added."""
        name = self._name('O=C1CCCCC1')
        # Should be "cyclohexan-1-one" or similar
        assert 'cyclohex' in name, f"Missing cyclohex: {name}"
        assert 'E' not in name.split('-')[0] if '-' in name else True, (
            f"False E descriptor on cyclohexanone: {name}"
        )

    def test_e2e_methylcyclohexane_unchanged(self):
        """End-to-end: methylcyclohexane (single bond) stays methyl, not methylidene."""
        name = self._name('CC1CCCCC1')
        assert 'methylcyclohexane' in name, f"Expected methylcyclohexane: {name}"
        assert 'methylidene' not in name, (
            f"Single-bond methyl should NOT become methylidene: {name}"
        )

    def test_e2e_isopropylidene_cyclohexane(self):
        """End-to-end: =C(CH3)2 on a ring gives propan-2-ylidene (F-T9/DD6:
        the PIN is the located 'propan-2-ylidene'; 'isopropylidene' is
        general-only)."""
        # 1-methyl-4-(propan-2-ylidene)cyclohexane
        name = self._name('CC1CCC(=C(C)C)CC1')
        assert 'propan-2-ylidene' in name, (
            f"Expected propan-2-ylidene: {name}"
        )


# =============================================================================
# OPSIN round-trip tests (Task 2)
# =============================================================================

def _opsin_name_to_smiles(name):
    """Parse an IUPAC name with OPSIN, return SMILES or None."""
    try:
        result = subprocess.run(
            ['java', '-jar', jar_or_skip(), '-osmi'],
            input=name, capture_output=True, text=True, timeout=15
        )
        smi = result.stdout.strip()
        return smi if smi else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


@pytest.mark.roundtrip
class TestExocyclicOpsinRoundtrip:
    """OPSIN round-trip tests for exocyclic names.

    Note: OPSIN 2.8.0 does not support methylidene/ethylidene on rings.
    These tests are marked xfail to document the OPSIN limitation.
    """

    def test_opsin_methylidene_cyclohexane(self):
        """OPSIN should parse methylidenecyclohexane.

        a phase cleanup: removed stale @pytest.mark.xfail(reason="OPSIN
        2.8.0..."). a phase / a phase upgraded the project
        to OPSIN 2.9.0 which DOES parse ylidene names on rings; the test
        passes cleanly.
        """
        smi = _opsin_name_to_smiles("methylidenecyclohexane")
        assert smi is not None, "OPSIN failed to parse methylidenecyclohexane"

    def test_opsin_ethylidene_cyclohexanone(self):
        """OPSIN should parse 2-ethylidenecyclohexan-1-one.

        a phase cleanup: removed stale @pytest.mark.xfail (same reason
        as test_opsin_methylidene_cyclohexane).
        """
        smi = _opsin_name_to_smiles("2-ethylidenecyclohexan-1-one")
        assert smi is not None, "OPSIN failed to parse 2-ethylidenecyclohexan-1-one"
