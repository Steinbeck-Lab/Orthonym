"""
Tests for ring-exocyclic double bond naming.

Exocyclic double bonds (=CH2, =CHR, =NR) on ring systems require:
- ylidene suffix (-ylidene instead of -yl) per IUPAC P-31.1.3.1
- E/Z stereodescriptors when the exocyclic bond has asymmetric substitution
- No false E/Z in substituent/prefix contexts

These tests cover STER-04 (deferred from Phase 92-03).
"""

import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler


# =============================================================================
# Test exocyclic =C substituent naming (ethylidene, methylidene, etc.)
# =============================================================================

class TestExocyclicYlideneNaming:
    """Tests that exocyclic =C bonds produce ylidene names, not yl names."""

    def test_ethylidene_cyclohexanone_contains_ethylidene(self):
        """Exocyclic =CHCH3 on cyclohexanone should be named ethylidene, not ethyl.

        IUPAC P-31.1.3.1: The suffix -ylidene is used for substituents attached
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

        IUPAC P-31.1.3.1: =CH2 -> methylidene
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

        IUPAC P-91.2: E/Z locant for exocyclic bonds uses endocyclic atom position.
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

        Phase 92-03 deferred this: naive extension caused false (1E) on
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
