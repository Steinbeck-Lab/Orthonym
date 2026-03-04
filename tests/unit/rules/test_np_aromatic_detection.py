"""Unit tests for AROMATIC bond detection in _find_scaffold_unsaturation().

Phase 89 Task 2: The function currently only detects DOUBLE and TRIPLE
bond types, missing AROMATIC bonds in aromatic rings of natural product
scaffolds (e.g., aromatic A-ring in estrane steroids).

IUPAC Blue Book P-31.1.3.4: Aromatic ring bonds in partially aromatic
scaffolds are expressed as ene positions (e.g., estra-1,3,5(10)-triene).
"""

import pytest
from rdkit import Chem

from orthonym.rules.natural_products import _find_scaffold_unsaturation


def _make_scaffold_inputs(smiles: str):
    """Create mol, matched_set, and numbering for a molecule.

    For testing purposes, we treat ALL heavy atoms as scaffold atoms
    and assign sequential IUPAC locants starting from 1.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    matched_set = set(range(mol.GetNumAtoms()))
    numbering = {i: i + 1 for i in range(mol.GetNumAtoms())}
    return mol, matched_set, numbering


class TestAromaticBondDetection:
    """Tests for AROMATIC bond type detection as ene unsaturation."""

    def test_aromatic_ring_produces_ene_locants(self):
        """Benzene (fully aromatic) should produce ene locants for C=C bonds."""
        mol, matched, numbering = _make_scaffold_inputs('c1ccccc1')
        result = _find_scaffold_unsaturation(mol, matched, numbering)
        # Benzene has 6 aromatic C-C bonds, all should be detected as ene
        assert len(result['ene']) > 0, (
            "AROMATIC bonds not detected: benzene should have ene locants"
        )

    def test_aromatic_steroid_a_ring(self):
        """Estrone-like aromatic A-ring should produce ene locants.

        Estra-1,3,5(10)-trien-17-one has 3 aromatic C=C bonds in the A-ring.
        The remaining rings B/C/D are saturated.
        """
        # Estrone: aromatic A-ring steroid
        estrone_smi = 'C[C@]12CC[C@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CCC2=O'
        mol = Chem.MolFromSmiles(estrone_smi)
        assert mol is not None
        # Use all carbon atoms in scaffold
        matched = set(range(mol.GetNumAtoms()))
        numbering = {i: i + 1 for i in range(mol.GetNumAtoms())}
        result = _find_scaffold_unsaturation(mol, matched, numbering)
        # Aromatic A-ring should give ene locants for the C-C aromatic bonds
        assert len(result['ene']) >= 3, (
            f"Expected >= 3 ene locants for aromatic A-ring, got {result['ene']}"
        )

    def test_saturated_steroid_no_ene(self):
        """Fully saturated steroid (androstane) should produce no ene locants."""
        # 5alpha-Androstane: fully saturated
        androstane_smi = 'C[C@]12CC[C@H]3[C@@H](CC[C@@H]4CCCC[C@@H]34)[C@@H]1CCC2'
        mol, matched, numbering = _make_scaffold_inputs(androstane_smi)
        result = _find_scaffold_unsaturation(mol, matched, numbering)
        assert result['ene'] == [], (
            f"Saturated steroid should have no ene locants, got {result['ene']}"
        )
        assert result['yne'] == [], (
            f"Saturated steroid should have no yne locants, got {result['yne']}"
        )

    def test_explicit_double_bond_still_detected(self):
        """Steroid with explicit C=C double bond (non-aromatic) still works."""
        # Cholest-5-ene skeleton fragment (simplified)
        mol, matched, numbering = _make_scaffold_inputs('C1CC=CCC1')
        result = _find_scaffold_unsaturation(mol, matched, numbering)
        assert len(result['ene']) == 1, (
            f"Explicit double bond should give 1 ene locant, got {result['ene']}"
        )

    def test_no_yne_for_aromatic(self):
        """Aromatic bonds should NOT produce yne locants."""
        mol, matched, numbering = _make_scaffold_inputs('c1ccccc1')
        result = _find_scaffold_unsaturation(mol, matched, numbering)
        assert result['yne'] == [], (
            f"Aromatic bonds should not produce yne locants, got {result['yne']}"
        )

    def test_mixed_aromatic_and_saturated(self):
        """Naphthalene fused with saturated ring: only aromatic part has ene."""
        # Tetrahydronaphthalene (tetralin): one aromatic + one saturated ring
        mol, matched, numbering = _make_scaffold_inputs('c1ccc2c(c1)CCCC2')
        result = _find_scaffold_unsaturation(mol, matched, numbering)
        # 4 aromatic C-C bonds in the aromatic ring, 0 in the saturated ring
        assert len(result['ene']) > 0, (
            "Mixed system should detect aromatic C-C bonds as ene"
        )
        # Should have exactly the aromatic C-C bonds (not C-C bonds in sat ring)
        # The aromatic ring has C atoms at positions sharing with sat ring
        assert len(result['ene']) <= 6, (
            "Too many ene locants for a single aromatic ring"
        )
