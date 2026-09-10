"""
Unit tests for ring classification.

Tests classify_ring returns correct type for each ring category:
- heterocyclic_aromatic: pyridine, furan, thiophene, imidazole
- heterocyclic_saturated: piperidine, THF, thiolane, morpholine
- aromatic: benzene (carbocyclic aromatic)
- cycloalkane: cyclohexane (carbocyclic saturated)
- cycloalkene: cyclohexene (carbocyclic unsaturated, non-aromatic)

Also verifies the startswith('heterocyclic') contract that all callers rely on.
"""

import pytest
from rdkit import Chem
from orthonym.perception.rings import classify_ring


def _get_first_ring(smiles: str):
    """Helper: parse SMILES, return (mol, first_ring_atoms)."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Failed to parse SMILES: {smiles}"
    rings = mol.GetRingInfo().AtomRings()
    assert len(rings) >= 1, f"No rings found in: {smiles}"
    return mol, tuple(rings[0])


# ── Heterocyclic aromatic ────────────────────────────────────────────

@pytest.mark.parametrize("smiles,name", [
    ("c1ccncc1", "pyridine"),
    ("c1ccoc1", "furan"),
    ("c1ccsc1", "thiophene"),
    ("c1cnc[nH]1", "imidazole"),
])
def test_heterocyclic_aromatic(smiles, name):
    """Aromatic heterocycles must be classified as 'heterocyclic_aromatic'."""
    mol, ring = _get_first_ring(smiles)
    result = classify_ring(mol, ring)
    assert result == "heterocyclic_aromatic", (
        f"{name} ({smiles}): expected 'heterocyclic_aromatic', got '{result}'"
    )


# ── Heterocyclic saturated ───────────────────────────────────────────

@pytest.mark.parametrize("smiles,name", [
    ("C1CCNCC1", "piperidine"),
    ("C1CCOC1", "THF"),
    ("C1CCSC1", "thiolane"),
    ("C1COCCN1", "morpholine"),
])
def test_heterocyclic_saturated(smiles, name):
    """Saturated heterocycles must be classified as 'heterocyclic_saturated'."""
    mol, ring = _get_first_ring(smiles)
    result = classify_ring(mol, ring)
    assert result == "heterocyclic_saturated", (
        f"{name} ({smiles}): expected 'heterocyclic_saturated', got '{result}'"
    )


# ── Carbocyclic rings ────────────────────────────────────────────────

def test_benzene_classified_as_aromatic():
    """Benzene (no heteroatoms) must be classified as 'aromatic'."""
    mol, ring = _get_first_ring("c1ccccc1")
    assert classify_ring(mol, ring) == "aromatic"


def test_cyclohexane_classified_as_cycloalkane():
    """Cyclohexane must be classified as 'cycloalkane'."""
    mol, ring = _get_first_ring("C1CCCCC1")
    assert classify_ring(mol, ring) == "cycloalkane"


def test_cyclohexene_classified_as_cycloalkene():
    """Cyclohexene must be classified as 'cycloalkene'."""
    mol, ring = _get_first_ring("C1=CCCCC1")
    assert classify_ring(mol, ring) == "cycloalkene"


# ── startswith('heterocyclic') contract ──────────────────────────────

@pytest.mark.parametrize("smiles,name", [
    ("c1ccncc1", "pyridine"),
    ("C1CCNCC1", "piperidine"),
    ("c1ccoc1", "furan"),
    ("C1CCOC1", "THF"),
    ("c1ccsc1", "thiophene"),
    ("C1CCSC1", "thiolane"),
    ("c1cnc[nH]1", "imidazole"),
    ("C1COCCN1", "morpholine"),
])
def test_heterocyclic_startswith_contract(smiles, name):
    """All heterocyclic classifications must satisfy startswith('heterocyclic').

    This is the backward-compatibility contract: callers use
    ``ring_type.startswith('heterocyclic')`` to match both aromatic
    and saturated heterocycles.
    """
    mol, ring = _get_first_ring(smiles)
    result = classify_ring(mol, ring)
    assert result.startswith("heterocyclic"), (
        f"{name} ({smiles}): '{result}' does not startswith('heterocyclic')"
    )


# ── Negative contract: carbocyclic rings must NOT startswith('heterocyclic') ──

@pytest.mark.parametrize("smiles,name", [
    ("c1ccccc1", "benzene"),
    ("C1CCCCC1", "cyclohexane"),
    ("C1=CCCCC1", "cyclohexene"),
])
def test_carbocyclic_not_heterocyclic(smiles, name):
    """Carbocyclic rings must NOT startswith('heterocyclic')."""
    mol, ring = _get_first_ring(smiles)
    result = classify_ring(mol, ring)
    assert not result.startswith("heterocyclic"), (
        f"{name} ({smiles}): '{result}' unexpectedly startswith('heterocyclic')"
    )
