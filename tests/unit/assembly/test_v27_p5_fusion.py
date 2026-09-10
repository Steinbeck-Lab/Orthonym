""" a phase — general fusion-nomenclature PIN upgrade (P-25.3).

The complete-tier engine ships a von-Baeyer POLYENE for mancude fused ring
systems the default path abstains on; a phase prefers a STRUCTURALLY-VERIFIED
fusion PIN (catalog exact-match, or a computed candidate that AFFIRMATIVELY
round-trips) and falls back to the VB polyene otherwise (0-wrong, non-regressing).
"""
import pytest
from rdkit import Chem

from orthonym.namer import Orthonym, is_failure_name
from orthonym.assembly.general_fusion import (
    name_fusion_parent, _affirmative_roundtrip,
)

# benzo[h]quinoline (the fusion machinery mis-orients this to benzo[f]; the
# generate-and-verify picks the correct side-letter by round-trip).
BENZO_H_QUINOLINE = "N1=CC=CC2=CC=C3C(=C12)C=CC=C3"
CYCLOPENTA_B_NAPHTHALENE = "C1C=CC=2C1=CC1=CC=CC=C1C2"


# NOTE: end-to-end namer assertions are NOT used here — the unit-test harness
# disables the OPSIN validity gate (conftest `_disable_opsin_validity_gate_for_tests`),
# so the default path's UNGATED wrong candidate (benzo[f]) would preempt. The
# Phase-5 producer is tested DIRECTLY (its own affirmative RT is gate-independent);
# end-to-end is verified via scripts/diagnose.py + the milestone gate.

@pytest.mark.unit
def test_affirmative_rt_rejects_wrong_orientation(opsin_available):
    """The 0-wrong floor: the WRONG orientation (benzo[f]) is rejected, the
    correct one (benzo[h]) accepted."""
    if not opsin_available:
        pytest.skip("OPSIN/Java unavailable")
    assert _affirmative_roundtrip("benzo[f]quinoline", BENZO_H_QUINOLINE) is False
    assert _affirmative_roundtrip("benzo[h]quinoline", BENZO_H_QUINOLINE) is True


@pytest.mark.unit
def test_benzo_annulation_produces_correct_pin(opsin_available):
    """The producer upgrades the VB polyene to the verified fusion PIN by
    picking the correct side-letter via round-trip (benzo[h], not benzo[f])."""
    if not opsin_available:
        pytest.skip("OPSIN/Java unavailable")
    mol = Chem.MolFromSmiles(BENZO_H_QUINOLINE)
    ring_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    assert name_fusion_parent(mol, ring_atoms) == "benzo[h]quinoline"


@pytest.mark.unit
def test_name_fusion_parent_bare_guard():
    """name_fusion_parent only fires for a BARE parent (cage == all heavy atoms);
    a substituted molecule returns None (-> the tail/VB path handles it)."""
    mol = Chem.MolFromSmiles("Cc1ccc2ccccc2c1")  # 2-methylnaphthalene (substituted)
    ring_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    assert name_fusion_parent(mol, ring_atoms) is None


@pytest.mark.unit
def test_catalog_exact_match_java_free():
    """A vetted-catalog PAH resolves by exact canonical-SMILES match with NO
    OPSIN consultation (0-wrong by construction)."""
    # triphenylene is a POLYCYCLIC_DATA entry
    mol = Chem.MolFromSmiles("c1ccc2c(c1)c1ccccc1c1ccccc21")
    ring_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    out = name_fusion_parent(mol, ring_atoms)
    assert out == "triphenylene"


@pytest.mark.unit
def test_cyclopenta_fails_closed():
    """A cyclopenta-fused system this phase does not build fails closed (None)
    -> the caller keeps the VB polyene. Never a wrong fusion string."""
    mol = Chem.MolFromSmiles(CYCLOPENTA_B_NAPHTHALENE)
    ring_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    assert name_fusion_parent(mol, ring_atoms) is None
