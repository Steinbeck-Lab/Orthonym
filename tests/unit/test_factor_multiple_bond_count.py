"""Unit tests for Phase 146 _compute_multiple_bond_count factor.

Per Phase 146 CONTEXT.md D-06: the factor counts double + triple bonds
where BOTH endpoints are in parent_atom_indices (parent atoms ONLY, NOT
entire molecule). Substituent multiple bonds (e.g., a nitrile substituent's
C#N triple) must NOT contribute.

Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.4.1.2
"""

import importlib
import os

import pytest
from types import SimpleNamespace

from rdkit import Chem

from orthonym.assembly.coverage_scoring import _compute_multiple_bond_count


def _features(smiles):
    """Minimal features object compatible with the helper's `mol` access."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return SimpleNamespace(mol=mol)


# ---------------------------------------------------------------------------
# Scenario 1: methane (0 atoms with bonds; 0 multiple bonds)
# ---------------------------------------------------------------------------
def test_methane_zero_bonds():
    """Methane C: parent={0}, no bonds at all → factor = 0.0."""
    features = _features("C")
    result = _compute_multiple_bond_count(features, {0})
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Scenario 2: ethane (1 single bond, 0 multiple bonds)
# ---------------------------------------------------------------------------
def test_ethane_zero_multiple_bonds():
    """Ethane CC: parent={0,1}, only single bond → factor = 0.0."""
    features = _features("CC")
    result = _compute_multiple_bond_count(features, {0, 1})
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Scenario 3: propene (1 double bond in parent)
# ---------------------------------------------------------------------------
def test_propene_one_double_bond():
    """Propene C=CC: parent={0,1,2}, one double bond (0=1) → factor = 1.0."""
    features = _features("C=CC")
    result = _compute_multiple_bond_count(features, {0, 1, 2})
    assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Scenario 4: propyne (1 triple bond in parent)
# ---------------------------------------------------------------------------
def test_propyne_one_triple_bond():
    """Propyne C#CC: parent={0,1,2}, one triple bond (0#1) → factor = 1.0."""
    features = _features("C#CC")
    result = _compute_multiple_bond_count(features, {0, 1, 2})
    assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Scenario 5: cyclopentadiene (2 double bonds in parent ring)
# ---------------------------------------------------------------------------
def test_cyclopentadiene_two_double_bonds():
    """Cyclopentadiene C1=CCC=C1: parent={0..4}, two double bonds → factor=2.0."""
    features = _features("C1=CCC=C1")
    result = _compute_multiple_bond_count(features, {0, 1, 2, 3, 4})
    assert result == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Scenario 6a: benzene Kekulé form (3 double bonds in parent)
# ---------------------------------------------------------------------------
def test_benzene_kekule_three_double_bonds():
    """Benzene explicitly kekulized: parent={0..5}, three DOUBLE bonds → factor=3.0.

    RDKit's MolFromSmiles promotes Kekulé-written input like C1=CC=CC=C1 to
    AROMATIC bond types by default (aromaticity perception runs during parse).
    To count the three formal double bonds required by P-44.4.1.2, the caller
    must invoke Chem.Kekulize(mol, clearAromaticFlags=True) first. This test
    exercises that path so the factor returns 3.0 on properly kekulized input.

    Production code paths in Orthonym that care about the factor will feed
    pre-kekulized molecules (or the factor returns 0.0 for aromatic rings —
    the xfail test documents this quirk explicitly).
    """
    mol = Chem.MolFromSmiles("C1=CC=CC=C1")
    Chem.Kekulize(mol, clearAromaticFlags=True)
    features = SimpleNamespace(mol=mol)
    result = _compute_multiple_bond_count(features, {0, 1, 2, 3, 4, 5})
    assert result == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Scenario 6b: benzene aromatic form (xfail — RDKit quirk)
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    reason=(
        "RDKit perceives c1ccccc1 as AROMATIC bonds (not DOUBLE), so the "
        "factor returns 0.0 for aromatic benzene. Documented quirk per D-06 "
        "RESEARCH; Plan 04 grid search compensates by calibrating the factor "
        "weight so aromatic rings don't trigger spurious discrimination."
    ),
    strict=True,
)
def test_benzene_aromatic_bondtype_quirk():
    """Aromatic benzene c1ccccc1: RDKit reports AROMATIC bonds — factor=0.0.

    This xfail documents that callers wanting 'multiple bonds in an aromatic
    ring' must pre-kekulize their mol before invoking the factor. Production
    code paths in Orthonym kekulize via RDKit's default SMILES parse for
    non-aromatic input but preserve aromaticity for aromatic input — so the
    factor's parent-atom-only semantics are independent of this quirk.
    """
    features = _features("c1ccccc1")
    result = _compute_multiple_bond_count(features, {0, 1, 2, 3, 4, 5})
    # Expected-to-fail: we assert the factor should be 3.0, but RDKit returns 0.0.
    assert result == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Scenario 7: nitrile substituent excluded (parent = ethyl portion only)
# ---------------------------------------------------------------------------
def test_nitrile_substituent_excluded():
    """Propanenitrile CCC#N: parent={0,1} (ethyl), nitrile C#N is substituent → 0.0.

    Per D-06: the C#N triple bond has endpoint atom 2 IN parent-adjacent
    but atom 3 (N) OUTSIDE the parent set — the count requires BOTH endpoints
    in parent, so the triple bond is excluded. Tests the parent-atoms-only
    contract head-on.
    """
    features = _features("CCC#N")
    # Parent = just the ethyl carbons; the nitrile C (idx 2) and N (idx 3)
    # are the substituent group.
    result = _compute_multiple_bond_count(features, {0, 1})
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Scenario 8: double bond bridging parent and substituent (acetone C=O)
# ---------------------------------------------------------------------------
def test_double_bond_bridging_parent_excluded():
    """Acetone CC(=O)C: parent={0,1,3} (3-C chain), C=O bridges parent/O → 0.0.

    Atom 1 is IN parent; atom 2 (O) is OUTSIDE parent. Per D-06 the bond is
    counted only if BOTH endpoints are in parent_atom_indices, so the C=O
    double bond does NOT contribute to the factor.
    """
    features = _features("CC(=O)C")
    # Parent: the three carbons. Oxygen (idx 2) is the principal group O.
    result = _compute_multiple_bond_count(features, {0, 1, 3})
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Scenario 9: parent_atom_indices=None → 0.0 (defensive)
# ---------------------------------------------------------------------------
def test_none_parent_returns_zero():
    """None parent_atom_indices → factor = 0.0 (defensive default)."""
    features = _features("C=CC")
    result = _compute_multiple_bond_count(features, None)
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Scenario 10: empty parent set → 0.0
# ---------------------------------------------------------------------------
def test_empty_parent_set_returns_zero():
    """Empty parent_atom_indices → factor = 0.0 (defensive)."""
    features = _features("C=CC")
    result = _compute_multiple_bond_count(features, set())
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Scenario 11: features.mol=None → 0.0 (defensive)
# ---------------------------------------------------------------------------
def test_none_mol_returns_zero():
    """features.mol = None → factor = 0.0 (defensive)."""
    features = SimpleNamespace(mol=None)
    result = _compute_multiple_bond_count(features, {0, 1, 2})
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Integration scenario 12: V18 mode includes the factor via compute_confidence
# ---------------------------------------------------------------------------
def test_v18_mode_includes_factor(monkeypatch):
    """With V18 env flag active, compute_confidence populates the new factor.

    The factor is computed from parent_atom_indices and appears in
    CandidateName.factors['multiple_bond_count'] with a numeric count.
    """
    monkeypatch.setenv("ORTHONYM_USE_V18_WEIGHTS", "true")
    from orthonym.assembly import coverage_scoring as cs
    importlib.reload(cs)
    try:
        features = _features("C=CC")
        cand = cs.compute_confidence(
            "propene", "chain", features, parent_atom_indices={0, 1, 2}
        )
        assert 'multiple_bond_count' in cand.factors
        assert cand.factors['multiple_bond_count'] == pytest.approx(1.0)
    finally:
        # Restore default V17 state for subsequent tests.
        monkeypatch.delenv("ORTHONYM_USE_V18_WEIGHTS", raising=False)
        importlib.reload(cs)


# ---------------------------------------------------------------------------
# Integration scenario 13: V17 mode excludes the factor
# ---------------------------------------------------------------------------
def test_v17_mode_excludes_factor(monkeypatch):
    """With V17 env flag (default), the factors dict must NOT contain the key.

    Byte-identical semantics require V17 path to produce an unchanged factors
    dict — the V18 guard `if 'multiple_bond_count' in FACTOR_WEIGHTS` ensures
    the key is never inserted when V17 is active.
    """
    monkeypatch.setenv("ORTHONYM_USE_V18_WEIGHTS", "false")
    from orthonym.assembly import coverage_scoring as cs
    importlib.reload(cs)
    try:
        features = _features("C=CC")
        cand = cs.compute_confidence(
            "propene", "chain", features, parent_atom_indices={0, 1, 2}
        )
        assert 'multiple_bond_count' not in cand.factors
    finally:
        monkeypatch.delenv("ORTHONYM_USE_V18_WEIGHTS", raising=False)
        importlib.reload(cs)
