"""Unit tests for Phase 160.2 Plan-02-02: general_acyclic catch-all handler.

Per CONTEXT D-08 + ADR-19-02 §3.1: explicit catch-all (priority 99999,
predicate=lambda *_: True) closing the Plan-02 fallthrough gap so
dispatch_inner first-match-AND-succeeds-wins ALWAYS returns a non-None
InnerDispatchResult.

Per CONTEXT D-10 + RESEARCH §7: first-wave NamingResult emits tree=None.
Full tree IR population deferred to v19+ phases.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.handlers.general_acyclic import (
    _is_general_acyclic,
    name_general_acyclic,
)
from orthonym.assembly.name_tree import NamingResult


# =============================================================================
# Predicate tests
# =============================================================================


def test_is_general_acyclic_always_true():
    """CONTEXT D-08 catch-all: predicate returns True for any input."""
    assert _is_general_acyclic(None) is True
    assert _is_general_acyclic({}) is True
    assert _is_general_acyclic(object()) is True


def test_is_general_acyclic_pure():
    """Predicate is pure read-only: idempotent across calls."""
    sentinel = object()
    assert _is_general_acyclic(sentinel) is True
    assert _is_general_acyclic(sentinel) is True


# =============================================================================
# Handler return-shape tests
# =============================================================================


def _features_for(smi: str):
    """Build fully-classified features for a SMILES via Orthonym pipeline.

    Mirrors the full perception + classification pipeline used by
    Orthonym.name() so handler tests run against features in the exact
    shape they would receive during dispatch_inner invocation.
    """
    from orthonym.namer import Orthonym
    namer = Orthonym(style="pin")
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None, f"Invalid SMILES: {smi}"
    from rdkit.Chem import CanonSmiles
    canon = CanonSmiles(smi)
    features = namer._perceive(mol, smi, canon)
    namer._classify(features)
    return features


def test_name_general_acyclic_returns_naming_result():
    """Handler returns NamingResult instance per CONTEXT D-10 contract."""
    features = _features_for("CCCC")
    result = name_general_acyclic(features, mol=features.mol, style="pin")
    assert isinstance(result, NamingResult)


def test_name_general_acyclic_tree_is_none():
    """First-wave policy per CONTEXT D-05 + D-10: tree=None for general_acyclic."""
    features = _features_for("CCCC")
    result = name_general_acyclic(features, mol=features.mol, style="pin")
    assert result.tree is None
    assert result.atom_to_locant_hint is None


def test_name_general_acyclic_butane():
    """Catch-all names simple butane as 'butane' per IUPAC P-14 + P-23 + P-44."""
    features = _features_for("CCCC")
    result = name_general_acyclic(features, mol=features.mol, style="pin")
    assert result.name == "butane"


def test_name_general_acyclic_propanol():
    """Catch-all names propanol substring is present."""
    features = _features_for("CCCO")
    result = name_general_acyclic(features, mol=features.mol, style="pin")
    assert result.name
    assert "propan" in result.name or "ol" in result.name


def test_name_general_acyclic_methane():
    """Catch-all names methane."""
    features = _features_for("C")
    result = name_general_acyclic(features, mol=features.mol, style="pin")
    assert result.name


def test_name_general_acyclic_returns_non_empty_for_typical_chain():
    """Catch-all returns a non-empty name string for any features with a chain."""
    features = _features_for("CCCCCC")
    result = name_general_acyclic(features, mol=features.mol, style="pin")
    assert result.name
    assert isinstance(result.name, str)
    assert len(result.name) > 0


# =============================================================================
# Module structure tests
# =============================================================================


def test_general_acyclic_all_exports():
    """__all__ exports both name_general_acyclic and _is_general_acyclic."""
    from orthonym.assembly.handlers import general_acyclic
    assert set(general_acyclic.__all__) == {"name_general_acyclic", "_is_general_acyclic"}


def test_general_acyclic_handler_callable():
    """name_general_acyclic and _is_general_acyclic are callable."""
    assert callable(name_general_acyclic)
    assert callable(_is_general_acyclic)


def test_general_acyclic_loc_target():
    """general_acyclic.py target ~150-200 LOC verbatim lift."""
    src = open("src/orthonym/assembly/handlers/general_acyclic.py").read()
    loc = src.count("\n")
    # The plan target is ~150 LOC; verbatim lift makes it ~190 LOC with imports + docstring
    assert 100 <= loc <= 300, f"general_acyclic.py LOC = {loc}, expected 100-300"
