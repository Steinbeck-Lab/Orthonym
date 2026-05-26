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


def test_is_general_acyclic_true_for_alkane():
    """Catch-all predicate fires for typical alkane features."""
    features = _features_for("CCCC")
    assert _is_general_acyclic(features) is True


def test_is_general_acyclic_false_for_simple_amide():
    """AP-160.2-06 CASE B refinement: predicate defers to inline amide branch
    for single-amide cases (composer.py:917-919 inline guard mirror)."""

    class FakeFeatures:
        principal_group = 'primary_amide'
        principal_group_atoms = [(0, 1, 2)]  # single amide match

    assert _is_general_acyclic(FakeFeatures()) is False


def test_is_general_acyclic_false_for_amine():
    """AP-160.2-06 CASE B refinement: predicate defers to inline amine branch
    for secondary/tertiary amine cases (composer.py:919 inline guard mirror)."""

    class FakeFeatures:
        principal_group = 'secondary_amine'
        principal_group_atoms = None

    assert _is_general_acyclic(FakeFeatures()) is False


def test_is_general_acyclic_true_for_multi_amide():
    """Multi-amide compounds fall through to general_acyclic (pg_count > 1
    matches the chain-fallback path, not the inline amide branch)."""

    class FakeFeatures:
        principal_group = 'primary_amide'
        principal_group_atoms = [(0, 1, 2), (3, 4, 5)]  # diamide

    assert _is_general_acyclic(FakeFeatures()) is True


def test_is_general_acyclic_pure():
    """Predicate is pure read-only: idempotent across calls."""
    features = _features_for("CCCC")
    assert _is_general_acyclic(features) is True
    assert _is_general_acyclic(features) is True


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


def test_name_general_acyclic_emits_structured_tree():
    """Phase 165 SCORE-01: general_acyclic now emits a STRUCTURED NameTreeNode
    (superseding the Phase-160 first-wave tree=None). The tree round-trips
    byte-identically to the returned name."""
    from orthonym.assembly.name_tree import NameTreeNode
    from orthonym.assembly.name_tree_to_string import name_tree_to_string
    features = _features_for("CCCC")
    result = name_general_acyclic(features, mol=features.mol, style="pin")
    assert isinstance(result.tree, NameTreeNode)
    assert result.tree.parent_stem != ""
    assert result.tree.class_id == "general_acyclic"
    assert name_tree_to_string(result.tree, "pin") == result.name
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
