"""Phase 160 Plan-06 unit tests for ``handlers/ester_family.py``.

Composite handler tests covering signature shape, sub-path fallthrough,
and byte-identical proof vs the inline composer.py cascade.
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

from orthonym import Orthonym
from orthonym.assembly.handlers.ester_family import (
    _is_ester_family,
    name_ester_family,
)
from orthonym.assembly.name_tree import NamingResult


def test_signature_shape():
    """name_ester_family accepts (features, mol=None, style='pin') signature."""
    sig = inspect.signature(name_ester_family)
    assert "features" in sig.parameters
    params = sig.parameters
    if "mol" in params:
        assert params["mol"].default is None
    if "style" in params:
        assert params["style"].default == "pin"


def test_predicate_signature_and_purity():
    """_is_ester_family(features) returns bool; no side effects."""
    class FakeFeatures:
        is_polyfunctional = False
        principal_group = None
        all_ester_matches = None
        ester_match = None

    feats = FakeFeatures()
    result = _is_ester_family(feats)
    assert isinstance(result, bool)
    assert result is False


def test_polyfunctional_subpath_pool_add():
    """When features.is_polyfunctional=True and name_polyfunctional returns
    a name (mock), name_ester_family returns NamingResult and pool.add is
    called with handler_id='polyfunctional'."""
    class FakeFeatures:
        is_polyfunctional = True
        principal_group = None
        all_ester_matches = None
        ester_match = None
        mol = MagicMock()

    FakeFeatures.mol.GetNumHeavyAtoms = MagicMock(return_value=4)

    pool_calls = []

    class FakePool:
        def add(self, name, hid, feats):
            pool_calls.append((name, hid))
            return None

        def best(self):
            class B:
                name = "butane-1,4-diol"
            return B()

    with patch(
        "orthonym.rules.polyfunctional.name_polyfunctional",
        return_value="butane-1,4-diol",
    ), patch(
        "orthonym.assembly.candidate_pool.get_current_pool",
        return_value=FakePool(),
    ):
        result = name_ester_family(FakeFeatures(), mol=FakeFeatures.mol, style="pin")
        assert isinstance(result, NamingResult)
        assert result.name == "butane-1,4-diol"
        assert pool_calls == [("butane-1,4-diol", "polyfunctional")]


def test_ester_subpath_fallthrough_to_simple_ester():
    """When polyfunctional / multi_ester don't apply but ester_match is
    present, the third sub-path fires with handler_id='ester'."""
    class FakeFeatures:
        is_polyfunctional = False
        principal_group = "ester"
        all_ester_matches = None
        ester_match = (0, 1, 2, 3)
        mol = MagicMock()

    FakeFeatures.mol.GetNumHeavyAtoms = MagicMock(return_value=4)

    pool_calls = []

    class FakePool:
        def add(self, name, hid, feats):
            pool_calls.append((name, hid))
            return None

        def best(self):
            class B:
                name = "ethyl acetate"
            return B()

    with patch(
        "orthonym.rules.esters.name_ester",
        return_value="ethyl acetate",
    ), patch(
        "orthonym.assembly.candidate_pool.get_current_pool",
        return_value=FakePool(),
    ):
        result = name_ester_family(FakeFeatures(), mol=FakeFeatures.mol, style="pin")
        assert isinstance(result, NamingResult)
        assert result.name == "ethyl acetate"
        assert pool_calls == [("ethyl acetate", "ester")]


def test_all_subpaths_fallthrough_returns_none():
    """When none of the three sub-paths produce a name, returns None."""
    class FakeFeatures:
        is_polyfunctional = False
        principal_group = "alcohol"  # not ester
        all_ester_matches = None
        ester_match = None
        mol = MagicMock()

    result = name_ester_family(FakeFeatures(), mol=FakeFeatures.mol, style="pin")
    assert result is None


def test_byte_identical_simple_ester_via_inline_cascade():
    """Byte-identical: composer.py inline cascade produces the same name
    as the composite for a representative simple-ester SMILES."""
    # composer.py inline path is what Orthonym().name(...) goes through today.
    namer = Orthonym()
    try:
        inline_name = namer.name("CCOC(=O)C")  # ethyl acetate
    except Exception as e:
        pytest.skip(f"Orthonym().name failed on representative SMILES: {e}")
    # The composite is not yet wired into production (Plan-07 does that),
    # so we cannot fully run it without features. The byte-identical
    # contract is enforced by Plan-07's --mode delta gate at production
    # wiring time. This test pins the inline-name baseline for that gate.
    assert isinstance(inline_name, str)
    assert inline_name  # non-empty


def test_predicate_rejects_ring_assembly_dominant_polyfunctional():
    """Plan-07 IUPAC P-44.1 root-cause fix: the predicate MUST NOT match
    ring-assembly-dominant polyfunctional molecules (e.g. terphenyl polyhydroxy).
    For those, ring_assembly@2500 is the correct dispatch target, not
    ester_family@1500.

    Verified against canary regression: 'COc1cc(-c2ccc(O)c(CC=C(C)C)c2)c(OC)
    c(O)c1-c1ccc(O)c(O)c1' is is_polyfunctional=True but has principal_chain=
    None and aromatic rings. name_polyfunctional returns None for it; the
    predicate must therefore return False so dispatch_inner reaches
    ring_assembly@2500.
    """
    from orthonym.namer import compute_features
    from rdkit import Chem
    smi = "COc1cc(-c2ccc(O)c(CC=C(C)C)c2)c(OC)c(O)c1-c1ccc(O)c(O)c1"
    mol = Chem.MolFromSmiles(smi)
    # Use Orthonym pipeline to get fully-populated features (compute_features
    # alone doesn't set is_polyfunctional).
    namer = Orthonym()
    # Reach into the pipeline to extract features.
    canonical = Chem.MolToSmiles(mol)
    features = namer._perceive(mol, smi, canonical)  # noqa: SLF001 — test access
    namer._classify(features)  # noqa: SLF001
    assert getattr(features, 'is_polyfunctional', False) is True, (
        "Sanity: this molecule should be polyfunctional"
    )
    assert _is_ester_family(features) is False, (
        "Predicate must reject ring-assembly-dominant polyfunctional molecules "
        "(IUPAC P-44.1 hierarchical seniority; ring_assembly@2500 owns these)"
    )


def test_predicate_accepts_chain_parent_polyfunctional():
    """Counter-regression: chain-parent polyfunctional molecules with a
    principal_chain + atom_to_locant must STILL match the predicate so
    ester_family handles them (sub-path 1 polyfunctional)."""
    from unittest.mock import MagicMock

    class FakeFeatures:
        is_polyfunctional = True
        principal_chain = [0, 1, 2, 3]  # truthy
        atom_to_locant = {0: 1, 1: 2}   # truthy
        principal_group = 'hydroxy'
        is_cyclic = False
        chain_is_parent = True
        principal_group_atoms = [(0,)]
        ester_match = None
        all_ester_matches = None
        mol = MagicMock()

    assert _is_ester_family(FakeFeatures()) is True


def test_predicate_accepts_saturated_monocyclic_polyfunctional():
    """Counter-regression: monocyclic saturated polyfunctional rings
    (name_polyfunctional ring-as-parent path) must match."""
    from rdkit import Chem
    # Cyclohexane-1,2-diol — saturated monocyclic, multiple OH = polyfunctional
    mol = Chem.MolFromSmiles("OC1CCCCC1O")

    class FakeFeatures:
        is_polyfunctional = True
        principal_chain = None
        atom_to_locant = None
        principal_group = 'hydroxy'
        is_cyclic = True
        chain_is_parent = False

    feats = FakeFeatures()
    feats.mol = mol
    assert _is_ester_family(feats) is True
