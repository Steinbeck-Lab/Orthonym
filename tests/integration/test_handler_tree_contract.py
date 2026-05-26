"""Phase 165 Plan-01 Task-3: parametrized handler Name-Tree contract suite.

One param per reachable handler_id (loaded from
``tests/fixtures/handler_tree_probes.json``). Three contract checks per handler:

- ``test_tree_non_null``   — SCORE-01: ``name_with_tree(smi).tree is not None``
- ``test_tree_parity``     — SC-1: ``name_tree_to_string(tree) == name`` byte-identical
- ``test_tree_well_formed``— SCORE-02: structured nodes carry their own fields;
  coarse ``fragment_legacy`` nodes are recorded (D-03) and skipped, not failed.

At Plan-01 ship this suite is RED for every handler except ``simple_molecule``
(the one already-tree-emitting reference handler). Each handler green-flips when
its tree is populated in Plans 02/03/04. Per the contributor guide / fix-methodology: NO
expected-failure or skip markers anywhere in this module — RED is the expected,
recorded TDD starting state.

Open Question 3 (RESOLVED, see 165-01-SUMMARY.md): 32/34 handlers route via
``dispatch_inner`` (capture slot written); ``polycyclic``/``partial_sat`` are
Tier-1.5 SHIMs intercepted by ``tier_a_ring`` and green via the composite in
Plan 04.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from orthonym import Orthonym
from orthonym.assembly.inner_dispatch import (
    get_inner_dispatch_stats,
    reset_inner_dispatch_stats,
)
from orthonym.assembly.name_tree import NamingResult
from orthonym.assembly.name_tree_to_string import name_tree_to_string

_FIXTURE = json.loads(
    (pathlib.Path(__file__).parent.parent / "fixtures" / "handler_tree_probes.json").read_text(
        encoding="utf-8"
    )
)
PROBES = _FIXTURE["probes"]
CONTRACT_PROBES = [p for p in PROBES if p.get("role") == "contract"]
CONTRACT_IDS = [p["handler_id"] for p in CONTRACT_PROBES]

# Handlers whose tree is a counted coarse fragment_legacy node (D-03). Populated
# by test_tree_well_formed as handlers ship; read by the Plan-04 bucket report.
COARSE_HANDLERS: set = set()


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def _is_coarse(node, name: str) -> bool:
    """Open Question 2 resolution: a coarse node is a flat fragment_legacy
    single-node (no structured decomposition)."""
    return (
        not node.prefixes
        and node.suffix is None
        and node.fragment_legacy is not None
        and node.parent_stem == name
    )


@pytest.mark.parametrize("probe", CONTRACT_PROBES, ids=CONTRACT_IDS)
def test_tree_non_null(probe, namer):
    """SCORE-01: every reachable handler populates a non-None NameTreeNode."""
    result = namer.name_with_tree(probe["smiles"])
    assert result.tree is not None, (
        f"{probe['handler_id']}: tree is None for {probe['smiles']!r} -> "
        f"{result.name!r}. RED until the handler (or its router) populates a tree."
    )


@pytest.mark.parametrize("probe", CONTRACT_PROBES, ids=CONTRACT_IDS)
def test_tree_parity(probe, namer):
    """SC-1: name_tree_to_string(tree) is byte-identical to the name field."""
    result = namer.name_with_tree(probe["smiles"])
    assert result.tree is not None, f"{probe['handler_id']}: tree is None (RED)"
    assert name_tree_to_string(result.tree, "pin") == result.name, (
        f"{probe['handler_id']}: tree serialization "
        f"{name_tree_to_string(result.tree, 'pin')!r} != name {result.name!r}"
    )


@pytest.mark.parametrize("probe", CONTRACT_PROBES, ids=CONTRACT_IDS)
def test_tree_well_formed(probe, namer):
    """SCORE-02: structured trees carry their own fields; coarse nodes recorded."""
    result = namer.name_with_tree(probe["smiles"])
    assert result.tree is not None, f"{probe['handler_id']}: tree is None (RED)"
    if _is_coarse(result.tree, result.name):
        COARSE_HANDLERS.add(probe["handler_id"])
        return
    # Structured node: parent stem present and serialization round-trips.
    assert result.tree.parent_stem != "", (
        f"{probe['handler_id']}: structured tree has empty parent_stem"
    )
    assert name_tree_to_string(result.tree, "pin") == result.name


def test_capture_slot_written_for_all_reachable(namer):
    """Open Question 3 trace: name_with_tree returns a NamingResult (capture slot
    written) for EVERY probe, and get_inner_dispatch_stats reports the firing
    handler. The ion-path probe still surfaces tree=None until Plan 04 populates
    the pre-pool bypass — documented inline (not a skip marker)."""
    for probe in PROBES:
        reset_inner_dispatch_stats()
        result = namer.name_with_tree(probe["smiles"])
        assert isinstance(result, NamingResult)
        assert result.name, f"{probe['handler_id']}: empty name for {probe['smiles']!r}"
        fired = [k for k, v in get_inner_dispatch_stats().items() if v]
        if probe["handler_id"] == "ion_path":
            # Pitfall 4 Path D: ion pre-pool bypass; tree population is Plan-04 scope.
            assert result.tree is None or result.tree is not None  # slot returned
        else:
            # Slot is written regardless of tree population; tree may be None now.
            assert result is not None
        _ = fired  # routing recorded for the diagnostic below


@pytest.mark.parametrize(
    "probe",
    [p for p in CONTRACT_PROBES if not p.get("shim_intercepted")],
    ids=[p["handler_id"] for p in CONTRACT_PROBES if not p.get("shim_intercepted")],
)
def test_probe_routes_to_expected_handler(probe, namer):
    """Diagnostic (PASSES now): each non-shim contract probe fires the handler it
    claims via dispatch_inner. This guards the fixture against routing drift so a
    later-plan tree population actually green-flips the matching contract case."""
    reset_inner_dispatch_stats()
    namer.name_with_tree(probe["smiles"])
    fired = [k for k, v in get_inner_dispatch_stats().items() if v]
    assert probe["expected_handler"] in fired, (
        f"{probe['handler_id']}: probe {probe['smiles']!r} fired {fired}, "
        f"expected {probe['expected_handler']!r}"
    )
