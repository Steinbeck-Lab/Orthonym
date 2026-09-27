"""a phase (-03 /) — standing regression tripwire for the
name-tree-serializer production flip.

JVM-free, frozen-snapshot guard (mirrors tests/unit/rules/test_among_rings_gold.py):
``tests/fixtures/serializer_flip_snapshot.csv`` holds (smiles, production_name,
class_id) for the flipped general_acyclic proof set + two carrier anchors. For
every FLIPPED-class row this asserts that the EXPLICIT-field serializer path
(``_assemble_explicit_fields`` — NOT ``name_tree_to_string``, which would
short-circuit on a str carrier) reproduces the frozen production name
byte-for-byte. If a future change ever makes a flipped class diverge, this
FAILS — that is the regression guard.

It also asserts the carrier manifest is stable: each carrier-anchor row's
class_id is NOT in ``SERIALIZER_PRODUCTION_CLASSES`` (an accidental future flip
without 100% byte-identity would trip this).

JVM-free: the OPSIN validity gate is disabled at import (it is a no-op for these
well-formed molecules — they all OPSIN-parse), so the test needs no JVM and the
frozen names equal the true production names.
"""
from __future__ import annotations

import csv
import os
import pathlib

# Disable the OPSIN validity gate BEFORE importing orthonym -> JVM-free
# (the gate is a no-op for these molecules; names are unchanged)...
#... and put the environment back right after the import. Under xdist every
# worker imports every test module at collection, so a module-level write that
# stayed in os.environ switched the gates off in every child process the rest of
# the suite spawned (TRIAGE C1: subprocess tests saw gate-off names). The namer
# reads these variables once, at import, so restoring them changes nothing here.
_ENV_BEFORE = {"ORTHONYM_DISABLE_OPSIN_VALIDITY_GATE":
               os.environ.get("ORTHONYM_DISABLE_OPSIN_VALIDITY_GATE")}
os.environ.setdefault("ORTHONYM_DISABLE_OPSIN_VALIDITY_GATE", "1")

import pytest

from rdkit import RDLogger

from orthonym.namer import name_with_tree
from orthonym.assembly.name_tree_to_string import (
    SERIALIZER_PRODUCTION_CLASSES,
    _assemble_explicit_fields,
)

for _k, _v in _ENV_BEFORE.items():   # restore: see the note above the import
    if _v is None:
        os.environ.pop(_k, None)
    else:
        os.environ[_k] = _v

RDLogger.DisableLog("rdApp.*")

_SNAPSHOT = (
    pathlib.Path(__file__).resolve().parents[2]  #.../tests
    / "fixtures" / "serializer_flip_snapshot.csv"
)


def _load_snapshot():
    with open(_SNAPSHOT, newline="") as f:
        return list(csv.DictReader(f))


_ROWS = _load_snapshot()
_FLIPPED = [r for r in _ROWS if r["class_id"] in SERIALIZER_PRODUCTION_CLASSES]
_CARRIERS = [r for r in _ROWS if r["class_id"] not in SERIALIZER_PRODUCTION_CLASSES]


pytestmark = pytest.mark.unit


def test_snapshot_has_flipped_and_carrier_rows():
    """The frozen snapshot must cover BOTH a flipped class and carrier anchors."""
    assert len(_FLIPPED) >= 12, f"expected >=12 flipped rows, got {len(_FLIPPED)}"
    assert len(_CARRIERS) >= 2, f"expected >=2 carrier anchors, got {len(_CARRIERS)}"


@pytest.mark.parametrize(
    "row", _FLIPPED, ids=[r["smiles"] for r in _FLIPPED]
)
def test_flipped_class_explicit_field_byte_identical(row):
    """A flipped class is serialized via the EXPLICIT-field path byte-identically.

    Asserts on _assemble_explicit_fields directly (NOT name_tree_to_string) so a
    regression that re-introduces a str carrier — making the flip a silent no-op —
    is caught (Pitfall 2 guard)."""
    r = name_with_tree(row["smiles"])
    tree = r.tree
    assert tree is not None, f"no tree for {row['smiles']}"
    assert tree.class_id in SERIALIZER_PRODUCTION_CLASSES, (
        f"{row['smiles']} class_id={tree.class_id!r} drifted out of the flip set"
    )
    # The flip MUST have dropped the str carrier so the explicit path runs.
    assert tree.fragment_legacy is None, (
        f"{row['smiles']}: a flipped class must carry fragment_legacy=None "
        f"(got {tree.fragment_legacy!r}) — the flip would be a silent no-op"
    )
    out = _assemble_explicit_fields(tree, "pin")
    assert out == row["production_name"], (
        f"{row['smiles']}: explicit-field serializer diverged from the frozen "
        f"production name (got {out!r}, expected {row['production_name']!r})"
    )
    # And the production name itself is unchanged from the frozen snapshot.
    assert r.name == row["production_name"], (
        f"{row['smiles']}: production name drifted (got {r.name!r}, "
        f"frozen {row['production_name']!r})"
    )


@pytest.mark.parametrize(
    "row", _CARRIERS, ids=[r["smiles"] for r in _CARRIERS]
)
def test_carrier_manifest_stable(row):
    """A carrier-anchor row must stay OUT of the flip set (manifest stability)."""
    r = name_with_tree(row["smiles"])
    tree = r.tree
    assert tree is not None
    assert tree.class_id not in SERIALIZER_PRODUCTION_CLASSES, (
        f"{row['smiles']} class_id={tree.class_id!r} was accidentally flipped"
    )
    assert r.name == row["production_name"], (
        f"{row['smiles']}: carrier name drifted (got {r.name!r}, "
        f"frozen {row['production_name']!r})"
    )
