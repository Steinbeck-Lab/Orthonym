""" a phase Task 6 (C4): known-positive control tests for the stereo-honesty
closure (C1 predicate + C2a steroid guard + C2b amino-acid guard + C3 gate
backstop).

Two paired controls: a FLAT (stereo-undefined) input that must NOT emit a
config-implying stereo name (the fabrication class that motivated the closure),
and a fully STEREO-DEFINED input of the same family that must still get its
correct, specific PIN -- proving the guard is not overbroad and did not regress
a real positive into an abstention.

Reference:.the workflow tooling/sdd/2026-08-16--phase1-stereo-honesty/task-6-brief.md
"""
import logging

logging.disable(logging.CRITICAL)

from orthonym.namer import Orthonym


def test_flat_cholesterol_not_fabricated_default():
    """Cholesterol with all stereo stripped must not be named as a specific
    'cholest...' stereoparent derivative on the DEFAULT (PIN) path -- that would
    assert a stereochemical configuration the flat input never specified."""
    nm = Orthonym()
    out = nm.name_tiered(
        "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C"
    ).get("name") or ""
    assert "cholest" not in out, (
        f"flat (stereo-undefined) cholesterol skeleton fabricated a cholest- name: {out!r}"
    )


def test_defined_androstenedione_kept():
    """The fully stereo-defined androstenedione must still emit its exact,
    specific PIN -- the guard must not suppress a legitimate stereo-defined
    natural-product name."""
    nm = Orthonym()
    smi = "C[C@@]12C(CC[C@H]1[C@@H]1CCC3=CC(CC[C@]3(C)[C@H]1CC2)=O)=O"
    assert nm.name_tiered(smi).get("name") == "androst-4-ene-3,17-dione"
