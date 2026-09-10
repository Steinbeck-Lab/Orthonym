"""The mismatch classifier must separate mechanisms, not restate the symptom.

`stereo_only` is the load-bearing case: a name that drops a stereodescriptor
yields a DIFFERENT InChIKey while the constitution is identical, so lumping it
with atom drops would size a stereo defect as an atom-coverage defect.

Import style follows tests/unit/test_cluster_coverage_tell.py:22 -- `eval/` has
no __init__.py, so `from eval.classify_self01 import...` raises
ModuleNotFoundError even with pythonpath=["."].
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from classify_self01 import classify_mismatch  # noqa: E402


@pytest.mark.parametrize("smi_in,smi_out,expected", [
    # 13 heavy atoms named as 4 -> atoms dropped
    ("CCNc1nc(O)nc(NCC)n1", "CC(=O)O", "atom_drop"),
    # more atoms out than in
    ("CCO", "CCOCCOCC", "atom_fabricated"),
    # same heavy-atom count, different connectivity and different formula
    ("CC(=O)OC", "CCCCC", "same_count_wrong"),
    # identical constitution, stereo dropped
    ("C/C=C/C", "CC=CC", "stereo_only"),
    # OPSIN produced nothing
    ("CCO", None, "unparseable"),
    ("CCO", "", "unparseable"),
    # OPSIN produced something RDKit cannot read
    ("CCO", "C1CC", "unparseable"),
    # unreadable input
    ("not-a-smiles", "CCO", "input_unreadable"),
])
def test_classify_mismatch(smi_in, smi_out, expected):
    assert classify_mismatch(smi_in, smi_out) == expected


def test_identical_molecules_are_not_a_mismatch():
    with pytest.raises(ValueError):
        classify_mismatch("CCO", "CCO")


def test_same_formula_different_connectivity_is_flagged_as_such():
    """An isomer is a distinct mechanism from a miscounted skeleton.

    methyl acetate vs propanoic acid: same formula C3H6O2, same heavy-atom
    count, different constitution. Calling that `same_count_wrong` alongside a
    genuine skeleton error would merge a rearrangement defect with a counting
    defect, and they need different fixes.
    """
    assert classify_mismatch("CC(=O)OC", "CCC(=O)O") == "tautomer_or_charge"


def test_stereo_is_checked_before_the_atom_count():
    """Ordering guard: a stereo-only difference has delta 0, so a naive
    count-first classifier would fall through to `same_count_wrong` and hide the
    whole stereo class inside a constitution class."""
    assert classify_mismatch("N[C@@H](C)C(=O)O", "NC(C)C(=O)O") == "stereo_only"
