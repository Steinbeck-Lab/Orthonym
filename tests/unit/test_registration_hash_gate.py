"""v30 C6: SELF-01 verdict uses a RegistrationHash-backed stereo layer.

The InChIKey skeleton block is stereo-insensitive (ADR-18-07), so the pre-C6
verdict judged a wrong-stereoisomer name as "ok". C6 keeps the gold-safe
constitution+charge logic and ADDS a stereo-CONFLICT guard on the stereo-strict
primary path, while the BBR-GATE stereo carve-out stays stereo-insensitive via
``ignore_stereo=True``.
"""
import pytest
from orthonym import namer


@pytest.mark.parametrize("a,b", [
    ("CCO", "OCC"),
    ("OC(=O)C", "CC(=O)O"),
])
def test_same_molecule_ok(a, b):
    assert namer._self_consistency_verdict(a, b) == "ok"


@pytest.mark.parametrize("a,b", [
    ("c1ccccc1", "C1CCCCC1"),   # benzene vs cyclohexane
    ("CCO", "CCC"),             # ethanol vs propane
])
def test_constitution_difference_mismatch(a, b):
    assert namer._self_consistency_verdict(a, b) == "mismatch"


def test_stereo_conflict_is_mismatch_on_primary_path():
    # L- vs D-alanine: same constitution, CONFLICTING stereo -> mismatch (C6 improvement).
    assert namer._self_consistency_verdict(
        "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O") == "mismatch"


def test_stereo_omission_is_tolerated_on_primary_path():
    # A name that under-specifies stereo (parse drops the centre) is NOT a
    # constitutional error -> ok (breadth-preserving; 0-wrong is about constitution).
    assert namer._self_consistency_verdict(
        "C[C@H](N)C(=O)O", "CC(N)C(=O)O") == "ok"


def test_stereo_difference_is_ok_when_ignore_stereo():
    # The :1173 carve-out compares full-stereo input vs a stereo-STRIPPED parse.
    assert namer._self_consistency_verdict(
        "C[C@H](N)C(=O)O", "CC(N)C(=O)O", ignore_stereo=True) == "ok"


def test_unparseable_is_inconclusive():
    assert namer._self_consistency_verdict("not_a_smiles", "CCO") == "inconclusive"


def test_neutral_input_charge_ambiguous_name_still_ok():
    # methyl phosphate (neutral) vs its OPSIN dianion parse: protonation-ambiguous,
    # NOT a wrong molecule -> must stay ok (documented exempt case).
    assert namer._self_consistency_verdict(
        "COP(=O)(O)O", "COP(=O)([O-])[O-]") == "ok"


def test_charged_input_charge_drop_is_mismatch():
    # a charged input whose name drops the charge IS a leak.
    assert namer._self_consistency_verdict("[O-]O", "OO") == "mismatch"
