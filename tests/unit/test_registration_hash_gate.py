""" C6: verdict uses a RegistrationHash-backed stereo layer.

The InChIKey skeleton block is stereo-insensitive (-07), so the pre-C6
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
    # Policy reversed by 0c4d4a2d3 ("reject stereo OMISSION as a mismatch (0-wrong; 20/500
    # default stereo-drops)"): a name that under-specifies stereo (the parse drops the
    # defined centre) describes a less specific, different molecule, so on the primary
    # path it is a mismatch. The ignore_stereo=True carve-out below stays tolerant.
    assert namer._self_consistency_verdict(
        "C[C@H](N)C(=O)O", "CC(N)C(=O)O") == "mismatch"


def test_stereo_difference_is_ok_when_ignore_stereo():
    # The:1173 carve-out compares full-stereo input vs a stereo-STRIPPED parse.
    assert namer._self_consistency_verdict(
        "C[C@H](N)C(=O)O", "CC(N)C(=O)O", ignore_stereo=True) == "ok"


def test_unparseable_is_inconclusive():
    assert namer._self_consistency_verdict("not_a_smiles", "CCO") == "inconclusive"


def test_neutral_input_charge_ambiguous_name_still_ok():
    # Policy reversed by baeb8d1ab (" compares the net charge and the protonation
    # flag for every input"): a neutral input has no protonation exemption. OPSIN reads
    # 'methyl phosphate' as the dianion; the neutral ester COP(=O)(O)O is 'methyl dihydrogen
    # phosphate' "Esters of mononuclear noncarbon oxoacids", the Blue Book
    # 'P(O)(O-CH3)(OH)2 methyl dihydrogen phosphate (PIN)'), so the dianion parse is a
    # different species -> mismatch.
    assert namer._self_consistency_verdict(
        "COP(=O)(O)O", "COP(=O)([O-])[O-]") == "mismatch"


def test_charged_input_charge_drop_is_mismatch():
    # a charged input whose name drops the charge IS a leak.
    assert namer._self_consistency_verdict("[O-]O", "OO") == "mismatch"


def test_tautomer_difference_with_same_stereo_is_ok():
    # 5'-inosinic acid: input is the 6-oxo purine, OPSIN parses the retained name
    # to the 6-hydroxy tautomer; the sugar stereo is IDENTICAL. A tautomer diff
    # must NOT read as a stereo conflict (regression guard — this broke the 1652
    # gold row before TAUTOMER_HASH replaced CANONICAL_SMILES).
    inp = "O=c1[nH]cnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O"
    opsin = "O=P(O)(O)OC[C@H]1O[C@@H](n2cnc3c(O)ncnc32)[C@H](O)[C@@H]1O"
    assert namer._self_consistency_verdict(inp, opsin) == "ok"
