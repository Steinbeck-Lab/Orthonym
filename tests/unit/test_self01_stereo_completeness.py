# tests/unit/test_self01_stereo_completeness.py
from orthonym import namer


def test_omission_detected_azo():
    # input has a defined E on N=N; the stereo-free name drops it
    assert namer._name_omits_input_stereo(
        r"C(/N=N/c1ccccc1)1=CC=CC=C1",          # input: (E)-azobenzene
        "C1(=CC=CC=C1)N=NC1=CC=CC=C1") is True   # name parse: no E/Z


def test_no_omission_when_complete():
    # a fully-specified name over a fully-specified input
    s = r"C[C@H](N)C(=O)O"
    assert namer._name_omits_input_stereo(s, s) is False


def test_stereo_free_input_is_not_omission():
    # input defines nothing -> a name cannot "omit" -> not an omission
    assert namer._name_omits_input_stereo("NCCC(N)C(=O)O", "NCCC(N)C(=O)O") is False


def test_overspecified_is_not_omission():
    # name defines MORE than input -> not an omission (separate case)
    assert namer._name_omits_input_stereo("CC(N)C(=O)O", r"C[C@H](N)C(=O)O") is False


def test_unparseable_is_not_omission():
    assert namer._name_omits_input_stereo("not a smiles", "also not") is False
