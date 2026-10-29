"""The stereo of a radical's name is composed with OPSIN's ``-r``, in the radical offer and nowhere else.

The best-effort tier composes the stereo of the input onto the name of its stereo-free structure: it re-anchors the
descriptors to the locants OPSIN gives the name (``validation.opsin_roundtrip.opsin_atom_locant_map``) and checks the
candidate with ``opsin_roundtrip_check``. OPSIN reads a name with a free valence ('(1-methoxy-1-oxopropan-2-yl)amino')
only with its ``-r`` flag, and ``validation/read_back_policy.py`` gives that flag to a structure that has radical
electrons. Both calls read the name without it, so the composition gave nothing for every radical with stereo, and the
radical kept its abstention although the stereo-free structure had a name.

The reading with ``-r`` is the radical offer's (``read_back_policy.radical_last_resort``): the last resort of the
tier, made only for a molecule that nothing else names, so no name that ships today depends on it. And OPSIN's SMILES
of a name does not mark a radical, so the full InChIKey, which sees neither radical electrons nor bond order, is not
enough with ``-r``: the check also compares the radical graph (``validation.radical_identity``).
"""
import pytest
from rdkit import Chem

from orthonym.validation.opsin_roundtrip import opsin_atom_locant_map, opsin_roundtrip_check
from orthonym.validation.radical_identity import radical_identity_verdict
from orthonym.validation.read_back_policy import radical_last_resort
from tests.support.rt_assert import _independent_parse, name_best_effort, name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


def _assert_named(smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] == expected, res
    assert res["tier"] == "systematic_verified" and not res["is_pin"], res
    assert name_is_rt_exact(res["name"], smiles), res
    parsed = _independent_parse(res["name"], smiles)
    assert parsed is not None and radical_identity_verdict(smiles, parsed) == "ok", (res, parsed)


STEREO_ROWS = [
    ("C[C@@H](C(=O)OC)[NH]", "(2S)-(1-methoxy-1-oxopropan-2-yl)amino"),
    ("C/C=N\\[O]", "(1Z)-(ethylideneamino)oxy"),
]


@pytest.mark.parametrize("smiles,expected", STEREO_ROWS)
def test_the_stereo_of_a_radical_is_composed_on_its_name(smiles, expected):
    """The stereo of a radical is composed on its name: every radical with stereo kept its abstention before."""
    _assert_named(smiles, expected)


def test_the_stereo_re_anchoring_reads_a_radical_name_with_its_radical_flag():
    smiles, name = "C[C@@H](C(=O)OC)[NH]", "(2S)-(1-methoxy-1-oxopropan-2-yl)amino"
    with radical_last_resort():
        assert opsin_roundtrip_check(smiles, name)["passed"] is True
        assert opsin_roundtrip_check(smiles, "(2R)-(1-methoxy-1-oxopropan-2-yl)amino")["passed"] is False
        assert opsin_atom_locant_map("(1-methoxy-1-oxopropan-2-yl)amino", Chem.MolFromSmiles(smiles))


def test_outside_the_radical_offer_a_radical_name_is_read_as_it_always_was():
    """No name that ships today depends on the reading with ``-r``: outside the offer a name with a free valence fails
    closed, as it did."""
    smiles, name = "C[C@@H](C(=O)OC)[NH]", "(2S)-(1-methoxy-1-oxopropan-2-yl)amino"
    assert opsin_roundtrip_check(smiles, name)["error"] == "opsin_parse_failed"
    assert opsin_atom_locant_map("(1-methoxy-1-oxopropan-2-yl)amino", Chem.MolFromSmiles(smiles)) is None


def test_the_full_key_is_not_the_radical_graph():
    """'but-3-en-2-yl' parses to C[CH]C=C, which has the full InChIKey of [CH2]C=CC (the key sees neither radical
    electrons nor bond order): another molecule, which the read of the name with -r must not accept."""
    for reading in (False, True):
        if reading:
            with radical_last_resort():
                check = opsin_roundtrip_check("[CH2]C=CC", "but-3-en-2-yl")
        else:
            check = opsin_roundtrip_check("[CH2]C=CC", "but-3-en-2-yl")
        assert check["passed"] is False, (reading, check)
    with radical_last_resort():
        assert opsin_roundtrip_check("[CH2]C=CC", "but-2-en-1-yl")["passed"] is True
        assert opsin_roundtrip_check("C[CH]C=C", "but-3-en-2-yl")["passed"] is True


def test_a_closed_shell_structure_is_still_read_without_the_radical_flag():
    assert opsin_roundtrip_check("CCO", "ethanol")["passed"] is True
    with radical_last_resort():
        assert opsin_roundtrip_check("CCO", "ethanol")["passed"] is True
        # a name with a free valence is not the name of a closed-shell molecule, whatever the offer reads
        assert opsin_roundtrip_check("CCO", "ethyl")["passed"] is False
        assert opsin_atom_locant_map("ethyl", Chem.MolFromSmiles("CC")) is None


# --- the stereo cascade of the floor reads a radical with -r, in the stereo pass of the radical offer ----------------

NESTED_STEREO_ROWS = [
    # three stereocentres in three nested substituents: the composition at the front of the name cites '(1S,2R,1R)'
    # with repeated locants (declined), and each descriptor belongs in the bracket of its own group
    ("C[C@H](C1=CC=CC=C1)N2C[C@@H]2[C@H]([C]3[CH][CH][CH][CH]3)O",
     "1-[(1S)-hydroxy{(2R)-1-[(1R)-1-phenylethyl]aziridin-2-yl}methyl]cyclopentane-1,2,3,4,5-pentayl"),
]


@pytest.mark.parametrize("smiles,expected", NESTED_STEREO_ROWS)
def test_the_descriptors_of_every_branch_are_cited_in_the_bracket_of_the_branch(smiles, expected):
    """``### **** NAMING OF STEREOISOMERS`` (the Blue Book), sentence:44643: "In preferred IUPAC names,
    stereodescriptors are placed immediately at the front of the part of the name to which they relate". The floor's stereo
    cascade (``universal_substituent._resolve_floor_stereo``) builds that name and verifies it with OPSIN's ``-r`` in the
    stereo pass of the radical offer (``read_back_policy.radical_floor_stereo``)."""
    _assert_named(smiles, expected)


def test_the_stereo_pass_does_not_change_a_name_that_the_composition_gives():
    """The stereo pass is the last class of the offer: a molecule whose plain name takes the stereo at the front keeps it."""
    _assert_named("C[C@@H](O)[C]1[CH][CH][CH][CH]1", "(1R)-1-(1-hydroxyethyl)cyclopentane-1,2,3,4,5-pentayl")


def test_the_atom_coverage_reads_a_radical_name_with_its_flag_only_in_the_stereo_pass():
    from orthonym.validation.atom_coverage import find_opsin_jar, validate_atom_coverage
    from orthonym.validation.read_back_policy import radical_floor_stereo
    mol = Chem.MolFromSmiles("C[C@@H](O)[C]1[CH][CH][CH][CH]1")
    name = "1-[(1R)-1-hydroxyethyl]cyclopentane-1,2,3,4,5-pentayl"
    if find_opsin_jar() is None:
        pytest.skip("no OPSIN")
    assert validate_atom_coverage(mol, name).method != "parse_back"      # read without -r: a free valence does not parse
    with radical_floor_stereo():
        assert validate_atom_coverage(mol, name).method == "parse_back"


def test_the_stereo_pass_is_not_active_while_the_offer_reads_a_name():
    """The pass builds its names inside the ``radical_floor_stereo`` block and yields them after it, so the offer's own checks
    of a name (the exit check, the composition) read a radical as they always read."""
    from orthonym.assembly.radical_floor import _further_names
    from orthonym.validation.read_back_policy import radical_floor_stereo_active
    mol = Chem.MolFromSmiles("C[C@@H](O)[C]1[CH][CH][CH][CH]1")
    seen = []
    with radical_last_resort():
        for _name in _further_names(mol):
            seen.append(radical_floor_stereo_active())
    assert seen and not any(seen)
