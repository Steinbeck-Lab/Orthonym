"""Biaryl stereogenic axes (``validation/axis_identity.py``).

 "Stereogenic axis" (the Blue Book,:44824): "Stereodescriptors 'M' and 'P' are used in preferred
IUPAC names. The lowest locant on the axis is cited before the stereodescriptor." "Specification of
stereogenic axes" (:49797) PIN examples: '(1M)-6,6′-dinitro[1,1′-biphenyl]-2,2′-dicarboxylic acid' (:49803),
'(1P)-2′,5′-dimethoxy-6-nitro[1,1′-biphenyl]-2-carboxylic acid' (:49805), '(1M)-2′,6-diamino-6′-methoxy[1,1′-biphenyl]-
2-carboxylic acid' (:49807). A name that leaves an axis out is not the PIN,:44624). OPSIN 2.9.0 reads no
axial descriptor; OPSIN main reads '(M)-' without a locant (dan2097/opsin#356).
"""
import pytest
from rdkit import Chem

from orthonym.rules.stereochemistry import strip_stereo
from orthonym.validation import axis_identity as ax
from tests.support.opsin_marks import needs_axial_opsin, release_axial_only

BINOL_M = "Oc1ccc2ccccc2c1-c1c(O)ccc2ccccc12 |wU:10.11|"
BINOL_P = "Oc1ccc2ccccc2c1-c1c(O)ccc2ccccc12 |wD:10.11|"
BINOL = "Oc1ccc2ccccc2c1-c1c(O)ccc2ccccc12"
#: the three biphenyl PINs of, drawn from OPSIN main's reading of their locant-free form
BOOK = [
    ("O=C(O)c1cccc([N+](=O)[O-])c1-c1c(C(=O)O)cccc1[N+](=O)[O-] |wU:11.11|",
     "(1M)-6,6'-dinitro[1,1'-biphenyl]-2,2'-dicarboxylic acid"),
    ("COc1ccc(OC)c(-c2c(C(=O)O)cccc2[N+](=O)[O-])c1 |wU:8.7|",
     "(1P)-2',5'-dimethoxy-6-nitro[1,1'-biphenyl]-2-carboxylic acid"),
    ("COc1cccc(N)c1-c1c(N)cccc1C(=O)O |wD:8.7|",
     "(1M)-2',6-diamino-6'-methoxy[1,1'-biphenyl]-2-carboxylic acid"),
]
#: an axis to an aryl SUBSTITUENT (the book's '(2P)' phenol,:49828), an axis inside an ester's alcohol part (:49830)
#: and a molecule with a second biaryl bond: no descriptor is written (OPSIN main and RDKit label an aryl-substituent
#: axis differently, so the read-back never matches), the name ships labelled below the PIN
OMITTED = [
    ("Cc1cc(C)c(-c2c(CO)ccc3ccccc23)c(O)c1 |wU:6.6|", "2-[2-(hydroxymethyl)naphthalen-1-yl]-3,5-dimethylphenol"),
    ("CC1(C)C[C@@H]1C(=O)Oc1ccc2ccccc2c1-c1c(O)ccc2ccccc12 |wU:17.19|",
     "1-(2-hydroxynaphthalen-1-yl)naphthalen-2-yl (1S)-2,2-dimethylcyclopropane-1-carboxylate"),
    ("Cc1ccccc1-c1ccc2ccccc2c1-c1c(C)ccc2ccccc12 |wD:17.20|",
     "1-(2-methylnaphthalen-1-yl)-2-(2-methylphenyl)naphthalene"),
]


def _best_effort():
    from orthonym.cli import _tiered_namer
    return _tiered_namer("pin", "best-effort")


def _pin_tier():
    from orthonym.cli import _tiered_namer
    return _tiered_namer("pin", "pin")


@pytest.mark.parametrize("name, twin", [
    ("(1M)-[1,1'-binaphthalene]-2,2'-diol", "(M)-[1,1'-binaphthalene]-2,2'-diol"),
    ("(2P)-2-[2-(hydroxymethyl)naphthalen-1-yl]-3,5-dimethylphenol",
     "(P)-2-[2-(hydroxymethyl)naphthalen-1-yl]-3,5-dimethylphenol"),
    ("(11aM)-1,11-dinitro-5,7-dihydro-6H-dibenzo[a,c][7]annulen-6-one",
     "(M)-1,11-dinitro-5,7-dihydro-6H-dibenzo[a,c][7]annulen-6-one"),
    ("(1M,2S)-x", "(M,2S)-x"),
    ("(1Ra)-y", "(Ra)-y"),
])
def test_twin_leaves_out_the_locant_of_one_axial_descriptor(name, twin):
    assert ax.axial_locant_twin(name) == twin


@pytest.mark.parametrize("name", [
    "(M)-[1,1'-binaphthalene]-2,2'-diol",             # no locant
    "(1M,6P)-1,8-dichloroocta-1,2,6,7-tetraene",      # two axial descriptors: no twin
    "(2S)-butan-2-ol", "[1,1'-binaphthalene]-2,2'-diol", "(4-methylphenyl)methanol",
])
def test_no_twin(name):
    assert ax.axial_locant_twin(name) == name


def test_axes_key_is_the_same_for_the_two_halves_of_a_symmetric_biaryl():
    m, p = ax._axes(ax._axis_mol(BINOL_M)), ax._axes(ax._axis_mol(BINOL_P))
    assert list(m.values()) == ["M"] and list(p.values()) == ["P"] and m.keys() == p.keys()


def test_kekule_wedge_of_opsin_is_kept():
    # OPSIN main's reading of '(M)-6,6'-dinitro[1,1'-biphenyl]-2,2'-dicarboxylic acid': RDKit reads this string
    # with no atropisomer bond (the wedge sits on a Kekule double bond)
    ext = ("[N+](=O)([O-])C=1C=CC=C(C1C=1C(=CC=CC1[N+](=O)[O-])C(=O)O)C(=O)O "
           "|$_AV:N;O;O';6;5;4;3;2;1;1';2';3';4';5';6';N';O'';O''';C';O''';O'';C;O';O$,wD:8.8|")
    assert not any(b.GetStereo() in ax._ATROP for b in Chem.MolFromSmiles(ext).GetBonds())
    read = ax._read_extended(ext)
    assert ax._axes(read) == ax._axes(ax._axis_mol(BOOK[0][0]))


def test_verdict_without_opsin():
    assert ax.axis_verdict(BINOL, "[1,1'-binaphthalene]-2,2'-diol") == "n/a"
    assert ax.axis_verdict(BINOL_M, "[1,1'-binaphthalene]-2,2'-diol") == "omission"


def test_strip_stereo_keeps_an_axial_block():
    # the stereo-stripped carve-out of the validity gate must never judge an axial descriptor on constitution alone
    assert strip_stereo("(1M)-[1,1'-binaphthalene]-2,2'-diol") == "(1M)-[1,1'-binaphthalene]-2,2'-diol"


@needs_axial_opsin
@pytest.mark.opsin_gate
def test_verdict_reads_the_axis():
    assert ax.axis_verdict(BINOL_M, "(1M)-[1,1'-binaphthalene]-2,2'-diol") == "match"
    assert ax.axis_verdict(BINOL_M, "(1P)-[1,1'-binaphthalene]-2,2'-diol") == "mismatch"
    assert ax.axis_decorated(BINOL_P, "[1,1'-binaphthalene]-2,2'-diol") == "(1P)-[1,1'-binaphthalene]-2,2'-diol"


@needs_axial_opsin
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, name", [(BINOL_M, "(1M)-[1,1'-binaphthalene]-2,2'-diol"),
                                          (BINOL_P, "(1P)-[1,1'-binaphthalene]-2,2'-diol")] + BOOK)
def test_book_pins(smiles, name):
    for namer in (_best_effort(), _pin_tier()):
        row = namer.name_tiered(smiles)
        assert (row["name"], row["tier"], row["stereo_unexpressed"]) == (name, "pin_verified", False)


@needs_axial_opsin
@pytest.mark.opsin_gate
def test_one_process_keeps_each_configuration():
    namer = _best_effort()
    got = [namer.name_tiered(s)["name"] for s in (BINOL_M, BINOL_P, BINOL_M, BINOL, BINOL_P, BINOL_P)]
    assert got == ["(1M)-[1,1'-binaphthalene]-2,2'-diol", "(1P)-[1,1'-binaphthalene]-2,2'-diol",
                   "(1M)-[1,1'-binaphthalene]-2,2'-diol", "[1,1'-binaphthalene]-2,2'-diol",
                   "(1P)-[1,1'-binaphthalene]-2,2'-diol", "(1P)-[1,1'-binaphthalene]-2,2'-diol"]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, name", OMITTED)
def test_axis_left_out_is_no_pin(smiles, name):
    row = _best_effort().name_tiered(smiles)
    assert (row["name"], row["tier"], row["stereo_unexpressed"]) == (name, "systematic_verified", True)


@release_axial_only
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, name", [(BINOL_M, "[1,1'-binaphthalene]-2,2'-diol"),
                                          (BOOK[0][0], "6,6'-dinitro[1,1'-biphenyl]-2,2'-dicarboxylic acid")])
def test_release_opsin_names_the_axis_input_below_the_pin(smiles, name):
    row = _best_effort().name_tiered(smiles)
    assert (row["name"], row["tier"], row["stereo_unexpressed"]) == (name, "systematic_verified", True)
    assert _pin_tier().name_tiered(smiles)["tier"] == "abstain"
    assert _best_effort().name_tiered(BINOL)["tier"] == "pin_verified"     # no axis: unchanged


@pytest.mark.parametrize("name", ["dihydrogen (32P)phosphate", "(226Ra)radium", "N,P,P-trimethylphosphinic amide",
                                  "dichlorido(κ2N,P)platinum", "(2S)-butan-2-ol"])
def test_isotope_and_ligand_letters_are_no_axial_descriptor(name):
    #: '(32P)' is a nuclide descriptor; an axial descriptor is an item of a stereo group
    assert ax.axial_descriptors(name) == 0 and ax.axial_locant_twin(name) == name


@pytest.mark.opsin_gate
def test_isotope_labelled_axis_input_keeps_its_name():
    # review finding: '(32P)' was read as an axial descriptor, and the name of this input was withdrawn
    row = _best_effort().name_tiered("O[32P](=O)(O)Oc1ccc2ccccc2c1-c1c(O)ccc2ccccc12 |wD:14.14|")
    assert (row["name"], row["tier"], row["stereo_unexpressed"]) == (
        "1-(2-hydroxynaphthalen-1-yl)naphthalen-2-yl dihydrogen (32P)phosphate", "systematic_verified", True)


@pytest.mark.opsin_gate
def test_redundant_brackets_keep_the_axis():
    # review finding: the redundant-bracket rewrite of the input wrote no CX layer, and the axis was lost
    row = _best_effort().name_tiered("[OH]c1ccc2ccccc2c1-c1c(O)ccc2ccccc12 |wU:10.11|")
    assert row["stereo_unexpressed"] is not True or row["tier"] != "pin_verified"
    assert row["name"] in ("[1,1'-binaphthalene]-2,2'-diol", "(1M)-[1,1'-binaphthalene]-2,2'-diol")
    assert (row["name"] == "(1M)-[1,1'-binaphthalene]-2,2'-diol") == (row["tier"] == "pin_verified")
