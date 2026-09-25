"""A decomposition amide glue must not convert an acid group it did not cut
(pre-existing-failures plan, Task 4).

The decomposition engine names an amide by cutting its C(=O)-N bond, naming the
two fragments, and joining the names. For a POLY-acid fragment both joins
converted EVERY acid group, although only ONE was cut: the acyl prefix
'butanedioyl' is the DIVALENT -CO-CH2-CH2-CO-, the Blue Book), so
the free CH2-COOH vanished from 'N-({(2S)-2-[methyl(7-nitro2,1,3-benzoxadiazol-
4-yl)amino]butanedioyl})(2S)-2-aminopropanoic acid' (TRIAGE row 17,
CHEBI:139249), and the amide 'butanediamide' makes the free acid an amide. The
free acid is the senior class,:18158: acids above amides), so
neither form is the name; both joins now decline, as the ester assembler already
did.

Measured and NOT changed: an 'N-<acyl>' float onto an amine fragment whose name
has no amine suffix ('N-octadecanoyl(2S,3R,4E)-2-amino-...-1,3-diol'). It is not
IUPAC grammar,:26225, puts N- on the amine suffix), but OPSIN reads it
as intended and it round-trips exactly on 10 m1500 rows, so refusing it is a
spelling change, not a 0-wrong fix.
"""
import pytest

from orthonym.decomposition.fragment_assembly import _assemble_amide, _is_poly_acid_name

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("name,expected", [
    ("acetic acid", False),
    ("(2S)-3-phenylpropanoic acid", False),
    ("cyclohexanecarboxylic acid", False),
    ("(2S)-2-[methyl(7-nitro-2,1,3-benzoxadiazol-4-yl)amino]butanedioic acid", True),
    ("benzene-1,4-dicarboxylic acid", True),
    ("propane-1,2,3-tricarboxylic acid", True),
])
def test_is_poly_acid_name(name, expected):
    assert _is_poly_acid_name(name) is expected


def test_poly_acid_acyl_float_declines():
    # row 17 (CHEBI:139249): the recorded call, verbatim
    got = _assemble_amide(
        {"amine": "(2S)-2-aminopropanoic acid",
         "acid": "(2S)-2-[methyl(7-nitro2,1,3-benzoxadiazol-4-yl)amino]butanedioic acid"},
        "pin",
        {"amine": "C[C@H](N)C(=O)O",
         "acid": "CN(c1ccc([N+](=O)[O-])c2nonc12)[C@@H](CC(=O)O)C(=O)O"})
    assert got is None


def test_poly_acid_amide_suffix_declines():
    # the simple-amine branch would write 'N-methylbutanediamide' for ONE amide bond
    got = _assemble_amide(
        {"amine": "methanamine", "acid": "butanedioic acid"}, "pin",
        {"amine": "CN", "acid": "OC(=O)CCC(=O)O"})
    assert got is None


@pytest.mark.parametrize("names,smiles,expected", [
    ({"acid": "acetic acid", "amine": "methanamine"},
     {"acid": "CC(=O)O", "amine": "CN"}, "N-methylacetamide"),
    ({"acid": "benzoic acid", "amine": "cyclohexanamine"},
     {"acid": "OC(=O)c1ccccc1", "amine": "NC1CCCCC1"}, "N-cyclohexylbenzamide"),
])
def test_valid_amide_glue_unchanged(names, smiles, expected):
    assert _assemble_amide(names, "pin", smiles) == expected
