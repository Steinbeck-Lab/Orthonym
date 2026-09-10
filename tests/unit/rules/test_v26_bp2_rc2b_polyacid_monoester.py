""" (accuracy) — poly-acid mono-ester is acid-senior, not an ester.

The decomposition ester assembler (_assemble_ester) blindly built 'alkyl...ate'
from fragment names. For a mono-ester of a di-acid (e.g. ethyl hydrogen phthalate
CCOC(=O)c1ccccc1C(=O)O) it produced 'ethyl benzene-1,2-dicarboxylate' — which
drops the 'hydrogen' the free -COOH needs, so OPSIN reads it as an anion
(RT-MISMATCH). Per (carboxylic acid > ester) /, the un-esterified
free acid is the senior principal group, so the ester functional-class name is not
the PIN.

The guard declines when the acid fragment is a poly-acid (its systematic name
carries a multiplied '…dicarboxylic acid' / '…dioic acid' suffix). The molecule
then fails closed (accuracy #1: never emit a wrong name) rather than shipping the
underspecified ester name. The chain analogue was already correct
('6-ethoxy-6-oxohexanoic acid'); full diesters (diethyl …dicarboxylate) and
simple mono-acid esters (ethyl benzoate) are unaffected.
"""
from orthonym.decomposition.fragment_assembly import _assemble_ester
from orthonym.namer import name_compound


# --- pure-function contract (authoritative) --------------------------------

def test_poly_acid_mono_ester_declined():
    assert _assemble_ester(
        {"acid": "benzene-1,2-dicarboxylic acid", "alkyl": "ethanol"}, "pin"
    ) is None
    assert _assemble_ester(
        {"acid": "butanedioic acid", "alkyl": "methanol"}, "pin"
    ) is None


def test_mono_acid_ester_unaffected():
    assert _assemble_ester(
        {"acid": "benzoic acid", "alkyl": "ethanol"}, "pin"
    ) == "ethyl benzoate"
    assert _assemble_ester(
        {"acid": "acetic acid", "alkyl": "ethanol"}, "pin"
    ) == "ethyl acetate"


# --- full name: wrong name gone; regressions preserved ---------------------

def test_ring_mono_ester_no_longer_wrong_name():
    # was 'ethyl benzene-1,2-dicarboxylate' (RT-MISMATCH); now fails closed.
    out = name_compound("CCOC(=O)c1ccccc1C(=O)O")
    assert "dicarboxylate" not in out  # the wrong ester name must not ship


def test_full_diester_unaffected():
    assert name_compound("CCOC(=O)c1ccc(cc1)C(=O)OCC") == \
        "diethyl benzene-1,4-dicarboxylate"


def test_chain_mono_ester_acid_senior_correct():
    assert name_compound("OC(=O)CCCCC(=O)OCC") == "6-ethoxy-6-oxohexanoic acid"
    assert name_compound("CCOC(=O)CC(=O)O") == "3-ethoxy-3-oxopropanoic acid"


def test_simple_esters_unaffected():
    assert name_compound("CCOC(=O)C") == "ethyl acetate"
    assert name_compound("COC(=O)c1ccccc1") == "methyl benzoate"
