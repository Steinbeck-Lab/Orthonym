from rdkit import Chem
from orthonym.rules.natural_products import name_natural_product


def _m(smi):
    return Chem.MolFromSmiles(smi)


def test_flat_steroid_declines_retained_name():
    # flat cholesterol: undefined ring stereocentres -> must NOT emit 'cholest-...'
    out = name_natural_product(_m("CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C"))
    assert out is None or "cholest" not in out


def test_flat_androstanediol_declines():
    out = name_natural_product(_m("CC12CCC3C(CCC4CC(O)CCC43C)C1CCC2O"))
    assert out is None or "androstan" not in out


def test_defined_steroid_keeps_retained_name():
    # natural-config androstenedione still names as the steroid parent
    smi = "C[C@@]12C(CC[C@H]1[C@@H]1CCC3=CC(CC[C@]3(C)[C@H]1CC2)=O)=O"
    out = name_natural_product(_m(smi))
    assert out == "androst-4-ene-3,17-dione"


def test_partially_defined_steroid_keeps_honest_name():
    """-02 regression (fix a performance pass, 2026-08-16 coordinator ruling): a steroid with the
    ring stereocentres DEFINED but one substituent centre (C-20, atom idx 1) left
    UNDEFINED must NOT decline. a performance pass's guard declined whenever ANY matched
    stereocentre was undefined, which fabricated a fabrication-guard false positive on
    this partially-defined case -- the correct, measured criterion (all 25 fabrication
    witnesses have ZERO defined centres) is "zero defined AND >=1 undefined", never "any
    undefined". OPSIN(5α-pregnane-3β,20-diol) round-trips to the same molecule
    (same full InChIKey), confirming the per-locant alpha/beta name that omits the
    undefined C-20 is honest, not a fabrication.
    """
    smi = "CC([C@H]1CC[C@H]2[C@@H]3CC[C@H]4C[C@H](CC[C@]4(C)[C@H]3CC[C@]12C)O)O"
    out = name_natural_product(_m(smi))
    assert out == "5α-pregnane-3β,20-diol"
