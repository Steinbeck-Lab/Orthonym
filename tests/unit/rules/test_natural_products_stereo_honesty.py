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
