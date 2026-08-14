"""Unit tests for the acyclic polyol / polyether / polyester T4 producer
(v30 tail #21). Each emitted name is asserted to round-trip to the full
InChIKey via OPSIN (the 0-wrong contract); every out-of-scope shape declines.
"""
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.rules.polyol_polyester import name_acyclic_polyol_polyester
from orthonym.validation.opsin_roundtrip import opsin_parse


def _full_rt(smiles: str, name: str) -> bool:
    osmi = opsin_parse(name)
    if not osmi:
        return False
    a = inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))
    b = inchi.MolToInchiKey(Chem.MolFromSmiles(osmi))
    return a == b


def test_tail21_hexitol_trimethacrylate_names_and_round_trips():
    smi = ("CC(=C)C(=O)OCC(COCC(C(C(C(COCC(COC(=O)C(=C)C)O)OCC("
           "COC(=O)C(=C)C)O)O)O)O)O")
    name = name_acyclic_polyol_polyester(Chem.MolFromSmiles(smi))
    assert name == (
        "1,5,6-tris(2-hydroxy-3-(2-methylprop-2-enoyloxy)propoxy)"
        "hexane-2,3,4-triol")
    assert _full_rt(smi, name)


def test_glyceryl_monoacetate_names_and_round_trips():
    smi = "CC(=O)OCC(O)CO"
    name = name_acyclic_polyol_polyester(Chem.MolFromSmiles(smi))
    assert name == "3-(acetyloxy)propane-1,2-diol"
    assert _full_rt(smi, name)


def test_pure_polyol_declines_composer_owns_it():
    # No ester -> the composer already names this as a PIN; the producer must
    # never intercept it.
    smi = "OCC(COCC(C(C(C(COCC(CO)O)OCC(CO)O)O)O)O)O"
    assert name_acyclic_polyol_polyester(Chem.MolFromSmiles(smi)) is None


def test_plain_alkane_declines():
    assert name_acyclic_polyol_polyester(Chem.MolFromSmiles("CCCCCC")) is None


def test_ring_containing_declines():
    # A cyclic ester is out of scope (ring guard).
    assert name_acyclic_polyol_polyester(
        Chem.MolFromSmiles("O=C1OCC1")) is None


def test_nitrogen_bearing_declines():
    # Any non-C/H/O atom -> decline (fail closed).
    assert name_acyclic_polyol_polyester(
        Chem.MolFromSmiles("CC(=O)OCC(O)CN")) is None
