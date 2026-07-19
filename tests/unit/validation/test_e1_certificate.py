# tests/unit/validation/test_e1_certificate.py
"""v25 G1: E1 certificate — every heavy atom binds exactly once."""
import pytest
from rdkit import Chem

from orthonym.assembly.general_engine import GeneralEngineResult, TokenBinding
from orthonym.validation.e1_certificate import E1Verdict, verify_certificate

pytestmark = pytest.mark.unit

ETHANOL = Chem.MolFromSmiles("CCO")  # atoms: 0 C, 1 C, 2 O


def _res(bindings, name="ethan-1-ol"):
    return GeneralEngineResult(name=name, bindings=tuple(bindings))


def test_complete_partition_ok():
    v = verify_certificate(ETHANOL, _res([
        TokenBinding((0, 1), "eth", "parent"),
        TokenBinding((2,), "ol", "suffix"),
    ]))
    assert v.ok


def test_unbound_atom_fails():
    v = verify_certificate(ETHANOL, _res([
        TokenBinding((0, 1), "eth", "parent"),
    ]))
    assert not v.ok and "unbound" in v.reason


def test_double_bound_atom_fails():
    v = verify_certificate(ETHANOL, _res([
        TokenBinding((0, 1), "eth", "parent"),
        TokenBinding((1, 2), "ol", "suffix"),
    ]))
    assert not v.ok and "twice" in v.reason


def test_token_missing_from_name_fails():
    v = verify_certificate(ETHANOL, _res([
        TokenBinding((0, 1), "propan", "parent"),
        TokenBinding((2,), "ol", "suffix"),
    ]))
    assert not v.ok and "token" in v.reason


def test_charged_mol_fails():
    mol = Chem.MolFromSmiles("CC[O-]")
    v = verify_certificate(mol, _res([
        TokenBinding((0, 1), "eth", "parent"),
        TokenBinding((2,), "olate", "suffix"),
    ], name="ethanolate"))
    assert not v.ok and "charge" in v.reason
