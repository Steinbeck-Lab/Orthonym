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


# --------------------------------------------------------------------------
# F-E1: per-token element soundness (sound-by-refusal).
# --------------------------------------------------------------------------
from orthonym.validation.e1_certificate import _token_is_confidently_all_carbon


def test_all_carbon_token_bound_to_heteroatom_is_rejected():
    """The synthetic gap: `ethyl`/`eth` tokens fabricated onto N and O atoms of
    CCN(CC)N=O used to certify as `1-ethylethane`. Now rejected."""
    mol = Chem.MolFromSmiles("CCN(CC)N=O")  # 0C 1C 2N 3C 4C 5N 6O
    v = verify_certificate(mol, _res([
        TokenBinding((2, 3, 4, 5, 6), "ethyl", "parent"),
        TokenBinding((0, 1), "eth", "parent"),
    ], name="1-ethylethane"), allow_charged=True)
    assert not v.ok and "ethyl" in v.reason


def test_real_general_result_with_heteroatom_token_still_certifies():
    """A composite token that legitimately declares heteroatoms (`...3-oxa-1,2-
    diaza...`) is NOT confidently all-carbon, so it is skipped -- the real
    general-engine result for CCN(CC)N=O still certifies."""
    mol = Chem.MolFromSmiles("CCN(CC)N=O")
    v = verify_certificate(mol, _res([
        TokenBinding((2, 3, 4, 5, 6), "1-ethyl-3-oxa-1,2-diazaprop-2-en-1-yl", "parent"),
        TokenBinding((0, 1), "eth", "parent"),
    ], name="1-(1-ethyl-3-oxa-1,2-diazaprop-2-en-1-yl)ethane"), allow_charged=True)
    assert v.ok


def test_heteroatom_suffix_token_on_heteroatom_not_rejected():
    """`ol` on the O of ethanol is a heteroatom-declaring token -> skipped, not
    rejected (the existing complete-partition test also covers this)."""
    v = verify_certificate(ETHANOL, _res([
        TokenBinding((0, 1), "eth", "parent"),
        TokenBinding((2,), "ol", "suffix"),
    ]))
    assert v.ok


@pytest.mark.parametrize("token,expected", [
    ("ethyl", True), ("methyl", True), ("propyl", True), ("ethane", True),
    ("cyclohexyl", True), ("phenyl", True), ("naphthyl", True), ("benzyl", True),
    # heteroatom / functional tokens must be skipped (never confidently all-carbon)
    ("methoxy", False), ("ethanol", False), ("methanamine", False),
    ("oxan-2-yl", False), ("pyridin-3-yl", False), ("furyl", False),
    ("thienyl", False), ("acetyl", False), ("formyl", False), ("chloro", False),
    ("hydroxy", False), ("sulfanyl", False), ("carbamoyl", False), ("ol", False),
])
def test_confidently_all_carbon_classifier(token, expected):
    assert _token_is_confidently_all_carbon(token) is expected
