"""Phase 1 Part A: the shared best-effort certification gate.

Proves ``certify_general_result`` is STRICTLY STRONGER than E1 alone -- it
rejects the swap-witness class E1 misses (a name whose bindings partition the
atoms correctly but re-fragment a ring), which was shippable via the two
unwired lanes before this gate existed.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.general_engine import GeneralEngineResult, TokenBinding
from orthonym.validation.coverage_gate import certify_general_result
from orthonym.validation.e1_certificate import verify_certificate

pytestmark = pytest.mark.unit

CYCLOHEXANE = Chem.MolFromSmiles("C1CCCCC1")  # ring: 0-1-2-3-4-5-0
ETHANOL = Chem.MolFromSmiles("CCO")           # 0 C, 1 C, 2 O


def test_clean_result_is_certified():
    res = GeneralEngineResult(name="cyclohexane", bindings=(
        TokenBinding((0, 1, 2, 3, 4, 5), "cyclohexane", "parent"),
    ))
    assert certify_general_result(CYCLOHEXANE, res) is True


def test_clean_ethanol_is_certified():
    res = GeneralEngineResult(name="ethan-1-ol", bindings=(
        TokenBinding((0, 1), "eth", "parent"),
        TokenBinding((2,), "ol", "suffix"),
    ))
    assert certify_general_result(ETHANOL, res) is True


def test_ring_refragment_swap_witness_passes_E1_but_gate_voids_it():
    """The canonical swap-witness: cyclohexane spelled as two disjoint propyl
    halves. Atom coverage is complete and disjoint, so E1 accepts it -- but the
    ring bonds joining the halves are unaccounted, so the binding-spine bond
    totality proof (P2) rejects it. The shared gate must return False."""
    res = GeneralEngineResult(name="dipropyl", bindings=(
        TokenBinding((0, 1, 2), "propyl", "prefix"),
        TokenBinding((3, 4, 5), "propyl", "prefix"),
    ))
    # E1 alone is fooled -- documents exactly why the spine half is needed.
    assert verify_certificate(CYCLOHEXANE, res).ok is True
    # The shared gate is strictly stronger and voids it.
    assert certify_general_result(CYCLOHEXANE, res) is False


def test_e1_failure_short_circuits_to_false():
    # An atom left unbound fails E1; the gate returns False without needing the
    # spine half.
    res = GeneralEngineResult(name="ethane", bindings=(
        TokenBinding((0, 1), "eth", "parent"),
    ))
    assert verify_certificate(ETHANOL, res).ok is False
    assert certify_general_result(ETHANOL, res) is False


def test_none_result_is_not_certified():
    assert certify_general_result(ETHANOL, None) is False
