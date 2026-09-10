"""a phase B5 general lever: the RT-mismatch -> cascade rescue is
best-effort-ONLY, so it can never fire on the PIN/complete/valid tiers (0-wrong
critical -- it must not perturb the default path).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("flags", [
    dict(),                                              # PIN default
    dict(general_fallback=True),                         # valid tier
    dict(general_fallback=True, allow_aromatic_general=True),  # complete tier
])
def test_rescue_is_a_noop_off_best_effort(flags):
    """Without ``general_fallback_unverified`` (best-effort), the rescue returns
    None before any work -- so it cannot change a PIN/complete/valid result."""
    nm = Orthonym(**flags)
    mol = Chem.MolFromSmiles("CCO")
    # feats=None is safe: the best-effort guard returns before touching it.
    assert nm._try_t4_rescue(mol, None, "CCO") is None


def test_rescue_gated_on_live_jar_or_test_mode():
    """Best-effort but no jar and not in gate-bypass -> None (mirrors the T4
    handoff's jar guard; a jarless run stays byte-identical)."""
    nm = Orthonym(general_fallback=True, general_fallback_unverified=True,
                   allow_aromatic_general=True)
    # Force the jarless / non-bypass branch deterministically.
    nm._disable_opsin_validity_gate = False
    import orthonym.namer as N
    orig = N._validity_gate_jar_present
    N._validity_gate_jar_present = lambda: False
    try:
        mol = Chem.MolFromSmiles("CCO")
        assert nm._try_t4_rescue(mol, None, "CCO") is None
    finally:
        N._validity_gate_jar_present = orig
