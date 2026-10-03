"""An N-carbon-substituted hydroxylamine is named on the senior amine; the functional-class
hydroxylamine name is not the PIN.

 (the Blue Book): "Substituted hydroxylamines of the type R-NH-OH or RR'N-OH
are named as N-derivatives of the senior amine"; 'N-hydroxymethanamine (PIN) N-methylhydroxylamine'
(:38314). When the amine form cannot be built yet (a base amine with a locant, the alphanumerical
merge of 'N-hydroxy' into it), the producer falls back to 'N-(2-chloroethyl)hydroxylamine', which
shipped pin_verified; it is now labelled below the PIN (best-effort keeps it, the default tier
declines). O-substituted hydroxylamines keep the hydroxylamine parent,
'O-methylhydroxylamine (PIN)',:38350).
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers, name_breadth, name_default
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


def test_functional_class_fallback_is_below_the_pin():
    smi = "ONCCCl"
    d = name_default(smi)
    assert d.get("tier") == "abstain" and d.get("limit_code") == "NO_VERIFIED_PIN", d
    b = name_breadth(smi)
    assert (b.get("name"), b.get("tier")) == ("N-(2-chloroethyl)hydroxylamine", "systematic_verified"), b
    assert name_is_rt_exact(b["name"], smi)


@pytest.mark.parametrize("smiles,pin", [
    ("CCNO", "N-hydroxyethanamine"),
    ("CNO", "N-hydroxymethanamine"),
    ("CN(C)O", "N-hydroxy-N-methylmethanamine"),
    ("CON", "O-methylhydroxylamine"),
])
def test_amine_and_o_substituted_forms_keep_their_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
