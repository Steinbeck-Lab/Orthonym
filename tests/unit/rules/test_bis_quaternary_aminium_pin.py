"""Two quaternary N+ on the ends of one carbon chain take the substitutive
'-bis(aminium)' PIN, which the round-trip parser reads, so it is emitted and verified.

 names a carbon-substituted N+ by the cationic suffix 'aminium'
('*N*,*N*,*N*-trimethylmethanaminium (PIN)', the Blue Book, not
'tetramethylazanium':41354). "Polycations with cationic centers on
characteristic groups are named by substitutive nomenclature or multiplicative
nomenclature" (:42138): '*N*^1,*N*^1,*N*^3,*N*^3,*N*^3-hexamethylpropanebis(amidium)
(PIN)' (:42154), 'butanebis(nitrilium) (PIN)' beside the non-PIN multiplicative
'butanediylidynebis(azanium)' (:42160-42162); '3-(azaniumylmethyl)pentane-1,5-
bis(aminium) (PIN)' (:42366). N-locants carry the chain locant:26348).
The multiplicative 'hexane-1,6-diylbis(trimethylazanium)' stays as the verified
general-tier fallback (emit_bis_quaternary_ammonium).
"""
import pytest

from orthonym import Orthonym
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate


def _row(smiles, namer=None):
    with jvm_slots(1, purpose="test-bis-aminium"):
        return (namer or Orthonym(style="pin")).name_tiered(smiles)


@pytest.mark.parametrize("smiles,expected", [
    ("C[N+](C)(C)CCCCCC[N+](C)(C)C",
     "N1,N1,N1,N6,N6,N6-hexamethylhexane-1,6-bis(aminium)"),
    ("C[N+](C)(C)CCCCC[N+](C)(C)C",
     "N1,N1,N1,N5,N5,N5-hexamethylpentane-1,5-bis(aminium)"),
    ("C[N+](C)(C)CC[N+](C)(C)C",
     "N1,N1,N1,N2,N2,N2-hexamethylethane-1,2-bis(aminium)"),
    ("CC[N+](C)(C)CCCCCC[N+](C)(C)CC",
     "N1,N6-diethyl-N1,N1,N6,N6-tetramethylhexane-1,6-bis(aminium)"),
    ("CC(C)[N+](C)(C)CCC[N+](C)(C)C(C)C",
     "N1,N1,N3,N3-tetramethyl-N1,N3-di(propan-2-yl)propane-1,3-bis(aminium)"),
    # a speculative sort key names '(2-hydroxyethyl)dimethylazaniumyl' while
    # choosing the chain; it must not demote the substitutive PIN that ships
    ("OCC[N+](C)(C)CCCCCC[N+](C)(C)CCO",
     "N1,N6-bis(2-hydroxyethyl)-N1,N1,N6,N6-tetramethylhexane-1,6-bis(aminium)"),
])
def test_bis_quaternary_aminium_pin(smiles, expected):
    row = _row(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified" and row["is_pin"]
    assert name_is_rt_exact(expected, smiles)


def test_asymmetric_bis_quaternary_still_fails_closed():
    # different N-substituents on the two ends: the numbering direction needs the
    # prefix tie-break, not built -- fail closed, never a wrong name
    assert _row("C[N+](C)(C)CCCCCC[N+](C)(CC)CC")["tier"] == "abstain"


def test_warm_instance_keeps_the_pin_label():
    # a demoted betaine named first in the same instance must not leak its label
    namer = Orthonym(style="pin")
    for smiles, is_pin in (("C[N+](C)(C)CC(=O)[O-]", False),
                           ("C[N+](C)(C)CCCCCC[N+](C)(C)C", True),
                           ("OCC[N+](C)(C)CCCCCC[N+](C)(C)CCO", True),
                           ("C[N+](C)(C)CC(=O)[O-]", False),
                           ("C[N+](C)(C)CCCCCC[N+](C)(C)C", True)):
        row = _row(smiles, namer)
        assert row["is_pin"] is is_pin, (smiles, row["name"], row["tier"])
        assert (row["tier"] == "pin_verified") is is_pin, (smiles, row["tier"])
