"""A substituent prefix on a retained contracted alkoxy prefix makes a compound prefix.

 (the Blue Book): 'methoxy', 'ethoxy', 'propoxy', 'butoxy' and 'phenoxy' "are
fully substitutable (with the exception of tert-butoxy) and are considered as simple
prefixes". One of them carrying a substituent prefix is a compound prefix,:15762),
and (:7232) "Parentheses are used around compound (see and complex (see
 prefixes": '1-(chloromethoxy)-4-nitrobenzene (PIN)' (:27711), '[2-(carboxymethoxy)
ethoxy]acetic acid' (:23112). Without a locant the digit test did not see it, so the bridged
fused, the partially hydrogenated fused and the cyclohexane prefix paths cited
'methoxymethoxy' and 'chloromethoxy' bare. The substituent prefix in front of the stem may
itself be enclosed: the prefix producers pass '(methylsulfanyl)methoxy', which takes the next
mark around it ('[(4-methylphenyl)methoxy]benzene',,:21618). OPSIN reads both
spellings, so the spelling rests on this test; every name below was read back by OPSIN 2.9.0
to the input's full InChIKey (S3 planning notes, fix pass; the nested row in the S3 fix round
1 ledger)."""
import pytest

from orthonym import Orthonym
from orthonym.assembly.naming_utils import is_complex_substituent, needs_brackets
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots

COMPOUND = ["methoxymethoxy", "ethoxyethoxy", "chloromethoxy", "hydroxyethoxy",
            "carboxymethoxy", "phenylmethoxy", "chlorophenoxy", "methylsulfanylmethoxy",
            "(methylsulfanyl)methoxy", "(trifluoromethyl)methoxy"]
SIMPLE = ["methoxy", "ethoxy", "propoxy", "butoxy", "phenoxy", "tert-butoxy", "isopropoxy",
          "dimethoxy"]
NAMES = [
    ("COCOC1CC2CCC1c1ccccc12", "2-(methoxymethoxy)-1,2,3,4-tetrahydro-1,4-ethanonaphthalene"),
    ("COCOC1CCc2ccccc2C1", "2-(methoxymethoxy)-1,2,3,4-tetrahydronaphthalene"),
    ("COCOC1CCCCC1", "(methoxymethoxy)cyclohexane"),
    ("ClCOC1CCCCC1", "(chloromethoxy)cyclohexane"),
    ("COCOC1CCCCC1C", "1-(methoxymethoxy)-2-methylcyclohexane"),
    ("CSCOC1CCCCC1", "[(methylsulfanyl)methoxy]cyclohexane"),
]


@pytest.mark.parametrize("name", COMPOUND)
def test_a_substituted_contracted_alkoxy_is_a_compound_prefix(name):
    assert (needs_brackets(name), is_complex_substituent(name)) == (True, True), name


@pytest.mark.parametrize("name", SIMPLE)
def test_the_contracted_alkoxy_prefixes_stay_simple(name):
    assert (needs_brackets(name), is_complex_substituent(name)) == (False, False), name


def _row(smiles, tier):
    with jvm_slots(1, purpose="alkoxy-enclosure"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", NAMES)
def test_the_compound_prefix_is_enclosed(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])
