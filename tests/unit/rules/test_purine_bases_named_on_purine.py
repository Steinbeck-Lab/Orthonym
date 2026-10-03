"""Purine bases get the purine PIN.

'adenine', 'guanine', 'hypoxanthine' and 'xanthine' do not occur in the Blue Book (0 hits,
case-insensitive, as for 'uracil', 'thymine' and 'cytosine', the deny rows); purine is a
 retained name with special numbering, "the PIN is 7H-purine" (the Blue Book).
The bases are denied in data/iupac_2013_pin_list.json (--trivial keeps them through
GENERAL_RETAINED_NAMES), the six fused-catalogue entries that carried the trivial names (and
scrambled maps) are gone, the 7H tautomer entry is '7H-purine', and the purine producers
(rules/purine.py, rules/purine_oxo.py) name the bare bases. A name built on the catalogue
'xanthine' core (functional substituents the dione producer does not take) is recorded as not
the PIN. Every name keeps the input's tautomer (OPSIN 2.9.0 fixed-H InChI exact): the N3-H
tautomer is '...-3,7-dihydro-6H-purin-6-one', not '1,7-dihydro'.
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers, name_breadth, name_default
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

PURINE_PIN_ROWS = [
    ("c1ncc2[nH]cnc2n1", "7H-purine"),
    ("Nc1ncnc2nc[nH]c12", "7H-purin-6-amine"),
    ("Nc1ncnc2[nH]cnc12", "9H-purin-6-amine"),
    ("Nc1nc2[nH]cnc2c(=O)[nH]1", "2-amino-1,9-dihydro-6H-purin-6-one"),
    ("Nc1nc(=O)c2[nH]cnc2[nH]1", "2-amino-3,7-dihydro-6H-purin-6-one"),
    ("O=c1[nH]cnc2nc[nH]c12", "1,7-dihydro-6H-purin-6-one"),
    ("O=c1[nH]cnc2[nH]cnc12", "1,9-dihydro-6H-purin-6-one"),
    ("O=c1[nH]c(=O)c2[nH]cnc2[nH]1", "3,7-dihydro-1H-purine-2,6-dione"),
    ("O=c1[nH]c(=O)c2nc[nH]c2[nH]1", "3,9-dihydro-1H-purine-2,6-dione"),
]


@pytest.mark.parametrize("smiles,pin", PURINE_PIN_ROWS)
def test_purine_base_is_named_on_purine(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


def test_trivial_base_names_stay_general_only():
    from rdkit import Chem
    from orthonym.data import ALL_RETAINED_NAMES, GENERAL_RETAINED_NAMES, _PIN_DENY
    for name, smi in [("adenine", "Nc1ncnc2[nH]cnc12"), ("guanine", "Nc1nc2[nH]cnc2c(=O)[nH]1"),
                      ("hypoxanthine", "O=c1[nH]cnc2[nH]cnc12"),
                      ("xanthine", "O=c1[nH]c(=O)c2[nH]cnc2[nH]1")]:
        assert name in _PIN_DENY, name
        assert Chem.CanonSmiles(smi) not in ALL_RETAINED_NAMES, name
        assert GENERAL_RETAINED_NAMES.get(Chem.CanonSmiles(smi)) == name


def test_a_xanthine_core_name_is_not_labelled_a_pin():
    smi = "OCN1C(N(C=2N=CN(C2C1=O)C)C)=O"
    for res in (name_default(smi), name_breadth(smi)):
        assert res.get("tier") != "pin_verified", res
    b = name_breadth(smi)
    assert b.get("name") and name_is_rt_exact(b["name"], smi), b
