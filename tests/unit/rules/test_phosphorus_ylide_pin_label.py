"""A phosphorus ylide drawn R3P=CR2 has no PIN of the double-bond kind; the zwitterion is the PIN.

 'Ylides' (the Blue Book): the ylides "may be named in different ways...: (1) as
zwitterionic compounds; (2) by applying the λ-convention when X = P, As, Sb, Bi, S, Se or Te; (3) by
functional class nomenclature... Method (1) is applicable to all 'ylides' and leads to preferred IUPAC
names." The phosphorus example (:42519): '(1) 2-(trimethylphosphaniumyl)propan-2-ide (PIN)'; the
λ-convention name '(2) trimethyl(propan-2-ylidene)-λ5-phosphane' carries no PIN label.

The strict path named the uncharged drawing '(triethylphosphanylidene)ethane' (a prefix for a P with five
bonds without its λ5) and labelled it pin_verified. The names built from the double-bond drawing are
recorded as PIN-form names whose preferred status is not certified: the default tier declines them, the
best-effort tier keeps them (pin_unverified). The charge-separated drawing keeps its zwitterion PIN.
"""
import pytest
from rdkit import Chem

from orthonym.perception.ylides import is_uncharged_group15_ylide
from tests.support.pin_tiers import name_breadth, name_default
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles, expected", [
    ("CC=P(CC)(CC)CC", True), ("CC(C)=P(C)(C)C", True), ("C=P(c1ccccc1)(c1ccccc1)c1ccccc1", True),
    ("C=[As](C)(C)C", True),
    ("[CH2-][P+](C)(C)C", False),            # the zwitterion drawing
    ("C=P(C)C", False),                      # four neighbours are the λ5 centre; three are not
    ("CP(C)C", False), ("O=P(C)(C)C", False),
    ("C1=CC=CC=P1", False),                  # a ring bond P=C: the ring of a phosphinine
    ("C=S(C)C", False),                      # the sulfur ylide has its own PIN
])
def test_the_ylide_predicate(smiles, expected):
    assert is_uncharged_group15_ylide(Chem.MolFromSmiles(smiles)) is expected


@pytest.mark.parametrize("smiles, name", [
    ("CC=P(CC)(CC)CC", "triethyl(ethylidene)-λ5-phosphane"),
    ("CCP(=CC)(CC)CC", "triethyl(ethylidene)-λ5-phosphane"),      # the spelling of the milestone1500 row
    ("CC(C)=P(C)(C)C", "trimethyl(propan-2-ylidene)-λ5-phosphane"),
])
def test_an_uncharged_phosphorus_ylide_has_no_pin_at_the_default_tier(smiles, name):
    d = name_default(smiles)
    assert d.get("tier") == "abstain" and d.get("limit_code") == "NO_VERIFIED_PIN", d
    b = name_breadth(smiles)
    assert (b.get("name"), b.get("tier"), b.get("is_pin")) == (name, "pin_unverified", False), b
    assert_full_rt(b["name"], smiles)


def test_the_zwitterion_drawing_keeps_its_pin():
    smiles = "[CH2-][P+](C)(C)C"
    for res in (name_default(smiles), name_breadth(smiles)):
        assert (res.get("name"), res.get("tier"), res.get("is_pin")) == (
            "(trimethylphosphaniumyl)methanide", "pin_verified", True), res
    assert_full_rt("(trimethylphosphaniumyl)methanide", smiles)
