"""A mancude fused system with any number of identical bridges has a bridged fused PIN (Task 2).

 "Citation of bridges" (the Blue Book,:14173): "Two or more identical bridges are
indicated by the numerical prefixes 'di', 'tri', etc.... The locant pairs of identical bridges are
separated by colons."; "Choice of attachment locants" (:14221): (a) the lowest set of
locants for all the bridge attachment points considered as a set (:14225); (:23710) and
 (:23843) put the bridged fused name above the von Baeyer name.

The bridged fused PIN builder (``rules/bridged_fused_pin``) names these at main; this file pins the
class so that a later change to the von Baeyer path, which reaches the same molecules first, cannot
turn a bridged fused PIN into a polyene cage. Independent OPSIN read-back: MATCH for both anthracene
names (``scripts/large_polycycles/independent_rt.py``).
"""
import pytest

from tests.support.large_polycycles import be, ladder, pin
from tests.support.rt_assert import assert_full_rt

pytestmark = pytest.mark.opsin_gate
L = ladder()


@pytest.mark.unit
@pytest.mark.parametrize("label,expected", [
    ("B01_trimethanoanthracene", "1,4:5,8:9,10-trimethanoanthracene"),   #:14173
    ("B02_1458dimethanoanthracene", "1,4:5,8-dimethanoanthracene"),      # (a):14225
])
def test_k_bridge_mancude_pin(label, expected):
    r = pin(L[label])
    assert (r["name"], r["tier"]) == (expected, "pin_verified"), r
    assert_full_rt(expected, L[label])
    assert be(L[label])["name"] == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("C12=CC=C(C3=CC=CC=C13)C2", "1,4-methanonaphthalene"),
    ("C12=CC=C(C=3C4=CC=C(C13)C4)C2", "1,4:5,8-dimethanonaphthalene"),
    ("C12=CC=C(C=3C4=CC=C(C13)CC4)CC2", "1,4:5,8-diethanonaphthalene"),
    ("C12=CC=C(C=3C4=CC=C(C13)O4)O2", "1,4:5,8-diepoxynaphthalene"),
])
def test_existing_bridged_pins_unchanged(smiles, expected):
    assert pin(smiles)["name"] == expected
