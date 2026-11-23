"""The E/Z locant of a double bond between a numbered and a lettered ring atom ('9a' beside 9) is the lower one.

``rules.stereochemistry.collect_stereodescriptors`` compared an int with a fusion locant string and raised a TypeError,
so the fused-ring PIN builder declined and the von Baeyer name shipped. numbers the interior atoms with
letters ('4a' after 4); the composite locant key orders them (``_composite_locant_sort_key``).
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact


def test_a_lettered_locant_beside_a_number():
    from orthonym.rules.stereochemistry import _composite_locant_sort_key as key
    assert min(10, "9a", key=lambda loc: key((loc, ""))) == "9a"
    assert min("4a", 2, key=lambda loc: key((loc, ""))) == 2


@pytest.mark.opsin_gate
def test_the_macrocyclic_lactone_takes_its_fusion_pin():
    smiles = "C=C(C)[C@H]1CC[C@@]23O[C@@H]2[C@@H](C/C(C)=C2/C(=O)C=C(C)[C@@]2(O)CC1)OC3=O"
    r = name_best_effort(smiles)
    assert r["tier"] == "pin_verified" and "cyclododeca" in r["name"] and "tetracyclo" not in r["name"], r
    assert name_is_rt_exact(r["name"], smiles)
