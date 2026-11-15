"""Substituted arsanium, stibanium and bismuthanium cations (``rules/pnictogen_onium.py``).

 (the Blue Book): the cation of a mononuclear parent hydride takes the suffix 'ium' and its groups
are prefixes, '[6-(dimethylsulfaniumyl)hexyl]tri(methyl)phosphanium (PIN)' (:42358). A group with a heteroatom
reached no producer; the name is systematic_verified (built by the all-organic program, never a PIN).
"""
import pytest

from tests.support.rt_assert import name_best_effort

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name", [
    ("C[As+](C)(C)CCO", "(2-hydroxyethyl)tri(methyl)arsanium"),
    ("C[As+](C)(C)CCO.[I-]", "(2-hydroxyethyl)tri(methyl)arsanium iodide"),
    ("CCC(=O)C[As+](c1ccccc1)(c1ccccc1)c1ccccc1", "(2-oxobutyl)tri(phenyl)arsanium"),
    ("CC(=O)NC1=CC(=CC=C1)[As+](C)(C)C", "(3-acetamidophenyl)tri(methyl)arsanium"),
])
def test_substituted_arsanium(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "systematic_verified" and not r["is_pin"], r


def test_a_hydrocarbyl_arsanium_keeps_its_route_and_label():
    r = name_best_effort("C[As+](C)(C)C")
    assert r["name"] == "tetramethylarsanium" and r["tier"] == "pin_verified"
