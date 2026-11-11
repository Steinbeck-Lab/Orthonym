"""The tie-break of ``perception/chains.py`` computes its key once per symmetry class of the tied chains and only
for the classes with the lowest prefix locants (still-unnamed PubChem plan of 2026-11-11, Topic 1). /
 (the Blue Book): among chains that tie on every criterion, the prefixes decide; chains that an
automorphism of the molecule maps onto each other give one name, so naming the prefixes of each is repeated work.
The chosen chain and the name are the ones the plain ``min`` over every chain gives (``ORTHONYM_P2_VERIFY=1`` checks
that inside the selector)."""
import pytest

import orthonym.assembly.substituent_enumerator as enumerator
from orthonym import Orthonym

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles, name, most", [
    ("CCOCC(COCC)(COCC)COCC", "1,3-diethoxy-2,2-bis(ethoxymethyl)propane", 30),
    ("CC(C)CC(CC(C)C)(CC(C)C)CO", "4-methyl-2,2-bis(2-methylpropyl)pentan-1-ol", 20),
    ("CCCC(CCC)(CCC)C(=O)O", "2,2-dipropylpentanoic acid", 4),
    ("OC(CCl)(CCl)CCBr", "4-bromo-1-chloro-2-(chloromethyl)butan-2-ol", 4),
])
def test_same_name_fewer_prefix_namings(monkeypatch, smiles, name, most):
    calls = [0]
    original = enumerator.name_substituent

    def counted(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(enumerator, "name_substituent", counted)
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")
    assert calls[0] <= most


@pytest.mark.parametrize("smiles, name", [
    ("CCOCC(COCC)(COCC)COCC", "1,3-diethoxy-2,2-bis(ethoxymethyl)propane"),
    ("OC(CCl)(CCl)CCBr", "4-bromo-1-chloro-2-(chloromethyl)butan-2-ol"),
    ("OCC(CCBr)CCCl", "2-(2-bromoethyl)-4-chlorobutan-1-ol"),
])
def test_verify_switch_gives_the_same_name(monkeypatch, smiles, name):
    """With ``ORTHONYM_P2_VERIFY=1`` the plain ``min`` runs beside the staged one and asserts the same chain."""
    monkeypatch.setenv("ORTHONYM_P2_VERIFY", "1")
    assert Orthonym().name_tiered(smiles)["name"] == name
