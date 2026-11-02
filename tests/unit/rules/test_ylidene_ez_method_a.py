"""The E/Z of a ring '-ylidene' double bond on a chain parent is cited at the front of the name (method (a)).

 (1) (the Blue Book): "(a) the double bond is considered as an integral part of the parent
structure; the stereodescriptor is placed at the front of the substitutive name, preceded by the locant indicating its
point of attachment to the parent structure; (b) [...] The stereodescriptor is cited at the front of the name of the
substituent group [...] Method (a) generates preferred IUPAC names." The ring prefix leaves the bond to its chain host,
which cites it with the chain locant; the locant-free C2 parents cite it bare, as the alpha carbon's '(R)-' of
'(R)-{...}acetic acid (PIN)' (:45731). A ring parent keeps method (a); an ylidene group inside a prefix keeps its own
descriptor,:44639).
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from tests.support.rt_assert import name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _row(smiles, tier):
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", [
    ("C/C(C=O)=C1/CCCC[C@@H]1C", "(2E)-2-[(2S)-2-methylcyclohexylidene]propanal"),
    ("C[C@H]1CCCC/C1=C(\\C)C=O", "(2Z)-2-[(2S)-2-methylcyclohexylidene]propanal"),
    ("OC(=O)C/C=C1/CCCC[C@@H]1C", "(3Z)-3-[(2S)-2-methylcyclohexylidene]propanoic acid"),
    ("OC(=O)/C=C1/CCC[C@H](C)C1", "(Z)-[(3S)-3-methylcyclohexylidene]acetic acid"),
    ("O=C/C=C1/CCC[C@H](C)C1", "(Z)-[(3S)-3-methylcyclohexylidene]acetaldehyde"),
    ("N#C/C=C1/CCC[C@H](C)C1", "(Z)-[(3S)-3-methylcyclohexylidene]acetonitrile"),
    ("C/C=C1\\CCC[C@@H](C)C1CC(=O)O", "[(2E,6R)-2-ethylidene-6-methylcyclohexyl]acetic acid"),
])
def test_method_a_is_the_pin(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert name_is_rt_exact(name, smiles)


def test_a_ring_parent_keeps_method_a():
    row = _row("CC/C(C)=C1\\C=Cc2ccccc21", "pin")
    assert (row["name"], row["tier"]) == ("(1E)-1-(butan-2-ylidene)-1H-indene", "pin_verified"), row
