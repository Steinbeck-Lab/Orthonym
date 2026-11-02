"""The E/Z of a ring '-ylidene' double bond on a chain parent cited inside the prefix is not the PIN.

 (1) (the Blue Book): "(a) the double bond is considered as an integral part of the parent
structure; the stereodescriptor is placed at the front of the substitutive name, preceded by the locant indicating its
point of attachment to the parent structure; (b) [...] The stereodescriptor is cited at the front of the name of the
substituent group [...] Method (a) generates preferred IUPAC names." The engine builds method (b) for a ring ylidene
that has its own stereocentre, so that name ships below the PIN; a ring parent keeps method (a).
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _row(smiles, tier):
    return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.parametrize("smiles,name", [
    ("C/C(C=O)=C1/CCCC[C@@H]1C", "2-[(1E,2S)-2-methylcyclohexylidene]propanal"),
    ("OC(=O)/C=C1/CCC[C@H](C)C1", "[(1Z,3S)-3-methylcyclohexylidene]acetic acid"),
])
def test_method_b_is_below_the_pin(smiles, name):
    assert _row(smiles, "pin")["tier"] == "abstain"
    row = _row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (name, "systematic_verified"), row


def test_a_ring_parent_keeps_method_a():
    row = _row("CC/C(C)=C1\\C=Cc2ccccc21", "pin")
    assert (row["name"], row["tier"]) == ("(1E)-1-(butan-2-ylidene)-1H-indene", "pin_verified"), row
