"""A carbenium on a fused ring system outside the numbering tables is named through its '-yl' group: '-ylium'.

: a cation formed by loss of a hydride takes the suffix 'ylium' on the parent hydride, as 'yl' names
the group of the same atom ('9H-fluoren-9-ylium'). The localized fused form is the PIN form of
('1-methyl-5H-cyclohepta[b]pyridin-1-ium-5-ylium (PIN)', the Blue Book). The wider tiers ask for the fusion
prefix before their von Baeyer cage namer: (the Blue Book) a fusion name, not von Baeyer, for a
system of two or more rings of five or more members.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from tests.support.rt_assert import name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

ROWS = [
    ("c1ccc2ccc[cH+]c-2cc1", "1H-benzo[7]annulen-1-ylium"),
    ("c1ccc2c[cH+]ccc-2cc1", "2H-benzo[7]annulen-2-ylium"),
    ("C1=CC=C2C=CC=C[CH+]C2=C1", "5H-benzo[7]annulen-5-ylium"),
    ("Cc1ccc2ccc[cH+]c-2cc1", "7-methyl-1H-benzo[7]annulen-1-ylium"),
    ("[CH]1C=CC2=CC=CC=CC2=C1", "2H-benzo[7]annulen-2-yl"),
]


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", ROWS)
def test_fused_ylium_and_yl(smiles, name, tier):
    row = Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert name_is_rt_exact(name, smiles)


def test_table_cations_unchanged():
    namer = Orthonym(style="pin", **_emit_tier_flags("pin"))
    assert namer.name_tiered("c1ccc2c(c1)[cH+]c1ccccc12")["name"] == "9H-fluoren-9-ylium"
    assert namer.name_tiered("c1cc[cH+]ccc1")["name"] == "cyclohepta-2,4,6-trien-1-ylium"


@pytest.mark.parametrize("smiles", ["C1=C[C+]2C=CC=CC2C=C1", "C1=C[C+]2C=c3ccccc3=CC2=CC1"])
def test_a_fusion_atom_cation_is_not_named_through_a_hydro_yl_group(smiles):
    """: the cation at a fusion atom of a mancude ring takes added hydrogen, 'naphthalen-4a(8aH)-ylium',
    where its '-yl' group takes hydro prefixes, '4a,8a-dihydronaphthalen-4a-yl'; the '-ylium' is not built from it."""
    for tier in ("pin", "best-effort"):
        row = Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)
        assert row["tier"] != "pin_verified" and "dihydro" not in (row["name"] or ""), row
