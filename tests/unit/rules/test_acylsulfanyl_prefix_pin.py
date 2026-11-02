"""The thioester group -S-CO-R is the preferred prefix 'acylsulfanyl' at the PIN tier.

 (the Blue Book): "an ester group is indicated by preferred prefixes [...] such as
'acylsulfanyl' for the group -S-CO-R".: 'CH3-CH2-CO-OO-S-CO-CH3' is
'1-[(acetylsulfanyl)peroxy]propan-1-one (PIN)' (:39499), the parent the longer acyl chain.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from tests.support.rt_assert import name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", [
    ("CCC(=O)OOSC(C)=O", "1-[(acetylsulfanyl)peroxy]propan-1-one"),
    ("CC(=O)OOSC(C)=O", "1-[(acetylperoxy)sulfanyl]ethan-1-one"),
    ("CC(=O)SCC(=O)O", "(acetylsulfanyl)acetic acid"),
])
def test_acylsulfanyl_prefix(smiles, name, tier):
    row = Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert name_is_rt_exact(name, smiles)
