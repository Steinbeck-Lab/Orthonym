"""A sulfinyl stereodescriptor fronted on a name with locants is not labelled a PIN.

 (the Blue Book): stereodescriptors "are placed at the front of the complete name when
related to the parent structure... When they relate to substituent groups, they are cited at the front
of the corresponding prefix. They are preceded by a numerical or letter locant... when such locants are
present"; the book's own locanted prefix: '1-methyl-4-[(R)-phenyl(18O1)methanesulfonyl]benzene (PIN)'
(:46030). The unlocanted single prefix is the boundary: '(S)-(methanesulfinyl)benzene (PIN)' (:46266),
where the front of the name is the front of the prefix. The stereo backstop put the bare '(R)-' of a
sulfoxide S in front of names such as '4-(methanesulfinyl)benzoic acid' at pin_verified. The name is
valid (OPSIN 2.9.0 reads it to the full InChIKey) but is not the PIN, so the best-effort tier keeps it
below pin_verified and the PIN tier declines.
"""
import pytest

pytestmark = pytest.mark.opsin_gate


def _row(smiles, tier):
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    namer = Orthonym() if tier == "pin" else Orthonym(style="pin", **_emit_tier_flags(tier))
    return namer.name_tiered(smiles)


@pytest.mark.parametrize("smiles,name", [
    ("C[S@@](=O)c1ccc(C(=O)O)cc1", "(R)-4-(methanesulfinyl)benzoic acid"),
    ("C[S@@](=O)c1ccc(O)cc1", "(R)-4-(methanesulfinyl)phenol"),
    ("CC[S@](=O)c1ccc(C(=O)O)cc1", "(S)-4-(ethanesulfinyl)benzoic acid"),
])
def test_fronted_descriptor_of_a_locanted_prefix_is_not_a_pin(smiles, name):
    from tests.support.rt_assert import name_is_rt_exact
    be = _row(smiles, "best-effort")
    assert be["name"] == name and name_is_rt_exact(name, smiles)
    assert be["tier"] != "pin_verified" and be["is_pin"] is False, be["tier"]
    pin = _row(smiles, "pin")
    assert pin["tier"] != "pin_verified", pin


@pytest.mark.parametrize("smiles,pin", [
    ("C[S@@](=O)c1ccccc1", "(R)-(methanesulfinyl)benzene"),
    ("CC[S@](=O)C", "(R)-(methanesulfinyl)ethane"),
])
def test_the_unlocanted_single_prefix_keeps_its_pin(smiles, pin):
    from tests.support.pin_tiers import assert_pin_at_both_tiers
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,be_name", [
    # the best-effort tier already builds the descriptor inside the prefix (the promotion
    # re-run's producers): the shape of:46030, kept below pin_verified there
    ("C[S@@](=O)c1ccc(CC(=O)O)cc1", "{4-[(R)-methanesulfinyl]phenyl}acetic acid"),
    ("C[S@@](=O)c1ccncc1", "4-[(R)-methanesulfinyl]pyridine"),
])
def test_the_pin_tier_declines_the_fronted_spelling(smiles, be_name):
    from tests.support.rt_assert import name_is_rt_exact
    pin = _row(smiles, "pin")
    assert pin["tier"] != "pin_verified", pin
    be = _row(smiles, "best-effort")
    assert be["name"] == be_name and name_is_rt_exact(be_name, smiles)
