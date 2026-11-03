"""Linear polyanhydrides with three or more links: the method (1) dianhydride.

 (the Blue Book) "Method (1) generates preferred IUPAC names": the preferred dicarboxylic acid and its
two adjoining acids, one of which carries the rest of the chain as an acyloxy prefix, cited in their order of
occurrence beginning with the end acid lower in alphabetical order. The acid of such a word alone is named with the
anhydride as 'acyloxy' + 'oxo' beside the senior acid,:32382, 'acid is senior to anhydride')."""
import pytest

from orthonym.cli import _tiered_namer
from tests.support.rt_assert import name_best_effort

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    #:32402 (the book cites the prefixes of the third word out of alphanumerical order; orders them)
    ("CC(=O)OC(=O)CCC(=O)OC(=O)CCC(=O)OC(=O)CC", "acetic butanedioic 4-oxo-4-(propanoyloxy)butanoic dianhydride"),
    # the longer chain is the preferred dicarboxylic acid; the end acid lower in alphabetical order is cited first
    ("CC(=O)OC(=O)CC(=O)OC(=O)CCC(=O)OC(=O)C", "acetic butanedioic 3-(acetyloxy)-3-oxopropanoic dianhydride"),
    # four links: both adjoining acids carry a link
    ("CC(=O)OC(=O)CCC(=O)OC(=O)CCC(=O)OC(=O)CCC(=O)OC(=O)C",
     "bis[4-(acetyloxy)-4-oxobutanoic] butanedioic dianhydride"),
    # the acid words alone
    ("CCC(=O)OC(=O)CCC(=O)O", "4-oxo-4-(propanoyloxy)butanoic acid"),
    ("CC(=O)OC(=O)CCC(=O)O", "4-(acetyloxy)-4-oxobutanoic acid"),
])
def test_method_one_pins(smiles, name):
    row = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


def test_a_one_link_name_of_a_polyanhydride_is_no_pin():
    """A substituted, unsymmetrical centre needs locants on the acid words (:32413), which are not built: the PIN tier
    declines, and the wider tiers keep the one-link name below the PIN."""
    smiles = "CC(=O)OC(=O)CCC(=O)OC(=O)CC(C)C(=O)OC(=O)CC"
    assert _tiered_namer("pin", "pin").name_tiered(smiles)["tier"] == "abstain"
    row = name_best_effort(smiles)
    assert row["name"] and not row["is_pin"], row
