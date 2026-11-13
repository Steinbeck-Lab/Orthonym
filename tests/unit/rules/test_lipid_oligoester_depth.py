"""The acyl groups of a linear oligoester of a hydroxy acid are named down to the innermost acid it needs.

 "Esters cited as prefixes" (the Blue Book): an ester group is indicated by an 'acyloxy' prefix when
another group has priority, e.g. "3-(benzoyloxy)propanoic acid (PIN)". The 3-hydroxybutanoate pentamer of milestone1500
names its acid fragments three levels deep (``rules.lipids._MAX_FRAGMENT_DEPTH``); with two the inner acid fell to the
general engine, which records its name as a non-PIN fragment, and the same name was labelled systematic_verified.
"""
import pytest

from tests.support.pin_tiers import tier_row

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

PENTAMER = "C[C@H](CC(=O)[O-])OC(=O)C[C@@H](C)OC(=O)C[C@@H](C)OC(=O)C[C@@H](C)OC(=O)C[C@@H](C)O"
NAME = ("(3R)-3-{[(3R)-3-{[(3R)-3-{[(3R)-3-{[(3R)-3-hydroxybutanoyl]oxy}butanoyl]oxy}butanoyl]oxy}butanoyl]oxy}"
        "butanoate")


def test_the_hydroxybutanoate_pentamer_keeps_its_pin_label():
    row = tier_row(PENTAMER, "best-effort")
    assert row["name"] == NAME
    assert row["tier"] in ("pin_verified", "pin_unverified"), row["tier"]
