"""A located ring-yl descriptor OPSIN cannot read in a multiplied prefix falls back to the bare '(R)' on the book retry.

 (the Blue Book): the descriptor of a substituent group is cited at the front of its prefix with a locant,
'[(3R)-pyrrolidin-3-yl]'. OPSIN 2.9.0 reads that spelling in a prefix cited once or twice and rejects it when the
prefix is multiplied twice ('N4,N6-bis(2,4-bis{[(3R)-pyrrolidin-3-yl]oxy}phenyl)...'), so the name had no round trip and
the molecule abstained. The located spelling is a book form (assembly.book_prefixes): the row no round trip confirmed is
named again with the mechanical spellings, the bare '(R)-' of ``_add_substituent_stereo_text`` (below the PIN) among them.
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

R068 = ("C1CNC[C@@H]1OC2=CC(=C(C=C2C(F)(F)F)NC(=O)C3=CC(=NC=N3)C(=O)NC4=C(C=C(C(=C4)C(F)(F)F)"
        "O[C@@H]5CCNC5)O[C@@H]6CCNC6)O[C@@H]7CCNC7")


@pytest.mark.opsin_gate
def test_four_located_descriptors_in_multiplied_prefixes_are_named_by_the_retry():
    res = name_best_effort(R068)
    assert res["name"] and "[(R)-pyrrolidin-3-yl]" in res["name"] and res["tier"] != "pin_verified", res
    assert name_is_rt_exact(res["name"], R068)


@pytest.mark.opsin_gate
def test_a_row_that_round_trips_keeps_the_located_descriptor():
    res = name_best_effort("OC[C@H]1CCCO1")
    assert res["name"] == "[(2R)-oxolan-2-yl]methanol", res
