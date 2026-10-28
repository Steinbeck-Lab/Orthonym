"""Rows the von Baeyer main-ring search used to abstain on at ``perf_op_budget`` and now names.

The pair searches of the main-ring search (``rules/main_ring_pairs.py``) charged one perf-budget unit
per pair of bridgehead paths, which is quadratic in the thousands of paths of a ring system of 13 or
more rings "Selection of the main ring", ``the Blue Book Blue Book``;
"Selection of the main bridge", ``:9603``; "Selection of the main bridge and secondary
bridges", ``:9655``). Each row below is a PubChem row that abstained with ``perf_op_budget`` and is
named since. What makes a name here a correct structure is the independent read-back
(``assert_full_rt``: a fresh OPSIN call, the full InChIKey of the input). The name and the tier are pins
of what the engine ships.

The tier is ``systematic_verified`` for all four, none is labelled a PIN. Every cage here is above the
PIN ceilings (``vonbaeyer_universal.beyond_pin_caps``: 40 skeletal atoms, 8 rings), and
``name_polycyclic_complete`` records the descriptor of such a cage as not a PIN. The reason is rule
 of "Selection of the main bridge and secondary bridges" (``:9685``: "The
superscript locants for the secondary bridges must be as low as possible when considered as a set in
ascending numerical order, the decision being made at the first point of difference"). The main-ring
search keeps one decomposition per bridgehead pair and the numbering is chosen among the decompositions
it kept, so the rule is not applied to every numbering the cage allows. ``CID20598612`` is the example:
its name carries ``0^1,18`` where a descriptor of the same cage with ``0^1,4`` reads back to the same
structure, which the rule prefers. Before the cap was applied on this path the label was ``pin_verified`` for
``CID20598612`` (and for ``CID170637327``, not pinned here); the names are unchanged.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.opsin_gate]

ROWS = {
    'CID20598612': (
        'C1C2C3CC4C35C26C1C7C68C59C4C1C92C83C7C4C33C22C1C1C22C33C4CC3C2C1',
        'octadecacyclo[14.12.0.0^1,18.0^2,13.0^2,15.0^3,12.0^3,28.0^4,11.0^4,27.0^5,10.0^5,26.0^6,9.0^6,25.0^7,24.0^19,28.0^20,27.0^21,26.0^22,25]octacosane',
        'systematic_verified',
    ),
    'CID11332020': (
        'CC(C)(C)C1=CC2=C3C(=C1)C4=C(C=C5C6=C4C7=C3C8=C9C2=CC(=CC9=C1C=C(C=C2C1=C8C1=C7C3=C6C(=CC(=C3C3=CC(=CC2=C31)C(C)(C)C)C(C)(C)C)C(=O)C5=O)C(C)(C)C)C(C)(C)C)C(C)(C)C',
        '7,12,21,26,30,37-hexa-tert-butyltetradecacyclo[22.18.2.0^2,39.0^3,16.0^4,9.0^5,38.0^10,15.0^14,19.0^17,43.0^18,23.0^28,44.0^29,42.0^32,41.0^35,40]tetratetraconta-1(43),2(39),3,5(38),6,8,10,12,14(19),15,17,20,22,24(44),25,27,29(42),30,32(41),35(40),36-henicosaene-33,34-dione',
        'systematic_verified',
    ),
    'CID102281701': (
        'CC(C)(C)C1=CC2=C(C(=C1)CC3=C(C(=CC(=C3)C(C)(C)C)NC4=CC(=CC(=C4OC)CC5=CC(=CC(=C5OC)CC6=C(C(=CC(=C6)C(C)(C)C)CC7=C(C(=CC(=C7)C(C)(C)C)NC8=CC(=CC(=C8OC)CC9=CC(=CC(=C9OC)CC1=C(C(=CC(=C1)C(C)(C)C)CC1=C(C(=CC(=C1)C(C)(C)C)NC1=CC(=CC(=C1OC)CC1=CC(=CC(=C1OC)C2)C(C)(C)C)C(C)(C)C)OC)OC)C(C)(C)C)C(C)(C)C)OC)OC)C(C)(C)C)C(C)(C)C)OC)OC',
        '5,11,17,23,29,35,41,47,53,59,65,71-dodeca-tert-butyl-73,74,75,76,77,78,79,80,81,82,83,84-dodecamethoxy-2,26,50-triazatridecacyclo[67.3.1.1^3,7.1^9,13.1^15,19.1^21,25.1^27,31.1^33,37.1^39,43.1^45,49.1^51,55.1^57,61.1^63,67]tetraoctaconta-1(72),3,5,7(84),9,11,13(83),15(82),16,18,21(81),22,24,27,29,31(80),33,35,37(79),39(78),40,42,45(77),46,48,51,53,55(76),57,59,61(75),63(74),64,66,69(73),70-hexatriacontaene',
        'systematic_verified',
    ),
    'CID91280663': (
        'CC1(OC2C3CC(C2O1)C4=C5C=C6C7=C(C8CC7C9C8OC(O9)(C)C)C(=CC1=NC(=CC2=C7C8C=CC(C7=C(N2)CC(=C34)N5)C2C8OC(O2)(C)C)C2=C1C1C=CC2C2C1OC(O2)(C)C)N6)C',
        '10,10,22,22,34,34,46,46-octamethyl-9,11,21,23,33,35,45,47-octaoxa-51,52,54,56-tetraazaheptadecacyclo[41.5.2.2^7,13.1^3,41.1^5,15.1^17,27.1^19,25.1^29,39.1^31,37.0^2,42.0^6,14.0^8,12.0^18,26.0^20,24.0^30,38.0^32,36.0^44,48]octapentaconta-2(42),3,5,14,17,26,28,30(38),39,41(51),49,57-dodecaene',
        'systematic_verified',
    ),
}


@pytest.mark.parametrize("cid", list(ROWS))
def test_row_that_ran_out_of_the_perf_budget_keeps_its_name(cid):
    from orthonym.cli import _emit_tier_flags
    from orthonym.namer import Orthonym
    from tests.support.jars import jar_or_skip
    from tests.support.rt_assert import assert_full_rt

    jar_or_skip("opsin")
    smiles, expected, tier = ROWS[cid]
    namer = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    namer.name_tiered("CCO")      # as the run tool does: the JVM is up before the molecule is named
    result = namer.name_tiered(smiles)
    assert result["name"] == expected
    assert result["tier"] == tier
    assert_full_rt(result["name"], smiles, cid)
