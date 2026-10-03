"""Spelling of the bridged fused parent hydride (slice S2): the hyphen before a parent name
that begins with a locant, and hydro prefixes past 'hexadecahydro'."""
from orthonym.rules.bridged_fused_pin import BridgedParent
from orthonym.rules.bridged_fused_pin.hydro import HydroState, hydro_text


def test_a_hyphen_separates_the_bridge_prefix_from_a_locant():
    # (a) (the Blue Book) "to separate locants from words or word
    # fragments"::14648 '2H,7H-4a,7-ethano-1-benzopyran (PIN)',:24653
    # '1H,3H-3a,7a-methano-2-benzofuran (PIN)'. OPSIN 2.9.0 also reads the name without the
    # hyphen to the same structure, so only this test holds the spelling.
    p = BridgedParent("", "2H,7H", "4a,7-ethano", "1-benzopyran", {}, True)
    assert p.text() == "2H,7H-4a,7-ethano-1-benzopyran"
    q = BridgedParent("", "", "4,7-methano", "azulene", {}, True)
    assert q.text() == "4,7-methanoazulene"


def test_total_hydrogenation_words_past_sixteen():
    # (:16880) hydro prefixes in pairs; (:17026) total hydrogenation
    # without locants: 'octadecahydro' (BB:17040 'octadecahydro-7,14-methano-...')
    state = HydroState(frozenset(), frozenset(range(18)), 0, 9, (frozenset(),))
    assert hydro_text(state, frozenset(range(18)), {}) == "octadecahydro"
    state20 = HydroState(frozenset(), frozenset(range(20)), 0, 10, (frozenset(),))
    assert hydro_text(state20, frozenset(range(20)), {}) == "icosahydro"
