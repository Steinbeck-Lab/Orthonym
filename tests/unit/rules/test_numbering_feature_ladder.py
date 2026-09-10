""" numbering-feature ladder in compare_numbering (Wave-2 P0c
Task 11). BB the Blue Book: (b) heteroatoms before (c) indicated
hydrogen before (d) principal group suffix.
"""
from orthonym.rules.locants import compare_numbering


class TestLadderOrderP59110:
    def test_heteroatoms_outrank_pcg(self):
        # (b) before (d): candidate A wins on heteroatoms even though its
        # suffix locant is higher.
        a = {"heteroatoms": [(1, "N")], "pcg": [3]}
        b = {"heteroatoms": [(2, "N")], "pcg": [1]}
        assert compare_numbering(a, b) == -1
        assert compare_numbering(b, a) == 1

    def test_indicated_h_outranks_pcg(self):
        # (c) before (d).
        a = {"indicated_h": [1], "pcg": [4]}
        b = {"indicated_h": [2], "pcg": [1]}
        assert compare_numbering(a, b) == -1

    def test_heteroatoms_outrank_indicated_h(self):
        # (b) before (c).
        a = {"heteroatoms": [(1, "N")], "indicated_h": [9]}
        b = {"heteroatoms": [(3, "N")], "indicated_h": [1]}
        assert compare_numbering(a, b) == -1

    def test_pcg_still_decides_when_earlier_tiers_tie(self):
        a = {"heteroatoms": [(1, "N")], "pcg": [2]}
        b = {"heteroatoms": [(1, "N")], "pcg": [4]}
        assert compare_numbering(a, b) == -1

    def test_single_tier_calls_unchanged(self):
        # Callers passing only pcg / only substituents keep exact semantics.
        assert compare_numbering({"pcg": [1]}, {"pcg": [3]}) == -1
        assert compare_numbering({"substituents": [2, 3]},
                                 {"substituents": [2, 4]}) == -1
        assert compare_numbering({}, {}) == 0
