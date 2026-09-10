""": identical-Roman-letter prefixes cited lowest-locant-first
(Wave-2 P0c Task 7). BB the Blue Book + worked example line 3531.
"""
from orthonym.assembly.naming_utils import prefix_citation_sort_key


class TestPrefixCitationSortKey:
    def test_identical_letters_lowest_locant_first(self):
        # the Blue Book: pentan-2-yl cited before pentan-3-yl.
        a = prefix_citation_sort_key("(pentan-2-yl)")
        b = prefix_citation_sort_key("(pentan-3-yl)")
        assert a[0] == b[0]          # identical Roman letters
        assert a < b                 # locant 2 < 3 decides citation order

    def test_leading_parent_locants_do_not_decide(self):
        # The prefix's PARENT locant ('4-') is stripped; only the group's
        # own contained locants participate: "the group that
        # CONTAINS the lowest locants").
        #
        # -FIX Item 2: the convention is now DECLARED at the call rather
        # than guessed from the string. These two inputs are already-RENDERED
        # prefix strings, so their leading locants are parent locants -> the
        # caller says `parent_locants=True`. A bare substituent NAME leads with
        # its own locant, which must see; one silent strip served both
        # and that is what made the key non-injective (see
        # test_the_two_conventions_disagree_and_must_both_be_expressible).
        a = prefix_citation_sort_key("4-(pentan-2-yl)", parent_locants=True)
        b = prefix_citation_sort_key("1-(pentan-3-yl)", parent_locants=True)
        assert a < b

    def test_the_two_conventions_disagree_and_must_both_be_expressible(self):
        """The reason the flag exists: on the SAME string the two readings give
        opposite answers, so no single default can be right for both callers."""
        rendered_a = prefix_citation_sort_key("4-(pentan-2-yl)",
                                             parent_locants=True)
        rendered_b = prefix_citation_sort_key("1-(pentan-3-yl)",
                                             parent_locants=True)
        bare_a = prefix_citation_sort_key("4-(pentan-2-yl)")
        bare_b = prefix_citation_sort_key("1-(pentan-3-yl)")
        assert rendered_a < rendered_b        # by the GROUP's own locants: 2 < 3
        assert bare_b < bare_a                # by the leading locant: 1 < 4

    def test_the_key_is_total_so_a_tie_cannot_reach_atom_order(self):
        """Tiers 1-2 can tie; the full-string tier must still separate them, or a
        stable ``sorted`` resolves the tie by RDKit neighbour order."""
        a = prefix_citation_sort_key("cyclohexylmethyl")
        b = prefix_citation_sort_key("(cyclohexyl)methyl")
        assert a[:2] == b[:2]                 # tiers 1-2 tie
        assert a != b                         #...tier 3 does not

    def test_alpha_still_dominates(self):
        # Different letters: alphabetical order decides as before.
        assert prefix_citation_sort_key("2-chloroethyl") < \
            prefix_citation_sort_key("1-methylethyl")

    def test_sorting_a_list(self):
        prefixes = ["(pentan-3-yl)", "(pentan-2-yl)", "ethyl"]
        ordered = sorted(prefixes, key=prefix_citation_sort_key)
        assert ordered == ["ethyl", "(pentan-2-yl)", "(pentan-3-yl)"]
