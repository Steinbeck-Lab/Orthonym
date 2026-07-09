"""P-14.5.4: identical-Roman-letter prefixes cited lowest-locant-first
(Wave-2 P0c Task 7). BB BlueBookV2.md:3517 + worked example line 3531.
"""
from orthonym.assembly.naming_utils import prefix_citation_sort_key


class TestPrefixCitationSortKey:
    def test_identical_letters_lowest_locant_first(self):
        # BB line 3531: pentan-2-yl cited before pentan-3-yl.
        a = prefix_citation_sort_key("(pentan-2-yl)")
        b = prefix_citation_sort_key("(pentan-3-yl)")
        assert a[0] == b[0]          # identical Roman letters
        assert a < b                 # locant 2 < 3 decides citation order

    def test_leading_parent_locants_do_not_decide(self):
        # The prefix's PARENT locant ('4-') is stripped; only the group's
        # own contained locants participate (P-14.5.4: "the group that
        # CONTAINS the lowest locants").
        a = prefix_citation_sort_key("4-(pentan-2-yl)")
        b = prefix_citation_sort_key("1-(pentan-3-yl)")
        assert a < b

    def test_alpha_still_dominates(self):
        # Different letters: P-14.5.2 alphabetical order decides as before.
        assert prefix_citation_sort_key("2-chloroethyl") < \
            prefix_citation_sort_key("1-methylethyl")

    def test_sorting_a_list(self):
        prefixes = ["(pentan-3-yl)", "(pentan-2-yl)", "ethyl"]
        ordered = sorted(prefixes, key=prefix_citation_sort_key)
        assert ordered == ["ethyl", "(pentan-2-yl)", "(pentan-3-yl)"]
