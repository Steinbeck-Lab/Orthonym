""" -- Hantzsch-Widman heteroatom locant CITATION ORDER.

``build_hw_name`` used to join every heteroatom locant as one globally ascending
run, which discards which locant belongs to which 'a' prefix. For an O-Si-O
ring that spells ``1,2,3-dioxasilolane`` -- asserting an O-O bond that does not
exist (OPSIN: "Atom is in unphysical valency state! Element: O valency: 4").

**** (``the Blue Book Blue Book``), last sentence of the rule:

    "Locants are cited at the front of the name, in the order of citation of the
    skeletal replacement ('a') prefixes."

The rule's own example block settles it beyond a degenerate reading -- the locant
SET is chosen for lowness, but the CITATION is grouped per element in prefix
order:

    1,6,2-dioxazepane (PIN) (:8300) -- dioxa takes 1 and 6, aza takes 2
    1,3,2-dioxaboretane (PIN) (:37178) -- dioxa takes 1 and 3, bora takes 2

Both are non-degenerate: a global ascending sort spells them ``1,2,6-`` and
``1,2,3-`` respectively.
"""

import pytest

from orthonym.rules.heterocycles import build_hw_name


# ---------------------------------------------------------------------------
# The defect: citation order must follow 'a'-prefix order, not a global sort
# ---------------------------------------------------------------------------

class TestLocantsFollowPrefixCitationOrder:

    @pytest.mark.unit
    def test_dioxaboretane_is_a_verbatim_blue_book_pin(self):
        """1,3,2-dioxaboretane (PIN) -- the Blue Book.

        O-B-O in a 4-ring. ``dioxa`` is cited first and owns locants 1 and 3;
        ``bora`` owns 2. A global sort would spell ``1,2,3-``.
        """
        assert build_hw_name([(1, 'O'), (2, 'B'), (3, 'O')], 4, True, False) == \
            "1,3,2-dioxaboretane"

    @pytest.mark.unit
    def test_dioxazepane_is_a_verbatim_blue_book_pin(self):
        """1,6,2-dioxazepane (PIN) -- the Blue Book.

        The Blue Book prints the "not" list as locant SETS ('1,2,6' is lower
        than '1,3,4') while the PIN itself is CITED '1,6,2' -- the clearest
        statement in the section that set-selection and citation are two
        different steps.
        """
        assert build_hw_name([(1, 'O'), (2, 'N'), (6, 'O')], 7, True, False) == \
            "1,6,2-dioxazepane"

    @pytest.mark.unit
    def test_o_si_o_silolane_no_longer_asserts_an_o_o_bond(self):
        """The reported defect: O-Si-O must be 1,3,2- not 1,2,3-."""
        assert build_hw_name([(1, 'O'), (2, 'Si'), (3, 'O')], 5, True, False) == \
            "1,3,2-dioxasilolane"

    @pytest.mark.unit
    def test_end_to_end_tetramethyl_dioxasilolane(self):
        """Whole-molecule check -- the string must reach the emitted name."""
        from orthonym.namer import Orthonym
        namer = Orthonym(
            style="pin",
            general_fallback=True,
            general_fallback_unverified=True,
            allow_aromatic_general=True,
        )
        row = namer.name_tiered("CC1C(O[Si](O1)(C)C)C")
        assert row["name"] == "2,2,4,5-tetramethyl-1,3,2-dioxasilolane"


# ---------------------------------------------------------------------------
# The fix must be a NO-OP wherever the senior element already holds the low slot
# ---------------------------------------------------------------------------

class TestCitationOrderIsANoOpWhenAlreadyGrouped:

    @pytest.mark.unit
    @pytest.mark.parametrize("heteroatoms,ring_size,saturated,aromatic,expected", [
        # Blue Book PIN, example block (:8296) -- oxa owns 1,
        # dithia owns 2 and 6; prefix order and ascending order agree here.
        ([(1, 'O'), (2, 'S'), (6, 'S')], 7, True, False, "1,2,6-oxadithiepane"),
        # Blue Book PIN (:8312) -- one locant per element, order agrees.
        ([(1, 'O'), (2, 'N'), (5, 'P')], 5, False, True, "1,2,5-oxazaphosphole"),
        # Single-element rings: nothing to regroup.
        ([(1, 'O'), (3, 'O')], 5, True, False, "1,3-dioxolane"),
        ([(1, 'O'), (3, 'O'), (5, 'O')], 6, True, False, "1,3,5-trioxane"),
        ([(1, 'S'), (2, 'S')], 5, True, False, "1,2-dithiolane"),
        # Mixed, senior element already lowest.
        ([(1, 'O'), (2, 'S')], 5, True, False, "1,2-oxathiolane"),
        ([(1, 'O'), (3, 'N')], 5, True, False, "1,3-oxazolidine"),
        ([(1, 'O'), (4, 'N')], 6, True, False, "1,4-oxazinane"),
        ([(1, 'O'), (3, 'S'), (5, 'N')], 7, True, False, "1,3,5-oxathiazepane"),
        # Single heteroatom: no locant at all.
        ([(1, 'Si')], 5, True, False, "silolane"),
        ([(1, 'O')], 5, True, False, "oxolane"),
    ])
    def test_unchanged(self, heteroatoms, ring_size, saturated, aromatic, expected):
        assert build_hw_name(heteroatoms, ring_size, saturated, aromatic) == expected
