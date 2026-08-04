"""P-22.2.2.1.6: a six-membered HW stem follows the LEAST SENIOR heteroatom.

Verbatim, `BlueBookV2/BlueBookV2.md:8411`, section heading "P-22.2.2.1.6
Selecting Hantzsch-Widman names for six-membered rings":

    "The stem for six-membered rings depends on the least senior heteroatom in
    the ring, i.e., the heteroatom whose name directly precedes the stem.
    Heteroatoms are divided into three groups, A, B, and C, each corresponding
    to a stem for the unsaturated and for the saturated compound (Table 2.5).
    The stem is selected in accordance with the group to which the least senior
    heteroatom belongs."

Table 2.5 (`:8259-8261`) -- note Bi is group 6A, not 6C:

    6A (O, S, Se, Te, Bi)                          ine    / ane
    6B (N, Si, Ge, Sn, Pb)                         ine    / inane
    6C (F, Cl, Br, I, P, As, Sb, B, Al, Ga, In, Tl) inine / inane

Seniority for citation, P-22.2.2.1.3 (`:8284`): "their order of citation follows
the sequence: F, Cl, Br, I, O, S, Se, Te, N, P, As, Sb, Bi, Si, Ge, Sn, Pb, B,
Al, Ga, In, Tl." The LEAST senior heteroatom is the one occurring LAST in it.

What was wrong: `hw_stems.get_hw_stem` chose the saturated 6-ring stem from the
single `heteroatom` argument -- the MOST senior heteroatom -- and its caller
(`rules/heterocycles.py:811-813`) passed `'N'` whenever nitrogen was present,
else the most senior element. That is verbatim correct for 3-, 4- and 5-membered
rings (P-22.2.2.1.5.2, `:8394`: "The stems 'iridine', 'etidine', and 'olidine'
are used when nitrogen atoms are present in the ring") and it coincides with the
six-ring rule whenever the least senior heteroatom happens to share a group with
the most senior one -- which is why `1,3-oxazinane` and `1,3-oxaselenane` were
already right. It is wrong for O+As: O is 6A, so the stem came out `ane` and we
emitted `1,3-oxarsane`, which OPSIN cannot parse, so the gate suppressed it and
the molecule fell through to "arsenic compound (not supported)".

Every expectation below is a Blue Book (PIN) or (preselected name) example; the
line number of each is cited beside it.
"""
import pytest

from orthonym.data.hw_stems import get_hw_stem

# P-22.2.2.1.3 (:8284) -- the citation sequence, most senior first.
BB_SENIORITY = ["F", "Cl", "Br", "I", "O", "S", "Se", "Te", "N", "P", "As",
                "Sb", "Bi", "Si", "Ge", "Sn", "Pb", "B", "Al", "Ga", "In", "Tl"]


# (heteroatom set, saturated, expected stem, the BB name it comes from)
BB_SIX_RING_EXAMPLES = [
    ({"O"},        False, "ine",   "1,4-dioxine (PIN), :8417"),
    ({"O"},        True,  "ane",   "1,4-dioxane"),
    ({"S", "Se"},  True,  "ane",   "1,3-thiaselenane (PIN), :8421"),
    ({"N"},        False, "ine",   "1,3,5-triazine (PIN), :8429"),
    ({"O", "N"},   True,  "inane", "1,3-oxazinane (PIN), :8431"),
    ({"P"},        False, "inine", "1,3,5-triphosphinine (PIN), :8437"),
    ({"O", "As"},  True,  "inane", "1,3-oxarsinane (PIN), :8455"),
    ({"Si"},       True,  "inane", "hexasilinane (preselected), :8459"),
    ({"P", "B"},   True,  "inane", "1,3,5,2,4,6-triphosphatriborinane, :8466"),
    ({"P", "B"},   False, "inine", "1,3,5,2,4,6-triphosphatriborinine, :8464"),
    ({"O", "Si"},  True,  "inane", "1,3,5,2,4,6-trioxatrisilinane (PIN), :2082"),
    ({"O", "B"},   True,  "inane", "1,3,5,2,4,6-trioxatriborinane, :37210"),
    ({"S", "B"},   True,  "inane", "1,3,5,2,4,6-trithiatriborinane, :37218"),
    ({"Al"},       True,  "inane", "aluminane, :7942 (see :7938)"),
    ({"Te"},       True,  "ane",   "6A: Te is group A"),
    ({"Bi"},       True,  "ane",   "6A: Bi is group A, NOT group C"),
    ({"Bi"},       False, "ine",   "6A: Bi unsaturated"),
]


@pytest.mark.parametrize("hset,saturated,expected,source", BB_SIX_RING_EXAMPLES)
def test_six_ring_stem_follows_least_senior_heteroatom(
        hset, saturated, expected, source):
    least = max(hset, key=BB_SENIORITY.index)
    assert get_hw_stem(6, saturated, least, ring_heteroatoms=hset) == expected, source


def test_the_most_senior_heteroatom_must_not_drive_the_stem():
    """O+As: O is 6A ('ane'), As is 6C ('inane'). As is least senior, so 'inane'.

    Passing the MOST senior atom as `heteroatom` must not change the answer once
    the full ring set is supplied -- otherwise the caller's choice of "principal"
    heteroatom silently decides a stem the Blue Book assigns from the set.
    """
    assert get_hw_stem(6, True, "O", ring_heteroatoms={"O", "As"}) == "inane"
    assert get_hw_stem(6, True, "As", ring_heteroatoms={"O", "As"}) == "inane"


def test_as_plus_bi_takes_the_group_a_stem_because_bi_is_least_senior():
    """The reverse direction: a 6C atom present does NOT force a 6C stem.

    As (6C) is more senior than Bi (6A) in the P-22.2.2.1.3 sequence, so Bi is
    the least senior and Table 2.5 group A applies. The old "any 6C atom present
    wins" heuristic returned 'inane'/'inine' here.
    """
    assert get_hw_stem(6, True, "As", ring_heteroatoms={"As", "Bi"}) == "ane"
    assert get_hw_stem(6, False, "As", ring_heteroatoms={"As", "Bi"}) == "ine"


def test_single_heteroatom_backwards_compatible_without_the_set():
    """Callers that pass no ring set keep the documented single-atom behaviour."""
    assert get_hw_stem(6, True, "O") == "ane"
    assert get_hw_stem(6, True, "N") == "inane"
    assert get_hw_stem(6, False, "O") == "ine"
    assert get_hw_stem(6, False, "P") == "inine"


def test_smaller_rings_still_use_the_nitrogen_rule():
    """P-22.2.2.1.5.2 (:8394) is a DIFFERENT rule and must not be disturbed:
    "The stems 'iridine', 'etidine', and 'olidine' are used when nitrogen atoms
    are present in the ring; otherwise the 'ane' stems are used."
    """
    assert get_hw_stem(5, True, "N") == "olidine"
    assert get_hw_stem(5, True, "O") == "olane"
    assert get_hw_stem(4, True, "N") == "etidine"
    assert get_hw_stem(3, True, "N") == "iridine"


def test_unknown_element_does_not_crash_or_silently_reclassify():
    """Hg/Zn/Cd appear in no Table 2.5 group and in no citation sequence.

    `rules/heterocycles.py:772` refuses them before a stem is ever needed, but
    the selector must not raise if reached, and must not promote an unlisted
    element to 'least senior' and thereby change a listed atom's stem.
    """
    assert get_hw_stem(6, True, "O", ring_heteroatoms={"O", "Hg"}) == "ane"
