""" a phase Task 3 — invented ``-yl`` substituent prefixes (defect class C3).

``parent_to_prefix`` converted a *functional parent* name into a substituent
prefix by string surgery, with two failures measured at ``:

1. Three branches (``-ol``, ``-amine``, ``-one``) ended in a permissive
   ``else: stem = base`` catch-all, so a name that merely *ended in those
   letters* but belonged to a different functional class was captured and
   mangled: ``ethaneperoxol`` -> ``hydroxyethaneperoxyl``,
   ``methanethiol`` -> ``hydroxymethanethiyl`` (an -OH asserted where the
   molecule has -SH), ``O-methylhydroxylamine`` -> ``aminoO-methylhydroxylyl``.
2. The multi-FG branches elided a stem ending ``an``/``a`` but not ``ane``,
   against (the Blue Book) "with elision of the final letter 'e' of parent
   hydrides, when present" -> ``ethane`` + ``yl`` = ``ethaneyl``.

and the terminal fallback appended ``yl`` to aldehyde functional parents that
have a *defined* prefix instead: (the Blue Book) "a -CHO group is
expressed by the preferred prefix 'oxo' if located at an end of a carbon
chain, or, otherwise, by the preferred prefix 'formyl'".

None of ``formaldehydyl``, ``acetaldehydyl``, ``ethaneyl``, ``ethaneperoxyl``
or ``hydroxylylidene`` occurs anywhere in ``the Blue Book Blue Book``.
"""

import pytest

from orthonym.assembly.substituent_naming import (
    ATTACH_LOCANT_UNKNOWN,
    _add_substituent_stereo,
    _elide_parent_hydride_ending,
    parent_to_prefix,
)


# --------------------------------------------------------------------------
# 1. The correct prefix for -CHO is `formyl`, and it is now produced
# --------------------------------------------------------------------------

def test_formaldehyde_yields_formyl_not_formaldehydyl():
    """ (the Blue Book): a -CHO not at a chain end is 'formyl'.

    Formaldehyde has ONE carbon, so the only substituent derivable from it is
    -CHO; the conversion is unambiguous. (the Blue Book) confirms the
    group is spelled 'formyl' and that its H is substitutable.
    """
    assert parent_to_prefix("formaldehyde", 1, attach_locant=ATTACH_LOCANT_UNKNOWN) == "formyl"


def test_formyl_from_formic_acid_still_works():
    """Control: the pre-existing retained-acyl route must not regress."""
    assert parent_to_prefix("formic acid", 1, attach_locant=ATTACH_LOCANT_UNKNOWN) == "formyl"


def test_hydrogen_cyanide_yields_cyano_not_hydrogen_cyanidyl():
    """ (the Blue Book): the -CN group's preferred prefix is 'cyano'.

    The sibling defect named verbatim in rules/ring_assemblies.py's veto
    comment. the Blue Book derives nitriles "from hydrocyanic acid, H-C=N"; HCN has
    one removable H on carbon, so the conversion is unambiguous.
    """
    assert parent_to_prefix("hydrogen cyanide", 1, attach_locant=ATTACH_LOCANT_UNKNOWN) == "cyano"


@pytest.mark.parametrize("name", [
    "water", "ammonia", "carbon dioxide", "hydrogen peroxide",
])
def test_inorganic_parents_fail_closed(name):
    """Not parent hydrides -> no '-yl' form (was 'wateryl', 'ammoniayl',...)."""
    assert parent_to_prefix(name, 1, attach_locant=ATTACH_LOCANT_UNKNOWN) is None


@pytest.mark.parametrize("parent,clen,expected", [
    ("formic acid", 1, "formyl"),
    ("acetic acid", 2, "acetyl"),
])
def test_space_bearing_parent_still_converts(parent, clen, expected):
    """Guard against over-reach: a "contains a space" rule would break these.

     residue Task A re-pointed this from 'henicosyl prop-2-enoate' ->
    '23-carboxytricosyl'. That output is no longer emitted, but NOT because of
    the space: the carboxy locant 23 was read off the whole-fragment carbon
    COUNT, which is not a proof of the fragment's shape. The multi-word parents
    below still convert, so the over-reach guard this test exists for is intact.
    """
    assert parent_to_prefix(
        parent, clen, attach_locant=ATTACH_LOCANT_UNKNOWN) == expected


# --------------------------------------------------------------------------
# 2. Fail closed when no valid prefix exists -- return None, never a word
# --------------------------------------------------------------------------

# (parent_name, chain_length, what the Blue Book uses instead)
FAIL_CLOSED_CASES = [
    # ambiguous attachment: -C(=O)CH3 is 'acetyl', -CH2-CHO is '2-oxoethyl',
    # and parent_to_prefix receives no attachment context to choose between them
    ("acetaldehyde", 2, "acetyl / 2-oxoethyl -- attachment unknown"),
    # (the Blue Book): the prefix for -OOH is 'hydroperoxy'
    ("ethaneperoxol", 2, "hydroperoxy (P-63.4.1)"),
    # -SH is 'sulfanyl'; the old code asserted 'hydroxy', the wrong element
    ("methanethiol", 1, "sulfanyl"),
    ("2-methylundecane-2-thiol", 12, "sulfanyl"),
    # (the Blue Book): =N-OH is 'hydroxyimino', =N-OR '(alkoxyimino)'
    ("hydroxylamine", 0, "hydroxyimino (P-68.3.1.1.2)"),
    ("O-methylhydroxylamine", 1, "methoxyimino (P-68.3.1.1.2)"),
    ("N-methylhydroxylamine", 1, "hydroxyimino family"),
    ("N,N-dimethylhydroxylamine", 2, "hydroxyimino family"),
    # a functional-class name, not a parent hydride
    ("ethylene glycol", 2, "not a parent hydride"),
]


@pytest.mark.parametrize("name,clen,bb_alternative", FAIL_CLOSED_CASES)
def test_fails_closed_returning_none(name, clen, bb_alternative):
    """No valid `-yl` prefix exists -> return None so the caller abstains."""
    assert parent_to_prefix(name, clen, attach_locant=ATTACH_LOCANT_UNKNOWN) is None, (
        f"{name!r} has no '-yl' prefix form; the Blue Book uses "
        f"{bb_alternative}. Emitting a manufactured word is the C3 defect."
    )


def test_caller_propagates_none_rather_than_emitting_a_word():
    """The unguarded call site must abstain, not fabricate.

    ``name_substituent_fragment`` (substituent_naming.py:4596) passes the
    prefix straight into ``_add_substituent_stereo`` without an ``if prefix:``
    guard, so None-tolerance there is what makes failing closed safe.
    """
    assert _add_substituent_stereo(object(), [1, 2], None) is None


# --------------------------------------------------------------------------
# 3. (the Blue Book) elision of the final 'e' -- no `...aneyl`
# --------------------------------------------------------------------------

# residue Task A: the multi-FG branches that used to drive these cases now
# decline -- every locant in 'ethane-1,2-diol' belongs to the CAPPED molecule's
# numbering, the Blue Book), and the count/name pair is not injective over
# fragments. The ELISION rule itself is unchanged and still live, so the cases
# are re-pointed at `_elide_parent_hydride_ending`, which performs it.
ELISION_CASES = [
    ("ethane", "eth"),
    ("propane", "prop"),
    ("hexane", "hex"),
    ("cyclohexane", "cyclohex"),
]


@pytest.mark.parametrize("stem,expected", ELISION_CASES)
def test_final_e_is_elided_before_yl(stem, expected):
    """ method (1) (the Blue Book): 'yl' REPLACES the ending 'ane'."""
    assert _elide_parent_hydride_ending(stem) == expected


def test_no_output_ends_in_aneyl():
    """Class invariant: nothing this module can still emit ends in 'aneyl'."""
    probes = [
        "ethane-1,2-diol", "propane-1,2-diol", "ethane-1,2-diamine",
        "propane-1,2,3-triol", "butane-1,4-diamine", "hexane-2,5-dione",
        "propane", "cyclohexane", "pyridine", "methanol", "acetamide",
    ]
    checked = 0
    for name in probes:
        out = parent_to_prefix(name, 4, attach_locant=ATTACH_LOCANT_UNKNOWN)
        if out is None:
            continue
        checked += 1
        assert not out.endswith("aneyl"), f"{name!r} -> {out!r} violates P-29.2"
        assert not out.endswith("eneyl"), f"{name!r} -> {out!r} violates P-29.2"
    assert checked >= 4, (
        f"only {checked} probes produced a prefix; the loop would be vacuous"
    )


# --------------------------------------------------------------------------
# 4. Regression controls -- these must be byte-identical to d475f8c6
# --------------------------------------------------------------------------

UNCHANGED = [
    ("ethanol", 2, "hydroxyethyl"),
    ("methanol", 1, "hydroxymethyl"),
    # base ends 'an' -> still elidable, so the silanol route is untouched
    ("trimethylsilanol", 3, "hydroxytrimethylsilyl"),
    ("dimethylsilanol", 2, "hydroxydimethylsilyl"),
    ("methanamine", 1, "aminomethyl"),
    ("propane", 3, "propyl"),
    ("cyclohexane", 6, "cyclohexyl"),
    ("acetic acid", 2, "acetyl"),
    # a legitimate parent hydride whose prefix IS '<name>yl' -- must survive
    ("methylhydrazine", 1, "methylhydrazinyl"),
]


@pytest.mark.parametrize("name,clen,expected", UNCHANGED)
def test_unaffected_conversions_are_unchanged(name, clen, expected):
    assert parent_to_prefix(name, clen, attach_locant=ATTACH_LOCANT_UNKNOWN) == expected


# residue Task A: four rows moved out of UNCHANGED. Each borrowed its locant
# from the CAPPED molecule's numbering, and (name, count) is not injective over
# fragments -- '-CH2CH2CH2OH' and '-CH(OH)CH2CH3' both cap to 'propan-1-ol' with
# count 3, and OPSIN 2.9.0 makes the single old answer EXACT for one and a
# DIFFERENT MOLECULE for the other. (the Blue Book) requires the free valence
# to take the lowest locant, which only a caller holding the molecule can honour.
BORROWED_LOCANT = [
    ("propan-2-ol", 3, "was '2-hydroxypropyl'"),
    ("propan-1-amine", 3, "was '1-aminopropyl'"),
    ("propanal", 3, "was '3-oxopropyl' (count-derived)"),
    ("butan-2-one", 4, "was '2-oxobutyl'"),
]


@pytest.mark.parametrize("name,clen,_was", BORROWED_LOCANT)
def test_borrowed_locant_conversions_fail_closed(name, clen, _was):
    assert parent_to_prefix(
        name, clen, attach_locant=ATTACH_LOCANT_UNKNOWN) is None


# --------------------------------------------------------------------------
# 5. The invented words must never be produced, for any probe in this module
# --------------------------------------------------------------------------

INVENTED = ("formaldehydyl", "acetaldehydyl", "ethaneyl", "ethaneperoxyl",
            "hydroxylyl", "thiyl", "glycyl")


def test_no_probe_produces_an_invented_token():
    probes = (
        [(n, c) for n, c, _ in FAIL_CLOSED_CASES]
        + [(n, 4) for n, _ in ELISION_CASES]
        + [(n, c) for n, c, _ in UNCHANGED]
        + [("formaldehyde", 1)]
    )
    assert len(probes) >= 20, "vacuous: probe set unexpectedly small"
    produced = [(n, parent_to_prefix(n, c, attach_locant=ATTACH_LOCANT_UNKNOWN)) for n, c in probes]
    non_none = [(n, o) for n, o in produced if o]
    assert non_none, "vacuous: every probe returned None"
    for name, out in non_none:
        for bad in INVENTED:
            assert bad not in out, f"{name!r} -> {out!r} contains {bad!r}"
