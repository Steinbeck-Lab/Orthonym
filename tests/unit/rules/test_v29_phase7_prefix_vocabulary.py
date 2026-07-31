"""v29 Phase 7 Task 3 — invented ``-yl`` substituent prefixes (defect class C3).

``parent_to_prefix`` converted a *functional parent* name into a substituent
prefix by string surgery, with two failures measured at ``d475f8c6``:

1. Three branches (``-ol``, ``-amine``, ``-one``) ended in a permissive
   ``else: stem = base`` catch-all, so a name that merely *ended in those
   letters* but belonged to a different functional class was captured and
   mangled: ``ethaneperoxol`` -> ``hydroxyethaneperoxyl``,
   ``methanethiol`` -> ``hydroxymethanethiyl`` (an -OH asserted where the
   molecule has -SH), ``O-methylhydroxylamine`` -> ``aminoO-methylhydroxylyl``.
2. The multi-FG branches elided a stem ending ``an``/``a`` but not ``ane``,
   against P-29.2 (BB:15811) "with elision of the final letter 'e' of parent
   hydrides, when present" -> ``ethane`` + ``yl`` = ``ethaneyl``.

and the terminal fallback appended ``yl`` to aldehyde functional parents that
have a *defined* prefix instead: P-66.6.1.3 (BB:35000) "a -CHO group is
expressed by the preferred prefix 'oxo' if located at an end of a carbon
chain, or, otherwise, by the preferred prefix 'formyl'".

None of ``formaldehydyl``, ``acetaldehydyl``, ``ethaneyl``, ``ethaneperoxyl``
or ``hydroxylylidene`` occurs anywhere in ``BlueBookV2/BlueBookV2.md``.
"""

import pytest

from orthonym.assembly.substituent_naming import (
    _add_substituent_stereo,
    parent_to_prefix,
)


# --------------------------------------------------------------------------
# 1. The correct prefix for -CHO is `formyl`, and it is now produced
# --------------------------------------------------------------------------

def test_formaldehyde_yields_formyl_not_formaldehydyl():
    """P-66.6.1.3 (BB:35000): a -CHO not at a chain end is 'formyl'.

    Formaldehyde has ONE carbon, so the only substituent derivable from it is
    -CHO; the conversion is unambiguous.  P-65.1.8.3 (BB:30702) confirms the
    group is spelled 'formyl' and that its H is substitutable.
    """
    assert parent_to_prefix("formaldehyde", 1) == "formyl"


def test_formyl_from_formic_acid_still_works():
    """Control: the pre-existing retained-acyl route must not regress."""
    assert parent_to_prefix("formic acid", 1) == "formyl"


def test_hydrogen_cyanide_yields_cyano_not_hydrogen_cyanidyl():
    """P-66.5.1.1.4 (BB:34734): the -CN group's preferred prefix is 'cyano'.

    The sibling defect named verbatim in rules/ring_assemblies.py's veto
    comment. BB:34687 derives nitriles "from hydrocyanic acid, H-C=N"; HCN has
    one removable H on carbon, so the conversion is unambiguous.
    """
    assert parent_to_prefix("hydrogen cyanide", 1) == "cyano"


@pytest.mark.parametrize("name", [
    "water", "ammonia", "carbon dioxide", "hydrogen peroxide",
])
def test_inorganic_parents_fail_closed(name):
    """Not parent hydrides -> no '-yl' form (was 'wateryl', 'ammoniayl', ...)."""
    assert parent_to_prefix(name, 1) is None


def test_space_bearing_ester_parent_still_converts():
    """Guard against over-reach: a "contains a space" rule would break this.

    Measured in the corpus census: 'henicosyl prop-2-enoate' converts through
    the '-oate' branch, so multi-word names must NOT be denied wholesale.
    """
    assert parent_to_prefix("henicosyl prop-2-enoate", 24) == "23-carboxytricosyl"


# --------------------------------------------------------------------------
# 2. Fail closed when no valid prefix exists -- return None, never a word
# --------------------------------------------------------------------------

# (parent_name, chain_length, what the Blue Book uses instead)
FAIL_CLOSED_CASES = [
    # ambiguous attachment: -C(=O)CH3 is 'acetyl', -CH2-CHO is '2-oxoethyl',
    # and parent_to_prefix receives no attachment context to choose between them
    ("acetaldehyde", 2, "acetyl / 2-oxoethyl -- attachment unknown"),
    # P-63.4.1 (BB:27944): the prefix for -OOH is 'hydroperoxy'
    ("ethaneperoxol", 2, "hydroperoxy (P-63.4.1)"),
    # -SH is 'sulfanyl'; the old code asserted 'hydroxy', the wrong element
    ("methanethiol", 1, "sulfanyl"),
    ("2-methylundecane-2-thiol", 12, "sulfanyl"),
    # P-68.3.1.1.2 (BB:38460): =N-OH is 'hydroxyimino', =N-OR '(alkoxyimino)'
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
    assert parent_to_prefix(name, clen) is None, (
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
# 3. P-29.2 (BB:15811) elision of the final 'e' -- no `...aneyl`
# --------------------------------------------------------------------------

ELISION_CASES = [
    ("ethane-1,2-diol", 2, "1,2-dihydroxyethyl"),
    ("propane-1,2-diol", 3, "1,2-dihydroxypropyl"),
    ("ethane-1,2-diamine", 2, "1,2-diaminoethyl"),
]


@pytest.mark.parametrize("name,clen,expected", ELISION_CASES)
def test_final_e_is_elided_before_yl(name, clen, expected):
    """P-29.2 method (1) (BB:15813): 'yl' REPLACES the ending 'ane'."""
    assert parent_to_prefix(name, clen) == expected


def test_no_output_ends_in_aneyl():
    """Class invariant over every case this module exercises."""
    probes = [n for n, _c, _e in ELISION_CASES] + [
        "propane-1,2,3-triol", "butane-1,4-diamine", "hexane-2,5-dione",
    ]
    assert probes, "vacuous: nothing probed"
    checked = 0
    for name in probes:
        out = parent_to_prefix(name, 4)
        if out is None:
            continue
        checked += 1
        assert not out.endswith("aneyl"), f"{name!r} -> {out!r} violates P-29.2"
        assert not out.endswith("eneyl"), f"{name!r} -> {out!r} violates P-29.2"
    assert checked >= len(ELISION_CASES), (
        f"only {checked} probes produced a prefix; the loop would be vacuous"
    )


# --------------------------------------------------------------------------
# 4. Regression controls -- these must be byte-identical to d475f8c6
# --------------------------------------------------------------------------

UNCHANGED = [
    ("ethanol", 2, "hydroxyethyl"),
    ("methanol", 1, "hydroxymethyl"),
    ("propan-2-ol", 3, "2-hydroxypropyl"),
    # base ends 'an' -> still elidable, so the silanol route is untouched
    ("trimethylsilanol", 3, "hydroxytrimethylsilyl"),
    ("dimethylsilanol", 2, "hydroxydimethylsilyl"),
    ("methanamine", 1, "aminomethyl"),
    ("propan-1-amine", 3, "1-aminopropyl"),
    ("propane", 3, "propyl"),
    ("cyclohexane", 6, "cyclohexyl"),
    ("acetic acid", 2, "acetyl"),
    ("propanal", 3, "3-oxopropyl"),
    ("butan-2-one", 4, "2-oxobutyl"),
    # a legitimate parent hydride whose prefix IS '<name>yl' -- must survive
    ("methylhydrazine", 1, "methylhydrazinyl"),
]


@pytest.mark.parametrize("name,clen,expected", UNCHANGED)
def test_unaffected_conversions_are_unchanged(name, clen, expected):
    assert parent_to_prefix(name, clen) == expected


# --------------------------------------------------------------------------
# 5. The invented words must never be produced, for any probe in this module
# --------------------------------------------------------------------------

INVENTED = ("formaldehydyl", "acetaldehydyl", "ethaneyl", "ethaneperoxyl",
            "hydroxylyl", "thiyl", "glycyl")


def test_no_probe_produces_an_invented_token():
    probes = (
        [(n, c) for n, c, _ in FAIL_CLOSED_CASES]
        + [(n, c) for n, c, _ in ELISION_CASES]
        + [(n, c) for n, c, _ in UNCHANGED]
        + [("formaldehyde", 1)]
    )
    assert len(probes) >= 20, "vacuous: probe set unexpectedly small"
    produced = [(n, parent_to_prefix(n, c)) for n, c in probes]
    non_none = [(n, o) for n, o in produced if o]
    assert non_none, "vacuous: every probe returned None"
    for name, out in non_none:
        for bad in INVENTED:
            assert bad not in out, f"{name!r} -> {out!r} contains {bad!r}"
