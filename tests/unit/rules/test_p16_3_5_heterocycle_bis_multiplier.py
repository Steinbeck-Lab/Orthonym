"""The heterocycle substituent formatters must choose ``bis``/``tris`` by P-16.3.5(a).

``rules/heterocycles.py`` kept a private ``SIMPLE_MULTIPLIERS.get(count, ...)`` lookup in
``_format_c_substituent`` and ``_format_n_substituent``, so those two producers **could not
emit ``bis`` at all** — every multiplied substituent prefix on a heterocyclic parent came
back with ``di``/``tri``/``tetra`` regardless of whether the prefix was substituted.
``BrCc1cccc(CBr)n1`` was named ``2,6-di(bromomethyl)pyridine``.

THE RULE
--------
``BlueBookV2/BlueBookV2.md:7104``, §**P-16.3.5**: "The numerical prefixes 'bis', 'tris',
'tetrakis', etc. are used to indicate a multiplicity of: **(a) compound or complex (i.e.
substituted) prefixes**", and its own example list prints, verbatim::

    bis(bromomethyl) (preferred prefix, see P-61.3.1)

with ``:25811`` carrying the assembled PIN ``1,2-bis(bromomethyl)benzene (PIN)``.  The
carbocyclic producer (``rules/benzene.py``) already routes through the shared primitive and
spells that row correctly; only the heterocyclic producer diverged.

WHY THIS IS NOT A LOCANT OR ENCLOSURE QUESTION
----------------------------------------------
Two different predicates are in play and they genuinely disagree, which is what makes this
easy to get backwards (``naming_utils.get_multiplier_prefix`` carries a ⛔ note about a
commit that regressed in both directions by conflating them):

* **Enclosure** — P-16.3.4 — ``is_complex_substituent``: does the prefix need ``(...)``?
* **Multiplier** — P-16.3.2(c)/P-16.3.5(a) — ``is_substituted_substituent``: is the prefix
  *substituted*?

``propan-2-yl`` is enclosed but NOT substituted, so it takes the simple multiplier:
``:25721`` ``1,4-di(propan-2-yl)cyclohexane (PIN)``.  ``bromomethyl`` is substituted, so it
takes the derived one.  The negative cases below exist to pin that boundary — a fix that
merely swaps in ``COMPLEX_MULTIPLIERS`` would turn ``di(propan-2-yl)`` into
``bis(propan-2-yl)`` and break the PIN.

The fix routes both formatters through ``naming_utils.multiplied_component``, THE shared
join of multiplier + enclosure + P-16.2.4 hyphen, rather than re-deciding locally.
"""

import pytest

from orthonym import name_compound


@pytest.mark.parametrize(
    "smiles,expected",
    [
        # P-16.3.5(a), verbatim BB preferred prefix `bis(bromomethyl)` (:7104)
        ("BrCc1cccc(CBr)n1", "2,6-bis(bromomethyl)pyridine"),
        ("ClCc1cccc(CCl)n1", "2,6-bis(chloromethyl)pyridine"),
        # same class on other heterocyclic parents -- the defect was in the shared
        # formatter, so it was never pyridine-specific
        ("BrCc1ccc(CBr)o1", "2,5-bis(bromomethyl)furan"),
        ("BrCc1ccc(CBr)s1", "2,5-bis(bromomethyl)thiophene"),
    ],
)
def test_substituted_prefix_takes_bis(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.parametrize(
    "smiles,expected",
    [
        # UNsubstituted prefix -> simple multiplier.  P-16.3.2(a).
        ("Cc1cccc(C)n1", "2,6-dimethylpyridine"),
        # Enclosed but unsubstituted -> STILL simple.  :25721
        # `1,4-di(propan-2-yl)cyclohexane (PIN)` is the governing precedent.
        ("CC(C)c1cccc(C(C)C)n1", "2,6-di(propan-2-yl)pyridine"),
        # Halogen prefixes are simple.
        ("Clc1cccc(Cl)n1", "2,6-dichloropyridine"),
    ],
)
def test_unsubstituted_prefix_keeps_di(smiles, expected):
    """The boundary cases: these must NOT move to bis/tris."""
    assert name_compound(smiles) == expected


def test_tert_butyl_keeps_its_hyphen_and_simple_multiplier():
    """P-16.3.2(a) names `tert-butyl` among the UNSUBSTITUTED prefixes, and
    P-16.2.4.1(d) (``:6964``, verbatim ``di-tert-butyl``) keeps the hyphen.

    This is the case that made the ordering inside ``get_multiplier_prefix``
    load-bearing, so it is asserted end-to-end on a heterocyclic parent too.
    """
    assert name_compound("CC(C)(C)c1cccc(C(C)(C)C)n1") == "2,6-di-tert-butylpyridine"
