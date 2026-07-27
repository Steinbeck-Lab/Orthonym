"""v29 P3-FIX Item 2 — the P-14.5.4 citation tie-break must be a TOTAL order.

Phase 3 replaced several raw ``sorted()`` calls with ``prefix_citation_sort_key``
to get Blue Book citation order. The key was right about the letters and wrong
about the locants, and — decisively — it was **non-injective**: two prefixes that
differ only in their own internal locant compared EQUAL, so ``sorted()`` (stable)
fell back to Counter/dict insertion order, which is RDKit neighbour order, which
is how the SMILES happened to be written. One molecule, two spellings, two names.

Measured before the fix, all three the same molecule::

    CCC(C)CSSSCCC(C)C  -> 1-(2-methylbutyl)-3-(3-methylbutyl)trisulfane
    CC(C)CCSSSCC(CC)C  -> 1-(3-methylbutyl)-3-(2-methylbutyl)trisulfane

The nomenclature rule, with its heading — ``### **P-14.5** ALPHANUMERICAL ORDER``
→ ``**P-14.5.4**`` (``BlueBookV2.md:3517``): "*When two or more prefixes consist
of identical Roman letters, priority for order of citation is given to the group
that contains the lowest locant(s) at the first point of difference.*" Its own
first example is exactly this pair (``BlueBookV2.md:3521``):

    4-(2-methylbutyl)-N-(3-methylbutyl)aniline (PIN)
    "(for ordering the substituents '2' is lower than '3'; the fact that 'N' is
     lower than '4' is irrelevant)"

and ``BlueBookV2.md:3533`` fixes the COMPARISON to be term-by-term in order of
appearance rather than on a sorted set:

    1-(2-methylpentan-3-yl)-1-(3-methylpentan-2-yl)cyclopentane (PIN)
    "[not ...; the locant set '2,3' is lower than '3,2']"

Root cause of the non-injectivity: ``prefix_citation_sort_key`` was stripping the
prefix's LEADING locant set before collecting locants, so ``2-methylbutyl`` and
``3-methylbutyl`` both keyed to ``('methylbutyl', ())``. That strip is correct for
the OTHER convention this one function is called with — a fully rendered prefix
string like ``3,5-dichloro``, whose leading locants are PARENT locants assigned BY
citation order and so may not decide it — and wrong for a bare substituent name,
whose leading locant is its own. The two conventions are now explicit, and a final
full-string tier makes the order total so no tie can ever reach atom order again.
The full-string tier is an ENGINEERING requirement (determinism), never a
nomenclature claim: it is only ever consulted after P-14.5 has been exhausted.
"""

import pytest
from rdkit import Chem

from orthonym.assembly.naming_utils import prefix_citation_sort_key


# --------------------------------------------------------------------------
# 1. the key itself: injective, and ordered the way P-14.5.4 says
# --------------------------------------------------------------------------

# Every one of these is a DISTINCT substituent prefix, so no two may share a key.
DISTINCT_PREFIXES = [
    'methyl', 'ethyl', 'propyl', 'butyl', 'octyl', 'decyl', 'dodecyl',
    '2-methylbutyl', '3-methylbutyl',
    'pentan-2-yl', 'pentan-3-yl', 'hexan-2-yl', 'hexan-3-yl', 'decan-2-yl',
    '2-methylpentan-2-yl', '3-methylpentan-3-yl',
    '2-methylpentan-3-yl', '3-methylpentan-2-yl',
    '1-chloroethyl', '2-chloroethyl',
    '2-methylphenyl', '3-methylphenyl', '4-methylphenyl',
    '2-methylcyclohexyl', '3-methylcyclohexyl',
    'tert-butyl', 'cyclohexylmethyl', '(4-methylphenyl)methyl',
]


def test_the_citation_key_is_injective():
    """A non-injective key is what let a tie fall through to RDKit atom order."""
    keys = {}
    for p in DISTINCT_PREFIXES:
        k = prefix_citation_sort_key(p)
        assert k not in keys, (
            "key collision: %r and %r both -> %r" % (keys.get(k), p, k))
        keys[k] = p
    assert len(keys) == len(DISTINCT_PREFIXES)


@pytest.mark.parametrize("pair,expected", [
    # BB 3521, the rule's own first example: '2' is lower than '3'.
    (['3-methylbutyl', '2-methylbutyl'], ['2-methylbutyl', '3-methylbutyl']),
    # BB 3529: `1-(pentan-2-yl)-4-(pentan-3-yl)benzene (PIN)`.
    (['pentan-3-yl', 'pentan-2-yl'], ['pentan-2-yl', 'pentan-3-yl']),
    # BB 3531: the locant set '2,2' is lower than '3,3'.
    (['3-methylpentan-3-yl', '2-methylpentan-2-yl'],
     ['2-methylpentan-2-yl', '3-methylpentan-3-yl']),
    # BB 3533: compared IN ORDER OF APPEARANCE -- '2,3' is lower than '3,2'.
    (['3-methylpentan-2-yl', '2-methylpentan-3-yl'],
     ['2-methylpentan-3-yl', '3-methylpentan-2-yl']),
    # BB 3527: `6-(1-chloroethyl)-5-(2-chloroethyl)-1H-indole (PIN)`
    # "(the locant '1' is lower than '2')".
    (['2-chloroethyl', '1-chloroethyl'], ['1-chloroethyl', '2-chloroethyl']),
])
def test_identical_letters_resolve_by_lowest_internal_locant(pair, expected):
    assert sorted(pair, key=prefix_citation_sort_key) == expected


def test_a_double_digit_locant_is_compared_as_a_NUMBER_not_a_string():
    """The total-order tier must never be allowed to decide a P-14.5.4 question:
    as strings '10-' sorts before '2-', but locant 2 is lower than locant 10."""
    got = sorted(['10-methylundecyl', '2-methylundecyl'],
                 key=prefix_citation_sort_key)
    assert got == ['2-methylundecyl', '10-methylundecyl'], got


def test_letters_still_win_over_locants():
    """P-14.5.4 only applies once the Roman letters are IDENTICAL; it must not
    reorder prefixes that differ in letters (``### **P-14.5** ALPHANUMERICAL
    ORDER`` preamble, ``BlueBookV2.md:3442``)."""
    got = sorted(['2-methylbutyl', 'ethyl'], key=prefix_citation_sort_key)
    assert got == ['ethyl', '2-methylbutyl'], got


# --------------------------------------------------------------------------
# 2. end to end: one molecule, many spellings, ONE name
# --------------------------------------------------------------------------
# The two arms always have IDENTICAL Roman letters and different internal
# locants, which is the only shape that reached the collision.

SPELLING_WITNESSES = [
    # (family, smiles) -- each is ONE molecule; the harness re-spells it itself.
    ('polychalcogen trisulfane',      'CCC(C)CSSSCCC(C)C'),
    ('polyazane hydrazine',           'CCC(C)CNNCCC(C)C'),
    ('polyazane triazane',            'CCC(C)CNNNCCC(C)C'),
    ('mononuclear hydride arsane',    'CCC(C)C[As](C)CCC(C)C'),
    ('group-14 silyl prefix',         'CCC(C)C[Si](CCC(C)C)(C)CC(=O)O'),
    # branched secondary alkyls: BB 3533's '2,3' vs '3,2' shape
    ('trisulfane branched alkyls',    'CCC(C)C(C)SSSC(C)C(C)CC'),
    # ring-bearing arms
    ('hydrazine ring-bearing arms',   'Cc1ccccc1CNNCc1cccc(C)c1'),
    ('trisulfane ring-bearing arms',  'Cc1ccccc1CSSSCc1cccc(C)c1'),
]


@pytest.mark.parametrize("family,smiles", SPELLING_WITNESSES)
def test_one_molecule_gets_one_name_however_it_is_spelled(family, smiles):
    """Probed the way the gate probes: input + RDKit canonical + 6 seeded
    re-spellings (``, the gate's own
    n_rand=6 / seed=0 defaults). ``determinism_eval``'s stated policy is that a
    non-quarantined probe yielding >1 name is a HARD FAIL."""
    import importlib.util
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        '_det_eval', root / 'scripts' / 'determinism_eval.py')
    det = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(det)

    from orthonym.namer import Orthonym
    namer = Orthonym()
    res = det.probe_smiles(smiles, lambda s: namer.name(s),
                           n_rand=6, seed=0, timeout=120.0)
    assert res['n_distinct'] == 1, (
        "%s: %d distinct names for one molecule: %r"
        % (family, res['n_distinct'], res['names']))


def test_the_trisulfane_witness_is_cited_in_the_blue_book_order():
    """Not just stable — stable on the RIGHT one. BB 3521: '2' before '3'."""
    from orthonym.namer import Orthonym
    namer = Orthonym()
    got = namer.name(Chem.CanonSmiles('CCC(C)CSSSCCC(C)C'))
    assert got == '1-(2-methylbutyl)-3-(3-methylbutyl)trisulfane', got


def test_the_locant_tier_is_in_order_of_appearance_not_sorted():
    """``BlueBookV2.md:3533`` compares the locants term-by-term AS CITED:

        1-(2-methylpentan-3-yl)-1-(3-methylpentan-2-yl)cyclopentane (PIN)
        "[not 1-(3-methylpentan-2-yl)-1-(2-methylpentan-3-yl)cyclopentane;
          the locant set '2,3' is lower than '3,2']"

    '3,2' is only "higher" than '2,3' if the two are NOT sorted first — sorting
    would make both ``(2,3)`` and lose the distinction the rule turns on. Pinned
    on the key directly because the natural naming witness is rescued by the
    total-order tier and so cannot see the difference.
    """
    from orthonym.assembly.name_comparison import locant_sort_key
    descending = prefix_citation_sort_key('3-methylpentan-2-yl')[1]
    assert descending == (locant_sort_key('3'), locant_sort_key('2')), descending
    ascending = prefix_citation_sort_key('2-methylpentan-3-yl')[1]
    assert ascending == (locant_sort_key('2'), locant_sort_key('3')), ascending
    assert ascending < descending


def test_the_aniline_site_cites_by_the_groups_own_locant_not_the_parent_one():
    """``**P-14.5.4**``'s own first example (``BlueBookV2.md:3521``) is an aniline:

        4-(2-methylbutyl)-N-(3-methylbutyl)aniline (PIN)
        "(for ordering the substituents '2' is lower than '3'; the fact that 'N'
          is lower than '4' is irrelevant)"

    This is the site that sorts already-RENDERED prefix strings, where the leading
    locant is a PARENT locant and must NOT decide. Read the rendered string as if
    the leading locant were the group's own and the order inverts to
    ``N-3-methylbutyl-4-(2-methylbutyl)aniline`` — measured, so this pins the
    convention rather than merely restating it.

    Only the ORDER is asserted. The N-substituent's missing enclosing marks
    (``N-3-methylbutyl`` where BB writes ``N-(3-methylbutyl)``) is a separate,
    pre-existing enclosure gap on this path and is deliberately not pinned here.
    """
    from orthonym.namer import Orthonym
    namer = Orthonym()
    got = namer.name(Chem.CanonSmiles('CCC(C)Cc1ccc(NCCC(C)C)cc1'))
    assert '2-methylbutyl' in got and '3-methylbutyl' in got, got
    assert got.index('2-methylbutyl') < got.index('3-methylbutyl'), got
    assert got.startswith('4-(2-methylbutyl)'), got
