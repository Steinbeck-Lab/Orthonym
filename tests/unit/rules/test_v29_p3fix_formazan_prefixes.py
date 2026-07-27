"""v29 P3-FIX Item 5 — the formazan site had ZERO behavioural tests.

The Phase 3 organyl migration widened `_formazan_substituent`, so newly-admitted
compound prefixes reached a composer that was still on pre-migration raw code. It
got three things wrong at once, and nothing caught it because a whole-`tests/` grep
for `formazan` returned only the bare-parent trivial-name row and a JSON fixture:

  1. a LOCAL `_SUB_MULTIPLIER` table -- so no `bis`/`tris` for a substituted
     prefix (`**P-16.3.5**`, ``BlueBookV2.md:7104``: the numerical prefixes
     'bis', 'tris', 'tetrakis' "*are used to indicate a multiplicity of (a)
     compound or complex (i.e. substituted) prefixes*", examples
     ``bis(bromomethyl)``, ``bis(2-chloropropan-2-yl)``, both "preferred prefix")
     and no hyphen for an italicized one (`**P-16.2.4.1**`(d), ``:6957/:6964``);
  2. a bare ``f"({nm})"`` instead of the escalating `enclose_if_compound`, so a
     name already containing parentheses got a second same-level pair.
     ``grep -c 'bis((' BlueBookV2.md`` is **0** against 7 hits for
     ``bis[(...)...]``, per `## **P-16.5.4** Multiple types of enclosing marks`
     -> `**P-16.5.4.1.5**` (``:7509``);
  3. `alpha_sort_key` for the citation order, which ties on identical letters.

Shipped before this fix, both OPSIN-clean so SELF-01 passed them:
``1,5-ditert-butylformazan`` and ``1,5-di((3-methylphenyl)methyl)formazan``.

`formazan` is a retained name that IS a preferred IUPAC name "*fully substitutable
by suffixes and prefixes*" (``BlueBookV2.md:16521``), and
`## **P-68.3.1.3.5.1** Derivatives of formazan` (``:38930``) constructs preferred
names "*systematically as derivatives of formazan when prefixes only are present*".
So this block is an ordinary substituent-prefix block and belongs to the canonical
`format_substituent_prefix` -- which is what the sibling `_name_diazene_oxide`
100 lines up in the same module already uses.
"""

import pytest
from rdkit import Chem


@pytest.fixture
def namer():
    from orthonym.namer import Orthonym
    return Orthonym()


@pytest.mark.parametrize("smiles,expected", [
    # An ITALICIZED simple prefix: hyphen, simple `di` (P-16.2.4.1(d) +
    # P-16.3.2(a)). Was `1,5-ditert-butylformazan`.
    ("CC(C)(C)NN=CN=NC(C)(C)C", "1,5-di-tert-butylformazan"),
    # A SUBSTITUTED (compound) prefix that already carries parentheses: `bis`
    # per P-16.3.5(a), escalated to BRACKETS per P-16.5.4.1.5.
    # Was `1,5-di((3-methylphenyl)methyl)formazan`.
    ("c1ccc(C)cc1CNN=CN=NCc1cc(C)ccc1",
     "1,5-bis[(3-methylphenyl)methyl]formazan"),
    # A substituted prefix WITHOUT inner marks: `bis(...)`, plain parentheses.
    ("C1CCCCC1CNN=CN=NCC1CCCCC1", "1,5-bis(cyclohexylmethyl)formazan"),
    # Unsubstituted simple prefixes are byte-identical to before -- the BB's own
    # derivative example is this shape (`1,3-diphenylformazan (PIN)`).
    ("CCNN=CN=NCC", "1,5-diethylformazan"),
    ("c1ccccc1NN=CN=Nc1ccccc1", "1,5-diphenylformazan"),
])
def test_formazan_prefixes_are_spelled_the_blue_book_way(namer, smiles, expected):
    assert namer.name(Chem.CanonSmiles(smiles)) == expected


def test_no_formazan_name_double_encloses_at_one_level():
    """P-16.5.4.1.5: once parentheses are used inside a prefix, the outer mark
    escalates to brackets. `bis((` occurs ZERO times in the Blue Book."""
    from orthonym.namer import Orthonym
    got = Orthonym().name(
        Chem.CanonSmiles("c1ccc(C)cc1CNN=CN=NCc1cc(C)ccc1"))
    assert 'bis((' not in got and 'di((' not in got, got
    assert 'bis[(' in got, got


def test_the_italicized_prefix_keeps_its_hyphen_under_a_multiplier():
    from orthonym.namer import Orthonym
    got = Orthonym().name(Chem.CanonSmiles("CC(C)(C)NN=CN=NC(C)(C)C"))
    assert 'ditert' not in got, got
    assert 'di-tert-butyl' in got, got


def test_the_bare_parent_still_reaches_the_retained_name(namer):
    """The composer must not claim an UNsubstituted formazan -- that row belongs
    to the retained-name table, and returning early is how it gets there."""
    got = namer.name(Chem.CanonSmiles("NN=CN=N"))
    assert 'formazan' in got, got
    assert not got.startswith('1,5-'), got
