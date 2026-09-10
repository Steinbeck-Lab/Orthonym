"""A composed heteroatom prefix must name its organyl by CONSTITUTION.

A composed prefix -- alkyl+amino, alkyl+oxy, alkyl+sulfanyl -- used to name its
organyl half by COUNTING ITS CARBONS and indexing an alkyl-stem table
(``get_alkyl_name(n)`` / ``ALKOXY_NAMES[n]``). A carbon count is not a
constitution: butyl, 2-methylpropyl, butan-2-yl and tert-butyl are FOUR
different C4H9 groups and the count named all four 'butyl', so three of every
four such names described a molecule other than the one drawn.

The producers were NOT the six ``_name_*_branch`` helpers in
substituent_enumerator (verified by wrapping all six: none fires for these
inputs). They are:
  * ``composer._check_for_acylamino`` -- the non-carbonyl tail, whose own
    comment admitted it kept ``_count_carbon_chain`` + ``get_alkyl_name``
    "BYTE-IDENTICAL";
  * ``composer._name_n_attached_substituent_fallback`` -- the same shape;
  * ``substituent_prefix_forms.get_alkoxy_prefix`` -- ``ALKOXY_NAMES[count]``.

Blue Book authority for every expected name below (``the Blue Book Blue Book``):

  PIN status of the four C4H9 prefixes
    * ``tert-butyl`` IS a preferred prefix and is cited BARE -- the Blue Book
      "tert-butyl* (unsubstituted) = 2-methylpropan-2-yl | (CH3)3C- |
      ; ", and the PIN example at the Blue Book
      "*tert*-butyldi(methyl)phosphane (PIN)".
    * ``isopropyl`` is NOT a PIN: the Blue Book "isopropyl = propan-2-yl* =
      1-methylethyl | ", and at the Blue Book "The prefixes
      isopropyl, isopropylidene, and trityl are retained for use in general
      nomenclature but no substitution of any kind is allowed." PIN =
      ``propan-2-yl``.
    * ``sec-butyl`` is NOT a PIN: the Blue Book "butan-2-yl* = 1-methylpropyl (not
      sec-butyl; not but-2-yl) | ", and the Blue Book lists *sec*-butyl among
      prefixes "no longer recommended; the first name in parentheses is the
      preferred prefix name". PIN = ``butan-2-yl``.
    * ``isobutyl`` is NOT a PIN: the Blue Book "2-methylpropyl (preferred prefix)
      (not isobutyl)" and the Blue Book. PIN = ``2-methylpropyl``.

  Enclosing marks -- around the ORGANYL, suffix OUTSIDE
    * the Blue Book "-NH-CH3 methylamino (preferred prefix)" -- a SIMPLE organyl is
      bare; the Blue Book "-NH-CH2Cl (chloromethyl)amino (preferred prefix)" -- a
      COMPOUND organyl is enclosed and 'amino' stays outside the marks.

  Alkoxy morphology, verbatim (the Blue Book-27691)
    * "CH3-[CH2]3-O- butoxy (preferred prefix)" -- retained CONTRACTED, bare;
    * "(CH3)3C-O- *tert*-butoxy (preferred prefix) (no substitution)", and
      the Blue Book marks 'tert-butyloxy' as "not";
    * "(CH3)2CH-O-... (propan-2-yl)oxy (preferred prefix)";
    * "CH3-CH2-CH(CH3)-O- (butan-2-yl)oxy (preferred prefix) (not *sec*-butoxy)";
    * "(CH3)2CH-CH2-O- 2-methylpropoxy (preferred prefix) (not isobutoxy)" --
      note this one CONTRACTS and is BARE, it is not '(2-methylpropyl)oxy'.

  Multiplicity (a) vs (a))
    * 'di' for a simple prefix that merely carries a locant -- the Blue Book
      "2-[di(butan-2-yl)amino]butan-2-ol (PIN)", the Blue Book
      "1,4-di(propan-2-yl)cyclohexane (PIN)";
    * 'bis' for a compound (substituted) prefix -- the Blue Book
      "bis(2-methylpropyl)", the Blue Book "bis(chloromethyl)aminoxyl (PIN)".

  Mixed simple/compound citation
    * the Blue Book "3-[methyl(phenyl)amino]phenol" -- a simple organyl is bare when
      FIRST-cited and parenthesized afterwards.

Every expected end-to-end name here is OPSIN-round-trip clean (structure
MATCH). OPSIN proves VALIDITY only; the PIN authority is the Blue Book above.

These names ship in the JVM-absent mode, where the constitutional gate
fails OPEN and cannot suppress a wrong constitution. The autouse conftest
fixture disables that gate, so these tests run in exactly that mode.
"""
import pytest

from orthonym.assembly.substituent_enumerator import (
    cite_organyl_in_composed_prefix,
    composed_alkoxy_prefix,
    composed_prefix_multiplier,
)
from orthonym.namer import name_compound

# --------------------------------------------------------------------------
# The four C4H9 constitutions plus the C3 pair. ONE carbon count (4) covers
# rows 1-4: before the fix all four collapsed onto a single string.
# --------------------------------------------------------------------------
C4H9_AMINO = [
    ("CC(C)(C)NCCO", "2-(tert-butylamino)ethan-1-ol", "tert-butyl (P-29.6.1)"),
    ("CC(C)CNCCO", "2-[(2-methylpropyl)amino]ethan-1-ol", "isobutyl -> BB:16412"),
    ("CC(CC)NCCO", "2-[(butan-2-yl)amino]ethan-1-ol", "sec-butyl -> BB:55642"),
    ("CCCCNCCO", "2-(butylamino)ethan-1-ol", "n-butyl, unchanged"),
]

C3H7_AMINO = [
    ("CC(C)NCCO", "2-[(propan-2-yl)amino]ethan-1-ol", "isopropyl -> BB:56251"),
    ("CCCNCCO", "2-(propylamino)ethan-1-ol", "n-propyl, unchanged"),
]

C4H9_ALKOXY = [
    ("CC(C)(C)OCCO", "2-tert-butoxyethan-1-ol", "BB:27679 tert-butoxy"),
    ("CC(C)COCCO", "2-(2-methylpropoxy)ethan-1-ol", "BB:27689 2-methylpropoxy"),
    ("CC(CC)OCCO", "2-[(butan-2-yl)oxy]ethan-1-ol", "BB:27687 (butan-2-yl)oxy"),
]


@pytest.mark.parametrize("smiles,expected,why", C4H9_AMINO + C3H7_AMINO)
def test_amino_prefix_names_the_constitution(smiles, expected, why):
    """Each alkylamino prefix spells its own constitution (not its carbon count)."""
    assert name_compound(smiles) == expected, why


@pytest.mark.parametrize("smiles,expected,why", C4H9_ALKOXY)
def test_alkoxy_prefix_names_the_constitution(smiles, expected, why):
    """: each C4H9-oxy group gets its own preferred prefix."""
    assert name_compound(smiles) == expected, why


def test_four_c4h9_amino_isomers_get_four_distinct_names():
    """The regression tripwire: a carbon count gave these ONE name for FOUR
    molecules. Distinctness is necessary but not sufficient, so the exact PINs
    are asserted above as well."""
    names = [name_compound(smi) for smi, _, _ in C4H9_AMINO]
    assert len(set(names)) == 4, f"isomers collapsed onto {names}"


def test_four_c4h9_alkoxy_isomers_are_distinct_from_n_butoxy():
    """'butoxy' belongs to CH3-[CH2]3-O- alone (the Blue Book); no branched isomer
    may claim it."""
    for smi, expected, _ in C4H9_ALKOXY:
        assert "2-butoxyethan-1-ol" != name_compound(smi)
        assert expected == name_compound(smi)


def test_dialkylamino_multiplicity_is_bluebook_di_not_bis():
    """the Blue Book '2-[di(butan-2-yl)amino]butan-2-ol (PIN)': a locant-bearing
    SIMPLE prefix takes 'di', not 'bis'."""
    assert name_compound("CC(C)N(C(C)C)CCO") == "2-[di(propan-2-yl)amino]ethan-1-ol"


def test_mixed_simple_and_italicized_branches_cite_per_p16_5_1_3_1():
    """the Blue Book '3-[methyl(phenyl)amino]phenol': the first-cited simple organyl
    is bare, the next is parenthesized. 'tert-butyl' is simple."""
    assert name_compound("CC(C)(C)N(C)CCO") == "2-[tert-butyl(methyl)amino]ethan-1-ol"


# --------------------------------------------------------------------------
# The primitive, driven directly off the Blue Book tables.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("token,expected", [
    ("methyl", "methyl"),                       # the Blue Book bare
    ("butyl", "butyl"),
    ("cyclohexyl", "cyclohexyl"),
    ("tert-butyl", "tert-butyl"),               # the Blue Book cited bare
    ("sec-butyl", "sec-butyl"),
    ("propan-2-yl", "(propan-2-yl)"),
    ("butan-2-yl", "(butan-2-yl)"),
    ("2-methylpropyl", "(2-methylpropyl)"),
    ("chloromethyl", "(chloromethyl)"),         # the Blue Book, hyphen-free compound
    ("tert-butylsulfanyl", "(tert-butylsulfanyl)"),
])
def test_cite_organyl_marks(token, expected):
    """: marks around a COMPOUND organyl, none around a simple one."""
    assert cite_organyl_in_composed_prefix(token) == expected


@pytest.mark.parametrize("token,expected", [
    ("methyl", "methoxy"),
    ("ethyl", "ethoxy"),
    ("propyl", "propoxy"),
    ("butyl", "butoxy"),
    ("tert-butyl", "tert-butoxy"),              # NOT tert-butyloxy (the Blue Book)
    ("propan-2-yl", "(propan-2-yl)oxy"),
    ("butan-2-yl", "(butan-2-yl)oxy"),
    ("2-methylpropyl", "2-methylpropoxy"),      # contracts AND stays bare
    ("pentyl", "pentyloxy"),                    # C5+ does not contract
    ("3-methylbutyl", "3-methylbutoxy"),
])
def test_composed_alkoxy_prefix_matches_p63_2_2_2(token, expected):
    """The table (the Blue Book-27691), transcribed."""
    assert composed_alkoxy_prefix(token) == expected


@pytest.mark.parametrize("token,expected", [
    ("methyl", "di"),
    ("tert-butyl", "di"),
    ("propan-2-yl", "di"),                      # the Blue Book di(propan-2-yl)
    ("butan-2-yl", "di"),                       # the Blue Book di(butan-2-yl)amino
    ("prop-1-en-2-yl", "di"),                   # the Blue Book di(prop-1-en-2-yl)
    ("2-methylpropyl", "bis"),                  # the Blue Book bis(2-methylpropyl)
    ("chloromethyl", "bis"),                    # the Blue Book bis(chloromethyl)
    ("2-chloropropan-2-yl", "bis"),             # the Blue Book
])
def test_multiplier_di_vs_bis(token, expected):
    """(a) 'di' for a locant-bearing simple prefix vs (a) 'bis'
    for a substituted one."""
    assert composed_prefix_multiplier(token, 2) == expected


@pytest.mark.parametrize("sentinel", [
    "substituent",                      # the cascade placeholder
    "unknown organic compound",         # whole-molecule refusal
    "zinc compound (not supported)",    # descriptive fallback
    "",
])
def test_refusal_sentinel_is_never_welded_into_a_composed_prefix(monkeypatch,
                                                                sentinel):
    """A sentinel accepted into a prefix slot becomes part of a name that reads
    as success ('zinc compound (not supported)ylethane'). Recognition is
    delegated to the ONE shared errors.is_refusal_sentinel predicate."""
    from rdkit import Chem

    from orthonym.assembly import substituent_enumerator as se
    mol = Chem.MolFromSmiles("CCO")
    monkeypatch.setattr(se, "name_substituent", lambda *a, **k: sentinel)
    assert se.composed_prefix_organyl_name(mol, [0, 1], 0) is None


def test_multi_word_organyl_is_refused():
    """A whole compound name is not a prefix token."""
    from rdkit import Chem

    from orthonym.assembly import substituent_enumerator as se
    mol = Chem.MolFromSmiles("CCO")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(se, "name_substituent", lambda *a, **k: "methylboronic acid")
        assert se.composed_prefix_organyl_name(mol, [0, 1], 0) is None


def test_empty_fragment_fails_closed():
    from rdkit import Chem

    from orthonym.assembly.substituent_enumerator import (
        composed_prefix_organyl_name,
    )
    assert composed_prefix_organyl_name(Chem.MolFromSmiles("CCO"), [], 0) is None
