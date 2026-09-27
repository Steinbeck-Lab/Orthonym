""" a phase (Job 1) — the italicized-prefix carve-out, decided in ONE place.

a phase (``) unified five open-coded copies of the compound-hyphen
predicate behind ``assembly.naming_utils.has_structural_hyphen`` and added a
structural tripwire. A review then found the unification INCOMPLETE and the
tripwire unable to have caught the survivors: it searched only for
``startswith(("tert-``, so it was blind to

* a raw ``'-' in t`` with no carve-out at all (three sites),
* ``startswith('tert-')`` in its single-argument form,
* ``for skip in ('tert-', 'sec-')`` loop forms,
* character-set membership tests whose class contains a hyphen
  (``any(c in nm for c in '-0123456789')``),
* and a SECOND hand-written carve-out list (``r_name not in ('tert-butyl',
  'sec-butyl')``), the one construct guaranteed to drift from the primitive.

Two of the survivors were shipping defects: ``(tert-butyl)(tert-butyl)zinc`` and
the outright malformed ``ditert-butylzinc`` / ``ditert-butylmethylsilane``.

**(b)/(d) / ** — the hyphen of a leading italicized structural prefix is
part of a SIMPLE retained name. Verbatim, on-disk ``the Blue Book``:

* ``:16282`` "The retained name '*tert*-butyl' has never been recommended for
  further substitution... Acceptable locants have never been adopted for this
  name." — so its hyphen can never be a locant boundary.
* ``:16286`` ``*tert*-butyldi(methyl)phosphane (PIN)`` — cited BARE.
* ``:18929`` ``*tert*-butyldi(methyl)(oxiranylmethoxy)silane (PIN)``.
* ``:25717`` ``1,2-di-*tert*-butylbenzene (PIN)`` — ``di-``, hyphenated; not
  ``bis(tert-butyl)`` and certainly not ``ditert-butyl``.
* ``:3507`` ``1-(butan-2-yl)-3-*tert*-butylbenzene (PIN)`` — a bare ``tert-``
  beside an ENCLOSED ``butan-2-yl``, and ``tert-`` ignored in the ordering.

**** (``:27665``) — the retained ``R-O–`` contractions, "used both as
preferred IUPAC prefixes" and "considered as simple prefixes requiring the
numerical prefixes 'di', 'tri'", listing ``(CH3)3C-O– *tert*-butoxy (preferred
prefix) (no substitution)``. The index rejects the alternatives by name:
``:55662`` ``tert-butoxy* (unsubstituted) =... (not tert-butyloxy)`` and
``:55646`` ``(butan-2-yl)oxy*... (not sec-butoxy; not sec-butyloxy)``.

Every namer assertion runs with the OPSIN validity gate explicitly DISABLED —
the no-JRE mode where the gate fails OPEN and nothing downstream can rescue a
wrong producer output.
"""

import re
from pathlib import Path

import pytest

import orthonym
from orthonym.assembly.naming_utils import (
    enclose_if_compound,
    format_substituent_prefix,
    get_multiplier_prefix,
    has_structural_hyphen,
    is_complex_substituent,
    italicized_prefix_is_bare,
    multiplier_needs_hyphen,
    needs_brackets,
    strip_italicized_structural_prefix,
)
from orthonym.errors import is_failure_name, is_refusal_sentinel
from orthonym.namer import Orthonym


@pytest.fixture
def ungated_namer(monkeypatch):
    """A namer with the OPSIN validity gate explicitly DISABLED.

    The gate fails OPEN when no JRE is present, so every safety property here is
    asserted in the mode where nothing downstream can suppress a wrong producer
    output. Both defects in this module were leaking in exactly that mode.
    """
    import orthonym.namer as _namer
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", True, raising=False)
    return Orthonym()


# ==========================================================================
# part A: the primitives
# ==========================================================================

@pytest.mark.parametrize("name", ["tert-butyl", "sec-butyl"])
def test_italicized_led_simple_name_is_bare(name):
    """: no marks, SIMPLE multiplier, and the hyphen is not structural."""
    assert has_structural_hyphen(name) is False
    assert italicized_prefix_is_bare(name) is True
    assert needs_brackets(name) is False
    assert is_complex_substituent(name) is False
    assert enclose_if_compound(name) == name        # cited BARE (BB 16286)
    # -FIX Item 4: the (d) hyphen is now part of the
    # multiplied TOKEN and is produced by this one primitive, so the
    # multiplier comes back as `di-`. The assertion's point is unchanged:
    # the SIMPLE `di` and not the derived `bis` (a)).
    assert get_multiplier_prefix(2, name) == "di-"
    assert not get_multiplier_prefix(2, name).startswith("bis")
    assert multiplier_needs_hyphen(name) is True    # 'di-tert-butyl'


@pytest.mark.parametrize("name", [
    "tert-butylsulfanyl",        # compound chalcogen prefix
    "tert-butylamino",           # compound two-prefix name
    "tert-butyl-dimethylsilyl",  # a SECOND, genuinely structural hyphen
    "2-tert-butyl",              # a locant
])
def test_the_carve_out_never_swallows_a_compound_name(name):
    """The carve-out is about the PREFIX only; the REMAINDER still decides.

    This is the property that keeps the fix narrower than the open-code it
    replaced: 'tert-butyl' reduces to the simple 'butyl' and is cited bare, but
    'tert-butylsulfanyl' reduces to 'butylsulfanyl' — still compound under the
     chalcogen rule, so it keeps its marks exactly as '(methylsulfanyl)'
    does.
    """
    assert italicized_prefix_is_bare(name) is False
    assert enclose_if_compound(name).startswith(("(", "["))


def test_italicized_prefix_is_bare_agrees_with_the_canonical_predicate():
    """``italicized_prefix_is_bare`` must never say "bare" where the canonical
    enclosing predicate says "compound".

    It is offered to sites that keep their OWN compound test, so a disagreement
    would ship two spellings of one name. Judging the remainder by
    ``needs_brackets`` ALONE did exactly that: needs_brackets misses the bare
    two-prefix compound 'butylamino', so 'tert-butylamino' came back bare while
    ``enclose_if_compound`` enclosed it.
    """
    for name in ["tert-butyl", "sec-butyl", "tert-butylsulfanyl",
                 "tert-butylamino", "tert-butyl-dimethylsilyl", "2-tert-butyl",
                 "methyl", "phenyl", "2-methylpropyl", "cyclohexylmethyl"]:
        if italicized_prefix_is_bare(name):
            assert enclose_if_compound(name) == name, name


def test_is_complex_substituent_decides_on_the_stripped_remainder():
    """MINOR from the review: the carve-out was only an EARLY-EXIT gate, so every
    suffix rule below it stayed anchored to the RAW name and the predicate
    disagreed with itself — '_ALKYLAMINO_RE' is '^'-anchored, so 'tert-' pushed
    the alkyl root off the anchor.
    """
    assert is_complex_substituent("butylamino") is True
    assert is_complex_substituent("tert-butylamino") is True     # was False
    assert is_complex_substituent("methylsulfanyl") is True
    assert is_complex_substituent("tert-butylsulfanyl") is True
    # and the simple retained name is still simple
    assert is_complex_substituent("tert-butyl") is False


@pytest.mark.parametrize("name,expected", [
    ("tert-butyl", "2,6-di-tert-butyl"),
    ("sec-butyl", "2,6-di-sec-butyl"),
])
def test_multiplied_italicized_prefix_keeps_its_hyphen(name, expected):
    """P-16.3.3(b)/P-16.2.4.1(d) — BB '1,2-di-tert-butylbenzene' (PIN) is the witness."""
    assert format_substituent_prefix(name, [2, 6], 2) == expected


# ==========================================================================
# part B: the mononuclear pair (sites 1 + 2), end to end
# ==========================================================================

def test_mononuclear_multi_prefix_transform_survives_a_tert_butyl(ungated_namer):
    """Sites 1+2: the raw hyphen test corrupted EVERY prefix, not just the tert- one.

    ``_is_simple_prefix`` feeds an ``all(...)``, so classing 'tert-butyl' compound
    flipped the whole guard False and silently switched OFF the
    first-bare/rest-enclosed transform for every prefix in the name. BB 16286
    writes the directly analogous phosphane as
    '*tert*-butyldi(methyl)phosphane' (PIN): first group bare, each later group
    enclosed, multiplier OUTSIDE the marks. OPSIN-exact against the input.
    """
    assert (ungated_namer.name("CC(C)(C)[Si](C)(C)OCC1CO1")
            == "tert-butyldi(methyl)(oxiranylmethoxy)silane")


def test_mononuclear_multiplied_italicized_prefix_is_not_malformed(ungated_namer):
    """The multiplier leg, live and MALFORMED before this phase.

    ``CC(C)(C)[SiH](C)C(C)(C)C`` shipped 'ditert-butylmethylsilane' — 'di' fused
    straight onto 'tert-' with no boundary at all. (b)/(d) requires the hyphen:
    'di-tert-butyl'. OPSIN-exact (C(C)(C)(C)[SiH](C)C(C)(C)C).
    """
    assert (ungated_namer.name("CC(C)(C)[SiH](C)C(C)(C)C")
            == "di-tert-butylmethylsilane")


def test_the_multiplied_simple_prefix_carve_out_still_holds(ungated_namer):
    """PROTECT: the documented conservative branch of the mononuclear rule (when
    ANY prefix is multiplied, all are left bare) must not move."""
    assert ungated_namer.name("C(Br)(Cl)(Cl)F") == "bromodichlorofluoromethane"


def test_apply_mononuclear_enclosing_is_the_one_implementation():
    """Direct witness for sites 1+2 (the duplicated pair).

    A mutation run showed the end-to-end silane name is produced by the
    ORGANOMETALLIC path, not this one, so no namer-level assertion actually
    exercised ``apply_mononuclear_enclosing`` -- the very function the pair was
    collapsed into. This calls it directly, which is also the only way to show the
    damage the raw hyphen did: ``_is_simple_prefix`` feeds an ``all(...)``, so ONE
    italicized prefix turned the transform off for EVERY prefix.
    """
    from orthonym.assembly.composition_primitives import (
        apply_mononuclear_enclosing as encl,
    )
    # first cited BARE, second and further EACH enclosed (BB 16286 shape)
    assert encl(["tert-butyl", "methyl", "chloro"], True) == [
        "tert-butyl", "(methyl)", "(chloro)"]
    assert encl(["sec-butyl", "methyl"], True) == ["sec-butyl", "(methyl)"]
    #... and the carve-out stays narrow: a COMPOUND remainder is still compound,
    # so the transform correctly does not fire for it
    assert encl(["tert-butylsulfanyl", "methyl"], True) == [
        "tert-butylsulfanyl", "methyl"]
    # unchanged for prefixes that never had a hyphen
    assert encl(["bromo", "chloro", "fluoro"], True) == [
        "bromo", "(chloro)", "(fluoro)"]
    # and the rule is gated on a mononuclear parent with >= 2 prefixes
    assert encl(["tert-butyl", "methyl"], False) == ["tert-butyl", "methyl"]
    assert encl(["tert-butyl"], True) == ["tert-butyl"]


def test_alkoxy_prefix_fails_closed_on_a_revoked_contraction(monkeypatch):
    """Direct witness for the fail-closed leg.

    ``sec-butyl`` cannot reach ``_alkoxy_prefix`` today -- the substituent namer
    returns the locanted ``butan-2-yl``, which is the PIN -- so a mutation run
    showed the guard had no test that could fail. The guard still has to exist:
     revokes BOTH ``sec-butoxy`` and ``sec-butyloxy`` by name, so if any
    producer ever hands this function an italicized-led R that is not in the
    retained table, the only correct spelling is one this function cannot build.
    Refuse rather than invent a rejected contraction.
    """
    from rdkit import Chem
    from orthonym.assembly.handlers import hydroxylamine as H

    monkeypatch.setattr(H, "name_substituent_fragment", None, raising=False)
    mol = Chem.MolFromSmiles("CCC(C)ONC")
    o_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O"][0]
    r_c = [nb.GetIdx() for nb in mol.GetAtomWithIdx(o_idx).GetNeighbors()
           if nb.GetSymbol() == "C"][0]

    import orthonym.assembly.substituent_naming as SN
    monkeypatch.setattr(SN, "name_substituent_fragment",
                        lambda *a, **k: "sec-butyl")
    assert H._alkoxy_prefix(mol, o_idx, r_c) is None      # never 'sec-butyloxy'

    # a retained contraction is still produced, so the guard is not a blanket veto
    monkeypatch.setattr(SN, "name_substituent_fragment",
                        lambda *a, **k: "tert-butyl")
    assert H._alkoxy_prefix(mol, o_idx, r_c) == "tert-butoxy"


# ==========================================================================
# part C: site 3, the alkoxy prefix
# ==========================================================================

def test_tert_butyl_oxy_prefix_is_the_blue_book_contraction(ungated_namer):
    """: '(CH3)3C-O– *tert*-butoxy (preferred prefix) (no substitution)'.

    ``_alkoxy_prefix`` used a raw ``"-" in alkyl`` and emitted '(tert-butyl)oxy'.
    The Blue Book rejects both that and 'tert-butyloxy' by name (index:55662,
    :55671). OPSIN-exact: 'N-tert-butoxymethanamine' -> C(C)(C)(C)ONC.
    """
    assert ungated_namer.name("CC(C)(C)ONC") == "N-tert-butoxymethanamine"
    assert ungated_namer.name("CC(C)(C)ONCC") == "N-tert-butoxyethanamine"


@pytest.mark.parametrize("smiles,expected", [
    # PROTECT: the other five contractions must not move.
    ("CNOC", "N-methoxymethanamine"),
    ("CCONCC", "N-ethoxyethanamine"),
    ("CCCCONC", "N-butoxymethanamine"),
    # and a locanted R stays ENCLOSED — revokes 'sec-butoxy' and gives
    # '(butan-2-yl)oxy' as the PIN, which is what the substituent namer produces.
    ("CCC(C)ONC", "N-(butan-2-yl)oxymethanamine"),
    ("CC(C)ONC", "N-(propan-2-yl)oxymethanamine"),
])
def test_the_other_retained_alkoxy_contractions_are_unchanged(
        ungated_namer, smiles, expected):
    assert ungated_namer.name(smiles) == expected


def test_an_italicized_alkyl_with_no_retained_contraction_fails_closed():
    """ revokes 'sec-butoxy' AND 'sec-butyloxy' (index:55646).

    So an italicized-led R that is not in the retained table has no spelling this
    function is entitled to emit, and the carve-out must not be read as licence
    to invent one. Fail closed instead.
    """
    from orthonym.assembly.handlers.hydroxylamine import _CONTRACTED_ALKOXY
    assert _CONTRACTED_ALKOXY["tert-butyl"] == "tert-butoxy"
    assert "sec-butyl" not in _CONTRACTED_ALKOXY
    # the guard is the shared primitive, so it covers the whole class
    assert strip_italicized_structural_prefix("sec-butyl") == ("butyl", True)


# ==========================================================================
# part D: the copies the 3A tripwire could not see
# ==========================================================================

def test_organometallic_multiplied_ligand_is_not_malformed(ungated_namer):
    """``organometallics`` carried THREE multiplier-join sites and a raw ``-`` leg.

    ``:771``'s char-set test re-opened, in the same file, the defect its sibling
    ``_compound_ligand`` was fixed for: 'tert-butyl' matched ``-``, so two
    IDENTICAL SIMPLE ligands were routed away from the multiplied form and came
    back '(tert-butyl)(tert-butyl)zinc'. OPSIN-exact:
    C(C)(C)(C)[Zn]C(C)(C)C.
    """
    assert ungated_namer.name("CC(C)(C)[Zn]C(C)(C)C") == "di-tert-butylzinc"


@pytest.mark.parametrize("smiles,expected", [
    ("C[Zn]C", "dimethylzinc"),
    ("c1ccccc1[Hg]c1ccccc1", "diphenylmercury"),
    ("CC[Pb](CC)(CC)CC", "tetraethylplumbane"),
    ("CC[Sn](CC)(CC)CC", "tetraethylstannane"),
])
def test_simple_organometallic_names_are_unchanged(ungated_namer, smiles, expected):
    """PROTECT: dropping the raw ``-`` leg must not move a name that never had a
    hyphen in it."""
    assert ungated_namer.name(smiles) == expected


def test_alphabetization_copies_now_cover_sec_as_well():
    """``organometallics._alphabetize_simple_ligands`` open-coded
    ``name[5:] if name.startswith('tert-')`` — it handled 'tert-' and silently
    MISSED 'sec-', so 'sec-butyl' sorted at 's' instead of 'b'.
    """
    from orthonym.rules.organometallics import _alphabetize_simple_ligands
    got = _alphabetize_simple_ligands([(1, "sec-butyl"), (1, "methyl")])
    assert [n for _c, n in got] == ["sec-butyl", "methyl"]   # b before m
    got2 = _alphabetize_simple_ligands([(1, "tert-butyl"), (1, "methyl")])
    assert [n for _c, n in got2] == ["tert-butyl", "methyl"]


def test_general_engine_multiplier_and_enclosure_agree_on_an_italicized_prefix():
    """``general_engine._COMPLEX_PREFIX_RE`` includes a hyphen and drove BOTH the
    enclosure choice and the di-vs-bis choice, so one raw test got a 'tert-butyl'
    wrong twice: 'bis(tert-butyl)' where the Blue Book writes 'di-tert-butyl'.
    """
    from orthonym.assembly.general_engine import _is_complex_prefix, _mult_prefix
    assert _is_complex_prefix("tert-butyl") is False
    assert _is_complex_prefix("tert-butylsulfanyl") is True
    assert _is_complex_prefix("2-methylpropyl") is True
    assert _mult_prefix(2, "tert-butyl") == "di-tert-butyl"
    assert _mult_prefix(1, "tert-butyl") == "tert-butyl"
    assert _mult_prefix(2, "methyl") == "dimethyl"


@pytest.mark.parametrize("smiles,expected", [
    # PROTECT: sites whose carve-out was hand-written or missing but whose output
    # was already correct. Converting them to the primitive is drift-prevention,
    # so every one of these must stay byte-identical.
    ("CC(C)(C)B(O)O", "tert-butylboronic acid"),        # composer hard-coded list
    ("CCC(C)B(O)O", "(butan-2-yl)boronic acid"),
    ("CC(C)(C)c1ccc(cc1)C#N", "4-tert-butylbenzonitrile"),   # composer char-set
    ("CC(C)(C)C1CCC(CC1)O", "4-tert-butylcyclohexan-1-ol"),  # ring_substituents
    # MULTI-substituent variants: a mutation run showed the single-substituent
    # rows above do not reach the two char-set predicates at all (their
    # enclosing decision is only consulted once a second prefix is present), so
    # these are the rows that actually witness composer._is_compound_prefix and
    # ring_substituents._is_complex_prefix.
    ("CC(C)(C)c1cc(C)ccc1C#N", "2-tert-butyl-4-methylbenzonitrile"),
    ("CC(C)(C)c1cc(Cl)ccc1C#N", "2-tert-butyl-4-chlorobenzonitrile"),
    ("CC(C)(C)C1CC(C)CCC1O", "2-tert-butyl-4-methylcyclohexan-1-ol"),
    ("CC(C)(C)C1CC(C)C(O)CC1", "4-tert-butyl-2-methylcyclohexan-1-ol"),
    ("CC(C)(C)C1CC(C(C)(C)C)CCC1", "1,3-di-tert-butylcyclohexane"),
    ("CC(C)(C)c1ccccc1", "tert-butylbenzene"),
    ("CC(C)(C)c1cccc(C(C)(C)C)c1O", "2,6-di-tert-butylphenol"),
    ("CC(C)(C)P(=O)(O)O", "tert-butylphosphonic acid"),
    ("CC(C)(C)P(C)(=O)O", "tert-butyl(methyl)phosphinic acid"),
    # and the compound remainder still takes its marks
    ("CC(C)(C)Sc1ccccc1", "(tert-butylsulfanyl)benzene"),
])
def test_converted_sites_stay_byte_identical(ungated_namer, smiles, expected):
    assert ungated_namer.name(smiles) == expected


# ==========================================================================
# part E: the TRIPWIRE, rebuilt so it would have caught the survivors
# ==========================================================================
#
# The 3A tripwire searched for ONE syntactic shape. This one inverts the burden of
# proof: it finds every construct in the tree that could be a copy of the
# decision and requires each to be listed below WITH A REASON. A new copy in any
# shape fails the test, because a new line is simply not in the list. The
# regexes deliberately over-match (a plain `'-' in name` is usually innocent);
# over-matching is what makes the list, and therefore the reasoning, complete.

_TRIPWIRE_PATTERNS = {
    # `'-' in name`, `"-" not in name`
    'hyphen_in': re.compile(r"""['"]-['"]\s+(?:not\s+)?in\b"""),
    # any `'tert-'` / `"sec-"` literal, in ANY shape: startswith(x), startswith((
    # x, y)), `for skip in (x, y)`, a membership list, a slice constant
    'italic_literal': re.compile(r"""['"](?:tert|sec)-['"]"""),
    # `any(c in nm for c in '-0123456789')` - a character class holding a hyphen
    'charclass_iter': re.compile(r"""\bfor\s+\w+\s+in\s+['"][^'"]*-[^'"]*['"]"""),
}

# (path relative to the package root, exact stripped source line) -> why it is OK.
# "GUARDED" = the raw test survives but the carve-out is applied to the
# same expression via the shared primitive. "NOT THE DECISION" = the hyphen is
# not a compoundness test at all.
_TRIPWIRE_ALLOWLIST = {
    # ---- the primitive itself ----
    ("assembly/naming_utils.py",
     '_ITALICIZED_STRUCTURAL_PREFIXES = ("sec-", "tert-")'):
        "THE ONE definition of the carve-out.",
    ("assembly/naming_utils.py", 'return "-" in remainder'):
        "has_structural_hyphen's own body — decides on the STRIPPED remainder.",
    ("assembly/naming_utils.py", "if '-' in name:"):
        "needs_brackets, reached only AFTER the strip-and-recurse above it.",

    # ---- GUARDED: raw test kept, carve-out applied to the same expression ----
    ("assembly/composition_primitives.py",
     "or ('-' in t and not _ital_bare)):"):
        "GUARDED by italicized_prefix_is_bare on the same expression.",
    ("assembly/composer.py", "needs_parens = (any(c in r_name for c in '-,')"):
        "GUARDED — the next line ANDs in not italicized_prefix_is_bare(r_name).",
    ("assembly/composer.py",
     "or any(c in nm for c in '-()[]0123456789'))"):
        "GUARDED — italicized_prefix_is_bare early-returns False above it (j7: the "
        "line was reflowed when 'oxycarbonimidoyl' joined 'oxycarbonyl'). "
        "UNWITNESSED: a mutation run plus a provenance probe showed the enclosing "
        "_assemble_aromatic_benzonitrile is not reached for any tert-butyl "
        "benzonitrile tried (single- or multi-substituted; another handler names "
        "them), so reverting the guard changes no output either. Kept as "
        "drift-prevention, NOT as a verified fix.",
    ("rules/ring_substituents.py", "return any(c in nm for c in '-()[]0123456789')"):
        "GUARDED — italicized_prefix_is_bare early-returns False above it. "
        "UNWITNESSED for the same reason: decorated_ring_substituent_name is not "
        "reached for any tert-butyl-substituted ring tried.",
    ("assembly/handlers/hydroxylamine.py",
     'if any(ch.isdigit() for ch in alkyl) or "-" in alkyl or "(" in alkyl:'):
        "GUARDED — the italicized class is resolved (contraction) or refused above.",
    # j7 (TRIAGE g7 C20): three raw tests that DID decide for an italicized-led
    # name now call the shared primitive first (spellings OPSIN 2.9.0 exact,
    # '1,2-di-*tert*-butylbenzene (PIN)' the Blue Book).
    ("assembly/substituent_naming.py",
     "if (any(ch.isdigit() for ch in tok) or '-' in tok or ' ' in tok"):
        "GUARDED — _fg_enclose returns an italicized_prefix_is_bare token bare above it.",
    ("rules/phosphorus.py",
     'and (any(ch in token for ch in "()[]-, 0123456789")'):
        "GUARDED — ANDed with not italicized_prefix_is_bare(token) on the line above; "
        "the simple branch joins with multiplier_needs_hyphen: 'tri-tert-butyl "
        "phosphite' (was 'tris(tert-butyl) phosphite').",
    ("rules/salts.py", "if '-' in name:"):
        "GUARDED — _ion_needs_enclosing_multiplier returns False for an "
        "italicized_prefix_is_bare ion word above it, and _apply_stoichiometric_prefix "
        "adds the hyphen: 'magnesium di-tert-butoxide' (was 'bis(tert-butoxide)').",

    # ---- carve-out UNREACHABLE: the producer cannot emit an italicized-led name ----
    ("rules/benzene.py",
     "_complex = any(ch.isdigit() for ch in _amido) or '-' in _amido"):
        "acyl_amido_prefix_from_branch yields an -amido/-anilide prefix built from "
        "an ACID stem; no acid stem is spelled with a leading tert-/sec-, so the "
        "carve-out cannot fire. Replacing the raw test with has_structural_hyphen "
        "alone would UNDER-enclose (this site's only compound signals are digits "
        "and hyphen), so it is left broad on purpose.",
    ("rules/benzene.py",
     "_complex = any(ch.isdigit() for ch in _imid) or '-' in _imid"):
        "Same, imidamido branch.",
    ("assembly/composer.py", "or '-' in _amido):"):
        "Same, acid_name_to_amido_prefix.",
    ("rules/ring_assemblies.py",
     "if '-' in sub_yl_name and not sub_yl_name.startswith('('):"):
        "get_ring_substituent_name yields a RING-yl name ('pyridin-2-yl'); a ring "
        "prefix is never spelled with a leading tert-/sec-.",
    ("data/sugar_names.py", 'if n > 1 and any(ch in word for ch in "- ("):'):
        "Operates on a sugar ACYL ESTER WORD, never an alkyl prefix.",
    ("data/sugar_names.py", 'if any(ch in word for ch in " -("):'):
        "Same, name_sugar_ester.",
    ("data/sugar_names.py", 'if any(ch.isdigit() for ch in base) or "-" in base:'):
        "Operates on a catalogued sugar BASE name (glucopyranose, 2-acetamido-2-deoxy-"
        "...), never an alkyl prefix; a decorated base declines to the '[...]oxy' form "
        "(j7 review).",

    # ---- NOT THE DECISION ----
    ("assembly/substituent_naming.py",
     "base_name = name.split('-')[-1] if '-' in name else name"):
        "NOT THE DECISION — strips a locant block for a hetero-ring set lookup.",
    ("data/sugar_names.py", 'if (anomer or config) and "-" in base_name:'):
        "NOT THE DECISION — finds a prefix block to insert the a/b-D descriptor.",
    ("decomposition/engine.py", 'has_hyphens = "-" in name'):
        "NOT THE DECISION — name-DETAIL heuristic (is a big molecule's name "
        "specific enough to skip decomposition).",
    ("rules/fusion_descriptors.py", "if '-' in inner:"):
        "NOT THE DECISION — splits a fusion descriptor '[3,2-b]' into its child "
        "locants and edge letter for the citation sort key (j7 review).",
    ("rules/oligosaccharides.py", 'if "-" in base:'):
        "NOT THE DECISION — splits a decorated sugar base into its prefix block and "
        "stem so the configurational descriptor goes before the stem (j7 review).",
    ("rules/polycyclics.py",
     "'core_name': heterocycle_name.split('-')[-1] if '-' in heterocycle_name "
     "else heterocycle_name,"):
        "NOT THE DECISION — splits off a locant block; also in "
        "identify_fused_system, which has no callers.",
}


def _scan_for_hyphen_predicate_copies(root: Path):
    """Every line in the package matching any tripwire pattern."""
    found = []
    for py in sorted(root.rglob("*.py")):
        for lineno, line in enumerate(py.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith(">>>"):
                continue      # a comment may quote the retired open-code
            if "``" in line:
                continue      # rst inline code inside a docstring, not code
            for pattern in _TRIPWIRE_PATTERNS.values():
                if pattern.search(line):
                    found.append((str(py.relative_to(root)), stripped, lineno))
                    break
    return found


def test_no_unreviewed_copy_of_the_italicized_prefix_decision():
    """Structural tripwire, rebuilt: every copy must be listed WITH A REASON.

    The 3A version searched for ``startswith(("tert-`` and therefore could not see
    any of the survivors — three raw ``'-' in t`` tests, a single-argument
    ``startswith('tert-')``, two ``for x in ('tert-', 'sec-')`` loops, three
    character-set tests, and a second hand-written carve-out list. An allowlist
    cannot be defeated that way: a new copy in ANY shape is simply not in the
    list.
    """
    root = Path(orthonym.__file__).parent
    offenders = [
        f"{path}:{lineno}: {line}"
        for path, line, lineno in _scan_for_hyphen_predicate_copies(root)
        if (path, line) not in _TRIPWIRE_ALLOWLIST
    ]
    assert not offenders, (
        "Unreviewed hyphen/italicized-prefix test(s). Either route the decision "
        "through naming_utils.strip_italicized_structural_prefix / "
        "has_structural_hyphen / italicized_prefix_is_bare, or add the line to "
        "_TRIPWIRE_ALLOWLIST with the reason it is safe:\n" + "\n".join(offenders)
    )


def test_the_tripwire_allowlist_has_no_stale_entries():
    """An allowlist that outlives its line silently loses its teeth: the entry
    would keep excusing a line that has since been reworded into a NEW copy."""
    root = Path(orthonym.__file__).parent
    live = {(path, line)
            for path, line, _lineno in _scan_for_hyphen_predicate_copies(root)}
    stale = sorted(k for k in _TRIPWIRE_ALLOWLIST if k not in live)
    assert not stale, f"stale _TRIPWIRE_ALLOWLIST entries: {stale}"


def test_the_tripwire_actually_fires(tmp_path):
    """MUTATION TEST OF THE TRIPWIRE ITSELF.

    A prior tripwire in this repo could never fire. This asserts the scanner sees
    each of the shapes that got past 3A — including the two loop forms and the
    character-set form its regex was blind to — and that a listed line is excused.
    """
    pkg = tmp_path / "orthonym"
    pkg.mkdir()
    (pkg / "mod.py").write_text(
        "def f(t, nm, p):\n"
        "    if '-' in t:\n"                                  # raw hyphen
        "        return 1\n"
        "    if p.startswith('tert-'):\n"                      # single-arg
        "        return 2\n"
        "    if p.startswith(('tert-', 'sec-')):\n"            # tuple form (3A)
        "        return 3\n"
        "    for skip in ('tert-', 'sec-'):\n"                 # loop form
        "        pass\n"
        "    return any(c in nm for c in '-()[]')\n"           # char class
    )
    hits = _scan_for_hyphen_predicate_copies(pkg)
    lines = [line for _path, line, _lineno in hits]
    assert "if '-' in t:" in lines
    assert "if p.startswith('tert-'):" in lines
    assert "if p.startswith(('tert-', 'sec-')):" in lines
    assert "for skip in ('tert-', 'sec-'):" in lines
    assert "return any(c in nm for c in '-()[]')" in lines
    assert len(hits) == 5, hits


