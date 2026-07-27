"""v29 Phase 3A — the organyl prefix guard routes to the shared chokepoint.

``rules/substituent_purity`` is the shared "is this an organyl, and what is its
prefix name?" guard used by the parent-hydride and oxoacid namers.  It used to
wrap ``rules.phosphorus._characterize_substituent``, which follows ONLY the
carbon skeleton and therefore miscounts a ring or a branch as a linear chain
(benzyl -> 'heptyl', cyclohexyl -> 'hexyl', propan-2-yl -> 'propyl').  To keep
those wrong names from shipping, the guard REFUSED every ring-bearing, branched,
long or unsaturated organyl.  The guard was right; the walker under it was the
defect.

This module pins the routing to ``assembly.substituent_enumerator.name_substituent``
-- the audited 5-tier chokepoint that every general-engine locus already uses --
and the two Blue Book rules that make the routing correct:

* **P-29.6.1** (BB ``BlueBookV2.md:16272``): "The traditional prefixes benzyl,
  benzylidene, benzylidyne are retained preferred prefixes, but are not to be
  substituted".  Verbatim PIN pair (``:16280``)::

      2-(4-bromobenzyl)pyridine        2-[(4-bromophenyl)methyl]pyridine (PIN)

  so an UNsubstituted benzyl keeps the retained prefix, and a substituted one
  must become ``(...phenyl)methyl`` in square brackets.
* **P-67.1.1.2 / BB L36051-L36066** the ``-onic``/``-inic`` oxoacid stems, with
  ``methyl(phenyl)arsinic acid`` (PIN) verbatim for the two-organyl form.

The class this phase opens is deliberately bounded to fragments built ONLY from
carbon, hydrogen and halogen -- see
``test_heteroatom_organyl_is_still_refused`` for the P-41 seniority reason a
heteroatom-bearing organyl must keep failing closed.
"""

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from orthonym.perception.functional_groups import detect_functional_groups
from orthonym.rules.substituent_purity import (
    is_simple_unbranched_organyl, organyl_prefix_name,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _hub_and_start(smiles, hub_symbol='P'):
    """Return ``(mol, start_idx, hub_idx)`` for ``R-<hub>`` with ONE C on the hub."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    hubs = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == hub_symbol]
    assert len(hubs) == 1, smiles
    hub = hubs[0]
    cs = [n.GetIdx() for n in mol.GetAtomWithIdx(hub).GetNeighbors()
          if n.GetSymbol() == 'C']
    assert len(cs) == 1, smiles
    return mol, cs[0], hub


@pytest.fixture
def ungated_namer(monkeypatch):
    """A namer with the SELF-01 OPSIN validity gate explicitly DISABLED.

    The gate fails OPEN when no JRE is present, so every safety property in this
    module is asserted in the mode where nothing downstream can rescue a wrong
    producer output.  This is the mode in which the sibling ``heptyl`` bugs were
    shipping wrong structures before Phase B.
    """
    import orthonym.namer as _namer
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", True, raising=False)
    return Orthonym()


# --------------------------------------------------------------------------
# 1. The primitive: previously-refused organyls are now NAMED
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    # ring-bearing (the docstring's case (c): benzyl was counted as 'heptyl')
    ("C1=CC=C(C=C1)CP(=O)(O)O",  "benzyl"),
    ("C1CCCCC1P(=O)(O)O",        "cyclohexyl"),
    ("C1CCCCC1CP(=O)(O)O",       "cyclohexylmethyl"),
    ("c1ccccc1CCP(=O)(O)O",      "2-phenylethyl"),
    ("c1ccccc1C(C)P(=O)(O)O",    "1-phenylethyl"),
    # internal attachment (case (b): propan-2-yl was mislabelled 'propyl')
    ("CC(C)P(=O)(O)O",           "propan-2-yl"),
    ("CCC(C)P(=O)(O)O",          "butan-2-yl"),
    ("CC(C)(C)P(=O)(O)O",        "tert-butyl"),
    # unsaturation (the walker refused every non-single bond)
    ("C=CP(=O)(O)O",             "ethenyl"),
    # a chain longer than the private walker's table
    ("CCCCCCCCCCCCP(=O)(O)O",    "dodecyl"),
    # halogen: prefix-only per P-59 Table 28, so it cannot demand a suffix
    ("BrCP(=O)(O)O",             "bromomethyl"),
    # P-29.6.1: a SUBSTITUTED benzyl must lose the retained prefix
    ("Brc1ccc(CP(=O)(O)O)cc1",   "(4-bromophenyl)methyl"),
    ("Cc1ccc(CP(=O)(O)O)cc1",    "(4-methylphenyl)methyl"),
])
def test_previously_refused_organyl_is_now_named(smiles, expected):
    mol, start, hub = _hub_and_start(smiles)
    assert organyl_prefix_name(mol, start, hub) == expected


# --------------------------------------------------------------------------
# 2. Byte-identity on every fragment the private walker already accepted
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("CP(=O)(O)O",       "methyl"),
    ("CCP(=O)(O)O",      "ethyl"),
    ("CCCP(=O)(O)O",     "propyl"),
    ("CCCCP(=O)(O)O",    "butyl"),
    ("CCCCCP(=O)(O)O",   "pentyl"),
    ("c1ccccc1P(=O)(O)O", "phenyl"),
])
def test_byte_identical_on_the_classes_the_old_walker_accepted(smiles, expected):
    """The routing must not move a name that was already correct.

    Every prefix the private walker accepted (unbranched alkyl attached at a
    terminus, phenyl) must come back byte-identical from the chokepoint, or the
    widening would be a silent re-spelling of shipped PINs.
    """
    mol, start, hub = _hub_and_start(smiles)
    assert organyl_prefix_name(mol, start, hub) == expected


def test_naphthyl_is_upgraded_to_the_pin_locanted_form():
    """The one accepted class whose spelling CHANGES -- and it was wrong before.

    ``_characterize_substituent`` returned the bare ``naphthyl`` for any
    10-carbon aromatic system.  ``naphthyl`` is not a preferred prefix and it is
    positionally AMBIGUOUS: P-25.3.1 numbers the fused system, so C10H7- is
    ``naphthalen-1-yl`` or ``naphthalen-2-yl``.  The chokepoint spells the
    locant, which is the PIN.
    """
    mol, start, hub = _hub_and_start("c1ccc2ccccc2c1P(=O)(O)O")
    assert organyl_prefix_name(mol, start, hub) == "naphthalen-1-yl"


# --------------------------------------------------------------------------
# 3. What must STILL fail closed
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles", [
    "OCCP(=O)(O)O",         # 2-hydroxyethyl
    "OC(=O)CP(=O)(O)O",     # carboxymethyl
    "NCCP(=O)(O)O",         # 2-aminoethyl
    "COCP(=O)(O)O",         # methoxymethyl
])
def test_heteroatom_organyl_is_still_refused(smiles):
    """A heteroatom in the organyl can carry a SUFFIXABLE characteristic group.

    P-41 makes the principal characteristic group -- not the hub -- decide the
    parent, so a heteroatom-bearing organyl is not freely a prefix:

    * ``HO-CH2CH2-NH-NH2``: the alcohol is senior to the amine class, so the PIN
      is ``2-hydrazinylethan-1-ol``, NOT ``(2-hydroxyethyl)hydrazine``.
    * ``HOOC-CH2-P(=O)(OH)2``: within P-41 class 7 a carboxylic acid outranks a
      phosphonic acid, so the carboxy group takes the suffix -- NOT
      ``(carboxymethyl)phosphonic acid``.

    Deciding that per family needs a seniority comparison this primitive does
    not have, so the whole heteroatom class fails closed here.  Hydrocarbon and
    halogen fragments carry NO suffixable characteristic group (P-59 Table 28
    lists the halogens among the groups cited only as prefixes), which is
    exactly why the widened class is bounded to them.
    """
    mol, start, hub = _hub_and_start(smiles)
    assert organyl_prefix_name(mol, start, hub) is None


@pytest.mark.parametrize("smiles", [
    "[CH2-]P(=O)(O)O",      # carbanion: an ion is a SENIOR class (P-41)
])
def test_charged_organyl_is_refused(smiles):
    mol, start, hub = _hub_and_start(smiles)
    assert organyl_prefix_name(mol, start, hub) is None


def test_exclude_idx_must_be_the_attachment_neighbour():
    """``start_idx`` is fragment-side and ``exclude_idx`` is its parent-side
    neighbour at every call site.  A call that violates the convention (a
    parent-side ``start_idx``, or a non-adjacent hub) must fail closed instead of
    naming some other subtree.
    """
    mol, start, hub = _hub_and_start("C1=CC=C(C=C1)CP(=O)(O)O")
    far = [a.GetIdx() for a in mol.GetAtoms()
           if a.GetSymbol() == 'O' and a.GetIdx() != hub]
    assert organyl_prefix_name(mol, start, far[0]) is None   # not adjacent
    assert organyl_prefix_name(mol, hub, start) is None      # hub is not an organyl


# --------------------------------------------------------------------------
# 4. The narrow predicate stays narrow (routing decisions must not widen)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,narrow", [
    ("CP(=O)(O)O",              True),    # methyl
    ("CCCP(=O)(O)O",            True),    # propyl
    ("c1ccccc1P(=O)(O)O",       True),    # phenyl
    ("C1=CC=C(C=C1)CP(=O)(O)O", False),   # benzyl -- ring
    ("CC(C)P(=O)(O)O",          False),   # propan-2-yl -- internal attachment
    ("C=CP(=O)(O)O",            False),   # ethenyl -- non-single bond
    ("OCCP(=O)(O)O",            False),   # heteroatom
])
def test_is_simple_unbranched_organyl_keeps_the_old_narrow_semantics(smiles, narrow):
    """``assembly/substituent_naming._group14_neighbour_prefix`` uses the guard as
    a ROUTING TEST for the contracted alkoxy prefix producer, not as a namer.
    Widening a routing test changes which producer claims a molecule, so that
    site keeps the narrow semantics under this explicitly narrow name.
    """
    mol, start, hub = _hub_and_start(smiles)
    assert is_simple_unbranched_organyl(mol, start, hub) is narrow


# --------------------------------------------------------------------------
# 5. Acceptance: the user-reported molecules, end to end, WITHOUT the gate
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("C1=CC=C(C=C1)CP(=O)(O)O",   "benzylphosphonic acid"),
    ("C1=CC=C(C=C1)C[As](=O)(O)O", "benzylarsonic acid"),
    ("C1=CC=C(C=C1)C[Sb](=O)(O)O", "benzylstibonic acid"),
])
def test_benzyl_oxoacid_acceptance(ungated_namer, smiles, expected):
    """P-29.6.1 retained preferred prefix ``benzyl`` + the P-67 oxoacid stems
    (BB L36051-L36054).  ``2-benzylpyridine (PIN)`` (BB ``:16280``) is the
    verbatim precedent for citing the unsubstituted benzyl bare, with no
    enclosing marks.
    """
    assert ungated_namer.name(smiles) == expected


@pytest.mark.parametrize("smiles,fg,producer,expected", [
    ("C1=CC=C(C=C1)CP(C)(=O)O",  "phosphinic_acid", "name_phosphinic_acid",
     "benzyl(methyl)phosphinic acid"),
    ("C1=CC=C(C=C1)C[As](C)(=O)O", "arsinic_acid", "name_arsinic_acid",
     "benzyl(methyl)arsinic acid"),
    ("C1CCCCC1P(C)(=O)O",        "phosphinic_acid", "name_phosphinic_acid",
     "cyclohexyl(methyl)phosphinic acid"),
])
def test_inic_acid_two_organyl_form(smiles, fg, producer, expected):
    """P-16.5.1.3 mononuclear enclosing marks, exactly as BB L36066 writes
    ``methyl(phenyl)arsinic acid`` (PIN): first cited group bare, each
    subsequent one enclosed.
    """
    from orthonym.rules import phosphorus
    mol = Chem.MolFromSmiles(smiles)
    matches = detect_functional_groups(mol).get(fg, [])
    assert matches, f"{smiles} should perceive {fg}"
    assert getattr(phosphorus, producer)(mol, matches[0]) == expected


# --------------------------------------------------------------------------
# 6. The substitution restriction must survive the routing (P-29.6.1)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,forbidden", [
    ("Brc1ccc(CP(=O)(O)O)cc1", "bromobenzyl"),
    ("Cc1ccc(CP(=O)(O)O)cc1",  "methylbenzyl"),
])
def test_substituted_benzyl_never_keeps_the_retained_prefix(
    ungated_namer, smiles, forbidden,
):
    """BB ``:16280`` verbatim: ``2-(4-bromobenzyl)pyridine`` is general
    nomenclature; ``2-[(4-bromophenyl)methyl]pyridine`` is the PIN.  A
    substituted benzyl must therefore be spelled ``(...phenyl)methyl``, and the
    compound prefix takes enclosing marks that escalate to SQUARE brackets
    because the name already contains parentheses (P-16.5.1.1 / P-16.5.4.1).
    """
    name = ungated_namer.name(smiles)
    assert "(4-" in name and ")methyl" in name, name
    assert forbidden not in name, name
    assert name.endswith("phosphonic acid"), name


# --------------------------------------------------------------------------
# 7. The old fabricated-chain bug must stay dead, with no gate to rescue it
# --------------------------------------------------------------------------

@pytest.mark.parametrize("smiles", [
    "C1=CC=C(C=C1)CP(=O)(O)O",
    "C1=CC=C(C=C1)C[As](=O)(O)O",
    "C1=CC=C(C=C1)C[Sb](=O)(O)O",
    "C1=CC=C(C=C1)CP(C)(=O)O",
    "C1=CC=C(C=C1)C[As](C)(=O)O",
    "C1CCCCC1P(C)(=O)O",
    "C1CCCCC1[As](C)(=O)O",
    "C1CCCCC1P(=O)(O)O",
])
def test_no_fabricated_linear_chain_in_any_ring_bearing_organyl(
    ungated_namer, smiles,
):
    """REGRESSION guard, kept from Phase B and strengthened.

    ``_characterize_substituent`` counted the carbons of a non-aromatic subtree
    as a LINEAR chain: benzyl (7 C, six in a ring) came back as 'heptyl' and
    cyclohexyl as 'hexyl' -- a WRONG CONSTITUTION.  Phase B stopped that by
    REFUSING; this phase stops it by naming the ring correctly.  Either way the
    fabricated token must never appear, and the assertion runs with the OPSIN
    gate disabled because that gate fails open with no JRE.
    """
    name = ungated_namer.name(smiles)
    assert "heptyl" not in name, name
    assert "hexyl" not in name or "cyclohexyl" in name, name
    assert "methylbenzene" not in name and "toluene" not in name, name


# --------------------------------------------------------------------------
# 8. Provenance: the name comes from the routed handler, via the chokepoint
# --------------------------------------------------------------------------

def test_provenance_benzylphosphonic_comes_from_the_routed_handler(monkeypatch):
    """A right answer from an unintended path is not the fix.

    Two independent witnesses: the ``phosphonic_acid`` producer must be the one
    that returns the string, and the routed guard must be CALLED and must return
    the ``benzyl`` token that ends up in the name.
    """
    import orthonym.namer as _namer
    import orthonym.rules.phosphorus as ph
    import orthonym.rules.substituent_purity as sp
    monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", True, raising=False)

    guard_calls = []
    real_guard = sp.organyl_prefix_name

    def spy_guard(mol, start_idx, exclude_idx):
        out = real_guard(mol, start_idx, exclude_idx)
        guard_calls.append(out)
        return out

    producer_calls = []
    real_producer = ph.name_phosphonic_acid

    def spy_producer(mol, match_atoms):
        out = real_producer(mol, match_atoms)
        producer_calls.append(out)
        return out

    monkeypatch.setattr(sp, "organyl_prefix_name", spy_guard)
    monkeypatch.setattr(ph, "name_phosphonic_acid", spy_producer)

    name = Orthonym().name("C1=CC=C(C=C1)CP(=O)(O)O")
    assert name == "benzylphosphonic acid"
    assert "benzylphosphonic acid" in producer_calls, producer_calls
    assert "benzyl" in guard_calls, guard_calls


# --------------------------------------------------------------------------
# 9. P-16.3.3(b)/P-16.2.4.1(d) / P-29.6.1: a retained ITALICIZED-PREFIX name is cited BARE
#
# The routing above made `tert-butyl` reachable as a prefix for the first time
# and exposed an over-enclosure defect: '(tert-butyl)arsonic acid'.  The Blue
# Book cites the retained preferred prefix bare, hyphen and all:
#
#   BB 16286   ***tert*-butyldi(methyl)phosphane  (PIN)      <- no marks
#   BB 16270   P-29.6.1 "Retained prefixes that ARE preferred prefixes"
#   BB 16282   "The retained name '*tert*-butyl' has never been recommended for
#              further substitution ... Acceptable locants have never been
#              adopted for this name."   -> its hyphen is not a compound boundary
#   P-16.3.3(b)/P-16.2.4.1(d)   'di-tert-butyl' (NOT 'bis(tert-butyl)'), 'N-tert-butyl'
#              (cited bare per BB 3465)
#
# Root cause: the carve-out was open-coded in three divergent places and MISSING
# from `needs_brackets`, whose blanket `'-' in name` then won inside the union
# predicate `enclose_if_compound`.  It now lives in exactly ONE primitive.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["tert-butyl", "sec-butyl"])
def test_retained_italicized_prefix_is_not_compound(name):
    """P-16.3.3(b)/P-16.2.4.1(d): the leading italicized prefix's hyphen does not make the name
    compound, so no enclosing marks and the SIMPLE di/tri multiplier."""
    from orthonym.assembly.naming_utils import (
        enclose_if_compound, get_multiplier_prefix, has_structural_hyphen,
        is_complex_substituent, needs_brackets,
    )
    assert has_structural_hyphen(name) is False
    assert needs_brackets(name) is False
    assert is_complex_substituent(name) is False
    assert enclose_if_compound(name) == name      # cited BARE (BB 16286)
    # v29 P3-FIX Item 4: the P-16.2.4.1(d) hyphen is now part of the multiplied
    # TOKEN and is produced by this one primitive, so the multiplier comes back
    # as `di-`. The assertion's point is unchanged: the SIMPLE `di` and not the
    # derived `bis` (P-16.3.2(a), BlueBookV2.md:7033).
    assert get_multiplier_prefix(2, name) == "di-"
    assert not get_multiplier_prefix(2, name).startswith("bis")


@pytest.mark.parametrize("name,expected", [
    # P-16.3.4: multiplied retained prefix keeps the hyphen boundary and the
    # SIMPLE multiplier -- BB '1,2-di-tert-butylbenzene' (PIN) is the witness.
    ("tert-butyl", "2,6-di-tert-butyl"),
    ("sec-butyl", "2,6-di-sec-butyl"),
])
def test_di_tert_butyl_multiplication(name, expected):
    from orthonym.assembly.naming_utils import format_substituent_prefix
    assert format_substituent_prefix(name, [2, 6], 2) == expected


@pytest.mark.parametrize("name", [
    "tert-butylsulfanyl",        # compound chalcogen prefix (P-16.3.3)
    "tert-butyl-dimethylsilyl",  # a SECOND, structural hyphen
    "2-tert-butyl",              # a locant
])
def test_the_carve_out_does_not_swallow_a_genuinely_compound_name(name):
    """The carve-out must be about the PREFIX only, never the remainder.

    Stripping 'tert-' and re-deciding on the remainder is what keeps this
    narrow: 'tert-butyl' -> 'butyl' (simple), but 'tert-butylsulfanyl' ->
    'butylsulfanyl', still compound under the P-16.3.3 chalcogen rule and still
    enclosed, exactly as '(methylsulfanyl)' is.
    """
    from orthonym.assembly.naming_utils import needs_brackets
    assert needs_brackets(name) is True


def test_the_italicized_prefix_carve_out_lives_in_exactly_one_place():
    """Structural tripwire: no open-coded DETECTION of an italicized prefix.

    The defect was caused by divergent copies -- is_complex_substituent's inline
    strip, format_substituent_prefix's `startswith(("tert-","sec-"))`,
    organometallics' raw `'-' in name`, the multiplier-hyphen rule's own
    `startswith`, and needs_brackets' missing case.  Detection now lives only in
    `strip_italicized_structural_prefix`; every consumer must call that (or
    `has_structural_hyphen`).  Downstream rules may still DIFFER -- withholding
    marks vs. inserting a hyphen -- they just may not re-derive the test.

    This test found the fifth copy when it was first written.

    SUPERSEDED (v29 P3B): it searches for ONE syntactic shape and was therefore
    blind to every copy a later review found — three raw ``'-' in t`` tests, a
    single-argument ``startswith('tert-')``, two ``for x in ('tert-', 'sec-')``
    loops, three character-set tests and a second hand-written carve-out list. The
    complete check is the allowlist tripwire
    ``test_v29_p3b_shared_predicates.test_no_unreviewed_copy_of_the_italicized_prefix_decision``;
    this one is kept as a cheap fast-fail on the shape it does cover.
    """
    import re
    from pathlib import Path
    import orthonym
    root = Path(orthonym.__file__).parent
    offenders = []
    for py in root.rglob("*.py"):
        text = py.read_text()
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue          # a comment may quote the retired open-code
            if re.search(r'startswith\(\s*\(\s*["\']tert-', line):
                offenders.append(f"{py.relative_to(root)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "open-coded tert-/sec- carve-out found; call "
        "naming_utils.strip_italicized_structural_prefix instead:\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize("smiles,expected", [
    # P-67 oxoacid stems (BB L36051-L36054) + the bare retained prefix.
    ("CC(C)(C)P(=O)(O)O",    "tert-butylphosphonic acid"),
    ("CC(C)(C)[As](=O)(O)O", "tert-butylarsonic acid"),
    ("CC(C)(C)[Sb](=O)(O)O", "tert-butylstibonic acid"),
    # P-16.5.1.3: FIRST cited group bare (and here it needs no marks anyway),
    # each subsequent one enclosed -- BB L36066 'methyl(phenyl)arsinic acid'.
    ("CC(C)(C)P(C)(=O)O",    "tert-butyl(methyl)phosphinic acid"),
])
def test_tert_butyl_oxoacid_cites_the_retained_prefix_bare(ungated_namer, smiles, expected):
    """BB 16286 '*tert*-butyldi(methyl)phosphane' (PIN) cites tert-butyl BARE;
    the prefix is cited BARE (BB 3465 / BB 16286). (NB: 'N-tert-butyl' is NOT a Blue Book string -- verified absent in 4 encodings against 30 hits for `*tert*-butyl`; the bare form follows from BB 3465 and the rules are P-16.3.3(b)/BB 7070 + P-16.2.4.1(d)/BB 6964, not P-16.3.4.)  All four names are
    OPSIN-exact against the input structure."""
    assert ungated_namer.name(smiles) == expected


def test_organometallic_ligand_join_cites_the_retained_prefix_bare(ungated_namer):
    """The FOURTH copy of the compound predicate, found while fixing the third.

    ``organometallics._ligand_token`` tested a raw ``'-' in name``, so the
    Group-14 hydride-parent join emitted the non-PIN
    '(tert-butyl)di(methyl)(oxiranylmethoxy)silane'.  BB 16286 writes the
    directly analogous phosphane as '*tert*-butyldi(methyl)phosphane' (PIN) --
    first group bare, each later group enclosed with its multiplier OUTSIDE the
    marks.  OPSIN-exact: C(C)(C)(C)[Si](OCC1OC1)(C)C.
    """
    assert (ungated_namer.name("CC(C)(C)[Si](C)(C)OCC1CO1")
            == "tert-butyldi(methyl)(oxiranylmethoxy)silane")


@pytest.mark.parametrize("smiles,expected", [
    # Protect rows: names that were ALREADY correct must not move.
    ("CC(C)(C)c1ccccc1",              "tert-butylbenzene"),
    ("CC(C)(C)c1cccc(C(C)(C)C)c1O",   "2,6-di-tert-butylphenol"),
])
def test_already_correct_tert_butyl_names_are_unchanged(ungated_namer, smiles, expected):
    """`2,6-di-tert-butylphenol` is the P-16.3.3(b)/P-16.2.4.1(d) multiplication witness (BB
    '1,2-di-tert-butylbenzene' (PIN)); both were correct before the carve-out
    moved and must stay byte-identical."""
    assert ungated_namer.name(smiles) == expected


# ==========================================================================
# FAMILY 2 -- rules/catenated_hydrides.name_heterochalcogen_aba (1 site)
#
# The a[ba]n pure-chalcogen parent hydride (P-68.4.2.1 / P-21.2.3.1):
# HS-O-SH -> `dithioxane`, and a terminal chalcogen may bear ONE organyl.
# The organyl guard was the narrow walker, so every branched / unsaturated /
# long terminal organyl was refused (`unknown organic compound`).
#
# Widening the guard also puts prefixes carrying LOCANTS and HYPHENS through
# this handler's own prefix-composition block for the first time.  That block
# open-coded `sorted(counts)` + bare concatenation, which is correct only for
# the letters-only simple prefixes the narrow walker could return.  It now uses
# the shared primitives, so:
#
#   BB 25719   `1,4-di(propan-2-yl)cyclohexane` (PIN)  -> the compound prefix is
#              enclosed and the SIMPLE multiplier sits OUTSIDE the marks
#   BB 16286   `*tert*-butyldi(methyl)phosphane` (PIN) -> a retained italicized
#              prefix is cited BARE, and P-16.3.4 keeps its hyphen under a
#              multiplier (`di-tert-butyl`, never `ditert-butyl`)
#   P-14.5.2   alphanumerical order ignores the italicized prefix and the marks
# ==========================================================================

def _aba(smiles):
    from orthonym.rules.catenated_hydrides import name_heterochalcogen_aba
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return name_heterochalcogen_aba(mol)


@pytest.mark.parametrize("smiles,expected", [
    # branched: the narrow walker refused an INTERNAL attachment outright
    ("CC(C)SOS",             "(propan-2-yl)dithioxane"),
    ("CC(C)(C)SOS",          "tert-butyldithioxane"),
    # unsaturated: the narrow walker refused every non-single bond
    ("C=CSOS",               "ethenyldithioxane"),
    # longer than the private walker's chain table
    ("CCCCCCCCCCCCSOS",      "dodecyldithioxane"),
    # two identical compound organyls -> di OUTSIDE the marks (BB 25719)
    ("CC(C)SOSC(C)C",        "di(propan-2-yl)dithioxane"),
    # P-16.3.3(b)/P-16.2.4.1(d): the italicized prefix keeps its hyphen under the multiplier
    ("CC(C)(C)SOSC(C)(C)C",  "di-tert-butyldithioxane"),
    # two DIFFERENT organyls: P-14.5.2 order ('methyl' < 'propanyl'), and
    # P-16.5.1.3 encloses the second cited group
    ("CSOSC(C)C",            "methyl(propan-2-yl)dithioxane"),
])
def test_f2_catenated_aba_previously_refused_organyl_is_now_named(smiles, expected):
    assert _aba(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    ("SOS",         "dithioxane"),
    ("CSOS",        "methyldithioxane"),
    ("CSOSC",       "dimethyldithioxane"),
    ("CCSOS",       "ethyldithioxane"),
    ("CSOSCC",      "ethyl(methyl)dithioxane"),
    ("SOSOS",       "trithioxane"),
    ("CSOSOS",      "methyltrithioxane"),
    ("CSOSOSC",     "dimethyltrithioxane"),
])
def test_f2_catenated_aba_existing_names_are_byte_identical(smiles, expected):
    """PROTECT rows.  Every name this handler already emitted must be unchanged:
    the widening must add emissions, never re-spell one.  ``ethyl(methyl)``
    also pins that the switch from raw string order to P-14.5.2 alphanumerical
    order is a no-op on the letters-only class the narrow walker allowed."""
    assert _aba(smiles) == expected


@pytest.mark.parametrize("smiles,why", [
    ("c1ccccc1SOS",   "ring: the handler's own whole-molecule ring guard"),
    ("C1CCCCC1SOS",   "ring: the handler's own whole-molecule ring guard"),
    ("c1ccccc1CSOS",  "ring: the handler's own whole-molecule ring guard"),
    ("BrCSOS",        "halogen: the handler's own C-or-chalcogen-only scan"),
    ("OCCSOS",        "heteroatom organyl: P-41 seniority, guard keeps it closed"),
])
def test_f2_catenated_aba_other_guards_still_fail_closed(smiles, why):
    """TRIPWIRE.  The migration widens ONE guard, not the handler's scope.

    ``name_heterochalcogen_aba`` carries two further whole-molecule guards --
    ``NumRings() > 0`` and "every heavy atom is a chalcogen or a carbon" -- that
    independently refuse ring-bearing and halogenated organyls.  They are NOT in
    this task's scope, so those classes must still fail closed here; if one of
    them starts emitting, the migration reached further than the call site.
    """
    assert _aba(smiles) is None, why


# ==========================================================================
# FAMILY 3 -- rules/pseudoketones.name_acyl_hetero_pseudoketone (1 site)
#
# An acyl group on a P/As hub is a pseudoketone: the carbonyl is the parent
# (`-one`) and the hub is a `phosphanyl`/`arsanyl` substituent carrying its own
# organyls.  The organyl guard was the narrow walker, so a branched, cyclic,
# unsaturated or long organyl on the hub refused the WHOLE handler.
#
# Widening it also puts compound prefixes through this handler's own hub-prefix
# composition for the first time, which was raw `sorted()` + bare concatenation
# + an unconditional `f"({hubyl})"`.  Two Blue Book rules govern that block and
# BOTH are quoted with their headings, because one of them makes a
# previously-emitted name change:
#
#   BB 7272  **P-16.5.1.3.1** "For mononuclear parent hydrides with two or more
#            substituents the first cited substituent never has enclosing marks
#            unless it includes a locant.  The second and further substituents
#            are each enclosed with parentheses EVEN FOR SIMPLE SUBSTITUENTS.
#            When the simple substituent groups are accompanied by
#            multiplicative prefixes such as 'di' and 'tri', the multiplicative
#            prefixes are NOT included in the parentheses."
#            -> `ethyl(methyl)(propyl)phosphane` (PIN), BB 7282
#   BB 39228 `4-[ethyl(methyl)phosphanyl]-1*H*-imidazole` (PIN)
#            -> the rule holds for the SUBSTITUENT-PREFIX form too, which is the
#            shape this handler emits, and the outer marks ESCALATE to square
#            brackets over the inner parentheses (P-16.5.4.1).
#            The silyl analogue is `3-[amino(methyl)silyl]...` (PIN), BB 3545.
#
# So `1-(ethylmethylphosphanyl)propan-1-one` was NON-PIN: it dropped the
# P-16.5.1.3.1 marks.  Its correction is a wrong-name -> better-name change, not
# a refusal -> emission, and it is pinned separately below so the gate can
# attribute it.
# ==========================================================================

def _pk(smiles):
    from orthonym.rules.pseudoketones import name_acyl_hetero_pseudoketone
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return name_acyl_hetero_pseudoketone(mol)


@pytest.mark.parametrize("smiles,expected", [
    # branched -- the narrow walker refused an internal attachment
    ("CCC(=O)P(C(C)C)C(C)C",       "1-[di(propan-2-yl)phosphanyl]propan-1-one"),
    ("CCC(=O)P(C(C)C)C",           "1-[methyl(propan-2-yl)phosphanyl]propan-1-one"),
    # P-16.3.3(b)/P-16.2.4.1(d) retained italicized prefix: bare, hyphen kept under 'di'
    ("CCC(=O)P(C(C)(C)C)C(C)(C)C", "1-(di-tert-butylphosphanyl)propan-1-one"),
    # ring-bearing -- the walker miscounted these as linear chains
    ("CCC(=O)P(C1CCCCC1)C1CCCCC1", "1-(dicyclohexylphosphanyl)propan-1-one"),
    ("CCC(=O)P(Cc1ccccc1)Cc1ccccc1", "1-(dibenzylphosphanyl)propan-1-one"),
    # unsaturated
    ("CCC(=O)P(C=C)C=C",           "1-(diethenylphosphanyl)propan-1-one"),
    # longer than the private walker's chain table, + P-16.5.1.3.1 marks
    ("CCC(=O)P(CCCCCCCCCCCC)C",    "1-[dodecyl(methyl)phosphanyl]propan-1-one"),
    # the As hub shares the code path
    ("CCC(=O)[As](C(C)C)C",        "1-[methyl(propan-2-yl)arsanyl]propan-1-one"),
])
def test_f3_pseudoketone_previously_refused_hub_organyl_is_now_named(smiles, expected):
    assert _pk(smiles) == expected


@pytest.mark.parametrize("smiles,before,after", [
    ("CCC(=O)P(C)CC",        "1-(ethylmethylphosphanyl)propan-1-one",
                             "1-[ethyl(methyl)phosphanyl]propan-1-one"),
    ("CCC(=O)P(c1ccccc1)C",  "1-(methylphenylphosphanyl)propan-1-one",
                             "1-[methyl(phenyl)phosphanyl]propan-1-one"),
])
def test_f3_pseudoketone_two_different_organyls_gain_the_p1651331_marks(
    smiles, before, after,
):
    """A WRONG-name -> BETTER-name change, recorded with both strings.

    BB 7272 requires the second and further substituents of a mononuclear hub to
    be enclosed "even for simple substituents", and BB 39228
    `4-[ethyl(methyl)phosphanyl]-1*H*-imidazole` (PIN) shows the rule applying to
    exactly this substituent-prefix shape.  The bare-concatenated form was
    reachable before this migration (two SIMPLE organyls pass the narrow walker),
    so it is called out separately from the refusal -> emission rows: the gate
    must attribute this one string move to P-16.5.1.3.1 and not to the widening.
    """
    assert _pk(smiles) == after
    assert _pk(smiles) != before


@pytest.mark.parametrize("smiles,expected", [
    # ONE distinct group -> first-cited is bare, so no marks and the SIMPLE
    # multiplier stays attached (BB 7272: multipliers sit outside the marks).
    ("CCC(=O)P(C)C",            "1-(dimethylphosphanyl)propan-1-one"),
    ("CCC(=O)PC",               "1-(methylphosphanyl)propan-1-one"),
    ("CCC(=O)P(c1ccccc1)c1ccccc1", "1-(diphenylphosphanyl)propan-1-one"),
    ("CCC(=O)[As](C)C",         "1-(dimethylarsanyl)propan-1-one"),
    ("CC(=O)P(C)C",             "1-(dimethylphosphanyl)ethan-1-one"),
    ("CCCC(=O)P(C)C",           "1-(dimethylphosphanyl)butan-1-one"),
    # the UNSUBSTITUTED hub keeps its bare retained prefix (a gold target:
    # `1-phosphanylbutan-1-one` is in )
    ("CCC(=O)P",                "1-phosphanylpropan-1-one"),
    # the Si/Ge branch is a DIFFERENT producer and must be untouched
    ("CCC(=O)[Si](C)(C)C",      "1-(trimethylsilyl)propan-1-one"),
])
def test_f3_pseudoketone_existing_names_are_byte_identical(smiles, expected):
    """PROTECT rows.  Every other name this handler emitted must be unchanged."""
    assert _pk(smiles) == expected


@pytest.mark.parametrize("smiles,why", [
    ("CCC(=O)P(CO)C",   "heteroatom organyl on the hub: P-41 seniority"),
    ("CCC(=O)P(CCl)C",  "the handler's own carbon-only hub_frag scan"),
    ("CCC(=O)SC",       "plain carbon on a chalcogen hub: thioester, must decline"),
])
def test_f3_pseudoketone_other_guards_still_fail_closed(smiles, why):
    """TRIPWIRE.  The widening must not enlarge the handler's scope.

    The hub-scope guard ("every hub_frag atom is carbon") and the chalcogen-hub
    thioester carve-out are independent of the organyl guard and must keep
    refusing; the thioester row in particular protects a SENIOR class that
    another handler owns.
    """
    assert _pk(smiles) is None, why


# ==========================================================================
# FAMILY 4 -- assembly/substituent_naming (2 sites: ONE PREDICATE, ONE NAMING)
#
# `_group14_neighbour_prefix` names ONE neighbour of a Si/Ge substituent centre.
# Its two calls are the worked example of the whole task, and they are opposite
# kinds:
#
#   site 4211  PREDICATE -- gates `get_alkoxy_prefix`, a DIFFERENT producer (the
#              contracted alkoxy builder).  Migrated to
#              `is_simple_unbranched_organyl`, which is the same walker, so the
#              routing is bit-for-bit unchanged.  Widening it would change which
#              producer claims an -O-R, with no nomenclature justification.
#   site 4220  NAMING -- the return value IS the prefix.  Widened to
#              `organyl_prefix_name`.
#
# ** The refusal at 4220 did NOT fail closed. **  It handed the fragment to a
# sibling Group-14 producer that fabricates a linear chain from the carbon
# skeleton, so `CC(C)[Si](C)(C)CC(=O)O` shipped as
# `(dimethylpropylsilyl)acetic acid` -- propan-2-yl spelled `propyl`, a WRONG
# CONSTITUTION.  It was masked only by the SELF-01 OPSIN gate, which fails OPEN
# when no JRE is present, so these assertions run ungated.  This is invariant 11
# in its original direction: the refusal was already unmasking a worse generator.
# ==========================================================================

def _g14_frag(smiles):
    """``(mol, si_idx, parent_c_idx, frag_set)`` for ``R[Si](A)(B)-CH2-COOH``."""
    from orthonym.assembly.substituent_naming import _collect_branch_atoms
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    patt = Chem.MolFromSmarts('[Si,Ge]-[CH2]-C(=O)-[OX2H1]')
    match = mol.GetSubstructMatch(patt)
    assert match, f"{smiles} must be an R3Si-CH2-COOH witness"
    si, parent_c = match[0], match[1]
    return mol, si, parent_c, set(_collect_branch_atoms(mol, si, parent_c))


def _g14_name(smiles):
    from orthonym.assembly.substituent_naming import _name_group14_substituent
    mol, si, parent_c, frag = _g14_frag(smiles)
    return _name_group14_substituent(mol, sorted(frag), si)


# --- site 4220: NAMING ----------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("CC(C)[Si](C)(C)CC(=O)O",     "dimethyl(propan-2-yl)silyl"),
    # v29 P3-CLOSEOUT Item A re-baseline: was `methylbis(propan-2-yl)silyl`.
    # `propan-2-yl` is an UNSUBSTITUTED simple prefix that merely carries a
    # locant, so P-16.3.4(a) parenthesises it while P-16.3.2(a) multiplies it
    # with the SIMPLE `di`.  The Blue Book gives this exact shape verbatim on a
    # silane: `ethyldi(propanG2Gyl)silane!(PIN)` at BlueBookV2.md:7294, and
    # `di(propanG2Gyl)!(preferred!prefix)` at :7087.  `bis` is reserved by
    # `**P-16.3.5**`(a) (:7104) for a SUBSTITUTED prefix -- contrast the BB's own
    # `1,4-bis(2-chloropropan-2-yl)benzene (PIN)` (:25793) against
    # `1,4-di(propan-2-yl)cyclohexane (PIN)` (:25721), one chloro apart.
    ("CC(C)[Si](C(C)C)(C)CC(=O)O", "methyldi(propan-2-yl)silyl"),
])
def test_f4_group14_naming_site_replaces_a_fabricated_chain(smiles, expected):
    """The producer now NAMES the branched organyl instead of declining it."""
    assert _g14_name(smiles) == expected


@pytest.mark.parametrize("smiles,wrong,right", [
    ("CC(C)[Si](C)(C)CC(=O)O",     "(dimethylpropylsilyl)acetic acid",
                                   "[dimethyl(propan-2-yl)silyl]acetic acid"),
    # Item A re-baseline, same authority as above (BB 7294 `ethyldi(propan-2-yl)silane` (PIN)).
    ("CC(C)[Si](C(C)C)(C)CC(=O)O", "(methyldipropylsilyl)acetic acid",
                                   "[methyldi(propan-2-yl)silyl]acetic acid"),
    ("CC(C)[Si](C)(C)CCO",         "2-(dimethylpropylsilyl)ethan-1-ol",
                                   "2-[dimethyl(propan-2-yl)silyl]ethan-1-ol"),
])
def test_f4_wrong_constitution_no_longer_ships(ungated_namer, smiles, wrong, right):
    """A WRONG CONSTITUTION is replaced by the correct name, asserted UNGATED.

    `propan-2-yl` was spelled `propyl` -- a different compound.  Both new names
    are OPSIN-exact against the input structure.  The gate is disabled because it
    fails open with no JRE, which is exactly how the wrong string was reachable.
    """
    name = ungated_namer.name(smiles)
    assert name == right
    assert name != wrong
    assert "propylsilyl" not in name, name


@pytest.mark.parametrize("smiles,before,after", [
    ("Cc1ccc(C[Si](C)(C)CC(=O)O)cc1",
     "{[(4-methylphenyl)methyl]di(methyl)silyl}acetic acid",
     "{dimethyl[(4-methylphenyl)methyl]silyl}acetic acid"),
    ("Cc1ccccc1[Si](C)(C)CC(=O)O",
     "[(2-methylphenyl)di(methyl)silyl]acetic acid",
     "[dimethyl(2-methylphenyl)silyl]acetic acid"),
])
def test_f4_prefix_citation_order_follows_p1452(ungated_namer, smiles, before, after):
    """Two names are RE-ORDERED, from non-preferred to preferred.

    P-14.5.2 orders prefixes alphanumerically, so `methyl` is cited before
    `methylphenyl` (a shorter name that is a prefix of a longer one comes first).
    BB 3545/28174 shows the comparison running over the full prefix string
    INCLUDING its marks, with a letter preferred to an open parenthesis:
    `3-[amino(methyl)silyl]-3-[(aminomethyl)silyl]cyclopentan-1-ol` (PIN) {not the
    reverse; ... at the fourth character of the name, the letter 'a' is preferred
    to an open parenthesis}.  Both spellings denote the same structure and both
    round-trip; only the citation order moves.
    """
    name = ungated_namer.name(smiles)
    assert name == after
    assert name != before


@pytest.mark.parametrize("smiles,expected", [
    ("C[Si](C)(C)CC(=O)O",          "trimethylsilyl"),
    ("Cl[Si](C)(C)CC(=O)O",         "chlorodimethylsilyl"),
    ("O[Si](C)(C)CC(=O)O",          "hydroxydimethylsilyl"),
    ("N[Si](C)(C)CC(=O)O",          "aminodimethylsilyl"),
    ("CCO[Si](OCC)(OCC)CC(=O)O",    "triethoxysilyl"),
    ("C[Ge](C)(C)CC(=O)O",          "trimethylgermyl"),
    ("CC(C)(C)[Si](C)(C)CC(=O)O",   "tert-butyldimethylsilyl"),
    ("C1CCCCC1[Si](C)(C)CC(=O)O",   "cyclohexyldimethylsilyl"),
    ("C=C[Si](C)(C)CC(=O)O",        "ethenyldimethylsilyl"),
    ("c1ccccc1[Si](C)(C)CC(=O)O",   "dimethylphenylsilyl"),
    ("CCCCCCCCCCCC[Si](C)(C)CC(=O)O", "dodecyldimethylsilyl"),
])
def test_f4_group14_existing_prefixes_are_byte_identical(smiles, expected):
    """PROTECT rows.

    `tert-butyldimethylsilyl` is the one that pins the ORDER primitive: raw
    `sorted()` keys `tert-butyl` on its 't' and would emit
    `dimethyltert-butylsilyl`.  `chlorodimethylsilyl` and `hydroxydimethylsilyl`
    pin that the P-16.5.1.3.1 leg was deliberately NOT applied here -- this
    producer competes with a sibling that emits the bare form for that shape.
    """
    assert _g14_name(smiles) == expected


# --- site 4211: ROUTING PREDICATE ----------------------------------------

def _o_neighbour_prefix(smiles):
    """Run `_group14_neighbour_prefix` on the alkoxy OXYGEN of the Si centre."""
    from orthonym.assembly.substituent_naming import _group14_neighbour_prefix
    mol, si, parent_c, frag = _g14_frag(smiles)
    os_ = [n.GetIdx() for n in mol.GetAtomWithIdx(si).GetNeighbors()
           if n.GetSymbol() == 'O' and n.GetIdx() in frag]
    assert len(os_) == 1, smiles
    return _group14_neighbour_prefix(mol, os_[0], si, frag)


@pytest.mark.parametrize("smiles,expected", [
    ("CO[Si](C)(C)CC(=O)O",   "methoxy"),
    ("CCO[Si](C)(C)CC(=O)O",  "ethoxy"),
    ("CCCO[Si](C)(C)CC(=O)O", "propoxy"),
    # phenyl/naphthyl ARE in the narrow class, so phenoxy was always admitted
    # here; it is a protect row, not a widening.
    ("c1ccccc1O[Si](C)(C)CC(=O)O", "phenoxy"),
])
def test_f4_predicate_still_admits_the_simple_alkoxy(smiles, expected):
    """The contracted alkoxy producer keeps the cases it already handled
    (P-68.2.6.2, BB 38245 `-Ge(OEt)3` -> `triethoxygermyl`)."""
    assert _o_neighbour_prefix(smiles) == expected


@pytest.mark.parametrize("smiles,why", [
    ("CC(C)O[Si](C)(C)CC(=O)O",      "branched R -- internal attachment"),
    ("CC(C)(C)O[Si](C)(C)CC(=O)O",   "branched R -- tert-butoxy"),
    ("C1CCCCC1O[Si](C)(C)CC(=O)O",   "ring-bearing R"),
    ("C=CO[Si](C)(C)CC(=O)O",        "unsaturated R"),
])
def test_f4_predicate_did_not_silently_widen(smiles, why):
    """THE tripwire for the PREDICATE site (4211).

    A ring-bearing, branched or unsaturated -O-R must still NOT take the
    contracted-alkoxy path.  `is_simple_unbranched_organyl` is the same walker
    the deprecated namer used, so this routing decision is bit-for-bit what it
    was; if the site were widened to `organyl_prefix_name` these would start
    returning an alkoxy prefix and a different producer would claim the molecule
    -- a behaviour change with no nomenclature justification.
    """
    assert _o_neighbour_prefix(smiles) is None, why


@pytest.mark.parametrize("smiles", [
    "CC(C)O[Si](C)(C)CC(=O)O",
    "C1CCCCC1O[Si](C)(C)CC(=O)O",
    "C=CO[Si](C)(C)CC(=O)O",
])
def test_f4_predicate_refusal_leaves_the_whole_group14_namer_closed(smiles):
    """The predicate's `None` propagates: `_group14_neighbour_prefix` returns None
    for that neighbour, so `_name_group14_substituent` declines the fragment and
    the cascade continues, exactly as before the migration."""
    assert _g14_name(smiles) is None


# ==========================================================================
# FAMILY 5 -- rules/inorganic_acids (2 sites, both NAMING)
#
#   line 338  `_amido_n_prefix`     -> the N-locant prefix block of a
#             P-oxoacid amide: `N,N-dimethyl`, `N-ethyl-N-methyl`.
#   line 445  `name_p_oxoacid_frn`  -> the organyl cited in front of the
#             functional-replacement acid stem: `methylphosphonochloridic acid`.
#
# Both strings are concatenated straight into the emitted name (site 445 through
# `build_p_frn_acid_name`, which does no marking of its own), so a widened prefix
# carrying a locant would have run into the stem.  Enclosing marks come from the
# shared `enclose_if_compound`:
#
#   BB 32784  `*N*-(propan-2-yl)acetamide` (PIN)  -> a locanted organyl on an
#             italic-N locant IS enclosed.
#   BB 3465   `4-butyl-4-*tert*-butylcyclohexan-1-ol` (PIN) -> the retained
#             italicized prefix is cited BARE even directly after a locant
#             (P-16.3.3(b) + P-16.2.4.1(d)), so the N-substituent is cited bare per BB 3465.
# ==========================================================================

def _f5(smiles):
    """End-to-end at DEFAULT settings is what these assert; the producer is
    reached through the normal cascade."""
    from orthonym.rules.inorganic_acids import name_p_oxoacid_frn
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return name_p_oxoacid_frn(mol)


@pytest.mark.parametrize("smiles,expected", [
    # site 445 -- organyl on P
    ("CC(C)P(=O)(Cl)O",        "(propan-2-yl)phosphonochloridic acid"),
    ("CC(C)(C)P(=O)(Cl)O",     "tert-butylphosphonochloridic acid"),
    ("C1CCCCC1P(=O)(Cl)O",     "cyclohexylphosphonochloridic acid"),
    ("c1ccccc1CP(=O)(Cl)O",    "benzylphosphonochloridic acid"),
    ("CCCCCCCCCCCCP(=O)(Cl)O", "dodecylphosphonochloridic acid"),
    ("CC(C)P(=O)(OC#N)O",      "(propan-2-yl)phosphonocyanatidic acid"),
])
def test_f5_frn_organyl_previously_refused_is_now_named(smiles, expected):
    assert _f5(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    # site 338 -- N-substituent of the amido nitrogen
    ("CC(C)NP(=O)(O)O",     "N-(propan-2-yl)phosphoramidic acid"),
    ("CC(C)(C)NP(=O)(O)O",  "N-tert-butylphosphoramidic acid"),
    ("C1CCCCC1NP(=O)(O)O",  "N-cyclohexylphosphoramidic acid"),
    ("C=CNP(=O)(O)O",       "N-ethenylphosphoramidic acid"),
    # P-16.3.3(b)/P-16.2.4.1(d): the multiplier keeps the italicized prefix's hyphen
    ("CC(C)(C)N(C(C)(C)C)P(=O)(O)O", "N,N-di-tert-butylphosphoramidic acid"),
    # BB 25719 `1,4-di(propan-2-yl)cyclohexane` (PIN): SIMPLE multiplier OUTSIDE
    # the marks of a compound prefix
    ("CC(C)N(C(C)C)P(=O)(O)O",       "N,N-di(propan-2-yl)phosphoramidic acid"),
    # P-14.5.2 order across a simple and a compound prefix ('methyl' < 'propanyl')
    ("CC(C)N(C)P(=O)(O)O",  "N-methyl-N-(propan-2-yl)phosphoramidic acid"),
])
def test_f5_amido_n_substituent_previously_refused_is_now_named(smiles, expected):
    """BB 32784 `*N*-(propan-2-yl)acetamide` (PIN) for the enclosed locanted
    prefix; BB 3465 for `tert-butyl` cited bare after a locant (P-16.3.3(b)/P-16.2.4.1(d))."""
    assert _f5(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    ("NP(=O)(O)O",         "phosphoramidic acid"),
    ("CNP(=O)(O)O",        "N-methylphosphoramidic acid"),
    ("CN(C)P(=O)(O)O",     "N,N-dimethylphosphoramidic acid"),
    ("CCN(CC)P(=O)(O)O",   "N,N-diethylphosphoramidic acid"),
    ("CCN(C)P(=O)(O)O",    "N-ethyl-N-methylphosphoramidic acid"),
    ("c1ccccc1NP(=O)(O)O", "N-phenylphosphoramidic acid"),
    ("CP(=O)(Cl)O",        "methylphosphonochloridic acid"),
    ("c1ccccc1P(=O)(Cl)O", "phenylphosphonochloridic acid"),
    ("CP(=O)(OC#N)O",      "methylphosphonocyanatidic acid"),
    ("CN(C)P(=O)(Cl)O",    "N,N-dimethylphosphoramidochloridic acid"),
])
def test_f5_existing_names_are_byte_identical(smiles, expected):
    """PROTECT rows.  `N-ethyl-N-methyl` pins that the P-14.5.2 citation order is
    unchanged, and every simple prefix must stay BARE -- `enclose_if_compound` is
    a no-op on the letters-only class the narrow walker allowed."""
    assert _f5(smiles) == expected


@pytest.mark.parametrize("smiles,why", [
    ("OCCNP(=O)(O)O",      "heteroatom organyl on N: P-41 seniority"),
    ("CC(C)P(=O)(O)O",     "no class group at all -> the phosphonic suffix path owns it"),
    ("CC(C)P(=O)(Cl)N",    "no -OH left -> not class 'acid'"),
    ("CC(C)P(=O)(Cl)OC",   "an ester -OR neighbour"),
])
def test_f5_other_guards_still_fail_closed(smiles, why):
    """TRIPWIRE.  The widening must not enlarge `name_p_oxoacid_frn`'s scope: the
    class-group requirement, the >=1 -OH 'acid' requirement and the no-ester rule
    are independent of the organyl guard.  The `CC(C)P(=O)(O)O` row matters most --
    a plain phosphonic acid must keep routing to the suffix path (which names it
    `propan-2-ylphosphonic acid`), not be captured as an FRN acid."""
    assert _f5(smiles) is None, why


# ==========================================================================
# FAMILY 6 -- rules/polychalcogen (3 sites, all NAMING)
#
#   line 105  `_terminal_substituents` -> the chain-parent prefix block
#   line 259  the P-68.4.2.2/.3 parent+suffix layer (`methyldisulfanol`)
#   line 445  `name_polysulfoxide_sulfone` -> the lambda/oxo prefix block
#
# Both locanted prefix blocks were open-coded and are now ONE helper,
# `_cite_locanted_prefixes`, because the widening exposed three defects in them:
#
#  1. segments were joined by '' -> `1-ethyl3-methyltrisulfane`.  BB 21649
#     `3-ethyl-2-methylhexane` (PIN) is the witness for the HYPHEN separator --
#     and for citing alphanumerically even when the locants then DESCEND, which
#     the old "ascending first-locant order (matches alpha here)" only got right
#     for the letters-only class the narrow walker could return.
#  2. no P-16.5.1.1 marks, so a locanted prefix ran into the stem.
#  3. no P-16.3.3(b)/P-16.2.4.1(d) carve-out, so `tert-butyl` lost its multiplier hyphen.
#
# The widening also changes WHICH nomenclature claims two molecules, and the Blue
# Book says the new one is preferred -- BB 23385, verbatim:
#
#   1,1'-(ethane-1,2-diyl)bis(3-methyltrisulfane) (PIN) (not
#   2,3,4,7,8,9-hexathiadecane; trisulfane, HS-S-SH, is a parent hydride and is
#   NOT ALLOWED TO BE A HETEROUNIT)
#
# so the skeletal-replacement `trithia` spelling is the explicitly rejected form.
# Before this migration the choice between the two was decided by nothing more
# than whether the narrow walker happened to accept the substituent: `CSSSC` got
# the trisulfane name, its dodecyl analogue got `2,3,4-trithiahexadecane`.
#
# The 2-chalcogen chains stay excluded (BB 27864: substituted `disulfane` names
# "are not recommended"; BB 2979 `(bromodisulfanyl)methane` (PIN) `not
# 1-bromo-2-methyldisulfane`) -- this module already requires n>=3, and only the
# lambda-oxidised sulfoxide/sulfone layer uses a locanted disulfane, which BB
# 29415 `1-methyl-2-phenyl-1lambda6,2lambda6-disulfane-1,1,2,2-tetrone` (PIN)
# confirms.
# ==========================================================================

@pytest.mark.parametrize("smiles,expected", [
    # site 105 -- chain parent
    ("CC(C)SSSC(C)C",           "1,3-di(propan-2-yl)trisulfane"),
    ("CC(C)SSSC",               "1-methyl-3-(propan-2-yl)trisulfane"),
    ("CC(C)(C)SSSC(C)(C)C",     "1,3-di-tert-butyltrisulfane"),
    # site 259 -- parent + -ol/-thiol suffix layer
    ("CC(C)SSO",                "(propan-2-yl)disulfanol"),
    ("CC(C)SSSO",               "(propan-2-yl)trisulfanol"),
    # site 445 -- polysulfoxide / sulfone
    ("CC(C)S(=O)S(=O)C(C)C",
     "1,2-di(propan-2-yl)-1lambda4,2lambda4-disulfane-1,2-dione"),
    ("CC(C)S(=O)S(=O)C",
     "1-methyl-2-(propan-2-yl)-1lambda4,2lambda4-disulfane-1,2-dione"),
])
def test_f6_polychalcogen_previously_refused_organyl_is_now_named(
    ungated_namer, smiles, expected,
):
    assert ungated_namer.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    ("CC(C)(C)SSSC",   "1-tert-butyl-3-methyltrisulfane"),
    ("CC(C)(C)SSSCC",  "1-tert-butyl-3-ethyltrisulfane"),
    ("CC(C)(C)S(=O)S(=O)C",
     "1-tert-butyl-2-methyl-1lambda4,2lambda4-disulfane-1,2-dione"),
])
def test_f6_citation_order_ignores_the_italicized_prefix(
    ungated_namer, smiles, expected,
):
    """P-14.5.2 keys on the LETTERS, so `tert-butyl` sorts under 'butyl'.

    These are the rows that distinguish the shared citation key from a raw string
    sort: raw order puts 'methyl'/'ethyl' before 'tert-butyl' (on 'm'/'e' < 't'),
    while P-14.5.2 compares 'butyl' < 'ethyl' < 'methyl' and cites tert-butyl
    FIRST -- which then also takes locant 1 under P-14.4 (g).  Without a row like
    this the raw-sort mutant survives, because every other witness here happens to
    sort identically both ways.
    """
    assert ungated_namer.name(smiles) == expected


@pytest.mark.parametrize("smiles,before,after", [
    ("CSSSCC",   "1-ethyl3-methyltrisulfane",   "1-ethyl-3-methyltrisulfane"),
    ("CSSSSSCC", "1-ethyl5-methylpentasulfane", "1-ethyl-5-methylpentasulfane"),
])
def test_f6_prefix_segments_are_hyphen_separated(ungated_namer, smiles, before, after):
    """A malformed name that SHIPPED is repaired.

    Two different simple organyls both pass the retired narrow walker, so
    `1-ethyl3-methyltrisulfane` was reachable and emitted before this migration --
    OPSIN even parses it.  BB 21649 `3-ethyl-2-methylhexane` (PIN) is the witness
    for the hyphen.  Recorded with both strings so the gate attributes the move to
    the separator fix and not to the widening.
    """
    name = ungated_namer.name(smiles)
    assert name == after
    assert name != before


@pytest.mark.parametrize("smiles,before,after", [
    ("CCCCCCCCCCCCSSSC", "2,3,4-trithiahexadecane",
                         "1-dodecyl-3-methyltrisulfane"),
    ("C=CSSSC=C",        "3,4,5-trithiahepta-1,6-diene",
                         "1,3-diethenyltrisulfane"),
])
def test_f6_polysulfane_parent_beats_skeletal_replacement(
    ungated_namer, smiles, before, after,
):
    """The widening changes WHICH nomenclature claims the molecule, toward the PIN.

    BB 23385 verbatim: `1,1'-(ethane-1,2-diyl)bis(3-methyltrisulfane)` (PIN) `(not
    2,3,4,7,8,9-hexathiadecane; trisulfane, HS-S-SH, is a parent hydride and is
    not allowed to be a heterounit)`.  The `trithia` replacement spelling is
    therefore the rejected form, and these two molecules were only getting it
    because the narrow walker refused their substituents (dodecyl was longer than
    its chain table; ethenyl carried a double bond), which made this handler
    decline and let skeletal replacement win.  `CSSSC` already took the trisulfane
    name, so the old behaviour was inconsistent within one structural class.
    """
    name = ungated_namer.name(smiles)
    assert name == after
    assert name != before


@pytest.mark.parametrize("smiles,expected", [
    ("SSS",               "trisulfane"),
    ("CSSS",              "1-methyltrisulfane"),
    ("CSSSC",             "1,3-dimethyltrisulfane"),
    ("CSSSSC",            "1,4-dimethyltetrasulfane"),
    ("CSSSSSC",           "1,5-dimethylpentasulfane"),
    ("CSSO",              "methyldisulfanol"),
    ("CSSSO",             "methyltrisulfanol"),
    ("CS(=O)S(=O)C",      "1,2-dimethyl-1lambda4,2lambda4-disulfane-1,2-dione"),
    ("CS(=O)S(=O)S(=O)C",
     "1,3-dimethyl-1lambda4,2lambda4,3lambda4-trisulfane-1,2,3-trione"),
])
def test_f6_polychalcogen_existing_names_are_byte_identical(
    ungated_namer, smiles, expected,
):
    """PROTECT rows across all three sites."""
    assert ungated_namer.name(smiles) == expected


@pytest.mark.parametrize("smiles,forbidden", [
    # BB 27864 / BB 2979: substituted 2-chalcogen chains are NOT named as
    # `disulfane`; this module requires n>=3 and that guard is untouched.
    ("CC(C)SSC(C)C", "disulfane"),
    ("CSSC",         "disulfane"),
    ("CSSc1ccccc1",  "disulfane"),
])
def test_f6_two_chalcogen_chains_are_still_not_named_as_disulfane(
    ungated_namer, smiles, forbidden,
):
    """TRIPWIRE.  BB 2979 `(bromodisulfanyl)methane` (PIN) `(not
    bromo(methyl)disulfane ...; not 1-bromo-2-methyldisulfane)`.  The widening must
    not pull the n==2 disulfides into the parent-hydride path -- only the
    lambda-oxidised sulfoxide/sulfone layer may use a locanted disulfane."""
    assert forbidden not in ungated_namer.name(smiles)


@pytest.mark.parametrize("smiles", [
    "CSSSCC", "CCSSSC", "CSSSSSCC", "CC(C)SSSC", "CSSSC",
    "CC(C)S(=O)S(=O)C", "CS(=O)S(=O)C",
])
def test_f6_locants_do_not_depend_on_smiles_atom_order(ungated_namer, smiles):
    """P-14.4 (g) (BB 3307) makes the chain orientation DETERMINISTIC.

    Both prefix blocks chose the orientation by comparing only the locant
    multiset; on a tie (the common case for a symmetric chain, e.g. {1,3} either
    way) the choice fell through to RDKit atom order, so the same compound got
    `1-ethyl-3-methyltrisulfane` or `3-ethyl-1-methyltrisulfane` depending on how
    its SMILES happened to be written.  BB 3307 P-14.4 (g) supplies the missing
    criterion -- "lowest locants for the substituent cited first as a prefix in
    the name" -- witnessed verbatim by BB 29956
    `1-hydroxy-3-oxopropane-1,2,3-tricarboxylic acid` (PIN) [not
    `3-hydroxy-1-oxo...`].  Naming the raw SMILES and its canonical form must now
    agree.
    """
    from rdkit import Chem
    assert ungated_namer.name(smiles) == ungated_namer.name(Chem.CanonSmiles(smiles))


# ==========================================================================
# FAMILY 7 -- rules/polyazane (4 sites, all NAMING)
#
#   line 108  `_collect_substituents`   -> the N-chain prefix block
#   lines 256/257 `_name_azoxy`         -> the two organyls of R-N=N(O)-R
#   line 298  `_formazan_substituent`   -> one organyl on a formazan skeleton
#
# `_format_n2_substituents` carried the same three defects as polychalcogen's
# blocks (no hyphen separator -> `1-ethyl2-methylhydrazine` SHIPPED; no P-16.3.3
# marks; raw sort), plus the missing P-14.4 (g) orientation tie-break, and
# `_format_substituted_polyazane`/`_name_azoxy` open-coded `f"({name})"` which
# cannot escalate marks and has no P-16.3.4 carve-out.
#
# The widening also moves five names from a CARBON parent to the HYDRAZINE
# parent, which is the preferred direction:
#
#   P-44.1.2 (BB 18917)   class seniority order `N > P > As > ... > O > S > Se > Te > C`
#             -- nitrogen before carbon.
#   BB 18950  `1-(2H-pyran-3-yl)-2-(silolan-2-yl)hydrazine` (PIN) `( N > Si > O)`
#             -- hydrazine is the parent even against two RING substituents.
#
# and `methylhydrazine` / `phenylhydrazine` (BB PINs, P-68.3.1.2) were already
# emitted, so before this migration the class split purely on whether the narrow
# walker accepted the substituent.  (BB 19376 `2-hydrazinylpyridine` (PIN) is not
# a counterexample: P-44.1.2.2 criterion (1) (BB 19336/19340) makes the RING
    # senior to the chain when both hold the same senior element -- BB 19378
    # says exactly "(ring is senior to chain)", not anything about heterocycles.
    # P-44.2.1 is ring-vs-RING and does not govern this. [v29 P3-FIX Item 9])
# ==========================================================================

@pytest.mark.parametrize("smiles,expected", [
    # site 108 -- N-chain prefixes
    ("CC(C)NNC(C)C",         "1,2-di(propan-2-yl)hydrazine"),
    ("CC(C)NNC",             "1-methyl-2-(propan-2-yl)hydrazine"),
    ("CC(C)(C)NNC(C)(C)C",   "1,2-di-tert-butylhydrazine"),
    ("CC(C)NNN",             "1-(propan-2-yl)triazane"),
    ("CC(C)N=NC(C)C",        "1,2-di(propan-2-yl)diazene"),
    # >=3-N chain with two DIFFERENT prefixes: exercises that composer's own
    # P-14.4 (g) tie-break (methyl is cited first, so it takes locant 1)
    ("CC(C)NNNC",            "1-methyl-3-(propan-2-yl)triazane"),
    ("CNNNCC",               "1-ethyl-3-methyltriazane"),
    # MULTIPLIED prefixes on the >=3-N composer: P-16.3.4 hyphen + BB 25719
    # SIMPLE multiplier outside the marks of a compound prefix
    ("CC(C)(C)NNNC(C)(C)C",  "1,3-di-tert-butyltriazane"),
    ("CC(C)NNNC(C)C",        "1,3-di(propan-2-yl)triazane"),
    # sites 256/257 -- azoxy
    ("CC(C)N=[N+]([O-])C(C)C",       "di(propan-2-yl)diazene oxide"),
    ("C1CCCCC1N=[N+]([O-])C1CCCCC1", "dicyclohexyldiazene oxide"),
])
def test_f7_polyazane_previously_refused_organyl_is_now_named(
    ungated_namer, smiles, expected,
):
    assert ungated_namer.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    ("CC(C)(C)NNC",  "1-tert-butyl-2-methylhydrazine"),
    ("CC(C)(C)NNCC", "1-tert-butyl-2-ethylhydrazine"),
    ("CC(C)(C)N=NC", "1-tert-butyl-2-methyldiazene"),
    ("CC(C)(C)NNN",  "1-tert-butyltriazane"),
    # the >=3-N block is a SEPARATE composer and needs its own rows
    ("CC(C)(C)NNNC", "1-tert-butyl-3-methyltriazane"),
    # P-16.3.4 under a multiplier, on the azoxy composer
    ("CC(C)(C)N=[N+]([O-])C(C)(C)C", "di-tert-butyldiazene oxide"),
])
def test_f7_citation_order_ignores_the_italicized_prefix(
    ungated_namer, smiles, expected,
):
    """P-14.5.2 keys on the LETTERS, so `tert-butyl` sorts under 'butyl'.

    These rows distinguish the shared citation key from a raw string sort, which
    would put 'methyl'/'ethyl' first (on 'm'/'e' < 't').  Every other witness in
    this family sorts identically both ways, so without a row like this the
    raw-sort mutant survives.
    """
    assert ungated_namer.name(smiles) == expected


@pytest.mark.parametrize("smiles,before,after", [
    ("CNNCC", "1-ethyl2-methylhydrazine", "1-ethyl-2-methylhydrazine"),
])
def test_f7_prefix_segments_are_hyphen_separated(ungated_namer, smiles, before, after):
    """A malformed name that SHIPPED is repaired (BB 21649
    `3-ethyl-2-methylhexane` (PIN) for the separator).  Two different simple
    organyls both pass the retired narrow walker, so this was reachable before the
    migration."""
    name = ungated_namer.name(smiles)
    assert name == after
    assert name != before


@pytest.mark.parametrize("smiles,before,after", [
    ("CC(C)NN",          "2-hydrazinylpropane",          "(propan-2-yl)hydrazine"),
    ("CC(C)(C)NN",       "2-hydrazinyl-2-methylpropane", "tert-butylhydrazine"),
    ("C1CCCCC1NN",       "hydrazinylcyclohexane",        "cyclohexylhydrazine"),
    ("C=CNN",            "hydrazinylethene",             "ethenylhydrazine"),
    ("CCCCCCCCCCCCNN",   "1-hydrazinyldodecane",         "dodecylhydrazine"),
])
def test_f7_hydrazine_is_the_senior_parent_over_carbon(
    ungated_namer, smiles, before, after,
):
    """Five names move from a CARBON parent to the HYDRAZINE parent.

    P-44.1.2 (BB 18917) gives the class seniority order `N > P > As > ... > O > S > Se > Te >
    C` -- nitrogen BEFORE carbon -- and BB 18950
    `1-(2H-pyran-3-yl)-2-(silolan-2-yl)hydrazine` (PIN) `( N > Si > O)` keeps
    hydrazine as the parent even against two ring substituents.  `methylhydrazine`
    and `phenylhydrazine` (BB PINs, P-68.3.1.2) were already emitted, so the old
    behaviour split one class on nothing but whether the narrow walker accepted
    the substituent.  All five new names are OPSIN-exact.
    """
    name = ungated_namer.name(smiles)
    assert name == after
    assert name != before


@pytest.mark.parametrize("smiles,expected", [
    ("NN",                          "hydrazine"),
    ("CNN",                         "methylhydrazine"),
    ("c1ccccc1NN",                  "phenylhydrazine"),
    ("CNNC",                        "1,2-dimethylhydrazine"),
    ("CN(C)N",                      "1,1-dimethylhydrazine"),
    ("NNN",                         "triazane"),
    ("CNNN",                        "1-methyltriazane"),
    ("CNNNC",                       "1,3-dimethyltriazane"),
    ("CN=NC",                       "1,2-dimethyldiazene"),
    ("c1ccccc1N=Nc1ccccc1",         "1,2-diphenyldiazene"),
    ("c1ccccc1N=[N+]([O-])c1ccccc1", "diphenyldiazene oxide"),
    ("CN=[N+]([O-])C",              "dimethyldiazene oxide"),
])
def test_f7_polyazane_existing_names_are_byte_identical(
    ungated_namer, smiles, expected,
):
    """PROTECT rows.  `phenylhydrazine` matters most: it shows a RING substituent
    was already cited as a prefix on the hydrazine parent, which is what makes
    `cyclohexylhydrazine` the consistent answer rather than a new convention."""
    assert ungated_namer.name(smiles) == expected


@pytest.mark.parametrize("smiles", [
    "CNNCC", "CCNNC", "CC(C)NNC", "CNNC", "CC(C)N=NC", "CC(C)NNN",
    # the >=3-N composer has its OWN orientation key, so it needs its own rows
    # with two DIFFERENT substituents for the tie-break to be exercised
    "CC(C)NNNC", "CNNNCC", "CC(C)(C)NNNC",
])
def test_f7_locants_do_not_depend_on_smiles_atom_order(ungated_namer, smiles):
    """P-14.4 (g) (BB 3307) makes the N-chain direction DETERMINISTIC.

    Both `_format_n2_substituents` and `_format_substituted_polyazane` compared
    only the locant SET; on the symmetric 2-N parent that is {1,2} either way, so
    the direction fell through to RDKit atom order.  BB 29956
    `1-hydroxy-3-oxopropane-1,2,3-tricarboxylic acid` (PIN) [not
    `3-hydroxy-1-oxo...`] is the verbatim witness for the criterion.
    """
    from rdkit import Chem
    assert ungated_namer.name(smiles) == ungated_namer.name(Chem.CanonSmiles(smiles))


@pytest.mark.parametrize("smiles,why", [
    ("OCCNN",   "heteroatom organyl: P-41 seniority keeps it closed"),
    ("CC(C)N=[N+]([O-])C", "unsymmetric azoxy needs the NNO/ONN locant machinery"),
])
def test_f7_other_guards_still_fail_closed(smiles, why):
    """TRIPWIRE.  The azoxy producer only builds the SYMMETRIC case (its `ra != rb`
    guard); widening the organyl namer must not make it emit a locant-ambiguous
    unsymmetric name.  The heteroatom organyl stays refused by the primitive."""
    from orthonym.rules.polyazane import _name_azoxy
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    assert _name_azoxy(mol) is None, why


# ==========================================================================
# FAMILY 8 -- rules/mononuclear_hydrides (10 sites, all NAMING)
#
# The largest family: `_classify_organyls`, `_classify_mixed_halo_organyl`,
# `name_arsanyl_substituent`, `_classify_phosphane_subs`, `name_heterone`,
# `name_heteroimine`, the phosphanimine hub/N pair, and the lambda-sulfane
# imine/oxide hub/N pair.
#
# ** What the refusal was actually doing (invariant 11, ungated probe) **
# A `None` here does not fail closed -- it hands the molecule to generators that
# DROP the heteroatom.  With the SELF-01 gate disabled the producers emitted
# `propane` for CC(C)[AsH2] (the arsenic simply gone), `cyclohexane` for
# C1CCCCC1[AsH2], `(methylmethyl)methane` for C[Si](C)(C)N=P, and `propylsilane`
# / `tetrapropylsilane` (propan-2-yl spelled propyl) for the silanes.
# At DEFAULT settings the gate suppressed all of those, so they were NOT shipped
# names -- they emitted `unknown organic compound` / `<element> compound (not
# supported)`.  The shipped effect of this migration is therefore
# REFUSAL -> CORRECT NAME; the fabrications were a producer defect that the gate
# caught, and this phase removes the defect rather than relying on the gate.
#
# Four composers are repaired: `_build_mixed_substituent_string`,
# `_compose_io_prefixes`, the phosphanimine `_block`, and `name_heteroimine`'s
# single-organyl join -- plus `phosphorus._build_substituent_string`, which family
# 8 feeds and which produced `dipropan-2-yl`, `ditert-butyl` and
# `tert-butyl(dimethyl)`.  BB 7272 (P-16.5.1.3.1) is verbatim on the last one:
# "When the simple substituent groups are accompanied by multiplicative prefixes
# such as 'di' and 'tri', the multiplicative prefixes are NOT included in the
# parentheses" -- BB 16286 `tert-butyldi(methyl)phosphane` (PIN) is the shape.
# ==========================================================================

@pytest.mark.parametrize("smiles,expected", [
    # pnictogen hydrides -- these are the atom-drop cases
    ("CC(C)[AsH2]",       "(propan-2-yl)arsane"),
    ("CC(C)[As](C)C",     "dimethyl(propan-2-yl)arsane"),
    ("C1CCCCC1[AsH2]",    "cyclohexylarsane"),
    ("CC(C)[SbH2]",       "(propan-2-yl)stibane"),
    ("CC(C)[BiH2]",       "(propan-2-yl)bismuthane"),
    # heterone
    ("CC(C)[Si](C(C)C)=O", "di(propan-2-yl)silanone"),
    # heteroimine + phosphanimine hub/N
    ("CC(C)P=N",          "1-(propan-2-yl)phosphanimine"),
    ("CC(C)P(C(C)C)(C(C)C)=N",
     "P,P,P-tri(propan-2-yl)-lambda5-phosphanimine"),
    ("CC(C)P(C)(C)=NC(C)C",
     "P,P-dimethyl-P-(propan-2-yl)-N-(propan-2-yl)-lambda5-phosphanimine"),
    # lambda-sulfane imine/oxide
    ("CC(C)S(=N)(=O)C(C)C", "iminodi(propan-2-yl)-lambda6-sulfanone"),
    ("CS(=N)(=O)C(C)C",     "imino(methyl)(propan-2-yl)-lambda6-sulfanone"),
    # phosphane with a silyl co-substituent
    ("CC(C)P([SiH3])C",   "methyl(propan-2-yl)(silyl)phosphane"),
    # mixed halo+organyl with the COMPOUND prefix in the FIRST (unmarked) slot:
    # P-14.5.2 puts 'cyclohexylmethyl'/'butanyl' before 'fluoro', and
    # P-16.5.1.3.1 withholds only the SEPARATING marks, so the compound prefix
    # still takes its own P-16.5.1.1 marks there.
    ("C1CCCCC1C[Si](F)(F)F", "(cyclohexylmethyl)tri(fluoro)silane"),
    ("CCC(C)[Si](F)(F)F",    "(butan-2-yl)tri(fluoro)silane"),
    ("CC(C)[Si](F)(F)F",     "trifluoro(propan-2-yl)silane"),
])
def test_f8_mononuclear_previously_refused_organyl_is_now_named(
    ungated_namer, smiles, expected,
):
    assert ungated_namer.name(smiles) == expected


@pytest.mark.parametrize("smiles,dropped_element", [
    ("CC(C)[AsH2]",    "As"),
    ("C1CCCCC1[AsH2]", "As"),
    ("CC(C)[SbH2]",    "Sb"),
    ("CC(C)[BiH2]",    "Bi"),
    ("CC(C)P=N",       "P"),
])
def test_f8_the_hub_element_is_never_dropped_from_the_name(
    ungated_namer, smiles, dropped_element,
):
    """REGRESSION guard for the invariant-11 failure this family carried.

    With the guard refusing, the fallback generators emitted `propane` for
    CC(C)[AsH2] and `cyclohexane` for C1CCCCC1[AsH2] -- the hub element silently
    GONE, a wrong constitution that only the OPSIN gate stopped.  The emitted name
    must now carry the hub's parent-hydride stem, and this runs UNGATED so nothing
    downstream can rescue it.
    """
    stem = {"As": "arsane", "Sb": "stibane", "Bi": "bismuthane",
            "P": "phosphan"}[dropped_element]
    name = ungated_namer.name(smiles)
    assert stem in name, name
    assert name not in ("propane", "cyclohexane"), name


@pytest.mark.parametrize("smiles,before,after", [
    # P-44.1.2 (BB 18917) class seniority: Si is senior to C, so the SILANE is the parent
    ("Cl[Si](C)(C)C(C)C",    "2-(chlorodimethylsilyl)propane",
                             "chlorodi(methyl)(propan-2-yl)silane"),
    ("Cl[Si](C)(C)C(C)(C)C", "2-(chlorodimethylsilyl)-2-methylpropane",
                             "tert-butyl(chloro)di(methyl)silane"),
    ("CC(C)[Si](Cl)(Cl)Cl",  "2-(trichlorosilyl)propane",
                             "trichloro(propan-2-yl)silane"),
    ("CC(C)[Si](C)=O",       "2-methylsilanonylpropane",
                             "methyl(propan-2-yl)silanone"),
])
def test_f8_silicon_is_the_senior_parent_over_carbon(
    ungated_namer, smiles, before, after,
):
    """Four names move from a CARBON parent to the SILANE parent, toward the PIN.

    P-44.1.2 (BB 18917) gives the class seniority order `N > P > As > Sb > Bi > Si > Ge > Sn >
    Pb > B > ... > O > S > Se > Te > C` -- silicon BEFORE carbon.  The codebase
    already emitted `methylsilane`, `cyclohexylsilane`, `trichloro(methyl)silane`
    and `chlorotri(methyl)silane` on the silane parent, so the old behaviour split
    one class on nothing but whether the narrow walker accepted the substituent.
    All four new names are OPSIN-exact.
    """
    name = ungated_namer.name(smiles)
    assert name == after
    assert name != before


def test_f8_multiplier_sits_outside_the_enclosing_marks(ungated_namer):
    """BB 7272 verbatim: "the multiplicative prefixes are NOT included in the
    parentheses".  `imino(dimethyl)-lambda6-sulfanone` put `di` inside them; the
    shape BB 16286 spells is `tert-butyldi(methyl)phosphane` (PIN)."""
    name = ungated_namer.name("CS(=N)(=O)C")
    assert name == "iminodi(methyl)-lambda6-sulfanone"
    assert name != "imino(dimethyl)-lambda6-sulfanone"


@pytest.mark.parametrize("smiles,expected", [
    ("C[SiH3]",                 "methylsilane"),
    ("C[Si](C)(C)C",            "tetramethylsilane"),
    ("c1ccccc1[SiH3]",          "phenylsilane"),
    ("CC(C)(C)[SiH3]",          "tert-butylsilane"),
    ("C1CCCCC1[SiH3]",          "cyclohexylsilane"),
    ("C=C[SiH3]",               "ethenylsilane"),
    ("C[AsH2]",                 "methylarsane"),
    ("C[As](C)C",               "trimethylarsane"),
    ("C[SbH2]",                 "methylstibane"),
    ("C[BiH2]",                 "methylbismuthane"),
    ("Cl[Si](C)(C)C",           "chlorotri(methyl)silane"),
    ("Cl[Si](Cl)(C)C",          "dichlorodi(methyl)silane"),
    ("Cl[Si](Cl)(Cl)C",         "trichloro(methyl)silane"),
    ("C[Si](C)=O",              "dimethylsilanone"),
    ("CP=N",                    "1-methylphosphanimine"),
    ("CP(C)(C)=N",              "P,P,P-trimethyl-lambda5-phosphanimine"),
    ("CP(C)(C)=NC",             "P,P,P-trimethyl-N-methyl-lambda5-phosphanimine"),
    ("C[PH2]",                  "methylphosphane"),
    ("C[P](C)C",                "trimethylphosphane"),
    ("[SiH3]P([SiH3])[SiH3]",   "trisilylphosphane"),
    ("CP([SiH3])C",             "dimethyl(silyl)phosphane"),
])
def test_f8_mononuclear_existing_names_are_byte_identical(
    ungated_namer, smiles, expected,
):
    """PROTECT rows across all ten sites and four composers.

    `chlorotri(methyl)silane` / `dichlorodi(methyl)silane` /
    `trichloro(methyl)silane` are the BB-cited mixed halo+organyl forms (BB 35754 /
    39668 / 25866) and pin that the multiplier stayed outside the marks and the
    first group unmarked.  `tert-butylsilane` pins the P-14.5.2 letters-only key.
    """
    assert ungated_namer.name(smiles) == expected


def test_f8_shared_phosphorus_composer_matches_the_bb_shapes():
    """`phosphorus._build_substituent_string` is shared with family 1, so its
    three widened-input defects are pinned directly.

    * BB 25719 `1,4-di(propan-2-yl)cyclohexane` (PIN) -- marks kept, SIMPLE
      multiplier outside  (was `dipropan-2-yl`)
    * P-16.3.3(b)/P-16.2.4.1(d) -- the italicized hyphen survives multiplication
      (was `ditert-butyl`)
    * BB 16286 `tert-butyldi(methyl)phosphane` (PIN) -- multiplier OUTSIDE the
      marks of the SECOND cited group  (was `tert-butyl(dimethyl)`)
    """
    from orthonym.rules.phosphorus import _build_substituent_string as b
    assert b(["propan-2-yl", "propan-2-yl"]) == "di(propan-2-yl)"
    assert b(["tert-butyl", "tert-butyl"]) == "di-tert-butyl"
    assert b(["tert-butyl", "methyl", "methyl"]) == "tert-butyldi(methyl)"
    # protect: the shapes family 1 already shipped must not move
    assert b(["methyl", "methyl", "methyl"]) == "trimethyl"
    assert b(["methyl", "propan-2-yl"]) == "methyl(propan-2-yl)"
    assert b(["methyl", "phenyl"]) == "methyl(phenyl)"
    assert b(["benzyl", "methyl"]) == "benzyl(methyl)"
    assert b(["cyclohexyl", "cyclohexyl"]) == "dicyclohexyl"


def test_f8_io_prefix_composer_keeps_its_documented_cases():
    """`_compose_io_prefixes`' own docstring examples must be byte-identical after
    moving the multiplier outside the marks (BB 7272)."""
    from orthonym.rules.mononuclear_hydrides import _compose_io_prefixes as c
    assert c([("methylimino", True)]) == "(methylimino)"
    assert c([("methyl", False), ("methyl", False),
              ("phenylimino", True)]) == "dimethyl(phenylimino)"
    assert c([("phenyl", False), ("phenyl", False)]) == "diphenyl"
    assert c([("ethyl", False), ("methyl", False)]) == "ethyl(methyl)"
    # widened: the multiplier leaves the marks instead of hiding inside them
    assert c([("propan-2-yl", False), ("propan-2-yl", False),
              ("imino", False)]) == "iminodi(propan-2-yl)"


@pytest.mark.parametrize("smiles,why", [
    ("OCC[SiH3]",       "heteroatom organyl: P-41 seniority keeps it closed"),
    ("CO[Si](C)(C)C",   "a single-bonded heteroatom on the hub is not this family"),
])
def test_f8_other_guards_still_fail_closed(smiles, why):
    """TRIPWIRE.  `_classify_organyls` refuses any stray heteroatom and
    `_classify_mixed_halo_organyl` refuses a non-halogen heteroatom on the hub;
    widening the organyl namer must not enlarge either."""
    from orthonym.rules.mononuclear_hydrides import name_mononuclear_hydride
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    assert name_mononuclear_hydride(mol) is None, why


# ==========================================================================
# The deprecated narrow namer is GONE (all 23 call sites migrated)
# ==========================================================================

def test_the_deprecated_narrow_namer_no_longer_exists():
    """Structural tripwire: `pure_organyl_prefix_name` must stay deleted.

    It was a shim carrying the narrow walker while the families migrated. With
    all 23 call sites moved to `organyl_prefix_name` (naming) or
    `is_simple_unbranched_organyl` (routing) it has no consumer, and a
    zero-caller deprecated shim is exactly how a later session acquires new
    call sites. `_narrow_walk_name` stays private and backs only the predicate.
    """
    import orthonym.rules.substituent_purity as sp
    assert not hasattr(sp, "pure_organyl_prefix_name")
    assert "pure_organyl_prefix_name" not in sp.__all__
    assert sp.__all__ == ["organyl_prefix_name", "is_simple_unbranched_organyl"]
    # the predicate's backing walker must still be there
    assert callable(sp._narrow_walk_name)


def test_no_source_file_calls_the_deprecated_narrow_namer():
    """No CALL to the retired name anywhere under src/ (prose may mention it)."""
    import re
    from pathlib import Path
    import orthonym
    root = Path(orthonym.__file__).parent
    offenders = []
    for py in root.rglob("*.py"):
        for lineno, line in enumerate(py.read_text().splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if re.search(r'\bpure_organyl_prefix_name\s*\(', line) or \
                    re.search(r'import\s+pure_organyl_prefix_name', line):
                offenders.append(f"{py.relative_to(root)}:{lineno}: {stripped}")
    assert not offenders, (
        "the retired narrow namer is referenced again; use "
        "organyl_prefix_name (naming) or is_simple_unbranched_organyl "
        "(routing):\n" + "\n".join(offenders)
    )
