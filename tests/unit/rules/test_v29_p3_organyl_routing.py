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
    because the name already contains parentheses (P-16.3.3 / P-16.5.4.1).
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
# 9. P-16.3.4 / P-29.6.1: a retained ITALICIZED-PREFIX name is cited BARE
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
#   P-16.3.4   'di-tert-butyl' (NOT 'bis(tert-butyl)'), 'N-tert-butyl'
#              (NOT 'N-(tert-butyl)')
#
# Root cause: the carve-out was open-coded in three divergent places and MISSING
# from `needs_brackets`, whose blanket `'-' in name` then won inside the union
# predicate `enclose_if_compound`.  It now lives in exactly ONE primitive.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["tert-butyl", "sec-butyl"])
def test_retained_italicized_prefix_is_not_compound(name):
    """P-16.3.4: the leading italicized prefix's hyphen does not make the name
    compound, so no enclosing marks and the SIMPLE di/tri multiplier."""
    from orthonym.assembly.naming_utils import (
        enclose_if_compound, get_multiplier_prefix, has_structural_hyphen,
        is_complex_substituent, needs_brackets,
    )
    assert has_structural_hyphen(name) is False
    assert needs_brackets(name) is False
    assert is_complex_substituent(name) is False
    assert enclose_if_compound(name) == name      # cited BARE (BB 16286)
    assert get_multiplier_prefix(2, name) == "di"  # di-, not bis- (P-16.3.4)


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
    P-16.3.4 spells out 'N-tert-butyl' NOT 'N-(tert-butyl)'.  All four names are
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
    """`2,6-di-tert-butylphenol` is the P-16.3.4 multiplication witness (BB
    '1,2-di-tert-butylbenzene' (PIN)); both were correct before the carve-out
    moved and must stay byte-identical."""
    assert ungated_namer.name(smiles) == expected
