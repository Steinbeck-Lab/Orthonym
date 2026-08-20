"""Two independent coverage gaps, both measured at ``86e69737``.

    CC(=O)NC1CCCCC1        -> N-cyclohexylacetamide      works
    CC1(CCCCC1)NC(C)=O     -> unknown organic compound   GAP A
    NC(=O)NC1CCCCC1        -> N-cyclohexylurea           works
    NC(=S)NC1CCCCC1        -> unknown organic compound   GAP B

An inherited framing said *"no 1,1-disubstituted cycloalkyl substituent is
nameable in this tree"*.  That is REFUTED by measurement --
``(1-methylcyclohexyl)methanol``, ``1-methylcyclohexan-1-amine``,
``1,1-dimethylcyclohexane`` and ``1-methylcyclohexane-1-carboxylic acid`` all
name correctly, and so does ``N-(4,4-dimethylcyclohexyl)acetamide``.  The
measured boundary is far narrower and is pinned by
``test_boundary_gem_away_from_attachment_always_worked`` below:

    **gem-disubstitution AT THE ATTACHMENT CARBON of a ring substituent.**

GAP A root cause (``rules/ring_substituents.py`` ``decorated_ring_substituent_name``)
------------------------------------------------------------------------------
The function skipped every exocyclic neighbour of the attachment atom::

    parent_nbrs = {nbr.GetIdx() for nbr in
                   mol.GetAtomWithIdx(attachment_atom).GetNeighbors()
                   if nbr.GetIdx() not in ring_set}

Exactly one of those bonds is the free valence to the parent; the rest are
genuine decorations.  On a mono-substituted attachment carbon there is only
one, so skipping "all of them" and skipping "the parent bond" coincide and the
defect is invisible.  On a **gem-disubstituted** attachment carbon the ring's
own substituent is discarded with it, ``atom_prefixes`` comes back empty, and
the ``if not any(atom_prefixes.values())`` guard returns None.  The caller
(``rules/amides.py:_name_n_substituent``) then fell through to
``name_substituent``, which returned the unnameable sentinel ``'substituent'``,
and ``_enrich_ring_n_substituent`` prepended the very decoration that had been
dropped -- emitting ``N-(1-methylsubstituent)acetamide``.  Only the OPSIN
validity gate stopped it, which is why every assertion here is made at the
PRODUCER.

The fix identifies the parent bond STRUCTURALLY -- the exocyclic neighbour that
lies outside the fragment being named -- never by a count or an index.

GAP B root cause (``assembly/composer.py`` ``_enrich_handler_name``)
--------------------------------------------------------------------
``_try_name_thiourea`` already returned the correct ``N-cyclohexylthiourea``
(measured: 4 calls, 4 correct returns).  ``_enrich_handler_name`` then
re-discovered the handler's OWN, already-spelled thiourea core as a ring
substituent and prepended it, yielding
``1-(carbamothioylamino)N-cyclohexylthiourea`` -- the same unit spelled twice,
a different molecule, suppressed by SELF-01.  The exclusion list at
``composer.py:281`` is a hand-maintained tuple that had gone stale: it carries
``'urea'`` but not ``'thiourea'`` (nor ``cyanamide`` / ``imidate`` /
``chalcogen_ester``).  The fix derives the exclusion from the handler's own
``handler_id`` so the list cannot go stale again.

Blue Book, quotations verified by opening ``BlueBookV2/BlueBookV2.md``:

* **P-66.1.6.1.3 "Chalcogen analogues of urea and isourea"** (:33437),
  **P-66.1.6.1.3.1** (:33439): *"Chalcogen analogues of urea are named by
  functional replacement nomenclature using the prefixes 'thio', 'seleno', and
  'telluro'. Preferred IUPAC names use the letter locants N, and N'. Numerical
  locants may be used for thiourea in general nomenclature."*  Restated at
  :33446 *"Numerical locants are no longer used for thiourea in the IUPAC
  preferred name."*  => the ``1,3-``/``1-`` numeral spelling is general
  nomenclature, NOT the PIN.
* The mono-N-substituted class exemplar (:33451):
  **``N-(butan-2-yl)selenourea (PIN)``** -- a singly substituted chalcogen urea
  that KEEPS its ``N`` locant.  ``N-cyclohexylthiourea`` follows it directly.
* **P-29.2** free-valence morphology and **P-14.4** (:3221) criterion (c),
  free valence low, which fixes the attachment carbon of a carbocyclic
  substituent at locant 1 -- hence ``1-methylcyclohexyl``.

OPSIN proves the structure, never the spelling; every expected name below was
checked BOTH ways -- OPSIN 2.9.0 round-trip to an InChIKey identical to the
input's, AND against the governing rule above.
"""

import pytest
from rdkit import Chem


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ring_and_frag(smiles, smarts, ring_smarts=None):
    """Return (mol, ring_tuple, attach_idx, frag_set) for the N-substituent."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return mol


def _n_substituent_fragment(smiles):
    """(mol, ring, attach_idx, frag_set) for the ring borne on the amide/urea N.

    Located structurally: find the N that carries the ring, walk the fragment
    on the far side of that N-C bond.  No atom indices are hard-coded.
    """
    mol = Chem.MolFromSmiles(smiles)
    patt = Chem.MolFromSmarts("[NX3][CX4;R]")
    matches = mol.GetSubstructMatches(patt)
    assert matches, f"no N-C(ring) bond in {smiles}"
    n_idx, attach_idx = matches[0]
    # BFS the substituent fragment, boundary = the nitrogen
    seen, stack = {attach_idx}, [attach_idx]
    while stack:
        cur = stack.pop()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            i = nb.GetIdx()
            if i == n_idx or i in seen or nb.GetAtomicNum() <= 1:
                continue
            seen.add(i)
            stack.append(i)
    ring = next(r for r in mol.GetRingInfo().AtomRings() if attach_idx in r)
    return mol, ring, attach_idx, seen


def _features_for(smiles, fn_name):
    """Capture the real MolecularFeatures handed to a composer builder."""
    import orthonym.assembly.composer as C
    from orthonym import Orthonym as _OS
    n_atoms = Chem.MolFromSmiles(smiles).GetNumAtoms()
    grabbed = []
    orig = getattr(C, fn_name)

    def spy(f):
        if getattr(f, "mol", None) is not None and f.mol.GetNumAtoms() == n_atoms:
            grabbed.append(f)
        return orig(f)

    setattr(C, fn_name, spy)
    try:
        _OS(style="pin").name_tiered(smiles)
    finally:
        setattr(C, fn_name, orig)
    return grabbed[0] if grabbed else None


@pytest.fixture(scope="module")
def namer():
    from orthonym import Orthonym
    return Orthonym(style="pin")


def _name(namer, smiles):
    return namer.name_tiered(smiles)["name"]


# ===========================================================================
# GAP A -- at the producer
# ===========================================================================

def test_gap_a_producer_names_the_gem_disubstituted_attachment_carbon():
    """PRODUCER-level.  ``decorated_ring_substituent_name`` returned None for a
    gem-disubstituted attachment carbon because it skipped the decoration along
    with the parent bond.  Asserted here, not on the shipped name, because the
    OPSIN gate suppressed the wrong output the old path produced.
    """
    from orthonym.rules.ring_substituents import decorated_ring_substituent_name
    mol, ring, attach, frag = _n_substituent_fragment("CC(=O)NC1(C)CCCCC1")
    got = decorated_ring_substituent_name(mol, ring, attach, expected_atoms=frag)
    assert got == "1-methylcyclohexyl", got


def test_gap_a_producer_covers_exactly_the_requested_fragment():
    """The decoration must be COVERED, not dropped: a name that silently omits
    the gem-methyl would describe cyclohexyl, a different molecule.  Proven by
    asking for a fragment that does NOT include the methyl -- it must refuse.
    """
    from orthonym.rules.ring_substituents import decorated_ring_substituent_name
    mol, ring, attach, frag = _n_substituent_fragment("CC(=O)NC1(C)CCCCC1")
    ring_only = set(ring)
    assert ring_only != frag
    assert decorated_ring_substituent_name(
        mol, ring, attach, expected_atoms=ring_only) is None


def test_gap_a_no_sentinel_ever_reaches_a_name():
    """The unnameable sentinel ``'substituent'`` must never be enriched into a
    name.  ``N-(1-methylsubstituent)acetamide`` was built by
    ``rules/amides.py:_name_n_substituent`` treating the sentinel as truthy.
    """
    from orthonym.rules.amides import _name_n_substituent
    mol, ring, attach, frag = _n_substituent_fragment("CC(=O)NC1(C)CCCCC1")
    ordered = [attach] + sorted(frag - {attach})
    got = _name_n_substituent(mol, ordered, sum(
        1 for i in frag if mol.GetAtomWithIdx(i).GetSymbol() == "C"))
    assert got is None or "substituent" not in got, got


def _attachment(smiles):
    mol = Chem.MolFromSmiles(smiles)
    _, att = mol.GetSubstructMatches(Chem.MolFromSmarts("[NX3][CX4;R]"))[0]
    ring = next(r for r in mol.GetRingInfo().AtomRings() if att in r)
    return mol, ring, att


def test_gap_a_fails_closed_when_the_parent_bond_is_undecidable():
    """PRODUCER-level, and it pins a WRONG CONSTITUTION, not just a refusal.

    With no ``expected_atoms`` scope supplied, which exocyclic bond leaves the
    fragment is genuinely undecidable.  Dropping the guard does not merely lose
    a name -- on a ring carrying a SECOND decoration it silently drops the
    gem substituent and emits ``4-methylcyclohexyl`` for a
    **1,4-dimethyl**cyclohexyl fragment, a different molecule.  Measured: with
    the guard removed this exact call returns ``'4-methylcyclohexyl'``.

    Every production caller passes ``expected_atoms``, so this branch is
    defensive; it is asserted here because the signature makes the argument
    optional and a future caller may omit it.
    """
    from orthonym.rules.ring_substituents import decorated_ring_substituent_name
    mol, ring, att = _attachment("CC(=O)NC1(C)CCC(C)CC1")
    assert decorated_ring_substituent_name(mol, ring, att) is None
    # the same call WITH a scope resolves the ambiguity and must still name
    _, _, att2 = _attachment("CC(=O)NC1CCC(C)CC1")
    mol2, ring2, att2 = _attachment("CC(=O)NC1CCC(C)CC1")
    assert decorated_ring_substituent_name(mol2, ring2, att2) == "4-methylcyclohexyl"


@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)NC1(C)CCCCC1",        "N-(1-methylcyclohexyl)acetamide"),
    ("CC1(CCCCC1)NC(C)=O",        "N-(1-methylcyclohexyl)acetamide"),
    ("CC(=O)NC1(C)CCCC1",         "N-(1-methylcyclopentyl)acetamide"),
    ("CC(=O)NC1(C)CCC1",          "N-(1-methylcyclobutyl)acetamide"),
    ("CC(=O)NC1(C)CC1",           "N-(1-methylcyclopropyl)acetamide"),
    ("CC(=O)NC1(C)CCCCCC1",       "N-(1-methylcycloheptyl)acetamide"),
    ("CC(=O)NC1(C)CCCCCCC1",      "N-(1-methylcyclooctyl)acetamide"),
    ("CCC1(CCCCC1)NC(C)=O",       "N-(1-ethylcyclohexyl)acetamide"),
])
def test_gap_a_family_ring_sizes_3_to_8(namer, smiles, expected):
    assert _name(namer, smiles) == expected


def test_boundary_gem_away_from_attachment_always_worked(namer):
    """The measured boundary.  gem-disubstitution AWAY from the attachment
    carbon was never broken -- this is what refutes the inherited
    "no 1,1-disubstituted cycloalkyl is nameable" framing.
    """
    assert _name(namer, "CC(=O)NC1CCC(C)(C)CC1") == \
        "N-(4,4-dimethylcyclohexyl)acetamide"


# ===========================================================================
# GAP B -- at the producer
# ===========================================================================

def test_gap_b_producer_already_worked():
    """``_try_name_thiourea`` was never the defect: it returns the right name.
    Pinning this makes it impossible to "fix" Gap B in the wrong module.
    """
    from orthonym.assembly.composer import _try_name_thiourea
    features = _features_for("NC(=S)NC1CCCCC1", "_try_name_thiourea")
    assert features is not None, "the thiourea builder was never reached"
    assert _try_name_thiourea(features) == "N-cyclohexylthiourea"


def test_gap_b_handler_does_not_enrich_a_complete_name():
    """PRODUCER-level.  ``_try_name_thiourea`` builds a COMPLETE name -- the
    retained parent plus every N-substituent, refusing rather than skipping an
    un-nameable one -- so enrichment can only spell an atom a second time.  It
    re-discovered the handler's own core and prepended
    ``1-(carbamothioylamino)``.  SELF-01 suppressed the result, so asserting on
    the shipped name would have judged the GATE, not the generator.

    The second assertion keeps this from going vacuous: it shows enrichment
    STILL corrupts the name if called, so the handler's skip is load-bearing
    rather than a no-op that would pass either way.
    """
    from orthonym.assembly.composer import _enrich_handler_name
    from orthonym.assembly.handlers.thiourea import name_thiourea
    features = _features_for("NC(=S)NC1CCCCC1", "_try_name_thiourea")
    assert features is not None

    result = name_thiourea(features)
    assert result is not None and result.name == "N-cyclohexylthiourea", result

    corrupted = _enrich_handler_name(
        features, "N-cyclohexylthiourea", "thiourea")
    assert corrupted.startswith("1-(carbamothioylamino)"), corrupted


def test_gap_b_urea_sibling_enrichment_is_unchanged():
    """The oxo sibling was already correct and must stay byte-identical."""
    from orthonym.assembly.composer import _enrich_handler_name
    features = _features_for("NC(=O)NC1CCCCC1", "_try_name_urea")
    assert features is not None
    assert _enrich_handler_name(
        features, "N-cyclohexylurea", "urea") == "N-cyclohexylurea"


@pytest.mark.parametrize("smiles,expected", [
    ("NC(=S)NC1CCCCC1",   "N-cyclohexylthiourea"),
    ("NC(=S)NC1CCCC1",    "N-cyclopentylthiourea"),
    ("NC(=S)NC1CCC1",     "N-cyclobutylthiourea"),
    ("NC(=S)NC1CC1",      "N-cyclopropylthiourea"),
    ("NC(=S)NC1CCCCCC1",  "N-cycloheptylthiourea"),
    ("NC(=S)NC1CCCCCCC1", "N-cyclooctylthiourea"),
])
def test_gap_b_family_ring_sizes_3_to_8(namer, smiles, expected):
    assert _name(namer, smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    # These abstained before: the handler's complete name was enriched with a
    # prefix numbered against a parent (urea) that has no atom 1 or 4 at all
    # -- `1-methylN-(1-methylcyclohexyl)urea` -- which SELF-01 suppressed.
    ("NC(=S)NC1(C)CCCCC1",  "N-(1-methylcyclohexyl)thiourea"),
    ("NC(=O)NC1(C)CCCCC1",  "N-(1-methylcyclohexyl)urea"),
    ("NC(=O)NC1CCC(C)CC1",  "N-(4-methylcyclohexyl)urea"),
    ("NC(=S)NC1CCC(C)CC1",  "N-(4-methylcyclohexyl)thiourea"),
])
def test_decorated_ring_on_a_retained_urea_parent(namer, smiles, expected):
    """The urea/thiourea divergence, reconciled: both families now carry a
    decorated ring N-substituent, and neither cites a numeral against a parent
    that has no numbered skeleton."""
    got = _name(namer, smiles)
    assert got == expected
    assert not got[0].isdigit(), got


@pytest.mark.parametrize("smiles,expected", [
    ("NC#N",                  "cyanamide"),
    ("CC(C)NC#N",             "(propan-2-yl)cyanamide"),
    ("CCN(CC)C#N",            "diethylcyanamide"),
    # REGRESSION GUARD. Excluding the handler's own FG atoms from enrichment
    # (the first attempt at Gap B) truncated the fragment hanging off the ring
    # and turned this correct name into
    # `1-[(1S)-ethyl]((S)-1-cyclohexylethyl)cyanamide`.
    #
    # v33 Engine 4 change-asserted-value: `(S)-` -> `(1S)-`. The carrier chain
    # numbering the prefix cites was not reaching the stereo emitter, so the
    # descriptor shipped unlocanted. Rule (VERIFIED, `BlueBookV2.md:44643`,
    # heading `## **P-91.3** NAMING OF STEREOISOMERS`): a substituent-group
    # stereodescriptor is "preceded by a numerical or letter locant to describe
    # the position of the stereogenic unit *when such locants are present*" --
    # worked `(PIN)` example `[(1R)-1-chloropropyl]benzene`; the unlocanted form
    # is for a name with NO locant (`(R)-bromo(chloro)fluoromethane` (PIN)).
    # `1-cyclohexylethyl` carries locant 1, so `(1S)` is the PIN spelling.
    # OPSIN 2.9.0 parses BOTH spellings to the input's full InChIKey, so this is
    # a pure conformance change (0-wrong-neutral).
    # NOTE the outer `(` is a PRE-EXISTING enclosing-mark defect, unrelated and
    # unchanged: P-16.5.4.1.3 (`BlueBookV2.md:7478`) counts stereo parentheses
    # toward nesting, so P-16.5.4.1.4 requires `[...]` here. `_STEREO_PAREN_RE`
    # STRIPS them (and matches only the unlocanted `(S)`/`(R)`), which is why the
    # `(S)-` spelling never tripped the grammar check and `(1S)-` now does. The
    # name is still emitted per D-11/D-15 and round-trips exactly.
    ("C[C@@H](C1CCCCC1)NC#N", "((1S)-1-cyclohexylethyl)cyanamide"),
    # A ring bonded DIRECTLY to the cyanamide N: enrichment corrupted both of
    # these into abstentions, so they also pin that the handler skips it.
    # (Both round-trip through OPSIN 2.9.0 to the input constitution.)
    ("N#CNC1CCCCC1",          "cyclohexylcyanamide"),
    ("N#CNC1(C)CCCCC1",       "(1-methylcyclohexyl)cyanamide"),
])
def test_cyanamide_complete_names_are_not_enriched(namer, smiles, expected):
    assert _name(namer, smiles) == expected


def test_retained_tert_butyl_never_names_a_ring():
    """PRODUCER-level. ``tert-butyl`` is ACYCLIC (P-29.6.1, BB:16196/:16286).

    The retained matcher accepted any fragment with 4 carbons, 3 carbon
    neighbours at the attachment atom, and no heteroatom -- and
    **1-methylcyclopropyl** satisfies all three (its attachment carbon has two
    RING neighbours plus the methyl). So `NC(=O)NC1(C)CC1` was named
    `N-tert-butylurea`: a cyclopropane ring opened into a chain, a different
    molecule with the same C4H9 formula, caught only by SELF-01.

    Exactly the class of
     -- the same
    matcher had already been hardened once, against heteroatoms, without
    anyone adding a ring check.
    """
    from orthonym.assembly.substituent_naming import _check_retained_substituent
    from orthonym.assembly.substituent_enumerator import name_substituent
    mol = Chem.MolFromSmiles("NC(=O)NC1(C)CC1")
    frag = [4, 5, 6, 7]                       # 1-methylcyclopropyl, attach = 4
    assert mol.GetAtomWithIdx(4).IsInRing()
    assert _check_retained_substituent(mol, frag, 4) != "tert-butyl"
    assert name_substituent(mol, set(frag), 4) == "1-methylcyclopropyl"
    # the genuine acyclic tert-butyl must still be retained
    tb = Chem.MolFromSmiles("CC(C)(C)NC(=O)N")
    assert _check_retained_substituent(tb, [0, 1, 2, 3], 1) == "tert-butyl"


@pytest.mark.parametrize("smiles,expected", [
    ("NC(=O)NC1(C)CC1",  "N-(1-methylcyclopropyl)urea"),
    ("NC(=S)NC1(C)CC1",  "N-(1-methylcyclopropyl)thiourea"),
    ("CC(C)(C)NC(=O)N",  "N-tert-butylurea"),
    ("CC(C)(C)NC(=S)N",  "N-tert-butylthiourea"),
])
def test_cyclopropyl_is_not_flattened_to_tert_butyl(namer, smiles, expected):
    assert _name(namer, smiles) == expected


def test_gap_b_letter_locant_not_a_numeral(namer):
    """P-66.1.6.1.3.1 (:33439/:33446): numerals are general nomenclature only."""
    got = _name(namer, "NC(=S)NC1CCCCC1")
    assert got.startswith("N-"), got
    assert not got[0].isdigit(), got


# ===========================================================================
# The retained parent has NO numbered skeleton
# ===========================================================================

def test_retained_urea_parent_rejects_a_numeric_front_of_name_stereo_block():
    """PRODUCER-level, on the shared stereo owner.

    ``urea``/``thiourea``/``cyanamide`` have no numbered skeleton -- their only
    locants are the italic letters *N* / *N*' (P-66.1.6.1.3.1, BB:33439; :33446
    "Numerical locants are no longer used for thiourea in the IUPAC preferred
    name").  A front-of-name block citing 1/3/5 therefore denotes nothing in
    that parent (P-14.3.3 "Citation of locants", BB:2869) and OPSIN cannot
    parse it.

    Measured over the 144 stereo-bearing urea/thiourea rows of pubchem_2000 +
    chebi_5000: such a block is 4/4 CORRECT on parents that DO have a numbered
    skeleton, and 0/2 on the retained parent.
    """
    from orthonym.assembly.composer import _inject_stereo_if_missing
    features = _features_for("C[C@@H]([C@H]1C[C@@H]2CC[C@H]1C2)NC(=S)NCC=C",
                             "_try_name_thiourea")
    assert features is not None
    base = "N-[1-(bicyclo[2.2.1]heptan-2-yl)ethyl]-N'-(prop-2-en-1-yl)thiourea"
    # the scope this handler declares must leave the name alone ...
    assert _inject_stereo_if_missing(
        features, base, atom_to_locant=None,
        parent_scope='retained_no_locants') == base
    # ... and the undeclared scope is what used to prepend the bad block, so the
    # test would be vacuous if it could not still be produced.
    assert _inject_stereo_if_missing(
        features, base, atom_to_locant=None).startswith("(1S,3R,5S)-")


@pytest.mark.parametrize("smiles,expected", [
    # stereo lives in the N-substituent and is cited THERE -- unchanged
    # v33 Engine 4 change-asserted-value: `(S)-` -> `(1S)-`, same P-91.3
    # citation as the cyanamide row above (`BlueBookV2.md:44643`; `(PIN)`
    # example `[(1R)-1-chloropropyl]benzene`). The property this row exists to
    # pin -- the descriptor stays INSIDE the N-substituent bracket and no locant
    # is hung on the retained parent -- is unchanged, and the `not
    # got.startswith("(")` assertion below still holds. Both spellings
    # round-trip through OPSIN 2.9.0 to the input's full InChIKey.
    ("C[C@@H](C1CCCCC1)NC(=S)NCC=C",
     "N-[(1S)-1-cyclohexylethyl]-N'-(prop-2-en-1-yl)thiourea"),
    # a substituent whose stereo the prefix namer cannot carry: the constitution
    # is still right and no locant that denotes nothing is cited
    ("C[C@@H]([C@H]1C[C@@H]2CC[C@H]1C2)NC(=S)NCC=C",
     "N-[1-(bicyclo[2.2.1]heptan-2-yl)ethyl]-N'-(prop-2-en-1-yl)thiourea"),
    ("C[C@@H]([C@H]1C[C@@H]2CC[C@H]1C2)NC(=O)NCC=C",
     "N-[1-(bicyclo[2.2.1]heptan-2-yl)ethyl]-N'-(prop-2-en-1-yl)urea"),
])
def test_no_unresolvable_locant_on_a_retained_parent(namer, smiles, expected):
    got = _name(namer, smiles)
    assert got == expected
    assert not got.startswith("("), got


# ===========================================================================
# Controls -- must stay byte-identical
# ===========================================================================

@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)NC1CCCCC1",   "N-cyclohexylacetamide"),
    ("NC(=O)NC1CCCCC1",   "N-cyclohexylurea"),
    ("CC(=O)NC",          "N-methylacetamide"),
    ("CC(=O)NC(C)C",      "N-(propan-2-yl)acetamide"),
    ("NC(=S)N",           "thiourea"),
    ("CC1(C)CCCCC1",      "1,1-dimethylcyclohexane"),
])
def test_controls_unchanged(namer, smiles, expected):
    assert _name(namer, smiles) == expected
