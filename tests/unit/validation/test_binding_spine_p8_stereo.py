"""Phase 0c Task 3: P8 -- atom-indexed stereo completeness/correctness.

Every fixture here is atom-indexed on purpose: the molecule's real CIP labels
are computed once (via ``collect_stereodescriptors``/``terminal_fragment_name``,
the SAME functions the real producers use), then the test name string is
hand-corrupted to prove P8 resolves through ``stereo_atom_to_locant`` to real
atom/bond identity rather than counting descriptors in the string. Test (b)
in particular -- same COUNT of descriptors, wrong locant/atom assignment -- is
the one a pure cardinality scan (the forbidden anti-pattern,
``count_expressed_stereo_descriptors``) could never catch.
"""
import pytest
from rdkit import Chem

from orthonym.rules.stereochemistry import collect_stereodescriptors
from orthonym.rules.terminal_fragment import terminal_fragment_name
from orthonym.validation import binding_spine as bs

pytestmark = pytest.mark.unit


def _b(token, kind, atoms, **kw):
    return bs.SpineBinding(token=token, kind=kind,
                           atom_ids=frozenset(atoms), **kw)


def _spine(*roots, stereo_atom_to_locant=None):
    return bs.BindingSpine(roots=tuple(roots),
                           stereo_atom_to_locant=dict(stereo_atom_to_locant or {}))


# ---------------------------------------------------------------------------
# Parent-scope R/S fixture: C[C@H](Cl)[C@H](Br)C, backbone atoms 0,1,3,5 ->
# locants 1,2,3,4. Ground truth computed via collect_stereodescriptors (the
# same function _stereo_prefix uses), never hand-guessed.
# ---------------------------------------------------------------------------
_MOL = Chem.MolFromSmiles("C[C@H](Cl)[C@H](Br)C")
_ATOM_TO_LOCANT = {0: 1, 1: 2, 3: 3, 5: 4}
_REAL = collect_stereodescriptors(_MOL, _ATOM_TO_LOCANT)  # [(2, 'S'), (3, 'R')]


def _parent_scope_spine(name: str) -> bs.BindingSpine:
    return _spine(
        _b("but", bs.BindingKind.PARENT, [0, 1, 3, 5]),
        _b("chloro", bs.BindingKind.PREFIX, [2]),
        _b("bromo", bs.BindingKind.PREFIX, [4]),
        stereo_atom_to_locant=_ATOM_TO_LOCANT,
    )


def test_real_ground_truth_is_two_distinct_cip_letters():
    """Guard the fixture itself: the swap test below is meaningless if the
    two real centres happen to share one CIP letter."""
    assert _REAL == [(2, 'S'), (3, 'R')]


def test_a_omitted_descriptor_is_stereo_missing():
    """(a): a candidate binding a stereo-bearing PARENT token but omitting
    the descriptor block entirely -> P8 forward finding."""
    p = bs.verify_spine(_MOL, _parent_scope_spine("3-bromo-2-chlorobutane"),
                        "3-bromo-2-chlorobutane", mode="audit")
    assert bs.STEREO_DESCRIPTOR_MISSING in p.codes()
    assert bs.STEREO_UNVERIFIED not in p.codes()  # map WAS threaded


def test_b_wrong_descriptor_right_count_wrong_atom_is_mismatch():
    """(b): the emitted block has the RIGHT COUNT (2) of descriptors but the
    CIP letters are swapped between the two real locants -- a pure
    cardinality scan (2 emitted == 2 defined) would report this clean. P8
    must not, because it resolves each pair to real atom identity."""
    name = "(2R,3S)-3-bromo-2-chlorobutane"  # real is (2S,3R): fully swapped
    p = bs.verify_spine(_MOL, _parent_scope_spine(name), name, mode="audit")
    codes = p.codes()
    assert bs.STEREO_DESCRIPTOR_MISMATCH in codes, codes
    assert bs.STEREO_DESCRIPTOR_MISSING in codes, codes
    # same total descriptor count both sides -- the point of the test
    assert len([c for c in codes if c == bs.STEREO_DESCRIPTOR_MISMATCH]) == 2
    assert len([c for c in codes if c == bs.STEREO_DESCRIPTOR_MISSING]) == 2


def test_c_complete_correct_candidate_has_no_p8_finding():
    """(c): the real, correctly-ordered descriptor block -> no P8 finding at
    all (forward or reverse)."""
    name = "(2S,3R)-3-bromo-2-chlorobutane"
    p = bs.verify_spine(_MOL, _parent_scope_spine(name), name, mode="audit")
    codes = p.codes()
    assert bs.STEREO_DESCRIPTOR_MISSING not in codes
    assert bs.STEREO_DESCRIPTOR_MISMATCH not in codes
    assert bs.STEREO_UNVERIFIED not in codes


def test_untreaded_map_with_leading_block_is_unverified_not_wrong():
    """An empty ``stereo_atom_to_locant`` (a producer that never threaded one,
    e.g. the terminal-ring bare-parent tier) must NEVER be read as "wrong" --
    there is no ground truth to compare against, so P8 reports the weaker,
    unproven-not-disproven ``STEREO_UNVERIFIED`` (mirrors P3's
    ``CHARGE_UNVERIFIED``), not a confident MISMATCH."""
    name = "(2S,3R)-3-bromo-2-chlorobutane"
    spine = _spine(
        _b("but", bs.BindingKind.PARENT, [0, 1, 3, 5]),
        _b("chloro", bs.BindingKind.PREFIX, [2]),
        _b("bromo", bs.BindingKind.PREFIX, [4]),
    )  # no stereo_atom_to_locant
    p = bs.verify_spine(_MOL, spine, name, mode="audit")
    codes = p.codes()
    assert bs.STEREO_UNVERIFIED in codes
    assert bs.STEREO_DESCRIPTOR_MISSING not in codes
    assert bs.STEREO_DESCRIPTOR_MISMATCH not in codes


def test_no_leading_block_and_no_map_is_silent():
    """An achiral name with no threaded map and no descriptor block: P8 has
    nothing to say (not even STEREO_UNVERIFIED -- there is no block to be
    unverified about)."""
    mol = Chem.MolFromSmiles("CCCC")
    spine = _spine(_b("but", bs.BindingKind.PARENT, [0, 1, 2, 3]))
    p = bs.verify_spine(mol, spine, "butane", mode="audit")
    stereo_codes = [c for c in p.codes() if c.startswith("STEREO")]
    assert stereo_codes == []


# ---------------------------------------------------------------------------
# (d) Second check: a PREFIX binding's own embedded (nE)/(nZ) block, using a
# REAL embedded-E/Z witness (the shipped v30 internal-C=C acyl lever):
# C/C=C/C(=O)N -> terminal_fragment_name -> "(3E)-2-oxo-1-azapent-3-en-1-yl".
# ---------------------------------------------------------------------------
_EZ_MOL = Chem.MolFromSmiles("C/C=C/C(=O)N")
_EZ_FRAG = frozenset(range(_EZ_MOL.GetNumAtoms()))
_EZ_REAL = terminal_fragment_name(_EZ_MOL, set(_EZ_FRAG), 5)


def test_real_embedded_ez_witness_shape():
    assert _EZ_REAL is not None
    assert _EZ_REAL.name == "(3E)-2-oxo-1-azapent-3-en-1-yl"


def _ez_spine(token: str) -> bs.BindingSpine:
    return _spine(_b(token, bs.BindingKind.PREFIX, _EZ_FRAG))


def test_d_embedded_ez_correct_token_has_no_finding():
    p = bs.verify_spine(_EZ_MOL, _ez_spine(_EZ_REAL.name), _EZ_REAL.name,
                        mode="audit")
    codes = [c for c in p.codes() if c.startswith("SUBSTITUENT_STEREO")]
    assert codes == []


def test_d_embedded_ez_dropped_descriptor_is_missing():
    stripped = "2-oxo-1-azapent-3-en-1-yl"  # real token minus the (3E)- block
    p = bs.verify_spine(_EZ_MOL, _ez_spine(stripped), stripped, mode="audit")
    assert bs.SUBSTITUENT_STEREO_MISSING in p.codes()


def test_d_embedded_ez_wrong_letter_is_mismatch():
    wrong = "(3Z)-2-oxo-1-azapent-3-en-1-yl"  # real bond is E, not Z
    p = bs.verify_spine(_EZ_MOL, _ez_spine(wrong), wrong, mode="audit")
    assert bs.SUBSTITUENT_STEREO_MISMATCH in p.codes()


def test_d_fabricated_descriptor_with_no_real_bond_is_mismatch():
    """A token claiming (nE)/(nZ) over an atom set with NO internal defined
    E/Z bond at all -- the fabrication case, not merely a wrong letter."""
    saturated = Chem.MolFromSmiles("CCCC(=O)N")
    frag = frozenset(range(saturated.GetNumAtoms()))
    fabricated = "(3E)-2-oxo-1-azapentyl"
    p = bs.verify_spine(saturated, _spine(
        _b(fabricated, bs.BindingKind.PREFIX, frag)), fabricated, mode="audit")
    assert bs.SUBSTITUENT_STEREO_MISMATCH in p.codes()


def test_p8_findings_are_warn_severity_in_audit_mode():
    """Part C: audit-only this task -- P8 findings must never flip ``ok`` to
    False under mode='audit' (the mode every production call site uses), or
    T4 emissions would change this round."""
    name = "3-bromo-2-chlorobutane"  # descriptor dropped -> real P8 finding
    p = bs.verify_spine(_MOL, _parent_scope_spine(name), name, mode="audit")
    stereo_findings = [f for f in p.findings if f.code.startswith("STEREO")]
    assert stereo_findings, "fixture must actually raise a P8 finding"
    assert all(f.severity == "warn" for f in stereo_findings)
    assert p.ok, "a warn-severity finding must never flip ok to False"


def test_p8_findings_escalate_to_error_in_strict_mode():
    name = "3-bromo-2-chlorobutane"
    p = bs.verify_spine(_MOL, _parent_scope_spine(name), name, mode="strict")
    stereo_findings = [f for f in p.findings if f.code.startswith("STEREO")]
    assert stereo_findings
    assert all(f.severity == "error" for f in stereo_findings)
    assert not p.ok


# ---------------------------------------------------------------------------
# Phase 0c Task 4: the 8 dev500 false positives Task 3's diagnostic found,
# root-caused to 3 bugs in P8's OWN parser (never a molecule defect). Each
# fixture below reproduces the BUG SHAPE via a hand-built minimal molecule
# (the file's own established convention -- ground truth computed via
# ``collect_stereodescriptors``/``terminal_fragment_name``, never hand-
# guessed), with the real dev500 witness (SMILES + emitted name) quoted for
# traceability. All 8 witnesses were re-verified directly against these fixes
# by a dev500 best-effort diagnostic sweep before landing (0 confident P8
# findings, down from 8; see the Task 4 report).
# ---------------------------------------------------------------------------

def test_p8b_ignores_embedded_rs_descriptor_not_its_scope():
    """Root cause 1 (6 of the 8 false positives): a PREFIX token embedding a
    legitimate R/S descriptor from a DIFFERENT mechanism (a substituent's own
    atom stereocentre, not P8b's concern -- P8b's only ground truth is a
    binding's own internal E/Z BOND) must raise nothing.

    Real witness: ``CC(C)CCC[C@@H](C)[C@H]1CC...`` ->
    ``...14-((2R)-6-methylheptan-2-yl)...tetracyclo[...]...`` --
    ``_p8b_substituent_stereo`` used to reuse the R/S-admitting
    ``_parse_leading_stereo_block`` and misread the embedded ``(2R)`` as a
    fabricated E/Z claim (``SUBSTITUENT_STEREO_MISMATCH``) even though the
    token's own claimed atoms carry NO internal E/Z bond at all. Reproduced
    here on ``_MOL`` (real R/S, real CIP letters via ``_REAL``, zero E/Z
    bonds), embedding its own real (2, 'S') pair in a PREFIX token's text.
    """
    locant, cip = _REAL[0]  # (2, 'S') -- a REAL centre on this molecule
    token = f"({locant}{cip})-3-bromo-2-chlorobutyl"
    atoms = frozenset(range(_MOL.GetNumAtoms()))
    spine = _spine(_b(token, bs.BindingKind.PREFIX, atoms))
    p = bs.verify_spine(_MOL, spine, token, mode="audit")
    codes = [c for c in p.codes() if c.startswith("SUBSTITUENT_STEREO")]
    assert codes == [], codes


def test_p8b_finds_embedded_ez_mid_token_not_only_at_index_zero():
    """Root cause 2 (1 of the 8): a real embedded ``(nE)/(nZ)`` block sitting
    AFTER other text in a large flat composite PREFIX token (a nested
    sub-fragment) must still be found.

    Real witness: ``C/C=C/C1=CC2=...`` ->
    ``...6-(1-oxooctyl)-11-[(1E)-prop-1-en-1-yl]-4-oxa-12-azatricyclo[...]``
    -- the real, correctly-expressed ``(1E)`` sat mid-token behind
    ``6-(1-oxooctyl)-11-[...``, past index 0, so the old anchored parse
    reported the genuinely-present descriptor as MISSING. Reproduced here by
    prefixing the real embedded-E/Z witness token with unrelated text.
    """
    token = f"6-(1-oxooctyl)-{_EZ_REAL.name}"
    spine = _spine(_b(token, bs.BindingKind.PREFIX, _EZ_FRAG))
    p = bs.verify_spine(_EZ_MOL, spine, token, mode="audit")
    codes = [c for c in p.codes() if c.startswith("SUBSTITUENT_STEREO")]
    assert codes == [], codes


def test_p8a_finds_leading_block_after_functional_class_word():
    """Root cause 3 (1 of the 8): a functional-class TWO-WORD ester name
    (P-65.6.3.2.1) puts the ester alkyl group's own word before the parent's
    leading descriptor block, so the position-0 anchor finds nothing and
    P8a misreported every real centre as MISSING.

    Real witness: ``CC[C@]12C=CCN3CC[C@]4(...)...`` ->
    ``methyl (1R,12R,19S)-1-ethyl-5,15-diazapentacyclo[...]...-3-carboxylate``.
    Reproduced on the existing parent-scope fixture with a leading alkyl
    word prepended to its own real, correct descriptor block.
    """
    name = "methyl (2S,3R)-3-bromo-2-chlorobutanoate"
    p = bs.verify_spine(_MOL, _parent_scope_spine(name), name, mode="audit")
    codes = p.codes()
    assert bs.STEREO_DESCRIPTOR_MISSING not in codes, codes
    assert bs.STEREO_DESCRIPTOR_MISMATCH not in codes, codes


def test_p8a_still_flags_wrong_letters_after_functional_class_word():
    """The two-word fix must not become a free pass: a two-word name whose
    block is genuinely wrong still gets caught."""
    name = "methyl (2R,3S)-3-bromo-2-chlorobutanoate"  # swapped vs real (2S,3R)
    p = bs.verify_spine(_MOL, _parent_scope_spine(name), name, mode="audit")
    codes = p.codes()
    assert bs.STEREO_DESCRIPTOR_MISMATCH in codes, codes
    assert bs.STEREO_DESCRIPTOR_MISSING in codes, codes


def test_p8a_selects_parent_block_when_first_word_has_its_own_stereo():
    """FIX-ROUND regression (reviewer-found, real end-to-end reproduction):
    a functional-class two-word name whose FIRST word is not a bare alkyl
    identifier but a fully general substituent carrying ITS OWN leading
    ``(nR)/(nS)`` block -- e.g. a chiral ester alkyl group -- must not be
    mistaken for the parent's block. "First non-empty block found" used to
    return the alkyl word's own descriptor and never reach the parent's
    real block, reporting all 3 real parent centres MISSING and (once
    escalated) voiding a fully correct T4 candidate.

    Real witness:
    ``CC[C@]12C=CCN3CC[C@]4(C(=C(C(=O)O[C@@H](C)CC)C1)Nc1ccccc14)[C@@H]32``
    -> ``(2S)-butan-2-yl (1R,12R,19S)-12-ethyl-8,16-diazapentacyclo[...]
    ...-10-carboxylate``. Reproduced on the existing parent-scope fixture:
    a foreign ``(5R)-pentyl`` word (a locant/CIP that matches NEITHER real
    parent centre, so a naive first-match would be unambiguously wrong)
    precedes the parent's own real, correct block.
    """
    name = "(5R)-pentyl (2S,3R)-3-bromo-2-chlorobutanoate"
    p = bs.verify_spine(_MOL, _parent_scope_spine(name), name, mode="audit")
    codes = p.codes()
    assert bs.STEREO_DESCRIPTOR_MISSING not in codes, codes
    assert bs.STEREO_DESCRIPTOR_MISMATCH not in codes, codes


def test_p8a_not_blinded_by_foreign_word_when_parent_block_is_wrong():
    """Negative control for the same shape: the PARENT's block (not the
    foreign alkyl word's) is genuinely wrong -- P8 must still catch it,
    proving the identity-based selection locks onto the parent's own block
    rather than merely refusing to be fooled by the alkyl word."""
    name = "(5R)-pentyl (2S,3S)-3-bromo-2-chlorobutanoate"  # real is (2S,3R)
    p = bs.verify_spine(_MOL, _parent_scope_spine(name), name, mode="audit")
    codes = p.codes()
    assert bs.STEREO_DESCRIPTOR_MISMATCH in codes, codes
    assert bs.STEREO_DESCRIPTOR_MISSING in codes, codes


def test_p8a_not_masked_when_parent_block_is_fabricated_and_foreign_word_overlaps():
    """Task 4 FIX-ROUND 2 (reviewer-found, MORE SEVERE than the round-1
    regression): the round-1 fix selected the candidate block by MAXIMUM
    OVERLAP against ``expected_set``. That is exploitable: locants are
    small ints and CIP is binary, so a foreign word can coincidentally
    share a real ``(locant, cip)`` pair with the parent's true centres. If
    the parent's OWN block is genuinely fabricated but scores WORSE on
    overlap than a foreign word's, overlap-based selection would pick the
    foreign word and the genuine ``STEREO_DESCRIPTOR_MISMATCH`` would never
    fire -- SILENTLY HIDING a wrong name under the gate that is supposed to
    BE the 0-wrong backstop (strictly worse than round 1's bug, which only
    voided a correct name into a safe abstain).

    Reviewer's exact witness: real centres are ``{(2,'S'),(3,'R')}``; the
    parent's own block ``(4S,5R)`` is completely fabricated (0 overlap);
    the foreign ``pentyl`` word's own block is ``(2S)`` (1 point of
    overlap, since locant 2 CIP 'S' happens to be one of the real pairs).
    Overlap-based selection would return the foreign ``(2S)`` and miss the
    fabrication entirely. The fix (positional anchoring to the PARENT
    binding's own P4 span) must select the PARENT's ``(4S,5R)`` regardless
    of which candidate scores better, and correctly report the
    fabrication.
    """
    name = "(2S)-pentyl (4S,5R)-3-bromo-2-chlorobutanoate"
    p = bs.verify_spine(_MOL, _parent_scope_spine(name), name, mode="audit",
                        escalate=bs.STRICT_STEREO_CHARGE_AXES)
    codes = p.codes()
    assert bs.STEREO_DESCRIPTOR_MISMATCH in codes, codes
    assert bs.STEREO_DESCRIPTOR_MISSING in codes, codes
    assert not p.ok, "a fabricated parent block must not be masked"


# ---------------------------------------------------------------------------
# Task 4 FIX-ROUND 3 (re-review-found residual, SAFE-SIDE false-abstain): a
# chiral resolving group esterified to an ACHIRAL acid -- round 2's
# "rightmost candidate at or before the parent's span" rule wrongly picked
# the pentyl word's own descriptor for the achiral parent (it was the ONLY
# candidate anywhere, since the achiral parent has none of its own), raising
# a false STEREO_DESCRIPTOR_MISMATCH on a fully correct name.
# ---------------------------------------------------------------------------
_ACHIRAL_MOL = Chem.MolFromSmiles("CCCC")  # butane -- no real stereocentres
_ACHIRAL_ATOM_TO_LOCANT = {0: 1, 1: 2, 2: 3, 3: 4}


def test_p8a_no_false_mismatch_when_achiral_parent_precedes_by_foreign_word():
    """The reviewer's exact shape: ``"(2S)-pentyl butanoate"``. The parent
    (``butanoate``, achiral -- ``expected_set`` is empty) has no leading
    block of its own; the ONLY block anywhere in the string is the pentyl
    word's own ``(2S)``. ADJACENCY selection must recognise this block does
    NOT sit at the parent's own word boundary and return ``[]`` for the
    parent -- an empty ``emitted`` against an empty ``expected_set`` is a
    clean pass, not a fabricated mismatch.
    """
    name = "(2S)-pentyl butanoate"
    spine = _spine(
        _b("but", bs.BindingKind.PARENT, [0, 1, 2, 3]),
        stereo_atom_to_locant=_ACHIRAL_ATOM_TO_LOCANT,
    )
    p = bs.verify_spine(_ACHIRAL_MOL, spine, name, mode="audit",
                        escalate=bs.STRICT_STEREO_CHARGE_AXES)
    codes = [c for c in p.codes() if c.startswith("STEREO")]
    assert codes == [], codes
    assert p.ok, p.findings


def test_p8a_still_flags_genuine_missing_when_parent_has_centres_but_no_block():
    """Negative control: ``[]`` for "no block at the parent's own word" must
    NOT become a blanket pass when the parent DOES have real centres --
    only when it genuinely has none. Real centres are
    ``{(2,'S'),(3,'R')}``; the name has a foreign ``(5R)-pentyl`` word but
    NO block at all before the parent's own ``3-bromo-2-chlorobutanoate``
    word -- the parent's stereo was genuinely dropped, and P8a must still
    catch it (both centres MISSING)."""
    name = "(5R)-pentyl 3-bromo-2-chlorobutanoate"
    p = bs.verify_spine(_MOL, _parent_scope_spine(name), name, mode="audit",
                        escalate=bs.STRICT_STEREO_CHARGE_AXES)
    codes = p.codes()
    assert codes.count(bs.STEREO_DESCRIPTOR_MISSING) == 2, codes
    assert bs.STEREO_DESCRIPTOR_MISMATCH not in codes, codes
    assert not p.ok


def test_p8a_ambiguous_when_parent_span_unresolvable():
    """Fail-safe branch: when the PARENT binding's own token cannot be
    located among P4's resolved spans (e.g. a spine whose parent token
    text does not actually occur in ``name``) AND there are 2+ candidate
    blocks, P8a must never guess -- it reports
    ``STEREO_PARENT_BLOCK_AMBIGUOUS`` (unproven, not disproven, exactly
    like ``STEREO_UNVERIFIED``) rather than silently picking one."""
    name = "(2S)-pentyl (2S,3R)-3-bromo-2-chlorobutanoate"  # real block, correct
    spine = _spine(
        _b("nonexistent-parent-token", bs.BindingKind.PARENT, [0, 1, 3, 5]),
        _b("chloro", bs.BindingKind.PREFIX, [2]),
        _b("bromo", bs.BindingKind.PREFIX, [4]),
        stereo_atom_to_locant=_ATOM_TO_LOCANT,
    )
    p = bs.verify_spine(_MOL, spine, name, mode="audit")
    assert bs.STEREO_PARENT_BLOCK_AMBIGUOUS in p.codes()
    assert bs.STEREO_DESCRIPTOR_MISSING not in p.codes()
    assert bs.STEREO_DESCRIPTOR_MISMATCH not in p.codes()

    p_strict = bs.verify_spine(_MOL, spine, name, mode="strict")
    ambiguous = [f for f in p_strict.findings
                if f.code == bs.STEREO_PARENT_BLOCK_AMBIGUOUS]
    assert ambiguous and all(f.severity == "error" for f in ambiguous)
    assert not p_strict.ok


def test_p8b_still_flags_genuine_mismatch_after_the_ez_only_fix():
    """The R/S-exclusion fix must not blind P8b to a genuine E/Z fabrication
    (the pre-existing (d) tests already cover this; this pins it survives
    alongside the new ``_iter_embedded_ez_pairs`` scan)."""
    wrong = "(3Z)-2-oxo-1-azapent-3-en-1-yl"  # real bond is E, not Z
    p = bs.verify_spine(_EZ_MOL, _ez_spine(wrong), wrong, mode="audit")
    assert bs.SUBSTITUENT_STEREO_MISMATCH in p.codes()


# ---------------------------------------------------------------------------
# Task 3 review Minor: a single-centre P-14.3.4 locant-omitted descriptor
# (``(R)-...`` instead of ``(2R)-...``) must resolve correctly, not read as
# BOTH a missing real pair and a mismatched fabricated one. Latent today
# (the formatter always emits the locant) but must not be a landmine for the
# next producer that legally omits it.
# ---------------------------------------------------------------------------
_SINGLE_MOL = Chem.MolFromSmiles("C[C@H](Cl)Br")
_SINGLE_ATOM_TO_LOCANT = {0: 1, 1: 2}
_SINGLE_REAL = collect_stereodescriptors(_SINGLE_MOL, _SINGLE_ATOM_TO_LOCANT)


def _single_centre_spine(stereo_map=None) -> bs.BindingSpine:
    return _spine(
        _b("ethane", bs.BindingKind.PARENT, [0, 1]),
        _b("chloro", bs.BindingKind.PREFIX, [2]),
        _b("bromo", bs.BindingKind.PREFIX, [3]),
        stereo_atom_to_locant=(stereo_map if stereo_map is not None
                               else _SINGLE_ATOM_TO_LOCANT),
    )


def test_real_single_centre_fixture_has_exactly_one_descriptor():
    """Guard the fixture: the omission test below is meaningless on a
    molecule with zero or more than one real stereocentre."""
    assert len(_SINGLE_REAL) == 1


def test_p8_locant_omitted_single_centre_descriptor_resolves():
    _, cip = _SINGLE_REAL[0]
    name = f"({cip})-1-bromo-1-chloroethane"
    p = bs.verify_spine(_SINGLE_MOL, _single_centre_spine(), name,
                        mode="audit")
    codes = p.codes()
    assert bs.STEREO_DESCRIPTOR_MISSING not in codes, codes
    assert bs.STEREO_DESCRIPTOR_MISMATCH not in codes, codes


def test_p8_locant_omitted_wrong_letter_still_mismatches():
    """The omission normalisation must only forgive a MATCHING CIP letter --
    a bare descriptor with the WRONG letter is still a genuine over-claim
    and must still be caught (both halves: the real centre undischarged,
    and the fabricated pair unresolved)."""
    _, cip = _SINGLE_REAL[0]
    wrong_cip = "S" if cip == "R" else "R"
    name = f"({wrong_cip})-1-bromo-1-chloroethane"
    p = bs.verify_spine(_SINGLE_MOL, _single_centre_spine(), name,
                        mode="audit")
    codes = p.codes()
    assert bs.STEREO_DESCRIPTOR_MISMATCH in codes, codes
    assert bs.STEREO_DESCRIPTOR_MISSING in codes, codes


# ---------------------------------------------------------------------------
# Phase 0c Task 4 Part C: the stereo+charge axis promotion mechanism.
# ``escalate=STRICT_STEREO_CHARGE_AXES`` must force P8's codes to "error"
# under ``mode="audit"`` (the mode the T4 wiring actually keeps), without
# needing a full ``mode="strict"`` flip.
# ---------------------------------------------------------------------------

def test_escalate_promotes_p8_findings_to_error_under_audit_mode():
    name = "3-bromo-2-chlorobutane"  # descriptor dropped -> real P8 finding
    p = bs.verify_spine(_MOL, _parent_scope_spine(name), name, mode="audit",
                        escalate=bs.STRICT_STEREO_CHARGE_AXES)
    stereo_findings = [f for f in p.findings if f.code.startswith("STEREO")]
    assert stereo_findings
    assert all(f.severity == "error" for f in stereo_findings)
    assert not p.ok


def test_escalate_leaves_correct_names_passing():
    """The escalation must never turn a CORRECT name's warn-free P8 pass
    into a failure -- it only promotes severities on findings that already
    fired; a clean proof has none to promote."""
    name = "(2S,3R)-3-bromo-2-chlorobutane"
    p = bs.verify_spine(_MOL, _parent_scope_spine(name), name, mode="audit",
                        escalate=bs.STRICT_STEREO_CHARGE_AXES)
    assert p.ok, p.findings


def test_escalate_never_promotes_substituent_stereo_unverified_info():
    """The ambiguous-multiplicity ``SUBSTITUENT_STEREO_UNVERIFIED`` code is
    deliberately excluded from ``STRICT_STEREO_CHARGE_AXES`` -- it stays
    "info" even escalated, because disambiguating multiplicity >=2 is not
    resolvable here regardless of mode (never confidently wrong)."""
    assert bs.SUBSTITUENT_STEREO_UNVERIFIED not in bs.STRICT_STEREO_CHARGE_AXES
    mol = Chem.MolFromSmiles("CCCCCCC")  # heptane -- no real E/Z bonds at all
    frag = frozenset(range(mol.GetNumAtoms()))
    # 2 embedded descriptors alone is enough to trigger the >=2 branch,
    # regardless of how many real E/Z bonds the fragment has. The binding's
    # own token starts with the block (so P8b sees it); `name` deliberately
    # does NOT start with it (a leading digit defeats both the position-0
    # and functional-class-word leading-block parses), which keeps this
    # fixture isolated to P8b -- an unrelated P8a STEREO_UNVERIFIED (this
    # spine threads no stereo_atom_to_locant at all) is not what this test
    # is about.
    token = "(2E,5E)-hepta-2,5-dienyl"
    name = f"5-{token}"
    spine = _spine(_b(token, bs.BindingKind.PREFIX, frag))
    p = bs.verify_spine(mol, spine, name, mode="audit",
                        escalate=bs.STRICT_STEREO_CHARGE_AXES)
    unverified = [f for f in p.findings
                 if f.code == bs.SUBSTITUENT_STEREO_UNVERIFIED]
    assert unverified, "fixture must actually trigger the >=2 branch"
    assert all(f.severity == "info" for f in unverified)
    assert bs.STEREO_UNVERIFIED not in p.codes(), (
        "fixture confound: P8a fired too, not isolating P8b")
