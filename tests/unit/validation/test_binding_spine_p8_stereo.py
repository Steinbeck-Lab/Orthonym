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
