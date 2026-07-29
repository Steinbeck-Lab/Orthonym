"""P-14.3.4.5 inside a SUBSTITUENT (enclosing-mark) scope -- v29 Phase C Task 5a.

Governing rule chain, verbatim from ``BlueBookV2/BlueBookV2.md`` with headings.

``P-14.3.3`` "Citation of locants" (``:2869``) is the **DENY-DEFAULT**, and its
scoping clause is the entire mechanism of this class --

    "In preferred IUPAC names, if any locants are essential for defining the structure
     of the parent structure or of a unit of structure **as defined by its appropriate
     enclosing marks**, then all locants must be cited for the parent structure or
     that structural unit."

``P-14.3.4.5`` (``:3007``), under ``P-14.3.4`` "Omission of locants", grants the
licence -- and is the ONLY one of the six sub-licences whose text says *"compounds or
**substituent groups**"* --

    "All locants are omitted in compounds or substituent groups in which all
     substitutable positions are completely substituted or modified, for example, by
     hydro, in the same way. Except for hydrogen atoms attached to chalcogen atoms,
     such as in acids, alcohols, and to the carbon atoms of formyl groups (aldehydes),
     all hydrogen atoms are considered substitutable."

and its counter-clause ``:3009`` is the tripwire --

    "In case of partial substitution or modification, all numerical prefixes must be
     indicated. The prefix 'per-' is no longer recommended."

★ ONE MOLECULE, TWO SCOPES, OPPOSITE ANSWERS. The single verbatim ``(PIN)`` witness is
``:3023`` ``1-chloro-2-(pentafluoroethyl)benzene (PIN)``: inside the parentheses the
ethyl group is completely and uniformly substituted and omits, while outside the
benzene ring is only partially substituted and keeps ``1,2``. (NOT ``:3027`` -- that
line is an OCR image placeholder, ``![](_page_73_Picture_8.jpeg)``.)

★ THE ARITHMETIC that makes the licence count HYDROGENS, not positions: ethyl has FIVE
substitutable hydrogens after the free valence is formed -- 2 at C1, 3 at C2 (P-29.2,
``:15813``: *"the atom with the free valence terminates a chain and always has the
locant '1', which is omitted from the name"*). That is why the Blue Book prints
``penta``fluoro. Measured here as ``*CC -> {1: 2, 2: 3}``.

THE CLASS IS OPEN. ``:3009`` retires the ``per-`` contraction, which *was* exactly a
closed-list mechanism; the 2013 recommendations replaced it with counting. ``:3023`` is
the ONLY printed instance of any of ``pentafluoroethyl`` / ``pentachloroethyl`` /
``heptafluoropropyl`` / ``pentafluorophenyl`` (verified by grep over the whole Blue
Book), so ``pentachloroethyl`` and ``pentabromoethyl`` below are DERIVED, not verbatim.
A lookup keyed on the spelling would be wrong on its complement by construction.

MEASURED CODE PATH (call-spy validated on 3 known positives + 2 known negatives --
``ethanol`` and ``benzene`` record ZERO calls at every candidate site). The two live
sites sit on TWO ENTIRELY DIFFERENT cascades, which is why both must be wired:
  * ``substituent_naming._located_acyclic_alkyl_name`` -- the sole productive namer for
    the benzene target, reached from ``substituent_enumerator.py:1513``. For that
    molecule ``_name_saturated_substituted_chain`` is never even CALLED;
  * ``substituent_naming._name_saturated_substituted_chain`` -- the sole productive
    namer for the cyclohexane target, reached from ``name_substituent_fragment``
    Step 2c. For that molecule ``_located_acyclic_alkyl_name`` is called 10x and is
    productive ZERO times.
REFUTED as off-path (0 productive calls across 16 probe molecules):
``_name_branched_alkenyl_substituent``, ``_name_aryl_vinyl_substituent`` (58 calls, 0
productive), ``_name_branched_polyfunctional_substituent`` (never called),
``_name_polyfunctional_acyclic_substituent``, ``rules.ring_substituents
.decorated_ring_substituent_name`` (never called -- a fully substituted ring resolves
with the ring as PARENT, so it never becomes a substituent).

Invariant 11: removing a locant can unmask something worse -- in v29 a fail-closed
prefix turned a fabrication into a silent atom drop, four separate times. Every guard
below asserts the FULL emitted name, never merely that a locant vanished.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.locant_omission import (
    forced_locant_scope,
    isotopic_naming_scope,
)
from orthonym.assembly.substituent_naming import (
    _l5_chain_parent_hydride,
    _l5_substituent_prefix,
    _located_acyclic_alkyl_name,
    _name_saturated_substituted_chain,
)
from orthonym.namer import Orthonym


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def _name(namer, smiles):
    return namer.name(smiles)


def _split_ring_and_branch(smiles):
    """(mol, sub_atoms, attach_idx, parent_set) for the LARGEST acyclic branch hung
    off the molecule's ring system.

    Derived from the structure, never hardcoded atom indices -- and per-branch, so a
    molecule with more than one ring substituent (the ``:3023`` target carries both a
    chloro and the pentafluoroethyl) yields the branch under test rather than an
    assertion. Fragments are grown by BFS that never crosses back into the ring.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    ring_info = mol.GetRingInfo()
    ring = {a.GetIdx() for a in mol.GetAtoms() if ring_info.NumAtomRings(a.GetIdx())}
    assert ring, f"{smiles}: no ring to hang a substituent off"

    branches = []
    for i in (a.GetIdx() for a in mol.GetAtoms()):
        if i in ring:
            continue
        if not any(n.GetIdx() in ring for n in mol.GetAtomWithIdx(i).GetNeighbors()):
            continue
        frag, stack = [], [i]
        seen = set(ring)
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            frag.append(cur)
            for n in mol.GetAtomWithIdx(cur).GetNeighbors():
                if n.GetIdx() not in seen:
                    stack.append(n.GetIdx())
        branches.append((len(frag), i, sorted(frag)))
    assert branches, f"{smiles}: no acyclic branch found"
    branches.sort(reverse=True)
    _n, attach, frag = branches[0]
    return mol, frag, attach, ring


# --------------------------------------------------------------------------- #
# 1. The licence fires -- whole emitted name asserted                          #
# --------------------------------------------------------------------------- #
class TestLicensedOmission:
    def test_the_verbatim_pin_witness(self, namer):
        """★ ``:3023`` ``1-chloro-2-(pentafluoroethyl)benzene (PIN)``.

        Both scopes at once: the substituent omits, the ring keeps ``1,2``.
        """
        assert _name(namer, "Clc1ccccc1C(F)(F)C(F)(F)F") == \
            "1-chloro-2-(pentafluoroethyl)benzene"

    def test_second_live_site_cyclohexane(self, namer):
        """Same licence, the OTHER cascade (``_name_saturated_substituted_chain``)."""
        assert _name(namer, "FC(F)(F)C(F)(F)C1CCCCC1") == \
            "(pentafluoroethyl)cyclohexane"

    @pytest.mark.parametrize("smiles,expected", [
        # DERIVED from :3007 -- the class is OPEN, none of these is printed in the BB.
        ("ClC(Cl)(Cl)C(Cl)(Cl)c1ccccc1", "(pentachloroethyl)benzene"),
        ("BrC(Br)(Br)C(Br)(Br)c1ccccc1", "(pentabromoethyl)benzene"),
        ("FC(F)(F)C(F)(F)C1CCC1", "(pentafluoroethyl)cyclobutane"),
    ])
    def test_open_class_entailed_by_the_predicate(self, namer, smiles, expected):
        assert _name(namer, smiles) == expected

    def test_enclosing_marks_survive_the_locant_removal(self, namer):
        """★ INVARIANT 11 REGRESSION, caught by measurement and fixed at root.

        Removing the locants first produced ``heptafluoropropylbenzene`` -- the
        enclosing marks were LOST. ``is_complex_substituent`` returns True on the
        first digit it sees, so while these names carried locants the digit check was
        the gate and ``_HALOALKYL_RE`` never mattered; its multiplier alternation
        stopped at ``hexa``, so ``hepta``+ fell through to "simple". ``:3023`` prints
        ``1-chloro-2-(pentafluoroethyl)benzene (PIN)`` WITH the marks, and bare
        ``heptafluoropropylbenzene`` also reads as ``heptafluoro`` +
        ``propylbenzene``. Fixed by completing the multiplier set from
        ``SIMPLE_MULTIPLIERS``.
        """
        assert _name(namer, "FC(F)(F)C(F)(F)C(F)(F)c1ccccc1") == \
            "(heptafluoropropyl)benzene"

    def test_uniformly_complete_methyl_stays_locant_free(self, namer):
        """``trichloromethyl`` was already locant-free before this task; the licence
        must not disturb it (it is the 1-carbon member of the same class)."""
        assert _name(namer, "ClC(Cl)(Cl)c1ccccc1") == "(trichloromethyl)benzene"
        assert _name(namer, "FC(F)(F)c1ccccc1") == "(trifluoromethyl)benzene"


# --------------------------------------------------------------------------- #
# 2. The deny-default holds -- every reachable Blue Book negative              #
# --------------------------------------------------------------------------- #
class TestDenyByDefault:
    def test_flagship_parent_scope_tripwire(self, namer):
        """``:3019`` ``2,2,3,3,3-pentafluoropropan-1-ol (PIN)``.

        ★ The same five fluorines as the target, a DIFFERENT scope: here the parent
        is fluorinated and C1 (``-CH2-OH``) keeps two hydrogens => partial => ``:3009``.
        """
        assert _name(namer, "OCC(F)(F)C(F)(F)F") == "2,2,3,3,3-pentafluoropropan-1-ol"

    def test_the_near_miss_attachment_carbon_keeps_its_hydrogen(self, namer):
        """★ ``:46359`` ``(1,1,1,3,3,3-hexafluoropropan-2-yl)oxy...``.

        A *substituent group*, fully fluorinated at C1 and C3, but the ATTACHMENT
        carbon C2 keeps its H. Guards against a predicate that only checks terminal
        carbons.
        """
        assert _name(namer, "FC(F)(F)C(C1CCCCC1)C(F)(F)F") == \
            "(1,1,1,3,3,3-hexafluoropropan-2-yl)cyclohexane"

    def test_the_minimal_pair_same_ethyl_skeleton(self, namer):
        """★ ``:41664`` ``(2,2-dichloroethyl)sulfanylium (PIN)`` -- the same ethyl
        skeleton as the target, differing only in that 2 of 5 H are replaced."""
        assert _name(namer, "ClC(Cl)CC1CCCCC1") == "(2,2-dichloroethyl)cyclohexane"

    @pytest.mark.parametrize("smiles,expected", [
        # :3529 '6-(1-chloroethyl)-5-(2-chloroethyl)-1H-indole (PIN)' -- 1 of 5 H.
        ("CC(Cl)c1ccccc1", "(1-chloroethyl)benzene"),
        ("ClCCc1ccccc1", "(2-chloroethyl)benzene"),
        # 4 of 5 H -- the last hydrogen is what keeps every locant (:3009).
        ("FC(F)C(F)(F)c1ccccc1", "(1,1,2,2-tetrafluoroethyl)benzene"),
        ("CC(F)(F)C(F)(F)c1ccccc1", "(1,1,2,2-tetrafluoropropyl)benzene"),
        # partial, and NOT uniform.
        ("ClCC(Cl)Cc1ccccc1", "(2,3-dichloropropyl)benzene"),
    ])
    def test_partial_substitution_keeps_every_locant(self, namer, smiles, expected):
        assert _name(namer, smiles) == expected

    def test_internal_free_valence_keeps_locants_end_to_end(self, namer):
        """★ The k != 1 boundary, asserted on the EMITTED name.

        All seven substitutable hydrogens ARE uniformly replaced, so P-14.3.4.5's
        completeness test is satisfied -- yet the scope must still cite them, because
        the free-valence locant ``2`` is itself essential (it distinguishes
        propan-2-yl from propan-1-yl) and P-14.3.3 (``:2869``) then requires *"all
        locants ... for that structural unit"*. The Blue Book prints no
        fully-substituted substituent group with an internal free valence, so
        deny-by-default picks retention. If this ever elides to
        ``(heptafluoropropan-2-yl)benzene``, that boundary was crossed deliberately
        and needs its own citation.
        """
        assert _name(namer, "FC(F)(F)C(F)(C(F)(F)F)c1ccccc1") == \
            "(1,1,1,2,3,3,3-heptafluoropropan-2-yl)benzene"

    def test_zero_h_everywhere_but_not_by_the_same_group(self, namer):
        """★ ``:29619`` -- the Blue Book cites P-14.3.4.5 BY NAME to explain a
        NEGATIVE: every carbon has 0 H, yet the locants are required, because C1 is
        substituted by something other than fluoro. Guards against a
        'zero-H-everywhere' predicate instead of 'zero-H-everywhere BY THE SAME
        GROUP'.

        Asserted structurally: the emitted name form differs from the Blue Book's
        (Orthonym names this as an N-acylpiperidine), so pinning the exact string
        would enshrine an unrelated spelling. What must hold is that the fluoro
        locants are all cited.
        """
        got = _name(namer, "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)"
                           "C(F)(F)C(=O)N1CCCCC1")
        assert "pentadecafluoro" in got, got
        assert "2,2,3,3,4,4,5,5,6,6,7,7,8,8,8-pentadecafluoro" in got, \
            f"the fluoro locants are required, see P-14.3.4.5 (:29619): {got!r}"

    def test_amide_nh_are_substitutable_so_the_amide_keeps(self, namer):
        """★ The sharpest boundary pair in Phase C: identical fluorination, the ACID
        omits (Task 5b, parent scope) and the AMIDE keeps -- because an acid ``-OH``
        hydrogen sits on a chalcogen (excluded, ``:3007``) while an amide ``N-H``
        does not (included; proven independently by ``:2889``
        ``N1,N3-dimethylpropanediamide (PIN)``). Substituent scope should not reach
        it -- asserted anyway."""
        assert _name(namer, "FC(F)(F)C(F)(F)C(=O)N") == \
            "2,2,3,3,3-pentafluoropropanamide"

    @pytest.mark.parametrize("smiles,expected", [
        # Parent-scope L5 -- Task 5b, NOT this task. Must be untouched here.
        ("OC(=O)C(F)(F)C(F)(F)C(F)(F)F", "2,2,3,3,4,4,4-heptafluorobutanoic acid"),
        ("OC(=O)C(F)(F)C(F)(F)F", "2,2,3,3,3-pentafluoropropanoic acid"),
        # Correct today for other reasons -- census tripwires.
        ("OC(=O)C(F)(F)F", "trifluoroacetic acid"),
        ("OC(=O)CC(F)(F)F", "3,3,3-trifluoropropanoic acid"),
        ("OC(=O)C(F)C(F)(F)F", "2,3,3,3-tetrafluoropropanoic acid"),
        ("FC(F)(F)CO", "2,2,2-trifluoroethan-1-ol"),
        ("Cc1c(C)c(C)c(C)c(C)c1C", "hexamethylbenzene"),
        ("Fc1c(F)c(F)c(F)c(F)c1F", "hexafluorobenzene"),
        ("Clc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl", "hexachlorobenzene"),
        ("Oc1c(O)c(O)c(O)c(O)c1O", "benzenehexol"),
        ("CNC(=O)N", "N-methylurea"),
        ("CNC(=O)CC(=O)NC", "N1,N3-dimethylpropanediamide"),
        # Plain alkyl / branched substituents the licence must never touch.
        ("CC(CC)c1ccccc1", "(butan-2-yl)benzene"),
        ("CC(C)(C)c1ccccc1", "tert-butylbenzene"),
        ("ClCCCCc1ccccc1", "(4-chlorobutyl)benzene"),
        ("ClCc1ccccc1", "(chloromethyl)benzene"),
    ])
    def test_census_tripwires_unchanged(self, namer, smiles, expected):
        assert _name(namer, smiles) == expected


# --------------------------------------------------------------------------- #
# 3. The helper in isolation -- the C8 dummy-atom marshalling                  #
# --------------------------------------------------------------------------- #
class TestParentHydrideMarshalling:
    @pytest.mark.parametrize("chain_len,k,expected_h", [
        (1, 1, {1: 3}),
        (2, 1, {1: 2, 2: 3}),            # ★ ethyl = 5 => 'penta'fluoro
        (3, 1, {1: 2, 2: 2, 3: 3}),
        (3, 2, {1: 3, 2: 1, 3: 3}),
        (5, 3, {1: 3, 2: 2, 3: 1, 4: 2, 5: 3}),
    ])
    def test_free_valence_consumes_exactly_one_hydrogen(
        self, chain_len, k, expected_h
    ):
        from orthonym.assembly.locant_omission import substitutable_h_count
        hydride = _l5_chain_parent_hydride(chain_len, k)
        assert hydride is not None
        got = {i + 1: substitutable_h_count(hydride, i) for i in range(chain_len)}
        assert got == expected_h
        dummy = hydride.GetAtomWithIdx(chain_len)
        assert dummy.GetSymbol() == "*"
        assert dummy.GetTotalNumHs() == 0, \
            "the free valence must not contribute a substitutable hydrogen"

    def test_ethyl_hydride_is_the_brief_c8_form(self):
        assert Chem.MolToSmiles(_l5_chain_parent_hydride(2, 1)) == "*CC"


class TestHelperPredicate:
    def test_uniform_complete_ethyl_licenses(self):
        mol, sub, attach, _ = _split_ring_and_branch("FC(F)(F)C(F)(F)C1CCCCC1")
        chain = [attach] + [
            n.GetIdx() for n in mol.GetAtomWithIdx(attach).GetNeighbors()
            if n.GetSymbol() == "C" and n.GetIdx() in set(sub)
        ]
        groups = {"fluoro": [1, 1, 2, 2, 2]}
        assert _l5_substituent_prefix(mol, sub, chain, 1, groups) == "pentafluoro"

    def test_hexafluoropropan_2_yl_shape_returns_none(self):
        """★ The ``:46359`` shape at the helper level: internal free valence, and
        the attachment carbon keeps a hydrogen. TWO independent reasons to deny."""
        mol, sub, attach, _ = _split_ring_and_branch(
            "FC(F)(F)C(C1CCCCC1)C(F)(F)F")
        cs = [n.GetIdx() for n in mol.GetAtomWithIdx(attach).GetNeighbors()
              if n.GetSymbol() == "C" and n.GetIdx() in set(sub)]
        chain = [cs[0], attach, cs[1]]
        groups = {"fluoro": [1, 1, 1, 3, 3, 3]}
        assert _l5_substituent_prefix(mol, sub, chain, 2, groups) is None

    def test_internal_free_valence_is_denied_even_when_complete(self):
        """k >= 2 cites the free-valence locant, which is essential; P-14.3.3
        (``:2869``) then requires every locant in that structural unit."""
        mol, sub, attach, _ = _split_ring_and_branch(
            "FC(F)(F)C(F)(C(F)(F)F)c1ccccc1")
        cs = [n.GetIdx() for n in mol.GetAtomWithIdx(attach).GetNeighbors()
              if n.GetSymbol() == "C" and n.GetIdx() in set(sub)]
        chain = [cs[0], attach, cs[1]]
        groups = {"fluoro": [1, 1, 1, 2, 3, 3, 3]}
        assert _l5_substituent_prefix(mol, sub, chain, 2, groups) is None

    def test_partial_substitution_returns_none(self):
        mol, sub, attach, _ = _split_ring_and_branch("ClC(Cl)CC1CCCCC1")
        chain = [attach] + [
            n.GetIdx() for n in mol.GetAtomWithIdx(attach).GetNeighbors()
            if n.GetSymbol() == "C" and n.GetIdx() in set(sub)
        ]
        assert _l5_substituent_prefix(mol, sub, chain, 1,
                                      {"chloro": [2, 2]}) is None

    @pytest.mark.parametrize("groups", [
        # ★ EVERY position mixed, and every position's kind-set has IDENTICAL
        # contents. Designed from the FAILURE MODE: without the per-position
        # uniformity clause the code picks an arbitrary kind per position, and equal
        # sets of the same strings iterate identically within a process -- so both
        # positions pick the SAME kind, the global len(kinds) check passes, the counts
        # match, and the licence fires while silently DROPPING the other halogen
        # (invariant 11: a locant decision became a structure drop). This witness
        # therefore catches the mutation on EVERY hash seed.
        #
        # The naive witness ({"fluoro": [1,2,2,2], "chloro": [1]}) does NOT: only one
        # position is mixed, so the outcome is 50/50 on set iteration order --
        # measured 2 of 3 runs, then 3 of 4 runs, before this was redesigned.
        # locant 1 (2 H): 1 fluoro + 1 chloro = 2; locant 2 (3 H): 2 fluoro + 1 chloro
        {"fluoro": [1, 2, 2], "chloro": [1, 2]},
        {"chloro": [1, 2], "fluoro": [1, 2, 2]},
    ])
    def test_two_kinds_at_one_position_returns_none(self, groups):
        """Not 'in the same way' (``:3007``) => ``:3009`` restores every locant."""
        mol, sub, attach, _ = _split_ring_and_branch("FC(F)(F)C(F)(F)C1CCCCC1")
        chain = [attach] + [
            n.GetIdx() for n in mol.GetAtomWithIdx(attach).GetNeighbors()
            if n.GetSymbol() == "C" and n.GetIdx() in set(sub)
        ]
        assert _l5_substituent_prefix(mol, sub, chain, 1, dict(groups)) is None

    def test_two_kinds_on_different_positions_returns_none(self):
        """Global heterogeneity -- caught by ``l5_uniform_complete``'s own
        ``len(kinds) != 1`` check rather than the per-position one."""
        mol, sub, attach, _ = _split_ring_and_branch("FC(F)(F)C(F)(F)C1CCCCC1")
        chain = [attach] + [
            n.GetIdx() for n in mol.GetAtomWithIdx(attach).GetNeighbors()
            if n.GetSymbol() == "C" and n.GetIdx() in set(sub)
        ]
        assert _l5_substituent_prefix(
            mol, sub, chain, 1,
            {"chloro": [1, 1], "fluoro": [2, 2, 2]}) is None

    def test_out_of_range_locant_returns_none(self):
        mol, sub, attach, _ = _split_ring_and_branch("FC(F)(F)C(F)(F)C1CCCCC1")
        chain = [attach] + [
            n.GetIdx() for n in mol.GetAtomWithIdx(attach).GetNeighbors()
            if n.GetSymbol() == "C" and n.GetIdx() in set(sub)
        ]
        assert _l5_substituent_prefix(mol, sub, chain, 1,
                                      {"fluoro": [1, 1, 2, 2, 3]}) is None

    def test_empty_groups_returns_none(self):
        mol, sub, attach, _ = _split_ring_and_branch("FC(F)(F)C(F)(F)C1CCCCC1")
        chain = [attach] + [
            n.GetIdx() for n in mol.GetAtomWithIdx(attach).GetNeighbors()
            if n.GetSymbol() == "C" and n.GetIdx() in set(sub)
        ]
        assert _l5_substituent_prefix(mol, sub, chain, 1, {}) is None


class TestForcedLocantScope:
    def test_helper_declines_inside_a_forced_scope(self):
        """P-14.3.3 as an ambient scope. The isotope path names an isotope-STRIPPED
        molecule, so a structural ``has_isotope`` test at this depth is blind by
        construction (measured: ``GetIsotope()`` reads 0 for every scope atom even
        for a 13C input). This ContextVar is the ONLY live guard, and neither
        SELF-01 (which ``namer.py`` states verbatim *"ignores isotopes"*) nor the
        gold set can see a failure here -- gold exposure for this class is zero.
        """
        mol, sub, attach, _ = _split_ring_and_branch("FC(F)(F)C(F)(F)C1CCCCC1")
        chain = [attach] + [
            n.GetIdx() for n in mol.GetAtomWithIdx(attach).GetNeighbors()
            if n.GetSymbol() == "C" and n.GetIdx() in set(sub)
        ]
        groups = {"fluoro": [1, 1, 2, 2, 2]}
        # known positive first, so a silently-broken harness cannot pass this test
        assert _l5_substituent_prefix(mol, sub, chain, 1, groups) == "pentafluoro"
        with forced_locant_scope("test"):
            assert _l5_substituent_prefix(mol, sub, chain, 1, groups) is None
        # and the scope is properly unwound
        assert _l5_substituent_prefix(mol, sub, chain, 1, groups) == "pentafluoro"

    def test_helper_declines_inside_an_isotopic_scope(self):
        """★ ``forced_locant_scope`` ALONE IS NOT ENOUGH -- measured 2026-07-29.

        ``rules/isotopes.py`` enters the FORCED scope only once ``_enumerate`` has
        established that the descriptor needs a locant. For
        ``FC(F)(F)[13C](F)(F)C1CCCCC1`` it never does, so at this depth
        ``locants_are_forced()`` is False AND every ``GetIsotope()`` reads 0 -- both
        signals negative, while the finished name still carries ``(13C1)``. Keying
        the licence on the forced flag alone emptied that scope of all its locants,
        which P-82.6.1.1 (``:44180``) forbids because the ethyl group's two carbons
        are inequivalent.
        """
        mol, sub, attach, _ = _split_ring_and_branch("FC(F)(F)C(F)(F)C1CCCCC1")
        chain = [attach] + [
            n.GetIdx() for n in mol.GetAtomWithIdx(attach).GetNeighbors()
            if n.GetSymbol() == "C" and n.GetIdx() in set(sub)
        ]
        groups = {"fluoro": [1, 1, 2, 2, 2]}
        assert _l5_substituent_prefix(mol, sub, chain, 1, groups) == "pentafluoro"
        with isotopic_naming_scope("isotope"):
            assert _l5_substituent_prefix(mol, sub, chain, 1, groups) is None
        assert _l5_substituent_prefix(mol, sub, chain, 1, groups) == "pentafluoro"


class TestIsotopeEndToEnd:
    """The licence must not strip the last locants out of a labelled scope, and must
    not disturb the labelled omissions that are already correct."""

    @pytest.mark.parametrize("smiles,expected", [
        # ★ The regression this guard exists for: locants_are_forced() is False here.
        ("FC(F)(F)[13C](F)(F)C1CCCCC1",
         "(1,1,2,2,2-pentafluoro(13C1)ethyl)cyclohexane"),
        ("FC(F)(F)[13C](F)(F)c1ccccc1",
         "(1,1,2,2,2-pentafluoro(13C1)ethyl)benzene"),
        # forced scope IS active here (the descriptor needed locant 2).
        ("Clc1ccccc1C(F)(F)[13C](F)(F)F",
         "1-chloro-2-(1,1,2,2,2-pentafluoro(2-13C1)ethyl)benzene"),
        # partial substitution: denied on its own merits, label or no label.
        ("FC(F)(F)C(F)([2H])C1CCCCC1", "(1,2,2,2-tetrafluoro(2H1)ethyl)cyclohexane"),
        # P-82.6.1.3 omissions that must SURVIVE -- the weaker isotopic flag is
        # deliberately not consulted by rules/benzene.py.
        ("Cc1c(C)c(C)c(C)c(C)[13c]1C", "hexamethyl(13C1)benzene"),
        ("[13CH3]CO", "(2-13C1)ethan-1-ol"),
    ])
    def test_labelled_scopes(self, namer, smiles, expected):
        assert _name(namer, smiles) == expected


# --------------------------------------------------------------------------- #
# 4. One test per MEASURED-LIVE site, so a refactor that drops one fails       #
# --------------------------------------------------------------------------- #
class TestLiveSitesStayWired:
    def test_site_located_acyclic_alkyl_name(self):
        """The benzene target's sole productive namer (spy-measured, reached from
        ``substituent_enumerator.py:1513``). ``_name_saturated_substituted_chain``
        is never called for this molecule."""
        mol, sub, attach, _ = _split_ring_and_branch("Clc1ccccc1C(F)(F)C(F)(F)F")
        assert _located_acyclic_alkyl_name(mol, sub, attach) == \
            ("pentafluoroethyl", 1)

    def test_site_name_saturated_substituted_chain(self):
        """The cyclohexane target's sole productive namer (spy-measured, reached
        from ``name_substituent_fragment`` Step 2c). For this molecule
        ``_located_acyclic_alkyl_name`` is called 10x and productive 0 times."""
        mol, sub, attach, parent = _split_ring_and_branch(
            "FC(F)(F)C(F)(F)C1CCCCC1")
        assert _name_saturated_substituted_chain(mol, sub, attach, parent) == \
            "pentafluoroethyl"

    def test_both_sites_still_retain_on_a_partial_fragment(self):
        """The same two entry points, on partial fragments: the join they own must
        still be reachable and still cite locants."""
        mol, sub, attach, _ = _split_ring_and_branch("CC(Cl)c1ccccc1")
        assert _located_acyclic_alkyl_name(mol, sub, attach) == ("1-chloroethyl", 1)
        mol, sub, attach, parent = _split_ring_and_branch("ClC(Cl)CC1CCCCC1")
        assert _name_saturated_substituted_chain(mol, sub, attach, parent) == \
            "2,2-dichloroethyl"
