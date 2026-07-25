"""P-29.2 free-valence morphology for CARBON substituent prefixes.

Phase 1b. A substituent prefix names a free valence, and IUPAC P-29.2 fixes
its morphology by the NUMBER of free valences on the attachment atom:

    -yl      one free valence     (single bond to the parent)
    -ylidene two on the same atom (double bond to the parent)
    -ylidyne three on the same atom (triple bond to the parent)

``name_substituent`` is the chokepoint that holds both the fragment and the
attachment atom, so it is the only place where that number is knowable. Before
this phase every tier of its cascade returned a ``-yl`` token regardless of the
attachment bond order, so an exocyclic ``=CH2`` was named ``methyl`` -- a wrong
STRUCTURE, not merely a wrong style.

Two things are pinned here:

  * the CLASS is built -- ``methylidene`` / ``methylidyne`` / ``ethylidene`` /
    ``propan-2-ylidene`` come out of the structure, never out of a lookup keyed
    on a molecule;
  * everything the class does NOT cover FAILS CLOSED. A doubly-bonded free
    valence must never leave this function as a single-valence ``-yl`` token,
    because that names a different molecule.

The single-bond controls are the byte-identity half: for ``free_valence == 1``
the new gate must be a strict no-op.

References: IUPAC 2013 P-29.2, P-29.3.2 (free valence gets the lowest locant
consistent with the chain numbering), P-29.6.2.3 (retained ``propyl``-style
unbranched stems, hence ``propylidene`` rather than ``propan-1-ylidene``).
"""
import pytest
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import (
    _carbon_ylidene_prefix,
    _descriptive_fallback,
    _free_valence_at_attachment,
    name_substituent,
)


def _mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"bad test SMILES {smiles!r}"
    return mol


# ======================================================================
# The structural primitive: free valence read off the attachment bond
# ======================================================================


class TestFreeValenceAtAttachment:
    """The bond ORDER out of the fragment, never the hydrogen count."""

    @pytest.mark.parametrize("smiles,frag,attach,expected", [
        # -CH3 on a ring: one free valence
        ("CC1CCCCC1", {0}, 0, 1),
        # =CH2 on a ring: two free valences on the same atom
        ("C=C1CCCCC1", {0}, 0, 2),
        # CH3-CH= on a ring: two, on the attachment atom of a 2-atom fragment
        ("CC=C1CCCCC1", {0, 1}, 1, 2),
        # (CH3)2C= on a ring: two
        ("CC(C)=C1CCCCC1", {0, 1, 2}, 1, 2),
        # -C#N: the nitrile carbon has ONE free valence toward the ring
        ("N#CC1CCCCC1", {0, 1}, 1, 1),
        # =O on a ring carbon: two
        ("O=C1CCCCC1", {0}, 0, 2),
    ])
    def test_reads_the_attachment_bond_order(self, smiles, frag, attach,
                                             expected):
        assert _free_valence_at_attachment(_mol(smiles), frag, attach) == expected

    def test_hydrogen_count_does_not_decide(self):
        """=CH2 and -CH3 differ in H count AND in bond order; only the bond
        order is allowed to be the discriminator, so a fragment whose H count
        would suggest ``-yl`` but whose bond is double must read 2."""
        # CH2 with 2 H attached by a DOUBLE bond -> 2, not 1.
        assert _free_valence_at_attachment(_mol("C=C1CCCCC1"), {0}, 0) == 2
        # CH with 1 H attached by a SINGLE bond (isopropyl CH) -> 1, not 2.
        assert _free_valence_at_attachment(_mol("CC(C)C1CCCCC1"),
                                           {0, 1, 2}, 1) == 1

    def test_multi_point_attachment_is_undecidable(self):
        """A fragment bonded to the parent at more than one place is a bridge
        (P-25), not a substituent prefix. Return None -- out of scope, and the
        caller must not treat it as a single free valence."""
        # spiro: the CH2CH2CH2CH2 fragment touches the parent ring twice
        mol = _mol("C1CCC2(CC1)CCCC2")
        ri = mol.GetRingInfo()
        spiro = [a.GetIdx() for a in mol.GetAtoms()
                 if ri.NumAtomRings(a.GetIdx()) == 2]
        assert len(spiro) == 1
        second = [r for r in ri.AtomRings() if spiro[0] in r][1]
        frag = set(second) - {spiro[0]}
        attach = next(i for i in frag
                      if mol.GetBondBetweenAtoms(i, spiro[0]) is not None)
        assert _free_valence_at_attachment(mol, frag, attach) is None

    def test_aromatic_linkage_is_undecidable(self):
        """An aromatic linkage bond has no integer order, so no morphology can
        be asserted. Never guess 1."""
        mol = _mol("c1ccccc1")
        frag = {0}
        assert _free_valence_at_attachment(mol, frag, 0) is None


# ======================================================================
# The class: carbon ylidene / ylidyne prefixes
# ======================================================================


class TestCarbonYlidenePrefixes:
    """P-29.2 morphology emitted from the actual attachment bond order."""

    def test_exocyclic_methylene_is_methylidene_not_methyl(self):
        """THE defect. =CH2 on a ring carbon is ``methylidene``; ``methyl``
        would name a different molecule."""
        assert name_substituent(_mol("C=C1CCCCC1"), {0}, 0) == "methylidene"

    def test_ethylidene(self):
        assert name_substituent(_mol("CC=C1CCCCC1"), {0, 1}, 1) == "ethylidene"

    def test_propan_2_ylidene(self):
        """Free valence in the middle of the chain: the lowest locant it can
        take is 2 (P-29.3.2), so the enclosing-mark-free token is
        ``propan-2-ylidene``, not ``propylidene``."""
        assert name_substituent(_mol("CC(C)=C1CCCCC1"),
                                {0, 1, 2}, 1) == "propan-2-ylidene"

    def test_propylidene_when_the_free_valence_is_terminal(self):
        """Terminal free valence keeps the retained unbranched stem form."""
        assert name_substituent(_mol("CCC=C1CCCCC1"),
                                {0, 1, 2}, 2) == "propylidene"

    def test_methylidyne(self):
        """Three free valences on one carbon -- P-29.2 ``-ylidyne``."""
        # HC(triple)C-  :  the terminal CH of a chain triple-bonded to a
        # fragment boundary. Build it explicitly: propyne, fragment = {0}
        # (the CH), attached to C1 by a triple bond.
        mol = _mol("C#CC1CCCCC1")
        assert name_substituent(mol, {0}, 0) == "methylidyne"

    def test_ethylidyne(self):
        mol = _mol("CC#CC1CCCCC1")
        assert name_substituent(mol, {0, 1}, 1) == "ethylidyne"

    def test_constructor_is_not_a_yl_producer(self):
        """``_carbon_ylidene_prefix`` owns the MULTI-valent morphology only.
        A single free valence belongs to the retained-name cascade (which knows
        ``propan-2-yl`` and its retained relatives), so asking this constructor
        for one must be refused, not answered."""
        mol = _mol("CC(C)C1CCCCC1")
        assert _carbon_ylidene_prefix(mol, {0, 1, 2}, 1, 1) is None
        assert _carbon_ylidene_prefix(_mol("CC1CCCCC1"), {0}, 0, 1) is None
        # ... while the multivalent asks are answered from the same fragment
        # shape, so the refusal above is about the VALENCE, not the fragment.
        assert _carbon_ylidene_prefix(
            _mol("CC(C)=C1CCCCC1"), {0, 1, 2}, 1, 2) == "propan-2-ylidene"

    def test_descriptive_fallback_single_carbon_reads_bond_order(self):
        """The originally-reported root-cause site, pinned directly."""
        assert _descriptive_fallback(_mol("C=C1CCCCC1"), {0}, 0) == "methylidene"
        assert _descriptive_fallback(_mol("C#CC1CCCCC1"), {0}, 0) == "methylidyne"
        assert _descriptive_fallback(_mol("CC1CCCCC1"), {0}, 0) == "methyl"


# ======================================================================
# Byte-identity half: single-bond attachments are untouched
# ======================================================================


class TestSingleValenceUnchanged:
    """free_valence == 1 must route through the pre-existing cascade
    unchanged -- this is what keeps PIN default output byte-identical."""

    @pytest.mark.parametrize("smiles,frag,attach,expected", [
        ("CC1CCCCC1", {0}, 0, "methyl"),
        ("CCC1CCCCC1", {0, 1}, 1, "ethyl"),
        ("CC(C)C1CCCCC1", {0, 1, 2}, 1, "propan-2-yl"),
        ("OC1CCCCC1", {0}, 0, "hydroxy"),
        ("ClC1CCCCC1", {0}, 0, "chloro"),
        ("N#CC1CCCCC1", {0, 1}, 1, "cyano"),
    ])
    def test_single_bond_prefixes_unchanged(self, smiles, frag, attach,
                                            expected):
        assert name_substituent(_mol(smiles), frag, attach) == expected


class TestNonCarbonYlideneUnchanged:
    """Chalcogen/nitrogen ylidenes already had correct morphology and must
    keep it -- the gate must not fail them closed."""

    @pytest.mark.parametrize("smiles,frag,attach,expected", [
        ("O=C1CCCCC1", {0}, 0, "oxo"),
        ("S=C1CCCCC1", {0}, 0, "sulfanylidene"),
        ("N=C1CCCCC1", {0}, 0, "imino"),
    ])
    def test_double_bonded_heteroatom_prefixes_unchanged(
            self, smiles, frag, attach, expected):
        assert name_substituent(_mol(smiles), frag, attach) == expected


# ======================================================================
# Fail closed outside the class
# ======================================================================


class TestFailsClosedOutsideTheClass:
    """Never a ``-yl`` token on a multi-order free valence."""

    def _is_single_valence_token(self, tok):
        return isinstance(tok, str) and tok.endswith("yl")

    @pytest.mark.parametrize("smiles,frag,attach", [
        # branched alkylidene not covered by the chain-through-attachment
        # class: 2-methylpropylidene
        ("CC(C)C=C1CCCCC1", {0, 1, 2, 3}, 3),
        # ring-bearing ylidene fragment (cyclohexylidene-style decoration);
        # the ring-closure atom 5 carries the exocyclic double bond to atom 6
        ("C1CCCCC1=C1CCCCC1", {0, 1, 2, 3, 4, 5}, 5),
        # unsaturation inside the ylidene fragment
        ("C=CC=C1CCCCC1", {0, 1, 2}, 2),
    ])
    def test_never_emits_single_valence_token_for_double_attachment(
            self, smiles, frag, attach):
        mol = _mol(smiles)
        assert _free_valence_at_attachment(mol, frag, attach) == 2
        for mancude in (False, True):
            got = name_substituent(mol, frag, attach, allow_mancude=mancude)
            assert not self._is_single_valence_token(got), (
                f"emitted single-valence token {got!r} for a DOUBLE-bonded "
                f"free valence -- that names a different molecule"
            )

    def test_mancude_tier_abstains_with_none(self):
        """Under the best-effort tier the abstention is a clean None, matching
        the Tier-4.5 de-masking convention.

        The example moved. This used to use the branched 2-methylpropylidene
        shape, which the class did not cover when it was written; the decorated
        constructor now names it ((2-methylpropylidene)cyclohexane, OPSIN
        round-trips the exact input), so it is no longer an abstention and
        cannot demonstrate one. The boronic-acid ylidene below still is: the
        recursive namer produces no readable single-valence reading for it, so
        the gate has nothing to give the P-29.2 morpheme to.
        """
        mol = _mol("OB(O)C=C1CCCCC1")
        assert _free_valence_at_attachment(mol, {0, 1, 2, 3}, 3) == 2
        assert name_substituent(mol, {0, 1, 2, 3}, 3,
                                allow_mancude=True) is None
