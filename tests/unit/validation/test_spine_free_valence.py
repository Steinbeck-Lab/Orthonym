"""P7: a prefix token's free-valence morphology vs the real linkage bond order.

a phase, spine half.

P2 proves every bond is CLAIMED exactly once. It never looks at a bond's
ORDER, so the spine happily certified ``methyl`` bound to a carbon that is
DOUBLE-bonded to the ring -- the exact wrong-structure defect a phase exists
to stop. P1-P6 all passed it, which means the proof could not have caught the
bug that motivated the fix, and any later enforce mode would have shipped it.

P7 closes that. For a PREFIX binding it reads the actual linkage bond out of
the graph and compares its order with the morphology the token's own text
asserts: ``-yl`` one, ``-ylidene`` two, ``-ylidyne`` three).

The discipline is ``token_arity``'s: a CONFIDENT verdict must be correct, so
anything the morphology oracle cannot decide -- a token that spells no
free-valence morpheme at all (``oxo``, ``hydroxy``), a multi-point attachment,
an aromatic linkage -- is reported ``"info"`` and never blocks. Only a
confidently-known morphology that DISAGREES with the graph is an ``"error"``.
"""
import pytest
from rdkit import Chem

from orthonym.validation import binding_spine as bs

pytestmark = pytest.mark.unit

# methylidenecyclohexane: atom 0 is the exocyclic =CH2, atoms 1-6 the ring.
METHYLIDENECYCLOHEXANE = Chem.MolFromSmiles("C=C1CCCCC1")
# methylcyclohexane: atom 0 is the -CH3, atoms 1-6 the ring.
METHYLCYCLOHEXANE = Chem.MolFromSmiles("CC1CCCCC1")
# cyclohexanone: atom 0 is the =O.
CYCLOHEXANONE = Chem.MolFromSmiles("O=C1CCCCC1")


def _b(token, kind, atoms, **kw):
    return bs.SpineBinding(token=token, kind=kind,
                           atom_ids=frozenset(atoms), **kw)


def _spine(*roots):
    return bs.BindingSpine(roots=tuple(roots))


class TestTheDefectIsCaught:

    def test_methyl_on_a_double_bonded_carbon_is_an_error(self):
        """THE regression this proof exists for. Every atom is claimed, every
        bond is claimed, the name spans anchor -- and the name is still wrong,
        because ``methyl`` asserts one free valence and the graph has two."""
        proof = bs.verify_spine(METHYLIDENECYCLOHEXANE, _spine(
            _b("cyclohexane", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6]),
            _b("methyl", bs.BindingKind.PREFIX, [0]),
        ), "methylcyclohexane")
        assert bs.FREE_VALENCE_MISMATCH in proof.codes()
        assert not proof.ok

    def test_p1_to_p6_alone_would_have_passed_it(self):
        """Mutation guard: without P7 the very same spine is clean. If this
        ever fails, P7 is no longer the thing doing the work here."""
        findings = []
        stats = {}
        spine = _spine(
            _b("cyclohexane", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6]),
            _b("methyl", bs.BindingKind.PREFIX, [0]),
        )
        bs._p1_atom_partition(METHYLIDENECYCLOHEXANE, spine, False,
                              findings, stats)
        bs._p2_bond_totality(METHYLIDENECYCLOHEXANE, spine, "audit",
                             findings, stats)
        assert not [f for f in findings if f.severity == "error"], (
            "P1/P2 already reject this spine, so the P7 test above would pass "
            "for the wrong reason"
        )

    def test_methylidene_on_the_same_graph_passes(self):
        proof = bs.verify_spine(METHYLIDENECYCLOHEXANE, _spine(
            _b("cyclohexane", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6]),
            _b("methylidene", bs.BindingKind.PREFIX, [0]),
        ), "methylidenecyclohexane")
        assert bs.FREE_VALENCE_MISMATCH not in proof.codes()
        assert proof.ok, proof.findings

    def test_methylidene_on_a_single_bond_is_an_error(self):
        """The mirror image: the morphology over-states the bond order."""
        proof = bs.verify_spine(METHYLCYCLOHEXANE, _spine(
            _b("cyclohexane", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6]),
            _b("methylidene", bs.BindingKind.PREFIX, [0]),
        ), "methylidenecyclohexane")
        assert bs.FREE_VALENCE_MISMATCH in proof.codes()
        assert not proof.ok


class TestConfidentVerdictsAreCorrect:

    def test_methyl_on_a_single_bond_passes(self):
        proof = bs.verify_spine(METHYLCYCLOHEXANE, _spine(
            _b("cyclohexane", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6]),
            _b("methyl", bs.BindingKind.PREFIX, [0]),
        ), "methylcyclohexane")
        assert proof.ok, proof.findings
        assert bs.FREE_VALENCE_UNVERIFIED not in proof.codes()

    def test_ylidyne_morphology_is_read(self):
        """CH3-C(triple)-: atoms 0,1 are the ``ethylidyne`` fragment and the
        bond 1-2 out of it is a TRIPLE bond, which is the three free valences
        the token asserts."""
        mol = Chem.MolFromSmiles("CC#CC1CCCCC1")
        assert (mol.GetBondBetweenAtoms(1, 2).GetBondType()
                == Chem.BondType.TRIPLE)
        proof = bs.verify_spine(mol, _spine(
            _b("parent", bs.BindingKind.PARENT, [2, 3, 4, 5, 6, 7, 8]),
            _b("ethylidyne", bs.BindingKind.PREFIX, [0, 1]),
        ), "ethylidyneparent")
        assert bs.FREE_VALENCE_MISMATCH not in proof.codes()
        assert proof.stats["free_valence_confident"] == 1

    def test_ylidyne_token_on_a_single_bond_is_an_error(self):
        proof = bs.verify_spine(METHYLCYCLOHEXANE, _spine(
            _b("cyclohexane", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6]),
            _b("methylidyne", bs.BindingKind.PREFIX, [0]),
        ), "methylidynecyclohexane")
        assert bs.FREE_VALENCE_MISMATCH in proof.codes()


class TestUnverifiedIsNeverAnError:
    """Absence of evidence, exactly as ``ARITY_UNVERIFIED``."""

    def test_token_spelling_no_free_valence_morpheme_is_info(self):
        """``oxo`` is a correct prefix for a doubly-bonded O, and it spells no
        ``-yl`` morpheme, so P7 has nothing to compare and must not guess."""
        proof = bs.verify_spine(CYCLOHEXANONE, _spine(
            _b("cyclohexane", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6]),
            _b("oxo", bs.BindingKind.PREFIX, [0]),
        ), "oxocyclohexane")
        assert bs.FREE_VALENCE_MISMATCH not in proof.codes()
        assert bs.FREE_VALENCE_UNVERIFIED in proof.codes()
        assert all(f.severity == "info" for f in proof.findings
                   if f.code == bs.FREE_VALENCE_UNVERIFIED)
        assert all(f.severity == "info"
                   for f in proof.findings
                   if f.code == bs.FREE_VALENCE_UNVERIFIED)

    def test_multi_point_attachment_is_info(self):
        """Two linkage bonds is a bridge/spiro shape, where the simple
        -yl/-ylidene/-ylidyne rule does not apply. Never a confident verdict."""
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CCCC2")  # spiro[4.5]decane
        ri = mol.GetRingInfo()
        spiro = next(a.GetIdx() for a in mol.GetAtoms()
                     if ri.NumAtomRings(a.GetIdx()) == 2)
        second = [r for r in ri.AtomRings() if spiro in r][1]
        frag = sorted(set(second) - {spiro})
        rest = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in frag]
        proof = bs.verify_spine(mol, _spine(
            _b("cyclohexane", bs.BindingKind.PARENT, rest),
            _b("butyl", bs.BindingKind.PREFIX, frag),
        ), "butylcyclohexane")
        assert bs.FREE_VALENCE_MISMATCH not in proof.codes()
        assert bs.FREE_VALENCE_UNVERIFIED in proof.codes()
        assert all(f.severity == "info" for f in proof.findings
                   if f.code == bs.FREE_VALENCE_UNVERIFIED)

    def test_multiplied_yl_token_is_info(self):
        """``-diyl`` spreads its free valences over two atoms, so the single
        attachment-bond rule cannot judge it."""
        mol = Chem.MolFromSmiles("C1CCC2(CC1)CCCC2")
        ri = mol.GetRingInfo()
        spiro = next(a.GetIdx() for a in mol.GetAtoms()
                     if ri.NumAtomRings(a.GetIdx()) == 2)
        second = [r for r in ri.AtomRings() if spiro in r][1]
        frag = sorted(set(second) - {spiro})
        rest = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in frag]
        proof = bs.verify_spine(mol, _spine(
            _b("cyclohexane", bs.BindingKind.PARENT, rest),
            _b("butane-1,4-diyl", bs.BindingKind.PREFIX, frag),
        ), "butane-1,4-diylcyclohexane")
        assert bs.FREE_VALENCE_MISMATCH not in proof.codes()

    def test_parent_and_suffix_kinds_are_out_of_scope(self):
        """P7 judges PREFIX bindings only: a parent has no free valence and a
        suffix's morphology is a different lexicon."""
        proof = bs.verify_spine(Chem.MolFromSmiles("CCO"), _spine(
            _b("eth", bs.BindingKind.PARENT, [0, 1]),
            _b("ol", bs.BindingKind.SUFFIX, [2]),
        ), "ethan-1-ol")
        assert bs.FREE_VALENCE_MISMATCH not in proof.codes()
        assert bs.FREE_VALENCE_UNVERIFIED not in proof.codes()


class TestProofBookkeeping:

    def test_p7_is_listed_in_the_proofs_that_ran(self):
        proof = bs.verify_spine(METHYLCYCLOHEXANE, _spine(
            _b("cyclohexane", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6]),
            _b("methyl", bs.BindingKind.PREFIX, [0]),
        ), "methylcyclohexane")
        assert "P7" in proof.stats["proofs"]

    def test_counters_are_reported(self):
        proof = bs.verify_spine(CYCLOHEXANONE, _spine(
            _b("cyclohexane", bs.BindingKind.PARENT, [1, 2, 3, 4, 5, 6]),
            _b("oxo", bs.BindingKind.PREFIX, [0]),
        ), "oxocyclohexane")
        assert proof.stats["free_valence_confident"] == 0
        assert proof.stats["free_valence_unverified"] == 1
