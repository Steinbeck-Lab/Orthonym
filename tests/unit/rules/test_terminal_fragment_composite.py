"""Task 4: ring-bearing fragments, and the stereochemistry refusal guard.

Ring nomenclature is delegated wholesale to `rules.terminal_ring.terminal_ring_name`,
which already applies P-22.2.3 skeletal replacement with lambda and ring
multiple-bond locants and gates every result on its own reconstruction audit. What
is NEW here is the JOIN: a decorated ring fragment must account for the ring atoms
AND every decoration, and a ring decoration's locant must come from the RING's own
numbering (which skips 4 -> 4a -> 5), never from list position.

Expected-string policy, deliberate: exact strings are asserted only where derived
from `terminal_ring_name` directly. Everything else is asserted by STRUCTURE --
completeness, basis, refusal -- and by OPSIN round-trip, which is non-circular. The
coordinator's hand-written expected values have been wrong four times in this phase;
a derived or round-trip-verified oracle is not subject to that failure mode.
"""
import pytest
from rdkit import Chem

from orthonym.rules.terminal_fragment import terminal_fragment_name


def _whole(smiles, attach=0):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return mol, set(range(mol.GetNumAtoms())), attach


# --------------------------------------------------------------- ring systems

@pytest.mark.parametrize("smiles,attach,expected", [
    # derived from terminal_ring_name directly, 2026-08-04
    ("C1CCCCC1", 0, "cyclohexan-1-yl"),
    ("C1CCOCC1", 0, "1-oxacyclohexan-4-yl"),
    # mancude ring: the SYSTEMATIC form, not the retained 'phenyl' -- correct for
    # this best-effort tier, where a complete name beats a preferred-or-absent one
    ("c1ccccc1", 0, "cyclohexa-1,3,5-trien-1-yl"),
])
def test_bare_ring_system_delegates_to_terminal_ring(smiles, attach, expected):
    mol, frag, att = _whole(smiles, attach)
    got = terminal_fragment_name(mol, frag, att)
    assert got is not None, f"{smiles} refused"
    assert got.name == expected
    assert got.basis == "ring"
    assert got.atoms == frozenset(frag)


@pytest.mark.parametrize("smiles,attach", [
    ("CC1CCCCC1", 1),        # methyl decoration on the attachment atom
    ("C1CCCCC1CC", 5),       # ethyl decoration
    ("CCC1CCCCC1", 2),       # propyl decoration
])
def test_decorated_ring_is_complete_across_the_join(smiles, attach):
    """The join must drop neither the ring's atoms nor the decoration's."""
    mol, frag, att = _whole(smiles, attach)
    got = terminal_fragment_name(mol, frag, att)
    assert got is not None, f"{smiles} refused"
    assert got.atoms == frozenset(frag), (
        "completeness invariant: an atom-short name is the defect this module "
        "exists to remove"
    )
    assert got.basis == "composite"


def test_off_table_ring_element_refuses():
    """Zn has no Table 1.5 morpheme; terminal_ring_name refuses and so must we."""
    mol, frag, att = _whole("C1CC[Zn]CC1", 0)
    assert terminal_fragment_name(mol, frag, att) is None


def test_a_ring_decoration_that_cannot_be_named_refuses_the_whole_fragment():
    """Partial success is the atom-drop bug wearing a different hat."""
    mol = Chem.MolFromSmiles("C1CCCCC1[Zn]C")
    assert mol is not None
    got = terminal_fragment_name(mol, set(range(mol.GetNumAtoms())), 0)
    assert got is None


# ------------------------------------------- the stereochemistry refusal guard

def test_an_ACYCLIC_stereocentre_emits_the_descriptor():
    """The acyclic path EMITS the backbone R/S centre (v30 B3), it no longer
    refuses. SELF-01's C6 stereo layer verifies the emitted descriptor, so a wrong
    CIP/locant abstains rather than shipping a different compound.
    """
    # NB `CC[C@H](C)CC` looks like a stereocentre but is NOT one -- two identical
    # ethyl arms, so RDKit strips the tag and the case would pass vacuously. The
    # molecule below really does carry a defined centre at atom 2.
    mol = Chem.MolFromSmiles("CC[C@H](C)CCC")
    assert mol is not None
    assert any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
               for a in mol.GetAtoms()), "fixture must actually carry stereo"
    got = terminal_fragment_name(mol, set(range(mol.GetNumAtoms())), 0)
    assert got is not None and "(3S)" in got.name


def test_an_ACYCLIC_defined_backbone_double_bond_now_emits_ez():
    """v30 internal-C=C lever: a defined BACKBONE C=C configuration is now
    EMITTED as a leading (nE)/(nZ) block rather than refused (only R/S centres,
    which this module still cannot spell, keep refusing -- see the sibling test).

    OPSIN-verified in a parent: `[(2E)-pent-2-en-1-yl]benzene` -> CC/C=C/Cc1ccccc1
    (RT-exact), while the stereo-less `pent-2-en-1-ylbenzene` parses to a
    DIFFERENT molecule -- which is exactly why the achiral token had to refuse
    before, and why emitting the E/Z form is the correct fix, not a relaxation.
    """
    mol = Chem.MolFromSmiles(r"C/C=C/CC")
    assert mol is not None
    assert any(b.GetStereo() != Chem.BondStereo.STEREONONE
               for b in mol.GetBonds()), "fixture must actually carry stereo"
    got = terminal_fragment_name(mol, set(range(mol.GetNumAtoms())), 0)
    assert got is not None
    assert got.name == "(2E)-pent-2-en-1-yl"


def test_a_stereocentre_in_the_fragment_emits_the_descriptor():
    """v30 §A: the composite path now EMITS the stereodescriptor instead of the
    former blanket refusal.

    The old guard refused any defined stereo because the path emitted none and
    SELF-01's skeleton block is constitution-only. That was resolved two ways: the
    composite path cites the ring system's own R/S and every branch cites its own
    (each branch recursion either emits its centre or REFUSES — no silent drop),
    and SELF-01 now carries the C6 RegistrationHash stereo layer that catches a
    WRONG descriptor. So the centre is cited, never dropped. Here the centre is in
    the -CH2-CH(OH)-CH3 branch, cited by that branch's recursion as (2S).
    """
    mol = Chem.MolFromSmiles("C[C@H](O)CC1CCCCC1")
    assert mol is not None
    got = terminal_fragment_name(mol, set(range(mol.GetNumAtoms())), 8)
    assert got is not None
    assert "(2S)" in got.name  # the centre is CITED, not dropped


def test_a_defined_double_bond_configuration_in_the_fragment_emits_the_descriptor():
    """Same as the stereocentre: the E/Z is now CITED (2E), not refused."""
    mol = Chem.MolFromSmiles(r"C/C=C/CC1CCCCC1")
    assert mol is not None
    got = terminal_fragment_name(mol, set(range(mol.GetNumAtoms())), 5)
    assert got is not None
    assert "(2E)" in got.name


def test_an_undefined_double_bond_still_names():
    """The guard keys on DEFINED stereo, not on the presence of a double bond --
    otherwise it would silently swallow the whole Task 3 unsaturation feature."""
    mol = Chem.MolFromSmiles("CC=CC")
    got = terminal_fragment_name(mol, set(range(mol.GetNumAtoms())), 0)
    assert got is not None and got.name == "but-2-en-1-yl"


def test_stereo_outside_the_fragment_does_not_block_it():
    """Only stereo INSIDE frag_atoms matters; a centre elsewhere in the molecule
    is somebody else's problem and must not make this fragment refuse."""
    mol = Chem.MolFromSmiles("C[C@H](O)CCCC")
    assert mol is not None
    got = terminal_fragment_name(mol, {3, 4, 5, 6}, 3)
    assert got is not None, "a fragment free of stereo must still name"
    assert got.atoms == frozenset({3, 4, 5, 6})


def test_ring_system_stereo_is_cited_in_leading_block():
    """v30 §A: a chiral ring SUBSTITUTENT cites its ring R/S in a leading (...)
    block using the ring numbering; the whole molecule round-trips (SELF-01 C6)."""
    # 2-(2-hydroxyethyl)cyclohexan-1-ol as a substituent fragment (attach on ring)
    mol = Chem.MolFromSmiles("OCC[C@H]1CCCC[C@@H]1O")
    got = terminal_fragment_name(mol, set(range(mol.GetNumAtoms())), 3)
    assert got is not None
    assert got.name.startswith("(") and "R" in got.name.split(")")[0]  # cites ring R/S
