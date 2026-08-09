"""v30 BLOCKER fix — ring-ASSEMBLY as a substituent takes the P-28.3 primed
free-valence form ``[1,1'-biphenyl]-4-yl``, never the yl-less parent hydride
``1,1'-biphenyl`` (an OPSIN-unparseable substituent token).

Root cause (pre-fix): ``name_ring_system_substituent``'s enumerator fallback
routed a biphenyl fragment through the generic cascade, which returned the
PARENT hydride ``1,1'-biphenyl``; wrapping it shipped ``(1,1'-biphenyl)methyl``
etc. — SELF-01 suppressed it on the PIN path (abstain) and T4 shipped it
unverified (wrong). Fix routes ring-assembly substituent fragments through the
existing ``name_ring_assembly_prefix`` (the same P-28.3 builder the whole-
molecule composer already trusts), with a fail-closed backstop.

Reference note: and decompose biphenyl-as-substituent to
the non-preferred ``4-phenylphenyl``; the bracketed primed ``[1,1'-biphenyl]-4-yl``
is a net PIN-correctness edge for Orthonym (connector-seeded
prime-the-second-ring numbering, but the brackets + free-valence demotion are ours).
"""
from rdkit import Chem

from orthonym.rules.ring_substituents import (
    get_ring_substituent_name,
    name_ring_system_substituent,
    _fragment_is_ring_assembly,
)
from orthonym.assembly.substituent_enumerator import name_substituent


def _biphenyl_attach(smi):
    """(mol, ring_atoms, attach_ring_atom, ch3) for a methyl-marked assembly."""
    mol = Chem.MolFromSmiles(smi)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    ri = mol.GetRingInfo()
    ring_atoms = tuple(a.GetIdx() for a in mol.GetAtoms()
                       if ri.NumAtomRings(a.GetIdx()) > 0)
    ch3 = [a.GetIdx() for a in mol.GetAtoms()
           if a.GetSymbol() == 'C' and ri.NumAtomRings(a.GetIdx()) == 0][0]
    attach = [n.GetIdx() for n in mol.GetAtomWithIdx(ch3).GetNeighbors()
              if n.GetIdx() in ring_atoms][0]
    return mol, ring_atoms, attach, ch3


def test_biphenyl_para_substituent_is_bracketed_yl():
    mol, ring_atoms, attach, _ = _biphenyl_attach("Cc1ccc(-c2ccccc2)cc1")
    assert get_ring_substituent_name(mol, ring_atoms, attach) == "[1,1'-biphenyl]-4-yl"


def test_biphenyl_meta_and_ortho_locants():
    mol, ring_atoms, attach, _ = _biphenyl_attach("Cc1cccc(-c2ccccc2)c1")
    assert get_ring_substituent_name(mol, ring_atoms, attach) == "[1,1'-biphenyl]-3-yl"
    mol, ring_atoms, attach, _ = _biphenyl_attach("Cc1ccccc1-c1ccccc1")
    assert get_ring_substituent_name(mol, ring_atoms, attach) == "[1,1'-biphenyl]-2-yl"


def test_biphenyl_on_methylene_composes_with_enclosing_marks():
    mol, ring_atoms, attach, ch3 = _biphenyl_attach("Cc1ccc(-c2ccccc2)cc1")
    frag = list(ring_atoms) + [ch3]
    assert name_substituent(mol, frag, ch3, allow_mancude=True) \
        == "([1,1'-biphenyl]-4-yl)methyl"


def test_never_emits_yl_less_parent_hydride():
    """The whole point: the substituent namer must not return '1,1'-biphenyl'."""
    mol, ring_atoms, attach, _ = _biphenyl_attach("Cc1ccc(-c2ccccc2)cc1")
    for am in (False, True):
        out = name_ring_system_substituent(mol, list(ring_atoms), attach,
                                            allow_mancude=am)
        assert out == "[1,1'-biphenyl]-4-yl"
        assert out != "1,1'-biphenyl"


def test_fused_polycycle_is_not_an_assembly():
    """Naphthalene (one fused system) must NOT be claimed as a ring assembly —
    it keeps its own naphthalen-2-yl producer."""
    mol = Chem.MolFromSmiles("Cc1ccc2ccccc2c1")
    ri = mol.GetRingInfo()
    ring_atoms = [a.GetIdx() for a in mol.GetAtoms()
                  if ri.NumAtomRings(a.GetIdx()) > 0]
    assert _fragment_is_ring_assembly(mol, ring_atoms) is False


def test_terphenyl_detected_as_assembly():
    mol, ring_atoms, _, _ = _biphenyl_attach(
        "Cc1ccc(-c2ccc(-c3ccccc3)cc2)cc1")
    assert _fragment_is_ring_assembly(mol, ring_atoms) is True
