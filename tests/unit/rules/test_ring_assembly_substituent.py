""" BLOCKER fix — ring-ASSEMBLY as a substituent takes the primed
free-valence form ``[1,1'-biphenyl]-4-yl``, never the yl-less parent hydride
``1,1'-biphenyl`` (an OPSIN-unparseable substituent token).

Root cause (pre-fix): ``name_ring_system_substituent``'s enumerator fallback
routed a biphenyl fragment through the generic cascade, which returned the
PARENT hydride ``1,1'-biphenyl``; wrapping it shipped ``(1,1'-biphenyl)methyl``
etc. — suppressed it on the PIN path (abstain) and shipped it
unverified (wrong). Fix routes ring-assembly substituent fragments through the
existing ``name_ring_assembly_prefix`` (the same builder the whole-
molecule composer already trusts), with a fail-closed backstop.

Note: a naive decomposition renders biphenyl-as-substituent as
the non-preferred ``4-phenylphenyl``; the bracketed primed ``[1,1'-biphenyl]-4-yl``
is a net PIN-correctness edge for Orthonym (connector-seeded
prime-the-second-ring numbering, but the brackets + free-valence demotion are ours).
"""
import pytest
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


@pytest.mark.parametrize("smi,expected", [
    ("Cc1ccc(-c2ccccn2)nc1", "[2,2'-bipyridin]-5-yl"),
    ("Cc1ccc(-c2cccs2)s1", "[2,2'-bithiophen]-5-yl"),
])
def test_heteroaromatic_assembly_free_valence_not_on_junction(smi, expected):
    """review BLOCKER 1: the -yl locant must use the junction-aware per-system
    numbering, not orient_heterocycle-in-isolation which placed it ON the 2,2'
    junction ('[2,2'-bithiophen]-2-yl'). Correct is 5-yl (the far position)."""
    m = Chem.MolFromSmiles(smi)
    ri = m.GetRingInfo()
    ring = tuple(a.GetIdx() for a in m.GetAtoms() if ri.NumAtomRings(a.GetIdx()) > 0)
    ch3 = [a.GetIdx() for a in m.GetAtoms()
           if a.GetSymbol() == 'C' and ri.NumAtomRings(a.GetIdx()) == 0][0]
    att = [n.GetIdx() for n in m.GetAtomWithIdx(ch3).GetNeighbors()
           if n.GetIdx() in ring][0]
    assert get_ring_substituent_name(m, ring, att) == expected


def test_fused_component_assembly_fails_closed():
    """review BLOCKER 3: a MULTI-RING FUSED assembly component (binaphthalene)
    degrades the per-system numbering to the unreliable per-atom fallback and
    emitted a PARSEABLE-WRONG '[1,1'-binaphthalen]-2-yl' (a 2,2'-binaphthalene).
    Never guess a fused-system attachment locant — fail closed."""
    m = Chem.MolFromSmiles("Cc1ccc2cc(-c3ccc4ccccc4c3)ccc2c1")
    ri = m.GetRingInfo()
    ring = tuple(a.GetIdx() for a in m.GetAtoms() if ri.NumAtomRings(a.GetIdx()) > 0)
    ch3 = [a.GetIdx() for a in m.GetAtoms()
           if a.GetSymbol() == 'C' and ri.NumAtomRings(a.GetIdx()) == 0][0]
    att = [n.GetIdx() for n in m.GetAtomWithIdx(ch3).GetNeighbors()
           if n.GetIdx() in ring][0]
    assert get_ring_substituent_name(m, ring, att) is None


def test_terphenyl_detected_as_assembly():
    mol, ring_atoms, _, _ = _biphenyl_attach(
        "Cc1ccc(-c2ccc(-c3ccccc3)cc2)cc1")
    assert _fragment_is_ring_assembly(mol, ring_atoms) is True


@pytest.mark.parametrize("smi,expected", [
    ("CC1CCCCC1C1CCCCC1", "[1,1'-bi(cyclohexan)]-2-yl"),
    ("CC1CCC1C1CCC1", "[1,1'-bi(cyclobutan)]-2-yl"),
    ("CC1CC1C1CC1", "[1,1'-bi(cyclopropan)]-2-yl"),
])
def test_saturated_assembly_component_is_parenthesised(smi, expected):
    """ (the Blue Book '[1,1'-bi(cyclohexan)]-4-yl' preferred prefix): a
    NON-retained (cycloalkane / von Baeyer) assembly component takes parentheses
    in the SUBSTITUENT prefix too, mirroring the parent path's _enclose_component
    — not the buggy paren-less '[1,1'-bicyclohexan]-2-yl' (RT-valid, so it was
    invisible to every gate; the spelling-layer blind spot, review-found)."""
    m = Chem.MolFromSmiles(smi)
    ri = m.GetRingInfo()
    ring = tuple(a.GetIdx() for a in m.GetAtoms() if ri.NumAtomRings(a.GetIdx()) > 0)
    ch3 = [a.GetIdx() for a in m.GetAtoms()
           if a.GetSymbol() == 'C' and ri.NumAtomRings(a.GetIdx()) == 0][0]
    att = [n.GetIdx() for n in m.GetAtomWithIdx(ch3).GetNeighbors()
           if n.GetIdx() in ring][0]
    assert get_ring_substituent_name(m, ring, att) == expected


def test_retained_mancude_assembly_stays_bare():
    """The parens fix must NOT touch retained mancude stems — biphenyl is
    '[1,1'-biphenyl]-4-yl' with NO parens (byte-identical)."""
    mol, ring_atoms, attach, _ = _biphenyl_attach("Cc1ccc(-c2ccccc2)cc1")
    assert get_ring_substituent_name(mol, ring_atoms, attach) == "[1,1'-biphenyl]-4-yl"


def test_terphenyl_substituent_middle_ring_locant():
    """151-03 / bug fixed on the substituent path too: para-terphenyl's
    MIDDLE ring back-attachment locant must be 4' ('1,1':4',1'''), not the buggy
    per-pair '1,1':1',1''' that OPSIN parses to a different (spiro) molecule."""
    mol, ring_atoms, attach, _ = _biphenyl_attach(
        "Cc1ccc(-c2ccc(-c3ccccc3)cc2)cc1")
    assert get_ring_substituent_name(mol, ring_atoms, attach) \
        == "[1,1':4',1''-terphenyl]-4-yl"
