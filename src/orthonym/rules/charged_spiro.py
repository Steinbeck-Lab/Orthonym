"""The name of a spiro cation: the spiro parent, its groups and the suffix 'ylium' or 'ium'.

 SKELETAL REPLACEMENT ('a') NOMENCLATURE FOR CATIONS (the Blue Book), method (1): the compound is named
with the neutral 'a' prefixes and the cationic centre by the suffix 'ium' or 'ylium'. A quaternary onium at a spiro
junction has four bonds, all in rings: one above the standard bonding number of nitrogen, phosphorus, arsenic and
antimony, so no neutral parent hydride has it. The book's name is that of the cation as the hydride ion lost from the
λ5 hydride, '5λ5-arsaspiro[4.4]nonan-5-ylium (PIN)' (:41802), and '1H-2λ5-spiro[isoquinoline-2,2'-pyrido[1,2-a]
pyrazin]-2-ylium (PIN)' (:41808) for the nitrogen; the 'ium' of a parent with the standard valence ('1-methyl-
1-azabicyclo[2.2.1]heptan-1-ium (PIN)',:41773) is for the cation that a hydron makes.

The spiro parent is named by ``spiro.name_spiro_system`` off the λ5 hydride of the cation (the cation with its charge
replaced by a hydrogen atom: the numbering is the one of the molecule, and the replacement prefix cites the λ of the
atom by the rule of the neutral spiro route). ``name_spiro_system`` hands back the parent alone (its
``substituents_included`` is False): a group on a ring atom is cited as a prefix of the parent on the numbering it
chose, the way the neutral spiro route cites it (``composer._enrich_complex_ring_with_subs``).
"""
from typing import Optional

from rdkit import Chem, rdBase

#: the elements whose standard bonding number is three: a junction cation with four bonds is the ylium of the λ5 hydride
_YLIUM_ELEMENTS = frozenset({'N', 'P', 'As', 'Sb'})


def lambda_hydride(mol, cation_idx: int) -> Optional[Chem.Mol]:
    """The λ5 hydride of the cation: the molecule with the charge of the atom replaced by one hydrogen atom, so that
    its bonding number is five. Not sanitized (the atom is over-valent for RDKit); None when it cannot be built."""
    try:
        with rdBase.BlockLogs():
            rw = Chem.RWMol(mol)
            atom = rw.GetAtomWithIdx(cation_idx)
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
            hydride = rw.GetMol()
            hydride.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(hydride)
        return hydride
    except (RuntimeError, ValueError):
        return None


def name_charged_spiro(mol, cation_idx: int) -> str:
    """The candidate name of the spiro system whose junction carries the one cationic ring heteroatom
    ``cation_idx`` (see the module docstring), or '' when the shape is out of scope: not a ring heteroatom of charge
    +1, not the junction of a pure spiro system, or a parent that cannot be named. No read-back is made here."""
    from ..perception.rings import get_spiro_atoms
    from .spiro import name_spiro_system
    try:
        atom = mol.GetAtomWithIdx(cation_idx)
    except (RuntimeError, IndexError, OverflowError):
        return ''
    # a single cationic (+1) ring HETEROATOM: a spiro ring carbon is already four-coordinate, and only a skeletal
    # heteroatom takes the replacement 'a' prefix
    if atom.GetFormalCharge() != 1 or not atom.IsInRing() or atom.GetSymbol() == 'C':
        return ''
    if not ring_system_is_pure_spiro(mol, cation_idx):
        return ''
    # the junction: all four bonds in rings, so the neutral atom is over-valent and has no group to sever
    if cation_idx not in get_spiro_atoms(mol):
        return ''
    ylium = atom.GetSymbol() in _YLIUM_ELEMENTS and atom.GetTotalValence() == 4
    named = lambda_hydride(mol, cation_idx) if ylium else mol
    result = name_spiro_system(named) if named is not None else None
    if result is None:
        return ''
    parent_name, ring_atoms, atom_to_locant, _subs = result
    locant = atom_to_locant.get(cation_idx)
    if cation_idx not in ring_atoms or locant is None:
        return ''
    return ium_with_ring_groups(mol, parent_name, locant, ring_atoms, atom_to_locant,
                                'ylium' if ylium else 'ium')


def ring_system_is_pure_spiro(mol, cation_idx: int) -> bool:
    """Whether the ring system of the cation is a pure spiro system (N spiro atoms joining N+1 rings) and holds
    every spiro atom of the molecule. The rings of the groups on it (a phenyl, a piperidinyl) are no part of the
    system: ``spiro.is_spiro_system`` counts every ring of the molecule and declined the spiro ammonium whose
    groups held a ring."""
    from ..perception.rings import get_spiro_atoms
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    system = [r for r in rings if cation_idx in r]
    if not system:
        return False
    grown = True
    while grown:
        grown = False
        for r in rings:
            if r not in system and any(r & other for other in system):
                system.append(r)
                grown = True
    atoms = set().union(*system)
    spiro = get_spiro_atoms(mol)
    return bool(spiro) and spiro <= atoms and len(system) == len(spiro) + 1


def ium_with_ring_groups(mol, parent_name: str, locant, ring_atoms, atom_to_locant, suffix: str = 'ium') -> str:
    """``<groups><parent>-<locant>-<suffix>``, the 'e' of the parent elided before the suffix ('ium' or 'ylium',,
    the Blue Book 'General rule for systematically naming cationic centers in parent hydrides'), or ''
    when the groups cannot be cited. The caller reads the name back to the full InChIKey."""
    from ..assembly.composer import _enrich_complex_ring_with_subs
    stem = parent_name[:-1] if parent_name.endswith('e') else parent_name
    return _enrich_complex_ring_with_subs(mol, f"{stem}-{locant}-{suffix}", ring_atoms, atom_to_locant) or ''
