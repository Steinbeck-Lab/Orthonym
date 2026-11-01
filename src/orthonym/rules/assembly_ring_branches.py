"""A two-ring assembly that is the parent of a molecule whose other rings hang on it by a chain.

``### **** SENIORITY ORDER FOR PARENT STRUCTURES`` (the Blue Book) applies the criteria for the senior
parent "in order, until a decision is reached", before "those applicable to rings and ring systems (see ":
 the principal characteristic group, **** (:18917) the senior atom ("N > P > As > Sb > Bi > Si > Ge >
Sn > Pb > B > Al > Ga > In > Tl > O > S > Se > Te > C"), then ``### **** Systems composed of rings and chains``
(:19336), (1):19340 "Within the same class, a ring or ring system has seniority over a chain". Among rings, ``****`` (:20571) "Seniority criteria for ring
assemblies are based on the appropriate criteria for rings and ring systems given in ", whose criterion
"(d) has the greater number of rings" (``****``,:19415) makes a two-ring assembly senior to a single ring of
the same kind. The printed row is ``4-methoxy-1,1'-biphenyl (PIN) (not 4-phenylanisole; no substitution on anisole
for PINs) [not 1-methoxy-4-phenylbenzene; the biphenyl ring system is senior to a single benzene ring]`` (:27722,
under ``### **** Retained names``); a third ring on a chain changes nothing in it: the ring on the chain is
one more single ring, junior to the assembly, and the chain is cited with the prefix of the ring it holds
(``C6H5-CH2-O-`` is "benzyloxy (preferred prefix)",:24591 under ``## **** CONSTRUCTION OF LINEAR COMPOUND AND/OR
COMPLEX SUBSTITUENT PREFIXES``).

``rules.ring_assemblies.detect_ring_assembly`` takes a molecule only when every ring system of it joins the
assembly, so a molecule with one more ring (``benzyloxy``, ``phenoxy``, ``benzyl``, ``cyclohexylmethoxy``) reached
the benzene writer, which names the assembly as a benzene with a phenyl group ('1-(benzyloxy)-4-phenylbenzene');
``rules.ring_assembly_screen`` takes the PIN label from such a name. This module is the caller of
``detect_ring_assembly`` for the molecule named as a whole that finds the assembly among the ring systems, by the
criteria above, and offers the name of the assembly with the other rings in its substituent prefixes.

The assembly is the parent when

* the molecule is one component with no charge, isotope, radical or stereo;
* no chain is the parent (``features.chain_is_parent``) and every other ring hangs on the assembly by a chain: a ring
  bonded to a ring of the assembly directly is a matter of ``****`` and ``****`` (:24153, "When
  assemblies of otherwise identical rings contain both mancude and saturated rings, the use of hydro prefixes is
  preferred", the case of 'cyclohexyl' on a biphenyl), which the other detectors of ``rules.ring_assemblies`` read;
* the molecule has a principal characteristic group: every one sits on the assembly and on no other ring system, and
  the assembly is the parent whatever the other rings are: the parent has "the maximum number of
  substituents corresponding to the principal characteristic group"; the criteria for rings come after it); or the
  molecule has none: the atoms outside the assembly are carbon, oxygen, sulfur and halogen atoms (a more senior
  atom of could be a parent of its own) and the ring systems joined by direct bonds, identical ones merged
  (``assembly_units``), have one unit of two ring systems that is strictly senior to every other unit by
  ``ring_selection.ring_system_score`` (a) to (g), then the type order of: a tie is no decision
  and is declined).

Only a two-ring assembly: three to six identical systems take composite locants in the PIN (``****``,
:15653), which ``rules.composite_assembly`` writes where the OPSIN in use reads them and which a molecule with other
rings is outside of. The name is built by ``name_ring_assembly`` and returned only when OPSIN 2.9.0 reads it back to
the full InChIKey of the molecule, as ``detect_hydro_ring_assembly`` does: a molecule this module declines keeps the
classification it had.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from rdkit import Chem

__all__ = ["detect_assembly_with_ring_branches"]

#: (:18917): the atoms outside the assembly that cannot be the senior atom of a parent of their own
#: (O, S and halogen follow the carbon chains and rings; a nitrogen, silicon,... atom would precede them).
_PLAIN_ELEMENTS = frozenset({6, 8, 16, 9, 17, 35, 53})


def _plain(mol) -> bool:
    """One component with no charge, isotope, radical or stereo."""
    if len(Chem.GetMolFrags(mol)) != 1:
        return False
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() or atom.GetIsotope() or atom.GetNumRadicalElectrons():
            return False
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return False
    return all(b.GetStereo() == Chem.BondStereo.STEREONONE for b in mol.GetBonds())


def _principal_groups_on(mol, features, assembly: Set[int], groups) -> bool:
    """True when every principal characteristic group of ``features`` sits on ``assembly`` and on no other ring
    system."""
    from .parent_selection import is_principal_group_on_ring
    pg = getattr(features, 'principal_group', None)
    others = [set(g) for g in groups if not set(g) <= assembly]
    for match in features.principal_group_atoms:
        if not is_principal_group_on_ring(mol, assembly, [match], pg):
            return False
        if any(is_principal_group_on_ring(mol, other, [match], pg) for other in others):
            return False
    return True


def _joined_directly(mol, groups, assembly: Set[int]) -> bool:
    """True when a bond joins an atom of ``assembly`` to an atom of another ring system."""
    others = set().union(*[set(g) for g in groups if not set(g) <= assembly])
    return any((b.GetBeginAtomIdx() in assembly and b.GetEndAtomIdx() in others)
               or (b.GetEndAtomIdx() in assembly and b.GetBeginAtomIdx() in others) for b in mol.GetBonds())


def _two_ring_units(mol, groups) -> List[Set[int]]:
    from ..assembly.complex_ring_group import assembly_units
    units = [set(u) for u in assembly_units(mol, groups)]
    return [u for u in units if sum(1 for g in groups if set(g) <= u) == 2]


def _senior_unit(mol, groups, features) -> Optional[Set[int]]:
    """The two-ring assembly that is the parent of ``mol`` among its ring systems, or None (see the module
    docstring)."""
    if getattr(features, 'principal_group_atoms', None):
        holders = [u for u in _two_ring_units(mol, groups) if _principal_groups_on(mol, features, u, groups)]
        return holders[0] if len(holders) == 1 else None
    if getattr(features, 'principal_group', None):
        return None
    from ..assembly.complex_ring_group import assembly_units
    from .ring_selection import ring_system_score
    units = assembly_units(mol, groups)
    scored = sorted((ring_system_score(mol, set(u)), min(u)) for u in units)
    if len(scored) < 2 or scored[0][0] == scored[1][0]:
        return None
    best = set(next(u for u in units if min(u) == scored[0][1]))
    if best not in _two_ring_units(mol, groups):
        return None
    #: a more senior atom outside the assembly could be the parent of its own
    plain = all(a.GetIdx() in best or a.GetAtomicNum() in _PLAIN_ELEMENTS for a in mol.GetAtoms())
    return best if plain else None


def detect_assembly_with_ring_branches(mol, ring_systems: List[Set[int]], features=None) -> Optional[Dict]:
    """The assembly info of the two-ring assembly that is the parent of ``mol`` among its ring systems, with its
    name (``name_ring_assembly``, read back by OPSIN 2.9.0 to the full InChIKey of ``mol``), or None (see the module
    docstring). ``features`` is the molecular features of the whole molecule: without it the principal
    characteristic groups are unknown and nothing is offered."""
    if features is None or getattr(features, 'chain_is_parent', False) or len(ring_systems) < 3:
        return None
    try:
        return _detect(mol, ring_systems, features)
    except Exception:  # noqa: BLE001 - a structure this reader cannot name is declined and keeps its classification
        return None


def _detect(mol, ring_systems: List[Set[int]], features) -> Optional[Dict]:
    from ..assembly.complex_ring_group import ring_groups
    from .ring_assemblies import _hydro_certified, detect_ring_assembly, name_ring_assembly
    groups = ring_groups(mol)
    if len(groups) < 3:
        return None
    assembly = _senior_unit(mol, groups, features) if _plain(mol) else None
    if assembly is None or _joined_directly(mol, groups, assembly):
        return None
    members = [set(rs) for rs in ring_systems if set(rs) <= assembly]
    info = detect_ring_assembly(mol, members)
    if (info is None or info.get('count') != 2 or info.get('double_bond_junction') or info.get('replacement')
            or info.get('mancude_tautomer') or info.get('hydro')):
        return None
    # A principal group that the assembly builders can only cite as a prefix leaves the single-ring name standing
    # (``name_ring_assembly`` returns None for a ``pcg_prefix_declines`` request instead of a below-PIN name).
    info['pcg_prefix_declines'] = True
    name = name_ring_assembly(mol, info, features)
    if not name or not _hydro_certified(mol, name):
        return None
    info['ring_branches'] = True
    info['name'] = name
    return info
