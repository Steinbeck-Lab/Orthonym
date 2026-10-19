"""A benzene ring fused to a heteromonocycle with a halogen ring atom in the lambda state.

The benziodoxolones and their relatives: ``1-methoxy-1λ3,2-benziodoxol-3(1H)-one (PIN)``
 "Nomenclature based on halogen parent hydrides", the Blue Book, the example at
:39674) and ``1H-1λ3-benziodole (PIN)`` (:39692). Before this module the fusion producers had no
component for a ring halogen, so every such system went unnamed.

The name is the one of a benzoheterocycle, "Heteromonocyclic components fused to a
benzene ring" (:11813): "the locant(s) indicating the position(s) of the heteroatom(s) at the front
of the name consisting of the fusion prefix 'benzo' followed by... a Hantzsch-Widman systematic
name... The locants cited correspond to the full bicyclic structure. As in Hantzsch-Widman names,
locants are placed in the order corresponding to the order of citation of the heteroatoms in the
heterocyclic component. The locant '1' is always assigned to the atom of the heterocyclic component
next to a fusion atom... The letter 'o' of the 'benzo' prefix is elided when followed by a vowel.
Indicated hydrogen is placed at the front of the name, when required."

The lambda symbol of a fused system, "Fused ring systems with skeletal atoms with nonstandard
bonding numbers" (:14531) and (:12452), "follows the locant of the atom with the
nonstandard bonding number in accordance with the numbering of the fused system". In a benzo name
the locant of the halogen is one of the heteroatom locants at the front, so the symbol sits inside
that list: ``1λ4,5-benzodithiepine (PIN)`` and ``3H-3λ3,2,4-benziodadioxepine (PIN)`` (:14549,
:14557). The Hantzsch-Widman builder (``heterocycles.build_hw_name``) already writes that list for a
monocycle (``1,3λ5-oxaphosphole``,; this module gives it the locants of the fusion
numbering and puts 'benzo' in front of its stem.

Everything else is the rules the other fused producers share: the fusion numbering
(``fusion_numbering.compute_fused_numbering``), the indicated, added and hydro hydrogen of the
mancude parent (``ring_hydrogen``, which counts the halogen with its own bonding number:
``lambda_convention.ring_lambda_valence``) and the assembly of the substituents and the suffix
(``fused_rings._assemble_fused_heterocycle_name``). Scope, everything else is declined (``None``):
two rings, one a benzene ring and the other of five to seven members with a halogen of
``lambda_convention.FUSED_RING_LAMBDA_HALOGENS``; no charged, radical or isotope-labelled ring atom;
no ring atom of nonstandard bonding number but those halogens; every heteroatom at a numbered
(not a lettered) position.
"""
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem

#: ring sizes of the heteromonocycle of a benzoheterocycle name that this producer builds
_HETERO_RING_SIZES = (5, 6, 7)


def _rings_of_system(mol) -> Optional[Tuple[Set[int], Set[int]]]:
    """(benzene ring, heteromonocycle) of the two-ring ortho-fused system that holds a ring
    halogen in the lambda state, else None. Every other ring of the molecule must stand apart from
    the two (a ring of a substituent is named as the substituent it is)."""
    from .lambda_convention import ring_lambda_valence
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    for hetero in rings:
        if (len(hetero) not in _HETERO_RING_SIZES
                or not any(ring_lambda_valence(mol, a) is not None for a in hetero)):
            continue
        for benzo in rings:
            shared = hetero & benzo
            if (benzo is hetero or len(shared) != 2 or len(benzo) != 6
                    or mol.GetBondBetweenAtoms(*sorted(shared)) is None
                    or not all(mol.GetAtomWithIdx(a).GetSymbol() == 'C'
                               and mol.GetAtomWithIdx(a).GetIsAromatic() for a in benzo)):
                continue
            if any(r is not hetero and r is not benzo and r & (hetero | benzo) for r in rings):
                return None
            return benzo, hetero
    return None


def _plain_atoms(mol, ring_atoms: Set[int]) -> bool:
    """No charged, radical or isotope-labelled ring atom, and no lambda atom but an admitted
    ring halogen: the atoms this producer's parent describes as they are."""
    from .lambda_convention import nonstandard_bonding_number, ring_lambda_valence
    for a in ring_atoms:
        atom = mol.GetAtomWithIdx(a)
        if atom.GetFormalCharge() or atom.GetIsotope() or atom.GetNumRadicalElectrons():
            return False
        if nonstandard_bonding_number(mol, a) is not None and ring_lambda_valence(mol, a) is None:
            return False
    return True


def benzo_hw_name(mol, hetero_ring: Set[int], atom_to_locant: Dict[int, object]) -> Optional[str]:
    """The mancude benzoheterocycle name of the ring system under ``atom_to_locant``:
    '1λ3,2-benziodoxole', '1λ3-benziodole', '3λ3,2,4-benziodadioxepine'. None when a heteroatom
    has no numbered locant or the Hantzsch-Widman builder has no name for the ring."""
    from .heterocycles import build_hw_name
    from .lambda_convention import fused_lambda_entries
    hetero: List[Tuple[int, str]] = []
    for a in sorted(hetero_ring):
        sym = mol.GetAtomWithIdx(a).GetSymbol()
        if sym == 'C':
            continue
        loc = atom_to_locant.get(a)
        if loc is None or not str(loc).isdigit():
            return None
        hetero.append((int(loc), sym))
    ring_atoms = set(atom_to_locant)
    entries = fused_lambda_entries(mol, ring_atoms, atom_to_locant)
    if not entries or not hetero:
        return None
    lam = {loc: n for loc, n in entries}
    # the stem and the 'a' prefixes are those of the heteromonocycle; its own locants are the
    # fusion locants, cited once at the front, each lambda symbol after its locant
    hw = build_hw_name(hetero, len(hetero_ring), is_saturated=False, is_aromatic=True,
                       lambda_by_locant=lam, cite_locants=True)
    if not hw or '-' not in hw:
        return None
    front, stem = hw.split('-', 1)
    if not stem or not stem[0].isalpha():
        return None
    benzo = 'benz' if stem[0] in 'aeiou' else 'benzo'
    return f"{front}-{benzo}{stem}"


def name_halogen_benzo_system(mol) -> Optional[str]:
    """The name of a benzene ring fused to a heteromonocycle with a halogen ring atom in the
    lambda state, with its substituents, suffix and hydrogen; None (the caller keeps its own
    answer) when the system is outside the scope of the module docstring."""
    from . import fused_rings as fr
    from . import ring_hydrogen
    from .fusion_numbering import compute_fused_numbering
    pair = _rings_of_system(mol)
    if pair is None:
        return None
    benzo, hetero = pair
    ring_atoms = benzo | hetero
    if not _plain_atoms(mol, ring_atoms) or not fr._exocyclic_atoms_accounted(mol, ring_atoms):
        return None
    canonical = compute_fused_numbering(mol, ring_atoms)
    if not canonical:
        return None
    canon_rank = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    atoms_by_rank = sorted(ring_atoms, key=lambda a: canon_rank[a])
    chosen = fr._select_numbering_with_ring_hydrogen(
        mol, ring_atoms, canonical, atoms_by_rank, any_ring_atoms=True)
    if chosen is None:
        return None
    atom_to_locant, subs = chosen
    suffix = fr._ring_hydrogen_suffix_atoms(
        subs, atom_to_locant, ring_hydrogen.exocyclic_double_atoms(mol, ring_atoms))
    if suffix is None:
        return None
    rh = ring_hydrogen.ring_hydrogen(mol, ring_atoms, suffix, atom_to_locant)
    if rh is None:
        return None
    parent = benzo_hw_name(mol, hetero, atom_to_locant)
    front = ring_hydrogen.parent_prefix(rh, atom_to_locant)
    if parent is None or front is None:
        return None
    from ..data.halogen_ring_parents import PRINTED_PIN_PARENTS
    if parent not in PRINTED_PIN_PARENTS:
        # valid, but the book prints no PIN on this ring system: labelled below the PIN, name-scoped
        fr._record_non_pin_spelling(parent[:-1])
    return fr._assemble_fused_heterocycle_name(
        mol, fr._join_hydrogen_prefix(front, parent),
        fr._hydrogen_marked_substituents(subs, rh, atom_to_locant), atom_to_locant)
