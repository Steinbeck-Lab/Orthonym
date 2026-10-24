"""Numbering of a decorated ring prefix among the numberings the free valence leaves open.

``substituent_enumerator._recursive_fragment_substituent_name`` numbers a ring core by the
criteria it models (heteroatoms, free valence, ene/yne, the prefix locants taken together). Several
numberings can tie on all of them (the ring of ``2,3-dihydroxy-4,5,6-tris(...)cyclohexyl`` runs
either way round from C-1). the Blue Book, " NUMBERING" (:3219), decides the tie in this order:

(f) "detachable alphabetized prefixes, all considered together in a series of increasing numerical
order" (:3300; the caller's key);
(g) "lowest locants for the substituent cited first as a prefix in the name" (:3307; the examples
``4-methyl-5-nitrooctanedioic acid (PIN)``, ``1-methyl-4-nitronaphthalene (PIN) (not 4-methyl-1-
nitronaphthalene)``);
(j) "the lower locant is assigned to CIP stereodescriptors Z, R, M, and r (pseudoasymmetry) that are
preferred to E, S, P, and s" (:3346).

Without this step the first of the tied numberings found decided, and that is the order of the atoms
in the input.

The numberings that tie are the numbering found composed with an automorphism of the core that keeps
the attachment atom, each atom's element, charge, hydrogen count, aromaticity and exocyclic double
bonds (so every locant set of the core name is unchanged) and maps the carriers of the decorations
onto themselves.
"""
from __future__ import annotations

from typing import Dict, List

from rdkit import Chem


def core_automorphisms(mol, core, attach_idx) -> List[Dict[int, int]]:
    """The automorphisms of the ring core that fix the attach atom and keep each atom's element, charge,
    hydrogen count, aromaticity and number of exocyclic double bonds (an automorphism then keeps every
    locant set of the core name: indicated hydrogen, hydro, ene, oxo positions)."""
    core = set(core)
    colour = {}
    for a in core:
        atom = mol.GetAtomWithIdx(a)
        exo = sum(1 for b in atom.GetBonds()
                  if b.GetOtherAtomIdx(a) not in core and b.GetBondTypeAsDouble() >= 2.0)
        colour[a] = (atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetTotalNumHs(),
                     atom.GetIsAromatic(), exo, a == attach_idx)
    ids = {c: i + 1 for i, c in enumerate(sorted(set(colour.values())))}
    rw = Chem.RWMol(mol)
    for a in sorted(set(range(mol.GetNumAtoms())) - core, reverse=True):
        rw.RemoveAtom(a)
    order = sorted(core)
    sub = rw.GetMol()
    for k, a in enumerate(order):
        atom = sub.GetAtomWithIdx(k)
        atom.SetIsotope(ids[colour[a]])
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    sub.UpdatePropertyCache(strict=False)
    matches = sub.GetSubstructMatches(sub, uniquify=False, useChirality=False, maxMatches=5000)
    return [{order[i]: order[m[i]] for i in range(len(order))} for m in matches]


def tied_numberings(mol, core, attach_idx, pos, carriers) -> List[Dict[int, int]]:
    """Numberings equal to ``pos`` on every criterion before (g): ``pos`` composed with an automorphism
    of the core that maps the set of carrier atoms onto itself."""
    carrier_set = set(carriers)
    out = []
    for aut in core_automorphisms(mol, core, attach_idx):
        if {aut[c] for c in carrier_set} == carrier_set:
            out.append({a: pos[aut[a]] for a in pos})
    return out


def stereo_atoms(mol, core) -> List[int]:
    """The core atoms whose locant a stereodescriptor cites: stereocentres and the ends of a stereogenic
    ring double bond."""
    atoms = set()
    for a in core:
        atom = mol.GetAtomWithIdx(a)
        if atom.HasProp('_CIPCode') or atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            atoms.add(a)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in core and j in core and bond.GetStereo() != Chem.BondStereo.STEREONONE:
            atoms.update((i, j))
    return sorted(atoms)


def g_key(numbering, named):
    """ (g): the locants of the prefixes in the order they are cited (alphanumerical order)."""
    from .naming_utils import alpha_sort_key
    names = sorted({nm for _, nm in named}, key=alpha_sort_key)
    return tuple(tuple(sorted(numbering[ra] for ra, nm in named if nm == n)) for n in names)


def g_winners(mol, core, attach_idx, pos, named) -> List[Dict[int, int]]:
    """The numberings that tie with ``pos`` on (a)-(f) and have the lowest (g) key."""
    tied = tied_numberings(mol, core, attach_idx, pos, [ra for ra, _ in named])
    if len(tied) <= 1:
        return [pos]
    keys = [g_key(numbering, named) for numbering in tied]
    best = min(keys)
    return [numbering for numbering, key in zip(tied, keys) if key == best]


def j_key(mol, core, numbering):
    """ (j): ``(locants of the Z/R centres and bonds, locants of the E/S ones)``; lower is
    preferred. A stereogenic ring double bond is cited at the lower locant of its two ends (the higher
    when it closes the ring). Only the CIP labels the molecule carries are read."""
    n = len(core)
    preferred, other = [], []
    for a in core:
        code = mol.GetAtomWithIdx(a).GetPropsAsDict().get('_CIPCode')
        if code in ('R', 'S'):
            (preferred if code == 'R' else other).append(numbering[a])
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        code = bond.GetPropsAsDict().get('_CIPCode')
        if i in numbering and j in numbering and code in ('E', 'Z'):
            lo, hi = sorted((numbering[i], numbering[j]))
            loc = hi if (lo == 1 and hi == n) else lo
            (preferred if code == 'Z' else other).append(loc)
    return tuple(sorted(preferred)), tuple(sorted(other))


def prefix_order_numbering(mol, core, attach_idx, pos, named):
    """``pos`` resolved by (g), then (j), among the numberings that tie on (a)-(f). A tie that
    remains keeps ``pos`` when it is among the winners (the numbering found first), else the first
    winner."""
    winners = g_winners(mol, core, attach_idx, pos, named)
    if len(winners) > 1:
        keys = [j_key(mol, core, numbering) for numbering in winners]
        best = min(keys)
        winners = [numbering for numbering, key in zip(winners, keys) if key == best]
    return pos if pos in winners else winners[0]
