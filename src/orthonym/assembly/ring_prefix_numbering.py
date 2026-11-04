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


def fusion_atom_locants(mol, core, attach_idx, carriers, pos) -> Dict[int, str]:
    """``{atom: lettered locant}`` of the fusion atoms of the fusion name of a fused core
    (``rules.fused_forms.fused_system_form``), on the numbering ``pos`` (the form's numbering composed with an
    automorphism of the core, as the ties of (g) are resolved above), or ``{}``.

    ``pos`` holds the numbered atoms only; the fusion atoms of a fused ring ('3a', '7a' of indene) have no entry
    in it, so the descriptors of their stereocentres were never cited. "Numbering of peripheral
    skeletal atoms" (the Blue Book): "Each fusion carbon atom is given the same number as the immediately
    preceding nonfusion skeletal atom, modified by a Roman letter 'a', 'b', 'c', 'd', etc.";
    (:44624) cites every stereogenic unit. Every automorphism that reproduces ``pos`` must give the same letters,
    else ``{}``."""
    from ..rules.fused_forms import fused_system_form
    try:
        form = fused_system_form(mol, sorted(core), attach_idx, list(carriers))
    except Exception:  # noqa: BLE001 - a producer that raises is a decline
        return {}
    if form is None:
        return {}
    lettered = [a for a in sorted(core) if isinstance(form.numbering.get(a), str)]
    found = {tuple(form.numbering[aut[a]] for a in lettered)
             for aut in core_automorphisms(mol, core, attach_idx)
             if all(form.numbering.get(aut[a]) == loc for a, loc in pos.items())}
    if len(found) != 1 or not all(isinstance(loc, str) for loc in next(iter(found))):
        return {}
    return dict(zip(lettered, next(iter(found))))


def with_fusion_locants(mol, core, attach_idx, carriers, pos):
    """``pos`` with the lettered locants of the stereogenic fusion atoms of a fused ring core, for the descriptors of
    the prefix: the centres at '3a' and '7a' of ``(1R,2S,3aR,7aS)-...octahydro-1H-inden-1-yl`` were left out of its
    descriptors. ``pos`` itself when no stereocentre of the core lacks a locant."""
    missing = [a for a in core if a not in pos and mol.GetAtomWithIdx(a).HasProp('_CIPCode')]
    if not missing:
        return pos
    letters = fusion_atom_locants(mol, core, attach_idx, carriers, pos)
    return {**pos, **{a: letters[a] for a in missing if a in letters}}


def citation_key(name):
    """The order in which the groups of a ring prefix are cited: (alphanumerical order) and then, for
    groups that tie on it because leaves the descriptors out (``(6R)-...`` and ``(6S)-...``), R before S
    , ``:22603``, under ``### **** CRITERIA RELATED ONLY TO CONFIGURATION``; see
    ``naming_utils.alpha_sort_key_total``). The order is a total one: a ``sorted`` that stops at the
    alphanumerical key keeps the order its input had, which for a set of names is the order of their hashes
    (the hash seed's) and for the groups of a prefix the order of the atoms of the input."""
    from .naming_utils import alpha_sort_key_total
    return alpha_sort_key_total(name)


def g_key(numbering, named):
    """ (g): the locants of the prefixes in the order they are cited (``citation_key``)."""
    names = sorted({nm for _, nm in named}, key=citation_key)
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


def senior_heteroatom_at_one(mol, hetero, first_atom, rank) -> int:
    """0 when the numbering of a heteromonocycle that starts at ``first_atom`` gives the locant 1 to a heteroatom of the
    most senior element of its ring (``rank`` maps an atom index to its rank in the seniority sequence of the 'a'
    prefixes: lower is senior), else 1; 0 for a ring with no heteroatom. (the Blue Book): "The locant
    '1' is given to a heteroatom that occurs first in the seniority sequence used for citation of the skeletal
    replacement ('a') prefixes. The numbering is then chosen to give lowest locants to heteroatoms considered as a set
    in ascending numerical order": the oxygen of '1,3,4-oxadiazol-2-yl' is 1, not the nitrogen of the lower set
    {1,2,4}."""
    return 0 if not hetero or (first_atom in hetero and rank(first_atom) == min(rank(a) for a in hetero)) else 1

