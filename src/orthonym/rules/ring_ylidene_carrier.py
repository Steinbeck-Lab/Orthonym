"""A ring-YLIDENE substituent on a chain carrier: '(cyclohexylidenemethyl)', '1-cyclohexylideneethyl'.

 "Unsymmetrical substituent groups having terminal atoms with different types of
free valencies" (the Blue Book) names

    (cyclohexylidenemethyl)benzene (PIN)
    [not (phenylmethylidene)cyclohexane; benzene is senior to cyclohexane (see ] (:6390)

The substituent group of the parent ring is the carrier -CH= (a chain, 'methyl'); the ring that is
double-bonded to it is cited as the prefix 'cyclohexylidene' on the carrier, and a longer carrier
numbers its free valence 1, '1-cyclohexylideneethyl', '2-cyclohexylideneethyl'.

``rules.ring_substituents._compound_ring_on_chain_substituent`` builds '(ring-yl)alkyl' for a carrier
whose bonds are all single; this module is the same shape with ONE double bond, the one between a
carrier carbon and a ring carbon. ``ring_on_chain_substituent_name`` is the entry point of both: the
ring-ylidene form first, every other shape by the single-bond producer, unchanged.

Fails closed (``None``, so the caller falls back) unless the whole fragment is one carbon carrier path
rooted at the attachment and one ring system, joined by one plain C=C bond without a stereo flag, with
the ring producer's own token spelling the two free valences.
"""
from typing import Dict, Optional

from rdkit import Chem


def ring_on_chain_substituent_name(
    mol, frag_atoms, frag_set, frag_ring_atoms, attach_idx, ring_info,
    allow_mancude: bool = False,
    pos_out: Optional[Dict[int, int]] = None,
) -> Optional[str]:
    """The ring-on-chain substituent name of a chain-rooted fragment carrying a ring system, or None."""
    from .ring_substituents import _compound_ring_on_chain_substituent
    name = ring_ylidene_on_chain_substituent(
        mol, frag_set, frag_ring_atoms, attach_idx,
        allow_mancude=allow_mancude, pos_out=pos_out)
    if name is not None:
        return name
    return _compound_ring_on_chain_substituent(
        mol, frag_atoms, frag_set, frag_ring_atoms, attach_idx, ring_info,
        allow_mancude=allow_mancude, pos_out=pos_out)


def _ring_ylidene_bonds(mol, frag_set, ring_atoms):
    """The plain C=C bonds of the fragment between a ring carbon and a non-ring carbon."""
    found = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i not in frag_set or j not in frag_set or b.GetBondType() != Chem.BondType.DOUBLE:
            continue
        if (i in ring_atoms) == (j in ring_atoms):
            continue
        ring_c, chain_c = (mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j))
        if i not in ring_atoms:
            ring_c, chain_c = chain_c, ring_c
        if (ring_c.GetSymbol() != 'C' or chain_c.GetSymbol() != 'C' or ring_c.GetIsAromatic()
                or chain_c.GetIsAromatic() or chain_c.IsInRing() or b.GetIsAromatic()
                or b.GetStereo() != Chem.BondStereo.STEREONONE):
            continue
        found.append((chain_c.GetIdx(), ring_c.GetIdx()))
    return found


def _carrier_path(mol, carrier, attach_idx):
    """The carrier as a simple path of single-bonded carbons rooted at ``attach_idx``, or None."""
    adj = {a: [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() in carrier]
           for a in carrier}
    if len(adj.get(attach_idx, [])) > 1:
        return None
    path = [attach_idx]
    while True:
        nxt = [n for n in adj[path[-1]] if n not in path]
        if not nxt:
            break
        if len(nxt) > 1:
            return None
        path.append(nxt[0])
    if len(path) != len(carrier):
        return None
    for a, b in zip(path, path[1:]):
        if mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() != 1.0:
            return None
    return path


def ring_ylidene_on_chain_substituent(
    mol, frag_set, frag_ring_atoms, attach_idx,
    allow_mancude: bool = False,
    pos_out: Optional[Dict[int, int]] = None,
) -> Optional[str]:
    """'(ring-ylidene)alkyl' for a carbon carrier rooted at ``attach_idx`` with one ring system on
    it by a double bond, or None. ``pos_out`` is filled as in ``_compound_ring_on_chain_substituent``
    (the carrier numbering, free valence 1) when the name cites a carrier locant."""
    if attach_idx not in frag_set or attach_idx in frag_ring_atoms:
        return None
    ylidene = _ring_ylidene_bonds(mol, frag_set, set(frag_ring_atoms))
    if len(ylidene) != 1:
        return None
    from .ring_substituents import (
        _carries_stereo,
        _fold_nonring_decorations,
        name_ring_system_substituent,
    )
    # A ring decoration is part of the ring-ylidene prefix ('4-methylcyclohexylidene'); the
    # carrier is what is left of the fragment.
    ring_atoms = _fold_nonring_decorations(mol, frag_set, frag_ring_atoms, attach_idx)
    if ring_atoms is None:
        return None
    if not allow_mancude and _carries_stereo(mol, ring_atoms - set(frag_ring_atoms)):
        return None  # the strict ring producers do not cite a stereodescriptor inside a decoration
    carrier = set(frag_set) - ring_atoms
    if attach_idx not in carrier or any(
            mol.GetAtomWithIdx(a).GetSymbol() != 'C' or mol.GetAtomWithIdx(a).GetFormalCharge()
            or mol.GetAtomWithIdx(a).GetNumRadicalElectrons() for a in carrier):
        return None
    # The attachment bond is a single bond: a carbon double-bonded to the ring and to the parent
    # is a cumulene carbon, which this prefix does not describe.
    if any(n.GetIdx() not in frag_set
           and mol.GetBondBetweenAtoms(attach_idx, n.GetIdx()).GetBondTypeAsDouble() != 1.0
           for n in mol.GetAtomWithIdx(attach_idx).GetNeighbors()):
        return None
    path = _carrier_path(mol, carrier, attach_idx)
    chain_c, ring_c = ylidene[0]
    if path is None or chain_c not in path or ring_c not in ring_atoms:
        return None
    # The ring system hangs off ONE carrier atom, and only through the ylidene bond.
    ring_bonds = [(a, n.GetIdx()) for a in path for n in mol.GetAtomWithIdx(a).GetNeighbors()
                  if n.GetIdx() in ring_atoms]
    if ring_bonds != [(chain_c, ring_c)]:
        return None
    ring_name = name_ring_system_substituent(
        mol, sorted(ring_atoms), ring_c, allow_enumerator_fallback=True, allow_mancude=allow_mancude)
    if not ring_name:
        return None
    # The ring producer must have SPELLED the double bond: a '-yl' token for a ring joined by a
    # double bond would describe a different molecule.
    from ..validation.name_morphemes import free_valence_morphology
    fv = free_valence_morphology(ring_name)
    if not (fv.confident and fv.free_valences == 2):
        return None
    from ..assembly.naming_utils import (
        apply_enclosing_marks,
        get_alkyl_name,
        is_complex_substituent,
    )
    from ..assembly.prefix_derivation import built
    alkyl = get_alkyl_name(len(path))
    if not alkyl:
        return None
    #: the ring-ylidene prefix is enclosed only when it is itself complex.
    inner = apply_enclosing_marks(ring_name, -1) if is_complex_substituent(ring_name) else ring_name
    if len(path) == 1:
        return built(f'{inner}methyl', substituted=True)
    if pos_out is not None:
        pos_out.update({a: i for i, a in enumerate(path, start=1)})
    return built(f'{path.index(chain_c) + 1}-{inner}{alkyl}', substituted=True)
