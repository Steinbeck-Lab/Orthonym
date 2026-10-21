"""'-ylidene' prefix of an UNSATURATED monocyclic ring fragment: 'cyclohex-2-en-1-ylidene',
'cyclohexa-2,4-dien-1-ylidene', '2H-pyrrol-2-ylidene', '4H-thiopyran-4-ylidene'.

 gives a substituent group joined to its parent by a double bond the ending 'ylidene' in
place of 'yl', on the same parent hydride, the Blue Book, "All mancude rings and
ring systems are named by method (2) in "). The saturated carbocycle ('cyclohexylidene') is
built by ``substituent_enumerator._carbon_ylidene_prefix``; the unsaturated ring had no producer,
so the gate of ``carbon_free_valence_prefix`` refused it ('-yl' would describe a different molecule)
and a molecule whose senior ring carries such a prefix had no default-tier name.

* Carbocycle: the free valence takes locant 1 (the lowest,, the ring is numbered for the
  lowest 'ene' locants: the token of the '-yl' producer
  (``ring_substituents._unsaturated_carbocyclic_substituent``) with its ending exchanged.
* One-heteroatom ring: the ylidene atom is the INDICATED-HYDROGEN position of the mancude parent,
  '4H-thiopyran-4-ylidene (PIN)', the Blue Book); the numbering, the 'hydro'
  prefixes and the indicated hydrogen are those of the '-yl' producer
  (``ring_substituents._unsaturated_heteromonocyclic_substituent(..., ylidene=True)``). A ylidene
  atom that is a hydro position instead needs the added-hydrogen form,
  'naphthalen-1(2H)-ylidene'), which is not built: the producer declines.

Only a bare ring is named here (the fragment IS the ring: a decorated ring is named by the decorated
ring producers), and only a plain neutral ring with no isotope, no CIP label and no stereo double
bond. ``carbon_free_valence_prefix`` re-reads the token's own text (two free valences) before it is
used, so a token that spelled one would never ship.
"""
from typing import Optional

from rdkit import Chem


def unsaturated_monocycle_ylidene_prefix(mol, frag_atoms, attach_idx, free_valence) -> Optional[str]:
    """The ylidene prefix of the unsaturated monocycle ``frag_atoms``, or None (decline)."""
    if free_valence != 2:
        return None
    frag = set(frag_atoms)
    ring_info = mol.GetRingInfo()
    rings = [r for r in ring_info.AtomRings() if attach_idx in r]
    if len(rings) != 1 or set(rings[0]) != frag or attach_idx not in frag:
        return None
    n_hetero = 0
    for idx in frag:
        atom = mol.GetAtomWithIdx(idx)
        if (atom.GetIsAromatic() or atom.GetFormalCharge() or atom.GetNumRadicalElectrons()
                or atom.GetIsotope() or atom.HasProp('_CIPCode')
                or ring_info.NumAtomRings(idx) != 1):
            return None
        n_hetero += atom.GetAtomicNum() != 6
    has_double = False
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in frag and bond.GetEndAtomIdx() in frag:
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                if bond.GetStereo() != Chem.BondStereo.STEREONONE or bond.HasProp('_CIPCode'):
                    return None
                has_double = True
            elif bond.GetBondType() != Chem.BondType.SINGLE:
                return None
    if not has_double:
        return None  # saturated: the saturated-ring producer owns it
    ring = tuple(sorted(frag))
    from .ring_substituents import (
        _unsaturated_carbocyclic_substituent,
        _unsaturated_heteromonocyclic_substituent,
    )
    if n_hetero == 0:
        # the ylidene atom carries no ring double bond (it has two exocyclic valences)
        if any(mol.GetBondBetweenAtoms(attach_idx, n.GetIdx()).GetBondTypeAsDouble() != 1.0
               for n in mol.GetAtomWithIdx(attach_idx).GetNeighbors() if n.GetIdx() in frag):
            return None
        base = _unsaturated_carbocyclic_substituent(mol, ring, attach_idx)
        if not base or not base.endswith('-1-yl'):
            return None
        return base[:-len('yl')] + 'ylidene'
    return _unsaturated_heteromonocyclic_substituent(mol, ring, attach_idx, ylidene=True)
