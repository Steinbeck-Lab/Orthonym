"""The numbering of a bare fused ring that an ionic suffix is added to, read from the ring's own name.

``ions._ring_iupac_locants`` asks the namer's ring-locant supplier (``namer._build_ring_info_for_parent_selection``)
for the numbering of the neutral ring. That supplier numbers a fused ring system that is not in its catalogue with the
orienter of a single ring (its Branch 4 is reached for any one ring system with a heteroatom), so a ring cation whose
fused parent the catalogue misses got a locant from another numbering than its name: 'C[n+]1cccn2ncnc21' was composed
as '6-methyl[1,2,4]triazolo[1,5-a]pyrimidin-6-ium' (C-6 is a carbon), the round trip refused it and the PIN tier
declined. The ring's name fixes its numbering "Numbering fused ring systems", the Blue Book: the
fused name is numbered by those rules, and the locants of the name are that numbering), and OPSIN's reading of that
name gives each atom its locant (``validation.opsin_roundtrip.opsin_atom_locant_map``, the re-anchor primitive): '4-methyl[1,2,4]triazolo[1,5-a]
pyrimidin-4-ium', the Blue Book, the '-ium' at the position of the added hydron, here the substituted
nitrogen).

Scope: a bare ring (every heavy atom in one ring system, as the DEMOTE branch of ``ions`` builds it) of two or more
rings, fused or bridged ('7,7-dimethyl-7-azabicyclo[2.2.1]heptan-7-ium'). A monocycle, and a ring with substituents, keep the supplier. None when the ring has no name or OPSIN cannot
map the name onto it; the caller then uses the supplier as before.
"""
from typing import Dict, Optional

from rdkit import Chem


def bare_fused_ring_locants(ring_mol, mol=None, ring_system=()) -> Optional[Dict[int, object]]:
    """``{atom index: locant}`` of ``ring_mol`` in the numbering of its own name, or None (see the module docstring).

    ``mol`` and ``ring_system``: the ion and its ring atoms before the substituents were cut off. An aromatic ring atom
    of ``mol`` with an exocyclic double bond ('C[n+]1cn(R)c2nc(N)[nH]c(=O)c21', the 6-oxo of 7-methylguanosine) gives
    None: cut off, its '=O' leaves a mancude bare ring, whose name states none of the hydro prefixes the oxo needs
     "Indicated hydrogen, 'added indicated hydrogen', and hydro prefixes", the Blue Book;
    '2,2-dimethyl-1,3-dioxo-2,3-dihydro-1H-isoindol-2-ium (PIN)',:41447), so the cation
    would be spelled '...-6-oxo-1H-purin-7-ium' without them and labelled a PIN it is not."""
    if ring_mol is None or ring_mol.GetRingInfo().NumRings() < 2:
        return None
    if mol is not None and any(
            mol.GetAtomWithIdx(i).GetIsAromatic() and any(
                b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtomIdx(i) not in set(ring_system)
                for b in mol.GetAtomWithIdx(i).GetBonds())
            for i in ring_system):
        return None
    heavy = [a for a in ring_mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy or not all(a.IsInRing() for a in heavy) or len(Chem.GetMolFrags(ring_mol)) != 1:
        return None
    try:
        from ..errors import is_failure_name
        from ..namer import Orthonym
        from ..validation.opsin_roundtrip import opsin_atom_locant_map
        name = Orthonym(style='pin', _disable_opsin_validity_gate=True).name(Chem.MolToSmiles(ring_mol))
        if not name or is_failure_name(name):
            return None
        locants = opsin_atom_locant_map(name, ring_mol)
    except (RecursionError, ValueError, RuntimeError, KeyError):
        return None
    if not locants or set(locants) != {a.GetIdx() for a in heavy}:
        return None
    return locants
