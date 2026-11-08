"""A carbon substituent of a polycyclic arene parent that the arene writer's own table does not name: the shared prefix.

``rules.polycyclics._identify_pah_substituent`` names alkyl chains, a few functionalised chains ('hydroxymethyl') and the
suffix groups on the ring; any other carbon group left the whole molecule unnamed at the PIN tier ('3-(methoxycarbonyl)
naphthalene-2-carboxylic acid', where the benzene writer names '2-(methoxycarbonyl)benzoic acid'). The prefix of such a
group is the one the shared substituent namer builds for every ring parent (``assembly.substituent_naming.
name_substituent_fragment``): 'methoxycarbonyl', '(acetylsulfanyl)carbonyl', '3-methoxy-3-oxopropyl'; an ester is a
prefix when an acid holds the suffix:31709, 'alkoxy...oxo' / 'alkoxycarbonyl'), and names the
thioanhydride link of 'acetic 3-[(acetylsulfanyl)carbonyl]naphthalene-2-carboxylic anhydride (PIN)' (:32459) the same way.

The prefix is declined when the group holds an atom of the molecule's principal characteristic group:18169): that
group is then a suffix of another parent, or a second suffix of this one, which this writer does not build.
"""
from typing import Dict, Optional, Sequence, Set


def shared_pah_prefix(mol, start_idx: int, core_atoms: Set[int], frag_atoms: Sequence[int]) -> Optional[Dict]:
    """``{'name', 'atoms'}`` for the carbon group ``frag_atoms`` (attached at ``start_idx``) of an arene parent, else None."""
    frag = set(frag_atoms)
    from rdkit import Chem
    if not _shape_has_book_prefix(mol, start_idx, frag):
        return None
    from ..perception.functional_groups import detect_functional_groups
    from .seniority import get_principal_group
    probe = Chem.Mol(mol)
    try:
        probe.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(probe)       # the caller's molecule may carry no ring information
        _pg, matches = get_principal_group(probe, detect_functional_groups(probe))
    except Exception:
        return None
    if any(frag & set(m) for m in matches or ()):
        return None
    from ..assembly.substituent_naming import name_substituent_fragment
    try:
        name = name_substituent_fragment(mol, sorted(frag), start_idx, sorted(core_atoms))
    except Exception:
        return None
    if not name or name == 'substituent' or ' ' in name or 'unknown' in name:
        return None
    if _acyl_head(mol, start_idx, frag):
        from ..assembly.acyl_prefix import is_acyl_prefix
        if not is_acyl_prefix(name):
            return None     # '3-methoxy-1,3-dioxopropyl' is the general form of '3-methoxy-3-oxopropanoyl':30430)
    return {'name': name, 'atoms': sorted(frag)}


def _acyl_head(mol, start_idx: int, frag) -> bool:
    """True when the attachment carbon is a carbonyl carbon that continues into another carbon of the group (an acyl)."""
    from rdkit import Chem
    head = mol.GetAtomWithIdx(start_idx)
    oxo = any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(head).GetSymbol() == 'O' for b in head.GetBonds())
    return oxo and any(nb.GetSymbol() == 'C' and nb.GetIdx() in frag for nb in head.GetNeighbors())


def _shape_has_book_prefix(mol, start_idx: int, frag) -> bool:
    """False for the shapes whose book prefix the shared namer does not spell: a hydrazide ('hydrazinecarbonyl',
     :33990, not 'aminocarbamoyl') or any N-N bond, and a C=N group ('C-methoxycarbonimidoyl', not
    '[imino(methoxy)methyl]')."""
    from rdkit import Chem
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i not in frag or j not in frag:
            continue
        pair = {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()}
        if pair == {'N'} or (pair == {'C', 'N'} and b.GetBondType() != Chem.BondType.SINGLE):
            return False
    return True
