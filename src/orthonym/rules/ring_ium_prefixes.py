"""The '-ium' name of a protonated ring nitrogen whose neutral molecule takes a suffix ('4-carboxypiperidin-1-ium').

 (the Blue Book-18170) ranks '6 Cations' before '7 Acids' and every later class, so the ring
cation is the principal characteristic group: the parent is the ring hydride with the '-ium' suffix
:41366), and a group the neutral molecule names as a suffix is cited as a prefix ('4-carboxy-1-methylpyridin-1-ium
chloride (PIN)',:29914; an acyl group on a ring nitrogen is an 'acyl' prefix of the ring-ium parent,
:41447). The neutral name cannot be reused here: its suffix ('piperidine-4-carboxylic acid') or its other parent (the
pseudoketone '2-ethoxy-1-(4-ethylpiperazin-1-yl)ethan-1-one') does not hold the '-ium'. The name is built in one
numbering of the ring, as for a ring cation with no hydrogen (``ions._emit_ring_cumulative_suffix``, DEMOTE): the
cation's low locant, then the prefixes, every substituent of the ring a prefix, the bare ring the stem.
"""
import re


def ring_ium_with_prefixes(mol, ring_mol, ring_system, center_idx: int) -> str:
    """'4-(ethoxyacetyl)-1-ethylpiperazin-1-ium' for the protonated ring nitrogen ``center_idx`` of ``mol``, else ''.

    ``ring_mol`` is ``mol`` with that nitrogen neutral, index-preserving; it is numbered (a protonated ring atom
    read on ``mol`` looks like the indicated hydrogen of a parent). Declined for a stem with an indicated
    hydrogen, which this composition cannot place."""
    from .ions import _elide_terminal_e, _join_prefix_stem, _p74_bare_ring_stem, _p74_ring_substituent_prefix
    from .ring_cation_hydro import ring_cation_locants
    if mol.GetAtomWithIdx(center_idx).GetSymbol() != 'N' or not mol.GetAtomWithIdx(center_idx).GetTotalNumHs():
        return ''
    if _tautomer_competes(mol, ring_system, center_idx):
        return ''
    try:
        locants, indicated_h = ring_cation_locants(ring_mol, ring_system, center_idx)
    except (RuntimeError, ValueError, KeyError, IndexError):
        return ''
    if not locants or center_idx not in locants or indicated_h is not None:
        return ''
    prefix = _p74_ring_substituent_prefix(mol, ring_system, locants, set())
    if not prefix:
        return ''
    stem = _p74_bare_ring_stem(mol, ring_system, center_idx)
    if not stem or re.match(r'^\d+[Hh]-', stem):
        return ''
    return f"{_join_prefix_stem(prefix, _elide_terminal_e(stem))}-{locants[center_idx]}-ium"


def _tautomer_competes(mol, ring_system, center_idx: int) -> bool:
    """True when an exocyclic N, O or S that carries hydrogen stands on an sp2 atom of the ring ('2-amino-4,5-dihydro-
    1,3-thiazol-3-ium', '4-aminopyridin-1-ium', '3-hydroxypyridin-1-ium'), or the cationic nitrogen carries an sp2 or
    aryl carbon outside the ring (a protonated enamine: '1-(3-iminocyclohex-1-en-1-yl)piperidin-1-ium' is also the
    iminium of its resonance form). The hydron can then be read on another atom ('1,3-thiazolidin-2-iminium'), and
    the Blue Book sets no skeletal 'ium' against such a cationic suffix (d), the Blue Book; decision
    'internal notes): no PIN is claimed, the molecule is left to the other producers."""
    from rdkit import Chem
    ring = set(ring_system)
    for nb in mol.GetAtomWithIdx(center_idx).GetNeighbors():
        if nb.GetIdx() not in ring and (nb.GetIsAromatic() or any(
                b.GetBondType() != Chem.BondType.SINGLE for b in nb.GetBonds())):
            return True
    for i in ring:
        atom = mol.GetAtomWithIdx(i)
        if not (atom.GetIsAromatic() or any(b.GetBondType() == Chem.BondType.DOUBLE for b in atom.GetBonds())):
            continue
        for nb in atom.GetNeighbors():
            if nb.GetIdx() not in ring and nb.GetSymbol() in ('N', 'O', 'S') and nb.GetTotalNumHs():
                return True
    return False


def ring_acyl_token(mol, root: int, ring_atom: int, sub_atoms):
    """The acyl prefix of a carbonyl-rooted substituent ``sub_atoms`` of a ring cation ('acetyl', 'ethoxyacetyl',
    'benzoyl'), else None. (the Blue Book): the acyl prefix, not the '2-ethoxy-1-oxoethyl' of
    general nomenclature:30430). The numbering tie key and the name read the same token
    (``ions._p74_ring_substituent_groups``)."""
    from rdkit import Chem
    a = mol.GetAtomWithIdx(root)
    if a.GetSymbol() != 'C' or not any(
            b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(a).GetSymbol() == 'O' for b in a.GetBonds()):
        return None
    from ..assembly.substituent_naming import acyl_prefix_from_branch
    return acyl_prefix_from_branch(mol, root, ring_atom, set(sub_atoms))
