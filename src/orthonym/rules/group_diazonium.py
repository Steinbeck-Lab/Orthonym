"""The diazonium cation on a group the parent hydride does not carry: '[(4-methoxyphenyl)methylidene]azanediazonium',
'(hexyloxy)diazonium', '(4-oxo-2-sulfanylidene-1,3-thiazolidin-3-yl)diazonium'.

 "Diazonium ions" (the Blue Book): "Cations containing an -N2+ group attached to a parent hydride are
traditionally named according to the principles of substitutive nomenclature by using the suffix 'diazonium'"
(:41609), 'methanediazonium (PIN)' (:41613). A -N2+ group on an oxygen, a sulfur or a ring heteroatom is on the parent
hydride of the group that holds it (oxidane, sulfane, the heteroring), which the substitutive path does not build for a
group that is no hydrocarbon: the group is named as a prefix, as for the diazonium ion on a nitrogen
(``azanediazonium``), and 'diazonium' ends the name. A group on a carbon is the parent hydride's own
('naphthalene-1-diazonium', ``charged_router._name_diazonium``) and is left to it. The Blue Book gives no example, so
the name is recorded as not the PIN; it is read back to the full InChIKey of the molecule before it is returned.
"""
from rdkit import Chem


def _ion_sites_are_only(mol, cation_idx: int) -> bool:
    """True when the one genuine ionic centre of ``mol`` is the cation at ``cation_idx`` (the charges of a nitro or
    N-oxide group are internal and not ionic centres, ``perception.ions``)."""
    from ..perception.ions import get_ion_sites
    sites = get_ion_sites(mol)
    cations = sites.get('cations') or []
    return not sites.get('anions') and len(cations) == 1 and cations[0]['atom_idx'] == cation_idx


def name_group_diazonium(mol, cation_idx: int, proximal_n: int, terminal_n: int, attach_idx: int) -> str:
    """'<group>diazonium' for the -N2+ group (``proximal_n`` bonded to ``attach_idx``, ``terminal_n`` the other
    nitrogen, ``cation_idx`` the one that carries the charge in the drawing) on the group that ``attach_idx`` heads,
    else ''. The group is named as a prefix by the namers of ``group_prefixes`` (the first that reads back wins)."""
    from ..assembly.naming_utils import enclose_if_compound
    from .group_prefixes import first_name_that_reads_back, group_prefix_candidates
    if (not _ion_sites_are_only(mol, cation_idx) or len(Chem.GetMolFrags(mol)) != 1
            or mol.GetAtomWithIdx(attach_idx).GetSymbol() == 'C'):   # a carbon group is the parent hydride's
        return ''
    side, stack = {attach_idx}, [attach_idx]
    while stack:
        for n in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
            if n.GetIdx() not in side and n.GetIdx() not in (proximal_n, terminal_n):
                side.add(n.GetIdx())
                stack.append(n.GetIdx())
    if {proximal_n, terminal_n} & side or proximal_n not in (a.GetIdx() for a in mol.GetAtomWithIdx(attach_idx).GetNeighbors()):
        return ''
    prefixes = group_prefix_candidates(mol, sorted(side), attach_idx, 'yl', [proximal_n])
    return first_name_that_reads_back(mol, (f"{enclose_if_compound(p)}diazonium" for p in prefixes))
