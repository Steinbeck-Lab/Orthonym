"""The substituents of a fused heterocyclic core cited as a substituent prefix
(``composer._generate_ring_substituent_prefixes``): the prefix string of the groups this module names, or None when
a neighbour of the core is a group it does not name, so that the core is never cited bare while it carries one.

Kept out of ``composer.py`` so that file stays within its size budget."""
from collections import defaultdict
from typing import Dict, List

from ..data.locant_text import render_locants
from .naming_utils import alpha_sort_key
from .straight_alkyl import straight_alkyl_name


def _collect_pure_alkyl_atoms(mol, start_idx, excluded):
    from .composer import _collect_pure_alkyl_atoms as collect
    return collect(mol, start_idx, excluded)


def detect_fused_het_inner_subs(
    mol, core_atom_set, atom_mapping, ring_atom_set, chain_set,
    attach_ring_idx
) -> str:
    """Detect substituents on a fused heterocycle core and return formatted prefix string.

    Finds atoms bonded to core atoms that are NOT part of the core and NOT in
    the chain. Maps their positions to IUPAC locants via atom_mapping.

    Returns:
        Formatted inner substituent prefix string (e.g., "5-methyl-") or ""
        if no inner substituents found; None when a neighbour of the core is a
        substituent this detector does not name (a phenyl, a branched alkyl, a
        sulfanyl group, an atom of a ring joined to the core): the caller then
        leaves the core to the general ring namer instead of citing it bare
        ('1-benzofuran-5-yl' for a 2-phenylbenzofuran-5-yl lost the phenyl).
    """
    _HALOGEN_PREFIX = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}
    _ALKOXY_DIRECT = {1: 'methoxy', 2: 'ethoxy', 3: 'propoxy'}

    from ..assembly.naming_utils import get_multiplier_prefix as _gmp

    sub_groups: Dict[str, List] = defaultdict(list)

    for core_idx in core_atom_set:
        # the attachment atom too: its bond to the chain is skipped below (``chain_set``), any other group on it
        # is a substituent of the core ('2-methyl-...-2-yl' of a 2-methylchroman-2-yl)
        iupac_locant = atom_mapping.get(core_idx)
        if iupac_locant is None:
            continue
        atom = mol.GetAtomWithIdx(core_idx)
        for nbr in atom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni in core_atom_set or ni in chain_set:
                continue
            if ni in ring_atom_set:
                return None     # a ring joined to the core: not named here
            named_before = sum(len(v) for v in sub_groups.values())
            sym = nbr.GetSymbol()
            if sym in _HALOGEN_PREFIX:
                sub_groups[_HALOGEN_PREFIX[sym]].append(iupac_locant)
            elif sym == 'C' and not nbr.GetIsAromatic():
                # Non-core carbon neighbor -- classify by substructure

                cn_nbrs = [n for n in nbr.GetNeighbors()
                           if n.GetIdx() != core_idx
                           and n.GetIdx() not in core_atom_set]

                # (a) Cyano detection: C with triple bond to N, no H on C
                if (len(cn_nbrs) == 1 and cn_nbrs[0].GetSymbol() == 'N'
                        and nbr.GetTotalNumHs() == 0):
                    bond_cn = mol.GetBondBetweenAtoms(ni, cn_nbrs[0].GetIdx())
                    if bond_cn and bond_cn.GetBondTypeAsDouble() == 3.0:
                        sub_groups['cyano'].append(iupac_locant)
                        continue

                # (b) Haloalkyl detection: C bonded only to halogens (no H, no non-hal)
                hal_map = _HALOGEN_PREFIX
                hal_nbrs = [n for n in cn_nbrs if n.GetSymbol() in hal_map]
                non_hal = [n for n in cn_nbrs
                           if n.GetSymbol() not in hal_map
                           and n.GetSymbol() != 'H']
                if hal_nbrs and not non_hal and nbr.GetTotalNumHs() == 0:
                    hal_counts: Dict[str, int] = defaultdict(int)
                    for h in hal_nbrs:
                        hal_counts[h.GetSymbol()] += 1
                    parts = []
                    for h_sym in sorted(hal_counts.keys(),
                                        key=lambda s: hal_map[s]):
                        cnt = hal_counts[h_sym]
                        mp = _gmp(cnt, hal_map[h_sym]) if cnt > 1 else ''
                        parts.append(f'{mp}{hal_map[h_sym]}')
                    haloalkyl_name = '(' + ''.join(parts) + 'methyl)'
                    sub_groups[haloalkyl_name].append(iupac_locant)
                    continue

                # (c) An unbranched alkyl chain bound by its end: the count names that shape only
                # (assembly/straight_alkyl; (1), the Blue Book)
                alkyl_atoms = _collect_pure_alkyl_atoms(
                    mol, ni, core_atom_set | chain_set | ring_atom_set
                )
                alkyl_name = straight_alkyl_name(mol, alkyl_atoms) if alkyl_atoms else None
                if alkyl_name:
                    sub_groups[alkyl_name].append(iupac_locant)

            elif sym == 'O':
                h_count = nbr.GetTotalNumHs()
                o_nbrs = [n for n in nbr.GetNeighbors()
                          if n.GetIdx() != core_idx
                          and n.GetIdx() not in core_atom_set]
                if h_count == 1 and len(o_nbrs) == 0:
                    sub_groups['hydroxy'].append(iupac_locant)
                elif (h_count == 0 and len(o_nbrs) == 1
                      and o_nbrs[0].GetSymbol() == 'C'):
                    # Alkoxy of an unbranched chain bound by its end (assembly/straight_alkyl)
                    c_start_idx = o_nbrs[0].GetIdx()
                    alkyl_atoms = _collect_pure_alkyl_atoms(
                        mol, c_start_idx,
                        core_atom_set | chain_set | ring_atom_set | {ni}
                    )
                    c_count = (len(alkyl_atoms) if alkyl_atoms
                               and straight_alkyl_name(mol, alkyl_atoms) else None)
                    if c_count:
                        if c_count in _ALKOXY_DIRECT:
                            sub_groups[_ALKOXY_DIRECT[c_count]].append(
                                iupac_locant)
                        else:
                            from ..data.chain_names import get_chain_prefix
                            sub_groups[
                                f'{get_chain_prefix(c_count)}oxy'
                            ].append(iupac_locant)

            elif sym == 'N':
                # Nitrogen neighbor -- check nitro vs amino
                n_nbrs = [n for n in nbr.GetNeighbors()
                          if n.GetIdx() != core_idx
                          and n.GetIdx() not in core_atom_set]
                if (nbr.GetFormalCharge() == 1
                        and len(n_nbrs) == 2
                        and all(n.GetSymbol() == 'O' for n in n_nbrs)):
                    sub_groups['nitro'].append(iupac_locant)
                elif nbr.GetTotalNumHs() == 2 and len(n_nbrs) == 0:
                    sub_groups['amino'].append(iupac_locant)
            if sum(len(v) for v in sub_groups.values()) == named_before:
                return None     # a substituent no branch above names: never cite the core bare

    if not sub_groups:
        return ""

    # Build prefix string sorted by IUPAC alphabetization
    from ..assembly.naming_utils import enclose_if_compound
    prefix_parts = []
    for name in sorted(sub_groups.keys(), key=alpha_sort_key):
        locs = sorted(sub_groups[name], key=lambda x: (int(x) if str(x).isdigit() else 999, str(x)))
        count = len(locs)
        loc_str = render_locants(locs)
        if count == 1:
            prefix_parts.append(f'{loc_str}-{enclose_if_compound(name)}')
        else:
            from .naming_utils import multiplied_component as _mc
            prefix_parts.append(f'{loc_str}-{_mc(count, name, enclose_if_compound(name))}')

    return '-'.join(prefix_parts) + '-' if prefix_parts else ""
