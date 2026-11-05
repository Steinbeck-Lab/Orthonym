"""The prefixes on the chain carbons of a chain diamide whose amide nitrogens are substituted.

'2-chloro-N1,N4-dimethylbutanediamide', 'N1,N4,2-trimethylbutanediamide': ``amides.name_chain_diamide`` names the
N-substituents of a saturated chain diamide; a group on a chain carbon was left to the general path, which declines
the combination. (the Blue Book): the N-substituents are cited with N-locants; (:2877): the
terminal locants of a diamide are not cited ('2-methylpropanediamide (PIN)',:2887);: identical prefixes are
one multiplied prefix with all their locants, the N-locants first ('N,2-dimethylpropanamide',:21624).
"""
from typing import List, Optional, Tuple

from rdkit import Chem


def chain_carbon_prefixes(mol: Chem.Mol, chain, covered: set, atom_to_locant: dict, *,
                          cite_stereo: bool = False) -> Optional[List[Tuple[int, str]]]:
    """``[(chain locant, prefix name),...]`` for every branch on a chain carbon outside ``covered``, or None when a
    branch has no name (or is bonded by more than a single bond). A structure with a stereo layer is left to the
    general path unless the caller cites the descriptors of the chain itself (``cite_stereo``:
    ``amides.name_chain_diamide`` when the amide nitrogens are substituted, a class the general path does not
    name)."""
    from ..assembly.substituent_enumerator import name_substituent
    chain_set, out, seen = set(chain), [], set(covered)
    for idx in chain:
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            j = nb.GetIdx()
            if j in seen or nb.GetAtomicNum() == 1:
                continue
            if mol.GetBondBetweenAtoms(idx, j).GetBondType() != Chem.BondType.SINGLE:
                return None
            branch, stack = set(), [j]
            while stack:
                a = stack.pop()
                if a in branch or a in chain_set:
                    continue
                branch.add(a)
                stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors())
            name = name_substituent(mol, branch, j)
            if not name or name == "substituent":
                return None
            seen |= branch
            out.append((atom_to_locant[idx], str(name)))
    if any(a.GetAtomicNum() > 1 and a.GetIdx() not in seen for a in mol.GetAtoms()):
        return None
    if out and not cite_stereo and (any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms())
                                    or any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds())):
        return None     # the descriptors are cited by the general path ('(2S,3S)-2,3-dihydroxybutanediamide')
    return out


def merged_prefix_parts(n_groups: dict, c_prefixes: List[Tuple[int, str]], render_n, sort_key) -> List[str]:
    """The cited prefixes, alphanumerical: an N-prefix group ``{name: [N locants]}`` and the chain prefixes; a simple
    name on both is one prefix with the N-locants first ('N1,N4,2-trimethyl')."""
    from ..assembly.naming_utils import format_substituent_prefix, multiplied_component
    from ..data.locant_text import render_locants
    c_groups: dict = {}
    for loc, name in c_prefixes:
        c_groups.setdefault(name, []).append(loc)
    parts = []
    for name, locs in n_groups.items():
        if name in c_groups and name.isalpha():
            all_locs = [f"N{loc}" for loc in sorted(locs)] + [str(x) for x in sorted(c_groups.pop(name))]
            parts.append((sort_key(name), f"{render_locants(all_locs)}-{multiplied_component(len(all_locs), name, name)}"))
        else:
            parts.append((sort_key(name), render_n(name, sorted(locs))))
    for name, locs in c_groups.items():
        parts.append((sort_key(name), format_substituent_prefix(name, sorted(locs), len(locs))))
    parts.sort(key=lambda p: p[0])
    return [p[1] for p in parts]
