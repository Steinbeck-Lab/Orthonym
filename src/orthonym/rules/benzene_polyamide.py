"""Benzene di- and polyamides whose nitrogens carry groups: each group cited with the N-locant of its own nitrogen.

 'N-Substitution', (the Blue Book): "In di- and polyamides, except for geminal
diamides, N locants with superscripted arabic numbers, which are the locants of the parent structure, are used to
differentiate the nitrogen atoms"; 'N1-methylbutanediamide (PIN)', 'N3-ethyl-N1-methylnaphthalene-1,3-disulfonamide
(PIN)'; 'benzene-1,4-dicarboxamide (PIN)' (:32718). Kept apart from ``rules.benzene`` (a module at its size budget),
whose benzene polyamine writer, the same superscript convention) builds the name with the amide suffix.
The prefix part of those names (``n_locant_prefix_part``) is also the one of the polycyclic polyamides
('N2,N6-dimethylnaphthalene-2,6-dicarboxamide', ``polycyclic_polyamide_prefixes``).
"""
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

#: the key of the per-locant N-group table of a suffix in ``n_substituents_map``
_BY_LOCANT = "__by_locant__:"
#: the multiplied amide suffixes this writer cites
_AMIDE_SUFFIXES = ("carboxamide", "sulfonamide")


def note_n_substituents(n_substituents_map: Dict, sub_info: Dict, locant: int) -> None:
    """Record the N-groups of a suffix group: under the suffix name (the one-group writers read it) and per locant."""
    n_subs = sub_info.get('n_substituents')
    if n_subs:
        n_substituents_map[sub_info['suffix_name']] = n_subs
        n_substituents_map.setdefault(_BY_LOCANT + sub_info['suffix_name'], {})[locant] = list(n_subs)


def name_benzene_polyamide(chosen_suffix: str, chosen_locants: List[int], prefix_groups: Dict[str, List[int]],
                           n_substituents_map: Dict) -> Optional[str]:
    """'N1-methylbenzene-1,4-dicarboxamide' for two or more ring amides of which at least one nitrogen carries a
    group; None for any other case (one amide: the 'benzamide' writer; no N-group: the plain multiplied suffix)."""
    if chosen_suffix not in _AMIDE_SUFFIXES or len(chosen_locants) < 2:
        return None
    by_locant = n_substituents_map.get(_BY_LOCANT + chosen_suffix) or {}
    if not by_locant:
        return None
    from .benzene import _name_substituted_benzenediamine
    return _name_substituted_benzenediamine(prefix_groups, list(chosen_locants), dict(by_locant), chosen_suffix)


def n_locant_prefix_part(ring_groups: Dict[str, List[int]], n_by_locant: Dict[int, List[str]]) -> str:
    """The prefix part of a poly-amine or polyamide name: every N group with the N-locant of its own nitrogen
    , the Blue Book;,:32774), one name over the ring and the nitrogens multiplied
    together, the italic N locants first,:7027;,:3195: 'N1,N4,2-trimethyl'), all prefixes in
    one alphanumerical series: '4-chloro-N1-methyl'. '' when there is no prefix."""
    from ..assembly.identical_prefixes import shared_prefix_groups
    from ..assembly.naming_utils import (alpha_sort_key, enclose_if_compound, format_substituent_prefix,
                                         multiplied_component, prefix_citation_sort_key, _wrap_n_substituent)
    from ..data.locant_text import render_locants
    n_locs_by_name: Dict[str, List[int]] = {}
    for n_loc in sorted(n_by_locant):
        for nm in n_by_locant[n_loc]:
            n_locs_by_name.setdefault(nm, []).append(n_loc)
    ring_groups, n_locs_by_name, shared_entries = shared_prefix_groups(
        ring_groups, n_locs_by_name, lambda loc: f"N{loc}")
    entries: List[Tuple[str, str]] = []
    for nm, locs in n_locs_by_name.items():
        locs = sorted(locs)
        body = _wrap_n_substituent(enclose_if_compound(nm))
        entries.append((alpha_sort_key(nm), f"{render_locants(f'N{loc}' for loc in locs)}-"
                                            f"{multiplied_component(len(locs), nm, body)}"))
    for name in ring_groups:
        locants = sorted(ring_groups[name])
        entries.append((alpha_sort_key(name), format_substituent_prefix(name, locants, len(locants))))
    entries += shared_entries
    # an entry is an already rendered prefix, its locants parent locants; ``[1:]`` keeps the total order
    entries.sort(key=lambda e: (e[0], prefix_citation_sort_key(e[1], parent_locants=True)[1:]))
    return "-".join(rendered for _k, rendered in entries)


def polycyclic_polyamide_prefixes(mol, substituents: Dict[int, List[Dict]], atom_to_locant: Dict[int, object],
                                  ring_groups: Dict[str, List[int]], fixed: bool) -> Optional[str]:
    """The prefix part of a polycyclic hydrocarbon with two or more ring carboxamides of which at least one nitrogen
    carries a group ('N2,N6-dimethyl' of 'N2,N6-dimethylnaphthalene-2,6-dicarboxamide'), or None for any other case.

    The numbering of the ring system is fixed up to its symmetry; of the symmetry images that keep every suffix locant
    and the locants of every ring prefix, the one chosen gives the substituted nitrogens the lowest locants
    after the ring locants: 'N4,2-dimethylpentane-2,4-diamine (PIN)', the Blue Book; 'N1-methylbutanediamide
    (PIN)',:32774), then the N group cited first the lowest (g)). ``fixed``: keep the given numbering (the
    name cites stereodescriptors with its locants)."""
    by_locant = {pos: list(info['n_substituents']) for pos, infos in substituents.items() for info in infos
                 if info.get('is_suffix') and info.get('name') == 'carboxamide' and info.get('n_substituents')}
    amide_locants = sorted(pos for pos, infos in substituents.items() for info in infos
                           if info.get('is_suffix') and info.get('name') == 'carboxamide')
    if len(amide_locants) < 2 or not by_locant:
        return None
    atom_of = {loc: a for a, loc in atom_to_locant.items()}
    if not all(isinstance(loc, int) and loc in atom_of for loc in amide_locants):
        return None
    images = [dict(atom_to_locant)] if fixed else _symmetry_images(mol, atom_to_locant)

    def moved(image, loc):
        return image[atom_of[loc]]

    from ..assembly.naming_utils import alpha_sort_key
    best = None
    for image in images:
        if sorted(moved(image, loc) for loc in amide_locants) != amide_locants:
            continue
        if any(sorted(moved(image, loc) for loc in locs) != sorted(locs) for locs in ring_groups.values()):
            continue
        new_map = {moved(image, loc): names for loc, names in by_locant.items()}
        all_n = sorted(loc for loc, names in new_map.items() for _ in names)
        first_cited = tuple(tuple(sorted(loc for loc, names in new_map.items() if nm in names))
                            for nm in sorted({n for names in new_map.values() for n in names}, key=alpha_sort_key))
        key = (all_n, first_cited)
        if best is None or key < best[0]:
            best = (key, new_map)
    if best is None:
        return None
    return n_locant_prefix_part({k: list(v) for k, v in ring_groups.items()}, best[1])


def _symmetry_images(mol, atom_to_locant: Dict[int, object]) -> List[Dict[int, object]]:
    """Every numbering of the ring system that a symmetry of its atoms gives: ``{atom: locant[image of atom]}``."""
    core = sorted(atom_to_locant)
    query = Chem.MolFromSmarts(Chem.MolFragmentToSmarts(mol, atomsToUse=core))
    if query is None:
        return [dict(atom_to_locant)]
    matches = [m for m in mol.GetSubstructMatches(query, uniquify=False, maxMatches=1000) if set(m) == set(core)]
    if not matches:
        return [dict(atom_to_locant)]
    base = matches[0]
    images = []
    for m in matches:
        perm = dict(zip(base, m))
        images.append({a: atom_to_locant[perm[a]] for a in core})
    return images
