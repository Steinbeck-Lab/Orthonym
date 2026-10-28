"""The prefixes of a benzene ring as its numbering sees them: (g), (i) and (j) of ``### **** NUMBERING``.

``the Blue Book``. (g) (``:3307``) "lowest locants for the substituent cited first as a prefix in the name" is
decided by alphanumerical order, which leaves the stereodescriptors out (``:3446``): two prefixes that differ only
in R and S are ONE prefix to (g) and their locants one set. (i) (``:3336``) gives the lowest locants to the isotopically
modified atoms, and only then (j) (``:3346``) the lower locant to the CIP stereodescriptors cited inside the prefixes
(``1-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]benzene (PIN)``, ``:3399``): "isotopic modifications have seniority for low
locants over stereodescriptors described in (j) below" (``:3346``).
"""
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


def prefix_citation_locants(
    oriented: List[int],
    substituents: Dict[int, List[Dict]],
    principal_group_positions: Optional[Set[int]] = None,
) -> Tuple[Tuple[int, ...], ...]:
    """The per-prefix locant sets in citation order (``benzene.benzene_prefix_citation_locants``)."""
    by_name: Dict[str, List[int]] = defaultdict(list)
    for locant, name in prefix_entries(oriented, substituents, principal_group_positions):
        by_name[name].append(locant)
    return _prefixes_by_citation(by_name)


def narrow_by_isotopes_and_descriptors(mol, candidates, substituents, principal_group_positions=None):
    """The orientations of ``candidates`` that stand after (i) and then (j): the lowest locants to the
    isotopically modified atoms, the lower locant to the R prefix (the descriptors of the prefixes at the locants
    they hang on)."""
    from ..assembly.locant_omission import isotope_locant_key
    from ..assembly.naming_utils import numbering_cip_key

    def i_key(oriented):
        return isotope_locant_key(mol, {a: i + 1 for i, a in enumerate(oriented)})

    best = min(i_key(o) for o in candidates)
    candidates = [o for o in candidates if i_key(o) == best]
    if len(candidates) == 1:
        return candidates

    def j_key(oriented):
        return numbering_cip_key((), prefix_entries(oriented, substituents, principal_group_positions))

    best = min(j_key(o) for o in candidates)
    return [o for o in candidates if j_key(o) == best]


def prefix_entries(
    oriented: List[int],
    substituents: Dict[int, List[Dict]],
    principal_group_positions: Optional[Set[int]] = None,
) -> List[Tuple[int, str]]:
    """``(locant, name)`` of every prefix the name cites for this orientation of the ring (the prefixes
    :func:`benzene_prefix_citation_locants` describes), the name with its stereodescriptors."""
    from .benzene import _SUFFIX_TO_PREFIX
    pcg_set = set(principal_group_positions or ())
    entries: List[Tuple[int, str]] = []

    def _n_substituent_names(sub):
        """The italic-N prefixes this suffix instance will cite, if any."""
        names = sub.get('n_substituents')
        if not names:
            candidate = sub.get('amine_candidate')
            if isinstance(candidate, dict):
                names = candidate.get('n_substituents')
        return names or ()

    for i, atom_idx in enumerate(oriented):
        for sub in substituents.get(atom_idx) or ():
            if not isinstance(sub, dict):
                continue
            if atom_idx in pcg_set:
                for n_name in _n_substituent_names(sub):
                    if n_name:
                        entries.append((i + 1, n_name))
                continue
            if sub.get('is_suffix'):
                if not pcg_set:
                    continue
                suffix_name = sub.get('suffix_name')
                name = _SUFFIX_TO_PREFIX.get(suffix_name, suffix_name)
            else:
                name = sub.get('name')
            if name:
                entries.append((i + 1, name))
    return entries


def _prefixes_by_citation(by_name: Dict[str, List[int]]) -> Tuple[Tuple[int, ...], ...]:
    """The locants of each prefix, the prefixes in the order they are cited.

    Prefixes that differ only in their stereodescriptors are ONE prefix here: ``****`` (``:3446``)
    leaves the descriptors out of the alphanumerical order, so criterion (g) (``:3307``) cannot tell them
    apart; the choice between them is (j)'s (``:3346``), after (i) (``:3336``), and the locants of the twin
    prefixes are one set (``1-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]benzene (PIN)``, ``:3399``)."""
    from ..assembly.naming_utils import alpha_sort_key
    by_class: Dict[str, List[int]] = defaultdict(list)
    first: Dict[str, str] = {}

    # Ordered by the SHARED total order, not by the raw ``alpha_sort_key``
    # string. 's preamble (:3442) makes this a two-stage comparison --
    # "Nonitalic Roman letters are considered first... When all the Roman
    # letters are identical, the set of locants... are compared" -- and a single
    # key string conflates the stages, because it still carries digits and,
    # before the enclosing-mark strip, brackets. `4-[(1R)-1-chloroethyl]phenoxy`
    # keyed as `1-chloroethylphenoxy`, whose leading '1' (ASCII 49) sorts under
    # every letter, so it was cited ahead of `chloroethyl` and took locant 1 --
    # the wrong numbering for gold row W2F-P8-01. `prefix_citation_sort_key`
    # separates the stages (letters, then locants, then (j) configuration)
    # and is the same authority the rest of the cascade consults.
    from ..assembly.naming_utils import prefix_citation_sort_key
    for name in sorted(by_name, key=prefix_citation_sort_key):
        k = alpha_sort_key(name)
        by_class[k].extend(by_name[name])
        first.setdefault(k, name)
    return tuple(tuple(sorted(by_class[k])) for k in sorted(by_class, key=lambda k: prefix_citation_sort_key(first[k])))
