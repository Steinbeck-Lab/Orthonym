"""Two carboxamides of one molecule that tie as parent structures:, then, on the names of their prefixes.

``rules.amide_parent`` ranks the amides of a molecule by,,,, and,
which are criteria of structures. Two amides that tie on all of them are told apart by the criteria that read the names of
the prefixes (the same step ``rules.ring_tie_names`` takes for two rings):

* "the senior parent structure that has the lower locant or set of locants for substituents cited as prefixes to the
  parent structure (other than 'hydro/dehydro' prefixes) in their order of citation in the name" (the Blue Book, under
  " CRITERIA RELATED TO NUMBER AND LOCATION OF SUBSTITUENT GROUPS",:21596): the locants of the prefixes as the name
  cites them, group by group; the locant 'N' is lower than a numeral "Lowest set of locants",:3189: "Italic capital
  and lower-case letter locants are lower than Greek letter locants, which, in turn, are lower than numerals"). In
  'N-ethyl-2-[2-(methylamino)acetamido]acetamide' the prefixes are cited 'ethyl, (methylamino)acetamido' at 'N,2'; the other
  parent of the same molecule cites 'N,2' too ('N-[2-(ethylamino)-2-oxoethyl]-2-(methylamino)acetamide'), and the amide of
  'C[C@@H]1CCCC[C@@H]1NC(=O)CNCC(=O)NC' that has the N-methyl group cites 'methyl' before 'methylcyclohexylamino...' at 'N,2'
  where the other cites '2,N'.
* and (:22172,:22208) come between the two criteria. They are not worked out: a prefix with a nonstandard
  bonding number (a 'lambda' in its name) or an isotopically modified atom outside the parents leaves a tie of
  undecided.
* "CRITERIA RELATED TO ALPHANUMERICAL ORDER OF NAMES" (:22234): "The preferred IUPAC name is the name that is earlier in
  alphanumerical order (see. Alphabetic letters are considered first in the order that they appear in the name; all
  Roman letters are considered before any italic letters, unless the latter are used as locants or are a part of a compound
  or composite locant, for example, 'N' and '4a'. Then, if still there is a choice, numerical locants are considered in the
  order of their appearance in the name." Two amides that tie have the same parent skeleton, so the letters that decide are
  those of the prefixes as the name cites them; a string of letters that is the start of the other is left undecided (the
  letters of the parent name would decide, and they are not built here).

The parents are chains of acyclic carbon atoms from the carbonyl carbon (C-1): the locant of a prefix is its position on the
chain, or N on the amide nitrogen. A carboxamide on a ring is not worked out (the numbering of the ring decides the locants),
and a prefix without a name, or a shape this does not read, leaves the tie undecided (None) for the caller to label. The
prefixes are named as the sort keys of ``perception.chains`` are (``name_substituent_for_ordering``, inside
``isolated_provenance``, which leaves no trace in the name of the molecule).
"""
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ..errors import is_failure_name
from .lambda_convention import LAMBDA

#: the locant of a prefix on the amide nitrogen, and the one of a prefix on the chain, as the lowest-set comparison reads them
#:: the italic capital 'N' is lower than every numeral)
_N_LOCANT = "N"


def _locant_rank(locant) -> Tuple[int, int]:
    """``(0, 0)`` for the locant N, ``(2, n)`` for the numeral n."""
    return (0, 0) if locant == _N_LOCANT else (2, int(locant))


def _best_chains(mol, carbonyl: int) -> List[Tuple[int, ...]]:
    """The acyl chains of the amide at ``carbonyl`` that tie as parents: those of the smallest ``_path_key``."""
    from .amide_parent import _acyl_paths, _label_map, _path_key
    atom = mol.GetAtomWithIdx(carbonyl)
    nitrogen = next((nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 7), None)
    oxygen = next((nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 8
                   and mol.GetBondBetweenAtoms(carbonyl, nb.GetIdx()).GetBondTypeAsDouble() == 2.0), None)
    labels = _label_map(mol)
    keyed = [(_path_key(mol, p, nitrogen, oxygen, labels), p) for p in _acyl_paths(mol, carbonyl)]
    if not keyed:
        return []
    best = min(k for k, _ in keyed)
    return [p for k, p in keyed if k == best]


def _prefixes(mol, carbonyl: int, path: Tuple[int, ...]) -> Optional[List[Tuple[object, str]]]:
    """``(locant, name)`` of every prefix of the amide parent ``path`` (the chain from ``carbonyl``): each heavy neighbour of
    a chain atom off the chain (C-1, the carbonyl carbon, holds the group of the suffix) and each heavy substituent of the
    amide nitrogen, named whole with everything beyond it. None when one has no name or two of them share an atom."""
    from ..perception.chains import _bfs_substituent
    atom = mol.GetAtomWithIdx(carbonyl)
    nitrogen = next((nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 7), None)
    oxygen = [nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 8
              and mol.GetBondBetweenAtoms(carbonyl, nb.GetIdx()).GetBondTypeAsDouble() == 2.0]
    if nitrogen is None or len(oxygen) != 1:
        return None
    parent = set(path) | {nitrogen} | set(oxygen)
    found: List[Tuple[object, int]] = []
    for pos, a in enumerate(path, start=1):
        if pos == 1:
            continue
        found += [(pos, nb.GetIdx()) for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                  if nb.GetIdx() not in parent and nb.GetAtomicNum() > 1]
    found += [(_N_LOCANT, nb.GetIdx()) for nb in mol.GetAtomWithIdx(nitrogen).GetNeighbors()
              if nb.GetIdx() != carbonyl and nb.GetAtomicNum() > 1]
    prefixes: List[Tuple[object, str]] = []
    used: Set[int] = set()
    for locant, j in found:
        atoms = _bfs_substituent(mol, j, parent)
        if used & set(atoms):
            return None
        used |= set(atoms)
        name = _prefix_name(mol, atoms, j)
        if not name or is_failure_name(name):
            return None
        prefixes.append((locant, name))
    return prefixes


def _prefix_name(mol, atoms, attachment: int) -> Optional[str]:
    """The name of a prefix as a sort key: the name of the tier being run and, where that tier names none (the PIN tier does
    not name a chain of amino and amide groups that holds a ring: '{[2-(cyclohexylamino)-2-oxoethyl]amino}'), the name of the
    best-effort tier. The key decides which amide is the parent, not what a name is called, and the two tiers spell such a
    group alike; a group that is named differently by the two (the wider tier's general names) is read as the tier being
    run reads it where it can."""
    from ..assembly.substituent_enumerator import name_substituent_for_ordering
    from ..metrics.provenance import best_effort_ctx
    name = name_substituent_for_ordering(mol, atoms, attachment)
    if name or best_effort_ctx.get():
        return name
    token = best_effort_ctx.set(True)
    try:
        return name_substituent_for_ordering(mol, atoms, attachment)
    finally:
        best_effort_ctx.reset(token)


def _cited(prefixes: Sequence[Tuple[object, str]]) -> List[Tuple[str, Tuple[object, ...]]]:
    """``(text, locants)`` of the groups of the prefixes in the order the name cites them: equal names multiplied, the groups
    in alphanumerical order (``rules.ring_tie_names._cited_groups``), the locants of a group in their lowest-set order."""
    from .ring_tie_names import _cited_groups
    groups = _cited_groups([(0, name) for _, name in prefixes])
    return [(text, tuple(sorted((loc for loc, nm in prefixes if nm == name), key=_locant_rank)))
            for text, name in groups]


def _parent_key(mol, carbonyl: int) -> Optional[tuple]:
    """`` locants in the order of citation, Roman letters, italic prefixes, numerals, between)`` of the amide parent
    at ``carbonyl`` (the smallest over the chains that tie), or None when it cannot be read. ``between`` is True when
    or would be the next criterion (a nonstandard bonding number in a name, an isotope outside the parent)."""
    from ..assembly.naming_utils import prefix_roman_and_italic
    from .ring_tie_names import _numerals
    keys = []
    for path in _best_chains(mol, carbonyl):
        prefixes = _prefixes(mol, carbonyl, path)
        if prefixes is None:
            return None
        cited = _cited(prefixes)
        cite = tuple(_locant_rank(loc) for _, locs in cited for loc in locs)
        parts = [prefix_roman_and_italic(text) for text, _ in cited]
        numerals = tuple(x for (text, locs), _ in zip(cited, parts)
                         for x in tuple(int(loc) for loc in locs if loc != _N_LOCANT) + _numerals(text))
        parent = set(path)
        between = (any(LAMBDA in name for _, name in prefixes)
                   or any(a.GetIsotope() for a in mol.GetAtoms() if a.GetIdx() not in parent))
        keys.append((cite, "".join(p[0] for p in parts), tuple(i for p in parts for i in p[1]), numerals, between))
    return min(keys) if keys else None


def senior_amide_by_names(mol, carbonyls: Sequence[int]) -> Optional[int]:
    """The carbonyl carbon, among ``carbonyls``, of the amide that and then make the parent; None when they
    do not decide (a ring amide, a prefix without a name, equal keys, one string of letters the start of the other, or
     next after a tie of the locants). ``carbonyls`` are amides that tie on every criterion of
    ``rules.amide_parent.amide_parent_key``, one for each class of atoms the symmetry of the molecule maps onto each other."""
    try:
        from ..metrics.provenance import isolated_provenance
        from ..perception.chains import _P45_NAMING_BRANCHES
        if len(carbonyls) < 2 or _P45_NAMING_BRANCHES.get():
            return None
        if any(nb.IsInRing() and nb.GetAtomicNum() == 6 for c in carbonyls for nb in mol.GetAtomWithIdx(c).GetNeighbors()):
            return None
        token = _P45_NAMING_BRANCHES.set(True)      # a comparison started inside these namings is not made
        try:
            with isolated_provenance():
                keys: Dict[int, Optional[tuple]] = {c: _parent_key(mol, c) for c in carbonyls}
        finally:
            _P45_NAMING_BRANCHES.reset(token)
        if any(k is None for k in keys.values()):
            return None
        lowest = min(keys[c][0] for c in carbonyls)                         #
        tied = [c for c in carbonyls if keys[c][0] == lowest]
        if len(tied) == 1:
            return tied[0]
        if any(keys[c][4] for c in tied):                                    # / come next
            return None
        letters = sorted({keys[c][1] for c in tied})                         #: the Roman letters first
        if len(letters) > 1:
            if letters[1].startswith(letters[0]):
                return None
            return next(c for c in tied if keys[c][1] == letters[0])
        for tier in (2, 3):                                                  # then the italic letters, the numerals
            values = sorted({keys[c][tier] for c in tied})
            if len(values) > 1:
                return next(c for c in tied if keys[c][tier] == values[0])
        return None
    except Exception:  # noqa: BLE001 -- a comparison that cannot be made decides nothing
        return None
