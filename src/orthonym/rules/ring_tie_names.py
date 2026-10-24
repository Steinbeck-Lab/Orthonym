"""Two rings of equal seniority that tie on every criterion that counts or locates groups:, then.

The two rings are joined by one =CH- linker, so either ring is the parent of the name and the other ring
and the linker are one compound prefix on it ('1-[(4-ethylcyclohexyl)methylidene]-4-methylcyclohexane' and
'1-ethyl-4-[(4-methylcyclohexylidene)methyl]cyclohexane' name one molecule). ``ring_selection`` has applied
, (h) and (k), and to the two rings; this is what the book applies next, to the
two NAMES:

* "the senior parent structure that has the lower locant or set of locants for substituents cited
  as prefixes to the parent structure (other than 'hydro/dehydro' prefixes) in their order of citation in
  the name" (the Blue Book, under " CRITERIA RELATED TO NUMBER AND LOCATION OF SUBSTITUENT
  GROUPS",:21596): the locants of the prefixes as the name cites them, group by group ('1,5,1,6' is lower
  than '1,6,1,5' for 'bromo, (bromomethyl), dichloro', the example '1-bromo-5-(bromomethyl)-1,6-
  dichlorohexane (PIN)',:22128). Each ring is numbered as (:3219) numbers it: the suffix first (c),
  the prefixes together (f), then the prefix cited first (g).
* "CRITERIA RELATED ONLY TO SUBSTITUENTS WITH NONSTANDARD BONDING NUMBERS" (:22172) and
  "CRITERIA RELATED ONLY TO ISOTOPIC MODIFICATION OF SUBSTITUENTS" (:22208) come between the two. They are not
  worked out: where a prefix has a nonstandard bonding number (a 'lambda' in its name) or an atom of the
  molecule is isotopically modified, a tie of is left undecided.
* "Criteria related to alphanumerical order of names" (:22234): "The preferred IUPAC name is the name
  that is earlier in alphanumerical order (see. Alphabetic letters are considered first in the order
  that they appear in the name; all Roman letters are considered before any italic letters, unless the
  latter are used as locants or are a part of a compound or composite locant, for example, 'N' and '4a'.
  Then, if still there is a choice, numerical locants are considered in the order of their appearance in
  the name." The multiplying prefixes are letters of the name ('bromo' is earlier than 'dibromo', the
  example (1) of,:22243). Two rings of one size and the same principal groups have the same parent
  hydride and the same suffix, so the letters that decide are those of the prefixes in the order the name
  cites them; a string of letters that is the start of the other is left undecided (the rest of the name,
  which is the same, would decide, and it is not built here).

The prefixes of a ring are named as the sort keys of ``perception.chains`` are (``name_substituent_for_ordering``,
which leaves no trace, inside ``isolated_provenance``; the name of the molecule itself is built afterwards by
the ordinary path on the ring this chooses). A prefix without a name, or a shape this does not work out, leaves
the tie undecided (None) and the caller keeps its honest decline.
"""
import re
from collections import Counter
from typing import List, Optional, Set, Tuple

from ..errors import is_failure_name
from .lambda_convention import LAMBDA

#: ``(ring atom, name)`` of the prefixes of one ring
Prefixes = List[Tuple[int, str]]


def _ring_prefixes(mol, ring_atoms: Set[int], group_atoms: Set[int]) -> Optional[Prefixes]:
    """``(ring atom, name)`` of every prefix of the ring, named as a sort key; None when one has no name.

    A prefix is a heavy neighbour outside the ring that is not an atom of a principal group (the suffix);
    it is named whole, with everything beyond it (the other ring, for the linker)."""
    from ..assembly.substituent_enumerator import name_substituent_for_ordering
    from ..perception.chains import _bfs_substituent
    prefixes: Prefixes = []
    for a in sorted(ring_atoms):
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            j = nb.GetIdx()
            if j in ring_atoms or j in group_atoms or nb.GetAtomicNum() == 1:
                continue
            name = name_substituent_for_ordering(mol, _bfs_substituent(mol, j, ring_atoms), j)
            if not name or is_failure_name(name):
                return None
            prefixes.append((a, name))
    return prefixes


def _cited_groups(prefixes: Prefixes) -> List[Tuple[str, str]]:
    """``(text, name)`` of the distinct prefixes in the order the name cites them: equal names multiplied,
    the groups in alphanumerical order (``perception.chains._cited_prefixes``, the one home of that order)."""
    from ..assembly.naming_utils import get_multiplier_prefix
    from ..perception.chains import _cited_prefixes
    names = [nm for _, nm in prefixes]
    text_of = {nm: ("" if n == 1 else get_multiplier_prefix(n, nm)) + nm for nm, n in Counter(names).items()}
    by_text = {text: nm for nm, text in text_of.items()}
    return [(text, by_text[text]) for text in _cited_prefixes(names)]


def _numbered(cyc: List[int], suffix_atoms: List[int], prefixes: Prefixes,
              groups: List[Tuple[str, str]]) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]:
    """``(locants of the suffix, locants of each cited group of prefixes)`` of the ring numbered as has
    it: lowest locants for the suffix (c), then for all the prefixes together (f), then for the prefix cited
    first (g) (the numbering left that has the lower locants term by term, group by group, in citation order)."""
    n = len(cyc)
    where = [cyc.index(a) for a in suffix_atoms]
    at = [(cyc.index(a), nm) for a, nm in prefixes]
    best = None
    for start in range(n):
        for step in (1, -1):
            def loc(j):
                return ((j - start) * step) % n + 1
            sufl = tuple(sorted(loc(j) for j in where))
            together = tuple(sorted(loc(j) for j, _ in at))
            cited = tuple(tuple(sorted(loc(j) for j, m in at if m == nm)) for _, nm in groups)
            cand = ((sufl, together, tuple(x for g in cited for x in g)), sufl, cited)
            if best is None or cand < best:
                best = cand
    return best[1], best[2]


def tied_numberings(cyc: List[int], suffix_atoms: List[int], prefixes: Prefixes,
                    groups: Optional[List[Tuple[str, str]]] = None) -> List[Tuple[int, int]]:
    """``(start, step)`` of every numbering of the ring (``cyc`` in the order round the ring) that is lowest by
     (:3219): the suffix (c), all the prefixes together (f) and, with ``groups`` (``_cited_groups``), the
    prefixes in the order the name cites them (g); the numberings the isotopic criterion (i) chooses among."""
    n = len(cyc)
    where = [cyc.index(a) for a in suffix_atoms]
    at = [(cyc.index(a), nm) for a, nm in prefixes]
    best, tied = None, []
    for start in range(n):
        for step in (1, -1):
            def loc(j):
                return ((j - start) * step) % n + 1
            key = (tuple(sorted(loc(j) for j in where)), tuple(sorted(loc(j) for j, _ in at)))
            if groups is not None:
                key += (tuple(x for _, nm in groups for x in sorted(loc(j) for j, m in at if m == nm)),)
            if best is None or key < best:
                best, tied = key, [(start, step)]
            elif key == best:
                tied.append((start, step))
    return tied


def prefix_attachments(mol, ring_atoms: Set[int], group_atoms: Set[int]) -> Prefixes:
    """``(ring atom, '')`` of every prefix of the ring, unnamed (the positions the numbering of the ring reads)."""
    return [(a, "") for a in sorted(ring_atoms) for nb in mol.GetAtomWithIdx(a).GetNeighbors()
            if nb.GetIdx() not in ring_atoms and nb.GetIdx() not in group_atoms and nb.GetAtomicNum() != 1]


def named_prefixes(mol, rings: List[Set[int]], group_atoms: Set[int]) -> Optional[List[Prefixes]]:
    """The prefixes of each of the two rings with their names (``_ring_prefixes``), named as sort keys inside
    ``isolated_provenance`` with no comparison started inside those namings; None when one has no name or the
    naming is itself inside such a naming."""
    from ..metrics.provenance import isolated_provenance
    from ..perception.chains import _P45_NAMING_BRANCHES
    if _P45_NAMING_BRANCHES.get():
        return None
    named: List[Prefixes] = []
    token = _P45_NAMING_BRANCHES.set(True)      # a comparison started inside these namings is not made
    try:
        with isolated_provenance():
            for k in (0, 1):
                prefixes = _ring_prefixes(mol, rings[k], group_atoms)
                if prefixes is None:
                    return None
                named.append(prefixes)
    finally:
        _P45_NAMING_BRANCHES.reset(token)
    return named


def _numerals(text: str) -> Tuple[int, ...]:
    """The numerals of a name in the order they appear, a stereodescriptor not being one,:3446)."""
    from ..assembly.naming_utils import strip_alphanumerical_noise
    return tuple(int(x) for x in re.findall(r"\d+", strip_alphanumerical_noise(text.lower())))


def decide_by_names(cycles: List[List[int]], suffix: List[List[int]], named: List[Prefixes],
                    between: bool = False) -> Optional[int]:
    """Index (0 or 1) of the ring that and then make the parent, or None when they do not
    decide (the same locants, letters and numerals; one string of letters the start of the other; the locants
    tie and ``between`` says or would be the next criterion).

    ``cycles`` are the two rings in the order of their atoms (rings of one size), ``suffix`` the ring atoms
    each cites its principal groups at, ``named`` the prefixes of each ring with their names, ``between``
    that a prefix has a nonstandard bonding number or an atom is isotopically modified."""
    from ..assembly.naming_utils import prefix_roman_and_italic
    groups = [_cited_groups(prefixes) for prefixes in named]
    numbered = [_numbered(cycles[k], suffix[k], named[k], groups[k]) for k in (0, 1)]
    cite = [tuple(x for g in cited for x in g) for _, cited in numbered]
    if cite[0] != cite[1]:
        return 0 if cite[0] < cite[1] else 1
    if between:
        return None
    letters, italics, numerals = [], [], []
    for k in (0, 1):
        parts = [prefix_roman_and_italic(text) for text, _ in groups[k]]
        letters.append("".join(p[0] for p in parts))
        italics.append(tuple(i for p in parts for i in p[1]))
        sufl, cited = numbered[k]
        numerals.append(tuple(x for (text, _), locs in zip(groups[k], cited)
                              for x in locs + _numerals(text)) + sufl)
    a, b = letters
    if a != b:
        if a.startswith(b) or b.startswith(a):
            return None
        return 0 if a < b else 1
    for tier in (italics, numerals):
        if tier[0] != tier[1]:
            return 0 if tier[0] < tier[1] else 1
    return None


def senior_ring_by_names(mol, rings: List[Set[int]], cycles: List[List[int]], suffix: List[List[int]],
                         group_atoms: Set[int]) -> Optional[int]:
    """Index (0 or 1) of the ring that and then make the parent, or None when they do not
    decide (a prefix without a name, a name that cannot be compared, the same name by every criterion).

    ``rings`` are the atoms of the two rings, ``cycles`` the same in the order round the ring (saturated
    carbocyclic monocycles, the only rings ``ring_selection`` numbers), ``suffix`` the ring atoms each cites
    its principal groups at, ``group_atoms`` the atoms of the principal groups."""
    try:
        if len(cycles[0]) != len(cycles[1]):
            return None
        named = named_prefixes(mol, rings, group_atoms)
        if named is None:
            return None
        from ..assembly.locant_omission import skeleton_label_on_atoms
        #: an isotopic modification of a SUBSTITUENT; the labels of the parents were compared by
        # (``ring_tie_isotopes``) before began
        outside = [a.GetIdx() for a in mol.GetAtoms()
                   if a.GetIdx() not in rings[0] and a.GetIdx() not in rings[1] and a.GetIdx() not in group_atoms]
        between = (any(mol.GetAtomWithIdx(a).GetIsotope() for a in outside)
                   or skeleton_label_on_atoms(mol, outside)
                   or any(LAMBDA in name for prefixes in named for _, name in prefixes))
        return decide_by_names(cycles, suffix, named, between)
    except Exception:  # noqa: BLE001 -- a comparison that cannot be made decides nothing
        return None
