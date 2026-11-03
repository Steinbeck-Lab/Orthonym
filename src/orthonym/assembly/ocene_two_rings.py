"""An ocene with groups on both rings, named from the cyclopentyl drawing of each ring.

 (the Blue Book) "Ocenes are considered as heterocycles... These names are substituted in accordance
with the principles, rules, and conventions of substitutive nomenclature, using suffixes or prefixes"; the positions
of the second ring are primed: '1,1′′-(ethane-1,2-diyl)bis(1′-methylruthenocene) (PIN)' (:40139). Each substituted
ring is named alone through its cyclopentyl drawing (``ocene_drawing``), and the two names are joined:

* both rings are the parent and carry prefixes only: every group is a prefix on the ocene, each ring numbered for the
  lowest locants: then the prefix cited first in alphanumerical order), the ring with the lower set
  unprimed ('1,1′-dimethylferrocene');
* both rings carry one and the same suffix and nothing else: 'ferrocene-1,1′-dicarboxylic acid';
* one ring is the parent with a suffix, or carries the group of another parent: the groups of the other ring are
  primed prefixes on it ('1′-methylferrocene-1-carboxylic acid', '[1′-(diphenylphosphanyl)ferrocen-1-yl]...').

Every candidate is a name for the caller to read back to the input's full standard InChIKey; none is a PIN here (the
numbering ties of that cross the two rings and the multiplicative names of are not built).
"""
import re
from typing import List, Optional, Tuple

from ..data.locant_text import render_locants

_UNIT_START = re.compile(r"(\d+'*(?:,\d+'*)*)-")


def _loc_key(loc: str) -> Tuple[int, int]:
    """'1' < "1'" < '2': an unprimed locant before the primed one of the same number."""
    return int(loc.rstrip("'")), loc.count("'")


def ring_prefixes(mol, ring) -> Optional[List[Tuple[int, str]]]:
    """``(index in ring, prefix name)`` of every group on the ring of a cyclopentyl drawing, or None when the engine
    has no prefix name for one of them."""
    from ..errors import is_failure_name
    from .substituent_enumerator import name_substituent
    out = []
    for k, i in enumerate(ring):
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            if nb.GetIdx() in ring:
                continue
            seen, stack = {nb.GetIdx()}, [nb.GetIdx()]
            while stack:
                for x in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
                    if x.GetIdx() not in seen and x.GetIdx() not in ring:
                        seen.add(x.GetIdx())
                        stack.append(x.GetIdx())
            name = name_substituent(mol, sorted(seen), nb.GetIdx())
            if not name or name == 'substituent' or is_failure_name(name):
                return None
            out.append((k, name))
    return out


def numbered(placed: List[Tuple[int, str]], prime: str = '') -> List[Tuple[str, str]]:
    """``(locant, name)`` for groups at ring indices 0-4, in the numbering of the five-membered ring with the lowest
    locants, then the lowest for the prefix cited first in alphanumerical order."""
    from .naming_utils import alpha_sort_key
    best = None
    for start in range(5):
        for step in (1, -1):
            pos = {(start + step * n) % 5: n + 1 for n in range(5)}
            locs = [(pos[k], name) for k, name in placed]
            key = (sorted(n for n, _ in locs), [n for n, _ in sorted(locs, key=lambda e: (alpha_sort_key(e[1]), e[0]))])
            if best is None or key < best[0]:
                best = (key, locs)
    return [(f"{n}{prime}", name) for n, name in sorted(best[1])]


def _units(text: str) -> Optional[List[Tuple[List[str], str]]]:
    """The locanted prefixes of ``text`` ('2-(diphenylphosphanyl)-4-methyl') as ``(locants, body)``, the body with its
    multiplier; None when a prefix carries no locant or the text is not such a list."""
    out, pos = [], 0
    while pos < len(text):
        m = _UNIT_START.match(text, pos)
        if m is None:
            return None
        depth, end = 0, m.end()
        while end < len(text):
            c = text[end]
            depth += 1 if c in '([{' else -1 if c in ')]}' else 0
            if depth == 0 and c == '-' and _UNIT_START.match(text, end + 1):
                break
            end += 1
        out.append((m.group(1).split(','), text[m.end():end]))
        pos = end + 1
    return out or None


def _base(locs: List[str], body: str) -> Optional[Tuple[str, str]]:
    """``(bare name, marked name)`` of one prefix of ``body`` cited ``len(locs)`` times, or None."""
    from ..data.chain_names import multiplying_prefix
    from .naming_utils import is_fully_enclosed
    n = len(locs)
    if n > 1:
        for mult in (multiplying_prefix(n), multiplying_prefix(n, 'complex')):
            if mult and body.startswith(mult) and len(body) > len(mult):
                body = body[len(mult):].lstrip('-')
                break
        else:
            return None
    bare = body[1:-1] if is_fully_enclosed(body) else body
    return bare, body


def render(entries: List[Tuple[str, str, str]]) -> str:
    """``(locant, bare, marked)`` entries as one alphanumerically ordered prefix list ('1,1'-dimethyl')."""
    from .naming_utils import alpha_sort_key, enclose_if_compound, multiplied_component
    grouped = {}
    for loc, bare, marked in entries:
        grouped.setdefault(bare, (marked, []))[1].append(loc)
    parts = []
    for bare, (marked, locs) in sorted(grouped.items(), key=lambda kv: (alpha_sort_key(kv[0]), kv[1][1])):
        locs = sorted(locs, key=_loc_key)
        token = multiplied_component(len(locs), bare, marked) if len(locs) > 1 else (
            marked if marked != bare else enclose_if_compound(bare))
        parts.append(f"{render_locants(locs)}-{token}")
    return '-'.join(parts)


def _primed(placed) -> List[Tuple[str, str, str]]:
    from .naming_utils import enclose_if_compound
    return [(loc, name, enclose_if_compound(name)) for loc, name in numbered(placed, "'")]


def ferrocene_parent_name(stem: str, a_placed, b_placed) -> str:
    """Both rings the parent, prefixes only: the ring whose locants are lower (then whose first-cited prefix is) is
    unprimed ('1,1',2-trimethylferrocene')."""
    from .naming_utils import alpha_sort_key, enclose_if_compound
    best = None
    for first, second in ((a_placed, b_placed), (b_placed, a_placed)):
        entries = [(loc, n, enclose_if_compound(n)) for loc, n in numbered(first)] + _primed(second)
        key = (sorted(_loc_key(e[0]) for e in entries),
               [_loc_key(e[0]) for e in sorted(entries, key=lambda e: (alpha_sort_key(e[1]), _loc_key(e[0])))])
        if best is None or key < best[0]:
            best = (key, entries)
    entries = best[1]
    if len(entries) == 10 and len({bare for _loc, bare, _marked in entries}) == 1:
        # (the Blue Book): "All locants are omitted in compounds... in which all substitutable
        # positions are completely substituted or modified... in the same way" ('decamethylferrocene')
        from .naming_utils import multiplied_component
        return f"{multiplied_component(10, entries[0][1], entries[0][2])}{stem}e"
    return f"{render(entries)}{stem}e"


def same_suffix_name(stem: str, name_a: str, name_b: str) -> Optional[str]:
    """'cyclopentanecarboxylic acid' on both rings -> 'ferrocene-1,1'-dicarboxylic acid'; None otherwise."""
    m = re.fullmatch(r"cyclopentane?([a-z][a-z ]*)", name_a)
    if name_a != name_b or m is None:
        return None
    return f"{stem}e-1,1'-di{m.group(1)}"


def merged_prefixes(text: str, b_placed) -> Optional[str]:
    """The locanted prefixes of ``text`` (of ring 1, as the engine numbered them) with the groups of the other ring
    as primed prefixes, in one alphanumerical list; None when ``text`` is not such a list."""
    entries = []
    if text:
        units = _units(text)
        if units is None:
            return None
        for locs, body in units:
            base = _base(locs, body)
            if base is None:
                return None
            entries.extend((loc, base[0], base[1]) for loc in locs)
    return render(entries + _primed(b_placed))


#: principal groups that are parent hydrides, not suffixes: a second one on the other ring is a prefix there
_HYDRIDE_CLASSES = frozenset({'primary_phosphine', 'secondary_phosphine', 'tertiary_phosphine', 'arsine',
                              'stibine', 'silane', 'borane'})
_PARENT = re.compile(r"(.*?)cyclopentan(e|(?=-\d)|(?=ol$)|(?=amine$))(.*)")


def _principal(mol) -> Optional[str]:
    from ..perception.functional_groups import detect_functional_groups
    from ..rules.seniority import get_principal_group
    group = get_principal_group(mol, detect_functional_groups(mol))[0]
    return None if group in _HYDRIDE_CLASSES else group


def _parent_form(name):
    m = _PARENT.fullmatch(name) if name and name.count('cyclopent') == 1 else None
    return m if m and not m.group(1).startswith('(') else None


def _suffixed(stem: str, m, b_placed) -> Optional[str]:
    """Ring 1 the parent with its suffix ('2-methylcyclopentane-1-carboxylic acid'), the other ring's groups primed
    prefixes on it; the suffix takes its locant once the ring carries another group."""
    head = merged_prefixes(m.group(1).rstrip('-'), b_placed)
    if head is None:
        return None
    suffix = m.group(3)
    if not re.match(r"-\d", suffix):
        suffix = f"-1-{suffix}"
    return f"{head}{stem}{m.group(2)}{suffix}"


def _with_ocenyl(name: str, stem: str, groups: int, b_placed) -> Optional[str]:
    """The 'cyclopentyl' prefix of ``name`` (ring 1 with ``groups`` groups) written as the ocenyl prefix that carries
    the other ring's groups as primed prefixes ('[1'-(diphenylphosphanyl)ferrocen-1-yl]di(phenyl)phosphane')."""
    from .naming_utils import enclosing_opener
    from .ocene_drawing import _in_order, _raise_enclosing_marks
    hits = list(re.finditer('cyclopentyl', name))
    if len(hits) != 1 or re.match(r'(?:id|ium)', name[hits[0].end():]):
        return None
    i, j = hits[0].start(), hits[0].end()
    before, after = name[:i], name[j:]
    if groups > 1:
        opener = enclosing_opener(name, i)
        if opener is None:
            return None
        inner = merged_prefixes(name[opener + 1:i], b_placed)
        if inner is None:
            return None
        return _in_order(_raise_enclosing_marks(f"{name[:opener + 1]}{inner}{stem}-1-yl{after}", opener + 1) or '')
    if before[-1:].isalpha() and not before.endswith('-'):
        return None
    inner = f"{merged_prefixes('', b_placed)}{stem}-1-yl"
    if (not before or before.endswith(' ')) and (not after or after.startswith(' ')):
        return f"{before}{inner}{after}"            # an ester's organyl word takes no marks
    if before.endswith('(') and after.startswith(')'):
        text, start = f"{before}{inner}{after}", len(before)
    else:
        text, start = f"{before}({inner}){after}", len(before) + 1
    return _in_order(_raise_enclosing_marks(text, start) or '')


def _multiplied(stem: str, name: str) -> Optional[str]:
    """Both rings the same group of another parent: the multiplicative name, '1,1'-(ferrocene-1,1'-diyl)
    di(ethan-1-one) (PIN)' (the Blue Book); a unit with prefixes of its own is cited with 'bis'."""
    from .naming_utils import enclose_if_compound
    from ..validation.spelling.models import read_units
    if name.count('cyclopentyl') != 1:
        return None
    i = name.index('cyclopentyl')
    before, after = name[:i], name[i + len('cyclopentyl'):]
    m = re.search(r"(?:^|-)(\d+)-$", before)
    lead = before[:m.start(1)] if m else before
    if re.search(r"[()\[\]{}]", lead) or (lead and not lead.endswith('-')) or not after[:1].isalpha():
        return None                     # a prefix of depth 0 only, then the parent with what follows it
    unit = lead + after
    parsed = read_units(unit)
    if parsed is None:
        return None
    loc = m.group(1) if m else None
    locs = f"{loc},{loc}'-" if loc else ''
    if any(p.prefixes for p in parsed[2]):
        marked = enclose_if_compound(unit)
        token = f"bis{marked if marked != unit else '(' + unit + ')'}"
    else:
        token = f"di({unit})" if re.search(r'\d', unit) else f"di{unit}"
    return f"{locs}({stem}e-1,1'-diyl){token}"


def two_ring_candidates(namer, stem: str, rings) -> List[str]:
    """Candidate names, best first, for an ocene with groups on both rings; ``rings`` holds ``(cyclopentyl drawing,
    ring atoms, its SMILES)`` of each ring. The caller ships the first that reads back to the input."""
    from rdkit import Chem
    names = []
    for mol, ring, smi in rings:
        row = namer.name_tiered(smi)
        names.append(None if row.get('tier') == 'abstain' else row.get('name'))
    placed = [ring_prefixes(mol, ring) for mol, ring, _s in rings]
    forms = [_parent_form(n) for n in names]
    out = []
    if all(placed) and all(f is not None and f.group(2) == 'e' and not f.group(3) for f in forms):
        out.append(ferrocene_parent_name(stem, placed[0], placed[1]))
    if all(names) and (same := same_suffix_name(stem, names[0], names[1])):
        out.append(same)
    symmetric = Chem.MolToSmiles(rings[0][0]) == Chem.MolToSmiles(rings[1][0])
    if symmetric and names[0] and forms[0] is None and (multiplied := _multiplied(stem, names[0])):
        out.append(multiplied)
    order = sorted((0, 1), key=lambda k: -len(placed[k] or ()))
    for a in order:
        b = 1 - a
        if not names[a] or not placed[b] or _principal(rings[b][0]) is not None:
            continue                    # a principal group of the other ring would be cited as a prefix
        if forms[a] is not None:
            if forms[a].group(3):
                out.append(_suffixed(stem, forms[a], placed[b]))
        else:
            out.append(_with_ocenyl(names[a], stem, len(placed[a]), placed[b]))
    return [n for n in dict.fromkeys(out) if n]
