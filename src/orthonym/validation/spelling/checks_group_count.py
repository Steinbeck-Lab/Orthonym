"""Spelling check of the parent of a name by the number of principal characteristic groups it carries
(registered with:func:`orthonym.validation.pin_spelling.register`).

A round trip proves the molecule, never the parent: OPSIN reads 'N-(2-aminoethyl)aniline' back to the
structure of the preferred 'N1-phenylethane-1,2-diamine'.

* (the Blue Book): "The senior parent structure has the maximum number of substituents
  corresponding to the principal characteristic group (suffix) or senior parent hydride in accord with
  the seniority of classes and the seniority of suffixes." (:26375): "Complex
  polyamines, i.e., compounds in which a choice between two or more parent hydrides must be made, are
  systems composed of two or more secondary and/or tertiary amines.... The senior parent amine structure
  is chosen in accordance with the choice of a principal chain or a senior ring or ring system, as
  described in "; the example 'N1-(aminomethyl)ethane-1,2-diamine (PIN)' (the diamine is the parent,
  the monoamine a prefix). A secondary or tertiary amine nitrogen is bonded to a carbon of each parent
  it can have, so it counts as a group of each of them,:26247).

The structure side counts, for the senior class present, the groups every candidate parent carries:
each ring system, and every chain of acyclic carbons (the best path of the acyclic skeleton). A group
is carried by a chain through its carbon (acid, amide, nitrile, aldehyde, ketone: the carbon of the group;
alcohol and amine: a carbon bonded to the oxygen or the nitrogen) and by a ring system through an atom of
the ring (a ketone), or through a ring atom bonded to the carbon of the group (the appended suffixes
'carboxylic acid', 'carboxamide', 'carbonitrile', 'carbaldehyde'), or through a ring atom bonded to the
oxygen or the nitrogen. The name side is the suffix the name cites (its multiplier and its locants) or the
retained name ('aniline', 'phenol', 'benzoic acid',...). When the best candidate carries more groups than
the name's parent, the name is not the PIN. Names of other classes, functional class names, charged
species and anything the reader cannot read with certainty are left alone.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register
from .checks_parent import _CLASS_PAT, _unit_suffix, principal_group_count_check
from .lexer import normalise
from .models import P10, read_units
from .units import STEMS

#: class -> (SMARTS of the group with its key atom first, how a candidate carries it)
#: 'carbon': the key atom is the carbon of the group (a chain carries it through that carbon);
#: 'hetero': the key atom is the oxygen or nitrogen, the carbons bonded to it carry it
_GROUPS = {
    'acid': ('[CX3;!R](=O)[OX2H1]', 'carbon', True),
    'amide': ('[CX3;!R](=O)[NX3;!R;+0;!$(N-[#7])]', 'carbon', True),
    'nitrile': ('[CX2;!R]#[NX1]', 'carbon', True),
    'aldehyde': ('[CX3H1;!R](=O)[#6]', 'carbon', True),
    'ketone': ('[CX3](=O)([#6])[#6]', 'carbon', False),
    'alcohol': ('[OX2H1][#6;!$(C=[O,S,N])]', 'hetero', None),
    'amine': ('[NX3;!R;+0;!$(N-[#6,#16,#15]=[O,S,N]);!$(N-[#7,#8,#16,#15]);!$(N=*)]', 'hetero', None),
}
_PAT = {c: (Chem.MolFromSmarts(s), kind, appended) for c, (s, kind, appended) in _GROUPS.items()}
#: the retained names that cite one group of a class without a suffix
_RETAINED = [
    (re.compile(r"aniline$"), 'amine'),
    (re.compile(r"phenol$"), 'alcohol'),
    (re.compile(r"(?:benzoic|acetic|formic) acid$"), 'acid'),
    (re.compile(r"(?:benz|acet|form)amide$"), 'amide'),
    (re.compile(r"(?:benzonitrile|acetonitrile)$"), 'nitrile'),
    (re.compile(r"(?:benzaldehyde|acetaldehyde)$"), 'aldehyde'),
]
_RANK = {'acid': 6, 'amide': 10, 'nitrile': 13, 'aldehyde': 14, 'ketone': 15, 'alcohol': 16, 'amine': 18}
#: a multiplier elided in front of the suffix ('hexol', 'tetramine'): the count it spells
_ELIDED = re.compile(r"(di|tri|tetr|pent|hex|hept|oct|non|dec)(ol|amine)$")
_ELIDED_COUNT = {'di': 2, 'tri': 3, 'tetr': 4, 'pent': 5, 'hex': 6, 'hept': 7, 'oct': 8, 'non': 9, 'dec': 10}
#: names this check does not judge: retained trivial names of several groups ('oxamide', 'xylitol',
#: 'inositol') and the multiplicative names of / ('N,N'-methylenediethanamine': the
#: parent of such a name is the monoamine that the multiplier repeats, so the groups it holds are not
#: those of one parent)
_NOT_JUDGED = re.compile(r"(?:itol|oxamide|oxalonitrile|oxamic acid|oxalic acid)$"
                         r"|(?:di|tri|tetra|bis|tris)(?:" + "|".join(sorted(STEMS, key=len, reverse=True)) +
                         r")an(?:amine|ol|amide|nitrile|al|one)$")
_MAX_SKELETON = 120      # acyclic carbons of one component, above which the paths are not enumerated


def _groups(mol, cls) -> List[Set[int]]:
    """For each group of class ``cls``: the set of atoms through which a candidate parent carries it."""
    pat, kind, _appended = _PAT[cls]
    out: List[tuple] = []
    seen = set()
    for match in mol.GetSubstructMatches(pat):
        key = match[0]
        if key in seen:
            continue
        seen.add(key)
        if kind == 'carbon':
            out.append(({key}, key))
        else:
            carbons = {nb.GetIdx() for nb in mol.GetAtomWithIdx(key).GetNeighbors() if nb.GetSymbol() == 'C'}
            if carbons:
                out.append((carbons, key))
    return out


def _ring_systems(mol) -> List[Set[int]]:
    systems: List[Set[int]] = []
    for ring in mol.GetRingInfo().AtomRings():
        ring = set(ring)
        merged = [s for s in systems if s & ring]
        for s in merged:
            systems.remove(s)
            ring |= s
        systems.append(ring)
    return systems


def _best_count(mol, cls) -> Optional[int]:
    """The most groups of ``cls`` one candidate parent carries (None when it cannot be told)."""
    groups = _groups(mol, cls)
    if not groups:
        return 0
    _pat, kind, appended = _PAT[cls]
    best = 0
    # ring systems
    for system in _ring_systems(mol):
        n = 0
        for carriers, key in groups:
            if kind == 'hetero':
                hit = bool(carriers & system)
            elif appended:
                hit = any(nb.GetIdx() in system for c in carriers
                          for nb in mol.GetAtomWithIdx(c).GetNeighbors())
            else:
                hit = bool(carriers & system)
            n += hit
        best = max(best, n)
    # chains: the best simple path of the acyclic carbon skeleton (a forest)
    skeleton = {a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'C' and not a.IsInRing()}
    by_node: Dict[int, Set[int]] = {i: set() for i in skeleton}
    for g, (carriers, _key) in enumerate(groups):
        for c in carriers:
            if c in by_node:
                by_node[c].add(g)
    if len(skeleton) > _MAX_SKELETON * 4:
        return None
    adj = {i: [nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors() if nb.GetIdx() in skeleton]
           for i in skeleton}
    for start in skeleton:
        if not by_node[start] and not any(by_node[j] for j in adj[start]):
            continue
        stack = [(start, -1, frozenset(by_node[start]))]
        steps = 0
        while stack:
            node, parent, held = stack.pop()
            best = max(best, len(held))
            steps += 1
            if steps > 20000:
                return None
            for nxt in adj[node]:
                if nxt != parent:
                    stack.append((nxt, node, held | by_node[nxt]))
    return best


def _name_group(name: str):
    """(class, count) of the principal group the name's parent carries, or None."""
    n = normalise(name).strip()
    if P10.search(n) or len(n.split(' ')) > 1 and not n.endswith(' acid'):
        return None
    parsed = read_units(n)
    if parsed is None:
        return None
    _s, encs, ps = parsed
    if any(e.kind in ('isotope', 'isotope_br') for e in encs):
        return None
    words = [p for p in ps if p.unit.encl is None]
    top = [p for p in words if p.parent and p.parent != 'acid']
    if len(top) != 1 or 'multiplicative' in top[0].notes:
        return None
    if _NOT_JUDGED.search(n):
        return None
    sc = _unit_suffix(top[0], [w.unit.text for w in words])
    elided = _ELIDED.search(n)
    if elided and (sc is None or sc[1] < _ELIDED_COUNT[elided.group(1)]):
        cls = 'alcohol' if elided.group(2) == 'ol' else 'amine'
        return cls, _ELIDED_COUNT[elided.group(1)]
    if sc is not None:
        cls, count, junior = sc
        return None if junior or cls not in _PAT else (cls, count)
    for rx, cls in _RETAINED:
        if rx.search(n):
            return cls, 1
    return None


@register("P-44.1.1")
def parent_group_count_check(mol, name) -> Optional[SpellingFailure]:
    """ (:18875): the parent of the name carries the maximum number of principal groups."""
    if mol is None or not name:
        return None
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return None
    if '.' in Chem.MolToSmiles(mol):
        return None
    cited = _name_group(name)
    if cited is None:
        return None
    # the check of the parent's own prefixes (checks_parent) reports the same rule: one finding per name
    if principal_group_count_check(mol, name) is not None:
        return None
    cls, count = cited
    # the senior class present: the class table, which holds the primary and the N-alkyl amines only,
    # plus the amines of this check; the name must cite that class, else the class checks decide
    ranks = [rank for rank, _c, pat in _CLASS_PAT if mol.HasSubstructMatch(pat)]
    if _groups(mol, 'amine'):
        ranks.append(_RANK['amine'])
    if not ranks or min(ranks) != _RANK[cls]:
        return None
    best = _best_count(mol, cls)
    if best is None or best <= count:
        return None
    return SpellingFailure(
        "P-44.1.1",
        f"the parent of the name carries {count} '{cls}' group(s); a chain or ring of the structure carries "
        f"{best}, and the senior parent has the maximum number of principal characteristic groups")
