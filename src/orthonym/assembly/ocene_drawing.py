"""A substituted metallocene drawn as a metal(2+) and two cyclopentadienide anions, named with its ocenyl prefix.

The ocene table (``data.organometallics``) names the bare sandwich ('ferrocene'); a substituted one was named as a
salt whose anion spelling is not that of the compound ('iron(II) 1-({[4-(cyclopenta-1,3-dien-5-id-1-yl)phenyl]...'),
or not at all. OPSIN reads the ocenyl prefix names ('ferrocen-1-ylmethanol', '4-(ferrocen-1-yl)benzoic acid',
'2-(osmocen-1-yl)ethan-1-ol') to the same ionic drawing, so such a name can be read back to the input.

The ocenyl group has no chokepoint among the producers of substituent prefixes, so it is named through a drawing:
the input with the metal and the bare ring removed and the substituted ring saturated is a cyclopentyl compound; the
engine names it, and the one 'cyclopentyl' prefix of that name is written '(<ocene>-1-yl)'. That is a token rewrite
with a check of everything the rewrite can change:

* Alphanumerical order and the numbering it decides (g)): declined when any other word of the name
  sorts between 'cyclopentyl' and the ocenyl name.
* Enclosing marks: only a whole prefix (not 'cyclopentyloxy', not 'dicyclopentyl') is rewritten; the ocenyl prefix
  has a locant, so it is enclosed, and every mark around it steps up as nests them.
* The structure: the name ships only when OPSIN reads it to the input's full standard InChIKey.

Scope (fail closed, None otherwise): three fragments, a metal of the ocene table with charge 2+, a bare
cyclopentadienide and a cyclopentadienide with its groups on ring carbons (no fused ring), no other charge; a wider tier
only (the default tier keeps its own answer). The label is systematic_verified: an ocene has no PIN here.
"""
import re
from typing import Optional

from rdkit import Chem

from ..data.locant_text import render_locants

#: the metal(2+) -> the ocene stem of the bare sandwich's name ('ferrocene')
_OCENE_STEMS = {'Fe': 'ferrocen', 'Ru': 'ruthenocen', 'Os': 'osmocen', 'Co': 'cobaltocen', 'Ni': 'nickelocen',
                'Cr': 'chromocen', 'V': 'vanadocen', 'Mn': 'manganocen'}
_SURROGATE = 'cyclopentyl'
_PAIR = {('N', 1): ('O', -1), ('O', -1): ('N', 1)}
_MULTIPLIER = r'(?:di|tri|tetra|penta|hexa|bis|tris|tetrakis)'


def _cp_ring(frag):
    """The five ring atoms of a cyclopentadienide fragment (all-carbon five-ring, one ring carbanion, two ring double
    bonds) or None."""
    rings = [r for r in frag.GetRingInfo().AtomRings() if len(r) == 5]
    for r in rings:
        atoms = [frag.GetAtomWithIdx(i) for i in r]
        if any(a.GetSymbol() != 'C' for a in atoms):
            continue
        if [a.GetFormalCharge() for a in atoms].count(-1) != 1 or any(a.GetFormalCharge() not in (0, -1) for a in atoms):
            continue
        km = Chem.Mol(frag)
        try:
            Chem.Kekulize(km, clearAromaticFlags=True)
        except Exception:  # noqa: BLE001 -- not this shape
            continue
        doubles = sum(1 for b in km.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
                      and b.GetBeginAtomIdx() in r and b.GetEndAtomIdx() in r)
        if doubles == 2:
            return r
    return None


def _surrogate(smiles: str):
    """``(ocene stem, [(cyclopentyl drawing, ring atoms, its SMILES, groups)])`` for an input in scope, one entry per
    substituted ring, else None."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) != 3:
        return None
    metals = [f for f in frags if f.GetNumAtoms() == 1 and f.GetAtomWithIdx(0).GetSymbol() in _OCENE_STEMS]
    if len(metals) != 1 or metals[0].GetAtomWithIdx(0).GetFormalCharge() != 2:
        return None
    rest = [f for f in frags if f is not metals[0]]
    rings = []
    for frag in rest:
        if frag.GetNumAtoms() == 5 and _cp_ring(frag) is not None:
            continue                    # a bare ring
        drawn = _saturated(frag)
        if drawn is None:
            return None
        rings.append(drawn)
    if not rings or len(rings) + sum(1 for f in rest if f.GetNumAtoms() == 5 and _cp_ring(f) is not None) != 2:
        return None
    return _OCENE_STEMS[metals[0].GetAtomWithIdx(0).GetSymbol()], rings


def _saturated(sub):
    """``(cyclopentyl drawing, ring atoms, its SMILES, groups)`` of a substituted cyclopentadienide, else None."""
    ring = _cp_ring(sub)
    if ring is None or sub.GetNumAtoms() <= 5:
        return None
    if any(a.GetFormalCharge() and (a.GetSymbol(), a.GetFormalCharge()) not in _PAIR
           or a.GetFormalCharge() and not any((nb.GetSymbol(), nb.GetFormalCharge()) == _PAIR[(a.GetSymbol(), a.GetFormalCharge())]
                                              for nb in a.GetNeighbors())
           for a in sub.GetAtoms() if a.GetIdx() not in ring):
        return None                     # a charge of the group is one of a bonded N+/O- pair ('nitro'), or out of scope
    if sum(a.GetFormalCharge() for a in sub.GetAtoms() if a.GetIdx() not in ring):
        return None
    out = [(i, nb.GetIdx()) for i in ring for nb in sub.GetAtomWithIdx(i).GetNeighbors() if nb.GetIdx() not in ring]
    if not out or any(sub.GetBondBetweenAtoms(*o).GetBondType() != Chem.BondType.SINGLE
                      or sub.GetBondBetweenAtoms(*o).IsInRing() for o in out):
        return None                     # a ring fused to the cyclopentadienide (an indenide) is another ligand
    if any(a.GetIsotope() or a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in sub.GetAtoms() if a.GetIdx() in ring):
        return None
    rw = Chem.RWMol(sub)
    for b in rw.GetBonds():
        if b.GetBeginAtomIdx() in ring and b.GetEndAtomIdx() in ring:
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetIsAromatic(False)
    for i in ring:
        a = rw.GetAtomWithIdx(i)
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
    try:
        surrogate = rw.GetMol()
        Chem.SanitizeMol(surrogate)
    except Exception:  # noqa: BLE001 -- not this shape
        return None
    if sum(1 for r in surrogate.GetRingInfo().AtomRings()
           if len(r) == 5 and all(surrogate.GetAtomWithIdx(i).GetSymbol() == 'C' for i in r)) != 1:
        return None                     # another cyclopentane would make the 'cyclopentyl' token ambiguous
    return surrogate, ring, Chem.MolToSmiles(surrogate), len(out)


def rewrite_surrogate_prefix(name: str, stem: str, groups: int = 1) -> Optional[str]:
    """``name`` with its one 'cyclopentyl' prefix written '(<stem>-1-yl)', or None when the rewrite could change
    anything but that prefix (see the module docstring); ``groups`` is the number of groups on the ring. When the
    cyclopentane is the parent ('cyclopentanecarboxylic acid') the ocene is ('ferrocenecarboxylic acid'):
    (the Blue Book) "Ocenes are considered as heterocycles", 'ferrocenecarbaldehyde' (:3963); its five ring
    positions are numbered as those of cyclopentane, so the locants stay ('1,2-dimethylferrocene'). A ring with more
    groups as a prefix is a substituted 'cyclopentyl' whose attachment is position 1 ('[2-(phenylselanyl)ferrocen-
    1-yl]'), already enclosed. The order of every rewritten name is checked by the spelling layer's check."""
    parent = re.fullmatch(r"(.*?)cyclopentan(e|(?=-\d)|(?=ol$)|(?=amine$))(.*)", name)
    if parent and 'cyclopent' not in parent.group(1) + parent.group(3):
        return f"{parent.group(1)}{stem}{parent.group(2)}{parent.group(3)}"
    hits = [m for m in re.finditer(_SURROGATE, name)]
    if len(hits) != 1:
        return None
    i, j = hits[0].start(), hits[0].end()
    before, after = name[:i], name[j:]
    if re.search('(?:^|[^a-z])' + _MULTIPLIER + '$', before) or re.match(r'(?:id|ium)', after):
        return None                     # 'dicyclopentyl', 'cyclopentylidene', 'cyclopentylium'; the group of a
                                        # compound prefix ('cyclopentylmethyl', 'cyclopentyloxy') is rewritten
    depth = sum(1 if c in '([{' else -1 if c in ')]}' else 0 for c in before)
    if groups > 1:
        # the substituted 'cyclopentyl' of a compound prefix: its own prefixes stand before it, its marks exist
        if not depth or not (before[-1:].isalpha() or before[-1:] in ')]}'):
            return None
        return _in_order(f"{before}{stem}-1-yl{after}")
    if before[-1:].isalpha() and not before.endswith('-'):
        return None                     # one group only: nothing is cited on the ring
    if before.endswith('(') and after.startswith(')'):
        return _in_order(f"{before}{stem}-1-yl{after}")     # its own marks exist: '(cyclopentyl)phosphane'
    if not depth and (resorted := _resorted(name, stem)) is not None:
        return resorted                 # a whole depth-0 prefix, cited in its place among the others
    if not depth and after[:1].isalpha() and not re.match(_MULTIPLIER + r'\(', after):
        return None                     # an unenclosed compound prefix ('1-cyclopentyloxybenzene'); a multiplied
                                        # prefix after it is another prefix ('cyclopentyldi(phenyl)phosphane')
    if not depth and (not before or before.endswith(' ')) and (not after or after.startswith(' ')):
        return f"{before}{stem}-1-yl{after}"   # a word of its own (an ester's organyl group) takes no marks
    if not depth:
        return _in_order(f"{before}({stem}-1-yl){after}")
    raised = _raise_enclosing_marks(f"{before}({stem}-1-yl){after}", len(before))
    return _in_order(raised) if raised else None


def _in_order(name: str) -> Optional[str]:
    """``name`` when its prefixes stand in alphanumerical order, the spelling layer's check: siblings of
    one level only), else None: an ocenyl prefix that would change places with another is not ours to move."""
    from ..validation.spelling.checks_marks import alphanumerical_order_check
    return name if alphanumerical_order_check(None, name) is None else None


_LOCANTS = r"(?:\d+[a-z]?'*|[A-Z]\d*'*)(?:,(?:\d+[a-z]?'*|[A-Z]\d*'*))*"


def _resorted(name: str, stem: str) -> Optional[str]:
    """``name`` with its depth-0 prefixes cited again in alphanumerical order after 'cyclopentyl' is written
    '(<stem>-1-yl)': '3-(2,6-dimethylanilino)-2-(ferrocen-1-yl)-6-methylimidazo[1,2-a]pyridine'. Every depth-0 prefix
    must carry a locant, the split must give back the name, and the engine's own order (``alpha_sort_key``) must be
    the order of the name as it came; None otherwise. The numbering is the surrogate's: a tie of (g) that the
    new order would decide otherwise is not looked at, and the name is never the PIN (see the module text)."""
    from ..validation.spelling.units import parse_name
    from .naming_utils import alpha_sort_key
    lead = re.match(r"\((?:\d+[a-z]?'*[EZRS],?)+\)-", name)      # the stereodescriptors of the whole name
    if lead:
        rest = _resorted(name[lead.end():], stem)
        return lead.group(0) + rest if rest else None
    try:
        _s, _encs, parsed = parse_name(name)
    except Exception:  # noqa: BLE001 -- a name the parser cannot read is not rewritten
        return None
    top = next((p for p in parsed if p.unit.encl is None and p.unit.word == 0), None)
    if top is None or ' ' in name or not top.prefixes or any(not p.locs for p in top.prefixes):
        return None
    texts = []
    for pref in top.prefixes:
        if pref.kind == 'simple':
            texts.append(pref.raw)
        elif pref.kind == 'enclosed' and pref.enc is not None and pref.mult == 1:
            texts.append(name[pref.enc.start:pref.enc.end + 1])
        else:
            return None
    pos, units = 0, []
    for pref, text in zip(top.prefixes, texts):
        found = re.compile(_LOCANTS + '-').match(name, pos)
        if found is None or found.group(0)[:-1] != render_locants(pref.locs) or not name.startswith(text, found.end()):
            return None
        units.append((found.group(0), text))
        pos = found.end() + len(text)
        if len(units) < len(top.prefixes):
            if name[pos:pos + 1] != '-':
                return None
            pos += 1
    parent = name[pos:]     # the parent, with any hydro prefixes, which are not alphabetized ('-3,4-dihydro...')
    if (not parent or parent[0] in '([' or (parent[0] == '-' and not re.compile('-' + _LOCANTS + '-').match(parent))
            or sum(1 for _l, t in units if t == _SURROGATE) != 1):
        return None
    keys = [alpha_sort_key(t) for _l, t in units]
    if keys != sorted(keys):
        return None                     # the order of the name is not the one this key gives: not ours to change
    units = [(loc, f'({stem}-1-yl)' if t == _SURROGATE else t) for loc, t in units]
    units.sort(key=lambda u: alpha_sort_key(u[1]))
    return '-'.join(loc + t for loc, t in units) + parent


_LEVEL = {'(': 1, '[': 2, '{': 3}
_OPEN = {1: '(', 2: '[', 3: '{'}
_CLOSE = {'(': ')', '[': ']', '{': '}'}


def _raise_enclosing_marks(name: str, start: int) -> Optional[str]:
    """``name`` with every pair of marks that encloses the new '(...)' at ``start`` raised, inner to outer, to one
    level above the highest pair inside it: (the Blue Book) nests , , { } in that order
    ('1-[(ferrocen-1-yl)methyl]piperidine'). None when a pair would need a level above { }."""
    pairs, stack = {}, []
    for i, c in enumerate(name):
        if c in '([{':
            stack.append(i)
        elif c in ')]}':
            if not stack:
                return None
            pairs[stack.pop()] = i
    if stack:
        return None
    chars = list(name)
    chain = sorted((o for o, cl in pairs.items() if o < start and cl > start), reverse=True)   # inner to outer
    for o in chain:
        cl = pairs[o]
        inner = max((_LEVEL[chars[x]] for x, y in pairs.items() if o < x and y < cl), default=0)
        need = max(_LEVEL[chars[o]], inner + 1)
        if need > 3:
            return None
        if need != _LEVEL[chars[o]]:
            chars[o], chars[cl] = _OPEN[need], _CLOSE[_OPEN[need]]
    return ''.join(chars)


def ocene_drawing_row(namer, smiles: str) -> Optional[dict]:
    """The ``name_tiered`` row of a substituted ocene in scope at a wider tier, or None (the caller names it)."""
    if not isinstance(smiles, str) or namer._default_tier_policy_applies():
        return None                     # an RDKit Mol input is named through its own SMILES by the caller
    found = _surrogate(smiles)
    if found is None:
        return None
    from ..errors import is_failure_name
    if not is_failure_name(namer._strict_pin_twin_name(smiles)):
        return None                     # the default tier names it ('decamethylferrocene (PIN)'): its answer stands
    stem, rings = found
    if len(rings) == 2:
        from .ocene_two_rings import two_ring_candidates
        candidates = two_ring_candidates(namer, stem, rings)
        row = {}
    else:
        _mol, _ring, surrogate, groups = rings[0]
        row = namer.name_tiered(surrogate)
        if row.get('tier') == 'abstain' or not row.get('name'):
            return None
        candidates = [rewrite_surrogate_prefix(row['name'], stem, groups)]
    from ..validation.opsin_roundtrip import opsin_parse
    key = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
    for name in filter(None, candidates):
        parsed = opsin_parse(name)
        back = Chem.MolFromSmiles(parsed) if parsed else None
        if back is not None and Chem.MolToInchiKey(back) == key:
            break
    else:
        return None
    if not row:
        row = namer.name_tiered(rings[0][2])    # the record fields of a run of the engine
    from rdkit.Chem import rdMolDescriptors
    return {**row, 'name': name, 'tier': 'systematic_verified', 'is_pin': False, 'source': 'ocene_drawing',
            'formula': rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smiles)), 'verified': 'opsin',
            'spelling_failures': [], 'limit_code': None}
