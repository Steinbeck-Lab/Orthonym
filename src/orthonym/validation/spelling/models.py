"""Shared readings of a name for the spelling checks: the abstention marks, the parsed
units of a name, and the numbering models of parents whose numbering symmetries are known from
the name alone (an unbranched carbon chain, a carbocyclic, retained or Hantzsch-Widman
monocycle, naphthalene, anthracene). No naming code, no OPSIN."""
from __future__ import annotations

import re

from .lexer import UnbalancedMarksError
from .units import MULT_RE, REPL_ELEMENT, STEM_RE, STEMS, parse_name

#: lexical marks of the classes (carbohydrates, nucleosides, stereoparents, amino acids and
#: peptides, carotenoids, coordination names): the substitutive checks abstain on them
P10 = re.compile(r"(pyranos|furanos|osyl|uronic|ose\b|osid|glycer|L-|D-|DL-|\bα-|\bβ-|adenosin|guanosin|uridin|cytidin|"
                 r"thymidin|inosin|cholest|androst|pregn|estr|gonan|morphinan|ergolin|abieta|labda|kaura|"
                 r"carotene|retin|prostan|leukotrien|tocopher|alanyl|glycyl|valyl|leucyl|lysyl|prolyl|seryl|"
                 r"threonyl|tyrosyl|phenylalanyl|tryptophyl|cysteinyl|methionyl|histidyl|arginyl|aspartyl|"
                 r"glutamyl|glutaminyl|asparaginyl|isoleucyl|glycine|alanine|valine|leucine|serine|threonine|"
                 r"cysteine|methionine|proline|tyrosine|tryptophan|histidine|lysine|arginine|aspartic|glutamic|"
                 r"glutamine|asparagine|isoleucine|phenylalanine|κ|η|μ)")


def show(text):
    """A unit's text for a failure detail: child enclosures shown as '(...)'."""
    return re.sub(r"\x01\d+\x02", "(...)", text).replace("\x03", "[...]").replace("\x04", "[").replace(
        "\x05", "]").replace("\x06", "(").replace("\x07", ")")


def read_units(name):
    """parse_name(name), or None when the marks do not pair (the check reports that)."""
    try:
        return parse_name(name)
    except UnbalancedMarksError:
        return None


#: retained and Hantzsch-Widman monocycles: name stem -> (size, {locant: element}, mancude)
RINGS = {
    'pyridin': (6, {1: 'N'}, True), 'piperidin': (6, {1: 'N'}, False), 'piperazin': (6, {1: 'N', 4: 'N'}, False),
    'morpholin': (6, {1: 'O', 4: 'N'}, False), 'thiomorpholin': (6, {1: 'S', 4: 'N'}, False),
    'pyrimidin': (6, {1: 'N', 3: 'N'}, True), 'pyrazin': (6, {1: 'N', 4: 'N'}, True),
    'pyridazin': (6, {1: 'N', 2: 'N'}, True),
    'furan': (5, {1: 'O'}, True), 'thiophen': (5, {1: 'S'}, True), 'pyrrol': (5, {1: 'N'}, True),
    'oxolan': (5, {1: 'O'}, False), 'thiolan': (5, {1: 'S'}, False), 'pyrrolidin': (5, {1: 'N'}, False),
    'oxan': (6, {1: 'O'}, False), 'thian': (6, {1: 'S'}, False), 'azetidin': (4, {1: 'N'}, False),
    'oxetan': (4, {1: 'O'}, False), 'thietan': (4, {1: 'S'}, False), 'aziridin': (3, {1: 'N'}, False),
    'oxiran': (3, {1: 'O'}, False), 'thiiran': (3, {1: 'S'}, False), 'azepan': (7, {1: 'N'}, False),
    'azocan': (8, {1: 'N'}, False), 'oxepan': (7, {1: 'O'}, False),
    'imidazol': (5, {1: 'N', 3: 'N'}, True), 'pyrazol': (5, {1: 'N', 2: 'N'}, True),
    'imidazolidin': (5, {1: 'N', 3: 'N'}, False), 'pyrazolidin': (5, {1: 'N', 2: 'N'}, False),
    '1,3-thiazol': (5, {1: 'S', 3: 'N'}, True), '1,3-oxazol': (5, {1: 'O', 3: 'N'}, True),
    '1,2-oxazol': (5, {1: 'O', 2: 'N'}, True), '1,2-thiazol': (5, {1: 'S', 2: 'N'}, True),
    '1,3-thiazolidin': (5, {1: 'S', 3: 'N'}, False), '1,3-oxazolidin': (5, {1: 'O', 3: 'N'}, False),
    '1,2-oxazolidin': (5, {1: 'O', 2: 'N'}, False),
    '1,4-dioxan': (6, {1: 'O', 4: 'O'}, False), '1,3-dioxan': (6, {1: 'O', 3: 'O'}, False),
    '1,3-dioxolan': (5, {1: 'O', 3: 'O'}, False), '1,3-dithiolan': (5, {1: 'S', 3: 'S'}, False),
    '1,3,5-triazin': (6, {1: 'N', 3: 'N', 5: 'N'}, True), '1,2,4-triazin': (6, {1: 'N', 2: 'N', 4: 'N'}, True),
    '1,2,4-triazol': (5, {1: 'N', 2: 'N', 4: 'N'}, True), '1,2,3-triazol': (5, {1: 'N', 2: 'N', 3: 'N'}, True),
    'tetrazol': (5, {1: 'N', 2: 'N', 3: 'N', 4: 'N'}, True),
    '1,2,4-oxadiazol': (5, {1: 'O', 2: 'N', 4: 'N'}, True), '1,3,4-oxadiazol': (5, {1: 'O', 3: 'N', 4: 'N'}, True),
    '1,3,4-thiadiazol': (5, {1: 'S', 3: 'N', 4: 'N'}, True), '1,2,4-thiadiazol': (5, {1: 'S', 2: 'N', 4: 'N'}, True),
    '1,2,5-oxadiazol': (5, {1: 'O', 2: 'N', 5: 'N'}, True), '1,2,5-thiadiazol': (5, {1: 'S', 2: 'N', 5: 'N'}, True),
}
#: (the Blue Book) "their order of citation follows the sequence: F, Cl, Br, I, O, S,
#: Se, Te, N, P, As, Sb, Bi, Si, Ge, Sn, Pb, B, Al, Ga, In, Tl. The locant '1' is given to a heteroatom
#: that occurs first in the seniority sequence"; the same order for 'a' names, (:8484)
_SENIOR = ['F', 'Cl', 'Br', 'I', 'O', 'S', 'Se', 'Te', 'N', 'P', 'As', 'Sb', 'Bi', 'Si', 'Ge', 'Sn', 'Pb', 'B',
           'Al', 'Ga', 'In', 'Tl']
#: retained functional parents with the principal group fixed at C1 of benzene
_BENZ_FUNC = {'benzoic': 1, 'benzoate': 1, 'benzaldehyde': 1, 'benzonitrile': 1, 'benzamide': 1, 'benzoyl': 1,
              'phenol': 1, 'phenoxy': 1, 'aniline': 1, 'anilino': 1, 'phenyl': 1, 'benzenesulfonic': 1,
              'benzenesulfonamide': 1, 'benzenesulfonyl': 1, 'benzohydrazide': 1, 'benzenethiol': 1,
              'benzamido': 1, 'phenolate': 1, 'benzoyloxy': 1, 'phenylsulfanyl': None}
UNSAT = re.compile(r"^(?:(?:" + MULT_RE + r")?(?:en|ene|yn|yne))$")
_UNSAT_SUFFIX = re.compile(r"^((?:" + MULT_RE + r")?(?:en|yn))(e?)(di)?(oic|oate|oyl|oyloxy|al|amide|amido|nitrile|"
                           r"ohydrazide|hydrazide|oyl chloride|imidamide|ethioic|oxamide|thioamide|ohydroxamic)?$")


def _ring_perms(n):
    perms = []
    for start in range(n):
        for direction in (1, -1):
            perms.append({(start + direction * i) % n + 1: i + 1 for i in range(n)})
    return perms


def _chain_perms(n):
    return [{i: i for i in range(1, n + 1)}, {i: n + 1 - i for i in range(1, n + 1)}]


#: an element locant ('N', 'N4', "N'", 'O'): its element, superscript and primes
ELEMENT_LOCANT = re.compile(r"([A-Z][a-z]?)(\d*)('*)")


def int_locs(locs):
    out = []
    for loc in locs:
        if not re.fullmatch(r"\d+", loc):
            return None
        out.append(int(loc))
    return out


#: the symmetry of the fixed numbering of naphthalene (1-8) and anthracene (1-10)
#:, peripheral nonfusion atoms; read off the book's numbering)
_FUSED_PERMS = {
    'naphthalene': [{i: i for i in range(1, 9)},
                    {1: 4, 2: 3, 3: 2, 4: 1, 5: 8, 6: 7, 7: 6, 8: 5},
                    {1: 5, 2: 6, 3: 7, 4: 8, 5: 1, 6: 2, 7: 3, 8: 4},
                    {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}],
    'anthracene': [{i: i for i in range(1, 11)},
                   {1: 4, 2: 3, 3: 2, 4: 1, 5: 8, 6: 7, 7: 6, 8: 5, 9: 9, 10: 10},
                   {1: 5, 2: 6, 3: 7, 4: 8, 5: 1, 6: 2, 7: 3, 8: 4, 9: 10, 10: 9},
                   {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1, 9: 10, 10: 9}],
}


def ring_or_chain_model(p, is_subst_unit):  # noqa: C901 -- one reading per parent kind
    """(kind, n, perms, feats) of a unit whose parent's numbering symmetries are known from the
    name alone (an unbranched carbon chain, a carbocyclic or retained / Hantzsch-Widman
    monocycle, naphthalene, anthracene), or None. ``feats['prefix']`` holds (key, locants) of the
    prefixes on numbered positions, ``feats['prefix_src']`` the prefixes themselves in the same
    order (a prefix whose locants are all element locants goes to ``feats['letter']``)."""
    par = p.parent
    if not par or 'multiplicative' in p.notes or 'isotope_or_other' in p.notes or 'enc_after_parent' in p.notes:
        return None
    feats = {'hetero': {}, 'suffix': [], 'addedH': [], 'unsat_b': [], 'double_b': [], 'hydro': [],
             'prefix': [], 'prefix_src': [], 'ih': []}
    implied = []
    for locs, raw in p.repl:
        mm = re.fullmatch(r"(?:(" + MULT_RE + r"))?(.+)", raw)
        el = REPL_ELEMENT.get(mm.group(2))
        il = int_locs(locs)
        if el is None or il is None:
            return None
        for x in il:
            feats['hetero'][x] = el
    m = re.fullmatch(r"cyclo(" + STEM_RE + r")(a|an|ane|ene|en|yl|ylidene|ane?(?:carbonitrile|carboxylic|"
                     r"carboxamide|carbaldehyde|thiol|ol|amine|one|carbonyl))?", par)
    mc = re.fullmatch(r"(" + STEM_RE + r")(a|an|ane|ene|en|yl|ylidene|anoic|anoate|anoyl|anal|anamide|anenitrile|"
                      r"anedioic|anedioate|anediamide|anedinitrile|anedioyl|anamido|ane?thiol|anoyloxy)?", par)
    if m and STEMS[m.group(1)] >= 3:
        n = STEMS[m.group(1)]
        kind = 'ring'
        tail = m.group(2) or ''
        if tail in ('yl', 'ylidene') or (tail.startswith('an') and len(tail) > 3):
            implied = [1]
    elif par in ('benzene', 'benzen'):
        n, kind = 6, 'ring'
    elif par in _BENZ_FUNC:
        if _BENZ_FUNC[par] is None:
            return None
        n, kind = 6, 'ring'
        implied = [1]
    elif mc and not (is_subst_unit and mc.group(2) in (None, '')):
        n = STEMS[mc.group(1)]
        kind = 'chain'
        tail = mc.group(2) or ''
        if tail in ('yl', 'ylidene', 'anoic', 'anoate', 'anoyl', 'anal', 'anamide', 'anenitrile', 'anamido',
                    'anoyloxy'):
            implied = [1]
        elif tail.startswith('anedi'):
            implied = [1, n]
    elif re.fullmatch(r"naphthalen(?:e)?", par) and not p.hydro and not p.ih:
        n, kind = 'naphthalene', 'fused'
    elif re.fullmatch(r"anthracen(?:e)?", par) and not p.hydro and not p.ih:
        n, kind = 'anthracene', 'fused'
    else:
        key = None
        pl = ','.join(p.parent_locs) if p.parent_locs else ''
        for cand in ((pl + '-' + par) if pl else None, par):
            if cand is None:
                continue
            for stem in sorted(RINGS, key=len, reverse=True):
                if cand.startswith(stem) and re.fullmatch(r"e?", cand[len(stem):]):
                    key = stem
                    break
            if key:
                break
        if not key:
            return None
        n, het, _manc = RINGS[key]
        feats['hetero'].update(het)
        kind = 'ring'
    feats['suffix'] += implied
    kinds = set()
    for locs, txt, added in p.endings:
        il = int_locs(locs)
        if il is None:
            return None
        txt = txt or ''
        mu = _UNSAT_SUFFIX.match(txt)
        if UNSAT.match(txt) or (mu and mu.group(4)):
            feats['unsat_b'] += il
            if 'en' in (mu.group(1) if mu else txt):
                feats['double_b'] += il
            if mu and mu.group(4):
                feats['suffix'] += [1] if not mu.group(3) else [1, n]
                kinds.add(mu.group(4))
            continue
        if txt or added:
            feats['suffix'] += il
        if txt:
            kinds.add(re.sub(r"^(?:" + MULT_RE + r")", '', txt))
        if added:
            feats['addedH'] += [int(x) for x in re.findall(r"(\d+)H", added)]
    if len(kinds) > 1:
        # suffixes of two kinds (a cationic and a radical centre, two free valences of different
        # bond order): criterion (c) of holds them as one set, and the book ranks the kinds
        # within it ('4,4-dimethylpiperazin-4-ium-1-ylium (PIN)' the Blue Book,
        # 'anthracen-9(10H)-yl-10-ylidene (PIN)':24711) by rules this model does not hold
        return None
    for locs, _raw in p.hydro:
        il = int_locs(locs)
        if il is None:
            return None
        feats['hydro'] += il
    for g in p.ih:
        il = int_locs([x[:-1] for x in g])
        if il is None:
            return None
        feats['ih'] += il
    feats['letter'] = []
    for pr in p.prefixes:
        if not pr.locs:
            return None
        mms = [ELEMENT_LOCANT.fullmatch(loc) for loc in pr.locs]
        if all(mms):
            # N-, O-, S- prefixes sit on the suffix group: they take part in criterion (f) with
            # their superscript (the suffix position)
            for mm in mms:
                feats['letter'].append((mm.group(1), int(mm.group(2)) if mm.group(2) else None))
            continue
        il = int_locs(pr.locs)
        if il is None:
            return None
        feats['prefix'].append((pr.key, il))
        feats['prefix_src'].append(pr)
    allpos = feats['suffix'] + feats['unsat_b'] + feats['hydro'] + feats['ih'] + feats['addedH'] + \
        [x for _, lst in feats['prefix'] for x in lst] + list(feats['hetero'])
    if kind == 'fused':
        perms = _FUSED_PERMS[n]
        if any(x not in perms[0] for x in allpos) or feats['unsat_b']:
            return None
        return kind, len(perms[0]), perms, feats
    if any(x < 1 or x > n for x in allpos):
        return None
    perms = _chain_perms(n) if kind == 'chain' else _ring_perms(n)
    return kind, n, perms, feats


def bond_loc(a, b, n, kind):
    if kind == 'ring' and {a, b} == {1, n}:
        return n
    return min(a, b)


def vector(feats, mp, n, kind, letters_first=True):
    """The locant sets of a unit under numbering ``mp``, in the order of the criteria.
    ``letters_first``: an element locant ('N4') ranks below a numeral,:3195); False
    ranks it above (the order the PIN 'N4,2-dimethylpentane-2,4-diamine':26371 needs)."""
    def tr(lst):
        return sorted(mp[x] for x in lst)

    def trb(lst):
        out = []
        for k in lst:
            k2 = k % n + 1 if kind == 'ring' else k + 1
            out.append(bond_loc(mp[k], mp[k2], n, kind))
        return sorted(out)

    het = feats['hetero']
    senior_first = []
    if kind == 'ring' and het:
        top = min(_SENIOR.index(e) if e in _SENIOR else 99 for e in het.values())
        inv = {mp[x]: x for x in mp}
        a1 = inv[1]
        senior_first = [0 if (a1 in het and (_SENIOR.index(het[a1]) if het[a1] in _SENIOR else 99) == top) else 1]
    lrank, nrank = (0, 1) if letters_first else (1, 0)
    letters_ = sorted((lrank, el, mp[s] if s is not None else 0) for el, s in feats.get('letter', []))
    numerals = [(nrank, '', x) for x in tr([x for _, lst in feats['prefix'] for x in lst])]
    v = [senior_first, tr(list(het)),
         [mp[x] for x in sorted(het, key=lambda x: (_SENIOR.index(het[x]) if het[x] in _SENIOR else 99, mp[x]))],
         tr(feats['ih']), tr(feats['suffix']), tr(feats['addedH']),
         sorted(tr(feats['hydro']) + trb(feats['unsat_b'])), trb(feats['double_b']),
         sorted(letters_ + numerals)]
    for _, lst in feats['prefix']:
        v.append(tr(lst))
    return v


LABELS = ['senior heteroatom at 1', 'heteroatoms', 'heteroatoms by seniority', 'IH', 'suffix/free valence',
           'added H', 'hydro/ene/yne', 'double bonds', 'all prefixes']
