"""Spelling checks of the parent choice,,,,,;
registered with:func:`orthonym.validation.pin_spelling.register`).

The senior class present comes from RDKit SMARTS; the class, parent and prefixes the name cites
are read lexically. No naming code, no OPSIN.
"""
from __future__ import annotations

import re

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register
from .lexer import normalise
from .models import P10, RINGS, UNSAT, int_locs, read_units, ring_or_chain_model, show
from .units import MULT, MULT_RE, STEM_RE, STEMS


# ------------------------------------------------------------------ / principal chain
def _chain_model(p, enclosed):
    """``ring_or_chain_model`` of a unit, reading an enclosed free-valence chain whose parent is a
    bare stem with its endings ('hex-5-yn-2-yl', 'prop-2-en-1-yl') as the chain it is."""
    mdl = ring_or_chain_model(p, enclosed)
    if mdl is None and enclosed and any(t in ('yl', 'ylidene') for _, t, _ in p.endings):
        mdl = ring_or_chain_model(p, False)
    return mdl


class _Branch:
    """A prefix that is an unbranched carbon chain attached at its C1: ``m`` atoms, the lower
    atom locants of its multiple bonds ``mult`` (``dbl``: the double bonds), the locants of its own
    prefixes ``subs``, all in its own numbering from the attachment atom."""

    def __init__(self, m, mult, dbl, subs):
        self.m, self.mult, self.dbl, self.subs = m, sorted(mult), sorted(dbl), sorted(subs)


def _branch(pr, by_enc):
    """The:class:`_Branch` of prefix ``pr``, or None when it is not an unbranched carbon chain
    attached at C1 that the reader holds with certainty: 'propyl', 'prop-2-en-1-yl',
    '1-hydroxyethyl', 'hydroxymethyl' (the locant of a one-carbon group is omitted)."""
    if pr.kind != 'enclosed':
        ms = re.fullmatch(r"(" + STEM_RE + r")yl", pr.key)
        if ms and not pr.ital:
            return _Branch(STEMS[ms.group(1)], [], [], [])
        return None
    sub = by_enc.get(id(pr.enc))
    if sub is None or sub.repl or sub.hydro or sub.ih or sub.parent_locs or set(sub.notes) - {'parent_from_last_prefix'}:
        return None
    par = sub.parent or ''
    mult, dbl = [], []
    ms = re.fullmatch(r"(" + STEM_RE + r")yl", par)
    if ms:
        if sub.endings:
            return None
        m = STEMS[ms.group(1)]
    elif re.fullmatch(STEM_RE, par):
        m = STEMS[par]
        free = 0
        for locs, txt, _added in sub.endings:
            il = int_locs(locs)
            if il is None or not txt:
                return None
            if txt == 'yl':
                if il != [1]:
                    return None
                free += 1
            elif UNSAT.match(txt):
                mult += il
                if 'en' in txt:
                    dbl += il
            else:
                return None
        if free != 1:
            return None
    else:
        return None
    subs = []
    for x in sub.prefixes:
        if x.locs:
            il = int_locs(x.locs)
            if il is None or len(il) != max(1, x.mult):
                return None
            subs += il
        elif m == 1:
            subs += [1] * max(1, x.mult)
        else:
            return None
    if any(x < 1 or x > m for x in subs) or any(x < 1 or x >= m for x in mult):
        return None
    return _Branch(m, mult, dbl, subs)


def _best_numbering(n, sfx, mult, dbl, subs):
    """``(sfx, mult, dbl, subs)`` of a chain of ``n`` atoms in the better of its two numberings."""
    forward = (sorted(sfx), sorted(mult), sorted(dbl), sorted(subs))
    reverse = (sorted(n + 1 - x for x in sfx), sorted(n - x for x in mult), sorted(n - x for x in dbl),
               sorted(n + 1 - x for x in subs))
    return min(forward, reverse)


def _alternative_chain(n, k, side, sfx, mult, dbl, subs, br):
    """The principal chain of a unit of ``n`` atoms read through its prefix ``br`` at locant ``k``
    instead of through the tail on ``side`` ('up': the atoms after ``k``; 'down': those before),
    for a branch as long as the tail it replaces: ``(sfx, mult, dbl, subs)`` of that chain in its best
    numbering (suffix and free-valence locants, multiple bonds, double bonds, substituents, each
    compared as a sorted list). The tail becomes one substituent at ``k``; the branch's own
    prefixes join the chain. ``subs`` is the unit's substituent positions with the branch's at ``k``
    once removed."""
    if side == 'up':
        mult_kept = [x for x in mult if x <= k - 1]
        dbl_kept = [x for x in dbl if x <= k - 1]
        subs_kept = [x for x in subs if x <= k]
        at = lambda j: k + j                # noqa: E731 -- position of branch atom j
        bond = lambda j: k + j              # noqa: E731 -- lower locant of branch bond j (atom j-atom j+1)
    else:
        mult_kept = [x for x in mult if x >= k]
        dbl_kept = [x for x in dbl if x >= k]
        subs_kept = [x for x in subs if x >= k]
        at = lambda j: k - j                # noqa: E731
        bond = lambda j: k - j - 1          # noqa: E731
    mult2 = mult_kept + [bond(j) for j in br.mult]
    dbl2 = dbl_kept + [bond(j) for j in br.dbl]
    subs2 = subs_kept + [k] + [at(j) for j in br.subs]
    return _best_numbering(n, sfx, mult2, dbl2, subs2)


@register('P-45.2.1')
def principal_chain_check(mol, name):
    """The principal chain of a name is the one the criteria choose.

     (:20916) the principal chain has the greater number of skeletal atoms, length before
    unsaturation (:18865; '2-ethylideneoctanoic acid (PIN) [not 2-hexylbut-2-enoic acid...]'
    :29938); then, for chains as long, the cascade of (:21016) for a parent chain and of
     'THE PRINCIPAL SUBSTITUENT CHAIN' (:22632) for a substituent chain:
    (d) the greater number of multiple bonds regardless of type, then of double bonds
    :21033,:21066;:22682),
    (i) the lowest locants for multiple bonds regardless of type, then for double bonds
     :21366;:22726; 'hept-1-en-6-yn-4-yl (preferred prefix)':17256),
    (k) the maximum number of substituents cited as prefixes:21604, 'N,N,2-trimethyl-3-
    {...}propanamide (PIN)':21624;:22740) and
    (l) their lowest locants:21698;:22768, '4-hydroxy-3-(2-hydroxyethyl)pentan-
    2-yl (preferred prefix)':22780).

    A prefix at locant i of an unbranched carbon chain that is itself an acyclic carbon chain
    attached at its C1 gives one alternative parent through it, replacing the tail of the chain on
    one side of i; the name fails when the alternative wins at the first criterion that tells the
    two apart. A tie, or a branch the reader does not hold with certainty, passes."""
    if P10.search(name):
        return None
    parsed = read_units(name)
    if parsed is None:
        return None
    _s, _encs, ps = parsed
    by_enc = {id(p.unit.encl): p for p in ps if p.unit.encl is not None}
    for p in ps:
        enclosed = p.unit.encl is not None
        mdl = _chain_model(p, enclosed)
        if mdl is None or mdl[0] != 'chain':
            continue
        _kind, n, _perms, feats = mdl
        if feats['hetero']:
            continue
        # a free-valence chain is a substituent:; a parent chain:
        subst = enclosed or any(t in ('yl', 'ylidene') for _, t, _ in p.endings) \
            or (p.parent or '').endswith('yl')
        occupied_suffix = set(feats['suffix'])
        sfx = list(feats['suffix'])
        mult = list(feats['unsat_b'])
        dbl = list(feats['double_b'])
        subs = sorted(x for _key, locs in feats['prefix'] for x in locs)
        total = len(subs)
        # the prefixes on numbered positions, as the model read them (element-locant prefixes
        # sit on the suffix group and are not part of the chain)
        for pr in feats['prefix_src']:
            if len(pr.locs) != 1 or not pr.locs[0].isdigit() or pr.mult != 1:
                continue
            br = _branch(pr, by_enc)
            if br is None:
                continue
            k = int(pr.locs[0])
            for side in ('up', 'down'):
                tail = list(range(k + 1, n + 1)) if side == 'up' else list(range(1, k))
                if not tail or (set(tail) & occupied_suffix):
                    continue
                t = len(tail)
                if br.m > t:
                    return SpellingFailure('P-44.3.2', f"substituent '{pr.raw[:40]}' at {k} of '{p.parent}' "
                                                       f"carries a longer chain ({br.m} > {t})")
                if br.m < t:
                    continue
                subs_wo = list(subs)
                subs_wo.remove(k)
                _sfx2, mult2, dbl2, subs2 = _alternative_chain(n, k, side, sfx, mult, dbl, subs_wo, br)
                # the unit's own numbering is judged by the numbering checks; the chains are
                # compared in their better numberings
                _sfx1, mult1, dbl1, subs1 = _best_numbering(n, sfx, mult, dbl, subs)
                where = f"'{pr.raw[:40]}' at {k} of '{p.parent}'"
                if len(mult2) != len(mult1):
                    if len(mult2) > len(mult1):
                        return SpellingFailure('P-46.1.4' if subst else 'P-44.4.1.1',
                                               f"{where}: the chain through it has {len(mult2)} multiple "
                                               f"bond(s) vs {len(mult1)}")
                    continue
                if len(dbl2) != len(dbl1):
                    if len(dbl2) > len(dbl1):
                        return SpellingFailure('P-46.1.4' if subst else 'P-44.4.1.2',
                                               f"{where}: the chain through it has {len(dbl2)} double "
                                               f"bond(s) vs {len(dbl1)}")
                    continue
                if mult2 != mult1:
                    if mult2 < mult1:
                        return SpellingFailure('P-46.1.9' if subst else 'P-44.4.1.10.1',
                                               f"{where}: the chain through it has the multiple bonds at "
                                               f"{mult2} vs {mult1}")
                    continue
                if dbl2 != dbl1:
                    if dbl2 < dbl1:
                        return SpellingFailure('P-46.1.9' if subst else 'P-44.4.1.10.1',
                                               f"{where}: the chain through it has the double bonds at "
                                               f"{dbl2} vs {dbl1}")
                    continue
                if len(subs2) != total:
                    if len(subs2) > total:
                        return SpellingFailure('P-45.2.1', f"{where}: chain through it has {len(subs2)} "
                                                           f"prefixes vs {total}")
                    continue
                if subs2 < subs1:
                    return SpellingFailure('P-45.2.2', f"{where}: the chain through it has the prefixes at "
                                                       f"{subs2} vs {subs1}")
    return None


@register('P-45.2.1')
def n_aryl_parent_check(mol, name):
    """ (:21604), example (:21610) '4-methoxy-N-phenylaniline (PIN) [not
    N-(4-methoxyphenyl)aniline...]': of two identical rings joined through the amine nitrogen,
    the parent carries more prefixes."""
    if P10.search(name):
        return None
    parsed = read_units(name)
    if parsed is None:
        return None
    _s, _encs, ps = parsed
    by_enc = {id(p.unit.encl): p for p in ps if p.unit.encl is not None}
    for p in ps:
        if p.unit.encl is not None:
            continue
        pair = None
        if p.parent == 'aniline':
            pair = 'phenyl'
        else:
            mm = re.fullmatch(r"(cyclo(?:" + STEM_RE + r"))anamine", p.parent or '')
            if mm:
                pair = mm.group(1) + 'yl'
        if not pair:
            continue
        ring_pref = sum(max(len(x.locs), x.mult) for x in p.prefixes
                        if x.locs and all(loc.isdigit() for loc in x.locs))
        for pr in p.prefixes:
            if pr.kind != 'enclosed' or pr.locs != ('N',):
                continue
            sub = by_enc.get(id(pr.enc))
            if sub is None or sub.parent != pair or sub.hydro or sub.repl:
                continue
            k = sum(max(len(x.locs), x.mult) for x in sub.prefixes)
            if k > ring_pref:
                return SpellingFailure('P-45.2.1', f"N-substituent '{pr.raw[:40]}' has {k} prefixes, the parent "
                                                   f"ring {ring_pref}: the substituted ring is the parent")
    return None


# ------------------------------------------------------------------ / principal characteristic group
#: (rank, class, SMARTS); a lower rank is senior, Table 4.1,:18162-:18219). A carbamate
#: R-NH-CO-O-R' is an ester (class 9); an acyl on a ring nitrogen and a lactam or lactone
#: carbonyl are ring ketones ('cyclic amides are named as heterocycles',:18184)
_CLASS_SMARTS = [
    (6, 'acid', '[CX3](=[O,S])[OX2H1,SX2H1]'),
    (6, 'acid', '[SX4](=O)(=O)[OX2H1]'),
    (6, 'acid', '[SX3](=O)[OX2H1]'),
    (6, 'acid', '[PX4](=O)([OX2H1])'),
    (7, 'anhydride', '[CX3](=O)-!@[OX2]-!@[CX3](=O)'),
    (8, 'ester', '[CX3](=O)-!@[OX2][#6]'),
    (8, 'ester', '[SX4](=O)(=O)-!@[OX2][#6]'),
    (9, 'acid_halide', '[CX3](=O)[F,Cl,Br,I]'),
    (10, 'amide', '[CX3](=O)-!@[NX3;!R;!$(N-N)]'),
    (10, 'amide', '[SX4](=O)(=O)-!@[NX3;!R;!$(N-N)]'),
    (13, 'nitrile', '[#6][CX2]#[NX1]'),
    (14, 'aldehyde', '[CX3H1](=O)[#6]'),
    (15, 'ketone', '[#6][CX3](=O)[#6]'),
    (15, 'ketone', '[CX3;R](=O)[N,O,S;R]'),
    (16, 'alcohol', '[OX2H1][#6;!$(C=[O,S,N])]'),
    (16, 'alcohol', '[SX2H1][#6;!$(C=[O,S,N])]'),
    (17, 'hydroperoxide', '[OX2H1][OX2]'),
    (18, 'amine', '[NX3;H2,H1,H0;!$(N-[C,S,P]=[O,S,N]);!$(N-[a]);!$(N=*);!$(N#*);!$(N-N);!$(N-O);!$([N+])]([#6])'),
    (18, 'amine', '[NX3;H2;$(N-a)]'),
    (19, 'imine', '[CX3;!R]=[NX2;!R][H,#6]'),
]
_CLASS_PAT = [(r, c, Chem.MolFromSmarts(s)) for r, c, s in _CLASS_SMARTS]
_NAME_CLASS = [
    (re.compile(r"(?:oic|carboxylic|sulfonic|sulfinic|phosphonic|carbothioic|carbodithioic|ic) acid$"), 6, 'acid'),
    (re.compile(r"(?:amide|carboxamide|sulfonamide|carbohydrazide|hydrazide)$"), 10, 'amide'),
    (re.compile(r"(?:nitrile|carbonitrile)$"), 13, 'nitrile'),
    (re.compile(r"(?:carbaldehyde|[a-z]al)$"), 14, 'aldehyde'),
    (re.compile(r"(?:one)$"), 15, 'ketone'),
    (re.compile(r"(?:ol|thiol)$"), 16, 'alcohol'),
    (re.compile(r"(?:amine)$"), 18, 'amine'),
    (re.compile(r"(?:imine)$"), 19, 'imine'),
]


@register('P-41')
def principal_class_check(mol, name):
    """ (:18162) seniority of classes: the class the name cites as its suffix is the senior
    class present in the structure. Functional class names (esters, anhydrides, salts), charged
    species and radicals are not read here."""
    if P10.search(name) or '.' in Chem.MolToSmiles(mol):
        return None
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return None
    n = re.sub(r"\s*\(\d+/\d+\)$", '', normalise(name).strip())
    if len(n.split(' ')) > 1 and not n.endswith(' acid'):
        return None
    cls = next(((rank, c) for rx, rank, c in _NAME_CLASS if rx.search(n)), None)
    if cls is None:
        return None
    senior = None
    for rank, c, pat in _CLASS_PAT:
        if mol.HasSubstructMatch(pat) and (senior is None or rank < senior[0]):
            senior = (rank, c)
    if senior and senior[0] < cls[0]:
        return SpellingFailure('P-41', f"name expresses '{cls[1]}' as the principal class; the structure "
                                       f"has '{senior[1]}'")
    return None


_CLASS_PREFIX = {'acid': ('carboxy',), 'amide': ('carbamoyl',), 'nitrile': ('cyano',), 'aldehyde': ('formyl',),
                 'ketone': ('oxo',), 'alcohol': ('hydroxy',), 'thiol': ('sulfanyl',), 'amine': ('amino',),
                 'imine': ('imino',)}
_SUFFIX_CLASS = [
    (re.compile(r"(?:(" + MULT_RE + r"))?(?:oic acid|carboxylic acid)$"), 'acid'),
    (re.compile(r"(?:(" + MULT_RE + r"))?(?:carboxamide|amide)$"), 'amide'),
    (re.compile(r"(?:(" + MULT_RE + r"))?(?:carbonitrile|nitrile)$"), 'nitrile'),
    (re.compile(r"(?:(" + MULT_RE + r"))?(?:carbaldehyde|al)$"), 'aldehyde'),
    (re.compile(r"(?:(" + MULT_RE + r"))?one$"), 'ketone'),
    (re.compile(r"(?:(" + MULT_RE + r"))?thiol$"), 'thiol'),
    (re.compile(r"(?:(" + MULT_RE + r"))?ol$"), 'alcohol'),
    (re.compile(r"(?:(" + MULT_RE + r"))?amine$"), 'amine'),
]


#: (heading:18462):18470 "The order of seniority for suffixes... is based on the seniority
#: of classes 7 through 20 given in Table 4.1 and includes suffixes modified by functional
#: replacement"; Table 4.4 (:18597) lists them in decreasing order: 'carboxylic acid' / 'oic acid'
#: (1) before the peroxoic and chalcogen ('thioic', 'dithioic', 'selenoic') acids (:18603-:18623),
#: 'carboxamide' / 'amide' (16,:18759) before 'carbothioamide' / 'thioamide' (:18762-:18763),
#: carboximidamides (17,:18764), carbohydrazonamides (18,:18767), sulfonamides (19,:18769) and
#: the other amides down to tellurinamides (30,:18790), 'al' before 'thial', 'selenal', 'tellural'
#: (47,:18823-:18830), 'one' before 'thione', 'selone', 'tellone' (48,:18831-:18835), 'ol' before
#: 'selenol', 'tellurol' (49,:18836-:18840) and 'peroxol' (50,:18841). A suffix of the class that
#: matches one of these patterns is junior to the suffix that the class prefix of _CLASS_PREFIX
#: expresses ('carbamoyl' is the carboxamide, 'oxo' the ketone, 'hydroxy' the alcohol).
_JUNIOR_SUFFIX = {
    'acid': re.compile(r"(?:thi|selen|tellur|perox)oic acid$"),
    'amide': re.compile(r"(?:thio|seleno|telluro|imid|hydrazon|sulfon|sulfin|selenon|selenin|telluron|"
                        r"tellurin)amide$"),
    'aldehyde': re.compile(r"(?:thi|selen|tellur)al$"),
    'ketone': re.compile(r"(?:thi|sel|tell)one$"),
    'alcohol': re.compile(r"(?:selen|tellur|perox)ol$"),
}


def _unit_suffix(p, words):
    """(class, count, junior) of the principal suffix of a top-level unit; ``junior`` is the
    suffix itself when it is junior to the suffix the class prefix expresses, else ''."""
    cands = [t for _, t, _ in p.endings if t] or [p.parent or '']
    last = cands[-1]
    if words and words[-1] == 'acid' and last.endswith(('oic', 'carboxylic')):
        last = last + ' acid'
    for rx, cls in _SUFFIX_CLASS:
        m = rx.search(last)
        if m:
            cnt = MULT.get(m.group(1), 1) if m.group(1) else 1
            locs = [loc for locs, t, _ in p.endings
                    if t and rx.search(t if not t.endswith(('oic', 'carboxylic')) else t + ' acid') for loc in locs]
            if locs:
                cnt = max(cnt, len(locs))
            mj = _JUNIOR_SUFFIX[cls].search(last) if cls in _JUNIOR_SUFFIX else None
            return cls, cnt, (mj.group(0) if mj else '')
    return None


def _class_prefixes(p, cls):
    """The simple prefixes of class ``cls`` on carbon positions of the unit's parent."""
    names = _CLASS_PREFIX.get(cls, ())
    oxo_locs = {loc for x in p.prefixes if x.key == 'oxo' for loc in x.locs}
    het_locs = {loc for x in p.prefixes if x.key in ('hydroxy', 'amino', 'oxo') for loc in x.locs}
    k = 0
    for x in p.prefixes:
        if x.kind != 'simple' or x.key not in names:
            continue
        if any(not loc.isdigit() for loc in x.locs):
            continue            # N-, O- locants: not a carbon of the parent
        if cls in ('alcohol', 'amine') and any(loc in oxo_locs for loc in x.locs):
            continue            # hydroxy or amino with oxo on one carbon: an acid or amide group
        if cls == 'ketone' and any(loc in (het_locs - oxo_locs) for loc in x.locs):
            continue
        k += max(len(x.locs), x.mult)
    return k


@register('P-44.1.1')
def principal_group_count_check(mol, name):
    """ (:18875) "The senior parent structure has the maximum number of substituents
    corresponding to the principal characteristic group (suffix)... in accord with the seniority
    of classes and the seniority of suffixes ": a prefix of the suffix's class
    on the parent itself ('2-carboxybenzoic acid', not the PIN 'benzene-1,2-dicarboxylic acid',
    :4975), or a substituent that carries more of them; (:24096) "for the same number of
    characteristic groups cited as the principal characteristic group, a ring is always selected
    as the parent hydride". When the cited suffix is a junior suffix of the class (a 'carbamoyl'
    prefix beside a 'sulfonamide' suffix, 'oxo' beside 'thione'), the name already fails at the
    choice of the principal characteristic group, (:18470, Table 4.4:18597): the failure is
    reported as in each of the three cases."""
    if P10.search(name):
        return None
    if any(a.GetFormalCharge() for a in mol.GetAtoms()):
        return None
    parsed = read_units(name)
    if parsed is None:
        return None
    _s, encs, ps = parsed
    if any(e.kind in ('isotope', 'isotope_br') for e in encs):
        return None     #: an isotopically modified group is cited separately on purpose
    words = [p for p in ps if p.unit.encl is None]
    top = [p for p in words if p.parent and p.parent != 'acid']
    if len(top) != 1 or 'multiplicative' in top[0].notes:
        return None
    t = top[0]
    sc = _unit_suffix(t, [w.unit.text for w in words])
    if sc is None:
        return None
    cls, m, junior = sc
    pfx = _CLASS_PREFIX[cls][0]
    if _class_prefixes(t, cls):
        if junior:
            return SpellingFailure('P-43', f"'{pfx}' prefix on the parent: its suffix is senior to the cited "
                                           f"'{junior}' suffix")
        return SpellingFailure('P-44.1.1', f"'{pfx}' prefix on the parent that carries the "
                                           f"'{cls}' suffix")
    top_is_chain = re.fullmatch(r"(?:" + STEM_RE + r")(?:a|an|ane|ene|en)?.*", t.parent or '') is not None \
        and not t.repl and not re.match(r"cyclo", t.parent or '')
    ring_alt = r"phenyl|cyclo(?:" + STEM_RE + r")yl|(?:" + '|'.join(sorted(RINGS, key=len, reverse=True)) + r")"
    for p in ps:
        if p is t or p.unit.encl is None:
            continue
        mdl_ring = re.fullmatch(ring_alt, p.parent or '') or (p.parent or '').startswith(tuple(RINGS))
        mdl_chain = re.fullmatch(r"(?:" + STEM_RE + r")yl", p.parent or '') and not p.repl
        if not (mdl_ring or mdl_chain):
            continue
        k = _class_prefixes(p, cls)
        fails = k > m or (k == m and k > 0 and top_is_chain and mdl_ring)
        if fails and junior:
            return SpellingFailure('P-43', f"substituent '{show(p.unit.text)[:40]}' carries '{pfx}', whose "
                                           f"suffix is senior to the cited '{junior}' suffix")
        if k > m:
            return SpellingFailure('P-44.1.1', f"substituent '{show(p.unit.text)[:40]}' carries {k} "
                                               f"'{pfx}' groups, the parent {m}")
        if fails:
            return SpellingFailure('P-52.2.8', f"ring substituent '{show(p.unit.text)[:40]}' carries as many "
                                               f"'{cls}' groups ({k}) as the chain parent")
    return None
