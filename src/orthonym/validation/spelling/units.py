"""Unit parser on top of:mod:`.lexer`: per unit (a word of the name or the content of a
substituent enclosure), its detachable prefixes with their locants and alphanumerical key,
hydro prefixes, indicated hydrogen, skeletal replacement prefixes, the parent token and the
endings (suffix, ene/yne, free valence).

 (the Blue Book) defines the unit: "the parent structure or... a unit of
structure as defined by its appropriate enclosing marks". The parser reads a closed lexicon of
simple prefixes; a word it cannot split leaves the rest as the parent token, and the checks
abstain on a parent they do not know.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from .lexer import Tok, letters, units

STEMS = {'meth': 1, 'eth': 2, 'prop': 3, 'but': 4, 'pent': 5, 'hex': 6, 'hept': 7, 'oct': 8, 'non': 9,
         'dec': 10, 'undec': 11, 'dodec': 12, 'tridec': 13, 'tetradec': 14, 'pentadec': 15, 'hexadec': 16,
         'heptadec': 17, 'octadec': 18, 'nonadec': 19, 'icos': 20, 'henicos': 21, 'docos': 22, 'tricos': 23,
         'tetracos': 24, 'pentacos': 25, 'hexacos': 26, 'heptacos': 27, 'octacos': 28, 'nonacos': 29,
         'triacont': 30, 'hentriacont': 31, 'dotriacont': 32, 'tritriacont': 33, 'tetratriacont': 34,
         'pentatriacont': 35, 'hexatriacont': 36, 'heptatriacont': 37, 'octatriacont': 38, 'nonatriacont': 39,
         'tetracont': 40}
STEM_RE = '|'.join(sorted(STEMS, key=len, reverse=True))
MULT = {'di': 2, 'tri': 3, 'tetra': 4, 'penta': 5, 'hexa': 6, 'hepta': 7, 'octa': 8, 'nona': 9, 'deca': 10,
        'undeca': 11, 'dodeca': 12, 'trideca': 13, 'tetradeca': 14, 'pentadeca': 15, 'hexadeca': 16,
        'bis': 2, 'tris': 3, 'tetrakis': 4, 'pentakis': 5, 'hexakis': 6, 'heptakis': 7, 'octakis': 8}
MULT_RE = '|'.join(sorted(MULT, key=len, reverse=True))
#: skeletal replacement ('a') prefixes the reader knows, with the element each one names: the
#: spellings of Table 1.5, general replacement) and Table 2.4,
#: Hantzsch-Widman; 'aluma' and 'indiga' there, the Blue Book), without 'carba'. The reader
#: keeps its own copy, since a check must not inherit a defect of the writer's tables; a test
#: holds it equal to the two tables.
REPL_ELEMENT = {
    'fluora': 'F', 'chlora': 'Cl', 'broma': 'Br', 'ioda': 'I', 'astata': 'At', 'oxa': 'O', 'thia': 'S',
    'selena': 'Se', 'tellura': 'Te', 'polona': 'Po', 'aza': 'N', 'phospha': 'P', 'arsa': 'As', 'stiba': 'Sb',
    'bisma': 'Bi', 'sila': 'Si', 'germa': 'Ge', 'stanna': 'Sn', 'plumba': 'Pb', 'bora': 'B', 'alumina': 'Al',
    'aluma': 'Al', 'galla': 'Ga', 'inda': 'In', 'indiga': 'In', 'thalla': 'Tl'}
REPL = tuple(REPL_ELEMENT)
REPL_RE = '|'.join(sorted(REPL, key=len, reverse=True))
_FUNC_REPL_RE = r"(?:imido|hydrazido|nitrido|peroxo)"
BRIDGE_RE = (r"(?:epoxymethano|methanoxy|epoxyethano|epiminomethano|epithiomethano|methanooxymethano|epidioxy|"
             r"epoxy|epimino|epithio|episulfano|epiazano|methano|ethano|propano|butano|pentano|hexano|metheno|"
             r"etheno|propeno|buteno|benzeno|epidisulfano)")

#: simple substituent prefixes,, -; a prefix outside this lexicon ends the
#: prefix zone, so an unknown word is read as part of the parent, never split by guess
SIMPLE = """fluoro chloro bromo iodo astato oxo thioxo selanylidene sulfanylidene tellanylidene selenoxo imino
hydrazinylidene diazo azido nitro nitroso cyano isocyano cyanato isocyanato thiocyanato isothiocyanato
selenocyanato isoselenocyanato amino hydroxy hydroperoxy sulfanyl disulfanyl trisulfanyl selanyl tellanyl sulfo sulfino
sulfeno carboxy phosphono phosphanyl phosphoryl phosphinyl phosphinato phosphonato arsanyl stibanyl silyl disilanyl
germyl stannyl plumbyl boranyl hydrazinyl diazenyl triazanyl diphosphanyl diselanyl ditellanyl digermyl distannyl
diboranyl diarsanyl triazenyl disilyl azanyl azanylidene oxido sulfido oxy dioxy trioxy
peroxy sulfanediyl ethenyl ethynyl ethenylidene methylidene methylene carbonyl sulfonyl sulfinyl
phenyl phenoxy phenylene benzoyl benzyl benzylidene acetyl formyl carbamoyl sulfamoyl carbamimidoyl
acetamido benzamido formamido carbamoylamino anilino ureido guanidino ammonio azaniumyl oxidanyl
sulfonato carboxylato olato sulfanylato nitrilo phosphanylidene silylidene ylidene
cyclopropyl cyclobutyl cyclopentyl cyclohexyl cycloheptyl cyclooctyl hydroxyimino methoxyimino
oxidanylidene sulfanidyl aminooxy hydroxyamino azanylidyne tert sec""".split()
_SIMPLE_ALTS = sorted(set(SIMPLE), key=len, reverse=True)
_SIMPLE_RE = re.compile(r"(" + '|'.join(_SIMPLE_ALTS) + r")")
_ALKYL_RE = re.compile(r"((?:cyclo)?(?:" + STEM_RE + r")(?:ylidyne|ylidene|yl|oxy|anoyloxy|anoyl|anamido))")
#: a parent name may begin with these prefix strings ('oxolane', 'aminium', 'phenylene')
_PARENT_GUARD = {'oxo': ('lan', 'can', 'nan', 'lane', 'cane', 'nane'), 'hydroxy': ('l',),
                 'diazo': ('xan', 'l', 'n', 't'), 'diazenyl': (), 'azido': (), 'nitro': ('us',),
                 'amino': ('xide', 'ium'), 'phenyl': ('ene',), 'methyl': ('ene', 'ium'), 'oxy': ('gen',),
                 'carbonyl': (), 'benzyl': ('idene',)}
#: a rest that begins with an ending belongs to the parent's own name, so the string before it is
#: no prefix: the radical and ion parents 'oxyl' and 'aminoxyl', 'methoxyl (PIN)',
#: 'bis(chloromethyl)aminoxyl (PIN)' the Blue Book,:40703), '-ylidene', '-ylidyne'
#: ('methoxyboranylidene':36271, 'oxidoazaniumylidyne':43386), '-ylium', '-ium', '-ide', '-uide'
#: ('phenyldisulfanylium':41668)
_ENDING_START = re.compile(r"(?:l|xyl)$|(?:idene|idyne|ylidene|ylidyne|ylium|ium|ide|uide)")
#: the tail of a Hantzsch-Widman name after a replacement prefix whose final 'a' is elided before
#: the stem's vowel: 'ioda' + 'ocine' is 'iodocine', 'oxa' + 'ocine' 'oxocine',
#: 'oxa' + 'onine' 'oxonine', 'broma' + 'olane' 'bromolane'; such a word is the parent, not a prefix
#: ending in 'o' ('iodo', 'oxo', 'bromo') before an unknown word
_HW_TAIL = re.compile(r"(?:cin|can|lan|lidin|nin|nan|l)e?$")


def split_simple(txt: str, nlocs=None):
    """Greedy split of the leading simple prefixes of ``txt``: ([(multiplier, prefix)], rest).
    ``nlocs`` (the locants in front) prefers a reading whose multiplier matches their count."""
    out = []
    rest = txt
    while True:
        cands = []
        for mult in [None] + sorted(MULT, key=len, reverse=True):
            r = rest
            if mult:
                if not r.startswith(mult):
                    continue
                r = r[len(mult):]
            ms = [x for x in (_SIMPLE_RE.match(r), _ALKYL_RE.match(r)) if x]
            if not ms:
                continue
            p = max(ms, key=lambda x: len(x.group(1))).group(1)
            for alt in _SIMPLE_ALTS:
                if r.startswith(alt) and len(alt) > len(p):
                    p = alt
            after = r[len(p):]
            g = _PARENT_GUARD.get(p)
            if g and any(after.startswith(x) for x in g):
                continue
            if _ENDING_START.match(after) or (p.endswith('o') and _HW_TAIL.match(after)):
                continue
            cands.append((len(mult or '') + len(p), mult, p))
        if not cands:
            break
        if nlocs and not out:
            cands = [c for c in cands if (c[1] is None and nlocs == 1) or (c[1] and MULT[c[1]] == nlocs)] or \
                    [c for c in cands if c[1] is None]
            if not cands:
                break
        cands.sort(key=lambda c: (-c[0], c[1] is not None))
        if nlocs and nlocs > 1 and not out:
            mc = [c for c in cands if c[1] and MULT[c[1]] == nlocs]
            if mc:
                cands = mc + cands
        tot, mult, p = cands[0]
        out.append((mult, p))
        rest = rest[tot:]
        nlocs = None
        if not rest:
            break
    return out, rest


@dataclass
class Pref:
    """A detachable prefix of a unit."""
    locs: tuple
    key: str           # the alphanumerical key
    kind: str          # 'simple' | 'enclosed'
    raw: str
    mult: int = 1
    enc: object = None
    nested: bool = False
    ital: tuple = ()


@dataclass
class Parsed:
    """One unit read into prefixes, hydro prefixes, indicated hydrogen, parent and endings."""
    unit: object
    prefixes: list = field(default_factory=list)
    hydro: list = field(default_factory=list)      # [(locs, raw)]
    ih: list = field(default_factory=list)
    repl: list = field(default_factory=list)       # [(locs, raw)] skeletal replacement and bridge prefixes
    parent: str = ''
    parent_locs: tuple = ()                        # locants in front of the parent (Hantzsch-Widman, fusion)
    endings: list = field(default_factory=list)    # [(locs, text, added_ih)]
    order_after_hydro: list = field(default_factory=list)
    stereo: list = field(default_factory=list)
    multiplied: object = None
    notes: list = field(default_factory=list)


def _split_trailing_mult(toks):
    out = []
    for k, t in enumerate(toks):
        if t.kind == 'TXT' and k + 1 < len(toks) and toks[k + 1].kind == 'ENC':
            m = re.search(r"(" + MULT_RE + r")$", t.text)
            if m and m.start() > 0:
                out.append(Tok('TXT', t.text[:m.start()]))
                out.append(Tok('TXT', m.group(1)))
                continue
        out.append(t)
    return out


def _parse_unit(u) -> Parsed:  # noqa: C901 -- one pass over the token stream
    p = Parsed(u)
    toks = _split_trailing_mult(list(u.toks))
    i = 0
    pend_locs = ()
    pend_ital = []
    seen_hydro = False
    state = 'prefix'
    while i < len(toks):
        t = toks[i]
        if state != 'prefix':
            if t.kind == 'LOC':
                if i + 1 < len(toks) and toks[i + 1].kind == 'TXT':
                    p.endings.append((t.locs, toks[i + 1].text, t.enc))
                    i += 2
                    continue
                p.endings.append((t.locs, '', t.enc))
            elif t.kind == 'TXT':
                p.endings.append(((), t.text, None))
            elif t.kind == 'ENC':
                p.notes.append('enc_after_parent')
            i += 1
            continue
        if t.kind == 'STEREO':
            p.stereo.append(t.text)
            i += 1
            continue
        if t.kind == 'LOC':
            if t.enc:      # added indicated hydrogen right after a locant: the ending zone
                state = 'end'
                continue
            pend_locs = t.locs
            i += 1
            continue
        if t.kind == 'ITAL':
            pend_ital.append(t.text)
            i += 1
            continue
        if t.kind == 'IH':
            p.ih.append(t.locs)
            pend_locs = ()
            i += 1
            continue
        if t.kind == 'ENC' and re.fullmatch(BRIDGE_RE, t.text) and pend_locs and len(pend_locs) >= 2:
            p.repl.append((pend_locs, t.text))
            p.notes.append('bridged_fused')
            pend_locs = ()
            i += 1
            continue
        if t.kind == 'ENC':
            pr = Pref(pend_locs, letters(t.text), 'enclosed', t.text, len(pend_locs) or 1, t.enc)
            if seen_hydro:
                p.order_after_hydro.append(pr)
            p.prefixes.append(pr)
            pend_locs = ()
            pend_ital = []
            i += 1
            continue
        if t.kind == 'ASM':
            p.parent = t.text
            p.parent_locs = pend_locs
            pend_locs = ()
            state = 'end'
            i += 1
            continue
        if t.kind == 'OTHER':
            p.notes.append('isotope_or_other')
            i += 1
            continue
        if t.kind != 'TXT':
            i += 1
            continue
        txt = t.text
        # a multiplier in front of an enclosure: 'bis', 'di', 'tri'
        if i + 1 < len(toks) and toks[i + 1].kind == 'ENC' and re.fullmatch(MULT_RE, txt):
            i += 1
            pr_t = toks[i]
            pr = Pref(pend_locs, letters(pr_t.text), 'enclosed', pr_t.text, MULT[txt], pr_t.enc)
            if seen_hydro:
                p.order_after_hydro.append(pr)
            p.prefixes.append(pr)
            pend_locs = ()
            i += 1
            continue
        # skeletal replacement ('a') prefixes: part of the parent
        mr = re.match(r"(?:(" + MULT_RE + r"))?(" + REPL_RE + r")(?![a-z]*?yl$)", txt)
        after = txt[len(mr.group(0)):] if mr else ''
        while mr and after:
            m2 = re.match(r"(?:(" + MULT_RE + r"))?(" + REPL_RE + r")", after)
            if not m2:
                break
            after = after[m2.end():]
        repl_ok = mr and (after == '' or re.match(
            r"(?:(?:bi|tri|tetra|penta|hexa|hepta|octa|nona|deca|undeca|dodeca)?cyclo|(?:di|tri)?spiro|"
            + STEM_RE + r")", after))
        if repl_ok:
            p.repl.append((pend_locs, mr.group(0)))
            pend_locs = ()
            rest = txt[len(mr.group(0)):]
            if rest:
                p.parent = rest[1:] if rest.startswith('-') else rest
                state = 'end'
            i += 1
            continue
        mf = re.match(r"(?:(" + MULT_RE + r"))?" + _FUNC_REPL_RE + r"(?=[a-z]|$)", txt)
        if mf:
            p.repl.append((pend_locs, mf.group(0)))
            p.parent = txt
            p.notes.append('functional_replacement')
            pend_locs = ()
            state = 'end'
            i += 1
            continue
        mh = re.match(r"(?:(" + MULT_RE + r"))?(de)?hydro(?!xy|peroxy|xyl|gen)", txt)
        if mh:
            p.hydro.append((pend_locs, mh.group(0)))
            seen_hydro = True
            pend_locs = ()
            rest = txt[len(mh.group(0)):]
            if rest:
                p.parent = rest
                state = 'end'
            i += 1
            continue
        mb = re.match(r"(?:(" + MULT_RE + r"))?(" + BRIDGE_RE + r")", txt)
        if mb and pend_locs and len(pend_locs) >= 2 and \
                not re.match(r"(?:ic|ate|yl|oyl|ylidene|l\b|ne\b|nitrile|amide|ate)", txt[len(mb.group(0)):]):
            # a bridge prefix: part of the bridged fused parent
            p.repl.append((pend_locs, mb.group(0)))
            p.notes.append('bridged_fused')
            pend_locs = ()
            rest = txt[len(mb.group(0)):]
            if rest:
                p.parent = rest
                state = 'end'
            i += 1
            continue
        pre, rest = split_simple(txt, len(pend_locs) if pend_locs else None)
        for k, (mult, pfx) in enumerate(pre):
            if pfx in ('tert', 'sec'):
                continue
            pr = Pref(pend_locs if k == 0 else (), pfx, 'simple', (mult or '') + pfx,
                      MULT.get(mult, 1) if mult else 1)
            if k == 0 and pend_ital:
                pr.ital = tuple(pend_ital)
            if pre and pre[0][1] in ('tert', 'sec') and k == 1:
                pr.ital = (pre[0][1],)
            if k > 0 and not rest:
                pr.nested = True
            if seen_hydro:
                p.order_after_hydro.append(pr)
            p.prefixes.append(pr)
            if k == 0:
                pend_locs = ()
        if pre and pre[-1][1] in ('tert', 'sec') and not rest:
            i += 1
            continue
        if rest:
            p.parent = rest
            p.parent_locs = pend_locs
            pend_locs = ()
            state = 'end'
        i += 1
    if pend_locs and state == 'prefix':
        p.notes.append('dangling_locs')
    if any(t.kind == 'OTHER' for t in toks):
        p.notes.append('isotope_or_other')
    if not p.parent and p.prefixes and p.prefixes[-1].kind == 'enclosed' and p.prefixes[-1].mult > 1:
        # a multiplicative name, 'bis(4-bromobenzene)': the last enclosure is the multiplied parent
        p.multiplied = p.prefixes.pop()
        p.parent = '<multiplied>'
        p.notes.append('multiplicative')
    if not p.parent and p.prefixes and p.prefixes[-1].kind == 'simple':
        last = p.prefixes.pop()
        if last in p.order_after_hydro:
            p.order_after_hydro.remove(last)
        p.parent = last.raw
        p.parent_locs = last.locs
        p.notes.append('parent_from_last_prefix')
    return p


@lru_cache(maxsize=512)
def parse_name(name: str):
    """(normalised name, enclosures, [Parsed per unit]) of ``name``. Raises
    :class:`.lexer.UnbalancedMarksError` for marks that do not pair. Cached per string: the result is
    shared by every check and must not be changed by a caller."""
    s, encs, us, _mapping = units(name)
    ps = [_parse_unit(u) for u in us]
    mult_encs = {id(p.multiplied.enc) for p in ps if p.multiplied is not None and p.multiplied.enc is not None}
    for u in us:
        tk = _split_trailing_mult(list(u.toks))
        if len(tk) >= 2 and tk[-1].kind == 'ENC' and tk[-2].kind == 'TXT' and re.fullmatch(MULT_RE, tk[-2].text):
            mult_encs.add(id(tk[-1].enc))
            for p in ps:
                if p.unit is u and 'multiplicative' not in p.notes:
                    p.notes.append('multiplicative')
    for p in ps:
        if p.unit.encl is not None and id(p.unit.encl) in mult_encs:
            p.notes.append('multiplied_parent')
            p.notes.append('multiplicative')
    return s, encs, ps
