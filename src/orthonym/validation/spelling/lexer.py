"""Lexical reader of a substitutive name, for the spelling checks (no naming code, no OPSIN).

*:func:`normalise` -- primes as ``'``, no superscript digits, no markdown marks.
*:func:`enclosures` -- every pair of enclosing marks, as a tree, each classified as a
  *counted* mark (a substituent enclosure, a stereodescriptor set, a compound locant, an
  isotope descriptor) or an *integral* one (fusion, von Baeyer, spiro and ring-assembly
  descriptors, added indicated hydrogen), the distinction of -.4
  (the Blue Book-:7506).
*:func:`units` -- the words of the name and the content of every substituent enclosure,
  each read as a token stream with its child enclosures collapsed to placeholders.
*:func:`letters` -- the nonitalic Roman letters of a prefix for alphanumerical order
  ,:3442).

A reader that cannot make sense of a string never guesses::class:`UnbalancedMarksError` is raised
for marks that do not pair, and the checks treat any other unknown shape as an abstention.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_SUPER = {0x2070: '0', 0xB9: '1', 0xB2: '2', 0xB3: '3', 0x2074: '4', 0x2075: '5', 0x2076: '6',
          0x2077: '7', 0x2078: '8', 0x2079: '9'}


def normalise(s: str) -> str:
    """The name with primes written ``'`` and superscript digits as plain digits."""
    s = s.replace('′', "'").replace('″', "''").replace('‴', "'''").replace('`', "'")
    s = re.sub(r'</?sup>', '', s)
    s = s.replace('*', '')
    s = ''.join(_SUPER.get(ord(c), c) for c in s)
    s = s.replace('–', '-').replace('−', '-')
    return s


OPEN = {'(': ')', '[': ']', '{': '}'}
CLOSE = {v: k for k, v in OPEN.items()}

_STEREO_PIECE = re.compile(r"^(?:(?:\d+[a-z]?'*(?:\(\d+'*\))?|[A-Z][a-z]?\d*'*|\d+[a-z]?'*,\d+'*)?"
                           r"(?:R|S|E|Z|r|s|M|P|RS|SR|Ra|Sa|Rp|Sp|ξ|R\*|S\*|EZ|ZE)\*?)$")
_ADDED_IH = re.compile(r"^\d+[a-z]?'*H(?:,\d+[a-z]?'*H)*$")
#: an isotope descriptor, the Blue Book; '(N-2H1)acetamide (PIN)':43828):
#: nuclides with their count, after locants that may be numerals or element locants ('(13C)',
#: '(2-2H1)', '(N-2H1)', '(N,N-2H2)', '(O-18O)')
_ISOTOPE = re.compile(r"^\s*(?:(?:\d+[a-z]?'*|[A-Z][a-z]?\d*'*)(?:\s*,\s*(?:\d+[a-z]?'*|[A-Z][a-z]?\d*'*))*\s*-\s*)?"
                      r"(?:\d+\s*(?:H|C|N|O|S|P|F|Cl|Br|I|Se|B)\s*\d*\s*,?\s*)+$")
_ASSEMBLY = re.compile(r"^\d+'*[a-z]?,\d+'*[a-z]?(?:[:,]\d+'*[a-z]?)*-"
                       r"(?:bi(?!s)|ter|quater|quinque|sexi|septi|octi|novi|deci)")


@dataclass
class Encl:
    """One pair of enclosing marks of the normalised name."""
    typ: str            # '(' '[' '{'
    start: int          # index of the opening mark
    end: int            # index of the closing mark
    content: str
    kind: str = ''      # subst | stereo | added_ih | compound_locant | fusion | vonbaeyer | spiro
    # | assembly | assembly_paren | annulene | component_loc | isotope
    # | isotope_br | ratio | other_int
    children: list = field(default_factory=list)
    parent: object = None

    @property
    def counted(self) -> bool:
        """/.2 (the Blue Book,:7469): the parentheses of added indicated
        hydrogen and the brackets of fusion, spiro, ring-assembly and von Baeyer descriptors are
        ignored in the nesting order; /.4 (:7478,:7501): those of compound locants,
        stereodescriptors and isotope descriptors are taken into consideration."""
        return self.kind in ('subst', 'stereo', 'compound_locant', 'isotope')


class UnbalancedMarksError(Exception):
    """The enclosing marks of the name do not pair."""


def enclosures(s: str) -> list:
    """Every pair of enclosing marks of ``s`` (a normalised name), with its tree and kind."""
    stack, out = [], []
    for i, c in enumerate(s):
        if c in OPEN:
            stack.append((c, i))
        elif c in CLOSE:
            if not stack or stack[-1][0] != CLOSE[c]:
                raise UnbalancedMarksError(f'{c!r} at {i}')
            t, j = stack.pop()
            out.append(Encl(t, j, i, s[j + 1:i]))
    if stack:
        raise UnbalancedMarksError(f'open {stack[-1][0]!r} at {stack[-1][1]}')
    out.sort(key=lambda e: e.start)
    for e in out:
        best = None
        for f in out:
            if f is not e and f.start < e.start and e.end < f.end:
                if best is None or f.start > best.start:
                    best = f
        e.parent = best
        if best is not None:
            best.children.append(e)
    for e in out:
        _classify(s, e)
    return out


def _classify(s: str, e: Encl) -> None:
    c = e.content
    before = s[:e.start]
    after = s[e.end + 1:]
    prevch = before[-1:] if before else ''
    if e.typ == '(':
        pieces = re.split(r',(?![^(]*\))', c)
        if _ADDED_IH.match(c) and prevch and (prevch.isdigit() or prevch in "'" or prevch.isalpha()):
            e.kind = 'added_ih'
        elif re.fullmatch(r"\d+'*[a-z]?", c) and prevch and (prevch.isdigit() or prevch == "'"):
            e.kind = 'compound_locant'
        elif c and (all(_STEREO_PIECE.match(p.strip()) for p in pieces)
                    or re.fullmatch(r"(?:T|SP|OC|TB|TBPY|SPY|TPR|PBPY|CU|SS|TS)-\d+(?:-[\dA-Z]+)?", c)) \
                and not re.search(r'[a-z]{3}', c):
            e.kind = 'stereo'
        elif re.fullmatch(r"\d+/\d+(?:/\d+)*", c):
            e.kind = 'ratio'
        elif re.search(r"(?:^|[-'\d])(?:bi|ter|quater)$", before) and not re.search(r"[-,]\d", c[:1]):
            e.kind = 'assembly_paren'
        elif _ISOTOPE.match(c):
            e.kind = 'isotope'
        elif re.fullmatch(r"[\d,']+", c) and prevch == '':
            e.kind = 'other_int'
        else:
            e.kind = 'subst'
    elif e.typ == '[':
        tail = re.search(r'([A-Za-z]+)$', before)
        w = tail.group(1).lower() if tail else ''
        if w.endswith('cyclo') and re.fullmatch(r"[\d.,^'\s]+", c):
            e.kind = 'vonbaeyer'
        elif re.search(r'spiro(?:bi|ter|quater)?$', w) and re.fullmatch(r"[\d.,^'\s]+", c):
            e.kind = 'vonbaeyer'
        elif re.search(r'spiro(?:bi|ter|quater)?$', w):
            e.kind = 'spiro'
        elif re.fullmatch(r"\s*(?:\d+[a-z]?'*\s*-\s*)?(?:\d+\s*[A-Z][a-z]?\s*\d*\s*,?\s*)+", c):
            e.kind = 'isotope_br'
        elif _ASSEMBLY.match(c):
            e.kind = 'assembly'
        elif re.fullmatch(r"[\d,'a-z:;.\-\s]+", c) and w and prevch.isalpha():
            e.kind = 'fusion'
        elif re.fullmatch(r"\d+", c) and after[:1].isalpha():
            e.kind = 'annulene'           # [10]annulene, benzo[8]annulene
        elif re.fullmatch(r"[\d,']+", c) and after[:1].isalpha():
            e.kind = 'component_loc'      # [1,2,4]triazolo, [1,3]dioxolo
        elif _ISOTOPE.match(c):
            e.kind = 'isotope'
        elif re.fullmatch(r"[\d,'a-z:;.\-]+", c) and prevch.isalpha():
            e.kind = 'fusion'
        else:
            e.kind = 'subst'
    else:
        e.kind = 'subst'


# ----------------------------------------------------------------- units and tokens
@dataclass
class Tok:
    """A token of a unit: LOC (a locant set), IH (an indicated-hydrogen group), ENC (a child
    substituent enclosure), STEREO, OTHER (isotope, ratio), ITAL (an italic prefix), ASM (a
    ring-assembly descriptor), TXT (letters)."""
    kind: str
    text: str
    locs: tuple = ()
    enc: object = None


@dataclass
class Unit:
    """A word of the name (``encl`` None) or the content of a substituent enclosure."""
    text: str
    toks: list
    encl: object = None
    word: int = 0


_ELEM_LOC = ('N', 'O', 'S', 'Se', 'Te', 'P', 'As', 'Sb', 'Bi', 'B', 'Si', 'Ge', 'Sn', 'C', 'Hg', 'Pb', 'Al')
_LOC_ONE = (r"(?:\d+[a-z]?'*(?:λ\d+)?(?:\(\d+'*\))?|(?:" + '|'.join(sorted(_ELEM_LOC, key=len, reverse=True))
            + r")(?:\d+[a-z]?)?'*|[α-ωΑ-Ω]\d*'*|λ\d+)")
_LOCSET = re.compile(r"(" + _LOC_ONE + r"(?:," + _LOC_ONE + r")*)((?:\x06[^\x07]*\x07)?)-")
_IHSET = re.compile(r"(\d+[a-z]?'*H(?:,\d+[a-z]?'*H)*)-")
_PH = re.compile(r"\x01(\d+)\x02")
_INT = '\x03'


def _collapse(s, e_start, e_end, encs, mapping):
    """``s[e_start:e_end]`` with each top child enclosure replaced: substituent, stereo, compound
    locant, isotope and ratio enclosures by a placeholder (``mapping`` keeps them), assemblies by
    their text between \\x04 and \\x05, added indicated hydrogen between \\x06 and \\x07, and the
    integral descriptors (fusion, von Baeyer, spiro, annulene, component locants) by one masked
    character."""
    out, i = [], e_start
    kids = [e for e in encs if e_start <= e.start and e.end < e_end and
            (e.parent is None or not (e_start <= e.parent.start and e.parent.end < e_end))]
    kids.sort(key=lambda e: e.start)
    for k in kids:
        out.append(s[i:k.start])
        if k.kind in ('subst', 'stereo', 'compound_locant', 'isotope', 'isotope_br', 'ratio', 'other_int'):
            mapping.append(k)
            out.append(f'\x01{len(mapping) - 1}\x02')
        elif k.kind in ('assembly', 'assembly_paren'):
            out.append('\x04' + k.content + '\x05')
        elif k.kind == 'added_ih':
            out.append('\x06' + k.content + '\x07')
        else:
            out.append(_INT)
        i = k.end + 1
    out.append(s[i:e_end])
    return ''.join(out)


def _tokenize(t, mapping):
    toks, i = [], 0
    while i < len(t):
        c = t[i]
        if c in '- ':
            i += 1
            continue
        m = _PH.match(t, i)
        if m:
            e = mapping[int(m.group(1))]
            if e.kind == 'stereo':
                toks.append(Tok('STEREO', e.content, enc=e))
            elif e.kind == 'compound_locant':
                if toks:
                    last = toks[-1]
                    toks[-1] = Tok(last.kind, last.text + '(' + e.content + ')', last.locs, last.enc)
            elif e.kind in ('isotope', 'isotope_br', 'ratio', 'other_int'):
                toks.append(Tok('OTHER', e.content, enc=e))
            else:
                toks.append(Tok('ENC', e.content, enc=e))
            i = m.end()
            continue
        m = _IHSET.match(t, i)
        if m and (i == 0 or t[i - 1] in '- \x02'):
            toks.append(Tok('IH', m.group(1), tuple(m.group(1).split(','))))
            i = m.end()
            continue
        m = _LOCSET.match(t, i)
        if m and (i == 0 or t[i - 1] in '- \x02,'):
            toks.append(Tok('LOC', m.group(1), tuple(m.group(1).split(',')), enc=(m.group(2)[1:-1] or None)))
            i = m.end()
            continue
        m = re.match(r"(tert|sec|cis|trans|rel|rac|endo|exo|syn|anti|meso|erythro|threo|ent|D|L|DL|as|s|r|c|t)"
                     r"-(?=[a-z\x01])", t[i:])
        if m and (i == 0 or t[i - 1] in '- \x02'):
            toks.append(Tok('ITAL', m.group(1)))
            i += m.end()
            continue
        if c == '\x04':
            j = t.index('\x05', i)
            toks.append(Tok('ASM', t[i + 1:j]))
            i = j + 1
            continue
        j = i
        while j < len(t) and t[j] not in '- \x01\x04':
            j += 1
        toks.append(Tok('TXT', t[i:j]))
        i = j
    return toks


def units(name: str):
    """(normalised name, enclosures, units, placeholder mapping) of ``name``."""
    s = normalise(name)
    encs = enclosures(s)
    out, mapping = [], []
    top = _collapse(s, 0, len(s), encs, mapping)
    for wi, w in enumerate(top.split(' ')):
        out.append(Unit(w, _tokenize(w, mapping), None, wi))
    for e in encs:
        if e.kind != 'subst':
            continue
        sub = _collapse(s, e.start + 1, e.end, encs, mapping)
        out.append(Unit(sub, _tokenize(sub, mapping), e, -1))
    return s, encs, out, mapping


def letters(text: str) -> str:
    """The nonitalic Roman letters of a prefix name for alphanumerical order,:3442):
    locants, italic prefixes, stereodescriptors, indicated hydrogen, isotope descriptors and
    the integral descriptors removed; the letters of enclosed substituent names are kept
    ,:3477: 'the first letter of its complete name')."""
    t = normalise(text)
    try:
        encs = enclosures(t)
    except UnbalancedMarksError:
        encs = []
    drop = [(e.start, e.end) for e in encs
            if e.kind in ('stereo', 'added_ih', 'fusion', 'vonbaeyer', 'isotope', 'isotope_br',
                          'compound_locant', 'annulene', 'component_loc', 'ratio')]
    t = ''.join(c for i, c in enumerate(t) if not any(a <= i <= z for a, z in drop))
    t = re.sub(r"(?:(?<=^)|(?<=[-\s(\[{,]))(?:tert|sec|cis|trans|rel|rac|endo|exo|syn|anti|meso|ent|as|s|D|L|DL|"
               r"N|O|S|Se|Te|P|As|B|Si|C)(?:\d+)?'*(?=[-,])", '', t)
    t = re.sub(r"\d+[a-z]?'*H(?=[-,])", '', t)
    t = re.sub(r"λ\d+", '', t)
    t = re.sub(r"(?<![A-Za-z])\d+[a-z]?'*(?=[,\-()\[\]{}]|$)", '', t)
    return re.sub(r"[^A-Za-z]", '', t)
