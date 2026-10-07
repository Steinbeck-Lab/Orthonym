"""Spelling checks of enclosing marks, multiplying prefixes, hyphens and the order of prefixes
,,,; registered with:func:`orthonym.validation.pin_spelling.register`).

All read the name lexically (:mod:`.lexer`,:mod:`.units`); no naming code, no OPSIN.
"""
from __future__ import annotations

import re

from ..pin_spelling import SpellingFailure, register
from .lexer import UnbalancedMarksError, enclosures, normalise
from .models import P10, read_units, show
from .units import split_simple

# ------------------------------------------------------------------ enclosing marks
_CYCLE = ['(', '[', '{']


@register('P-16.5')
def enclosing_marks_check(mol, name):
    """ (:7220) "Enclosing marks must not be omitted from preferred IUPAC names"; the marks
    must pair. (:7446) nesting order ``{[({})]}``: the level of a counted mark is one
    more than the highest level among its counted descendants; the parentheses of added indicated
    hydrogen and the brackets of fusion, spiro, ring-assembly and von Baeyer descriptors are
    ignored /.2,:7465,:7469); (:7509) consecutive marks of one level
    take the next level. (:7368) "Parentheses are used to enclose terms modified by the
    numerical prefixes 'bis', 'tris', 'tetrakis', etc."."""
    s = normalise(name)
    try:
        encs = enclosures(s)
    except UnbalancedMarksError as e:
        return SpellingFailure('P-16.5', f'unbalanced enclosing marks: {e}')
    if P10.search(name):
        return None
    lev = {}

    def kids(e):
        out = []
        for c in e.children:
            if c.counted:
                out.append(c)
            else:
                out.extend(kids(c))
        return out

    def level(e):
        if id(e) in lev:
            return lev[id(e)]
        ks = kids(e)
        lv = 1 + max((level(k) for k in ks), default=0)
        if e.kind == 'subst':
            first = [k for k in ks if k.start == e.start + 1]
            if first and first[0].typ == _CYCLE[(lv - 1) % 3]:
                lv += 1
        lev[id(e)] = lv
        return lv

    for e in encs:
        if e.kind != 'subst':
            continue
        lv = level(e)
        exp = _CYCLE[(lv - 1) % 3]
        if e.typ != exp:
            return SpellingFailure('P-16.5.4', f"'{e.typ}' at nesting level {lv} (expected '{exp}') "
                                               f"around '{e.content[:50]}'")
    for m in re.finditer(r"(?:^|[-\s\])}',\d])(bis|tris|tetrakis|pentakis|hexakis)(?=[a-z])", s):
        word = re.match(r"[a-z]+", s[m.end(1):]).group(0)
        pre, rest = split_simple(word)
        if not pre or rest:
            continue      # 'trispiro', 'trisila', 'bistibinane': no multiplied substituent prefix
        return SpellingFailure('P-16.5.1.10', f"'{m.group(1)}' not followed by an enclosing mark: "
                                             f"...{s[m.start(1):m.start(1) + 20]}")
    return None


_DIVALENT = {'oxy', 'dioxy', 'peroxy', 'carbonyl', 'sulfonyl', 'sulfinyl', 'carbonothioyl', 'sulfanediyl',
             'azanediyl', 'diazenyl', 'methylene', 'phosphoryl'}


@register('P-16.5.1.1')
def compound_prefix_enclosure_check(mol, name):
    """ (:7232) parentheses around compound and complex prefixes. A run of three or more
    simple components in which a divalent component follows the first is a compound prefix
    whose leading part is not enclosed ('pentyloxycarbonyl' for '(pentyloxy)carbonyl':
    (:27667) keeps only methoxy, ethoxy, propoxy, butoxy, phenoxy and tert-butoxy as simple alkoxy
    prefixes; '(benzyloxy)carbonyl (preferred prefix)':18116); one locant in front of three
    concatenated components is a locant of the second ('2-methoxyethylsulfamoyl')."""
    if P10.search(name):
        return None
    parsed = read_units(name)
    if parsed is None:
        return None
    _s, _encs, ps = parsed
    for p in ps:
        toks = p.unit.toks
        for i, t in enumerate(toks):
            if t.kind != 'TXT':
                continue
            pre, rest = split_simple(t.text)
            comps = [x for x in pre if x[1] not in ('tert', 'sec')]
            if len(comps) < 3 or rest:
                continue
            if any(x[1] in _DIVALENT for x in comps[1:]):
                return SpellingFailure('P-16.5.1.1', f"compound component not enclosed in '{t.text}'")
            if i > 0 and toks[i - 1].kind == 'LOC' and len(toks[i - 1].locs) == 1 and comps[0][0] is None:
                return SpellingFailure('P-16.5.1.1', f"compound component not enclosed in "
                                                     f"'{toks[i - 1].text}-{t.text}'")
        for i, t in enumerate(toks):
            if t.kind != 'TXT' or i < 2 or toks[i - 1].kind != 'ENC' or toks[i - 2].kind != 'LOC':
                continue
            if not all(loc.isdigit() for loc in toks[i - 2].locs):
                continue
            pre, rest = split_simple(t.text)
            comps = [x for x in pre if x[1] not in ('tert', 'sec')]
            if len(comps) >= 2 and not rest and comps[-1][1] in _DIVALENT | {'sulfanyl', 'selanyl', 'amino'}:
                return SpellingFailure('P-16.5.1.1', f"'{toks[i - 2].text}-({toks[i - 1].text[:25]})' sits on "
                                                     f"'{comps[0][1]}' of '{t.text}', which is not enclosed")
    return None


# ------------------------------------------------------------------ alphanumerical order
def _inner_locs(raw):
    """The locants inside a prefix in order of appearance (stereodescriptors removed)."""
    t = re.sub(r"\([^()]*?(?:R|S|E|Z)\)-?", '', normalise(raw))
    out = [int(m.group(1)) for m in re.finditer(r"(?<![A-Za-z\d])(\d+)(?=[,\-])", t)]
    return out or None


@register('P-14.5')
def alphanumerical_order_check(mol, name):
    """ (:3448) simple prefixes are arranged alphabetically, multiplying prefixes do not
    alter the order; (:3477) a substituent prefix begins with the first letter of its
    complete name; (:3517) with identical letters, the lower locants at the first point
    of difference come first; (:16872) hydro prefixes are cited after the alphabetized
    prefixes."""
    if P10.search(name):
        return None
    parsed = read_units(name)
    if parsed is None:
        return None
    _s, _encs, ps = parsed
    for p in ps:
        keys = [pr for pr in p.prefixes if not pr.nested]
        for a, b in zip(keys, keys[1:]):
            ka, kb = a.key.lower(), b.key.lower()
            if not ka or not kb:
                continue
            n = min(len(ka), len(kb))
            if ka[:n] > kb[:n]:
                return SpellingFailure('P-14.5', f"'{a.raw}' cited before '{b.raw}' in '{show(p.unit.text)[:60]}'")
            if ka == kb and a.kind == b.kind == 'enclosed':
                ia = re.findall(r"(?:^|[-(\[{])(as|s|sec|tert|cis|trans|N|O|S)-", normalise(a.raw))
                ib = re.findall(r"(?:^|[-(\[{])(as|s|sec|tert|cis|trans|N|O|S)-", normalise(b.raw))
                if ia != ib:
                    continue        #: italic letters decide before locants
                la, lb = _inner_locs(a.raw), _inner_locs(b.raw)
                if la is not None and lb is not None and len(la) == len(lb) and la != lb and la > lb:
                    return SpellingFailure('P-14.5.4', f"'{a.raw}' cited before '{b.raw}': same letters, "
                                                       f"higher locants first")
        if p.order_after_hydro:
            return SpellingFailure('P-31.2.1', f"detachable prefix '{p.order_after_hydro[0].raw}' cited "
                                               f"after a hydro prefix")
    return None


# ------------------------------------------------------------------ 'bis' for substituted prefixes
@register('P-16.3.5')
def multiplying_prefix_check(mol, name):
    """ (a) (:7104) "The numerical prefixes 'bis', 'tris', 'tetrakis', etc. are used to
    indicate a multiplicity of: (a) compound or complex (i.e. substituted) prefixes", e.g.
    'bis(2-chloropropan-2-yl)'; (c) (:7035) "any component which is substituted
    automatically requires use of the multiplicative forms 'bis', 'tris', etc."; hydro prefixes
    count ('bis(4,5-dihydrothiophen-2-yl)di(methyl)germane (PIN)':38232). 'di', 'tri',... before
    an enclosure whose unit carries a detachable prefix fails; 'di([4-2H]benzoyl)' (an isotope
    descriptor) and 'di(1H-imidazol-1-yl)' (indicated hydrogen) are simple."""
    if P10.search(name):
        return None
    parsed = read_units(name)
    if parsed is None:
        return None
    s, _encs, ps = parsed
    for m in re.finditer(r"(?:^|[-\s(\[{,'\d])(di|tri|tetra|penta|hexa)(?=[(\[{])", s):
        start = m.end(1)
        enc = next((p for p in ps if p.unit.encl is not None and p.unit.encl.start == start), None)
        if enc is None or 'multiplicative' in enc.notes:
            continue
        if enc.prefixes or enc.hydro:
            return SpellingFailure('P-16.3.5', f"'{m.group(1)}' before the substituted component "
                                               f"'{show(enc.unit.text)[:50]}' (needs 'bis', 'tris', ...)")
    return None


@register('P-16.3.4')
def bis_before_simple_component_check(mol, name):
    """ (f) (:7104) parentheses with 'di', 'tri',... enclose "simple components
    containing brackets": 'di(bicyclo[3.2.1]octan-3-yl) (preferred prefix)', '3,5-di([1,1'-
    biphenyl]-3-yl)pyridine (PIN)' (:23913); 'bis', 'tris',... are for substituted components
     (a)) and the ambiguities of (:7134). 'bis' before an unsubstituted
    component whose brackets are a ring-assembly, von Baeyer or spiro descriptor fails, unless
    the component carries skeletal replacement ('a') prefixes (c)), a fusion
    descriptor (d)) or begins with a multiplying prefix (e))."""
    if P10.search(name):
        return None
    res = read_units(name)
    if res is None:
        return None
    s, _encs, ps = res
    for m in re.finditer(r"(?:^|[-\s(\[{,'\d])(bis|tris|tetrakis)(?=[(\[{])", s):
        start = m.end(1)
        enc = next((p for p in ps if p.unit.encl is not None and p.unit.encl.start == start), None)
        if enc is None or 'multiplicative' in enc.notes or enc.prefixes or enc.hydro or enc.repl:
            continue
        kinds = {c.kind for c in enc.unit.encl.children}
        if not kinds & {'assembly', 'vonbaeyer', 'spiro'} or kinds & {'fusion', 'component_loc', 'annulene'}:
            continue
        if re.match(r"(?:\d+[a-z]?'*(?:,\d+[a-z]?'*)*-)?(?:di|tri|tetra|bi)(?!cyclo|spiro)", enc.unit.encl.content):
            continue
        return SpellingFailure('P-16.3.4', f"'{m.group(1)}' before the simple component "
                                           f"'{enc.unit.encl.content[:50]}' (needs 'di', 'tri', ...)")
    return None


#: parents with one substitutable position or a mononuclear parent hydride /.2,
#::7272,:7304): every substituent prefix but the first is enclosed
_ONE_POSITION_PARENTS = re.compile(
    r"^(?:methane|silane|germane|stannane|plumbane|borane|phosphane|arsane|stibane|bismuthane|"
    r"methanol|methanethiol|methanone|methanethione|silanol|silanecarboxylic|acetonitrile|acetic|"
    r"phosphinic)$")
_CARRIER = re.compile(r"(?:yl|ylidene|oxy|carbonyl|sulfonyl|sulfinyl)$")


@register('P-16.5.1.1')
def one_position_parent_enclosure_check(mol, name):
    """On a mononuclear parent or a parent with one substitutable position, the leading run of
    letters of the name splits into two or more substituent prefixes and ends in a group that
    carries the earlier ones ('methyldiselanyl(methylsulfanyl)methane' for the PIN
    '(methyldiselanyl)(methylsulfanyl)methane',:18278): a compound prefix without its
    parentheses,:7232) or a second substituent without them,:7272,
    '(chloromethyl)(methyl)silane (PIN)', 'chloro(methyl)silane (PIN)'). A run of terminal
    prefixes ('bromodichlorofluoromethane') is not read: the book prints both
    'bromo(chloro)fluoromethane (PIN)' (:44645) and the form."""
    if P10.search(name):
        return None
    res = read_units(name)
    if res is None:
        return None
    _s, _encs, ps = res
    for p in ps:
        if p.unit.encl is not None or not _ONE_POSITION_PARENTS.match(p.parent or ''):
            continue
        if any(t.kind in ('LOC', 'IH', 'ASM', 'OTHER') for t in p.unit.toks):
            continue
        first = next((t for t in p.unit.toks if t.kind == 'TXT'), None)
        if first is None:
            continue
        pre, _rest = split_simple(first.text)
        comps = [x for x in pre if x[1] not in ('tert', 'sec')]
        if len(comps) >= 2 and _CARRIER.search(comps[-1][1]) and first.text != p.parent:
            return SpellingFailure('P-16.5.1.1', f"'{first.text}' on '{p.parent}': a compound or second "
                                                 f"substituent prefix without parentheses")
    return None


# ------------------------------------------------------------------ (b) hyphen after an enclosing mark
_CLOSE_THEN_LOCANT = re.compile(r"[)\]}](?=\d)")


@register('P-16.2.4.1')
def hyphen_after_enclosure_check(mol, name):
    """ (b) (:6944): hyphens are used "after parentheses, if the closing parenthesis is
    followed by a locant" ('1-(chloromethyl)-4-nitrobenzene (PIN)'). A closing mark directly
    followed by a digit has lost it ('N-{...}1H-indazole-3-carboxamide'). Fusion and von Baeyer
    brackets are followed by letters, superscripts by ']'."""
    s = normalise(name)
    m = _CLOSE_THEN_LOCANT.search(s)
    if m:
        return SpellingFailure('P-16.2.4.1', f"no hyphen between '{s[m.start()]}' and the locant: "
                                             f"...{s[max(0, m.start() - 15):m.start() + 6]}")
    return None
