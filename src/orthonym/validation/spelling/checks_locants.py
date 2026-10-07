"""Spelling checks of locants and numbering,,; registered with
:func:`orthonym.validation.pin_spelling.register`).

All read the name lexically against the numbering models of:mod:`.models`; no naming code, no
OPSIN.
"""
from __future__ import annotations

import re

from ..pin_spelling import SpellingFailure, register
from .models import LABELS, P10, UNSAT, read_units, ring_or_chain_model, show, vector
from .units import MULT, STEM_RE, STEMS, split_simple


@register('P-14.4')
def numbering_check(mol, name):
    """ (:3221) low locants in the order (a) heteroatoms, (b) indicated hydrogen, (c)
    principal characteristic groups and free valences, (d) added hydrogen, (e) hydro/ene/yne
    together then double bonds, (f) detachable prefixes together, (g) the prefix cited first. For a
    unit whose parent's numbering symmetries are known, the name fails when another numbering is
    lower at the first point of difference.

    The book ranks an element locant against a numeral both ways: italic letters are lower
    ,:3195,:3215 "N,α,1,2 is lower than 1,2,4,6"), but the PIN
    'N4,2-dimethylpentane-2,4-diamine' (:26371) is lowest only with numerals first. The check
    fails a name only when a lower numbering exists under both rankings."""
    if P10.search(name):
        return None
    parsed = read_units(name)
    if parsed is None:
        return None
    _s, _encs, ps = parsed
    for p in ps:
        mdl = ring_or_chain_model(p, p.unit.encl is not None)
        if mdl is None:
            continue
        kind, n, perms, feats = mdl
        ident = perms[0] if kind == 'fused' else {i: i for i in range(1, n + 1)}
        first = {}
        for letters_first in (True, False):
            v0 = vector(feats, ident, n, kind, letters_first)
            for mp in perms:
                if mp != ident:
                    v = vector(feats, mp, n, kind, letters_first)
                    if v < v0:
                        first[letters_first] = (v0, v)
                        break
            if not feats['letter']:
                first[False] = first.get(True)
                break
        if first.get(True) and first.get(False):
            v0, v = first[True]
            labels = LABELS + [f'prefix {k}' for k, _ in feats['prefix']]
            for lab, a, b in zip(labels, v0, v):
                if a != b:
                    return SpellingFailure('P-14.4', f"{kind} '{p.parent}' in '{show(p.unit.text)[:60]}': "
                                                     f"{lab} {a} > {b}")
    return None


# ------------------------------------------------------------------ omission of locants
_SYM_MONO = {'benzene', 'pyrazine', '1,3,5-triazine', '1,2,4,5-tetrazine', 'coronene'}


def _subst_count(p):
    c = sum(max(len(pr.locs), pr.mult) for pr in p.prefixes)
    for locs, txt, _added in p.endings:
        if txt and not UNSAT.match(txt):
            c += max(1, len(locs))
    return c


@register('P-14.3.4')
def locant_omission_check(mol, name):  # noqa: C901 -- one branch per subrule
    """ (:2877) no terminal locants for acyclic mono- and dicarboxylic acids and their
    derivatives; (:2891) locant '1' omitted in substituted mononuclear parents and
    monosubstituted two-atom chains and homogeneous monocycles; (:2939) monosubstituted
    symmetric parents; (:3007) "All locants are omitted in compounds or substituent
    groups in which all substitutable positions are completely substituted"; (:3031)
    acetic acid and parents whose substitutable hydrogens share one locant."""
    if P10.search(name):
        return None
    parsed = read_units(name)
    if parsed is None:
        return None
    _s, encs, ps = parsed
    if any(e.kind in ('isotope', 'isotope_br') for e in encs):
        return None
    for p in ps:
        par = p.parent
        if not par or 'multiplicative' in p.notes or 'isotope_or_other' in p.notes:
            continue
        allocs = [loc for pr in p.prefixes for loc in pr.locs] + [loc for locs, _, _ in p.endings for loc in locs] + \
                 [loc for locs, _ in p.hydro for loc in locs]
        nsub = _subst_count(p)
        unsat = any(UNSAT.match(t or '') for _, t, _ in p.endings)
        homo_mono = re.fullmatch(r"cyclo(?:" + STEM_RE + r")(?:an|ane)?|benzen|benzene|ethan|ethane|methan|methane|"
                                 r"silan|silane", par)
        # imines are left out, as in the engine's own (c) licence (no verbatim book row)
        if homo_mono and nsub == 1 and allocs == ['1'] and not unsat and not p.hydro and not p.ih \
                and not any((t or '').endswith('imine') for _, t, _ in p.endings):
            return SpellingFailure('P-14.3.4.2', f"locant '1' cited on monosubstituted '{par}' in "
                                                 f"'{show(p.unit.text)[:60]}'")
        # with a stereodescriptor on the unit the book prints both forms: '(R)-{bis[...]amino}{...}acetic
        # acid (PIN)' (:45731) and '(2R)-2-chloro-2-{4-[...]phenyl}acetic acid (PIN)' (:46729)
        if re.fullmatch(r"acetic|acetate|acetyl|acetonitrile|acetaldehyde", par) and p.prefixes and \
                any(pr.locs for pr in p.prefixes) and not p.stereo and \
                not any(loc.startswith(('N', 'O')) for pr in p.prefixes for loc in pr.locs):
            return SpellingFailure('P-14.3.4.6', f"locants cited on '{par}' prefixes in '{show(p.unit.text)[:60]}'")
        mch = re.fullmatch(r"(" + STEM_RE + r")(yl|ane)", par)
        if mch and len(p.prefixes) == 1 and not p.endings and p.prefixes[0].locs:
            n_c = STEMS[mch.group(1)]
            cap = 2 * n_c + (1 if mch.group(2) == 'yl' else 2)
            if len(p.prefixes[0].locs) == cap and p.prefixes[0].kind == 'simple':
                return SpellingFailure('P-14.3.4.5', f"all {cap} positions of '{par}' substituted alike, "
                                                     f"locants cited")
        if par in ('phenyl', 'benzene') and len(p.prefixes) == 1 and p.prefixes[0].locs and \
                p.prefixes[0].kind == 'simple' and len(p.prefixes[0].locs) == (5 if par == 'phenyl' else 6):
            return SpellingFailure('P-14.3.4.5', f"all positions of '{par}' substituted alike, locants cited")
        for locs, txt, _added in p.endings:
            if not p.repl and re.fullmatch(r"(?:" + STEM_RE + r")an(?:e)?", par) and \
                    re.fullmatch(r"(?:di)?(?:oic|oate|al|nitrile|amide|oyl|oyl chloride|ohydrazide)", txt or ''):
                n_c = STEMS[re.fullmatch(r"(" + STEM_RE + r")ane?", par).group(1)]
                # (:2869) all or none: another ending with a locant keeps the terminal one
                others = [l2 for l2, t2, _ in p.endings if (l2, t2) != (locs, txt) and l2]
                if set(locs) <= {'1', str(n_c)} and not others:
                    return SpellingFailure('P-14.3.4.1', f"terminal locants cited for "
                                                         f"'{par}-{','.join(locs)}-{txt}'")
        pl = (','.join(p.parent_locs) + '-' if p.parent_locs else '') + par
        if re.sub(r'e$', '', pl) in {re.sub(r'e$', '', x) for x in _SYM_MONO} and nsub == 1 and allocs \
                and not p.ih and not p.hydro and not p.prefixes and par not in ('benzene', 'benzen') \
                and p.unit.encl is None and not any(t in ('yl', 'ylidene', 'diyl') for _, t, _ in p.endings):
            return SpellingFailure('P-14.3.4.3', f"locant cited on monosubstituted symmetric '{pl}'")
    return None


# ------------------------------------------------------------------ all locants of a unit or none
_RING_SUFFIX_UNLOCANTED = re.compile(
    r"^(?:cyclo(?:" + STEM_RE + r")an(?:e)?|benzen(?:e)?)"
    r"(ol|thiol|selenol|peroxol|amine|imine|one|thione|carbonitrile|carboxylic|carbaldehyde|carboxamide|"
    r"carbothioamide|carbothioic|carbonyl|carbohydrazide|carboximidamide|carboxylate|sulfonic|sulfinic|"
    r"sulfonamide|sulfonyl|sulfinyl|sulfonate|thiolate|olate)$")
_CHAIN_SUFFIX_UNLOCANTED = re.compile(
    r"^(?:" + '|'.join(k for k in sorted(STEMS, key=len, reverse=True) if k != 'meth') + r")an(?:e)?"
    r"(ol|thiol|selenol|peroxol|amine|imine|one|thione)$")


#: simple prefixes multiplied by 'di', 'tri',... (b),:7067): halo, alkyl, and the
#: common characteristic-group prefixes
_MULTIPLIABLE = re.compile(r"fluoro|chloro|bromo|iodo|nitro|nitroso|amino|hydroxy|oxo|imino|cyano|"
                           r"methoxy|ethoxy|propoxy|butoxy|phenoxy|phenyl|methylidene|sulfanylidene|"
                           r"carboxy|formyl|acetyl|(?:" + STEM_RE + r")yl")


def _numeric(locs):
    return any(re.match(r"\d", loc) for loc in locs)


@register('P-14.3.3')
def locant_citation_check(mol, name):
    """ (:2869) "if any locants are essential for defining the structure of the parent
    structure or of a unit of structure as defined by its appropriate enclosing marks, then all
    locants must be cited for the parent structure or that structural unit. For example, the
    omission of the locant '1' in 2-chloroethanol... is not allowed in preferred IUPAC names, thus
    the name 2-chloroethan-1-ol is the PIN". A unit that cites a numeric locant for a prefix or a
    hydro prefix but writes its suffix without one, on a carbon chain (other than the mononuclear
    methane and the acids and their derivatives) or a carbocyclic monocycle; the book
    writes 'cyclohexane-1-carboxylic acid', 'benzene-1-sulfonic acid', 'ethan-1-ol' beside other
    locants (:35009,:31174,:28156). And a multiplied simple prefix whose locants do not match its
    multiplying prefix ('2-dichloro', '4-dimethylamino')."""
    if P10.search(name):
        return None
    parsed = read_units(name)
    if parsed is None:
        return None
    _s, encs, ps = parsed
    if any(e.kind in ('isotope', 'isotope_br') for e in encs) or re.search(r"[A-Z]'* \d|[,\-^] ", name):
        return None     # isotopes; a locant set split by a space ("N' 3", "N1, N^ 3": text extraction)
    for p in ps:
        if 'multiplicative' in p.notes or 'isotope_or_other' in p.notes:
            continue
        toks = p.unit.toks
        for i, t in enumerate(toks[1:], 1):
            # a multiplied simple prefix ('di', 'tri'...) after a locant set of another size;
            # a divalent prefix of a multiplicative name ('1,1'-oxydibenzene') has no multiplier,
            # and a prefix with an inherent multiplying syllable ('trioxidanyl') is not read
            if t.kind != 'TXT' or toks[i - 1].kind != 'LOC' or toks[i - 1].enc:
                continue
            pre, _rest = split_simple(t.text)
            if pre and pre[0][0] and _MULTIPLIABLE.fullmatch(pre[0][1]) \
                    and MULT[pre[0][0]] != len(toks[i - 1].locs):
                return SpellingFailure('P-14.3.3', f"{len(toks[i - 1].locs)} locant(s) {toks[i - 1].text} for "
                                                   f"'{pre[0][0]}{pre[0][1]}' in '{show(p.unit.text)[:60]}'")
        par = p.parent or ''
        if p.endings or p.repl or not (_RING_SUFFIX_UNLOCANTED.match(par) or _CHAIN_SUFFIX_UNLOCANTED.match(par)):
            continue
        cited = any(_numeric(pr.locs) for pr in p.prefixes) or any(_numeric(locs) for locs, _ in p.hydro)
        if cited:
            return SpellingFailure('P-14.3.3', f"locants are cited in '{show(p.unit.text)[:60]}' but the "
                                               f"suffix of '{par}' is not")
    return None
