"""Spelling checks of where a name places a stereodescriptor and an enclosing mark.

 "NAMING OF STEREOISOMERS" (the Blue Book): "In preferred IUPAC names, stereodescriptors are
placed immediately at the front of the part of the name to which they relate. They are placed at the
front of the complete name when related to the parent structure; they are cited in parentheses
followed by a hyphen. When they relate to substituent groups, they are cited at the front of the
corresponding prefix. They are preceded by a numerical or letter locant to describe the position of
the stereogenic unit when such locants are present."

A descriptor with a numerical locant that the parent of its own unit has no position for does not
relate to that parent: '(1R,4S)-2-(4-methoxyphenyl)-N-{...}ethan-1-amine' puts the '4S' of a
substituent ring before an ethane that has two positions. The check reads the unit's parent from
the name alone (an unbranched carbon chain, a carbocyclic, retained or Hantzsch-Widman monocycle,
naphthalene, anthracene::func:`.models.ring_or_chain_model`) and abstains on any other parent, so
a reader that does not know a parent can only pass it. It does not decide which prefix a descriptor
belongs to, only that it is not the unit's own.

A mononuclear parent of a heteroatom (borane, silane, phosphane, the boronic and phosphonic acids and
the like) has no numbered position at all, so a descriptor with a numerical locant at the front of its
unit relates to a substituent whatever the parent: '(1E,4S)-(4-hydroxypent-1-en-1-yl)boronic acid' is
'[(1E,4S)-4-hydroxypent-1-en-1-yl]boronic acid' (:func:`descriptor_locant_at_a_parent_without_positions`).

 "Parentheses" (the Blue Book): "Parentheses are used around compound and complex
prefixes". A prefix cited with an N-locant that begins with a numerical locant is such a prefix:
'N-(2-methoxyethyl)', never 'N-2-methoxyethyl' (the reading of that spelling is 'N-2' of a
methoxyethyl: ``2-methoxyethyl`` has its own locant). No naming code, no OPSIN.
"""
from __future__ import annotations

import re

from ..pin_spelling import ABSTAIN, PASS, SpellingFailure, register
from .models import P10, read_units, ring_or_chain_model, show

#: the numerical locant of one descriptor of a set: '4S' of '(1R,4S)', '4aR', '2E'
_NUMERIC_LOCANT = re.compile(r"(\d+)[a-z]?(?=[RSEZrsez]|\Z)")


@register('P-91.3')
def descriptor_locants_fit_their_unit(mol, name):
    """A stereodescriptor set at the front of a unit whose parent has fewer positions than a locant
    of the set,:44643): the descriptor relates to a substituent, and is cited in the
    unit of that prefix."""
    if P10.search(name):
        return None
    parsed = read_units(name)
    if parsed is None:
        return None
    _s, _encs, ps = parsed
    for p in ps:
        if not p.stereo:
            continue
        model = ring_or_chain_model(p, p.unit.encl is not None)
        if model is None:
            continue
        positions = model[1]
        for token in p.stereo:
            for item in token.split(','):
                m = _NUMERIC_LOCANT.match(item.strip())
                if m and int(m.group(1)) > positions:
                    return SpellingFailure(
                        'P-91.3', f"descriptor '{item.strip()}' at the front of '{show(p.parent or '')}', "
                                  f"a parent of {positions} position(s): it relates to a substituent")
    return None


#: the parent tokens of the mononuclear parent hydrides of Groups 13-15 and of their acids, anions and
#: esters: the names 'borane', 'silane', 'phosphane'... /, the mononuclear parent hydrides,
#: and / the oxoacids of boron, phosphorus, arsenic and antimony with the endings
#: 'onic', 'inic', 'onous', 'inous' and their 'ate' and 'ite' forms). A parent of one skeletal atom has no
#: position to cite. 'disilane', 'diphosphane' (two atoms, positions 1 and 2) are not in the set.
_NO_POSITION_PARENT = re.compile(
    r"(?:(?:bor|silan|german|stannan|plumban|phosphan|arsan|stiban|bismuthan|aluman|gallan|indigan|thallan)e"
    r"|(?:bor|phosph|ars|stib)(?:on|in)(?:ic|ous|ate|ite))")


@register('P-91.3')
def descriptor_locant_at_a_parent_without_positions(mol, name):
    """A stereodescriptor with a numerical locant set at the front of a unit whose parent is a mononuclear
    heteroatom parent,:44643): "stereodescriptors are placed immediately at the front of the part
    of the name to which they relate", and a parent of one atom has no position for a numerical locant, so
    the descriptor relates to a substituent and is cited at the front of that prefix. PASS when every unit
    with a locanted descriptor has a parent whose positions are known (a mononuclear parent, or one
    :func:`descriptor_locants_fit_their_unit` counts); ABSTAIN on another parent, on a name of the
     classes and on a name whose marks do not pair."""
    if P10.search(name):
        return ABSTAIN
    parsed = read_units(name)
    if parsed is None:
        return ABSTAIN
    _s, _encs, ps = parsed
    decided = True
    for p in ps:
        items = [item.strip() for token in p.stereo for item in token.split(',')]
        locanted = [item for item in items if _NUMERIC_LOCANT.match(item)]
        if not locanted:
            continue
        if _NO_POSITION_PARENT.fullmatch(p.parent or ''):
            return SpellingFailure(
                'P-91.3', f"descriptor '{locanted[0]}' at the front of '{show(p.parent)}', a parent with no "
                          f"numbered position: it relates to a substituent")
        if ring_or_chain_model(p, p.unit.encl is not None) is None:
            decided = False
    return PASS if decided else ABSTAIN


#: one item of a stereodescriptor set: a locant (numerical, lettered, primed or N) and a descriptor, as
#: ``decomposition.fragment_assembly._FRONT_STEREO_RE`` reads the block at the front of a name
_ITEM = r"(?:(?:[0-9]+[a-z]*'*|N'*)?(?:[RSEZrsMP]\*?|RS|SR))"
#: a set written straight after a letter, a digit or a closing mark, or after the hyphen of a prefix, i.e.
#: between a prefix and the rest of the name: 'N-methyl(1E,2E)-but-2-en-1-imine'. A set at the front of the
#: name, after a space (an ester or salt word), after an opening mark (the front of an enclosed prefix) or after
#: 'rel-' / 'rac-' does not count.
_BLOCK_AFTER_PREFIX = re.compile(rf"(?<=[a-z0-9\)\]\}}])(-?)\(({_ITEM}(?:,{_ITEM})*)\)-")


@register('P-91.3')
def descriptor_block_after_a_prefix(mol, name):
    """A stereodescriptor set between a prefix and the part it opens,:44643): the parent's
    descriptors are "placed at the front of the complete name", a substituent's "at the front of the
    corresponding prefix", so a set never follows a prefix: '(1E,2E)-N-methylbut-2-en-1-imine', not
    'N-methyl(1E,2E)-but-2-en-1-imine'. A sulfur isotope label of the structure ('3-methyl(34S)-2H-thiopyran',
     has the same spelling as a set '(34S)' and is not one."""
    if P10.search(name):
        return None
    sulfur_masses = ({str(a.GetIsotope()) for a in mol.GetAtoms() if a.GetSymbol() == 'S' and a.GetIsotope()}
                     if mol is not None else set())
    for m in _BLOCK_AFTER_PREFIX.finditer(name):
        if m.group(1) and name[:m.start()].endswith(('rel', 'rac')):
            continue
        items = m.group(2).split(',')
        if all(i.endswith('S') and i[:-1] in sulfur_masses for i in items):
            continue
        return SpellingFailure('P-91.3', f"descriptor set '({m.group(2)})' after the prefix "
                                         f"'{show(name[:m.start()][-30:])}': it belongs at the front")
    return None


#: an italic N locant (or a set 'N,N', "N,N'") joined by a hyphen to a prefix that begins with its own
#: numerical locant and a letter: 'N-2-methoxyethyl', 'N,N-1-phenylethyl'. 'N,2-dimethyl' (a numerical
#: locant in the set), 'N-1H-indol-3-yl' (an indicated hydrogen) and the isotope descriptor 'N-2H1' do not match.
_UNENCLOSED_N_PREFIX = re.compile(r"(?:(?<=[-\s(\[{])|^)N'*(?:,N'*)*-\d+[a-z]?(?:,\d+[a-z]?)*-[a-z]")


@register('P-16.5.1.1')
def n_prefix_with_a_locant_is_enclosed(mol, name):
    """A prefix of an N-locant that begins with a numerical locant is a compound prefix and takes its
    enclosing marks,:7232)."""
    if P10.search(name):
        return None
    m = _UNENCLOSED_N_PREFIX.search(name)
    if m:
        return SpellingFailure('P-16.5.1.1', f"'{m.group(0)}...' cites a prefix with its own locant after an "
                                             f"N-locant without enclosing marks")
    return None
