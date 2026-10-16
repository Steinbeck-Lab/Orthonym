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

 "Parentheses" (the Blue Book): "Parentheses are used around compound and complex
prefixes". A prefix cited with an N-locant that begins with a numerical locant is such a prefix:
'N-(2-methoxyethyl)', never 'N-2-methoxyethyl' (the reading of that spelling is 'N-2' of a
methoxyethyl: ``2-methoxyethyl`` has its own locant). No naming code, no OPSIN.
"""
from __future__ import annotations

import re

from ..pin_spelling import SpellingFailure, register
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
