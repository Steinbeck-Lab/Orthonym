"""Spelling check of the parent of a secondary or tertiary amine (registered with
:func:`orthonym.validation.pin_spelling.register`).

A round trip proves the molecule, never the spelling: OPSIN reads 'N-phenyl-1-(phenylsulfanyl)butan-2-amine'
back to the structure of the preferred 'N-[1-(phenylsulfanyl)butan-2-yl]aniline'.

* "Selection of the principal chain or senior ring system in secondary and tertiary
  amines" (the Blue Book): "When a choice for parent hydride is possible between a ring and
  a chain, the ring is preferred." Its examples (:26288,:26292): 'N-cyclohexylaniline (PIN)',
  'N-butylcyclopropanamine (PIN)' (not 'N-cyclopropylbutan-1-amine'). An amine whose nitrogen is
  bonded to a ring atom and to a chain carbon is named on the ring, the chain a prefix of the
  nitrogen; a name that takes the chain as the parent is the general spelling.

The structure side is the one acyclic amine nitrogen of the molecule: neutral, three single bonds
to carbon or hydrogen, no carbon of a carbonyl, thiocarbonyl or imine group next to it (an amide
is another class), bonded to a ring atom and to an acyclic carbon. The name side is a word whose
parent is an acyclic chain with the suffix 'amine' (a chain stem, with or without 'an'/'en'/'yn').
A molecule with a second amine nitrogen is not judged: the suffix might be that nitrogen's, and
 (the maximum number of principal characteristic groups) decides before the ring does.
"""
from __future__ import annotations

import re
from typing import Optional

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register
from .checks_prefixes import _belongs_to_a_ring_name
from .lexer import normalise
from .units import STEM_RE

#: an amine nitrogen: acyclic, neutral, three single bonds, every neighbour a carbon (or no
#: neighbour but hydrogens), none of them the carbon of a C=O, C=S or C=N group
_AMINE_N = Chem.MolFromSmarts(
    "[NX3;!R;+0;!$(N=*);!$(N-[!#6;!#1]);!$(N-[#6]=[O,S,N])]")
#: the end of a name whose parent is an acyclic chain with the suffix 'amine': a chain stem, its
#: saturation or unsaturation ('an', 'prop-2-en', 'hex-5-yn', 'penta-1,3-dien'), the suffix locants
#: and the suffix ('-2-amine', '-1,3-diamine')
_CHAIN_AMINE_END = re.compile(
    rf"(?P<stem>{STEM_RE})(?:an(?:e)?|a?-\d+(?:,\d+)*-(?:di|tri|tetra)?(?:en|yn)(?:e)?"
    rf"(?:-\d+(?:,\d+)*-(?:en|yn)(?:e)?)?)?-?(?:\d+(?:,\d+)*-)?(?:di|tri|tetra)?amine$")


def _bridging_amine(mol) -> bool:
    """True when the molecule has exactly one amine nitrogen and it is bonded to a ring atom
    and to an acyclic carbon."""
    nitrogens = [m[0] for m in mol.GetSubstructMatches(_AMINE_N)]
    if len(nitrogens) != 1:
        return False
    carbons = [a for a in mol.GetAtomWithIdx(nitrogens[0]).GetNeighbors()]
    return (any(a.IsInRing() for a in carbons)
            and any(not a.IsInRing() for a in carbons))


@register("P-62.2.2.2")
def amine_ring_parent_check(mol, name) -> Optional[SpellingFailure]:
    """ (:26247): the ring is the parent of an amine whose nitrogen is bonded to a ring
    and to a chain ('N-[1-(phenylsulfanyl)butan-2-yl]aniline', not 'N-phenyl-1-(phenylsulfanyl)butan-2-amine')."""
    if not _bridging_amine(mol):
        return None
    text = normalise(name)
    end = _CHAIN_AMINE_END.search(text)
    # the stem ends the name, not a ring name or a replacement name ('cyclobutanamine', '2-oxabutan-1-amine')
    if end is None or _belongs_to_a_ring_name(text, end.start()):
        return None
    return SpellingFailure(
        "P-62.2.2.2",
        f"the chain '{end.group('stem')}' is the parent of an amine whose nitrogen is also bonded to a "
        f"ring; the ring is preferred as the parent")
