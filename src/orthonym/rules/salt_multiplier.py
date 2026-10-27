"""The multiplier of an ion word that only carries locants or stereodescriptors,.

 General methodology (the Blue Book): the "simple components" ("unsubstituted parent hydrides...;
functionalized parent hydrides, such as benzenesulfonic acid; or retained names, such as acetic acid") "are multiplied by
the multiplicative prefixes 'di', 'tri', etc." (:7033), and "(c) any component which is substituted automatically requires
use of the multiplicative forms 'bis', 'tris', etc." (:7035). (:7085) puts "(e) functionalized parent hydrides
with locants" in parentheses, 'di(benzene-1-sulfonic acid)' (:7104), and the Blue Book prints a multiplied anion of that
kind as '(2R)-3-hydroxypropane-1,2-diyl di[(3E,5E)-hepta-3,5-dienoate] (PIN)' (:47993). So the anion of an unsubstituted
chain or ring with locants and stereodescriptors, 'octadec-9-enoate', 'hexa-2,4-dienoate', is multiplied by 'di',
'tri' and enclosed, 'zinc di[(9Z)-octadec-9-enoate]'; an anion with a substituent prefix, '2-ethylhexanoate',
takes 'bis' (c); 'propane-1,3-diyl bis(chloroacetate) (PIN)',:31825).
"""
from __future__ import annotations

import re

#: the leading stereodescriptors of an ion word: '(9Z)-', '(2E,4E)-', '(3E,5E)-'
_STEREO = re.compile(r"^\((?:\d+['′]?[A-Za-z](?:,|(?=\))))+\)-")
#: the parent of an unsubstituted word: an acyclic or cyclic hydrocarbon stem, or a ring parent that has no 'a' stem
_PARENT = (r"(?:cyclo)?(?:meth|eth|prop|but|pent|hex|hept|oct|non|(?:un|do|tri|tetra|penta|hexa|hepta|octa|nona)?dec"
           r"|(?:icos|(?:tri|tetra|penta|hexa|hepta|octa|nona)cont|hect))a?"
           r"|benzene|naphthalene|anthracene|phenanthrene|pyridine|thiophene|furan")
#: the parent, then the endings, suffixes and their locants: '-9-enoate', '-2,4-dienoate', '-2-ene-1-carboxylate'
_UNSUBSTITUTED = re.compile(rf"^(?:{_PARENT})(?:-[0-9,]+-[a-z]+)*[a-z]*$")


def is_unsubstituted_locanted_word(name: str) -> bool:
    """True when ``name`` is an ion word of an unsubstituted parent that cites locants (or leading stereodescriptors): no
    substituent prefix, no enclosing mark, no space. A word with no locant is not one (it takes the plain multiplier)."""
    core = _STEREO.sub("", name or "", count=1)
    return any(ch.isdigit() for ch in name or "") and bool(_UNSUBSTITUTED.match(core))
