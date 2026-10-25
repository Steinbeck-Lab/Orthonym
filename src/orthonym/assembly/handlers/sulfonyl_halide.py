"""W3-P11 sulfonyl/sulfinyl halide handler — direct-return shim.

Wrapper around ``rules.sulfur.name_sulfonyl_halide`` (mirrors the
acid_halide shim discipline). The acid halide of a sulfonic / sulfinic
acid is a two-word functional-class name '{stem}sulfonyl {halide}' /
'{stem}sulfinyl {halide}'.

IUPAC cite: / / (acyl halides of
sulfonic/sulfinic acids; BB 'ethanesulfonyl chloride',
'propane-1-sulfonyl chloride', '4-isocyanatobenzene-1-sulfonyl chloride
(PIN)'); (the cyanides of the sulfonic, sulfinic,
selenonic, seleninic, telluronic and tellurinic acids; BB 'methanesulfonyl
cyanide (PIN)', 'benzeneseleninyl cyanide (PIN)').
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ...data.oxoacid_cyanides import OXOACID_CYANIDE_GROUPS
from ..name_tree import NameTreeNode, NamingResult

logger = logging.getLogger(__name__)


def _is_sulfonyl_halide(features: Any) -> bool:
    """True iff the principal group is a sulfonyl/sulfinyl acid halide, or the
    cyanide of a sulfonic, sulfinic, selenonic, seleninic, telluronic or
    tellurinic acid (v42 P4-3, P-66.5.1.3.2 — same cap-and-rename body)."""
    return getattr(features, 'principal_group', None) in (
        ('sulfonyl_halide', 'sulfinyl_halide') + OXOACID_CYANIDE_GROUPS
    )


def _cite_chalcogen_descriptor(features: Any, name: str) -> str:
    """The R / S descriptor of a stereogenic E (sulfinyl, seleninyl, tellurinyl) of an acyl cyanide,
    cited at the front of the name when E is the one stereogenic unit of the molecule.

    The name is that of the acid-derived acyl group with the class word 'cyanide': E is a centre of the
    acid group of the parent, not of a 'sulfinyl' prefix of a sulfoxide, so its descriptor stands at the
    front of the acid-derived name even where the name cites locants, "The chirality symbols
    'R' and 'S'" (the Blue Book, 'ethyl (R)-4-nitrobenzene-1-sulfinate (PIN)';:46266, '(S)-
    (methanesulfinyl)benzene (PIN)' is the sulfoxide, a prefix). ``_inject_stereo_if_missing`` cannot
    place it where the name has a numbered parent (its ring and chain scopes disagree) and the
    universal backstop (``namer._final_stereo_check``) fronts it but labels it below the PIN whenever
    the name cites a locant, because it reads the centre as a sulfoxide. The CIP label is the engine's
    (``features.stereocenters``); with another stereogenic unit in the molecule the name is left as it
    is and the read-back gate declines it."""
    group = getattr(features, 'principal_group', None)
    if group not in OXOACID_CYANIDE_GROUPS or getattr(features, 'double_bond_stereo', None):
        return name
    centres = getattr(features, 'stereocenters', None) or []
    matches = features.functional_groups.get(group, [])
    if len(centres) != 1 or len(matches) != 1 or centres[0].get('idx') != matches[0][0]:
        return name
    cip = centres[0].get('cip')
    if cip not in ('R', 'S') or name.startswith(('(R)-', '(S)-')):
        return name
    return f"({cip})-{name}"


def name_sulfonyl_halide(
    features: Any, mol: Any = None, style: str = "pin",
) -> Optional[NamingResult]:
    """Direct-return sulfonyl/sulfinyl-halide handler (cap-and-rename body in
    rules.sulfur.name_sulfonyl_halide). Lazy import keeps the
    handlers -> rules.sulfur -> namer chain off the import-time graph."""
    from ...rules.sulfur import name_sulfonyl_halide as _name_sulfonyl_halide
    from ..candidate_pool import get_current_pool
    from ..composer import _inject_stereo_if_missing

    halide_name = _name_sulfonyl_halide(features, style=style)
    if not halide_name:
        return None

    pool = get_current_pool()
    pool.add(halide_name, "sulfonyl_halide", features)
    final_name = _inject_stereo_if_missing(
        features, pool.best().name, atom_to_locant=None,
    )
    final_name = _cite_chalcogen_descriptor(features, final_name)
    return NamingResult(
        name=final_name,
        tree=NameTreeNode(
            parent_stem=final_name, class_id="sulfonyl_halide",
            iupac_section_cite="P-67.1.4.4.1", fragment_legacy=final_name,
        ),
        atom_to_locant_hint=None,
    )


__all__ = ["name_sulfonyl_halide", "_is_sulfonyl_halide"]
