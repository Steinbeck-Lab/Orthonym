"""One alphanumerically ordered block of locanted prefixes, N locants and numbers together.

 / (f): detachable prefixes are cited in alphanumerical order of their names, whatever
their locants are; 'N,4-dimethyl-N-(3-methylphenyl)benzamide (PIN)' (the Blue Book) and
'3-chloro-N-(2-chlorophenyl)naphthalene-2-sulfonamide (PIN)' (:32881) put the italic N and the numbers
in ONE list. encloses a complex prefix, multiplies identical ones,
escalates to square brackets around an inner parenthesis.

For pieces that are all ``N`` this is ``rules.amides.format_n_substitution`` ('N-methyl',
'N,N-dimethyl', 'N-ethyl-N-methyl'); the same primitives (``is_complex_substituent``,
``_wrap_n_substituent``, ``multiplied_component``) do the enclosing and the multiplying.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional, Tuple

from ..assembly.naming_utils import (
    _wrap_n_substituent,
    alpha_sort_key,
    is_complex_substituent,
    multiplied_component,
)
from ..data.locant_text import render_locants
from ..errors import is_refusal_sentinel


def locanted_prefix_block(pieces: "List[Tuple[str, str]]") -> Optional[str]:
    """``[(name, locant),...]`` -> 'N-methyl-1-phenyl' (no trailing hyphen), or None for a refusal sentinel."""
    groups: "OrderedDict[str, List[str]]" = OrderedDict()
    for name, loc in pieces:
        if is_refusal_sentinel(name):
            return None
        groups.setdefault(name, []).append(loc)
    parts = []
    for name in sorted(groups, key=alpha_sort_key):
        locs = sorted(groups[name], key=lambda s: (s[:1].isdigit(), int(s) if s.isdigit() else 0, s))
        if is_complex_substituent(name):
            display = _wrap_n_substituent(name) if "(" in name else f"({name})"
        else:
            display = _wrap_n_substituent(name)
        token = display if len(locs) == 1 else multiplied_component(len(locs), name, display)
        parts.append(f"{render_locants(locs)}-{token}")
    return "-".join(parts)


__all__ = ["locanted_prefix_block"]
