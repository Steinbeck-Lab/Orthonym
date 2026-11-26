"""A multiplied, enclosed substituent prefix in front of a core makes a substituted prefix (a)).

'2,4,6-tri(propan-2-yl)phenyl', '3,5-bis(trifluoromethyl)phenyl' and 'di(propan-2-yl)amino' each carry a
detachable substituent prefix on their core, whatever multiplies it, so a multiple of them takes the derived
multiplier: 'bis[2,4,6-tri(propan-2-yl)phenyl]', not 'di[2,4,6-tri(propan-2-yl)phenyl]'.
(the Blue Book): "For compound or complex substituent groups (see and, the multiplicative
prefixes 'bis', 'tris', 'tetrakis-', etc. are used as described in (a)". 'propan-2-yl' and
'di(propan-2-yl)' alone have nothing after the enclosed part and stay simple ('di(propan-2-yl)', (a)).
Read by:func:`orthonym.assembly.naming_utils.is_substituted_substituent`.
"""
from __future__ import annotations


def multiplied_enclosed_leader(body: str) -> bool:
    """``body`` (lower case, an optional leading locant set) starts with a multiplying syllable and an enclosed
    substituent prefix followed by more name ('tri(propan-2-yl)phenyl', 'di(propan-2-yl)amino')."""
    from .naming_utils import (_LEADING_LOCANT_RE, _LEADING_MULTIPLIER_SYLLABLES,
                               _enclosed_leader_is_a_substituent)
    m = _LEADING_LOCANT_RE.match(body)
    if m:
        body = body[m.end():]
    for mult in _LEADING_MULTIPLIER_SYLLABLES:
        rest = body[len(mult):]
        if body.startswith(mult) and rest[:1] in ('(', '[') and _enclosed_leader_is_a_substituent(rest):
            return True
    return False
