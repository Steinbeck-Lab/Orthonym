"""An ester whose carbonyl carbon is in the principal chain, beside a principal group senior to it, as 'oxo' + 'alkoxy'.

``## **** Esters cited as prefixes`` (the Blue Book): "When, in an ester..., another group is present
that has priority for citation as the principal group..., an ester group is indicated by prefixes";
'3-ethoxy-N,N,N-trimethyl-3-oxopropan-1-aminium bromide (PIN)' (:31708): the carbonyl carbon is a chain atom, its =O
the prefix 'oxo' and its alkoxy group a branch of its own. The chain-prefix generator had no prefix for the 'ester'
group, so the =O was left out and only the 'ethoxy' branch was cited (a name of another molecule, which the read-back
refused). The principal groups are those of ``rules.polyfunctional._SPLIT_DEFAULT_PGS``, where the same split is made.
"""
from typing import Any, List, Optional, Tuple


def ester_oxo_matches(features: Any, fg_name: str, matches) -> Tuple[str, list]:
    """``('ketone', [(carbonyl C, =O),...])`` for the esters to cite as 'oxo' on the principal chain, else
    ``(fg_name, matches)`` unchanged."""
    pairs = _oxo_pairs(features, fg_name, matches)
    return ("ketone", pairs) if pairs is not None else (fg_name, matches)


def _oxo_pairs(features: Any, fg_name: str, matches) -> Optional[List[tuple]]:
    if fg_name != "ester" or not matches:
        return None
    from ..rules.polyfunctional import _SPLIT_DEFAULT_PGS
    if getattr(features, "principal_group", None) not in _SPLIT_DEFAULT_PGS:
        return None
    chain = set(getattr(features, "principal_chain", None) or ())
    mol = features.mol
    pairs = []
    for match in matches:
        if not match or match[0] not in chain:
            return None
        carbon = mol.GetAtomWithIdx(match[0])
        oxo = [b.GetOtherAtomIdx(match[0]) for b in carbon.GetBonds()
               if b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(carbon).GetSymbol() == "O"]
        if len(oxo) != 1:
            return None
        pairs.append((match[0], oxo[0]))
    return pairs
