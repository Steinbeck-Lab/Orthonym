"""An ester whose carbonyl carbon is in the principal chain, beside a principal group senior to it, as 'oxo' + 'alkoxy'.

``## **** Esters cited as prefixes`` (the Blue Book): "When, in an ester..., another group is present
that has priority for citation as the principal group..., an ester group is indicated by prefixes";
'3-ethoxy-N,N,N-trimethyl-3-oxopropan-1-aminium bromide (PIN)' (:31708): the carbonyl carbon is a chain atom, its =O
the prefix 'oxo' and its alkoxy group a branch of its own. The chain-prefix generator had no prefix for the 'ester'
group, so the =O was left out and only the 'ethoxy' branch was cited (a name of another molecule, which the read-back
refused). The principal groups are those of ``rules.polyfunctional._SPLIT_DEFAULT_PGS``, where the same split is made.

An anhydride beside a senior acid is cited the same way, its carbonyl in the chain as 'oxo' and its other acyl group
as 'acyloxy': (:32382) 'bis(acetyloxy)borinic acid (PIN; acid is senior to anhydride)', and the acid words
of method (1) (:32394), '4-oxo-4-(propanoyloxy)butanoic' (:32402).

An amide on a terminal atom of the chain too: ``## **** Substituents of the types -CO-NH2 and -CO-CO-NH2``
(:32920), "(1) by using the two prefixes 'amino' and 'oxo' to denote such groups on terminal atoms of carbon chains
having more than one carbon atom... method (1) is preferred for chains and method (2) ['carbamoyl'] for rings and ring
systems, heterogeneous chains, and on nonterminal atoms of carbon chains. Carbamoyl and amino groups may be
substituted in the normal way"; '5-(2-amino-2-oxoethyl)furan-2-carboxylic acid (PIN)' (:32940).
"""
# the N-substituted amides: a primary amide is cited as 'amino' + 'oxo' by its own prefix form already; a hydroxamic
# acid is an N-hydroxy amide
_AMIDES = ("secondary_amide", "tertiary_amide", "hydroxamic_acid")
from typing import Any, List, Optional, Tuple


def ester_oxo_matches(features: Any, fg_name: str, matches) -> Tuple[str, list]:
    """``('ketone', [(carbonyl C, =O),...])`` for the esters to cite as 'oxo' on the principal chain, else
    ``(fg_name, matches)`` unchanged."""
    pairs = _oxo_pairs(features, fg_name, matches)
    return ("ketone", pairs) if pairs is not None else (fg_name, matches)


def _oxo_pairs(features: Any, fg_name: str, matches) -> Optional[List[tuple]]:
    if fg_name not in ("ester", "anhydride") + _AMIDES or not matches:
        return None
    from ..rules.polyfunctional import _SPLIT_DEFAULT_PGS
    if getattr(features, "principal_group", None) not in _SPLIT_DEFAULT_PGS:
        return None
    order = list(getattr(features, "principal_chain", None) or ())
    chain = set(order)
    mol = features.mol
    pairs = []
    for match in matches:
        # an anhydride match is (carbonyl C, =O, bridge O, carbonyl C, =O): one of its carbonyls is a chain atom
        inside = [c for c in ((match[0], match[3]) if fg_name == "anhydride" and len(match) == 5 else match[:1])
                  if c in chain]
        if not inside and fg_name in _AMIDES:
            continue                    # an amide of a branch: the branch's own prefix names it
        if len(inside) != 1 or (fg_name in _AMIDES and (len(order) < 2 or inside[0] not in (order[0], order[-1]))):
            return None                 # an amide on a nonterminal chain atom is 'carbamoyl'
        carbon = mol.GetAtomWithIdx(inside[0])
        oxo = [b.GetOtherAtomIdx(inside[0]) for b in carbon.GetBonds()
               if b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(carbon).GetSymbol() == "O"]
        if len(oxo) != 1:
            return None
        pairs.append((inside[0], oxo[0]))
    return pairs or None
