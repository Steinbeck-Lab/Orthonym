"""The acyloxy prefix of a chain substituent whose acid side holds a ring ('{2-(benzoyloxy)ethyl}').

 "Esters cited as prefixes" (the Blue Book): an ester group R-CO-O- on the skeleton of a substituent is
the 'acyloxy' prefix ('3-(benzoyloxy)propanoic acid (PIN)',:31711; '3-[(pyridine-3-carbonyl)oxy]propanoic acid (PIN)',
:31723). The acyloxy pass of ``substituent_naming._name_polyfunctional_acyclic_substituent_impl`` runs after the ring
branches of the chain are read, so a ring inside the acyl group was taken for a ring branch of the chain and the fragment
declined. This pass consumes such a group first, as ``sulfur_acid_ester_prefixes.consume_sulfur_acid_oxy`` does for the
sulfonic esters (program 11); an acyl group without a ring is left to the later pass.
"""
from typing import Dict


def _acid_side(mol, carbonyl_c: int, ester_o: int) -> set:
    side, stack = set(), [carbonyl_c]
    while stack:
        x = stack.pop()
        if x in side or x == ester_o:
            continue
        side.add(x)
        stack.extend(nb.GetIdx() for nb in mol.GetAtomWithIdx(x).GetNeighbors())
    return side


def consume_ring_acyloxy(mol, sub_set, consumed, add_prefix) -> bool:
    """Add '(R-carbonyl)oxy' / 'benzoyloxy' with ``add_prefix(host, token)`` for each ester O between a non-ring skeleton
    carbon of the fragment and an acyl carbon whose acid side holds a ring, and mark the O and the acid side ``consumed``.
    False when one token would repeat (the caller cannot spell 'bis'), so the fragment declines."""
    from ..rules.lipids import _acyloxy_for_site
    from .naming_utils import enclose_if_compound
    seen: Dict[str, int] = {}
    for idx in sorted(sub_set):
        o = mol.GetAtomWithIdx(idx)
        if idx in consumed or o.GetSymbol() != "O" or o.GetDegree() != 2 or o.IsInRing():
            continue
        # a carboxylic acyl R-CO-: the carbonyl carbon bears the two oxygens and one carbon (a carbamate or carbonate
        # -O-CO-N/-O-CO-O is no acyloxy group of this pass)
        acyl = [n for n in o.GetNeighbors() if n.GetSymbol() == "C" and not n.IsInRing() and any(
            x.GetSymbol() == "O" and x.GetDegree() == 1
            and mol.GetBondBetweenAtoms(n.GetIdx(), x.GetIdx()).GetBondTypeAsDouble() == 2.0 for x in n.GetNeighbors())
            and sorted(x.GetSymbol() for x in n.GetNeighbors()) == ["C", "O", "O"]]
        acyl_idx = {n.GetIdx() for n in acyl}
        hosts = [n.GetIdx() for n in o.GetNeighbors() if n.GetSymbol() == "C" and n.GetIdx() not in acyl_idx]
        if len(acyl) != 1 or len(hosts) != 1 or hosts[0] in consumed or mol.GetAtomWithIdx(hosts[0]).IsInRing():
            continue
        side = _acid_side(mol, acyl[0].GetIdx(), idx)
        if (not side <= set(sub_set) or hosts[0] in side or side & consumed
                or not any(mol.GetAtomWithIdx(a).IsInRing() for a in side)):
            continue
        acyloxy = _acyloxy_for_site(mol, ("acyl", acyl[0].GetIdx(), idx))
        if not acyloxy:
            continue
        token = enclose_if_compound(acyloxy)
        consumed.add(idx)
        consumed.update(side)
        add_prefix(hosts[0], token)
        seen[token] = seen.get(token, 0) + 1
    return all(n < 2 for n in seen.values())
