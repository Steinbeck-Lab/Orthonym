"""Label-only guard for the benzene parent choice of ``benzene._preferred_benzene_parent_ring``.

 "the maximum number of substituents cited as prefixes" (the Blue Book), then the lower locant
set (:21698), the lower locants in order of citation (:21791) and the earlier name in alphanumerical
order (:22234; '1-bromo-3-[(3-chlorophenyl)methyl]benzene (PIN) [not 1-[(3-bromophenyl)methyl]-3-chlorobenzene]',:6279).
The candidates of ``_benzene_parent_candidates`` are pre-filtered by a 'carbon-linked' tier, a determinism device with
no rule behind it: of two rings that tie on, the ring on the CH2 of '(phenoxymethyl)benzene' is kept and the
O-linked ring is never compared, though '(benzyloxy)benzene' is earlier. Here every ring that ties on is named
the same way and the criteria above are applied to the names; when the shipped choice loses, its name is kept below
the PIN label (the name itself does not change). A tied ring that cannot be named, a name with a suffix, or an
isotopically modified molecule decides before leaves the label as it was.
"""
from __future__ import annotations

import re

_LOCANTS = re.compile(r"(\d+(?:,\d+)*)-")


def _prefix_locants(name: str):
    """The locants of the prefixes in order of citation (the depth-0 locant sets in front of 'benzene'), or None."""
    if not name.endswith('benzene'):
        return None
    depth, out, i = 0, [], 0
    while i < len(name):
        ch = name[i]
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        elif depth == 0 and ch.isdigit() and (i == 0 or not name[i - 1].isalnum()):
            m = _LOCANTS.match(name, i)
            if m:
                out += [int(x) for x in m.group(1).split(',')]
                i = m.end()
                continue
        i += 1
    return tuple(out)


def _ring_name(mol, ring_atoms, principal_group):
    """The name of ``mol`` on the benzene ring ``ring_atoms``, built as ``_preferred_benzene_parent_ring`` builds it."""
    from .benzene import (get_benzene_substituents, name_substituted_benzene, orient_benzene,
                          principal_group_ring_atoms)
    substituents = get_benzene_substituents(mol, ring_atoms)
    if not substituents:
        return "benzene"
    positions = principal_group_ring_atoms(mol, ring_atoms, substituents, principal_group=principal_group)
    oriented = orient_benzene(mol, ring_atoms, substituents, principal_group_positions=positions or None)
    return name_substituted_benzene(mol, ring_atoms, oriented, substituents, assembly_label=False)


def _key(name: str, count: int, alnum_key):
    locs = _prefix_locants(name)
    if locs is None or len(locs) not in (0, count):
        return None
    if not locs:
        if count != 1:
            return None
        locs = (1,)
    return (tuple(sorted(locs)), locs, alnum_key(name))


def chosen_loses_the_tie(mol, scored, chosen, is_benzene_ring, alnum_key):
    """The rule ('', '' or '') by which another benzene ring that ties on gives the
    preferred name, else None."""
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        return None
    rings = [r for r in mol.GetRingInfo().AtomRings() if is_benzene_ring(mol, r)]
    if len(rings) < 2:
        return None

    def _count(ring):
        ring_set = set(ring)
        return sum(1 for a in ring for nbr in mol.GetAtomWithIdx(a).GetNeighbors()
                   if nbr.GetIdx() not in ring_set and nbr.GetAtomicNum() > 1)
    most = max(_count(r) for r in rings)
    tied = [r for r in rings if _count(r) == most]
    if len(tied) < 2:
        return None
    names = {frozenset(r): nm for r, nm in scored}
    missing = [r for r in tied if frozenset(r) not in names]
    if not missing:
        return None                     # every tied ring was compared by the selector
    from .benzene import _molecule_principal_group
    from ..metrics.provenance import isolated_provenance
    from ..validation.coverage_gate import _full_inchikey_roundtrips
    principal = _molecule_principal_group(mol)
    for r in missing:
        with isolated_provenance():        # a rival's records never reach the name that ships
            nm = _ring_name(mol, r, principal)
        # a rival name must describe the whole molecule (a name that drops atoms decides nothing)
        names[frozenset(r)] = nm if nm and _full_inchikey_roundtrips(mol, nm) else None
    keys = {}
    for r in tied:
        nm = names.get(frozenset(r))
        k = _key(nm, most, alnum_key) if nm else None
        if k is None:
            return None                 # a tied ring without a comparable name: no answer, no label change
        keys[frozenset(r)] = k
    best = min(keys.values())
    mine = keys.get(frozenset(chosen[0]))
    if mine is None or mine == best or list(keys.values()).count(best) != 1:
        return None
    return 'P-45.2.2' if mine[0] != best[0] else 'P-45.2.3' if mine[1] != best[1] else 'P-45.5'


def note_unresolved_benzene_tie(mol, scored, chosen, is_benzene_ring, alnum_key=None):
    """Record the chosen name below the PIN label when it loses the tie (``chosen_loses_the_tie``); returns
    ``chosen`` unchanged."""
    if chosen and alnum_key is not None:
        try:
            rule = chosen_loses_the_tie(mol, scored, chosen, is_benzene_ring, alnum_key)
        except Exception:  # noqa: BLE001 -- no answer: no label change
            rule = None
        if rule:
            from ..metrics.provenance import record_non_pin_label
            record_non_pin_label(chosen[1], rule=rule,
                                 detail="another benzene ring that ties on the number of prefixes gives the preferred name")
    return chosen
