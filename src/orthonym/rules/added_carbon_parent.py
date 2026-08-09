"""Added-carbon multi-suffix parent naming (DD1 Fix 2, Blue Book P-65.1.1.1 /
P-66.1.1.1.1.2 / P-66.5.1.1.2).

When **three or more** carboxylic-acid / carboxamide / carbonitrile groups are
present on an acyclic skeleton they cannot all be expressed as chain-terminal
``-oic acid`` / ``-amide`` / ``-nitrile`` suffixes (a chain has only two ends),
so the PIN names the **parent hydride formed by removing the carbonyl/nitrile
carbons** and appends a multiplied ``carbo*`` *added-carbon* suffix, citing one
locant per attachment atom::

    NC(=O)C(C(=O)N)C(=O)N      -> methanetricarboxamide
    CCCC(C#N)(C#N)C#N          -> butane-1,1,1-tricarbonitrile
    OC(=O)CC(C(=O)O)CC(=O)O    -> propane-1,2,3-tricarboxylic acid

Why ``n >= 3`` and NOT geminal-2: two such groups can ALWAYS be routed as the two
termini of a single chain (``HOOC-CH2-COOH`` -> propanedioic acid, the carbonyl
carbons + the bridging carbon form a 3-atom chain), so the chain-suffix form wins
for n<=2 (P-65.1.1.2). Only at n>=3 does the chain run out of ends, forcing the
added-carbon form that expresses the maximum number of groups as the suffix
(P-65.1.1.1: maximum number of skeletal/principal groups expressed as suffix).

The namer is deliberately SCOPED to a clean acyclic carbon parent (every parent
carbon's only non-chain heavy neighbours are the added carbons). A decorated
parent (extra substituents, heteroatoms, unsaturation, rings) returns ``None`` so
the caller falls through to the legacy path — never a wrong name. Ring-attached
added-carbon suffixes (``cyclohexane-1,2-dicarboxylic acid``) are already handled
by the ring suffix path and are intentionally excluded here.
"""
from __future__ import annotations

from typing import Any, List, Optional

from ..assembly.naming_utils import SIMPLE_MULTIPLIERS
from ..data.chain_names import get_chain_prefix
from ..perception.chains import find_longest_carbon_chain
from .seniority import SUFFIX_FORMS

# Principal-group FG names that take an added-carbon ``carbo*`` suffix form and
# whose match tuple carries the carbonyl/nitrile carbon at index 0. The value is
# the added-carbon (ring/multi) suffix string from SUFFIX_FORMS[name][1].
_ADDED_CARBON_FGS = ("carboxylic_acid", "primary_amide", "nitrile")


def _added_carbon_suffix(fg_name: str) -> Optional[str]:
    forms = SUFFIX_FORMS.get(fg_name)
    if not forms:
        return None
    return forms[1]  # the ring/added-carbon ('carbo*') form


def requires_added_carbon_suffix(mol, fg_name: Optional[str], pg_matches) -> bool:
    """Cheap structural gate: principal group is acid/amide/nitrile and there are
    >= 3 distinct instances. The rigorous acyclic-clean-parent validation lives in
    :func:`name_added_carbon_parent`, which returns ``None`` to fall through.
    """
    if fg_name not in _ADDED_CARBON_FGS:
        return False
    if not pg_matches:
        return False
    added = {m[0] for m in pg_matches if m}
    return len(added) >= 3


def _added_carbons_and_attachments(mol, pg_matches):
    """Return (sorted added-carbon idxs, {added_carbon -> skeleton-attachment C})
    or ``None`` if any added carbon is not a carbon bonded to exactly one acyclic
    non-added skeleton carbon (the cases this namer does not own)."""
    added = sorted({m[0] for m in pg_matches if m})
    added_set = set(added)
    attach = {}
    for ac in added:
        atom = mol.GetAtomWithIdx(ac)
        if atom.GetSymbol() != "C":
            return None
        carbon_nbrs = [
            n.GetIdx()
            for n in atom.GetNeighbors()
            if n.GetSymbol() == "C" and n.GetIdx() not in added_set
        ]
        if len(carbon_nbrs) != 1:
            return None  # added C bonded to 0 or >1 skeleton carbons -> not our case
        sk = carbon_nbrs[0]
        if mol.GetAtomWithIdx(sk).IsInRing():
            return None  # ring-attached -> ring suffix path owns it
        attach[ac] = sk
    return added, attach


def name_added_carbon_parent(features: Any, style: str = "pin") -> Optional[str]:
    """Name an acyclic >=3-group added-carbon multi-suffix molecule, or ``None``
    to fall through to the legacy path when the structure is outside the clean
    acyclic-parent class this namer owns."""
    mol = features.mol
    fg_name = features.principal_group
    pg_matches = features.principal_group_atoms
    carbo = _added_carbon_suffix(fg_name)
    if carbo is None or not pg_matches:
        return None

    parsed = _added_carbons_and_attachments(mol, pg_matches)
    if parsed is None:
        return None
    added, attach = parsed
    n_groups = len(added)
    if n_groups < 3:
        return None
    added_set = set(added)

    # Parent hydride = the carbon skeleton with the added carbons removed.
    chain = find_longest_carbon_chain(mol, exclude_atoms=added_set)
    if not chain:
        return None

    # All non-added carbons must lie ON this single chain (no branches). Every
    # parent carbon's heavy neighbour must be a parent/added carbon OR a nameable
    # exocyclic SUBSTITUENT (v30: citric-acid family -- a 2-hydroxy on the core;
    # previously ANY extra neighbour returned None -> the molecule fell to a wrong
    # pentanedioic-chain candidate). Collect each substituent fragment and name it
    # via the recursive substituent namer; fail closed on anything un-nameable so a
    # wrong/atom-dropped name is never emitted.
    from rdkit import Chem

    skeleton_carbons = {
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetSymbol() == "C" and a.GetIdx() not in added_set
    }
    chain_set = set(chain)
    if skeleton_carbons != chain_set:
        return None
    # substituent = (chain_atom_idx, attach_neighbour_idx, frozenset(frag_atoms))
    substituents: List[tuple] = []
    for c in chain:
        catom = mol.GetAtomWithIdx(c)
        if catom.IsInRing():
            return None
        for nbr in catom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni in chain_set or ni in added_set or nbr.GetSymbol() == "H":
                continue
            # BFS the substituent fragment from ni, never crossing back into the
            # parent chain or the added (suffix) carbons.
            frag = set()
            stack = [ni]
            while stack:
                a = stack.pop()
                if a in frag or a in chain_set or a in added_set:
                    continue
                frag.add(a)
                for nn in mol.GetAtomWithIdx(a).GetNeighbors():
                    j = nn.GetIdx()
                    if j not in frag and j not in chain_set and j not in added_set:
                        stack.append(j)
            substituents.append((c, ni, frozenset(frag)))

    # Reject any non-single bond inside the parent chain (unsaturated parents are
    # out of this clean-saturated-skeleton scope; aconitic-family needs a separate
    # unsaturated extension).
    for i in range(len(chain) - 1):
        bond = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return None

    length = len(chain)
    stem = get_chain_prefix(length)
    parent = f"{stem}ane"
    multiplier = SIMPLE_MULTIPLIERS.get(n_groups)
    if multiplier is None:
        return None

    # Name every substituent (fail closed if any is un-nameable) BEFORE choosing
    # the numbering, so a molecule this namer cannot fully express is declined here
    # rather than emitting a partial (atom-dropping) name.
    named_subs: List[tuple] = []  # (chain_atom_idx, bare_prefix_name)
    if substituents:
        from ..assembly.substituent_enumerator import name_substituent
        for chain_atom, ni, frag in substituents:
            sub_name = name_substituent(mol, sorted(frag), ni)
            if not sub_name or sub_name == "substituent":
                return None  # un-nameable substituent -> fail closed (0-wrong)
            named_subs.append((chain_atom, sub_name))

    # P-31.1.4 / P-14.4 numbering: lowest locants to the added-carbon (suffix)
    # attachments FIRST, then to the substituents. Choose the direction minimising
    # (sorted suffix locants, then sorted substituent locants).
    pos = {atom_idx: i for i, atom_idx in enumerate(chain)}

    def loc(atom_idx: int, reverse: bool) -> int:
        p = pos[atom_idx]
        return (length - 1 - p) + 1 if reverse else p + 1

    def key_for(reverse: bool):
        suf = sorted(loc(attach[ac], reverse) for ac in added)
        sub = sorted(loc(ca, reverse) for ca, _ in named_subs)
        return (suf, sub)

    reverse = key_for(True) < key_for(False)
    suffix_locants = sorted(loc(attach[ac], reverse) for ac in added)

    # Assemble the substituent-prefix string (P-16.3.3 enclosure, P-14.5.2 alpha
    # order, P-16.3.4 multipliers) via the shared naming utilities.
    prefix_str = ""
    if named_subs:
        from collections import defaultdict
        from ..assembly.naming_utils import (
            alpha_sort_key, is_complex_substituent, get_multiplier_prefix,
        )
        by_name: dict = defaultdict(list)
        for ca, nm in named_subs:
            by_name[nm].append(loc(ca, reverse))
        parts = []
        for nm in sorted(by_name, key=alpha_sort_key):
            locs = sorted(by_name[nm])
            complex_ = is_complex_substituent(nm)
            disp = f"({nm})" if complex_ and not (nm.startswith("(") and nm.endswith(")")) else nm
            mult = get_multiplier_prefix(len(locs), complex_)
            locstr = ",".join(str(x) for x in locs)
            parts.append(f"{locstr}-{mult}{disp}")
        # P-16.3.4: hyphen-join the ordered prefix fragments.
        prefix_str = "-".join(parts) if len(parts) > 1 else parts[0]
        # a trailing enclosure/letter meets a digit of the parent locant run below;
        # the parent stem starts with a letter, so no hyphen needed after prefix_str.

    if length == 1:
        return f"{prefix_str}{parent}{multiplier}{carbo}"
    locant_str = ",".join(str(x) for x in suffix_locants)
    return f"{prefix_str}{parent}-{locant_str}-{multiplier}{carbo}"


__all__ = ["requires_added_carbon_suffix", "name_added_carbon_parent"]
