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

    # All non-added carbons must lie ON this single chain (no branches), and every
    # parent carbon's heavy neighbours must be parent carbons or added carbons
    # only (no other substituents / heteroatoms / unsaturation we cannot express).
    skeleton_carbons = {
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetSymbol() == "C" and a.GetIdx() not in added_set
    }
    chain_set = set(chain)
    if skeleton_carbons != chain_set:
        return None
    for c in chain:
        catom = mol.GetAtomWithIdx(c)
        if catom.IsInRing():
            return None
        for nbr in catom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni in chain_set or ni in added_set:
                continue
            if nbr.GetSymbol() == "H":
                continue
            return None  # an extra substituent/heteroatom we do not own here
    # Reject any non-single bond inside the parent chain (unsaturated parents are
    # out of this clean-saturated-skeleton scope).
    from rdkit import Chem

    for i in range(len(chain) - 1):
        bond = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return None

    length = len(chain)

    # Lowest locants to the added-carbon attachment points (P-31.1.4): choose the
    # numbering direction giving the lexicographically smallest sorted locant set.
    pos = {atom_idx: i for i, atom_idx in enumerate(chain)}

    def locants_for(reverse: bool) -> List[int]:
        out = []
        for ac in added:
            p = pos[attach[ac]]
            out.append((length - 1 - p) + 1 if reverse else p + 1)
        return sorted(out)

    fwd = locants_for(reverse=False)
    rev = locants_for(reverse=True)
    locants = min(fwd, rev)

    stem = get_chain_prefix(length)
    parent = f"{stem}ane"
    multiplier = SIMPLE_MULTIPLIERS.get(n_groups)
    if multiplier is None:
        return None

    # methane (single-carbon parent): no locants possible/required.
    if length == 1:
        return f"{parent}{multiplier}{carbo}"
    locant_str = ",".join(str(loc) for loc in locants)
    return f"{parent}-{locant_str}-{multiplier}{carbo}"


__all__ = ["requires_added_carbon_suffix", "name_added_carbon_parent"]
