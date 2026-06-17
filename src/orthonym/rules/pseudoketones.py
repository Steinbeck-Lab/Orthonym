"""Pseudoketone routing for acyl-on-ring-nitrogen "hidden amides" (DD1 Fix 3,
Blue Book P-66.1.3 / P-64.3.2 / P-66.1.4.3).

An acyl group on a *ring-system* nitrogen (or any skeletal heteroatom that is not
an acyclic amine N) is NOT named as an amide: the ring N is a skeletal atom of a
ring parent hydride, so the C=O carbon becomes the principal group as a KETONE
(pseudoketone) and the ring fragment is cited as an N-yl substituent on the
carbonyl carbon::

    CC(=O)N1CCCCC1   -> 1-(piperidin-1-yl)ethan-1-one      (P-66.1.3)
    CCC(=O)N1CCCCC1  -> 1-(piperidin-1-yl)propan-1-one
    CC(=S)N1CCCC1    -> 1-(pyrrolidin-1-yl)ethane-1-thione  (P-66.1.4.3)

Detection is on the original graph: the amide N must be a RING member and the
carbonyl carbon must be EXOCYCLIC to that ring (so a lactam, whose C=O is itself
in the ring, is excluded and stays a cyclic amide/one). An ordinary acyclic
amide (acetamide, N,N-dimethylacetamide, benzamide) has a non-ring amide N and is
never a pseudoketone.

Scope (returns ``None`` to fall through otherwise — never a wrong name): a single
amide group whose acyl side is a clean carbon chain (no further substituents /
heteroatoms / unsaturation on the acyl carbons). The ring fragment is named via
the existing ring-substituent namer, so substituents ON the ring are handled.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

from ..assembly.naming_utils import apply_vowel_elision
from ..data.chain_names import get_chain_prefix
from ..perception.chains import find_longest_carbon_chain
from .ring_substituents import name_ring_system_substituent

# Amide-family principal groups that can be a hidden amide (the N is bonded to
# the acyl C). A primary amide N (2 H) cannot be a ring member, so only the
# secondary/tertiary and chalcogen variants ever qualify, but we accept the whole
# family and let the ring-membership test decide.
_AMIDE_FGS = (
    "primary_amide", "secondary_amide", "tertiary_amide",
    "thioamide", "selenoamide", "telluroamide",
)

# carbonyl chalcogen symbol -> ketone (pseudoketone) suffix per P-66.1.3 / P-66.1.4.3.
_CHALCOGEN_KETONE_SUFFIX = {"O": "one", "S": "thione", "Se": "selone", "Te": "tellone"}
_CHALCOGENS = ("O", "S", "Se", "Te")


def _ring_system_atoms(mol, seed_idx: int) -> Tuple[int, ...]:
    """All atoms of the fused ring system reachable from ``seed_idx`` via ring
    bonds (a single ring for piperidine/pyrrolidine/morpholine)."""
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    system = set()
    frontier = [r for r in rings if seed_idx in r]
    seen_rings = []
    while frontier:
        r = frontier.pop()
        if r in seen_rings:
            continue
        seen_rings.append(r)
        system |= r
        for other in rings:
            if other not in seen_rings and (other & system):
                frontier.append(other)
    return tuple(sorted(system))


def is_hidden_amide(mol, amide_match) -> Optional[Tuple[int, int, str]]:
    """Return ``(carbonyl_C_idx, ring_N_idx, chalcogen_symbol)`` if ``amide_match``
    is an acyl-on-ring-N hidden amide, else ``None``.

    Criteria: the amide N is a ring member; the carbonyl carbon is bonded to that
    N and double-bonded to a chalcogen (O/S/Se/Te) and is itself NOT in a ring
    (exocyclic acyl — excludes lactams)."""
    n_idx = None
    for idx in amide_match:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == "N":
            n_idx = idx
            break
    if n_idx is None:
        return None
    n_atom = mol.GetAtomWithIdx(n_idx)
    if not n_atom.IsInRing():
        return None

    # carbonyl carbon = a carbon neighbour of N that is double-bonded to a chalcogen
    from rdkit import Chem

    for nbr in n_atom.GetNeighbors():
        if nbr.GetSymbol() != "C":
            continue
        c_idx = nbr.GetIdx()
        chalcogen = None
        for cn in nbr.GetNeighbors():
            if cn.GetSymbol() in _CHALCOGENS:
                bond = mol.GetBondBetweenAtoms(c_idx, cn.GetIdx())
                if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                    chalcogen = cn.GetSymbol()
                    break
        if chalcogen is None:
            continue
        if mol.GetAtomWithIdx(c_idx).IsInRing():
            return None  # lactam: carbonyl C in the ring -> not a pseudoketone
        return (c_idx, n_idx, chalcogen)
    return None


def name_pseudoketone(features: Any, style: str = "pin") -> Optional[str]:
    """Name an acyl-on-ring-N hidden amide as a pseudoketone, or ``None`` to fall
    through when the structure is outside the clean-acyl-chain class."""
    mol = features.mol
    pg = features.principal_group
    matches = features.principal_group_atoms
    if pg not in _AMIDE_FGS or not matches:
        return None
    if len(matches) != 1:
        return None  # multiple amide groups -> not the simple pseudoketone case

    parsed = is_hidden_amide(mol, matches[0])
    if parsed is None:
        return None
    carbonyl_c, ring_n, chalcogen = parsed
    ketone_suffix = _CHALCOGEN_KETONE_SUFFIX.get(chalcogen)
    if ketone_suffix is None:
        return None

    ring_atoms = _ring_system_atoms(mol, ring_n)
    if not ring_atoms:
        return None

    # The N-side fragment = the whole connected component on the nitrogen side
    # after cutting the N->carbonyl bond (ring system + any substituents on it).
    from collections import deque

    frag_seen = {carbonyl_c}
    frag = []
    queue = deque([ring_n])
    while queue:
        a = queue.popleft()
        if a in frag_seen:
            continue
        frag_seen.add(a)
        frag.append(a)
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            if nb.GetIdx() not in frag_seen:
                queue.append(nb.GetIdx())

    # Deterministic constitutional safety guard (no external tool): only emit when
    # the N-side fragment is the BARE ring system (no exocyclic substituents on the
    # ring). For an unsubstituted ring, name_ring_system_substituent reliably yields
    # a constitutionally-correct N-yl name (piperidin-1-yl / pyrrolidin-1-yl /
    # morpholin-4-yl). On a SUBSTITUTED symmetric ring it can drop the attachment
    # locant (e.g. '1-methylpiperazinyl', which denotes a DIFFERENT molecule), so a
    # decorated ring falls through to the legacy path rather than risk a wrong name.
    # (Substituted-ring pseudoketones are a follow-on, gated on the ring-substituent
    # namer gaining correct attachment numbering for symmetric N-heterocycles.)
    if set(frag) != set(ring_atoms):
        return None

    # Acyl parent chain: longest carbon chain that excludes the entire N-side
    # fragment (ring + its substituents); the carbonyl carbon (the ketone, locant
    # 1) must be one terminus of it.
    frag_set = set(frag)
    acyl_chain = find_longest_carbon_chain(mol, exclude_atoms=frag_set)
    if not acyl_chain or carbonyl_c not in acyl_chain:
        return None
    if acyl_chain[0] != carbonyl_c and acyl_chain[-1] != carbonyl_c:
        return None  # carbonyl C must be a chain terminus to be position 1
    chain = acyl_chain if acyl_chain[0] == carbonyl_c else list(reversed(acyl_chain))
    chain_set = set(chain)

    # Clean-acyl guard: the only non-chain heavy neighbours allowed are the
    # carbonyl chalcogen and the ring N (both on the carbonyl carbon). Any other
    # substituent / heteroatom on an acyl carbon -> fall through (not our scope).
    for c in chain:
        catom = mol.GetAtomWithIdx(c)
        for nbr in catom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni in chain_set:
                continue
            if nbr.GetSymbol() == "H":
                continue
            if c == carbonyl_c and (ni == ring_n or nbr.GetSymbol() in _CHALCOGENS):
                continue
            return None

    ringyl = name_ring_system_substituent(mol, frag, ring_n)
    if not ringyl or " " in ringyl:
        return None

    length = len(chain)
    base = f"{get_chain_prefix(length)}ane"  # 'ethane', 'propane', ...
    joined = apply_vowel_elision(base, ketone_suffix)  # 'ethanone' / 'ethanethione'
    stem_part = joined[: len(joined) - len(ketone_suffix)]  # 'ethan' / 'ethane'
    parent = f"{stem_part}-1-{ketone_suffix}"  # 'ethan-1-one' / 'ethane-1-thione'

    return f"1-({ringyl}){parent}"


__all__ = ["is_hidden_amide", "name_pseudoketone"]
