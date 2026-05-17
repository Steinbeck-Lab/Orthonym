"""Phase 160.1 substituent prefix-form table for non-principal FG-bearing substituents.

Per IUPAC 2013 §P-65 + §P-66, a NON-PRINCIPAL functional group MUST be
expressed via its IUPAC-defined prefix form when it appears as a substituent:

    -C(=O)OCH3   ->  methoxycarbonyl    (P-65.6.3)
    -C(=O)NH2    ->  carbamoyl          (P-66.6.1)
    -C(=O)NHCH3  ->  N-methylcarbamoyl  (P-66.6.1)
    -NHC(=O)OCH3 ->  (methoxycarbonyl)amino  (P-66.6.4)
    -OCH3        ->  methoxy            (P-63.2.5)
    -S(=O)CH3    ->  methylsulfinyl     (P-63.6)
    ... (14 rows total — see 160.1-AUDIT-SUBENUM.md §1)

This module is the SHARED chemistry-rule table consulted by:

  1. ``rules/polyfunctional.py`` (principal-chain context; existing caller;
     backwards-compat preserved via re-export shim per Phase 160.1 CONTEXT D-03).
  2. ``assembly/substituent_enumerator.py:name_substituent`` Tier-0.5 hook
     (sub-fragment context; NEW caller per Phase 160.1 CONTEXT D-04).

5 generators lifted verbatim from ``rules/polyfunctional.py`` per CONTEXT D-03;
3 NEW generators will be added in Plan-02-02 per RESEARCH §6 (secondary/tertiary
amide carbamoyl forms + carbamate orientation-checking form);
1 dispatcher ``get_substituent_prefix_form(fg_name, mol, atoms, principal_chain)``
mirrors ``rules/polyfunctional.get_fg_prefix_form`` per CONTEXT D-03 + RESEARCH §6.

All functions are PURE: read-only on (mol, atoms, principal_chain);
no side effects; no pool.add(); no MolecularFeatures mutation. Per CONTEXT
D-25 inheritance from Phase 160 + Phase 158 D-26.

None-guard for principal_chain: per RESEARCH §4, every lifted function adds
``chain_set = set(principal_chain) if principal_chain else set()`` at entry,
allowing the Tier-0.5 caller to pass principal_chain=None for sub-fragment
context (the substituent has no principal-chain).
"""
from __future__ import annotations

import threading

# Phase 160.2 Plan-04-02 WR-04 closure: removed unused ``import logging``
# and ``logger = logging.getLogger(__name__)`` — no ``logger.*`` call sites
# in 1041 LOC (verified via grep). Maintainers who want to add debug
# instrumentation should re-add the import + module-local logger then.
from collections import deque
from typing import List, Optional, Set

from .naming_utils import ALKYL_NAMES, alpha_sort_key, get_alkyl_name
from ..rules.seniority import PREFIX_FORMS


# Chain prefixes for ether alkoxy naming (match ALKYL_NAMES pattern).
# Lifted from rules/polyfunctional.py:52-63 verbatim.
ALKOXY_NAMES = {
    1: "methoxy",
    2: "ethoxy",
    3: "propoxy",
    4: "butoxy",
    5: "pentyloxy",
    6: "hexyloxy",
    7: "heptyloxy",
    8: "octyloxy",
    9: "nonyloxy",
    10: "decyloxy",
}


def _count_fragment_atoms(
    mol,
    start_atom: int,
    exclude: Set[int],
    carbons_only: bool = False,
) -> int:
    """Count atoms in a fragment via BFS.

    Lifted from rules/polyfunctional.py:600-642 verbatim. Helper for the 5
    lifted generators below — kept as a private module-local function so the
    new module is self-contained and the lift target has no dependency on
    rules/polyfunctional.py (avoids a circular import when polyfunctional.py
    re-exports from this module).

    Args:
        mol: RDKit Mol object.
        start_atom: Starting atom index.
        exclude: Atoms to not cross (boundary).
        carbons_only: If True, only count carbon atoms.

    Returns:
        Number of atoms (or carbons if carbons_only=True).
    """
    visited: Set[int] = set()
    queue = deque([start_atom])
    count = 0

    while queue:
        atom_idx = queue.popleft()
        if atom_idx in visited or atom_idx in exclude:
            continue
        visited.add(atom_idx)

        atom = mol.GetAtomWithIdx(atom_idx)
        if carbons_only:
            if atom.GetSymbol() == "C":
                count += 1
        else:
            count += 1

        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in exclude:
                queue.append(nbr_idx)

    return count


def get_alkoxy_prefix(
    mol,
    ether_atoms: tuple,
    principal_chain: Optional[List[int]] = None,
) -> Optional[str]:
    """Determine the alkoxy prefix for an ether (IUPAC P-63.1 / P-63.2.5).

    The ether SMARTS ``[OX2]([CX4])[CX4]`` matches both carbons.
    We need to determine which side is the substituent (smaller/not in chain)
    and name it as alkoxy.

    Lifted from rules/polyfunctional._get_alkoxy_prefix verbatim with a NEW
    None-guard at function entry per Phase 160.1 RESEARCH §4 lift-blocking
    analysis: when ``principal_chain`` is None (the Tier-0.5 sub-fragment
    caller passes None), ``chain_set`` becomes empty and the orientation
    guard falls back cleanly to the smaller-fragment selection.

    Args:
        mol: RDKit Mol object.
        ether_atoms: Atom indices from ether SMARTS match (O, C, C).
        principal_chain: Atom indices of the principal chain; may be None for
            the Tier-0.5 sub-fragment caller per Phase 160.1 CONTEXT D-04.

    Returns:
        Alkoxy prefix (e.g., ``"methoxy"``, ``"ethoxy"``), or None if naming
        fails. Never returns the literal string ``"alkoxy"`` (not valid IUPAC).
    """
    chain_set: Set[int] = set(principal_chain) if principal_chain else set()

    # ether_atoms from SMARTS "[OX2]([CX4])[CX4]" = (O, C1, C2)
    if len(ether_atoms) < 3:
        return None  # Defensive guard: let caller skip this ether

    oxygen_idx = ether_atoms[0]
    carbon1_idx = ether_atoms[1]
    carbon2_idx = ether_atoms[2]

    # Determine which carbon is the substituent (not in principal chain)
    # or the smaller fragment if both/neither in chain
    carbon1_in_chain = carbon1_idx in chain_set
    carbon2_in_chain = carbon2_idx in chain_set

    if carbon1_in_chain and not carbon2_in_chain:
        # carbon2 is the substituent
        sub_carbon = carbon2_idx
    elif carbon2_in_chain and not carbon1_in_chain:
        # carbon1 is the substituent
        sub_carbon = carbon1_idx
    else:
        # Neither or both in chain - use smaller fragment
        # Count atoms on each side via BFS
        frag1_size = _count_fragment_atoms(mol, carbon1_idx, {oxygen_idx})
        frag2_size = _count_fragment_atoms(mol, carbon2_idx, {oxygen_idx})
        sub_carbon = carbon1_idx if frag1_size <= frag2_size else carbon2_idx

    # Check if the substituent is aromatic (phenoxy, benzyloxy)
    sub_atom = mol.GetAtomWithIdx(sub_carbon)

    # Case A: O -> aromatic C in 6-membered all-carbon ring -> "phenoxy"
    if sub_atom.GetIsAromatic():
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if sub_carbon in ring and len(ring) == 6:
                if all(
                    mol.GetAtomWithIdx(r).GetIsAromatic()
                    and mol.GetAtomWithIdx(r).GetSymbol() == "C"
                    for r in ring
                ):
                    return "phenoxy"
        # Fallback for other aromatic ethers
        return "phenoxy"

    # Case B: O -> CH2 -> aromatic ring -> "benzyloxy"
    if (
        not sub_atom.GetIsAromatic()
        and sub_atom.GetSymbol() == "C"
        and sub_atom.GetTotalNumHs() >= 1
    ):
        arom_nbrs = [
            n
            for n in sub_atom.GetNeighbors()
            if n.GetIdx() != oxygen_idx and n.GetIsAromatic()
        ]
        non_h_non_arom = [
            n
            for n in sub_atom.GetNeighbors()
            if n.GetIdx() != oxygen_idx
            and not n.GetIsAromatic()
            and n.GetSymbol() != "H"
        ]
        if arom_nbrs and not non_h_non_arom:
            return "benzyloxy"

    # Count carbons in the substituent fragment
    carbon_count = _count_fragment_atoms(
        mol, sub_carbon, {oxygen_idx}, carbons_only=True
    )

    # Get alkoxy name
    if carbon_count in ALKOXY_NAMES:
        return ALKOXY_NAMES[carbon_count]

    # For larger groups, build name from alkyl
    try:
        alkyl = get_alkyl_name(carbon_count)
        # Remove 'yl' and add 'yloxy' for larger alkoxy groups
        if alkyl.endswith("yl"):
            return alkyl[:-2] + "yloxy"
        return alkyl + "oxy"
    except (ValueError, KeyError):
        # get_alkyl_name failed; try get_chain_prefix for arbitrary counts
        try:
            from ..data.chain_names import get_chain_prefix

            prefix = get_chain_prefix(carbon_count)
            return f"{prefix}yloxy"
        except (ValueError, KeyError):
            return None  # Not "alkoxy" -- let caller handle absence


def get_alkoxycarbonyl_prefix(
    mol,
    ester_atoms: tuple,
    principal_chain: Optional[List[int]] = None,
) -> Optional[str]:
    """Generate alkoxycarbonyl prefix for ester-as-non-principal-group (IUPAC P-65.6.3).

    Per IUPAC P-65.6.3, when an ester group ``-C(=O)-O-R`` is not the principal
    characteristic group, it is expressed as an alkoxycarbonyl prefix::

        -COOCH3   -> methoxycarbonyl
        -COOC2H5  -> ethoxycarbonyl
        -COOPh    -> phenoxycarbonyl

    Lifted from rules/polyfunctional._get_alkoxycarbonyl_prefix verbatim with a
    NEW None-guard at function entry per Phase 160.1 RESEARCH §4: when
    ``principal_chain`` is None (Tier-0.5 sub-fragment caller), ``chain_set``
    becomes empty and the orientation guard auto-fails to the SMARTS-based
    alkyl-side discrimination.

    Args:
        mol: RDKit Mol object.
        ester_atoms: Tuple from ester SMARTS ``[CX3](=O)[OX2][#6]``:
                     (carbonyl_C, carbonyl_O, ester_O, alkyl_C).
        principal_chain: Atom indices of the principal chain; may be None for
            the Tier-0.5 sub-fragment caller per Phase 160.1 CONTEXT D-04.

    Returns:
        Alkoxycarbonyl prefix string, or None if this ester should not
        be named as alkoxycarbonyl (e.g., lactones, backbone esters,
        heteroatom-bearing alkyl, oversized alkyl fragments).
    """
    from ..rules.esters import parse_ester_fragments, is_lactone

    chain_set: Set[int] = set(principal_chain) if principal_chain else set()

    # Guard 1: lactones are named differently, not as alkoxycarbonyl
    if is_lactone(mol, ester_atoms):
        return None

    # Guard 2: orientation check — alkoxycarbonyl only applies when the
    # carbonyl C is on or bonded to the principal chain, meaning the ester
    # extends as -C(=O)-O-R away from the chain.  When the ester O is on
    # the chain instead (chain-O-C(=O)-R), the ester is an acyloxy
    # substituent, handled by _check_for_acyloxy() in composer.py.
    carbonyl_c = ester_atoms[0]
    ester_o = ester_atoms[2] if len(ester_atoms) > 2 else None
    if chain_set and ester_o is not None:
        c_on_chain = carbonyl_c in chain_set
        o_on_chain = ester_o in chain_set
        c_adj_chain = c_on_chain or any(
            nbr.GetIdx() in chain_set
            for nbr in mol.GetAtomWithIdx(carbonyl_c).GetNeighbors()
            if nbr.GetIdx() != ester_atoms[1]  # exclude carbonyl O
            and nbr.GetIdx() != ester_o
        )
        if not c_adj_chain and o_on_chain:
            # O faces the chain, C(=O) faces away -> acyloxy, not alkoxycarbonyl
            return None
        if not c_adj_chain and not o_on_chain:
            # Neither end touches the chain; ester is in an isolated branch
            return None
        if c_on_chain and o_on_chain:
            # Both ends of ester on principal chain -- ester is backbone,
            # not a substituent. Reject alkoxycarbonyl naming.
            return None

    # Split ester into acid and alkyl (OR) fragments
    acid_atoms, alkyl_atoms = parse_ester_fragments(mol, ester_atoms)
    if not alkyl_atoms:
        return None

    if ester_o is None:
        return None

    # Early return for aromatic alkyl: phenoxycarbonyl, (benzyloxy)carbonyl
    first_alkyl = alkyl_atoms[0]
    first_atom = mol.GetAtomWithIdx(first_alkyl)

    if first_atom.GetIsAromatic():
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if first_alkyl in ring and len(ring) == 6:
                if all(
                    mol.GetAtomWithIdx(r).GetIsAromatic()
                    and mol.GetAtomWithIdx(r).GetSymbol() == "C"
                    for r in ring
                ):
                    return "phenoxycarbonyl"
        return "phenoxycarbonyl"

    if (
        not first_atom.GetIsAromatic()
        and first_atom.GetSymbol() == "C"
        and first_atom.GetTotalNumHs() >= 1
    ):
        arom_nbrs = [
            n
            for n in first_atom.GetNeighbors()
            if n.GetIdx() != ester_o and n.GetIsAromatic()
        ]
        non_h_non_arom = [
            n
            for n in first_atom.GetNeighbors()
            if n.GetIdx() != ester_o
            and not n.GetIsAromatic()
            and n.GetSymbol() != "H"
        ]
        if arom_nbrs and not non_h_non_arom:
            return "(benzyloxy)carbonyl"

    # Guard 3: the alkyl (OR) fragment must be pure carbon (no heteroatoms)
    # for simple alkoxycarbonyl naming.  Heteroatom-containing fragments
    # (e.g., amino acid side chains) need complex naming beyond the scope
    # of this prefix generator.
    has_heteroatom = any(
        mol.GetAtomWithIdx(a).GetSymbol() not in ("C", "H") for a in alkyl_atoms
    )
    if has_heteroatom:
        return None

    # Guard 4: size sanity — if the OR fragment is larger than the principal
    # chain, this ester should be handled by functional-class naming.
    carbon_count = sum(
        1 for a in alkyl_atoms if mol.GetAtomWithIdx(a).GetSymbol() == "C"
    )
    chain_len = len(principal_chain) if principal_chain else 0
    if chain_len > 0 and carbon_count > chain_len:
        return None
    # Reject non-aromatic ring-containing alkyl fragments (macrocyclic esters).
    # Aromatic rings (phenyl) are already handled above.
    ring_info = mol.GetRingInfo()
    has_non_aromatic_ring = any(
        ring_info.NumAtomRings(a) > 0 and not mol.GetAtomWithIdx(a).GetIsAromatic()
        for a in alkyl_atoms
    )
    if has_non_aromatic_ring:
        return None

    if carbon_count == 0:
        return None

    # Build the alkoxycarbonyl name from ALKOXY_NAMES (same table as ethers)
    if carbon_count in ALKOXY_NAMES:
        alkoxy = ALKOXY_NAMES[carbon_count]
        return f"{alkoxy}carbonyl"

    # For larger/unknown sizes, build from alkyl name
    try:
        alkyl_name = get_alkyl_name(carbon_count)
        if alkyl_name.endswith("yl"):
            return f"{alkyl_name[:-2]}yloxy" + "carbonyl"
        return f"{alkyl_name}oxy" + "carbonyl"
    except (ValueError, KeyError):
        try:
            from ..data.chain_names import get_chain_prefix

            prefix = get_chain_prefix(carbon_count)
            return f"{prefix}yloxycarbonyl"
        except (ValueError, KeyError):
            return None


def get_sulfinyl_prefix(
    mol,
    sulfoxide_atoms: tuple,
    principal_chain: Optional[List[int]] = None,
) -> Optional[str]:
    """Generate (alkyl)sulfinyl prefix for sulfoxide as non-principal group (IUPAC P-63.6).

    IUPAC P-63.6: ``R-S(=O)-R'`` when not the principal group is expressed as
    an (alkyl)sulfinyl prefix on the parent chain.

    SMARTS ``[SX3](=[OX1])([#6])[#6]`` matches (S, O, C1, C2).

    Lifted from rules/polyfunctional._get_sulfinyl_prefix verbatim with a NEW
    None-guard at function entry per Phase 160.1 RESEARCH §4.

    Args:
        mol: RDKit Mol object.
        sulfoxide_atoms: Atom indices from sulfoxide SMARTS match.
        principal_chain: Atom indices of the principal chain; may be None.

    Returns:
        Compound prefix string (e.g., ``"methylsulfinyl"``), or None on failure.
    """
    chain_set: Set[int] = set(principal_chain) if principal_chain else set()

    if len(sulfoxide_atoms) < 3:
        return None

    sulfur_idx = sulfoxide_atoms[0]

    # Find the two C neighbors of S (skip O neighbors)
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    c_neighbors = [n for n in sulfur.GetNeighbors() if n.GetSymbol() == "C"]
    if len(c_neighbors) < 2:
        return None

    # Determine which C is on the chain vs substituent
    c1, c2 = c_neighbors[0].GetIdx(), c_neighbors[1].GetIdx()
    c1_on_chain = c1 in chain_set
    c2_on_chain = c2 in chain_set

    if c1_on_chain and not c2_on_chain:
        sub_carbon = c2
    elif c2_on_chain and not c1_on_chain:
        sub_carbon = c1
    else:
        # Neither or both on chain -- use smaller fragment
        frag1 = _count_fragment_atoms(mol, c1, {sulfur_idx})
        frag2 = _count_fragment_atoms(mol, c2, {sulfur_idx})
        sub_carbon = c1 if frag1 <= frag2 else c2

    # Count carbons in substituent fragment
    carbon_count = _count_fragment_atoms(
        mol, sub_carbon, {sulfur_idx}, carbons_only=True
    )
    if carbon_count == 0:
        return None

    # Build compound prefix: methylsulfinyl, ethylsulfinyl, etc.
    try:
        alkyl = get_alkyl_name(carbon_count)
        return f"{alkyl}sulfinyl"
    except (ValueError, KeyError):
        return None


def get_sulfonyl_prefix(
    mol,
    sulfone_atoms: tuple,
    principal_chain: Optional[List[int]] = None,
) -> Optional[str]:
    """Generate (alkyl)sulfonyl prefix for sulfone as non-principal group (IUPAC P-63.6).

    IUPAC P-63.6: ``R-S(=O)(=O)-R'`` when not the principal group is expressed
    as an (alkyl)sulfonyl prefix on the parent chain.

    SMARTS ``[SX4](=[OX1])(=[OX1])([#6])[#6]`` matches (S, O1, O2, C1, C2).

    Lifted from rules/polyfunctional._get_sulfonyl_prefix verbatim with a NEW
    None-guard at function entry per Phase 160.1 RESEARCH §4.

    Args:
        mol: RDKit Mol object.
        sulfone_atoms: Atom indices from sulfone SMARTS match.
        principal_chain: Atom indices of the principal chain; may be None.

    Returns:
        Compound prefix string (e.g., ``"methylsulfonyl"``), or None on failure.
    """
    chain_set: Set[int] = set(principal_chain) if principal_chain else set()

    if len(sulfone_atoms) < 3:
        return None

    sulfur_idx = sulfone_atoms[0]

    # Find the two C neighbors of S (skip O neighbors)
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    c_neighbors = [n for n in sulfur.GetNeighbors() if n.GetSymbol() == "C"]
    if len(c_neighbors) < 2:
        return None

    # Determine which C is on the chain vs substituent
    c1, c2 = c_neighbors[0].GetIdx(), c_neighbors[1].GetIdx()
    c1_on_chain = c1 in chain_set
    c2_on_chain = c2 in chain_set

    if c1_on_chain and not c2_on_chain:
        sub_carbon = c2
    elif c2_on_chain and not c1_on_chain:
        sub_carbon = c1
    else:
        # Neither or both on chain -- use smaller fragment
        frag1 = _count_fragment_atoms(mol, c1, {sulfur_idx})
        frag2 = _count_fragment_atoms(mol, c2, {sulfur_idx})
        sub_carbon = c1 if frag1 <= frag2 else c2

    # Count carbons in substituent fragment
    carbon_count = _count_fragment_atoms(
        mol, sub_carbon, {sulfur_idx}, carbons_only=True
    )
    if carbon_count == 0:
        return None

    # Build compound prefix: methylsulfonyl, ethylsulfonyl, etc.
    try:
        alkyl = get_alkyl_name(carbon_count)
        return f"{alkyl}sulfonyl"
    except (ValueError, KeyError):
        return None


def get_sulfanyl_prefix(
    mol,
    thioether_atoms: tuple,
    principal_chain: Optional[List[int]] = None,
) -> Optional[str]:
    """Generate (alkyl)sulfanyl prefix for thioether as non-principal group (IUPAC P-63.2.5).

    IUPAC P-63.2.5: ``R-S-R'`` when not the principal group is expressed as
    an (alkyl)sulfanyl prefix on the parent chain.

    SMARTS ``[SX2]([#6])[#6]`` matches (S, C1, C2).

    Lifted from rules/polyfunctional._get_sulfanyl_prefix verbatim with a NEW
    None-guard at function entry per Phase 160.1 RESEARCH §4.

    Args:
        mol: RDKit Mol object.
        thioether_atoms: Atom indices from thioether SMARTS match.
        principal_chain: Atom indices of the principal chain; may be None.

    Returns:
        Compound prefix string (e.g., ``"methylsulfanyl"``), or None on failure.
    """
    chain_set: Set[int] = set(principal_chain) if principal_chain else set()

    if len(thioether_atoms) < 3:
        return None

    sulfur_idx = thioether_atoms[0]

    # Find the two C neighbors of S
    sulfur = mol.GetAtomWithIdx(sulfur_idx)
    c_neighbors = [n for n in sulfur.GetNeighbors() if n.GetSymbol() == "C"]
    if len(c_neighbors) < 2:
        return None

    # Determine which C is on the chain vs substituent
    c1, c2 = c_neighbors[0].GetIdx(), c_neighbors[1].GetIdx()
    c1_on_chain = c1 in chain_set
    c2_on_chain = c2 in chain_set

    if c1_on_chain and not c2_on_chain:
        sub_carbon = c2
    elif c2_on_chain and not c1_on_chain:
        sub_carbon = c1
    else:
        # Neither or both on chain -- use smaller fragment
        frag1 = _count_fragment_atoms(mol, c1, {sulfur_idx})
        frag2 = _count_fragment_atoms(mol, c2, {sulfur_idx})
        sub_carbon = c1 if frag1 <= frag2 else c2

    # Count carbons in substituent fragment
    carbon_count = _count_fragment_atoms(
        mol, sub_carbon, {sulfur_idx}, carbons_only=True
    )
    if carbon_count == 0:
        return None

    # Build compound prefix: methylsulfanyl, ethylsulfanyl, etc.
    try:
        alkyl = get_alkyl_name(carbon_count)
        return f"{alkyl}sulfanyl"
    except (ValueError, KeyError):
        return None


def _alkoxy_name_for_branch(
    mol,
    alkyl_atom: int,
    exclude: Set[int],
) -> Optional[str]:
    """Helper: produce the alkoxy-group name (e.g., "methoxy", "ethoxy") for the
    fragment starting at ``alkyl_atom`` and bounded by ``exclude``.

    Used by ``get_carbamoyloxy_prefix`` Branch A to name the R-O- portion of
    ``-NHC(=O)OR`` substituents per IUPAC P-65.6.3 / P-66.6.4. Mirrors the
    ALKOXY_NAMES lookup pattern used by ``get_alkoxycarbonyl_prefix``.

    Returns None if the fragment is heteroatom-bearing, empty, or oversized.
    """
    visited: Set[int] = set()
    queue = deque([alkyl_atom])
    while queue:
        atom_idx = queue.popleft()
        if atom_idx in visited or atom_idx in exclude:
            continue
        visited.add(atom_idx)
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() not in ("C", "H"):
            return None
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in exclude:
                queue.append(nbr_idx)

    carbon_count = _count_fragment_atoms(
        mol, alkyl_atom, exclude, carbons_only=True
    )
    if carbon_count == 0:
        return None
    if carbon_count in ALKOXY_NAMES:
        return ALKOXY_NAMES[carbon_count]
    try:
        alkyl = get_alkyl_name(carbon_count)
        if alkyl.endswith("yl"):
            return alkyl[:-2] + "yloxy"
        return alkyl + "oxy"
    except (ValueError, KeyError):
        return None


def _name_alkyl_branch_from_atom(
    mol,
    alkyl_atom: int,
    exclude: Set[int],
) -> Optional[str]:
    """Helper: produce the alkyl-group name (e.g., "methyl", "ethyl") for the
    fragment starting at ``alkyl_atom`` and bounded by ``exclude``.

    Used by the 3 NEW Phase 160.1 Plan-02-02 generators to name the alkyl
    portion attached to the amide-N or carbamate-N/O. The carbon count is
    derived from a BFS over the fragment (carbons only); ``get_alkyl_name``
    maps the count to the IUPAC stem ("methyl", "ethyl", "propyl", ...).

    Heteroatom-bearing alkyl fragments return None (the caller falls through
    to a different prefix form).
    """
    # Reject heteroatom-bearing alkyl side per IUPAC P-66.6.1 substituent
    # naming (analog to get_alkoxycarbonyl_prefix Guard 3).
    visited: Set[int] = set()
    queue = deque([alkyl_atom])
    while queue:
        atom_idx = queue.popleft()
        if atom_idx in visited or atom_idx in exclude:
            continue
        visited.add(atom_idx)
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() not in ("C", "H"):
            return None
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in exclude:
                queue.append(nbr_idx)

    carbon_count = _count_fragment_atoms(
        mol, alkyl_atom, exclude, carbons_only=True
    )
    if carbon_count == 0:
        return None
    try:
        return get_alkyl_name(carbon_count)
    except (ValueError, KeyError):
        return None


def get_n_alkyl_carbamoyl_prefix(
    mol,
    amide_atoms: tuple,
    principal_chain: Optional[List[int]] = None,
) -> Optional[str]:
    """Generate N-(alkyl)carbamoyl prefix for secondary-amide-as-substituent.

    Per IUPAC P-66.6.1, when a secondary amide ``-C(=O)NHR`` is not the
    principal group AND attached to the parent through the carbonyl-C,
    the prefix form is ``N-(alkyl)carbamoyl``::

        -C(=O)NHCH3   -> N-methylcarbamoyl
        -C(=O)NHC2H5  -> N-ethylcarbamoyl

    Returns None for:

      * Lactam (cyclic amide; handled by ring handler)
      * Amide attached through N (acetylamino path; handled by
        ``_name_amino_branch``)
      * Backbone amide (both ends on principal chain)

    Args:
        mol: RDKit Mol (the full molecule, not the substituent fragment).
        amide_atoms: 4-tuple ``(carbonyl_C, carbonyl_O, amide_N, alkyl_C)``
            from SMARTS ``[CX3](=O)[NX3H1][#6]`` match.
        principal_chain: principal-chain atom indices, or None for
            sub-fragment context (the Tier-0.5 caller; in that case the
            orientation gate is the caller's responsibility — this function
            assumes carbonyl-C-attached orientation and returns the
            carbamoyl form).

    Returns:
        IUPAC-canonical ``"N-(alkyl)carbamoyl"`` string, or None.
    """
    chain_set: Set[int] = set(principal_chain) if principal_chain else set()

    # Guard 1: 4-atom tuple required (FUNCTIONAL_GROUP_SMARTS["secondary_amide"])
    if len(amide_atoms) < 4:
        return None
    carbonyl_c, carbonyl_o, amide_n, alkyl_c = amide_atoms[:4]

    # Guard 2: lactam exclusion — if both amide_N and carbonyl_C are in a ring,
    # this is a cyclic amide (lactam) handled by the ring handler.
    if (
        mol.GetAtomWithIdx(amide_n).IsInRing()
        and mol.GetAtomWithIdx(carbonyl_c).IsInRing()
    ):
        return None

    # Guard 3: orientation — carbamoyl form requires attach through carbonyl_C.
    # In principal-chain context: reject N-attached and backbone-amide cases.
    # In sub-fragment context (principal_chain=None): trust the caller — the
    # Tier-0.5 hook (Plan-02-04) pre-filters by attach_idx.
    if chain_set:
        c_on_chain = carbonyl_c in chain_set
        n_on_chain = amide_n in chain_set
        if n_on_chain and not c_on_chain:
            return None  # N-attached → acetylamino path, not carbamoyl
        if c_on_chain and n_on_chain:
            return None  # backbone amide → reject

    # Guard 4: name the alkyl group on N (must be pure-C fragment)
    alkyl_name = _name_alkyl_branch_from_atom(
        mol, alkyl_c, exclude={carbonyl_c, carbonyl_o, amide_n}
    )
    if not alkyl_name:
        return None

    return f"N-{alkyl_name}carbamoyl"


def get_n_n_dialkyl_carbamoyl_prefix(
    mol,
    amide_atoms: tuple,
    principal_chain: Optional[List[int]] = None,
) -> Optional[str]:
    """Generate N,N-(dialkyl)carbamoyl prefix for tertiary-amide-as-substituent.

    Per IUPAC P-66.6.1::

        -C(=O)N(CH3)2      -> N,N-dimethylcarbamoyl
        -C(=O)N(CH3)(C2H5) -> N-ethyl-N-methylcarbamoyl  (alphabetized)
        -C(=O)N(C2H5)2     -> N,N-diethylcarbamoyl

    Returns None for cyclic tertiary amide, backbone amide, or N-attached
    orientation (same gates as ``get_n_alkyl_carbamoyl_prefix``).
    """
    chain_set: Set[int] = set(principal_chain) if principal_chain else set()

    if len(amide_atoms) < 5:
        return None
    carbonyl_c, carbonyl_o, amide_n, alkyl_c1, alkyl_c2 = amide_atoms[:5]

    if (
        mol.GetAtomWithIdx(amide_n).IsInRing()
        and mol.GetAtomWithIdx(carbonyl_c).IsInRing()
    ):
        return None

    if chain_set:
        c_on_chain = carbonyl_c in chain_set
        n_on_chain = amide_n in chain_set
        if n_on_chain and not c_on_chain:
            return None
        if c_on_chain and n_on_chain:
            return None

    # Name both alkyl groups on N
    excl_base = {carbonyl_c, carbonyl_o, amide_n}
    alkyl1 = _name_alkyl_branch_from_atom(
        mol, alkyl_c1, exclude=excl_base | {alkyl_c2}
    )
    alkyl2 = _name_alkyl_branch_from_atom(
        mol, alkyl_c2, exclude=excl_base | {alkyl_c1}
    )
    if not alkyl1 or not alkyl2:
        return None

    if alkyl1 == alkyl2:
        return f"N,N-di{alkyl1}carbamoyl"

    # Alphabetize ascending: N-{lower}-N-{higher}carbamoyl
    if alpha_sort_key(alkyl1) < alpha_sort_key(alkyl2):
        first, second = alkyl1, alkyl2
    else:
        first, second = alkyl2, alkyl1
    return f"N-{first}-N-{second}carbamoyl"


def get_carbamoyloxy_prefix(
    mol,
    carbamate_atoms: tuple,
    principal_chain: Optional[List[int]] = None,
) -> Optional[str]:
    """Generate carbamate prefix per IUPAC P-66.6.4.

    The carbamate ``-NHC(=O)O-`` has TWO orientation forms:

      * **Branch A**: connects through N → ``-NHC(=O)OCH3`` → ``"(methoxycarbonyl)amino"``
      * **Branch B**: connects through O → ``-OC(=O)NH2`` → ``"carbamoyloxy"``

    Returns None for cyclic carbamate (oxazolidinone; handled by ring
    handler) or backbone carbamate (all atoms on principal chain).

    In sub-fragment context (``principal_chain=None``) defaults to Branch A
    naming for ``-NHC(=O)OR`` fragments (the more-common substituent shape per
    CONTEXT D-05 row 11); Branch B is invoked by the Tier-0.5 caller when
    attach_idx == ester_O.

    Args:
        mol: RDKit Mol object.
        carbamate_atoms: 5-tuple ``(amide_N, carbonyl_C, carbonyl_O, ester_O,
            alkyl_C)`` from SMARTS ``[NX3][CX3](=O)[OX2][#6]`` match.
        principal_chain: principal-chain atom indices, or None.
    """
    chain_set: Set[int] = set(principal_chain) if principal_chain else set()

    if len(carbamate_atoms) < 5:
        return None
    amide_n, carbonyl_c, carbonyl_o, ester_o, alkyl_c = carbamate_atoms[:5]

    # Guard: cyclic carbamate (oxazolidinone) — deferred to ring handler
    if (
        mol.GetAtomWithIdx(amide_n).IsInRing()
        and mol.GetAtomWithIdx(carbonyl_c).IsInRing()
        and mol.GetAtomWithIdx(ester_o).IsInRing()
    ):
        return None

    exclude_for_alkyl = {amide_n, carbonyl_c, carbonyl_o, ester_o}

    if chain_set:
        n_on_chain = amide_n in chain_set
        o_on_chain = ester_o in chain_set
        if n_on_chain and o_on_chain:
            return None  # backbone carbamate
        if n_on_chain:
            # Branch A: -NHC(=O)OR → "(R-oxycarbonyl)amino" (IUPAC P-66.6.4)
            alkoxy_part = _alkoxy_name_for_branch(mol, alkyl_c, exclude_for_alkyl)
            if not alkoxy_part:
                return None
            return f"({alkoxy_part}carbonyl)amino"
        if o_on_chain:
            # Branch B: -OC(=O)NH2 → "carbamoyloxy"
            return "carbamoyloxy"
        return None

    # Sub-fragment context (principal_chain=None): default to Branch A.
    # The Tier-0.5 caller selects Branch B explicitly by detecting attach_idx
    # == ester_O before invoking this generator.
    alkoxy_part = _alkoxy_name_for_branch(mol, alkyl_c, exclude_for_alkyl)
    if not alkoxy_part:
        return None
    return f"({alkoxy_part}carbonyl)amino"


def get_substituent_prefix_form(
    fg_name: str,
    mol,
    atoms: tuple,
    principal_chain: Optional[List[int]] = None,
) -> Optional[str]:
    """Dispatcher across the 14-row IUPAC P-65/P-66 prefix-form table.

    Returns the IUPAC-canonical prefix-form string for ``fg_name`` applied to
    ``atoms``, or ``None`` if no prefix-form rule applies (caller falls back
    to its default behavior).

    Per Phase 160.1 CONTEXT D-05 closed-set: extending beyond these 14 rows
    is v19+1 scope. Sibling phases (Phase 163 FRN P-25.3 functional
    replacement, v19+1 hydroxamic / hydrazide / phosphorus) attach by adding
    ADDITIONAL rows below — never deleting or modifying existing rows.

    Args:
        fg_name: Functional-group name from ``FUNCTIONAL_GROUP_SMARTS``
            (e.g., ``"ester"``, ``"primary_amide"``, ``"thioether"``).
        mol: RDKit Mol object (the full molecule, not the substituent fragment).
        atoms: Atom indices from the SMARTS match for ``fg_name``.
        principal_chain: Atom indices of the principal chain; may be None for
            the Tier-0.5 sub-fragment caller (substituent_enumerator) per
            Phase 160.1 CONTEXT D-04. Lifted generators internally None-guard.

    Returns:
        IUPAC-canonical prefix-form string, or None when no rule applies.
    """
    # --- Dynamic (lifted) generators — rows 1, 2, 7, 8, 9, 10 ---
    if fg_name == "ester":
        return get_alkoxycarbonyl_prefix(mol, atoms, principal_chain)
    if fg_name in ("ether", "vinyl_ether", "aromatic_ether"):
        return get_alkoxy_prefix(mol, atoms, principal_chain)
    if fg_name == "sulfoxide":
        return get_sulfinyl_prefix(mol, atoms, principal_chain)
    if fg_name == "sulfone":
        return get_sulfonyl_prefix(mol, atoms, principal_chain)
    if fg_name == "thioether":
        return get_sulfanyl_prefix(mol, atoms, principal_chain)

    # --- Dynamic (NEW Plan-02-02) — rows 4, 5, 11 ---
    if fg_name == "secondary_amide":
        return get_n_alkyl_carbamoyl_prefix(mol, atoms, principal_chain)
    if fg_name == "tertiary_amide":
        return get_n_n_dialkyl_carbamoyl_prefix(mol, atoms, principal_chain)
    if fg_name == "carbamate":
        return get_carbamoyloxy_prefix(mol, atoms, principal_chain)

    # --- Static-table lookups — rows 3, 6, 12, 13, 14 ---
    if fg_name == "primary_amide":
        return PREFIX_FORMS.get("primary_amide")  # "carbamoyl"
    if fg_name == "nitrile":
        return PREFIX_FORMS.get("nitrile")  # "cyano"
    if fg_name == "urea":
        return PREFIX_FORMS.get("urea")  # "carbamoylamino"
    if fg_name == "isocyanate":
        return PREFIX_FORMS.get("isocyanate")  # "isocyanato"
    if fg_name == "isothiocyanate":
        return PREFIX_FORMS.get("isothiocyanate")  # "isothiocyanato"

    # --- Phase 163 FRN attachment slot ---
    # Phase 163 FRN attachment: thio/seleno/telluro/imino chalcogen replacement
    # (additive on dispatcher; NO Plan-02 row deletion; NO signature change).

    # Unknown / out-of-table → fall through; caller may consult PREFIX_FORMS
    # directly for any other static prefix (e.g., carboxylic_acid → "carboxy",
    # hydroxyl → "hydroxy", amino, etc.). Returning None signals "this row
    # is not in the 14-row Phase 160.1 closed-set".
    return None


# ====================================================================
# Tier-0.5 hook — Phase 160.1 CONTEXT D-04 substituent_enumerator wiring
# ====================================================================

# The 14-row prefix-form FG names in dispatcher order. Lazy-compiled SMARTS
# patterns are cached at first call per RESEARCH §11 Risk F (avoid repeated
# Chem.MolFromSmarts cost per Tier-0.5 invocation; ~ 14 × 1µs cache lookup
# instead of ~ 14 × 50µs compile + match cost).
_PREFIX_FORM_FG_NAMES = (
    "ester",
    "ether", "vinyl_ether", "aromatic_ether",
    "primary_amide", "secondary_amide", "tertiary_amide",
    "nitrile",
    "sulfoxide", "sulfone", "thioether",
    "carbamate", "urea",
    "isocyanate", "isothiocyanate",
)

_PREFIX_FORM_PATTERNS: dict = {}  # Lazily populated on first call

# Phase 160.2 Plan-04-03a WR-05 closure: lock guarding the lazy-init of
# ``_PREFIX_FORM_PATTERNS``. Industry-standard double-check pattern
# (first check WITHOUT lock for fast-path; re-check INSIDE lock to ensure
# one-time init under concurrent first-call). Closes the thread-safety
# gap documented at 160.1-REVIEW.md WR-05 — concurrent first callers no
# longer race ``Chem.MolFromSmarts`` 28 times.
_PREFIX_FORM_CACHE_LOCK = threading.Lock()


def _ensure_patterns_cached() -> None:
    """Lazy-compile the 14 SMARTS patterns once per process.

    Phase 160.2 Plan-04-03a WR-05 closure: thread-safe via
    ``threading.Lock`` double-check pattern. Concurrent first calls
    re-acquire-and-recheck under the lock so the population loop runs
    exactly once across all threads (mirroring the standard
    double-checked-locking idiom).
    """
    if _PREFIX_FORM_PATTERNS:
        return  # First check (no lock; fast path for the common warm-cache case)
    with _PREFIX_FORM_CACHE_LOCK:
        if _PREFIX_FORM_PATTERNS:
            return  # Second check (inside lock; ensures one-time init)
        from rdkit import Chem
        from ..perception.functional_groups import FUNCTIONAL_GROUP_SMARTS

        for fg_name in _PREFIX_FORM_FG_NAMES:
            if fg_name in FUNCTIONAL_GROUP_SMARTS:
                _PREFIX_FORM_PATTERNS[fg_name] = Chem.MolFromSmarts(
                    FUNCTIONAL_GROUP_SMARTS[fg_name]
                )


def _check_substituent_prefix_form(
    mol,
    frag_atoms_set: Set[int],
    attach_idx: int,
) -> Optional[str]:
    """Tier-0.5 prefix-form check for FG-bearing substituent fragments.

    Per Phase 160.1 CONTEXT D-04 + IUPAC P-65 / P-66 prefix-form rules, a
    substituent fragment that ENTIRELY contains one of the 14 non-principal
    functional groups is named via the IUPAC-canonical prefix form (e.g.,
    ``-C(=O)OCH3 → "methoxycarbonyl"`` per P-65.6.3), short-circuiting the
    Tier-4 recursive ``name_substituent_fragment`` path that produces
    ``"methyl formatyl"`` / ``"hydroxymethyl"`` incorrectly per RESEARCH §3
    root-cause bug trace.

    PURE: read-only on (mol, frag_atoms_set, attach_idx); no mutation;
    no ``pool.add()``; no ``MolecularFeatures`` touch.

    Args:
        mol: RDKit Mol (the full molecule, not the substituent fragment).
        frag_atoms_set: atom indices of the substituent fragment.
        attach_idx: index of the atom in ``frag_atoms_set`` that bonds to
            the parent. Used only for orientation discrimination on the
            carbamate row 11 (Branch A vs Branch B); ignored otherwise.

    Returns:
        IUPAC-canonical prefix form (e.g., ``"methoxycarbonyl"``,
        ``"carbamoyl"``) if a 14-row match applies entirely within the
        fragment; None otherwise (caller falls through to Tier-1).
    """
    _ensure_patterns_cached()
    # Tighter than subset: the SMARTS match must be exactly the fragment
    # (no extra atoms). This prevents large heterosubstituent fragments
    # containing an embedded FG bond from being mis-named as that FG's
    # prefix (e.g., a 47-atom branch that contains an ether bond should
    # NOT be named "heptatetracontyloxy"). Linker-bearing substituents
    # like -CH2-C(=O)OCH3 fall through to Tier-1+ which correctly names
    # the methylene linker around the FG.
    for fg_name in _PREFIX_FORM_FG_NAMES:
        pattern = _PREFIX_FORM_PATTERNS.get(fg_name)
        if pattern is None:
            continue
        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            match_set = set(match)
            # --- Phase 160.2 Plan-04-01: CR-01 fix per IUPAC P-66.6.4 ---
            # Branch B carbamate (-OC(=O)NH2): when attach_idx points at the
            # ester_O (match[3]) the prefix is ``carbamoyloxy`` (P-66.6.4),
            # NOT ``(R-oxycarbonyl)amino`` (Branch A) or some unrelated
            # acyloxy form.
            #
            # SMARTS ``[NX3][CX3](=O)[OX2][#6]`` matches 5 atoms; the trailing
            # ``[#6]`` is the alkyl_C parent attach point (match[4]) which
            # may live OUTSIDE the substituent fragment (live case
            # ``O=C(N)OCCCC(=O)O`` where the substituent is the 4-atom
            # -OC(=O)NH2 group and alkyl_C is on the principal chain), or
            # INSIDE the substituent fragment (synthetic-direct call where
            # the fragment IS the full SMARTS match). Accept both shapes so
            # the documented Tier-0.5 → Branch B routing fires uniformly.
            # See 160.2-AUDIT-DECOMP-CLOSURE.md §5 + 160.1-REVIEW.md CR-01.
            if (
                fg_name == "carbamate"
                and len(match) >= 5
                and attach_idx == match[3]
                and (
                    # Substituent-context (live): fragment is match - {alkyl_C}
                    frag_atoms_set == match_set - {match[4]}
                    # Full-match context (unit / direct call): frag == match
                    or frag_atoms_set == match_set
                )
            ):
                return "carbamoyloxy"
            # --- End CR-01 fix ---
            # FG must EQUAL the fragment (no extra atoms). This is the
            # IUPAC P-65/P-66 prefix-form precondition: the substituent
            # fragment must be the FG itself, not a larger group containing
            # the FG (CONTEXT D-04 strict scope).
            if match_set != frag_atoms_set:
                continue
            prefix = get_substituent_prefix_form(
                fg_name, mol, tuple(match), principal_chain=None
            )
            if prefix is not None:
                return prefix
    return None
