"""
Parent selection logic for ring vs chain compounds (IUPAC P-44.1).

This module implements the IUPAC 2013 rules for selecting between a ring
and a chain as the parent structure in organic compound naming.

Key rule (P-44.1): "The principal characteristic group cited as suffix
must be attached to the principal chain or ring system."

Key rule (P-52.2.8): "When the ring and the chain contain the same number
of skeletal atoms in the ring or chain, the ring system is always preferred
as the principal chain."

This means:
- If -COOH is on the chain, chain MUST be parent
- If -COOH is directly on the ring, ring MUST be parent
- For hydrocarbons (no FG), rings have seniority over chains (P-44.1.2.2)
- When FG count is tied, P-44.1 cascade (chain length > multiple bonds) applied before P-52.2.8 ring default
- When multiple ring systems exist, the most senior one is the parent (P-44.2)
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from rdkit import Chem

from ..perception.chains import find_longest_skeletal_chain
from ..perception.natural_products import detect_natural_product
from .locants import compare_locant_sets
from .ring_selection import ring_system_score
from .seniority import PG_ATTACHMENT_INDICES

# WS-A (P-66.6.1): these suffixes decorate a SKELETAL atom of the
# parent — there is no exocyclic-carbon '-one' suffix (unlike the carbo-
# suffixes -carboxylic acid / -carbaldehyde / -carbonitrile, whose carbon
# is exocyclic by design). For these PGs, "on ring" means the suffix atom
# IS a ring atom; the bonded-to-ring relaxation is invalid and mis-parented
# every aryl ketone (O=CCc1cnc[nH]1 named bare 'benzene').
SKELETAL_SUFFIX_PGS = {
    "ketone",
    "thioketone",
    "selenoketone",
    "telluroketone",
    # Wave2 T3a (P-52.2.8 / P-29.4.2): the -ol/-thiol/-selenol/-tellurol/
    # -amine/-imine suffixes likewise decorate a skeletal atom — there is no
    # exocyclic-carbon form (no "-carbinol"). An exocyclic carbon bearing one
    # of these can never be expressed as a ring suffix, so "on ring" is
    # membership-only. Every entry here whose SMARTS leads with the
    # heteroatom REQUIRES a paired PG_ATTACHMENT_INDICES override pointing at
    # the locant-bearing carbon (seniority.py), else genuine ring suffixes
    # (4-methylcyclohexan-1-ol) go false-negative. imine already leads with C.
    "alcohol",
    "primary_alcohol",
    "secondary_alcohol",
    "tertiary_alcohol",
    "thiol",
    "selenol",
    "tellurol",
    "primary_amine",
    "secondary_amine",
    "tertiary_amine",
    "imine",
}

# : the PGs whose characteristic heteroatom can carry MORE THAN ONE
# bearing carbon, and can therefore BRIDGE two parent candidates. These are the
# only subtypes for which the RC-4 (heteroatom, bearing-C) normalisation in
# seniority._normalize_pcg_match loses information -- it keeps one carbon and
# discards the rest, tie-broken by atom index. Alcohols, primary amines and
# aromatic amines have exactly one bearing carbon, so they are excluded: nothing
# is lost for them and the membership loop already answers them.
_BRIDGING_HETEROATOM_PGS = frozenset({
    "secondary_amine",
    "tertiary_amine",
})

logger = logging.getLogger(__name__)


def _pg_attachment_atoms(
    fg_name: Optional[str], pg_atoms: Tuple[int, ...]
) -> List[int]:
    """Return molecular atom indices that bear the IUPAC locant for ``fg_name``.

    Most FG SMARTS lead with the locant-bearing atom (e.g. the C of -COOH in
    ``[CX3](=O)[OX2H1]``). For those, this returns ``[pg_atoms[0]]``.

    Some FG SMARTS lead with a flanking atom; ``PG_ATTACHMENT_INDICES``
    overrides the SMARTS atom indices to use. The motivating case is
    ``disulfide`` (SMARTS ``[#6][SX2][SX2][#6]``): atom 0 is a flanking C,
    but IUPAC P-31.1.4 says the locant set uses the heteroatoms (S, S);
    ``PG_ATTACHMENT_INDICES["disulfide"] = [1, 2]`` restores correct
    behaviour. Downstream comparators take ``min(locants)`` over the
    returned atoms when the FG spans multiple positions.

    Args:
        fg_name: Functional group name from SENIORITY_ORDER. ``None`` is
                 treated as default (back-compat for callers that don't
                 thread the name through).
        pg_atoms: SMARTS-match tuple of molecular atom indices.

    Returns:
        List of molecular atom indices (always non-empty for a valid
        ``pg_atoms``). Order is the SMARTS-index order from the override.

    Source: IM-01 fix; agent-5-regression-class.md root cause analysis.
    """
    if not pg_atoms:
        return []
    indices = PG_ATTACHMENT_INDICES.get(fg_name) if fg_name else None
    if indices is None:
        return [pg_atoms[0]]
    # Filter out any indices that exceed the SMARTS match length (defensive
    # against malformed SMARTS / future overrides). At minimum return atom 0
    # so the comparator never sees an empty locant set for a present PG.
    out = [pg_atoms[i] for i in indices if i < len(pg_atoms)]
    return out if out else [pg_atoms[0]]


def _build_ring_pos(ring_set: Set[int], ring_info: dict = None) -> dict:
    """Build atom-to-locant map using IUPAC ring numbering.

    Per ASML-19 / /: uses actual IUPAC ring numbering
    when available, instead of sorted atom index positional proxy.

     extension: accepts both int locants and ``(int, str)`` tuple
    locants (for fusion atoms like ``'4a' -> (4, 'a')``). When ANY tuple
    value is present, ALL int values are coerced to ``(n, '')`` tuples
    so the returned dict is homogeneous — this is REQUIRED because
    ``_compare_multiple_bond_locants`` (line ~416 below) does
    ``min(ring_pos[a], ring_pos[b])``, which raises ``TypeError`` on
    mixed int+tuple comparison in Python 3 (RESEARCH §3 Risk 3).

    Priority:
    1. iupac_locants from ring_info (authoritative IUPAC numbering,
       either pure-int or homogeneously-coerced tuple)
    2. Fallback: sorted atom indices mapped to 1-indexed positions
       (correct for carbocyclic rings where any consistent numbering
        produces equivalent comparison results due to ring symmetry,
        preserved for back-compat with pre-147 callers per)

    Args:
        ring_set: Set of atom indices in the ring system
        ring_info: Optional dict with 'iupac_locants' key mapping
                   atom_idx -> locant. Value may be int or ``(int, str)``
                   tuple locant (for fusion atoms). Legacy string locants
                   (e.g. ``'3a'``) are filtered out and trigger sorted
                   fallback if coverage becomes incomplete. May contain
                   atoms outside ring_set; only ring atoms with int-
                   or tuple-typed locants are used.

    Returns:
        Dict mapping atom_idx -> locant. The dict is HOMOGENEOUS: either
        all int values or all ``(int, str)`` tuple values. Partial
        coverage falls through to sorted-int fallback (back-compat).

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.4.1.4+
    Source: https://iupac.qmul.ac.uk/BlueBook/P1.html P-14.5.2, P-14.7
    Source: CONTEXT (tuple encoding), (back-compat),
            RESEARCH §3 Risk 3 (min() hazard at:416 site).
    """
    if ring_info and ring_info.get("iupac_locants"):
        iupac = ring_info["iupac_locants"]
        # Only include atoms that are in ring_set; accept int OR tuple
        # locants. Other types (e.g. legacy strings like '3a')
        # are filtered and may trigger the sorted fallback below.
        ring_pos = {}
        for atom_idx in ring_set:
            if atom_idx in iupac:
                locant = iupac[atom_idx]
                if isinstance(locant, (int, tuple)):
                    ring_pos[atom_idx] = locant
        # Use authoritative locants only when coverage is complete.
        if len(ring_pos) == len(ring_set):
            # : enforce homogeneity invariant. If any tuple
            # locant is present, coerce all int values to (n, '') tuples
            # so downstream min()/sort() operations on ring_pos values
            # never TypeError on mixed int/tuple comparison.
            if any(isinstance(v, tuple) for v in ring_pos.values()):
                ring_pos = {
                    k: (v, '') if isinstance(v, int) else v
                    for k, v in ring_pos.items()
                }
            return ring_pos
        # Partial coverage -> sorted fallback (back-compat with pre-147
        # callers; cascade step 6 in candidate_pool.py gates on complete
        # coverage via _has_iupac_locants).

    # Fallback: sorted atom indices (correct for carbocyclic rings).
    ring_sorted = sorted(ring_set)
    return {atom_idx: i + 1 for i, atom_idx in enumerate(ring_sorted)}


@dataclass
class ParentSelectionResult:
    """Result of parent structure selection.

    Attributes:
        parent_type: 'ring' or 'chain'
        parent_atoms: Atom indices of the parent structure
        substituent_rings: Rings that become substituents (when chain is parent)
        reasoning: Explanation for debugging
        principal_ring_system: The senior ring system chosen by the single
            authoritative among-rings computation (P-44.2). Populated once by
            namer._classify (chokepoint consolidation); the
            derived ``senior_ring_system`` / ``principal_ring`` feature fields
            are read from this one value. ``None`` until populated (e.g. pure
            acyclic, or before the post-pass runs).
    """
    parent_type: str  # 'ring' or 'chain'
    parent_atoms: List[int]
    substituent_rings: List[Tuple[int, ...]]  # Rings that become substituents
    reasoning: str  # For debugging
    # : one authoritative principal-ring-system computation.
    principal_ring_system: Optional[Tuple[int, ...]] = None
    # (offer-not-return): the SIZE of the ranked P-44
    # parent pool this result was chosen from. 1 for the pre-empt branches and a
    # single-candidate pool; >=2 when several ring/chain candidates competed.
    # Pure metadata — no naming logic reads it — surfaced so the best-effort
    # top-level namer can, ONLY when ``ranked[0]`` abstains, retry the junior
    # pool members (``_forced_parent_rank``) under an RT gate. ``ranked[0]`` and
    # every PIN-nameable molecule are byte-identical; this never reorders.
    parent_pool_size: int = 1


def is_principal_group_on_ring(
    mol,
    ring_atoms: Set[int],
    principal_group_atoms: List[tuple],
    principal_group: Optional[str] = None,
) -> bool:
    """
    Check if the principal functional group is directly attached to the ring.

    For carboxylic acid: the carbonyl carbon must be bonded to a ring atom.
    For alcohol: the carbon bearing -OH must be a ring atom.

    The key distinction is:
    - "on ring" = FG attachment point is bonded to a ring carbon
    - NOT just anywhere connected through chain to ring

    Args:
        mol: RDKit Mol object
        ring_atoms: Set of atom indices in the ring
        principal_group_atoms: List of tuples, each tuple is a SMARTS match
        principal_group: Optional FG name; when provided, attachment lookup
                         honours ``PG_ATTACHMENT_INDICES`` (IM-01).

    Returns:
        True if principal group is directly attached to ring
    """
    if not principal_group_atoms:
        return False

    ring_atoms_set = set(ring_atoms)

    for pg_atoms in principal_group_atoms:
        if not pg_atoms:
            continue

        # IM-01: per-FG attachment indices instead of literal pg_atoms[0].
        # For most FGs returns [pg_atoms[0]]; for ``disulfide`` returns
        # the two S atoms so a ring-bound S-S is correctly recognised as
        # "on ring" via the heteroatoms, not a flanking carbon.
        for attachment_atom in _pg_attachment_atoms(principal_group, pg_atoms):
            # Self-check: attachment atom itself may be a ring atom
            if attachment_atom in ring_atoms_set:
                return True

            # WS-A: skeletal suffixes (-one family) have no exocyclic
            # form — membership above is the ONLY way they can be on-ring.
            if principal_group in SKELETAL_SUFFIX_PGS:
                continue

            # Check if attachment atom is directly bonded to a ring atom
            atom = mol.GetAtomWithIdx(attachment_atom)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() in ring_atoms_set:
                    return True

    # (P-44.1): a characteristic heteroatom that BRIDGES two parent
    # candidates -- a secondary/tertiary amine N bonded to a carbon of EACH ring
    # -- is attached to BOTH of them, so P-44.1 disqualifies neither and the
    # P-44.2 ring-seniority tiebreak is what decides (namer.py:4321 already says
    # so in prose). The loop above could not see that: the RC-4 union in
    # get_principal_group normalises every amine match to a ``(N, bearing-C)``
    # 2-tuple (``_normalize_pcg_match``), which KEEPS ONLY ONE of the N's
    # carbons -- picked by heavy-neighbour count, tie-broken by ATOM INDEX. For a
    # bridging N the candidate carbons tie, so the survivor -- and therefore this
    # predicate's answer -- flipped with the input atom order. Measured on
    # Brc1ccccc1Nc1ccccn1: 8 of 12 randomised orderings named it, 4 abstained,
    # because ``pg_on_senior`` went False whenever the surviving carbon sat in
    # the JUNIOR ring and the senior pyridine was then discarded for
    # ``atom_rings[0]``. ``PG_ATTACHMENT_INDICES['secondary_amine'] = [1, 2]``
    # was written for the raw 3-tuple and is dead against the normalised shape.
    #
    # Ask the characteristic heteroatom instead: it is order-independent and it
    # sees EVERY bearing carbon. For this family the heteroatom's heavy
    # neighbours ARE its bearing carbons, so "het is in / bonded to this ring" is
    # exactly "some bearing carbon is in this ring" -- the P-44.1 question --
    # and it can never report a group as on-ring that is not attached to it.
    # Scoped to the two subtypes whose heteroatom can carry more than one carbon;
    # alcohols, primary and aromatic amines have a single bearing carbon, so the
    # loop above already answers them and this block is a no-op there.
    # Both members of _BRIDGING_HETEROATOM_PGS are also in SKELETAL_SUFFIX_PGS,
    # where "on ring" is MEMBERSHIP-only: the locant-bearing carbon must BE a
    # ring atom, never merely bonded to one. That semantics is preserved exactly
    # -- the heteroatom's heavy neighbours are its bearing carbons, so the test
    # below is "some bearing carbon of this N IS an atom of this ring", which is
    # the same question the loop above asks, only over ALL the carbons instead of
    # the single one that survived normalisation. The N itself is deliberately
    # NOT accepted by membership: a ring-skeletal nitrogen is a ring heteroatom,
    # not an amine suffix on that ring.
    if principal_group in _BRIDGING_HETEROATOM_PGS:
        for pg_atoms in principal_group_atoms:
            if not pg_atoms:
                continue
            het = mol.GetAtomWithIdx(pg_atoms[0])
            if het.GetSymbol() != 'N':
                continue  # unexpected match shape -- fail closed to the loop above
            for neighbor in het.GetNeighbors():
                if (neighbor.GetSymbol() == 'C'
                        and neighbor.GetIdx() in ring_atoms_set):
                    return True

    return False


def is_principal_group_on_chain(
    mol,
    chain_atoms: List[int],
    principal_group_atoms: List[tuple],
    principal_group: Optional[str] = None,
) -> bool:
    """
    Check if the principal functional group is on the chain.

    For carboxylic acid: the carbonyl carbon must be in chain_atoms.
    For alcohol: the carbon bearing -OH must be in chain_atoms.

    Args:
        mol: RDKit Mol object
        chain_atoms: List of atom indices in the principal chain
        principal_group_atoms: List of tuples, each tuple is a SMARTS match
        principal_group: Optional FG name; when provided, attachment lookup
                         honours ``PG_ATTACHMENT_INDICES`` (IM-01).

    Returns:
        True if principal group is on the chain
    """
    if not principal_group_atoms or not chain_atoms:
        return False

    chain_set = set(chain_atoms)

    for pg_atoms in principal_group_atoms:
        if not pg_atoms:
            continue

        # IM-01: per-FG attachment indices (see _pg_attachment_atoms docstring).
        for attachment_atom in _pg_attachment_atoms(principal_group, pg_atoms):
            # Check if attachment atom is in the chain
            if attachment_atom in chain_set:
                return True

            # Also check if attachment atom is bonded to chain
            # (for cases where FG is terminal, e.g., -CH2-COOH)
            atom = mol.GetAtomWithIdx(attachment_atom)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() in chain_set:
                    # The attachment point is bonded to chain
                    return True

    return False


def _count_multiple_bonds(mol, atom_set: Set[int]) -> int:
    """Count double + triple bonds where both atoms are in atom_set.

    Args:
        mol: RDKit Mol object
        atom_set: Set of atom indices to consider

    Returns:
        Number of double or triple bonds within atom_set
    """
    count = 0
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set:
            bt = bond.GetBondType()
            if bt == Chem.BondType.DOUBLE or bt == Chem.BondType.TRIPLE:
                count += 1
    return count


def _count_double_bonds(mol, atom_set: Set[int]) -> int:
    """Count double bonds (not triple) where both atoms are in atom_set."""
    count = 0
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set:
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                count += 1
    return count


def _get_largest_individual_ring_size(mol, ring_system_atoms: Set[int]) -> int:
    """Get size of largest individual SSSR ring in a ring system.

    Per P-52.2.8, ring-vs-chain comparison uses individual ring size,
    not total fused system atom count.

    Args:
        mol: RDKit Mol object
        ring_system_atoms: Set of atom indices in the ring system

    Returns:
        Size of the largest individual ring, or total atoms if no rings found
    """
    ri = mol.GetRingInfo()
    max_size = 0
    for ring in ri.AtomRings():
        ring_set = set(ring)
        if ring_set.issubset(ring_system_atoms):
            max_size = max(max_size, len(ring))
    return max_size if max_size > 0 else len(ring_system_atoms)


def _count_substituents_on_atoms(mol, atom_set: Set[int]) -> int:
    """Count non-H substituents attached to atoms in atom_set but not in it."""
    count = 0
    for idx in atom_set:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() not in atom_set and nbr.GetSymbol() != 'H':
                count += 1
    return count


# ============================================================================
# Composable P-44.1 Comparators (ring vs chain)
# ============================================================================
# Each returns: 1 (chain wins), -1 (ring wins), 0 (tie)


def _compare_chain_length(chain_len: int, ring_size: int) -> int:
    """P-44.1(c): Maximum chain length / ring skeletal atoms."""
    if chain_len > ring_size:
        return 1
    elif ring_size > chain_len:
        return -1
    return 0


def _compare_multiple_bonds(mol, chain_set: Set[int], ring_set: Set[int]) -> int:
    """P-44.1(d): Maximum number of multiple bonds (double + triple)."""
    chain_mult = _count_multiple_bonds(mol, chain_set)
    ring_mult = _count_multiple_bonds(mol, ring_set)
    if chain_mult > ring_mult:
        return 1
    elif ring_mult > chain_mult:
        return -1
    return 0


def _compare_double_bonds(mol, chain_set: Set[int], ring_set: Set[int]) -> int:
    """P-44.1(e): Maximum number of double bonds."""
    chain_db = _count_double_bonds(mol, chain_set)
    ring_db = _count_double_bonds(mol, ring_set)
    if chain_db > ring_db:
        return 1
    elif ring_db > chain_db:
        return -1
    return 0


def _compare_pg_locants(
    mol, chain: List[int], ring_set: Set[int],
    principal_group_atoms: List[tuple],
    ring_info: dict = None,
    principal_group: Optional[str] = None,
) -> int:
    """P-44.1(f): Lowest locants for principal groups.

    Compares the locant sets for principal characteristic group attachment
    points on the chain versus on the ring, using first-point-of-difference
    comparison (IUPAC P-14.7).

    Locants are 1-indexed IUPAC-style positions (per P-14.7):
    - Chain: position along the chain list (atom at index 0 -> locant 1).
    - Ring: IUPAC ring numbering when available (via ring_info), otherwise
      sorted atom indices mapped to 1-indexed positions as fallback.

    For multi-atom PGs (e.g. ``disulfide`` whose locant atoms are two
    sulfurs), the per-instance locant is ``min`` over the FG's attachment
    atoms, matching IUPAC P-31.1.4 ("lowest locant rule applies to the PG
    as a set of attachments"). See ``_pg_attachment_atoms``.

    Args:
        mol: RDKit Mol object
        chain: Ordered list of atom indices forming the principal chain
        ring_set: Set of atom indices in the ring system
        principal_group_atoms: List of tuples of atom indices from SMARTS matches
        ring_info: Optional dict with 'iupac_locants' for IUPAC ring numbering
        principal_group: Optional FG name; when provided, attachment lookup
                         honours ``PG_ATTACHMENT_INDICES`` (IM-01).

    Returns:
        1 if chain has lower PG locants (chain wins)
        -1 if ring has lower PG locants (ring wins)
        0 if tied or neither has PG locants
    """
    chain_set = set(chain)

    # Build 1-indexed position maps (IUPAC P-14.7: locants start at 1)
    chain_pos = {atom_idx: i + 1 for i, atom_idx in enumerate(chain)}
    ring_pos = _build_ring_pos(ring_set, ring_info=ring_info)

    def _instance_locant(attachments: List[int], pos_map: dict, scaffold_set):
        """Lowest locant over FG attachment atoms reaching ``scaffold_set``.

        For each attachment atom: if the atom itself is in the scaffold
        (chain or ring), the locant is ``pos_map[atom]``. Else if it has
        a neighbour in the scaffold, take that neighbour's locant. Returns
        the ``min`` over all attachments that hit, or ``None`` if none do.
        """
        candidates = []
        for attachment in attachments:
            if attachment in scaffold_set:
                candidates.append(pos_map[attachment])
                continue
            atom = mol.GetAtomWithIdx(attachment)
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() in scaffold_set:
                    candidates.append(pos_map[nbr.GetIdx()])
                    break
        return min(candidates) if candidates else None

    # Get PG locants on chain (1-indexed IUPAC locants)
    chain_pg_locants = []
    for pg_atoms in principal_group_atoms:
        if not pg_atoms:
            continue
        # IM-01: per-FG attachment indices, then min over attachments.
        attachments = _pg_attachment_atoms(principal_group, pg_atoms)
        loc = _instance_locant(attachments, chain_pos, chain_set)
        if loc is not None:
            chain_pg_locants.append(loc)

    # Get PG locants on ring (1-indexed positional proxy)
    ring_pg_locants = []
    for pg_atoms in principal_group_atoms:
        if not pg_atoms:
            continue
        attachments = _pg_attachment_atoms(principal_group, pg_atoms)
        loc = _instance_locant(attachments, ring_pos, ring_set)
        if loc is not None:
            ring_pg_locants.append(loc)

    if not chain_pg_locants and not ring_pg_locants:
        return 0

    # Use compare_locant_sets for first-point-of-difference comparison
    # compare_locant_sets returns: -1 (a preferred), 0 (tie), 1 (b preferred)
    # chain=a, ring=b: -1 -> return 1 (chain wins), 1 -> return -1 (ring wins)
    cmp = compare_locant_sets(chain_pg_locants, ring_pg_locants)
    return -cmp


def _compare_multiple_bond_locants(
    mol, chain: List[int], ring_set: Set[int],
    ring_info: dict = None
) -> int:
    """P-44.1(g): Lowest locants for multiple bonds.

    Compares the locant sets for double and triple bonds on the chain
    versus on the ring, using first-point-of-difference (IUPAC P-14.7).

    For chain: bond between chain positions i and i+1 gets locant i+1
    (1-indexed, using the lower position per IUPAC convention).
    For ring: IUPAC ring numbering when available (via ring_info), otherwise
    sorted atom indices mapped to 1-indexed positions as fallback.
    Bond locant is the lower position of the two bonded atoms.

    Args:
        mol: RDKit Mol object
        chain: Ordered list of atom indices forming the principal chain
        ring_set: Set of atom indices in the ring system
        ring_info: Optional dict with 'iupac_locants' for IUPAC ring numbering

    Returns:
        1 if chain has lower bond locants (chain wins)
        -1 if ring has lower bond locants (ring wins)
        0 if tied or neither has multiple bonds
    """
    chain_set = set(chain)

    # Build position maps (1-indexed)
    chain_pos = {atom_idx: i + 1 for i, atom_idx in enumerate(chain)}
    ring_pos = _build_ring_pos(ring_set, ring_info=ring_info)

    # Collect bond locants for chain and ring
    chain_bond_locants = []
    ring_bond_locants = []

    for bond in mol.GetBonds():
        bt = bond.GetBondType()
        if bt != Chem.BondType.DOUBLE and bt != Chem.BondType.TRIPLE:
            continue
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        # Check if bond is on chain (both atoms in chain)
        if a in chain_set and b in chain_set:
            locant = min(chain_pos[a], chain_pos[b])
            chain_bond_locants.append(locant)

        # Check if bond is on ring (both atoms in ring)
        if a in ring_set and b in ring_set:
            locant = min(ring_pos[a], ring_pos[b])
            ring_bond_locants.append(locant)

    if not chain_bond_locants and not ring_bond_locants:
        return 0

    # Use compare_locant_sets for first-point-of-difference comparison
    # compare_locant_sets returns: -1 (a preferred), 0 (tie), 1 (b preferred)
    # We pass chain as a, ring as b:
    # -1 (chain preferred) -> return 1 (chain wins)
    # 1 (ring preferred) -> return -1 (ring wins)
    # 0 -> return 0
    cmp = compare_locant_sets(chain_bond_locants, ring_bond_locants)
    return -cmp


def _compare_substituent_locants(
    mol, chain: List[int], ring_set: Set[int],
    ring_info: dict = None
) -> int:
    """P-44.1(i): Lowest locants for substituents (detachable prefixes).

    Compares the locant sets for substituent attachment points on the chain
    versus on the ring, using first-point-of-difference (IUPAC P-14.7).

    A substituent is any non-hydrogen atom bonded to a chain/ring atom
    but NOT itself part of the chain/ring.

    For chain: substituent at chain position i gets locant i+1 (1-indexed).
    For ring: IUPAC ring numbering when available (via ring_info), otherwise
    sorted atom indices mapped to 1-indexed positions as fallback.

    Args:
        mol: RDKit Mol object
        chain: Ordered list of atom indices forming the principal chain
        ring_set: Set of atom indices in the ring system
        ring_info: Optional dict with 'iupac_locants' for IUPAC ring numbering

    Returns:
        1 if chain has lower substituent locants (chain wins)
        -1 if ring has lower substituent locants (ring wins)
        0 if tied or neither has substituents
    """
    chain_set = set(chain)

    # Build position maps (1-indexed)
    chain_pos = {atom_idx: i + 1 for i, atom_idx in enumerate(chain)}
    ring_pos = _build_ring_pos(ring_set, ring_info=ring_info)

    # Collect substituent locants for chain
    chain_sub_locants = []
    for idx in chain:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() not in chain_set and nbr.GetSymbol() != 'H':
                # This chain atom has an external substituent
                chain_sub_locants.append(chain_pos[idx])
                break  # Only count the position once, even with multiple substituents

    # Collect substituent locants for ring
    ring_sub_locants = []
    for idx in sorted(ring_set):
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() not in ring_set and nbr.GetSymbol() != 'H':
                ring_sub_locants.append(ring_pos[idx])
                break

    if not chain_sub_locants and not ring_sub_locants:
        return 0

    # compare_locant_sets: -1 (a preferred), 0 (tie), 1 (b preferred)
    # chain=a, ring=b: -1 -> return 1 (chain wins), 1 -> return -1 (ring wins)
    cmp = compare_locant_sets(chain_sub_locants, ring_sub_locants)
    return -cmp


def _count_bridging_heteroatoms(mol, chain: List[int]) -> int:
    """Count bridging heteroatoms (non-C with >=2 chain neighbours) in a
    skeletal chain. WS-A: P-51.4 admits chain replacement
    nomenclature ('2,5,8,11-tetraoxadodecane') at >= 4 hetero units — the
    no-PG skeletal-chain candidacy gate."""
    chain_set = set(chain)
    count = 0
    for idx in chain:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 6:
            continue
        chain_nbrs = sum(
            1 for nbr in atom.GetNeighbors() if nbr.GetIdx() in chain_set
        )
        if chain_nbrs >= 2:
            count += 1
    return count


def _has_bridging_heteroatom(mol, chain: List[int]) -> bool:
    """Check if a skeletal chain contains a genuine bridging heteroatom.

    A bridging heteroatom is a non-carbon atom (N, O, S) that has at least
    2 neighbors within the chain, meaning it connects two carbon segments
    (e.g., C-O-C ether, C-NH-C amine). Terminal heteroatoms from functional
    groups (e.g., C=O carbonyl oxygen) have only 1 chain neighbor and are
    NOT bridging.

    This guard prevents skeletal chains from being preferred just because
    they pick up a terminal functional group atom (.1 lesson).

    Args:
        mol: RDKit Mol object
        chain: Ordered list of atom indices forming the skeletal chain

    Returns:
        True if chain contains at least one bridging heteroatom
    """
    chain_set = set(chain)
    for idx in chain:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 6:
            continue  # Skip carbons
        chain_nbrs = sum(
            1 for nbr in atom.GetNeighbors() if nbr.GetIdx() in chain_set
        )
        if chain_nbrs >= 2:
            return True
    return False


def _compare_substituent_count(mol, chain_set: Set[int], ring_set: Set[int]) -> int:
    """P-44.1(h): Maximum number of substituents."""
    chain_subs = _count_substituents_on_atoms(mol, chain_set)
    ring_subs = _count_substituents_on_atoms(mol, ring_set)
    if chain_subs > ring_subs:
        return 1
    elif ring_subs > chain_subs:
        return -1
    return 0


def select_parent(
    mol,
    ring_systems: List[set],
    principal_chain: List[int],
    principal_group: Optional[str],
    principal_group_atoms: List[tuple],
    ring_info: dict = None,
    _offer_rank: int = 0,
) -> ParentSelectionResult:
    """
    Select parent structure per IUPAC P-44.

     G1: the staged class-specific cascade (with its fail-open
    default-to-ring) was root-cause-replaced by ONE pooled Blue Book P-44
    comparator over ring+chain candidates -- see
    ``rules/p44_scorer.select_parent_unified`` for the rule cascade
    (P-44.1.1 PG count -> P-44.1.2 senior class -> P-44.1.2.2 ring>chain
    -> P-44.2 among rings / P-44.3 among chains -> P-44.4 tiebreaks) and
    the preserved pre-empts (single-carbon PCG chain, P-31.1.3.4 natural
    products, P-51.4 admission).

    Args:
        mol: RDKit Mol object
        ring_systems: List of sets of atom indices for each ring
        principal_chain: Pre-computed principal chain (from namer.py)
        principal_group: Name of principal functional group (or None)
        principal_group_atoms: List of tuples of atom indices
        ring_info: Optional dict with 'iupac_locants' for IUPAC ring numbering
                   (from fused heterocycle data). Used by locant comparison
                   functions per ASML-19.

    Returns:
        ParentSelectionResult with decision and metadata
    """
    from .p44_scorer import select_parent_unified
    return select_parent_unified(
        mol, ring_systems, principal_chain, principal_group,
        principal_group_atoms, ring_info, _offer_rank=_offer_rank,
    )


def _count_pg_on_ring(
    mol,
    ring_atoms: Set[int],
    principal_group_atoms: List[tuple],
    principal_group: Optional[str] = None,
) -> int:
    """Count how many distinct principal group instances are on the ring.

    PSEL-01: Deduplicates by match tuple to avoid counting the same
    physical FG instance multiple times from overlapping SMARTS matches.

    IM-01: Multi-atom PGs (e.g. disulfide) count as ON the ring when ANY
    attachment atom (or its neighbour) is in the ring.
    """
    seen_matches = set()
    count = 0
    for pg_atoms in principal_group_atoms:
        if not pg_atoms:
            continue
        key = tuple(sorted(pg_atoms))
        if key in seen_matches:
            continue
        seen_matches.add(key)
        for attachment in _pg_attachment_atoms(principal_group, pg_atoms):
            # Self-check: attachment atom itself may be a ring atom
            if attachment in ring_atoms:
                count += 1
                break
            # WS-A: skeletal suffixes (-one family) count on-ring
            # ONLY by membership — no bonded-to-ring relaxation.
            if principal_group in SKELETAL_SUFFIX_PGS:
                continue
            atom = mol.GetAtomWithIdx(attachment)
            hit = False
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() in ring_atoms:
                    count += 1
                    hit = True
                    break
            if hit:
                break
    return count


def _count_pg_on_chain(
    mol,
    chain_atoms: List[int],
    principal_group_atoms: List[tuple],
    principal_group: Optional[str] = None,
) -> int:
    """Count how many distinct principal group instances are on the chain.

    PSEL-01: Deduplicates by match tuple to avoid counting the same
    physical FG instance multiple times from overlapping SMARTS matches.

    IM-01: Multi-atom PGs (e.g. disulfide) count as ON the chain when ANY
    attachment atom (or its neighbour) is in the chain.
    """
    chain_set = set(chain_atoms)
    seen_matches = set()
    count = 0
    for pg_atoms in principal_group_atoms:
        if not pg_atoms:
            continue
        key = tuple(sorted(pg_atoms))
        if key in seen_matches:
            continue
        seen_matches.add(key)
        for attachment in _pg_attachment_atoms(principal_group, pg_atoms):
            if attachment in chain_set:
                count += 1
                break
            atom = mol.GetAtomWithIdx(attachment)
            hit = False
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() in chain_set:
                    count += 1
                    hit = True
                    break
            if hit:
                break
    return count


def _ring_system_substituent_alpha_key(mol, system) -> tuple:
    """P-45.5 / P-14.5.2 alphanumeric tiebreak key for a candidate ring-system
    parent: the sorted tuple of ``alpha_sort_key`` of every substituent borne by
    the ring system. Lexicographically-lower = the ring system whose substituent
    citation is alphanumerically first (the deterministic PIN choice when two ring
    systems tie through all earlier P-44/P-45 criteria). Built robustly (any naming
    failure -> a neutral 'zzz' placeholder) so it never perturbs the reachable
    non-tie cases, where the (score, -pg_attachments) sort resolves before this key
    is consulted. D-FOLLOWON item 12 — replaces the bare RDKit-ring-enumeration
    index as the tie discriminator (the index is retained as the final, totality
    tiebreak)."""
    try:
        from ..assembly.naming_utils import alpha_sort_key
        from ..assembly.substituent_enumerator import name_substituent
    except Exception:
        return ()
    keys = []
    sysset = set(system)
    for a in system:
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            if nb.GetIdx() in sysset:
                continue
            frag, seen, stack = [], set(sysset), [nb.GetIdx()]
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                frag.append(x)
                stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(x).GetNeighbors()
                             if n.GetIdx() not in seen)
            try:
                nm = name_substituent(mol, frag, nb.GetIdx())
            except Exception:
                nm = None
            keys.append(alpha_sort_key(nm) if nm else "zzz")
    return tuple(sorted(keys))


def _select_best_ring_system(
    mol,
    ring_systems: List[set],
    principal_group_atoms: Optional[List[tuple]] = None,
    principal_group: Optional[str] = None,
) -> Tuple[set, List[set]]:
    """Select the most senior ring system from multiple candidates.

    When a molecule has disconnected ring systems (e.g., pyridine + cyclohexane),
    only the most senior ring system should be the parent. Others become
    substituents.

    Uses ring_system_score() from ring_selection.py (P-44.2) to compare.
    Tiebreaker: prefer the ring system with more principal group attachment points.

    Args:
        mol: RDKit Mol object
        ring_systems: List of sets of atom indices for each ring system
        principal_group_atoms: Optional list of FG atom tuples for tiebreaking
        principal_group: Optional FG name; when provided, attachment lookup
                         honours ``PG_ATTACHMENT_INDICES`` (IM-01).

    Returns:
        Tuple of (best_ring_system_atoms, other_ring_systems)
    """
    if len(ring_systems) <= 1:
        return (ring_systems[0] if ring_systems else set(), [])

    # Score each ring system via P-44.2 criteria
    scored = []
    for i, system in enumerate(ring_systems):
        score = ring_system_score(mol, system)

        # Tiebreaker: count principal group attachment points on this ring system
        # Weight 2 for PG atom directly IN the ring (e.g., ring ketone C=O where C is in ring)
        # Weight 1 for PG atom bonded TO a ring atom (e.g., -COOH where C(=O) is bonded to ring C)
        # This ensures direct-on-ring PG wins over adjacent-to-ring PG (PRNT-05)
        pg_attachments = 0
        if principal_group_atoms:
            for pg_atoms in principal_group_atoms:
                if not pg_atoms:
                    continue
                # IM-01: weight per FG instance is the MAX over its
                # attachment atoms (direct=2, adjacent=1, neither=0).
                instance_weight = 0
                for attachment in _pg_attachment_atoms(
                    principal_group, pg_atoms
                ):
                    if attachment in system:
                        instance_weight = max(instance_weight, 2)
                        continue
                    atom = mol.GetAtomWithIdx(attachment)
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetIdx() in system:
                            instance_weight = max(instance_weight, 1)
                            break
                pg_attachments += instance_weight

        # Append negative pg_attachments so min() prefers more attachments
        scored.append((score, -pg_attachments, i))

    scored.sort()
    best_idx = scored[0][2]

    # P-45.5 / P-14.5.2 tiebreak (D-FOLLOWON item 12): when 2+ ring systems tie on
    # the P-44.2 score AND the pg-attachment count, the winner is the one cited
    # alphanumerically first (lowest substituent alpha key), with the input index
    # kept as the final totality tiebreak — NOT the raw RDKit ring-enumeration
    # index alone (which is SMILES-order-dependent). Computed only for the tied
    # group, so the reachable non-tie cases are byte-identical. (Defensive today:
    # no nameable molecule reaches a true ring-vs-equal-ring tie — the diaryl
    # scaffolds that would fail upstream first — so this is inert on the corpus but
    # makes the parent choice a deterministic graph criterion.)
    top_score, top_pg = scored[0][0], scored[0][1]
    tied = [s for s in scored if s[0] == top_score and s[1] == top_pg]
    if len(tied) > 1:
        best_idx = min(
            tied,
            key=lambda s: (_ring_system_substituent_alpha_key(mol, ring_systems[s[2]]), s[2]),
        )[2]

    best_system = ring_systems[best_idx]
    other_systems = [ring_systems[i] for i in range(len(ring_systems)) if i != best_idx]

    return (best_system, other_systems)


def _ring_system_has_nitrogen(mol, ring_atoms: Set[int]) -> bool:
    """Check if any atom in the ring system is nitrogen."""
    for idx in ring_atoms:
        if mol.GetAtomWithIdx(idx).GetAtomicNum() == 7:
            return True
    return False
