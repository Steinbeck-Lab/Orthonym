"""Centralized substituent fragment naming module.

Provides name_substituent_fragment() -- the single entry point for naming
any substituent (linear, branched, functionalized, or ring-containing) from
its atom indices within a parent molecule.

Architecture:
  1. Fast path: linear terminal alkyl substituents use get_alkyl_name() directly.
  2. Retained PREFERRED names: phenyl, benzyl, retained cycloalkyls, tert-butyl
     (F-T9/DD6 RET-02: isopropyl/sec-butyl/isobutyl/neopentyl are NOT retained —
     their located PINs come from _located_acyclic_alkyl_name in step 3/2d).
  3. Located / recursive path: a branched or internally-attached acyclic alkyl is
     named by its own principal chain numbered from the free valence
     (_located_acyclic_alkyl_name: propan-2-yl, butan-2-yl, 2-methylpropyl);
     other fragments extract SMILES and convert via parent_to_prefix().

The function returns RAW prefix names WITHOUT enclosing marks (parentheses/brackets).
The caller (format_substituent_prefix in naming_utils.py) handles wrapping based on
is_complex_substituent() and multiplier logic.

References:
    IUPAC 2013 P-31.1.3 (substituent prefix naming)
    IUPAC 2013 P-14.5.2 (compound substituent enclosing marks)
"""

import re
import logging
from typing import List, Optional, Set

from rdkit import Chem
from ..perception.stereo import assign_stereochemistry
from .naming_utils import get_alkyl_name, SIMPLE_MULTIPLIERS, alpha_sort_key
from .fragment_naming import name_fragment_recursively

logger = logging.getLogger(__name__)


# ============================================================================
# Linear Alkyl Detection (Fast Path Guard)
# ============================================================================


def _is_linear_alkyl(mol, sub_atoms: List[int]) -> bool:
    """Check if a substituent is a straight-chain pure saturated alkyl group.

    Returns True if ALL atoms in sub_atoms are carbon, no carbon has
    more than 2 carbon neighbors within sub_atoms (i.e., no branching),
    AND all bonds between fragment atoms are single bonds.

    Chains with C=C or C#C bonds are NOT linear alkyl — they are
    alkenyl/alkynyl substituents that need the unsaturated naming path
    to produce correct prefix forms (ethenyl, prop-2-en-1-yl, ethynyl).

    This is the fast-path guard: if True, use get_alkyl_name(carbon_count)
    directly, avoiding unnecessary recursion.

    Args:
        mol: RDKit Mol object.
        sub_atoms: Atom indices of the substituent.

    Returns:
        True if the substituent is a linear (unbranched) saturated pure-carbon chain.
    """
    if not sub_atoms:
        return False

    sub_set = set(sub_atoms)

    # Reject if any atom is in a ring (cyclic substituents are not linear alkyl)
    ring_info = mol.GetRingInfo()
    for idx in sub_atoms:
        if ring_info.NumAtomRings(idx) > 0:
            return False

    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)

        # Any non-carbon atom means it's not pure alkyl
        if atom.GetSymbol() != 'C':
            return False

        # Count carbon neighbors within the substituent
        c_nbrs_in_sub = sum(
            1 for nbr in atom.GetNeighbors()
            if nbr.GetIdx() in sub_set and nbr.GetSymbol() == 'C'
        )

        # A linear chain carbon has at most 2 C neighbors within the fragment
        # (or 1 at the terminal). More than 2 means branching.
        if c_nbrs_in_sub > 2:
            return False

        # Reject if any intra-fragment bond is unsaturated (C=C or C#C).
        # These must go through the unsaturated naming path to produce
        # correct alkenyl/alkynyl names.
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in sub_set:
                bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                if bond and bond.GetBondTypeAsDouble() != 1.0:
                    return False

    return True


def _attach_is_chain_terminus(mol, sub_atoms: List[int], attach_idx) -> bool:
    """True when the substituent's attachment atom is a chain TERMINUS.

    A terminus has at most one carbon neighbour WITHIN the fragment, so the free
    valence sits at locant 1 of an unbranched chain (the elided ``-yl`` form). An
    internal attachment (2 in-fragment carbon neighbours) makes the free valence
    an interior locant (``alkan-k-yl``), which the linear fast path cannot express
    (DD5 RC-6 / SEN-04). Returns True when ``attach_idx`` is None so callers that
    lack attachment context keep their prior fast-path behaviour.
    """
    if attach_idx is None:
        return True
    sub_set = set(sub_atoms)
    c_nbrs = sum(
        1 for nbr in mol.GetAtomWithIdx(attach_idx).GetNeighbors()
        if nbr.GetIdx() in sub_set and nbr.GetSymbol() == 'C'
    )
    return c_nbrs <= 1


# ============================================================================
# Unsaturated Linear Chain Naming (IUPAC P-31.1.3)
# ============================================================================


def _name_aryl_vinyl_substituent(
    mol,
    sub_atoms: List[int],
    attach_idx: int,
    parent_set: Set[int],
) -> Optional[str]:
    """Wave2 T3c (P-31.1.3.4 / P-29.6): name an acyclic UNSATURATED all-carbon
    chain substituent that carries one or more RING substituents — the styryl /
    aryl-vinyl class ``Ar-CH=CH-`` -> ``(E)-2-phenylethenyl`` (PIN
    ``2-phenylethen-1-yl``).

    The plain ``_name_unsaturated_chain`` rejects any ring atom (its all-carbon-
    acyclic guard), and the ring chokepoint declines because the attachment is on
    the acyclic vinyl carbon, not a ring — so this positive namer fills exactly
    that gap. Fail-closed (return None) for: a saturated chain (defer to the
    located/linear namers), a branched chain, a heteroatom in the chain, a ring
    fused into the chain (attached to >1 chain carbon), or any ring the ring
    engine cannot name — the caller's Wave2 T3a ring-fragment guard then keeps
    it honest.

    Returns the raw prefix WITHOUT enclosing marks (the caller's needs_brackets
    adds them); the E/Z descriptor is prepended here (``_add_substituent_stereo``
    handles only atom R/S, not bond E/Z).
    """
    if attach_idx is None or len(sub_atoms) < 3:
        return None
    sub_set = set(sub_atoms)
    if attach_idx not in sub_set:
        return None
    from collections import deque as _dq
    ri = mol.GetRingInfo()

    # (1) The maximal acyclic all-carbon chain through the attachment: BFS over
    # non-ring carbons only. Require unbranched (<=2 chain-C neighbours each) and
    # at least one C=C / C#C.
    chain_atoms = set()
    q = _dq([attach_idx])
    while q:
        i = q.popleft()
        if i in chain_atoms:
            continue
        a = mol.GetAtomWithIdx(i)
        if a.GetSymbol() != 'C' or ri.NumAtomRings(i) > 0:
            return None
        chain_atoms.add(i)
        for nb in a.GetNeighbors():
            ni = nb.GetIdx()
            if (ni in sub_set and ni not in chain_atoms
                    and nb.GetSymbol() == 'C' and ri.NumAtomRings(ni) == 0):
                q.append(ni)
    has_unsat = False
    for ci in chain_atoms:
        cc = 0
        for nb in mol.GetAtomWithIdx(ci).GetNeighbors():
            if nb.GetIdx() in chain_atoms:
                cc += 1
                b = mol.GetBondBetweenAtoms(ci, nb.GetIdx())
                if b is not None and b.GetBondTypeAsDouble() != 1.0:
                    has_unsat = True
        if cc > 2:
            return None  # branched chain
    if not has_unsat:
        return None

    # (2) Every remaining sub atom must belong to a ring system attached to
    # EXACTLY ONE chain carbon by a single bond. Collect (chain_carbon, ring_seed).
    ring_seeds = []  # (chain_carbon_idx, ring_atom_idx)
    for ci in chain_atoms:
        for nb in mol.GetAtomWithIdx(ci).GetNeighbors():
            ni = nb.GetIdx()
            if ni in chain_atoms or ni not in sub_set:
                continue
            if ri.NumAtomRings(ni) == 0:
                return None  # a non-ring atom off the chain -> not this class
            b = mol.GetBondBetweenAtoms(ci, ni)
            if b is None or b.GetBondTypeAsDouble() != 1.0:
                return None
            ring_seeds.append((ci, ni))
    if not ring_seeds:
        return None  # no aryl arm -> plain unsaturated chain (other namer)

    # (3) Order the chain linearly; number so the free valence is lowest, then
    # lowest unsaturation locants (mirror _name_unsaturated_chain).
    terminal = None
    for ci in chain_atoms:
        if sum(1 for nb in mol.GetAtomWithIdx(ci).GetNeighbors()
               if nb.GetIdx() in chain_atoms) <= 1:
            terminal = ci
            break
    if terminal is None:
        return None
    ordered = [terminal]
    seen = {terminal}
    while len(ordered) < len(chain_atoms):
        nxt = None
        for nb in mol.GetAtomWithIdx(ordered[-1]).GetNeighbors():
            if nb.GetIdx() in chain_atoms and nb.GetIdx() not in seen:
                nxt = nb.GetIdx()
                break
        if nxt is None:
            break
        ordered.append(nxt)
        seen.add(nxt)
    if len(ordered) != len(chain_atoms):
        return None
    n = len(ordered)

    def _key(order):
        a_loc = order.index(attach_idx) + 1
        unsat = []
        for i in range(n - 1):
            b = mol.GetBondBetweenAtoms(order[i], order[i + 1])
            if b is not None and b.GetBondTypeAsDouble() in (2.0, 3.0):
                unsat.append(i + 1)
        return (a_loc, sorted(unsat))

    fwd, rev = ordered, list(reversed(ordered))
    chain = fwd if _key(fwd) <= _key(rev) else rev
    pos = {a: i + 1 for i, a in enumerate(chain)}
    a_locant = pos[attach_idx]
    double_locs, triple_locs = [], []
    for i in range(n - 1):
        b = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
        if b is None:
            continue
        bt = b.GetBondTypeAsDouble()
        if bt == 2.0:
            double_locs.append(i + 1)
        elif bt == 3.0:
            triple_locs.append(i + 1)
    stem = _build_alkenyl_name(n, a_locant, double_locs, triple_locs)
    if not stem:
        return None

    # (4) Name each ring substituent via the trustworthy ring engine + its
    # chain-position locant.
    from .substituent_enumerator import name_substituent as _name_sub
    from ..rules.ring_substituents import name_ring_system_substituent
    from collections import defaultdict as _dd
    ring_prefix_groups = _dd(list)
    for ci, seed in ring_seeds:
        ring_atoms = sorted(_ring_system_atoms(mol, seed, chain_atoms))
        rn = name_ring_system_substituent(
            mol, ring_atoms, seed, allow_enumerator_fallback=False)
        if not rn:
            return None
        ring_prefix_groups[rn].append(pos[ci])

    # (5) Assemble locant-sorted ring-substituent prefixes + stem.
    from .composer import _format_prefix_groups
    prefix = _format_prefix_groups(dict(ring_prefix_groups))
    name = f"{prefix}{stem}"

    # (6) E/Z descriptor for stereogenic chain double bonds (bond CIP, not atom).
    from ..perception.stereo import get_double_bond_stereo
    ez = []
    for sb in get_double_bond_stereo(mol):
        a, b = sb['atoms']
        if a in chain_atoms and b in chain_atoms:
            loc = min(pos[a], pos[b])
            ez.append((loc, sb['stereo']))
    if ez:
        ez.sort()
        if len(ez) == 1:
            name = f"({ez[0][1]})-{name}"
        else:
            name = "(" + ",".join(f"{l}{s}" for l, s in ez) + ")-" + name
    return name


def _ring_system_atoms(mol, seed, exclude):
    """All atoms of the ring system containing ``seed``, staying out of
    ``exclude`` (the chain). BFS over ring-member atoms and their ring
    system (fused partners), plus non-ring decorations reachable without
    crossing the chain — so a substituted aryl arm is named whole."""
    from collections import deque as _dq
    ri = mol.GetRingInfo()
    atoms = set()
    q = _dq([seed])
    while q:
        i = q.popleft()
        if i in atoms or i in exclude:
            continue
        atoms.add(i)
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            ni = nb.GetIdx()
            if ni in exclude or ni in atoms:
                continue
            # include ring atoms + any atom reachable off the ring (decorations)
            q.append(ni)
    return atoms


def _name_unsaturated_chain(
    mol,
    sub_atoms: List[int],
    attach_idx: int,
    parent_set: Set[int],
) -> Optional[str]:
    """Name an unsaturated linear chain substituent per IUPAC P-31.1.3.

    Constructs the correct alkenyl/alkynyl prefix name by:
    1. Verifying the fragment is an unbranched all-carbon chain with unsaturation
    2. Tracing the chain from the attachment point
    3. Trying both numbering directions
    4. Choosing the direction that gives lowest locant to the free-valence
       (attachment point) first, then lowest locants to unsaturation

    Examples:
        CH2=CH-  (attached at CH=)  -> ethenyl
        CH2=CH-CH2- (attached at CH2) -> prop-2-en-1-yl
        CH3-CH=CH- (attached at CH=)  -> prop-1-en-1-yl
        CH2=C(CH3)- (attached at C=) -> prop-1-en-2-yl
        HC#C- (attached at C)         -> ethynyl

    Args:
        mol: RDKit Mol object.
        sub_atoms: Atom indices of the substituent fragment.
        attach_idx: Index of the first atom of the substituent (bonded to parent).
        parent_set: Set of atom indices in the parent chain/ring.

    Returns:
        Prefix-form name (e.g., "ethenyl", "prop-2-en-1-yl") or None if this
        is not an unsaturated linear chain.
    """
    if len(sub_atoms) < 2:
        return None

    sub_set = set(sub_atoms)
    ring_info = mol.GetRingInfo()

    # Verify: all carbon, no rings, no branching, has unsaturation
    has_unsat = False
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            return None
        if ring_info.NumAtomRings(idx) > 0:
            return None
        c_nbrs = sum(
            1 for nbr in atom.GetNeighbors()
            if nbr.GetIdx() in sub_set and nbr.GetSymbol() == 'C'
        )
        if c_nbrs > 2:
            return None  # branched
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in sub_set:
                bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                if bond and bond.GetBondTypeAsDouble() != 1.0:
                    has_unsat = True

    if not has_unsat:
        return None

    carbon_count = len(sub_atoms)

    # Trace the full chain from one terminal to the other.
    # Find a terminal atom (exactly 1 neighbor in fragment).
    terminal = None
    for idx in sub_atoms:
        frag_nbrs = [
            nbr.GetIdx() for nbr in mol.GetAtomWithIdx(idx).GetNeighbors()
            if nbr.GetIdx() in sub_set
        ]
        if len(frag_nbrs) == 1:
            terminal = idx
            break

    if terminal is None:
        return None  # no terminal found (cycle?), shouldn't happen

    ordered = [terminal]
    visited = {terminal}
    current = terminal
    while len(ordered) < carbon_count:
        atom = mol.GetAtomWithIdx(current)
        next_atom = None
        for nbr in atom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni in sub_set and ni not in visited:
                next_atom = ni
                break
        if next_atom is None:
            break
        ordered.append(next_atom)
        visited.add(next_atom)
        current = next_atom

    if len(ordered) != carbon_count:
        return None

    # Try both numbering directions and pick the best per IUPAC rules.
    fwd = ordered
    rev = list(reversed(ordered))

    best_name = None
    best_key = None

    for chain in [fwd, rev]:
        try:
            a_locant = chain.index(attach_idx) + 1
        except ValueError:
            continue

        # Collect double/triple bond locants (lower-numbered atom in each bond)
        double_locs = []
        triple_locs = []
        for i in range(len(chain) - 1):
            bond = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
            if bond is None:
                continue
            bt = bond.GetBondTypeAsDouble()
            if bt == 2.0:
                double_locs.append(i + 1)
            elif bt == 3.0:
                triple_locs.append(i + 1)

        all_unsat = sorted(double_locs + triple_locs)
        # IUPAC P-31.1.3.4: lowest to free valence first, then to unsaturation
        key = (a_locant, all_unsat)

        if best_key is None or key < best_key:
            best_key = key
            # Build the name for this direction
            best_name = _build_alkenyl_name(
                carbon_count, a_locant, double_locs, triple_locs
            )

    return best_name


def _build_alkenyl_name(
    carbon_count: int,
    attach_locant: int,
    double_locants: List[int],
    triple_locants: List[int],
) -> str:
    """Build an alkenyl/alkynyl prefix name from chain data.

    Args:
        carbon_count: Number of carbons in the substituent chain.
        attach_locant: 1-indexed position of the free valence (attachment point).
        double_locants: Sorted list of double bond locants.
        triple_locants: Sorted list of triple bond locants.

    Returns:
        Prefix-form name, e.g., "ethenyl", "prop-2-en-1-yl", "ethynyl".
    """
    from ..data.chain_names import get_chain_prefix

    stem = get_chain_prefix(carbon_count)
    double_locants = sorted(double_locants)
    triple_locants = sorted(triple_locants)

    MULT = {2: "di", 3: "tri", 4: "tetra", 5: "penta"}

    # For 2-carbon chains: no locants needed for unsaturation
    if carbon_count == 2:
        if double_locants:
            return f"{stem}enyl"
        if triple_locants:
            return f"{stem}ynyl"
        return f"{stem}yl"

    # For longer chains: build with locants using structured assembly.
    # Each segment is (locant_str, multiplier, bond_suffix).
    # The 'a' euphonic connector is added when multiple bonds have multiplied
    # locants (diene, diyne) per IUPAC P-31.1.3.4.
    num_double = len(double_locants)
    num_triple = len(triple_locants)
    needs_a = (num_double > 1) or (num_triple > 1 and num_double == 0)

    segments = []
    if double_locants:
        loc_str = ",".join(str(l) for l in double_locants)
        mult = MULT.get(num_double, str(num_double)) if num_double > 1 else ""
        segments.append((loc_str, mult, "en"))

    if triple_locants:
        loc_str = ",".join(str(l) for l in triple_locants)
        mult = MULT.get(num_triple, str(num_triple)) if num_triple > 1 else ""
        segments.append((loc_str, mult, "yn"))

    # Assemble infix: "a" (if needed) then "-locants-[mult]bond" per segment
    infix = ""
    if needs_a:
        infix = "a"
    for loc_str, mult, bond in segments:
        infix += f"-{loc_str}-{mult}{bond}"

    if not needs_a and infix:
        # infix already starts with "-" from the first segment
        pass
    elif not infix:
        infix = ""

    name = f"{stem}{infix}-{attach_locant}-yl"

    return name


_HALOGEN_PREFIX = {"F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}


def _name_saturated_substituted_chain(
    mol,
    sub_atoms: List[int],
    attach_idx: int,
    parent_set: Set[int],
) -> Optional[str]:
    """Name a saturated linear carbon chain bearing only halogen substituents,
    numbered from the attachment point (IUPAC P-46.1.8: the free valence takes
    locant 1; P-46 substituents are numbered from there).

    Fixes the DEF-8 / §3.5 defect: the recursive path (Step 3-5) extracts the
    fragment and renames it as a FREE molecule, losing the attachment constraint
    ('CCCCCl' -> '1-chlorobutane' -> 'chlorobutyl', locant dropped). Here the
    backbone is numbered so the attachment carbon is locant 1, so a chloro four
    carbons away is '4-' -> '4-chlorobutyl'; 'ClCC(Cl)C-' -> '2,3-dichloropropyl';
    '-CH2CH2Cl' -> '2-chloroethyl'.

    Returns None (fall through to the richer recursive path) for anything that is
    not a LINEAR all-carbon backbone + terminal halogens with a TERMINAL (primary)
    attachment — branched backbones, other heteroatoms, rings, and secondary
    attachment keep their existing naming.
    """
    if not sub_atoms:
        return None
    sub_set = set(sub_atoms)
    ring_info = mol.GetRingInfo()

    backbone = []                # carbon backbone atom indices
    halogens = []                # (halogen_idx, attached_carbon_idx)
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if ring_info.NumAtomRings(idx) > 0:
            return None          # ring -> not a simple chain
        sym = atom.GetSymbol()
        if sym == 'C':
            backbone.append(idx)
        elif sym in _HALOGEN_PREFIX:
            c_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in sub_set]
            if len(c_nbrs) != 1:
                return None      # bridging/abnormal halogen
            halogens.append((idx, c_nbrs[0]))
        else:
            return None          # any other heteroatom -> richer case, fall through

    if not backbone or not halogens:
        return None              # plain alkyl is the fast path; need >=1 halogen here
    backbone_set = set(backbone)
    if any(c not in backbone_set for _h, c in halogens):
        return None

    # Linear backbone: each backbone carbon has <= 2 backbone-carbon neighbours.
    for idx in backbone:
        c_nbrs = sum(1 for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                     if n.GetIdx() in backbone_set)
        if c_nbrs > 2:
            return None          # branched backbone -> fall through

    # Attachment must be a backbone terminal (primary substituent).
    if attach_idx not in backbone_set:
        return None
    attach_c_nbrs = sum(1 for n in mol.GetAtomWithIdx(attach_idx).GetNeighbors()
                        if n.GetIdx() in backbone_set)
    if attach_c_nbrs != 1:
        return None              # secondary/internal attachment -> fall through

    # Trace the backbone from the attachment terminal (attach = locant 1).
    ordered = [attach_idx]
    visited = {attach_idx}
    current = attach_idx
    while len(ordered) < len(backbone):
        nxt = None
        for n in mol.GetAtomWithIdx(current).GetNeighbors():
            ni = n.GetIdx()
            if ni in backbone_set and ni not in visited:
                nxt = ni
                break
        if nxt is None:
            break
        ordered.append(nxt)
        visited.add(nxt)
        current = nxt
    if len(ordered) != len(backbone):
        return None              # disconnected backbone -> fall through

    pos = {idx: i + 1 for i, idx in enumerate(ordered)}   # attach = 1

    from collections import defaultdict
    from ..data.chain_names import get_chain_prefix
    groups: dict = defaultdict(list)
    for h_idx, c_idx in halogens:
        groups[_HALOGEN_PREFIX[mol.GetAtomWithIdx(h_idx).GetSymbol()]].append(pos[c_idx])

    _MULT = {1: "", 2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa"}
    # Alphabetical order of the (base) halogen prefixes (P-14.5.1: the di/tri
    # multiplier on a simple substituent is ignored for ordering).
    part_strings = []
    for prefix in sorted(groups.keys()):
        locs = sorted(groups[prefix])
        loc_str = ",".join(str(loc) for loc in locs)
        mult = _MULT.get(len(locs), f"{len(locs)}")
        part_strings.append(f"{loc_str}-{mult}{prefix}")

    stem = get_chain_prefix(len(backbone))
    return f"{'-'.join(part_strings)}{stem}yl"


# Single-atom detachable prefixes this namer enumerates from STRUCTURE, keyed
# by (element, role). Carboxy is multi-atom (handled separately: its carbon is
# NOT a backbone carbon — P-65.1.1 expresses -COOH on a substituent as the
# detachable prefix "carboxy", excluding the acid carbon from the parent chain).
_POLYFUNC_OXO_PREFIX = "oxo"
_POLYFUNC_HYDROXY_PREFIX = "hydroxy"
_POLYFUNC_AMINO_PREFIX = "amino"
_POLYFUNC_CARBOXY_PREFIX = "carboxy"


def _name_carbamoylamino_chain_substituent(
    mol, sub_atoms: List[int], attach_idx: int,
) -> Optional[str]:
    """P-66.1.6.1.1.3 (BB 33338/55463, W2E-P1FG Task 11): an unbranched
    saturated all-carbon chain rooted at ``attach_idx`` and terminated by a
    single -NH-C(=O)-NH2 unit -> '{loc}-(carbamoylamino){chain}yl' ('not
    ureido'), e.g. -CH2CH2CH2-NH-C(=O)-NH2 -> '3-(carbamoylamino)propyl'.

    Fail-closed: exactly one urea unit, a terminal unsubstituted NH2, an
    unsubstituted bridge NH, and a pure linear carbon chain from the free
    valence; anything else -> None (caller falls through)."""
    sub_set = set(sub_atoms)
    if attach_idx not in sub_set:
        return None
    if any(mol.GetAtomWithIdx(i).IsInRing() for i in sub_set):
        return None
    # Locate the urea unit: an -NH- bridge N bonded to a carbamoyl C(=O)NH2.
    bridge_n = carbonyl_c = term_n = None
    for i in sub_set:
        a = mol.GetAtomWithIdx(i)
        if a.GetAtomicNum() != 7 or a.GetFormalCharge() != 0:
            continue
        heavy = [nb for nb in a.GetNeighbors() if nb.GetAtomicNum() > 1]
        # bridge N: exactly one C-chain neighbour + one carbonyl-C neighbour
        c_nbrs = [nb for nb in heavy if nb.GetAtomicNum() == 6]
        if len(c_nbrs) != 2 or a.GetTotalNumHs() != 1:
            continue
        for _c in c_nbrs:
            o2 = [n for n in _c.GetNeighbors() if n.GetAtomicNum() == 8 and
                  mol.GetBondBetweenAtoms(_c.GetIdx(), n.GetIdx())
                     .GetBondTypeAsDouble() == 2.0]
            nh2 = [n for n in _c.GetNeighbors() if n.GetAtomicNum() == 7
                   and n.GetIdx() != i and n.GetDegree() == 1
                   and n.GetTotalNumHs() == 2 and n.GetFormalCharge() == 0]
            if len(o2) == 1 and len(nh2) == 1 and _c.GetDegree() == 3:
                bridge_n, carbonyl_c, term_n = i, _c.GetIdx(), nh2[0].GetIdx()
                o_idx = o2[0].GetIdx()
                break
        if bridge_n is not None:
            break
    if bridge_n is None:
        return None
    urea_atoms = {bridge_n, carbonyl_c, term_n, o_idx}
    # The remaining fragment atoms form the carbon chain from the free valence.
    chain = [i for i in sub_set if i not in urea_atoms]
    if not chain or any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in chain):
        return None
    # Linear chain: trace from attach (free valence = locant 1) to bridge_n.
    ordered = [attach_idx]
    visited = {attach_idx}
    cur = attach_idx
    while len(ordered) < len(chain):
        nxt = None
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            ni = nb.GetIdx()
            if ni in chain and ni not in visited:
                nxt = ni
                break
        if nxt is None:
            return None
        ordered.append(nxt)
        visited.add(nxt)
        cur = nxt
    if len(ordered) != len(chain):
        return None
    # The last chain carbon must bear the bridge N (urea at the far end).
    if bridge_n not in [n.GetIdx() for n in
                        mol.GetAtomWithIdx(ordered[-1]).GetNeighbors()]:
        return None
    # No branching on the chain (each interior C bonds exactly 2 chain C).
    for pos, ci in enumerate(ordered):
        c_in_chain = sum(1 for nb in mol.GetAtomWithIdx(ci).GetNeighbors()
                         if nb.GetIdx() in chain)
        max_chain = 1 if pos in (0, len(ordered) - 1) else 2
        if c_in_chain > max_chain:
            return None
    loc = len(ordered)  # bridge-N carbon is the far end
    from ..data.chain_names import get_chain_prefix
    stem = get_chain_prefix(len(chain))
    return f"{loc}-(carbamoylamino){stem}yl"


def _name_polyfunctional_acyclic_substituent(
    mol,
    sub_atoms: List[int],
    attach_idx: int,
    parent_set: Set[int],
) -> Optional[str]:
    """Name an acyclic SATURATED carbon-chain substituent bearing >=1 simple
    detachable functional-group prefixes (carboxy / amino / hydroxy / oxo /
    halogen), numbered from the free valence (P-29.2 / P-31.1.4.3.4: the free
    valence is locant 1; every characteristic group on a substituent is cited
    as a detachable prefix per P-29.3.2 / P-65.1.1).

    ROOT-CAUSE fix for the substituent structure-loss / locant-drop bug: the
    recursive Tier-4 path names the H-capped fragment as a FREE molecule
    ('(2R)-2-aminopropanoic acid' / 'propan-1-ol' / 'acetic acid') and
    ``parent_to_prefix`` then (a) rebuilds the stem from the carbon COUNT alone,
    DROPPING every secondary prefix (-> '(R)-2-carboxyethyl', the amino lost),
    (b) inherits the parent's LOWEST-LOCANT numbering instead of re-numbering
    from the free valence (-CH2CH2CH2OH -> 'propan-1-ol' -> '1-hydroxypropyl',
    a WRONG locant — the hydroxy is 3 C from the attachment), or (c) maps a
    1-carbon acid to its retained acyl name (-CH2COOH -> 'acetic acid' ->
    'acetyl', a DIFFERENT MOLECULE) — while the Tier-2 cache maps the capped
    fragment to a whole-molecule retained name ('alanine' -> 'alaninyl'). All
    lose structure or mis-locate. Here every group on the chain is enumerated
    directly FROM STRUCTURE, numbered from the free valence, so nothing is
    dropped and the locants are attachment-correct.

    Fail-closed: returns None (caller falls through unchanged) for rings,
    unsaturated or branched backbones, non-primary (internal) attachment,
    secondary/tertiary amines, amides / esters / ethers / thioethers / nitriles
    (existing tiers own those), charged atoms, any unrecognised atom (S / P / B
    / ...), or a single oxo on the free-valence carbon (= an acyl group, named
    acetyl / propanoyl by the retained/acyl tier). Single-FG forms that were
    already correct elsewhere stay byte-identical: a 1-carbon backbone elides the
    locant ('hydroxymethyl' / 'aminomethyl' / 'carboxymethyl'), and pure-halogen
    chains are handled upstream by the halogen path (Step 2c) before this runs.

    Worked examples:
        ``-CH2-CH(NH2)-COOH`` -> "2-amino-2-carboxyethyl"  (P-65.1.1, ChEBI PS)
        ``-CH2-CH2-OH``       -> "2-hydroxyethyl"           (locant restored)
        ``-CH2-CH2-CH2-OH``   -> "3-hydroxypropyl"          (locant corrected)
        ``-CH2-COOH``         -> "carboxymethyl"            (was 'acetyl')
        ``-CH2-C(=O)-CH3``    -> "2-oxopropyl"              (oxo, not on attach)
    """
    if not sub_atoms or attach_idx is None:
        return None
    sub_set = set(sub_atoms)
    if attach_idx not in sub_set:
        return None
    ring_info = mol.GetRingInfo()
    if any(ring_info.NumAtomRings(i) > 0 for i in sub_set):
        return None  # ring fragment -> the ring engine owns it

    consumed: Set[int] = set()      # atoms absorbed by a carboxy group
    # backbone_carbon_idx -> list of prefix strings sitting on that carbon
    prefix_on: dict = {}

    def _add_prefix(carbon_idx: int, prefix: str) -> None:
        prefix_on.setdefault(carbon_idx, []).append(prefix)

    # ---- Pass 1: carboxy (-C(=O)OH / -C(=O)O-). The carboxy carbon and its two
    # oxygens are consumed; the prefix is recorded on the carboxy carbon's single
    # carbon neighbour (a backbone carbon). ----
    for idx in sub_set:
        a = mol.GetAtomWithIdx(idx)
        if a.GetSymbol() != 'C' or a.GetFormalCharge() != 0:
            continue
        dbl_o = []
        oh_o = []
        c_nbr = []
        clean = True
        for b in a.GetBonds():
            nb = b.GetOtherAtom(a)
            ni = nb.GetIdx()
            if ni not in sub_set:
                # only the free-valence bond (at the attachment atom) may leave
                # the fragment; a carboxy carbon is never the attachment here
                clean = clean and (idx == attach_idx)
                continue
            bt = b.GetBondType()
            if nb.GetSymbol() == 'O' and bt == Chem.BondType.DOUBLE:
                dbl_o.append(ni)
            elif nb.GetSymbol() == 'O' and bt == Chem.BondType.SINGLE:
                oh_o.append(nb)
            elif nb.GetSymbol() == 'C' and bt == Chem.BondType.SINGLE:
                c_nbr.append(ni)
            else:
                clean = False
        if not clean:
            continue
        if len(dbl_o) == 1 and len(oh_o) == 1 and len(c_nbr) == 1:
            o_single = oh_o[0]
            if (o_single.GetDegree() == 1
                    and (o_single.GetTotalNumHs() >= 1
                         or o_single.GetFormalCharge() < 0)):
                consumed.add(idx)
                consumed.add(dbl_o[0])
                consumed.add(o_single.GetIdx())
                _add_prefix(c_nbr[0], _POLYFUNC_CARBOXY_PREFIX)

    # Backbone = every carbon not consumed by a carboxy group.
    backbone = [
        i for i in sub_set
        if mol.GetAtomWithIdx(i).GetSymbol() == 'C' and i not in consumed
    ]
    backbone_set = set(backbone)
    if not backbone:
        return None
    if attach_idx not in backbone_set:
        return None  # attachment is inside a carboxy/prefix group -> decline

    # Carboxy must attach to a backbone carbon (else the fragment is malformed).
    for c_idx in list(prefix_on):
        if c_idx not in backbone_set:
            return None

    # ---- Pass 2: every remaining (non-consumed) atom must be a recognised
    # single-atom prefix on a backbone carbon; backbone carbons must be saturated
    # and bond only to backbone carbons / recognised prefix atoms. ----
    fg_count = sum(len(v) for v in prefix_on.values())  # carboxy groups so far
    for idx in sub_set:
        if idx in consumed:
            continue
        a = mol.GetAtomWithIdx(idx)
        sym = a.GetSymbol()
        if a.GetFormalCharge() != 0:
            return None
        if idx in backbone_set:
            # Validate this backbone carbon's bonds.
            for b in a.GetBonds():
                nb = b.GetOtherAtom(a)
                ni = nb.GetIdx()
                bt = b.GetBondType()
                if ni not in sub_set:
                    if idx != attach_idx:
                        return None      # only the free valence leaves the frag
                    continue
                if ni in consumed:
                    continue             # bond into the carboxy group (the C=O–OH)
                if bt == Chem.BondType.SINGLE:
                    continue             # to backbone C or to a prefix heteroatom
                if (bt == Chem.BondType.DOUBLE and nb.GetSymbol() == 'O'
                        and ni not in backbone_set):
                    continue             # ketone/aldehyde C=O (oxo) — handled below
                if (bt == Chem.BondType.DOUBLE
                        and nb.GetSymbol() in ('S', 'Se', 'Te')
                        and ni not in backbone_set):
                    continue             # C=S/Se/Te ylidene (Wave-2 C) — handled below
                return None              # C=C / C#C / C#N / C=N etc. -> decline
            continue
        # Non-backbone, non-consumed heavy atom -> must be a simple prefix.
        if a.GetIsotope():
            return None
        # find its single backbone-carbon neighbour
        heavy_nbrs = [n for n in a.GetNeighbors()]
        c_host = [n.GetIdx() for n in heavy_nbrs
                  if n.GetIdx() in backbone_set]
        in_frag_nbrs = [n.GetIdx() for n in heavy_nbrs if n.GetIdx() in sub_set]
        if sym == 'O':
            # hydroxy (-OH, single bond, terminal) or oxo (=O on a backbone C)
            if len(in_frag_nbrs) != 1 or len(c_host) != 1:
                return None
            bond = mol.GetBondBetweenAtoms(idx, c_host[0])
            if bond.GetBondType() == Chem.BondType.SINGLE:
                if a.GetTotalNumHs() < 1:
                    return None          # alkoxide / ether-like -> decline
                _add_prefix(c_host[0], _POLYFUNC_HYDROXY_PREFIX)
                fg_count += 1
            elif bond.GetBondType() == Chem.BondType.DOUBLE:
                if a.GetTotalNumHs() != 0:
                    return None
                _add_prefix(c_host[0], _POLYFUNC_OXO_PREFIX)
                fg_count += 1
            else:
                return None
        elif sym == 'N':
            # primary amine -NH2 only (neutral, terminal, single bond, 2 H);
            # the host carbon must NOT be a carbonyl (that is an amide).
            if len(in_frag_nbrs) != 1 or len(c_host) != 1:
                return None
            if a.GetTotalNumHs() != 2:
                return None
            bond = mol.GetBondBetweenAtoms(idx, c_host[0])
            if bond.GetBondType() != Chem.BondType.SINGLE:
                return None
            host = mol.GetAtomWithIdx(c_host[0])
            if any(bb.GetBondType() == Chem.BondType.DOUBLE
                   and bb.GetOtherAtom(host).GetSymbol() == 'O'
                   for bb in host.GetBonds()):
                return None              # -C(=O)NH2 amide -> decline (carbamoyl)
            _add_prefix(c_host[0], _POLYFUNC_AMINO_PREFIX)
            fg_count += 1
        elif sym in _HALOGEN_PREFIX:
            if len(in_frag_nbrs) != 1 or len(c_host) != 1:
                return None
            bond = mol.GetBondBetweenAtoms(idx, c_host[0])
            if bond.GetBondType() != Chem.BondType.SINGLE:
                return None
            _add_prefix(c_host[0], _HALOGEN_PREFIX[sym])
            fg_count += 1
        elif sym in ('S', 'Se', 'Te'):
            # Wave-2 completion C (P-64.6.1): a TERMINAL =S/=Se/=Te on a
            # backbone C is the chalcogenylidene prefix (2-sulfanylidenebutyl,
            # BB P-64.7.3 verbatim witness). Single-bonded chalcogens
            # (sulfanyl / thioether) stay declined — other tiers own them.
            if len(in_frag_nbrs) != 1 or len(c_host) != 1:
                return None
            bond = mol.GetBondBetweenAtoms(idx, c_host[0])
            if (bond.GetBondType() != Chem.BondType.DOUBLE
                    or a.GetTotalNumHs() != 0):
                return None
            _add_prefix(c_host[0], {'S': 'sulfanylidene',
                                    'Se': 'selanylidene',
                                    'Te': 'tellanylidene'}[sym])
            fg_count += 1
        else:
            return None                  # P / B / etc. -> decline

    # A single carbon bearing BOTH oxo and hydroxy is a carboxyl carbon (-C(=O)OH)
    # that Pass-1 did not consume as 'carboxy' (e.g. its only non-O neighbour is the
    # non-carbon parent — a carbamic acid C-N). Never emit 'hydroxyoxomethyl' for a
    # carboxylic acid: fail-closed so another tier names it (P-65.1.1).
    _YLIDENES = ('sulfanylidene', 'selanylidene', 'tellanylidene')
    for _prefixes in prefix_on.values():
        if _POLYFUNC_OXO_PREFIX in _prefixes and _POLYFUNC_HYDROXY_PREFIX in _prefixes:
            return None
        # Chalcogenylidene + hydroxy/oxo on ONE carbon = a thioic-acid-type
        # carbon (Wave-2 C) — other tiers own those; never 'hydroxysulfanylidene...'.
        if any(y in _prefixes for y in _YLIDENES) and (
                _POLYFUNC_HYDROXY_PREFIX in _prefixes
                or _POLYFUNC_OXO_PREFIX in _prefixes):
            return None

    if fg_count < 1:
        return None  # no detachable group -> plain alkyl, the fast/located tiers own it

    # An oxo on the FREE-VALENCE (attachment) carbon makes the fragment an ACYL
    # group (-C(=O)-R): its PIN is the acyl prefix (acetyl / propanoyl / formyl,
    # P-66.6.3 / P-65.3.1), NOT '1-oxoalkyl'. The single-FG path (fg_count == 1)
    # would otherwise mis-name a bare acyl substituent, so fail-closed and let the
    # retained / acyl tier own it. (The >=2-FG behaviour shipped in a8dd06ff is
    # left untouched: an acyl bearing a second detachable group is a rare edge that
    # those golds do not exercise -> follow-on.)
    if fg_count == 1 and _POLYFUNC_OXO_PREFIX in prefix_on.get(attach_idx, []):
        return None
    # Same rule for a thioacyl attachment (-C(=S)-R = alkanethioyl, Wave-2 C).
    if fg_count == 1 and any(
            y in prefix_on.get(attach_idx, []) for y in _YLIDENES):
        return None

    # ---- Backbone must be a single linear chain attached at a terminus. ----
    for idx in backbone:
        n_bb = sum(1 for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                   if n.GetIdx() in backbone_set)
        if n_bb > 2:
            return None  # branched backbone -> follow-on (located polyfunctional)
    n_attach_bb = sum(1 for n in mol.GetAtomWithIdx(attach_idx).GetNeighbors()
                      if n.GetIdx() in backbone_set)
    if n_attach_bb > 1:
        return None  # internal/secondary attachment -> follow-on

    # Trace the chain from the attachment terminus (attach = locant 1).
    ordered = [attach_idx]
    visited = {attach_idx}
    current = attach_idx
    while len(ordered) < len(backbone):
        nxt = None
        for n in mol.GetAtomWithIdx(current).GetNeighbors():
            ni = n.GetIdx()
            if ni in backbone_set and ni not in visited:
                nxt = ni
                break
        if nxt is None:
            break
        ordered.append(nxt)
        visited.add(nxt)
        current = nxt
    if len(ordered) != len(backbone):
        return None  # disconnected backbone -> decline

    pos = {idx: i + 1 for i, idx in enumerate(ordered)}  # attach = 1
    single_position = (len(backbone) == 1)

    from collections import defaultdict
    from ..data.chain_names import get_chain_prefix
    groups: dict = defaultdict(list)  # prefix -> sorted locant list
    for c_idx, prefixes in prefix_on.items():
        for pfx in prefixes:
            groups[pfx].append(pos[c_idx])

    _MULT = {1: "", 2: "di", 3: "tri", 4: "tetra",
             5: "penta", 6: "hexa"}
    parts = []
    for prefix in sorted(groups.keys(), key=alpha_sort_key):
        locs = sorted(groups[prefix])
        mult = _MULT.get(len(locs), SIMPLE_MULTIPLIERS.get(len(locs), ""))
        if single_position:
            parts.append(f"{mult}{prefix}")           # methyl: locant elided
        else:
            loc_str = ",".join(str(loc) for loc in locs)
            parts.append(f"{loc_str}-{mult}{prefix}")

    stem = get_chain_prefix(len(backbone))
    # Join consecutive prefix parts; a leading digit after a non-digit needs a
    # hyphen ("...amino-2-carboxy..."), handled by the locant prefix itself.
    return f"{''.join(_joined_prefix_parts(parts))}{stem}yl"


def _joined_prefix_parts(parts: List[str]) -> List[str]:
    """Join alphabetised prefix parts inserting a hyphen between a letter and a
    following locant digit (e.g. 'amino' + '2-carboxy' -> 'amino-2-carboxy')."""
    out = []
    for i, p in enumerate(parts):
        if i > 0 and out and out[-1][-1].isalpha() and p[:1].isdigit():
            out.append("-")
        out.append(p)
    return out


def _name_ether_substituted_chain(
    mol,
    sub_atoms: List[int],
    attach_idx: int,
    parent_set: Set[int],
) -> Optional[str]:
    """Name a saturated all-carbon chain substituent bearing ether ``-O-R``
    group(s), numbered from the attachment point (free valence = locant 1,
    P-29.2 / P-46; the ether expressed as an (R-oxy) prefix, P-63.2.2.2).

    v22 C-T2 (V-3, V1 constitutional-fidelity theme). Fixes the defect where the
    recursive path (Step 3-4) caps the fragment to a FREE molecule whose retained
    name is then naively ``-yl``-ed, losing the attachment locus:
    ``-CH2-O-C6H5`` -> capped ``anisole`` -> ``anisolyl`` (a DIFFERENT
    constitution — implies the free valence is on the ring). Here the backbone
    carbon bearing each ether is numbered from the free valence and each ``-O-R``
    is named via :func:`get_alkoxy_prefix`:

      - ``-CH2-O-C6H5``   -> ``phenoxymethyl``
      - ``-CH2-O-CH3``    -> ``methoxymethyl``
      - ``-CH2CH2-O-C6H5``-> ``2-phenoxyethyl``

    Returns ``None`` (fall through to the richer recursive path — fail-closed)
    for anything that is not a LINEAR all-carbon backbone with a TERMINAL
    (primary) attachment whose only heteroatoms are neutral, divalent, acyclic
    ether oxygens, each bridging exactly one backbone carbon to a nameable R
    group (no other functional groups, no branches, no unsaturation, no rings on
    the backbone).
    """
    if not sub_atoms:
        return None
    sub_set = set(sub_atoms)
    ring_info = mol.GetRingInfo()

    attach_atom = mol.GetAtomWithIdx(attach_idx)
    if attach_atom.GetSymbol() != 'C' or ring_info.NumAtomRings(attach_idx) > 0:
        return None  # attach via heteroatom / ring -> not this handler

    # Identify ether-type links: neutral, divalent, acyclic, both neighbours in
    # the fragment, both bonds single. P-63.2.5/P-29.5.2: O -> (R)oxy prefix, S
    # -> (R)sulfanyl prefix (W2E-P1FC Task 8, the -CH2-S-R concatenation). Any
    # other decoration (charged, =O, -OH, ring, peroxide -O-O-/-S-S-) disqualifies
    # the whole fragment (fail-closed).
    ether_os: List[int] = []
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()
        if sym == 'C':
            continue
        if sym not in ('O', 'S'):
            return None  # any non-C, non-O/S heteroatom -> richer case
        if (atom.GetFormalCharge() != 0 or atom.GetTotalNumHs() != 0
                or atom.GetDegree() != 2 or ring_info.NumAtomRings(idx) > 0):
            return None
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        if not all(n in sub_set for n in nbrs):
            return None
        if any(mol.GetBondBetweenAtoms(idx, n).GetBondType()
               != Chem.BondType.SINGLE for n in nbrs):
            return None
        # Exclude chalcogen-chalcogen catenation (peroxide/disulfide): that is a
        # (R)peroxy/(R)disulfanyl, owned by a different producer.
        if any(mol.GetAtomWithIdx(n).GetSymbol() in ('O', 'S') for n in nbrs):
            return None
        ether_os.append(idx)
    if not ether_os:
        return None  # plain alkyl is the fast path; need >=1 ether here
    ether_set = set(ether_os)

    # Backbone = carbons reachable from the attachment WITHOUT crossing an ether
    # oxygen. Must be all-carbon, acyclic, saturated (single C-C bonds only).
    backbone: List[int] = []
    seen = {attach_idx}
    stack = [attach_idx]
    while stack:
        cur = stack.pop()
        cur_atom = mol.GetAtomWithIdx(cur)
        if cur_atom.GetSymbol() != 'C' or ring_info.NumAtomRings(cur) > 0:
            return None
        backbone.append(cur)
        for nbr in cur_atom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni in ether_set or ni not in sub_set or ni in seen:
                continue
            if nbr.GetSymbol() != 'C':
                return None  # non-C, non-ether neighbour on the backbone
            bond = mol.GetBondBetweenAtoms(cur, ni)
            if bond.GetBondType() != Chem.BondType.SINGLE:
                return None  # unsaturated backbone -> fall through
            seen.add(ni)
            stack.append(ni)
    backbone_set = set(backbone)

    # Linear backbone, attachment at a terminal (primary) carbon.
    for idx in backbone:
        c_nbrs = sum(1 for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                     if n.GetIdx() in backbone_set)
        if c_nbrs > 2:
            return None  # branched backbone -> fall through
    attach_c_nbrs = sum(1 for n in attach_atom.GetNeighbors()
                        if n.GetIdx() in backbone_set)
    if attach_c_nbrs > 1:
        return None  # secondary/internal attachment -> fall through

    # Every fragment atom must be backbone, an ether O, or reachable only through
    # an ether O (i.e. part of an R group). Map each ether O to the backbone
    # carbon it decorates + its R-side neighbour.
    accounted = backbone_set | ether_set
    ether_links: List[tuple] = []  # (backbone_carbon, ether_O, r_side_atom)
    for o_idx in ether_os:
        nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(o_idx).GetNeighbors()]
        bb_side = [n for n in nbrs if n in backbone_set]
        r_side = [n for n in nbrs if n not in backbone_set]
        if len(bb_side) != 1 or len(r_side) != 1:
            return None  # O not bridging backbone -> R (e.g. R-O-R' both off backbone)
        ether_links.append((bb_side[0], o_idx, r_side[0]))
        # Walk the R group, marking atoms accounted-for.
        r_stack = [r_side[0]]
        r_seen = {o_idx}
        while r_stack:
            ra = r_stack.pop()
            if ra in r_seen:
                continue
            r_seen.add(ra)
            accounted.add(ra)
            for n in mol.GetAtomWithIdx(ra).GetNeighbors():
                if n.GetIdx() in sub_set and n.GetIdx() not in r_seen:
                    r_stack.append(n.GetIdx())
    if accounted != sub_set:
        return None  # unaccounted atoms -> richer case, fall through

    # Number the backbone from the attachment terminal (attach = locant 1).
    ordered = [attach_idx]
    visited = {attach_idx}
    current = attach_idx
    while len(ordered) < len(backbone):
        nxt = None
        for n in mol.GetAtomWithIdx(current).GetNeighbors():
            ni = n.GetIdx()
            if ni in backbone_set and ni not in visited:
                nxt = ni
                break
        if nxt is None:
            break
        ordered.append(nxt)
        visited.add(nxt)
        current = nxt
    if len(ordered) != len(backbone):
        return None  # disconnected backbone -> fall through
    pos = {idx: i + 1 for i, idx in enumerate(ordered)}

    # Name each ether-type link as an (R-oxy)/(R-sulfanyl) prefix at its
    # backbone-carbon locant.
    from collections import defaultdict
    from ..data.chain_names import get_chain_prefix
    from .substituent_prefix_forms import get_alkoxy_prefix, get_sulfanyl_prefix
    from .naming_utils import is_complex_substituent, apply_enclosing_marks
    groups: dict = defaultdict(list)
    for bb_c, o_idx, r_side in ether_links:
        if mol.GetAtomWithIdx(o_idx).GetSymbol() == 'O':
            oxy = get_alkoxy_prefix(mol, (o_idx, bb_c, r_side), backbone)
            if not oxy or oxy == "alkoxy":
                return None  # un-nameable R -> fall through (fail-closed)
        else:  # S -> (R)sulfanyl (P-29.5.2 concatenation)
            # Name the R side with the proper substituent namer so a RING-bearing
            # R (benzyl -> 'benzyl') is not flattened to a carbon count
            # (get_sulfanyl_prefix mis-counts benzyl's 7 carbons as 'heptyl').
            # Collect R's atoms (everything on the far side of the S link).
            _r_atoms: List[int] = []
            _r_seen = {o_idx}
            _stack = [r_side]
            while _stack:
                _cur = _stack.pop()
                if _cur in _r_seen:
                    continue
                _r_seen.add(_cur)
                _r_atoms.append(_cur)
                for _n in mol.GetAtomWithIdx(_cur).GetNeighbors():
                    if _n.GetIdx() in sub_set and _n.GetIdx() not in _r_seen:
                        _stack.append(_n.GetIdx())
            _r_name = name_substituent_fragment(mol, _r_atoms, r_side, [o_idx])
            if not _r_name or ' ' in _r_name:
                return None  # un-nameable R -> fall through (fail-closed)
            # P-16.3.3/P-16.5: '{R}sulfanyl' concatenated onto the backbone stem
            # is a compound substituent prefix, so it is ALWAYS enclosed
            # ('(benzylsulfanyl)', '(methylsulfanyl)'); the outer enclosing mark
            # then escalates on the parent ('(benzylsulfanyl)methyl' ->
            # '[(benzylsulfanyl)methyl]benzoic acid').
            oxy = apply_enclosing_marks(f"{_r_name}sulfanyl", 0)
        groups[oxy].append(pos[bb_c])

    _MULT = {1: "", 2: "bis", 3: "tris", 4: "tetrakis"}
    cite_locants = len(backbone) > 1  # methyl: single position -> elide locant
    part_strings = []
    # Alphanumerical order by the (compound) oxy prefix name (P-14.5.2).
    for prefix in sorted(groups.keys(), key=alpha_sort_key):
        locs = sorted(groups[prefix])
        mult = _MULT.get(len(locs), f"{len(locs)}")
        # Compound oxy prefixes (phenoxy, benzyloxy, methoxy) are enclosed when
        # multiplied; single occurrence is cited bare.
        body = f"({prefix})" if len(locs) > 1 else prefix
        if cite_locants:
            loc_str = ",".join(str(loc) for loc in locs)
            part_strings.append(f"{loc_str}-{mult}{body}")
        else:
            part_strings.append(f"{mult}{body}")

    stem = get_chain_prefix(len(backbone))
    return f"{''.join(part_strings) if not cite_locants else '-'.join(part_strings)}{stem}yl"


# ============================================================================
# Fragment SMILES Extraction
# ============================================================================


def _extract_fragment_smiles(
    mol,
    sub_atoms: List[int],
    attach_idx: int,
    parent_chain: Set[int],
) -> Optional[str]:
    """Extract a valid SMILES for a substituent fragment.

    Uses RDKit's MolFragmentToSmiles to extract the substituent as a
    standalone molecule. The attachment point atom gets its valence
    satisfied implicitly (MolFragmentToSmiles caps the cut bond with H).

    Args:
        mol: RDKit Mol object.
        sub_atoms: Atom indices of the substituent fragment.
        attach_idx: Index of the first atom of the substituent (bonded to parent).
        parent_chain: Set of atom indices in the parent chain.

    Returns:
        SMILES string of the fragment, or None if extraction fails.
    """
    if not sub_atoms:
        return None

    try:
        frag_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=sub_atoms)
        if not frag_smi:
            return None

        # Validate: the fragment SMILES must parse back to a valid molecule
        frag_mol = Chem.MolFromSmiles(frag_smi)
        if frag_mol is None:
            return None

        return frag_smi
    except Exception as e:
        logger.debug(
            "DROP-15 substituent_skip: reason=extract_exception atom_count=%d error=%s",
            len(sub_atoms), e,
        )
        return None


# ============================================================================
# Parent-to-Prefix Conversion (IUPAC P-31.1.3)
# ============================================================================

# ASML-17: Retained name -> correct IUPAC substituent prefix form.
# These names have special IUPAC-defined substituent forms that cannot be
# derived by simple suffix-stripping. Checked BEFORE the regex cascade.
# Reference: IUPAC 2013 Blue Book P-31.1.3.4, P-68.3
_RETAINED_NAME_PREFIX = {
    # Heterocyclic retained names with IUPAC-defined substituent forms
    'adenine': 'adenin-9-yl',           # purine derivative, attachment at N-9
    'guanine': 'guanin-9-yl',           # purine derivative, attachment at N-9
    'indole': '1H-indol-3-yl',          # standard attachment at C-3
    'purine': 'purin-9-yl',             # standard attachment at N-9
    'uracil': 'uracil-1-yl',            # pyrimidine-2,4(1H,3H)-dione
    'thymine': 'thymin-1-yl',            # 5-methyluracil
    'cytosine': 'cytosin-1-yl',          # pyrimidine derivative
    'xanthine': 'xanthin-7-yl',          # purine-2,6-dione
    'hypoxanthine': 'hypoxanthin-9-yl',  # purine-6-ol

    # Acid-derived retained names -> acyl prefix forms
    'glutaric acid': 'glutaryl',         # pentanedioyl
    'succinic acid': 'succinyl',         # butanedioyl
    'malonic acid': 'malonyl',           # propanedioyl
    'maleic acid': 'maleoyl',            # cis-butenedioyl
    'fumaric acid': 'fumaryl',           # trans-butenedioyl
    'oxalic acid': 'oxalyl',             # ethanedioyl
    'phthalic acid': 'phthaloyl',        # benzene-1,2-dicarbonyl

    # Note: acetamide, formamide, benzamide are handled by the -amide regex
    # cascade (producing carbamoyl prefix form per IUPAC P-66.1.1.4).
    # They do NOT need lookup table entries.
}


# ============================================================================
# Acyl-nitrogen prefix subsystem — IUPAC P-66.1.1.4.3 (Wave 2 T1c)
# ============================================================================
# For R-CO-NH- on a parent with a senior characteristic group, the preferred
# prefix (method (1)) is the amide name with its final 'e' changed to 'o':
# amide -> amido, carboxamide -> carboxamido. Method (2) '{acyl}amino'
# ('pentanoylamino', 'benzoylamino') does NOT generate preferred IUPAC names.
# Blue Book examples: 4-formamidobenzoic acid (PIN), (4-acetamido-3-
# methylphenyl)arsonic acid (PIN), 4-benzamidobenzene-1-sulfonic acid (PIN).

# formamide/acetamide are retained amide PINs (P-66.1.1.1), so their amido
# prefixes keep the retained stems instead of 'methanamido'/'ethanamido'.
_AMIDO_BY_ACYL_CARBONS = {1: 'formamido', 2: 'acetamido'}

_ACID_TO_AMIDO_RETAINED = {
    'formic acid': 'formamido',
    'acetic acid': 'acetamido',
    # benzoic acid resolves via the generic '...oic acid' rule -> benzamido
}


def acyl_carbons_to_amido_prefix(acyl_carbons: int) -> Optional[str]:
    """P-66.1.1.4.3 method (1) prefix for a linear saturated acyl R-CO-.

    ``acyl_carbons`` counts the acyl carbons INCLUDING the carbonyl carbon:
    1 -> 'formamido', 2 -> 'acetamido', n>=3 -> '{stem}anamido'
    (propanamido, butanamido, ...). Returns None when no chain stem exists.
    """
    if acyl_carbons in _AMIDO_BY_ACYL_CARBONS:
        return _AMIDO_BY_ACYL_CARBONS[acyl_carbons]
    if acyl_carbons < 3:
        return None
    try:
        from ..data.chain_names import get_chain_prefix
        stem = get_chain_prefix(acyl_carbons)
    except (ImportError, ValueError, KeyError):
        return None
    return f"{stem}anamido" if stem else None


def acid_name_to_amido_prefix(acid_name: str) -> Optional[str]:
    """P-66.1.1.4.3 method (1): acid name -> amido prefix via the amide name.

    'benzoic acid' -> 'benzamido', 'pentanoic acid' -> 'pentanamido',
    'naphthalene-1-carboxylic acid' -> 'naphthalene-1-carboxamido'
    (the amide names benzamide/pentanamide/naphthalene-1-carboxamide have
    their final 'e' changed to 'o'). Returns None when no safe transform
    exists — poly-acids ('...dioic acid', '...dicarboxylic acid') and
    functional-replacement acids fail closed because a single amido prefix
    cannot describe them.
    """
    if not acid_name:
        return None
    name = acid_name.strip()
    low = name.lower()
    retained = _ACID_TO_AMIDO_RETAINED.get(low)
    if retained:
        return retained
    # Functional-replacement / peroxy acids: the plain amido transform would
    # misdescribe them; fail closed (callers keep their legacy fallback).
    if low.endswith(('thioic acid', 'selenoic acid', 'telluroic acid',
                     'peroxoic acid', 'imidic acid', 'ohydroximic acid')):
        return None
    if low.endswith('carboxylic acid'):
        stem = name[:-len('carboxylic acid')]
        # 'X-1,2-dicarboxylic acid' etc. has >1 acid group; reject.
        if re.search(r'(?:di|tri|tetra|penta|hexa)[-,\d]*$', stem.lower()):
            return None
        return stem + 'carboxamido'
    if low.endswith('dioic acid'):
        return None  # two acid groups — not describable by one amido prefix
    if low.endswith('oic acid'):
        return name[:-len('oic acid')] + 'amido'
    return None


def oxamoyl_branch_name(mol, n_idx: int, acyl_c_idx: int) -> Optional[str]:
    """P-66.1.1.4.5.1 (BB 33071/55479): recognize the EXACT H2N-CO-CO- branch
    hanging from an imine N -> 'oxamoyl' (the preferred prefix for the
    H2N-CO-CO-N= group's acyl part). Returns None for anything else
    (fail-closed): the first carbon must be a carbonyl (=O, no other
    substituents besides the =N-bearing N and the second carbonyl C); the
    second carbon must be a carbamoyl (=O + terminal NH2)."""
    a = mol.GetAtomWithIdx(acyl_c_idx)
    if a.GetAtomicNum() != 6:
        return None
    nbrs = {n.GetIdx(): n for n in a.GetNeighbors() if n.GetIdx() != n_idx}
    dbl_o = [i for i, n in nbrs.items() if n.GetAtomicNum() == 8 and
             mol.GetBondBetweenAtoms(acyl_c_idx, i).GetBondTypeAsDouble() == 2.0]
    c2 = [i for i, n in nbrs.items() if n.GetAtomicNum() == 6]
    if len(dbl_o) != 1 or len(c2) != 1 or len(nbrs) != 2:
        return None
    c2a = mol.GetAtomWithIdx(c2[0])
    o2 = [n for n in c2a.GetNeighbors() if n.GetAtomicNum() == 8 and
          mol.GetBondBetweenAtoms(c2[0], n.GetIdx()).GetBondTypeAsDouble() == 2.0]
    n2 = [n for n in c2a.GetNeighbors() if n.GetAtomicNum() == 7 and
          n.GetIdx() != acyl_c_idx and n.GetDegree() == 1 and n.GetTotalNumHs() == 2]
    if len(o2) == 1 and len(n2) == 1 and c2a.GetDegree() == 3:
        return "oxamoyl"
    return None


def linear_acyl_amido_prefix(mol, carbonyl_c: int, n_idx: int,
                             sub_atoms) -> Optional[str]:
    """Amido prefix (P-66.1.1.4.3 method (1)) for an N-acyl substituent.

    Emits 'formamido' (HCO-NH-), 'acetamido' (CH3-CO-NH-) or
    '{stem}anamido' (>=3 C) ONLY when the substituent is exactly
    -NH-CO-R with R an unbranched, saturated, acyclic, all-carbon chain
    and every substituent atom accounted for (the N, the carbonyl O and
    the chain carbons — nothing dropped). Returns None otherwise, so the
    strict builder can never mint a name for a branch it cannot fully
    describe; callers keep their legacy fallback for those.

    Args:
        mol: RDKit Mol.
        carbonyl_c: atom index of the acyl C=O carbon.
        n_idx: atom index of the amide nitrogen (the attachment atom).
        sub_atoms: all atom indices of the substituent (including n_idx).
    """
    sub_set = set(sub_atoms)
    if carbonyl_c not in sub_set or n_idx not in sub_set:
        return None
    ring_info = mol.GetRingInfo()
    branch = set()
    carbonyl_o = None
    stack = [carbonyl_c]
    while stack:
        a_idx = stack.pop()
        if a_idx in branch or a_idx == n_idx:
            continue
        if a_idx not in sub_set:
            return None  # branch escapes the substituent
        atom = mol.GetAtomWithIdx(a_idx)
        sym = atom.GetSymbol()
        if sym == 'O':
            # allow exactly the carbonyl =O on the acyl carbon
            bond = mol.GetBondBetweenAtoms(carbonyl_c, a_idx)
            if (carbonyl_o is None and bond is not None
                    and bond.GetBondTypeAsDouble() == 2.0):
                carbonyl_o = a_idx
                branch.add(a_idx)
                continue
            return None
        if sym != 'C':
            return None
        if atom.GetIsAromatic() or ring_info.NumAtomRings(a_idx) > 0:
            return None
        if atom.GetFormalCharge() or atom.GetNumRadicalElectrons():
            return None
        branch.add(a_idx)
        for nbr in atom.GetNeighbors():
            ni = nbr.GetIdx()
            if ni == n_idx or ni in branch:
                continue
            bond = mol.GetBondBetweenAtoms(a_idx, ni)
            if (nbr.GetSymbol() == 'C' and bond is not None
                    and bond.GetBondTypeAsDouble() != 1.0):
                return None  # unsaturated acyl (prop-2-enamido) not built here
            stack.append(ni)
    if carbonyl_o is None:
        return None
    # every substituent atom must be the N, the carbonyl O, or a chain C
    if sub_set != branch | {n_idx}:
        return None
    carbons = [a for a in branch
               if mol.GetAtomWithIdx(a).GetSymbol() == 'C']
    # unbranched: each chain carbon has <=2 carbon neighbours in-branch and
    # the acyl carbon at most one
    for c_idx in carbons:
        c_nbrs = sum(
            1 for nbr in mol.GetAtomWithIdx(c_idx).GetNeighbors()
            if nbr.GetIdx() in branch and nbr.GetSymbol() == 'C'
        )
        if c_nbrs > (1 if c_idx == carbonyl_c else 2):
            return None
    return acyl_carbons_to_amido_prefix(len(carbons))


def acyl_amido_prefix_from_branch(mol, n_idx: int, carbonyl_c: int,
                                  sub_atoms) -> Optional[str]:
    """P-66.1.1.4.3 method (1) amido prefix for a full N-attached acyl branch.

    ``sub_atoms`` is the ENTIRE substituent (the amide N plus the whole acyl
    fragment). Fast path: :func:`linear_acyl_amido_prefix`. General path
    (ring / substituted acyls): take ALL branch atoms except the N as the
    acyl fragment — nothing can be silently dropped — convert it to the
    corresponding acid by adding an -OH at the carbonyl carbon, name that
    acid recursively, then apply :func:`acid_name_to_amido_prefix`
    ('benzoic acid' -> 'benzamido', '4-methylbenzoic acid' ->
    '4-methylbenzamido'). Returns the BARE prefix (callers add enclosing
    marks for locant-bearing forms) or None (fail closed).
    """
    sub_set = set(sub_atoms)
    if n_idx not in sub_set or carbonyl_c not in sub_set:
        return None

    linear = linear_acyl_amido_prefix(mol, carbonyl_c, n_idx, sub_atoms)
    if linear:
        return linear

    n_atom = mol.GetAtomWithIdx(n_idx)
    if n_atom.GetFormalCharge() != 0:
        return None
    # the N must connect to the branch ONLY through the acyl carbon
    # (-N(H)-CO-R; N-substituted amido forms are not built here)
    in_branch_nbrs = {
        nb.GetIdx() for nb in n_atom.GetNeighbors() if nb.GetIdx() in sub_set
    }
    if in_branch_nbrs != {carbonyl_c}:
        return None
    # confirm the carbonyl =O inside the branch
    has_carbonyl_o = any(
        nb.GetSymbol() == 'O' and nb.GetIdx() in sub_set
        and mol.GetBondBetweenAtoms(carbonyl_c, nb.GetIdx()) is not None
        and mol.GetBondBetweenAtoms(
            carbonyl_c, nb.GetIdx()).GetBondTypeAsDouble() == 2.0
        for nb in mol.GetAtomWithIdx(carbonyl_c).GetNeighbors()
    )
    if not has_carbonyl_o:
        return None
    # the fragment must be closed: every acyl atom's neighbours stay inside
    # the substituent (otherwise MolFragmentToSmiles would silently cut a
    # bond and the name would describe a different molecule)
    for a_idx in sub_set - {n_idx}:
        for nb in mol.GetAtomWithIdx(a_idx).GetNeighbors():
            if nb.GetIdx() not in sub_set:
                return None
    try:
        rw = Chem.RWMol(mol)
        oh = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(carbonyl_c, oh, Chem.BondType.SINGLE)
        hh = rw.AddAtom(Chem.Atom(1))
        rw.AddBond(oh, hh, Chem.BondType.SINGLE)
        frag_atoms = sorted((sub_set - {n_idx}) | {oh, hh})
        frag_smi = Chem.MolFragmentToSmiles(rw, frag_atoms, canonical=True)
        if not frag_smi:
            return None
        from .fragment_naming import name_fragment_recursively
        acid_name = name_fragment_recursively(frag_smi)
        if not acid_name:
            return None
        return acid_name_to_amido_prefix(acid_name)
    except Exception:
        return None


def imidamide_name_to_imidamido_prefix(name: str) -> Optional[str]:
    """Wave2 T3d (P-66.4.1.3.5 method 1): turn an amidine parent name
    (imidamide / carboximidamide) into the non-principal prefix by changing the
    final 'e' -> 'o': 'ethanimidamide' -> 'ethanimidamido', 'benzenecarboximidamide'
    -> 'benzenecarboximidamido'. Fail-closed (None) for di/poly-imidamide names
    (one imidamido cannot describe a poly-amidine) or names carrying N-locants
    the branch cannot describe."""
    if not name:
        return None
    low = name.lower()
    if not (low.endswith('imidamide')):
        return None
    if 'diimidamide' in low or 'dicarboximidamide' in low:
        return None
    if low.startswith('n-') or low.startswith("n'") or ',n' in low:
        return None
    return name[:-1] + 'o'


def imidoyl_amido_prefix_from_branch(mol, n_idx: int, imino_c: int,
                                     sub_atoms) -> Optional[str]:
    """Wave2 T3d (P-66.4.1.3.5): imidamido prefix for a full N-attached amidine
    branch ``-N(H)-C(=NH)-R`` (the amidine's AMINO nitrogen is the ring/chain
    attachment). Mirrors :func:`acyl_amido_prefix_from_branch` but keyed on the
    imino C=N instead of a carbonyl C=O. Reconstructs the imidamide parent
    R-C(=NH)-NH2, names it recursively, then applies
    :func:`imidamide_name_to_imidamido_prefix`. Returns the BARE prefix or None.

    Guards (fail-closed): the imino carbon must have exactly one =N (double) whose
    N bears only H/C (reject amidrazone -C(=N-NH2)-); exactly one single-bonded N
    == n_idx (reject guanidine's second amino N); the attachment N connects to the
    branch ONLY through this carbon; full-branch coverage (no dropped atoms)."""
    sub_set = set(sub_atoms)
    if n_idx not in sub_set or imino_c not in sub_set:
        return None
    n_atom = mol.GetAtomWithIdx(n_idx)
    if n_atom.GetFormalCharge() != 0:
        return None
    if {nb.GetIdx() for nb in n_atom.GetNeighbors() if nb.GetIdx() in sub_set} != {imino_c}:
        return None
    c_atom = mol.GetAtomWithIdx(imino_c)
    if c_atom.GetFormalCharge() != 0:
        return None
    # exactly one imino =N (H/C only) + exactly one single-bonded N (== n_idx)
    imino_n = None
    single_ns = []
    for nb in c_atom.GetNeighbors():
        b = mol.GetBondBetweenAtoms(imino_c, nb.GetIdx())
        if nb.GetSymbol() == 'N':
            if b.GetBondTypeAsDouble() == 2.0:
                if imino_n is not None:
                    return None
                imino_n = nb
            elif b.GetBondTypeAsDouble() == 1.0:
                single_ns.append(nb.GetIdx())
    if imino_n is None or single_ns != [n_idx]:
        return None
    # imino N must bear only H/C (reject amidrazone -C(=N-NH2)-)
    for nb in imino_n.GetNeighbors():
        if nb.GetIdx() != imino_c and nb.GetSymbol() not in ('C',):
            return None
    # closed fragment: every acyl-side atom's neighbours stay in the branch
    for a_idx in sub_set - {n_idx}:
        for nb in mol.GetAtomWithIdx(a_idx).GetNeighbors():
            if nb.GetIdx() not in sub_set:
                return None
    try:
        rw = Chem.RWMol(mol)
        nh2 = rw.AddAtom(Chem.Atom(7))
        rw.AddBond(imino_c, nh2, Chem.BondType.SINGLE)
        frag_atoms = sorted((sub_set - {n_idx}) | {nh2})
        frag_smi = Chem.MolFragmentToSmiles(rw, frag_atoms, canonical=True)
        if not frag_smi:
            return None
        from .fragment_naming import name_fragment_recursively
        parent = name_fragment_recursively(frag_smi)
        if not parent:
            return None
        return imidamide_name_to_imidamido_prefix(parent)
    except Exception:
        return None


def hydrazonamide_name_to_hydrazonamido_prefix(name: str) -> Optional[str]:
    """PF-2 (P-66.4.2.3.5): turn an amidrazone parent name (hydrazonamide) into
    the non-principal prefix by changing the final 'e' -> 'o':
    'ethanehydrazonamide' -> 'ethanehydrazonamido'. Fail-closed (None) for
    di/poly names or names carrying N-locants the branch cannot describe."""
    if not name:
        return None
    low = name.lower()
    if not low.endswith('hydrazonamide'):
        return None
    if 'dihydrazonamide' in low or 'dicarbohydrazonamide' in low:
        return None
    if low.startswith('n-') or low.startswith("n'") or ',n' in low:
        return None
    return name[:-1] + 'o'


def hydrazonoyl_amido_prefix_from_branch(mol, n_idx: int, imino_c: int,
                                         sub_atoms) -> Optional[str]:
    """PF-2 (P-66.4.2.3.5): hydrazonamido prefix for a full N-attached amidrazone
    branch ``-N(H)-C(=N-NH2)-R`` (the amidrazone AMINO nitrogen is the ring/chain
    attachment). Sibling of :func:`imidoyl_amido_prefix_from_branch`, but keyed
    on the hydrazono ``C=N-NH2`` — the terminal ``NH2`` on the imino N is the
    discriminant vs a plain amidine (which that sibling rejects by design).
    Reconstructs the amidrazone parent R-C(=N-NH2)-NH2, names it recursively
    (-> 'ethanehydrazonamide'), then applies the e->o transform. Returns the
    BARE prefix or None.

    Guards (fail-closed): the imino carbon has exactly one =N (double) whose N
    bears exactly one terminal degree-1 NH2 (neutral); exactly one single-bonded
    N == n_idx (rejects hydrazidine/guanidine); the attachment N connects to the
    branch ONLY through this carbon; zero formal charges; full-branch coverage."""
    sub_set = set(sub_atoms)
    if n_idx not in sub_set or imino_c not in sub_set:
        return None
    n_atom = mol.GetAtomWithIdx(n_idx)
    if n_atom.GetFormalCharge() != 0:
        return None
    if {nb.GetIdx() for nb in n_atom.GetNeighbors()
            if nb.GetIdx() in sub_set} != {imino_c}:
        return None
    c_atom = mol.GetAtomWithIdx(imino_c)
    if c_atom.GetFormalCharge() != 0:
        return None
    imino_n = None
    single_ns = []
    for nb in c_atom.GetNeighbors():
        b = mol.GetBondBetweenAtoms(imino_c, nb.GetIdx())
        if nb.GetSymbol() == 'N':
            if b.GetBondTypeAsDouble() == 2.0:
                if imino_n is not None:
                    return None
                imino_n = nb
            elif b.GetBondTypeAsDouble() == 1.0:
                single_ns.append(nb.GetIdx())
    if imino_n is None or single_ns != [n_idx]:
        return None
    # hydrazono discriminant: the imino =N's sole non-imino_c neighbour is a
    # single terminal degree-1 neutral NH2 (rejects plain amidine =NH / =N-C).
    imino_n_heavy = [nb for nb in imino_n.GetNeighbors()
                     if nb.GetIdx() != imino_c]
    if (imino_n.GetFormalCharge() != 0 or len(imino_n_heavy) != 1
            or imino_n_heavy[0].GetSymbol() != 'N'
            or imino_n_heavy[0].GetDegree() != 1
            or imino_n_heavy[0].GetFormalCharge() != 0):
        return None
    # closed fragment: every acyl-side atom's neighbours stay in the branch
    for a_idx in sub_set - {n_idx}:
        for nb in mol.GetAtomWithIdx(a_idx).GetNeighbors():
            if nb.GetIdx() not in sub_set:
                return None
    try:
        rw = Chem.RWMol(mol)
        nh2 = rw.AddAtom(Chem.Atom(7))
        rw.AddBond(imino_c, nh2, Chem.BondType.SINGLE)
        frag_atoms = sorted((sub_set - {n_idx}) | {nh2})
        frag_smi = Chem.MolFragmentToSmiles(rw, frag_atoms, canonical=True)
        if not frag_smi:
            return None
        from .fragment_naming import name_fragment_recursively
        parent = name_fragment_recursively(frag_smi)
        if not parent:
            return None
        return hydrazonamide_name_to_hydrazonamido_prefix(parent)
    except Exception:
        return None


def _pure_linear_alkyl_len_local(mol, start_idx, exclude) -> Optional[int]:
    """Length of a pure, unbranched, acyclic, saturated all-carbon chain
    (local copy of composer._pure_linear_alkyl_len to avoid an import cycle).
    Returns None for branched/cyclic/unsaturated/hetero-decorated fragments."""
    from collections import deque
    ri = mol.GetRingInfo()
    frag = set()
    queue = deque([start_idx])
    while queue:
        a = queue.popleft()
        if a in frag or a in exclude:
            continue
        at = mol.GetAtomWithIdx(a)
        if at.GetSymbol() != 'C' or ri.NumAtomRings(a) > 0:
            return None
        frag.add(a)
        for nb in at.GetNeighbors():
            ni = nb.GetIdx()
            if ni in exclude:
                continue
            bond = mol.GetBondBetweenAtoms(a, ni)
            if nb.GetSymbol() == 'C':
                if bond.GetBondTypeAsDouble() != 1.0:
                    return None
                queue.append(ni)
            else:
                return None
    # unbranched: no in-fragment carbon has > 2 fragment-carbon neighbours
    for a in frag:
        c_nbrs = sum(1 for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                     if nb.GetIdx() in frag)
        if c_nbrs > 2:
            return None
    return len(frag) if frag else None


def _isolated_benzene_wholly_in(mol, c_idx, sub_set) -> bool:
    """True when ``c_idx`` is in an isolated (non-fused) all-carbon aromatic
    6-ring wholly contained in ``sub_set``."""
    a = mol.GetAtomWithIdx(c_idx)
    if not (a.GetIsAromatic() and a.GetSymbol() == 'C'):
        return False
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if (c_idx in ring and len(ring) == 6
                and all(r in sub_set for r in ring)
                and all(mol.GetAtomWithIdx(r).GetIsAromatic()
                        and mol.GetAtomWithIdx(r).GetSymbol() == 'C'
                        for r in ring)):
            rset = set(ring)
            if not any(set(o) != rset and set(o) & rset for o in ri.AtomRings()):
                return True
    return False


def sulfino_hydrazonoyl_amido_prefix_from_branch(mol, n_idx: int, s_idx: int,
                                                 sub_atoms) -> Optional[str]:
    """P-66.4.2.3.5 / P-66.4.3.2 (plan P1AM Task 8): N-attached
    R-S(=N-NH2)(-NH-)[=O]? branch -> '{R-stem}sulfinohydrazonamido' (no =O)
    or '{R-stem}sulfonohydrazonamido' (one =O). Fail-closed None on any
    deviation (charges, extra substitution, unnameable R).
    """
    sub_set = set(sub_atoms)
    s = mol.GetAtomWithIdx(s_idx)
    if s.GetSymbol() != 'S' or s.GetFormalCharge() != 0:
        return None
    dbl_o = []
    dbl_n = []
    r_root = None
    for nb in s.GetNeighbors():
        bt = mol.GetBondBetweenAtoms(s_idx, nb.GetIdx()).GetBondTypeAsDouble()
        if nb.GetIdx() == n_idx:
            if bt != 1.0:
                return None
            continue
        if bt == 2.0 and nb.GetSymbol() == 'O':
            dbl_o.append(nb.GetIdx())
        elif bt == 2.0 and nb.GetSymbol() == 'N':
            dbl_n.append(nb.GetIdx())
        elif bt == 1.0 and nb.GetSymbol() == 'C':
            if r_root is not None:
                return None
            r_root = nb.GetIdx()
        else:
            return None
    if len(dbl_n) != 1 or r_root is None or len(dbl_o) > 1:
        return None
    # =N-NH2 hydrazono arm: exactly one terminal NH2 on the =N
    hz = mol.GetAtomWithIdx(dbl_n[0])
    if hz.GetFormalCharge() != 0:
        return None
    hz_tails = [nb for nb in hz.GetNeighbors() if nb.GetIdx() != s_idx
                and nb.GetAtomicNum() > 1]
    if (len(hz_tails) != 1 or hz_tails[0].GetSymbol() != 'N'
            or hz_tails[0].GetDegree() != 1
            or hz_tails[0].GetFormalCharge() != 0):
        return None
    # attachment N carries nothing else heavy besides the parent + S
    # (N-substituted forms fail closed)
    n_atom = mol.GetAtomWithIdx(n_idx)
    if n_atom.GetFormalCharge() != 0:
        return None
    n_heavy_in_sub = [nb.GetIdx() for nb in n_atom.GetNeighbors()
                      if nb.GetIdx() in sub_set and nb.GetAtomicNum() > 1]
    if n_heavy_in_sub != [s_idx]:
        return None
    # R must be an isolated benzene ring or an unbranched n-alkyl wholly in sub.
    if _isolated_benzene_wholly_in(mol, r_root, sub_set):
        stem = "benzene"
    else:
        n_len = _pure_linear_alkyl_len_local(mol, r_root, {s_idx})
        if n_len is None or n_len < 1:
            return None
        from ..data.chain_names import get_chain_prefix
        stem = f"{get_chain_prefix(n_len)}ane"
    word = "sulfono" if len(dbl_o) == 1 else "sulfino"
    return f"{stem}{word}hydrazonamido"


def parent_to_prefix(parent_name: str, chain_length: int, attach_locant: int = 1) -> str:
    """Convert a parent compound name to substituent prefix form.

    Per IUPAC P-31.1.3, the parent compound name is transformed into
    a substituent prefix by:
    1. Removing the suffix (e.g., -oic acid, -ol, -one)
    2. Converting the suffix to its prefix form (e.g., -ol -> hydroxy)
    3. Adding the prefix at the correct locant
    4. Appending -yl at the free-valence position

    Per IUPAC P-46.2, the point of free valency receives the lowest
    possible locant consistent with any fixed numbering of the parent
    hydride. For chain-derived substituents with no fixed numbering,
    the chain is oriented so the attachment point is at locant 1.

    Args:
        parent_name: Parent compound IUPAC name (e.g., "propan-2-ol").
        chain_length: Number of carbons in the substituent chain.
        attach_locant: Locant of the free-valence carbon (default 1).

    Returns:
        Prefix-form name (e.g., "2-hydroxypropyl"). Returns raw name
        WITHOUT enclosing marks.
    """
    if not parent_name:
        return ""

    name = parent_name.strip()

    # ASML-17: Check retained name lookup FIRST (before regex cascade)
    name_lower = name.lower()
    if name_lower in _RETAINED_NAME_PREFIX:
        return _RETAINED_NAME_PREFIX[name_lower]

    # ---- Carboxylic acids: -oic acid / -anoic acid ----
    # e.g., "butanoic acid" -> "3-carboxypropyl"
    m_oic = re.match(r'^(.+?)(?:an)?oic acid$', name)
    if m_oic:
        stem = m_oic.group(1)
        # Carboxy goes on the terminal carbon (chain_length for original chain)
        # The stem is shortened by one carbon (the COOH carbon becomes "carboxy")
        carboxy_locant = chain_length
        # Build: {locant}-carboxy{shortened_stem}yl
        # For butanoic acid (4C): carboxy at C4, stem = prop (3C), result = 3-carboxypropyl
        # Actually: butanoic acid -> carboxy replaces the acid, chain becomes 3C (propyl)
        # The parent chain of the substituent WITHOUT the carboxylic acid is chain_length - 1
        shortened_length = chain_length - 1
        if shortened_length >= 1:
            from ..data.chain_names import get_chain_prefix
            short_stem = get_chain_prefix(shortened_length)
            # carboxy locant is the shortened chain length (farthest from attachment)
            return f"{shortened_length}-carboxy{short_stem}yl"
        else:
            return "carboxy"

    # ---- Locanted alcohol: -N-ol ----
    # Matches saturated (-an-N-ol), unsaturated (-en-N-ol, -yn-N-ol),
    # and bare (-N-ol) patterns.
    # e.g., "propan-2-ol" -> "2-hydroxypropyl"
    # e.g., "3-methylbut-2-en-1-ol" -> "1-hydroxy-3-methylbut-2-en-1-yl"
    m_ol = re.search(r'-(\d+)-ol$', name)
    if m_ol:
        locant = m_ol.group(1)
        # Get the stem (everything before "-N-ol")
        stem = name[:m_ol.start()]
        # Convert saturated suffix to yl: -an -> -yl (propan -> propyl)
        # Keep unsaturation: -en stays as -en, -yn stays as -yn
        if stem.endswith('an'):
            stem = stem[:-2]  # propan -> prop
        # Insert hyphen between hydroxy and stem when stem starts with a
        # digit (e.g., "3-methylbut-2-en") to avoid "hydroxy3-methylbut"
        sep = "-" if stem and stem[0].isdigit() else ""
        return f"{locant}-hydroxy{sep}{stem}yl"

    # ---- Multi-FG alcohol: -diol, -triol ----
    # e.g., "propane-1,2-diol" -> "2,3-dihydroxypropyl"
    # e.g., "ethane-1,2-diol" -> "2-hydroxy-1-(hydroxymethyl)" ... complex
    # P-63.1.2 elides the multiplier-final 'a' before '-ol' (tetra+ol -> tetrol),
    # so match BOTH spellings (tetra? = tetr|tetra) and derive the hydroxy
    # multiplier from the locant COUNT — group(2) 'tetr' must NOT become the
    # wrong 'tetrhydroxy'. (v22 G2 follow-on: keeps this converter in sync with
    # the elision fix in naming_utils._join_multiplied_suffix.)
    m_diol = re.search(r'[,-](\d+(?:,\d+)*)-(?:di|tri|tetra?)ol$', name)
    if m_diol:
        locants = m_diol.group(1)
        count = locants.count(',') + 1
        multiplier = SIMPLE_MULTIPLIERS.get(count, '') if count > 1 else ''
        stem = name[:m_diol.start()]
        if stem.endswith('an'):
            stem = stem[:-2]
        elif stem.endswith('a'):
            stem = stem[:-1]
        prefix = "hydroxy" if multiplier == '' else f"{multiplier}hydroxy"
        return f"{locants}-{prefix}{stem}yl"

    # ---- Multi-FG ketone: -dione, -trione ----
    m_dione = re.search(r'[,-](\d+(?:,\d+)*)-([dt]i|tri|tetra)one$', name)
    if m_dione:
        locants = m_dione.group(1)
        multiplier = m_dione.group(2)
        stem = name[:m_dione.start()]
        if stem.endswith('an'):
            stem = stem[:-2]
        elif stem.endswith('a'):
            stem = stem[:-1]
        return f"{locants}-{multiplier}oxo{stem}yl"

    # ---- Multi-FG amine: -diamine, -triamine ----
    m_diamine = re.search(r'[,-](\d+(?:,\d+)*)-([dt]i|tri|tetra)amine$', name)
    if m_diamine:
        locants = m_diamine.group(1)
        multiplier = m_diamine.group(2)
        stem = name[:m_diamine.start()]
        if stem.endswith('an'):
            stem = stem[:-2]
        elif stem.endswith('a'):
            stem = stem[:-1]
        return f"{locants}-{multiplier}amino{stem}yl"

    # ---- Unlocanted alcohol: ends in -ol (e.g., "ethanol", "methanol") ----
    if name.endswith('ol') and not name.endswith('diol') and not name.endswith('triol'):
        # Strip -ol, check for -an prefix
        base = name[:-2]  # remove "ol"
        if base.endswith('an'):
            stem = base[:-2]  # remove "an"
        elif base.endswith('a'):
            # e.g., "methan" case (methanol -> methan -> meth)
            stem = base[:-1]
        else:
            stem = base
        # Alcohol: add hydroxy prefix
        return f"hydroxy{stem}yl"

    # ---- Locanted ketone: -an-N-one ----
    # e.g., "butan-2-one" -> "2-oxobutyl"
    m_one = re.search(r'an?-(\d+)-one$', name)
    if m_one:
        locant = m_one.group(1)
        stem_end = m_one.start()
        stem = name[:stem_end].rstrip('-')
        return f"{locant}-oxo{stem}yl"

    # ---- Unlocanted ketone: ends in -one ----
    if name.endswith('one') and not name.endswith('none'):
        base = name[:-3]  # remove "one"
        if base.endswith('an'):
            stem = base[:-2]
        elif base.endswith('a'):
            stem = base[:-1]
        else:
            stem = base
        return f"oxo{stem}yl"

    # ---- Locanted amine: -an-N-amine ----
    # e.g., "propan-1-amine" -> "1-aminopropyl"
    m_amine = re.search(r'an?-(\d+)-amine$', name)
    if m_amine:
        locant = m_amine.group(1)
        stem_end = m_amine.start()
        stem = name[:stem_end].rstrip('-')
        return f"{locant}-amino{stem}yl"

    # ---- Unlocanted amine: ends in -amine / -anamine ----
    if name.endswith('amine'):
        base = name[:-5]  # remove "amine"
        if base.endswith('an'):
            stem = base[:-2]
        elif base.endswith('a'):
            stem = base[:-1]
        else:
            stem = base
        return f"amino{stem}yl"

    # ---- Aldehyde: ends in -al or -anal ----
    # e.g., "propanal" -> "3-oxopropyl"  (moving-base-atom: the -CHO carbon is
    # absorbed into the substituent chain and expressed as 'oxo', NOT 'formyl').
    if (name.endswith('al') and not name.endswith('nal')) or name.endswith('anal'):
        if name.endswith('anal'):
            stem = name[:-4]  # remove "anal"
        elif name.endswith('al'):
            base = name[:-2]  # remove "al"
            if base.endswith('an'):
                stem = base[:-2]
            elif base.endswith('a'):
                stem = base[:-1]
            else:
                stem = base
        # IUPAC P-66.6.1 + Table-28.1 note (m): the former -CHO carbon sits at the
        # chain terminus OPPOSITE the attachment (attachment = locant 1, P-29 /
        # P-31.1.4.3.4), so the oxo locant is the LAST carbon = chain_length.
        # '1-oxo...yl' (oxo at the attachment/acyl carbon) is the disfavoured CAS
        # acyl form, explicitly NOT a preferred IUPAC prefix.
        return f"{chain_length}-oxo{stem}yl"

    # ---- Ester: -oate suffix ---- (IUPAC P-65.6.3)
    # e.g., "propanoate" -> carboxy prefix form
    # When ester is not the principal group, the acid portion uses "carboxy"
    m_oate = re.search(r'(?:an)?oate$', name)
    if m_oate:
        shortened = chain_length - 1
        if shortened >= 1:
            from ..data.chain_names import get_chain_prefix
            short_stem = get_chain_prefix(shortened)
            return f"{shortened}-carboxy{short_stem}yl"
        return "carboxy"

    # ---- Amidine: -carboximidamide / -imidamide ---- (IUPAC P-66.4.1.3.1)
    # MUST precede the general -amide regex which would wrongly match "-imidamide"
    # and return "carbamoyl" (amide prefix) for an amidine group.
    # e.g., "methanimidamide" -> "carbamimidoyl" (terminal C1; chain_length-1 = 0)
    # e.g., "propanimidamide" -> "carbamimidoyl" with chain stem for longer chains
    if name.endswith('carboximidamide'):
        stem = name[:-15]  # remove "carboximidamide"
        if stem:
            return f"carbamimidoyl{stem}yl"
        return "carbamimidoyl"
    m_imidamide = re.search(r'(?:an)?imidamide$', name)
    if m_imidamide:
        shortened = chain_length - 1
        if shortened >= 1:
            from ..data.chain_names import get_chain_prefix
            short_stem = get_chain_prefix(shortened)
            return f"{shortened}-carbamimidoyl{short_stem}yl"
        return "carbamimidoyl"

    # ---- Amide: -carboxamide (most specific first) ---- (IUPAC P-66.1.1.4)
    # e.g., "benzcarboxamide" -> "carbamoyl" prefix
    if name.endswith('carboxamide'):
        stem = name[:-11]  # remove "carboxamide"
        if stem:
            return f"carbamoyl{stem}yl"
        return "carbamoyl"

    # ---- Amide: general -amide suffix ---- (IUPAC P-66.1.1.4)
    # e.g., "propanamide" -> "2-carbamoylethyl", "acetamide" -> "carbamoylmethyl"
    m_amide = re.search(r'(?:an)?amide$', name)
    if m_amide:
        shortened = chain_length - 1
        if shortened >= 1:
            from ..data.chain_names import get_chain_prefix
            short_stem = get_chain_prefix(shortened)
            return f"{shortened}-carbamoyl{short_stem}yl"
        return "carbamoyl"

    # ---- Nitrile: -carbonitrile (most specific first) ---- (IUPAC P-66.1.4.1)
    # e.g., "benzonitrile" -> "cyanophenyl" (cyano + stem + yl)
    if name.endswith('carbonitrile'):
        stem = name[:-12]  # remove "carbonitrile"
        if stem:
            return f"cyano{stem}yl"
        return "cyano"

    # ---- Nitrile: general -nitrile suffix ---- (IUPAC P-66.1.4.1)
    # e.g., "propanenitrile" -> "2-cyanoethyl", "acetonitrile" -> "cyanomethyl"
    if name.endswith('nitrile') and not name.endswith('carbonitrile'):
        shortened = chain_length - 1
        if shortened >= 1:
            from ..data.chain_names import get_chain_prefix
            short_stem = get_chain_prefix(shortened)
            return f"{shortened}-cyano{short_stem}yl"
        return "cyano"

    # ---- Simple acid names: convert to acyl prefix ---- (IUPAC P-65.1.7)
    # Only handle simple retained acid names that appear as substituents.
    # e.g., "formic acid" -> "formyl", "acetic acid" -> "acetyl"
    _ACID_TO_ACYL_PREFIX = {
        'formic acid': 'formyl',
        'acetic acid': 'acetyl',
        'propionic acid': 'propionyl',
        'butyric acid': 'butyryl',
        'benzoic acid': 'benzoyl',
    }
    if name in _ACID_TO_ACYL_PREFIX:
        return _ACID_TO_ACYL_PREFIX[name]

    # ---- Cyclic names: cyclo...ane -> cyclo...yl ---- (IUPAC P-31.1.3)
    # e.g., "cyclohexane" -> "cyclohexyl", "cyclopentane" -> "cyclopentyl"
    if 'cyclo' in name and name.endswith('ane'):
        stem = name[:-3]  # remove "ane"
        return f"{stem}yl"

    # ---- Heterocyclic -ane ending ---- (IUPAC P-31.1.3)
    # Heterocyclic ring names (oxirane, thiirane, oxetane, thietane, oxolane,
    # oxane, thiane, etc.) replace -e with -yl, NOT strip -ane and add -yl.
    # e.g., "oxirane" -> "oxiranyl" (not "oxiryl"), "oxane" -> "oxanyl" (not "oxyl")
    _HETERO_ANE_RINGS = {
        'oxirane', 'thiirane', 'oxetane', 'thietane', 'oxolane',
        'oxane', 'thiane', 'thiolane',
        'dioxane', 'dioxolane', 'dithiane', 'dithiolane', 'trioxane',
        'borolane', 'boroxane', 'silolane',
    }
    base_name = name.split('-')[-1] if '-' in name else name
    if base_name in _HETERO_ANE_RINGS:
        return name[:-1] + "yl"  # replace -e with -yl

    # ---- Alkane: -ane or -e ending ----
    # e.g., "propane" -> "propyl", "2-methylpropane" -> "2-methylpropyl"
    if name.endswith('ane'):
        stem = name[:-3]  # remove "ane"
        return f"{stem}yl"

    # ---- Heterocyclic -ine ending ---- (IUPAC P-31.1.3)
    # e.g., "pyridine" -> "pyridinyl", "piperidine" -> "piperidinyl"
    # Note: -ine must come BEFORE the generic -e fallback
    if name.endswith('ine'):
        return name[:-1] + "yl"  # pyridine -> pyridinyl

    # ---- Fallback: strip terminal -e if present, add -yl ----
    # Guard: names ending in -amide/-imide should NOT use naive -e stripping
    if name.endswith('e'):
        if name.endswith('amide'):
            # amide -> amido form (e.g., "formamide" -> "formamido")
            return name[:-1] + 'o'  # -amide -> -amido
        if name.endswith('imide'):
            # imide -> imido form (e.g., "succinimide" -> "succinimido")
            return name[:-1] + 'o'  # -imide -> -imido
        return name[:-1] + "yl"

    # If name already ends in -yl, return as-is
    if name.endswith('yl'):
        return name

    return name + "yl"


# ============================================================================
# Retained Substituent Names
# ============================================================================


def _is_plain_phenyl(mol, aromatic_idx: int, central_c_idx: int) -> bool:
    """True if ``aromatic_idx`` is in an unsubstituted benzene ring (C6H5-).

    "Unsubstituted" = a 6-membered all-aromatic-carbon ring whose only exocyclic
    heavy attachment is ``central_c_idx``. Used by ``_name_aryl_methyl_ether`` to
    confirm a genuine *di-phenyl* methyl before emitting ``diphenylmethoxy`` (a
    tolyl/halophenyl neighbour must NOT be called diphenylmethoxy).
    """
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if aromatic_idx not in ring or len(ring) != 6:
            continue
        if not all(
            mol.GetAtomWithIdx(r).GetIsAromatic()
            and mol.GetAtomWithIdx(r).GetSymbol() == "C"
            for r in ring
        ):
            continue
        ring_set = set(ring)
        for r in ring:
            for nb in mol.GetAtomWithIdx(r).GetNeighbors():
                if nb.GetIdx() in ring_set or nb.GetAtomicNum() <= 1:
                    continue
                if nb.GetIdx() != central_c_idx:
                    return False  # ring carries a substituent -> not plain phenyl
        return True
    return False


def _name_aryl_methyl_ether(mol, central_c_idx: int, oxygen_idx: int) -> Optional[str]:
    """Name an ``-O-CH(aryl)ₙ`` ether substituent by counting the central
    carbon's aromatic neighbours (HYG-04, Phase 167).

    Single source of truth for the benzyloxy/diphenylmethoxy decision, replacing
    three byte-duplicated sites (``substituent_enumerator._name_alkoxy_branch``
    Case B and two ``composer`` sites) that returned ``benzyloxy`` for *any* aryl
    neighbour and so mis-named ``Ph₂CH-O-`` (diphenylmethyl ether) as benzyloxy.

    Returns:
      - ``"benzyloxy"``      — 1 aromatic neighbour (legacy behaviour, preserved),
      - ``"diphenylmethoxy"``— exactly 2 *unsubstituted phenyl* neighbours (IUPAC
        2013 PIN; D-07 — NOT the Beilstein ``benzhydryloxy``),
      - ``None``             — not a clean aryl-methyl ether (defer to existing logic).

    Byte-identical to the legacy guard (``central is non-aromatic C with ≥1 H and
    no non-H/non-aromatic heavy neighbour besides O``) except that the genuine
    diphenylmethyl case is upgraded from benzyloxy to diphenylmethoxy.
    """
    central = mol.GetAtomWithIdx(central_c_idx)
    if (
        central.GetIsAromatic()
        or central.GetSymbol() != "C"
        or central.GetTotalNumHs() < 1
    ):
        return None
    arom_nbrs = [
        n for n in central.GetNeighbors()
        if n.GetIdx() != oxygen_idx and n.GetIsAromatic()
    ]
    non_h_non_arom = [
        n for n in central.GetNeighbors()
        if n.GetIdx() != oxygen_idx
        and not n.GetIsAromatic()
        and n.GetSymbol() != "H"
    ]
    if not arom_nbrs or non_h_non_arom:
        return None
    if len(arom_nbrs) == 2 and all(
        _is_plain_phenyl(mol, n.GetIdx(), central_c_idx) for n in arom_nbrs
    ):
        return "diphenylmethoxy"
    return "benzyloxy"


def _check_retained_substituent(
    mol,
    sub_atoms: List[int],
    attach_idx: int,
) -> Optional[str]:
    """Check for the RETAINED PREFERRED substituent names only.

    Detects phenyl, benzyl, the retained cycloalkyls (cyclopropyl…cyclooctyl),
    and tert-butyl by analyzing the branching pattern at the attachment point.

    F-T9 / DD6 RET-02: the no-longer-recommended branched short-chain prefixes
    (isopropyl P-29.6.2.2; sec-butyl / isobutyl / neopentyl P-29.6.3) are NOT
    returned here. They are general-nomenclature-only forms whose PINs are the
    located/systematic names (propan-2-yl, butan-2-yl, 2-methylpropyl,
    2,2-dimethylpropyl), produced downstream by ``_located_acyclic_alkyl_name``
    (Step 2d / Tier 1.8). tert-butyl STAYS — it IS a preferred prefix per
    P-29.6.1 (Blue Book 16196 / 16286).

    Args:
        mol: RDKit Mol object.
        sub_atoms: Atom indices of the substituent.
        attach_idx: First atom of the substituent (bonded to parent chain).

    Returns:
        Retained name string, or None if no retained name applies.
    """
    frag_set = set(sub_atoms)

    # --- Phenyl detection ---
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        ring_set = set(ring)
        if ring_set.issubset(frag_set) and len(ring) == 6:
            if all(mol.GetAtomWithIdx(r).GetIsAromatic() and
                   mol.GetAtomWithIdx(r).GetSymbol() == 'C' for r in ring):
                # Has a benzene ring
                # Phase 125 fix: count ALL non-ring heavy atoms, not
                # just carbons.  Heteroatom substituents (Cl, OH, NH2,
                # F, Br, NO2) on the ring were invisible to the old
                # carbon-only check, causing "phenyl" to be returned
                # for substituted rings like 4-chlorophenyl.
                non_ring_heavy = sum(
                    1 for i in sub_atoms
                    if i not in ring_set
                    and mol.GetAtomWithIdx(i).GetAtomicNum() > 1
                )
                if non_ring_heavy == 0:
                    return "phenyl"
                elif non_ring_heavy == 1:
                    # Benzyl only if the attachment point is a non-ring
                    # carbon (CH2 bridging parent to ring). If the
                    # attachment point is a ring carbon, this is a
                    # substituted phenyl (e.g., 4-methylphenyl or
                    # 4-chlorophenyl), not benzyl.
                    non_ring_atoms = [
                        i for i in sub_atoms
                        if i not in ring_set
                        and mol.GetAtomWithIdx(i).GetAtomicNum() > 1
                    ]
                    if (len(non_ring_atoms) == 1
                            and mol.GetAtomWithIdx(non_ring_atoms[0]).GetSymbol() == 'C'
                            and attach_idx not in ring_set):
                        return "benzyl"

    # --- Cycloalkyl detection (IUPAC P-31.1.3.4) ---
    # Saturated carbocyclic rings used as substituents: cyclopropyl, cyclobutyl,
    # cyclopentyl, cyclohexyl, cycloheptyl, cyclooctyl.
    # Must be all-carbon, all-single-bond, no extra non-ring heavy atoms.
    _CYCLO_RETAINED = {
        3: 'cyclopropyl', 4: 'cyclobutyl', 5: 'cyclopentyl',
        6: 'cyclohexyl', 7: 'cycloheptyl', 8: 'cyclooctyl',
    }
    for ring in ring_info.AtomRings():
        ring_set = set(ring)
        if not ring_set.issubset(frag_set):
            continue
        ring_size = len(ring)
        if ring_size not in _CYCLO_RETAINED:
            continue
        # All ring atoms must be carbon
        if not all(mol.GetAtomWithIdx(r).GetSymbol() == 'C' for r in ring):
            continue
        # All bonds in ring must be single
        all_single = True
        for i in range(ring_size):
            bond = mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % ring_size])
            if bond and bond.GetBondTypeAsDouble() != 1.0:
                all_single = False
                break
        if not all_single:
            continue
        # No non-ring heavy atoms (unsubstituted ring only)
        non_ring_heavy = sum(
            1 for i in sub_atoms
            if i not in ring_set and mol.GetAtomWithIdx(i).GetAtomicNum() > 1
        )
        if non_ring_heavy == 0 and attach_idx in ring_set:
            return _CYCLO_RETAINED[ring_size]

    # --- Alkyl branching detection ---
    # Retained alkyl names (isopropyl, tert-butyl, etc.) only apply to
    # saturated fragments.  If any bond within the fragment is double or
    # triple, this is an unsaturated substituent (alkenyl/alkynyl) that
    # must be named systematically.
    carbon_atoms = [i for i in sub_atoms if mol.GetAtomWithIdx(i).GetSymbol() == 'C']
    carbon_count = len(carbon_atoms)

    if carbon_count == 0:
        return None

    for idx in sub_atoms:
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nbr.GetIdx() in frag_set:
                bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                if bond and bond.GetBondTypeAsDouble() != 1.0:
                    return None  # Unsaturated — use systematic naming

    attach_atom = mol.GetAtomWithIdx(attach_idx)
    if attach_atom.GetSymbol() != 'C':
        return None

    # Count carbon neighbors of the attachment atom within the fragment
    c_neighbors_in_frag = [
        nbr.GetIdx() for nbr in attach_atom.GetNeighbors()
        if nbr.GetIdx() in frag_set and nbr.GetSymbol() == 'C'
    ]

    # F-T9 / DD6 RET-02: only tert-butyl is a RETAINED PREFERRED prefix
    # (P-29.6.1, Blue Book 16196 / 16286). isopropyl (P-29.6.2.2) and
    # sec-butyl / isobutyl / neopentyl (P-29.6.3, no-longer-recommended) are
    # deliberately NOT returned: their PINs are the located/systematic forms
    # (propan-2-yl, butan-2-yl, 2-methylpropyl, 2,2-dimethylpropyl), built by
    # the general structure-derived producer `_located_acyclic_alkyl_name`
    # (Step 2d / Tier 1.8, shipped in E2/SEN-04) once this returns None.
    # Wave2 T5b hardening: the fragment must be EXACTLY the C4H9 skeleton —
    # a heteroatom-bearing fragment (e.g. -C(CH3)2CH2OH) satisfied the old
    # carbon-count test and was silently flattened to 'tert-butyl', DROPPING
    # the heteroatom (wrong constitution; SELF-01 was the only safety net).
    if (carbon_count == 4 and len(c_neighbors_in_frag) == 3
            and all(mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                    for i in sub_atoms)):
        # tert-butyl: C(CH3)3 -- 3 carbon branches at the attachment carbon.
        return "tert-butyl"

    return None


# ============================================================================
# Substituent CIP Stereo Descriptor
# ============================================================================


def _located_acyclic_alkyl_name(mol, sub_atoms, attach_idx):
    """Name an acyclic, all-carbon, saturated substituent by its OWN principal
    chain, numbered from the free valence (P-29.2 / P-46.1.8 / P-46.1.12).

    GENERAL structure-derived rule (DD5 RC-6 / SEN-04), no name table and NOT
    gated on a stereocentre. Returns ``(name, k)`` where ``name`` is the full
    located prefix and ``k`` is the free-valence locant, or ``None`` when the
    substituent is not an acyclic, all-carbon, saturated alkyl (caller then
    falls through to the recursive path).

    - The parent is the LONGEST carbon chain THROUGH the free-valence atom
      (P-29.2: a substituent's principal chain includes the atom with the free
      valence and is the longest such chain).
    - The free valence gets the LOWEST locant ``k`` (P-46.1.8).
    - The substituent's OWN substituents (off-chain branches) are numbered from
      that same chain and cited as prefixes (P-46.1.12), via the shared
      substituent namer.

    Forms (P-29.2 / P-29.6.2.3):
      - free valence terminal (k == 1): ``<sub-prefixes><stem>yl`` with the
        free-valence locant elided — e.g. ``3-methylbutyl``, ``4-methylpentyl``.
        (A k==1 chain with NO branches is a plain unbranched alkyl handled by
        the linear fast path; this deriver only reaches it defensively.)
      - free valence internal (k >= 2): ``<sub-prefixes><stem>an-k-yl`` — e.g.
        ``pentan-3-yl``, ``hexan-2-yl``, ``4-methylhexan-2-yl``.

    Replaces the former hardcoded ``sec-butyl -> butan-2-yl`` table and the
    branch-blind longest-chain deriver (which dropped a substituent's own methyl,
    e.g. ``3-methylbutyl`` was mis-named ``butyl``).
    """
    if attach_idx is None or not sub_atoms:
        return None
    sub_set = set(sub_atoms)
    if attach_idx not in sub_set:
        return None

    # Acyclic, saturated, uncharged; the CHAIN is all-carbon but degree-1
    # halogens and hydroxyl oxygens are allowed as simple BRANCHES (Wave2 T5b,
    # P-46.1.12: '2-bromo-4-chloropentan-3-yl', '1-hydroxypropan-2-yl').
    # Any other heteroatom shape (ether O, amino N, carbonyl via the bond-order
    # check, charged atoms) declines fail-closed to the recursive path.
    ring_info = mol.GetRingInfo()
    carbon_set = set()
    for idx in sub_set:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetFormalCharge() != 0:
            return None
        if ring_info.NumAtomRings(idx) > 0:
            return None
        sym = atom.GetSymbol()
        if sym == 'C':
            carbon_set.add(idx)
        elif sym in ('F', 'Cl', 'Br', 'I'):
            in_frag = [n for n in atom.GetNeighbors() if n.GetIdx() in sub_set]
            if atom.GetDegree() != 1 or len(in_frag) != 1 \
                    or in_frag[0].GetSymbol() != 'C':
                return None
        elif sym == 'O':
            # hydroxyl only: degree-1 O with H, bonded to a fragment carbon
            in_frag = [n for n in atom.GetNeighbors() if n.GetIdx() in sub_set]
            if atom.GetDegree() != 1 or atom.GetTotalNumHs() < 1 \
                    or len(in_frag) != 1 or in_frag[0].GetSymbol() != 'C':
                return None
        else:
            return None
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in sub_set:
                bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                if bond and bond.GetBondTypeAsDouble() != 1.0:
                    return None
    if attach_idx not in carbon_set:
        return None

    # Longest carbon chain THROUGH the free valence: the two longest arms out of
    # the attachment atom joined through it (the attachment is an interior or
    # terminal vertex of that chain). Iterative (explicit-stack post-order) so a
    # pathologically long chain cannot raise RecursionError (WR-03); result and
    # tie behaviour are identical to the recursive form (first-of-ties kept).
    def _longest_arm(start, came_from):
        best_child: Dict[int, List[int]] = {}
        stack = [(start, came_from, False)]
        while stack:
            node, parent, processed = stack.pop()
            if processed:
                best = [node]
                for nbr in mol.GetAtomWithIdx(node).GetNeighbors():
                    nidx = nbr.GetIdx()
                    if nidx in carbon_set and nidx != parent:
                        cand = [node] + best_child.get(nidx, [nidx])
                        if len(cand) > len(best):
                            best = cand
                best_child[node] = best
            else:
                stack.append((node, parent, True))
                for nbr in mol.GetAtomWithIdx(node).GetNeighbors():
                    nidx = nbr.GetIdx()
                    if nidx in carbon_set and nidx != parent:
                        stack.append((nidx, node, False))
        return best_child[start]

    arms = []
    for nbr in mol.GetAtomWithIdx(attach_idx).GetNeighbors():
        if nbr.GetIdx() in carbon_set:
            arms.append(_longest_arm(nbr.GetIdx(), attach_idx))
    arms.sort(key=len, reverse=True)

    def _branch_locs(order):
        oset = set(order)
        opos = {a: i + 1 for i, a in enumerate(order)}
        locs = []
        for ci in order:
            for nb in mol.GetAtomWithIdx(ci).GetNeighbors():
                ni = nb.GetIdx()
                if ni in oset or ni not in sub_set:
                    continue
                locs.append(opos[ci])
        return sorted(locs)

    def _orient(chain):
        """Number so the free valence gets the LOWEST locant (P-46.1.8);
        on an exact-centre tie, break toward the LOWEST side-chain locant
        set at first point of difference (Wave2 T3c, P-29.4.1):
        '2-methylpentan-3-yl' not '4-methylpentan-3-yl'."""
        _fwd_k = chain.index(attach_idx) + 1
        _rev_k = len(chain) - chain.index(attach_idx)
        if _rev_k < _fwd_k:
            return list(reversed(chain))
        if _rev_k == _fwd_k and \
                _branch_locs(list(reversed(chain))) < _branch_locs(chain):
            return list(reversed(chain))
        return chain

    # Wave2 T5b: when several arm PAIRS tie for the longest chain (a
    # quaternary/branched attach with equal arms), the pick was atom-order-
    # dependent. Enumerate every maximal-length pair and choose by the P-44.4
    # cascade tail: (free-valence locant k) -> (greatest number of detachable
    # prefixes) -> (lowest branch-locant set): -C(CH3)2CH2OH must pick the
    # chain THROUGH the CH2 so the name is '1-hydroxy-2-methylpropan-2-yl',
    # never the lossy-locant '(hydroxymethyl)' form. Pure-alkyl fragments are
    # unaffected (tied pairs there are symmetric).
    if len(arms) >= 2:
        best_total = len(arms[0]) + len(arms[1])
        candidates = [
            list(reversed(arms[i])) + [attach_idx] + arms[j]
            for i in range(len(arms))
            for j in range(i + 1, len(arms))
            if len(arms[i]) + len(arms[j]) == best_total
        ]
    elif len(arms) == 1:
        candidates = [[attach_idx] + arms[0]]
    else:
        candidates = [[attach_idx]]

    chain = None
    _best_key = None
    for cand in candidates:
        oriented = _orient(cand)
        locs = _branch_locs(oriented)
        key = (oriented.index(attach_idx) + 1, -len(locs), locs)
        if _best_key is None or key < _best_key:
            _best_key = key
            chain = oriented

    chain_len = len(chain)
    if chain_len < 2:
        return None
    k = chain.index(attach_idx) + 1
    chain_set_c = set(chain)
    chain_pos = {a: i + 1 for i, a in enumerate(chain)}

    # Off-chain branches -> the substituent's own substituents (P-46.1.12),
    # named by the shared substituent namer and located on this chain.
    from collections import defaultdict
    from .substituent_enumerator import name_substituent
    branch_groups: dict = defaultdict(list)
    for chain_atom in chain:
        for nbr in mol.GetAtomWithIdx(chain_atom).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in chain_set_c or nidx not in sub_set:
                continue
            # BFS the branch fragment within the substituent (excluding the chain).
            frag = []
            seen = set(chain_set_c)
            stack = [nidx]
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                frag.append(cur)
                for nn in mol.GetAtomWithIdx(cur).GetNeighbors():
                    if nn.GetIdx() in sub_set and nn.GetIdx() not in seen:
                        stack.append(nn.GetIdx())
            # Single-atom simple branches (Wave2 T5b): the guard loop above
            # admitted only degree-1 halogens / hydroxyl O, so a 1-atom
            # non-carbon frag maps directly to its prefix.
            _SIMPLE_BRANCH = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo',
                              'I': 'iodo', 'O': 'hydroxy'}
            _sym0 = mol.GetAtomWithIdx(nidx).GetSymbol()
            if len(frag) == 1 and _sym0 in _SIMPLE_BRANCH:
                branch_groups[_SIMPLE_BRANCH[_sym0]].append(
                    chain_pos[chain_atom])
                continue
            try:
                bname = name_substituent(mol, frag, nidx)
            except Exception as exc:  # noqa: BLE001
                # Missing-beats-wrong: a branch we cannot name -> decline the
                # whole located form (fall through to the recursive path). Logged
                # at debug so a genuine namer bug here stays observable (IN-02).
                logger.debug("located-alkyl branch naming failed: %s", exc)
                return None
            if not bname or bname == "substituent":
                return None
            branch_groups[bname].append(chain_pos[chain_atom])

    # Wave2 T5b (P-14.5.2(e)): when the free valence sits at the exact chain
    # centre AND the branch-locant multiset is direction-invariant (the T3c
    # first-point-of-difference tie-break above could not decide), assign the
    # lowest locants to the substituent cited FIRST in alphanumerical order:
    # Br@2/Cl@4 -> '2-bromo-4-chloropentan-3-yl', never '4-bromo-2-chloro'.
    # Renumbering position l on the reversed chain is chain_len+1-l, so the
    # flip is a pure remap — no re-collection (branch membership is
    # direction-independent). Also removes a latent atom-order dependence for
    # different-name equal-set branches.
    if branch_groups and k == chain_len + 1 - k:
        _cur_all = sorted(l for locs in branch_groups.values() for l in locs)
        _flip_all = sorted(chain_len + 1 - l for l in _cur_all)
        if _cur_all == _flip_all:
            from .naming_utils import alpha_sort_key
            _names_alpha = sorted(branch_groups, key=alpha_sort_key)
            _cur_seq = [tuple(sorted(branch_groups[n])) for n in _names_alpha]
            _flip_seq = [
                tuple(sorted(chain_len + 1 - l for l in branch_groups[n]))
                for n in _names_alpha
            ]
            if _flip_seq < _cur_seq:
                branch_groups = {
                    n: sorted(chain_len + 1 - l for l in locs)
                    for n, locs in branch_groups.items()
                }

    from ..data.chain_names import get_chain_prefix
    if k == 1 and not branch_groups:
        # Unbranched primary alkyl (normally handled upstream); plain name.
        try:
            return (get_alkyl_name(chain_len), 1)
        except (ValueError, KeyError):
            return None
    try:
        stem = get_chain_prefix(chain_len)
    except (ValueError, KeyError):
        return None

    from .composer import _format_prefix_groups
    prefix = _format_prefix_groups(branch_groups) if branch_groups else ""

    if k == 1:
        try:
            base = get_alkyl_name(chain_len)  # 'butyl', 'pentyl', ...
        except (ValueError, KeyError):
            base = f"{stem}yl"
        return (f"{prefix}{base}", 1)
    return (f"{prefix}{stem}an-{k}-yl", k)


def _acyclic_alkyl_located_stereo_name(mol, sub_atoms, attach_idx):
    """Stereo-path adapter for the located acyclic-alkyl deriver.

    Thin wrapper kept for ``_add_substituent_stereo`` (which prepends the
    ``({k}{cip})-`` descriptor only when the attachment atom IS the stereocentre,
    i.e. ``s_idx == attach_idx``). The structure derivation — including the
    substituent's OWN substituents and free-valence numbering — now lives in the
    general, un-gated ``_located_acyclic_alkyl_name`` (DD5 RC-6 / SEN-04). The old
    ``HasProp('_CIPCode')`` gate here is no longer needed: the caller's
    ``s_idx == attach_idx`` check already restricts descriptor emission to the
    stereocentre-at-attachment case.

    Worked examples (BlueBook P-29.6.2.3): ``[C@@H](C)CC`` -> ("butan-2-yl", 2),
    ``[C@@H](C)CCC`` -> ("pentan-2-yl", 2).
    """
    return _located_acyclic_alkyl_name(mol, sub_atoms, attach_idx)


def _add_substituent_stereo(mol, sub_atoms, name, attach_idx=None):
    """Add CIP stereodescriptors to a substituent name if stereocenters exist.

    When a substituent contains one or more stereocenters with defined CIP
    labels (R/S), the descriptor is prepended: e.g. a stereogenic secondary
    acyclic alkyl becomes "(2S)-butan-2-yl"; a parent-hydride-style substituent
    with a unique stereo position becomes "(R)-name".

    For a single stereocenter the format is "(R)-name" or "({k}R)-name".
    For multiple stereocenters the format uses locants threaded from the
    substituent's own numbering (or emits none — D-09).

    This is a systemic fix: any substituent on any parent (chain, ring,
    heterocycle) that has a stereocenter gets the descriptor.

    Args:
        mol: RDKit Mol object (CIP labels must already be assigned).
        sub_atoms: Atom indices of the substituent fragment.
        name: The substituent name without stereo (e.g., "sec-butyl").
        attach_idx: The substituent's attachment atom index, when known. Used to
            derive the located descriptor (locant + PIN systematic name) for an
            acyclic-alkyl substituent from STRUCTURE. None when the caller has no
            attachment context (then the located form is not derivable and the
            bare "(R)-"/"(S)-" form is used per the unique-position rule).

    Returns:
        Name with stereo prefix if stereocenters found, otherwise unchanged.
    """
    if not sub_atoms or not name:
        return name

    # Ensure CIP labels are assigned (idempotent guard)
    assign_stereochemistry(mol)

    # Collect CIP-labeled atoms within the substituent
    stereo_atoms = []
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.HasProp('_CIPCode'):
            # SUB-04/D-15: inherit the perception verdict verbatim (the single
            # source of truth). The former force-uppercase of pseudo-asymmetric
            # r/s was a latent D-15 violation — it would CORRUPT a genuine
            # substituent-position pseudo-asymmetric centre. Verified dead on the
            # whole corpus (0 hits across the genuine-13 + ceramide leaks + 400
            # stereo molecules), so removal is zero-regression.
            cip = atom.GetProp('_CIPCode')
            stereo_atoms.append((idx, cip))

    if not stereo_atoms:
        return name

    if len(stereo_atoms) == 1:
        # Single stereocenter.
        s_idx, cip = stereo_atoms[0]
        # STEREO-03 (Phase 177 WSB-02 D-09): a stereogenic acyclic-alkyl
        # substituent needs the PIN systematic name + the descriptor at its
        # attachment locant (P-29.2 / P-31.1.4.3.4 / P-91). The name and the
        # locant are derived FROM STRUCTURE (the substituent atoms + attachment
        # atom) by `_acyclic_alkyl_located_stereo_name` — NOT a hardcoded
        # name->PIN table and NOT a BFS-from-attachment heuristic. This covers
        # sec-butyl (-> butan-2-yl, k=2), pentan-2-yl (-> pentan-2-yl, k=2), and
        # any stereogenic secondary acyclic alkyl. Examples (BlueBook P-29.6.2.3:
        # `butan-2-yl` is the preferred prefix; `sec-butyl` is NOT a PIN):
        #   CC(=O)N[C@@H](C)CC  -> N-[(2S)-butan-2-yl]acetamide
        #   CC(=O)N[C@@H](C)CCC -> N-[(2S)-pentan-2-yl]acetamide
        located = _acyclic_alkyl_located_stereo_name(mol, sub_atoms, attach_idx)
        if located is not None and s_idx == attach_idx:
            pin_form, loc = located
            return f"({loc}{cip})-{pin_form}"
        # Otherwise: a single stereocenter on a parent-hydride-style substituent
        # (or any case where a structure-derived located form is not available)
        # takes the bare "(R)-"/"(S)-" (no locant) per the P-91 unique-position
        # rule. Per D-09 a missing locant is never fabricated.
        return f"({cip})-{name}"

    # Multiple stereocenters within one substituent.
    #
    # WR-06 fix (Phase 177 WSB-02 D-07/D-09): the former raw-atom-index
    # positional locants (`idx_to_internal = {idx: pos+1 for ...}`) were WRONG —
    # they number a leading heteroatom "1" and bear no relation to the IUPAC
    # numbering the substituent's NAME actually used. The correct locant must be
    # threaded from the SAME chain/ring numbering the substituent name used
    # (name_fragment_recursively / parent_to_prefix), which is not available at
    # this signature without a recursion-contract change. Re-deriving an
    # independent numbering (e.g. BFS-from-attachment) is explicitly rejected as
    # "a DIFFERENT wrong heuristic — a band-aid the fix-methodology forbids."
    #
    # Per D-09 (missing beats wrong): when the substituent's own numbering
    # cannot be threaded, emit NO multi-centre stereo block — return the name
    # unchanged. A multi-centre stereo descriptor placed with fabricated locants
    # is worse than none (it would round-trip to the WRONG diastereomer). The
    # single-centre branch above still emits the correct unlocanted "(R)-"/"(S)-".
    return name


# ============================================================================
# Main Entry Point
# ============================================================================


def _find_amine_oxide_n(mol, sub_atoms):
    """P-62.5(2) detector: an amine-oxide nitrogen inside a substituent
    fragment — N with formal charge +1, ALL bonds single (excludes nitro,
    azide, aromatic N-oxides), exactly one terminal single-bonded O- inside
    the fragment. Returns (n_idx, o_idx) or None. Two oxide-N in one
    fragment -> None (out of scope, fail closed at the caller)."""
    from rdkit import Chem
    sub_set = set(sub_atoms)
    found = None
    for idx in sub_atoms:
        a = mol.GetAtomWithIdx(idx)
        if a.GetSymbol() != 'N' or a.GetFormalCharge() != 1 or a.GetIsAromatic():
            continue
        if any(b.GetBondType() != Chem.BondType.SINGLE for b in a.GetBonds()):
            continue
        o_minus = [n.GetIdx() for n in a.GetNeighbors()
                   if n.GetSymbol() == 'O' and n.GetFormalCharge() == -1
                   and n.GetDegree() == 1 and n.GetIdx() in sub_set]
        if len(o_minus) != 1:
            continue
        if found is not None:
            return None
        found = (idx, o_minus[0])
    return found


def _collect_plain_alkyl_branch(mol, start_idx, blocked):
    """BFS a plain saturated acyclic all-carbon branch; None if anything
    else is reached (fail-closed helper for _lambda5_azanyl_prefix). Distinct
    from _collect_branch_atoms (which uses a single block_idx and does not
    reject heteroatoms)."""
    from rdkit import Chem
    seen, queue = set(), [start_idx]
    while queue:
        i = queue.pop()
        if i in seen or i in blocked:
            continue
        a = mol.GetAtomWithIdx(i)
        if (a.GetSymbol() != 'C' or a.GetIsAromatic() or a.IsInRing()
                or a.GetFormalCharge() != 0):
            return None
        if any(b.GetBondType() != Chem.BondType.SINGLE for b in a.GetBonds()):
            return None
        seen.add(i)
        queue.extend(nb.GetIdx() for nb in a.GetNeighbors()
                     if nb.GetIdx() not in blocked)
    return sorted(seen)


def _lambda5_azanyl_prefix(mol, sub_atoms, attach_idx):
    """P-62.5(2): build the lambda5-azane substituent prefix for a fragment
    whose terminal atom is an amine-oxide N reached through an UNBRANCHED
    saturated all-carbon chain from the attachment atom:
      -CH2-N+(CH3)2(O-)   -> '[dimethyl(oxo)-lambda5-azanyl]methyl'
      -CH2-CH2-NH2+(O-)   -> '2-(oxo-lambda5-azanyl)ethyl'
    Fail-closed: branched chains, ring/aromatic/charged chain atoms,
    non-alkyl N-substituents, or uncovered fragment atoms -> None."""
    from rdkit import Chem
    from .naming_utils import get_multiplier_prefix, get_alkyl_name
    hit = _find_amine_oxide_n(mol, sub_atoms)
    if hit is None:
        return None
    n_idx, o_idx = hit
    # Determinism guard (P-62.5 / P-44): choosing the principal among two
    # equal-seniority amine-oxide centres is an atom-order-dependent parent
    # tie-break that is not yet built. Naming the substituent-side oxide while
    # that choice is nondeterministic would make the whole-molecule name
    # order-dependent -> fail closed (deterministic unknown) for molecules with
    # >1 amine-oxide nitrogen. Single-substituent-oxide molecules are unaffected.
    _all_ox_n = sum(
        1 for a in mol.GetAtoms()
        if a.GetSymbol() == 'N' and a.GetFormalCharge() == 1
        and not a.GetIsAromatic()
        and all(b.GetBondType() == Chem.BondType.SINGLE for b in a.GetBonds())
        and sum(1 for nb in a.GetNeighbors()
                if nb.GetSymbol() == 'O' and nb.GetFormalCharge() == -1
                and nb.GetDegree() == 1) == 1
    )
    if _all_ox_n > 1:
        return None
    sub_set = set(sub_atoms)
    # walk attach -> ... -> C bonded to N (plain saturated unbranched C chain)
    chain = []
    prev, cur = None, attach_idx
    while True:
        a = mol.GetAtomWithIdx(cur)
        if (a.GetSymbol() != 'C' or a.GetIsAromatic() or a.IsInRing()
                or a.GetFormalCharge() != 0):
            return None
        chain.append(cur)
        nxt = [nb.GetIdx() for nb in a.GetNeighbors()
               if nb.GetIdx() in sub_set and nb.GetIdx() != prev]
        if any(mol.GetBondBetweenAtoms(cur, i).GetBondType()
               != Chem.BondType.SINGLE for i in nxt):
            return None
        if nxt == [n_idx]:
            break
        if len(nxt) != 1:
            return None
        prev, cur = cur, nxt[0]
    # N substituents besides the chain carbon and the oxide O: plain alkyls
    covered = set(chain) | {n_idx, o_idx}
    alkyl_names = []
    n_atom = mol.GetAtomWithIdx(n_idx)
    for nb in n_atom.GetNeighbors():
        i = nb.GetIdx()
        if i in (o_idx, chain[-1]):
            continue
        if i not in sub_set:
            return None
        branch = _collect_plain_alkyl_branch(mol, i, {n_idx})
        if branch is None:
            return None
        bname = name_substituent_fragment(mol, branch, i, [n_idx])
        if bname is None or not bname.isalpha():
            return None  # only simple unlocanted alkyls (methyl, ethyl, ...)
        alkyl_names.append(bname)
        covered.update(branch)
    if covered != sub_set:
        return None  # never drop an atom silently
    alkyl_names.sort()
    if not alkyl_names:
        core = "(oxo-lambda5-azanyl)"
    elif len(alkyl_names) == 2 and alkyl_names[0] == alkyl_names[1]:
        mp = get_multiplier_prefix(2, alkyl_names[0])
        core = f"[{mp}{alkyl_names[0]}(oxo)-lambda5-azanyl]"
    elif len(alkyl_names) == 1:
        core = f"[{alkyl_names[0]}(oxo)-lambda5-azanyl]"
    else:
        return None
    chain_len = len(chain)
    alkyl = get_alkyl_name(chain_len)  # 'methyl', 'ethyl', ...
    if alkyl is None:
        return None
    if chain_len == 1:
        return f"{core}{alkyl}"
    return f"{chain_len}-{core}{alkyl}"


def name_substituent_fragment(
    mol,
    sub_atoms: List[int],
    attach_idx: int,
    parent_chain: list,
) -> Optional[str]:
    """Name a substituent fragment, routing to the appropriate naming path.

    This is the centralized entry point for all substituent naming.
    It detects the complexity of the substituent and routes accordingly:
    1. Retained PREFERRED names: phenyl, benzyl, retained cycloalkyls, tert-butyl.
    2. Linear terminal alkyl (fast path): get_alkyl_name() directly.
    2d. Located acyclic alkyl: branched/internal attachment -> _located_acyclic_alkyl_name
        (propan-2-yl, butan-2-yl, 2-methylpropyl) — the F-T9/DD6 RET-02 PINs.
    3. Recursive naming: extract SMILES, name recursively, convert to prefix.

    Args:
        mol: RDKit Mol object.
        sub_atoms: Atom indices of the substituent fragment.
        attach_idx: First atom of the substituent (bonded to parent chain).
        parent_chain: List of atom indices in the parent chain.

    Returns:
        Raw prefix name (e.g., "methyl", "propan-2-yl", "2-methylpropyl")
        WITHOUT enclosing marks. Returns None if naming fails.
    """
    if not sub_atoms:
        return None

    parent_set = set(parent_chain) if parent_chain else set()

    # Step 0 (Wave-2 C2, P-63.2.2): an O-ATTACHED fragment is an R-oxy prefix.
    # The generic Steps 3-5 mis-anchored it ('1-hydroxy-1-methoxymethyl' for
    # -O-CH2-O-CH3, constitution-wrong and SELF-01-suppressed) — recurse on
    # the carbon portion and convert yl -> oxy; fail closed on anything the
    # recursion cannot express (never garbage).
    if len(sub_atoms) >= 2:
        _a = mol.GetAtomWithIdx(attach_idx)
        if (_a.GetSymbol() == 'O' and _a.GetFormalCharge() == 0
                and _a.GetTotalNumHs() == 0 and not _a.IsInRing()):
            _inner_nbrs = [n.GetIdx() for n in _a.GetNeighbors()
                           if n.GetIdx() in set(sub_atoms)]
            if len(_inner_nbrs) == 1 and mol.GetAtomWithIdx(
                    _inner_nbrs[0]).GetSymbol() == 'C':
                _rest = [i for i in sub_atoms if i != attach_idx]
                _inner = name_substituent_fragment(
                    mol, _rest, _inner_nbrs[0], parent_chain + [attach_idx])
                # Fail closed on generic-path mis-expressions: 'hydroxy'/
                # 'oxo' garbage for embedded ethers the C-attach recursion
                # cannot yet express (depth >=2 nesting stays closed).
                if (not _inner or _inner == 'substituent'
                        or 'hydroxy' in _inner or 'oxo' in _inner
                        or _inner.endswith('ylyl')):
                    return None
                if _inner.endswith('yl'):
                    _stem = _inner[:-2] + 'oxy'
                else:
                    _stem = f'({_inner})oxy'
                return _stem

    # Step 0b (P-62.5(2)): a fragment carrying an amine-oxide nitrogen is
    # named ONLY via the lambda5-azanyl builder. The generic recursion drops
    # the oxide (raw HEAD emitted '(methylmethyl)methyl' for
    # -CH2-N+(CH3)2(O-), a different molecule) -> if the builder declines,
    # FAIL CLOSED rather than fall through.
    if _find_amine_oxide_n(mol, sub_atoms) is not None:
        return _lambda5_azanyl_prefix(mol, sub_atoms, attach_idx)

    # Step 1: Check retained PREFERRED substituent names FIRST (phenyl, benzyl,
    # retained cycloalkyls, tert-butyl). F-T9/DD6 RET-02: isopropyl/sec-butyl/
    # isobutyl/neopentyl are NOT returned (their located PINs come from Step 2d).
    # The retained check runs before the linear fast path, which cannot
    # distinguish e.g. propyl from a branched 3-carbon attachment.
    retained = _check_retained_substituent(mol, sub_atoms, attach_idx)
    if retained:
        return _add_substituent_stereo(mol, sub_atoms, retained, attach_idx=attach_idx)

    # Step 1b (BBR-PERC, 169.7): chalcogen-ether substituent -Se-R / -Te-R →
    # (alkyl)selanyl / (alkyl)tellanyl (P-63.6). Without this, Step 4's recursive
    # path names it as the parent hydride "methaneselenol" → "methaneselenolyl",
    # which OPSIN cannot parse (so the validity gate then suppresses the whole
    # name to "unknown"). The chalcogen-agnostic prefix builder walks the C
    # neighbours of the attach chalcogen; the substituent C (not on the parent
    # chain) is named the (alkyl) stem.
    _attach = mol.GetAtomWithIdx(attach_idx)
    if _attach.GetSymbol() in ('Se', 'Te'):
        from .substituent_prefix_forms import get_sulfanyl_prefix
        _suffix = 'selanyl' if _attach.GetSymbol() == 'Se' else 'tellanyl'
        _nbrs = [n.GetIdx() for n in _attach.GetNeighbors()]
        if len(_nbrs) >= 2:
            _chal = get_sulfanyl_prefix(
                mol, (attach_idx, _nbrs[0], _nbrs[1]), parent_chain, suffix=_suffix
            )
            if _chal:
                return _add_substituent_stereo(mol, sub_atoms, _chal, attach_idx=attach_idx)

    # Step 1d (v23 Phase 8, P-68.2.2): Group-14 (Si/Ge) substituent ->
    # (prefixes)silyl / (prefixes)germyl. Without this, Step 4's recursive path
    # names a bare -SiH3 as the free molecule ('unknown organic compound') + 'yl'
    # = 'unknown organic compoundyl' (then dropped on a senior carbon parent,
    # losing the silyl), and an -Si(OH)3 as the parent-hydride -ol suffix
    # ('silanetriol') + 'yl' = 'silanetriolyl' (OH wrongly kept as a suffix).
    # Returns the correct silyl/germyl prefix, or None (fall through) for a
    # multivalent / complex centre — byte-identical for the simple-organyl forms
    # (trimethylsilyl) the recursive path already produces. (Same two-namer wiring
    # as the Phase-4 SUBST-01 lesson: name_substituent AND this function.)
    if _attach.GetSymbol() in ('Si', 'Ge'):
        _g14 = _name_group14_substituent(mol, sub_atoms, attach_idx)
        if _g14:
            return _g14

    # Wave2 T3c (P-31.1.3.4): aryl-vinyl / styryl — an acyclic UNSATURATED chain
    # bearing a ring substituent (Ar-CH=CH- -> (E)-2-phenylethenyl). MUST precede
    # the Step 1c ring chokepoint (which declines because the free valence is on
    # the acyclic vinyl C, not a ring) and the recursive path (which mis-anchors
    # the free valence onto the ring -> the invalid 'ethenylbenzenyl'). Returns
    # None for saturated / branched / non-aryl-arm cases -> falls through.
    if attach_idx is not None:
        _av = _name_aryl_vinyl_substituent(mol, sub_atoms, attach_idx, parent_set)
        if _av is not None:
            return _av

    # Step 1c (Phase 4 SUBST-01): ring-bearing fragment -> the trustworthy ring
    # chokepoint, BEFORE the recursive / cache paths which drop ene/yne locants
    # ('cyclohexenyl' for cyclohex-1-en-1-yl) or name a ring-on-chain as a
    # different molecule ('methylcyclohexyl' for cyclohexylmethyl) or guess a
    # monocycle for a polycyclic (cycloheptyl for norbornane). The benzene /
    # ring-parent assemblers route their substituents here, so this is the locus
    # that fixes them. allow_enumerator_fallback=False = recursion guard (the
    # chokepoint's own fallback is the enumerator's name_substituent, which calls
    # back here at Tier 4).
    if attach_idx is not None:
        try:
            _ri = mol.GetRingInfo()
            if any(_ri.NumAtomRings(a) > 0 for a in sub_atoms):
                from ..rules.ring_substituents import name_ring_system_substituent
                _ring_nm = name_ring_system_substituent(
                    mol, sorted(sub_atoms), attach_idx,
                    allow_enumerator_fallback=False,
                )
                if _ring_nm:
                    return _ring_nm
        except Exception:
            pass

    # Step 2: Fast path -- linear saturated alkyl, attached at a chain TERMINUS.
    # DD5 RC-6 / SEN-04: a linear chain attached at an INTERNAL carbon (e.g. the
    # central C of pentan-3-yl, the 2-C of hexan-2-yl) is NOT a terminal alkyl —
    # it must be numbered from the free valence (Step 2d). The free-valence-locant
    # is only elided (the bare 'pentyl' form) when the attachment is terminal.
    if _is_linear_alkyl(mol, sub_atoms) and _attach_is_chain_terminus(
        mol, sub_atoms, attach_idx
    ):
        carbon_count = sum(
            1 for i in sub_atoms
            if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
        )
        if carbon_count > 0:
            try:
                return get_alkyl_name(carbon_count)
            except (ValueError, KeyError):
                return None
        return None

    # Step 2b: Unsaturated linear chain (all C, no branching, has C=C or C#C).
    # Must be handled before general recursion because the recursive path
    # doesn't know the attachment point, producing wrong locants
    # (e.g., "prop-1-enyl" instead of "prop-2-en-1-yl" for allyl).
    unsat_name = _name_unsaturated_chain(mol, sub_atoms, attach_idx, parent_set)
    if unsat_name is not None:
        return _add_substituent_stereo(mol, sub_atoms, unsat_name, attach_idx=attach_idx)

    # Step 2c (Phase 171 BBR-ASM, DEF-8 / P-46): saturated linear chain with halogen
    # substituents, numbered from the attachment point. MUST precede the recursive
    # path, which renames the extracted fragment as a free molecule and loses the
    # attachment constraint ('CCCCCl' -> '1-chlorobutane' -> 'chlorobutyl', no locant).
    halo_name = _name_saturated_substituted_chain(mol, sub_atoms, attach_idx, parent_set)
    if halo_name is not None:
        return _add_substituent_stereo(mol, sub_atoms, halo_name, attach_idx=attach_idx)

    # Step 2c-poly (v23 SL — substituent structure-loss / locant-drop): a saturated
    # acyclic carbon chain bearing >=1 simple detachable prefixes (carboxy/amino/
    # hydroxy/oxo/halogen), numbered from the free valence. MUST precede the
    # recursive path, which caps the fragment to a free molecule and lets
    # parent_to_prefix DROP the secondary prefixes (serine-O -CH2CH(NH2)COOH ->
    # '(R)-2-carboxyethyl', amino lost), inherit the parent's lowest-locant
    # numbering (-CH2CH2CH2OH -> '1-hydroxypropyl', wrong end), or map a 1-carbon
    # acid to its retained acyl (-CH2COOH -> 'acetyl'). Fail-closed for rings /
    # branched / unsaturated / amides / ethers / bare-acyl (oxo on the attach C).
    poly_name = _name_polyfunctional_acyclic_substituent(
        mol, sub_atoms, attach_idx, parent_set
    )
    if poly_name is not None:
        return _add_substituent_stereo(mol, sub_atoms, poly_name, attach_idx=attach_idx)

    # Step 2c-ether (v22 C-T2 / V-3, P-63.2.2.2): saturated all-carbon chain
    # bearing ether -O-R substituent(s), numbered from the attachment. MUST
    # precede the recursive path, which caps the fragment to a free molecule and
    # mis-names a retained ether as '<molecule>yl' (-CH2-O-C6H5 -> 'anisolyl', a
    # different constitution). Returns None for anything else -> falls through.
    ether_name = _name_ether_substituted_chain(mol, sub_atoms, attach_idx, parent_set)
    if ether_name is not None:
        return _add_substituent_stereo(mol, sub_atoms, ether_name, attach_idx=attach_idx)

    # Step 2d (DD5 RC-6 / SEN-04, P-29.2 / P-46): branched / secondary acyclic
    # all-carbon alkyl substituent named by its OWN principal chain numbered from
    # the free valence (hexan-2-yl, pentan-3-yl, 3-methylbutyl). MUST precede the
    # recursive Step 3, which caps the free valence with H and renames the fragment
    # as a free molecule — losing the attachment so a secondary attachment becomes
    # a terminal alkyl (hexan-2-yl -> 'hexyl') and a branch is numbered from the
    # wrong end (3-methylbutyl -> '2-methylbutyl', a DIFFERENT constitution).
    # Returns None for anything not an acyclic all-carbon saturated alkyl -> falls
    # through. Retained prefixes (isopropyl/sec-butyl/…) are already handled by
    # Step 1, so they are not reached here.
    located = _located_acyclic_alkyl_name(mol, sub_atoms, attach_idx)
    if located is not None:
        located_name, _k = located
        return _add_substituent_stereo(mol, sub_atoms, located_name, attach_idx=attach_idx)

    # Wave2 T3a constitution-conservation guard: a ring-bearing fragment
    # reaching this point means the trustworthy ring chokepoint (Step 1c)
    # DECLINED it. Steps 3-5 would cap the fragment, name it as a free
    # molecule, and string-surgery the suffix into a prefix — anchoring the
    # free valence wherever the name form implies, NOT at attach_idx
    # (-CH2-O-CH2-Ar(OH) became 'hydroxy4-(methoxymethyl)phenyl': a direct
    # ring-parent bond, a DIFFERENT constitution). Fail closed instead; the
    # ring engine is the only namer that anchors ring fragments honestly.
    try:
        _ri_guard = mol.GetRingInfo()
        if any(_ri_guard.NumAtomRings(a) > 0 for a in sub_atoms):
            logger.debug(
                "DROP-24 substituent_skip: reason=ring_fragment_declined_by_"
                "ring_engine atom_count=%d", len(sub_atoms),
            )
            return None
    except Exception:
        pass

    # Step 3: Extract fragment SMILES and name recursively
    frag_smiles = _extract_fragment_smiles(mol, sub_atoms, attach_idx, parent_set)
    if frag_smiles is None:
        # Fallback: try simple carbon count for pure-carbon substituents
        carbon_count = sum(
            1 for i in sub_atoms
            if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
        )
        if carbon_count > 0:
            try:
                return get_alkyl_name(carbon_count)
            except (ValueError, KeyError):
                pass
        logger.warning(
            "DROP-11 substituent_skip: reason=smiles_extraction_failure atom_count=%d",
            len(sub_atoms),
        )
        return None

    # Step 4: Recursive naming via name_fragment_recursively()
    parent_name = name_fragment_recursively(frag_smiles)
    if parent_name is None:
        # Recursion depth limit or naming failure: fallback to carbon count
        carbon_count = sum(
            1 for i in sub_atoms
            if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
        )
        if carbon_count > 0:
            try:
                return get_alkyl_name(carbon_count)
            except (ValueError, KeyError):
                pass
        logger.warning(
            "DROP-12 substituent_skip: reason=recursion_depth_fallback frag_smiles=%s",
            frag_smiles[:60],
        )
        return None

    # Step 5: Convert parent name to prefix form
    carbon_count = sum(
        1 for i in sub_atoms
        if mol.GetAtomWithIdx(i).GetSymbol() == 'C'
    )
    prefix_name = parent_to_prefix(parent_name, chain_length=carbon_count)

    return _add_substituent_stereo(mol, sub_atoms, prefix_name, attach_idx=attach_idx)


# ============================================================================
# Convenience: check if substituent needs recursive naming
# ============================================================================


def needs_recursive_naming(mol, sub_atoms: List[int]) -> bool:
    """Check whether a substituent requires recursive naming.

    Returns True if the substituent is branched or contains heteroatoms
    (i.e., not a simple linear alkyl chain).

    Args:
        mol: RDKit Mol object.
        sub_atoms: Atom indices of the substituent.

    Returns:
        True if recursive naming may be needed, False for simple linear alkyls.
    """
    return not _is_linear_alkyl(mol, sub_atoms)


# ============================================================================
# CATION-AS-SUBSTITUENT PREFIX PRODUCER (P-74.0 / P-74.1.3) — 169.6-04
# ============================================================================
#
# P-74.0 (verbatim, BlueBookV2 line 42411): "an anionic center has priority
# over a cationic center in zwitterions ... anionic centers ... become the
# parent structure, into which the cationic part is substituted." P-74.1.3
# (line 42464): when the cationic and anionic centers sit on DIFFERENT parent
# structures, the cation is "prefix[ed] ... to the name of the anionic parent
# structure."
#
# This is the NEW structured producer the chokepoint had ZERO of before
# (parent_to_prefix above is neutral-only — no aminiumyl/ammoniumyl/azaniumyl
# capability). It is GENERAL (fix-methodology.md): no per-molecule branch, no
# hardcoded f-string. It REPLACES the deleted salts.py betaine literal +
# "2-azaniumyl{base}" carbon-counting f-string.
#
# FORM CHOICE (OPSIN-grounded, 169.6-04 deviation, documented in SUMMARY):
# The Blue Book PIN for a quaternary-ammonium cation substituent is the
# `-aminiumyl` form (`(N,N-dimethylmethanaminiumyl)acetate`, line 42470). OPSIN
# 2.9.0 does NOT implement the `-aminiumyl` substituent-suffix form (verified:
# `(N,N-dimethylmethanaminiumyl)acetate`, `(methanaminiumyl)acetate` are both
# UNPARSEABLE), but it DOES round-trip the equivalent `azane`-based PIN form
# `(trimethylazaniumyl)acetate` -> `C[N+](C)(C)CC(=O)[O-]` (and the multi-N-
# substituent variants). Both name the SAME cation (P-74.1.3 lists
# `(trimethylammoniumyl)`/azanium as the accepted equivalent). Accuracy is the
# #1 priority (the contributor guide) and the byte-identical/RT gate forbids shipping an
# OPSIN-unparseable string when a round-tripping equivalent exists, so the
# producer emits the azane-based `…azaniumyl` form (RT=1) — a strict
# improvement over the old `betaine`->`unknown organic compound` (RT=0).


def cation_to_prefix(mol, cation_idx: int, parent_attach_idx: int) -> str:
    """Build the cation-as-substituent prefix for a zwitterion (P-74.1.3).

    The cationic atom (``cation_idx``, e.g. a quaternary ammonium N) is the
    attachment atom of the substituent prefix; ``parent_attach_idx`` is its
    neighbour that lies on the path into the anionic parent (the bond that is
    "consumed" by the attachment). Every OTHER neighbour branch of the cation
    atom becomes an N-substituent prefix, named by the existing structured
    substituent machinery (``name_substituent_fragment``), then composed as:

        {alphabetized, multiplied N-substituent prefixes}azaniumyl

    ``azane`` (NH3) is the parent hydride of a nitrogen cation; ``azanium`` =
    NH4+ (P-73.1.1.1); ``azaniumyl`` = the N-attached cationic substituent
    (P-74.1.3, OPSIN-parseable equivalent of the `-aminiumyl` PIN — see module
    note above). Structured, NOT a hardcoded f-string.

    Args:
        mol: RDKit Mol of the whole zwitterion.
        cation_idx: Atom index of the (non-internal) cationic centre.
        parent_attach_idx: Neighbour atom index on the path to the anion.

    Returns:
        The cation prefix WITHOUT enclosing marks (e.g. ``trimethylazaniumyl``),
        or '' for an out-of-scope cation (ylide / non-N onium / amine-oxide /
        1,n-dipolar — P-74.2 deferred, D-06 honest-fail).
    """
    cat = mol.GetAtomWithIdx(cation_idx)

    # SCOPE (D-06): only a nitrogen cation maps to the azaniumyl family. Onium
    # cations on other elements (oxonium/sulfonium/phosphonium), ylides and
    # amine-oxides are deferred (return '' -> legacy fallthrough / honest-fail).
    if cat.GetSymbol() != 'N' or cat.GetFormalCharge() <= 0:
        return ''

    # Collect each N-substituent branch (every neighbour except the one leading
    # to the anionic parent). Each branch is named as a substituent prefix.
    sub_prefixes: List[str] = []
    for nb in cat.GetNeighbors():
        if nb.GetIdx() == parent_attach_idx:
            continue
        frag_atoms = _collect_branch_atoms(mol, start_idx=nb.GetIdx(),
                                           block_idx=cation_idx)
        name = name_substituent_fragment(mol, frag_atoms, nb.GetIdx(), [])
        if not name:
            return ''  # an unnameable N-substituent -> bail (honest-fail)
        sub_prefixes.append(name)

    if not sub_prefixes:
        # Bare protonated nitrogen with no extra substituents ([NH3+]-parent) is
        # the `azaniumyl` group itself (P-74.1.3): e.g. glycine zwitterion's
        # cation. The caller decides whether to use this or keep the amino-acid
        # neutral/retained form (sequenced first, D-06).
        return 'azaniumyl'

    return _compose_n_substituent_prefix(sub_prefixes) + 'azaniumyl'


def _collect_branch_atoms(mol, start_idx: int, block_idx: int) -> List[int]:
    """Collect the connected atom branch rooted at ``start_idx`` WITHOUT crossing
    back through ``block_idx`` (the cationic atom). Returns the branch atom
    indices (the substituent fragment hanging off the cation centre)."""
    visited = {block_idx}
    stack = [start_idx]
    frag: List[int] = []
    while stack:
        a = stack.pop()
        if a in visited:
            continue
        visited.add(a)
        frag.append(a)
        for nn in mol.GetAtomWithIdx(a).GetNeighbors():
            if nn.GetIdx() not in visited:
                stack.append(nn.GetIdx())
    return frag


# v23 Phase 8 (P-68.2.2): Group-14 substituent groups -XH3 named by method (2)
# of P-29.2 -> silyl / germyl, with the substituents on the X centre cited as
# prefixes (trimethylsilyl, trihydroxysilyl). Scoped to Si/Ge (the Phase-8 scope;
# Sn/Pb stannyl/plumbyl deferred).
_GROUP14_SUBSTITUENT_STEM = {'Si': 'silyl', 'Ge': 'germyl'}
# Terminal (X-bonded only) neutral heteroatom neighbour -> its substituent prefix.
_GROUP14_TERMINAL_HETERO_PREFIX = {
    ('O', 1): 'hydroxy',
    ('S', 1): 'sulfanyl',
    ('Se', 1): 'selanyl',
    ('Te', 1): 'tellanyl',
    ('N', 2): 'amino',
}
_GROUP14_HALO_PREFIX = {'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo', 'I': 'iodo'}


def _group14_neighbour_prefix(mol, idx, exclude_idx, frag_set) -> Optional[str]:
    """Prefix name for ONE neighbour atom of a Group-14 substituent centre, or
    None (fail-closed) for anything not a simple terminal hetero / halide / pure
    organyl."""
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetFormalCharge() != 0:
        return None
    sym = atom.GetSymbol()
    heavy_in_frag = [nb for nb in atom.GetNeighbors()
                     if nb.GetIdx() in frag_set and nb.GetSymbol() != 'H']
    # A TERMINAL hetero/halide neighbour (the Group-14 centre is its sole heavy
    # neighbour) is cited as hydroxy / sulfanyl / amino / fluoro / ... A
    # NON-terminal -O-R (silyl ether / silicic ester) is NOT this class -> None.
    if len(heavy_in_frag) == 1 and heavy_in_frag[0].GetIdx() == exclude_idx:
        if sym in _GROUP14_HALO_PREFIX:
            return _GROUP14_HALO_PREFIX[sym]
        hp = _GROUP14_TERMINAL_HETERO_PREFIX.get((sym, atom.GetTotalNumHs()))
        if hp is not None:
            return hp
    # Pure unbranched-alkyl / phenyl-naphthyl organyl (reuse the Phase-7 guard).
    from ..rules.substituent_purity import pure_organyl_prefix_name
    return pure_organyl_prefix_name(mol, idx, exclude_idx)


def _name_group14_substituent(mol, frag_atoms, attach_idx) -> Optional[str]:
    """Name a MONOVALENT Group-14 (Si/Ge) substituent as ``(prefixes)silyl`` /
    ``(prefixes)germyl`` (P-68.2.2 / P-29.2 method 2), else None (fail-closed).

    Fixes (the substituent cascade otherwise drops or mis-suffixes these):
      -SiH3      -> silyl            (HEAD: 'substituent', dropped on a senior C)
      -Si(OH)3   -> trihydroxysilyl  (HEAD: 'silanetriolyl', OH kept as a -ol suffix)
      -Si(CH3)3  -> trimethylsilyl   (already worked via Tier-4; kept byte-identical)

    Fail-closed for: a non-Si/Ge attach atom, a charged/radical/ring centre, a
    MULTIVALENT free valence (silanediyl bridge), or any neighbour that is not H /
    terminal -OH,-SH,-SeH,-TeH,-NH2,-halide / pure organyl (a silyl ether / silicic
    ester Si-O-C fails the terminal check -> None -> handled elsewhere).
    """
    frag_set = set(frag_atoms)
    a = mol.GetAtomWithIdx(attach_idx)
    stem = _GROUP14_SUBSTITUENT_STEM.get(a.GetSymbol())
    if stem is None:
        return None
    if (a.GetFormalCharge() != 0 or a.GetNumRadicalElectrons() != 0
            or a.IsInRing()):
        return None
    # Monovalent: exactly one bond leaves the fragment (to the parent). A
    # silanediyl / silanetriyl bridge (>=2 external bonds) is out of scope.
    external = [n for n in a.GetNeighbors() if n.GetIdx() not in frag_set]
    if len(external) != 1:
        return None
    prefixes: List[str] = []
    for n in a.GetNeighbors():
        if n.GetIdx() not in frag_set:
            continue  # the parent attachment
        p = _group14_neighbour_prefix(mol, n.GetIdx(), attach_idx, frag_set)
        if p is None:
            return None
        prefixes.append(p)
    if not prefixes:
        return stem  # bare -SiH3 -> 'silyl'
    return _compose_n_substituent_prefix(prefixes) + stem


def _compose_n_substituent_prefix(sub_prefixes: List[str]) -> str:
    """Compose multiple N-substituent prefixes into one alphabetized, multiplied
    string (P-14.5.2 alphanumerical order; P-16.3.3 multiplying prefixes).

    e.g. ['methyl','methyl','methyl'] -> 'trimethyl';
         ['ethyl','methyl','methyl']  -> 'ethyldimethyl'.
    """
    from collections import Counter
    from .naming_utils import get_multiplier_prefix, is_complex_substituent

    counts = Counter(sub_prefixes)
    # Alphabetize by the substituent name (ignoring the multiplier, per IUPAC).
    parts = []
    for name in sorted(counts.keys()):
        count = counts[name]
        mult = get_multiplier_prefix(count, name)
        if is_complex_substituent(name) and count > 1:
            parts.append((name, f"{mult}({name})"))
        else:
            parts.append((name, f"{mult}{name}"))
    return ''.join(p[1] for p in parts)
