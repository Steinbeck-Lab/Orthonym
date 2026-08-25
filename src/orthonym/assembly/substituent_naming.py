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
    IUPAC 2013 P-16.5.1.1 (compound substituent enclosing marks)
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem
from ..perception.stereo import assign_stereochemistry
from .naming_utils import get_alkyl_name, SIMPLE_MULTIPLIERS, alpha_sort_key, simple_multiplier_word
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
    best_chain = None

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
            best_chain = chain
            # Build the name for this direction
            best_name = _build_alkenyl_name(
                carbon_count, a_locant, double_locs, triple_locs
            )

    if best_name is None or best_chain is None:
        return None

    # v29 P3-FIX Item 1: the located stereodescriptor block. This producer is the
    # ONLY place that holds this substituent's own numbering, so it is the only
    # place that can put a locant on a descriptor -- and P-91.2.1.2.1 makes the
    # locant mandatory. `_add_substituent_stereo` (the generic emitter the caller
    # runs next) reads only ATOM _CIPCode, so without this the E/Z geometry was
    # dropped in silence and both isomers shipped one name.
    #
    # The fragment is unbranched and all-carbon by the guards above, so every
    # fragment atom lies on `best_chain`: the block is COMPLETE by construction
    # and needs no fail-closed branch here.
    return _located_stereo_block(
        mol, {a: i + 1 for i, a in enumerate(best_chain)}) + best_name


def _located_stereo_block(mol, locant_of: Dict[int, int]) -> str:
    """The ``(2R,3E)-`` stereodescriptor block for a substituent whose OWN
    numbering is ``locant_of`` (atom index -> substituent locant), or ``''``.

    Covers both kinds of stereogenic unit in one block, which is what the Blue
    Book does: under ``## **P-46.3** PRINCIPAL SUBSTITUENT CHAINS IN COMPOUNDS
    WITH STEREOGENIC CENTERS`` (``BlueBookV2.md:23014``) the preferred name is
    ``[(2Z,4R,5E)-4-methylhepta-2,5-dien-4-yl]siline (PIN)`` -- R/S and E/Z
    interleaved in ONE parenthesis and cited in LOCANT order (2, 4, 5), not
    grouped by kind. ``### P-91.2.1.2.1 Stereodescriptors used in substitutive
    nomenclature`` (``BlueBookV2.md:44624``) supplies the locant requirement:
    "*In preferred IUPAC names, stereodescriptors, preceded by a locant, must be
    cited to specify each stereogenic unit*".

    A double bond is located at the LOWER of its two atoms' locants, the same
    convention ``_build_alkenyl_name`` uses for the ``-en-`` locant itself, so
    the descriptor locant and the ene locant always agree -- ``(1E)-prop-1-en-1-yl``
    (``BlueBookV2.md:44686``, ``## **P-91.3** NAMING OF STEREOISOMERS``) and
    ``(2E)-but-2-en-1-yl`` (``:45437``) are the verbatim PIN tokens.

    Only DEFINED elements are cited: an undefined centre carries no ``_CIPCode``
    and a PIN may not invent a configuration the structure does not specify.
    Only atoms/bonds inside ``locant_of`` are considered, so a stereo element
    elsewhere in the molecule can never leak into a substituent's block.

    Pure: no mol mutation beyond the idempotent CIP assignment.
    """
    from ..perception.stereo import assign_stereochemistry, get_double_bond_stereo
    assign_stereochemistry(mol)

    terms: List[Tuple[int, str]] = []
    for idx, loc in locant_of.items():
        atom = mol.GetAtomWithIdx(idx)
        if atom.HasProp('_CIPCode'):
            terms.append((loc, atom.GetProp('_CIPCode')))
    for sb in get_double_bond_stereo(mol):
        a, b = sb['atoms']
        if a in locant_of and b in locant_of:
            terms.append((min(locant_of[a], locant_of[b]), sb['stereo']))
    if not terms:
        return ''
    terms.sort()
    return '(' + ','.join(f'{loc}{code}' for loc, code in terms) + ')-'


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
        mult = simple_multiplier_word(num_double) or ""
        segments.append((loc_str, mult, "en"))

    if triple_locants:
        loc_str = ",".join(str(l) for l in triple_locants)
        mult = simple_multiplier_word(num_triple) or ""
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


def _name_branched_alkenyl_substituent(mol, sub_atoms, attach_idx):
    """P-29.2 / P-31.1.4.3 / P-32.1.1: a BRANCHED acyclic ALL-CARBON substituent
    bearing >=1 C=C/C#C, named by its own principal chain THROUGH the free valence
    with the free-valence locant explicitly cited (P-32.1.1(1): free valences have
    priority for low locants and locant '1' MUST be cited for acyclic groups).

    The linear ``_name_unsaturated_chain`` declines branched fragments (its
    ``c_nbrs > 2`` guard), so before this handler they fell through to the free-
    molecule recursion + ``parent_to_prefix``, whose ``-e``->``-yl`` fallback DROPS
    the free-valence locant (``2-methylprop-1-enyl`` instead of ``2-methylprop-1-
    en-1-yl``) or emits a wrong-connectivity name (SELF-01 -> ``unknown`` for prenyl).

    Scope (fail-closed -> next handler / recursion otherwise): acyclic, all-carbon,
    uncharged, has unsaturation AND branching. Reuses the existing
    ``_build_alkenyl_name`` for the stem and ``name_substituent`` for off-chain
    branches. Returns the raw prefix name (no enclosing marks) or None.
    """
    if attach_idx is None or len(sub_atoms) < 3:
        return None
    sub_set = set(sub_atoms)
    if attach_idx not in sub_set:
        return None
    ring_info = mol.GetRingInfo()
    for idx in sub_set:
        a = mol.GetAtomWithIdx(idx)
        if a.GetSymbol() != 'C' or a.GetFormalCharge() != 0:
            return None
        if ring_info.NumAtomRings(idx) > 0:
            return None

    # Require BOTH unsaturation and branching (the linear namer owns the rest).
    has_unsat = False
    branched = False
    for idx in sub_set:
        c_nbrs = 0
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            ni = nbr.GetIdx()
            if ni in sub_set:
                c_nbrs += 1
                bond = mol.GetBondBetweenAtoms(idx, ni)
                if bond and bond.GetBondTypeAsDouble() != 1.0:
                    has_unsat = True
        if c_nbrs > 2:
            branched = True
    if not (has_unsat and branched):
        return None

    # Principal chain = longest simple path THROUGH the free valence, tie-broken
    # by MAX on-chain unsaturation (P-31.1.4.3.4). The fragment is an acyclic tree,
    # so the path between any two atoms is unique; enumerate atom pairs (bounded:
    # substituents are small) and keep the best chain that contains attach_idx.
    from collections import deque

    def _tree_path(a, b):
        prev = {a: None}
        dq = deque([a])
        while dq:
            x = dq.popleft()
            if x == b:
                break
            for nbr in mol.GetAtomWithIdx(x).GetNeighbors():
                ni = nbr.GetIdx()
                if ni in sub_set and ni not in prev:
                    prev[ni] = x
                    dq.append(ni)
        if b not in prev:
            return None
        path = []
        cur = b
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        return list(reversed(path))

    def _onchain_unsat(chain):
        n = 0
        for i in range(len(chain) - 1):
            bond = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
            if bond and bond.GetBondTypeAsDouble() != 1.0:
                n += 1
        return n

    atoms = sorted(sub_set)  # deterministic order for tie-breaking
    best = None  # ((length, unsat), chain)
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            p = _tree_path(atoms[i], atoms[j])
            if not p or attach_idx not in p:
                continue
            key = (len(p), _onchain_unsat(p))
            if best is None or key > best[0]:
                best = (key, p)
    if best is None:
        return None
    chain = best[1]

    def _dir_key(ch):
        k = ch.index(attach_idx) + 1
        unsat = []
        for i in range(len(ch) - 1):
            bond = mol.GetBondBetweenAtoms(ch[i], ch[i + 1])
            bt = bond.GetBondTypeAsDouble() if bond else 1.0
            if bt in (2.0, 3.0):
                unsat.append(i + 1)
        # P-32.1.1(1): free valence lowest first, then unsaturation lowest.
        return (k, sorted(unsat))

    fwd = chain
    rev = list(reversed(chain))
    chain = fwd if _dir_key(fwd) <= _dir_key(rev) else rev

    k = chain.index(attach_idx) + 1
    chain_set_c = set(chain)
    chain_pos = {a: i + 1 for i, a in enumerate(chain)}
    double_locs, triple_locs = [], []
    for i in range(len(chain) - 1):
        bond = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
        bt = bond.GetBondTypeAsDouble() if bond else 1.0
        if bt == 2.0:
            double_locs.append(i + 1)
        elif bt == 3.0:
            triple_locs.append(i + 1)

    # Off-chain branches -> the substituent's own substituents (P-46.1.12),
    # named recursively and located on this chain (mirror _located_acyclic_alkyl_name).
    from collections import defaultdict
    from .substituent_enumerator import name_substituent
    branch_groups: dict = defaultdict(list)
    for chain_atom in chain:
        for nbr in mol.GetAtomWithIdx(chain_atom).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in chain_set_c or nidx not in sub_set:
                continue
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
            try:
                bname = name_substituent(mol, frag, nidx)
            except Exception as exc:  # noqa: BLE001 — missing beats wrong
                logger.debug("branched-alkenyl branch naming failed: %s", exc)
                return None
            if not bname or bname == "substituent":
                return None
            branch_groups[bname].append(chain_pos[chain_atom])

    core = _build_alkenyl_name(len(chain), k, double_locs, triple_locs)
    if not branch_groups:
        body = core  # no off-chain branch after all -> plain alkenyl
    else:
        from .composer import _format_prefix_groups
        prefix = _format_prefix_groups(branch_groups)
        body = f"{prefix}{core}"

    # v29 P3-FIX Item 1: the located stereodescriptor block, from the SAME
    # `chain_pos` numbering the name was built with -- see `_located_stereo_block`
    # for the citations. Only CHAIN-borne elements are cited here; a stereocentre
    # inside an off-chain branch is expressed by that branch's own recursive
    # `name_substituent` call (nested block), so the two together cover the whole
    # fragment without either fabricating a locant it does not own.
    return _located_stereo_block(mol, chain_pos) + body


def _name_unsaturated_oxo_substituent(mol, sub_atoms, attach_idx, parent_set):
    """v26 BP-2 RC-3 (P-33 oxo / P-14.4): an ACYCLIC all-carbon substituent chain
    that is UNSATURATED (>=1 C=C/C#C) and carries >=1 in-chain/terminal carbonyl
    (aldehyde or ketone) -> the carbonyl is the detachable prefix 'oxo' on the
    chain numbered from the free valence, e.g. -C(=CH2)-CHO -> '3-oxoprop-1-en-2-yl'.

    The unsaturated analogue of the saturated oxo handling in
    _name_polyfunctional_acyclic_substituent (which declines unsaturation and
    internal attachment). Fail-closed (None) for: rings; any heteroatom other
    than a terminal ketone/aldehyde =O; a branched or non-single-chain backbone;
    no unsaturation (saturated tier owns it); no oxo (pure-alkenyl tier owns it);
    or an oxo ON the free-valence carbon (that is an acyl group -C(=O)-R, owned
    by the acyl tier — P-66 note (m) forbids the '1-oxo...yl' form).
    """
    if not sub_atoms or attach_idx is None or attach_idx not in set(sub_atoms):
        return None
    sub_set = set(sub_atoms)
    ri = mol.GetRingInfo()
    if any(ri.NumAtomRings(i) > 0 for i in sub_set):
        return None
    backbone = []
    oxo_hosts = {}                      # host C idx -> count of =O
    for idx in sub_set:
        a = mol.GetAtomWithIdx(idx)
        if a.GetFormalCharge() != 0:
            return None
        sym = a.GetSymbol()
        if sym == 'C':
            backbone.append(idx)
        elif sym == 'O':
            # only a terminal ketone/aldehyde =O (degree 1, 0 H, double to a C)
            nb = list(a.GetNeighbors())
            if len(nb) != 1 or a.GetTotalNumHs() != 0:
                return None
            b = mol.GetBondBetweenAtoms(idx, nb[0].GetIdx())
            if b.GetBondType() != Chem.BondType.DOUBLE or nb[0].GetSymbol() != 'C':
                return None
            if nb[0].GetIdx() not in sub_set:
                return None
            oxo_hosts[nb[0].GetIdx()] = oxo_hosts.get(nb[0].GetIdx(), 0) + 1
        else:
            return None                 # any other heteroatom -> fail closed
    bset = set(backbone)
    if not backbone or not oxo_hosts:
        return None
    if any(h not in bset for h in oxo_hosts):
        return None
    # linear single chain: every backbone C has <=2 backbone-C neighbours
    for idx in backbone:
        if sum(1 for n in mol.GetAtomWithIdx(idx).GetNeighbors()
               if n.GetIdx() in bset) > 2:
            return None
    # must be unsaturated within the backbone
    has_unsat = any(
        mol.GetBondBetweenAtoms(i, j.GetIdx()).GetBondTypeAsDouble() != 1.0
        for i in backbone for j in mol.GetAtomWithIdx(i).GetNeighbors()
        if j.GetIdx() in bset and j.GetIdx() > i)
    if not has_unsat:
        return None                     # saturated oxo -> polyfunctional tier owns it
    # an oxo on the free-valence carbon = acyl -> decline (P-66 note (m))
    if attach_idx in oxo_hosts:
        return None
    # trace the chain terminal -> terminal
    term = next((i for i in backbone
                 if sum(1 for n in mol.GetAtomWithIdx(i).GetNeighbors()
                        if n.GetIdx() in bset) == 1), None)
    if term is None:
        return None
    ordered, seen, cur = [term], {term}, term
    while len(ordered) < len(backbone):
        nxt = next((n.GetIdx() for n in mol.GetAtomWithIdx(cur).GetNeighbors()
                    if n.GetIdx() in bset and n.GetIdx() not in seen), None)
        if nxt is None:
            break
        ordered.append(nxt)
        seen.add(nxt)
        cur = nxt
    if len(ordered) != len(backbone):
        return None
    n = len(backbone)
    best = None                         # (key, name)
    for chain in (ordered, list(reversed(ordered))):
        fv = chain.index(attach_idx) + 1
        dbl, trp = [], []
        for i in range(n - 1):
            bt = mol.GetBondBetweenAtoms(
                chain[i], chain[i + 1]).GetBondTypeAsDouble()
            if bt == 2.0:
                dbl.append(i + 1)
            elif bt == 3.0:
                trp.append(i + 1)
        pos = {idx: i + 1 for i, idx in enumerate(chain)}
        oxo_locs = sorted(pos[h] for h in oxo_hosts for _ in range(oxo_hosts[h]))
        unsat = sorted(dbl + trp)
        # P-14.4: (c) free valence, (e) unsaturation, (f) detachable prefix oxo
        key = (fv, unsat, oxo_locs)
        if best is None or key < best[0]:
            stem = _build_alkenyl_name(n, fv, dbl, trp)   # 'prop-1-en-2-yl'
            _M = {1: "", 2: "di", 3: "tri", 4: "tetra"}
            ostr = f"{','.join(map(str, oxo_locs))}-{_M.get(len(oxo_locs), '')}oxo"
            best = (key, f"{ostr}{stem}")                 # '3-oxo' + 'prop-1-en-2-yl'
    return best[1] if best else None


_HALOGEN_PREFIX = {"F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}


# ============================================================================
# P-14.3.4.5 inside a SUBSTITUENT (enclosing-mark) scope
# ============================================================================
# ``P-14.3.4.5`` (``BlueBookV2/BlueBookV2.md:3007``), under ``P-14.3.4``
# "Omission of locants":
#
#     "All locants are omitted in compounds or substituent groups in which all
#      substitutable positions are completely substituted or modified, for
#      example, by hydro, in the same way. Except for hydrogen atoms attached to
#      chalcogen atoms, such as in acids, alcohols, and to the carbon atoms of
#      formyl groups (aldehydes), all hydrogen atoms are considered
#      substitutable."
#
# and its counter-clause ``:3009``: *"In case of partial substitution or
# modification, all numerical prefixes must be indicated. The prefix 'per-' is no
# longer recommended."*
#
# ★ It is the ONLY one of the six P-14.3.4 licences whose text says *"compounds or
# **substituent groups**"* -- the others speak only of parent structures. That word
# is what licenses ``1-chloro-2-(pentafluoroethyl)benzene (PIN)`` (``:3023``).
#
# The licence is carved out of the DENY-DEFAULT ``P-14.3.3`` "Citation of locants"
# (``:2869``), which is scoped *"as defined by its appropriate enclosing marks"*.
# That scoping sentence is the whole mechanism here: in
# ``1-chloro-2-(pentafluoroethyl)benzene`` the ethyl group inside the parentheses is
# completely and uniformly substituted and omits, while the benzene ring outside is
# only partially substituted and keeps ``1,2``. ONE molecule, TWO scopes, opposite
# answers -- so the licence must be applied at the SUBSTITUENT sites, where the
# enclosing-mark scope is known, and never inside the shared
# ``composer._format_prefix_groups`` (whose ``composer.py:216`` caller is
# parent-level and would strip ``benzenehexol``'s siblings).
#
# The class is **OPEN**: ``:3009`` retires the ``per-`` contraction, which *was*
# exactly a closed-list mechanism, and the 2013 recommendations replaced it with
# counting. So a table keyed on ``"pentafluoroethyl"`` is wrong on its complement by
# construction -- ``heptafluoropropyl``, ``pentachloroethyl``,
# ``heptafluoropropan-2-yl`` are all entailed and none is printed in the Blue Book
# (``:3023`` is the ONLY printed instance of any of them; verified by grep).
# Structural predicate only.
_L5_CHAIN_HYDRIDE_CACHE: Dict[Tuple[int, int], object] = {}


def _l5_chain_parent_hydride(chain_len: int, k: int):
    """Parent hydride of an acyclic saturated all-carbon substituent chain.

    ``chain_len`` carbons, free valence at 1-based locant ``k``, the free valence
    represented as a **dummy atom**. Atom index ``i`` <-> chain locant ``i + 1``;
    the dummy is the last atom and carries no hydrogen, so it is never a
    substitutable position.

    P-14.3.4.5 speaks of the substitutable positions of the PARENT, so the licence
    has to be measured against the UNDECORATED chain: on the input molecule a fully
    substituted carbon has zero hydrogens and the count that decides the licence
    would be lost. Same reason, and the same idiom, as
    ``rules/benzene.py::_benzene_parent_hydride()`` -- which is the site behind the
    ``benzenehexol`` (omits) vs ``cyclohexane-1,2,3,4,5,6-hexol`` (retains) pair.

    The dummy consumes exactly one hydrogen (**P-29.2**, ``:15813``: *"The atom with
    the free valence terminates a chain and always has the locant '1', which is
    omitted from the name"*), and the attachment carbon's REMAINING hydrogens still
    count. That arithmetic is what makes the Blue Book print ``penta``fluoroethyl:
    ethyl has 5 substitutable H, 2 at C1 and 3 at C2. Measured here as
    ``*CC -> {1: 2, 2: 3}``.
    """
    key = (chain_len, k)
    cached = _L5_CHAIN_HYDRIDE_CACHE.get(key)
    if cached is not None:
        return cached
    rw = Chem.RWMol()
    for _ in range(chain_len):
        rw.AddAtom(Chem.Atom(6))
    for i in range(chain_len - 1):
        rw.AddBond(i, i + 1, Chem.BondType.SINGLE)
    dummy = rw.AddAtom(Chem.Atom(0))
    rw.AddBond(k - 1, dummy, Chem.BondType.SINGLE)
    hydride = rw.GetMol()
    try:
        Chem.SanitizeMol(hydride)
    except Exception:  # noqa: BLE001  -- deny-by-default
        return None
    _L5_CHAIN_HYDRIDE_CACHE[key] = hydride
    return hydride


def _l5_substituent_prefix(mol, sub_atoms, chain, k, groups) -> Optional[str]:
    """``P-14.3.4.5`` (``:3007``) for ONE substituent enclosing-mark scope.

    Returns the **locant-free** prefix string (``'pentafluoro'``) when the licence
    positively fires, or ``None`` meaning *"keep the locants"*.

    ⚠ **DENY BY DEFAULT.** This is not a locant stripper: every omission is a
    positively-licensed structural predicate, and anything that cannot be
    established returns ``None``. The **completeness/uniformity test** itself is NOT
    re-derived here -- it is delegated to
    ``assembly.locant_omission.l5_uniform_complete``, the one place the Blue Book
    licences live.

    ⚠ It does **more than marshal**, and the docstring said otherwise until a review
    caught it (2026-07-29). Beyond building the parent hydride and the decoration
    map, this function independently decides two further rule questions:

    * the **two ambient P-14.3.3 scopes** (``locants_are_forced`` and the weaker
      ``scope_has_isotopic_modification``) -- either one vetoes;
    * the **k >= 2 internal-free-valence boundary**: for a free valence anywhere but
      position 1 the valence locant is itself essential, so P-14.3.3's *"then all
      locants must be cited ... for that structural unit"* restores the substitution
      locants and the licence declines. This is a deny-by-default reading, not a
      printed Blue Book example (the BB prints no fully substituted substituent group
      with an internal free valence), and it is recorded as ASSUMED, not VERIFIED;
    * plus ``scope_forces_locants`` for the ordinary essential-locant cases.

    Args:
        mol: the whole molecule the substituent lives in.
        sub_atoms: every atom of the substituent scope (chain + decorations).
        chain: ordered principal-chain atom indices; ``chain[i]`` is locant
            ``i + 1``.
        k: the free-valence locant, 1-based.
        groups: ``{prefix name: [locant, ...]}`` with ONE ENTRY PER OCCURRENCE,
            exactly as both call sites already build it (``pentafluoroethyl`` is
            ``{'fluoro': [1, 1, 2, 2, 2]}``).
    """
    from .locant_omission import (
        l5_uniform_complete, locants_are_forced, scope_forces_locants,
        scope_has_isotopic_modification,
    )
    from .naming_utils import format_substituent_prefix

    # P-14.3.3 (``:2869``) as an AMBIENT scope. The isotope path names an
    # isotope-STRIPPED molecule -- measured: at both live sites every
    # ``GetIsotope()`` in the scope reads 0 even for a 13C input -- so the
    # structural ``has_isotope`` below is blind by construction and THIS is the
    # live guard. A licence that skips it elides a locant P-82.6.1.1 (``:44180``)
    # requires, and neither SELF-01 (``namer.py`` states verbatim that it *"ignores
    # isotopes"*) nor the gold set can see it: gold exposure for this whole class
    # is zero.
    if locants_are_forced():
        return None

    # ⚠ AND the weaker isotopic declaration, because ``locants_are_forced()`` ALONE
    # IS NOT ENOUGH -- measured 2026-07-29. The isotope decorator enters the forced
    # scope only after establishing that the descriptor needs a locant, so for
    # ``FC(F)(F)[13C](F)(F)C1CCCCC1`` it is False here while the finished name still
    # carries ``(13C1)``. Consulting only the forced flag turned
    # ``(1,1,2,2,2-pentafluoro(13C1)ethyl)cyclohexane`` into
    # ``(pentafluoro(13C1)ethyl)cyclohexane``, emptying of ALL locants a scope whose
    # two carbons are inequivalent -- exactly what P-82.6.1.1 (``:44180``) forbids.
    # This licence empties a scope completely, so it must decline on the weaker flag
    # too. (``rules/benzene.py`` must NOT: its locant-free form is correct under
    # P-82.6.1.3, ``(13C1)benzenehexol``.)
    if scope_has_isotopic_modification():
        return None

    if mol is None or not chain or not groups or not sub_atoms:
        return None
    chain_len = len(chain)
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= chain_len:
        return None

    # ★ TERMINAL FREE VALENCE ONLY (k == 1). This is a rule boundary, not a
    # convenience narrowing, and it is the conservative side of a genuine ambiguity
    # the source does not settle by a printed example.
    #
    # P-29.2 (``:15813``) *"the atom with the free valence terminates a chain and
    # always has the locant '1', which is omitted from the name"* -- so for k == 1
    # the enclosing-mark scope contains NO locant of its own, and P-14.3.4.5 can
    # empty it. That is precisely the shape of the only printed positive,
    # ``(pentafluoroethyl)`` (``:3023``).
    #
    # For k >= 2 the scope must cite the free-valence locant itself
    # (``propan-2-yl``), and that locant IS essential -- it distinguishes
    # propan-2-yl from propan-1-yl. P-14.3.3 (``:2869``) then applies verbatim:
    # *"if any locants are essential for defining the structure of ... a unit of
    # structure as defined by its appropriate enclosing marks, then all locants must
    # be cited for ... that structural unit."* So the substitution locants come
    # back, and ``1,1,1,2,3,3,3-heptafluoropropan-2-yl`` KEEPS them.
    #
    # Consistency check: the Blue Book's flagship negative ``:46359``
    # ``(1,1,1,3,3,3-hexafluoropropan-2-yl)oxy`` is k == 2 and keeps -- reachable
    # both by this clause and by its C2 retaining a hydrogen, so the two readings
    # agree there. The Blue Book prints NO fully-substituted substituent group with
    # an internal free valence, so nothing decides between them; deny-by-default
    # picks retention.
    if k != 1:
        return None

    sub_set = set(sub_atoms)
    if not set(chain) <= sub_set or len(set(chain)) != chain_len:
        return None

    # Independently re-establish the preconditions the parent hydride encodes, so a
    # third call site cannot be wired up against a scope this shape does not fit:
    # acyclic, all-carbon, saturated, and connected in the given order.
    ring_info = mol.GetRingInfo()
    for idx in chain:
        try:
            atom = mol.GetAtomWithIdx(int(idx))
        except (OverflowError, RuntimeError, ValueError, TypeError):
            return None
        if atom.GetSymbol() != 'C' or atom.GetFormalCharge() != 0:
            return None
        if ring_info.NumAtomRings(int(idx)) > 0:
            return None
    for a, b in zip(chain, chain[1:]):
        bond = mol.GetBondBetweenAtoms(int(a), int(b))
        if bond is None or bond.GetBondTypeAsDouble() != 1.0:
            return None
    # Any unsaturation ANYWHERE in the scope changes the hydrogen count the licence
    # is measured against, so it is a precondition of the whole scope, not just the
    # chain.
    for bond in mol.GetBonds():
        if (bond.GetBeginAtomIdx() in sub_set and bond.GetEndAtomIdx() in sub_set
                and bond.GetBondTypeAsDouble() != 1.0):
            return None

    # Marshal: locant -> the ONE decoration kind there, plus its multiplicity.
    # ``counts`` is mandatory: ethyl's C1 has TWO substitutable hydrogens, so
    # without it a doubly-fluorinated C1 reads as partial and the licence is
    # (correctly, but uselessly) denied. Every decoration is included, not just
    # halogens -- otherwise a scope whose uniformity is broken by a NON-halogen
    # substituent could not be detected, which is the ``:29619`` failure mode.
    kind_at: Dict[int, set] = {}
    count_at: Dict[int, int] = {}
    total = 0
    for kind, locants in groups.items():
        if kind is None or not str(kind).strip():
            return None
        for loc in (locants or []):
            if isinstance(loc, bool) or not isinstance(loc, int):
                return None
            if not 1 <= loc <= chain_len:
                return None
            kind_at.setdefault(loc - 1, set()).add(kind)
            count_at[loc - 1] = count_at.get(loc - 1, 0) + 1
            total += 1
    if not kind_at or any(len(kinds) != 1 for kinds in kind_at.values()):
        return None                      # two kinds at one position -> not "in the
                                         # same way" -> :3009
    decoration_of = {idx: next(iter(kinds)) for idx, kinds in kind_at.items()}

    # P-14.3.3: anything ESSENTIAL in the same scope restores every locant.
    # Uniform complete substitution structurally precludes a stereocentre on the
    # chain (a carbon bearing two identical decorations is not stereogenic), but
    # that is checked rather than assumed -- a complex uniform decoration could
    # carry one of its own.
    has_stereo = any(
        mol.GetAtomWithIdx(int(i)).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for i in sub_atoms
    )
    if scope_forces_locants(
        prefix_locants=[loc for locs in groups.values() for loc in (locs or [])],
        # A substituent prefix scope carries no suffix of its own.
        suffix_locants=[],
        stereo_text='stereo' if has_stereo else '',
        # Indicated hydrogen is a ring/tautomer device; the scope is verified
        # acyclic above. Multiplicative names and ring assemblies name their scope
        # elsewhere and never arrive here as one acyclic chain. Skeletal
        # replacement needs a heteroatom IN the chain; verified all-carbon above.
        has_indicated_h=False,
        has_isotope=any(mol.GetAtomWithIdx(int(i)).GetIsotope() for i in sub_atoms),
        is_multiplicative=False,
        is_ring_assembly=False,
        has_skeletal_replacement=False,
    ):
        return None

    hydride = _l5_chain_parent_hydride(chain_len, k)
    if hydride is None:
        return None
    if not l5_uniform_complete(hydride, decoration_of=decoration_of,
                               counts=count_at):
        return None

    # Render through the ONE canonical substituent-prefix formatter, which already
    # documents the empty-locant case as the P-14.3.4 elision path
    # (``naming_utils.py``: *"An empty locant list (elided per P-14.3.4, e.g. a
    # mononuclear parent: phenylmethanol) takes no hyphen"*). Passing ``[]``
    # therefore reuses its multiplier, bis/tris and
    # enclosing-mark logic instead of adding a tenth inline copy of the join.
    only_kind = next(iter(set(decoration_of.values())))
    rendered = format_substituent_prefix(only_kind, [], total)
    return rendered or None


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

    v29 P3-FIX Item 1: ...and anything UNSATURATED. The docstring said "saturated"
    from the start but nothing enforced it, so a fragment carrying a C=C was named
    by the alkane stem and the double bond vanished: ``-CH2-CH=CH-Cl`` came back
    ``3-chloropropyl`` (an alkane) instead of ``3-chloroprop-2-en-1-yl``, and
    ``-CH=CH-Cl`` came back ``2-chloroethyl``. Those name a DIFFERENT COMPOUND, not
    merely a stereo-underspecified one, so this is a constitution guard rather than
    a stereo one. Surfaced by the stereo-expression obligation in
    ``rules.substituent_purity.organyl_prefix_name``: a dropped double bond drops
    its geometry too, so the count identity there failed and led here. The
    unsaturated cases belong to ``_name_unsaturated_chain`` /
    ``_name_branched_alkenyl_substituent``, which is where declining sends them.
    """
    if not sub_atoms:
        return None
    sub_set = set(sub_atoms)
    ring_info = mol.GetRingInfo()

    # SATURATED is in this function's name and contract: decline otherwise.
    for _b in mol.GetBonds():
        if (_b.GetBeginAtomIdx() in sub_set and _b.GetEndAtomIdx() in sub_set
                and _b.GetBondTypeAsDouble() != 1.0):
            return None

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

    stem = get_chain_prefix(len(backbone))

    # P-14.3.4.5 (``:3007``), substituent enclosing-mark scope -- MEASURED-LIVE site
    # for ``(pentafluoroethyl)cyclohexane`` (a validated call-spy recorded this
    # function as the sole productive namer of that fragment, reached from
    # ``name_substituent_fragment`` Step 2c; ``_located_acyclic_alkyl_name`` is
    # never productive for it). Deny-by-default: the helper returns None unless the
    # licence positively fires, and the locant-multiplier join below is then
    # BYPASSED, never post-processed. ``ordered`` is attach-first, so the free
    # valence is locant 1.
    _l5 = _l5_substituent_prefix(mol, sub_atoms, ordered, 1, groups)
    if _l5:
        return f"{_l5}{stem}yl"

    _MULT = {1: "", 2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa"}
    # Alphabetical order of the (base) halogen prefixes (P-14.5.1: the di/tri
    # multiplier on a simple substituent is ignored for ordering).
    part_strings = []
    for prefix in sorted(groups.keys()):
        locs = sorted(groups[prefix])
        loc_str = ",".join(str(loc) for loc in locs)
        mult = _MULT.get(len(locs), f"{len(locs)}")
        part_strings.append(f"{loc_str}-{mult}{prefix}")

    return f"{'-'.join(part_strings)}{stem}yl"


# Single-atom detachable prefixes this namer enumerates from STRUCTURE, keyed
# by (element, role). Carboxy is multi-atom (handled separately: its carbon is
# NOT a backbone carbon — P-65.1.1 expresses -COOH on a substituent as the
# detachable prefix "carboxy", excluding the acid carbon from the parent chain).
_POLYFUNC_OXO_PREFIX = "oxo"
_POLYFUNC_HYDROXY_PREFIX = "hydroxy"
_POLYFUNC_AMINO_PREFIX = "amino"
_POLYFUNC_CARBOXY_PREFIX = "carboxy"
# v33 Phase 3 (glucosinolate/thiohydroximate O-sulfate anion): the S-analogue of
# hydroxy (P-63.1.5) for a terminal -SH branch on a backbone carbon
# ('2-sulfanylethyl', the direct parallel to '2-hydroxyethyl'). Documented as
# declined at the point below until this fix ("other tiers own them" was
# unverified -- SPY showed no tier names a chain+thiol branch at all, not even
# the 2-atom case).
_POLYFUNC_SULFANYL_PREFIX = "sulfanyl"
# v33 Phase 6 Wave 2 (#5b, P-66.5.1): the nitro group, charge-separated in the
# graph (-N+(=O)[O-]) but net-neutral as a substituent -- consumed by the
# dedicated Pass 1c2 below BEFORE Pass 1d's generic cation loop and Pass 2's
# blanket charge decline can see it, so it is numbered from the free valence
# exactly like hydroxy/amino/halogen ('2-nitroethyl', not the bare 'nitroethyl'
# or the wrong-end '1-nitropropyl').
_POLYFUNC_NITRO_PREFIX = "nitro"


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


def _longest_carbon_chain(mol, backbone_set, attach):
    """Longest simple carbon path in `backbone_set` starting at `attach` (a
    terminus). Deterministic: ties broken by the lowest CanonicalRankAtoms
    sequence along the path (P-29.2 numbering starts at the free valence)."""
    from rdkit import Chem
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    best: list = []
    best_key = None

    def dfs(cur, path, seen):
        nonlocal best, best_key
        extended = False
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            ni = nb.GetIdx()
            if ni in backbone_set and ni not in seen:
                extended = True
                dfs(ni, path + [ni], seen | {ni})
        if not extended:
            key = (-len(path), tuple(ranks[i] for i in path))
            if best_key is None or key < best_key:
                best_key = key
                best = list(path)

    dfs(attach, [attach], {attach})
    return best


def _collect_branch_subtree(mol, seed, chain_set):
    """All atoms reachable from `seed` without crossing into `chain_set`
    (branch carbons + their attached FG atoms)."""
    seen = {seed}
    stack = [seed]
    while stack:
        c = stack.pop()
        for nb in mol.GetAtomWithIdx(c).GetNeighbors():
            ni = nb.GetIdx()
            if ni in chain_set or ni in seen:
                continue
            seen.add(ni)
            stack.append(ni)
    return seen


def _branch_carbons_unbranched(mol, subtree, seed):
    """True iff the branch's carbon skeleton is a linear chain rooted at `seed`
    (so name_substituent yields a clean prefix and never reaches the enumerator's
    malformed compound-substituent path). v1 fail-closed otherwise."""
    cset = {i for i in subtree if mol.GetAtomWithIdx(i).GetSymbol() == 'C'}
    for i in cset:
        deg = sum(1 for nb in mol.GetAtomWithIdx(i).GetNeighbors()
                  if nb.GetIdx() in cset)
        limit = 1 if i == seed else 2
        if deg > limit:
            return False
    return True


def _name_branched_polyfunctional_substituent(
    mol, backbone_set, attach_idx, prefix_on,
):
    """W2F-P3 (P-59.2.1.8 / P-29.2 / P-16.3.3): name a BRANCHED acyclic saturated
    carbon substituent bearing detachable FG prefixes, numbered from the free
    valence (attach = locant 1). Principal chain = the longest carbon path from
    the (primary-terminus) attachment; FG prefixes on chain carbons are located;
    off-chain carbons + their FG atoms form simple UNBRANCHED sub-branches named
    via name_substituent (hydroxymethyl, methyl, ...). Every atom is already
    validated as a recognised FG / carbon by the caller's Pass-1/Pass-2, so this
    only partitions + assembles. Fail-closed (None) outside v1: secondary
    attachment, a sub-branch whose carbon skeleton is itself branched, an off-chain
    FG host not absorbed by a branch, or a repeated complex branch prefix."""
    from collections import defaultdict
    from ..data.chain_names import get_chain_prefix
    from .substituent_enumerator import name_substituent
    from .naming_utils import needs_brackets

    if sum(1 for n in mol.GetAtomWithIdx(attach_idx).GetNeighbors()
           if n.GetIdx() in backbone_set) > 1:
        return None  # secondary / internal attachment -> follow-on
    chain = _longest_carbon_chain(mol, backbone_set, attach_idx)
    if not chain or chain[0] != attach_idx:
        return None
    chain_set = set(chain)
    pos = {idx: i + 1 for i, idx in enumerate(chain)}  # attach = 1

    groups: dict = defaultdict(list)  # located-prefix -> [locant,...]
    # (1) FG prefixes whose host carbon is ON the principal chain.
    for c_idx, prefixes in prefix_on.items():
        if c_idx in chain_set:
            for pfx in prefixes:
                groups[pfx].append(pos[c_idx])

    # (2) off-chain backbone carbons = branch subtrees rooted at a chain carbon.
    offchain = set(backbone_set) - chain_set
    covered: set = set()
    for seed in sorted(offchain):
        if seed in covered:
            continue
        root_cs = [n.GetIdx() for n in mol.GetAtomWithIdx(seed).GetNeighbors()
                   if n.GetIdx() in chain_set]
        if len(root_cs) != 1:
            continue  # interior branch atom -> collected via its own root
        subtree = _collect_branch_subtree(mol, seed, chain_set)
        if not _branch_carbons_unbranched(mol, subtree, seed):
            return None  # branched sub-branch -> v1 fail-closed
        covered |= subtree
        bname = name_substituent(mol, subtree, seed)
        if not bname:
            return None
        if needs_brackets(bname):
            bname = f"({bname})"
        if bname.startswith("(") and groups.get(bname):
            return None  # repeated complex branch (bis/tris) -> follow-on
        groups[bname].append(pos[root_cs[0]])

    if offchain - covered:
        return None  # an off-chain atom is not part of any branch subtree
    for c_idx in prefix_on:
        if c_idx not in chain_set and c_idx not in covered:
            return None  # FG host neither on-chain nor inside a named branch

    _MULT = {1: "", 2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa"}
    parts = []
    for prefix in sorted(groups.keys(), key=alpha_sort_key):
        locs = sorted(groups[prefix])
        mult = _MULT.get(len(locs), SIMPLE_MULTIPLIERS.get(len(locs), ""))
        loc_str = ",".join(str(loc) for loc in locs)
        parts.append(f"{loc_str}-{mult}{prefix}")
    stem = get_chain_prefix(len(chain))
    return f"{''.join(_joined_prefix_parts(parts))}{stem}yl"


# ----------------------------------------------------------------------------
# Per-molecule memoization for _name_polyfunctional_acyclic_substituent.
#
# ROOT-CAUSE PERFORMANCE FIX (2026-08-20): naming a long peptide/depsipeptide
# chain calls this function ~20,000+ times for as few as ~30 DISTINCT
# (frozenset(sub_atoms), attach_idx) keys -- the recursive substituent namer
# re-derives the same overlapping sub-fragment's name over and over (a single
# key was recomputed 11,552 times on one 209-heavy-atom peptide), the classic
# exponential-recursion-over-overlapping-subproblems shape. The function has
# no side effects and its result depends ONLY on (mol, sub_atoms, attach_idx)
# -- `parent_set` is accepted but never read in the body below, confirmed by
# inspection -- so memoizing is a pure speedup with byte-identical output.
#
# Scope: PER MOLECULE OBJECT, not global. `sub_atoms`/`attach_idx` are atom
# INDICES that only mean anything relative to one `mol`; a cache keyed on
# indices alone would return a name computed for a different molecule's atoms
# once a second molecule is named in the same process. `mol` is threaded
# UNCHANGED as the single full-molecule object through the whole recursive
# substituent-naming tree (verified: every call site -- the Step 2c-poly
# dispatch in name_substituent_fragment and the Tier-1.96 call in
# substituent_enumerator.py -- passes the same top-level `mol` with a SUBSET
# of its atom indices, never a fragment-extracted new Mol), so identity
# (`is`) on `mol` is the correct scope key. The held module-level reference to
# the current mol additionally prevents a same-`id()` false match after the
# previous mol is garbage-collected and a new object happens to reuse the
# freed address.
#
# `None` is a legitimate (and, per the profiler, extremely common) result and
# is cached like any other value via a sentinel, so a failing subproblem does
# not keep re-failing at full cost either.
# ----------------------------------------------------------------------------
_POLYFUNC_MEMO_MOL = None
_POLYFUNC_MEMO_CACHE: dict = {}
_POLYFUNC_MEMO_MISS = object()


def _name_polyfunctional_acyclic_substituent(
    mol,
    sub_atoms: List[int],
    attach_idx: int,
    parent_set: Set[int],
) -> Optional[str]:
    """Memoizing wrapper around :func:`_name_polyfunctional_acyclic_substituent_impl`.

    See the module comment above the cache globals for the scoping and
    correctness argument. Delegates all naming logic unchanged to the `_impl`
    function; this wrapper only adds a per-mol memo keyed on
    ``(frozenset(sub_atoms), attach_idx)``.
    """
    global _POLYFUNC_MEMO_MOL, _POLYFUNC_MEMO_CACHE
    if _POLYFUNC_MEMO_MOL is not mol:
        _POLYFUNC_MEMO_CACHE = {}
        _POLYFUNC_MEMO_MOL = mol
    key = (frozenset(sub_atoms), attach_idx)
    cached = _POLYFUNC_MEMO_CACHE.get(key, _POLYFUNC_MEMO_MISS)
    if cached is not _POLYFUNC_MEMO_MISS:
        return cached
    result = _name_polyfunctional_acyclic_substituent_impl(
        mol, sub_atoms, attach_idx, parent_set
    )
    _POLYFUNC_MEMO_CACHE[key] = result
    return result


def _name_polyfunctional_acyclic_substituent_impl(
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

    # ---- Pass 1b (v30 RISK 5 Class 2, P-16.5.1.3.1 / P-62.2.1.1): a SUBSTITUTED
    # amine -N(R)(R') on a backbone carbon -> a composed '(dialkylamino)' PREFIX,
    # consuming the N and its alkyl branches so they never enter the backbone.
    # Delegates every ordering/marking decision to the ONE amino-prefix assembler
    # (`_assemble_amino_prefix_core`), the same one the PIN path uses -- so the
    # general-engine substituent path stops fabricating the OPSIN-unparseable
    # `amino-N,N-diethylethyl` string-surgery token. Primary -NH2 (no branch) is
    # left for the Pass-2 'amino' branch below. Fail-closed (skip this N, leaving
    # the fragment to decline as before) for anything outside the v1 envelope:
    # charged / ring / imine-or-amide N, >1 carbon host, or a branch the composer
    # cannot name (ring/hetero/complex). ----
    _amine_import_ok = True
    try:
        from .substituent_enumerator import (
            _assemble_amino_prefix_core, composed_prefix_organyl_name,
            carbon_free_valence_prefix)
    except Exception:
        _amine_import_ok = False
    _ring_info_pf = mol.GetRingInfo()
    if _amine_import_ok:
        for idx in list(sub_set):
            if idx in consumed:
                continue
            a = mol.GetAtomWithIdx(idx)
            if a.GetSymbol() != 'N' or a.GetFormalCharge() != 0:
                continue
            if _ring_info_pf.NumAtomRings(idx) > 0:
                continue  # ring N -> not this handler
            if any(b.GetBondType() != Chem.BondType.SINGLE for b in a.GetBonds()):
                continue  # imine / amide-C=N / nitrile N -> other tiers
            in_frag = [n.GetIdx() for n in a.GetNeighbors()
                       if n.GetIdx() in sub_set and n.GetIdx() not in consumed]

            def _component(seed):
                # atoms reachable from `seed` within sub_set WITHOUT crossing this N
                comp: Set[int] = set()
                st = [seed]
                while st:
                    x = st.pop()
                    if x in comp or x == idx or x not in sub_set:
                        continue
                    comp.add(x)
                    for n in mol.GetAtomWithIdx(x).GetNeighbors():
                        nj = n.GetIdx()
                        if nj != idx and nj not in comp:
                            st.append(nj)
                return comp

            # The HOST is the N-neighbour whose side holds the free valence; every
            # other N-neighbour is a BRANCH of the amino prefix.
            host = None
            host_multi = False
            branch_seeds = []
            for nb in in_frag:
                comp = _component(nb)
                if attach_idx in comp:
                    if host is not None:
                        host_multi = True
                    host = nb
                else:
                    branch_seeds.append((nb, comp))
            if host is None or host_multi:
                continue  # N in-chain/aza (not a pendant amine) -> decline
            if mol.GetAtomWithIdx(host).GetSymbol() != 'C':
                continue  # amino prefix must sit on a carbon
            host_atom = mol.GetAtomWithIdx(host)
            if any(bb.GetBondType() == Chem.BondType.DOUBLE
                   and bb.GetOtherAtom(host_atom).GetSymbol() == 'O'
                   for bb in host_atom.GetBonds()):
                continue  # host is a carbonyl carbon -> amide, not amine
            if not branch_seeds:
                continue  # primary -NH2 -> Pass-2 'amino' branch handles it
            entries = []
            branch_atoms: Set[int] = set()
            ok = True
            for b0, visited in branch_seeds:
                if any(mol.GetAtomWithIdx(v).GetSymbol() != 'C'
                       or _ring_info_pf.NumAtomRings(v) > 0
                       or mol.GetAtomWithIdx(v).GetFormalCharge() != 0
                       for v in visited):
                    ok = False
                    break
                _fv = carbon_free_valence_prefix(mol, visited, b0)
                nm = _fv.prefix
                if nm is None:
                    if _fv.must_fail_closed:
                        ok = False
                        break
                    nm = composed_prefix_organyl_name(mol, visited, b0)
                if nm is None:
                    ok = False
                    break
                _complex = bool(re.search(r"[()\-]", nm)) or nm[:1].isdigit()
                entries.append((nm, _complex))
                branch_atoms |= visited
            if not ok or not entries:
                continue
            core = _assemble_amino_prefix_core(entries)
            if not core:
                continue
            consumed.add(idx)
            consumed.update(branch_atoms)
            _add_prefix(host, f"({core})")

    # ---- Pass 1e (v33 Phase 6 E2b, P-66.1.1.4.3): a secondary/tertiary amide
    # -C(=O)-N(H)(R)- IN THE CHAIN -- the carbonyl carbon is a plain BACKBONE
    # atom, not a terminal -C(=O)OH/-C(=O)NH2 Pass-1/Pass-2 already own -- is
    # expressed as an 'oxo' prefix (the =O; Pass 2 below adds it unchanged,
    # this pass never touches it) PLUS an '(R-amino)'/'[(R)amino]' compound
    # prefix at the SAME locant for the amide -N(H)(R)- branch:
    # '3-oxo-3-[(2-sulfanylethyl)amino]propyl' for -CH2CH2C(=O)NHCH2CH2SH
    # (pantetheine's N-substituent arm). Without this pass the amide N reaches
    # Pass 2's plain-N branch below, which requires a bare -NH2 (2 H) and hard
    # DECLINES the WHOLE fragment for any -NH-R/-NR2 amide nitrogen (DROP-09).
    #
    # Consumes the amide N + its OWN substituent branch(es) (named via the
    # SAME organyl-prefix builders Pass 1b already uses, and assembled through
    # the ONE amino-prefix assembler `_assemble_decorated_amino_prefix` so the
    # enclosing-mark escalation -- '(2-sulfanylethyl)amino' nested inside
    # '[(2-sulfanylethyl)amino]' -- is the shared P-16.5.2.4 logic, not a
    # second hand-rolled copy). Fail-closed (skip this N) for: a ring amide N,
    # a bare primary amide (-C(=O)NH2, 0 branches -- the existing carbamoyl
    # tiers own that), a branch this project's organyl namer cannot express,
    # a carbonyl carbon that is not a PLAIN acyl centre (an imide/urea/
    # carbamate second N or O on the same carbon -- a different functional
    # class other tiers own), or a branch that loops back into the backbone
    # (bridging, not a pendant R).
    if _amine_import_ok:
        for idx in list(sub_set):
            if idx in consumed:
                continue
            a = mol.GetAtomWithIdx(idx)
            if a.GetSymbol() != 'N' or a.GetFormalCharge() != 0 or a.GetIsotope():
                continue
            if _ring_info_pf.NumAtomRings(idx) > 0:
                continue
            if any(b.GetBondType() != Chem.BondType.SINGLE for b in a.GetBonds()):
                continue
            in_frag = [n.GetIdx() for n in a.GetNeighbors()
                       if n.GetIdx() in sub_set and n.GetIdx() not in consumed]
            host = None
            branch_seeds = []
            for nb in in_frag:
                nb_atom = mol.GetAtomWithIdx(nb)
                is_carbonyl = False
                if nb_atom.GetSymbol() == 'C' and nb_atom.GetFormalCharge() == 0:
                    _dbl_o = [bb.GetOtherAtom(nb_atom) for bb in nb_atom.GetBonds()
                              if bb.GetBondType() == Chem.BondType.DOUBLE
                              and bb.GetOtherAtom(nb_atom).GetSymbol() == 'O']
                    _extra_n = sum(
                        1 for bb in nb_atom.GetBonds()
                        if bb.GetOtherAtom(nb_atom).GetSymbol() == 'N'
                        and bb.GetOtherAtom(nb_atom).GetIdx() != idx)
                    _extra_o_single = sum(
                        1 for bb in nb_atom.GetBonds()
                        if bb.GetBondType() == Chem.BondType.SINGLE
                        and bb.GetOtherAtom(nb_atom).GetSymbol() == 'O')
                    if (len(_dbl_o) == 1 and _dbl_o[0].GetDegree() == 1
                            and _dbl_o[0].GetFormalCharge() == 0
                            and _extra_n == 0 and _extra_o_single == 0):
                        is_carbonyl = True
                        _oxo_idx = _dbl_o[0].GetIdx()
                if is_carbonyl:
                    if host is not None:
                        host = None  # >1 carbonyl neighbour -> not a simple amide N
                        break
                    host, oxo_idx = nb, _oxo_idx
                else:
                    branch_seeds.append(nb)
            if host is None or host in consumed:
                continue
            if not branch_seeds:
                continue  # bare primary amide -- the carbamoyl tiers own it

            def _n_branch_component(seed, _block=idx):
                comp: Set[int] = set()
                st = [seed]
                while st:
                    x = st.pop()
                    if x in comp or x == _block or x not in sub_set:
                        continue
                    comp.add(x)
                    for n in mol.GetAtomWithIdx(x).GetNeighbors():
                        nj = n.GetIdx()
                        if nj != _block and nj not in comp:
                            st.append(nj)
                return comp

            entries = []
            branch_atoms: Set[int] = set()
            ok = True
            for b0 in branch_seeds:
                comp = _n_branch_component(b0)
                if host in comp or attach_idx in comp:
                    ok = False  # bridges back into the backbone -> not pendant
                    break
                if any(mol.GetAtomWithIdx(v).GetFormalCharge() != 0 for v in comp):
                    ok = False
                    break
                _fv = carbon_free_valence_prefix(mol, comp, b0)
                nm = _fv.prefix
                if nm is None:
                    if _fv.must_fail_closed:
                        ok = False
                        break
                    nm = composed_prefix_organyl_name(mol, comp, b0)
                if nm is None:
                    ok = False
                    break
                _complex = bool(re.search(r"[()\-]", nm)) or nm[:1].isdigit()
                entries.append((nm, _complex))
                branch_atoms |= comp
            if not ok or not entries:
                continue
            from .composer import _assemble_decorated_amino_prefix
            amino_pfx = _assemble_decorated_amino_prefix(entries, enclose=True)
            if not amino_pfx:
                continue
            consumed.add(idx)
            consumed.update(branch_atoms)
            _add_prefix(host, amino_pfx)

    # ---- Pass 1f (v33 Phase 6 E2c, P-65.3.1 / P-66.1.1.4.2, WRONG-MOLECULE
    # RISK): a sulfonamide -S(=O)(=O)-NH2/-NHR/-NR2 hanging off a backbone
    # carbon is the retained 'sulfamoyl' prefix, or its N-substituted
    # '(R-sulfamoyl)'/'(dialkylsulfamoyl)' form -- '(methylsulfamoyl)methyl'
    # for -CH2-SO2-NHCH3. Consumed HERE so it can never reach the generic
    # recursive Tier-4 path + `parent_to_prefix`'s '-amide' string transform,
    # which reads 'methanesulfonamide' as if it ended in the CARBOXAMIDE
    # suffix and COLLAPSES THE SULFONYL TO 'carbamoyl' -- a WRONG
    # CONSTITUTION (measured: '-CH2-SO2-NHCH3' -> the wrong 'carbamoylmethyl',
    # a different molecule SELF-01 must catch). Reuses
    # `rules.benzene._build_n_substituted_sulfamoyl_prefix` (the ring-side
    # '{N-subs}sulfamoyl' builder) rather than re-deriving it, for the
    # CHAIN-side attachment. Fail-closed (skip this S) for: a ring-borne
    # sulfonyl, a charged/hypervalent/isotope-labelled S, a ring amide N,
    # DISTINCT N,N-disubstituents (the shared builder's own conservative
    # envelope -- nested 'ethyl(methyl)sulfamoyl' is not built), or any shape
    # outside the plain sulfonamide envelope (a bridging N, a branch this
    # project's organyl namer cannot express).
    if _amine_import_ok:
        for idx in list(sub_set):
            if idx in consumed:
                continue
            a = mol.GetAtomWithIdx(idx)
            if a.GetSymbol() != 'S' or a.GetFormalCharge() != 0 or a.GetIsotope():
                continue
            if _ring_info_pf.NumAtomRings(idx) > 0:
                continue
            nbrs = list(a.GetNeighbors())
            if len(nbrs) != 4 or any(n.GetIdx() not in sub_set for n in nbrs):
                continue
            dbl_o = []
            host_c = None
            n_idx = None
            ok = True
            for n in nbrs:
                ni = n.GetIdx()
                b = mol.GetBondBetweenAtoms(idx, ni)
                if (n.GetSymbol() == 'O' and b.GetBondType() == Chem.BondType.DOUBLE
                        and n.GetFormalCharge() == 0 and n.GetDegree() == 1):
                    dbl_o.append(ni)
                elif (n.GetSymbol() == 'C' and b.GetBondType() == Chem.BondType.SINGLE
                        and host_c is None):
                    host_c = ni
                elif (n.GetSymbol() == 'N' and b.GetBondType() == Chem.BondType.SINGLE
                        and n_idx is None):
                    n_idx = ni
                else:
                    ok = False
                    break
            if not ok or len(dbl_o) != 2 or host_c is None or n_idx is None:
                continue
            if host_c in consumed:
                continue
            n_atom = mol.GetAtomWithIdx(n_idx)
            if (_ring_info_pf.NumAtomRings(n_idx) > 0
                    or n_atom.GetFormalCharge() != 0 or n_atom.GetIsotope()):
                continue
            if any(b.GetBondType() != Chem.BondType.SINGLE for b in n_atom.GetBonds()):
                continue
            n_branches = [nb.GetIdx() for nb in n_atom.GetNeighbors()
                          if nb.GetIdx() != idx]
            if any(b not in sub_set for b in n_branches):
                continue

            def _s_branch_component(seed, _block=None):
                _blocked = _block or {idx, n_idx}
                comp: Set[int] = set()
                st = [seed]
                while st:
                    x = st.pop()
                    if x in comp or x in _blocked or x not in sub_set:
                        continue
                    comp.add(x)
                    for n in mol.GetAtomWithIdx(x).GetNeighbors():
                        nj = n.GetIdx()
                        if nj not in _blocked and nj not in comp:
                            st.append(nj)
                return comp

            names = []
            branch_atoms: Set[int] = set()
            ok2 = True
            for b0 in n_branches:
                comp = _s_branch_component(b0)
                if host_c in comp or attach_idx in comp:
                    ok2 = False
                    break
                if any(mol.GetAtomWithIdx(v).GetFormalCharge() != 0 for v in comp):
                    ok2 = False
                    break
                _fv = carbon_free_valence_prefix(mol, comp, b0)
                nm = _fv.prefix
                if nm is None:
                    if _fv.must_fail_closed:
                        ok2 = False
                        break
                    nm = composed_prefix_organyl_name(mol, comp, b0)
                if nm is None:
                    ok2 = False
                    break
                names.append(nm)
                branch_atoms |= comp
            if not ok2:
                continue
            if not names:
                sulfamoyl_pfx = 'sulfamoyl'
            else:
                from ..rules.benzene import _build_n_substituted_sulfamoyl_prefix
                sulfamoyl_pfx = _build_n_substituted_sulfamoyl_prefix(names)
            if not sulfamoyl_pfx:
                continue
            consumed.add(idx)
            consumed.update(dbl_o)
            consumed.add(n_idx)
            consumed.update(branch_atoms)
            _add_prefix(host_c, sulfamoyl_pfx)

    # ---- Pass 1c (v30 tail #7/#21, P-65.6.3.2.3 / P-16.3.3): an ester whose
    # OXYGEN sits on a backbone carbon (-C-O-C(=O)-R) is the ACYLOXY detachable
    # prefix '(Racyloxy)' -- '(acetyloxy)methyl' for -CH2-O-C(=O)CH3,
    # '3-(2-methylprop-2-enoyloxy)propyl' for the #21 arm. Tier 1.95 already owns
    # the case where the ester O IS the free valence (a bare '-O-C(=O)R'
    # substituent); this pass is its INTERIOR sibling -- the ester O bridges two
    # in-fragment carbons, so both neighbours live inside sub_set. The whole acyl
    # side (carbonyl C + its =O + the R chain) is consumed; the acyl is named by
    # the shared acid engine (`_acyloxy_for_site` -> full name_compound on the
    # isolated acid, so branched/unsaturated acyls like methacryloyl name
    # correctly), never a carbon count. Carbamate / carbonate / xanthate
    # (-O-C(=O)-N / -O-C(=O)-O / -O-C(=S)-) are EXCLUDED here (the carbonyl C
    # must carry exactly one terminal =O and at most one all-carbon R): those are
    # owned by the Tier-0.5 carbamoyloxy path -- feeding them to the acyl namer
    # would mis-spell them. Fail-closed on any non-plain-ester shape. ----
    _acyloxy_import_ok = True
    try:
        from ..rules.lipids import _acyloxy_for_site
        from .naming_utils import enclose_if_compound as _enclose_if_compound
    except Exception:
        _acyloxy_import_ok = False
    if _acyloxy_import_ok:
        _acyloxy_tokens: dict = {}   # token -> count (compound-multiplicity guard)
        for idx in list(sub_set):
            if idx in consumed:
                continue
            a = mol.GetAtomWithIdx(idx)
            if (a.GetSymbol() != 'O' or a.GetFormalCharge() != 0
                    or a.GetTotalNumHs() != 0
                    or ring_info.NumAtomRings(idx) > 0):
                continue
            nbrs = [n.GetIdx() for n in a.GetNeighbors()]
            if len(nbrs) != 2 or any(n not in sub_set for n in nbrs):
                continue   # not an INTERIOR ester O (both sides in fragment)
            if any(mol.GetBondBetweenAtoms(idx, n).GetBondType()
                   != Chem.BondType.SINGLE for n in nbrs):
                continue
            # one neighbour is the carbonyl carbon C(=O), the other the host.
            carbonyl_c = host_c = None
            for n in nbrs:
                na = mol.GetAtomWithIdx(n)
                if na.GetSymbol() != 'C':
                    carbonyl_c = host_c = None
                    break
                if any(b.GetBondType() == Chem.BondType.DOUBLE
                       and b.GetOtherAtom(na).GetSymbol() == 'O'
                       for b in na.GetBonds()):
                    carbonyl_c = n
                else:
                    host_c = n
            if carbonyl_c is None or host_c is None or host_c in consumed:
                continue
            # The carbonyl carbon must be a PLAIN acyl: exactly one terminal =O,
            # at most one all-carbon R, nothing else (excludes carbamate/carbonate
            # /thiono). Its whole R side must live inside the fragment.
            cc = mol.GetAtomWithIdx(carbonyl_c)
            oxo_n = []
            r_side = []
            bad = False
            for b in cc.GetBonds():
                o = b.GetOtherAtom(cc)
                oi = o.GetIdx()
                if oi == idx:
                    continue
                if (b.GetBondType() == Chem.BondType.DOUBLE
                        and o.GetSymbol() == 'O' and o.GetDegree() == 1):
                    oxo_n.append(oi)
                elif (b.GetBondType() == Chem.BondType.SINGLE
                      and o.GetSymbol() == 'C'):
                    r_side.append(oi)
                elif o.GetSymbol() == 'H':
                    continue
                else:
                    bad = True
                    break
            if bad or len(oxo_n) != 1 or len(r_side) > 1:
                continue
            # Collect the entire acyl side (from the carbonyl C, not crossing the
            # ester O); it must be self-contained and must not re-enter the host.
            acyl_side: Optional[Set[int]] = set()
            st = [carbonyl_c]
            while st:
                x = st.pop()
                if x in acyl_side or x == idx:
                    continue
                if x not in sub_set:
                    acyl_side = None
                    break
                acyl_side.add(x)
                for n in mol.GetAtomWithIdx(x).GetNeighbors():
                    ni = n.GetIdx()
                    if ni != idx and ni not in acyl_side:
                        st.append(ni)
            if acyl_side is None or host_c in acyl_side:
                continue
            acyloxy = _acyloxy_for_site(mol, ("acyl", carbonyl_c, idx))
            if not acyloxy or ' ' in acyloxy:
                continue
            token = _enclose_if_compound(acyloxy)
            consumed.add(idx)
            consumed.update(acyl_side)
            _add_prefix(host_c, token)
            _acyloxy_tokens[token] = _acyloxy_tokens.get(token, 0) + 1
        # A COMPOUND acyloxy prefix that repeats needs bis/tris, which the simple
        # assembly multiplier below cannot spell -- fail closed rather than emit
        # the wrong 'di(acetyloxy)'. (Single occurrence per token is the tail-row
        # shape; a multiplicative acyloxy is a follow-on.)
        if any(cnt >= 2 for cnt in _acyloxy_tokens.values()):
            return None

    # ---- Pass 1c2 (v33 Phase 6 Wave 2, #5b, P-61.5.1 / P-29.3.2): a pendant
    # NITRO group on a backbone carbon (-CH2-N+(=O)[O-], the charge-separated
    # graph form of -NO2) is the detachable 'nitro' prefix, consumed HERE --
    # before Pass 1d's generic cation loop just below, which ALSO matches on
    # this N (its formal charge is +1, so it satisfies Pass 1d's
    # ``GetFormalCharge() <= 0: continue`` filter). Pass 1d hands any such atom
    # to ``cation_to_prefix``, an onium-family builder (ammonium/oxonium/
    # phosphonium, P-74.1.3) that structurally declines a nitro N (no host
    # carries the onium's expected all-single-bond substituent pattern) and
    # returns None -- a SILENT decline: Pass 1d does not consume the N or its
    # two oxygens on a miss, so nothing here or in Pass 1d places it, and the
    # fragment reaches Pass 2's blanket ``GetFormalCharge() != 0: return None``
    # check, which declines the WHOLE fragment (nitro's charge-separated N+/O-
    # never nets to zero ATOM BY ATOM, only in sum over the pair). That decline
    # is why nitro -- alone among hydroxy/amino/carboxy/halogen -- never reached
    # the free-valence-numbered builder below and fell back to the OLDER
    # recursive whole-molecule namer instead, whose numbering anchors the
    # SUBSTITUENT at ITS OWN lowest locant rather than the chain's free valence:
    # '-CH2-CH2-CH2-NO2' capped to the free molecule 'nitropropane' numbers the
    # nitro at C1 (both chain ends are termini once the attachment is capped
    # with H), then the alkyl/alkoxy conversion re-roots the free valence at
    # locant 1 WITHOUT renumbering the nitro -- '1-nitropropyl' instead of
    # '3-nitropropyl', the SAME wrong-end bug the hydroxy/carboxy passes above
    # exist to fix (P-29.3.2: the free valence takes locant 1).
    #
    # Structural match only, so this is a no-op for every other N+ shape
    # (ammonium/oxonium cations fall through unchanged to Pass 1d): the N must
    # carry formal charge +1, exactly 3 neighbours all inside the fragment --
    # ONE single-bonded backbone carbon (the host, not yet consumed), ONE
    # neutral terminal =O, and ONE terminal -O- of formal charge -1 -- and
    # nothing else. Any deviation (extra substituent, ring N, wrong bond order/
    # charge on either oxygen) is left alone for Pass 1d / Pass 2 to decline
    # exactly as before this pass existed.
    for _nidx in list(sub_set):
        if _nidx in consumed:
            continue
        _na = mol.GetAtomWithIdx(_nidx)
        if _na.GetSymbol() != 'N' or _na.GetFormalCharge() != 1:
            continue
        _nbrs = list(_na.GetNeighbors())
        if len(_nbrs) != 3:
            continue
        _host_c = _dbl_o = _neg_o = None
        _ok = True
        for _nb in _nbrs:
            _ni = _nb.GetIdx()
            if _ni not in sub_set:
                _ok = False
                break
            _bond = mol.GetBondBetweenAtoms(_nidx, _ni)
            if (_nb.GetSymbol() == 'C' and _bond.GetBondType() == Chem.BondType.SINGLE
                    and _host_c is None):
                _host_c = _ni
            elif (_nb.GetSymbol() == 'O' and _bond.GetBondType() == Chem.BondType.DOUBLE
                    and _nb.GetFormalCharge() == 0 and _nb.GetDegree() == 1
                    and _dbl_o is None):
                _dbl_o = _ni
            elif (_nb.GetSymbol() == 'O' and _bond.GetBondType() == Chem.BondType.SINGLE
                    and _nb.GetFormalCharge() == -1 and _nb.GetDegree() == 1
                    and _neg_o is None):
                _neg_o = _ni
            else:
                _ok = False
                break
        if not (_ok and _host_c is not None and _dbl_o is not None
                and _neg_o is not None) or _host_c in consumed:
            continue
        consumed.add(_nidx)
        consumed.add(_dbl_o)
        consumed.add(_neg_o)
        _add_prefix(_host_c, _POLYFUNC_NITRO_PREFIX)

    # ---- Pass 1d (v33 Phase 3 enabler, P-74.1.3 / P-73): a pendant ONIUM --
    # cation branch off a backbone carbon (choline's -CH2-CH2-N+(CH3)3) is a
    # locanted detachable '(...)azaniumyl'-family prefix, consumed here just
    # like the substituted-amino (Pass 1b) / acyloxy (Pass 1c) branches above.
    #
    # This is NOT the DIRECT-ATTACHMENT shape (the cation itself IS
    # ``attach_idx``) -- that is a separate, already-shipped mechanism
    # (`rules/charged_router.py`/`rules/ions.py` calling `cation_to_prefix`
    # directly) and stays untouched: the component search below BLOCKS on the
    # cation atom, so when the cation IS ``attach_idx`` no neighbour's
    # component can ever contain it, ``host`` stays None, and this pass is a
    # clean no-op (falls through to Pass-2's existing charge decline exactly
    # as before this pass existed).
    #
    # ``cation_to_prefix`` (P-74.1.3, the existing structured primitive
    # defined later in this same module) already builds the whole
    # '{N-substituents}azaniumyl'-family token from the cation's OWN
    # neighbours; this pass only has to (a) find the ONE neighbour branch
    # that leads back to ``attach_idx`` (the "host" backbone carbon the
    # prefix locants onto) and (b) consume the cation + every OTHER branch
    # (the cation's own substituents, which ``cation_to_prefix`` names and
    # owns internally) so Pass 2 below never sees the charge.
    #
    # Fail-closed (skip this cation -- Pass 2's pre-existing
    # ``GetFormalCharge() != 0`` guard then declines the whole fragment,
    # exactly as it did before this pass existed) for: a ring-borne onium
    # (pyridinium etc. -- the ring engine's job), a cation bridging TWO
    # backbone-side paths (in-chain, not a pendant branch), a host that is
    # not carbon, or any shape ``cation_to_prefix`` itself declines (ylide /
    # non-onium / P-74.2 1,n-dipolar). A NEUTRAL fragment never enters this
    # loop at all (no atom has formal charge > 0) -- a byte-identical no-op
    # for every existing hydroxy/amino/halogen/etc class.
    _ring_info_cat = mol.GetRingInfo()
    for idx in list(sub_set):
        if idx in consumed:
            continue
        a = mol.GetAtomWithIdx(idx)
        if a.GetFormalCharge() <= 0:
            continue
        if _ring_info_cat.NumAtomRings(idx) > 0:
            continue  # defensive no-op: upstream (line 1563) already excludes ring-bearing fragments

        def _cation_component(seed, _block=idx):
            comp: Set[int] = set()
            st = [seed]
            while st:
                x = st.pop()
                if x in comp or x == _block or x not in sub_set:
                    continue
                comp.add(x)
                for n in mol.GetAtomWithIdx(x).GetNeighbors():
                    nj = n.GetIdx()
                    if nj != _block and nj not in comp:
                        st.append(nj)
            return comp

        in_frag = [n.GetIdx() for n in a.GetNeighbors()
                   if n.GetIdx() in sub_set and n.GetIdx() not in consumed]
        host = None
        host_multi = False
        branch_atoms_cat: Set[int] = set()
        for nb in in_frag:
            comp = _cation_component(nb)
            if attach_idx in comp:
                if host is not None:
                    host_multi = True
                host = nb
            else:
                branch_atoms_cat |= comp
        if host is None or host_multi:
            continue  # not a pendant branch (bridging / walled off) -> decline
        if mol.GetAtomWithIdx(host).GetSymbol() != 'C' or host in consumed:
            continue
        if mol.GetBondBetweenAtoms(idx, host).GetBondType() != Chem.BondType.SINGLE:
            continue
        _cp = cation_to_prefix(mol, idx, host)
        if not _cp:
            continue  # cation_to_prefix declined -> fail closed, fragment falls through
        consumed.add(idx)
        consumed.update(branch_atoms_cat)
        _add_prefix(host, f"({_cp})")

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
    # single-atom prefix on a backbone carbon; backbone carbons bond only to
    # backbone carbons / recognised prefix atoms. A backbone C=C is RECORDED (v30
    # B2: unsaturated substituent chains, the aconitic/lignin-monomer family) and
    # named below; any other multiplicity (C#C / C=N / branch-C=C) still declines.
    core_double_bonds: Set[frozenset] = set()
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
                if (bt == Chem.BondType.DOUBLE and nb.GetSymbol() == 'C'
                        and ni in backbone_set):
                    core_double_bonds.add(frozenset((idx, ni)))  # backbone C=C -> named below
                    continue
                return None              # C#C / C#N / C=N / branch-C=C etc. -> decline
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
        elif (sym == 'S' and len(in_frag_nbrs) == 1 and len(c_host) == 1
                and mol.GetBondBetweenAtoms(idx, c_host[0]).GetBondType()
                    == Chem.BondType.SINGLE
                and a.GetTotalNumHs() >= 1):
            # v33 Phase 3 (glucosinolate/thiohydroximate O-sulfate anion,
            # P-63.1.5): a terminal -SH (thiol) on a backbone carbon is the
            # 'sulfanyl' detachable prefix -- the direct S-analogue of the
            # 'hydroxy' branch above ('2-sulfanylethyl' parallels
            # '2-hydroxyethyl'). MEASURED: no other tier names this shape
            # (the comment on the chalcogenylidene branch below claiming
            # "other tiers own them" was unverified for the single-bond
            # case -- SPY found the fragment cache/recursive/descriptive
            # tiers all decline a chain+thiol branch, even the 2-atom
            # '2-sulfanylethyl'). A charged / isotope-labelled / bridging S
            # never reaches this branch (guarded above); anything else
            # (0 H -- a thioether bridge or -S- ester owner) falls through
            # unchanged to the chalcogenylidene branch next, preserving its
            # existing decline.
            _add_prefix(c_host[0], _POLYFUNC_SULFANYL_PREFIX)
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
        elif sym == 'P':
            # W2F-P7 (P-68.3): a phosphanyl group as a detachable prefix on a
            # backbone carbon (-CH2-PH2 -> 'phosphanylmethyl'). This roots the
            # substituent at the free-valence CARBON (correct), whereas the Tier-4
            # recursive namer names the capped fragment as a FREE molecule
            # (CH3-PH2 -> 'methylphosphane' -> the wrong 'methylphosphyl'). Only
            # the STANDARD-valence terminal hydride P is named here (simple prefix,
            # no enclosing marks); a λ5-hydride / organyl P returns None so the
            # whole fragment falls through to the recursive tier, which roots it
            # correctly as '(lambda5-phosphanyl)methyl'. The name_phosphanyl_
            # substituent guard EXCLUDES a phosphoryl/phosphonic P=O.
            if len(in_frag_nbrs) != 1 or len(c_host) != 1:
                return None
            bond = mol.GetBondBetweenAtoms(idx, c_host[0])
            if bond.GetBondType() != Chem.BondType.SINGLE:
                return None
            from ..rules.phosphorus import name_phosphanyl_substituent
            _phn = name_phosphanyl_substituent(mol, [idx], idx)
            if _phn != 'phosphanyl':      # λ5 / dialkyl / P=O -> recursive tier
                return None
            _add_prefix(c_host[0], _phn)
            fg_count += 1
        else:
            return None                  # B / etc. -> decline

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
        # v33 Phase 3: oxo + sulfanyl on ONE carbon is the same thioic-S-acid
        # shape (-C(=O)-SH, P-65.1.1.4) as the hydroxy/oxo carboxyl-carbon
        # case above -- never emit '1-oxo-1-sulfanyl...'; defer to the
        # dedicated S-acid prefix ('sulfanylcarbonyl', PREFIX_FORMS) tier.
        if _POLYFUNC_OXO_PREFIX in _prefixes and _POLYFUNC_SULFANYL_PREFIX in _prefixes:
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
        # v30 tail: a bare acyl -C(=O)-R (the sole FG is the oxo on the
        # free-valence carbon) is the acyl PREFIX -- 'acetyl'/'propanoyl'/... --
        # per P-66.6 (BB:17762 'acetyl (preferred prefix)', :17944 "acyl groups
        # such as 'acetyl', for -CO-CH3, derived from acetic acid"). Emit it
        # directly for a LINEAR SATURATED backbone: the old fail-close left the
        # 2-carbon case to cap to acetaldehyde and garble as the OPSIN-unparseable
        # '1-methyl-2-oxaeth-1-en-1-yl' (SELF-01-vetoed -> abstain). Branched /
        # unsaturated acyls (2-methylpropanoyl, prop-2-enoyl) still fail closed
        # here -> handled by later tiers / a follow-on. attach must BE the
        # carbonyl carbon (the oxo sits on it), so this never fires on the
        # -CH2-C(=O)- '2-oxoethyl' orientation.
        _bb_linear = all(
            sum(1 for _n in mol.GetAtomWithIdx(_i).GetNeighbors()
                if _n.GetIdx() in backbone_set) <= 2
            for _i in backbone)
        if (_bb_linear and not core_double_bonds
                and attach_idx in backbone_set and len(backbone) >= 1):
            from ..data.chain_names import get_chain_prefix as _gcp_acyl
            _nac = len(backbone)
            _acyl_pfx = {1: 'formyl', 2: 'acetyl'}.get(_nac)
            if _acyl_pfx is None:
                _acyl_pfx = f"{_gcp_acyl(_nac)}anoyl"
            if _acyl_pfx and ' ' not in _acyl_pfx:
                return _acyl_pfx
        return None
    # Same rule for a thioacyl attachment (-C(=S)-R = alkanethioyl, Wave-2 C).
    if fg_count == 1 and any(
            y in prefix_on.get(attach_idx, []) for y in _YLIDENES):
        return None

    # ---- Backbone: linear (fast path below) or branched (W2F-P3, P-59.2.1.8). ----
    is_branched = any(
        sum(1 for n in mol.GetAtomWithIdx(idx).GetNeighbors()
            if n.GetIdx() in backbone_set) > 2
        for idx in backbone
    )
    if core_double_bonds and is_branched:
        return None  # v30 B2 v1: an unsaturated substituent chain is linear-only
    if is_branched:
        # Branched acyclic FG-bearing substituent: select a principal chain from
        # the free valence and name off-chain sub-branches (the "located
        # polyfunctional" follow-on this guard used to defer). Fail-closed (None)
        # outside the v1 envelope; every atom is already Pass-2-validated.
        return _name_branched_polyfunctional_substituent(
            mol, backbone_set, attach_idx, prefix_on,
        )
    n_attach_bb = sum(1 for n in mol.GetAtomWithIdx(attach_idx).GetNeighbors()
                      if n.GetIdx() in backbone_set)
    if n_attach_bb > 1:
        return None  # internal/secondary attachment on a linear backbone -> follow-on

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

    # v30 tail #22 (P-16.5.1.3.1 / P-16.3.3): a MONONUCLEAR substituent core
    # (single backbone carbon, locants elided) bearing >=2 simple detachable
    # prefixes must cite the FIRST bare and enclose EACH of the rest. The bare
    # concatenation is ambiguous -- '-CH(NH2)(COOH)' spelled 'aminocarboxymethyl'
    # RT-parses as amino + carboxymethyl (a different molecule), while
    # 'amino(carboxy)methyl' round-trips. BB witnesses: '[amino(imino)methyl]',
    # '[hydroxy(imino)methyl]' (:34268, :33425). Scoped to single_position: a
    # multi-carbon backbone carries locants that already disambiguate
    # ('2-amino-2-carboxyethyl', unchanged). The multiplied-simple carve-out
    # (bromodichlorofluoromethyl) is preserved inside apply_mononuclear_enclosing.
    if single_position and len(parts) >= 2:
        from .composition_primitives import apply_mononuclear_enclosing
        parts = apply_mononuclear_enclosing(parts, is_mononuclear=True)

    stem = get_chain_prefix(len(backbone))
    # Join consecutive prefix parts; a leading digit after a non-digit needs a
    # hyphen ("...amino-2-carboxy..."), handled by the locant prefix itself.
    joined = ''.join(_joined_prefix_parts(parts))
    if not core_double_bonds:
        return f"{joined}{stem}yl"
    # v30 B2: unsaturated substituent chain (aconitic/lignin-monomer family).
    # Add the 'ene' infix + free-valence yl locant + (nE/nZ) descriptor, numbered
    # from the free valence (locant 1, P-29.2 -- NOT the references' longest-chain
    # OPSIN-optimisation, which mis-roots the free valence: refconsult-b2 Q4.2).
    return _unsaturated_substituent_name(
        mol, ordered, pos, sub_set, core_double_bonds, joined, stem,
    )


def _unsaturated_substituent_name(
    mol, ordered, pos, sub_set, core_double_bonds, joined, stem,
):
    """Finish an UNSATURATED acyclic-chain substituent prefix (v30 B2).

    Called only from :func:`_name_polyfunctional_acyclic_substituent` once the
    backbone, the detachable prefixes and the chain trace are already validated and
    the backbone carries >=1 C=C. Numbering is from the free valence (``ordered[0]``
    is locant 1, P-29.2), so the ene locant and the yl (attachment) locant are read
    from ``pos``. Emits ``{prefixes}{stem}-{ene}-en-{yl}-yl`` with a leading
    ``(nE)/(nZ)`` descriptor when the double bond geometry is defined
    (``3-hydroxyprop-1-en-1-yl`` / ``(1E)-3-hydroxyprop-1-en-1-yl``).

    Handles a mono-ene (``prop-1-en-1-yl``) OR a polyene chain (``penta-1,3-dien-1-yl``,
    the euphonic-'a' multiplied form, mirroring the aconitic branch in
    ``added_carbon_parent``). Each stereogenic C=C contributes one ``(nE)/(nZ)`` to a
    single locant-ordered leading descriptor block (``(1E,3E)``).

    Fail closed (return None -> the cascade falls through unchanged):
      * >4 double bonds (multiplier table caps at tetraene);
      * a defined R/S stereocentre in the fragment ONLY WHEN we also emit an E/Z
        descriptor -- the caller's ``_stereo_route`` double-apply guard drops the
        R/S once this name leads with a ``(...)`` block, so a merged R/S+E/Z case
        can't be expressed and must fail closed. When no C=C geometry is defined
        (no descriptor emitted) the R/S is left for ``_stereo_route`` to add, so it
        is ALLOWED (fable-b2 RISK 3: e.g. ``(S)-3-hydroxybut-1-en-1-yl``);
      * a stereo bond that is not a backbone C=C (off-chain geometry);
      * a stereogenic C=C whose CIP code is neither E nor Z.
    Constitution is correct by construction (the prefixes were enumerated from the
    real bonds, like the saturated sibling), and E/Z follows deterministically from
    the bond CIP code, so no separate re-anchor is needed; the exact spelling is
    covered by unit tests + the substituent gold-diff.
    """
    from ..perception.stereo import assign_stereochemistry

    assign_stereochemistry(mol)
    core_bond_ids = set()
    ene_locs = []          # lowest locant of each backbone C=C
    for pair in core_double_bonds:
        a_idx, b_idx = tuple(pair)
        bond = mol.GetBondBetweenAtoms(a_idx, b_idx)
        core_bond_ids.add(bond.GetIdx())
        ene_locs.append(min(pos[a_idx], pos[b_idx]))
    ene_locs.sort()

    # No stereo bond other than a backbone C=C (off-chain geometry can't be
    # mapped to a chain locant).
    for bnd in mol.GetBonds():
        if (bnd.GetStereo() != Chem.BondStereo.STEREONONE
                and bnd.GetIdx() not in core_bond_ids
                and bnd.GetBeginAtomIdx() in sub_set
                and bnd.GetEndAtomIdx() in sub_set):
            return None

    yl_loc = pos[ordered[0]]  # free valence = locant 1 (P-29.2)

    # One (locant, E/Z) per stereogenic backbone C=C -> merged leading block.
    ez_bits = []
    for pair in core_double_bonds:
        a_idx, b_idx = tuple(pair)
        bond = mol.GetBondBetweenAtoms(a_idx, b_idx)
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        if not bond.HasProp('_CIPCode'):
            return None
        ez = bond.GetProp('_CIPCode')
        if ez not in ('E', 'Z'):
            return None
        ez_bits.append((min(pos[a_idx], pos[b_idx]), ez))
    descriptor = ""
    if ez_bits:
        ez_bits.sort()
        descriptor = "(" + ",".join(f"{lc}{ez}" for lc, ez in ez_bits) + ")-"

    # Chain R/S stereocentres: `_stereo_route` (the caller's post-processor) can add
    # a descriptor for exactly ONE centre and ONLY when we do not already lead with
    # an E/Z block. So fail closed when EITHER we emit an E/Z descriptor and any R/S
    # exists (double-apply guard would drop the R/S), OR there are >=2 R/S centres
    # (`_add_substituent_stereo`'s multi-centre branch returns the name UNCHANGED,
    # dropping ALL R/S -> a stereo-dropped WRONG molecule; fable-polyene BLOCKER 2,
    # invariant 9). A single centre with no descriptor is left for `_stereo_route`
    # to locate (RT-verified, e.g. `(S)-...`).
    n_chiral = sum(
        1 for i in sub_set
        if mol.GetAtomWithIdx(i).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)
    if n_chiral >= 2 or (descriptor and n_chiral >= 1):
        return None

    # Build the ene stem: mono -> 'prop-1-en'; poly -> euphonic-'a' 'penta-1,3-dien'.
    eloc_str = ",".join(str(x) for x in ene_locs)
    if len(ene_locs) == 1:
        ene_stem = f"{stem}-{eloc_str}-en"
    else:
        emult = {2: "di", 3: "tri", 4: "tetra"}.get(len(ene_locs))
        if emult is None:
            return None  # > tetraene -> out of scope
        ene_stem = f"{stem}a-{eloc_str}-{emult}en"

    return f"{descriptor}{joined}{ene_stem}-{yl_loc}-yl"


# --- P-16.2.4.1(a): the hyphen between a word fragment and a following locant --
#
# BB:6936, section heading "P-16.2.4 Hyphens":
#   "P-16.2.4.1 Hyphens are used in substitutive names:
#    (a) to separate locants from words or word fragments;
#    Example: 2-chloro-2-methylpropane (PIN, P-61.3.1)"
#
# Which strings count as a locant is fixed by BB:2847, section heading
# "P-14.3.1 Types of locants":
#   "Traditional types of locants are arabic numbers, for example, 1, 2, 3;
#    primed locants, for example, 1', 1''', 2''; locants including a lower case
#    Roman letter, for example, 3a, 3b; ITALICIZED ROMAN LETTERS, for example,
#    O, N, P; ..."
# and the same paragraph admits the composite forms 'N^2' and 'O^3'.
#
# So an italic element locant takes the separating hyphen exactly as an arabic
# one does -- confirmed verbatim by the PIN example under P-16.2.4.1(b),
# BB:6944: "N1-(2-aminoethyl)-N1,N2,N2-trimethylethane-1,2-diamine (PIN,
# P-62.2.4.1.3)".
#
# The counter-rule that bounds this is P-16.2.4.2 (BB:6968): "No hyphen is
# placed after a numerical prefix cited in front of a compound substituent
# enclosed by parentheses, even if that substituent begins with locants" --
# example "N,1-bis(4-chlorophenyl)methanimine (PIN)". A '(' is not a locant, so
# the predicate below declines it and 'bis(' is never split.
_ITALIC_ELEMENT_LOCANT_RE = re.compile(r"[NOSP]\d*['′]*\Z")


def _starts_with_locant(text: str) -> bool:
    """True when ``text`` opens with a cited locant set (P-14.3.1).

    Recognises the arabic forms ('2-methyl', '2,12-dimethyl') and the italic
    element forms ('N-methyl', 'O-methyl', 'N1-(2-aminoethyl)', "N2'-..."), the
    latter only when the whole hyphen-terminated head parses as locants -- so a
    plain word fragment ('methyl', 'tert-butyl', 'beta-D-glucopyranosyl') is
    declined and configurational/CIP letters (D, L, R, S as '(S)-') never reach
    it, being parenthesised.

    The leading-digit arm is deliberately the unchanged legacy test, so every
    arabic case keeps its historical answer byte-for-byte; the italic arm is
    purely additive.
    """
    if text[:1].isdigit():
        return True
    head, sep, _rest = text.partition("-")
    if not sep or not head:
        return False
    return all(_ITALIC_ELEMENT_LOCANT_RE.fullmatch(tok) for tok in head.split(","))


def _prefix_stem_yl(prefix: str, stem: str) -> str:
    """Assemble ``<prefix><stem>yl`` under P-16.2.4.1(a).

    ``prefix`` is a word fragment that may itself carry a leading locant
    ('3-amino'); ``stem`` is a parent-hydride stem that may itself *begin* with
    one ('2,12-dimethyltetradec', 'N1-(2-aminoethyl)-N2-methyleth'). When it
    does, the two must be separated by a hyphen or the locant fuses into the
    preceding word and the name denotes nothing:
    '3-amino' + '2,12-dimethyltetradec' + 'yl' must be
    '3-amino-2,12-dimethyltetradecyl', not '3-amino2,12-dimethyltetradecyl'.
    """
    if (prefix and stem and (prefix[-1].isalpha() or prefix[-1] in ")]}")
            and _starts_with_locant(stem)):
        return f"{prefix}-{stem}yl"
    return f"{prefix}{stem}yl"


def _joined_prefix_parts(parts: List[str]) -> List[str]:
    """Join alphabetised prefix parts inserting a hyphen between a letter (or a
    closing enclosing mark) and a following locant (e.g. 'amino' +
    '2-carboxy' -> 'amino-2-carboxy'; '2-(hydroxymethyl)' + '5-oxo' ->
    '2-(hydroxymethyl)-5-oxo'). W2F-P3: the closing-bracket case is needed by the
    branched-substituent path, whose complex sub-branch parts end in ')'; the
    linear path never emits bracketed parts, so it is byte-identical.

    v29 P7 T4: the locant test is now the shared ``_starts_with_locant`` so that
    italic element locants ('N1-', 'O-') take the P-16.2.4.1(a) hyphen too. The
    arabic answer is unchanged by construction (see that function's docstring)."""
    out = []
    for i, p in enumerate(parts):
        if (i > 0 and out and (out[-1][-1].isalpha() or out[-1][-1] in ")]}")
                and _starts_with_locant(p)):
            out.append("-")
        out.append(p)
    return out


def _ether_chain_locants_omitted(mol, sub_atoms, backbone, groups,
                                  attach_locant: int = 1) -> bool:
    """P-14.3.4 for the ether-substituted-chain SUBSTITUENT scope.

    True => this substituent group's own enclosing-mark scope cites NO locants.

    ⚠ Replaces a hand-rolled ``cite_locants = len(backbone) > 1`` that appeared
    nowhere in the Blue Book (recorded as an unaudited private licence in
    ``). The
    ANSWER it gave for a one-carbon backbone is right; the REASON was missing, and
    on a >1-carbon backbone it denied a licence P-14.3.4.5 grants.

    Two licences can fire, both delegated whole to ``assembly.locant_omission``:

    * `**P-14.3.4.6**` (BB:3031) *"All locants are omitted for parent compounds when
      all substitutable hydrogen atoms have the same locant."* -- a ONE-carbon
      backbone puts every substitutable hydrogen of the ``methyl`` parent hydride at
      locant 1, so ``-CH(O-CH3)2`` is ``dimethoxymethyl``, not
      ``1,1-dimethoxymethyl``.

      ★ **The printed positive is the Blue Book's own preferred prefix**
      ``diphenylmethyl`` (BB:16408 ``diphenylmethyl (preferred prefix) (not
      benzhydryl)``; BB:55912 ``| diphenylmethyl* | (C6H5)2CH– | P-29.6.3 |``).
      (C6H5)2CH- is the identical shape: a ``methyl`` substituent group with 2 of
      its 3 substitutable hydrogens replaced -- **partial** substitution, so
      P-14.3.4.5 does NOT apply and BB:3009's *"In case of partial substitution ...
      all numerical prefixes must be indicated"* would demand a locant -- yet the
      preferred prefix carries none. P-14.3.4.6 is the licence that explains it,
      and this is also why the rule's *"parent compounds"* wording is read as
      reaching a substituent group: P-14.3.4.5 one clause earlier says
      *"compounds **or substituent groups**"*, and BB:40566
      ``dichloromethylidene (PIN)`` is a second locant-free substituent-group row.

    * `**P-14.3.4.5**` (BB:3007) uniform COMPLETE substitution, the
      ``(pentafluoroethyl)`` case, for a backbone longer than one carbon whose every
      substitutable hydrogen is replaced by the same oxy prefix.

    Deny-by-default: anything not positively established returns False (cite).

    The two ambient P-14.3.3 scopes are consulted because this licence empties a
    scope of ALL its locants -- see
    ``.
    The **fragment-boundary observation** (``_naming_call_produces_a_name_component``)
    is deliberately NOT consulted, and that is not a two-of-three omission: it asks
    *"is the molecule I measured the molecule whose name I am editing?"*, and here the
    unit being measured IS this substituent's own enclosing-mark scope, which
    P-14.3.3 (BB:2869) scopes to *"a unit of structure as defined by its appropriate
    enclosing marks"*. Consulting it would gag the licence unconditionally (a
    substituent naming is always nested inside a larger name) and would therefore
    also break ``(pentafluoroethyl)cyclohexane``. Same reading as the sibling
    substituent-scope licence ``_l5_substituent_prefix`` above, which predates the
    observation and likewise does not consult it.
    """
    from .locant_omission import (
        l5_uniform_complete, l6_all_substitutable_h_share_one_locant,
        locants_are_forced, scope_forces_locants,
        scope_has_isotopic_modification,
    )

    # The two ambient P-14.3.3 declarations (isotope path names a label-STRIPPED
    # skeleton, so no structural test below can see the label).
    if locants_are_forced():
        return False
    if scope_has_isotopic_modification():
        return False

    if mol is None or not backbone or not groups or not sub_atoms:
        return False

    # v33 Phase 6 E2d: the caller now also reaches this for a NON-TERMINAL free
    # valence (`attach_locant` >= 2, e.g. 'propan-2-yl'). For k >= 2 that locant
    # is itself essential -- it distinguishes e.g. propan-2-yl from
    # propan-1-yl -- so P-14.3.3 (BB:2869) *"if any locants are essential ...
    # then all locants must be cited ... for that structural unit"* restores
    # the substitution locants and this licence declines. This mirrors the
    # identical, already-reviewed reading in the sibling all-carbon licence
    # `_l5_substituent_prefix` (see its own "TERMINAL FREE VALENCE ONLY"
    # derivation); the Blue Book prints no fully-substituted substituent group
    # with an internal free valence, so deny-by-default picks retention here
    # too. k==1 is the ONLY shape the two P-14.3.4 licences below were ever
    # derived against, so this guard also keeps them byte-identical for every
    # terminal-attachment caller that predates E2d.
    if not isinstance(attach_locant, int) or isinstance(attach_locant, bool) \
            or attach_locant != 1:
        return False

    chain_len = len(backbone)
    # Free valence at locant 1 (P-29.2, BB:15813 *"the atom with the free valence
    # terminates a chain and always has the locant '1'"*). Established BY
    # CONSTRUCTION in the caller (verified via `attach_locant == 1` just above,
    # not merely assumed from a terminal-only restriction that no longer
    # exists). Only `len(backbone)` is read here, so the BFS order of
    # `backbone` is irrelevant; every locant coming in via `groups` is
    # range-checked below.
    hydride = _l5_chain_parent_hydride(chain_len, 1)
    if hydride is None:
        return False

    # Marshal {hydride atom index -> the ONE decoration kind there} + multiplicity.
    kind_at: Dict[int, set] = {}
    count_at: Dict[int, int] = {}
    for kind, locants in groups.items():
        if kind is None or not str(kind).strip():
            return False
        for loc in (locants or []):
            if isinstance(loc, bool) or not isinstance(loc, int):
                return False
            if not 1 <= loc <= chain_len:
                return False
            kind_at.setdefault(loc - 1, set()).add(kind)
            count_at[loc - 1] = count_at.get(loc - 1, 0) + 1
    if not kind_at:
        return False

    # P-14.3.3: anything ESSENTIAL in the same scope restores every locant.
    has_stereo = any(
        mol.GetAtomWithIdx(int(i)).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for i in sub_atoms
    )
    if scope_forces_locants(
        prefix_locants=[loc for locs in groups.values() for loc in (locs or [])],
        suffix_locants=[],          # a substituent prefix scope has no suffix
        stereo_text='stereo' if has_stereo else '',
        has_indicated_h=False,      # backbone verified acyclic by the caller
        has_isotope=any(mol.GetAtomWithIdx(int(i)).GetIsotope() for i in sub_atoms),
        is_multiplicative=False,
        is_ring_assembly=False,
        has_skeletal_replacement=False,
    ):
        return False

    # P-14.3.4.6 -- every substitutable hydrogen of the parent hydride at one locant.
    atom_to_locant = {a.GetIdx(): a.GetIdx() + 1 for a in hydride.GetAtoms()}
    if l6_all_substitutable_h_share_one_locant(hydride, atom_to_locant):
        return True

    # P-14.3.4.5 -- uniform COMPLETE substitution of the parent hydride.
    #
    # ⚠ MUTATION-SURVIVING, DELIBERATELY KEPT. Measured 2026-07-30: replacing this
    # whole leg with ``return False`` changes no name, because it is unreachable
    # today from BOTH sides -- for a one-carbon backbone P-14.3.4.6 above has already
    # returned True, and the >1-carbon shape that would need it (every substitutable
    # hydrogen of an ethyl backbone replaced by the same oxy prefix,
    # ``c1ccccc1C(OC)(OC)C(OC)(OC)OC``) is declined upstream by this producer's own
    # accounting and names ``unknown organic compound``. It is the same rule the
    # sibling ``_l5_substituent_prefix`` implements for halogens, it is the correct
    # answer if that shape ever becomes nameable, and it is a *narrowing* condition on
    # an already-denied path -- so keeping it cannot license anything the rule does
    # not. Do not "simplify" it away without re-running that witness.
    if any(len(kinds) != 1 for kinds in kind_at.values()):
        return False                # two kinds at one position -> not "in the same
                                    # way" -> BB:3009
    decoration_of = {idx: next(iter(kinds)) for idx, kinds in kind_at.items()}
    return bool(l5_uniform_complete(hydride, decoration_of=decoration_of,
                                    counts=count_at))


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

    # Backbone = carbons reachable from the attachment WITHOUT crossing ANY
    # heteroatom. Must be all-carbon, acyclic, saturated. A heteroatom
    # neighbour simply stops the walk here -- it is classified below as a
    # potential backbone-attached ether/thioether LINK. (v33 Phase 6 DROP-24
    # aryloxymethyl fix: this walk used to depend on a whole-fragment
    # heteroatom front filter that rejected on ANY non-O/S atom anywhere in
    # the fragment, including one buried inside an R group -- e.g. the nitro N
    # of `4-nitrophenoxy` -- which aborted the whole handler before it ever
    # reached the R-group namer. The walk no longer needs that precomputed
    # set: a non-carbon neighbour is simply not part of the backbone.)
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
            if ni not in sub_set or ni in seen:
                continue
            if nbr.GetSymbol() != 'C':
                continue  # heteroatom neighbour -> a link candidate, not backbone
            bond = mol.GetBondBetweenAtoms(cur, ni)
            if bond.GetBondType() != Chem.BondType.SINGLE:
                return None  # unsaturated backbone -> fall through
            seen.add(ni)
            stack.append(ni)
    backbone_set = set(backbone)

    # Identify ether-type links: neutral, divalent, acyclic, both neighbours in
    # the fragment, both bonds single. P-63.2.5/P-29.5.2: O -> (R)oxy prefix, S
    # -> (R)sulfanyl prefix (W2E-P1FC Task 8, the -CH2-S-R concatenation). Any
    # other decoration (charged, =O, -OH, ring, peroxide -O-O-/-S-S-) directly
    # ON THE BACKBONE disqualifies the whole fragment (fail-closed) -- a richer
    # producer owns it.
    #
    # Only heteroatoms DIRECTLY bonded to a backbone carbon are held to this
    # strict test. A heteroatom with NO backbone neighbour is INTERIOR to an R
    # group (e.g. that same nitro N, or a ring heteroatom) and is unrestricted
    # here: it is accounted for by the recursive R-group namer
    # (`get_alkoxy_prefix` / `name_substituent_fragment`) below, never by this
    # front filter, and by the atom-coverage check that follows (no atom may
    # go unaccounted -> no silent drop).
    link_candidates: Set[int] = set()
    for idx in backbone:
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            ni = nbr.GetIdx()
            if ni in sub_set and ni not in backbone_set:
                link_candidates.add(ni)

    # P-46/P-57.1.6.2: an O/S with BOTH neighbours on the backbone is an
    # in-backbone (replacement) heteroatom -> not this handler (fail closed).
    ether_os: List[int] = []
    for idx in link_candidates:
        atom = mol.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()
        if sym not in ('O', 'S'):
            return None  # non-ether functional group directly on the backbone
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
        bb_n = sum(1 for n in nbrs if n in backbone_set)
        if bb_n == 2:
            return None  # in-backbone ether -> replacement parent, not a link
        ether_os.append(idx)
    if not ether_os:
        return None  # no backbone-bridging ether -> fall through
    ether_set = set(ether_os)

    # Linear (unbranched) backbone -- every backbone atom has at most 2
    # backbone neighbours, so the backbone as a whole is a simple path. The
    # attachment may sit at EITHER a terminal (primary, k==1) or an INTERNAL
    # (k>=2) position of that path (v33 Phase 6 E2d): a non-terminal free
    # valence is numbered from whichever end gives it the lowest locant
    # (P-46.1.8), exactly like the all-carbon `_located_acyclic_alkyl_name`
    # sibling -- '1,1-dimethoxypropan-2-yl' for -CH(CH3)-CH(OCH3)2 attached at
    # the middle carbon (measured: the OLD terminal-only restriction declined
    # this shape entirely, so it fell through to the recursive Tier-4 path,
    # which capped + renamed the H-capped free molecule and silently dropped
    # the attachment locant -> '1,1-dimethoxypropyl', a locant-omission bug).
    for idx in backbone:
        c_nbrs = sum(1 for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                     if n.GetIdx() in backbone_set)
        if c_nbrs > 2:
            return None  # branched backbone -> fall through

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

    # Number the linear backbone. For a TERMINAL attachment the walk starting
    # at attach_idx already gives it locant 1 (byte-identical to the prior
    # behaviour). For an INTERNAL attachment, walk the path from either end
    # and orient so the free valence gets the LOWEST locant (P-46.1.8); on an
    # exact-centre tie, break toward the lower ether-substituent locant set at
    # the first point of difference (P-29.4.1 -- mirrors
    # `_located_acyclic_alkyl_name`'s `_orient`).
    if len(backbone) == 1:
        ordered = [attach_idx]
    else:
        endpoints = [
            idx for idx in backbone
            if sum(1 for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                   if n.GetIdx() in backbone_set) <= 1
        ]
        if len(endpoints) != 2:
            return None  # not a simple path -> fall through (defensive)
        walk = [endpoints[0]]
        seen_w = {endpoints[0]}
        cur = endpoints[0]
        while len(walk) < len(backbone):
            nxt = None
            for n in mol.GetAtomWithIdx(cur).GetNeighbors():
                ni = n.GetIdx()
                if ni in backbone_set and ni not in seen_w:
                    nxt = ni
                    break
            if nxt is None:
                break
            walk.append(nxt)
            seen_w.add(nxt)
            cur = nxt
        if len(walk) != len(backbone) or attach_idx not in walk:
            return None  # disconnected backbone -> fall through
        idx_a = walk.index(attach_idx)
        k_fwd = idx_a + 1
        k_rev = len(walk) - idx_a
        if k_rev < k_fwd:
            ordered = list(reversed(walk))
        elif k_rev > k_fwd:
            ordered = walk
        else:
            # Exact-centre tie: prefer the direction giving the lower sorted
            # set of ether-bearing-carbon locants (the substituents' own
            # positions), reusing `ether_links` already built above.
            def _ether_locs(order):
                p = {a: i + 1 for i, a in enumerate(order)}
                return sorted(p[bb_c] for bb_c, _, _ in ether_links)
            rev_walk = list(reversed(walk))
            ordered = rev_walk if _ether_locs(rev_walk) < _ether_locs(walk) \
                else walk
    if len(ordered) != len(backbone):
        return None  # disconnected backbone -> fall through
    pos = {idx: i + 1 for i, idx in enumerate(ordered)}
    k_attach = pos[attach_idx]

    # Name each ether-type link as an (R-oxy)/(R-sulfanyl) prefix at its
    # backbone-carbon locant.
    from collections import defaultdict
    from ..data.chain_names import get_chain_prefix
    from .substituent_prefix_forms import get_alkoxy_prefix, get_sulfanyl_prefix
    from .naming_utils import (
        is_complex_substituent, apply_enclosing_marks, _is_fully_enclosed,
        starts_with_locant,
    )
    groups: dict = defaultdict(list)
    for bb_c, o_idx, r_side in ether_links:
        if mol.GetAtomWithIdx(o_idx).GetSymbol() == 'O':
            oxy = get_alkoxy_prefix(mol, (o_idx, bb_c, r_side), backbone)
            if not oxy or oxy == "alkoxy":
                return None  # un-nameable R -> fall through (fail-closed)
            # P-16.3.3/P-16.5: a CITED-LOCANT oxy prefix -- whether it already
            # carries marks from a nested recursion ('(4-methoxyphenyl)methoxy')
            # OR is a bare-but-substituted retained contraction ('4-nitrophenoxy',
            # '4-methylphenoxy' -- P-63.2.2.2 keeps the CONTRACTED spelling even
            # substituted, e.g. BB verbatim '(4-chlorophenoxy)benzene',
            # '(2-nitrophenoxy)borane') -- must be enclosed AS A UNIT before
            # concatenating the backbone stem: '(4-nitrophenoxy)methyl', mirroring
            # the BB-verbatim sub-component pattern for the analogous (X-phenyl)
            # case ('[(3-chlorophenyl)methyl]benzene', '2-[(4-bromophenyl)methyl]
            # pyridine', '4-[(4-hydroxyphenyl)methyl]phenol' -- the retained
            # contraction changes the SPELLING of the atomic unit, not whether a
            # decorated, LOCANTED version of it needs its own marks as a
            # sub-component).
            #
            # A RETAINED oxy prefix that carries NO locant of its own must stay
            # bare, even though it is structurally a fused two-part compound:
            # 'phenoxy', 'methoxy', 'benzyloxy', 'cyclohexyloxy' (P-63.2.2.2 /
            # P-29.6.1, BB '2-benzylpyridine (PIN)' -- the retained-ROOT-word
            # test, not a general compound-substituent test). ``is_complex_
            # substituent``/``enclose_if_compound`` are the wrong primitive
            # here -- a FIRST ATTEMPT used ``enclose_if_compound`` and it
            # regressed a gold row: 'benzyloxy' and 'cyclohexyloxy' trip
            # ``is_complex_substituent`` as a two-morpheme compound (like
            # 'cyclohexylmethyl') even though they carry no substituent locant,
            # over-nesting 'Oc1ccc(COCc2ccccc2)cc1' from the gold
            # '4-(benzyloxymethyl)phenol' to a wrong '4-[(benzyloxy)methyl]
            # phenol'. The decision this call site needs is narrower than
            # "is this compound" -- it is "does this oxy prefix CITE A LOCANT" --
            # so use ``starts_with_locant`` (the shared primitive for exactly
            # that test, e.g. distinguishing 'chloro' from '4-chloro') instead.
            if (not _is_fully_enclosed(oxy)
                    and (('(' in oxy or '[' in oxy) or starts_with_locant(oxy))):
                oxy = apply_enclosing_marks(oxy, -1)
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
            # P-16.3.3/P-16.5: a COMPOUND R ('methoxymethyl') must itself be
            # enclosed before concatenating 'sulfanyl' -> '(methoxymethyl)
            # sulfanyl'; a simple/retained R ('benzyl','methyl') stays bare ->
            # 'benzylsulfanyl'. Then '{R}sulfanyl' is a compound prefix and is
            # ALWAYS enclosed (escalating when it already carries brackets):
            # '(benzylsulfanyl)', '[(methoxymethyl)sulfanyl]'. The outer mark
            # escalates again on the parent.
            _r_enclosed = (apply_enclosing_marks(_r_name, -1)
                           if is_complex_substituent(_r_name) else _r_name)
            oxy = apply_enclosing_marks(f"{_r_enclosed}sulfanyl", -1)
        groups[oxy].append(pos[bb_c])

    # ---- P-16.3 multiplier + P-14.3.4 locant licence, both DELEGATED. ----
    # This block used to carry a private ``_MULT = {2: 'bis', 3: 'tris', ...}`` (one
    # of the 27 divergent multiplier tables measured in
    # ``) and a
    # hand-rolled ``cite_locants = len(backbone) > 1`` locant licence that appears
    # nowhere in the Blue Book. Both decisions now go to the shared primitives.
    from .naming_utils import (
        _has_stereo_prefix, is_substituted_substituent, multiplied_component,
    )
    cite_locants = not _ether_chain_locants_omitted(
        mol, sub_atoms, backbone, groups, attach_locant=k_attach)
    part_strings = []
    # Alphanumerical order by the (compound) oxy prefix name (P-14.5.2).
    for prefix in sorted(groups.keys(), key=alpha_sort_key):
        locs = sorted(groups[prefix])
        n_occ = len(locs)
        # `**P-16.3.5**` (BB:7104) "The numerical prefixes 'bis', 'tris',
        # 'tetrakis', etc. are used to indicate a multiplicity of: (a) compound or
        # complex (i.e. substituted) prefixes" -- so the multiplier turns on
        # "is the component SUBSTITUTED?", which is exactly what
        # ``get_multiplier_prefix`` (via ``multiplied_component``) already decides
        # for every other producer. The private table above answered 'bis'
        # UNCONDITIONALLY, so it emitted ``bis(methoxy)`` where the Blue Book
        # prints the CONTRACTED prefix's simple multiplier:
        #   BB:5098   CH3-CH2-CH(O-CH3)2       1,1-dimethoxypropane (PIN)
        #   BB:35344  CH3(CH2)3-CH(S-CH3)2     1,1-bis(methylsulfanyl)pentane (PIN)
        # One skeleton shape, two spellings: ``methoxy`` is derived "from a
        # contracted name" (BB:17958, verbatim, P-63.2.2.2) and is therefore a
        # SIMPLE prefix taking ``di``/no marks, while the uncontracted
        # ``methylsulfanyl`` is substituted and takes ``bis(...)``. ``bis(methoxy)``
        # occurs ZERO times in the Blue Book; ``dimethoxy`` occurs in five printed
        # PINs/preferred prefixes (:5098, :27705, :27754, :36275, :37045).
        # Enclosure accompanies the complex multiplier (P-16.3.5(a)'s own examples
        # are all enclosed: ``bis(bromomethyl)``, ``bis(dimethylamino)``), and a
        # prefix this producer already enclosed whole (the S-branch's
        # ``(methylsulfanyl)``) is NOT re-enclosed -- that would give
        # ``bis[(methylsulfanyl)]`` against BB:35344.
        if n_occ > 1 and (is_substituted_substituent(prefix)
                          or is_complex_substituent(prefix)
                          or _has_stereo_prefix(prefix)) \
                and not _is_fully_enclosed(prefix):
            marked = apply_enclosing_marks(prefix, -1)
        else:
            marked = prefix
        head = multiplied_component(n_occ, prefix, marked)
        if cite_locants:
            loc_str = ",".join(str(loc) for loc in locs)
            part_strings.append(f"{loc_str}-{head}")
        else:
            part_strings.append(head)

    stem = get_chain_prefix(len(backbone))
    if cite_locants:
        joined = '-'.join(part_strings)
    elif len(part_strings) >= 2:
        # P-16.3.3 / P-16.3.4: >=2 DISTINCT substituent prefixes co-cited on the
        # same LOCANT-ELIDED position must be set off by enclosing marks. A bare
        # ''.join fuses adjacent (R)oxy prefixes into ONE compound (chained)
        # prefix -- 'ethoxymethoxy' reads as an ethoxy-substituted methoxy CHAIN
        # (CH3CH2-O-CH2-O-, a DIFFERENT constitution), so OPSIN mis-parses
        # 'ethoxymethoxymethyl' and SELF-01 correctly rejected it, dropping the
        # engine to the ugly a-replacement rescue. Separate them: the first
        # prefix stays bare, each LATER distinct prefix is enclosed (already-
        # enclosed tokens, e.g. an S-branch '(methylsulfanyl)', are left as-is --
        # they already provide the separation) -> 'ethoxy(methoxy)methyl'
        # (OPSIN-RT-verified). A single distinct prefix -- incl. the 'di'-
        # multiplied 'dimethoxy' from -CH(OMe)2 -- is len==1 and stays bare, so
        # no over-enclosure of the single/identical-group paths.
        joined = part_strings[0] + ''.join(
            p if _is_fully_enclosed(p) else apply_enclosing_marks(p, -1)
            for p in part_strings[1:])
    else:
        joined = ''.join(part_strings)
    # v33 Phase 6 E2d: a NON-TERMINAL free valence (k_attach >= 2) must cite its
    # own locant on the parent-hydride stem -- 'propan-2-yl', not 'propyl'
    # (P-29.2, BB:15813 elides the locant only when the free valence
    # "terminates a chain", i.e. k==1). A hyphen is needed before the digit
    # whenever the joined prefix string ends in a letter or a closing mark.
    if k_attach == 1:
        return f"{joined}{stem}yl"
    return f"{joined}{stem}an-{k_attach}-yl"


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

    # Aldehyde functional parent -> the DEFINED prefix, not a '-yl' transform.
    # P-66.6.1.3 (BB:35000, under P-66.6 ALDEHYDES): "In the presence of a
    # characteristic group having priority to be cited as a suffix or when
    # present on a side chain, a -CHO group is expressed by the preferred
    # prefix 'oxo' if located at an end of a carbon chain, or, otherwise, by
    # the preferred prefix 'formyl'."  A substituent named through this
    # converter is by construction NOT part of the parent chain, so the
    # 'otherwise' arm applies (cf. BB:35009 '4-formylcyclohexane-1-carboxylic
    # acid (PIN)').  P-65.1.8.3 (BB:30702) confirms the spelling and that the
    # H of -CHO is substitutable.  Formaldehyde has ONE carbon, so -CHO is the
    # only substituent derivable from it -- the conversion is unambiguous.
    # ACETaldehyde is deliberately absent: with two carbons the prefix is
    # 'acetyl' (attachment at the carbonyl C) or '2-oxoethyl' (attachment at
    # the methyl C), and this converter receives no attachment context to
    # choose between them, so it fails closed below instead.
    'formaldehyde': 'formyl',

    # HCN as a substituent -> the DEFINED prefix. P-66.5.1.1.4 (BB:34734):
    # "When a group is present that has priority for citation as the principal
    # characteristic group or when all -CN groups cannot be expressed as the
    # principal characteristic group, the -CN group is designated by the
    # preferred prefix 'cyano'."  BB:34687 derives nitriles "from hydrocyanic
    # acid, H-C=N".  One carbon, one removable H, so unambiguous.  Without this
    # the terminal fallback produced 'hydrogen cyanidyl' -- the sibling defect
    # named in rules/ring_assemblies.py's veto comment.
    'hydrogen cyanide': 'cyano',

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
                     'peroxoic acid', 'imidic acid', 'ohydroximic acid',
                     'thioacetic acid', 'selenoacetic acid', 'telluroacetic acid',
                     'peracetic acid', 'peroxyacetic acid')) or 'perox' in low:
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
    # Retained acetamide family (P-66.1.1.4.3 method (1), BB:32995): the prefix is
    # the AMIDE name with final 'e'->'o'. 'acetamide' is a retained amide PIN, and
    # its substituted amide REQUIRES the C2 locant (acetamide has two substitutable
    # sites, N and C2, and 'N-phenylacetamide' = acetanilide is a DIFFERENT molecule,
    # BB:32859) -- so 'phenylacetic acid' -> '2-phenylacetamide' -> '2-phenylacetamido'
    # (verified: the amide of NC(=O)Cc1ccccc1 is named '2-phenylacetamide'; the acid
    # legitimately omits the locant per P-14.3.4.6, the amide may not). The acid name
    # already carries an explicit locant when systematic ('2-phenylacetic acid'), so
    # insert '2-' ONLY when the stem has no leading locant of its own. Bare
    # 'acetic acid' is the retained table above. A multiplied stem ('diacetic acid',
    # 'oxydiacetic acid') is a poly-acid one amido prefix cannot describe -> fail
    # closed, mirroring the 'carboxylic acid' / 'dioic acid' guards.
    if low.endswith('acetic acid'):
        stem = name[:-len('acetic acid')]
        if re.search(r'(?:di|tri|tetra|penta|hexa)[-,\d]*$', stem.lower()):
            return None
        if stem and not stem[0].isdigit():
            stem = '2-' + stem
        return stem + 'acetamido'
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


def n_substituted_acyl_amido_prefix(mol, n_idx: int, sub_atoms,
                                    parent_atoms) -> Optional[str]:
    """P-66.1.1.4.3 method (1): an N-SUBSTITUTED acylamino branch
    ``-N(R')-C(=O)-R`` is the ``{N-R'}{acyl}amido`` PREFIX -- N-methylacetamido,
    N-methylformamido, N-ethylpropanamido (BB:32995; verbatim
    ``2-(N-methylpropanamido)benzene-1-sulfonic acid (PIN)`` :33040).

    The unsubstituted sibling ``linear_acyl_amido_prefix`` fails closed on a
    substituted N (its ``sub_set == branch | {n_idx}`` check), and the legacy
    count fallback in ``_name_amino_branch`` refuses a non-mono-substituted N
    (37cd122d F1), so before this builder the whole ``-N(R')C(=O)R`` fragment
    dropped to the ugly general replacement name (``1,2-dimethyl-3-oxa-1-
    azaprop-2-en-1-yl``) or abstained.

    PIN ONLY as a PREFIX: this fires in ``_name_amino_branch`` (a substituent
    namer), reached only when parent selection already made the amide a prefix
    (a senior characteristic group is elsewhere). When the amide is the
    PRINCIPAL group it is the SUFFIX -- BB:33048 marks ``4-(N-methylacetamido)
    quinoline`` explicitly NOT PIN, and Orthonym already names that molecule
    ``N-methyl-N-(quinolin-2-yl)acetamide`` via the suffix path, which this
    builder never sees.

    Strict / fail-closed. Fires ONLY when, with FULL atom coverage:
      * N is uncharged/unradical/unlabelled, NOT a ring member, all single
        bonds, and has EXACTLY one bond into the parent (a bare tertiary amide N);
      * N carries EXACTLY two non-parent heavy branches -- the acyl C (a clean
        C=O) and one other substituent R';
      * the acyl side names via ``linear_acyl_amido_prefix`` (an unbranched
        saturated acyclic all-carbon acyl -> formamido/acetamido/{stem}anamido);
      * R' names via the recursive substituent namer;
      * the fragment is EXACTLY {N} + acyl-subtree + R'-subtree (nothing dropped).

    Returns the BARE prefix core ``'N-{R'}{amido}'`` (the caller applies the
    P-16.5.1.1 enclosing marks -- ``needs_brackets('N-methylacetamido')`` is
    True, so it renders ``(N-methylacetamido)``); None otherwise.
    """
    sub_set = set(sub_atoms)
    parent_set = set(parent_atoms)
    if n_idx not in sub_set:
        return None
    n_atom = mol.GetAtomWithIdx(n_idx)
    # A bare tertiary amide N usable as a prefix (mirror the sulfonamido self-guards).
    if n_atom.GetAtomicNum() != 7 or n_atom.GetIsAromatic() or n_atom.IsInRing():
        return None
    if (n_atom.GetFormalCharge() or n_atom.GetNumRadicalElectrons()
            or n_atom.GetIsotope()):
        return None
    if any(b.GetBondTypeAsDouble() != 1.0 for b in n_atom.GetBonds()):
        return None
    parent_nbrs = [nb.GetIdx() for nb in n_atom.GetNeighbors()
                   if nb.GetIdx() in parent_set]
    if len(parent_nbrs) != 1:
        return None
    branches = [nb.GetIdx() for nb in n_atom.GetNeighbors()
                if nb.GetIdx() not in parent_set and nb.GetIdx() in sub_set]
    if len(branches) != 2:
        return None

    def _is_acyl_c(i):
        a = mol.GetAtomWithIdx(i)
        if a.GetAtomicNum() != 6:
            return False
        return any(
            nb.GetAtomicNum() == 8
            and mol.GetBondBetweenAtoms(i, nb.GetIdx()).GetBondTypeAsDouble() == 2.0
            for nb in a.GetNeighbors()
        )

    acyl_candidates = [b for b in branches if _is_acyl_c(b)]
    if len(acyl_candidates) != 1:
        return None  # 0 (no acyl) or 2 (an imide -N(C=O)(C=O)) -> not this class
    acyl_c = acyl_candidates[0]
    r_prime = next(b for b in branches if b != acyl_c)

    def _subtree(start):
        seen = set()
        stack = [start]
        while stack:
            i = stack.pop()
            if i in seen or i == n_idx:
                continue
            seen.add(i)
            for nb in mol.GetAtomWithIdx(i).GetNeighbors():
                if nb.GetIdx() != n_idx and nb.GetIdx() not in seen:
                    stack.append(nb.GetIdx())
        return seen

    acyl_sub = _subtree(acyl_c)
    rp_sub = _subtree(r_prime)
    # Disjoint, and together with N exactly the fragment (no atom dropped/shared).
    if acyl_sub & rp_sub:
        return None
    if ({n_idx} | acyl_sub | rp_sub) != sub_set:
        return None

    # Isotope hole (fable review of the first F-amido cut): `linear_acyl_amido_prefix`
    # checks charge/radical on chain carbons but NOT isotope, and this producer must be
    # honest without the gate (the 8afa533c lesson). An amido stem drops any label, so a
    # labelled fragment would name a wrong isotopologue -- fail closed.
    if any(mol.GetAtomWithIdx(i).GetIsotope() for i in sub_set):
        return None

    # R' must be a plain substituent, NOT a second acyl-like group -- a carbon
    # double-bonded to a chalcogen (=O/=S/=Se/=Te). `-N(C=O)(C=S)` is a mixed imide,
    # not P-66.1.1.4.3 method (1); `name_substituent` would spell the thioacyl as
    # replacement-nomenclature junk (`1-methyl-2-thiaeth-1-en-1-yl`, non-PIN, gate-blind
    # because it round-trips). Mirror the 2-acyl imide refusal (fable review).
    rp_atom = mol.GetAtomWithIdx(r_prime)
    if rp_atom.GetAtomicNum() == 6 and any(
        nb.GetAtomicNum() in (8, 16, 34, 52)
        and mol.GetBondBetweenAtoms(
            r_prime, nb.GetIdx()).GetBondTypeAsDouble() == 2.0
        for nb in rp_atom.GetNeighbors()
    ):
        return None

    amido_core = linear_acyl_amido_prefix(mol, acyl_c, n_idx, {n_idx} | acyl_sub)
    if not amido_core:
        return None  # branched / unsaturated / ring / hetero acyl -> fail closed

    # Name R' AND enclose it with the SAME machinery the amide-SUFFIX path uses
    # (`_name_n_substituent` + `format_n_substitution`): it applies the P-16.5.1.1
    # enclosing marks, the P-16.5.4 nesting escalation, and embeds/escalates a stereo
    # descriptor (`N-[(S)-1-phenylethyl]`) correctly -- verified identical to
    # `CC(=O)N(C)[C@@H](C)c1ccccc1 -> N-methyl-N-[(S)-1-phenylethyl]acetamide`. The first
    # F-amido cut interpolated R' RAW (`f"N-{r_name}..."`), which left it UNBRACKETED and
    # (a) regressed a valid RT-exact emission to an abstention (unparseable
    # `N-2,2-dimethylpropylacetamido` -> SELF-01 suppressed -> the general fallback never
    # ran) and (b) shipped an OPSIN-unparseable stereo name via the stereo carve-out
    # (fable review). `_name_n_substituent` wants the attach atom first in the list.
    from ..rules.amides import _name_n_substituent, format_n_substitution
    rp_list = [r_prime] + [i for i in rp_sub if i != r_prime]
    carbon_count = sum(
        1 for i in rp_sub if mol.GetAtomWithIdx(i).GetAtomicNum() == 6)
    r_name = _name_n_substituent(mol, rp_list, carbon_count)
    if not r_name:
        return None
    from ..errors import is_refusal_sentinel
    if is_refusal_sentinel(r_name):
        return None  # recursion/depth fallback placeholder -> never splice `N-substituent...`
    n_seg = format_n_substitution(
        [{"atoms": rp_sub, "name": r_name, "carbon_count": carbon_count}])
    if not n_seg or is_refusal_sentinel(n_seg):
        return None
    prefix = f"{n_seg}{amido_core}"

    # Gate-INDEPENDENT re-anchor (the 8afa533c / F6 guard-2 precedent, mandatory because
    # best-effort T4 ships on the certificate, NOT on OPSIN-RT -- the producer must be
    # honest on its own). `name_substituent` can MIS-NAME R': e.g. `-CH2-S-CH3` ->
    # `methylsulfanyl` (a pre-existing fragment-namer defect that drops the CH2; the O
    # analog `-CH2-O-CH3` -> `methoxymethyl` is correct), so the composed prefix would
    # denote a DIFFERENT molecule (fable re-review BLOCKER: gate-on it regressed 2
    # HEAD-RT-exact rows to abstention, gate-off it shipped the wrong molecule). Re-anchor:
    # build the corresponding AMIDE (this prefix's parent characteristic group,
    # `...amido`->`...amide`), OPSIN-parse it, and require the SAME InChIKey as the
    # substituent fragment capped at N with an implicit H. Fail closed on mismatch or when
    # OPSIN is unavailable -> the cascade/general fallback then supplies a valid
    # replacement name (restoring HEAD behaviour), never a wrong or atom-dropped prefix.
    try:
        capped_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(sub_set))
        capped = Chem.MolFromSmiles(capped_smiles)
        if capped is None:
            return None
        capped_key = Chem.MolToInchiKey(capped)
    except Exception:
        return None
    if not amido_core.endswith("amido"):
        return None
    amide_name = prefix[:-1] + "e"  # trailing '...amido' -> '...amide'
    from ..namer import _validity_gate_name_to_smiles
    reparsed = _validity_gate_name_to_smiles(amide_name)
    if reparsed is None:
        return None
    # The InChIKey compare inherits InChI mobile-H (tautomer) equivalence -- a
    # wrong-TAUTOMER R' name would pass. That is the same equivalence the headline
    # round-trip metric uses (SELF-01's skeleton is looser still), so the guard sits
    # AT the project's correctness bar, not below it; no tautomer-divergent R' name is
    # reachable today (the R' namer refuses those with the `substituent` sentinel).
    try:
        rp_key = Chem.MolToInchiKey(Chem.MolFromSmiles(reparsed))
    except Exception:
        return None
    if not rp_key or rp_key != capped_key:
        return None
    return prefix


def sulfonamido_prefix_from_n_branch(mol, n_idx: int, sub_atoms,
                                     parent_atoms) -> Optional[str]:
    """P-66.1.1.4.3 (BB:32995): an N-attached ``-NH-SO2-R`` branch is the
    ``{R}sulfonamido`` PREFIX (methanesulfonamido / ethanesulfonamido /
    benzenesulfonamido / cyclohexanesulfonamido) — NOT an ``amino`` split of the
    N, and NOT the cascade's ``carbamoyl`` misroot (which swaps S->C and DROPS
    S, the two =O and R: a different molecule).

    The stem is built by the SAME acid-stem sulfonyl primitive the sulfone-prefix
    path uses (``_acid_stem_oxide_prefix(mol, r_carbon, s_idx, 'sulfonyl')`` ->
    'methanesulfonyl' / 'benzenesulfonyl' / 'cyclohexanesulfonyl'), with the
    suffix rewrite sulfonyl -> sulfonamido. That primitive returns None (so this
    fails closed) for a substituted-arene / CF3 / branched-hetero R -> those keep
    abstaining rather than shipping a wrong or atom-dropped name.

    Strict / fail-closed. Fires ONLY when, with FULL atom coverage:
      * N carries exactly ONE non-parent heavy neighbour, the sulfonyl S (a bare
        ``-NH-``, never N,N-disubstituted); N-S is a single bond,
      * S is a clean sulfonyl: exactly two terminal ``=O`` (both in the fragment)
        and exactly one further heavy neighbour R, a carbon,
      * the fragment is EXACTLY {N, S, the two =O} plus the R subtree — nothing
        dropped.

    Returns the BARE prefix core (e.g. 'methanesulfonamido'), matching every
    sibling return in ``_name_amino_branch``; None otherwise.
    """
    from .substituent_prefix_forms import _acid_stem_oxide_prefix

    sub_set = set(sub_atoms)
    parent_set = set(parent_atoms)
    if n_idx not in sub_set:
        return None
    n_atom = mol.GetAtomWithIdx(n_idx)
    # This is a SHARED primitive: one caller (`_name_compound_substituent`) guards
    # only `GetSymbol()=='N'`, so the helper must self-guard every way a fragment
    # can be a non-`-NH-` attachment (fable review of b5e4d3da). A monovalent
    # `-NH-SO2R` prefix requires: uncharged/unradical/unlabelled N, NOT a ring
    # member, ALL single bonds (an `=N-SO2R` sulfonimidoyl is a different bond
    # order / H count), and EXACTLY one bond into the parent (a bridging
    # `parent-N(-parent')-SO2R` is divalent, not a prefix).
    if (n_atom.GetSymbol() != 'N' or n_atom.GetFormalCharge()
            or n_atom.GetNumRadicalElectrons() or n_atom.GetIsotope()
            or n_atom.IsInRing()):
        return None
    if any(b.GetBondTypeAsDouble() != 1.0 for b in n_atom.GetBonds()):
        return None
    if sum(1 for nb in n_atom.GetNeighbors()
           if nb.GetIdx() in parent_set) != 1:
        return None
    branches = [nb.GetIdx() for nb in n_atom.GetNeighbors()
                if nb.GetIdx() not in parent_set and nb.GetIdx() in sub_set]
    if len(branches) != 1:
        return None
    s_idx = branches[0]
    s_atom = mol.GetAtomWithIdx(s_idx)
    if (s_atom.GetSymbol() != 'S' or s_atom.GetFormalCharge()
            or s_atom.GetNumRadicalElectrons() or s_atom.GetIsotope()):
        return None
    ns_bond = mol.GetBondBetweenAtoms(n_idx, s_idx)
    if ns_bond is None or ns_bond.GetBondTypeAsDouble() != 1.0:
        return None
    dbl_o = []
    r_side = []
    for nb in s_atom.GetNeighbors():
        ni = nb.GetIdx()
        if ni == n_idx:
            continue
        bond = mol.GetBondBetweenAtoms(s_idx, ni)
        if (nb.GetSymbol() == 'O' and nb.GetDegree() == 1
                and bond.GetBondTypeAsDouble() == 2.0):
            dbl_o.append(ni)
            continue
        r_side.append(ni)
    if len(dbl_o) != 2 or len(r_side) != 1:
        return None
    if any(o not in sub_set for o in dbl_o):
        return None
    r_idx = r_side[0]
    if r_idx not in sub_set or mol.GetAtomWithIdx(r_idx).GetSymbol() != 'C':
        return None
    # Atom coverage: the fragment must be EXACTLY {N, S, two =O} + the R subtree
    # (reachable from R without recrossing S). Any leftover atom -> silent drop.
    # `_acid_stem_oxide_prefix` guards the R constitution by ELEMENT only, so a
    # charged / radical / isotopically-labelled R would be spelled as the plain
    # stem and drop the label -> a different molecule; reject it here (the sibling
    # `linear_acyl_amido_prefix` rejects charge/radical the same way).
    r_subtree = set()
    stack = [r_idx]
    while stack:
        a = stack.pop()
        if a in r_subtree or a == s_idx:
            continue
        if a not in sub_set:
            return None  # R escapes the fragment
        _ra = mol.GetAtomWithIdx(a)
        if _ra.GetFormalCharge() or _ra.GetNumRadicalElectrons() or _ra.GetIsotope():
            return None
        r_subtree.add(a)
        for nb in _ra.GetNeighbors():
            if nb.GetIdx() != s_idx:
                stack.append(nb.GetIdx())
    if sub_set != {n_idx, s_idx, dbl_o[0], dbl_o[1]} | r_subtree:
        return None

    # `_acid_stem_oxide_prefix` already delegates to `_acid_stem_unsaturated_oxide_prefix`
    # for a substituted / unsaturated R (substituent_prefix_forms.py), so a substituted-
    # arene R (4-aminobenzene) reaches that builder here. v30 #29 gap-a fixed that
    # builder's piece-selection (it now picks the capped-sulfonyl-S fragment, not a
    # detached ring-sulfur host), which unblocks `2-(4-aminobenzene-1-sulfonamido)-
    # 1,3-thiazole-5-carboxylic acid`.
    stem = _acid_stem_oxide_prefix(mol, r_idx, s_idx, 'sulfonyl')
    if not stem or not stem.endswith('sulfonyl'):
        return None
    return stem[:-len('sulfonyl')] + 'sulfonamido'


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


def _elide_parent_hydride_ending(base: str):
    """Elide a parent hydride's ending so ``-yl`` can be appended (P-29.2).

    P-29.2 "GENERAL METHODOLOGY FOR NAMING SUBSTITUENT GROUPS" (BB:15811):
    *"Systematic names are formed by using the suffixes 'yl', 'ylidene' and
    'ylidyne', **with elision of the final letter 'e' of parent hydrides, when
    present**, according to two methods"* — method (1) (BB:15813): *"The
    suffixes 'yl', 'ylidene', and 'ylidyne' **replace the ending 'ane'** of the
    parent hydride name."*

    The callers previously elided only ``an``/``a`` and left ``ane`` intact,
    which is how ``ethane`` + ``yl`` became ``ethaneyl`` (BB occurrences of
    that string: zero).

    Returns the elided stem, or ``None`` when ``base`` carries no elidable
    parent-hydride ending — in which case the caller keeps its previous
    behaviour rather than guessing.
    """
    if base.endswith('ane'):
        return base[:-3]          # method (1): 'yl' replaces 'ane'
    if base.endswith('an'):
        return base[:-2]
    if base.endswith('ene') or base.endswith('yne'):
        return base[:-1]          # method (2): elide only the final 'e'
    if base.endswith('a'):
        return base[:-1]
    return None


# --- Functional parents that have NO '-yl' form ------------------------------
# P-29.2 (BB:15811, heading "GENERAL METHODOLOGY FOR NAMING SUBSTITUENT
# GROUPS") licenses the 'yl'/'ylidene'/'ylidyne' suffixes only for a PARENT
# HYDRIDE.  A FUNCTIONAL parent is not a parent hydride, so no '<name>yl' form
# exists for it — the Blue Book gives each of these classes a DEFINED prefix
# instead.
#
# This guard must run BEFORE the suffix branches below, because two of them
# match on a bare string ending and would otherwise CAPTURE these names: the
# unlocanted '-ol' branch turned 'ethaneperoxol' into 'hydroxyethaneperoxyl'
# and 'methanethiol' into 'hydroxymethanethiyl' (asserting an -OH where the
# molecule has -SH), and the unlocanted '-amine' branch turned
# 'O-methylhydroxylamine' into 'aminoO-methylhydroxylyl'.  Positioning, not
# absence, was the defect: the pre-existing RC-1 guard further down is correct
# but sits downstream of the branches that mis-capture.
#
# Only classes whose ending denotes a DIFFERENT characteristic group are listed.
# Genuine alcohols with unsystematic stems ('menthol', 'cholesterol',
# 'inositol', 'glycerol') are deliberately NOT here: 'hydroxy...yl' at least
# names the right element for them, and denying them would change names outside
# this defect class.
_FUNCTIONAL_PARENT_NO_YL_FORM = (
    # (name ending, the prefix the Blue Book uses instead)
    ('aldehyde',
     "'oxo' at a chain end, else 'formyl' (P-66.6.1.3, BB:35000)"),
    ('thioperoxol',
     "chalcogen analogue of 'hydroperoxy' (P-63.4.2, BB:27961)"),
    ('peroxol',
     "'hydroperoxy' (P-63.4.1, BB:27944)"),
    ('thiol',
     "'sulfanyl' — the group is -SH; 'hydroxy' would assert -OH"),
    ('hydroxylamine',
     "'hydroxyimino' / '(alkoxyimino)' (P-68.3.1.1.2, BB:38460)"),
    ('glycol',
     'a functional-class name, not a parent hydride'),
)

# Inorganic / functional-class parents matched as WHOLE names.  Deliberately
# exact rather than a "contains a space" rule: the census shows space-bearing
# ester parents ('henicosyl prop-2-enoate' -> '23-carboxytricosyl') convert
# legitimately through the '-oate' branch, so a blanket space rule would break
# them.  Each of these otherwise reached the terminal fallback and produced
# 'wateryl' / 'ammoniayl' / 'carbon dioxidyl'.
_NON_HYDRIDE_WHOLE_NAMES = frozenset({
    'water',            # as a substituent the group is -OH, prefix 'hydroxy'
    'ammonia',          # as a substituent the group is -NH2, prefix 'amino'
    'carbon dioxide',
    'carbon monoxide',
    'hydrogen peroxide',  # -OOH is 'hydroperoxy' (P-63.4.1, BB:27944)
})


class _AttachLocantUnknown:
    """Sentinel type for ``parent_to_prefix(attach_locant=...)``."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return "ATTACH_LOCANT_UNKNOWN"

    def __bool__(self) -> bool:
        return False


#: Explicit "this caller cannot prove where the free valence sits" value for
#: ``parent_to_prefix``. It is a distinct object rather than ``None`` so that a
#: caller which simply has no locant is never confused with one that computed
#: ``0``/``None`` by accident.
ATTACH_LOCANT_UNKNOWN = _AttachLocantUnknown()


def parent_to_prefix(parent_name: str, chain_length: int, *, attach_locant) -> str:
    """Convert a parent compound name to substituent prefix form.

    Per IUPAC P-29.2 (BB:15811), the parent compound name is transformed into
    a substituent prefix by:
    1. Removing the suffix (e.g., -oic acid, -ol, -one)
    2. Converting the suffix to its prefix form (e.g., -ol -> hydroxy)
    3. Adding the prefix at the correct locant
    4. Appending -yl at the free-valence position

    ⚠ **This converter may only emit locants it can justify.** (v29 residue Task A.)

    It is handed a *name string* and a *carbon count*, and nothing else. The
    string was produced by naming the fragment as a free molecule after capping
    its free valence with H, so every locant inside it belongs to the CAPPED
    molecule's numbering -- which was chosen to favour that molecule's own
    principal characteristic group. P-46.1 criterion (h) / **P-46.1.8** require
    the opposite for a substituent group: *"The principal substituent chain has
    the lowest locants for free valences of any kind."*

    The two numberings genuinely disagree. Measured witness (R8.2), fragment
    ``-C(CH3)(C2H5)-(CH2)8-CH(NH2)-CH(CH3)2``::

        capped + named as a molecule : 2,12-dimethyltetradecan-3-amine
        string-surgered to a prefix  : 3-amino-2,12-dimethyltetradecyl
        numbered from the free valence: 12-amino-3,13-dimethyltetradecan-3-yl

    The chain is numbered from opposite ends, so the amino locant and both
    methyl locants differ. **Splicing a free-valence locant onto the borrowed
    stem therefore cannot repair these branches** -- it would produce
    ``3-amino-2,12-dimethyltetradecan-3-yl``, one name in two numberings. The
    numbering has to be recomputed from the structure, which only a caller
    holding the molecule can do (see ``_located_acyclic_alkyl_name``).

    The count-derived branches fail the same way for a second reason. A whole
    fragment carbon COUNT is not a proof of the fragment's shape: for the
    branched acyl ``-C(=O)CH(CH3)2`` the count is 4 while the principal chain is
    3, so ``f"{chain_length}-oxo"`` spliced locant **4** onto a three-carbon
    ``propyl`` stem (``4-oxo-2-methylpropyl``).

    So both families now DECLINE (return ``None``) rather than fabricate. A
    one-position stem is the exception that needs no proof: **P-14.3.4.6**
    (BB:3031) *"All locants are omitted for parent compounds when all
    substitutable hydrogen atoms have the same locant"* -- ``carbamoylmethyl``,
    never ``1-carbamoylmethyl``.

    Args:
        parent_name: Parent compound IUPAC name (e.g., "propan-2-ol").
        chain_length: Number of carbons in the substituent chain.
        attach_locant: Locant of the free-valence atom, or
            ``ATTACH_LOCANT_UNKNOWN`` when the caller cannot prove one.
            **Required** -- "make a prefix unrenderable without its locant"
            (`:345``). It used to default to ``1`` and was read by
            nothing, so all six call sites silently omitted it.

    Returns:
        Prefix-form name (e.g., "hydroxymethyl"), ``""``/``None`` when this
        converter cannot express the fragment. Returns the raw name WITHOUT
        enclosing marks.
    """
    # Read the parameter that used to be dead. A proven integer still cannot
    # rescue the foreign-numbering branches below (see the docstring), but it is
    # recorded here so the decline is attributable and so no future caller can
    # re-enter the fabrication path by simply forgetting the argument.
    _attach_proven = isinstance(attach_locant, int) and not isinstance(
        attach_locant, bool)

    def _decline_unjustifiable(kind: str, detail: str):
        """A locant this converter cannot justify -> fail closed (never fabricate)."""
        logger.debug(
            "TaskA fail-closed: %s locant is not derivable from (name=%r, "
            "chain_length=%d, attach_locant=%r); %s",
            kind, parent_name, chain_length, attach_locant, detail,
        )
        return None

    if not parent_name:
        return ""

    name = parent_name.strip()

    # ASML-17: Check retained name lookup FIRST (before regex cascade)
    name_lower = name.lower()
    if name_lower in _RETAINED_NAME_PREFIX:
        return _RETAINED_NAME_PREFIX[name_lower]

    # v29 P7 C3: fail closed on FUNCTIONAL parents that have no '-yl' form.
    # Hoisted ABOVE the suffix cascade on purpose — the '-ol' and '-amine'
    # branches match on a bare string ending and would otherwise capture these
    # (see _FUNCTIONAL_PARENT_NO_YL_FORM). Returning None makes the caller
    # abstain; every call site tolerates it (5 guard with `if prefix:`, and
    # name_substituent_fragment's _add_substituent_stereo returns None for a
    # None name), so the SELF-01 gate sees an abstention rather than an
    # OPSIN-unparseable fabrication.
    if name_lower in _NON_HYDRIDE_WHOLE_NAMES:
        logger.debug("C3 fail-closed: %r is not a parent hydride", name)
        return None
    for _ending, _bb_alternative in _FUNCTIONAL_PARENT_NO_YL_FORM:
        if name_lower.endswith(_ending):
            logger.debug(
                "C3 fail-closed: %r is a functional parent with no '-yl' form; "
                "the Blue Book uses %s", name, _bb_alternative,
            )
            return None

    # ---- Carboxylic acids: -oic acid / -anoic acid ----
    # e.g., "butanoic acid" -> "3-carboxypropyl"
    m_oic = re.match(r'^(.+?)(?:an)?oic acid$', name)
    if m_oic:
        stem = m_oic.group(1)
        # P-59.2.1.5 (W2E-P1FC Task 10) fail-closed guard: this chain converter
        # must NEVER flatten a RING acid ('benzoic acid', 'naphthoic acid') into
        # a carboxy-alkyl chain ('6-carboxyhexyl') — that describes a DIFFERENT
        # molecule (the historical phenyl->hexyl corruption). A systematic chain
        # acid is '{chainstem}anoic acid' whose bare stem IS a known chain prefix
        # (meth/eth/prop/...); a trivial ring acid ('benz', 'naphth') is not.
        # Decline (return "") when the stem is not a recognized chain prefix so
        # the caller falls through to the ring-substituent namer / fails closed.
        from ..data.chain_names import get_chain_prefix as _gcp
        _valid_chain_stems = {_gcp(i) for i in range(1, 31)}
        if stem.lower() not in _valid_chain_stems:
            return ""
        # Carboxy goes on the terminal carbon (chain_length for original chain)
        # The stem is shortened by one carbon (the COOH carbon becomes "carboxy")
        carboxy_locant = chain_length
        # Build: {locant}-carboxy{shortened_stem}yl
        # For butanoic acid (4C): carboxy at C4, stem = prop (3C), result = 3-carboxypropyl
        # Actually: butanoic acid -> carboxy replaces the acid, chain becomes 3C (propyl)
        # The parent chain of the substituent WITHOUT the carboxylic acid is chain_length - 1
        shortened_length = chain_length - 1
        if shortened_length == 1:
            # P-14.3.4.6 (BB:3031): a one-carbon stem has all its substitutable
            # hydrogens at the same locant, so NO locant is cited -> the count
            # cannot be wrong here. 'carboxymethyl', never '1-carboxymethyl'.
            return "carboxymethyl"
        if shortened_length >= 2:
            # The carboxy locant was read off the whole-fragment carbon COUNT,
            # which is not a proof of the fragment's shape (a branched fragment
            # has a shorter principal chain than its carbon count).
            return _decline_unjustifiable(
                "carboxy", "count-derived locant on a multi-position stem")
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
        # P-46.1.8: `locant` here is the HYDROXY position in the capped
        # molecule's numbering, which was chosen to give the OH the lowest
        # locant -- the opposite of what a substituent group requires. It
        # produced '1-hydroxypropyl' for -CH2CH2CH2OH (wrong end) and
        # '3-hydroxy(3S)-oct-1-enyl' for the prostaglandin side chain (no
        # free-valence locant at all).
        return _decline_unjustifiable(
            "hydroxy", "locant borrowed from the capped molecule's numbering")

    # ---- Multi-FG alcohol: -diol, -triol ----
    # e.g., "propane-1,2-diol" -> "1,2-dihydroxypropyl"
    # e.g., "ethane-1,2-diol" -> "1,2-dihydroxyethyl"
    # P-63.1.2 elides the multiplier-final 'a' before '-ol' (tetra+ol -> tetrol),
    # so match BOTH spellings (tetra? = tetr|tetra) and derive the hydroxy
    # multiplier from the locant COUNT — group(2) 'tetr' must NOT become the
    # wrong 'tetrhydroxy'. (v22 G2 follow-on: keeps this converter in sync with
    # the elision fix in naming_utils._join_multiplied_suffix.)
    # Every locant in these three multi-FG forms is read straight out of the
    # capped molecule's numbering (P-46.1.8 decline -- see the docstring). The
    # -diamine arm is the R4 witness: it kept the parent's N1/N2 italic locants
    # in a prefix scope that has no N1/N2, giving the OPSIN-unparseable
    # '1,2-diamino-N1-(2-aminoethyl)-N2-methylethyl'.
    m_diol = re.search(r'[,-](\d+(?:,\d+)*)-(?:di|tri|tetra?)ol$', name)
    if m_diol:
        return _decline_unjustifiable(
            "polyhydroxy", "locants borrowed from the capped molecule's numbering")

    # ---- Multi-FG ketone: -dione, -trione ----
    m_dione = re.search(r'[,-](\d+(?:,\d+)*)-([dt]i|tri|tetra)one$', name)
    if m_dione:
        return _decline_unjustifiable(
            "polyoxo", "locants borrowed from the capped molecule's numbering")

    # ---- Multi-FG amine: -diamine, -triamine ----
    m_diamine = re.search(r'[,-](\d+(?:,\d+)*)-([dt]i|tri|tetra)amine$', name)
    if m_diamine:
        return _decline_unjustifiable(
            "polyamino", "locants borrowed from the capped molecule's numbering")

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
        return _prefix_stem_yl("hydroxy", stem)

    # ---- Locanted ketone: -N-one ----
    # e.g., "butan-2-one" -> "2-oxobutyl"
    # Task L3-0 (v33 Phase 0): was anchored to a literal 'a'(+optional 'n')
    # immediately before the locant digit (`an?-(\d+)-one$`), which requires
    # the locant to sit right after a saturated '-an-' infix. An unsaturated
    # chain hides the locant behind '-en-'/'-yn-' instead (e.g.
    # "hept-2-en-4-one" -- the char before "-4-one" is the 'n' of 'en', not
    # of 'an'), so the narrow anchor missed it and execution fell through to
    # the "Unlocanted ketone" branch below, which silently spliced 'oxo' onto
    # the stem with NO locant at all -- dropping a real, load-bearing locant
    # rather than declining ('oxo-2-methylhept-2-en-4-yl'; OPSIN then places
    # the uncited oxo at the lowest available position, a different molecule).
    # Widened to match ANY locant immediately before '-one$', mirroring the
    # already-correct sibling "Locanted alcohol" pattern two blocks up
    # (`r'-(\d+)-ol$'`, documented there to match saturated/unsaturated/bare
    # forms alike).
    m_one = re.search(r'-(\d+)-one$', name)
    if m_one:
        return _decline_unjustifiable(
            "oxo", "locant borrowed from the capped molecule's numbering")

    # ---- Unlocanted ketone: ends in -one ----
    if name.endswith('one') and not name.endswith('none'):
        base = name[:-3]  # remove "one"
        if base.endswith('an'):
            stem = base[:-2]
        elif base.endswith('a'):
            stem = base[:-1]
        else:
            stem = base
        return _prefix_stem_yl("oxo", stem)

    # ---- Locanted amine: -N-amine ----
    # e.g., "propan-1-amine" -> "1-aminopropyl"
    # Task L3-0 (v33 Phase 0): same narrow-anchor blind spot as `m_one` above
    # (`an?-(\d+)-amine$` misses an unsaturated "-en-N-amine"/"-yn-N-amine"
    # chain, e.g. "hept-2-en-4-amine"), widened the same way.
    m_amine = re.search(r'-(\d+)-amine$', name)
    if m_amine:
        # The R8.2 witness: '2,12-dimethyltetradecan-3-amine' ->
        # '3-amino-2,12-dimethyltetradecyl'. Numbered from the free valence the
        # SAME fragment is '12-amino-3,13-dimethyltetradecan-3-yl' -- a different
        # locant for every prefix, because the chain runs the other way.
        return _decline_unjustifiable(
            "amino", "locant borrowed from the capped molecule's numbering")

    # ---- Unlocanted amine: ends in -amine / -anamine ----
    if name.endswith('amine'):
        base = name[:-5]  # remove "amine"
        if base.endswith('an'):
            stem = base[:-2]
        elif base.endswith('a'):
            stem = base[:-1]
        else:
            stem = base
        return _prefix_stem_yl("amino", stem)

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
        # The oxo locant used to be `chain_length` -- the whole fragment's carbon
        # COUNT -- on the reasoning that the -CHO carbon sits at the terminus
        # opposite the attachment. That holds only for an UNBRANCHED chain. A
        # count is not a proof of shape: '2-methylpropanal' counts 4 carbons but
        # its stem is the 3-carbon 'prop', so the converter spliced locant 4 onto
        # a three-position stem ('4-oxo-2-methylpropyl', R12.1). A one-carbon
        # stem needs no locant at all (P-14.3.4.6, BB:3031) and is kept.
        if stem and not _starts_with_locant(stem) and chain_length == 1:
            return _prefix_stem_yl("oxo", stem)
        return _decline_unjustifiable(
            "oxo", "count-derived aldehyde locant is not a proof of chain length")

    # ---- Ester: -oate suffix ---- (IUPAC P-65.6.3)
    # e.g., "propanoate" -> carboxy prefix form
    # When ester is not the principal group, the acid portion uses "carboxy"
    m_oate = re.search(r'(?:an)?oate$', name)
    if m_oate:
        shortened = chain_length - 1
        if shortened == 1:
            return "carboxymethyl"          # P-14.3.4.6: one position, no locant
        if shortened >= 2:
            return _decline_unjustifiable(
                "carboxy", "count-derived locant on a multi-position stem")
        return "carboxy"

    # ---- Amidine: -carboximidamide / -imidamide ---- (IUPAC P-66.4.1.3.1)
    # MUST precede the general -amide regex which would wrongly match "-imidamide"
    # and return "carbamoyl" (amide prefix) for an amidine group.
    # e.g., "methanimidamide" -> "carbamimidoyl" (terminal C1; chain_length-1 = 0)
    # e.g., "propanimidamide" -> "carbamimidoyl" with chain stem for longer chains
    if name.endswith('carboximidamide'):
        stem = name[:-15]  # remove "carboximidamide"
        if stem:
            return _prefix_stem_yl("carbamimidoyl", stem)
        return "carbamimidoyl"
    m_imidamide = re.search(r'(?:an)?imidamide$', name)
    if m_imidamide:
        shortened = chain_length - 1
        if shortened == 1:
            return "carbamimidoylmethyl"    # P-14.3.4.6: one position, no locant
        if shortened >= 2:
            return _decline_unjustifiable(
                "carbamimidoyl", "count-derived locant on a multi-position stem")
        return "carbamimidoyl"

    # ---- Amide: -carboxamide (most specific first) ---- (IUPAC P-66.1.1.4)
    # e.g., "benzcarboxamide" -> "carbamoyl" prefix
    if name.endswith('carboxamide'):
        stem = name[:-11]  # remove "carboxamide"
        if stem:
            return _prefix_stem_yl("carbamoyl", stem)
        return "carbamoyl"

    # ---- v33 Phase 6 E2c fail-closed guard, WRONG-MOLECULE RISK: a
    # SULFONAMIDE/SULFINAMIDE (or any other non-carboxamide '...amide'-suffix
    # functional class) must NEVER reach the generic '-amide' -> 'carbamoyl'
    # transform below. 'methanesulfonamide' ends in the literal substring
    # 'amide' exactly like 'ethanamide' does, but its amide nitrogen sits on
    # S(=O)(=O)-, not C(=O)- -- collapsing it to 'carbamoyl' silently swaps the
    # sulfonyl for a carbonyl, a DIFFERENT constitution (measured:
    # '-CH2-SO2-NHCH3' capped + named as the free molecule
    # 'N-methylmethanesulfonamide' -> the WRONG 'carbamoylmethyl'). The
    # dedicated 'sulfamoyl'/'(R-sulfamoyl)' conversion
    # (P-65.3.1 / P-66.1.1.4.2) is a STRUCTURAL primitive that runs upstream of
    # this string converter (`_name_polyfunctional_acyclic_substituent`'s Pass
    # 1f); anything reaching here for that class fails closed rather than
    # guess -- 0-wrong over breadth.
    if name_lower.endswith('sulfonamide') or name_lower.endswith('sulfinamide'):
        logger.debug(
            "E2c fail-closed: %r is a sulfonamide/sulfinamide, not a "
            "carboxamide -- the generic '-amide' transform must not fire",
            name,
        )
        return None

    # ---- Amide: general -amide suffix ---- (IUPAC P-66.1.1.4)
    # e.g., "propanamide" -> "2-carbamoylethyl", "acetamide" -> "carbamoylmethyl"
    m_amide = re.search(r'(?:an)?amide$', name)
    if m_amide:
        shortened = chain_length - 1
        if shortened == 1:
            return "carbamoylmethyl"        # P-14.3.4.6: one position, no locant
        if shortened >= 2:
            return _decline_unjustifiable(
                "carbamoyl", "count-derived locant on a multi-position stem")
        return "carbamoyl"

    # ---- Nitrile: -carbonitrile (most specific first) ---- (IUPAC P-66.1.4.1)
    # e.g., "benzonitrile" -> "cyanophenyl" (cyano + stem + yl)
    if name.endswith('carbonitrile'):
        stem = name[:-12]  # remove "carbonitrile"
        if stem:
            return _prefix_stem_yl("cyano", stem)
        return "cyano"

    # ---- Nitrile: general -nitrile suffix ---- (IUPAC P-66.1.4.1)
    # e.g., "propanenitrile" -> "2-cyanoethyl", "acetonitrile" -> "cyanomethyl"
    if name.endswith('nitrile') and not name.endswith('carbonitrile'):
        shortened = chain_length - 1
        if shortened == 1:
            return "cyanomethyl"            # P-14.3.4.6: one position, no locant
        if shortened >= 2:
            return _decline_unjustifiable(
                "cyano", "count-derived locant on a multi-position stem")
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

    # ---- Aniline family: the RETAINED prefix, not a '-yl' transform ----
    # P-62.2.1.1.1 (BB:26139): "The prefix name 'anilino' is retained as the
    # preferred prefix for C6H5-NH- with full substitution allowed." Aniline is a
    # carbocyclic amine, not a heterocycle, so the P-31.1.3 '-ine' -> '-inyl' rule
    # below must not reach it: it produced '4-methyl-N-methylanilinyl', a morpheme
    # that appears nowhere in the Blue Book. Must precede the '-ine' branch.
    if name.endswith('aniline'):
        from ..rules.ring_substituents import anilino_prefix_from_aniline_name
        return anilino_prefix_from_aniline_name(name)

    # ---- Heterocyclic -ine ending ---- (IUPAC P-31.1.3)
    # e.g., "pyridine" -> "pyridinyl", "piperidine" -> "piperidinyl"
    # Note: -ine must come BEFORE the generic -e fallback
    if name.endswith('ine'):
        return name[:-1] + "yl"  # pyridine -> pyridinyl

    # ---- RC-1 fail-closed guard (v26 BP-2) ----
    # A FUNCTIONAL-PARENT name that reaches the terminal fallbacks below has NO
    # valid '+yl' prefix — the correct prefix is a defined form (carboxy /
    # alkoxycarbonyl / isothiocyanato / oxo ...), never '«acid»yl' / '«formate»yl'
    # / '«enal»yl'. BB P-65.1.1 (acid-as-substituent is 'carboxy', never
    # '«acid»yl') and P-66 note (p) ('1-oxopropyl'-type acyl strings are not
    # preferred prefixes). The correct converters (retained-acyl, -oate->carboxy,
    # -amide, -nitrile, -al saturated aldehyde, -one, -ol, -amine) all run ABOVE
    # this point, so any residue reaching here is a class this string converter
    # cannot express: DECLINE (return None) so the caller fails closed to the
    # SELF-01 gate instead of fabricating an OPSIN-unparseable string. Every
    # call site tolerates None (5 guard `if prefix:`; and the terminal
    # `return _add_substituent_stereo(...)` in `name_substituent_fragment`
    # is safe because `_add_substituent_stereo` opens with
    # `if not sub_atoms or not name: return name`, so a None name comes back
    # None). Cited by SYMBOL, not by line: the two line numbers that used to
    # stand here (`:3604` and `line 3011`) were both stale — every insertion
    # above them moves them, and v29 P7 alone shifted this file three times.
    # Class-specific to functional residues — never touches
    # -ene/-yne/-ane/-ine/-yl.
    if (' ' in name             # v30 RISK 5 Class 3: a SPACE marks a functional-class
                                # multi-word name ('urea oxime', 'taxifoline acetate') --
                                # never a valid single substituent token; the '-e'->'-yl'
                                # fallback below would fabricate the unparseable 'urea oximyl'.
            or ' acid' in name
            or name.endswith('ate')      # functional-class / residual ester (formate, carbamate, ...)
            or name.endswith('urea')
            or name.endswith('al')):     # unconverted (unsaturated) aldehyde residue (prop-2-enal)
        return None

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


def _substituted_aryl_ring_name(mol, aryl_idx: int, central_c_idx: int) -> Optional[str]:
    """Name the aryl side of a benzylic ether as a ring-system substituent
    ('4-methoxyphenyl'), or None (fail closed) — w2f p1 Task 7 (P-29.6.1:
    ring substitution kills retained benzyloxy in PINs).

    BFS the aryl-side fragment from ``aryl_idx`` (never crossing the benzylic
    carbon) and delegate to rules.ring_substituents.name_ring_system_substituent
    (the ring-engine producer, which itself fails closed on anything it cannot
    fully describe). Guards: None / space-bearing output / bare 'phenyl'
    (a plain ring must have taken the retained-benzyloxy path upstream)."""
    frag = []
    seen = {central_c_idx}
    stack = [aryl_idx]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        frag.append(cur)
        for n in mol.GetAtomWithIdx(cur).GetNeighbors():
            if n.GetIdx() not in seen and n.GetAtomicNum() > 1:
                stack.append(n.GetIdx())
    if not frag:
        return None
    from ..rules.ring_substituents import name_ring_system_substituent
    name = name_ring_system_substituent(mol, sorted(frag), aryl_idx)
    if not name or ' ' in name or name == 'phenyl':
        return None
    return name


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
    if len(arom_nbrs) == 1:
        aryl_idx = arom_nbrs[0].GetIdx()
        if _is_plain_phenyl(mol, aryl_idx, central_c_idx):
            return "benzyloxy"  # P-29.6.1 retained preferred prefix (bare ring)
        # w2f p1 (P-29.6.1, BB 16274): ring substitution is not allowed on
        # retained benzyl(oxy) in PINs -> systematic '(4-methoxyphenyl)methoxy'
        # (P-35.4.1, BB 18112: '(4-chlorophenyl)methoxy (preferred prefix)').
        # Fail-closed: None when the decorated ring cannot be fully named
        # (the old unconditional 'benzyloxy' DROPPED the ring substituent —
        # a different molecule).
        ring_name = _substituted_aryl_ring_name(mol, aryl_idx, central_c_idx)
        if ring_name is None:
            return None
        return f"({ring_name})methoxy"
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
    # R3 follow-up: the fragment must also be ACYCLIC. `tert-butyl` is an
    # acyclic prefix (P-29.6.1, Blue Book 16196 / 16286), but the three tests
    # above are all satisfied by **1-methylcyclopropyl**, whose attachment
    # carbon has two RING neighbours plus a methyl -- 4 carbons, 3 carbon
    # neighbours, no heteroatom. `NC(=O)NC1(C)CC1` was therefore named
    # `N-tert-butylurea`: the cyclopropane ring opened into a chain, a
    # different molecule that happens to share the C4H9 formula, and SELF-01
    # was the only thing that caught it.
    #
    # Same class as the heteroatom hardening noted just above, and of
    # : a COUNT is not
    # a constitution. The ring test is structural (RDKit ring membership), not
    # another tally.
    if (carbon_count == 4 and len(c_neighbors_in_frag) == 3
            and all(mol.GetAtomWithIdx(i).GetSymbol() == 'C'
                    for i in sub_atoms)
            and not any(mol.GetAtomWithIdx(i).IsInRing() for i in sub_atoms)):
        # tert-butyl: C(CH3)3 -- 3 carbon branches at the attachment carbon.
        return "tert-butyl"

    return None


# ============================================================================
# Substituent CIP Stereo Descriptor
# ============================================================================


def _located_acyclic_alkyl_name(mol, sub_atoms, attach_idx, with_pos=False,
                                allow_functional=False):
    """Name an acyclic, all-carbon, saturated substituent by its OWN principal
    chain, numbered from the free valence (P-29.2 / P-46.1.8 / P-46.1.12).

    GENERAL structure-derived rule (DD5 RC-6 / SEN-04), no name table and NOT
    gated on a stereocentre. Returns ``(name, k)`` where ``name`` is the full
    located prefix and ``k`` is the free-valence locant, or ``None`` when the
    substituent is not an acyclic, all-carbon, saturated alkyl (caller then
    falls through to the recursive path).

    When ``with_pos`` is set the return is the 3-tuple ``(name, k, chain_pos)``
    where ``chain_pos`` maps each principal-chain atom index to its substituent
    locant (attach=1..n). This is the SANCTIONED numbering the name itself uses
    (P-46.1.8), so a stereodescriptor can be cited at a stereocentre's true
    locant even when that centre is not the attachment atom (P-91.3) -- without
    the forbidden re-derived BFS-from-attachment heuristic. Only the stereo
    adapter passes it; the two 2-tuple callers are unaffected.

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
    if allow_functional:
        # v33 re-rooted FG-capable substituent (P-46.1.12): the CHAIN is the
        # acyclic, single-bonded carbon skeleton through the free valence; EVERY
        # other atom -- ring atoms, chain-carbon =O (oxo), -O-R ethers/esters,
        # -N< amines/amides, -S-R thioethers, phospho subgraphs -- is permitted
        # and cited as a structurally-numbered detachable PREFIX (or recursed
        # branch) below, never a suffix. This reuses the SAME free-valence
        # numbering and branch machinery as the strict alkyl path, so a fragment
        # rooted at an arbitrary attachment atom is named as a proper substituent
        # (not string-surgered from a molecule name). 0-wrong is preserved by the
        # top-level SELF-01/OPSIN gate. Defer (return None) on a charged atom or a
        # C=C/C#C in the carbon skeleton (chain unsaturation not handled here yet).
        for idx in sub_set:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
                return None
            if atom.GetSymbol() != 'C' or ring_info.NumAtomRings(idx) > 0:
                continue
            cc_unsat = any(
                b.GetBondTypeAsDouble() > 1.0
                and b.GetOtherAtom(atom).GetSymbol() == 'C'
                and b.GetOtherAtom(atom).GetIdx() in sub_set
                for b in atom.GetBonds())
            if cc_unsat:
                return None
            carbon_set.add(idx)
        if attach_idx not in carbon_set:
            return None
        return _located_fg_assemble(mol, sub_atoms, attach_idx, carbon_set,
                                    ring_info, with_pos)
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
        elif sym == 'N':
            # PRIMARY amine only: a degree-1 -NH2 hanging off a fragment carbon
            # is a simple detachable prefix in exactly the way hydroxyl and the
            # halogens above already are (P-46.1.12), so the same chain
            # selection applies and 'amino' is the whole of its contribution.
            #
            # This is a PROOF of the admissible shape, not a deny-list: exactly
            # two hydrogens, degree 1, uncharged, no radical, carbon-bonded.
            # -NH-R (a secondary amine) fails it because 'amino' would silently
            # drop R; nitro, nitrile, imine and N-oxide fail on degree, H-count
            # or the bond-order check below. Everything unproven still declines
            # to the recursive path.
            in_frag = [n for n in atom.GetNeighbors() if n.GetIdx() in sub_set]
            if (atom.GetDegree() != 1 or atom.GetTotalNumHs() != 2
                    or atom.GetNumRadicalElectrons() != 0
                    or len(in_frag) != 1 or in_frag[0].GetSymbol() != 'C'):
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

    def _ret(nm, kk):
        # Attach ``chain_pos`` only for the stereo adapter (``with_pos``); the
        # two 2-tuple callers keep their ``(name, k)`` contract.
        return (nm, kk, chain_pos) if with_pos else (nm, kk)

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
                              'I': 'iodo', 'O': 'hydroxy', 'N': 'amino'}
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

    # Wave2 T5b (P-14.4(g), BB 3307 -- P-14.5.2 has NO lettered sub-items):
    # when the free valence sits at the exact chain
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
            return _ret(get_alkyl_name(chain_len), 1)
        except (ValueError, KeyError):
            return None
    try:
        stem = get_chain_prefix(chain_len)
    except (ValueError, KeyError):
        return None

    # P-14.3.4.5 (``:3007``), substituent enclosing-mark scope -- MEASURED-LIVE site
    # for ``1-chloro-2-(pentafluoroethyl)benzene`` (``:3023``). A validated call-spy
    # recorded this function as the sole productive namer of that fragment, reached
    # from ``substituent_enumerator.py:1513``; ``_name_saturated_substituted_chain``
    # is never even CALLED for it -- the two live sites sit on two entirely
    # different cascades, which is why both have to be wired.
    # Deny-by-default: the licence is consulted, and only a positive answer replaces
    # the located join. It fires only for k == 1 (see the helper's derivation), so
    # the ``an-k-yl`` branch below is unaffected by construction.
    _l5 = _l5_substituent_prefix(mol, sub_atoms, chain, k, branch_groups) \
        if branch_groups else None

    from .composer import _format_prefix_groups
    prefix = _l5 or (_format_prefix_groups(branch_groups) if branch_groups else "")

    if k == 1:
        try:
            base = get_alkyl_name(chain_len)  # 'butyl', 'pentyl', ...
        except (ValueError, KeyError):
            base = f"{stem}yl"
        return _ret(f"{prefix}{base}", 1)
    return _ret(f"{prefix}{stem}an-{k}-yl", k)


def _located_fg_assemble(mol, sub_atoms, attach_idx, carbon_set, ring_info, with_pos):
    """v33 FG-capable located substituent assembly (see ``_located_acyclic_alkyl_name``
    ``allow_functional``). The parent is the longest ACYCLIC carbon chain through the
    free valence (P-29.2, numbered from the free valence P-46.1.8); every off-chain
    atom -- a chain-carbon ``=O`` (oxo), ``-OH`` (hydroxy), ``-NH2`` (amino), and any
    ``-O-R`` / ``-N<`` / ``-S-R`` / ring / phospho branch -- is cited as a
    structurally-numbered detachable PREFIX via the shared substituent namer
    (P-46.1.12), recursed. Returns ``(name, k[, chain_pos])`` or None (missing-beats-
    wrong: any unnameable branch declines the whole form). Stereo descriptors are
    added by the caller from ``chain_pos``.
    """
    sub_set = set(sub_atoms)

    def _longest_arm(start, came_from):
        best_child = {}
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
    if len(arms) >= 2:
        chain = list(reversed(arms[0])) + [attach_idx] + arms[1]
    elif len(arms) == 1:
        chain = [attach_idx] + arms[0]
    else:
        chain = [attach_idx]
    # free valence gets the lowest locant (P-46.1.8)
    if (len(chain) - chain.index(attach_idx)) < (chain.index(attach_idx) + 1):
        chain = list(reversed(chain))
    chain_len = len(chain)
    if chain_len < 1:
        return None
    k = chain.index(attach_idx) + 1
    chain_set_c = set(chain)
    chain_pos = {a: i + 1 for i, a in enumerate(chain)}

    from collections import defaultdict
    branch_groups: dict = defaultdict(list)
    for c in chain:
        for nbr in mol.GetAtomWithIdx(c).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx in chain_set_c or nidx not in sub_set:
                continue
            bond = mol.GetBondBetweenAtoms(c, nidx)
            sym = nbr.GetSymbol()
            # chain-carbon =O -> oxo (P-64.2.1 as a prefix); =N -> imino
            if bond.GetBondTypeAsDouble() == 2.0 and nbr.GetDegree() == 1:
                if sym == 'O':
                    branch_groups['oxo'].append(chain_pos[c]); continue
                if sym == 'N':
                    branch_groups['imino'].append(chain_pos[c]); continue
            # collect the whole off-chain branch
            frag = []
            seen = set(chain_set_c)
            stk = [nidx]
            while stk:
                cur = stk.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                frag.append(cur)
                for nn in mol.GetAtomWithIdx(cur).GetNeighbors():
                    if nn.GetIdx() in sub_set and nn.GetIdx() not in seen:
                        stk.append(nn.GetIdx())
            if len(frag) == 1 and sym in ('F', 'Cl', 'Br', 'I'):
                branch_groups[{'F': 'fluoro', 'Cl': 'chloro', 'Br': 'bromo',
                               'I': 'iodo'}[sym]].append(chain_pos[c]); continue
            if len(frag) == 1 and sym == 'O' and nbr.GetTotalNumHs() >= 1:
                branch_groups['hydroxy'].append(chain_pos[c]); continue
            if len(frag) == 1 and sym == 'N' and nbr.GetTotalNumHs() == 2:
                branch_groups['amino'].append(chain_pos[c]); continue
            try:
                # recurse through the central substituent entry so the branch
                # reaches the same tiers (rings, phospho Step 0, and Tier FG for a
                # nested FG chain), rooted at the branch's own attachment atom.
                bname = name_substituent_fragment(mol, sorted(frag), nidx, list(chain))
            except Exception as exc:  # noqa: BLE001
                logger.debug("located-fg branch naming failed: %s", exc)
                return None
            if not bname or bname == 'substituent':
                return None
            branch_groups[bname].append(chain_pos[c])

    from .composer import _format_prefix_groups
    prefix = _format_prefix_groups(branch_groups) if branch_groups else ""
    try:
        if k == 1:
            base = get_alkyl_name(chain_len)
            name = f"{prefix}{base}"
        else:
            from ..data.chain_names import get_chain_prefix
            stem = get_chain_prefix(chain_len)
            name = f"{prefix}{stem}an-{k}-yl"
    except (ValueError, KeyError):
        return None
    # v33: cite this chain's OWN stereodescriptors from the SAME free-valence
    # numbering the name was built with (`chain_pos`), exactly as the sibling
    # located derivers do (`_located_acyclic_alkyl_name` at the alkenyl path,
    # BlueBookV2 P-91.2.1.2.1 / P-46.3). The docstring formerly said "stereo is
    # added by the caller", but the Tier-FG caller's `_add_substituent_stereo`
    # does not fire when this chain is reached as a deep `-O-<chain>` (butoxy)
    # substituent -- so an on-chain stereocentre (e.g. the acyl-CoA pantetheine
    # 3-hydroxy centre) was silently dropped -> the whole molecule abstained on
    # a stereo mismatch. `_located_stereo_block` cites ONLY atoms in `chain_pos`
    # and ONLY DEFINED (`_CIPCode`) centres, so off-chain-branch stereocentres
    # (expressed by their own nested prefix) are never double-counted, and an
    # undefined centre is never fabricated. The caller's later
    # `_add_substituent_stereo` is idempotent here (its
    # `count_expressed_stereo_descriptors` guard suppresses a second block).
    stereo = _located_stereo_block(mol, chain_pos)
    if stereo:
        name = f"{stereo}{name}"
    return (name, k, chain_pos) if with_pos else (name, k)


def _fg_enclose(tok: str) -> str:
    """Enclose a composed substituent token when it carries a locant or its own
    marks (P-16.3.3); a simple one-word token stays bare."""
    if not tok:
        return tok
    if tok[0] == '(' or tok[0] == '[':
        return tok
    if any(ch.isdigit() for ch in tok) or '-' in tok or ' ' in tok:
        return f"({tok})"
    return tok


def _fg_branch_atoms(mol, start, block, sub_set):
    br = []
    seen = {block}
    stk = [start]
    while stk:
        cur = stk.pop()
        if cur in seen or cur not in sub_set:
            continue
        seen.add(cur)
        br.append(cur)
        for nn in mol.GetAtomWithIdx(cur).GetNeighbors():
            if nn.GetIdx() != block and nn.GetIdx() in sub_set and nn.GetIdx() not in seen:
                stk.append(nn.GetIdx())
    return br


def _located_fg_hetero_root(mol, sub_atoms, attach_idx):
    """v33: FG-capable substituent rooted at a HETERO atom -- ``-O-R`` -> R-oxy,
    ``-S-R`` -> (R)sulfanyl, ``-N(<)`` -> (R)amino -- with R recursed through the
    central substituent entry (so a nested FG chain reaches Tier FG). Returns the
    prefix string or None (fail-closed)."""
    sub_set = set(sub_atoms)
    a = mol.GetAtomWithIdx(attach_idx)
    if a.GetFormalCharge() != 0 or a.GetNumRadicalElectrons() != 0:
        return None
    sym = a.GetSymbol()
    heavy = [n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() in sub_set]

    def _name_r(r_root):
        return name_substituent_fragment(
            mol, sorted(_fg_branch_atoms(mol, r_root, attach_idx, sub_set)),
            r_root, [attach_idx])

    if sym == 'O':
        if not heavy:
            return 'hydroxy'
        if len(heavy) != 1:
            return None
        r = _name_r(heavy[0])
        if not r:
            return None
        from .substituent_enumerator import alkoxy_prefix_from_substituent
        return alkoxy_prefix_from_substituent(r) or (f"{_fg_enclose(r)}oxy")
    if sym == 'S':
        if not heavy:
            return 'sulfanyl'
        if len(heavy) != 1:
            return None
        r = _name_r(heavy[0])
        if not r:
            return None
        return f"{_fg_enclose(r)}sulfanyl"
    if sym == 'N':
        subs = []
        for h in heavy:
            r = _name_r(h)
            if not r:
                return None
            subs.append(r)
        if not subs:
            return 'amino'
        from .naming_utils import alpha_sort_key
        inner = ''.join(_fg_enclose(s) for s in sorted(subs, key=alpha_sort_key))
        return f"{inner}amino"
    return None


def located_map_completes_substituent_stereo(mol, sub_atoms, name, pos) -> bool:
    """Would citing descriptors for exactly the centres ``pos`` covers make
    ``name`` express EVERY defined stereo element of ``sub_atoms``?

    The admission test for a producer-supplied ``located`` map (see
    ``_add_substituent_stereo``'s ``located`` argument). A producer that owns a
    CHAIN numbering can only cite its own chain's centres; a ring-yl or
    heteroatom-branch centre it cannot reach must ALREADY be spelled inside
    ``name``. When that is not the case, citing the reachable subset would ship a
    PARTIALLY stereo-specified prefix -- a name that claims one configuration and
    leaves the rest silent. Under D-09 ("missing beats wrong") and the same
    all-or-nothing principle as ``general_engine_stereo_complete``, that must
    fail CLOSED: the pre-existing descriptor-less name is emitted instead, and no
    partial configuration is ever asserted.

    PRECONDITION (caller's responsibility -- the count identity cannot see a
    violation): the centres reachable through ``pos`` must be DISJOINT from the
    centres already expressed inside ``name``. Every current caller
    (``_compound_ring_on_chain_substituent`` via Tier 1.95) satisfies this by
    construction -- ``pos`` holds only carrier-CHAIN carbons while ``name``
    expresses only the nested RING-yl centres, two disjoint sets. A future caller
    passing an OVERLAPPING map would get a false ADMIT (one centre double-cited,
    another left silent), so it must re-establish the disjointness first.

    Counts DEFINED elements only: ``_CIPCode`` on an atom, or on a bond with both
    ends inside the fragment. NOTE this atoms-AND-bonds population is WIDER than
    the atoms-only ``stereo_atoms`` that ``_add_substituent_stereo`` cites over; a
    defined stereo BOND therefore pushes the sum away from equality and DECLINES
    (safe direction -- carriers are all-single by the producer's own guard, so no
    such bond reaches here today). Carbohydrate ``alpha-D-`` notation is NOT
    counted as expressed by ``count_expressed_stereo_descriptors``, so a
    glycosyl-bearing fragment under-counts and therefore declines -- safe again.
    """
    from ..rules.stereochemistry import count_expressed_stereo_descriptors
    sub_set = set(sub_atoms)
    defined = sum(1 for i in sub_set
                  if mol.GetAtomWithIdx(i).HasProp('_CIPCode'))
    for b in mol.GetBonds():
        if (b.GetBeginAtomIdx() in sub_set and b.GetEndAtomIdx() in sub_set
                and b.HasProp('_CIPCode')):
            defined += 1
    reachable = sum(1 for i in sub_set
                    if i in pos and mol.GetAtomWithIdx(i).HasProp('_CIPCode'))
    return count_expressed_stereo_descriptors(name) + reachable == defined


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

    Worked examples (BlueBook P-29.6.2.3): ``[C@@H](C)CC`` -> ("butan-2-yl", 2,
    {...}), ``[C@@H](C)CCC`` -> ("pentan-2-yl", 2, {...}). Returns the 3-tuple
    ``(name, k, chain_pos)`` so the emitter can cite a stereodescriptor at its
    true substituent locant even off the attachment atom (P-91.3).
    """
    return _located_acyclic_alkyl_name(
        mol, sub_atoms, attach_idx, with_pos=True)


def _add_substituent_stereo(mol, sub_atoms, name, attach_idx=None, located=None):
    """Add CIP stereodescriptors to a substituent name if stereocenters exist.

    When a substituent contains one or more stereocenters with defined CIP
    labels (R/S), the descriptor is prepended: e.g. a stereogenic secondary
    acyclic alkyl becomes "(2S)-butan-2-yl"; a parent-hydride-style substituent
    with a unique stereo position becomes "(R)-name".

    For a single stereocenter the format is "(R)-name" or "({k}R)-name".
    For multiple stereocenters the format uses locants threaded from the
    substituent's own numbering (any stereocentre this numbering does not
    reach is presumed already expressed by a nested recursively-named
    branch — see the ``located`` bugfix note below — and is left alone).

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
        located: An optional pre-computed ``(pin_form, k, pos)`` triple — the
            same contract ``_acyclic_alkyl_located_stereo_name`` returns — to
            use IN PLACE OF re-deriving one from ``attach_idx``. v33 bugfix:
            the Tier-FG caller (``allow_functional=True``) already builds
            exactly this triple to produce ``name`` itself; passing it here
            guarantees ``pin_form`` stays byte-identical to ``name`` (no risk
            of a second, differently-tie-broken recomputation silently
            replacing a correct name) while still exposing the FG chain's own
            ``pos`` map, which the plain ``attach_idx``-only path below cannot
            reach (it calls the STRICT, non-FG deriver only — the FG shape
            declines fail-closed to plain ``None``, which is exactly how a
            chain stereocentre in an FG-shaped fragment, e.g. the pantetheine
            3-hydroxy centre of an acyl-CoA thioester substituent, used to be
            silently dropped instead of just missing-a-locant). Every other
            caller passes nothing and keeps the prior strict-only behaviour
            unchanged.
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

    # v27 Phase S Task 3: double-apply guard for NESTED descriptor blocks.
    # The _stereo_route re.match guard only catches a LEADING "(...)" block, so
    # a substituent whose stereocentre is already expressed inside a nested
    # sub-substituent block (e.g. "2-[(1R)-1-hydroxyethyl]cyclohexyl") would
    # otherwise get a SPURIOUS bare "(R)-" prepended here — double-counting the
    # SAME centre and shipping a wrong/over-specified stereoisomer. If the name
    # already carries at least as many descriptor tokens as the fragment has
    # defined stereo elements, the stereo is already expressed -> leave it.
    from ..rules.stereochemistry import count_expressed_stereo_descriptors
    if count_expressed_stereo_descriptors(name) >= len(stereo_atoms):
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
        _loc = located if located is not None \
            else _acyclic_alkyl_located_stereo_name(mol, sub_atoms, attach_idx)
        if _loc is not None:
            # `_located_acyclic_alkyl_name` gives the substituent's OWN principal-
            # chain numbering (attach=1..n). Cite the descriptor at the
            # stereocentre's TRUE locant in that numbering -- P-91.3
            # (`BlueBookV2.md:44686`, "## **P-91.3** NAMING OF STEREOISOMERS":
            # a substituent-group stereodescriptor is "preceded by a numerical or
            # letter locant to describe the position of the stereogenic unit when
            # such locants are present"). This covers the stereocentre-at-
            # attachment case (`butan-2-yl` -> `(2S)-butan-2-yl`, loc == k) AND
            # the off-attachment case the old `s_idx == attach_idx` gate dropped
            # (`3-hydroxybutyl` -> `(3S)-3-hydroxybutyl`). The locant comes from
            # the SANCTIONED structure-derived chain, never a re-derived BFS.
            pin_form, _k, pos = _loc
            loc = pos.get(s_idx)
            if loc is not None:
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
    # v33 Phase 0 L3-2a: for the ACYCLIC-ALKYL shape specifically, that
    # recursion-contract change already exists. `_acyclic_alkyl_located_stereo_name`
    # -- the SAME deriver the single-centre branch above trusts (it returns
    # `pin_form` and ships it in place of the caller's `name`) -- exposes the
    # substituent's own chain-position map (`pos`, P-29.2/P-46.1.8/.12 numbering)
    # via `with_pos=True`. That map IS "the same numbering the substituent name
    # used" for this shape: `_located_acyclic_alkyl_name` is the deriver of
    # `name` itself here (a plain or hydroxy/halogen/amino-decorated acyclic
    # alkyl chain, e.g. '1,2,3-trihydroxypropyl' or the branched
    # '5-(propan-2-yl)heptan-2-yl'), so its `pos` is self-consistently the
    # numbering the emitted text already carries -- not a re-derived guess.
    # It declines (`None`) for every other shape (ring atoms, ethers, charged
    # atoms, non-single bonds), so the D-09 "missing beats wrong" fallback
    # below is unchanged for those.
    #
    # v33 bugfix (acyl-CoA pantetheine 3-hydroxy centre): a stereocentre this
    # deriver's OWN numbering does not cover is NOT necessarily unfindable —
    # it is typically one already expressed inside a nested recursively-named
    # branch prefix (a branch atom is never itself a member of `pos`, which
    # maps ONLY this level's own chain; branches are named+stereo-decorated by
    # their OWN independent `_add_substituent_stereo` call before this outer
    # one runs). Requiring ALL stereocentres in the whole fragment subtree to
    # be in `pos` before emitting ANY of them was therefore wrong: it silently
    # DROPPED an own-chain descriptor whenever the fragment ALSO contained an
    # already-resolved nested one (e.g. a Tier-FG '...-4-oxobutyl' chain whose
    # C3 stereocentre is on-chain, alongside a nested acyl-sulfanyl branch
    # whose OWN stereocentre was already baked into `name`). Emit locants for
    # whichever stereocentres `pos` DOES cover; the rest are presumed already
    # expressed and are left untouched -- never fabricated, never double-counted.
    _loc = located if located is not None \
        else _acyclic_alkyl_located_stereo_name(mol, sub_atoms, attach_idx)
    if _loc is not None:
        pin_form, _k, pos = _loc
        own_stereo = [(idx, cip) for idx, cip in stereo_atoms if idx in pos]
        if own_stereo:
            from ..rules.stereochemistry import format_stereodescriptor_string
            descr = sorted((pos[idx], cip) for idx, cip in own_stereo)
            block = format_stereodescriptor_string(descr)
            if block:
                return f"{block}{pin_form}"

    # Per D-09 (missing beats wrong): when the substituent's own numbering
    # cannot be threaded (or covers none of this fragment's stereocentres),
    # emit NO multi-centre stereo block here — return the name unchanged. A
    # multi-centre stereo descriptor placed with fabricated locants is worse
    # than none (it would round-trip to the WRONG diastereomer). The
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
    from ..rules.lambda_convention import LAMBDA
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
        core = f"(oxo-{LAMBDA}5-azanyl)"
    elif len(alkyl_names) == 2 and alkyl_names[0] == alkyl_names[1]:
        mp = get_multiplier_prefix(2, alkyl_names[0])
        core = f"[{mp}{alkyl_names[0]}(oxo)-{LAMBDA}5-azanyl]"
    elif len(alkyl_names) == 1:
        core = f"[{alkyl_names[0]}(oxo)-{LAMBDA}5-azanyl]"
    else:
        return None
    chain_len = len(chain)
    alkyl = get_alkyl_name(chain_len)  # 'methyl', 'ethyl', ...
    if alkyl is None:
        return None
    if chain_len == 1:
        return f"{core}{alkyl}"
    return f"{chain_len}-{core}{alkyl}"


def fragment_is_linear_terminal_alkyl(mol, sub_atoms, attach_idx: int) -> bool:
    """Is ``get_alkyl_name(carbon_count)`` an HONEST name for this fragment?

    ``get_alkyl_name(n)`` can only ever spell an unbranched saturated acyclic
    chain attached at a terminus ('propyl'). The two carbon-count fallbacks below
    used to call it for ANY fragment whose carbon count was non-zero, so a
    fragment they could not name recursively was renamed by counting its carbons
    and DISCARDING everything else: ``CCS[Zn]SCC`` came out ``'ethylethane'`` and
    ``CCO[Zn]OCC`` came out ``'oxylethane'``. A name must never claim atoms it
    dropped, so the fallback is now gated on the fragment actually being the one
    shape the function can spell.

    Requires: every atom carbon, none in a ring, every internal bond single, the
    induced subgraph a simple path, and ``attach_idx`` one of its two ends.
    """
    frag = set(sub_atoms)
    if attach_idx not in frag:
        return False
    for idx in frag:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C' or atom.IsInRing():
            return False
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return False
        degree = 0
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() not in frag:
                continue
            degree += 1
            bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
            if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
                return False
        limit = 1 if idx == attach_idx else 2
        if degree > limit:
            return False
    return True


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

    # ---- P-29.2 free-valence gate (the SAME gate name_substituent carries) ----
    # This is the project's OTHER general substituent chokepoint: it takes
    # attach_idx but every route below builds its token from the fragment
    # ALONE, so none of them can see whether the free valence is single
    # (-yl), double (-ylidene) or triple (-ylidyne). It reached the fused-ring
    # detectors, which is how 'C=C1Cc2ccccc2C1' was named '2-methylindane' --
    # a different molecule -- long after the sibling chokepoint was fixed.
    #
    # Same shared primitive, same three-way contract: a no-op for single bonds
    # (so every existing name stays byte-identical), the constructed prefix for
    # a double/triple carbon attachment, and fail closed rather than hand back
    # a -yl token that names something else.
    from .substituent_enumerator import (
        carbon_free_valence_prefix, _gate_is_reentrant)

    # When the gate itself is the caller it wants this function's ordinary
    # single-valence reading, which it will then give the right morpheme.
    # Consulting the gate again here is what would not terminate.
    if not _gate_is_reentrant():
        _fv = carbon_free_valence_prefix(mol, sub_atoms, attach_idx)
        if _fv.prefix is not None:
            return _fv.prefix
        if _fv.must_fail_closed:
            logger.debug("P-29.2 gate (fragment namer): %s", _fv.basis)
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
            # v33 (P-67.2.6): an -O-P(=O)(…)… phosphoanhydride subgraph as a
            # substituent. OST already names a terminal -O-P(=O)(OH)2 as
            # 'phosphonooxy' via FG perception, but a P-O-P bridge/chain (di/tri…
            # phosphate ester, as in acyl-CoA) matched no FG and fell through to a
            # generic mis-expression -> 'unknown'. Route it to the recursive
            # phosphoryl-oxy namer (best-effort systematic form; the top-level
            # SELF-01/OPSIN gate keeps 0-wrong).
            if len(_inner_nbrs) == 1 and mol.GetAtomWithIdx(
                    _inner_nbrs[0]).GetSymbol() == 'P':
                _parent_nbrs = [n.GetIdx() for n in _a.GetNeighbors()
                                if n.GetIdx() not in set(sub_atoms)]
                if len(_parent_nbrs) == 1:
                    from ..rules.phosphorus import (
                        name_phosphoxane_oxy_substituent,
                        name_phosphoanhydride_oxy_substituent)
                    # PIN first (P-67.2.6 method 2, the diphosphoxane skeletal
                    # parent); fall back to the valid method-1 recursive-phosphoryl
                    # form if the chain is not a clean phosphoxane (e.g. a single P
                    # -> phosphonooxy via method-1).
                    _ph = name_phosphoxane_oxy_substituent(
                        mol, attach_idx, _parent_nbrs[0])
                    if _ph is None:
                        _ph = name_phosphoanhydride_oxy_substituent(
                            mol, attach_idx, _parent_nbrs[0])
                    if _ph is not None:
                        return _ph
                return None
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
                # P-63.2.2 morphology via the BB-verbatim composed_alkoxy_prefix:
                # 'methoxymethyl' -> 'methoxymethoxy' (contract) but a ring/
                # locant-bearing '-yl' keeps its marks ('oxan-2-yl' ->
                # '(oxan-2-yl)oxy', NOT 'oxan-2-oxy') (F-spell-oxy).
                from .substituent_enumerator import alkoxy_prefix_from_substituent
                if _inner.endswith('yl'):
                    _stem = alkoxy_prefix_from_substituent(_inner)
                else:
                    _stem = f'({_inner})oxy'
                if _stem is None:
                    return None
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

    # Step 1b (BBR-PERC, 169.7; widened v33 Phase 6 E2a to include plain 'S'):
    # chalcogen-ether substituent -S-R / -Se-R / -Te-R → (alkyl)sulfanyl /
    # (alkyl)selanyl / (alkyl)tellanyl (P-63.2.5 / P-63.6). Without this,
    # Step 4's recursive path names it as the parent hydride "methanethiol" /
    # "methaneselenol" → "methanethiolyl" / "methaneselenolyl", which OPSIN
    # cannot parse (so the validity gate then suppresses the whole name to
    # "unknown") -- the plain-S case was already caught cleanly by the C3
    # `_FUNCTIONAL_PARENT_NO_YL_FORM` fail-closed guard in `parent_to_prefix`
    # rather than emitting that garbage, but it built nothing either.
    #
    # 'S' was excluded from this branch historically; there is no evidence it
    # was excluded for a STRUCTURAL reason (the builder below is explicitly
    # documented as "chalcogen-agnostic", already defaults its own `suffix`
    # param to `"sulfanyl"`, and already recurses the substituent side through
    # THIS SAME function -- `get_sulfanyl_prefix` calls
    # `name_substituent_fragment` on the arm, so a RING-bearing arm (DROP-24,
    # e.g. `-S-c1ccc(O)cc1O` -> '(2,5-dihydroxyphenyl)sulfanyl') is named by
    # the SAME trustworthy ring chokepoint (Step 1c) this function already
    # runs for a standalone ring fragment -- "delegate the ring-side to the
    # ring namer while keeping the chain-side attachment"). The chalcogen-
    # agnostic prefix builder walks the C neighbours of the attach chalcogen;
    # the substituent C (not on the parent chain) is named via the shared
    # substituent namer, whether it is a plain alkyl, a ring, or a compound
    # (methoxymethyl-style) arm.
    _attach = mol.GetAtomWithIdx(attach_idx)
    if _attach.GetSymbol() in ('S', 'Se', 'Te'):
        from .substituent_prefix_forms import get_sulfanyl_prefix
        _suffix = {'S': 'sulfanyl', 'Se': 'selanyl',
                   'Te': 'tellanyl'}[_attach.GetSymbol()]
        _nbrs = [n.GetIdx() for n in _attach.GetNeighbors()]
        if len(_nbrs) >= 2:
            # The R side is UNAMBIGUOUS from this function's own contract: it
            # is whichever neighbour of the chalcogen lies WITHIN `sub_atoms`
            # (this fragment's own boundary); the other neighbour is, by
            # construction, the parent-side attachment. Some callers pass an
            # EMPTY `parent_chain` at this tier (`_name_substituent_cascade`'s
            # Tier 4, `substituent_enumerator.py:2377-2379`), which left
            # `get_sulfanyl_prefix`'s internal "neither side is on the chain"
            # tie-break (smaller-fragment heuristic) to guess -- and it guessed
            # WRONG for a same-atom-count pair: -S-CH2-C6H5 vs the tolyl parent
            # (7 heavy atoms each side) returned '(4-methylphenyl)sulfanyl'
            # instead of 'benzylsulfanyl', a DIFFERENT molecule. Folding the
            # fragment boundary into the chain hint makes exactly one side
            # "on chain" by construction, so the ambiguous size tie-break is
            # never consulted.
            _sub_set_chal = set(sub_atoms)
            _chain_hint = set(parent_chain) if parent_chain else set()
            _chain_hint |= {n for n in _nbrs if n not in _sub_set_chal}
            _chal = get_sulfanyl_prefix(
                mol, (attach_idx, _nbrs[0], _nbrs[1]), list(_chain_hint),
                suffix=_suffix,
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

    # Step 1e (v26 BP-2 RC-2a, P-66.5.1.2 / BB 1710): the terminal pseudohalide
    # groups -N=C=O, -N=C=S, -N#C and -S-C#N are ALWAYS cited as substituent
    # prefixes (isocyanato / isothiocyanato / isocyano / thiocyanato) in PINs
    # ("added to the list of characteristic groups that are always cited as
    # prefixes ... in preferred IUPAC names"). The generic Step-3..5 recursion
    # names them as a free acid ('isothiocyanic acid') then fabricates
    # 'isothiocyanic acidyl' (OPSIN-unparseable -> SELF-01 -> unknown). Detect the
    # group structurally and return the authoritative prefix from seniority.
    # Only fires when the fragment is EXACTLY the pseudohalide group anchored at
    # attach_idx (any extra decoration falls through -> fail closed).
    _sub_set = set(sub_atoms)
    if 2 <= len(_sub_set) <= 3:
        _PSEUDOHALIDES = (
            ('isothiocyanate', '[NX2]=[CX2]=[SX1]'),
            ('isocyanate', '[NX2]=[CX2]=[OX1]'),
            ('isocyanide', '[NX2]#[CX1]'),
            ('thiocyanate', '[SX2][CX2]#[NX1]'),
        )
        from ..rules.seniority import get_prefix as _pseudo_get_prefix
        for _fg, _sm in _PSEUDOHALIDES:
            _pat = Chem.MolFromSmarts(_sm)
            if _pat is None:
                continue
            for _m in mol.GetSubstructMatches(_pat):
                if set(_m) == _sub_set and _m[0] == attach_idx:
                    _pfx = _pseudo_get_prefix(_fg)
                    if _pfx:
                        return _pfx

    # Step 1e2 (v33 Phase 6 E2e, P-66.5): a bare terminal NITROSO group -N=O is
    # the retained substituent prefix 'nitroso' (P-61.5, e.g. the amidine
    # N-substituent 'N-nitrosocarbamimidoyl', CHEBI:138933's
    # N(5)-(N-nitrosocarbamimidoyl)-L-ornithine). Without this, Step 3-4's
    # recursive path names the 2-atom fragment as its own isolated molecule
    # and fabricates an OPSIN-unparseable token (SELF-01-suppressed, safe but
    # incomplete). Only fires when the fragment is EXACTLY {N, O} with attach
    # at the neutral N and a terminal, neutral, doubly-bonded O -- a nitrite
    # ester -O-N=O (attach O) or an N-oxide never match (guarded on which atom
    # is attach_idx + the bond order, mirroring `substituent_enumerator`'s
    # analogous whole-fragment nitroso detector).
    if len(_sub_set) == 2:
        _no_attach = mol.GetAtomWithIdx(attach_idx)
        if _no_attach.GetSymbol() == 'N' and _no_attach.GetFormalCharge() == 0:
            _other = [i for i in _sub_set if i != attach_idx]
            if len(_other) == 1:
                _o_atom = mol.GetAtomWithIdx(_other[0])
                if (_o_atom.GetSymbol() == 'O' and _o_atom.GetDegree() == 1
                        and _o_atom.GetFormalCharge() == 0
                        and mol.GetBondBetweenAtoms(
                            attach_idx, _other[0]).GetBondType()
                        == Chem.BondType.DOUBLE):
                    return 'nitroso'

    # W3-P02-2 (P-65.1.3.1.2(1), BB 30033 / acyl table 56178): a substituent
    # that is the imidic-acid carbon -C(=NH)-OH attached to a ring/ring-system
    # (or any parent) is the compound acyl prefix 'C-hydroxycarbonimidoyl'
    # (HO-C(=NH)-; the italic 'C-' marks the hydroxy ON CARBON, distinguishing
    # it from an N-hydroxy form). Without this the recursive fallback names the
    # fragment 'methanimidic acid' + 'yl' = 'methanimidic acidyl' (OPSIN-
    # unparseable -> SELF-01 -> unknown). Detect: the attach C carries exactly a
    # terminal neutral =NH (degree-1, charge-0, unsubstituted) + a terminal -OH
    # (degree-1, one H) + one bond to the parent; the whole substituent is those
    # three atoms. Fail closed on anything else (N-substituted imino, O-alkyl,
    # extra branch) so a wrong name is never emitted.
    if attach_idx is not None and len(sub_atoms) == 3:
        _ic = mol.GetAtomWithIdx(attach_idx)
        if _ic.GetSymbol() == 'C' and _ic.GetFormalCharge() == 0:
            _sub_set = set(sub_atoms)
            _im_n = None
            _oh_o = None
            _ic_ok = True
            for _nb in _ic.GetNeighbors():
                _ni = _nb.GetIdx()
                if _ni not in _sub_set:
                    continue  # the single bond back to the parent
                _b = mol.GetBondBetweenAtoms(attach_idx, _ni)
                if (_nb.GetSymbol() == 'N'
                        and _b.GetBondTypeAsDouble() == 2.0
                        and _nb.GetFormalCharge() == 0
                        and _nb.GetDegree() == 1):
                    _im_n = _ni
                elif (_nb.GetSymbol() == 'O'
                        and _b.GetBondTypeAsDouble() == 1.0
                        and _nb.GetDegree() == 1
                        and _nb.GetTotalNumHs() == 1):
                    _oh_o = _ni
                else:
                    _ic_ok = False
                    break
            if (_ic_ok and _im_n is not None and _oh_o is not None
                    and _sub_set == {attach_idx, _im_n, _oh_o}):
                # Only DEMOTE the imidic acid to this acyl prefix when a SENIOR
                # group is present (mode 1, P-65.1.3.1.2). If imidic acid is the
                # molecule's principal characteristic group it belongs in the
                # suffix ('benzenecarboximidic acid'), so fail closed here rather
                # than emit the valid-but-non-PIN '(C-hydroxycarbonimidoyl)benzene'.
                from ..perception.functional_groups import (
                    detect_functional_groups as _dfg,
                )
                from ..rules.seniority import get_principal_group as _gpg
                _pg, _ = _gpg(mol, _dfg(mol))
                if _pg == 'imidic_acid':
                    return None
                return 'C-hydroxycarbonimidoyl'

    # W3-P02-7 (P-65.1.7.2.2, BB 30462): a substituent that is an imidoyl carbon
    # R-C(=NH)- (R = linear alkyl) is the '{stem}animidoyl' acyl prefix —
    # 'ethanimidoyl' for CH3-C(=NH)-, 'methanimidoyl' for HC(=NH)-. Used e.g. as
    # an amide N-substituent: 'N-ethanimidoyl-N-methylacetamide'. Mirrors the AM-5
    # detector (benzene.py) at the universal substituent namer. Detect: the attach
    # C (acyclic, neutral) bears exactly ONE terminal neutral unsubstituted =NH
    # (degree-1, charge-0, one H); the remainder of the fragment (sub_atoms minus
    # the imino N) is a pure ACYCLIC UNBRANCHED alkyl chain. The parent-attachment
    # atom is the attach C neighbour NOT in sub_atoms (parent_chain is [] here, so
    # the walk is confined to sub_atoms). Fail closed for ring / branched /
    # substituted-imino / hetero remainders (other classes, e.g. the
    # C-hydroxycarbonimidoyl detector above owns the -OH variant).
    if attach_idx is not None and len(sub_atoms) >= 2:
        _mc = mol.GetAtomWithIdx(attach_idx)
        _sub_set2 = set(sub_atoms)
        if (_mc.GetSymbol() == 'C' and _mc.GetFormalCharge() == 0
                and not _mc.IsInRing()):
            _im_ns = [
                n.GetIdx() for n in _mc.GetNeighbors()
                if n.GetSymbol() == 'N'
                and n.GetIdx() in _sub_set2
                and mol.GetBondBetweenAtoms(
                    attach_idx, n.GetIdx()).GetBondTypeAsDouble() == 2.0
            ]
            if len(_im_ns) == 1:
                _im_n2 = mol.GetAtomWithIdx(_im_ns[0])
                _im_heavy = [y for y in _im_n2.GetNeighbors()
                             if y.GetIdx() != attach_idx]
                # The alkyl part is the whole fragment minus the imino N.
                _alkyl_set = _sub_set2 - {_im_ns[0]}
                _alkyl_ok = bool(_alkyl_set)
                for _a in _alkyl_set:
                    _aa = mol.GetAtomWithIdx(_a)
                    if _aa.GetSymbol() != 'C' or _aa.IsInRing():
                        _alkyl_ok = False
                        break
                    # linear chain: <=2 heavy neighbours WITHIN the alkyl part
                    _ndeg = sum(1 for nb in _aa.GetNeighbors()
                                if nb.GetIdx() in _alkyl_set)
                    if _ndeg > 2:
                        _alkyl_ok = False
                        break
                if (_alkyl_ok and _im_n2.GetFormalCharge() == 0
                        and not _im_heavy and _im_n2.GetTotalNumHs() == 1):
                    from ..data.chain_names import get_chain_prefix
                    _stem = get_chain_prefix(len(_alkyl_set))
                    if _stem:
                        return f"{_stem}animidoyl"

    # W3-P02-8 (v33 Phase 6 E2e, P-66.4.1.3.1 / P-66.4.1.2): a carbamimidoyl
    # substituent -C(=NH)-NH-R attached to the PARENT via one of its own amino
    # nitrogens -- the N(5)-substituent-of-ornithine shape (CHEBI:138933,
    # N(5)-(N-nitrosocarbamimidoyl)-L-ornithine). From the WHOLE molecule's
    # view the amidine carbon carries THREE nitrogens (one imino =NH, two
    # amino -NH-), but ONE amino nitrogen IS the parent attachment itself --
    # `sub_atoms` (by construction excluding the parent atom) only ever holds
    # the OTHER two: the bare imino N plus the remaining amino N's own R
    # substituent. That reduces to the ordinary 2-N carbamimidoyl shape,
    # N-substituted with R -- 'N-nitrosocarbamimidoyl' for -NH-N=O. Mirrors
    # the ring-attached AM-5/D1 detector (`rules.benzene
    # ._detect_amidine_n_substituents` / `_build_amidine_n_prefix`), reused
    # here for its OUTPUT FORMATTER only: that detector assumes the amidine
    # carbon bonds to the ring/chain PARENT directly (a C-C/C-ring bond), so
    # its own N-walk cannot be reused as-is when the attachment bond is
    # itself one of the amidine's own nitrogens -- a fresh, narrowly-scoped
    # structural match is used instead.
    #
    # Without this, Step 3-4's recursive path caps the fragment (losing the
    # parent's own nitrogen, so the ISOLATED capped molecule is genuinely
    # smaller than what the fragment encodes) and the general engine has no
    # N-nitroso-amidine reading at all -- measured: it silently DROPS the
    # nitroso and reports the plain 'methanimidamide' (2 atoms short of the
    # 5-atom fragment), which SELF-01's atom-count mismatch already catches
    # and safely fails closed (0-wrong preserved; only breadth was missing).
    #
    # Detect: attach_idx is an acyclic, neutral carbon with EXACTLY two
    # neighbours WITHIN sub_atoms -- one via a DOUBLE bond to a terminal,
    # neutral, unsubstituted N (=NH, degree 1: the imino nitrogen), and one
    # via a SINGLE bond to another neutral, non-ring N (the amino nitrogen).
    # The amino nitrogen's own substituent (if any, beyond attach_idx) is
    # named via the SAME recursive substituent namer used everywhere else in
    # this cascade -- 0 substituents (bare -NH2) gives plain 'carbamimidoyl',
    # exactly 1 gives 'N-{R}carbamimidoyl'. Fail-closed (returns None, falls
    # through) for >1 amino-N substituent (a tertiary amino nitrogen -- not a
    # plain amidine), a charged/ring atom anywhere in the shape, an
    # unaccounted atom (no silent drop), or an amino-N substituent the
    # recursive namer cannot express.
    if attach_idx is not None and len(sub_atoms) >= 2:
        _amc = mol.GetAtomWithIdx(attach_idx)
        _sub_set3 = set(sub_atoms)
        if (_amc.GetSymbol() == 'C' and _amc.GetFormalCharge() == 0
                and not _amc.IsInRing()):
            _in_frag_n = [
                n for n in _amc.GetNeighbors()
                if n.GetIdx() in _sub_set3 and n.GetSymbol() == 'N'
            ]
            if len(_in_frag_n) == 2:
                _bonds = [
                    (n, mol.GetBondBetweenAtoms(attach_idx, n.GetIdx()))
                    for n in _in_frag_n
                ]
                _imino_ns = [n for n, b in _bonds
                             if b.GetBondTypeAsDouble() == 2.0]
                _amino_ns = [n for n, b in _bonds
                             if b.GetBondTypeAsDouble() == 1.0]
                if len(_imino_ns) == 1 and len(_amino_ns) == 1:
                    _im_n = _imino_ns[0]
                    _am_n = _amino_ns[0]
                    _im_heavy = [y for y in _im_n.GetNeighbors()
                                 if y.GetIdx() != attach_idx]
                    if (_im_n.GetFormalCharge() == 0 and not _im_heavy
                            and _im_n.GetTotalNumHs() == 1
                            and not _im_n.IsInRing()
                            and _am_n.GetFormalCharge() == 0
                            and not _am_n.IsInRing()):
                        _am_subs = [
                            y.GetIdx() for y in _am_n.GetNeighbors()
                            if y.GetIdx() != attach_idx
                        ]
                        if not _am_subs:
                            return 'carbamimidoyl'
                        if len(_am_subs) == 1:
                            _r_seed = _am_subs[0]
                            _block3 = {_im_n.GetIdx(), _am_n.GetIdx(), attach_idx}
                            _r_atoms: List[int] = []
                            _seen_r = set(_block3)
                            _stack_r = [_r_seed]
                            while _stack_r:
                                _cur = _stack_r.pop()
                                if _cur in _seen_r:
                                    continue
                                _seen_r.add(_cur)
                                _r_atoms.append(_cur)
                                for _nb in mol.GetAtomWithIdx(_cur).GetNeighbors():
                                    if (_nb.GetIdx() in _sub_set3
                                            and _nb.GetIdx() not in _seen_r):
                                        _stack_r.append(_nb.GetIdx())
                            # Every fragment atom must be accounted for by the
                            # amidine core + the R branch -- no silent drop.
                            if set(_r_atoms) | _block3 == _sub_set3:
                                _r_name = name_substituent_fragment(
                                    mol, _r_atoms, _r_seed,
                                    list(parent_set | {attach_idx, _am_n.GetIdx()}),
                                )
                                if _r_name and ' ' not in _r_name:
                                    from ..rules.benzene import (
                                        _build_amidine_n_prefix,
                                    )
                                    _n_pfx = _build_amidine_n_prefix(
                                        [('N', _r_name)])
                                    if _n_pfx:
                                        return f"{_n_pfx}carbamimidoyl"

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
                # v33 (giants engine, acyl-CoA): a chain-rooted ring-bearing
                # fragment (P-29.1.2, ``-CH2-[ring]``) where the ring ALSO
                # carries its own further ring-system substituent (e.g. the
                # ribose ring's purine base, non-fused, joined by a single
                # bond) only resolves through this chokepoint's
                # ``_compound_ring_on_chain_substituent`` / decorated-fused
                # branches when ``allow_mancude=True`` -- measured directly
                # (isolated adenosine-3'-phosphate fragment: None without,
                # '[(2R,3S,4R,5R)-5-(6-amino-9H-purin-9-yl)-4-hydroxy-3-
                # (phosphonooxy)oxolan-2-yl]methyl' with). That capability is
                # new and unproven against the PIN gold set (same reasoning
                # as the existing ``allow_mancude`` best-effort gate a few
                # lines below in this same function), so gate it on
                # ``best_effort_ctx`` -- a non-best-effort call is
                # byte-identical to before this existed.
                _ring_mancude = False
                try:
                    from ..metrics.provenance import best_effort_ctx as _rbe_ctx
                    _ring_mancude = bool(_rbe_ctx.get())
                except Exception:
                    _ring_mancude = False
                _ring_nm = name_ring_system_substituent(
                    mol, sorted(sub_atoms), attach_idx,
                    allow_enumerator_fallback=False,
                    allow_mancude=_ring_mancude,
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

    # Step 2b-branched (v28 Cluster A Fix 4, P-32.1.1(1)): a BRANCHED acyclic
    # all-carbon alkenyl/alkynyl substituent. The linear namer above declines
    # branching; without this the recursion + parent_to_prefix drop the free-
    # valence locant ('2-methylprop-1-enyl') or mis-name the connectivity.
    branched_unsat = _name_branched_alkenyl_substituent(mol, sub_atoms, attach_idx)
    if branched_unsat is not None:
        return _add_substituent_stereo(mol, sub_atoms, branched_unsat, attach_idx=attach_idx)

    # Step 2b-oxo (v26 BP-2 RC-3, P-33 / P-14.4): unsaturated all-carbon chain
    # carrying an in-chain aldehyde/ketone -> oxo prefix numbered from the free
    # valence (-C(=CH2)CHO -> '3-oxoprop-1-en-2-yl'). Sits between the pure-alkenyl
    # namer (declines: =O is not C) and the saturated polyfunctional builder
    # (declines: unsaturation + internal attachment). Fail-closed otherwise.
    unsat_oxo = _name_unsaturated_oxo_substituent(
        mol, sub_atoms, attach_idx, parent_set)
    if unsat_oxo is not None:
        return _add_substituent_stereo(mol, sub_atoms, unsat_oxo, attach_idx=attach_idx)

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

    # Tier FG (v33): the PROPER re-rooted FG-capable located substituent namer.
    # Placed here -- BEFORE the Wave2 T3a ring-atom guard just below -- because
    # that guard's "any ring atom anywhere in sub_atoms -> decline" check also
    # catches a shape T3a never intended to block: a fragment whose attach_idx
    # is an ORDINARY ACYCLIC atom and the ring is reached only several bonds
    # downstream (e.g. a giant acyl-CoA's C(=O)-S-CH2CH2-NH-...-O-CH2-[ribose
    # ring]-[purine] "S-alkyl" ester group). T3a's own worked example
    # (-CH2-O-CH2-Ar(OH)) is a fragment with NO acyclic backbone at all beyond
    # the ring -- Tier FG's carbon-only backbone walk (`_located_fg_assemble`)
    # simply never engages for it (attach_idx would not land in `carbon_set`),
    # so moving this block earlier cannot resurrect that defect. For the
    # genuine chain-reaches-a-ring shape, `_located_fg_assemble` builds the
    # acyclic backbone through the free valence and, on reaching the ring,
    # recurses through `name_substituent_fragment` on JUST the ring branch
    # (a small, PROPERLY re-anchored sub-fragment where the ring chokepoint
    # -- Step 1c above -- gets a fair, correctly-scoped shot) instead of
    # collapsing ring+backbone into one opaque blob and string-surgering a
    # free-molecule name (parent_to_prefix's fail-closed contract, untouched).
    # Names an acyclic-carbon-backbone fragment rooted at the attachment atom,
    # citing every FG / ring / hetero group as a structurally-numbered
    # detachable prefix (P-46.1.12). Gated on best_effort_ctx, so a non-best-
    # effort (PIN) call is byte-identical to before this block existed; the
    # top-level SELF-01/OPSIN gate voids any non-RT candidate, keeping
    # 0-wrong.
    _fg_best_effort = False
    try:
        from ..metrics.provenance import best_effort_ctx
        _fg_best_effort = bool(best_effort_ctx.get())
    except Exception:
        _fg_best_effort = False
    if _fg_best_effort and attach_idx is not None and attach_idx in set(sub_atoms):
        _fg_name = None
        _fg_located = None
        try:
            _asym = mol.GetAtomWithIdx(attach_idx).GetSymbol()
            if _asym in ('O', 'S', 'N'):
                _fg_name = _located_fg_hetero_root(mol, list(sub_atoms), attach_idx)
            else:
                # v33 bugfix: request `with_pos=True` in the SAME call that
                # builds `_fg_name`, and hand the resulting (name, k, pos)
                # triple to `_add_substituent_stereo` as `located`. This chain
                # is the ONLY deriver that can locate a stereocentre sitting on
                # an FG-shaped chain (the strict, non-FG deriver
                # `_add_substituent_stereo` falls back to declines it, which
                # is how a chain stereocentre here -- e.g. the acyl-CoA
                # pantetheine 3-hydroxy centre -- used to be silently dropped
                # instead of just missing a locant). Passing the SAME triple
                # already used for `_fg_name` (rather than letting the callee
                # re-derive one from `attach_idx` alone) guarantees the name
                # `_add_substituent_stereo` decorates is byte-identical to
                # this one -- no risk of a second, differently-tie-broken
                # recomputation silently replacing a correct name.
                _r = _located_acyclic_alkyl_name(
                    mol, list(sub_atoms), attach_idx, allow_functional=True,
                    with_pos=True)
                _fg_name = _r[0] if _r is not None else None
                _fg_located = _r if _r is not None else None
        except Exception as _exc:  # noqa: BLE001
            logger.debug("Tier FG located namer failed: %s", _exc)
            _fg_name = None
            _fg_located = None
        if _fg_name is not None:
            return _add_substituent_stereo(
                mol, sub_atoms, _fg_name, attach_idx=attach_idx,
                located=_fg_located)

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

    # Step 2e (v32 Phase 3A-b, P-74.1.3): a fragment with a REAL net formal
    # charge (a genuine onium cation on the branch -- e.g. the choline/
    # trimethylammonium head of a phosphatidylcholine-like lipid), NOT an
    # internal charge-separated pair that cancels within the same fragment
    # (nitro/N-oxide/azide/diazo net to 0 and are untouched by this guard).
    # Steps 3-5 below are CHARGE-BLIND: name_fragment_recursively correctly
    # builds a valid whole-molecule name for the ISOLATED fragment carrying
    # the ionic '-ium' suffix (e.g. 'N,N,N-trimethylethan-1-aminium'), then
    # parent_to_prefix does STRING-LEVEL suffix surgery to turn it into a
    # prefix -- which either produces the OPSIN-unparseable '-aminiumyl'
    # ending, or silently reinterprets the attachment bond as an extra
    # substituent. Measured: a real choline-phosphate compound substituent
    # came out '(2-phosphonooxy-N,N,N-trimethylethan-1-aminium)yl',
    # SELF-01-suppressed (a 0-wrong defect this project's PIN tiers must not
    # rely on the OPSIN backstop alone to catch, the contributor guide invariant 1).
    #
    # cation_to_prefix (P-74.1.3, the existing structured primitive, wired
    # previously only at charged_router.py:478 / rules/ions.py:4305) builds
    # the OPSIN-valid 'azaniumyl'-family prefix directly off the cation
    # atom's own substituents. It is shaped for the DIRECT-ATTACHMENT case
    # (the cation IS this fragment's own attach atom, with the parent one
    # bond away) -- route exactly that shape through it. Any other charged
    # shape (the charge sits deeper in the fragment, behind a chain this
    # project has no nested-substituent composer for yet, or the net charge
    # is an anion) fails closed rather than guessing -- never emit a name for
    # a different molecule. Neutral fragments (net charge 0) are UNCHANGED --
    # this block is a no-op for them (byte-identical).
    if sum(mol.GetAtomWithIdx(a).GetFormalCharge() for a in sub_atoms) != 0:
        _cat_atom = mol.GetAtomWithIdx(attach_idx)
        if _cat_atom.GetFormalCharge() > 0:
            _parent_nbrs = [n.GetIdx() for n in _cat_atom.GetNeighbors()
                            if n.GetIdx() in parent_set]
            if len(_parent_nbrs) == 1:
                _cp = cation_to_prefix(mol, attach_idx, _parent_nbrs[0])
                if _cp:
                    return _add_substituent_stereo(
                        mol, sub_atoms, _cp, attach_idx=attach_idx)
        logger.debug(
            "DROP-26 substituent_skip: reason=charged_fragment_not_direct_"
            "cation_attach atom_count=%d", len(sub_atoms),
        )
        return None

    # Step 3: Extract fragment SMILES and name recursively
    frag_smiles = _extract_fragment_smiles(mol, sub_atoms, attach_idx, parent_set)
    if frag_smiles is None:
        # Fallback: simple carbon count — ONLY when that name is honest for this
        # fragment (see fragment_is_linear_terminal_alkyl); otherwise fail closed
        # rather than drop the atoms the count ignores.
        carbon_count = len(sub_atoms)
        if (carbon_count > 0
                and fragment_is_linear_terminal_alkyl(mol, sub_atoms, attach_idx)):
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
        # Recursion depth limit, cycle guard, or a REFUSAL SENTINEL from the
        # recursive namer. Fall back to the carbon count ONLY when an unbranched
        # terminal alkyl name is honest for this fragment: this line is what turned
        # the CCS[Zn]SCC sentinel refusal into the structurally WRONG 'ethylethane'
        # once the sentinel stopped leaking through as a string.
        carbon_count = len(sub_atoms)
        if (carbon_count > 0
                and fragment_is_linear_terminal_alkyl(mol, sub_atoms, attach_idx)):
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
    prefix_name = parent_to_prefix(
        parent_name, chain_length=carbon_count,
        attach_locant=ATTACH_LOCANT_UNKNOWN)

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


# W4-I4 (P-74.1.3 / P-74.2.1.1): parent-hydride cation stem for the onium
# substituent prefix (stem + 'iumyl'). Nitrogen keeps 'azaniumyl' (the amine-based
# '-methanaminiumyl' PIN is OPSIN-unparseable; azaniumyl is the OPSIN-valid form).
_ONIUM_PREFIX_STEM = {
    'N': 'azan', 'P': 'phosphan', 'As': 'arsan', 'Sb': 'stiban',
    'O': 'oxidan', 'S': 'sulfan', 'Se': 'selan', 'Te': 'tellan',
}


def cation_to_prefix(mol, cation_idx: int, parent_attach_idx: int,
                      as_free_ion: bool = False) -> str:
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
        as_free_ion: when True, build the standalone onium-cation UNIT name
            (stem + ``ium``, e.g. ``trimethylazanium``) instead of the
            ``-iumyl`` substituent-PREFIX form. Used by the P-73.5.1.1/.2
            multiplicative bis(...)/tris(...) polycation assembly (v33 Phase
            3, e.g. ``hexane-1,6-diylbis(trimethylazanium)``), where the
            repeated cationic UNIT is cited as a complete parent-cation name,
            not a ``-yl`` substituent (cf. BB PIN example
            ``(1,4-phenylene)bis(phosphanium)``). The N-substituent
            composition (alphabetized/multiplied) is identical either way;
            only the trailing suffix differs.

    Returns:
        The cation prefix WITHOUT enclosing marks (e.g. ``trimethylazaniumyl``,
        or ``trimethylazanium`` when ``as_free_ion``), or '' for an
        out-of-scope cation (ylide / non-N onium / amine-oxide / 1,n-dipolar —
        P-74.2 deferred, D-06 honest-fail).
    """
    cat = mol.GetAtomWithIdx(cation_idx)

    # SCOPE: the onium cation-substituent prefix is the parent-hydride cation stem
    # + 'iumyl' (P-73.1.1.1 / P-74.1.3 / P-74.2.1.1). W4-I4 GENERALIZED from
    # nitrogen-only (azaniumyl) to the P/O/S/Se/Te/As onium family:
    #   N -> azaniumyl   P -> phosphaniumyl   O -> oxidaniumyl   S -> sulfaniumyl
    # The nitrogen PIN is the amine-based '<...>methanaminiumyl', but that form is
    # OPSIN-unparseable; 'azaniumyl' is the documented OPSIN-valid equivalent, so
    # nitrogen keeps 'azaniumyl' here (W4-I4 note).
    stem = _ONIUM_PREFIX_STEM.get(cat.GetSymbol())
    if stem is None or cat.GetFormalCharge() <= 0:
        return ''
    _onium_suffix = (stem + 'ium') if as_free_ion else (stem + 'iumyl')

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
        # Bare protonated heteroatom with no extra substituents ([NH3+]-parent) is
        # the `<stem>iumyl` group itself (P-74.1.3): e.g. glycine zwitterion's
        # cation is `azaniumyl`. The caller decides whether to use this or keep the
        # amino-acid neutral/retained form (sequenced first, D-06).
        return _onium_suffix

    return _compose_n_substituent_prefix(sub_prefixes) + _onium_suffix


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
    # A TERMINAL -O-alkyl (alkoxy) neighbour: the O is bonded ONLY to the
    # Group-14 centre + one pure-organyl carbon (a silicic/germanic ester O-R),
    # neutral, no H. Cited via the contracted alkoxy prefix (P-68.2.6.2, BB
    # 38245: -Ge(OEt)3 -> 'triethoxygermyl'). A -O-R where R is not a pure
    # organyl (nested ether), or an O bearing extra heavy neighbours, -> None.
    if (sym == 'O' and atom.GetTotalNumHs() == 0 and len(heavy_in_frag) == 2
            and any(nb.GetIdx() == exclude_idx for nb in heavy_in_frag)):
        others = [nb for nb in heavy_in_frag if nb.GetIdx() != exclude_idx]
        if len(others) == 1 and others[0].GetSymbol() == 'C':
            # ROUTING PREDICATE, not a namer: the answer selects a DIFFERENT
            # producer (`get_alkoxy_prefix`, the contracted alkoxy builder), whose
            # behaviour on ring-bearing / branched / unsaturated R is not
            # established. Widening this test would change which producer claims a
            # molecule rather than turn a refusal into an emission, so it keeps the
            # narrow semantics under the explicitly narrow name (v29 P3).
            from ..rules.substituent_purity import is_simple_unbranched_organyl
            r_c = others[0].GetIdx()
            if is_simple_unbranched_organyl(mol, r_c, idx):
                from .substituent_prefix_forms import get_alkoxy_prefix
                alk = get_alkoxy_prefix(mol, (idx, exclude_idx, r_c),
                                        principal_chain=[exclude_idx])
                if alk is not None:
                    return alk
        return None
    # The organyl neighbour, NAMED (this string is the returned prefix). v29 P3:
    # routed to the shared chokepoint, so a ring-bearing, branched, unsaturated or
    # long organyl on the Group-14 centre is named instead of refused. The refusal
    # here did NOT fail closed -- it handed the fragment to a sibling producer that
    # fabricated a linear chain, shipping `(dimethylpropylsilyl)acetic acid` for
    # the propan-2-yl compound (a WRONG CONSTITUTION, masked only by the SELF-01
    # OPSIN gate, which fails OPEN with no JRE).
    from ..rules.substituent_purity import organyl_prefix_name
    return organyl_prefix_name(mol, idx, exclude_idx)


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
    return _compose_group14_prefixes(prefixes) + stem


def _compose_group14_prefixes(prefixes: List[str]) -> str:
    """Compose the prefix block of a Group-14 substituent centre.

    Split out of ``_compose_n_substituent_prefix`` (which stays with its ONE
    remaining caller, the onium cation-prefix path) because v29 P3 widened the
    organyl guard feeding it: a prefix here may now carry locants
    (``propan-2-yl``), a retained italicized prefix (``tert-butyl``) or its own
    enclosing marks (``(4-methylphenyl)methyl``), none of which the shared
    N-substituent composer handled.

    Two shared primitives replace raw string work, and BOTH are no-ops on the
    letters-only class the retired narrow walker could return -- which is what
    keeps every previously-emitted Group-14 prefix byte-identical:

    * ``prefix_citation_sort_key`` -- P-14.5.2/P-14.5.4 citation order. Raw
      ``sorted()`` ordered ``tert-butyl`` on its 't' and would have emitted
      ``dimethyltert-butylsilyl`` for the compound that is spelled
      ``tert-butyldimethylsilyl``.
    * ``enclose_if_compound`` -- P-16.5.1.1 marks for a compound prefix, so
      ``propan-2-yl`` is cited ``(propan-2-yl)`` and cannot run into the
      neighbouring token.

    NOT applied here, and v29 P3-FIX Item 9 replaces the reason. The
    P-16.5.1.3.1 (BB 7272) "second and further substituents are each enclosed even
    for simple substituents" leg would respell ``chlorodimethylsilyl`` as
    ``chlorodi(methyl)silyl``. The old justification -- "a sibling Group-14
    producer emits the BARE form for that shape" -- could not be substantiated: no
    such sibling was found in this module (only this composer and
    ``_compose_n_substituent_prefix``, the latter documented as onium-only).

    The rule genuinely does not reach here, for a reason the rule states itself:
    its scope sentence is "*For mononuclear parent HYDRIDES with two or more
    substituents...*", and what is being built here is a substituent PREFIX, not a
    parent hydride. The Blue Book spells the prefix bare in exactly this shape and
    contrasts it with the marked parent form:
    ``1-(trimethylsilyl)ethan-1-one (PIN)`` against the alternative
    ``acetyltri(methyl)silane`` (BlueBookV2.md:29380);
    ``[2-(1,3-dioxolan-2-yl)ethyl]tri(methyl)silane (PIN)`` (:35326);
    ``(tert-butylperoxy)dimethylsilyl propanoate (PIN)`` (:23389), a bare
    ``dimethyl`` on a silyl PREFIX; ``3-(trimethoxysilyl)propane-1-thiol (PIN)``
    (:28164). So the bare form is BB-supported here and the parenthesised form is
    what the rule requires of the parent-hydride name.
    """
    from collections import Counter

    from .naming_utils import (enclose_if_compound, get_multiplier_prefix,
                               is_complex_substituent, prefix_citation_sort_key)

    counts = Counter(prefixes)
    parts = []
    for name in sorted(counts.keys(), key=prefix_citation_sort_key):
        count = counts[name]
        mult = get_multiplier_prefix(count, name)
        if is_complex_substituent(name) and count > 1:
            parts.append(f"{mult}({name})")
        else:
            # v29 P3-FIX Item 4: a SIMPLE multiplier joined to a name that leads
            # with an ITALICIZED structural prefix keeps the hyphen boundary.
            # `**P-16.2.4** Hyphens` -> `**P-16.2.4.1** Hyphens are used in
            # substitutive names:` -> clause `(d) to separate italic letters from
            # Roman letters` (BlueBookV2.md:6957) gives the verbatim example
            # `di-tert-butyl (P-61.2.3)` (:6964), and `### **P-61.2.2** Cyclic
            # hydrocarbons` gives `1,2-di-tert-butylbenzene (PIN)` (:25717).
            # `tert-butyl` is SIMPLE, so the multiplier is `di` and not `bis`:
            # `**P-16.3.2** General methodology` clause (a) (:7033) names it
            # explicitly -- "Simple components are ... unsubstituted prefixes,
            # such as ethyl or tert-butyl ... multiplied by ... 'di', 'tri'".
            # Without this the composer emitted `ditert-butyl`, which appears
            # ZERO times in the Blue Book, while a SIBLING Group-14 producer
            # spelled the same fragment `di-tert-butyl` -- one fragment, two
            # spellings, both OPSIN-clean, so the gate could not see it.
            # `multiplier_needs_hyphen` is the shared primitive the six other
            # composers in this class already use; this was the site that
            # open-coded around it.
            parts.append(f"{mult}{enclose_if_compound(name)}")
    return ''.join(parts)


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
