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
        else:
            return None                  # S / P / B / etc. -> decline

    # A single carbon bearing BOTH oxo and hydroxy is a carboxyl carbon (-C(=O)OH)
    # that Pass-1 did not consume as 'carboxy' (e.g. its only non-O neighbour is the
    # non-carbon parent — a carbamic acid C-N). Never emit 'hydroxyoxomethyl' for a
    # carboxylic acid: fail-closed so another tier names it (P-65.1.1).
    for _prefixes in prefix_on.values():
        if _POLYFUNC_OXO_PREFIX in _prefixes and _POLYFUNC_HYDROXY_PREFIX in _prefixes:
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

    # Identify ether oxygens: neutral, divalent, acyclic, both neighbours in the
    # fragment, both bonds single. Anything else (charged O, =O, -OH, ring O,
    # peroxide -O-O-) disqualifies the whole fragment (fail-closed).
    ether_os: List[int] = []
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()
        if sym == 'C':
            continue
        if sym != 'O':
            return None  # any non-C, non-O heteroatom -> richer case
        if (atom.GetFormalCharge() != 0 or atom.GetTotalNumHs() != 0
                or atom.GetDegree() != 2 or ring_info.NumAtomRings(idx) > 0):
            return None
        nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
        if not all(n in sub_set for n in nbrs):
            return None
        if any(mol.GetBondBetweenAtoms(idx, n).GetBondType()
               != Chem.BondType.SINGLE for n in nbrs):
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

    # Name each ether as an (R-oxy) prefix at its backbone-carbon locant.
    from collections import defaultdict
    from ..data.chain_names import get_chain_prefix
    from .substituent_prefix_forms import get_alkoxy_prefix
    groups: dict = defaultdict(list)
    for bb_c, o_idx, r_side in ether_links:
        oxy = get_alkoxy_prefix(mol, (o_idx, bb_c, r_side), backbone)
        if not oxy or oxy == "alkoxy":
            return None  # un-nameable R -> fall through (fail-closed)
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
    if carbon_count == 4 and len(c_neighbors_in_frag) == 3:
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

    # Acyclic, all-carbon, fully saturated within the fragment.
    ring_info = mol.GetRingInfo()
    for idx in sub_set:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            return None
        if ring_info.NumAtomRings(idx) > 0:
            return None
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in sub_set:
                bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                if bond and bond.GetBondTypeAsDouble() != 1.0:
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
                    if nidx in sub_set and nidx != parent:
                        cand = [node] + best_child.get(nidx, [nidx])
                        if len(cand) > len(best):
                            best = cand
                best_child[node] = best
            else:
                stack.append((node, parent, True))
                for nbr in mol.GetAtomWithIdx(node).GetNeighbors():
                    nidx = nbr.GetIdx()
                    if nidx in sub_set and nidx != parent:
                        stack.append((nidx, node, False))
        return best_child[start]

    arms = []
    for nbr in mol.GetAtomWithIdx(attach_idx).GetNeighbors():
        if nbr.GetIdx() in sub_set:
            arms.append(_longest_arm(nbr.GetIdx(), attach_idx))
    arms.sort(key=len, reverse=True)
    if len(arms) >= 2:
        chain = list(reversed(arms[0])) + [attach_idx] + arms[1]
    elif len(arms) == 1:
        chain = [attach_idx] + arms[0]
    else:
        chain = [attach_idx]
    chain_len = len(chain)
    if chain_len < 2:
        return None

    # Number so the free valence gets the LOWEST locant (P-46.1.8).
    if (chain_len - chain.index(attach_idx)) < (chain.index(attach_idx) + 1):
        chain = list(reversed(chain))
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
