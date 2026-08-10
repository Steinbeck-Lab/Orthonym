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

from .naming_utils import (
    ALKYL_NAMES, alpha_sort_key, enclose_if_compound, get_alkyl_name)
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


def _collect_fragment_atoms(mol, start_atom: int, exclude: Set[int]) -> Set[int]:
    """Return the set of atom indices in the fragment reachable from
    ``start_atom`` without crossing any atom in ``exclude`` (BFS). Companion to
    :func:`_count_fragment_atoms` used by the aryloxy decorator (P-45.6)."""
    visited: Set[int] = set()
    queue = deque([start_atom])
    while queue:
        idx = queue.popleft()
        if idx in visited or idx in exclude:
            continue
        visited.add(idx)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            ni = nbr.GetIdx()
            if ni not in visited and ni not in exclude:
                queue.append(ni)
    return visited


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

    # Case A: O -> aromatic ring C. A BARE phenyl -> retained 'phenoxy'
    # (P-63.2.2.2). A SUBSTITUTED aryloxy ring MUST carry its ring substituents
    # (P-45.6 / P-45.2.1): the old unconditional 'phenoxy' silently DROPPED them
    # (4-chlorophenoxy -> 'phenoxy'), an atom-drop the OPSIN self-consistency
    # gate suppresses to 'unknown' WITH Java and SHIPS wrong without it. The old
    # blanket fallback also mis-named EVERY other aromatic ether 'phenoxy'
    # (heteroaryl / fused / naphthyl) — now fail-closed instead of wrong.
    if sub_atom.GetIsAromatic():
        ring_info = mol.GetRingInfo()
        arom_ring = None
        for ring in ring_info.AtomRings():
            if sub_carbon in ring and all(
                mol.GetAtomWithIdx(r).GetIsAromatic() for r in ring
            ):
                arom_ring = ring
                break
        if arom_ring is None:
            return None  # aromatic C not in a fully-aromatic ring -> fail closed
        all_carbon_6 = len(arom_ring) == 6 and all(
            mol.GetAtomWithIdx(r).GetSymbol() == "C" for r in arom_ring
        )
        frag = _collect_fragment_atoms(mol, sub_carbon, {oxygen_idx})
        from ..rules.ring_substituents import decorated_ring_substituent_name
        dec = decorated_ring_substituent_name(
            mol, list(arom_ring), sub_carbon, expected_atoms=frag,
        )
        if dec is not None:
            if dec.endswith("phenyl"):
                # '4-chlorophenyl' -> '4-chlorophenoxy' (retained contraction)
                return dec[:-6] + "phenoxy"
            if dec.endswith("yl"):
                # decorated heteroaryl: 'pyridin-2-yl' -> '(pyridin-2-yl)oxy'
                # (P-63.2.2.2 / BB:27641 "(pyridin-2-yl)oxy (preferred prefix)";
                # BB:6267 "(5-chloropyridin-2-yl)oxy"). The locant-bearing free
                # valence keeps the whole '-yl' inside marks (F-spell-oxy).
                from .substituent_enumerator import alkoxy_prefix_from_substituent
                return alkoxy_prefix_from_substituent(dec)
            return None
        # dec is None -> a BARE ring (decorated_ring_substituent_name only handles
        # DECORATED rings). Bare benzene contracts to the retained 'phenoxy'; a bare
        # heteroaryl or fused-aryl ring is named from its full aromatic ring SYSTEM
        # as '<ring>-yloxy' (W8-P2 Task 2.1, P-63.2.2.2 / P-29.3.1) instead of being
        # dropped (the drop shipped a bare-parent name like 'ethanoic acid' gate-off).
        # Atom-drop veto: name ONLY when the perceived aromatic ring system is exactly
        # the collected fragment — any undecorated extra atom means an unhandled
        # substituent, so fail closed (never drop).
        _rings = [set(r) for r in ring_info.AtomRings()]
        _sys = set(next(r for r in _rings if sub_carbon in r))
        _changed = True
        while _changed:
            _changed = False
            for _r in _rings:
                if _r & _sys and not _r <= _sys:
                    _sys |= _r
                    _changed = True
        if set(frag) != _sys:
            return None  # undecorated extra atoms -> fail closed
        if all_carbon_6 and len(_sys) == 6:
            return "phenoxy"
        from ..rules.ring_substituents import get_ring_substituent_name
        _nm = get_ring_substituent_name(mol, tuple(_sys), sub_carbon)
        if _nm and _nm.endswith("yl") and _nm != "phenyl":
            # '(naphthalen-1-yl)oxy' / '(quinolin-2-yl)oxy' — the bare ring
            # '-yl' keeps its enclosing marks (P-63.2.2.2, BB:27641/22098);
            # a locant-free carbocycle concatenates ('cyclohexyloxy', BB:27768)
            # (F-spell-oxy).
            from .substituent_enumerator import alkoxy_prefix_from_substituent
            return alkoxy_prefix_from_substituent(_nm)
        return None

    # Case B: O -> CH(aryl)n -> benzyloxy (1 aryl) / diphenylmethoxy (2 phenyl).
    # HYG-04 (Phase 167): single shared aryl-count helper (was inline benzyloxy here).
    # Local import: file documents circular-import sensitivity (see header).
    from .substituent_naming import _name_aryl_methyl_ether
    _aryl_ether = _name_aryl_methyl_ether(mol, sub_carbon, oxygen_idx)
    if _aryl_ether is not None:
        return _aryl_ether

    # P-57.1.6.2 / P-46.1.3: the R side of the ether may itself be an
    # ether/thioether-bearing (replacement-numbered) chain, e.g.
    # -O-CH2CH2-O-CH3 -> '(2-methoxyethoxy)'. A bare carbon count would DROP
    # the interior heteroatom (constitutionally WRONG: -> 'propoxy'). When the
    # R fragment contains an interior ether O or thioether S, name R with the
    # full substituent namer (which handles the nested (R-oxy)/(R-sulfanyl)
    # recursion) and cite '({R}oxy)'.
    r_atoms: List[int] = []
    _r_seen = {oxygen_idx}
    _stack = [sub_carbon]
    while _stack:
        _cur = _stack.pop()
        if _cur in _r_seen:
            continue
        _r_seen.add(_cur)
        r_atoms.append(_cur)
        for _n in mol.GetAtomWithIdx(_cur).GetNeighbors():
            if _n.GetIdx() != oxygen_idx and _n.GetIdx() not in _r_seen:
                _stack.append(_n.GetIdx())
    _has_interior_hetero = any(
        mol.GetAtomWithIdx(a).GetSymbol() in ('O', 'S')
        for a in r_atoms
    )
    if _has_interior_hetero:
        from .substituent_naming import name_substituent_fragment
        from .naming_utils import apply_enclosing_marks
        _r_name = name_substituent_fragment(mol, r_atoms, sub_carbon, [oxygen_idx])
        # Only accept a clean, single-token (bracketable) recursive name; a
        # name with a space (a full compound name) means the R side is not a
        # nameable substituent here -> fall through / fail-closed.
        if _r_name and ' ' not in _r_name:
            # R -> R-oxy via the BB-verbatim P-63.2.2 morphology
            # (composed_alkoxy_prefix): the retained set contracts
            # ('methoxymethyl' -> 'methoxymethoxy'), a locant-bearing / ring
            # free valence keeps the whole '-yl' inside marks
            # ('oxan-2-yl' -> '(oxan-2-yl)oxy', NOT the mangled 'oxan-2-oxy'),
            # and C5+ concatenates ('pentyl' -> 'pentyloxy'). The compound
            # '{R}oxy' prefix is then enclosed by the caller (F-spell-oxy).
            from .substituent_enumerator import alkoxy_prefix_from_substituent
            _r_oxy = (alkoxy_prefix_from_substituent(_r_name)
                      if _r_name.endswith('yl') else _r_name + 'oxy')
            if _r_oxy is None:
                return None
            return apply_enclosing_marks(_r_oxy, 0)
        return None  # un-nameable ether-bearing R -> fail closed (never drop a hetero)

    # Name the R group by its CONSTITUTION, then apply the P-63.2.2.2 alkoxy
    # morphology.
    #
    # This used to be `ALKOXY_NAMES[carbon_count]` / `get_alkyl_name(count)` -- a
    # pure carbon COUNT with no branching test. A count cannot distinguish the
    # four C4H9 groups, so (CH3)3C-O-, (CH3)2CH-CH2-O-, CH3-CH2-CH(CH3)-O- and
    # CH3-[CH2]3-O- were ALL named 'butoxy': one string for four different
    # molecules. The Blue Book gives each its own preferred prefix
    # (P-63.2.2.2, BB 27665-27691) and composed_alkoxy_prefix spells that table.
    from .substituent_enumerator import (
        composed_alkoxy_prefix,
        composed_prefix_organyl_name,
    )
    _alkyl = composed_prefix_organyl_name(mol, r_atoms, sub_carbon)
    if _alkyl is None:
        return None  # un-nameable R -> fail closed (never a count-based guess)
    return composed_alkoxy_prefix(_alkyl)


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
        if c_on_chain:
            # W2F-P2 (P-65.6.3.3.5 method (1), BB 31958-31962): a CHAIN-MEMBER
            # ester carbonyl is expressed substitutively as oxo + alkoxy on the
            # acid parent ('5-butoxy-2-methyl-5-oxopentanoic acid'), never as
            # R-oxycarbonyl (double-counts the carbonyl carbon = names a
            # one-carbon-longer homolog). The oxo+alkoxy pair is emitted by the
            # polyfunctional group-splitting branch (OPSIN-RT gated).
            return None
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

    # Wave2 T3b (P-63.6): the PIN prefix is the ACID-STEM form built on the
    # parent hydride ('methanesulfinyl', BB 18284/28150-verbatim family;
    # 'methylsulfinyl' is the non-PIN alternative). The side classifier also
    # supplies the constitution guard the old carbons_only count lacked (a
    # branched or hetero-bearing R' was silently flattened to a linear alkyl
    # count). None -> the caller drops to its own fail-closed handling.
    return _acid_stem_oxide_prefix(mol, sub_carbon, sulfur_idx, 'sulfinyl')


def _acid_stem_oxide_prefix(
    mol, sub_carbon: int, sulfur_idx: int, oxide_kind: str,
) -> Optional[str]:
    """Shared Wave2 T3b builder: '{parent-hydride-stem}sulfinyl/sulfonyl'
    (methanesulfinyl / benzenesulfonyl / cyclohexanesulfinyl) with the
    _classify_oxide_side constitution guard. None when the R' side is not an
    honestly-nameable shape."""
    from ..rules.sulfur import (
        _classify_oxide_side,
        _acid_stem_unsaturated_oxide_prefix,
    )
    side = _classify_oxide_side(mol, sub_carbon, sulfur_idx)
    if side is None:
        # BP-3 C2-B (P-63.6): unsaturated / branched / hetero arm —
        # _classify_oxide_side only handles saturated-linear / benzene /
        # cycloalkane. Build the acid-stem PIN ('prop-2-ene-1-sulfinyl') by
        # naming the arm-derived sulfinic/sulfonic acid with the full engine and
        # rewriting the suffix. Returns None (fail closed) on anything that does
        # not name as a clean '... sulfinic/sulfonic acid'.
        return _acid_stem_unsaturated_oxide_prefix(
            mol, sub_carbon, sulfur_idx, oxide_kind
        )
    stem, _kind, _atoms = side
    return f"{stem}{oxide_kind}"


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

    # Wave2 T3b (P-63.6): ACID-STEM PIN form + constitution guard — see
    # get_sulfinyl_prefix ('2-(methanesulfonyl)ethan-1-ol', BB 28150 verbatim).
    return _acid_stem_oxide_prefix(mol, sub_carbon, sulfur_idx, 'sulfonyl')


def _acyl_on_chalcogen_name(mol, acyl_c: int, chalcogen_idx: int) -> Optional[str]:
    """Name the R-CO- group bonded to a chalcogen as an acyl prefix stem
    (P-35.5.1 'acetylsulfanyl', BB 18128).

    v1 scope (None outside it, fail-closed): the acyl carbon carries exactly
    one terminal =O, the single chalcogen link, and at most one R; R is either
    an unsubstituted LINEAR acyclic all-carbon chain -> retained formyl/acetyl
    or systematic alkanoyl via rules/acid_halides._build_acyl_name
    (P-65.1.7.2.1) — the single acyl-name authority, NOT a local table — or an
    unsubstituted phenyl -> retained 'benzoyl' (P-66.6.1.2).
    """
    atom = mol.GetAtomWithIdx(acyl_c)
    if atom.GetSymbol() != "C" or atom.IsInRing():
        return None
    oxo = []
    r_side = []
    for b in atom.GetBonds():
        other = b.GetOtherAtom(atom)
        if other.GetIdx() == chalcogen_idx:
            if b.GetBondTypeAsDouble() != 1.0:
                return None
            continue
        if (
            b.GetBondTypeAsDouble() == 2.0
            and other.GetSymbol() == "O"
            and other.GetDegree() == 1
        ):
            oxo.append(other.GetIdx())
            continue
        if b.GetBondTypeAsDouble() in (1.0, 1.5) and other.GetSymbol() == "C":
            r_side.append(other.GetIdx())
            continue
        return None  # =S, N, second O-function, charged O, ... -> fail-closed
    if len(oxo) != 1 or len(r_side) > 1:
        return None

    from ..rules.acid_halides import _build_acyl_name  # Pattern-S3 lazy import

    if not r_side:
        return _build_acyl_name(1)  # H-CO- -> retained 'formyl'

    r0 = mol.GetAtomWithIdx(r_side[0])
    if r0.GetIsAromatic():
        # 'benzoyl' iff plain UNSUBSTITUTED benzene (P-66.6.1.2); substituted
        # aroyl assembly is out of v1 -> None.
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if r0.GetIdx() not in ring or len(ring) != 6:
                continue
            if not all(
                mol.GetAtomWithIdx(i).GetIsAromatic()
                and mol.GetAtomWithIdx(i).GetSymbol() == "C"
                for i in ring
            ):
                return None
            for i in ring:
                for nb in mol.GetAtomWithIdx(i).GetNeighbors():
                    if (
                        nb.GetIdx() not in ring
                        and nb.GetIdx() != acyl_c
                        and nb.GetAtomicNum() > 1
                    ):
                        return None
            return "benzoyl"
        return None

    # Linear unsubstituted alkanoyl walk (acyclic, pure C, unbranched,
    # all single bonds). n counts the acyl carbon + the chain carbons.
    n = 1
    prev, cur = acyl_c, r_side[0]
    while True:
        a = mol.GetAtomWithIdx(cur)
        if a.GetSymbol() != "C" or a.GetIsAromatic() or a.IsInRing():
            return None
        nxt = []
        for b in a.GetBonds():
            other = b.GetOtherAtom(a)
            if other.GetIdx() == prev:
                if b.GetBondTypeAsDouble() != 1.0:
                    return None
                continue
            if other.GetAtomicNum() <= 1:
                continue
            if other.GetSymbol() != "C" or b.GetBondTypeAsDouble() != 1.0:
                return None
            nxt.append(other.GetIdx())
        n += 1
        if not nxt:
            break
        if len(nxt) > 1:
            return None  # branched -> v1 fail-closed
        prev, cur = cur, nxt[0]
    return _build_acyl_name(n)


def get_sulfanyl_prefix(
    mol,
    thioether_atoms: tuple,
    principal_chain: Optional[List[int]] = None,
    suffix: str = "sulfanyl",
) -> Optional[str]:
    """Generate (alkyl)sulfanyl/selanyl/tellanyl prefix for a chalcogen ether as
    non-principal group (IUPAC P-63.2.5 / P-63.6).

    IUPAC P-63.2.5: ``R-S-R'`` (or ``R-Se-R'`` / ``R-Te-R'``) when not the
    principal group is expressed as an (alkyl)chalcogenyl prefix on the parent
    chain. The algorithm is chalcogen-agnostic (it walks the C neighbours of the
    chalcogen atom at ``thioether_atoms[0]``); only the suffix differs:
    ``sulfanyl`` (S), ``selanyl`` (Se), ``tellanyl`` (Te).

    SMARTS ``[SX2]([#6])[#6]`` / ``[SeX2]([#6])[#6]`` / ``[TeX2]([#6])[#6]``
    each match (chalcogen, C1, C2).

    Lifted from rules/polyfunctional._get_sulfanyl_prefix verbatim with a NEW
    None-guard at function entry per Phase 160.1 RESEARCH §4. BBR-PERC (169.7)
    added the ``suffix`` param (default "sulfanyl" → backward-compatible).

    Args:
        mol: RDKit Mol object.
        thioether_atoms: Atom indices from the chalcogen-ether SMARTS match.
        principal_chain: Atom indices of the principal chain; may be None.
        suffix: chalcogen prefix stem ("sulfanyl" | "selanyl" | "tellanyl").

    Returns:
        Compound prefix string (e.g., ``"methylsulfanyl"`` / ``"methylselanyl"``),
        or None on failure.
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

    # W2F-P2 (P-35.5.1, BB 18128): acyl-on-chalcogen. When the substituent-side
    # carbon bears a =O/=S it is an ACYL group (thioester S-side); the alkyl
    # counter below would collapse the carbonyl into the alkyl count
    # ('ethylsulfanyl' for CH3-CO-S- — a DIFFERENT molecule). Name it as the
    # acyl prefix ('acetylsulfanyl') or fail closed.
    _sub_atom = mol.GetAtomWithIdx(sub_carbon)
    _dbl_chalc = [
        b.GetOtherAtom(_sub_atom).GetSymbol()
        for b in _sub_atom.GetBonds()
        if b.GetBondTypeAsDouble() == 2.0
        and b.GetOtherAtom(_sub_atom).GetSymbol() in ("O", "S")
    ]
    if _dbl_chalc:
        if suffix != "sulfanyl" or "S" in _dbl_chalc:
            # v1: acylselanyl/acyltellanyl spellings and thioacyl (C=S) stems
            # are not OPSIN-verified -> refuse rather than emit a collapsed
            # alkyl name (fail-closed, accuracy-first).
            return None
        acyl = _acyl_on_chalcogen_name(mol, sub_carbon, sulfur_idx)
        if not acyl:
            return None
        return f"{acyl}{suffix}"

    # BP-3 C2-A (P-63.2.5 / P-63.1.5): name the arm RECURSIVELY as a substituent
    # group instead of by carbon-count. The old get_alkyl_name(carbon_count)
    # collapsed every arm to a saturated linear alkyl — allyl -> 'propyl',
    # isobutyl -> 'butyl', 2-hydroxyethyl -> 'ethyl', phenyl -> 'hexyl' — all
    # constitution-wrong (SELF-01-suppressed). name_substituent_fragment is the
    # project's centralized substituent namer and returns None on any shape it
    # cannot name, so this stays fail-closed.
    from .substituent_naming import name_substituent_fragment
    from .naming_utils import is_complex_substituent, apply_enclosing_marks

    arm_atoms = _collect_fragment_atoms(mol, sub_carbon, {sulfur_idx})
    if not arm_atoms:
        return None
    arm = name_substituent_fragment(
        mol, list(arm_atoms), sub_carbon, list(chain_set | {sulfur_idx})
    )
    if not arm:
        return None  # fail closed: un-nameable arm -> caller drops -> SELF-01

    # P-16.3.3: a COMPLEX arm (locants / compound: prop-2-en-1-yl, 2-methylpropyl,
    # 2-hydroxyethyl, penta-1,4-dien-3-yl) is enclosed BEFORE the chalcogen stem
    # (BB 25713 '(prop-2-en-1-yl)cyclohexane'; BB 27836 '[(penta-1,4-dien-3-yl)
    # sulfanyl]cyclobutane'). A SIMPLE arm (methyl/ethyl/propyl/phenyl/cyclopentyl)
    # stays bare -> 'methylsulfanyl' (byte-identical to the old plain-alkyl output;
    # BB 27828). The caller (format_fg_prefix / composer) supplies the OUTER
    # enclosure + locant and escalates '(...)sulfanyl' -> '[(...)sulfanyl]'.
    if is_complex_substituent(arm):
        arm = apply_enclosing_marks(arm, 0)
    return f"{arm}{suffix}"


def get_alkoxysulfinyl_prefix(
    mol,
    s_idx: int,
    attach_idx: int,
) -> Optional[str]:
    """Build the O-alkyl (alkoxy)sulfinyl compound prefix (P-16.3.3 / P-35.5.1).

    For an S(=O) centre attached to the parent through ``attach_idx`` and
    carrying exactly ONE ``-O-alkyl`` arm and the single ``=O`` oxo (no C
    neighbours), name the O-alkyl arm as ``<alkyl>oxy`` and concatenate the
    additive ``sulfinyl`` stem -> ``ethoxysulfinyl``. The compound prefix is
    returned already enclosed in parentheses per P-16.5.1.1
    (``(ethoxysulfinyl)``); the caller's N-substituent wrapper leaves a
    balanced single-paren name unchanged.

    The C-linked ``R-S(=O)-R'`` case is handled by ``get_sulfinyl_prefix``
    (which requires two C-neighbours on S and declines here — S has an O and
    the attachment N, zero C). This assembler is the O-linked complement.

    Args:
        mol: RDKit Mol object.
        s_idx: Atom index of the sulfinyl sulfur.
        attach_idx: Atom index of the parent atom the sulfur is bonded to
            (excluded from the arm walk — e.g. the aniline nitrogen).

    Returns:
        ``"(ethoxysulfinyl)"`` etc., or None (fail-closed) if the shape is not
        exactly one O-alkyl arm + one =O oxo + the single attachment.
    """
    s_atom = mol.GetAtomWithIdx(s_idx)
    if s_atom.GetSymbol() != "S":
        return None
    # S(=O) sulfinyl: exactly one double-bonded terminal oxo.
    oxo = []
    ether_o = []
    other = []
    for nbr in s_atom.GetNeighbors():
        nidx = nbr.GetIdx()
        bond = mol.GetBondBetweenAtoms(s_idx, nidx)
        if nidx == attach_idx:
            continue
        if (nbr.GetSymbol() == "O" and nbr.GetDegree() == 1
                and bond is not None
                and bond.GetBondTypeAsDouble() == 2.0):
            oxo.append(nidx)
        elif (nbr.GetSymbol() == "O" and nbr.GetDegree() == 2
                and bond is not None
                and bond.GetBondTypeAsDouble() == 1.0):
            ether_o.append(nidx)
        else:
            other.append(nidx)
    # Fail-closed: exactly one =O oxo, exactly one -O-alkyl arm, nothing else.
    if len(oxo) != 1 or len(ether_o) != 1 or other:
        return None
    o_idx = ether_o[0]
    o_atom = mol.GetAtomWithIdx(o_idx)
    # The alkyl carbon on the far side of the ether oxygen.
    alkyl_c = [n.GetIdx() for n in o_atom.GetNeighbors()
               if n.GetIdx() != s_idx]
    if len(alkyl_c) != 1:
        return None
    # Reuse the ether alkoxy namer via its (O, C1, C2) tuple contract; pass the
    # ether O and both its carbons (S side, alkyl side). With no principal
    # chain it selects the smaller (alkyl) fragment automatically.
    alkoxy = get_alkoxy_prefix(mol, (o_idx, s_idx, alkyl_c[0]), None)
    if not alkoxy:
        return None
    return f"({alkoxy}sulfinyl)"


def get_phosphoryl_prefix(
    mol,
    p_idx: int,
    attach_idx: int,
) -> Optional[str]:
    """Build the [(...)phosphoryl] additive compound prefix (P-16.3.3 / P-16.3.6).

    For a ``P(=O)`` centre attached to the parent through ``attach_idx``, name
    every remaining (non-oxo, non-attachment) P substituent as a substitutive
    prefix, apply the multiplicative disambiguation (``bis(sulfanyl)`` for two
    ``-SH`` arms), and concatenate the additive ``phosphoryl`` stem ->
    ``bis(sulfanyl)phosphoryl``. Returned WITHOUT an outer enclosure; the
    caller's N-substituent wrapper escalates the parens-bearing name to square
    brackets per P-16.5.1.1 -> ``[bis(sulfanyl)phosphoryl]``.

    Only ``-SH`` arms are recognised today (the sole verified class); any other
    arm shape fails closed so a partial/ambiguous name never leaks.

    Args:
        mol: RDKit Mol object.
        p_idx: Atom index of the phosphoryl phosphorus.
        attach_idx: Atom index of the parent atom the phosphorus is bonded to.

    Returns:
        ``"bis(sulfanyl)phosphoryl"`` etc., or None (fail-closed).
    """
    from .naming_utils import get_multiplier_prefix, apply_enclosing_marks

    p_atom = mol.GetAtomWithIdx(p_idx)
    if p_atom.GetSymbol() != "P":
        return None
    oxo = []
    arms: List[int] = []
    for nbr in p_atom.GetNeighbors():
        nidx = nbr.GetIdx()
        if nidx == attach_idx:
            continue
        bond = mol.GetBondBetweenAtoms(p_idx, nidx)
        if (nbr.GetSymbol() == "O" and nbr.GetDegree() == 1
                and bond is not None
                and bond.GetBondTypeAsDouble() == 2.0):
            oxo.append(nidx)
        else:
            arms.append(nidx)
    # Fail-closed: exactly one P=O oxo required for the 'phosphoryl' core.
    if len(oxo) != 1 or not arms:
        return None
    # Name each remaining arm as a substitutive prefix. Only bare -SH (sulfanyl)
    # is honestly nameable here; anything else -> fail closed.
    arm_names: List[str] = []
    for a_idx in arms:
        a_atom = mol.GetAtomWithIdx(a_idx)
        if (a_atom.GetSymbol() == "S" and a_atom.GetDegree() == 1
                and a_atom.GetTotalNumHs() == 1
                and a_atom.GetFormalCharge() == 0):
            arm_names.append("sulfanyl")
        else:
            return None
    if not arm_names:
        return None
    # Group identical arms with the P-16.3.6 multiplicative prefix. All-identical
    # is the only verified case; distinct arms would need locants -> fail closed.
    if len(set(arm_names)) != 1:
        return None
    name = arm_names[0]
    count = len(arm_names)
    if count == 1:
        core = f"{name}phosphoryl"
    else:
        mult = get_multiplier_prefix(count, name)
        core = f"{mult}{apply_enclosing_marks(name, 0)}phosphoryl"
    return core


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

    # C4d: a carbon COUNT cannot describe this branch, and the ALKOXY_NAMES
    # lookup it fed spelled five of six real Boc-family branches as a different
    # group: tert-butyl read as 4 carbons became n-'butoxy', benzyl as 7 became
    # 'heptyloxy' (a straight chain for a ring), phenyl as 6 became 'hexyloxy',
    # allyl as 3 became saturated 'propoxy' (losing the double bond), and
    # propan-2-yl became n-'propoxy'. Only methyl was right, and only by
    # coincidence. PERCEIVE the branch instead, through the same BB-cited
    # primitives the rest of the composed-prefix system uses -- P-63.2.2.2
    # tabulates the morphology (BB 27667-27691) and composed_alkoxy_prefix
    # spells it, including the retained contraction tert-butoxy (BB 27679,
    # "not tert-butyloxy").
    if not visited:
        return None
    from .substituent_enumerator import (
        composed_alkoxy_prefix,
        composed_prefix_organyl_name,
    )
    token = composed_prefix_organyl_name(mol, sorted(visited), alkyl_atom)
    if not token:
        return None
    return composed_alkoxy_prefix(token)


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

    ⚠ **A carbon COUNT only names an UNBRANCHED ACYCLIC chain.**
    ``get_alkyl_name(n)`` spells the straight-chain stem, so handing it the
    count of a branched or cyclic fragment renames the molecule:
    ``-C(CH3)3`` (4 carbons) came back ``"butyl"`` and phenyl (6) came back
    ``"hexyl"`` — different constitutions, found while building the R3
    thiourea prefix (this is the count-anti-pattern class of
    ``). The count
    path is now gated on the fragment actually BEING an unbranched SATURATED
    acyclic chain attached at one of its termini — a walk over the real bonds,
    not a tally — and everything else is routed to the substituent chokepoint
    that already produces located PINs (``tert-butyl``, ``phenyl``,
    ``propan-2-yl``). Un-nameable there -> None (fail closed).
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

    # Structure proof that the count is a faithful description. This used to be
    # a local walk that checked rings, branching and the attachment terminus but
    # NEVER BOND ORDER, so an allyl / propargyl / butenyl fragment -- an
    # unbranched acyclic all-carbon chain attached at a terminus -- passed and
    # was spelled as the saturated alkane stem: 'OC(=O)CCNC(=O)NCCC' (propyl),
    # '...NCC=C' (allyl) and '...NCC#C' (propargyl) all emitted the identical
    # '3-[(propylcarbamoyl)amino]propanoic acid'. Three molecules, one name --
    # no function of (fragment, carbon-count) can be right there.
    #
    # ``fragment_is_linear_terminal_alkyl`` is the shared primitive that already
    # asks exactly this question ("is get_alkyl_name(n) an HONEST name for this
    # fragment?") and it DOES check bond order, so route through it rather than
    # grow a second guard list that can drift out of step with the first.
    from .substituent_naming import fragment_is_linear_terminal_alkyl

    if fragment_is_linear_terminal_alkyl(mol, list(visited), alkyl_atom):
        carbon_count = _count_fragment_atoms(
            mol, alkyl_atom, exclude, carbons_only=True
        )
        if carbon_count == 0:
            return None
        try:
            return get_alkyl_name(carbon_count)
        except (ValueError, KeyError):
            return None

    # Branched or cyclic: name it properly instead of counting it.
    try:
        from .substituent_enumerator import name_substituent as _ns
        prefix = _ns(mol, set(visited), alkyl_atom)
    except Exception:
        return None
    if prefix and prefix != "substituent":
        return prefix
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

    # P-16.3.4 (BlueBookV2.md:7085) "Parentheses (round brackets) ... are used
    # to enclose multiplied components that are: ... (b) simple substituent
    # prefixes modified by 'ene' and 'yne' endings and that have locants"
    # (:7104, example 'di(prop-1-en-2-yl)' (preferred prefix)). Until the bond-
    # order guard above was fixed, every name reaching this interpolation was a
    # bare saturated stem ('propyl') that needs no marks; now that an alkenyl /
    # alkynyl / located prefix can arrive, it must be enclosed or the locant
    # runs into the 'carbamoyl' that follows it. enclose_if_compound is a no-op
    # for a simple stem, so every pre-existing name stays byte-identical.
    return f"N-{enclose_if_compound(alkyl_name)}carbamoyl"


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

    # 'di', NOT 'bis': P-16.3.5 (BlueBookV2.md:7104) reserves 'bis'/'tris' for
    # "(a) compound or complex (i.e. substituted) prefixes" -- every clause of
    # it is gated on the component being SUBSTITUTED, and an unsubstituted
    # alkenyl prefix is not. P-16.3.4(b) (:7104) supplies the parentheses, and
    # BB :38230 carries the PIN '1,1-dimethyl-3,4-di(prop-1-en-2-yl)germolane'.
    if alkyl1 == alkyl2:
        return f"N,N-di{enclose_if_compound(alkyl1)}carbamoyl"

    # Alphabetize ascending: N-{lower}-N-{higher}carbamoyl
    if alpha_sort_key(alkyl1) < alpha_sort_key(alkyl2):
        first, second = alkyl1, alkyl2
    else:
        first, second = alkyl2, alkyl1
    return (f"N-{enclose_if_compound(first)}"
            f"-N-{enclose_if_compound(second)}carbamoyl")


def _urea_distal_n(mol, urea_atoms, principal_chain) -> Optional[int]:
    """P-66.1.6.1.1.3 proximal/distal disambiguation for a urea substituent.

    A urea substituent ``parent-N(H?)-C(=O)-N(H?)(R?)`` attaches to the parent
    through the PROXIMAL N; the DISTAL N carries the substituents that decorate
    the carbamoyl acyl name (``(R-carbamoyl)amino``). The urea SMARTS match
    tuple ``[NX3][CX3](=O)[NX3]`` is NOT orientation-stable, so identify the
    distal N structurally:

      * principal_chain given -> proximal N is the one bonded into the chain;
      * else the distal N is the pure-substituent side (its external branches
        are all nameable simple substituents) while the parent side is not.

    Returns the distal N atom index, or None when the orientation is ambiguous
    (fail closed).
    """
    urea_set = set(urea_atoms)
    n_idxs = [i for i in urea_atoms
              if mol.GetAtomWithIdx(i).GetSymbol() == 'N']
    if len(n_idxs) != 2:
        return None
    na, nb = n_idxs

    def _external_heavy(n_idx):
        return [nbr.GetIdx()
                for nbr in mol.GetAtomWithIdx(n_idx).GetNeighbors()
                if nbr.GetIdx() not in urea_set and nbr.GetAtomicNum() > 1]

    # Cue 1: principal chain -> the proximal N touches the chain.
    if principal_chain:
        cs = set(principal_chain)
        na_prox = na in cs or any(x in cs for x in _external_heavy(na))
        nb_prox = nb in cs or any(x in cs for x in _external_heavy(nb))
        if na_prox and not nb_prox:
            return nb
        if nb_prox and not na_prox:
            return na

    # Cue 2: the distal N is a pure-substituent side (all external branches are
    # nameable simple substituents); the proximal/parent side is not. An
    # unsubstituted distal N (no external heavy neighbours) is vacuously pure.
    def _pure_side(n_idx):
        ext = _external_heavy(n_idx)
        excl = urea_set | {n_idx}
        for x in ext:
            if _name_alkyl_branch_from_atom(
                    mol, x, exclude=excl | (set(ext) - {x})) is None:
                return False
        return True

    na_pure = _pure_side(na)
    nb_pure = _pure_side(nb)
    if na_pure and not nb_pure:
        return na
    if nb_pure and not na_pure:
        return nb

    # Cue 3 (degenerate no-parent-context tie-break, e.g. a bare urea molecule
    # where both sides are nameable): the -CO-NH2 end is the carbamoyl terminus
    # (distal), the substituted N is treated as the parent-attachment
    # (proximal). Both-substituted -> genuinely ambiguous -> None (fail closed).
    # An unsubstituted N is always pure, so "both not-pure" implies both
    # substituted and falls through to None.
    na_ext = _external_heavy(na)
    nb_ext = _external_heavy(nb)
    if not na_ext and nb_ext:
        return na
    if not nb_ext and na_ext:
        return nb
    if not na_ext and not nb_ext:
        return na  # both unsubstituted -> carbamoylamino either way
    return None


def _carbamoyl_with_distal_substituents(
    mol, distal_n: int, urea_set: Set[int], stem: str = "carbamoyl"
) -> Optional[str]:
    """Name the distal-N substituents into the carbamoyl acyl (P-66.1.6.1.1.3).

    Unsubstituted distal N -> ``"carbamoyl"``. Substituted -> the substituents
    cited with plain di-/tri- multipliers, alphanumerically ordered, prefixed
    to ``carbamoyl`` (``methylcarbamoyl`` / ``dimethylcarbamoyl``; the enclosed
    carbamoyl form drops N-locants per BB 33354). Any un-nameable substituent
    -> None (fail closed).

    ``stem`` selects the acyl: ``"carbamoyl"`` for urea (P-66.1.6.1.1.3) or
    ``"carbamothioyl"`` for thiourea (P-66.1.6.1.3.3 :33489 lists the
    unsubstituted ``carbamothioylamino``; ``carbamothioyl`` itself is a
    retained preferred prefix per P-65.2.1.5 :30869 and P-66.1.4.4 :33198,
    *"carbamothioyl (not thiocarbamoyl) for -CS-NH2"*). Defaulted so the urea
    caller is byte-identical.
    """
    from collections import Counter
    from .naming_utils import get_multiplier_prefix

    ext = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(distal_n).GetNeighbors()
           if nbr.GetIdx() not in urea_set and nbr.GetAtomicNum() > 1]
    if not ext:
        return stem
    names: List[str] = []
    for x in ext:
        excl = urea_set | {distal_n} | (set(ext) - {x})
        nm = _name_alkyl_branch_from_atom(mol, x, exclude=excl)
        if not nm:
            return None
        names.append(nm)
    counts = Counter(names)
    parts = []
    for base in sorted(counts, key=alpha_sort_key):
        c = counts[base]
        parts.append(base if c == 1 else f"{get_multiplier_prefix(c, base)}{base}")
    return "".join(parts) + stem


def get_n_substituted_carbamoylamino_prefix(
    mol, atoms: tuple, principal_chain: Optional[List[int]] = None
) -> Optional[str]:
    """P-66.1.6.1.1.3: ``-NH-CO-NR2`` substituent -> ``(carbamoyl-decorated)amino``.

    Replaces the fixed ``PREFIX_FORMS['urea'] = 'carbamoylamino'`` with a
    dynamic builder that enumerates the distal N's substituents:

      * unsubstituted distal N -> ``"carbamoylamino"`` (BB preselected prefix,
        NOT 'ureido'/'3-methylureido');
      * substituted distal N -> ``"(methylcarbamoyl)amino"`` /
        ``"(dimethylcarbamoyl)amino"`` (compound-prefix enclosure, P-16.5.1.1);
      * un-nameable distal substituent or ambiguous orientation -> None
        (fail closed).
    """
    if not atoms:
        return None
    urea_set = set(atoms)
    distal_n = _urea_distal_n(mol, atoms, principal_chain)
    if distal_n is None:
        return None
    core = _carbamoyl_with_distal_substituents(mol, distal_n, urea_set)
    if core is None:
        return None
    if core == "carbamoyl":
        return "carbamoylamino"
    return f"({core})amino"


def _chalcogen_of_thiourea_match(mol, atoms: tuple) -> Optional[str]:
    """The chalcogen element symbol of an N-C(=X)-N match, or None."""
    for i in atoms:
        sym = mol.GetAtomWithIdx(i).GetSymbol()
        if sym in ("S", "Se", "Te"):
            return sym
    return None


def get_n_substituted_carbamothioylamino_prefix(
    mol, atoms: tuple, principal_chain: Optional[List[int]] = None
) -> Optional[str]:
    """P-66.1.6.1.3.3: ``-NH-CS-NR2`` substituent -> ``(carbamothioyl…)amino``.

    Residue R3 root cause. This row used to be the STATIC lookup
    ``PREFIX_FORMS['thiourea']`` -> ``"carbamothioylamino"``, returned
    unconditionally and ignoring ``atoms`` entirely. ``carbamothioylamino`` is
    the MONOVALENT group ``H2N-CS-NH-`` (P-66.1.6.1.3.3, BlueBookV2.md :33489).
    When the 4-atom core bridges two parts of the molecule — both nitrogens
    substituted — the static string named a two-attachment bridge with a
    one-attachment prefix: everything past the distal nitrogen was orphaned and
    re-attached elsewhere, and the core was consumed a second time from the
    other direction. Measured on ``CC(C)(C)NC(=S)NC1CCCCC1``: 31 calls, 31
    ``carbamothioylamino`` returns, output
    ``1-(2-(carbamothioylamino)-2-carbamothioylamino-2-methylpropyl)cyclohexane``
    — one thiourea unit spelled twice, and a ring->N bond rendered ring->C.

    This is now the exact mirror of the oxo sibling
    ``get_n_substituted_carbamoylamino_prefix`` (P-66.1.6.1.1.3), which never
    had the defect because it resolves the distal nitrogen structurally and
    returns None when the orientation is ambiguous — measured 12/12 None on the
    urea analogue of the same molecule, which then correctly re-parents onto
    urea.

      * unsubstituted distal N -> ``"carbamothioylamino"``
        (the enumerated preferred prefix, :33489; BB example :33501
        ``3-(carbamothioylamino)propanoic acid (PIN)``);
      * substituted distal N -> ``"(methylcarbamothioyl)amino"`` /
        ``"(dimethylcarbamothioyl)amino"``. P-66.1.6.1.3.3 enumerates NO
        substituted-distal-N row — the form is derived across the
        P-66.1.6.1.3.1 chalcogen-replacement relationship (:33439) from the oxo
        `(PIN)` example ``2-[(methylcarbamoyl)amino]naphthalene-1-carboxylic
        acid`` (:33354);
      * both nitrogens substituted / ambiguous orientation -> None. The caller
        falls through and the molecule re-parents onto the retained thiourea
        (P-44.1.1 :18875 over P-41 Table 4.1: amides class 11 :18184 outrank
        carbon rings class 40 :18216).

    Se/Te analogues return None. P-66.1.6.1.3.3 enumerates only the sulfur
    prefix, and no ``carbamoselenoyl``/``carbamotelluroyl`` spelling appears
    anywhere in the Blue Book, so inventing one would risk a wrong name; the
    Se/Te RETAINED PARENT is built instead (P-66.1.6.1.3.1 :33451
    ``N-(butan-2-yl)selenourea (PIN)``).
    """
    if not atoms:
        return None
    if _chalcogen_of_thiourea_match(mol, atoms) != "S":
        return None
    core_set = set(atoms)
    distal_n = _urea_distal_n(mol, atoms, principal_chain)
    if distal_n is None:
        return None
    core = _carbamoyl_with_distal_substituents(
        mol, distal_n, core_set, stem="carbamothioyl"
    )
    if core is None:
        return None
    if core == "carbamothioyl":
        return "carbamothioylamino"
    return f"({core})amino"


def _compute_branch_b_carbamoyloxy_name(
    mol,
    carbamate_atoms: tuple,
) -> Optional[str]:
    """Compute Branch B carbamoyloxy name with N-substitution per IUPAC P-66.6.4.

    For ``-OC(=O)NR1R2`` substituents (attached through the ester-O):

      * ``-OC(=O)NH2``           → ``"carbamoyloxy"``
      * ``-OC(=O)NHCH3``         → ``"(N-methylcarbamoyl)oxy"``
      * ``-OC(=O)NHC2H5``        → ``"(N-ethylcarbamoyl)oxy"``
      * ``-OC(=O)N(CH3)2``       → ``"(N,N-dimethylcarbamoyl)oxy"``
      * ``-OC(=O)N(CH3)(C2H5)``  → ``"(N-ethyl-N-methylcarbamoyl)oxy"``  (alphabetized)

    Mirrors ``get_n_alkyl_carbamoyl_prefix`` + ``get_n_n_dialkyl_carbamoyl_prefix``
    conventions (uses explicit ``N-`` locants) then appends ``oxy``.

    Returns None for heteroatom-bearing N-substituents (caller falls through).
    """
    if len(carbamate_atoms) < 5:
        return None
    amide_n, carbonyl_c, carbonyl_o, _ester_o, _alkyl_c = carbamate_atoms[:5]

    # Find N's non-H, non-carbonyl-C neighbours (the N-substituents).
    n_atom = mol.GetAtomWithIdx(amide_n)
    n_alkyl_neighbors: List[int] = []
    for nbr in n_atom.GetNeighbors():
        if nbr.GetIdx() == carbonyl_c:
            continue
        if nbr.GetAtomicNum() == 1:
            continue
        n_alkyl_neighbors.append(nbr.GetIdx())

    # Unsubstituted: -OC(=O)NH2 → carbamoyloxy
    if not n_alkyl_neighbors:
        return "carbamoyloxy"

    excl_base = {carbonyl_c, carbonyl_o, amide_n}

    # Mono-N-substituted: -OC(=O)NHR → (N-Rcarbamoyl)oxy
    if len(n_alkyl_neighbors) == 1:
        alkyl_name = _name_alkyl_branch_from_atom(
            mol, n_alkyl_neighbors[0], exclude=excl_base
        )
        if not alkyl_name:
            return None
        # Same P-16.3.4 / P-16.3.5 reasoning as get_n_alkyl_carbamoyl_prefix.
        return f"(N-{enclose_if_compound(alkyl_name)}carbamoyl)oxy"

    # Di-N-substituted: -OC(=O)NR1R2 → (N,N-(R1)(R2)carbamoyl)oxy
    if len(n_alkyl_neighbors) == 2:
        a1, a2 = n_alkyl_neighbors
        alkyl1 = _name_alkyl_branch_from_atom(mol, a1, exclude=excl_base | {a2})
        alkyl2 = _name_alkyl_branch_from_atom(mol, a2, exclude=excl_base | {a1})
        if not alkyl1 or not alkyl2:
            return None
        if alkyl1 == alkyl2:
            return f"(N,N-di{enclose_if_compound(alkyl1)}carbamoyl)oxy"
        if alpha_sort_key(alkyl1) < alpha_sort_key(alkyl2):
            first, second = alkyl1, alkyl2
        else:
            first, second = alkyl2, alkyl1
        return (f"(N-{enclose_if_compound(first)}"
                f"-N-{enclose_if_compound(second)}carbamoyl)oxy")

    return None


def get_carbamoyloxy_prefix(
    mol,
    carbamate_atoms: tuple,
    principal_chain: Optional[List[int]] = None,
) -> Optional[str]:
    """Generate carbamate prefix per IUPAC P-66.6.4.

    The carbamate ``-NHC(=O)O-`` has TWO orientation forms:

      * **Branch A**: connects through N → ``-NHC(=O)OCH3`` → ``"(methoxycarbonyl)amino"``
      * **Branch B**: connects through O → ``-OC(=O)NH2`` → ``"carbamoyloxy"`` (N-substituted
        variants per ``_compute_branch_b_carbamoyloxy_name``: ``(N-methylcarbamoyl)oxy`` etc.)

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
            # Branch B: -OC(=O)NR1R2 → carbamoyloxy / (N-Rcarbamoyl)oxy
            return _compute_branch_b_carbamoyloxy_name(mol, carbamate_atoms)
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
    # BBR-PERC (169.7): Se/Te ether analogues → (alkyl)selanyl/tellanyl (P-63.6).
    if fg_name == "selenoether":
        return get_sulfanyl_prefix(mol, atoms, principal_chain, suffix="selanyl")
    if fg_name == "telluroether":
        return get_sulfanyl_prefix(mol, atoms, principal_chain, suffix="tellanyl")

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
        # P-66.1.6.1.1.3 (BlueBookV2.md:33338,33354): dynamic N-substituted
        # urea prefix. Distal-N substituents decorate the carbamoyl acyl:
        # -NH-CO-NH2 -> carbamoylamino; -NH-CO-NHMe -> (methylcarbamoyl)amino;
        # -NH-CO-NMe2 -> (dimethylcarbamoyl)amino. Un-nameable/ambiguous -> None.
        return get_n_substituted_carbamoylamino_prefix(mol, atoms, principal_chain)
    if fg_name == "thiourea":
        # R3: was a STATIC, unparametrised PREFIX_FORMS lookup that ignored
        # `atoms` and so named a BRIDGING N-CS-N core with the monovalent
        # prefix. Now the exact mirror of the `urea` row two lines up.
        return get_n_substituted_carbamothioylamino_prefix(
            mol, atoms, principal_chain
        )
    if fg_name == "isocyanate":
        return PREFIX_FORMS.get("isocyanate")  # "isocyanato"
    if fg_name == "isothiocyanate":
        return PREFIX_FORMS.get("isothiocyanate")  # "isothiocyanato"
    # R8b (P-66.4.1.3.1): amidine as non-principal substituent → "carbamimidoyl"
    if fg_name == "amidine":
        return PREFIX_FORMS.get("amidine")  # "carbamimidoyl"

    # Wave2 T2b (P-68.3.1.1.1.5): UNSUBSTITUTED -NH-OH as a non-principal
    # substituent → 'hydroxyamino' (preselected prefix; BB PIN
    # '4-(hydroxyamino)phenol'). Restricted to the unsubstituted form: the
    # matched N must carry exactly OH + one C + one H. N-substituted
    # hydroxylamines (R-N(CH3)-OH) return None → the static
    # PREFIX_FORMS['hydroxylamine'] stays None → the group drops and the
    # validity gate fails closed (their PIN needs a composed
    # [hydroxy(methyl)amino] builder, not built here).
    if fg_name == "hydroxylamine":
        for _i in atoms:
            _a = mol.GetAtomWithIdx(_i)
            if _a.GetSymbol() == 'N':
                if _a.GetDegree() == 2 and _a.GetTotalNumHs() == 1:
                    return "hydroxyamino"
                return None
        return None

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
    "selenoether", "telluroether",  # BBR-PERC (169.7)
    "carbamate", "urea", "thiourea",
    "isocyanate", "isothiocyanate",
    "amidine",  # R8b (P-66.4.1.3.1): carbamimidoyl prefix
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
    allow_higher_sulfur: bool = False,
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
            ):
                # The carbamate 4-atom core (amide_N, carbonyl_C, carbonyl_O,
                # ester_O) must live inside the substituent fragment. The
                # trailing SMARTS atom (alkyl_C = match[4]) is the parent
                # attachment point, which may be EITHER inside or outside the
                # fragment. Any extra atoms in the fragment beyond the core
                # are valid N-substituents (NHR / NR1R2 cases handled by
                # ``_compute_branch_b_carbamoyloxy_name``).
                core_set = match_set - {match[4]}
                if core_set.issubset(frag_atoms_set):
                    # Phase 160.2 follow-up: N-substituted Branch B per
                    # IUPAC P-66.6.4. NH2 → "carbamoyloxy"; NHR →
                    # "(N-Rcarbamoyl)oxy"; NR1R2 →
                    # "(N,N-(R1)(R2)carbamoyl)oxy" (alphabetized).
                    branch_b = _compute_branch_b_carbamoyloxy_name(mol, match)
                    if branch_b is not None:
                        return branch_b
            # --- End CR-01 fix ---
            # v27 P2 (P-63.6): a sulfoxide/sulfone SUBSTITUENT attaches through
            # its SULFUR, so the SMARTS' *other* carbon (R' on the parent side)
            # lives OUTSIDE the fragment. The strict match_set==frag_atoms_set
            # test below would then reject the match and the fragment would fall
            # through to a lower tier that DROPS the S and its =O atoms (naming a
            # bare 'methyl' for -S(=O)(=O)-CH3 -> a DIFFERENT molecule, CR-01
            # class). Accept the match when the S is the attach atom and every
            # match atom except that single parent-side carbon is inside the
            # fragment; name it (R)sulfinyl / (R)sulfonyl with the parent carbon
            # marked as the "chain" so the builder picks the in-fragment R.
            if (fg_name in ("sulfoxide", "sulfone")
                    and allow_higher_sulfur
                    and attach_idx is not None
                    and len(match) >= 4
                    and match[0] == attach_idx):
                s_atom = mol.GetAtomWithIdx(match[0])
                c_nbrs = [n.GetIdx() for n in s_atom.GetNeighbors()
                          if n.GetSymbol() == "C"]
                parent_cs = [c for c in c_nbrs if c not in frag_atoms_set]
                sub_cs = [c for c in c_nbrs if c in frag_atoms_set]
                # The =O atoms (SMARTS positions after S) must be part of the
                # substituent fragment; only the single parent-side carbon may
                # sit outside it. (Containment, not equality: the R' side may be
                # multi-atom -- ethyl, phenyl -- so match captures just its first
                # carbon while the fragment holds the whole R'.)
                o_atoms = [i for i in match
                           if mol.GetAtomWithIdx(i).GetSymbol() == "O"]
                if (len(parent_cs) == 1 and len(sub_cs) >= 1
                        and match[0] in frag_atoms_set
                        and all(o in frag_atoms_set for o in o_atoms)):
                    prefix = get_substituent_prefix_form(
                        fg_name, mol, tuple(match),
                        principal_chain=[parent_cs[0]],
                    )
                    if prefix is not None:
                        return prefix
                continue
            # FG must EQUAL the fragment (no extra atoms). This is the
            # IUPAC P-65/P-66 prefix-form precondition: the substituent
            # fragment must be the FG itself, not a larger group containing
            # the FG (CONTEXT D-04 strict scope).
            if match_set != frag_atoms_set:
                continue
            # v22 C-T2 (V-3): the ether (alkoxy) prefix form is an attach-via-O
            # prefix (R-O-). It is only valid when the substituent actually
            # attaches through the ether oxygen (match[0] for the O-ether SMARTS).
            # When the fragment attaches via a CARBON (e.g. -CH2-O-CH3,
            # methoxymethyl), the ether is internal to the chain and must be named
            # (R-oxy)alkyl by the downstream chain handler — NOT collapsed to
            # 'methoxy' (which drops the attachment carbon -> a different
            # constitution). Fall through. Scoped to O-ethers only: the
            # thioether/selenoether/telluroether SMARTS put a CARBON at match[0]
            # (S/Se/Te at match[1]), so this match[0] test does not apply to them,
            # and no carbon-attached chalcogen-ether handler exists yet (C-T2 is O).
            if (fg_name in ("ether", "vinyl_ether", "aromatic_ether")
                    and attach_idx is not None
                    and len(match) >= 1 and attach_idx != match[0]):
                continue
            # Wave-2 completion C2 (P-15.2.1.1, same bug shape as the ether
            # guard above): the isocyanate/isothiocyanate SMARTS includes the
            # linker carbon at match[0]; a benzyl fragment {CH2,N,C,S} passed
            # the strict-scope test and returned bare 'isothiocyanato',
            # DROPPING the CH2. The prefix is valid only when the fragment
            # attaches through the FG nitrogen (match[1]).
            if (fg_name in ("isocyanate", "isothiocyanate")
                    and attach_idx is not None
                    and len(match) >= 2 and attach_idx != match[1]):
                continue
            # v31 (same bug shape as the ether + isocyanate guards above, P-63.6):
            # the (R)sulfinyl / (R)sulfonyl prefix form attaches through the SULFUR
            # (match[0] for both the sulfoxide `[SX3](=O)([#6])[#6]` and sulfone
            # SMARTS). When the fragment attaches via a CARBON CARRIER instead
            # (e.g. -CH2-S(=O)-CH3, benzyl methyl sulfoxide), the S(=O)x is internal
            # and this prefix would DROP the carrier carbon -> 'methanesulfinyl' for
            # -CH2-S(=O)-CH3, a DIFFERENT molecule (CS(=O)c…). Valid only when the
            # fragment attaches through the sulfur; otherwise fall through to the
            # recursive namer, which names the carrier carbon bearing an
            # (alkylsulfinyl)/(alkylsulfonyl) decoration ((methanesulfinylmethyl)).
            if (fg_name in ("sulfoxide", "sulfone")
                    and attach_idx is not None
                    and len(match) >= 1 and attach_idx != match[0]):
                continue
            prefix = get_substituent_prefix_form(
                fg_name, mol, tuple(match), principal_chain=None
            )
            if prefix is not None:
                return prefix
    return None
