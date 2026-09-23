"""Unified substituent enumeration module with ReplaceCore-based extraction.

Provides a single enumeration path for all substituents on both ring and chain
parent structures. Replaces the previously fragmented three-path system that
caused silent drops, double-counting, and wrong locants.

Architecture:
  - Ring parents: ReplaceCore(mol, core_from_ring_atoms) -> fragment mols
  - Chain parents: Branch-point enumeration from features.substituents dict
  - All fragments: classify -> name via existing naming infrastructure

Public API:
  - discover_substituents(mol, parent_atoms, parent_type,...) [a phase]
  - extract_ring_substituents(mol, ring_atoms, oriented_ring)
  - extract_chain_substituents(mol, principal_chain, substituents_dict)
  - classify_and_name_fragment(mol, frag_info, parent_atoms, features=None)
  - collect_substituent_atom_set(substituent_infos)

References:
    IUPAC 2013 (detachable prefixes)
    IUPAC 2013 (parent selection determines what's a substituent)
"""

import logging
import re
import threading
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import List, Optional

# a phase Plan-04-02 closure: removed unused
# ``from typing import List, Optional, Set, Dict`` — none of the four
# names are referenced anywhere in 1608 LOC (verified via AST scan;
# they appeared only in docstrings, not annotations). Re-add narrowly
# scoped imports here when real annotations are added. (v52: ``List`` re-added
# for the chalcogen-chain walk in ``_name_disulfanyl_branch``.)
from rdkit import Chem
from rdkit.Chem import RWMol

# candidate ledger. Leaf module (threading + typing only), so cycle-free;
# OFF unless a consumer calls enable, so the production cost is one boolean test.
from ..metrics.candidate_ledger import Scope as _LedgerScope
from ..metrics.candidate_ledger import Stage as _LedgerStage
from ..metrics.candidate_ledger import record_candidate as _ledger_record
from ..perception.molcache import atoms_of, bonds_of  # audit 2026-09-03 (S2): per-call atom/bond tuples
from ..rules.seniority import get_prefix
from .naming_utils import SIMPLE_MULTIPLIERS, get_alkyl_name
from .substituent_naming import _name_aryl_methyl_ether, name_substituent_fragment
from .substituent_prefix_forms import _check_substituent_prefix_form
from ..perception.smarts_cache import compiled as _compiled_smarts
from ..metrics.provenance import best_effort_ctx  # perf lever A10 (2026-09-13): hoisted (14,478 executions per 300 molecules)
from ..metrics.provenance import (  # a lever: breadth flags enter the name_substituent memo key
    allow_aromatic_general_ctx,
    full_coverage_ctx,
    general_fallback_ctx,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

SubstituentInfo = namedtuple(
    'SubstituentInfo',
    ['frag_mol', 'locant', 'attach_mol_idx', 'frag_atoms']
)
"""
Represents a single substituent on a parent structure.

Fields:
    frag_mol: RDKit Mol of the isolated fragment (with dummy atom at attachment),
              or None for chain-parent substituents where fragment is inline.
    locant: IUPAC locant (1-indexed integer) on the parent.
    attach_mol_idx: Original mol atom index of the attachment point on parent.
    frag_atoms: Set of original mol atom indices belonging to this substituent.
"""


# ============================================================================
# Universal Substituent Discovery (a phase)
# ============================================================================


def discover_substituents(
    mol,
    parent_atoms,
    parent_type="auto",
    oriented_ring=None,
    principal_chain=None,
    atom_to_locant=None,
    ring_atom_to_locant=None,
    general_fallback=False,
):
    """Discover ALL substituents on a parent structure.

    Universal entry point that replaces six parallel substituent discovery
    systems. Every non-parent, non-hydrogen atom in mol is assigned to
    exactly one SubstituentInfo. No silent drops, no size limits.

    Args:
        mol: RDKit Mol object.
        parent_atoms: Set of atom indices defining the parent structure.
        parent_type: ``"ring"``, ``"chain"``, or ``"auto"`` (auto-detects).
        oriented_ring: Ring atom indices in IUPAC order (required for ring parents).
        principal_chain: Chain atom indices in order (required for chain parents).
        atom_to_locant: Optional mapping of atom idx -> IUPAC locant. Consumed
            by the CHAIN path only.
        ring_atom_to_locant: Optional mapping of atom idx -> IUPAC locant for a
            RING parent, inherited from whichever producer spelled the parent
            name. Overrides the ``oriented_ring`` position arithmetic per atom
            (``'4a'``/``'8a'`` and any fused-parent numbering cannot be
            expressed as a position). ``None`` -> unchanged behaviour.
        general_fallback: Composer1 Task 5 gating flag. Passed ``True`` ONLY
            from the general-engine (complete/best-effort) call sites. Controls
            the failure mode when a non-parent heavy atom cannot be assigned to
            any substituent fragment — e.g. a substituent hanging off a SUFFIX/FG
            heteroatom (the N-aryl ring of an amide anilide), which the parent-
            chain/ring walk cannot reach:

              * ``False`` (PIN default): ``_verify_completeness`` HARD-asserts on
                the unassigned atoms exactly as before (byte-identical). The
                narrow PIN handlers own these molecules; a partition gap there is
                a genuine invariant violation the PIN callers already catch.
              * ``True`` (general fallback): FAIL CLOSED to a clean sentinel —
                return ``None`` instead of raising, and NEVER silently drop the
                unassigned atoms (dropping them would ship a name of the wrong
                constitution). The general caller converts the ``None`` to a
                clean ``_refuse`` -> the engine abstains, no exception.

    Returns:
        List[SubstituentInfo] with one entry per substituent fragment, or
        ``None`` when ``general_fallback=True`` and the partition is incomplete
        (unassigned / overlapping atoms) — signalling the caller to fail closed.
    """
    parent_set = set(parent_atoms)

    if parent_type == "auto":
        parent_type = _detect_parent_type(mol, parent_set)

    if parent_type == "ring":
        results = extract_ring_substituents(
            mol, tuple(parent_set), oriented_ring,
            atom_to_locant=ring_atom_to_locant,
        )
    else:
        results = _discover_chain_substituents(
            mol, parent_set, principal_chain, atom_to_locant
        )

    complete = _verify_completeness(
        mol, parent_set, results, general_fallback=general_fallback
    )
    if general_fallback and not complete:
        # Partition gap under the general-fallback context (e.g. a substituent
        # off a suffix/FG heteroatom the walk cannot reach). Fail closed to a
        # sentinel the general caller handles; never drop the atoms.
        return None

    return results


def _detect_parent_type(mol, parent_atoms):
    """Auto-detect whether parent_atoms represent a ring or chain parent.

    Checks if any complete ring in the molecule is a subset of parent_atoms.
    If so, returns ``"ring"``; otherwise ``"chain"``.

    Args:
        mol: RDKit Mol object.
        parent_atoms: Set of atom indices defining the parent structure.

    Returns:
        ``"ring"`` or ``"chain"``.
    """
    parent_set = set(parent_atoms)
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if set(ring).issubset(parent_set):
            return "ring"
    return "chain"


def _discover_chain_substituents(mol, parent_atoms, principal_chain,
                                  atom_to_locant=None):
    """BFS-based substituent discovery for chain parents.

    For each atom on the principal chain, finds non-parent neighbors and
    BFS-collects complete substituent fragments. Walks through ALL atom
    types (no carbon-only restriction). No size limit.

    Args:
        mol: RDKit Mol object.
        parent_atoms: Set of atom indices in the parent chain.
        principal_chain: List of atom indices in chain order.
        atom_to_locant: Optional mapping of atom idx -> IUPAC locant.

    Returns:
        List[SubstituentInfo] namedtuples.
    """
    results = []
    parent_set = set(parent_atoms)
    assigned = set()

    if principal_chain is None:
        principal_chain = sorted(parent_set)

    for chain_pos, chain_atom_idx in enumerate(principal_chain):
        chain_atom = mol.GetAtomWithIdx(chain_atom_idx)
        for nbr in chain_atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in parent_set or nbr_idx in assigned:
                continue
            if nbr.GetAtomicNum() == 1:
                continue

            frag_atoms = _bfs_collect_fragment(
                mol, nbr_idx, parent_set, assigned
            )
            if not frag_atoms:
                continue
            assigned.update(frag_atoms)

            locant = chain_pos + 1
            if atom_to_locant and chain_atom_idx in atom_to_locant:
                locant = atom_to_locant[chain_atom_idx]

            results.append(SubstituentInfo(
                frag_mol=None,
                locant=locant,
                attach_mol_idx=chain_atom_idx,
                frag_atoms=frozenset(frag_atoms),
            ))

    return results


def _bfs_collect_fragment(mol, start_idx, parent_set, already_assigned):
    """BFS from start_idx, collecting all non-parent heavy atoms.

    CRITICAL: Does NOT stop at heteroatoms. Collects O, N, S, P and all
    atoms reachable through them. This ensures FG-containing substituents
    are discovered as compound fragments . No size limit.

    Args:
        mol: RDKit Mol object.
        start_idx: Atom index to start BFS from.
        parent_set: Set of parent atom indices (BFS boundary).
        already_assigned: Set of atom indices already claimed by another
            substituent.

    Returns:
        Set of atom indices in the collected fragment.
    """
    visited = set()
    queue = deque([start_idx])
    while queue:
        idx = queue.popleft()
        if idx in visited or idx in parent_set or idx in already_assigned:
            continue
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 1:
            continue
        visited.add(idx)
        for nbr in atom.GetNeighbors():
            queue.append(nbr.GetIdx())
    return visited


def _verify_completeness(mol, parent_atoms, substituents,
                         general_fallback=False):
    """Verify that all non-parent heavy atoms are accounted for.

    Every non-parent, non-hydrogen atom must be assigned to exactly one
    SubstituentInfo. Detects double-assigned, missed, and extra atoms.

    Args:
        mol: RDKit Mol object.
        parent_atoms: Set of parent atom indices.
        substituents: List of SubstituentInfo namedtuples.
        general_fallback: Composer1 Task 5 gating flag (see
            ``discover_substituents``). Selects the failure mode:

              * ``False`` (PIN default): a partition violation HARD-asserts
                exactly as before — byte-identical raise + message. This path
                is unchanged for every PIN-context caller.
              * ``True`` (general fallback): NO exception — a partition
                violation returns ``False`` (incomplete) so the general caller
                can fail closed to a clean abstain instead of crashing.

    Returns:
        bool: ``True`` if the partition is complete (every non-parent heavy
        atom assigned exactly once), else ``False`` (only reachable under
        ``general_fallback=True``; otherwise an ``AssertionError`` is raised
        before returning).

    Raises:
        AssertionError: If ``general_fallback=False`` and atoms are
            double-assigned, missed, or extra (byte-identical to the prior
            behavior).
    """
    parent_set = set(parent_atoms)
    all_sub_atoms = set()
    for sub in substituents:
        overlap = all_sub_atoms & set(sub.frag_atoms)
        if overlap:
            if general_fallback:
                return False
            assert not overlap, (
                f"Double-assigned atoms: {overlap}"
            )
        all_sub_atoms.update(sub.frag_atoms)

    expected = set()
    for atom in atoms_of(mol):
        if atom.GetAtomicNum() > 1 and atom.GetIdx() not in parent_set:
            expected.add(atom.GetIdx())

    missed = expected - all_sub_atoms
    if missed:
        if general_fallback:
            return False
        assert not missed, f"Unassigned atoms: {missed}"
    extra = all_sub_atoms - expected
    if extra:
        if general_fallback:
            return False
        assert not extra, f"Extra atoms not in molecule: {extra}"
    return True


# ============================================================================
# free-valence morphology (a phase)
# ============================================================================
#
# IUPAC 2013 fixes a substituent prefix's ENDING by the number of free
# valences on the attachment atom:
#
# one -> -yl (methyl, single bond to the parent)
# two -> -ylidene (methylidene, double bond to the parent)
# three -> -ylidyne (methylidyne, triple bond to the parent)
#
# That number is a property of the ATTACHMENT BOND, not of the fragment's
# composition and not of its hydrogen count. Every tier of the naming cascade
# below builds its token from the fragment alone, so none of them can know it;
# ``name_substituent`` is the only place that holds the fragment AND the
# attachment atom together, which is why the morphology is decided here, once,
# for every tier.
#
# also settles the spelling: ``methylidene`` is the PIN for =CH2 as a
# prefix. ``methylene`` is the retained/general form and is NOT the PIN, so it
# is never emitted.

#: free-valence count -> suffix morpheme.
_FREE_VALENCE_SUFFIX = {1: 'yl', 2: 'ylidene', 3: 'ylidyne'}

#: Bond order -> free-valence count. Aromatic and dative bonds are absent on
#: purpose: they carry no integer free valence, so no morphology can be
#: asserted for them and the caller must treat them as undecidable.
_BOND_FREE_VALENCE = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
}


def _free_valence_at_attachment(mol, frag_atoms, attach_idx):
    """How many free valences the fragment's attachment atom carries.

    Reads the ACTUAL bond order of the single bond leaving the fragment, per
    . Deliberately not the hydrogen count: H count merely correlates with
    bond order (``-CH3`` vs ``=CH2``), and a correlation is what produced the
    ``methyl``-for-``=CH2`` defect in the first place.

    Args:
        mol: RDKit Mol of the FULL molecule.
        frag_atoms: Iterable of atom indices forming the substituent fragment.
        attach_idx: The fragment atom bonded to the enclosing parent. Note this
            is an atom OF the fragment, so the linkage bond is the one from it
            to a neighbour OUTSIDE ``frag_atoms``.

    Returns:
        1, 2 or 3 -- or ``None`` when the shape is outside the
        single-attachment-atom case: the fragment touches the parent at more
        than one bond (a bridge/spiro, rather than a prefix), the linkage
        is aromatic or dative (no integer order), or there is no linkage at
        all. ``None`` means UNDECIDABLE and must never be read as 1.
    """
    frag = set(frag_atoms)
    if attach_idx not in frag:
        return None
    linkage = []
    for idx in frag:
        atom = mol.GetAtomWithIdx(idx)
        for bond in atom.GetBonds():
            other = bond.GetOtherAtomIdx(idx)
            if other in frag:
                continue
            if mol.GetAtomWithIdx(other).GetAtomicNum() <= 1:
                continue  # explicit H is part of the fragment's own saturation
            linkage.append((idx, bond))
    # Exactly one bond out of the fragment is what a -yl/-ylidene/-ylidyne
    # prefix describes. Two or more is a bridge, whatever their orders.
    if len(linkage) != 1:
        return None
    src, bond = linkage[0]
    if src != attach_idx:
        return None
    return _BOND_FREE_VALENCE.get(bond.GetBondType())


def _token_asserts_single_free_valence(token):
    """True when ``token``'s own text CONFIDENTLY spells one free valence.

    A valence-conservation guard, never a producer: a token this returns True
    for may not be emitted for a double or triple attachment, because it names
    a different molecule.

    Delegates to the shared text oracle so the producer and the proof
    spine's P7 cannot drift apart about what a token asserts. Anything the
    oracle refuses (``oxo``, ``hydroxy``, a multiplied ``-diyl`` ending) is not
    claimed here, so the guard never suppresses a prefix on a reading it could
    not defend.
    """
    from ..validation.name_morphemes import free_valence_morphology
    estimate = free_valence_morphology(token)
    return bool(estimate.confident and estimate.free_valences == 1)


def _carbon_ylidene_prefix(mol, frag_atoms, attach_idx, free_valence):
    """ ``-ylidene`` / ``-ylidyne`` prefix for a CARBON free valence.

    Builds the token from the structure -- the parent hydride the fragment IS,
    plus the morpheme -- for two shapes:

    *Acyclic*: the chain running through the attachment atom, numbered so the
    free valence takes the lowest locant it can.

        =CH2 -> methylidene
        CH3-CH= -> ethylidene
        CH3-CH2-CH= -> propylidene retained stem)
        (CH3)2C= -> propan-2-ylidene
        CH3-C(triple) -> ethylidyne

    *Monocyclic*: an unsubstituted saturated carbocycle, whose free valence is
    on a ring atom and needs no locant -- every ring atom of an
    otherwise-bare cycloalkane is equivalent).

        -(CH2)5C= -> cyclohexylidene

    Fails closed (``None``) on everything else -- a heteroatom attachment,
    unsaturation or a charge inside the fragment, a branch hanging off the
    chain, a decorated or polycyclic ring, or any CIP stereo the ``-yl`` stereo
    emitter would have to describe. Callers must NOT degrade a ``None`` into a
    ``-yl`` token.
    """
    suffix = _FREE_VALENCE_SUFFIX.get(free_valence)
    if suffix is None or free_valence == 1:
        return None
    frag = set(frag_atoms)
    if attach_idx not in frag:
        return None

    from ..data.chain_names import get_chain_prefix

    # -- fragment must be neutral, saturated, all-carbon, stereo-free --
    ring_info = mol.GetRingInfo()
    in_ring = 0
    for idx in frag:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() != 6:
            return None
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons():
            return None
        if atom.GetIsotope():
            return None
        if atom.HasProp('_CIPCode'):
            return None
        if atom.IsInRing():
            in_ring += 1
            # A ring atom in more than one ring is a fused/spiro/bridged
            # system, whose parent hydride this constructor does not build.
            if ring_info.NumAtomRings(idx) != 1:
                return None
    for bond in bonds_of(mol):
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in frag and j in frag:
            if bond.GetBondType() != Chem.BondType.SINGLE:
                return None
            if bond.HasProp('_CIPCode'):
                return None

    # -- monocyclic branch: a bare saturated carbocycle --
    if in_ring:
        # Every fragment atom must be in the SAME single ring: a ring carrying
        # any substituent would need that substituent's own prefix and locant.
        if in_ring != len(frag):
            return None
        rings = [r for r in ring_info.AtomRings() if attach_idx in r]
        if len(rings) != 1 or set(rings[0]) != frag:
            return None
        stem = get_chain_prefix(len(frag))
        return f"cyclo{stem}{suffix}" if stem else None

    # -- longest chain through the attachment atom, free valence lowest --
    # Depth of each branch leaving the attachment atom (the fragment is a tree,
    # so a plain DFS depth is the longest simple path into that branch).
    def _depth(start, banned):
        best, stack = 0, [(start, banned, 1)]
        while stack:
            cur, seen, dist = stack.pop()
            best = max(best, dist)
            for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
                k = nb.GetIdx()
                if k in frag and k not in seen:
                    stack.append((k, seen | {k}, dist + 1))
        return best

    branches = sorted(
        (_depth(nb.GetIdx(), {attach_idx, nb.GetIdx()})
         for nb in mol.GetAtomWithIdx(attach_idx).GetNeighbors()
         if nb.GetIdx() in frag),
        reverse=True)
    longest = branches[0] if branches else 0
    second = branches[1] if len(branches) > 1 else 0
    chain_len = 1 + longest + second
    # Numbering from the SHORTER side gives the free valence its lowest locant.
    locant = second + 1

    # The chain must BE the fragment. A branch hanging off it would need its
    # own prefix and locant, which is a different construction than the one
    # built here -- fail closed rather than lose it.
    if chain_len != len(frag):
        return None

    from ..data.chain_names import get_chain_prefix
    stem = get_chain_prefix(chain_len)
    if not stem:
        return None
    if locant == 1:
        # Retained unbranched form: methylidene / ethylidene / propylidene.
        return f"{stem}{suffix}"
    return f"{stem}an-{locant}-{suffix}"


def _subtree_atoms(mol, frag, start, banned):
    """Atoms of ``frag`` reachable from ``start`` without crossing ``banned``.

    The fragment is a tree at the substituent level, so this is the branch
    hanging off ``start`` once the bond back toward ``banned`` is cut.
    """
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            k = nb.GetIdx()
            if k in frag and k not in banned and k not in seen:
                seen.add(k)
                stack.append(k)
    return seen


def _nitrogen_ylidene_prefix(mol, frag_atoms, attach_idx, free_valence):
    """ / BB(:1703) ``-ylidene`` prefix for a NITROGEN free valence, or ``None``.

    A nitrogen attached to the parent by a DOUBLE bond is an ``ylidene`` free
    valence, not a ``-yl``. Two shapes, both left unnamed by the carbon
    constructor (non-carbon root) and by the ``-NH-R`` amino intercept (which
    requires all-single bonds), which is why they used to reach the guard
    as ``...hydrazinyl`` and fail closed -- the measured decorated-steroid /
    hydrazone breadth class::

        =N-N< -> '{2-substituents}hydrazin-1-ylidene'
                          (the attached N takes locant 1 -- the free valence gets
                          the lowest locant, -- and the OTHER nitrogen is
                          locant 2 and carries its own substituents)
        =N-R / =NH -> '{R}imino' / 'imino'

    The deprecated ``hydrazono`` prefix is NEVER emitted (BB "Changes from the
    1979 edition":1703 makes ``hydrazinylidene`` the systematic form). Every
    result is a candidate that (``_rt_match``) round-trip-verifies
    downstream, so a wrong locant assignment abstains rather than ships.

    Returns ``None`` (caller fails closed) for a triple-bond N free valence, a
    charged / isotopic / radical attachment, or any substituent the recursive
    namer declines.
    """
    if free_valence != 2:
        return None  # =N with a triple bond (nitrilo) is a separate, rare class
    frag = set(frag_atoms)
    if attach_idx not in frag:
        return None
    na = mol.GetAtomWithIdx(attach_idx)
    if (na.GetSymbol() != 'N' or na.GetFormalCharge() != 0
            or na.GetNumRadicalElectrons() or na.GetIsotope()):
        return None

    # The attached N's neighbours INSIDE the fragment (the parent is outside it).
    inner = [nb.GetIdx() for nb in na.GetNeighbors() if nb.GetIdx() in frag]

    from .naming_utils import alpha_sort_key, enclose_if_compound, get_multiplier_prefix

    def _name_branch(root, banned):
        """The recursive substituent name for the branch rooted at ``root``."""
        sub = _subtree_atoms(mol, frag, root, banned)
        token = name_substituent(mol, sub, root)
        from ..errors import is_refusal_sentinel
        if not token or is_refusal_sentinel(token) or ' ' in token:
            return None
        return token

    # --- hydrazinylidene: the attached N bonds to exactly one other N -----
    n_neighbours = [i for i in inner
                    if mol.GetAtomWithIdx(i).GetSymbol() == 'N']
    if len(inner) == 1 and len(n_neighbours) == 1:
        nb_idx = n_neighbours[0]
        nb = mol.GetAtomWithIdx(nb_idx)
        if (nb.GetFormalCharge() != 0 or nb.GetNumRadicalElectrons()
                or nb.GetIsotope()):
            return None
        # every bond inside the hydrazine backbone + its decorations must be
        # single except the parent attachment (a =N-N= diylidene is a different
        # class named on hydrazine as a whole).
        if mol.GetBondBetweenAtoms(attach_idx, nb_idx).GetBondType() != \
                Chem.BondType.SINGLE:
            return None
        # substituents on the FAR nitrogen (locant 2)
        names = []
        for r in nb.GetNeighbors():
            k = r.GetIdx()
            if k == attach_idx or k not in frag:
                continue
            token = _name_branch(k, {nb_idx})
            if token is None:
                return None
            names.append(token)
        if not names:
            return "hydrazinylidene"
        # locant omission (the Blue Book verbatim "(dimethylcarbamoyl)
        # hydrazinylidene (preferred prefix)"; the Blue Book makes 'hydrazinylidene' the
        # systematic prefix for H2N-N=): the free valence is at N1, which is
        # valence-FULL (=parent double bond + N2 single bond), so EVERY substituent
        # must sit on N2 -- the '1' (free valence) and '2' (substituent) locants are
        # unambiguous and carry no information, so both are omitted:
        # '(propan-2-ylidene)hydrazinylidene', 'dimethylhydrazinylidene', never
        # '2-(propan-2-ylidene)hydrazin-1-ylidene'. Identical tokens keep their
        # multiplier; each candidate is still /OPSIN-RT verified downstream,
        # so a hypothetical ambiguous omission abstains rather than ships wrong.
        grouped = {}
        for t in names:
            grouped.setdefault(t, 0)
            grouped[t] += 1
        parts = []
        for t, count in grouped.items():
            mult = get_multiplier_prefix(count, t)
            parts.append((alpha_sort_key(t), f"{mult}{enclose_if_compound(t)}"))
        block = "".join(p for _, p in sorted(parts))
        return f"{block}hydrazinylidene"

    # --- imino: the attached N bonds to zero further N (=N-R or =NH) -------
    if len(n_neighbours) == 0:
        if not inner:
            return "imino"           # =NH
        if len(inner) != 1:
            return None              # a neutral =N cannot carry two single bonds
        token = _name_branch(inner[0], {attach_idx})
        if token is None:
            return None
        return f"{enclose_if_compound(token)}imino"

    return None


# ---------------------------------------------------------------------------
# The ONE verdict every substituent detector asks for
# ---------------------------------------------------------------------------
#
# Six ring-substituent detectors independently walk from a parent atom to an
# exocyclic neighbour, trace the fragment, count its carbons and mint a ``-yl``
# prefix from the count alone. The count cannot distinguish ``-CH3`` from
# ``=CH2``, so every one of them named a different molecule the moment the
# attachment bond was double -- and each fix written detector-by-detector was
# another private copy of "read the bond order here". The copies are the defect:
# a detector added tomorrow starts out without one.
#
# So the bond-order read (``_free_valence_at_attachment``), the constructor
# (``_carbon_ylidene_prefix``) and the text check (``free_valence_morphology``)
# are composed ONCE, here, into a single three-way verdict that a detector
# consumes without re-deriving anything:
#
# defers -> single (or undecidable) free valence. The caller's own
# existing ``-yl`` naming is -correct; carry on
# unchanged. This is the overwhelmingly common case and
# is what keeps every existing name byte-identical.
# prefix is set -> two or three free valences AND a prefix whose own text
# spells exactly that many. Use it verbatim.
# must_fail_closed -> two or three free valences and no such prefix. The
# caller MUST abstain. Emitting ``-yl`` here names a
# different molecule, and dropping the fragment names a
# different molecule too.


#: Re-entrancy flag for the gate.
#:
#: ``_ylidene_from_substituent_name`` below asks the ordinary fragment namer for
#: the fragment's SINGLE-valence reading, and that namer's first act is to
#: consult this gate -- which would ask again, forever. The flag says "the gate
#: is already being applied by an outer frame; give me the plain -yl name". It
#: is thread-local because naming runs per-thread in the corpus harnesses.
_GATE_REENTRY = threading.local()


def _gate_is_reentrant() -> bool:
    """True while the gate is asking a namer for a fragment's ``-yl`` reading."""
    return getattr(_GATE_REENTRY, "active", False)


@dataclass(frozen=True)
class FreeValencePrefix:
    """The verdict for one substituent fragment's attachment bond.

    ``free_valence`` is always the TRUE bond order at the attachment (1/2/3),
    or ``None`` when the shape is outside the single-attachment-atom case a
    free-valence prefix describes at all (a bridge, an aromatic linkage).
    ``in_class`` is False when the attachment atom is not carbon, where the
     carbon morphology does not apply and the caller's own heteroatom
    machinery owns the naming.

    Callers branch on the two PROPERTIES, never on the fields: reading the
    fields directly is how a caller re-invents the policy this class holds --
    and getting ``in_class`` wrong turns every ring ketone into an abstention.
    """

    free_valence: Optional[int]
    prefix: Optional[str]
    basis: str
    in_class: bool = True

    @property
    def defers(self) -> bool:
        """True when the caller's own existing naming path is -correct."""
        return (not self.in_class
                or self.free_valence is None
                or self.free_valence == 1)

    @property
    def must_fail_closed(self) -> bool:
        """True when needs a multivalent prefix and none could be built."""
        return not self.defers and self.prefix is None


def _ylidene_from_substituent_name(mol, frag_atoms, attach_idx, free_valence):
    """``-ylidene``/``-ylidyne`` for a DECORATED fragment, or ``None``.

    ``_carbon_ylidene_prefix`` builds the token from scratch and so is limited to
    bare shapes -- a plain chain, a bare cycloalkane. A great many real free
    valences sit on a decorated carbon: the diketopiperazine natural product
    ``O=C1NC(Cc2c[nH]c3ccccc23)C(=O)N/C1=C/c1cnc[nH]1`` carries an exocyclic
    ``=CH-`` bearing an imidazole, whose prefix is
    ``(1H-imidazol-5-yl)methylidene``. Refusing those would fail closed on a
    large and entirely nameable part of the class.

     says the free-valence morpheme is a SUFFIX on the fragment's parent
    hydride, and the parent hydride does not depend on how many valences are
    free. So the fragment's own ``-yl`` name -- which the project's recursive
    substituent namer already builds correctly, decorations and all -- is the
    same name this needs, with one morpheme exchanged.

    That exchange is guarded on both sides, which is what keeps it a derivation
    rather than a string edit:

    * the ``-yl`` reading must CONFIDENTLY spell exactly one free valence (the
      shared morpheme oracle decides, not a string test), so a token whose
      ending cannot be read -- ``oxo``, a multiplied ``-diyl`` -- is refused;
    * only that trailing morpheme is replaced, and the caller re-reads the
      RESULT through the same oracle and requires it to spell the bond's actual
      order.

    Returns ``None`` for anything the namer declines or the oracle cannot read,
    so the caller still fails closed.
    """
    suffix = _FREE_VALENCE_SUFFIX.get(free_valence)
    if suffix is None or free_valence == 1:
        return None
    # An outer frame is already inside the gate; recursing would not terminate.
    if _gate_is_reentrant():
        return None

    from ..validation.name_morphemes import free_valence_morphology

    _GATE_REENTRY.active = True
    try:
        base = name_substituent_fragment(
            mol, sorted(frag_atoms), attach_idx, [])
    except Exception:  # noqa: BLE001 - a namer failure is an abstention
        return None
    finally:
        _GATE_REENTRY.active = False

    estimate = free_valence_morphology(base)
    if not (estimate.confident and estimate.free_valences == 1):
        return None
    # The '-yl' the oracle just confirmed is the token's own ending.
    return f"{base[:-2]}{suffix}"


def carbon_free_valence_prefix(mol, frag_atoms, attach_idx) -> FreeValencePrefix:
    """ verdict for a CARBON-attached substituent fragment.

    The shared entry point for every substituent detector -- monocyclic,
    heterocyclic, bicyclo, von Baeyer polycyclic, spiro, fused. Use it wherever
    a fragment is about to be named from its carbon count::

        verdict = carbon_free_valence_prefix(mol, sub_atoms, first_atom)
        if verdict.prefix:
            name = verdict.prefix
        elif verdict.must_fail_closed:
            ...abstain... # never get_alkyl_name(carbon_count)
        else:
            name = get_alkyl_name(carbon_count) # unchanged legacy path

    Args:
        mol: RDKit Mol of the FULL molecule.
        frag_atoms: Atom indices of the substituent fragment.
        attach_idx: The FRAGMENT atom bonded to the parent (not the parent
            atom -- the linkage bond runs from this atom OUT of the fragment).

    Scope -- deliberately CARBON only. The multivalent heteroatom prefixes
    (``oxo``, ``sulfanylidene``, ``imino``) are a separate class that the
    detectors' own heteroatom machinery already names correctly, and their
    tokens spell no morpheme at all, so the text check below cannot
    confirm them. Including them would turn every ring ketone into a
    fail-closed abstention. A non-carbon attachment therefore DEFERS, leaving
    the existing chalcogen/nitrogen paths untouched.

    The prefix is accepted only when its own TEXT spells the number of free
    valences the BOND has. The two facts come from independent places -- the
    count from the molecular graph, the reading from the shared morpheme oracle
    that the proof spine's P7 also uses -- so their agreement is evidence, and a
    future constructor change that quietly emitted a ``-yl`` token would be
    caught here rather than shipped.

    Deliberately calls the CONSTRUCTOR, not the full naming cascade. Two
    reasons: no cascade tier mints a carbon ``-ylidene`` token (they all build
    single-valence prefixes, which the check would reject anyway), and this
    function is called FROM inside the cascade's own chokepoints, so routing it
    back through them would recurse without terminating.

    References: IUPAC 2013,,.
    """
    from ..validation.name_morphemes import free_valence_morphology

    frag = set(frag_atoms)
    free_valence = _free_valence_at_attachment(mol, frag, attach_idx)
    if free_valence is None or free_valence == 1:
        return FreeValencePrefix(
            free_valence, None,
            "single or undecidable free valence: caller's own naming applies")

    if attach_idx is None or mol.GetAtomWithIdx(attach_idx).GetAtomicNum() != 6:
        # Out of the CARBON class (see the scope note above).
        return FreeValencePrefix(
            free_valence, None,
            "non-carbon attachment: heteroatom multivalent prefixes are a "
            "separate class, deferred to the caller",
            in_class=False)

    token = _carbon_ylidene_prefix(mol, frag, attach_idx, free_valence)
    if token is None:
        token = _ylidene_from_substituent_name(
            mol, frag, attach_idx, free_valence)
    morphology = free_valence_morphology(token)
    if morphology.confident and morphology.free_valences == free_valence:
        return FreeValencePrefix(free_valence, token, morphology.basis)
    return FreeValencePrefix(
        free_valence, None,
        f"no P-29.2 prefix for a free valence of {free_valence}: "
        f"{token!r} ({morphology.basis})")


# ============================================================================
# Universal Substituent Naming (a phase)
# ============================================================================


def _route_fragment_to_general_engine(mol, frag_atoms, attach_idx):
    """M2 Task 2: last-resort route for a branch BOTH the cascade and the
    Phase-B3 universal-substituent fallback declined (`internal notes
    M2-PRODUCER-a trace.md``: 71 unique root fragments the cascade declines, 59 of
    them (83%) already nameable by the whole-molecule general engine -- the
    fused/spiro ring shapes ``universal_substituent``'s own narrower ring
    logic does not reach, since that module explicitly does NOT reuse
    ``general_engine``'s substituent-recursion tail).

    Strategy: cap the free valence with an exocyclic ``-OH`` (NOT a bare H)
    and ask ``name_compound`` to name the capped fragment standalone. The
    trick is that principal-characteristic-group numbering gives the
    lowest locant to THAT ``-OH`` -- the same atom that would carry the free
    valence -- so for a fragment with no OTHER senior group or pre-existing
    ``-OH``, the alcohol's own locant IS the free-valence locant
    would assign, and ``_alcohol_to_alkyl`` (already used by the amide/ester
    decomposition path for exactly this ``-ol`` -> ``-yl`` conversion) carries
    it over unchanged. A bare-H cap would lose that locant entirely -- with no
    functional group at the attachment atom, the capped fragment's own
    numbering has nothing to anchor a "this is atom 1" reading, so there is no
    way to recover which digit in the resulting name is our atom.

    Guarded to fail closed (``None``) rather than ever risk naming a
    different molecule:

    * the attach atom must be an uncharged CARBON -- adding ``-OH`` to any
      other element changes the fragment's chemistry (N-OH is a
      hydroxylamine, O-OH a peroxide,...);
    * no ring may be split by the extraction (a ring atom set only PARTIALLY
      inside ``frag_atoms`` means ``MolFragmentToSmiles`` would sever a ring
      bond -- one of the a trace's 4 fragmentation artifacts, describing a
      DIFFERENT, ring-opened molecule);
    * ``frag_atoms`` must have exactly ONE bond to the rest of the molecule
      -- the single point of attachment a ``-yl`` prefix describes;
    * the capped fragment's name must come back as a SINGLE, unmultiplied
      ``-ol`` (not ``diol``/``triol``/... and not some OTHER suffix) -- any
      other shape means the injected ``-OH`` did not win seniority (a
      senior group already in the fragment out-ranks it) or the fragment
      already carried its own ``-OH`` (now two), and in both cases the
      resulting locant is not provably the attachment atom's, so this
      declines rather than guess.

    The caller's OPSIN round-trip  is still the final 0-wrong net;
    this only decides whether a candidate is OFFERED.

    Pure: no mutation of ``mol`` (works on a private ``RWMol`` copy).
    """
    if attach_idx is None:
        return None
    frag_set = set(frag_atoms)
    if attach_idx not in frag_set or not (0 <= attach_idx < mol.GetNumAtoms()):
        return None
    attach_atom = mol.GetAtomWithIdx(attach_idx)
    if attach_atom.GetSymbol() != 'C' or attach_atom.GetFormalCharge() != 0:
        return None
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        ring_set = set(ring)
        if ring_set & frag_set and not ring_set.issubset(frag_set):
            return None  # extraction would cut this ring -- decline
    ext_bonds = [
        (a_idx, nb.GetIdx())
        for a_idx in frag_atoms
        for nb in mol.GetAtomWithIdx(a_idx).GetNeighbors()
        if nb.GetIdx() not in frag_set
    ]
    if len(ext_bonds) != 1:
        return None  # not a single clean point of attachment
    try:
        rw = Chem.RWMol(mol)
        o_idx = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(attach_idx, o_idx, Chem.BondType.SINGLE)
        h_idx = rw.AddAtom(Chem.Atom(1))
        rw.AddBond(o_idx, h_idx, Chem.BondType.SINGLE)
        cap_atoms = sorted(frag_set | {o_idx, h_idx})
        frag_smi = Chem.MolFragmentToSmiles(rw, cap_atoms, canonical=True)
    except Exception:  # a producer bug must never crash branch naming
        return None
    if not frag_smi:
        return None
    from ..errors import is_refusal_sentinel
    from ..namer import name_compound
    try:
        alcohol_name = name_compound(
            frag_smi, general_fallback=True,
            general_fallback_unverified=True, allow_aromatic_general=True)
    except Exception:
        return None
    if not alcohol_name or is_refusal_sentinel(alcohol_name) or ' ' in alcohol_name:
        return None
    if alcohol_name.lower().count('hydroxy') > 1:
        return None  # the fragment already carried its own -OH (now 2+)
    if alcohol_name.endswith('ol') and 'hydroxy' not in alcohol_name.lower():
        # Case A: the injected -OH won suffix seniority (a plain chain
        # or simple monocycle -- e.g. "cyclohexan-4-ol"). _alcohol_to_alkyl
        # already carries the -ol locant over to -yl unchanged.
        if re.search(r'(?:di|tri|tetra|penta)ol\b', alcohol_name.lower()):
            return None  # a multiplied -ol: our -OH is not the only one
        from ..decomposition.fragment_assembly import _alcohol_to_alkyl
        yl_name = _alcohol_to_alkyl(alcohol_name)
        if not yl_name or ' ' in yl_name or not yl_name.endswith('yl'):
            return None
        return yl_name
    # Case B: the fused/spiro/polycyclic ring producers in this codebase
    # build SUFFIX-FREE names (`allow_suffix_free`) -- a plain exocyclic -OH
    # never becomes a principal-characteristic-group suffix there, only a
    # "<locant>-hydroxy" PREFIX. That locant is still governed by the same
    # "lowest locant to the (sole) substituent" numbering a free valence
    # would earn, so recover it from the prefix instead of the suffix: strip
    # the "<locant>-hydroxy" prefix text out of the name (and the hyphen that
    # joined it to its neighbour, on whichever side survives) and APPEND
    # "-<locant>-yl" after the parent stem AS-IS. OPSIN 2.9.0 (verified
    # directly, not assumed) accepts the free-valence locant+"-yl" appended
    # to the UNMODIFIED terminal "-e" for a multi-component bracketed parent
    # ("...azepane]" + "-4-yl" -> "...azepane]-4-yl" parses correctly) --
    # unlike a von Baeyer/simple-ring name with no closing bracket at the
    # very end, where the "-e" -> "-yl" contraction is the standard spelling
    # but OPSIN parses the unstripped "-e-locant-yl" form too (verified:
    # "cyclohexane-4-yl" and "cyclohexan-4-yl" both parse to the same
    # structure). So appending unconditionally, without ever stripping the
    # "-e", is round-trip-safe for both shapes and avoids having to detect
    # which shape this name is.
    m = re.search(r'(\d+)-hydroxy', alcohol_name)
    if not m:
        return None  # no locant found (or hydroxy unlocanted) -- don't guess
    locant = m.group(1)
    start, end = m.span()
    if start > 0 and alcohol_name[start - 1] == '-':
        start -= 1
    remainder = (alcohol_name[:start] + alcohol_name[end:]).lstrip('-')
    if not remainder:
        return None
    yl_name = f'{remainder}-{locant}-yl'
    if ' ' in yl_name:
        return None
    return yl_name


_ORDER_INT_RE = re.compile(r"\d+")   # Lever G: parse RDKit's '[0,1,2,]' output-order prop


def _substituent_memo_key(mol, frag_atoms, attach_idx, allow_mancude,
                          best_effort_val):
    """Complete cache key for:func:`name_substituent` (M1 Lever C1).

    Every component is a proven wrong-name mechanism if omitted
    (`internal notes` Part 1 Lever C):

    1. rooted canonical isomeric fragment SMILES -- structure, stereo parities,
       charges, isotopes AND attachment position (``pentyl`` vs ``pentan-2-yl``);
    2. attach-in-fragment flag (rooted vs unrooted);
    3. external free-valence order (single vs double attachment -> ``-yl`` vs
       ``-ylidene``,;
    4. effective ``allow_mancude`` + best-effort tier (what Tier-4.5 may return);
    5. per-fragment-atom + per-in-fragment-bond ``_CIPCode`` state, in the
       canonical fragment-SMILES output order, so a difference in whether
       ``assign_stereochemistry`` has run on THIS mol object is a cache MISS,
       never a wrong-descriptor hit.

    Returns ``None`` to SKIP caching (fail-open -- a miss can never corrupt) when
    the canonical atom output-order prop is unavailable; the caller treats a
    raised exception the same way.
    """
    attach_in_frag = attach_idx in frag_atoms
    smi = Chem.MolFragmentToSmiles(
        mol, atomsToUse=sorted(frag_atoms),
        rootedAtAtom=(attach_idx if attach_in_frag else -1),
        isomericSmiles=True, canonical=True)
    frag_set = set(frag_atoms)
    _a = mol.GetAtomWithIdx(attach_idx)
    ext_free_valence = sum(
        int(b.GetBondTypeAsDouble()) for b in _a.GetBonds()
        if b.GetOtherAtom(_a).GetIdx() not in frag_set)
    if not mol.HasProp('_smilesAtomOutputOrder'):
        return None
    _order = [int(x) for x in _ORDER_INT_RE.findall(mol.GetProp('_smilesAtomOutputOrder'))]
    # Lever G (2026-09-12): one materialised atom tuple instead of two GetAtomWithIdx calls per
    # atom here and two per external bond below. Same atoms, same values, same tuple.
    _atoms = atoms_of(mol)
    atom_cips = tuple(
        (_atoms[i].GetProp('_CIPCode') if _atoms[i].HasProp('_CIPCode') else None)
        for i in _order)
    _rank = {idx: pos for pos, idx in enumerate(_order)}
    _bond_items = []
    for _b in bonds_of(mol):
        _bi, _ei = _b.GetBeginAtomIdx(), _b.GetEndAtomIdx()
        if _bi in frag_set and _ei in frag_set:
            _cip = _b.GetProp('_CIPCode') if _b.HasProp('_CIPCode') else None
            _bond_items.append(((_rank.get(_bi, -1), _rank.get(_ei, -1)), _cip))
    _bond_items.sort(key=lambda t: t[0])
    bond_cips = tuple(c for _, c in _bond_items)
    # Item-1 fix : MolFragmentToSmiles drops each fragment atom's bonds to
    # atoms OUTSIDE the fragment and back-fills implicit H, so two structurally
    # distinct substituents (terminal formyl-on-N `formamido` vs an acyl-bridge
    # `carbamoyl`) collide on the same `smi`. `ext_free_valence` above measures
    # only the ATTACH atom, so a non-attach atom's external bond was invisible.
    # Capture every fragment atom's external-bond-order multiset, in the canonical
    # fragment-SMILES output order, so the key is COMPLETE. Finer keys only turn a
    # false HIT into a MISS (recompute) -> byte-identity-safe.
    ext_bond_orders = tuple(
        tuple(sorted(
            b.GetBondTypeAsDouble()
            for b in _atoms[i].GetBonds()
            if b.GetOtherAtomIdx(i) not in frag_set))
        for i in _order)
    # Item-1 fix, a performance pass (, a review C1): the fragment SMILES + external-bond
    # ORDERS still describe only the fragment and the multiplicities of its
    # outward bonds -- NOT what those bonds lead to. Producers on the cascade read
    # BEYOND the fragment: ``_name_amino_branch`` walks the acyl carbon's external
    # chain and counts its carbons (in ONE molecule, a propanoyl and a butanoyl
    # ``-NH-C(=O)-`` bridge both present the ``{N,C,O}`` sub-fragment), and the
    # ring-cut checks read ``RingInfo`` for rings that close through external atoms
    # (a pyrrolidine C4 vs an acyclic C4 diyl). Those distinct inputs cannot be
    # told apart by any FRAGMENT-only signature.
    #
    # The complete-by-construction fix (mirroring ``_POLYFUNC_MEMO_CACHE``'s
    # ``(frozenset(sub_atoms), attach_idx)`` key, substituent_naming.py:1584): the
    # memo scope is ONE top-level naming call, so within it the producer's input is
    # exactly ``(mol, frag_atoms, attach_idx, flags)``. ``frozenset(frag_atoms)``
    # is a PERFECT fragment discriminator within a molecule -- two different
    # substituents are two different atom-index sets, so the propanoyl and butanoyl
    # ``{N,C,O}`` fragments (distinct positions) get distinct keys -- and raw
    # ``attach_idx`` splits the symmetric same-fragment/different-end case (C1.3,
    # ``butan-1-yl`` vs ``butan-4-yl``). The retained structural bits (``smi``,
    # ``ext_bond_orders``, CIP stamp) still discriminate mol OBJECTS, so a copy of
    # the mol with coinciding indices but a different structure cannot false-hit.
    # ``frozenset(frag_atoms)`` strictly subsumes what a canonical-rank signature
    # did (it separates even symmetric fragments a rank collapses), so the key is
    # only FINER -> a false HIT can become only a MISS (recompute) -> byte-identity
    # preserved.
    # a lever: the strict PIN twin (namer.py) runs with the four breadth flags OFF,
    # so a value cached at one configuration must never be served to another when the
    # twin shares this run's memo scope. ``best_effort_val`` is already the
    # ``best_effort_ctx`` bit; append the other three. Finer key => only MORE misses
    # (recompute), never a false hit => byte-identity-safe.
    return (smi, attach_in_frag, ext_free_valence,
            bool(allow_mancude), best_effort_val, atom_cips, bond_cips,
            ext_bond_orders, frozenset(frag_atoms), attach_idx,
            general_fallback_ctx.get(), allow_aromatic_general_ctx.get(),
            full_coverage_ctx.get())


def name_substituent(mol, frag_atoms, attach_idx, allow_mancude: bool = False):
    """Name a substituent fragment with the free-valence morphology requires.

    Thin gate over:func:`_name_substituent_cascade`, which holds the
    five-tier naming logic. The split exists because the cascade's tiers all
    build their token from the FRAGMENT alone -- they never see the attachment
    bond -- so none of them can know whether the free valence is single
    (``-yl``), double (``-ylidene``) or triple (``-ylidyne``). This function is
    the one place that holds the fragment and the attachment atom together, so
    the morphology is decided here, once, for every tier.

    Behaviour by free valence:

    * **one** (or undecidable -- a bridge, an aromatic linkage): the cascade
      runs and its answer is returned untouched. This is the overwhelmingly
      common case and is byte-identical to the pre- behaviour.
    * **two or three**: the carbon ylidene/ylidyne constructor gets first
      refusal; if it declines, the cascade runs and its answer is accepted ONLY
      if the token does not spell a single free valence. A ``-yl`` token on a
      doubly-bonded free valence names a DIFFERENT molecule, so it is refused
      rather than shipped -- ``None`` under ``allow_mancude`` (the Tier-4.5
      de-masking convention) and the ``'substituent'`` unnameable sentinel
      otherwise, both of which fail closed downstream.

    The correct multivalent heteroatom prefixes (``oxo``, ``sulfanylidene``,
    ``imino``,...) spell no ``-yl`` morpheme, so they pass the guard untouched.

    References:
        IUPAC 2013 (free-valence morphology), (lowest locant
        for the free valence).
    """
    # M2 Task 1 (tier propagation): the whole-molecule best-effort path
    # publishes its tier via ``best_effort_ctx`` (metrics/provenance.py; set
    # at namer.py:3021-3022, reset at:3404-3405). ~25 callers invoke
    # ``name_substituent`` with the default ``allow_mancude=False``, so a
    # best-effort whole-molecule run still gated substituent naming at the
    # DEFAULT tier here and declined fragments the cascade names at
    # best-effort (measured: internal notes).
    # Honor the ambient context here, once, so every caller inherits it.
    # DEFAULT/PIN tier is unchanged: ``best_effort_ctx`` is unset there, so
    # ``.get`` is falsy and this is byte-identical to the prior behaviour.
    _best_effort_val = best_effort_ctx.get()
    allow_mancude = allow_mancude or bool(_best_effort_val)

    def _body(mol=mol, frag_atoms=frag_atoms, attach_idx=attach_idx, allow_mancude=allow_mancude):
        # The bond-order read and the ylidene construction come from the SHARED
        # primitive, not from a private copy here -- this function and
        # carbon_free_valence_prefix used to hold the same two calls, which is the
        # duplication the class is meant to remove. ``free_valence`` below is the
        # TRUE bond order even for a non-carbon attachment (the primitive only
        # declines to CONSTRUCT for those), so the heteroatom guard further down is
        # unchanged.
        frag_atoms_set = set(frag_atoms)
        # ⚠ NORMALIZE the attachment, so the documented contract is self-enforcing.
        #
        # This function's contract is that ``attach_idx`` is the atom WITHIN
        # ``frag_atoms`` that bonds to the parent, but nothing enforced it, and TWO
        # conventions circulate in this tree: ``SubstituentInfo.attach_mol_idx`` is the
        # PARENT-side atom. Production callers each normalize on their own -- explicitly
        # at ``rules/ring_substituents.py:1869-1879``, implicitly at
        # ``assembly/general_engine.py:760`` (neighbours of ``attach_mol_idx``
        # restricted to the fragment) -- so this is a NO-OP for them, verified by an
        # unchanged a dev split.
        #
        # It is here because an unenforced convention silently returns a WRONG or None
        # token instead of failing, and it already cost real work twice in one session:
        # a `carboxy` guard that was green in a unit test and dead in production, and a
        # measurement that overstated the substituent-coverage bucket by 34 rows
        # (109 real vs 143 reported) because probe scripts passed the raw parent-side
        # atom. Measured on 8 probes with the raw value: only 2 named; after
        # normalising, 8 named -- `2-oxoethyl`, `2-carboxyethyl`, `8-oxooctyl` and
        # `1-hydroxyethyl` (locant present) among them.
        # Wrapped, because `attach_idx` is not guaranteed to be a valid atom index:
        # `test_invalid_attach_idx_does_not_crash` passes an out-of-range one on
        # purpose, and `GetAtomWithIdx` raises `RuntimeError: Range Error` rather than
        # returning None. Leaving it to raise turned a documented degradation contract
        # into a crash -- caught by that test, which is exactly why it exists.
        if attach_idx is not None and attach_idx not in frag_atoms_set:
            try:
                _fs = next(
                    (n.GetIdx() for n in mol.GetAtomWithIdx(attach_idx).GetNeighbors()
                     if n.GetIdx() in frag_atoms_set),
                    None,
                )
            except Exception:  # noqa: BLE001 - an invalid index is a decline, not a crash
                _fs = None
            if _fs is not None:
                attach_idx = _fs
        verdict = carbon_free_valence_prefix(mol, frag_atoms_set, attach_idx)
        if verdict.prefix is not None:
            return verdict.prefix
        free_valence = verdict.free_valence

        # Phase1 B1 + BB:1703): a NITROGEN attached by a DOUBLE bond is
        # an ylidene free valence (hydrazin-1-ylidene for =N-N<, {R}imino for =N-R),
        # NOT a -yl. carbon_free_valence_prefix declines the non-carbon root and the
        # -NH-R amino intercept below requires all-single bonds, so without this the
        # cascade builds '...hydrazinyl' and the guard refuses it -> the whole
        # branch abstains (the measured decorated-steroid / hydrazone class). The
        # constructor is candidate-only; round-trip-verifies downstream.
        if (free_valence == 2 and attach_idx is not None
                and 0 <= attach_idx < mol.GetNumAtoms()
                and mol.GetAtomWithIdx(attach_idx).GetSymbol() == 'N'):
            _nyl = _nitrogen_ylidene_prefix(
                mol, frag_atoms_set, attach_idx, free_valence)
            if _nyl is not None:
                return _nyl

        # breadth: an O-ROOTED ether substituent -O-R is an ALKOXY
        # prefix ('methoxy'/'ethoxy'/'phenoxy'), NOT the cascade's carbon-rooted
        # 'hydroxy(R)'. `carbon_free_valence_prefix` declines a non-carbon root, so
        # without this the cascade mis-roots -O-CH3 as `hydroxymethyl` -- a DIFFERENT
        # molecule (-O-CH3 vs -CH2-OH), which then abstains on (breadth loss,
        # latent wrong-molecule). The PIN path already names it correctly via a
        # different entry; this fixes the general-engine `name_substituent` path.
        # Delegates to the existing `get_alkoxy_prefix` (handles alkyl/aryl R and
        # fails closed on shapes it cannot name). Uses an element-dispatched
        # oxygen-subgraph prefix construction.
        if attach_idx is not None and 0 <= attach_idx < mol.GetNumAtoms():
            _root = mol.GetAtomWithIdx(attach_idx)
            if (_root.GetSymbol() == 'O' and _root.GetFormalCharge() == 0
                    and _root.GetTotalNumHs() == 0
                    and not _root.IsInRing()):  # a ring O spanning the parent would
                    # cut the ring if named as an acyclic alkoxy -> a DIFFERENT
                    # molecule; exclude by construction (a review finding 5 -- no
                    # reachable case found, but this makes the intercept airtight
                    # rather than relying on the recovery-lane RT to catch it).
                _o_nbrs = list(_root.GetNeighbors())
                _parent = [n.GetIdx() for n in _o_nbrs
                           if n.GetIdx() not in frag_atoms_set]
                _rside = [n.GetIdx() for n in _o_nbrs
                          if n.GetIdx() in frag_atoms_set]
                if (len(_parent) == 1 and len(_rside) == 1
                        and mol.GetAtomWithIdx(_rside[0]).GetSymbol() == 'C'
                        and all(mol.GetBondBetweenAtoms(attach_idx, x).GetBondType()
                                == Chem.BondType.SINGLE
                                for x in (_parent[0], _rside[0]))):
                    # The frag-side neighbour MUST be carbon -- a true alkoxy -O-C(...).
                    # A peroxide -O-O-R has an O frag-side and its PIN is '(alkylperoxy)'
                    #, NOT '(alkoxyoxy)': feeding the O side to get_alkoxy_prefix
                    # regressed the (ethylperoxy)benzene / (methylperoxy) golds
                    # (same molecule, wrong PIN). Peroxides fall through to the cascade,
                    # which already names them correctly.
                    #
                    # tail #25: -O-C(=O)-/-C(=S)-/-C(=N)- is an ACYLOXY (ester,
                    # carbamate, carbonate, thioester), NOT a plain alkoxy. Feeding it
                    # to get_alkoxy_prefix mis-names it -- -OC(=O)NHMe became the
                    # WRONG 'carbamoylmethoxy' (a different molecule, -vetoed).
                    # Defer any acyloxy R-side carbonyl to the cascade, whose Tier-0.5
                    # prefix-form check names the carbamate '(N-Rcarbamoyl)oxy'.
                    _rc = mol.GetAtomWithIdx(_rside[0])
                    _is_acyloxy = any(
                        b.GetBondTypeAsDouble() == 2.0
                        and b.GetOtherAtom(_rc).GetSymbol() in ('O', 'S', 'N')
                        for b in _rc.GetBonds())
                    if not _is_acyloxy:
                        from .substituent_prefix_forms import get_alkoxy_prefix
                        _alk = get_alkoxy_prefix(
                            mol, (attach_idx, _parent[0], _rside[0]), [_parent[0]])
                        if _alk:
                            return _alk

                # Phase1 B5: SILYLOXY -O-[Si]< is '{silyl}oxy'
                # ('(trimethylsilyl)oxy'), NOT the cascade's mis-rooted
                # 'hydroxy{silyl}' (a DIFFERENT molecule -Si-OH). Silicon is a
                # substitutive-nomenclature parent hydride (silane -> silyl, /,
                # and the DIRECT -Si case already names correctly, so this routes the
                # O-attached case through the SAME recursive namer + an 'oxy' morpheme.
                # Handles TMS/TBS/TIPS uniformly; RT-backstopped by downstream.
                elif (len(_parent) == 1 and len(_rside) == 1
                        and mol.GetAtomWithIdx(_rside[0]).GetSymbol() == 'Si'
                        and all(mol.GetBondBetweenAtoms(attach_idx, x).GetBondType()
                                == Chem.BondType.SINGLE
                                for x in (_parent[0], _rside[0]))):
                    from ..errors import is_refusal_sentinel
                    from .naming_utils import enclose_if_compound
                    _si_frag = _subtree_atoms(
                        mol, frag_atoms_set, _rside[0], {attach_idx})
                    _si = name_substituent(mol, sorted(_si_frag), _rside[0])
                    if _si and not is_refusal_sentinel(_si) and ' ' not in _si:
                        return f"{enclose_if_compound(_si)}oxy"

            # breadth /: an N-ROOTED substituent -NH-R /
            # -N(R)R' / -NH-C(=O)R is an amino/amido PREFIX (methylamino / dimethylamino
            # / acetamido), NOT the cascade's carbon-rooted misroot -- `carbon_free_valence_
            # prefix` declines the non-carbon root, so the cascade re-roots -NH-CH3 as
            # `aminomethyl` (that is -CH2-NH2, a DIFFERENT molecule), -N(CH3)2 as the
            # unparseable `amino-N-methylmethyl`, -NH-C(=O)CH3 as `carbamoylmethyl`. Same
            # class the alkoxy intercept above fixes; delegates to the existing
            # `_name_amino_branch` (the builder the PIN path already uses). It returns None
            # for hydrazinyl / diazenyl / N-oxide / sulfonamido, which then fall through
            # unchanged (already handled elsewhere, or a named follow-up). Guards mirror the
            # alkoxy/peroxide ones: uncharged, not in a ring, all single bonds (an imine /
            # diazo N is double-bonded and must not be read as an amine).
            elif (_root.GetSymbol() == 'N' and _root.GetFormalCharge() == 0
                    and not _root.IsInRing()
                    and all(b.GetBondType() == Chem.BondType.SINGLE
                            for b in _root.GetBonds())):
                _parent_n = {n.GetIdx() for n in _root.GetNeighbors()
                             if n.GetIdx() not in frag_atoms_set}
                if _parent_n:
                    _amino = _name_amino_branch(
                        mol, frag_atoms_set, attach_idx, _parent_n)
                    if _amino:
                        return _amino
                    # a phase (11C2 #17),: a BARE terminal -NH-NH2
                    # free-valence leaf is the RETAINED substituent prefix
                    # 'hydrazinyl' (H2N-NH-), not the compositional 'aminoamino' the
                    # amino cascade falls through to (amino recursing into amino).
                    # Fires ONLY for the two-nitrogen -NH-NH2 shape: the attach N
                    # (single-bonded, uncharged, non-ring -- gated by the elif above)
                    # has exactly ONE in-fragment neighbour, that neighbour is a
                    # TERMINAL -NH2 nitrogen (no heavy neighbour but the attach N,
                    # uncharged, no radical/isotope, non-ring, all single bonds), and
                    # the fragment is exactly those two atoms. A SUBSTITUTED far
                    # nitrogen (2-R-hydrazinyl), an =N-N< (hydrazinylidene, handled in
                    # _nitrogen_ylidene_prefix), and an acyl hydrazide named as the
                    # senior -hydrazide SUFFIX (which is claimed before this
                    # prefix-leaf namer is ever reached) are all NOT matched and fall
                    # through unchanged. Offer-RT-gated downstream.
                    _inner_n = [nb.GetIdx() for nb in _root.GetNeighbors()
                                if nb.GetIdx() in frag_atoms_set
                                and mol.GetAtomWithIdx(nb.GetIdx()).GetSymbol() == 'N']
                    _inner_all = [nb.GetIdx() for nb in _root.GetNeighbors()
                                  if nb.GetIdx() in frag_atoms_set]
                    if (len(frag_atoms_set) == 2 and len(_inner_n) == 1
                            and len(_inner_all) == 1):
                        _far = mol.GetAtomWithIdx(_inner_n[0])
                        _far_heavy = [nb for nb in _far.GetNeighbors()
                                      if nb.GetAtomicNum() > 1
                                      and nb.GetIdx() != attach_idx]
                        if (_far.GetFormalCharge() == 0
                                and not _far.GetNumRadicalElectrons()
                                and not _far.GetIsotope()
                                and not _far.IsInRing()
                                and not _far_heavy
                                and all(b.GetBondType() == Chem.BondType.SINGLE
                                        for b in _far.GetBonds())):
                            return "hydrazinyl"
                    # sub-lever A: best-effort -NH-R with a RING-bearing R
                    # (cyclohexylamino, benzylamino, [(4-fluorophenyl)methyl]amino).
                    # _name_amino_branch declines a saturated-ring/ring-on-chain R;
                    # mirror the O-rooted alkoxy path which already recurses ring R.
                    if allow_mancude:
                        _amino_ring = _name_amino_ring_branch(
                            mol, frag_atoms_set, attach_idx, _parent_n)
                        if _amino_ring:
                            return _amino_ring

            # sub-lever A: best-effort -S-R with a RING-bearing R
            # (cyclohexylsulfanyl, benzylsulfanyl, (4-hydroxycyclohexyl)sulfanyl).
            # The chalcogen-rooted sulfanyl cascade tier declines a saturated-ring /
            # ring-on-chain R; mirror the O-rooted alkoxy + N-rooted amino ring paths.
            # A non-ring -S-R never matches (the helper requires a ring) -> falls
            # through to the cascade unchanged; PIN default is byte-identical.
            elif (allow_mancude and _root.GetSymbol() == 'S'
                    and _root.GetFormalCharge() == 0
                    and _root.GetDegree() == 2
                    and all(b.GetBondType() == Chem.BondType.SINGLE
                            for b in _root.GetBonds())):
                _thio_ring = _name_thio_ring_branch(
                    mol, frag_atoms_set, attach_idx)
                if _thio_ring:
                    return _thio_ring

        token = _name_substituent_cascade(
            mol, frag_atoms, attach_idx, allow_mancude=allow_mancude)

        # candidate ledger: FRAGMENT scope, recorded at the cascade's single
        # production call site rather than by wrapping the cascade itself (a rename
        # would break the AST assertions in
        # tests/unit/assembly/test_substituent_enumerator_tier_05.py).
        #
        # Scope matters more here than anywhere else in the instrument. Measured
        # (PE1 plan): this cascade produces genuinely CORRECT names like
        # 'N,N-diethylethanamine' for a fragment of a molecule whose whole-molecule
        # candidate was sentinel-spliced. Round-tripping a fragment name against the
        # whole input would file it as a wrong molecule and manufacture a large fake
        # producer-correctness class, so consumers must filter on scope.
        if token is not None:
            _ledger_record("substituent_enumerator._name_substituent_cascade",
                           _LedgerStage.PRODUCED, token,
                           scope=_LedgerScope.FRAGMENT,
                           detail=f"allow_mancude:{allow_mancude}")

        # (generalizes SP2.1'): a PLAIN acyloxy ester substituent (``-O-C(=O)-R``
        # attached via its ester O) whose acyl is SYSTEMATIC is named by the cascade,
        # NOT as the ``<acyl>oxy`` PIN prefix, but as an oxa-replacement
        # chain (``…-2-oxo-1-oxabutyl``) -- because Tier-1.95 above intercepts only
        # RETAINED acyls (a static ``FRAGMENT_NAME_CACHE`` lookup) and a systematic acyl
        # falls through. trace (a project rule) REFUTED the SP2.1' report's
        # "OPSIN-grammar-invalid" premise: that oxa-chain form round-trips EXACT in
        # OPSIN and already ships on the best-effort tier -- so this is a SPELLING gap
        # there, and on the DEFAULT/PIN tier a BREADTH gap (the systematic acyl frag is
        # instead dropped -> the whole molecule abstains). Route it through the SAME
        # recognizer SP2.1' built (``composer._acyloxy_prefix_for_frag`` ->
        # ``rules.lipids._acyloxy_for_site`` -> the full acid engine), which is
        # FAIL-CLOSED (ester O = attach; exactly one terminal ``=O`` and <=1 all-carbon
        # R on the carbonyl C; carbamate/carbonate/thiono excluded; the whole acyl side
        # must be self-contained -- else ``None``, so no atom is dropped by claiming the
        # shape). The override fires ONLY when it DIFFERS from the cascade token, so a
        # RETAINED acyl (Tier-1.95 already returns ``acetyloxy``) and a pure-ether
        # oxa-replacement chain (no carbonyl -> recognizer ``None``) stay BYTE-IDENTICAL.
        # Placed BEFORE the B2 stereo-completeness guard below so a stereo-dropped acyl
        # name is still refused; the caller's OPSIN round-trip is the 0-wrong
        # net on the assembled name. The recognizer is imported lazily to avoid the
        # composer<->substituent_enumerator import cycle (same pattern as the
        # ``from.composer import …`` calls elsewhere in this module).
        if (attach_idx is not None and 0 <= attach_idx < mol.GetNumAtoms()
                and mol.GetAtomWithIdx(attach_idx).GetAtomicNum() == 8):
            try:
                from .composer import _acyloxy_prefix_for_frag
                _acyloxy = _acyloxy_prefix_for_frag(mol, frag_atoms, attach_idx)
            except Exception:  # a producer helper must never crash branch naming
                _acyloxy = None
            if _acyloxy and _acyloxy != token and ' ' not in _acyloxy:
                token = _acyloxy

        if free_valence in (2, 3) and _token_asserts_single_free_valence(token):
            # Decorated ring-ylidene /: a DECORATED monocyclic
            # carbocycle joined to its parent by a DOUBLE bond is an '-ylidene'.
            # carbon_free_valence_prefix builds the BARE 'cyclohexylidene' but declines
            # any decoration, so the cascade produced the '-yl' token
            # ('4-fluorocyclohexyl'); the free valence sits on a NON-AROMATIC RING
            # CARBON, so swap the TERMINAL free-valence morpheme (re-anchored to the
            # SAME attachment atom -- nothing else moves) and VERIFY the result via the
            # morphology reader. RT / backstop any error. Only free-valence 2:
            # a ring carbon cannot carry an '-ylidyne' (free valence 3).
            if free_valence == 2:
                _a = mol.GetAtomWithIdx(attach_idx)
                # 0-WRONG guard (a project rule): the decorated-ring substituent namer
                # DROPS defined ring stereo, so a stereo-bearing fragment would yield a
                # stereo-STRIPPED '-ylidene' (a different molecule -- #18's full InChIKey
                # loses its -HOWMLNFFSA stereo layer). Convert only when the fragment
                # carries NO defined stereo; a stereo case abstains safely until the
                # composite ring-substituent stereo path is built.
                # A stereo-bearing fragment is safe to convert ONLY if the token
                # already CARRIES its stereo -- else the ylidene STRIPS it (and
                # does NOT catch a strip: its RT compares connectivity). Count the
                # fragment's defined stereo (stereocentres + fully-internal stereo
                # bonds) and the token's cited descriptors; convert only when the token
                # COVERS them. The descriptor block may be NESTED ('2-[(1Z)-…]ethyl'),
                # so a startswith('(') test is insufficient.
                _frag_stereo_n = sum(
                    1 for _i in frag_atoms_set
                    if mol.GetAtomWithIdx(_i).GetChiralTag()
                    != Chem.ChiralType.CHI_UNSPECIFIED
                ) + sum(
                    1 for b in bonds_of(mol)
                    if b.GetStereo() in (Chem.BondStereo.STEREOE,
                                         Chem.BondStereo.STEREOZ,
                                         Chem.BondStereo.STEREOCIS,
                                         Chem.BondStereo.STEREOTRANS)
                    and b.GetBeginAtomIdx() in frag_atoms_set
                    and b.GetEndAtomIdx() in frag_atoms_set)
                _token_descr_n = len(re.findall(r'[RSrsEZ](?=[,)])', token))
                _stereo_ok = (_frag_stereo_n == 0) or (_token_descr_n >= _frag_stereo_n)
                # The free valence may sit on a RING carbon (cyclohexylidene) OR a
                # CHAIN carbon (a decorated '…ethyl' whose C1 double-bonds the parent,
                # e.g. a seco-steroid side chain '2-[…cyclohexylidene]ethylidene'). Both
                # are ylidene; bare cases are owned by carbon_free_valence_prefix
                # upstream, so only DECORATED tokens reach here. Aromatic ring carbons
                # are excluded (a quinoid/mancude ylidene is a different construction).
                if (_a.GetSymbol() == 'C' and not _a.GetIsAromatic() and _stereo_ok
                        and token.endswith('yl') and not token.endswith('idene')):
                    from ..validation.name_morphemes import free_valence_morphology
                    _cand = token[:-2] + 'ylidene'
                    _est = free_valence_morphology(_cand)
                    if _est.confident and _est.free_valences == 2:
                        return _cand
            from ..metrics.abstention import AbstentionCode, record_abstention
            record_abstention(AbstentionCode.BRANCH_UNNAMEABLE,
                              detail='p29_2_free_valence_morphology')
            logger.debug(
                "P-29.2 guard: refused %r for a free valence of %d at atom %d",
                token, free_valence, attach_idx)
            return None if allow_mancude else "substituent"

        # B2 stereo-completeness guard: a substituent whose fragment carries a
        # DEFINED-stereo double bond MUST cite its E/Z descriptor, else the token names
        # a different (stereo-dropped) molecule. Legacy alkenyl tiers emit an
        # `Xethenyl` form without the descriptor (e.g. -CH=CH-OMe -> `methoxyethenyl`);
        # the OPSIN validity gate's stereo carve-out then SHIPS it as a wrong molecule
        # (a project rule -- a review-b2 BLOCKER 1). Decline so the caller fails closed. The
        # descriptor block is `(1E)`/`(E)`/`(1E,2R)`; a legitimate stereogenic-C=C name
        # always carries it, so this can only turn a wrong output into an abstention.
        if token and token != "substituent":
            import re as _re
            if not _re.search(r"[(,]\d*[EZ][),]", token) and any(
                    b.GetBondType() == Chem.BondType.DOUBLE
                    and b.GetStereo() in (Chem.BondStereo.STEREOE,
                                          Chem.BondStereo.STEREOZ,
                                          Chem.BondStereo.STEREOCIS,
                                          Chem.BondStereo.STEREOTRANS)
                    and b.GetBeginAtomIdx() in frag_atoms_set
                    and b.GetEndAtomIdx() in frag_atoms_set
                    for b in bonds_of(mol)):
                logger.debug("B2 stereo-completeness: refused %r (defined C=C, no E/Z)",
                             token)
                return None if allow_mancude else "substituent"

        # Phase B3 -- branch-fallback via the unconditional universal recursive
        # namer. The five-tier cascade above declined this branch (``token`` is the
        # ``'substituent'`` sentinel, or ``None`` under ``allow_mancude``), which
        # ``general_engine`` translates into ``_refuse("branch unnameable...")`` --
        # failing the WHOLE enclosing candidate. RE-a trace (task-B34) confirmed this
        # ``name_substituent`` decline is THE branch site that cascades to
        # whole-molecule abstention (309 declines over the 44 a dev split best-effort
        # abstainers; ``composer.classify_and_name_fragment`` recorded 0). Rather
        # than decline, re-enter the universal namer on JUST this branch subgraph
        # and render an ugly-but-valid ``-yl``/``-ylidene``/``-ylidyne`` prefix
        # (cited at the branch's own attachment locant, correct morphology
        # from ``free_valence``). The general engine binds the WHOLE ``frag`` atom
        # set to this returned string (``TokenBinding(frag, prefix)``), so E1 atom
        # coverage holds regardless of the token content; (the caller's
        # OPSIN round-trip / ``_rt_match`` superset ladder) is the 0-wrong net --
        # a branch name that does not round-trip (or a stereo CONFLICT) is
        # suppressed to abstain, a constitution-correct one (incl. a safe
        # stereo-OMISSION) ships. Fail-closed: a universal ``None`` (genuine
        # residual) or a malformed/multi-word result keeps the existing decline.
        #
        # BEST-EFFORT ONLY: gated on ``best_effort_ctx`` (True iff
        # ``general_fallback_unverified``), so the PIN / valid / complete tiers --
        # which also call ``name_substituent`` -- are byte-identical. An ugly
        # universal ``-yl`` is a best-effort degrade, never a PIN component; a
        # PIN-tier branch decline must still abstain, exactly as before.
        #
        # BOND-ORDER GUARD: ``free_valence`` (= ``_free_valence_at_attachment`` via
        # ``verdict.free_valence``) is ``None`` for a bridge/spiro, 2+ linkage
        # bonds), an aromatic or dative linkage, or no linkage at all -- its
        # docstring (:380) states "None means UNDECIDABLE and must never be read as
        # 1", yet ``_render_as_substituent`` would silently default ``None`` ->
        # ``"yl"`` (``universal_substituent.py`` ``{1:"yl",2:"ylidene",3:"ylidyne"}
        #.get(bond_order, "yl")``), naming a DIFFERENT free-valence morphology. So
        # only offer the universal fallback when the free valence is a decidable
        # single-attachment 1/2/3; on ``None`` (or any other value) keep the
        # existing decline (fall through to ``return token``) rather than emit a
        # wrong-morphology candidate that only the round-trip net would catch.
        from ..errors import is_refusal_sentinel
        from ..metrics.provenance import best_effort_ctx
        if (best_effort_ctx.get()
                and (token is None or is_refusal_sentinel(token))
                and free_valence in (1, 2, 3)):
            try:
                from .universal_substituent import name_universal_substituent_prefix
                _uni = name_universal_substituent_prefix(
                    mol, frag_atoms_set, attach_idx, bond_order=free_valence)
            except Exception:  # a producer bug must never crash branch naming
                _uni = None
            if _uni and not is_refusal_sentinel(_uni) and ' ' not in _uni:
                return _uni

            # M2 Task 2: Phase-B3 above also declined (its own narrower ring
            # logic doesn't reach fused/spiro shapes the FULL general engine
            # already names) -- try routing the branch to the whole-molecule
            # engine via name_compound + an -OH-locant-carried -yl conversion.
            # Scoped to free_valence == 1 (a plain -yl): the OH-capping trick
            # only recovers a SINGLE free valence's locant, never an ylidene/
            # ylidyne's. See _route_fragment_to_general_engine's docstring.
            if free_valence == 1:
                _routed = _route_fragment_to_general_engine(
                    mol, frag_atoms, attach_idx)
                if _routed:
                    return _routed

        return token

    # --- M1 Lever C1: scoped-per-call substituent-naming memo ----------
    # The dispatch cascade, the gate-rejection retries and the recursive Tier-4
    # cascade all re-name the SAME fragment many times inside ONE molecule
    # (measured 288/371 duplicate calls, ~12% of marginal runtime). Memoize on a
    # key that is COMPLETE for the emitted name (see ``_substituent_memo_key``),
    # so a hit is byte-identical to a recompute -- verified continuously by
    # ``ORTHONYM_MEMO=verify``. Key construction is fail-open: any error, or an
    # unavailable canonical output-order prop, skips the cache and recomputes (a
    # miss can never corrupt). Scope + modes live in ``assembly/memo.py``; plan in
    # `internal notes` Step 1.
    try:
        _memo_key = _substituent_memo_key(
            mol, frag_atoms, attach_idx, allow_mancude, _best_effort_val)
    except Exception:  # noqa: BLE001 -- a key-construction error is a cache skip
        _memo_key = None
    if _memo_key is None:
        return _body()
    from .memo import cache_or_compute
    return cache_or_compute("name_substituent", _memo_key, _body)


def name_ylidene_substituent(mol, frag_atoms, attach_idx):
    """Name a fragment whose bond to its parent is DOUBLE, or ``None``.

    The shared entry point for the namers that cite a doubly-bonded fragment --
    hydrazone, semicarbazone, azine, the cumulative ium/ide chain, the (3)
    ketene branch. Each of them used to spell the morphology itself::

        yl = name_substituent(mol, frag, c)
        if not yl.endswith("yl"):
            return None
        ... f"{yl}idene"...

    which decided the free valence a second time, in the consumer, by rewriting
    a token. It only worked while the pipeline was returning the WRONG
    single-valence token for a double bond; correcting the pipeline made every
    such consumer reject its own correct input.

    Here the producer owns the morphology and the consumer VERIFIES it: the
    token comes back from ``name_substituent`` already carrying the
    ending its attachment bond earned, and is returned only if its own text
    confirms the two free valences. Nothing is appended, so there is no second
    place for the two to disagree.
    """
    from ..validation.name_morphemes import free_valence_morphology
    token = name_substituent(mol, frag_atoms, attach_idx)
    estimate = free_valence_morphology(token)
    if estimate.confident and estimate.free_valences == 2:
        return token
    return None


def name_ylidyne_substituent(mol, frag_atoms, attach_idx):
    """Name a fragment whose bond to its parent is TRIPLE, or ``None``.

    The ``-ylidyne`` sibling of:func:`name_ylidene_substituent`: for a
    fragment attached by a triple bond (three free valences), e.g. the
    ``CH3-C#`` of a nitrile imide ``CC#[N+][N-]C`` -> ``ethylidyne``. As with
    the ylidene entry point, the token comes back from:func:`name_substituent`
    already carrying the morphology its attachment bond earned, and is returned
    only if its own text confirms the three free valences -- nothing is
    appended, so producer and consumer cannot drift about the free valence.
    """
    from ..validation.name_morphemes import free_valence_morphology
    token = name_substituent(mol, frag_atoms, attach_idx)
    estimate = free_valence_morphology(token)
    if estimate.confident and estimate.free_valences == 3:
        return token
    return None


# ---------------------------------------------------------------------------
# The COMPOSED-PREFIX organyl chokepoint
#
# A composed heteroatom prefix -- alkyl+amino, alkyl+oxy, alkyl+sulfanyl,
# alkyl+peroxy,... -- has an organyl half and a composing half. Every producer
# of such a prefix used to name the organyl half by COUNTING ITS CARBONS and
# indexing an alkyl-stem table (``get_alkyl_name(n)`` / ``ALKOXY_NAMES[n]``).
#
# A carbon count is not a constitution. butyl, 2-methylpropyl, butan-2-yl and
# tert-butyl are FOUR different C4H9 groups; counting carbons named all four
# 'butyl', so three of the four names described a molecule other than the one
# that was drawn. The three functions below are the one place the organyl half
# is named, cited and multiplied, so the class cannot regrow one site at a time.
# ---------------------------------------------------------------------------

# (BB 27667-27691): the retained CONTRACTED alkoxy prefixes. These
# are the only alkyl stems whose 'yl' contracts with 'oxy'; C5+ keeps the alkyl
# name whole ('pentyloxy'). 'tert-butoxy' is listed separately because it is a
# retained name in its own right and is explicitly NOT 'tert-butyloxy'
# (BB 55662 "tert-butoxy* (unsubstituted)... (not tert-butyloxy)").
_ALKOXY_CONTRACTED_STEMS = {
    'methyl': 'methoxy',
    'ethyl': 'ethoxy',
    'propyl': 'propoxy',
    'butyl': 'butoxy',
}

# A free valence that carries an EXPLICIT locant ('propan-2-yl', 'butan-2-yl',
# 'hex-5-en-2-yl'). Such a name is cited WHOLE inside enclosing marks with the
# composing suffix outside them -- '(propan-2-yl)oxy' -- because contracting it
# would bury the locant that identifies the attachment carbon.
_LOCANT_BEARING_YL_RE = re.compile(r'-\d+(?:,\d+)*-yl$')


def composed_prefix_organyl_name(mol, frag_atoms, attach_idx):
    """The PIN substituent-prefix token for an organyl fragment, or ``None``.

    THE organyl-naming step for every composed heteroatom prefix, and the
    root-cause replacement for ``get_alkyl_name(carbon_count)`` at those sites.
    The fragment is routed through the shared audited cascade
    (:func:`name_substituent`), which PERCEIVES the branching a count erases.

    Strict acceptance filter, following the Phase-3
    ``substituent_purity.organyl_prefix_name`` pattern: a single prefix TOKEN or
    nothing. Every refusal sentinel is recognised by the ONE shared
    :func:`errors.is_refusal_sentinel` predicate -- a sentinel welded into a
    prefix slot becomes part of a name that reads as success -- and a multi-word
    result is refused too, because a composed prefix must never be assembled
    around a whole compound name.

    Pure: no mol mutation.
    """
    if not frag_atoms:
        return None
    name = name_substituent(mol, sorted(frag_atoms), attach_idx,
                            allow_mancude=False)
    from ..errors import is_refusal_sentinel
    if is_refusal_sentinel(name) or ' ' in name:
        return None
    return name


def _assemble_amino_prefix_core(branch_entries):
    """The amino-prefix CORE ('methyl(propan-2-yl)amino'), or ``None``.

    A two-line delegation to ``composer._assemble_decorated_amino_prefix`` --
    deliberately NOT a second assembler. That function is documented as "the ONE
    assembler" and already owns every ordering and marking decision:

      * (the Blue Book, section " ENCLOSING MARKS") *"For
        mononuclear parent hydrides with two or more substituents the first
        cited substituent never has enclosing marks unless it includes a locant.
        The second and further substituents are each enclosed with parentheses
        even for simple substituents."* -- the nitrogen is that mononuclear
        parent, which is why 'ethyl(methyl)amino' carries an inner pair the
        symmetric 'dimethylamino' does not;
      * POLYFUNCTIONAL COMPOUNDS, the Blue Book
        ``2-[di(butan-2-yl)amino]butan-2-ol (PIN)`` -- the outer bracket over
        the whole prefix is applied downstream, so this returns the core
        UNENCLOSED, matching every sibling return in ``_name_amino_branch``.

    The import is lazy because ``composer`` imports this module.
    """
    from .composer import _assemble_decorated_amino_prefix
    return _assemble_decorated_amino_prefix(branch_entries, enclose=False)


def cite_organyl_in_composed_prefix(token):
    """Cite ``token`` inside a composed prefix with its enclosing marks.

    A SIMPLE prefix is cited BARE: 'butylamino' and 'methylsulfanyl'
    (BB 18089 "-NH-CH3 methylamino (preferred prefix)"; BB 27651 "CH3-S-
    methylsulfanyl (preferred prefix)"). A retained ITALICISED prefix is simple
    too -- 'tert-butyl' is a preferred prefix cited bare in
    'tert-butyldi(methyl)phosphane (PIN)', BB 16286).

    A COMPOUND prefix takes enclosing marks with the composing suffix OUTSIDE
    them: '(propan-2-yl)oxy' and '(butan-2-yl)oxy' are the preferred prefixes
    , BB 27683/27687) and '(chloromethyl)amino' likewise (BB 18112).
    The marks go around the ORGANYL, never around the whole composed prefix.

    The italicised-prefix carve-out is decided by the ONE shared
    :func:`naming_utils.italicized_prefix_is_bare`, never re-spelled here.
    """
    if not token:
        return None
    from .naming_utils import enclose_if_compound, italicized_prefix_is_bare
    # The retained italicised prefixes are simple and cited bare ('tert-butyl');
    # the primitive judges the REMAINDER, so 'tert-butylsulfanyl' still encloses.
    if italicized_prefix_is_bare(token):
        return token
    # The compound test is the SHARED enclose_if_compound, which already unions
    # needs_brackets with is_complex_substituent. Neither is complete alone --
    # is_complex_substituent('hydroxymethyl') is False while needs_brackets is
    # True -- and spelling a private union here is how this class regrew before.
    return enclose_if_compound(token)


def _is_bare_locant_bearing_alkyl(token):
    """True iff ``token`` is an UNSUBSTITUTED chain alkyl whose free valence
    carries an explicit locant -- 'propan-2-yl', 'butan-2-yl', 'hex-5-en-2-yl'.

    Grounded in the real chain-stem table rather than a morphological guess: the
    token must begin with a tabulated stem ('prop', 'but',...) immediately
    followed by its 'an'/'en'/'yn' saturation syllable, so a decorated name like
    '2-chloropropan-2-yl' (which begins with a locant) can never match.

    This is the (a) class -- "simple substituent prefixes having
    locants" (BB 7085; clause (b) is the separate ene/yne case) --
    as distinct from the (a) compound class.
    """
    if not token or not _LOCANT_BEARING_YL_RE.search(token):
        return False
    from ..data.chain_names import get_chain_prefix
    for n in range(1, 31):
        stem = get_chain_prefix(n)
        # 'propan-2-yl' = prop + an + -2-yl
        # 'prop-1-en-2-yl' = prop + -1-en + -2-yl (BB 7104 'di(...)')
        # 'hex-5-en-2-yl' = hex + -5-en + -2-yl
        if re.fullmatch(
            rf'{stem}(?:an)?(?:-\d+(?:,\d+)*-(?:en|yn))*-\d+(?:,\d+)*-yl',
            token,
        ):
            return True
    return False


def composed_prefix_multiplier(token, count):
    """The multiplicative prefix for ``count`` copies of ``token``, or ``None``.

    (a) / (a). A SIMPLE substituent prefix -- including one that
    merely carries a locant -- takes 'di'/'tri': BB 25719
    '1,4-di(propan-2-yl)cyclohexane (PIN)' and BB 28170
    '2-[di(butan-2-yl)amino]butan-2-ol (PIN)'. A COMPOUND (substituted) prefix
    takes 'bis'/'tris': BB 41118 'bis(2-methylpropyl)', BB 40703
    'bis(chloromethyl)aminoxyl (PIN)' and BB 7104
    'bis(2-chloropropan-2-yl) (preferred prefix)'.

    ``None`` when the multiplicity has no tabulated prefix -- the caller must
    then fail closed rather than invent one.
    """
    from .naming_utils import COMPLEX_MULTIPLIERS
    if count < 2:
        return ''
    # Derived from the CITATION decision, not re-tested here, so the marks and
    # the multiplier can never disagree about whether a prefix is compound.
    marked = cite_organyl_in_composed_prefix(token)
    compound = marked is not None and marked != token
    if compound and _is_bare_locant_bearing_alkyl(token):
        compound = False
    table = COMPLEX_MULTIPLIERS if compound else SIMPLE_MULTIPLIERS
    return table.get(count)


def composed_alkoxy_prefix(token):
    """Turn an organyl ``token`` into its alkoxy prefix, or ``None``.

    The Blue Book tabulates this morphology verbatim (BB 27667-27691):

      * ``CH3-[CH2]3-O-`` **butoxy** -- the retained CONTRACTED prefixes
        (methoxy, ethoxy, propoxy, butoxy) are simple and cited bare;
      * ``(CH3)3C-O-`` ***tert*-butoxy** "(preferred prefix) (no substitution)",
        explicitly NOT 'tert-butyloxy' (BB 55662);
      * ``(CH3)2CH-O-`` **(propan-2-yl)oxy** and ``CH3-CH2-CH(CH3)-O-``
        **(butan-2-yl)oxy** -- a locant-bearing free valence keeps the alkyl
        name whole inside marks, suffix outside;
      * ``(CH3)2CH-CH2-O-`` **2-methylpropoxy** "(preferred prefix) (not
        isobutoxy)" -- a free valence at position 1 CONTRACTS even when the
        chain is branched, and is cited bare.

    ``None`` when the token is not an alkyl the table covers, so the caller
    fails closed instead of guessing a morphology.
    """
    if not token:
        return None
    if token == 'tert-butyl':
        return 'tert-butoxy'
    # 'phenoxy' is retained as the PREFERRED prefix, so C6H5-O- never spells out
    # as 'phenyloxy': BB 17796 "phenoxy (preferred prefix) (full substitution;
    # see " and BB 24567 "phenoxy (preferred prefix) (a retained
    # simple prefix derived from phenol; substitution allowed)". Only the BARE
    # ring contracts -- BB 24607 gives "([1,1'-biphenyl]-4-yl)oxy (preferred
    # prefix) (not 4-phenylphenoxy)", so a token that merely ENDS in 'phenyl'
    # must not be contracted here.
    if token == 'phenyl':
        return 'phenoxy'
    # An explicit free-valence locant -> cite the alkyl whole inside marks.
    if _LOCANT_BEARING_YL_RE.search(token):
        cited = cite_organyl_in_composed_prefix(token)
        return f"{cited}oxy" if cited else None
    if not token.endswith('yl'):
        return None
    for stem, contracted in _ALKOXY_CONTRACTED_STEMS.items():
        if token == stem:
            return contracted
        # The contraction is a straight-CHAIN morphology: it applies only when
        # 'propyl'/'butyl' is the parent chain stem, not when it is the tail of a
        # RING name. 'cyclopropyl'/'cyclobutyl' end in the stem but are rings
        # attached at a ring carbon -> concatenate ('cyclopropyloxy', like
        # 'cyclohexyloxy' the Blue Book; 'cyclopropoxy'/'cyclobutoxy' occur 0x in the
        # BB). A genuine substituent prefix before the stem ('2-methylpropyl',
        # 'cyclopropylmethyl') does NOT end in 'cyclo', so it still contracts.
        if token.endswith(stem) and not token[:-len(stem)].endswith('cyclo'):
            # '2-methylpropyl' -> '2-methyl' + 'propoxy' = '2-methylpropoxy'
            return token[:-len(stem)] + contracted
    # C5+ and cycloalkyls keep the whole name ('pentyloxy', 'cyclopropyloxy').
    return f"{token}oxy"


def alkoxy_prefix_from_substituent(token):
    """A '...yl' substituent NAME -> its R-oxy prefix.

    Thin wrapper over:func:`composed_alkoxy_prefix` that ALSO applies the
    decorated-phenyl -> phenoxy contraction the primitive deliberately declines.
    ``composed_alkoxy_prefix`` stays conservative on any '...phenyl' token
    (the biphenyl '4-phenylphenyl' shape, BB 24607 -> '([1,1'-biphenyl]-4-yl)oxy')
    and leaves the contraction to the caller. A DECORATED benzene is the retained,
    fully-substitutable 'phenoxy' / BB 17796 "phenoxy... full
    substitution"): '4-methylphenyl' -> '4-methylphenoxy'. A locant-bearing
    biphenyl reaches us as '[1,1'-biphenyl]-4-yl' (a '-N-yl' token, routed by
    composed_alkoxy_prefix's locant branch), never as '...phenyl', so contracting
    a bare '...phenyl' here is safe. Use this at every alkoxy emitter that hands
    in a general substituent name (F-spell-oxy).
    """
    if not token:
        return None
    if token.endswith('phenyl'):
        return token[:-2] + 'oxy'   # '4-methylphenyl' -> '4-methylphenoxy'
    return composed_alkoxy_prefix(token)


# (BB 27914) /: the free valence each NON-oxygen divalent
# chalcogen contributes to a composed prefix. Oxygen is absent on purpose --
# its morphology is the CONTRACTED one and is spelled by the BB-cited
# `composed_alkoxy_prefix` above, never by appending a bare 'oxy' here.
_CHALCOGEN_YL_SUFFIX = {'S': 'sulfanyl', 'Se': 'selanyl', 'Te': 'tellanyl'}


def composed_chalcogen_group_prefix(mol, frag_atoms, chalcogen_idx, boundary):
    """The PIN prefix token for a MONO-chalcogen-rooted group ``-X-R``, or None.

    ``chalcogen_idx`` is a divalent O/S/Se/Te whose single continuation inside
    ``frag_atoms`` (not crossing ``boundary``) is an ORGANYL group. The Blue
    Book spells both morphologies verbatim:

      * ``-O-CH3`` -> ``methoxy``, BB 27671)
      * ``-O-C(CH3)3`` -> ``tert-butoxy`` (BB 27679, "not tert-butyloxy")
      * ``-O-CH(CH3)2`` -> ``(propan-2-yl)oxy`` (BB 27683)
      * ``-S-CH3`` -> ``methylsulfanyl`` (BB 27651, BB 25021)
      * ``-Se-CH3`` -> ``methylselanyl``

    THIS IS THE PRIMITIVE THE GENERIC CASCADE CANNOT SUPPLY. Handed a chalcogen
    attachment,:func:`name_substituent` RE-ROOTS the fragment at a carbon and
    names the chalcogen as a hydroxy/sulfanyl SUBSTITUENT on that carbon, so
    ``tert-Bu-O-`` came back as ``2-hydroxy-2-methylpropyl`` -- a different
    CONSTITUTION, not merely a different spelling. Any caller holding a
    chalcogen attachment must come here rather than guess with the cascade.

    Scope is deliberately the MONO chalcogen. A second chalcogen beyond
    ``chalcogen_idx`` is the peroxy / disulfanyl class, which the
    cascade already names correctly and whole ('methylperoxy',
    'tert-butyldisulfanyl'); this returns None there so the caller keeps that
    working path.

    Fails closed (``None``) when the atom is not a divalent chalcogen, has other
    than exactly one in-fragment continuation, that continuation is not carbon,
    or the organyl half cannot be named as a single prefix token.
    """
    frag_set = set(frag_atoms)
    if chalcogen_idx not in frag_set:
        return None
    hub = mol.GetAtomWithIdx(chalcogen_idx)
    sym = hub.GetSymbol()
    if sym != 'O' and sym not in _CHALCOGEN_YL_SUFFIX:
        return None
    if not _is_divalent_chalcogen_atom(hub):
        return None

    _bound = set(boundary) | {chalcogen_idx}
    onward = [n.GetIdx() for n in hub.GetNeighbors()
              if n.GetIdx() in frag_set and n.GetIdx() not in _bound]
    if len(onward) != 1:
        return None  # terminal (-OH/-SH) or a branched hub -> fail closed
    r_start = onward[0]
    if mol.GetAtomWithIdx(r_start).GetSymbol() != 'C':
        return None  # di-chalcogen: the cascade owns that class

    # The organyl half, bounded so the walk can never re-enter the hub.
    r_atoms = set()
    stack = [r_start]
    while stack:
        idx = stack.pop()
        if idx in r_atoms or idx in _bound or idx not in frag_set:
            continue
        r_atoms.add(idx)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nbr.GetIdx()
            if nidx not in r_atoms and nidx not in _bound:
                stack.append(nidx)
    if not r_atoms:
        return None

    token = composed_prefix_organyl_name(mol, sorted(r_atoms), r_start)
    if not token:
        return None
    if sym == 'O':
        # The contracted R-O- morphology, spelled by the BB-cited primitive.
        return composed_alkoxy_prefix(token)
    cited = cite_organyl_in_composed_prefix(token)
    if not cited:
        return None
    return f"{cited}{_CHALCOGEN_YL_SUFFIX[sym]}"


# C4d: the sentinel separating "not the chalcogen-rooted acyl class" from "that
# class, and it must FAIL CLOSED". Returning None for both lets a caller's
# carbon-count fallback spell a carbamate as 'methanoylamino' -- a formyl C-H
# the molecule does not have.
ACYL_CHALCOGEN_UNNAMEABLE = object()


def chalcogen_rooted_acyl_amino_core(mol, carbonyl_c, n_idx, frag_atoms):
    """The prefix core for a CHALCOGEN-rooted acyl on nitrogen, ``R-X-CO-NH-``.

    Returns ``'(tert-butoxycarbonyl)amino'`` and friends WITHOUT the outer
    enclosing marks (each caller applies its own, since the two call sites wrap
    at different depths), ``None`` when the acyl is not chalcogen-rooted so the
    caller keeps its carbon path, or:data:`ACYL_CHALCOGEN_UNNAMEABLE` when it
    IS this class but cannot be spelled -- the caller must then fail closed.

    THE CLASS THE COUNTS CANNOT REACH. ``R-O-CO-`` is an ester of carbamic acid
    (Boc, Cbz, Fmoc, methoxycarbonyl,...), not an acyl of a carboxylic acid, so
    no carbon count describes it. Both count-based callers proved that: one
    stops at the heteroatom and counts only the carbonyl carbon, returning 1 for
    a true formyl ``H-CO-NH-`` and for ``(CH3)3C-O-CO-NH-`` alike; the other
    WALKS ACROSS the heteroatom and counts the organyl beyond it as if it were
    acyl carbon. The first spelled Boc 'methanoylamino', the second spelled
    ``CH3-S-CO-NH-`` 'ethanoylamino'.

    The Blue Book builds this prefix by CONCATENATION onto the
    chalcogen-group prefix, verbatim::

        -CO-O-CH2-C6H5 (benzyloxy)carbonyl (preferred prefix) [BB 18116]
        CH3-CO-S-CO- (acetylsulfanyl)carbonyl (preferred prefix) [BB 18128]

    and (BB 31698) names ``-CO-OR'`` 'alkoxycarbonyl'. Whether the
    chalcogen prefix is cited BARE or in marks is the /
     simple-vs-compound distinction -- the retained contractions are
    SIMPLE (BB 27667 "considered as simple prefixes"), so *tert*-butoxy
    concatenates bare, BB 54417 ``N2-(tert-butoxycarbonyl)-L-lysine``, while a
    concatenated ``benzyloxy`` is COMPOUND (BB 27633) and takes its marks,
    BB 54422 ``N5-acetyl-N2-[(benzyloxy)carbonyl]-L-glutamine`` (both.
    That decision is NOT re-spelled here: it is the ONE shared
    ``enclose_if_compound``, so these marks cannot drift from the rest of the
    system.

    ``amino`` is the morpheme (BB 26314); with the caller's marks the
    result is the Blue Book's own ``[(acyl)amino]acetic acid`` shape, BB 33213
    ``[(methanesulfinothioyl)amino]acetic acid (PIN)``.

    Pure: no mol mutation.
    """
    from .naming_utils import apply_enclosing_marks, enclose_if_compound

    frag_set = set(frag_atoms)
    if carbonyl_c not in frag_set or n_idx not in frag_set:
        return None
    c_at = mol.GetAtomWithIdx(carbonyl_c)

    # The carbonyl's own doubly-bonded O, inside the fragment.
    carbonyl_o = None
    for nb in c_at.GetNeighbors():
        if (nb.GetSymbol() == 'O' and nb.GetIdx() in frag_set
                and mol.GetBondBetweenAtoms(
                    carbonyl_c, nb.GetIdx()).GetBondTypeAsDouble() == 2.0):
            carbonyl_o = nb.GetIdx()
            break
    if carbonyl_o is None:
        return None

    # Exactly one heavy neighbour besides the amide N and that O, and it must
    # lie inside the fragment -- an acyl reaching outside is not describable here.
    rest = [nb.GetIdx() for nb in c_at.GetNeighbors()
            if nb.GetIdx() not in (n_idx, carbonyl_o) and nb.GetAtomicNum() > 1]
    if len(rest) != 1:
        return None
    x_idx = rest[0]
    if x_idx not in frag_set:
        return None
    if mol.GetAtomWithIdx(x_idx).GetSymbol() not in ('O', 'S', 'Se', 'Te'):
        return None  # a carbon acyl -- the caller's own paths own that class

    # From here the class is CLAIMED: every exit below fails closed.
    # The nitrogen carries the attachment, this acyl, and at most ONE further
    # organyl. That extra branch is cited alongside the acyl by
    # method (2) -- BB 33042 ``2-[methyl(propanoyl)amino]benzene-1-sulfonic
    # acid`` -- which is the only method available here, since a carbamate has
    # no amide name for method (1) to alter. Failing closed instead is NOT the
    # safe choice: the caller's fallback then drops the whole branch and still
    # emits, so an N-methyl Boc came back as bare '3-fluoropropanoic acid'.
    n_at = mol.GetAtomWithIdx(n_idx)
    if n_at.GetFormalCharge() != 0:
        return ACYL_CHALCOGEN_UNNAMEABLE
    n_heavy = [nb.GetIdx() for nb in n_at.GetNeighbors() if nb.GetAtomicNum() > 1]
    if carbonyl_c not in n_heavy or len(n_heavy) - 1 != len(
            [i for i in n_heavy if i in frag_set]):
        return ACYL_CHALCOGEN_UNNAMEABLE  # not exactly one attachment outside
    n_extra = [i for i in n_heavy if i in frag_set and i != carbonyl_c]
    if len(n_extra) > 1:
        return ACYL_CHALCOGEN_UNNAMEABLE

    # The organyl half, walked from the chalcogen and bounded by the carbonyl.
    organyl = set()
    stack = [x_idx]
    while stack:
        a = stack.pop()
        if a in organyl or a == carbonyl_c or a not in frag_set:
            continue
        organyl.add(a)
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            if nb.GetIdx() not in organyl and nb.GetIdx() != carbonyl_c:
                stack.append(nb.GetIdx())
    if n_idx in organyl:
        return ACYL_CHALCOGEN_UNNAMEABLE  # cyclic carbamate -- ring handler's

    # The N-substituent branch, walked from the nitrogen.
    extra = set()
    for e in n_extra:
        stack = [e]
        while stack:
            a = stack.pop()
            if a in extra or a == n_idx or a not in frag_set:
                continue
            extra.add(a)
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                if nb.GetIdx() not in extra and nb.GetIdx() != n_idx:
                    stack.append(nb.GetIdx())
    if extra & (organyl | {carbonyl_c, carbonyl_o}):
        return ACYL_CHALCOGEN_UNNAMEABLE  # fused back into the acyl -- cyclic

    # The prefix must claim EVERY atom of the fragment, or it would read as a
    # complete description of a branch it silently truncated.
    if organyl | extra | {carbonyl_c, carbonyl_o, n_idx} != frag_set:
        return ACYL_CHALCOGEN_UNNAMEABLE

    token = composed_chalcogen_group_prefix(mol, sorted(organyl), x_idx,
                                           {carbonyl_c})
    if not token:
        return ACYL_CHALCOGEN_UNNAMEABLE
    acyl_core = f"{enclose_if_compound(token)}carbonyl"

    if not n_extra:
        return f"{apply_enclosing_marks(acyl_core, -1)}amino"

    # An N-substituted carbamate: cite both N-substituents through the ONE
    # shared assembler, so their alphanumerical order and their
    # per-branch marks are the same decisions the rest of the system makes.
    branch = composed_prefix_organyl_name(mol, sorted(extra), n_extra[0])
    if not branch:
        return ACYL_CHALCOGEN_UNNAMEABLE
    from .composer import _assemble_decorated_amino_prefix
    core = _assemble_decorated_amino_prefix(
        [(branch, False), (acyl_core, True)], enclose=False)
    return core if core else ACYL_CHALCOGEN_UNNAMEABLE


def is_dichalcogen_bridge_attach(mol, attach_idx, frag_atoms_set,
                                 require_different: bool = False):
    """True when ``attach_idx`` is a divalent chalcogen bonded, inside the
    fragment, to a second divalent chalcogen -- the / bridge
    shapes ``-OO-``, ``-SS-``, ``-OS-``, ``-SO-``, ``-OSe-``...

    ``require_different=True`` narrows it to the MIXED bridge only (the two
    chalcogens are different elements), which is the shape the generic
    substituent tiers must never be allowed to mangle.

    One walker for both questions on purpose: the two predicates differ by a
    single element comparison, and spelling them separately is how this class
    regrew a site at a time before.
    """
    _CHALCOGENS = ('O', 'S', 'Se', 'Te')
    a = mol.GetAtomWithIdx(attach_idx)
    sym = a.GetSymbol()
    if sym not in _CHALCOGENS or not _is_divalent_chalcogen_atom(a):
        return False
    return any(
        n.GetIdx() in frag_atoms_set
        and n.GetSymbol() in _CHALCOGENS
        and (n.GetSymbol() != sym if require_different else True)
        and _is_divalent_chalcogen_atom(n)
        for n in a.GetNeighbors()
    )


def _name_substituent_cascade(mol, frag_atoms, attach_idx,
                              allow_mancude: bool = False):
    """Name any substituent fragment.

    Five-tier naming cascade:
      1. Retained substituent names (isopropyl, phenyl, etc.) -- IUPAC preferred
      2. Static fragment cache (FRAGMENT_NAME_CACHE) -- O(1) lookup
      3. Linear alkyl fast path (chain_prefixes table)
      4. Recursive compound naming (name_substituent_fragment)
      5. Descriptive fallback (guaranteed non-None)

    Args:
        mol: RDKit Mol of the full molecule.
        frag_atoms: Set/list of atom indices belonging to the substituent.
        attach_idx: Atom index WITHIN frag_atoms that bonds to the parent.
        allow_mancude: opt-in (complete/best-effort engine tier only).
            Threaded to the ring chokepoint so a multi-ring cage substituent the
            narrow PIN namers decline (tricyclo+/adamantane, mancude fused
            aromatics) is named via the universal von-Baeyer cage engine instead
            of failing closed. Default False -> PIN-default byte-identical.

    Returns:
        str: IUPAC prefix name. Under the PIN-default path
            (``allow_mancude=False``) always non-None, always non-empty --
            the five-tier cascade guarantees Tier 5's descriptive fallback as
            the terminal case. Under ``allow_mancude=True`` (Composer #1,
            Tier 4.5) may return ``None``: a clean abstention when the
            recursive decoration composer also declines and the fragment is
            ring-bearing with no honest systematic name available -- the
            'substituent' sentinel is de-masked to ``None`` rather than
            shipped. Simple non-ring fragments still resolve via the
            descriptive fallback even under ``allow_mancude=True``.

    References:
        IUPAC 2013 (detachable prefixes)
        a phase design: five-tier cascade with guaranteed fallback
    """
    import re as _re

    from .fragment_naming import FRAGMENT_NAME_CACHE
    from .substituent_naming import (
        ATTACH_LOCANT_UNKNOWN,
        _add_substituent_stereo,
        _check_retained_substituent,
        _is_linear_alkyl,
        parent_to_prefix,
    )

    frag_atoms_set = set(frag_atoms)

    # Edge case: empty fragment
    if not frag_atoms_set:
        return "substituent"

    # ---- (a phase -02): stereo-dropping tier double-apply guard ----
    # Tiers 0.5/1/1.5/1.6/2/3 build their prefix from a canonicalised fragment
    # (e.g. Tier-2's MolFragmentToSmiles strips @/@@ before the cache lookup),
    # so they SHORT-CIRCUIT Tier-4 — the only tier that natively reaches
    # _add_substituent_stereo. Without this guard a stereogenic substituent that
    # resolves via an early tier ships descriptor-less. _stereo_route routes a
    # tier return through _add_substituent_stereo IFF the fragment carries CIP
    # stereo AND the candidate prefix does not ALREADY carry a "(...)" stereo
    # block (the re.match double-apply guard — RESEARCH Q#2: a fragment-scope
    # check, NOT a needs_stereo_injection call).
    _STEREO_BLOCK_RE = _re.compile(r'^\(\d*[a-z]?[RSrsEZez](,\d*[a-z]?[RSrsEZez])*\)-')

    def _frag_has_cip_stereo() -> bool:
        for _i in frag_atoms_set:
            _a = mol.GetAtomWithIdx(_i)
            if _a.HasProp('_CIPCode'):
                return True
        for _b in bonds_of(mol):
            if (_b.GetBeginAtomIdx() in frag_atoms_set
                    and _b.GetEndAtomIdx() in frag_atoms_set
                    and _b.HasProp('_CIPCode')):
                return True
        return False

    def _stereo_route(prefix: str, located=None) -> str:
        # Route a stereo-dropping tier's return through the substituent stereo
        # emitter, unless the prefix already carries a leading "(...)" descriptor
        # (double-apply guard) or the fragment has no CIP stereo. `attach_idx` is
        # threaded so the emitter can derive the located descriptor (PIN name +
        # attachment locant) for an acyclic-alkyl substituent from STRUCTURE.
        #
        # `located` is the OPTIONAL `(pin_form, k, pos)` triple documented on
        # `_add_substituent_stereo`: a tier that ALREADY knows the numbering its
        # own name used passes it here instead of letting the emitter re-derive
        # one. Only Tier 1.95 does so today (its ring-on-chain producer owns a
        # CARRIER chain numbering the acyclic-only deriver cannot reach, so
        # without it every carrier-borne descriptor was dropped). `pin_form` is
        # the tier's own `prefix`, so the emitted text stays byte-identical.
        if not prefix or _STEREO_BLOCK_RE.match(prefix):
            return prefix
        if not _frag_has_cip_stereo():
            return prefix
        return _add_substituent_stereo(
            mol, list(frag_atoms_set), prefix, attach_idx=attach_idx,
            located=located,
        )

    # ---- Tier 0.5 (a phase): IUPAC / prefix-form check ----
    # PURE read-only check. Returns the IUPAC-canonical prefix form for any
    # fragment that ENTIRELY contains one of the 14 non-principal functional
    # groups (ester, ether, amide, sulfoxide, sulfone, thioether, nitrile,
    # carbamate, urea, isocyanate, isothiocyanate). Short-circuits Tier-1..5
    # for FG-bearing fragments, eliminating the 'methyl formatyl' /
    # 'hydroxymethyl' bug per RESEARCH root-cause fix.
    try:
        #: allow_mancude (complete/best-effort tier) also lifts
        # the S-attached sulfoxide/sulfone prefix-form guard so a ring-borne
        # -S(=O)(=O)-R / -S(=O)-R substituent is named (R)sulfonyl / (R)sulfinyl
        # instead of dropping the S and its =O. Gated so the PIN default path is
        # byte-identical (the pre-existing polyfunctional route still owns it).
        prefix_form = _check_substituent_prefix_form(
            mol, frag_atoms_set, attach_idx,
            allow_higher_sulfur=allow_mancude,
        )
        if prefix_form is not None:
            return _stereo_route(prefix_form)
    except Exception:
        # Defensive: any unexpected SMARTS / RDKit error falls through to Tier-1
        pass

    # ---- Tier 0.55 (a phase,: S-oxoacid acyl-oxy/-amino ----
    # An -O-S(oxoacid) or -NH-S(oxoacid) tail, where the parent carries a group
    # senior to the sulfur acid, is a substituent PREFIX (sulfooxy,
    # (chlorosulfonyl)oxy, sulfamoyloxy, (aminosulfinyl)oxy, (methoxysulfinyl)oxy,
    # (methoxysulfonyl)amino). The generic tiers name it by skeletal ('a')
    # replacement ('…-1,3-dioxa-2λ6-thiapropyl') — a valid, round-tripping, but
    # NON-PIN form (the Blue Book Blue Book). Route BOTH linker shapes to
    # the S-oxoacid namer; it fails closed on any S outside the neutral
    # mono/di-oxo acyl class, so plain thioethers, carbon-R sulfonyls and parent
    # sulfate/sulfamate esters are untouched.
    if attach_idx is not None:
        _att = mol.GetAtomWithIdx(attach_idx)
        if (_att.GetSymbol() == 'O' and _att.GetFormalCharge() == 0
                and _att.GetTotalNumHs() == 0 and not _att.IsInRing()):
            _parent = [n.GetIdx() for n in _att.GetNeighbors()
                       if n.GetIdx() not in frag_atoms_set]
            _inner = [n.GetIdx() for n in _att.GetNeighbors()
                      if n.GetIdx() in frag_atoms_set]
            if (len(_parent) == 1 and len(_inner) == 1
                    and mol.GetAtomWithIdx(_inner[0]).GetSymbol() == 'S'):
                from ..rules.sulfur_oxoacid import (
                    name_sulfur_oxoacid_oxy_substituent,
                )
                _so = name_sulfur_oxoacid_oxy_substituent(
                    mol, attach_idx, _parent[0])
                if _so is not None:
                    return _stereo_route(_so)
        elif _att.GetSymbol() == 'S':
            from ..rules.sulfur_oxoacid import (
                name_sulfur_oxoacid_acyl_for_amino,
            )
            _sa = name_sulfur_oxoacid_acyl_for_amino(
                mol, attach_idx, frag_atoms_set)
            if _sa is not None:
                return _stereo_route(_sa)

    # ---- Tier 0.6 / BB 1710): thiocyanato pseudohalide ----
    # The terminal thiocyanate group -S-C#N is ALWAYS cited as the substituent
    # prefix 'thiocyanato' in PINs ("added to the list of characteristic groups
    # that are always cited as prefixes... in preferred IUPAC names"). Tier 0.5
    # already emits isocyanato / isothiocyanato / isocyano for the three
    # N-anchored pseudohalides, but -S-C#N falls through it and the generic
    # tiers name it 'cyanosulfanyl' (a valid but non-PIN cyano+sulfanyl
    # concatenation). This mirrors the identical detector on the sibling
    # substituent namer (substituent_naming.py Step 1e). Fires only when the
    # fragment is EXACTLY {S, C, N} anchored at the divalent S; any extra
    # decoration falls through -> fail closed.
    if attach_idx is not None and len(frag_atoms_set) == 3:
        _tc_pat = _compiled_smarts('[SX2][CX2]#[NX1]')
        if _tc_pat is not None:
            for _m in mol.GetSubstructMatches(_tc_pat):
                if set(_m) == frag_atoms_set and _m[0] == attach_idx:
                    from ..rules.seniority import get_prefix as _pseudo_get_prefix
                    _pfx = _pseudo_get_prefix('thiocyanate')
                    if _pfx:
                        return _stereo_route(_pfx)

    # ---- Tier 1: Retained substituent names ----
    # Checked first per IUPAC: retained names (phenyl, isopropyl, etc.)
    # are the preferred forms and must take priority over cache-derived
    # parent-to-prefix conversions (e.g., "phenyl" not "benzenyl").
    try:
        retained = _check_retained_substituent(
            mol, list(frag_atoms_set), attach_idx
        )
        if retained:
            return _stereo_route(retained)
    except Exception:
        pass

    # ---- Tier 1.5 (a phase L1): monocyclic heteroaryl PIN locant ----
    # A heteroaryl ring substituent (pyridine, imidazole, furan,...) takes
    # free-valence numbering — pyridin-3-yl, 1H-imidazol-5-yl — instead of the
    # locant-less parent_to_prefix form (pyridinyl / imidazolyl) that the cache
    # (Tier 2) or recursive namer (Tier 4) would otherwise emit. Guarded:
    # returns None (so we fall through unchanged) unless the locant is provably
    # PIN-correct (IUPAC.
    if attach_idx is not None:
        try:
            from ..rules.ring_substituents import pin_heteroaryl_substituent_name
            for ring in mol.GetRingInfo().AtomRings():
                if attach_idx in ring and set(ring) <= frag_atoms_set:
                    pin = pin_heteroaryl_substituent_name(mol, ring, attach_idx)
                    if pin is not None:
                        return _stereo_route(pin)
                    break
        except Exception:
            pass

    # ---- Tier 1.6 (.2): decorated monocyclic ring substituent ----
    # A ring fragment carrying its own substituents must keep them with
    # attachment-correct numbering ('2-oxocyclohexyl'), instead of the
    # cache/recursive parent_to_prefix form that keeps the PARENT numbering
    # ('1-oxocyclohexyl' — structurally impossible) or drops the group.
    # Guarded: returns None (fall through unchanged) unless the ring is a
    # supported simple monocycle AND the decorated name covers EXACTLY the
    # fragment atoms numbering; see rules/ring_substituents.py).
    if attach_idx is not None:
        try:
            from ..rules.ring_substituents import decorated_ring_substituent_name
            for ring in mol.GetRingInfo().AtomRings():
                if attach_idx in ring and set(ring) <= frag_atoms_set:
                    dec = decorated_ring_substituent_name(
                        mol, ring, attach_idx, expected_atoms=frag_atoms_set)
                    if dec is not None:
                        return _stereo_route(dec)
                    break
        except Exception:
            pass

    # ---- Tier 1.7 (DD2 Fix B, Phase D): peroxy / disulfanyl substituent ----
    # A -O-O-R (peroxy) / -S-S-R (disulfanyl) substituent: the attach atom is a
    # divalent chalcogen bonded to a second like chalcogen inside the fragment.
    # Named (R)peroxy / (R)disulfanyl per (1). Placed before the cache /
    # recursive tiers, which otherwise mangle the -O-O-/-S-S- into a bogus
    # 'peroxyl'/'dithioperoxyl' fragment. Reachable from EVERY caller (the chain
    # GENERAL path, the benzene/ring-substituent path), so the substitutive
    # peroxide/disulfide is named identically wherever it appears.
    if attach_idx is not None and attach_idx in frag_atoms_set:
        _attach_atom = mol.GetAtomWithIdx(attach_idx)
        _sym = _attach_atom.GetSymbol()
        if _sym in ('O', 'S') and any(
            n.GetSymbol() == _sym and n.GetIdx() in frag_atoms_set
            for n in _attach_atom.GetNeighbors()
        ):
            _parent_atoms = set(range(mol.GetNumAtoms())) - frag_atoms_set
            _frag_list = list(frag_atoms_set)
            _chal = (
                _name_peroxy_branch(mol, _frag_list, attach_idx, _parent_atoms)
                if _sym == 'O'
                else _name_disulfanyl_branch(mol, _frag_list, attach_idx, _parent_atoms)
            )
            if _chal:
                return _stereo_route(_chal)
        # Wave2: MIXED divalent-chalcogen bridge (-O-S-R etc.).
        # Positive name for the O-attached sulfanyl case ((methylsulfanyl)oxy);
        # every other mixed shape is a TERMINAL decline — the generic tiers
        # below mangle the bridge into a wrong-constitution fragment, so
        # falling through is never allowed for this shape.
        if _is_mixed_chalcogen_bridge_attach(mol, attach_idx, frag_atoms_set):
            _parent_atoms = set(range(mol.GetNumAtoms())) - frag_atoms_set
            _mixed = _name_mixed_chalcogen_branch(
                mol, list(frag_atoms_set), attach_idx, _parent_atoms)
            return _stereo_route(_mixed) if _mixed else None

    # ---- Tier 1.75: O-glycosyl (glycosyloxy) substituent ----
    # A sugar O-linked to a non-sugar aglycone is a COMPOUND substituent prefix
    # 'glycosyl' + 'oxy' (BB:53915), cited at the aglycone's attachment locant
    # -- BB's worked example is 1-[4-(beta-D-glucopyranosyloxy)phenyl]ethan-1-one
    # (:53927). Naming it HERE is what single-counts the glycosidic oxygen: the
    # ordinary parent+prefix machinery then assigns the locant and cites only the
    # REMAINING hydroxy groups. Without this tier the fragment reaches Tier 5 and
    # collapses to the 'substituent' placeholder, the assembled name is rejected
    # as garbled, and the pre-general decomposition fallback glues the sugar onto
    # the aglycone's ALCOHOL name -- expressing the glycosidic O twice and citing
    # no locant. Placed before the cache / recursive tiers, which name the
    # fragment as a free molecule (a '...pyranose' PARENT, not a prefix).
    # Disjoint from Tier 1.7 by construction: a glycosidic O's in-fragment
    # neighbour is a carbon, so neither the peroxy (O-O) nor the mixed-chalcogen
    # predicate can match. Fails closed (falls through) on every shape whose PIN
    # morphology is a different construction -- see glycosyl_substituent_prefix.
    if attach_idx is not None and attach_idx in frag_atoms_set:
        try:
            from ..data.sugar_names import glycosyl_substituent_prefix
            _glyco = glycosyl_substituent_prefix(
                mol, frag_atoms_set, attach_idx)
            if _glyco:
                # Returned WITHOUT _stereo_route on purpose. The carbohydrate
                # descriptors the prefix already carries (the anomeric 'alpha'/
                # 'beta' plus the 'D'/'L' configurational prefix, name
                # every stereocentre of the glycosyl group. _stereo_route's
                # double-apply guard only recognises a LEADING '(...)' CIP block,
                # so it would not see them and would prepend a second, redundant
                # R/S block -- '(2R,3R,4S,5S,6R)-beta-D-glucopyranosyloxy'.
                return _glyco
        except Exception:
            pass

    # ---- Tier 1.8 (DD5 /): located acyclic alkyl ----
    # A BRANCHED or INTERNALLY-attached acyclic all-carbon saturated alkyl
    # substituent is named by its OWN principal chain numbered from the free
    # valence (hexan-2-yl, pentan-3-yl, 3-methylbutyl) per /. This
    # MUST precede the fragment cache (Tier 2) and the linear fast path (Tier 3),
    # both of which name the fragment as a FREE molecule and lose the attachment
    # (-> 'hexyl', 'pentyl', '2-methylbutyl' — a wrong locant or constitution).
    # Scoped to the cases where the located form DIFFERS from the plain alkyl
    # (internal attachment OR a branch): a TERMINAL unbranched chain keeps the
    # fast path byte-identical. Returns None for rings / heteroatoms / unsaturated
    # -> falls through unchanged.
    if attach_idx is not None and attach_idx in frag_atoms_set:
        try:
            from .substituent_naming import (
                _attach_is_chain_terminus,
                _is_linear_alkyl,
                _located_acyclic_alkyl_name,
            )
            _frag_list = list(frag_atoms_set)
            _terminal_linear = (
                _is_linear_alkyl(mol, _frag_list)
                and _attach_is_chain_terminus(mol, _frag_list, attach_idx)
            )
            if not _terminal_linear:
                _located = _located_acyclic_alkyl_name(mol, _frag_list, attach_idx)
                if _located is not None:
                    return _stereo_route(_located[0])
        except Exception:
            pass

    # ---- Tier 1.9 (C- / V-3): ether-substituted carbon chain ----
    # A saturated all-carbon chain bearing ether -O-R substituent(s), numbered
    # from the free valence, named (R-oxy)alkyl per (phenoxymethyl,
    # 2-phenoxyethyl, methoxymethyl). MUST precede the cache (Tier 2): the cache
    # maps the capped fragment SMILES to a whole-molecule retained name
    # (COc1ccccc1 -> 'anisole') which parent_to_prefix then mangles to 'anisolyl'
    # — a DIFFERENT constitution (free valence on the ring). Reachable from every
    # caller so the ether substituent is named identically wherever it appears.
    # Returns None (fall through) for anything not this narrow class (fail-closed).
    if attach_idx is not None and attach_idx in frag_atoms_set:
        try:
            from .substituent_naming import _name_ether_substituted_chain
            _ether = _name_ether_substituted_chain(
                mol, list(frag_atoms_set), attach_idx, set()
            )
            if _ether:
                return _stereo_route(_ether)
        except Exception:
            pass

    # ---- Tier 1.92 (a phase,: Group-14 silyl/germyl substituent --
    # A monovalent Si/Ge substituent is named (prefixes)silyl / (prefixes)germyl,
    # with the substituents on the Si/Ge centre cited as prefixes. MUST precede the
    # Tier-2 cache and Tier-4 recursive namer, which drop a bare -SiH3 ('substituent')
    # or keep an -OH as a parent-hydride -ol suffix (-Si(OH)3 -> 'silanetriolyl').
    # Gated on the attach atom being Si/Ge (rare -> contained blast radius);
    # fail-closed (None -> fall through unchanged) for every other case.
    if attach_idx is not None and attach_idx in frag_atoms_set:
        if mol.GetAtomWithIdx(attach_idx).GetSymbol() in ('Si', 'Ge'):
            try:
                from .substituent_naming import _name_group14_substituent
                _g14 = _name_group14_substituent(
                    mol, list(frag_atoms_set), attach_idx
                )
                if _g14:
                    return _stereo_route(_g14)
            except Exception:
                pass

    # ---- Tier 1.93 (W2F-P7,: phosphanyl (P-rooted) substituent ----
    # A phosphorus-rooted substituent (-PH2 -> phosphanyl, -PR2 -> dialkyl/
    # diarylphosphanyl) is cited via rules/phosphorus.name_phosphanyl_substituent.
    # MUST precede the Tier-5 descriptive fallback, which returns the
    # 'substituent' sentinel for a lone P — dropping it and failing the molecule
    # closed ('OC(=O)CCP' -> propanoic acid -> rejects -> 'unknown').
    # The helper fires ONLY for a clean neutral organyl/hydride P (excludes a
    # phosphoryl/phosphonic P=O, named by the oxoacid subsystem) -> fail-closed
    # (falls through to 'substituent') on any decline. The λ5 branch is Task 3.
    if attach_idx is not None and attach_idx in frag_atoms_set:
        if mol.GetAtomWithIdx(attach_idx).GetSymbol() == 'P':
            # substituent_recursion_depth_exceeded /: a
            # CARBON-FREE P-oxo fragment
            # -- bare -P(=O)(OH)2 (phosphono) / -P(=O)(O-)2 (phosphonato). The
            # phosphanyl helper below EXCLUDES a phosphonic P=O (it names only a
            # clean neutral organyl/hydride P), so this shape used to fall through
            # to the 'substituent' sentinel and be DROPPED -- the same carbon-free
            # reject as name_substituent_fragment substituent_recursion_depth_exceeded (the whole-molecule
            # namer calls 'O=P(O)O' inorganic). This restores the retained prefix
            # the FG-on-parent-chain path already emits; the top-level /
            # OPSIN gate voids any non-RT composed name (0-wrong). Fail-closed on
            # any other P-oxo shape (see carbon_free_phospho_prefix).
            try:
                from ..rules.phosphorus import carbon_free_phospho_prefix
                _pp = carbon_free_phospho_prefix(
                    mol, list(frag_atoms_set), attach_idx
                )
                if _pp:
                    return _stereo_route(_pp)
            except Exception:
                pass
            try:
                from ..rules.phosphorus import name_phosphanyl_substituent
                _ph = name_phosphanyl_substituent(
                    mol, list(frag_atoms_set), attach_idx
                )
                if _ph:
                    return _stereo_route(_ph)
            except Exception:
                pass

    # ---- Tier 1.93b (W3-P10, /: arsanyl (As-rooted) ----
    # An arsenic-rooted substituent (-As(OH)2 -> dihydroxyarsanyl) cited as a
    # prefix under a senior organic group (-COOH). MUST precede the Tier-5
    # descriptive fallback, which returns the 'inorganic compound (not supported)'
    # sentinel for a lone As and fails the whole molecule closed. Fail-closed
    # (falls through) for any non-As / decorated-As shape.
    if attach_idx is not None and attach_idx in frag_atoms_set:
        if mol.GetAtomWithIdx(attach_idx).GetSymbol() in ('As', 'Sb', 'Bi'):
            try:
                from ..rules.mononuclear_hydrides import name_arsanyl_substituent
                _as = name_arsanyl_substituent(
                    mol, list(frag_atoms_set), attach_idx
                )
                if _as:
                    return _stereo_route(_as)
            except Exception:
                pass

    # ---- Wave2 T3c: aryl-vinyl / styryl (SUBST-01 two-namer rule) ----
    # Mirror name_substituent_fragment's aryl-vinyl handler here so the benzene
    # generic-C fallback (benzene.py, which calls name_substituent and rejects the
    # 'substituent' sentinel) also emits (E)-2-phenylethenyl. MUST precede the
    # ring chokepoint below (which declines an acyclic-attached aryl-vinyl arm).
    if attach_idx is not None and attach_idx in frag_atoms_set:
        try:
            from .substituent_naming import _name_aryl_vinyl_substituent
            _av = _name_aryl_vinyl_substituent(
                mol, list(frag_atoms_set), attach_idx, set())
            if _av is not None:
                return _stereo_route(_av)
        except Exception:
            pass

    # ---- Tier 1.95 (a phase SUBST-01): ring-system substituent chokepoint ----
    # A ring-bearing fragment is named by the trustworthy ring engine
    # (get_ring_substituent_name + _compound_ring_on_chain_substituent) BEFORE the
    # Tier-2 cache, which would otherwise (a) drop ene/yne locants
    # ('cyclohexenyl' for cyclohex-1-en-1-yl via parent_to_prefix), or (b) name a
    # ring-on-chain as a different molecule ('methylcyclohexyl' for
    # cyclohexylmethyl). allow_enumerator_fallback=False makes the chokepoint
    # return None on a decline instead of re-entering this cascade (recursion
    # guard); we then fall through to the existing tiers unchanged.
    if attach_idx is not None and attach_idx in frag_atoms_set:
        try:
            _ri = mol.GetRingInfo()
            if any(_ri.NumAtomRings(a) > 0 for a in frag_atoms_set):
                from ..rules.ring_substituents import name_ring_system_substituent
                # `_carrier_pos` is filled by the ring-on-chain producer with the
                # CARRIER chain numbering its own name cites (free valence = 1,
                #. A carrier atom can itself be a stereocentre -- e.g. the
                # `-CH(CH3)-` bridging a lactone ring to a pentacyclic parent in
                # CHEBI:2364 -- and the acyclic-only deriver inside
                # `_add_substituent_stereo` DECLINES on any ring-bearing fragment,
                # so that descriptor used to be silently dropped (a stereo-
                # incomplete, i.e. WRONG-diastereomer, prefix). Handing the
                # producer's own map over as `located` cites it at the locant the
                # name really used, never a re-derived one. Empty map (every other
                # producer, and the mononuclear-carrier branches whose descriptor
                # is unlocanted) => byte-identical to before.
                _carrier_pos: dict = {}
                _ring_nm = name_ring_system_substituent(
                    mol, sorted(frag_atoms_set), attach_idx,
                    allow_enumerator_fallback=False,
                    allow_mancude=allow_mancude,
                    pos_out=_carrier_pos,
                )
                if _ring_nm:
                    # Fail closed on a PARTIAL expression: the carrier map can
                    # only reach the carrier's own centres, so hand it over only
                    # when doing so spells EVERY defined stereo element of the
                    # fragment (the rest already inside the nested ring-yl
                    # prefix). Otherwise the pre-existing descriptor-less name
                    # stands -- e.g. '1-(bicyclo[2.2.1]heptan-2-yl)ethyl', whose
                    # 3 ring centres no producer here can cite: claiming only
                    # the carrier's would assert one configuration and leave
                    # three silent (, missing beats wrong).
                    from .substituent_naming import located_map_completes_substituent_stereo
                    _use_pos = bool(_carrier_pos) and \
                        located_map_completes_substituent_stereo(
                            mol, frag_atoms_set, _ring_nm, _carrier_pos)
                    return _stereo_route(
                        _ring_nm,
                        located=((_ring_nm, 1, _carrier_pos)
                                 if _use_pos else None),
                    )
        except Exception:
            pass

    # ---- Tier 1.96 (SL): acyclic substituent with detachable prefixes ----
    # A saturated acyclic carbon chain bearing >=1 simple detachable prefixes
    # (carboxy/amino/hydroxy/oxo/halogen) is named from STRUCTURE, numbered from
    # the free valence. MUST precede the Tier-2 cache, which maps the H-capped
    # fragment to a whole-molecule retained name (serine-O -CH2CH(NH2)COOH caps
    # to 'alanine' -> 'alaninyl'; -CH2COOH caps to 'acetic acid' -> 'acetyl', a
    # different molecule), and the Tier-4 recursive path, which lets
    # parent_to_prefix DROP secondary prefixes ('(R)-2-carboxyethyl', amino lost)
    # or inherit the parent's lowest-locant numbering (-CH2CH2CH2OH ->
    # '1-hydroxypropyl', wrong end). Fail-closed (None -> fall through) for
    # rings / branched / unsaturated / amides / esters / ethers / bare-acyl.
    if attach_idx is not None and attach_idx in frag_atoms_set:
        try:
            from .substituent_naming import _name_polyfunctional_acyclic_substituent
            _poly = _name_polyfunctional_acyclic_substituent(
                mol, list(frag_atoms_set), attach_idx, set()
            )
            if _poly:
                return _stereo_route(_poly)
        except Exception:
            pass

    # W2E-P1FG Task 11: a carbon chain terminated by a
    # -NH-C(=O)-NH2 urea unit -> '{loc}-(carbamoylamino){chain}yl' ('not
    # ureido'). MUST precede the Tier-2 cache / Tier-4 recursive path, which
    # name the H-capped fragment as 'propylurea' -> 'propylureayl'.
    if attach_idx is not None and attach_idx in frag_atoms_set:
        try:
            from .substituent_naming import _name_carbamoylamino_chain_substituent
            _cba = _name_carbamoylamino_chain_substituent(
                mol, list(frag_atoms_set), attach_idx
            )
            if _cba:
                return _stereo_route(_cba)
        except Exception:
            pass

    # ---- Tier 1.97 (Composer1,: chalcogen-rooted sulfanyl ----
    # A monovalent substituent attached VIA a divalent sulfur (-S-R) is the
    # (R)sulfanyl prefix. name_substituent_fragment's chalcogen-ether handler
    # (Step 1b) covers -Se-R/-Te-R but OMITS S, so an S-rooted -S-R reaching
    # the Tier-4 recursive path is mis-named (a bare -S-CH3 becomes the garbage
    # 'hydroxymethanethiyl' — a spaceless token that would slip past the caller
    # guard). This is the substituent-decoration analog of the ring/chain
    # (R)sulfanyl the parent-anchored paths already emit; the recursive
    # composer (Tier 5) re-enters name_substituent for every decoration, so a
    # -S-R decoration must resolve here. GATED on allow_mancude (complete/
    # best-effort tier) -> PIN default byte-identical; fail-closed (falls
    # through) for any non-monovalent / non-carbon-flanked / higher-valence S.
    if (allow_mancude and attach_idx is not None
            and attach_idx in frag_atoms_set):
        _sa = mol.GetAtomWithIdx(attach_idx)
        if (_sa.GetSymbol() == 'S' and _sa.GetFormalCharge() == 0
                and _sa.GetTotalNumHs() == 0
                and _sa.GetNumRadicalElectrons() == 0
                and all(b.GetBondTypeAsDouble() == 1.0 for b in _sa.GetBonds())):
            _s_in = [n.GetIdx() for n in _sa.GetNeighbors()
                     if n.GetIdx() in frag_atoms_set and n.GetAtomicNum() > 1]
            _s_ext = [n.GetIdx() for n in _sa.GetNeighbors()
                      if n.GetIdx() not in frag_atoms_set]
            # Monovalent -S-R: exactly one heavy (carbon) neighbour inside the
            # fragment (the R group) and exactly one bond leaving to the parent
            # (also a carbon, so get_sulfanyl_prefix's two-carbon SMARTS holds).
            if (len(_s_in) == 1 and len(_s_ext) == 1
                    and mol.GetAtomWithIdx(_s_in[0]).GetSymbol() == 'C'
                    and mol.GetAtomWithIdx(_s_ext[0]).GetSymbol() == 'C'):
                try:
                    from .substituent_prefix_forms import get_sulfanyl_prefix
                    _parent_side = sorted(
                        set(range(mol.GetNumAtoms())) - frag_atoms_set)
                    _sf = get_sulfanyl_prefix(
                        mol, (attach_idx, _s_in[0], _s_ext[0]),
                        principal_chain=_parent_side, suffix='sulfanyl')
                    if _sf and _sf != 'substituent' and ' ' not in _sf:
                        return _stereo_route(_sf)
                except Exception:
                    pass
            # Monovalent -S-X where the sole in-fragment continuation is a
            # HALOGEN: the sulfenyl-halide substituent, spelled by substituting
            # the sulfanyl (-SH, H with the halogen prefix -> e.g.
            # -S-Cl 'chlorosulfanyl'. get_sulfanyl_prefix's two-carbon SMARTS
            # cannot see this shape, so build it directly. OPSIN round-trips
            # '(chlorosulfanyl)cyclohexane' / '1-(chlorosulfanyl)decahydro-
            # naphthalene'; guards it. Halogen terminal (degree 1) only.
            elif (len(_s_in) == 1 and len(_s_ext) == 1
                    and mol.GetAtomWithIdx(_s_ext[0]).GetSymbol() == 'C'):
                _x = mol.GetAtomWithIdx(_s_in[0])
                if (_x.GetSymbol() in _HALOGEN_MAP and _x.GetDegree() == 1
                        and _x.GetFormalCharge() == 0):
                    return _stereo_route(
                        f"{_HALOGEN_MAP[_x.GetSymbol()]}sulfanyl")

    # ---- Tier 1.85 (Wave F): SULFINYL / SULFONYL-rooted substituent -----
    # A fragment rooted at a sulfinyl ``-S(=O)-`` or sulfonyl ``-S(=O)(=O)-``
    # sulfur bridging the parent to exactly ONE in-fragment carbon subtree R is
    # the substitutive prefix ``{R}sulfinyl`` / ``{R}sulfonyl`` -- NOT
    # a skeletal-replacement chain. Without this the sulfinyl-S attach reaches
    # the last-resort ``terminal_fragment`` replacement generator, which spells
    # the whole -S(=O)-CH2-Ar half as ``2-[...]-1-oxo-1-thiaethyl`` (an
    # 'a'-replacement name that round-trips but is NOT the retained/substitutive
    # PIN spelling): the pantoprazole class. PREFERENCE ORDER: the acid-stem PIN
    # form (``methanesulfinyl``, ``get_sulfinyl_prefix``) FIRST -- so a simple
    # ``-S(=O)CH3`` that ever reaches here keeps its PIN spelling and never
    # regresses to the non-PIN ``methylsulfinyl`` -- then the additive
    # ``[Rsub]sulfinyl`` form (R named by re-entering ``name_substituent`` on
    # the strictly-smaller carbon subtree) when the acid-stem producer declines
    # a compound R. GATED on ``allow_mancude`` (complete/best-effort tier) ->
    # PIN default byte-identical; fail-closed (falls through to the replacement
    # generator) for any non-sulfinyl/sulfonyl shape. backstops.
    if (allow_mancude and attach_idx is not None
            and attach_idx in frag_atoms_set):
        _oa = mol.GetAtomWithIdx(attach_idx)
        if (_oa.GetSymbol() == 'S' and _oa.GetFormalCharge() == 0
                and _oa.GetTotalNumHs() == 0
                and _oa.GetNumRadicalElectrons() == 0):
            _dbl_o = [n for n in _oa.GetNeighbors()
                      if n.GetSymbol() == 'O' and n.GetDegree() == 1
                      and n.GetIdx() in frag_atoms_set
                      and mol.GetBondBetweenAtoms(attach_idx, n.GetIdx())
                      .GetBondTypeAsDouble() == 2.0]
            _c_in = [n.GetIdx() for n in _oa.GetNeighbors()
                     if n.GetIdx() in frag_atoms_set and n.GetAtomicNum() == 6]
            _s_ext = [n.GetIdx() for n in _oa.GetNeighbors()
                      if n.GetIdx() not in frag_atoms_set]
            _kind = {1: 'sulfinyl', 2: 'sulfonyl'}.get(len(_dbl_o))
            # Exactly: n double-bond oxygens (both in-fragment) + one in-fragment
            # carbon subtree R + one bond leaving to a parent atom, and NO other
            # neighbour (so every S neighbour is accounted -> no silent atom drop).
            #
            # CQ1 Task B (B1, best-effort): the parent-side atom may be C (a
            # sulfone) OR N (a ring-N sulfonamide, e.g. the N-sulfonyl of a
            # diaza-spiro/-cycloalkane). The substituent prefix `{R}sulfonyl` /
            # `{R}sulfinyl` is the SAME regardless of the atom it attaches
            # TO -- that atom is on the PARENT, not part of the substituent. Before
            # this the N-parent case (`CCS(=O)(=O)N1CC2(CCNC2)C1`) failed the guard,
            # fell to the skeletal-replacement generator, and emitted a
            # constitution-WRONG `1-(1-oxo-2-oxa-1λ6-thiaeth-1-en-1-yl)ethyl` token
            # (an extra in-chain oxa + one dropped =O) that the whole-graph RT gate
            # then voided -> abstain. Allowing N converts the ring-N-sulfonyl class
            # to `{R}sulfonyl`, which OPSIN round-trips. Still best-effort-gated
            # (`allow_mancude`) so PIN is byte-identical, and /RT backstops
            # any shape whose additive R does not round-trip. O/other parent-side
            # atoms stay fail-closed (sulfonate-ester ambiguity), unchanged.
            if (_kind is not None and len(_c_in) == 1 and len(_s_ext) == 1
                    and mol.GetAtomWithIdx(_s_ext[0]).GetSymbol() in ('C', 'N')
                    and _oa.GetDegree() == len(_dbl_o) + 2):
                _sulf = None
                try:
                    from .substituent_prefix_forms import get_sulfinyl_prefix, get_sulfonyl_prefix
                    _parent_side = sorted(
                        set(range(mol.GetNumAtoms())) - frag_atoms_set)
                    _getter = (get_sulfinyl_prefix if _kind == 'sulfinyl'
                               else get_sulfonyl_prefix)
                    _acid = _getter(
                        mol, (attach_idx, _dbl_o[0].GetIdx(), _c_in[0],
                              _s_ext[0]), principal_chain=_parent_side)
                    if _acid and _acid.endswith(_kind) and ' ' not in _acid:
                        _sulf = _acid
                except Exception:
                    _sulf = None
                if _sulf is None:
                    # Additive [Rsub]{kind}: R is the in-fragment carbon subtree
                    # (fragment minus S and its =O oxygens), named substitutively.
                    _r_atoms = (frag_atoms_set - {attach_idx}
                                - {o.GetIdx() for o in _dbl_o})
                    if _r_atoms and len(_r_atoms) < len(frag_atoms_set):
                        _rname = name_substituent(
                            mol, sorted(_r_atoms), _c_in[0], allow_mancude=True)
                        if (_rname and _rname != 'substituent'
                                and ' ' not in _rname
                                and 'unknown' not in _rname.lower()):
                            from .naming_utils import enclose_if_compound
                            _sulf = enclose_if_compound(_rname) + _kind
                if _sulf and _sulf != 'substituent' and ' ' not in _sulf:
                    return _stereo_route(_sulf)

    # ---- Tier 1.9: the intact -C(=O)OH group is `carboxy`, never `formyl` ----
    #. A graph-shape guard, and it has to be here rather than in
    # `parent_to_prefix`, because that function receives only a NAME.
    #
    # Without it, Tier 4 names the fragment recursively as a whole compound --
    # ``O=CO`` names as *formic acid* -- and `parent_to_prefix` then converts that
    # through ``_ACID_TO_ACYL_PREFIX = {'formic acid': 'formyl',...}``
    # (`substituent_naming.py:3460`). That mapping is right for an ACYL group and
    # wrong for an intact acid, and the reasoning is decisive rather than stylistic:
    # `formyl` is ``HC(=O)-``, whose fragment is two atoms and names as
    # *formaldehyde*. If the recursive namer said *formic acid*, the fragment still
    # carries its hydroxyl, so `carboxy` is right and `formyl` cannot be.
    #
    # Measured cost of not having it: composing von Baeyer parents with these
    # prefixes emitted ``5,7-diformyl-...`` for an input whose only such group is
    # ``C(=O)O`` -- a name two oxygens short, i.e. a different molecule that
    # then has to suppress. One confirmed cause of 34/71 wrong composed rows
    # (`benchmarks/assembly_yield_dev500.json`).
    #
    # `rules/ring_assemblies.py:1526` already does exactly this for the ring-assembly
    # path and is the model; this is the same predicate reused on the shared
    # substituent path rather than a second implementation of it.
    #
    # `parent_to_prefix` is deliberately left alone: `rules/acid_halides.py:16`
    # documents ``formic acid -> formyl`` as correct for acid halides,
    # `decomposition/fragment_assembly.py:136` holds its own copy, and
    # `tests/unit/rules/test_v29_phase7_prefix_vocabulary.py:51` asserts its current
    # return value.
    # ⚠ TWO attachment conventions circulate and the guard must survive both.
    # `_is_carboxyl_substituent` wants the FRAGMENT-side carbon; several callers pass
    # the PARENT-side atom instead (`SubstituentInfo.attach_mol_idx`), which
    # `rules/ring_substituents.py:1869-1879` normalizes with the same three lines.
    # My first version of this guard used the raw value, so it fired in a unit test
    # that happened to pass a fragment-side index and NEVER fired in production --
    # green for the wrong reason, and the corpus measurement was unchanged, which is
    # what exposed it.
    try:
        from ..rules.ring_assemblies import _is_carboxyl_substituent
        _cx_attach = attach_idx
        if _cx_attach not in frag_atoms_set:
            _cx_attach = next(
                (n.GetIdx()
                 for n in mol.GetAtomWithIdx(_cx_attach).GetNeighbors()
                 if n.GetIdx() in frag_atoms_set),
                None,
            )
        if (_cx_attach is not None
                and _is_carboxyl_substituent(
                    mol, list(frag_atoms_set), _cx_attach)):
            return _stereo_route('carboxy')
    except Exception:  # pragma: no cover - a guard must never break naming
        pass

    # ---- Tier 1.95: an ESTER-OXYGEN attachment is `<acyl>oxy` ----
    # The sibling of the carboxy guard above, and the SAME root cause: the acid
    # name -> acyl prefix table (`substituent_naming.py:3460`) is applied to a
    # fragment that retains an oxygen the acyl group does not have. Which oxygen
    # is lost depends only on where the fragment attaches:
    #
    # attached via the carbonyl C, keeping -OH -> `carboxy` (guard above)
    # attached via the ester O, keeping -O- -> `<acyl>oxy` (here)
    #
    # Measured: `-O-C(=O)CH3` came out `acetyl`, which is `-C(=O)CH3`, an oxygen
    # SHORT -- and OPSIN confirms the two differ, `(acetyl)benzene` parsing to
    # `C(C)(=O)C1=CC=CC=C1` against `(acetyloxy)benzene`'s
    # `C(C)(=O)OC1=CC=CC=C1`. So this is a wrong molecule, not a spelling choice.
    #
    # Composing acyl + `oxy` is the nomenclature operation specifies
    # for an ester cited as a prefix, NOT string surgery on an emitted name: the
    # acyl morpheme is looked up for the fragment and the `oxy` morpheme is
    # appended, which is how the prefix is formed.
    #
    # Longer acyloxy chains (`-OC(=O)CH2CH3`) never reach here -- the chain path
    # already spells them `2-oxo-1-oxabutyl`, verified RT-exact -- so this fires
    # only for the short acids the retained table intercepts first.
    try:
        if len(frag_atoms_set) >= 3:
            _o = mol.GetAtomWithIdx(attach_idx)
            if (_o.GetSymbol() == 'O' and _o.GetFormalCharge() == 0
                    and _o.GetTotalNumHs() == 0):
                _nbrs = [n.GetIdx() for n in _o.GetNeighbors()
                         if n.GetIdx() in frag_atoms_set]
                if len(_nbrs) == 1:
                    _c = mol.GetAtomWithIdx(_nbrs[0])
                    _has_oxo = any(
                        b.GetBondType() == Chem.BondType.DOUBLE
                        and b.GetOtherAtom(_c).GetSymbol() == 'O'
                        and b.GetOtherAtom(_c).GetIdx() in frag_atoms_set
                        for b in _c.GetBonds())
                    if _c.GetSymbol() == 'C' and _has_oxo:
                        from .fragment_naming import FRAGMENT_NAME_CACHE
                        from .substituent_naming import (
                            ATTACH_LOCANT_UNKNOWN,
                            parent_to_prefix,
                        )
                        _fs = Chem.MolFragmentToSmiles(
                            mol, list(frag_atoms_set))
                        _acid = FRAGMENT_NAME_CACHE.get(Chem.CanonSmiles(_fs))
                        if _acid:
                            _n_c = sum(
                                1 for i in frag_atoms_set
                                if mol.GetAtomWithIdx(i).GetAtomicNum() == 6)
                            _acyl = parent_to_prefix(
                                _acid, chain_length=_n_c,
                                attach_locant=ATTACH_LOCANT_UNKNOWN)
                            if (_acyl and _acyl.endswith('yl')
                                    and 'oxy' not in _acyl):
                                return _stereo_route(f'{_acyl}oxy')
    except Exception:  # pragma: no cover - a guard must never break naming
        pass

    # ---- Tier 2: Static fragment cache (O(1)) ----
    try:
        frag_smiles = Chem.MolFragmentToSmiles(mol, list(frag_atoms_set))
        if frag_smiles:
            canonical = Chem.CanonSmiles(frag_smiles)
            if canonical:
                cached = FRAGMENT_NAME_CACHE.get(canonical)
                if cached:
                    # Convert parent name to prefix form
                    carbon_count = sum(
                        1 for i in frag_atoms_set
                        if mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                    )
                    prefix = parent_to_prefix(
                        cached, chain_length=carbon_count,
                        attach_locant=ATTACH_LOCANT_UNKNOWN)
                    if prefix:
                        return _stereo_route(prefix)
    except Exception:
        pass  # Cache miss is fine, continue to next tier

    # ---- Tier 3: Linear alkyl fast path (attached at a chain TERMINUS) ----
    # DD5 /: a linear chain attached at an INTERNAL carbon
    # (pentan-3-yl, hexan-2-yl) is NOT a terminal alkyl — defer to Tier 4's
    # located deriver so the free valence becomes the numbering basis.
    try:
        from .substituent_naming import _attach_is_chain_terminus
        if _is_linear_alkyl(mol, list(frag_atoms_set)) and _attach_is_chain_terminus(
            mol, list(frag_atoms_set), attach_idx
        ):
            carbon_count = sum(
                1 for i in frag_atoms_set
                if mol.GetAtomWithIdx(i).GetAtomicNum() == 6
            )
            if carbon_count > 0:
                return _stereo_route(get_alkyl_name(carbon_count))
    except Exception:
        pass

    # ---- Tier 4: Recursive compound naming ----
    # Skip Tier 4 for single-atom non-carbon fragments (halogens, -OH, -NH2,
    # =O, etc.) where the recursive namer produces garbled results like
    # "ammoniayl" or "unknown organic compoundyl". The descriptive fallback
    # (Tier 5) handles these correctly.
    _skip_tier4 = False
    if len(frag_atoms_set) == 1:
        _single_atom = mol.GetAtomWithIdx(next(iter(frag_atoms_set)))
        if _single_atom.GetAtomicNum() != 6:
            _skip_tier4 = True

    if not _skip_tier4:
        try:
            result = name_substituent_fragment(
                mol, list(frag_atoms_set), attach_idx, []
            )
            if result and "unknown" not in result.lower():
                return result
        except Exception:
            pass

    # ---- Tier 4.5 (Composer1 Task 2): recursive decoration composition --
    # Every narrow producer has declined. Under the general-fallback context
    # (allow_mancude, threaded from the complete/best-effort engine) the
    # recursive composer partitions a ring-bearing fragment into its ring CORE
    # (named via the general ring engine) + DECORATIONS (each named by
    # RE-ENTERING name_substituent on a strictly-smaller atom set) and
    # assembles one hyphenated `-yl` token, so a decorated ring stops abstaining
    # instead of collapsing to the 'substituent' sentinel. If it cannot
    # decompose the fragment it returns None; we then de-mask the descriptive
    # 'substituent' sentinel to a clean None (a ring-bearing fragment no honest
    # namer could express) while STILL letting the descriptive fallback name a
    # simple non-ring fragment (hydroxy / amino / methyl / halogen). The PIN
    # default (allow_mancude False) keeps the EXACT existing sentinel behavior
    # -> byte-identical.
    if allow_mancude:
        _rec = _recursive_fragment_substituent_name(
            mol, frag_atoms_set, attach_idx, allow_mancude=True)
        if _rec is not None:
            return _rec
        # ---- Tier 4.9 (-T1b): the AUDITED systematic ring generator ----
        # The polarity inversion, localised. Every narrow producer AND the
        # recursive composer have declined, so the next line used to be the
        # ``'substituent'`` refusal sentinel (``errors.py:224``: "a REFUSAL, not
        # a name"). Put the systematic generator there instead: a von
        # Baeyer/spiro descriptor or a replacement monocycle, with a
        # locant for every skeletal heteroatom (λ where hypervalent) and every
        # ring multiple bond, gated on a RECONSTRUCTION AUDIT of the emitted
        # string (``rules/terminal_ring``).
        #
        # Restricted to a fragment that IS exactly one connected ring system:
        # this generator names a RING, so calling it on a decorated fragment
        # would silently DROP the decorations (a wrong structure, not an uglier
        # name). Decorated fragments are the composer's job above -- and its
        # monocycle core tail now falls through to the same generator, so they
        # are covered there rather than here.
        _term = _terminal_bare_ring_substituent(
            mol, frag_atoms_set, attach_idx)
        if _term is not None:
            return _term
        #: the DECORATED-fragment sibling of the generator above. Where
        # ``_terminal_bare_ring_substituent`` requires the fragment to BE exactly
        # one bare ring system, this one names a ring system PLUS its decorations,
        # or a complex acyclic fragment, and is complete by construction: it
        # returns None rather than any name that fails its own atom-coverage
        # invariant, so it cannot do the silent-drop that the comment above warns
        # about.
        #
        # Wired on THIS side only -- ``allow_mancude`` is True here, i.e. the
        # complete/best-effort tier. Deliberately NOT wired at the Tier-5 return
        # below, which is the PIN/DEFAULT path: emitting a systematic
        # 'a'-replacement name there would assert PIN status for a non-PIN name
        # (``1-oxacyclohexan-4-yl`` where the PIN is ``oxan-4-yl``) and break PIN
        # byte-identity. See the comment on that return.
        # ORDER IS LOAD-BEARING: LAST resort, never first.
        # ``_descriptive_fallback`` is not only a sentinel factory -- it produces
        # the CORRECT specific prefix for small fragments (``hydroxy``, ``amino``,
        # ``oxo``, ``cyano``, the halogens) and returns the bare word
        # ``substituent`` only when everything else has declined. Calling the
        # systematic generator BEFORE it preempts those: measured on the 500-row
        # census, that ordering rewrote ``2-hydroxy-…`` to ``2-(1-oxamethyl)-…``,
        # emitted ``1-oxa-2,3-diphosphapropyl`` where specific prefixes existed,
        # and LOST two correct names outright (EMIT -> ABSTAIN). Textbook
        # a project rule: removing a refusal path unmasked a worse generator.
        # So it runs only where the sentinel would otherwise be returned.
        _desc = _descriptive_fallback(mol, frag_atoms_set, attach_idx)
        if _desc != 'substituent':
            return _desc
        from ..rules.terminal_fragment import terminal_fragment_name
        _tf = terminal_fragment_name(mol, frag_atoms_set, attach_idx)
        if _tf is not None:
            return _tf.name
        # ---- C4 : decorated acyclic-chain substituent -----------------
        # LAST resort for a chain-rooted multi-functional fragment the narrow
        # tiers, the ring composer, the descriptive fallback and the terminal
        # chain namer all declined. Demotes every characteristic group to a
        # prefix (structure-based), so a fragment that names standalone only via
        # a SUFFIX stops dropping its atoms. None -> clean abstain (unchanged).
        _chain = _recursive_chain_fragment_substituent_name(
            mol, frag_atoms_set, attach_idx, allow_mancude=True)
        if _chain is not None:
            return _chain
        # census-attribution fix: the best-effort ladder is now FULLY
        # exhausted for this branch (ring composer, bare-ring, descriptive,
        # terminal and chain composers all declined). Record the
        # ``enumerator_ring_fallback`` census blocker HERE, gated on the same
        # ring-bearing condition the descriptive fallback used, so it reflects the
        # TRUE best-effort failure -- not an exploratory PIN attempt (:2944) and
        # not a branch the chain composer would have named. Telemetry only; never
        # alters naming output (returns None either way).
        try:
            _ri = mol.GetRingInfo()
            if any(_ri.NumAtomRings(a) > 0 for a in frag_atoms_set):
                from ..metrics.abstention import AbstentionCode, record_abstention
                record_abstention(AbstentionCode.BRANCH_UNNAMEABLE,
                                  detail='enumerator_ring_fallback')
        except Exception:
            pass
        return None

    # ---- Tier 5: Descriptive fallback (guaranteed non-None) ----
    #
    # ⚠ -T1b, deliberately NOT the wiring site for the systematic ring
    # generator, although this is where the sentinel is returned from. Reaching
    # here means ``allow_mancude`` is False -- the general-fallback branch above
    # returns unconditionally -- i.e. this is the PIN/DEFAULT path. A von Baeyer
    # polyene or an 'a'-replacement monocycle is a VALID name but not the
    # PREFERRED one (``thian-3-yl`` is the PIN, ``1-thiacyclohexan-3-yl`` is
    # not), so emitting one here would assert PIN status for a non-PIN name and
    # would break PIN byte-identity. The generator is wired on the
    # ``allow_mancude`` (complete / best-effort tier) side only, three lines up.
    return _descriptive_fallback(mol, frag_atoms_set, attach_idx)


def _terminal_bare_ring_substituent(mol, frag_atoms, attach_idx):
    """``rules.terminal_ring`` applied to a fragment that is exactly ONE
    connected ring system, or ``None``.

    The atom-conservation guard is the point: this returns a name for the RING,
    so it may only be used when the fragment has no non-ring heavy atom. A
    decorated fragment returns ``None`` here and is handled by the recursive
    composer, which owns the decorations.
    """
    if attach_idx is None:
        return None
    frag = {a for a in frag_atoms
            if mol.GetAtomWithIdx(a).GetAtomicNum() > 1}
    if not frag or attach_idx not in frag:
        return None
    ri = mol.GetRingInfo()
    if any(ri.NumAtomRings(a) == 0 for a in frag):
        return None  # a non-ring heavy atom -> not a bare ring system
    if ri.NumAtomRings(attach_idx) == 0:
        return None
    # one CONNECTED ring system (a fragment holding two separate rings joined by
    # nothing is not a single ring system and has no single descriptor)
    seen = {attach_idx}
    stack = [attach_idx]
    while stack:
        cur = stack.pop()
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            j = nb.GetIdx()
            if j in frag and j not in seen and ri.NumAtomRings(j) > 0:
                seen.add(j)
                stack.append(j)
    if seen != frag:
        return None
    from ..rules.terminal_ring import terminal_ring_name
    result = terminal_ring_name(mol, sorted(frag), attach_idx)
    return result.name if result is not None else None


def _descriptive_fallback(mol, frag_atoms, attach_idx):
    """Produce a compositional description for unnameable fragments.

    Analyzes fragment atoms directly to build a best-effort prefix name.
    For simple fragments (1-3 atoms), produces specific names like
    "hydroxy", "amino", "methyl". For complex unnameable fragments,
    returns "substituent" as absolute last resort.

    Args:
        mol: RDKit Mol object.
        frag_atoms: Set of atom indices in the fragment.
        attach_idx: Attachment atom index.

    Returns:
        str: Always non-None, always non-empty.
    """
    if not frag_atoms:
        return "substituent"

    # Wave2 constitution-conservation guard: a ring-bearing fragment down
    # here was declined by every honest namer (incl. the ring engine). The
    # carbon-count alkyl branch below would flatten it into a linear chain
    # (methylcyclohexyl -> 'heptyl', a DIFFERENT constitution). Return the
    # explicit unnameable marker instead.
    try:
        _ri = mol.GetRingInfo()
        if any(_ri.NumAtomRings(a) > 0 for a in frag_atoms):
            # Task 0.1: a ring-bearing branch reaching this last resort is
            # declined by the descriptive namer; return the explicit unnameable
            # marker rather than the alkyl-flatten below (which would be a wrong
            # constitution). census-attribution fix: the
            # ``enumerator_ring_fallback`` BRANCH_UNNAMEABLE record is NO LONGER
            # made here. This function is reached from BOTH the PIN Tier-5 return
            # (name_substituent:2944, allow_mancude=False) and the best-effort
            # ladder (:2917) BEFORE its own terminal/chain composers run, so a
            # record here is first-writer-wins-set by an exploratory PIN attempt
            # or by a branch the chain composer goes on to name (a review label-leak
            # finding). The record now fires only at the best-effort ladder's TRUE
            # exhaustion in name_substituent (after the chain composer), so
            # blocker_detail reflects the real best-effort blocker.
            return "substituent"
    except Exception:
        pass

    # Analyze fragment composition
    carbons = 0
    heteroatoms = {}
    for idx in frag_atoms:
        atom = mol.GetAtomWithIdx(idx)
        anum = atom.GetAtomicNum()
        if anum == 1:
            continue
        if anum == 6:
            carbons += 1
        else:
            sym = atom.GetSymbol()
            heteroatoms[sym] = heteroatoms.get(sym, 0) + 1

    # Single-atom fragments
    if len(frag_atoms) == 1:
        idx = next(iter(frag_atoms))
        atom = mol.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()
        total_hs = atom.GetTotalNumHs()

        # Halogens
        if sym in _HALOGEN_MAP:
            return _HALOGEN_MAP[sym]
        # Oxygen
        if sym == 'O':
            return 'hydroxy' if total_hs >= 1 else 'oxo'
        # Nitrogen
        if sym == 'N':
            if total_hs >= 2:
                return 'amino'
            elif total_hs == 1:
                return 'imino'
            else:
                return 'azanyl'
        # Sulfur
        if sym == 'S':
            return 'sulfanyl' if total_hs >= 1 else 'sulfanylidene'
        # Single carbon --: the morphology follows the ATTACHMENT BOND
        # ORDER, not the hydrogen count. The neighbouring branches above
        # discriminate on H count, which merely correlates; that correlation is
        # what made this branch emit 'methyl' unconditionally and name an
        # exocyclic =CH2 as a singly-bonded methyl -- a wrong STRUCTURE.
        if sym == 'C':
            free_valence = _free_valence_at_attachment(
                mol, frag_atoms, attach_idx)
            suffix = _FREE_VALENCE_SUFFIX.get(free_valence)
            if suffix is None:
                # Undecidable shape (bridge, aromatic linkage). Naming it
                # 'methyl' would assert a single free valence this function
                # cannot see; abstain instead.
                return "substituent"
            return f"meth{suffix}"

    # Carbon-only fragments: use alkyl names
    if carbons > 0 and not heteroatoms:
        try:
            return get_alkyl_name(carbons)
        except (ValueError, KeyError):
            pass

    # Multi-atom heteroatom-only fragments
    if carbons == 0 and heteroatoms:
        symbols = sorted(heteroatoms.keys())
        # -NO2 (nitro)
        if symbols == ['N', 'O'] and heteroatoms.get('N', 0) == 1 and heteroatoms.get('O', 0) == 2:
            return 'nitro'
        # -N=O (nitroso): attach is a NEUTRAL N carrying exactly one terminal,
        # doubly-bonded O. Guarded on the attach atom + bond order so a
        # nitrite -O-N=O (attach O, O count 2) or an N-oxide can never match.
        if (symbols == ['N', 'O'] and heteroatoms.get('N', 0) == 1
                and heteroatoms.get('O', 0) == 1):
            _a = mol.GetAtomWithIdx(attach_idx)
            if _a.GetSymbol() == 'N' and _a.GetFormalCharge() == 0:
                _os = [n for n in _a.GetNeighbors() if n.GetSymbol() == 'O']
                if (len(_os) == 1 and _os[0].GetDegree() == 1
                        and _os[0].GetFormalCharge() == 0
                        and mol.GetBondBetweenAtoms(attach_idx, _os[0].GetIdx())
                            .GetBondType() == Chem.BondType.DOUBLE):
                    return 'nitroso'
        # -N3 (azido)
        if symbols == ['N'] and heteroatoms.get('N', 0) == 3:
            return 'azido'
        # Single heteroatom type
        if len(symbols) == 1:
            sym = symbols[0]
            if sym == 'O':
                return 'hydroxy'
            if sym == 'N':
                return 'amino'
            if sym == 'S':
                return 'sulfanyl'
            if sym in _HALOGEN_MAP:
                return _HALOGEN_MAP[sym]

    # ---- Compound substituents: carbon + heteroatom combinations ----
    # IUPAC: compound prefix names built from
    # heteroatom-prefix + alkyl-stem (e.g., hydroxymethyl, aminoethyl).
    # Restricted to avoid positional ambiguity.

    if carbons > 0 and heteroatoms:
        # Cyano: exactly 1C + 1N with triple bond (nitrile substituent)
        if carbons == 1 and heteroatoms == {'N': 1}:
            for idx in frag_atoms:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == 'C':
                    for nbr in atom.GetNeighbors():
                        if nbr.GetIdx() in frag_atoms and nbr.GetSymbol() == 'N':
                            bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
                            if bond and bond.GetBondTypeAsDouble() == 3.0:
                                return 'cyano'

        # 2-HA fragments (1 carbon + 1 heteroatom): unambiguous position
        total_ha = carbons + sum(heteroatoms.values())
        if total_ha <= 2 and carbons == 1:
            hetero_prefix = None
            for idx in frag_atoms:
                atom = mol.GetAtomWithIdx(idx)
                sym = atom.GetSymbol()
                if sym == 'O':
                    hetero_prefix = 'hydroxy' if atom.GetTotalNumHs() >= 1 else 'oxo'
                    break
                elif sym == 'N':
                    hetero_prefix = 'amino' if atom.GetTotalNumHs() >= 2 else 'imino'
                    break
                elif sym == 'S':
                    hetero_prefix = 'sulfanyl' if atom.GetTotalNumHs() >= 1 else 'thio'
                    break
                elif sym in _HALOGEN_MAP:
                    hetero_prefix = _HALOGEN_MAP[sym]
                    break

            if hetero_prefix:
                try:
                    alkyl_stem = get_alkyl_name(carbons)
                    return f"{hetero_prefix}{alkyl_stem}"
                except (ValueError, KeyError):
                    pass

    # Absolute last resort
    # Task 0.1: no tier could express this branch.
    from ..metrics.abstention import AbstentionCode, record_abstention
    record_abstention(AbstentionCode.BRANCH_UNNAMEABLE,
                      detail='enumerator_last_resort')
    return "substituent"


# ============================================================================
# Composer1 Task 1: detach-and-name ring-substituent primitive
# ============================================================================


def _detach_and_name_ring_substituent(mol, frag_atoms, attach_idx,
                                       allow_mancude: bool = False):
    """Name the RING-SYSTEM CORE of a substituent fragment as a ``-yl`` token.

    NOTE (Composer #1 final review, I2): retained as a tested T1
    primitive (see ``tests/unit/rules/test_v28_composer1.py``) but SUPERSEDED
    in production by ``_recursive_fragment_substituent_name``'s own core
    numbering (``_monocycle_position_map`` / ``polycyclic_core_numbering``),
    which can place a DECORATION locant on the core -- something this
    primitive's conjoined ``-yl`` token cannot carry. Not on any production
    call path today; kept for reuse/testing.

    Reusable FIRST primitive for the always-emit recursive substituent
    composer: locates the ring atoms within ``frag_atoms`` and routes them
    through the existing detached-submol von-Baeyer/spiro/cage/monocycle
    engine (``ring_substituents.get_ring_substituent_name``, which itself
    dispatches to ``_polycyclic_substituent_name`` ->
    ``_universal_cage_substituent_name`` / ``_universal_spiro_substituent_name``
    under ``allow_mancude``, and to the monocyclic/mancude-heteromonocyclic
    producers otherwise). The detachment itself (submol construction with the
    broken bond left as an implicit-H valence cap, NOT raw atom-index copying)
    is performed by that existing, already-audited machinery via
    ``ring_substituents._extract_ring_submol`` — this primitive does not
    duplicate it.

    Does NOT reimplement von-Baeyer/spiro/cage naming; does NOT walk
    substituents hanging off the ring core (that recursion is later
    composer tasks). Not wired into ``name_substituent`` yet  —
    PIN-default (``allow_mancude=False`` callers) is unaffected by this
    addition.

    Args:
        mol: RDKit Mol of the full molecule.
        frag_atoms: Iterable of atom indices belonging to the substituent
            fragment (may include non-ring decoration atoms; only the atoms
            that are IN A RING are used as the ring core).
        attach_idx: Atom index (must be one of ``frag_atoms``, and must be a
            ring atom) marking the open-valence attachment point.
        allow_mancude: / opt-in (complete/best-effort engine tier
            only) threaded straight to ``get_ring_substituent_name`` so a
            cage (tricyclo+/adamantane) or mancude fused-aromatic ring core
            is named instead of failing closed. Default False.

    Returns:
        str: A ``...-<loc>-yl`` (or ``-ylidene``) substituent token for the
            ring core, or ``None`` (fail-closed) if the fragment has no
            nameable ring core -- attach_idx is not a ring atom within
            frag_atoms, or every ring namer declines.
    """
    from ..rules.ring_substituents import get_ring_substituent_name

    frag_set = set(frag_atoms)
    if attach_idx is None or attach_idx not in frag_set:
        return None

    ring_info = mol.GetRingInfo()
    ring_atoms = tuple(sorted(
        a for a in frag_set if ring_info.NumAtomRings(a) > 0
    ))
    if not ring_atoms or attach_idx not in ring_atoms:
        # No ring core in this fragment, or the attachment point itself is
        # not part of the ring core -- nothing for this primitive to name.
        return None

    try:
        name = get_ring_substituent_name(
            mol, ring_atoms, attach_idx, allow_mancude=allow_mancude
        )
    except Exception:
        name = None

    if not name or name == 'substituent' or ' ' in name:
        return None
    return name


# ============================================================================
# Composer1 Task 2: recursive decoration composition
# ============================================================================


def _ring_system_core_atoms(mol, attach_idx, frag_set, ring_info):
    """Ring-system atoms (within ``frag_set``) connected to ``attach_idx``
    through shared-ring membership (the fused component containing the
    attachment). Returns a ``set`` of atom indices, or ``None`` if the
    attachment is not a ring atom of the fragment."""
    if ring_info.NumAtomRings(attach_idx) == 0:
        return None
    frag_ring = {a for a in frag_set if ring_info.NumAtomRings(a) > 0}
    if attach_idx not in frag_ring:
        return None
    core = {attach_idx}
    changed = True
    while changed:
        changed = False
        for ring in ring_info.AtomRings():
            rs = set(ring)
            if not rs <= frag_ring:
                continue
            if (rs & core) and not (rs <= core):
                core |= rs
                changed = True
    return core


def _monocycle_position_map(mol, core, attach_idx, deco_carriers, ring_info):
    """ / free-valence numbering of a SIMPLE monocyclic
    ring ``core``. Returns ``{atom_idx: locant}`` for the winning numbering, or
    ``None`` for anything that is not a simple monocycle (fused / spiro /
    bridged atom, broken cycle). Key order (lowest wins): heteroatom locant set
    -> heteroatom element-seniority locants -> indicated-H locant -> free
    valence locant -> decoration-carrier locant set. This mirrors the validated
    numbering of ``ring_substituents._decorated_heteroaryl_substituent_name``.
    """
    from ..rules.ring_substituents import _HETEROATOM_SENIORITY
    core = list(core)
    n = len(core)
    core_set = set(core)
    adj = {}
    for i in core:
        if ring_info.NumAtomRings(i) != 1:
            return None  # fused / spiro / bridged -> not a simple monocycle
        nb = [x.GetIdx() for x in mol.GetAtomWithIdx(i).GetNeighbors()
              if x.GetIdx() in core_set]
        if len(nb) != 2:
            return None
        adj[i] = nb
    # Walk the cyclic order.
    order = [core[0], adj[core[0]][0]]
    while len(order) < n:
        prev, cur = order[-2], order[-1]
        nxt = [x for x in adj[cur] if x != prev]
        if not nxt:
            return None
        order.append(nxt[0])
    if len(order) != n or order[0] not in adj[order[-1]]:
        return None  # not a single closed cycle
    het = [i for i in core if mol.GetAtomWithIdx(i).GetSymbol() != 'C']
    # tranche T3: indicated hydrogen is a mancude-ring concept — a fully
    # SATURATED ring has none, so its ring N-H atoms must NOT be read as
    # ambiguous indicated H (this was declining piperazine's two N-H, blocking
    # every decorated saturated N/O-heterocycle substituent). Account for
    # indicated H only on an aromatic/mancude core.
    aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in core)
    ih = ([i for i in het if mol.GetAtomWithIdx(i).GetTotalNumHs() >= 1]
          if aromatic else [])
    if len(ih) > 1:
        return None  # ambiguous indicated hydrogen -> do not guess
    ih_atom = ih[0] if ih else None
    deco_set = set(deco_carriers)
    best_key = None
    best = None
    for start in range(n):
        for direction in (1, -1):
            a2p = {order[(start + direction * p) % n]: p + 1 for p in range(n)}
            het_locs = tuple(sorted(a2p[i] for i in het))
            sen_locs = tuple(a2p[i] for i in sorted(
                het, key=lambda a: (_HETEROATOM_SENIORITY.get(
                    mol.GetAtomWithIdx(a).GetSymbol(), 99), a2p[a])))
            ih_loc = a2p[ih_atom] if ih_atom is not None else 0
            fv_loc = a2p[attach_idx]
            deco_locs = tuple(sorted(a2p[c] for c in deco_set))
            key = (het_locs, sen_locs, ih_loc, fv_loc, deco_locs)
            if best_key is None or key < best_key:
                best_key = key
                best = a2p
    return best


def _borrow_heteroarene_stem(mol, core, attach_idx):
    """ tranche T1: extract the numbering-INDEPENDENT ring-skeleton stem
    (e.g. ``'1,2-oxazol'``) from the authoritative ``get_ring_substituent_name``
    output for the BARE core, or ``None``.

    Only the stem is borrowed; the free-valence and decoration locants are
    (re)cited by the caller on the recursion's own numbering. Used when the small
    ``_PIN_HETEROARYL_STEMS`` table lacks a systematic azole that the dispatcher
    can name (isoxazole/oxazole/thiazole/triazole...). Fail-closed (``None``) for
    a multi-word or non ``{stem}-<loc>-yl`` shape.
    """
    import re

    from ..rules.ring_substituents import get_ring_substituent_name
    try:
        bare = get_ring_substituent_name(mol, tuple(sorted(core)), attach_idx)
    except Exception:
        return None
    if not bare or ' ' in bare:
        return None
    m = re.match(r'^(?:\d+H-)?(?P<stem>.+?)-\d+-yl$', bare)
    if not m:
        return None
    return m.group('stem')


def _monocycle_core_tail(mol, core, attach_idx, pos, ring_info):
    """Build the bare monocyclic ring-substituent tail (``phenyl`` /
    ``pyridin-3-yl`` / ``cyclohexyl`` / ``1H-pyrrol-2-yl``...) using the PIN
    stem tables, with the free-valence locant taken from ``pos``. Fail-closed
    (``None``) for a ring that is not a confidently-PIN monocyclic stem
    (partially unsaturated, ambiguous indicated H, or stem not in the table).
    Carbocyclic aromatic-6 / saturated forms cite the free valence implicitly
    at position 1 (``phenyl`` / ``cyclohexyl``), so those require
    ``pos[attach] == 1``.
    """
    from ..data.chain_names import get_chain_prefix
    from ..rules.ring_substituents import _PIN_HETEROARYL_STEMS, identify_ring_system
    core_set = set(core)
    het = [i for i in core if mol.GetAtomWithIdx(i).GetSymbol() != 'C']
    aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in core)

    def _all_single_ring_bonds():
        for a in core:
            for b in mol.GetAtomWithIdx(a).GetNeighbors():
                if b.GetIdx() in core_set and a < b.GetIdx():
                    bd = mol.GetBondBetweenAtoms(a, b.GetIdx())
                    if bd.GetIsAromatic() or bd.GetBondTypeAsDouble() != 1.0:
                        return False
        return True

    if not het:
        if aromatic and len(core) == 6:
            return 'phenyl' if pos[attach_idx] == 1 else None
        if _all_single_ring_bonds():
            if pos[attach_idx] != 1:
                return None
            return f'cyclo{get_chain_prefix(len(core))}yl'
        return None  # mixed-saturation carbocycle -> ene locants needed

    # Heterocycle: aromatic OR fully saturated only (mixed -> ene-locants).
    if not aromatic and not _all_single_ring_bonds():
        return None
    ring_name = identify_ring_system(mol, tuple(sorted(core)))
    stem = _PIN_HETEROARYL_STEMS.get(ring_name)
    # Pyrazole vs imidazole: identify_ring_system reports both N,N 5-rings as
    # 'imidazole'; adjacent ring nitrogens => pyrazole.
    if ring_name == 'imidazole':
        n_idx = [i for i in het if mol.GetAtomWithIdx(i).GetSymbol() == 'N']
        if len(n_idx) == 2 and mol.GetBondBetweenAtoms(
                n_idx[0], n_idx[1]) is not None:
            stem = 'pyrazol'
    if stem is None:
        # tranche (resolves Composer #1 I1 duplication): the small
        # `_PIN_HETEROARYL_STEMS` table does not cover the systematic azoles
        # (isoxazole/oxazole/thiazole/triazole...) that `identify_ring_system`
        # reports as None. The authoritative dispatcher `get_ring_substituent_name`
        # names those via its retained/Hantzsch-Widman path. Borrow its
        # numbering-INDEPENDENT ring-skeleton stem here and re-cite the free
        # valence (+ the caller's decorations) on THIS recursion's own numbering.
        # This function is reached ONLY from the allow_mancude-gated
        # `_recursive_fragment_substituent_name`, so the PIN default is byte-
        # identical; a wrong stem is caught downstream by.
        stem = _borrow_heteroarene_stem(mol, core, attach_idx)
    if stem is None:
        return None
    # tranche T3: indicated H only on an aromatic/mancude core (a saturated
    # heterocycle's N-H is not indicated H) — mirrors _monocycle_position_map.
    ih = ([i for i in het if mol.GetAtomWithIdx(i).GetTotalNumHs() >= 1]
          if aromatic else [])
    if len(ih) > 1:
        return None
    ih_prefix = f'{pos[ih[0]]}H-' if ih else ''
    return f'{ih_prefix}{stem}-{pos[attach_idx]}-yl'


def _terminal_monocycle_core_tail(mol, core, attach_idx, pos):
    """-T1b: the AUDITED replacement tail for a monocyclic core
    the PIN stem tables declined, on the caller's OWN ``pos`` numbering.

    ``None`` on any refusal, including a failed reconstruction audit. Kekulizes
    an index-preserving copy first -- RDKit reports an aromatic bond order as
    **1.5**, so an un-kekulized mancude ring would lose every double bond and the
    emitted name would denote the saturated ring (a wrong structure). The audit
    also refuses order 1.5 outright, so this is proven twice.
    """
    from rdkit import Chem

    from ..rules.terminal_ring import (
        audit_monocycle_replacement_name,
        build_monocycle_replacement_name,
    )
    core_set = set(core)
    if attach_idx not in core_set or set(pos) != core_set:
        return None
    try:
        rw = Chem.RWMol(mol)
        Chem.Kekulize(rw, clearAromaticFlags=True)
        kek = rw.GetMol()
    except Exception:  # noqa: BLE001 - kekulization must degrade, never raise
        return None
    for i in core_set:
        if kek.GetAtomWithIdx(i).GetFormalCharge() != 0:
            return None  # class; no 'a'-prefix name expresses a ring ion
    name = build_monocycle_replacement_name(
        kek, sorted(core_set), pos, pos[attach_idx])
    if name is None:
        return None
    if not audit_monocycle_replacement_name(
            kek, sorted(core_set), pos, name, attach_idx):
        logger.info("terminal monocycle tail %r failed the reconstruction "
                    "audit; refuse", name)
        return None
    return name


def _recursive_fragment_substituent_name(mol, frag_atoms, attach_idx,
                                          allow_mancude: bool = False):
    """General recursive composer: name a ring-bearing substituent fragment as
    ``{decorations}{ring-core}-yl`` by partitioning it into a ring CORE + its
    DECORATIONS and recursing.

     always-emit keystone (Composer #1, Task 2). Reached from
    ``name_substituent`` ONLY under the general-fallback context
    (``allow_mancude=True``) after every narrow tier declines, so the PIN
    default path is byte-identical. Algorithm:

    1. CORE = the (fused) ring system within ``frag_atoms`` containing the
       attachment. (``attach_idx`` must be a ring atom; a chain-rooted or
       ring-on-chain fragment returns ``None`` -> Task 3 / Composer #2.)
    2. DECORATIONS = each maximal branch hanging off a core atom, collected
       with the shared ``_bfs_collect_fragment`` walk (boundary = the core).
       Every non-core heavy atom MUST land in exactly one decoration
       (COMPOSITION-CONTRACT coverage check) or we fail closed.
    3. Number the core (simple monocycle: free-valence enumeration;
       fused/spiro cores are out of this task -> ``None``).
    4. Name each decoration by RE-ENTERING ``name_substituent`` on a STRICTLY
       SMALLER atom set (termination invariant asserted); a decoration the
       recursion cannot name (or a multi-word / sentinel token) -> fail closed.
    5. Assemble one hyphenated ``-yl`` substituent token: alphabetized,
       multiplied, enclosing-marked decoration prefixes + the bare core tail.

    Returns the composed token, or ``None`` (clean abstain) when the core is
    unnameable, the coverage/termination invariant is violated, or any
    decoration cannot be named. NEVER returns the ``'substituent'`` sentinel
    and NEVER a multi-word token.
    """
    if not allow_mancude:
        return None
    frag_set = set(frag_atoms)
    if not frag_set or attach_idx is None or attach_idx not in frag_set:
        return None

    ring_info = mol.GetRingInfo()
    core = _ring_system_core_atoms(mol, attach_idx, frag_set, ring_info)
    if not core or attach_idx not in core:
        return None

    # ---- Step 2: enumerate decorations (shared fragment walk) --------------
    assigned = set(core)
    decorations = []  # (carrier_core_atom, decoration_attach_atom, atom_set)
    for ra in core:
        for nb in mol.GetAtomWithIdx(ra).GetNeighbors():
            ni = nb.GetIdx()
            if (ni in core or ni not in frag_set or nb.GetAtomicNum() <= 1
                    or ni in assigned):
                continue
            branch = _bfs_collect_fragment(mol, ni, core, assigned)
            if not branch:
                continue
            assigned |= branch
            decorations.append((ra, ni, branch))

    # COMPOSITION CONTRACT: every non-core heavy fragment atom must be covered
    # by exactly one decoration (the shared walk already prevents overlap).
    heavy = {a for a in frag_set if mol.GetAtomWithIdx(a).GetAtomicNum() > 1}
    covered = set(core) | {a for _, _, branch in decorations for a in branch}
    if covered != heavy:
        return None  # an unaccounted heavy atom -> fail closed

    # ---- Step 3: number the core -------------------------------------------
    deco_carriers = [ra for ra, _, _ in decorations]
    pos = _monocycle_position_map(
        mol, core, attach_idx, deco_carriers, ring_info)
    if pos is not None:
        core_tail = _monocycle_core_tail(mol, core, attach_idx, pos, ring_info)
        if not core_tail or ' ' in core_tail:
            # -T1b: the PIN stem tables declined -- a mixed-saturation
            # carbocycle (needs ene locants), a non-aromatic heterocycle with a
            # ring multiple bond, or a stem no table carries. That used to abstain
            # for the WHOLE fragment. Fall through to the AUDITED systematic
            # replacement name instead, on THIS SAME ``pos`` numbering, so the
            # decoration locants read off ``pos`` below stay consistent with the
            # core tail by construction rather than by two numberings agreeing.
            core_tail = _terminal_monocycle_core_tail(mol, core, attach_idx, pos)
            if not core_tail or ' ' in core_tail:
                return None
    else:
        #: POLYCYCLIC (fused / bridged / cage) decorated core. The
        # monocycle numberer declined, so route the core through the shared
        # polycyclic numberer, which returns ONE consistent {atom: locant} map +
        # the bare `...-<fv>-yl` tail (fused-carbocyclic PAH + von-Baeyer cage;
        # other polycyclic classes stay deferred -> None -> fail closed). ATTEMPT
        # a covered candidate; (the production RT gate) arbitrates any
        # uncertain locant so 0-wrong holds without proving numbering perfect.
        from ..rules.ring_substituents import polycyclic_core_numbering
        _poly = polycyclic_core_numbering(
            mol, tuple(sorted(core)), attach_idx, deco_carriers,
            allow_mancude=allow_mancude)
        if _poly is None:
            return None  # un-numberable polycyclic core -> clean abstain
        pos, core_tail = _poly
        if not core_tail or ' ' in core_tail:
            return None

    # No decorations: the bare-core tail IS the answer (earlier tiers normally
    # own this, but stay correct if we reach here for a bare ring).
    if not decorations:
        return core_tail

    # ---- Step 4: name each decoration by recursion -------------------------
    from collections import defaultdict

    from .naming_utils import alpha_sort_key, enclose_if_compound, get_multiplier_prefix
    groups = defaultdict(list)
    for ra, ni, branch in decorations:
        # Termination invariant: recurse only on a STRICTLY SMALLER atom set.
        if not len(branch) < len(frag_set):
            return None
        dname = name_substituent(
            mol, sorted(branch), ni, allow_mancude=allow_mancude)
        if (not dname or dname == 'substituent' or ' ' in dname
                or 'unknown' in dname.lower()):
            return None  # unnameable decoration -> fail closed
        groups[dname].append(pos[ra])

    # ---- Step 5: assemble one hyphenated token -----------------------------
    parts = []  # (alpha_key, text)
    for dname, locs in groups.items():
        locs = sorted(locs)
        token = enclose_if_compound(dname)
        mult = get_multiplier_prefix(len(locs), dname)
        text = f"{','.join(str(l) for l in locs)}-{mult}{token}"
        parts.append((alpha_sort_key(dname), text))
    parts.sort(key=lambda x: x[0])
    body = '-'.join(p[1] for p in parts)
    # Hyphen before the tail only when the tail is digit-initial (indicated H,
    # e.g. '2-methyl-1H-pyrrol-2-yl'); elide before a letter-initial stem.
    sep = '-' if core_tail[0].isdigit() else ''
    result = f"{body}{sep}{core_tail}"
    if ' ' in result or result == 'substituent':
        return None
    # Composite ring-substituent stereo: cite the ring's CIP descriptors
    # at the FRONT of the prefix, on THIS numbering (pos; free valence = locant 1).
    # collect_stereodescriptors only cites atoms/bonds whose locants are in pos, so
    # a boundary bond (an exocyclic ylidene attachment) is left to the caller.
    # Adds only descriptors the token lacks -> can only make a stereo-bearing
    # decorated ring MORE correct (a stereo-stripped token was a wrong molecule the
    # RT gate suppressed); backstops a wrong label.
    result = _prepend_ring_substituent_stereo(mol, pos, result, frag_set)
    return result


def _prepend_ring_substituent_stereo(mol, pos, token, frag_set=None):
    """Prepend the CIP stereodescriptor prefix (``(3S,4S,5R)-``) to a decorated
    ring/chain-substituent token, using the substituent numbering ``pos``. No-op
    when there is no cited stereo or the token already leads with a descriptor.

    ``collect_stereodescriptors`` cites R/S centres + double bonds with BOTH ends
    in ``pos`` (and ring-exocyclic bonds). A substituent's OWN characteristic
    double bond -- an oxime/imine C=N, or a C=C -- has its carbon in ``pos`` but
    its other end (the =N-OR nitrogen, or a decoration carbon) OUTSIDE ``pos``;
     cites that E/Z on the substituent's prefix, at the carbon's locant.
    ``collect_stereodescriptors`` fails closed on it (correct for PARENT scope, a
    different rule), so this SUBSTITUENT-scoped emitter adds it here, guarded by
    ``frag_set`` so the parent-attachment bond (other end NOT in the fragment) is
    never cited.
    """
    if not pos or not token or token.startswith('('):
        return token
    try:
        from ..rules.stereochemistry import (
            collect_stereodescriptors,
            format_stereodescriptor_string,
        )
        descr = list(collect_stereodescriptors(mol, pos))
        _seen = {loc for loc, _ in descr}
        if frag_set is not None:
            for bond in bonds_of(mol):
                if not (bond.HasProp('_CIPCode')
                        and bond.GetProp('_CIPCode') in ('E', 'Z')):
                    continue
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                # exactly one end in pos, the other a DECORATION atom of the same
                # fragment (in frag_set) -> the substituent's own E/Z, cite on the
                # in-pos carbon's locant. Excludes the parent-attachment bond
                # (parent atom not in frag_set).
                in_i, in_j = i in pos, j in pos
                if in_i == in_j:
                    continue
                c_in = i if in_i else j
                other = j if in_i else i
                # Restrict to a HETEROATOM other-end (an oxime/imine C=N, C=N-N
                # hydrazone,...): that is the substituent's own characteristic
                # double bond whose E/Z cites on the prefix and which
                # collect_stereodescriptors skips. A C=C whose other end is a
                # decoration CARBON is NOT cited here -- that bond belongs to the
                # decoration's own recursively-built name (citing it on the core
                # locant double-counted it and broke the secosteroid trienes).
                if (other in frag_set
                        and mol.GetAtomWithIdx(other).GetAtomicNum() != 6
                        and pos[c_in] not in _seen):
                    descr.append((pos[c_in], bond.GetProp('_CIPCode')))
                    _seen.add(pos[c_in])
        if descr:
            sp = format_stereodescriptor_string(descr)
            if sp:
                return f"{sp}{token}"
    except Exception:
        pass
    return token


def _longest_carbon_path_from(mol, frag_set, start):
    """Longest simple path of ACYCLIC carbons in ``frag_set`` beginning at
    ``start``, or ``None``.

    ``start`` is the free-valence atom and becomes one endpoint of the returned
    path, so it takes locant 1 lowest locant for the free valence).
    Backbone bonds may be single, double or triple: the stem is spelled by
    ``_stem_block``, which encodes the ene/yne locants from the SAME numbering,
    so an unsaturated backbone is named correctly rather than as a saturated
    stem (which would be a different molecule). Acyclic carbons form a forest,
    so the DFS terminates.
    """
    carbons = {i for i in frag_set
               if mol.GetAtomWithIdx(i).GetAtomicNum() == 6
               and not mol.GetAtomWithIdx(i).IsInRing()}
    if start not in carbons:
        return None
    best = [start]

    def _dfs(node, path, visited):
        nonlocal best
        if len(path) > len(best):
            best = list(path)
        for nb in mol.GetAtomWithIdx(node).GetNeighbors():
            j = nb.GetIdx()
            if j not in carbons or j in visited:
                continue
            visited.add(j)
            path.append(j)
            _dfs(j, path, visited)
            path.pop()
            visited.discard(j)

    _dfs(start, [start], {start})
    return best


def _recursive_chain_fragment_substituent_name(mol, frag_atoms, attach_idx,
                                               allow_mancude: bool = False):
    """Acyclic sibling of:func:`_recursive_fragment_substituent_name`: name a
    CHAIN-rooted multi-functional substituent as ``{decorations}{alkyl}-yl`` by
    demoting every characteristic group to a detachable prefix.

     C4 (keystone). Reached from ``name_substituent`` ONLY under the
    general-fallback context (``allow_mancude=True``) after every narrow tier,
    the ring composer, the descriptive fallback and ``terminal_fragment_name``
    have declined, so the PIN default path is byte-identical. The gap this fills:
    a fragment whose principal characteristic group would be a SUFFIX standalone
    (``…-amide``/``…-amine``/``…-oic acid``) has no suffix as a substituent, so
    the group must become a prefix (``carbamoyl``/``amino``/``carboxy``) and the
    attach carbon carries the free valence. The dead ``parent_to_prefix`` string
    surgery could not do that demotion; this composer does it from STRUCTURE.

    Algorithm (mirrors the ring composer, chain core instead of ring core):

    1. CORE = the longest all-single-bond acyclic carbon path with the attach
       atom as an endpoint (free valence = locant 1). Requires an acyclic carbon
       attach and a backbone of >= 2 carbons; else ``None`` (earlier tiers /
       ``terminal_fragment_name`` own the mono-carbon and ring cases).
    2. DECORATIONS = each maximal branch hanging off a core carbon, via the
       shared ``_bfs_collect_fragment`` walk. Every non-core heavy atom MUST land
       in exactly one decoration (coverage contract) or fail closed.
    3. Name each decoration by RE-ENTERING ``name_substituent`` on a strictly
       smaller atom set; an unnameable / multi-word / sentinel decoration -> fail
       closed (never a silent drop).
    4. Assemble one hyphenated ``-yl`` token: alphabetized, multiplied,
       enclosing-marked decoration prefixes + the ``get_alkyl_name`` core stem.

    Returns the composed token, or ``None`` (clean abstain). NEVER the
    ``'substituent'`` sentinel and NEVER a multi-word token.
    """
    if not allow_mancude:
        return None
    frag_set = set(frag_atoms)
    if not frag_set or attach_idx is None or attach_idx not in frag_set:
        return None
    a = mol.GetAtomWithIdx(attach_idx)
    if a.GetAtomicNum() != 6 or a.IsInRing():
        return None  # ring / heteroatom attach -> not this composer's job

    chain = _longest_carbon_path_from(mol, frag_set, attach_idx)
    if chain is None or len(chain) < 2:
        return None
    pos = {atom: i + 1 for i, atom in enumerate(chain)}
    chain_set = set(chain)

    # ---- decorations (shared fragment walk) --------------------------------
    assigned = set(chain_set)
    decorations = []  # (carrier_chain_atom, decoration_attach_atom, atom_set)
    for ca in chain:
        for nb in mol.GetAtomWithIdx(ca).GetNeighbors():
            ni = nb.GetIdx()
            if (ni in chain_set or ni not in frag_set or nb.GetAtomicNum() <= 1
                    or ni in assigned):
                continue
            branch = _bfs_collect_fragment(mol, ni, chain_set, assigned)
            if not branch:
                continue
            assigned |= branch
            decorations.append((ca, ni, branch))

    # COVERAGE CONTRACT: every non-core heavy atom covered by exactly one
    # decoration (the shared walk already prevents overlap).
    heavy = {x for x in frag_set if mol.GetAtomWithIdx(x).GetAtomicNum() > 1}
    covered = set(chain_set) | {x for _, _, branch in decorations for x in branch}
    if covered != heavy:
        return None  # an unaccounted heavy atom -> fail closed

    # Core stem, free valence at locant 1. _stem_block spells the ene/yne
    # locants from the SAME `pos`, so an unsaturated backbone is named correctly;
    # a saturated backbone reuses get_alkyl_name (byte-identical to the prior
    # saturated-only path). Import lazily -- general_engine imports this module.
    from .general_engine import _stem_block
    _stem = _stem_block(mol, chain, pos)
    if _stem is None:
        return None  # unsupported backbone bond pattern -> fail closed
    _base, _hydride = _stem
    if _hydride == _base + "ane":
        core_tail = get_alkyl_name(len(chain))  # 'pentyl' (fv=1, elided)
    elif _hydride.endswith("e"):
        core_tail = _hydride[:-1] + "-1-yl"      # 'pent-2-ene' -> 'pent-2-en-1-yl'
    else:
        return None
    if not core_tail:
        return None
    if not decorations:
        return core_tail

    # ---- name each decoration by recursion ---------------------------------
    from collections import defaultdict

    from .naming_utils import alpha_sort_key, enclose_if_compound, get_multiplier_prefix
    groups = defaultdict(list)
    for ca, ni, branch in decorations:
        if not len(branch) < len(frag_set):
            return None  # termination invariant: strictly smaller
        dname = name_substituent(
            mol, sorted(branch), ni, allow_mancude=allow_mancude)
        if (not dname or dname == 'substituent' or ' ' in dname
                or 'unknown' in dname.lower()):
            return None  # unnameable decoration -> fail closed
        groups[dname].append(pos[ca])

    # ---- assemble one hyphenated token -------------------------------------
    parts = []  # (alpha_key, text)
    for dname, locs in groups.items():
        locs = sorted(locs)
        token = enclose_if_compound(dname)
        mult = get_multiplier_prefix(len(locs), dname)
        text = f"{','.join(str(l) for l in locs)}-{mult}{token}"
        parts.append((alpha_sort_key(dname), text))
    parts.sort(key=lambda x: x[0])
    body = '-'.join(p[1] for p in parts)
    # elide before a letter-initial stem ('...5-oxopentyl'); keep the hyphen only
    # before a digit-initial tail (mirrors the ring composer's separator rule).
    sep = '-' if core_tail[0].isdigit() else ''
    result = f"{body}{sep}{core_tail}"
    if ' ' in result or result == 'substituent':
        return None
    # Composite CHAIN-substituent stereo, symmetric with the ring sibling
    # _recursive_fragment_substituent_name: this composer dropped a defined-stereo
    # stereocentre in the chain core ('1-amino-3-(hydroxymethyl)pentyl' from a
    # [C@@H] fragment), and does NOT catch a strip (its RT compares
    # connectivity). Cite the core's CIP descriptors on THIS numbering (pos; free
    # valence = locant 1). Adds only descriptors the token lacks -> can only make a
    # stereo-bearing chain substituent MORE correct; backstops a wrong label.
    result = _prepend_ring_substituent_stereo(mol, pos, result, frag_set)
    return result


# ============================================================================
# Halogen Name Map (for fg_only classification)
# ============================================================================

_HALOGEN_MAP = {
    'F': 'fluoro',
    'Cl': 'chloro',
    'Br': 'bromo',
    'I': 'iodo',
}


# ============================================================================
# Ring Substituent Extraction (ReplaceCore-based)
# ============================================================================


def extract_ring_substituents(mol, ring_atoms, oriented_ring,
                              atom_to_locant=None):
    """Extract all substituent fragments from a ring parent using ReplaceCore.

    Builds a core mol from ring_atoms, calls ReplaceCore to extract all
    non-ring fragments as separate mol objects with isotope-labeled dummy
    atoms indicating attachment points.

    Args:
        mol: RDKit Mol object.
        ring_atoms: Tuple or list of ring atom indices (from principal_ring).
        oriented_ring: List of ring atom indices in IUPAC numbering order.
        atom_to_locant: Optional ``{atom idx -> IUPAC locant}`` INHERITED from
            the producer that spelled the parent name. Where it covers an
            attachment atom it decides that atom's locant; elsewhere the
            ``oriented_ring`` position arithmetic below is used, unchanged.

            This exists because ``oriented_ring`` is a bare ordering and
            ``_get_locant_from_oriented_ring`` can only ever return ``pos + 1``.
            A fused parent's numbering is not a position sequence -- it skips
            (``4`` -> ``4a`` -> ``5``) and it depends on which of several
            automorphic numberings the parent name was spelled from. Deriving a
            substituent locant from position arithmetic while the parent name
            was spelled from a different numbering names a DIFFERENT molecule,
            which is exactly what happened to 2-substituted tetralins.
    """
    ring_set = set(ring_atoms)

    # Build core mol from ring atoms
    core = RWMol()
    idx_map = {}  # original mol idx -> core mol idx
    for atom_idx in ring_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        new_idx = core.AddAtom(Chem.Atom(atom.GetAtomicNum()))
        # Preserve aromaticity for correct matching
        core.GetAtomWithIdx(new_idx).SetIsAromatic(atom.GetIsAromatic())
        idx_map[atom_idx] = new_idx

    # Copy bonds between ring atoms
    added_bonds = set()
    for atom_idx in ring_atoms:
        for bond in mol.GetAtomWithIdx(atom_idx).GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            if begin in ring_set and end in ring_set:
                bond_key = (min(begin, end), max(begin, end))
                if bond_key not in added_bonds:
                    core.AddBond(
                        idx_map[begin], idx_map[end], bond.GetBondType()
                    )
                    added_bonds.add(bond_key)

    core_mol = core.GetMol()

    # Match tuple maps core atom index -> original mol atom index
    match = tuple(ring_atoms)

    # ReplaceCore: removes core, returns fragments with isotope-labeled dummies
    frags = Chem.ReplaceCore(mol, core_mol, match, labelByIndex=True)
    if frags is None:
        return []

    # Split into individual fragment mols
    subfgs = Chem.GetMolFrags(frags, asMols=True, sanitizeFrags=False)
    if not subfgs:
        return []

    # Pre-compute per-branch atom sets for geminal substituent disambiguation
    # Key: ring_atom_idx -> list of frozensets (one per separate branch from that atom)
    branch_map = _compute_branch_map(mol, ring_set)

    # Track which branches have been claimed (for geminal disambiguation)
    claimed_branches = set()  # set of (attach_idx, branch_id) tuples

    results = []
    for frag in subfgs:
        # Find the dummy atom(s) to determine attachment point
        for atom in frag.GetAtoms():
            if atom.GetAtomicNum() == 0:  # dummy atom
                # CRITICAL: isotope 0 is valid (maps to match position 0)
                core_pos = atom.GetIsotope()
                if core_pos < len(match):
                    mol_atom_idx = match[core_pos]

                    # Inherited numbering wins where it covers this atom;
                    # otherwise map to a locant via oriented_ring as before.
                    if atom_to_locant is not None and mol_atom_idx in atom_to_locant:
                        locant = atom_to_locant[mol_atom_idx]
                    else:
                        locant = _get_locant_from_oriented_ring(
                            mol_atom_idx, oriented_ring
                        )

                    # Collect original mol atom indices for this fragment
                    # Use per-branch disambiguation for geminal substituents
                    frag_atoms = _collect_frag_atoms_for_fragment(
                        mol, ring_set, mol_atom_idx, frag,
                        branch_map, claimed_branches
                    )

                    if locant is not None:
                        results.append(SubstituentInfo(
                            frag_mol=frag,
                            locant=locant,
                            attach_mol_idx=mol_atom_idx,
                            frag_atoms=frag_atoms,
                        ))
                break  # only process first dummy atom per fragment

    return results


def _get_locant_from_oriented_ring(mol_atom_idx, oriented_ring):
    """Map a mol atom index to its IUPAC locant via oriented_ring.

    Args:
        mol_atom_idx: Atom index in the original mol.
        oriented_ring: List of atom indices in IUPAC numbering order.

    Returns:
        1-indexed IUPAC locant, or None if not found.
    """
    for pos, ring_atom in enumerate(oriented_ring):
        if ring_atom == mol_atom_idx:
            return pos + 1
    return None


def _compute_branch_map(mol, ring_set):
    """Compute per-branch atom sets for each ring atom.

    For geminal substituents (two substituents on the same ring atom), each
    separate branch needs its own atom set. This function BFS-es from each
    individual non-ring neighbor of each ring atom to produce separate sets.

    Args:
        mol: RDKit Mol object.
        ring_set: Set of ring atom indices.

    Returns:
        Dict mapping ring_atom_idx -> list of frozensets (one per branch).
    """
    branch_map = {}
    for ring_idx in ring_set:
        branches = []
        ring_atom = mol.GetAtomWithIdx(ring_idx)
        for nbr in ring_atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in ring_set:
                continue
            # BFS from this specific neighbor to collect its branch
            visited = set()
            stack = [nbr_idx]
            while stack:
                idx = stack.pop()
                if idx in visited or idx in ring_set:
                    continue
                visited.add(idx)
                for nbr2 in mol.GetAtomWithIdx(idx).GetNeighbors():
                    nbr2_idx = nbr2.GetIdx()
                    if nbr2_idx not in visited and nbr2_idx not in ring_set:
                        stack.append(nbr2_idx)
            if visited:
                branches.append(frozenset(visited))
        if branches:
            branch_map[ring_idx] = branches
    return branch_map


def _collect_frag_atoms_for_fragment(mol, ring_set, attach_ring_idx, frag_mol,
                                      branch_map, claimed_branches):
    """Collect original mol atom indices for a specific fragment.

    Handles geminal substituents by matching frag_mol composition against
    individual branches from branch_map, and tracking which branches have
    already been claimed.

    Args:
        mol: RDKit Mol object.
        ring_set: Set of ring atom indices.
        attach_ring_idx: Ring atom index the fragment is attached to.
        frag_mol: RDKit Mol of the isolated fragment.
        branch_map: Dict from _compute_branch_map.
        claimed_branches: Mutable set of (attach_idx, branch_idx) tuples already used.

    Returns:
        Frozenset of original mol atom indices.
    """
    branches = branch_map.get(attach_ring_idx, [])

    if len(branches) <= 1:
        # Single substituent at this position -- use all non-ring neighbors
        return branches[0] if branches else frozenset()

    # Multiple branches (geminal): match fragment composition to find the right one
    # Count non-dummy heavy atoms in frag_mol
    frag_heavy = _count_frag_heavy_atoms(frag_mol)

    for branch_idx, branch_atoms in enumerate(branches):
        key = (attach_ring_idx, branch_idx)
        if key in claimed_branches:
            continue

        # Count heavy atoms in this branch from original mol
        branch_heavy = {}
        for idx in branch_atoms:
            atom = mol.GetAtomWithIdx(idx)
            anum = atom.GetAtomicNum()
            if anum != 1:  # skip hydrogen
                sym = atom.GetSymbol()
                branch_heavy[sym] = branch_heavy.get(sym, 0) + 1

        if branch_heavy == frag_heavy:
            claimed_branches.add(key)
            return branch_atoms

    # Fallback: if no exact match, claim the first unclaimed branch
    for branch_idx, branch_atoms in enumerate(branches):
        key = (attach_ring_idx, branch_idx)
        if key not in claimed_branches:
            claimed_branches.add(key)
            return branch_atoms

    # Last resort: return union of all branches (shouldn't happen)
    all_atoms = set()
    for b in branches:
        all_atoms.update(b)
    return frozenset(all_atoms)


def _count_frag_heavy_atoms(frag_mol):
    """Count non-dummy, non-H atoms in a fragment mol by element symbol.

    Returns:
        Dict of {symbol: count} for heavy atoms in the fragment.
    """
    counts = {}
    if frag_mol is None:
        return counts
    for atom in frag_mol.GetAtoms():
        anum = atom.GetAtomicNum()
        if anum == 0 or anum == 1:
            continue
        sym = atom.GetSymbol()
        counts[sym] = counts.get(sym, 0) + 1
    return counts


def _collect_frag_original_atoms(mol, ring_set, attach_ring_idx):
    """BFS from a ring atom to collect all non-ring substituent atoms.

    Note: This collects ALL substituent atoms from the ring atom. For geminal
    substituents, use _collect_frag_atoms_for_fragment instead.

    Args:
        mol: RDKit Mol object.
        ring_set: Set of ring atom indices.
        attach_ring_idx: Ring atom index that the substituent is attached to.

    Returns:
        Frozenset of original mol atom indices belonging to the substituent.
    """
    visited = set()
    stack = []

    # Start from neighbors of the ring atom that are NOT in the ring
    ring_atom = mol.GetAtomWithIdx(attach_ring_idx)
    for nbr in ring_atom.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx not in ring_set:
            stack.append(nbr_idx)

    while stack:
        idx = stack.pop()
        if idx in visited or idx in ring_set:
            continue
        visited.add(idx)
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx not in visited and nbr_idx not in ring_set:
                stack.append(nbr_idx)

    return frozenset(visited)


# ============================================================================
# Chain Substituent Extraction
# ============================================================================


def extract_chain_substituents(mol, principal_chain, substituents_dict):
    """Convert a substituents dict to a list of SubstituentInfo namedtuples.

    Takes the existing features.substituents dict (position -> list of
    atom index lists) and wraps each entry as a SubstituentInfo for
    unified downstream processing.

    Args:
        mol: RDKit Mol object.
        principal_chain: List of atom indices in the principal chain.
        substituents_dict: Dict mapping chain position (1-indexed) to list
            of substituent atom index lists.

    Returns:
        List of SubstituentInfo namedtuples.
    """
    if not substituents_dict:
        return []

    chain_set = set(principal_chain)
    results = []

    for position, sub_atom_lists in substituents_dict.items():
        for sub_atoms in sub_atom_lists:
            if not sub_atoms:
                continue

            # The first atom in sub_atoms is the one bonded to the chain
            attach_idx = sub_atoms[0] if isinstance(sub_atoms, (list, tuple)) else sub_atoms

            # Compute the actual atom index on the chain that this attaches to
            # position is 1-indexed into principal_chain
            if isinstance(position, int) and 1 <= position <= len(principal_chain):
                chain_atom_idx = principal_chain[position - 1]
            else:
                chain_atom_idx = attach_idx

            frag_atoms = frozenset(sub_atoms) if isinstance(sub_atoms, (list, tuple)) else frozenset([sub_atoms])

            results.append(SubstituentInfo(
                frag_mol=None,  # Chain substituents use atom indices, not frag mols
                locant=position,
                attach_mol_idx=chain_atom_idx,
                frag_atoms=frag_atoms,
            ))

    return results


# ============================================================================
# Fragment Classification and Naming
# ============================================================================


def classify_and_name_fragment(mol, frag_info, parent_atoms, features=None):
    """Classify a substituent fragment and produce its IUPAC prefix name.

    Routes each fragment to the appropriate naming function based on its
    composition:
      - fg_only: No carbon atoms (halogens, -OH, -NH2, -NO2, etc.)
      - pure_alkyl: Only carbon atoms (methyl, ethyl, etc.)
      - compound: Carbon + heteroatoms (trifluoromethyl, hydroxymethyl, etc.)

    a phase: Tier-0.5 prefix-form check runs FIRST so that any
    fragment matching the 14-row IUPAC / prefix-form table is
    named via the canonical prefix form (e.g., -C(=O)OCH3 -> methoxycarbonyl)
    before falling through to compound/pure_alkyl/fg_only classification.
    This eliminates the polyfunctional-path duplicate-name bug
    (hydroxymethyl + methoxycarbonyl on the same ester atoms) per
    RESEARCH root-cause fix.

    Args:
        mol: RDKit Mol object of the full molecule.
        frag_info: SubstituentInfo namedtuple for this substituent.
        parent_atoms: Set of atom indices in the parent structure.
        features: Optional features object (for additional context).

    Returns:
        IUPAC prefix name string (e.g., "methyl", "hydroxy", "trifluoromethyl"),
        or None if naming fails (with WARNING logged).
    """
    frag_mol = frag_info.frag_mol
    frag_atoms = frag_info.frag_atoms

    # ---- Tier 0.5 (a phase): IUPAC / prefix-form check ----
    # Pure read-only check. Applies to fragments that entirely contain one of
    # the 14 non-principal functional groups. The polyfunctional handler routes
    # substituent fragments here (via _name_compound_substituent fallback);
    # without this gate the compound-substituent path generates "hydroxymethyl"
    # for the methyl-ester fragment per RESEARCH bug trace.
    try:
        frag_atom_set = set(frag_atoms) if not isinstance(frag_atoms, set) else frag_atoms
        attach_idx = getattr(frag_info, "attach_mol_idx", None)
        if attach_idx is None and frag_atom_set:
            attach_idx = next(iter(frag_atom_set))
        prefix_form = _check_substituent_prefix_form(
            mol, frag_atom_set, attach_idx
        )
        if prefix_form is not None:
            return prefix_form
    except Exception:
        pass

    # task 9: ring-containing fragments go to the single
    # ring-substituent chokepoint free-valence locant:
    # naphthalen-2-yl, pyridin-2-yl,...) — composition-based naming below
    # would count a ring's carbons as a chain. Recursion-safe:
    # name_ring_system_substituent only uses get_ring_substituent_name and
    # the name_substituent cascade, never this router.
    try:
        _ri = mol.GetRingInfo()
        if attach_idx is not None and any(
                _ri.NumAtomRings(a) > 0 for a in frag_atom_set):
            # composition lever: under the best-effort tier, let the ring-
            # substituent chokepoint name a decorated (hetero)aryl branch that
            # name_substituent can only build with allow_mancude=True (e.g.
            # [4-(methanesulfonyl)phenyl]methyl). PIN default leaves
            # best_effort_ctx unset -> allow_mancude=False here -> byte-identical.
            # A malformed / non-round-tripping name is still suppressed by the
            # validity + rt_exact gate downstream, so 0-wrong holds.
            from ..metrics.provenance import best_effort_ctx
            from ..rules.ring_substituents import name_ring_system_substituent
            _ring_nm = name_ring_system_substituent(
                mol, sorted(frag_atom_set), attach_idx,
                allow_mancude=bool(best_effort_ctx.get()),
            )
            if _ring_nm:
                return _ring_nm
    except Exception:
        pass

    # Classify the fragment by composition
    category = _classify_fragment(mol, frag_mol, frag_atoms)

    if category == 'fg_only':
        return _name_fg_only(mol, frag_mol, frag_atoms)
    elif category == 'pure_alkyl':
        return _name_pure_alkyl(mol, frag_info, parent_atoms)
    elif category == 'compound':
        return _name_compound_substituent(mol, frag_info, parent_atoms)
    else:
        # Unknown -- log warning, never silently drop
        _frag_smiles = _get_frag_smiles(mol, frag_atoms)
        logger.warning(
            "Unrecognized substituent fragment at locant %s: %s",
            frag_info.locant, _frag_smiles
        )
        return None


def _classify_fragment(mol, frag_mol, frag_atoms):
    """Classify a fragment as fg_only, pure_alkyl, compound, or unknown.

    Uses the fragment mol if available (ring parent), otherwise uses
    original mol atom indices.

    Args:
        mol: RDKit Mol of the full molecule.
        frag_mol: RDKit Mol of the isolated fragment (may be None).
        frag_atoms: Set/frozenset of original mol atom indices.

    Returns:
        String category: 'fg_only', 'pure_alkyl', 'compound', or 'unknown'.
    """
    carbons = 0
    heteroatoms = 0

    if frag_mol is not None:
        # Use fragment mol (from ReplaceCore)
        for atom in frag_mol.GetAtoms():
            anum = atom.GetAtomicNum()
            if anum == 0:
                continue  # skip dummy atom
            if anum == 6:
                carbons += 1
            elif anum != 1:
                heteroatoms += 1
    else:
        # Use original mol indices
        for idx in frag_atoms:
            atom = mol.GetAtomWithIdx(idx)
            anum = atom.GetAtomicNum()
            if anum == 6:
                carbons += 1
            elif anum != 1:
                heteroatoms += 1

    if carbons == 0 and heteroatoms > 0:
        return 'fg_only'
    elif carbons > 0 and heteroatoms == 0:
        return 'pure_alkyl'
    elif carbons > 0 and heteroatoms > 0:
        return 'compound'
    else:
        return 'unknown'


def _name_fg_only(mol, frag_mol, frag_atoms):
    """Name a fragment that is a pure functional group (no carbon).

    Checks halogens by symbol, then common FG patterns.

    Args:
        mol: RDKit Mol of full molecule.
        frag_mol: RDKit Mol of isolated fragment (may be None).
        frag_atoms: Set of original mol atom indices.

    Returns:
        IUPAC prefix name string, or None if unrecognized.
    """
    # Collect non-dummy, non-hydrogen atoms
    if frag_mol is not None:
        atoms_info = []
        for atom in frag_mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                continue  # skip dummy
            atoms_info.append({
                'symbol': atom.GetSymbol(),
                'atomic_num': atom.GetAtomicNum(),
                'total_hs': atom.GetTotalNumHs(),
                'num_bonds': atom.GetDegree(),
            })
    else:
        atoms_info = []
        for idx in frag_atoms:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() == 1:
                continue
            atoms_info.append({
                'symbol': atom.GetSymbol(),
                'atomic_num': atom.GetAtomicNum(),
                'total_hs': atom.GetTotalNumHs(),
                'num_bonds': atom.GetDegree(),
            })

    if len(atoms_info) == 0:
        return None

    # Single atom cases
    if len(atoms_info) == 1:
        ai = atoms_info[0]
        sym = ai['symbol']

        # Halogens
        if sym in _HALOGEN_MAP:
            return _HALOGEN_MAP[sym]

        # -OH (oxygen with 1 H)
        if sym == 'O' and ai['total_hs'] >= 1:
            return 'hydroxy'

        # =O (oxo -- oxygen with no H, double bonded)
        if sym == 'O' and ai['total_hs'] == 0:
            return 'oxo'

        # -NH2 (nitrogen with 2 H)
        if sym == 'N' and ai['total_hs'] >= 2:
            return 'amino'

        # =NH (imino -- nitrogen with 1 H)
        if sym == 'N' and ai['total_hs'] == 1:
            return 'imino'

        # -SH (sulfanyl)
        if sym == 'S' and ai['total_hs'] >= 1:
            return 'sulfanyl'

        # =S (sulfanylidene)
        if sym == 'S' and ai['total_hs'] == 0:
            return 'sulfanylidene'

    # Multi-atom FG-only patterns
    if len(atoms_info) >= 2:
        symbols = sorted(ai['symbol'] for ai in atoms_info)

        # -NO2 (nitro): N + 2O
        if symbols == ['N', 'O', 'O']:
            return 'nitro'

        # -N3 (azido): 3 N atoms
        if symbols == ['N', 'N', 'N']:
            return 'azido'

    # Try seniority.get_prefix as fallback
    # For FG-only fragments on ring parents, try SMARTS matching
    if frag_mol is not None:
        from ..perception.functional_groups import FUNCTIONAL_GROUP_SMARTS
        for fg_name, smarts_str in FUNCTIONAL_GROUP_SMARTS.items():
            pattern = _compiled_smarts(smarts_str)
            if pattern and frag_mol.HasSubstructMatch(pattern):
                prefix = get_prefix(fg_name)
                if prefix:
                    return prefix

    # Last resort for single-atom halogens on original mol
    for idx in frag_atoms:
        atom = mol.GetAtomWithIdx(idx)
        sym = atom.GetSymbol()
        if sym in _HALOGEN_MAP:
            return _HALOGEN_MAP[sym]
        if sym == 'O' and atom.GetTotalNumHs() >= 1:
            return 'hydroxy'
        if sym == 'O' and atom.GetTotalNumHs() == 0:
            return 'oxo'
        if sym == 'N' and atom.GetTotalNumHs() >= 2:
            return 'amino'
        if sym == 'S' and atom.GetTotalNumHs() >= 1:
            return 'sulfanyl'

    logger.warning(
        "Could not name fg_only fragment with atoms: %s",
        [ai['symbol'] for ai in atoms_info]
    )
    return None


def _fragment_is_saturated_acyclic(mol, frag_atoms) -> bool:
    """True iff the fragment is a plain saturated acyclic carbon chain: no ring
    atom and every intra-fragment bond single. The carbon-COUNT fallback in
    _name_pure_alkyl is only honest for this shape -- an unsaturated or cyclic
    fragment counted by carbons alone silently drops its double bonds / ring
    (a wrong constitution; today masked by the 0-wrong backstop -> abstain).
    a review review / carotenoid trace 2026-09-09.
    """
    fset = set(frag_atoms)
    for idx in fset:
        if mol.GetAtomWithIdx(idx).IsInRing():
            return False
    for b in mol.GetBonds():
        a, z = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a in fset and z in fset and b.GetBondTypeAsDouble() != 1.0:
            return False
    return True


def _name_pure_alkyl(mol, frag_info, parent_atoms):
    """Name a pure alkyl substituent (carbon-only).

    Uses name_substituent_fragment for retained names (isopropyl, etc.)
    and recursive naming, with get_alkyl_name as fallback.

    Args:
        mol: RDKit Mol of full molecule.
        frag_info: SubstituentInfo namedtuple.
        parent_atoms: Set of parent atom indices.

    Returns:
        Alkyl prefix name string, or None.
    """
    frag_atoms = list(frag_info.frag_atoms)
    if not frag_atoms:
        return None

    # Find the attachment atom within the fragment
    attach_idx = _find_attach_atom_in_frag(mol, frag_atoms, parent_atoms)
    if attach_idx is None and frag_atoms:
        attach_idx = frag_atoms[0]

    # Delegate to existing naming infrastructure
    parent_list = list(parent_atoms) if parent_atoms else []
    name = name_substituent_fragment(mol, frag_atoms, attach_idx, parent_list)
    if name:
        return name

    # Fallback: count carbons -- ONLY for a plain saturated acyclic chain. A
    # fragment with any double/triple bond or ring atom must NOT be flattened by
    # carbon count (that drops the unsaturation/ring -> wrong constitution, e.g.
    # a carotenoid polyene named 'hentriacontyl'); fail closed so the recursive/
    # cascade namer or an abstention owns it (0-wrong).
    carbon_count = sum(
        1 for idx in frag_atoms
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
    )
    if carbon_count > 0 and _fragment_is_saturated_acyclic(mol, frag_atoms):
        try:
            return get_alkyl_name(carbon_count)
        except (ValueError, KeyError):
            pass

    return None


def _name_compound_substituent(mol, frag_info, parent_atoms):
    """Name a compound substituent (carbon + heteroatoms).

    Handles haloalkyl (trifluoromethyl), hydroxyalkyl, aminoalkyl,
    alkoxy, sulfanylalkyl, and other compound types.

    Args:
        mol: RDKit Mol of full molecule.
        frag_info: SubstituentInfo namedtuple.
        parent_atoms: Set of parent atom indices.

    Returns:
        Compound prefix name string, or None.
    """
    frag_atoms = list(frag_info.frag_atoms)
    if not frag_atoms:
        return None

    # Find the attachment atom
    attach_idx = _find_attach_atom_in_frag(mol, frag_atoms, parent_atoms)
    if attach_idx is None and frag_atoms:
        attach_idx = frag_atoms[0]

    frag_set_for_chalcogen = set(frag_atoms)

    # (generalizes SP2.1'): a PLAIN acyloxy ester '-O-C(=O)-R' with a SYSTEMATIC
    # acyl must be caught HERE, before the alkoxy branch below mis-reads the ester O
    # as a plain ether (measured: '4-(pentyloxy)benzoic acid' for the
    # -O-C(=O)-CH2-C(CH3)2-OH frag -- a wrong molecule that then suppresses,
    # so the whole molecule abstains). trace (a project rule) refuted the SP2.1'
    # 'OPSIN-invalid' premise (the oxa-chain the best-effort tier emits round-trips
    # exact); the real gap is a DEFAULT-tier abstain + best-effort ugly spelling.
    # Route it through the SAME recognizer name_substituent uses
    # (composer._acyloxy_prefix_for_frag -> rules.lipids._acyloxy_for_site -> the
    # full acid engine) for the '<acyl>oxy' PIN prefix. The recognizer
    # is HIGHLY specific and FAIL-CLOSED: the attach atom must be a bare ester O, the
    # frag-side carbon a carbonyl with exactly one terminal '=O' and <=1 all-carbon
    # R, the whole acyl side self-contained (else None) -- so it fires ONLY for a
    # plain acyloxy and never for an ether/peroxide/carbamate/carbonate. A RETAINED
    # acyloxy is intercepted upstream (Tier 0.5 prefix-form in
    # classify_and_name_fragment) and never reaches here, so retained spellings are
    # BYTE-IDENTICAL; the caller's OPSIN round-trip is the 0-wrong net.
    if attach_idx is not None:
        try:
            from .composer import _acyloxy_prefix_for_frag
            _acyloxy_early = _acyloxy_prefix_for_frag(mol, frag_atoms, attach_idx)
        except Exception:  # a producer helper must never crash branch naming
            _acyloxy_early = None
        if _acyloxy_early and ' ' not in _acyloxy_early:
            return _acyloxy_early

    # DD2 Fix B (Phase D, (1)): peroxy branch -O-O-R -> (R)peroxy.
    # Checked BEFORE the alkoxy branch because both attach through a divalent O;
    # only the peroxide case has a second O on the far side of the attach O.
    if attach_idx is not None:
        attach_atom = mol.GetAtomWithIdx(attach_idx)
        if attach_atom.GetSymbol() == 'O' and attach_atom.GetDegree() == 2 and any(
            n.GetSymbol() == 'O' and n.GetIdx() in frag_set_for_chalcogen
            and n.GetIdx() not in parent_atoms
            for n in attach_atom.GetNeighbors()
        ):
            name = _name_peroxy_branch(mol, frag_atoms, attach_idx, parent_atoms)
            if name:
                return name

    # DD2 Fix B (Phase D): disulfanyl branch -S-S-R -> (R)disulfanyl, before the
    # thioether (sulfanyl) branch (both attach through a divalent S).
    if attach_idx is not None:
        attach_atom = mol.GetAtomWithIdx(attach_idx)
        if attach_atom.GetSymbol() == 'S' and any(
            n.GetSymbol() == 'S' and n.GetIdx() in frag_set_for_chalcogen
            and n.GetIdx() not in parent_atoms
            for n in attach_atom.GetNeighbors()
        ):
            name = _name_disulfanyl_branch(mol, frag_atoms, attach_idx, parent_atoms)
            if name:
                return name

    # Wave2: MIXED divalent-chalcogen bridge — must be decided
    # BEFORE the alkoxy/sulfanyl branches (an -O-S-R attach O would otherwise
    # be mis-read as a plain alkoxy with a mangled R). Terminal decline on a
    # detected bridge the namer can't express; never fall through.
    if attach_idx is not None and _is_mixed_chalcogen_bridge_attach(
            mol, attach_idx, frag_set_for_chalcogen):
        return _name_mixed_chalcogen_branch(
            mol, frag_atoms, attach_idx, parent_atoms)

    # Special case: O-attached branches (ether substituents)
    # -O-R -> "alkoxy" (e.g., methoxy, ethoxy, phenoxy)
    if attach_idx is not None:
        attach_atom = mol.GetAtomWithIdx(attach_idx)
        if attach_atom.GetSymbol() == 'O' and attach_atom.GetDegree() == 2:
            name = _name_alkoxy_branch(mol, frag_atoms, attach_idx, parent_atoms)
            if name:
                return name

    # Special case: S-attached branches (thioether substituents)
    # -S-R -> "alkylsulfanyl" (e.g., methylsulfanyl, ethylsulfanyl)
    if attach_idx is not None:
        attach_atom = mol.GetAtomWithIdx(attach_idx)
        if attach_atom.GetSymbol() == 'S' and attach_atom.GetTotalNumHs() == 0:
            name = _name_sulfanyl_branch(mol, frag_atoms, attach_idx, parent_atoms)
            if name:
                return name

    # Special case: N-attached branches (amino substituents)
    # -NH-R -> "alkylamino" (e.g., methylamino, phenylamino/anilino)
    # -NH-C(=O)-R -> "acylamino" (e.g., acetylamino, benzoylamino)
    if attach_idx is not None:
        attach_atom = mol.GetAtomWithIdx(attach_idx)
        if attach_atom.GetSymbol() == 'N':
            name = _name_amino_branch(mol, frag_atoms, attach_idx, parent_atoms)
            if name:
                return name

    # Delegate to existing naming infrastructure
    parent_list = list(parent_atoms) if parent_atoms else []
    name = name_substituent_fragment(mol, frag_atoms, attach_idx, parent_list)
    if name:
        # Reject garbled names from S/P-attached branches where
        # name_substituent_fragment doesn't handle the root heteroatom
        if attach_idx is not None:
            attach_sym = mol.GetAtomWithIdx(attach_idx).GetSymbol()
            if attach_sym in ('S', 'P') and 'thiyl' in name:
                pass  # fall through to warning
            else:
                return name

    # a phase-b, 0-wrong): a fragment with a REAL net formal
    # charge (a genuine onium cation on the branch, e.g. a choline/
    # trimethylammonium head), NOT an internal charge-separated pair that
    # cancels within the same fragment (nitro/N-oxide/azide/diazo net to 0
    # and are untouched by this guard). The recursive fallback below is
    # CHARGE-BLIND: name_fragment_recursively names the isolated fragment as
    # a whole molecule carrying the ionic '-ium' suffix, then parent_to_prefix
    # does string-level suffix surgery that either produces the
    # OPSIN-unparseable '-aminiumyl' ending or silently reinterprets the
    # attachment bond as an extra substituent (mirrors the sibling fix in
    # substituent_naming.py::name_substituent_fragment Step 2e). Route the
    # DIRECT-ATTACHMENT shape (cation IS attach_idx, parent one bond away)
    # through the existing structured cation_to_prefix primitive,
    # rules/ions.py:4305 / charged_router.py:478); any other charged shape
    # fails closed here (falls to the final decline below) rather than
    # guessing. Neutral fragments (net charge 0) are unchanged -- no-op.
    if sum(mol.GetAtomWithIdx(a).GetFormalCharge() for a in frag_atoms) != 0:
        from .substituent_naming import cation_to_prefix
        if attach_idx is not None:
            _cat_atom = mol.GetAtomWithIdx(attach_idx)
            if _cat_atom.GetFormalCharge() > 0:
                _parent_nbrs = [n.GetIdx() for n in _cat_atom.GetNeighbors()
                                if n.GetIdx() in parent_atoms]
                if len(_parent_nbrs) == 1:
                    _cp = cation_to_prefix(mol, attach_idx, _parent_nbrs[0])
                    if _cp:
                        return _cp
        logger.debug(
            "charged_fragment_not_directly_nameable substituent_skip: reason=charged_fragment_not_direct_"
            "cation_attach atom_count=%d", len(frag_atoms),
        )
        return None

    # Fallback: recursive naming for ring-containing compound fragments.
    # Use name_fragment_recursively which has cycle detection via visited set.
    # This handles cases where the fragment is a ring system with heteroatoms
    # that the simpler naming paths above cannot handle
    # (n_branch_ring_substituent_unnameable / c_branch_ring_substituent_unnameable /
    # ring_fragment_declined_by_ring_engine / amine_n_substituent_unnameable).
    #
    #: the old ``if len(frag_atoms) <= 25`` size cap here dropped a
    # legitimately nameable large fragment on size alone (the 27-heavy-atom
    # disaccharide chain hung off a steroid aglycone; SP4 witnesses g1/g3).
    # Size is NOT the right guard: nameability is. Attempt the recursive name for
    # any compound fragment and let the RT gate (whole-molecule /OPSIN
    # validity at the top level, and name_compound's own fragment-level check)
    # decide — a fragment whose name does not round-trip is refused and the whole
    # molecule abstains, so 0-wrong is preserved by the gate, not by a constant.
    # Cost is bounded WITHOUT a size cap: name_fragment_recursively charges the
    # per-top-level work budget (``spend_fragment_work``, the giant-molecule
    # hang fix) and enforces the ``_MAX_VISITED_SIZE`` depth net, so an
    # unbounded/expensive fragment abstains fast rather than dropping cheaply.
    frag_smiles = _get_frag_smiles(mol, frag_atoms)
    if frag_smiles and frag_smiles != "unknown":
        try:
            from .fragment_naming import name_fragment_recursively
            from .substituent_naming import ATTACH_LOCANT_UNKNOWN, parent_to_prefix
            frag_name = name_fragment_recursively(frag_smiles)
            if frag_name:
                # Convert parent name to prefix form (e.g., "benzoic acid" -> not useful,
                # but "pyridine" -> "pyridinyl", "cyclohexanone" -> "oxocyclohexyl")
                carbon_count = sum(
                    1 for idx in frag_atoms
                    if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
                )
                prefix_name = parent_to_prefix(
                    frag_name, chain_length=carbon_count,
                    attach_locant=ATTACH_LOCANT_UNKNOWN)
                if prefix_name:
                    logger.debug(
                        "ring_substituent_recursive_naming_fallback: recursive naming for %s -> %s",
                        frag_smiles, prefix_name,
                    )
                    return prefix_name
        except Exception:
            pass  # Keep falling through to warning

    # If naming infrastructure couldn't handle it, log warning
    frag_smiles = _get_frag_smiles(mol, frag_atoms)
    logger.warning(
        "Could not name compound substituent at locant %s: %s",
        frag_info.locant, frag_smiles
    )
    return None


def _name_alkoxy_branch(mol, frag_atoms, attach_idx, parent_atoms):
    """Name an alkoxy-attached branch: -O-R -> alkoxy.

    IUPAC: Ether substituents named as alkoxy when the oxygen
    is the attachment point to the parent. Examples:
      -O-CH3 -> methoxy
      -O-C2H5 -> ethoxy
      -O-phenyl -> phenoxy
      -O-CH2-phenyl -> benzyloxy

    Args:
        mol: RDKit Mol.
        frag_atoms: List of atom indices in the fragment.
        attach_idx: Atom index of the O attachment atom.
        parent_atoms: Set of parent atom indices.

    Returns:
        Alkoxy prefix name, or None if not a simple case.
    """
    from ..data.chain_names import get_chain_prefix

    frag_set = set(frag_atoms)
    o_atom = mol.GetAtomWithIdx(attach_idx)

    # Find non-parent neighbor of O (the R group)
    alkyl_start = None
    for nbr in o_atom.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx in parent_atoms:
            continue
        if nbr_idx in frag_set:
            alkyl_start = nbr_idx
            break

    if alkyl_start is None:
        # Bare -O- with no R group (shouldn't happen for compound substituent)
        return None

    alkyl_atom = mol.GetAtomWithIdx(alkyl_start)

    # Case A: O -> aromatic C in 6-membered all-carbon ring -> "phenoxy"
    if alkyl_atom.GetIsAromatic():
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            if alkyl_start in ring and len(ring) == 6:
                if all(mol.GetAtomWithIdx(r).GetIsAromatic()
                       and mol.GetAtomWithIdx(r).GetSymbol() == 'C'
                       for r in ring):
                    return "phenoxy"
        return "phenoxy"

    # Case B: O -> CH(aryl)n -> benzyloxy (1 aryl) / diphenylmethoxy (2 phenyl).
    # (a phase): single shared aryl-count helper (was inline benzyloxy here).
    _aryl_ether = _name_aryl_methyl_ether(mol, alkyl_start, attach_idx)
    if _aryl_ether is not None:
        return _aryl_ether

    # Case C: O -> simple alkyl chain -> "methoxy", "ethoxy", etc.
    # Count carbons in the alkyl part (BFS from alkyl_start excluding O)
    visited = set()
    stack = [alkyl_start]
    carbon_count = 0
    has_heteroatom = False
    has_ring = False
    ring_info = mol.GetRingInfo()

    while stack:
        idx = stack.pop()
        if idx in visited or idx == attach_idx:
            continue
        if idx not in frag_set:
            continue
        visited.add(idx)
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'C':
            carbon_count += 1
        elif atom.GetAtomicNum() != 1:
            has_heteroatom = True
        if ring_info.NumAtomRings(idx) > 0:
            has_ring = True
        for n in atom.GetNeighbors():
            if n.GetIdx() not in visited and n.GetIdx() != attach_idx:
                stack.append(n.GetIdx())

    # Only name as alkoxy if the R group is a pure alkyl chain (no heteroatoms, no rings)
    if has_heteroatom or has_ring or carbon_count == 0:
        return None

    # +: branched alkyl groups must use (alkan-n-yl)oxy form
    # with enclosing marks, NOT the retained n-alkyl names (propoxy, butoxy, etc.).
    # A secondary or tertiary alkyl group attaches to O via a non-terminal carbon:
    # the attachment carbon (alkyl_start) has >= 2 carbon neighbours within the
    # fragment. Straight-chain alkyls always attach via a terminal carbon (1 C
    # neighbour in the fragment). This correctly distinguishes isopropyl (2 C-nbrs
    # at attach) from n-propyl (1 C-nbr at attach), without false positives for the
    # middle carbon of n-propyl (which has 2 C-nbrs but is NOT the attachment point).
    # The call to name_substituent_fragment is safe here because alkyl_sub_atoms
    # excludes attach_idx (the ether O), so the recursed fragment has no O and
    # _name_alkoxy_branch's O-check will not trigger inside that call.
    frag_carbons = {idx for idx in visited if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'}
    attach_c_nbrs_in_frag = [
        n.GetIdx() for n in mol.GetAtomWithIdx(alkyl_start).GetNeighbors()
        if n.GetIdx() in frag_carbons
    ]
    is_branched = len(attach_c_nbrs_in_frag) >= 2

    if is_branched:
        # Build the alkyl sub-fragment (all fragment atoms except the ether O)
        alkyl_sub_atoms = [a for a in frag_atoms if a != attach_idx]
        alkyl_name = name_substituent_fragment(
            mol, alkyl_sub_atoms, alkyl_start, list(parent_atoms) + [attach_idx]
        )
        if alkyl_name and alkyl_name.endswith("yl"):
            #: the oxy suffix is appended to the alkyl name, then the
            # whole compound substituent name is wrapped in enclosing marks.
            # Correct: (propan-2-yloxy), NOT (propan-2-yl)oxy.
            return f"({alkyl_name[:-2]}yloxy)"
        # name_substituent_fragment failed — fall through to ALKOXY_NAMES approximation

    # TODO: _check_for_alkoxy (composer.py) and get_alkoxy_prefix
    # (substituent_prefix_forms.py) share the same ALKOXY_NAMES table and have the
    # same branching bug; fix them too if those paths are ever extended to pure-ether
    # whole-molecule naming.
    ALKOXY_NAMES = {
        1: "methoxy", 2: "ethoxy", 3: "propoxy", 4: "butoxy",
        5: "pentyloxy", 6: "hexyloxy", 7: "heptyloxy", 8: "octyloxy",
        9: "nonyloxy", 10: "decyloxy",
    }

    if carbon_count in ALKOXY_NAMES:
        return ALKOXY_NAMES[carbon_count]
    elif carbon_count > 10:
        return get_chain_prefix(carbon_count) + "yloxy"
    return None


def _name_amino_ring_branch(mol, frag_set, root_idx, parent_set):
    """Best-effort: a MONO- or DI-substituted N carrying a RING-bearing R ->
    '(R-yl)amino' / 'R1(R2)amino' (cyclohexylamino, benzylamino,
    [(4-fluorophenyl)methyl]amino, (4-hydroxycyclohexyl)amino; and disubstituted:
    cyclohexyl(methyl)amino, dicyclohexylamino, benzyl(cyclohexyl)amino).

    ``_name_amino_branch`` names alkyl/aryl/acyl R but declines a saturated-ring or
    ring-on-chain R, so a secondary amine whose R contains a ring fell through to the
    ugly replacement name. This mirrors the O-rooted ``get_alkoxy_prefix`` path (which
    already recurses ring R -> ``cyclohexyloxy``) by recursing ``name_substituent`` on
    R and wrapping the connective 'amino'.

    Reached only under the best-effort tier (the caller gates on ``allow_mancude``), so
    the PIN default is byte-identical. v1 scope: exactly ONE fragment-side branch on a
    neutral, single-bonded N; the branch CONTAINS a ring and is nameable by the
    recursion; the whole fragment is covered by N + the branch. Fail closed (None) on a
    disubstituted N, an acyl branch (owned by ``_name_amino_branch``), a charged/
    multivalent N, an unnameable R, or a coverage gap -> never a wrong or partial name.
    """
    root = mol.GetAtomWithIdx(root_idx)
    if (root.GetSymbol() != 'N' or root.GetFormalCharge() != 0
            or root.GetNumRadicalElectrons() != 0):
        return None
    if any(b.GetBondTypeAsDouble() != 1.0 for b in root.GetBonds()):
        return None
    branches = [n.GetIdx() for n in root.GetNeighbors()
                if n.GetIdx() in frag_set and n.GetIdx() not in parent_set]
    if len(branches) not in (1, 2):
        return None  # -NH-R (mono) or -N(R)(R') (di); v2 stops at two branches
    ring_info = mol.GetRingInfo()
    # Collect each branch's fragment atoms (boundary = the root N).
    branch_atomsets = []
    all_batoms = set()
    for b in branches:
        batoms = set()
        stack = [b]
        while stack:
            a = stack.pop()
            if a in batoms:
                continue
            batoms.add(a)
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                ni = nb.GetIdx()
                if ni != root_idx and ni in frag_set and ni not in batoms:
                    stack.append(ni)
        branch_atomsets.append(batoms)
        all_batoms |= batoms
    # Coverage: root + all branches must account for every heavy fragment atom.
    heavy = {a for a in frag_set if mol.GetAtomWithIdx(a).GetAtomicNum() > 1}
    if ({root_idx} | all_batoms) & heavy != heavy:
        return None
    # At least ONE branch must contain a ring — otherwise this is the pure-alkyl
    # amino case _name_amino_branch already owns (it declined for another reason,
    # and this ring extension must not silently re-claim it).
    if not any(any(ring_info.NumAtomRings(a) > 0 for a in bs)
               for bs in branch_atomsets):
        return None

    from rdkit import Chem as _Chem

    from ..rules.benzene import _reanchor_name_to_mol
    from .naming_utils import apply_enclosing_marks, is_complex_substituent

    names = []
    for b, batoms in zip(branches, branch_atomsets):
        battach = mol.GetAtomWithIdx(b)
        # A HETEROATOM-rooted branch on the amino N is not a simple amino
        # substituent — it is hydrazine (-N-N-), hydroxylamine (-N-O-), nitroso
        # (-N=O) etc., a different retained nomenclature class. name_substituent
        # names such a branch with an OPSIN-lenient but INVALID replacement string
        # ('2-oxa-1-azaeth-1-en-1-yl' for -N=O; a review RISK 5) that the per-branch
        # re-anchor accepts (constitution-correct). Require each R to be
        # carbon-rooted so those classes fall through to their own producers.
        if battach.GetSymbol() != 'C':
            return None
        # An acyl branch (-N-C(=O)-R) is the amido family, owned by
        # _name_amino_branch — never build 'amino' over it.
        if battach.GetSymbol() == 'C' and any(
                nb.GetSymbol() == 'O'
                and mol.GetBondBetweenAtoms(
                    b, nb.GetIdx()).GetBondTypeAsDouble() == 2.0
                for nb in battach.GetNeighbors()):
            return None
        ryl = name_substituent(mol, sorted(batoms), b, allow_mancude=True)
        if (not ryl or ryl == 'substituent' or ' ' in ryl
                or 'unknown' in ryl.lower()):
            return None
        # Gate-INDEPENDENT honesty, applied PER BRANCH (8afa533c guard-2 / F-B
        # _reanchor precedent): re-anchor a probe amine H2N-R, named
        # "{cited}amine", against the standalone R-amine molecule. A genuine ring
        # substituent round-trips (cyclohexyl -> cyclohexylamine == cyclohexanamine);
        # a mis-numbered or yl-less ring-yl does not. Fail closed on mismatch /
        # no-parse / NO-JAR so the producer stays honest even with the gate off.
        cited = (apply_enclosing_marks(ryl, -1)
                 if is_complex_substituent(ryl) else ryl)
        try:
            _probe_smi = _Chem.MolFragmentToSmiles(
                mol, sorted({root_idx} | batoms))
            _probe_mol = _Chem.MolFromSmiles(_probe_smi) if _probe_smi else None
        except Exception:  # noqa: BLE001 - a malformed probe is a decline
            _probe_mol = None
        if (_probe_mol is None
                or _reanchor_name_to_mol(_probe_mol, f'{cited}amine') is None):
            return None
        names.append(ryl)

    if len(names) == 1:
        inner = (apply_enclosing_marks(names[0], -1)
                 if is_complex_substituent(names[0]) else names[0])
        return f'{inner}amino'
    # Disubstituted N: hand BOTH R names to the ONE amino-prefix assembler
    # ordering + marking + di/bis multiplicity):
    # cyclohexyl(methyl)amino, dicyclohexylamino, benzyl(cyclohexyl)amino.
    return _assemble_amino_prefix_core([(nm, False) for nm in names])


def _name_thio_ring_branch(mol, frag_set, root_idx):
    """Best-effort: a divalent -S-R with a RING-bearing R -> '(R-yl)sulfanyl'
    (cyclohexylsulfanyl, benzylsulfanyl, (4-hydroxycyclohexyl)sulfanyl, phenylsulfanyl).

    The chalcogen-rooted sulfanyl cascade tier declines a saturated-ring /
    ring-on-chain R, so a thioether whose R contains a ring fell through to the
    ugly replacement name. Mirrors the O-rooted alkoxy (`cyclohexyloxy`) and
    N-rooted amino (`cyclohexylamino`) ring paths: recurse `name_substituent` on R
    and wrap the connective 'sulfanyl'. Gated best-effort (caller checks
    allow_mancude) -> PIN byte-identical. Fail closed (None) on a disubstituted /
    higher-valence S, an R with no ring, an unnameable R, a coverage gap, or a
    gate-independent re-anchor mismatch (the yl-less ring-assembly F1 class).
    """
    root = mol.GetAtomWithIdx(root_idx)
    branches = [n.GetIdx() for n in root.GetNeighbors() if n.GetIdx() in frag_set]
    if len(branches) != 1:
        return None  # -S- must bridge exactly one in-fragment R and the parent
    b = branches[0]
    batoms = set()
    stack = [b]
    while stack:
        a = stack.pop()
        if a in batoms:
            continue
        batoms.add(a)
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            ni = nb.GetIdx()
            if ni != root_idx and ni in frag_set and ni not in batoms:
                stack.append(ni)
    heavy = {a for a in frag_set if mol.GetAtomWithIdx(a).GetAtomicNum() > 1}
    if ({root_idx} | batoms) & heavy != heavy:
        return None
    if not any(mol.GetRingInfo().NumAtomRings(a) > 0 for a in batoms):
        return None  # no ring -> the cascade's simple-sulfanyl tier owns it
    ryl = name_substituent(mol, sorted(batoms), b, allow_mancude=True)
    if (not ryl or ryl == 'substituent' or ' ' in ryl
            or 'unknown' in ryl.lower()):
        return None
    from .naming_utils import apply_enclosing_marks, is_complex_substituent
    inner = (apply_enclosing_marks(ryl, -1)
             if is_complex_substituent(ryl) else ryl)
    # Gate-INDEPENDENT honesty (the ed52fa98 / 8afa533c precedent): name_substituent
    # can return a yl-LESS PARENT-HYDRIDE for a ring ASSEMBLY (biphenyl ->
    # "1,1'-biphenyl") that the token checks miss; wrapping it ships an
    # OPSIN-unparseable name. Re-anchor a probe thioether CH3-S-R named
    # "({inner}sulfanyl)methane" against the standalone CH3-S-R molecule; a genuine
    # ring substituent round-trips, the parent hydride does not. Fail closed on
    # mismatch / no-parse / NO-JAR so the producer stays honest gate-off.
    from rdkit import Chem as _Chem
    try:
        _rw = _Chem.RWMol(mol)
        # Detach the S from its PARENT-side neighbour (the atom outside the
        # fragment) before adding the probe methyl. Without this the S keeps
        # three bonds in the RWMol; MolFragmentToSmiles then cuts the excluded
        # parent bond and RDKit fills the freed valence with an H, yielding a
        # TRIVALENT `C[SH]...` probe (a thiol-like species the divalent-S name
        # can never re-anchor to). Removing it first leaves a clean divalent
        # thioether CH3-S-R. (Only reached under allow_mancude -> PIN-safe.)
        for _pn in [n.GetIdx() for n in mol.GetAtomWithIdx(root_idx).GetNeighbors()
                    if n.GetIdx() not in frag_set]:
            _rw.RemoveBond(root_idx, _pn)
        _newc = _rw.AddAtom(_Chem.Atom(6))
        _rw.AddBond(root_idx, _newc, _Chem.BondType.SINGLE)
        _probe_smi = _Chem.MolFragmentToSmiles(
            _rw.GetMol(), sorted(batoms | {root_idx, _newc}))
        _probe_mol = _Chem.MolFromSmiles(_probe_smi) if _probe_smi else None
    except Exception:  # noqa: BLE001 - a malformed probe is a decline, not a crash
        _probe_mol = None
    if _probe_mol is None:
        return None
    from ..rules.benzene import _reanchor_name_to_mol
    if _reanchor_name_to_mol(_probe_mol, f'({inner}sulfanyl)methane') is None:
        return None
    return f'{inner}sulfanyl'


def _name_amino_branch(mol, frag_atoms, attach_idx, parent_atoms):
    """Name an N-attached branch: -NH-R -> alkylamino, phenylamino/anilino.

    IUPAC: Amine substituents named as amino when nitrogen is the
    attachment point. Also handles acylamino (-NH-C(=O)-R).

    Key patterns:
      -NH2 -> amino (handled as fg_only, not here)
      -NH-CH3 -> methylamino
      -NH-phenyl -> anilino (retained name for phenylamino)
      -NH-C(=O)-R -> amido family (acetamido, hexanamido;
                     method (1) = PIN), legacy acylamino fallback otherwise
      -N(CH3)2 -> dimethylamino

    Args:
        mol: RDKit Mol.
        frag_atoms: List of atom indices in the fragment.
        attach_idx: Atom index of the N attachment atom.
        parent_atoms: Set of parent atom indices.

    Returns:
        Amino prefix name, or None if pattern not recognized.
    """
    from ..data.chain_names import get_chain_prefix

    frag_set = set(frag_atoms)
    n_atom = mol.GetAtomWithIdx(attach_idx)

    # Collect non-parent, non-N neighbors
    branches = []
    for nbr in n_atom.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx in parent_atoms:
            continue
        if nbr_idx in frag_set:
            branches.append(nbr_idx)

    if not branches:
        return None

    ring_info = mol.GetRingInfo()

    # Check for acylamino: -NH-C(=O)-R
    for branch_start in branches:
        branch_atom = mol.GetAtomWithIdx(branch_start)
        if branch_atom.GetSymbol() != 'C':
            continue
        # Check if this C has a =O (carbonyl)
        has_carbonyl = False
        for nbr in branch_atom.GetNeighbors():
            if nbr.GetIdx() == attach_idx:
                continue
            if nbr.GetSymbol() == 'O':
                bond = mol.GetBondBetweenAtoms(branch_start, nbr.GetIdx())
                if bond and bond.GetBondTypeAsDouble() == 2.0:
                    has_carbonyl = True
                    break
        if has_carbonyl:
            # Wave2: method (1) amido prefix is the PIN
            # (formamido/acetamido/{stem}anamido). The strict builder emits
            # only for an exact -NH-CO-(unbranched saturated carbon chain)
            # branch with full atom coverage; anything else keeps the legacy
            # method-(2) fallback below.
            from .substituent_naming import linear_acyl_amido_prefix
            amido_name = linear_acyl_amido_prefix(
                mol, branch_start, attach_idx, frag_atoms
            )
            if amido_name:
                return amido_name

            # F-amido method (1), the Blue Book): the strict builder
            # above refuses a SUBSTITUTED amide N (`-N(R')-C(=O)-R`), and the legacy
            # count fallback below refuses a non-mono-substituted N (37cd122d F1), so
            # `-N(CH3)C(=O)CH3` dropped to the ugly general replacement name. Build the
            # `{N-R'}{acyl}amido` PIN prefix (N-methylacetamido / N-methylformamido).
            # Fail-closed on a branched/unsaturated/ring/hetero acyl or an unnameable
            # R'. PIN only as a prefix -- reached only when parent selection already
            # made the amide a prefix (the amide-as-principal case is the suffix path).
            from .substituent_naming import n_substituted_acyl_amido_prefix
            _n_amido = n_substituted_acyl_amido_prefix(
                mol, attach_idx, frag_atoms, parent_atoms
            )
            if _n_amido:
                return _n_amido

            # C4d: an acyl rooted on a CHALCOGEN -- the carbamate
            # class R-X-CO-NH- covering every alkoxycarbonyl -- has no
            # hydrocarbyl acyl name. The count below cannot express it and,
            # worse, WALKS ACROSS the heteroatom (it traverses every fragment
            # atom and counts the carbons), so it spelled CH3-S-CO-NH- as
            # 'ethanoylamino' and tBuO-CO-NH- as a 5-carbon acyl. Route to the
            # ONE shared primitive; never fall through once the class is
            # claimed.
            _cg = chalcogen_rooted_acyl_amino_core(
                mol, branch_start, attach_idx, frag_atoms)
            if _cg is ACYL_CHALCOGEN_UNNAMEABLE:
                return None
            if _cg is not None:
                return _cg

            # C4d soundness gate. What follows spells a COUNT as an unbranched
            # saturated stem, so it is faithful only when the acyl really is an
            # unbranched, acyclic, saturated, all-carbon chain. Without it,
            # acryloyl became 'propanoylamino' (saturated) and isobutyryl
            # 'butanoylamino' (n-butyryl) -- constitutionally DIFFERENT acyls.
            # The carbonyl's own =O must be excluded, or the predicate reads it
            # as a hetero decoration and refuses every acyl.
            #
            # Coverage guard (a review review of 9bb3a532): the count fallback below
            # spells ONLY this one acyl branch as `{stem}anoylamino`, so if the N
            # carries a SECOND substituent (`-N(CH3)C(=O)R`, or the diacyl imide
            # `-N(C(=O)R)2`) that substituent is silently DROPPED -> a different
            # molecule (`ethanoylamino` for `-N(CH3)COCH3`, the whole N-methyl
            # gone). The strict sibling `linear_acyl_amido_prefix` already proves
            # full coverage; this legacy count path did not. Fail closed unless
            # the N is mono-substituted (the sole heavy branch is this acyl C).
            if len(branches) != 1:
                return None
            _co = next(
                (nb.GetIdx() for nb in branch_atom.GetNeighbors()
                 if nb.GetSymbol() == 'O'
                 and mol.GetBondBetweenAtoms(
                     branch_start, nb.GetIdx()).GetBondTypeAsDouble() == 2.0),
                None,
            )
            if _co is None:
                return None
            from .composer import _pure_linear_alkyl_len
            _n = _pure_linear_alkyl_len(
                mol, branch_start, set(parent_atoms) | {attach_idx, _co})
            if _n is None:
                return None
            # The count includes the carbonyl C, which is the stem's C1.
            return get_chain_prefix(_n) + "anoylamino"

    # (the Blue Book): -NH-SO2-R -> '{R}sulfonamido' (methanesulfonamido
    # / benzenesulfonamido / cyclohexanesulfonamido). The sulfonyl S is not a
    # carbon, so the acylamino loop above skips it; without this branch the
    # cascade re-roots the fragment as `carbamoyl` (swaps S->C, drops S, the two
    # =O and R -- a different molecule). Shared primitive; fails closed (None)
    # for substituted-arene / CF3 / N,N-disubstituted R (those keep abstaining).
    from .substituent_naming import sulfonamido_prefix_from_n_branch
    _sulfonamido = sulfonamido_prefix_from_n_branch(
        mol, attach_idx, frag_atoms, parent_atoms)
    if _sulfonamido:
        return _sulfonamido

    # Check for anilino: -NH-phenyl (isolated benzene ring directly on N).
    # (the Blue Book) 'anilino' is the retained PREFERRED PREFIX for
    # C6H5-NH- WITH FULL SUBSTITUTION ALLOWED (the Blue Book '4-chloroanilino
    # (preferred prefix) | (4-chlorophenyl)amino'). This was the THIRD copy of the
    # unguarded `return "anilino"`: the ring test never looked at the ring's own
    # substituents, so OCCCCNc1ccc(Cl)cc1 was named '4-anilinobutan-1-ol' — the
    # chloro dropped, a different molecule. Routed through the shared primitive,
    # which proves atom coverage and otherwise fails closed.
    # enclose=False: this producer's contract is to return the BARE prefix core and
    # let the caller apply the marks (its sibling returns are bare too,
    # e.g. `get_chain_prefix(_n) + "anoylamino"`). Returning a pre-enclosed prefix
    # here double-wrapped it — '4-ethylcyclohexan-1-yl[(4-ethylanilino)]methanethioic
    # O-acid' for CCC1CCC(CC1)NC(=S)NC2=CC=C(C=C2)CC.
    from ..rules.ring_substituents import anilino_prefix_from_n_branch
    _anilino = anilino_prefix_from_n_branch(
        mol, attach_idx, frag_atoms, enclose=False)
    if _anilino is not None:
        return _anilino

    # Simple amino: -NH-alkyl or -N(alkyl)2
    if len(branches) == 1:
        # BFS from branch to count carbons
        visited = set()
        stack_c = [branches[0]]
        carbon_count = 0
        has_hetero = False
        has_ring = False
        while stack_c:
            idx = stack_c.pop()
            if idx in visited or idx == attach_idx:
                continue
            if idx not in frag_set:
                continue
            visited.add(idx)
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == 'C':
                carbon_count += 1
            elif atom.GetAtomicNum() != 1:
                has_hetero = True
            if ring_info.NumAtomRings(idx) > 0:
                has_ring = True
            for n in atom.GetNeighbors():
                if n.GetIdx() not in visited and n.GetIdx() != attach_idx:
                    stack_c.append(n.GetIdx())
        if carbon_count > 0 and not has_ring:
            #: the branch's bond to the nitrogen may be DOUBLE
            # (-N=CH-CH3), where the prefix is 'ethylideneamino'. The carbon
            # walk above crosses that bond without noticing it, so 'CC=NCC(=O)O'
            # was named (ethylamino)iminoacetic acid -- a different molecule.
            _fv = carbon_free_valence_prefix(mol, visited, branches[0])
            if _fv.prefix is not None:
                return f"{_fv.prefix}amino"
            if _fv.must_fail_closed:
                return None
            # NEVER get_alkyl_name(carbon_count) here. A COUNT named butyl,
            # 2-methylpropyl, butan-2-yl and tert-butyl all 'butyl', so three of
            # every four C4H9 amino prefixes described a molecule other than the
            # one drawn -- 'CC(C)NCC(=O)N' -> '2-(propylamino)acetamide' and
            # 'CC(C)(C)NCC(=O)N' -> '2-(butylamino)acetamide', both a different
            # constitution. This was the THIRD carbon-count amino site;
            # composer._composed_amino_branch_name's docstring converted the
            # other two and this one was missed. Route through the SAME
            # constitution-perceiving primitive the acid path uses, then through
            # the ONE assembler, so the amide and the acid cannot disagree.
            #
            # a phase (E3 Task 4): `has_hetero` used to gate this call OFF
            # entirely, so a branch carrying its own heteroatom decoration
            # (amino, carboxy, hydroxy,...) beyond the attachment N -- e.g.
            # saccharopine's N-(5-amino-5-carboxypentyl) arm -- fell straight
            # through to `return None` and the whole molecule was dropped
            # (suppressed the atom-incomplete parent-only candidate).
            # `composed_prefix_organyl_name` already routes through
            # `name_substituent`, the SAME cascade that names '2-carboxyethyl'
            # and '5-amino-5-carboxypentyl' as compound alkyl prefixes
            # elsewhere in this tree, and it fails closed (None) on anything
            # it cannot prove -- so dropping the heteroatom restriction only
            # ADDS coverage; a pure-alkyl branch is unaffected (byte-identical
            # path, unconditional on has_hetero).
            _organyl = composed_prefix_organyl_name(mol, visited, branches[0])
            if _organyl is None:
                return None
            return _assemble_amino_prefix_core([(_organyl, False)])

    # -N(alkyl)2+ -> dialkylamino (e.g. dimethylamino). site#2 (a phase):
    # the disubstituted case the docstring promised but was never implemented, so
    # N,N-dialkylamino substituents on a chain parent fell to `return None` and were
    # mis-walked into a spurious amino+alkylamino split. Multiplicity, ordering
    # and marking are all delegated to the ONE assembler; the local
    # `Counter` + `SIMPLE_MULTIPLIERS` copy that used to live here spelled
    # 'ethylmethylamino' for an ASYMMETRIC pair, which OPSIN reads as the single
    # substituent 2-ethylmethyl -- a different constitution. Principal-amine path
    # (_assemble_amine_name) is untouched (Pitfall 3).
    if len(branches) >= 2:
        branch_names = []
        all_pure = True
        for branch_start in branches:
            visited = set()
            stack_c = [branch_start]
            carbon_count = 0
            has_hetero = False
            has_ring = False
            while stack_c:
                idx = stack_c.pop()
                if idx in visited or idx == attach_idx:
                    continue
                if idx not in frag_set:
                    continue
                visited.add(idx)
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == 'C':
                    carbon_count += 1
                elif atom.GetAtomicNum() != 1:
                    has_hetero = True
                if ring_info.NumAtomRings(idx) > 0:
                    has_ring = True
                for n in atom.GetNeighbors():
                    if n.GetIdx() not in visited and n.GetIdx() != attach_idx:
                        stack_c.append(n.GetIdx())
            if carbon_count <= 0 or has_hetero or has_ring:
                all_pure = False
                break
            #, same reasoning as the single-branch case above.
            _fv = carbon_free_valence_prefix(mol, visited, branch_start)
            if _fv.prefix is not None:
                branch_names.append(_fv.prefix)
                continue
            if _fv.must_fail_closed:
                all_pure = False
                break
            # Same carbon-count hazard as the single-branch case, and the same
            # cure -- the shared constitution-perceiving organyl namer.
            _organyl = composed_prefix_organyl_name(mol, visited, branch_start)
            if _organyl is None:
                all_pure = False
                break
            branch_names.append(_organyl)
        if all_pure and branch_names:
            return _assemble_amino_prefix_core(
                [(nm, False) for nm in branch_names])

    return None


def _name_sulfanyl_branch(mol, frag_atoms, attach_idx, parent_atoms):
    """Name a sulfanyl-attached branch: -S-R -> alkylsulfanyl.

    For simple -S-alkyl branches, produces IUPAC substitutive prefix names:
    -S-CH3 -> methylsulfanyl
    -S-C2H5 -> ethylsulfanyl

    Args:
        mol: RDKit Mol.
        frag_atoms: List of atom indices in the fragment.
        attach_idx: Atom index of the S attachment atom.
        parent_atoms: Set of parent atom indices.

    Returns:
        Sulfanyl prefix name, or None if not a simple case.
    """
    frag_set = set(frag_atoms)
    parent_set = set(parent_atoms) if parent_atoms else set()
    s_atom = mol.GetAtomWithIdx(attach_idx)

    # Collect the WHOLE S-side fragment -- every heavy atom reachable from S
    # inside frag_set without crossing S or the parent.
    #
    # This walk used to stop at the first heteroatom, because it only pushed
    # neighbours from INSIDE an ``if symbol == 'C'`` block. That had two
    # consequences, and the second is the defect:
    #
    # 1. ``alkyl_atoms`` held carbons BY CONSTRUCTION, so the guard below it
    # ("any atom in alkyl_atoms that is not C or H -> defer") could never
    # be True. It was VACUOUS -- dead code wearing the costume of a check.
    # 2. Because the walk never crossed a heteroatom, it never VISITED the
    # atoms it was dropping, so nothing downstream could notice the loss.
    # On CHEBI:131797 the S-side is a whole cysteinyl-glycine arm and this
    # returned 'propylsulfanyl' while silently discarding 7 heavy atoms
    # (C,C,N,N,O,O,O) -- a name asserting -S-CH2CH2CH3 for something else.
    #
    # Walking the real fragment first is what makes the proof below possible:
    # you cannot check atoms you never looked at.
    side_atoms = set()
    stack = [
        n.GetIdx() for n in s_atom.GetNeighbors()
        if n.GetIdx() in frag_set and n.GetIdx() not in parent_set
        and n.GetIdx() != attach_idx
    ]
    while stack:
        idx = stack.pop()
        if idx in side_atoms or idx == attach_idx or idx in parent_set:
            continue
        if idx not in frag_set:
            continue
        if mol.GetAtomWithIdx(idx).GetAtomicNum() <= 1:
            continue
        side_atoms.add(idx)
        for n in mol.GetAtomWithIdx(idx).GetNeighbors():
            if n.GetIdx() not in side_atoms:
                stack.append(n.GetIdx())

    if not side_atoms:
        return None

    # The alkyl stem must attach to S at exactly one atom, and that atom must be
    # the terminus of a genuine unbranched saturated acyclic all-carbon chain --
    # the ONLY shape ``get_alkyl_name(n)`` can spell. ``fragment_is_linear_
    # terminal_alkyl`` is the shared primitive for exactly that question; use it
    # rather than re-deriving a guard list here.
    from .substituent_naming import fragment_is_linear_terminal_alkyl

    anchors = [
        idx for idx in side_atoms
        if mol.GetBondBetweenAtoms(idx, attach_idx) is not None
    ]
    if len(anchors) != 1:
        return None
    if not fragment_is_linear_terminal_alkyl(mol, list(side_atoms), anchors[0]):
        return None

    try:
        alkyl_name = get_alkyl_name(len(side_atoms))
        return f"{alkyl_name}sulfanyl"
    except (ValueError, KeyError):
        return None


_CHALCOGEN_ATOMIC_NUMS = frozenset({8, 16, 34, 52})  # O, S, Se, Te


def _is_divalent_chalcogen_atom(atom) -> bool:
    """True iff *atom* is a neutral, non-aromatic divalent chalcogen with only
    single bonds — the ``-O-``/``-S-`` ether-oxidation state of a peroxide /
    disulfide / thioperoxol linkage.

     guard (Phase D code review): without this, a higher-oxidation-state S
    (sulfinyl ``-S(=O)-`` / sulfonyl ``-S(=O)(=O)-`` / thiosulfonate) was claimed
    as a disulfide and its ``=O`` atoms dropped, producing a parseable name for a
    DIFFERENT molecule. Mirrors ``skeletal_replacement._dichalcogen_bond_set`` and
    the ``benzene.py`` ring-substituent guard.
    """
    if atom.GetAtomicNum() not in _CHALCOGEN_ATOMIC_NUMS:
        return False
    if atom.GetFormalCharge() != 0 or atom.GetIsAromatic():
        return False
    from rdkit import Chem as _Chem
    return all(b.GetBondType() == _Chem.BondType.SINGLE for b in atom.GetBonds())


def _name_peroxy_or_disulfanyl_R(mol, r_start, boundary, frag_set):
    """Name the R group of a -O-O-R / -S-S-R substituent as a substituent prefix.

    Collects the R fragment (atoms in ``frag_set`` reachable from ``r_start``
    without crossing the two-chalcogen ``boundary``) and delegates to the shared
    ``name_substituent`` cascade so alkyl (methyl/ethyl), aryl (phenyl), and
    branched R groups are all named via one root-cause path. Returns None if R
    is empty.
    """
    visited = set()
    stack = [r_start]
    while stack:
        idx = stack.pop()
        if idx in visited or idx in boundary or idx not in frag_set:
            continue
        visited.add(idx)
        for n in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = n.GetIdx()
            if nidx not in visited and nidx not in boundary:
                stack.append(nidx)
    if not visited:
        return None
    r_name = name_substituent(mol, sorted(visited), r_start)
    if not r_name:
        return None
    # (Phase D code review): a COMPOUND R needs enclosing marks before the
    # outer peroxy/disulfanyl suffix is appended, so '[(methylperoxy)methyl]peroxy'
    # not the ambiguous '(methylperoxy)methylperoxy'. apply_enclosing_marks does the
    # ->->{} nesting; a simple/retained R (methyl, phenyl) is returned bare.
    from .naming_utils import apply_enclosing_marks, is_complex_substituent
    _needs_marks = (
        is_complex_substituent(r_name)
        or '(' in r_name or '[' in r_name  # embedded enclosing marks (nested peroxy/disulfanyl)
    )
    if _needs_marks and not (
        r_name.startswith('(') and r_name.endswith(')')
        and r_name.count('(') == 1
    ):
        r_name = apply_enclosing_marks(r_name, depth=-1)
    return r_name


def _name_peroxy_branch(mol, frag_atoms, attach_idx, parent_atoms):
    """DD2 Fix B (Phase D, (1)): name a peroxy branch -O-O-R -> (R)peroxy.

    The substituent attaches to the parent through a divalent O whose other
    bond is to a second O (the peroxide linkage). The far side R is named as a
    substituent prefix and suffixed with ``peroxy``:
      -O-O-CH3 -> methylperoxy
      -O-O-C2H5 -> ethylperoxy

    Returns None if the attach atom is not a peroxide O or R cannot be named.
    """
    frag_set = set(frag_atoms)
    o1 = mol.GetAtomWithIdx(attach_idx)
    if o1.GetSymbol() != 'O' or not _is_divalent_chalcogen_atom(o1):
        return None
    o2_idx = None
    for nbr in o1.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx in parent_atoms or nbr_idx not in frag_set:
            continue
        if nbr.GetSymbol() == 'O' and _is_divalent_chalcogen_atom(nbr):
            o2_idx = nbr_idx
            break
    if o2_idx is None:
        return None
    o2 = mol.GetAtomWithIdx(o2_idx)
    r_start = None
    for nbr in o2.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx == attach_idx or nbr_idx not in frag_set:
            continue
        r_start = nbr_idx
        break
    if r_start is None:
        return None
    r_name = _name_peroxy_or_disulfanyl_R(mol, r_start, {attach_idx, o2_idx}, frag_set)
    if not r_name:
        return None
    return f"{r_name}peroxy"


# substituent-chain multipliers for a homogeneous contiguous chalcogen
# chain named ``...<mult>sulfanyl``: k=2 disulfanyl, k=3 trisulfanyl,... Capped at
# octa (each verified OPSIN-parseable, 2026-09-23); a longer chain fails closed.
_CHALCOGEN_CHAIN_SUB_MULT = {
    2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta',
    6: 'hexa', 7: 'hepta', 8: 'octa',
}


def _name_disulfanyl_branch(mol, frag_atoms, attach_idx, parent_atoms):
    """§**** (BB:39325/:39327) / (1) /: name a
    homogeneous contiguous S-chain substituent, MAXIMISING the chain.

    *"Compounds with three or more contiguous identical chalcogen atoms are treated
    as parent hydrides in substitutive nomenclature"* — so the whole run of
    contiguous divalent sulfurs is ONE chain named ``<mult>sulfanyl`` (di-, tri-,
    tetra-, …), never split into a nested sulfanyl on a shorter chain::

      -S-S-CH3 -> methyldisulfanyl -S-SH -> disulfanyl
      -S-S-S-CH3 -> methyltrisulfanyl -S-S-SH -> trisulfanyl
      -S-S-S-S-C2H5 -> ethyltetrasulfanyl

    (Before this maximisation ``-S-S-S-CH2CH3`` was split into
    ``(ethylsulfanyl)disulfanyl`` — the RIGHT molecule, a non-PIN spelling.)

    Fail-closed (None) on a branched / hypervalent chain atom, a >1-substituent
    terminus, or a chain longer than the octa multiplier table. Mirrors
    :func:`_name_peroxy_branch`, which keeps the 2-atom -O-O-R peroxy form.
    """
    frag_set = set(frag_atoms)
    s1 = mol.GetAtomWithIdx(attach_idx)
    if s1.GetSymbol() != 'S' or not _is_divalent_chalcogen_atom(s1):
        return None
    # Walk the maximal contiguous linear chain of divalent S atoms, moving AWAY
    # from the parent (the parent-side neighbour is not in frag_set). Every atom
    # before the terminus must continue to EXACTLY one further chain S and carry no
    # other in-fragment heavy neighbour; a branch or hypervalent hub fails closed.
    chain = [attach_idx]
    prev = None
    cur = attach_idx
    while True:
        cur_atom = mol.GetAtomWithIdx(cur)
        onward_s: List[int] = []
        other_heavy: List[int] = []
        for n in cur_atom.GetNeighbors():
            ni = n.GetIdx()
            if ni == prev or ni not in frag_set or n.GetSymbol() == 'H':
                continue
            if n.GetSymbol() == 'S' and _is_divalent_chalcogen_atom(n):
                onward_s.append(ni)
            else:
                other_heavy.append(ni)
        if len(onward_s) == 1 and not other_heavy:
            prev, cur = cur, onward_s[0]
            chain.append(cur)
            continue
        if len(onward_s) == 0:
            break                              # terminus (other_heavy is R or empty)
        return None                            # forked / branched S hub -> decline
    k = len(chain)
    mult = _CHALCOGEN_CHAIN_SUB_MULT.get(k)
    if mult is None:
        return None                            # k<2 (unreachable via Tier 1.7) or >8
    stem = f"{mult}sulfanyl"
    # R on the terminal chain atom (at most one non-S heavy neighbour); none -> a
    # terminal -S…S-H, the bare <mult>sulfanyl.
    last = mol.GetAtomWithIdx(chain[-1])
    r_start = None
    for n in last.GetNeighbors():
        ni = n.GetIdx()
        if ni == chain[-2] or ni not in frag_set or n.GetSymbol() == 'H':
            continue
        if r_start is not None:
            return None                        # >1 terminal substituent -> decline
        r_start = ni
    if r_start is None:
        return stem
    r_name = _name_peroxy_or_disulfanyl_R(mol, r_start, set(chain), frag_set)
    if not r_name:
        return None
    return f"{r_name}{stem}"


def _name_mixed_chalcogen_branch(mol, frag_atoms, attach_idx, parent_atoms):
    """Wave2: name a MIXED divalent-chalcogen bridge substituent.

    Supported positively:
      * attach through O with an S inner atom, ``-O-S-R -> (Rsulfanyl)oxy``
        ([(methylsulfanyl)oxy]ethane for CCOSC, OPSIN-verified);
      * attach through S with a chalcogen (O/Se/Te) inner atom, ``-S-R ->
        {(R)}sulfanyl`` — the whole R beyond the S is named by the shared
        substituent cascade and enclosed if compound: ``-S-O-O-CH3 ->
        (methylperoxy)sulfanyl`` /, the compound-sulfanyl
        substituent of the acyl-hetero pseudoketone).

    Every other mixed shape (terminal-H inner chalcogen, branched inner atom)
    returns None; the CALLERS treat a detected mixed bridge whose namer declined
    as a terminal decline (fail closed), never falling through to the generic
    tiers that mangle the bridge into a bogus fragment.
    """
    frag_set = set(frag_atoms)
    a1 = mol.GetAtomWithIdx(attach_idx)
    if not _is_divalent_chalcogen_atom(a1):
        return None
    attach_sym = a1.GetSymbol()

    # -O-S-R (attach through O, inner S) -> (Rsulfanyl)oxy
    if attach_sym == 'O':
        s_idx = None
        for nbr in a1.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in parent_atoms or nbr_idx not in frag_set:
                continue
            if nbr.GetSymbol() == 'S' and _is_divalent_chalcogen_atom(nbr):
                s_idx = nbr_idx
                break
        if s_idx is None:
            return None
        s_atom = mol.GetAtomWithIdx(s_idx)
        r_start = None
        for nbr in s_atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx == attach_idx or nbr_idx not in frag_set:
                continue
            r_start = nbr_idx
            break
        if r_start is None:
            return None  # terminal -O-SH is the os_thioperoxol suffix, not this path
        r_name = _name_peroxy_or_disulfanyl_R(
            mol, r_start, {attach_idx, s_idx}, frag_set)
        if not r_name:
            return None
        return f"({r_name}sulfanyl)oxy"

    # -[S/Se/Te]-R (attach through a chalcogen, inner DIFFERENT chalcogen) ->
    # {(R)}sulfanyl / selanyl / tellanyl. The mixed-bridge gate guarantees the
    # inner neighbour is a DIFFERENT divalent chalcogen (so -S-S-R disulfanyl and
    # -S-C alkylsulfanyl never reach here); name the whole R group beyond the hub
    # via the shared cascade and enclose a compound R.
    _CHALCOGEN_YL = _CHALCOGEN_YL_SUFFIX
    if attach_sym in _CHALCOGEN_YL:
        r_start = None
        for nbr in a1.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in parent_atoms or nbr_idx not in frag_set:
                continue
            if r_start is not None:
                return None  # branched hub -> not this simple sulfanyl shape
            r_start = nbr_idx
        if r_start is None:
            return None
        # (BB 27914): when the inner chalcogen is a MONO one -- its own
        # continuation is a carbon -- the R side is an -O-R / -S-R group whose PIN
        # morphology is the CONTRACTED one: '(methoxysulfanyl)cyclohexane (PIN)'
        # for C6H11-S-O-CH3 is 'methoxy' + 'sulfanyl'. The generic cascade cannot
        # name that half: handed the chalcogen it re-roots at a carbon and returns
        # a HYDROXYALKYL, which turned tert-Bu-O-S- into
        # '(2-hydroxy-2-methylpropyl)sulfanyl' -- the O-S bridge rewritten as a
        # C-S bond, a different molecule that still round-tripped as *a* valid
        # name. Route the mono-chalcogen R side to the shared composed primitive.
        # A DI-chalcogen inner (-S-O-O-R) is NOT redirected: the cascade names
        # that class correctly and whole ('(methylperoxy)sulfanyl').
        _mono = composed_chalcogen_group_prefix(
            mol, sorted(frag_set), r_start, {attach_idx})
        if _mono:
            #: this concatenated compound prefix takes enclosing marks --
            # BB 27914 spells the PIN '(methoxysulfanyl)cyclohexane', not
            # 'methoxysulfanylcyclohexane'. Marked HERE so the producer is correct
            # on its own rather than relying on a caller to recognise the shape;
            # bare, '<alkoxy>sulfanyl' is not merely non-PIN but ambiguous, and the
            # aryl sibling '<aryl>disulfanyl' re-parses as a different molecule.
            # This does not double-enclose when the shared root rule in
            # naming_utils.needs_brackets also covers the token: that predicate
            # returns False for an already-enclosed name, so no caller adds a
            # second level (verified for all four morphologies below).
            # The pre-existing cascade path further down is deliberately NOT
            # re-marked -- it already returns shapes its callers escalate
            # correctly ('(methylperoxy)sulfanyl' -> '[...]').
            from .naming_utils import apply_enclosing_marks as _aem
            return _aem(f"{_mono}{_CHALCOGEN_YL[attach_sym]}", depth=-1)
        r_name = name_substituent(mol, sorted(frag_set - {attach_idx}), r_start)
        if not r_name:
            return None
        from .naming_utils import apply_enclosing_marks, is_complex_substituent
        if is_complex_substituent(r_name) or '(' in r_name or '[' in r_name:
            r_name = apply_enclosing_marks(r_name, depth=-1)
        return f"{r_name}{_CHALCOGEN_YL[attach_sym]}"

    return None


def _is_mixed_chalcogen_bridge_attach(mol, attach_idx, frag_atoms_set):
    """True when ``attach_idx`` is a divalent chalcogen whose in-fragment
    neighbor is a DIFFERENT divalent chalcogen — the mixed-bridge
    shape the generic substituent tiers must never be allowed to mangle.

    The MIXED case of the shared:func:`is_dichalcogen_bridge_attach` walker;
    kept as a named predicate for its call sites, never as a second walker.
    """
    return is_dichalcogen_bridge_attach(
        mol, attach_idx, frag_atoms_set, require_different=True)


def _find_attach_atom_in_frag(mol, frag_atoms, parent_atoms):
    """Find the fragment atom that is bonded to the parent structure.

    Args:
        mol: RDKit Mol.
        frag_atoms: List of atom indices in the fragment.
        parent_atoms: Set of atom indices in the parent.

    Returns:
        Atom index of the attachment atom, or None.
    """
    frag_set = set(frag_atoms)
    for idx in frag_atoms:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in parent_atoms:
                return idx
    return None


def _get_frag_smiles(mol, frag_atoms):
    """Get SMILES for a fragment defined by atom indices.

    Args:
        mol: RDKit Mol.
        frag_atoms: Collection of atom indices.

    Returns:
        SMILES string, or "unknown" if extraction fails.
    """
    try:
        atoms = list(frag_atoms)
        if atoms:
            smi = Chem.MolFragmentToSmiles(mol, atomsToUse=atoms)
            return smi if smi else "unknown"
    except Exception:
        pass
    return "unknown"


# ============================================================================
# Atom Set Collection (for deduplication)
# ============================================================================


def collect_substituent_atom_set(substituent_infos):
    """Collect the union of all substituent atom indices.

    Used by callers to prevent double-counting: FGs whose atoms are
    entirely within a named branch should be skipped in standalone
    FG prefix generation.

    Args:
        substituent_infos: List of SubstituentInfo namedtuples.

    Returns:
        Frozenset of all original mol atom indices covered by substituents.
    """
    all_atoms = set()
    for info in substituent_infos:
        if info.frag_atoms:
            all_atoms.update(info.frag_atoms)
    return frozenset(all_atoms)
