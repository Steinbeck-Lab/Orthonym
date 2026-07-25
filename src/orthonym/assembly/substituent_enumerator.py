"""Unified substituent enumeration module with ReplaceCore-based extraction.

Provides a single enumeration path for all substituents on both ring and chain
parent structures. Replaces the previously fragmented three-path system that
caused silent drops, double-counting, and wrong locants.

Architecture:
  - Ring parents: ReplaceCore(mol, core_from_ring_atoms) -> fragment mols
  - Chain parents: Branch-point enumeration from features.substituents dict
  - All fragments: classify -> name via existing naming infrastructure

Public API:
  - discover_substituents(mol, parent_atoms, parent_type, ...) [Phase 84]
  - extract_ring_substituents(mol, ring_atoms, oriented_ring)
  - extract_chain_substituents(mol, principal_chain, substituents_dict)
  - classify_and_name_fragment(mol, frag_info, parent_atoms, features=None)
  - collect_substituent_atom_set(substituent_infos)

References:
    IUPAC 2013 P-31.1 (detachable prefixes)
    IUPAC 2013 P-44 (parent selection determines what's a substituent)
"""

import logging
import threading
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import Optional

# Phase 160.2 Plan-04-02 WR-04 closure: removed unused
# ``from typing import List, Optional, Set, Dict`` — none of the four
# names are referenced anywhere in 1608 LOC (verified via AST scan;
# they appeared only in docstrings, not annotations). Re-add narrowly
# scoped imports here when real annotations are added.

from rdkit import Chem
from rdkit.Chem import RWMol

from .naming_utils import get_alkyl_name, SIMPLE_MULTIPLIERS
from .substituent_naming import name_substituent_fragment, _name_aryl_methyl_ether
from .substituent_prefix_forms import _check_substituent_prefix_form
from ..rules.seniority import get_prefix

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
# Universal Substituent Discovery (Phase 84)
# ============================================================================


def discover_substituents(
    mol,
    parent_atoms,
    parent_type="auto",
    oriented_ring=None,
    principal_chain=None,
    atom_to_locant=None,
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
        atom_to_locant: Optional mapping of atom idx -> IUPAC locant.
        general_fallback: v28 Composer1 Task 5 gating flag. Passed ``True`` ONLY
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
            mol, tuple(parent_set), oriented_ring
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
    are discovered as compound fragments (USUB-04). No size limit.

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
        general_fallback: v28 Composer1 Task 5 gating flag (see
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
    for atom in mol.GetAtoms():
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
# P-29.2 free-valence morphology (Phase 1b)
# ============================================================================
#
# IUPAC 2013 P-29.2 fixes a substituent prefix's ENDING by the number of free
# valences on the attachment atom:
#
#     one   -> -yl       (methyl,      single bond to the parent)
#     two   -> -ylidene  (methylidene, double bond to the parent)
#     three -> -ylidyne  (methylidyne, triple bond to the parent)
#
# That number is a property of the ATTACHMENT BOND, not of the fragment's
# composition and not of its hydrogen count. Every tier of the naming cascade
# below builds its token from the fragment alone, so none of them can know it;
# ``name_substituent`` is the only place that holds the fragment AND the
# attachment atom together, which is why the morphology is decided here, once,
# for every tier.
#
# P-29.2 also settles the spelling: ``methylidene`` is the PIN for =CH2 as a
# prefix. ``methylene`` is the retained/general form and is NOT the PIN, so it
# is never emitted.

#: P-29.2 free-valence count -> suffix morpheme.
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
    P-29.2. Deliberately not the hydrogen count: H count merely correlates with
    bond order (``-CH3`` vs ``=CH2``), and a correlation is what produced the
    ``methyl``-for-``=CH2`` defect in the first place.

    Args:
        mol: RDKit Mol of the FULL molecule.
        frag_atoms: Iterable of atom indices forming the substituent fragment.
        attach_idx: The fragment atom bonded to the enclosing parent. Note this
            is an atom OF the fragment, so the linkage bond is the one from it
            to a neighbour OUTSIDE ``frag_atoms``.

    Returns:
        1, 2 or 3 -- or ``None`` when the shape is outside the P-29.2
        single-attachment-atom case: the fragment touches the parent at more
        than one bond (a bridge/spiro, P-25 rather than a prefix), the linkage
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
    # prefix describes. Two or more is a bridge (P-25), whatever their orders.
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

    Delegates to the shared P-29.2 text oracle so the producer and the proof
    spine's P7 cannot drift apart about what a token asserts. Anything the
    oracle refuses (``oxo``, ``hydroxy``, a multiplied ``-diyl`` ending) is not
    claimed here, so the guard never suppresses a prefix on a reading it could
    not defend.
    """
    from ..validation.name_morphemes import free_valence_morphology
    estimate = free_valence_morphology(token)
    return bool(estimate.confident and estimate.free_valences == 1)


def _carbon_ylidene_prefix(mol, frag_atoms, attach_idx, free_valence):
    """P-29.2 ``-ylidene`` / ``-ylidyne`` prefix for a CARBON free valence.

    Builds the token from the structure -- the parent hydride the fragment IS,
    plus the P-29.2 morpheme -- for two shapes:

    *Acyclic*: the chain running through the attachment atom, numbered so the
    free valence takes the lowest locant it can (P-29.3.2).

        =CH2                -> methylidene
        CH3-CH=             -> ethylidene
        CH3-CH2-CH=         -> propylidene       (P-29.6.2.3 retained stem)
        (CH3)2C=            -> propan-2-ylidene
        CH3-C(triple)       -> ethylidyne

    *Monocyclic*: an unsubstituted saturated carbocycle, whose free valence is
    on a ring atom and needs no locant (P-29.3.3 -- every ring atom of an
    otherwise-bare cycloalkane is equivalent).

        -(CH2)5C=           -> cyclohexylidene

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
    for bond in mol.GetBonds():
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


# ---------------------------------------------------------------------------
# The ONE P-29.2 verdict every substituent detector asks for
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
#   defers           -> single (or undecidable) free valence. The caller's own
#                       existing ``-yl`` naming is P-29.2-correct; carry on
#                       unchanged. This is the overwhelmingly common case and
#                       is what keeps every existing name byte-identical.
#   prefix is set    -> two or three free valences AND a prefix whose own text
#                       spells exactly that many. Use it verbatim.
#   must_fail_closed -> two or three free valences and no such prefix. The
#                       caller MUST abstain. Emitting ``-yl`` here names a
#                       different molecule, and dropping the fragment names a
#                       different molecule too.


#: Re-entrancy flag for the P-29.2 gate.
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
    """The P-29.2 verdict for one substituent fragment's attachment bond.

    ``free_valence`` is always the TRUE bond order at the attachment (1/2/3),
    or ``None`` when the shape is outside the single-attachment-atom case a
    free-valence prefix describes at all (a bridge, an aromatic linkage).
    ``in_class`` is False when the attachment atom is not carbon, where the
    P-29.2 carbon morphology does not apply and the caller's own heteroatom
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
        """True when the caller's own existing naming path is P-29.2-correct."""
        return (not self.in_class
                or self.free_valence is None
                or self.free_valence == 1)

    @property
    def must_fail_closed(self) -> bool:
        """True when P-29.2 needs a multivalent prefix and none could be built."""
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

    P-29.2 says the free-valence morpheme is a SUFFIX on the fragment's parent
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
    """P-29.2 verdict for a CARBON-attached substituent fragment.

    The shared entry point for every substituent detector -- monocyclic,
    heterocyclic, bicyclo, von Baeyer polycyclic, spiro, fused. Use it wherever
    a fragment is about to be named from its carbon count::

        verdict = carbon_free_valence_prefix(mol, sub_atoms, first_atom)
        if verdict.prefix:
            name = verdict.prefix
        elif verdict.must_fail_closed:
            ...abstain...        # never get_alkyl_name(carbon_count)
        else:
            name = get_alkyl_name(carbon_count)   # unchanged legacy path

    Args:
        mol: RDKit Mol of the FULL molecule.
        frag_atoms: Atom indices of the substituent fragment.
        attach_idx: The FRAGMENT atom bonded to the parent (not the parent
            atom -- the linkage bond runs from this atom OUT of the fragment).

    Scope -- deliberately CARBON only. The multivalent heteroatom prefixes
    (``oxo``, ``sulfanylidene``, ``imino``) are a separate class that the
    detectors' own heteroatom machinery already names correctly, and their
    tokens spell no P-29.2 morpheme at all, so the text check below cannot
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

    References: IUPAC 2013 P-29.2, P-29.3.2, P-29.3.3.
    """
    from ..validation.name_morphemes import free_valence_morphology

    frag = set(frag_atoms)
    free_valence = _free_valence_at_attachment(mol, frag, attach_idx)
    if free_valence is None or free_valence == 1:
        return FreeValencePrefix(
            free_valence, None,
            "single or undecidable free valence: caller's own naming applies")

    if attach_idx is None or mol.GetAtomWithIdx(attach_idx).GetAtomicNum() != 6:
        # Out of the P-29.2 CARBON class (see the scope note above).
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
# Universal Substituent Naming (Phase 85)
# ============================================================================


def name_substituent(mol, frag_atoms, attach_idx, allow_mancude: bool = False):
    """Name a substituent fragment with the free-valence morphology P-29.2 requires.

    Thin P-29.2 gate over :func:`_name_substituent_cascade`, which holds the
    five-tier naming logic. The split exists because the cascade's tiers all
    build their token from the FRAGMENT alone -- they never see the attachment
    bond -- so none of them can know whether the free valence is single
    (``-yl``), double (``-ylidene``) or triple (``-ylidyne``). This function is
    the one place that holds the fragment and the attachment atom together, so
    the morphology is decided here, once, for every tier.

    Behaviour by free valence:

    * **one** (or undecidable -- a bridge, an aromatic linkage): the cascade
      runs and its answer is returned untouched. This is the overwhelmingly
      common case and is byte-identical to the pre-P-29.2 behaviour.
    * **two or three**: the carbon ylidene/ylidyne constructor gets first
      refusal; if it declines, the cascade runs and its answer is accepted ONLY
      if the token does not spell a single free valence. A ``-yl`` token on a
      doubly-bonded free valence names a DIFFERENT molecule, so it is refused
      rather than shipped -- ``None`` under ``allow_mancude`` (the Tier-4.5
      de-masking convention) and the ``'substituent'`` unnameable sentinel
      otherwise, both of which fail closed downstream.

    The correct multivalent heteroatom prefixes (``oxo``, ``sulfanylidene``,
    ``imino``, ...) spell no ``-yl`` morpheme, so they pass the guard untouched.

    References:
        IUPAC 2013 P-29.2 (free-valence morphology), P-29.3.2 (lowest locant
        for the free valence).
    """
    # The bond-order read and the ylidene construction come from the SHARED
    # primitive, not from a private copy here -- this function and
    # carbon_free_valence_prefix used to hold the same two calls, which is the
    # duplication the class is meant to remove. ``free_valence`` below is the
    # TRUE bond order even for a non-carbon attachment (the primitive only
    # declines to CONSTRUCT for those), so the heteroatom guard further down is
    # unchanged.
    frag_atoms_set = set(frag_atoms)
    verdict = carbon_free_valence_prefix(mol, frag_atoms_set, attach_idx)
    if verdict.prefix is not None:
        return verdict.prefix
    free_valence = verdict.free_valence

    token = _name_substituent_cascade(
        mol, frag_atoms, attach_idx, allow_mancude=allow_mancude)

    if free_valence in (2, 3) and _token_asserts_single_free_valence(token):
        from ..metrics.abstention import AbstentionCode, record_abstention
        record_abstention(AbstentionCode.BRANCH_UNNAMEABLE,
                          detail='p29_2_free_valence_morphology')
        logger.debug(
            "P-29.2 guard: refused %r for a free valence of %d at atom %d",
            token, free_valence, attach_idx)
        return None if allow_mancude else "substituent"

    return token


def name_ylidene_substituent(mol, frag_atoms, attach_idx):
    """Name a fragment whose bond to its parent is DOUBLE, or ``None``.

    The shared entry point for the namers that cite a doubly-bonded fragment --
    hydrazone, semicarbazone, azine, the cumulative ium/ide chain, the P-64.5(3)
    ketene branch. Each of them used to spell the morphology itself::

        yl = name_substituent(mol, frag, c)
        if not yl.endswith("yl"):
            return None
        ... f"{yl}idene" ...

    which decided the free valence a second time, in the consumer, by rewriting
    a token. It only worked while the pipeline was returning the WRONG
    single-valence token for a double bond; correcting the pipeline made every
    such consumer reject its own correct input.

    Here the producer owns the morphology and the consumer VERIFIES it: the
    token comes back from ``name_substituent`` already carrying the P-29.2
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
        allow_mancude: v27 P1 opt-in (complete/best-effort engine tier only).
            Threaded to the ring chokepoint so a multi-ring cage substituent the
            narrow PIN namers decline (tricyclo+/adamantane, mancude fused
            aromatics) is named via the universal von-Baeyer cage engine instead
            of failing closed. Default False -> PIN-default byte-identical.

    Returns:
        str: IUPAC prefix name. Under the PIN-default path
            (``allow_mancude=False``) always non-None, always non-empty --
            the five-tier cascade guarantees Tier 5's descriptive fallback as
            the terminal case. Under ``allow_mancude=True`` (v28 Composer #1,
            Tier 4.5) may return ``None``: a clean abstention when the
            recursive decoration composer also declines and the fragment is
            ring-bearing with no honest systematic name available -- the
            'substituent' sentinel is de-masked to ``None`` rather than
            shipped. Simple non-ring fragments still resolve via the
            descriptive fallback even under ``allow_mancude=True``.

    References:
        IUPAC 2013 P-31.1 (detachable prefixes)
        Phase 85 design: five-tier cascade with guaranteed fallback
    """
    import re as _re
    from .fragment_naming import FRAGMENT_NAME_CACHE
    from .substituent_naming import (
        parent_to_prefix,
        _check_retained_substituent,
        _is_linear_alkyl,
        _add_substituent_stereo,
    )

    frag_atoms_set = set(frag_atoms)

    # Edge case: empty fragment
    if not frag_atoms_set:
        return "substituent"

    # ---- D-08 (Phase 177 WSB-02): stereo-dropping tier double-apply guard ----
    # Tiers 0.5/1/1.5/1.6/2/3 build their prefix from a canonicalised fragment
    # (e.g. Tier-2's MolFragmentToSmiles strips @/@@ before the cache lookup),
    # so they SHORT-CIRCUIT Tier-4 — the only tier that natively reaches
    # _add_substituent_stereo. Without this guard a stereogenic substituent that
    # resolves via an early tier ships descriptor-less. _stereo_route() routes a
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
        for _b in mol.GetBonds():
            if (_b.GetBeginAtomIdx() in frag_atoms_set
                    and _b.GetEndAtomIdx() in frag_atoms_set
                    and _b.HasProp('_CIPCode')):
                return True
        return False

    def _stereo_route(prefix: str) -> str:
        # Route a stereo-dropping tier's return through the substituent stereo
        # emitter, unless the prefix already carries a leading "(...)" descriptor
        # (double-apply guard) or the fragment has no CIP stereo. `attach_idx` is
        # threaded so the emitter can derive the located descriptor (PIN name +
        # attachment locant) for an acyclic-alkyl substituent from STRUCTURE.
        if not prefix or _STEREO_BLOCK_RE.match(prefix):
            return prefix
        if not _frag_has_cip_stereo():
            return prefix
        return _add_substituent_stereo(
            mol, list(frag_atoms_set), prefix, attach_idx=attach_idx
        )

    # ---- Tier 0.5 (Phase 160.1 D-04): IUPAC P-65 / P-66 prefix-form check ----
    # PURE read-only check. Returns the IUPAC-canonical prefix form for any
    # fragment that ENTIRELY contains one of the 14 non-principal functional
    # groups (ester, ether, amide, sulfoxide, sulfone, thioether, nitrile,
    # carbamate, urea, isocyanate, isothiocyanate). Short-circuits Tier-1..5
    # for FG-bearing fragments, eliminating the 'methyl formatyl' /
    # 'hydroxymethyl' bug per RESEARCH §3 root-cause fix.
    try:
        # v27 P2 (P-63.6): allow_mancude (complete/best-effort tier) also lifts
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

    # ---- Tier 1.5 (Phase 173.5 L1): monocyclic heteroaryl PIN locant ----
    # A heteroaryl ring substituent (pyridine, imidazole, furan, ...) takes
    # free-valence numbering — pyridin-3-yl, 1H-imidazol-5-yl — instead of the
    # locant-less parent_to_prefix form (pyridinyl / imidazolyl) that the cache
    # (Tier 2) or recursive namer (Tier 4) would otherwise emit. Guarded:
    # returns None (so we fall through unchanged) unless the locant is provably
    # PIN-correct (IUPAC P-31.1.4.3.4).
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

    # ---- Tier 1.6 (WS-A.2): decorated monocyclic ring substituent ----
    # A ring fragment carrying its own substituents must keep them with
    # attachment-correct numbering ('2-oxocyclohexyl'), instead of the
    # cache/recursive parent_to_prefix form that keeps the PARENT numbering
    # ('1-oxocyclohexyl' — structurally impossible) or drops the group.
    # Guarded: returns None (fall through unchanged) unless the ring is a
    # supported simple monocycle AND the decorated name covers EXACTLY the
    # fragment atoms (P-14.4 numbering; see rules/ring_substituents.py).
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
    # Named (R)peroxy / (R)disulfanyl per P-63.3.1(1). Placed before the cache /
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
        # Wave2 T6c (P-63.3.2): MIXED divalent-chalcogen bridge (-O-S-R etc.).
        # Positive name for the O-attached sulfanyl case ((methylsulfanyl)oxy);
        # every other mixed shape is a TERMINAL decline — the generic tiers
        # below mangle the bridge into a wrong-constitution fragment, so
        # falling through is never allowed for this shape.
        if _is_mixed_chalcogen_bridge_attach(mol, attach_idx, frag_atoms_set):
            _parent_atoms = set(range(mol.GetNumAtoms())) - frag_atoms_set
            _mixed = _name_mixed_chalcogen_branch(
                mol, list(frag_atoms_set), attach_idx, _parent_atoms)
            return _stereo_route(_mixed) if _mixed else None

    # ---- Tier 1.8 (DD5 RC-6 / SEN-04): located acyclic alkyl ----
    # A BRANCHED or INTERNALLY-attached acyclic all-carbon saturated alkyl
    # substituent is named by its OWN principal chain numbered from the free
    # valence (hexan-2-yl, pentan-3-yl, 3-methylbutyl) per P-29.2 / P-46. This
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
                _located_acyclic_alkyl_name,
                _attach_is_chain_terminus,
                _is_linear_alkyl,
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

    # ---- Tier 1.9 (v22 C-T2 / V-3): ether-substituted carbon chain ----
    # A saturated all-carbon chain bearing ether -O-R substituent(s), numbered
    # from the free valence, named (R-oxy)alkyl per P-63.2.2.2 (phenoxymethyl,
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

    # ---- Tier 1.92 (v23 Phase 8, P-68.2.2): Group-14 silyl/germyl substituent --
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

    # ---- Tier 1.93 (W2F-P7, P-68.3): phosphanyl (P-rooted) substituent ----
    # A phosphorus-rooted substituent (-PH2 -> phosphanyl, -PR2 -> dialkyl/
    # diarylphosphanyl) is cited via rules/phosphorus.name_phosphanyl_substituent.
    # MUST precede the Tier-5 descriptive fallback, which returns the
    # 'substituent' sentinel for a lone P — dropping it and failing the molecule
    # closed ('OC(=O)CCP' -> propanoic acid -> SELF-01 rejects -> 'unknown').
    # The helper fires ONLY for a clean neutral organyl/hydride P (excludes a
    # phosphoryl/phosphonic P=O, named by the oxoacid subsystem) -> fail-closed
    # (falls through to 'substituent') on any decline. The λ5 branch is Task 3.
    if attach_idx is not None and attach_idx in frag_atoms_set:
        if mol.GetAtomWithIdx(attach_idx).GetSymbol() == 'P':
            try:
                from ..rules.phosphorus import name_phosphanyl_substituent
                _ph = name_phosphanyl_substituent(
                    mol, list(frag_atoms_set), attach_idx
                )
                if _ph:
                    return _stereo_route(_ph)
            except Exception:
                pass

    # ---- Tier 1.93b (W3-P10, P-67.1.5.1 / P-68.3): arsanyl (As-rooted) ----
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

    # ---- Tier 1.95 (Phase 4 SUBST-01): ring-system substituent chokepoint ----
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
                _ring_nm = name_ring_system_substituent(
                    mol, sorted(frag_atoms_set), attach_idx,
                    allow_enumerator_fallback=False,
                    allow_mancude=allow_mancude,
                )
                if _ring_nm:
                    return _stereo_route(_ring_nm)
        except Exception:
            pass

    # ---- Tier 1.96 (v23 SL): acyclic substituent with detachable prefixes ----
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

    # W2E-P1FG Task 11 (P-66.1.6.1.1.3): a carbon chain terminated by a
    # -NH-C(=O)-NH2 urea unit -> '{loc}-(carbamoylamino){chain}yl' ('not
    # ureido'). MUST precede the Tier-2 cache / Tier-4 recursive path, which
    # name the H-capped fragment as 'N-propylurea' -> 'N-propylureayl'.
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

    # ---- Tier 1.97 (v28 Composer1, P-63.2.5): chalcogen-rooted sulfanyl ----
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
                    prefix = parent_to_prefix(cached, chain_length=carbon_count)
                    if prefix:
                        return _stereo_route(prefix)
    except Exception:
        pass  # Cache miss is fine, continue to next tier

    # ---- Tier 3: Linear alkyl fast path (attached at a chain TERMINUS) ----
    # DD5 RC-6 / SEN-04: a linear chain attached at an INTERNAL carbon
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

    # ---- Tier 4.5 (v28 Composer1 Task 2): recursive decoration composition --
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
        _desc = _descriptive_fallback(mol, frag_atoms_set, attach_idx)
        return None if _desc == 'substituent' else _desc

    # ---- Tier 5: Descriptive fallback (guaranteed non-None) ----
    return _descriptive_fallback(mol, frag_atoms_set, attach_idx)


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

    # Wave2 T3a constitution-conservation guard: a ring-bearing fragment down
    # here was declined by every honest namer (incl. the ring engine). The
    # carbon-count alkyl branch below would flatten it into a linear chain
    # (methylcyclohexyl -> 'heptyl', a DIFFERENT constitution). Return the
    # explicit unnameable marker instead.
    try:
        _ri = mol.GetRingInfo()
        if any(_ri.NumAtomRings(a) > 0 for a in frag_atoms):
            # v25 P0 Task 0.1: ring-bearing branch declined by every honest
            # namer — the E2 recursive-namer census bucket.
            from ..metrics.abstention import AbstentionCode, record_abstention
            record_abstention(AbstentionCode.BRANCH_UNNAMEABLE,
                              detail='enumerator_ring_fallback')
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
        # Single carbon -- P-29.2: the morphology follows the ATTACHMENT BOND
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
    # IUPAC P-31.1.3: compound prefix names built from
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
    # v25 P0 Task 0.1: no tier could express this branch.
    from ..metrics.abstention import AbstentionCode, record_abstention
    record_abstention(AbstentionCode.BRANCH_UNNAMEABLE,
                      detail='enumerator_last_resort')
    return "substituent"


# ============================================================================
# v28 Composer1 Task 1: detach-and-name ring-substituent primitive
# ============================================================================


def _detach_and_name_ring_substituent(mol, frag_atoms, attach_idx,
                                       allow_mancude: bool = False):
    """Name the RING-SYSTEM CORE of a substituent fragment as a ``-yl`` token.

    NOTE (v28 Composer #1 final review, I2): retained as a tested T1
    primitive (see ``tests/unit/rules/test_v28_composer1.py``) but SUPERSEDED
    in production by ``_recursive_fragment_substituent_name``'s own core
    numbering (``_monocycle_position_map`` / ``polycyclic_core_numbering``),
    which can place a DECORATION locant on the core -- something this
    primitive's conjoined ``-yl`` token cannot carry. Not on any production
    call path today; kept for reuse/testing.

    Reusable FIRST primitive for the v28 always-emit recursive substituent
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
    substituents hanging off the ring core (that recursion is later v28
    composer tasks). Not wired into ``name_substituent`` yet (v28 Task 4) —
    PIN-default (``allow_mancude=False`` callers) is unaffected by this
    addition.

    Args:
        mol: RDKit Mol of the full molecule.
        frag_atoms: Iterable of atom indices belonging to the substituent
            fragment (may include non-ring decoration atoms; only the atoms
            that are IN A RING are used as the ring core).
        attach_idx: Atom index (must be one of ``frag_atoms``, and must be a
            ring atom) marking the open-valence attachment point.
        allow_mancude: v27/v28 opt-in (complete/best-effort engine tier
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
# v28 Composer1 Task 2: recursive decoration composition
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
    """P-14.4 / P-31.1.4.3.4 free-valence numbering of a SIMPLE monocyclic
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
    # v28 tranche T3: indicated hydrogen is a mancude-ring concept — a fully
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
    """v28 tranche T1: extract the numbering-INDEPENDENT ring-skeleton stem
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
    ``pyridin-3-yl`` / ``cyclohexyl`` / ``1H-pyrrol-2-yl`` ...) using the PIN
    stem tables, with the free-valence locant taken from ``pos``. Fail-closed
    (``None``) for a ring that is not a confidently-PIN monocyclic stem
    (partially unsaturated, ambiguous indicated H, or stem not in the table).
    Carbocyclic aromatic-6 / saturated forms cite the free valence implicitly
    at position 1 (``phenyl`` / ``cyclohexyl``), so those require
    ``pos[attach] == 1``.
    """
    from ..rules.ring_substituents import (
        identify_ring_system, _PIN_HETEROARYL_STEMS)
    from ..data.chain_names import get_chain_prefix
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
    # 'imidazole'; adjacent ring nitrogens => pyrazole (P-25.2.1).
    if ring_name == 'imidazole':
        n_idx = [i for i in het if mol.GetAtomWithIdx(i).GetSymbol() == 'N']
        if len(n_idx) == 2 and mol.GetBondBetweenAtoms(
                n_idx[0], n_idx[1]) is not None:
            stem = 'pyrazol'
    if stem is None:
        # v28 tranche T1 (resolves Composer #1 I1 duplication): the small
        # `_PIN_HETEROARYL_STEMS` table does not cover the systematic azoles
        # (isoxazole/oxazole/thiazole/triazole...) that `identify_ring_system`
        # reports as None. The authoritative dispatcher `get_ring_substituent_name`
        # names those via its retained/Hantzsch-Widman path. Borrow its
        # numbering-INDEPENDENT ring-skeleton stem here and re-cite the free
        # valence (+ the caller's decorations) on THIS recursion's own numbering.
        # This function is reached ONLY from the allow_mancude-gated
        # `_recursive_fragment_substituent_name`, so the PIN default is byte-
        # identical; a wrong stem is caught downstream by SELF-01.
        stem = _borrow_heteroarene_stem(mol, core, attach_idx)
    if stem is None:
        return None
    # v28 tranche T3: indicated H only on an aromatic/mancude core (a saturated
    # heterocycle's N-H is not indicated H) — mirrors _monocycle_position_map.
    ih = ([i for i in het if mol.GetAtomWithIdx(i).GetTotalNumHs() >= 1]
          if aromatic else [])
    if len(ih) > 1:
        return None
    ih_prefix = f'{pos[ih[0]]}H-' if ih else ''
    return f'{ih_prefix}{stem}-{pos[attach_idx]}-yl'


def _recursive_fragment_substituent_name(mol, frag_atoms, attach_idx,
                                          allow_mancude: bool = False):
    """General recursive composer: name a ring-bearing substituent fragment as
    ``{decorations}{ring-core}-yl`` by partitioning it into a ring CORE + its
    DECORATIONS and recursing.

    v28 always-emit keystone (Composer #1, Task 2). Reached from
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
    3. Number the core (simple monocycle: free-valence P-14.4 enumeration;
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
            return None
    else:
        # v28 Task 2b: POLYCYCLIC (fused / bridged / cage) decorated core. The
        # monocycle numberer declined, so route the core through the shared
        # polycyclic numberer, which returns ONE consistent {atom: locant} map +
        # the bare `...-<fv>-yl` tail (fused-carbocyclic PAH + von-Baeyer cage;
        # other polycyclic classes stay deferred -> None -> fail closed). ATTEMPT
        # a covered candidate; SELF-01 (the production RT gate) arbitrates any
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
    from .naming_utils import (
        apply_enclosing_marks, is_complex_substituent, get_multiplier_prefix,
        alpha_sort_key)
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
        token = (apply_enclosing_marks(dname, -1)
                 if is_complex_substituent(dname) else dname)
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


def extract_ring_substituents(mol, ring_atoms, oriented_ring):
    """Extract all substituent fragments from a ring parent using ReplaceCore.

    Builds a core mol from ring_atoms, calls ReplaceCore to extract all
    non-ring fragments as separate mol objects with isotope-labeled dummy
    atoms indicating attachment points.

    Args:
        mol: RDKit Mol object.
        ring_atoms: Tuple or list of ring atom indices (from principal_ring).
        oriented_ring: List of ring atom indices in IUPAC numbering order.

    Returns:
        List of SubstituentInfo namedtuples, one per substituent fragment.
        Empty list if ring has no substituents.
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

                    # Map to IUPAC locant via oriented_ring
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

    Phase 160.1 D-04: Tier-0.5 prefix-form check runs FIRST so that any
    fragment matching the 14-row IUPAC P-65/P-66 prefix-form table is
    named via the canonical prefix form (e.g., -C(=O)OCH3 -> methoxycarbonyl)
    before falling through to compound/pure_alkyl/fg_only classification.
    This eliminates the polyfunctional-path duplicate-name bug
    (hydroxymethyl + methoxycarbonyl on the same ester atoms) per
    RESEARCH §3 root-cause fix.

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

    # ---- Tier 0.5 (Phase 160.1 D-04): IUPAC P-65 / P-66 prefix-form check ----
    # Pure read-only check. Applies to fragments that entirely contain one of
    # the 14 non-principal functional groups. The polyfunctional handler routes
    # substituent fragments here (via _name_compound_substituent fallback);
    # without this gate the compound-substituent path generates "hydroxymethyl"
    # for the methyl-ester fragment per RESEARCH §3 bug trace.
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

    # WS-A task 9: ring-containing fragments go to the single
    # ring-substituent chokepoint (P-29.2 free-valence locant:
    # naphthalen-2-yl, pyridin-2-yl, ...) — composition-based naming below
    # would count a ring's carbons as a chain. Recursion-safe:
    # name_ring_system_substituent only uses get_ring_substituent_name and
    # the name_substituent cascade, never this router.
    try:
        _ri = mol.GetRingInfo()
        if attach_idx is not None and any(
                _ri.NumAtomRings(a) > 0 for a in frag_atom_set):
            from ..rules.ring_substituents import name_ring_system_substituent
            _ring_nm = name_ring_system_substituent(
                mol, sorted(frag_atom_set), attach_idx
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
            pattern = Chem.MolFromSmarts(smarts_str)
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

    # Fallback: count carbons
    carbon_count = sum(
        1 for idx in frag_atoms
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
    )
    if carbon_count > 0:
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

    # DD2 Fix B (Phase D, P-63.3.1(1)): peroxy branch -O-O-R -> (R)peroxy.
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

    # Wave2 T6c (P-63.3.2): MIXED divalent-chalcogen bridge — must be decided
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

    # Fallback: recursive naming for ring-containing compound fragments.
    # Use name_fragment_recursively() which has cycle detection via visited set.
    # This handles cases where the fragment is a ring system with heteroatoms
    # that the simpler naming paths above cannot handle (DROP-18/19/24/25).
    # Guard: only for moderately-sized fragments (<=25 atoms).
    if len(frag_atoms) <= 25:
        frag_smiles = _get_frag_smiles(mol, frag_atoms)
        if frag_smiles and frag_smiles != "unknown":
            try:
                from .fragment_naming import name_fragment_recursively
                from .substituent_naming import parent_to_prefix
                frag_name = name_fragment_recursively(frag_smiles)
                if frag_name:
                    # Convert parent name to prefix form (e.g., "benzoic acid" -> not useful,
                    # but "pyridine" -> "pyridinyl", "cyclohexanone" -> "oxocyclohexyl")
                    carbon_count = sum(
                        1 for idx in frag_atoms
                        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
                    )
                    prefix_name = parent_to_prefix(frag_name, chain_length=carbon_count)
                    if prefix_name:
                        logger.debug(
                            "DROP-18/19 fallback: recursive naming for %s -> %s",
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

    IUPAC P-63.2.3: Ether substituents named as alkoxy when the oxygen
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
    # HYG-04 (Phase 167): single shared aryl-count helper (was inline benzyloxy here).
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

    # P-63.2.3.2 + P-14.5.2: branched alkyl groups must use (alkan-n-yl)oxy form
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
            # P-14.5.2: the oxy suffix is appended to the alkyl name, then the
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


def _name_amino_branch(mol, frag_atoms, attach_idx, parent_atoms):
    """Name an N-attached branch: -NH-R -> alkylamino, phenylamino/anilino.

    IUPAC P-62.2.3: Amine substituents named as amino when nitrogen is the
    attachment point. Also handles acylamino (-NH-C(=O)-R).

    Key patterns:
      -NH2 -> amino (handled as fg_only, not here)
      -NH-CH3 -> methylamino
      -NH-phenyl -> anilino (retained name for phenylamino)
      -NH-C(=O)-R -> amido family (acetamido, hexanamido; P-66.1.1.4.3
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
            # Wave2 T1c (P-66.1.1.4.3): method (1) amido prefix is the PIN
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

            # Count carbons in the acyl R-group (excluding the carbonyl C and =O)
            acyl_carbons = 0
            visited = set()
            stack_c = [branch_start]
            while stack_c:
                idx = stack_c.pop()
                if idx in visited or idx == attach_idx:
                    continue
                if idx not in frag_set:
                    continue
                visited.add(idx)
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == 'C':
                    acyl_carbons += 1
                for n in atom.GetNeighbors():
                    if n.GetIdx() not in visited and n.GetIdx() != attach_idx:
                        stack_c.append(n.GetIdx())
            if acyl_carbons >= 1:
                # acyl_carbons includes the carbonyl C
                # Legacy fallback (method (2), non-preferred):
                # chain_prefix + "anoyl" + "amino"
                acyl_name = get_chain_prefix(acyl_carbons) + "anoylamino"
                return acyl_name

    # Check for anilino: -NH-phenyl (isolated benzene ring directly on N)
    for branch_start in branches:
        branch_atom = mol.GetAtomWithIdx(branch_start)
        if branch_atom.GetIsAromatic() and branch_atom.GetSymbol() == 'C':
            for ring in ring_info.AtomRings():
                if branch_start in ring and len(ring) == 6:
                    all_arom = all(mol.GetAtomWithIdx(r).GetIsAromatic() for r in ring)
                    all_c = all(mol.GetAtomWithIdx(r).GetSymbol() == 'C' for r in ring)
                    if all_arom and all_c:
                        # Verify isolated (not fused)
                        ring_set_check = set(ring)
                        is_fused = any(
                            set(other) != ring_set_check and set(other) & ring_set_check
                            for other in ring_info.AtomRings()
                        )
                        if not is_fused:
                            return "anilino"

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
        if carbon_count > 0 and not has_hetero and not has_ring:
            # P-29.2: the branch's bond to the nitrogen may be DOUBLE
            # (-N=CH-CH3), where the prefix is 'ethylideneamino'. The carbon
            # walk above crosses that bond without noticing it, so 'CC=NCC(=O)O'
            # was named (ethylamino)iminoacetic acid -- a different molecule.
            _fv = carbon_free_valence_prefix(mol, visited, branches[0])
            if _fv.prefix is not None:
                return f"{_fv.prefix}amino"
            if _fv.must_fail_closed:
                return None
            try:
                alkyl_name = get_alkyl_name(carbon_count)
                return f"{alkyl_name}amino"
            except (ValueError, KeyError):
                pass

    # -N(alkyl)2+ -> dialkylamino (e.g. dimethylamino). HYG-04 site#2 (Phase 167):
    # the disubstituted case the docstring promised but was never implemented, so
    # N,N-dialkylamino substituents on a chain parent fell to `return None` and were
    # mis-walked into a spurious amino+alkylamino split. Mirror the WORKING
    # principal-amine multiplicity (composer._assemble_amine_name: Counter +
    # SIMPLE_MULTIPLIERS) as a PREFIX form (no N- locants). Principal-amine path
    # (_assemble_amine_name) is untouched (Pitfall 3).
    if len(branches) >= 2:
        from collections import Counter

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
            # P-29.2, same reasoning as the single-branch case above.
            _fv = carbon_free_valence_prefix(mol, visited, branch_start)
            if _fv.prefix is not None:
                branch_names.append(_fv.prefix)
                continue
            if _fv.must_fail_closed:
                all_pure = False
                break
            try:
                branch_names.append(get_alkyl_name(carbon_count))
            except (ValueError, KeyError):
                all_pure = False
                break
        if all_pure and branch_names:
            counts = Counter(branch_names)
            parts = []
            # Alphabetical by alkyl stem (di-/tri- are ignored for ordering,
            # matching the principal-amine analog's sorted assembly).
            for nm in sorted(counts.keys()):
                count = counts[nm]
                if count == 1:
                    parts.append(nm)
                else:
                    mult = SIMPLE_MULTIPLIERS.get(count, str(count))
                    parts.append(f"{mult}{nm}")
            return f"{''.join(parts)}amino"

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
    s_atom = mol.GetAtomWithIdx(attach_idx)

    # Collect carbon atoms bonded to S (excluding parent)
    alkyl_atoms = []
    for nbr in s_atom.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx in parent_atoms:
            continue
        if nbr_idx in frag_set and nbr.GetSymbol() == 'C':
            # BFS from this C to collect all connected carbons in fragment
            visited = set()
            stack = [nbr_idx]
            while stack:
                idx = stack.pop()
                if idx in visited or idx == attach_idx:
                    continue
                if idx not in frag_set:
                    continue
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() == 'C':
                    visited.add(idx)
                    for n in atom.GetNeighbors():
                        if n.GetIdx() not in visited and n.GetIdx() != attach_idx:
                            stack.append(n.GetIdx())
            alkyl_atoms.extend(visited)

    if not alkyl_atoms:
        return None

    # Check for non-C non-H atoms in the alkyl portion
    has_hetero_in_alkyl = any(
        mol.GetAtomWithIdx(idx).GetSymbol() not in ('C', 'H')
        for idx in alkyl_atoms
    )
    if has_hetero_in_alkyl:
        return None  # Complex case, defer

    carbon_count = len(alkyl_atoms)
    if carbon_count == 0:
        return None

    try:
        alkyl_name = get_alkyl_name(carbon_count)
        return f"{alkyl_name}sulfanyl"
    except (ValueError, KeyError):
        return None


_CHALCOGEN_ATOMIC_NUMS = frozenset({8, 16, 34, 52})  # O, S, Se, Te


def _is_divalent_chalcogen_atom(atom) -> bool:
    """True iff *atom* is a neutral, non-aromatic divalent chalcogen with only
    single bonds — the ``-O-``/``-S-`` ether-oxidation state of a peroxide /
    disulfide / thioperoxol linkage.

    CR-01 guard (Phase D code review): without this, a higher-oxidation-state S
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
    # WR-01 (Phase D code review): a COMPOUND R needs enclosing marks before the
    # outer peroxy/disulfanyl suffix is appended, so '[(methylperoxy)methyl]peroxy'
    # not the ambiguous '(methylperoxy)methylperoxy'. apply_enclosing_marks does the
    # ()->[]->{} nesting; a simple/retained R (methyl, phenyl) is returned bare.
    from .naming_utils import is_complex_substituent, apply_enclosing_marks
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
    """DD2 Fix B (Phase D, P-63.3.1(1)): name a peroxy branch -O-O-R -> (R)peroxy.

    The substituent attaches to the parent through a divalent O whose other
    bond is to a second O (the peroxide linkage). The far side R is named as a
    substituent prefix and suffixed with ``peroxy``:
      -O-O-CH3   -> methylperoxy
      -O-O-C2H5  -> ethylperoxy

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


def _name_disulfanyl_branch(mol, frag_atoms, attach_idx, parent_atoms):
    """DD2 Fix B (Phase D, P-63.3.1(1) / P-35.2.2): name a disulfanyl branch
    -S-S-R -> (R)disulfanyl; terminal -S-SH -> disulfanyl.

    Mirrors ``_name_peroxy_branch`` for the S-S linkage:
      -S-S-CH3 -> methyldisulfanyl
      -S-SH    -> disulfanyl (terminal; the H-bearing S carries no R)
    """
    frag_set = set(frag_atoms)
    s1 = mol.GetAtomWithIdx(attach_idx)
    if s1.GetSymbol() != 'S' or not _is_divalent_chalcogen_atom(s1):
        return None
    s2_idx = None
    for nbr in s1.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx in parent_atoms or nbr_idx not in frag_set:
            continue
        if nbr.GetSymbol() == 'S' and _is_divalent_chalcogen_atom(nbr):
            s2_idx = nbr_idx
            break
    if s2_idx is None:
        return None
    s2 = mol.GetAtomWithIdx(s2_idx)
    r_start = None
    for nbr in s2.GetNeighbors():
        nbr_idx = nbr.GetIdx()
        if nbr_idx == attach_idx or nbr_idx not in frag_set:
            continue
        r_start = nbr_idx
        break
    if r_start is None:
        # Terminal -S-SH: no R group -> bare disulfanyl (P-35.2.2).
        return "disulfanyl"
    r_name = _name_peroxy_or_disulfanyl_R(mol, r_start, {attach_idx, s2_idx}, frag_set)
    if not r_name:
        return None
    return f"{r_name}disulfanyl"


def _name_mixed_chalcogen_branch(mol, frag_atoms, attach_idx, parent_atoms):
    """Wave2 T6c (P-63.3.2): name a MIXED divalent-chalcogen bridge substituent.

    Supported positively:
      * attach through O with an S inner atom, ``-O-S-R -> (Rsulfanyl)oxy``
        ([(methylsulfanyl)oxy]ethane for CCOSC, OPSIN-verified);
      * attach through S with a chalcogen (O/Se/Te) inner atom, ``-S-R ->
        {(R)}sulfanyl`` — the whole R beyond the S is named by the shared
        substituent cascade and enclosed if compound: ``-S-O-O-CH3 ->
        (methylperoxy)sulfanyl`` (P-63.3.2 / P-65.6.3.4.2, the compound-sulfanyl
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
    _CHALCOGEN_YL = {'S': 'sulfanyl', 'Se': 'selanyl', 'Te': 'tellanyl'}
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
        r_name = name_substituent(mol, sorted(frag_set - {attach_idx}), r_start)
        if not r_name:
            return None
        from .naming_utils import is_complex_substituent, apply_enclosing_marks
        if is_complex_substituent(r_name) or '(' in r_name or '[' in r_name:
            r_name = apply_enclosing_marks(r_name, depth=-1)
        return f"{r_name}{_CHALCOGEN_YL[attach_sym]}"

    return None


def _is_mixed_chalcogen_bridge_attach(mol, attach_idx, frag_atoms_set):
    """True when ``attach_idx`` is a divalent chalcogen whose in-fragment
    neighbor is a DIFFERENT divalent chalcogen — the P-63.3.2 mixed-bridge
    shape the generic substituent tiers must never be allowed to mangle."""
    _CHALCOGENS = ('O', 'S', 'Se', 'Te')
    a = mol.GetAtomWithIdx(attach_idx)
    sym = a.GetSymbol()
    if sym not in _CHALCOGENS or not _is_divalent_chalcogen_atom(a):
        return False
    return any(
        n.GetIdx() in frag_atoms_set
        and n.GetSymbol() in _CHALCOGENS
        and n.GetSymbol() != sym
        and _is_divalent_chalcogen_atom(n)
        for n in a.GetNeighbors()
    )


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
