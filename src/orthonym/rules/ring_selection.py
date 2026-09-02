"""
Ring system type classification, scoring, and selection per IUPAC P-44.2.

Provides the foundation for correct principal ring system selection.
Implements:
- P-44.2.2 type hierarchy (RingSystemType enum)
- P-44.2.1 general criteria (ring_system_score tuple)
- Principal ring system selection (select_principal_ring_system)

These are pure functions that can be tested independently before integration.

Reference: IUPAC 2013 Blue Book, P-44.2 (Selection of Preferred Ring System)
"""

from enum import IntEnum
from typing import List, Optional, Set, Tuple

from rdkit import Chem

from ..perception.rings import (
    get_ring_info,
    get_ring_systems,
    get_spiro_atoms,
    is_heterocyclic,
)


# ============================================================================
# P-44.2.2 Ring System Type Hierarchy
# ============================================================================


class RingSystemType(IntEnum):
    """P-44.2.2 type hierarchy. Lower value = more senior.

    IUPAC 2013 P-44.2.2.2:
    1. Spiro ring systems (P-24)
    2. Cyclic phane parent hydrides (P-26.4)
    3. Fused ring systems (P-25)
    4. Bridged fused ring systems (P-25.7)
    5. Von Baeyer ring systems (P-23)
    6. Linear phane parent hydrides (P-26)
    7. Ring assemblies (P-28)

    MONOCYCLIC is not in P-44.2.2 but needed as fallback for simple rings.
    """
    SPIRO = 1           # P-44.2.2.2.1
    CYCLIC_PHANE = 2    # P-44.2.2.2.2 (stub)
    FUSED = 3           # P-44.2.2.2.3
    BRIDGED_FUSED = 4   # P-44.2.2.2.4
    VON_BAEYER = 5      # P-44.2.2.2.5
    LINEAR_PHANE = 6    # P-44.2.2.2.6 (stub)
    RING_ASSEMBLY = 7   # P-44.2.2.2.7
    MONOCYCLIC = 8      # Simple monocyclic (not in P-44.2.2 hierarchy)


# ============================================================================
# Heteroatom Seniority for P-44.2.1
# ============================================================================

# Higher value = more senior. Used negated in scoring tuple so min() wins.
# P-44.2.1 ring selection uses P-18(b) heteroatom order plus halogens.
# Expanded per IUPAC 2013 errata (BBerrors.html) to include all 20 elements
# that can appear as heteroatoms in ring systems.
_HETEROATOM_SENIORITY = {
    'N': 20,     # Most senior heteroatom per P-18(b)
    'F': 19,     # Halogen (P-44.2.1 ring comparison)
    'Cl': 18,    # Halogen
    'Br': 17,    # Halogen
    'I': 16,     # Halogen
    'O': 15,     # P-18(b) Group 16
    'S': 14,
    'Se': 13,
    'Te': 12,
    'P': 11,     # P-18(b) Group 15
    'As': 10,    # Per P-18(b) errata
    'Sb': 9,
    'Bi': 8,
    'Si': 7,     # Group 14
    'Ge': 6,
    'Sn': 5,
    'Pb': 4,
    'B': 3,      # Group 13
    'Al': 2,
    'Ga': 1,
}

# Seniority order for P-44.2.1(g) term-by-term variety comparison.
# Tuple position i represents the count of element _HETEROATOM_VARIETY_ORDER[i].
# Negated counts so min() selects ring with MORE of senior element.
_HETEROATOM_VARIETY_ORDER = [
    'F', 'Cl', 'Br', 'I', 'O', 'S', 'Se', 'Te', 'N',
    'P', 'As', 'Sb', 'Bi', 'Si', 'Ge', 'Sn', 'Pb', 'B', 'Al', 'Ga'
]


# ============================================================================
# Ring System Type Classification
# ============================================================================


def classify_ring_system_type(
    mol: Chem.Mol,
    ring_system_atoms: Set[int]
) -> RingSystemType:
    """Classify a ring system into its P-44.2.2 type.

    Reuses existing detection functions from the codebase, operating on a
    sub-molecule built from the ring system atoms when necessary.

    Classification priority (same as _classify_complex_ring in composer.py):
    1. Spiro junction detected -> SPIRO
    2. Fused core + extra bridges -> BRIDGED_FUSED
    3. Ortho-fused or ortho-peri-fused -> FUSED
    4. Bicyclo or polycyclic bridged -> VON_BAEYER
    5. Fallback -> MONOCYCLIC

    Args:
        mol: RDKit Mol object (full molecule)
        ring_system_atoms: Set of atom indices belonging to this ring system

    Returns:
        RingSystemType enum value
    """
    if not ring_system_atoms:
        return RingSystemType.MONOCYCLIC

    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    # Count how many SSSR rings are fully contained in this system
    contained_rings = [
        ring for ring in atom_rings
        if set(ring).issubset(ring_system_atoms)
    ]
    num_rings = len(contained_rings)

    if num_rings <= 1:
        # Single ring or no rings -> check for spiro (spiro connects 2 rings
        # but with include_spiro=True they become one system)
        # A spiro system has a spiro atom shared by exactly 2 rings
        spiro_atoms = get_spiro_atoms(mol)
        spiro_in_system = spiro_atoms & ring_system_atoms
        if spiro_in_system:
            return RingSystemType.SPIRO
        return RingSystemType.MONOCYCLIC

    # Multiple rings in this system -- build a sub-molecule for detection
    sub_mol = _build_submol(mol, ring_system_atoms)
    if sub_mol is None:
        return RingSystemType.MONOCYCLIC

    # Check for spiro first (shared single atom between two rings)
    spiro_atoms_in_system = get_spiro_atoms(mol)
    spiro_in_system = spiro_atoms_in_system & ring_system_atoms
    if spiro_in_system:
        return RingSystemType.SPIRO

    # .A +: Cyclophane classification fires after spiro and
    # before bridged-fused (P-44.2.2 hierarchy: SPIRO=1 < CYCLIC_PHANE=2 < FUSED=3).
    # Source: 155-CONTEXT.md,,; ring_selection.py:48 enum.
    # NOTE (root-cause-only, ISS-005): narrow exception scope to ImportError
    # only -- circular-import-safe lazy import idiom (matches multiplicative.py
    # lazy-import pattern). Runtime errors from is_cyclophane MUST bubble up;
    # do NOT swallow them. is_cyclophane already returns False (not raises) for
    # non-cyclophane mol per topology gate, so the try/except handles
    # ONLY the bootstrap ImportError case.
    try:
        from .phane import is_cyclophane
    except ImportError:
        pass
    else:
        if is_cyclophane(mol):
            all_systems = get_ring_systems(mol)
            if any(ring_system_atoms == s for s in all_systems):
                return RingSystemType.CYCLIC_PHANE

    # Classification priority (adapted from _classify_complex_ring in composer.py):
    # 1. bridged-fused (detect_bridged_fused) FIRST
    # 2. bicyclo (is_bicyclo_system) -- pure 2-ring bridged
    # 3. classify_fused_system 'bridged-fused' -- catches cases missed by
    # detect_bridged_fused (e.g., benzonorbornadiene). Must come BEFORE
    # is_polycyclic_system because that function would catch these as VB.
    # 4. polycyclic-bridged (is_polycyclic_system) -- tricyclo+ pure VB
    # Must come after bridged-fused checks to avoid misclassification.
    # 5. fused (ortho-fused/ortho-peri-fused)

    from .bridged_fused import detect_bridged_fused
    from .fused_rings import classify_fused_system
    from .bicyclo import is_bicyclo_system
    from .polycyclic import is_polycyclic_system

    # Step 1: Check bridged-fused via the dedicated detector
    try:
        if detect_bridged_fused(sub_mol):
            return RingSystemType.BRIDGED_FUSED
    except Exception:
        pass

    # Step 2: Check bicyclo (2-ring bridged)
    # This catches norbornane, bicyclo[2.2.2]octane, etc. BEFORE the
    # classify_fused_system check which incorrectly flags them.
    try:
        if is_bicyclo_system(sub_mol):
            return RingSystemType.VON_BAEYER
    except Exception:
        pass

    # Step 3: Get classify_fused_system result
    fused_type = 'not-fused'
    try:
        fused_type = classify_fused_system(sub_mol)
    except Exception:
        pass

    # Step 3b: Check for bridged-fused via classify_fused_system
    # This catches systems like benzonorbornadiene where detect_bridged_fused
    # misses but classify_fused_system correctly identifies 'bridged-fused'.
    # Pure VB systems (norbornane) are already caught at step 2.
    if fused_type == 'bridged-fused':
        return RingSystemType.BRIDGED_FUSED

    # Step 4: Check polycyclic bridged (tricyclo+)
    try:
        if is_polycyclic_system(sub_mol):
            return RingSystemType.VON_BAEYER
    except Exception:
        pass

    # Step 5: Check fused (ortho-fused or ortho-peri-fused)
    if fused_type in ('ortho-fused', 'ortho-peri-fused'):
        return RingSystemType.FUSED

    # Multi-ring but doesn't match any specific type
    # Could be ring assembly or monocyclic fallback
    return RingSystemType.MONOCYCLIC


def _build_submol(mol: Chem.Mol, atom_indices: Set[int]) -> Optional[Chem.Mol]:
    """Build an RWMol sub-molecule from a subset of atoms.

    Creates a new molecule containing only the specified atoms and the
    bonds between them. Preserves atom properties (element, aromaticity,
    formal charge, etc.) and bond properties (type, aromaticity).

    Args:
        mol: Source RDKit Mol object
        atom_indices: Set of atom indices to include

    Returns:
        New RDKit Mol object, or None on failure
    """
    if not atom_indices:
        return None

    try:
        rw = Chem.RWMol()
        # Map old atom index -> new atom index
        old_to_new = {}

        # Add atoms
        for old_idx in sorted(atom_indices):
            old_atom = mol.GetAtomWithIdx(old_idx)
            new_idx = rw.AddAtom(Chem.Atom(old_atom.GetAtomicNum()))
            new_atom = rw.GetAtomWithIdx(new_idx)
            new_atom.SetIsAromatic(old_atom.GetIsAromatic())
            new_atom.SetFormalCharge(old_atom.GetFormalCharge())
            new_atom.SetNoImplicit(old_atom.GetNoImplicit())
            new_atom.SetNumExplicitHs(old_atom.GetNumExplicitHs())
            old_to_new[old_idx] = new_idx

        # Add bonds
        seen_bonds = set()
        for old_idx in atom_indices:
            for bond in mol.GetAtomWithIdx(old_idx).GetBonds():
                begin = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                if begin in atom_indices and end in atom_indices:
                    bond_key = (min(begin, end), max(begin, end))
                    if bond_key not in seen_bonds:
                        seen_bonds.add(bond_key)
                        new_begin = old_to_new[begin]
                        new_end = old_to_new[end]
                        rw.AddBond(new_begin, new_end, bond.GetBondType())
                        new_bond = rw.GetBondBetweenAtoms(new_begin, new_end)
                        if new_bond is not None:
                            new_bond.SetIsAromatic(bond.GetIsAromatic())

        # Sanitize
        try:
            Chem.SanitizeMol(rw)
        except Exception:
            # If sanitization fails, try without kekulization
            try:
                Chem.SanitizeMol(
                    rw,
                    Chem.SanitizeFlags.SANITIZE_ALL
                    ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                )
            except Exception:
                pass

        return rw.GetMol()
    except Exception:
        return None


# ============================================================================
# Ring System Scoring (P-44.2.1 General Criteria)
# ============================================================================


def _spiro_fusion_count(mol: Chem.Mol, system_atoms: Set[int]) -> int:
    """P-44.2.2.2.1.1: number of spiro fusions in this ring system (spiro atoms
    that lie within the system). 0 for a non-spiro system. Deterministic
    (depends only on the atom set, not SMILES order)."""
    return len(get_spiro_atoms(mol) & set(system_atoms))


# Fixed width for the spiro-atom locant tuple term (padded with a high sentinel
# so tuples stay length-comparable and a non-spiro / shorter set never spuriously
# wins the "lower locant" tier). No real spiro system exceeds this many atoms.
_SPIRO_LOCANT_WIDTH = 8
_SPIRO_LOCANT_SENTINEL = 10 ** 6


def _is_saturated_monocyclic_spiro(mol: Chem.Mol, system_atoms: Set[int]) -> bool:
    """P-44.2.2.2.1.2 criterion (b): the system is a spiro system whose every
    component ring is a SATURATED MONOCYCLE (no ring multiple/aromatic bonds).
    Deterministic (atom-set only)."""
    spiro_in = get_spiro_atoms(mol) & set(system_atoms)
    if not spiro_in:
        return False
    ri = mol.GetRingInfo()
    rings_in = [set(r) for r in ri.AtomRings() if set(r) <= set(system_atoms)]
    if not rings_in:
        return False
    # each ring must be a monocycle (no fused edge = shares >=2 atoms with another
    # ring in the same system) and fully saturated.
    for i, ra in enumerate(rings_in):
        for j, rb in enumerate(rings_in):
            if i != j and len(ra & rb) >= 2:
                return False  # fused component, not monocyclic
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in system_atoms and b in system_atoms:
            if (bond.GetIsAromatic()
                    or bond.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)):
                return False
    return True


def _spiro_atom_locant_set(mol: Chem.Mol, system_atoms: Set[int]) -> Tuple[int, ...]:
    """P-44.2.2.2.1.2: the spiro-atom locant set (increasing, fixed-width padded).
    Uses the deterministic spiro numbering (get_spiro_numbering /
    _get_polyspiro_numbering, both CanonicalRankAtoms-tiebroken). Returns an
    all-sentinel tuple for a non-spiro system (so it never wins the tier).
    Spelling-independent."""
    spiro_in = sorted(get_spiro_atoms(mol) & set(system_atoms))
    pad = [_SPIRO_LOCANT_SENTINEL] * _SPIRO_LOCANT_WIDTH
    if not spiro_in:
        return tuple(pad)
    try:
        from .spiro import get_spiro_numbering, _get_polyspiro_numbering
        if len(spiro_in) == 1:
            numbering = get_spiro_numbering(mol, spiro_in[0])
        else:
            numbering = _get_polyspiro_numbering(mol, set(spiro_in))
    except Exception:
        return tuple(pad)
    if not numbering:
        return tuple(pad)
    locs = sorted(numbering[a] for a in spiro_in if a in numbering)
    if len(locs) != len(spiro_in):
        return tuple(pad)
    out = (locs + pad)[:_SPIRO_LOCANT_WIDTH]
    return tuple(out)


import re as _re

# Fixed width for the fusion-descriptor letter / number terms. Padded with a
# high sentinel so a system WITH (lower) descriptor letters/numbers wins the tie
# and tuples stay length-comparable.
_FUSION_LETTER_WIDTH = 8
_FUSION_LETTER_SENTINEL = 999
_FUSION_NUMBER_WIDTH = 8
_FUSION_NUMBER_SENTINEL = 10 ** 6


def _fused_system_name(mol: Chem.Mol, system_atoms: Set[int]) -> Optional[str]:
    """Deterministically produce the fused-ring name of ``system_atoms`` (the
    sub-system) so its von-Baeyer-style fusion descriptor can be parsed. Returns
    None when the system is not a nameable fused ring system. Depends only on the
    atom set (the underlying namers are spelling-independent by design)."""
    ri = mol.GetRingInfo()
    rings_in = [set(r) for r in ri.AtomRings() if set(r) <= set(system_atoms)]
    if len(rings_in) < 2:
        return None  # monocyclic / non-fused -> no fusion descriptor
    try:
        from .spiro import _name_fused_component
        rings = [list(r) for r in rings_in]
        named = _name_fused_component(mol, rings)
    except Exception:
        return None
    if named is None:
        return None
    return named[0]


def _fusion_descriptor_letters(mol: Chem.Mol, system_atoms: Set[int]) -> Tuple[int, ...]:
    """P-44.2.2.2.3.3 criterion (c): the fused system's italic fusion-descriptor
    LETTERS, compared as a set (lower letters = senior). Returns a fixed-width
    tuple of letter ordinals (a>=1), padded with a sentinel. ``()``-equivalent
    (all sentinel) when the system has no explicit fusion descriptor (retained
    names like quinoline/isoquinoline, or non-fused). Spelling-independent."""
    pad = (_FUSION_LETTER_SENTINEL,) * _FUSION_LETTER_WIDTH
    name = _fused_system_name(mol, system_atoms)
    if not name:
        return pad
    # fusion descriptors carry italic letters inside the bracket, e.g. furo[3,2-b]
    letters = sorted(
        ord(m.group(1)) - ord('a') + 1
        for m in _re.finditer(r"-([a-z])\]", name)
    )
    if not letters:
        return pad
    out = (letters + list(pad))[:_FUSION_LETTER_WIDTH]
    return tuple(out)


def _fusion_descriptor_numbers(mol: Chem.Mol, system_atoms: Set[int]) -> Tuple[int, ...]:
    """P-44.2.2.2.3.4 criterion (d): the fused system's fusion-descriptor NUMBERS
    (attachment locants inside the bracket, in citation order; lower = senior).
    Returns a fixed-width tuple padded with a sentinel; all-sentinel when no
    explicit descriptor. Spelling-independent (deterministic namer)."""
    pad = (_FUSION_NUMBER_SENTINEL,) * _FUSION_NUMBER_WIDTH
    name = _fused_system_name(mol, system_atoms)
    if not name:
        return pad
    nums: List[int] = []
    # descriptor numbers appear as "[2,3-c]" (and multi-attachment "[3,2-b:...]")
    for m in _re.finditer(r"\[([0-9,]+)-[a-z]", name):
        for tok in m.group(1).split(','):
            if tok.isdigit():
                nums.append(int(tok))
    if not nums:
        return pad
    out = (nums + list(pad))[:_FUSION_NUMBER_WIDTH]
    return tuple(out)


_P25_8_HET_LOCANT_WIDTH = 8
_P25_8_SENTINEL = 10 ** 6


def _p25_8_component_rank(mol: Chem.Mol, system_atoms: Set[int]) -> Tuple[int, ...]:
    """P-44.2.2.2.3.5 criterion (e): the senior ring COMPONENT per P-25.8. Two
    fused systems that tie on all prior criteria are separated by the seniority
    of their (base) component. The P-25.8 sub-criterion that distinguishes e.g.
    quinoline (N at locant 1) from isoquinoline (N at 2) is the HETEROATOM
    LOCANT SET in the component's own numbering (lower = senior). Returns a
    fixed-width tuple (lower = senior), all-sentinel when not resolvable so the
    term is INERT (fail-safe: never introduces a spelling-dependent difference).
    Deterministic (the component namer is spelling-independent)."""
    pad = (_P25_8_SENTINEL,) * _P25_8_HET_LOCANT_WIDTH
    ri = mol.GetRingInfo()
    rings_in = [set(r) for r in ri.AtomRings() if set(r) <= set(system_atoms)]
    if len(rings_in) < 2:
        return pad  # monocyclic / non-fused -> no component seniority tiebreak
    # any heteroatom present? if not, the (e) heteroatom-locant tiebreak is inert
    if not any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in system_atoms):
        return pad
    try:
        from .spiro import _name_fused_component
        named = _name_fused_component(mol, [list(r) for r in rings_in])
    except Exception:
        return pad
    if named is None:
        return pad
    _name, a2l = named

    def _base_int(v):
        # CP2 widened the fused-component locant map to also carry lettered
        # fusion locants ('4a') and (int, primes) tuples. This P-25.8 (e)-criterion
        # het-locant tiebreak must use the BASE integer of each (as it did pre-CP2,
        # when the map was coerced to int upstream) so a heteroatom on a ring-fusion
        # carbon still contributes its number -- not be silently dropped by an
        # int-only filter. Returns None only for a locant with no integer part.
        if isinstance(v, int):
            return v
        if isinstance(v, tuple) and v and isinstance(v[0], int):
            return v[0]
        if isinstance(v, str):
            digits = ''.join(c for c in v if c.isdigit())
            return int(digits) if digits else None
        return None

    het_locs = sorted(
        b for i in a2l
        if i in system_atoms and mol.GetAtomWithIdx(i).GetAtomicNum() != 6
        for b in (_base_int(a2l[i]),)
        if b is not None
    )
    if not het_locs:
        return pad
    out = (het_locs + list(pad))[:_P25_8_HET_LOCANT_WIDTH]
    return tuple(out)


def _bridged_fused_prebridge_metrics(
    mol: Chem.Mol, system_atoms: Set[int]
) -> Tuple[int, int, int, int]:
    """P-44.2.2.2.4 criteria (a),(b),(c),(n) — the cheaply + deterministically
    computable subset of the 14 bridged-fused tiebreakers:
      (a) more rings, (b) more ring atoms, (c) fewer heteroatoms,
      (n) more noncumulative double bonds — scoped to the bridged-fused system.
    Criteria (d)-(m) need the full bridge parse (attachment locants, composite/
    dependent-bridge classification) and are DEFERRED (fail-closed): the scorer
    never emits a spelling-dependent difference for them (p5_bridged engine).
    Returns (0,0,0,0) for a non-bridged-fused system so the term is inert
    elsewhere. Deterministic (atom-set only)."""
    zero = (0, 0, 0, 0)
    try:
        from .bridged_fused import detect_bridged_fused
    except ImportError:
        return zero
    sub = _build_submol(mol, system_atoms)
    if sub is None:
        return zero
    try:
        if not detect_bridged_fused(sub):
            return zero
    except Exception:
        return zero
    ri = mol.GetRingInfo()
    num_rings = sum(1 for r in ri.AtomRings() if set(r) <= set(system_atoms))
    num_ring_atoms = len(system_atoms)
    num_hetero = sum(
        1 for i in system_atoms if mol.GetAtomWithIdx(i).GetAtomicNum() != 6
    )
    num_double = 0
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in system_atoms and b in system_atoms:
            if bond.GetIsAromatic() or bond.GetBondType() == Chem.BondType.DOUBLE:
                num_double += 1
    # (a) more rings, (b) more atoms, (n) more double bonds -> negate for min();
    # (c) FEWER heteroatoms -> keep positive so fewer sorts first.
    return (-num_rings, -num_ring_atoms, num_hetero, -num_double)


def ring_system_score(
    mol: Chem.Mol,
    system_atoms: Set[int]
) -> tuple:
    """Score a ring system for principal ring system selection.

    Returns a scoring tuple where ALL values are arranged so that
    ``min()`` selects the most senior ring system.

    IUPAC P-44.2: General criteria (P-44.2.1) are applied BEFORE type
    hierarchy (P-44.2.2). Type hierarchy is a tiebreaker within the
    same general criteria class. P-44.4.1 unsaturation is a FURTHER
    tiebreaker, applied only after P-44.2.2 type seniority (so e.g.
    spiro > phane > fused stays senior to a mere double-bond difference).

    Tuple ordering (39 elements; Tasks 12-17 + the P-44.2.2.2.4 pre-bridge
    metrics append P-44.2.2.2.x tiebreakers AFTER the P-44.4.1 tier so they only
    break within-type ties):
    - [0] -has_heteroatom: P-44.2.1(a) heterocyclic preferred (negated)
    - [1] -has_nitrogen: P-44.2.1(b) N-containing preferred (negated)
    - [2] -senior_heteroatom_rank: P-44.2.1(c) most senior heteroatom (negated)
    - [3] -num_rings: P-44.2.1(d) more rings = senior (negated)
    - [4] -num_skeletal_atoms: P-44.2.1(e) more atoms = senior (negated)
    - [5] -num_heteroatoms: P-44.2.1(f) more heteroatoms = senior (negated)
    - [6..25] heteroatom_variety_tuple: P-44.2.1(g) term-by-term comparison
              (20 elements: -count_N, -count_F, -count_Cl, -count_Br, -count_I,
               -count_O, -count_S, -count_Se, -count_Te, -count_P,...)
    - [26] type_rank: P-44.2.2 type hierarchy (tiebreaker, lower = senior)
    - [27] -num_multiple_bonds: P-44.4.1.1 max ring multiple bonds (negated)
    - [28] -num_double_bonds: P-44.4.1.2 then max double bonds (negated)
    - [29] -spiro_fusions: P-44.2.2.2.1.1 more spiro fusions = senior (negated)
    - [30] -sat_monocyclic_spiro: P-44.2.2.2.1.2(b) all-sat-monocyclic (negated)
    - [31] spiro-atom locant set: P-44.2.2.2.1.2 lower locants (nested tuple)
    - [32] fusion-descriptor letters: P-44.2.2.2.3.3 lower letters (nested tuple)
    - [33] fusion-descriptor numbers: P-44.2.2.2.3.4 lower numbers (nested tuple)
    - [34] P-25.8 component rank: P-44.2.2.2.3.5 senior component (nested tuple;
           het-locant set — quinoline<isoquinoline)
    - [35..38] bridged-fused pre-bridge metrics (a,b,c,n): P-44.2.2.2.4

    The unsaturation tier (S1, WS-A.1) breaks the among-equal-carbocycle
    tie that previously made ``C1CCCCC1c1ccccc1`` resolve to the arbitrary
    list-order winner ``phenylcyclohexane``; benzene now wins on unsaturation
    (P-44.4.1.1) -> ``cyclohexylbenzene``. Appended AFTER type_rank so it can
    never override P-44.2.2 type seniority. RDKit reports benzene bonds as
    AROMATIC, so the counter must treat AROMATIC as multiple (a naive
    DOUBLE-only count gives benzene zero).

    Args:
        mol: RDKit Mol object
        system_atoms: Set of atom indices in this ring system

    Returns:
        Tuple suitable for comparison with min() to select most senior
    """
    if not system_atoms:
        return (
            (0, 0, 0, 0, 0, 0) + (0,) * len(_HETEROATOM_VARIETY_ORDER)
            + (999, 0, 0, 0, 0)
            + ((_SPIRO_LOCANT_SENTINEL,) * _SPIRO_LOCANT_WIDTH,)
            + ((_FUSION_LETTER_SENTINEL,) * _FUSION_LETTER_WIDTH,)
            + ((_FUSION_NUMBER_SENTINEL,) * _FUSION_NUMBER_WIDTH,)
            + ((_P25_8_SENTINEL,) * _P25_8_HET_LOCANT_WIDTH,)
            + (0, 0, 0, 0)
        )

    # 1. Type rank
    type_rank = int(classify_ring_system_type(mol, system_atoms))

    # 2-7. Analyze atoms in the system
    has_heteroatom = False
    has_nitrogen = False
    senior_heteroatom_rank = 0
    num_heteroatoms = 0
    heteroatom_counts = {}  # element -> count

    for idx in system_atoms:
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        if symbol != 'C':
            has_heteroatom = True
            num_heteroatoms += 1
            heteroatom_counts[symbol] = heteroatom_counts.get(symbol, 0) + 1
            if symbol == 'N':
                has_nitrogen = True
            rank = _HETEROATOM_SENIORITY.get(symbol, 0)
            if rank > senior_heteroatom_rank:
                senior_heteroatom_rank = rank

    # Number of SSSR rings fully contained in this system
    ri = mol.GetRingInfo()
    num_rings = sum(
        1 for ring in ri.AtomRings()
        if set(ring).issubset(system_atoms)
    )

    # Number of skeletal atoms
    num_skeletal_atoms = len(system_atoms)

    # P-44.4.1 unsaturation (aromatic-aware): count ring bonds that are
    # DOUBLE / TRIPLE / AROMATIC. RDKit kekulizes benzene to AROMATIC bonds,
    # so AROMATIC must count as multiple or an aromatic ring scores zero.
    num_multiple_bonds = 0
    num_double_bonds = 0
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a not in system_atoms or b not in system_atoms:
            continue
        bt = bond.GetBondType()
        if bond.GetIsAromatic() or bt in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE):
            num_multiple_bonds += 1
            if bond.GetIsAromatic() or bt == Chem.BondType.DOUBLE:
                num_double_bonds += 1

    # P-44.2.1(g): heteroatom variety -- term-by-term comparison by seniority
    # Build tuple: (-count_of_N, -count_of_F,..., -count_of_P)
    # Negated so min() selects ring with MORE of the most-senior element
    heteroatom_variety_tuple = tuple(
        -heteroatom_counts.get(elem, 0)
        for elem in _HETEROATOM_VARIETY_ORDER
    )

    # P-44.2.2.2.1.1: number of spiro fusions (more = senior). Appended
    # AFTER the P-44.4.1 unsaturation tier so it only breaks a WITHIN-spiro tie
    # and never overrides type/unsaturation seniority. Deterministic (atom-set
    # only), so it introduces no spelling dependence.
    spiro_fusions = _spiro_fusion_count(mol, system_atoms)
    # P-44.2.2.2.1.2: (b) all-saturated-monocyclic-spiro preferred,
    # then the lower spiro-atom locant set. Applied AFTER the Task-12 spiro-
    # fusion term (P-44.2.2.2.1 "applied successively"). Both are deterministic.
    sat_mono = _is_saturated_monocyclic_spiro(mol, system_atoms)
    spiro_locants = _spiro_atom_locant_set(mol, system_atoms)

    # P-44.2: General criteria (P-44.2.1) applied BEFORE type hierarchy (P-44.2.2)
    return (
        -int(has_heteroatom),               # P-44.2.1(a): heterocyclic preferred
        -int(has_nitrogen),                 # P-44.2.1(b): N-containing preferred
        -senior_heteroatom_rank,            # P-44.2.1(c): most senior heteroatom
        -num_rings,                         # P-44.2.1(d): more rings = senior
        -num_skeletal_atoms,                # P-44.2.1(e): more atoms = senior
        -num_heteroatoms,                   # P-44.2.1(f): more heteroatoms
        *heteroatom_variety_tuple,          # P-44.2.1(g): 20 elements, term-by-term
        type_rank,                          # P-44.2.2: type hierarchy (tiebreaker)
        -num_multiple_bonds,                # P-44.4.1.1: max ring multiple bonds
        -num_double_bonds,                  # P-44.4.1.2: then max double bonds
        -spiro_fusions,                     # P-44.2.2.2.1.1: more spiro fusions
        -int(sat_mono),                     # P-44.2.2.2.1.2(b): sat-monocyclic-spiro
        spiro_locants,                      # P-44.2.2.2.1.2: lower spiro-atom locants
        _fusion_descriptor_letters(mol, system_atoms),  # P-44.2.2.2.3.3: fusion letters
        _fusion_descriptor_numbers(mol, system_atoms),  # P-44.2.2.2.3.4: fusion numbers
        _p25_8_component_rank(mol, system_atoms),        # P-44.2.2.2.3.5: P-25.8 component
        *_bridged_fused_prebridge_metrics(mol, system_atoms),  # P-44.2.2.2.4 (a,b,c,n)
    )


# ============================================================================
# Principal Ring System Selection
# ============================================================================


def _ylidene_linked_parent_ring(
    mol: Chem.Mol,
    ring_systems: List[Set[int]],
) -> Optional[Set[int]]:
    """Return the ring system that is the parent under P-31.1.4 ylidene linkage.

    Detects a single exocyclic methine/methanediyl carbon that is double-bonded
    into exactly ONE ring system and single-bonded into exactly ONE OTHER ring
    system (and bonded to nothing else heavy). The double-bonded ring is the
    parent; the single-bonded ring + the methine carbon form the ylidene
    substituent (P-29.6.1 benzylidene for a bare phenyl).

    Fail-closed: returns None (no override) unless there is exactly one such
    linking carbon connecting exactly two of the given ring systems, with the
    two rings distinct and the double/single sides unambiguous. Any other shape
    (>2 rings involved, branching on the methine, no double bond, ring-directly-
    bonded-ring) leaves the P-44.2 score selector in charge.
    """
    def _ring_of(atom_idx: int) -> Optional[int]:
        for i, sysset in enumerate(ring_systems):
            if atom_idx in sysset:
                return i
        return None

    matches: List[Tuple[int, int]] = []  # (double_ring_idx, single_ring_idx)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'C' or atom.IsInRing():
            continue
        if atom.GetFormalCharge() != 0:
            continue
        heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        # exactly two heavy neighbours: one via '=' into a ring, one via '-'
        # into a different ring (a bare =CH- or =C< methine bridging two rings).
        if len(heavy) != 2:
            continue
        dbl_ring = None
        sgl_ring = None
        ok = True
        for n in heavy:
            nb = mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx())
            r = _ring_of(n.GetIdx())
            if r is None:
                ok = False
                break
            if nb.GetBondTypeAsDouble() == 2.0:
                if dbl_ring is not None:
                    ok = False
                    break
                dbl_ring = r
            elif nb.GetBondTypeAsDouble() == 1.0:
                if sgl_ring is not None:
                    ok = False
                    break
                sgl_ring = r
            else:
                ok = False
                break
        if not ok or dbl_ring is None or sgl_ring is None:
            continue
        if dbl_ring == sgl_ring:
            continue
        matches.append((dbl_ring, sgl_ring))

    # Exactly one unambiguous ylidene linker between two distinct rings.
    if len(matches) != 1:
        return None
    dbl_ring, _sgl = matches[0]
    return ring_systems[dbl_ring]


def _is_carbon_fused_system(mol, atoms: Set[int]) -> bool:
    """True iff ``atoms`` form an all-carbon ring system of >=2 fused rings
    (a triterpene/steroid-type aglycone core). Used ONLY by the best-effort
    glycoside-parent preference in:func:`select_principal_ring_system`."""
    ri = mol.GetRingInfo()
    n_rings = sum(1 for r in ri.AtomRings() if set(r) <= set(atoms))
    if n_rings < 2:
        return False
    return all(mol.GetAtomWithIdx(a).GetSymbol() == 'C' for a in atoms)


def _is_monosaccharide_ring(mol, atoms: Set[int]) -> bool:
    """True iff ``atoms`` form a single pyranose/furanose-like ring: one ring,
    exactly one ring oxygen, no other ring heteroatom, and >=2 exocyclic
    hydroxy / hydroxymethyl groups (a stereo-defined OR generic sugar). Used ONLY
    by the best-effort glycoside-parent preference."""
    ri = mol.GetRingInfo()
    aset = set(atoms)
    rings = [r for r in ri.AtomRings() if set(r) <= aset]
    if len(rings) != 1 or len(aset) not in (5, 6):
        return False
    ring_o = [a for a in aset if mol.GetAtomWithIdx(a).GetSymbol() == 'O']
    if len(ring_o) != 1:
        return False
    if any(mol.GetAtomWithIdx(a).GetSymbol() not in ('C', 'O') for a in aset):
        return False
    # count exocyclic -OH / -CH2OH on the ring carbons
    oh = 0
    for a in aset:
        at = mol.GetAtomWithIdx(a)
        if at.GetSymbol() != 'C':
            continue
        for nb in at.GetNeighbors():
            if nb.GetIdx() in aset:
                continue
            if nb.GetSymbol() == 'O' and nb.GetTotalNumHs() >= 1:
                oh += 1
            elif (nb.GetSymbol() == 'C' and nb.GetTotalNumHs() == 2
                  and any(x.GetSymbol() == 'O' and x.GetTotalNumHs() >= 1
                          for x in nb.GetNeighbors())):
                oh += 1  # -CH2OH
    return oh >= 2


def select_principal_ring_system(
    mol: Chem.Mol,
    ring_systems: List[Set[int]],
    principal_group_atoms=None,
) -> Tuple[int, ...]:
    """Select the most senior ring system from a list of candidates.

    Uses ring_system_score() to compare candidates. The system with the
    minimum score tuple is the most senior (P-44.2 hierarchy).

    Args:
        mol: RDKit Mol object
        ring_systems: List of sets, each set contains atom indices in
                      one ring system (from get_ring_systems())

    Returns:
        Tuple of sorted atom indices of the most senior ring system,
        or empty tuple if no ring systems provided
    """
    if not ring_systems:
        return ()

    if len(ring_systems) == 1:
        return tuple(sorted(ring_systems[0]))

    # tail (glycoside convention, best-effort only): a GLYCOSIDE names its
    # AGLYCONE as the parent and every sugar as a glycosyloxy substituent
    # (P-102 / the natural-product convention), even though strict P-44.2 makes
    # a heterocyclic sugar ring senior to an all-carbon ring system. When exactly
    # one all-carbon FUSED ring system (>=2 rings) competes with ONLY
    # monosaccharide-like rings (a single ring, exactly one ring O, no other ring
    # heteroatom, bearing >=2 exocyclic hydroxy/CH2OH -- i.e. a pyranose/furanose),
    # prefer the carbon core so the sugars become substituents. GATED on the
    # best-effort context so the PIN default is byte-identical (the 1652 gate is
    # untouched); SELF-01 round-trip is the 0-wrong net. Verified: #11/#12 saponins
    # name FULL-InChIKey only via this preference.
    try:
        from ..metrics.provenance import best_effort_ctx
        if best_effort_ctx.get():
            carbon_fused = [sy for sy in ring_systems
                            if _is_carbon_fused_system(mol, sy)]
            # A glycosylated large carbon core (triterpene/steroid, >=3 fused
            # carbon rings) that is strictly the largest ring system present, with
            # at least one bona-fide monosaccharide ring attached, is an aglycone:
            # prefer it as parent so every sugar (and the aglycone's own small
            # O-rings) becomes a substituent. The >=3-ring + largest + sugar-present
            # trident keeps this to the glycoside class.
            if len(carbon_fused) == 1:
                core = carbon_fused[0]
                others = [sy for sy in ring_systems if sy is not core]
                core_rings = sum(
                    1 for r in mol.GetRingInfo().AtomRings()
                    if set(r) <= set(core))
                if (others and core_rings >= 3
                        and len(core) >= max(len(sy) for sy in others)
                        and any(_is_monosaccharide_ring(mol, sy)
                                for sy in others)):
                    return tuple(sorted(core))
    except Exception:  # pragma: no cover - a hint must never break selection
        pass

    # P-44.1: the senior parent bears the principal characteristic group, and
    # that outranks the P-44.2 ring-type hierarchy scored below. ``ring_system_
    # score`` takes only the ring atoms, so it cannot honour P-44.1; the optional
    # ``principal_group_atoms`` hint supplies it. POSITIVE EVIDENCE only -- the
    # override fires ONLY when EXACTLY ONE ring system bears the group, so an
    # absent, empty, or ambiguous hint falls through to the unchanged scoring
    # (every pre-existing caller passes no hint -> byte-identical). A ring system
    # "bears" the group when a group atom is in it OR is bonded to it (covers a
    # ring-form suffix and an appended suffix like a ring -carboxylic acid).
    if principal_group_atoms:
        pg_atoms: Set[int] = set()
        for m in principal_group_atoms:
            if isinstance(m, int):
                pg_atoms.add(m)
            else:
                try:
                    pg_atoms.update(int(a) for a in m)
                except TypeError:
                    pass
        n_atoms = mol.GetNumAtoms()
        pg_atoms = {a for a in pg_atoms if 0 <= a < n_atoms}
        if pg_atoms:
            bearing = []
            for sy in ring_systems:
                sset = set(sy)
                if any(a in sset
                       or any(nb.GetIdx() in sset
                              for nb in mol.GetAtomWithIdx(a).GetNeighbors())
                       for a in pg_atoms):
                    bearing.append(sy)
            if len(bearing) == 1:
                return tuple(sorted(bearing[0]))

    # W2E-D2 (P-29.6.1 / P-31.1.4): a substituent attached to a ring by an
    # exocyclic DOUBLE bond (ylidene) belongs to the ring on the double-bond
    # side; the ring it is single-bonded to is cited as the ylidene substituent
    # (e.g. benzylidenecyclohexane: the =CH-C6H5 methine double-bonds cyclohexane
    # and single-bonds benzene, so cyclohexane is parent and 'benzylidene' the
    # substituent). The P-44.2 ring_system_score below would otherwise pick the
    # senior aromatic ring as parent and try to name the aliphatic ring as a
    # (never-nameable) ylidene substituent, failing closed. Force the
    # double-bonded ring here. Fail-closed (no override) on any shape but the
    # single-methine-between-exactly-two-ring-systems case.
    _ylidene_ring = _ylidene_linked_parent_ring(mol, ring_systems)
    if _ylidene_ring is not None:
        return tuple(sorted(_ylidene_ring))

    # Score each ring system and select the one with minimum score
    best_idx = 0
    best_score = ring_system_score(mol, ring_systems[0])

    for i in range(1, len(ring_systems)):
        score = ring_system_score(mol, ring_systems[i])
        if score < best_score:
            best_score = score
            best_idx = i

    return tuple(sorted(ring_systems[best_idx]))
