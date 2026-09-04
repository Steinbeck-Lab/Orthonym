"""
Locant assignment utilities for IUPAC nomenclature.

Locants map atom positions in the principal chain to 1-indexed IUPAC locant
numbers. This module provides:

1. build_atom_to_locant: Convert atom indices to locant mapping
2. compare_locant_sets: First-point-of-difference comparison
3. orient_chain: Apply IUPAC 2013 chain orientation criteria
4. get_functional_group_locants: Resolve FG atoms to chain locants

IUPAC 2013 chain orientation criteria (applied in order):
    a. Lowest locants for principal characteristic group
    b. Lowest locants for multiple bonds (as a set)
    c. Lowest locants for double bonds (if tie with triple bonds)
    d. Lowest locants for substituents (detachable prefixes)

Reference: IUPAC 2013 Blue Book, P-14.4, P-14.6, P-14.7
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union


from .lambda_convention import nonstandard_bonding_number

# ---------------------------------------------------------------------------
# Element-seniority order for numbering tie-breaks (DD4 /.
#
# Single source of truth for the IUPAC 2013 element-seniority sequence used in
# numbering decisions (P-15.4.1.2 / P-15.4.3.2.1 skeletal-replacement; P-44.3.3
# senior-acyclic-heteroatom; P-25.3.3.1.2(b) fused-ring). Lower rank = SENIOR
# (gets the lower locant when a positional set ties).
#
#   F < Cl < Br < I < At < O < S < Se < Te < Po < N < P < As < Sb < Bi
#     < C < Si < Ge < Sn < Pb < B < Al < Ga < In < Tl
#
# Before E1 there were 3+ divergent heteroatom-seniority tables
# (ring_substituents._HETEROATOM_SENIORITY, ring_selection._HETEROATOM_SENIORITY,
# fusion_descriptors.HETERO_PRIORITY). This constant is the canonical numbering
# table; the numbering consumer (ring_substituents._HETEROATOM_SENIORITY) is
# derived from it. ring_selection's table is a genuinely-different P-18
# ring-selection order (lock) and is intentionally left separate.
# Source: https://iupac.qmul.ac.uk/BlueBook/P1.html P-15.4.1.2; DD4.
# ---------------------------------------------------------------------------
_ELEMENT_NUMBERING_ORDER: List[str] = [
    'F', 'Cl', 'Br', 'I', 'At',
    'O', 'S', 'Se', 'Te', 'Po',
    'N', 'P', 'As', 'Sb', 'Bi',
    'C', 'Si', 'Ge', 'Sn', 'Pb',
    'B', 'Al', 'Ga', 'In', 'Tl',
]
ELEMENT_NUMBERING_SENIORITY: Dict[str, int] = {
    sym: rank for rank, sym in enumerate(_ELEMENT_NUMBERING_ORDER)
}
# Unknown elements sort AFTER every listed element (least senior) but
# deterministically among themselves (by symbol, applied by callers).
_ELEMENT_SENIORITY_DEFAULT: int = len(_ELEMENT_NUMBERING_ORDER)


def element_seniority_rank(symbol: str) -> int:
    """Numbering-seniority rank of an element symbol (lower = senior).

    Unlisted symbols return a sentinel rank past every listed element so they
    sort last in a deterministic, total order. Source: P-15.4.1.2 (DD4).
    """
    return ELEMENT_NUMBERING_SENIORITY.get(symbol, _ELEMENT_SENIORITY_DEFAULT)

# Phase 147 /: locant type system extension.
# Locants may be plain ints (e.g. 4) or (int_base, str_suffix) tuples for
# fusion atoms (e.g. '4a' -> (4, 'a')). The empty string '' sorts
# lexicographically before any letter, so (4, '') < (4, 'a') < (5, ''),
# matching the IUPAC convention that locant 4 is "lower" than 4a.
_Locant = Union[int, Tuple[int, str]]


def _assert_homogeneous_locants(locants: List[_Locant]) -> None:
    """Raise ValueError if ``locants`` contains a mix of int and tuple types.

    Per Phase 146: after coercion, a locant list must be uniformly
    int OR uniformly tuple. Mixed types indicate a caller bug (e.g.,
    ring_info populated only partially) and would cause Python's
    ``sorted()`` / ``min()`` to raise ``TypeError`` on mixed int/tuple
    comparison.

    Empty lists are vacuously homogeneous and return silently.

    Args:
        locants: List of int or (int, str) tuple locants.

    Raises:
        ValueError: if ``locants`` contains both int and tuple values.
                    The message includes the first offending int and the
                    first offending tuple to aid debugging.

    Source: https://iupac.qmul.ac.uk/BlueBook/P1.html P-14.7
            (locant set comparison semantics)
    Source: Phase 146 (locant type safety lock-in)
    """
    has_int = any(isinstance(x, int) for x in locants)
    has_tuple = any(isinstance(x, tuple) for x in locants)
    if has_int and has_tuple:
        first_int = next(x for x in locants if isinstance(x, int))
        first_tuple = next(x for x in locants if isinstance(x, tuple))
        raise ValueError(
            f"Locant list contains mixed int and tuple types: "
            f"{first_int!r} and {first_tuple!r}. "
            f"Coerce all to tuples or all to ints before passing to "
            f"compare_locant_sets."
        )


def build_atom_to_locant(principal_chain: List[int]) -> Dict[int, int]:
    """
    Create a mapping from RDKit atom indices to 1-indexed IUPAC locants.

    The locant numbering follows the order of the principal chain: the first
    atom in the chain gets locant 1, the second gets locant 2, etc.

    Args:
        principal_chain: Ordered list of atom indices forming the principal chain.
                         The ordering must already reflect the correct numbering
                         direction (see orient_chain).

    Returns:
        Dict mapping atom_idx -> locant (1-indexed).
        Empty dict if principal_chain is empty.

    Examples:
        >>> build_atom_to_locant([5, 3, 1, 0])
        {5: 1, 3: 2, 1: 3, 0: 4}
        >>> build_atom_to_locant([])
        {}
        >>> build_atom_to_locant([0])
        {0: 1}
    """
    return {atom_idx: locant for locant, atom_idx in enumerate(principal_chain, 1)}


def compare_locant_sets(
    set_a: List[_Locant],
    set_b: List[_Locant],
) -> int:
    """
    Compare two locant sets using IUPAC first-point-of-difference rule.

    This is NOT the sum-of-locants method. Both sets are sorted ascending,
    then compared term-by-term. The set with the lower value at the first
    point of difference is preferred.

    If all compared elements are equal, the shorter set wins (fewer locants
    needed means simpler name). If completely identical, returns 0.

    Phase 147 extension: also accepts ``List[Tuple[int, str]]``
    with first-point-of-difference semantics for fusion atoms. When one
    list contains tuples and the other contains ints, all ints are coerced
    to ``(n, '')`` tuples internally — empty string sorts before any
    letter, preserving the IUPAC convention that locant ``4`` is "lower"
    than locant ``4a``. Truly heterogeneous lists are rejected by
    ``_assert_homogeneous_locants`` (raises ``ValueError``).

    Args:
        set_a: First locant set (unsorted or sorted). Elements are int
               or ``(int, str)`` tuples; mixed allowed only if all ints
               can be coerced via the ``(n, '')`` padding.
        set_b: Second locant set (same type contract as set_a).

    Returns:
        -1 if set_a is preferred (lower at first difference)
         0 if sets are equal
         1 if set_b is preferred (lower at first difference)

    Examples:
        >>> compare_locant_sets([2, 3, 5], [3, 4, 6])
        -1
        >>> compare_locant_sets([2, 4, 5], [2, 3, 5])
        1
        >>> compare_locant_sets([2, 3], [2, 3])
        0
        >>> compare_locant_sets([(4, ''), (5, '')], [(4, 'a'), (5, '')])
        -1
        >>> compare_locant_sets([(4, 'a'), (5, '')], [(4, 'b'), (5, '')])
        -1

    Source: https://iupac.qmul.ac.uk/BlueBook/P1.html P-14.5.2, P-14.7
    Source: Phase 146 (locant type safety); Phase 147 /
    """
    # Phase 147: tuple-coercion entry path. If either list contains a
    # tuple locant, coerce all ints to (n, '') tuples in BOTH lists so
    # Python's sorted()/comparison operators stay type-safe. Pure-int
    # lists fall through to the back-compat fast path unchanged.
    any_tuple = (
        any(isinstance(x, tuple) for x in set_a)
        or any(isinstance(x, tuple) for x in set_b)
    )
    if any_tuple:
        a_coerced = [(x, '') if isinstance(x, int) else x for x in set_a]
        b_coerced = [(x, '') if isinstance(x, int) else x for x in set_b]
        _assert_homogeneous_locants(a_coerced)
        _assert_homogeneous_locants(b_coerced)
        a_sorted = sorted(a_coerced)
        b_sorted = sorted(b_coerced)
    else:
        a_sorted = sorted(set_a)
        b_sorted = sorted(set_b)

    for a, b in zip(a_sorted, b_sorted):
        if a < b:
            return -1  # set_a preferred
        if a > b:
            return 1   # set_b preferred

    # All compared elements equal -- shorter set wins
    if len(a_sorted) < len(b_sorted):
        return -1
    if len(a_sorted) > len(b_sorted):
        return 1

    return 0  # Identical


def _compare_heteroatom_seniority(
    a_pairs: List[Tuple[_Locant, str]],
    b_pairs: List[Tuple[_Locant, str]],
) -> int:
    """Compare two heteroatom (locant, element) lists for numbering.

    Implements the skeletal/replacement + fused-ring numbering rule:
      1. lowest heteroatom locant SET, kind-agnostic (P-15.4.3.2.1 /
         P-25.3.3.1.2(a)) — via ``compare_locant_sets`` on the positions;
      2. on a positional tie, the lower locant goes to the element highest in
         the element-seniority order (P-15.4.1.2 / P-25.3.3.1.2(b)) — compared
         element-by-element from most senior to least.

    Worked example (BlueBook P-15.4.1.2): ``2-oxa-4,6,8-trisilanonane`` — the
    locant sets ``{2,4,6,8}`` tie, so O (senior to Si) takes locant 2.

    Returns -1 if ``a`` is preferred, +1 if ``b``, 0 if genuinely equivalent.
    """
    a_locs = [loc for loc, _ in a_pairs]
    b_locs = [loc for loc, _ in b_pairs]
    positional = compare_locant_sets(a_locs, b_locs)
    if positional != 0:
        return positional

    # Positional tie -> element-seniority tie-break.
    a_by: Dict[str, List[_Locant]] = defaultdict(list)
    b_by: Dict[str, List[_Locant]] = defaultdict(list)
    for loc, sym in a_pairs:
        a_by[sym].append(loc)
    for loc, sym in b_pairs:
        b_by[sym].append(loc)

    # Iterate elements most-senior first; the first element whose locant set
    # differs decides (the senior element wants the lowest locants).
    elements = sorted(set(a_by) | set(b_by), key=lambda s: (element_seniority_rank(s), s))
    for sym in elements:
        result = compare_locant_sets(a_by.get(sym, []), b_by.get(sym, []))
        if result != 0:
            return result
    return 0


def compare_numbering(candidate_a: Dict, candidate_b: Dict) -> int:
    """Fully-ordered deterministic numbering tie-break (P-14.3.5 + P-14.4 layers).

    The single shared lowest-locant comparator routed through by all three
    numbering engines (skeletal-replacement, fused-ring/PAH, benzene ring) so
    that the same structure always yields the same numbering regardless of
    input SMILES order. It NEVER falls through to input order: when every active
    tier ties, the two numberings are genuinely equivalent (the molecule has
    that symmetry) and either is correct & byte-identical.

    Each candidate is a dict carrying only the tiers relevant to the call; a
    missing key means that tier is unconstrained (ties). Tiers, applied in the
    P-59.1.10 order:

      'heteroatoms'  list of (locant, element) — positional set    (P-59.1.10(b)
                     then element-seniority lowest-locant            / P-15.4.1.2)
      'indicated_h'  indicated-hydrogen locant set                  (P-59.1.10(c))
      'pcg'          principal-characteristic-group locant set      (P-59.1.10(d))
      'substituents' detachable-prefix locant set                   (P-14.4(f))
      'alpha'        sortable key giving the alphabetically-first    (P-14.4(g))
                     prefix the lowest locant

    Returns -1 if ``candidate_a`` is preferred, +1 if ``candidate_b``, 0 if
    every active tier ties. ``compare_locant_sets`` is reused unchanged as the
    positional primitive — this comparator is additive alongside it.
    """
    # Tier 1 — heteroatom set (P-59.1.10(b) / P-14.4(b)): positional, then
    # element seniority. Heteroatoms are part of the parent hydride and
    # outrank the suffix for numbering.
    a_het, b_het = candidate_a.get('heteroatoms'), candidate_b.get('heteroatoms')
    if a_het is not None or b_het is not None:
        result = _compare_heteroatom_seniority(a_het or [], b_het or [])
        if result != 0:
            return result

    # Tier 2 — indicated hydrogen (P-59.1.10(c)): lowest locants for
    # indicated hydrogen, before the principal-group suffix tier.
    a_ih, b_ih = candidate_a.get('indicated_h'), candidate_b.get('indicated_h')
    if a_ih is not None or b_ih is not None:
        result = compare_locant_sets(a_ih or [], b_ih or [])
        if result != 0:
            return result

    # Tier 3 — principal characteristic group (P-59.1.10(d) / P-14.4(d)).
    a_pcg, b_pcg = candidate_a.get('pcg'), candidate_b.get('pcg')
    if a_pcg is not None or b_pcg is not None:
        result = compare_locant_sets(a_pcg or [], b_pcg or [])
        if result != 0:
            return result

    # Tier 4 — detachable-substituent set (P-14.4(f)).
    a_sub, b_sub = candidate_a.get('substituents'), candidate_b.get('substituents')
    if a_sub is not None or b_sub is not None:
        result = compare_locant_sets(a_sub or [], b_sub or [])
        if result != 0:
            return result

    # Tier 5 — alphabetically-first prefix lowest locant (P-14.4(g)).
    a_alpha, b_alpha = candidate_a.get('alpha'), candidate_b.get('alpha')
    if a_alpha is not None or b_alpha is not None:
        if a_alpha is None:
            return 1
        if b_alpha is None:
            return -1
        if a_alpha < b_alpha:
            return -1
        if a_alpha > b_alpha:
            return 1

    return 0


def orient_chain(
    chain: List[int],
    mol,
    principal_group_atoms: Set[int],
    double_bonds: List[Tuple[int, int]],
    triple_bonds: List[Tuple[int, int]],
    substituent_positions: Optional[Dict[int, List]] = None,
) -> List[int]:
    """
    Orient the principal chain to produce the lowest IUPAC locant set.

    Applies IUPAC 2013 criteria in strict order:
        a. Lowest locants for principal characteristic group
        b. Lowest locants for multiple bonds (double + triple, as a set)
        c. Lowest locants for double bonds specifically
        d. Lowest locants for substituents (detachable prefixes)
        e. Lowest locant for alphabetically first substituent (P-14.4(g))

    For pure hydrocarbons (no principal group), criterion (a) is skipped.

    Args:
        chain: Ordered list of atom indices forming the candidate chain.
               This function evaluates forward vs reversed ordering.
        mol: RDKit Mol object (used to detect bonds on the chain).
        principal_group_atoms: Set of atom indices belonging to the principal
                                characteristic group (may be empty).
        double_bonds: List of (atom_idx, atom_idx) tuples for C=C bonds.
        triple_bonds: List of (atom_idx, atom_idx) tuples for C#C bonds.
        substituent_positions: Optional dict mapping chain atom index to list
                                of substituent groups. If None, criterion (d)
                                is skipped.

    Returns:
        The chain in the preferred orientation (may be the same list or
        reversed).
    """
    if len(chain) <= 1:
        return chain

    forward = chain
    reverse = list(reversed(chain))

    # Build locant maps for both directions
    fwd_map = build_atom_to_locant(forward)
    rev_map = build_atom_to_locant(reverse)

    chain_set = set(chain)

    # --- Criterion (a): Lowest locants for principal characteristic group ---
    if principal_group_atoms:
        pg_on_chain = principal_group_atoms & chain_set
        if not pg_on_chain:
            # P-14.4(a) / P-31.1.4: a principal group whose defining atoms are NOT
            # chain atoms (-SO3H / -PO3H2: the S/P and its O's hang OFF a chain
            # carbon) is located by the CHAIN CARBON that bears it. Without this,
            # criterion (a) was skipped for such groups and a mere substituent stole
            # C1 ('1-hydroxyethanesulfonic acid' instead of the PIN
            # '2-hydroxyethanesulfonic acid'). Anchor on the attachment carbon(s) so
            # the principal group still drives the lowest-locant rule. (Groups whose
            # match includes the chain carbon -- acids/ketones/alcohols/amines -- keep
            # a non-empty pg_on_chain above and are unaffected: byte-identical.)
            pg_on_chain = {
                a for a in chain
                if any(nb.GetIdx() in principal_group_atoms
                       for nb in mol.GetAtomWithIdx(a).GetNeighbors())
            }
        if pg_on_chain:
            fwd_locants = sorted(fwd_map[a] for a in pg_on_chain)
            rev_locants = sorted(rev_map[a] for a in pg_on_chain)
            result = compare_locant_sets(fwd_locants, rev_locants)
            if result == -1:
                return forward
            if result == 1:
                return reverse
            # result == 0 -> tie, continue to next criterion

    # --- Criterion (b): Lowest locants for multiple bonds (combined set) ---
    # Must compute bond locant atoms separately for forward and reverse chains,
    # since the "lower position" atom of each bond depends on chain direction
    all_bonds = double_bonds + triple_bonds
    fwd_bond_atoms = _get_bond_locant_atoms(forward, all_bonds)
    rev_bond_atoms = _get_bond_locant_atoms(reverse, all_bonds)
    if fwd_bond_atoms or rev_bond_atoms:
        fwd_locants = sorted(fwd_map[a] for a in fwd_bond_atoms)
        rev_locants = sorted(rev_map[a] for a in rev_bond_atoms)
        result = compare_locant_sets(fwd_locants, rev_locants)
        if result == -1:
            return forward
        if result == 1:
            return reverse

    # --- Criterion (c): Lowest locants for double bonds specifically ---
    fwd_double_atoms = _get_bond_locant_atoms(forward, double_bonds)
    rev_double_atoms = _get_bond_locant_atoms(reverse, double_bonds)
    if fwd_double_atoms or rev_double_atoms:
        fwd_locants = sorted(fwd_map[a] for a in fwd_double_atoms)
        rev_locants = sorted(rev_map[a] for a in rev_double_atoms)
        result = compare_locant_sets(fwd_locants, rev_locants)
        if result == -1:
            return forward
        if result == 1:
            return reverse

    # --- Criterion (d): Lowest locants for substituents ---
    if substituent_positions:
        # Substituent positions are already keyed by atom index
        sub_atoms = set(substituent_positions.keys()) & chain_set
        if sub_atoms:
            fwd_locants = sorted(fwd_map[a] for a in sub_atoms)
            rev_locants = sorted(rev_map[a] for a in sub_atoms)
            result = compare_locant_sets(fwd_locants, rev_locants)
            if result == -1:
                return forward
            if result == 1:
                return reverse

    # --- Criterion (d.5): P-14.4(h) nonstandard-valence atom lower locant ---
    # When the substituent locant SETS tie, the chain position bearing a
    # substituent whose attachment atom is in a NONSTANDARD (λ) valence state
    # takes the lower locant (BB P-14.4(h), BlueBookV2.md:3318,3334:
    # 'OC(C[PH4])CP' -> '1-(λ5-phosphanyl)-3-phosphanylpropan-2-ol'; the λ5 arm
    # is given C1). Runs BEFORE the alphanumerical criterion (e). Canonical:
    # driven by the perceived λ bonding number, not atom order.
    if substituent_positions:
        lam_chain_atoms = set()
        for c_idx in (set(substituent_positions.keys()) & chain_set):
            sub_atom_idxs: List[int] = []
            for s in substituent_positions[c_idx]:
                if isinstance(s, (list, tuple, set, frozenset)):
                    sub_atom_idxs.extend(s)
                else:
                    sub_atom_idxs.append(s)
            if any(nonstandard_bonding_number(mol, ai) is not None
                   for ai in sub_atom_idxs):
                lam_chain_atoms.add(c_idx)
        if lam_chain_atoms:
            fwd_locants = sorted(fwd_map[a] for a in lam_chain_atoms)
            rev_locants = sorted(rev_map[a] for a in lam_chain_atoms)
            result = compare_locant_sets(fwd_locants, rev_locants)
            if result == -1:
                return forward
            if result == 1:
                return reverse

    # --- Criterion (e): Lowest locant for alphabetically first substituent ---
    # P-14.4(g): when substituent locant sets are identical in both directions,
    # prefer the orientation giving the lowest locant to the first-cited prefix
    # (alphabetically first substituent).
    if substituent_positions:
        sub_atoms = set(substituent_positions.keys()) & chain_set
        if len(sub_atoms) >= 2:  # Need 2+ substituent positions for this to matter
            fwd_alpha = _alphabetical_tiebreaker(forward, fwd_map, substituent_positions, mol)
            rev_alpha = _alphabetical_tiebreaker(reverse, rev_map, substituent_positions, mol)
            if fwd_alpha < rev_alpha:
                return forward
            if rev_alpha < fwd_alpha:
                return reverse

    # --- Criterion (f): P-45.6.3 — 'R' before 'S' at first difference ---
    # Reached only when (a)-(e) all tie: the two orientations yield names
    # identical except for the stereodescriptor sequence (meso-type
    # symmetry). BB P-45.6.3 (BlueBookV2.md:22603). CIP labels are computed
    # on a COPY (never mutate the shared mol); guarded to molecules that
    # actually carry chiral tags so achiral chains stay byte-identical.
    from rdkit import Chem as _Chem
    if any(a.GetChiralTag() != _Chem.ChiralType.CHI_UNSPECIFIED
           for a in mol.GetAtoms()):
        try:
            from rdkit.Chem import rdCIPLabeler
            probe = _Chem.Mol(mol)
            rdCIPLabeler.AssignCIPLabels(probe)
            fwd_seq = _cip_sequence(forward, fwd_map, probe)
            rev_seq = _cip_sequence(reverse, rev_map, probe)
            if fwd_seq != rev_seq:
                return forward if fwd_seq < rev_seq else reverse
        except Exception:
            pass  # fail-closed to the deterministic forward fallback

    # All criteria tied -- return forward (arbitrary but deterministic)
    return forward


def get_functional_group_locants(
    chain: List[int],
    fg_atom_tuples: List[Tuple[int, ...]],
    atom_to_locant: Dict[int, int],
    mol=None,
) -> List[int]:
    """
    Resolve functional group SMARTS match atoms to chain locants.

    Given FG match tuples from SMARTS pattern matching, find the
    locant-defining atom for each match on the principal chain.

    The locant-defining atom is the carbon on the chain that bears the
    functional group -- e.g. the carbonyl carbon for ketones, the carbon
    bearing -OH for alcohols, the carboxyl carbon for acids.

    When an RDKit mol is provided, the function selects the on-chain
    carbon with the most bonds to heteroatoms (O, N, S, etc.) within the
    match.  This correctly handles SMARTS patterns where a neighbor carbon
    appears before the key carbon in the match tuple (e.g. the ketone
    pattern ``[#6][CX3](=O)[#6]``).

    Without a mol, falls back to picking the first on-chain atom.

    Args:
        chain: Ordered principal chain atom indices.
        fg_atom_tuples: List of atom index tuples from SMARTS matching.
                        Each tuple contains indices of atoms in one FG instance.
        atom_to_locant: Mapping from atom index to locant (from build_atom_to_locant).
        mol: Optional RDKit Mol object.  When provided, used to score
             candidate carbon atoms by their heteroatom connectivity.

    Returns:
        Sorted list of locants where the functional group attaches to the chain.
        Empty list if no FG atoms are on the chain.
    """
    chain_set = set(chain)
    locants = []

    for match_tuple in fg_atom_tuples:
        best_idx = _pick_locant_atom(match_tuple, chain_set, mol)
        if best_idx is not None and best_idx in atom_to_locant:
            locants.append(atom_to_locant[best_idx])

    return sorted(locants)


def _pick_locant_atom(
    match_tuple: Tuple[int, ...],
    chain_set: Set[int],
    mol=None,
) -> Optional[int]:
    """
    Pick the locant-defining atom from a SMARTS match tuple.

    Strategy:
        1. Collect all carbon atoms in the match that lie on the chain.
        2. If *mol* is available, score each candidate by the number of
           bonds it has to heteroatoms (non-C, non-H) within the same
           match tuple.  The atom with the highest score is the
           functional-group carbon.
        3. On tie (or when mol is unavailable), return the first
           candidate in match-tuple order.

    Returns:
        Atom index of the locant-defining atom, or ``None`` if no atom
        in the match is on the chain.
    """
    if mol is None:
        # Fallback: first atom on chain
        for atom_idx in match_tuple:
            if atom_idx in chain_set:
                return atom_idx
        return None

    match_set = set(match_tuple)
    candidates: List[int] = []
    for atom_idx in match_tuple:
        if atom_idx in chain_set:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetAtomicNum() == 6:  # carbon
                candidates.append(atom_idx)

    if not candidates:
        # No carbon *inside the match* lies on the chain. This is the case for
        # SMARTS that deliberately exclude the attachment carbon from the match
        # arity and assert it only via a recursive environment -- e.g. the
        # sulfinic / sulfonic oxoacid patterns
        #   '[SX3;$([SX3][#6])](=O)[OX2H1]' / '[SX4;$([SX4][#6])](=O)(=O)[OX2H1]'
        # match only (S, O, O[, O]); the chain carbon that BEARS the group is a
        # NEIGHBOR of a match atom, not a member of the match (the recursive
        # $(...) keeps the inorganic-oxoacid distinction). Walk the match atoms'
        # neighbors and return the chain carbon that bears the group so its
        # suffix locant is cited (P-14.3.4 / P-65.3.1: 'butane-2-sulfinic acid',
        # 'butane-2-sulfonic acid' -- not the locant-dropped 'butanesulfinic acid'
        # that mis-names a secondary-carbon attachment as a primary one).
        for atom_idx in match_tuple:
            for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx in chain_set and nbr.GetAtomicNum() == 6:
                    return nbr_idx
        # Last resort: any match atom on the chain (legacy behavior).
        for atom_idx in match_tuple:
            if atom_idx in chain_set:
                return atom_idx
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Score: count bonds to non-carbon atoms within the match
    def _hetero_score(idx: int) -> int:
        atom = mol.GetAtomWithIdx(idx)
        score = 0
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in match_set and nbr.GetAtomicNum() != 6:
                score += 1
        return score

    # Pick highest score; on tie, preserve match-tuple order
    best = max(candidates, key=_hetero_score)
    return best


def get_bond_locants(
    chain: List[int],
    bonds: List[Tuple[int, int]],
    atom_to_locant: Dict[int, int],
) -> List[int]:
    """
    Get locants for bonds (double or triple) within the chain.

    For each bond (atom_a, atom_b), both atoms must be in the chain.
    The locant is the LOWER of the two atom locants (per IUPAC convention,
    a bond between positions i and i+1 is cited by locant i).

    Args:
        chain: Ordered list of atom indices in the principal chain.
        bonds: List of (atom_idx_a, atom_idx_b) tuples for bonds.
        atom_to_locant: Mapping from atom index to locant (1-indexed).

    Returns:
        Sorted list of locants for the bonds.

    Examples:
        >>> chain = [0, 1, 2, 3]
        >>> bonds = [(1, 2)]  # bond between atoms 1 and 2
        >>> atom_to_locant = {0: 1, 1: 2, 2: 3, 3: 4}
        >>> get_bond_locants(chain, bonds, atom_to_locant)
        [2]
    """
    bond_atoms = _get_bond_locant_atoms(chain, bonds)
    locants = [atom_to_locant[atom_idx] for atom_idx in bond_atoms
               if atom_idx in atom_to_locant]
    return sorted(locants)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _alphabetical_tiebreaker(
    chain: List[int],
    atom_to_locant: Dict[int, int],
    substituent_positions: Dict[int, List],
    mol,
) -> tuple:
    """Build a comparison key for the alphabetical tiebreaker (criterion e).

    For each substituent attachment point on the chain, determine the
    substituent name (based on carbon count for simple alkyl groups) and
    its locant. Return a tuple of (alpha_sort_key, locant) pairs sorted
    by alpha_sort_key first, so that lexicographic comparison of tuples
    selects the orientation giving the lowest locant to the
    alphabetically first substituent.

    Args:
        chain: Ordered chain atom indices.
        atom_to_locant: Mapping from atom index to 1-indexed locant.
        substituent_positions: Dict mapping chain atom index to list of
            substituent atom groups.
        mol: RDKit Mol object.

    Returns:
        Tuple for lexicographic comparison. Lower = preferred.
    """
    from ..assembly.naming_utils import alpha_sort_key, get_alkyl_name

    chain_set = set(chain)
    entries = []

    for atom_idx, sub_groups in substituent_positions.items():
        if atom_idx not in chain_set:
            continue
        locant = atom_to_locant.get(atom_idx)
        if locant is None:
            continue

        # Count carbon atoms in the first substituent group at this position
        # (sufficient for the alphabetical comparison of simple alkyls)
        for sub_atoms in sub_groups:
            name = None
            # WS-A task 9 (P-14.5.2): a RING substituent must be compared by
            # its REAL cited prefix name, not a name fabricated from its
            # carbon count (phenyl is NOT 'hexyl', thiophen-2-yl is NOT
            # 'butyl' — the fabricated keys inverted the orientation of
            # 1-phenyl-4-(thiophen-2-yl)butane-1,4-dione). Derive the
            # attachment as the fragment atom bonded to the chain atom
            # (sub_atoms comes from a set — position 0 is arbitrary).
            if sub_atoms and any(
                    mol.GetAtomWithIdx(a).IsInRing() for a in sub_atoms):
                try:
                    attach_idx = next(
                        (a for a in sub_atoms
                         if any(nbr.GetIdx() == atom_idx
                                for nbr in mol.GetAtomWithIdx(a).GetNeighbors())),
                        None,
                    )
                    if attach_idx is not None:
                        from ..assembly.substituent_enumerator import (
                            name_substituent,
                        )
                        name = name_substituent(mol, list(sub_atoms), attach_idx)
                except Exception:
                    name = None
            if not name:
                carbon_count = sum(
                    1 for a in sub_atoms
                    if mol.GetAtomWithIdx(a).GetSymbol() == "C"
                )
                if carbon_count > 0:
                    name = get_alkyl_name(carbon_count)
                else:
                    # Non-carbon substituent (e.g., halogen): use atom symbol
                    if sub_atoms:
                        name = mol.GetAtomWithIdx(sub_atoms[0]).GetSymbol().lower()
                    else:
                        name = "zzz"
            entries.append((alpha_sort_key(name), locant))

    # Sort by alpha key first, then by locant
    entries.sort()
    # Return as flat tuple for lexicographic comparison:
    # the first entry's (alpha_key, locant) dominates
    return tuple(item for pair in entries for item in pair)


def _get_bond_locant_atoms(
    chain: List[int],
    bonds: List[Tuple[int, int]],
) -> Set[int]:
    """
    Find atoms on the chain that are part of specified bonds.

    For IUPAC locant purposes, a multiple bond between atoms at positions i
    and i+1 in the chain is cited by the lower locant (position i). This
    function returns the set of lower-numbered atoms (by chain position) for
    each bond that lies entirely on the chain.

    Args:
        chain: Ordered list of atom indices in the principal chain.
        bonds: List of (atom_idx_a, atom_idx_b) tuples for bonds.

    Returns:
        Set of atom indices (the lower-position atom of each on-chain bond).
    """
    chain_set = set(chain)
    # Build position lookup: atom_idx -> chain position (0-indexed)
    pos = {atom_idx: i for i, atom_idx in enumerate(chain)}
    result = set()

    for a, b in bonds:
        if a in chain_set and b in chain_set:
            # Return the atom with the lower chain position
            if pos[a] < pos[b]:
                result.add(a)
            else:
                result.add(b)

    return result


def _cip_sequence(chain: List[int], atom_to_locant: Dict[int, int],
                  labeled_mol) -> tuple:
    """P-45.6.3 key: (locant, CIP code) pairs for chain stereocentres,
    sorted by locant. 'R' < 'S' lexicographically, so tuple comparison
    implements 'alphabetic order of the stereochemical descriptors'."""
    pairs = []
    for a in chain:
        atom = labeled_mol.GetAtomWithIdx(a)
        if atom.HasProp('_CIPCode'):
            pairs.append((atom_to_locant[a], atom.GetProp('_CIPCode')))
    pairs.sort()
    return tuple(pairs)
