"""
Stereochemistry rules - IUPAC stereodescriptor collection and formatting.

This module handles the mapping from RDKit stereochemistry (indexed by atom
indices) to IUPAC nomenclature (indexed by locants on the principal chain/ring).

Key functions:
- collect_stereodescriptors: Get R/S and E/Z descriptors with IUPAC locants
- format_stereodescriptor_string: Format as "(2R,3S)-" prefix
- get_double_bond_locant: Get lower locant for E/Z double bond

Ring junction stereochemistry functions:
- get_bridgehead_atoms: Find ring fusion stereocenter atoms
- collect_ring_junction_stereo: Collect junction stereo with 'a' suffix locants
- format_ring_junction_stereo: Format as "(4aR,8aS)-" or "(4ar,8ac)-"
- determine_simple_cis_trans: Return "cis" or "trans" for simple bicyclics

IMPORTANT: This module requires atom_to_locant mapping to be provided by the
caller (computed during chain/ring classification). It does NOT fall back to
atom indices as those are NOT valid IUPAC locants.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Union

from rdkit import Chem
from ..perception.stereo import assign_stereochemistry, detect_axial_chirality

logger = logging.getLogger(__name__)


def _composite_locant_sort_key(
    item: Tuple[Union[int, str], str],
) -> Tuple[int, str]:
    """Sort key for descriptor lists that may mix int and '<int><letter>' locants.

    Mirrors _junction_locant_sort_key (line ~727, Phase 130) so that the
    universal collector (collect_stereodescriptors, line ~33) and the
    ring-junction collector (collect_ring_junction_stereo, line ~680) share
    one sort discipline.

    Per IUPAC P-91.1, composite locants like '3a' sort BETWEEN integer 3
    and integer 4 (after 3 because '' < 'a' in tuple-lex comparison).

    Phase 153 D-03 -- single source of truth for descriptor ordering. Per
    Pitfall 2, branches on isinstance(locant, int) FIRST so int input does
    NOT fall into the int('3'[:-1]) = int('') ValueError trap.

    Behaviour on existing int-only locants (regression invariant):
    byte-identical sort order to lambda x: x[0]. Verified by
    TestStereoBackstopRegressionInvariant in
    tests/unit/rules/test_handler_stereo_injection.py.
    """
    locant = item[0]
    if isinstance(locant, int):
        return (locant, '')
    # str path -- '3a' / '7a' / '12b'
    if locant and locant[-1].isalpha():
        return (int(locant[:-1]), locant[-1])
    return (int(locant), '')


def collect_stereodescriptors(
    mol,
    atom_to_locant: Dict[int, int],
    include_near_parent_ez: bool = False
) -> List[Tuple[int, str]]:
    """
    Collect all stereodescriptors from a molecule using IUPAC locants.

    Args:
        mol: RDKit Mol object (stereochemistry should already be assigned)
        atom_to_locant: Mapping from atom index to IUPAC locant number.
                       Only atoms in this mapping are considered (principal
                       chain/ring atoms).
        include_near_parent_ez: If True, also collect E/Z bonds that are
                       one hop away from the parent (i.e., neither bond atom
                       is in atom_to_locant, but one has a neighbor that is).
                       Per IUPAC P-93.5.2, substituent E/Z attached directly
                       to the parent should be reported. Only enable for
                       top-level naming, not decomposition fragments.

    Returns:
        List of (locant, cip_code) tuples, sorted by locant ascending.
        cip_code is 'R', 'S', 'r', 's' (lowercase for pseudoasymmetric),
        or 'E', 'Z' for double bonds.

    Example:
        >>> mol = Chem.MolFromSmiles('C[C@H](O)CC')
        >>> rdCIPLabeler.AssignCIPLabels(mol)
        >>> atom_to_locant = {0: 4, 1: 3, 3: 2, 4: 1}  # chain oriented
        >>> collect_stereodescriptors(mol, atom_to_locant)
        [(3, 'R')]
    """
    # Ensure stereochemistry is assigned (idempotent guard)
    assign_stereochemistry(mol)

    descriptors: List[Tuple[int, str]] = []

    # Collect R/S stereocenters (atom-based)
    for atom in mol.GetAtoms():
        if atom.HasProp('_CIPCode'):
            atom_idx = atom.GetIdx()
            # Only include atoms on principal chain/ring
            if atom_idx in atom_to_locant:
                locant = atom_to_locant[atom_idx]
                cip_code = atom.GetProp('_CIPCode')
                # Preserve CIP code as-is: R/S for normal stereocenters,
                # r/s for pseudoasymmetric centers per IUPAC P-92.1.4.2.
                descriptors.append((locant, cip_code))

    # Collect E/Z double bonds (bond-based)
    for bond in mol.GetBonds():
        if bond.HasProp('_CIPCode'):
            # Skip ring-constrained double bonds in small rings: double bonds
            # in rings of size 7 or fewer have geometry fixed by ring strain.
            # Macrocyclic rings (8+ members) CAN have meaningful E/Z geometry
            # per P-31.1.3 errata (Sep 2024).
            if bond.IsInRing():
                ri = mol.GetRingInfo()
                min_ring_size = float('inf')
                for ring in ri.BondRings():
                    if bond.GetIdx() in ring:
                        min_ring_size = min(min_ring_size, len(ring))
                if min_ring_size < 8:
                    continue

            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()

            # Determine locant for this E/Z bond.
            # Standard case: both atoms in atom_to_locant -> use lower locant.
            # Exocyclic case (IUPAC P-91.2): one atom in a ring that is
            # in atom_to_locant, the other outside the ring -> use the
            # ring atom's locant.  Only applies to true ring-exocyclic
            # bonds (prevents false E/Z on chain substituent bonds).
            if begin_idx in atom_to_locant and end_idx in atom_to_locant:
                # Both in parent -- standard behaviour
                locant = min(atom_to_locant[begin_idx],
                             atom_to_locant[end_idx])
            elif begin_idx in atom_to_locant and end_idx not in atom_to_locant:
                # Only include if the in-mapping atom is in a ring
                # (true exocyclic bond, not a chain substituent bond)
                if not mol.GetAtomWithIdx(begin_idx).IsInRing():
                    continue
                locant = atom_to_locant[begin_idx]
            elif end_idx in atom_to_locant and begin_idx not in atom_to_locant:
                if not mol.GetAtomWithIdx(end_idx).IsInRing():
                    continue
                locant = atom_to_locant[end_idx]
            else:
                # Neither atom in parent.
                if not include_near_parent_ez:
                    continue
                # One-hop case (IUPAC P-93.5.2): E/Z bonds in
                # substituents attached directly to the parent should
                # be reported, referencing the parent locant at the
                # attachment point.  Example: a styrenyl substituent
                # on a ring has C=C entirely off the ring, but one
                # bond atom is directly bonded to a ring atom that
                # IS in atom_to_locant.
                locant = None
                for idx in (begin_idx, end_idx):
                    atom = mol.GetAtomWithIdx(idx)
                    for nbr in atom.GetNeighbors():
                        nbr_idx = nbr.GetIdx()
                        if nbr_idx in atom_to_locant:
                            candidate = atom_to_locant[nbr_idx]
                            if locant is None or (isinstance(candidate, int) and
                                                  isinstance(locant, int) and
                                                  candidate < locant):
                                locant = candidate
                if locant is None:
                    continue

            cip_code = bond.GetProp('_CIPCode')  # 'E' or 'Z'
            descriptors.append((locant, cip_code))

    # Collect axial chirality (Ra/Sa) -- atropisomers and allenes (IUPAC P-93.5)
    axial_elements = detect_axial_chirality(mol)
    for element in axial_elements:
        cip = element.get('cip')
        if cip is None:
            continue  # Undetermined chirality -- skip
        locant_atom = element.get('locant_atom')
        if locant_atom is not None and locant_atom in atom_to_locant:
            locant = atom_to_locant[locant_atom]
            descriptors.append((locant, cip))
        # If locant_atom not in atom_to_locant, the axial chirality element
        # is not on the principal chain/ring -- skip (same filtering as R/S)

    # Sort by locant ascending. D-03: use the composite-locant safe key so
    # mixed int / '<int><letter>' (e.g. '3a', '7a' from ComplexRingResult.
    # atom_to_locant per fused_rings.py:317) sort per IUPAC P-91.1
    # ('3a' BETWEEN integer 3 and 4). Byte-identical to the pre-153
    # `lambda x: x[0]` for int-only inputs (locked by
    # TestStereoBackstopRegressionInvariant in tests/unit/rules/
    # test_handler_stereo_injection.py).
    descriptors.sort(key=_composite_locant_sort_key)

    # Filter out stereo descriptors with invalid locants (locant 0 or > parent size)
    if descriptors and atom_to_locant:
        int_locants = [v for v in atom_to_locant.values() if isinstance(v, int)]
        parent_size = max(int_locants) if int_locants else 0
        if parent_size > 0:
            from .locant_validation import validate_stereo_locants
            descriptors = validate_stereo_locants(descriptors, parent_size)

    return descriptors


def get_double_bond_locant(
    bond,
    atom_to_locant: Dict[int, int]
) -> Optional[int]:
    """
    Get the IUPAC locant for a double bond.

    Per IUPAC convention, the locant is the lower of the two atom locants.

    Args:
        bond: RDKit Bond object
        atom_to_locant: Mapping from atom index to IUPAC locant

    Returns:
        Lower locant of the two bond atoms, or None if either atom is
        not in the mapping (not on principal chain/ring).

    Example:
        >>> mol = Chem.MolFromSmiles('C/C=C/C')
        >>> bond = mol.GetBondWithIdx(1)  # the C=C bond
        >>> atom_to_locant = {0: 1, 1: 2, 2: 3, 3: 4}
        >>> get_double_bond_locant(bond, atom_to_locant)
        2
    """
    begin_idx = bond.GetBeginAtomIdx()
    end_idx = bond.GetEndAtomIdx()

    # Both atoms must be in mapping
    if begin_idx not in atom_to_locant or end_idx not in atom_to_locant:
        return None

    locant_a = atom_to_locant[begin_idx]
    locant_b = atom_to_locant[end_idx]

    return min(locant_a, locant_b)


def format_stereodescriptor_string(
    descriptors: List[Tuple[int, str]]
) -> str:
    """
    Format stereodescriptors as an IUPAC name prefix.

    STEREO-04: Multiple descriptors in single block with comma separation.

    Args:
        descriptors: List of (locant, cip_code) tuples, sorted by locant.

    Returns:
        Formatted string like "(2R)-", "(2R,3S)-", "(2E,3R,5Z)-"
        Returns empty string if no descriptors.

    Examples:
        >>> format_stereodescriptor_string([])
        ''
        >>> format_stereodescriptor_string([(2, 'R')])
        '(2R)-'
        >>> format_stereodescriptor_string([(2, 'R'), (3, 'S')])
        '(2R,3S)-'
        >>> format_stereodescriptor_string([(2, 'E'), (3, 'R'), (5, 'Z')])
        '(2E,3R,5Z)-'
        >>> format_stereodescriptor_string([(2, 'r'), (3, 's')])
        '(2r,3s)-'
    """
    if not descriptors:
        return ""

    # Build comma-separated list of "locantCIP"
    parts = [f"{locant}{cip}" for locant, cip in descriptors]

    return f"({','.join(parts)})-"


# ---------------------------------------------------------------------------
# Phase 152: Handler-level stereo injection (D-01 / D-04 predicate-first wiring)
# ---------------------------------------------------------------------------

# D-03: detection regex set REUSED VERBATIM from namer.py:62-75. DO NOT BROADEN.
# Pattern A — prefix form (already-stereoed name; matched against name start)
_STEREO_PREFIX_RE = re.compile(
    # Phase 153 D-03 / Pitfall 8: accept composite locants like '7a', '3a'
    # in addition to plain digits, so an already-stereoed name like
    # `(7aS)-2,3a-dichloro-...benzofuran-...` is correctly recognized as
    # already-stereoed and the injector does not double-emit the prefix.
    # Compatible with the existing `(2R)-` / `(R)-` / `(E)-` / `(2R,3S)-`
    # / `(2r,3s)-` matches (the locant prefix is optional via \d*).
    r'\(\d*[a-z]?[RSrsEZez](,\d*[a-z]?[RSrsEZez])*\)-'
)
# Pattern B — embedded block (descriptor block anywhere in the name body)
_STEREO_EMBEDDED_RE = re.compile(
    # Phase 153 D-03: also accept composite locants like '7a', '3a' in
    # embedded blocks (e.g., 'something-(3aR,7aS)-else').
    r'\(\d*[a-z]?[RSEZrsez](,\d*[a-z]?[RSEZrsez])*\)'
)
# Pattern C — carbohydrate / amino-acid traditional notation
_CARBOHYDRATE_STEREO_RE = re.compile(r'(alpha|beta|alfa)-[DL]-', re.IGNORECASE)

# BBR-GATE (Phase 169.7): a LEADING stereo / relative-configuration descriptor-block
# matcher for strip_stereo. Mirrors  (the
# validation precedent). Matches a leading (...)- block whose contents are PURELY
# stereo descriptors (digits, optional composite-locant letter, r/s/e/z/R/S/E/Z/*,
# RS/SR, commas/hyphens/+/space), OR a bare rel-/rac-/cis-/trans-/(±)- prefix. A
# substituent enclosing group like "(2-chloroethyl)-" does NOT match (its content has
# non-descriptor letters). Deliberately SEPARATE from _STEREO_PREFIX_RE (do NOT broaden
# that one — Phase 153 D-03).
_STRIP_STEREO_LEADING_RE = re.compile(
    r"^(\((?:[0-9]+[a-z]?[rsezRSEZ*]|[rsezRSEZ]|RS|SR|[,\-+ ])+\)|rel|rac|cis|trans|\(±\))-",
    re.IGNORECASE,
)


def strip_stereo(name: str) -> str:
    """Return *name* with leading stereo / relative-config descriptor blocks removed
    (to a fixpoint) — the BBR-GATE "where does OPSIN fail" probe (CONTEXT D-05).

    READ-ONLY: this is NOT a postprocessor on shipped names. The validity gate uses it
    only to test whether a name's CONSTITUTIONAL (stereo-stripped) form parses; the
    SHIPPED name keeps its stereo. Strips leading ``(2R)-`` / ``(1s,4s)-`` / ``(E)-`` /
    ``rel-`` / ``rac-`` / ``cis-`` / ``trans-`` / ``(±)-`` blocks; leaves a name with no
    leading descriptor (``hexane``) and substituent enclosing groups (``(2-chloroethyl)``)
    untouched.
    """
    if not name:
        return name
    s = name.strip()
    prev = None
    while prev != s:
        prev = s
        s = _STRIP_STEREO_LEADING_RE.sub("", s).strip()
    return s


def needs_stereo_injection(mol, name: str) -> bool:
    """Return True iff *mol* carries CIP stereo not represented in *name*.

    Pure read-only predicate (D-05) used both by the namer.py backstop
    (after Phase 152 refactor) and by inject_stereo_from_locant_map.

    Per D-03, the three name-side detection patterns are byte-identical to
    namer.py:_final_stereo_check lines 62-75. Per D-20, do NOT broaden.

    Args:
        mol: RDKit Mol object (may be None).
        name: Generated IUPAC name string (may be empty / 'unknown').

    Returns:
        True iff (a) mol is not None and name is non-empty and != 'unknown'
        AND (b) name does NOT match any of the three stereo-detection patterns
        AND (c) mol has at least one atom or bond with _CIPCode set after
        idempotent assign_stereochemistry.
    """
    if mol is None or not name or name == 'unknown':
        return False

    # Pattern A — prefix form (already-stereoed name; D-03)
    if _STEREO_PREFIX_RE.match(name):
        return False
    # Pattern B — embedded block
    if _STEREO_EMBEDDED_RE.search(name):
        return False
    # Pattern C — carbohydrate / amino acid traditional notation
    if _CARBOHYDRATE_STEREO_RE.search(name):
        return False

    # Idempotent CIP assignment (D-05 read-only — assign_stereochemistry uses
    # _Orthonym_CIPAssigned marker so re-entry is cheap and side-effect free
    # beyond what RDKit's CIP labeler already does).
    assign_stereochemistry(mol)
    has_atom_stereo = any(a.HasProp('_CIPCode') for a in mol.GetAtoms())
    has_bond_stereo = any(b.HasProp('_CIPCode') for b in mol.GetBonds())
    return has_atom_stereo or has_bond_stereo


def inject_stereo_from_locant_map(
    name: str,
    mol,
    atom_to_locant: Optional[Dict[int, int]],
    *,
    include_near_parent_ez: bool = True,
) -> str:
    """Prepend a P-91 stereo descriptor block to *name* using authoritative locants.

    Per IUPAC P-91 / P-91.1, prepends a `(R/S/E/Z)-` block built from
    collect_stereodescriptors + format_stereodescriptor_string.

    Per D-09, **no atom-index fallback**: when atom_to_locant is None, empty,
    or all-zero (degenerate), returns *name* unchanged and emits a single
    DEBUG log line. The backstop in namer.py will still log WARNING in this
    case so handler attribution is preserved.

    Args:
        name: The candidate IUPAC name from a handler.
        mol: RDKit Mol object with stereo info.
        atom_to_locant: Authoritative {atom_idx: 1-indexed locant} map from
            the handler's own perception (heterocycle / benzene / cycloalkane
            / cycloalkene). Must NOT be derived from raw atom indices (D-09).
        include_near_parent_ez: When True (default -- preserves benzene /
            heterocycle Tier-A behaviour), exocyclic E/Z bonds one hop from
            the parent are attributed to the lowest neighbouring locant per
            P-93.5.2. When False (cycloalkane / cycloalkene caller post-
            BL-02 fix), exocyclic E/Z bonds are NOT attributed to ring
            locants; only ring-atom R/S and ring-bond E/Z are emitted. This
            is the conservative gate per D-09 ("better a missing stereo
            block than a wrong one") for handlers where exocyclic E/Z can
            be mis-attributed via include_near_parent_ez=True.

    Returns:
        name unchanged (predicate False / no locant map / no descriptors)
        OR  prefix + name where prefix is e.g. '(2R)-', '(2R,3S)-',
        '(2E,3R,5Z)-', '(2r,3s)-' per P-91.

    Example:
        >>> mol = Chem.MolFromSmiles('C[C@@H](O)CC')
        >>> rdCIPLabeler.AssignCIPLabels(mol)
        >>> inject_stereo_from_locant_map('butan-2-ol', mol, {1: 2})
        '(2R)-butan-2-ol'
    """
    if not needs_stereo_injection(mol, name):
        return name

    # D-09: hard precondition — no atom-index fallback.
    # WR-02 fix (Phase 152-02, 2026-05-03): also reject all-zero / non-positive
    # locant maps. A locant of 0 or negative is IUPAC-malformed (locants are
    # 1-indexed); accepting it would emit '(0R)-name' or '(-1R)-name' garbage.
    # Per D-09 ("missing > wrong"), skip injection.
    if not atom_to_locant or not any(
        isinstance(v, int) and v > 0 for v in atom_to_locant.values()
    ):
        logger.debug(
            "inject_stereo: skipped, no locant map / all-zero locants (name=%r)",
            name[:50],
        )
        return name

    # D-11: include_near_parent_ez defaults to True for P-93.5.2 compliance
    # (top-level only; is_top_level_naming guard at the call site enforces
    # this). BL-02 fix (Phase 152-02, 2026-05-03): cycloalkane / cycloalkene
    # caller passes include_near_parent_ez=_ring_is_whole_molecule so chain-
    # side exocyclic E/Z is NOT attributed to ring locants.
    descriptors = collect_stereodescriptors(
        mol, atom_to_locant, include_near_parent_ez=include_near_parent_ez,
    )
    if not descriptors:
        return name

    prefix = format_stereodescriptor_string(descriptors)
    return f"{prefix}{name}"


def _ring_atom_to_locant_from_oriented(oriented_ring: List[int]) -> Dict[int, int]:
    """Build atom-idx → 1-indexed locant map from oriented_ring.

    Single source of truth for the {idx: pos+1} formula previously inlined
    at composer.py:7184. Used by benzene / cycloalkane / cycloalkene
    handler wiring per D-08.

    Args:
        oriented_ring: List of atom indices in IUPAC ring-position order
            (position 1 first).

    Returns:
        Dict mapping each atom idx to its 1-indexed locant. When duplicate
        atom indices appear, the LAST occurrence wins (matches dict semantics
        of the original one-liner at composer.py:7184). WR-03 fix
        (Phase 152-02, 2026-05-03): also emits a WARNING log when duplicates
        are present so a buggy upstream orientator does not silently produce
        a wrong locant map.

    Example:
        >>> _ring_atom_to_locant_from_oriented([10, 11, 12, 13, 14, 15])
        {10: 1, 11: 2, 12: 3, 13: 4, 14: 5, 15: 6}
    """
    if oriented_ring and len(set(oriented_ring)) != len(oriented_ring):
        logger.warning(
            "_ring_atom_to_locant_from_oriented: duplicate atom indices in "
            "oriented_ring=%r -- last position wins (upstream orientator may have a bug)",
            oriented_ring,
        )
    return {idx: pos + 1 for pos, idx in enumerate(oriented_ring)}


def collect_ring_stereodescriptors(
    mol,
    ring_atom_to_locant: Dict[int, int]
) -> List[Tuple[int, str]]:
    """
    Collect stereodescriptors for ring compounds.

    Same as collect_stereodescriptors but specifically for ring naming,
    where only atoms in the ring are considered.

    Args:
        mol: RDKit Mol object (stereochemistry should already be assigned)
        ring_atom_to_locant: Mapping from ring atom indices to IUPAC locants.
                            Only atoms that are keys in this dict are included.

    Returns:
        List of (locant, cip_code) tuples, sorted by locant ascending.

    Note:
        For rings, double bond E/Z is less common (ring strain), but
        we still handle it for completeness.
    """
    # Reuse the main function - it already handles the filtering correctly
    return collect_stereodescriptors(mol, ring_atom_to_locant)


def determine_ring_cis_trans(
    mol,
    ring_atoms: List[int],
    sub1_idx: int,
    sub2_idx: int
) -> Optional[str]:
    """
    Determine if two substituents on a ring are cis or trans.

    For a ring with exactly 2 substituents at specified positions,
    determine their relative stereochemistry based on CIP labels.

    The rule for 1,2-disubstituted rings:
    - SAME CIP codes (R,R or S,S) -> CIS (substituents on same face)
    - DIFFERENT CIP codes (R,S or S,R) -> TRANS (substituents on opposite faces)

    This is because in a ring, atoms with the same absolute configuration
    at adjacent positions have their substituents on the same face.

    Args:
        mol: RDKit Mol object (CIP labels should already be assigned)
        ring_atoms: List of atom indices that form the ring
        sub1_idx: Atom index of first substituted ring carbon
        sub2_idx: Atom index of second substituted ring carbon

    Returns:
        'cis' or 'trans', or None if cannot be determined (missing CIP labels)

    Example:
        >>> mol = Chem.MolFromSmiles('C[C@H]1CCCC[C@@H]1C')  # cis-1,2-dimethylcyclohexane
        >>> rdCIPLabeler.AssignCIPLabels(mol)
        >>> ring_atoms = [1, 2, 3, 4, 5, 6]
        >>> determine_ring_cis_trans(mol, ring_atoms, 1, 6)
        'cis'
    """
    # Both atoms must be in the ring
    if sub1_idx not in ring_atoms or sub2_idx not in ring_atoms:
        return None

    # Get atoms
    atom1 = mol.GetAtomWithIdx(sub1_idx)
    atom2 = mol.GetAtomWithIdx(sub2_idx)

    # Both must have CIP labels
    if not atom1.HasProp('_CIPCode') or not atom2.HasProp('_CIPCode'):
        return None

    cip1 = atom1.GetProp('_CIPCode')
    cip2 = atom2.GetProp('_CIPCode')

    # Only handle R/S (not r/s pseudoasymmetric for now)
    if cip1 not in ['R', 'S'] or cip2 not in ['R', 'S']:
        return None

    # Same CIP = cis, Different CIP = trans
    if cip1 == cip2:
        return 'cis'
    else:
        return 'trans'


def get_simple_ring_stereo(
    mol,
    ring_atoms: List[int],
    ring_substituents: Dict[int, str]
) -> Optional[str]:
    """
    Get cis/trans prefix for simple disubstituted rings.

    This function handles the common case of exactly 2 substituted positions
    on a ring, returning a cis- or trans- prefix for the name.

    Note: This is a simplification. More complex rings (3+ substituents)
    would need the full IUPAC r/c/t reference system, which is deferred.

    Args:
        mol: RDKit Mol object (CIP labels should already be assigned)
        ring_atoms: List of atom indices forming the ring
        ring_substituents: Dict mapping ring atom locant -> substituent name.
                          Only keys (locants) are used to identify substituted positions.

    Returns:
        'cis-' or 'trans-' prefix string, or None if:
        - Not exactly 2 substituted positions
        - Cannot determine stereochemistry

    Example:
        >>> mol = Chem.MolFromSmiles('C[C@H]1CCCC[C@@H]1C')
        >>> rdCIPLabeler.AssignCIPLabels(mol)
        >>> ring_atoms = [1, 2, 3, 4, 5, 6]
        >>> # Assuming oriented_ring maps locant -> atom_idx
        >>> ring_substituents = {1: 'methyl', 2: 'methyl'}
        >>> get_simple_ring_stereo(mol, ring_atoms, ring_substituents)
        'cis-'
    """
    # Need exactly 2 substituted positions for simple cis/trans
    if len(ring_substituents) != 2:
        return None

    # Get the locants with substituents
    locants = list(ring_substituents.keys())

    # We need to find which atom indices correspond to these locants
    # This requires knowing the atom_to_locant mapping
    # Since we don't have it directly, we'll need the caller to provide
    # the actual atom indices

    # For now, return None and let the caller handle the mapping
    # This function needs the actual atom indices, not locants
    return None


def get_simple_ring_stereo_from_atoms(
    mol,
    ring_atoms: List[int],
    substituted_atom_indices: List[int]
) -> Optional[str]:
    """
    Get cis/trans prefix given the actual atom indices of substituted positions.

    Args:
        mol: RDKit Mol object (CIP labels should already be assigned)
        ring_atoms: List of atom indices forming the ring
        substituted_atom_indices: List of exactly 2 atom indices that have substituents

    Returns:
        'cis-' or 'trans-' prefix string, or None if cannot determine

    Example:
        >>> mol = Chem.MolFromSmiles('C[C@H]1CCCC[C@@H]1C')
        >>> rdCIPLabeler.AssignCIPLabels(mol)
        >>> ring_atoms = [1, 2, 3, 4, 5, 6]
        >>> get_simple_ring_stereo_from_atoms(mol, ring_atoms, [1, 6])
        'cis-'
    """
    if len(substituted_atom_indices) != 2:
        return None

    result = determine_ring_cis_trans(
        mol,
        ring_atoms,
        substituted_atom_indices[0],
        substituted_atom_indices[1]
    )

    if result:
        return f'{result}-'
    return None


def format_ring_stereo_with_descriptors(
    ring_cis_trans: Optional[str],
    descriptors: List[Tuple[int, str]]
) -> str:
    """
    Format ring stereo prefix with optional R/S descriptors.

    IUPAC format for ring stereo:
    - If only cis/trans: "cis-1,2-dimethylcyclohexane"
    - If cis/trans + R/S: "cis-(1R,2S)-1,2-dimethylcyclohexane"
    - The cis/trans goes BEFORE the stereodescriptors

    Args:
        ring_cis_trans: 'cis-' or 'trans-' prefix, or None
        descriptors: List of (locant, cip_code) tuples from collect_stereodescriptors

    Returns:
        Combined prefix string like "cis-", "trans-(1R,2S)-", etc.
        Empty string if no stereo information.

    Example:
        >>> format_ring_stereo_with_descriptors('cis-', [(1, 'R'), (2, 'S')])
        'cis-(1R,2S)-'
        >>> format_ring_stereo_with_descriptors('trans-', [])
        'trans-'
        >>> format_ring_stereo_with_descriptors(None, [(1, 'R')])
        '(1R)-'
    """
    rs_string = format_stereodescriptor_string(descriptors)

    if ring_cis_trans:
        if rs_string:
            return f'{ring_cis_trans}{rs_string}'
        else:
            return ring_cis_trans
    else:
        return rs_string


# =============================================================================
# Ring Junction Stereochemistry Functions
# =============================================================================

def get_bridgehead_atoms(mol) -> List[int]:
    """
    Find atoms at ring fusion points (bridgehead atoms).

    Bridgehead atoms are atoms that are shared by multiple rings. For fused
    ring systems like decalin, these are the atoms at the junction where
    rings meet.

    Args:
        mol: RDKit Mol object

    Returns:
        List of atom indices that are bridgehead atoms (in 2+ rings and sp3)

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CCC2CCCCC2C1')  # decalin
        >>> bridgeheads = get_bridgehead_atoms(mol)
        >>> len(bridgeheads)  # 2 bridgehead atoms
        2
        >>> mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')  # naphthalene (aromatic)
        >>> bridgeheads = get_bridgehead_atoms(mol)
        >>> len(bridgeheads)  # 0 - sp2 atoms are not stereocenters
        0
    """
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    if len(atom_rings) < 2:
        return []

    # Count how many rings each atom belongs to
    atom_ring_count: Dict[int, int] = {}
    for ring in atom_rings:
        for atom_idx in ring:
            atom_ring_count[atom_idx] = atom_ring_count.get(atom_idx, 0) + 1

    # Find atoms in 2+ rings that are sp3 (potential stereocenters)
    bridgehead_atoms = []
    for atom_idx, count in atom_ring_count.items():
        if count >= 2:
            atom = mol.GetAtomWithIdx(atom_idx)
            # Check if sp3 (saturated) - these can be stereocenters
            # Hybridization SP3 indicates tetrahedral geometry
            hybridization = atom.GetHybridization()
            if hybridization == Chem.HybridizationType.SP3:
                bridgehead_atoms.append(atom_idx)

    return sorted(bridgehead_atoms)


def collect_ring_junction_stereo(
    mol,
    bridgehead_atoms: List[int],
    atom_to_locant: Dict[int, Union[int, str]]
) -> List[Tuple[str, str]]:
    """
    Collect stereodescriptors for ring junction (bridgehead) atoms.

    Ring junction atoms in fused systems use 'a' suffix locants in IUPAC naming.
    For example, in decahydronaphthalene, the junction atoms are 4a and 8a.

    Args:
        mol: RDKit Mol object (CIP labels should already be assigned)
        bridgehead_atoms: List of atom indices at ring junctions
        atom_to_locant: Mapping from atom index to IUPAC locant (int or str)
                       The locant should already include 'a' suffix if needed

    Returns:
        List of (locant_str, cip_code) tuples, sorted by locant.
        Locant is string to handle 'a' suffix (e.g., '4a', '8a').

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@@H]2C1')  # cis-decalin
        >>> rdCIPLabeler.AssignCIPLabels(mol)
        >>> bridgeheads = get_bridgehead_atoms(mol)
        >>> atom_to_locant = {3: '4a', 8: '8a'}  # junction atoms with 'a' suffix
        >>> stereo = collect_ring_junction_stereo(mol, bridgeheads, atom_to_locant)
        >>> # Returns [('4a', 'S'), ('8a', 'S')] or similar
    """
    # Ensure CIP labels are assigned (idempotent guard)
    assign_stereochemistry(mol)

    descriptors: List[Tuple[str, str]] = []

    for atom_idx in bridgehead_atoms:
        if atom_idx not in atom_to_locant:
            continue

        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.HasProp('_CIPCode'):
            locant = atom_to_locant[atom_idx]
            # Convert to string if int, to handle both int and 'a' suffix locants
            locant_str = str(locant)
            cip_code = atom.GetProp('_CIPCode')
            descriptors.append((locant_str, cip_code))

    # Sort by locant (handle '4a', '8a' style locants)
    def _junction_locant_sort_key(item: Tuple[str, str]) -> Tuple[int, str]:
        """Sort key for junction locants like '4a', '8a'."""
        locant_str = item[0]
        if locant_str and locant_str[-1].isalpha():
            return (int(locant_str[:-1]), locant_str[-1])
        return (int(locant_str), '')

    descriptors.sort(key=_junction_locant_sort_key)

    return descriptors


def format_ring_junction_stereo(
    descriptors: List[Tuple[str, str]],
    use_rct: bool = False
) -> str:
    """
    Format ring junction stereodescriptors as IUPAC name prefix.

    IUPAC 2013 provides two notations for ring junction stereo:
    1. R/S notation (PIN style): "(4aR,8aS)-"
    2. r/c/t notation (reference plane): "(4ar,8ac)-"

    The r/c/t notation uses:
    - r: reference stereocenter (first one)
    - c: cis to reference (same CIP code)
    - t: trans to reference (different CIP code)

    Args:
        descriptors: List of (locant_str, cip_code) tuples from collect_ring_junction_stereo
        use_rct: If True, use r/c/t notation; if False (default), use R/S

    Returns:
        Formatted string like "(4aR,8aS)-" or "(4ar,8ac)-"
        Returns empty string if no descriptors.

    Examples:
        >>> format_ring_junction_stereo([('4a', 'S'), ('8a', 'S')])
        '(4aS,8aS)-'
        >>> format_ring_junction_stereo([('4a', 'S'), ('8a', 'S')], use_rct=True)
        '(4ar,8ac)-'
        >>> format_ring_junction_stereo([('4a', 'R'), ('8a', 'S')], use_rct=True)
        '(4ar,8at)-'
    """
    if not descriptors:
        return ""

    if use_rct:
        # Convert to r/c/t notation
        # First stereocenter is reference (r)
        # Same CIP as reference = cis (c)
        # Different CIP = trans (t)
        rct_parts = []
        ref_cip = None

        for i, (locant, cip) in enumerate(descriptors):
            if i == 0:
                # First is reference
                ref_cip = cip
                rct_parts.append(f"{locant}r")
            else:
                # Compare to reference
                if cip == ref_cip:
                    rct_parts.append(f"{locant}c")  # cis
                else:
                    rct_parts.append(f"{locant}t")  # trans

        return f"({','.join(rct_parts)})-"
    else:
        # Standard R/S notation
        parts = [f"{locant}{cip}" for locant, cip in descriptors]
        return f"({','.join(parts)})-"


def determine_simple_cis_trans(
    mol,
    junction_atoms: List[int]
) -> Optional[str]:
    """
    Determine cis/trans for simple bicyclic systems (decalin type).

    For simple fused bicyclics with exactly 2 junction atoms:
    - Same chiral tag (both CCW or both CW) = cis (both H on same face)
    - Different chiral tags (one CCW, one CW) = trans (H on opposite faces)

    Note: CIP codes (R/S) are NOT reliable for cis/trans determination in
    symmetric fused systems like decalin, because the molecular symmetry
    can cause both junction atoms to have the same CIP code even in trans.
    Instead, we use the ChiralTag which reflects the actual tetrahedral
    configuration.

    Args:
        mol: RDKit Mol object (stereochemistry should already be assigned)
        junction_atoms: List of exactly 2 atom indices at ring junction

    Returns:
        'cis' or 'trans', or None if:
        - Not exactly 2 junction atoms
        - Missing stereochemistry on junction atoms

    Examples:
        >>> # cis-decalin: both [C@@H] = same ChiralTag = cis
        >>> mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@@H]2C1')
        >>> bridgeheads = get_bridgehead_atoms(mol)
        >>> determine_simple_cis_trans(mol, bridgeheads)
        'cis'
        >>> # trans-decalin: [C@@H]...[C@H] = different ChiralTag = trans
        >>> mol = Chem.MolFromSmiles('C1CC[C@@H]2CCCC[C@H]2C1')
        >>> bridgeheads = get_bridgehead_atoms(mol)
        >>> determine_simple_cis_trans(mol, bridgeheads)
        'trans'
    """
    if len(junction_atoms) != 2:
        return None

    atom1 = mol.GetAtomWithIdx(junction_atoms[0])
    atom2 = mol.GetAtomWithIdx(junction_atoms[1])

    # Get chiral tags
    tag1 = atom1.GetChiralTag()
    tag2 = atom2.GetChiralTag()

    # Both must have defined chirality
    unspecified = Chem.ChiralType.CHI_UNSPECIFIED
    if tag1 == unspecified or tag2 == unspecified:
        return None

    # Also reject CHI_OTHER which indicates unknown/invalid
    other = Chem.ChiralType.CHI_OTHER
    if tag1 == other or tag2 == other:
        return None

    # Compare chiral tags:
    # CHI_TETRAHEDRAL_CW and CHI_TETRAHEDRAL_CCW are the common stereo tags
    # Same tag = cis (both H on same face)
    # Different tag = trans (H on opposite faces)
    if tag1 == tag2:
        return 'cis'
    else:
        return 'trans'


def get_junction_locants_for_fused_system(
    mol,
    bridgehead_atoms: List[int],
    ring_size_1: int,
    ring_size_2: int
) -> Dict[int, str]:
    """
    Generate IUPAC 'a' suffix locants for junction atoms in fused systems.

    For ortho-fused bicyclics like decahydronaphthalene:
    - Ring 1 (6-membered): positions 1-4, then 4a
    - Ring 2 (6-membered): positions 4a-8, then 8a
    - Junction atoms get 'a' suffix locants (4a, 8a)

    This is a simplified implementation for common cases.

    Args:
        mol: RDKit Mol object
        bridgehead_atoms: List of atom indices at ring junctions
        ring_size_1: Size of first ring
        ring_size_2: Size of second ring

    Returns:
        Dict mapping atom index to locant string (e.g., {3: '4a', 8: '8a'})

    Note:
        This is a simplified mapping. Full IUPAC numbering requires
        consideration of ring system orientation and heteroatom positions.
    """
    # For 6,6-fused (decalin/naphthalene type):
    # Total unique positions = ring1 + ring2 - 2 shared atoms
    # Standard IUPAC: 4a and 8a for 6,6-fused
    if ring_size_1 == 6 and ring_size_2 == 6 and len(bridgehead_atoms) == 2:
        # Simple case: assign 4a and 8a
        return {
            bridgehead_atoms[0]: '4a',
            bridgehead_atoms[1]: '8a'
        }

    # For 5,6-fused (indane type): 3a and 7a
    if (ring_size_1 == 5 and ring_size_2 == 6) or (ring_size_1 == 6 and ring_size_2 == 5):
        if len(bridgehead_atoms) == 2:
            return {
                bridgehead_atoms[0]: '3a',
                bridgehead_atoms[1]: '7a'
            }

    # For 5,5-fused (azulene type): 3a and 8a typically
    if ring_size_1 == 5 and ring_size_2 == 5 and len(bridgehead_atoms) == 2:
        return {
            bridgehead_atoms[0]: '3a',
            bridgehead_atoms[1]: '5a'
        }

    # Generic fallback: use sequential 'a' suffix locants
    result = {}
    # Calculate locant based on position in combined ring system
    # For generic case, use (ring_size_1 - 1)a pattern
    if len(bridgehead_atoms) >= 1:
        result[bridgehead_atoms[0]] = f'{ring_size_1 - 1}a'
    if len(bridgehead_atoms) >= 2:
        total_positions = ring_size_1 + ring_size_2 - 2
        result[bridgehead_atoms[1]] = f'{total_positions}a'

    return result
