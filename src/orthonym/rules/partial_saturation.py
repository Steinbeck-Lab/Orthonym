"""
Partial saturation detection and prefix generation for fused ring systems.

This module handles IUPAC 2013 hydro prefixes for partially saturated
fused heterocycles and polycyclic compounds:
- dihydro- (2 H added)
- tetrahydro- (4 H added)
- hexahydro- (6 H added)
- octahydro- (8 H added)
- decahydro- (10 H added)
- dodecahydro- (12 H added)
- perhydro- (fully saturated, no locants needed)

IUPAC 2013 Blue Book:
"Prefixes 'dihydro', 'tetrahydro', etc. indicate the addition of hydrogen
to specified positions of an otherwise unsaturated parent structure."

Key Rules:
1. Always include locants for partial saturation (2,3-dihydro, not just dihydro)
2. Perhydro means ALL ring atoms saturated - no locants used
3. Saturation prefix comes LAST before parent name, AFTER substituents
4. Order: [substituents]-[saturation prefix]-[indicated H]-[parent]

Reference: IUPAC 2013 Blue Book
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from rdkit import Chem

from ..perception.molcache import atoms_of, bonds_of  # audit 2026-09-03 (S2): per-call atom/bond tuples
from ..perception.smarts_cache import compiled as _compiled_smarts

# Saturation prefix mapping based on number of added hydrogens
# Each sp3 carbon in ring adds 2 hydrogens vs aromatic parent
SATURATION_PREFIXES: Dict[int, str] = {
    2: 'dihydro',
    4: 'tetrahydro',
    6: 'hexahydro',
    8: 'octahydro',
    10: 'decahydro',
    12: 'dodecahydro',
    14: 'tetradecahydro',
    16: 'hexadecahydro',
}


def detect_partial_saturation(
    mol: Chem.Mol,
    aromatic_parent_smiles: str
) -> Optional[Dict[str, Any]]:
    """
    Detect partial saturation by comparing molecule to aromatic parent.

    Counts sp3-hybridized atoms in the ring system that would be sp2/aromatic
    in the parent structure. The IUPAC hydro prefix (dihydro-, tetrahydro-, etc.)
    indicates the number of hydrogen atoms added, which corresponds to
    2 * (number of sp3 ring atoms).

    For fused ring systems:
    - tetrahydroquinoline: 4 positions saturated = 4 sp3 atoms in reduced ring
    - indoline (2,3-dihydroindole): 2 sp3 atoms at positions 2,3
    - decahydronaphthalene (decalin): all 10 ring atoms sp3 = perhydro

    Args:
        mol: RDKit molecule to analyze
        aromatic_parent_smiles: SMILES of the fully aromatic parent structure

    Returns:
        Dict with saturation info, or None if no saturation detected:
        - 'sp3_count': Number of sp3 atoms in ring system
        - 'hydrogen_count': Number of added hydrogens (sp3_count * 2)
        - 'prefix': Saturation prefix string ('dihydro', 'tetrahydro', etc.)
        - 'saturated_indices': List of atom indices that are sp3
        - 'is_perhydro': True if fully saturated

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCN2') # tetrahydroquinoline
        >>> result = detect_partial_saturation(mol, 'c1ccc2ncccc2c1')
        >>> result['prefix']
        'hexahydro' # 3 sp3 carbons * 2 = 6H
    """
    if mol is None:
        return None

    if aromatic_parent_smiles is None:
        return None

    parent = Chem.MolFromSmiles(aromatic_parent_smiles)
    if parent is None:
        return None

    # Get ring atoms in the molecule
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)

    # Count aromatic atoms in the parent
    parent_aromatic_count = sum(1 for atom in parent.GetAtoms() if atom.GetIsAromatic())

    # Find sp3-hybridized atoms in the ring system
    # These are the "saturated positions" where hydrogen was added
    sp3_indices = []

    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)

        # Count sp3 atoms (saturated positions)
        if atom.GetHybridization() == Chem.HybridizationType.SP3:
            sp3_indices.append(idx)

    sp3_count = len(sp3_indices)

    # No saturation detected
    if sp3_count == 0:
        return None

    # Calculate hydrogen count based on actual atom valence
    # Carbon sp3 adds 2H vs aromatic, but N adds only 1H (trivalent)
    hydrogen_count = 0
    for idx in sp3_indices:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 7:  # Nitrogen
            hydrogen_count += 1  # N-H (trivalent N in ring)
        else:
            hydrogen_count += 2  # C adds 2H when saturated

    # Check for perhydro (fully saturated)
    # When all ring atoms are sp3
    is_perhydro = sp3_count >= parent_aromatic_count

    # Get prefix
    if is_perhydro:
        prefix = 'perhydro'
    else:
        prefix = get_saturation_prefix(hydrogen_count)

    if prefix is None:
        return None

    return {
        'sp3_count': sp3_count,
        'hydrogen_count': hydrogen_count,
        'prefix': prefix,
        'saturated_indices': sp3_indices,
        'is_perhydro': is_perhydro,
        'parent_aromatic_count': parent_aromatic_count,
    }


def get_saturation_prefix(hydrogen_count: int) -> Optional[str]:
    """
    Get the IUPAC saturation prefix for a given hydrogen count.

    Args:
        hydrogen_count: Number of added hydrogens (typically 2, 4, 6, 8, 10, 12)

    Returns:
        Saturation prefix string, or None if not a standard count

    Examples:
        >>> get_saturation_prefix(2)
        'dihydro'
        >>> get_saturation_prefix(4)
        'tetrahydro'
        >>> get_saturation_prefix(10)
        'decahydro'
    """
    return SATURATION_PREFIXES.get(hydrogen_count)


def get_saturation_locants(
    mol: Chem.Mol,
    sp3_atom_indices: List[int],
    atom_to_locant: Dict[int, Union[int, str]]
) -> List[Union[int, str]]:
    """
    Get IUPAC locants for saturated positions.

    Converts atom indices to IUPAC locants using the provided mapping,
    then sorts them according to IUPAC rules (lowest locant set).

    Args:
        mol: RDKit molecule
        sp3_atom_indices: List of atom indices that are sp3 (saturated)
        atom_to_locant: Mapping from atom index to IUPAC locant

    Returns:
        Sorted list of locants for saturation prefix

    Examples:
        >>> atom_to_locant = {0: 1, 1: 2, 2: 3, 3: 4,...}
        >>> get_saturation_locants(mol, [0, 1, 2, 3], atom_to_locant)
        [1, 2, 3, 4]
    """
    locants = []

    for idx in sp3_atom_indices:
        locant = atom_to_locant.get(idx)
        if locant is not None:
            locants.append(locant)

    # Sort locants using IUPAC rules
    return _sort_locants(locants)


def _sort_locants(locants: List[Union[int, str]]) -> List[Union[int, str]]:
    """
    Sort locants according to IUPAC rules.

    Handles mixed int/str locants (e.g., 1, 2, '3a', '4a').
    Numeric locants come first in ascending order, then fusion locants.

    Args:
        locants: List of locants (int or str like '3a', '7a')

    Returns:
        Sorted list of locants
    """
    def _locant_sort_key(loc: Union[int, str]) -> Tuple[int, str]:
        """Sort key: (numeric_part, alpha_suffix)."""
        if isinstance(loc, int):
            return (loc, '')
        elif isinstance(loc, str):
            # Parse '3a' -> (3, 'a'), '7a' -> (7, 'a')
            if loc and loc[-1].isalpha():
                try:
                    return (int(loc[:-1]), loc[-1])
                except ValueError:
                    return (999, loc)
            try:
                return (int(loc), '')
            except ValueError:
                return (999, loc)
        return (999, str(loc))

    return sorted(locants, key=_locant_sort_key)


def format_saturation_prefix(
    prefix: str,
    locants: Optional[List[Union[int, str]]] = None
) -> str:
    """
    Format saturation prefix with locants for IUPAC name.

    IUPAC rules:
    - Perhydro: no locants (perhydro-)
    - All others: locants required (2,3-dihydro-, 1,2,3,4-tetrahydro-)

    Args:
        prefix: Saturation prefix ('dihydro', 'tetrahydro', 'perhydro', etc.)
        locants: Optional list of locants (not used for perhydro)

    Returns:
        Formatted prefix string ready for name assembly

    Examples:
        >>> format_saturation_prefix('perhydro')
        'perhydro'
        >>> format_saturation_prefix('tetrahydro', [1, 2, 3, 4])
        '1,2,3,4-tetrahydro'
        >>> format_saturation_prefix('dihydro', [2, 3])
        '2,3-dihydro'
    """
    # Perhydro never has locants
    if prefix == 'perhydro':
        return 'perhydro'

    # All other saturation prefixes MUST have locants per IUPAC
    if locants:
        sorted_locants = _sort_locants(locants)
        locant_str = ','.join(str(loc) for loc in sorted_locants)
        return f"{locant_str}-{prefix}"

    # Fallback: return prefix without locants (not ideal, but better than nothing)
    return prefix


def get_saturated_position_locants(
    mol: Chem.Mol,
    saturated_indices: List[int],
    atom_to_locant: Dict[int, Union[int, str]]
) -> List[Union[int, str]]:
    """
    Get IUPAC locants for saturated positions.

    This is an alias for get_saturation_locants for clearer naming.

    Args:
        mol: RDKit molecule
        saturated_indices: List of atom indices that are saturated
        atom_to_locant: Mapping from atom index to IUPAC locant

    Returns:
        Sorted list of locants for saturation prefix
    """
    return get_saturation_locants(mol, saturated_indices, atom_to_locant)


def analyze_saturation_for_naming(
    mol: Chem.Mol,
    aromatic_parent_smiles: str,
    atom_to_locant: Dict[int, Union[int, str]]
) -> Optional[str]:
    """
    Complete analysis for saturation prefix generation in naming.

    This is the main entry point for the composer module. It combines
    detection, locant assignment, and formatting into a single call.

    Args:
        mol: RDKit molecule to analyze
        aromatic_parent_smiles: SMILES of the aromatic parent structure
        atom_to_locant: Mapping from atom index to IUPAC locant

    Returns:
        Formatted saturation prefix string, or None if no saturation

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCN2') # tetrahydroquinoline
        >>> parent = 'c1ccc2ncccc2c1' # quinoline
        >>> atom_to_locant = {...} # IUPAC locant mapping
        >>> analyze_saturation_for_naming(mol, parent, atom_to_locant)
        '1,2,3,4-tetrahydro'
    """
    # Detect saturation
    result = detect_partial_saturation(mol, aromatic_parent_smiles)
    if result is None:
        return None

    prefix = result['prefix']
    saturated_indices = result['saturated_indices']
    is_perhydro = result['is_perhydro']

    if is_perhydro:
        # Perhydro: no locants needed
        return 'perhydro'

    # Get locants for saturated positions
    locants = get_saturated_position_locants(mol, saturated_indices, atom_to_locant)

    if not locants:
        # No locants available - return prefix only (suboptimal)
        return prefix

    return format_saturation_prefix(prefix, locants)


def is_fully_saturated(mol: Chem.Mol, aromatic_parent_smiles: str) -> bool:
    """
    Check if a molecule is fully saturated (perhydro) relative to parent.

    Args:
        mol: RDKit molecule to check
        aromatic_parent_smiles: SMILES of the aromatic parent

    Returns:
        True if the molecule is fully saturated (perhydro)

    Examples:
        >>> mol = Chem.MolFromSmiles('C1CCCC2CCCCC12') # decalin (perhydronaphthalene)
        >>> is_fully_saturated(mol, 'c1ccc2ccccc2c1') # naphthalene
        True
    """
    result = detect_partial_saturation(mol, aromatic_parent_smiles)
    if result is None:
        # No saturation info - could be already fully aromatic or no match
        return False
    return result.get('is_perhydro', False)


def count_ring_sp3_atoms(mol: Chem.Mol) -> int:
    """
    Count sp3-hybridized atoms in ring systems.

    Utility function for saturation analysis.

    Args:
        mol: RDKit molecule

    Returns:
        Number of sp3 atoms in rings
    """
    ri = mol.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)

    count = 0
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetHybridization() == Chem.HybridizationType.SP3:
            count += 1

    return count


def get_ring_saturation_level(
    mol: Chem.Mol,
    aromatic_parent_smiles: str
) -> str:
    """
    Get a human-readable saturation level description.

    Args:
        mol: RDKit molecule
        aromatic_parent_smiles: SMILES of aromatic parent

    Returns:
        Description string: 'aromatic', 'partially saturated', or 'fully saturated'
    """
    result = detect_partial_saturation(mol, aromatic_parent_smiles)

    if result is None:
        return 'aromatic'
    elif result.get('is_perhydro', False):
        return 'fully saturated'
    else:
        return 'partially saturated'


# =============================================================================
# Carbocyclic Partial Saturation (PAH systems)
# =============================================================================


@lru_cache(maxsize=None)
def _carbocyclic_mancude_parents_by_size(n_ring_atoms: int) -> Tuple[str, ...]:
    """All-carbon mancude parents in ``POLYCYCLIC_DATA`` that carry a populated
    ``iupac_numbering`` and are a *pure* ring system of exactly ``n_ring_atoms``
    atoms.

    Returning every same-size candidate (e.g. naphthalene AND azulene at 10;
    anthracene AND phenanthrene at 14) is safe: the bond-order-agnostic
    automorphism match in:func:`_match_mancude_parent_numbering` only succeeds
    for the parent whose *connectivity* matches, so a mismatched same-size
    parent is silently skipped (fail-closed — never a wrong parent guess).
    """
    from ..data.polycyclic_data import POLYCYCLIC_DATA

    out: List[str] = []
    for name, entry in POLYCYCLIC_DATA.items():
        numbering = entry.get('iupac_numbering')
        csmiles = entry.get('canonical_smiles')
        if not numbering or not csmiles:
            continue
        cmol = Chem.MolFromSmiles(csmiles)
        if cmol is None or cmol.GetNumAtoms() != n_ring_atoms:
            continue
        if any(a.GetSymbol() != 'C' for a in cmol.GetAtoms()):
            continue
        out.append(name)
    return tuple(out)


def _match_mancude_parent_numbering(
    mol: Chem.Mol,
    ring_atoms: Set[int],
    sp3_atoms: Set[int],
    ring_double_bonds: List[Tuple[int, int]],
    candidate_parents: List[str],
    substituent_ring_atoms: Optional[Set[int]] = None,
    pcg_ring_atoms: Optional[Set[int]] = None,
) -> Optional[Tuple[str, Dict[int, Any]]]:
    """Number a (partly) saturated fused carbocycle by its mancude parent.

    For each candidate parent, do a bond-order-agnostic automorphism match of
    the stored aromatic parent skeleton onto ``mol`` and read off the parent's
    fixed ``iupac_numbering`` (incl. the lettered fusion locants 4a/8a/...).
    Among ALL matching numberings (across all candidates) pick the one giving
    the LOWEST locant set FIRST to the principal-characteristic-group ring atoms
    (``pcg_ring_atoms``, — the suffix outranks hydro), then to the
    hydro positions (the sp3 ring atoms), then to the residual ring double bonds
    — IUPAC 2013.

    Returns ``(parent_name, {atom_idx: locant})`` for the winning numbering, or
    ``None`` if no candidate's skeleton matches the ring system exactly (fail
    closed — never emit a wrong parent/locant).

    This is the shared primitive behind both the aromatic-bearing partial-PAH
    path (:func:`detect_carbocyclic_partial_saturation`) and the all-saturated
    residual-ene path (:func:`name_hydrogenated_fused_carbocycle`); it replaces
    the old naive sp3-walk that mis-numbered non-adjacent hydro positions (e.g.
    1,4-dihydronaphthalene → wrong "1,2-").
    """
    from ..data.polycyclic_data import POLYCYCLIC_DATA

    ring_atom_set = set(ring_atoms)
    best_key = None
    best: Optional[Tuple[str, Dict[int, Any]]] = None

    for parent in candidate_parents:
        entry = POLYCYCLIC_DATA.get(parent)
        if not entry:
            continue
        numbering = entry.get('iupac_numbering') or {}
        csmiles = entry.get('canonical_smiles')
        if not numbering or not csmiles:
            continue
        cmol = Chem.MolFromSmiles(csmiles)
        if cmol is None:
            continue
        n_canonical = cmol.GetNumAtoms()
        if n_canonical != len(ring_atom_set):
            continue

        params = Chem.AdjustQueryParameters.NoAdjustments()
        params.makeBondsGeneric = True
        params.aromatizeIfPossible = False
        params.adjustDegree = False
        query = Chem.AdjustQueryProperties(cmol, params)

        for match in mol.GetSubstructMatches(query, uniquify=False):
            if len(match) != n_canonical:
                continue
            atom_to_locant = {
                match[c_idx]: loc
                for c_idx, loc in numbering.items()
                if c_idx < len(match)
            }
            # The match must cover EXACTLY the ring system (not some other subset).
            if set(atom_to_locant) != ring_atom_set:
                continue
            hydro_locs = sorted(_locant_key(atom_to_locant[a]) for a in sp3_atoms)
            ene_locs = sorted(
                min(_locant_key(atom_to_locant[i]), _locant_key(atom_to_locant[j]))
                for i, j in ring_double_bonds
            )
            #: a ring principal-characteristic-group suffix takes the
            # lowest locant FIRST — before hydro and before detachable prefixes.
            if pcg_ring_atoms:
                pcg_locs = sorted(
                    _locant_key(atom_to_locant[a]) for a in pcg_ring_atoms
                    if a in atom_to_locant
                )
            else:
                pcg_locs = []
            # /: after PCG + hydro + ene, the detachable
            # substituents take the lowest locants (breaks ties among equivalent
            # numberings, e.g. 5-methyl- vs 8-methyl-tetrahydronaphthalene).
            if substituent_ring_atoms:
                sub_locs = sorted(
                    _locant_key(atom_to_locant[a]) for a in substituent_ring_atoms
                    if a in atom_to_locant
                )
            else:
                sub_locs = []
            key = (pcg_locs, hydro_locs, ene_locs, sub_locs)
            if best_key is None or key < best_key:
                best_key = key
                best = (parent, atom_to_locant)

    return best


#: a ring carbon bearing an exocyclic -C(=O)OH is a ring
# 'carboxylic acid' PCG. Named as the '-carboxylic acid' suffix on a
# partially-saturated named-carbocycle parent; the acid carbon is exocyclic
# (not a ring atom), so it never enters the ring numbering — the ring carbon it
# hangs off is the locant-bearing atom.
_RING_COOH_SMARTS = Chem.MolFromSmarts('[#6;R][CX3](=O)[OX2H1]')


def _partial_sat_pcg_ring_atoms(mol, fused_ring_atoms: Set[int]) -> Set[int]:
    """Ring atoms bearing a ring-attached carboxylic-acid PCG ('-carboxylic
    acid' suffix). Empty set when none. Only ring carbons in the fused system
    count; the exocyclic acid carbon/oxygens are not ring atoms."""
    out: Set[int] = set()
    if _RING_COOH_SMARTS is None:
        return out
    for match in mol.GetSubstructMatches(_RING_COOH_SMARTS):
        ring_c = match[0]
        if ring_c in fused_ring_atoms:
            out.add(ring_c)
    return out


# "HYDROXY COMPOUNDS AND CHALCOGEN ANALOGUES": a ring carbon bearing an
# exocyclic -OH is a ring 'ol' PCG, named with the '-ol' suffix on the
# hydro-prefixed parent. BB verbatim PIN (the Blue Book):
# "(1) 5,6,7,8-tetrahydronaphthalen-2-ol (PIN)". The hydroxy oxygen is
# exocyclic, so the ring carbon it hangs off is the locant-bearing atom.
_RING_OH_SMARTS = Chem.MolFromSmarts('[#6;R][OX2H1]')
# A PRIMARY amine (-NH2) on a ring carbon -> the '-amine' suffix /
# -adjacent; `1,2,3,4-tetrahydronaphthalen-1-amine` is the PIN at
# the Blue Book). Neutral, exactly two H, single bond.
_RING_NH2_SMARTS = Chem.MolFromSmarts('[#6;R][NX3;H2;+0]')

# Suffix spelling per PCG kind. The '-ol' form elides the parent's terminal 'e'
# only when no multiplying prefix intervenes (a)): the Blue Book prints
# BOTH `naphthalen-4a(2H)-ol (PIN)` and `naphthalene-4a,8a-diol (PIN)`
# (the Blue Book).
_PARTIAL_SAT_PCG_SUFFIX = {
    'carboxylic_acid': ('carboxylic acid', False),
    'ol': ('ol', True),
    # A vowel-initial suffix, so it elides the parent's terminal 'e'
    # (`naphthalen-1-amine`); a multiplied form keeps it (`naphthalene-1,5-diamine`).
    'amine': ('amine', True),
}


def _partial_sat_pcg(
    mol, fused_ring_atoms: Set[int]
) -> Tuple[Optional[str], Set[int]]:
    """The SENIOR ring principal-characteristic-group of a partially-saturated
    fused carbocycle, as ``(kind, ring_atoms)`` — ``(None, set)`` when there
    is none.

    ``kind`` is ``'carboxylic_acid'`` or ``'ol'``.
    seniority puts acids above alcohols, so a ring -COOH wins outright and any
    ring -OH then stays a detachable ``hydroxy`` prefix — two suffixes are never
    produced.

    The ``'ol'`` class is scoped FAIL-CLOSED to molecules whose only heteroatoms
    are the ring-attached hydroxy oxygens, one per ring carbon. That makes -ol
    provably the senior characteristic group present, so the suffix
    choice cannot be wrong, and it keeps out of scope an -OH on an EXOCYCLIC
    carbon — which makes that carbon the parent instead
    (``(1,2,3,4-tetrahydronaphthalen-2-yl)methanol``, not an ``-ol``).

    Single source of truth: both the numbering primitive (which must give the
    PCG the lowest locants, and the assembler (which spells the
    suffix) read this one answer, so the two can never disagree.
    """
    acid = _partial_sat_pcg_ring_atoms(mol, fused_ring_atoms)
    if acid:
        return ('carboxylic_acid', acid)
    if _RING_OH_SMARTS is None:
        return (None, set())
    ol_ring: Set[int] = set()
    ol_oxygens: Set[int] = set()
    for match in mol.GetSubstructMatches(_RING_OH_SMARTS):
        ring_c, oxygen = match[0], match[1]
        if ring_c in fused_ring_atoms:
            ol_ring.add(ring_c)
            ol_oxygens.add(oxygen)
    if not ol_ring:
        # No ring -OH: the next-junior scoped suffix this path spells is the
        # primary amine puts amines below alcohols, so -ol above always
        # wins when present).
        return _partial_sat_pcg_amine(mol, fused_ring_atoms)
    # Exactly one -OH per PCG ring carbon: a geminal diol would need a
    # different construction than one locant per suffix position.
    if len(ol_oxygens) != len(ol_ring):
        return (None, set())
    # Every heteroatom in the molecule must be one of those hydroxy oxygens,
    # and the species must be neutral and non-radical — otherwise -ol is not
    # provably the senior group and this path must not claim the suffix.
    for atom in atoms_of(mol):
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return (None, set())
        if atom.GetAtomicNum() != 6 and atom.GetIdx() not in ol_oxygens:
            return (None, set())
    return ('ol', ol_ring)


def _partial_sat_pcg_amine(
    mol, fused_ring_atoms: Set[int]
) -> Tuple[Optional[str], Set[int]]:
    """The '-amine' PCG of a partially-saturated fused carbocycle:
    a ring carbon bearing a primary -NH2. Scoped FAIL-CLOSED, exactly like the
    '-ol' path — the only heteroatoms present must be those amine nitrogens (so
    -amine is provably the senior characteristic group, puts it below
    alcohols/acids, both already handled above), one -NH2 per ring carbon,
    neutral non-radical. An -NH2 on an EXOCYCLIC carbon is out of scope (that
    carbon is the parent), and an N-substituted amine is deferred (v1 primary
    only). ``(None, set)`` when it does not apply."""
    if _RING_NH2_SMARTS is None:
        return (None, set())
    am_ring: Set[int] = set()
    am_nitrogens: Set[int] = set()
    for match in mol.GetSubstructMatches(_RING_NH2_SMARTS):
        ring_c, nitrogen = match[0], match[1]
        if ring_c in fused_ring_atoms:
            am_ring.add(ring_c)
            am_nitrogens.add(nitrogen)
    if not am_ring or len(am_nitrogens) != len(am_ring):
        return (None, set())
    for atom in atoms_of(mol):
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return (None, set())
        if atom.GetAtomicNum() != 6 and atom.GetIdx() not in am_nitrogens:
            return (None, set())
    return ('amine', am_ring)


def _partial_sat_pcg_exocyclic_atoms(
    mol, ring_set: Set[int], pcg_kind: Optional[str], pcg_ring_atoms: Set[int]
) -> Set[int]:
    """The off-ring atoms consumed by the PCG suffix — the carboxyl C plus its
    two oxygens for ``'carboxylic_acid'``, the hydroxy oxygen for ``'ol'``.

    Shared by the assembler (so they are not ALSO named as detachable prefixes)
    and by the ``partial_sat_sp3_substituent_drop`` conservation veto (so an
    atom the suffix already accounts for is not counted as a silent drop)."""
    out: Set[int] = set()
    if not pcg_kind or not pcg_ring_atoms:
        return out
    for rc in pcg_ring_atoms:
        for n in mol.GetAtomWithIdx(rc).GetNeighbors():
            if n.GetIdx() in ring_set:
                continue
            if pcg_kind == 'carboxylic_acid' and n.GetSymbol() == 'C':
                # the carboxyl carbon + its two oxygens
                out.add(n.GetIdx())
                for nn in n.GetNeighbors():
                    if nn.GetIdx() != rc and nn.GetSymbol() == 'O':
                        out.add(nn.GetIdx())
            elif pcg_kind == 'ol' and n.GetSymbol() == 'O':
                out.add(n.GetIdx())
            elif pcg_kind == 'amine' and n.GetSymbol() == 'N':
                out.add(n.GetIdx())
    return out


def detect_carbocyclic_partial_saturation(
    mol: Chem.Mol,
    fused_ring_atoms: Set[int]
) -> Optional[Dict[str, Any]]:
    """
    Detect partial saturation in carbocyclic fused systems.

    This function identifies partially saturated polycyclic aromatic hydrocarbons
    like tetrahydronaphthalene, dihydroanthracene, etc. It analyzes the fused
    ring system to detect if it's a partially saturated version of a known
    aromatic parent (naphthalene, anthracene, phenanthrene).

    The detection works by:
    1. Checking all ring atoms are carbons (pure carbocycle)
    2. Counting aromatic vs sp3 atoms
    3. Matching the ring system size and structure to known parents
    4. Computing the saturation prefix based on sp3 count

    IUPAC 2013 Blue Book:
    - tetrahydronaphthalene: 4 sp3 atoms = tetrahydro prefix
    - dihydronaphthalene: 2 sp3 atoms = dihydro prefix
    - decahydronaphthalene (decalin): all sp3 = perhydro or decahydro

    Args:
        mol: RDKit molecule
        fused_ring_atoms: Set of atom indices in the fused ring system

    Returns:
        Dict with saturation info, or None if not a recognized partially
        saturated carbocycle:
        - 'parent_name': Name of aromatic parent ('naphthalene', etc.)
        - 'parent_smiles': SMILES of aromatic parent
        - 'prefix': Saturation prefix string ('tetrahydro', etc.)
        - 'saturated_indices': List of atom indices that are sp3
        - 'sp3_count': Number of sp3 atoms
        - 'hydrogen_count': Number of added hydrogens (sp3_count * 2)
        - 'is_perhydro': True if fully saturated
        - 'atom_to_locant': Mapping from atom index to IUPAC locant (if
          available). This IS the numbering the name is spelled from —
          consumers must inherit it rather than re-derive one.
        - 'pcg_kind': 'carboxylic_acid' | 'ol' | None — the senior ring
          principal characteristic group, which was given the lowest locants
          (see:func:`_partial_sat_pcg`)
        - 'pcg_ring_atoms': the ring atoms carrying it (empty when None)

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCC2') # tetrahydronaphthalene
        >>> ri = mol.GetRingInfo
        >>> ring_atoms = set
        >>> for ring in ri.AtomRings:
        ... ring_atoms.update(ring)
        >>> result = detect_carbocyclic_partial_saturation(mol, ring_atoms)
        >>> result['prefix']
        'tetrahydro'
        >>> result['parent_name']
        'naphthalene'
    """
    if mol is None or not fused_ring_atoms:
        return None

    # Check if all ring atoms are carbons (carbocyclic)
    for idx in fused_ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            # Has heteroatom - not a pure carbocycle
            return None

    # Count sp3 and aromatic atoms in the fused ring system
    sp3_indices = []
    aromatic_indices = []

    for idx in fused_ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetHybridization() == Chem.HybridizationType.SP3:
            sp3_indices.append(idx)
        elif atom.GetIsAromatic():
            aromatic_indices.append(idx)

    sp3_count = len(sp3_indices)
    aromatic_count = len(aromatic_indices)
    total_ring_atoms = len(fused_ring_atoms)

    # No sp3 atoms = fully aromatic, not partially saturated
    if sp3_count == 0:
        return None

    # For fully saturated systems (aromatic_count == 0), we need to distinguish:
    # - Fused bicyclic systems (decalin = 2 fused 6-rings = perhydronaphthalene)
    # - Simple large rings (cyclodecane = 1 ring with 10 atoms)
    #
    # Only fused systems should be named as perhydro-aromatics.
    if aromatic_count == 0:
        # Check if this is a fused ring system (multiple rings sharing atoms)
        ri = mol.GetRingInfo()
        atom_rings = ri.AtomRings()

        # Count rings that overlap with fused_ring_atoms
        rings_in_system = []
        for ring in atom_rings:
            ring_set = set(ring)
            if ring_set & fused_ring_atoms:
                rings_in_system.append(ring_set)

        # If there's only 1 ring, it's a simple cycloalkane, not a fused system
        if len(rings_in_system) <= 1:
            return None

        # Check if rings are actually fused (share atoms)
        is_fused = False
        for i, ring1 in enumerate(rings_in_system):
            for j, ring2 in enumerate(rings_in_system):
                if i < j and len(ring1 & ring2) >= 2:
                    is_fused = True
                    break
            if is_fused:
                break

        if not is_fused:
            return None

    # Residual non-aromatic ring C=C double bonds — needed both to choose the
    # lowest-locant numbering and to recognise full saturation.
    ring_double_bonds: List[Tuple[int, int]] = []
    for bond in bonds_of(mol):
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetIsAromatic():
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in fused_ring_atoms and b in fused_ring_atoms:
            ring_double_bonds.append((a, b))

    # Identify the mancude parent AND its IUPAC numbering by a bond-order-agnostic
    # automorphism match: the all-carbon parent skeleton whose
    # connectivity matches this ring system, numbered to give the lowest locants
    # first to the hydro (sp3) positions, then to any residual ring double bond.
    # Fail closed if no all-carbon parent of this exact ring size matches — never
    # guess the parent by ring-atom count (the old size->name guess mislabelled
    # phenanthrene as anthracene) and never number by a naive sp3 walk (the old
    # _build_naphthalene_type_locants mis-numbered non-adjacent hydro positions,
    # e.g. 1,4-dihydronaphthalene -> wrong "1,2-").
    sp3_set = set(sp3_indices)
    candidates = _carbocyclic_mancude_parents_by_size(total_ring_atoms)
    # tiebreak: ring atoms bearing an off-ring (substituent) heavy atom.
    # Among numberings tied on hydro+ene locants, the one giving these the
    # lowest locants wins (5-methyl- not 8-methyl-tetrahydronaphthalene).
    substituent_ring_atoms = {
        idx for idx in fused_ring_atoms
        if any(n.GetIdx() not in fused_ring_atoms
               for n in mol.GetAtomWithIdx(idx).GetNeighbors())
    }
    # "Nondetachable hydro prefixes vs. indicated hydrogen"
    # (the Blue Book), on the PIN `5,8-dioxo-5,6,7,8-tetrahydro-
    # naphthalene-2-carboxylic acid`: "detachable but nonalphabetized hydro
    # prefixes do not have precedence over the principal characteristic group
    # for low numbering, but has precedence over other detachable prefixes."
    # So the PCG ring atoms must be numbered LOWEST — before the hydro set,
    # which in turn outranks the detachable substituent prefixes. Detected
    # ONCE here so the numbering primitive can prioritise them and the
    # assembler can spell the suffix from the SAME answer.
    pcg_kind, pcg_ring_atoms = _partial_sat_pcg(mol, fused_ring_atoms)
    matched = _match_mancude_parent_numbering(
        mol, fused_ring_atoms, sp3_set, ring_double_bonds, candidates,
        substituent_ring_atoms=substituent_ring_atoms,
        pcg_ring_atoms=pcg_ring_atoms,
    )
    if matched is None:
        return None
    parent_name, atom_to_locant = matched

    from ..data.polycyclic_data import POLYCYCLIC_DATA
    parent_smiles = POLYCYCLIC_DATA[parent_name].get('canonical_smiles')

    # In a mancude carbocycle every ring atom is sp2; each sp3 ring atom is one
    # added hydrogen, so hydrogen_count == sp3_count.
    hydrogen_count = sp3_count
    is_perhydro = (sp3_count == total_ring_atoms)

    if is_perhydro:
        prefix = 'perhydro'
    else:
        prefix = get_saturation_prefix(hydrogen_count)

    if prefix is None:
        return None

    return {
        'parent_name': parent_name,
        'parent_smiles': parent_smiles,
        'prefix': prefix,
        'saturated_indices': sp3_indices,
        'sp3_count': sp3_count,
        'hydrogen_count': hydrogen_count,
        'is_perhydro': is_perhydro,
        'atom_to_locant': atom_to_locant,
        'pcg_kind': pcg_kind,
        'pcg_ring_atoms': pcg_ring_atoms,
    }


def _build_naphthalene_type_locants(
    mol: Chem.Mol,
    ring_atoms: Set[int],
    sp3_indices: List[int]
) -> Dict[int, int]:
    """
    Build IUPAC locant mapping for naphthalene-type fused systems.

    For tetrahydronaphthalene, IUPAC numbering is:
    - Positions 1-4: saturated ring (the sp3 atoms)
    - Positions 4a, 5-8, 8a: aromatic ring

    For simplicity, we assign:
    - sp3 atoms get locants 1, 2, 3, 4 (in order around the saturated ring)
    - Aromatic atoms get locants 5, 6, 7, 8 (in order)

    Args:
        mol: RDKit molecule
        ring_atoms: Set of atom indices in the fused system
        sp3_indices: List of sp3 atom indices

    Returns:
        Dict mapping atom index to IUPAC locant (1-indexed)
    """
    from collections import defaultdict

    if not sp3_indices:
        return {idx: i + 1 for i, idx in enumerate(sorted(ring_atoms))}

    # Get ring info
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()

    # Find the ring containing sp3 atoms (the saturated ring)
    saturated_ring = None
    for ring in atom_rings:
        ring_set = set(ring)
        if not (ring_set & ring_atoms):
            continue
        ring_sp3 = ring_set & set(sp3_indices)
        if len(ring_sp3) >= 2:
            saturated_ring = ring
            break

    if saturated_ring is None:
        return {idx: i + 1 for i, idx in enumerate(sorted(ring_atoms))}

    # Find fusion atoms (shared between rings)
    atom_ring_count = defaultdict(int)
    for ring in atom_rings:
        ring_set = set(ring)
        if ring_set & ring_atoms:
            for idx in ring:
                if idx in ring_atoms:
                    atom_ring_count[idx] += 1

    fusion_atoms = {idx for idx, count in atom_ring_count.items() if count > 1}

    # Build ordered traversal around the saturated ring
    # Start from an sp3 atom that's adjacent to a fusion atom
    sp3_set = set(sp3_indices)

    # Find starting sp3 atom adjacent to fusion
    start_atom = None
    for idx in sp3_indices:
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() in fusion_atoms:
                start_atom = idx
                break
        if start_atom:
            break

    if start_atom is None:
        start_atom = sp3_indices[0]

    # Traverse the sp3 atoms in order (simple chain traversal)
    visited = set()
    sp3_order = []
    current = start_atom

    while len(sp3_order) < len(sp3_indices):
        if current in visited:
            break
        visited.add(current)
        sp3_order.append(current)

        # Find next unvisited sp3 neighbor
        atom = mol.GetAtomWithIdx(current)
        found_next = False
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in sp3_set and nbr_idx not in visited:
                current = nbr_idx
                found_next = True
                break

        if not found_next:
            # No more sp3 neighbors, break
            break

    # Add any missed sp3 atoms
    for idx in sp3_indices:
        if idx not in visited:
            sp3_order.append(idx)

    # Build locant mapping: sp3 atoms get locants 1,2,3,4
    atom_to_locant = {}
    for i, idx in enumerate(sp3_order):
        atom_to_locant[idx] = i + 1

    # Non-sp3 atoms get higher locants (5, 6, 7, 8,...)
    non_sp3 = [idx for idx in ring_atoms if idx not in sp3_set]
    next_locant = len(sp3_order) + 1
    for idx in sorted(non_sp3):  # Simple ordering for now
        atom_to_locant[idx] = next_locant
        next_locant += 1

    return atom_to_locant


def is_tetrahydronaphthalene(mol: Chem.Mol, fused_ring_atoms: Set[int]) -> bool:
    """
    Check if fused system is tetrahydronaphthalene-type.

    Tetrahydronaphthalene has:
    - 10 ring atoms total
    - 4 sp3 carbons (saturated ring)
    - 6 aromatic carbons (benzene ring)
    - All carbons (no heteroatoms)

    Args:
        mol: RDKit molecule
        fused_ring_atoms: Set of atom indices in the fused ring system

    Returns:
        True if the system is tetrahydronaphthalene-type

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccc2c(c1)CCCC2')
        >>> ri = mol.GetRingInfo
        >>> ring_atoms = set
        >>> for ring in ri.AtomRings:
        ... ring_atoms.update(ring)
        >>> is_tetrahydronaphthalene(mol, ring_atoms)
        True
    """
    if len(fused_ring_atoms) != 10:
        return False

    sp3_count = 0
    aromatic_count = 0

    for idx in fused_ring_atoms:
        atom = mol.GetAtomWithIdx(idx)

        # Must be carbon
        if atom.GetSymbol() != 'C':
            return False

        if atom.GetHybridization() == Chem.HybridizationType.SP3:
            sp3_count += 1
        elif atom.GetIsAromatic():
            aromatic_count += 1

    # Tetrahydronaphthalene: 4 sp3 + 6 aromatic
    return sp3_count == 4 and aromatic_count == 6


# =============================================================================
# Hydrogenated fused carbocycles with residual ring double bonds (V-5 / V2 theme)
# =============================================================================

# Number of ring atoms -> mancude parent that has a populated ``iupac_numbering``
# map in POLYCYCLIC_DATA. Only parents whose numbering (incl. the lettered fusion
# locants 4a/8a/...) is stored can be numbered correctly; everything else fails
# closed (no wrong locants). 10 = naphthalene covers the common V-5 class.
_HYDRO_FUSED_PARENT_BY_SIZE: Dict[int, str] = {
    10: 'naphthalene',
}


def _locant_key(loc: Union[int, str, Tuple[int, str]]) -> Tuple[int, str]:
    """Sortable key for a locant that may be an int, a ``(int, str)`` tuple, or a
    string like ``'4a'`` — so that ``4 < 4a < 5`` and integers/tuples mix cleanly."""
    if isinstance(loc, tuple):
        return (int(loc[0]), str(loc[1]))
    if isinstance(loc, int):
        return (loc, '')
    if isinstance(loc, str):
        i = 0
        while i < len(loc) and loc[i].isdigit():
            i += 1
        if i == 0:
            return (999, loc)
        return (int(loc[:i]), loc[i:])
    return (999, str(loc))


def _locant_display(loc: Union[int, str, Tuple[int, str]]) -> str:
    """Render a locant (int / ``(int, str)`` tuple / ``'4a'`` string) as the
    string used in a name (``4a``, ``8a``, ``1``...)."""
    if isinstance(loc, tuple):
        return f"{loc[0]}{loc[1]}"
    return str(loc)


def name_hydrogenated_fused_carbocycle(mol: Chem.Mol) -> Optional[str]:
    """Name a *partially* saturated fused bicyclic carbocycle (naphthalene-type).

    This covers the V-5 / V2-theme defect: a fused carbocycle that is the
    hydrogenated form of a mancude parent (e.g. naphthalene) but still retains
    one or more *isolated* (non-aromatic) ring C=C double bonds. The legacy
    ``_name_saturated_fused_carbocyclic`` blindly emits the fully-saturated
    (``decahydro``) name for any non-aromatic two-ring carbocycle, dropping the
    residual double bond and naming a different molecule.

    Algorithm (IUPAC 2013:
      1. Require a two-ring, all-carbon, non-aromatic system with >= 1 residual
         ring double bond (fully-saturated systems are handled as ``perhydro``
         elsewhere -> return None here).
      2. Identify the mancude parent by ring-atom count; require a populated
         ``iupac_numbering`` map (with lettered fusion locants).
      3. Enumerate every automorphic numbering of the parent skeleton onto the
         molecule (bond-order-agnostic substructure match), and pick the one
         giving the LOWEST locant set to the ``hydro`` prefixes (the sp3 ring
         atoms), then to the residual double bonds.
      4. Emit ``<locants>-<count>hydro<parent>`` (e.g.
         ``1,2,3,4,4a,5,6,8a-octahydronaphthalene``).

    Fails closed (returns None) for any system it cannot number correctly so it
    never emits a wrong name.

    Args:
        mol: RDKit molecule (a two-ring carbocyclic, non-aromatic fused system).

    Returns:
        The hydro-prefixed parent name, or None if not handled (fail closed).
    """
    from ..data.polycyclic_data import POLYCYCLIC_DATA

    if mol is None:
        return None

    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    if len(atom_rings) != 2:
        return None

    ring_atoms: Set[int] = set()
    for ring in atom_rings:
        ring_atoms.update(ring)

    # Carbocyclic only; no aromatic ring atoms (the aromatic-bearing partial
    # case is handled by the polycyclics PAH path).
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            return None
        if atom.GetIsAromatic():
            return None

    # Residual ring double bonds (non-aromatic C=C with both atoms in the ring).
    ring_double_bonds: List[Tuple[int, int]] = []
    for bond in bonds_of(mol):
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetIsAromatic():
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in ring_atoms and b in ring_atoms:
            ring_double_bonds.append((a, b))

    # No residual double bond -> fully saturated; perhydro is handled elsewhere.
    if not ring_double_bonds:
        return None

    # Atom-conservation veto (R12,: this emitter produces only a bare
    # `<locants>-<prefix><parent>` name with NO substituent slot, so any exocyclic
    # heavy-atom neighbour on a ring atom (e.g. the two ring-fusion -OH of
    # naphthalene-4a,8a-diol) would be silently DROPPED, naming a different
    # molecule. Gated, catches it; gate-off (no Java) it would ship the
    # atom-dropped name. Fail closed here at the source so a substituted system is
    # never mis-named as its bare hydro-parent. (Unsubstituted hydro-fused
    # carbocycles have no off-ring heavy neighbour and are unaffected.)
    for idx in ring_atoms:
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetIdx() not in ring_atoms and nb.GetAtomicNum() > 1:
                return None

    # The sp3 (saturated / "hydro") ring atoms.
    sp3_atoms = {
        idx for idx in ring_atoms
        if mol.GetAtomWithIdx(idx).GetHybridization() == Chem.HybridizationType.SP3
    }
    sp2_atoms = ring_atoms - sp3_atoms

    # Every sp2 ring atom must be accounted for by a ring double bond (no
    # exocyclic unsaturation such as an exocyclic =CH2 or =O on the ring) — else
    # the hydro count would be wrong. Fail closed otherwise.
    if len(sp2_atoms) != 2 * len(ring_double_bonds):
        return None

    parent = _HYDRO_FUSED_PARENT_BY_SIZE.get(len(ring_atoms))
    if parent is None:
        return None
    entry = POLYCYCLIC_DATA.get(parent)
    if entry is None:
        return None
    numbering = entry.get('iupac_numbering') or {}
    canonical_smiles = entry.get('canonical_smiles')
    if not numbering or not canonical_smiles:
        return None

    canonical_mol = Chem.MolFromSmiles(canonical_smiles)
    if canonical_mol is None:
        return None

    # Bond-order-agnostic query: the stored parent SMILES is aromatic, but the
    # molecule is (partly) saturated, so match on connectivity only while keeping
    # the canonical atom order so ``numbering`` (keyed by canonical index) applies.
    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.makeBondsGeneric = True
    params.aromatizeIfPossible = False
    params.adjustDegree = False
    query = Chem.AdjustQueryProperties(canonical_mol, params)

    matches = mol.GetSubstructMatches(query, uniquify=False)
    if not matches:
        return None

    n_hydro = len(sp3_atoms)
    prefix = SATURATION_PREFIXES.get(n_hydro)
    if prefix is None:
        return None

    n_canonical = canonical_mol.GetNumAtoms()

    # Choose the numbering minimizing (hydro-locant set, then residual-double-bond
    # locant set) per (lowest locants to hydro prefixes + ene).
    best_key = None
    best_map: Optional[Dict[int, Any]] = None
    for match in matches:
        if len(match) != n_canonical:
            continue
        atom_to_locant = {
            match[c_idx]: loc
            for c_idx, loc in numbering.items()
            if c_idx < len(match)
        }
        if len(atom_to_locant) != len(ring_atoms):
            continue
        hydro_locs = sorted((_locant_key(atom_to_locant[a]) for a in sp3_atoms))
        ene_locs = sorted(
            min(_locant_key(atom_to_locant[i]), _locant_key(atom_to_locant[j]))
            for i, j in ring_double_bonds
        )
        key = (hydro_locs, ene_locs)
        if best_key is None or key < best_key:
            best_key = key
            best_map = atom_to_locant

    if best_map is None:
        return None

    hydro_display = sorted((best_map[a] for a in sp3_atoms), key=_locant_key)
    locant_str = ','.join(_locant_display(loc) for loc in hydro_display)
    return f"{locant_str}-{prefix}{parent}"


# =============================================================================
# Added-indicated-hydrogen -ol / -amine suffix on a mancude naphthalene
# / — the -ol/-amine sibling of name_cyclic_oxo_compound
# =============================================================================

# Suffix word by PCG kind + count. -ol/-amine are vowel-initial (a)
# elides the parent's terminal 'e' at count 1); the multiplied 'di-/tri-' forms
# are consonant-initial (keep it) — the two BB PINs are `naphthalen-4a(2H)-ol`
# and `naphthalene-4a,8a-diol` (the Blue Book).
_ADDED_H_SUFFIX_WORDS: Dict[str, Dict[int, str]] = {
    'ol':    {1: 'ol', 2: 'diol', 3: 'triol', 4: 'tetrol'},
    'amine': {1: 'amine', 2: 'diamine', 3: 'triamine', 4: 'tetramine'},
}


def name_added_h_fused_carbocycle_suffix(mol: Chem.Mol) -> Optional[str]:
    """Name a mancude naphthalene bearing an -ol/-amine (di-) suffix that needs
    'added indicated hydrogen' /, e.g. ``naphthalen-4a(2H)-ol``,
    ``naphthalen-4a(2H)-amine``, ``naphthalene-2,4a(2H)-diamine``,
    ``naphthalene-4a,8a-diol``.

    The Blue Book gives these the added-indicated-hydrogen form as the PIN
    (the Blue Book), NOT the equivalent hydro form
    (``2,4a-dihydronaphthalen-4a-ol``); both parse to the same structure through
    OPSIN, so the round-trip gate cannot choose between them and this producer
    must spell the PIN directly.

    This is the -ol/-amine sibling of:func:`name_cyclic_oxo_compound`: the -one
    suffix carbon is sp2 (exocyclic C=O) whereas the -ol/-amine suffix carbon is
    itself sp3, but the added-indicated-hydrogen mechanism max
    noncumulative double bonds; a pair of suffixes that removes a
    double bond needs no added H) is the same, so the maximum matching of the
    reduced ring atoms is reused.

    Scope (fail-closed -> None otherwise): a single neutral non-radical fragment,
    no specified stereo, whose ring system is exactly the naphthalene skeleton
    (two ortho-fused 6-membered all-carbon rings, 10 atoms), NON-aromatic (an
    aromatic ring routes to the PAH partial-saturation path). The senior ring PCG
    must be an -ol or -amine (via:func:`_partial_sat_pcg`, which already scopes
    fail-closed to molecules whose only heteroatoms are the suffix O/N). Every sp2
    ring atom must be covered by an intra-ring C=C (no exocyclic unsaturation),
    every suffix carbon must be sp3, and there must be NO hydro prefix (every
    reduced adjacent pair is a suffix pair removing a double bond) — the
    hydro-prefixed forms are a separate build.

    Numbering + added-indicated-H: the reduced (sp3) ring atoms are matched
     adjacent pair = removed double bond -> no added H); an unmatched
    reduced atom needs one added/indicated hydrogen, cited ``(nH)`` after the
    suffix locant(s), UNLESS it is a ring-fusion carbon already bearing the suffix
    (its hydrogen is consumed by the suffix and marked only by the suffix locant).
    The naphthalene fixed numbering is chosen to give lowest locants to the suffix,
    then to the added indicated hydrogen. Fail-closed on any indeterminacy
    (the downstream / round-trip gate is the constitution backstop).
    """
    from ..data.polycyclic_data import POLYCYCLIC_DATA

    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    for atom in atoms_of(mol):
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None
    # Specified stereo would be silently dropped by this bare-numbering path.
    for atom in atoms_of(mol):
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return None
    for bond in bonds_of(mol):
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            return None

    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    # Exactly the naphthalene skeleton: two ortho-fused 6-rings, 10 ring atoms.
    if len(atom_rings) != 2:
        return None
    if any(len(r) != 6 for r in atom_rings):
        return None
    if len(set(atom_rings[0]) & set(atom_rings[1])) != 2:  # ortho fusion = 1 shared edge
        return None
    ring_atoms: Set[int] = set()
    for r in atom_rings:
        ring_atoms.update(r)
    if len(ring_atoms) != 10:
        return None
    # Carbocyclic, non-aromatic (an aromatic ring routes to the PAH partial-sat path).
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C' or atom.GetIsAromatic():
            return None

    # The senior ring PCG must be an -ol or -amine /. The helper
    # scopes fail-closed to molecules whose only heteroatoms are the suffix O/N
    # (one per ring carbon, neutral non-radical), so the suffix choice is safe.
    pcg_kind, pcg_ring_atoms = _partial_sat_pcg(mol, ring_atoms)
    if pcg_kind not in ('ol', 'amine') or not pcg_ring_atoms:
        return None
    suffix_atoms = set(pcg_ring_atoms)

    # This producer renders ONLY the bare naphthalene parent + the -ol/-amine
    # suffix; it cannot render a substituent. So every OFF-ring heavy atom must be
    # a suffix heteroatom (the -ol O / -amine N on a suffix ring carbon). Any other
    # off-ring heavy atom — e.g. a ring methyl — would be SILENTLY DROPPED (0-wrong
    # would rest only on the downstream gate). Fail closed. (_partial_sat_pcg
    # only guarantees no non-suffix HETEROatom exists; a carbon substituent, and a
    # substituent on an sp2 ring atom, both slip past the sp3-only check below.)
    suffix_hetero: Set[int] = set()
    for c in suffix_atoms:
        for nb in mol.GetAtomWithIdx(c).GetNeighbors():
            if nb.GetIdx() not in ring_atoms and nb.GetSymbol() in ('O', 'N'):
                suffix_hetero.add(nb.GetIdx())
    for atom in atoms_of(mol):
        idx = atom.GetIdx()
        if atom.GetAtomicNum() <= 1 or idx in ring_atoms or idx in suffix_hetero:
            continue
        return None  # off-ring heavy atom that is not a suffix O/N -> would be dropped

    # Residual ring C=C required; every double bond involving a ring atom must be
    # intra-ring (no exocyclic =CH2/=O — keeps the mancude interpretation).
    ring_double_bonds: List[Tuple[int, int]] = []
    for bond in bonds_of(mol):
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        a_in, b_in = a in ring_atoms, b in ring_atoms
        if a_in != b_in:
            return None  # exocyclic double bond on a ring atom -> out of scope
        if a_in and b_in:
            ring_double_bonds.append((a, b))
    if not ring_double_bonds:
        return None

    # Reduced (sp3) ring atoms: suffix carbons + plain hydrocarbon >CH2 / >CH-.
    sp3_atoms = {
        idx for idx in ring_atoms
        if mol.GetAtomWithIdx(idx).GetHybridization() == Chem.HybridizationType.SP3
    }
    # Every suffix carbon must be a reduced (sp3) position (an sp2 enol/enamine
    # =C(OH)- is a different, non-added-H case -> fail closed).
    if not suffix_atoms.issubset(sp3_atoms):
        return None
    # Every sp2 ring atom is covered by an intra-ring C=C (no stray unsaturation).
    sp2_atoms = ring_atoms - sp3_atoms
    if len(sp2_atoms) != 2 * len(ring_double_bonds):
        return None
    # A plain (non-suffix) sp3 ring atom must be a pure hydrocarbon (no off-ring
    # heavy neighbour) — _partial_sat_pcg already guarantees no non-suffix
    # heteroatom exists, so this is belt-and-braces.
    for idx in sp3_atoms - suffix_atoms:
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetIdx() not in ring_atoms and nb.GetAtomicNum() > 1:
                return None

    # --- naphthalene fixed numbering (bond-order-agnostic automorphism match) ---
    parent = _HYDRO_FUSED_PARENT_BY_SIZE.get(len(ring_atoms))
    if parent is None:
        return None
    entry = POLYCYCLIC_DATA.get(parent)
    if entry is None:
        return None
    numbering = entry.get('iupac_numbering') or {}
    canonical_smiles = entry.get('canonical_smiles')
    if not numbering or not canonical_smiles:
        return None
    canonical_mol = Chem.MolFromSmiles(canonical_smiles)
    if canonical_mol is None:
        return None
    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.makeBondsGeneric = True
    params.aromatizeIfPossible = False
    params.adjustDegree = False
    query = Chem.AdjustQueryProperties(canonical_mol, params)
    matches = mol.GetSubstructMatches(query, uniquify=False)
    if not matches:
        return None
    n_canonical = canonical_mol.GetNumAtoms()

    # Ring degree: a fusion carbon carries 3 ring bonds. A suffix on such a carbon
    # (0 mancude H) consumes its added hydrogen, so it is not itself cited as (nH).
    ring_degree = {
        idx: sum(1 for nb in mol.GetAtomWithIdx(idx).GetNeighbors()
                 if nb.GetIdx() in ring_atoms)
        for idx in ring_atoms
    }

    # Adjacency among the reduced set (for the maximum matching,.
    sat_adj: Dict[int, Set[int]] = {i: set() for i in sp3_atoms}
    for bond in bonds_of(mol):
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in sp3_atoms and j in sp3_atoms:
            sat_adj[i].add(j)
            sat_adj[j].add(i)

    best_key = None
    best: Optional[Tuple[List[Any], List[Any]]] = None
    for match in matches:
        if len(match) != n_canonical:
            continue
        loc = {match[c]: l for c, l in numbering.items() if c < len(match)}
        if set(loc) != ring_atoms:
            continue
        # Maximum matching of the reduced ring atoms: matched adjacent pairs remove
        # a ring double bond -> no added H); unmatched atoms each need
        # one added/indicated H.
        matched, unmatched = _max_oxo_matching(sat_adj, sp3_atoms, loc)
        # A matched (double-bond-removing) pair must be TWO suffix carbons; a
        # matched plain atom is a 'hydro' (dihydro) form -> a separate build.
        if any(a not in suffix_atoms for a in matched):
            continue
        # Added indicated hydrogen: an unmatched reduced atom, EXCEPT a ring-fusion
        # carbon that bears the suffix (its hydrogen is consumed by the suffix).
        added_ih = {a for a in unmatched
                    if not (ring_degree[a] >= 3 and a in suffix_atoms)}
        # Any unmatched atom NOT cited as added-IH must be a suffix carbon (a
        # fusion suffix) — otherwise a saturated position is unexplained.
        if any(a not in suffix_atoms for a in (unmatched - added_ih)):
            continue
        suffix_locs = sorted((loc[a] for a in suffix_atoms), key=_locant_key)
        ih_locs = sorted((loc[a] for a in added_ih), key=_locant_key)
        key = (
            [_locant_key(x) for x in suffix_locs],
            len(ih_locs),
            [_locant_key(x) for x in ih_locs],
        )
        if best_key is None or key < best_key:
            best_key = key
            best = (suffix_locs, ih_locs)
    if best is None:
        return None
    suffix_locs, ih_locs = best

    count = len(suffix_atoms)
    words = _ADDED_H_SUFFIX_WORDS.get(pcg_kind)
    suffix_word = words.get(count) if words else None
    if suffix_word is None:
        return None
    # (a): elide the parent's terminal 'e' only before a vowel-initial
    # suffix (single -ol/-amine); the multiplied di-/tri- forms keep it.
    stem = parent[:-1] if (parent.endswith('e') and suffix_word[0] in 'aeiou') else parent
    sloc = ','.join(_locant_display(x) for x in suffix_locs)
    ih_str = ('(' + ','.join(f"{_locant_display(x)}H" for x in ih_locs) + ')') if ih_locs else ''
    return f"{stem}-{sloc}{ih_str}-{suffix_word}"


# =============================================================================
# Ring peroxol suffix on a partially-saturated fused carbocycle
# =============================================================================


def name_hydro_fused_chalcogen_suffix(mol: Chem.Mol) -> Optional[str]:
    """ (BB 27935): -OOH on an sp3 carbon of a partially saturated
    fused carbocycle -> '<hydro-parent>-<locant>-peroxol' (BB verbatim PIN:
    1,2,3,4-tetrahydronaphthalene-1-peroxol).

    Fail-closed (accuracy-first): exactly one -OOH, no other heteroatoms and
    no other substituents; the skeleton (molecule minus the two O) must be the
    numbering-verified tetralin family (names to '1,2,3,4-tetrahydronaphthalene'
    via the normal pipeline) and the OOH carbon must sit at locant 1 (the sp3
    ring carbon bonded to an aromatic fusion carbon). Structured so -OH can join
    later; this task ships only the -OOH case."""
    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    for atom in atoms_of(mol):
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None
    patt = _compiled_smarts("[OX2H][OX2][CX4;R]")
    matches = mol.GetSubstructMatches(patt)
    if len(matches) != 1:
        return None
    oh, o2, c = matches[0]
    # whole molecule minus the two -OOH oxygens must be an all-carbon skeleton
    skeleton = Chem.RWMol(mol)
    for idx in sorted((oh, o2), reverse=True):
        skeleton.RemoveAtom(idx)
    sk = skeleton.GetMol()
    try:
        Chem.SanitizeMol(sk)
    except Exception:
        return None
    for atom in sk.GetAtoms():
        if atom.GetAtomicNum() != 6:
            return None      # any residual heteroatom/other substituent -> fail closed
    # Name the bare skeleton through the normal pipeline; only the
    # numbering-verified tetralin family is in scope for this task.
    from ..namer import name_compound
    base = name_compound(Chem.MolToSmiles(sk), style="pin")
    if base != "1,2,3,4-tetrahydronaphthalene":
        return None
    # The OOH carbon must sit at locant 1 of the saturated ring: it is the sp3
    # carbon bonded to an aromatic fusion carbon (verify structurally).
    c_atom = mol.GetAtomWithIdx(c)
    if not any(n.GetIsAromatic() for n in c_atom.GetNeighbors()):
        return None
    return f"{base}-1-peroxol"


# =============================================================================
# Added indicated hydrogen + ring-ketone suffix /
# =============================================================================


def _ring_carbonyl_carbons(mol, ring_set: Set[int]) -> List[int]:
    """Ring carbons bearing an exocyclic ketone-type =O (suitable for an -one /
    -dione suffix). Covers ketone, lactam, and lactone carbonyls on mancude
    rings — all named with the -one suffix per P-64.2.2.2."""
    out: List[int] = []
    for idx in ring_set:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            continue
        for bond in atom.GetBonds():
            o = bond.GetOtherAtom(atom)
            if (bond.GetBondType() == Chem.BondType.DOUBLE
                    and o.GetSymbol() == 'O'
                    and o.GetIdx() not in ring_set
                    and o.GetDegree() == 1):
                out.append(idx)
                break
    return out


def _ring_imine_carbons(mol, ring_set: Set[int],
                        include_nsub: bool = False) -> List[int]:
    """Ring carbons bearing an exocyclic imine =NH / =N-R, suitable
    for the -imine / -diimine suffix with the same added-indicated-H numbering as
    -one.

    The nitrogen must be exocyclic, double-bonded and neutral. By default only a
    BARE imine (degree 1, =NH) qualifies — the degree-1 requirement is the gate
    that keeps N-substituted ring imines off the bare-imine path. With
    ``include_nsub=True`` an N-substituted imine (degree 2, =N-R where the single
    non-ring neighbour is a carbon substituent) also qualifies; the N-substituent
    is then rendered by the caller as an ``N``-locanted prefix
    (``N1,N4-dimethylnaphthalene-1,4-diimine``, /. A charged
    =N(+) is out of scope in both modes (fail closed)."""
    out: List[int] = []
    for idx in ring_set:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            continue
        for bond in atom.GetBonds():
            o = bond.GetOtherAtom(atom)
            if not (bond.GetBondType() == Chem.BondType.DOUBLE
                    and o.GetSymbol() == 'N'
                    and o.GetIdx() not in ring_set
                    and o.GetFormalCharge() == 0):
                continue
            if o.GetDegree() == 1:
                out.append(idx)
                break
            if include_nsub and o.GetDegree() == 2 and o.GetTotalNumHs() == 0:
                others = [nb for nb in o.GetNeighbors() if nb.GetIdx() != idx]
                if (len(others) == 1 and others[0].GetSymbol() == 'C'
                        and others[0].GetIdx() not in ring_set):
                    out.append(idx)
                    break
    return out


def _ring_chalcogenone_carbons(mol, ring_set: Set[int], symbol: str) -> List[int]:
    """: ring carbons bearing an exocyclic ketone-type ``=S``/``=Se``
    (thione / selone), the heavier-chalcogen analogues of the ``-one`` ketone
    suffix chalcogen replacement; e.g. ``pyridine-2(1H)-thione``,
    ``pyrimidine-2,4(1H,3H)-dithione``). Mirrors ``_ring_carbonyl_carbons``
    exactly but for ``symbol`` in {'S','Se'} (and 'O' for parity); the exocyclic
    chalcogen must be terminal (degree 1) so a RING sulfur (thiophene) or a
    substituent-bearing =S(R) is never mistaken for a thione suffix."""
    out: List[int] = []
    for idx in ring_set:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() != 'C':
            continue
        for bond in atom.GetBonds():
            o = bond.GetOtherAtom(atom)
            if (bond.GetBondType() == Chem.BondType.DOUBLE
                    and o.GetSymbol() == symbol
                    and o.GetIdx() not in ring_set
                    and o.GetDegree() == 1):
                out.append(idx)
                break
    return out


def _fused_ring_system_atoms(rings, seed_atoms) -> Set[int]:
    """Return the atoms of the single FUSED ring system that contains a seed atom.

    Rings are merged into one system when they share an edge (>=2 atoms = ortho-
    or peri-fusion); rings connected only by a single bond (a pendant phenyl) or a
    spiro atom (1 shared atom) are SEPARATE systems. This isolates the parent
    ring system of a ring ketone from pendant ring substituents (the phenyl of a
    3-phenylchroman-4-one). Returns the merged atom set whose system contains any
    ``seed_atoms`` atom, or an empty set if none match.
    """
    ring_sets = [set(r) for r in rings]
    parent = list(range(len(rings)))

    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(ring_sets[i] & ring_sets[j]) >= 2:  # shared edge = fused
                parent[_find(i)] = _find(j)

    systems: Dict[int, Set[int]] = {}
    for i in range(len(rings)):
        systems.setdefault(_find(i), set()).update(ring_sets[i])
    for atoms in systems.values():
        if atoms & seed_atoms:
            return atoms
    return set()


def _collect_subtree(mol, start: int, blocked: Set[int]) -> Set[int]:
    """BFS the connected atoms reachable from ``start`` without crossing any atom
    in ``blocked``. Used to gather an exocyclic N-substituent subtree (the imine
    N is the blocked cut point), so the group can be named on its own."""
    seen: Set[int] = set()
    stack = [start]
    while stack:
        i = stack.pop()
        if i in seen or i in blocked:
            continue
        seen.add(i)
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            if nb.GetIdx() not in blocked:
                stack.append(nb.GetIdx())
    return seen


def _elide_terminal_e(stem: str) -> str:
    """Elide a terminal 'e' before a vowel-initial suffix (a)): naphthalene
    -> naphthalen (before -one); kept before -dione/-trione (consonant)."""
    return stem[:-1] if stem.endswith('e') else stem


def _max_oxo_matching(adj, nodes, locant):
    """Max-cardinality matching over ``nodes`` (ring sub-graph ``adj``); among
    maximum matchings, minimise the sorted unmatched-locant tuple. Returns
    ``(matched_set, unmatched_set)``. The matched atoms pair into reduced ring
    double bonds (hydro); the unmatched atoms are the indicated-H positions."""
    from functools import lru_cache
    nodes = frozenset(nodes)

    @lru_cache(maxsize=None)
    def rec(avail):
        if not avail:
            return (0, (), frozenset())
        a = min(avail)
        rest = avail - {a}
        best = rec(rest)
        best = (best[0], tuple(sorted(best[1] + (_locant_key(locant[a]),))), best[2])
        for b in adj[a]:
            if b in rest:
                s, uk, mset = rec(rest - {b})
                cand = (s + 1, uk, mset | {a, b})
                if (cand[0] > best[0]) or (cand[0] == best[0] and cand[1] < best[1]):
                    best = cand
        return best

    _, _, matched = rec(nodes)
    return set(matched), set(nodes) - set(matched)


def _mancude_ring_parent(mol, ring_atoms):
    """Aromatic mancude parent of the whole ring system (drop every exocyclic
    bond, force all ring atoms+bonds aromatic, re-sanitise). Returns
    ``(parent_mol, old_to_new)`` or ``(None, None)`` if it cannot aromatize."""
    ring = set(ring_atoms)
    em = Chem.RWMol()
    old_to_new = {}
    for idx in sorted(ring):
        old_to_new[idx] = em.AddAtom(Chem.Atom(mol.GetAtomWithIdx(idx).GetAtomicNum()))
    for bond in bonds_of(mol):
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in ring and j in ring:
            em.AddBond(old_to_new[i], old_to_new[j], Chem.BondType.AROMATIC)
            em.GetAtomWithIdx(old_to_new[i]).SetIsAromatic(True)
            em.GetAtomWithIdx(old_to_new[j]).SetIsAromatic(True)
    parent = em.GetMol()
    try:
        Chem.SanitizeMol(parent)
    except Exception:
        return None, None
    if not all(a.GetIsAromatic() for a in parent.GetAtoms()):
        return None, None
    return parent, old_to_new


def _mancude_ring_parent_nh(mol, ring_atoms):
    """The mancude parent of a monocycle whose ONLY ring nitrogen must carry the
    indicated hydrogen for the ring to be mancude (the 1H-pyrrole type, which
    _mancude_ring_parent cannot build: a bare aromatic 'n' in a 5-ring does not
    kekulize). Returns ``(parent_mol, old_to_new, n_atom)`` or
    ``(None, None, None)``."""
    ring = set(ring_atoms)
    ns = [i for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() == 7]
    if len(ns) != 1:
        return None, None, None
    em = Chem.RWMol()
    old_to_new = {}
    for idx in sorted(ring):
        at = Chem.Atom(mol.GetAtomWithIdx(idx).GetAtomicNum())
        if idx == ns[0]:
            at.SetNumExplicitHs(1)
        old_to_new[idx] = em.AddAtom(at)
    for bond in bonds_of(mol):
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in ring and j in ring:
            em.AddBond(old_to_new[i], old_to_new[j], Chem.BondType.AROMATIC)
            em.GetAtomWithIdx(old_to_new[i]).SetIsAromatic(True)
            em.GetAtomWithIdx(old_to_new[j]).SetIsAromatic(True)
    parent = em.GetMol()
    try:
        Chem.SanitizeMol(parent)
    except Exception:
        return None, None, None
    if not all(a.GetIsAromatic() for a in parent.GetAtoms()):
        return None, None, None
    return parent, old_to_new, ns[0]


# Single-chalcogen pyran family: 2H-/4H-pyran and chalcogen analogues are PINs
# (BB Table 2.2, lines 2164/8141). These mancude parents do NOT aromatize, so
# they are built here rather than via _mancude_ring_parent.
_CHALCOGEN_PYRAN_BASE = {'O': 'pyran', 'S': 'thiopyran',
                         'Se': 'selenopyran', 'Te': 'telluropyran'}


def _monocyclic_intrinsic_ih_oxo_parents(mol, ring_set):
    """Intrinsic-indicated-H mancude parent candidates for a NON-aromatizing
    6-membered monocycle bearing exactly one chalcogen (O/S/Se/Te) — the pyran
    family. A ring ketone on such a parent is named by DIRECT substitution of
    the indicated-H >CH2: ``4H-pyran-4-one`` / ``2H-pyran-2-one``
    and chalcogen analogues. The only mancude (max non-cumulative double bond)
    indicated-H positions of a 6-ring with the heteroatom at locant 1 are 2 and
    4 (positions 3/5 cannot carry two noncumulative double bonds; 6 ≡ 2), so two
    parent isomers per direction are offered. Returns
    ``[(name, [(loc, {})], ih_locants),...]``; the carbonyl-at-indicated-H
    constraint in:func:`name_cyclic_oxo_compound` selects the correct isomer
    and ring direction. Returns ```` (fail-closed) for any other ring.
    """
    from .heterocycles import _macrocycle_ordered_ring
    if len(ring_set) != 6:
        return []
    het = [i for i in ring_set
           if mol.GetAtomWithIdx(i).GetSymbol() in _CHALCOGEN_PYRAN_BASE]
    if len(het) != 1:
        return []
    x = het[0]
    if any(mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in ring_set if i != x):
        return []
    base = _CHALCOGEN_PYRAN_BASE[mol.GetAtomWithIdx(x).GetSymbol()]
    ordered = _macrocycle_ordered_ring(mol, ring_set)
    if ordered is None or len(ordered) != 6:
        return []
    n = 6
    out = []
    for start in range(n):
        if ordered[start] != x:
            continue
        for direction in (1, -1):
            seq = [ordered[(start + k * direction) % n] for k in range(n)]
            loc = {a: idx + 1 for idx, a in enumerate(seq)}  # heteroatom -> 1
            for ih in (2, 4):  # the only valid mancude indicated-H positions
                out.append((f"{ih}H-{base}", [(loc, {})], frozenset({ih})))
    return out


def _resolve_oxo_parent(mol, ring_atoms, monocyclic: Optional[bool] = None,
                        alternative_indicated_h: bool = False):
    """Resolve the mancude parent(s) of a ring-ketone's ring system.

    Returns a list of ``(parent_name, [(locant_map, parentH_map),...], ih_locants)``
    candidates (empty list if none resolve). ``locant_map``: mol-atom -> IUPAC
    locant. ``ih_locants``: the set of locants carrying INTRINSIC indicated
    hydrogen in this parent (a >CH2 in the otherwise mancude system, e.g. {2} for
    2H-chromene, {4} for 4H-chromene, {2}/{4} for the pyran isomers); empty for a
    fully-mancude parent (naphthalene, pyridine). For an intrinsic-IH parent the
    carbonyl must sit AT one of ``ih_locants`` — the ketone is the direct
    substitution of that >CH2 — which disambiguates the parent
    isomer (4H-chromen-4-one, not 2H-chromen-3-one). ``parentH_map`` is retained
    for shape compatibility (currently unused downstream).

    ``alternative_indicated_h``: a fused catalog parent with ONE intrinsic
    indicated hydrogen is also offered with that hydrogen at every other
    peripheral carbon where the ring system stays mancude ('2H-indole' beside the
    catalog's '1H-indole'), so a ketone there can take it,
    :func:`_alternative_indicated_h_parents`).
    """
    from ..data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
    from ..data.polycyclic_data import POLYCYCLIC_DATA
    from .heterocycles import (
        _macrocycle_ordered_ring,
        get_heteroatom_priority,
        name_heterocycle,
    )

    ring_set = set(ring_atoms)
    n = len(ring_set)
    rings = mol.GetRingInfo().AtomRings()
    if monocyclic is None:
        # historical test: the WHOLE molecule has one ring (a pendant ring sends
        # a monocycle to the fused-catalog branch below, which declines it)
        monocyclic = len(rings) == 1

    # --- monocyclic: build the mancude (aromatic) parent + free numbering ---
    if monocyclic:
        if all(mol.GetAtomWithIdx(i).GetSymbol() == 'C' for i in ring_set):
            return []  # carbocyclic monocycle -> cycloalkanone path
        # a phase (A): "mancude parent + added indicated hydrogen"
        # is a Hantzsch-Widman-range concept (rings of size
        # 3-10 -- the same bound name_heterocycle's own `ring_size > 10`
        # branch and _name_lambda_heteromonocycle already enforce). Past
        # that, RDKit can still force-sanitize a monocyclic ring as fully
        # aromatic (_mancude_ring_parent succeeds), but citing it as a
        # mancude parent is WRONG nomenclature for a macrocycle -- the PIN
        # method there is direct ene/dione locant citation on the
        # replacement-nomenclature stem. Measured: an unsaturated
        # 13-membered oxa-lactone ring (ONE real ring C=C) built a bogus
        # 6-double-bond "mancude" parent and named a different molecule
        # (a hyper-unsaturated hexaene + decahydro form), silently
        # preempting the correct ring_heterocycle path via the KIH
        # short-circuit in tier_a_ring. Decline here so the caller falls
        # through to it.
        if n > 10:
            return []
        parent, old_to_new = _mancude_ring_parent(mol, ring_set)
        nh_atom = None
        if parent is None:
            # A 1H-pyrrole-type monocycle: mancude only with the indicated
            # hydrogen on its ring nitrogen. The candidate declares that N's
            # locant as its intrinsic indicated hydrogen; name_cyclic_oxo_compound
            # uses it only in the case (the suffix pair just removes
            # ring double bonds: '1H-pyrrole-2,5-dione (PIN)', the Blue Book).
            parent, old_to_new, nh_atom = _mancude_ring_parent_nh(mol, ring_set)
        if parent is None:
            # non-aromatizing mancude monocycle (pyran-type intrinsic-IH parent)
            return _monocyclic_intrinsic_ih_oxo_parents(mol, ring_set)
        pr = parent.GetRingInfo().AtomRings()
        if len(pr) != 1:
            return []
        pname = name_heterocycle(parent, pr[0])
        if not pname:
            return []
        parentH = {a: parent.GetAtomWithIdx(old_to_new[a]).GetTotalNumHs() for a in ring_set}
        ordered = _macrocycle_ordered_ring(mol, ring_set)
        if ordered is None or len(ordered) != n:
            return []
        het = {i for i in ring_set if mol.GetAtomWithIdx(i).GetSymbol() != 'C'}
        nums = []
        for start in range(n):
            for direction in (1, -1):
                seq = [ordered[(start + k * direction) % n] for k in range(n)]
                nums.append({a: idx + 1 for idx, a in enumerate(seq)})

        def _hk(loc):
            return (tuple(sorted(loc[a] for a in het)),
                    tuple(sorted((get_heteroatom_priority(mol.GetAtomWithIdx(a).GetSymbol()), loc[a]) for a in het)))
        if het:
            best_h = min(_hk(l) for l in nums)
            nums = [l for l in nums if _hk(l) == best_h]
        if nh_atom is not None:
            # The parent name spells its indicated hydrogen ('1H-pyrrole'); keep
            # only the numberings that put that locant on the ring nitrogen.
            import re as _re
            _m = _re.match(r'^(\d+)H-', pname)
            if not _m:
                return []
            ih = int(_m.group(1))
            nums = [l for l in nums if l[nh_atom] == ih]
            if not nums:
                return []
            return [(pname, [(l, parentH) for l in nums], frozenset({ih}))]
        # An aromatic mancude monocycle has no intrinsic indicated hydrogen.
        return [(pname, [(l, parentH) for l in nums], frozenset())]

    # --- fused: bond-generic skeleton match vs PAH then fused-heterocycle data ---
    # Only MANCUDE parents are valid oxo-parents: the ketone substitutes an
    # indicated-H >CH2 and the added-IH/hydro is computed RELATIVE to the mancude
    # ring. A SATURATED catalog entry (chromane, thiochromane, 2,3-dihydro-1-
    # benzofuran,...) is NOT a mancude parent — matching it would make a fused
    # saturated ketone (chroman-4-one) inherit the saturated parent's hydro
    # pattern instead of re-deriving the lowest-locant form. Skip
    # any entry flagged saturated by its name ('hydro') or ring_system.
    cands = []
    for nm, e in POLYCYCLIC_DATA.items():
        if e.get('iupac_numbering') and e.get('canonical_smiles'):
            cands.append((nm, e['canonical_smiles'], e['iupac_numbering']))
    for cs, e in FUSED_HETEROCYCLE_DATA.items():
        if not e.get('iupac_locants'):
            continue
        if 'hydro' in e['name'].lower() or 'saturated' in e.get('ring_system', '').lower():
            continue  # saturated/partially-hydro entry — not a mancude oxo-parent
        cands.append((e['name'], cs, e['iupac_locants']))
    out = []
    for nm, cs, numbering in cands:
        cmol = Chem.MolFromSmiles(cs)
        if cmol is None or cmol.GetNumAtoms() != n:
            continue
        # Intrinsic indicated-H locants of THIS parent: ring carbons that are NOT
        # in a ring double bond of the kekulised parent (the >CH2 of the mancude
        # system). Independent of the match; computed once per candidate.
        ih_locants = set()
        kek = Chem.Mol(cmol)
        try:
            Chem.Kekulize(kek, clearAromaticFlags=True)
            cring = set().union(*[set(r) for r in cmol.GetRingInfo().AtomRings()]) \
                if cmol.GetRingInfo().AtomRings() else set()
            in_ring_db = set()
            for bond in kek.GetBonds():
                if bond.GetBondType() == Chem.BondType.DOUBLE:
                    i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    if i in cring and j in cring:
                        in_ring_db.add(i)
                        in_ring_db.add(j)
            for c, loc in numbering.items():
                # A mancude parent's intrinsic indicated-hydrogen positions are ring
                # atoms attached only by single ring bonds (not in a ring double
                # bond) that carry hydrogen. These sit on CARBON (1H-indene,
                # 2H-/4H-chromene) OR on NITROGEN (9H-purine, 1H-indole,
                # 1H-benzimidazole, 9H-carbazole,...). Nitrogen was previously
                # excluded, so an N-indicated-H parent reported NO intrinsic IH; the
                # carbonyl-at-IH validity check below was then bypassed and the
                # catalog's baked-in tautomer label (e.g. "9H-") was emitted without
                # verifying it fits the target — over-saturating the ring (the
                # caffeine "9H-purine-2,6(1H,3H,7H)-dione" bug, caught only by the
                # backstop). Detecting N here restores the check for the
                # whole N-indicated-H parent class.
                if (c < cmol.GetNumAtoms() and isinstance(loc, int)
                        and cmol.GetAtomWithIdx(c).GetSymbol() in ('C', 'N')
                        and cmol.GetAtomWithIdx(c).GetTotalNumHs() > 0
                        and c not in in_ring_db):
                    ih_locants.add(loc)
        except Exception:
            ih_locants = set()
        params = Chem.AdjustQueryParameters.NoAdjustments()
        params.makeBondsGeneric = True
        params.aromatizeIfPossible = False
        params.adjustDegree = False
        q = Chem.AdjustQueryProperties(cmol, params)
        pairs = []
        for match in mol.GetSubstructMatches(q, uniquify=False):
            a2l = {match[c]: loc for c, loc in numbering.items() if c < len(match)}
            if set(a2l) != ring_set:
                continue
            if not all(mol.GetAtomWithIdx(match[c]).GetAtomicNum() == cmol.GetAtomWithIdx(c).GetAtomicNum()
                       for c in numbering if c < len(match)):
                continue
            pH = {match[c]: cmol.GetAtomWithIdx(c).GetTotalNumHs() for c in numbering if c < len(match)}
            pairs.append((a2l, pH))
        if pairs:
            out.append((nm, pairs, frozenset(ih_locants)))
            if alternative_indicated_h:
                for alt_nm, alt_ih in _alternative_indicated_h_parents(
                        nm, cmol, numbering, ih_locants):
                    # 4th field: only for the case (see the caller)
                    out.append((alt_nm, pairs, frozenset({alt_ih}), True))
    return out


def _has_perfect_matching(mol, nodes) -> bool:
    """True iff the atoms ``nodes`` of ``mol`` can be paired off completely along
    bonds between them (a Kekule arrangement of double bonds over them)."""
    from functools import lru_cache
    nbrs = {a: frozenset(nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                         if nb.GetIdx() in nodes) for a in nodes}

    @lru_cache(maxsize=None)
    def rec(avail):
        if not avail:
            return True
        a = min(avail)
        rest = avail - {a}
        return any(rec(rest - {b}) for b in nbrs[a] if b in rest)

    return rec(frozenset(nodes))


def _alternative_indicated_h_parents(name, cmol, numbering, ih_locants):
    """The other single-indicated-hydrogen forms of a fused catalog parent.

     (the Blue Book): "When there are an equal number of
    indicated hydrogen atoms and principal characteristic groups..., the
    indicated hydrogen atoms are placed at peripheral atoms that will accommodate
    these principal characteristic groups... Locants for hydro prefixes are those
    of the saturated positions" ('2-(1,3,4,5-tetrahydro-2H-2-benzazepin-2-yl)
    ethan-1-ol (PIN)',:24774). The catalog holds one tautomer per ring system
    ('1H-indole'); a ketone at C-2 needs '2H-indole' -> '1,3-dihydro-2H-indol-2-
    one'. Offered at a peripheral CARBON p (two ring bonds, a double-bond atom of
    the catalog form) when the ring system with p as its only sp3 atom still has
    a Kekule arrangement over every other double-bond atom, the catalog's own
    indicated-hydrogen atom included (it must be a C or N with two ring bonds).
    Returns ``[(name, locant)]``;  for anything else."""
    import re as _re
    if len(ih_locants) != 1:
        return []
    (h,) = tuple(ih_locants)
    m = _re.match(rf'^{h}H-(.+)$', name)
    if not m:
        return []
    stem = m.group(1)
    n_atoms = cmol.GetNumAtoms()
    ring = set()
    for r in cmol.GetRingInfo().AtomRings():
        ring.update(r)
    kek = Chem.Mol(cmol)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:
        return []
    db_atoms = set()
    for bond in kek.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in ring and j in ring:
                db_atoms.update((i, j))

    def _ring_degree(a):
        return sum(1 for nb in cmol.GetAtomWithIdx(a).GetNeighbors()
                   if nb.GetIdx() in ring)

    h_atoms = [c for c, loc in numbering.items() if loc == h and c < n_atoms]
    if len(h_atoms) != 1:
        return []
    h_atom = h_atoms[0]
    if (cmol.GetAtomWithIdx(h_atom).GetSymbol() not in ('C', 'N')
            or _ring_degree(h_atom) != 2 or h_atom in db_atoms):
        return []
    base = db_atoms | {h_atom}
    out = []
    for c, loc in numbering.items():
        if not isinstance(loc, int) or loc == h or c >= n_atoms:
            continue
        if (cmol.GetAtomWithIdx(c).GetSymbol() != 'C' or c not in db_atoms
                or _ring_degree(c) != 2):
            continue
        if _has_perfect_matching(cmol, base - {c}):
            out.append((f"{loc}H-{stem}", loc))
    return out


def _ring_carbonyl_rings_atoms(mol, parent_ring_set):
    """Atoms of the OTHER ring systems of ``mol`` that carry a ring carbonyl (an
    exocyclic =O on a ring carbon) -- the ketone / pseudoketone rings ranks
    with the parent's (branch review fixes)."""
    ri = mol.GetRingInfo()
    systems = []
    for r in ri.AtomRings():
        r = set(r)
        for other in [x for x in systems if x & r]:
            systems.remove(other)
            r |= other
        systems.append(r)
    out = set()
    for sy in systems:
        if sy & set(parent_ring_set):
            continue
        for a in sy:
            atom = mol.GetAtomWithIdx(a)
            if atom.GetSymbol() == 'C' and any(
                    b.GetBondType() == Chem.BondType.DOUBLE
                    and b.GetOtherAtom(atom).GetSymbol() == 'O'
                    and b.GetOtherAtom(atom).GetIdx() not in sy
                    for b in atom.GetBonds()):
                out |= sy
                break
    return out


def name_cyclic_oxo_compound(mol: Chem.Mol,
                             pseudoketone_rings_equal: bool = False) -> Optional[str]:
    """Name an UNSUBSTITUTED cyclic ketone / dione on a mancude ring system,
    emitting the preferred IUPAC name with added indicated hydrogen and/or hydro
    prefixes (IUPAC / / /.

    Worked examples (all OPSIN-2.9.0 round-trip + Blue-Book verified):
        ``O=c1cccc[nH]1`` -> ``pyridin-2(1H)-one``
        ``O=c1ccc2ccccc2[nH]1`` -> ``quinolin-2(1H)-one``
        ``O=c1c2ccccc2[nH]c2ccccc12`` -> ``acridin-9(10H)-one``
        ``O=C1CC=Cc2ccccc21`` -> ``naphthalen-1(2H)-one``
        ``O=C1CCCc2ccccc21`` -> ``3,4-dihydronaphthalen-1(2H)-one``
        ``O=C1C=CC(=O)c2ccccc21`` -> ``naphthalene-1,4-dione`` (no added-H,
        ``O=c1[nH]c(=O)c2ccccc2[nH]1`` -> ``quinazoline-2,4(1H,3H)-dione``
        ``O=C1C=Cc2ccccc21`` -> ``1H-inden-1-one`` (intrinsic-IH parent)
        ``O=C1CCc2ccccc21`` -> ``2,3-dihydro-1H-inden-1-one``

    Method: identify the mancude parent + numbering; the carbonyl C(s) take the
    -one/-dione suffix; ring atoms carrying an EXTRA hydrogen vs the mancude
    parent (an N-H or a >CH2) split, by a maximum matching of the ring graph,
    into hydro positions (matched pairs = reduced ring C=C) and added-indicated-H
    positions (unmatched, cited as ``(nH)`` after the suffix locant). Lowest
    locants go to the suffix, then added-IH, then hydro /.

    Tightly fail-closed (returns None — never a wrong name):
      * >= 1 ring carbonyl; UNSUBSTITUTED (ring + carbonyl O's only);
      * the ring must retain residual unsaturation (an aromatic ring atom OR a
        non-carbonyl ring C=C) — a fully-saturated ring carbonyl is a saturated
        lactam/lactone/ketone named on the saturated parent (piperidin-2-one,
        oxolan-2-one), NOT here;
      * the whole molecule has no retained PIN name (so uracil / maleimide are
        left to their retained entries);
      * the mancude parent must resolve (aromatic monocycle, or a stored PAH /
        fused-heterocycle skeleton).
    """
    from ..data.retained_names import get_retained_name

    rings = mol.GetRingInfo().AtomRings()
    if not rings:
        return None
    all_ring_atoms: Set[int] = set()
    for r in rings:
        all_ring_atoms.update(r)

    # The parent ring system is the FUSED ring system bearing the ring carbonyl —
    # NOT every ring in the molecule. A pendant ring (a phenyl on a chromanone,
    # connected by a single bond, sharing 0-1 atoms) is a SUBSTITUENT, not part of
    # the parent, so it must be excluded from ring_set (else _resolve_oxo_parent
    # can't match the parent skeleton, and the carbonyl/substituent split breaks).
    # Suffix source: a ring exocyclic =O (ketone/lactam/lactone -one,
    # OR, when there is NO ring =O anywhere, a ring exocyclic bare =NH (imine,
    #. Imine is the LAST seniority class #52), so a
    # ring bearing BOTH =O and =NH keeps the -one suffix + 'imino' prefix — that
    # path stays on the =O branch below and is byte-identical. The added-indicated-
    # hydrogen engine (matching + numbering + (nH) formatting) is suffix-agnostic.
    suffix_symbol = 'O'
    suffix_table = {1: 'one', 2: 'dione', 3: 'trione', 4: 'tetrone'}
    suffix_hetero_name = 'oxo'
    all_carbonyls = set(_ring_carbonyl_carbons(mol, all_ring_atoms))
    if not all_carbonyls:
        #: heavier-chalcogen ketone analogues replacement) — thione
        # (=S), then selone (=Se) — are senior to imine and take the -thione /
        # -selone suffix with the SAME added-indicated-H numbering as -one (the
        # engine is suffix-agnostic). Previously these fell through to the
        # substitutive 'sulfanylidene'/'selanylidene' PREFIX (a non-PIN form:
        # `2-sulfanylidene-1H-pyridine` instead of the PIN `pyridine-2(1H)-thione`).
        for _sym, _tab, _hn in (
            ('S', {1: 'thione', 2: 'dithione', 3: 'trithione', 4: 'tetrathione'},
             'sulfanylidene'),
            ('Se', {1: 'selone', 2: 'diselone', 3: 'triselone', 4: 'tetraselone'},
             'selanylidene'),
        ):
            _cs = set(_ring_chalcogenone_carbons(mol, all_ring_atoms, _sym))
            if _cs:
                all_carbonyls = _cs
                suffix_symbol = _sym
                suffix_table = _tab
                suffix_hetero_name = _hn
                break
    nsub_imine_mode = False
    if not all_carbonyls:
        all_carbonyls = set(_ring_imine_carbons(mol, all_ring_atoms))
        if not all_carbonyls:
            # N-substituted ring imine (=N-R), rendered with an N-locant prefix
            # (N1,N4-dimethylnaphthalene-1,4-diimine, /. The
            # quinoid-diimine mirror of the =O quinone path above; the -imine
            # suffix machinery is suffix-agnostic and the N-substituent(s) are
            # rendered after the base name below.
            all_carbonyls = set(_ring_imine_carbons(mol, all_ring_atoms,
                                                    include_nsub=True))
            if not all_carbonyls:
                return None
            nsub_imine_mode = True
        suffix_symbol = 'N'
        suffix_table = {1: 'imine', 2: 'diimine', 3: 'triimine'}
        suffix_hetero_name = 'imino'

    def _suffix_carbons(rs):
        if suffix_symbol == 'O':
            return _ring_carbonyl_carbons(mol, rs)
        if suffix_symbol == 'N':
            return _ring_imine_carbons(mol, rs, include_nsub=nsub_imine_mode)
        return _ring_chalcogenone_carbons(mol, rs, suffix_symbol)

    ring_set: Set[int] = _fused_ring_system_atoms(rings, all_carbonyls)
    if pseudoketone_rings_equal:
        # (branch review fixes): among the ring systems that carry ring
        # carbonyls, the parent is the one with more of them, then the senior one
        # by (the Blue Book-19417) -- not the first one met.
        from .lactones import _ring_system_seniority_key
        _cands = []
        for _c in sorted(all_carbonyls):
            _sy = _fused_ring_system_atoms(rings, {_c})
            if _sy and not any(_sy == _x for _x in _cands):
                _cands.append(_sy)
        if _cands:
            _rs = [set(r) for r in rings]
            ring_set = max(_cands, key=lambda _sy: (
                len(all_carbonyls & _sy), _ring_system_seniority_key(mol, _sy, _rs)))
    if not ring_set:
        return None

    # Wave-B BUILD-1 (parent-selection robustness; C3-a trace Site-3): the carbonyl
    # here is named on its FUSED ring system only (``ring_set``). When that system is
    # SPIRO- or BRIDGE-joined to further ring atoms — an SSSR ring that shares a
    # junction atom with ``ring_set`` yet also reaches outside it — this added-
    # indicated-H parent CANNOT express the joined partner and would silently drop
    # it, leaving an atom-incomplete partial (a wrong molecule that only /the
    # RT gate catches -> abstain). Fail closed so the WHOLE shared-atom ring system
    # routes through the complex_ring/spiro namer instead (which names it when it can,
    # else the RT gate abstains — never a silent drop). A single-bond-linked ring
    # substituent shares NO atom with ``ring_set`` and is unaffected (still named).
    for _r in rings:
        _rset = set(_r)
        if (_rset & ring_set) and (_rset - ring_set):
            return None

    carbonyls = set(_suffix_carbons(ring_set))
    if not carbonyls:
        return None
    # v1 imine gate (fail closed): every imine C's RING neighbours must
    # be carbon — a ring-heteroatom neighbour is a cyclic amidine/imidate shape
    # whose seniority is unresolved (buildable follow-on), never a bare ring imine.
    if suffix_symbol == 'N':
        for c in carbonyls:
            for nb in mol.GetAtomWithIdx(c).GetNeighbors():
                if nb.GetIdx() in ring_set and nb.GetSymbol() != 'C':
                    return None
    # The suffix heteroatom(s) (=O for -one, =NH for -imine) belong to the suffix,
    # not to substituents.
    carbonyl_oxygens: Set[int] = set()
    for c in carbonyls:
        ca = mol.GetAtomWithIdx(c)
        for bond in ca.GetBonds():
            o = bond.GetOtherAtom(ca)
            if (bond.GetBondType() == Chem.BondType.DOUBLE and o.GetSymbol() == suffix_symbol
                    and o.GetIdx() not in ring_set and o.GetDegree() == 1):
                carbonyl_oxygens.add(o.GetIdx())
    # N-substituted ring imine: the imine N is degree-2 (not caught by the
    # degree-1 loop above), so add it to the suffix-atom set and collect its
    # exocyclic substituent subtree for N-locant prefix rendering. nsub_by_carbon
    # maps each imine C -> (N idx, frozenset(substituent atoms), attach C).
    nsub_by_carbon: Dict[int, tuple] = {}
    nsub_all: Set[int] = set()
    if nsub_imine_mode:
        for c in carbonyls:
            ca = mol.GetAtomWithIdx(c)
            n_atom = None
            for bond in ca.GetBonds():
                o = bond.GetOtherAtom(ca)
                if (bond.GetBondType() == Chem.BondType.DOUBLE
                        and o.GetSymbol() == 'N' and o.GetIdx() not in ring_set
                        and o.GetFormalCharge() == 0):
                    n_atom = o
                    break
            if n_atom is None:
                return None
            carbonyl_oxygens.add(n_atom.GetIdx())
            if n_atom.GetDegree() == 1:
                continue  # a bare =NH mixed in — no N-substituent to render
            attach = [nb for nb in n_atom.GetNeighbors() if nb.GetIdx() != c]
            if (len(attach) != 1 or attach[0].GetSymbol() != 'C'
                    or attach[0].GetIdx() in ring_set):
                return None
            sub_atoms = _collect_subtree(mol, attach[0].GetIdx(),
                                         blocked={n_atom.GetIdx()})
            if sub_atoms & (all_ring_atoms | carbonyl_oxygens):
                return None  # substituent loops back into the ring / another suffix N
            # (the Blue Book): an ACYL (or any heteroatom-bearing)
            # N-substituent makes a group SENIOR to the ring imine — =N-C(=O)R is an
            # N-ylidene amide (amide class 11 > imine class 20), not an N-imine
            # prefix. Only a plain alkyl/alkenyl (all-carbon) N-substituent is an
            # N-prefix here; fail closed on anything else, else the -imine suffix
            # would falsely outrank the senior amide (correct N-acyl->amide naming
            # is a buildable follow-on).
            if any(mol.GetAtomWithIdx(a).GetSymbol() != 'C' for a in sub_atoms):
                return None
            nsub_by_carbon[c] = (n_atom.GetIdx(), frozenset(sub_atoms),
                                 attach[0].GetIdx())
            nsub_all |= sub_atoms
    substituent_atoms = {a.GetIdx() for a in atoms_of(mol)
                         if a.GetIdx() not in ring_set
                         and a.GetIdx() not in carbonyl_oxygens
                         and a.GetIdx() not in nsub_all}
    substituted = bool(substituent_atoms)
    # v1 N-substituted-imine scope: the ONLY substituents are on the imine
    # nitrogen(s). A ring substituent alongside them is a buildable follow-on —
    # fail closed (the general renderer would silently drop the N-substituents).
    if nsub_imine_mode and substituted:
        return None
    if substituted:
        # Substituted ring-ketones (monocyclic OR fused): the engine fires only
        # when the ring ketone(s) IS the principal characteristic group, verified
        # via the authoritative seniority machinery (get_principal_group) — NOT a
        # heuristic, since this recognizer PREEMPTS the normal dispatch. Then every
        # substituent is a junior prefix and the heterocycle substituent assembler
        # renders them against this numbering. A pendant ring (the phenyl of a
        # 3-phenylchroman-4-one) is already excluded from ring_set, so it appears as
        # a substituent. Fail closed when the ketone is not the PCG (a senior
        # aldehyde/acid/ester/amide/nitrile substituent) OR when any substituent
        # bears its own acyl/imine/nitrile/thiocarbonyl carbon (a rival the
        # same-class PCG check would not separate).
        from ..perception.functional_groups import detect_functional_groups
        from .seniority import get_principal_group
        # The principal characteristic group must sit on the PARENT ring system —
        # i.e. the ring carbonyl(s) named by the -one suffix (ketone, OR a lactam /
        # lactone ring C=O, all of which names as -one). If the PCG's
        # atoms touch a SUBSTITUENT, a senior group lives there (a substituent acid /
        # aldehyde / sulfonic acid) and the ring -one is not the principal group ->
        # fail closed. (Checking PCG-location, not name, keeps lactam pyridinones /
        # lactone chromen-2-ones in scope while excluding senior substituents.)
        _pcg_name, _pcg_atoms = get_principal_group(mol, detect_functional_groups(mol))
        _pcg_idx = {i for tup in (_pcg_atoms or []) for i in tup}
        # Only the group's CORE locates it: its heteroatoms and its carbons that carry
        # a double bond to a heteroatom. A lactam's tertiary-amide match also lists
        # the N-alkyl carbon (1-methylisatin: (C=O, O, N, CH3, ring C)), which is a
        # substituent on the ring N, not a senior group outside the ring.
        def _is_core(i):
            at = mol.GetAtomWithIdx(i)
            if at.GetAtomicNum() != 6:
                return True
            return any(b.GetBondType() == Chem.BondType.DOUBLE
                       and b.GetOtherAtom(at).GetAtomicNum() not in (1, 6)
                       for b in at.GetBonds())
        # Branch review fixes, the Blue Book): with
        # ``pseudoketone_rings_equal`` (the caller has found THIS ring system the
        # senior one) a ring carbonyl of another ring system -- a lactone, a lactam
        # or a ring ketone -- is a pseudoketone of the same class as the -one here,
        # cited as an 'oxo...yl' prefix: '2-(5-oxooxolan-2-yl)-2,3-dihydro-4H-1-
        # benzopyran-4-one'. The atoms of those ring systems (and their exocyclic
        # =O) are then not a senior rival.
        _peer = (_ring_carbonyl_rings_atoms(mol, ring_set)
                 if pseudoketone_rings_equal else set())
        if _peer:
            _peer |= {n.GetIdx() for a in _peer for n in mol.GetAtomWithIdx(a).GetNeighbors()
                      if n.GetSymbol() == 'O' and not n.IsInRing()}
        if ({i for i in _pcg_idx if _is_core(i)} & substituent_atoms) - _peer:
            return None
        # Belt-and-braces for an EQUAL-class rival the PCG union may not localise: a
        # substituent bearing its own acyl / imine / nitrile / thiocarbonyl carbon.
        for a in substituent_atoms:
            at = mol.GetAtomWithIdx(a)
            if at.GetSymbol() != 'C' or a in _peer:
                continue
            for b in at.GetBonds():
                if (b.GetBondTypeAsDouble() in (2.0, 3.0)
                        and b.GetOtherAtom(at).GetSymbol() in ('O', 'N', 'S', 'Se', 'Te')):
                    return None  # substituent acyl / imine / nitrile / thiocarbonyl
    if get_retained_name(Chem.MolToSmiles(mol)):
        return None  # a retained PIN (uracil, maleimide) owns the name

    # Residual unsaturation required (else fully-saturated lactam/lactone/ketone).
    has_residual = any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_set)
    if has_residual:
        # Branch review fixes: RDKit's aromaticity model also flags a ring whose
        # every double bond is EXOCYCLIC -- the 3,6-bis(ylidene) diketopiperazine
        # 'O=c1[nH]c(=Cc2ccccc2)c(=O)[nH]c1=Cc1ccccc1' -- although no ring bond is
        # double in its (only) Kekule structure. Such a ring is saturated: its ring
        # ketone takes the saturated parent, 'piperazine-2,5-dione' as for the
        # mono-ylidene (':28267 piperidin-2-one (PIN)', ':29325
        # imidazolidine-2,4-dione (PIN)', ':29344 1,3-diazinane-2,4,6-trione (PIN)
        # pyrimidine-2,4,6(1H,3H,5H)-trione'), never a mancude parent with hydro
        # prefixes ('1,3,4,6-tetrahydropyrazine-2,5-dione').
        try:
            _kek = Chem.RWMol(mol)
            Chem.Kekulize(_kek, clearAromaticFlags=True)
            has_residual = any(
                b.GetBondType() == Chem.BondType.DOUBLE
                and b.GetBeginAtomIdx() in ring_set and b.GetEndAtomIdx() in ring_set
                for b in _kek.GetBonds())
        except Exception:  # noqa: BLE001 -- not kekulizable: keep the aromatic reading
            has_residual = True
    if not has_residual:
        for bond in bonds_of(mol):
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if (i in ring_set and j in ring_set
                        and i not in carbonyls and j not in carbonyls):
                    has_residual = True
                    break
    if not has_residual:
        return None

    # The parent is the ring system bearing the carbonyl(s), so a single-ring
    # parent is a monocycle even when a PENDANT ring exists elsewhere (a 1-benzyl
    # or 1-glycosyl uracil, a 1-phenylmaleimide). The historical whole-molecule
    # ring count sent those to the fused catalog, which declines them, and the
    # composer then spelled a Hantzsch-Widman or dihydro form at the PIN tier
    # ('1-benzyl-1H-1,3-diazine-2,4-dione', '1-phenyl-2,5-dihydro-1H-pyrrole-
    # 2,5-dione'). Same test as oxo_prefix_parent_numberings.
    single_ring = any(set(r) == ring_set for r in rings)
    candidates = _resolve_oxo_parent(mol, ring_set, monocyclic=single_ring,
                                     alternative_indicated_h=True)
    if not candidates:
        return None

    # SAT = the ring atoms that are SATURATED relative to the mancude parent
    # (added-IH + hydro), detected STRUCTURALLY: non-carbonyl ring C/N that are
    # NOT in a ring double bond in the KEKULISED molecule. Structural (not H-count)
    # so a SUBSTITUTED indicated-H position still counts (1-methylpyridin-2(1H)-one
    # keeps its (1H); the parent O/S never enters — only C/N can be a hydro/IH
    # position). Independent of numbering, so computed once.
    kek = Chem.Mol(mol)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:
        return None
    mol_ring_db: Set[int] = set()
    for bond in kek.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in ring_set and j in ring_set:
                mol_ring_db.add(i)
                mol_ring_db.add(j)
    sat = {i for i in ring_set
           if i not in carbonyls and i not in mol_ring_db
           and mol.GetAtomWithIdx(i).GetSymbol() in ('C', 'N')}
    adj = {i: set() for i in sat}
    for bond in bonds_of(mol):
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in sat and j in sat:
            adj[i].add(j)
            adj[j].add(i)

    # (g) (the Blue Book): when every earlier criterion ties,
    # "lowest locants for the substituent cited first as a prefix in the name".
    # Name each ring substituent once (its real prefix name, as cited) so the
    # numbering key below can end with (alpha key, locant) pairs -- without it a
    # tie fell to candidate order: COc1cc(OC)c2c(=O)c3c(O)cc(C)cc3oc2c1 shipped
    # '8-hydroxy-1,3-dimethoxy-6-methyl-9H-xanthen-9-one' (PIN '1-hydroxy-6,8-
    # dimethoxy-3-methyl-...', both sets {1,3,6,8}; hydroxy is cited first).
    _sub_names = []  # (ring atom, prefix name)
    if substituent_atoms:
        from ..assembly.naming_utils import alpha_sort_key as _alpha_sort_key
        from ..assembly.substituent_enumerator import name_substituent as _name_sub
        from ..errors import is_refusal_sentinel as _is_refusal
        for _r in ring_set:
            for _nb in mol.GetAtomWithIdx(_r).GetNeighbors():
                _start = _nb.GetIdx()
                if _start not in substituent_atoms:
                    continue
                _frag, _stack = {_start}, [_start]
                while _stack:
                    _x = _stack.pop()
                    for _y in mol.GetAtomWithIdx(_x).GetNeighbors():
                        _yi = _y.GetIdx()
                        if _yi in substituent_atoms and _yi not in _frag:
                            _frag.add(_yi)
                            _stack.append(_yi)
                try:
                    _nm = _name_sub(mol, sorted(_frag), _start)
                except Exception:  # noqa: BLE001 -- no name: no (g) key for it
                    _nm = None
                if _nm and not _is_refusal(_nm):
                    _sub_names.append((_r, _alpha_sort_key(_nm)))

    best = None
    for _cand in candidates:
        parent_name, pairs, ih_locants = _cand[:3]
        # an alternative indicated-hydrogen tautomer ('2H-indole') exists only for
        # (as many indicated hydrogens as suffixes, all at them);
        # with fewer, (1) (:24808) keeps the parent's own lowest-
        # locant indicated hydrogen ('3,7-dihydro-1H-purine-2,6-dione', not
        # '2H-purine-2,6(1H,3H,7H)-dione')
        _alt_only_at_suffixes = len(_cand) > 3 and _cand[3]
        for loc, parentH in pairs:
            if _alt_only_at_suffixes and not (
                    len(ih_locants) == len(carbonyls)
                    and all(loc[c] in ih_locants for c in carbonyls)):
                continue
            # Intrinsic-IH parent (1H-indene, 2H-/4H-chromene, the 2H-/4H-pyran
            # isomers): the ketone is the DIRECT substitution of the parent's
            # >CH2, so a carbonyl MUST sit at one of its indicated-H locants
            #. This disambiguates 4H-chromen-4-one from
            # 2H-chromen-3-one and selects the 2H-/4H-pyran isomer + ring
            # direction. No constraint for a fully mancude parent (ih_locants
            # empty) — its added-IH comes from the maximum matching below.
            if ih_locants and not any(loc[c] in ih_locants for c in carbonyls):
                # (the Blue Book): "'Added indicated hydrogen'
                # atoms are not cited when the accommodation of a pair of
                # principal characteristic groups or free valences simply
                # removes a double bond (directly or after rearrangement of
                # double bonds) from the parent ring structure." A parent whose
                # indicated hydrogen sits on a ring NITROGEN keeps it (the N-H or
                # N-R of the molecule), and the suffixes take the other ring
                # positions: '1H-pyrrole-2,5-dione (PIN)' (:33843), '1-hydroxy-
                # 1H-pyrrole-2,5-dione (PIN)' (:29597). Only that case is named
                # here: every indicated-H atom is an N at a saturated position
                # and NO other ring atom is saturated (no added indicated
                # hydrogen, no hydro prefix); anything else stays declined.
                ih_atoms = {a for a in ring_set if loc[a] in ih_locants}
                if not (ih_atoms and ih_atoms <= sat and not (sat - ih_atoms)
                        and all(mol.GetAtomWithIdx(a).GetSymbol() == 'N'
                                for a in ih_atoms)):
                    continue
                matched, unmatched = set(), set()
            elif len(ih_locants) == len(carbonyls) and all(
                    loc[c] in ih_locants for c in carbonyls):
                # (the Blue Book): as many indicated hydrogen
                # atoms as suffixes, all placed at the suffix positions -- "Locants
                # for hydro prefixes are those of the saturated positions", whether
                # or not they are neighbours: '1,3-dihydro-2H-indol-2-one', not an
                # added indicated hydrogen (TRIAGE j12 finding 5). Mancude parent
                # and molecule both pair off every other double-bond atom, so the
                # saturated set is even; an odd one is not this case.
                if len(sat) % 2:
                    continue
                matched, unmatched = set(sat), set()
            else:
                matched, unmatched = _max_oxo_matching(adj, sat, loc)
            added_ih, hydro = unmatched, matched
            # Substituent locants are the LAST numbering criterion: after
            # the suffix, added-IH and hydro).
            sub_ring = {i for i in ring_set
                        if any(nb.GetIdx() in substituent_atoms
                               for nb in mol.GetAtomWithIdx(i).GetNeighbors())}
            # (4) (the Blue Book): "indicated hydrogen atoms
            #... ha[ve] seniority over 'added indicated hydrogen' for lower
            # locants", and (1) (:24808) puts the parent's indicated hydrogen at the
            # lowest locant consistent with the mancude system: so between two
            # tautomeric parents of one ring system the lower indicated-H locant
            # wins before added-indicated-H is compared ('1H-isoindole-1,3(2H)-
            # dione (PIN)':33853 over '2H-isoindole-1,3-dione'; '1H-cyclopenta
            # [a]naphthalene-1,2(3H)-dione (PIN) (not 3H-...-1,2-dione)':24822).
            # Empty for a fully mancude parent, so it never reorders those.
            key = (sorted(_locant_key(loc[c]) for c in carbonyls),
                   sorted(_locant_key(l) for l in ih_locants),
                   sorted(_locant_key(loc[a]) for a in added_ih),
                   sorted(_locant_key(loc[a]) for a in hydro),
                   sorted(_locant_key(loc[a]) for a in sub_ring),
                   sorted((_ak, _locant_key(loc[_r])) for _r, _ak in _sub_names))
            if best is None or key < best[0]:
                best = (key, loc, added_ih, hydro, parent_name)
    if best is None:
        return None
    _, loc, added_ih, hydro, parent_name = best

    carb_sorted = sorted(carbonyls, key=lambda c: _locant_key(loc[c]))
    carb_str = ','.join(_locant_display(loc[c]) for c in carb_sorted)
    suffix = suffix_table.get(len(carbonyls))
    if suffix is None:
        return None
    if added_ih:
        ih_sorted = sorted(added_ih, key=lambda a: _locant_key(loc[a]))
        ih_str = '(' + ','.join(f"{_locant_display(loc[a])}H" for a in ih_sorted) + ')'
    else:
        ih_str = ''
    # (a): elide terminal 'e' only before a vowel-initial suffix. For the
    # oxo table this is exactly the single-carbonyl '-one' case; for imine it is
    # single '-imine' (vowel-initial) but NOT '-diimine'/'-triimine' (consonant).
    elide = parent_name.endswith('e') and bool(suffix) and suffix[0] in 'aeiou'
    stem = parent_name[:-1] if elide else parent_name
    name = f"{stem}-{carb_str}{ih_str}-{suffix}"
    if hydro:
        prefix = SATURATION_PREFIXES.get(len(hydro))
        if prefix is None:
            return None
        hy_sorted = sorted(hydro, key=lambda a: _locant_key(loc[a]))
        sep = '-' if name[:1].isdigit() else ''  # hyphen before a digit-initial parent (1H-inden...)
        name = f"{','.join(_locant_display(loc[a]) for a in hy_sorted)}-{prefix}{sep}{name}"

    # N-substituted ring imine: render each N-substituent as an N-locant prefix
    # (N1,N4-dimethylnaphthalene-1,4-diimine), grouped by name with N-locants
    # sorted and groups alphabetized. v1 scope: simple (pure-alpha)
    # N-substituents only, and no hydro parent (the detachable-vs-hydro front
    # ordering is a buildable follow-on) — fail closed on either, keeping 0-wrong.
    if nsub_imine_mode and nsub_by_carbon:
        if hydro:
            return None
        from ..assembly.substituent_enumerator import name_substituent
        from ..assembly.naming_utils import get_multiplier_prefix, alpha_sort_key
        from collections import defaultdict
        #: the N-locant carries a number only to distinguish among
        # several imine nitrogens (N1,N4-dimethyl...); a single imine N is cited
        # as the bare italic 'N' (N-methylnaphthalen-2(1H)-imine).
        number_n = len(carbonyls) >= 2
        by_name: Dict[str, List[str]] = defaultdict(list)
        for c, (_n_idx, sub_atoms, attach_idx) in nsub_by_carbon.items():
            sub_name = name_substituent(mol, set(sub_atoms), attach_idx)
            if not sub_name or not sub_name.isalpha():
                return None  # complex N-substituent: out of v1 scope, fail closed
            nloc = f"N{_locant_display(loc[c])}" if number_n else "N"
            by_name[sub_name].append(nloc)
        parts = []
        for nm in sorted(by_name, key=alpha_sort_key):
            nlocs = sorted(by_name[nm], key=lambda s: _locant_key(s[1:]))
            from ..assembly.naming_utils import multiplied_component as _mc
            parts.append(','.join(nlocs) + '-' + _mc(len(nlocs), nm, nm))
        # (a) (the Blue Book): a hyphen separates the prefix from a
        # DIGIT-initial parent ('N-methyl' + '1H-inden-1-imine' ->
        # 'N-methyl-1H-inden-1-imine'); no hyphen before a letter-initial parent
        # ('N1,N4-dimethyl' + 'naphthalene-1,4-diimine'). Mirrors the hydro render.
        sep = '-' if name[:1].isdigit() else ''
        name = '-'.join(parts) + sep + name

    # Substituents (monocyclic, hydrocarbon/halogen): render them as alphabetized
    # prefixes against THIS numbering, with the carbonyl 'oxo' excluded (it is the
    # -one/-dione suffix), reusing the heterocycle substituent assembler. The
    # base `name` (parent + added-IH + suffix, possibly hydro-prefixed) is passed
    # as the parent so prefixes are prepended -> e.g. 5-methylpyridin-2(1H)-one.
    if substituted:
        from .heterocycles import (
            get_heterocycle_substituents,
            name_substituted_heterocycle,
        )
        oriented = [a for a, _ in sorted(loc.items(), key=lambda kv: _locant_key(kv[1]))]
        subs = get_heterocycle_substituents(mol, list(ring_set), oriented, loc)
        filtered = {}
        for locant, slist in subs.items():
            keep = [s for s in slist
                    if s.get('hetero_name') != suffix_hetero_name
                    and not any(a in carbonyl_oxygens for a in s.get('atoms', []))]
            if keep:
                filtered[locant] = keep
        if filtered:
            name = name_substituted_heterocycle(mol, list(ring_set), name, filtered, loc)
    return name


def oxo_prefix_parent_numberings(mol: Chem.Mol, ring_atoms) -> List[tuple]:
    """Parent hydrides for a ring system whose ring C=O is NOT the principal
    characteristic group, so the =O is an 'oxo' PREFIX (the prefix-mode sibling
    of:func:`name_cyclic_oxo_compound`).

     "Prefix nomenclature" (the Blue Book): "After the
    introduction of indicated and 'added indicated hydrogen' atoms, all
    substituent groups not expressed as suffixes are cited as prefixes". An oxo
    prefix substitutes the two hydrogen atoms of a saturated ring position, so
    every ring carbonyl carbon is a SATURATED position of the parent hydride,
    exactly like a ring >CH2 or >N-R: the parent's own indicated hydrogen
     :24639, "in preferred IUPAC names indicated hydrogen must always
    be cited") sits on one of them and the rest are hydro prefixes, cited in
    front of the parent. (PIN) examples: '9,10-dioxo-9,10-
    dihydroanthracene-2-carboxylic acid' (:29471), '5,8-dioxo-5,6,7,8-
    tetrahydronaphthalene-2-carboxylic acid' (:24890), '5-oxo-2,5-dihydrofuran-
    2-carboxylic acid' (:29276), '1,3-dioxo-1,3-dihydro-2H-isoindole-2,5-diyl'
    (:24868), '2,2-dimethyl-1,3-dioxo-2,3-dihydro-1H-isoindol-2-ium' (:41447),
    '2-methyl-4-oxo-3,4-dihydro-1H-2-benzoselenopyran-2-ium-3-ide' (:42439).

    Returns ``[(parent_name, locant_map, ih_atoms, hydro_atoms),...]`` -- one
    entry per mancude parent and numbering the structure allows -- or ````
    (fail closed). ``parent_name`` already carries the hydro prefixes and the
    indicated hydrogen ('3,4-dihydro-2H-1-benzopyran', '1,4-dihydroquinoline',
    '4H-1-benzopyran'); the caller chooses the numbering (b) indicated
    hydrogen, (c) suffixes, (e) hydro prefixes, (f) detachable prefixes).

    Scope (everything else returns ````): ``ring_atoms`` is one whole fused
    (or monocyclic) ring system bearing >=1 ring C=O; every ring atom is
    neutral; every saturated ring position is a C=O carbon, an sp3 carbon or a
    neutral single-bonded nitrogen (no =NR / =S / =CR2 on the ring); the
    mancude parent resolves (:func:`_resolve_oxo_parent`); the parent's
    indicated-hydrogen atoms are saturated in the molecule; the hydro count is
    even (a single left-over position needs 'added indicated hydrogen', which
    only a suffix position may carry,.
    """
    ring_set = set(ring_atoms)
    if not ring_set:
        return []
    rings = mol.GetRingInfo().AtomRings()
    for r in rings:
        rs = set(r)
        if (rs & ring_set) and not rs <= ring_set:
            return []  # part of a larger (fused/spiro/bridged) ring system
    carbonyls = set(_ring_carbonyl_carbons(mol, ring_set))
    if not carbonyls:
        return []
    kek = Chem.Mol(mol)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:
        return []
    ring_db: Set[int] = set()
    for bond in kek.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in ring_set and j in ring_set:
                ring_db.update((i, j))
    sat: Set[int] = set()
    for a in ring_set:
        at = mol.GetAtomWithIdx(a)
        if at.GetFormalCharge() != 0 or at.GetNumRadicalElectrons() != 0:
            return []
        if a in ring_db:
            continue
        if a in carbonyls:
            sat.add(a)
            continue
        kat = kek.GetAtomWithIdx(a)
        if any(b.GetBondType() != Chem.BondType.SINGLE for b in kat.GetBonds()):
            return []  # an exocyclic =X other than the carbonyl O
        sym = at.GetSymbol()
        if sym in ('C', 'N'):
            sat.add(a)
            continue
        ring_deg = sum(1 for nb in at.GetNeighbors() if nb.GetIdx() in ring_set)
        if sym in ('O', 'S', 'Se', 'Te') and ring_deg == 2 and at.GetTotalNumHs() == 0:
            continue  # divalent ring chalcogen: never a hydro position
        return []
    out = []
    single_ring = any(set(r) == ring_set for r in rings)
    for parent_name, pairs, ih_locants in _resolve_oxo_parent(
            mol, ring_set, monocyclic=single_ring):
        for loc, _parent_h in pairs:
            if set(loc) != ring_set:
                continue
            ih_atoms = {a for a in ring_set if loc[a] in ih_locants}
            if len(ih_atoms) != len(ih_locants) or not ih_atoms <= sat:
                continue
            hydro = sat - ih_atoms
            if len(hydro) % 2:
                continue
            name = parent_name
            if hydro:
                prefix = SATURATION_PREFIXES.get(len(hydro))
                if prefix is None:
                    continue
                hy = ','.join(_locant_display(loc[a]) for a in
                              sorted(hydro, key=lambda a: _locant_key(loc[a])))
                sep = '-' if name[:1].isdigit() else ''
                name = f"{hy}-{prefix}{sep}{name}"
            out.append((name, dict(loc), frozenset(ih_atoms), frozenset(hydro)))
    return out


# Back-compat alias: the original (narrower) Phase-2 entry point, now backed by
# the general cyclic-oxo engine.
def name_ring_ketone_with_added_indicated_h(mol: Chem.Mol) -> Optional[str]:
    return name_cyclic_oxo_compound(mol)


# =============================================================================
# Hydro forms of an intrinsic-indicated-H mancude fused carbocycle /
# — the suffix-free sibling of name_cyclic_oxo_compound
# =============================================================================


def name_hydro_mancude_fused_carbocycle(mol: Chem.Mol) -> Optional[str]:
    """Name an UNSUBSTITUTED all-carbon FUSED ring system that is a hydro
    (part- or more-saturated) form of a mancude parent carrying INTRINSIC
    indicated hydrogen.

    Worked targets (all OPSIN-2.9.0 round-trip verified):
        ``c1ccc2c(c1)CCc1ccccc1C2`` -> ``10,11-dihydro-5H-dibenzo[a,d][7]annulene``
            (dibenzosuberane; the amitriptyline / nortriptyline / protriptyline core)
        ``C1=Cc2ccccc2CCC1`` -> ``6,7-dihydro-5H-benzo[7]annulene``
        ``c1ccc2c(c1)CCCCC2`` -> ``6,7,8,9-tetrahydro-5H-benzo[7]annulene``
            (benzosuberane; the benzsuberone precursor)

    This is the suffix-free sibling of:func:`name_cyclic_oxo_compound`: it
    reuses the SAME mancude-parent resolution (:func:`_resolve_oxo_parent`), but
    with NO ring characteristic group. The added 'hydro' positions are the sp3
    ring carbons that are NOT the parent's intrinsic indicated hydrogen, cited by
    lowest locants (b) then (e)) — they need not be an adjacent
    reduced-C=C pair, so no max-matching is used here. The mancude parent already
    spells its own intrinsic
    indicated hydrogen in its name (``5H-...``, (the Blue Book) /
     (:3721)); this function prepends the detachable, nonalphabetized
    ``x,y-dihydro`` hydro prefix / (:1682)), placed just before
    the indicated hydrogen and numbered by lowest locants AFTER it. Order:
    ``[hydro]-[indicated H]-[parent]``.

    Tightly fail-closed (returns None — never a wrong name):
      * one fragment, neutral, non-radical;
      * the WHOLE molecule is one edge-fused ring system: every heavy atom is a
        ring carbon and no ring atom bears an off-ring heavy neighbour, so the
        bare-parent name accounts for every atom (0 silent drop). A substituted
        drug (amitriptyline) therefore declines here and abstains — never wrong;
      * the mancude parent must resolve with a NON-EMPTY intrinsic indicated-H
        locant set — this scopes the path to the annulene / cyclopenta class and
        leaves the naphthalene-family tetralin path (no intrinsic IH) untouched;
      * there must be REAL added hydro (>= 1 reduced ring C=C); the bare mancude
        parent itself is named on the fused-heterocycle path (B1), not here;
      * the parent's declared indicated-H locant(s) must land on sp3 (>CH2)
        ring atoms of the molecule, so the parent name's ``zH-`` already accounts
        for them and no moved / extra added-indicated hydrogen is needed (that
        more general case is deferred, fail closed).
    """
    from ..data.retained_names import get_retained_name

    if mol is None:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    for atom in atoms_of(mol):
        if atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() != 0:
            return None

    # Specified stereo (any @ or /\ in the input): this bare-parent path spells
    # no stereodescriptors, so it would silently OMIT them (a stereo-incomplete
    # name the RT gate would then abstain). Fail closed — never drop stereo.
    for atom in atoms_of(mol):
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return None
    for bond in bonds_of(mol):
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            return None

    rings = mol.GetRingInfo().AtomRings()
    if len(rings) < 2:
        return None
    ring_set: Set[int] = set()
    for r in rings:
        ring_set.update(r)

    # WHOLE-molecule bare carbocycle: every heavy atom is a ring carbon, and no
    # ring atom carries an off-ring heavy neighbour (atom-conservation, 0 drop).
    for atom in atoms_of(mol):
        if atom.GetAtomicNum() <= 1:
            continue
        if atom.GetIdx() not in ring_set or atom.GetSymbol() != 'C':
            return None

    # The rings must all merge into ONE ortho-/peri-fused system (>= 2 rings
    # sharing an edge) — this rejects biphenyl-type single-bond-linked rings and
    # spiro (1 shared atom) systems, which are not one bare fused parent.
    if _fused_ring_system_atoms(rings, ring_set) != ring_set:
        return None

    # A retained-name molecule owns its own name (belt-and-braces).
    if get_retained_name(Chem.MolToSmiles(mol)):
        return None

    candidates = _resolve_oxo_parent(mol, ring_set)
    if not candidates:
        return None

    # SAT = ring carbons NOT in a ring double bond of the kekulised molecule (the
    # sp3 "hydro" / indicated-H positions), exactly as name_cyclic_oxo_compound.
    kek = Chem.Mol(mol)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:
        return None
    mol_ring_db: Set[int] = set()
    for bond in kek.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in ring_set and j in ring_set:
                mol_ring_db.add(i)
                mol_ring_db.add(j)
    # Residual unsaturation is REQUIRED: a fully-saturated (perhydro) fused
    # carbocycle — hydrindane (octahydro-1H-indene), decalin
    # (decahydronaphthalene) — has NO ring double bond and is named in COUNT form
    # (no per-position hydro locants) with stereo by the existing perhydro path.
    # This path is only for HYDRO forms that retain mancude (aromatic or residual
    # C=C) character. Without this guard the producer over-fires on perhydro rings
    # and emits a locant-form, stereo-dropping name (the hydrindane regression).
    if not mol_ring_db:
        return None
    sat = {i for i in ring_set
           if i not in mol_ring_db and mol.GetAtomWithIdx(i).GetSymbol() == 'C'}
    if not sat:
        return None

    # The sp3 ring carbons `sat` are the parent's own intrinsic indicated-H
    # position(s) PLUS the added-'hydro' positions. Each candidate parent
    # declares its indicated-H locants (`ih_locants`); a numbering is valid only
    # when those locants land on sp3 (`sat`) atoms — the parent name's `zH-`
    # >CH2 must be a real sp3 position in the molecule. The remaining sp3 atoms
    # are the added hydro. The hydro positions need NOT be an adjacent
    # (reduced-C=C) pair: a non-adjacent 'added hydrogen' set is standard
    # (1,4-dihydronaphthalene (PIN), the Blue Book), so the hydro locants
    # are chosen directly by lowest-locant sets, NOT by a max-matching (which
    # only realises adjacent pairs and so mis-numbered the 1,4-type isomer).
    # NUMBERING (the Blue Book) assigns low locants in decreasing
    # seniority: (b) indicated hydrogen for unsubstituted compounds (:3246) then
    # (e)(i) 'hydro'/'dehydro' prefixes (:3288). So rank by (ih locants, hydro
    # locants) and take the minimum.
    best = None
    for parent_name, pairs, ih_locants in candidates:
        if not ih_locants:
            continue  # scoped to intrinsic-indicated-H parents only
        ih_key = {_locant_key(loc) for loc in ih_locants}
        for loc, _parentH in pairs:
            ih_atoms = {a for a in ring_set if _locant_key(loc[a]) in ih_key}
            # The parent's declared indicated hydrogen must be a real sp3 >CH2 in
            # the molecule; otherwise this parent/orientation cannot describe it
            # with pure hydro prefixes (a moved/added-indicated H is deferred).
            if not ih_atoms.issubset(sat):
                continue
            hydro = sat - ih_atoms
            if not hydro:
                continue  # no added hydro -> the bare parent (named by B1)
            # Hydro atoms come in even multiples (each 'dihydro' = 2 H); an odd
            # count means this is not a clean hydro form of the parent -> skip.
            if len(hydro) not in SATURATION_PREFIXES:
                continue
            key = (tuple(sorted(_locant_key(loc[a]) for a in ih_atoms)),
                   tuple(sorted(_locant_key(loc[a]) for a in hydro)))
            if best is None or key < best[0]:
                best = (key, loc, hydro, parent_name)
    if best is None:
        return None
    _, loc, hydro, parent_name = best

    prefix = SATURATION_PREFIXES.get(len(hydro))
    if prefix is None:
        return None
    hy_sorted = sorted(hydro, key=lambda a: _locant_key(loc[a]))
    hy = ','.join(_locant_display(loc[a]) for a in hy_sorted)
    # A hyphen before a digit-initial parent (the intrinsic-IH `5H-...` stem).
    sep = '-' if parent_name[:1].isdigit() else ''
    return f"{hy}-{prefix}{sep}{parent_name}"
