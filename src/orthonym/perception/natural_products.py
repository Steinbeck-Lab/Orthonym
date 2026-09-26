"""
Natural product scaffold detection via RDKit substructure matching.

IUPAC: Retained names for natural product ring systems (steroids,
terpenoids, alkaloids). These parent scaffolds have IUPAC-recommended retained
names that take precedence over systematic von Baeyer nomenclature.

Uses flexible query patterns (bond-generic) so that unsaturated derivatives
(e.g. cholesterol with C=C in ring A) still match their saturated parent
scaffold (cholestane). Stereochemistry is also relaxed in patterns to allow
matching molecules with undefined or different stereocenters.

IUPAC: Steroid parent hydrides use retained names (androstane,
pregnane, cholestane, etc.) with modification for unsaturation.

Provides:
- detect_natural_product: Main detection entry point
- get_non_scaffold_atoms: Identify atoms not part of matched scaffold
- get_scaffold_substituents: Identify substituents attached to scaffold
- is_steroid: Quick steroid check
- is_alkaloid: Quick alkaloid check
"""

import logging
from collections import deque
from typing import Dict, List, Optional, Set

from rdkit import Chem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level cache for flexible query patterns
# ---------------------------------------------------------------------------
_FLEXIBLE_PATTERNS: Dict[str, Chem.Mol] = {}
_PATTERNS_INITIALIZED = False


def _init_flexible_patterns() -> None:
    """Build bond-generic, stereo-free query molecules for scaffold matching.

    Each scaffold SMILES is converted to a query where:
    - Bond types are generic (single matches double, etc.)
    - Stereochemistry is removed
    This allows derivatives with unsaturation or different stereo to still
    match their parent scaffold.
    """
    global _PATTERNS_INITIALIZED
    if _PATTERNS_INITIALIZED:
        return

    try:
        from orthonym.data.natural_products import (
            NATURAL_PRODUCT_SCAFFOLDS,
            get_scaffold_patterns,
        )
    except ImportError:
        logger.warning("Natural product data module not available")
        _PATTERNS_INITIALIZED = True
        return

    raw_patterns = get_scaffold_patterns()
    for smiles, mol in raw_patterns.items():
        # Create a copy to avoid modifying the data module's patterns
        mol_copy = Chem.RWMol(mol)
        Chem.RemoveStereochemistry(mol_copy)
        params = Chem.AdjustQueryParameters()
        params.makeBondsGeneric = True
        params.adjustDegree = False
        params.adjustRingCount = False
        query = Chem.AdjustQueryProperties(mol_copy, params)
        if query is not None:
            _FLEXIBLE_PATTERNS[smiles] = query

    _PATTERNS_INITIALIZED = True


def _get_flexible_patterns() -> Dict[str, Chem.Mol]:
    """Return flexible query patterns, initializing on first call."""
    if not _PATTERNS_INITIALIZED:
        _init_flexible_patterns()
    return _FLEXIBLE_PATTERNS


# ---------------------------------------------------------------------------
# BFS helper for substituent collection
# ---------------------------------------------------------------------------

def _bfs_substituent(mol, start_idx: int, exclude_set: set) -> list:
    """BFS from start_idx collecting connected non-scaffold atoms.

    Args:
        mol: RDKit Mol object.
        start_idx: Starting atom index (first atom of substituent).
        exclude_set: Set of atom indices to exclude (scaffold atoms).

    Returns:
        List of atom indices in the substituent fragment.
    """
    visited = {start_idx}
    queue = deque([start_idx])
    result = [start_idx]
    while queue:
        current = queue.popleft()
        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in exclude_set:
                visited.add(nbr_idx)
                queue.append(nbr_idx)
                result.append(nbr_idx)
    return result


# ---------------------------------------------------------------------------
# Main detection functions
# ---------------------------------------------------------------------------

def detect_natural_product(mol) -> Optional[Dict]:
    """Memoising front of:func:`_detect_natural_product_impl` (perf lever A7, 2026-09-13).

    ``classify_compound_class``, ``name_natural_product`` and ``select_parent_unified`` each
    call this on the same Mol in one pipeline pass (5,787 calls per 300 molecules, 1.2 s).
    The flexible scaffold queries are bond-generic and stereo-free, so the match depends on
    the atoms' elements and charges (editable in place) and on connectivity (not editable on
    a ``Chem.Mol``); the element/charge tuple is the memo key. The dict is copied on every
    call (its ``non_scaffold_atoms`` set is mutable); ``matched_atoms`` is a tuple.
    """
    if mol is None:
        return None
    try:
        from .molcache import atoms_of, cached_by_key
        sig = tuple((a.GetAtomicNum(), a.GetFormalCharge()) for a in atoms_of(mol))
    except Exception:
        return _detect_natural_product_impl(mol)
    hit = cached_by_key(mol, "natural_product", sig, lambda: _detect_natural_product_impl(mol))
    if hit is None:
        return None
    return {**hit, "non_scaffold_atoms": set(hit["non_scaffold_atoms"])}


def _detect_natural_product_impl(mol) -> Optional[Dict]:
    """Detect natural product scaffold in a molecule via substructure matching.

    Uses flexible (bond-generic, stereo-free) query patterns so that both
    saturated parent scaffolds and unsaturated derivatives are correctly
    identified. When multiple scaffolds match, the LARGEST match (most atoms)
    wins, giving the most specific scaffold identification.

    Args:
        mol: RDKit Mol object. Returns None if mol is None.

    Returns:
        Dict with keys:
            - scaffold_name: str (e.g. "androstane")
            - scaffold_stem: str (e.g. "androst")
            - scaffold_class: str (e.g. "steroid")
            - scaffold_smiles: str (canonical SMILES of matched scaffold)
            - matched_atoms: tuple of int (atom indices in scaffold)
            - non_scaffold_atoms: set of int (atom indices NOT in scaffold)
        Returns None if no scaffold matches or if inputs are invalid.
    """
    if mol is None:
        return None

    try:
        from orthonym.data.natural_products import NATURAL_PRODUCT_SCAFFOLDS
    except ImportError:
        logger.warning("Natural product data module not available")
        return None

    patterns = _get_flexible_patterns()
    if not patterns:
        logger.warning("No scaffold patterns compiled")
        return None

    # Collect all matches. When multiple scaffolds match, prefer by class
    # priority (steroid > alkaloid > terpene), then by size within same class.
    # This prevents large terpene scaffolds (prostane: 20 atoms) from
    # outranking smaller but more specific steroid scaffolds (estrane: 18).
    _CLASS_PRIORITY = {"steroid": 0, "alkaloid": 1, "terpene": 2}
    best_match = None
    best_match_size = 0
    best_smiles = None
    best_class_priority = 99

    for smiles, query_mol in patterns.items():
        if mol.HasSubstructMatch(query_mol):
            match = mol.GetSubstructMatch(query_mol)
            scaffold_info = NATURAL_PRODUCT_SCAFFOLDS[smiles]
            cls_priority = _CLASS_PRIORITY.get(scaffold_info["class"], 50)
            # Prefer higher-priority class; within same class, prefer larger
            if (cls_priority < best_class_priority or
                    (cls_priority == best_class_priority and len(match) > best_match_size)):
                best_match = match
                best_match_size = len(match)
                best_smiles = smiles
                best_class_priority = cls_priority

    if best_match is None:
        return None

    best_match = _canonical_scaffold_match(mol, patterns[best_smiles], best_smiles, best_match)
    scaffold_info = NATURAL_PRODUCT_SCAFFOLDS[best_smiles]
    non_scaffold = get_non_scaffold_atoms(mol, best_match)

    return {
        "scaffold_name": scaffold_info["name"],
        "scaffold_stem": scaffold_info["stem"],
        "scaffold_class": scaffold_info["class"],
        "scaffold_smiles": best_smiles,
        "matched_atoms": best_match,
        "non_scaffold_atoms": non_scaffold,
    }


# classes of one scaffold decoration, in the order the NP assembler
# (rules/natural_products.py) expresses them: an ester makes the scaffold the
# '-yl' of the ester name (free valence, `_assemble_np_ester_name`); otherwise a
# ketone is the '-one' suffix with hydroxy demoted to a prefix; otherwise a
# hydroxy is the '-ol' suffix.
_NP_SUFFIX_RANK = {'fv': 0, 'one': 1, 'ol': 2}
_NP_DEMOTED_PREFIX = {'one': 'oxo', 'ol': 'hydroxy'}
_NP_HALOGEN_PREFIX = {9: 'fluoro', 17: 'chloro', 35: 'bromo', 53: 'iodo'}


def _np_decoration_kind(mol, idx: int, nbr, in_scaffold: set) -> Optional[str]:
    """Class of the decoration ``nbr`` on scaffold atom ``idx``: a suffix class
    ('fv' / 'one' / 'ol'), a prefix name the NP assembler cites (fluoro, chloro,
    bromo, iodo, methyl, methoxy), or ``None`` for any other prefix (still a
    detachable prefix for (f); unnamed, so it blocks the (g) tier)."""
    from rdkit import Chem as _Chem
    bond = mol.GetBondBetweenAtoms(idx, nbr.GetIdx())
    z = nbr.GetAtomicNum()
    if z == 8:
        if bond.GetBondType() == _Chem.BondType.DOUBLE:
            return 'one'
        others = [x for x in nbr.GetNeighbors() if x.GetIdx() != idx]
        if not others:
            return 'ol'
        if any(x.GetIdx() in in_scaffold for x in others):
            return None  # an O bridging two scaffold atoms
        for x in others:
            if x.GetAtomicNum() in (6, 15, 16) and any(
                    y.GetAtomicNum() in (8, 16)
                    and mol.GetBondBetweenAtoms(x.GetIdx(), y.GetIdx()).GetBondType()
                    == _Chem.BondType.DOUBLE
                    for y in x.GetNeighbors()):
                return 'fv'  # O-acyl / O-sulfonyl / O-phosphoryl: the scaffold is the '-yl'
        if (len(others) == 1 and others[0].GetAtomicNum() == 6
                and others[0].GetDegree() == 1):
            return 'methoxy'
        return None
    if z in _NP_HALOGEN_PREFIX and bond.GetBondType() == _Chem.BondType.SINGLE:
        return _NP_HALOGEN_PREFIX[z]
    if (z == 6 and nbr.GetDegree() == 1
            and bond.GetBondType() == _Chem.BondType.SINGLE):
        return 'methyl'
    return None


def _canonical_scaffold_match(mol, query_mol, scaffold_smiles: str, first_match: tuple) -> tuple:
    """Choose the scaffold match by the numbering rules, not by atom order.

    ``GetSubstructMatch`` returns the FIRST match RDKit finds, and for a
    symmetric scaffold (tropane: C1<->C5, C2<->C4, C6<->C7) which match comes
    first depends on how the input SMILES is written, so the same molecule got
    '(1R,3r,5S)-tropan-3-yl...' or '(1S,3r,5R)-tropan-3-yl...' (full gate,
    determinism). The derivative is named by the rules of to
    , the Blue Book), so among all matches pick by
    ``### **** NUMBERING`` (:3219), in order:

      (c):3256 the suffix / free valence -- the senior of ester '-yl', '-one',
          '-ol' present, exactly as the NP assembler chooses it;
      (e):3288 "low locants are given first to multiple bonds as a set and then
          to double bonds";
      (f):3301 the detachable prefixes "all considered together" (a demoted
          ketone/hydroxy counts as oxo/hydroxy here);
      (g):3307 the prefix cited first, by alphanumerical name (only when every
          prefix has a known name, so a partial list never ranks against the rule);
      (j):3346 the CIP stereodescriptors with their locants
          (``naming_utils.cip_locant_rank_key``);
      then the canonical atom ranks, which do not depend on the input spelling
      (any match still tied here gives the same name).

    Lumping the suffix in with the prefixes (the old first tier) put the prefix
    first: '(1R,2S,5S,7S)-2-methyltropan-7-ol' where (c) gives
    '(1S,4S,5R,6S)-4-methyltropan-6-ol'.
    """
    try:
        from orthonym.data.natural_products import get_scaffold_numbering
        numbering = get_scaffold_numbering(scaffold_smiles)
        matches = mol.GetSubstructMatches(query_mol, uniquify=False, maxMatches=256)
    except Exception:
        return first_match
    if not numbering or len(matches) < 2:
        return first_match
    try:
        from rdkit import Chem as _Chem
        from orthonym.perception.stereo import assign_stereochemistry
        from orthonym.assembly.naming_utils import cip_locant_rank_key
        assign_stereochemistry(mol)
        ranks = list(_Chem.CanonicalRankAtoms(mol, breakTies=True))
    except Exception:
        return first_match

    def _loc_key(loc):
        s = str(loc)
        digits = ''.join(ch for ch in s if ch.isdigit())
        return (int(digits) if digits else 0, s)

    def key(match):
        in_scaffold = set(match)
        loc = {idx: numbering.get(pos) for pos, idx in enumerate(match)}
        decos = []  # (kind, locant key), one per scaffold -> outside bond
        for idx in match:
            if loc.get(idx) is None:
                continue
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                if nbr.GetIdx() not in in_scaffold and nbr.GetAtomicNum() > 1:
                    decos.append((_np_decoration_kind(mol, idx, nbr, in_scaffold),
                                  _loc_key(loc[idx])))
        present = [_NP_SUFFIX_RANK[k] for k, _ in decos if k in _NP_SUFFIX_RANK]
        top = min(present) if present else None
        principal = tuple(sorted(l for k, l in decos
                                 if top is not None and _NP_SUFFIX_RANK.get(k) == top))
        multiple, double = [], []
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if (a in in_scaffold and b in in_scaffold
                    and loc.get(a) is not None and loc.get(b) is not None
                    and not bond.GetIsAromatic()
                    and bond.GetBondType() in (_Chem.BondType.DOUBLE,
                                               _Chem.BondType.TRIPLE)):
                pair = tuple(sorted((_loc_key(loc[a]), _loc_key(loc[b]))))
                multiple.append(pair)
                if bond.GetBondType() == _Chem.BondType.DOUBLE:
                    double.append(pair)
        prefixes = [(_NP_DEMOTED_PREFIX.get(k, k), l) for k, l in decos
                    if top is None or _NP_SUFFIX_RANK.get(k) != top]
        prefix_locs = tuple(sorted(l for _, l in prefixes))
        first_cited = ()
        if prefixes and all(isinstance(p, str) and p not in _NP_SUFFIX_RANK
                            for p, _ in prefixes):
            first_cited = tuple(sorted(prefixes))
        cip = [(_loc_key(loc[idx]), mol.GetAtomWithIdx(idx).GetProp('_CIPCode'))
               for idx in match
               if loc.get(idx) is not None and mol.GetAtomWithIdx(idx).HasProp('_CIPCode')]
        return (principal, tuple(sorted(multiple)), tuple(sorted(double)),
                prefix_locs, first_cited, cip_locant_rank_key(cip),
                tuple(ranks[i] for i in match))

    return min(matches, key=key)


def get_non_scaffold_atoms(mol, matched_atoms: tuple) -> set:
    """Return atom indices NOT part of the matched scaffold.

    Args:
        mol: RDKit Mol object.
        matched_atoms: Tuple of atom indices that are part of the scaffold.

    Returns:
        Set of atom indices not in the scaffold.
    """
    all_atoms = set(range(mol.GetNumAtoms()))
    return all_atoms - set(matched_atoms)


def get_scaffold_substituents(mol, matched_atoms: tuple) -> List[Dict]:
    """Identify substituents attached to the scaffold.

    For each scaffold atom, checks its neighbors. If a neighbor is not part
    of the scaffold, a BFS is performed to collect the full substituent group.

    Args:
        mol: RDKit Mol object.
        matched_atoms: Tuple of atom indices that are part of the scaffold.

    Returns:
        List of dicts, each with:
            - attachment_atom: int (scaffold atom idx where substituent attaches)
            - substituent_atoms: list of int (all atoms in the substituent)
            - first_atom: int (first atom of substituent, bonded to scaffold)
    """
    scaffold_set = set(matched_atoms)
    visited_substituent_atoms: Set[int] = set()
    substituents = []

    for scaffold_idx in matched_atoms:
        atom = mol.GetAtomWithIdx(scaffold_idx)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in scaffold_set and nbr_idx not in visited_substituent_atoms:
                # Found a new substituent starting at nbr_idx
                sub_atoms = _bfs_substituent(mol, nbr_idx, scaffold_set)
                visited_substituent_atoms.update(sub_atoms)
                substituents.append({
                    "attachment_atom": scaffold_idx,
                    "substituent_atoms": sub_atoms,
                    "first_atom": nbr_idx,
                })

    return substituents


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def is_steroid(mol) -> bool:
    """Quick check: does molecule contain a steroid scaffold?

    Args:
        mol: RDKit Mol object. Returns False if mol is None.

    Returns:
        True if molecule contains a steroid scaffold.
    """
    if mol is None:
        return False
    result = detect_natural_product(mol)
    if result is None:
        return False
    return result["scaffold_class"] == "steroid"


def is_alkaloid(mol) -> bool:
    """Quick check: does molecule contain an alkaloid scaffold?

    Args:
        mol: RDKit Mol object. Returns False if mol is None.

    Returns:
        True if molecule contains an alkaloid scaffold.
    """
    if mol is None:
        return False
    result = detect_natural_product(mol)
    if result is None:
        return False
    return result["scaffold_class"] == "alkaloid"
