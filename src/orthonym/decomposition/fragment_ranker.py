"""Fragment seniority ranking for decomposition parent/substituent assignment.

Implements IUPAC seniority-based fragment ranking to determine which
fragment becomes the parent structure in decomposition naming. Uses existing
seniority.py and functional_groups.py infrastructure -- this is a thin wrapper.

Ranking criteria (applied in order until tie-broken):
1. Most principal characteristic groups
2. Most senior principal group type
3. Senior heteroatom: N > O > S
4. Ring system present beats chain-only
5. Most heavy atoms size tiebreaker)

Scores are tuples where LOWER = MORE SENIOR (natural sort order).
"""

from typing import Dict, List, Optional, Tuple

from rdkit import Chem

from ..perception.functional_groups import detect_functional_groups
from ..rules.seniority import SENIORITY_ORDER, get_principal_group

# Sentinel rank for fragments with no detectable principal group
_NO_FG_RANK = 999


def score_fragment_seniority(frag_smiles: Optional[str]) -> Tuple[int, int, int, int, int]:
    """Score a fragment SMILES by seniority.

    Returns a comparison tuple where LOWER = MORE SENIOR:
        (-pg_count, pg_seniority_rank, heteroatom_rank, -has_ring, -num_heavy_atoms)

    Args:
        frag_smiles: SMILES string for the fragment. None or invalid
            SMILES returns the lowest possible seniority.

    Returns:
        5-tuple of integers for comparison. Lower tuple = more senior fragment.
    """
    # Handle None / empty / invalid
    if not frag_smiles:
        return (0, _NO_FG_RANK, 3, 0, 0)

    mol = Chem.MolFromSmiles(frag_smiles)
    if mol is None:
        return (0, _NO_FG_RANK, 3, 0, 0)

    # Detect functional groups using the existing perception layer
    fgs = detect_functional_groups(mol)

    # Get the most senior principal group
    pg_name, pg_matches = get_principal_group(mol, fgs)

    if pg_name is not None:
        pg_count = len(pg_matches)
        try:
            pg_rank = SENIORITY_ORDER.index(pg_name)
        except ValueError:
            pg_rank = _NO_FG_RANK
    else:
        pg_count = 0
        pg_rank = _NO_FG_RANK

    # Heteroatom rank: N > O > S > none senior heteroatom)
    hetero_rank = _heteroatom_rank(mol)

    # Ring presence
    has_ring = 1 if mol.GetRingInfo().NumRings() > 0 else 0

    # Heavy atom count size tiebreaker)
    n_heavy = mol.GetNumHeavyAtoms()

    # Return tuple: LOWER = MORE SENIOR
    # Negate pg_count, has_ring, n_heavy so that MORE = LOWER
    return (-pg_count, pg_rank, hetero_rank, -has_ring, -n_heavy)


def _heteroatom_rank(mol) -> int:
    """Determine the senior heteroatom rank for a fragment.

    Per: N > O > S > none.

    Returns:
        0 for N present, 1 for O present, 2 for S present, 3 for none.
    """
    has_n = False
    has_o = False
    has_s = False
    for atom in mol.GetAtoms():
        anum = atom.GetAtomicNum()
        if anum == 7:
            has_n = True
        elif anum == 8:
            has_o = True
        elif anum == 16:
            has_s = True

    if has_n:
        return 0
    if has_o:
        return 1
    if has_s:
        return 2
    return 3


def rank_fragments(fragments: List[Dict]) -> List[Dict]:
    """Rank fragment dicts by seniority (most senior first).

    Args:
        fragments: List of dicts, each must have a "smiles" key.
            All other keys are preserved.

    Returns:
        New list sorted by seniority (most senior first = lowest score).
    """
    return sorted(fragments, key=lambda f: score_fragment_seniority(f.get("smiles")))


def acid_is_more_senior(acid_smiles: str, other_smiles: str) -> bool:
    """Check if the acid fragment is more senior or equal to the other fragment.

    Convenience function for engine.py assembly decisions. When this returns
    False, the other fragment (amine/alkyl) is strictly more senior and should
    become the parent.

    Args:
        acid_smiles: SMILES of the acid-side fragment.
        other_smiles: SMILES of the other fragment (amine/alkyl).

    Returns:
        True if acid is more senior or equal (normal assembly).
        False if other is strictly more senior (reverse assembly).
    """
    acid_score = score_fragment_seniority(acid_smiles)
    other_score = score_fragment_seniority(other_smiles)
    return acid_score <= other_score
