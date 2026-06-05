"""
Amide naming rules for IUPAC nomenclature.

Amide naming follows these patterns:
- Primary amides: stem + 'amide' (acetamide, propanamide)
- Secondary amides: N-substituent + stem + 'amide' (N-methylacetamide)
- Tertiary amides: N,N-disubstituent + stem + 'amide' (N,N-dimethylacetamide)
- Ring-attached amides: parent + 'carboxamide' (cyclohexanecarboxamide)

Based on IUPAC 2013 Blue Book P-66.1.
"""

import re
from typing import List, Optional, Dict, Any
from collections import defaultdict

from rdkit import Chem

from ..assembly.naming_utils import (
    get_alkyl_name,
    get_multiplier_prefix,
    alpha_sort_key,
    is_complex_substituent,
    ALKYL_NAMES,
)

# Pattern matching a leading positional locant (digit(s)) in a substituent
# name.  Used to detect compound N-substituent names that need
# parenthesization per IUPAC P-14.5.2.
_POSITIONAL_LOCANT_RE = re.compile(r'(?:^|\b)\d')


def _has_positional_locants(name: str) -> bool:
    """Whether an N-substituent needs enclosing marks (parentheses).

    Phase 171 BBR-ASM (DEF-8 consolidation): this is no longer a divergent
    digit-only test. The audit (06 §3.1) found that the parenthesization decision
    here disagreed with the bis/tris multiplier decision (get_multiplier_prefix ->
    is_complex_substituent) for digit-less complex substituents like 'chloroethyl'
    -> 'N,N-bischloroethyl' (malformed; PIN 'N,N-bis(2-chloroethyl)'). The fix is
    to unify both decisions onto the ONE correct predicate is_complex_substituent
    (P-16.3.5 / P-16.5.1.1). This delegate is retained so the consolidation tripwire
    (tests/unit/assembly/test_needs_parens_consolidation.py) sees the two views agree.
    """
    return is_complex_substituent(name)


# Standard stems - delegated to centralized chain_names module
from ..data.chain_names import get_chain_prefix as _get_chain_prefix
STEM_PREFIXES = {i: _get_chain_prefix(i) for i in range(1, 21)}


def get_amide_type(mol, amide_atoms: tuple) -> str:
    """
    Determine if an amide is primary, secondary, or tertiary.

    The amide nitrogen determines the type:
    - Primary: -C(=O)NH2 (N has 2 hydrogens)
    - Secondary: -C(=O)NHR (N has 1 hydrogen, 1 substituent)
    - Tertiary: -C(=O)NR2 (N has 0 hydrogens, 2 substituents)

    Args:
        mol: RDKit Mol object
        amide_atoms: Atom indices from amide SMARTS match

    Returns:
        "primary", "secondary", or "tertiary"
    """
    # Find the nitrogen atom in the amide
    nitrogen_idx = None
    for idx in amide_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'N':
            nitrogen_idx = idx
            break

    if nitrogen_idx is None:
        return "primary"

    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)

    # Count hydrogens (explicit + implicit)
    num_h = nitrogen.GetTotalNumHs()

    if num_h >= 2:
        return "primary"
    elif num_h == 1:
        return "secondary"
    else:
        return "tertiary"


def get_n_substituents(mol, amide_atoms: tuple) -> List[Dict]:
    """
    Get substituents attached to the amide nitrogen.

    For secondary/tertiary amides, find the carbon substituents
    attached to nitrogen (not the carbonyl carbon).

    Args:
        mol: RDKit Mol object
        amide_atoms: Atom indices from amide SMARTS match

    Returns:
        List of dicts: [{"atoms": [atom_indices], "name": "methyl"}, ...]
    """
    substituents = []

    # Find the nitrogen and carbonyl carbon
    nitrogen_idx = None
    carbonyl_carbon_idx = None

    for idx in amide_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'N':
            nitrogen_idx = idx
        elif atom.GetSymbol() == 'C':
            # Find the chalcogen-double-bonded carbon (=O for regular amides,
            # =S/=Se/=Te for Phase 163 chalcogen amides).
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() in _AMIDE_CHALCOGEN_ELEMENTS:
                    bond = mol.GetBondBetweenAtoms(idx, neighbor.GetIdx())
                    if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                        carbonyl_carbon_idx = idx
                        break

    if nitrogen_idx is None:
        return substituents

    nitrogen = mol.GetAtomWithIdx(nitrogen_idx)

    # Find carbon substituents on nitrogen (excluding the carbonyl carbon)
    for neighbor in nitrogen.GetNeighbors():
        nbr_idx = neighbor.GetIdx()
        if neighbor.GetSymbol() != 'C':
            continue
        if nbr_idx == carbonyl_carbon_idx:
            continue

        # This is a substituent on nitrogen
        sub_atoms = _bfs_substituent(mol, nbr_idx, {nitrogen_idx, carbonyl_carbon_idx})
        carbon_count = sum(1 for idx in sub_atoms if mol.GetAtomWithIdx(idx).GetSymbol() == 'C')

        # Check if substituent is aromatic ring-based
        sub_name = _name_n_substituent(mol, sub_atoms, carbon_count)
        if sub_name:
            substituents.append({
                "atoms": sub_atoms,
                "name": sub_name,
                "carbon_count": carbon_count,
            })

    return substituents


def _name_n_substituent(mol, sub_atoms: List[int], carbon_count: int) -> Optional[str]:
    """Name an N-substituent by delegating to the universal naming pipeline.

    All N-substituent naming is handled by ``name_substituent()`` from the
    universal pipeline (Phase 85), which provides correct IUPAC names via
    a five-tier cascade: retained names (isopropyl, phenyl, tert-butyl),
    fragment cache, linear alkyl fast path, recursive naming, and fallback.

    This replaces the previous approach that used only carbon count to
    produce linear alkyl names, causing cyclopentyl -> "pentyl",
    isopropyl -> "propyl", pyridinyl -> "pentyl", etc.
    """
    if not sub_atoms:
        return None

    from ..assembly.substituent_enumerator import name_substituent
    sub_set = set(sub_atoms)
    # attach_idx is the first atom in BFS order (directly bonded to nitrogen)
    attach_idx = sub_atoms[0]
    result = name_substituent(mol, sub_set, attach_idx)
    if result:
        result = _enrich_ring_n_substituent(mol, result, sub_atoms)
        return result

    return None


def _enrich_ring_n_substituent(mol, base_name: str, sub_atoms: List[int]) -> str:
    """Enrich a ring-based N-substituent name with sub-substituent prefixes.

    If the N-substituent fragment contains a ring with additional non-ring
    atoms (sub-substituents), this function discovers those sub-substituents
    via the universal pipeline and prepends their IUPAC-locanted prefixes
    to the base ring name.

    Follows the ``_enrich_complex_ring_with_subs()`` pattern from Phase 119.

    Args:
        mol: RDKit Mol object.
        base_name: Name returned by ``name_substituent()`` (e.g., "cyclohexyl").
        sub_atoms: List of atom indices in the N-substituent fragment (BFS order).

    Returns:
        Enriched name with locanted sub-substituent prefixes, or the original
        ``base_name`` unchanged if no ring or no sub-substituents found.
    """
    from ..assembly.substituent_enumerator import discover_substituents, name_substituent

    sub_set = set(sub_atoms)
    ring_info = mol.GetRingInfo()

    # Find the first ring whose atoms are entirely within the fragment
    frag_ring_atoms = None
    for ring in ring_info.AtomRings():
        ring_set = set(ring)
        if ring_set.issubset(sub_set):
            frag_ring_atoms = ring_set
            break

    if not frag_ring_atoms:
        return base_name  # No ring in fragment

    # Check if there are non-ring atoms in the fragment (sub-substituents)
    non_ring_in_frag = sub_set - frag_ring_atoms
    if not non_ring_in_frag:
        return base_name  # Ring-only, nothing to enrich

    # Orient the ring starting from the attachment atom (sub_atoms[0], bonded
    # to nitrogen) as locant 1.  Choose the traversal direction that yields
    # the lowest locant set for sub-substituents (IUPAC P-31.1.3).
    attach_atom = sub_atoms[0]  # first atom in BFS = bonded to N
    oriented_ring = _orient_ring_from_attachment(mol, frag_ring_atoms, attach_atom)

    if not oriented_ring:
        return base_name

    # Discover sub-substituents on the ring
    try:
        subs = discover_substituents(
            mol, frag_ring_atoms, "ring", oriented_ring=tuple(oriented_ring)
        )
    except Exception:
        return base_name

    if not subs:
        return base_name

    # Name and collect sub-substituent prefixes, filtering to only those
    # whose atoms are within the N-substituent fragment scope
    prefix_groups = defaultdict(list)  # name -> [locant, ...]
    for sub_info in subs:
        frag_atoms_set = set(sub_info.frag_atoms)
        # Skip sub-substituents outside the N-substituent scope (e.g., the
        # amide chain attached back through nitrogen)
        if not frag_atoms_set.issubset(sub_set):
            continue
        # Skip oversized fragments (same guard as _enrich_complex_ring_with_subs)
        if len(sub_info.frag_atoms) > 12:
            continue

        a_idx = sub_info.attach_mol_idx
        if a_idx not in frag_atoms_set:
            a_idx = next(iter(frag_atoms_set))

        prefix_name = name_substituent(mol, sub_info.frag_atoms, a_idx)
        if not prefix_name or prefix_name == "substituent":
            continue
        if ' ' in prefix_name:
            continue

        locant = sub_info.locant
        if locant is not None:
            prefix_groups[prefix_name].append(locant)

    if not prefix_groups:
        return base_name

    # Build prefix parts with locants and multiplier prefixes
    prefix_parts = []
    for name, locants in prefix_groups.items():
        locants.sort(key=lambda x: (0, x) if isinstance(x, int) else (1, str(x)))
        count = len(locants)
        if count == 1:
            prefix_parts.append(f"{locants[0]}-{name}")
        else:
            locant_str = ",".join(str(loc) for loc in locants)
            multiplier = get_multiplier_prefix(count, name)
            prefix_parts.append(f"{locant_str}-{multiplier}{name}")

    prefix_parts.sort(key=lambda x: alpha_sort_key(x))

    # Use the ring stem name as the base (e.g., "cyclohexyl")
    ring_base = _extract_ring_base_name(base_name, frag_ring_atoms, mol)
    prefix_str = "-".join(prefix_parts)
    return f"{prefix_str}{ring_base}" if prefix_str else base_name


def _extract_ring_base_name(full_name: str, ring_atoms: set, mol) -> str:
    """Extract the bare ring stem name from a potentially enriched name."""
    ring_size = len(ring_atoms)
    is_aromatic = any(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring_atoms)
    if is_aromatic:
        return full_name
    from ..data.chain_names import get_chain_prefix
    stem = get_chain_prefix(ring_size)
    base = f"cyclo{stem}yl"
    if base in full_name:
        return base
    return full_name


def _orient_ring_from_attachment(mol, ring_atoms: set, attach_atom: int) -> List[int]:
    """Orient a ring starting from the attachment atom, choosing the direction
    that gives the lowest locant set for sub-substituents."""
    if attach_atom not in ring_atoms:
        for nbr in mol.GetAtomWithIdx(attach_atom).GetNeighbors():
            if nbr.GetIdx() in ring_atoms:
                attach_atom = nbr.GetIdx()
                break
        else:
            return []

    ring_adj = {}
    for idx in ring_atoms:
        atom = mol.GetAtomWithIdx(idx)
        ring_adj[idx] = [
            nbr.GetIdx() for nbr in atom.GetNeighbors()
            if nbr.GetIdx() in ring_atoms
        ]

    start_neighbors = ring_adj.get(attach_atom, [])
    if len(start_neighbors) < 2:
        return sorted(ring_atoms)

    def _traverse(start, first_neighbor):
        path = [start, first_neighbor]
        prev, current = start, first_neighbor
        while len(path) < len(ring_atoms):
            nxt = [n for n in ring_adj[current] if n != prev]
            if not nxt:
                break
            path.append(nxt[0])
            prev, current = current, nxt[0]
        return path

    dir1 = _traverse(attach_atom, start_neighbors[0])
    dir2 = _traverse(attach_atom, start_neighbors[1])

    def _sub_locants(oriented):
        locant_map = {a: i + 1 for i, a in enumerate(oriented)}
        locs = []
        for idx in oriented:
            atom = mol.GetAtomWithIdx(idx)
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() not in ring_atoms and nbr.GetAtomicNum() > 1:
                    locs.append(locant_map[idx])
                    break
        return sorted(locs)

    locs1 = _sub_locants(dir1)
    locs2 = _sub_locants(dir2)

    for a, b in zip(locs1, locs2):
        if a < b:
            return dir1
        elif b < a:
            return dir2

    if len(locs1) <= len(locs2):
        return dir1
    return dir2


def _bfs_substituent(mol, start_idx: int, exclude: set) -> List[int]:
    """BFS to find all atoms in a substituent."""
    from collections import deque

    visited = set()
    queue = deque([start_idx])
    atoms = []

    while queue:
        atom_idx = queue.popleft()
        if atom_idx in visited or atom_idx in exclude:
            continue
        visited.add(atom_idx)
        atoms.append(atom_idx)

        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in exclude:
                queue.append(nbr_idx)

    return atoms


def format_n_substitution(substituents: List[Dict]) -> str:
    """
    Format N-substituents as IUPAC prefix.

    Rules:
    - Single substituent: "N-methyl"
    - Two identical: "N,N-dimethyl"
    - Two different: "N-ethyl-N-methyl" (alphabetized)

    Args:
        substituents: List of dicts from get_n_substituents()

    Returns:
        Formatted N-substitution prefix (e.g., "N-methyl", "N,N-dimethyl")
    """
    if not substituents:
        return ""

    # Group by name
    groups: Dict[str, int] = defaultdict(int)
    for sub in substituents:
        groups[sub["name"]] += 1

    # Build prefix parts
    parts = []
    for name in sorted(groups.keys(), key=alpha_sort_key):
        count = groups[name]
        # IUPAC P-16.3.5 / P-16.5.1.1 (Phase 171 BBR-ASM, DEF-8): a complex
        # (substituted/compound) substituent is enclosed in parentheses whenever
        # cited — at count 1 AND when multiplied with bis/tris. The paren decision
        # MUST use the SAME predicate (is_complex_substituent) that
        # get_multiplier_prefix uses to choose bis/tris, otherwise digit-less complex
        # names glue ('N,N-bischloroethyl' instead of 'N,N-bis(2-chloroethyl)').
        is_complex = is_complex_substituent(name)
        display_name = f"({name})" if is_complex else name
        # Apply P-16.3.3 bracket escalation for N-substituents with parens
        from ..assembly.naming_utils import _wrap_n_substituent
        display_name = _wrap_n_substituent(display_name)
        if count == 1:
            parts.append(f"N-{display_name}")
        else:
            # Multiple of same: N,N-di...
            multiplier = get_multiplier_prefix(count, name)
            n_locants = ",".join(["N"] * count)
            parts.append(f"{n_locants}-{multiplier}{display_name}")

    # Join parts with hyphen
    return "-".join(parts)


def is_ring_attached_amide(mol, amide_atoms: tuple) -> bool:
    """
    Check if an amide is attached to a ring.

    A ring-attached amide has its carbonyl carbon directly
    bonded to a ring carbon. These use -carboxamide suffix.

    Args:
        mol: RDKit Mol object
        amide_atoms: Atom indices from amide SMARTS match

    Returns:
        True if the amide is ring-attached
    """
    # Find the carbonyl-like carbon (=O regular amide, or =S/=Se/=Te
    # for Phase 163 chalcogen amides)
    carbonyl_carbon_idx = None
    for idx in amide_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'C':
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() in _AMIDE_CHALCOGEN_ELEMENTS:
                    bond = mol.GetBondBetweenAtoms(idx, neighbor.GetIdx())
                    if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                        carbonyl_carbon_idx = idx
                        break
            if carbonyl_carbon_idx:
                break

    if carbonyl_carbon_idx is None:
        return False

    carbonyl = mol.GetAtomWithIdx(carbonyl_carbon_idx)

    # Check if carbonyl carbon has a neighbor in a ring (excluding the
    # principal-group leaf N and the double-bonded chalcogen).
    for neighbor in carbonyl.GetNeighbors():
        if neighbor.GetSymbol() == 'N':
            continue  # Skip the nitrogen
        if neighbor.GetSymbol() in _AMIDE_CHALCOGEN_ELEMENTS:
            continue  # Skip the carbonyl chalcogen (=O / =S / =Se / =Te)
        if neighbor.IsInRing():
            return True

    return False


_AMIDE_CHALCOGEN_ELEMENTS = ('O', 'S', 'Se', 'Te')


def get_amide_chain_length(mol, amide_atoms: tuple) -> int:
    """
    Get the chain length for an amide (including carbonyl carbon).

    Works for regular amides (=O) and Phase 163 chalcogen amides
    (=S thioamide / =Se selenoamide / =Te telluroamide) per P-66.1.4.1.1
    + P-66.6.3.

    The chain length determines the parent name:
    - 1: formamide / methanethioamide / methaneselenoamide
    - 2: acetamide / ethanethioamide / ethaneselenoamide
    - 3: propanamide / propanethioamide / propaneselenoamide
    etc.

    Args:
        mol: RDKit Mol object
        amide_atoms: Atom indices from amide SMARTS match

    Returns:
        Chain length (number of carbons in parent chain)
    """
    # Find the carbonyl carbon (the C with a double bond to any chalcogen
    # O / S / Se / Te) and the nitrogen.
    carbonyl_carbon_idx = None
    nitrogen_idx = None

    for idx in amide_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == 'N':
            nitrogen_idx = idx
        elif atom.GetSymbol() == 'C':
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() in _AMIDE_CHALCOGEN_ELEMENTS:
                    bond = mol.GetBondBetweenAtoms(idx, neighbor.GetIdx())
                    if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                        carbonyl_carbon_idx = idx
                        break

    if carbonyl_carbon_idx is None:
        return 1

    # BFS to find longest chain from carbonyl carbon (excluding nitrogen direction
    # and the double-bonded chalcogen, since that's the principal-group leaf).
    exclude = {nitrogen_idx} if nitrogen_idx else set()
    for idx in amide_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() in _AMIDE_CHALCOGEN_ELEMENTS:
            exclude.add(idx)

    chain = _find_longest_carbon_chain(mol, carbonyl_carbon_idx, exclude)
    return len(chain)


def _find_longest_carbon_chain(mol, start_idx: int, exclude: set) -> List[int]:
    """Find longest carbon chain from a starting atom."""
    from collections import deque

    best_chain = [start_idx]

    def dfs(current_idx: int, path: List[int], visited: set):
        nonlocal best_chain

        if len(path) > len(best_chain):
            best_chain = path.copy()

        atom = mol.GetAtomWithIdx(current_idx)
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in visited or nbr_idx in exclude:
                continue
            if neighbor.GetSymbol() != 'C':
                continue

            visited.add(nbr_idx)
            path.append(nbr_idx)
            dfs(nbr_idx, path, visited)
            path.pop()
            visited.discard(nbr_idx)

    visited = {start_idx}
    dfs(start_idx, [start_idx], visited)

    return best_chain


def get_amide_parent_name(
    chain_length: int,
    is_ring: bool = False,
    suffix_form: str = "amide",
) -> str:
    """
    Get the parent amide name based on chain length and suffix form.

    Args:
        chain_length: Number of carbons in parent chain
        is_ring: If True, use -carboxamide form
        suffix_form: One of "amide", "thioamide", "selenoamide", "telluroamide"
            (P-66.1.4.1.1 + P-66.6.3 functional replacement). For chalcogen forms
            the IUPAC PIN preserves the parent-stem terminal 'e' (e.g.,
            'propanethioamide') because the suffix starts with a consonant.

    Returns:
        Parent amide name (e.g., "formamide", "acetamide", "propanamide",
        "ethanethioamide", "propaneselenoamide").
    """
    if is_ring:
        # Ring-attached amides: parent + carboxamide / carbothioamide / etc.
        stem = _get_chain_prefix(chain_length)
        if suffix_form == "amide":
            return f"cyclo{stem}anecarboxamide"
        # P-66.1.4.1.1 chalcogen analogs: "-carbothioamide", "-carboselenoamide",
        # "-carbotelluroamide". OPSIN-confirmed PINs.
        return f"cyclo{stem}anecarbo{suffix_form}"

    # Chain amides
    if suffix_form == "amide":
        # Retained PIN names for regular amides (P-66.1.1.1.1)
        if chain_length == 1:
            return "formamide"
        elif chain_length == 2:
            return "acetamide"
        else:
            stem = _get_chain_prefix(chain_length)
            return f"{stem}anamide"

    # Phase 163 chalcogen amides (-thioamide / -selenoamide / -telluroamide):
    # NO retained "thioformamide" / "thioacetamide" forms — systematic only per
    # OPSIN-confirmed PINs methanethioamide / ethanethioamide / propanethioamide.
    # Suffix starts with consonant, so terminal 'e' of '{stem}ane' is preserved
    # per IUPAC P-16.3.3 vowel-elision rule.
    stem = _get_chain_prefix(chain_length)
    return f"{stem}ane{suffix_form}"


def name_amide(mol, amide_atoms: tuple, suffix_form: str = "amide") -> str:
    """
    Generate IUPAC name for an amide compound.

    Handles:
    - Primary amides: acetamide
    - Secondary amides: N-methylacetamide
    - Tertiary amides: N,N-dimethylformamide
    - Ring-attached amides: cyclohexanecarboxamide
    - Phase 163 chalcogen amides (suffix_form="thioamide"/"selenoamide"/"telluroamide"):
      ethanethioamide, N-methylpropaneselenoamide, etc.

    Args:
        mol: RDKit Mol object
        amide_atoms: Atom indices from amide SMARTS match
        suffix_form: PIN suffix variant — "amide" (default) or chalcogen analog
            ("thioamide" / "selenoamide" / "telluroamide", per P-66.1.4.1.1).

    Returns:
        IUPAC name for the amide
    """
    # Check if ring-attached
    is_ring = is_ring_attached_amide(mol, amide_atoms)

    if is_ring:
        # Find the ring and get its size
        carbonyl_carbon_idx = None
        for idx in amide_atoms:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetSymbol() == 'C':
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'O':
                        bond = mol.GetBondBetweenAtoms(idx, neighbor.GetIdx())
                        if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                            carbonyl_carbon_idx = idx
                            break

        ring_size = 6  # Default
        if carbonyl_carbon_idx is not None:
            carbonyl = mol.GetAtomWithIdx(carbonyl_carbon_idx)
            for neighbor in carbonyl.GetNeighbors():
                if neighbor.IsInRing():
                    ring_info = mol.GetRingInfo()
                    for ring in ring_info.AtomRings():
                        if neighbor.GetIdx() in ring:
                            ring_size = len(ring)
                            break
                    break

        parent_name = get_amide_parent_name(
            ring_size, is_ring=True, suffix_form=suffix_form,
        )

        # Check for N-substitution
        amide_type = get_amide_type(mol, amide_atoms)
        if amide_type in ("secondary", "tertiary"):
            n_subs = get_n_substituents(mol, amide_atoms)
            n_prefix = format_n_substitution(n_subs)
            if n_prefix:
                return f"{n_prefix}{parent_name}"

        return parent_name

    # Chain amide
    chain_length = get_amide_chain_length(mol, amide_atoms)
    parent_name = get_amide_parent_name(chain_length, suffix_form=suffix_form)

    # Check for N-substitution
    amide_type = get_amide_type(mol, amide_atoms)
    if amide_type in ("secondary", "tertiary"):
        n_subs = get_n_substituents(mol, amide_atoms)
        n_prefix = format_n_substitution(n_subs)
        if n_prefix:
            return f"{n_prefix}{parent_name}"

    return parent_name
