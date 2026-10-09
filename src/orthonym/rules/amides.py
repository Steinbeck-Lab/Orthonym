"""
Amide naming rules for IUPAC nomenclature.

Amide naming follows these patterns:
- Primary amides: stem + 'amide' (acetamide, propanamide)
- Secondary amides: N-substituent + stem + 'amide' (N-methylacetamide)
- Tertiary amides: N,N-disubstituent + stem + 'amide' (N,N-dimethylacetamide)
- Ring-attached amides: parent + 'carboxamide' (cyclohexanecarboxamide)

Based on IUPAC 2013 Blue Book.
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional

from rdkit import Chem

from ..assembly.naming_utils import (
    alpha_sort_key,
    get_multiplier_prefix,
    is_complex_substituent,
)
from ..errors import is_refusal_sentinel

# Pattern matching a leading positional locant (digit(s)) in a substituent
# name. Used to detect compound N-substituent names that need
# parenthesization per IUPAC.
_POSITIONAL_LOCANT_RE = re.compile(r'(?:^|\b)\d')


def _has_positional_locants(name: str) -> bool:
    """Whether an N-substituent needs enclosing marks (parentheses).

    a phase assembly/parenthesisation fix (consolidation): this is no longer a divergent
    digit-only test. The audit (06) found that the parenthesization decision
    here disagreed with the bis/tris multiplier decision (get_multiplier_prefix ->
    is_complex_substituent) for digit-less complex substituents like 'chloroethyl'
    -> 'N,N-bischloroethyl' (malformed; PIN 'N,N-bis(2-chloroethyl)'). The fix is
    to unify both decisions onto the ONE correct predicate is_complex_substituent
     /. This delegate is retained so the consolidation tripwire
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
        List of dicts: [{"atoms": [atom_indices], "name": "methyl"},...]
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
            # =S/=Se/=Te for a phase chalcogen amides).
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


def _name_n_substituent(mol, sub_atoms: List[int], carbon_count: int, *,
                        refusal_as_none: bool = False) -> Optional[str]:
    """Name an N-substituent by delegating to the universal naming pipeline.

    ``refusal_as_none``: where the pipeline declines the fragment (the cascade's own
    refusal, ``errors.is_cascade_refusal``) the answer is ``None`` rather than the
    placeholder word; a caller that cites the name asks for it.

    All N-substituent naming is handled by ``name_substituent`` from the
    universal pipeline (a phase), which provides correct IUPAC names via
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

    #.2: when the decorated-ring primitive can name the WHOLE fragment
    # (ring + its substituents, coverage-checked), use that name directly and
    # skip _enrich_ring_n_substituent — the enricher would prepend the same
    # prefixes a second time ('4-methyl' + '4-methylphenyl'). This also
    # replaces the pre-existing double-count of the enricher around retained
    # forms ('4-methyl' + 'toluenyl').
    from ..rules.ring_substituents import decorated_ring_substituent_name
    for ring in mol.GetRingInfo().AtomRings():
        if attach_idx in ring and set(ring) <= sub_set:
            dec = decorated_ring_substituent_name(
                mol, ring, attach_idx, expected_atoms=sub_set)
            if dec is not None:
                return dec
            break

    result = name_substituent(mol, sub_set, attach_idx)
    if result:
        from ..errors import is_cascade_refusal
        if refusal_as_none and is_cascade_refusal(result):
            return None
        result = _enrich_ring_n_substituent(mol, result, sub_atoms)
        return result

    return None


def _ring_system_within(mol, ring_atoms: set, allowed: set) -> set:
    """The atoms of ``allowed`` joined to ``ring_atoms`` through ring bonds: the ring
    system (rings that share atoms) that holds ``ring_atoms``."""
    out, stack = set(ring_atoms), list(ring_atoms)
    while stack:
        cur = stack.pop()
        for bond in mol.GetAtomWithIdx(cur).GetBonds():
            j = bond.GetOtherAtomIdx(cur)
            if bond.IsInRing() and j in allowed and j not in out:
                out.add(j)
                stack.append(j)
    return out


def _enrich_ring_n_substituent(mol, base_name: str, sub_atoms: List[int]) -> str:
    """Enrich a ring-based N-substituent name with sub-substituent prefixes.

    If the N-substituent fragment contains a ring with additional non-ring
    atoms (sub-substituents), this function discovers those sub-substituents
    via the universal pipeline and prepends their IUPAC-locanted prefixes
    to the base ring name.

    Follows the ``_enrich_complex_ring_with_subs`` pattern from a phase.

    Args:
        mol: RDKit Mol object.
        base_name: Name returned by ``name_substituent`` (e.g., "cyclohexyl").
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

    # A ring that is one ring of a polycyclic ring system (fused, bridged fused, von
    # Baeyer, spiro) is not the whole ring stem: name_substituent names such a system
    # whole ('bicyclo[2.2.1]heptan-2-yl', '1,2,3,4-tetrahydro-1,4-ethanonaphthalen-2-yl'),
    # and the system's other ring atoms are not sub-substituents of this one ring
    # (reading them as such built '4-ethan-2-ylbicyclo[2.2.1]heptan-2-yl' and
    # '2-phenyl1,2,3,4-tetrahydro-1,4-ethanonaphthalen-2-yl', different molecules).
    if _ring_system_within(mol, frag_ring_atoms, sub_set) != frag_ring_atoms:
        return base_name

    # name_substituent already FULLY decorates an AROMATIC ring substituent
    # (e.g. '(4-hydroxyphenyl)methyl', '(4-methylphenyl)methyl'), so
    # re-discovering its ring substituents here and prepending them
    # DOUBLE-COUNTS them: '4-hydroxy(4-hydroxyphenyl)methyl' — a parseable
    # WRONG molecule (found by the lever-A gate-OFF honesty sweep on the
    # N-(4-hydroxybenzyl) amide). The enricher is only needed for ALIPHATIC
    # ring stems that name_substituent returns bare (e.g. 'cyclohexyl' ->
    # '4-methylcyclohexyl'); _extract_ring_base_name already returns the full
    # name for aromatic rings, so this branch was pure double-count. Trust the
    # aromatic decoration and return it unchanged.
    if any(mol.GetAtomWithIdx(i).GetIsAromatic() for i in frag_ring_atoms):
        return base_name

    # Check if there are non-ring atoms in the fragment (sub-substituents)
    non_ring_in_frag = sub_set - frag_ring_atoms
    if not non_ring_in_frag:
        return base_name  # Ring-only, nothing to enrich

    # Orient the ring starting from the attachment atom (sub_atoms[0], bonded
    # to nitrogen) as locant 1. Choose the traversal direction that yields
    # the lowest locant set for sub-substituents (IUPAC.
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
    prefix_groups = defaultdict(list)  # name -> [locant,...]
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
            from ..assembly.naming_utils import multiplied_component as _mc
            prefix_parts.append(f"{locant_str}-{_mc(count, name, name)}")

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


def format_n_substitution(substituents: List[Dict]) -> Optional[str]:
    """
    Format N-substituents as IUPAC prefix.

    Rules:
    - Single substituent: "N-methyl"
    - Two identical: "N,N-dimethyl"
    - Two different: "N-ethyl-N-methyl" (alphabetized)

    Args:
        substituents: List of dicts from get_n_substituents

    Returns:
        Formatted N-substitution prefix (e.g., "N-methyl", "N,N-dimethyl"), or
        ``None`` if any substituent name is a refusal sentinel (M2 Task 3
        splice guard, below) -- never a string containing the sentinel.
    """
    if not substituents:
        return ""

    # Group by name
    groups: Dict[str, int] = defaultdict(int)
    for sub in substituents:
        groups[sub["name"]] += 1

    # M2 Task 3 fail-closed splice guard: a substituent name that is a refusal
    # sentinel (the substituent cascade's bare 'substituent' placeholder,
    # 'unknown...', '(not supported)', empty) must never be woven into the
    # N-prefix -- e.g. get_multiplier_prefix(2, 'substituent') builds the
    # literal string 'disubstituent'. VOID this candidate instead; callers
    # already treat a falsy return as "no N-prefix available" and fall back
    # or abstain. Mirrors the guard fragment_naming.py already applies at its
    # name_compound splice point.
    if any(is_refusal_sentinel(name) for name in groups):
        return None

    # Build prefix parts
    parts = []
    for name in sorted(groups.keys(), key=alpha_sort_key):
        count = groups[name]
        # IUPAC / (a phase assembly/parenthesisation fix,): a complex
        # (substituted/compound) substituent is enclosed in parentheses whenever
        # cited — at count 1 AND when multiplied with bis/tris. The paren decision
        # MUST use the SAME predicate (is_complex_substituent) that
        # get_multiplier_prefix uses to choose bis/tris, otherwise digit-less complex
        # names glue ('N,N-bischloroethyl' instead of 'N,N-bis(2-chloroethyl)').
        is_complex = is_complex_substituent(name)
        from ..assembly.naming_utils import _wrap_n_substituent
        if is_complex:
            # nesting ORDER (BB 7444; escalation, BB 7509) under the marks requirement (BB 7232): if the substituent ALREADY carries an
            # inner "(...)" (e.g. a "(2S)-" stereo descriptor), the outer
            # enclosure must escalate to square brackets — "[(2S)-butan-2-yl]",
            # NOT "((2S)-butan-2-yl)". _wrap_n_substituent picks  when an inner
            # paren is present and  otherwise. Applying it to the BARE name (not
            # a pre-parenthesised one) lets it make that choice correctly; a
            # pre-wrap "(name)" would be mis-read as already-balanced and the
            # escalation would be skipped.
            if '(' in name:
                display_name = _wrap_n_substituent(name)
            else:
                display_name = f"({name})"
        else:
            # Simple substituent: no enclosing marks, but still allow bracket
            # escalation when a stereo descriptor introduced an inner paren.
            display_name = _wrap_n_substituent(name)
        if count == 1:
            parts.append(f"N-{display_name}")
        else:
            # Multiple of same: N,N-di... The shared primitive joins the
            # multiplier (and the (c)/(d) marks of 'di(dodecyl)').
            from ..assembly.naming_utils import multiplied_component
            n_locants = ",".join(["N"] * count)
            parts.append(f"{n_locants}-{multiplied_component(count, name, display_name)}")

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
    # for a phase chalcogen amides)
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

    Works for regular amides (=O) and a phase chalcogen amides
    (=S thioamide / =Se selenoamide / =Te telluroamide) per
    +.

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

    #: the acyl parent chain is acyclic — exclude every ring atom
    # (except the acyl carbon itself, always acyclic on this path) so the walker
    # never absorbs a substituent ring into the chain. Mirrors the general cyclic
    # path in namer.py (find_principal_chain(exclude_atoms=all_ring_atoms)).
    for atom in mol.GetAtoms():
        if atom.IsInRing() and atom.GetIdx() != carbonyl_carbon_idx:
            exclude.add(atom.GetIdx())

    chain = _find_longest_carbon_chain(mol, carbonyl_carbon_idx, exclude)
    return len(chain)


def _find_longest_carbon_chain(mol, start_idx: int, exclude: set) -> List[int]:
    """Find longest carbon chain from a starting atom."""

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
            if neighbor.IsInRing():
                # / (1)): the acyclic acyl parent chain of
                # an amide cannot traverse ring atoms — a ring is a separate
                # parent/substituent (heptylbenzene, not a C13 chain). On this
                # path the acyl carbon is guaranteed acyclic (ring-attached amides
                # route through is_ring_attached_amide -> -carboxamide), so a ring
                # atom reached here is always a substituent ring, never chain.
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
             + functional replacement). For chalcogen forms
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
        # chalcogen analogs: "-carbothioamide", "-carboselenoamide",
        # "-carbotelluroamide". OPSIN-confirmed PINs.
        return f"cyclo{stem}anecarbo{suffix_form}"

    # Chain amides
    if suffix_form == "amide":
        # Retained PIN names for regular amides
        if chain_length == 1:
            return "formamide"
        elif chain_length == 2:
            return "acetamide"
        else:
            stem = _get_chain_prefix(chain_length)
            return f"{stem}anamide"

    # a phase chalcogen amides (-thioamide / -selenoamide / -telluroamide):
    # NO retained "thioformamide" / "thioacetamide" forms — systematic only per
    # OPSIN-confirmed PINs methanethioamide / ethanethioamide / propanethioamide.
    # Suffix starts with consonant, so terminal 'e' of '{stem}ane' is preserved
    # per IUPAC (a) vowel-elision rule.
    stem = _get_chain_prefix(chain_length)
    return f"{stem}ane{suffix_form}"


class RingAmideParts:
    """The structural parts of a ring carboxamide name (branch review fixes).

    ``ring_hydride`` ('cyclohexane') and ``suffix`` ('carboxamide', 'carbothioamide')
    for a saturated cycloalkane, or the retained ``benzamide``; ``n_substituents``
    as ``get_n_substituents`` returns them (empty for a primary amide). A composer
    assembles the name from these parts -- never by cutting an emitted name apart.
    """

    __slots__ = ("ring_hydride", "suffix", "retained", "n_substituents")

    def __init__(self, ring_hydride, suffix, retained, n_substituents):
        self.ring_hydride = ring_hydride
        self.suffix = suffix
        self.retained = retained
        self.n_substituents = tuple(n_substituents or ())

    def parent_word(self, suffix_locant: bool = False) -> str:
        """The parent word. ``suffix_locant``: a ring prefix is cited as well, so the
        suffix keeps its locant, the Blue Book "... then all locants
        must be cited"): 'cyclohexane-1-carboxamide'; alone it is omitted
         (c),:2891): 'cyclohexanecarboxamide'. The retained 'benzamide'
        numbers its carbonyl-bearing carbon 1 and cites no suffix locant."""
        if self.retained:
            return self.retained
        if suffix_locant:
            return f"{self.ring_hydride}-1-{self.suffix}"
        return f"{self.ring_hydride}{self.suffix}"


def ring_amide_parts(mol, amide_atoms: tuple,
                     suffix_form: str = "amide") -> Optional[RingAmideParts]:
    """``RingAmideParts`` for a ring-attached amide ``name_amide`` can spell, else
    None (see ``name_amide``)."""
    if not is_ring_attached_amide(mol, amide_atoms):
        return None
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
    is_aromatic_benzene = False
    # The two parents this branch can spell are benzamide and
    # cyclo<alk>anecarboxamide: an ISOLATED all-carbon ring, aromatic benzene
    # or fully saturated. Anything else -- a heteroring (pyridine), a ring with
    # a double bond, a ring fused or spiro-joined to another -- would be spelled
    # as that saturated carbocycle, a different molecule (pyridine-4-carbox-
    # amide was named 'cyclohexanecarboxamide').
    # (the Blue Book,:32680 'thiophene-2-carboxamide (PIN)'): the
    # suffix 'carboxamide' goes on the ring's own name. Decline (None) so a
    # producer that names the ring runs instead.
    spellable = False
    if carbonyl_carbon_idx is not None:
        carbonyl = mol.GetAtomWithIdx(carbonyl_carbon_idx)
        for neighbor in carbonyl.GetNeighbors():
            if neighbor.IsInRing():
                ring_info = mol.GetRingInfo()
                for ring in ring_info.AtomRings():
                    if neighbor.GetIdx() in ring:
                        ring_size = len(ring)
                        ring_atoms = [mol.GetAtomWithIdx(i) for i in ring]
                        _all_c = all(a.GetSymbol() == 'C' for a in ring_atoms)
                        _isolated = all(ring_info.NumAtomRings(i) == 1 for i in ring)
                        _saturated = all(
                            not a.GetIsAromatic() for a in ring_atoms) and all(
                            mol.GetBondBetweenAtoms(ring[k], ring[(k + 1) % len(ring)])
                            .GetBondType() == Chem.BondType.SINGLE
                            for k in range(len(ring)))
                        if ring_size == 6:
                            if (all(a.GetIsAromatic() for a in ring_atoms)
                                    and _all_c):
                                is_aromatic_benzene = True
                        spellable = _all_c and _isolated and (
                            is_aromatic_benzene or _saturated)
                        break
                break
    if not spellable:
        return None
    n_subs = ()
    if get_amide_type(mol, amide_atoms) in ("secondary", "tertiary"):
        n_subs = get_n_substituents(mol, amide_atoms)
    if is_aromatic_benzene:
        return RingAmideParts(None, None, "benzamide", n_subs)
    stem = _get_chain_prefix(ring_size)
    suffix = "carboxamide" if suffix_form == "amide" else f"carbo{suffix_form}"
    return RingAmideParts(f"cyclo{stem}ane", suffix, None, n_subs)


def name_amide(mol, amide_atoms: tuple, suffix_form: str = "amide") -> Optional[str]:
    """
    Generate IUPAC name for an amide compound.

    Handles:
    - Primary amides: acetamide
    - Secondary amides: N-methylacetamide
    - Tertiary amides: N,N-dimethylacetamide; dimethylformamide (a formamide with the same group on both N-H cites no N locants)
    - Ring-attached amides: cyclohexanecarboxamide
    - a phase chalcogen amides (suffix_form="thioamide"/"selenoamide"/"telluroamide"):
      ethanethioamide, N-methylpropaneselenoamide, etc.

    Args:
        mol: RDKit Mol object
        amide_atoms: Atom indices from amide SMARTS match
        suffix_form: PIN suffix variant — "amide" (default) or chalcogen analog
            ("thioamide" / "selenoamide" / "telluroamide", per.

    Returns:
        IUPAC name for the amide
    """
    # Check if ring-attached
    is_ring = is_ring_attached_amide(mol, amide_atoms)

    if is_ring:
        parts = ring_amide_parts(mol, amide_atoms, suffix_form=suffix_form)
        if parts is None:
            return None
        parent_name = parts.parent_word()
        if parts.n_substituents:
            n_prefix = format_n_substitution(list(parts.n_substituents))
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
        _complete = _formamide_completely_substituted(parent_name, amide_type, n_subs)
        if _complete:
            return f"{_complete}{parent_name}"
        n_prefix = format_n_substitution(n_subs)
        if n_prefix:
            return f"{n_prefix}{parent_name}"

    return parent_name


def _formamide_completely_substituted(parent_name, amide_type, n_subs) -> Optional[str]:
    """The N-prefix of a formamide whose two N-H are both replaced by the same
    substituent, cited without locants; None otherwise.

     (the Blue Book): "All locants are omitted in compounds... in
    which all substitutable positions are completely substituted... in the same
    way"; formamide's substitutable hydrogens are the two N-H -- a C-substituted
    formamide takes another parent ('carbonochloridic amide (PIN) (not
    1-chloroformamide)',:32707) -- so prints 'dimethylformamide
    (PIN)' (:32782), while a single N-substituent keeps its locant ('N-phenyl
    formamide (PIN)',:32855). Deny-by-default: two different substituents, a
    stereo-bearing or isotopic scope, a scope that is part of a larger name."""
    try:
        if parent_name != "formamide" or amide_type != "tertiary":
            return None
        if not n_subs or len(n_subs) != 2:
            return None
        names = {sub.get("name") for sub in n_subs}
        if len(names) != 1:
            return None
        name = next(iter(names))
        if not name or is_refusal_sentinel(name):
            return None
        from ..assembly.locant_omission import (
            locants_are_forced,
            scope_has_isotopic_modification,
        )
        if locants_are_forced() or scope_has_isotopic_modification():
            return None
        from ..assembly.handlers._handler_shared import locant_scope_is_a_name_component
        if locant_scope_is_a_name_component():
            return None
        from ..assembly.naming_utils import enclose_if_compound, multiplied_component
        return multiplied_component(2, name, enclose_if_compound(name))
    except Exception:  # noqa: BLE001 -- deny by default
        return None


def name_chain_diamide(
    mol, all_amide_matches: List[tuple], chain: List[int],
    atom_to_locant: Dict[int, int],
) -> Optional[str]:
    """Name an acyclic chain diamide with optional N-substituents (D3).

    Handles the mixed primary + N-substituted (and symmetric primary /
    symmetric secondary) acyclic diamide class per BB
    (parent = ``{stem}anediamide``) and (N-substituents cited
    as ``N{locant}`` prefixes where the locant is the chain-carbon locant of
    the amide carbonyl; identical substituents on both ends give the
    ``N1,N4-di...`` form).

    Fail-closed: returns ``None`` (so the caller falls through to the existing
    polyfunctional / general_acyclic path unchanged) whenever any precondition
    is not met, or any N-substituent cannot be named.

    Args:
        mol: RDKit Mol object.
        all_amide_matches: list of amide SMARTS match tuples (primary and/or
            secondary), carbonyl carbon at tuple index 0.
        chain: the principal chain atom-index list (the diamide backbone).
        atom_to_locant: mapping from chain atom index -> locant.

    Returns:
        The PIN name string, or ``None`` (fail-closed).
    """
    # TIGHT predicate: exactly two amide groups.
    if not all_amide_matches or len(all_amide_matches) != 2:
        return None
    chain_len = len(chain)
    if chain_len < 2:
        return None

    end_locants = {1, chain_len}

    # Both carbonyl carbons (tuple index 0) must sit at chain-END locants.
    # Collect (locant, match) so the amide identity travels with its position.
    positioned = []
    for match in all_amide_matches:
        carbonyl_c = match[0]
        locant = atom_to_locant.get(carbonyl_c)
        if locant is None or locant not in end_locants:
            return None
        positioned.append((locant, match))

    # Guard against both amides mapping to the same chain end (degenerate).
    if {loc for loc, _ in positioned} != end_locants:
        return None

    # Extract N-substituents at each end: (locant, [names]).
    per_end = []  # list of (locant, [sub_name,...])
    covered = set(chain)
    for locant, match in positioned:
        covered.update(match)
        subs = get_n_substituents(mol, match)
        names = []
        for sub in subs:
            name = sub.get("name")
            if not name:
                return None  # un-nameable N-substituent -> fail-closed
            names.append(name)
            covered.update(sub.get("atoms") or ())
        per_end.append((locant, names))

    # This body cites the N-substituents of a saturated chain only. A
    # substituent on a chain carbon ('2-methylpropanediamide (PIN)',,
    # the Blue Book) or a chain double or triple bond ('but-2-enediamide')
    # is left to the general path, which cites it; naming the bare saturated
    # parent here dropped it.
    if any(atom.GetAtomicNum() > 1 and atom.GetIdx() not in covered
           for atom in mol.GetAtoms()):
        return None
    _chain_set = set(chain)
    for _bond in mol.GetBonds():
        if (_bond.GetBeginAtomIdx() in _chain_set
                and _bond.GetEndAtomIdx() in _chain_set
                and _bond.GetBondTypeAsDouble() != 1.0):
            return None

    # Orientation: choose forward vs reversed numbering so the
    # N-substituents get the lowest locant set. The chain carries two
    # numbering directions; the amide carbonyls are fixed at {1, chain_len}.
    # For each candidate orientation compute the N-locant assigned to each
    # substituted end and pick the lower sorted locant set.
    def _n_locants_for(reverse: bool):
        out = []
        for locant, names in per_end:
            eff = (chain_len + 1 - locant) if reverse else locant
            for _ in names:
                out.append(eff)
        return sorted(out)

    fwd = _n_locants_for(False)
    rev = _n_locants_for(True)
    reverse = rev < fwd

    # Build (effective_locant, name) pairs across both ends.
    placed = []  # (eff_locant, name)
    for locant, names in per_end:
        eff = (chain_len + 1 - locant) if reverse else locant
        for name in names:
            placed.append((eff, name))

    # Parent name: {stem}anediamide. chain_len==2 -> ethanediamide.
    stem = _get_chain_prefix(chain_len)
    if not stem:
        return None
    parent_name = f"{stem}anediamide"

    if not placed:
        # Symmetric unsubstituted diamide (e.g. butanediamide) — plain parent.
        return parent_name

    # Group by substituent name, collecting its N-locants.
    groups: Dict[str, List[int]] = defaultdict(list)
    for eff_locant, name in placed:
        groups[name].append(eff_locant)

    # Build prefix parts, alphabetised by substituent name.
    parts = []  # (sort_key, prefix_string)
    for name in groups:
        locants = sorted(groups[name])
        is_complex = is_complex_substituent(name)
        from ..assembly.naming_utils import _wrap_n_substituent
        if is_complex:
            display_name = _wrap_n_substituent(name) if '(' in name else f"({name})"
        else:
            display_name = _wrap_n_substituent(name)
        n_locant_str = ",".join(f"N{loc}" for loc in locants)
        if len(locants) == 1:
            prefix = f"{n_locant_str}-{display_name}"
        else:
            from ..assembly.naming_utils import multiplied_component
            prefix = (f"{n_locant_str}-"
                      f"{multiplied_component(len(locants), name, display_name)}")
        parts.append((alpha_sort_key(name), prefix))

    parts.sort(key=lambda p: p[0])
    n_prefix_string = "-".join(p[1] for p in parts)

    return f"{n_prefix_string}{parent_name}"
