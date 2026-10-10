"""
Chain detection and principal chain selection.

Implements IUPAC 2013 rules for selecting the principal chain.
Key change in IUPAC 2013: Chain length takes priority over unsaturation!
"""

import os as _os
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem

from ..rules.lambda_convention import nonstandard_bonding_number
from .molcache import atoms_of
from ..rules.locants import compare_locant_sets  # perf lever A10 (2026-09-13): hoisted (23,785 executions per 300 molecules)

# a phase (heteroatom-only-suffix acids): characteristic-heteroatom
# (atomic number) for FG classes whose SMARTS matches S/P + O with NO carbon
# atom in the match tuple. sulfonic_acid, sulfinic_acid, phosphonic_acid and
# the sulfonic-family imidic/peroxoic/thioic S variants all use a RECURSIVE
# carbon guard (e.g. sulfonic_acid "[SX4;$([SX4][#6])](=O)(=O)[OX2H1]") -- the
# bonded carbon is only a validity filter, never a literal pattern atom, so it
# is ABSENT from the match tuple (verified: GetSubstructMatches on
# CCCCS(=O)(=O)O returns only the S,O,O,O indices, no carbon).
#
# Without a characteristic-heteroatom entry for these FG names, the `_het_z`
# lookup in find_principal_chain (below) stayed None for them, so its "het not
# found" legacy fallback registered only the S/P/O atoms into fg_atoms and
# NEVER the bearing carbon (the carbon directly bonded to S/P). Criterion 1
# ("chain contains the principal characteristic group", was then False
# for every candidate carbon chain -- tying at 0 -- so criterion 3 (max chain
# length) handed the parent to a longer chain that does not carry the acid at
# all (e.g. the acyl chain of a taurine amide: CCC(=O)NCCS(=O)(=O)O wrongly
# parented as "propanesulfonic acid" instead of the ethanesulfonic-acid-bearing
# chain). This mapping feeds the SAME `_het_z` bearing-carbon branch already
# used for alcohol/amine (seniority._CLASS_CHARACTERISTIC_Z), so their bearing
# carbons are now computed identically -- registering the acid-bearing carbon
# into fg_atoms lets criterion 1 pick the correct chain before length is ever
# consulted.
#
# phosphinic_acid's SMARTS ("[PX4](=O)([OX2H1])([#6])[#6]") already carries
# both carbons as LITERAL match atoms (not recursive), so it has no live bug
# here, but is included for consistency with its sulfonic/phosphonic-family
# siblings -- verified harmless: the het-found branch recomputes the same
# carbons as bearing carbons, and P/O atoms (present in fg_atoms either way)
# can never appear in a carbon-chain's atom set, so fg_count/fg_atoms are
# unaffected by the branch switch.
_HETEROACID_CHARACTERISTIC_Z: Dict[str, int] = {
    "sulfonic_acid": 16, "sulfinic_acid": 16,
    "sulfonoperoxoic_acid": 16, "sulfonothioic_S_acid": 16,
    "sulfonimidic_acid": 16, "sulfinimidic_acid": 16,
    "phosphonic_acid": 15, "phosphinic_acid": 15,
    # a phase Wave 2 (#4): same bearing-carbon defect for the remaining
    # Group-15/16 oxoacid FG classes (functional_groups.py ~94-114). arsonic_acid
    # and stibonic_acid use the same recursive carbon guard as sulfonic_acid
    # ("[AsX4;$([AsX4][#6])](=O)([OX2H1])[OX2H1]") -- carbon absent from the
    # match tuple -- so As/Sb were already MASKED (they name via a substituent-
    # prefix fallback) rather than defect-free; added here for parity/robustness.
    # selenonic_acid/seleninic_acid/telluronic_acid/tellurinic_acid are the SAME
    # recursive-guard shape and were live abstentions (longer competing carbon
    # chain wins the parent) until this entry. arsinic_acid/stibinic_acid carry
    # both carbons as LITERAL match atoms (like phosphinic_acid) so have no live
    # bug, but are included for consistency -- harmless, see phosphinic_acid note
    # above.
    "arsonic_acid": 33, "arsinic_acid": 33,
    "stibonic_acid": 51, "stibinic_acid": 51,
    "selenonic_acid": 34, "seleninic_acid": 34,
    "telluronic_acid": 52, "tellurinic_acid": 52,
    # a phase: trivalent -ous analogues — same central-heteroatom Z as their
    # -onic/-inic siblings (P=15, As=33, Sb=51), same recursive carbon-guard shape.
    "phosphonous_acid": 15, "phosphinous_acid": 15,
    "arsonous_acid": 33, "arsinous_acid": 33,
    "stibonous_acid": 51, "stibinous_acid": 51,
}


def find_all_carbon_chains(
    mol,
    min_length: int = 1,
    exclude_atoms: Optional[Set[int]] = None
) -> List[List[int]]:
    """
    Find all carbon chains in a molecule using DFS.

    Args:
        mol: RDKit Mol object
        min_length: Minimum chain length to return
        exclude_atoms: Optional set of atom indices to skip (e.g., ring atoms)

    Returns:
        List of lists, each inner list contains atom indices of a chain
    """
    chains = []
    exclude = exclude_atoms or set()

    def dfs(atom_idx: int, visited: Set[int], path: List[int]):
        # Skip excluded atoms (e.g., ring atoms when finding chain through ring)
        if atom_idx in exclude:
            return

        atom = mol.GetAtomWithIdx(atom_idx)

        # Only follow carbon atoms (not heteroatoms)
        if atom.GetSymbol() != 'C':
            return

        visited.add(atom_idx)
        path.append(atom_idx)

        # Record this path if it meets minimum length
        if len(path) >= min_length:
            chains.append(path.copy())

        # Explore neighbors
        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited:
                dfs(nbr_idx, visited, path)

        # Backtrack
        path.pop()
        visited.discard(atom_idx)

    # Start DFS from each carbon atom to find all possible chains
    for atom in atoms_of(mol):
        if atom.GetSymbol() == 'C' and atom.GetIdx() not in exclude:
            dfs(atom.GetIdx(), set(), [])

    return chains


def find_maximal_carbon_chains(
    mol,
    exclude_atoms: Optional[Set[int]] = None
) -> List[List[int]]:
    """The paths of ``find_all_carbon_chains(mol, 1, exclude_atoms)`` that cannot be
    extended at either end, in the SAME relative order as in that list.

    A path is extendable when an end atom has a carbon neighbour that is neither
    excluded nor on the path; the extended path is then itself one of the
    enumerated paths (the DFS below walks every simple path of non-excluded carbons
    from every start). ``find_principal_chain`` only ever needs these: see the
    proof at its call site.

    Cost: the full enumeration stores every simple path, n^2 paths of mean length
    n/3 for an unbranched C_n chain, and ``find_principal_chain`` scored each one in
    O(n) -- O(n^3) per molecule (C145: 21025 paths, 25 s). This walks the same DFS
    but records a path only when both of its ends are closed, and skips the DFS
    from a start atom that provably closes no path (below), so an unbranched chain
    costs O(n).

    Traversal order is the same as ``find_all_carbon_chains``: start atoms in atom
    order, neighbours in RDKit neighbour order, a path recorded when the DFS reaches
    its last atom. Only which paths are recorded differs.
    """
    exclude = exclude_atoms or set()
    atoms = list(atoms_of(mol))

    def _eligible(atom) -> bool:
        # The same test find_all_carbon_chains' dfs applies on entry.
        return atom.GetSymbol() == 'C' and atom.GetIdx() not in exclude

    adj: Dict[int, List[int]] = {}
    for atom in atoms:
        if _eligible(atom):
            adj[atom.GetIdx()] = [
                nbr.GetIdx() for nbr in atom.GetNeighbors() if _eligible(nbr)
            ]

    ring_info = mol.GetRingInfo()

    def _can_start_closed(idx: int) -> bool:
        # A path starting at s is closed at s only if every eligible neighbour of s
        # lies on the path. With >= 2 of them the path leaves s through one and
        # reaches another later, so s lies on a cycle and hence in a ring. An atom
        # in no ring with >= 2 eligible neighbours therefore starts no closed path.
        if len(adj[idx]) <= 1:
            return True
        try:
            return ring_info.NumAtomRings(idx) > 0
        except Exception:  # ring info not initialised: do not prune
            return True

    chains: List[List[int]] = []
    for atom in atoms:
        start = atom.GetIdx()
        if start not in adj or not _can_start_closed(start):
            continue
        start_nbrs = adj[start]
        visited: Set[int] = {start}
        path: List[int] = [start]
        stack = [iter(start_nbrs)]
        if not start_nbrs:
            chains.append(path.copy())
        while stack:
            nxt = None
            for nbr in stack[-1]:
                if nbr not in visited:
                    nxt = nbr
                    break
            if nxt is None:
                stack.pop()
                visited.discard(path.pop())
                continue
            visited.add(nxt)
            path.append(nxt)
            nxt_nbrs = adj[nxt]
            if (all(n in visited for n in nxt_nbrs)
                    and all(n in visited for n in start_nbrs)):
                chains.append(path.copy())
            stack.append(iter(nxt_nbrs))

    return chains


def find_longest_carbon_chain(
    mol,
    exclude_atoms: Optional[Set[int]] = None
) -> List[int]:
    """
    Find the longest continuous carbon chain.

    Args:
        mol: RDKit Mol object
        exclude_atoms: Optional set of atom indices to skip (e.g., ring atoms)

    Returns:
        List of atom indices forming the longest chain
    """
    exclude = exclude_atoms or set()

    def dfs(atom_idx: int, visited: Set[int], path: List[int], results: List[List[int]]):
        # Skip excluded atoms
        if atom_idx in exclude:
            return

        atom = mol.GetAtomWithIdx(atom_idx)

        if atom.GetSymbol() != 'C':
            return

        visited.add(atom_idx)
        path.append(atom_idx)

        # Update longest if this path is longer
        if len(path) > len(results[0]):
            results[0] = path.copy()

        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited:
                dfs(nbr_idx, visited, path, results)

        path.pop()
        visited.discard(atom_idx)

    results = [[]]
    for atom in atoms_of(mol):
        if atom.GetSymbol() == 'C' and atom.GetIdx() not in exclude:
            dfs(atom.GetIdx(), set(), [], results)

    return results[0]


# ============================================================================
# Skeletal Chain Finding (C, O, N, S) — IUPAC /
# ============================================================================

# Per IUPAC, skeletal replacement nomenclature considers
# O, N, S as part of the principal chain backbone.
_SKELETAL_ATOMS = {6, 7, 8, 16}  # C, N, O, S


def find_all_skeletal_chains(
    mol,
    min_length: int = 1,
    exclude_atoms: Optional[Set[int]] = None,
    max_chains: int = 10000
) -> List[List[int]]:
    """Find all skeletal chains following C, O, N, S atoms.

    Per IUPAC and, skeletal replacement nomenclature
    considers O, N, S as part of the principal chain backbone.

    This function is used specifically for parent selection chain-vs-ring
    comparison. It does NOT replace find_all_carbon_chains which is
    used for standard chain-based naming.

    Scope: Chain FINDING only. Oxa/aza/thia prefix generation is
    deferred to a later phase.

    Args:
        mol: RDKit Mol object
        min_length: Minimum chain length to return
        exclude_atoms: Optional set of atom indices to skip (e.g., ring atoms)
        max_chains: Maximum chains to find (prevents combinatorial explosion)

    Returns:
        List of lists, each inner list contains atom indices of a skeletal chain
    """
    chains: List[List[int]] = []
    exclude = exclude_atoms or set()
    chain_limit_hit = False

    def dfs(atom_idx: int, visited: Set[int], path: List[int]):
        nonlocal chain_limit_hit
        if chain_limit_hit:
            return

        if atom_idx in exclude:
            return

        atom = mol.GetAtomWithIdx(atom_idx)

        #: Follow C, N, O, S atoms only
        if atom.GetAtomicNum() not in _SKELETAL_ATOMS:
            return

        visited.add(atom_idx)
        path.append(atom_idx)

        if len(path) >= min_length:
            chains.append(path.copy())
            if len(chains) >= max_chains:
                chain_limit_hit = True
                path.pop()
                visited.discard(atom_idx)
                return

        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited:
                dfs(nbr_idx, visited, path)
                if chain_limit_hit:
                    break

        path.pop()
        visited.discard(atom_idx)

    # Start DFS from each skeletal atom
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in _SKELETAL_ATOMS and atom.GetIdx() not in exclude:
            dfs(atom.GetIdx(), set(), [])
            if chain_limit_hit:
                break

    return chains


def find_longest_skeletal_chain(
    mol,
    exclude_atoms: Optional[Set[int]] = None
) -> List[int]:
    """Find the longest continuous skeletal chain (C, O, N, S).

    Convenience function parallel to find_longest_carbon_chain.
    Used for parent selection comparison when heteroatom chains
    may be longer than carbon-only chains.

    Args:
        mol: RDKit Mol object
        exclude_atoms: Optional set of atom indices to skip

    Returns:
        List of atom indices forming the longest skeletal chain
    """
    exclude = exclude_atoms or set()

    def dfs(atom_idx: int, visited: Set[int], path: List[int], results: List[List[int]]):
        if atom_idx in exclude:
            return

        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetAtomicNum() not in _SKELETAL_ATOMS:
            return

        visited.add(atom_idx)
        path.append(atom_idx)

        if len(path) > len(results[0]):
            results[0] = path.copy()

        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited:
                dfs(nbr_idx, visited, path, results)

        path.pop()
        visited.discard(atom_idx)

    results: List[List[int]] = [[]]
    for atom in atoms_of(mol):
        if atom.GetAtomicNum() in _SKELETAL_ATOMS and atom.GetIdx() not in exclude:
            dfs(atom.GetIdx(), set(), [], results)

    return results[0]


def _get_non_principal_terminal_carbons(
    mol,
    functional_groups: Dict[str, List[tuple]],
    principal_group: Optional[str] = None,
) -> Set[int]:
    """Identify terminal FG carbons of non-principal groups whose prefix includes C.

    Only FGs where the non-principal prefix represents the entire terminal group
    including its carbon are excluded from chain enumeration:
    - carbamoyl (-C(=O)NH2): prefix includes C
    - carboxy (-COOH): prefix includes C (when non-principal acid)
    - carbonochloridoyl (-C(=O)Cl): prefix includes C
    - cyano (-C#N): the nitrile carbon belongs to the cyano prefix, NOT the parent
      chain, whenever the nitrile is non-principal (a senior group is present).
      Per the cyano carbon is excluded from the parent (e.g.
      N#CCCC(=O)O -> 3-cyanopropanoic acid, not 4-cyanobutanoic acid). When the
      nitrile IS the principal group its carbon stays in the chain via the
      `fg_name == principal_group` skip below (-nitrile suffix counts that C).

    FGs where the prefix represents only the heteroatom attachment are NOT excluded:
    - oxo (=O on chain C): the aldehyde/keto carbon IS a chain member and is
      expressed as 'oxo' in-chain (e.g. O=CCC(=O)O -> 3-oxopropanoic acid, the
      PIN per; it is NOT excised to a 'formyl' prefix on acyclic chains.

    Per IUPAC 2013 (c),,.

    Args:
        mol: RDKit Mol object
        functional_groups: Dict from detect_functional_groups
        principal_group: Name of principal functional group (or None)

    Returns:
        Set of atom indices for non-principal FG terminal carbons
    """
    # NOTE: acid_chloride/bromide/fluoride are deliberately NOT in this set.
    # On a CHAIN parent a non-principal acyl halide keeps its carbon IN the
    # chain, expressed as 'oxo' (=O) + 'halo' (X): Blue Book worked
    # examples — "methyl 4-chloro-4-oxobutanoate" (PIN, line 5108),
    # "3-chloro-3-oxopropanoic acid" (PIN, line 31531) — NOT the longer-prefix
    # "...carbonochloridoyl...". (The 'carbonochloridoyl'/'chlorocarbonyl'
    # prefix IS the PIN only on a RING parent, e.g. "2-carbonochloridoyl-
    # benzoic acid" line 31533, where the carbon cannot be a ring member; that
    # path does not use chain enumeration so it is unaffected.)
    _TERMINAL_C_FGS = {
        'carboxylic_acid': 0,
        # W3-P03-5, BB 30384): primary_amide is DELIBERATELY NOT in
        # this set. On a CHAIN parent a non-principal -CO-NH2 at a chain end keeps
        # its carbon IN the chain, expressed as 'oxo' (=O) + 'amino' (-NH2):
        # "4-amino-4-oxobutanoic acid" (PIN) -- NOT the longer 'carbamoyl' prefix
        # (the general/non-PIN alternative "3-carbamoylpropanoic acid", and the PIN
        # only on a RING parent where the carbon cannot join the ring,
        # "2-carbamoylbenzoic acid" L30377; ring parents do not use chain
        # enumeration so they are unaffected). Mirrors the acid-halide (oxo+halo)
        # and amidine (amino+imino) decisions above/below; the chain-end oxo+amino
        # split is emitted in rules/polyfunctional.py.
        #: a non-principal nitrile is the 'cyano' prefix whose carbon
        # is excluded from the parent chain. SMARTS '[CX2]#[NX1]' -> index 0 = C.
        # Skipped automatically when nitrile IS the principal group (suffix path).
        'nitrile': 0,
        #, BB 34338): amidine is DELIBERATELY NOT in this set.
        # "When the carbon atom of the H2N-C(=NH)- group terminates a chain,
        # -NH2 and =NH are designated amino and imino" — so on a CHAIN parent the
        # amidine carbon stays IN the chain and is expressed via 'amino' + 'imino'
        # prefixes (methyl 4-(dimethylamino)-4-(ethylimino)butanoate), mirroring
        # the acid-halide note above. The 'carbamimidoyl' prefix is the PIN only
        # for RING parents / genuinely off-chain amidine carbons (BB 34332);
        # those never use chain enumeration so they are unaffected.
        # (BB 34498, plan P1AM Task 6): the amidrazone
        # (hydrazonamide) carbon at a chain end stays IN the chain and is
        # expressed via 'amino' + 'hydrazinylidene' prefixes, mirroring the
        # amidine note above. 'carbamohydrazonoyl' remains the PIN
        # prefix only for ring/off-chain amidrazone carbons.
        # ('hydrazidine' stays excluded: its chain-end split form
        # hydrazinyl+hydrazinylidene) is Task 7 scope-checked
        # and currently fails closed.)
        'hydrazidine': 0,
    }

    principal_carbons: Set[int] = set()
    if principal_group and principal_group in functional_groups:
        for match in functional_groups[principal_group]:
            if principal_group in _TERMINAL_C_FGS:
                c_idx = _TERMINAL_C_FGS[principal_group]
                if c_idx < len(match):
                    principal_carbons.add(match[c_idx])

    terminal_carbons: Set[int] = set()
    for fg_name, c_idx in _TERMINAL_C_FGS.items():
        if fg_name == principal_group:
            continue
        if fg_name not in functional_groups:
            continue
        for match in functional_groups[fg_name]:
            if c_idx >= len(match):
                continue
            carbon_atom_idx = match[c_idx]
            if carbon_atom_idx in principal_carbons:
                continue
            atom = mol.GetAtomWithIdx(carbon_atom_idx)
            if atom.GetSymbol() != 'C':
                continue
            carbon_nbr_count = sum(
                1 for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'C'
            )
            if carbon_nbr_count <= 1:
                terminal_carbons.add(carbon_atom_idx)

    return terminal_carbons


def find_principal_chain(
    mol,
    functional_groups: Dict[str, List[tuple]],
    principal_group: Optional[str] = None,
    exclude_atoms: Optional[Set[int]] = None,
    *,
    ring_prefixes: bool = False,
) -> Optional[List[int]]:
    """
    Find the principal chain following IUPAC 2013 rules.

    Selection criteria (in order of priority):
    1. Contains principal characteristic group
    2. Maximum number of principal groups
    3. Maximum chain length (IUPAC 2013: length BEFORE unsaturation!)
    4. Maximum multiple bonds (double + triple)
    5. Maximum double bonds
    6. Lowest locants for principal groups (first point of difference)
    7. Lowest locants for multiple bonds
    8. Maximum substituents
    9. Lowest locants for substituents

    Non-principal suffix-capable FG terminal carbons (e.g., the C in -C(=O)NH2
    when amide is not the principal group) are excluded from chain enumeration
    to prevent chain length inflation (IUPAC.

    Args:
        mol: RDKit Mol object
        functional_groups: Dict from detect_functional_groups
        principal_group: Name of principal functional group (or None)
        exclude_atoms: Optional set of atom indices to skip (e.g., ring atoms)
        ring_prefixes: False (every production caller): an atom of
            ``exclude_atoms`` bonded to a chain atom is neither counted nor
            located nor named as a substituent by criteria 8, 9 and the
            tie-break. True: such an atom (a ring bonded to the chain) is a
            substituent cited as a prefix like any other, as to
             require (the Blue Book,:21698,:21791; the chain
            parent of the example at:21624 counts its ring
            substituent). In this mode the tie-break names each whole branch
            with the side-effect-free ``name_substituent_for_ordering``, and the
            call returns None when two different chains reach that tie-break
            and a branch has no name, or when two different chains still tie
            after it (the comparison cannot be decided). Used by
            ``chain_parent_prefix_seniority`` to check the chain a producer
            was given.

    Returns:
        List of atom indices forming the principal chain, in order (None only
        with ``ring_prefixes=True``, see above)
    """
    np_terminal_carbons = _get_non_principal_terminal_carbons(
        mol, functional_groups, principal_group
    )
    combined_exclude = set(exclude_atoms) if exclude_atoms else set()
    combined_exclude |= np_terminal_carbons

    # Only the paths that cannot be extended at either end can win, so only those
    # are enumerated (find_maximal_carbon_chains; the chosen chain is unchanged).
    # Proof: extend a path P by one eligible carbon at an end to P'. P' is itself
    # enumerated, and chain_score(P') > chain_score(P) in the tuple order: terms 0
    # (contains_fg) and 1 (fg_count) only read set membership of chain atoms, so
    # they cannot drop on a superset, and term 2 (length) grows by one. So an
    # extendable path is never in `top` below, and `top` -- the chains with the best
    # score, in enumeration order, which the tie-breaks read -- is the same list in
    # the same order. Measured equal on the gate, breadth and chain corpora
    # (TRIAGE 'Suite fix -- j3-long-alkanes'). The full enumeration cost O(n^3) for
    # an unbranched C_n chain (C145: 21025 paths scored, 25 s); this costs O(n).
    chains = find_maximal_carbon_chains(
        mol, exclude_atoms=combined_exclude if combined_exclude else None
    )

    if not chains:
        return []

    # Get atoms belonging to principal functional group.
    # Keep both the flat atom set (for contains-check) and the match
    # tuples list (for correct instance counting per (b)).
    fg_atoms = set()
    fg_matches: List[tuple] = []
    # Characteristic heteroatom set for the principal-group class (N for amine,
    # O for alcohol, …). Populated below when the PCG is a heteroatom-suffix
    # class; used by criterion 8 (max prefix substituents) to count the
    # N-substituents on a suffix amine nitrogen — see _count_substituents.
    fg_hetero_atoms: Set[int] = set()
    # (b) counting support: one entry per PCG INSTANCE, each the SET of
    # carbons DIRECTLY bonded to that instance's characteristic heteroatom (its
    # true bearing carbons). fg_count(chain) counts an instance iff the chain
    # contains ANY of its bearing carbons. This is the IUPAC-correct semantics:
    # a suffix group counts for a parent chain when a carbon that directly bears
    # it is on the chain. For a secondary amine/alcohol whose heteroatom bridges
    # TWO carbons (N-CH2-N junction, e.g. NCCNCN), BOTH carbons are bearing
    # carbons, so the instance counts for whichever of the two chains is chosen
    # as parent — instead of being collapsed to ONE arbitrary (order-dependent)
    # carbon by _normalize_pcg_match, which deflated the other chain's count and
    # let a 1-carbon "chain" out-score the genuine ethane-1,2-diamine backbone.
    fg_bearing_carbons: List[Set[int]] = []
    # The non-carbon atoms of the principal group's own matches (a ketone's '=O', an
    # alcohol's 'O', an amide's 'N'): the characteristic group the chain carries as its
    # suffix, which the key of ``ring_prefixes`` mode must not cite as a prefix.
    pcg_own_hetero: Set[int] = set()
    if principal_group and principal_group in functional_groups:
        # DD5: count the principal characteristic group over its
        # WHOLE equal-seniority class (e.g. primary + secondary OH = two hydroxy
        # PCGs), so the chain bearing all of them wins on PCG count (the di-OH
        # chain -> 3-(4-chlorobutyl)pentane-1,4-diol, not the longer 1-OH chain).
        # Non-equalized groups fall back to the single subtype (byte-identical).
        from ..rules.seniority import (
            _CLASS_CHARACTERISTIC_Z,
            _RC4_UNION_CLASSES,
            _SENIORITY_CLASS_MEMBERS,
            _SENIORITY_PARENT,
            _normalize_pcg_match,
        )
        _parent_class = _SENIORITY_PARENT.get(principal_group)
        _members = _SENIORITY_CLASS_MEMBERS.get(_parent_class)
        _het_z = _CLASS_CHARACTERISTIC_Z.get(_parent_class)
        if _het_z is None:
            # a phase: heteroatom-only-suffix acid classes (sulfonic/
            # sulfinic/phosphonic/phosphinic + the sulfonic-family imidic/
            # peroxoic/thioic S variants) are singleton classes -- never a
            # _SENIORITY_PARENT value -- so _parent_class is always None for
            # them and the lookup above always misses. Fall back to a direct
            # principal_group lookup in _HETEROACID_CHARACTERISTIC_Z (module
            # level, above). Harmless for every other FG name: that dict only
            # has keys for the classes named there.
            _het_z = _HETEROACID_CHARACTERISTIC_Z.get(principal_group)
        if _members is None or _parent_class not in _RC4_UNION_CLASSES:
            fg_matches = functional_groups[principal_group]
        else:
            # Union over the equal-seniority class, normalized to the
            # (heteroatom, bearing-carbon) shape so fg_atoms / locant scoring
            # anchor at the (single) bearing carbon — the raw secondary-OH SMARTS
            # includes FLANKING carbons that would otherwise falsely place the
            # group for a longer chain that does not actually carry the OH.
            fg_matches = [
                _normalize_pcg_match(mol, m, _het_z)
                for sub in _members for m in functional_groups.get(sub, [])
            ]
        # R1: a SKELETAL-suffix PCG -- the '-one' family -- puts
        # the characteristic group's OWN atom into the parent hydride, so only
        # that atom may satisfy "this chain bears the PCG". The ketone SMARTS
        # '[#6][CX3](=O)[#6]' carries BOTH FLANKING carbons, and the whole-match
        # semantics below let a chain through a mere NEIGHBOUR of the carbonyl
        # score contains_fg=1 / fg_count=1. Criterion 3 (length) then handed the
        # win to a longer carbonyl-FREE chain, the acyl carbons were dropped as
        # an unnameable substituent (universal_pipeline_unnameable_substituent) and the '=O' was re-expressed on
        # the attachment atom -- a SILENT ATOM DROP:
        # CCCCCCCCC(CCCC)C(C)(CC(C)C)C(=O)C (C21H42O)
        # -> '5-butyl-2,4-dimethyltridecan-4-one' (C19H38O, 2 C GONE)
        #
        # "Acyclic ketones" (the Blue Book Blue Book) -- "(1)
        # substitutively, using the suffix 'one'... Method (1) generates
        # preferred IUPAC names"; its examples `butan-2-one (PIN)`,
        # `heptan-3-one (PIN)` and `5-methylhexan-2-one (PIN)` all number the
        # CARBONYL CARBON as a skeletal atom of the parent chain.
        # (:18875) -- "The senior parent structure has the maximum
        # number of substituents corresponding to the principal characteristic
        # group (suffix)"; (:18873) -- these criteria "must always be
        # applied before those applicable to... chains (see ". A chain
        # without the carbonyl carbon bears ZERO ketones, so it loses at
        # before chain length is ever consulted.
        #
        # This is the same correction ``_pg_is_on_ring`` already applies to
        # RINGS via SKELETAL_SUFFIX_PGS (parent_selection.py:265, "the
        # bonded-to-ring relaxation is invalid and mis-parented every aryl
        # ketone"). The chain selector never received it. Both the primitive
        # and the membership set are reused, not reinvented.
        #
        # Scoped to the SKELETAL_SUFFIX_PGS members that reach the het_z-is-None
        # fallback: alcohol, imine, ketone, selenoketone, selenol, telluroketone,
        # tellurol, thioketone, thiol. For every one of those except the ketone
        # family the only non-anchor match atom is a HETEROATOM, which can never
        # be a member of a carbon chain -- so the restriction is byte-identical
        # there and bites exactly the flanking-carbon case it was derived for.
        # The alcohol/amine classes have het_z set and keep the bearing-carbon
        # branch below (the correct semantics for an exocyclic heteroatom).
        # Function-local import: rules.parent_selection imports perception.chains
        # at module scope, so a top-level import here would be circular.
        from ..rules.parent_selection import (
            SKELETAL_SUFFIX_PGS,
            _pg_attachment_atoms,
        )
        _skeletal_suffix = (
            _het_z is None and principal_group in SKELETAL_SUFFIX_PGS
        )

        def _pcg_anchor_atoms(match) -> Set[int]:
            """Atoms of ``match`` that can make a chain bear this PCG."""
            if not _skeletal_suffix:
                return set(match)
            return set(_pg_attachment_atoms(principal_group, tuple(match)))

        for match in fg_matches:
            fg_atoms.update(_pcg_anchor_atoms(match))
            pcg_own_hetero.update(
                a for a in match if mol.GetAtomWithIdx(a).GetAtomicNum() != 6)
        # Build the bearing-carbon set for each PCG instance (deduped by
        # heteroatom so a group present under >1 SMARTS subtype counts once).
        # Bearing carbons = carbons DIRECTLY bonded to the characteristic
        # heteroatom (never flanking carbons). When the heteroatom cannot be
        # located (het_z is None, e.g. carbonyl/acid classes) the instance keeps
        # the legacy whole-match semantics so their (b) count is unchanged.
        _seen_het: Set[int] = set()
        for match in fg_matches:
            het = None
            if _het_z is not None:
                het = next(
                    (a for a in match
                     if mol.GetAtomWithIdx(a).GetAtomicNum() == _het_z),
                    None,
                )
            if het is None:
                # Legacy fallback: count where any match atom is on the chain --
                # narrowed by R1 to the PCG's own skeletal atom for the
                # SKELETAL_SUFFIX_PGS families (see the note above), so
                # a chain through a flanking carbon no longer counts the group.
                fg_bearing_carbons.append(_pcg_anchor_atoms(match))
                continue
            if het in _seen_het:
                continue
            _seen_het.add(het)
            fg_hetero_atoms.add(het)
            bearing = {
                nb.GetIdx()
                for nb in mol.GetAtomWithIdx(het).GetNeighbors()
                if nb.GetAtomicNum() == 6
            }
            fg_bearing_carbons.append(bearing if bearing else {het})
            # Also register EVERY bearing carbon in fg_atoms so criterion-6
            # (lowest-PCG-locant, _compute_fg_locant_score) and chain ORIENTATION
            # recognise the PCG on whichever chain carries it. _normalize_pcg_match
            # collapsed a bridging secondary amine (N bonded to two candidate
            # chain carbons) to ONE order-dependent carbon, so the OTHER chain saw
            # one fewer on-chain PCG position and lost criterion 6 spelling-
            # dependently. Only carbons DIRECTLY bonded to the heteroatom are
            # added (never flanking carbons), preserving the anti-over-count the
            # original normalization intended.
            fg_atoms.update(bearing)

    def count_bonds_in_chain(chain: List[int]) -> Tuple[int, int]:
        """Count double and triple bonds within the chain."""
        double_bonds = 0
        triple_bonds = 0

        for i in range(len(chain) - 1):
            bond = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
            if bond:
                bond_type = bond.GetBondType()
                if bond_type == Chem.BondType.DOUBLE:
                    double_bonds += 1
                elif bond_type == Chem.BondType.TRIPLE:
                    triple_bonds += 1

        return double_bonds, triple_bonds

    exclude = exclude_atoms or set()
    # Atoms that are never a substituent of the chain for criteria 8 and 9: the
    # excluded atoms, unless ``ring_prefixes`` asks for the count in which
    # a ring bonded to the chain is a prefix too.
    not_prefix = set() if ring_prefixes else exclude
    #: ring_prefixes mode: attachment atoms of branches the tie-break could not name
    unnamed_branches: List[int] = []

    def _compute_fg_locant_score(chain: List[int]) -> tuple:
        """Criterion 6: Lowest locants for principal group."""
        chain_set = set(chain)
        on_chain = fg_atoms & chain_set
        if not on_chain:
            return (0,)
        # Try both orientations, take better one
        fwd = sorted(chain.index(a) for a in on_chain)
        rev = sorted(len(chain) - 1 - chain.index(a) for a in on_chain)
        fwd_score = (1,) + tuple(-p for p in fwd)
        rev_score = (1,) + tuple(-p for p in rev)
        return max(fwd_score, rev_score)

    def _compute_bond_locant_score(chain: List[int]) -> tuple:
        """Criterion 7: Lowest locants for multiple bonds."""
        positions = []
        for i in range(len(chain) - 1):
            bond = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
            if bond:
                bt = bond.GetBondType()
                if bt == Chem.BondType.DOUBLE or bt == Chem.BondType.TRIPLE:
                    positions.append(i)
        if not positions:
            return (0,)
        # Try both orientations
        fwd = sorted(positions)
        rev = sorted(len(chain) - 2 - p for p in positions)
        fwd_score = (1,) + tuple(-p for p in fwd)
        rev_score = (1,) + tuple(-p for p in rev)
        return max(fwd_score, rev_score)

    def _count_substituents(chain: List[int]) -> int:
        """Criterion 8: Maximum number of substituents cited as
        prefixes on the chain.

        Counts every heavy chain-carbon neighbour off the chain (as before) AND,
        additionally, the prefix substituents hanging off a SUFFIX heteroatom (an
        amine N / alcohol O in ``fg_hetero_atoms``) directly bonded to the chain:
        N-methyl, N-(2-aminoethyl), … are genuine prefix substituents. This is
        strictly additive to the previous count (the heteroatom is still counted
        once as before), so it only ever changes the RESULT of a tie between two
        equal-length equal-PCG chains — never a chain that was already uniquely
        best. It breaks the tie for ``CN(C)CCN(C)CCN``: the middle
        ethane-1,2-diamine (N's carry 3 methyls + a 2-aminoethyl -> +4) beats the
        terminal one (1 methyl + a 2-(dimethylamino)ethyl -> +2) deterministically
        and spelling-independently. The old counter saw only the two amine N's
        (2 vs 2) and left the winner to enumeration order.
        """
        chain_set = set(chain)
        count = 0
        for atom_idx in chain:
            atom = mol.GetAtomWithIdx(atom_idx)
            for nbr in atom.GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx not in chain_set and nbr_idx not in not_prefix:
                    if nbr.GetSymbol() != 'H':
                        count += 1
                        if nbr_idx in fg_hetero_atoms:
                            # Suffix heteroatom: also count the prefix
                            # substituents ON it (its heavy neighbours other
                            # than this chain carbon)..
                            for sub in nbr.GetNeighbors():
                                si = sub.GetIdx()
                                if si == atom_idx or si in not_prefix:
                                    continue
                                if sub.GetAtomicNum() > 1:
                                    count += 1
        return count

    def _cascade_reverse(chain: List[int]) -> bool:
        """True if the REVERSED chain gives lower locants by the orientation
        cascade PCG -> multiple bonds -> double bonds -> substituents
         numbering order).

        Used so the substituent-locant criterion (idx 9) is read in the SAME
        orientation the higher-priority criteria fix, instead of independently
        re-minimizing (the old ``max(fwd, rev)`` evaluated a fictional
        orientation and mis-ranked chains tying on the higher criteria).
        """
        cset = set(chain)
        rev = list(reversed(chain))

        def _fg(ch):
            return sorted(i + 1 for i, a in enumerate(ch) if a in fg_atoms)

        def _mb(ch):
            out = []
            for i in range(len(ch) - 1):
                b = mol.GetBondBetweenAtoms(ch[i], ch[i + 1])
                if b and b.GetBondType() in (
                    Chem.BondType.DOUBLE, Chem.BondType.TRIPLE
                ):
                    out.append(i + 1)
            return sorted(out)

        def _db(ch):
            out = []
            for i in range(len(ch) - 1):
                b = mol.GetBondBetweenAtoms(ch[i], ch[i + 1])
                if b and b.GetBondType() == Chem.BondType.DOUBLE:
                    out.append(i + 1)
            return sorted(out)

        def _sub(ch):
            out = []
            for i, a in enumerate(ch):
                for nbr in mol.GetAtomWithIdx(a).GetNeighbors():
                    ni = nbr.GetIdx()
                    if ni not in cset and ni not in not_prefix and nbr.GetSymbol() != 'H':
                        out.append(i + 1)
                        break
            return sorted(out)

        for setf in (_fg, _mb, _db, _sub):
            c = compare_locant_sets(setf(chain), setf(rev))
            if c < 0:
                return False
            if c > 0:
                return True
        return False

    def _compute_sub_locant_score(chain: List[int]) -> tuple:
        """Criterion 9 /: lowest substituent locants, read in the
        orientation fixed by the higher-priority criteria (PCG -> multiple bonds ->
        double bonds), NOT independently minimized.

        The old independent ``max(fwd, rev)`` evaluated a fictional orientation and
        mis-ranked chains that tie on the higher criteria — e.g. for
        ``C=CCC(C=C(C)C)C(C)=CC`` (both length-7, both 1,5-diene) it preferred the
        ``{4,6}`` carving over the correct ``{4,5}`` (5-methyl-4-(2-methylprop-1-en-
        1-yl)hepta-1,5-diene). For chains with no PCG/bonds the cascade falls
        through to substituents, so a pure substituted alkane still orients to the
        lowest substituent locants (byte-identical to the old behaviour).
        """
        chain_set = set(chain)
        ch = list(reversed(chain)) if _cascade_reverse(chain) else chain
        positions = []
        for i, atom_idx in enumerate(ch):
            atom = mol.GetAtomWithIdx(atom_idx)
            for nbr in atom.GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx not in chain_set and nbr_idx not in not_prefix and nbr.GetSymbol() != 'H':
                    positions.append(i)
                    # ring_prefixes mode: compares the locant of every
                    # prefix ('2,2,3' is not '2,3'), so a carbon with two
                    # prefixes gives its locant twice.
                    if not ring_prefixes:
                        break
        if not positions:
            return ()
        return tuple(-p for p in sorted(positions))

    def _compute_double_bond_locant_score(chain: List[int]) -> tuple:
        """Criterion 7.5 (h)): Lowest locants for double bonds only.

        When two chains have the same combined multiple-bond locant set,
        the chain with lower double-bond-only locants is preferred.
        This breaks ties between en-yne orientations.
        """
        positions = []
        for i in range(len(chain) - 1):
            bond = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
            if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                positions.append(i)
        if not positions:
            return (0,)
        fwd = sorted(positions)
        rev = sorted(len(chain) - 2 - p for p in positions)
        fwd_score = (1,) + tuple(-p for p in fwd)
        rev_score = (1,) + tuple(-p for p in rev)
        return max(fwd_score, rev_score)

    def chain_score(chain: List[int]) -> tuple:
        """
        Calculate selection score for a chain.
        Returns 10-element tuple for comparison (higher = better).
        Implements all IUPAC 2013 criteria.

        Tuple elements:
          0: contains_fg (bool as int)
          1: fg_count (FG instances, not atoms -- (b))
          2: length (chain length -- (a))
          3: multiple_bonds (double + triple count -- (c))
          4: double_bonds (double bond count -- (d))
          5: fg_locants_score (lowest FG locants -- (f))
          6: bond_locants_score (lowest multiple bond locants -- (g))
          7: double_bond_locants_score (lowest double bond locants -- (h))
          8: sub_count (substituent count -- (i))
          9: sub_locants_score (lowest substituent locants -- (j))
        """
        chain_set = set(chain)

        # Criterion 1: Contains principal group
        contains_fg = 1 if (fg_atoms & chain_set) else 0

        # Criterion 2: Count of principal group INSTANCES in chain.
        # (b): count how many PCG INSTANCES are borne by this chain — an
        # instance counts iff a carbon DIRECTLY bearing its characteristic
        # heteroatom is on the chain (fg_bearing_carbons). This counts instances
        # (e.g. 2 for two COOH) rather than atoms, AND correctly counts a
        # secondary amine/alcohol whose heteroatom bridges two chain-carbon
        # candidates for whichever chain is chosen as the parent (see the
        # NCCNCN ethane-1,2-diamine case at fg_bearing_carbons construction).
        fg_count = sum(
            1 for bearing in fg_bearing_carbons if bearing & chain_set
        )

        # Criterion 3: Chain length (IUPAC 2013 prioritizes length!)
        length = len(chain)

        # Criterion 4 & 5: Count multiple bonds
        double_bonds, triple_bonds = count_bonds_in_chain(chain)
        multiple_bonds = double_bonds + triple_bonds

        # Criterion 6: Lowest locants for principal group
        fg_locants_score = _compute_fg_locant_score(chain)

        # Criterion 7: Lowest locants for multiple bonds (combined)
        bond_locants_score = _compute_bond_locant_score(chain)

        # Criterion 7.5 (h)): Lowest locants for double bonds only.
        # Breaks ties when combined bond locants are equal but double bond
        # positions differ (e.g., en-yne orientation).
        double_bond_locants_score = _compute_double_bond_locant_score(chain)

        # Criterion 8: Maximum substituents
        sub_count = _count_substituents(chain)

        # Criterion 9: Lowest locants for substituents
        sub_locants_score = _compute_sub_locant_score(chain)

        return (contains_fg, fg_count, length, multiple_bonds, double_bonds,
                fg_locants_score, bond_locants_score, double_bond_locants_score,
                sub_count, sub_locants_score)

    def _p45_alpha_key(chain: List[int]) -> tuple:
        """DD5 / /: deterministic candidate
        comparator for chains that TIE on every chain_score term.

        Replaces the arbitrary ``max`` fall-through (which returned whichever
        chain ``find_all_carbon_chains`` happened to enumerate first). The key is
        ``(full_substituent_locant_set, locants_in_alphanumerical_citation_order)``,
        both read in the chain's PCG-/substituent-lowest orientation, so the chain
        whose prefixes get the lowest locants in order of citation wins (lower key
        preferred): ``OCC(CCBr)CCCl`` -> ``2-(2-bromoethyl)-4-chlorobutan-1-ol``
        (not ``4-bromo-2-(2-chloroethyl)…``). Substituent names are resolved by the
        shared namer; the multiplier-free base name is the alpha sort key.
        """
        from ..assembly.naming_utils import alpha_sort_key
        from ..assembly.substituent_enumerator import (
            name_substituent,
            name_substituent_for_ordering,
        )

        #: orient via the SAME full cascade `_compute_sub_locant_score` uses
        # (PCG -> multiple-bonds -> double-bonds -> substituents), not just the PCG,
        # so the citation-order tie-break is read in one consistent orientation; and
        # use the chain-enumeration boundary `combined_exclude` (chain set + the
        # non-principal FG terminal carbons removed from enumeration) so the named
        # substituent fragment matches what the real enumerator sees.
        oriented = list(reversed(chain)) if _cascade_reverse(chain) else chain
        cset = set(oriented)
        # ring_prefixes mode: each branch is named whole (a ring bonded to the
        # chain, or a branch that reaches one, is one prefix: '(3-nitrophenyl)methyl',
        # '3-nitrophenyl'), so only the chain bounds it.
        bfs_boundary = cset if ring_prefixes else cset | combined_exclude
        pos = {a: i + 1 for i, a in enumerate(oriented)}
        entries = []  # (alpha_key, locant)
        sub_locants = []
        raw_names = []
        for chain_atom in oriented:
            for nbr in mol.GetAtomWithIdx(chain_atom).GetNeighbors():
                ni = nbr.GetIdx()
                if ni in cset or ni in fg_atoms:
                    continue
                if ring_prefixes and ni in pcg_own_hetero:
                    # (the Blue Book) orders "substituents cited as
                    # prefixes"; the principal group is the suffix ('-one', '-ol',
                    # '-amide'), never an 'oxo' / 'hydroxy' / 'amino' prefix
                    continue
                if not ring_prefixes and ni in combined_exclude:
                    continue
                if nbr.GetSymbol() == 'H':
                    continue
                frag = _bfs_substituent(mol, ni, bfs_boundary)
                if ring_prefixes:
                    try:
                        nm = name_substituent_for_ordering(mol, frag, ni)
                    except Exception:
                        nm = None
                    if not nm:
                        unnamed_branches.append(ni)
                else:
                    try:
                        nm = name_substituent(mol, frag, ni)
                    except Exception:
                        nm = "zzz"
                entries.append((alpha_sort_key(nm or "zzz"), pos[chain_atom]))
                sub_locants.append(pos[chain_atom])
                raw_names.append(nm or "")
        entries.sort()
        #: when every structural criterion ties, the PIN is the name
        # that comes first in alphanumerical order. The locant tuple alone
        # cannot express that -- two candidate chains through the SAME molecule
        # can carry the same locant set and the same locants-in-citation-order
        # and still give different names, because the substituent NAMES differ.
        #
        # Gold row W2E-P0CF-01 is exactly that: heptanoic acid whose C4 branch
        # can be read as `(1,2-difluoropropyl)` (chain through the nitro side)
        # or as `(1,2-dinitropropyl)` (chain through the fluoro side). Both give
        # locants 4,5,6 in citation order, so the old key tied and the winner
        # was whichever chain `find_all_carbon_chains` enumerated first. The
        # names decide it: 'difluoropropyl' < 'dinitropropyl' at the third
        # letter, so the PIN is `4-(1,2-difluoropropyl)-5,6-dinitroheptanoic
        # acid`. Appending the citation SEQUENCE makes that comparison explicit
        # instead of accidental.
        if ring_prefixes:
            # (the Blue Book): "Alphabetic letters are considered first
            # in the order that they appear in the name; all Roman letters are
            # considered before any italic letters". The prefixes as cited: equal
            # names multiplied ('dibromo', 'bis(...)' -- 'bromo' is earlier than
            # 'dibromo',:22245), in alphanumerical order; their Roman
            # letters in that order. A tie here between different chains is left
            # undecided by the caller.
            return (sorted(sub_locants),
                    tuple(loc for _a, loc in entries),
                    _cited_prefix_letters(raw_names),
                    _cited_prefix_italics(raw_names))
        return (sorted(sub_locants),
                tuple(loc for _a, loc in entries),
                tuple(a for a, _loc in entries))

    def _lambda_direct_key(chain: List[int]) -> tuple:
        """ (BB 22172-22182): among chains tying on every term,
        the PIN parent bears the substituent group of the HIGHEST bonding number
        directly connected to it (λ5 > λ3). Score = the multiset of nonstandard
        (λ) bonding numbers of the DIRECTLY-attached substituent atoms, sorted
        descending; the chain with the lexicographically-greatest tuple wins.

        For ``OC(=O)C(CP)C[PH4]`` the chain through the -CH2-PH4 arm makes λ5-P a
        direct substituent (key ``(5,)``); the chain through the -CH2-PH2 arm has
        no directly-attached λ atom (key ````) — so the former is the parent,
        giving ``3-(λ5-phosphanyl)-2-(phosphanylmethyl)propanoic acid`` (PIN), not
        the ``[not] 3-phosphanyl-2-(λ5-phosphanylmethyl)…`` alternative.
        """
        cset = set(chain)
        lams = []
        for atom_idx in chain:
            for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                ni = nbr.GetIdx()
                if ni in cset or ni in combined_exclude or ni in fg_atoms:
                    continue
                if nbr.GetSymbol() == 'H':
                    continue
                lam = nonstandard_bonding_number(mol, ni)
                if lam is not None:
                    lams.append(lam)
        return tuple(sorted(lams, reverse=True))

    # reverse-pair memo (PURE SPEEDUP — provably output-identical).
    # `chain_score` is orientation-invariant: criteria 1-2 read set membership,
    # 3-5 count length/bonds (both orientation-free), 6-8 take a fwd/rev `max`,
    # and criterion 9 (`_compute_sub_locant_score`) returns the same tuple for a
    # chain and its reverse — it only differs between the two orientations when
    # they do NOT tie the cascade, and in that case both orientations pick
    # the SAME canonical orientation; when they DO tie, the substituent-position
    # pattern is palindromic so `sorted(positions)` is identical either way.
    # `find_all_carbon_chains` emits every chain in BOTH orientations (~47% of
    # the list are exact reverses — measured 210/441, 462/981), so scoring the
    # reverse-canonical form once and reusing it for the mirror chain returns the
    # IDENTICAL tuple with ~half the work. The cache is LOCAL to this
    # find_principal_chain call (a new call = a new closure = a new dict), so it
    # can never serve a stale score for a different molecule / PCG / exclude set.
    _score_cache: Dict[tuple, tuple] = {}
    _P2_VERIFY = _os.environ.get("ORTHONYM_P2_VERIFY") == "1"

    def _chain_score_cached(chain: List[int]) -> tuple:
        t = tuple(chain)
        rt = t[::-1]
        key = t if t <= rt else rt
        cached = _score_cache.get(key)
        if cached is None:
            cached = chain_score(chain)
            _score_cache[key] = cached
        elif _P2_VERIFY:
            # Debug self-check: the memo must return exactly what the un-memoized
            # code would. Off by default (zero cost); flip ORTHONYM_P2_VERIFY=1
            # to assert orientation-invariance on real data across a whole run.
            fresh = chain_score(chain)
            assert fresh == cached, (chain, fresh, cached)
        return cached

    # Find chain with highest score; break exact ties deterministically.
    scored = [(_chain_score_cached(c), c) for c in chains]
    best_score = max(s for s, _ in scored)
    top = [c for s, c in scored if s == best_score]
    if len(top) == 1:
        best_chain = top[0]
    else:
        # nonstandard-bonding-number criterion runs BEFORE the
        # alphanumerical comparator (all terms already tied within `top`).
        best_lambda = max(_lambda_direct_key(c) for c in top)
        top = [c for c in top if _lambda_direct_key(c) == best_lambda]
        if not ring_prefixes:
            best_chain = top[0] if len(top) == 1 else min(top, key=_p45_alpha_key)
        elif len(top) == 1 or len({chain_symmetry_key(mol, c) for c in top}) == 1:
            # one chain, or chains that a symmetry of the molecule maps onto each other
            # (a chain and its reverse, the two methyl ends of an isopropyl group): the
            # caller reads only the length and the symmetry key of the chain, which are
            # the same for each of them, so no prefix is named
            best_chain = top[0]
        else:
            # ring_prefixes mode: the tie is decided only between different chains
            # (a chain and its own reverse are one chain, and so are two chains
            # that a symmetry of the molecule maps onto each other: the two methyl
            # ends of an isopropyl group). It is undecided (None) when a branch
            # without a name took part, or when two different chains keep the same
            # key: the key compares prefix names without their locants, so the
            # full-name comparison of (the Blue Book) that would
            # decide between them is not made here.
            keyed = [(_p45_alpha_key(c), c) for c in top]
            best_key = min(k for k, _c in keyed)
            best = [c for k, c in keyed if k == best_key]
            if (unnamed_branches
                    and len({chain_symmetry_key(mol, c) for c in top}) > 1):
                return None
            if len({chain_symmetry_key(mol, c) for c in best}) > 1:
                return None
            best_chain = best[0]

    # Determine numbering direction (lowest locants for principal group)
    if principal_group and fg_atoms:
        best_chain = _orient_chain_for_lowest_locants(best_chain, fg_atoms)

    return best_chain


#: attribute that memoises ``chain_parent_prefix_seniority`` on one features object
_PREFIX_SENIORITY_ATTR = "_p45_ring_prefix_seniority"
#: attribute that memoises ``principal_chain_with_ring_prefixes`` on one features object
_SENIOR_CHAIN_ATTR = "_p45_ring_prefix_chain"


def _cited_prefixes(names):
    """The prefixes of a chain as a name cites them: equal names multiplied
    (``assembly.naming_utils.get_multiplier_prefix``: ``di`` for a simple prefix, ``bis``
    for a substituted one), the groups in alphanumerical order (``alpha_sort_key``)."""
    from ..assembly.naming_utils import alpha_sort_key, get_multiplier_prefix
    counts: Dict[str, int] = {}
    for nm in names:
        counts[nm] = counts.get(nm, 0) + 1
    cited = []
    for nm, n in counts.items():
        mult = "" if n == 1 else get_multiplier_prefix(n, nm)
        cited.append((alpha_sort_key(nm or "zzz"), mult + nm))
    cited.sort()
    return [text for _k, text in cited]


def _cited_prefix_letters(names) -> str:
    """The Roman letters of a chain's prefixes as a name cites them, in citation order
    , the Blue Book; 'bromo' is earlier than 'dibromo',:22245). The letters
    are those of every other key (``assembly.naming_utils.prefix_roman_and_italic``):
    locants, letter locants, fusion letters, Greek letters and descriptors are not part
    of them."""
    from ..assembly.naming_utils import prefix_roman_and_italic
    return "".join(prefix_roman_and_italic(text)[0] for text in _cited_prefixes(names))


def _cited_prefix_italics(names) -> tuple:
    """The italic prefixes ('sec', 'tert') of a chain's cited prefixes, in citation order:
    considered only when the Roman letters tie, absence first, the Blue Book;
    '3-(butan-2-yl)-5-tert-butyl...':3513)."""
    from ..assembly.naming_utils import prefix_roman_and_italic
    return tuple(i for text in _cited_prefixes(names) for i in prefix_roman_and_italic(text)[1])


def chain_symmetry_key(mol, chain) -> tuple:
    """The chain as its sequence of atom symmetry classes (constitution only, no
    tie-breaking), read in the lower of its two directions: two chains with the same
    key name the molecule alike (the two methyl ends of an isopropyl group; ASSUMED
    for a sequence of equal classes that no symmetry of the whole molecule maps
    onto each other, a case that leaves a name below the PIN at worst)."""
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=False))
    seq = tuple(ranks[int(a)] for a in chain)
    return min(seq, seq[::-1])


def chain_parent_prefix_seniority(features) -> Optional[str]:
    """Check a chain parent against to where a ring is bonded
    to the chain or to another chain of the same rank.

    ``find_principal_chain`` chooses the chain of a cyclic molecule with the ring
    atoms excluded, and its criteria 8 and 9 and its tie-break then skip
    every ring atom: a ring bonded to a chain atom is never a prefix there, and a
    branch that reaches a ring is named only up to the ring ('methyl' for
    '(3-nitrophenyl)methyl'). The book counts, locates and orders a ring prefix
    like any other:, "the maximum number of substituents cited as
    prefixes" (the Blue Book; the chain parent of example (4),:21624,
    counts its ring substituent), (:21698), and (:21791), "the
    lower locant or set of locants for substituents cited as prefixes... in
    their order of citation in the name" ('3-bromo-2-(2-bromo-1-hydroxyethyl)-4-
    hydroxybutanoic acid (PIN)',:22108). So for
    ``N#CC(CO)Cc1cccc([N+](=O)[O-])c1`` the selector keeps the CH2OH chain,
    '3-hydroxy-2-[(3-nitrophenyl)methyl]propanenitrile' (prefix locants 3,2 in
    citation order), while the PIN is '2-(hydroxymethyl)-3-(3-nitrophenyl)
    propanenitrile' (2,3); and for ``OC(=O)C(C)Cc1ccccc1`` it gives
    '2-benzylpropanoic acid' (one prefix) for the PIN '2-methyl-3-
    phenylpropanoic acid' (two).

    The same selector, asked to count ring prefixes (``ring_prefixes=True``),
    answers which chain makes senior. Returns ``'senior_alternative'``
    when that is another chain of the same length (a name built on the
    features' chain is not the PIN), ``'undecided'`` when the comparison cannot
    be made (a branch without a name in a tie, two different chains that still
    tie, or an error), and None when the
    chain stands or the check does not apply (no ring, a ring parent, or a chain
    this selector did not choose: one with a heteroatom, or of another length).
    The answer is kept on ``features``, keyed by its chain and principal group.
    """
    key = (bool(getattr(features, 'chain_is_parent', False)),
           tuple(getattr(features, 'principal_chain', None) or ()),
           getattr(features, 'principal_group', None))
    cached = getattr(features, _PREFIX_SENIORITY_ATTR, None)
    if cached is not None and cached[0] == key:
        return cached[1]
    status = _chain_parent_prefix_seniority(features)
    try:
        setattr(features, _PREFIX_SENIORITY_ATTR, (key, status))
    except AttributeError:
        pass
    return status


def principal_chain_with_ring_prefixes(features) -> Optional[List[int]]:
    """The principal chain of a cyclic molecule as chooses it: the chain
    ``find_principal_chain`` gives with the ring atoms excluded from the chain and
    counted, located and ordered as prefixes (``ring_prefixes=True``).

     (the Blue Book) 'The preferred IUPAC name is based on the senior
    parent structure that has the maximum number of substituents cited as prefixes'
    (the hydro prefixes excepted, as the rule says); its example (4),:21624,
    counts the ring substituent of the parent chain; (:21698),
    (:21791). With the ring atoms left out of the count, two chains of the same
    length bearing one prefix each tie and the input's atom order decides:
    ``COC(=O)C(C)Cc1ccccc1`` (methyl 2-methyl-3-phenylpropanoate, the PIN) gave
    'methyl 2-benzylpropanoate' for some spellings.

    None when the comparison cannot be made (a branch without a name in a tie, two
    different chains that still tie, no chain, or an error); the caller then keeps
    the chain of the call without ring prefixes. The answer is kept on ``features``,
    keyed by the principal group and the ring atoms, so that the check of a chain
    parent (:func:`chain_parent_prefix_seniority`) does not make it twice."""
    mol = getattr(features, 'mol', None)
    ring_systems = getattr(features, 'ring_systems', None) or ()
    if mol is None or not ring_systems:
        return None
    principal_group = getattr(features, 'principal_group', None)
    ring_atoms: Set[int] = set()
    for rs in ring_systems:
        ring_atoms.update(int(a) for a in rs)
    key = (principal_group, tuple(sorted(ring_atoms)))
    cached = getattr(features, _SENIOR_CHAIN_ATTR, None)
    if cached is not None and cached[0] == key:
        return cached[1]
    try:
        from ..metrics.provenance import isolated_provenance
        with isolated_provenance():
            senior = find_principal_chain(
                mol,
                getattr(features, 'functional_groups', None) or {},
                principal_group,
                exclude_atoms=ring_atoms,
                ring_prefixes=True,
            )
    except Exception:                            # fail closed: undecided
        senior = None
    senior = list(senior) if senior else None
    try:
        setattr(features, _SENIOR_CHAIN_ATTR, (key, senior))
    except AttributeError:
        pass
    return senior


#: Principal groups whose producer builds the name on ``features.principal_chain``, the chain
#: it is given, and for which:func:`p45_principal_chain` hands parent selection the chain
#: makes senior. Each group is here because 20 random SMILES orders of a molecule with
#: a tie between two chains of one length (``CC(Cc1ccccc1)X`` with X the group, a ring on one
#: of the chains) gave the PIN spelling, labelled pin_verified, for every order, where the
#: chain without ring prefixes gave 'benzyl' spellings for some orders
#: (``tests/unit/namer/test_leads_l1_n8b_chain_prefixes.py``). A group that is not here keeps
#: the chain of the call without ring prefixes, as before; among them are the producers that
#: choose the chain themselves: the ester (``rules/esters.py:_find_acid_principal_chain``, a
#: breadth-first walk whose first longest path wins, so a tie is decided by the input's atom
#: order: 'methyl 2-benzylpropanoate' in 7 of 20 orders of 'COC(=O)C(C)Cc1ccccc1'; given the
#: senior chain the check of:func:`chain_parent_prefix_seniority` passes and the name is
#: labelled pin_verified), the thio-, seleno- and telluroester handler, the oxime handler (the
#: same, for 'CC(Cc1ccccc1)C=NO'), the imidate and anhydride producers. A group joins the set
#: when its producer takes the chain it is given.
GIVEN_CHAIN_GROUPS = frozenset({
    'carboxylic_acid', 'acid_chloride', 'acid_bromide',
    'primary_amide', 'secondary_amide', 'tertiary_amide', 'imide', 'hydrazide', 'amidine',
    'nitrile', 'aldehyde', 'ketone', 'imine',
    'primary_alcohol', 'secondary_alcohol', 'tertiary_alcohol', 'thiol',
    'primary_amine', 'secondary_amine', 'tertiary_amine',
    'sulfonic_acid', 'primary_sulfonamide', 'thioamide', 'thioic_S_acid', 'peroxy_acid',
})


def p45_principal_chain(features, default_chain: Optional[List[int]]) -> Optional[List[int]]:
    """The chain parent selection is given for a cyclic molecule: ``default_chain``
    (``find_principal_chain`` with the ring atoms excluded and not counted) replaced by the
    chain makes senior (:func:`principal_chain_with_ring_prefixes`) when that is a
    chain of the same length and the group's producer builds on the chain it is given
    (:data:`GIVEN_CHAIN_GROUPS`). ``default_chain`` stands where the comparison is
    undecided too. The caller decides whether the molecule is the one the call names
    (``namer.Orthonym._classify``): a fragment named inside another name keeps the chain
    of the call without ring prefixes, because the string of a part built on another chain
    than the whole's would be recorded as a non-PIN part of the name that contains it
    (measured: '2-hydroxyethyl 2-methyl-3-phenylpropanoate', whose acid is a nested name,
    labelled systematic_verified for the 7 of 10 orders in which it was pin_verified)."""
    if not default_chain:
        return default_chain
    if getattr(features, 'principal_group', None) not in GIVEN_CHAIN_GROUPS:
        return default_chain
    senior = principal_chain_with_ring_prefixes(features)
    if not senior or len(senior) != len(default_chain):
        return default_chain
    #: the chain carries the principal characteristic group before is reached.
    # The selector counts an instance for any chain that holds a carbon bearing it, so for
    # 'CC(=O)N[C@@H](Cc1ccccc1)c1cc(OC)cc(=O)o1' the CH-CH2 chain of the nitrogen's other
    # substituent ties with the acetyl chain and wins on its prefixes: a chain that does not
    # hold the atoms of the group the default chain holds is not taken.
    group_atoms = {int(a) for match in (getattr(features, 'principal_group_atoms', None) or ())
                   for a in match}
    if ({int(a) for a in senior} & group_atoms) != ({int(a) for a in default_chain} & group_atoms):
        return default_chain
    return senior


def _chain_parent_prefix_seniority(features) -> Optional[str]:
    if not (getattr(features, 'chain_is_parent', False)
            and getattr(features, 'is_cyclic', False)):
        return None
    chain = getattr(features, 'principal_chain', None)
    ring_systems = getattr(features, 'ring_systems', None) or ()
    mol = getattr(features, 'mol', None)
    if not chain or not ring_systems or mol is None:
        return None
    try:
        if any(mol.GetAtomWithIdx(int(a)).GetAtomicNum() != 6 for a in chain):
            return None
    except Exception:                            # fail closed
        return 'undecided'
    senior = principal_chain_with_ring_prefixes(features)
    if senior is None:
        return 'undecided'
    try:
        if len(senior) != len(chain):
            return None
        if chain_symmetry_key(mol, senior) != chain_symmetry_key(mol, chain):
            return 'senior_alternative'
    except Exception:                            # fail closed
        return 'undecided'
    return None


def _orient_chain_for_lowest_locants(chain: List[int], priority_atoms: Set[int]) -> List[int]:
    """
    Orient chain so priority atoms have lowest locants.
    
    Uses first-point-of-difference rule.
    """
    forward_locants = [
        i + 1 for i, idx in enumerate(chain)
        if idx in priority_atoms
    ]
    reverse_locants = [
        len(chain) - i for i, idx in enumerate(chain)
        if idx in priority_atoms
    ]

    # Compare using first-point-of-difference
    if _compare_locants(reverse_locants, forward_locants):
        return list(reversed(chain))
    return chain


def _compare_locants(set_a: List[int], set_b: List[int]) -> bool:
    """
    Compare two locant sets using first-point-of-difference rule.
    
    Returns True if set_a is preferred (lower).
    """
    a_sorted = sorted(set_a)
    b_sorted = sorted(set_b)

    for a, b in zip(a_sorted, b_sorted):
        if a < b:
            return True
        if a > b:
            return False

    # If all compared elements are equal, prefer shorter or equal set
    # (deterministic tiebreaker for symmetric molecules)
    return len(a_sorted) <= len(b_sorted)


def get_substituents(mol, main_chain: List[int]) -> Dict[int, List[List[int]]]:
    """
    Find substituents attached to the main chain.
    
    Args:
        mol: RDKit Mol object
        main_chain: List of atom indices in main chain (ordered)
        
    Returns:
        Dict mapping chain position (1-indexed) to list of substituent atom lists.
        Each substituent is represented as a list of its atom indices.
    """
    chain_set = set(main_chain)
    substituents = {}

    for position, chain_idx in enumerate(main_chain, 1):
        chain_atom = mol.GetAtomWithIdx(chain_idx)
        position_subs = []

        for neighbor in chain_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()

            # Skip atoms that are part of the main chain
            if nbr_idx in chain_set:
                continue

            # BFS to find full substituent
            sub_atoms = _bfs_substituent(mol, nbr_idx, chain_set)
            position_subs.append(sub_atoms)

        if position_subs:
            substituents[position] = position_subs

    return substituents


def _bfs_substituent(mol, start_idx: int, exclude_set: Set[int]) -> List[int]:
    """
    Find all atoms in a substituent using BFS.
    
    Args:
        mol: RDKit Mol object
        start_idx: Starting atom index (first atom of substituent)
        exclude_set: Set of atom indices to exclude (main chain atoms)
        
    Returns:
        List of atom indices in the substituent
    """
    visited = {start_idx}
    queue = deque([start_idx])

    while queue:
        current = queue.popleft()
        atom = mol.GetAtomWithIdx(current)

        for neighbor in atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx not in visited and nbr_idx not in exclude_set:
                visited.add(nbr_idx)
                queue.append(nbr_idx)

    return list(visited)


def get_chain_atoms_with_locants(chain: List[int]) -> Dict[int, int]:
    """
    Create mapping from atom index to locant number.

    Args:
        chain: Ordered list of atom indices

    Returns:
        Dict mapping atom_idx -> locant (1-indexed)
    """
    return {atom_idx: locant for locant, atom_idx in enumerate(chain, 1)}


def is_ring_substituent(mol, sub_atoms: List[int], parent_atoms: Set[int]) -> bool:
    """
    Check if substituent atoms form a complete ring.

    A substituent is considered a ring substituent if all atoms of at least
    one ring in the molecule are contained within the substituent atoms
    (excluding the parent structure atoms).

    Args:
        mol: RDKit Mol object
        sub_atoms: Atom indices of the substituent
        parent_atoms: Atoms of the parent structure (to exclude from consideration)

    Returns:
        True if the substituent contains a complete ring, False otherwise

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccccc1C') # toluene
        >>> # Phenyl atoms: 0-5, Methyl: 6
        >>> is_ring_substituent(mol, [0, 1, 2, 3, 4, 5], {6})
        True
        >>> mol2 = Chem.MolFromSmiles('CCCCC') # pentane
        >>> is_ring_substituent(mol2, [0, 1, 2], set)
        False
    """
    if not sub_atoms:
        return False

    sub_set = set(sub_atoms)
    ri = mol.GetRingInfo()

    # Check if any ring in the molecule is entirely within the substituent atoms
    for ring in ri.AtomRings():
        ring_set = set(ring)
        # Ring must be entirely within sub_atoms (not overlapping with parent)
        if ring_set.issubset(sub_set) and not ring_set.intersection(parent_atoms):
            return True

    return False


def classify_substituent(mol, sub_atoms: List[int], parent_atoms: Set[int]) -> Dict:
    """
    Classify a substituent as ring or alkyl chain.

    This function determines whether a substituent is a ring system (and if so,
    what kind) or an alkyl chain. It's used to correctly name ring substituents
    (phenyl, cyclohexyl, piperidinyl) instead of incorrectly counting carbons
    (hexyl, pentyl).

    Args:
        mol: RDKit Mol object
        sub_atoms: Atom indices of the substituent
        parent_atoms: Atoms of the parent structure (to exclude)

    Returns:
        Dict with:
        - 'type': 'ring' or 'alkyl'
        - 'name': substituent name (e.g., 'phenyl', 'cyclohexyl', 'methyl')
        - 'atoms': list of atom indices
        - 'ring_atoms': tuple of ring atom indices (only if type='ring')

    Examples:
        >>> mol = Chem.MolFromSmiles('c1ccccc1CCC(=O)O') # phenylpropanoic acid
        >>> classify_substituent(mol, [0,1,2,3,4,5], {6,7,8,9,10})
        {'type': 'ring', 'name': 'phenyl', 'atoms': [0,1,2,3,4,5], 'ring_atoms': (0,1,2,3,4,5)}
    """
    from ..rules.ring_substituents import get_ring_substituent_name

    if not sub_atoms:
        return {'type': 'alkyl', 'name': '', 'atoms': []}

    sub_set = set(sub_atoms)
    ri = mol.GetRingInfo()

    # Check if substituent contains a complete ring
    contained_ring = None
    for ring in ri.AtomRings():
        ring_set = set(ring)
        # Ring must be entirely within sub_atoms (not overlapping with parent)
        if ring_set.issubset(sub_set) and not ring_set.intersection(parent_atoms):
            contained_ring = ring
            break

    if contained_ring:
        # This is a ring substituent. task 9: name the WHOLE fragment
        # through the single ring-substituent chokepoint with its attachment
        # atom — the old per-first-SSSR-ring lookup truncated a fused system
        # to its first ring (naphthalenyl -> 'phenyl', a DIFFERENT group) and
        # never carried the free-valence locant.
        attach_idx = next(
            (a for a in sub_atoms
             for nbr in mol.GetAtomWithIdx(a).GetNeighbors()
             if nbr.GetIdx() in parent_atoms),
            sub_atoms[0],
        )
        from ..rules.ring_substituents import name_ring_system_substituent
        ring_name = name_ring_system_substituent(mol, sub_atoms, attach_idx)
        if not ring_name and set(contained_ring) == sub_set:
            # Last resort: legacy single-ring lookup (locant-less). Only
            # sound when the fragment IS that single ring.
            ring_name = get_ring_substituent_name(mol, contained_ring)

        return {
            'type': 'ring',
            'name': ring_name or '',
            'atoms': sub_atoms,
            'ring_atoms': contained_ring,
        }

    # A nitrile substituent -C#N (attached through the carbon) is the detachable
    # prefix 'cyano', NOT an alkyl carbon-count. The carbon-count
    # fallback below sees only the ONE carbon of -C#N and silently drops the N,
    # mis-naming it 'methyl' -> a WRONG MOLECULE (e.g. the tricyanomethanide
    # carbanion `N#C[C-](C#N)C#N` emitted '1,1,1-trimethylmethanide'). Detect the
    # exact -C#N shape here and return 'cyano'.
    if len(sub_set) == 2:
        a0 = mol.GetAtomWithIdx(sub_atoms[0])
        a1 = mol.GetAtomWithIdx(sub_atoms[1])
        c_at = a0 if a0.GetSymbol() == 'C' else (a1 if a1.GetSymbol() == 'C' else None)
        n_at = a0 if a0.GetSymbol() == 'N' else (a1 if a1.GetSymbol() == 'N' else None)
        if c_at is not None and n_at is not None:
            cn = mol.GetBondBetweenAtoms(c_at.GetIdx(), n_at.GetIdx())
            # cyano: C#N with a terminal N (degree 1, no H) and the carbon carrying
            # the bond back to the parent (i.e. attachment is THROUGH the carbon).
            c_to_parent = any(nb.GetIdx() in parent_atoms
                              for nb in c_at.GetNeighbors())
            if (cn is not None and cn.GetBondType() == Chem.BondType.TRIPLE
                    and n_at.GetDegree() == 1 and n_at.GetTotalNumHs() == 0
                    and c_to_parent):
                return {'type': 'alkyl', 'name': 'cyano', 'atoms': sub_atoms}

    # Not a ring - count carbons for alkyl naming
    carbon_count = sum(
        1 for idx in sub_atoms
        if mol.GetAtomWithIdx(idx).GetSymbol() == 'C'
    )

    # Get alkyl name
    from ..data.chain_names import get_alkyl_name as _chain_alkyl_name
    try:
        alkyl_name = _chain_alkyl_name(carbon_count)
    except ValueError:
        # Unsupported carbon count, return generic name
        alkyl_name = f"{carbon_count}C-yl" if carbon_count > 0 else ""

    return {
        'type': 'alkyl',
        'name': alkyl_name,
        'atoms': sub_atoms,
    }
