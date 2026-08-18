"""
Chain detection and principal chain selection.

Implements IUPAC 2013 rules for selecting the principal chain.
Key change in IUPAC 2013: Chain length takes priority over unsaturation!
"""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple
from rdkit import Chem

from ..rules.lambda_convention import nonstandard_bonding_number

# v33 Phase 6 (heteroatom-only-suffix acids): characteristic-heteroatom
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
# ("chain contains the principal characteristic group", P-44.1) was then False
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
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C' and atom.GetIdx() not in exclude:
            dfs(atom.GetIdx(), set(), [])

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
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C' and atom.GetIdx() not in exclude:
            dfs(atom.GetIdx(), set(), [], results)

    return results[0]


# ============================================================================
# Skeletal Chain Finding (C, O, N, S) — IUPAC P-44.3 / P-15.4
# ============================================================================

# Per IUPAC P-15.4, skeletal replacement nomenclature considers
# O, N, S as part of the principal chain backbone.
_SKELETAL_ATOMS = {6, 7, 8, 16}  # C, N, O, S


def find_all_skeletal_chains(
    mol,
    min_length: int = 1,
    exclude_atoms: Optional[Set[int]] = None,
    max_chains: int = 10000
) -> List[List[int]]:
    """Find all skeletal chains following C, O, N, S atoms.

    Per IUPAC P-44.3 and P-15.4, skeletal replacement nomenclature
    considers O, N, S as part of the principal chain backbone.

    This function is used specifically for parent selection chain-vs-ring
    comparison. It does NOT replace find_all_carbon_chains() which is
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

        # P-15.4: Follow C, N, O, S atoms only
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

    Convenience function parallel to find_longest_carbon_chain().
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
    for atom in mol.GetAtoms():
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
      Per P-66.5.1.1.4 the cyano carbon is excluded from the parent (e.g.
      N#CCCC(=O)O -> 3-cyanopropanoic acid, not 4-cyanobutanoic acid). When the
      nitrile IS the principal group its carbon stays in the chain via the
      `fg_name == principal_group` skip below (-nitrile suffix counts that C).

    FGs where the prefix represents only the heteroatom attachment are NOT excluded:
    - oxo (=O on chain C): the aldehyde/keto carbon IS a chain member and is
      expressed as 'oxo' in-chain (e.g. O=CCC(=O)O -> 3-oxopropanoic acid, the
      PIN per P-66.6.1); it is NOT excised to a 'formyl' prefix on acyclic chains.

    Per IUPAC 2013 P-66.1(c), P-65.1.1.1, P-66.5.1.1.4.

    Args:
        mol: RDKit Mol object
        functional_groups: Dict from detect_functional_groups()
        principal_group: Name of principal functional group (or None)

    Returns:
        Set of atom indices for non-principal FG terminal carbons
    """
    # NOTE: acid_chloride/bromide/fluoride are deliberately NOT in this set.
    # On a CHAIN parent a non-principal acyl halide keeps its carbon IN the
    # chain, expressed as 'oxo' (=O) + 'halo' (X): Blue Book P-65.5.4 worked
    # examples — "methyl 4-chloro-4-oxobutanoate" (PIN, line 5108),
    # "3-chloro-3-oxopropanoic acid" (PIN, line 31531) — NOT the longer-prefix
    # "...carbonochloridoyl...".  (The 'carbonochloridoyl'/'chlorocarbonyl'
    # prefix IS the PIN only on a RING parent, e.g. "2-carbonochloridoyl-
    # benzoic acid" line 31533, where the carbon cannot be a ring member; that
    # path does not use chain enumeration so it is unaffected.)
    _TERMINAL_C_FGS = {
        'carboxylic_acid': 0,
        # W3-P03-5 (P-65.1.6.1, BB 30384): primary_amide is DELIBERATELY NOT in
        # this set. On a CHAIN parent a non-principal -CO-NH2 at a chain end keeps
        # its carbon IN the chain, expressed as 'oxo' (=O) + 'amino' (-NH2):
        # "4-amino-4-oxobutanoic acid" (PIN) -- NOT the longer 'carbamoyl' prefix
        # (the general/non-PIN alternative "3-carbamoylpropanoic acid", and the PIN
        # only on a RING parent where the carbon cannot join the ring,
        # "2-carbamoylbenzoic acid" L30377; ring parents do not use chain
        # enumeration so they are unaffected). Mirrors the acid-halide (oxo+halo)
        # and amidine (amino+imino) decisions above/below; the chain-end oxo+amino
        # split is emitted in rules/polyfunctional.py.
        # P-66.5.1.1.4: a non-principal nitrile is the 'cyano' prefix whose carbon
        # is excluded from the parent chain. SMARTS '[CX2]#[NX1]' -> index 0 = C.
        # Skipped automatically when nitrile IS the principal group (suffix path).
        'nitrile': 0,
        # AM-4 (P-66.4.1.3.2, BB 34338): amidine is DELIBERATELY NOT in this set.
        # "When the carbon atom of the H2N-C(=NH)- group terminates a chain,
        # -NH2 and =NH are designated amino and imino" — so on a CHAIN parent the
        # amidine carbon stays IN the chain and is expressed via 'amino' + 'imino'
        # prefixes (methyl 4-(dimethylamino)-4-(ethylimino)butanoate), mirroring
        # the acid-halide note above. The 'carbamimidoyl' prefix is the PIN only
        # for RING parents / genuinely off-chain amidine carbons (BB 34332);
        # those never use chain enumeration so they are unaffected.
        # P-66.4.2.3.2 (BB 34498, plan P1AM Task 6): the amidrazone
        # (hydrazonamide) carbon at a chain end stays IN the chain and is
        # expressed via 'amino' + 'hydrazinylidene' prefixes, mirroring the
        # AM-4 amidine note above. 'carbamohydrazonoyl' remains the PIN
        # prefix only for ring/off-chain amidrazone carbons.
        # ('hydrazidine' stays excluded: its chain-end split form
        # (P-66.4.3.4.1 hydrazinyl+hydrazinylidene) is Task 7 scope-checked
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
    exclude_atoms: Optional[Set[int]] = None
) -> List[int]:
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
    to prevent chain length inflation (IUPAC P-44.3).

    Args:
        mol: RDKit Mol object
        functional_groups: Dict from detect_functional_groups()
        principal_group: Name of principal functional group (or None)
        exclude_atoms: Optional set of atom indices to skip (e.g., ring atoms)

    Returns:
        List of atom indices forming the principal chain, in order
    """
    np_terminal_carbons = _get_non_principal_terminal_carbons(
        mol, functional_groups, principal_group
    )
    combined_exclude = set(exclude_atoms) if exclude_atoms else set()
    combined_exclude |= np_terminal_carbons

    chains = find_all_carbon_chains(
        mol, min_length=1,
        exclude_atoms=combined_exclude if combined_exclude else None
    )
    
    if not chains:
        return []
    
    # Get atoms belonging to principal functional group.
    # Keep both the flat atom set (for contains-check) and the match
    # tuples list (for correct instance counting per P-44.1(b)).
    fg_atoms = set()
    fg_matches: List[tuple] = []
    # Characteristic heteroatom set for the principal-group class (N for amine,
    # O for alcohol, …). Populated below when the PCG is a heteroatom-suffix
    # class; used by criterion 8 (max prefix substituents) to count the
    # N-substituents on a suffix amine nitrogen — see _count_substituents.
    fg_hetero_atoms: Set[int] = set()
    # P-44.1(b) counting support: one entry per PCG INSTANCE, each the SET of
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
    if principal_group and principal_group in functional_groups:
        # DD5 RC-4 (P-44.1.1): count the principal characteristic group over its
        # WHOLE equal-seniority class (e.g. primary + secondary OH = two hydroxy
        # PCGs), so the chain bearing all of them wins on PCG count (the di-OH
        # chain -> 3-(4-chlorobutyl)pentane-1,4-diol, not the longer 1-OH chain).
        # Non-equalized groups fall back to the single subtype (byte-identical).
        from ..rules.seniority import (
            _SENIORITY_PARENT, _SENIORITY_CLASS_MEMBERS, _RC4_UNION_CLASSES,
            _CLASS_CHARACTERISTIC_Z, _normalize_pcg_match,
        )
        _parent_class = _SENIORITY_PARENT.get(principal_group)
        _members = _SENIORITY_CLASS_MEMBERS.get(_parent_class)
        _het_z = _CLASS_CHARACTERISTIC_Z.get(_parent_class)
        if _het_z is None:
            # v33 Phase 6: heteroatom-only-suffix acid classes (sulfonic/
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
        # v29 R1 (P-44.1.1): a SKELETAL-suffix PCG -- the '-one' family -- puts
        # the characteristic group's OWN atom into the parent hydride, so only
        # that atom may satisfy "this chain bears the PCG". The ketone SMARTS
        # '[#6][CX3](=O)[#6]' carries BOTH FLANKING carbons, and the whole-match
        # semantics below let a chain through a mere NEIGHBOUR of the carbonyl
        # score contains_fg=1 / fg_count=1. Criterion 3 (length) then handed the
        # win to a longer carbonyl-FREE chain, the acyl carbons were dropped as
        # an unnameable substituent (DROP-09) and the '=O' was re-expressed on
        # the attachment atom -- a SILENT ATOM DROP:
        #   CCCCCCCCC(CCCC)C(C)(CC(C)C)C(=O)C  (C21H42O)
        #     -> '5-butyl-2,4-dimethyltridecan-4-one'  (C19H38O, 2 C GONE)
        #
        # P-64.2.2.1 "Acyclic ketones" (BlueBookV2/BlueBookV2.md:28346) -- "(1)
        # substitutively, using the suffix 'one' ... Method (1) generates
        # preferred IUPAC names"; its examples `butan-2-one (PIN)`,
        # `heptan-3-one (PIN)` and `5-methylhexan-2-one (PIN)` all number the
        # CARBONYL CARBON as a skeletal atom of the parent chain.
        # P-44.1.1 (:18875) -- "The senior parent structure has the maximum
        # number of substituents corresponding to the principal characteristic
        # group (suffix)"; P-44.1 (:18873) -- these criteria "must always be
        # applied before those applicable to ... chains (see P-44.3)". A chain
        # without the carbonyl carbon bears ZERO ketones, so it loses at
        # P-44.1.1 before chain length is ever consulted.
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
            SKELETAL_SUFFIX_PGS, _pg_attachment_atoms,
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
        # Build the bearing-carbon set for each PCG instance (deduped by
        # heteroatom so a group present under >1 SMARTS subtype counts once).
        # Bearing carbons = carbons DIRECTLY bonded to the characteristic
        # heteroatom (never flanking carbons). When the heteroatom cannot be
        # located (het_z is None, e.g. carbonyl/acid classes) the instance keeps
        # the legacy whole-match semantics so their P-44.1(b) count is unchanged.
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
                # narrowed by v29 R1 to the PCG's own skeletal atom for the
                # SKELETAL_SUFFIX_PGS families (see the P-44.1.1 note above), so
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

    def _compute_fg_locant_score(chain: List[int]) -> tuple:
        """Criterion 6 (P-44.4h): Lowest locants for principal group."""
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
        """Criterion 7 (P-44.4j): Lowest locants for multiple bonds."""
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
        """Criterion 8 (P-44.4.1.1): Maximum number of substituents cited as
        prefixes on the chain.

        Counts every heavy chain-carbon neighbour off the chain (as before) AND,
        additionally, the prefix substituents hanging off a SUFFIX heteroatom (an
        amine N / alcohol O in ``fg_hetero_atoms``) directly bonded to the chain:
        N-methyl, N-(2-aminoethyl), … are genuine prefix substituents. This is
        strictly additive to the previous count (the heteroatom is still counted
        once as before), so it only ever changes the RESULT of a tie between two
        equal-length equal-PCG chains — never a chain that was already uniquely
        best. It breaks the P-44.4.1.1 tie for ``CN(C)CCN(C)CCN``: the middle
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
                if nbr_idx not in chain_set and nbr_idx not in exclude:
                    if nbr.GetSymbol() != 'H':
                        count += 1
                        if nbr_idx in fg_hetero_atoms:
                            # Suffix heteroatom: also count the prefix
                            # substituents ON it (its heavy neighbours other
                            # than this chain carbon). P-44.4.1.1.
                            for sub in nbr.GetNeighbors():
                                si = sub.GetIdx()
                                if si == atom_idx or si in exclude:
                                    continue
                                if sub.GetAtomicNum() > 1:
                                    count += 1
        return count

    def _cascade_reverse(chain: List[int]) -> bool:
        """True if the REVERSED chain gives lower locants by the orientation
        cascade PCG -> multiple bonds -> double bonds -> substituents
        (P-31.1.4 numbering order).

        Used so the substituent-locant criterion (idx 9) is read in the SAME
        orientation the higher-priority criteria fix, instead of independently
        re-minimizing (the old ``max(fwd, rev)`` evaluated a fictional
        orientation and mis-ranked chains tying on the higher criteria).
        """
        from ..rules.locants import compare_locant_sets
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
                    if ni not in cset and ni not in exclude and nbr.GetSymbol() != 'H':
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
        """Criterion 9 (P-44.4 / P-45.2.2): lowest substituent locants, read in the
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
                if nbr_idx not in chain_set and nbr_idx not in exclude and nbr.GetSymbol() != 'H':
                    positions.append(i)
                    break
        if not positions:
            return ()
        return tuple(-p for p in sorted(positions))

    def _compute_double_bond_locant_score(chain: List[int]) -> tuple:
        """Criterion 7.5 (P-44.1(h)): Lowest locants for double bonds only.

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
        Implements all IUPAC 2013 P-44 criteria.

        Tuple elements:
          0: contains_fg (bool as int)
          1: fg_count (FG instances, not atoms -- P-44.1(b))
          2: length (chain length -- P-44.1(a))
          3: multiple_bonds (double + triple count -- P-44.1(c))
          4: double_bonds (double bond count -- P-44.1(d))
          5: fg_locants_score (lowest FG locants -- P-44.1(f))
          6: bond_locants_score (lowest multiple bond locants -- P-44.1(g))
          7: double_bond_locants_score (lowest double bond locants -- P-44.1(h))
          8: sub_count (substituent count -- P-44.1(i))
          9: sub_locants_score (lowest substituent locants -- P-44.1(j))
        """
        chain_set = set(chain)

        # Criterion 1: Contains principal group
        contains_fg = 1 if (fg_atoms & chain_set) else 0

        # Criterion 2: Count of principal group INSTANCES in chain.
        # P-44.1(b): count how many PCG INSTANCES are borne by this chain — an
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

        # Criterion 7.5 (P-44.1(h)): Lowest locants for double bonds only.
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
        """DD5 RC-1 (P-45.2.3 / P-45.2.2 / P-45.5): deterministic candidate
        comparator for chains that TIE on every chain_score term.

        Replaces the arbitrary ``max()`` fall-through (which returned whichever
        chain ``find_all_carbon_chains`` happened to enumerate first). The key is
        ``(full_substituent_locant_set, locants_in_alphanumerical_citation_order)``,
        both read in the chain's PCG-/substituent-lowest orientation, so the chain
        whose prefixes get the lowest locants in order of citation wins (lower key
        preferred): ``OCC(CCBr)CCCl`` -> ``2-(2-bromoethyl)-4-chlorobutan-1-ol``
        (not ``4-bromo-2-(2-chloroethyl)…``). Substituent names are resolved by the
        shared namer; the multiplier-free base name is the alpha sort key.
        """
        from ..assembly.naming_utils import alpha_sort_key
        from ..assembly.substituent_enumerator import name_substituent

        # WR-05: orient via the SAME full cascade `_compute_sub_locant_score` uses
        # (PCG -> multiple-bonds -> double-bonds -> substituents), not just the PCG,
        # so the citation-order tie-break is read in one consistent orientation; and
        # use the chain-enumeration boundary `combined_exclude` (chain set + the
        # non-principal FG terminal carbons removed from enumeration) so the named
        # substituent fragment matches what the real enumerator sees.
        oriented = list(reversed(chain)) if _cascade_reverse(chain) else chain
        cset = set(oriented)
        bfs_boundary = cset | combined_exclude
        pos = {a: i + 1 for i, a in enumerate(oriented)}
        entries = []  # (alpha_key, locant)
        sub_locants = []
        for chain_atom in oriented:
            for nbr in mol.GetAtomWithIdx(chain_atom).GetNeighbors():
                ni = nbr.GetIdx()
                if ni in cset or ni in combined_exclude or ni in fg_atoms:
                    continue
                if nbr.GetSymbol() == 'H':
                    continue
                frag = _bfs_substituent(mol, ni, bfs_boundary)
                try:
                    nm = name_substituent(mol, frag, ni)
                except Exception:
                    nm = "zzz"
                entries.append((alpha_sort_key(nm or "zzz"), pos[chain_atom]))
                sub_locants.append(pos[chain_atom])
        entries.sort()
        # P-45.6.1: when every structural criterion ties, the PIN is the name
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
        return (sorted(sub_locants),
                tuple(loc for _a, loc in entries),
                tuple(a for a, _loc in entries))

    def _lambda_direct_key(chain: List[int]) -> tuple:
        """P-45.3.1 (BB 22172-22182): among chains tying on every P-44 term,
        the PIN parent bears the substituent group of the HIGHEST bonding number
        directly connected to it (λ5 > λ3). Score = the multiset of nonstandard
        (λ) bonding numbers of the DIRECTLY-attached substituent atoms, sorted
        descending; the chain with the lexicographically-greatest tuple wins.

        For ``OC(=O)C(CP)C[PH4]`` the chain through the -CH2-PH4 arm makes λ5-P a
        direct substituent (key ``(5,)``); the chain through the -CH2-PH2 arm has
        no directly-attached λ atom (key ``()``) — so the former is the parent,
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

    # Find chain with highest score; break exact ties deterministically (P-45).
    scored = [(chain_score(c), c) for c in chains]
    best_score = max(s for s, _ in scored)
    top = [c for s, c in scored if s == best_score]
    if len(top) == 1:
        best_chain = top[0]
    else:
        # P-45.3.1 nonstandard-bonding-number criterion runs BEFORE the
        # alphanumerical comparator (all P-44 terms already tied within `top`).
        best_lambda = max(_lambda_direct_key(c) for c in top)
        top = [c for c in top if _lambda_direct_key(c) == best_lambda]
        best_chain = top[0] if len(top) == 1 else min(top, key=_p45_alpha_key)

    # Determine numbering direction (lowest locants for principal group)
    if principal_group and fg_atoms:
        best_chain = _orient_chain_for_lowest_locants(best_chain, fg_atoms)

    return best_chain


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
        >>> mol = Chem.MolFromSmiles('c1ccccc1C')  # toluene
        >>> # Phenyl atoms: 0-5, Methyl: 6
        >>> is_ring_substituent(mol, [0, 1, 2, 3, 4, 5], {6})
        True
        >>> mol2 = Chem.MolFromSmiles('CCCCC')  # pentane
        >>> is_ring_substituent(mol2, [0, 1, 2], set())
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
        >>> mol = Chem.MolFromSmiles('c1ccccc1CCC(=O)O')  # phenylpropanoic acid
        >>> classify_substituent(mol, [0,1,2,3,4,5], {6,7,8,9,10})
        {'type': 'ring', 'name': 'phenyl', 'atoms': [0,1,2,3,4,5], 'ring_atoms': (0,1,2,3,4,5)}
    """
    from ..rules.ring_substituents import get_ring_substituent_name, identify_ring_system

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
        # This is a ring substituent. WS-A task 9: name the WHOLE fragment
        # through the single ring-substituent chokepoint with its attachment
        # atom — the old per-first-SSSR-ring lookup truncated a fused system
        # to its first ring (naphthalenyl -> 'phenyl', a DIFFERENT group) and
        # never carried the P-29.2 free-valence locant.
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
