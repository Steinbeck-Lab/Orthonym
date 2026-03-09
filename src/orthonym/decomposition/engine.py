"""
Decomposition engine with quality gating and orchestration.

The engine decides WHETHER to decompose a molecule (quality gate),
calls bond_cleavage and fragment_capping to split it, names each
fragment recursively, and assembles the final multi-component name.

Quality gate prevents regressions: molecules that the existing pipeline
names correctly are left alone. Only molecules with poor names (unknown,
suspiciously short, etc.) are decomposed.
"""

import re
from collections import deque
from typing import Dict, List, Optional

from rdkit import Chem


# ---------------------------------------------------------------------------
# Performance guard
# ---------------------------------------------------------------------------

MAX_CLEAVABLE_BONDS = 12  # Skip decomposition if more than this many bonds
MAX_BOND_RETRY_ATTEMPTS = 5  # Max bonds to try when multi-bond retry is active


# ---------------------------------------------------------------------------
# Retained-name whitelist for quality gate
# ---------------------------------------------------------------------------

# Known retained names that correctly identify a core substructure even
# in large molecules (nucleotide cofactors, natural products, etc.).
# These bypass the "no digits and no hyphens" rejection for heavy_atoms > 20.
_RETAINED_CORE_NAMES = frozenset({
    'adenine', 'guanine', 'thymine', 'cytosine', 'uracil',
    'xanthine', 'hypoxanthine', 'purine', 'pyrimidine',
    'indole', 'quinoline', 'isoquinoline', 'acridine',
    'phenothiazine', 'xanthene', 'phenoxazine', 'thianthrene',
    '1h-indole',
})


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def _name_quality_is_acceptable(name: str, mol) -> bool:
    """Check if an existing name is good enough (no decomposition needed).

    Returns True if the name looks acceptable, False if decomposition
    should be attempted.

    Criteria for UNACCEPTABLE names:
    - None, empty, or "unknown"
    - Suspiciously short for a complex molecule (heavy_atoms > 15,
      name shorter than heavy_atoms // 2)
    - Inadequate char/atom ratio for very large molecules (heavy_atoms > 25,
      ratio < 0.45)
    - No digits and no hyphens for a large molecule (heavy_atoms > 20),
      which suggests only a retained name for one fragment was returned

    Args:
        name: The existing pipeline name (may be None).
        mol: RDKit Mol object for the molecule.

    Returns:
        True if name is acceptable (skip decomposition).
        False if name is poor (try decomposition).
    """
    if not name or name == "unknown":
        return False

    # Whitelist: known retained names that correctly identify a core
    # substructure even in large molecules (e.g., adenine in nucleotide
    # cofactors). These bypass all size-based rejection checks.
    if name.lower() in _RETAINED_CORE_NAMES:
        return True

    heavy_atoms = mol.GetNumHeavyAtoms()

    # Suspiciously short name for a complex molecule
    if heavy_atoms > 15 and len(name) < heavy_atoms // 2:
        return False

    # For very large molecules, an adequate name should have at least
    # 0.45 characters per heavy atom (locants, prefixes, parent name)
    if heavy_atoms > 25 and len(name) / heavy_atoms < 0.45:
        return False

    # Large molecule with no digits and no hyphens: likely just a retained
    # name for one fragment (e.g., "benzene" for a 25-atom ester)
    # BUT: skip this check for known retained core names that correctly
    # identify a core substructure even in large molecules (e.g., adenine
    # in nucleotide cofactors).
    if heavy_atoms > 20 and name.lower() not in _RETAINED_CORE_NAMES:
        has_digits = any(c.isdigit() for c in name)
        has_hyphens = "-" in name
        if not has_digits and not has_hyphens:
            return False

    # Multi-amide under-naming detection (PEP-01)
    # If molecule has multiple DISTINCT amide carbonyls but the name only
    # references one, the name is partial and decomposition should be attempted.
    # We count distinct carbonyl C atoms (acid_atom) rather than raw amide
    # bond count so that ureas (NC(=O)N -- one carbonyl, two C-N bonds)
    # are NOT falsely flagged as multi-amide.
    from .bond_cleavage import find_cleavable_bonds

    try:
        bonds = find_cleavable_bonds(mol)
        amide_bonds = [b for b in bonds if b.get("type") == "amide"]
        # Distinct carbonyl carbons involved in amide bonds
        distinct_amide_carbonyls = len(
            set(b["acid_atom"] for b in amide_bonds)
        )

        if distinct_amide_carbonyls >= 2:
            # Count how many amide-related patterns the name captures.
            # Each named amide bond should produce "amino", "amido",
            # "amide", "acetamid", "formamid", or similar.
            # Multiplier prefixes (di-, tri-, tetra-) before amide tokens
            # indicate multiple groups, so count them accordingly.
            _MULT_MAP = {'di': 2, 'tri': 3, 'tetra': 4, 'penta': 5}
            amide_refs = 0
            for m in re.finditer(
                r'(di|tri|tetra|penta)?(amino|amido|amide|acetamid|formamid)',
                name, re.IGNORECASE,
            ):
                prefix = m.group(1)
                amide_refs += _MULT_MAP.get(prefix.lower(), 1) if prefix else 1
            # If the name captures fewer amide references than distinct
            # amide carbonyls, the name is partial -- reject it
            if amide_refs < distinct_amide_carbonyls:
                return False

    except Exception:
        pass  # If bond detection fails, don't block on it

    # Multi-ring fragment drop detection: if the molecule has a cleavable bond
    # separating exactly two distinct ring systems, but the name only references
    # one ring system, reject the name as incomplete.
    #
    # Conservative approach: only trigger when ALL conditions are met:
    # 1. Molecule has exactly 2 ring systems separated by cleavable bonds
    # 2. At least one ring system on each side has >= 5 ring atoms
    # 3. The name's char/heavy-atom ratio is < 1.5
    # Exactly 2 ring systems avoids over-triggering on glycosides and
    # natural products with 3+ ring systems where a single name is correct.
    try:
        # Reuse bonds from above if available, otherwise re-detect
        if 'bonds' not in dir():
            bonds = find_cleavable_bonds(mol)

        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()

        if bonds and len(atom_rings) >= 2:
            ring_atom_sets = [set(r) for r in atom_rings]
            # Merge overlapping ring sets (fused rings are one system)
            merged_systems = []
            for rs in ring_atom_sets:
                merged = False
                for ms in merged_systems:
                    if rs & ms:
                        ms.update(rs)
                        merged = True
                        break
                if not merged:
                    merged_systems.append(set(rs))

            if len(merged_systems) == 2:
                # Only trigger for exactly 2 ring systems. Molecules with 3+
                # ring systems (glycosides, polycyclic natural products) often
                # have a single name that correctly covers all rings.
                for bond_info in bonds:
                    bond_idx = bond_info.get("bond_idx")
                    if bond_idx is None:
                        continue
                    bond_obj = mol.GetBondWithIdx(bond_idx)
                    a1 = bond_obj.GetBeginAtomIdx()
                    a2 = bond_obj.GetEndAtomIdx()

                    # BFS from each side of the cleavable bond to find
                    # which ring systems are reachable from each side
                    def _reachable_atoms(start, exclude):
                        visited = set()
                        queue = deque([start])
                        while queue:
                            a = queue.popleft()
                            if a in visited:
                                continue
                            visited.add(a)
                            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                                nidx = nb.GetIdx()
                                if nidx != exclude and nidx not in visited:
                                    queue.append(nidx)
                        return visited

                    side1_atoms = _reachable_atoms(a1, a2)
                    side2_atoms = _reachable_atoms(a2, a1)

                    # Find ring systems on each side
                    side1_ring_systems = [
                        ms for ms in merged_systems
                        if ms & side1_atoms and len(ms) >= 5
                    ]
                    side2_ring_systems = [
                        ms for ms in merged_systems
                        if ms & side2_atoms and len(ms) >= 5
                    ]

                    if side1_ring_systems and side2_ring_systems:
                        # Both sides have substantial ring systems.
                        # A name covering both ring systems needs enough
                        # characters to name each (parent + prefix/locants).
                        # Use a higher ratio threshold (1.5) because naming
                        # two ring systems requires substantially more chars
                        # than naming one ring + simple substituents.
                        ratio = len(name) / heavy_atoms if heavy_atoms else 999
                        if ratio < 1.5:
                            return False
    except Exception:
        pass  # Guard: never let ring detection crash the quality gate

    # Detect half-decomposition artifacts: consecutive duplicate words
    # e.g., "palmitate palmitate" indicates same fragment named twice
    words = name.split()
    for i in range(len(words) - 1):
        if words[i] == words[i + 1] and len(words[i]) > 3:
            return False

    # Detect garbled decomposition names with duplicated parent-like tokens.
    # E.g., "(ethanamide)-1-(...tetrahydropyranyl)ethanamide" has "ethanamide"
    # twice, indicating the assembly merged fragments incorrectly.
    # Substituent-like tokens (ending in -oxy, -yl, -amino, -ido) can
    # legitimately repeat (di-glucopyranosyloxy, dimethyl, etc.).
    _PARENT_SUFFIXES = ('amide', 'amine', 'anol', 'anone', 'anediol',
                        'anedione', 'anoic', 'anoate')
    _dup_words = re.findall(r'[a-z]{6,}', name.lower())
    if _dup_words:
        from collections import Counter as _DupCtr
        _dup_counts = _DupCtr(_dup_words)
        for _dw, _dcnt in _dup_counts.items():
            if _dcnt >= 2 and any(_dw.endswith(sfx) for sfx in _PARENT_SUFFIXES):
                return False

    return True


def _decomposition_is_worse(decomp_name: str, existing_name: str, mol) -> bool:
    """Check if decomposition produced a worse name than the existing pipeline.

    Returns True if existing_name should be preferred over decomp_name.
    This prevents decomposition from replacing an acceptable (if incomplete)
    name with a garbled or malformed one.

    Args:
        decomp_name: The decomposition result name.
        existing_name: The existing pipeline name (from name_fragment_recursively).
        mol: RDKit Mol object for the molecule.

    Returns:
        True if existing_name is better (decomposition is worse).
    """
    # Garbled token detection: names containing fragments that indicate
    # malformed assembly (e.g., "anedicarboxamide", "aneyl", "cycloane")
    garbled_patterns = [
        'anedicarboxamide', 'aneyl', 'unknown', 'cycloane',
        'acidyl',       # "phosphonic acidyl" etc. -- malformed decomposition assembly
        'thioateyl',    # malformed thioester assembly
        'sulfonamideyl',  # malformed sulfonamide assembly
        'phosphateyl',  # malformed phosphodiester assembly
    ]
    for pattern in garbled_patterns:
        if pattern in decomp_name.lower() and pattern not in existing_name.lower():
            return True

    # Bracket mismatch: more open than close or vice versa
    for open_ch, close_ch in [('(', ')'), ('[', ']'), ('{', '}')]:
        if decomp_name.count(open_ch) != decomp_name.count(close_ch):
            return True

    # Duplicate parent-like tokens: if the decomposed name has the same
    # parent suffix name appearing more than once (e.g., "ethanamide...ethanamide"),
    # the assembly is likely garbled. Only flag parent-like tokens (amide, amine, etc.),
    # not substituent prefixes that can legitimately repeat.
    _PARENT_SFXS = ('amide', 'amine', 'anol', 'anone', 'anediol',
                    'anedione', 'anoic', 'anoate')
    _dwords = re.findall(r'[a-z]{6,}', decomp_name.lower())
    if _dwords:
        from collections import Counter as _DCtr
        _dcounts = _DCtr(_dwords)
        for _dw, _dcnt in _dcounts.items():
            if _dcnt >= 2 and any(_dw.endswith(sfx) for sfx in _PARENT_SFXS):
                return True

    # Detect duplicated structural fragment names: if a long token (>= 15 chars)
    # appears 2+ times in the decomposition but not in the existing name, the
    # decomposition likely split the molecule into two copies of the same
    # structural fragment (garbled assembly).
    if _dwords:
        _existing_words = set(re.findall(r'[a-z]{6,}', existing_name.lower()))
        for _dw, _dcnt in _dcounts.items():
            if (_dcnt >= 2 and len(_dw) >= 15
                    and _dw not in _existing_words):
                return True

    return False


# ---------------------------------------------------------------------------
# Coverage-based quality gate for decomposition results
# ---------------------------------------------------------------------------

def _coverage_is_adequate(name: str, mol) -> bool:
    """Check if a decomposed name covers enough of the molecule.

    Uses name length as a proxy for atom coverage: a well-named molecule
    should have roughly 2+ characters per heavy atom (locants, prefixes,
    parent name, substituents). Names with less than the expected minimum
    length are considered inadequate coverage.

    NOTE: The locked decision specified leveraging coverage_scoring.py's
    retrieve_confidence(). However, retrieve_confidence() is only populated
    by the composer pipeline (compute_confidence()), not the decomposition
    path. This heuristic achieves the same rejection goal without requiring
    architectural changes to thread confidence through decomposition.
    See 87-RESEARCH.md Open Question 2.

    Applied ONLY to decomposition results, NOT to existing pipeline names.

    Args:
        name: The decomposition-produced name (may be None).
        mol: RDKit Mol object for the molecule.

    Returns:
        True if coverage is adequate.
        False if the name is too short for the molecule's size.
    """
    if not name:
        return False

    heavy_atoms = mol.GetNumHeavyAtoms()

    # Small molecules (<=10 heavy atoms) always pass -- coverage
    # heuristic is unreliable for tiny molecules.
    if heavy_atoms <= 10:
        return True

    # Heuristic: expect ~2.0 chars per heavy atom for a substitutive name.
    # However, functional class names (ester/amide decomposition) are more
    # compact (e.g., "phenyl palmitate" = 16 chars for 24 heavy atoms = 0.67).
    # Use 0.6 chars/HA threshold to avoid false positives on valid compact
    # names while still catching truly inadequate coverage (e.g., "methane"
    # for a 30-atom molecule = 0.23 chars/HA).
    expected_min = int(heavy_atoms * 0.6)
    return len(name) >= expected_min


# ---------------------------------------------------------------------------
# Bond selection
# ---------------------------------------------------------------------------

# Priority order for bond types (lower = higher priority)
_BOND_TYPE_PRIORITY = {
    "ester": 1,
    "amide": 2,
    "phosphodiester": 3,
    "thioester": 4,
    "glycosidic": 5,
    "sulfonamide": 6,
    "carbamate": 7,
    "ether": 8,
}


def _name_sugar_fragment(smiles: str) -> Optional[str]:
    """Try to name a fragment as a sugar using the retained names lookup.

    If the fragment's canonical SMILES matches a known sugar, returns
    the glycosyloxy prefix (e.g., "beta-D-glucopyranosyloxy").
    Returns None if not a recognized sugar.

    Args:
        smiles: Canonical SMILES of the sugar fragment.

    Returns:
        Glycosyloxy prefix string, or None if not a known sugar.
    """
    from ..data.sugar_names import lookup_sugar, sugar_to_glycosyloxy_prefix

    sugar_info = lookup_sugar(smiles)
    if sugar_info:
        anomer, config, base_name = sugar_info
        return sugar_to_glycosyloxy_prefix(anomer, config, base_name)
    return None


def _select_best_bond(mol, bonds: List[Dict]) -> Dict:
    """Select the single best bond to cleave.

    Priority:
    1. By bond type: ester > amide > phosphodiester > thioester >
       glycosidic > sulfonamide > carbamate > ether
    2. Among same type: prefer most balanced split (smallest
       abs(frag1_atoms - frag2_atoms))

    Args:
        mol: RDKit Mol object.
        bonds: List of bond info dicts from find_cleavable_bonds().

    Returns:
        The single best bond dict to cleave.
    """
    if len(bonds) == 1:
        return bonds[0]

    def _balance_score(bond_info: Dict) -> int:
        """Estimate how balanced a split would be.

        Uses a BFS from each side of the bond to count atoms
        reachable without crossing the bond.
        """
        bond_idx = bond_info["bond_idx"]
        rdkit_bond = mol.GetBondWithIdx(bond_idx)
        a1 = rdkit_bond.GetBeginAtomIdx()
        a2 = rdkit_bond.GetEndAtomIdx()

        # BFS from a1 without crossing the bond
        visited1 = set()
        queue = deque([a1])
        while queue:
            curr = queue.popleft()
            if curr in visited1:
                continue
            visited1.add(curr)
            atom = mol.GetAtomWithIdx(curr)
            for nbr in atom.GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx not in visited1:
                    # Don't cross the cleavage bond
                    if (curr == a1 and nidx == a2) or (curr == a2 and nidx == a1):
                        continue
                    queue.append(nidx)

        visited2 = set()
        queue = deque([a2])
        while queue:
            curr = queue.popleft()
            if curr in visited2:
                continue
            visited2.add(curr)
            atom = mol.GetAtomWithIdx(curr)
            for nbr in atom.GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx not in visited2:
                    if (curr == a1 and nidx == a2) or (curr == a2 and nidx == a1):
                        continue
                    queue.append(nidx)

        return abs(len(visited1) - len(visited2))

    # Sort: first by type priority, then by balance (smaller = better)
    return min(
        bonds,
        key=lambda b: (_BOND_TYPE_PRIORITY.get(b["type"], 99), _balance_score(b)),
    )


# ---------------------------------------------------------------------------
# Single-bond decomposition helper
# ---------------------------------------------------------------------------

def _try_single_bond_decompose(mol, bond: Dict, style: str = "pin") -> Optional[str]:
    """Attempt decomposition using a single bond.

    Extracted from try_decompose() Steps 4-8. Contains the full
    cleave-cap-name-assemble pipeline for one bond. Handles all 8
    bond types: ester, amide, phosphodiester, thioester, glycosidic,
    sulfonamide, carbamate, ether.

    Args:
        mol: RDKit Mol object to decompose.
        bond: Bond info dict from find_cleavable_bonds().
        style: Naming style ("pin" for preferred IUPAC names).

    Returns:
        Assembled multi-component IUPAC name, or None if decomposition
        fails at any step (capping, size guard, fragment naming, assembly).
    """
    from .fragment_capping import cleave_and_cap
    from ..assembly.fragment_naming import name_fragment_recursively
    from .fragment_assembly import assemble_fragment_name

    # Cleave and cap
    # For ether bonds, the acid side (larger fragment) gets H-cap (no OH),
    # while the alkyl side naturally keeps the ether oxygen as an alcohol.
    acid_oh = bond["type"] != "ether"
    fragments = cleave_and_cap(mol, [bond], acid_side_oh=acid_oh)
    if not fragments or len(fragments) < 2:
        return None

    # Size guard -- each fragment must be strictly smaller than parent
    parent_heavy = mol.GetNumHeavyAtoms()
    for frag in fragments:
        frag_mol = Chem.MolFromSmiles(frag["smiles"])
        if frag_mol and frag_mol.GetNumHeavyAtoms() >= parent_heavy:
            return None  # Fragment not smaller -- abort

    # Sort fragments smallest-first (by heavy atom count) so smaller
    # fragments populate the runtime cache before larger ones that may
    # contain similar structural motifs (IUPAC P-51 reuse principle).
    def _frag_sort_key(frag):
        frag_mol = Chem.MolFromSmiles(frag["smiles"])
        return frag_mol.GetNumHeavyAtoms() if frag_mol else 999
    fragments.sort(key=_frag_sort_key)

    # Name each fragment recursively (with sugar intercept for glycosidic bonds)
    fragment_names = {}
    for frag in fragments:
        frag_name = None

        # Sugar intercept: for glycosidic bonds, try sugar lookup on acid-side fragment
        if bond["type"] == "glycosidic" and frag["side"] == "acid":
            frag_name = _name_sugar_fragment(frag["smiles"])

        # Fall through to recursive naming if sugar lookup failed or non-sugar fragment
        if not frag_name:
            frag_name = name_fragment_recursively(frag["smiles"])

        if not frag_name or "unknown" in frag_name.lower():
            return None  # Cannot name a fragment -- abort
        fragment_names[frag["side"]] = frag_name

    # Substitutive naming preference: when the acid fragment already has a
    # principal group (sulfonic acid, phosphonic acid), prefer substitutive
    # naming over functional class assembly. This handles cases like sulfa
    # drugs where the acid fragment is already a well-formed parent name.
    #
    # IMPORTANT: Do NOT apply substitutive preference for esters or amides.
    # Esters use functional class naming ("alkyl alkanoate") and amides use
    # amide naming ("N-alkylalkanamide") -- these are the correct IUPAC forms.
    # Substitutive preference is only for bond types where the acid fragment
    # is already a principal-group parent that should receive a prefix.
    _SUBSTITUTIVE_BOND_TYPES = frozenset({
        "sulfonamide", "thioester", "phosphodiester",
    })
    if bond["type"] in _SUBSTITUTIVE_BOND_TYPES:
        acid_name = fragment_names.get("acid", "")
        _PG_INDICATORS = ("-ic acid", "-sulfonic acid", "-phosphonic acid",
                          "-carboxylic acid")
        acid_has_pg = any(acid_name.lower().endswith(ind) for ind in _PG_INDICATORS)

        if acid_has_pg:
            from .fragment_assembly import (_alcohol_to_alkyl, _amine_to_prefix,
                                            _join_components)
            alkyl_name = fragment_names.get("alkyl") or fragment_names.get("amine")
            if alkyl_name:
                sub_prefix = _alcohol_to_alkyl(alkyl_name) or _amine_to_prefix(alkyl_name)
                if sub_prefix:
                    substitutive_name = _join_components(sub_prefix, acid_name)
                    # Validate substitutive attempt: must pass quality gate and
                    # not be worse than functional class assembly
                    if substitutive_name and _name_quality_is_acceptable(substitutive_name, mol):
                        return substitutive_name

    # Amide seniority-based assembly: when amine fragment has higher
    # P-44.1.1 seniority than acid fragment, use substitutive naming
    # (amine becomes parent, acid becomes acyl prefix).
    if bond["type"] == "amide":
        try:
            from .fragment_ranker import acid_is_more_senior
            acid_frag = next((f for f in fragments if f["side"] == "acid"), None)
            amine_frag = next((f for f in fragments if f["side"] == "amine"), None)
            if acid_frag and amine_frag:
                if not acid_is_more_senior(acid_frag["smiles"], amine_frag["smiles"]):
                    # Amine is more senior -> substitutive naming
                    from .fragment_assembly import _acid_to_acyl, _join_components
                    acid_name = fragment_names.get("acid", "")
                    amine_name = fragment_names.get("amine", "")
                    acyl = _acid_to_acyl(acid_name)
                    if acyl and amine_name:
                        sub_name = f"N-{_join_components(acyl, amine_name)}"
                        if sub_name and _name_quality_is_acceptable(sub_name, mol):
                            return sub_name
        except Exception:
            pass  # Any failure: fall through to normal assembly

    # Assemble (delegate to fragment_assembly module)
    return assemble_fragment_name(bond["type"], fragment_names, style=style)


# ---------------------------------------------------------------------------
# Main decomposition entry point
# ---------------------------------------------------------------------------

def try_decompose(mol, style: str = "pin") -> Optional[str]:
    """Attempt decomposition of a molecule into named fragments.

    This is the main entry point for the decomposition engine. It:
    1. Finds cleavable bonds (ester, amide, phosphodiester, thioester,
       glycosidic, sulfonamide, carbamate, ether -- 8 types)
    2. Checks if the existing pipeline name is acceptable (quality gate)
    3. Selects the best bond to cleave
    4. Cleaves and caps the fragments
    5. Names each fragment recursively
    6. Assembles the multi-component IUPAC name

    Returns None if:
    - No cleavable bonds exist
    - The existing pipeline name is good enough (quality gate passes)
    - Fragment naming fails
    - Size guard detects non-shrinking fragments

    Args:
        mol: RDKit Mol object to decompose.
        style: Naming style ("pin" for preferred IUPAC names).

    Returns:
        Multi-component IUPAC name string, or None to fall through
        to the existing naming pipeline.
    """
    from .bond_cleavage import find_cleavable_bonds

    # Step 1: Find cleavable bonds
    bonds = find_cleavable_bonds(mol)
    if not bonds:
        return None  # No cleavable bonds, fall through

    # Step 1b: Performance guard -- skip if too many cleavable bonds
    if len(bonds) > MAX_CLEAVABLE_BONDS:
        return None  # Performance guard: too complex for decomposition

    # Step 2: Try existing pipeline first (via name_fragment_recursively
    # to respect the cycle guard).
    # If this SMILES is already in the visited set (being named up the
    # call stack), skip the probe — the caller already determined the
    # assembled name was inadequate, so proceed directly to decomposition.
    from ..assembly.fragment_naming import name_fragment_recursively, _fragment_guard, _get_visited

    existing_smiles = Chem.MolToSmiles(mol)
    visited = _get_visited()
    if existing_smiles in visited:
        # Already being named up the call stack — cycle detected.
        # Return None to let the caller's assembly pipeline handle naming
        # instead of decomposing (which would produce garbled results).
        return None
    else:
        # Temporarily disable the runtime fragment cache during this probe
        # so that intermediate results don't contaminate later naming.
        _saved_cache = getattr(_fragment_guard, 'cache', None)
        _fragment_guard.cache = None
        try:
            existing_name = name_fragment_recursively(existing_smiles)
        finally:
            _fragment_guard.cache = _saved_cache

    # Step 3: Quality gate -- only decompose if existing name is poor
    quality_ok = existing_name and _name_quality_is_acceptable(existing_name, mol)

    # Step 3b: Glycoside bypass -- if a glycosidic bond leads to a known sugar,
    # decomposition will produce a better name (retained sugar name vs systematic
    # oxane/tetrahydropyran). Only bypass when sugar lookup would succeed.
    glycoside_bypass = False
    if quality_ok:
        glycosidic_bonds = [b for b in bonds if b.get("type") == "glycosidic"]
        if glycosidic_bonds:
            from .fragment_capping import cleave_and_cap as _probe_cleave
            probe_bond = _select_best_bond(mol, glycosidic_bonds)
            probe_frags = _probe_cleave(mol, [probe_bond], acid_side_oh=True)
            if probe_frags:
                for pf in probe_frags:
                    if pf["side"] == "acid" and _name_sugar_fragment(pf["smiles"]):
                        glycoside_bypass = True
                        break

    if quality_ok and not glycoside_bypass:
        return None  # Existing name is good enough

    # Step 4: Choose ONE bond to cleave (the most significant one)
    best_bond = _select_best_bond(mol, bonds)

    # Steps 4-8: Single-bond attempt via helper
    single_result = _try_single_bond_decompose(mol, best_bond, style)

    # Coverage gate: reject decomposition results that don't cover enough
    # of the molecule's heavy atoms (applied only to decomposition output).
    if single_result and not _coverage_is_adequate(single_result, mol):
        single_result = None  # Coverage inadequate, discard this result

    # DECP-05: single-bond path returns result directly (backward compat).
    # Quality comparison: if decomposition produced a worse name than the
    # existing pipeline (garbled tokens, bracket mismatches), prefer the
    # existing name to avoid replacing a parseable name with garbage.
    if len(bonds) == 1:
        if single_result and existing_name:
            if _decomposition_is_worse(single_result, existing_name, mol):
                return existing_name
        return single_result

    # If single-bond result is adequate and not worse than existing, return it
    if single_result and _name_quality_is_acceptable(single_result, mol):
        if not (existing_name and _decomposition_is_worse(single_result, existing_name, mol)):
            return single_result

    # MULTI-BOND RETRY (DECP-01): try alternative bonds
    tried_indices = {best_bond["bond_idx"]}
    for bond in bonds:
        if bond["bond_idx"] in tried_indices:
            continue
        if len(tried_indices) >= MAX_BOND_RETRY_ATTEMPTS:
            break
        tried_indices.add(bond["bond_idx"])
        alt_result = _try_single_bond_decompose(mol, bond, style)
        # Coverage gate on retry results
        if alt_result and not _coverage_is_adequate(alt_result, mol):
            continue
        if alt_result and _name_quality_is_acceptable(alt_result, mol):
            # Also check that the alternative is not worse than existing name
            if existing_name and _decomposition_is_worse(alt_result, existing_name, mol):
                continue
            return alt_result

    # Compare decomposition result against existing pipeline name:
    # if decomposition produced a worse name (garbled, bracket-mismatched,
    # or containing malformed tokens), fall back to existing pipeline name.
    if single_result and existing_name:
        if _decomposition_is_worse(single_result, existing_name, mol):
            return None  # Let existing pipeline name be used

    return single_result  # Best effort fallback
