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

MAX_CLEAVABLE_BONDS = 20  # Phase 099: raised from 12, with structural complexity check
MAX_BOND_RETRY_ATTEMPTS = 5  # Max bonds to try when multi-bond retry is active
MAX_DECOMP_LEVELS = 3  # Phase 107: max iterative decomposition levels for mixed bond types


# ---------------------------------------------------------------------------
# Fragment naming with fallback (Phase 127)
# ---------------------------------------------------------------------------


def _name_fragment_with_fallback(smiles: str):
    """Name a fragment: recursive naming first, pipeline-only fallback.

    Per D-01: When name_fragment_recursively() returns None (depth/cycle limit hit),
    fall back to name_pipeline_only() instead of aborting the entire decomposition.
    name_pipeline_only() uses the full IUPAC pipeline without triggering decomposition
    recursion, preserving all substituents.

    Returns:
        IUPAC name string, or None if both attempts fail.
    """
    from ..assembly.fragment_naming import name_fragment_recursively
    from ..namer import name_pipeline_only

    # Primary: recursive naming (may trigger sub-decomposition)
    name = name_fragment_recursively(smiles)
    if name and "unknown" not in name.lower():
        return name

    # Fallback: full systematic pipeline without decomposition (D-01)
    name = name_pipeline_only(smiles)
    if name and "unknown" not in name.lower():
        return name

    return None


def _get_max_decomp_levels(mol) -> int:
    """Return max decomposition levels based on molecule size.

    Per D-04: HA > 50 molecules (phospholipids, polysaccharides) need one
    extra level to fully decompose mixed bond types.
    """
    if mol.GetNumHeavyAtoms() > 50:
        return 4
    return MAX_DECOMP_LEVELS  # Default: 3


# ---------------------------------------------------------------------------
# Bond type classification for tiered coverage thresholds
# ---------------------------------------------------------------------------

# Functional-class bond types produce compact names (e.g., "phenyl palmitate")
# and use a lower chars/HA coverage threshold (0.6).
# Substitutive bond types (sulfonamide, phosphodiester, ether, default) produce
# longer names and require a higher threshold (0.8).
_FUNCTIONAL_CLASS_TYPES = frozenset({"ester", "amide", "glycosidic", "carbamate", "thioester"})


# ---------------------------------------------------------------------------
# Bond-type-specific thresholds for multi-bond decomposition (Phase 099-04)
# ---------------------------------------------------------------------------

# Glycosidic bonds: threshold 2 (disaccharide + aglycone benefits from multi-bond).
# Ester/amide: threshold 3 (2-bond molecules better handled by single-bond).
# Phase 099-02 confirmed count>=2 causes regressions on 2-ester phospholipids.
_MULTI_BOND_THRESHOLD = {
    "ester": 2,       # Lowered from 3 per DECO-22/D-07: enables diester decomposition
    "glycosidic": 2,
    "amide": 3,
}


# ---------------------------------------------------------------------------
# Retained-name whitelist for quality gate
# ---------------------------------------------------------------------------

# Known retained names that correctly identify a core substructure even
# in large molecules (nucleotide cofactors, natural products, etc.).
# These bypass the "no digits and no hyphens" rejection for heavy_atoms > 25.
# Expanded in Phase 099 with fused heterocycle names.
_RETAINED_CORE_NAMES = frozenset({
    'adenine', 'guanine', 'thymine', 'cytosine', 'uracil',
    'xanthine', 'hypoxanthine', 'purine', 'pyrimidine',
    'indole', 'quinoline', 'isoquinoline', 'acridine',
    'phenothiazine', 'xanthene', 'phenoxazine', 'thianthrene',
    '1h-indole',
    # Phase 099 additions: fused heterocycles and polycyclics
    'flavone', 'chromone', 'coumarin', 'pteridine', 'phenazine',
    'carbazole', 'phenanthridine',
    # v28 Cluster B: the PIN forms carry the [b,d] fusion descriptor; the exact
    # match at :377/:425 needs them so a >25-heavy-atom core still passes the
    # quality gate. Bare forms kept (harmless) for any legacy path.
    'dibenzo[b,d]furan', 'dibenzo[b,d]thiophene',
    'dibenzofuran', 'dibenzothiophene',
    'fluorene', 'fluorenone',
    'anthracene', 'phenanthrene', 'chrysene',
})


# ---------------------------------------------------------------------------
# Ring-system token detection for quality gate
# ---------------------------------------------------------------------------

# Recognized ring-system name tokens. Names containing any of these tokens
# are considered structurally informative even without digits/hyphens.
# Used to bypass the no-digits/no-hyphens rejection in _name_quality_is_acceptable().
_RING_SYSTEM_TOKENS = frozenset({
    'pyridine', 'pyrimidine', 'pyrazine', 'pyridazine',
    'benzene', 'toluene', 'naphthalene', 'anthracene', 'phenanthrene',
    'morpholine', 'piperidine', 'piperazine', 'pyrrolidine',
    'indole', 'quinoline', 'isoquinoline', 'quinoxaline', 'quinazoline',
    'thiophene', 'furan', 'pyrrole', 'oxazole', 'thiazole', 'isoxazole',
    'imidazole', 'triazole', 'tetrazole',
    'carbazole', 'acridine', 'phenothiazine', 'phenoxazine',
    'flavone', 'chromone', 'coumarin', 'xanthene',
    'purine', 'pteridine', 'phenazine',
    'dibenzofuran', 'dibenzothiophene',
})


# ---------------------------------------------------------------------------
# Bond-type token matching for quality gate (Phase 132 - DECO-20)
# ---------------------------------------------------------------------------

# Token mapping: what name tokens indicate each bond type.
# Amide tokens include acyl prefixes (anoyl, enoyl, oyl)
# since N-acyl naming IS amide naming (IUPAC P-66.6.3).
# Moved to module level from _name_quality_is_acceptable() for testability.
_BOND_TYPE_TOKENS = {
    "ester": {"ester", "oate", "ate", "oyloxy",
              "acetyloxy", "benzoyloxy", "acetyl",
              "benzoyl"},
    "amide": {"amide", "amino", "amido", "acetamid",
              "formamid", "carbamoyl", "anilino",
              "anoyl", "enoyl", "oyl", "acyl"},
    "glycosidic": {"glycos", "pyranosyl", "furanosyl",
                   "glucos", "galactos", "mannos", "rhamn",
                   "fucos", "sugar", "osyl"},
    "phosphodiester": {"phosph", "nucleotid"},
    "thioester": {"thio"},
    "sulfonamide": {"sulfonamid", "sulfamid"},
    "carbamate": {"carbamat", "urethane"},
    # Ethers are common and don't always produce distinct
    # name tokens (ether O becomes "oxa" or is absorbed
    # into alkoxy prefixes). Don't penalize.
    "ether": set(),
    # Thioethers produce "thio" prefix in substitutive names
    "thioether": {"thio", "sulfanyl"},
    # Secondary amines produce "amino" prefix
    "sec_amine": {"amino"},
}


def _compile_token_patterns(token_dict):
    """Build regex patterns for IUPAC morpheme-aware token matching.

    Per D-14: tokens match at IUPAC nomenclature boundaries (after hyphen,
    after opening paren, at start of name, before closing paren, at end of name).
    Per D-15: short suffix tokens ('ate', 'oyl') match at word/morpheme end only,
    to prevent false positives from common English words.
    Per D-16: known false positive patterns ('polyester', 'polyamide', etc.) are
    excluded via a separate false-positive check BEFORE regex matching.

    Token matching strategy:
    - Short suffix tokens ('ate', 'oyl'): match at morpheme end only
    - Longer IUPAC morphemes (>= 4 chars): substring match is safe because
      these morphemes are specific enough. False positives from polymer names
      are handled by the _FALSE_POSITIVES exclusion list.
    """
    _SUFFIX_ONLY_TOKENS = {"ate", "oyl"}
    _FALSE_POSITIVES = {"polyester", "polyamide", "polyurethane", "polycarbonate"}

    compiled = {}
    for bond_type, tokens in token_dict.items():
        patterns = []
        for tok in tokens:
            escaped = re.escape(tok)
            if tok in _SUFFIX_ONLY_TOKENS:
                # Short suffix: match at morpheme end only to avoid
                # "calculate" matching "ate", "royal" matching "oyl"
                patterns.append(re.compile(
                    rf'{escaped}(?:$|[-)\s,])', re.IGNORECASE
                ))
            else:
                # Longer IUPAC morphemes: substring match is safe.
                # False positives from polymer names handled by exclusion list.
                patterns.append(re.compile(
                    rf'{escaped}', re.IGNORECASE
                ))
        compiled[bond_type] = patterns
    compiled["_false_positives"] = _FALSE_POSITIVES
    return compiled


# Pre-compiled patterns for use by quality gate
_COMPILED_TOKEN_PATTERNS = _compile_token_patterns(_BOND_TYPE_TOKENS)


def _token_matches_name(name_lower, token_patterns, bond_type):
    """Check if any token for this bond type matches in the name.

    Per D-16: first checks for false positive patterns.
    Per D-17: never raises -- returns True on any exception (benefit of doubt).
    """
    try:
        false_pos = token_patterns.get("_false_positives", set())
        for fp in false_pos:
            if fp in name_lower:
                return False

        patterns = token_patterns.get(bond_type, [])
        if not patterns:
            return True  # Empty token set (e.g., ether): benefit of doubt
        return any(p.search(name_lower) for p in patterns)
    except Exception:
        return True  # Guard: never crash the quality gate



def _name_has_ring_system_token(name: str) -> bool:
    """Check if a name contains a recognized ring-system token.

    Used by the quality gate to distinguish legitimate retained names
    (e.g., "phenothiazine" for a 26-atom molecule) from partial names
    that only cover a small fragment.

    Args:
        name: IUPAC name string.

    Returns:
        True if the name contains at least one recognized ring-system token.
    """
    name_lower = name.lower()
    return any(token in name_lower for token in _RING_SYSTEM_TOKENS)


# ---------------------------------------------------------------------------
# Name-size coverage heuristic (Phase 099-04)
# ---------------------------------------------------------------------------

def _name_covers_molecule(name: str, mol) -> bool:
    """Check if a pipeline name plausibly covers the whole molecule.

    ⚠ NOT a coverage measurement (Task Z2). The "estimated coverage" below is
    ``len(name) / 1.5 / heavy_atoms`` -- a CHARACTER COUNT rescaled by a
    constant. It is anti-correlated with real coverage: the correct retained
    name ``cholesterol`` scores 0.393 on its own 28-atom molecule, while a name
    that INVENTS atoms scores higher the more verbose it is. Real coverage is
    decided by constitution (InChIKey skeleton), never by a ratio --
    ``validation/atom_coverage.py``. That oracle spawns a JVM per call, so it
    cannot be used here: this function is called inside the decomposition
    recursion. Treat the number below as a cheap triage heuristic and nothing
    more; do not report it, or anything derived from it, as coverage.

    Only rejects when ALL conditions are met:
    (a) molecule has > 20 heavy atoms (small/medium always pass)
    (b) not in decomposition context (visited set empty)
    (c) molecule has cleavable bonds (alternative exists)
    (d) name is under 30 characters
    (e) estimated coverage < 0.45

    (a), (d) and (e) are the LIVE constants, re-read from the code 2026-08-02.
    This docstring previously said "> 15", "below 55%" and "< 0.55"; none of
    those values exists in the body, and the 0.55 was copied into the
    outward-facing  from
    here. Measured: (d) alone bypasses this function on 10 of the 16 molecules
    that reach it, and the rejection at (e) fired 0 times across 40 molecules.

    Args:
        name: The pipeline name.
        mol: RDKit Mol object for the molecule.

    Returns:
        True if name plausibly covers the molecule, False if it
        likely only describes a substructure.
    """
    try:
        heavy_atoms = mol.GetNumHeavyAtoms()

        # Small/medium molecule bypass: always accept for HA <= 20.
        # Calibrated at 20 (raised from 15) to avoid false positives on
        # molecules like heptanamide (HA=18, amide bond) where the pipeline
        # name is correct despite low coverage ratio.
        if heavy_atoms <= 20:
            return True

        # Decomposition context bypass: if visited set is non-empty,
        # we're evaluating a decomposition result -- always accept.
        from ..assembly.fragment_naming import _get_visited
        visited = _get_visited()
        if len(visited) > 0:
            return True

        # No-cleavable-bonds bypass: if no cleavable bonds exist,
        # the pipeline name is the best we can do regardless of coverage.
        from .bond_cleavage import find_cleavable_bonds
        bonds = find_cleavable_bonds(mol)
        if not bonds:
            return True

        # Long name bypass: names >= 30 chars describe something substantial
        # even for very large molecules. This prevents false positives on
        # names like "N-2-hydroxydocosanoyltetracosanolate" (36 chars, 60 HA)
        # which describe complex multi-functional compounds.
        if len(name) >= 30:
            return True

        # Coverage estimation: approximate how many heavy atoms the
        # name describes. IUPAC names use ~1.5 chars per HA for
        # retained names and ~2.0 for substitutive. Use conservative
        # 1.5 as divisor to avoid over-rejection.
        estimated_ha = len(name) / 1.5
        coverage = estimated_ha / heavy_atoms

        # If estimated coverage is below threshold, the name is partial.
        # Calibrated at 0.45 (lowered from 0.55 via benchmark-driven tuning:
        # 0.55 -> rejected "2-methylhexadecanoate" (0.52), 0.50 -> rejected
        # "17-phenylheptadecyl acetate" (0.49)). 0.45 correctly rejects
        # "5-chloroquinoline" (0.44) and "(22E)-stigmasta-7,22-diene" (0.42)
        # while accepting legitimate decomposition names.
        if coverage < 0.45:
            return False

        return True

    except Exception:
        return True  # Never crash the quality gate


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
      (unless the name contains a recognized ring-system token)

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
    # cofactors).
    #
    # Coverage guard (Phase 099-03): retained names are valid only if they
    # plausibly describe the whole molecule. "adenine" (7 chars) naming a
    # 58-HA molecule (ratio 0.12) indicates a partial match -- OPSIN
    # round-trips to adenine (10 HA), proving the name only covers 17%.
    # For molecules with HA > 20, require >= 0.25 chars/HA. This catches
    # adenine-in-CoA (58 HA, ratio 0.12) while preserving adenine
    # standalone (10 HA), phenothiazine (13 chars/14 HA = 0.93), and
    # flavone (7 chars/22 HA mock = 0.32).
    if name.lower() in _RETAINED_CORE_NAMES:
        heavy_atoms = mol.GetNumHeavyAtoms()
        if heavy_atoms <= 20 or len(name) / heavy_atoms >= 0.25:
            return True
        # Fall through to other checks (may trigger decomposition)

    heavy_atoms = mol.GetNumHeavyAtoms()

    # Suspiciously short name for a complex molecule.
    #
    # ⚠ THIS IS THE DOMINANT GUARD IN THIS FUNCTION (Task Z2). It is a
    # character count -- `len(name) < heavy_atoms // 2` is `chars/HA < ~0.5`
    # written without a division, which is why a grep for `len(name) /` misses
    # it. Because it runs first and is stricter than the chars/HA tests below,
    # it SHADOWS them: it is what rejects `cholesterol` on a 28-heavy-atom
    # steroid (11 < 14), not the 0.45 test that used to sit at the end of this
    # function. Before re-tuning anything below, check whether this line has
    # already returned.
    if heavy_atoms > 15 and len(name) < heavy_atoms // 2:
        return False

    # D-04: chars/HA check for medium molecules (15-30 HA).
    #
    # ⚠ The example this comment used to give -- "catches names like
    # 'quinoline' (0.39 chars/HA) for a 23 HA molecule" -- is FALSE, measured
    # 2026-08-02: 'quinoline' is in _RETAINED_CORE_NAMES and 0.391 >= 0.25, so
    # the retained-name branch above returns True and this block is never
    # reached for it. What this block does catch is the opposite shape: a name
    # long enough to clear the // 2 test above but still terse, e.g.
    # 'ethyl stearate' (0.636 on 22 HA) -- a CORRECT name, sent to
    # decomposition for being short. This is the one chars/HA test in this
    # function measured to fire on the naming path.
    # Threshold 0.65 (lowered from 0.7 to avoid rejecting "2-methylicosane" at 0.68).
    # Cleavable-bond bypass: only reject if decomposition is actually possible.
    # "hexadecane" (0.625) for 16 HA has no cleavable bonds -> no point rejecting.
    if 15 < heavy_atoms <= 30:
        chars_per_ha = len(name) / heavy_atoms
        if chars_per_ha < 0.65:
            from .bond_cleavage import find_cleavable_bonds
            try:
                bonds = find_cleavable_bonds(mol)
                if bonds:
                    return False  # Low ratio + cleavable bonds -> try decomposition
            except Exception:
                return False  # On error, still reject

    # REMOVED (Task Z2, 2026-08-02): a `heavy_atoms > 25 and
    # len(name)/heavy_atoms < 0.45 -> return False` test stood here. It was
    # UNREACHABLE, and provably so rather than by sampling: any name short
    # enough to trip it (len < 0.45*HA) is also short enough to trip the
    # `len(name) < heavy_atoms // 2` test above (len < HA//2), which returns
    # first. Checked exhaustively for every HA in 26..4000 -- no value admits
    # the one without the other. The spy agreed: 68 evaluations across 17 of
    # 40 molecules, 0 rejections. Deleting it changes no name.
    # Do not reintroduce a ratio test here; re-derive from the guard above.

    # Large molecule with no digits and no hyphens: likely just a retained
    # name for one fragment (e.g., "benzene" for a 25-atom ester).
    # Phase 099: expanded _RETAINED_CORE_NAMES handles known ring-system
    # retained names (phenothiazine, carbazole, flavone, etc.) via early
    # whitelist bypass above. For other names, the no-digits/no-hyphens
    # check remains at HA>20 to catch incomplete names.
    #
    # Fragment-aware threshold (Phase 099-02): when naming a fragment during
    # decomposition (visited set non-empty), use HA>30 to be more lenient --
    # medium-sized fragments with retained names are valid in decomposition
    # context. At top level (visited set empty), keep HA>20 for strictness.
    from ..assembly.fragment_naming import _get_visited
    visited = _get_visited()
    effective_threshold = 30 if len(visited) > 0 else 20
    if heavy_atoms > effective_threshold and name.lower() not in _RETAINED_CORE_NAMES:
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
            # C1 fix (P-66.3.1.1): find_cleavable_bonds labels each hydrazide
            # C(=O)-N bond as type 'amide', so a dihydrazide has
            # distinct_amide_carbonyls==2. The name token 'hydrazide' contains
            # no amide/amido token, so without counting it the complete GENERAL
            # name ('butanedihydrazide') was falsely rejected as under-naming.
            # Count the hydrazide suffix tokens too. Longest-first so
            # 'carbohydrazide' is not partial-matched as 'hydrazide'. Additive:
            # this can only RAISE amide_refs (make the gate more lenient), never
            # newly reject a name that passed before.
            for m in re.finditer(
                r'(di|tri|tetra|penta)?'
                r'(carbohydrazide|hydrazide|hydrazid|amino|amido|amide|acetamid|formamid)',
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
                    # C4 (P-62.2.2): an amine bridge between two ring systems is
                    # named SUBSTITUTIVELY (aniline parent + N-aryl prefix, e.g.
                    # diphenylamine -> 'N-phenylaniline'), NOT by decomposition.
                    # The substitutive name is legitimately compact (ratio ~1.15
                    # for N-phenylaniline / 13 HA), so the char-ratio proxy below
                    # would wrongly reject it and trigger a garbled cleavage
                    # ('phenylphenol'). The amine bridge is not a decomposition
                    # linkage (unlike ester/amide/ether); skip the ring-drop
                    # rejection for it. Scoped strictly to amine bond types.
                    if bond_info.get("type") in ("sec_amine", "tert_amine"):
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
                        #
                        # ⚠ MEASURED INERT (Task Z2, 2026-08-02): reached 3
                        # times on 1 of 40 molecules, rejected 0 times. Unlike
                        # the 0.45 test deleted above, this is NOT provably
                        # unreachable -- it is a character count that happens
                        # never to have fired on the sample -- so it was left
                        # in place rather than removed on a 40-molecule zero.
                        # Removing it admits new behaviour and needs the full
                        # gate. It is also anti-correlated with what it claims
                        # to check: a verbose name naming ONE ring passes,
                        # a terse name naming BOTH fails.
                        ratio = len(name) / heavy_atoms if heavy_atoms else 999
                        if ratio < 1.5:
                            return False
    except Exception:
        pass  # Guard: never let ring detection crash the quality gate

    # Multi-bond under-coverage detection (Phase 099-03): if molecule has
    # multiple distinct cleavable bond types but the name references fewer
    # than half, the name likely describes only one fragment of a
    # multi-component molecule.
    # Example: molecule with ester + phosphodiester bonds named "butanedioic
    # acid" only covers the acid fragment, not the ester or phosphodiester
    # linkages.
    #
    # IMPORTANT: This check is gated on coverage ratio < 0.8 to avoid
    # rejecting decomposition results that adequately describe the molecule.
    # Decomposition names like "N-2,3-dihydroxyhexacosanoylaminocyclohexane-
    # tetraol" (ratio 0.86) are complete despite not mentioning every bond
    # type, because they result from cleaving ONE bond and naming each half.
    # Only pipeline names with low coverage (< 0.8) are suspect.
    #
    # ⚠ THE PRE-FILTER IS A CHARACTER COUNT AND IT KEEPS THIS BLOCK SHUT
    # (Task Z2, 2026-08-02). `coverage_ratio` is `len(name)/heavy_atoms`, so
    # what "< 0.8" actually buys is: a name VERBOSE enough is exempt from the
    # bond-token check below -- which is a real structural test. Measured: the
    # ratio is computed 34 times across 13 of 40 molecules and the branch is
    # entered ZERO times. Neither of the two tests that name this block reach
    # it either -- test_quality_gate.py:360 is rejected earlier by the
    # `len(name) < heavy_atoms // 2` guard (its own comment concedes this) and
    # :331 by the D-04 chars/HA reject. So this block has no test coverage and
    # no observed execution.
    # NOT removed: a reachable window exists (ratio in [0.65, 0.8) for
    # 15 < HA <= 30), so unlike the 0.45 test deleted above it is not provably
    # dead, and dropping the pre-filter would START rejecting names. Either
    # change needs the full gate.
    try:
        # Reuse bonds from the multi-amide block above if available
        if 'bonds' not in dir():
            bonds = find_cleavable_bonds(mol)
        coverage_ratio = len(name) / heavy_atoms if heavy_atoms else 999
        if bonds and heavy_atoms > 15 and coverage_ratio < 0.8:
            distinct_bond_types = set(b["type"] for b in bonds)
            if len(distinct_bond_types) >= 2:
                # Use module-level boundary-aware token matching (Phase 132)
                # instead of substring matching to prevent false positives
                # like "polyester" triggering "ester" token.
                name_lower = name.lower()
                represented_types = 0
                for bt in distinct_bond_types:
                    tokens = _BOND_TYPE_TOKENS.get(bt, set())
                    if not tokens:
                        # ether: give benefit of doubt
                        represented_types += 1
                        continue
                    if _token_matches_name(name_lower, _COMPILED_TOKEN_PATTERNS, bt):
                        represented_types += 1
                # If less than half of distinct bond types are represented,
                # the name is partial -- reject it
                if represented_types < len(distinct_bond_types) / 2:
                    return False
    except Exception:
        pass  # Guard: never let bond detection crash the quality gate

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

    # Name-size coverage heuristic (Phase 099-04): reject names that
    # describe less than ~55% of molecule heavy atoms when cleavable bonds
    # exist and decomposition hasn't been attempted yet.
    if not _name_covers_molecule(name, mol):
        return False

    return True


def _decomposition_is_worse(decomp_name: str, existing_name: str, mol) -> bool:
    """Check if decomposition produced a worse name than the existing pipeline.

    Returns True if existing_name should be preferred over decomp_name.
    This prevents decomposition from replacing an acceptable (if incomplete)
    name with a garbled or malformed one.

    Args:
        decomp_name: The decomposition result name.
        existing_name: The existing pipeline name (from name_pipeline_only).
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

def _coverage_is_adequate(name: str, mol, bond_type: str = "") -> bool:
    """Check if a decomposed name covers enough of the molecule.

    Uses name length as a proxy for atom coverage: a well-named molecule
    should have roughly 2+ characters per heavy atom (locants, prefixes,
    parent name, substituents). Names with less than the expected minimum
    length are considered inadequate coverage.

    Tiered thresholds (Phase 099):
    - Functional-class types (ester, amide, glycosidic, carbamate, thioester)
      use 0.6 chars/HA -- these produce compact names like "phenyl palmitate".
    - Substitutive types (sulfonamide, phosphodiester, ether, default)
      use 0.8 chars/HA -- these produce longer substitutive names.

    Applied ONLY to decomposition results, NOT to existing pipeline names.

    Args:
        name: The decomposition-produced name (may be None).
        mol: RDKit Mol object for the molecule.
        bond_type: Bond type string from bond info dict (default="" uses 0.8).

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

    # Tiered threshold: functional-class bond types produce compact names
    # (e.g., "phenyl palmitate" = 0.67 chars/HA) and use a lower threshold.
    # Substitutive bond types produce longer names and need a higher threshold.
    threshold = 0.6 if bond_type in _FUNCTIONAL_CLASS_TYPES else 0.8
    expected_min = int(heavy_atoms * threshold)

    # Phase 107: Retained-name coverage bonus.
    # Names containing retained-name tokens (adenine, cholesterol, etc.)
    # describe more structure than their character count suggests -- retained
    # names are intentionally shorter than systematic names. Apply a 1.5x
    # multiplier to effective name length for coverage comparison.
    # This bonus applies ONLY at the fragment-level coverage check, NOT
    # to the final assembled name quality check.
    name_lower = name.lower()
    retained_bonus = 1.0
    for rn in _RETAINED_CORE_NAMES:
        if rn in name_lower:
            retained_bonus = 1.5
            break

    effective_length = len(name) * retained_bonus
    return effective_length >= expected_min


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
    "thioether": 9,
    "sec_amine": 10,
}


def _validate_assembly_tokens(assembled: str, fragment_names: list) -> bool:
    """Validate assembled name references tokens from ALL input fragment names.

    Per D-02: Each input fragment should contribute at least one morpheme
    (word token) to the assembled result. If a fragment's tokens are entirely
    absent, the assembly lost that fragment.

    Uses stem-level matching: for each fragment token, checks if a stem
    (first 3+ chars) appears in the assembled name. This handles IUPAC
    suffix transformations (e.g., "hexadecanoic" -> "hexadecanoate" shares
    stem "hexadecano", "ethanol" -> "ethyl" shares stem "eth").

    Args:
        assembled: The assembled IUPAC name string.
        fragment_names: List of individual fragment name strings.

    Returns:
        True if all fragments are represented in the assembly.
    """
    assembled_lower = assembled.lower()
    for name in fragment_names:
        # Extract significant tokens (>= 4 chars to skip locants/prefixes)
        tokens = [t for t in re.split(r'[-\s,()]+', name.lower()) if len(t) >= 4]
        if not tokens:
            continue  # Short-token-only fragments pass by default
        # Check if any token or its stem is present in the assembled name.
        # Uses progressively shorter stems to handle IUPAC suffix transformations
        # (e.g., "ethanol" -> "eth" stem in "ethyl", "propanoic" -> "propan" in "propanoate")
        found = False
        for tok in tokens:
            # Full token first
            if tok in assembled_lower:
                found = True
                break
            # Stem match: try stems of decreasing length down to 3 chars
            # (3-char stems like "eth" are safe for IUPAC roots: eth/meth/prop/etc.)
            for stem_len in range(len(tok) - 1, 2, -1):
                stem = tok[:stem_len]
                if stem in assembled_lower:
                    found = True
                    break
            if found:
                break
        if not found:
            return False
    return True


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

    # Sugar-detection bypass: if any glycosidic bond leads to a known sugar,
    # prefer it. Sugar fragments get retained names (beta-D-glucopyranosyloxy)
    # which are better than systematic oxane/tetrahydropyran names.
    # Cap probe at 3 glycosidic bonds to limit performance impact.
    glycosidic_candidates = [b for b in bonds if b.get("type") == "glycosidic"][:3]
    for bond in glycosidic_candidates:
        from .fragment_capping import cleave_and_cap
        probe_frags = cleave_and_cap(mol, [bond], acid_side_oh=True)
        if probe_frags:
            for pf in probe_frags:
                if pf["side"] == "acid" and _name_sugar_fragment(pf["smiles"]):
                    return bond  # Sugar bond takes priority

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

    # Name each fragment with fallback (Phase 127: D-01)
    # Phase 176 / D-03: build a parallel fragment_smiles dict in lockstep with
    # fragment_names, keyed by the SAME frag["side"], so the glycoside assembler
    # can inspect fragment structure (the sugar-skeleton deriver and the
    # aglycone seniority guard both need the SMILES, not just the names). This
    # is purely additive -- non-glycoside callers never read fragment_smiles.
    fragment_names = {}
    fragment_smiles = {}
    for frag in fragments:
        frag_name = None

        # Sugar intercept: for glycosidic bonds, try sugar lookup on acid-side fragment
        if bond["type"] == "glycosidic" and frag["side"] == "acid":
            frag_name = _name_sugar_fragment(frag["smiles"])

        # Fall through to fallback naming if sugar lookup failed or non-sugar fragment
        if not frag_name:
            frag_name = _name_fragment_with_fallback(frag["smiles"])

        if not frag_name or "unknown" in frag_name.lower():
            return None  # Truly unnameable -- abort
        fragment_names[frag["side"]] = frag_name
        fragment_smiles[frag["side"]] = frag["smiles"]

    # --- Seniority-based substitutive assembly for swapped-role bonds ---
    # When _maybe_swap_parent_roles() detected that the non-acid fragment
    # is the correct parent (roles_swapped=True), attempt substitutive
    # naming: acid fragment becomes a prefix on the larger parent fragment.
    # The acid_atom/alkyl_atom in the bond dict are NOT swapped -- only the
    # flag is set. We swap the naming roles here at assembly time.
    if bond.get("roles_swapped"):
        try:
            from .fragment_assembly import _acid_to_acyl, _join_components
            acid_name = fragment_names.get("acid", "")
            other_name = (fragment_names.get("alkyl")
                          or fragment_names.get("amine", ""))

            if bond["type"] == "ester":
                # Acyloxy prefix: "acetic acid" -> "(acetyloxy)" prefix on parent
                acyl = _acid_to_acyl(acid_name)
                if acyl and other_name:
                    sub_name = _join_components(f"({acyl}oxy)", other_name)
                    if sub_name and _name_quality_is_acceptable(sub_name, mol):
                        return sub_name

            elif bond["type"] == "thioester":
                # Acylthio prefix: "acetic acid" -> "(acetylthio)" prefix on parent
                acyl = _acid_to_acyl(acid_name)
                if acyl and other_name:
                    sub_name = _join_components(f"({acyl}thio)", other_name)
                    if sub_name and _name_quality_is_acceptable(sub_name, mol):
                        return sub_name

            elif bond["type"] in ("phosphodiester", "sulfonamide", "carbamate"):
                # For uncommon bond types, fall through to normal assembly.
                # The roles_swapped flag is informational; the existing assembler
                # will attempt naming with the original (un-swapped) atom indices.
                pass

            # Fallback: if substitutive naming failed, fall through to normal assembly
        except Exception:
            pass  # Any failure: fall through to normal assembly

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
    # Guard: skip if roles_swapped is already True (detection-time swap
    # already handled the seniority assignment -- swapping again would
    # double-invert). In practice, amide roles_swapped is always False
    # (amides are exempt from detection-time swap), but this guard
    # provides defense-in-depth.
    if bond["type"] == "amide" and not bond.get("roles_swapped"):
        try:
            from .fragment_ranker import acid_is_more_senior
            acid_frag = next((f for f in fragments if f["side"] == "acid"), None)
            amine_frag = next((f for f in fragments if f["side"] == "amine"), None)
            if acid_frag and amine_frag:
                if not acid_is_more_senior(acid_frag["smiles"], amine_frag["smiles"]):
                    # Amine is more senior -> substitutive naming
                    from .fragment_assembly import _acid_to_acyl, _join_components
                    from ..assembly.naming_utils import _wrap_n_substituent
                    acid_name = fragment_names.get("acid", "")
                    amine_name = fragment_names.get("amine", "")
                    acyl = _acid_to_acyl(acid_name)
                    if acyl and amine_name:
                        wrapped_acyl = _wrap_n_substituent(acyl)
                        sub_name = f"N-{_join_components(wrapped_acyl, amine_name)}"
                        if sub_name and _name_quality_is_acceptable(sub_name, mol):
                            return sub_name
        except Exception:
            pass  # Any failure: fall through to normal assembly

    # Assemble (delegate to fragment_assembly module)
    # Phase 176 / D-03: thread fragment_smiles additively; only the glycoside
    # assembler reads it, every other assembler ignores the kwarg.
    return assemble_fragment_name(
        bond["type"], fragment_names, style=style, fragment_smiles=fragment_smiles
    )


# ---------------------------------------------------------------------------
# Multi-bond same-type decomposition helper
# ---------------------------------------------------------------------------

def _try_multi_bond_decompose(
    mol, bonds: List[Dict], style: str = "pin"
) -> Optional[str]:
    """Attempt decomposition using multiple same-type bonds simultaneously.

    Cleaves all provided bonds at once, names each fragment, and assembles
    a multi-component name. Currently supports ester bonds (polyester naming
    per IUPAC P-65.6.3.4). Other bond types fall through to None.

    Args:
        mol: RDKit Mol object to decompose.
        bonds: List of bond info dicts (must be >= 2, all same type).
        style: Naming style ("pin" for preferred IUPAC names).

    Returns:
        Multi-component IUPAC name, or None if decomposition fails.
    """
    # Validate: need >= 2 bonds, all same type
    if not bonds or len(bonds) < 2:
        return None

    bond_type = bonds[0]["type"]
    if not all(b["type"] == bond_type for b in bonds):
        return None

    from .fragment_capping import cleave_and_cap
    from .fragment_assembly import (
        _assemble_multi_ester,
        _assemble_multi_glycoside,
        _assemble_multi_amide,
    )

    # Cleave with acid_side_oh = True for ester/amide/glycosidic, False for ether
    acid_oh = bond_type != "ether"
    fragments = cleave_and_cap(mol, bonds, acid_side_oh=acid_oh)
    if not fragments or len(fragments) < 2:
        return None

    # Size guard: every fragment must be strictly smaller than parent
    parent_heavy = mol.GetNumHeavyAtoms()
    for frag in fragments:
        frag_mol = Chem.MolFromSmiles(frag["smiles"])
        if frag_mol and frag_mol.GetNumHeavyAtoms() >= parent_heavy:
            return None

    # Sort fragments smallest-first for cache warming
    def _frag_sort_key(frag):
        frag_mol = Chem.MolFromSmiles(frag["smiles"])
        return frag_mol.GetNumHeavyAtoms() if frag_mol else 999
    fragments.sort(key=_frag_sort_key)

    # Name each fragment (sugar intercept for glycosidic)
    # DECO-17: use list-of-tuples to preserve identical SMILES duplicates
    # DECO-25: partial assembly -- collect successful fragments, track failures
    import logging
    logger = logging.getLogger(__name__)

    named_fragments = []  # List[Tuple[Dict, str]] -- preserves all including duplicates
    failed_count = 0
    for frag in fragments:
        frag_name = None

        # Sugar intercept for glycosidic bonds
        if bond_type == "glycosidic" and frag["side"] == "acid":
            frag_name = _name_sugar_fragment(frag["smiles"])

        if not frag_name:
            frag_name = _name_fragment_with_fallback(frag["smiles"])

        if not frag_name or "unknown" in frag_name.lower():
            # DECO-25: track failure but don't abort yet
            failed_count += 1
            logger.debug("Fragment naming failed for %s (side=%s)",
                         frag["smiles"], frag.get("side", "?"))
            continue

        # D-05: Fragment naming size validation -- reject names that cover
        # less than 60% of the fragment's heavy atoms.
        # Bypass for retained core names (adenine, purine, etc.) which correctly
        # identify the key structural feature even when the fragment includes
        # additional atoms (e.g., adenosine-phosphate fragment named "adenine").
        frag_mol = Chem.MolFromSmiles(frag["smiles"])
        if frag_mol:
            frag_ha = frag_mol.GetNumHeavyAtoms()
            is_retained_core = frag_name.lower() in _RETAINED_CORE_NAMES
            if (frag_ha > 5
                    and not is_retained_core
                    and not _name_covers_molecule(frag_name, frag_mol)):
                logger.debug("Fragment name '%s' rejected: poor coverage for %s (%d HA)",
                             frag_name, frag["smiles"], frag_ha)
                failed_count += 1
                continue

        named_fragments.append((frag, frag_name))

    # DECO-25: require at least 2 named fragments for assembly
    if len(named_fragments) < 2:
        return None  # Not enough fragments to assemble

    # Log partial assembly if some fragments failed
    if failed_count > 0:
        logger.info("Partial assembly: %d/%d fragments named (%d failed)",
                    len(named_fragments), len(fragments), failed_count)

    # Dispatch to bond-type-specific multi-fragment assembler
    if bond_type == "ester":
        result = _assemble_multi_ester(named_fragments, style)
    elif bond_type == "glycosidic":
        result = _assemble_multi_glycoside(named_fragments, style)
    elif bond_type == "amide":
        result = _assemble_multi_amide(named_fragments, style)
    else:
        return None

    if not result:
        return None

    # Quality checks on the assembled result
    if not _coverage_is_adequate(result, mol, bond_type=bond_type):
        return None

    return result


# ---------------------------------------------------------------------------
# Iterative mixed-type decomposition (Phase 107)
# ---------------------------------------------------------------------------

def _try_iterative_mixed_decompose(
    mol, bonds: List[Dict], style: str = "pin"
) -> Optional[str]:
    """Iteratively decompose large molecules with mixed bond types.

    After initial single-bond cleavage, re-scan each resulting fragment
    for additional cleavable bonds of DIFFERENT types. This handles
    phospholipid-type molecules (phosphodiester + ester bonds) and
    glycoside-ester hybrids.

    Algorithm:
    1. Initial cleavage with _select_best_bond()
    2. For each fragment with HA > 30 that still contains cleavable bonds
       of a DIFFERENT type, cleave the best sub-bond
    3. Maximum MAX_DECOMP_LEVELS levels (iterative, not recursive)
    4. Quality gate comparison at each level

    Args:
        mol: RDKit Mol object.
        bonds: List of all cleavable bond dicts.
        style: Naming style.

    Returns:
        Assembled name string, or None if decomposition fails or is worse.
    """
    import logging
    from .bond_cleavage import find_cleavable_bonds
    from .fragment_capping import cleave_and_cap

    logger = logging.getLogger(__name__)

    if not bonds or len(bonds) < 2:
        return None

    # Need at least 2 different bond types for mixed decomposition
    bond_types_present = set(b["type"] for b in bonds)
    if len(bond_types_present) < 2:
        return None

    # Initial cleavage
    best_bond = _select_best_bond(mol, bonds)
    initial_frags = cleave_and_cap(mol, [best_bond], acid_side_oh=True)
    if not initial_frags or len(initial_frags) < 2:
        return None

    used_bond_types = {best_bond["type"]}

    # Track bond type metadata on fragments (DECO-23)
    all_fragments = []
    for frag in initial_frags:
        frag['parent_bond_type'] = best_bond['type']
        all_fragments.append(frag)

    parent_heavy = mol.GetNumHeavyAtoms()

    # Iterative decomposition across levels (Phase 127: D-04 conditional levels)
    max_levels = _get_max_decomp_levels(mol)
    for level in range(max_levels - 1):  # Already did level 0
        new_fragments = []
        changed = False
        for frag in all_fragments:
            frag_mol = Chem.MolFromSmiles(frag["smiles"])
            if frag_mol is None:
                new_fragments.append(frag)
                continue
            frag_ha = frag_mol.GetNumHeavyAtoms()
            if frag_ha <= 30:
                new_fragments.append(frag)
                continue
            # Re-scan for cleavable bonds of DIFFERENT types
            sub_bonds = find_cleavable_bonds(frag_mol)
            sub_bonds = [b for b in sub_bonds if b["type"] not in used_bond_types]
            if not sub_bonds:
                new_fragments.append(frag)
                continue
            # Cleave the best sub-bond
            sub_best = _select_best_bond(frag_mol, sub_bonds)
            sub_frags = cleave_and_cap(frag_mol, [sub_best], acid_side_oh=True)
            if sub_frags and len(sub_frags) >= 2:
                # Size guard: sub-fragments must be smaller than original
                all_smaller = all(
                    (Chem.MolFromSmiles(sf["smiles"]) is None or
                     Chem.MolFromSmiles(sf["smiles"]).GetNumHeavyAtoms() < frag_ha)
                    for sf in sub_frags
                )
                if all_smaller:
                    # Tag new fragments with their bond type (DECO-23)
                    for sf in sub_frags:
                        sf['parent_bond_type'] = sub_best['type']
                    new_fragments.extend(sub_frags)
                    used_bond_types.add(sub_best["type"])
                    changed = True
                else:
                    new_fragments.append(frag)
            else:
                new_fragments.append(frag)
        all_fragments = new_fragments
        if not changed:
            break

    # Name each fragment (Phase 127: D-01 fallback)
    # DECO-17/Pitfall 5: use list-of-tuples to preserve identical SMILES duplicates
    # DECO-25: partial assembly -- collect successful fragments, track failures
    named_fragments = []
    failed_count = 0
    for frag in all_fragments:
        frag_name = _name_sugar_fragment(frag["smiles"])
        if not frag_name:
            frag_name = _name_fragment_with_fallback(frag["smiles"])
        if not frag_name or "unknown" in frag_name.lower():
            # DECO-25: track failure but don't abort yet
            failed_count += 1
            logger.debug("Mixed-decomp fragment naming failed for %s", frag["smiles"])
            continue

        # D-05: Fragment naming size validation (with retained core name bypass)
        frag_mol = Chem.MolFromSmiles(frag["smiles"])
        if frag_mol:
            frag_ha = frag_mol.GetNumHeavyAtoms()
            is_retained_core = frag_name.lower() in _RETAINED_CORE_NAMES
            if (frag_ha > 5
                    and not is_retained_core
                    and not _name_covers_molecule(frag_name, frag_mol)):
                logger.debug("Mixed-decomp fragment name '%s' rejected: poor coverage for %s (%d HA)",
                             frag_name, frag["smiles"], frag_ha)
                failed_count += 1
                continue

        named_fragments.append((frag, frag_name))

    # DECO-25: require at least 2 named fragments for assembly
    if len(named_fragments) < 2:
        return None  # Not enough fragments to assemble

    if failed_count > 0:
        logger.info("Mixed-decomp partial assembly: %d/%d fragments named (%d failed)",
                    len(named_fragments), len(all_fragments), failed_count)

    # Bond-type-aware assembly (DECO-23): route fragment pairs through
    # bond-type-specific assemblers instead of naive space-join
    from .fragment_assembly import _assemble_by_bond_type

    assembled = _assemble_by_bond_type(named_fragments, used_bond_types, style)

    if not assembled:
        # Fallback: join with spaces (functional class style)
        sorted_named = sorted(
            named_fragments,
            key=lambda pair: len(pair[1]),
            reverse=True,
        )
        if len(sorted_named) <= 1:
            return None
        parts = [name for _, name in sorted_named]
        assembled = " ".join(parts)

    # Token validation (DECO-27): reject if assembly lost a fragment
    fragment_names = [name for _, name in named_fragments]
    if not _validate_assembly_tokens(assembled, fragment_names):
        # Bond-type assembly lost a fragment -- try space-join fallback
        sorted_named = sorted(
            named_fragments,
            key=lambda pair: len(pair[1]),
            reverse=True,
        )
        parts = [name for _, name in sorted_named]
        assembled = " ".join(parts)
        # Re-validate the fallback
        if not _validate_assembly_tokens(assembled, fragment_names):
            return None  # Assembly lost a fragment -- reject

    # Quality gate: the assembled name must be acceptable
    if not _name_quality_is_acceptable(assembled, mol):
        return None

    # Coverage gate
    if not _coverage_is_adequate(assembled, mol, bond_type=best_bond["type"]):
        return None

    return assembled


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

    # Step 2: Get existing pipeline name via systematic-only path.
    # name_pipeline_only() skips decomposition, so it cannot recurse back
    # into try_decompose(). No cache isolation needed.
    # If this SMILES is already in the visited set (being named up the
    # call stack), skip the probe — the caller already determined the
    # assembled name was inadequate, so proceed directly to decomposition.
    from ..namer import name_pipeline_only
    from ..assembly.fragment_naming import _get_visited

    existing_smiles = Chem.MolToSmiles(mol)
    visited = _get_visited()
    if existing_smiles in visited:
        # Already being named up the call stack — cycle detected.
        # Return None to let the caller's assembly pipeline handle naming
        # instead of decomposing (which would produce garbled results).
        return None
    existing_name = name_pipeline_only(existing_smiles, style=style)

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
    if single_result and not _coverage_is_adequate(single_result, mol, bond_type=best_bond["type"]):
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
        if alt_result and not _coverage_is_adequate(alt_result, mol, bond_type=bond["type"]):
            continue
        if alt_result and _name_quality_is_acceptable(alt_result, mol):
            # Also check that the alternative is not worse than existing name
            if existing_name and _decomposition_is_worse(alt_result, existing_name, mol):
                continue
            return alt_result

    # Phase 107: Iterative mixed-type decomposition.
    # If single-bond retry produced no acceptable result AND the molecule has
    # cleavable bonds of multiple types, try iterative decomposition.
    if not single_result or not _name_quality_is_acceptable(single_result, mol):
        bond_types_present = set(b["type"] for b in bonds)
        if len(bond_types_present) >= 2 and mol.GetNumHeavyAtoms() > 30:
            iterative_result = _try_iterative_mixed_decompose(mol, bonds, style)
            if iterative_result:
                if not (existing_name and _decomposition_is_worse(iterative_result, existing_name, mol)):
                    return iterative_result

    # MULTI-BOND SAME-TYPE cleavage (Phase 099): if N+ bonds of the same
    # type exist, try cleaving all same-type bonds simultaneously
    # (IUPAC P-65.6.3.4 polyesters / triglycerides, P-68 glycosides).
    # Bond-type-specific thresholds (Phase 099-04):
    # - glycosidic: 2 (disaccharide + aglycone)
    # - ester/amide: 3 (2-bond molecules better handled by single-bond)
    # Only attempt when single-bond produced no result at all.
    from collections import Counter as _BondCounter
    bond_type_counts = _BondCounter(b["type"] for b in bonds)
    if not single_result:
        for bond_type_key, count in bond_type_counts.most_common():
            threshold = _MULTI_BOND_THRESHOLD.get(bond_type_key, 99)
            if count >= threshold:
                same_type_bonds = [b for b in bonds if b["type"] == bond_type_key]
                multi_result = _try_multi_bond_decompose(mol, same_type_bonds, style)
                if multi_result:
                    if not (existing_name and _decomposition_is_worse(multi_result, existing_name, mol)):
                        return multi_result

    # Phase 107: Mixed-bond threshold relaxation.
    # When total cleavable bonds >= 3 across all types and no single type
    # reached its threshold above, relax the ester threshold to 2 for
    # mixed-type molecules (e.g., 2 esters + 1 glycosidic).
    # Only for ester type (glycosidic already at 2, amide stays at 3).
    # Guard: only when single-bond produced no acceptable result.
    if not single_result or not _name_quality_is_acceptable(single_result, mol):
        total_cleavable = sum(bond_type_counts.values())
        ester_count = bond_type_counts.get("ester", 0)
        if total_cleavable >= 3 and ester_count >= 2 and len(bond_type_counts) >= 2:
            # Only apply relaxation if standard threshold was NOT reached
            if ester_count < _MULTI_BOND_THRESHOLD.get("ester", 99):
                ester_bonds = [b for b in bonds if b["type"] == "ester"]
                multi_result = _try_multi_bond_decompose(mol, ester_bonds, style)
                if multi_result:
                    if not (existing_name and _decomposition_is_worse(multi_result, existing_name, mol)):
                        return multi_result

    # Ester-specific fallback: when single_result exists but failed quality,
    # multi-bond ester may produce a better name by exposing the clean core
    # (e.g., removing 3 peripheral acetyloxy groups from a tetracyclic ring).
    if single_result and not _name_quality_is_acceptable(single_result, mol):
        ester_count = bond_type_counts.get("ester", 0)
        if ester_count >= _MULTI_BOND_THRESHOLD.get("ester", 99):
            ester_bonds = [b for b in bonds if b["type"] == "ester"]
            multi_result = _try_multi_bond_decompose(mol, ester_bonds, style)
            if multi_result and _name_quality_is_acceptable(multi_result, mol):
                if not (existing_name and _decomposition_is_worse(multi_result, existing_name, mol)):
                    return multi_result

    # Compare decomposition result against existing pipeline name:
    # if decomposition produced a worse name (garbled, bracket-mismatched,
    # or containing malformed tokens), fall back to existing pipeline name.
    if single_result and existing_name:
        if _decomposition_is_worse(single_result, existing_name, mol):
            return None  # Let existing pipeline name be used

    return single_result  # Best effort fallback
