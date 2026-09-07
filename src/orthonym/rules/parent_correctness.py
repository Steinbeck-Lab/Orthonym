"""Parent-correctness scorer for Phase 145.1.

Scaffolds the 5th confidence factor (`parent_correctness`) in the candidate
scoring pipeline. Phase 145.1 wires this into FACTOR_WEIGHTS with weight 0.0
so the factor is computed and logged but does NOT influence selection
(byte-identical safe by D-14 IEEE 754 + Python 3.7+ dict-order). Phase 146
raises the weight after 80/20 train/test calibration to activate it.

REFERENCE SOURCE (D-04, locked in 145.1-CONTEXT.md):
    OPSIN round-trip of the reference name. The only non-circular option:
    - Option A (chosen): OPSIN parses the reference name -> reference SMILES.
      OPSIN is the inverse of Orthonym; reference names from ChEBI / PubChem
      / OPSIN self-test are external ground truth when they parse.
    - Option B (rejected -- Phase 146's work): a separate rule-based selector
      would pre-empt Phase 146.
    - Option C (rejected -- too narrow): hand-curated parent map covers only
      the 500-compound opsin_selftest corpus.

EXTRACTION PIPELINE (D-05, CD-01 -- locked from RESEARCH §4.2):
    E1: regex parent-token + OPSIN re-parse + RDKit substructure match +
        canonical-rank tiebreak. Pure Python + subprocess + RDKit;
        no Java<->Python bridge dependency. Verified on 6/9 test cases;
        remaining 3 fall back to 0.5 (no-decision = safe).

THREAD-LOCAL I/O (D-07, locked):
    Module-level _pc_context = threading.local() mirrors the established
    coverage_scoring._confidence_store pattern at coverage_scoring.py:404.
    Benchmark runners call set_reference_name(name) BEFORE orthonym.name(smiles).
    Production callers (no reference set) immediately return 0.5 with zero
    OPSIN cost.

INVARIANT (security threat T-145.1-01 mitigation):
    Production code path NEVER invokes OPSIN subprocess. Scorer.score()
    short-circuits to 0.5 when _pc_context.reference_name is None
    (the production default). OPSIN is invoked ONLY in benchmark mode
    when set_reference_name() has been explicitly called by the
    benchmark runner.

FAILURE MODES (RESEARCH §4.4 -- all return 0.5 no-decision):
    - OPSIN can't parse reference name
    - Parent token regex returns empty
    - OPSIN parses parent token to invalid SMILES
    - Substructure match returns 0 hits (ChEBI noise)
    - Substructure match returns >1 hits -> canonical-rank tiebreak
    - OPSIN CLI subprocess timeout (caught)
    - _pc_context.reference_name is None (production path)
    - candidate.parent_atom_indices is None (handler didn't report)
"""

import glob
import logging
import re
import subprocess
import threading
from pathlib import Path
from typing import Any, Optional, Set

from rdkit import Chem

from orthonym.jvm_flags import JVM_HYGIENE_FLAGS

from ..assembly.coverage_scoring import CandidateName

logger = logging.getLogger(__name__)

# OPSIN jar lives at repo root. Resolve via glob so any version
# (e.g., opsin-cli-2.9.0..., opsin-cli-2.10.0...) is picked up.
# IM-07: previously hardcoded to 2.9.0, breaking when a newer JAR
# replaced it. Pattern matches validation/atom_coverage.py:62.
_REPO_ROOT = Path(__file__).parent.parent.parent.parent

def _resolve_opsin_jar() -> Path:
    """Return path to the first matching OPSIN CLI JAR at repo root.

    Falls back to the canonical 2.9.0 filename if no match is found, so
    the FileNotFoundError raised by ``_opsin_to_smi`` (caught at line 122)
    still names a meaningful path in logs.
    """
    candidates = sorted(glob.glob(
        str(_REPO_ROOT / "opsin-cli-*-jar-with-dependencies.jar")
    ))
    if candidates:
        # Prefer highest-versioned (sort by name, take last).
        return Path(candidates[-1])
    return _REPO_ROOT / "opsin-cli-2.9.0-jar-with-dependencies.jar"

OPSIN_JAR = _resolve_opsin_jar()

# Match Phase 145 D-09 / benchmark_multi_corpus.py:DEFAULT_OPSIN_TIMEOUT (CD-05)
OPSIN_TIMEOUT: float = 10.0


# ---------------------------------------------------------------------------
# Thread-local context (mirrors coverage_scoring._confidence_store pattern)
# ---------------------------------------------------------------------------

_pc_context = threading.local()


def set_reference_name(name: Optional[str]) -> None:
    """Set the reference IUPAC name for the current naming call.

    Call BEFORE orthonym.name(smiles) in benchmark runs. Pass None to clear.
    Production callers (orthonym.name() in REPL/library use) MUST NOT call
    this -- leaving _pc_context.reference_name unset preserves byte-identical
    confidence values and avoids OPSIN subprocess cost.
    """
    _pc_context.reference_name = name


def clear_reference_name() -> None:
    """Clear the thread-local reference name (idempotent)."""
    _pc_context.reference_name = None


# ---------------------------------------------------------------------------
# OPSIN subprocess wrapper (CD-05 -- match benchmark_multi_corpus.py defaults)
# ---------------------------------------------------------------------------

def _opsin_to_smi(name: str) -> Optional[str]:
    """Parse name -> SMILES via OPSIN CLI subprocess.

    Returns the SMILES string on success, None on any failure (timeout,
    OPSIN can't parse, OPSIN jar missing, OS error). All failure paths
    log at DEBUG only -- no production-path noise.
    """
    try:
        p = subprocess.run(
            ["java", *JVM_HYGIENE_FLAGS, "-jar", str(OPSIN_JAR), "-o", "smi"],
            input=name + "\n",
            capture_output=True,
            text=True,
            timeout=OPSIN_TIMEOUT,
        )
        out = p.stdout.strip()
        return out if out else None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug("OPSIN failure on %r: %s", name, e)
        return None


# ---------------------------------------------------------------------------
# Parent-token extraction (CD-01 -- regex heuristic per RESEARCH §4.2)
# ---------------------------------------------------------------------------

# Strip leading locant cluster (e.g. "1,2-", "3a-", "2-")
_LOCANT_PREFIX_RE = re.compile(r'^\d+[a-z]?,?(\d+[a-z]?,?)*-?')


def _extract_parent_token(name: Optional[str]) -> Optional[str]:
    """Extract the parent-hydride token from an IUPAC name.

    Heuristic (CD-01): the parent token starts after the LAST top-level
    closing parenthesis (substituents are bracketed; parent is unbracketed
    at the end of the name). Strips leading locant clusters.

    Returns None if the heuristic produces nothing usable; scorer treats
    None as no-decision (returns 0.5).

    Verified test cases (RESEARCH §4.2 lines 666-678):
        "4-oxo-4-(prop-2-enoyloxy)but-2-enoic acid" -> "but-2-enoic acid"
        "ethanol" -> "ethanol"
        "3-(2-methoxyethyl)hexan-1-ol" -> "hexan-1-ol"
        "4-(4-chlorophenyl)butan-2-one" -> "butan-2-one"
        "benzene-1,2-diol" -> "benzene-1,2-diol"
        "1H-indole" -> "indole"  (1H- stripped)
        "2-methylpropanal" -> "methylpropanal" (no parens -- heuristic fails;
                              caller should treat as no-decision via OPSIN
                              re-parse downstream)
    """
    if not name:
        return None
    # Track parenthesis depth, find position after last top-level close paren
    depth = 0
    last_close = -1
    for i, c in enumerate(name):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                last_close = i
    parent = name[last_close + 1:] if last_close >= 0 else name
    # Strip leading locant cluster
    parent = _LOCANT_PREFIX_RE.sub('', parent).strip()
    return parent if parent else None


def opsin_reference_mol(ref_name: str) -> Optional[Any]:
    """OPSIN-parse the FULL reference name once -> reference RDKit mol.

    Phase 166 SCORE-03 (A1 strategy, 166-AUDIT §A1 OPSIN-Cost Prototype):
    the per-node scorer parses the reference name ONCE per compound, then
    does RDKit fragment-submol matching per node (NOT one OPSIN call per
    node). Returns None on any OPSIN/parse failure (mirrors _opsin_to_smi's
    caught-exception contract at :135-137). Reference names come from
    trusted corpora; OPSIN runs on STDIN (no shell), OPSIN_TIMEOUT=10.0.
    """
    smi = _opsin_to_smi(ref_name)
    if not smi:
        return None
    return Chem.MolFromSmiles(smi)


def match_token_atoms_in_mol(token: str, input_mol: Any) -> Optional[Set[int]]:
    """OPSIN-parse a parent-stem/fragment token and substructure-match it
    into input_mol, returning the matched atom-index set (or None).

    Phase 166 SCORE-03: the reusable per-node generalization of the
    atom-alignment step (extracted verbatim from the old inline body of
    _extract_reference_parent_atoms). 0 hits -> None; >1 hits ->
    CanonicalRankAtoms(breakTies=True) deterministic tiebreak (load-bearing
    per RESEARCH Pitfall 4 — set-order non-determinism caused a real Phase
    145.1 byte-diff); any failure -> None. OPSIN runs on STDIN via
    _opsin_to_smi (no shell, OPSIN_TIMEOUT=10.0).
    """
    if not token:
        return None
    token_smi = _opsin_to_smi(token)
    if not token_smi:
        return None
    token_mol = Chem.MolFromSmiles(token_smi)
    if token_mol is None:
        return None
    try:
        matches = input_mol.GetSubstructMatches(token_mol)
    except Exception as e:
        logger.debug("Substructure match failed: %s", e)
        return None
    if not matches:
        return None
    if len(matches) == 1:
        return set(matches[0])
    # Ambiguous: canonical-rank tiebreak (deterministic)
    try:
        canon_ranks = list(Chem.CanonicalRankAtoms(input_mol, breakTies=True))
        best = min(matches, key=lambda m: tuple(sorted(canon_ranks[a] for a in m)))
        return set(best)
    except Exception as e:
        logger.debug("Canonical rank tiebreak failed: %s", e)
        # Deterministic fallback: first match
        return set(matches[0])


def _extract_reference_parent_atoms(
    ref_name: str, input_mol: Any
) -> Optional[Set[int]]:
    """Extract reference parent atom indices via OPSIN round-trip.

    Pipeline:
      1. Extract parent-hydride token from ref_name (heuristic)
      2. match_token_atoms_in_mol: OPSIN-parse the token, RDKit-substructure
         match into input_mol, canonical-rank tiebreak on ambiguity.

    Returns set of atom indices in input_mol comprising the reference
    parent. Returns None on any pipeline failure. Behavior is UNCHANGED from
    the pre-Phase-166 inline implementation (now delegated to the reusable
    match_token_atoms_in_mol helper so the per-node scorer shares one code
    path).
    """
    parent_token = _extract_parent_token(ref_name)
    if not parent_token:
        return None
    return match_token_atoms_in_mol(parent_token, input_mol)


# ---------------------------------------------------------------------------
# ParentCorrectnessScorer
# ---------------------------------------------------------------------------

class ParentCorrectnessScorer:
    """Computes the parent_correctness factor for a candidate name.

    Returns:
      1.0  if candidate's parent atoms match OPSIN-extracted reference parent
      0.0  if mismatch
      0.5  on no-decision (any failure mode -- see module docstring)

    Production callers (no _pc_context.reference_name set) immediately
    return 0.5 with zero OPSIN cost. Benchmark runners that have called
    set_reference_name() pay one OPSIN parse per scored candidate.
    """

    @staticmethod
    def score(candidate: CandidateName, mol: Any) -> float:
        """Score a single candidate against the thread-local reference."""
        ref_name = getattr(_pc_context, 'reference_name', None)
        if ref_name is None:
            # Production path -- no reference; no signal possible
            return 0.5
        if candidate.parent_atom_indices is None:
            # Handler didn't report parent atoms (most direct-return)
            return 0.5
        # Extract reference parent atoms via OPSIN round-trip
        ref_parent_atoms = _extract_reference_parent_atoms(ref_name, mol)
        if ref_parent_atoms is None:
            # Pipeline failed (OPSIN unparseable, no substructure match, etc.)
            return 0.5
        # Compare sets
        if set(candidate.parent_atom_indices) == ref_parent_atoms:
            return 1.0
        return 0.0
