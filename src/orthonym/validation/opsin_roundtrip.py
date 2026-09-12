"""
OPSIN round-trip diagnostic tool.

: Provides opsin_parse, opsin_roundtrip_check, and
opsin_parse_both_versions for validating generated IUPAC names
against the OPSIN parser.

Uses subprocess to invoke the OPSIN CLI JAR. Requires Java runtime
and OPSIN JAR file(s) in the project root.
"""

import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from orthonym.jvm_flags import JVM_HYGIENE_FLAGS
from ..jvm_bridge import REJECTED, opsin_extended_smiles  # Lever A: module-level so tests can monkeypatch

# Project root: 4 levels up from this file
# (src/orthonym/validation/opsin_roundtrip.py -> project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def _find_opsin_jar(version: str = "2.9.0") -> Optional[str]:
    """Find OPSIN JAR file for the specified version.

    Args:
        version: OPSIN version string (e.g., "2.9.0", "2.8.0").

    Returns:
        Path to JAR file, or None if not found.
    """
    jar_name = f"opsin-cli-{version}-jar-with-dependencies.jar"
    jar_path = PROJECT_ROOT / jar_name
    if jar_path.exists():
        return str(jar_path)
    return None


@lru_cache(maxsize=1)
def _java_available() -> bool:
    """Check if Java runtime is available.

    ⚠ **CACHED.** Measured 2026-07-30: this probe re-spawned a JVM on every call
    purely to re-answer "is Java installed?". A Java runtime cannot appear or
    disappear inside one process, so caching is **provably output-neutral** — no name
    can change. ``maxsize=1`` matches the idiom in ``name_morphemes.py``; the sibling
    probe in ``perception/centres_bridge.py`` is cached identically and carries the
    full measurement.
    """
    try:
        proc = subprocess.run(
            ["java", *JVM_HYGIENE_FLAGS, "-version"],
            capture_output=True, text=True, timeout=5,
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def opsin_parse(name: str, jar_version: str = "2.9.0") -> Optional[str]:
    """Parse an IUPAC name with OPSIN CLI JAR.

    Args:
        name: IUPAC name to parse.
        jar_version: OPSIN version to use (default "2.9.0").

    Returns:
        SMILES string if OPSIN successfully parsed the name,
        or None if parsing failed.
    """
    if not name or not name.strip():
        return None

    jar_path = _find_opsin_jar(jar_version)
    if jar_path is None:
        return None

    # PERF: prefer the ONE in-process JVM (jvm_bridge, JPype) over a ~130 ms
    # process launch per name. It returns EXACTLY the bytes this `java -jar`
    # invocation would have written to stdout, so the parsing below is shared
    # verbatim between the two sources and cannot drift. `served` is False
    # whenever the in-process path cannot serve the call (jpype absent, jar
    # unresolvable, a different OPSIN version requested, an embedded newline,
    # a Java-side error) -- we then fall through to the subprocess exactly as
    # before, so correctness never depends on the optimization.
    stdout_text: Optional[str] = None
    served = False
    try:
        from ..jvm_bridge import opsin_stdout
        stdout_text, served = opsin_stdout(name, allow_radicals=False,
                                           jar_path=jar_path)
    except ImportError:  # pragma: no cover - jvm_bridge always present
        served = False

    if not served:
        try:
            result = subprocess.run(
                ["java", *JVM_HYGIENE_FLAGS, "-jar", jar_path, "-osmi"],
                input=name,
                capture_output=True,
                text=True,
                timeout=15,
            )
            stdout_text = result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return None

    lines = (stdout_text or "").strip().split("\n")
    out = lines[-1].strip() if lines else ""

    if not out:
        return None
    if "could not be interpreted" in out.lower():
        return None
    if "unsure of the meaning" in out.lower():
        return None

    return out


_AV_LINE_RE = re.compile(r"^(\S+)\s+\|\$_AV:(.*)\$\|\s*$")
_AV_TOKEN_RE = re.compile(r"^(\d+)('*)([a-z]?)$")

# A locant token as OPSIN's ``$_AV`` field spells it, re-typed for the stereo
# map format the injector accepts (int / (int, "'") tuple / '<int><letter>'
# string / (int, "'", 'a') tuple). See rules/stereochemistry.py
# _render_locant_token + _composite_locant_sort_key.
Locant = Union[int, str, Tuple]


def _parse_av_token(tok: str) -> Optional[Locant]:
    """Convert one OPSIN ``$_AV`` locant token to the stereo-map format.

    ``'1'`` -> 1; ``"1'"`` -> (1, "'"); ``'7a'`` -> '7a';
    ``"3'a"`` -> (3, "'", 'a'). Returns None for anything unrecognised
    (a compound/bridge locant OPSIN may emit) — the caller drops it, so a
    stereocentre with an unmappable locant loses its descriptor (missing beats
    wrong), never gets a fabricated one."""
    m = _AV_TOKEN_RE.match(tok)
    if not m:
        return None
    n = int(m.group(1))
    primes, letter = m.group(2), m.group(3)
    if not primes and not letter:
        return n
    if primes and not letter:
        return (n, primes)
    if letter and not primes:
        return f"{n}{letter}"
    return (n, primes, letter)


def _extended_smiles(name: str, jar_version: str = "2.9.0") -> Optional[str]:
    """OPSIN ``-o extendedsmi`` output line for *name* (in-process fast path;
    subprocess fallback only when the in-process path cannot serve the call), or
    None if OPSIN rejected the name / was unavailable. Memoised per name inside the
    naming scope (Lever A, 2026-09-12): the stereo re-anchor asks for the same name
    several times per molecule, and a definitive rejection used to start a fresh
    3.5 s Java process that printed the same empty line."""
    from ..assembly.memo import cache_or_compute
    return cache_or_compute("opsin_extended_smiles", (name, jar_version),
                            lambda: _extended_smiles_uncached(name, jar_version))


def _extended_smiles_uncached(name: str, jar_version: str = "2.9.0") -> Optional[str]:
    jar_path = _find_opsin_jar(jar_version)
    if jar_path is None:
        return None
    text, served = opsin_extended_smiles(name, jar_path=jar_path)   # module-level import (monkeypatchable)
    if served:
        return None if text == REJECTED else text
    try:
        result = subprocess.run(
            ["java", *JVM_HYGIENE_FLAGS, "-jar", jar_path, "-o", "extendedsmi"],
            input=name, capture_output=True, text=True, timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
    lines = (result.stdout or "").strip().split("\n")
    out = lines[-1].strip() if lines else ""
    if not out or "could not be interpreted" in out.lower():
        return None
    return out


def opsin_atom_locant_map(
    name: str, mol, jar_version: str = "2.9.0",
) -> Optional[Dict[int, Locant]]:
    """Re-anchor an IUPAC *name*'s numbering onto *mol*'s atoms via OPSIN.

    Parses *name* with ``-o extendedsmi`` to obtain OPSIN's OWN per-atom locants
    (the authoritative numbering the name will be read back with), then maps
    those atoms onto *mol* by constitutional (stereo-insensitive) subgraph
    isomorphism. Returns ``{mol_atom_idx: locant_token}`` (tokens in the
    injector's accepted format) or None when it cannot be built (OPSIN
    unavailable/rejecting, atom-count/parse mismatch, or no isomorphism).

    This is the ``re-anchor + audit`` primitive (the contributor guide a project rule): the
    map is derived FROM the name+structure, never trusted from a builder's
    internal numbering, and the caller RT-verifies any name decorated with it.
    """
    if mol is None:
        return None
    ext = _extended_smiles(name, jar_version=jar_version)
    if not ext:
        return None
    m = _AV_LINE_RE.match(ext.strip())
    if not m:
        return None
    smi_part, av = m.group(1), m.group(2)
    tokens = av.split(";")
    try:
        from rdkit import Chem
        omol = Chem.MolFromSmiles(smi_part)
    except ImportError:  # pragma: no cover
        return None
    if omol is None or omol.GetNumAtoms() != len(tokens):
        return None
    # Constitutional isomorphism onto the input graph (chirality ignored — the
    # locants are constitutional; the stereo layer is what we are re-deriving).
    #
    # DETERMINISM (a review -P2 F2 / a project rule): a symmetric molecule has more
    # than one automorphism, and ``GetSubstructMatch`` returns an ARBITRARY one
    # whose choice depends on the input atom order — so two SMILES spellings of,
    # e.g., meso-butane-2,3-diol would map the OPSIN locants onto different atoms
    # and yield different R/S descriptors. Both round-trip (0-wrong holds) but the
    # output is not deterministic. Enumerate the automorphisms and pick the one
    # canonical under ``CanonicalRankAtoms`` (an atom-order-INVARIANT total order),
    # so the same molecule always yields the same locant map regardless of how its
    # SMILES was written.
    matches = mol.GetSubstructMatches(
        omol, useChirality=False, uniquify=False, maxMatches=100_000)
    matches = [m for m in matches if len(m) == omol.GetNumAtoms()]
    if not matches:
        return None
    if len(matches) == 1:
        match = matches[0]
    else:
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
        match = min(matches, key=lambda mm: tuple(ranks[i] for i in mm))
    result: Dict[int, Locant] = {}
    for opsin_idx, mol_idx in enumerate(match):
        loc = _parse_av_token(tokens[opsin_idx].strip())
        if loc is not None:
            result[mol_idx] = loc
    return result or None


def opsin_roundtrip_check(
    smiles: str, name: str, jar_version: str = "2.9.0"
) -> dict:
    """Full round-trip validation: name -> OPSIN -> SMILES -> InChI comparison.

    Args:
        smiles: Original SMILES string.
        name: Generated IUPAC name to validate.
        jar_version: OPSIN version to use.

    Returns:
        Dict with keys:
            - passed: bool -- True if InChI matches
            - opsin_smiles: str or None -- SMILES from OPSIN
            - inchi_match: bool -- whether InChI strings match
            - error: str or None -- error description if failed
    """
    result = {
        "passed": False,
        "opsin_smiles": None,
        "inchi_match": False,
        "error": None,
    }

    # 1. Parse name with OPSIN
    opsin_smiles = opsin_parse(name, jar_version=jar_version)
    if opsin_smiles is None:
        result["error"] = "opsin_parse_failed"
        return result

    result["opsin_smiles"] = opsin_smiles

    # 2. Convert both SMILES to InChI using RDKit
    try:
        from rdkit import Chem
        from rdkit.Chem.inchi import MolToInchi

        # Original SMILES -> InChI
        mol_orig = Chem.MolFromSmiles(smiles)
        if mol_orig is None:
            result["error"] = "invalid_original_smiles"
            return result
        inchi_orig = MolToInchi(mol_orig)

        # OPSIN SMILES -> InChI
        mol_opsin = Chem.MolFromSmiles(opsin_smiles)
        if mol_opsin is None:
            result["error"] = "invalid_opsin_smiles"
            return result
        inchi_opsin = MolToInchi(mol_opsin)

        # 3. Compare InChI strings
        if inchi_orig and inchi_opsin and inchi_orig == inchi_opsin:
            result["inchi_match"] = True
            result["passed"] = True
        else:
            result["error"] = "inchi_mismatch"

    except ImportError:
        result["error"] = "rdkit_not_available"
    except Exception as e:
        result["error"] = f"comparison_error: {str(e)}"

    return result


def opsin_parse_both_versions(name: str) -> dict:
    """Cross-validate name against OPSIN 2.8.0 and 2.9.0.

    Per: Test against both versions to identify version-specific
    parsing differences.

    Args:
        name: IUPAC name to parse.

    Returns:
        Dict with keys:
            -: str or None -- SMILES from OPSIN 2.8.0
            -: str or None -- SMILES from OPSIN 2.9.0
            - agree: bool -- whether both versions produce the same result
    """
    v28 = opsin_parse(name, jar_version="2.8.0")
    v29 = opsin_parse(name, jar_version="2.9.0")

    # Both None = agree (both failed); both same string = agree
    agree = v28 == v29

    return {
        "v28": v28,
        "v29": v29,
        "agree": agree,
    }
