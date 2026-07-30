"""
OPSIN round-trip diagnostic tool.

FMT-06: Provides opsin_parse(), opsin_roundtrip_check(), and
opsin_parse_both_versions() for validating generated IUPAC names
against the OPSIN parser.

Uses subprocess to invoke the OPSIN CLI JAR. Requires Java runtime
and OPSIN JAR file(s) in the project root.
"""

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional


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
            ["java", "-version"],
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
                ["java", "-jar", jar_path, "-osmi"],
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

    Per D-19: Test against both versions to identify version-specific
    parsing differences.

    Args:
        name: IUPAC name to parse.

    Returns:
        Dict with keys:
            - v28: str or None -- SMILES from OPSIN 2.8.0
            - v29: str or None -- SMILES from OPSIN 2.9.0
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
