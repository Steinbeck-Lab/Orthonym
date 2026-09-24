"""
Combined OPSIN + PubChem validation logic.

Strategy: Try OPSIN first (fast, local). If OPSIN cannot parse the name,
fall back to PubChem PUG-REST API lookup. A compound passes round-trip
validation if EITHER validator produces an InChI match with the original
molecule.

This dual approach recovers RT credit for correct names that OPSIN's
parser cannot handle (e.g., complex decomposition names, unusual
substituent patterns).
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi

from orthonym.jvm_flags import JVM_HYGIENE_FLAGS

from .pubchem_validator import lookup_name_pubchem


@dataclass
class DualResult:
    """Result of dual OPSIN+PubChem validation for a single compound."""

    opsin_parsed: bool = False
    opsin_rt: bool = False
    pubchem_resolved: bool = False
    pubchem_rt: bool = False
    combined_rt: bool = False
    opsin_smiles: Optional[str] = None
    pubchem_inchi: Optional[str] = None


def _get_inchi(smiles: str) -> Optional[str]:
    """Compute InChI from SMILES. Returns None on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MolToInchi(mol)
    except Exception:
        return None


def _parse_name_with_opsin(
    name: str, opsin_jar: Optional[str] = None
) -> Optional[str]:
    """
    Parse a single IUPAC name to SMILES using OPSIN.

    Args:
        name: IUPAC name to parse.
        opsin_jar: Path to OPSIN JAR file. Auto-detected if None.

    Returns:
        Canonical SMILES string or None if parsing failed.
    """
    if opsin_jar is None:
        opsin_jar = _find_opsin_jar()
    if opsin_jar is None:
        return None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(name + "\n")
            temp_input = f.name

        cmd = ["java", *JVM_HYGIENE_FLAGS, "-jar", opsin_jar, "-osmi", temp_input]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )

        output = proc.stdout.strip()
        if output:
            mol = Chem.MolFromSmiles(output)
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        return None
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
        return None
    finally:
        try:
            os.unlink(temp_input)
        except OSError:
            pass


def _parse_batch_with_opsin(
    names: List[str], opsin_jar: str, timeout: float = 120.0
) -> Dict[str, Optional[str]]:
    """
    Parse multiple IUPAC names to SMILES using OPSIN in batch mode.

    Args:
        names: List of IUPAC names to parse.
        opsin_jar: Path to OPSIN JAR file.
        timeout: Maximum time in seconds for the batch.

    Returns:
        Dict mapping name -> canonical SMILES (or None if failed).
    """
    results: Dict[str, Optional[str]] = {}
    if not names:
        return results

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            for name in names:
                f.write(name + "\n")
            temp_input = f.name

        cmd = ["java", *JVM_HYGIENE_FLAGS, "-jar", opsin_jar, "-osmi", temp_input]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )

        stdout_lines = proc.stdout.split("\n") if proc.stdout else []
        for i, name in enumerate(names):
            if i < len(stdout_lines):
                smiles = stdout_lines[i].strip()
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        results[name] = Chem.MolToSmiles(mol, canonical=True)
                    else:
                        results[name] = None
                else:
                    results[name] = None
            else:
                results[name] = None

    except (subprocess.TimeoutExpired, subprocess.SubprocessError, OSError):
        for name in names:
            if name not in results:
                results[name] = None
    finally:
        try:
            os.unlink(temp_input)
        except OSError:
            pass

    return results


def _find_opsin_jar() -> Optional[str]:
    """Path to the pinned OPSIN jar (``orthonym.jars``), or None without a Java runtime."""
    import shutil

    if not shutil.which("java"):
        return None
    from ..jars import find_jar
    return find_jar("opsin")


def validate_compound(
    smiles: str,
    name: str,
    opsin_jar: Optional[str] = None,
    pubchem_cache: Optional[dict] = None,
    skip_pubchem_api: bool = False,
) -> DualResult:
    """
    Validate a generated IUPAC name using OPSIN (primary) and PubChem (fallback).

    Strategy:
    1. Compute InChI of the original SMILES.
    2. Try OPSIN parse-back: parse name -> compute InChI -> compare.
    3. If OPSIN fails to parse, try PubChem lookup: get InChI -> compare.
    4. combined_rt = opsin_rt OR pubchem_rt.

    Args:
        smiles: Original SMILES string.
        name: Generated IUPAC name to validate.
        opsin_jar: Path to OPSIN JAR file (auto-detected if None).
        pubchem_cache: Mutable dict used as PubChem lookup cache.
        skip_pubchem_api: If True, use only cached PubChem results.

    Returns:
        DualResult with all validation fields populated.
    """
    result = DualResult()

    if pubchem_cache is None:
        pubchem_cache = {}

    # Compute original InChI
    original_inchi = _get_inchi(smiles)
    if original_inchi is None:
        return result

    # --- OPSIN path ---
    opsin_smiles = _parse_name_with_opsin(name, opsin_jar=opsin_jar)
    if opsin_smiles is not None:
        result.opsin_parsed = True
        result.opsin_smiles = opsin_smiles
        opsin_inchi = _get_inchi(opsin_smiles)
        if opsin_inchi and opsin_inchi == original_inchi:
            result.opsin_rt = True

    # --- PubChem path (only if OPSIN did NOT parse) ---
    if not result.opsin_parsed:
        pc_result = lookup_name_pubchem(
            name, pubchem_cache, skip_api=skip_pubchem_api
        )
        if pc_result is not None:
            result.pubchem_resolved = True
            pc_inchi = pc_result.get("inchi", "")
            result.pubchem_inchi = pc_inchi
            if pc_inchi and pc_inchi == original_inchi:
                result.pubchem_rt = True

    # --- Combined result ---
    result.combined_rt = result.opsin_rt or result.pubchem_rt

    return result


def validate_batch(
    compounds: List[Tuple[str, str]],
    opsin_jar: Optional[str] = None,
    pubchem_cache: Optional[dict] = None,
    skip_pubchem_api: bool = False,
) -> List[DualResult]:
    """
    Validate a batch of compounds using batch OPSIN parsing for efficiency.

    Args:
        compounds: List of (smiles, name) tuples.
        opsin_jar: Path to OPSIN JAR file.
        pubchem_cache: Mutable dict for PubChem cache.
        skip_pubchem_api: If True, skip PubChem API calls.

    Returns:
        List of DualResult, one per compound in same order.
    """
    if pubchem_cache is None:
        pubchem_cache = {}

    if opsin_jar is None:
        opsin_jar = _find_opsin_jar()

    # Compute original InChIs
    original_inchis = []
    for smi, _ in compounds:
        original_inchis.append(_get_inchi(smi))

    # Batch OPSIN parse
    names = [name for _, name in compounds]
    opsin_results: Dict[str, Optional[str]] = {}
    if opsin_jar:
        opsin_results = _parse_batch_with_opsin(names, opsin_jar)

    # Build results
    results = []
    for i, (smi, name) in enumerate(compounds):
        dr = DualResult()
        orig_inchi = original_inchis[i]
        if orig_inchi is None:
            results.append(dr)
            continue

        # OPSIN path
        opsin_smi = opsin_results.get(name)
        if opsin_smi is not None:
            dr.opsin_parsed = True
            dr.opsin_smiles = opsin_smi
            opsin_inchi = _get_inchi(opsin_smi)
            if opsin_inchi and opsin_inchi == orig_inchi:
                dr.opsin_rt = True

        # PubChem fallback (only if OPSIN did NOT parse)
        if not dr.opsin_parsed:
            pc_result = lookup_name_pubchem(
                name, pubchem_cache, skip_api=skip_pubchem_api
            )
            if pc_result is not None:
                dr.pubchem_resolved = True
                pc_inchi = pc_result.get("inchi", "")
                dr.pubchem_inchi = pc_inchi
                if pc_inchi and pc_inchi == orig_inchi:
                    dr.pubchem_rt = True

        dr.combined_rt = dr.opsin_rt or dr.pubchem_rt
        results.append(dr)

    return results
