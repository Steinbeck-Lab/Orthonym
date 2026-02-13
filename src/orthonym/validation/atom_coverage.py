"""
Atom coverage validator for Orthonym.

Validates how completely a generated IUPAC name describes the input molecule
by parsing the name back to a structure and comparing heavy atom counts.

This is a diagnostic tool -- it reports coverage scores but never blocks
naming output. Phase 47 adds hard gating once parent selection is improved.

Two validation approaches:
  1. Parse-back (OPSIN): Parse name via OPSIN JAR, compare heavy atom counts.
  2. Fallback: If OPSIN unavailable or parse fails, return 'unavailable' result.
"""

import glob as glob_mod
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional, Set

from rdkit import Chem


@dataclass
class CoverageResult:
    """Result of atom coverage validation.

    Attributes:
        total_heavy_atoms: Heavy atom count of the input molecule.
        claimed_atoms: Number of heavy atoms accounted for in the name.
        unclaimed_atoms: Number of heavy atoms NOT accounted for.
        coverage_ratio: claimed / total (0.0 to 1.0).
        is_complete: True when coverage_ratio >= 0.80.
        method: Validation method used ('parse_back', 'substructure', or 'unavailable').
        claimed_atom_indices: Indices of atoms accounted for (may be empty
            if only atom-count comparison was used).
        unclaimed_atom_indices: Indices of atoms NOT accounted for (may be
            empty if only atom-count comparison was used).
    """

    total_heavy_atoms: int
    claimed_atoms: int
    unclaimed_atoms: int
    coverage_ratio: float
    is_complete: bool
    method: str
    claimed_atom_indices: Set[int] = field(default_factory=set)
    unclaimed_atom_indices: Set[int] = field(default_factory=set)


def find_opsin_jar() -> Optional[str]:
    """Find OPSIN JAR file in project root or standard locations.

    Uses the same search patterns as ``.
    """
    # Determine project root (go up from src/orthonym/validation/)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))

    search_patterns = [
        os.path.join(project_root, "opsin-cli-*-jar-with-dependencies.jar"),
        os.path.join(project_root, "opsin-cli-*.jar"),
        os.path.join(project_root, "opsin.jar"),
        os.path.join(
            project_root,
            "opsin",
            "opsin-cli",
            "target",
            "opsin-cli-*-jar-with-dependencies.jar",
        ),
    ]

    for pattern in search_patterns:
        matches = glob_mod.glob(pattern)
        if matches:
            return matches[0]
    return None


def _parse_name_with_opsin(name: str, opsin_jar: str) -> Optional[str]:
    """Parse a single IUPAC name to SMILES using OPSIN JAR.

    Mirrors the batch approach from ``
    (stdin/stdout mode with a single name written to a temp file).

    Args:
        name: IUPAC name to parse.
        opsin_jar: Path to OPSIN CLI JAR file.

    Returns:
        Canonical SMILES string if parse succeeds, else ``None``.
    """
    if not name or not opsin_jar:
        return None

    # Write single name to temp file
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(name + "\n")
            temp_input = f.name
    except OSError:
        return None

    try:
        cmd = ["java", "-jar", opsin_jar, "-osmi", temp_input]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30.0,
        )

        stdout_lines = proc.stdout.strip().split("\n") if proc.stdout else []
        if stdout_lines and stdout_lines[0].strip():
            smiles = stdout_lines[0].strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
        return None

    except (subprocess.TimeoutExpired, OSError, Exception):
        return None
    finally:
        try:
            os.unlink(temp_input)
        except OSError:
            pass


def validate_atom_coverage(
    mol,
    name: str,
    opsin_jar: Optional[str] = None,
) -> CoverageResult:
    """Validate how completely *name* describes *mol* by atom coverage.

    Uses a parse-back approach: parse *name* back to a molecule (via OPSIN)
    and compare heavy atom counts.

    Args:
        mol: RDKit Mol object for the input molecule.
        name: Generated IUPAC name to validate.
        opsin_jar: Path to OPSIN JAR. If ``None``, attempts auto-discovery.
            If OPSIN is unavailable, returns a result with method='unavailable'.

    Returns:
        A :class:`CoverageResult` with coverage metrics.
    """
    total_heavy = mol.GetNumHeavyAtoms()

    if total_heavy == 0:
        return CoverageResult(
            total_heavy_atoms=0,
            claimed_atoms=0,
            unclaimed_atoms=0,
            coverage_ratio=1.0,
            is_complete=True,
            method="trivial",
        )

    # Try auto-discovery if no JAR path given
    if opsin_jar is None:
        opsin_jar = find_opsin_jar()

    # Check Java availability
    if opsin_jar is not None:
        import shutil

        if not shutil.which("java"):
            opsin_jar = None

    # Parse-back approach
    if opsin_jar is not None and name:
        parsed_smiles = _parse_name_with_opsin(name, opsin_jar)
        if parsed_smiles is not None:
            parsed_mol = Chem.MolFromSmiles(parsed_smiles)
            if parsed_mol is not None:
                parsed_heavy = parsed_mol.GetNumHeavyAtoms()
                ratio = min(parsed_heavy / total_heavy, 1.0)
                claimed = min(parsed_heavy, total_heavy)
                unclaimed = total_heavy - claimed

                return CoverageResult(
                    total_heavy_atoms=total_heavy,
                    claimed_atoms=claimed,
                    unclaimed_atoms=unclaimed,
                    coverage_ratio=ratio,
                    is_complete=(ratio >= 0.80),
                    method="parse_back",
                )

    # Fallback: OPSIN unavailable or parse failed
    return CoverageResult(
        total_heavy_atoms=total_heavy,
        claimed_atoms=0,
        unclaimed_atoms=total_heavy,
        coverage_ratio=0.0,
        is_complete=False,
        method="unavailable",
    )
