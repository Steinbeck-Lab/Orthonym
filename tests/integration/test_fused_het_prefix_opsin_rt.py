"""
OPSIN round-trip tests for fused heterocycle prefix stems.

Phase 78 Plan 01 — Verifies every generated prefix stem round-trips
through OPSIN when embedded in a test compound name.

Format: "4-({stem}-{locant}-yl)butanoic acid" → OPSIN → valid SMILES
"""

import subprocess
import pytest
from pathlib import Path
from rdkit import Chem

from orthonym.data.fused_heterocycles import (
    FUSED_HETEROCYCLE_DATA,
    FUSED_HETEROCYCLE_PREFIX_STEMS,
    match_fused_heterocycle_core,
)


# === OPSIN Setup ===

PROJECT_ROOT = Path(__file__).parent.parent.parent
OPSIN_JAR = PROJECT_ROOT / "opsin-cli-2.9.0-jar-with-dependencies.jar"


def _opsin_available() -> bool:
    """Check if OPSIN CLI JAR is available and Java is installed."""
    if not OPSIN_JAR.exists():
        return False
    try:
        proc = subprocess.run(
            ["java", "-version"],
            capture_output=True, text=True, timeout=5
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _opsin_parse(name: str) -> str:
    """Parse an IUPAC name through OPSIN and return SMILES (or empty string)."""
    if not name or not OPSIN_JAR.exists():
        return ""
    try:
        proc = subprocess.run(
            ["java", "-jar", str(OPSIN_JAR), "-osmi"],
            input=name, capture_output=True, text=True, timeout=15
        )
        return proc.stdout.strip()
    except (subprocess.TimeoutExpired, Exception):
        return ""


HAS_OPSIN = _opsin_available()
skip_no_opsin = pytest.mark.skipif(
    not HAS_OPSIN, reason="OPSIN JAR not available"
)


def _find_carbon_locant(smiles: str) -> str:
    """Find a valid carbon attachment locant for OPSIN RT testing.

    Picks the first peripheral carbon (not heteroatom, not fusion position)
    from iupac_locants for maximum OPSIN compatibility.
    """
    data = FUSED_HETEROCYCLE_DATA.get(smiles)
    if data is None:
        return "2"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "2"

    locants = data.get('iupac_locants', {})

    # Prefer integer locants on carbon atoms
    for pattern_idx, locant in sorted(locants.items()):
        if isinstance(locant, str):
            # Skip fusion locants (3a, 7a, etc.) and special locants (=O, N6)
            continue
        if pattern_idx < mol.GetNumAtoms():
            atom = mol.GetAtomWithIdx(pattern_idx)
            if atom.GetAtomicNum() == 6:  # Carbon
                return str(locant)

    # Fallback: first integer locant
    for locant in locants.values():
        if isinstance(locant, int):
            return str(locant)

    return "2"


# Ring systems that OPSIN does not recognize (rare fusions not in its vocabulary)
_OPSIN_UNKNOWN_RINGS = {
    'pyrido[3,4-b]pyridazine',  # Rare ring, not in OPSIN arylGroups.xml
}


def _build_test_params():
    """Build parametrize list for OPSIN round-trip tests."""
    params = []
    for smiles, stem in FUSED_HETEROCYCLE_PREFIX_STEMS.items():
        locant = _find_carbon_locant(smiles)
        data = FUSED_HETEROCYCLE_DATA.get(smiles, {})
        name = data.get('name', smiles)
        marks = []
        if name in _OPSIN_UNKNOWN_RINGS:
            marks.append(pytest.mark.xfail(
                reason=f"OPSIN does not recognize {name}", strict=True
            ))
        params.append(pytest.param(
            smiles, stem, locant, name,
            id=name,
            marks=marks,
        ))
    return params


@pytest.mark.integration
class TestPrefixStemOpsinRoundtrip:
    """Verify all fused heterocycle prefix stems round-trip through OPSIN."""

    @skip_no_opsin
    @pytest.mark.parametrize("smiles,stem,locant,het_name", _build_test_params())
    def test_prefix_stem_opsin_roundtrip(self, smiles, stem, locant, het_name):
        """Test that '4-({stem}-{locant}-yl)butanoic acid' parses through OPSIN."""
        # Build test name: 4-(prefix)butanoic acid
        prefix = f"{stem}-{locant}-yl"
        test_name = f"4-({prefix})butanoic acid"

        opsin_result = _opsin_parse(test_name)

        assert opsin_result, (
            f"OPSIN failed to parse '{test_name}' "
            f"(heterocycle: {het_name}, stem: {stem})"
        )

        # Verify OPSIN returned a valid SMILES
        mol = Chem.MolFromSmiles(opsin_result)
        assert mol is not None, (
            f"OPSIN returned invalid SMILES '{opsin_result}' "
            f"for name '{test_name}'"
        )
