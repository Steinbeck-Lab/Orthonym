"""
Consistency tests for chain naming data.

Two categories:
1. Static analysis: Verify no module re-introduces local CHAIN_PREFIXES
   or ALKYL_NAMES dicts that should come from canonical sources.
2. IUPAC value verification: Verify 40+ chain prefix values match
   IUPAC 2013 Blue Book compositional rules.
"""

import pathlib
import re

import pytest

from orthonym.data.chain_names import get_chain_prefix

SRC_DIR = pathlib.Path(__file__).parents[3] / "src" / "orthonym"


# ============================================================================
# Static Analysis Tests
# ============================================================================

@pytest.mark.unit
class TestNoLocalChainPrefixDicts:
    """Verify no module defines its own CHAIN_PREFIXES or ALKYL_NAMES dict."""

    def test_no_local_chain_prefix_dicts(self):
        """No Python file outside chain_names.py should define CHAIN_PREFIXES."""
        canonical = SRC_DIR / "data" / "chain_names.py"
        pattern = re.compile(r'^\s*_?CHAIN_PREFIXES\s*=\s*[\{\(]', re.MULTILINE)
        violations = []
        for py_file in SRC_DIR.rglob("*.py"):
            if py_file == canonical:
                continue
            content = py_file.read_text()
            if pattern.search(content):
                violations.append(str(py_file.relative_to(SRC_DIR)))
        assert not violations, (
            f"Local CHAIN_PREFIXES dicts found in: {violations}. "
            f"Use data.chain_names.get_chain_prefix() instead."
        )

    def test_no_local_alkyl_names_dicts(self):
        """No Python file outside naming_utils.py should define ALKYL_NAMES."""
        canonical = SRC_DIR / "assembly" / "naming_utils.py"
        pattern = re.compile(r'^\s*ALKYL_NAMES\s*=\s*[\{\(]', re.MULTILINE)
        violations = []
        for py_file in SRC_DIR.rglob("*.py"):
            if py_file == canonical:
                continue
            content = py_file.read_text()
            if pattern.search(content):
                violations.append(str(py_file.relative_to(SRC_DIR)))
        assert not violations, (
            f"Local ALKYL_NAMES dicts found in: {violations}. "
            f"Use assembly.naming_utils.get_alkyl_name() instead."
        )


# ============================================================================
# IUPAC Value Verification Tests
# ============================================================================

@pytest.mark.unit
class TestIUPACValueVerification:
    """Verify chain_names.py returns correct IUPAC Blue Book values.

    Tests cover:
    - All 20 standard retained prefixes
    - Representative compositional twenties
    - All decade prefixes (30-90)
    - Representative tens+units compositions
    - All pure hundred prefixes (100-900)
    - Representative hundreds compositions
    - All pure thousand prefixes (1000-9000)
    - Representative thousands compositions
    """

    @pytest.mark.parametrize("n,expected", [
        # All 20 standard prefixes (IUPAC Table A6.1)
        (1, "meth"), (2, "eth"), (3, "prop"), (4, "but"), (5, "pent"),
        (6, "hex"), (7, "hept"), (8, "oct"), (9, "non"), (10, "dec"),
        (11, "undec"), (12, "dodec"), (13, "tridec"), (14, "tetradec"),
        (15, "pentadec"), (16, "hexadec"), (17, "heptadec"), (18, "octadec"),
        (19, "nonadec"), (20, "icos"),
        # Representative twenties
        (21, "henicos"), (22, "docos"), (29, "nonacos"),
        # Every decade
        (30, "triacont"), (40, "tetracont"), (50, "pentacont"),
        (60, "hexacont"), (70, "heptacont"), (80, "octacont"), (90, "nonacont"),
        # Representative tens+units compositions
        (32, "dotriacont"), (53, "tripentacont"), (76, "hexaheptacont"),
        (99, "nonanonacont"),
        # Every pure hundred
        (100, "hect"), (200, "dict"), (300, "trict"), (400, "tetract"),
        (500, "pentact"), (600, "hexact"), (700, "heptact"), (800, "octact"),
        (900, "nonact"),
        # Representative hundreds compositions
        (111, "undecahect"), (132, "dotriacontahect"),
        (231, "hentriacontadict"), (486, "hexaoctacontatetract"),
        (999, "nonanonacontanonact"),
        # Pure thousands (IUPAC
        (1000, "kili"), (2000, "dili"), (3000, "trili"),
        (4000, "tetrali"), (5000, "pentali"),
        (6000, "hexali"), (7000, "heptali"), (8000, "octali"),
        (9000, "nonali"),
        # Thousands compositions
        (1001, "henakili"),
        (1010, "decakili"),
        (1100, "hectakili"),
        (9999, "nonanonacontanonactanonali"),
    ])
    def test_chain_prefix_iupac_values(self, n, expected):
        assert get_chain_prefix(n) == expected
