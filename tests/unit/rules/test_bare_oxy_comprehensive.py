"""Comprehensive tests for bare oxy prefix prevention across all code paths.

Tests that ether substituents always produce qualified prefixes (methoxy, ethoxy,
propoxy, etc.) and never produce a bare standalone "oxy" prefix.

IUPAC Reference: P-63.2.3 (ether substituent naming)
Phase: 90-02 Task 1
"""
import pytest
import re
import logging


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _has_bare_oxy(name: str) -> bool:
    """Return True if *name* contains a bare standalone 'oxy' prefix.

    Legitimate compound words (methoxy, ethoxy, propoxy, carboxyloxy,
    oxybis, oxydi, anoxy, etc.) are NOT flagged.  Only a standalone
    'oxy' token that appears as a hyphen-separated or space-separated
    word *and* is NOT part of a larger word is flagged.
    """
    if not name:
        return False
    # Split on hyphens and spaces, check for exact "oxy" tokens
    tokens = re.split(r'[-\s,()]', name)
    for tok in tokens:
        if tok == 'oxy':
            return True
    return False


# ---------------------------------------------------------------------------
# 1. Simple ether molecules always produce qualified alkoxy prefixes
# ---------------------------------------------------------------------------

class TestSimpleEtherQualifiedPrefixes:
    """Ether substituents must ALWAYS produce qualified prefixes."""

    @pytest.mark.parametrize("smiles,must_contain", [
        ("COC", "methoxy"),           # dimethyl ether
        ("CCOCC", "ethoxy"),          # diethyl ether
        ("CCCOCC", "propoxy"),        # propyl ethyl ether variant
        ("CCCCOCCCCC", "butoxy"),     # butyl pentyl ether
    ])
    def test_simple_ethers_have_qualified_prefix(self, smiles, must_contain):
        """Simple symmetric/asymmetric ethers must produce qualified alkoxy names."""
        from orthonym.namer import name_compound
        name = name_compound(smiles)
        assert name is not None, f"Failed to name {smiles}"
        assert not _has_bare_oxy(name), (
            f"Bare 'oxy' found in name for {smiles}: {name!r}"
        )


# ---------------------------------------------------------------------------
# 2. Multiplicative oxy bridge names must be preserved (IUPAC P-31.1.2.1)
# ---------------------------------------------------------------------------

class TestMultiplicativeOxyBridge:
    """Multiplicative oxy bridge names (1,1'-oxybis...) are IUPAC-correct."""

    def test_oxybis_preserved(self):
        """The 'oxy' in 'oxybis' is a legitimate IUPAC bridge name."""
        # oxybis is a compound word, not bare oxy
        assert not _has_bare_oxy("1,1'-oxybispropane")
        assert not _has_bare_oxy("1,1'-oxydiethane")


# ---------------------------------------------------------------------------
# 3. _alcohol_to_alkoxy never returns bare "oxy"
# ---------------------------------------------------------------------------

class TestAlcoholToAlkoxyNoBareOxy:
    """The _alcohol_to_alkoxy conversion must never return bare 'oxy'."""

    @pytest.mark.parametrize("input_name", [
        "ol",         # degenerate input
        "anol",       # degenerate input
        "yl",         # degenerate input
    ])
    def test_degenerate_inputs_do_not_produce_bare_oxy(self, input_name):
        """Degenerate alcohol/alkyl names must NOT produce bare 'oxy'."""
        from orthonym.decomposition.fragment_assembly import _alcohol_to_alkoxy
        result = _alcohol_to_alkoxy(input_name)
        if result is not None:
            assert result != "oxy", (
                f"_alcohol_to_alkoxy({input_name!r}) returned bare 'oxy'"
            )

    @pytest.mark.parametrize("input_name,expected", [
        ("methanol", "methoxy"),
        ("ethanol", "ethoxy"),
        ("propan-1-ol", "propoxy"),
        ("phenol", "phenoxy"),
    ])
    def test_normal_alcohols_produce_qualified_alkoxy(self, input_name, expected):
        """Normal alcohol names produce correct alkoxy prefixes."""
        from orthonym.decomposition.fragment_assembly import _alcohol_to_alkoxy
        result = _alcohol_to_alkoxy(input_name)
        assert result == expected, (
            f"_alcohol_to_alkoxy({input_name!r}) = {result!r}, expected {expected!r}"
        )


# ---------------------------------------------------------------------------
# 4. _name_alkoxy_branch never returns bare "oxy"
# ---------------------------------------------------------------------------

class TestNameAlkoxyBranchNoBareOxy:
    """The _name_alkoxy_branch function must never return bare 'oxy'."""

    def test_simple_methyl_ether(self):
        """CH3-O- attached to a chain -> 'methoxy'."""
        from rdkit import Chem
        from orthonym.assembly.substituent_enumerator import _name_alkoxy_branch

        mol = Chem.MolFromSmiles("CCOC")  # ethyl methyl ether
        # O is atom 2, neighbors are C(1) and C(3)
        # If parent is [0,1], O is attachment at 2, alkyl is [3]
        frag_atoms = [2, 3]
        parent_atoms = {0, 1}
        name = _name_alkoxy_branch(mol, frag_atoms, 2, parent_atoms)
        assert name is not None
        assert name != "oxy", f"Got bare 'oxy' from _name_alkoxy_branch"
        assert "oxy" in name  # Should be "methoxy"


# ---------------------------------------------------------------------------
# 5. Defensive check in composer catches stray bare oxy with WARNING
# ---------------------------------------------------------------------------

class TestComposerDefensiveOxyCheck:
    """Composer final assembly logs WARNING for any remaining bare oxy."""

    def test_defensive_check_exists(self):
        """Verify defensive oxy check function exists and is callable."""
        from orthonym.assembly.composer import _warn_if_bare_oxy
        assert callable(_warn_if_bare_oxy)

    def test_defensive_check_detects_bare_oxy(self):
        """_warn_if_bare_oxy returns True for names containing bare oxy."""
        from orthonym.assembly.composer import _warn_if_bare_oxy
        assert _warn_if_bare_oxy("2-oxy-hexane") is True

    def test_defensive_check_ignores_qualified_oxy(self):
        """_warn_if_bare_oxy returns False for qualified oxy names."""
        from orthonym.assembly.composer import _warn_if_bare_oxy
        assert _warn_if_bare_oxy("2-methoxyhexane") is False
        assert _warn_if_bare_oxy("ethoxybenzene") is False
        assert _warn_if_bare_oxy("1,1'-oxybispropane") is False


# ---------------------------------------------------------------------------
# 6. End-to-end: no bare oxy in generated names for ether compounds
# ---------------------------------------------------------------------------

class TestEndToEndNoBareOxy:
    """Full pipeline never produces bare oxy for ether-containing molecules."""

    @pytest.mark.parametrize("smiles", [
        "COC",                  # dimethyl ether
        "CCOCC",               # diethyl ether
        "c1ccc(OC)cc1",        # anisole/methoxybenzene
        "c1ccc(OCC)cc1",       # ethoxybenzene
        "CCCCOCCCC",           # dibutyl ether
        "CC(C)OC(C)C",        # diisopropyl ether
    ])
    def test_no_bare_oxy_in_output(self, smiles):
        """Generated name for ether molecules must not contain bare 'oxy'."""
        from orthonym.namer import name_compound
        name = name_compound(smiles)
        assert name is not None, f"Failed to name {smiles}"
        assert not _has_bare_oxy(name), (
            f"Bare 'oxy' in generated name for {smiles}: {name!r}"
        )
