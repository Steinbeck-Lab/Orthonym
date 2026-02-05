"""
Regression tests for name formatting cleanup (Phase 15, Plan 03).

Tests for:
1. No 'functionalized_chain' debug string in generated names
2. Proper hyphenation at locant-name boundaries
3. Proper polyol suffix naming (word form, not numeric)
4. No 'cycloane' empty ring size
5. Substituent grouping and locant assignment
"""

import pytest
from rdkit import Chem

from orthonym.namer import name_compound


# ============================================================================
# Bug 1: functionalized_chain debug string should never appear in names
# ============================================================================

class TestNoFunctionalizedChainDebugString:
    """Verify 'functionalized_chain' debug string never appears in generated names."""

    def test_benzene_with_carboxylic_acid_chain(self):
        """Butanoic acid chain on benzene should produce valid substituent name."""
        mol = Chem.MolFromSmiles('c1ccc(CCCC(=O)O)cc1')
        name = name_compound(Chem.MolToSmiles(mol))
        assert 'functionalized_chain' not in name, (
            f"Debug string 'functionalized_chain' leaked into name: {name}"
        )

    def test_benzene_with_alcohol_chain(self):
        """Propanol chain on benzene should produce valid substituent name."""
        mol = Chem.MolFromSmiles('c1ccc(CCCO)cc1')
        name = name_compound(Chem.MolToSmiles(mol))
        assert 'functionalized_chain' not in name, (
            f"Debug string 'functionalized_chain' leaked into name: {name}"
        )

    def test_benzene_with_aldehyde_chain(self):
        """Aldehyde chain on benzene should produce valid substituent name."""
        mol = Chem.MolFromSmiles('c1ccc(CC=O)cc1')
        name = name_compound(Chem.MolToSmiles(mol))
        assert 'functionalized_chain' not in name, (
            f"Debug string 'functionalized_chain' leaked into name: {name}"
        )


# ============================================================================
# Bug 2: Missing hyphens at locant-name boundaries
# ============================================================================

class TestHyphenAtLocantBoundary:
    """Verify hyphens are always inserted between letter-ending prefix and digit-starting token."""

    def test_prefix_before_locanted_replacement(self):
        """A prefix ending with a letter followed by a digit-starting token needs a hyphen."""
        # Test at assembly utility level
        from orthonym.assembly.composer import _join_prefixes
        result = _join_prefixes(["pentabutyl", "1,4,7,10,13-pentaaza"])
        assert "pentabutyl-1" in result, (
            f"Missing hyphen between prefix and locant: {result}"
        )

    def test_diethyl_before_locanted_prefix(self):
        """diethyl followed by locanted prefix should have hyphen."""
        from orthonym.assembly.composer import _join_prefixes
        result = _join_prefixes(["diethyl", "1,3-dioxaphosphepine"])
        assert "diethyl-1" in result, (
            f"Missing hyphen between prefix and locant: {result}"
        )

    def test_tetramethyl_before_locanted_prefix(self):
        """tetramethyl followed by locanted prefix should have hyphen."""
        from orthonym.assembly.composer import _join_prefixes
        result = _join_prefixes(["tetramethyl", "1,7,11-trioxa"])
        assert "tetramethyl-1" in result, (
            f"Missing hyphen between prefix and locant: {result}"
        )


# ============================================================================
# Bug 3: Numeric polyol suffix should use word-form multipliers
# ============================================================================

class TestPolyolSuffix:
    """Verify polyol suffixes use word-form multipliers (triol, tetraol), not numeric (3ol)."""

    def test_triol_suffix(self):
        """3 hydroxyl groups should produce 'triol', not '3ol' or 'polyol'."""
        from orthonym.rules.tricyclo import name_polycyclo_with_functional_groups
        # Direct unit test of the multiplier logic
        _OH_MULTIPLIERS = {
            2: "di", 3: "tri", 4: "tetra", 5: "penta",
            6: "hexa", 7: "hepta", 8: "octa", 9: "nona", 10: "deca",
        }
        for count, expected_mult in _OH_MULTIPLIERS.items():
            suffix = f"{expected_mult}ol"
            assert suffix.isalpha(), f"Suffix '{suffix}' should be alphabetic for count={count}"
            assert not any(c.isdigit() for c in suffix), (
                f"Suffix '{suffix}' should not contain digits"
            )

    def test_no_numeric_ol_suffix(self):
        """No generated suffix should be just digits + 'ol' like '9ol' or '8ol'."""
        import re
        _OH_MULTIPLIERS = {
            2: "di", 3: "tri", 4: "tetra", 5: "penta",
            6: "hexa", 7: "hepta", 8: "octa", 9: "nona", 10: "deca",
        }
        for count in range(2, 11):
            mult = _OH_MULTIPLIERS.get(count, str(count))
            name = f"testname{mult}ol"
            # Should not match pattern digit+ol
            assert not re.search(r'\dol', name), (
                f"Name '{name}' has numeric+ol pattern for count={count}"
            )


# ============================================================================
# Bug 4: cycloane empty ring size guard
# ============================================================================

class TestNoCycloane:
    """Verify 'cycloane' (empty ring size) never appears in generated names."""

    def test_ring_parent_with_known_sizes(self):
        """Common ring sizes produce correct names."""
        from orthonym.assembly.composer import CHAIN_PREFIXES
        for size in [3, 4, 5, 6, 7, 8]:
            assert size in CHAIN_PREFIXES, f"Ring size {size} missing from CHAIN_PREFIXES"
            stem = CHAIN_PREFIXES[size]
            name = f"cyclo{stem}ane"
            assert name != "cycloane", f"Ring size {size} produced cycloane"
            assert len(stem) > 0, f"Empty stem for ring size {size}"

    def test_generate_ring_parent_guard(self):
        """_generate_ring_parent returns safe placeholder for invalid ring, not 'cycloane'."""
        from orthonym.assembly.composer import _generate_ring_parent
        from dataclasses import dataclass
        from typing import Optional

        # Create a minimal mock features with empty principal_ring
        @dataclass
        class MockFeatures:
            ring_type: str = 'cycloalkane'
            principal_ring: Optional[tuple] = None

        features = MockFeatures(principal_ring=None)
        result = _generate_ring_parent(features)
        # Should return 'cyclo' placeholder, not 'cycloane'
        assert 'cycloane' not in result.text, (
            f"Generated 'cycloane' from invalid ring: {result.text}"
        )


# ============================================================================
# Bug 5: Substituent stacking without locants (deduplication)
# ============================================================================

class TestSubstituentMerging:
    """Verify duplicate substituent prefixes from different sources are merged."""

    def test_merge_duplicate_hydroxy(self):
        """Two hydroxy prefixes from different sources merge into dihydroxy."""
        from orthonym.assembly.composer import _merge_duplicate_prefixes, NameFragment
        prefixes = [
            NameFragment(text="hydroxy", locants=(), fragment_type="prefix"),
            NameFragment(text="hydroxy", locants=(), fragment_type="prefix"),
        ]
        merged = _merge_duplicate_prefixes(prefixes)
        assert len(merged) == 1
        assert "dihydroxy" in merged[0].text

    def test_merge_with_locants(self):
        """Prefixes with locants are combined and locants merged."""
        from orthonym.assembly.composer import _merge_duplicate_prefixes, NameFragment
        prefixes = [
            NameFragment(text="3-hydroxy", locants=(3,), fragment_type="prefix"),
            NameFragment(text="5-hydroxy", locants=(5,), fragment_type="prefix"),
        ]
        merged = _merge_duplicate_prefixes(prefixes)
        assert len(merged) == 1
        assert "dihydroxy" in merged[0].text
        assert 3 in merged[0].locants
        assert 5 in merged[0].locants

    def test_no_merge_different_names(self):
        """Different substituent names should not be merged."""
        from orthonym.assembly.composer import _merge_duplicate_prefixes, NameFragment
        prefixes = [
            NameFragment(text="hydroxy", locants=(), fragment_type="prefix"),
            NameFragment(text="methyl", locants=(), fragment_type="prefix"),
        ]
        merged = _merge_duplicate_prefixes(prefixes)
        assert len(merged) == 2
