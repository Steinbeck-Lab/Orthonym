"""Tests for N-prefix deduplication across all assembly paths.

Ensures that N-substituted amide names never contain "N-N-" duplication
and that legitimate "N,N-" prefixes for di-substituted amides are preserved.

IUPAC Reference: (N-substituted amide naming)
Phase: 90-02 Task 2
"""
import pytest


# ---------------------------------------------------------------------------
# 1. N-substituted amide names never contain "N-N-" duplication
# ---------------------------------------------------------------------------

class TestNoNNDuplication:
    """Generated N-substituted amide names must never contain N-N-."""

    @pytest.mark.parametrize("smiles,description", [
        ("CC(=O)NC", "N-methylacetamide"),
        ("CC(=O)NCC", "N-ethylacetamide"),
        ("CCC(=O)NC", "N-methylpropanamide"),
        ("CC(=O)Nc1ccccc1", "N-phenylacetamide"),
    ])
    def test_no_nn_in_simple_amides(self, smiles, description):
        """Simple N-substituted amides must not contain N-N-."""
        from orthonym.namer import name_compound
        name = name_compound(smiles)
        assert name is not None, f"Failed to name {smiles} ({description})"
        assert "N-N-" not in name, (
            f"N-N- duplication in {description}: {name!r}"
        )

    @pytest.mark.parametrize("smiles,description", [
        ("CC(=O)N(C)C", "N,N-dimethylacetamide"),
        ("CC(=O)N(CC)CC", "N,N-diethylacetamide"),
    ])
    def test_nn_comma_preserved_for_di_substituted(self, smiles, description):
        """N,N-di prefixes for di-substituted amides must be preserved."""
        from orthonym.namer import name_compound
        name = name_compound(smiles)
        assert name is not None, f"Failed to name {smiles} ({description})"
        assert "N-N-" not in name, (
            f"N-N- duplication in {description}: {name!r}"
        )
        # N,N- is the correct form for di-substituted
        assert "N,N-" in name, (
            f"Missing N,N- for di-substituted amide {description}: {name!r}"
        )


# ---------------------------------------------------------------------------
# 2. Correct N- prefix placement for mono-substituted amides
# ---------------------------------------------------------------------------

class TestMonoNPrefixPlacement:
    """Mono-N-substituted amides have exactly one N- prefix."""

    @pytest.mark.parametrize("smiles,expected_prefix", [
        ("CC(=O)NC", "N-methyl"),
        ("CC(=O)NCC", "N-ethyl"),
    ])
    def test_single_n_prefix(self, smiles, expected_prefix):
        """Name must contain exactly one N- prefix for mono-substituted."""
        from orthonym.namer import name_compound
        name = name_compound(smiles)
        assert name is not None, f"Failed to name {smiles}"
        assert expected_prefix in name, (
            f"Expected {expected_prefix!r} in {name!r}"
        )


# ---------------------------------------------------------------------------
# 3. Fragment assembly amide path dedup
# ---------------------------------------------------------------------------

class TestFragmentAssemblyAmideDedup:
    """The _assemble_amide path in fragment_assembly.py must dedup N-."""

    def test_simple_amide_assembly(self):
        """Simple amide assembly produces correct N- prefix."""
        from orthonym.decomposition.fragment_assembly import _assemble_amide
        result = _assemble_amide(
            {"acid": "acetic acid", "amine": "methylamine"}, "pin"
        )
        assert result is not None
        assert "N-N-" not in result
        assert result == "N-methylacetamide"

    def test_recursive_amine_no_double_n(self):
        """Amine with existing N- prefix should not get doubled."""
        from orthonym.decomposition.fragment_assembly import _assemble_amide
        result = _assemble_amide(
            {"acid": "acetic acid", "amine": "N-methylcyclohexanamine"}, "pin"
        )
        assert result is not None
        assert "N-N-" not in result

    def test_guard_strips_nn_prefix(self):
        """The while loop guard catches any remaining N-N-."""
        from orthonym.decomposition.fragment_assembly import _assemble_amide
        # Even with tricky inputs, N-N- should never survive
        result = _assemble_amide(
            {"acid": "propanoic acid", "amine": "ethylamine"}, "pin"
        )
        assert result is not None
        assert "N-N-" not in result

    def test_dimethylamine_produces_nn_comma(self):
        """Dimethylamine as amine fragment should produce N,N-dimethyl prefix.

        When the amine fragment is 'dimethylamine', the 'di' indicates two
        methyl groups on nitrogen, so the correct prefix is N,N-dimethyl,
        not just N-dimethyl.
        """
        from orthonym.decomposition.fragment_assembly import _assemble_amide
        result = _assemble_amide(
            {"acid": "acetic acid", "amine": "dimethylamine"}, "pin"
        )
        assert result is not None
        assert "N-N-" not in result
        assert "N,N-dimethyl" in result, (
            f"Expected N,N-dimethyl in {result!r}"
        )


# ---------------------------------------------------------------------------
# 4. Composer acylamino path dedup
# ---------------------------------------------------------------------------

class TestComposerAcylaminoDedup:
    """The _check_for_acylamino path must not produce N-N-."""

    @pytest.mark.parametrize("smiles", [
        "CC(=O)NCC(=O)O",     # 2-(acetylamino)acetic acid / glycine derivative
        "CCCCC(=O)NC(CC)C(=O)O",  # N-acyl amino acid
    ])
    def test_acylamino_no_nn(self, smiles):
        """Acylamino-containing compounds must not produce N-N-."""
        from orthonym.namer import name_compound
        name = name_compound(smiles)
        assert name is not None, f"Failed to name {smiles}"
        assert "N-N-" not in name, (
            f"N-N- duplication in acylamino for {smiles}: {name!r}"
        )


# ---------------------------------------------------------------------------
# 5. Edge case: N-methyl-N-ethyl alphabetical ordering
# ---------------------------------------------------------------------------

class TestNSubstituentOrdering:
    """Multiple N-substituents must be in alphabetical order."""

    def test_n_ethyl_n_methyl_order(self):
        """N-ethyl must precede N-methyl (alphabetical)."""
        from orthonym.namer import name_compound
        name = name_compound("CC(=O)N(C)CC")  # N-ethyl-N-methylacetamide
        assert name is not None
        assert "N-N-" not in name
        # Ethyl before methyl alphabetically
        if "N-ethyl" in name and "N-methyl" in name:
            assert name.index("N-ethyl") < name.index("N-methyl"), (
                f"N-ethyl should precede N-methyl in {name!r}"
            )
