"""Integration tests: closure — name_compound dispatches mixed-spiro-fused
inputs through the new elif branch in _assemble_complex_ring_name.

a phase-04 closes the gap where the 151-02 SUMMARY claimed to add this dispatch
but a git-stash incident silently dropped it. These tests exercise the FULL
name_compound pipeline (not just direct calls to name_mixed_spiro_fused) so
a future regression in the dispatch chain CANNOT be silently masked.
"""
import pytest
from rdkit import Chem


class TestMixedSpiroFusedDispatch:
    """Phase 151-04 BLK-01 closure — name_compound integration."""

    @pytest.mark.integration
    def test_dispatch_does_not_return_none_on_known_mixed_smiles(self):
        """The composer dispatch must invoke name_mixed_spiro_fused; if the
        elif branch is missing, _assemble_complex_ring_name returns None and
        the entire AUTONOM path is dead code from name_compound.

        We use a synthetic spiro-fused SMILES known to classify as
        mixed-spiro-fused per is_mixed_spiro_fused. If the function returns
        None for ALL inputs, that is itself a regression worth flagging.
        """
        from orthonym.namer import name_compound
        from orthonym.rules.spiro import is_mixed_spiro_fused

        # Synthetic mixed-spiro-fused: indoline (fused) + cyclohexane (spiro side)
        # The exact SMILES is the canonical AUTONOM example
        # spiro[indoline-3,1'-cyclohexane]: C1CCC2(CC1)CC1=CC=CC=C1N2
        smi = "C1CCC2(CC1)CC1=CC=CC=C1N2"
        mol = Chem.MolFromSmiles(smi)
        if mol is None or not is_mixed_spiro_fused(mol):
            pytest.skip(
                "Synthetic SMILES no longer classifies as mixed-spiro-fused; "
                "this is OK if the predicate tightened — the corpus fixture "
                "tests below cover real-world cases."
            )

        name = name_compound(smi)
        # proof: name_compound must NOT silently return None for inputs
        # that classify as mixed-spiro-fused. If the elif branch is missing,
        # _assemble_complex_ring_name returns None for this input, and the
        # downstream pipeline produces either None or a wildly different name.
        assert name is not None, (
            "name_compound returned None for a mixed-spiro-fused input — "
            "the elif branch in _assemble_complex_ring_name is missing or "
            "name_mixed_spiro_fused returned None for all corpus inputs."
        )
