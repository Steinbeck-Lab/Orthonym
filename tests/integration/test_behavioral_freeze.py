"""
Behavioral freeze test: verify name_compound output has not changed
for simple chain SMILES (C*n, n=1..500) compared to a committed fixture.

The fixture at tests/fixtures/chain_freeze_1_500.json was generated before
the a phase data consolidation changes. This test ensures zero behavioral
regression from chain prefix cleanup and extension.

Marked as @pytest.mark.slow since 500 RDKit calls take ~18s.
Run with: pytest -m slow tests/integration/test_behavioral_freeze.py -v
"""
import json
import pathlib

import pytest

from orthonym.namer import name_compound

FIXTURE_PATH = pathlib.Path(__file__).parents[1] / "fixtures" / "chain_freeze_1_500.json"


@pytest.mark.slow
class TestBehavioralFreeze:
    """Validate name_compound produces identical output to pre-change fixture."""

    @classmethod
    def setup_class(cls):
        with open(FIXTURE_PATH) as f:
            cls.expected = json.load(f)

    @pytest.mark.parametrize("n", range(1, 501))
    def test_chain_name_unchanged(self, n):
        smiles = "C" * n
        result = name_compound(smiles)
        expected = self.expected[str(n)]
        assert result == expected, (
            f"Chain {n}: expected '{expected}', got '{result}'"
        )
