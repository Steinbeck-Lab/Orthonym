"""
Behavioral freeze test: verify name_compound() output has not changed
for ester SMILES compared to a committed fixture.

The fixture at tests/fixtures/ester_freeze.json was generated before
the Phase 22 fragment-aware naming changes. This test ensures zero
behavioral regression from ester/amide routing modifications.

Marked as @pytest.mark.integration since 53 RDKit calls are fast (<5s).
Run with: pytest -m integration tests/integration/test_ester_behavioral_freeze.py -v
"""
import json
import pathlib

import pytest

from orthonym.namer import name_compound

FIXTURE_PATH = pathlib.Path(__file__).parents[1] / "fixtures" / "ester_freeze.json"


@pytest.mark.integration
class TestEsterBehavioralFreeze:
    """Validate name_compound() produces identical output to pre-change fixture."""

    @classmethod
    def setup_class(cls):
        with open(FIXTURE_PATH) as f:
            cls.expected = json.load(f)

    @pytest.mark.parametrize(
        "smiles",
        sorted(json.load(open(FIXTURE_PATH)).keys()),
        ids=[
            f"{v}|{k}"
            for k, v in sorted(json.load(open(FIXTURE_PATH)).items())
        ],
    )
    def test_ester_name_unchanged(self, smiles):
        result = name_compound(smiles)
        expected = self.expected[smiles]
        assert result == expected, (
            f"Ester freeze regression: expected '{expected}', got '{result}' for {smiles}"
        )
