"""Integration tests for the confidence metadata API (Phase 81, Task 3).

Verifies:
1. Orthonym.name_with_confidence() returns correct dict shape
2. name_compound(include_confidence=True) returns dict
3. name_compound() default still returns str (backward compat)
4. Confidence values are in valid range
5. Names match between name() and name_with_confidence()
6. CLI --confidence flag works
"""

import subprocess
import sys

import pytest
from orthonym import name_compound, Orthonym


class TestNameWithConfidence:
    """Tests for Orthonym.name_with_confidence()."""

    def test_returns_dict(self):
        """name_with_confidence returns a dict with all required keys."""
        namer = Orthonym()
        result = namer.name_with_confidence("CCO")
        assert isinstance(result, dict)
        assert 'name' in result
        assert 'confidence' in result
        assert 'factors' in result
        assert 'handler' in result

    def test_ethanol_high_confidence(self):
        """Ethanol (retained name) should get high confidence."""
        namer = Orthonym()
        result = namer.name_with_confidence("CCO")
        assert result['name'] == "ethanol"
        # Early return path for retained names -> confidence 1.0
        assert result['confidence'] == 1.0

    def test_invalid_smiles_raises(self):
        """Invalid SMILES raises ValueError."""
        namer = Orthonym()
        with pytest.raises(ValueError):
            namer.name_with_confidence("INVALID_SMILES_XYZ")

    def test_complex_molecule_has_factors(self):
        """Complex molecule returns all 5 factor keys.

        Phase 145.1 ISS-004 update: 'parent_correctness' is the 5th factor
        added at LAST position in FACTOR_WEIGHTS (Plan 02 D-14). It must be
        present in the factors dict — value is 0.5 in production path
        (FACTOR_WEIGHTS['parent_correctness']=0.0 zeros its contribution to
        confidence aggregation, preserving byte-identical behavior).
        """
        namer = Orthonym()
        # Substituted quinoline -- goes through candidate collection
        result = namer.name_with_confidence(
            "CCCCCCCCCc1cc(=O)c2ccccc2n1C"
        )
        assert isinstance(result['factors'], dict)
        expected_keys = {'ratio', 'atom_coverage',
                         'fg_recognition', 'substituent_completeness',
                         'parent_correctness'}
        assert set(result['factors'].keys()) == expected_keys


class TestNameCompoundConfidence:
    """Tests for name_compound(include_confidence=True)."""

    def test_include_confidence_returns_dict(self):
        """name_compound with include_confidence=True returns dict."""
        result = name_compound("CCO", include_confidence=True)
        assert isinstance(result, dict)
        assert 'name' in result
        assert 'confidence' in result
        assert 'factors' in result
        assert 'handler' in result

    def test_default_returns_str(self):
        """name_compound() without include_confidence returns str."""
        result = name_compound("CCO")
        assert isinstance(result, str)
        assert result == "ethanol"

    def test_confidence_values_in_range(self):
        """All confidence and factor values are in [0.0, 1.0]."""
        test_smiles = [
            "CCO",           # ethanol
            "c1ccccc1",      # benzene
            "c1ccncc1",      # pyridine
            "CC(=O)O",       # acetic acid
            "Nc1ncnc2nc[nH]c12",  # adenine
        ]
        for smi in test_smiles:
            result = name_compound(smi, include_confidence=True)
            assert 0.0 <= result['confidence'] <= 1.0, \
                f"confidence out of range for {smi}: {result['confidence']}"
            for k, v in result.get('factors', {}).items():
                assert 0.0 <= v <= 1.0, \
                    f"factor {k} out of range for {smi}: {v}"


class TestAPIParity:
    """Verify name() and name_with_confidence()['name'] produce same results."""

    DIVERSE_SMILES = [
        "CCO",                          # ethanol
        "c1ccccc1",                     # benzene
        "CC(=O)O",                      # acetic acid
        "c1ccncc1",                     # pyridine
        "Nc1ncnc2nc[nH]c12",           # adenine
        "c1ccc2ccccc2c1",              # naphthalene
        "O=C(O)c1ccccc1",             # benzoic acid
        "CCN(CC)CC",                    # triethylamine
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O", # caffeine
        "Cc1ccccc1",                    # toluene
    ]

    @pytest.mark.parametrize("smiles", DIVERSE_SMILES)
    def test_names_match(self, smiles):
        """name() and name_with_confidence()['name'] must agree."""
        namer = Orthonym()
        plain_name = namer.name(smiles)
        conf_result = namer.name_with_confidence(smiles)
        assert plain_name == conf_result['name'], \
            f"Mismatch for {smiles}: name()={plain_name!r} vs " \
            f"name_with_confidence()={conf_result['name']!r}"


class TestCLIConfidence:
    """Tests for the CLI --confidence flag."""

    def test_cli_confidence_flag(self):
        """CLI --confidence flag prints confidence metadata."""
        from orthonym.cli import main
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            ret = main(["c1ccccc1", "--confidence"])
        assert ret == 0
        output = buf.getvalue()
        assert "Name:" in output
        assert "Confidence:" in output
        assert "Handler:" in output

    def test_cli_default_no_confidence(self):
        """CLI without --confidence prints only the name."""
        from orthonym.cli import main
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            ret = main(["c1ccccc1"])
        assert ret == 0
        assert buf.getvalue().strip() == "benzene"
