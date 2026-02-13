"""
Unit tests for PubChem validator and dual validator.

All tests use mocked HTTP calls -- no real PubChem API requests are made.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orthonym.validation.pubchem_validator import (
    load_cache,
    lookup_name_pubchem,
    save_cache,
)
from orthonym.validation.dual_validator import DualResult


# =============================================================================
# load_cache tests
# =============================================================================


class TestLoadCache:
    """Tests for load_cache function."""

    def test_missing_file_returns_empty_dict(self, tmp_path):
        """load_cache returns {} when cache file does not exist."""
        result = load_cache(tmp_path / "nonexistent.json")
        assert result == {}

    def test_corrupt_file_returns_empty_dict(self, tmp_path):
        """load_cache returns {} when cache file contains invalid JSON."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{{", encoding="utf-8")
        result = load_cache(bad_file)
        assert result == {}

    def test_valid_cache_loaded(self, tmp_path):
        """load_cache correctly loads a valid JSON cache file."""
        cache_file = tmp_path / "cache.json"
        data = {"ethanol": {"smiles": "CCO", "inchi": "InChI=1S/C2H6O/..."}}
        cache_file.write_text(json.dumps(data), encoding="utf-8")
        result = load_cache(cache_file)
        assert result == data

    def test_non_dict_json_returns_empty_dict(self, tmp_path):
        """load_cache returns {} when JSON is a list instead of dict."""
        cache_file = tmp_path / "list.json"
        cache_file.write_text("[1, 2, 3]", encoding="utf-8")
        result = load_cache(cache_file)
        assert result == {}

    def test_cache_with_none_values(self, tmp_path):
        """load_cache preserves None values (negative cache entries)."""
        cache_file = tmp_path / "cache.json"
        data = {"ethanol": {"smiles": "CCO", "inchi": "InChI=1S/..."}, "bogus-name": None}
        cache_file.write_text(json.dumps(data), encoding="utf-8")
        result = load_cache(cache_file)
        assert result["bogus-name"] is None
        assert result["ethanol"]["smiles"] == "CCO"


# =============================================================================
# save_cache tests
# =============================================================================


class TestSaveCache:
    """Tests for save_cache function."""

    def test_writes_valid_json(self, tmp_path):
        """save_cache writes a JSON file that can be loaded back."""
        cache_file = tmp_path / "cache.json"
        data = {"ethanol": {"smiles": "CCO", "inchi": "InChI=1S/..."}}
        save_cache(data, cache_file)
        with open(cache_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_creates_parent_directories(self, tmp_path):
        """save_cache creates parent dirs if they don't exist."""
        cache_file = tmp_path / "subdir" / "nested" / "cache.json"
        save_cache({"test": "data"}, cache_file)
        assert cache_file.exists()
        with open(cache_file, "r", encoding="utf-8") as f:
            assert json.load(f) == {"test": "data"}

    def test_preserves_none_values(self, tmp_path):
        """save_cache correctly serializes None values."""
        cache_file = tmp_path / "cache.json"
        data = {"good-name": {"smiles": "CCO", "inchi": "X"}, "bad-name": None}
        save_cache(data, cache_file)
        with open(cache_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["bad-name"] is None


# =============================================================================
# lookup_name_pubchem tests
# =============================================================================


class TestLookupNamePubchem:
    """Tests for lookup_name_pubchem function."""

    def test_returns_cached_result_without_api_call(self):
        """lookup_name_pubchem returns cached result without network call."""
        cache = {"ethanol": {"smiles": "CCO", "inchi": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"}}
        result = lookup_name_pubchem("ethanol", cache, skip_api=True)
        assert result is not None
        assert result["smiles"] == "CCO"

    def test_returns_cached_none_for_negative_results(self):
        """lookup_name_pubchem returns None for cached negative results."""
        cache = {"nonexistent-compound": None}
        result = lookup_name_pubchem("nonexistent-compound", cache, skip_api=True)
        assert result is None

    def test_skip_api_returns_none_on_cache_miss(self):
        """skip_api=True returns None for names not in cache."""
        cache = {}
        result = lookup_name_pubchem("unknown-name", cache, skip_api=True)
        assert result is None

    @patch("orthonym.validation.pubchem_validator.requests.get")
    @patch("orthonym.validation.pubchem_validator.time.sleep")
    def test_api_call_on_cache_miss(self, mock_sleep, mock_get):
        """lookup_name_pubchem makes API call on cache miss."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "PropertyTable": {
                "Properties": [
                    {
                        "IsomericSMILES": "CCO",
                        "InChI": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        cache = {}
        result = lookup_name_pubchem("ethanol", cache)

        assert result is not None
        assert result["smiles"] == "CCO"
        assert result["inchi"] == "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
        # Result should be cached now
        assert "ethanol" in cache
        assert cache["ethanol"]["smiles"] == "CCO"
        # Rate limiting sleep should have been called
        mock_sleep.assert_called_once_with(0.25)

    @patch("orthonym.validation.pubchem_validator.requests.get")
    @patch("orthonym.validation.pubchem_validator.time.sleep")
    def test_api_call_with_smiles_key_fallback(self, mock_sleep, mock_get):
        """lookup_name_pubchem handles PubChem returning 'SMILES' instead of 'IsomericSMILES'."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "PropertyTable": {
                "Properties": [
                    {
                        "SMILES": "CCO",
                        "InChI": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        cache = {}
        result = lookup_name_pubchem("ethanol", cache)

        assert result is not None
        assert result["smiles"] == "CCO"

    @patch("orthonym.validation.pubchem_validator.requests.get")
    @patch("orthonym.validation.pubchem_validator.time.sleep")
    def test_404_caches_none(self, mock_sleep, mock_get):
        """lookup_name_pubchem caches None for 404 responses."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        cache = {}
        result = lookup_name_pubchem("nonexistent-compound", cache)

        assert result is None
        assert "nonexistent-compound" in cache
        assert cache["nonexistent-compound"] is None

    @patch("orthonym.validation.pubchem_validator.requests.get")
    @patch("orthonym.validation.pubchem_validator.time.sleep")
    def test_exception_does_not_cache(self, mock_sleep, mock_get):
        """lookup_name_pubchem does NOT cache results on network error."""
        import requests as req

        mock_get.side_effect = req.ConnectionError("Network error")

        cache = {}
        result = lookup_name_pubchem("ethanol", cache)

        assert result is None
        assert "ethanol" not in cache  # Should NOT be cached

    @patch("orthonym.validation.pubchem_validator.requests.get")
    @patch("orthonym.validation.pubchem_validator.time.sleep")
    def test_timeout_does_not_cache(self, mock_sleep, mock_get):
        """lookup_name_pubchem does NOT cache results on timeout."""
        import requests as req

        mock_get.side_effect = req.Timeout("Request timed out")

        cache = {}
        result = lookup_name_pubchem("ethanol", cache)

        assert result is None
        assert "ethanol" not in cache


# =============================================================================
# DualResult tests
# =============================================================================


class TestDualResult:
    """Tests for DualResult dataclass."""

    def test_default_construction(self):
        """DualResult defaults to all False/None."""
        dr = DualResult()
        assert dr.opsin_parsed is False
        assert dr.opsin_rt is False
        assert dr.pubchem_resolved is False
        assert dr.pubchem_rt is False
        assert dr.combined_rt is False
        assert dr.opsin_smiles is None
        assert dr.pubchem_inchi is None

    def test_combined_rt_with_opsin_only(self):
        """combined_rt is True when only OPSIN matches."""
        dr = DualResult(opsin_parsed=True, opsin_rt=True, combined_rt=True)
        assert dr.combined_rt is True

    def test_combined_rt_with_pubchem_only(self):
        """combined_rt is True when only PubChem matches."""
        dr = DualResult(pubchem_resolved=True, pubchem_rt=True, combined_rt=True)
        assert dr.combined_rt is True

    def test_combined_rt_false_when_neither_matches(self):
        """combined_rt is False when neither validator matches."""
        dr = DualResult(opsin_parsed=True, opsin_rt=False)
        assert dr.combined_rt is False

    def test_fields_populated(self):
        """DualResult can store SMILES and InChI from validators."""
        dr = DualResult(
            opsin_parsed=True,
            opsin_rt=True,
            opsin_smiles="CCO",
            pubchem_resolved=False,
            pubchem_inchi=None,
            combined_rt=True,
        )
        assert dr.opsin_smiles == "CCO"
        assert dr.pubchem_inchi is None
