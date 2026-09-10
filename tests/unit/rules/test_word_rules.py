"""
Unit tests for OPSIN word rules data .

Verifies that the OPSIN_WORD_RULES data imported from wordRules.xml
has the expected structure and content.
"""

import pytest

from orthonym.data.opsin_imports.word_rules import OPSIN_WORD_RULES


class TestWordRulesData:
    """Tests for OPSIN_WORD_RULES data structure."""

    def test_minimum_rule_count(self):
        """Must have at least 21 rule entries (OPSIN has 21 rule names)."""
        assert len(OPSIN_WORD_RULES) >= 21, (
            f"Expected >= 21 word rules, got {len(OPSIN_WORD_RULES)}"
        )

    def test_rule_entry_structure(self):
        """Each entry must have 'name', 'type', and 'words' keys."""
        for i, entry in enumerate(OPSIN_WORD_RULES):
            assert "name" in entry, f"Entry {i} missing 'name' key"
            assert "type" in entry, f"Entry {i} missing 'type' key"
            assert "words" in entry, f"Entry {i} missing 'words' key"

    def test_words_is_list(self):
        """The 'words' field must be a list."""
        for i, entry in enumerate(OPSIN_WORD_RULES):
            assert isinstance(entry["words"], list), (
                f"Entry {i} 'words' is not a list: {type(entry['words'])}"
            )

    def test_words_entries_are_dicts(self):
        """Each word in 'words' must be a dict with at least 'type'."""
        for i, entry in enumerate(OPSIN_WORD_RULES):
            for j, word in enumerate(entry["words"]):
                assert isinstance(word, dict), (
                    f"Entry {i} word {j} is not a dict"
                )
                assert "type" in word, (
                    f"Entry {i} word {j} missing 'type' key"
                )

    def test_known_rule_names_present(self):
        """Key rule names from OPSIN must be present."""
        rule_names = {entry["name"] for entry in OPSIN_WORD_RULES}
        expected_names = {"ester", "multiEster", "anhydride"}
        missing = expected_names - rule_names
        assert not missing, (
            f"Missing expected rule names: {missing}. "
            f"Found: {sorted(rule_names)}"
        )

    def test_additional_rule_names(self):
        """Additional known rule names should be present."""
        rule_names = {entry["name"] for entry in OPSIN_WORD_RULES}
        # These are known OPSIN word rule names
        expected = {
            "monovalentFunctionalGroup",
            "divalentFunctionalGroup",
            "oxide",
            "acidHalideOrPseudoHalide",
        }
        missing = expected - rule_names
        assert not missing, (
            f"Missing expected rule names: {missing}. "
            f"Found: {sorted(rule_names)}"
        )

    def test_ester_rule_has_ate_pattern(self):
        """Ester rules should reference 'ateGroup' endsWith."""
        ester_rules = [
            r for r in OPSIN_WORD_RULES if r["name"] == "ester"
        ]
        assert len(ester_rules) >= 1, "No ester rules found"

        # At least one ester rule should have endsWith 'ateGroup'
        has_ate = False
        for rule in ester_rules:
            for word in rule["words"]:
                if word.get("endsWith") == "ateGroup":
                    has_ate = True
                    break
        assert has_ate, (
            "No ester rule found with endsWith='ateGroup'"
        )

    def test_unique_rule_name_count(self):
        """Should have at least 15 unique rule names."""
        rule_names = {entry["name"] for entry in OPSIN_WORD_RULES}
        assert len(rule_names) >= 15, (
            f"Expected >= 15 unique rule names, got {len(rule_names)}: "
            f"{sorted(rule_names)}"
        )
