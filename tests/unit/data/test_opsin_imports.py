"""
Tests for OPSIN XML data imports.

Comprehensive spot-check tests for all generated data modules.
Verifies through requirements.
"""

import json
import random
from pathlib import Path

import pytest
from rdkit import Chem

# ============================================================================
# Module existence and minimum counts (through)
# ============================================================================


@pytest.mark.unit
class TestModuleCounts:
    """Verify minimum entry counts for each generated data module."""

    def test_aryl_groups_count(self):
        """/: arylGroups.xml yields 350+ unique entries."""
        from orthonym.data.opsin_imports.aryl_groups import OPSIN_ARYL_GROUPS

        assert len(OPSIN_ARYL_GROUPS) >= 350, (
            f"Expected >= 350 aryl group entries, got {len(OPSIN_ARYL_GROUPS)}"
        )

    def test_simple_groups_count(self):
        """/: simpleGroups.xml yields 400+ unique entries."""
        from orthonym.data.opsin_imports.simple_groups import OPSIN_SIMPLE_GROUPS

        assert len(OPSIN_SIMPLE_GROUPS) >= 400, (
            f"Expected >= 400 simple group entries, got {len(OPSIN_SIMPLE_GROUPS)}"
        )

    def test_acid_stems_count(self):
        """: carboxylicAcids.xml yields 240+ unique entries."""
        from orthonym.data.opsin_imports.carboxylic_acids_opsin import OPSIN_ACID_STEMS

        assert len(OPSIN_ACID_STEMS) >= 240, (
            f"Expected >= 240 acid stem entries, got {len(OPSIN_ACID_STEMS)}"
        )

    def test_substituent_names_count(self):
        """: simpleSubstituents.xml + substituents.xml yields 400+ entries."""
        from orthonym.data.opsin_imports.substituent_names_opsin import (
            OPSIN_SUBSTITUENT_NAMES,
        )

        assert len(OPSIN_SUBSTITUENT_NAMES) >= 400, (
            f"Expected >= 400 substituent entries, got {len(OPSIN_SUBSTITUENT_NAMES)}"
        )

    def test_amino_acids_count(self):
        """: aminoAcids.xml yields 230+ unique entries."""
        from orthonym.data.opsin_imports.amino_acids_opsin import OPSIN_AMINO_ACIDS

        assert len(OPSIN_AMINO_ACIDS) >= 230, (
            f"Expected >= 230 amino acid entries, got {len(OPSIN_AMINO_ACIDS)}"
        )

    def test_carbohydrates_count(self):
        """: carbohydrates.xml yields 100+ unique entries."""
        from orthonym.data.opsin_imports.carbohydrates_opsin import OPSIN_CARBOHYDRATES

        assert len(OPSIN_CARBOHYDRATES) >= 100, (
            f"Expected >= 100 carbohydrate entries, got {len(OPSIN_CARBOHYDRATES)}"
        )

    def test_natural_products_count(self):
        """: naturalProducts.xml yields 100+ unique entries."""
        from orthonym.data.opsin_imports.natural_products_opsin import (
            OPSIN_NATURAL_PRODUCTS,
        )

        assert len(OPSIN_NATURAL_PRODUCTS) >= 100, (
            f"Expected >= 100 natural product entries, got {len(OPSIN_NATURAL_PRODUCTS)}"
        )

    def test_suffix_rules_count(self):
        """: suffixRules.xml yields 140+ rule entries."""
        from orthonym.data.opsin_imports.suffix_rules import OPSIN_SUFFIX_RULES

        assert len(OPSIN_SUFFIX_RULES) >= 140, (
            f"Expected >= 140 suffix rules, got {len(OPSIN_SUFFIX_RULES)}"
        )

    def test_suffix_applicability_exists(self):
        """: suffixApplicability.xml yields 250+ entries."""
        from orthonym.data.opsin_imports.suffix_rules import OPSIN_SUFFIX_APPLICABILITY

        assert len(OPSIN_SUFFIX_APPLICABILITY) >= 250, (
            f"Expected >= 250 suffix applicability entries, got {len(OPSIN_SUFFIX_APPLICABILITY)}"
        )

    def test_word_rules_count(self):
        """: wordRules.xml yields 55+ rule entries."""
        from orthonym.data.opsin_imports.word_rules import OPSIN_WORD_RULES

        assert len(OPSIN_WORD_RULES) >= 55, (
            f"Expected >= 55 word rules, got {len(OPSIN_WORD_RULES)}"
        )

    def test_cyclic_groups_count(self):
        """simpleCyclicGroups + cyclicUnsaturableHydrocarbon yields 140+ entries."""
        from orthonym.data.opsin_imports.cyclic_groups import OPSIN_CYCLIC_GROUPS

        assert len(OPSIN_CYCLIC_GROUPS) >= 140, (
            f"Expected >= 140 cyclic group entries, got {len(OPSIN_CYCLIC_GROUPS)}"
        )

    def test_functional_terms_count(self):
        """functionalTerms.xml yields 100+ entries."""
        from orthonym.data.opsin_imports.functional_terms import OPSIN_FUNCTIONAL_TERMS

        assert len(OPSIN_FUNCTIONAL_TERMS) >= 100, (
            f"Expected >= 100 functional term entries, got {len(OPSIN_FUNCTIONAL_TERMS)}"
        )

    def test_fusion_components_count(self):
        """fusionComponents.xml yields 120+ entries."""
        from orthonym.data.opsin_imports.fusion_components_opsin import (
            OPSIN_FUSION_COMPONENTS,
        )

        assert len(OPSIN_FUSION_COMPONENTS) >= 120, (
            f"Expected >= 120 fusion component entries, got {len(OPSIN_FUSION_COMPONENTS)}"
        )


# ============================================================================
# Known entry spot-checks
# ============================================================================


@pytest.mark.unit
class TestKnownEntries:
    """Verify specific high-value entries parsed correctly."""

    def test_benzene_in_aryl_groups(self):
        """Canonical SMILES for benzene is in OPSIN_ARYL_GROUPS with 'benz' name."""
        from orthonym.data.opsin_imports.aryl_groups import OPSIN_ARYL_GROUPS

        benzene_smiles = Chem.CanonSmiles("c1ccccc1")
        # Look for benzene with or without structural key suffix
        found = False
        for key, meta in OPSIN_ARYL_GROUPS.items():
            smiles = meta.get("smiles", key.split("||")[0])
            if smiles == benzene_smiles:
                if any("benz" in n for n in meta.get("names", [])):
                    found = True
                    break
        assert found, f"Benzene ({benzene_smiles}) with 'benz' name not found"

    def test_naphthalene_in_aryl_groups(self):
        """Canonical SMILES for naphthalene is present in OPSIN_ARYL_GROUPS."""
        from orthonym.data.opsin_imports.aryl_groups import OPSIN_ARYL_GROUPS

        naph_smiles = Chem.CanonSmiles("c1ccc2ccccc2c1")
        found = False
        for key, meta in OPSIN_ARYL_GROUPS.items():
            smiles = meta.get("smiles", key.split("||")[0])
            if smiles == naph_smiles:
                found = True
                break
        assert found, f"Naphthalene ({naph_smiles}) not found in aryl groups"

    def test_acet_acid_stem(self):
        """Acetic acid stem 'acet' present in OPSIN_ACID_STEMS."""
        from orthonym.data.opsin_imports.carboxylic_acids_opsin import OPSIN_ACID_STEMS

        found = False
        for key, meta in OPSIN_ACID_STEMS.items():
            if "acet" in meta.get("names", []):
                found = True
                break
        assert found, "'acet' acid stem not found in OPSIN_ACID_STEMS"

    def test_glycine_in_amino_acids(self):
        """Glycine stem 'glyc' present in OPSIN_AMINO_ACIDS."""
        from orthonym.data.opsin_imports.amino_acids_opsin import OPSIN_AMINO_ACIDS

        found = False
        for key, meta in OPSIN_AMINO_ACIDS.items():
            if "glyc" in meta.get("names", []):
                found = True
                break
        assert found, "'glyc' not found in OPSIN_AMINO_ACIDS"

    def test_glucose_in_carbohydrates(self):
        """A glucose-related entry exists in OPSIN_CARBOHYDRATES."""
        from orthonym.data.opsin_imports.carbohydrates_opsin import OPSIN_CARBOHYDRATES

        found = False
        for key, meta in OPSIN_CARBOHYDRATES.items():
            if any("gluc" in n for n in meta.get("names", [])):
                found = True
                break
        assert found, "No glucose-related entry found in OPSIN_CARBOHYDRATES"

    def test_cholestane_in_natural_products(self):
        """A steroid scaffold entry exists in OPSIN_NATURAL_PRODUCTS."""
        from orthonym.data.opsin_imports.natural_products_opsin import (
            OPSIN_NATURAL_PRODUCTS,
        )

        steroid_names = ["cholest", "androst", "estr", "pregn"]
        found = False
        for key, meta in OPSIN_NATURAL_PRODUCTS.items():
            for name in meta.get("names", []):
                if any(s in name for s in steroid_names):
                    found = True
                    break
            if found:
                break
        assert found, "No steroid scaffold entry found in OPSIN_NATURAL_PRODUCTS"


# ============================================================================
# SMILES canonicality checks
# ============================================================================


@pytest.mark.unit
class TestSMILESCanonicality:
    """Verify all SMILES keys are properly canonicalized."""

    def test_all_keys_are_canonical_smiles(self):
        """For 50 random aryl group entries, verify SMILES keys are canonical."""
        from orthonym.data.opsin_imports.aryl_groups import OPSIN_ARYL_GROUPS

        keys = list(OPSIN_ARYL_GROUPS.keys())
        sample = random.sample(keys, min(50, len(keys)))

        for key in sample:
            # Extract base SMILES from composite keys
            smiles = key.split("||")[0] if "||" in key else key
            canon = Chem.CanonSmiles(smiles)
            assert canon == smiles, (
                f"Non-canonical SMILES key: {smiles} -> {canon}"
            )

    def test_no_radical_prefix_in_keys(self):
        """No key in OPSIN_SUBSTITUENT_NAMES starts with '-' or '='."""
        from orthonym.data.opsin_imports.substituent_names_opsin import (
            OPSIN_SUBSTITUENT_NAMES,
        )

        for key in OPSIN_SUBSTITUENT_NAMES:
            smiles = key.split("||")[0] if "||" in key else key
            assert not smiles.startswith("-"), (
                f"Radical prefix '-' found in key: {key}"
            )
            assert not smiles.startswith("="), (
                f"Radical prefix '=' found in key: {key}"
            )

    def test_amino_acid_keys_are_canonical(self):
        """Amino acid SMILES keys are canonical with stereo preserved."""
        from orthonym.data.opsin_imports.amino_acids_opsin import OPSIN_AMINO_ACIDS

        keys = list(OPSIN_AMINO_ACIDS.keys())
        sample = random.sample(keys, min(30, len(keys)))

        for key in sample:
            smiles = key.split("||")[0] if "||" in key else key
            canon = Chem.CanonSmiles(smiles)
            assert canon == smiles, (
                f"Non-canonical amino acid SMILES: {smiles} -> {canon}"
            )


# ============================================================================
# Metadata structure checks
# ============================================================================


@pytest.mark.unit
class TestMetadataStructure:
    """Verify entries have correct metadata fields."""

    def test_acid_stem_has_suffix_applies_to(self):
        """At least 5 entries in OPSIN_ACID_STEMS have non-None suffixAppliesTo."""
        from orthonym.data.opsin_imports.carboxylic_acids_opsin import OPSIN_ACID_STEMS

        count = sum(
            1 for meta in OPSIN_ACID_STEMS.values()
            if meta.get("suffixAppliesTo") is not None
        )
        assert count >= 5, (
            f"Expected >= 5 acid stems with suffixAppliesTo, got {count}"
        )

    def test_natural_products_have_alpha_beta(self):
        """At least 5 natural products have alphaBetaClockWiseAtomOrdering."""
        from orthonym.data.opsin_imports.natural_products_opsin import (
            OPSIN_NATURAL_PRODUCTS,
        )

        count = sum(
            1 for meta in OPSIN_NATURAL_PRODUCTS.values()
            if meta.get("alphaBetaClockWiseAtomOrdering") is not None
        )
        assert count >= 5, (
            f"Expected >= 5 NPs with alphaBetaClockWiseAtomOrdering, got {count}"
        )

    def test_entry_has_is_pin_field(self):
        """All entries in OPSIN_ARYL_GROUPS have 'is_pin' key set to False."""
        from orthonym.data.opsin_imports.aryl_groups import OPSIN_ARYL_GROUPS

        for key, meta in OPSIN_ARYL_GROUPS.items():
            assert "is_pin" in meta, f"Entry {key} missing 'is_pin' field"
            assert meta["is_pin"] is False, (
                f"Entry {key} has is_pin={meta['is_pin']}, expected False"
            )

    def test_entry_has_names_list(self):
        """All entries have 'names' key which is a list of strings."""
        from orthonym.data.opsin_imports.aryl_groups import OPSIN_ARYL_GROUPS

        for key, meta in OPSIN_ARYL_GROUPS.items():
            assert "names" in meta, f"Entry {key} missing 'names' field"
            assert isinstance(meta["names"], list), (
                f"Entry {key} names is {type(meta['names'])}, expected list"
            )
            assert len(meta["names"]) > 0, f"Entry {key} has empty names list"
            for name in meta["names"]:
                assert isinstance(name, str), (
                    f"Entry {key} has non-string name: {name}"
                )

    def test_substituent_has_radical_type(self):
        """Substituent entries preserve radical_type metadata."""
        from orthonym.data.opsin_imports.substituent_names_opsin import (
            OPSIN_SUBSTITUENT_NAMES,
        )

        single_count = sum(
            1 for meta in OPSIN_SUBSTITUENT_NAMES.values()
            if meta.get("radical_type") == "single"
        )
        double_count = sum(
            1 for meta in OPSIN_SUBSTITUENT_NAMES.values()
            if meta.get("radical_type") == "double"
        )
        # Most substituents have single-bond radical prefix
        assert single_count >= 100, (
            f"Expected >= 100 single-radical substituents, got {single_count}"
        )
        assert double_count >= 10, (
            f"Expected >= 10 double-radical substituents, got {double_count}"
        )


# ============================================================================
# Consolidated export check
# ============================================================================


@pytest.mark.unit
class TestConsolidatedExport:
    """Verify the consolidated OPSIN_RETAINED_NAMES export."""

    def test_opsin_retained_names_consolidated(self):
        """OPSIN_RETAINED_NAMES importable and has 200+ entries.

        Note: Most OPSIN XML tokens are stems (not complete names) so only
        cyclic_groups and natural_products are included in the retained names
        merge. The 900+ figure from the original plan counted raw tokens
        including stems. See data/opsin_imports/__init__.py for the exclusion
        rationale.
        """
        from orthonym.data.opsin_imports import OPSIN_RETAINED_NAMES

        assert len(OPSIN_RETAINED_NAMES) >= 200, (
            f"Expected >= 200 retained names, got {len(OPSIN_RETAINED_NAMES)}"
        )

    def test_opsin_retained_names_values_are_strings(self):
        """OPSIN_RETAINED_NAMES values are all strings (name variants)."""
        from orthonym.data.opsin_imports import OPSIN_RETAINED_NAMES

        for smiles, name in OPSIN_RETAINED_NAMES.items():
            assert isinstance(smiles, str), f"Non-string key: {type(smiles)}"
            assert isinstance(name, str), f"Non-string value for {smiles}: {type(name)}"

    def test_opsin_retained_names_has_benzene(self):
        """OPSIN_RETAINED_NAMES contains benzene."""
        from orthonym.data.opsin_imports import OPSIN_RETAINED_NAMES

        benzene_smiles = Chem.CanonSmiles("c1ccccc1")
        assert benzene_smiles in OPSIN_RETAINED_NAMES, (
            f"Benzene ({benzene_smiles}) not in OPSIN_RETAINED_NAMES"
        )


# ============================================================================
# Validation report check
# ============================================================================


@pytest.mark.unit
class TestValidationReport:
    """Verify the validation report structure and content."""

    @pytest.fixture
    def report(self):
        report_path = (
            Path(__file__).resolve().parents[3]
            / "src"
            / "orthonym"
            / "data"
            / "opsin_imports"
            / "_validation_report.json"
        )
        with open(report_path) as f:
            return json.load(f)

    def test_validation_report_exists(self, report):
        """_validation_report.json loads as valid JSON."""
        assert report is not None
        assert "totals" in report

    def test_validation_report_has_per_file_stats(self, report):
        """Report has per_file_stats with at least 30 file entries."""
        assert "per_file_stats" in report
        assert len(report["per_file_stats"]) >= 30, (
            f"Expected >= 30 file stats, got {len(report['per_file_stats'])}"
        )

    def test_rejection_rate_reasonable(self, report):
        """Total rejected / total parsed < 15%."""
        totals = report["totals"]
        total_parsed = totals.get("total_parsed", totals.get("parsed", 0))
        total_rejected = totals.get("rejected", 0)
        if total_parsed > 0:
            rate = total_rejected / total_parsed
            assert rate < 0.15, (
                f"Rejection rate {rate:.1%} exceeds 15% threshold"
            )

    def test_validation_report_has_total_parsed(self, report):
        """Report contains total_parsed in totals."""
        totals = report["totals"]
        assert "total_parsed" in totals or "parsed" in totals, (
            "Report totals missing 'total_parsed' or 'parsed' field"
        )

    def test_validation_report_has_accepted(self, report):
        """Report contains accepted count."""
        totals = report["totals"]
        assert "accepted" in totals, "Report totals missing 'accepted' field"
        assert totals["accepted"] > 2000, (
            f"Expected > 2000 accepted entries, got {totals['accepted']}"
        )
