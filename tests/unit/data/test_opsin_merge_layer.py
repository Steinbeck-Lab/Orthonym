"""Tests for the OPSIN imports merge layer in data/__init__.py.

Verifies: hand-curated precedence, backward compatibility, PIN flags,
merged dictionary size, and register_retained_name behavior.
"""

import pytest
from rdkit import Chem


@pytest.mark.unit
class TestMergeLayerBehavior:
    """Tests for merge layer in data/__init__.py."""

    def test_hand_curated_wins_conflict(self):
        """Hand-curated entries take precedence over OPSIN imports (D-11)."""
        from orthonym.data import ALL_RETAINED_NAMES

        # benzene is in both hand-curated and OPSIN
        assert ALL_RETAINED_NAMES.get("c1ccccc1") == "benzene"

    def test_opsin_entries_accessible_via_merged(self):
        """OPSIN-only entries are accessible through ALL_RETAINED_NAMES."""
        from orthonym.data import ALL_RETAINED_NAMES
        from orthonym.data.retained_names import RETAINED_NAMES as hand_curated

        # Find an entry in ALL_RETAINED_NAMES that is NOT in hand-curated
        opsin_only = {
            k: v for k, v in ALL_RETAINED_NAMES.items()
            if k not in hand_curated
        }
        assert len(opsin_only) > 50, (
            f"Expected 50+ OPSIN-only entries, got {len(opsin_only)}"
        )

    def test_backward_compatible_alias(self):
        """RETAINED_NAMES is the same object as ALL_RETAINED_NAMES (D-12)."""
        from orthonym.data import RETAINED_NAMES, ALL_RETAINED_NAMES

        assert RETAINED_NAMES is ALL_RETAINED_NAMES

    def test_get_retained_name_uses_merged(self):
        """get_retained_name() searches the merged dictionary."""
        from orthonym.data import get_retained_name, ALL_RETAINED_NAMES
        from orthonym.data.retained_names import RETAINED_NAMES as hand_curated

        # Find an OPSIN-only SMILES
        opsin_only_smi = None
        for smi in ALL_RETAINED_NAMES:
            if smi not in hand_curated:
                opsin_only_smi = smi
                break

        assert opsin_only_smi is not None, "No OPSIN-only entries found"
        result = get_retained_name(opsin_only_smi)
        assert result is not None, (
            f"get_retained_name({opsin_only_smi}) returned None"
        )

    def test_has_retained_name_uses_merged(self):
        """has_retained_name() checks the merged dictionary."""
        from orthonym.data import has_retained_name, ALL_RETAINED_NAMES
        from orthonym.data.retained_names import RETAINED_NAMES as hand_curated

        # Find an OPSIN-only SMILES
        opsin_only_smi = None
        for smi in ALL_RETAINED_NAMES:
            if smi not in hand_curated:
                opsin_only_smi = smi
                break

        assert opsin_only_smi is not None
        assert has_retained_name(opsin_only_smi) is True

    def test_all_opsin_entries_have_is_pin_false(self):
        """All OPSIN import entries have is_pin=False (DATA-20)."""
        from orthonym.data.opsin_imports import (
            OPSIN_ARYL_GROUPS,
            OPSIN_SIMPLE_GROUPS,
            OPSIN_ACID_STEMS,
            OPSIN_AMINO_ACIDS,
            OPSIN_CARBOHYDRATES,
            OPSIN_NATURAL_PRODUCTS,
            OPSIN_CYCLIC_GROUPS,
        )

        for name, source in [
            ("aryl_groups", OPSIN_ARYL_GROUPS),
            ("simple_groups", OPSIN_SIMPLE_GROUPS),
            ("acid_stems", OPSIN_ACID_STEMS),
            ("amino_acids", OPSIN_AMINO_ACIDS),
            ("carbohydrates", OPSIN_CARBOHYDRATES),
            ("natural_products", OPSIN_NATURAL_PRODUCTS),
            ("cyclic_groups", OPSIN_CYCLIC_GROUPS),
        ]:
            for key, entry in source.items():
                assert "is_pin" in entry, (
                    f"{name}[{key}] missing is_pin field"
                )
                assert entry["is_pin"] is False, (
                    f"{name}[{key}] has is_pin={entry['is_pin']}, expected False"
                )

    def test_merged_count_reasonable(self):
        """ALL_RETAINED_NAMES has 350+ entries (hand-curated + OPSIN)."""
        from orthonym.data import ALL_RETAINED_NAMES

        assert len(ALL_RETAINED_NAMES) >= 350, (
            f"Expected >= 350 merged entries, got {len(ALL_RETAINED_NAMES)}"
        )

    def test_namer_uses_expanded_retained_names(self):
        """namer.py accesses the merged retained names dictionary."""
        from orthonym import namer

        assert hasattr(namer, "RETAINED_NAMES")
        assert len(namer.RETAINED_NAMES) >= 350, (
            f"namer.RETAINED_NAMES only has {len(namer.RETAINED_NAMES)} entries"
        )

    def test_register_retained_name_still_works(self):
        """register_retained_name() modifies ALL_RETAINED_NAMES."""
        from orthonym.data import (
            ALL_RETAINED_NAMES,
            register_retained_name,
        )

        test_smi = "CCCCCCCCCCCCCCC"  # pentadecane - unlikely to be in dict
        original = ALL_RETAINED_NAMES.get(test_smi)

        try:
            register_retained_name(test_smi, "test_compound")
            assert ALL_RETAINED_NAMES[test_smi] == "test_compound"
        finally:
            # Clean up
            if original is not None:
                ALL_RETAINED_NAMES[test_smi] = original
            else:
                ALL_RETAINED_NAMES.pop(test_smi, None)

    def test_no_opsin_stems_in_retained(self):
        """Verify OPSIN stems (incomplete names) are not in ALL_RETAINED_NAMES."""
        from orthonym.data import ALL_RETAINED_NAMES

        # These are known OPSIN stems that should NOT be retained names.
        # Note: "acetylene" is in hand-curated data (it's a valid retained
        # name for HC≡CH per IUPAC), so it's not in this exclusion list.
        known_stems = ["carbazol", "stilben", "phenetol", "butyrine",
                       "glycerone", "dihydrosuccinate", "ethylene",
                       "propylene"]
        for name in known_stems:
            found = [v for v in ALL_RETAINED_NAMES.values() if v == name]
            assert not found, (
                f"OPSIN stem '{name}' found in ALL_RETAINED_NAMES"
            )
