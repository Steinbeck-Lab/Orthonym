"""a phase Plan-01: seed-table integrity unit tests.

Mirrors the class-per-archaic discipline of tests/unit/data/test_retained_names_pin_cleanup.py
(PATTERNS.md analog). Asserts the locked seed table honors internal notes (shape +
Type coverage), (deny-list-by-data + + italic-locant ban), and
(canonicalization idempotence).

Class names are binding per internal notes per-task verification map.

Source: 168-internal notes //; internal notes class-per-archaic analog.
"""

import pytest
from rdkit import Chem

from orthonym.data import _PIN_DENY
from orthonym.data.triviality_controller_seed import (
    SEED_TABLE,
    SeedEntry,
    SubstitutionType,
)

# "no longer recommended" prefixes (a phase inheritance).
P29_6_3_DEPRECATED = frozenset(
    {"benzhydryl", "phenethyl", "isobutyl", "sec-butyl",
     "isopentyl", "tert-pentyl", "neopentyl"}
)


class TestSeedTableShape:
    """The seed table is non-empty, typed, and fully P-section cited ."""

    @pytest.mark.unit
    def test_seed_table_non_empty(self):
        assert len(SEED_TABLE) >= 45, f"SEED_TABLE has {len(SEED_TABLE)} entries; expected >= 45"

    @pytest.mark.unit
    def test_all_values_are_seed_entries(self):
        assert all(isinstance(e, SeedEntry) for e in SEED_TABLE.values())

    @pytest.mark.unit
    def test_every_entry_has_p_section(self):
        for e in SEED_TABLE.values():
            assert e.iupac_p_section, f"{e.retained_pin_name!r} missing iupac_p_section"


class TestSubstitutionTypeCoverage:
    """Type distribution matches the / seed enumeration."""

    @pytest.mark.unit
    def test_type_1_count(self):
        n = sum(1 for e in SEED_TABLE.values() if e.substitution_type is SubstitutionType.TYPE_1)
        assert n >= 30, f"type_1 count {n} < 30"

    @pytest.mark.unit
    def test_type_2a_count(self):
        n = sum(1 for e in SEED_TABLE.values() if e.substitution_type is SubstitutionType.TYPE_2A)
        assert n >= 5, f"type_2a count {n} < 5"

    @pytest.mark.unit
    def test_type_3_count(self):
        n = sum(1 for e in SEED_TABLE.values() if e.substitution_type is SubstitutionType.TYPE_3)
        assert n >= 3, f"type_3 count {n} < 3"


class TestType2aPrincipalGroupRequired:
    """Every Type 2a entry carries a principal_group_required ."""

    @pytest.mark.unit
    def test_type_2a_entries_have_principal_group(self):
        type_2a = [e for e in SEED_TABLE.values() if e.substitution_type is SubstitutionType.TYPE_2A]
        assert type_2a, "expected at least one Type 2a entry"
        for e in type_2a:
            assert e.principal_group_required, (
                f"Type 2a entry {e.retained_pin_name!r} missing principal_group_required"
            )


class TestType3LocantContext:
    """Every xylene Type 3 entry carries a locant_context of length >= 2 ."""

    @pytest.mark.unit
    def test_xylene_entries_have_locant_context(self):
        xylenes = [e for e in SEED_TABLE.values() if "xylene" in e.retained_pin_name]
        assert xylenes, "expected the xylene isomer entries"
        for e in xylenes:
            assert e.locant_context is not None, f"{e.retained_pin_name!r} missing locant_context"
            assert len(e.locant_context) >= 2, f"{e.retained_pin_name!r} locant_context too short"


class TestPINDenyGate:
    """No _PIN_DENY name reaches the seed table -- deny-list-by-data."""

    @pytest.mark.unit
    def test_no_deny_listed_name_in_seed(self):
        leaked = [
            e.retained_pin_name for e in SEED_TABLE.values()
            if e.retained_pin_name.lower().strip() in _PIN_DENY
        ]
        assert not leaked, f"deny-listed names leaked into seed table: {leaked}"


class TestP29_6_3DeprecatedNotInSeed:
    """The 7 deprecated prefixes never appear -- a phase inheritance."""

    @pytest.mark.unit
    def test_no_p29_6_3_deprecated_prefix_in_seed(self):
        names = {e.retained_pin_name.lower().strip() for e in SEED_TABLE.values()}
        leaked = sorted(P29_6_3_DEPRECATED & names)
        assert not leaked, f"P-29.6.3 deprecated prefixes in seed table: {leaked}"


class TestNoItalicLocants:
    """No o-/m-/p- italic-letter locant in any retained_pin_name +."""

    @pytest.mark.unit
    def test_no_italic_locants(self):
        offenders = []
        for e in SEED_TABLE.values():
            n = e.retained_pin_name.lower()
            if n.startswith(("o-", "m-", "p-")) or any(
                tok in n for tok in (" o-", " m-", " p-", "(o-", "(m-", "(p-")
            ):
                offenders.append(e.retained_pin_name)
        assert not offenders, f"italic-letter locants found: {offenders}"


class TestCanonicalSMILESForm:
    """Every canonical_smiles is Chem.CanonSmiles-stable (idempotence)."""

    @pytest.mark.unit
    def test_all_keys_canonical_idempotent(self):
        for e in SEED_TABLE.values():
            assert Chem.CanonSmiles(e.canonical_smiles) == e.canonical_smiles, (
                f"{e.retained_pin_name!r} canonical_smiles {e.canonical_smiles!r} not idempotent"
            )


class TestHeadlineEntriesPresent:
    """The headline aromatic / acid / heterocycle entries are keyed by canonical SMILES."""

    HEADLINE = [
        ("c1ccccc1", "benzene"),
        ("Oc1ccccc1", "phenol"),
        ("Nc1ccccc1", "aniline"),
        ("OC(=O)c1ccccc1", "benzoic acid"),
        ("CC(=O)O", "acetic acid"),
        ("c1ccncc1", "pyridine"),
        ("c1ccc2ccccc2c1", "naphthalene"),
        ("C1COCCN1", "morpholine"),
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles,name", HEADLINE)
    def test_headline_entry_present(self, smiles, name):
        key = Chem.CanonSmiles(smiles)
        assert key in SEED_TABLE, f"{name!r} ({key}) missing from SEED_TABLE"
        assert SEED_TABLE[key].retained_pin_name == name
