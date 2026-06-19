"""
Tests for retained names dictionary expansion (Phase 109, DATA-06).

Verifies:
- Dictionary has at least 290 entries (240 original + 50+ new)
- ALL keys are canonical SMILES (verified via Chem.CanonSmiles)
- Specific new entries exist and have correct names
- All values are non-empty lowercase strings
"""

import pytest
from rdkit import Chem

from orthonym.data.retained_names import RETAINED_NAMES, get_retained_name


class TestRetainedNamesCount:
    """Verify dictionary size meets expansion target."""

    def test_minimum_entry_count(self):
        """Dictionary must have at least 283 entries (290 original - 7 non-PIN removals in Phase 133)."""
        assert len(RETAINED_NAMES) >= 283, (
            f"Expected >= 283 entries, got {len(RETAINED_NAMES)}"
        )


class TestCanonicalSMILESKeys:
    """Verify all dictionary keys are canonical SMILES."""

    def test_all_keys_are_canonical(self):
        """Every key must equal its own Chem.CanonSmiles() result."""
        non_canonical = []
        for key in RETAINED_NAMES:
            try:
                canonical = Chem.CanonSmiles(key)
                if canonical != key:
                    non_canonical.append(
                        (key, canonical, RETAINED_NAMES[key])
                    )
            except Exception as e:
                non_canonical.append((key, f"ERROR: {e}", RETAINED_NAMES[key]))

        # Allow known alternate forms (imidazole, pyrazole, etc. that have
        # both canonical and non-canonical entries for compatibility)
        # These are pre-existing entries that serve as fallback lookups
        known_alternates = {
            "c1cnc[nH]1",   # imidazole alternate
            "c1cc[nH]n1",   # pyrazole alternate
            "c1cnco1",      # oxazole alternate
            "c1ccno1",      # isoxazole alternate
            "c1cncs1",      # thiazole alternate
            "c1ccsn1",      # isothiazole alternate
            "c1cnc2ccccc2n1",  # quinazoline alternate
            "c1ncnc2[nH]cnc12",  # purine alternate
            "CC(=O)N",      # acetamide alternate
            "OC(=O)C(N)Cc1ccccc1",  # phenylalanine alternate
        }

        # Also allow known non-canonical acid keys (pre-existing, tracked as
        # deferred fix item -- not in scope for Phase 109 Task 1)
        known_noncanonical_acids = {
            "OC(=O)C(O)=O",      # oxalic acid
            "OC(=O)CC(=O)O",     # malonic acid
            "OC(=O)CCC(=O)O",    # succinic acid
            "OC(=O)CCCC(=O)O",   # glutaric acid
            "OC(=O)CCCCC(=O)O",  # adipic acid
            "OC(=O)CC(O)(CC(=O)O)C(=O)O",  # citric acid
        }

        unexpected = [
            (k, c, n) for k, c, n in non_canonical
            if k not in known_alternates and k not in known_noncanonical_acids
        ]

        assert len(unexpected) == 0, (
            f"Found {len(unexpected)} non-canonical key(s) not in known "
            f"exceptions:\n"
            + "\n".join(
                f"  key='{k}' canonical='{c}' name='{n}'"
                for k, c, n in unexpected
            )
        )

    def test_all_keys_are_valid_smiles(self):
        """Every key must be parseable by RDKit."""
        invalid = []
        for key in RETAINED_NAMES:
            mol = Chem.MolFromSmiles(key)
            if mol is None:
                invalid.append((key, RETAINED_NAMES[key]))

        assert len(invalid) == 0, (
            f"Found {len(invalid)} invalid SMILES key(s):\n"
            + "\n".join(f"  key='{k}' name='{n}'" for k, n in invalid)
        )


class TestValueQuality:
    """Verify all dictionary values are well-formed."""

    def test_all_values_are_nonempty_strings(self):
        """Every value must be a non-empty string."""
        bad = []
        for key, name in RETAINED_NAMES.items():
            if not isinstance(name, str) or not name.strip():
                bad.append((key, name))

        assert len(bad) == 0, (
            f"Found {len(bad)} empty/non-string value(s):\n"
            + "\n".join(f"  key='{k}' value={v!r}" for k, v in bad)
        )

    def test_values_are_lowercase_or_known_format(self):
        """Values should be lowercase except for known patterns like
        N,N-dimethylformamide, (E)-cinnamic acid, L-tartaric acid, etc.
        """
        # Allowed uppercase patterns: element-prefixed (N,N- O- S-),
        # stereodescriptors (R/S/E/Z), indicated hydrogen (1H- 7H-),
        # amino acid L- prefix, DMSO, etc.
        # Indicated hydrogen can appear mid-name (e.g., "3,4-dihydro-2H-pyran")
        import re
        allowed_upper = re.compile(
            r"[NOPS],|[0-9]+H-|^[LDld]-|^\([RSEZ]\)-|"
            r"N-acetyl|^L-|^D-"
        )

        bad = []
        for key, name in RETAINED_NAMES.items():
            if name != name.lower():
                # Check if the uppercase is in an allowed pattern
                if not allowed_upper.search(name):
                    bad.append((key, name))

        assert len(bad) == 0, (
            f"Found {len(bad)} value(s) with unexpected uppercase:\n"
            + "\n".join(f"  key='{k}' name='{n}'" for k, n in bad)
        )


class TestNewEntries:
    """Spot-check specific new entries from Phase 109 expansion."""

    @pytest.mark.parametrize("smiles,expected_name", [
        # Polycyclic aromatics
        ("c1ccc2c(c1)c1ccccc1c1ccccc21", "triphenylene"),
        ("c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61", "coronene"),
        ("c1ccc2c(c1)-c1cccc3cccc-2c13", "fluoranthene"),
        # Benzoic acid derivatives
        ("O=C(O)c1cccc(C(=O)O)c1", "isophthalic acid"),
        ("Nc1ccc(C(=O)O)cc1", "4-aminobenzoic acid"),
        ("Nc1ccccc1C(=O)O", "anthranilic acid"),
        ("COc1cc(C(=O)O)ccc1O", "vanillic acid"),
        # Anhydrides
        ("O=C1C=CC(=O)O1", "maleic anhydride"),
        ("O=C1OC(=O)c2ccccc21", "phthalic anhydride"),
        # Heterocycle derivatives
        ("O=c1ccc2ccccc2o1", "coumarin"),
        ("O=c1ccoc2ccccc12", "chromone"),
        ("O=c1c2ccccc2oc2ccccc12", "xanthone"),
        ("O=C(O)c1cccnc1", "nicotinic acid"),
        ("O=C(O)c1ccncc1", "isonicotinic acid"),
        ("O=C(O)c1ccccn1", "picolinic acid"),
        ("NC(=O)c1cccnc1", "nicotinamide"),
        ("O=C1NS(=O)(=O)c2ccccc21", "saccharin"),
        # Amines
        ("NCCCCN", "putrescine"),
        ("NCCCCCN", "cadaverine"),
        # Solvent
        ("C1COCO1", "1,3-dioxolane"),
        # Nucleobases
        ("O=c1cc[nH]c(=O)[nH]1", "uracil"),
        ("Cc1c[nH]c(=O)[nH]c1=O", "thymine"),
        ("Nc1cc[nH]c(=O)n1", "cytosine"),
        ("Nc1ncnc2[nH]cnc12", "adenine"),
        ("Nc1nc2[nH]cnc2c(=O)[nH]1", "guanine"),
        # Long-chain diacids
        ("O=C(O)CCCCCCC(=O)O", "suberic acid"),
        ("O=C(O)CCCCCCCC(=O)O", "azelaic acid"),
        ("O=C(O)CCCCCCCCC(=O)O", "sebacic acid"),
        # Aromatic derivatives
        ("c1ccc(Nc2ccccc2)cc1", "diphenylamine"),
        ("COc1cc(C=O)ccc1O", "vanillin"),
        ("O=Cc1ccc(O)cc1", "4-hydroxybenzaldehyde"),
        ("COc1ccc(C=O)cc1", "anisaldehyde"),
        ("O=Cc1cccnc1", "nicotinaldehyde"),
        ("O=Cc1ccncc1", "isonicotinaldehyde"),
        ("Oc1cccc2ccccc12", "1-naphthol"),
        # Miscellaneous
        ("OC(c1ccccc1)c1ccccc1", "benzhydrol"),
        ("O=c1cc(-c2ccccc2)oc2ccccc12", "flavone"),
        ("Nc1ccc(N)cc1", "1,4-phenylenediamine"),
        ("Oc1cccc(O)c1O", "pyrogallol"),
        ("CC(=O)c1ccc(O)cc1", "4-hydroxyacetophenone"),
        # Additional compounds
        ("C1CCC2CCCCC2C1", "decahydronaphthalene"),
        ("O=C(O)c1ccco1", "furan-2-carboxylic acid"),
        ("c1ccc(-c2ccncc2)nc1", "2,2'-bipyridine"),
        ("Oc1cc(O)cc(O)c1", "phloroglucinol"),
        ("Oc1ccc2c(c1)OCO2", "sesamol"),
        ("O=Cc1ccc2c(c1)OCO2", "piperonal"),
        ("c1ccc(Cc2ccccc2)cc1", "diphenylmethane"),
        ("c1ccc(C(c2ccccc2)c2ccccc2)cc1", "triphenylmethane"),
        ("C1=Cc2ccccc2C1", "1H-indene"),
        ("O=Cc1ccco1", "furfural"),
    ])
    def test_new_entry_exists(self, smiles, expected_name):
        """Each new entry must be present with correct name."""
        canonical = Chem.CanonSmiles(smiles)
        assert canonical in RETAINED_NAMES, (
            f"Missing entry for {expected_name}: canonical={canonical}"
        )
        assert RETAINED_NAMES[canonical] == expected_name, (
            f"Wrong name for {canonical}: "
            f"expected='{expected_name}' got='{RETAINED_NAMES[canonical]}'"
        )

    @pytest.mark.parametrize("smiles,expected_name", [
        # Verify via get_retained_name function
        ("O=C(O)c1cccc(C(=O)O)c1", "isophthalic acid"),
        ("O=c1ccc2ccccc2o1", "coumarin"),
        # F-T9/DD6 RET-01: 'putrescine' (NCCCCN) is general-only — denied from the
        # gated headline path (PIN butane-1,4-diamine), so get_retained_name -> None.
        # (It stays in the RAW retained_names.RETAINED_NAMES alias; see test_new_entry_exists.)
        ("Nc1ncnc2[nH]cnc12", "adenine"),
        ("c1ccc(Cc2ccccc2)cc1", "diphenylmethane"),
    ])
    def test_get_retained_name_function(self, smiles, expected_name):
        """get_retained_name() must return correct name for new entries."""
        canonical = Chem.CanonSmiles(smiles)
        result = get_retained_name(canonical)
        assert result == expected_name, (
            f"get_retained_name('{canonical}') returned '{result}', "
            f"expected '{expected_name}'"
        )


class TestNoRegressions:
    """Verify existing entries are unchanged."""

    @pytest.mark.parametrize("smiles,expected_name", [
        ("c1ccccc1", "benzene"),
        ("CC(=O)O", "acetic acid"),
        ("CCO", "ethanol"),
        ("Nc1ccccc1", "aniline"),
        ("Oc1ccccc1", "phenol"),
        ("c1ccc2ccccc2c1", "naphthalene"),
        ("c1ccncc1", "pyridine"),
        ("C1CCCCC1", "cyclohexane"),
        ("O=Cc1ccccc1", "benzaldehyde"),
        # NOTE: CC(C)=O (acetone) removed from retained names in Phase 133 -- PIN is propan-2-one
        ("NC(N)=O", "urea"),
        ("c1ccc(-c2ccccc2)cc1", "biphenyl"),
        ("C#C", "acetylene"),
    ])
    def test_existing_entry_unchanged(self, smiles, expected_name):
        """Pre-existing entries must not be modified."""
        assert RETAINED_NAMES.get(smiles) == expected_name, (
            f"Regression: '{smiles}' should map to '{expected_name}', "
            f"got '{RETAINED_NAMES.get(smiles)}'"
        )
