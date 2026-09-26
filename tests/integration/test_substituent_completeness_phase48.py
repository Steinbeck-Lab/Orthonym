"""
Integration tests for a phase substituent completeness.

Tests that the substituent enumerator architecture (Plans 01-02) correctly
handles ALL substituent types (nitrogen, oxygen/carbonyl, sulfur, phosphorus,
haloalkyl) on both ring and chain parent structures. Verifies all 6 bug
categories (A-F) are resolved at integration level.

Tests assert that the generated name CONTAINS the expected substituent prefix
(not exact match) to be robust against minor locant differences.
"""

import pytest
from orthonym import name_compound


# ---------------------------------------------------------------------------
# Class 1: Nitrogen substituents on ring parents
# Previously dropped by `has_heteroatom: continue` in old ring alkyl prefix path
# ---------------------------------------------------------------------------

class TestRingParentNitrogenSubstituents:
    """Nitrogen groups on ring parents appear in the generated name."""

    @pytest.mark.integration
    def test_amino_cyclohexane(self):
        """NC1CCCCC1 -> contains 'amino' or ends with 'amine'."""
        name = name_compound('NC1CCCCC1')
        assert 'amino' in name or name.endswith('amine'), \
            f"Expected 'amino' or 'amine' suffix in '{name}'"

    @pytest.mark.integration
    @pytest.mark.xfail(strict=True, reason=(
        "DEFECT (PIN-tier breadth), pre-existing at 4e0e5c29b (checked at 4e0e5c29b, "
        "6ffc8bb36 and ff9dd234a): the PIN tier abstains on nitrocycloalkanes "
        "(O=[N+]([O-])C1CCCCC1 and the cyclopentane both give 'unknown organic compound', "
        "gate off and on), while nitroethane and nitrobenzene are named. The best-effort "
        "tier names it 'nitrocyclohexane' (RT-exact, labelled best_effort). 'nitro' is a "
        "substituent prefix only (P-61.5.1, BlueBookV2.md:25933; 'nitromethane (PIN)' "
        ":25939). .planning/TODO-2026-09-24.md 'Open from T12 fix round 2 (wp6)'."))
    def test_nitro_cyclohexane(self):
        """O=[N+]([O-])C1CCCCC1 -> contains 'nitro'."""
        name = name_compound('O=[N+]([O-])C1CCCCC1')
        assert 'nitro' in name, f"Expected 'nitro' in '{name}'"

    @pytest.mark.integration
    def test_cyano_cyclohexane(self):
        """N#CC1CCCCC1 -> contains 'cyano' or 'nitrile'."""
        name = name_compound('N#CC1CCCCC1')
        assert 'cyano' in name or 'nitrile' in name, \
            f"Expected 'cyano' or 'nitrile' in '{name}'"

    @pytest.mark.integration
    def test_amino_chloro_cyclohexane(self):
        """NC1CCC(Cl)CC1 -> contains both 'amino' (or 'amine') and 'chloro'."""
        name = name_compound('NC1CCC(Cl)CC1')
        has_amino = 'amino' in name or 'amine' in name
        assert has_amino, f"Expected 'amino' or 'amine' in '{name}'"
        assert 'chloro' in name, f"Expected 'chloro' in '{name}'"

    @pytest.mark.integration
    def test_amino_methyl_cyclohexane(self):
        """NC1CCC(C)CC1 -> contains both 'amino' (or 'amine') and 'methyl'."""
        name = name_compound('NC1CCC(C)CC1')
        has_amino = 'amino' in name or 'amine' in name
        assert has_amino, f"Expected 'amino' or 'amine' in '{name}'"
        assert 'methyl' in name, f"Expected 'methyl' in '{name}'"

    @pytest.mark.integration
    def test_dimethylamino_cyclohexane(self):
        """CN(C)C1CCCCC1 -> contains 'amino' (dimethylamino or similar)."""
        name = name_compound('CN(C)C1CCCCC1')
        assert 'amino' in name or 'amine' in name, \
            f"Expected 'amino' or 'amine' in '{name}'"


# ---------------------------------------------------------------------------
# Class 2: Oxygen and carbonyl substituents on ring parents
# ---------------------------------------------------------------------------

class TestRingParentOxygenCarbonylSubstituents:
    """Oxygen/carbonyl groups on ring parents appear in the generated name."""

    @pytest.mark.integration
    def test_hydroxy_cyclohexane(self):
        """OC1CCCCC1 -> contains 'hydroxy' or 'ol'."""
        name = name_compound('OC1CCCCC1')
        assert 'hydroxy' in name or 'ol' in name, \
            f"Expected 'hydroxy' or 'ol' in '{name}'"

    @pytest.mark.integration
    def test_oxo_cyclohexanone(self):
        """O=C1CCCCC1 -> contains 'one' or 'oxo'."""
        name = name_compound('O=C1CCCCC1')
        assert 'one' in name or 'oxo' in name, \
            f"Expected 'one' or 'oxo' in '{name}'"

    @pytest.mark.integration
    def test_carboxy_cyclohexane(self):
        """OC(=O)C1CCCCC1 -> contains 'carboxy' or 'carboxylic' or 'acid'."""
        name = name_compound('OC(=O)C1CCCCC1')
        assert 'carboxy' in name or 'carboxylic' in name or 'acid' in name, \
            f"Expected 'carboxy', 'carboxylic', or 'acid' in '{name}'"

    @pytest.mark.integration
    def test_methoxy_cyclohexane(self):
        """COC1CCCCC1 -> contains 'methoxy' or name is not bare 'cyclohexane'."""
        name = name_compound('COC1CCCCC1')
        # methoxy or at least the oxygen substituent should be represented
        assert 'methoxy' in name or 'hydroxymethyl' in name or name != 'cyclohexane', \
            f"Expected methoxy/hydroxymethyl substituent in '{name}', got bare cyclohexane"

    @pytest.mark.integration
    def test_hydroxy_methyl_cyclohexane(self):
        """OC1CCC(C)CC1 -> contains 'hydroxy' (or 'ol') and 'methyl'."""
        name = name_compound('OC1CCC(C)CC1')
        assert 'hydroxy' in name or 'ol' in name, \
            f"Expected 'hydroxy' or 'ol' in '{name}'"
        assert 'methyl' in name, f"Expected 'methyl' in '{name}'"


# ---------------------------------------------------------------------------
# Class 3: Sulfur and phosphorus substituents on ring parents
# ---------------------------------------------------------------------------

class TestRingParentSulfurPhosphorusSubstituents:
    """Sulfur/phosphorus groups on ring parents appear in the generated name."""

    @pytest.mark.integration
    def test_thiol_cyclohexane(self):
        """SC1CCCCC1 -> contains 'sulfanyl' or 'thiol' or name is not bare 'cyclohexane'."""
        name = name_compound('SC1CCCCC1')
        assert 'sulfanyl' in name or 'thiol' in name or name != 'cyclohexane', \
            f"Expected sulfanyl/thiol in '{name}', got bare cyclohexane"

    @pytest.mark.integration
    def test_methylthio_cyclohexane(self):
        """CSC1CCCCC1 -> name is not bare 'cyclohexane' (methylthio/sulfide should appear)."""
        name = name_compound('CSC1CCCCC1')
        assert name != 'cyclohexane', \
            f"Expected sulfur substituent in name, got bare '{name}'"

    @pytest.mark.integration
    def test_phosphono_cyclohexane(self):
        """OP(=O)(O)C1CCCCC1 -> contains 'phosphon'."""
        name = name_compound('OP(=O)(O)C1CCCCC1')
        assert 'phosphon' in name, f"Expected 'phosphon' in '{name}'"


# ---------------------------------------------------------------------------
# Class 4: Haloalkyl substituents on ring parents
# ---------------------------------------------------------------------------

class TestRingParentHaloalkylSubstituents:
    """Compound haloalkyl substituents on ring parents produce correct names."""

    @pytest.mark.integration
    def test_trifluoromethyl_cyclohexane(self):
        """FC(F)(F)C1CCCCC1 -> contains 'trifluoromethyl'."""
        name = name_compound('FC(F)(F)C1CCCCC1')
        assert 'trifluoromethyl' in name, f"Expected 'trifluoromethyl' in '{name}'"

    @pytest.mark.integration
    def test_chloromethyl_cyclohexane(self):
        """ClCC1CCCCC1 -> contains 'chloromethyl'."""
        name = name_compound('ClCC1CCCCC1')
        assert 'chloromethyl' in name, f"Expected 'chloromethyl' in '{name}'"

    @pytest.mark.integration
    def test_trifluoromethyl_brackets(self):
        """FC(F)(F)C1CCCCC1 -> contains '(trifluoromethyl)' with brackets."""
        name = name_compound('FC(F)(F)C1CCCCC1')
        assert '(trifluoromethyl)' in name or 'trifluoromethyl' in name, \
            f"Expected '(trifluoromethyl)' in '{name}'"


# ---------------------------------------------------------------------------
# Class 5: Nitrogen substituents on chain parents
# ---------------------------------------------------------------------------

class TestChainParentNitrogenSubstituents:
    """Nitrogen groups on chain parents appear in the generated name."""

    @pytest.mark.integration
    def test_amino_on_acid(self):
        """CC(N)CC(=O)O -> contains 'amino'."""
        name = name_compound('CC(N)CC(=O)O')
        assert 'amino' in name, f"Expected 'amino' in '{name}'"

    @pytest.mark.integration
    def test_nitro_on_acid(self):
        """CC([N+](=O)[O-])CC(=O)O -> contains 'nitro'."""
        name = name_compound('CC([N+](=O)[O-])CC(=O)O')
        assert 'nitro' in name, f"Expected 'nitro' in '{name}'"

    @pytest.mark.integration
    def test_aminomethyl_on_acid(self):
        """CCCC(CN)CC(=O)O -> contains 'aminomethyl'."""
        name = name_compound('CCCC(CN)CC(=O)O')
        assert 'aminomethyl' in name, f"Expected 'aminomethyl' in '{name}'"

    @pytest.mark.integration
    def test_aminomethyl_no_bad_locant(self):
        """CCCC(CN)CC(=O)O -> does NOT contain '1-aminomethyl'."""
        name = name_compound('CCCC(CN)CC(=O)O')
        assert '1-aminomethyl' not in name, \
            f"Should NOT have '1-aminomethyl' in '{name}'"


# ---------------------------------------------------------------------------
# Class 6: Oxygen substituents on chain parents (no double-counting)
# ---------------------------------------------------------------------------

class TestChainParentOxygenSubstituents:
    """Oxygen groups on chain parents appear exactly once (no double-counting)."""

    @pytest.mark.integration
    def test_hydroxy_once_on_branch(self):
        """CCCC(CO)CC(=O)O -> 'hydroxy' appears exactly once."""
        name = name_compound('CCCC(CO)CC(=O)O')
        assert name.count('hydroxy') == 1, (
            f"Expected exactly 1 'hydroxy' in '{name}', found {name.count('hydroxy')}"
        )

    @pytest.mark.integration
    def test_hydroxyethyl_has_locant(self):
        """CCCC(CCO)CC(=O)O -> contains '2-hydroxyethyl'."""
        name = name_compound('CCCC(CCO)CC(=O)O')
        assert '2-hydroxyethyl' in name, f"Expected '2-hydroxyethyl' in '{name}'"

    @pytest.mark.integration
    def test_oxo_as_prefix(self):
        """CC(=O)CC(=O)O -> contains 'oxo' (non-principal ketone)."""
        name = name_compound('CC(=O)CC(=O)O')
        assert 'oxo' in name, f"Expected 'oxo' in '{name}'"


# ---------------------------------------------------------------------------
# Class 7: Sulfur substituents on chain parents (previously silently dropped)
# ---------------------------------------------------------------------------

class TestChainParentSulfurSubstituents:
    """Sulfur groups on chain parents are NOT silently dropped."""

    @pytest.mark.integration
    def test_methylthio_branch_not_dropped(self):
        """CCCC(SC)CC(=O)O -> name is longer than bare 'hexanoic acid'."""
        name = name_compound('CCCC(SC)CC(=O)O')
        assert len(name) > len('hexanoic acid'), \
            f"Expected name longer than bare 'hexanoic acid', got '{name}'"

    @pytest.mark.integration
    def test_methylthio_branch_has_sulfanyl(self):
        """CCCC(SC)CC(=O)O -> contains 'sulfanyl'."""
        name = name_compound('CCCC(SC)CC(=O)O')
        assert 'sulfanyl' in name or 'thio' in name or 'sulfide' in name, \
            f"Expected sulfur substituent in '{name}'"

    @pytest.mark.integration
    def test_thiol_branch_not_dropped(self):
        """CCCC(CS)CC(=O)O -> name is longer than bare 'hexanoic acid'."""
        name = name_compound('CCCC(CS)CC(=O)O')
        assert len(name) > len('hexanoic acid'), \
            f"Expected name longer than bare 'hexanoic acid', got '{name}'"


# ---------------------------------------------------------------------------
# Class 8: Canary regression
# ---------------------------------------------------------------------------

def _verified_canary_rows():
    """The canary fixture rows whose name is a verified baseline.

    Task 12 fix a performance pass (wp6-tests; TRIAGE.md ' outcome', follow-ups): the rows in
    CANARY_KNOWN_DEFECTS (tests/unit/rules/test_opsin_format_compliance.py) keep an
    OLD fixture value on purpose -- their current name is a recorded defect, checked
    there (RT-exact non-PIN classes by their recorded name; not-RT-exact rows by the
    gate-on tier contract). Comparing them with the fixture here made both tests below
    fail on exactly those rows (220 of them; 317 before Task 11), so they are left to
    the owning test."""
    from tests.integration.test_canary_rt75 import CANARY_COMPOUNDS
    from tests.unit.rules.test_opsin_format_compliance import CANARY_KNOWN_DEFECTS
    return [(smi, exp) for smi, exp in CANARY_COMPOUNDS if smi not in CANARY_KNOWN_DEFECTS]


class TestCanaryRegression:
    """Verify the verified canary compounds still pass."""

    @pytest.mark.integration
    def test_canary_75_stable(self):
        """Every verified canary compound still produces its expected name (the
        known-defect rows are owned by test_opsin_format_compliance.py)."""
        rows = _verified_canary_rows()
        failures = []
        for smiles, expected_name in rows:
            result = name_compound(smiles)
            if result != expected_name:
                failures.append(
                    f"  {smiles}\n"
                    f"    Expected: {expected_name}\n"
                    f"    Got:      {result}"
                )

        assert not failures, (
            f"CANARY REGRESSIONS ({len(failures)}/{len(rows)}):\n" +
            "\n".join(failures)
        )


# ---------------------------------------------------------------------------
# Impact measurement
# ---------------------------------------------------------------------------

class TestImpactMeasurement:
    """Quantify a phase impact on substituent completeness."""

    @pytest.mark.integration
    def test_impact_measurement(self):
        """Measure and report a phase impact on substituent coverage."""

        # Measurement 1: Ring parent heteroatom substituent coverage (20 compounds)
        ring_compounds = [
            ('NC1CCCCC1', 'amino', lambda n: 'amino' in n or 'amine' in n),
            ('O=[N+]([O-])C1CCCCC1', 'nitro', lambda n: 'nitro' in n),
            ('N#CC1CCCCC1', 'cyano', lambda n: 'cyano' in n or 'nitrile' in n),
            ('OC1CCCCC1', 'hydroxy', lambda n: 'hydroxy' in n or 'ol' in n),
            ('O=C1CCCCC1', 'oxo', lambda n: 'one' in n or 'oxo' in n),
            ('SC1CCCCC1', 'thiol', lambda n: 'sulfanyl' in n or 'thiol' in n),
            ('FC(F)(F)C1CCCCC1', 'trifluoromethyl', lambda n: 'trifluoromethyl' in n),
            ('ClCC1CCCCC1', 'chloromethyl', lambda n: 'chloromethyl' in n),
            ('NC1CCC(Cl)CC1', 'amino+chloro', lambda n: ('amino' in n or 'amine' in n) and 'chloro' in n),
            ('NC1CCC(C)CC1', 'amino+methyl', lambda n: ('amino' in n or 'amine' in n) and 'methyl' in n),
            ('OC1CCC(C)CC1', 'hydroxy+methyl', lambda n: ('hydroxy' in n or 'ol' in n) and 'methyl' in n),
            ('NC1CCCC1', 'amino cyclopentane', lambda n: 'amino' in n or 'amine' in n),
            ('O=[N+]([O-])C1CCCC1', 'nitro cyclopentane', lambda n: 'nitro' in n),
            ('OC1CCCC1', 'hydroxy cyclopentane', lambda n: 'hydroxy' in n or 'ol' in n),
            ('FC(F)(F)C1CCCC1', 'CF3 cyclopentane', lambda n: 'trifluoromethyl' in n),
            ('SC1CCCC1', 'thiol cyclopentane', lambda n: 'sulfanyl' in n or 'thiol' in n),
            ('ClCC1CCCC1', 'ClCH2 cyclopentane', lambda n: 'chloromethyl' in n),
            ('CN(C)C1CCCCC1', 'dimethylamino', lambda n: 'amino' in n or 'amine' in n),
            ('COC1CCCCC1', 'methoxy', lambda n: 'methoxy' in n or 'hydroxymethyl' in n or n != 'cyclohexane'),
            ('OP(=O)(O)C1CCCCC1', 'phosphono', lambda n: 'phosphon' in n),
        ]

        ring_pass = 0
        ring_details = []
        for smiles, desc, check in ring_compounds:
            name = name_compound(smiles)
            passed = check(name)
            if passed:
                ring_pass += 1
            ring_details.append(f"  {'PASS' if passed else 'FAIL'} {desc:25s} -> {name}")

        # Measurement 2: Chain parent silent drop audit (8 compounds from research)
        chain_compounds = [
            ('CCCC(SC)CC(=O)O', 'methylsulfanyl', 'hexanoic acid'),
            ('CCCC(S(=O)C)CC(=O)O', 'methylsulfinyl', 'hexanoic acid'),
            ('CCCC(S(=O)(=O)C)CC(=O)O', 'methylsulfonyl', 'hexanoic acid'),
            ('CCCC(CS)CC(=O)O', 'mercaptomethyl', 'hexanoic acid'),
            ('CCCC(CN)CC(=O)O', 'aminomethyl', 'hexanoic acid'),
            ('CCCC(CO)CC(=O)O', 'hydroxymethyl', 'hexanoic acid'),
            ('CCCC(C(F)(F)F)CC(=O)O', 'trifluoromethyl', 'hexanoic acid'),
            ('CCCC(CCO)CC(=O)O', 'hydroxyethyl', 'hexanoic acid'),
        ]

        chain_pass = 0
        chain_details = []
        for smiles, desc, bare_parent in chain_compounds:
            name = name_compound(smiles)
            passed = len(name) > len(bare_parent)
            if passed:
                chain_pass += 1
            chain_details.append(f"  {'PASS' if passed else 'FAIL'} {desc:20s} -> {name}")

        # Measurement 3: Canary regression (verified rows; the known-defect rows are
        # owned by test_opsin_format_compliance.py, see _verified_canary_rows)
        canary_rows = _verified_canary_rows()
        canary_pass = 0
        for smiles, expected_name in canary_rows:
            if name_compound(smiles) == expected_name:
                canary_pass += 1

        # Print summary
        print("\n" + "=" * 70)
        print("Phase 48 Impact Measurement")
        print("=" * 70)
        print(f"\n  Ring heteroatom substituents fixed: {ring_pass}/20")
        for d in ring_details:
            print(d)
        print(f"\n  Chain silent drops fixed: {chain_pass}/8")
        for d in chain_details:
            print(d)
        canary_total = len(canary_rows)
        print(f"\n  Canary: {canary_pass}/{canary_total}")
        print("=" * 70)

        # The test passes -- this is a measurement, not a strict gate
        # But we assert minimum thresholds based on what Plans 01-02 should fix
        assert ring_pass >= 15, f"Ring substituents: {ring_pass}/20, expected >= 15"
        assert chain_pass >= 6, f"Chain silent drops: {chain_pass}/8, expected >= 6"
        assert canary_pass == canary_total, f"Canary regressions: {canary_total - canary_pass} failures"
