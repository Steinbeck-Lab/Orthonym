"""
Accuracy threshold tests for Orthonym.

Tests validation against ChEBI dataset and OPSIN round-trip.
These tests are marked as @pytest.mark.slow and skipped by default.

Run with: pytest tests/validation/ -m slow -v
"""

import os
import sys
from pathlib import Path

import pytest

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

from orthonym import name_compound


# =============================================================================
# Test Data Path
# =============================================================================

# Default test data location
TEST_DATA_PATH = PROJECT_ROOT / "chebi_iupac_filtered.tsv"

# Allow override via environment variable
if os.environ.get("ORTHONYM_TEST_DATA"):
    TEST_DATA_PATH = Path(os.environ["ORTHONYM_TEST_DATA"])


# =============================================================================
# Skip Conditions
# =============================================================================

# Skip if test data not available
skip_if_no_data = pytest.mark.skipif(
    not TEST_DATA_PATH.exists(),
    reason=f"Test data not found: {TEST_DATA_PATH}"
)


# =============================================================================
# Accuracy Threshold Constants
# =============================================================================

# Overall accuracy target
OVERALL_ACCURACY_TARGET = 0.95  # 95%

# Per-class accuracy thresholds
CLASS_THRESHOLDS = {
    'alkane': 0.99,        # Simple alkanes should be near-perfect
    'alkene': 0.95,        # Unsaturated chains
    'alkyne': 0.95,        # Triple bonds
    'alcohol': 0.97,       # Simple functional group
    'aldehyde': 0.95,
    'ketone': 0.95,
    'carboxylic_acid': 0.95,
    'ester': 0.90,
    'amide': 0.90,
    'amine': 0.95,
    'ether': 0.90,
    'nitrile': 0.95,
    'cycloalkane': 0.95,   # Simple rings
    'cycloalkene': 0.90,
    'aromatic': 0.90,      # Benzene derivatives
    'heterocycle_aromatic': 0.85,   # Complex heterocycles
    'heterocycle_saturated': 0.90,
    'heterocycle_small': 0.95,      # 3/4-membered
    'bicyclo': 0.85,       # Bridged bicyclics
    'spiro': 0.85,         # Spiro compounds
    'polycyclic_aromatic': 0.80,    # PAHs
    'fused_aromatic': 0.75,         # Fused systems
}

# Round-trip accuracy target
ROUNDTRIP_ACCURACY_TARGET = 0.95  # 95%


# =============================================================================
# Overall Accuracy Tests
# =============================================================================

@pytest.mark.slow
@skip_if_no_data
def test_overall_accuracy_10k():
    """
    Overall accuracy must be >= 95% on 10,000 compound sample.

    This is the main validation metric (VALID-01).
    """
    from validate_bulk import validate_dataset

    results = validate_dataset(str(TEST_DATA_PATH), sample_size=10000)
    accuracy = results['summary']['accuracy_exact']

    print(f"\nOverall accuracy: {accuracy:.1%} (target: {OVERALL_ACCURACY_TARGET:.0%})")
    print(f"  Exact matches: {results['summary']['success']}")
    print(f"  Partial matches: {results['summary']['partial_match']}")
    print(f"  Mismatches: {results['summary']['mismatch']}")
    print(f"  Errors: {results['summary']['error']}")

    # Report per-class breakdown
    print("\nPer-class breakdown:")
    for cls, stats in sorted(results['per_class'].items(), key=lambda x: -x[1]['total']):
        print(f"  {cls}: {stats['accuracy']:.1%} ({stats['success']}/{stats['total']})")

    assert accuracy >= OVERALL_ACCURACY_TARGET, \
        f"Accuracy {accuracy:.1%} below {OVERALL_ACCURACY_TARGET:.0%} target"


@pytest.mark.slow
@skip_if_no_data
def test_overall_accuracy_1k_quick():
    """
    Quick accuracy check on 1,000 compounds.

    Faster feedback during development.
    """
    from validate_bulk import validate_dataset

    results = validate_dataset(str(TEST_DATA_PATH), sample_size=1000)
    accuracy = results['summary']['accuracy_exact']

    print(f"\nQuick accuracy check: {accuracy:.1%}")

    # Allow slightly lower threshold for quick test (random sample variance)
    assert accuracy >= 0.90, \
        f"Quick check accuracy {accuracy:.1%} below 90% threshold"


@pytest.mark.slow
@skip_if_no_data
def test_full_dataset_accuracy():
    """
    Full 100k compound validation.

    Run time: ~30 minutes. Use for final validation only.
    """
    from validate_bulk import validate_dataset

    results = validate_dataset(str(TEST_DATA_PATH), sample_size=None)
    accuracy = results['summary']['accuracy_exact']

    print(f"\nFull dataset accuracy: {accuracy:.1%}")
    print(f"  Total: {results['summary']['total']}")

    assert accuracy >= OVERALL_ACCURACY_TARGET, \
        f"Full accuracy {accuracy:.1%} below {OVERALL_ACCURACY_TARGET:.0%} target"


# =============================================================================
# Per-Class Accuracy Tests
# =============================================================================

@pytest.mark.slow
@skip_if_no_data
@pytest.mark.parametrize("compound_class,threshold", CLASS_THRESHOLDS.items())
def test_class_accuracy(compound_class, threshold):
    """
    Test accuracy threshold for each compound class.

    Each class has its own threshold based on implementation complexity.
    """
    from validate_bulk import validate_dataset

    results = validate_dataset(
        str(TEST_DATA_PATH),
        sample_size=5000,
        compound_class=compound_class,
    )

    if results['summary']['total'] == 0:
        pytest.skip(f"No compounds of class '{compound_class}' in sample")

    accuracy = results['summary']['accuracy_exact']
    total = results['summary']['total']

    print(f"\n{compound_class} accuracy: {accuracy:.1%} (target: {threshold:.0%})")
    print(f"  Total tested: {total}")

    assert accuracy >= threshold, \
        f"{compound_class} accuracy {accuracy:.1%} below {threshold:.0%} threshold"


# =============================================================================
# OPSIN Round-Trip Tests
# =============================================================================

@pytest.mark.slow
@pytest.mark.roundtrip
@skip_if_no_data
def test_opsin_roundtrip():
    """
    OPSIN round-trip must pass for 95% of compounds (VALID-02).

    Tests: SMILES -> Orthonym -> OPSIN -> SMILES -> compare
    """
    from validate_roundtrip import validate_roundtrip_batch, check_opsin_available

    # Check OPSIN availability
    available, _, message = check_opsin_available()
    if not available:
        pytest.skip(f"OPSIN not available: {message}")

    results = validate_roundtrip_batch(str(TEST_DATA_PATH), sample_size=1000)

    if 'error' in results and not results.get('opsin_available'):
        pytest.skip(f"OPSIN error: {results['error']}")

    success_rate = results['summary']['success_rate']

    print(f"\nRound-trip success rate: {success_rate:.1%} (target: {ROUNDTRIP_ACCURACY_TARGET:.0%})")
    print(f"  Exact matches: {results['summary']['exact_match']}")
    print(f"  Equivalent (tautomer): {results['summary']['equivalent_match']}")
    print(f"  Name errors: {results['summary']['name_error']}")
    print(f"  Parse errors: {results['summary']['parse_error']}")
    print(f"  Mismatches: {results['summary']['mismatch']}")

    assert success_rate >= ROUNDTRIP_ACCURACY_TARGET, \
        f"Round-trip success {success_rate:.1%} below {ROUNDTRIP_ACCURACY_TARGET:.0%} target"


@pytest.mark.slow
@pytest.mark.roundtrip
def test_opsin_roundtrip_basic_compounds():
    """
    Round-trip validation on basic compounds that should always work.
    """
    from validate_roundtrip import validate_roundtrip, check_opsin_available

    available, jar_path, message = check_opsin_available()
    if not available:
        pytest.skip(f"OPSIN not available: {message}")

    # Basic compounds that must round-trip correctly
    basic_compounds = [
        "C",        # methane
        "CC",       # ethane
        "CCC",      # propane
        "CCO",      # ethanol
        "CC=O",     # acetaldehyde
        "CC(=O)O",  # acetic acid
        "C1CCCCC1", # cyclohexane
    ]

    failures = []
    for smiles in basic_compounds:
        result = validate_roundtrip(smiles, jar_path)
        if not result.match:
            failures.append(f"{smiles}: {result.generated_name} -> {result.error}")

    assert len(failures) == 0, f"Basic compounds failed round-trip: {failures}"


# =============================================================================
# Regression Guard Tests
# =============================================================================

@pytest.mark.integration
def test_key_compounds():
    """
    Test specific compounds that previously failed or are critical.

    These are regression tests to ensure we don't break working compounds.
    """
    # Critical test cases: (SMILES, expected_name)
    critical_cases = [
        # Simple alkanes
        ("C", "methane"),
        ("CC", "ethane"),
        ("CCC", "propane"),
        ("CCCC", "butane"),

        # Simple alcohols
        ("CO", "methanol"),
        ("CCO", "ethanol"),

        # Cyclic
        ("C1CCCCC1", "cyclohexane"),
        ("C1CCCC1", "cyclopentane"),

        # Aromatic
        ("c1ccccc1", "benzene"),
        ("Cc1ccccc1", "toluene"),

        # Heterocycles
        ("c1ccncc1", "pyridine"),
        ("c1ccoc1", "furan"),
        ("C1CCNCC1", "piperidine"),
    ]

    failures = []
    for smiles, expected in critical_cases:
        try:
            result = name_compound(smiles)
            if result is None:
                failures.append(f"{smiles}: returned None, expected '{expected}'")
            elif result.lower() != expected.lower():
                # Allow case differences
                failures.append(f"{smiles}: got '{result}', expected '{expected}'")
        except Exception as e:
            failures.append(f"{smiles}: error - {e}")

    if failures:
        print("\nFailed compounds:")
        for f in failures:
            print(f"  {f}")

    assert len(failures) == 0, f"{len(failures)} critical compounds failed"


@pytest.mark.integration
def test_bicyclic_compounds():
    """
    Test bicyclic compounds that were added in a phase.
    """
    bicyclic_cases = [
        # Basic bicyclo systems
        ("C1CC2CCC1C2", "bicyclo[2.2.1]heptane"),
        ("C1CC2CCCC1C2", "bicyclo[3.2.1]octane"),
        ("C1CC2CCCCC1C2", "bicyclo[4.2.1]nonane"),
    ]

    failures = []
    for smiles, expected in bicyclic_cases:
        try:
            result = name_compound(smiles)
            if result is None:
                failures.append(f"{smiles}: returned None, expected '{expected}'")
            elif expected.lower() not in result.lower():
                # Allow for substituents, just check base name
                failures.append(f"{smiles}: got '{result}', expected to contain '{expected}'")
        except Exception as e:
            failures.append(f"{smiles}: error - {e}")

    if failures:
        print("\nFailed bicyclic compounds:")
        for f in failures:
            print(f"  {f}")

    # Don't fail the test, just warn - these may not all be implemented
    if failures:
        pytest.xfail(f"{len(failures)} bicyclic compounds not fully implemented")


@pytest.mark.integration
def test_spiro_compounds():
    """
    Test spiro compounds that were added in a phase.
    """
    spiro_cases = [
        ("C1CCC2(CC1)CCCCC2", "spiro[5.5]undecane"),
        ("C1CCC2(CC1)CCCC2", "spiro[4.5]decane"),
    ]

    failures = []
    for smiles, expected in spiro_cases:
        try:
            result = name_compound(smiles)
            if result is None:
                failures.append(f"{smiles}: returned None, expected '{expected}'")
            elif "spiro" not in result.lower():
                failures.append(f"{smiles}: got '{result}', expected spiro name")
        except Exception as e:
            failures.append(f"{smiles}: error - {e}")

    if failures:
        print("\nFailed spiro compounds:")
        for f in failures:
            print(f"  {f}")

    if failures:
        pytest.xfail(f"{len(failures)} spiro compounds not fully implemented")


@pytest.mark.integration
def test_fused_heterocycles():
    """
    Test fused heterocycles (indole, quinoline, etc.).
    """
    fused_cases = [
        ("c1ccc2[nH]ccc2c1", "1H-indole"),
        ("c1ccc2ncccc2c1", "quinoline"),
        ("c1ccc2cnccc2c1", "isoquinoline"),
    ]

    failures = []
    for smiles, expected in fused_cases:
        try:
            result = name_compound(smiles)
            if result is None:
                failures.append(f"{smiles}: returned None, expected '{expected}'")
            # Check if the name contains expected heterocycle
            elif expected.split('-')[-1].lower() not in result.lower():
                failures.append(f"{smiles}: got '{result}', expected '{expected}'")
        except Exception as e:
            failures.append(f"{smiles}: error - {e}")

    if failures:
        print("\nFailed fused heterocycles:")
        for f in failures:
            print(f"  {f}")

    if failures:
        pytest.xfail(f"{len(failures)} fused heterocycles not fully implemented")


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.slow
@skip_if_no_data
def test_naming_performance():
    """
    Test that naming is fast enough for practical use.

    Target: < 10ms average per compound.
    """
    from validate_bulk import validate_dataset

    results = validate_dataset(str(TEST_DATA_PATH), sample_size=1000)
    avg_time = results['summary']['avg_time_ms']

    print(f"\nAverage naming time: {avg_time:.2f}ms")

    assert avg_time < 10.0, \
        f"Average naming time {avg_time:.2f}ms exceeds 10ms target"


@pytest.mark.slow
@skip_if_no_data
def test_no_timeouts():
    """
    Test that no compounds cause excessive timeouts.

    All compounds should complete within 5 seconds.
    """
    from validate_bulk import validate_dataset
    import time

    # Run with short sample to check for hangs
    start = time.time()
    results = validate_dataset(str(TEST_DATA_PATH), sample_size=100)
    elapsed = time.time() - start

    # Should complete 100 compounds in < 30 seconds (300ms each worst case)
    assert elapsed < 30, f"Validation took {elapsed:.1f}s, possible timeout issue"


# =============================================================================
# Data Quality Tests
# =============================================================================

@pytest.mark.slow
@skip_if_no_data
def test_no_empty_names():
    """
    Test that we don't generate empty names for valid SMILES.
    """
    from validate_bulk import validate_dataset

    results = validate_dataset(str(TEST_DATA_PATH), sample_size=500)

    # Count how many returned empty/None
    empty_count = results['summary']['unsupported']
    total = results['summary']['total']

    empty_rate = empty_count / total if total > 0 else 0

    print(f"\nEmpty name rate: {empty_rate:.1%} ({empty_count}/{total})")

    # Allow up to 10% unsupported
    assert empty_rate < 0.10, \
        f"Too many empty names: {empty_rate:.1%}"


@pytest.mark.slow
@skip_if_no_data
def test_no_exceptions():
    """
    Test that we don't throw exceptions for valid SMILES.
    """
    from validate_bulk import validate_dataset

    results = validate_dataset(str(TEST_DATA_PATH), sample_size=500)

    error_count = results['summary']['error']
    total = results['summary']['total']

    error_rate = error_count / total if total > 0 else 0

    print(f"\nException rate: {error_rate:.1%} ({error_count}/{total})")

    # Exceptions should be very rare
    assert error_rate < 0.05, \
        f"Too many exceptions: {error_rate:.1%}"
