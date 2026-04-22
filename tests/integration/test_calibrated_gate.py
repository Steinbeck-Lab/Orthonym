"""Integration tests for calibrated coverage gate weights.

Verifies that the calibrated FACTOR_WEIGHTS in coverage_scoring.py produce
correct behavior: proper sum, positivity, derivation documentation, and
regression anchors for key molecule types.
"""
import inspect
import pytest
from orthonym import name_compound
from orthonym.assembly.coverage_scoring import FACTOR_WEIGHTS


# ---------------------------------------------------------------------------
# Weight validity tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_weights_sum_to_one():
    """FACTOR_WEIGHTS must sum to 1.0 (within floating point tolerance)."""
    total = sum(FACTOR_WEIGHTS.values())
    assert abs(total - 1.0) < 0.01, f"FACTOR_WEIGHTS sum to {total}, expected 1.0"


@pytest.mark.integration
def test_weights_all_positive():
    """All calibrated factor weights must be > 0.

    Phase 145.1 ISS-004 update: 'parent_correctness' is the 5th factor
    (scaffolded with weight=0.0 for byte-identical safety per D-14;
    Phase 146 raises it to ~0.35 after train/test calibration). The
    scaffolding weight is excluded from this positivity check until
    Phase 146 calibrates it.
    """
    for key, val in FACTOR_WEIGHTS.items():
        if key == 'parent_correctness':
            # Phase 145.1 scaffolding key (weight=0.0 by D-14 byte-identical
            # contract). Phase 146 calibrates and removes this exception.
            assert val == 0.0, (
                f"parent_correctness must be exactly 0.0 in Phase 145.1 "
                f"scaffolding (D-14 byte-identical proof); got {val}"
            )
            continue
        assert val > 0, f"Weight '{key}' is {val}, must be > 0"


@pytest.mark.integration
def test_weights_have_derivation_comment():
    """coverage_scoring.py must contain calibration script derivation comment."""
    import orthonym.assembly.coverage_scoring as mod
    source = inspect.getsource(mod)
    assert "calibrate_coverage_gate.py" in source, (
        "Missing derivation comment referencing calibrate_coverage_gate.py"
    )


@pytest.mark.integration
def test_weights_have_five_keys():
    """FACTOR_WEIGHTS must have exactly 5 keys after Phase 145.1 scaffolding.

    Phase 145.1 ISS-004 update: 5th key 'parent_correctness' added at LAST
    position (Risk 2 mitigation -- preserves dict iteration order in
    compute_confidence's sum-loop; weight=0.0 keeps confidence values
    byte-identical per D-14).
    """
    expected = {
        'ratio', 'atom_coverage', 'fg_recognition',
        'substituent_completeness', 'parent_correctness',
    }
    assert set(FACTOR_WEIGHTS.keys()) == expected


# ---------------------------------------------------------------------------
# Confidence range tests
# ---------------------------------------------------------------------------

DIVERSE_SMILES = [
    "CC",                          # ethane (simple)
    "CCC",                         # propane
    "CCO",                         # ethanol
    "CC(=O)O",                     # acetic acid
    "c1ccccc1",                    # benzene
    "c1ccncc1",                    # pyridine
    "C1CCCCC1",                    # cyclohexane
    "CC(C)CC",                     # 2-methylbutane
    "OC(=O)CCCC(=O)O",            # glutaric acid
    "c1ccc2[nH]ccc2c1",           # 1H-indole
    "CC(=O)Nc1ccccc1",            # acetanilide
    "OC(=O)c1ccccc1",             # benzoic acid
    "Oc1ccccc1",                   # phenol
    "CC(C)(C)O",                   # 2-methylpropan-2-ol
    "CC=CC",                       # but-2-ene
    "C#CC",                        # propyne
    "CCCCCCCCCC",                  # decane
    "ClCCCl",                      # 1,2-dichloroethane
    "CCOC(=O)C",                   # ethyl acetate
    "CCN",                         # ethylamine
]


@pytest.mark.integration
def test_confidence_range_diverse_molecules():
    """Confidence scores for diverse molecules must be in [0.0, 1.0]."""
    for smi in DIVERSE_SMILES:
        result = name_compound(smi, include_confidence=True)
        assert isinstance(result, dict), f"Expected dict for {smi}"
        conf = result.get('confidence', -1)
        assert 0.0 <= conf <= 1.0, (
            f"Confidence {conf} out of range for {smi}"
        )


@pytest.mark.integration
def test_simple_molecules_high_confidence():
    """Simple molecules (ethane, propane, ethanol, acetic acid) should get confidence >= 0.7."""
    simple = ["CC", "CCC", "CCO", "CC(=O)O"]
    for smi in simple:
        result = name_compound(smi, include_confidence=True)
        conf = result.get('confidence', 0)
        assert conf >= 0.7, (
            f"Simple molecule {smi} has confidence {conf}, expected >= 0.7"
        )


# ---------------------------------------------------------------------------
# Name stability / regression anchors
# ---------------------------------------------------------------------------

STABILITY_ANCHORS = [
    # (SMILES, expected_name) -- curated mix of chain, ring, heterocycle, fused
    ("CC", "ethane"),
    ("CCC", "propane"),
    ("CCCC", "butane"),
    ("CCO", "ethanol"),
    ("CC(=O)O", "acetic acid"),
    ("c1ccccc1", "benzene"),
    ("c1ccncc1", "pyridine"),
    ("C1CCCCC1", "cyclohexane"),
    ("C(=O)O", "formic acid"),
    ("CC=O", "acetaldehyde"),
    ("CC(C)O", "propan-2-ol"),
    ("ClCCCl", "1,2-dichloroethane"),
    ("Oc1ccccc1", "phenol"),
    ("Cc1ccccc1", "toluene"),
    ("Nc1ccccc1", "aniline"),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected", STABILITY_ANCHORS,
                         ids=[s for s, _ in STABILITY_ANCHORS])
def test_name_stability_after_calibration(smiles, expected):
    """Curated molecules must produce exact expected names."""
    name = name_compound(smiles)
    assert name == expected, (
        f"Stability anchor failed: {smiles} -> '{name}' (expected '{expected}')"
    )


# ---------------------------------------------------------------------------
# Canary subset spot check
# ---------------------------------------------------------------------------

CANARY_SUBSET = [
    ("CC(=O)Oc1ccccc1C(=O)O", "aspirin"),
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "2-(4-isobutylphenyl)propanoic acid"),
    ("OC(=O)CC(O)(CC(=O)O)C(=O)O", "3-hydroxy-3-(hydroxymethyl)pentanetrioic acid"),
    ("OCCc1ccc(O)c(O)c1", "2-(3,4-dihydroxyphenyl)ethan-1-ol"),
    ("CC12CCC3C(CCC4CC(=O)CCC43C)C1CCC2O", "17-hydroxyandrostan-3-one"),
    ("OC(=O)c1ccc(N)cc1", "4-aminobenzoic acid"),
    ("CC(=O)Nc1ccc(O)cc1", "1-anilinoethanamide"),
    # Phase 125: old expected name "1-(N,N-diethylamino)-4-phenylbenzene" was
    # incorrect -- it dropped the N=N azo linkage because _check_retained_substituent
    # wrongly returned "phenyl" for the N=N-phenyl fragment.  The fix (counting all
    # non-ring heavy atoms, not just carbons) correctly rejects that shortcut.
    # Proper azo naming support is deferred.
    ("CCN(CC)c1ccc(N=Nc2ccccc2)cc1", "unknown organic compound"),
    ("CC(=O)O", "acetic acid"),
    ("Cc1ccc(O)c(C(C)C)c1", "2-isopropyl-4-methylphenol"),  # ASML-13: phenol suffix routing
    ("OC(=O)/C=C\\C(=O)O", "(2Z)-but-2-enedioic acid"),
    ("OC(=O)c1ccccc1O", "2-hydroxybenzoic acid"),
    ("c1ccc2c(c1)cc1ccc3ccccc3c1c2", "benz[a]anthracene"),
    ("OCCO", "ethylene glycol"),
    ("OC(=O)CCCCC(=O)O", "hexanedioic acid"),
    ("OC(=O)CCC(=O)O", "butanedioic acid"),
    ("CC(O)=O", "acetic acid"),
    ("CCC(=O)O", "propanoic acid"),
    ("c1ccncc1", "pyridine"),
    ("c1ccc(cc1)O", "phenol"),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected", CANARY_SUBSET,
                         ids=[f"canary_{i}" for i in range(len(CANARY_SUBSET))])
def test_canary_subset_stable(smiles, expected):
    """Spot-check 20 canary compounds for exact name match."""
    name = name_compound(smiles)
    assert name == expected, (
        f"Canary failed: {smiles} -> '{name}' (expected '{expected}')"
    )


# ---------------------------------------------------------------------------
# No empty names test
# ---------------------------------------------------------------------------

# Molecules that historically triggered DROP-20 (coverage gate reject)
DROP20_MOLECULES = [
    "OC(=O)c1cc(O)c(O)c(O)c1",            # gallic acid
    "CC(=O)Oc1ccccc1C(=O)O",               # aspirin
    "OC(=O)/C=C/c1ccc(O)c(O)c1",           # caffeic acid
    "Oc1cc(O)c2c(c1)OC(c1ccc(O)c(O)c1)CC2=O",  # eriodictyol
    "c1ccc2[nH]ccc2c1",                     # 1H-indole
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",          # ibuprofen
    "CC(O)C(=O)O",                          # lactic acid
    "OC(=O)CC(O)(CC(=O)O)C(=O)O",          # citric acid
    "CC(=O)Nc1ccc(O)cc1",                   # paracetamol
    "OC(=O)c1ccccc1O",                      # salicylic acid
    "NCCc1ccc(O)c(O)c1",                    # dopamine
    "OC(=O)CCC(=O)O",                       # succinic acid
    "OC(=O)CCCC(=O)O",                      # glutaric acid
    "OC(=O)CCCCC(=O)O",                     # adipic acid
    "OCCO",                                 # ethylene glycol
    "OC(=O)/C=C\\C(=O)O",                   # maleic acid
    "CC(=O)O",                              # acetic acid
    "CCC(=O)O",                             # propionic acid
    "CCCC(=O)O",                            # butyric acid
    "CC(C)(O)CC(=O)O",                      # mevalonic acid
    "OC(=O)c1ccc(N)cc1",                    # 4-aminobenzoic acid
    "c1ccncc1",                             # pyridine
    "Cc1ccccc1",                            # toluene
    "c1ccc2ccccc2c1",                       # naphthalene
    "O=C1CCCCC1",                           # cyclohexanone
    "OC1CCCCC1",                            # cyclohexanol
    "O=Cc1ccccc1",                          # benzaldehyde
    "CC(=O)c1ccccc1",                       # acetophenone
    "Nc1ccccc1",                            # aniline
    "c1ccc(cc1)c1ccccc1",                   # biphenyl
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles", DROP20_MOLECULES,
                         ids=[f"drop20_{i}" for i in range(len(DROP20_MOLECULES))])
def test_no_empty_names_from_gate(smiles):
    """Molecules that previously triggered DROP-20 must return a non-empty name."""
    name = name_compound(smiles)
    assert name, f"name_compound returned empty/None for {smiles}"
    assert len(name.strip()) > 0, f"name_compound returned whitespace-only for {smiles}"
