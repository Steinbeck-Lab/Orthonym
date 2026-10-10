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
    """FACTOR_WEIGHTS must sum to the remaining-calibration target.

    a phase -a.1 update: 'ratio' is demoted to 0.0 (zero IUPAC
    Blue Book justification). Remaining four active weights sum to 0.80.
    a phase recalibrates; byte-identical preserved until then via
    position-based pool.best selection (candidate_pool.py:329-335).
    a phase carried a 1.0 target under the original 4-factor
    calibration; after ratio demotion, the expected value is the sum of
    the surviving non-zero weights.
    """
    total = sum(FACTOR_WEIGHTS.values())
    # 0.0 (ratio) + 0.20 (atom_cov) + 0.35 (fg_rec) + 0.25 (sub_comp)
    # + 0.0 (parent_correctness, a phase scaffolding) = 0.80
    expected = 0.80
    assert abs(total - expected) < 0.01, (
        f"FACTOR_WEIGHTS sum to {total}, expected {expected} "
        f"(post-ratio-demotion Phase 145.2 D-09-a.1)"
    )


@pytest.mark.integration
def test_weights_all_positive():
    """All calibrated factor weights must be >= 0 (with documented zeros).

    a phase update: 'parent_correctness' is the 5th factor
    (scaffolded with weight=0.0 for byte-identical safety per;
    a phase raises it to ~0.35 after train/test calibration).
    a phase -a.1 update: 'ratio' is demoted to 0.0 because the
    name-length/HA heuristic has zero IUPAC Blue Book justification.
    Both zero-weight keys are documented exceptions; all other weights
    remain strictly positive until a phase recalibrates.
    """
    for key, val in FACTOR_WEIGHTS.items():
        if key == 'parent_correctness':
            # a phase scaffolding key (weight=0.0 by byte-identical
            # contract). a phase calibrates and removes this exception.
            assert val == 0.0, (
                f"parent_correctness must be exactly 0.0 in Phase 145.1 "
                f"scaffolding (D-14 byte-identical proof); got {val}"
            )
            continue
        if key == 'ratio':
            # a phase -a.1 demotion (zero IUPAC justification).
            # Position-based pool.best keeps byte-identical. a phase
            # may recalibrate or permanently remove.
            assert val == 0.0, (
                f"ratio must be exactly 0.0 per Phase 145.2 D-09-a.1 "
                f"demotion; got {val}"
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
    """FACTOR_WEIGHTS must have exactly 5 keys after a phase scaffolding.

    a phase update: 5th key 'parent_correctness' added at LAST
    position (Risk 2 mitigation -- preserves dict iteration order in
    compute_confidence's sum-loop; weight=0.0 keeps confidence values
    byte-identical per).
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
    """Confidence is either an in-range float or an explicit 'unverified'.

     C4: previously `0.0 <= conf <= 1.0`, which TypeErrors on the honest
    None and — more importantly — could never fail, because every molecule
    that took an early return reported a fabricated 1.0 from
    namer.py:2693-2699. Out-of-range floats are still rejected.
    """
    for smi in DIVERSE_SMILES:
        result = name_compound(smi, include_confidence=True)
        assert isinstance(result, dict), f"Expected dict for {smi}"
        conf = result.get('confidence', -1)
        if conf is None:
            assert result['verification'] == 'unverified', smi
            assert result['factors'] == {}, smi
        else:
            assert 0.0 <= conf <= 1.0, (
                f"Confidence {conf} out of range for {smi}"
            )


@pytest.mark.integration
def test_simple_molecules_report_unmeasured_not_a_high_score():
    """ C4 re-derivation: this test used to ENCODE THE DEFECT.

    It asserted `conf >= 0.7` for CC / CCC / CCO / CC(=O)O. All four take an
    early return, so nothing scores them; the >= 0.7 was satisfied purely by
    the fabricated confidence=1.0 built at namer.py:2693-2699 whenever no
    candidate had been scored. The same fabrication certified
    CC(C)(C)OOCCO -> 'ethan-1-ol' -- six of nine heavy atoms dropped -- at
    atom_coverage=1.0. A test demanding a high score from that machinery was
    requiring the system to keep fabricating.

    What is actually true, and worth locking: these molecules are named
    correctly, and the API admits it has no coverage measurement for them.
    See test_measured_candidate_scores_well below for the other half -- that
    a candidate which IS scored still scores well.
    """
    simple = {"CC": "ethane", "CCC": "propane",
              "CCO": "ethanol", "CC(=O)O": "acetic acid"}
    for smi, expected in simple.items():
        result = name_compound(smi, include_confidence=True)
        assert result['name'] == expected, smi
        assert result['confidence'] is None, (
            f"{smi} reported confidence {result['confidence']!r}; nothing "
            f"scored it, so no number is warranted"
        )
        assert result['verification'] == 'unverified', smi
        assert result['factors'] == {}, smi


@pytest.mark.integration
def test_measured_candidate_scores_well():
    """The other half of the re-derivation above.

    Replacing a fabricated high score with 'unverified' must not lose the
    ability to detect a real scoring regression. Caffeine and adenine DO route
    through candidate scoring, so they carry a populated factors dict and a
    real aggregate confidence -- and it should be high for molecules this
    well-handled. This is the assertion the old test was trying to make, on a
    molecule where it is actually meaningful.
    """
    for smi in ("Cn1c(=O)c2c(ncn2C)n(C)c1=O", "Nc1ncnc2nc[nH]c12"):
        result = name_compound(smi, include_confidence=True)
        assert result['confidence'] is not None, (
            f"{smi} no longer routes through candidate scoring -- if that is "
            f"intended, move it to the unmeasured test above; if not, this is "
            f"the scoring regression this test exists to catch"
        )
        assert result['factors'], smi
        assert result['confidence'] >= 0.7, (
            f"{smi} scored {result['confidence']}, expected >= 0.7"
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

# Expected values are the PINs the engine now emits (each OPSIN round-trips to the input):
# canary_0 'aspirin' is a brand name, withdrawn as a retained name by the PIN deny list
# (7c996fd24, iupac_2013_pin_list.json, citation; the ester-on-acid form
# is the PIN (the Blue Book under, '4-(acetyloxy)benzoic acid (PIN)').
# canary_1 'isobutyl' is not a preferred prefix: '2-methylpropyl (preferred prefix)
# (not isobutyl)', the Blue Book under.
# canary_2 citric acid: the Blue Book under.
# canary_3 two OH on the ring outrank one on the chain, the Blue Book.
# canary_6 amide is senior to alcohol (phenol); 'N-phenylacetamide (PIN)' the Blue Book.
# canary_9 'propan-2-yl (preferred prefix)', the Blue Book under; alphanumerical
# order puts methyl before propan-2-yl.
# canary_13 'ethylene glycol' is general nomenclature only: 'ethane-1,2-diol (PIN)',
# the Blue Book under.
CANARY_SUBSET = [
    ("CC(=O)Oc1ccccc1C(=O)O", "2-(acetyloxy)benzoic acid"),
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "2-[4-(2-methylpropyl)phenyl]propanoic acid"),
    ("OC(=O)CC(O)(CC(=O)O)C(=O)O", "2-hydroxypropane-1,2,3-tricarboxylic acid"),
    ("OCCc1ccc(O)c(O)c1", "4-(2-hydroxyethyl)benzene-1,2-diol"),
    ("CC12CCC3C(CCC4CC(=O)CCC43C)C1CCC2O", "17-hydroxyandrostan-3-one"),
    ("OC(=O)c1ccc(N)cc1", "4-aminobenzoic acid"),
    ("CC(=O)Nc1ccc(O)cc1", "N-(4-hydroxyphenyl)acetamide"),
    # a phase: old expected name "1-(N,N-diethylamino)-4-phenylbenzene" was
    # incorrect -- it dropped the N=N azo linkage because _check_retained_substituent
    # wrongly returned "phenyl" for the N=N-phenyl fragment. The fix (counting all
    # non-ring heavy atoms, not just carbons) correctly rejects that shortcut.
    # Leads program L3 (43e, change-asserted-value): the abstention that was pinned here is the
    # defect, and the row is the PIN now. 'Unsymmetrical monoazo compounds are
    # named in two ways' (the Blue Book),:38791: "Monoazo compounds with the general
    # structure R-N=N-R' in which R is substituted by a principal characteristic group are named
    # on the basis of the parent hydride, RH, substituted by an organyl diazenyl group, R'-N=N-";
    # '4-(phenyldiazenyl)benzene-1-sulfonic acid (PIN)' (:38798); the amine is aniline,
    #:17722). OPSIN 2.9.0 reads the name back to the input's full InChIKey (checked by a fresh
    # call in tests/unit/assembly/test_leads_l3_43e_organyl_diazenyl.py), and the old abstention
    # returns on the code before the change. The row runs under `opsin_gate`: with the validity
    # gate off (the suite default, tests/conftest.py) a producer outside the PIN path could ship a
    # wrong-structure name, which the gate suppresses.
    pytest.param("CCN(CC)c1ccc(N=Nc2ccccc2)cc1", "N,N-diethyl-4-(phenyldiazenyl)aniline",
                 marks=pytest.mark.opsin_gate),
    ("CC(=O)O", "acetic acid"),
    ("Cc1ccc(O)c(C(C)C)c1", "4-methyl-2-(propan-2-yl)phenol"),  #: phenol suffix routing
    ("OC(=O)/C=C\\C(=O)O", "(2Z)-but-2-enedioic acid"),
    ("OC(=O)c1ccccc1O", "2-hydroxybenzoic acid"),
    ("c1ccc2c(c1)cc1ccc3ccccc3c1c2", "tetraphene"),
    ("OCCO", "ethane-1,2-diol"),
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
def test_canary_subset_stable(smiles, expected, request):
    """Spot-check 20 canary compounds for exact name match."""
    if smiles == "CC12CCC3C(CCC4CC(=O)CCC43C)C1CCC2O":
        # Suite fix j6-breadth (TRIAGE g3 C17b): a stereo-free androstanolone.
        # 'androstane' implies the configuration of every chirality centre
        #, the Blue Book), which this input does not define,
        # so the expected stereoparent name is not the PIN; the PIN tier
        # abstains (hydro-cyclopenta[a]phenanthrene PIN not built). Best-effort
        # names it RT-exact (test_j6_breadth tier contract).
        request.applymarker(pytest.mark.xfail(strict=True, reason=(
            "PIN tier abstains: needs the hydro-cyclopenta[a]phenanthrene PIN "
            "for a stereo-free steroid (P-101.2.6); the expected "
            "'17-hydroxyandrostan-3-one' implies a configuration the input "
            "lacks -- TODO in TRIAGE.md 'Suite fix -- j6-breadth'")))
    name = name_compound(smiles)
    assert name == expected, (
        f"Canary failed: {smiles} -> '{name}' (expected '{expected}')"
    )


# ---------------------------------------------------------------------------
# No empty names test
# ---------------------------------------------------------------------------

# Molecules that historically triggered (coverage gate reject)
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
    """Molecules that previously triggered must return a non-empty name."""
    name = name_compound(smiles)
    assert name, f"name_compound returned empty/None for {smiles}"
    assert len(name.strip()) > 0, f"name_compound returned whitespace-only for {smiles}"
