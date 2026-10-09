""" C4 -- the atom-coverage HONESTY contract.

Project invariant, applied to atom coverage:

    A proof must never be CONFIDENTLY WRONG. Insufficient evidence =>
    'unverified', never 'pass' and never 'fail'.

The defect these tests lock out: ``name_with_confidence('CC(C)(C)OOCCO')``
(9 heavy atoms) returns ``'ethan-1-ol'`` (3 heavy atoms -- six atoms silently
dropped) and used to report ``confidence=1.0`` with
``factors['atom_coverage']=1.0`` and ``handler='direct'``, i.e. the PUBLIC API
certified PERFECT atom coverage for a name that describes a third of the
molecule. With a JRE the round-trip suppresses that name; without the
jar -- a supported mode -- it ships, and every Java-free defence failed.

These tests do NOT assert that coverage is measured (it is not: see
``test_atom_coverage_is_never_measured_in_the_production_path``). They assert
the weaker, achievable property that the system never CLAIMS a measurement it
does not have.
"""

import re
from pathlib import Path

import pytest

from orthonym.assembly.coverage_scoring import (
    COVERAGE_ESTIMATED_NAME_LENGTH,
    COVERAGE_MEASURED,
    COVERAGE_RETAINED_NAME_BOOST,
    VERIFICATION_UNVERIFIED,
    CandidateName,
    clear_confidence,
    compute_confidence,
    retrieve_confidence,
    store_confidence,
    unmeasured_confidence,
)
from orthonym.namer import Orthonym

# The reference defect: 9 heavy atoms in, a 3-heavy-atom name out.
DROPPING_SMILES = 'CC(C)(C)OOCCO'
DROPPED_NAME = 'ethan-1-ol'

SRC = Path(__file__).resolve().parents[2] / 'src' / 'orthonym'
NAMER_SRC = (SRC / 'namer.py').read_text()
SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'


@pytest.fixture
def namer():
    """A namer with the JVM-dependent gates off -- the unprotected mode."""
    return Orthonym()


@pytest.fixture(autouse=True)
def _clean_store():
    clear_confidence()
    yield
    clear_confidence()


# ---------------------------------------------------------------------------
# 1. The shared unmeasured record
# ---------------------------------------------------------------------------

def test_unmeasured_confidence_asserts_neither_pass_nor_fail():
    """The one honest record: no number that implies a verdict."""
    rec = unmeasured_confidence(name='ethan-1-ol')

    # NOT 1.0 (a fabricated pass) and NOT 0.0 (a fabricated fail).
    assert rec['confidence'] is None
    assert rec['verification'] == VERIFICATION_UNVERIFIED
    # Empty, so no consumer can read a coverage number that never existed.
    assert rec['factors'] == {}
    assert rec['coverage_provenance'] is None
    # 'direct' was the old value and is a real-looking handler id; it cleared
    # the quality gates' `!= 'unknown'` exclusion while carrying no factors.
    assert rec['handler'] == 'unmeasured'
    assert rec['name'] == 'ethan-1-ol'


def test_unmeasured_record_never_reports_atom_coverage():
    """`factors` must not carry a fabricated atom_coverage key at all.

    Absent is honest; ``atom_coverage: 1.0`` is the defect, and
    ``atom_coverage: 0.0`` would merely invert it.
    """
    assert 'atom_coverage' not in unmeasured_confidence(name='x')['factors']


# ---------------------------------------------------------------------------
# 2. The public API on the reference defect
# ---------------------------------------------------------------------------

def test_public_api_does_not_certify_the_atom_dropping_name(namer):
    """The headline regression test: no perfect score on a name that drops 6/9 atoms.

    The producer no longer drops the peroxy atoms: the name is now
    '2-(tert-butylperoxy)ethan-1-ol' (OPSIN 2.9.0 reads it back to the input's full
    an InChIKey; the dropped name 'ethan-1-ol' reads to
    an InChIKey, a different molecule), so the reference defect is no
    longer reachable through the real producer. This test keeps the honesty half on that name, which
    still has no measured coverage, and the next test restores the atom-dropping half
    by feeding the old dropped name through the same public API."""
    from tests.support.rt_assert import name_is_rt_exact
    md = namer.name_with_confidence(DROPPING_SMILES)

    assert md['name'] == '2-(tert-butylperoxy)ethan-1-ol'
    assert name_is_rt_exact(md['name'], DROPPING_SMILES)
    assert md['name'] != DROPPED_NAME
    assert md['confidence'] is None, (
        'nothing scored this name, so no confidence number is warranted'
    )
    assert md['verification'] == VERIFICATION_UNVERIFIED
    assert md['factors'] == {}
    assert md['factors'].get('atom_coverage') is None
    assert md['handler'] != 'direct'


def test_public_api_does_not_certify_a_producer_that_drops_atoms(namer, monkeypatch):
    """The atom-dropping half of the headline test, independent of the producer's state.

    ``_name_impl`` is made to return the old defective name 'ethan-1-ol' (3 heavy atoms)
    for the 9-heavy-atom input, as the producer did before the peroxy prefix was built; the
    public API must still not certify it with a confidence number, a verification or an
    atom-coverage factor, whatever the producer does."""
    monkeypatch.setattr(Orthonym, '_name_impl', lambda self, smiles: DROPPED_NAME)
    md = namer.name_with_confidence(DROPPING_SMILES)

    assert md['name'] == DROPPED_NAME
    assert md['confidence'] is None, (
        'nothing scored this name, so no confidence number is warranted'
    )
    assert md['verification'] == VERIFICATION_UNVERIFIED
    assert md['factors'] == {}
    assert md['factors'].get('atom_coverage') is None
    assert md['handler'] != 'direct'


@pytest.mark.parametrize('smiles, expected', [
    ('CCO', 'ethanol'),
    ('CC(=O)O', 'acetic acid'),
    ('c1ccccc1', 'benzene'),
    ('CC(C)(C)CC(=O)O', '3,3-dimethylbutanoic acid'),
])
def test_correct_names_are_unchanged_and_also_honest(namer, smiles, expected):
    """Making the contract honest must not alter any emitted name.

    These four are correct today and must stay byte-identical. Their metadata
    is `unverified` too -- which is the truth: nothing measures coverage for
    them either. An honest 'unverified' on a correct name is the intended
    outcome, not a degradation.
    """
    md = namer.name_with_confidence(smiles)
    assert md['name'] == expected
    assert md['confidence'] is None or isinstance(md['confidence'], float)
    if md['confidence'] is None:
        assert md['verification'] == VERIFICATION_UNVERIFIED
        assert md['factors'] == {}


def test_every_metadata_record_carries_the_verification_key(namer):
    """A consumer must always be able to ask 'was this verified?'."""
    for smiles in (DROPPING_SMILES, 'CCO', 'c1ccccc1', 'C1CCC2(CC1)CCCCC2'):
        md = namer.name_with_confidence(smiles)
        assert 'verification' in md, smiles
        assert md['verification'] in ('verified', 'unverified', 'refuted'), smiles


# ---------------------------------------------------------------------------
# 3. Coverage provenance -- the estimate must never pass as a measurement
# ---------------------------------------------------------------------------

class _FakeMol:
    def __init__(self, n):
        self._n = n

    def GetNumHeavyAtoms(self):
        return self._n


class _FakeFeatures:
    """Minimal stand-in: compute_confidence only needs mol + FG/substituent data."""

    def __init__(self, n_heavy):
        self.mol = _FakeMol(n_heavy)
        self.functional_groups = []
        self.substituents = []
        self.ring_systems = []


def test_compute_confidence_marks_a_real_measurement_as_measured():
    """Given parent atoms, atom_coverage IS a measurement and says so."""
    cand = compute_confidence(
        'ethan-1-ol', 'chain', _FakeFeatures(9), parent_atom_indices={0, 1, 2},
    )
    assert cand.coverage_provenance == COVERAGE_MEASURED
    # 3 of 9 heavy atoms -- the real coverage of the reference defect, and
    # well under the gate's 0.55 cut-off.
    assert cand.factors['atom_coverage'] == pytest.approx(0.3333, abs=1e-4)


def test_compute_confidence_marks_the_name_length_proxy_as_estimated():
    """Without parent atoms the number is a name-LENGTH proxy, not coverage."""
    cand = compute_confidence('ethan-1-ol', 'chain', _FakeFeatures(9))
    assert cand.coverage_provenance == COVERAGE_ESTIMATED_NAME_LENGTH
    # Identical to `ratio` by construction -- that is what makes it not a
    # coverage measurement.
    assert cand.factors['atom_coverage'] == cand.factors['ratio']
    # And it is 0.74, NOT the true 0.33: proof that the 0.55 gate cannot
    # catch the reference defect even where the gate is reachable.
    assert cand.factors['atom_coverage'] == pytest.approx(0.7407, abs=1e-4)
    assert cand.factors['atom_coverage'] > 0.55


def test_retained_name_boost_is_not_a_coverage_measurement():
    """The retained-name 1.0 is a claim about the NAME, not about atoms."""
    cand = compute_confidence('benzene', 'benzene', _FakeFeatures(6))
    assert cand.factors['atom_coverage'] == 1.0
    assert cand.coverage_provenance == COVERAGE_RETAINED_NAME_BOOST


def test_atom_coverage_is_never_measured_in_the_production_path():
    """No production caller supplies parent_atom_indices -- so nothing is measured.

    ``CandidatePool.add`` withholds it deliberately ("Risk 1": passing it
    would change atom_coverage and break the byte-identical guarantee) and
    attaches it to the CandidateName POST-HOC instead. This test pins that
    fact so it cannot be forgotten while reading ``atom_coverage``.
    """
    # Calls only -- never the `def compute_confidence(...)` signature, which
    # legitimately names the parameter.
    call_re = re.compile(r'(?<!def )\bcompute_confidence\(\s*[^)]*?\)', re.S)
    offenders = []
    for path in sorted(SRC.rglob('*.py')):
        for m in call_re.finditer(path.read_text()):
            call = m.group(0)
            if 'parent_atom_indices' in call:
                offenders.append(f'{path.name}: {call.strip()}')
    assert offenders == [], (
        'A production caller now passes parent_atom_indices, so atom_coverage '
        'may be a REAL measurement for it. That is an improvement -- but the '
        'gate/threshold comments and this contract must be updated together, '
        f'and the 0.55 cut-off re-derived on measured data. Found: {offenders}'
    )


def test_measured_coverage_would_catch_the_reference_defect():
    """The 0.55 gate is sound ON MEASURED DATA -- only the input is missing.

    This is the forward-looking half of the contract: it shows the fix is a
    producer-side atom-coverage record, not a threshold change. Given the REAL
    parent atom set for the reference defect (the 3-atom ethanol chain out of
    9 heavy atoms), atom_coverage is 0.33 and the existing 0.55 cut-off fires.
    Given the name-length proxy that production actually supplies, the same
    molecule scores 0.74 and passes. Same threshold, opposite verdicts.
    """
    real = compute_confidence(
        'ethan-1-ol', 'chain', _FakeFeatures(9), parent_atom_indices={0, 1, 2},
    )
    proxy = compute_confidence('ethan-1-ol', 'chain', _FakeFeatures(9))

    assert real.factors['atom_coverage'] < 0.55, 'measured coverage fires the gate'
    assert proxy.factors['atom_coverage'] > 0.55, 'the proxy does not'
    assert real.coverage_provenance == COVERAGE_MEASURED
    assert proxy.coverage_provenance == COVERAGE_ESTIMATED_NAME_LENGTH


def test_a_measured_full_coverage_candidate_still_scores_well():
    """Honesty must not mean pessimism: a fully-covering parent scores 1.0."""
    cand = compute_confidence(
        '2-methylheptan-2-ol', 'chain', _FakeFeatures(9),
        parent_atom_indices=set(range(9)),
    )
    assert cand.coverage_provenance == COVERAGE_MEASURED
    assert cand.factors['atom_coverage'] == 1.0


def test_retained_name_boost_overrides_even_a_real_measurement():
    """Documented, deliberately NOT changed here: the boost wins.

    When the boost applies, it sets atom_coverage to 1.0 unconditionally --
    discarding a real measurement if one was supplied. It is a second,
    independent way the number can overstate coverage, and it is why the
    gate excludes handler='retained_name'.

    The boost is itself size-limited (`is_core or (is_retained and
    total_heavy <= 15)`) precisely so a retained scaffold cannot inflate a
    large molecule -- so at 12 heavy atoms it overrides a real 6/12 = 0.5,
    while at 30 heavy atoms the same measurement survives untouched. Both
    directions are pinned below.

    Not changed in this task: the boost feeds the weighted sum, so altering it
    would move confidence values and break the byte-identical guarantee. It is
    recorded here so the next change to compute_confidence sees it, and so the
    provenance flag ('retained_name_boost', never 'measured') is pinned as the
    way to detect it.
    """
    half = {0, 1, 2, 3, 4, 5}

    # <= 15 heavy atoms: the boost fires and discards the measurement.
    boosted = compute_confidence(
        'cyclohexane', 'complex_ring', _FakeFeatures(12),
        parent_atom_indices=half,                  # a real 6/12 = 0.50
    )
    assert boosted.factors['atom_coverage'] == 1.0
    assert boosted.coverage_provenance == COVERAGE_RETAINED_NAME_BOOST
    assert boosted.coverage_provenance != COVERAGE_MEASURED

    # > 15 heavy atoms: the boost is withheld and the measurement survives.
    kept = compute_confidence(
        'cyclohexane', 'complex_ring', _FakeFeatures(30),
        parent_atom_indices=half,                  # a real 6/30 = 0.20
    )
    assert kept.factors['atom_coverage'] == pytest.approx(0.20)
    assert kept.coverage_provenance == COVERAGE_MEASURED


def test_verification_is_verified_only_when_coverage_was_measured():
    """The tri-state must track provenance, not handler identity."""
    measured = CandidateName(
        name='x', handler='chain', confidence=0.9,
        factors={'atom_coverage': 0.9}, coverage_provenance=COVERAGE_MEASURED,
    )
    store_confidence(measured)
    assert retrieve_confidence()['verification'] == 'verified'

    estimated = CandidateName(
        name='x', handler='chain', confidence=0.9,
        factors={'atom_coverage': 0.9},
        coverage_provenance=COVERAGE_ESTIMATED_NAME_LENGTH,
    )
    store_confidence(estimated)
    assert retrieve_confidence()['verification'] == VERIFICATION_UNVERIFIED


# ---------------------------------------------------------------------------
# 4. The empty store reports 'unverified', not a fabricated 0.0
# ---------------------------------------------------------------------------

def test_empty_store_is_unverified_not_a_zero_verdict():
    clear_confidence()
    rec = retrieve_confidence()
    assert rec['confidence'] is None, '0.0 is a fabricated FAIL'
    assert rec['verification'] == VERIFICATION_UNVERIFIED
    assert rec['factors'] == {}
    # Preserved on purpose: the name quality gates use handler != 'unknown'
    # as their "was anything scored?" guard, and the store's never-written
    # state is exactly that.
    assert rec['handler'] == 'unknown'
    assert rec['name'] == ''


# ---------------------------------------------------------------------------
# 5. The gate must not read a MISSING measurement as perfect coverage
# ---------------------------------------------------------------------------

def test_gate_has_no_fail_open_perfect_default():
    """`.get('atom_coverage', 1.0)` defaulted "unmeasured" to "perfect".

    This was LIVE, not latent: the chain-parent path publishes a candidate
    with handler='chain' and an EMPTY factors dict, which clears both of the
    gate's exclusions and then hits the default.
    """
    assert "get('atom_coverage', 1.0)" not in NAMER_SRC
    assert 'get("atom_coverage", 1.0)' not in NAMER_SRC
    # and the explicit unmeasured branch is present
    assert "atom_cov = _factors.get('atom_coverage')" in NAMER_SRC
    assert 'if atom_cov is None:' in NAMER_SRC


def test_chain_path_candidate_does_not_claim_perfect_confidence():
    """The atom_to_locant carrier must not broadcast a fabricated 1.0.

    It exists only to hand the stereo backstop a locant map; nothing scored
    it, and it carries no factors.
    """
    assert 'confidence=1.0,\n                        atom_to_locant=' not in NAMER_SRC
    assert 'confidence=None,\n                        atom_to_locant=' in NAMER_SRC


# ---------------------------------------------------------------------------
# 6. The calibration script must not manufacture its own input
# ---------------------------------------------------------------------------

def test_calibration_script_does_not_fabricate_factor_records():
    """It used to synthesise all-1.0 factors for the no-confidence path."""
    text = (SCRIPTS / 'calibrate_coverage_gate.py').read_text()
    # The fabricated literal block is gone...
    assert "'handler': 'direct'," not in text
    #...replaced by explicit exclusion of unmeasured records...
    assert "result.get('verification') == 'unverified'" in text
    assert 'n_unmeasured += 1' in text
    #...and a refusal floor that is actually COMPARED, not merely defined.
    # (Asserting only that the constant name appears would pass even if the
    # comparison were deleted -- that weakness was found by mutation M9.)
    assert 'MIN_CALIBRATION_POPULATION = ' in text
    assert 'if n_usable < MIN_CALIBRATION_POPULATION:' in text
    assert 'REFUSING TO CALIBRATE' in text
    assert 'sys.exit(2)' in text
