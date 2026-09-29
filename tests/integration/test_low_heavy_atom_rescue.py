"""a phase verification: low-heavy-atom rescue fallback preservation.

Per internal notes:
  The internal /1.5 fallback at composer.py (formerly
  `_MIN_RATIO_ACCEPT/1.5`) is PRESERVED — it is a low-heavy-atom rescue,
  NOT a weighted-sum gate.

a phase removes the top-level _MIN_RATIO_ACCEPT gate, but Plan 05
Task 2 preserves the inner fallback as `_MIN_RATIO_FALLBACK = 0.20` with
the same `if total_heavy <= 15 or ratio >= _MIN_RATIO_FALLBACK` semantics.

Tests in this module verify:

  (a) the inline _MIN_RATIO_FALLBACK constant literal exists in source,
  (b) the if-fallback condition exists in source,
  (c) _MIN_RATIO_ACCEPT and _CASCADE_RATIO_MIN constants are REMOVED,
  (d) 3 low-heavy-atom canary molecules produce valid names in BOTH
      V17 and V18 modes — the fallback is doing its job.

Source: https://iupac.qmul.ac.uk/BlueBook/P4.html
"""

import pytest

from tests.support.module_reload import reloaded


# Three low-heavy-atom canaries per internal notes ("add 3 low-heavy-atom
# molecules to canary to prove the fallback still fires").
LOW_HA_CANARIES = [
    # (SMILES, expected-name-substring)
    # Bicyclo[2.2.2]octane — 8 heavy atoms — small bridged carbocycle.
    # (Earlier planning notes mislabelled this as norbornane, which is
    # actually bicyclo[2.2.1]heptane — C1CC2CCC1C2.)
    ("C1CC2CCC1CC2", "bicyclo"),
    ("Fc1ccccc1", "fluorobenzene"),     # 7 heavy atoms — small aromatic
    ("Clc1ccncc1", "chloropyridine"),   # 7 heavy atoms — small heterocycle
]


@pytest.fixture(params=[
    ("false", "first_applicable"),  # V17
    ("true",  "score_based"),        # V18
], ids=["v17", "v18"])
def both_modes(request, monkeypatch):
    """Parametrized fixture: each test runs under V17 and V18.

    The two modules are reloaded to read the env vars at import time and put
    back after the test (``tests.support.module_reload.reloaded``, TRIAGE
    'Canary oxime -- test-order flake').
    """
    use_v18, sel_mode = request.param
    monkeypatch.setenv("ORTHONYM_USE_V18_WEIGHTS", use_v18)
    monkeypatch.setenv("ORTHONYM_SELECTION_MODE", sel_mode)
    from orthonym.assembly import coverage_scoring, candidate_pool
    with reloaded(coverage_scoring, candidate_pool):
        yield (use_v18, sel_mode)


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_substring", LOW_HA_CANARIES)
def test_low_ha_canaries_produce_name_in_both_modes(
    smiles, expected_substring, both_modes
):
    """: low-HA molecules MUST produce a non-empty name in both V17 and V18.

    The fallback `if total_heavy <= 15: return best.name` ensures small
    molecules (<=15 heavy atoms) always return the best Tier A candidate
    without being blocked by the old 0.20 ratio threshold.
    """
    from orthonym import name_compound
    name = name_compound(smiles)
    assert name, (
        f"D-10 violation: {smiles} produced empty name "
        f"in mode={both_modes} — the low-HA rescue fallback is not firing"
    )
    # Name should contain the expected retained-name substring (loose match).
    assert expected_substring.lower() in name.lower(), (
        f"D-10 verification: {smiles} expected '{expected_substring}' "
        f"in name; got '{name}' (mode={both_modes})"
    )


@pytest.mark.integration
def test_min_ratio_fallback_constant_exists():
    """Verify `_MIN_RATIO_FALLBACK = 0.20` exists in composer.py per.

    This is a source-level assertion: the constant must be present as a
    literal in the module source so static introspection confirms the
    preservation invariant. Renaming to a different literal (e.g., 0.15
    or 0.25) silently breaks and this test catches it.
    """
    import orthonym.assembly.composer as composer
    with open(composer.__file__) as fp:
        src = fp.read()
    assert "_MIN_RATIO_FALLBACK = 0.20" in src, (
        "D-10 PRESERVATION FAILURE: _MIN_RATIO_FALLBACK constant "
        "missing or not set to 0.20 in composer.py"
    )
    # Confirm the fallback condition is still present.
    assert "if total_heavy <= 15" in src, (
        "D-10 PRESERVATION FAILURE: low-heavy-atom rescue condition "
        "(`if total_heavy <= 15 or ...`) removed from composer.py"
    )


@pytest.mark.integration
def test_min_ratio_accept_constant_removed():
    """Confirm the top-level ratio-accept gate constant is gone .

    The CONSTANT `_MIN_RATIO_ACCEPT` must not appear anywhere in
    composer.py source. Comments referencing it in the commit history
    are fine, but not in the module source.
    """
    import orthonym.assembly.composer as composer
    with open(composer.__file__) as fp:
        src = fp.read()
    assert "_MIN_RATIO_ACCEPT" not in src, (
        "SC-4 FAILURE: _MIN_RATIO_ACCEPT constant still present in "
        "composer.py — the top-level ratio-accept gate was not removed"
    )


@pytest.mark.integration
def test_cascade_ratio_min_constant_removed():
    """Confirm the top-level cascade-ratio-min gate constant is gone .

    The CONSTANT `_CASCADE_RATIO_MIN` must not appear anywhere in
    composer.py source.
    """
    import orthonym.assembly.composer as composer
    with open(composer.__file__) as fp:
        src = fp.read()
    assert "_CASCADE_RATIO_MIN" not in src, (
        "SC-4 FAILURE: _CASCADE_RATIO_MIN constant still present in "
        "composer.py — the cascade-ratio-min gate was not removed"
    )
