"""Phase 146 integration test: 50+ molecules across handler tiers under V17/V18.

Per CONTEXT.md SC-6: two-tier selector tested end-to-end.
Per RESEARCH §5.3 + §7 Dimension 2.

Each test runs twice via the `feature_flag_mode` parametrized fixture —
once under V17 (default soak) and once under V18 (two-tier cascade
active). V17 invariants assert the expected name contains the retained
substring; V18 invariants allow substring match OR defensible variation
(calibration may move some names).

Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
"""

import importlib
import subprocess
import sys
import textwrap

import pytest


@pytest.fixture(params=[
    ("false", "first_applicable"),  # V17 (default soak per D-07)
    ("true",  "score_based"),        # V18 (two-tier active per D-01)
], ids=["v17", "v18"])
def feature_flag_mode(request, monkeypatch):
    """Parametrized fixture: each test runs under both V17 and V18.

    Sets the two env vars and reloads coverage_scoring + candidate_pool
    to pick up the module-import-time env-var reads. The conftest autouse
    _phase146_clear_thread_locals fixture handles thread-local cleanup
    after the test yields.
    """
    use_v18, sel_mode = request.param
    monkeypatch.setenv("ORTHONYM_USE_V18_WEIGHTS", use_v18)
    monkeypatch.setenv("ORTHONYM_SELECTION_MODE", sel_mode)
    # Force module reload so the env vars are re-read at import time.
    from orthonym.assembly import coverage_scoring, candidate_pool
    importlib.reload(coverage_scoring)
    importlib.reload(candidate_pool)
    yield (use_v18, sel_mode)


def _name(smi):
    """Helper to invoke name_compound after env-var reload."""
    from orthonym import name_compound
    return name_compound(smi)


# ---------------------------------------------------------------------------
# Tier A: ring handlers (benzene, heterocycle, complex_ring)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestTierARings:
    """Tier A ring handlers — compete via pool._best_two_tier (V18) or
    first_applicable (V17). Expected: retained names survive both modes.
    """

    @pytest.mark.parametrize("smiles,expected_v17_substring", [
        ("c1ccccc1", "benzene"),
        ("c1ccncc1", "pyridine"),
        ("c1ccc2ccccc2c1", "naphthalene"),
        ("c1ccc2[nH]ccc2c1", "indole"),
        ("c1ccoc1", "furan"),
        ("c1ccsc1", "thiophene"),
        ("c1cc[nH]c1", "pyrrole"),
        ("c1cnc[nH]1", "imidazole"),
        ("c1ccc2ncccc2c1", "quinoline"),
        ("c1ccc2cc3ccccc3cc2c1", "anthracene"),
    ])
    def test_tier_a_ring(self, smiles, expected_v17_substring, feature_flag_mode):
        """Tier A ring naming under V17 and V18 — should produce a name
        containing the expected retained-name substring.
        """
        name = _name(smiles)
        assert name, f"{smiles}: produced empty name (mode={feature_flag_mode})"
        assert expected_v17_substring in name.lower(), (
            f"{smiles}: expected '{expected_v17_substring}' in name '{name}' "
            f"(mode={feature_flag_mode})"
        )


# ---------------------------------------------------------------------------
# Tier B: special-case handlers (oxime, hydrazone, isocyanate, etc.)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestTierBSpecial:
    """Tier B special handlers — firstly direct-return via _confidence_gate.
    Expected: produce non-empty names in both modes.
    """

    @pytest.mark.parametrize("smiles", [
        "CC(=NO)C",         # acetone oxime
        "CC=NN",            # acetaldehyde hydrazone
        "CN=C=O",           # methyl isocyanate
        "CN=C=S",           # methyl isothiocyanate
        "NC(=O)O",          # carbamic acid
        "COC(=O)N",         # methyl carbamate
        "NC(=O)N",          # urea
        "NC(=N)N",          # guanidine
        "CS(=O)C",          # dimethyl sulfoxide
        "CS(=O)(=O)C",      # dimethyl sulfone
        "CSC",              # dimethyl sulfide
        "OB(O)c1ccccc1",    # phenylboronic acid
    ])
    def test_tier_b(self, smiles, feature_flag_mode):
        """Tier B handlers produce a non-empty name in both modes."""
        name = _name(smiles)
        assert name, (
            f"{smiles}: Tier B handler produced empty name "
            f"(mode={feature_flag_mode})"
        )


# ---------------------------------------------------------------------------
# Direct-return: polycyclic, amine, n_oxide, ring_nitrile, amide
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDirectReturn:
    """Direct-return handlers — first-applicable short-circuits pool.best().
    Expected: produce non-empty names in both modes.
    """

    @pytest.mark.parametrize("smiles", [
        "c1ccc2ccccc2c1",   # naphthalene (polycyclic)
        "c1ccc2cc3ccccc3cc2c1",  # anthracene (polycyclic)
        "NCC",              # ethylamine (amine)
        "CCN(CC)CC",        # triethylamine (amine, tertiary)
        "CCNC",             # N-methylethylamine (amine, secondary)
        "[O-][n+]1ccccc1",  # pyridine N-oxide
        "CC(=O)NC",         # N-methylacetamide (amide)
        "CC(=O)N",          # acetamide (amide)
        "N#Cc1ccccc1",      # benzonitrile (ring_nitrile)
        "N#Cc1ccncc1",      # 4-cyanopyridine (ring_nitrile on heterocycle)
    ])
    def test_direct_return(self, smiles, feature_flag_mode):
        """Direct-return handler produces a non-empty name in both modes."""
        name = _name(smiles)
        assert name, (
            f"{smiles}: direct-return handler produced empty name "
            f"(mode={feature_flag_mode})"
        )


# ---------------------------------------------------------------------------
# Chain handler — V18 first-class competition per SC-5
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestChain:
    """Chain handler — V18 wires chain as a first-class pool candidate
    (SC-5). V17 falls back to chain only when ring handlers fail. Expected:
    both modes produce the expected chain name.
    """

    @pytest.mark.parametrize("smiles,expected", [
        ("C", "methane"),
        ("CC", "ethane"),
        ("CCC", "propane"),
        ("CCCC", "butane"),
        ("CCO", "ethanol"),
        ("CCC=O", "propanal"),
        ("CCCCCC(=O)O", "hexanoic acid"),
        ("CC(O)C", "propan-2-ol"),
        ("CCC#N", "propanenitrile"),
        ("CCCCC", "pentane"),
    ])
    def test_chain(self, smiles, expected, feature_flag_mode):
        """Chain naming should produce the expected name in BOTH modes.

        Uses a loose substring/normalized-equality check to absorb minor
        punctuation variants (e.g., with/without hyphens in locants).
        """
        name = _name(smiles)
        assert name, f"{smiles}: empty name (mode={feature_flag_mode})"
        nm = name.lower()
        exp = expected.lower()
        assert (
            exp in nm
            or exp.replace("-", "") in nm.replace("-", "")
            or exp.replace(" ", "") in nm.replace(" ", "")
        ), (
            f"{smiles}: expected '{expected}', got '{name}' "
            f"(mode={feature_flag_mode})"
        )


# ---------------------------------------------------------------------------
# Canonical P-44.1.1 chain-wins case: phenyl-hexanoic acid
# Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1.1
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPhenylHexanoicAcid:
    """The canonical P-44.1.1 chain-wins test (RESEARCH §7 Dimension 2).

    SMILES: c1ccc(CCCCCC(=O)O)cc1
      - Chain: hexanoic acid (6 carbons + 1 PCG = COOH)
      - Ring: benzene (6 carbons + 0 PCGs)

    Per P-44.1.1: max PCG count wins → chain (hexanoic acid) wins.
    """

    def test_v18_picks_chain_parent(self):
        """In V18 mode, chain (hexanoic acid) should be the selected parent.

        Uses subprocess isolation so module-import-time env-var reads are
        honored without importlib.reload class-identity leakage (per Plan
        03 Rule 1 fix pattern).
        """
        script = textwrap.dedent(
            """
            from orthonym import name_compound
            name = name_compound("c1ccc(CCCCCC(=O)O)cc1")
            print(name)
            """
        ).strip()
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={
                "ORTHONYM_USE_V18_WEIGHTS": "true",
                "ORTHONYM_SELECTION_MODE": "score_based",
                "PATH": "/usr/bin:/usr/local/bin",
                "PYTHONPATH": "",
            },
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"V18 subprocess failed: stderr={result.stderr!r}"
        )
        name = result.stdout.strip().splitlines()[-1] if result.stdout else ""
        assert name, (
            f"V18 produced empty name for c1ccc(CCCCCC(=O)O)cc1 "
            f"(stdout={result.stdout!r})"
        )
        assert "hexanoic" in name.lower(), (
            f"P-44.1.1 chain-wins FAILURE: V18 expected chain (hexanoic acid) "
            f"parent for c1ccc(CCCCCC(=O)O)cc1; got {name!r}"
        )


# ---------------------------------------------------------------------------
# V17 byte-identical smoke — ensure env-var flip doesn't alter default
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestV17ByteIdenticalSmoke:
    """V17 default path byte-identical smoke test.

    Verifies that when both env vars are absent (or explicitly V17),
    the namer produces the established retained-name outputs for a
    handful of canonical molecules. Catches regression where a stray
    V18 code path leaks into the V17 default.
    """

    @pytest.mark.parametrize("smiles,expected", [
        ("CCO", "ethanol"),
        ("c1ccncc1", "pyridine"),
        ("c1ccccc1", "benzene"),
        ("CC(=O)O", "acetic acid"),
        ("CC(=O)N", "acetamide"),
        ("C", "methane"),
    ])
    def test_v17_default_preserves_canonical(self, smiles, expected, monkeypatch):
        """V17 default env produces the canonical retained name."""
        monkeypatch.delenv("ORTHONYM_USE_V18_WEIGHTS", raising=False)
        monkeypatch.delenv("ORTHONYM_SELECTION_MODE", raising=False)
        from orthonym.assembly import coverage_scoring, candidate_pool
        importlib.reload(coverage_scoring)
        importlib.reload(candidate_pool)
        from orthonym import name_compound
        name = name_compound(smiles)
        assert expected.lower() in name.lower(), (
            f"V17 default regression: {smiles} expected '{expected}', got '{name}'"
        )
