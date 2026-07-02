"""PIN-policy tests: --trivial fallback flag + fail-closed default (Task 1.9)
and ledger-driven demotion of non-PIN trivial headline names (Task 1.10).

Task 1.9 = the *mechanism*: the default PIN pipeline fails closed (returns the
descriptive "unknown organic compound" fallback when no PIN is derivable);
``trivial_fallback=True`` (CLI ``--trivial``) opts into a FALLBACK-ONLY path
that returns a general-only (PIN-denied) retained name *only* when the default
pipeline could not derive a PIN. ``--trivial`` never downgrades a derivable PIN.

Task 1.10 = the *data*: specific non-PIN trivial names are demoted from the
default headline via ``pin: false`` deny rows (glycerol/allyl alcohol/
chloroform/catechol/nicotinic acid/dihydroxalate/dihydrotartrate/glyoxal),
a phosgene -> carbonyl-dichloride retained-PIN rename, an oxalic-acid
canonical-key fix, and an oxamic-acid retained-PIN add. The demoted trivial
strings remain reachable only via the ``--trivial`` fallback.

See .superpowers/sdd/task-1.9-1.10-context.md for the controller resolutions
and the empirically-verified demote table.
"""

import pytest

from orthonym.namer import Orthonym, name_compound


# A molecule the systematic engine genuinely cannot name (returns the
# "unknown organic compound" fallback at HEAD, robustly — even with the OPSIN
# validity gate disabled, which is the state the test conftest autouse fixture
# forces). Used as the fail-closed / fallback SEAM: no real in-scope molecule
# has an underivable PIN but a trivial name (controller-verified), so the
# mechanism is exercised by injecting a synthetic entry into
# GENERAL_RETAINED_NAMES for this SMILES.
_SEAM_SMILES = "[Se]=[Se]"
_SEAM_NAME = "SYNTHETIC-SEAM-TRIVIAL-NAME"


@pytest.fixture
def seam_general_name(monkeypatch):
    """Inject a synthetic general-only (PIN-denied) retained name for a
    molecule the systematic engine returns 'unknown organic compound' for.

    Restores the real GENERAL_RETAINED_NAMES dict on teardown (never mutates
    the shared module dict permanently).
    """
    import orthonym.data as D
    from rdkit import Chem

    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(_SEAM_SMILES))
    patched = dict(D.GENERAL_RETAINED_NAMES)
    patched[canonical] = _SEAM_NAME
    monkeypatch.setattr(D, "GENERAL_RETAINED_NAMES", patched)
    return canonical


# --------------------------------------------------------------------------
# Task 1.9: mechanism — fail-closed default + --trivial fallback
# --------------------------------------------------------------------------


@pytest.mark.unit
class TestTrivialFallbackMechanism:
    def test_seam_returns_unknown_at_baseline(self):
        """Sanity: the seam molecule has no derivable PIN (fails closed)."""
        # name_compound normalizes the raw 'unknown' failure signal to the
        # descriptive fallback string.
        assert name_compound(_SEAM_SMILES, style="pin") == "unknown organic compound"

    def test_default_fails_closed_no_trivial(self, seam_general_name):
        """Default pipeline fails closed even when a general-only trivial
        name exists for the molecule."""
        assert name_compound(_SEAM_SMILES, style="pin") == "unknown organic compound"

    def test_trivial_flag_falls_back(self, seam_general_name):
        """--trivial opts into the general-only retained name when (and only
        when) the default pipeline could not derive a PIN."""
        n = Orthonym(style="pin", trivial_fallback=True)
        assert n.name(_SEAM_SMILES) == _SEAM_NAME

    def test_pin_preferred_even_with_trivial_flag(self):
        """--trivial never downgrades a derivable PIN."""
        n = Orthonym(style="pin", trivial_fallback=True)
        assert n.name("CCO") == "ethanol"

    def test_trivial_flag_pin_still_wins_glycerol(self):
        """--trivial on glycerol SMILES still returns the systematic PIN
        (propane-1,2,3-triol), not the demoted trivial 'glycerol'."""
        assert name_compound("OCC(O)CO", style="pin",
                             trivial_fallback=True) == "propane-1,2,3-triol"

    def test_trivial_flag_reaches_demoted_trivial_via_fallback(self):
        """A demoted trivial that IS the only name for its molecule remains
        reachable via --trivial. glycerol has a systematic PIN so it never
        falls back; use the mechanism directly: name_compound threads the
        flag and the seam fixture proves the fallback branch. Here we assert
        the flag is threaded end-to-end via name_compound without error."""
        # PIN wins for a derivable case (regression guard for the thread).
        assert name_compound("C=CCO", style="pin",
                             trivial_fallback=True) == "prop-2-en-1-ol"

    def test_trivial_fallback_implies_triviality_controller(self):
        """Per context resolution 3: trivial_fallback=True implies
        enable_triviality_controller=True (one user intent)."""
        n = Orthonym(style="pin", trivial_fallback=True)
        assert n._enable_triviality_controller is True

    def test_name_compound_accepts_trivial_fallback_kwarg(self, seam_general_name):
        """name_compound threads trivial_fallback through to Orthonym."""
        assert name_compound(_SEAM_SMILES, style="pin") == "unknown organic compound"
        assert name_compound(_SEAM_SMILES, style="pin",
                             trivial_fallback=True) == _SEAM_NAME


# --------------------------------------------------------------------------
# Task 1.10: data — demote non-PIN trivials from the default headline
# --------------------------------------------------------------------------


@pytest.mark.unit
class TestDemoteNonPinTrivials:
    def test_default_emits_pin_not_trivial(self):
        """Every demote-table row emits the systematic/retained PIN, not the
        deprecated trivial name (context-file table, verbatim)."""
        cases = {
            "CC(C)c1ccccc1": "(propan-2-yl)benzene",       # not cumene
            "OCC(O)CO": "propane-1,2,3-triol",              # not glycerol
            "C=CCO": "prop-2-en-1-ol",                       # not allyl alcohol
            "ClC(Cl)Cl": "trichloromethane",                 # not chloroform
            "ClC(=O)Cl": "carbonyl dichloride",              # not phosgene
            "Oc1ccccc1O": "benzene-1,2-diol",                # not catechol
            "OC(=O)c1cccnc1": "pyridine-3-carboxylic acid",  # not nicotinic acid
            "OC(=O)C(=O)O": "oxalic acid",                   # not dihydroxalate
            "OC(C(O)C(=O)O)C(=O)O": "2,3-dihydroxybutanedioic acid",  # not dihydrotartrate
            "O=CC=O": "ethanedial",                          # not glyoxal
            "NC(=O)C(=O)O": "oxamic acid",                   # not 1-carbamoylmethanoic acid
        }
        for smi, pin in cases.items():
            assert name_compound(smi, style="pin") == pin, (
                f"{smi} should emit PIN {pin!r}"
            )

    def test_oxalic_acid_both_key_forms(self):
        """Oxalic acid resolves under the canonical key regardless of input
        SMILES orientation (canonical-key fix, not the old non-canonical key)."""
        assert name_compound("OC(=O)C(=O)O", style="pin") == "oxalic acid"
        assert name_compound("O=C(O)C(=O)O", style="pin") == "oxalic acid"

    def test_phosgene_is_carbonyl_dichloride_pin(self):
        """phosgene SMILES emits the functional-class retained PIN
        carbonyl dichloride (P-65.5.5.1)."""
        assert name_compound("ClC(=O)Cl", style="pin") == "carbonyl dichloride"


@pytest.mark.unit
class TestDemotedTrivialsAreDenied:
    """The demoted trivial strings must not appear as headline names in the
    merged retained dict (PIN-deny gate)."""

    def test_demoted_names_absent_from_merged(self):
        from orthonym.data import ALL_RETAINED_NAMES

        demoted = {
            "glycerol", "allyl alcohol", "chloroform", "phosgene",
            "catechol", "nicotinic acid", "dihydroxalate",
            "dihydrotartrate", "glyoxal",
        }
        present = {v for v in ALL_RETAINED_NAMES.values()
                   if v.lower() in demoted}
        assert present == set(), f"demoted trivials still headline: {present}"

    def test_demoted_names_present_in_general_fallback(self):
        """The demoted trivials remain reachable via GENERAL_RETAINED_NAMES."""
        from orthonym.data import GENERAL_RETAINED_NAMES

        general_values = {v.lower() for v in GENERAL_RETAINED_NAMES.values()}
        for name in ("glycerol", "allyl alcohol", "chloroform", "catechol",
                     "nicotinic acid", "glyoxal"):
            assert name in general_values, (
                f"{name!r} not reachable via GENERAL_RETAINED_NAMES"
            )


@pytest.mark.unit
class TestDemoteRegressionGuards:
    """The 1.5-revert blast radius: names that must stay UNCHANGED."""

    def test_dormant_general_only_acids_stay_inactive(self):
        """malonic/succinic must NOT activate (they were the 1.5 breakage)."""
        assert name_compound("OC(=O)CC(=O)O", style="pin") == "propanedioic acid"
        assert name_compound("OC(=O)CCC(=O)O", style="pin") == "butanedioic acid"

    def test_unrelated_retained_names_unchanged(self):
        assert name_compound("CC(=O)O", style="pin") == "acetic acid"
        assert name_compound("CCO", style="pin") == "ethanol"
        assert name_compound("OC(=O)c1ccccc1", style="pin") == "benzoic acid"
        assert name_compound("COc1ccccc1", style="pin") == "methoxybenzene"
        assert name_compound("O=Cc1ccco1", style="pin") == "furfural"

    def test_purine_indicated_h_unchanged(self):
        """Purine catalog must not be re-keyed (9H->7H was the 1.5 regression)."""
        assert name_compound("c1ncc2[nH]cnc2n1", style="pin") == \
            "1H-imidazo[4,5-d]pyrimidine"
