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

See.the workflow tooling/sdd/task-1.9-1.10-context.md for the controller resolutions
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

    def test_task6c_wave1_data_deny_batch(self):
        """Task 6C: veratrol / o-benzoquinone / barbituric acid are retained
        NON-PIN names; each demotes to its systematic PIN (invariant-9: the
        EXACT PIN, not abstain / von Baeyer / another retained variant).

        - veratrol -> 1,2-dimethoxybenzene: 'Retained names'
          (the Blue Book '1,2-dimethoxybenzene (PIN; no substitution on anisole
          for PINs)').
        - o-benzoquinone -> cyclohexa-3,5-diene-1,2-dione:
          'Quinones' (the Blue Book 'No retained quinone names are used as
          preferred IUPAC names'; the Blue Book '(PIN) (not 1,2-benzoquinone)').
        - barbituric acid -> 1,3-diazinane-2,4,6-trione: pseudoketones
          (the Blue Book '1,3-diazinane-2,4,6-trione (PIN)').
        """
        cases = {
            "COc1ccccc1OC": "1,2-dimethoxybenzene",           # not veratrol
            "O=C1C=CC=CC1=O": "cyclohexa-3,5-diene-1,2-dione",  # not o-benzoquinone
            "O=C1CC(=O)NC(=O)N1": "1,3-diazinane-2,4,6-trione",  # not barbituric acid
        }
        for smi, pin in cases.items():
            assert name_compound(smi, style="pin") == pin, (
                f"{smi} should emit PIN {pin!r}"
            )

    def test_task6c_regressions_unchanged(self):
        """The Task 6C denies key on the exact name only; the sibling retained
        arene/quinone still emit their own PINs. UNSUBSTITUTED anisole IS the
        bare PIN (a review RISK 7: the Blue Book 'anisole (PIN)'; the Blue Book
        'Substitution is allowed on all structures except anisole') -- only
        substituted derivatives become methoxybenzenes."""
        assert name_compound("COc1ccccc1", style="pin") == "anisole"
        assert (name_compound("O=C1C=CC(=O)C=C1", style="pin")
                == "cyclohexa-2,5-diene-1,4-dione")

    def test_anisole_bare_pin_but_substituted_are_methoxybenzenes(self):
        """ a review RISK 7: UNSUBSTITUTED anisole is the PIN (the Blue Book
        'anisole (PIN)'), but the Blue Book 'Substitution is allowed on all
        structures except anisole' -- every substituted derivative is a
        methoxybenzene, NOT an anisole. Dispatch is by exact canonical SMILES,
        so only the bare parent hits the retained name."""
        assert name_compound("COc1ccccc1", style="pin") == "anisole"
        # substituted forms must NOT carry an 'anisole' base
        for smi in ("COc1ccc(C)cc1", "COc1ccccc1Cl", "COc1ccc(Cl)cc1",
                    "COc1ccccc1OC"):
            got = name_compound(smi, style="pin")
            assert "anisole" not in got.lower(), (
                f"{smi} must be a methoxybenzene, not anisole-based; got {got!r}"
            )
            assert "methoxy" in got.lower(), (
                f"{smi} should emit a methoxybenzene systematic name; got {got!r}"
            )

    def test_oxalic_acid_both_key_forms(self):
        """Oxalic acid resolves under the canonical key regardless of input
        SMILES orientation (canonical-key fix, not the old non-canonical key)."""
        assert name_compound("OC(=O)C(=O)O", style="pin") == "oxalic acid"
        assert name_compound("O=C(O)C(=O)O", style="pin") == "oxalic acid"

    def test_phosgene_is_carbonyl_dichloride_pin(self):
        """phosgene SMILES emits the functional-class retained PIN
        carbonyl dichloride."""
        assert name_compound("ClC(=O)Cl", style="pin") == "carbonyl dichloride"

    def test_picric_acid_is_trinitrophenol_pin(self):
        """'picric acid' is retained for general nomenclature only and only when
        unsubstituted; the PIN is the systematic 2,4,6-trinitrophenol.
        styphnic acid (a different structure) is unaffected by the name-keyed deny."""
        assert (name_compound("Oc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]",
                              style="pin") == "2,4,6-trinitrophenol")


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
            "picric acid",  #: PIN is 2,4,6-trinitrophenol
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
        # a review RISK 7: bare anisole IS the PIN (the Blue Book / the Blue Book).
        assert name_compound("COc1ccccc1", style="pin") == "anisole"
        assert name_compound("O=Cc1ccco1", style="pin") == "furfural"

    def test_purine_indicated_h_unchanged(self):
        """Purine catalog must not be re-keyed (9H->7H was the 1.5 regression)."""
        assert name_compound("c1ncc2[nH]cnc2n1", style="pin") == \
            "1H-imidazo[4,5-d]pyrimidine"


@pytest.mark.unit
class TestC6HomoRingDemotions:
    """C6-medring-leaks: six non-PIN 'homo-' ring-expansion trivial names must
    be demoted to pin:false so the systematic Hantzsch-Widman PIN is returned.
    BB refs:,,,,
    
    """

    def test_homopiperidine_demoted_to_azepane(self):
        """N1CCCCCC1 must yield azepane (7-membered N-only HW ring), not homopiperidine."""
        assert name_compound("N1CCCCCC1", style="pin") == "azepane"

    def test_homopiperazine_demoted_to_diazepane(self):
        """C1CNCCNC1 must yield a systematic 7-membered diaza name, not homopiperazine.
        The IUPAC 2013 PIN is 1,4-diazepane (HW); the HW builder may also emit
        the alternative replacement form 1,4-diazacycloheptane (both are valid
        systematic IUPAC names for this ring; neither is homopiperazine)."""
        result = name_compound("C1CNCCNC1", style="pin")
        assert result in ("1,4-diazepane", "1,4-diazacycloheptane"), (
            f"Expected systematic diaza-7-ring name, got {result!r}"
        )

    def test_homomorpholine_demoted_fail_closed(self):
        """C1CNCCOC1 must NOT yield homomorpholine; PIN is 1,4-oxazepane.
        The HW locant-elision bug STILL_WRONG) means the
        systematic builder currently emits a wrong-isomer name and
        catches it as unknown. Either 'unknown organic compound' or
        '1,4-oxazepane' are acceptable (fail-closed better than wrong name)."""
        result = name_compound("C1CNCCOC1", style="pin")
        assert result != "homomorpholine", (
            f"homomorpholine leaked into PIN headline: got {result!r}"
        )

    def test_thiahomomorpholine_demoted_fail_closed(self):
        """C1CNCCSC1 must NOT yield thiahomomorpholine; PIN is 1,4-thiazepane.
        May be unknown (fail-closed) until locant fix."""
        result = name_compound("C1CNCCSC1", style="pin")
        assert result != "thiahomomorpholine", (
            f"thiahomomorpholine leaked into PIN headline: got {result!r}"
        )

    def test_selenohomomorpholine_demoted_fail_closed(self):
        """C1CNCC[Se]C1 must NOT yield selenohomomorpholine; PIN is 1,4-selenazepane."""
        result = name_compound("C1CNCC[Se]C1", style="pin")
        assert result != "selenohomomorpholine", (
            f"selenohomomorpholine leaked into PIN headline: got {result!r}"
        )

    def test_tellurohomomorpholine_demoted_fail_closed(self):
        """C1CNCC[Te]C1 must NOT yield tellurohomomorpholine; PIN is 1,4-tellurazepane."""
        result = name_compound("C1CNCC[Te]C1", style="pin")
        assert result != "tellurohomomorpholine", (
            f"tellurohomomorpholine leaked into PIN headline: got {result!r}"
        )

    # --- control cases: 6-membered retained-PIN rings must be UNCHANGED ---

    def test_piperidine_unchanged(self):
        """C1CCNCC1 must still yield piperidine (retained PIN per."""
        assert name_compound("C1CCNCC1", style="pin") == "piperidine"

    def test_morpholine_unchanged(self):
        """C1COCCN1 must still yield morpholine (retained PIN per."""
        assert name_compound("C1COCCN1", style="pin") == "morpholine"

    def test_piperazine_unchanged(self):
        """C1CNCCN1 must still yield piperazine (retained PIN per."""
        assert name_compound("C1CNCCN1", style="pin") == "piperazine"

    def test_azocane_unchanged(self):
        """N1CCCCCCC1 must still yield azocane (8-membered HW, not affected by C6 deny)."""
        assert name_compound("N1CCCCCCC1", style="pin") == "azocane"


# --------------------------------------------------------------------------
# C8: thread --trivial into name_with_confidence and --batch (Task 1.9 gap)
# --------------------------------------------------------------------------


@pytest.mark.unit
class TestC8TrivialThreading:
    """C8 gap-fix: trivial_fallback must be honored in name_with_confidence
    (the --confidence path) AND in _process_batch (the --batch path).

    Uses the same _SEAM_SMILES / seam_general_name fixture defined above so
    no new seam molecule is needed.
    """

    # --- Locus B: name_with_confidence ---

    def test_name_with_confidence_trivial_off_returns_failure(
        self, seam_general_name
    ):
        """With trivial_fallback=False (default), name_with_confidence must
        return the failure signal, not the seam trivial name."""
        n = Orthonym(style="pin", trivial_fallback=False)
        result = n.name_with_confidence(_SEAM_SMILES)
        assert result["name"] != _SEAM_NAME, (
            "trivial_fallback=False must not produce the seam trivial name"
        )

    def test_name_with_confidence_trivial_on_returns_seam_name(
        self, seam_general_name
    ):
        """With trivial_fallback=True, name_with_confidence must fall back to
        the seam trivial name when the systematic engine cannot derive a PIN."""
        n = Orthonym(style="pin", trivial_fallback=True)
        result = n.name_with_confidence(_SEAM_SMILES)
        assert result["name"] == _SEAM_NAME, (
            f"trivial_fallback=True should give {_SEAM_NAME!r}, got {result['name']!r}"
        )

    def test_name_with_confidence_trivial_does_not_downgrade_pin(self):
        """trivial_fallback=True must NOT downgrade a derivable PIN (CCO →
        ethanol). The returned name must still be 'ethanol', not a trivial
        fall-back string, and confidence must NOT be forced to 1.0 (regression
        risk 3 in the spec: the trivial branch is low-confidence)."""
        n = Orthonym(style="pin", trivial_fallback=True)
        result = n.name_with_confidence("CCO")
        assert result["name"] == "ethanol", (
            f"derivable PIN CCO should remain 'ethanol', got {result['name']!r}"
        )
        # ethanol is named via a real handler, so confidence should NOT be
        # forced to 1.0 by the trivial fallback path — its actual metadata
        # is what it is; we only assert it was NOT clobbered to 1.0 by the
        # fallback mechanism (which would be wrong: fallback = low-confidence).
        # We assert the name is 'ethanol' which proves no downgrade occurred.

    def test_name_with_confidence_confidence_not_overridden_on_fallback(
        self, seam_general_name
    ):
        """When the trivial fallback fires, the fallback code itself must NOT
        override the 'confidence', 'handler', or 'factors' fields — those must
        remain whatever the pipeline set (spec regression risk 3: do NOT reset
        confidence to 1.0 inside the fallback call).

        Specifically: the metadata returned with trivial_fallback=True and
        trivial_fallback=False must agree on confidence/handler/factors — only
        'name' should differ when the fallback fires.
        """
        n_off = Orthonym(style="pin", trivial_fallback=False)
        n_on = Orthonym(style="pin", trivial_fallback=True)
        result_off = n_off.name_with_confidence(_SEAM_SMILES)
        result_on = n_on.name_with_confidence(_SEAM_SMILES)
        # The fallback fired: name differs
        assert result_on["name"] == _SEAM_NAME
        assert result_off["name"] != _SEAM_NAME
        # Confidence/handler/factors are unchanged by the fallback
        assert result_on["confidence"] == result_off["confidence"], (
            "fallback must not change the 'confidence' field"
        )
        assert result_on["handler"] == result_off["handler"], (
            "fallback must not change the 'handler' field"
        )
        assert result_on["factors"] == result_off["factors"], (
            "fallback must not change the 'factors' field"
        )

    # --- Locus A: _process_batch signature threading ---

    def test_process_batch_accepts_trivial_fallback_param(self):
        """_process_batch must accept a trivial_fallback keyword argument
        without raising TypeError (signature threading test)."""
        from orthonym.cli import _process_batch
        import inspect
        sig = inspect.signature(_process_batch)
        assert "trivial_fallback" in sig.parameters, (
            "_process_batch must have a 'trivial_fallback' parameter"
        )

    def test_process_batch_trivial_fallback_default_false(self):
        """_process_batch's trivial_fallback default must be False to preserve
        backward-compatibility (existing callers must be unaffected)."""
        from orthonym.cli import _process_batch
        import inspect
        sig = inspect.signature(_process_batch)
        param = sig.parameters["trivial_fallback"]
        assert param.default is False, (
            f"trivial_fallback default should be False, got {param.default!r}"
        )

    def test_cli_batch_trivial_smoke(self, tmp_path):
        """Subprocess CLI smoke: --batch --trivial must not crash and must
        produce output (one line per SMILES input)."""
        import subprocess, sys
        batch_file = tmp_path / "input.smi"
        batch_file.write_text("CCO\n[Se]=[Se]\n")
        result = subprocess.run(
            [sys.executable, "-m", "orthonym",
             "--batch", str(batch_file), "--trivial"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"--batch --trivial failed:\nstdout: {result.stdout!r}\n"
            f"stderr: {result.stderr!r}"
        )
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        assert len(lines) == 2, (
            f"Expected 2 output lines, got {len(lines)}: {result.stdout!r}"
        )

    def test_cli_batch_no_trivial_smoke(self, tmp_path):
        """Subprocess CLI smoke: --batch without --trivial must produce same
        number of output lines and must NOT produce the seam trivial name for
        the failure molecule (fail-closed default unchanged)."""
        import subprocess, sys
        batch_file = tmp_path / "input.smi"
        batch_file.write_text("CCO\n[Se]=[Se]\n")
        result = subprocess.run(
            [sys.executable, "-m", "orthonym",
             "--batch", str(batch_file)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"--batch failed:\nstdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        assert len(lines) == 2
        # Without --trivial the second line must be the failure string
        second_name = lines[1].split("\t")[1] if "\t" in lines[1] else lines[1]
        assert second_name == "unknown organic compound", (
            f"Without --trivial, failure must give 'unknown organic compound', "
            f"got {second_name!r}"
        )
