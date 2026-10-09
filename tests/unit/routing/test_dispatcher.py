"""a phase unit tests for ``orthonym.routing.dispatcher`` (ClassFirstRouter).

Per internal notes LOCKED test pyramid floor: ≥ 30 dispatcher behavior tests.
Per internal notes + the audit: env-var-gated audit log; default-OFF for
 stdout-byte-identical preservation.
Per internal notes +: per-instance dispatch_stats counter; no module-global state.
Per internal notes +: dispatch determinism + mol non-mutation invariant.
Per internal notes +: NO ``@pytest.mark.xfail`` markers.

Test classes per concern:

- ``TestRouterConstruction`` — ClassFirstRouter default + env-var gating + override.
- ``TestPerStoutClassDispatch`` — parametrized over (smiles, expected_name)
  representatives mined from the live canary corpus: every StoutClass dispatches
  for its representative SMILES.
- ``TestDispatchPurity`` — + invariants: determinism, no mol mutation,
  per-instance state isolation, defensive raise on impossible-state.
- ``TestStatsCounter`` — + telemetry counter contract.
- ``TestEdgeCases`` — empty/single-atom/dot-disconnected SMILES.
- ``TestAuditLogGating`` — env-var-gated INFO log; -preserving
  default-OFF behavior.

The Class 2 parametrize over ~ 19 STOUTCLASS_REPRESENTATIVES + ~ 16 explicit
tests in the other classes yields > 30 tests well above the floor.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import pytest
from rdkit import Chem

from orthonym import Orthonym, name_compound
from orthonym.routing.dispatch_table import DISPATCH_TABLE, StoutClass
from orthonym.routing.dispatcher import (
    ORTHONYM_DISPATCH_AUDIT_ENV_VAR,
    ClassFirstRouter,
)


# ---------------------------------------------------------------------------
# Representatives mined from the live canary corpus (the audit column
# byte_identical_canary_fixture_id, cross-referenced against
# tests/canary/canary_pre_cfr_158.csv via Plan-03 development bench).
#
# One (smiles, expected_name) per StoutClass; expected_name is the
# byte-identical Plan-01 frozen baseline output.
# ---------------------------------------------------------------------------

STOUTCLASS_REPRESENTATIVES: "OrderedDict[StoutClass, tuple]" = OrderedDict(
    [
        (StoutClass.SALT,                    ("[Na+].[Cl-]",            "sodium chloride")),
        (StoutClass.RADICAL,                 ("[CH3]",                  "methyl")),
        (StoutClass.ZWITTERION,              ("[NH3+]CC(=O)[O-]",       "glycine")),
        (StoutClass.ANION_RETAINED,          ("CC(=O)[O-]",             "acetate")),
        # 12541211e (cluster C): an alkylammonium is a substitutive aminium PIN, so
        # C[N+](C)(C)C routes via CATION_QUATERNARY@480 (below). CATION_RETAINED keeps the
        # parent-hydride cation: "Cations centers in parent hydrides",
        # "General rule for systematically naming cationic centers in parent hydrides"
        # (the Blue Book) 'H4N+ azanium (preselected name)'.
        (StoutClass.CATION_RETAINED,         ("[NH4+]",                 "azanium")),
        (StoutClass.CATION_QUATERNARY,       ("C[N+](C)(C)C",           "N,N,N-trimethylmethanaminium")),  #, the Blue Book '(PIN)'
        (StoutClass.ANION_SMALL,             ("C(=O)([O-])CCCCCC",      "heptanoate")),
        (StoutClass.POLY_ANION,              ("[O-]C(=O)CCCC(=O)[O-]",  "pentanedioate")),
        (StoutClass.MULTI_COMPONENT_NEUTRAL, ("CCO.OCC",                "ethanol ethanol")),
        (StoutClass.MULTIPLICATIVE,          ("c1ccc(Cc2ccccc2)cc1",    "1,1'-methylenedibenzene")),
        (StoutClass.CARBOHYDRATE_LOOKUP,     ("OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
                                              "α-D-glucopyranose")),
        (StoutClass.NATURAL_PRODUCT,         (
            "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@@]34C)[C@@H]1CC[C@@H]2O",
            "(8R,9S,10S,13S,14S,17S)-17-hydroxyandrost-4-en-3-one",
        )),
        #: peptide PIN is the SUBSTITUTIVE form (V38-PEPTIDE-PIN-VERDICT.md); RT verified.
        (StoutClass.PEPTIDE,                 ("NCC(=O)NCC(=O)O",        "(2-aminoacetamido)acetic acid")),
        (StoutClass.RETAINED_NAME,           ("CCO",                    "ethanol")),
        # A defined-configuration amino acid carries its D/L prefix: "The
        # stereodescriptors 'D' and 'L'" (the Blue Book-54295, "as drawn below for
        # L-alanine"); OPSIN parses 'L-alanine' to this input's full InChIKey.
        (StoutClass.AMINO_ACID,              ("C[C@H](N)C(=O)O",        "L-alanine")),
        # fix a performance pass: (the Blue Book) -- the 'a' name is the PIN
        # only with four or more heterounits, so the representative is the BB's own
        # four-O chain, "(4) 2,5,8,11-tetraoxadodecane (PIN)" (:27762); the former
        # 3-O representative COCCOCCOC is now substitutive (:27756).
        (StoutClass.SKELETAL_REPLACEMENT,    ("COCCOCCOCCOC",           "2,5,8,11-tetraoxadodecane")),
        # Wave2 fail-closed: the CYCLOPHANE entry still matches and
        # dispatches, but production REFUSES the composed bracket-prefix name
        # ('[3.3]orthocyclophane' — no phane form is OPSIN-parseable) via the
        # G0 UNSUPPORTED_RING_SYSTEM signal.
        (StoutClass.CYCLOPHANE,              ("C1CCc2ccccc2CCCc2ccccc21",
                                              "unknown organic compound")),
        #: the acyl word was 'palmitate'. Corrected to the PIN --
        # (the Blue Book) retains only formic/oxalic/acetic/
        # benzoic/oxamic as PINs, and (:29860) sends the rest to
        # general nomenclature;:29787 prints '(PIN)' on 'hexadecanoic acid'.
        # The row exists to pin DISPATCH, so the spelling is incidental to it.
        (StoutClass.DECOMPOSITION_PRE_GENERAL,
                                              ("CCCCCCCCCCCCCCCC(=O)OCCC",
                                              "propyl hexadecanoate")),
        # GENERAL: a SMILES whose final dispatch class (after cascade fall-through)
        # is GENERAL. Heptan-1-ol is not in retained names and falls through
        # to the GENERAL pipeline after DECOMPOSITION_PRE_GENERAL declines.
        (StoutClass.GENERAL,                 ("CCCCCCCO",               "heptan-1-ol")),
    ]
)


# ---------------------------------------------------------------------------
# Class 1 — Router construction (~ 3 tests)
# ---------------------------------------------------------------------------


class TestRouterConstruction:
    """internal notes +: ClassFirstRouter construction + env-var defaults."""

    def test_default_construction_no_audit_log(self, monkeypatch):
        """internal notes: default-OFF audit log preserves stdout-byte-identical."""
        monkeypatch.delenv(ORTHONYM_DISPATCH_AUDIT_ENV_VAR, raising=False)
        r = ClassFirstRouter()
        assert r._audit_log is False

    def test_audit_log_kwarg_overrides_env_var(self, monkeypatch):
        """internal notes: explicit ``_audit_log`` kwarg overrides env var (for tests)."""
        monkeypatch.setenv(ORTHONYM_DISPATCH_AUDIT_ENV_VAR, "1")
        r = ClassFirstRouter(_audit_log=False)
        assert r._audit_log is False

    def test_env_var_set_enables_audit_log(self, monkeypatch):
        """internal notes: ``ORTHONYM_DISPATCH_AUDIT=1`` enables INFO logging by default."""
        monkeypatch.setenv(ORTHONYM_DISPATCH_AUDIT_ENV_VAR, "1")
        r = ClassFirstRouter()
        assert r._audit_log is True


# ---------------------------------------------------------------------------
# Class 2 — Per-StoutClass dispatch (parametrized over STOUTCLASS_REPRESENTATIVES)
# ---------------------------------------------------------------------------


class TestPerStoutClassDispatch:
    """internal notes: each StoutClass produces its byte-identical name + dispatches to
    the expected entry (verified via the per-instance dispatch_stats counter)."""

    @pytest.mark.parametrize(
        "class_id,smiles,expected_name",
        [(c, s, n) for c, (s, n) in STOUTCLASS_REPRESENTATIVES.items()],
        ids=[c.name for c in STOUTCLASS_REPRESENTATIVES.keys()],
    )
    def test_per_stoutclass_dispatch(self, class_id, smiles, expected_name):
        """internal notes-CFR.md: each StoutClass dispatches for its representative SMILES.

        The byte-identical contract is on NAME OUTPUT only; the dispatch_stats
        counter confirms the routing decision (the audit column traceability).
        """
        namer = Orthonym(style="pin")
        name = namer.name(smiles)
        assert name == expected_name, (
            f"{class_id.name}: expected={expected_name!r}, got={name!r}; "
            f"SMILES={smiles!r}"
        )
        stats = namer.get_dispatch_stats()
        assert stats.get(class_id, 0) >= 1, (
            f"{class_id.name} did not appear in dispatch_stats for SMILES "
            f"{smiles!r}; got stats={dict(stats)}.  Either the predicate failed "
            f"to match or a higher-priority predicate consumed the SMILES."
        )


# ---------------------------------------------------------------------------
# Class 3 — Dispatch purity invariants (~ 4 tests)
# ---------------------------------------------------------------------------


class TestDispatchPurity:
    """CONTEXT D-26 + RL-5: predicate purity → dispatch determinism + mol non-mutation."""

    def test_dispatch_is_deterministic(self):
        """D-26: same input → same dispatch decision across independent calls."""
        router = ClassFirstRouter()
        mol = Chem.MolFromSmiles("CCO")
        smi = "CCO"
        canonical = Chem.MolToSmiles(mol, canonical=True)
        result1 = router.dispatch(mol, smi, canonical, None, _style="pin")
        result2 = router.dispatch(mol, smi, canonical, None, _style="pin")
        assert result1.class_id == result2.class_id

    def test_dispatch_does_not_mutate_mol(self):
        """ concrete check: dispatch must not mutate input mol properties."""
        router = ClassFirstRouter()
        mol = Chem.MolFromSmiles("CC(=O)O")
        before_props = mol.GetPropsAsDict()
        before_smi = Chem.MolToSmiles(mol, canonical=True)
        router.dispatch(mol, "CC(=O)O", before_smi, None, _style="pin")
        after_props = mol.GetPropsAsDict()
        after_smi = Chem.MolToSmiles(mol, canonical=True)
        assert before_props == after_props, (
            f"mol props mutated by dispatch; before={before_props}, after={after_props}"
        )
        assert before_smi == after_smi, (
            f"mol canonical SMILES drift; before={before_smi}, after={after_smi}"
        )

    def test_dispatch_pure_across_independent_routers(self):
        """D-16 + AP-6: per-instance counter — no leakage between router instances."""
        r1 = ClassFirstRouter()
        r2 = ClassFirstRouter()
        mol = Chem.MolFromSmiles("CCO")
        r1.dispatch(mol, "CCO", "CCO", None, _style="pin")
        # r2's stats remain empty
        assert r2.get_dispatch_stats() == {}
        assert sum(r1.get_dispatch_stats().values()) == 1

    def test_dispatch_raises_on_general_unreachable(self, monkeypatch):
        """ defensive raise: if DISPATCH_TABLE has no matching entry, raise.

        This is unreachable by construction in production (GENERAL's
        ``lambda *_: True`` always matches), but the defensive raise is part of
        the honest-fail-on-data contract — verify the code path.

        We mutate the dispatcher's view of the table via monkeypatch on the
        ``DISPATCH_TABLE`` symbol re-exported in ``dispatcher`` module; only
        the dispatcher's iteration is affected. No table fields are mutated.
        """
        # Build a replacement table that omits all entries — dispatch must raise.
        empty = OrderedDict()
        import orthonym.routing.dispatcher as dispatcher_mod
        monkeypatch.setattr(dispatcher_mod, "DISPATCH_TABLE", empty)
        router = ClassFirstRouter()
        mol = Chem.MolFromSmiles("CCO")
        with pytest.raises(RuntimeError, match="impossible state"):
            router.dispatch(mol, "CCO", "CCO", None)


# ---------------------------------------------------------------------------
# Class 4 — Stats counter contract (~ 4 tests)
# ---------------------------------------------------------------------------


class TestStatsCounter:
    """internal notes +: per-instance dispatch_stats Counter contract."""

    def test_stats_increment_on_dispatch(self):
        """: each dispatch call increments the matching class's counter."""
        router = ClassFirstRouter()
        mol = Chem.MolFromSmiles("CCO")
        for _ in range(5):
            router.dispatch(mol, "CCO", "CCO", None, _style="pin")
        stats = router.get_dispatch_stats()
        assert sum(stats.values()) == 5

    def test_reset_zeroes_stats(self):
        """: ``reset_dispatch_stats`` clears all counters."""
        router = ClassFirstRouter()
        mol = Chem.MolFromSmiles("CCO")
        router.dispatch(mol, "CCO", "CCO", None, _style="pin")
        assert sum(router.get_dispatch_stats().values()) == 1
        router.reset_dispatch_stats()
        assert router.get_dispatch_stats() == {}

    def test_stats_defensive_copy(self):
        """: ``get_dispatch_stats`` returns a defensive copy (no leak)."""
        router = ClassFirstRouter()
        mol = Chem.MolFromSmiles("CCO")
        router.dispatch(mol, "CCO", "CCO", None, _style="pin")
        s1 = router.get_dispatch_stats()
        # Mutate the caller's view; internal state must not be affected.
        s1[StoutClass.SALT] = 999
        s2 = router.get_dispatch_stats()
        assert s2.get(StoutClass.SALT, 0) != 999, (
            f"defensive copy violation; caller mutation leaked to router state."
        )

    def test_stat_keys_match_stoutclass(self):
        """: STAT_KEYS class constant enumerates StoutClass member values."""
        assert set(ClassFirstRouter.STAT_KEYS) == {c.value for c in StoutClass}


# ---------------------------------------------------------------------------
# Class 5 — Edge cases (~ 3 tests)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """ behavior preservation on edge-case SMILES."""

    def test_dispatch_on_empty_smiles_v18_unknown(self):
        """ contract preserved: empty SMILES returns "unknown" (no exception).

        This mirrors the cascade behavior — an empty SMILES parses to an
        empty molecule, falls through every predicate, and the GENERAL pipeline
        produces "unknown" via the _descriptive_fallback path. a phase
        substrate preserves this byte-identical per.
        """
        result = name_compound("")
        assert result == "unknown"

    def test_dispatch_on_single_atom(self):
        """ contract: single-atom SMILES (carbon) names to "methane" (retained)."""
        result = name_compound("C")
        assert result == "methane"

    def test_dispatch_on_dot_disconnected(self):
        """internal notes-CFR.md row 8: dot-disconnected neutral multi-atom fragments
        dispatch to MULTI_COMPONENT_NEUTRAL (verified end-to-end via name + stats)."""
        namer = Orthonym(style="pin")
        name = namer.name("CCO.OCC")
        # Byte-identical baseline (mined from canary corpus + Plan-03 dev bench)
        assert name == "ethanol ethanol"
        stats = namer.get_dispatch_stats()
        assert stats.get(StoutClass.MULTI_COMPONENT_NEUTRAL, 0) >= 1


# ---------------------------------------------------------------------------
# Class 6 — Audit log gating (~ 2 tests)
# ---------------------------------------------------------------------------


class TestAuditLogGating:
    """internal notes: env-var-gated INFO log; default-OFF for stdout-byte-identical."""

    def test_audit_log_emits_when_env_var_set(self, monkeypatch, caplog):
        """: ``ORTHONYM_DISPATCH_AUDIT=1`` causes one INFO record per dispatch."""
        monkeypatch.setenv(ORTHONYM_DISPATCH_AUDIT_ENV_VAR, "1")
        router = ClassFirstRouter()
        assert router._audit_log is True
        mol = Chem.MolFromSmiles("CCO")
        with caplog.at_level(logging.INFO, logger="orthonym.routing.dispatcher"):
            router.dispatch(mol, "CCO", "CCO", None, _style="pin")
        cfr_records = [
            r for r in caplog.records if "CFR dispatch:" in r.getMessage()
        ]
        assert len(cfr_records) >= 1, (
            f"expected ≥ 1 'CFR dispatch:' INFO record; got "
            f"{[r.getMessage() for r in caplog.records]!r}"
        )

    def test_audit_log_silent_when_env_var_unset(self, monkeypatch, caplog):
        """D-08: env var unset → ZERO INFO records (CFR-04 stdout-byte-identical preserved)."""
        monkeypatch.delenv(ORTHONYM_DISPATCH_AUDIT_ENV_VAR, raising=False)
        router = ClassFirstRouter()
        assert router._audit_log is False
        mol = Chem.MolFromSmiles("CCO")
        with caplog.at_level(logging.INFO, logger="orthonym.routing.dispatcher"):
            router.dispatch(mol, "CCO", "CCO", None, _style="pin")
        cfr_records = [
            r for r in caplog.records if "CFR dispatch:" in r.getMessage()
        ]
        assert len(cfr_records) == 0, (
            f"expected ZERO 'CFR dispatch:' INFO records (default-OFF); got "
            f"{[r.getMessage() for r in cfr_records]!r}"
        )
