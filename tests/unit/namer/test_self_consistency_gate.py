""" a phase  — constitutional self-consistency gate unit tests.

The gate suppresses an emitted name ONLY when OPSIN re-perceives it as a
CONSTITUTIONALLY DIFFERENT molecule than the input. It must be stereo-insensitive,
tautomer-/aromaticity-tolerant, and fail-OPEN on every inconclusive outcome. These
tests exercise the pure comparison helpers directly and the decision/gate logic with
the OPSIN oracle monkeypatched (no Java needed)."""
import pytest

import orthonym.namer as nm


# ---------------------------------------------------------------------------
# Pure constitutional-comparison helpers (RDKit only)
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestVerdict:
    def test_same_molecule_ok(self):
        assert nm._self_consistency_verdict("CCO", "CCO") == "ok"
        assert nm._self_consistency_verdict("CCO", "OCC") == "ok"  # same mol, diff spelling

    def test_different_molecule_mismatch(self):
        assert nm._self_consistency_verdict("CCO", "CCCO") == "mismatch"
        # the audit class: a dropped substituent (loses the F)
        assert nm._self_consistency_verdict("Fc1ccc2CCCCc2c1", "c1ccc2CCCCc2c1") == "mismatch"

    def test_stereo_only_is_ok(self):
        # R vs S — constitution identical; the gate must NOT suppress (-07)
        assert nm._self_consistency_verdict("C[C@H](O)CC", "C[C@@H](O)CC") == "ok"
        # E vs Z
        assert nm._self_consistency_verdict(r"C/C=C/C", r"C/C=C\C") == "ok"

    def test_tautomer_is_ok(self):
        # adenine 7H vs 9H tautomers normalize to the same standard-InChI skeleton
        assert nm._self_consistency_verdict(
            "Nc1ncnc2[nH]cnc12", "Nc1ncnc2nc[nH]c12") == "ok"

    def test_unparseable_is_inconclusive(self):
        assert nm._self_consistency_verdict("CCO", "this is not smiles!!!") == "inconclusive"
        assert nm._self_consistency_verdict("?!?", "CCO") == "inconclusive"

    def test_charge_balanced_salt_named_as_net_charged_species_is_mismatch(self):
        # fix a performance pass (wp1): a neutral SALT input is not the neutral-molecule
        # exemption. (the Blue Book) "Neutral salts of acids are
        # named by citing the name of the cation(s) followed by the name of the
        # anion": the name must denote a charge-balanced assembly. These are the
        # OPSIN 2.9.0 parses of 'trisodium 5-hydroxybenzene-1,3-disulfonate' and
        # 'trisodium (4-hydroxyphenyl)phosphonate' (net +1), which shipped as
        # pin_verified for the trianion salts.
        assert nm._self_consistency_verdict(
            "[O-]c1cc(cc(c1)S([O-])(=O)=O)S([O-])(=O)=O.[Na+].[Na+].[Na+]",
            "OC=1C=C(C=C(C1)S(=O)(=O)[O-])S(=O)(=O)[O-].[Na+].[Na+].[Na+]",
        ) == "mismatch"
        assert nm._self_consistency_verdict(
            "[O-]c1ccc(cc1)P(=O)([O-])[O-].[Na+].[Na+].[Na+]",
            "OC1=CC=C(C=C1)P([O-])([O-])=O.[Na+].[Na+].[Na+]",
        ) == "mismatch"

    def test_neutral_input_keeps_the_protonation_exemption(self):
        # 'methyl phosphate' round-trips to the dianion; the input has no formal
        # charge, so this is protonation ambiguity, not a wrong molecule.
        assert nm._self_consistency_verdict(
            "COP(=O)(O)O", "COP(=O)([O-])[O-]") == "ok"
        # a correct salt / zwitterion / hydrochloride name still passes
        assert nm._self_consistency_verdict(
            "CC(=O)[O-].[Na+]", "CC(=O)[O-].[Na+]") == "ok"
        assert nm._self_consistency_verdict("CC[NH3+].[Cl-]", "CCN.Cl") == "ok"

    def test_skeleton_stereo_insensitive(self):
        # the skeleton block is identical regardless of stereo descriptors
        assert nm._self_consistency_skeleton("C[C@H](O)CC") == nm._self_consistency_skeleton("CC(O)CC")


# ---------------------------------------------------------------------------
# Decision logic (mode behaviour) — opsin_smiles supplied directly, no Java
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestDecision:
    def _decide(self, monkeypatch, mode, name, smiles, opsin_smiles):
        monkeypatch.setattr(nm, "_SC_MODE", mode)
        return nm._self_consistency_decision(name, smiles, opsin_smiles, {})

    def test_off_is_noop(self, monkeypatch):
        # even a blatant mismatch ships when disabled
        assert self._decide(monkeypatch, "off", "propan-1-ol", "CCO", "CCCO") == "propan-1-ol"

    def test_warn_ships_but_counts(self, monkeypatch):
        monkeypatch.setattr(nm, "_SC_MODE", "warn")
        stats = {}
        out = nm._self_consistency_decision("propan-1-ol", "CCO", "CCCO", stats)
        assert out == "propan-1-ol"  # warn-only never changes output
        assert stats.get("self_consistency_mismatch") == 1
        assert "self_consistency_suppressed" not in stats

    def test_on_suppresses_mismatch(self, monkeypatch):
        monkeypatch.setattr(nm, "_SC_MODE", "on")
        stats = {}
        out = nm._self_consistency_decision("propan-1-ol", "CCO", "CCCO", stats)
        assert out != "propan-1-ol"          # suppressed to the descriptive fallback
        assert out == nm._descriptive_fallback("CCO")
        assert stats.get("self_consistency_suppressed") == 1

    def test_on_ships_correct_name(self, monkeypatch):
        # OPSIN re-perceives the SAME molecule -> ship even in suppress mode
        assert self._decide(monkeypatch, "on", "ethanol", "CCO", "CCO") == "ethanol"

    def test_on_ships_stereo_only_diff(self, monkeypatch):
        assert self._decide(monkeypatch, "on", "(R)-butan-2-ol",
                            "C[C@H](O)CC", "C[C@@H](O)CC") == "(R)-butan-2-ol"

    def test_on_fails_closed_when_inconclusive(self, monkeypatch):
        # OPSIN SMILES unparseable by RDKit -> inconclusive -> nothing was compared,
        # so the name ships only after a full-key round trip (claims conformance
        # R19, 2026-09-27; it used to fail OPEN). 'weird' has none -> withdrawn.
        monkeypatch.setattr(nm, "_validity_gate_name_to_smiles", lambda n: None)
        monkeypatch.setattr(nm, "_validity_gate_status", lambda n: "rejected")
        out = self._decide(monkeypatch, "on", "weird", "CCO", "@@bad@@")
        assert out != "weird"
        assert out == nm._descriptive_fallback("CCO")

    def test_on_inconclusive_ships_after_a_full_key_round_trip(self, monkeypatch):
        # The same unmade comparison, but the full-key round trip of the name
        # itself passes (OPSIN reads it back to the input) -> it ships.
        monkeypatch.setattr(nm, "_self_consistency_verdict",
                            lambda *a, **k: "inconclusive")
        monkeypatch.setattr(nm, "_validity_gate_name_to_smiles", lambda n: "CCO")
        assert self._decide(monkeypatch, "on", "ethanol", "CCO", "@@bad@@") == "ethanol"

    def test_warn_still_ships_when_inconclusive(self, monkeypatch):
        # the warn configuration keeps its documented meaning
        assert self._decide(monkeypatch, "warn", "weird", "CCO", "@@bad@@") == "weird"

    def test_no_input_smiles_ships(self, monkeypatch):
        assert self._decide(monkeypatch, "on", "ethanol", None, "CCCO") == "ethanol"


# ---------------------------------------------------------------------------
# Gate integration — monkeypatch the OPSIN oracle calls
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestGateIntegration:
    def test_parsed_self_consistent_ships(self, monkeypatch):
        monkeypatch.setattr(nm, "_DISABLE_VALIDITY_GATE", False)
        monkeypatch.setattr(nm, "_SC_MODE", "on")
        monkeypatch.setattr(nm, "_validity_gate_jar_present", lambda: True)
        monkeypatch.setattr(nm, "_validity_gate_name_to_smiles", lambda n: "CCO")  # OPSIN -> ethanol
        assert nm._final_opsin_validity_gate("ethanol", "CCO", {}) == "ethanol"

    def test_parsed_different_molecule_suppressed(self, monkeypatch):
        monkeypatch.setattr(nm, "_DISABLE_VALIDITY_GATE", False)
        monkeypatch.setattr(nm, "_SC_MODE", "on")
        monkeypatch.setattr(nm, "_validity_gate_jar_present", lambda: True)
        # OPSIN parses 'propan-1-ol' to CCCO but the input was CCO -> different molecule
        monkeypatch.setattr(nm, "_validity_gate_name_to_smiles", lambda n: "CCCO")
        out = nm._final_opsin_validity_gate("propan-1-ol", "CCO", {})
        assert out == nm._descriptive_fallback("CCO")

    def test_parsed_different_molecule_warn_ships(self, monkeypatch):
        monkeypatch.setattr(nm, "_DISABLE_VALIDITY_GATE", False)
        monkeypatch.setattr(nm, "_SC_MODE", "warn")
        monkeypatch.setattr(nm, "_validity_gate_jar_present", lambda: True)
        monkeypatch.setattr(nm, "_validity_gate_name_to_smiles", lambda n: "CCCO")
        assert nm._final_opsin_validity_gate("propan-1-ol", "CCO", {}) == "propan-1-ol"

    def test_jar_absent_fails_open(self, monkeypatch):
        monkeypatch.setattr(nm, "_DISABLE_VALIDITY_GATE", False)
        monkeypatch.setattr(nm, "_SC_MODE", "on")
        monkeypatch.setattr(nm, "_validity_gate_jar_present", lambda: False)
        # never even consult OPSIN; ship as-is
        assert nm._final_opsin_validity_gate("propan-1-ol", "CCO", {}) == "propan-1-ol"

    def test_rejected_name_still_suppressed(self, monkeypatch):
        # the EXISTING parseability gate is preserved: name_to_smiles None + rejected -> fallback
        monkeypatch.setattr(nm, "_DISABLE_VALIDITY_GATE", False)
        monkeypatch.setattr(nm, "_SC_MODE", "on")
        monkeypatch.setattr(nm, "_validity_gate_jar_present", lambda: True)
        monkeypatch.setattr(nm, "_validity_gate_name_to_smiles", lambda n: None)
        monkeypatch.setattr(nm, "_validity_gate_status", lambda n: "rejected")
        out = nm._final_opsin_validity_gate("not-a-real-name-xyz", "CCO", {})
        assert out == nm._descriptive_fallback("CCO")

    def test_unavailable_fails_closed(self, monkeypatch):
        # name_to_smiles None + parse_status 'unavailable' (transient, jar present,
        # the oracle's retry ladder exhausted) -> nothing verified -> suppress.
        # TRIAGE g7 C01 (2026-09-27): this used to ship the name (fail OPEN).
        monkeypatch.setattr(nm, "_DISABLE_VALIDITY_GATE", False)
        monkeypatch.setattr(nm, "_SC_MODE", "on")
        monkeypatch.setattr(nm, "_validity_gate_jar_present", lambda: True)
        monkeypatch.setattr(nm, "_validity_gate_name_to_smiles", lambda n: None)
        monkeypatch.setattr(nm, "_validity_gate_status", lambda n: "unavailable")
        assert nm._final_opsin_validity_gate("ethanol", "CCO", {}) == \
            nm._descriptive_fallback("CCO")
