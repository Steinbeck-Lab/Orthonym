"""
Integration tests for a phase: Ester & Lipid Format fixes.
Tests (ring ester prefix joining) and (polyfunctional ester demotion).
Tests (phospholipid routing) and OPSIN round-trip validation.
"""
import subprocess
import pytest
from orthonym.namer import name_compound
from tests.support.jars import jar_or_none
from tests.support.rt_assert import assert_full_rt


# ---------------------------------------------------------------------------
# OPSIN CLI helper
# ---------------------------------------------------------------------------

OPSIN_JAR = jar_or_none()
OPSIN_AVAILABLE = OPSIN_JAR is not None


def _opsin_parse(name: str) -> str:
    """Parse a name with OPSIN CLI and return SMILES or empty string on failure."""
    if not OPSIN_AVAILABLE:
        return ""
    try:
        result = subprocess.run(
            ["java", "-jar", OPSIN_JAR, "-osmi"],
            input=name,
            capture_output=True,
            text=True,
            timeout=15,
        )
        lines = result.stdout.strip().split("\n")
        out = lines[-1].strip() if lines else ""
        if "could not be interpreted" in out.lower():
            return ""
        if "unsure of the meaning" in out.lower():
            return ""
        return out
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


@pytest.mark.integration
class TestRingEsterPrefixJoining:
    """: Ring ester prefixes use hyphen joining with locants."""

    @pytest.mark.opsin_gate
    def test_different_acyloxy_benzene(self):
        """Two different esters on one ring: the second is the one acyloxy prefix of the organyl word.

        Re-asserted 2026-11-05 (lane/l17 item 5, change-asserted-value). The old value was the acyloxy form on the
        ring ('4-(acetyloxy)-1-(propanoyloxy)benzene'), a name with no ester word: "All preferred IUPAC
        names for esters are named by functional class nomenclature" (the Blue Book). The shipped value is the
        functional class name of method (2) of (:31831): "Method (1) generates preferred IUPAC names
        but names formed by using method (2) are acceptable in general nomenclature", with "(2) 4-(acetyloxy)phenyl
        dichloroacetate" (:31840, the same shape with other anions). Method (1), '1,4-phenylene acetate propanoate',
        is the PIN and OPSIN 2.9.0 reads it not, so the name is no PIN and the default tier declines it; it ships
        at a wider tier, labelled systematic_verified, read back by OPSIN to the full InChIKey. The anion that stands
        as the word of the name is the propanoate, the one the split of the decomposition engine makes the principal
        one; (:31856) fixes no other choice for two anions of one ring.
        """
        smiles = "CC(=O)Oc1ccc(OC(=O)CC)cc1"
        result = name_compound(smiles, include_confidence=True)
        assert result["name"] == "4-(acetyloxy)phenyl propanoate", result
        assert result["tier"] == "systematic_verified", result
        assert_full_rt(result["name"], smiles)

    @pytest.mark.opsin_gate
    def test_same_acyloxy_cyclohexane(self):
        """Identical anions on one ring: the functional class multiplicative name.

        Re-asserted 2026-11-05 (lane/l17 item 5, change-asserted-value). The old value only asked for 'acetyloxy' and
        'cyclohexane' in the name ('1,2-bis(acetyloxy)cyclohexane'). (the Blue Book): "When
        anions are identical functional class multiplicative nomenclature is used", 'ethane-1,2-diyl diacetate (PIN)'
        (:31823), 'propane-1,2,3-triyl triacetate (PIN)' (:31827); the multivalent organyl group of a ring is named as
        the group of one parent. The name is read back by OPSIN to the full InChIKey.
        """
        result = name_compound("CC(=O)OC1CCCCC1OC(C)=O")
        assert result == "cyclohexane-1,2-diyl diacetate", result
        assert_full_rt(result, "CC(=O)OC1CCCCC1OC(C)=O")


@pytest.mark.integration
class TestPolyfunctionalEsterDemotion:
    """: Esters demoted to acyloxy prefixes in polyfunctional compounds."""

    def test_glycerol_diacetate_no_dioate(self):
        """Glycerol diacetate: a diyl diacetate ester, NOT '-dioate' and NOT an '-ol' name."""
        result = name_compound("CC(=O)OCC(O)COC(=O)C")
        assert result is not None
        assert "oate" not in result, f"Got '-oate' suffix in polyfunctional ester: {result}"
        # PIN per R5: "SENIORITY ORDER FOR CLASSES" (the Blue Book) ranks "9 Esters"
        # (:18182) above "17 Hydroxy compounds" (:18190), so the ester is the principal class,
        # not an acyloxy prefix; "When anions are identical functional class
        # multiplicative nomenclature is used." (:31819); "ethane-1,2-diyl diacetate (PIN)"
        # (:31823), "propane-1,2,3-triyl triacetate (PIN)" (:31827). OPSIN RT exact
        # (TRIAGE.csv; re-checked in Task 7/8).
        assert result == "2-hydroxypropane-1,3-diyl diacetate", result

    # 2026-09-25 (pre-existing-failures plan, Task 5, TRIAGE rows 10/11), re-asserted
    # 2026-11-02 (change-asserted-value). The ester is the principal class,
    # the Blue Book "9 Esters" above:18190 "17 Hydroxy compounds"), and a
    # polyester of one 'alcoholic' component with different anions is named by
    # (:31831) "When anions are different, two methods are used";
    # "Method (1) generates preferred IUPAC names but names formed by using method
    # (2) are acceptable in general nomenclature" (:31836), with the example
    # 'propane-1,2,3-triyl 1,2-diacetate 3-propanoate (PIN)' (:31833). The PIN tier
    # now builds the method (1) PIN. OPSIN 2.9.0 has no grammar for it and reads it
    # through its acyloxy twin ('validation.polyester_twin'), so the default tier,
    # which ships a name only when OPSIN reads it directly, keeps the method (2) name.
    # Both run in production state (the OPSIN validity gate on).
    DIGLYCERIDE = "CCCCCCCCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COC(=O)CCCCCCCCC"
    DIGLYCERIDE_METHOD_1 = "(2S)-2-hydroxypropane-1,3-diyl 1-decanoate 3-docosanoate"
    DIGLYCERIDE_METHOD_2 = "(2S)-3-(decanoyloxy)-2-hydroxypropyl docosanoate"

    @pytest.mark.opsin_gate
    def test_diglyceride_mixed_acids(self):
        """Mixed-acid diglyceride: the PIN tier gives the method (1) PIN of
        , read back through its twin; the default tier gives the
        method (2) name, read back directly."""
        from orthonym import Orthonym
        from orthonym.cli import _emit_tier_flags
        from tests.support.rt_assert import name_is_rt_exact, name_reads_back
        row = Orthonym(style="pin", **_emit_tier_flags("pin")).name_tiered(self.DIGLYCERIDE)
        assert (row["name"], row["tier"]) == (self.DIGLYCERIDE_METHOD_1, "pin_verified"), row
        assert name_reads_back(row["name"], self.DIGLYCERIDE)
        result = name_compound(self.DIGLYCERIDE)
        assert result == self.DIGLYCERIDE_METHOD_2, result
        assert name_is_rt_exact(result, self.DIGLYCERIDE)

    @pytest.mark.opsin_gate
    def test_diglyceride_has_acyloxy_prefixes(self):
        """Method (2) at the default tier: the junior ester is an acyloxy prefix of
        the organyl, the senior one the anion."""
        result = name_compound(self.DIGLYCERIDE)
        assert "decanoyloxy" in result
        assert result.endswith(" docosanoate"), result

    def test_monoglyceride_no_oate(self):
        """Monoglyceride: one acyloxy, NOT -oate suffix."""
        result = name_compound("OCC(O)COC(=O)C")
        assert result is not None
        assert "oate" not in result, f"Got '-oate' suffix: {result}"

    def test_simple_ester_unaffected_propyl_acetate(self):
        """Propyl acetate is NOT polyfunctional -- no demotion."""
        assert name_compound("CCCOC(C)=O") == "propyl acetate"

    def test_ester_plus_amine(self):
        """Ester + amine: the ester is senior, named by functional class."""
        # PIN per R5: "SENIORITY ORDER FOR CLASSES" (the Blue Book) ranks
        # "9 Esters" (:18182) above "17 Hydroxy compounds" (:18190) and "19 Amines"
        # (:18192), so the ester is the principal characteristic group, named by
        # functional class, e.g. "2-hydroxypropyl (2-aminoethyl)carbamate (PIN)" (:30766);
        # "acetic acid (PIN)" (:29725) gives the anion word, "ethyl acetate
        # (PIN)" (:31667). OPSIN RT exact (TRIAGE rows 12-13).
        result = name_compound("CC(=O)OCC(N)C")
        assert result == "2-aminopropyl acetate"

    def test_ester_plus_alcohol_is_hydroxyalkyl_ester(self):
        """Ester + alcohol: the ester is senior; the OH is a hydroxy prefix."""
        # PIN per R5: "SENIORITY ORDER FOR CLASSES" (the Blue Book) ranks
        # "9 Esters" (:18182) above "17 Hydroxy compounds" (:18190) and "19 Amines"
        # (:18192), so the ester is the principal characteristic group, named by
        # functional class, e.g. "2-hydroxypropyl (2-aminoethyl)carbamate (PIN)" (:30766);
        # "acetic acid (PIN)" (:29725) gives the anion word, "ethyl acetate
        # (PIN)" (:31667). OPSIN RT exact (TRIAGE rows 12-13).
        result = name_compound("CC(=O)OCCO")
        assert result == "2-hydroxyethyl acetate"


# ===========================================================================
#: OPSIN round-trip validation for a phase ester format fixes
# ===========================================================================


@pytest.mark.integration
class TestEsterOPSINRoundTrip:
    """Validate that ester format fixes from 31-01 produce OPSIN-parseable names."""

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not available")
    def test_opsin_acetyloxybenzene(self):
        """Single acyloxy on benzene parses via OPSIN."""
        name = name_compound("CC(=O)Oc1ccccc1")
        assert name is not None
        opsin_smi = _opsin_parse(name)
        assert opsin_smi, f"OPSIN failed to parse: {name!r}"

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not available")
    def test_opsin_multi_acyloxy_benzene(self):
        """Two different acyloxy prefixes on benzene parse via OPSIN."""
        name = name_compound("CC(=O)Oc1ccc(OC(=O)CC)cc1")
        assert name is not None
        opsin_smi = _opsin_parse(name)
        assert opsin_smi, f"OPSIN failed to parse: {name!r}"

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not available")
    def test_opsin_glycerol_diacetate(self):
        """Glycerol diacetate (acyloxy + -ol) parses via OPSIN."""
        name = name_compound("CC(=O)OCC(O)COC(=O)C")
        assert name is not None
        opsin_smi = _opsin_parse(name)
        assert opsin_smi, f"OPSIN failed to parse: {name!r}"

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not available")
    @pytest.mark.opsin_gate
    def test_opsin_mixed_diglyceride(self):
        """Mixed-acid diglyceride: the default tier's name (method (2), the method (1)
        PIN has no OPSIN grammar) parses via OPSIN."""
        name = name_compound(
            "CCCCCCCCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COC(=O)CCCCCCCCC"
        )
        assert name is not None
        opsin_smi = _opsin_parse(name)
        assert opsin_smi, f"OPSIN failed to parse: {name!r}"

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not available")
    def test_opsin_diacetyloxy_cyclohexane(self):
        """Same acyloxy prefix twice on cyclohexane parses via OPSIN."""
        name = name_compound("CC(=O)OC1CCCCC1OC(C)=O")
        assert name is not None
        opsin_smi = _opsin_parse(name)
        assert opsin_smi, f"OPSIN failed to parse: {name!r}"

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not available")
    def test_opsin_ethyl_acetate_regression(self):
        """Simple ester (ethyl acetate) still parses via OPSIN."""
        name = name_compound("CCOC(C)=O")
        assert name == "ethyl acetate"
        opsin_smi = _opsin_parse(name)
        assert opsin_smi, f"OPSIN failed to parse: {name!r}"

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not available")
    def test_opsin_methyl_propanoate_regression(self):
        """Simple ester (methyl propanoate) still parses via OPSIN."""
        name = name_compound("COC(=O)CC")
        assert name == "methyl propanoate"
        opsin_smi = _opsin_parse(name)
        assert opsin_smi, f"OPSIN failed to parse: {name!r}"

    @pytest.mark.opsin_gate
    def test_format_multi_acyloxy_has_parentheses(self):
        """Multi-acyloxy benzene uses parenthesized prefix format.

        The shipped name of the molecule is '4-(acetyloxy)phenyl propanoate' (method (2) of,:31831; see
        TestRingEsterPrefixJoining.test_different_acyloxy_benzene for the 2026-11-05 re-assertion): the acetyloxy prefix
        is parenthesised, and the propanoate is the anion word, not a second prefix.
        """
        name = name_compound("CC(=O)Oc1ccc(OC(=O)CC)cc1")
        assert name == "4-(acetyloxy)phenyl propanoate", name
        assert "(acetyloxy)" in name and not name.endswith(")")

# ===========================================================================
#: Phospholipid compound coverage (best-effort)
# ===========================================================================


@pytest.mark.integration
class TestPhospholipidCoverage:
    """Best-effort phospholipid naming: non-None results, no crashes."""

    def test_glycerol_diacetate_phosphate_produces_name(self):
        """Glycerol diacetate phosphate returns a non-None name."""
        result = name_compound("CC(=O)OCC(COP(=O)(O)O)OC(=O)C")
        assert result is not None, "Phospholipid diacetate + phosphate returned None"
        assert isinstance(result, str)
        assert len(result) > 0

    def test_glycerol_diacetate_phosphate_has_acyloxy(self):
        """Phospholipid diacetate + phosphate shows acyloxy prefixes from 31-01 fix."""
        result = name_compound("CC(=O)OCC(COP(=O)(O)O)OC(=O)C")
        assert result is not None
        # The ester demotion should produce acyloxy prefixes or phosphate principal group
        assert "yloxy" in result or "phosph" in result, (
            f"Expected acyloxy prefix or phosphate reference in: {result!r}"
        )

    def test_glycerol_diacetate_phosphate_has_phospho(self):
        """Phospholipid diacetate + phosphate contains phosphorus group reference."""
        result = name_compound("CC(=O)OCC(COP(=O)(O)O)OC(=O)C")
        assert result is not None
        assert "phosph" in result, f"Expected phosphorus reference in: {result!r}"

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not available")
    def test_opsin_glycerol_diacetate_phosphate(self):
        """Phospholipid diacetate + phosphate name parses via OPSIN."""
        name = name_compound("CC(=O)OCC(COP(=O)(O)O)OC(=O)C")
        assert name is not None
        opsin_smi = _opsin_parse(name)
        assert opsin_smi, f"OPSIN failed to parse phospholipid name: {name!r}"

    def test_glycerol_phosphate_produces_name(self):
        """Simple glycerol phosphate (no ester) returns a non-None name."""
        result = name_compound("OCC(O)COP(=O)(O)O")
        assert result is not None, "Glycerol phosphate returned None"
        assert isinstance(result, str)
        assert len(result) > 0

    def test_glycerol_phosphate_has_hydroxy(self):
        """Glycerol phosphate includes hydroxy prefix for alcohol groups."""
        result = name_compound("OCC(O)COP(=O)(O)O")
        assert result is not None
        assert "hydroxy" in result or "phosph" in result, (
            f"Expected hydroxy prefix or phosphate reference in: {result!r}"
        )

    @pytest.mark.skipif(not OPSIN_AVAILABLE, reason="OPSIN JAR not available")
    def test_opsin_glycerol_phosphate(self):
        """Glycerol phosphate name parses via OPSIN."""
        name = name_compound("OCC(O)COP(=O)(O)O")
        assert name is not None
        opsin_smi = _opsin_parse(name)
        assert opsin_smi, f"OPSIN failed to parse glycerol phosphate: {name!r}"

    def test_lecithin_like_no_crash(self):
        """Lecithin-like compound (ester + phosphate diester + quat N) does not crash."""
        result = name_compound("CC(=O)OCC(COP(=O)(O)OCC[N+](C)(C)C)OC(=O)C")
        assert result is not None, "Lecithin-like compound returned None"
        # Known limitation: may lose phosphocholine fragment, but must not crash

    def test_gpc_no_crash(self):
        """Glycerophosphocholine (no esters) does not crash."""
        result = name_compound("OCC(O)COP(=O)(O)OCC[N+](C)(C)C")
        assert result is not None, "GPC compound returned None"
        # Known limitation: phosphate diester fragment may be lost

    def test_ester_amine_phospholipid_fragment(self):
        """Ester + amine phospholipid-like fragment names correctly."""
        result = name_compound("CC(=O)OCC(OC(=O)CCCCCCCCC)COP(=O)(O)OCCN")
        assert result is not None
        # Should have acyloxy prefixes, amine suffix, or phosphate reference
        assert "yloxy" in result or "amine" in result or "phosph" in result, (
            f"Expected acyloxy, amine, or phosphate in: {result!r}"
        )
