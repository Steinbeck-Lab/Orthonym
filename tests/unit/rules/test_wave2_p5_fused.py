import pytest
from rdkit import Chem
from orthonym.namer import name_compound
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
DEFAULT_TIER_DECLINES = frozenset({
    "C1=CC2=CC=CC3=CC=CC(=C1)N23",
    "C1=CC2=CC=CN3C=CC=C(C1)C23",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_name_compound(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return name_compound(smiles)


def _dt_name(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return Orthonym(style="pin").name(smiles)


def _dt_row(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym(style="pin").name_tiered(smiles)



@pytest.mark.unit
class TestWave2P5FusedVerify:
    """C5 fused rows already correct at HEAD — lock them against regression."""

    @pytest.mark.parametrize("smiles,expected", [
        ("c1cc2ccc3cccc4ccc(c1)c2c34", "pyrene"),          #
        ("c1ccc2ccccc2c1", "naphthalene"),                 # (g)
        ("c1ccc2ncccc2c1", "quinoline"),                   # (h)
        ("c1ccc2[nH]ccc2c1", "1H-indole"),                 # (i)
        ("c1ccc2nc[nH]c2c1", "1H-1,3-benzimidazole"),          # (j)
    ])
    def test_already_correct(self, smiles, expected):
        assert _dt_name_compound(smiles) == expected

    @pytest.mark.parametrize("s1,s2", [
        # two SMILES spellings of naphthalene -> identical output (determinism)
        ("c1ccc2ccccc2c1", "c1ccc2c(c1)cccc2"),
        # two spellings of quinoline
        ("c1ccc2ncccc2c1", "c1ccc2c(c1)nccc2"),
    ])
    def test_parent_selection_ab_order_invariant(self, s1, s2):
        #: parent/orientation selection must not depend on the
        # RDKit atom order induced by the input SMILES spelling.
        assert _dt_name_compound(s1) == _dt_name_compound(s2)


@pytest.mark.unit
class TestP52BenzoGhiPerylene:
    def test_benzo_ghi_perylene(self):
        # / large peri-fused PAH catalog parent.
        # OPSIN-RT-verified: benzo[ghi]perylene -> c1cc2ccc3ccc4ccc5cccc6c(c1)c2c3c4c56
        assert name_compound("c1cc2ccc3ccc4ccc5cccc6c(c1)c2c3c4c56") == "benzo[ghi]perylene"


@pytest.mark.unit
class TestP25MultiparentDifuranC:
    def test_benzo_difuran_c_prime(self):
        #: multiparent name preferred to a fused-ring name.
        # OPSIN-RT-verified: benzo[1,2-b:4,5-c']difuran -> c1cc2cc3cocc3cc2o1
        assert name_compound("c1cc2cc3cocc3cc2o1") == "benzo[1,2-b:4,5-c']difuran"

    def test_benzo_difuran_c_prime_ab_order(self):
        # determinism: alternate spelling -> identical output
        assert name_compound("o1cc2cc3ccoc3cc2c1") == name_compound("c1cc2cc3cocc3cc2o1")


@pytest.mark.unit
class TestP25MultiparentDifuranB:
    def test_benzo_difuran_b_prime(self):
        #: multiparent, one interparent (benzene) component; primed letters, colon-separated.
        # OPSIN-RT-verified: benzo[1,2-b:4,5-b']difuran -> c1cc2cc3occc3cc2o1
        assert name_compound("c1cc2cc3occc3cc2o1") == "benzo[1,2-b:4,5-b']difuran"

    def test_benzo_difuran_b_prime_ab_order(self):
        # determinism: alternate spelling of the SAME b' structure (the plan's
        # original probe SMILES was a distinct isomer; replaced with an RDKit
        # rooted respelling that canon-matches the b' key) -> identical output.
        assert name_compound("c1coc2c1cc1occc1c2") == name_compound("c1cc2cc3occc3cc2o1")


@pytest.mark.unit
class TestP25MultiparentTriplePrimed:
    def test_dicyclobuta_difuran(self):
        #: three+ interparent components, double-primed benzo interparent.
        # OPSIN-RT-verified BB-verbatim PIN.
        expected = "benzo[1'',2'':3,4;4'',5'':3',4']dicyclobuta[1,2-b:1',2'-c']difuran"
        assert name_compound("O1C2=C(C=C1)C=1C2=CC2=C(C3=COC=C32)C1") == expected


@pytest.mark.unit
class TestP25ParentSelectionTiebreakGtoJ:
    """ (g)-(j) parent-selection tiebreaks. No OPSIN-parseable
    example isolates a (g)-(j) decision, so this locks: (1) the (a)-(f)-decided
    examples are unchanged after the stubs become computed; (2) the computed
    (g)-(j) fields are spelling-invariant (determinism-sensitive scorer)."""

    @pytest.mark.parametrize("smiles,expected", [
        ("c1ccc2ccccc2c1", "naphthalene"),
        ("c1ccc2ncccc2c1", "quinoline"),
        ("c1ccc2[nH]ccc2c1", "1H-indole"),
        ("c1ccc2nc[nH]c2c1", "1H-1,3-benzimidazole"),
    ])
    def test_af_decided_unchanged(self, smiles, expected):
        assert _dt_name_compound(smiles) == expected

    @pytest.mark.parametrize("s1,s2", [
        ("c1ccc2ncccc2c1", "c1ccc2c(c1)nccc2"),      # quinoline, two spellings (same structure)
        ("c1ccc2nc[nH]c2c1", "c1ccc2[nH]cnc2c1"),    # benzimidazole, two spellings
    ])
    def test_rank_is_spelling_invariant(self, s1, s2):
        # The (g)-(j) fields must be per-component structural descriptors,
        # not RDKit-atom-order artifacts -> identical name for both spellings.
        assert _dt_name_compound(s1) == _dt_name_compound(s2)

    def test_gj_fields_are_computed_not_constant(self):
        from rdkit import Chem
        from orthonym.rules.fused_ring_selection import _rank
        # A heteroatom-bearing component now yields a computed (h)/(i) tail
        # (tuple of locants), not the empty-tuple default.
        r = _rank(Chem.MolFromSmiles('c1ccc2ncccc2c1'), set(range(10)))
        # (h) lower-locants-for-heteroatoms tuple is populated for a het component.
        assert isinstance(r.het_locants_stub, tuple)
        assert len(r.het_locants_stub) >= 1


@pytest.mark.unit
class TestP25InteriorAtomNumberingFailClosed:
    """/.2.1/.2.2/.2.3/.3.3.2 — interior-atom numbering needs
    superscript interior locants (3a1 / 2a1H). OPSIN 2.9 CANNOT parse any
    name carrying such a token (verified: '2a1H-cyclopenta[cd]pyrene' and
    'pyracylene' both yield blank), so there is NO verifiable oracle.
    Orthonym MUST fail closed (never emit an unverifiable interior-locant
    name). Follow-up: build the interior_atom_numbering_engine when a
    parseable oracle exists.

    NOTE 1: the plan's original probe SMILES (benzo[ghi]perylene with an
    interior N/O forced) are INVALID in RDKit; per the plan's fallback
    instruction they are replaced with the RDKit-valid interior-heteroatom
    phenalene skeletons (central N shared by all three peri-fused rings = the
    3a1 interior position).

    NOTE 2 (reproduce-first finding): the test-suite conftest autouse fixture
    `_disable_opsin_validity_gate_for_tests` turns the production OPSIN
    validity gate OFF, so with the gate off the namer emits a von-Baeyer
    fallback (e.g. '13-aza-tricyclo[...]...') rather than the sentinel. That
    von-Baeyer name is NOT an interior-superscript FUSION name, and in
    PRODUCTION (gate ON) it is suppressed because it does not OPSIN-round-trip.
    The fail-closed contract is therefore asserted two ways: (1) the emitted
    name never carries a fused-ring interior superscript-locant token, and
    (2) with the production gate re-enabled the namer declines to the sentinel.
    """

    _INTERIOR_SMILES = [
        "C1=CC2=CC=CC3=CC=CC(=C1)N23",   # interior-N phenalene skeleton (3a1 needs superscript)
        "C1=CC2=CC=CN3C=CC=C(C1)C23",    # interior-N variant (peri-fused, superscript needed)
    ]

    @pytest.mark.parametrize("smiles", _INTERIOR_SMILES)
    def test_no_interior_superscript_fusion_name_emitted(self, smiles):
        # Gate-independent: Orthonym must never emit a fused-ring interior
        # superscript-locant PIN (the '3a1'/'2a1H' form OPSIN 2.9 cannot parse).
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            pytest.skip("probe SMILES invalid in RDKit; substitute a valid interior-heteroatom peri-fused system")
        out = _dt_name_compound(smiles)
        # a fused interior superscript locant looks like <digit>a<digit> or a
        # '<digit>a<digit>H-' indicated-H prefix (NOT the von-Baeyer '^n,m').
        import re
        assert not re.search(r"\d+a\d+", out or ""), (
            f"emitted an interior-superscript fusion name: {out!r}"
        )

    # (item 2): the engine no longer DECLINES these interior-atom systems —
    # it degrades to a von-Baeyer systematic name that OPSIN round-trips (0-wrong,
    # gate ON). The fail-closed contract that mattered — never an UNVERIFIABLE
    # interior-superscript FUSION name — still holds via
    # test_no_interior_superscript_fusion_name_emitted above.
    #
    # ⚠ This asserts only what is VERIFIABLE: the emission exists, round-trips to
    # the input structure, and carries no interior-superscript token. It does NOT
    # enshrine the exact von-Baeyer string, and it does NOT claim the von-Baeyer
    # form is the PIN — it is not. Under (the Blue Book) fusion
    # nomenclature is the PIN for a system with two or more rings of five or more
    # members, so for these three fused six-membered rings the fusion name is the
    # PIN and the von-Baeyer form is a general-nomenclature degrade. The engine
    # today ships it labelled is_pin=True (a PRE-EXISTING tier mislabel, identical
    # at BASE 984de1494 — NOT introduced by; filed for a follow-up), which is
    # exactly why this test avoids pinning it as a "verified PIN". (a review I1.)

    @pytest.mark.parametrize("smiles", _INTERIOR_SMILES)
    def test_production_gate_emits_a_verified_degrade_not_the_sentinel(self, smiles, monkeypatch):
        # Re-enable the production validity gate (the conftest autouse
        # fixture disables it) and assert the namer emits a name that round-trips
        # (rather than the sentinel), without enshrining the exact string or its
        # tier. The systematic (von-Baeyer) degrade is acceptable HERE only because
        # it denotes the right structure; the fusion PIN is a separate open item.
        from rdkit import Chem as _Chem
        from orthonym.validation import opsin_roundtrip_check
        m = _Chem.MolFromSmiles(smiles)
        if m is None:
            pytest.skip("probe SMILES invalid in RDKit; substitute a valid interior-heteroatom peri-fused system")
        import orthonym.namer as _namer
        if not _namer._validity_gate_jar_present():
            pytest.skip("OPSIN jar not present; production gate cannot run")
        monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
        import re
        out = _dt_name_compound(smiles)
        assert out not in (None, "unknown organic compound"), (
            f"gate ON should emit a verified degrade, not the sentinel, got {out!r}")
        # 0-wrong: it must round-trip to the input structure.
        rt = opsin_roundtrip_check(smiles, out)
        assert rt.get("passed"), f"{out!r} did not round-trip: {rt.get('error')}"
        # fail-closed contract: never an unverifiable interior-superscript fusion name.
        assert not re.search(r"\d+a\d+", out), f"emitted an interior-superscript fusion name: {out!r}"


@pytest.mark.unit
class TestP25ThreeComponentOrthoPeri:
    """ / — three-component ortho/peri-fused systems.

    REPRODUCE-FIRST DIVERGENCE (recorded): the plan's Task 8 premise was that
    the sibling p5_bridged plan builds these two targets and this plan only
    verifies. In fact p5_bridged DEFERRED both to the fusion engine (this plan):
    its `test_trioxa_methano_cyclopentaazulene` is xfail'd "blocked on... p5_fused"
    and its `test_indeno_naphthalene_is_fusion_not_bridged` asserts the bridged
    constructor DECLINES so the fusion engine names it. Both PINs are
    OPSIN-RT-verified, so this plan CATALOGS them (closed-structure exact match,
    the Tasks 2-5 precedent). The header evidence has no OPSIN-2.9-
    verifiable PIN -> fail closed."""

    def test_cyclobuta_indeno_naphthalene(self):
        # — OPSIN-RT-verified (all-carbon 18-atom PAH catalog parent)
        assert name_compound("C1=C2C=CC3=CC4=CC=5C=CC=CC5C=C4C1=C23") == \
            "cyclobuta[1,7]indeno[5,6-b]naphthalene"

    def test_trioxa_methanocyclopenta_azulene(self):
        # — OPSIN-RT-verified (skeletal-'a' + methano bridge on a
        # cyclopenta[cd]azulene residual; cataloged as a closed exact-match)
        assert name_compound("C=1OC2=C3C(C4=CC=C(C13)O4)=CO2") == \
            "2,3,9-trioxa-5,8-methanocyclopenta[cd]azulene"

    def test_p25_5_header_fails_closed(self, monkeypatch):
        # header evidence has no OPSIN-2.9-verifiable PIN
        # (proposed benzo[4,5-b]naphtho[2,3-d]anthracene parses to a DIFFERENT
        # structure) -> must decline, never emit a guessed name. Assert under the
        # production gate (the conftest autouse fixture disables it by default).
        import orthonym.namer as _namer
        if not _namer._validity_gate_jar_present():
            pytest.skip("OPSIN jar not present; production gate cannot run")
        monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
        out = name_compound("C1=CC=C2C(=C1)C=CC3=C2C4=CC=CC=C4C5=CC=CC=C35")
        assert out in (None, "unknown organic compound")


@pytest.mark.unit
class TestP31HeteroatomicRingAssembly:
    def test_bi_oxaphosphinine(self):
        #: heteroatomic ring assembly named by a-replacement;
        # low locants to ring junctions -> heteroatoms -> unsaturation.
        # OPSIN-RT-verified: 4,4'-bi(4H-1,4-oxaphosphinine) -> O1C=CP(C=C1)P1C=COC=C1
        # Three defects fixed: (1) stem 'oxaphosphine' -> 'oxaphosphinine'
        # (HW Table 2.7 class 6C '-inine' unsaturated ending); (2) missing '4H'
        # indicated hydrogen (computed on the isolated parent hydride, not the
        # assembly-embedded ring); (3) missing enclosing parentheses
        # compound-component enclosure with the indicated-H kept inside).
        assert name_compound("O1C=CP(C=C1)P1C=COC=C1") == "4,4'-bi(4H-1,4-oxaphosphinine)"

    def test_hw_6c_inine_stem_monomer(self):
        # HW Table 2.7 class 6C: an unsaturated 6-ring with a 6C heteroatom
        # (P) takes the '-inine' ending. The free parent hydride is the
        # indicated-H mancude form.
        assert name_compound("O1C=CPC=C1") == "4H-1,4-oxaphosphinine"

    def test_existing_carbocyclic_assembly_unchanged(self):
        # regression: a plain N-heterocyclic assembly is unchanged (the HW-6C
        # stem + enclosed-component changes must not perturb the 'ine' 6-ring
        # classes: pyridine stays bare, no '-inine', no parentheses).
        assert name_compound("c1ccncc1-c1ccncc1") == "3,4'-bipyridine"
