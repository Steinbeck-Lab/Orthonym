import pytest
from rdkit import Chem
from orthonym.namer import name_compound


@pytest.mark.unit
class TestWave2P5FusedVerify:
    """C5 fused rows already correct at HEAD — lock them against regression."""

    @pytest.mark.parametrize("smiles,expected", [
        ("c1cc2ccc3cccc4ccc(c1)c2c34", "pyrene"),          # P-25.3.1.3
        ("c1ccc2ccccc2c1", "naphthalene"),                 # P-25.3.2.4 (g)
        ("c1ccc2ncccc2c1", "quinoline"),                   # P-25.3.2.4 (h)
        ("c1ccc2[nH]ccc2c1", "1H-indole"),                 # P-25.3.2.4 (i)
        ("c1ccc2nc[nH]c2c1", "1H-benzimidazole"),          # P-25.3.2.4 (j)
    ])
    def test_already_correct(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("s1,s2", [
        # two SMILES spellings of naphthalene -> identical output (determinism)
        ("c1ccc2ccccc2c1", "c1ccc2c(c1)cccc2"),
        # two spellings of quinoline
        ("c1ccc2ncccc2c1", "c1ccc2c(c1)nccc2"),
    ])
    def test_parent_selection_ab_order_invariant(self, s1, s2):
        # P-25.3.2.4: parent/orientation selection must not depend on the
        # RDKit atom order induced by the input SMILES spelling.
        assert name_compound(s1) == name_compound(s2)


@pytest.mark.unit
class TestP52BenzoGhiPerylene:
    def test_benzo_ghi_perylene(self):
        # P-52.2.4.2 / large peri-fused PAH catalog parent.
        # OPSIN-RT-verified: benzo[ghi]perylene -> c1cc2ccc3ccc4ccc5cccc6c(c1)c2c3c4c56
        assert name_compound("c1cc2ccc3ccc4ccc5cccc6c(c1)c2c3c4c56") == "benzo[ghi]perylene"


@pytest.mark.unit
class TestP25MultiparentDifuranC:
    def test_benzo_difuran_c_prime(self):
        # P-25.3.5.3: multiparent name preferred to a fused-ring name.
        # OPSIN-RT-verified: benzo[1,2-b:4,5-c']difuran -> c1cc2cc3cocc3cc2o1
        assert name_compound("c1cc2cc3cocc3cc2o1") == "benzo[1,2-b:4,5-c']difuran"

    def test_benzo_difuran_c_prime_ab_order(self):
        # determinism: alternate spelling -> identical output
        assert name_compound("o1cc2cc3ccoc3cc2c1") == name_compound("c1cc2cc3cocc3cc2o1")


@pytest.mark.unit
class TestP25MultiparentDifuranB:
    def test_benzo_difuran_b_prime(self):
        # P-25.3.7.1: multiparent, one interparent (benzene) component; primed letters, colon-separated.
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
        # P-25.3.7.3: three+ interparent components, double-primed benzo interparent.
        # OPSIN-RT-verified BB-verbatim PIN.
        expected = "benzo[1'',2'':3,4;4'',5'':3',4']dicyclobuta[1,2-b:1',2'-c']difuran"
        assert name_compound("O1C2=C(C=C1)C=1C2=CC2=C(C3=COC=C32)C1") == expected


@pytest.mark.unit
class TestP25ParentSelectionTiebreakGtoJ:
    """P-25.3.2.4 (g)-(j) parent-selection tiebreaks. No OPSIN-parseable
    example isolates a (g)-(j) decision, so this locks: (1) the (a)-(f)-decided
    examples are unchanged after the stubs become computed; (2) the computed
    (g)-(j) fields are spelling-invariant (determinism-sensitive scorer)."""

    @pytest.mark.parametrize("smiles,expected", [
        ("c1ccc2ccccc2c1", "naphthalene"),
        ("c1ccc2ncccc2c1", "quinoline"),
        ("c1ccc2[nH]ccc2c1", "1H-indole"),
        ("c1ccc2nc[nH]c2c1", "1H-benzimidazole"),
    ])
    def test_af_decided_unchanged(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("s1,s2", [
        ("c1ccc2ncccc2c1", "c1ccc2c(c1)nccc2"),      # quinoline, two spellings (same structure)
        ("c1ccc2nc[nH]c2c1", "c1ccc2[nH]cnc2c1"),    # benzimidazole, two spellings
    ])
    def test_rank_is_spelling_invariant(self, s1, s2):
        # The (g)-(j) fields must be per-component structural descriptors,
        # not RDKit-atom-order artifacts -> identical name for both spellings.
        assert name_compound(s1) == name_compound(s2)

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
    """P-25.3.3.2/.2.1/.2.2/.2.3/.3.3.2 — interior-atom numbering needs
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
    `_disable_opsin_validity_gate_for_tests` turns the production SELF-01 OPSIN
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
        out = name_compound(smiles)
        # a fused interior superscript locant looks like <digit>a<digit> or a
        # '<digit>a<digit>H-' indicated-H prefix (NOT the von-Baeyer '^n,m').
        import re
        assert not re.search(r"\d+a\d+", out or ""), (
            f"emitted an interior-superscript fusion name: {out!r}"
        )

    @pytest.mark.parametrize("smiles", _INTERIOR_SMILES)
    def test_production_gate_declines(self, smiles, monkeypatch):
        # Re-enable the production SELF-01 validity gate (the conftest autouse
        # fixture disables it) and assert the namer fails closed to the sentinel
        # for these no-verifiable-oracle interior-atom systems.
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            pytest.skip("probe SMILES invalid in RDKit; substitute a valid interior-heteroatom peri-fused system")
        import orthonym.namer as _namer
        if not _namer._validity_gate_jar_present():
            pytest.skip("OPSIN jar not present; production gate cannot run")
        monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
        out = name_compound(smiles)
        assert out in (None, "unknown organic compound"), (
            f"production gate should decline interior-atom system, got {out!r}"
        )
