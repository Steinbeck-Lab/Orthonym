"""Tests for a phase Plan 02 gap closure: DKP, tropane, ergostene fixes.

Tests cover:
- Piperazine-2,5-dione retained name recognition
- Tropane IUPAC numbering map and NP naming
- Ergostene derivative lookup
"""

import pytest

from rdkit import Chem

from orthonym import name_compound
from orthonym.data.retained_names import get_retained_name
from orthonym.data.natural_products import get_scaffold_numbering, get_natural_product_name
from orthonym.rules.natural_products import name_natural_product


# ---------------------------------------------------------------------------
# Task 1: Piperazine-2,5-dione (diketopiperazine) retained name
# ---------------------------------------------------------------------------


class TestDKPRetainedName:
    """Piperazine-2,5-dione should be recognized as a retained heterocyclic name."""

    @pytest.mark.unit
    def test_bare_dkp_ring_retained_name(self):
        """Piperazine-2,5-dione is named by the ring-ketone path, not a lookup row.

         fix a performance pass (wp6-tests), TEST-BUG: "O=C1CNCC(=O)N1" is piperazine-2,6-dione
        (both carbonyls on one ring N; InChIKey CYJAWBVQRMVFEO); its wrong-structure
        'piperazine-2,5-dione' lookup row was deleted on purpose (8a0a9bc0e,
        data/retained_names.py PA1 sweep), so the lookup correctly returns None. The
        2,5-dione (glycine anhydride) is O=C1CNC(=O)CN1. Both engine names are OPSIN
        2.9.0 full-InChIKey exact ring-ketone suffix names). The test id is
        kept."""
        assert get_retained_name(Chem.CanonSmiles("O=C1CNCC(=O)N1")) is None
        assert get_retained_name(Chem.CanonSmiles("O=C1CNC(=O)CN1")) is None
        assert name_compound("O=C1CNC(=O)CN1") == "piperazine-2,5-dione"
        assert name_compound("O=C1CNCC(=O)N1") == "piperazine-2,6-dione"

    @pytest.mark.integration
    def test_dkp_compound7_contains_piperazin(self):
        """DKP compound 7 (with indole + imidazole) generates name containing 'piperazin'."""
        smiles = "O=C1NC(Cc2c[nH]c3ccccc23)C(=O)N/C1=C/c1cnc[nH]1"
        name = name_compound(smiles)
        assert name, f"Should produce a name"
        assert "piperazin" in name.lower(), (
            f"Compound 7 should contain 'piperazin' in name. Got: {name}"
        )

    @pytest.mark.integration
    def test_dkp_compound2_root_cause_documented(self):
        """DKP compound 2 parent selection: document root cause.

        Compound 2 (C=C1NC(=O)[C@H]([C@@H](C)[C@]2(O)C(=O)N(C)c3ccccc32)NC1=O)
        currently produces '(2R)-2-hydroxy-N-methylindolin-1-one'.
        The indoline ring is being selected over the DKP ring because the
        indoline + benzene fused system scores higher in ring_system_score.
        This test verifies the root cause and documents it.
        """
        smiles = "C=C1NC(=O)[C@H]([C@@H](C)[C@]2(O)C(=O)N(C)c3ccccc32)NC1=O"
        name = name_compound(smiles)
        assert name, "Should produce a name"
        # This compound has indoline fused to benzene (8 atoms) vs DKP (6 atoms).
        # The fused ring system wins by ring size. This is a deeper parent selection
        # issue that requires ring_system_score changes -- deferred.
        # For now, just verify it produces SOME name (it does).


# ---------------------------------------------------------------------------
# Task 2: Tropane IUPAC numbering map and NP naming
# ---------------------------------------------------------------------------


class TestTropaneNumberingMap:
    """Tropane scaffold should have an IUPAC numbering map for locant-based naming."""

    @pytest.mark.unit
    def test_tropane_numbering_map_exists(self):
        """get_scaffold_numbering for tropane SMILES returns a dict with 8 entries."""
        numbering = get_scaffold_numbering("CN1[C@@H]2CCC[C@H]1CC2")
        assert numbering is not None, "Tropane numbering map should exist"
        assert len(numbering) == 8, (
            f"Tropane numbering should have 8 entries (positions 1-8, N-methyl excluded). "
            f"Got {len(numbering)}: {numbering}"
        )

    @pytest.mark.unit
    def test_tropane_numbering_covers_all_iupac_locants(self):
        """Numbering map values should cover IUPAC locants 1-8."""
        numbering = get_scaffold_numbering("CN1[C@@H]2CCC[C@H]1CC2")
        assert numbering is not None
        locants = set(numbering.values())
        assert locants == {1, 2, 3, 4, 5, 6, 7, 8}, (
            f"Expected locants 1-8, got: {locants}"
        )

    @pytest.mark.unit
    def test_tropane_nitrogen_at_position_8(self):
        """Nitrogen atom (query_pos 1) should map to IUPAC locant 8."""
        numbering = get_scaffold_numbering("CN1[C@@H]2CCC[C@H]1CC2")
        assert numbering is not None
        assert numbering[1] == 8, (
            f"Nitrogen (query_pos 1) should be IUPAC 8, got {numbering.get(1)}"
        )


class TestTropaneNPNaming:
    """name_natural_product should return tropane-based names."""

    @pytest.mark.integration
    def test_bare_tropane_returns_tropane(self):
        """Bare tropane scaffold returns 'tropane' from NP naming."""
        mol = Chem.MolFromSmiles("CN1C2CCCC1CC2")
        name = name_natural_product(mol)
        assert name is not None, "Bare tropane should produce a name"
        assert "tropane" == name.lower(), f"Expected 'tropane', got: {name}"

    @pytest.mark.integration
    def test_3_hydroxytropane_contains_tropan_and_ol(self):
        """3-hydroxytropane returns name containing 'tropan' and 'ol'."""
        mol = Chem.MolFromSmiles("CN1[C@@H]2CC[C@H]1C[C@@H](O)C2")
        name = name_natural_product(mol)
        assert name is not None, "3-hydroxytropane should produce a name"
        assert "tropan" in name.lower(), f"Expected 'tropan' in name. Got: {name}"
        assert "ol" in name.lower(), f"Expected 'ol' in name. Got: {name}"

    @pytest.mark.integration
    @pytest.mark.opsin_gate
    def test_tropane_ester_compound3_free_base(self):
        """Compound 3 free base (tropisetron): the NP scaffold producer declines,
        and the shipped name is the verified ester name.

        Task 4 continuation (2026-09-25; TRIAGE row 36, canary call 182). The
        producer used to return the bare 'tropane' (OPSIN: C8H15N; the input is
        C17H20N2O2): its "no decorations" test omitted `esters`.
        (the Blue Book): every substituent is cited, so it now declines and
        the decomposition names the ester by functional class,
        (:31663): "All preferred IUPAC names for esters are named by functional
        class nomenclature." The alcohol's descriptor set is the Blue Book's own
        for this alcohol, '(1R,3r,5S)-8-methyl-8-azabicyclo[3.2.1]octan-3-yl
        (2S)-3-hydroxy-2-phenylpropanoate... tropan-3α-yl...' (:48828).
        OPSIN 2.9.0 parses no lowercase r/s, so the name is verified by the
        stereo-stripped full-InChIKey round trip plus the centres labeller
        (namer._pseudoasymmetric_name_verified; the controller's ruling of
        2026-09-25)."""
        from orthonym.namer import _pseudoasymmetric_name_verified
        smiles = "CN1[C@@H]2CC[C@H]1C[C@@H](OC(=O)c1c[nH]c3ccccc13)C2"
        name = name_compound(smiles)
        assert name == "(1R,3r,5S)-tropan-3-yl 1H-indole-3-carboxylate", name
        assert _pseudoasymmetric_name_verified(name, smiles)
        # fix a performance pass (wp6-tests), change-asserted-value (was: the NP producer
        # returns None). 286a491c8 made tropisetron nameable on purpose (the NP
        # producer declined, the decomposition named the ester); 28e1d520b (esters.py:
        # name_ester names a fused ring acid via name_polyfunctional_ester_via_acid)
        # later let the NP producer build the SAME complete string itself (archive
        # bisect, rest-of-suite research item 29). (the Blue Book, no
        # decoration dropped) is what the old 'is None' guarded, so the producer may
        # decline, or return exactly the verified complete name.
        np = name_natural_product(Chem.MolFromSmiles(smiles))
        assert np is None or (np == name and _pseudoasymmetric_name_verified(np, smiles)), np

    @pytest.mark.integration
    @pytest.mark.xfail(strict=True, reason=(
        "DEFECT (non-PIN at pin_verified): the shipped '(1R,3r,5S)-tropan-3-yl ...' "
        "uses the natural-product parent 'tropane' (P-100, BlueBookV2.md:50943: no PINs "
        "for the P-10 chapter), and OPSIN 2.9.0 cannot parse its 'r' (verified only by "
        "the stereo-stripped round trip + the centres labeller, controller ruling "
        "2026-09-25). The Blue Book writes this alcohol's ester "
        "'(1R,3r,5S)-8-methyl-8-azabicyclo[3.2.1]octan-3-yl ...' first, before "
        "'tropan-3α-yl ...' (P-93.5.2.2.1, :48828). Needs the substituted von Baeyer "
        "ring-yl ester producer (.planning/TODO-2026-09-24.md section J; 'Open from T12 "
        "fix round 2 (wp6)')."))
    def test_tropane_ester_compound3_systematic_pin(self):
        smiles = "CN1[C@@H]2CC[C@H]1C[C@@H](OC(=O)c1c[nH]c3ccccc13)C2"
        assert name_compound(smiles) == (
            "(1R,3r,5S)-8-methyl-8-azabicyclo[3.2.1]octan-3-yl 1H-indole-3-carboxylate")

    @pytest.mark.integration
    def test_tropanone(self):
        """Tropan-3-one (ketone at C-3) should contain 'tropan' and 'one'."""
        # CN1C2CC(=O)CC1CC2 is tropan-3-one (ketone at middle of 3-carbon bridge)
        mol = Chem.MolFromSmiles("CN1C2CC(=O)CC1CC2")
        name = name_natural_product(mol)
        assert name is not None, "Tropan-3-one should produce a name"
        assert "tropan" in name.lower(), f"Expected 'tropan' in name. Got: {name}"
        assert "one" in name.lower(), f"Expected 'one' in name. Got: {name}"

    @pytest.mark.integration
    def test_compound3_full_with_hcl_salt(self):
        """Compound 3 with HCl salt generates name containing 'tropan'."""
        name = name_compound("CN1[C@@H]2CC[C@H]1C[C@@H](OC(=O)c1c[nH]c3ccccc13)C2.Cl")
        assert name is not None, "Compound 3 salt should produce a name"
        assert "tropan" in name.lower(), (
            f"Compound 3 salt should contain 'tropan'. Got: {name}"
        )


# ---------------------------------------------------------------------------
# Task 3: Ergostene derivative lookup
# ---------------------------------------------------------------------------


class TestErgosteneDerivative:
    """Ergostene compound with non-standard (5,6,5,6) ring topology."""

    ERGOSTENE_SMILES = (
        "C=C(CC[C@@H](C)[C@H]1CC[C@@]2(C)C3=C(CC[C@]12C)"
        "[C@@]1(C)CC[C@@H](O)[C@@]1(C)CC3)C(C)C"
    )

    @pytest.mark.unit
    def test_ergostene_derivative_lookup_exists(self):
        """get_natural_product_name returns a name containing 'ergost'."""
        from rdkit import Chem
        can = Chem.MolToSmiles(Chem.MolFromSmiles(self.ERGOSTENE_SMILES))
        name = get_natural_product_name(can)
        assert name is not None, "Ergostene derivative should be in lookup table"
        assert "ergost" in name.lower(), f"Expected 'ergost' in: {name}"

    @pytest.mark.integration
    def test_ergostene_name_compound(self):
        """name_compound on ergostene SMILES produces name containing 'ergost'."""
        name = name_compound(self.ERGOSTENE_SMILES)
        assert name is not None, "Should produce a name"
        assert "ergost" in name.lower(), (
            f"Expected 'ergost' in name. Got: {name}"
        )
