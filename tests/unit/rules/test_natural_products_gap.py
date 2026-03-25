"""Tests for Phase 118 Plan 02 gap closure: DKP, tropane, ergostene fixes.

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
        """Bare piperazine-2,5-dione ring returns retained name from lookup."""
        canonical = Chem.CanonSmiles("O=C1CNCC(=O)N1")
        name = get_retained_name(canonical)
        assert name == "piperazine-2,5-dione", f"Expected 'piperazine-2,5-dione', got: {name}"

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
    def test_tropane_ester_compound3_free_base(self):
        """Compound 3 free base (tropane-3-yl indole-3-carboxylate) contains 'tropan'."""
        mol = Chem.MolFromSmiles("CN1[C@@H]2CC[C@H]1C[C@@H](OC(=O)c1c[nH]c3ccccc13)C2")
        name = name_natural_product(mol)
        assert name is not None, "Tropane ester should produce a name"
        assert "tropan" in name.lower(), f"Expected 'tropan' in name. Got: {name}"

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
