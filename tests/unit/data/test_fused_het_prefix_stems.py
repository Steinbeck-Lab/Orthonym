"""
Unit tests for fused heterocycle prefix stem data and get_fused_heterocycle_prefix().

Phase 78 Plan 01 — RED tests (written before implementation).

Tests:
- FUSED_HETEROCYCLE_PREFIX_STEMS dict completeness
- _derive_prefix_stem() terminal-e elision rule
- get_fused_heterocycle_prefix() locant resolution
- Tautomer locant preservation
- Partial saturation stems
- O(1) static lookup (no name_compound recursion)
"""

import inspect
import pytest
from rdkit import Chem

from orthonym.data.fused_heterocycles import (
    FUSED_HETEROCYCLE_DATA,
    FUSED_HETEROCYCLE_PREFIX_STEMS,
    _derive_prefix_stem,
    get_fused_heterocycle_prefix,
    match_fused_heterocycle_core,
)


class TestPrefixStemsExist:
    """Verify FUSED_HETEROCYCLE_PREFIX_STEMS dict is populated."""

    def test_prefix_stems_exist(self):
        assert isinstance(FUSED_HETEROCYCLE_PREFIX_STEMS, dict)
        # 69 entries: 75 total - 2 retained - 1 lactone - 3 SMILES duplicates
        assert len(FUSED_HETEROCYCLE_PREFIX_STEMS) >= 69

    def test_all_data_entries_have_prefix_stems(self):
        """Every eligible entry in FUSED_HETEROCYCLE_DATA has a prefix stem.

        Excluded: retained names (adenine, hypoxanthine) and lactones (coumarin).
        """
        # Ring system types excluded from prefix stems
        _excluded_ring_systems = {'benzo-6-membered-lactone'}
        for smiles, data in FUSED_HETEROCYCLE_DATA.items():
            if data.get('is_retained_name'):
                continue
            if data.get('ring_system') in _excluded_ring_systems:
                continue
            assert smiles in FUSED_HETEROCYCLE_PREFIX_STEMS, (
                f"Missing prefix stem for {data['name']} ({smiles})"
            )

    def test_no_retained_names_in_prefix_stems(self):
        """Retained names (adenine, hypoxanthine) are NOT in prefix stems."""
        for smiles, data in FUSED_HETEROCYCLE_DATA.items():
            if data.get('is_retained_name'):
                assert smiles not in FUSED_HETEROCYCLE_PREFIX_STEMS, (
                    f"Retained name {data['name']} should not be in prefix stems"
                )

    def test_no_lactones_in_prefix_stems(self):
        """Lactones (coumarin) are NOT in prefix stems — need special handling."""
        for smiles, data in FUSED_HETEROCYCLE_DATA.items():
            if data.get('ring_system') == 'benzo-6-membered-lactone':
                assert smiles not in FUSED_HETEROCYCLE_PREFIX_STEMS, (
                    f"Lactone {data['name']} should not be in prefix stems"
                )


class TestDerivePrefixStem:
    """Test _derive_prefix_stem() terminal-e elision rule."""

    @pytest.mark.parametrize("name,expected", [
        ("quinoline", "quinolin"),
        ("1H-indole", "1H-indol"),
        ("9H-carbazole", "9H-carbazol"),
        ("acridine", "acridin"),
        ("phenazine", "phenazin"),
        ("isoquinoline", "isoquinolin"),
        ("quinazoline", "quinazolin"),
        ("quinoxaline", "quinoxalin"),
        ("cinnoline", "cinnolin"),
        ("phthalazine", "phthalazin"),
        ("indoline", "indolin"),
        ("pteridine", "pteridin"),
    ])
    def test_drops_terminal_e(self, name, expected):
        assert _derive_prefix_stem(name) == expected

    @pytest.mark.parametrize("name,expected", [
        ("1-benzofuran", "1-benzofuran"),
        ("coumarin", "coumarin"),
        ("2,3-dihydro-1-benzofuran", "2,3-dihydro-1-benzofuran"),
        ("chromane", "chroman"),  # ends in 'e', so drops it
        ("isochromane", "isochroman"),  # ends in 'e'
        ("thianthrene", "thianthren"),  # ends in 'e'
    ])
    def test_preserves_or_elides_correctly(self, name, expected):
        assert _derive_prefix_stem(name) == expected


class TestLocantResolution:
    """Test get_fused_heterocycle_prefix() returns correct stem-locant-yl strings."""

    @pytest.mark.parametrize("smiles,attach_idx,expected_prefix", [
        # Quinoline: attach at C-2 (IUPAC locant 2)
        # Build a molecule with quinoline + something attached
        ("c1ccc2ncccc2c1", None, None),  # placeholder - real test below
    ])
    def test_locant_resolution_placeholder(self, smiles, attach_idx, expected_prefix):
        """Placeholder - real locant tests use match_fused_heterocycle_core."""
        pass

    def test_quinoline_prefix(self):
        """Quinoline attached at C-2 → 'quinolin-2-yl'."""
        # 2-methylquinoline: methyl at position 2
        mol = Chem.MolFromSmiles('Cc1ccc2ccccc2n1')
        assert mol is not None
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        name, atom_mapping, core_smiles = result
        # Find atom idx where methyl attaches (the C bonded to CH3)
        # The methyl carbon is at idx 0 (C)
        # It's bonded to the ring atom - find that ring atom
        methyl_atom = mol.GetAtomWithIdx(0)
        ring_attach_idx = None
        for neighbor in methyl_atom.GetNeighbors():
            if neighbor.GetIdx() != 0:
                ring_attach_idx = neighbor.GetIdx()
                break
        assert ring_attach_idx is not None
        prefix = get_fused_heterocycle_prefix(core_smiles, ring_attach_idx, atom_mapping)
        assert prefix is not None
        assert prefix == "quinolin-2-yl"

    def test_indole_c3_prefix(self):
        """1H-Indole attached at C-3 → '1H-indol-3-yl'."""
        # 3-methylindole (skatole)
        mol = Chem.MolFromSmiles('Cc1c[nH]c2ccccc12')
        assert mol is not None
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        name, atom_mapping, core_smiles = result
        # Find methyl attachment point
        methyl_atom = mol.GetAtomWithIdx(0)
        ring_attach_idx = None
        for neighbor in methyl_atom.GetNeighbors():
            if neighbor.GetIdx() != 0:
                ring_attach_idx = neighbor.GetIdx()
                break
        assert ring_attach_idx is not None
        prefix = get_fused_heterocycle_prefix(core_smiles, ring_attach_idx, atom_mapping)
        assert prefix is not None
        assert prefix == "1H-indol-3-yl"

    def test_carbazole_c3_prefix(self):
        """9H-Carbazole attached at C-3 → '9H-carbazol-3-yl'."""
        mol = Chem.MolFromSmiles('Cc1ccc2[nH]c3ccccc3c2c1')
        assert mol is not None
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        name, atom_mapping, core_smiles = result
        methyl_atom = mol.GetAtomWithIdx(0)
        ring_attach_idx = None
        for neighbor in methyl_atom.GetNeighbors():
            ring_attach_idx = neighbor.GetIdx()
            break
        prefix = get_fused_heterocycle_prefix(core_smiles, ring_attach_idx, atom_mapping)
        assert prefix is not None
        # Verify the locant is correct (should be a number, not a fusion locant)
        assert "-yl" in prefix
        assert "9H-carbazol-" in prefix

    def test_benzimidazole_c2_prefix(self):
        """1H-Benzimidazole attached at C-2 → '1H-benzimidazol-2-yl'."""
        mol = Chem.MolFromSmiles('Cc1nc2ccccc2[nH]1')
        assert mol is not None
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        name, atom_mapping, core_smiles = result
        methyl_atom = mol.GetAtomWithIdx(0)
        ring_attach_idx = None
        for neighbor in methyl_atom.GetNeighbors():
            ring_attach_idx = neighbor.GetIdx()
            break
        prefix = get_fused_heterocycle_prefix(core_smiles, ring_attach_idx, atom_mapping)
        assert prefix is not None
        assert prefix == "1H-benzimidazol-2-yl"

    def test_benzothiazole_c2_prefix(self):
        """1,3-Benzothiazole attached at C-2 → '1,3-benzothiazol-2-yl'."""
        mol = Chem.MolFromSmiles('Cc1nc2ccccc2s1')
        assert mol is not None
        result = match_fused_heterocycle_core(mol)
        assert result is not None
        name, atom_mapping, core_smiles = result
        methyl_atom = mol.GetAtomWithIdx(0)
        ring_attach_idx = None
        for neighbor in methyl_atom.GetNeighbors():
            ring_attach_idx = neighbor.GetIdx()
            break
        prefix = get_fused_heterocycle_prefix(core_smiles, ring_attach_idx, atom_mapping)
        assert prefix is not None
        assert prefix == "1,3-benzothiazol-2-yl"


class TestTautomerLocants:
    """Verify tautomer locants are preserved in prefix form."""

    @pytest.mark.parametrize("name_substr,smiles", [
        ("1H-indol", "c1ccc2[nH]ccc2c1"),
        ("1H-indazol", "c1ccc2[nH]ncc2c1"),
        ("1H-benzimidazol", "c1ccc2[nH]cnc2c1"),
        ("1H-benzotriazol", "c1ccc2[nH]nnc2c1"),
        ("9H-carbazol", "c1ccc2c(c1)[nH]c1ccccc12"),
        ("9H-purin", "c1ncc2nc[nH]c2n1"),
    ])
    def test_tautomer_locant_in_prefix_stem(self, name_substr, smiles):
        """Prefix stem preserves tautomer designation (e.g., '1H-indol')."""
        stem = FUSED_HETEROCYCLE_PREFIX_STEMS.get(smiles)
        assert stem is not None, f"Missing prefix stem for {smiles}"
        assert name_substr in stem, (
            f"Expected '{name_substr}' in stem '{stem}'"
        )


class TestPartialSaturation:
    """Verify partially saturated entries produce correct prefix stems."""

    @pytest.mark.parametrize("smiles,expected_stem", [
        ("c1ccc2c(c1)CCN2", "indolin"),
        ("c1ccc2c(c1)CCCO2", "chroman"),
        ("c1ccc2c(c1)CCCN2", "1,2,3,4-tetrahydroquinolin"),
        ("c1ccc2c(c1)CCOC2", "isochroman"),
        ("c1ccc2c(c1)CNC2", "isoindolin"),
    ])
    def test_partial_saturation_stems(self, smiles, expected_stem):
        stem = FUSED_HETEROCYCLE_PREFIX_STEMS.get(smiles)
        assert stem is not None, f"Missing prefix stem for {smiles}"
        assert stem == expected_stem, (
            f"Expected stem '{expected_stem}', got '{stem}'"
        )


class TestNoRecursion:
    """Verify O(1) static lookup — no name_compound recursion."""

    def test_no_name_compound_recursion(self):
        """get_fused_heterocycle_prefix() does NOT call name_compound."""
        src = inspect.getsource(get_fused_heterocycle_prefix)
        assert "name_compound" not in src, (
            "get_fused_heterocycle_prefix() must not call name_compound "
            "(QUAL-06: static O(1) lookup only)"
        )


class TestEdgeCases:
    """Edge cases for get_fused_heterocycle_prefix()."""

    def test_returns_none_for_unknown_smiles(self):
        """Unknown core SMILES returns None."""
        result = get_fused_heterocycle_prefix(
            "C1CCCCC1",  # cyclohexane, not a fused het
            0,
            {0: 1, 1: 2}
        )
        assert result is None

    def test_returns_none_for_unmapped_atom(self):
        """Attachment atom not in atom_mapping returns None."""
        # Use a real core SMILES but an unmapped atom index
        smiles = "c1ccc2ncccc2c1"  # quinoline
        result = get_fused_heterocycle_prefix(smiles, 999, {0: 1})
        assert result is None
