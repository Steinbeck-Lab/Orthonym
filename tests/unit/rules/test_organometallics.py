"""Phase 161 unit tests for organometallic rules (P-69 + Salzer 1999).

Mirrors tests/unit/test_functional_groups.py rules-layer pattern. Tests
the ORGM_LIGAND_ORDER / METAL_NAMES / METAL_OXIDATION_STATE_HINTS tables
+ the assemble_organometallic_name + select_ligand_naming public API.

CONTEXT D-06 ENFORCEMENT: this module verifies that
src/orthonym/rules/organometallics.py does NOT import from
src/orthonym/rules/seniority.py — the two cascades MUST stay separate.

NEVER uses @pytest.mark.xfail (CONTEXT D-29) — honest-fail-on-data.
"""
import inspect
import pytest
from rdkit import Chem

from orthonym.rules.organometallics import (
    ORGM_LIGAND_ORDER,
    METAL_OXIDATION_STATE_HINTS,
    METAL_RANKING_FOR_PARENT_SELECTION,
    select_ligand_naming,
    assemble_organometallic_name,
)
from orthonym.data.organometallics import METAL_NAMES, LIGAND_NAMES, LIGAND_ETA_DEFAULTS
from orthonym.perception.metals import detect_metal_complex


# ---------------------------------------------------------------------------
# Class 1 — TestOrgmLigandOrder (≥ 5 tests)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOrgmLigandOrder:
    """Alphabetic ordering per Salzer 1999 §5.2."""

    def test_orgm_ligand_order_is_list(self):
        assert isinstance(ORGM_LIGAND_ORDER, list)

    def test_orgm_ligand_order_size_ge_15(self):
        """Salzer §5.2 alphabetic list has >= 15 ligand names."""
        assert len(ORGM_LIGAND_ORDER) >= 15

    def test_carbonyl_in_order(self):
        assert 'carbonyl' in ORGM_LIGAND_ORDER

    def test_cyclopentadienyl_in_order(self):
        assert 'cyclopentadienyl' in ORGM_LIGAND_ORDER

    def test_methyl_in_order(self):
        assert 'methyl' in ORGM_LIGAND_ORDER

    def test_phenyl_in_order(self):
        assert 'phenyl' in ORGM_LIGAND_ORDER

    def test_benzene_in_order(self):
        """Benzene is a π-ligand; appears as 'benzene' not 'phenyl' for arene complexes."""
        assert 'benzene' in ORGM_LIGAND_ORDER

    def test_no_seniority_import_in_rules_module(self):
        """CONTEXT D-06 HARD INVARIANT: rules/organometallics.py NEVER imports rules.seniority."""
        from orthonym.rules import organometallics
        src = inspect.getsource(organometallics)
        # Compiled grep gates per Plan-02 task 02-03
        # (allowed in docstring comments cautioning against doing it)
        forbidden_patterns = [
            'from ..rules.seniority',
            'from orthonym.rules.seniority',
            'import orthonym.rules.seniority',
        ]
        # Check non-comment lines
        import_lines = [
            line for line in src.splitlines()
            if line.strip().startswith(('from ', 'import '))
        ]
        for forbidden in forbidden_patterns:
            for line in import_lines:
                assert forbidden not in line, (
                    f"CONTEXT D-06 violation: '{forbidden}' found in "
                    f"rules/organometallics.py import:\n  {line}"
                )


# ---------------------------------------------------------------------------
# Class 2 — TestMetalNames (≥ 8 tests)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetalNames:
    """METAL_NAMES lookup integrity + naming_system branch coverage."""

    def test_iron_direct_name(self):
        assert METAL_NAMES['Fe']['direct'] == 'iron'
        assert METAL_NAMES['Fe']['naming_system'] == 'metal_direct'

    def test_tin_hydride_parent_name(self):
        """Sn uses hydride-parent system per IUPAC P-69.2 (stannane)."""
        assert METAL_NAMES['Sn']['hydride_parent'] == 'stannane'
        assert METAL_NAMES['Sn']['naming_system'] == 'hydride_parent'

    def test_lead_hydride_parent(self):
        """Pb → plumbane per P-69.2."""
        assert METAL_NAMES['Pb']['hydride_parent'] == 'plumbane'

    def test_silicon_hydride_parent(self):
        """Si → silane per P-69.2."""
        assert METAL_NAMES['Si']['hydride_parent'] == 'silane'

    def test_germanium_hydride_parent(self):
        """Ge → germane per P-69.2."""
        assert METAL_NAMES['Ge']['hydride_parent'] == 'germane'

    def test_lithium_metal_direct(self):
        """Li → lithium (metal-direct; alkali metals)."""
        assert METAL_NAMES['Li']['direct'] == 'lithium'
        assert METAL_NAMES['Li']['naming_system'] == 'metal_direct'

    def test_mercury_metal_direct(self):
        """Hg → mercury (Group 12, metal-direct)."""
        assert METAL_NAMES['Hg']['direct'] == 'mercury'

    def test_ruthenium_present(self):
        """Ru → ruthenium (Tier-1 metallocene metal)."""
        assert METAL_NAMES['Ru']['direct'] == 'ruthenium'

    def test_iron_no_hydride_parent(self):
        """Fe has no hydride-parent name (transition metal, metal-direct only)."""
        assert METAL_NAMES['Fe']['hydride_parent'] is None

    def test_all_tier_1_metals_present(self):
        """All 9 Tier-1 metallocene metals (per canary) must be in METAL_NAMES."""
        tier_1_metals = ['Fe', 'Ru', 'Os', 'Co', 'Ni', 'Cr', 'V', 'Mn']
        for sym in tier_1_metals:
            assert sym in METAL_NAMES, f"{sym} missing from METAL_NAMES"
            assert 'direct' in METAL_NAMES[sym]
            assert 'naming_system' in METAL_NAMES[sym]


# ---------------------------------------------------------------------------
# Class 3 — TestMetalOxidationStateHints (≥ 5 tests)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetalOxidationStateHints:
    """METAL_OXIDATION_STATE_HINTS per (metal, ligand_class) coverage per CONTEXT D-07."""

    def test_iron_cp2_stock_required_in_systematic(self):
        """Ferrocene systematic includes (II); PIN omits Stock."""
        hints = METAL_OXIDATION_STATE_HINTS[('Fe', 'Cp2')]
        assert hints['default_state'] == 2
        assert hints['stock_required_systematic'] is True
        assert hints['stock_required_pin'] is False

    def test_lithium_alkyl_never_stock(self):
        """Alkali metals always +1; Stock omitted per D-07."""
        hints = METAL_OXIDATION_STATE_HINTS[('Li', 'alkyl')]
        assert hints['stock_required_systematic'] is False
        assert hints['stock_required_pin'] is False

    def test_tin_alkyl4_never_stock(self):
        """Group 14 hydride-parent system; Stock implicit per P-69.2."""
        hints = METAL_OXIDATION_STATE_HINTS[('Sn', 'alkyl4')]
        assert hints['stock_required_systematic'] is False
        assert hints['default_state'] == 4

    def test_iron_co5_stock_required_in_systematic(self):
        """Fe(CO)5 systematic: pentacarbonyliron(0)."""
        hints = METAL_OXIDATION_STATE_HINTS[('Fe', 'CO5')]
        assert hints['default_state'] == 0
        assert hints['stock_required_systematic'] is True

    def test_chromium_bz2_stock_in_systematic(self):
        """bis(η⁶-benzene)chromium(0) systematic."""
        hints = METAL_OXIDATION_STATE_HINTS[('Cr', 'bz2')]
        assert hints['default_state'] == 0
        assert hints['stock_required_systematic'] is True

    def test_iron_cp2_cation_stock_state_3(self):
        """Ferrocenium: Fe(+3) form."""
        hints = METAL_OXIDATION_STATE_HINTS[('Fe', 'Cp2_cation')]
        assert hints['default_state'] == 3
        assert hints['stock_required_systematic'] is True

    def test_vanadium_co6_anion_stock_in_pin_too(self):
        """Anionic V(CO)6(-1) — Stock required in PIN per Ewens-Bassett."""
        hints = METAL_OXIDATION_STATE_HINTS[('V', 'CO6_anion')]
        assert hints['default_state'] == -1
        assert hints['stock_required_pin'] is True


# ---------------------------------------------------------------------------
# Class 4 — TestAssembleOrganometallicName (≥ 8 tests)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAssembleOrganometallicName:
    """End-to-end name assembly per fixture per CONTEXT D-01 tier scope."""

    def test_ferrocene_systematic(self):
        """Tier-1 ferrocene systematic form per Salzer §5.4."""
        mol = Chem.MolFromSmiles('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1')
        mc = detect_metal_complex(mol)
        assert mc is not None
        result = assemble_organometallic_name(mc, mol, style='systematic')
        assert result is not None
        name, _, _ = result
        assert name == 'bis(η⁵-cyclopentadienyl)iron(II)'

    def test_ruthenocene_systematic(self):
        """Tier-1 ruthenocene systematic form."""
        mol = Chem.MolFromSmiles('[Ru+2].c1cc[cH-]c1.c1cc[cH-]c1')
        mc = detect_metal_complex(mol)
        result = assemble_organometallic_name(mc, mol, style='systematic')
        assert result is not None
        name, _, _ = result
        assert name == 'bis(η⁵-cyclopentadienyl)ruthenium(II)'

    def test_methyllithium_pin(self):
        """Tier-3 σ-bonded alkali metal."""
        mol = Chem.MolFromSmiles('[Li][CH3]')
        mc = detect_metal_complex(mol)
        result = assemble_organometallic_name(mc, mol, style='pin')
        assert result is not None
        name, _, _ = result
        assert name == 'methyllithium'

    def test_tetramethylstannane_hydride_parent(self):
        """Risk R-06: Sn uses hydride-parent system; NOT 'tetramethyltin'."""
        mol = Chem.MolFromSmiles('C[Sn](C)(C)C')
        mc = detect_metal_complex(mol)
        result = assemble_organometallic_name(mc, mol, style='pin')
        assert result is not None
        name, _, _ = result
        assert name == 'tetramethylstannane'

    def test_pentacarbonyliron_pin(self):
        """Tier-2 mononuclear metal carbonyl Fe(CO)5."""
        mol = Chem.MolFromSmiles(
            '[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[Fe]'
        )
        mc = detect_metal_complex(mol)
        result = assemble_organometallic_name(mc, mol, style='pin')
        assert result is not None
        name, _, _ = result
        assert name == 'pentacarbonyliron'

    def test_pentacarbonyliron_systematic(self):
        """Tier-2 systematic: pentacarbonyliron(0)."""
        mol = Chem.MolFromSmiles(
            '[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[Fe]'
        )
        mc = detect_metal_complex(mol)
        result = assemble_organometallic_name(mc, mol, style='systematic')
        assert result is not None
        name, _, _ = result
        assert name == 'pentacarbonyliron(0)'

    def test_dimethylzinc_pin(self):
        """Tier-3 σ-bonded Group 12 dimer."""
        mol = Chem.MolFromSmiles('C[Zn]C')
        mc = detect_metal_complex(mol)
        result = assemble_organometallic_name(mc, mol, style='pin')
        assert result is not None
        name, _, _ = result
        assert name == 'dimethylzinc'

    def test_dibenzene_chromium_systematic(self):
        """Tier-4 bis-arene Cr(0) complex."""
        mol = Chem.MolFromSmiles('[Cr].c1ccccc1.c1ccccc1')
        mc = detect_metal_complex(mol)
        result = assemble_organometallic_name(mc, mol, style='systematic')
        assert result is not None
        name, _, _ = result
        assert name == 'bis(η⁶-benzene)chromium(0)'

    def test_seniority_py_not_imported(self):
        """CONTEXT D-06 enforcement: rules/organometallics.py NEVER imports rules.seniority.

        Restated test (also in TestOrgmLigandOrder; explicit per Plan-04 spec).
        """
        from orthonym.rules import organometallics
        src = inspect.getsource(organometallics)
        import_lines = [
            line for line in src.splitlines()
            if line.strip().startswith(('from ', 'import '))
        ]
        for line in import_lines:
            assert 'seniority' not in line


# ---------------------------------------------------------------------------
# Class 5 — TestSelectLigandNaming (≥ 5 tests)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSelectLigandNaming:
    """select_ligand_naming(): Plan-02 stub returns empty string.

    Plan-03 implementation kept the stub since the actual ligand-name lookup
    happens inside assemble_organometallic_name. Tests verify stub contract +
    that future Phase 161.1+ can extend without breaking the API.
    """

    def test_select_ligand_naming_returns_string(self):
        """Stub returns a string (empty)."""
        result = select_ligand_naming(None, 1)
        assert isinstance(result, str)

    def test_select_ligand_naming_default_style_pin(self):
        """Default style='pin' accepts (None, hapticity) signature."""
        result = select_ligand_naming(None, 5)
        # Stub returns empty string
        assert result == ''

    def test_select_ligand_naming_style_systematic(self):
        """style='systematic' accepted."""
        result = select_ligand_naming(None, 6, style='systematic')
        assert isinstance(result, str)

    def test_select_ligand_naming_does_not_raise(self):
        """Stub does not raise on (None, 1)."""
        try:
            select_ligand_naming(None, 1)
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"select_ligand_naming(None, 1) raised: {e!r}")

    def test_ligand_names_table_has_carbonyl(self):
        """LIGAND_NAMES is the canonical SMARTS→name table; carbonyl present."""
        assert LIGAND_NAMES['[C-]#[O+]'] == 'carbonyl'

    def test_ligand_names_table_has_methyl(self):
        """LIGAND_NAMES['CH3'] entry exists."""
        assert '[CH3]' in LIGAND_NAMES or 'methyl' in LIGAND_NAMES.values()


# ---------------------------------------------------------------------------
# Class 6 — TestMetalRankingForParentSelection (≥ 3 tests)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMetalRankingForParentSelection:
    """METAL_RANKING_FOR_PARENT_SELECTION per CONTEXT D-06 forward compat."""

    def test_iron_in_ranking(self):
        """Fe is part of the ranking table."""
        assert 'Fe' in METAL_RANKING_FOR_PARENT_SELECTION

    def test_lithium_in_ranking(self):
        assert 'Li' in METAL_RANKING_FOR_PARENT_SELECTION

    def test_tin_in_ranking(self):
        """Sn is part of the ranking table (hydride-parent metals included for forward compat)."""
        assert 'Sn' in METAL_RANKING_FOR_PARENT_SELECTION

    def test_ranking_uses_atomic_number(self):
        """Ranking is by atomic number (Z) for tie-break."""
        assert METAL_RANKING_FOR_PARENT_SELECTION['Fe'] == 26
        assert METAL_RANKING_FOR_PARENT_SELECTION['Li'] == 3
        assert METAL_RANKING_FOR_PARENT_SELECTION['Sn'] == 50
