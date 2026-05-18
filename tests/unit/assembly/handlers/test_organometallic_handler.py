"""Phase 161 unit tests for organometallic handler (P-69 + IR-10 + Salzer 1999).

Mirrors tests/unit/assembly/handlers/test_simple_molecule.py — the FIRST and
ONLY tree-emitting handler test from Phase 160.2 Plan-10 DECOMP-02. ORGM is
the SECOND tree-emitting handler in Orthonym per CONTEXT D-11.

CONTEXT D-11 BYTE-IDENTICAL CONTRACT: name_tree_to_string(result.tree) ==
result.name for ALL non-None returns. These tests verify the contract holds
per-tier (Tier-1 retained + Tier-2 carbonyl + Tier-3 σ-bonded + Tier-4
mixed η-bonded).

NEVER uses @pytest.mark.xfail (CONTEXT D-29) — honest-fail-on-data.
"""
import inspect
import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.assembly.handlers.organometallic import name_organometallic
from orthonym.assembly.name_tree import NameTreeNode, NamingResult
from orthonym.assembly.name_tree_to_string import name_tree_to_string
from orthonym.routing.dispatch_table import DISPATCH_TABLE, StoutClass


# ---------------------------------------------------------------------------
# Class 1 — TestNameOrganometallicSignature (≥ 4 tests)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNameOrganometallicSignature:
    """Per CONTEXT D-04 + D-11: name_organometallic(features, mol, style='pin') -> Optional[NamingResult]."""

    def test_signature_shape(self):
        """Handler has (features, mol, style) signature."""
        sig = inspect.signature(name_organometallic)
        params = list(sig.parameters.keys())
        assert len(params) >= 2, (
            f"Expected >= 2 parameters (features, mol[, style]); got {params}"
        )

    def test_returns_none_on_none_mol(self):
        """Defensive guard: name_organometallic(None, None) returns None."""
        assert name_organometallic(None, None) is None

    def test_returns_none_on_non_organometallic(self):
        """Acetic acid (no metal) cascades to None."""
        mol = Chem.MolFromSmiles('CC(=O)O')
        result = name_organometallic(None, mol)
        assert result is None

    def test_returns_none_on_ethanol(self):
        """Ethanol (no metal) cascades to None."""
        mol = Chem.MolFromSmiles('CCO')
        result = name_organometallic(None, mol)
        assert result is None

    def test_handler_registered_in_dispatch_table(self):
        """ORGM handler is registered under StoutClass.ORGANOMETALLIC at priority 50."""
        assert StoutClass.ORGANOMETALLIC in DISPATCH_TABLE
        entry = DISPATCH_TABLE[StoutClass.ORGANOMETALLIC]
        assert entry.priority == 50

    def test_handler_returns_namingresult_type(self):
        """For ORGM compounds, returns NamingResult (NamedTuple)."""
        mol = Chem.MolFromSmiles('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1')
        result = name_organometallic(None, mol)
        assert result is not None
        assert isinstance(result, NamingResult)
        assert hasattr(result, 'name')
        assert hasattr(result, 'tree')


# ---------------------------------------------------------------------------
# Class 2 — TestTier1Retained (≥ 8 tests for parent metallocenes)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTier1Retained:
    """Tier-1 unsubstituted parent metallocenes per CONTEXT D-01."""

    def test_ferrocene_pin(self):
        mol = Chem.MolFromSmiles('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1')
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert result.name == 'ferrocene'

    def test_ferrocene_systematic(self):
        mol = Chem.MolFromSmiles('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1')
        result = name_organometallic(None, mol, style='systematic')
        assert result is not None
        assert result.name == 'bis(η⁵-cyclopentadienyl)iron(II)'

    def test_ruthenocene_pin(self):
        mol = Chem.MolFromSmiles('[Ru+2].c1cc[cH-]c1.c1cc[cH-]c1')
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert result.name == 'ruthenocene'

    def test_osmocene_pin(self):
        mol = Chem.MolFromSmiles('[Os+2].c1cc[cH-]c1.c1cc[cH-]c1')
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert result.name == 'osmocene'

    def test_cobaltocene_pin(self):
        mol = Chem.MolFromSmiles('[Co+2].c1cc[cH-]c1.c1cc[cH-]c1')
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert result.name == 'cobaltocene'

    def test_nickelocene_pin(self):
        mol = Chem.MolFromSmiles('[Ni+2].c1cc[cH-]c1.c1cc[cH-]c1')
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert result.name == 'nickelocene'

    def test_chromocene_pin(self):
        mol = Chem.MolFromSmiles('[Cr+2].c1cc[cH-]c1.c1cc[cH-]c1')
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert result.name == 'chromocene'

    def test_vanadocene_pin(self):
        mol = Chem.MolFromSmiles('[V+2].c1cc[cH-]c1.c1cc[cH-]c1')
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert result.name == 'vanadocene'

    def test_manganocene_pin(self):
        mol = Chem.MolFromSmiles('[Mn+2].c1cc[cH-]c1.c1cc[cH-]c1')
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert result.name == 'manganocene'

    def test_ferrocenium_pin(self):
        """Fe(+3) form: ferrocenium retained PIN."""
        mol = Chem.MolFromSmiles('[Fe+3].c1cc[cH-]c1.c1cc[cH-]c1')
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert result.name == 'ferrocenium'

    def test_ferrocenium_systematic(self):
        """Fe(+3) form: bis(η⁵-cyclopentadienyl)iron(III)."""
        mol = Chem.MolFromSmiles('[Fe+3].c1cc[cH-]c1.c1cc[cH-]c1')
        result = name_organometallic(None, mol, style='systematic')
        assert result is not None
        assert result.name == 'bis(η⁵-cyclopentadienyl)iron(III)'

    def test_tree_class_id_organometallic(self):
        """All ORGM results emit NameTreeNode(class_id='organometallic')."""
        mol = Chem.MolFromSmiles('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1')
        result = name_organometallic(None, mol)
        assert result.tree is not None
        assert result.tree.class_id == 'organometallic'


# ---------------------------------------------------------------------------
# Class 3 — TestTier2Carbonyl (≥ 4 tests for metal carbonyls)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTier2Carbonyl:
    """Tier-2 mononuclear metal carbonyls per Salzer §6."""

    def test_pentacarbonyliron_pin(self):
        """Fe(CO)5 PIN."""
        mol = Chem.MolFromSmiles(
            '[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[Fe]'
        )
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert result.name == 'pentacarbonyliron'

    def test_pentacarbonyliron_systematic(self):
        """Fe(CO)5 systematic: includes (0) Stock."""
        mol = Chem.MolFromSmiles(
            '[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[Fe]'
        )
        result = name_organometallic(None, mol, style='systematic')
        assert result is not None
        assert result.name == 'pentacarbonyliron(0)'

    def test_tetracarbonylnickel_pin(self):
        """Ni(CO)4 PIN."""
        mol = Chem.MolFromSmiles(
            '[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[Ni]'
        )
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert result.name == 'tetracarbonylnickel'

    def test_hexacarbonylchromium_pin(self):
        """Cr(CO)6 PIN."""
        mol = Chem.MolFromSmiles(
            '[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[Cr]'
        )
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert result.name == 'hexacarbonylchromium'

    def test_hexacarbonylvanadium_anion(self):
        """V(CO)6(-1) PIN: Ewens-Bassett charge form."""
        mol = Chem.MolFromSmiles(
            '[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[V-]'
        )
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert result.name == 'hexacarbonylvanadium(1-)'


# ---------------------------------------------------------------------------
# Class 4 — TestTier3SigmaBonded (≥ 4 tests)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTier3SigmaBonded:
    """Tier-3 σ-bonded main-group organometallics."""

    def test_methyllithium_pin(self):
        mol = Chem.MolFromSmiles('[Li][CH3]')
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert result.name == 'methyllithium'

    def test_dimethylzinc_pin(self):
        mol = Chem.MolFromSmiles('C[Zn]C')
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert result.name == 'dimethylzinc'

    def test_tetramethylstannane_pin(self):
        """Sn hydride-parent system: tetramethylstannane."""
        mol = Chem.MolFromSmiles('C[Sn](C)(C)C')
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert result.name == 'tetramethylstannane'

    def test_tetramethylsilane_pin(self):
        """Si hydride-parent system: tetramethylsilane."""
        mol = Chem.MolFromSmiles('C[Si](C)(C)C')
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert result.name == 'tetramethylsilane'

    def test_ethylmagnesium_bromide_pin(self):
        """Grignard form: 'ethylmagnesium bromide' (space-separated)."""
        mol = Chem.MolFromSmiles('C[CH2][Mg][Br]')
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert result.name == 'ethylmagnesium bromide'


# ---------------------------------------------------------------------------
# Class 5 — TestNameTreeRoundTrip (≥ 5 tests per CONTEXT D-11)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNameTreeRoundTrip:
    """CONTEXT D-11 byte-identical contract: name_tree_to_string(tree) == result.name."""

    def test_ferrocene_round_trip(self):
        mol = Chem.MolFromSmiles('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1')
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert name_tree_to_string(result.tree) == result.name

    def test_ferrocene_systematic_round_trip(self):
        mol = Chem.MolFromSmiles('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1')
        result = name_organometallic(None, mol, style='systematic')
        assert result is not None
        assert name_tree_to_string(result.tree) == result.name

    def test_methyllithium_round_trip(self):
        mol = Chem.MolFromSmiles('[Li][CH3]')
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert name_tree_to_string(result.tree) == result.name

    def test_pentacarbonyliron_round_trip(self):
        mol = Chem.MolFromSmiles(
            '[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[Fe]'
        )
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert name_tree_to_string(result.tree) == result.name

    def test_tetramethylstannane_round_trip(self):
        mol = Chem.MolFromSmiles('C[Sn](C)(C)C')
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert name_tree_to_string(result.tree) == result.name

    def test_dimethylzinc_round_trip(self):
        mol = Chem.MolFromSmiles('C[Zn]C')
        result = name_organometallic(None, mol, style='pin')
        assert result is not None
        assert name_tree_to_string(result.tree) == result.name

    def test_iupac_section_cite_populated(self):
        """All ORGM trees include iupac_section_cite per CONTEXT D-11."""
        mol = Chem.MolFromSmiles('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1')
        result = name_organometallic(None, mol)
        assert result.tree.iupac_section_cite is not None
        assert (
            'P-69' in result.tree.iupac_section_cite
            or 'Salzer' in result.tree.iupac_section_cite
        )

    def test_byte_identical_name_via_namer(self):
        """Pipeline-level: name_compound for ferrocene returns 'ferrocene'."""
        result = name_compound('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1')
        assert result == 'ferrocene'

    def test_dispatch_entry_priority_50(self):
        """ORGM CFR entry is priority 50 per CONTEXT D-02."""
        entry = DISPATCH_TABLE[StoutClass.ORGANOMETALLIC]
        assert entry.priority == 50

    def test_dispatch_entry_side_effect_inventory_empty(self):
        """CONTEXT D-12 hard invariant: side_effect_inventory=() for ORGM."""
        entry = DISPATCH_TABLE[StoutClass.ORGANOMETALLIC]
        assert entry.side_effect_inventory == ()
