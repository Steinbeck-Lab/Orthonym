"""
Integration tests for Plan 15-06:
  1. Ion recursion fix (multi-charged species no longer crash)
  2. Zwitterion literal fix (no name is literally "zwitterion")
  3. Macrocyclic heterocycle completion (no names truncated at -oxa/-aza)

Tests cover:
  - Dianions, diammonium, calcium phosphate, CoA thioester fragments
  - Phospholipid zwitterions, carnitine, sphingomyelin-like
  - Crown ethers, large ring azamacrocycles, macrocyclic lactone rings
"""

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from orthonym.rules.heterocycles import name_heterocycle


# ============================================================================
# Bug 1: Ion recursion guards
# ============================================================================


class TestIonRecursionGuard:
    """Multi-charged species must not cause RecursionError."""

    @pytest.fixture(autouse=True)
    def _namer(self):
        self.namer = Orthonym()

    def test_dianion_succinate(self):
        """Succinate dianion should not crash."""
        result = self.namer.name('[O-]C(=O)CCC(=O)[O-]')
        assert result  # non-empty
        assert 'succinate' in result.lower() or 'butanedio' in result.lower()

    def test_diammonium_propane(self):
        """Propane-1,3-diammonium should not crash, names as neutral."""
        result = self.namer.name('[NH3+]CCC[NH3+]')
        assert result
        assert 'amine' in result.lower() or 'diamine' in result.lower()

    def test_calcium_phosphate_salt(self):
        """Calcium hydrogen phosphate should name as salt, not crash."""
        result = self.namer.name('[Ca+2].[O-]P([O-])(=O)O')
        assert result
        assert 'calcium' in result.lower()
        assert 'phosphate' in result.lower()

    def test_multi_cation_organic(self):
        """Ethane-1,2-diammonium should not crash."""
        result = self.namer.name('[NH3+]CC[NH3+]')
        assert result
        assert 'amine' in result.lower() or 'diamine' in result.lower()

    def test_iron_salt_returns_nonempty_or_empty(self):
        """Iron-containing salt should not crash (metallic = out of scope)."""
        try:
            result = self.namer.name('[Fe+3].[Cl-].[Cl-].[Cl-]')
            # Either produces a name or empty string, but no crash
            assert isinstance(result, str)
        except ValueError:
            # ValueError for invalid SMILES is acceptable
            pass


# ============================================================================
# Bug 2: Zwitterion literal elimination
# ============================================================================


class TestZwitterionLiteral:
    """No generated name should be the literal string 'zwitterion'."""

    @pytest.fixture(autouse=True)
    def _namer(self):
        self.namer = Orthonym()

    def test_carnitine_not_literal(self):
        """Carnitine zwitterion should not return 'zwitterion'."""
        result = self.namer.name('C[N+](C)(C)CC(O)CC([O-])=O')
        assert result != 'zwitterion'

    def test_betaine(self):
        """Betaine should be named 'betaine', not 'zwitterion'."""
        result = self.namer.name('C[N+](C)(C)CC([O-])=O')
        assert result == 'betaine'

    def test_glycine_zwitterion(self):
        """Glycine zwitterion should get a systematic name."""
        result = self.namer.name('[NH3+]CC([O-])=O')
        assert result
        assert result != 'zwitterion'
        assert 'azaniumyl' in result or 'glycine' in result.lower()

    def test_large_zwitterion_neutralize(self):
        """Large zwitterion (>6 C) should be named via neutral fallback."""
        result = self.namer.name('[NH3+]CCCCCCC([O-])=O')
        assert result
        assert result != 'zwitterion'

    def test_phosphocholine_fragment_not_literal(self):
        """Phosphocholine-like zwitterion must not return 'zwitterion'."""
        result = self.namer.name('C[N+](C)(C)CCOP([O-])(=O)O')
        assert result != 'zwitterion'


# ============================================================================
# Bug 3: Macrocyclic heterocycle name completion
# ============================================================================


class TestMacrocyclicNames:
    """Large heterocyclic ring names must not be truncated."""

    def _name_ring(self, smiles: str) -> str:
        """Name the first ring in the molecule."""
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"Invalid SMILES: {smiles}"
        ri = mol.GetRingInfo()
        rings = ri.AtomRings()
        assert len(rings) >= 1, f"No rings found in {smiles}"
        return name_heterocycle(mol, rings[0])

    def test_12_crown_4(self):
        """12-crown-4 should be 1,4,7,10-tetraoxacyclododecane."""
        name = self._name_ring('C1COCCOCCOCCO1')
        assert 'cyclododecane' in name
        assert 'tetraoxa' in name
        assert not name.endswith('oxa')

    def test_15_crown_5(self):
        """15-crown-5 should end with cyclo...ane, not truncated."""
        name = self._name_ring('C1COCCOCCOCCOCCOC1')
        assert 'cyclo' in name
        assert name.endswith('ane')
        assert 'pentaoxa' in name

    def test_18_crown_6(self):
        """18-crown-6 should end with cyclo...ane."""
        name = self._name_ring('C1COCCOCCOCCOCCOCCOC1')
        assert 'cyclo' in name
        assert name.endswith('ane')

    def test_large_aza_macrocycle(self):
        """Large aza macrocycle should get complete name."""
        name = self._name_ring('C1CCNCCNCCNCCNCC1')
        assert 'cyclo' in name
        assert 'aza' in name
        assert name.endswith('ane')
        assert not name.endswith('aza')

    def test_mixed_oxa_aza_macrocycle(self):
        """Mixed O/N macrocycle should have both prefixes and parent."""
        name = self._name_ring('C1CCOCCNCCOCCNCC1')
        assert 'oxa' in name
        assert 'aza' in name
        assert 'cyclo' in name
        assert name.endswith('ane')

    def test_no_truncation_at_prefix(self):
        """No macrocyclic name should end with just a replacement prefix."""
        test_cases = [
            'C1COCCOCCOCCO1',       # 12-crown-4
            'C1CCNCCNCCNCC1',       # triaza-12
            'C1CCSCCSCCSCC1',       # trithia-12
        ]
        for smiles in test_cases:
            name = self._name_ring(smiles)
            assert not name.endswith(('oxa', 'aza', 'thia')), \
                f"Truncated name '{name}' for SMILES {smiles}"

    def test_small_heterocycle_unchanged(self):
        """HW naming for small rings (3-10) should still work."""
        # Oxirane (3-membered)
        name = self._name_ring('C1CO1')
        assert name == 'oxirane'

        # Tetrahydropyran (6-membered, retained name for oxane)
        name = self._name_ring('C1CCOCC1')
        assert name in ('oxane', 'oxane')

    def test_10_membered_hw_still_works(self):
        """10-membered ring should still use HW naming."""
        # 10-membered ring with one O
        name = self._name_ring('C1CCCCCOCCC1')
        assert 'ecane' in name  # HW stem for 10-membered saturated

    def test_11_membered_uses_replacement(self):
        """11-membered ring should switch to replacement nomenclature."""
        name = self._name_ring('C1CCCCCCOCCCCC1')
        # Should NOT be empty or truncated
        assert len(name) > 3
        # Should end with a proper parent name
        assert 'cyclo' in name


# ============================================================================
# Cross-cutting: no regression on basic ion naming
# ============================================================================


class TestIonNamingRegression:
    """Basic ion naming should not be broken by the recursion guards."""

    @pytest.fixture(autouse=True)
    def _namer(self):
        self.namer = Orthonym()

    def test_ammonium(self):
        assert self.namer.name('[NH4+]') == 'ammonium'

    def test_acetate(self):
        assert self.namer.name('CC(=O)[O-]') == 'acetate'

    def test_sodium_acetate(self):
        result = self.namer.name('[Na+].CC(=O)[O-]')
        assert result == 'sodium acetate'

    def test_methylammonium(self):
        result = self.namer.name('C[NH3+]')
        assert result == 'methylammonium'
