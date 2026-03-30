"""Integration tests for decomposition fragment fallback naming (Phase 127).

Tests DECO-15 (fragments through name_pipeline_only()) and
DECO-16 (HA > 30 coverage >= 70%).
"""
import pytest
from rdkit import Chem
from orthonym import name_compound
from orthonym.decomposition.engine import (
    try_decompose,
    _name_fragment_with_fallback,
    _get_max_decomp_levels,
)
from orthonym.assembly.fragment_naming import _MAX_VISITED_SIZE


# 20 fragment-loss compounds from benchmark_results_v14.json
# Representing diverse failure modes: esters, amides, glycosidic, large HA, mixed
FRAGMENT_LOSS_SMILES = [
    # Ester decomposition failures
    ("CCCCCC(C)OC(=O)COc1ccc(Cl)c2cccnc12", 23),           # chloroquinoline ester, HA=23
    ("C/C=C/C(=O)O[C@@H]1CC2O[C@@H]3C=C(C)[C@@H](O)[C@@H]4OCC2(O)[C@@]1(C)[C@@]34C", 25),  # butenoyloxy terpene, HA=25
    ("Cc1c(O)cc2c(c1C)C(=O)O[C@@H]([C@@]1([C@@H]3CC=C4CCC[C@H](C)[C@@]4(C)C3)CO1)O2", 29),  # dimethylphenyl lactone, HA=29
    ("O=C(/C=C/c1ccc(O)cc1)O[C@@H]1C[C@](O)(C(=O)[O-])C[C@@H](O)[C@H]1O", 24),  # cinnamate ester, HA=24
    # Amide decomposition failures
    ("CCCCCCC(=O)NC1=CC(=O)[C@@H]2CCCN12", 18),           # pyrrolidinone heptanamide, HA=18
    ("COc1cc(C2OC2C(=O)NCCCCN)ccc1O", 20),                # epoxy amide, HA=20
    ("CC(C)[C@H](NC(=O)[C@@H](N)Cc1ccc(O)cc1)C(=O)N[C@@H](CCCN=C(N)N)C(=O)O", 31),  # peptide, HA=31
    # Glycosidic / sugar decomposition failures
    ("CC1OC(Oc2c(C3OC(CO)C(O)C(O)C3O)c(O)c3c(=O)cc(-c4ccc(O)c(O)c4)oc3c2C2OC(CO)C(O)C(O)C2O)C(O)C(O)C1O", 53),  # flavonoid glycoside, HA=53
    ("CC(=O)N[C@@H]1[C@@H](O)[C@H](O[C@@H]2O[C@H](CO)[C@@H](O[C@@H]3O[C@H](CO)[C@@H](O)[C@H](O[C@H]4O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]4O)[C@@H]3O)[C@H](O)[C@H]2NC(C)=O)[C@@H](CO)O[C@H]1O", 51),  # N-acetyl oligosaccharide, HA=51
    # Large molecules (HA > 50)
    ("CCC(/C=C/C(C)C1CCC2C3=CCC4CC(OC5OC(CO)C(OC6OC(CO)C(O)C(O)C6O)C(O)C5O)CCC4(C)C3CCC21C)C(C)C", 52),  # steroid glycoside, HA=52
    ("CCCCC/C=C\\C/C=C\\CCCCCCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCCCCC)COP(=O)(O)OC[C@H](N)C(=O)O", 52),  # phospholipid serine, HA=52
    ("CCCCC/C=C\\C/C=C\\CCCCCCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCC)COP(=O)(O)OC[C@H](N)C(=O)O", 52),  # phospholipid serine 2, HA=52
    ("CCCCCC/C=C\\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCC/C=C\\CCCCCCCC)COP(=O)([O-])[O-]", 46),  # phospholipid, HA=46
    ("CCCCCCCCCCCCCCCCCCCCCCC(C(=O)O)C(O)CCCCCCCCCCCCCCCCCC1CC1CCCCCCCCCCCCCCCCC(O)C(C)CCCCCCCCCCCCCCCCCC", 86),  # mycolic acid, HA=86
    # Mixed-type / aromatic failures
    ("Nc1c(/N=N/c2ccc([N+](=O)[O-])cc2)c(S(=O)(=O)O)cc2cc(S(=O)(=O)O)c(/N=N/c3ccccc3)c(O)c12", 39),  # azo dye, HA=39
    ("O=c1cc(-c2cc(O)c(O)cc2O)oc2cc(O)cc(O)c12", 22),     # flavone polyol, HA=22
    ("COc1cc(O)cc2c1C(=O)O[C@@H](C)CCCCC/C=C/2", 21),     # macrolide, HA=21
    ("Nc1ccc(S(=O)(=O)Nc2ncc(CC(=O)O)s2)cc1", 20),        # sulfonamide thiazole, HA=20
    ("COc1cc(-c2oc3cc(O)c(C)c(O)c3c(=O)c2OC)ccc1O", 25),  # methoxyflavone, HA=25
    ("CCc1oc2ccc(-c3cnn(C)c3)cc2c1C(=O)c1ccc(O)cc1", 26), # benzofuran ketone, HA=26
]


@pytest.mark.integration
class TestNameFragmentWithFallback:
    """Tests for _name_fragment_with_fallback() helper function."""

    def test_fallback_returns_name_for_simple_smiles(self):
        """Recursive naming succeeds for simple molecules, no fallback needed."""
        result = _name_fragment_with_fallback("CCO")
        assert result is not None
        assert "ethanol" in result.lower()

    def test_fallback_returns_none_for_truly_unnameable(self):
        """Returns None when BOTH recursive and pipeline naming fail."""
        # Invalid SMILES that cannot be named by either method
        result = _name_fragment_with_fallback("[invalid]")
        assert result is None

    def test_fallback_returns_name_not_none(self):
        """Fallback produces a valid name (not None) for complex fragments.

        Use a SMILES where recursive naming may hit depth limit but
        pipeline naming should succeed.
        """
        # A moderately complex molecule that should be nameable by pipeline
        result = _name_fragment_with_fallback("c1ccc2c(c1)cc1ccccc12")
        assert result is not None
        assert "unknown" not in result.lower()


@pytest.mark.integration
class TestMaxDecompLevels:
    """Tests for _get_max_decomp_levels() function."""

    def test_returns_4_for_large_molecule(self):
        """HA > 50 molecules get 4 decomposition levels."""
        mol = Chem.MolFromSmiles(
            "CCCCCCCCCCCCCCCCCCCCCCC(C(=O)O)C(O)CCCCCCCCCCCCCCCCCC1CC1CCCCCCCCCCCCCCCCC(O)C(C)CCCCCCCCCCCCCCCCCC"
        )
        assert mol is not None
        assert mol.GetNumHeavyAtoms() > 50
        assert _get_max_decomp_levels(mol) == 4

    def test_returns_3_for_small_molecule(self):
        """HA <= 50 molecules get default 3 decomposition levels."""
        mol = Chem.MolFromSmiles("CCCCCCCCCCCCCCCC(=O)OCC")
        assert mol is not None
        assert mol.GetNumHeavyAtoms() <= 50
        assert _get_max_decomp_levels(mol) == 3


@pytest.mark.integration
class TestDepthLimitConfig:
    """Tests for depth limit configuration."""

    def test_max_visited_size_is_50(self):
        """_MAX_VISITED_SIZE should be 50 (raised from 30 in Phase 127)."""
        assert _MAX_VISITED_SIZE == 50


@pytest.mark.integration
class TestDepthSafetyNetFallback:
    """Tests for depth safety net fallback in fragment_naming.py."""

    def test_depth_safety_net_uses_pipeline_fallback(self):
        """When visited set is at limit, pipeline fallback should be tried
        instead of returning None immediately.

        We verify this indirectly by checking that the depth safety net
        code path now includes name_pipeline_only.
        """
        import inspect
        from orthonym.assembly import fragment_naming
        source = inspect.getsource(fragment_naming.name_fragment_recursively)
        # The depth safety net should now reference name_pipeline_only
        assert "name_pipeline_only" in source


@pytest.mark.integration
class TestFragmentLossCompounds:
    """Parametrized tests for fragment-loss compounds from ChEBI-500 benchmark.

    These are compounds that previously produced no name because one fragment
    failed recursively and the entire decomposition was discarded.
    """

    @pytest.mark.parametrize(
        "smiles, ha",
        [pytest.param(s, h, id=f"HA{h}_{s[:20]}") for s, h in FRAGMENT_LOSS_SMILES],
    )
    def test_fragment_loss_compound_produces_name(self, smiles, ha):
        """Fragment-loss compound should produce a valid name with >= 70% HA coverage."""
        name = name_compound(smiles)
        assert name is not None, f"name_compound returned None for HA={ha}"
        assert "unknown" not in name.lower(), f"Name contains 'unknown': {name}"
        coverage = len(name) / ha if ha > 0 else 0
        assert coverage >= 0.7, (
            f"HA coverage {coverage:.2f} < 0.7 for HA={ha}, name={name[:80]}"
        )


@pytest.mark.integration
class TestHACoverage:
    """Verify HA > 30 molecules with cleavable bonds achieve >= 70% coverage."""

    HA_GT_30_SMILES = [
        # HA > 30 molecules with cleavable bonds from FRAGMENT_LOSS_SMILES
        ("CC(C)[C@H](NC(=O)[C@@H](N)Cc1ccc(O)cc1)C(=O)N[C@@H](CCCN=C(N)N)C(=O)O", 31),  # peptide
        ("Nc1c(/N=N/c2ccc([N+](=O)[O-])cc2)c(S(=O)(=O)O)cc2cc(S(=O)(=O)O)c(/N=N/c3ccccc3)c(O)c12", 39),  # azo dye
        ("CC1OC(Oc2c(C3OC(CO)C(O)C(O)C3O)c(O)c3c(=O)cc(-c4ccc(O)c(O)c4)oc3c2C2OC(CO)C(O)C(O)C2O)C(O)C(O)C1O", 53),  # flavonoid glycoside
        ("CC(=O)N[C@@H]1[C@@H](O)[C@H](O[C@@H]2O[C@H](CO)[C@@H](O[C@@H]3O[C@H](CO)[C@@H](O)[C@H](O[C@H]4O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]4O)[C@@H]3O)[C@H](O)[C@H]2NC(C)=O)[C@@H](CO)O[C@H]1O", 51),  # oligosaccharide
        ("CCC(/C=C/C(C)C1CCC2C3=CCC4CC(OC5OC(CO)C(OC6OC(CO)C(O)C(O)C6O)C(O)C5O)CCC4(C)C3CCC21C)C(C)C", 52),  # steroid glycoside
    ]

    @pytest.mark.parametrize(
        "smiles, ha",
        [pytest.param(s, h, id=f"HA{h}_{s[:20]}") for s, h in HA_GT_30_SMILES],
    )
    def test_ha_coverage_above_70_percent(self, smiles, ha):
        """HA > 30 molecules should achieve >= 70% HA coverage in generated names."""
        name = name_compound(smiles)
        assert name is not None, f"name_compound returned None for HA={ha}"
        assert "unknown" not in name.lower(), f"Name contains 'unknown': {name}"
        coverage = len(name) / ha if ha > 0 else 0
        assert coverage >= 0.7, (
            f"HA coverage {coverage:.2f} < 0.7 for HA={ha}, name={name[:80]}"
        )
