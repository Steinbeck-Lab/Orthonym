"""W2F-P1 Tasks 6, 8, 9 — α-substituted benzyl prefixes + the
/ citation-layer escalation both this class and the
asymmetric benzylic ether (Task 7) require.

BB (the Blue Book): 'bromo(4-methylphenyl)methyl (preferred
prefix)'. The enclosing marks are STRUCTURE-BEARING: OPSIN parses the
marks-dropped 'bromo(phenyl)methylbenzene' to a DIFFERENT molecule
(Brc1ccccc1Cc1ccccc1 — research A).

All expected names OPSIN-2.9-verified in
internal notes D (and D for Task-7 shapes).
"""
import pytest
from rdkit import Chem

from orthonym.namer import name_compound

UNKNOWN = "unknown organic compound"


@pytest.mark.unit
class TestCitationLayerEscalation:
    """Task 6: format_substituent_prefix must never cite a mark-bearing
    compound prefix bare, and must escalate (not double) the outer mark."""

    def test_interior_parens_escalate_to_brackets(self):
        from orthonym.assembly.naming_utils import format_substituent_prefix
        assert format_substituent_prefix("bromo(phenyl)methyl", [4], 1) == \
            "4-[bromo(phenyl)methyl]"

    def test_trailing_stem_after_brackets_escalates_to_braces(self):
        from orthonym.assembly.naming_utils import format_substituent_prefix
        assert format_substituent_prefix(
            "[(4-methoxyphenyl)methoxy]methyl", [4], 1
        ) == "4-{[(4-methoxyphenyl)methoxy]methyl}"

    def test_leading_paren_shape_unchanged(self):
        # pre-existing behavior (W2E-P1FC Task 8 branch) must be preserved
        from orthonym.assembly.naming_utils import format_substituent_prefix
        assert format_substituent_prefix("(benzylsulfanyl)methyl", [4], 1) == \
            "4-[(benzylsulfanyl)methyl]"

    def test_simple_and_complex_markless_shapes_unchanged(self):
        from orthonym.assembly.naming_utils import format_substituent_prefix
        assert format_substituent_prefix("methyl", [2, 2], 2) == "2,2-dimethyl"
        assert format_substituent_prefix("1-methylethyl", [4, 7], 2) == \
            "4,7-bis(1-methylethyl)"
        assert format_substituent_prefix("chloromethyl", [1], 1) == \
            "1-(chloromethyl)"

    def test_fusion_brackets_not_escalated(self):
        #: fusion brackets are nesting-IGNORED -> plain parens
        from orthonym.assembly.naming_utils import format_substituent_prefix
        assert format_substituent_prefix("furo[3,2-b]pyridin-2-yl", [4], 1) == \
            "4-(furo[3,2-b]pyridin-2-yl)"


@pytest.mark.unit
class TestAlphaHaloBenzylEndToEnd:
    """Task 8: research §3.D buildable rows (all OPSIN-verified)."""

    def test_curated_target(self):
        assert name_compound("Brc1ccc(C(Br)c2ccccc2)cc1") == \
            "1-bromo-4-[bromo(phenyl)methyl]benzene"

    def test_pcg_parent_benzoic_acid(self):
        #: the acid fixes the parent outright — no ring-choice question
        assert name_compound("OC(=O)c1ccc(C(Cl)c2ccccc2)cc1") == \
            "4-[chloro(phenyl)methyl]benzoic acid"

    def test_mixed_halogens_alpha_f_ring_br(self):
        # 'bromo' cited first on the parent (b < f)
        assert name_compound("FC(c1ccccc1)c1ccc(Br)cc1") == \
            "1-bromo-4-[fluoro(phenyl)methyl]benzene"

    def test_interleave_ring_yl_first_keeps_parens(self):
        # '(4-bromophenyl)' sorts as 'bromophenyl' < 'chloro' -> cited FIRST
        # but LOCANT-BEARING so it KEEPS its parens "unless it
        # includes a locant")
        assert name_compound("OC(=O)c1ccc(C(Cl)c2ccc(Br)cc2)cc1") == \
            "4-[(4-bromophenyl)(chloro)methyl]benzoic acid"

    def test_gem_dichloro_different_rings(self):
        # rings DIFFER -> substitutive IS the PIN; 'dichloro' sorts as
        # 'chloro' (multiplying prefix ignored for alphanumerical order)
        assert name_compound("Clc1ccc(C(Cl)(Cl)c2ccccc2)cc1") == \
            "1-chloro-4-[dichloro(phenyl)methyl]benzene"

    def test_gem_mixed_halogens(self):
        # first-cited 'chloro' bare; every further prefix parenthesized
        assert name_compound("Brc1ccc(C(F)(Cl)c2ccccc2)cc1") == \
            "1-bromo-4-[chloro(fluoro)(phenyl)methyl]benzene"


@pytest.mark.unit
class TestAlphaHaloBuilderDirect:
    """Task 8: producer level — the builder emits the BARE compound prefix
    (the Task-6 citation layer owns the outer bracket)."""

    def test_builder_emits_bare_compound_prefix(self):
        from orthonym.rules.ring_substituents import name_ring_system_substituent
        mol = Chem.MolFromSmiles("Brc1ccc(C(Br)c2ccccc2)cc1")
        # fragment seen from the Br-ring parent: CH(5) + Br(6) + phenyl(7-12)
        frag = [5, 6, 7, 8, 9, 10, 11, 12]
        assert name_ring_system_substituent(mol, frag, 5) == \
            "bromo(phenyl)methyl"

    def test_builder_undecorated_byte_identical(self):
        from orthonym.rules.ring_substituents import name_ring_system_substituent
        mol = Chem.MolFromSmiles("C(c1ccccc1)c1ccccn1")
        # retained-benzyl gate (:1294) untouched for bare-phenyl fragments
        assert name_ring_system_substituent(mol, [0, 1, 2, 3, 4, 5, 6], 0) == \
            "benzyl"


@pytest.mark.unit
class TestAlphaBenzylFailClosedAndGuards:
    """Task 9: research §3.D fail-closed + adjacent-class guard rows."""

    def test_alpha_nitro_stays_unknown_or_verified_heal(self):
        # nitro NOT in the v1 halogen whitelist (zwitterion-mask adjacency);
        # heal-optional: the OPSIN-verified nitro form (v1.5, out of scope).
        out = name_compound("Brc1ccc(C([N+](=O)[O-])c2ccccc2)cc1")
        assert out in (UNKNOWN, "1-bromo-4-[nitro(phenyl)methyl]benzene")

    def test_identical_units_gem_dichloro_stays_unknown(self):
        # THE leak hazard (research F): '[dichloro(phenyl)methyl]benzene'
        # is RT-valid but non-PIN (PIN = multiplicative
        # 1,1'-(dichloromethylene)dibenzene, not yet buildable). The
        # in-builder identical-units decline must keep this refused.
        assert name_compound("ClC(Cl)(c1ccccc1)c1ccccc1") == UNKNOWN

    def test_identical_units_sibling_multiplicative_untouched(self):
        #: identical units -> multiplicative@900 stays senior
        assert name_compound("c1ccccc1C(Br)c1ccccc1") == \
            "1,1'-(bromomethylene)dibenzene"

    def test_alpha_oh_stays_with_methanol_parent(self):
        # α-OH = PCG -> carbinol/methanol path owns it; the halogen
        # whitelist excludes O by construction. HEAD emits the RT-OK
        # '(4-bromophenyl)phenylmethanol' (pre-existing marks style defect,
        # research D — do NOT pin the exact marks, only the parent).
        out = name_compound("OC(c1ccccc1)c1ccc(Br)cc1")
        assert out.endswith("methanol")

    def test_halomethylene_bridge_gold_untouched(self):
        # gold W2C-MBRIDGE-02: identical PCG units + α-Cl on the BRIDGE ->
        # multiplicative@900; the identical-units decline ALSO protects it
        # if dispatch ever reorders.
        assert name_compound("Oc1ccc(C(Cl)c2ccc(O)cc2)cc1") == \
            "4,4'-(chloromethylene)diphenol"

    def test_retained_benzyl_untouched(self):
        # gold W2E-P1FC-04 neighbourhood (undecorated bare-phenyl fragment)
        assert name_compound("C(c1ccccc1)c1ccccn1") == "2-benzylpyridine"

    def test_ring_fold_path_untouched(self):
        # gold W2E-P1FC-05 shape: ring decoration folds BEFORE the new
        # decoration split — '(4-chlorophenyl)methyl' unchanged.
        assert name_compound("Clc1ccc(Cc2ccccn2)cc1") == \
            "2-[(4-chlorophenyl)methyl]pyridine"


@pytest.mark.unit
class TestAlphaBenzylDeterminism:
    """Task 9: research C det-probe protocol — evidence, sibling,
    gem-dichloro (d2 refusal), gem-mixed; 10+ random spellings each,
    byte-identical. All citation inputs are spelling-invariant strings
    (fixed halo-name map + letters-only key); any spread is a REAL bug."""

    @pytest.mark.parametrize("smi,expected", [
        ("Brc1ccc(C(Br)c2ccccc2)cc1",
         "1-bromo-4-[bromo(phenyl)methyl]benzene"),
        ("c1ccccc1C(Br)c1ccccc1", "1,1'-(bromomethylene)dibenzene"),
        ("ClC(Cl)(c1ccccc1)c1ccccc1", UNKNOWN),
        ("Brc1ccc(C(F)(Cl)c2ccccc2)cc1",
         "1-bromo-4-[chloro(fluoro)(phenyl)methyl]benzene"),
    ])
    def test_random_spellings_byte_identical(self, smi, expected):
        mol = Chem.MolFromSmiles(smi)
        outs = {
            name_compound(Chem.MolToSmiles(mol, doRandom=True, canonical=False))
            for _ in range(12)
        }
        assert outs == {expected}
