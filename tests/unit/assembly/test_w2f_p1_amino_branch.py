"""W2F-P1 Tasks 1-3 — P-35.4.1 decorated N-branch amino prefixes on CHAIN parents.

BB P-35.4.1 (the Blue Book): "-NH-CH2Cl (chloromethyl)amino (preferred
prefix)". The ring-parent path already implements this (rules/benzene.py:1421,
gold W2E-P1FC-10); the chain-parent path has TWO carbon-count-only sites in
assembly/composer.py (_check_for_acylamino no-carbonyl fallback and
_name_n_attached_substituent_fallback) that silently drop the decoration
('(methylamino)' for -NH-CH2Cl = a DIFFERENT molecule) and are then
SELF-01-suppressed to 'unknown organic compound'.

Every expected name is OPSIN-2.9-verified (RDKit-canonical round-trip MATCH)
in internal notes §1.D.
"""
import pytest
from rdkit import Chem

from orthonym.namer import name_compound

UNKNOWN = "unknown organic compound"


@pytest.fixture
def gated(monkeypatch):
    """Re-enable the production OPSIN-validity + SELF-01 gate for the
    end-to-end fail-closed rows (research §1.C/§1.D design the 'unknown'
    end state via the SELF-01/validity backstop: helper-None on the
    DECORATED path plus the constitutional gate on the non-decorated
    fallback paths — which are pre-existing/out-of-scope here). The autouse
    conftest fixture disables the gate for speed, so those boundary molecules
    surface their pre-suppression wrong names; re-enabling exercises the true
    production behavior. Mirrors tests/unit/namer/test_self_consistency_gate.py.
    Skips if the OPSIN jar is unavailable (portable)."""
    import orthonym.namer as _nm
    if not _nm._validity_gate_jar_present():
        pytest.skip("OPSIN jar unavailable for gate-inclusive fail-closed test")
    monkeypatch.setattr(_nm, "_DISABLE_VALIDITY_GATE", False, raising=False)
    monkeypatch.setattr(_nm, "_SC_MODE", "on", raising=False)
    yield


@pytest.mark.unit
class TestDecoratedAminoChainParent:
    """Task 1: single decorated N-branch, end-to-end (site 1 emits)."""

    def test_curated_target_chloromethyl_amino(self):
        assert name_compound("ClCNCCCCCCCC(=O)O") == \
            "8-[(chloromethyl)amino]octanoic acid"

    def test_hydroxymethyl_amino_structure_driven_parens(self):
        # is_complex_substituent('hydroxymethyl') is False (research §1.C
        # trap): inner parens must be STRUCTURE-driven (non-C heavy atom in
        # the branch), never is_complex-driven.
        assert name_compound("OCNCCCCCCCC(=O)O") == \
            "8-[(hydroxymethyl)amino]octanoic acid"

    def test_located_decoration_1_chloroethyl(self):
        assert name_compound("CC(Cl)NCCCCCCCC(=O)O") == \
            "8-[(1-chloroethyl)amino]octanoic acid"

    def test_locant_bearing_branch_2_hydroxyethyl(self):
        assert name_compound("OCCNCCCCCCCC(=O)O") == \
            "8-[(2-hydroxyethyl)amino]octanoic acid"


@pytest.mark.unit
class TestHelperContracts:
    """Task 1: the three new composer helpers, unit level."""

    def test_branch_name_raw_no_marks(self):
        from orthonym.assembly.composer import _name_decorated_amino_branch
        mol = Chem.MolFromSmiles("ClCNCCCCCCCC(=O)O")
        # SMILES atom order: 0=Cl 1=CH2 2=N 3..10=chain C 11,12=O
        assert _name_decorated_amino_branch(mol, 1, 2, set(range(3, 11))) == \
            "chloromethyl"

    def test_branch_name_locant_anchored_at_free_valence(self):
        from orthonym.assembly.composer import _name_decorated_amino_branch
        mol = Chem.MolFromSmiles("CC(Cl)NCCCCCCCC(=O)O")
        # 0=CH3 1=CH 2=Cl 3=N 4..11=chain C
        assert _name_decorated_amino_branch(mol, 1, 3, set(range(4, 12))) == \
            "1-chloroethyl"

    def test_ring_branch_declines(self):
        from orthonym.assembly.composer import _name_decorated_amino_branch
        mol = Chem.MolFromSmiles("c1ccccc1CNCCCCCCCC(=O)O")
        # 0-5=ring 6=CH2 7=N 8..15=chain C — ring guard (benzene.py:1440
        # rationale): parent-hydride competition belongs to parent selection.
        assert _name_decorated_amino_branch(mol, 6, 7, set(range(8, 16))) is None

    def test_space_garbage_declines(self):
        from orthonym.assembly.composer import _name_decorated_amino_branch
        mol = Chem.MolFromSmiles("OB(O)CNCCCCCCCC(=O)O")
        # 0=O 1=B 2=O 3=CH2 4=N 5..12=chain C — producer emits
        # 'methylboronic acidyl' (space) for -CH2-B(OH)2; space-guard refuses.
        assert _name_decorated_amino_branch(mol, 3, 4, set(range(5, 13))) is None

    def test_assembly_single_decorated(self):
        from orthonym.assembly.composer import _assemble_decorated_amino_prefix
        assert _assemble_decorated_amino_prefix([("chloromethyl", True)]) == \
            "[(chloromethyl)amino]"

    def test_assembly_bis_identical_decorated(self):
        # BB 40703 precedent: 'bis(chloromethyl)aminoxyl (PIN)'
        from orthonym.assembly.composer import _assemble_decorated_amino_prefix
        assert _assemble_decorated_amino_prefix(
            [("chloromethyl", True), ("chloromethyl", True)]
        ) == "[bis(chloromethyl)amino]"

    def test_assembly_mixed_alphanumerical_not_ascii(self):
        # '2-hydroxyethyl' sorts at 'h' (letters-only key), AFTER
        # 'chloromethyl' — raw sorted() would put '2-...' first (WRONG).
        from orthonym.assembly.composer import _assemble_decorated_amino_prefix
        assert _assemble_decorated_amino_prefix(
            [("2-hydroxyethyl", True), ("chloromethyl", True)]
        ) == "[(chloromethyl)(2-hydroxyethyl)amino]"


@pytest.mark.unit
class TestNNDecoratedAminoEndToEnd:
    """Task 2: N,N-di decorated branches, end-to-end."""

    def test_bis_identical_decorated(self):
        assert name_compound("ClCN(CCl)CCCCCCCC(=O)O") == \
            "8-[bis(chloromethyl)amino]octanoic acid"

    def test_mixed_decorated_alphanumerical(self):
        # citation c < h by the letters-only key (research §1.D)
        assert name_compound("ClCN(CCO)CCCCCCCC(=O)O") == \
            "8-[(chloromethyl)(2-hydroxyethyl)amino]octanoic acid"


@pytest.mark.unit
class TestSite2FallbackDirect:
    """Task 2: _name_n_attached_substituent_fallback, unit level."""

    def test_site2_decorated_single(self):
        from orthonym.assembly.composer import (
            _name_n_attached_substituent_fallback,
        )
        mol = Chem.MolFromSmiles("ClCNCCCCCCCC(=O)O")
        # sub = N(2) + CH2Cl branch {1, 0}; chain = 3..10; attach = the N
        assert _name_n_attached_substituent_fallback(
            mol, [2, 1, 0], {2, 1, 0}, set(range(3, 11)), 2
        ) == "[(chloromethyl)amino]"

    def test_site2_decorated_unnameable_fails_closed(self):
        from orthonym.assembly.composer import (
            _name_n_attached_substituent_fallback,
        )
        mol = Chem.MolFromSmiles("OB(O)CNCCCCCCCC(=O)O")
        # sub = N(4) + CH2-B(OH)2 branch {3, 1, 0, 2}; producer output is
        # garbled ('methylboronic acidyl') -> helper None -> site returns
        # None, NEVER the branch-dropping 'amino' NOR '(methylamino)'.
        assert _name_n_attached_substituent_fallback(
            mol, [4, 3, 1, 0, 2], {4, 3, 1, 0, 2}, set(range(5, 13)), 4
        ) is None

    def test_site2_pure_alkyl_byte_identical(self):
        from orthonym.assembly.composer import (
            _name_n_attached_substituent_fallback,
        )
        mol = Chem.MolFromSmiles("CNCCCCCCCC(=O)O")
        # 0=CH3 1=N 2..9=chain — legacy single-branch form preserved
        assert _name_n_attached_substituent_fallback(
            mol, [1, 0], {1, 0}, set(range(2, 10)), 1
        ) == "(methylamino)"

    def test_site2_nn_dimethyl_byte_identical(self):
        from orthonym.assembly.composer import (
            _name_n_attached_substituent_fallback,
        )
        mol = Chem.MolFromSmiles("CN(C)CCCCCCCC(=O)O")
        # 0=CH3 1=N 2=CH3 3..10=chain — legacy HYG-04 form preserved
        assert _name_n_attached_substituent_fallback(
            mol, [1, 0, 2], {1, 0, 2}, set(range(3, 11)), 1
        ) == "(dimethylamino)"


@pytest.mark.unit
@pytest.mark.usefixtures("gated")
class TestFailClosedBoundary:
    """Task 3: research §1.D fail-closed rows — refuse, never truncate.

    Gate-inclusive: the 'unknown'/'inorganic' end state is produced by the
    SELF-01/validity backstop (research §1.C/§1.D). Run with the production
    gate re-enabled (see the ``gated`` fixture) rather than the autouse
    gate-off default, so these assert TRUE production fail-closed behavior."""

    def test_boronic_branch_stays_unknown(self):
        assert name_compound("OB(O)CNCCCCCCCC(=O)O") == UNKNOWN

    def test_selanyl_branch_refuses_or_exact_heal(self):
        # HEAD producer emits structure-corrupt 'hydroxymethaneselenyl'
        # (no O in the branch!) — must refuse. Heal-optional acceptance:
        # the OPSIN-verified '8-[(selanylmethyl)amino]octanoic acid'.
        out = name_compound("[SeH]CNCCCCCCCC(=O)O")
        assert out in (UNKNOWN, "8-[(selanylmethyl)amino]octanoic acid")

    def test_arsanyl_branch_stays_refused(self):
        # refused upstream at dispatch (never reaches the new code) — pins
        # that name_substituent_fragment's atom-dropping 'propynyl' can
        # never surface (research §1.D row 3).
        # Phase B: the fallback label is now 'arsenic compound (not supported)'.
        # This molecule is a carbon-bearing ORGANOarsenic compound, so the old
        # 'inorganic' wording was factually wrong; 'As' was the only p-block
        # metalloid missing from errors._METAL_NAMES. The REFUSAL itself (the
        # point of this test -- 'propynyl' must never surface) is unchanged.
        out = name_compound("[AsH2]C#CCNCCCCCCCC(=O)O")
        assert out in (UNKNOWN, "arsenic compound (not supported)")


@pytest.mark.unit
class TestRegressionGuards:
    """Task 3: research §1.F gold-exposure rows — all HEAD-OK, must stay."""

    def test_ring_path_gold_w2e_p1fc_10(self):
        # rules/benzene.py path — file untouched by this plan
        assert name_compound("ClCNc1ccc(C(=O)O)cc1") == \
            "4-[(chloromethyl)amino]benzoic acid"

    def test_n_halo_golds_w2_npref(self):
        # N-halo/N-OH: no C branch -> the edited loops no-op (seniority path)
        assert name_compound("ClNCCCCCCCC(=O)O") == \
            "8-(chloroamino)octanoic acid"
        assert name_compound("FNCCCCCCCC(=O)O") == \
            "8-(fluoroamino)octanoic acid"
        assert name_compound("ONCCC(=O)O") == \
            "3-(hydroxyamino)propanoic acid"

    def test_dimethylamino_gold_w2c_d_am_04(self):
        # pure N,N-dimethyl path byte-identical (acylamino main path + site 2)
        assert name_compound("CCN=C(CCC(=O)OC)N(C)C") == \
            "methyl 4-(dimethylamino)-4-(ethylimino)butanoate"

    def test_carbamoylamino_gold_w2e_p1chainsa_09(self):
        # acylamino MAIN path (carbonyl found) — untouched by the edits
        assert name_compound("NC(=O)NCCCNC=O") == \
            "N-[3-(carbamoylamino)propyl]formamide"

    @pytest.mark.usefixtures("gated")
    def test_pure_branched_n_branch_stays_out_of_scope(self):
        # research §1.F latent-wrong adjacency: pure-carbon BRANCHED branch
        # (isopropyl) stays gated OUT of the helper; assert we did not start
        # emitting the linearized wrong name. Gate-inclusive: the legacy
        # pure-carbon path (unchanged by this plan) emits the wrong
        # '(propylamino)' which the SELF-01 backstop suppresses to 'unknown'
        # (A/B: byte-identical HEAD vs current — pre-existing, out of scope).
        out = name_compound("CC(C)NCCCCCCCC(=O)O")
        assert out != "8-(propylamino)octanoic acid"


@pytest.mark.unit
class TestMixedSimpleDecorated:
    """Task 3 decision rule (FIRST arm): the mixed simple+decorated surface
    is OPSIN-RT-verified at implementation time, so the grammar-derived name
    is kept (decorated always parenthesized; second-cited simple parenthesized
    per P-16.5.1.3.1). Not added to golds (not in the research's table)."""

    def test_mixed_simple_decorated_verified(self):
        # OPSIN-RT re-verified at implementation time (diagnose OK)
        assert name_compound("CN(CCl)CCCCCCCC(=O)O") == \
            "8-[(chloromethyl)(methyl)amino]octanoic acid"
