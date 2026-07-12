"""W2F-P1 Tasks 1-3 — P-35.4.1 decorated N-branch amino prefixes on CHAIN parents.

BB P-35.4.1 (BlueBookV2.md:18112): "-NH-CH2Cl (chloromethyl)amino (preferred
prefix)". The ring-parent path already implements this (rules/benzene.py:1421,
gold W2E-P1FC-10); the chain-parent path has TWO carbon-count-only sites in
assembly/composer.py (_check_for_acylamino no-carbonyl fallback and
_name_n_attached_substituent_fallback) that silently drop the decoration
('(methylamino)' for -NH-CH2Cl = a DIFFERENT molecule) and are then
SELF-01-suppressed to 'unknown organic compound'.

Every expected name is OPSIN-2.9-verified (RDKit-canonical round-trip MATCH)
in  §1.D.
"""
import pytest
from rdkit import Chem

from orthonym.namer import name_compound

UNKNOWN = "unknown organic compound"


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
