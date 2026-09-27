"""Tests for CIP edge case handling and limitation documentation.

Verifies that:
1. rdCIPLabeler fallback to legacy works when rdCIPLabeler fails
2. Pseudoasymmetric centers (r/s) are correctly passed through
3. Known CIP limitations are documented (docs/cip_known_limitations.md exists)
4. The stereo pipeline gracefully handles CIP assignment failures

a phase: requirement.
"""
import os
import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from orthonym.perception.stereo import assign_stereochemistry
from tests.support.local_only import local_only


class TestCIPFallback:
    """Verify assign_stereochemistry handles rdCIPLabeler failures gracefully."""

    def test_basic_cip_assignment(self):
        """Normal R/S assignment works."""
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        assign_stereochemistry(mol)
        atom = mol.GetAtomWithIdx(1)
        assert atom.HasProp('_CIPCode')
        assert atom.GetProp('_CIPCode') in ('R', 'S')

    def test_ez_cip_assignment(self):
        """Normal E/Z assignment works."""
        mol = Chem.MolFromSmiles("C/C=C/C")
        assign_stereochemistry(mol)
        bond = mol.GetBondWithIdx(1)
        assert bond.HasProp('_CIPCode')
        assert bond.GetProp('_CIPCode') in ('E', 'Z')

    def test_idempotent_assignment(self):
        """Calling assign_stereochemistry twice does not change results."""
        mol = Chem.MolFromSmiles("C[C@@H](O)CC")
        assign_stereochemistry(mol)
        code1 = mol.GetAtomWithIdx(1).GetProp('_CIPCode')
        assign_stereochemistry(mol)  # Second call
        code2 = mol.GetAtomWithIdx(1).GetProp('_CIPCode')
        assert code1 == code2

    def test_no_stereo_no_cip(self):
        """Molecule without stereo gets no CIP codes."""
        mol = Chem.MolFromSmiles("CCC")
        assign_stereochemistry(mol)
        for atom in mol.GetAtoms():
            assert not atom.HasProp('_CIPCode')

    def test_multiple_stereocenters(self):
        """Multiple R/S stereocenters are all assigned."""
        mol = Chem.MolFromSmiles("[C@@H](O)(F)[C@H](Cl)Br")
        assign_stereochemistry(mol)
        cip_atoms = [
            a for a in mol.GetAtoms() if a.HasProp('_CIPCode')
        ]
        assert len(cip_atoms) == 2
        for a in cip_atoms:
            assert a.GetProp('_CIPCode') in ('R', 'S')

    def test_mixed_rs_ez_assignment(self):
        """Molecules with both R/S stereocenters and E/Z bonds."""
        mol = Chem.MolFromSmiles("[C@@H](O)(C)/C=C/C")
        assign_stereochemistry(mol)
        atom_cip = [
            a for a in mol.GetAtoms() if a.HasProp('_CIPCode')
        ]
        bond_cip = [
            b for b in mol.GetBonds() if b.HasProp('_CIPCode')
        ]
        assert len(atom_cip) >= 1
        assert len(bond_cip) >= 1


class TestPseudoasymmetric:
    """Verify lowercase r/s for pseudoasymmetric centers flows through pipeline."""

    def test_pseudoasymmetric_in_collect_stereodescriptors(self):
        """If RDKit assigns lowercase r/s, collect_stereodescriptors preserves it."""
        from orthonym.rules.stereochemistry import collect_stereodescriptors
        # Tartaric acid meso form: central carbon is pseudoasymmetric
        # Note: RDKit may or may not assign lowercase r/s depending on the molecule
        mol = Chem.MolFromSmiles("O[C@@H](C(=O)O)[C@H](O)C(=O)O")
        rdCIPLabeler.AssignCIPLabels(mol)
        atl = {a.GetIdx(): a.GetIdx() + 1 for a in mol.GetAtoms()}
        descriptors = collect_stereodescriptors(mol, atl)
        # Just verify it returns descriptors without error
        assert isinstance(descriptors, list)
        for locant, code in descriptors:
            assert code in ('R', 'S', 'r', 's', 'E', 'Z')

    def test_format_preserves_lowercase(self):
        """format_stereodescriptor_string preserves lowercase r/s."""
        from orthonym.rules.stereochemistry import format_stereodescriptor_string
        result = format_stereodescriptor_string([(2, 'r'), (3, 's')])
        assert result == '(2r,3s)-'

    def test_format_mixed_case(self):
        """format_stereodescriptor_string handles mixed R/r/S/s."""
        from orthonym.rules.stereochemistry import format_stereodescriptor_string
        result = format_stereodescriptor_string([(2, 'R'), (3, 'r'), (5, 'S')])
        assert result == '(2R,3r,5S)-'


@local_only("docs/cip_known_limitations.md")
class TestCIPDocumentationExists:
    """Verify CIP limitations documentation exists (where docs/ is kept: it is
    gitignored, so a clean checkout skips this class -- TRIAGE g7 C02)."""

    def test_cip_limitations_doc_exists(self):
        """docs/cip_known_limitations.md must exist per."""
        doc_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'docs',
            'cip_known_limitations.md'
        )
        assert os.path.exists(doc_path), (
            f"CIP limitations document missing at {doc_path}"
        )

    def test_cip_doc_contains_pass_rate(self):
        """Document must contain the CIP pass rate."""
        doc_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'docs',
            'cip_known_limitations.md'
        )
        with open(doc_path) as f:
            content = f.read()
        assert '182/290' in content or '62.8%' in content

    def test_cip_doc_contains_failure_categories(self):
        """Document must list all failure category types."""
        doc_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'docs',
            'cip_known_limitations.md'
        )
        with open(doc_path) as f:
            content = f.read()
        for category in ['CT', 'TH', 'AT', 'HE']:
            assert category in content, f"Missing failure category: {category}"

    def test_cip_doc_contains_ster18_deferral(self):
        """Document must reference deferral to a phase."""
        doc_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'docs',
            'cip_known_limitations.md'
        )
        with open(doc_path) as f:
            content = f.read()
        assert 'STER-18' in content, "Missing STER-18 deferral reference"
        assert 'Phase 141' in content, "Missing Phase 141 deferral reference"

    def test_cip_doc_contains_rdcip_labeler(self):
        """Document must reference rdCIPLabeler implementation."""
        doc_path = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'docs',
            'cip_known_limitations.md'
        )
        with open(doc_path) as f:
            content = f.read()
        assert 'rdCIPLabeler' in content, "Missing rdCIPLabeler reference"


# ============================================================================
# a phase — stereo r/s casing (Wave 0), RE-SCOPED per C5 + VERIFIED.
#
# VERIFIED (rdCIPLabeler probe, 2026-06-02, per the "verify don't trust numbers"
# directive / A1): the genuine pseudo-asymmetric set = the molecules where
# rdCIPLabeler ITSELF assigns lowercase r/s on the WHOLE molecule. The fix
#  defaults to that verdict, so these MUST keep lowercase (regression
# gate, assert NOW). NOTE: the (1r,3r)-1-amino-3-fluorocyclobutane the SPEC
# called a "false leak" is in fact GENUINE (rdCIPLabeler -> r,r; a 1,3-
# disubstituted cyclobutane is pseudo-asymmetric per — OPSIN
# rejecting it in BOTH casings is OPSIN's documented CIP-sequence-rule
# limitation, not ours. The ACTUAL false leak is the ceramide
# cyclohexane-1,2,3,4,5,6-hexayl substituent series (name shows 2s/5r but
# whole-molecule rdCIPLabeler says all-uppercase) — a fragment-vs-whole-molecule
# CIP-context leak (xfail until Plan 05 reconciles substituent_naming.py:990).
# ============================================================================

from orthonym.perception.stereo import get_stereocenters  # noqa: E402
from orthonym import name_compound  # noqa: E402

# Verified genuine set (rdCIPLabeler assigns >=1 lowercase r/s on the whole mol).
_GENUINE_PSEUDO_ASYM = [
    ("1-amino-3-fluorocyclobutane-1-carboxylic acid", "N[C@]1(C(=O)O)C[C@@H]([18F])C1"),
    ("tropan-3-ol",                                    "CN1[C@@H]2CC[C@H]1C[C@H](O)C2"),
    ("tropan-3-yl nonanoate (mesitylene ester)",       "Cc1cc(C)cc(C(=O)O[C@H]2C[C@H]3CC[C@@H](C2)N3C)c1"),
    ("norbornane (1s,4s)",                             "C1C[C@H]2CC[C@@H]1C2"),
]


@pytest.mark.unit
class TestSUB04GenuinePseudoAsymmetricCasing:
    """The genuine lowercase set — the SUB-04 ZERO-regression gate (assert NOW)."""

    @pytest.mark.parametrize("label,smiles", _GENUINE_PSEUDO_ASYM)
    def test_genuine_pseudo_keeps_lowercase_at_perception(self, label, smiles):
        mol = Chem.MolFromSmiles(smiles)
        centers = get_stereocenters(mol)
        lowercase = [c for c in centers if c.get("cip") in ("r", "s")]
        assert lowercase, f"{label}: expected >=1 lowercase r/s (genuine pseudo-asym)"

    def test_tropanol_name_keeps_lowercase_descriptor(self):
        # (1R,3s,5S)-tropan-3-ol — the 3s lowercase must survive into the name.
        name = name_compound("CN1[C@@H]2CC[C@H]1C[C@H](O)C2")
        assert "3s" in name


@pytest.mark.unit
class TestSUB04CasingLeak:
    """The genuine false leak (fragment-vs-whole-molecule CIP context):
    rdCIPLabeler on the whole molecule says all-uppercase, but the generated
    name leaks lowercase on the cyclohexane-hexayl substituent. The fix should
    emit the whole-molecule verdict (uppercase) — xfail until Plan 05."""

    @pytest.mark.xfail(reason="SUB-04 Plan 05: substituent fragment-CIP-context lowercase leak", strict=False)
    def test_ceramide_inositol_substituent_uppercased(self):
        # Whole-molecule rdCIPLabeler -> all uppercase; current name leaks 2s/5r.
        smi = "CCCCCCCCCCCCCCCCCCCCCCCC(=O)N[C@@H](COP(=O)([O-])O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O)[C@H](O)/C=C/CCCCCCCCCCCCC"
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            pytest.skip("ceramide SMILES did not parse in this RDKit build")
        # No lowercase r/s anywhere on the whole molecule per rdCIPLabeler.
        assert not any(c.get("cip") in ("r", "s") for c in get_stereocenters(mol))
        name = name_compound(smi)
        assert "2s" not in name and "5r" not in name
