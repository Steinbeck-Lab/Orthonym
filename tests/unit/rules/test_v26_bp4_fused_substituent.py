"""v26 BP-4 Phase 1 — substituent support for the algorithmic 2-component
ortho-fused mancude heterocycle path.

Root cause fixed: ``_try_algorithmic_fusion_name`` previously hard-refused the
moment any exocyclic heavy atom was present, so a 2-component ortho-fused
mancude heterocycle whose core is NOT in the retained catalog lost its name as
soon as it carried a substituent. The fix discovers substituents against the
deterministic P-25.3.3 peripheral numbering (``compute_fused_numbering``),
applies the P-59.2.3 lowest-substituent-locant tie-break over the ring-system
automorphisms, and fails closed at source when any exocyclic branch is
unnameable.

Blue Book grounding (BlueBookV2/BlueBookV2.md):
- P-25.3.3.1.2(a)/(b): ring numbering fixed by heteroatoms (set, then element
  order O > S > Se > Te > N ...); substituents are NOT in that list.
- P-59.2.3.1 / line 25503: when a choice remains, low locants to detachable
  prefixes, then alphanumerical — the symmetric-parent tie-break.

These assert the NAME contract directly (via ``_try_algorithmic_fusion_name``)
rather than through the full namer's SELF-01 gate, which fails OPEN under OPSIN
subprocess contention (see NEXT-SESSION hazards). Each target PIN was verified
this session to OPSIN-round-trip to the input SMILES via ``.
"""
import pytest
from rdkit import Chem

from orthonym.rules.fused_rings import (
    _try_algorithmic_fusion_name,
    _try_polycomponent_fusion_name,
    _exocyclic_atoms_accounted,
)


def _ring_atoms(mol):
    atoms = set()
    for r in mol.GetRingInfo().AtomRings():
        atoms.update(r)
    return atoms


class TestBP4Phase1Targets:
    """The four blueprint targets now emit the expected PIN (were `unknown`)."""

    @pytest.mark.parametrize("smiles,expected", [
        # Symmetric parent: 2- and 5- are equivalent; PIN takes the lower (2).
        ("Cc1cc2ccoc2o1", "2-methylfuro[2,3-b]furan"),
        ("Cc1cc2occc2o1", "2-methylfuro[3,2-b]furan"),
        # Asymmetric parent (O < S): numbering fixed, no substituent freedom;
        # the methyl sits on the S-ring carbon = locant 5.
        ("Cc1cc2ccoc2s1", "5-methylthieno[2,3-b]furan"),
        # Halogen detachable prefix on the symmetric parent -> lowest locant 2.
        ("Clc1cc2ccoc2o1", "2-chlorofuro[2,3-b]furan"),
    ])
    def test_target_pin(self, smiles, expected):
        mol = Chem.MolFromSmiles(smiles)
        assert _try_algorithmic_fusion_name(mol) == expected


class TestBP4Phase1Generalizes:
    """The fix is a whole-class root-cause fix, not the four examples."""

    @pytest.mark.parametrize("smiles,expected", [
        ("Cc1cc2cc(C)oc2o1", "2,5-dimethylfuro[2,3-b]furan"),   # di-substituted
        ("CCc1cc2ccoc2o1", "2-ethylfuro[2,3-b]furan"),          # ethyl
        ("Cc1cc2ccoc2[se]1", "5-methylselenopheno[2,3-b]furan"),  # selenium ring
        ("Fc1cc2ccsc2o1", "2-fluorothieno[2,3-b]furan"),        # halogen, asym
        ("Nc1cc2ccoc2o1", "furo[2,3-b]furan-2-amine"),          # amino -> suffix
    ])
    def test_generalized_pin(self, smiles, expected):
        mol = Chem.MolFromSmiles(smiles)
        assert _try_algorithmic_fusion_name(mol) == expected

    def test_input_order_independent(self):
        """Two SMILES writings of the same molecule give the same PIN
        (the tie-break uses a SMILES-order-independent canonical signature)."""
        a = _try_algorithmic_fusion_name(Chem.MolFromSmiles("Cc1cc2occc2o1"))
        b = _try_algorithmic_fusion_name(Chem.MolFromSmiles("Cc1oc2c(c1)occ2"))
        assert a == b == "2-methylfuro[3,2-b]furan"


class TestBP4Phase1BaselinesUnchanged:
    """Bare systems keep the legacy path byte-for-byte; catalog cores still
    short-circuit before the algorithmic path (regression guard)."""

    @pytest.mark.parametrize("smiles,expected", [
        ("c1cc2ccoc2o1", "furo[2,3-b]furan"),
        ("c1cc2ccoc2s1", "thieno[2,3-b]furan"),
        ("c1cc2occc2o1", "furo[3,2-b]furan"),
    ])
    def test_bare_unchanged(self, smiles, expected):
        mol = Chem.MolFromSmiles(smiles)
        assert _try_algorithmic_fusion_name(mol) == expected


class TestBP4Phase1FailClosed:
    """Critic-required source-level completeness check: never emit a name that
    silently omits a substituent (that would denote a DIFFERENT molecule)."""

    @pytest.mark.parametrize("smiles", [
        "[Si](C)(C)c1cc2ccoc2o1",  # silyl: exotic element, unnameable branch
        "[B](O)(O)c1cc2ccoc2o1",   # boronic acid: exotic element B
    ])
    def test_unnameable_substituent_fails_closed(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert _exocyclic_atoms_accounted(mol, _ring_atoms(mol)) is False
        assert _try_algorithmic_fusion_name(mol) is None

    def test_accounted_true_for_simple_substituent(self):
        mol = Chem.MolFromSmiles("Cc1cc2ccoc2o1")
        assert _exocyclic_atoms_accounted(mol, _ring_atoms(mol)) is True

    def test_accounted_true_for_bare_system(self):
        mol = Chem.MolFromSmiles("c1cc2ccoc2o1")
        assert _exocyclic_atoms_accounted(mol, _ring_atoms(mol)) is True


class TestBP4Phase2Polycomponent:
    """Phase 2: substituent support for the 3+-component cata-fused star class
    (single heteroring base + >=2 monocyclic children), reusing the identical
    numbering + assembler machinery. Each PIN verified this session to OPSIN
    round-trip via """

    @pytest.mark.parametrize("smiles,expected", [
        ("Cc1cc2nc3ccoc3cc2o1", "2-methyldifuro[3,2-b:2',3'-e]pyridine"),
        ("Cc1coc2cc3occc3nc12", "3-methyldifuro[3,2-b:2',3'-e]pyridine"),
        ("Cc1c2occc2nc2ccoc12", "8-methyldifuro[3,2-b:2',3'-e]pyridine"),
        ("Clc1cc2nc3ccoc3cc2o1", "2-chlorodifuro[3,2-b:2',3'-e]pyridine"),
    ])
    def test_substituted_polycomponent_pin(self, smiles, expected):
        mol = Chem.MolFromSmiles(smiles)
        assert _try_polycomponent_fusion_name(mol) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("O1C=CC2=C1C=C1C(=N2)C=CO1", "difuro[3,2-b:2',3'-e]pyridine"),
        ("O1C=CC2=NC3=C(C=C21)SC=C3", "furo[3,2-b]thieno[2,3-e]pyridine"),
    ])
    def test_bare_polycomponent_unchanged(self, smiles, expected):
        mol = Chem.MolFromSmiles(smiles)
        assert _try_polycomponent_fusion_name(mol) == expected

    def test_unnameable_substituent_fails_closed(self):
        mol = Chem.MolFromSmiles("[Si](C)(C)c1cc2nc3ccoc3cc2o1")
        assert mol is not None
        assert _try_polycomponent_fusion_name(mol) is None
