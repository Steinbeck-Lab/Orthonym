"""Tests for cyclic imides (IUPAC P-66.2.1).

P-66.2.1 (the Blue Book): "Cyclic imides are preferably named as heterocyclic
pseudoketones." The PIN is therefore the ring-DIONE, and the trivial imide
names are general-nomenclature only:
  * succinimide -> pyrrolidine-2,5-dione (PIN, the Blue Book)
  * glutarimide -> piperidine-2,6-dione (PIN, the 6-membered homologue)
So succinimide and glutarimide are DEMOTED to GENERAL_RETAINED_NAMES (
a phase, iupac_2013_pin_list.json pin:false rows) and the systematic dione is
built by rules/heterocycles.py (the ring_imide_suffix pseudoketone branch).

maleimide (-> 1H-pyrrole-2,5-dione, the Blue Book) and phthalimide
(-> 1H-isoindole-1,3(2H)-dione) are NOT yet demoted: their mancude/fused rings
need the added-hydrogen machinery the monocyclic branch does not build, so they
keep their retained names for now (a project rule: don't unmask a worse spelling).
"""

import pytest
from rdkit import Chem

from orthonym.namer import name_compound
from orthonym.data.retained_names import get_retained_name
from orthonym.data import get_general_retained_name
from orthonym.perception.functional_groups import detect_functional_groups


@pytest.mark.unit
class TestCyclicImideRetainedNames:
    """Retained name lookup tests for cyclic imides."""

    def test_succinimide_demoted_to_general(self):
        """P-66.2.1: succinimide is general-only; the PIN retained lookup no
        longer serves it (the systematic PIN pyrrolidine-2,5-dione wins), but it
        stays reachable via GENERAL_RETAINED_NAMES / --trivial."""
        canonical = Chem.CanonSmiles("O=C1CCC(=O)N1")
        assert get_retained_name(canonical) is None
        assert get_general_retained_name(canonical) == "succinimide"

    def test_maleimide_retained_name_lookup(self):
        """Maleimide is not yet demoted (mancude added-H deferred), so it is
        still served by the PIN retained lookup."""
        canonical = Chem.CanonSmiles("O=C1C=CC(=O)N1")
        assert get_retained_name(canonical) == "maleimide"

    def test_glutarimide_demoted_to_general(self):
        """P-66.2.1: glutarimide is general-only; PIN is piperidine-2,6-dione."""
        canonical = Chem.CanonSmiles("O=C1CCCC(=O)N1")
        assert get_retained_name(canonical) is None
        assert get_general_retained_name(canonical) == "glutarimide"

    def test_phthalimide_retained_name_lookup(self):
        """Phthalimide not yet demoted (fused isoindole added-H deferred)."""
        canonical = Chem.CanonSmiles("O=C1NC(=O)c2ccccc21")
        assert get_retained_name(canonical) == "phthalimide"


@pytest.mark.unit
class TestCyclicImideNaming:
    """End-to-end naming tests for cyclic imides."""

    def test_succinimide(self):
        """P-66.2.1: O=C1CCC(=O)N1 -> pyrrolidine-2,5-dione (PIN, the Blue Book)."""
        assert name_compound("O=C1CCC(=O)N1") == "pyrrolidine-2,5-dione"

    def test_maleimide(self):
        """O=C1C=CC(=O)N1 -> maleimide (dione PIN deferred, see module docstring)."""
        assert name_compound("O=C1C=CC(=O)N1") == "maleimide"

    def test_glutarimide(self):
        """P-66.2.1: O=C1CCCC(=O)N1 -> piperidine-2,6-dione (PIN)."""
        assert name_compound("O=C1CCCC(=O)N1") == "piperidine-2,6-dione"

    def test_phthalimide(self):
        """Phthalimide via retained name (isoindole-dione PIN deferred)."""
        assert name_compound("O=C1NC(=O)c2ccccc21") == "phthalimide"

    def test_n_methyl_succinimide_not_retained(self):
        """N-methylsuccinimide should NOT match the retained name.
        It has different canonical SMILES due to N-methyl substituent."""
        canonical = Chem.CanonSmiles("O=C1CCC(=O)N1C")
        # Should not return "succinimide" -- different SMILES
        result = get_retained_name(canonical)
        assert result != "succinimide"

    def test_n_methyl_succinimide_naming(self):
        """N-methylsuccinimide should get systematic name, not retained name."""
        result = name_compound("O=C1CCC(=O)N1C")
        # Should NOT be just "succinimide" -- must include the N-methyl
        assert result != "succinimide"
        # The exact systematic name may vary, but it should contain methyl
        assert "methyl" in result.lower()

    def test_succinimide_alternate_input(self):
        """Alternative SMILES input for succinimide -> systematic dione PIN."""
        # Various equivalent representations
        assert name_compound("C1CC(=O)NC1=O") == "pyrrolidine-2,5-dione"

    def test_maleimide_alternate_input(self):
        """Alternative SMILES input for maleimide should also work."""
        assert name_compound("C1=CC(=O)NC1=O") == "maleimide"


@pytest.mark.unit
class TestImidePerception:
    """Test that imide FG detection suppresses overlapping amide matches."""

    def test_imide_suppresses_secondary_amide(self):
        """When imide detected, overlapping secondary_amide should be suppressed."""
        mol = Chem.MolFromSmiles('O=C1CCC(=O)N1')  # succinimide
        fgs = detect_functional_groups(mol)
        assert 'imide' in fgs
        assert 'secondary_amide' not in fgs

    def test_imide_suppresses_tertiary_amide(self):
        """When N-substituted imide detected, tertiary_amide should be suppressed."""
        mol = Chem.MolFromSmiles('CN1C(=O)CCC1=O')  # N-methylsuccinimide
        fgs = detect_functional_groups(mol)
        assert 'imide' in fgs
        assert 'tertiary_amide' not in fgs

    def test_glutarimide_imide_detected(self):
        """Glutarimide should be detected as imide."""
        mol = Chem.MolFromSmiles('O=C1CCCC(=O)N1')
        fgs = detect_functional_groups(mol)
        assert 'imide' in fgs
        assert 'secondary_amide' not in fgs
