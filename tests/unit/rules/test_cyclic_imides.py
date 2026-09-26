"""Tests for cyclic imides (IUPAC.

 (the Blue Book): "Cyclic imides are preferably named as heterocyclic
pseudoketones." The PIN is therefore the ring-DIONE, and the trivial imide
names are general-nomenclature only:
  * succinimide -> pyrrolidine-2,5-dione (PIN, the Blue Book)
  * glutarimide -> piperidine-2,6-dione (PIN, the 6-membered homologue)
So succinimide and glutarimide are DEMOTED to GENERAL_RETAINED_NAMES (
a phase, iupac_2013_pin_list.json pin:false rows) and the systematic dione is
built by rules/heterocycles.py (the ring_imide_suffix pseudoketone branch).

phthalimide (-> 1H-isoindole-1,3(2H)-dione, Task 5) and maleimide
(-> 1H-pyrrole-2,5-dione, the Blue Book; Task 12 fix a performance pass) are demoted too: the
mancude/fused added-hydrogen machinery of rules/partial_saturation
(name_cyclic_oxo_compound) builds their PINs.
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
        """: succinimide is general-only; the PIN retained lookup no
        longer serves it (the systematic PIN pyrrolidine-2,5-dione wins), but it
        stays reachable via GENERAL_RETAINED_NAMES / --trivial."""
        canonical = Chem.CanonSmiles("O=C1CCC(=O)N1")
        assert get_retained_name(canonical) is None
        assert get_general_retained_name(canonical) == "succinimide"

    def test_maleimide_retained_name_lookup(self):
        """Maleimide is demoted (PIN deny row): general nomenclature only."""
        # 2026-09-26 (pre-existing-failures plan, Task 12 fix a performance pass, item 4) change-asserted-value:
        # (the Blue Book) "Cyclic imides are preferably named as heterocyclic pseudoketones";
        # "1H-pyrrole-2,5-dione (PIN) pyrrole-2,5-dione" (:33843); (:24721) added indicated
        # hydrogen is not cited when the suffix pair only removes ring double bonds. OPSIN 2.9.0 RT exact.
        # Code-level mutation: without the deny row the lookup returns 'maleimide'.
        canonical = Chem.CanonSmiles("O=C1C=CC(=O)N1")
        assert get_retained_name(canonical) is None
        assert get_general_retained_name(canonical) == "maleimide"

    def test_glutarimide_demoted_to_general(self):
        """: glutarimide is general-only; PIN is piperidine-2,6-dione."""
        canonical = Chem.CanonSmiles("O=C1CCCC(=O)N1")
        assert get_retained_name(canonical) is None
        assert get_general_retained_name(canonical) == "glutarimide"

    def test_phthalimide_retained_name_lookup(self):
        """Phthalimide is now demoted (PIN deny row): the fused added-H PIN exists."""
        # 2026-09-25 (pre-existing-failures plan, Task 5) change-asserted-value:
        # R21: (the Blue Book) "Cyclic imides are preferably named as heterocyclic pseudoketones"; "2-phenyl-1H-isoindole-1,3(2H)-dione (PIN)... N-phenylphthalimide" (:33853); (:24689) added indicated hydrogen is preferred over hydro prefixes for PINs. Code-level mutation: without the deny row the lookup returns 'phthalimide'.
        canonical = Chem.CanonSmiles("O=C1NC(=O)c2ccccc21")
        assert get_retained_name(canonical) is None


@pytest.mark.unit
class TestCyclicImideNaming:
    """End-to-end naming tests for cyclic imides."""

    def test_succinimide(self):
        """: O=C1CCC(=O)N1 -> pyrrolidine-2,5-dione (PIN, the Blue Book)."""
        assert name_compound("O=C1CCC(=O)N1") == "pyrrolidine-2,5-dione"

    def test_maleimide(self):
        """O=C1C=CC(=O)N1 -> 1H-pyrrole-2,5-dione (PIN, the Blue Book)."""
        # 2026-09-26 (pre-existing-failures plan, Task 12 fix a performance pass, item 4) change-asserted-value:
        # (the Blue Book) "Cyclic imides are preferably named as heterocyclic pseudoketones";
        # "1H-pyrrole-2,5-dione (PIN) pyrrole-2,5-dione" (:33843); (:24721) added indicated
        # hydrogen is not cited when the suffix pair only removes ring double bonds. OPSIN 2.9.0 RT exact.
        assert name_compound("O=C1C=CC(=O)N1") == "1H-pyrrole-2,5-dione"

    def test_glutarimide(self):
        """: O=C1CCCC(=O)N1 -> piperidine-2,6-dione (PIN)."""
        assert name_compound("O=C1CCCC(=O)N1") == "piperidine-2,6-dione"

    def test_phthalimide(self):
        """Phthalimide -> the heterocyclic pseudoketone PIN."""
        # 2026-09-25 (pre-existing-failures plan, Task 5) change-asserted-value:
        # R21: (the Blue Book) "Cyclic imides are preferably named as heterocyclic pseudoketones"; "2-phenyl-1H-isoindole-1,3(2H)-dione (PIN)... N-phenylphthalimide" (:33853); (:24689) added indicated hydrogen is preferred over hydro prefixes for PINs. OPSIN RT exact.
        assert name_compound("O=C1NC(=O)c2ccccc21") == "1H-isoindole-1,3(2H)-dione"

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
        """Alternative SMILES input for maleimide -> the same dione PIN."""
        # 2026-09-26 (pre-existing-failures plan, Task 12 fix a performance pass, item 4) change-asserted-value:
        # (the Blue Book) "Cyclic imides are preferably named as heterocyclic pseudoketones";
        # "1H-pyrrole-2,5-dione (PIN) pyrrole-2,5-dione" (:33843); (:24721) added indicated
        # hydrogen is not cited when the suffix pair only removes ring double bonds. OPSIN 2.9.0 RT exact.
        assert name_compound("C1=CC(=O)NC1=O") == "1H-pyrrole-2,5-dione"

    @pytest.mark.parametrize("smiles,expected", [
        ("CN1C(=O)C=CC1=O", "1-methyl-1H-pyrrole-2,5-dione"),
        ("ON1C(=O)C=CC1=O", "1-hydroxy-1H-pyrrole-2,5-dione"),
        ("CC1=CC(=O)NC1=O", "3-methyl-1H-pyrrole-2,5-dione"),
        ("O=C1C(Cl)=C(Cl)C(=O)N1", "3,4-dichloro-1H-pyrrole-2,5-dione"),
        ("O=C1Nc2ccccc2C1=O", "1H-indole-2,3-dione"),
        # fix a performance pass (wp5): a PENDANT ring no longer sends the ring ketone to the
        # fused catalog (partial_saturation.name_cyclic_oxo_compound resolves a
        # single-ring parent as a monocycle), and an N-alkyl on a lactam N no longer
        # counts as a senior group outside the ring. Were '1-phenyl-2,5-dihydro-1H-
        # pyrrole-2,5-dione', '1-benzyl-2,5-dihydro-1H-pyrrole-2,5-dione', '1-methyl-2,3-
        # dihydro-1H-indole-2,3-dione', '1-benzyl-2,3-dihydro-1H-indole-2,3-dione'.
        ("O=C1C=CC(=O)N1c1ccccc1", "1-phenyl-1H-pyrrole-2,5-dione"),
        ("O=C1C=CC(=O)N1Cc1ccccc1", "1-benzyl-1H-pyrrole-2,5-dione"),
        ("CN1C(=O)C(=O)c2ccccc21", "1-methyl-1H-indole-2,3-dione"),
        ("O=C1C(=O)N(Cc2ccccc2)c2ccccc21", "1-benzyl-1H-indole-2,3-dione"),
    ])
    def test_n_indicated_h_dione_keeps_the_parent_indicated_hydrogen(self, smiles, expected):
        """ (the Blue Book): "'Added indicated hydrogen' atoms
        are not cited when the accommodation of a pair of principal
        characteristic groups or free valences simply removes a double bond
        (directly or after rearrangement of double bonds) from the parent ring
        structure." BB (PIN) rows: '1-hydroxy-1H-pyrrole-2,5-dione' (:29597),
        '1H-pyrrole-2,5-dione' (:33843). These names were
        '...-2,5-dihydro-1H-pyrrole-2,5-dione' and '3H-indole-2,3(1H)-dione'
        before fix a performance pass. OPSIN 2.9.0 full-InChIKey round trip: exact."""
        from tests.support.rt_assert import assert_full_rt
        name = name_compound(smiles)
        assert name == expected
        assert_full_rt(name, smiles)


    @pytest.mark.parametrize("smiles,expected", [
        ("O=c1ccn(Cc2ccccc2)c(=O)[nH]1", "1-benzylpyrimidine-2,4(1H,3H)-dione"),
        ("O=c1ccn(C2CCCO2)c(=O)[nH]1", "1-(oxolan-2-yl)pyrimidine-2,4(1H,3H)-dione"),
        ("CO[C@@H]1[C@H](O)[C@@H](CO)O[C@H]1n1ccc(=O)[nH]c1=O",
         "1-[(2R,3R,4R,5R)-4-hydroxy-5-(hydroxymethyl)-3-methoxyoxolan-2-yl]"
         "pyrimidine-2,4(1H,3H)-dione"),
        ("c1ccc(-c2ccc(=O)[nH]c2)cc1", "5-phenylpyridin-2(1H)-one"),
    ])
    def test_pendant_ring_keeps_the_added_hydrogen_form(self, smiles, expected):
        """A ring ketone with a pendant ring takes the same added-hydrogen name as its
        alkyl analogue ('1-methylpyrimidine-2,4(1H,3H)-dione'). Pyrimidine is a
        retained parent ('pyrimidine (PIN)',, the Blue Book), so the
        Hantzsch-Widman '1H-1,3-diazine-2,4-dione' it used to get is not a PIN; the
        added hydrogen follows: 'pyrimidine-4,6(1H,5H)-dione (PIN)'
        (:24709), 'pyridin-2(1H)-one (PIN)' (:21316). Were '1-benzyl-1H-1,3-diazine-
        2,4-dione', '1-(oxolan-2-yl)-1H-1,3-diazine-2,4-dione', the rest-of-suite s37
        name '...-1H-1,3-diazine-2,4-dione' and '5-phenyl-1H-pyridin-2-one'. OPSIN
        2.9.0 full-InChIKey round trip: exact."""
        from tests.support.rt_assert import assert_full_rt
        name = name_compound(smiles)
        assert name == expected
        assert_full_rt(name, smiles)


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
