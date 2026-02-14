"""Unit tests for phosphorus compound naming."""

import pytest
from rdkit import Chem

from src.orthonym.rules.phosphorus import (
    name_phosphine,
    name_phosphine_oxide,
    name_phosphonic_acid,
    name_phosphinic_acid,
    name_phosphate_ester,
    get_phosphorus_prefix,
    get_phosphanyl_prefix,
    _count_alkyl_carbons,
    _characterize_substituent,
)


class TestPhosphineNaming:
    """Tests for phosphine (phosphane) naming."""

    def test_methylphosphane(self):
        """CP -> methylphosphane"""
        mol = Chem.MolFromSmiles("CP")
        result = name_phosphine(mol, 1)  # P is at index 1
        assert result == "methylphosphane"

    def test_ethylphosphane(self):
        """CCP -> ethylphosphane"""
        mol = Chem.MolFromSmiles("CCP")
        result = name_phosphine(mol, 2)  # P is at index 2
        assert result == "ethylphosphane"

    def test_dimethylphosphane(self):
        """CPC -> dimethylphosphane (secondary phosphine)"""
        mol = Chem.MolFromSmiles("CPC")
        result = name_phosphine(mol, 1)  # P is at index 1
        assert result == "dimethylphosphane"

    def test_trimethylphosphane(self):
        """CP(C)C -> trimethylphosphane"""
        mol = Chem.MolFromSmiles("CP(C)C")
        result = name_phosphine(mol, 1)  # P is at index 1
        assert result == "trimethylphosphane"

    def test_triethylphosphane(self):
        """CCP(CC)CC -> triethylphosphane"""
        mol = Chem.MolFromSmiles("CCP(CC)CC")
        result = name_phosphine(mol, 2)  # P is at index 2
        assert result == "triethylphosphane"

    def test_ethylmethylphosphane(self):
        """CPCC -> ethylmethylphosphane (asymmetric secondary)"""
        mol = Chem.MolFromSmiles("CPCC")  # methyl and ethyl on P
        result = name_phosphine(mol, 1)
        assert result == "ethylmethylphosphane"

    def test_dimethylethylphosphane(self):
        """CCP(C)C -> ethyldimethylphosphane (asymmetric tertiary, with multiplier)"""
        mol = Chem.MolFromSmiles("CCP(C)C")
        result = name_phosphine(mol, 2)
        # Alphabetical: ethyl + dimethyl (multiplier for identical groups)
        assert result == "ethyldimethylphosphane"

    def test_parent_phosphane(self):
        """Pure phosphane (PH3)"""
        mol = Chem.MolFromSmiles("[PH3]")
        result = name_phosphine(mol, 0)
        assert result == "phosphane"


class TestPhosphineOxideNaming:
    """Tests for phosphine oxide naming."""

    def test_trimethylphosphane_oxide(self):
        """CP(C)(C)=O -> trimethylphosphane oxide"""
        mol = Chem.MolFromSmiles("CP(C)(C)=O")
        # SMARTS match gives indices - find P
        p_idx = None
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() == 'P':
                p_idx = i
                break
        result = name_phosphine_oxide(mol, tuple(range(mol.GetNumAtoms())))
        assert result == "trimethylphosphane oxide"

    def test_triethylphosphane_oxide(self):
        """CCP(CC)(CC)=O -> triethylphosphane oxide"""
        mol = Chem.MolFromSmiles("CCP(CC)(CC)=O")
        result = name_phosphine_oxide(mol, tuple(range(mol.GetNumAtoms())))
        assert result == "triethylphosphane oxide"

    def test_dimethylethylphosphane_oxide(self):
        """CCP(C)(C)=O -> asymmetric phosphine oxide"""
        mol = Chem.MolFromSmiles("CCP(C)(C)=O")
        result = name_phosphine_oxide(mol, tuple(range(mol.GetNumAtoms())))
        # Two methyl + one ethyl: ethyldimethylphosphane oxide (with multiplier)
        assert "phosphane oxide" in result
        assert result == "ethyldimethylphosphane oxide"


class TestPhosphonicAcidNaming:
    """Tests for phosphonic acid naming."""

    def test_methanephosphonic_acid(self):
        """CP(=O)(O)O -> methanephosphonic acid"""
        mol = Chem.MolFromSmiles("CP(=O)(O)O")
        result = name_phosphonic_acid(mol, (1, 2, 3, 4), "methane")
        assert result == "methanephosphonic acid"

    def test_ethanephosphonic_acid(self):
        """CCP(=O)(O)O -> ethanephosphonic acid"""
        mol = Chem.MolFromSmiles("CCP(=O)(O)O")
        result = name_phosphonic_acid(mol, (2, 3, 4, 5), "ethane")
        assert result == "ethanephosphonic acid"

    def test_phenylphosphonic_acid(self):
        """c1ccccc1P(=O)(O)O -> phenylphosphonic acid (note: phenyl not benzene)"""
        mol = Chem.MolFromSmiles("c1ccccc1P(=O)(O)O")
        result = name_phosphonic_acid(mol, (6, 7, 8, 9), "phenyl")
        assert result == "phenylphosphonic acid"

    def test_propanephosphonic_acid(self):
        """CCCP(=O)(O)O -> propanephosphonic acid"""
        mol = Chem.MolFromSmiles("CCCP(=O)(O)O")
        result = name_phosphonic_acid(mol, (3, 4, 5, 6), "propane")
        assert result == "propanephosphonic acid"


class TestPhosphinicAcidNaming:
    """Tests for phosphinic acid naming."""

    def test_dimethylphosphinic_acid(self):
        """CP(C)(=O)O -> dimethylphosphinic acid"""
        mol = Chem.MolFromSmiles("CP(C)(=O)O")
        result = name_phosphinic_acid(mol, tuple(range(mol.GetNumAtoms())))
        assert result == "dimethylphosphinic acid"

    def test_diethylphosphinic_acid(self):
        """CCP(CC)(=O)O -> diethylphosphinic acid"""
        mol = Chem.MolFromSmiles("CCP(CC)(=O)O")
        result = name_phosphinic_acid(mol, tuple(range(mol.GetNumAtoms())))
        assert result == "diethylphosphinic acid"

    def test_ethylmethylphosphinic_acid(self):
        """CCP(C)(=O)O -> ethylmethylphosphinic acid"""
        mol = Chem.MolFromSmiles("CCP(C)(=O)O")
        result = name_phosphinic_acid(mol, tuple(range(mol.GetNumAtoms())))
        assert result == "ethylmethylphosphinic acid"

    def test_dipropylphosphinic_acid(self):
        """CCCP(CCC)(=O)O -> dipropylphosphinic acid"""
        mol = Chem.MolFromSmiles("CCCP(CCC)(=O)O")
        result = name_phosphinic_acid(mol, tuple(range(mol.GetNumAtoms())))
        assert result == "dipropylphosphinic acid"


class TestPhosphateEsterNaming:
    """Tests for phosphate ester naming."""

    def test_methyl_phosphate(self):
        """COP(=O)(O)O -> methyl phosphate"""
        mol = Chem.MolFromSmiles("COP(=O)(O)O")
        # Find phosphorus index
        p_idx = None
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() == 'P':
                p_idx = i
                break
        result = name_phosphate_ester(mol, p_idx)
        assert result == "methyl phosphate"

    def test_dimethyl_phosphate(self):
        """COP(=O)(OC)O -> dimethyl phosphate"""
        mol = Chem.MolFromSmiles("COP(=O)(OC)O")
        p_idx = None
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() == 'P':
                p_idx = i
                break
        result = name_phosphate_ester(mol, p_idx)
        assert result == "dimethyl phosphate"

    def test_trimethyl_phosphate(self):
        """COP(=O)(OC)OC -> trimethyl phosphate"""
        mol = Chem.MolFromSmiles("COP(=O)(OC)OC")
        p_idx = None
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() == 'P':
                p_idx = i
                break
        result = name_phosphate_ester(mol, p_idx)
        assert result == "trimethyl phosphate"

    def test_ethyl_phosphate(self):
        """CCOP(=O)(O)O -> ethyl phosphate"""
        mol = Chem.MolFromSmiles("CCOP(=O)(O)O")
        p_idx = None
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() == 'P':
                p_idx = i
                break
        result = name_phosphate_ester(mol, p_idx)
        assert result == "ethyl phosphate"

    def test_diethyl_phosphate(self):
        """CCOP(=O)(OCC)O -> diethyl phosphate"""
        mol = Chem.MolFromSmiles("CCOP(=O)(OCC)O")
        p_idx = None
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() == 'P':
                p_idx = i
                break
        result = name_phosphate_ester(mol, p_idx)
        assert result == "diethyl phosphate"

    def test_triethyl_phosphate(self):
        """CCOP(=O)(OCC)OCC -> triethyl phosphate"""
        mol = Chem.MolFromSmiles("CCOP(=O)(OCC)OCC")
        p_idx = None
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() == 'P':
                p_idx = i
                break
        result = name_phosphate_ester(mol, p_idx)
        assert result == "triethyl phosphate"

    def test_ethyl_methyl_phosphate(self):
        """COP(=O)(OCC)O -> ethyl methyl phosphate"""
        mol = Chem.MolFromSmiles("COP(=O)(OCC)O")
        p_idx = None
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() == 'P':
                p_idx = i
                break
        result = name_phosphate_ester(mol, p_idx)
        assert result == "ethyl methyl phosphate"


class TestPhosphorusPrefix:
    """Tests for phosphorus group prefix forms."""

    def test_phosphonic_acid_prefix(self):
        """Phosphonic acid prefix is 'phosphono'."""
        assert get_phosphorus_prefix("phosphonic_acid") == "phosphono"

    def test_phosphinic_acid_prefix(self):
        """Phosphinic acid prefix is 'phosphino'."""
        assert get_phosphorus_prefix("phosphinic_acid") == "phosphino"

    def test_phosphine_phosphanyl_prefix(self):
        """Phosphines use phosphanyl prefix when P is a substituent."""
        assert get_phosphorus_prefix("tertiary_phosphine") == "phosphanyl"
        assert get_phosphorus_prefix("secondary_phosphine") == "phosphanyl"
        assert get_phosphorus_prefix("primary_phosphine") == "phosphanyl"

    def test_phosphine_oxide_no_prefix(self):
        """Phosphine oxides use functional class naming, no prefix."""
        assert get_phosphorus_prefix("phosphine_oxide") is None

    def test_phosphate_no_prefix(self):
        """Phosphates use functional class naming, no prefix."""
        assert get_phosphorus_prefix("phosphate_triester") is None
        assert get_phosphorus_prefix("phosphate_monoester") is None


class TestAlkylCounting:
    """Tests for alkyl carbon counting helper."""

    def test_methyl(self):
        """Single carbon = 1."""
        mol = Chem.MolFromSmiles("CP(C)C")
        count = _count_alkyl_carbons(mol, 0, {1})  # Start at C, exclude P
        assert count == 1

    def test_ethyl(self):
        """Two carbons = 2."""
        mol = Chem.MolFromSmiles("CCP(CC)CC")
        count = _count_alkyl_carbons(mol, 0, {2})  # Start at first C, exclude P
        assert count == 2

    def test_propyl(self):
        """Three carbons = 3."""
        mol = Chem.MolFromSmiles("CCCP(CCC)CCC")
        count = _count_alkyl_carbons(mol, 0, {3})
        assert count == 3

    def test_butyl(self):
        """Four carbons = 4."""
        mol = Chem.MolFromSmiles("CCCCP(CCCC)CCCC")
        count = _count_alkyl_carbons(mol, 0, {4})
        assert count == 4


class TestPhosphorusEdgeCases:
    """Edge case tests for phosphorus naming."""

    def test_primary_phosphine_single_carbon(self):
        """Primary phosphine with single methyl group."""
        mol = Chem.MolFromSmiles("CP")
        result = name_phosphine(mol, 1)
        assert result == "methylphosphane"

    def test_secondary_phosphine_two_carbons(self):
        """Secondary phosphine with two methyl groups."""
        mol = Chem.MolFromSmiles("CPC")
        result = name_phosphine(mol, 1)
        assert result == "dimethylphosphane"

    def test_phosphine_oxide_needs_three_carbons(self):
        """Phosphine oxide must have exactly 3 carbon substituents."""
        # A simple phosphine with only 2 carbons should return None
        mol = Chem.MolFromSmiles("CP(C)=O")  # only 2 carbons on P
        result = name_phosphine_oxide(mol, tuple(range(mol.GetNumAtoms())))
        assert result is None  # Not enough carbons

    def test_phosphate_not_phosphine_oxide(self):
        """Phosphate ester should not be named as phosphine oxide."""
        mol = Chem.MolFromSmiles("COP(=O)(OC)OC")  # trimethyl phosphate
        result = name_phosphine_oxide(mol, tuple(range(mol.GetNumAtoms())))
        # Has P=O but C bonded through O, not directly
        # So name_phosphine_oxide should return None
        assert result is None


class TestArylDetection:
    """Tests for _characterize_substituent() aryl vs alkyl detection."""

    def test_phenyl_detected(self):
        """Aromatic 6-C ring attached to P should be detected as phenyl."""
        mol = Chem.MolFromSmiles("c1ccccc1P")
        # Find P index
        p_idx = None
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() == 'P':
                p_idx = i
                break
        # Get the C neighbor of P
        p_atom = mol.GetAtomWithIdx(p_idx)
        c_neighbor = [n for n in p_atom.GetNeighbors() if n.GetSymbol() == 'C'][0]
        result = _characterize_substituent(mol, c_neighbor.GetIdx(), {p_idx})
        assert result == ("aryl", "phenyl")

    def test_methyl_still_alkyl(self):
        """Single non-aromatic carbon should be detected as methyl."""
        mol = Chem.MolFromSmiles("CP")
        result = _characterize_substituent(mol, 0, {1})
        assert result == ("alkyl", "methyl")

    def test_ethyl_still_alkyl(self):
        """Two-carbon chain should be detected as ethyl."""
        mol = Chem.MolFromSmiles("CCP")
        result = _characterize_substituent(mol, 0, {2})
        assert result == ("alkyl", "ethyl")

    def test_naphthyl_detected(self):
        """Naphthalene system attached to P should be detected as naphthyl."""
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1P")
        # Find P index
        p_idx = None
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() == 'P':
                p_idx = i
                break
        p_atom = mol.GetAtomWithIdx(p_idx)
        c_neighbor = [n for n in p_atom.GetNeighbors() if n.GetSymbol() == 'C'][0]
        result = _characterize_substituent(mol, c_neighbor.GetIdx(), {p_idx})
        assert result == ("aryl", "naphthyl")

    def test_hexyl_not_confused_with_phenyl(self):
        """6 linear carbons should be hexyl, not phenyl."""
        mol = Chem.MolFromSmiles("CCCCCCP")
        # Find P index
        p_idx = None
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() == 'P':
                p_idx = i
                break
        p_atom = mol.GetAtomWithIdx(p_idx)
        c_neighbor = [n for n in p_atom.GetNeighbors() if n.GetSymbol() == 'C'][0]
        result = _characterize_substituent(mol, c_neighbor.GetIdx(), {p_idx})
        assert result == ("alkyl", "hexyl")

    def test_propyl_alkyl(self):
        """Three-carbon chain should be propyl."""
        mol = Chem.MolFromSmiles("CCCP")
        result = _characterize_substituent(mol, 0, {3})
        assert result == ("alkyl", "propyl")

    def test_butyl_alkyl(self):
        """Four-carbon chain should be butyl."""
        mol = Chem.MolFromSmiles("CCCCP")
        result = _characterize_substituent(mol, 0, {4})
        assert result == ("alkyl", "butyl")


class TestArylPhosphineNaming:
    """Tests for name_phosphine with aryl substituents."""

    def test_triphenylphosphane(self):
        """PPh3 -> triphenylphosphane"""
        mol = Chem.MolFromSmiles("c1ccccc1P(c2ccccc2)c3ccccc3")
        # Find P index
        p_idx = None
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() == 'P':
                p_idx = i
                break
        result = name_phosphine(mol, p_idx)
        assert result == "triphenylphosphane"

    def test_diphenylmethylphosphane(self):
        """MePPh2 -> diphenylmethylphosphane (alphabetical: methyl after diphenyl -- but ignoring di)"""
        mol = Chem.MolFromSmiles("CP(c1ccccc1)c2ccccc2")
        p_idx = None
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() == 'P':
                p_idx = i
                break
        result = name_phosphine(mol, p_idx)
        # Alphabetical order ignoring "di": methyl < phenyl, so methyl first
        # But _build_substituent_string sorts unique names: methyl, phenyl
        # methyl comes before phenyl -> "methyldiphenylphosphane"
        # Wait: IUPAC says ignore di/tri for alphabetization.
        # "methyl" vs "phenyl": m < p, so methyl first.
        assert "methyl" in result
        assert "phenyl" in result
        assert result == "methyldiphenylphosphane"

    def test_phenylphosphane(self):
        """PhPH2 -> phenylphosphane (primary phosphine with phenyl)"""
        mol = Chem.MolFromSmiles("c1ccccc1[PH2]")
        p_idx = None
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() == 'P':
                p_idx = i
                break
        result = name_phosphine(mol, p_idx)
        assert result == "phenylphosphane"

    def test_diphenylphosphane(self):
        """Ph2PH -> diphenylphosphane (secondary phosphine with two phenyl)"""
        mol = Chem.MolFromSmiles("c1ccccc1[PH]c2ccccc2")
        p_idx = None
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() == 'P':
                p_idx = i
                break
        result = name_phosphine(mol, p_idx)
        assert result == "diphenylphosphane"

    def test_ethyldiphenylphosphane(self):
        """EtPPh2 -> ethyldiphenylphosphane"""
        mol = Chem.MolFromSmiles("CCP(c1ccccc1)c2ccccc2")
        p_idx = None
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() == 'P':
                p_idx = i
                break
        result = name_phosphine(mol, p_idx)
        assert result == "ethyldiphenylphosphane"


class TestArylPhosphineOxideNaming:
    """Tests for name_phosphine_oxide with aryl substituents."""

    def test_triphenylphosphane_oxide(self):
        """O=PPh3 -> triphenylphosphane oxide"""
        mol = Chem.MolFromSmiles("O=P(c1ccccc1)(c2ccccc2)c3ccccc3")
        result = name_phosphine_oxide(mol, tuple(range(mol.GetNumAtoms())))
        assert result == "triphenylphosphane oxide"

    def test_diphenylmethylphosphane_oxide(self):
        """O=P(Me)Ph2 -> methyldiphenylphosphane oxide"""
        mol = Chem.MolFromSmiles("O=P(C)(c1ccccc1)c2ccccc2")
        result = name_phosphine_oxide(mol, tuple(range(mol.GetNumAtoms())))
        assert "methyl" in result
        assert "phenyl" in result
        assert "phosphane oxide" in result
        assert result == "methyldiphenylphosphane oxide"

    def test_ethyldiphenylphosphane_oxide(self):
        """O=P(Et)Ph2 -> ethyldiphenylphosphane oxide"""
        mol = Chem.MolFromSmiles("O=P(CC)(c1ccccc1)c2ccccc2")
        result = name_phosphine_oxide(mol, tuple(range(mol.GetNumAtoms())))
        assert result == "ethyldiphenylphosphane oxide"


class TestArylPhosphinicAcid:
    """Tests for name_phosphinic_acid with aryl substituents."""

    def test_diphenylphosphinic_acid(self):
        """Ph2P(O)(OH) -> diphenylphosphinic acid"""
        mol = Chem.MolFromSmiles("O=P(O)(c1ccccc1)c2ccccc2")
        result = name_phosphinic_acid(mol, tuple(range(mol.GetNumAtoms())))
        assert result == "diphenylphosphinic acid"

    def test_methylphenylphosphinic_acid(self):
        """MePhP(O)(OH) -> methylphenylphosphinic acid (alphabetical)"""
        mol = Chem.MolFromSmiles("O=P(O)(C)c1ccccc1")
        result = name_phosphinic_acid(mol, tuple(range(mol.GetNumAtoms())))
        assert result == "methylphenylphosphinic acid"

    def test_ethylphenylphosphinic_acid(self):
        """EtPhP(O)(OH) -> ethylphenylphosphinic acid"""
        mol = Chem.MolFromSmiles("O=P(O)(CC)c1ccccc1")
        result = name_phosphinic_acid(mol, tuple(range(mol.GetNumAtoms())))
        assert result == "ethylphenylphosphinic acid"


class TestPhosphanylPrefix:
    """Tests for get_phosphanyl_prefix()."""

    def test_diphenylphosphanyl(self):
        """P with 2 phenyl -> diphenylphosphanyl"""
        mol = Chem.MolFromSmiles("c1ccccc1[PH]c2ccccc2")
        p_idx = None
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() == 'P':
                p_idx = i
                break
        result = get_phosphanyl_prefix(mol, p_idx)
        assert result == "diphenylphosphanyl"

    def test_dimethylphosphanyl(self):
        """P with 2 methyl -> dimethylphosphanyl"""
        mol = Chem.MolFromSmiles("CPC")
        p_idx = 1
        result = get_phosphanyl_prefix(mol, p_idx)
        assert result == "dimethylphosphanyl"

    def test_triphenylphosphanyl(self):
        """P with 3 phenyl -> triphenylphosphanyl"""
        mol = Chem.MolFromSmiles("c1ccccc1P(c2ccccc2)c3ccccc3")
        p_idx = None
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() == 'P':
                p_idx = i
                break
        result = get_phosphanyl_prefix(mol, p_idx)
        assert result == "triphenylphosphanyl"

    def test_methylphosphanyl(self):
        """P with 1 methyl -> methylphosphanyl"""
        mol = Chem.MolFromSmiles("CP")
        p_idx = 1
        result = get_phosphanyl_prefix(mol, p_idx)
        assert result == "methylphosphanyl"

    def test_bare_phosphanyl(self):
        """P with no C substituents -> phosphanyl"""
        mol = Chem.MolFromSmiles("[PH3]")
        result = get_phosphanyl_prefix(mol, 0)
        assert result == "phosphanyl"

    def test_ethylphosphanyl(self):
        """P with 1 ethyl -> ethylphosphanyl"""
        mol = Chem.MolFromSmiles("CCP")
        p_idx = 2
        result = get_phosphanyl_prefix(mol, p_idx)
        assert result == "ethylphosphanyl"

    def test_methyldiphenylphosphanyl(self):
        """P with 1 methyl + 2 phenyl -> methyldiphenylphosphanyl"""
        mol = Chem.MolFromSmiles("CP(c1ccccc1)c2ccccc2")
        p_idx = None
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetSymbol() == 'P':
                p_idx = i
                break
        result = get_phosphanyl_prefix(mol, p_idx)
        assert result == "methyldiphenylphosphanyl"
