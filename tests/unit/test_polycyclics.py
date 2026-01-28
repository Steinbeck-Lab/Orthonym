"""
Tests for polycyclic aromatic hydrocarbon (PAH) naming.

Tests cover:
- PAH identification (naphthalene, anthracene, phenanthrene, etc.)
- Unsubstituted PAH retained names
- Substituted PAH naming with correct IUPAC locants
- PAH numbering (alpha vs beta positions)
"""

import pytest
from rdkit import Chem
from orthonym import name_compound
from orthonym.rules.polycyclics import (
    identify_polycyclic,
    get_polycyclic_substituents,
    name_substituted_polycyclic,
    get_polycyclic_core_atoms,
)
from orthonym.data.polycyclic_data import (
    POLYCYCLIC_DATA,
    get_polycyclic_by_smiles,
    is_polycyclic_aromatic,
    get_pah_names,
)


class TestPolycyclicDataLookup:
    """Tests for polycyclic_data.py lookup functions."""

    def test_polycyclic_data_contains_common_pahs(self):
        """All common PAHs should be in POLYCYCLIC_DATA."""
        expected_pahs = [
            'naphthalene', 'anthracene', 'phenanthrene', 'pyrene',
            'fluorene', 'acenaphthene', 'acenaphthylene', 'chrysene'
        ]
        for pah in expected_pahs:
            assert pah in POLYCYCLIC_DATA, f"{pah} not in POLYCYCLIC_DATA"

    def test_get_polycyclic_by_smiles_naphthalene(self):
        """Naphthalene lookup by canonical SMILES."""
        result = get_polycyclic_by_smiles('c1ccc2ccccc2c1')
        assert result is not None
        assert result['name'] == 'naphthalene'

    def test_get_polycyclic_by_smiles_not_found(self):
        """Non-PAH SMILES returns None."""
        result = get_polycyclic_by_smiles('c1ccccc1')  # benzene
        assert result is None

    def test_is_polycyclic_aromatic_naphthalene(self):
        """Naphthalene is recognized as polycyclic aromatic."""
        assert is_polycyclic_aromatic('c1ccc2ccccc2c1') is True

    def test_is_polycyclic_aromatic_benzene(self):
        """Benzene is NOT a polycyclic aromatic."""
        assert is_polycyclic_aromatic('c1ccccc1') is False

    def test_get_pah_names(self):
        """get_pah_names returns list of all PAH names."""
        names = get_pah_names()
        assert isinstance(names, list)
        assert 'naphthalene' in names
        assert 'anthracene' in names
        assert len(names) >= 8


class TestPolycyclicIdentification:
    """Tests for identify_polycyclic function."""

    def test_identify_naphthalene(self):
        """Unsubstituted naphthalene is identified."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        assert identify_polycyclic(mol) == 'naphthalene'

    def test_identify_anthracene(self):
        """Unsubstituted anthracene is identified."""
        mol = Chem.MolFromSmiles('c1ccc2cc3ccccc3cc2c1')
        assert identify_polycyclic(mol) == 'anthracene'

    def test_identify_phenanthrene(self):
        """Unsubstituted phenanthrene is identified."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)ccc1ccccc12')
        assert identify_polycyclic(mol) == 'phenanthrene'

    def test_identify_pyrene(self):
        """Unsubstituted pyrene is identified."""
        mol = Chem.MolFromSmiles('c1cc2ccc3cccc4ccc(c1)c2c34')
        assert identify_polycyclic(mol) == 'pyrene'

    def test_identify_fluorene(self):
        """Unsubstituted fluorene is identified (has sp3 carbon)."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)Cc1ccccc1-2')
        assert identify_polycyclic(mol) == 'fluorene'

    def test_identify_chrysene(self):
        """Unsubstituted chrysene is identified."""
        mol = Chem.MolFromSmiles('c1ccc2c(c1)ccc1c3ccccc3ccc21')
        assert identify_polycyclic(mol) == 'chrysene'

    def test_identify_substituted_naphthalene(self):
        """Substituted naphthalene is identified as naphthalene."""
        mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')  # 2-methylnaphthalene
        assert identify_polycyclic(mol) == 'naphthalene'

    def test_not_polycyclic_benzene(self):
        """Benzene is not a polycyclic aromatic."""
        mol = Chem.MolFromSmiles('c1ccccc1')
        assert identify_polycyclic(mol) is None

    def test_not_polycyclic_cyclohexane(self):
        """Cyclohexane is not a polycyclic aromatic."""
        mol = Chem.MolFromSmiles('C1CCCCC1')
        assert identify_polycyclic(mol) is None


class TestPolycyclicRetainedNames:
    """Tests for PAH retained names via name_compound."""

    def test_naphthalene_retained(self):
        """c1ccc2ccccc2c1 -> naphthalene"""
        assert name_compound('c1ccc2ccccc2c1') == 'naphthalene'

    def test_anthracene_retained(self):
        """c1ccc2cc3ccccc3cc2c1 -> anthracene"""
        assert name_compound('c1ccc2cc3ccccc3cc2c1') == 'anthracene'

    def test_phenanthrene_retained(self):
        """c1ccc2c(c1)ccc1ccccc12 -> phenanthrene"""
        assert name_compound('c1ccc2c(c1)ccc1ccccc12') == 'phenanthrene'

    def test_pyrene_retained(self):
        """Pyrene -> pyrene"""
        assert name_compound('c1cc2ccc3cccc4ccc(c1)c2c34') == 'pyrene'

    def test_fluorene_retained(self):
        """Fluorene -> fluorene"""
        assert name_compound('c1ccc2c(c1)Cc1ccccc1-2') == 'fluorene'

    def test_acenaphthene_retained(self):
        """Acenaphthene -> acenaphthene"""
        assert name_compound('c1cc2c3c(cccc3c1)CC2') == 'acenaphthene'

    def test_acenaphthylene_retained(self):
        """Acenaphthylene -> acenaphthylene"""
        assert name_compound('C1=Cc2cccc3cccc1c23') == 'acenaphthylene'

    def test_chrysene_retained(self):
        """Chrysene -> chrysene"""
        assert name_compound('c1ccc2c(c1)ccc1c3ccccc3ccc21') == 'chrysene'


class TestSubstitutedPolycyclics:
    """Tests for substituted polycyclic aromatic naming."""

    def test_1_methylnaphthalene(self):
        """1-methylnaphthalene: methyl at alpha position."""
        result = name_compound('Cc1cccc2ccccc12')
        assert '1-methylnaphthalene' == result

    def test_2_methylnaphthalene(self):
        """2-methylnaphthalene: methyl at beta position."""
        result = name_compound('Cc1ccc2ccccc2c1')
        assert '2-methylnaphthalene' == result

    def test_1_chloronaphthalene(self):
        """1-chloronaphthalene: chloro at alpha position."""
        result = name_compound('Clc1cccc2ccccc12')
        assert '1-chloronaphthalene' == result

    def test_2_chloronaphthalene(self):
        """2-chloronaphthalene: chloro at beta position."""
        result = name_compound('Clc1ccc2ccccc2c1')
        assert '2-chloronaphthalene' == result

    def test_1_bromonaphthalene(self):
        """1-bromonaphthalene: bromo at alpha position."""
        result = name_compound('Brc1cccc2ccccc12')
        assert '1-bromonaphthalene' == result

    def test_2_bromonaphthalene(self):
        """2-bromonaphthalene: bromo at beta position."""
        result = name_compound('Brc1ccc2ccccc2c1')
        assert '2-bromonaphthalene' == result


class TestPolycyclicNumberingFixed:
    """Tests verifying PAH numbering follows IUPAC rules."""

    def test_alpha_position_gets_low_locant(self):
        """Substituent at alpha position gets locant 1 (not 4, 5, or 8)."""
        # 1-methylnaphthalene - methyl adjacent to fusion carbon
        mol = Chem.MolFromSmiles('Cc1cccc2ccccc12')
        subs = get_polycyclic_substituents(mol, 'naphthalene')
        # The locant should be 1, 4, 5, or 8 (alpha positions)
        locant = list(subs.keys())[0]
        assert locant == 1, f"Expected locant 1, got {locant}"

    def test_beta_position_gets_low_locant(self):
        """Substituent at beta position gets locant 2 (not 3, 6, or 7)."""
        # 2-methylnaphthalene - methyl NOT adjacent to fusion carbon
        mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')
        subs = get_polycyclic_substituents(mol, 'naphthalene')
        # The locant should be 2, 3, 6, or 7 (beta positions)
        locant = list(subs.keys())[0]
        assert locant == 2, f"Expected locant 2, got {locant}"

    def test_alpha_and_beta_are_different_positions(self):
        """1-methylnaphthalene and 2-methylnaphthalene are structurally different."""
        smiles_1_methyl = 'Cc1cccc2ccccc12'
        smiles_2_methyl = 'Cc1ccc2ccccc2c1'

        mol1 = Chem.MolFromSmiles(smiles_1_methyl)
        mol2 = Chem.MolFromSmiles(smiles_2_methyl)

        canon1 = Chem.MolToSmiles(mol1, canonical=True)
        canon2 = Chem.MolToSmiles(mol2, canonical=True)

        # They should have different canonical SMILES
        assert canon1 != canon2, "1-methyl and 2-methyl naphthalene should be different"


class TestPolycyclicSubstituentDetection:
    """Tests for get_polycyclic_substituents function."""

    def test_methylnaphthalene_substituent_detection(self):
        """Methyl group is correctly detected as substituent."""
        mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')
        subs = get_polycyclic_substituents(mol, 'naphthalene')

        assert len(subs) == 1
        sub_info = list(subs.values())[0][0]
        assert sub_info['name'] == 'methyl'

    def test_chloronaphthalene_substituent_detection(self):
        """Chloro group is correctly detected as substituent."""
        mol = Chem.MolFromSmiles('Clc1ccc2ccccc2c1')
        subs = get_polycyclic_substituents(mol, 'naphthalene')

        assert len(subs) == 1
        sub_info = list(subs.values())[0][0]
        assert sub_info['name'] == 'chloro'

    def test_unsubstituted_naphthalene_no_substituents(self):
        """Unsubstituted naphthalene has no substituents."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        subs = get_polycyclic_substituents(mol, 'naphthalene')

        assert len(subs) == 0


class TestPolycyclicCoreAtoms:
    """Tests for get_polycyclic_core_atoms function."""

    def test_naphthalene_core_has_10_atoms(self):
        """Naphthalene core has 10 atoms."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        core = get_polycyclic_core_atoms(mol, 'naphthalene')

        assert core is not None
        assert len(core) == 10

    def test_methylnaphthalene_core_excludes_methyl(self):
        """Methylnaphthalene core excludes the methyl carbon."""
        mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')
        core = get_polycyclic_core_atoms(mol, 'naphthalene')

        assert core is not None
        assert len(core) == 10  # Only naphthalene core, not methyl

    def test_anthracene_core_has_14_atoms(self):
        """Anthracene core has 14 atoms."""
        mol = Chem.MolFromSmiles('c1ccc2cc3ccccc3cc2c1')
        core = get_polycyclic_core_atoms(mol, 'anthracene')

        assert core is not None
        assert len(core) == 14


@pytest.mark.unit
class TestPolycyclicNamingFunction:
    """Tests for name_substituted_polycyclic function."""

    def test_name_substituted_naphthalene_methyl(self):
        """name_substituted_polycyclic generates correct name for methylnaphthalene."""
        mol = Chem.MolFromSmiles('Cc1ccc2ccccc2c1')
        subs = get_polycyclic_substituents(mol, 'naphthalene')
        name = name_substituted_polycyclic(mol, 'naphthalene', subs)

        assert 'methylnaphthalene' in name
        assert '2-' in name

    def test_name_substituted_naphthalene_chloro(self):
        """name_substituted_polycyclic generates correct name for chloronaphthalene."""
        mol = Chem.MolFromSmiles('Clc1ccc2ccccc2c1')
        subs = get_polycyclic_substituents(mol, 'naphthalene')
        name = name_substituted_polycyclic(mol, 'naphthalene', subs)

        assert 'chloronaphthalene' in name

    def test_name_unsubstituted_returns_parent_name(self):
        """name_substituted_polycyclic returns parent name when no substituents."""
        mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')
        subs = {}
        name = name_substituted_polycyclic(mol, 'naphthalene', subs)

        assert name == 'naphthalene'
