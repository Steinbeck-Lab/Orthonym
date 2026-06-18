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


class TestRetainedFusedStemAzulene:
    """v22 Phase C-T10 (V-1, theme T10): the empty/malformed-stem bug.

    Azulene is a retained fused-ring hydrocarbon (IUPAC 2013 P-25.1.1,
    Table 28.1): a 5-membered ring ortho-fused to a 7-membered ring, fully
    mancude (aromatic). It was *declared* in resolvers._NAMED_PAH_SYSTEMS but
    had no entry in POLYCYCLIC_DATA, so identify_polycyclic() returned None and
    the molecule fell through name_fused_heterocycle (None: carbocyclic) and
    name_ortho_fused_bicyclic (None: aromatic, so _name_saturated_fused_
    carbocyclic declines) to the acyclic chain catch-all, which emitted the
    malformed empty stem 'ane'. Fixed at the data/lookup source.

    These tests assert the OUTPUT (the produced name), not a classification
    flag. They cover the *bare* retained-fused-PAH naming path for the class
    (the gold V-1 azulene + the other bare retained PAHs), not just the one
    gold molecule. (Substituted-azulene locants route through the
    naphthalene-specific `_map_pah_atoms_to_iupac` mapper and are out of C-T10
    scope — owned by E1/DD4 fused-ring numbering.)
    """

    AZULENE_SMILES = 'C1=CC=C2C=CC=CC=C12'
    AZULENE_CANONICAL = 'c1ccc2cccc-2cc1'

    def test_azulene_in_polycyclic_data(self):
        """The data/lookup source now carries azulene (root-cause locus)."""
        assert 'azulene' in POLYCYCLIC_DATA
        entry = POLYCYCLIC_DATA['azulene']
        assert entry['num_atoms'] == 10
        assert entry['num_rings'] == 2

    def test_azulene_declared_named_pah_now_in_data(self):
        """Reconcile the specific azulene inconsistency: azulene was declared in
        resolvers._NAMED_PAH_SYSTEMS but absent from POLYCYCLIC_DATA (the cause
        of the empty-stem trap); it must now be present in both. (This checks
        only the azulene reconciliation, NOT a universal _NAMED_PAH_SYSTEMS
        invariant — fluoranthene is also declared but resolves via the fusion
        path rather than POLYCYCLIC_DATA membership; see the class-invariant
        test below for the no-empty-stem guarantee over the whole set.)"""
        from orthonym.assembly.resolvers import _NAMED_PAH_SYSTEMS
        assert 'azulene' in _NAMED_PAH_SYSTEMS
        assert 'azulene' in POLYCYCLIC_DATA

    def test_named_pah_systems_class_never_emits_empty_stem(self):
        """C-T10 CLASS INVARIANT: no retained PAH declared in
        resolvers._NAMED_PAH_SYSTEMS may collapse to the malformed empty stem
        'ane' / 'unknown' (the V-1 bug class). azulene was the declared member
        that did; this guards the whole declared set against recurrence,
        regardless of which path (POLYCYCLIC_DATA lookup or fusion) names it.
        Names present in POLYCYCLIC_DATA use their canonical SMILES; fluoranthene
        is declared but resolves via the fusion path (verified)."""
        from orthonym.assembly.resolvers import _NAMED_PAH_SYSTEMS
        # representative bare-parent SMILES for the one declared name not in
        # POLYCYCLIC_DATA (fluoranthene resolves via the fusion descriptor path).
        extra = {'fluoranthene': 'c1ccc-2c(c1)-c1cccc3cccc-2c13'}
        for nm in sorted(_NAMED_PAH_SYSTEMS):
            smi = (POLYCYCLIC_DATA[nm]['canonical_smiles']
                   if nm in POLYCYCLIC_DATA else extra.get(nm))
            assert smi is not None, f"no representative SMILES for declared PAH {nm!r}"
            out = name_compound(smi)
            assert out and out != 'ane' and 'unknown' not in out, (nm, out)

    def test_azulene_exact_canonical_lookup(self):
        """Bare azulene resolves via the exact canonical-SMILES lookup."""
        assert is_polycyclic_aromatic(self.AZULENE_CANONICAL) is True
        result = get_polycyclic_by_smiles(self.AZULENE_CANONICAL)
        assert result is not None and result['name'] == 'azulene'

    def test_identify_polycyclic_azulene(self):
        """identify_polycyclic resolves azulene (was None -> empty-stem bug)."""
        mol = Chem.MolFromSmiles(self.AZULENE_SMILES)
        assert identify_polycyclic(mol) == 'azulene'

    def test_azulene_end_to_end_gold(self):
        """V-1 gold: the whole-pipeline name is 'azulene', not 'ane'."""
        assert name_compound(self.AZULENE_SMILES) == 'azulene'

    def test_azulene_never_emits_malformed_stem(self):
        """The malformed empty stem 'ane' / 'unknown' must never be produced."""
        out = name_compound(self.AZULENE_SMILES)
        assert out == 'azulene'
        assert out != 'ane'
        assert 'unknown' not in out

    def test_azulene_input_order_deterministic(self):
        """Same structure, different SMILES spellings -> same name (the
        determinism gate this fix must hold)."""
        import random
        base = Chem.MolFromSmiles(self.AZULENE_SMILES)
        names = set()
        for seed in range(6):
            idx = list(range(base.GetNumAtoms()))
            random.Random(seed).shuffle(idx)
            respelled = Chem.MolToSmiles(Chem.RenumberAtoms(base, idx),
                                         canonical=False)
            names.add(name_compound(respelled))
        assert names == {'azulene'}, names

    @pytest.mark.parametrize("smiles,expected", [
        ('c1ccc2ccccc2c1', 'naphthalene'),       # bicyclic 6-6
        ('c1ccc2cc3ccccc3cc2c1', 'anthracene'),  # linear tricyclic
        ('c1ccc2c(c1)ccc1ccccc12', 'phenanthrene'),  # angular tricyclic
        ('C1=CC=C2C=CC=CC=C12', 'azulene'),       # bicyclic 5-7 (this fix)
    ])
    def test_retained_fused_carbocyclic_class(self, smiles, expected):
        """The shared bare retained-fused-PAH naming path is intact for the
        whole class (A8 rule-family coverage), including the new azulene
        member — none collapse to an empty stem."""
        assert name_compound(smiles) == expected
