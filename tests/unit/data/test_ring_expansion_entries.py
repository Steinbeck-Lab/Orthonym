"""Unit tests for Phase 89 ring system dictionary expansion entries.

Tests that all new dictionary entries have:
1. Correct canonical SMILES keys (round-trip verified)
2. Expected names
3. Correct parent_atoms counts matching heavy atom counts
4. iupac_locants dicts with correct number of entries
5. Proper routing (e.g., decalin does NOT produce VB notation)
"""

import pytest
from rdkit import Chem

from orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
from orthonym.data.natural_products import NATURAL_PRODUCT_DERIVATIVES
from orthonym.data.bicyclo_systems import BICYCLO_RETAINED_NAMES
from orthonym.data.partial_saturation_refs import AROMATIC_REFERENCES


def _canonical(smiles: str) -> str:
    """Return RDKit canonical SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    return Chem.MolToSmiles(mol, canonical=True)


# =========================================================================
# FUSED HETEROCYCLE ENTRIES (5 new entries)
# =========================================================================

class TestPhenanthridineEntry:
    """Phenanthridine: angular tricyclic, N at IUPAC position 5."""

    def test_entry_exists(self):
        can = _canonical('c1ccc2c(c1)ccc1cccnc12')
        assert can in FUSED_HETEROCYCLE_DATA

    def test_name(self):
        can = _canonical('c1ccc2c(c1)ccc1cccnc12')
        assert FUSED_HETEROCYCLE_DATA[can]['name'] == 'phenanthridine'

    def test_parent_atoms(self):
        can = _canonical('c1ccc2c(c1)ccc1cccnc12')
        data = FUSED_HETEROCYCLE_DATA[can]
        mol = Chem.MolFromSmiles(can)
        assert data['parent_atoms'] == mol.GetNumAtoms()

    def test_iupac_locants_count(self):
        can = _canonical('c1ccc2c(c1)ccc1cccnc12')
        data = FUSED_HETEROCYCLE_DATA[can]
        assert len(data['iupac_locants']) == data['parent_atoms']

    def test_canonical_roundtrip(self):
        can = _canonical('c1ccc2c(c1)ccc1cccnc12')
        assert _canonical(can) == can

    def test_ring_system(self):
        can = _canonical('c1ccc2c(c1)ccc1cccnc12')
        assert FUSED_HETEROCYCLE_DATA[can]['ring_system'] == 'tricyclic'


class TestBetaCarbolineEntry:
    """9H-beta-Carboline: pyrido[3,4-b]indole, tricyclic."""

    def test_entry_exists(self):
        can = _canonical('c1ccc2c(c1)[nH]c1cnccc12')
        assert can in FUSED_HETEROCYCLE_DATA

    def test_name_contains_carboline(self):
        can = _canonical('c1ccc2c(c1)[nH]c1cnccc12')
        name = FUSED_HETEROCYCLE_DATA[can]['name']
        assert 'carboline' in name.lower()

    def test_parent_atoms(self):
        can = _canonical('c1ccc2c(c1)[nH]c1cnccc12')
        data = FUSED_HETEROCYCLE_DATA[can]
        mol = Chem.MolFromSmiles(can)
        assert data['parent_atoms'] == mol.GetNumAtoms()

    def test_iupac_locants_count(self):
        can = _canonical('c1ccc2c(c1)[nH]c1cnccc12')
        data = FUSED_HETEROCYCLE_DATA[can]
        assert len(data['iupac_locants']) == data['parent_atoms']

    def test_tautomer_locant(self):
        can = _canonical('c1ccc2c(c1)[nH]c1cnccc12')
        data = FUSED_HETEROCYCLE_DATA[can]
        assert data['tautomer_locant'] == 9


class TestAcridoneEntry:
    """Acridone: acridin-9(10H)-one, tricyclic with exocyclic =O."""

    def test_entry_exists(self):
        can = _canonical('O=c1c2ccccc2[nH]c2ccccc12')
        assert can in FUSED_HETEROCYCLE_DATA

    def test_name(self):
        can = _canonical('O=c1c2ccccc2[nH]c2ccccc12')
        assert FUSED_HETEROCYCLE_DATA[can]['name'] == 'acridone'

    def test_parent_atoms(self):
        can = _canonical('O=c1c2ccccc2[nH]c2ccccc12')
        data = FUSED_HETEROCYCLE_DATA[can]
        mol = Chem.MolFromSmiles(can)
        assert data['parent_atoms'] == mol.GetNumAtoms()

    def test_iupac_locants_count(self):
        can = _canonical('O=c1c2ccccc2[nH]c2ccccc12')
        data = FUSED_HETEROCYCLE_DATA[can]
        assert len(data['iupac_locants']) == data['parent_atoms']

    def test_tautomer_locant(self):
        can = _canonical('O=c1c2ccccc2[nH]c2ccccc12')
        data = FUSED_HETEROCYCLE_DATA[can]
        assert data['tautomer_locant'] == 10


class TestQuinolizineEntry:
    """4H-Quinolizine: N-bridgehead bicyclic."""

    def test_entry_exists(self):
        can = _canonical('C1=CC2=CCC=CN2C=C1')
        assert can in FUSED_HETEROCYCLE_DATA

    def test_name(self):
        can = _canonical('C1=CC2=CCC=CN2C=C1')
        name = FUSED_HETEROCYCLE_DATA[can]['name']
        assert 'quinolizine' in name.lower()

    def test_parent_atoms(self):
        can = _canonical('C1=CC2=CCC=CN2C=C1')
        data = FUSED_HETEROCYCLE_DATA[can]
        assert data['parent_atoms'] == 10

    def test_iupac_locants_count(self):
        can = _canonical('C1=CC2=CCC=CN2C=C1')
        data = FUSED_HETEROCYCLE_DATA[can]
        assert len(data['iupac_locants']) == data['parent_atoms']

    def test_tautomer_locant(self):
        can = _canonical('C1=CC2=CCC=CN2C=C1')
        data = FUSED_HETEROCYCLE_DATA[can]
        assert data['tautomer_locant'] == 4


class TestQuinolizidineEntry:
    """Quinolizidine: fully saturated N-bridgehead bicyclic."""

    def test_entry_exists(self):
        can = _canonical('C1CCN2CCCCC2C1')
        assert can in FUSED_HETEROCYCLE_DATA

    def test_name(self):
        can = _canonical('C1CCN2CCCCC2C1')
        assert FUSED_HETEROCYCLE_DATA[can]['name'] == 'quinolizidine'

    def test_parent_atoms(self):
        can = _canonical('C1CCN2CCCCC2C1')
        data = FUSED_HETEROCYCLE_DATA[can]
        assert data['parent_atoms'] == 10

    def test_iupac_locants_count(self):
        can = _canonical('C1CCN2CCCCC2C1')
        data = FUSED_HETEROCYCLE_DATA[can]
        assert len(data['iupac_locants']) == data['parent_atoms']

    def test_tautomer_locant_none(self):
        can = _canonical('C1CCN2CCCCC2C1')
        data = FUSED_HETEROCYCLE_DATA[can]
        assert data['tautomer_locant'] is None


# =========================================================================
# NATURAL PRODUCT DERIVATIVE ENTRIES (7 new entries)
# =========================================================================

class TestFlavonoidEntries:
    """Flavonoid / chromene derivative entries in NATURAL_PRODUCT_DERIVATIVES."""

    def test_flavone(self):
        can = _canonical('O=c1cc(-c2ccccc2)oc2ccccc12')
        assert NATURAL_PRODUCT_DERIVATIVES.get(can) == 'flavone'

    def test_flavanone(self):
        can = _canonical('O=C1CC(c2ccccc2)Oc2ccccc21')
        assert NATURAL_PRODUCT_DERIVATIVES.get(can) == 'flavanone'

    def test_isoflavone(self):
        can = _canonical('O=c1c(-c2ccccc2)coc2ccccc12')
        assert NATURAL_PRODUCT_DERIVATIVES.get(can) == 'isoflavone'

    def test_chromanone(self):
        can = _canonical('O=C1CCOc2ccccc21')
        assert NATURAL_PRODUCT_DERIVATIVES.get(can) == 'chromanone'

    def test_chromone(self):
        can = _canonical('O=c1ccoc2ccccc12')
        assert NATURAL_PRODUCT_DERIVATIVES.get(can) == 'chromone'


class TestTerpenoidNPEntries:
    """Terpenoid entries: pinane and bornane."""

    def test_pinane(self):
        can = _canonical('CC1CCC2CC1C2(C)C')
        assert NATURAL_PRODUCT_DERIVATIVES.get(can) == 'pinane'

    def test_bornane(self):
        can = _canonical('CC1(C)C2CCC1(C)CC2')
        assert NATURAL_PRODUCT_DERIVATIVES.get(can) == 'bornane'


# =========================================================================
# BICYCLO RETAINED NAME ENTRIES (1 new entry)
# =========================================================================

class TestNorborneneEntry:
    """Norbornene: bicyclo[2.2.1]hept-2-ene."""

    def test_entry_exists(self):
        can = _canonical('C1=CC2CCC1C2')
        assert can in BICYCLO_RETAINED_NAMES

    def test_name(self):
        can = _canonical('C1=CC2CCC1C2')
        assert BICYCLO_RETAINED_NAMES[can] == 'norbornene'

    def test_canonical_roundtrip(self):
        can = _canonical('C1=CC2CCC1C2')
        assert _canonical(can) == can


# =========================================================================
# AROMATIC REFERENCE ENTRIES (3 new entries)
# =========================================================================

class TestAromaticReferenceEntries:
    """New entries in AROMATIC_REFERENCES."""

    def test_pyrazine_exists(self):
        assert 'pyrazine' in AROMATIC_REFERENCES

    def test_pyrazine_ring_atoms(self):
        assert AROMATIC_REFERENCES['pyrazine']['ring_atoms'] == 6

    def test_pyridazine_exists(self):
        assert 'pyridazine' in AROMATIC_REFERENCES

    def test_pyridazine_ring_atoms(self):
        assert AROMATIC_REFERENCES['pyridazine']['ring_atoms'] == 6

    def test_fluorene_exists(self):
        assert 'fluorene' in AROMATIC_REFERENCES

    def test_fluorene_ring_atoms(self):
        assert AROMATIC_REFERENCES['fluorene']['ring_atoms'] == 13

    def test_fluorene_is_carbocycle(self):
        assert AROMATIC_REFERENCES['fluorene'].get('is_carbocycle') is True


# =========================================================================
# CANONICAL SMILES VALIDATION (all new entries)
# =========================================================================

class TestCanonicalSmilesIntegrity:
    """Verify all new dictionary keys are truly canonical SMILES."""

    @pytest.mark.parametrize("smiles", [
        'c1ccc2c(c1)ccc1cccnc12',      # phenanthridine
        'c1ccc2c(c1)[nH]c1cnccc12',     # beta-carboline
        'O=c1c2ccccc2[nH]c2ccccc12',    # acridone
        'C1=CC2=CCC=CN2C=C1',           # 4H-quinolizine
        'C1CCN2CCCCC2C1',               # quinolizidine
        'O=c1cc(-c2ccccc2)oc2ccccc12',  # flavone
        'O=C1CC(c2ccccc2)Oc2ccccc21',   # flavanone
        'O=c1c(-c2ccccc2)coc2ccccc12',  # isoflavone
        'O=C1CCOc2ccccc21',             # chromanone
        'O=c1ccoc2ccccc12',             # chromone
        'C1=CC2CCC1C2',                 # norbornene
    ])
    def test_smiles_is_canonical(self, smiles):
        """Each SMILES key should round-trip to itself."""
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        canonical = Chem.MolToSmiles(mol, canonical=True)
        assert canonical == smiles, f"Not canonical: {smiles} -> {canonical}"


# =========================================================================
# DECALIN ROUTING CHECK (RING-04 coverage)
# =========================================================================

class TestDecalinRouting:
    """Verify decalin does NOT produce VB polycyclic notation."""

    def test_decalin_no_vb_notation(self):
        from orthonym.namer import name_compound
        result = name_compound('C1CCC2CCCCC2C1')
        assert result is not None
        assert 'bicyclo' not in result.lower(), (
            f"Decalin produced VB notation: {result}"
        )
