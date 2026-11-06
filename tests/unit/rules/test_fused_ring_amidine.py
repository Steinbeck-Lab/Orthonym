"""A fused hydro ring system whose N=C-N unit is part of the ring is named by fusion nomenclature.

The amidine group is the characteristic group "-C(=NH)-NH2" "attached to a cyclic parent hydride":
 "Suffixes for amidines" (the Blue Book),:34200 "All the other suffixes are used to
name acyclic polyamidines and all amidines having the suffix attached to a cyclic parent hydride".
A ring that holds the carbon and both nitrogens of the N=C-N unit has those three atoms as skeletal
atoms of the heterocycle, so no '-imidamide' suffix stands on it and the ring system is the parent.
``bridged_fused_pin.principal_suffix`` read the unit as the principal group, found no suffix for it
and declined, so 1,8-diazabicyclo[5.4.0]undec-7-ene (DBU) and 1,5-diazabicyclo[4.3.0]non-5-ene (DBN)
and their cations had no name at any tier; the same defect was fixed for the monocycles in
``ions._ring_bears_principal_group``. The names are the fusion names of with the hydro
prefixes of (:16872) and the numbering of (:3219); the cations add the suffix 'ium'
to the neutral parent "General rule for systematically naming cationic centers in
parent hydrides",:41366,:41368), at the fixed fusion locant of the protonated nitrogen.
"""
import pytest
from rdkit import Chem

from orthonym.perception.functional_groups import detect_functional_groups
from orthonym.rules.seniority import get_principal_group, without_ring_skeleton_amidine
from tests.support.jars import jar_or_skip
from tests.support.rt_assert import assert_full_rt
from tests.support.pin_tiers import tier_row

pytestmark = pytest.mark.unit

# (SMILES, the fusion name). The fusion numbering of each skeleton is fixed: N1 is the imine
# nitrogen of the pyrimido[1,2-a]azepine and pyrrolo[1,2-a]pyrimidine, so the cation is '-1-ium'.
NEUTRAL = [
    ("C1CCC2=NCCCN2CC1", "2,3,4,6,7,8,9,10-octahydropyrimido[1,2-a]azepine"),          # DBU
    ("C1CC2=NCCCN2C1", "2,3,4,6,7,8-hexahydropyrrolo[1,2-a]pyrimidine"),                # DBN
    ("C1CCN2CCN=C2C1", "2,3,5,6,7,8-hexahydroimidazo[1,2-a]pyridine"),
    ("CC1CCC2=NCCCN2CC1", "8-methyl-2,3,4,6,7,8,9,10-octahydropyrimido[1,2-a]azepine"),
    ("C[C@H]1CCC2=NCCCN2C1", "(7S)-7-methyl-3,4,6,7,8,9-hexahydro-2H-pyrido[1,2-a]pyrimidine"),
]
CATIONS = [
    ("C1CCC2=[NH+]CCCN2CC1", "2,3,4,6,7,8,9,10-octahydropyrimido[1,2-a]azepin-1-ium"),   # DBU-H+
    ("C1CC2=[NH+]CCCN2C1", "2,3,4,6,7,8-hexahydropyrrolo[1,2-a]pyrimidin-1-ium"),         # DBN-H+
    # the other resonance drawing of DBU-H+ (the charge on the bridgehead nitrogen): one name
    ("C1CCC2=[N+](CC1)CCCN2", "2,3,4,6,7,8,9,10-octahydropyrimido[1,2-a]azepin-1-ium"),
    ("C1CCN2CC[NH+]=C2C1", "2,3,5,6,7,8-hexahydroimidazo[1,2-a]pyridin-1-ium"),
    ("C1CCC2=[NH+]CCCN2CC1.[Cl-]", "2,3,4,6,7,8,9,10-octahydropyrimido[1,2-a]azepin-1-ium chloride"),
]


def _mol_groups(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol, detect_functional_groups(mol)


@pytest.mark.parametrize("smiles", ["C1CCC2=NCCCN2CC1", "C1CC2=NCCCN2C1", "C1CCN2CCN=C2C1"])
def test_the_ring_amidine_is_set_aside_when_all_three_atoms_are_ring_atoms(smiles):
    """The red side: ``get_principal_group`` names the ring N=C-N unit 'amidine'; with the unit set
    aside there is no principal group and the ring system is the parent:34200)."""
    mol, groups = _mol_groups(smiles)
    assert get_principal_group(mol, groups)[0] == "amidine"
    rest = without_ring_skeleton_amidine(groups, set(range(mol.GetNumAtoms())))
    assert "amidine" not in rest
    assert get_principal_group(mol, rest) == (None, [])


def test_an_amidine_with_an_atom_outside_the_ring_system_is_kept():
    """A '-carboximidamide' on a ring (benzamidine) and an amidine whose =NH is exocyclic are
    groups of the molecule, not the ring:34200 'attached to a cyclic parent
    hydride'); the dict is returned unchanged."""
    for smiles, ring in (("NC(=N)c1ccccc1", set(range(2, 8))), ("N=C1CCCC2CCCN12", set(range(1, 10)))):
        mol, groups = _mol_groups(smiles)
        assert groups.get("amidine"), smiles
        assert without_ring_skeleton_amidine(groups, ring) is groups, smiles


def test_one_ring_amidine_and_one_substituent_amidine_keep_the_group():
    """Every amidine of the molecule must be skeletal for the class to be set aside: the ring unit
    of DBU beside a carboximidamide substituent leaves the principal group an amidine."""
    mol, groups = _mol_groups("NC(=N)C1CCC2=NCCCN2CC1")
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    assert without_ring_skeleton_amidine(groups, ring) is groups


@pytest.mark.opsin_gate
def test_the_builder_names_the_fused_system_and_declines_a_substituent_amidine():
    jar_or_skip()
    from orthonym.rules.bridged_fused_pin import build_fused, principal_suffix, selection
    got = build_fused(Chem.MolFromSmiles("C1CCC2=NCCCN2CC1"))
    assert got is not None and got[0] == "2,3,4,6,7,8,9,10-octahydropyrimido[1,2-a]azepine", got
    mol = Chem.MolFromSmiles("NC(=N)C1CCC2=NCCCN2CC1")
    assert principal_suffix(mol, selection.ring_system(mol)) is None


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", NEUTRAL + CATIONS)
def test_the_default_tier_names_the_ring_amidine_and_the_name_reads_back(smiles, name):
    """pin_verified, and OPSIN reads the name back to the input's full InChIKey."""
    jar_or_skip()
    from orthonym import Orthonym
    row = Orthonym(style="pin").name_tiered(smiles)
    assert row["name"] == name and row["tier"] == "pin_verified", row
    assert_full_rt(name, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [("C1CCC2=NCCCN2CC1", NEUTRAL[0][1]),
                                         ("C1CCC2=[NH+]CCCN2CC1", CATIONS[0][1])])
def test_the_best_effort_tier_gives_the_same_name(smiles, name):
    """The fusion name is the PIN, so the wider tier keeps it and does not fall back to the von
    Baeyer '1,8-diazabicyclo[5.4.0]undec-7-ene' it gave before."""
    jar_or_skip()
    row = tier_row(smiles, "best-effort")
    assert row["name"] == name, row
