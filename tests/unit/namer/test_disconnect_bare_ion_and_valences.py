"""A bonded metal drawing is named on its ionic drawing when the metal ends as a bare ion that RDKit gives a radical
electron, when it takes a common oxidation state, and when a Group 1 or 2 metal is bonded to As, Sb or Te
(still-unnamed PubChem plan of 2026-11-11, Topics 5 and 6).

The count of atoms and charges (``perception/metals.py::atom_and_charge_ledger_ok``) compared the radical electrons
of the metal, and RDKit gives a bare ``[Fe+3]``, ``[Ti+3]``, ``[Cu+2]`` or ``[Cr+3]`` one radical electron from its
default valence, so ``Cl[Fe](Cl)Cl`` had no name while ``[Fe+3].[Cl-].[Cl-].[Cl-]`` had ``iron(III) chloride``. A
monoatomic ion's name carries no radical. The label is ``systematic_verified``: the drawing as written is not the
drawing named (the contributor guide, scope), and a name of a compound without carbon is never a PIN, the Blue Book).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.perception.metals import atom_and_charge_ledger_ok, single_bond_metal_heteroatom_sites

pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module")
def best_effort():
    return Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)


@pytest.mark.parametrize("smiles, name", [
    ("Cl[Ti](Cl)Cl", "titanium(III) chloride"),
    ("Cl[Fe](Cl)Cl", "iron(III) chloride"),
    ("Br[Fe](Br)Br", "iron(III) bromide"),
    ("[OH-].Cl[Fe+]Cl", "iron(III) dichloride hydroxide"),
    ("Cl[Cu]Cl", "copper(II) chloride"),
    ("Cl[Cr](Cl)Cl", "chromium(III) chloride"),
])
def test_bare_ion_radical_is_not_compared(best_effort, smiles, name):
    row = best_effort.name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "systematic_verified")


@pytest.mark.parametrize("smiles, name", [
    ("Cl[Ru](Cl)(Cl)Cl", "ruthenium(IV) chloride"),
    ("F[Ru](F)(F)(F)(F)F", "ruthenium(VI) fluoride"),
    ("Cl[Ti]Cl", "titanium(II) chloride"),
    ("Cl[Ni](Cl)Cl", "nickel(III) chloride"),
    ("F[Pt](F)(F)(F)(F)F", "platinum(VI) fluoride"),
    ("Cl[V]Cl", "vanadium(II) chloride"),
    ("I[Sm]I", "samarium(II) iodide"),
    ("Cl[Pd](Cl)(Cl)Cl", "palladium(IV) chloride"),
])
def test_common_oxidation_states(best_effort, smiles, name):
    row = best_effort.name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "systematic_verified")


@pytest.mark.parametrize("smiles, name", [
    ("[Na][TeH]", "sodium tellanide"),
    ("[Li][As](F)F", "lithium difluoroarsanide"),
    ("[Mg]([TeH])[TeH]", "magnesium bis(tellanide)"),
    ("[K][AsH2]", "potassium arsanide"),
])
def test_group_1_and_2_salts_of_metalloid_anions(best_effort, smiles, name):
    row = best_effort.name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "systematic_verified")


@pytest.mark.parametrize("smiles", ["Cl[Fe](Cl)Cl", "Cl[Ru](Cl)(Cl)Cl", "[Na][TeH]"])
def test_no_pin_label(smiles):
    assert Orthonym().name_tiered(smiles)["tier"] != "pin_verified"


def test_a_metal_that_keeps_a_bond_is_still_compared_on_radicals():
    """Only a metal that ends with no bond, and whose radical electrons are the ones RDKit infers for the bare ion,
    is not compared: a metal that keeps a bond, or a radical written on the bonded metal, is compared as before."""
    from orthonym.assembly.metal_offer import _break_bonds
    from orthonym.perception.metals import _bare_ion_bookkeeping_radicals
    before = Chem.MolFromSmiles("Cl[Fe](Cl)Cl")
    sites = single_bond_metal_heteroatom_sites(before)
    after = _break_bonds(before, sites)
    assert after.GetAtomWithIdx(1).GetNumRadicalElectrons() == 1      # RDKit's bare [Fe+3]
    assert atom_and_charge_ledger_ok(before, after, sites)
    bonded = Chem.MolFromSmiles("Cl[Fe+2]")
    assert not _bare_ion_bookkeeping_radicals(before.GetAtomWithIdx(1), bonded.GetAtomWithIdx(1))
    radical_before = Chem.MolFromSmiles("Cl[Fe]([Cl])Cl")
    radical_before.GetAtomWithIdx(1).SetNumRadicalElectrons(1)
    assert not _bare_ion_bookkeeping_radicals(radical_before.GetAtomWithIdx(1), after.GetAtomWithIdx(1))


def test_the_metalloid_partner_is_for_group_1_and_2_only():
    assert single_bond_metal_heteroatom_sites(Chem.MolFromSmiles("[Na][TeH]"))
    assert single_bond_metal_heteroatom_sites(Chem.MolFromSmiles("Cl[Fe][TeH]")) is None
    assert single_bond_metal_heteroatom_sites(Chem.MolFromSmiles("[Na][SiH3]")) is None


def test_no_rewrite_beside_a_chemical_radical(best_effort):
    """CID 87702906 (a charge-unbalanced record with a silylene beside Cl[Zr]Cl): every read-back of a molecule with a
    chemical radical compares the whole graph (validation/radical_identity.py), and the graph of the ionic drawing is
    not the input's, so no check could confirm the name against the drawing as written: no DISCONNECT name."""
    smiles = ("CC1=CC2=C([CH-]1)C=C(C(=C2C3=CC=CC=C3)C)C.CC1=CC2=C([CH-]1)C=C(C(=C2C3=CC=CC=C3)C)C."
              "C1=CC=C2C(=C1)C3=CC=C[C-]=C3[Si]2.Cl[Zr]Cl")
    row = best_effort.name_tiered(smiles)
    assert row["tier"] == "abstain" or "zirconium(2+)" not in (row["name"] or "")
