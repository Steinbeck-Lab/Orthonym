"""A salt whose nuclide labels all sit on one-atom anions names each such anion by the name of the lone ion
(still-unnamed PubChem plan of 2026-11-11, Topic 8).

 NAMES, (the Blue Book): "The name of an isotopically substituted compound is formed by adding or
inserting the nuclide symbol(s) enclosed in parentheses... before the part of the compound that is isotopically
substituted". OPSIN 2.9.0 reads no nuclide on the class word ('potassium (18F)fluoride' is unreadable) and reads the
name of the lone ion ('(18F)fluoranide', Table 8.1 'deuteride', ``rules.metal_atoms.lone_atom_name_and_basis``). That
spelling is not the PIN spelling, so the name is ``systematic_verified`` at most.
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module")
def best_effort():
    return Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)


@pytest.mark.parametrize("smiles, name", [
    ("[K+].[18F-]", "potassium (18F)fluoranide"),
    ("[Na+].[2H-]", "sodium deuteride"),
    ("[Li+].[3H-]", "lithium tritide"),
    ("C[N+](C)(C)C.[36Cl-]", "N,N,N-trimethylmethanaminium (36Cl)chloranide"),
    ("CC[N+](CC)(CC)CC.[18F-]", "N,N,N-triethylethanaminium (18F)fluoranide"),
    ("[Na+].[Na+].[35S-2]", "disodium (35S)sulfanediide"),
])
def test_labelled_counter_ion(best_effort, smiles, name):
    row = best_effort.name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "systematic_verified")


@pytest.mark.parametrize("smiles", ["C[N+](C)(C)C.[36Cl-]", "CC[N+](CC)(CC)CC.[18F-]"])
def test_the_default_tier_gives_no_pin(smiles):
    assert Orthonym().name_tiered(smiles)["tier"] != "pin_verified"


def test_unlabelled_salt_keeps_its_pin():
    row = Orthonym().name_tiered("CC[N+](CC)(CC)CC.[Cl-]")
    assert (row["name"], row["tier"]) == ("N,N,N-triethylethanaminium chloride", "pin_verified")
