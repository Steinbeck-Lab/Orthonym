"""A Group 1 metal drawn bonded to an oxygen, nitrogen or sulfur atom is named on the ionic drawing of the same
salt (program 22, 2026-11-09).

`data/metal_scope.py` admits Li, Na, K, Rb and Cs (not Fr: no row measured) to ``DISCONNECT``, as zinc and magnesium already are: the
bonded drawing 'C(=O)O[Li]' and the ionic drawing 'C(=O)[O-].[Li+]' have one standard InChIKey, so the name of the
ionic drawing ships when the count of atoms and charges agrees (`perception/metals.py::atom_and_charge_ledger_ok`).
The label is ``systematic_verified``: the drawing as written is not the drawing named (the contributor guide, scope). A salt is
named cation then anion: Salts (the Blue Book), 'sodium acetate (PIN)' (:43611); 'ethoxide' and
'tert-butoxide' are retained anion names,:28180).
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module")
def best_effort():
    return Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)


@pytest.mark.parametrize("smiles, name", [
    ("ClC1=C(N=NC(=C1)Cl)C(=O)O[Li]", "lithium 4,6-dichloropyridazine-3-carboxylate"),
    ("N1[C@@H](C1)C(=O)O[Li]", "lithium (2S)-aziridine-2-carboxylate"),
    ("OCCCCCCCCCCCC(=O)O[Na]", "sodium 12-hydroxydodecanoate"),
    ("FC(C(=O)O[Na])(C(F)(F)F)F", "sodium 2,2,3,3,3-pentafluoropropanoate"),
    ("COCCOCCO[Li]", "lithium 2-(2-methoxyethoxy)ethoxide"),
    ("C[Si](C)(C)N([Si](C)(C)C)[K]", "potassium bis(trimethylsilyl)azanide"),
    ("[Na]OS(=O)CCC(=O)OC", "sodium 3-methoxy-3-oxopropane-1-sulfinate"),
    ("F[Cs]", "caesium fluoride"),
])
def test_group1_bonded_drawing_named_as_its_salt(best_effort, smiles, name):
    row = best_effort.name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "systematic_verified")


@pytest.mark.parametrize("smiles", ["C(=O)O[Li]", "CCO[Na]"])
def test_no_pin_label_for_a_bonded_drawing(smiles):
    """The PIN tier names the drawing as written or declines; a name through another drawing is never a PIN."""
    row = Orthonym().name_tiered(smiles)
    assert row["tier"] != "pin_verified", row


@pytest.mark.parametrize("smiles", [
    "CCCC[Li].CO[Li]",        # a carbon-lithium bond anywhere refuses the rewrite of the whole molecule
    "[Li][Na]",               # metal-metal bond
    "C1CO[Li]O1",             # a lithium with two bonds would be Li(2+)
])
def test_no_ionic_name_for_drawings_the_rewrite_refuses(best_effort, smiles):
    """These drawings have no ionic drawing with a +1 Group 1 cation for every bond broken; the rewrite does not
    run, so no salt name of an ionic drawing is given for them."""
    from orthonym.perception.metals import single_bond_metal_heteroatom_sites
    from rdkit import Chem
    assert not single_bond_metal_heteroatom_sites(Chem.MolFromSmiles(smiles))
