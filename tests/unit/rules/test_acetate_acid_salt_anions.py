"""The mono-anion of an acid whose carboxylate is the acetate of 'X-CH2-COO(-)' (X = O, S, N)
(``rules.acid_salt_anions.name_acetate_acid_salt_anion``).

 "Acid salts of polybasic organic acids are named in two ways" (the Blue Book), method (1)
(:31593): "by substitutive nomenclature in which the free acid is cited as a prefix to the name of the anion", and
:31596 "Method (1) generates preferred IUPAC names, except when the structure of the acid salt is unknown"; method (2)
(:31594) is the neutral-salt form with the word 'hydrogen'. The anion is senior to the acid,:18167) and
 (:41261) takes the parent with the anionic centre; with the atom X ending the chain, that parent is the acetate.
The fully ionised name ('2,2',2''-nitrilotriacetate') has more charges than the input and was the name the anion route gave.

The name is a PIN (the maintainers' ruling of 2026-11-01): multiplicative nomenclature is out because an 'ate' and an
'ic acid' are two parents condition (3),:23184), so the free acid is a 'carboxymethyl' prefix of the acetate.

Every name is read back by an independent OPSIN call (the full standard InChIKey).
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from tests.support.rt_assert import name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

ROWS = [
    # the nitrilotriacetic acid mono-anion of the dev2000 population
    ("O=C([O-])CN(CC(=O)O)CC(=O)O", "[bis(carboxymethyl)amino]acetate"),
    ("OC(=O)CNCC(=O)[O-]", "[(carboxymethyl)amino]acetate"),
    ("OC(=O)CN(C)CC(=O)[O-]", "[(carboxymethyl)(methyl)amino]acetate"),
    ("OC(=O)COCC(=O)[O-]", "(carboxymethoxy)acetate"),
    ("OC(=O)CSCC(=O)[O-]", "[(carboxymethyl)sulfanyl]acetate"),
]


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles, name", ROWS)
def test_the_acetate_is_the_parent_and_the_free_acid_a_prefix(smiles, name, tier):
    row = Orthonym(style="pin", **(_emit_tier_flags(tier) if tier != "pin" else {})).name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert name_is_rt_exact(row["name"], smiles), row


@pytest.mark.parametrize("smiles", [
    "OC(=O)CC(=O)[O-]",                 # no atom X: the malonate, named on its own
    "OC(=O)CSC(C)C(=O)[O-]",            # the alpha carbon of the anion carries a methyl group
    "OC(=O)CN1CCN(CC(=O)O)CC1.[Na+]",   # not a single fragment
])
def test_other_shapes_are_not_named_by_this_builder(smiles):
    from rdkit import Chem
    from orthonym.perception.ions import get_ion_sites
    from orthonym.rules.acid_salt_anions import name_acetate_acid_salt_anion
    mol = Chem.MolFromSmiles(smiles)
    sites = get_ion_sites(mol)
    if sites["anions"]:
        assert name_acetate_acid_salt_anion(mol, sites["anions"][0]) == ""


def test_the_fully_ionised_nitrilotriacetate_keeps_its_name():
    row = Orthonym(style="pin").name_tiered("[O-]C(=O)CN(CC(=O)[O-])CC(=O)[O-]")
    assert (row["name"], row["tier"]) == ("2,2',2''-nitrilotriacetate", "pin_verified"), row
