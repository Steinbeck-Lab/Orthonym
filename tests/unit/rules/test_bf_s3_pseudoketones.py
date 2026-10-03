"""Slice S3 (Task S3.4): cyclic anhydrides, esters, amides and imides of the bridged fused
ring system as pseudoketones.

 (the Blue Book) "Cyclic anhydrides, esters and amides are named as
pseudoketones; the resulting names are preferred IUPAC names"; (:32477-:32486)
method (1) for cyclic anhydrides ('2-benzofuran-1,3-dione (PIN)'); cyclic imides
('hexahydro-1H-isoindole-1,3(2H)-dione (PIN)':33849); lactones (:32088); the
indicated and added hydrogen as for ketones (Task S3.3). (c) (:7619) elides "the
terminal letter 'a' in the names of numerical multiplicative prefixes when followed by a
suffix beginning with 'a' or 'o'": 'tetrone' ('tetrahydro-4,8-ethanopyrano[4,3-c]pyran-
1,3,5,7-tetrone (PIN)':32535), and for every suffix the builder spells, 'tetrol'
('benzenehexol (PIN, (not benzenehexaol)':7625) and 'tetramine'
:26348 "'tetramine', not 'tetraamine'"), where main built '-tetraol' and '-tetraamine'.
Every name was read back by OPSIN 2.9.0 to the input's full InChIKey (S3 planning notes,
ledger)."""
import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import build

PSEUDOKETONES = [
    ("O=C1OC(=O)C2C3C=CC(C3)C12", "3a,4,7,7a-tetrahydro-4,7-methano-2-benzofuran-1,3-dione"),
    ("O=C1OC(=O)C2C3CCC(C3)C12", "hexahydro-4,7-methano-2-benzofuran-1,3-dione"),
    ("O=C1NC(=O)C2C3C=CC(C3)C12", "3a,4,7,7a-tetrahydro-1H-4,7-methanoisoindole-1,3(2H)-dione"),
    ("O=C1N(c2ccccc2)C(=O)C2C3C=CC(C3)C12",
     "2-phenyl-3a,4,7,7a-tetrahydro-1H-4,7-methanoisoindole-1,3(2H)-dione"),
    ("O=C1NC(=O)C2C3C=CC(O3)C12", "3a,4,7,7a-tetrahydro-1H-4,7-epoxyisoindole-1,3(2H)-dione"),
    ("O=C1OCC2C3CCC(C3)C12", "hexahydro-4,7-methano-2-benzofuran-1(3H)-one"),
    ("O=C1NCC2C3CCC(C3)C12", "octahydro-1H-4,7-methanoisoindol-1-one"),
    ("O=C1CC2C3CCC(C3)C2C(=O)O1", "hexahydro-1H-5,8-methano-2-benzopyran-1,3(4H)-dione"),
    ("O=C1OC(=O)C23CCCC12C3", "5,6-dihydro-1H,3H,4H-3a,6a-methanocyclopenta[c]furan-1,3-dione"),
    ("O=C1OC(=O)C2C3CCC1C2C(=O)OC3=O", "tetrahydro-4,8-ethanopyrano[4,3-c]pyran-1,3,5,7-tetrone"),
]


@pytest.mark.parametrize("smiles,name", PSEUDOKETONES)
def test_pseudoketone_name(smiles, name):
    res = build(Chem.MolFromSmiles(smiles))
    assert res is not None and res[0] == name, res


ELISION = [
    ("OC1C(O)C2CC1c1c(O)c(O)ccc12", "1,2,3,4-tetrahydro-1,4-methanonaphthalene-2,3,5,6-tetrol"),
    ("NC1C(N)C2CC1c1c(N)c(N)ccc12", "1,2,3,4-tetrahydro-1,4-methanonaphthalene-2,3,5,6-tetramine"),
]


@pytest.mark.parametrize("smiles,name", ELISION)
def test_the_multiplier_a_is_elided_before_any_suffix_beginning_with_a_or_o(smiles, name):
    res = build(Chem.MolFromSmiles(smiles))
    assert res is not None and res[0] == name, res
