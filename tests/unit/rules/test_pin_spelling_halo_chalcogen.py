"""Spelling check (validation/spelling/checks_halo_chalcogen.py): a halogen on a λ4/λ6 S, Se or Te
written as a substituent prefix ('pentafluoro-λ6-sulfanyl') has no PIN in the book. "Halogen atoms
attached to heteroatoms": halo prefixes are for B, Al, In, Ga, Tl, Si, Ge, Sn, Pb and Bi,
the Blue Book); (:25876) "acid halides or esters of inorganic acids may be created when the
halogen atoms are linked to phosphorus or chalcogen atoms. Names must be based on the seniority of classes";
 "Sulfanes, selanes, and tellanes" (:39527) names λ chalcogen compounds on the λ4/λ6 parent hydrides.
The name is kept below the PIN label.
"""
import pytest
from rdkit import Chem

from orthonym.validation.pin_spelling import check_pin_spelling, registered_rules
from orthonym.validation.spelling.checks_halo_chalcogen import halo_lambda_chalcogenyl_prefix_check

from tests.support.pin_tiers import name_breadth, name_default

pytestmark = pytest.mark.unit

BAD = [
    ("FS(F)(F)(F)(F)C1CCCC2CCCCC12", "1-(pentafluoro-λ6-sulfanyl)decahydronaphthalene"),
    ("ClS(Cl)(Cl)C1CCCC2CCCCC12", "1-(trichloro-λ4-sulfanyl)decahydronaphthalene"),
    ("CC(=O)c1ccc(S(F)(F)(F)(F)F)cc1", "1-[4-(pentafluoro-λ6-sulfanyl)phenyl]ethan-1-one"),
    ("CS(F)(F)(F)(Cl)C1CCCCC1", "[chloro(trifluoro)methyl-λ6-sulfanyl]cyclohexane"),
    ("FS(F)(c1ccccc1)c1ccccc1", "1,1'-(difluoro-λ4-sulfanediyl)dibenzene"),
]


@pytest.mark.parametrize("smiles,name", BAD)
def test_the_halo_lambda_prefix_is_flagged(smiles, name):
    failure = halo_lambda_chalcogenyl_prefix_check(Chem.MolFromSmiles(smiles), name)
    assert failure is not None and failure.rule == "P-61.3.2.2", failure


@pytest.mark.parametrize("smiles,name", [
    ("CS(F)(F)(F)(F)F", "pentafluoro(methyl)-λ6-sulfane"),          # the parent spelling is not checked
    ("O=S(Cl)c1ccccc1", "benzenesulfinyl chloride"),                 # an acid halide: S=O
    ("O=S(=O)(F)c1ccc(C(C)=O)cc1", "4-acetylbenzene-1-sulfonyl fluoride"),
    ("CC(=O)c1ccc(SC)cc1", "1-[4-(methylsulfanyl)phenyl]ethan-1-one"),   # a bivalent S
    ("CS(C)(C)(C)(C)CC(C)=O", "1-(pentamethyl-λ6-sulfanyl)propan-2-one"),     # no halogen on the S
    ("ClCC(=O)c1ccc(S(C)(C)(C)(C)C)cc1", "2-chloro-1-[4-(pentamethyl-λ6-sulfanyl)phenyl]ethan-1-one"),
])
def test_other_words_and_structures_pass(smiles, name):
    assert halo_lambda_chalcogenyl_prefix_check(Chem.MolFromSmiles(smiles), name) is None


def test_the_check_is_registered():
    assert "P-61.3.2.2" in registered_rules()
    mol = Chem.MolFromSmiles("FS(F)(F)(F)(F)C1CCCC2CCCCC12")
    assert any(f.rule == "P-61.3.2.2" for f in check_pin_spelling(mol, BAD[0][1]))


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", BAD[:3])
def test_the_name_is_kept_below_the_pin(smiles, name):
    default = name_default(smiles)
    assert default.get("tier") != "pin_verified" and not default.get("is_pin"), default
    breadth = name_breadth(smiles)
    assert breadth.get("name") == name and breadth.get("tier") == "systematic_verified", breadth
