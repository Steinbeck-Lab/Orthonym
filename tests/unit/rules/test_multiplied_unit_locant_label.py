"""The head locants of a multiplicative name number the multiplied parent unit "Multiplicative name
formation", the Blue Book (a); '2,2'-{oxybis[(1E)-ethene-2,1-diyl-4,1-phenylene]}dipropanoic acid (PIN)',:50799).
A locant higher than the chain parent's length belongs to a ring prefix, so the name is no PIN
(validation/spelling/checks_multiplied_locants.py).
"""
import pytest
from rdkit import Chem

from orthonym.validation.spelling.checks_multiplied_locants import multiplied_unit_locant_check

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles,name", [
    ("CC(=O)c1ccc(Oc2ccc(C(C)=O)cc2)cc1", "4,4'-oxybis(1-phenylethan-1-one)"),
    ("ClCC(=O)Nc1ccc(Cc2ccc(NC(=O)CCl)cc2)cc1", "4,4'-methylenebis(2-chloro-N-phenylacetamide)"),
    ("O=CNc1ccc(Oc2ccc(NC=O)cc2)cc1", "4,4'-oxybis(N-phenylformamide)"),
    ("CC(C)(Oc1ccc(Oc2ccc(OC(C)(C)C(=O)O)cc2)cc1)C(=O)O", "4,4'-oxybis(2-methyl-2-phenoxypropanoic acid)"),
    ("OC(=O)Cc1ccc(Oc2ccc(CC(=O)O)c3ccccc23)c2ccccc12", "4,4'-oxybis[2-(naphthalen-1-yl)acetic acid]"),   # brackets
    ("C=C(C(=O)O)c1ccc(Oc2ccc(C(=C)C(=O)O)cc2)cc1", "4,4'-oxybis(2-phenylprop-2-enoic acid)"),
    ("OC(CCl)COc1ccc(C(c2ccc(OCC(O)CCl)cc2)c2ccc(OCC(O)CCl)cc2)cc1",
     "4,4',4''-methylidynetris(1-chloro-3-phenoxypropan-2-ol)"),
])
def test_a_ring_locant_on_a_chain_unit_is_flagged(smiles, name):
    failure = multiplied_unit_locant_check(Chem.MolFromSmiles(smiles), name)
    assert failure is not None and failure.rule == "P-15.3.1.3", name


@pytest.mark.parametrize("smiles,name", [
    ("CC(=O)c1ccc(Oc2ccc(C(C)=O)cc2)cc1", "1,1'-[oxydi(4,1-phenylene)]di(ethan-1-one)"),
    ("OCCCCCCCCCCCCCCCCSSCCCCCCCCCCCCCCCCO", "16,16'-disulfanediyldi(hexadecan-1-ol)"),
    ("OC(=O)c1ccc(Oc2ccc(C(=O)O)c(Br)c2)cc1Br", "4,4'-oxybis(2-bromobenzoic acid)"),                # a ring unit
    ("OC(c1ccc(Oc2ccc(C(O)(C(F)(F)F)C(F)(F)F)cc2)cc1)(C(F)(F)F)C(F)(F)F",
     "2,2'-[oxydi(4,1-phenylene)]bis(1,1,1,3,3,3-hexafluoropropan-2-ol)"),
    ("CN(C)CCOCCN(C)C", "2,2'-oxybis(N,N-dimethylethan-1-amine)"),
    ("O=C(Nc1ccccc1)C(=O)Nc1ccccc1", "N,N'-diphenylethanediamide"),
    ("C", "41,41'-oxybis(hentetracontan-1-ol)"),                     # a stem past the table: not judged
    ("C", "2,2'-oxybis(cyclohexan-1-one)"),                          # a ring parent
    ("C", "3,3'-oxybis(2,5,8-trioxadecane)"),                        # replacement nomenclature: chain of 10
])
def test_other_names_pass(smiles, name):
    assert multiplied_unit_locant_check(Chem.MolFromSmiles(smiles), name) is None, name
