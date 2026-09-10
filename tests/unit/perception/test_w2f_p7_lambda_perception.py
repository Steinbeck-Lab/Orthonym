"""W2F-P7 Task 2: λ-hydride phosphorus perception.

``is_lambda_hydride_phosphorus`` fires for a NEUTRAL, radical-free, acyclic P
whose bonding number is 5 (nonstandard) AND whose every non-attachment bond is
to hydrogen (the -PH4 hydride). It must EXCLUDE the phosphoryl/phosphonic P=O
(also valence 5 but with O neighbours), a charged [PH4+], and a ring P.
"""

from rdkit import Chem

from orthonym.rules.lambda_convention import nonstandard_bonding_number
from orthonym.perception.lambda_hydride import is_lambda_hydride_phosphorus


def _p_idx(smi):
    m = Chem.MolFromSmiles(smi)
    return m, [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "P"][0]


def test_ph4_bonding_number_is_5():
    mol, p = _p_idx("C[PH4]")
    assert nonstandard_bonding_number(mol, p) == 5


def test_ph4_is_lambda_hydride():
    mol, p = _p_idx("C[PH4]")
    assert is_lambda_hydride_phosphorus(mol, p)


def test_phosphoryl_excluded():
    # OP(O)(O)=O: valence 5 but NOT a lambda-hydride (all-H check excludes it)
    mol, p = _p_idx("OP(O)(O)=O")
    assert not is_lambda_hydride_phosphorus(mol, p)


def test_phosphine_oxide_excluded():
    # CP(C)(C)=O: valence 5 with organyl + =O -> not a lambda-hydride
    mol, p = _p_idx("CP(C)(C)=O")
    assert not is_lambda_hydride_phosphorus(mol, p)


def test_standard_phosphanyl_not_lambda_hydride():
    # -PH2 bonding number 3 (standard) -> not a lambda hydride
    mol, p = _p_idx("CCP")
    assert not is_lambda_hydride_phosphorus(mol, p)


def test_charged_phosphonium_excluded():
    # a charged P (phosphonium) must never carry a lambda descriptor
    mol, p = _p_idx("C[PH3+]")
    assert not is_lambda_hydride_phosphorus(mol, p)
