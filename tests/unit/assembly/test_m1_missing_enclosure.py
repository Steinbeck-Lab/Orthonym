import pytest
from orthonym import Orthonym

@pytest.fixture(scope="module")
def eng():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)

# W1 — bis() around a complex ring substituent (composer.py count>1 branch)
def test_w1_bis_3_chlorophenyl_enclosed(eng):
    smi = "C[C@@H]1C(=O)N[C@H](c2cccc(Cl)c2)C2(C(=O)Nc3cc(Cl)ccc32)[C@@H]1c1cccc(Cl)c1"
    name = eng.name(smi)
    assert name is not None and "unknown" not in name.lower()
    # the multiplied complex substituent must be enclosed
    assert "bis(3-chlorophenyl)" in name
    assert "bis3-chlorophenyl" not in name

# W3 — [] escalation around a substituent that already contains () (count==1 branch)
def test_w3_methanesulfonyl_quinazolinyl_enclosed(eng):
    smi = "COc1ccc2c(c1)C1(CC1)CN(c1ncnc3ccc(S(C)(=O)=O)cc13)C2"
    name = eng.name(smi)
    assert name is not None and "unknown" not in name.lower()
    assert "[6-(methanesulfonyl)quinazolin-4-yl]" in name

# NEGATIVE — a simple ring substituent must stay bare (no over-enclosure)
def test_simple_substituent_stays_bare(eng):
    # 1,3-dimethyl on a ring: 'methyl' is simple, must NOT gain parens
    name = eng.name("Cc1cccc(C)c1")
    assert name is not None
    assert "(methyl)" not in name
    assert "methyl" in name
