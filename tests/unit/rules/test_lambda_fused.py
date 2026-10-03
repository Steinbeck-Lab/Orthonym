""": lambda tokens cited at the beginning of fused-ring names.

1H-1lambda4-1-benzothiophene: the benzo name of (the Blue Book, '1-benzofuran
(PIN)':11827) with the lambda token and the one indicated hydrogen of the mancude system
('1H-1λ4-thiophene (PIN)',,:9167). OPSIN 2.9.0 reads it back to the input's full
InChIKey (it used to be the descriptor name '1H-1lambda4-benzo[b]thiophene', not the PIN).
"""
import pytest
from orthonym.namer import name_compound


@pytest.mark.unit
def test_lambda4_benzothiophene_pin():
    # BB mechanism + indicated-H, on the
    # algorithmic 2-component fusion path (benzo + thiophene).
    assert name_compound("[SH2]1C=CC2=C1C=CC=C2") == "1H-1λ4-1-benzothiophene"


@pytest.mark.unit
def test_standard_fused_unchanged():
    # protect — standard-valence fused system untouched by the lambda guard.
    #: PIN carries the S locant, the Blue Book
    # cites '1-benzothiophene' as the PIN reference name). OPSIN-RT clean.
    assert name_compound("c1ccc2sccc2c1") == "1-benzothiophene"


@pytest.mark.unit
def test_pyridothiazine_lambda_fails_closed():
    # BB example 3lambda4-pyrido[3,2-d][1,3]thiazine: the [1,3]thiazine fusion
    # component is unbuilt -> must stay refused, NEVER the one-ring
    # structure-loss name ('1,3-thiazine') the raw pipeline emits today.
    assert "unknown" in name_compound("N1=CS=CC2=C1C=CC=N2").lower()


@pytest.mark.unit
def test_lambda_fused_never_emits_lambdaless_name():
    # Raw-pipeline contract: the fused namer itself must not return a
    # lambda-less name for a lambda-bearing ring system.
    from rdkit import Chem
    from orthonym.rules.fused_rings import name_fused_heterocycle
    res = name_fused_heterocycle(Chem.MolFromSmiles("N1=CS=CC2=C1C=CC=N2"))
    assert res is None
    res2 = name_fused_heterocycle(Chem.MolFromSmiles("[SH2]1C=CC2=C1C=CC=C2"))
    assert res2 is not None and "λ4" in res2[0]
