"""A ring Si atom of lowered valence is cited by the lambda convention alone, not also as an '-ylidene'.

A silylene drawn ``[Si]`` in a ring (two ring bonds, two radical electrons) is the atom of its parent hydride with
a valence of two, which the lambda convention writes: '9λ2-sila...' "Nonstandard bonding numbers", the Blue Book, the symbol λn cited with the locant of the skeletal
atom prefix). The floor's radical rung named the whole heavy-atom set as an '-ylidene' prefix with the centre as its
attachment atom and gave 'spiro[2λ2-sila...]-2-ylidene', which cites the two electrons twice and which OPSIN reads
as another structure; the molecule abstained. The name of the molecule itself with the lambda convention
(assembly.radical_floor) is read back, radical graph included, by an independent OPSIN call.
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

ROWS = [
    ("C1=CC=C2C(=C1)C3=CC=CC=C3C24C5=CC=CC=C5[Si]C6=CC=CC=C46",
     "spiro[2λ2-silatricyclo[8.4.0.0^3,8]tetradeca-1(14),3,5,7,10,12-hexaene-9,9'-fluorene]"),
    ("C1=CC=C2C(=C1)C3=CC=CC=C3C24C5=C(C6=C(C=C5)SC7=CC=CC=C76)[Si]C8=CC=CC=C48", None),
    ("C1=CC=C2C(=C1)C3=CC=CC=C3C24C5=CC=CC=C5[Si]C6=C4C=C7C(=C6)C8=CC=CC=C8O7", None),
]


@pytest.mark.parametrize("smiles,expected", ROWS)
def test_ring_silylene_is_named_with_the_lambda_convention_only(smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] and "ylidene" not in res["name"] and "λ2-sila" in res["name"], res
    assert name_is_rt_exact(res["name"], smiles), res
    if expected:
        assert res["name"] == expected
