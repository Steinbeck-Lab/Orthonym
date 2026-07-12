"""P-66.1.6.1.1.3 - N-substituted urea substituent on a fused-aromatic acid parent
(W2F p6 Task 3).

BB P-66.1.6.1.1.3 (BlueBookV2.md:33338,33354); naphthalene fixed fusion numbering
(alpha 1,4,5,8 / beta 2,3,6,7) + P-31.1.4.3.4 low-locants-to-suffix -> acid=2,
urea-N=1. The plain-amino analog must stay working.
"""
import orthonym


def test_urea_on_naphthalene_acid():
    assert orthonym.name_compound("OC(=O)c1ccc2ccccc2c1NC(=O)NC", style="pin") == \
        "1-[(methylcarbamoyl)amino]naphthalene-2-carboxylic acid"


def test_unsubstituted_urea_on_naphthalene():
    assert orthonym.name_compound("OC(=O)c1ccc2ccccc2c1NC(=O)N", style="pin") == \
        "1-(carbamoylamino)naphthalene-2-carboxylic acid"


def test_plain_amino_naphthalene_regression():
    assert orthonym.name_compound("OC(=O)c1ccc2ccccc2c1N", style="pin") == \
        "1-aminonaphthalene-2-carboxylic acid"
