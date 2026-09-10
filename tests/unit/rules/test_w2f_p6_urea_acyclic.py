""" - acyclic N-substituted urea substituent end-to-end (W2F p6 Task 2).

BB (the Blue Book,33354): the urea substituent is
'(R-carbamoyl)amino'; the bracket escalation [(methylcarbamoyl)amino] follows
the enclosing-marks nesting rule. The shipped unsubstituted analog
'3-(carbamoylamino)propanoic acid' must stay working.
"""
import orthonym


def test_n_methylureido_propanoic_acid():
    assert orthonym.name_compound("CNC(=O)NCCC(=O)O", style="pin") == \
        "3-[(methylcarbamoyl)amino]propanoic acid"


def test_carbamoylamino_regression():
    # shipped analog must stay working
    assert orthonym.name_compound("NC(=O)NCCC(=O)O", style="pin") == \
        "3-(carbamoylamino)propanoic acid"
