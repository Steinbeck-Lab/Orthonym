""" — heteroatom-rooted ether substituent on an N-heteroarene ring.

-O-R / -S-R / -Se-R / -Te-R substituents on a heteroarene ring were NOT named:
classify_substituent counted the arm carbons and DROPPED the O/S/Se root, so the
coverage check declined the candidate -> unknown (methoxypyridine, methylsulfanyl-
pyridine, and the triazine flagship all failed). The get_heterocycle_substituents
loop now detects a single-bond O/S/Se/Te ether root and emits the (R)oxy /
(R)sulfanyl / (R)selanyl / (R)tellanyl prefix (P-63.2.2 / P-63.2.5): the O case
uses the fragment namer's contracting O-attach path (methoxy, not methyloxy); the
chalcogens name the arm + stem with enclosure for a complex arm. Fail-closed.
"""
import pytest
from orthonym.namer import name_compound


@pytest.mark.parametrize("smi,pin", [
    ("COc1ccccn1", "2-methoxypyridine"),
    ("CCOc1ccccn1", "2-ethoxypyridine"),
    ("COc1ccc(OC)nc1", "2,5-dimethoxypyridine"),
    ("CSc1ccccn1", "2-(methylsulfanyl)pyridine"),
    ("CSc1ncncn1", "2-(methylsulfanyl)-1,3,5-triazine"),
    ("C=CCSc1ccccn1", "2-[(prop-2-en-1-yl)sulfanyl]pyridine"),
])
def test_heteroaryl_ether_substituent(smi, pin):
    assert name_compound(smi) == pin


def test_triazine_flagship():
    # BP-3 §3.5 flagship: ring poly-amine N-locants + methylsulfanyl ring prefix.
    # (N2/N4 assignment follows the ring numbering; RT-verified either way.)
    assert name_compound("CSc1nc(NC2CC2)nc(NC(C)(C)C)n1") == \
        "N2-cyclopropyl-N4-tert-butyl-6-(methylsulfanyl)-1,3,5-triazine-2,4-diamine"


@pytest.mark.parametrize("smi,pin", [
    # carbon-rooted + halogen substituents unaffected
    ("Cc1ccccn1", "2-methylpyridine"),
    ("Clc1ccccn1", "2-chloropyridine"),
    # O-ether on benzene (different handler) unaffected
    # a review RISK 7: bare anisole IS the PIN (the Blue Book / the Blue Book)
    ("COc1ccccc1", "anisole"),
])
def test_regression_guards(smi, pin):
    assert name_compound(smi) == pin
