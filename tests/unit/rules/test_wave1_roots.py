"""Wave 1 root-fix tests — R9: heteroatom-variety order P-44.2.1.8."""
from orthonym.namer import name_compound


def test_r9_morpholine_senior_to_pyrimidine():
    # O (in morpholine) ranks above N (extra in pyrimidine) -> morpholine is the parent.
    out = name_compound("C1COCCN1Cc1cncnc1", style="pin")
    assert out.endswith("morpholine")          # e.g. 4-(pyrimidin-5-ylmethyl)morpholine
    assert "pyrimidin" not in out.split("morpholine")[0][-12:]  # pyrimidine is the substituent
