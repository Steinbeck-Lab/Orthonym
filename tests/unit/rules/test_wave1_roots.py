"""Wave 1 root-fix tests — R9: heteroatom-variety order P-44.2.1.8."""
from orthonym.namer import name_compound


def test_r9_morpholine_senior_to_pyrimidine():
    # O (in morpholine) ranks above N (extra in pyrimidine) -> morpholine is the parent.
    out = name_compound("C1COCCN1Cc1cncnc1", style="pin")
    assert out.endswith("morpholine")          # e.g. 4-(pyrimidin-5-ylmethyl)morpholine
    assert "pyrimidin" not in out.split("morpholine")[0][-12:]  # pyrimidine is the substituent


# ---------------------------------------------------------------------------
# R8a — acyclic/aromatic hydrazide suffix + terminal-1 elision (P-66.3.1.1)
# ---------------------------------------------------------------------------

def test_r8a_hydrazide_pins():
    """P-66.3.1.1: hydrazide suffix = chain stem + hydrazide (no locant-1).
    - pentanehydrazide: C5 chain; suffix 'hydrazide', terminal -> locant-1 elided,
      stem+ane kept (h is consonant -> no vowel elision).
    - acetohydrazide: retained acyl-stem for C2 (aceto-).
    - formohydrazide: retained acyl-stem for C1 (formo-).
    - benzohydrazide: benzene ring with C(=O)NN suffix -> retained PIN 'benzohydrazide'.
    """
    assert name_compound("CCCCC(=O)NN", style="pin") == "pentanehydrazide"
    assert name_compound("CC(=O)NN", style="pin") == "acetohydrazide"
    assert name_compound("O=CNN", style="pin") == "formohydrazide"
    assert name_compound("O=C(NN)c1ccccc1", style="pin") == "benzohydrazide"
