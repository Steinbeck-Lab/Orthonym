""" a review-1 — ring-attached sulfone/sulfoxide substituent is the acid-stem PIN.

: a ring-attached ``R-SO2-`` / ``R-SO-`` substituent is named as the
acid-stem oxide form (``methanesulfonyl`` / ``benzenesulfonyl`` / ``methanesulfinyl``),
NOT the ``alkyl``+``sulfonyl`` concatenation (``methylsulfonyl``). The benzene-parent
path already did this via name_chalcogen_oxide_substitutive; benzene.py's
_identify_sulfur_group (the substituted-benzene / phenol / benzoic-acid path) did not.
Every corrected form OPSIN-round-trips to the same structure (spelling-only PIN fix).
"""
import pytest

from orthonym import name_compound


@pytest.mark.integration
@pytest.mark.parametrize("smiles, expected", [
    # substituted benzoic acid (the reported case)
    ("O=S(=O)(C)c1ccccc1C(=O)O", "2-(methanesulfonyl)benzoic acid"),
    ("O=S(C)c1ccccc1C(=O)O", "2-(methanesulfinyl)benzoic acid"),
    ("O=S(=O)(CC)c1ccccc1C(=O)O", "2-(ethanesulfonyl)benzoic acid"),
    ("O=S(=O)(c1ccccc1)c1ccccc1C(=O)O", "2-(benzenesulfonyl)benzoic acid"),
    ("OC(=O)c1ccc(S(=O)(=O)C)cc1", "4-(methanesulfonyl)benzoic acid"),
    ("OC(=O)c1ccc(S(=O)C)cc1", "4-(methanesulfinyl)benzoic acid"),
    # phenol parent (same producer)
    ("O=S(=O)(C)c1ccccc1O", "2-(methanesulfonyl)phenol"),
    # benzene parent (must remain correct -- no regression)
    ("O=S(=O)(C)c1ccccc1", "(methanesulfonyl)benzene"),
])
def test_ring_sulfone_sulfoxide_is_acid_stem_pin(smiles, expected):
    assert name_compound(smiles) == expected, name_compound(smiles)


@pytest.mark.integration
def test_no_methyl_prefixed_sulfonyl_on_aromatic():
    # the non-PIN concatenated form must never appear on these aromatic parents
    for smi in ("O=S(=O)(C)c1ccccc1C(=O)O", "OC(=O)c1ccc(S(=O)(=O)C)cc1",
                "O=S(=O)(C)c1ccccc1O"):
        nm = name_compound(smi)
        assert "methylsulfonyl" not in nm and "methylsulfinyl" not in nm, nm
