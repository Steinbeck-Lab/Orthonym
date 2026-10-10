"""Leads program L6, item 43a (proposal for the owner of assembly/substituent_prefix_forms.py):
a substituted or ring acyl on a sulfur is an '(acyl)sulfanyl' prefix.

``get_sulfanyl_prefix`` named the acyl of a thioester S-side only for an unsubstituted linear acyl
or plain 'benzoyl' (``_acyl_on_chalcogen_name``, v1), so '(2R)-2-acetamido-3-[(2,3-dihydroxy-
propanoyl)sulfanyl]propanoic acid' was never built by the substitutive path: the PIN tier abstained
and the wider tiers named it through the decomposition fallback.

 (the Blue Book): "Parentheses are used around compound... and complex prefixes";
an acyl that carries a substituent is a compound prefix, enclosed before the chalcogen stem
('[(2,3-dihydroxypropanoyl)sulfanyl]'); the unsubstituted '(acetylsulfanyl)carbonyl' (:18128) stays
as it was. (:32991) for the 'acetamido' prefix of the same parent.
"""
import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers

PINS = [
    ("CC(=O)N[C@@H](CSC(=O)C(O)CO)C(=O)O",
     "(2R)-2-acetamido-3-[(2,3-dihydroxypropanoyl)sulfanyl]propanoic acid"),
    ("CC(=O)N[C@@H](CSC(=O)CO)C(=O)O",
     "(2R)-2-acetamido-3-[(hydroxyacetyl)sulfanyl]propanoic acid"),
    ("CC(=O)N[C@@H](CSC(=O)CCl)C(=O)O",
     "(2R)-2-acetamido-3-[(chloroacetyl)sulfanyl]propanoic acid"),
    ("CC(=O)N[C@@H](CSC(=O)C(C)O)C(=O)O",
     "(2R)-2-acetamido-3-[(2-hydroxypropanoyl)sulfanyl]propanoic acid"),
    ("CC(=O)N[C@@H](CSC(=O)C=C)C(=O)O",
     "(2R)-2-acetamido-3-[(prop-2-enoyl)sulfanyl]propanoic acid"),
    ("OCC(=O)SCCC(=O)O", "3-[(hydroxyacetyl)sulfanyl]propanoic acid"),
    # unchanged: an unsubstituted acyl keeps its single pair of marks
    ("CC(=O)N[C@@H](CSC(=O)CC)C(=O)O", "(2R)-2-acetamido-3-(propanoylsulfanyl)propanoic acid"),
    ("CC(=O)SCCC(=O)O", "3-(acetylsulfanyl)propanoic acid"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", PINS, ids=[s for s, _ in PINS])
def test_the_acylsulfanyl_prefix_of_a_substituted_acyl(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
