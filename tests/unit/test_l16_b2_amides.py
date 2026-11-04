"""An acylated nitrogen on a benzene ring is the amido prefix (lane/l16 b2).

 Substituents of the types -NH-CO-R and -NH-SO2-R (the Blue Book): "When a group having preference for
citation as a principal characteristic group is present, the group R-CO-NH-... of an N-substituted amide is named in two
ways: (1) substitutively, by using a prefix formed by changing the final letter 'e' in the complete name of the amide to
'o'... Method (1) generates preferred IUPAC names." The nitrogen of -N(C6H5)-CO-R is that group ('4-(N-phenylacetamido)
benzoic acid'), not the anilino prefix of (the Blue Book), whose 'N-<R>anilino' takes an
N-substituent that is no acyl ('3-(N-methylanilino)phenol (PIN)',:26308).
"""
import re

import pytest

from orthonym.cli import _tiered_namer
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

_TIERS = ["pin", "best-effort"]


def _row(smiles, tier):
    return _tiered_namer("pin", tier).name_tiered(smiles)


# an acylated nitrogen on a benzene ring that also carries a phenyl: the amido prefix of method (1), at both tiers
_N_ARYL_AMIDO = [
    ("OC(=O)c1ccc(N(C(C)=O)c2ccccc2)cc1", "4-(N-phenylacetamido)benzoic acid"),
    ("OC(=O)c1ccc(N(C(=O)COC)c2ccccc2)cc1", "4-(2-methoxy-N-phenylacetamido)benzoic acid"),
    ("COC(=O)c1ccc(N(C(C)=O)c2ccccc2)cc1", "methyl 4-(N-phenylacetamido)benzoate"),
    ("OC(=O)c1ccc(N(C(=O)c2ccccc2)c2ccccc2)cc1", "4-(N-phenylbenzamido)benzoic acid"),
    ("OC(=O)c1ccc(N(C=O)c2ccccc2)cc1", "4-(N-phenylformamido)benzoic acid"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name", _N_ARYL_AMIDO)
def test_an_acyl_nitrogen_that_carries_a_phenyl_is_the_amido_prefix_not_the_anilino(smiles, name, tier):
    """ (the Blue Book), method (1) 'generates preferred IUPAC names': '4-(N-phenylacetamido)benzoic
    acid', not '4-(N-acetylanilino)benzoic acid' (the book gives no 'N-acylanilino' name)."""
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert "anilino" not in row["name"], row
    assert_full_rt(row["name"], smiles)


# the amide of a three-carbon N-substituent whose acyl carries a ring: the PIN tier names the amide as the acyl chain
