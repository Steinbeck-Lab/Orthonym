"""The N-substituent of a secondary or tertiary amine cites its own stereodescriptor.

``composer._assemble_amine_name`` names each substituent of the amine nitrogen from its atoms; a stereocentre in
one ('N-(1-phenylethyl)') was left out, so the name described a molecule with that centre undefined. The OPSIN
validity gate withdraws such a name, and the molecule abstained at every tier (a secondary amine with a chiral
N-substituent) and, named through its neutral form with the gate off, as an anion or a salt. The descriptor is
cited with the substituent, the Blue Book: stereodescriptors of a substituent group are cited with
it; ``_add_substituent_stereo`` as for every other substituent). The locant of a centre in a substituent name that
carries locants is not derived ('[(R)-1-phenylethyl]' is the round-trip-verified form below the PIN,
substituent_naming._add_substituent_stereo_text), so the default tier still declines these. The suite runs with
the validity gate off, so each row sees the producer itself; every name is read back by an independent OPSIN call.
"""
import pytest

from tests.support.rt_assert import name_best_effort, name_is_rt_exact

ROWS = [
    ("C[C@@H](NC(C)(C)C)c1ccccc1", "2-methyl-N-[(1R)-1-phenylethyl]propan-2-amine"),
    ("CC(N[C@@H](C)c1ccccc1)c1ccccc1", None),
    ("C[C@@H](N[C@@H](CN(C)C)c1ccccc1)c1ccccc1", None),
    ("C[C@H](c1ccccc1)N[C@H](C)c1ccccc1", None),
]


@pytest.mark.parametrize("smiles,expected", ROWS)
def test_amine_with_a_chiral_n_substituent_is_named_with_its_stereo(smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] and name_is_rt_exact(res["name"], smiles), res
    if expected:
        assert res["name"] == expected


def test_anion_with_a_chiral_n_substituent():
    smiles = "C[C@H](C1=CC=CC=C1)[N-][C@@H](CN2CCCCC2)C3=CC=CC=C3"
    res = name_best_effort(smiles)
    assert res["name"] and name_is_rt_exact(res["name"], smiles), res


# (the Blue Book): "descriptors... are placed at the front of the complete name when related to the
# parent structure": the descriptor of a diamine parent stands before the prefixes, not after them.
@pytest.mark.parametrize("smiles,expected", [
    ("CN(C)C[C@@H](NC(C)C)c1ccccc1", "(2S)-N1,N1-dimethyl-2-phenyl-N2-(propan-2-yl)ethane-1,2-diamine"),
    ("C[C@@H](N[C@@H](CN(C)C)c1ccccc1)c1ccccc1",
     "(2R)-N1,N1-dimethyl-2-phenyl-N2-[(1R)-1-phenylethyl]ethane-1,2-diamine"),
])
def test_polyamine_descriptor_stands_at_the_front(smiles, expected):
    res = name_best_effort(smiles)
    assert res["name"] == expected, res
    assert name_is_rt_exact(expected, smiles)
