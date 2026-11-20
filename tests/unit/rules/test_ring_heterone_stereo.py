"""Ring sulfoxides with stereo are named as ring heterones (, ChEMBL 37 ring sulfoxides, 2026-11-19).

 (the Blue Book): in a ring, the oxygen atom on a ring sulfur is "expressed substitutively by the suffix
'-one'... in which the sulfur atoms are designated as lambda4 or lambda6 atoms... This method generates preferred
IUPAC names"., Table 2.3 (:8196): 'thiomorpholine (S instead of O) (PIN)'. (:44639): the descriptors
of the parent are cited at the front of the name.

``rules.ring_chalcogen_oxide._name_ring_heterone`` declined every input with stereo, so a chiral ring sulfoxide fell to
names below the PIN, and to two different parents by spelling ('1-oxo-1,4-thiazinane' and '1-oxothiomorpholine'). It
now names stereo on ring atoms (the labels the naming pipeline gave them); stereo in a substituent still declines. A
charge-separated drawing ('[S@@+]([O-])', as ChEMBL draws a sulfoxide) keeps the name of its own drawing:
``assembly.sulfoxide_drawing`` names the S=O drawing only for a drawing that gets no name as written."""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("C1[S@@](=O)C(C)(CN(C1)Cc1ccncc1)C", "(1R)-2,2-dimethyl-4-[(pyridin-4-yl)methyl]-1λ4-thiomorpholin-1-one"),
    ("O=[S@@]1CCN(C)C[C@@H]1C", "(1R,2S)-2,4-dimethyl-1λ4-thiomorpholin-1-one"),
    ("O=[S@]1CCCC[C@H]1C", "(1S,2R)-2-methyl-1λ4-thian-1-one"),
    # (j): the numbering that ties on every other rule gives R the lower locant ('(2R,4S)-2,4-difluoropentane')
    ("C[C@@H]1C[C@H](C)CS(=O)(=O)C1", "(3R,5S)-3,5-dimethyl-1λ6-thiane-1,1-dione"),
])
def test_a_ring_sulfoxide_with_stereo_is_a_heterone(smiles, name):
    for tier in ("pin", "best-effort"):
        row = tier_row(smiles, tier)
        assert (row["name"], row["tier"]) == (name, "pin_verified"), (tier, row)
    assert_full_rt(name, smiles)


@pytest.mark.parametrize("smiles,name,tier", [
    # review 2026-11-20: a selenium ring oxide keeps the PIN another producer gives it
    ("O=[Se@@]1CC[C@H](N)C1", "(1R,3S)-1-oxo-1λ4-selenolan-3-amine", "pin_verified"),
])
def test_ring_stereo_widening_is_for_sulfur_only(smiles, name, tier):
    row = tier_row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, tier), row


def test_stereo_in_a_substituent_still_declines_the_heterone():
    from rdkit import Chem

    from orthonym.rules.ring_chalcogen_oxide import _name_ring_heterone
    assert _name_ring_heterone(Chem.MolFromSmiles("O=[S@]1CCN([C@@H](C)CC)CC1")) is None
