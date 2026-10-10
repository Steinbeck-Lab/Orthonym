"""Leads program L3, item N8a: the numbering check still fails the perception-order spellings.

The ring-nitrile producer now numbers the ring from the suffix, so the engine no longer emits
'6-(4-ethylphenoxy)cyclohexane-1-carbonitrile' (the integration row that used to pin its label
lowering was removed with it, tests/integration/test_pin_spelling_label.py). The check that lowered
it is the label's second line of defence: it must keep failing the old spelling and passing the PIN.
 'NUMBERING' (the Blue Book): (c) "principal characteristic groups and free valences
(suffixes)" (:3256), then (f) "detachable alphabetized prefixes, all considered together in a series
of increasing numerical order" (:3301).
"""
import pytest
from rdkit import Chem

from orthonym.validation.pin_spelling import check_pin_spelling
from tests.support.rt_assert import name_is_rt_exact


def _rules(smiles, name):
    return [f.rule for f in check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True)]


ROWS = [
    ("CCc1ccc(OC2CCCCC2C#N)cc1", "6-(4-ethylphenoxy)cyclohexane-1-carbonitrile",
     "2-(4-ethylphenoxy)cyclohexane-1-carbonitrile"),
    ("C1(C#N)C(n2nc(C)c(Cl)c2C)CC(CCC)CC1",
     "6-(4-chloro-3,5-dimethyl-1H-pyrazol-1-yl)-4-propylcyclohexane-1-carbonitrile",
     "2-(4-chloro-3,5-dimethyl-1H-pyrazol-1-yl)-4-propylcyclohexane-1-carbonitrile"),
]


@pytest.mark.parametrize("smiles, old, pin", ROWS)
def test_the_perception_order_spelling_fails_and_the_pin_passes(smiles, old, pin):
    assert name_is_rt_exact(old, smiles) and name_is_rt_exact(pin, smiles)   # both are the molecule
    assert "P-14.4" in _rules(smiles, old)
    assert _rules(smiles, pin) == []
