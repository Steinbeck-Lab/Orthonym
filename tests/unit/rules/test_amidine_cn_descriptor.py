"""The C=N of an acyclic amidine is cited with its E/Z descriptor, though the InChIKey does not encode it.

IUPAC 2013 (the Blue Book): every stereogenic unit is cited in a PIN.
"""

import pytest

from tests.support.pin_tiers import assert_pin_at_both_tiers

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("C/C(N)=N/C", "(1Z)-N'-methylethanimidamide"),
    ("C/C(N)=N\\C", "(1E)-N'-methylethanimidamide"),
    ("CC(N)=NC", "N'-methylethanimidamide"),
    ("C/C(NC)=N/C", "(1Z)-N,N'-dimethylethanimidamide"),
])
def test_names(smiles, name):
    assert_pin_at_both_tiers(smiles, name)
