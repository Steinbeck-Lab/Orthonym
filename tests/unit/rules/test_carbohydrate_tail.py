"""P7 (Wave-8) sub-plan 7b — carbohydrate semisystematic tail (P-102).

7b.2 — open-chain uronic acid PINs (P-102.5.6.6.4.1).

Blue Book P-102.5.6.6.4.1 (BB:53779-53783): "Names of individual uronic acids
are formed by changing the ending 'ose' in the retained or systematic name of
the corresponding aldose to 'uronic acid'. The numbering of the aldose is kept
intact; the locant '1' is still assigned to the (potential) aldehydic group."
Example given: D-glucuronic acid. The RING form (alpha-D-glucopyranuronic acid)
already names correctly; the OPEN-CHAIN aldehydo form emitted the systematic
...-6-oxohexanoic acid. All target names are OPSIN-parseable (normal gate).

Structures are OPSIN name->structure authoritative (reproduce-first).
"""

import pytest
from rdkit import Chem

from orthonym.namer import Orthonym

RAW = Orthonym(_disable_opsin_validity_gate=True)
GATED = Orthonym()


@pytest.mark.parametrize("smiles,expected", [
    ("O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)C(=O)O", "D-glucuronic acid"),
    ("O=C[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)C(=O)O", "D-galacturonic acid"),
    ("O=C[C@@H](O)[C@@H](O)[C@H](O)[C@H](O)C(=O)O", "D-mannuronic acid"),
    ("O=C[C@H](O)[C@@H](O)[C@H](O)[C@@H](O)C(=O)O", "L-iduronic acid"),
    ("O=C[C@@H](O)[C@@H](O)[C@H](O)[C@@H](O)C(=O)O", "L-guluronic acid"),
])
def test_open_chain_uronic_acid_pin(smiles, expected):
    can = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    assert GATED.name(can) == expected
    assert RAW.name(can) == expected


def test_uronic_ring_form_unchanged():
    # Regression: the pyranuronic ring form must keep its existing PIN.
    can = Chem.MolToSmiles(Chem.MolFromSmiles(
        "O=C(O)[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O"))
    assert GATED.name(can) == "alpha-D-glucopyranuronic acid"
