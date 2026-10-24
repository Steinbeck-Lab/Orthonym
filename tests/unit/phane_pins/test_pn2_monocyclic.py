import pytest

from orthonym.rules.phane_pin import general_phane_name

from ._helpers import gold, mol


@pytest.mark.parametrize("line,name,smiles", gold("cycle", replaced=False, unsaturated=False),
                         ids=lambda v: str(v)[:30])
def test_book_monocyclic_hydrocarbon_pins(line, name, smiles):
    #,,, to: the names the book prints for its monocyclic skeletons
    assert general_phane_name(mol(smiles)) == name


def test_the_five_legacy_golds_agree_with_the_legacy_builder():
    from orthonym.rules.phane import build_phane_pin
    for smiles in ("C1Cc2ccc(cc2)CCc2ccc1cc2", "c1cc2cc(c1)CCc1cccc(c1)CC2",
                   "c1cc2ccc1CCCc1ccc(cc1)CCC2", "c1cc2ccc1CCCc1ccc(cc1)CC2",
                   "c1cc2cc(c1)CCCc1cccc(c1)CCC2"):
        assert general_phane_name(mol(smiles)) == build_phane_pin(mol(smiles))


@pytest.mark.parametrize("smiles", [
    "C1CC2=CC=C(C=C2)CCC2=CC=C1C=C2.[Na+]",      # two components
    "C1CC2=CC=C(C=C2)CC[CH]C2=CC=C1C=C2",        # a radical
    "[2H]C1CC2=CC=C(C=C2)CCC2=CC=C1C=C2",        # an isotope
    "C1CC2=CC=C(C=C2)CC[NH2+]C2=CC=C1C=C2",       # a charge
    "C1CCCCC1", "c1ccc2ccccc2c1",                # no phane
])
def test_out_of_scope_inputs_decline(smiles):
    assert general_phane_name(mol(smiles)) is None
