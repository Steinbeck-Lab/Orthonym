import pytest

from orthonym.rules.phane_pin import general_phane_name

from ._helpers import gold, mol


@pytest.mark.parametrize("line,name,smiles", gold("cycle", replaced=True), ids=lambda v: str(v)[:30])
def test_book_heterophane_pins(line, name, smiles):
    #,,
    assert general_phane_name(mol(smiles)) == name


@pytest.mark.parametrize("smiles", [
    "c1cc2ccc1C[S+](C)c1ccc(cc1)CC2",           # a charge on a chain atom
    "c1cc2ccc1CS(=O)c1ccc(cc1)CC2",             # a sulfoxide: not a standard valence
    "c1cc2ccc1C[Fe]c1ccc(cc1)CC2",              # an element outside
    "c1cc2ccc1C[GaH]c1ccc(cc1)CC2",             # lists gallium, but REPLACEMENT_TERMS has no term for it
])
def test_heteroatoms_the_rule_does_not_cover_decline(smiles):
    assert general_phane_name(mol(smiles)) is None


def test_the_element_order_is_the_one_of_p_26_5_4_2():
    # (:15246): O > S > Se > Te > N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In > Tl
    from orthonym.rules.skeletal_replacement import _A_CITATION_ORDER
    book = ["O", "S", "Se", "Te", "N", "P", "As", "Sb", "Bi", "Si", "Ge", "Sn", "Pb", "B", "Al", "Ga", "In", "Tl"]
    assert [e for e in _A_CITATION_ORDER if e in book] == book


def test_the_numbering_key_reads_the_same_order():
    from orthonym.rules.phane_pin.amplificants import HETERO_ORDER
    assert list(HETERO_ORDER) == ["O", "S", "Se", "Te", "N", "P", "As", "Sb", "Bi", "Si", "Ge", "Sn", "Pb", "B",
                                  "Al", "Ga", "In", "Tl"]
