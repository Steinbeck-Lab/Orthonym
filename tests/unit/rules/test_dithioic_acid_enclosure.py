"""P-65.1.5.1 / P-65.6.2.1 — 'dithioic acid'/'dithioate' are enclosed even at count 1.

The 'di' of 'dithio-' is visually a multiplicative 'di' on the suffix, so the Blue
Book parenthesises these two suffixes even for a single group (unlike plain 'thioic
acid', which stays bare):

    ...prop-2-ene(dithioic acid) (P-65.1.5.1, the Blue Book)
    sodium propane(dithioate) (P-65.6.2.1, the Blue Book)

Two coupled sites: format_suffix_with_locants adds the count=1 wrap on the neutral
acid; _acid_anion_from_neutral swaps the ending INSIDE the parens so the salt keeps
the marks. Plain 'thioic acid' at count 1 must stay bare (BB 'hexanethioic O-acid').

Every emitted name OPSIN-round-trips.
"""
import pytest
from rdkit import Chem

_FLAGS = dict(general_fallback=True, general_fallback_unverified=True,
              allow_aromatic_general=True)


def _inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None


@pytest.mark.parametrize("smiles,expected", [
    ("CCSC(N)=CC(=S)S", "3-amino-3-(ethylsulfanyl)prop-2-ene(dithioic acid)"),
    ("CCC(=S)[S-].[Na+]", "sodium propane(dithioate)"),
    ("CCC(=S)S", "propane(dithioic acid)"),  # bare-count-1 acid also gains the marks
])
def test_dithioic_acid_and_dithioate_enclosed(smiles, expected):
    from orthonym import Orthonym
    got = Orthonym(**_FLAGS).name_tiered(smiles)["name"]
    assert got == expected, f"{smiles}: got {got!r}"


def test_plain_thioic_acid_stays_bare():
    """Only the two 'di'-chalcogen tokens gain the count=1 wrap; plain thioic does not."""
    from orthonym import Orthonym
    got = Orthonym(**_FLAGS).name_tiered("CCCCCC(=O)S")["name"]
    assert "(" not in got.split(" ")[-1], got  # no parenthesised suffix
    assert "thioic" in got and "acid" in got, got


@pytest.mark.parametrize("smiles,expected", [
    ("CCSC(N)=CC(=S)S", "3-amino-3-(ethylsulfanyl)prop-2-ene(dithioic acid)"),
    ("CCC(=S)[S-].[Na+]", "sodium propane(dithioate)"),
])
def test_expected_pins_round_trip(smiles, expected):
    from orthonym.validation.opsin_roundtrip import opsin_parse
    rt = opsin_parse(expected)
    got = Chem.MolFromSmiles(rt) if rt else None
    assert got is not None and Chem.MolToInchiKey(got) == _inchikey(smiles), \
        f"RT failed for {expected!r}"
