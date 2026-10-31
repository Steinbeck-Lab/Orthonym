"""An amino acid, its anion and its ester are named by their systematic substitutive names (the PIN).

 INTRODUCTION (the Blue Book): "Preferred IUPAC names (PINs) are not identified for the compounds in this
Chapter", so 'L-alanine', 'glycinate' and 'methyl L-alaninate' are no PINs. "Systematic substitutive
names" (:54247): "When not denoted by a retained name, amino acids receive systematic substitutive names constructed by
applying the principles, rules and conventions of substitutive nomenclature"; Table 10.4 (:54186) prints that name
under each retained one ('2-aminopropanoic acid', 'aminoacetic acid'). The retained name stays the fallback of the
wider tiers when no systematic name reads back.
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


def _best_effort():
    return Orthonym(style="pin", **_emit_tier_flags("best-effort"))


@pytest.mark.parametrize("smiles,expected", [
    ("NCC(=O)O", "aminoacetic acid"),                                  # Table 10.4,:54202
    ("C[C@H](N)C(=O)O", "(2S)-2-aminopropanoic acid"),                 # Table 10.4,:54192
    ("C[C@@H](N)C(=O)O", "(2R)-2-aminopropanoic acid"),
    ("N[C@@H](CS)C(=O)O", "(2R)-2-amino-3-sulfanylpropanoic acid"),    # Table 10.4, cysteine
    ("N[C@@H](Cc1ccc(O)cc1)C(=O)O", "(2S)-2-amino-3-(4-hydroxyphenyl)propanoic acid"),
    ("NCC(=O)[O-]", "aminoacetate"),
    ("N[C@@H](CS)C(=O)[O-]", "(2R)-2-amino-3-sulfanylpropanoate"),
    ("C[C@H](N)C(=O)OC", "methyl (2S)-2-aminopropanoate"),
    ("NCC(=O)OCC", "ethyl aminoacetate"),
])
def test_the_pin_tier_names_the_systematic_name(smiles, expected):
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (expected, "pin_verified")
    assert _best_effort().name_tiered(smiles)["name"] == expected


@pytest.mark.parametrize("smiles,expected", [
    ("N[C@@H](CCC[C@H](N)C(=O)O)C(=O)O", "(2S,6S)-2,6-diaminoheptanedioic acid"),  # was 'l,l-diaminopimelic acid'
    ("NCCS", "2-aminoethane-1-thiol"),                                             # was 'cysteamine'
])
def test_a_trivial_name_of_the_tables_gives_way_to_the_pin(smiles, expected):
    assert Orthonym().name_tiered(smiles)["name"] == expected


@pytest.mark.parametrize("smiles,retained", [
    ("N[C@@H](CCSSCC[C@H](N)C(=O)O)C(=O)O", "homocystine"),
    ("N[C@@H](CSSC[C@H](N)C(=O)O)C(=O)O", "L-cystine"),
])
def test_a_molecule_whose_pin_is_multiplicative_keeps_its_retained_name_below_the_pin(smiles, retained):
    # "Preferred IUPAC multiplicative names" (the Blue Book): the substitutive name of two identical
    # halves is not the PIN, so it is not offered; the retained name is no PIN either,:50943)
    row = _best_effort().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (retained, "systematic_verified")
    assert Orthonym().name_tiered(smiles)["limit_code"] == "NO_VERIFIED_PIN"


def test_without_a_systematic_name_the_retained_name_ships_below_the_pin(monkeypatch):
    import orthonym.rules.amino_acids as amino_acids
    monkeypatch.setattr(amino_acids, "systematic_amino_acid_name", lambda mol: None)
    for smiles, retained in (("C[C@H](N)C(=O)O", "L-alanine"), ("NCC(=O)O", "glycine"),
                             ("N[C@@H](CCSSCC[C@H](N)C(=O)O)C(=O)O", "homocystine")):
        row = _best_effort().name_tiered(smiles)
        assert row["name"] == retained and row["tier"] == "systematic_verified", smiles
        assert Orthonym().name_tiered(smiles)["limit_code"] == "NO_VERIFIED_PIN", smiles


def test_a_peptide_keeps_its_residue_names():
    row = _best_effort().name_tiered("C[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)CN)C(=O)O")
    assert row["name"] == "glycylglutamylalanine"


def test_the_offer_is_only_for_the_whole_molecule():
    from rdkit import Chem

    from orthonym.rules.amino_acids import systematic_amino_acid_name
    assert systematic_amino_acid_name(Chem.MolFromSmiles("NCC(=O)O")) is None      # outside a name call
