"""A sulfonic or sulfinic acid or amide on a one-carbon chain that bears a ring: 'phenylmethanesulfonic acid'.

 (the Blue Book) puts the sulfonic acids in class 7a and the amides in class 11, senior to
the hydrocarbon ring, and gives the parent to the structure that holds the principal group: the CH2
bonded to the sulfur, not the ring, since the suffix of these acids is bonded straight to a skeletal
carbon (benzenesulfonic acid has the sulfur on the ring; here it is not). The one-carbon chain was dropped
for the ring ('methylcyclohexanesulfonic acid', '(sulfamoylmethyl)benzene', pin_verified) because the
parent-selection pre-empt knew only the skeletal -ol/-amine suffixes.

Locants: licences (the unsubstituted acid cites none, 'phenylmethanesulfonic acid'; a sulfonamide
cites '1-', as the book does for '1-hydrazinylmethanamine (PIN)':38535 and
'(1-cyclohexylmethanesulfonamido)acetic acid':33024). The sulfonyl PREFIX is 'phenylmethanesulfonyl'
; '1-methyl-4-[(R)-phenyl(18O1)methanesulfonyl]benzene (PIN)':46030).
"""
import pytest
from rdkit import Chem

from orthonym.rules.amides import format_n_substitution
from orthonym.rules.locanted_prefixes import locanted_prefix_block
from orthonym.rules.parent_selection import single_carbon_hosts_suffix
from orthonym.validation.opsin_roundtrip import opsin_parse
from tests.support.read_back import same_key

pytestmark = pytest.mark.opsin_gate


ROWS = [
    ("O=S(=O)(O)Cc1ccccc1", "phenylmethanesulfonic acid"),
    ("O=S(=O)(O)CC1CCCCC1", "cyclohexylmethanesulfonic acid"),
    ("OS(=O)Cc1ccccc1", "phenylmethanesulfinic acid"),
    ("O=S(=O)(N)Cc1ccccc1", "1-phenylmethanesulfonamide"),
    ("O=S(=O)(N)CC1CCCCC1", "1-cyclohexylmethanesulfonamide"),
    # N-substituted: the N locant and the ring locant stand in one alphanumerical list (:32879,:32881)
    ("CNS(=O)(=O)Cc1ccccc1", "N-methyl-1-phenylmethanesulfonamide"),
    ("CN(C)S(=O)(=O)Cc1ccccc1", "N,N-dimethyl-1-phenylmethanesulfonamide"),
    ("CCN(C)S(=O)(=O)Cc1ccccc1", "N-ethyl-N-methyl-1-phenylmethanesulfonamide"),
    ("CNS(=O)(=O)CC1CCCCC1", "1-cyclohexyl-N-methylmethanesulfonamide"),
    ("CC(C)NS(=O)(=O)Cc1ccccc1", "1-phenyl-N-(propan-2-yl)methanesulfonamide"),
    ("O=S(=O)(Cc1ccccc1)Nc1ccccc1", "N,1-diphenylmethanesulfonamide"),
    # dev2000 d1103 and d598 (6-deoxy-6-sulfo sugars), with the descriptors of the pipeline
    ("O=S(=O)(O)C[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
     "[(2S,3S,4S,5R,6R)-3,4,5,6-tetrahydroxyoxan-2-yl]methanesulfonic acid"),
    ("O=P(O)(O)OCC1(O)O[C@H](CS(=O)(=O)O)[C@@H](O)[C@@H]1O",
     "{(2S,3S,4S)-3,4,5-trihydroxy-5-[(phosphonooxy)methyl]oxolan-2-yl}methanesulfonic acid"),
]


@pytest.mark.parametrize("smiles,expected", ROWS)
def test_name_and_readback(namer, smiles, expected):
    row = namer.name_tiered(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified", row["name"]
    assert same_key(smiles, expected), expected


def test_the_sulfonamido_prefix_is_method_1_with_locant_1(namer):
    # (the Blue Book): '(1-cyclohexylmethanesulfonamido)acetic acid', method (1) = PIN;
    # '[(cyclohexylmethanesulfonyl)amino]acetic acid' would be the general-nomenclature method (2)
    row = namer.name_tiered("OC(=O)CNS(=O)(=O)CC1CCCCC1")
    assert row["name"] == "(1-cyclohexylmethanesulfonamido)acetic acid" and row["tier"] == "pin_verified", row
    assert namer.name("OC(=O)CCNS(=O)(=O)C") == "3-(methanesulfonamido)propanoic acid"      #:33019


def test_the_sulfonyl_prefix_is_the_methanesulfonyl_form(namer):
    # the same parent selection reaches the prefix: '1-[(2-methylphenyl)methanesulfonyl]piperidine...'
    smiles = "CC1=CC=CC=C1CS(=O)(=O)N1CCC(CC1)C(=O)O"
    name = namer.name(smiles)
    assert "(2-methylphenyl)methanesulfonyl" in name and "sulfonyl]" in name, name
    assert same_key(smiles, name), name


@pytest.mark.parametrize("smiles,expected", [
    ("OS(=O)(=O)c1ccccc1", "benzenesulfonic acid"),                      # sulfur on the ring: ring parent
    ("NS(=O)(=O)c1ccccc1", "benzenesulfonamide"),
    ("O=S(=O)(O)CCc1ccccc1", "2-phenylethane-1-sulfonic acid"),          # two carbons: unchanged
    ("OCc1ccccc1", "phenylmethanol"),
    ("NCC1CCCCC1", "1-cyclohexylmethanamine"),
    ("CNS(=O)(=O)c1ccccc1", "N-methylbenzenesulfonamide"),
    ("CNS(=O)(=O)C", "N-methylmethanesulfonamide"),
])
def test_neighbours_keep_their_names(namer, smiles, expected):
    assert namer.name(smiles) == expected


def test_hosts_suffix_helper():
    m = Chem.MolFromSmiles("O=S(=O)(O)Cc1ccccc1")
    ring = {a for r in m.GetRingInfo().AtomRings() for a in r}
    pg = [tuple(m.GetSubstructMatch(Chem.MolFromSmarts("[SX4](=O)(=O)[OX2H1]")))]
    assert single_carbon_hosts_suffix(m, "sulfonic_acid", pg, 4, ring) is True
    assert single_carbon_hosts_suffix(m, "carboxylic_acid", pg, 4, ring) is False   # not a sulfur suffix
    m2 = Chem.MolFromSmiles("OS(=O)(=O)c1ccccc1")                                   # sulfur on the ring
    ring2 = {a for r in m2.GetRingInfo().AtomRings() for a in r}
    pg2 = [tuple(m2.GetSubstructMatch(Chem.MolFromSmarts("[SX4](=O)(=O)[OX2H1]")))]
    assert single_carbon_hosts_suffix(m2, "sulfonic_acid", pg2, 4, ring2) is False


@pytest.mark.parametrize("subs", [["methyl"], ["methyl", "methyl"], ["methyl", "ethyl"],
                                  ["2-hydroxyethyl"], ["propan-2-yl", "propan-2-yl"], ["(2S)-butan-2-yl"]])
def test_block_equals_format_n_substitution_for_n_only_pieces(subs):
    assert locanted_prefix_block([(s, "N") for s in subs]) == format_n_substitution([{"name": s} for s in subs])


@pytest.mark.parametrize("name", [
    "4-(1-sulfanylideneethyl)benzoic acid",
    "4-chloro-7-[(3-chloro-4-nitroquinolin-7-yl)sulfanyl]-3-nitroquinoline",
    "2-(cyanosulfonyl)cyclohexane-1-carboxylic acid",
    "1-(2-oxophosphepan-3-yl)propan-1-one",
])
def test_the_four_refuted_names_are_not_pin_verified(namer, name):
    out = opsin_parse(name)
    if out:
        row = namer.name_tiered(out)
        assert not (row["tier"] == "pin_verified" and row["name"] == name)
