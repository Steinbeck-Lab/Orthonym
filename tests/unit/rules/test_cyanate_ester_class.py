"""Esters of cyanic, thiocyanic and selenocyanic acid outrank a junior principal group, the Blue Book,
"9 Esters"; "Cyanic acid",:30979, "classified as an acid, thus generating... esters"). (:32035):
"In the presence of a characteristic group having seniority for citation as suffix, an ester group is indicated by
appropriate prefixes" -- only then. 'propan-2-yl thiocyanate (PIN)' (:32033), '3-(thiocyanato)propanoic acid (PIN)'
(:31013), 'S-ethyl 3-(thiocyanato)propanethioate (PIN)' (:32041); junior-group precedent '3-oxobutyl bromate (PIN)'
(:35974). The spelling check (validation/spelling/checks_cyanate_ester.py) lowers the label of a prefix name; the
thiocyanate handler names one thiocyanate beside a junior group in the functional-class form.
"""
import pytest
from rdkit import Chem

from orthonym.validation.pin_spelling import registered_rules
from orthonym.validation.spelling.checks_cyanate_ester import cyanate_ester_prefix_check
from tests.support.pin_tiers import assert_pin_at_both_tiers, name_breadth, name_default

pytestmark = pytest.mark.unit

FLAGGED = [
    ("N#CSCCO", "2-(thiocyanato)ethan-1-ol"),
    ("CC(=O)Nc1ccc(SC#N)cc1", "N-[4-(thiocyanato)phenyl]acetamide"),
    ("N#CSCCSC#N", "1,2-bis(thiocyanato)ethane"),
    ("N#CSCCSC#N", "2-(thiocyanato)ethyl thiocyanate"),            # two esters, one written as a prefix
    ("O=C(Cl)C1=C(SC#N)CCCC1", "2-(thiocyanato)cyclohex-1-ene-1-carbonyl chloride"),   # acid halide, class 10
    ("N#COc1ccccc1", "cyanatobenzene"),
    ("N#CSC1CCOC1=O", "3-(thiocyanato)oxolan-2-one"),     # a lactone is a pseudoketone, junior to esters (:32112)
    ("N#COCCSC#N", "2-cyanatoethyl thiocyanate"),          # two esters, one written as a prefix
    ("N#C[Se]CCO", "2-(selenocyanato)ethan-1-ol"),
]
PASSING = [
    ("N#CSCCO", "2-hydroxyethyl thiocyanate"),
    ("OC(=O)CCSC#N", "3-(thiocyanato)propanoic acid"),             #:31013
    ("CC(C)SC#N", "propan-2-yl thiocyanate"),                      #:32033
    ("CCSC(=O)CCSC#N", "S-ethyl 3-(thiocyanato)propanethioate"),   #:32041
    ("N#CSCCSC#N", "ethane-1,2-diyl bis(thiocyanate)"),
    ("CC(=O)SC#N", "acetyl thiocyanate"),                          # an acyl thiocyanate is an anhydride
    ("C[N+](C)(C)CCSC#N", "N,N,N-trimethyl-2-(thiocyanato)ethan-1-aminium"),   # an ion outranks esters
    ("N#CCCN=C=S", "3-isothiocyanatopropanenitrile"),              # isothiocyanate, not an ester
    ("CC(C)(C)OC(=O)NCCSC#N", "tert-butyl [2-(thiocyanato)ethyl]carbamate"),   # carbonic acids outrank cyanic (:30746)
    ("COC(=O)CCSC#N", "methyl 3-(thiocyanato)propanoate"),          # a carboxylic ester outranks it
]


@pytest.mark.parametrize("smiles,name", FLAGGED)
def test_the_prefix_name_is_flagged(smiles, name):
    failure = cyanate_ester_prefix_check(Chem.MolFromSmiles(smiles), name)
    assert failure is not None and failure.rule == "P-65.6.3.3.7.2.2", (name, failure)


@pytest.mark.parametrize("smiles,name", PASSING)
def test_ester_names_and_senior_groups_pass(smiles, name):
    assert cyanate_ester_prefix_check(Chem.MolFromSmiles(smiles), name) is None, name


def test_the_check_is_registered():
    assert "P-65.6.3.3.7.2.2" in registered_rules()


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", [
    ("N#CSCCO", "2-hydroxyethyl thiocyanate"),
    ("CC(=O)Nc1ccc(SC#N)cc1", "4-acetamidophenyl thiocyanate"),
    ("CC(=O)CCSC#N", "3-oxobutyl thiocyanate"),
    ("N#CSc1ccc(N)cc1", "4-aminophenyl thiocyanate"),
    ("N#CSC1CCCCC1O", "2-hydroxycyclohexyl thiocyanate"),
    ("OC(=O)CCSC#N", "3-(thiocyanato)propanoic acid"),
    ("CC(C)SC#N", "propan-2-yl thiocyanate"),
])
def test_one_thiocyanate_beside_a_junior_group_is_the_ester(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", ["O=C(Cl)C1=C(SC#N)CCCC1", "N#COc1ccccc1"])
def test_prefix_names_are_kept_below_the_pin(smiles):
    default = name_default(smiles)
    assert default.get("tier") != "pin_verified" and not default.get("is_pin"), default
    breadth = name_breadth(smiles)
    assert breadth.get("name") and not breadth.get("is_pin"), breadth


@pytest.mark.opsin_gate
def test_dithioperoxycyanic_acid_is_no_ester():
    """HS-S-CN is 'dithioperoxycyanic acid', the Blue Book), not 'sulfanyl thiocyanate'."""
    res = name_breadth("N#CSS")
    assert res.get("name") != "sulfanyl thiocyanate", res
