"""A ring-assembly name whose component name begins with a locant takes a hyphen
between the multiplying prefix and that locant.

 (the Blue Book): an assembly of two identical cyclic systems is named
"by placing the prefix 'bi' (see before the name of the corresponding parent
hydride enclosed in parentheses, if necessary. Parentheses are used to avoid confusion
with von Baeyer names" (:15565). (a) (the Blue Book-6938): hyphens are
used "to separate locants from words or word fragments". The book's own ring-assembly
names with a locant-initial component all carry that hyphen:

  '2,2'-bi-3,1,5-benzoxadiarsepine (PIN)',:20912
  '2,2'-bi-1-naphthol',:27224 (and '3,7'-bi-1-naphthol',:27226)
  '2,2'-bi-2H-pyran', '1,1'-bi-1H-pyrrole' earlier-edition forms,,:15595,:15603

while the parentheses stay with the cycloalkane and von Baeyer components
('1,1'-bi(cyclopropane) (PIN)',:15573; '2,2'-bi(bicyclo[2.2.1]heptanylidene) (PIN)',
:15591) and a mancude component without a leading locant stays bare
('2,2'-bipyridine (PIN)', '1,1'-biphenyl (PIN)',:15577,:15575).
"""
import subprocess

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.rules import ring_assemblies as ra
from orthonym.validation.opsin_roundtrip import _find_opsin_jar

pytestmark = pytest.mark.opsin_gate

_BEST_EFFORT = dict(general_fallback=True, general_fallback_unverified=True,
                    allow_aromatic_general=True)

# (SMILES, the name). The first row is the Blue Book row (:20912).
LOCANT_INITIAL = [
    ("C1=[As]c2ccccc2[As]=C(C2=[As]c3ccccc3[As]=CO2)O1",
     "2,2'-bi-3,1,5-benzoxadiarsepine"),
    ("c1csc(n1)-c1nccs1", "2,2'-bi-1,3-thiazole"),
    ("c1ccc2sc(cc2c1)-c1cc2ccccc2s1", "2,2'-bi-1-benzothiophene"),
    ("C1COC(O1)C1OCCO1", "2,2'-bi-1,3-dioxolane"),
    # substituent prefixes on the assembly
    ("Cc1csc(n1)-c1nc(C)cs1", "4,4'-dimethyl-2,2'-bi-1,3-thiazole"),
    # the assembly carries the suffix (the enclosed parent, +
    ("OC(=O)c1csc(n1)-c1nc(C(O)=O)cs1",
     "[2,2'-bi-1,3-thiazole]-4,4'-dicarboxylic acid"),
    # the assembly is a substituent prefix ('-yl' on the enclosed assembly)
    ("CC(=O)c1csc(n1)-c1nccs1", "1-([2,2'-bi-1,3-thiazol]-4-yl)ethan-1-one"),
    # three and four components, the same prefix-to-name junction)
    ("c1csc(n1)-c1csc(n1)-c1nccs1", "2,4':2',2''-ter-1,3-thiazole"),
]

# Names whose component does not begin with a locant: unchanged.
CONTROLS = [
    ("c1ccc(nc1)-c1ccccn1", "2,2'-bipyridine"),
    ("c1ccccc1-c1ccccc1", "1,1'-biphenyl"),
    ("c1ccc2cc(ccc2c1)-c1ccc2ccccc2c1", "2,2'-binaphthalene"),
    ("C1CC1C1CC1", "1,1'-bi(cyclopropane)"),
    ("C1CCC(CC1)C1CCCCC1", "1,1'-bi(cyclohexane)"),
    ("C1CCC(C1)=C1CCCC1", "1,1'-bi(cyclopentylidene)"),
    ("c1ccc2[nH]c(cc2c1)-c1cc2ccccc2[nH]1", "1H,1'H-2,2'-biindole"),
    ("C1C=COC1=C1OC=CC1", "3H,3'H-2,2'-bifuranylidene"),
    # the middle ring takes the lower locant for its bond to the previous ring,
    # the Blue Book "the locant set 1,1':2',1'':3'',1''' is lower than 1,1':3',1'':2'',1'''";
    # the book's own '2,2':6',2'':6'',2'''-quaterpyridine':15667): '2,2':5',2''-terthiophene',
    # which OPSIN 2.9.0 reads back to the full InChIKey (leads L2 item N9c; was '2,5':2',2''-')
    ("s1cccc1-c1ccc(s1)-c1cccs1", "2,2':5',2''-terthiophene"),
]


@pytest.fixture(scope="module")
def best_effort():
    return Orthonym(**_BEST_EFFORT)


@pytest.fixture(scope="module")
def opsin_keys():
    jar = _find_opsin_jar("2.9.0")
    if jar is None:
        pytest.skip("OPSIN 2.9.0 jar not found")
    names = [name for _, name in LOCANT_INITIAL + CONTROLS]
    out = subprocess.run(["java", "-jar", str(jar), "-ostdinchikey"],
                         input="\n".join(names) + "\n", capture_output=True,
                         text=True, timeout=600).stdout.split("\n")
    return dict(zip(names, (line.strip() for line in out)))


@pytest.mark.parametrize("smiles,expected", LOCANT_INITIAL + CONTROLS,
                         ids=[name for _, name in LOCANT_INITIAL + CONTROLS])
def test_best_effort_name(best_effort, smiles, expected):
    result = best_effort.name_tiered(smiles)
    assert result["name"] == expected, result


@pytest.mark.parametrize("smiles,expected", LOCANT_INITIAL + CONTROLS,
                         ids=[name for _, name in LOCANT_INITIAL + CONTROLS])
def test_opsin_reads_the_name_back_to_the_input(opsin_keys, smiles, expected):
    assert opsin_keys[expected] == Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def test_the_writer_puts_a_hyphen_only_before_a_locant():
    join = ra._cite_after_multiplier
    assert join("bi", "3,1,5-benzoxadiarsepine") == "bi-3,1,5-benzoxadiarsepine"
    assert join("quater", "1,3-thiazole") == "quater-1,3-thiazole"
    assert join("bi", "pyridine") == "bipyridine"
    assert join("ter", "phenyl") == "terphenyl"
    # the parentheses keep the prefix and the component together
    assert join("bi", ra._enclose_component("bicyclo[2.2.1]heptane")) == \
        "bi(bicyclo[2.2.1]heptane)"
    assert join("bi", ra._enclose_component("cyclopropane")) == "bi(cyclopropane)"


# (the Blue Book): "The preferred numbering for ring assemblies
# composed of three or more identical cyclic systems uses composite locants rather
# than primed locants (see "; (:15655) recommends the composite
# locants for preferred IUPAC names and leaves the serially primed locants to general
# nomenclature, as in the PIN beside '1H,3''H-3,3':3',3''-terindole' (:15669). The writer
# spells these assemblies with primed locants, so such a name is never labelled a PIN:
# the default tier declines it and the wider tiers return it below the PIN label.
PRIMED_THREE_PLUS = [
    ("s1cccc1-c1ccc(s1)-c1cccs1", "2,2':5',2''-terthiophene"),
    ("c1csc(n1)-c1csc(n1)-c1nccs1", "2,4':2',2''-ter-1,3-thiazole"),
    ("c1ccc(cc1)-c1ccc(cc1)-c1ccccc1", "1,1':4',1''-terphenyl"),
    ("c1ccc(cc1)-c1ccc(cc1)-c1ccc(cc1)-c1ccccc1", "1,1':4',1'':4'',1'''-quaterphenyl"),
    # the assembly as the suffix parent and as a substituent prefix
    ("OC(=O)c1ccc(cc1)-c1ccc(cc1)-c1ccccc1", "[1,1':4',1''-terphenyl]-4-carboxylic acid"),
    ("OC(=O)CCCc1ccc(cc1)-c1ccc(cc1)-c1ccccc1",
     "4-([1,1':4',1''-terphenyl]-4-yl)butanoic acid"),
]

# Two-component assemblies: primed locants are the PIN numbering,:15573-15577).
TWO_COMPONENT_PINS = [
    ("C1=[As]c2ccccc2[As]=C(C2=[As]c3ccccc3[As]=CO2)O1",
     "2,2'-bi-3,1,5-benzoxadiarsepine"),
    ("c1csc(n1)-c1nccs1", "2,2'-bi-1,3-thiazole"),
    ("c1ccc(nc1)-c1ccccn1", "2,2'-bipyridine"),
    ("c1ccccc1-c1ccccc1", "1,1'-biphenyl"),
    ("OC(=O)c1ccc(cc1)-c1ccccc1", "[1,1'-biphenyl]-4-carboxylic acid"),
]


@pytest.fixture(scope="module")
def default_tier():
    return Orthonym()


@pytest.mark.parametrize("smiles,expected", PRIMED_THREE_PLUS,
                         ids=[name for _, name in PRIMED_THREE_PLUS])
def test_primed_three_plus_assembly_is_never_a_pin(best_effort, default_tier,
                                                   smiles, expected):
    from orthonym.errors import is_failure_name
    wide = best_effort.name_tiered(smiles)
    assert wide["name"] == expected, wide
    assert wide["tier"] == "systematic_verified" and not wide["is_pin"], wide
    pin = default_tier.name_tiered(smiles)
    assert pin["tier"] != "pin_verified" and not pin["is_pin"], pin
    assert is_failure_name(pin["name"]), pin


@pytest.mark.parametrize("smiles,expected", PRIMED_THREE_PLUS,
                         ids=[name for _, name in PRIMED_THREE_PLUS])
def test_the_decline_of_a_primed_three_plus_assembly_records_its_rule(best_effort, default_tier,
                                                                     smiles, expected):
    # the default tier's decline row, and the row of the name the wider tiers keep, give
    # as the reason (``spelling_failures``)
    pin = default_tier.name_tiered(smiles)
    assert (pin["tier"], pin["limit_code"]) == ("abstain", "NO_VERIFIED_PIN"), pin
    assert "P-52.2.7.2" in [f["rule"] for f in pin["spelling_failures"]], pin
    wide = best_effort.name_tiered(smiles)
    assert wide["name"] == expected, wide
    assert "P-52.2.7.2" in [f["rule"] for f in wide["spelling_failures"]], wide


@pytest.mark.parametrize("smiles,expected", TWO_COMPONENT_PINS,
                         ids=[name for _, name in TWO_COMPONENT_PINS])
def test_two_component_assembly_keeps_the_pin_label(default_tier, smiles, expected):
    result = default_tier.name_tiered(smiles)
    assert result["name"] == expected, result
    assert result["tier"] == "pin_verified" and result["is_pin"], result
    assert not result["spelling_failures"], result


def test_the_label_holds_on_a_warm_namer(best_effort):
    """The same namer, the same molecules again after others: the label does not
    depend on what the namer has cached."""
    for smiles, expected in PRIMED_THREE_PLUS + PRIMED_THREE_PLUS[::-1]:
        result = best_effort.name_tiered(smiles)
        assert result["name"] == expected and result["tier"] == "systematic_verified", result
    for smiles, expected in TWO_COMPONENT_PINS:
        result = best_effort.name_tiered(smiles)
        assert result["name"] == expected and result["tier"] == "pin_verified", result
