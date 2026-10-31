"""Lane E (multpin): the general multiplicative detector builds the multiplicative PIN where the
linker or the units were outside its scope, and no substitutive name stays labelled a PIN there.

 "Preferred IUPAC multiplicative names" (the Blue Book): "Multiplicative
nomenclature is preferred to substitutive nomenclature for generating preferred IUPAC names to
express multiple occurrences of identical parent structures, other than alkanes when (1) the
linking bonds... are identical, (2) the multiplicative groups, other than the central
multiplicative group, are symmetrically substituted; and (3) the locants of all substituent
groups on the identical parent structures, including suffix groups, are identical" (:23180-23184).
Every row below is a printed (PIN) of the Blue Book (the conformance oracle of
``benchmarks/bb_conformance``), apart from the ones marked 'derived'.

The classes (each row cites the section that prints it, with its heading):

* a substituted central or successive group, under " Simple multiplicative
  atoms or groups" (:5208) and (:5285): 'methylazanediyl', 'hydroxyazanediyl'
  "Hydroxylamine expressed as prefixes",:38436), 'acetylazanediyl' "Substituents of
  the types -NH-CO-R and -NH-SO2-R",:33062), '(2-methylpropane-1,3-diyl)' and 'oxybis(1-fluoro
  ethane-2,1-diyl)' "Assemblies of identical structural units",:5834,:5846),
  '(1,2-dibromoethane-1,2-diyl)' "HALOGEN COMPOUNDS",:25795),
  '(1,1,2,2-tetramethyldigermane-1,2-diyl)' "Substituted parent hydrides",:38243);
* a hetero atom or a run of hetero atoms as the group: 'phosphanediyl' and 'diarsenediyl'
   "Substitution of phosphanes, arsanes, and stibanes by organyl groups",:39230,
  :39234), 'diazenediyl',:38813), 'triselanediyl' and 'diselenoxanediyl'
   "Introduction",:39381,:39455), 'hydrazine-1,2-diyl' "Hydrazine and
  derivatives",:38547); an a(ba)n trio is the group 'dithioxanediyl',:8013);
* a group joined to a unit by a double bond: 'azanylylidene' and 'methanylylidene',
  :5265; "Substitutive names for imines",:26557,:26566,:26876);
* the units: a sulfonic acid, an amide with N-substituents "Substituted identical
  parent structures",:6227), a unit holding an 'oxo' prefix (derived), the ketone units of a
  diacyl derivative of trioxidane "Peroxyanhydrides and chalcogen analogues",:32362,
  :32366;:39393);
* the configuration of the double bond of the central chain "Stereodescriptors 'Z'
  and 'E', and 'cis' and 'trans'",:47100).

A linear phane (2),:23829) needs seven nodes: four ring systems and one oxygen make
five, so the multiplicative name stays the PIN and the substitutive name is no PIN (the lane's
first witness, ``WITNESS`` below).
"""
import pytest
from rdkit import Chem

from tests.support.pin_tiers import (
    assert_declined_at_default,
    assert_not_pin_labelled,
    assert_pin_at_both_tiers,
    name_breadth,
    name_default,
)
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

# (smiles, the PIN, where the book prints it)
PIN_ROWS = [
    # a substituted central group
    ("O=C(O)c1cccc(N(O)c2cccc(C(=O)O)c2)c1", "3,3'-(hydroxyazanediyl)dibenzoic acid"),          #:38436
    ("CC(=O)N(c1ccc(C(=O)O)cc1)c1ccc(C(=O)O)cc1", "4,4'-(acetylazanediyl)dibenzoic acid"),      #:33062
    ("CC(COc1ccc(O)cc1)COc1ccc(O)cc1", "4,4'-[(2-methylpropane-1,3-diyl)bis(oxy)]diphenol"),    #:5834
    ("O=S(=O)(O)C1CCC(C(F)COCC(F)C2CCC(S(=O)(=O)O)CC2)CC1",
     "4,4'-[oxybis(1-fluoroethane-2,1-diyl)]di(cyclohexane-1-sulfonic acid)"),                   #:5846
    ("BrC(c1ccccc1)C(Br)c1ccccc1", "1,1'-(1,2-dibromoethane-1,2-diyl)dibenzene"),               #:25795
    ("[CH3][Ge]([CH3])([CH2]CCCC=O)[Ge]([CH3])([CH3])[CH2]CCCC=O",
     "5,5'-(1,1,2,2-tetramethyldigermane-1,2-diyl)di(pentanal)"),                               #:38243
    # derived: the same rule with a carbon substituent and with a carbon chain that the free
    # valence of the first atom lengthens ('propane-1,2-diyl',,:15935)
    ("CN(c1ccc(C(=O)O)cc1)c1ccc(C(=O)O)cc1", "4,4'-(methylazanediyl)dibenzoic acid"),
    ("O=C(O)c1ccc(CC(C)c2ccc(C(=O)O)cc2)cc1", "4,4'-(propane-1,2-diyl)dibenzoic acid"),
    # a hetero atom, or a run of them, as the group
    ("O=C(O)c1ccc(Pc2ccc(C(=O)O)cc2)cc1", "4,4'-phosphanediyldibenzoic acid"),                  #:39230
    ("Oc1ccc([As]=[As]c2ccc(O)cc2)cc1", "4,4'-diarsenediyldiphenol"),                           #:39234
    ("O=C(O)c1ccc(N=Nc2ccc(C(=O)O)cc2)cc1", "4,4'-diazenediyldibenzoic acid"),                  #:38813
    ("O=C(O)c1ccc([Se][Se][Se]c2ccc(C(=O)O)cc2)cc1", "4,4'-triselanediyldibenzoic acid"),       #:39381
    ("Oc1ccc([Se]O[Se]c2ccc(O)cc2)cc1", "4,4'-diselenoxanediyldiphenol"),                       #:39455
    ("NC(=O)CCNNCCC(N)=O", "3,3'-(hydrazine-1,2-diyl)dipropanamide"),                           #:38547
    # a group joined to a unit by a double bond
    ("Oc1ccccc1C=NCCN=Cc1ccccc1O",
     "2,2'-[ethane-1,2-diylbis(azanylylidenemethanylylidene)]diphenol"),                        #:26566
    ("Oc1c(F)cccc1C=NCCN=Cc1cccc(F)c1O",
     "2,2'-[ethane-1,2-diylbis(azanylylidenemethanylylidene)]bis(6-fluorophenol)"),             #:26876
    ("O=C(O)CC=NCCN=CCC(=O)O", "3,3'-[ethane-1,2-diylbis(azanylylidene)]dipropanoic acid"),     #:26557
    # the units
    ("O=S(=O)(O)c1ccc(Oc2ccc(S(=O)(=O)O)cc2)cc1", "4,4'-oxydi(benzene-1-sulfonic acid)"),       #:5827
    ("NC(=O)C1CCCCC1NCCNC1CCCCC1C(N)=O",
     "2,2'-[ethane-1,2-diylbis(azanediyl)]di(cyclohexane-1-carboxamide)"),                      #:32908
    ("CN(C)C(=O)CCSCCC(=O)N(C)C", "3,3'-sulfanediylbis(N,N-dimethylpropanamide)"),              #:6227
    ("CCN(CC)C(=O)c1ccc(Oc2ccc(C(=O)N(CC)CC)cc2)cc1", "4,4'-oxybis(N,N-diethylbenzamide)"),     # derived
    ("OC(=O)CC(=O)c1ccc(cc1)-c1ccc(cc1)C(=O)CC(=O)O",
     "3,3'-([1,1'-biphenyl]-4,4'-diyl)bis(3-oxopropanoic acid)"),                               # derived
    ("CC(=O)OOOC(C)=O", "1,1'-trioxidanediyldi(ethan-1-one)"),                                  #:32362
    ("CC(=O)SOSC(C)=O", "1,1'-dithioxanediyldi(ethan-1-one)"),                                  #:32366
    ("CCC(=O)SSSC(=O)CC", "1,1'-trisulfanediyldi(propan-1-one)"),                               #:39393
    # the configuration of the double bond of the central chain
    ("Clc1ccc(/C(=C/c2ccccc2)c2ccccc2)cc1",
     "1,1'-[(1E)-1-(4-chlorophenyl)ethene-1,2-diyl]dibenzene"),                                 #:47100
    ("Clc1ccc(/C(=C\\c2ccccc2)c2ccccc2)cc1",
     "1,1'-[(1Z)-1-(4-chlorophenyl)ethene-1,2-diyl]dibenzene"),                                 # derived
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_pin_at_both_tiers(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


# Where the book's criteria do not give a multiplicative name, the name stays substitutive and no
# multiplicative token appears: (smiles, token).
NOT_MULTIPLICATIVE_ROWS = [
    # the group is a hydroxy on a hetero atom: a characteristic group of that atom, not a prefix
    # ('methyl bis(1-hydroxybutan-2-yl)borinate' is the borinic ester, an ester outranks an alcohol)
    ("CCC(CO)B(OC)C(CC)CO", "boranediyl"),
    ("C[Si](C)(C)[CH]([SnH]([OH])[CH]([Si](C)(C)C)[Si](C)(C)C)[Si](C)(C)C", "stannanediyl"),
    # two chalcogens between two acyl groups: an anhydride of a peroxy acid, (:32338)
    ("CC(=O)SSC(C)=O", "disulfanediyl"),
    # the two amides differ: not identical units, (3)
    ("CNC(=O)CCOCCC(=O)NCC", "oxydi"),
    # a branched alkyl on the atom next to a unit: the chain is not unbranched
    ("CC(C)C(c1ccccc1)c1ccccc1", "diyl"),
    # a linear phane of seven nodes (2),:23829)
    ("c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1", "phenylene"),
    # an amino branch on the nitrogen of the group makes it a hydrazine,:38513), never
    # 'aminoazanediyl'; the book prints no multiplicative name with hydrazine-1,1-diyl, so none is built
    ("O=C(O)c1ccc(N(N)c2ccc(C(=O)O)cc2)cc1", "aminoazanediyl"),
    ("NC(=O)CCN(N)CCC(N)=O", "aminoazanediyl"),
    # a Group 13 chain states its hydrogen count, 'diborane(4)',:37105,:37109)
    ("O=C(O)c1ccc(BBc2ccc(C(=O)O)cc2)cc1", "diborane-1,2-diyl"),
]


@pytest.mark.parametrize("smiles,token", NOT_MULTIPLICATIVE_ROWS)
def test_not_a_multiplicative_name(smiles, token):
    for res in (name_default(smiles), name_breadth(smiles)):
        assert token not in (res.get("name") or ""), res
    b = name_breadth(smiles)
    assert b.get("name") and name_is_rt_exact(b["name"], smiles), b


# Four ring systems in one line with fewer than seven nodes are no phane: the multiplicative name
# is the PIN,:23178), so the substitutive name the engine builds is not one. The
# multiplicative name needs a ring assembly as a successive group, which the detector does not
# write ('3,3'-{oxybis[([1,1'-biphenyl]-4',4-diyl)oxy]}dipropanoic acid' is the form,
#:5297, asks for); the default tier then declines, the wider tiers keep the verified name.
WITNESS = "OC(=O)CCOc1ccc(cc1)-c1ccc(Oc2ccc(cc2)-c2ccc(OCCC(=O)O)cc2)cc1"
WITNESS_SUBSTITUTIVE = ("3-[(4'-{[4'-(2-carboxyethoxy)[1,1'-biphenyl]-4-yl]oxy}"
                        "[1,1'-biphenyl]-4-yl)oxy]propanoic acid")


def test_witness_substitutive_name_is_no_pin():
    from orthonym.rules.pin_vocabulary import multiplicative_pin_expected
    assert multiplicative_pin_expected(Chem.MolFromSmiles(WITNESS))
    assert_not_pin_labelled(WITNESS, WITNESS_SUBSTITUTIVE)
    d = name_default(WITNESS)
    assert d.get("tier") != "pin_verified", d
    b = name_breadth(WITNESS)
    assert b.get("tier") == "systematic_verified", b


def test_four_ring_systems_of_seven_nodes_are_still_a_phane():
    # four benzene rings and three oxygens (:23959): a heptaphane, the multiplicative claim is withheld
    from orthonym.rules.pin_vocabulary import multiplicative_pin_expected
    smi = "OC(=O)CCOc1ccc(Oc2ccc(Oc3ccc(Oc4ccc(OCCC(=O)O)cc4)cc3)cc2)cc1"
    assert not multiplicative_pin_expected(Chem.MolFromSmiles(smi))


def test_linear_ring_chain_nodes():
    from orthonym.rules.multiplicative_general import linear_ring_chain_nodes
    assert linear_ring_chain_nodes(Chem.MolFromSmiles(WITNESS)) == 5
    assert linear_ring_chain_nodes(Chem.MolFromSmiles(
        "c1ccc(Oc2ccc(Oc3ccc(Oc4ccccc4)cc3)cc2)cc1")) == 7
    assert linear_ring_chain_nodes(Chem.MolFromSmiles("c1ccc(Oc2ccccc2)cc1")) == 0


def test_declined_at_default_when_the_name_is_not_a_pin():
    # the isopropyl on the atom next to the units keeps the chain branched: not built, so the
    # substitutive name is no PIN and the default tier declines it
    assert_declined_at_default("CC(C)C(c1ccccc1)c1ccccc1")
