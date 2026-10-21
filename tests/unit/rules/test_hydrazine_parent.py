"""HP lane: the hydrazine as the senior parent hydride.

 (the Blue Book): "The senior parent structure, whether cyclic or acyclic, has the senior
atom in accordance with the seniority of classes (see expressed by the following decreasing element
order: N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In > Tl > O > S > Se > Te > C. This
criterion is applied to select the senior atom in parents and to choose between rings and chains."
 (:18919): "A single senior atom is sufficient to give seniority to the parent hydride", with
'1-(2H-pyran-3-yl)-2-(silolan-2-yl)hydrazine (PIN) (N > Si > O)' (:18950). class 21
(:18197): "Nitrogen compounds: heterocycles, polyazanes, hydrazines (except for hydrazides), diazenes,
hydroxylamines, azanes (except for amides, amines, and imines)". "Hydrazine and
derivatives" (:38529): "Hydrazine is substituted by hydrocarbyl groups and characteristic groups
expressed by suffixes and prefixes", 'phenylhydrazine (PIN)' (:38535); the locants of the parent are 1
and 2, the lowest to the prefix cited first in the alphanumerical order:
NUMBERING (:3219), criterion (g) (:3307) "lowest locants for the substituent cited first as a prefix in
the name" (witness: '1-hydroxy-3-oxopropane-1,2,3-tricarboxylic acid (PIN)',:29956).

A molecule with no characteristic group of the classes 1 to 20 and no nitrogen ring is named on the
hydrazine, whatever its substituents hold of the elements the nitrogen outranks (a halogen, an ether or
sulfide, a nitro group, a ring or an atom of O, S, Si, P, B...). The producer is
``rules.polyazane.name_polyazane``; the question "is the hydrazine the parent" is asked by
``validation.spelling.polyazane_parent`` and by the spelling check of the 'hydrazinyl' prefix alike, so the
producer names exactly the molecules for which that check calls the prefix name wrong.
"""
import pytest
from rdkit import Chem

from orthonym.validation.spelling.polyazane_parent import hydrazine_is_senior_parent
from orthonym.rules.polyazane import name_polyazane
from tests.support.pin_tiers import (
    assert_declined_at_default, assert_pin_at_both_tiers, name_breadth, name_default)

# (smiles, the PIN both tiers ship at pin_verified; OPSIN 2.9.0 reads it back to the full key)
PIN_ROWS = [
    #:18950 verbatim: a ring of oxygen and a ring of silicon are both junior to the nitrogen of the hydrazine
    ("C1=COCC(NNC2CCC[SiH2]2)=C1", "1-(2H-pyran-3-yl)-2-(silolan-2-yl)hydrazine"),
    # a halogen on the ring: the retained 'phenylhydrazine' (:38535) with a substituent prefix on the ring
    ("NNc1ccc(Cl)cc1", "(4-chlorophenyl)hydrazine"),
    ("NNc1ccccc1Cl", "(2-chlorophenyl)hydrazine"),
    ("NNc1ccc(Br)cc1", "(4-bromophenyl)hydrazine"),
    # two substituents on the two nitrogens: 1,2-, the lowest locant to the prefix cited first (g)):
    # 'chlorophenyl' (c) before 'methyl' (m); 'methyl' (m) before 'trichlorophenyl' (t, the first letter of
    # the complete name of a compound prefix,
    ("CNNc1ccc(Cl)cc1", "1-(4-chlorophenyl)-2-methylhydrazine"),
    ("CNNC1=C(C=C(C=C1Cl)Cl)Cl", "1-methyl-2-(2,4,6-trichlorophenyl)hydrazine"),
    ("CNNc1ccc(OC)cc1", "1-(4-methoxyphenyl)-2-methylhydrazine"),
    # 1,1- and 2,2-: the locants are essential on a hydrazine, the lowest set {1,1,2}
    ("CN(C)Nc1ccc(Cl)cc1", "2-(4-chlorophenyl)-1,1-dimethylhydrazine"),
    ("CN(C)N(C)c1ccc(Cl)cc1", "1-(4-chlorophenyl)-1,2,2-trimethylhydrazine"),
    # a ring of oxygen: O is junior to N (:18917), the ring is a prefix of the hydrazine
    ("C1CCOC1NNc1ccccc1", "1-(oxolan-2-yl)-2-phenylhydrazine"),
    ("COC1=CC=CC(=C1OC)NNC2CCCO2", "1-(2,3-dimethoxyphenyl)-2-(oxolan-2-yl)hydrazine"),
    ("NNC1CCOCC1", "(oxan-4-yl)hydrazine"),
    ("NNc1ccco1", "(furan-2-yl)hydrazine"),
    ("NNc1cccs1", "(thiophen-2-yl)hydrazine"),
    # an ether or sulfide in a chain or on a ring (class 41), a selenide
    ("COCCNNC", "1-(2-methoxyethyl)-2-methylhydrazine"),
    ("CNNc1ccc(SC)cc1", "1-methyl-2-[4-(methylsulfanyl)phenyl]hydrazine"),
    ("CNNc1ccc([Se]C)cc1", "1-methyl-2-[4-(methylselanyl)phenyl]hydrazine"),
    # the nitro group is a prefix only (a group of no class), a sulfone is class 41,:18228)
    ("CNNc1ccc([N+](=O)[O-])cc1", "1-methyl-2-(4-nitrophenyl)hydrazine"),
    ("NNc1ccc([N+](=O)[O-])cc1[N+](=O)[O-]", "(2,4-dinitrophenyl)hydrazine"),
    ("CNNc1ccc(S(C)(=O)=O)cc1", "1-[4-(methanesulfonyl)phenyl]-2-methylhydrazine"),
    # the nitroso group on carbon is a prefix only,:25933), a peroxide and its chalcogen analogues are
    # class 42,:18218), below the nitrogen compounds, class 21 (:18197): the prefixes are
    # 'nitroso', '(tert-butylperoxy)' (:23389), '(methyldisulfanyl)' (:27874), '(methoxysulfanyl)' (:27914)
    ("NNc1ccc(N=O)cc1", "(4-nitrosophenyl)hydrazine"),
    ("CC(C)(C)OOc1ccc(NN)cc1", "[4-(tert-butylperoxy)phenyl]hydrazine"),
    ("CNNc1ccc(OOC)cc1", "1-methyl-2-[4-(methylperoxy)phenyl]hydrazine"),
    ("NNc1ccc(SSC)cc1", "[4-(methyldisulfanyl)phenyl]hydrazine"),
    ("NNCCSSCC", "[2-(ethyldisulfanyl)ethyl]hydrazine"),
    ("NNc1ccc([Se][Se]C)cc1", "[4-(methyldiselanyl)phenyl]hydrazine"),
    ("NNc1ccc(SOC)cc1", "[4-(methoxysulfanyl)phenyl]hydrazine"),
    # the atoms of a parent hydride that the nitrogen outranks: Si, Sn (:18917 "N > P >... > Si >... > Sn")
    ("CNNc1ccc([Si](C)(C)C)cc1", "1-methyl-2-[4-(trimethylsilyl)phenyl]hydrazine"),
    ("CNNc1ccccc1[Sn](C)(C)C", "1-methyl-2-[2-(trimethylstannyl)phenyl]hydrazine"),
    ("CNNc1ccc2ccccc2p1", "1-methyl-2-(phosphinolin-2-yl)hydrazine"),
    # a compound prefix of the ring of a prefix is in its own marks,:7232 "parentheses around
    # compound and complex prefixes"): '3-(methoxymethyl)oxiran-2-yl', not '3-methoxymethyloxiran-2-yl'
    # (the spelling the ring-prefix sequence gave for every compound prefix built on 'methoxy', 'dimethylamino',
    # 'cyclooctyloxy'..., which no check read on a ring stem of the Hantzsch-Widman family)
    ("COCC1C(O1)NN", "[3-(methoxymethyl)oxiran-2-yl]hydrazine"),
    ("NNc1ccc(COC)cc1", "[4-(methoxymethyl)phenyl]hydrazine"),
    ("NNC1CCC(COC)CC1", "[4-(methoxymethyl)cyclohexyl]hydrazine"),
    ("NNc1ccc(OC2CCCCCCC2)cc1", "[4-(cyclooctyloxy)phenyl]hydrazine"),
    # halogens on a chain; the multiplier of a substituted prefix is 'bis'
    ("ClCCNNCCCl", "1,2-bis(2-chloroethyl)hydrazine"),
    ("NNC(F)(F)F", "(trifluoromethyl)hydrazine"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_the_hydrazine_is_the_parent_at_both_tiers(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


# the names that were right and stay (:38535 'phenylhydrazine (PIN)'; the others by the same rule)
UNCHANGED = [
    ("NNc1ccccc1", "phenylhydrazine"),
    ("CNNc1ccccc1", "1-methyl-2-phenylhydrazine"),
    ("CNNc1ccc(C)cc1", "1-methyl-2-(4-methylphenyl)hydrazine"),
    ("CCNN", "ethylhydrazine"),
    ("CNNC", "1,2-dimethylhydrazine"),
    ("CN(C)N", "1,1-dimethylhydrazine"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,pin", UNCHANGED)
def test_the_names_that_were_right_stay(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


# A ring of silicon, phosphorus or boron is junior to the nitrogen as well (:18917), but the default tier
# has no name for the saturated ring as a prefix ('silolan-2-yl' stands in the book's own example, which the
# strict ring-prefix producers name together with the 2H-pyran ring). The name is built, and the label
# follows the producer of the ring prefix: the default tier declines, the best-effort tier ships the
# hydrazine-parent name at pin_unverified ("a name in PIN form that only a breadth producer built").
BREADTH_RING_ROWS = [
    ("CNNC1CCC[SiH2]1", "1-methyl-2-(silolan-2-yl)hydrazine"),
    ("CNNC1CCCP1", "1-methyl-2-(phospholan-2-yl)hydrazine"),
    ("CNNC1CCCB1", "1-(borolan-2-yl)-2-methylhydrazine"),
]


# The same for a prefix that only the best-effort producers name: the default tier declines the molecule, where
# the name that cites 'hydrazinyl' ('1-(ethylperoxy)-2-hydrazinylethane') is no PIN.
BREADTH_PEROXIDE_ROWS = [
    ("NNCCOOCC", "[2-(ethylperoxy)ethyl]hydrazine"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", BREADTH_PEROXIDE_ROWS)
def test_a_peroxide_prefix_only_the_breadth_producers_name_is_not_labelled_a_pin(smiles, name):
    d = name_default(smiles)
    assert d.get("tier") != "pin_verified" and "hydrazinyl" not in (d.get("name") or ""), d
    b = name_breadth(smiles)
    assert (b.get("name"), b.get("tier")) == (name, "pin_unverified"), b


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", BREADTH_RING_ROWS)
def test_a_ring_prefix_only_the_breadth_producers_name_is_not_labelled_a_pin(smiles, name):
    d = name_default(smiles)
    assert d.get("tier") != "pin_verified", d
    b = name_breadth(smiles)
    assert (b.get("name"), b.get("tier")) == (name, "pin_unverified"), b


# The hydrazine is NOT the parent: a characteristic group of the classes 1 to 20 is present or a nitrogen
# ring is (a heterocycle precedes the hydrazines in class 21,:18197). The producer declines, and the
# names that were right stay: the hydrazinyl prefix of the ring, the amine, the acid.
NOT_THE_PARENT = [
    ("NNc1ccccn1", "2-hydrazinylpyridine"),                        #:19376 (PIN): a nitrogen ring
    ("CNNc1ccc(C(=O)O)cc1", "4-(2-methylhydrazin-1-yl)benzoic acid"),  # an acid is senior
    ("CNNc1ccc(O)cc1", "4-(2-methylhydrazin-1-yl)phenol"),         # a hydroxy group is senior
    ("CNNc1ccc(N)cc1", "4-(2-methylhydrazin-1-yl)aniline"),        # an amine is senior
    ("CNNc1ccc(C#N)cc1", "4-(2-methylhydrazin-1-yl)benzonitrile"),
    ("CC(=O)NN", "acetohydrazide"),                                # a hydrazide is no hydrazine (:18197)
]


@pytest.mark.parametrize("smiles", [s for s, _ in NOT_THE_PARENT])
def test_the_producer_declines_where_the_hydrazine_is_not_the_parent(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert not hydrazine_is_senior_parent(mol)
    assert name_polyazane(mol) is None


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", NOT_THE_PARENT)
def test_the_names_where_the_hydrazine_is_not_the_parent_stay(smiles, name):
    assert_pin_at_both_tiers(smiles, name)


# Two hydrazine units are the multiplicative case of, which the producer does not build: it declines,
# and the default tier with it (the name that cites 'hydrazinyl' is the one rules out)
@pytest.mark.parametrize("smiles", ["NNc1ccccc1NN", "NNc1ccc(NN)cc1"])
def test_two_hydrazine_units_are_left_to_the_multiplicative_case(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert name_polyazane(mol) is None


@pytest.mark.opsin_gate
def test_two_hydrazine_units_are_declined_at_the_default_tier():
    assert_declined_at_default("NNc1ccccc1NN")


# the structural question, on its own
@pytest.mark.parametrize("smiles,expected", [
    ("NNc1ccc(Cl)cc1", True),
    ("CNNc1ccc(OC)cc1", True),
    ("C1=COCC(NNC2CCC[SiH2]2)=C1", True),                  #:18950
    ("CNNc1ccc([N+](=O)[O-])cc1", True),
    ("CNNc1ccc(S(C)(=O)=O)cc1", True),
    ("CNNc1ccc([Si](C)(C)C)cc1", True),
    ("CNNC", True),
    ("NNc1ccc(N=O)cc1", True),                              # nitroso on carbon: a prefix only (:25933)
    ("CC(C)(C)OOc1ccc(NN)cc1", True),                       # a peroxide: class 42 (:18218)
    ("NNc1ccc(SSC)cc1", True),                              # its chalcogen analogues
    ("NNc1ccc([Se][Se]C)cc1", True),
    ("NNc1ccc(SOC)cc1", True),
    ("NNC1CSSC1", True),
    ("NNc1ccc(OO)cc1", False),                              # a hydroperoxide is class 18 (:18216)
    ("NNc1ccc(SS)cc1", False),                              # so is its chalcogen analogue
    ("NNc1ccc(SSSC)cc1", False),                            # three chalcogen atoms: no such group is read
    ("NNc1ccc(ON=O)cc1", False),                            # nitroso on oxygen: a nitrite, derivative of nitrous acid
    ("NNN=O", False),                                       # nitroso on nitrogen
    ("NNc1ccccn1", False),                                  # a nitrogen ring
    ("CNNc1ccc(O)cc1", False),                              # a hydroxy group
    ("CNNc1ccc(C(=O)O)cc1", False),                         # a carboxylic acid
    ("CC(=O)NN", False),                                    # a hydrazide
    ("CC(C)=NN", False),                                    # a hydrazone
    ("NNN", False),                                         # a polyazane (:39017)
    ("c1ccccc1N=Nc1ccccc1", False),                         # a diazene
    ("c1cc[nH]n1", False),                                  # the N-N bond is in a ring
    ("CNNc1ccc(Cl)cc1.O", False),                           # a second component
    ("CNNc1ccc([Pt](C)C)cc1", False),                       # an element that is not listed
])
def test_the_hydrazine_is_the_senior_parent(smiles, expected):
    assert hydrazine_is_senior_parent(Chem.MolFromSmiles(smiles)) is expected


# the producer asked directly, for the prefixes the in-process substituent namer reads
@pytest.mark.parametrize("smiles,expected", [
    ("NNc1ccc(Cl)cc1", "(4-chlorophenyl)hydrazine"),
    ("CNNc1ccc(Cl)cc1", "1-(4-chlorophenyl)-2-methylhydrazine"),
    ("CNNC1=C(C=C(C=C1Cl)Cl)Cl", "1-methyl-2-(2,4,6-trichlorophenyl)hydrazine"),
    ("C1CCOC1NNc1ccccc1", "1-(oxolan-2-yl)-2-phenylhydrazine"),
    ("NNc1ccc([N+](=O)[O-])cc1[N+](=O)[O-]", "(2,4-dinitrophenyl)hydrazine"),
])
def test_the_producer_names_the_hydrazine(smiles, expected):
    assert name_polyazane(Chem.MolFromSmiles(smiles)) == expected


def _ring_prefix(smiles):
    """The prefix of the substituent on the nitrogen that carries one, by the five-tier namer."""
    from orthonym.assembly.substituent_enumerator import name_substituent
    mol = Chem.MolFromSmiles(smiles)
    n = next(a for a in mol.GetAtoms()
             if a.GetSymbol() == "N" and any(x.GetSymbol() == "C" for x in a.GetNeighbors()))
    start = next(x.GetIdx() for x in n.GetNeighbors() if x.GetSymbol() == "C")
    frag, stack, seen = [], [start], {n.GetIdx()}
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        frag.append(i)
        stack.extend(x.GetIdx() for x in mol.GetAtomWithIdx(i).GetNeighbors() if x.GetSymbol() != "H")
    return name_substituent(mol, frag, start, allow_mancude=False)


# (the Blue Book): the compound prefix of a decorated ring prefix is in marks. The sequence
# of prefixes (rules/ring_prefix_sequence.py) decided "compound" by a list of functional prefixes of its own;
# it asks the one primitive of the project, naming_utils.enclose_if_compound, now.
@pytest.mark.parametrize("smiles,prefix", [
    ("COCC1C(O1)NN", "3-(methoxymethyl)oxiran-2-yl"),
    ("NNc1ccc(COC)cc1", "4-(methoxymethyl)phenyl"),
    ("NNC1CCC(COC)CC1", "4-(methoxymethyl)cyclohexyl"),
    ("NNc1ccc(OC2CCCCCCC2)cc1", "4-(cyclooctyloxy)phenyl"),
    ("NNc1ccc(CCl)cc1", "4-(chloromethyl)phenyl"),           # a prefix of the functional list: unchanged
    ("NNc1ccc(C)cc1", "4-methylphenyl"),                      # a simple prefix stays bare
    ("NNc1ccc(OC)cc1", "4-methoxyphenyl"),                    # methoxy is a simple prefix
])
def test_a_compound_prefix_of_a_ring_prefix_is_enclosed(smiles, prefix):
    assert _ring_prefix(smiles) == prefix
