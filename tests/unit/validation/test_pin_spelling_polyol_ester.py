"""DT3 step B: esters of a single polyol are named by functional class multiplicative nomenclature.

 (the Blue Book): "When anions are identical functional class multiplicative nomenclature is
used" ('propane-1,3-diyl bis(chloroacetate) (PIN)',:31825). "When anions are different, two
methods are used:" (:31831): "Method (1) generates preferred IUPAC names but names formed by using method (2) are
acceptable in general nomenclature" ('(1) propane-1,2,3-triyl 2-acetate 1-hexadecanoate 3-[(9Z)-octadec-9-enoate]
(PIN) (2) 2-(acetyloxy)-3-(hexadecanoyloxy)propyl (9Z)-octadec-9-enoate',:31846; '(1) 1,4-phenylene acetate
dichloroacetate (PIN) (2) 4-(acetyloxy)phenyl dichloroacetate'). A polyester of several alcoholic components, or
of a polyol that is no multivalent group of one parent, is named by (:31860 ff.) with acyloxy
prefixes: '2-(acetyloxy)ethyl methyl butanedioate (PIN)', '2-[2-(acetyloxy)ethyl]phenyl acetate (PIN)'.
The Blue Book's own PINs must not be flagged (the whole book is read in ``tests_local``).

A polyol of more than one parent is a multiplicative organyl group when the carbons that carry the ester oxygens are one
symmetry class of it, or when its arms are identical and so are the anions (the central group may carry substituent groups
of its own,,:5208),:31815;,:23178; (2),:5141): '1,2-phenylenedi(propan-3,1-yl)
diacetate [PIN; see (2) and ]' (:31829), '2-[3-(formyloxy)propyl]cyclohexane-1,1-diyldi(propane-3,1-diyl)
diacetate (PIN)' (:31878). With unequal arms the polyester is named by substitutive nomenclature, (:31856):
'2-{2-[(acetyloxy)methyl]phenyl}ethyl acetate (PIN)' (:31899). A ring system of several rings is one parent
('5α-cholestane-3β,6α-diyl diacetate',:31979).
"""
import pytest
from rdkit import Chem

from orthonym.validation.pin_spelling import ABSTAIN, PASS, check_pin_spelling
from orthonym.validation.spelling.checks_polyol_ester import (
    polyol_esters_by_acyloxy_prefixes, single_polyol_ester)

pytestmark = pytest.mark.unit

GLYCERIDE = "CCCCCCCCCCCCCCCC(=O)OC[C@H](OC(C)=O)COC(=O)CCCCCCC/C=C\\CCCCCCCC"


def _verdict(smiles, name):
    return polyol_esters_by_acyloxy_prefixes(Chem.MolFromSmiles(smiles), name)


@pytest.mark.parametrize("smiles, name", [
    (GLYCERIDE, "2-(acetyloxy)-3-(hexadecanoyloxy)propyl (9Z)-octadec-9-enoate"),          #:31846 (2)
    ("CC(=O)Oc1ccc(OC(=O)C(Cl)Cl)cc1", "4-(acetyloxy)phenyl dichloroacetate"),               #:31837 (2)
    ("CC(=O)OCOC=O", "(formyloxy)methyl acetate"),                                           #:31837 (2)
    ("CC(=O)OCC(OC(C)=O)COC(=O)CC", "2,3-bis(acetyloxy)propyl propanoate"),                  #:31837 (2)
    ("ClCC(=O)OCCCOC(=O)CCl", "3-[(chloroacetyl)oxy]propyl chloroacetate"),                  # identical anions::31825
])
def test_a_polyol_ester_named_by_acyloxy_prefixes_is_flagged(smiles, name):
    failures = check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True)
    assert [f.rule for f in failures if f.rule.startswith("P-65.6.3.3.3")] == ["P-65.6.3.3.3"], name


@pytest.mark.parametrize("smiles, name", [
    (GLYCERIDE, "propane-1,2,3-triyl 2-acetate 1-hexadecanoate 3-[(9Z)-octadec-9-enoate]"),    #:31846 (1) PIN
    ("ClCC(=O)OCCCOC(=O)CCl", "propane-1,3-diyl bis(chloroacetate)"),                           #:31825 PIN
    ("CC(=O)OCC(OC(C)=O)COC(C)=O", "propane-1,2,3-triyl triacetate"),                           #:31825 PIN
    ("CC(=O)Oc1ccc(OC(=O)C(Cl)Cl)cc1", "1,4-phenylene acetate dichloroacetate"),                #:31837 (1) PIN
    ("CC(=O)OCOC=O", "methylene acetate formate"),                                              #:31837 (1) PIN
])
def test_the_functional_class_multiplicative_name_passes(smiles, name):
    assert _verdict(smiles, name) == PASS


@pytest.mark.parametrize("smiles, name", [
    ("COC(=O)CCC(=O)OCCOC(C)=O", "2-(acetyloxy)ethyl methyl butanedioate"),                     #:31860 PIN
    ("CC(=O)OCCOC(=O)CC(=O)OCCOC(C)=O", "bis[2-(acetyloxy)ethyl] propanedioate"),               #:31864 PIN
    ("CCC(=O)OCCOC(=O)CCC(=O)OCCOC(C)=O", "2-(acetyloxy)ethyl 2-(propanoyloxy)ethyl butanedioate"),   #:31866
    ("CCOC(=O)CC(=O)Oc1ccc(OC(=O)CCC(=O)OC)cc1", "4-[(3-ethoxy-3-oxopropanoyl)oxy]phenyl methyl butanedioate"),
    ("CC(=O)Oc1ccc(-c2ccc(OC(C)=O)cn2)cc1", "6-[4-(acetyloxy)phenyl]pyridin-3-yl acetate"),      #:31895 PIN
    ("CC(=O)OCCc1ccccc1OC(C)=O", "2-[2-(acetyloxy)ethyl]phenyl acetate"),                       #:31897 PIN
    ("CCOC(=O)C(C)OC(C)=O", "ethyl 2-(acetyloxy)propanoate"),                                   # second ester on the acid
])
def test_the_polyesters_of_p_65_6_3_3_4_are_not_flagged(smiles, name):
    """The book's PINs for a polyester of several alcoholic components, or of a polyol of more than one parent,
    cite acyloxy prefixes: the check does not decide them."""
    assert _verdict(smiles, name) in (PASS, ABSTAIN)
    assert not [f for f in check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True)
                if f.rule.startswith("P-65.6.3.3.3")]


# the symmetrical polyol of two parents: ring, hetero atom or chain between identical arms
_LINKED = [
    ("CC(=O)OCc1ccccc1COC(C)=O", "{2-[(acetyloxy)methyl]phenyl}methyl acetate"),
    ("O=C(OCc1ccccc1COC(=O)c1ccccc1)c1ccccc1", "{2-[(benzoyloxy)methyl]phenyl}methyl benzoate"),
    ("CC(=O)OCC1CCCCC1COC(C)=O", "{2-[(acetyloxy)methyl]cyclohexyl}methyl acetate"),
    ("CC(=O)OCc1ccc(COC(C)=O)o1", "{5-[(acetyloxy)methyl]furan-2-yl}methyl acetate"),
    ("CC(=O)OCCc1ccccc1CCOC(C)=O", "2-{2-[2-(acetyloxy)ethyl]phenyl}ethyl acetate"),
    ("CC(=O)OCCC1CC1CCOC(C)=O", "2-{2-[2-(acetyloxy)ethyl]cyclopropyl}ethyl acetate"),
    ("CC(=O)OCc1cccc(COC(C)=O)c1", "{3-[(acetyloxy)methyl]phenyl}methyl acetate"),
    ("CC(=O)OCc1ccc(COC(C)=O)cc1", "{4-[(acetyloxy)methyl]phenyl}methyl acetate"),
    ("CC(=O)OCC1CCC(COC(C)=O)CC1", "{4-[(acetyloxy)methyl]cyclohexyl}methyl acetate"),
    ("CC(=O)OCCOCCOC(C)=O", "2-[2-(acetyloxy)ethoxy]ethyl acetate"),
    ("CC(=O)Oc1ccc(-c2ccc(OC(C)=O)cc2)cc1", "4'-(acetyloxy)[1,1'-biphenyl]-4-yl acetate"),
    # different anions: method (1) of is the PIN, the acyloxy form (method (2)) is for general nomenclature
    ("CCC(=O)OCc1ccccc1COC(C)=O", "{2-[(acetyloxy)methyl]phenyl}methyl propanoate"),
]


@pytest.mark.parametrize("smiles, name", _LINKED)
def test_a_symmetrical_polyol_of_two_parents_named_by_acyloxy_prefixes_is_flagged(smiles, name):
    assert single_polyol_ester(Chem.MolFromSmiles(smiles))[0] == "linked", smiles
    failures = check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True)
    assert [f.rule for f in failures if f.rule.startswith("P-65.6.3.3.3")] == ["P-65.6.3.3.3"], name


@pytest.mark.parametrize("smiles, name", [
    ("CC(=O)OCc1ccccc1COC(C)=O", "1,2-phenylenebis(methylene) diacetate"),
    ("CC(=O)OCCCc1ccccc1CCCOC(C)=O", "1,2-phenylenedi(propane-3,1-diyl) diacetate"),               #:31829
    ("CC(=O)OCC1CCC(COC(C)=O)CC1", "cyclohexane-1,4-diylbis(methylene) diacetate"),
    ("CC(=O)OCCOCCOC(C)=O", "oxydi(ethane-2,1-diyl) diacetate"),
])
def test_the_multiplicative_organyl_name_of_a_symmetrical_polyol_passes(smiles, name):
    assert _verdict(smiles, name) == PASS


@pytest.mark.parametrize("smiles, name", [
    ("CC(=O)OCCc1ccccc1COC(C)=O", "2-{2-[(acetyloxy)methyl]phenyl}ethyl acetate"),               #:31899 PIN
    ("CCC(=O)OCCc1ccccc1COC(C)=O", "2-{2-[(acetyloxy)methyl]phenyl}ethyl propanoate"),
    ("CC(=O)OCCCC1(CCCOC(C)=O)CCCCC1CCCOC=O",
     "2-[3-(formyloxy)propyl]cyclohexane-1,1-diyldi(propane-3,1-diyl) diacetate"),                 #:31878 PIN
])
def test_a_polyol_of_unequal_arms_is_not_decided(smiles, name):
    """ (:31856): unequal arms (or a third ester group of another acid) give the substitutive name."""
    assert single_polyol_ester(Chem.MolFromSmiles(smiles))[0] == "other"
    assert _verdict(smiles, name) in (PASS, ABSTAIN)
    assert not [f for f in check_pin_spelling(Chem.MolFromSmiles(smiles), name, strict=True)
                if f.rule.startswith("P-65.6.3.3.3")]


_STEROID = "CC(=O)O[C@H]1CC[C@@]2(C)C(=CC[C@@H]3[C@@H]2CC[C@]2(C)[C@@H](OC(C)=O)CC[C@@H]32)C1"


def test_the_rings_of_a_ring_system_are_one_parent():
    """A steroid diacetate: the ring system is one parent ('5α-cholestane-3β,6α-diyl diacetate',:31979), so the name
    that cites one acetate as a prefix is not the functional class name of."""
    assert single_polyol_ester(Chem.MolFromSmiles(_STEROID)) == ("single", 2)
    name = "(3S,8R,9S,10R,13S,14S,17S)-17-(acetyloxy)-10,13-dimethyl-tetradecahydro-1H-cyclopenta[a]phenanthren-3-yl acetate"
    assert getattr(_verdict(_STEROID, name), "rule", None) == "P-65.6.3.3.3"
    assert _verdict(_STEROID, "1H-cyclopenta[a]phenanthrene-3,17-diyl diacetate") == PASS
    # two rings joined by a bond are two parents: a symmetrical pair is a linked polyol
    assert single_polyol_ester(Chem.MolFromSmiles("CC(=O)OC1CCC(CC1)C1CCC(OC(C)=O)CC1"))[0] == "linked"


def test_where_the_esters_sit():
    assert single_polyol_ester(Chem.MolFromSmiles(GLYCERIDE)) == ("single", 3)
    assert single_polyol_ester(Chem.MolFromSmiles("CCOC(=O)C(C)OC(C)=O"))[0] == "none"          # one per alcohol
    assert single_polyol_ester(Chem.MolFromSmiles("COC(=O)CCC(=O)OCCOC(C)=O"))[0] == "other"
    assert single_polyol_ester(Chem.MolFromSmiles("CCO"))[0] == "none"
    # a lactone is a heterocycle, not an ester of the class,:18182 "lactones and other cyclic esters
    # are named as heterocycles"): its ring ester is left out, and the open-chain esters beside it are read
    assert single_polyol_ester(Chem.MolFromSmiles("O=C1CCCCO1"))[0] == "none"
    assert single_polyol_ester(Chem.MolFromSmiles("O=C1OCCOC(=O)CCCC1"))[0] == "none"
    assert single_polyol_ester(Chem.MolFromSmiles("CC(=O)OCC(OC(C)=O)C1CCC(=O)O1")) == ("single", 2)


def test_a_molecule_with_a_group_senior_to_the_esters_is_not_decided():
    """ (:18162): an acid is senior to the esters; the esters are then prefixes,:31696)."""
    smiles = "CC(=O)OCC(OC(C)=O)C(=O)O"
    assert single_polyol_ester(Chem.MolFromSmiles(smiles)) == ("single", 2)
    assert _verdict(smiles, "2,3-bis(acetyloxy)propyl formate") == ABSTAIN          # a name an acid would not have


def test_a_name_of_the_p_10_classes_and_a_name_without_an_ester_word_abstain():
    assert _verdict(GLYCERIDE, "1,2-di-O-acetyl-3-O-hexadecanoyl-sn-glycerol glycerate") == ABSTAIN
    assert _verdict("CC(=O)OCC(OC(C)=O)COC(C)=O", "glycerol triacetate trimer") == ABSTAIN
    assert _verdict("CC(=O)OCC(OC(C)=O)COC(C)=O", "2,3-bis(acetyloxy)propyl") == ABSTAIN


def test_the_esters_beside_a_lactone_are_read():
    """A macrolactone with two acetates on its ring: the lactone is a heterocycle,:18182), the two
    acetates are esters of one polyol (the ring), and a name that cites one of them as an acyloxy prefix is not the
    PIN,:31819: identical anions take the multiplicative '...-9,13-diyl diacetate'). Before the
    lactone was left out, the check abstained on the molecule and the name shipped as pin_verified."""
    smiles = ("CC(=O)O[C@H]1[C@@H](C)[C@H](C)OC(=O)[C@H](Cc2ccccc2)N(C)C(=O)/C=C\\[C@@H](C)[C@H](OC(C)=O)"
              "/C(C)=C\\[C@@H]1C")
    name = ("(3S,6Z,8R,9S,10Z,12S,13R,14S,15S)-13-(acetyloxy)-3-benzyl-4,8,10,12,14,15-hexamethyl-2,5-dioxo-"
            "1-oxa-4-azacyclopentadeca-6,10-dien-9-yl acetate")
    assert single_polyol_ester(Chem.MolFromSmiles(smiles)) == ("single", 2)
    assert getattr(_verdict(smiles, name), "rule", None) == "P-65.6.3.3.3"


@pytest.mark.parametrize("smiles, old_name, group_name", [
    ("CC(=O)OCc1ccc(Cl)c(COC(C)=O)c1", "{3-[(acetyloxy)methyl]-4-chlorophenyl}methyl acetate",
     "(4-chloro-1,3-phenylene)bis(methylene) diacetate"),
    ("CC(=O)OCc1ccc(C)c(COC(C)=O)c1", "{3-[(acetyloxy)methyl]-4-methylphenyl}methyl acetate",
     "(4-methyl-1,3-phenylene)bis(methylene) diacetate"),
    ("CC(=O)OCc1ccc(O)c(COC(C)=O)c1", "{3-[(acetyloxy)methyl]-4-hydroxyphenyl}methyl acetate",
     "(4-hydroxy-1,3-phenylene)bis(methylene) diacetate"),
    ("CC(=O)OCc1cc(COC(C)=O)n[nH]1", "{3-[(acetyloxy)methyl]-1H-pyrazol-5-yl}methyl acetate",
     "1H-pyrazole-3,5-diylbis(methylene) diacetate"),
    ("CC(=O)OCc1ccc(Cl)cc1COC(C)=O", "{2-[(acetyloxy)methyl]-4-chlorophenyl}methyl acetate",
     "(4-chloro-1,2-phenylene)bis(methylene) diacetate"),
])
def test_identical_arms_on_a_substituted_central_group_are_a_linked_polyol(smiles, old_name, group_name):
    """ (:5208) "Substitution, expressed by prefixes or implied, is allowed on simple multiplicative groups,
    symmetrical or unsymmetrical", with (1) (:23178) identical linking bonds: the arms (a methylene group and an
    ester oxygen) are identical in a polyol whose central ring carries a chloro, methyl or hydroxy group, so the whole
    polyol need not be symmetrical. The name that cites one arm as an acyloxy prefix is no PIN."""
    mol = Chem.MolFromSmiles(smiles)
    assert single_polyol_ester(mol) == ("linked", 2)
    assert getattr(_verdict(smiles, old_name), "rule", None) == "P-65.6.3.3.3"
    assert _verdict(smiles, group_name) == PASS


@pytest.mark.parametrize("smiles", [
    "CC(=O)OCc1ccc(Cl)c(COC(=O)CC)c1",          # identical arms, two anions, a polyol with no symmetry: no locant of the
                                                # group says which arm carries which anion (:31878)
    "CC(=O)OCCc1ccc(Cl)c(COC(C)=O)c1",          # unequal arms
    "CC(=O)OCCc1ccc(Cl)cc1OC(C)=O",             # a ring and a chain
    "CC(=O)O[13CH2]c1ccc(Cl)c(COC(C)=O)c1",     # arms a nuclide makes unlike
])
def test_unlike_arms_or_anions_leave_the_substituted_polyol_undecided(smiles):
    assert single_polyol_ester(Chem.MolFromSmiles(smiles))[0] == "other"


@pytest.mark.parametrize("smiles, name", [
    ("O=C(OCCSC(C)=O)N1CCCCC1", "2-(acetylsulfanyl)ethyl piperidine-1-carboxylate"),
    ("CC(=O)OCCSC(C)=O", "2-(acetylsulfanyl)ethyl acetate"),
    ("CC(=O)SCCSC(C)=O", "S-[2-(acetylsulfanyl)ethyl] ethanethioate"),
    ("CC(=O)SC1CCC(CC1)OC(C)=O", "4-(acetylsulfanyl)cyclohexyl acetate"),
])
def test_an_s_ester_end_of_one_polyol_is_an_ester_group_too(smiles, name):
    """ "Definitions" (:31649): esters of "chalcogen analogues of alcohols (thiols, selenols, tellurols)
    ... are also included"; sets no oxygen-only condition, so the method (2) name with an
    'acylsulfanyl' prefix is no PIN (the book prints no method (1) name with an S end; ruling 2026-11-07)."""
    assert polyol_esters_by_acyloxy_prefixes(Chem.MolFromSmiles(smiles), name) not in (PASS, ABSTAIN)


@pytest.mark.parametrize("smiles, name", [
    ("CC(=O)SCC(=O)OC", "methyl (acetylsulfanyl)acetate"),      # the S-ester's 'alcohol' is the acid side: two components
    ("CC(=O)SCCO", "S-(2-hydroxyethyl) ethanethioate"),         # one ester group
    # an S-carbamoyl group is no ester of the polyol: the multiplicative name of the two acetates is the PIN
    ("CC(=O)OCC(COC(C)=O)SC(=O)N", "2-(carbamoylsulfanyl)propane-1,3-diyl diacetate"),
    ("CC(=O)Oc1ccc(SC(=O)N(C)C)cc1OC(C)=O", "4-[(dimethylcarbamoyl)sulfanyl]-1,2-phenylene diacetate"),
])
def test_an_s_ester_on_another_component_is_not_flagged(smiles, name):
    assert polyol_esters_by_acyloxy_prefixes(Chem.MolFromSmiles(smiles), name) is PASS


def test_a_simple_acyl_on_sulfur_is_not_enclosed():
    """ "Parentheses" (:7232) encloses compound prefixes only: 'carbamoylamino (preferred prefix)' (:33344),
    so 'carbamoylsulfanyl', not '(carbamoyl)sulfanyl'; a compound acyl is enclosed, '(cyclohexanecarbonyl)sulfanyl
    (preferred prefix)' (:24615)."""
    from tests.support.pin_tiers import tier_row
    assert tier_row("NC(=O)SCCC(=O)O", "pin")["name"] == "3-(carbamoylsulfanyl)propanoic acid"
    assert tier_row("OC(=O)c1ccc(SC(=O)C2CCCCC2)cc1", "pin")["name"] == \
        "4-[(cyclohexanecarbonyl)sulfanyl]benzoic acid"
