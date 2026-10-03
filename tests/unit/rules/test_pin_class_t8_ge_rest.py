"""PIN class program, Task 8: the remaining general-engine rows at the PIN tier.

Chalcogen-linked substituents on a benzene ring
-----------------------------------------------
 "Names of chalcogen analogues of ethers, i.e., sulfides, selenides and tellurides"
(the Blue Book): "(1) by prefixing the names of the substituent groups R'-S-, R'-Se-, or
R'-Te-, i.e., R'-sulfanyl, R'-selanyl, and R' tellanyl, respectively, to that of the parent
hydride, RH" (:27808); "Method (1), substitutive nomenclature, gives preferred IUPAC names"
(:27817). '(cyclopentylselanyl)benzene (PIN)' (:27834), '1-chloro-4-[(chloromethyl)selanyl]
benzene (PIN) (not α,4-dichloroselenoanisole)' (:27850), '(methylsulfanyl)benzene (PIN)'
(:27848).

 "Retained names" (:27665): '1-(chloromethoxy)-4-nitrobenzene (PIN; no substitution
on anisole for PINs)' (:27711).

 (:7232): "Parentheses are used around compound (see and complex (see
 prefixes".

At the base the benzene substituent collector had no branch for R-Se- / R-Te-, and for an R-O- /
R-S- whose R carries another atom (a halogen) it built a name that drops that atom ('1-methoxy-
4-nitrobenzene' for the chloromethoxy row); only the round trip held it back, so the default
tier declined and best-effort shipped the general engine's name at systematic_verified.

An R that holds a benzene ring ('(phenylselanyl)benzene') is not the collector's:
method (3) (:27810) and '1,1'-sulfanediyldibenzene (PIN)' (:27826) make the multiplicative name
the PIN for two identical rings; the multiplicative producer builds it (PIN class program,
Task 10).

A heteroatom group on a branch carbon of a ring parent
------------------------------------------------------
 (:43718): "The name of an isotopically substituted compound is formed by adding or
inserting the nuclide symbol(s) enclosed in parentheses, preceded by any necessary locant(s),
letters, and/or numerals, before the part of the compound that is isotopically substituted.";
'1-[amino(14C)methyl]cyclopentan-1-ol (PIN)' (:43742). The amino group sits on the branch
carbon, so the prefix is the whole branch, '(aminomethyl)', as '3-(2-iminopropyl)cyclohexane-
1-carboxylic acid (PIN)',:26549) cites its imino group inside the branch and
'2-(aminomethyl)propane-1,3-diamine (PIN)' (:26302) cites 'aminomethyl'.

At the base the ring-parent polyfunctional builder took the carbon of an -NH2 / -SH / -PH2 /
-NH-R match as the group's anchor when that carbon was a branch atom bonded to the ring, cited
the bare prefix ('amino') at the ring atom and dropped the branch carbon ('1-aminocyclopentan-
1-ol' for the unlabelled row); only the round trip held it back.

A carbon joined to a heteroatom of the match by a multiple bond stays part of the group:
"Acyl halides and pseudohalides as substituent groups" (:31517): "(1) by a prefix formed from the
name of the acid, for example, 'carbonochloridoyl'" (:31521); '(1) 2-carbonochloridoylbenzoic acid
(PIN)' (:31533), so '2-carbonochloridoylcyclopentane-1-carboxylic acid' is a control.

Not built here: -CH=NH on a ring parent. Its PIN prefix is 'methanimidoyl (preferred prefix)'
,:30468), but the ring-parent builder cites 'imino' at the ring atom ('3-imino-
cyclohexan-1-ol' for OC1CCCC(C1)C=N, another molecule); only the round trip holds it back, at the
base as now. The test guards that it is never labelled a PIN.

The 18O row of the same molecule, '1-(aminomethyl)cyclopentan-1-(18O)ol (PIN)' (:43744): the
nuclide on the oxygen of the '-ol' suffix goes immediately before that suffix,:43718,
"before the part of the compound that is isotopically substituted"); rules/isotopes.py places it
there (batch 2 fix a performance pass; the class test is test_pin_class_b2_isotope_xh_suffix.py).

A chain parent with a ring substituent and a halogen on the chain
-----------------------------------------------------------------
 (:3448): "Simple prefixes (i.e., those describing atoms and unsubstituted substituents)
are arranged alphabetically; multiplicative prefixes, if necessary, are then inserted and do
not alter the alphabetical order already established."; '3,3-dibromo-3-cyclohexylpropanoic acid
(PIN) (not 3-cyclohexyl-3,3-dibromopropanoic acid)' (:3459). (:3031) with
: 'bromo(chloro)acetic acid (PIN)' (:7312).

At the base, with the chain as the parent, the prefix builder read the ring-as-parent
substituent map (in which the chain itself is a branch of the ring) and took the chain's own
halogen for a group already cited inside a ring substituent, so it dropped it ('3-cyclohexyl-
propanoic acid'); only the round trip held it back.

A secondary alcohol on a substituent branch of a chain parent
-------------------------------------------------------------
 (:21791): "The preferred IUPAC name is based on the senior parent structure that has
the lower locant or set of locants for substituents cited as prefixes to the parent structure
(other than 'hydro/dehydro' prefixes) in their order of citation in the name.";
'3-bromo-2-(2-bromo-1-hydroxyethyl)-4-hydroxybutanoic acid (PIN) [not 4-bromo-2-(1-bromo-2-
hydroxyethyl)-3-hydroxybutanoic acid;...]' (:22108). The chain choice was already right (the
 key of perception/chains.py); the OH of the branch was cited a second time on the
chain ('...-2,4-dihydroxybutanoic acid'): the secondary-alcohol match also holds the carbinol
carbon's two neighbours, one of them the chain atom the branch hangs from, so the branch filter
kept it.

Rows already right at this task's base (protection)
---------------------------------------------------
 (:27274): '5-(1-hydroxy-2-sulfanylethyl)-2-sulfanylcyclohexan-1-ol (PIN) (ring
preferred to chain, see ' (:27599). (:9158): "The symbol λn, where n is the
bonding number, is cited immediately after the locant denoting the heteroatom with the
nonstandard bonding number."; '1-oxa-4λ4-thiacyclotetradecane (PIN)' (:9486), '1-oxa-
4,8λ4-dithiacyclododecane (PIN)' (:9492).

Owned by the large-polycycle plan (its Task 6, " von Baeyer PINs where fusion names
are not allowed")
------------------------------------------------------------------------------------------
 (:16633): 'bicyclo[4.1.0]hepta-1,3,5-triene (PIN) [not bicyclo[4.1.0]hepta-
1(6),2,4-triene]' (:16653). Not built here; best-effort keeps its RT-exact name.
"""
import pytest

from tests.support.pin_tiers import (
    assert_not_pin_labelled,
    assert_pin_at_both_tiers,
    name_breadth,
)
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

PIN_ROWS = [
    ("ClC[Se]c1ccc(Cl)cc1", "1-chloro-4-[(chloromethyl)selanyl]benzene"),              #:27850
    ("c1ccc([Se]C2CCCC2)cc1", "(cyclopentylselanyl)benzene"),                          #:27834
    ("O=[N+]([O-])c1ccc(OCCl)cc1", "1-(chloromethoxy)-4-nitrobenzene"),                #:27711
    ("N[14CH2]C1(O)CCCC1", "1-[amino(14C)methyl]cyclopentan-1-ol"),                    #:43742
    ("NCC1([18OH])CCCC1", "1-(aminomethyl)cyclopentan-1-(18O)ol"),                     #:43744
    ("O=C(O)CC(Br)(Br)C1CCCCC1", "3,3-dibromo-3-cyclohexylpropanoic acid"),            #:3459
    ("O=C(O)C(C(O)CBr)C(Br)CO", "3-bromo-2-(2-bromo-1-hydroxyethyl)-4-hydroxybutanoic acid"),  #:22108
]

# Class members (same producer, same rules); the base declined each at the default tier.
CLASS_ROWS = [
    ("ClC[Se]c1ccccc1", "[(chloromethyl)selanyl]benzene"),
    ("C[Se]c1ccccc1", "(methylselanyl)benzene"),                    # as '(methylsulfanyl)benzene':27848
    ("C[Se]c1ccc(Cl)cc1", "1-chloro-4-(methylselanyl)benzene"),
    ("c1ccc([Se]C2CCCCC2)cc1", "(cyclohexylselanyl)benzene"),
    ("c1ccc([Se]C(C)C)cc1", "[(propan-2-yl)selanyl]benzene"),      # as '1-[(propan-2-yl)selanyl]-...':27840
    ("C[Te]c1ccccc1", "(methyltellanyl)benzene"),
    ("ClCSc1ccc(Cl)cc1", "1-chloro-4-[(chloromethyl)sulfanyl]benzene"),
    ("O=[N+]([O-])c1ccc(SCCl)cc1", "1-[(chloromethyl)sulfanyl]-4-nitrobenzene"),
    ("c1ccc(SC(F)(F)F)cc1", "[(trifluoromethyl)sulfanyl]benzene"),
    ("ClCOc1ccccc1", "(chloromethoxy)benzene"),
    ("ClCOc1ccc(Cl)cc1", "1-chloro-4-(chloromethoxy)benzene"),
    ("c1ccc(OC(F)(F)F)cc1", "(trifluoromethoxy)benzene"),
    ("c1ccc(OCC(Cl)Cl)cc1", "(2,2-dichloroethoxy)benzene"),
    ("Clc1ccc(OCCCl)cc1", "1-chloro-4-(2-chloroethoxy)benzene"),
    ("OC(=O)c1ccc(OC(F)F)cc1", "4-(difluoromethoxy)benzoic acid"),
    ("FC(F)(F)Oc1cc(OC(F)(F)F)ccc1", "1,3-bis(trifluoromethoxy)benzene"),
    # a heteroatom group on a branch carbon of a ring parent: the prefix is the branch
    ("OC1(CN)CCCC1", "1-(aminomethyl)cyclopentan-1-ol"),
    ("N[13CH2]C1(O)CCCC1", "1-[amino(13C)methyl]cyclopentan-1-ol"),
    ("NCC1(O)CCCCC1", "1-(aminomethyl)cyclohexan-1-ol"),
    ("OC1(CN)CCC1", "1-(aminomethyl)cyclobutan-1-ol"),
    ("NCC1CCCCC1O", "2-(aminomethyl)cyclohexan-1-ol"),
    ("NC[C@H]1CCCC[C@@H]1O", "(1S,2R)-2-(aminomethyl)cyclohexan-1-ol"),
    ("OC1CCC(CN)CC1", "4-(aminomethyl)cyclohexan-1-ol"),
    ("OC1CC(CN)C1", "3-(aminomethyl)cyclobutan-1-ol"),
    ("O=C1CCCCC1CN", "2-(aminomethyl)cyclohexan-1-one"),
    ("OC(=O)C1(CN)CCCC1", "1-(aminomethyl)cyclopentane-1-carboxylic acid"),
    ("OC(=O)C1CCC(CN)CC1", "4-(aminomethyl)cyclohexane-1-carboxylic acid"),
    ("OC1(CNC)CCCC1", "1-[(methylamino)methyl]cyclopentan-1-ol"),
    ("OC1(CN(C)C)CCCC1", "1-[(dimethylamino)methyl]cyclopentan-1-ol"),
    ("OC1(CNCl)CCCC1", "1-[(chloroamino)methyl]cyclopentan-1-ol"),
    ("OC1(CS)CCCC1", "1-(sulfanylmethyl)cyclopentan-1-ol"),
    ("OC(=O)C1CCC(CS)CC1", "4-(sulfanylmethyl)cyclohexane-1-carboxylic acid"),
    ("OC1(CP)CCCC1", "1-(phosphanylmethyl)cyclopentan-1-ol"),
    # a chain parent with a ring substituent and a halogen on the chain
    ("O=C(O)CC(Br)C1CCCCC1", "3-bromo-3-cyclohexylpropanoic acid"),
    ("O=C(O)CC(Cl)C1CCCCC1", "3-chloro-3-cyclohexylpropanoic acid"),
    ("O=C(O)CC(Br)C1CCCC1", "3-bromo-3-cyclopentylpropanoic acid"),
    ("O=C(O)CC(F)(F)C1CCCC1", "3-cyclopentyl-3,3-difluoropropanoic acid"),
    ("CC(Br)(C1CCCCC1)CC(=O)O", "3-bromo-3-cyclohexylbutanoic acid"),
    ("O=C(O)CCC(Cl)C1CCCCC1", "4-chloro-4-cyclohexylbutanoic acid"),
    ("O=C(O)CC(Br)C1CCCCC1Cl", "3-bromo-3-(2-chlorocyclohexyl)propanoic acid"),
    ("O=C(O)C(Br)C1CCCCC1", "bromo(cyclohexyl)acetic acid"),                # as:7312
    ("ClC(C1CCCCC1)CC(=O)N", "3-chloro-3-cyclohexylpropanamide"),
    ("OCC(Br)C1CCCCC1", "2-bromo-2-cyclohexylethan-1-ol"),
    ("NCC(Cl)C1CCCCC1", "2-chloro-2-cyclohexylethan-1-amine"),
    # a secondary alcohol on a substituent branch of a chain parent
    ("BrCC(O)C(C(O)=O)C(Br)CO", "3-bromo-2-(2-bromo-1-hydroxyethyl)-4-hydroxybutanoic acid"),
    ("O=C(O)C(C(O)CCl)C(Cl)CO", "3-chloro-2-(2-chloro-1-hydroxyethyl)-4-hydroxybutanoic acid"),
    ("O=C(O)C(C(O)C)C(C)O", "3-hydroxy-2-(1-hydroxyethyl)butanoic acid"),
    ("OC(=O)CC(C(O)CBr)C(Br)CO", "4-bromo-3-(2-bromo-1-hydroxyethyl)-5-hydroxypentanoic acid"),
    ("CC(O)C(CCC(=O)O)C(C)O", "5-hydroxy-4-(1-hydroxyethyl)hexanoic acid"),
]

# Already the PIN at both tiers at the base (protection).
CONTROL_ROWS = [
    ("OC1CC(C(O)CS)CCC1S", "5-(1-hydroxy-2-sulfanylethyl)-2-sulfanylcyclohexan-1-ol"),  #:27599
    ("C1CCCCC[SH2]CCOCCCC1", "1-oxa-4λ4-thiacyclotetradecane"),                        #:9486
    ("C1CC[SH2]CCCSCCOC1", "1-oxa-4,8λ4-dithiacyclododecane"),                         #:9492
    ("CSc1ccc(Cl)cc1", "1-chloro-4-(methylsulfanyl)benzene"),
    ("CC(=O)c1ccc(OCCl)cc1", "1-[4-(chloromethoxy)phenyl]ethan-1-one"),
    ("c1ccc(Sc2ccccc2)cc1", "1,1'-sulfanediyldibenzene"),                              #:27826
    ("O=[N+]([O-])c1ccc(OC)cc1", "1-methoxy-4-nitrobenzene"),
    # the group's own carbon or nitrogen on the ring, or a group a branch deeper
    ("CC(=N)CC1CCCC(C(=O)O)C1", "3-(2-iminopropyl)cyclohexane-1-carboxylic acid"),  #:26549
    ("OC(=O)C1CCCC1C#N", "2-cyanocyclopentane-1-carboxylic acid"),
    ("OC(=O)C1CCCC1C(=O)Cl", "2-carbonochloridoylcyclopentane-1-carboxylic acid"),  # as:31533
    ("OC1CCCCC1N", "2-aminocyclohexan-1-ol"),
    ("OC1(CCN)CCCC1", "1-(2-aminoethyl)cyclopentan-1-ol"),
    ("OC1(C[N+](=O)[O-])CCCC1", "1-(nitromethyl)cyclopentan-1-ol"),
    ("OC1(CO)CCCC1", "1-(hydroxymethyl)cyclopentan-1-ol"),
    ("NCc1ccccc1O", "2-(aminomethyl)phenol"),
    ("OC(=O)c1ccc(CN)cc1", "4-(aminomethyl)benzoic acid"),
    ("O=C(O)CC(Cl)(Cl)c1ccccc1", "3,3-dichloro-3-phenylpropanoic acid"),
    ("O=C(O)CC(O)C1CCCCC1", "3-cyclohexyl-3-hydroxypropanoic acid"),
    ("O=C(O)CC(C)C1CCCCC1", "3-cyclohexylbutanoic acid"),
    ("O=C(O)C(C(O)CBr)C(Br)C", "4-bromo-2-(1-bromoethyl)-3-hydroxybutanoic acid"),
    ("O=C(O)C(CO)C(O)CBr", "4-bromo-3-hydroxy-2-(hydroxymethyl)butanoic acid"),
    ("O=C(O)C(C(O)CC)CCC", "3-hydroxy-2-propylpentanoic acid"),
    ("OCC(C(O)C)CCC", "2-propylbutane-1,3-diol"),
]

# Two identical rings: method (3) (:27810), the multiplicative PIN (Task 10).
MULTIPLICATIVE_ROWS = [
    ("c1ccc([Se]c2ccccc2)cc1", "1,1'-selanediyldibenzene"),
    ("c1ccc([Se]CC[Se]c2ccccc2)cc1", "1,1'-[ethane-1,2-diylbis(selanediyl)]dibenzene"),
]

# (smiles, a name of another molecule that the ring-parent builder constructs): never shipped as
# a PIN, and best-effort stays RT-exact. -CH=NH on the ring is the acyl group 'methanimidoyl
# (preferred prefix)',:30468); citing 'imino' at the ring atom drops the CH.
WRONG_CANDIDATE_ROWS = [
    ("OC1CCCC(C1)C=N", "3-iminocyclohexan-1-ol"),
]

# (smiles, PIN, reason): the PIN needs a producer this task does not build.
OWNED_ROWS = [
    ("c1ccc2c(c1)C2", "bicyclo[4.1.0]hepta-1,3,5-triene",
     "needs the large-polycycle plan Task 6: P-52.2.4.1 von Baeyer PINs where fusion names "
     "are not allowed (:16653, :23718)"),
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", CLASS_ROWS)
def test_class_member(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", CONTROL_ROWS)
def test_control(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", MULTIPLICATIVE_ROWS)
def test_two_identical_rings_multiplicative_pin(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,non_pin", WRONG_CANDIDATE_ROWS)
def test_wrong_candidate_never_labelled_pin(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)


@pytest.mark.parametrize("smiles,pin,owner", OWNED_ROWS)
def test_owned_row_best_effort_rt_exact(smiles, pin, owner):
    b = name_breadth(smiles)
    assert b.get("name") and name_is_rt_exact(b["name"], smiles), b


@pytest.mark.parametrize("smiles,pin,owner", [
    pytest.param(s, p, o, marks=pytest.mark.xfail(strict=True, reason=o)) for s, p, o in OWNED_ROWS
])
def test_owned_row_pin(smiles, pin, owner):
    assert_pin_at_both_tiers(smiles, pin)
