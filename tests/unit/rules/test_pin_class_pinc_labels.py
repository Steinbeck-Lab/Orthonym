"""PIN class program, the rest (lane PINC): names that are valid and read back but are no PIN, so the PIN tier
declines them and the best-effort tier keeps them read back (``assert_declined_at_default``); each row has the
controls of the same class that stay PINs.

* Added hydrogen at a ring nitrogen: "'Added indicated hydrogen'" (the Blue Book), "used to introduce a
  free valence, a radical or an ionic center, or a principal characteristic group into a fully unsaturated... fused
  polycyclic system in the absence of, or lack of, sufficient hydrogen atoms... or on equivalent heteroatoms such as
  -N=": '1-(3,4-dihydroquinolin-1(2H)-yl)ethan-1-one (PIN)',:29376; the plan's rows are:7467 and:33131),
  not '1-(1,2,3,4-tetrahydroquinolin-1-yl)ethan-1-one'. 'hydro' covers the other positions.
* Chalcone: under KETONES, Retained names (:28297), "The name 'chalcone' is the only retained name as a
  preferred IUPAC name and is limited to ring substitution only by characteristic groups lower than 'ketone'",
  "2',4'-dihydroxy-3,3'-dimethoxychalcone (PIN)" (:28303); '4-[(1E)-3-(2,4-dihydroxyphenyl)-3-oxoprop-1-en-1-yl]
  benzamide (PIN) (not 2',4'-dihydroxychalcone-4-carboxamide)' (:28305).
* Parent chain of two amide chains: (:21698), '3-[5-(3-amino-2-methyl-3-oxopropyl)-2-methylphenyl]-N-methyl
  propanamide (PIN) [not 2-methyl-3-{4-methyl-3-[3-(methylamino)-3-oxopropyl]phenyl}propanamide; the locant set
  'N,3' in the PIN is lower than '2,3']' (:21730); (:20916) '(b) has the greater number of skeletal atoms'.
* Diacyl chalcogen chain: (:32355), '1-[(acetylperoxy)sulfanyl]ethan-1-one (PIN) {not 1-[(acetylsulfanyl)
  peroxy]ethan-1-one; the PIN has the lower alphanumerical order}' (:32368); with two chains of different length the
  longer one is the parent, (:20916), '1-[(acetylsulfanyl)peroxy]propan-1-one (PIN)' (:39499).
* Halides and pseudohalides of noncarbon oxoacids: (:35706), '(CH3)2N-P(O)(NCO)Cl N,N-dimethylphosphor
  amidisocyanatidic chloride (PIN)' (:35730); ORGANOMETALLIC COMPOUNDS INVOLVING THE ELEMENTS OF GROUPS 13 THROUGH
  16 (:39737), 'BrSb(CH=CH2)2 bromodi(ethenyl)stibane (PIN; substitutive name)' (:39747), (:39053);
  'diphenylstibinous isocyanate (PIN)' (:35714).
* Arsinate and stibinate esters: (:46008) and its examples (:46022) are those of the phosphates, phosphonates and
  related compounds of phosphorus; no rule sentence or example covers an arsenic or antimony acid, so the descriptor
  form 'methyl (S)-[methyl(phenyl)arsinate]' is no verified PIN (the placement of the descriptor in the phosphonate
  diesters, 'ethyl methyl (R)-(methylphosphonate)', is ASSUMED from the phosphinate example, which the book prints).
* Amine oxides: (:26612), "amine and imine oxides are named by method (1) and all other amino groups, if present,
  are named as substituent groups by using the prefix 'amino'", '3,5-bis(dimethylamino)-N,N-dimethylpentan-1-amine N-oxide
  (PIN)' (:26632), '2-(3-carbamoyl-5-carboxyphenyl)ethan-1-amine N-oxide (PIN)' (:26636).
* '-dithioperoxol': (:27963), Table 6.1 (:27981), 'ethanedithioperoxol (PIN)' (:27998),
  'benzene-1,4-bis(dithioperoxolate) (PIN)' (:41041); (e) (:7134,:7196) 'bis(disulfanyl)... (not
  didisulfanyl)'; the prefix 'disulfanyl' of a group that is not the principal one, '3,4-bis(disulfanyl)benzamide (PIN)'
  (:28021).
"""
import pytest

from tests.support.pin_tiers import assert_declined_at_default, assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate

DECLINED_ROWS = [
    #:3725
    "CC(=O)N1CCCc2ccccc21",
    "CCC(=O)N1CCCc2ccccc21",
    "OC(=O)N1CCCc2ccccc21",
    "CC(=O)N1CCc2ccccc2C1",
    "CC(=O)N1CCNc2ccccc21",
    #:21730
    "CNC(=O)CCc1cc(CC(C)C(N)=O)ccc1C",
    "CNC(=O)CCc1ccccc1CC(C)C(N)=O",
    "NC(=O)CCc1ccccc1CCCC(=O)NC",              #: the N-methylamide chain is the longer one
    "NC(=O)CCc1ccc(CCC(=O)NC)cc1",             #:21604: equal chains, the N-methylamide has more prefixes
    "NC(=O)CCc1ccc(/C=C/C(=O)NC)cc1",          #:21033: the N-methylamide chain has the multiple bond
    #:32368,:39499
    "CCC(=O)OOSC(C)=O",
    #:35730,:39747
    "CN(C)P(=O)(Cl)N=C=O",
    "C=[CH][Sb]([Br])[CH]=C",
    #:46008
    "CO[As@](C)(=O)c1ccccc1",
    #:26632,:26636
    "CN(C)CCC(CC[N+](C)(C)[O-])N(C)C",
    "CN(C)CCCCC[N+](C)(C)[O-]",
    "NC(=O)c1cc(CC[NH2+][O-])cc(C(=O)O)c1",
    #:27963, (e):7134
    "SSc1ccc(SS)cc1",
    "SSc1ccccc1",
    "SSCCSS",
]

CONTROL_ROWS = [
    ("CN1CCCc2ccccc21", "1-methyl-1,2,3,4-tetrahydroquinoline"),            # the nitrogen is a prefix site: no added hydrogen
    ("NC1CCCc2ccccc21", "1,2,3,4-tetrahydronaphthalen-1-amine"),            #:26489, a carbon site
    ("O=C(/C=C/c1ccccc1)c1ccccc1", "chalcone"),                              #:28299
    ("O=C(/C=C\\c1ccccc1)c1ccccc1", "(2Z)-1,3-diphenylprop-2-en-1-one"),     # chalcone is the (E) isomer only
    ("COc1cccc(/C=C/C(=O)c2ccc(O)c(OC)c2O)c1", "2',4'-dihydroxy-3,3'-dimethoxychalcone"),   #:28303 (PIN)
    ("NC(=O)CCCc1ccccc1CCC(=O)N(C)C", "4-{2-[3-(dimethylamino)-3-oxopropyl]phenyl}butanamide"),   # the longer chain
    ("CC(=O)OOSC(C)=O", "1-[(acetylperoxy)sulfanyl]ethan-1-one"),            #:32368
    ("CN(C)P(=O)(Cl)Cl", "N,N-dimethylphosphoramidic dichloride"),
    ("C[Sb](C)N=C=O", "dimethylstibinous isocyanate"),                      #:35714
    ("ClP(c1ccccc1)c1ccccc1", "diphenylphosphinous chloride"),               #:35712
    ("C[N+](C)(C)[O-]", "N,N-dimethylmethanamine N-oxide"),                  #:26626
    ("CCSS", "ethanedithioperoxol"),                                         #:27998
    ("SSCC(=O)O", "disulfanylacetic acid"),                                  # the acid is senior: the prefix stands
    ("OCCSS", "2-disulfanylethan-1-ol"),
]


@pytest.mark.parametrize("smiles", DECLINED_ROWS)
def test_declined_at_the_pin_tier_and_kept_below(smiles):
    assert_declined_at_default(smiles)


@pytest.mark.parametrize("smiles,pin", CONTROL_ROWS)
def test_the_class_controls_stay_pins(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
