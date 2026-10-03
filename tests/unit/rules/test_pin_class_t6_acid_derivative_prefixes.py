"""PIN class program, Task 6: an acid parent with an acid-derivative prefix.

 (the Blue Book) "When another group is present that has seniority for
citation as principal group, the following prefixes are used: (1) the compound prefix
'C-hydroxycarbonimidoyl' used to denote the acyl group -C(=NH)-OH... (2) a combination of the
simple prefixes 'hydroxy' and 'imino' at the end of a carbon chain is used in preferred IUPAC
names rather than the compound prefix 'C-hydroxycarbonimidoyl'" (:30020,:30021). Rows:
'(2) 4-hydroxy-4-iminobutanoic acid (PIN)' (:30035) and '(1) 3-(C-hydroxycarbonimidoyl)
propanoic acid' (:30037, the non-PIN form of a chain end).

 (:34014) "When the -CO-NHNH2 group is at the end of a chain the use of the
prefixes 'hydrazinyl' and 'oxo' generates preferred IUPAC names." Row: '3-hydrazinyl-3-
oxopropanoic acid (PIN) (hydrazinecarbonyl)acetic acid carbonohydrazidoylacetic acid'
(:34018).

 (:30208) "When the position of chalcogen atoms is known, combinations of prefixes
such as 'hydroxy- and sulfanylidene-' or 'sulfanyl- and oxo-' are used in acyclic compounds"
(:30219). Rows: '4-oxo-4-sulfanylbutanoic acid (PIN) 3-(sulfanylcarbonyl)propanoic acid'
(:30250), 'HO-CS-CH2-CH2-COOH 4-hydroxy-4-sulfanylidenebutanoic acid (PIN)' (:30252),
'HS-CO-COOH oxo(sulfanyl)acetic acid (PIN)' (:30254).

 (:18489) orders the chalcogen acids by "(a) maximum number of oxygen atoms, then S, Se,
and Te atoms..." (:18491), with '-CS-OH carbothioic O-acid' above '-CO-SH carbothioic S-acid'
in Table 4.3 (:18520,:18521). Where two different chalcogen acids meet, the principal group
the engine picks is not checked against that order here, so those molecules stay declined at
the default tier (DECLINED_ROWS).

 (:29446) "Three types of prefixes are used in the formation of preferred IUPAC
names: (1) the prefix 'oxo' when the doubly bonded oxygen atom (ketone, pseudoketone or
heterone group) is not in position 1 of a side chain" (:29450); (:29496)
"Acyclic pseudoketones are named in the same way": '4-oxo-4-silylbutanoic acid (PIN)
3-(silanecarbonyl)propanoic acid' (:29500). (b) (:28263): pseudoketones are
"compounds in which an acyclic carbonyl group is bonded to one or two acyclic skeletal
heteroatoms, except nitrogen, halogen, or pseudohalogen atoms": '1-silylethan-1-one (PIN)',
'1-phosphanylpropan-1-one (PIN)', '1-(methoxydisulfanyl)ethan-1-one (PIN)' (:28273-:28277).
An -O- or an -S-O- partner makes an ester ('SO-methyl benzene(carbothioperoxoate) (PIN)',
:32001) and is not built here.

 (:2877) "Terminal locants are not cited in names for mono- and dicarboxylic acids
derived from acyclic hydrocarbons and their corresponding acyl halides, amides, hydrazides
... when unsubstituted or substituted on carbon atoms": 'H2N-CO-CH(CH3)-CO-NH2
2-methylpropanediamide (PIN)' (:2887), 'N1,N3-dimethylpropanediamide (PIN)' (:2889).

 (:26540) "The prefix 'imino' for =NH is used in presence of characteristic groups
having seniority over imines": '3-(2-iminopropyl)cyclohexane-1-carboxylic acid (PIN)'
(:26549). (:38607) "In the presence of functions having seniority the prefix
'hydrazinylidene' is used substitutively": '4-[(propan-2-ylidene)hydrazinylidene]cyclohexane-
1-carboxylic acid (PIN)' (:38613), '4-(phenylhydrazinylidene)cyclohexane-1-carboxylic acid
(PIN)' (:38579). (:38652): "The compound prefix 'carbamoylhydrazinylidene' is used
in the presence of a characteristic group that is preferred for citation as a suffix":
'4-[(dimethylcarbamoyl)hydrazinylidene]heptanoic acid (PIN)' (:38678); "The prefix
'carbamothioylhydrazinylidene' is preferred to the traditional prefix 'thiosemicarbazono'"
(:38680). (:33198): 'carbamothioyl (not thiocarbamoyl) for -CS-NH2'
('carbamothioyl (retained name; preferred prefix)',:30869). (:7232): compound
prefixes are enclosed: '2-[(methylcarbamoyl)amino]naphthalene-1-carboxylic acid (PIN)' (:33354),
'methyl [(methylimino)silyl]acetate (PIN)' (:26578).

At the base the default tier declined every PIN row and best-effort shipped the general
engine's names (the PIN strings) at systematic_verified. The polyfunctional handler gave the
imidic acid no prefix ('butanoic acid'), and cited the hydrazide and the thioic acids with the
acyl prefixes 'hydrazinecarbonyl', 'sulfanylcarbonyl', 'carbothioyl' on a chain that already
held their carbon (a different molecule; the round trip refused it). The =O of the
pseudoketone belongs to no perceived functional group, so the chain walk dropped it
('4-silylbutanoic acid', refused by the atom-coverage check). The chain-diamide handler
cited only N-substituents on a saturated parent, so '2-methylpropanediamide' came out as
'propanediamide' and 'but-2-enediamide' as 'butanediamide' (refused). The ring parent cited an
imine of a ring branch, and the bare =N of a ring or chain =N-N(R) / =N-R, as a ring-locant
'imino' / 'hydrazinylidene' prefix next to the branch's own name; a ring -C(=NH)-OH had no
prefix; the substituent walk named =N-N=C(CH3)2 '(propan-2-iminyl)amino' and -CH2-C(=NH)-CH3
'propan-2-iminyl' (different molecules).
"""
import pytest
from rdkit import Chem

from orthonym.assembly.substituent_naming import ATTACH_LOCANT_UNKNOWN, parent_to_prefix
from orthonym.perception.functional_groups import detect_functional_groups
from orthonym.rules.polyfunctional import _chain_end_acid_derivative_prefixes
from tests.support.pin_tiers import assert_declined_at_default, assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate

PIN_ROWS = [
    ("N=C(O)CCC(=O)O", "4-hydroxy-4-iminobutanoic acid"),                                #:30035
    ("NNC(=O)CC(=O)O", "3-hydrazinyl-3-oxopropanoic acid"),                              #:34018
    ("O=C(O)CCC(=O)S", "4-oxo-4-sulfanylbutanoic acid"),                                 #:30250
    ("O=C(O)CCC(O)=S", "4-hydroxy-4-sulfanylidenebutanoic acid"),                        #:30252
    ("O=C(O)CCC(=O)[SiH3]", "4-oxo-4-silylbutanoic acid"),                               #:29500
    ("CC(C(N)=O)C(N)=O", "2-methylpropanediamide"),                                      #:2887
    ("CCCC(CCC(=O)O)=NNC(=O)N(C)C", "4-[(dimethylcarbamoyl)hydrazinylidene]heptanoic acid"),  #:38678
    ("CC(=N)CC1CCCC(C(=O)O)C1", "3-(2-iminopropyl)cyclohexane-1-carboxylic acid"),         #:26549
    ("N=C(O)C1CCCC1C(=O)O", "2-(C-hydroxycarbonimidoyl)cyclopentane-1-carboxylic acid"),   #:30029
    ("CC(C)=NN=C1CCC(C(=O)O)CC1",
     "4-[(propan-2-ylidene)hydrazinylidene]cyclohexane-1-carboxylic acid"),               #:38613
]

# Members of the class beyond the Blue Book rows, built by the same rules (each expected
# name is read back by OPSIN to the input's full InChIKey inside the helper).
CLASS_ROWS = [
    ("N=C(O)CC(=O)O", "3-hydroxy-3-iminopropanoic acid"),
    ("N=C(O)C(=O)O", "hydroxy(imino)acetic acid"),                    # cf.:30254
    ("N=C(O)CCC(=O)S", "4-hydroxy-4-iminobutanethioic S-acid"),
    ("NNC(=O)CCCC(=O)O", "5-hydrazinyl-5-oxopentanoic acid"),
    ("NNC(=O)C(=O)O", "hydrazinyl(oxo)acetic acid"),
    ("NNC(=O)CC(N)C(=O)O", "2-amino-4-hydrazinyl-4-oxobutanoic acid"),
    ("OC(=O)CC(=S)S", "3-sulfanyl-3-sulfanylidenepropanoic acid"),
    ("OC(=O)CCCCC(=S)S", "6-sulfanyl-6-sulfanylidenehexanoic acid"),
    ("OC(=O)CCC(=O)[SeH]", "4-oxo-4-selanylbutanoic acid"),
    ("OC(=O)CCC(=[Se])O", "4-hydroxy-4-selanylidenebutanoic acid"),
    ("OC(=O)CC(=[Se])[SeH]", "3-selanyl-3-selanylidenepropanoic acid"),
    ("OC(=O)CCC(=[Te])[TeH]", "4-tellanyl-4-tellanylidenebutanoic acid"),
    ("OC(=O)CC(O)CC(=O)S", "3-hydroxy-5-oxo-5-sulfanylpentanoic acid"),
    ("OC(=O)C(C)CC(=S)O", "4-hydroxy-2-methyl-4-sulfanylidenebutanoic acid"),
    ("OC(=O)C=CC(=O)S", "4-oxo-4-sulfanylbut-2-enoic acid"),
    # acyclic pseudoketones (b)) under a senior group
    ("O=C(O)C(=O)[SiH3]", "oxo(silyl)acetic acid"),
    ("O=C(O)CCC(=O)[GeH3]", "4-germyl-4-oxobutanoic acid"),
    ("O=C(O)CCC(=O)[Si](C)(C)C", "4-oxo-4-(trimethylsilyl)butanoic acid"),
    ("O=C(O)CCC(=O)[Sn](C)(C)C", "4-oxo-4-(trimethylstannyl)butanoic acid"),
    ("O=C(O)CCC(=O)P", "4-oxo-4-phosphanylbutanoic acid"),
    ("O=C(O)CCC(=O)PC", "4-(methylphosphanyl)-4-oxobutanoic acid"),
    ("O=C(O)CCC(=O)P(=O)(O)O", "4-oxo-4-phosphonobutanoic acid"),
    ("O=C(O)CCC(=O)[As](C)C", "4-(dimethylarsanyl)-4-oxobutanoic acid"),
    ("O=C(O)CCC(=O)B(O)O", "4-borono-4-oxobutanoic acid"),                    # 'borono':36349
    ("O=C(O)CCC(=O)SSOC", "4-(methoxydisulfanyl)-4-oxobutanoic acid"),        # cf.:28277
    ("O=C(O)CCC(=O)[Se][Se]C", "4-(methyldiselanyl)-4-oxobutanoic acid"),
    ("N#CCCC(=O)[SiH3]", "4-oxo-4-silylbutanenitrile"),
    ("O=C(O)CC(O)CC(=O)[SiH3]", "3-hydroxy-5-oxo-5-silylpentanoic acid"),
    ("O=C(O)CC(=O)CC(=O)[SiH3]", "3,5-dioxo-5-silylpentanoic acid"),
    ("O=C(O)CC(N)CC(=O)[GeH3]", "3-amino-5-germyl-5-oxopentanoic acid"),
    ("OC(=O)CC(O)CC(=O)SSOC", "3-hydroxy-5-(methoxydisulfanyl)-5-oxopentanoic acid"),
    ("OC(=O)CC(O)CC[SiH3]", "3-hydroxy-5-silylpentanoic acid"),
    # chain diamides substituted on carbon or unsaturated
    ("NC(=O)C(O)C(N)=O", "2-hydroxypropanediamide"),
    ("NC(=O)C(Cl)CC(N)=O", "2-chlorobutanediamide"),
    ("NC(=O)C(C)(C)C(N)=O", "2,2-dimethylpropanediamide"),
    ("NC(=O)CC(C)CC(N)=O", "3-methylpentanediamide"),
    ("NC(=O)C(CC)C(N)=O", "2-ethylpropanediamide"),
    ("NC(=O)C(N)CC(N)=O", "2-aminobutanediamide"),
    ("NC(=O)C(Br)C(Br)C(N)=O", "2,3-dibromobutanediamide"),
    ("NC(=O)[C@@H](O)[C@H](O)C(N)=O", "(2S,3S)-2,3-dihydroxybutanediamide"),
    ("NC(=O)C=CC(N)=O", "but-2-enediamide"),
    ("NC(=O)/C=C/C(N)=O", "(2E)-but-2-enediamide"),
    ("NC(=O)C#CC(N)=O", "but-2-ynediamide"),
    # a -C(=NH)-OH on a ring (1))
    ("N=C(O)C1CCC(C(=O)O)CC1", "4-(C-hydroxycarbonimidoyl)cyclohexane-1-carboxylic acid"),
    # a group deeper in a ring branch is named with its branch
    ("NCCC1CCC(C(=O)O)CC1", "4-(2-aminoethyl)cyclohexane-1-carboxylic acid"),
    ("CC(=N)CCc1ccc(C(=O)O)cc1", "4-(3-iminobutyl)benzoic acid"),
    # a ring or chain =N- whose nitrogen side carries a substituent,
    #; '(R-imino)' as in '2-(tert-butylimino)...':25963)
    ("c1ccccc1NN=C1CCC(C(=O)O)CC1", "4-(phenylhydrazinylidene)cyclohexane-1-carboxylic acid"),  #:38579
    ("O=C(O)C1CCC(=NNc2ccccc2)C1", "3-(phenylhydrazinylidene)cyclopentane-1-carboxylic acid"),
    ("CNN=C1CCC(C(=O)O)CC1", "4-(methylhydrazinylidene)cyclohexane-1-carboxylic acid"),
    ("CN(C)N=C1CCC(C(=O)O)CC1", "4-(dimethylhydrazinylidene)cyclohexane-1-carboxylic acid"),
    ("CN(C)C(=O)NN=C1CCC(C(=O)O)CC1",
     "4-[(dimethylcarbamoyl)hydrazinylidene]cyclohexane-1-carboxylic acid"),
    ("NC(=O)NN=C1CCC(C(=O)O)CC1", "4-(carbamoylhydrazinylidene)cyclohexane-1-carboxylic acid"),
    ("NC(=S)NN=C1CCC(C(=O)O)CC1",
     "4-(carbamothioylhydrazinylidene)cyclohexane-1-carboxylic acid"),                    #:38680
    ("CC(C)=NN=C1CCCC1=O", "2-[(propan-2-ylidene)hydrazinylidene]cyclopentan-1-one"),
    ("CN=C1CCC(C(=O)O)CC1", "4-(methylimino)cyclohexane-1-carboxylic acid"),
    ("CON=C1CCC(C(=O)O)CC1", "4-(methoxyimino)cyclohexane-1-carboxylic acid"),
    ("CN=C1CCCCC1=O", "2-(methylimino)cyclohexan-1-one"),
    ("CC(=NNC)CC(=O)O", "3-(methylhydrazinylidene)butanoic acid"),
    ("CC(=NN(C)C)CC(=O)O", "3-(dimethylhydrazinylidene)butanoic acid"),
    ("CC(=NNC(N)=O)CC(=O)O", "3-(carbamoylhydrazinylidene)butanoic acid"),
    ("CC(=NNc1ccccc1)CC(=O)O", "3-(phenylhydrazinylidene)butanoic acid"),
    ("CC(=NN=C(C)C)CC(=O)O", "3-[(propan-2-ylidene)hydrazinylidene]butanoic acid"),
    ("CC(=NNC(N)=S)CC(=O)O", "3-(carbamothioylhydrazinylidene)butanoic acid"),         #:38680
    ("CC(=NNC(=S)NC)CC(=O)O", "3-[(methylcarbamothioyl)hydrazinylidene]butanoic acid"),
    ("CC(=NNC(=O)NC)CC(=O)O", "3-[(methylcarbamoyl)hydrazinylidene]butanoic acid"),
    # an acyl on the far nitrogen, as 'carbamoyl' in 'carbamoylhydrazinylidene' (:38669)
    ("CC(=NNC(=O)C)CC(=O)O", "3-(acetylhydrazinylidene)butanoic acid"),
]

# Already the PIN at both tiers at the base (protection).
CONTROL_ROWS = [
    ("NC(=O)CCC(=O)O", "4-amino-4-oxobutanoic acid"),
    ("O=CCCC(=O)O", "4-oxobutanoic acid"),
    ("OC(=O)C(=O)S", "oxo(sulfanyl)acetic acid"),                                        #:30254
    ("SC(=O)CC(=O)S", "propanebis(thioic acid)"),
    ("NC(=O)CCC(N)=O", "butanediamide"),
    ("CNC(=O)CC(N)=O", "N1-methylpropanediamide"),
    ("CNC(=O)CC(=O)NC", "N1,N3-dimethylpropanediamide"),                                 #:2889
    ("CCNC(=O)CCC(N)=O", "N1-ethylbutanediamide"),
    ("NC(=S)c1ccc(C(=O)O)cc1", "4-carbamothioylbenzoic acid"),
    ("NC(=S)CCC(=O)O", "4-amino-4-sulfanylidenebutanoic acid"),
    ("ON=C1CCC(C(=O)O)CC1", "4-(hydroxyimino)cyclohexane-1-carboxylic acid"),
    ("NN=C1CCC(C(=O)O)CC1", "4-hydrazinylidenecyclohexane-1-carboxylic acid"),
    ("N=C1CCC(C(=O)O)CC1", "4-iminocyclohexane-1-carboxylic acid"),
    ("OCCC1CCC(C(=O)O)CC1", "4-(2-hydroxyethyl)cyclohexane-1-carboxylic acid"),
    ("CC(=NN)CC(=O)O", "3-hydrazinylidenebutanoic acid"),
]

# (:7232) "Parentheses are used around compound... prefixes": a phosphanyl
# carrying its own prefix is enclosed, '3-(hydroxyphosphanyl)propanoic acid (PIN)' (:27272).
# At the base the bare 'methylphosphanyl' shipped at pin_verified.
ENCLOSURE_ROWS = [
    ("OC(=O)CCCPC", "4-(methylphosphanyl)butanoic acid"),
    ("OC(=O)CC(O)CPC", "3-hydroxy-4-(methylphosphanyl)butanoic acid"),
    ("OCCCPC", "3-(methylphosphanyl)propan-1-ol"),
    ("OC(=O)c1ccc(PC)cc1", "4-(methylphosphanyl)benzoic acid"),
    # unchanged: the bare parent-hydride prefix and names already enclosed
    ("NCCP", "2-phosphanylethan-1-amine"),
    ("CCP(C)c1c[nH]cn1", "4-[ethyl(methyl)phosphanyl]-1H-imidazole"),
    ("OC(=O)CC(PC)CPC", "3,4-bis(methylphosphanyl)butanoic acid"),
    ("CCC(=O)PC", "1-(methylphosphanyl)propan-1-one"),
    # an N-substituted carbamoyl is a compound prefix too ('[(methylcarbamoyl)amino]',:33354)
    ("CNC(=O)C1CCC(C(=O)O)CC1", "4-(methylcarbamoyl)cyclohexane-1-carboxylic acid"),
]

# Two different chalcogen acids, Table 4.3): declined at the default tier; the
# best-effort tier keeps an RT-exact name.
DECLINED_ROWS = [
    "OC(=S)CCC(=O)S",
    # a pseudoketone carbonyl is the suffix ('-one') when no group outranks ketones
    #,:29628 "There is no seniority order difference between ketones and
    # pseudoketones"); these names are not built at the default tier
    "CC(=O)CCC(=O)[SiH3]",
    "OCCC(=O)[SiH3]",
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


@pytest.mark.parametrize("smiles,pin", ENCLOSURE_ROWS)
def test_phosphanyl_enclosure(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles", DECLINED_ROWS)
def test_declined_at_the_pin_tier(smiles):
    assert_declined_at_default(smiles)


def test_the_chain_end_hydrazide_split_is_for_the_unsubstituted_group_only():
    """ names -CO-NH-NH2; a hydrazide with a substituent on either nitrogen is
    left to other paths (the split would drop the substituent)."""
    for smiles, expected in (
            ("NNC(=O)CC(=O)O", ({"hydrazinyl": [3], "oxo": [3]}, {0, 1, 3})),
            ("CNNC(=O)CC(=O)O", None),
            ("NN(C)C(=O)CC(=O)O", None)):
        mol = Chem.MolFromSmiles(smiles)
        matches = detect_functional_groups(mol)["hydrazide"]
        carbon = matches[0][0]
        got = _chain_end_acid_derivative_prefixes(
            mol, "hydrazide", matches, {carbon}, {carbon: 3}, "carboxylic_acid")
        assert got == expected, (smiles, got)


def test_a_thioamide_name_never_takes_the_amide_prefix_transform():
    """The capped-fragment converter read '...ethanethioamide' as an amide and gave
    '2-amino-2-oxoethyl' (the sulfur lost); a thioamide name has no such '-yl' form here."""
    for name in ("ethanethioamide", "N-methylethanethioamide"):
        assert parent_to_prefix(name, 2, attach_locant=ATTACH_LOCANT_UNKNOWN) is None, name
