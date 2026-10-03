"""PIN class program, Task 5: an acid parent with an ester, acyloxy, nitrooxy or thiocyanato prefix.

 (the Blue Book): "If a sulfur-containing group is attached by oxygen
(chalcogen) or nitrogen to a compound that contains also another substituent having priority
over the sulfur-containing group for citation as principal group, then the sulfur-containing
group is named by an appropriate prefix formed by concatenation or substitution (see
as described in and." Rows: '3-(sulfooxy)propanoic acid (PIN)'
(:36488), '3-[(methoxysulfinyl)oxy]propanoic acid (PIN)' (:36490), '3-[(chlorosulfonyl)oxy]
propanoic acid (PIN)' (:36492), '3-(sulfamoyloxy)propanoic acid (PIN)' (:36494),
'3-[(aminosulfinyl)oxy]propanoic acid (PIN) [not 3-(sulfinamidoyloxy)propanoic acid...]'
(:36500), '3-[(methoxysulfonyl)amino]propanoic acid (PIN)' (:36502).

 (:31276): the acyl groups 'sulfonyl' and 'sulfinyl' carry prefixes for the
characteristic groups attached to them, "This traditional method generates preferred IUPAC
names" (:31316): 'methoxysulfonyl (preferred prefix)' (:31322), 'chlorosulfinyl (preselected
prefix)' (:31326), 'sulfamoyl (preselected prefix) aminosulfonyl' (:31330). An acid named by
a suffix gives its own acyl prefix: '-SO-OH sulfino (preselected prefix)',:17969,
:17971), as '-SO2-OH' gives 'sulfo' ('HO-SO2-O- sulfooxy (preselected prefix)',:31342).

 (:36937): "In the presence of a characteristic group having precedence for citation
as principal group, polynuclear noncarbon oxoacids are cited as prefixes", method (2) "on the
basis of the names of the group which includes the greatest number of P, As, Sb, S, Se, and Te
central atoms": '(2) 3-[(1,3,3-trihydroxy-1,3-dioxo-1λ5,3λ5-diphosphoxan-1-yl)oxy]propanoic
acid (PIN)' (:36949).

 (:31696): "When, in an ester with the general structure R-CO-O-R' or
R-S(O)x-O-R', another group is present that has priority for citation as the principal group
..., an ester group is indicated by prefixes as 'acyloxy' for the group R-CO-O-":
'3-[(pyridine-3-carbonyl)oxy]propanoic acid (PIN) 3-(nicotinoyloxy)propanoic acid' (:31723).

 (:25957) with '2-(tert-butylimino)-3-methyl-3-(nitrooxy)butanoic acid (PIN)'
(:25963); (:26540): "The prefix 'imino' for =NH is used in presence of
characteristic groups having seniority over imines"; an N-substituted imino group is cited as
'(R-imino)': '3-(methylimino)butan-2-ylidene (PIN)' (:43352), 'methyl [(methylimino)silyl]
acetate (PIN)' (:26578).

 (:30977): "Preferred prefixes derived from cyanic acid are... 'thiocyanato' for
-S-CN... Parentheses are used to enclose chalcogen prefixes to avoid the possibility of
ambiguity." '3-(thiocyanato)propanoic acid (PIN)' (:31013).

At the base the default tier declined all nine rows and best-effort shipped the general
engine's names (the PIN strings) at systematic_verified. The PIN path split each prefix: the
branch walkers dropped the carbon-free -O-S(oxoacid) and -O-P-O-P branches and gave the
-NH-SO2-OCH3 branch the partial name 'amino'; the acyloxy walker had no ring acyl; the
thiocyanato branch was named twice ('3-(cyanosulfanyl)-3-(thiocyanato)'); an N-substituted
imine on the chain stopped the handler.

The prefixes are cited only when the principal characteristic group is senior to esters
: the acids and the anhydrides). Under a junior class the ester is named by functional
class nomenclature,:35916); those molecules stay declined at the default tier.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import name_acid_linked_branch
from orthonym.perception.functional_groups import detect_functional_groups
from tests.support.pin_tiers import assert_declined_at_default, assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate

PIN_ROWS = [
    ("N#CSCCC(=O)O", "3-(thiocyanato)propanoic acid"),                                   #:31013
    ("COS(=O)OCCC(=O)O", "3-[(methoxysulfinyl)oxy]propanoic acid"),                      #:36490
    ("CC(C)(C)N=C(C(=O)O)C(C)(C)O[N+](=O)[O-]",
     "2-(tert-butylimino)-3-methyl-3-(nitrooxy)butanoic acid"),                          #:25963
    ("O=C(O)CCOS(=O)(=O)Cl", "3-[(chlorosulfonyl)oxy]propanoic acid"),                   #:36492
    ("NS(=O)(=O)OCCC(=O)O", "3-(sulfamoyloxy)propanoic acid"),                           #:36494
    ("NS(=O)OCCC(=O)O", "3-[(aminosulfinyl)oxy]propanoic acid"),                         #:36500
    ("COS(=O)(=O)NCCC(=O)O", "3-[(methoxysulfonyl)amino]propanoic acid"),                #:36502
    ("O=C(O)CCOP(=O)(O)OP(=O)(O)O",
     "3-[(1,3,3-trihydroxy-1,3-dioxo-1λ5,3λ5-diphosphoxan-1-yl)oxy]propanoic acid"),     #:36949
    ("O=C(O)CCOC(=O)c1cccnc1", "3-[(pyridine-3-carbonyl)oxy]propanoic acid"),            #:31723
]

# Members of the class beyond the Blue Book rows, built by the same rules (each expected
# name is read back by OPSIN to the input's full InChIKey inside the helper).
CLASS_ROWS = [
    ("O=C(O)CCOS(=O)(=O)OC", "3-[(methoxysulfonyl)oxy]propanoic acid"),                  #:31322
    ("O=C(O)CCOS(=O)(=O)F", "3-[(fluorosulfonyl)oxy]propanoic acid"),
    ("O=C(O)CCOS(=O)Cl", "3-[(chlorosulfinyl)oxy]propanoic acid"),                       #:31326
    ("O=C(O)CCOS(=O)OCC", "3-[(ethoxysulfinyl)oxy]propanoic acid"),
    ("O=C(O)CCOS(=O)O", "3-(sulfinooxy)propanoic acid"),                                 #:17971
    ("O=C(O)CCNS(=O)(=O)O", "3-(sulfoamino)propanoic acid"),
    ("O=C(O)CCNS(N)(=O)=O", "3-(sulfamoylamino)propanoic acid"),                         #:31330
    ("O=C(O)CCNS(=O)(=O)Cl", "3-[(chlorosulfonyl)amino]propanoic acid"),
    ("O=C(O)CCOP(=O)(O)OP(=O)(O)OP(=O)(O)O",
     "3-[(1,3,5,5-tetrahydroxy-1,3,5-trioxo-1λ5,3λ5,5λ5-triphosphoxan-1-yl)oxy]propanoic acid"),
    ("O=C(O)CCOP(=O)(OC)OP(=O)(O)O",
     "3-[(3,3-dihydroxy-1-methoxy-1,3-dioxo-1λ5,3λ5-diphosphoxan-1-yl)oxy]propanoic acid"),
    ("O=C(O)CCOC(=O)c1ccccn1", "3-[(pyridine-2-carbonyl)oxy]propanoic acid"),
    ("O=C(O)CC(C)OC(=O)c1cccnc1", "3-[(pyridine-3-carbonyl)oxy]butanoic acid"),
    ("N#CSCCCC(=O)O", "4-(thiocyanato)butanoic acid"),
    ("CN=C(C)C(=O)O", "2-(methylimino)propanoic acid"),                                  #:43352
    ("CC(C)(C)N=CCC(=O)O", "3-(tert-butylimino)propanoic acid"),
    ("N#CSCC(=O)O", "(thiocyanato)acetic acid"),                     # cf. 'cyanoacetic acid (PIN)':30999
    ("O=C(O)CC(OS(=O)(=O)Cl)COS(=O)(=O)Cl", "3,4-bis[(chlorosulfonyl)oxy]butanoic acid"),
    ("O=C(O)CC(OS(N)(=O)=O)CO", "4-hydroxy-3-(sulfamoyloxy)butanoic acid"),
    ("O=C(O)C[C@H](OS(N)(=O)=O)C", "(3R)-3-(sulfamoyloxy)butanoic acid"),
    ("O=C(O)CC(OS(=O)(=O)Cl)CC(=O)O", "3-[(chlorosulfonyl)oxy]pentanedioic acid"),
    ("OS(=O)(=O)CCOS(=O)(=O)Cl", "2-[(chlorosulfonyl)oxy]ethane-1-sulfonic acid"),  # cf.:31713
    ("OC(=O)CCOC(=O)c1ccco1", "3-[(furan-2-carbonyl)oxy]propanoic acid"),
    # the ester of a noncarbon oxoacid on a carbon of a branch is cited in the branch prefix
    ("O=C(O)C(COP(=O)(O)O)CC", "2-[(phosphonooxy)methyl]butanoic acid"),
    ("O=C(O)C(CO[N+](=O)[O-])CC", "2-[(nitrooxy)methyl]butanoic acid"),
    ("O=C(O)[C@@](O)(COP(=O)(O)O)[C@H](O)[C@H](O)CO",
     "(2R,3R,4R)-2,3,4,5-tetrahydroxy-2-[(phosphonooxy)methyl]pentanoic acid"),     # dev2000
    # a ring parent takes the same prefixes
    ("O=C(O)C1CCC(OS(=O)(=O)Cl)CC1", "4-[(chlorosulfonyl)oxy]cyclohexane-1-carboxylic acid"),
    ("O=C(O)C1CCC(OS(=O)OC)CC1", "4-[(methoxysulfinyl)oxy]cyclohexane-1-carboxylic acid"),
    ("O=C(O)C1CCC(OP(=O)(O)OP(=O)(O)O)CC1",
     "4-[(1,3,3-trihydroxy-1,3-dioxo-1λ5,3λ5-diphosphoxan-1-yl)oxy]cyclohexane-1-carboxylic acid"),
]

# Already the PIN at both tiers at the base (protection).
CONTROL_ROWS = [
    ("O=C(O)CCOS(=O)(=O)O", "3-(sulfooxy)propanoic acid"),                               #:36488
    ("O=C(O)CCOP(=O)(O)O", "3-(phosphonooxy)propanoic acid"),
    ("O=C(O)CCO[N+](=O)[O-]", "3-(nitrooxy)propanoic acid"),
    ("CC(=O)OCCC(=O)O", "3-(acetyloxy)propanoic acid"),
    ("O=C(O)CCOC(=O)c1ccccc1", "3-(benzoyloxy)propanoic acid"),                          #:31711
    ("CS(=O)(=O)NCCC(=O)O", "3-(methanesulfonamido)propanoic acid"),
]

# A principal class junior to esters: the ester is named by functional class
# nomenclature,:35916), so the prefix name is not the PIN.
DECLINED_ROWS = [
    "OCCOS(=O)(=O)Cl",
    "NC(=O)CCOS(=O)(=O)Cl",
    "OCCOS(=O)O",
    "OCCOP(=O)(O)OP(=O)(O)O",
    "OC1CCC(OS(=O)(=O)Cl)CC1",
    "OCC(COP(=O)(O)O)CC",
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


@pytest.mark.parametrize("smiles", DECLINED_ROWS)
def test_declined_at_the_pin_tier(smiles):
    assert_declined_at_default(smiles)


def test_the_branch_builder_keeps_its_two_conditions():
    """The acyl-oxy builder names the -O-SO2-OH branch of 3-(sulfooxy)propanoic acid
    (atoms 5-9, linking O 5) only under a principal group senior to esters, and leaves
    a linking O that a perceived functional group holds ('sulfate_monoester') to that
    group's FG prefix."""
    mol = Chem.MolFromSmiles("O=C(O)CCOS(=O)(=O)O")
    branch = {5, 6, 7, 8, 9}
    assert name_acid_linked_branch(mol, branch, 5, "carboxylic_acid") == "sulfooxy"
    assert name_acid_linked_branch(mol, branch, 5, "primary_alcohol") is None
    assert name_acid_linked_branch(
        mol, branch, 5, "carboxylic_acid",
        functional_groups=detect_functional_groups(mol)) is None
