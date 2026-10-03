"""PIN class program, Task 9: spiro names.

 (the Blue Book, heading ' Linear polyspiro alicyclic ring systems'):
"The compound is numbered in the order in which the numbers of the von Baeyer spiro descriptor
are cited, including spiro atoms when encountered for the first time. Each time a spiro atom is
reached for the second time its locant, which has already been assigned, is cited as a
superscript number to the number of the preceding linking atoms." (:9977)
'dispiro[3.2.3^7.2^4]dodecane (PIN)' (:9981), 'dispiro[2.2.3^6.2^3]undecane (PIN)' (:9985).
:9989: "it is recommended that they be used to name all polyspiro compounds, especially when
IUPAC preferred names are required."
 (:10079): '6,7,13,14-tetraoxadispiro[4.2.4^8.2^5]tetradecane (PIN)' (:10087).
 (:19584): '5,11-diazadispiro[3.2.3^7.2^4]dodecane (PIN)' (:19597).
 (:49156): '(5R,7S)-1,8-dioxadispiro[4.1.4^7.2^5]tridecane (PIN)' (:49166).
The engine writes a superscript as '^n' (the branched polyspiro walk already does:
'trispiro[2.2.2^6.2.2^11.2^3]pentadecane').

 (:7469): "In determining the nesting order in a name the square brackets of ring
fusion, spiro fusion, ring assembly, or the extended von Baeyer names are ignored."
'tris(tetracyclo[3.2.0.0^2,7.0^4,6]heptane)' (:10824): a descriptor with superscript locants is
one of those square brackets, so a substituent prefix holding one is enclosed in parentheses.

 (the Blue Book): "Monospiro ring systems consisting of two identical polycyclic ring
components are named by placing the nondetachable prefix 'spirobi' before the name of the
component ring system enclosed in square brackets." (:10152): "the maximum number of
noncumulative double bonds is added (i.e., the system is made mancude) after construction of the
complete skeleton." Note (:10162): "brackets are used to enclose locants belonging to component
names (see ". (:10168): "If there is a choice for assigning primed locants,
the lower number at the spiro atom is unprimed." '3H,3'H-2,2'-spirobi[[1]benzothiophene] (PIN)'
(:10160), '2'H,3H-2,3'-spirobi[[1]benzothiophene] (PIN)' (:10176), '2,4'-spirobi[[1]benzopyran]
(PIN)' (:10182).
 (:10184, sentence:10186): "When ring components of 'spirobi' compounds are named by von
Baeyer nomenclature, heteroatoms are indicated by skeletal replacement ('a') nomenclature. The
spirobi ring system is named as the saturated bi- or polycyclic alicyclic hydrocarbon and the
heteroatoms are denoted by 'a' prefixes cited at the front of the completed 'spirobi' hydrocarbon
name. If there is a choice, low locants are given to the spiro atom, then to the heteroatoms".
'6-sila-2,2'-spirobi[bicyclo[2.2.1]heptane] (PIN)' (:10201), '6-oxa-6'-thia-2,2'-spirobi[bicyclo
[2.2.1]heptane] (PIN)' (:10203).
 (:16761): "the ending 'ene' is cited after the last bracket of the spiro name. The
final letter 'e' of the saturated hydrocarbon name is elided if followed by a vowel. If there is
a choice, low locants are assigned, in order, to spiro junction(s), heteroatoms and double bonds".
'2,2'-spirobi[bicyclo[2.2.1]heptan]-5-ene (PIN)' (:16771), '5,6'-dioxa-2,2'-spirobi[bicyclo
[2.2.2]octane]-7,7'-diene (PIN)' (:16775). (:3193): "Primed locants are placed
immediately after the corresponding unprimed locants in a set arranged in ascending order".

 (the Blue Book): "Monospiro ring systems with different ring components... are
formed by placing the ring component names in alphanumerical order within square brackets....
Locants of the second ring component are primed and thus any locants needed to name it are placed
in square brackets. Indicated hydrogen (see is cited in front of the name if needed in the
complete structure." (:10295): alphanumerical order, then "italic fusion
letters and numbers, heteroatom locants"; "All locants present in bicyclic fused benzo ring
component or Hantzsch-Widman named component are placed in brackets". '2'H,5H-spiro[thieno
[2,3-b]furan-4,3'-thieno[3,2-b]furan] (PIN)' (:10310). (:19601):
'2'H-spiro[cyclopentane-1,1'-isoquinoline] (PIN)' (:19607), '1'H-spiro[cyclopentane-1,2'-quinoline]
(PIN)' (:19632), '2'H-spiro[cyclopentane-1,3'-quinoline] (PIN)' (:19636). (:17042)
'4'a,5',6',7',8',8'a-hexahydro-1'H-spiro[imidazolidine-4,2'-quinoxaline] (PIN)' (:17050): hydro
prefixes, then indicated hydrogen, in front of the spiro name. (:49156):
'(1R)-5'H-spiro[indene-1,2'-[1,3]oxazole] (PIN)' (:49162). (b) (:3246): indicated hydrogen
takes the lowest locants among the descriptions of the structure. 'spiro[[1,3]dioxolane-2,1'-indene]
(PIN)' (:14910), 'spiro[piperidine-4,9'-xanthene] (PIN)' (:10268), '1'H-spiro[imidazolidine-4,2'-
quinoxaline] (PIN)' (:10270).
Where the component's preferred form cannot be built (a double bond of a von Baeyer or
carbocyclic component, cited as an 'ene' ending after the bracket,:16761 and
:17052), the older name is labelled below the PIN. A spiro name that cites the principal
characteristic group as a prefix ('trihydroxy', '3-oxo') is not the PIN either; the suffix
form 'spiro[4.5]decane-1,7-dione (PIN)',:28404): the default tier declines it.

 Cyclic ketones (the Blue Book): "Ketones resulting from the substitution of >CH2
groups are named substitutively using the suffix 'one' to designate the principal
characteristic group." (:28390) 'spiro[4.5]decane-1,7-dione (PIN)' (:28404). (:9941): the
spiro rules apply to "parent hydrides containing free spiro unions": the rings joined by a spiro
atom are one parent hydride, so every ketone on it is a suffix. (:25207): "if the number
of occurrences of the principal group is the same in two or more portions, the ring or ring system
is chosen as parent hydride".
"""
import pytest

from orthonym.assembly.naming_utils import apply_enclosing_marks, compute_nesting_depth
from tests.support.pin_tiers import (
    assert_declined_at_default,
    assert_not_pin_labelled,
    assert_pin_at_both_tiers,
    name_breadth,
)
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

# Build A: superscript locants in the dispiro descriptor.
DISPIRO_ROWS = [
    ("C1CC2(C1)CCC1(CCC1)CC2", "dispiro[3.2.3^7.2^4]dodecane"),                       #:9981
    ("C1CC2(C1)CCC1(CC2)CC1", "dispiro[2.2.3^6.2^3]undecane"),                        #:9985
    ("C1CCC2(C1)OOC1(CCCC1)OO2", "6,7,13,14-tetraoxadispiro[4.2.4^8.2^5]tetradecane"),  #:10087
    ("C1CC2(C1)CNC1(CCC1)CN2", "5,11-diazadispiro[3.2.3^7.2^4]dodecane"),              #:19597
]

# Class members: dispiro systems of other shapes (adjacent spiro atoms, a charged spiro atom).
DISPIRO_CLASS_ROWS = [
    ("C1CCC2(CC1)CCC1(CC2)CCC1", "dispiro[3.2.5^7.2^4]tetradecane"),
    ("C1CC2(CC1)CC1(CC2)CCCC1", "dispiro[4.1.4^7.2^5]tridecane"),
    ("C1CC2(C1)CC3(CC2)CC3", "dispiro[2.1.3^5.2^3]decane"),
    ("C1CC11COC11CCC1", "8-oxadispiro[2.0.3^4.2^3]nonane"),
    ("O=C1CC2(CC2)C11CO1", "1-oxadispiro[2.0.2^4.2^3]octan-8-one"),
    ("CC1CC11OCC11CN1", "5-methyl-7-oxa-1-azadispiro[2.0.2^4.2^3]octane"),
    ("C1CCCCC12[N+]1(CCCCC1)CCC2", "6-azadispiro[5.0.5^7.3^6]pentadecan-6-ium"),
]

# Build D: numbering of a spiro parent with prefixes. "(f) detachable alphabetized
# prefixes, all considered together in a series of increasing numerical order" (:3301), "(g)
# lowest locants for the substituent cited first as a prefix in the name" (:3307), "(j)... the
# lower locant is assigned to CIP stereodescriptors Z, R, M, and r... that are preferred to E,
# S, P, and s" (:3346). (:49156): '(5R,7S)-1,8-dioxadispiro[4.1.4^7.2^5]tridecane
# (PIN)' (:49166), '(1S,5R,7S)-1,7-dimethylspiro[4.5]decane (PIN)' (:49172).
NUMBERING_ROWS = [
    ("C1CO[C@@]2(C1)CC[C@]1(CCCO1)C2", "(5R,7S)-1,8-dioxadispiro[4.1.4^7.2^5]tridecane"),  #:49166
    ("C[C@H]1CCC[C@]2(CCC[C@@H]2C)C1", "(1S,5R,7S)-1,7-dimethylspiro[4.5]decane"),        #:49172
]

NUMBERING_CLASS_ROWS = [
    ("CC1CCCC12CCCCC2", "1-methylspiro[4.5]decane"),                      # (f), was 4-methyl
    ("CC1CCCC2(C1)CCCC2", "7-methylspiro[4.5]decane"),                    # (f), was 9-methyl
    ("ClC1CCC2(CCCCC2)CC1", "3-chlorospiro[5.5]undecane"),                # (f), was 9-chloro
    ("CC1CCCC2(CCCC2C)C1", "1,7-dimethylspiro[4.5]decane"),               # (f), was 4,9
    ("BrC1CCCC12CCCC2Cl", "1-bromo-6-chlorospiro[4.4]nonane"),            # (g)
    ("C1CO[C@@]2(C1)CC[C@@]1(CCCO1)C2", "(5S,7S)-1,8-dioxadispiro[4.1.4^7.2^5]tridecane"),
    ("C[C@@H]1CCC[C@]2(CCC[C@@H]2C)C1", "(1S,5R,7R)-1,7-dimethylspiro[4.5]decane"),
    ("CC1CC2(CC1)CC1(CC2)CC(C)C1", "2,8-dimethyldispiro[3.1.4^6.2^4]dodecane"),  # (f) on a dispiro
]

# Build B: 'spirobi' names.
SPIROBI_ROWS = [
    ("c1ccc2c(c1)CC1(Cc3ccccc3S1)S2", "3H,3'H-2,2'-spirobi[[1]benzothiophene]"),          #:10160
    ("c1ccc2c(c1)CC1(CSc3ccccc31)S2", "2'H,3H-2,3'-spirobi[[1]benzothiophene]"),          #:10176
    ("C1=CC2(C=Cc3ccccc3O2)c2ccccc2O1", "2,4'-spirobi[[1]benzopyran]"),                   #:10182
    ("C1CC2CC1CC21CC2C[SiH2]C1C2", "6-sila-2,2'-spirobi[bicyclo[2.2.1]heptane]"),          #:10201
    ("C1OC2CC1CC21CC2CSC1C2", "6-oxa-6'-thia-2,2'-spirobi[bicyclo[2.2.1]heptane]"),       #:10203
    ("C1=CC2CC1CC21CC2CCC1C2", "2,2'-spirobi[bicyclo[2.2.1]heptan]-5-ene"),               #:16771
    ("C1=CC2OCC1CC21CC2C=CC1CO2", "5,6'-dioxa-2,2'-spirobi[bicyclo[2.2.2]octane]-7,7'-diene"),  #:16775
]

SPIROBI_CLASS_ROWS = [
    ("C12CC3(CC(C1)OC2)CC1CC(C3)OC1", "6,6'-dioxa-3,3'-spirobi[bicyclo[3.2.1]octane]"),   #:10197
    ("C1CC2SCC1CC21CC2CCC1CS2", "5,6'-dithia-2,2'-spirobi[bicyclo[2.2.2]octane]"),        #:10195
    ("C1=CC2CC1CC21CC2C=CC1C2", "2,2'-spirobi[bicyclo[2.2.1]heptane]-5,5'-diene"),
    ("C1=Cc2ccccc2C12C=Cc1ccccc12", "1,1'-spirobi[indene]"),                              #:10164
    ("C1CC2CC1CC21CC2CCC1C2", "2,2'-spirobi[bicyclo[2.2.1]heptane]"),
]

# Build C: components by their mancude names, indicated hydrogen and hydro prefixes in front.
COMPONENT_ROWS = [
    ("c1cc2c(o1)SCC21COc2ccsc21", "2'H,5H-spiro[thieno[2,3-b]furan-4,3'-thieno[3,2-b]furan]"),  #:10310
    ("C1=Cc2ccccc2C2(CCCC2)N1", "2'H-spiro[cyclopentane-1,1'-isoquinoline]"),       #:19607
    ("C1=CC2(CCCC2)Nc2ccccc21", "1'H-spiro[cyclopentane-1,2'-quinoline]"),          #:19632
    ("C1=c2ccccc2=NCC12CCCC2", "2'H-spiro[cyclopentane-1,3'-quinoline]"),           #:19636
    ("C1=C[C@]2(N=CCO2)c2ccccc21", "(1R)-5'H-spiro[indene-1,2'-[1,3]oxazole]"),     #:49162
]

COMPONENT_CLASS_ROWS = [
    ("N1CNC2(C1)Nc1ccccc1N=C2", "1'H-spiro[imidazolidine-4,2'-quinoxaline]"),       #:10270
    ("C1COC2(O1)C=Cc1ccccc12", "spiro[[1,3]dioxolane-2,1'-indene]"),                #:14910
    ("C1CNCCC12c1ccccc1Oc1ccccc12", "spiro[piperidine-4,9'-xanthene]"),             #:10268
    ("C1CCC2(CC1)C=Cc1ccccc12", "spiro[cyclohexane-1,1'-indene]"),                  #:10266
    ("C1CNCCC12CNc1ccccc12", "1,2-dihydrospiro[indole-3,4'-piperidine]"),          # was spiro[2,3-dihydro-1H-indole-3,4'-piperidine]
]

# Not the PIN: the default tier declines, best-effort keeps a round-tripping name.
NOT_PIN_ROWS = [
    ("CCCCC[C@H]1O[C@]2(OCc3c(O)cccc32)[C@H](O)[C@@H]1O",
     "(1S,3'R,4'S,5'R)-3',4,4'-trihydroxy-5'-pentylspiro[1,3-dihydro-2-benzofuran-1,2'-oxolane]"),
    ("O=C1OC2(c3ccccc31)c1cc(Br)c(O)c(Br)c1Oc1c2cc(Br)c(O)c1Br",
     "2',4',5',7'-tetrabromo-3',6'-dihydroxy-3-oxospiro[1,3-dihydro-2-benzofuran-1,9'-xanthene]"),
    ("CC1(C)CCC(=O)C(C)(O)[C@@]12CCC(O)(CO)C(=O)C2",
     "(6R)-1,9-dihydroxy-9-(hydroxymethyl)-1,5,5-trimethyl-8-oxospiro[5.5]undecan-2-one"),
    # the other dev candidates (members.json, SPIRO_COMPONENT examples)
    ("COC1=CC(=O)O[C@]12Oc1cc(O)cc(C)c1C[C@H]2C",
     "(2S,3'R)-7'-hydroxy-3-methoxy-3',5'-dimethyl-5-oxospiro[2,5-dihydrofuran-2,2'-3,4-"
     "dihydro-2H-1-benzopyran]"),
    ("CCC[C@H]1C[C@H](O)[C@]2(OC(=O)c3c2cc(OC)c(OC)c3O)O1",
     "(1S,3'S,5'S)-3',4-dihydroxy-5,6-dimethoxy-3-oxo-5'-propylspiro[1,3-dihydro-2-"
     "benzofuran-1,2'-oxolane]"),
    ("O=C1c2c(O)cc(O)cc2OC12OCc1cc(O)c(O)cc1C2O",
     "4,4',6,6',7'-pentahydroxy-3-oxospiro[2,3-dihydro-1-benzofuran-2,3'-3,4-dihydro-1H-2-"
     "benzopyran]"),
]

# The principal characteristic group cited as a prefix on a spiro parent: declined at the
# default tier (the hoisted component name is still built at best-effort).
DECLINED_ROWS = [
    "CCCCC[C@H]1O[C@]2(OCc3c(O)cccc32)[C@H](O)[C@@H]1O",       # 3',4,4'-trihydroxy (PIN: -triol)
    "O=C1OC2(c3ccccc31)c1cc(Br)c(O)c(Br)c1Oc1c2cc(Br)c(O)c1Br",  # 3-oxo (PIN: -3-one)
    "OC1CCC2(CC1)c1ccccc1-c1ccccc12",                           # 4-hydroxy (PIN: -4-ol)
]

# Build E: every ketone of the spiro parent is a suffix.
DIONE_ROWS = [
    ("O=C1CCCC2(CCCC2=O)C1", "spiro[4.5]decane-1,7-dione"),  #:28404, was 1-oxospiro[4.5]decan-7-one
]

DIONE_CLASS_ROWS = [
    ("O=C1CCC2(CC1)CCC(=O)CC2", "spiro[5.5]undecane-3,9-dione"),      # was 9-oxo...-3-one
    ("O=C1CCCC12CCCCC2=O", "spiro[4.5]decane-1,6-dione"),             # was 1-oxo...-6-one
    ("O=C1CCC2(CC1)CCCC2=O", "spiro[4.5]decane-1,8-dione"),           # was 1-oxo...-8-one
    ("O=C1CCCCC12CCC(=O)C2", "spiro[4.5]decane-2,6-dione"),           # was 2-oxo...-6-one
    ("O=C1C=CC2(C=C1)CCC(=O)CC2", "spiro[5.5]undeca-1,4-diene-3,9-dione"),
    # dev candidate, was '(6R)-1,9-dihydroxy-9-(hydroxymethyl)-1,5,5-trimethyl-8-oxospiro[5.5]
    # undecan-2-one' at pin_verified
    ("CC1(C)CCC(=O)C(C)(O)[C@@]12CCC(O)(CO)C(=O)C2",
     "(6R)-1,9-dihydroxy-9-(hydroxymethyl)-1,5,5-trimethylspiro[5.5]undecane-2,8-dione"),
]

#: a ketone on a pendant CHAIN of the ring parent stays a prefix ('2-oxobutyl').
DIONE_CONTROL_ROWS = [
    ("CCC(=O)CC1CCCC(=O)C1=O", "3-(2-oxobutyl)cyclohexane-1,2-dione"),
]

CONTROL_ROWS = [
    ("C1CCC2(C1)CCCC2", "spiro[4.4]nonane"),                       #:9973
    ("C1CC12CCC1(CC1)CCC1(CC1)CC2", "trispiro[2.2.2^6.2.2^11.2^3]pentadecane"),
]

#: the substituent prefix around a superscripted descriptor takes parentheses.
ENCLOSURE_ROWS = [
    ("C1CCC12CCC1(CCC1)CC2CC(=O)O", "(dispiro[3.2.3^7.2^4]dodecan-5-yl)acetic acid"),
    ("OC(=O)CC1CC2CC1C1CC21", "(tricyclo[3.2.1.0^2,4]octan-6-yl)acetic acid"),
    ("OC(=O)CC1CC2CCC1C1CCC21", "(tricyclo[4.2.2.0^2,5]decan-8-yl)acetic acid"),
]


@pytest.mark.parametrize("smiles,pin", DISPIRO_ROWS)
def test_dispiro_superscripts(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", DISPIRO_CLASS_ROWS)
def test_dispiro_class_member(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", NUMBERING_ROWS)
def test_numbering(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", NUMBERING_CLASS_ROWS)
def test_numbering_class_member(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", SPIROBI_ROWS)
def test_spirobi(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", SPIROBI_CLASS_ROWS)
def test_spirobi_class_member(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", COMPONENT_ROWS)
def test_spiro_components(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", COMPONENT_CLASS_ROWS)
def test_spiro_components_class_member(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,non_pin", NOT_PIN_ROWS)
def test_not_pin_labelled(smiles, non_pin):
    assert_not_pin_labelled(smiles, non_pin)


@pytest.mark.parametrize("smiles", DECLINED_ROWS)
def test_prefix_cited_principal_group_is_declined(smiles):
    assert_declined_at_default(smiles)


@pytest.mark.parametrize("smiles,pin", DIONE_ROWS)
def test_spiro_ketones_are_suffixes(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", DIONE_CLASS_ROWS)
def test_spiro_ketones_class_member(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", DIONE_CONTROL_ROWS)
def test_chain_ketone_stays_a_prefix(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,pin", CONTROL_ROWS)
def test_control(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)


@pytest.mark.parametrize("smiles,name", ENCLOSURE_ROWS)
def test_superscripted_descriptor_does_not_nest(smiles, name):
    assert name_is_rt_exact(name, smiles)
    assert name_breadth(smiles).get("name") == name


@pytest.mark.parametrize("prefix", [
    "dispiro[3.2.3^7.2^4]dodecan-5-yl",
    "tricyclo[3.3.1.1^3,7]decan-1-yl",
    "2-methyltricyclo[3.2.1.0^2,4]octan-6-yl",
])
def test_superscripted_descriptor_bracket_is_not_a_nesting_mark(prefix):
    assert compute_nesting_depth(prefix) == 0
    assert apply_enclosing_marks(prefix) == f"({prefix})"


# Batch 2 fix a performance pass (F-06): the spliced two-component form built on the von Baeyer floor or the
# per-ring-system path ('allow_vonbaeyer_component' / 'restrict_atoms') is labelled below the PIN
# when a component keeps its own indicated hydrogen or hydro prefixes inside the brackets
#,:10152;,:10260), as on the default-argument path.
def _mixed_spiro_name_and_records(smiles, **kwargs):
    from rdkit import Chem

    import orthonym.metrics.provenance as provenance
    from orthonym.rules.spiro import name_mixed_spiro_fused
    mol = Chem.MolFromSmiles(smiles)
    if kwargs.pop("whole", False):
        kwargs["restrict_atoms"] = set(range(mol.GetNumAtoms()))
    token = provenance._NON_PIN_FRAGMENTS.set(())
    try:
        res = name_mixed_spiro_fused(mol, **kwargs)
        return (res[0] if res else None), provenance._NON_PIN_FRAGMENTS.get()
    finally:
        provenance._NON_PIN_FRAGMENTS.reset(token)


@pytest.mark.parametrize("smiles,spliced", [
    ("C1CCC2(CC1)CCc1ccccc12", "spiro[2,3-dihydro-1H-indene-1,1'-cyclohexane]"),
    ("C1CCC2(C1)CCOc1ccccc12", "spiro[3,4-dihydro-2H-1-benzopyran-4,1'-cyclopentane]"),
])
@pytest.mark.parametrize("kwargs", [
    {"allow_vonbaeyer_component": True},
    {"allow_vonbaeyer_component": True, "whole": True},
])
def test_spliced_form_on_floor_path_is_labelled_below_pin(smiles, spliced, kwargs):
    name, records = _mixed_spiro_name_and_records(smiles, **dict(kwargs))
    assert name == spliced
    assert spliced in records, records
    assert name_is_rt_exact(spliced, smiles)


@pytest.mark.parametrize("component,cites", [
    ("1H-indene", True), ("2,3-dihydro-1H-indene", True), ("2H-1-benzopyran", True),
    ("1,2,3,4-tetrahydronaphthalene", True),
    ("indene", False), ("bicyclo[2.2.1]heptane", False), ("cyclohexane", False),
    ("quinoline", False),
])
def test_component_cites_added_hydrogen(component, cites):
    from orthonym.rules.spiro import _component_cites_added_hydrogen
    assert _component_cites_added_hydrogen(component) is cites
