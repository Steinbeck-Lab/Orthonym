"""PIN class program, Task 7: ring and one-carbon ions at the PIN tier.

 "ANIONS FORMED BY ADDITION OF HYDRIDE IONS" (the Blue Book): "(1) the suffix
'uide' describes an anion formally derived by adding a hydride ion, H-, to a parent hydride
name" (:41093); "Method (1) leads to preferred IUPAC names." (:41098); '1,1-dimethylborinan-
1-uide (PIN) (not 1,1-dimethyl-1-boratacyclohexane)' (:41120). For anionic centres in rings:
"(1) by forming the name of the neutral compound according to skeletal replacement ('a')
nomenclature and using the suffixes 'ide' and 'uide' to describe the anionic centers"
(:41126), "Method (1) results in preferred IUPAC names." (:41129); '2,2-dimethyl-2-
boraspiro[4.5]decan-2-uide (PIN)' (:41143), '1-phosphabicyclo[2.2.2]octan-1-uide (PIN)'
(:41149).

 "Functional class nomenclature" (:40870): "Systematic names (see are
preferred IUPAC names." (:40872); 'acetyl anion 1-oxoethan-1-ide (PIN)' (:40878). The =O of
the chain is the 'oxo' prefix, (:28232) "the prefix 'oxo' is used when a characteristic
group having seniority is present" (:28234). Between two prefix blocks the locant takes a
hyphen, (a) "to separate locants from words or word fragments" (:6938).

 "General rule for systematically naming cationic centers in parent hydrides"
(:41366): "A cation derived formally by adding one or more hydrons to any position of a
neutral parent hydride... is named by replacing the final letter 'e' of the parent hydride
name, if any, by the suffix 'ium', preceded by multiplying prefixes 'di', 'tri', etc. to
denote the multiplicity of identical cationic centers." (:41368); 'tetramethyldiazene-1,2-
diium (PIN)' (:41407), '1,4-dioxane-1,4-diium (PIN)' (:41409). The 'e' stays before 'di'
 (a),:7595: elided only before a vowel). Numbering, (:3219): heteroatoms of
the ring first, then (c) suffixes (:3256), then (f) prefixes (:3301); the centres compare as
a set, (:3189,:3191).

At the base the default tier declined all five rows: the mononuclear '-uide' emitter declines
a ring centre (Task 1), the carbanion emitter declined the =O substituent, and the multi-cation
router had no ring parent construction; best-effort shipped the general engine's names at
systematic_verified.

Declined at the PIN tier (best-effort keeps its RT-exact name):
- '1-methoxy-1,3-dimethyl-1H-1-benzoborol-1-uide (PIN)' (:41122): the ring prefix helper
  declines a heteroatom-rooted group of more than one atom (methoxy);
- a ring holding an O+ and an N+ centre (two cation classes; the router keeps one class);
- several cationic centres on a parent named with indicated hydrogen or hydro prefixes
   criteria (b) and (e),:3246,:3288, are not modelled by that construction);
- a carbanion with a chloro prefix (the chain-substituent classifier does not name it);
- a ring '-uide' whose mancude parent the neutral namer spells without its indicated
  hydrogen ('1-methylphosphole'): a '-uide' adds a hydride, so the parent's indicated
  hydrogen stays in the name ('1H-1-benzoborol-1-uide',:41122). The phosphindole
  centre is named since the neutral parent is '1H-phosphindole' (the name-quality
  program, lane L1a): '1H-phosphindol-1-uide', as '1H-phosphol-1-uide'.
"""
import pytest
from rdkit import Chem

from orthonym.rules import ions
from tests.support.pin_tiers import assert_declined_at_default, assert_pin_at_both_tiers

pytestmark = pytest.mark.opsin_gate

PIN_ROWS = [
    ("C[B-]1(C)CCC2(CCCCC2)C1", "2,2-dimethyl-2-boraspiro[4.5]decan-2-uide"),   #:41143
    ("C[C-]=O", "1-oxoethan-1-ide"),                                              #:40878
    ("C1C[PH-]2CCC1CC2", "1-phosphabicyclo[2.2.2]octan-1-uide"),                  #:41149
    ("C1C[OH+]CC[OH+]1", "1,4-dioxane-1,4-diium"),                                #:41409
    ("C[B-]1(C)CCCCC1", "1,1-dimethylborinan-1-uide"),                            #:41120
]

# Class members (same producers, same rules); every name read back by OPSIN 2.9.0 to the
# input's full InChIKey. The base declined each at the default tier.
CLASS_ROWS = [
    # ring '-uide': in place (the centre carries H) and with the ligands as prefixes
    ("[BH2-]1CCCCC1", "borinan-1-uide"),
    ("C[BH-]1CCCCC1", "1-methylborinan-1-uide"),
    ("C[B-]1(C)CCOCC1", "4,4-dimethyl-1,4-oxaborinan-4-uide"),
    ("C[B-]1(C)CCCC1", "1,1-dimethylborolan-1-uide"),
    ("F[B-]1(F)CCCC1", "1,1-difluoroborolan-1-uide"),
    ("c1ccc([B-]2(c3ccccc3)CCCC2)cc1", "1,1-diphenylborolan-1-uide"),
    ("C[SiH2-]1CCCC1", "1-methylsilolan-1-uide"),
    ("C[P-]1(C)CCCCC1", "1,1-dimethylphosphinan-1-uide"),
    ("C1C[BH-]2CCC1CC2", "1-borabicyclo[2.2.2]octan-1-uide"),
    ("O[B-]1(O)OCCO1", "2,2-dihydroxy-1,3,2-dioxaborolan-2-uide"),
    # the parent's indicated hydrogen stays (a '-uide' adds a hydride), as in:41122
    ("C[B-]1(C)c2ccccc2-c2ccccc21", "5,5-dimethyl-5H-dibenzo[b,d]borol-5-uide"),  #:37250
    ("C[B-]1(C)CC=CC1", "1,1-dimethyl-2,5-dihydro-1H-borol-1-uide"),
    ("[BH2-]1C=CC=C1", "1H-borol-1-uide"),
    ("C[BH-]1C=CC=C1", "1-methyl-1H-borol-1-uide"),
    ("[PH2-]1C=CC=C1", "1H-phosphol-1-uide"),
    ("C[P-]1(C)C=CC=C1", "1,1-dimethyl-1H-phosphol-1-uide"),
    ("[PH2-]1C=Cc2ccccc21", "1H-phosphindol-1-uide"),
    ("C[P-]1(C)C=Cc2ccccc21", "1,1-dimethyl-1H-phosphindol-1-uide"),
    # / carbanions with 'oxo' prefixes
    ("CC[C-]=O", "1-oxopropan-1-ide"),
    ("CC(=O)[CH2-]", "2-oxopropan-1-ide"),
    ("CC(C)[C-]=O", "2-methyl-1-oxopropan-1-ide"),
    ("[CH-]=O", "oxomethanide"),
    ("O=C[CH-]C=O", "1,3-dioxopropan-2-ide"),
    ("CC(=O)[C-](C)C(C)=O", "3-methyl-2,4-dioxopentan-3-ide"),
    ("CC(C)(C)C(=O)[CH-]C(=O)C(C)(C)C", "2,2,6,6-tetramethyl-3,5-dioxoheptan-4-ide"),
    ("C=CC[C-]=O", "1-oxobut-3-en-1-ide"),
    ("N#CC[C-]=O", "2-cyano-1-oxoethan-1-ide"),
    ("CC[C-](C)C=O", "2-methyl-1-oxobutan-2-ide"),
    ("C[C-](C)C=O", "2-methyl-1-oxopropan-2-ide"),
    # carbanions with two kinds of prefix: the hyphen before the second locant
    ("CC[C-](CC)C(C)C", "3-ethyl-2-methylpentan-3-ide"),
    ("CCC(CC)[C-](C)CC", "4-ethyl-3-methylhexan-3-ide"),
    ("N#CC[C-](C)CC", "1-cyano-2-methylbutan-2-ide"),
    # several hydron-addition centres of one ring parent
    ("C1C[NH2+]CC[NH2+]1", "piperazine-1,4-diium"),
    ("C1C[SH+]CC[SH+]1", "1,4-dithiane-1,4-diium"),
    ("C1C[OH+]C[OH+]C1", "1,3-dioxane-1,3-diium"),
    ("C1C[OH+]CC[OH+]C1", "1,4-dioxepane-1,4-diium"),
    ("c1c[nH+]cc[nH+]1", "pyrazine-1,4-diium"),
    ("c1c[nH+]c[nH+]c1", "pyrimidine-1,3-diium"),
    ("C1C[NH2+]C[NH2+]C1", "1,3-diazinane-1,3-diium"),                           #:29344
    ("C1[NH2+]C[NH2+]C[NH2+]1", "1,3,5-triazinane-1,3,5-triium"),                #:29360
    ("C1[NH2+]C[NH2+]CN1", "1,3,5-triazinane-1,3-diium"),
    ("C1C[NH+]2CC[NH+]1CC2", "1,4-diazabicyclo[2.2.2]octane-1,4-diium"),
    ("C1C[NH+]2CC[NH+]1C2", "1,4-diazabicyclo[2.2.1]heptane-1,4-diium"),
    ("c1cc2c[nH+]ccc2c[nH+]1", "2,6-naphthyridine-2,6-diium"),
    # substituted: the centres first as a set, then the prefixes, in one numbering
    ("C[NH+]1CC[NH+](C)CC1", "1,4-dimethylpiperazine-1,4-diium"),
    ("C[N+]1(C)CC[N+](C)(C)CC1", "1,1,4,4-tetramethylpiperazine-1,4-diium"),
    ("C[N+]12CC[N+](C)(CC1)CC2", "1,4-dimethyl-1,4-diazabicyclo[2.2.2]octane-1,4-diium"),
    ("CC1C[NH2+]CC[NH2+]1", "2-methylpiperazine-1,4-diium"),
    ("Cc1c[nH+]cc[nH+]1", "2-methylpyrazine-1,4-diium"),
    ("C[n+]1c[nH+]ccc1", "1-methylpyrimidine-1,3-diium"),       # not 3-methyl (f))
    ("CN1C[NH2+]C[NH2+]C1", "5-methyl-1,3,5-triazinane-1,3-diium"),  # centres 1,3 before (f)
    # a ring P is the parent's own heteroatom, not a 'phosphine' group (the guard below)
    ("[PH2+]1CCCCC1", "phosphinan-1-ium"),
    ("C[PH+]1CCCCC1", "1-methylphosphinan-1-ium"),
]

# The Task 1 ion controls and the positives of this task's trace: already PIN at both tiers.
CONTROL_ROWS = [
    ("[B-](C)(C)(C)C", "tetramethylboranuide"),
    ("C[P-](C)(C)C", "tetramethylphosphanuide"),
    ("C1CCCCCSCCOCCCC1", "1-oxa-4-thiacyclotetradecane"),
    ("C1CC[SiH2]CCCCCCCCC[SiH2]CCC=CC=C1", "1,11-disilacycloicosa-4,6-diene"),
    ("C1CCCCC[NH2+]CCOCCCC1", "1-oxa-4-azacyclotetradecan-4-ium"),
    ("C[Si-]1CCCC1", "1-methylsilolan-1-ide"),
    ("C1CC[NH2+]CC1", "piperidin-1-ium"),
    ("C1CCCC[N+]12CCCC2", "5-azaspiro[4.5]decan-5-ium"),
    ("CCC(C)[CH-]C(C)C", "2,4-dimethylhexan-3-ide"),
]

DECLINED_ROWS = [
    "CO[B-]1(C)C=C(C)c2ccccc21",   # methoxy ligand (see the module docstring)
    "C[PH-]1C=CC=C1",              # the parent name lacks its indicated hydrogen
    "C1=C[NH2+]C=C[NH2+]1",        # a dication on a hydro-prefixed parent
    "C1C[OH+]CC[NH2+]1",           # two cation classes
    "ClCC[C-]=O",                  # chloro prefix on the carbanion chain
    # 'ethan-1-ide' is not the PIN: (b) omits the locant '1' on a two-atom chain
    # with one feature ('ethanol (PIN)':2907), and the book writes "'ethanide' not 'ethyl
    # anion' for CH3-CH2" (:40382); the spelling check of the label site lowers it, and the
    # best-effort tier keeps 'ethan-1-ide' (systematic_verified, full read-back)
    "[CH2-]C",
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


def _charged(mol):
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetFormalCharge()]


@pytest.mark.parametrize("smiles,name", [
    ("C[B-]1(C)CCCCC1", "1,1-dimethylborinan-1-uide"),
    ("C1C[PH-]2CCC1CC2", "1-phosphabicyclo[2.2.2]octan-1-uide"),
    ("C[B-]1(C)CCC2(CCCCC2)C1", "2,2-dimethyl-2-boraspiro[4.5]decan-2-uide"),
])
def test_ring_uide_emitter(smiles, name):
    mol = Chem.MolFromSmiles(smiles)
    assert ions.emit_parent_hydride_cumulative_suffix(mol, _charged(mol)[0], 'uide') == name


@pytest.mark.parametrize("smiles", [
    "[B-](C)(C)(C)C", "C[SiH4-]",          # open centres: the mononuclear emitter's
    "[CH2-]C",                             # a carbanion (hydron loss), not a '-uide'
    "[SiH2-]1C=CC=C1",                     # Si at its standard bonding number
    "c1c[nH+]c2[nH]ccc2c1",                # a cation
])
def test_uide_emitter_leaves_an_open_centre_to_the_mononuclear_emitter(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert ions.emit_parent_hydride_cumulative_suffix(mol, _charged(mol)[0], 'uide') == ''


def test_oxo_prefix_only_on_request():
    # The ylide composer splices its own prefix in front of the carbanion name without
    # re-sorting, so the default call keeps declining an =O on the chain.
    mol = Chem.MolFromSmiles("C[C-]=O")
    c = _charged(mol)[0]
    assert ions.emit_parent_hydride_cumulative_suffix(mol, c, 'ide') == ''
    assert ions.emit_parent_hydride_cumulative_suffix(
        mol, c, 'ide', oxo_prefixes=True) == '1-oxoethan-1-ide'


@pytest.mark.parametrize("smiles,name", [
    ("CC[C-](CC)C(C)C", "3-ethyl-2-methylpentan-3-ide"),
    ("CC(C)C[C-](CC)CC", "3-ethyl-5-methylhexan-3-ide"),
])
def test_carbanion_prefix_blocks_are_hyphenated(smiles, name):
    mol = Chem.MolFromSmiles(smiles)
    assert ions.emit_parent_hydride_cumulative_suffix(mol, _charged(mol)[0], 'ide') == name


@pytest.mark.parametrize("smiles,name", [
    ("C1C[OH+]CC[OH+]1", "1,4-dioxane-1,4-diium"),
    ("C[n+]1c[nH+]ccc1", "1-methylpyrimidine-1,3-diium"),
    ("CN1C[NH2+]C[NH2+]C1", "5-methyl-1,3,5-triazinane-1,3-diium"),
    ("C1C[NH+]2CC[NH+]1C2", "1,4-diazabicyclo[2.2.1]heptane-1,4-diium"),
])
def test_ring_poly_cation_emitter(smiles, name):
    mol = Chem.MolFromSmiles(smiles)
    assert ions.emit_ring_poly_cation_ium(mol, _charged(mol)) == name


@pytest.mark.parametrize("smiles", [
    "C1CC[NH2+]CC1",                       # one centre
    "c1cc(-c2cc[nH+]cc2)cc[nH+]1",         # two ring systems
    "C1C[CH+]CC[NH2+]1",                   # a carbon centre (hydride loss, not a hydron)
    "C1CC[SH3+]C[SH3+]CC1",                # the parent has no name here (nonstandard bonding)
    "O=C1C[NH2+]CC[NH2+]1",                # an exocyclic double bond
    "C1=C[NH2+]C=[NH+]1",                  # the parent cites indicated hydrogen
    "C1=C[NH2+]C=C[NH2+]1",                # the parent has hydro prefixes
])
def test_ring_poly_cation_emitter_declines(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert ions.emit_ring_poly_cation_ium(mol, _charged(mol)) == ''


def test_ring_phosphorus_is_not_a_principal_group_of_its_ring():
    ring = Chem.MolFromSmiles("C1CP2CCC1CC2")
    rs = set(range(ring.GetNumAtoms()))
    assert ions._ring_bears_principal_group(ring) is True
    assert ions._ring_bears_principal_group(ring, rs) is False
    # an exocyclic phosphanyl group is still a group of the ring's substituent
    exo = Chem.MolFromSmiles("PC1CCCCC1")
    ring_atoms = {a.GetIdx() for a in exo.GetAtoms() if a.IsInRing()}
    assert ions._ring_bears_principal_group(exo, ring_atoms) is True


@pytest.mark.parametrize("smiles,parent,cited", [
    ("C[PH-]1C=CC=C1", "1-methylphosphole", False),
    ("C[PH-]1C=CC=C1", "1-methyl-1H-phosphole", True),
    ("C[B-]1(C)CC=CC1", "2,5-dihydro-1H-borole", True),
    ("C[B-]1(C)CCCCC1", "borinane", True),          # saturated: nothing to cite
])
def test_uide_parent_must_cite_its_hydrogen(smiles, parent, cited):
    mol = Chem.MolFromSmiles(smiles)
    c = _charged(mol)[0]
    rs = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    assert ions._uide_parent_cites_hydrogen(mol, rs, c, parent) is cited
