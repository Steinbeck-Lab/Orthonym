"""Thioperoxol names,: a name ships only when something reads it back.

OPSIN 2.9.0 has no grammar for the suffixes '-SO-thioperoxol' and '-OS-thioperoxol', so the name of
every R-S-OH and R-O-SH used to ship with NO read-back (the ``thioperoxol`` grammar-gap carve-out: any
name that contained 'thioperoxol'). The CO audit (2026-10-17) read every carve-out name of a
PubChem / ChEBI / SureChEMBL sample through a check that does not use the producer: 52 of 308 were the
name of another molecule (a CH2 dropped between a ring and the suffix, ``C1CC1CSO`` named
``cyclopropane-1-SO-thioperoxol``; two substituents and a trimethoxymethyl on one carbon: five and six
bonds; a CF2-PH2 group named ``difluorophosphanylmethyl``; a second -SOH or -OSH group the name does not
cite; a cation or a protonation site the name does not state).

The rules:

* THE SUFFIXES 'SO-THIOPEROXOL', AND CHALOGEN ANALOGUES (the Blue Book): "The suffixes
  SO-thioperoxol and its chalcogen analogues are now introduced to replace the suffix 'sulfenic acid' and
  its chalcogen analogues that were discarded in the 1993 recommendations." and 'CH3-S-OH
  methane-SO-thioperoxol (PIN) (not methanesulfenic acid)' (:24326): the Blue Book equates the two names
  of the group -S-OH, so 'X-sulfenic acid' is a spelling of the same structure that OPSIN reads.
* Chalcogen analogues of hydroperoxides (:27961), (:27963): "Method (1) generates
  preferred IUPAC names." (:27970); Table 6.1 "Suffixes denoting peroxols (hydroperoxides) modified by
  functional replacement nomenclature" (:27972): -S-OH is '-SO-thioperoxol', -O-SH is '-OS-thioperoxol';
  'CH3-SOH (2) S-methyl thiohydroperoxide (no longer methanesulfenic acid)' (:27990).
* Basic functional suffixes (:17456): "The suffix 'peroxol', for -OOH, has been added to the list
  of basic suffixes. It is modified by functional replacement generating the suffixes '-OS-thioperoxol' for
  -OSH, and '-SO-thioperoxol' for -SOH." (:17460): '-OS-thioperoxol' is 'peroxol' with its second oxygen
  replaced by sulfur, so the name with 'peroxol' for the suffix is the name of the structure with that
  sulfur an oxygen.
* Anions derived from hydroxy compounds (:41011): 'CH3-CH2-S-O- ethane(SO-thioperoxolate)
  (PIN) (ethylsulfanyl)oxidanide (not ethanesulfenate; sulfenic acids are no longer recommended; see
  ' (:41035).

The engine reads the name back through the spelling of the same structure that OPSIN reads:
``X-sulfenic acid`` / ``X-sulfenate`` for -S-OH / -S-O-, ``X-peroxol`` (with the structure's -O-S-H made
-O-O-H) for -O-S-H. The tests read it again through another one, ``X-thiol`` and ``X-ol`` with the S-O
oxygen / O-S sulfur removed (the CO auditor's check), so no row passes by the same spelling twice.
"""
import re

import pytest
from rdkit import Chem

from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import _independent_parse, assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate

# (smiles, the name the default tier gave: right by the independent check, CO audit 255 of 308)
RIGHT = [
    ("CSO", "methane-SO-thioperoxol"),  #:24326 verbatim
    ("ClCCSO", "2-chloroethane-1-SO-thioperoxol"),  # the decorated mutant of the carve-out certificate
    ("C=COS", "eth-1-ene-OS-thioperoxol"),
    ("CCCSO", "propane-1-SO-thioperoxol"),
    ("C=CCOS", "prop-2-ene-1-OS-thioperoxol"),
    ("CC#COS", "prop-1-yne-1-OS-thioperoxol"),
    ("NC=CSO", "2-aminoeth-1-ene-1-SO-thioperoxol"),
    ("C(CSO)N", "2-aminoethane-1-SO-thioperoxol"),
    ("CCNCCOS", "2-(ethylamino)ethane-1-OS-thioperoxol"),
    ("C(SO)Cl", "chloromethane-SO-thioperoxol"),
    ("C1CC1SO", "cyclopropane-SO-thioperoxol"),      # (c), see RING_MONOSUBSTITUTED
    ("CC(N)SO", "1-aminoethane-1-SO-thioperoxol"),
    ("CCC(C)SO", "butane-2-SO-thioperoxol"),
    ("N/C=C\\SO", "(1Z)-2-aminoeth-1-ene-1-SO-thioperoxol"),
    ("C/C=C/SO", "(1E)-prop-1-ene-1-SO-thioperoxol"),
    ("CCC(C)OS", "butane-2-OS-thioperoxol"),
    ("CCOC(C)OS", "1-ethoxyethane-1-OS-thioperoxol"),
    ("C(N=C=O)OS", "isocyanatomethane-OS-thioperoxol"),
    ("C1C(CC1F)OS", "3-fluorocyclobutane-1-OS-thioperoxol"),
    ("CCCN(C)CCSO", "2-[methyl(propyl)amino]ethane-1-SO-thioperoxol"),
    ("C[C@@H]1CCC[C@@H]1OS", "(1S,2R)-2-methylcyclopentane-1-OS-thioperoxol"),
    ("CC[C@H](C)NCCCSO", "3-{[(2S)-butan-2-yl]amino}propane-1-SO-thioperoxol"),
    ("CN/C=C\\C(=C/COS)\\CN(C)C",
     "(2E,4Z)-3-[(dimethylamino)methyl]-5-(methylamino)penta-2,4-diene-1-OS-thioperoxol"),
    ("CC(C)OC(C)CCC(C)SO", "5-[(propan-2-yl)oxy]hexane-2-SO-thioperoxol"),
    ("SO[C@H](C)CC", "(2R)-butane-2-OS-thioperoxol"),
    ("C[C@H](SO)CC", "(2S)-butane-2-SO-thioperoxol"),
]
# the anions of the -S-OH group: the best-effort tier gave no name for any of them while the default tier did.
# Anions derived from hydroxy compounds (the Blue Book): 'CH3-CH2-S-O- ethane(SO-thioperoxolate)
# (PIN) (ethylsulfanyl)oxidanide (not ethanesulfenate;...)' (:41035): the parent without a locant is followed by the
# enclosed suffix; the producer spelled 'ethane-SO-thioperoxolate' and the name was labelled pin_verified.
RIGHT_ANIONS = [
    ("CS[O-]", "methane(SO-thioperoxolate)"),
    ("[Na+].CCS[O-]", "sodium ethane(SO-thioperoxolate)"),
    ("C(F)S[O-]", "fluoromethane(SO-thioperoxolate)"),
]
# no Blue Book row prints the enclosed suffix after a locant, so the spelling of these is not certified: the engine
# names them, a verified twin read-back, below the PIN (pin_unverified: the default tier declines them)
ANIONS_BELOW_PIN = [
    ("CCCS[O-]", "propane-1-SO-thioperoxolate"),
    ("N/C=C\\S[O-]", "(1Z)-2-aminoeth-1-ene-1-SO-thioperoxolate"),
    ("C=CS[O-]", "eth-1-ene-SO-thioperoxolate"),
]
# (c) (the Blue Book) "The locant '1' is omitted:... (c) in monosubstituted homogeneous monocyclic
# rings", 'cyclohexanethiol (PIN)' (:2917 examples): the ring carries the suffix and nothing else
RING_MONOSUBSTITUTED = [
    ("C1CCCCC1SO", "cyclohexane-SO-thioperoxol"),
    ("C1CCCCC1OS", "cyclohexane-OS-thioperoxol"),
    ("C1CC1OS", "cyclopropane-OS-thioperoxol"),
    ("C1CCCC1SO", "cyclopentane-SO-thioperoxol"),
]
# a second substituent restores the locant,:2869)
RING_DISUBSTITUTED = [
    ("CC1(SO)CCCCC1", "1-methylcyclohexane-1-SO-thioperoxol"),
    ("CC1CCCCC1OS", "2-methylcyclohexane-1-OS-thioperoxol"),
]

# a ring bonded to the one carbon that holds the suffix: the name of the ring with that carbon dropped was
# the name of another molecule (the audit's 'cyclopropane-1-SO-thioperoxol' for C1CC1CSO). The -SOH, -OSH and
# -SSH groups, like -OH and -SH, decorate the skeletal atom they are bonded to, there is no exocyclic-carbon
# suffix form of them, so the CH2 holds the suffix and is the parent: SENIORITY ORDER FOR PARENT
# STRUCTURES (the Blue Book), "The senior parent structure has the maximum number of substituents
# corresponding to the principal characteristic group (suffix)" (:18875); 'cyclohexylmethanol (PIN)' (:6692)
# is the same structure with -OH.
RING_CH2 = [
    ("C1CC1CSO", "cyclopropylmethane-SO-thioperoxol"),
    ("C1CCC(CC1)CSO", "cyclohexylmethane-SO-thioperoxol"),
    ("C1CCC(CC1)COS", "cyclohexylmethane-OS-thioperoxol"),
    ("C1C(CC1F)COS", "(3-fluorocyclobutyl)methane-OS-thioperoxol"),
    ("C1CCC(CC1)(COS)F", "(1-fluorocyclohexyl)methane-OS-thioperoxol"),
    ("c1ccccc1CSO", "phenylmethane-SO-thioperoxol"),
]
# the dithio- family, which OPSIN reads itself: the same defect declined the name there
RING_CH2_DITHIO = [
    ("C1CC1CSS", "cyclopropylmethanedithioperoxol"),
    ("c1ccccc1CSS", "phenylmethanedithioperoxol"),
]

# (smiles, the name the default tier gave and the CO audit proved to be the name of another molecule)
WRONG_BEFORE = [
    ('CCC(C)(C1CCC(CC1)C)OS', '4-methylcyclohexane-1-OS-thioperoxol'),
    ('C1C(CC1F)COS', '3-fluorocyclobutane-1-OS-thioperoxol'),
    ('CCCCCC(CCCCCCC(CCCCC)COS)CO[PH+]=S', '9-[(oxophosphanethiyloxy)methyl]-2-pentyltetradecane-1-OS-thioperoxol'),
    ('CC(C)(C1CCC(CC1)OS)C2CCC(CC2)OS', '4-nonylcyclohexane-1-OS-thioperoxol'),
    ('C1CCC(CC1)(COS)F', '1-fluorocyclohexane-1-OS-thioperoxol'),
    ('CC(C1=CC=CC=C1)C(=C(CSO)C2=CC=CC=C2)OC(=C(CSO)C3=CC=CC=C3)C(C)C4=CC=CC=C4', '2,4-diphenylpent-2-ene-1-SO-thioperoxol'),
    ('CC(C)CC(=C(CSO)C1=CC=CC=C1)OC(=C(CSO)C2=CC=CC=C2)CC(C)C', '5-methyl-2-phenylhex-2-ene-1-SO-thioperoxol'),
    ('CCC(OCCCOS)[Si](C)(OC)OC', '3-{[dimethoxy(methyl)(propyl)silyl]oxy}propane-1-OS-thioperoxol'),
    ('CCC(OCCCOS)[Si](OCC)(OCC)OCC', '3-{[triethoxy(propyl)silyl]oxy}propane-1-OS-thioperoxol'),
    ('CCC(OCCCOS)[Si](C)(OCC)OCC', '3-{[diethoxy(methyl)(propyl)silyl]oxy}propane-1-OS-thioperoxol'),
    ('C1C=C(C[C@H]([C@H]1C(F)(F)P)OS)N', '(1R,6S)-3-amino-6-(difluorophosphanylmethyl)cyclohex-3-ene-1-OS-thioperoxol'),
    ('CC(C1=CC(=CC(=C1)F)C(F)(F)P)NCCCSO', '3-({1-[3-(difluorophosphanylmethyl)-5-fluorophenyl]ethyl}amino)propane-1-SO-thioperoxol'),
    ('C1CC2=CC=CC=C2[C@H]1NCCCSO', '3-[(2,3-dihydro-1H-inden-1-yl)amino]propane-1-SO-thioperoxol'),
    ('CC(C1=CC=CC=C1C(F)(F)P)NCCCSO', '3-({1-[2-(difluorophosphanylmethyl)phenyl]ethyl}amino)propane-1-SO-thioperoxol'),
    ('CC[N+]1=CC=CC=C1/C(=C/C2=CC=C(C=C2)NCCSS)/C', '2-(4-decylidenecyclohexa-2,5-dien-1-iminyl)ethane-1-dithioperoxolium'),
    ('C[N+]1=CC=C(C=C1)/C=C/C2=CC=C(C=C2)NCCSS', '2-(4-octylidenecyclohexa-2,5-dien-1-iminyl)ethane-1-dithioperoxolium'),
    ('C[C@@H]1C/C(=C\\CP(=O)(C2=CC=CC=C2)C3=CC=CC=C3)/C[C@H](C1=C)OS', '(1R,3R,5E)-3-methyl-2-methylidene-5-tetradecylidenecyclohexane-1-OS-thioperoxol'),
    ('C1CCC(CC1)CSO', 'cyclohexane-1-SO-thioperoxol'),
    ('C1CC(CCC1C2=C3C=CC(=C(C4=CC=C(N4)C(=C5C=CC(=N5)C(=C6C=CC2=N6)C7CCC(CC7)SO)C8CCC(CC8)SO)C9CCC(CC9)SO)N3)SO', '4-octatriacontylcyclohexane-1-SO-thioperoxol'),
    ('CCCCCCCCC(CCCSO)C(CC)(OCCC)OCCC', '4-(1,1-dipropoxypropyl)-4,4-dipropoxydodecane-1-SO-thioperoxol'),
    ('CCCCCCCCC(CCCOS)C(OC)(OC)OC', '4,4,4-trimethoxy-4-(trimethoxymethyl)dodecane-1-OS-thioperoxol'),
    ('CCCCCCCCC(CCSO)C(C)(OCCC)OCCC', '3-(1,1-dipropoxyethyl)-3,3-dipropoxyundecane-1-SO-thioperoxol'),
    ('C(C(OS)OS)N', '2-aminoethane-1-OS-thioperoxol'),
    ('CCCCCCCCC(CCCSO)C(OC)(OC)OC', '4,4,4-trimethoxy-4-(trimethoxymethyl)dodecane-1-SO-thioperoxol'),
    ('C=COCC(CCCCSO)CC1=CC=CC=C1', '5-benzyl-6-ethoxyhexane-1-SO-thioperoxol'),
    ('CN(C)C1CC1COS', '2-aminocyclopropane-1-OS-thioperoxol'),
    ('CCC1=CC=C(C=C1)C2=CC=C(C=C2)CCSO', "2-([1,1'-biphenyl]-4-yl)ethane-1-SO-thioperoxol"),
    ('CO[Si](C)(CCCOS)OC', '3-(dimethoxymethylsilyl)propane-1-OS-thioperoxol'),
    ('C1=CC(=CC=C1C=C(OS)OS)[N+](=O)[O-]', '2-(4-nitrophenyl)eth-1-ene-1-OS-thioperoxol'),
    ('CC1=CN(C2=C1C=CC(=C2)C(C)C)CCCOS', '3-(3-methyl-6-propyl-1H-indol-1-yl)propane-1-OS-thioperoxol'),
    ('CC(C)(C)C(C)(CN)C(C=N)(C(C)(C)CSS)NC', '4-(aminomethyl)-3-imino-3-methanimidoyl-2,2,4,5,5-pentamethyl-3-(methylamino)hexane-1-dithioperoxol'),
    ('CC(C)C(=NOCSS)C(C)C', '(N-hydroxy-2,4-dimethylpentan-3-iminyl)methanedithioperoxol'),
    ('CC(CNC)CN/C(=C\\[NH3+])/SO', '(1E)-2-amino-1-{[2-methyl-3-(methylamino)propyl]amino}eth-1-ene-1-SO-thioperoxolium'),
    ('CN1C[C@H](C2=C(C1)C(=CC(=C2)Cl)Cl)C3CC=CC(=C3)SO', '3-decylcyclohexa-1,5-diene-1-SO-thioperoxol'),
    ('C/C=C(/C1(CC1C(C)CN)C)\\OS', '2-(1-aminopropan-2-yl)-1-methylcyclopropane-1-OS-thioperoxol'),
    ('C1COCCC1CN=C2C=CC(=CC2=[NH2+])SO', '3-imino-4-{[(oxan-4-yl)methyl]imino}cyclohexa-1,5-diene-1-SO-thioperoxolium'),
    ('CN(C)CC1(CC1)COS', '1-[(dimethylamino)methyl]cyclopropane-1-OS-thioperoxol'),
    ('C1=CC(=CC=C1C=C(OS)OS)OC(F)(F)F', '2-[4-(trifluoromethoxy)phenyl]eth-1-ene-1-OS-thioperoxol'),
    ('COC1=CC=C(C=C1)C=C(OS)OS', '2-(4-methoxyphenyl)eth-1-ene-1-OS-thioperoxol'),
    ('C1=CC(=CC=C1C=C(OS)OS)OC(F)F', '2-[4-(difluoromethoxy)phenyl]eth-1-ene-1-OS-thioperoxol'),
    ('CC(CCC1CC1OC(C)CC2CC2OS)OC3CC3F', '2-(1-{3-[(2-fluorocyclopropyl)oxy]butyl}-2-[(propan-2-yl)oxy]cyclopropyl)cyclopropane-1-OS-thioperoxol'),
    ('C1CCC(CC1)COS', 'cyclohexane-1-OS-thioperoxol'),
    ('CC[C@@H]1C[C@@H](CC1COS)C', '(2R,4S)-2-ethyl-4-methylcyclopentane-1-OS-thioperoxol'),
    ('CC\\1CC(C(C/C1=C/SS)OC)F', '(1Z)-4-fluoro-5-methoxy-2-methylcyclohexane-1-dithioperoxol'),
    ('C1=NC(=C2C(=N1)N(C=N2)CC(COS)OCl)N', '3-(6-amino-9H-purin-9-yl)propane-1-OS-thioperoxol'),
    ('C1CC1CSO', 'cyclopropane-1-SO-thioperoxol'),
    ('COCC1C(CC(O1)N2C=NC3=C(N=C(N=C32)N)CCSO)OC', '2-(2-amino-9H-purin-6-yl)ethane-1-SO-thioperoxol'),
    ('C1=COC(=C1)C(CSSCC(C2=CC=CO2)SS)SS', '3,8-di(furan-2-yl)-7-disulfanediyldecane-3,8-didithioperoxol'),
    ('CC(C)OC1=C(C=C(CC1OS)/C=C/C2=CC=C(C=C2)/C=C/C3=CC(=C(C(=C3)OC)OC(C)C)OC)OC', '5-[(E)-2-{4-[(E)-2-(3,5-dimethoxy-4-propoxyphenyl)ethenyl]phenyl}ethenyl]-3-methoxy-2-[(propan-2-yl)oxy]cyclohexa-2,4-diene-1-OS-thioperoxol'),
    ('N#CSO', '(thiocyanato)methane-SO-thioperoxol'),
    ('OSC#N', '(thiocyanato)methane-SO-thioperoxol'),
    ('[O-]SC#N', '(thiocyanato)methane-SO-thioperoxolate'),
]
# a cation named as the neutral molecule (the carve-out never compared a structure), and the -ium names the
# audit found wrong by their protonation site; no twin reads a thioperoxolium name, so none ships
CATIONS = [
    ("CC[SH+]O", "ethane-SO-thioperoxol"),
    ("CCS[OH2+]", "ethane-SO-thioperoxol"),
]

# (the name, its twin, the rule, the number of groups the suffix cites, whether it is an anion)
TWIN_CASES = [
    ("methane-SO-thioperoxol", "methane-sulfenic acid", "thioperoxol_sulfenic", 1, False),
    ("propane-1-SO-thioperoxol", "propane-1-sulfenic acid", "thioperoxol_sulfenic", 1, False),
    ("(1E)-prop-1-ene-1-SO-thioperoxol", "(1E)-prop-1-ene-1-sulfenic acid", "thioperoxol_sulfenic", 1, False),
    ("methane-SO-thioperoxolate", "methane-sulfenate", "thioperoxol_sulfenic", 1, True),
    ("sodium ethane-SO-thioperoxolate", "sodium ethane-sulfenate", "thioperoxol_sulfenic", 1, True),
    ("methane(SO-thioperoxolate)", "methanesulfenate", "thioperoxol_sulfenic", 1, True),
    ("sodium ethane(SO-thioperoxolate)", "sodium ethanesulfenate", "thioperoxol_sulfenic", 1, True),
    ("ethane-1,2-bis(SO-thioperoxol)", "ethane-1,2-bis(sulfenic acid)", "thioperoxol_sulfenic", 2, False),
    ("propane-1-OS-thioperoxol", "propane-1-peroxol", "thioperoxol_peroxol", 1, False),
    ("2-(sulfanyloxy)ethane-1-OS-thioperoxol", "2-(sulfanyloxy)ethane-1-peroxol", "thioperoxol_peroxol", 1, False),
    ("benzene-1,4-bis(OS-thioperoxol)", "benzene-1,4-bis(peroxol)", "thioperoxol_peroxol", 2, False),
]
# names no twin reads: the dithio- family OPSIN reads itself, the -ium names, the multiplied form glued
# onto the descriptor (a formatting defect,), an anion of -O-S-H (OPSIN reads 'peroxolate' as the
# neutral peroxol), and names that are no thioperoxol
NO_TWIN = [
    "ethanedithioperoxol",
    "2-aminoethane-1-dithioperoxolium",
    "propane-1-SO-thioperoxolium",
    "(1E)-2-amino-1-(methylamino)eth-1-ene-1-SO-thioperoxolium",
    "ethane-1,2-bisSO-thioperoxol",
    "propane-1-OS-thioperoxolate",
    "ethane(OS-thioperoxolate)",
    "ethanol",
    "1-(hydroxysulfanyl)butane",
]


def _independent_reads_back(name, smiles):
    """The CO auditor's check, a spelling the engine does not use: 'X-SO-thioperoxol' read as 'X-thiol'
    and 'X-OS-thioperoxol' as 'X-ol', compared with the input with the oxygen of each -S-OH (the sulfur
    of each -O-SH) removed. True when OPSIN reads the twin to the full InChIKey of that structure."""
    m = re.search(r"(?:(bis|tris|tetrakis)\(|\()?(SO|OS)-thioperoxol(ate)?\)?$", name)
    if m is None:
        return False
    count = {None: 1, "bis": 2, "tris": 3, "tetrakis": 4}[m.group(1)]
    kind = m.group(2)
    suffix = {"SO": "thiol", "OS": "ol"}[kind]   # an anion: the oxygen removed takes the charge with it
    if count > 1:
        suffix = m.group(1) + "(" + suffix + ")"
    twin = name[:m.start()] + suffix
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts("[!#1][SX2][OX2H1,OX1-]" if kind == "SO" else "[!#1][OX2][SX2H1,SX1-]")
    hits = mol.GetSubstructMatches(pattern)
    if len(hits) < count:
        return False
    drop = sorted({h[2] for h in hits}, reverse=True)[:count]
    rw = Chem.RWMol(mol)
    for atom in drop:
        rw.RemoveAtom(atom)
    try:
        edited = rw.GetMol()
        Chem.SanitizeMol(edited)
    except Exception:
        return False
    parsed = _independent_parse(twin, Chem.MolToSmiles(edited))
    if parsed is None or Chem.MolFromSmiles(parsed) is None:
        return False
    return Chem.MolToInchiKey(edited) == Chem.MolToInchiKey(Chem.MolFromSmiles(parsed))


@pytest.mark.parametrize("name,twin,rule,count,anion", TWIN_CASES)
def test_the_twin_is_the_name_with_the_suffix_of_the_same_group(name, twin, rule, count, anion):
    from orthonym.validation.thioperoxol_twin import thioperoxol_twin
    found = thioperoxol_twin(name)
    assert found is not None, name
    assert (found.name, found.rule, found.count, found.anion) == (twin, rule, count, anion)


@pytest.mark.parametrize("name", NO_TWIN)
def test_no_twin_for_a_name_that_is_not_a_well_formed_thioperoxol(name):
    from orthonym.validation.thioperoxol_twin import thioperoxol_twin
    assert thioperoxol_twin(name) is None


@pytest.mark.parametrize("smiles,name", RIGHT + RIGHT_ANIONS[:1] + ANIONS_BELOW_PIN[:2] + RING_MONOSUBSTITUTED)
def test_the_independent_check_reads_the_right_names(smiles, name):
    """The CO auditor's spelling reads these names to the input: they are right, so the twin must keep them."""
    assert _independent_reads_back(name, smiles)


@pytest.mark.parametrize("smiles,name", RIGHT + RIGHT_ANIONS + RING_MONOSUBSTITUTED + RING_DISUBSTITUTED)
def test_default_tier_ships_a_right_name_through_its_twin(smiles, name):
    row = default_tier_row(smiles)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified" and row["is_pin"] is True, row
    assert row["verified"] == "opsin_twin" and row["opsin"] == "verified_twin", row
    assert re.fullmatch(r"twin_verified:thioperoxol_(sulfenic|peroxol)", row["gate_outcome"]), row
    assert row["spelling_failures"] == [], row


@pytest.mark.parametrize("smiles,name", RIGHT + RIGHT_ANIONS + RING_MONOSUBSTITUTED + RING_DISUBSTITUTED)
def test_best_effort_names_every_row_the_default_tier_names(smiles, name):
    """The default tier named these rows; best-effort gave the systematic name of the neutral ones and no
    name at all for the anions. The twin-verified Preferred IUPAC Name ships at both tiers."""
    row = name_best_effort(smiles)
    assert row["name"] == name, row
    assert row["verified"] == "opsin_twin", row
    assert re.fullmatch(r"twin_verified:thioperoxol_(sulfenic|peroxol)", row["gate_outcome"]), row


@pytest.mark.parametrize("smiles,name", ANIONS_BELOW_PIN)
def test_an_anion_whose_spelling_no_book_row_certifies_is_not_a_pin(smiles, name):
    """The twin proves the structure; the spelling decides the label. The strict path builds the name and a verified
    twin reads it back (pin_unverified); the default tier declines it, the wider tier names it."""
    import tests.support.default_tier as dt
    strict = dt.strict_path_row(smiles)
    assert strict["name"] == name, strict
    assert strict["tier"] == "pin_unverified" and strict["is_pin"] is False, strict
    assert strict["verified"] == "opsin_twin", strict
    dt.assert_default_tier_declines(smiles)
    best = name_best_effort(smiles)
    assert best["name"] == name and best["tier"] == "pin_unverified" and best["verified"] == "opsin_twin", best


@pytest.mark.parametrize("name", ["cyclohexane-1-SO-thioperoxol", "cyclopropane-1-OS-thioperoxol",
                                  "methane-SO-thioperoxolate", "sodium ethane-SO-thioperoxolate"])
def test_the_spelling_check_reads_a_thioperoxol_name_that_breaks_the_book(name):
    """ (c) (:2913) and (:41035), read lexically (validation.pin_spelling)."""
    from orthonym.validation.pin_spelling import check_pin_spelling
    failures = check_pin_spelling(Chem.MolFromSmiles("CC"), name)
    assert any(f.rule.startswith(("P-14.3.4", "P-72.2.2.2.2")) and "thioperoxol" in f.detail for f in failures), (name, failures)


@pytest.mark.parametrize("name", ["cyclohexane-SO-thioperoxol", "1-methylcyclohexane-1-SO-thioperoxol",
                                  "methane(SO-thioperoxolate)", "sodium ethane(SO-thioperoxolate)",
                                  "propane-1-SO-thioperoxolate", "2-chloroethane-1-SO-thioperoxol"])
def test_the_spelling_check_keeps_the_spellings_of_the_book(name):
    from orthonym.validation.pin_spelling import check_pin_spelling
    failures = check_pin_spelling(Chem.MolFromSmiles("CC"), name)
    assert not any(f.rule.startswith(("P-14.3.4", "P-72.2.2.2.2")) and "thioperoxol" in f.detail for f in failures), failures


# M1: -OS-thioperoxol is 'peroxol' with its second oxygen replaced by sulfur,:17460); an input that also holds
# a -OOH group reads to the same peroxol structure whichever of the two groups the name puts the suffix on
SWAPPED = [
    ("1-(hydroperoxy)propane-2-OS-thioperoxol", "CC(OO)COS"),
    ("3-(hydroperoxy)butane-1-OS-thioperoxol", "CC(OS)CCOO"),
]


@pytest.mark.parametrize("name,smiles", SWAPPED)
def test_the_peroxol_twin_does_not_pass_a_suffix_and_prefix_swap(name, smiles):
    from orthonym.validation.thioperoxol_twin import thioperoxol_twin_round_trip
    assert thioperoxol_twin_round_trip(name, smiles) is None


@pytest.mark.parametrize("smiles", ["CC(OO)COS", "CC(OS)CCOO"])
def test_a_structure_with_both_groups_is_not_shipped_wrong(smiles):
    """The engine names these by the peroxol suffix (OPSIN reads it); no thioperoxol name ships for them."""
    for row in (default_tier_row(smiles), name_best_effort(smiles)):
        assert "thioperoxol" not in (row["name"] or "") or _independent_reads_back(row["name"], smiles), row


def test_the_O_S_H_group_is_one_on_a_carbon():
    """The suffix is cited for a group on a carbon: a sulfur on an oxygen on silicon is no thioperoxol."""
    from orthonym.validation.thioperoxol_twin import _peroxol_references
    assert list(_peroxol_references("C[Si](C)(C)OS", 1)) == []
    assert list(_peroxol_references("CCOS", 1)) == ["CCOO"]


# S1: the anion twin ('sulfenate') reads a name of an ester of the group as well: 'methyl ethane-sulfenate' is
# CCSOC. Mixed chalcogen analogues of peroxides (:27900): 'CH3-CH2-OS-CH3 (1) [(methylsulfanyl)oxy]ethane
# (PIN)... [not ethyl methanesulfenate]' (:27912): an ester is no anion. An anion twin needs an [S-O-] in the input.
def test_the_anion_twin_needs_an_anion_in_the_input():
    from orthonym.validation.thioperoxol_twin import thioperoxol_twin_round_trip
    assert thioperoxol_twin_round_trip("methyl ethane-SO-thioperoxolate", "CCSOC") is None
    assert thioperoxol_twin_round_trip("methyl ethane(SO-thioperoxolate)", "CCSOC") is None
    assert thioperoxol_twin_round_trip("ethane(SO-thioperoxolate)", "CCS[O-]") == "thioperoxol_sulfenic"
    assert thioperoxol_twin_round_trip("sodium ethane(SO-thioperoxolate)", "[Na+].CCS[O-]") == "thioperoxol_sulfenic"


@pytest.mark.parametrize("smiles,wrong", WRONG_BEFORE)
def test_the_independent_check_does_not_read_the_wrong_names(smiles, wrong):
    """The 52 audit rows: the name of another molecule (or one the check cannot read at all)."""
    assert not _independent_reads_back(wrong, smiles)


@pytest.mark.parametrize("smiles,wrong", WRONG_BEFORE)
def test_the_default_tier_does_not_ship_the_wrong_name(smiles, wrong):
    row = default_tier_row(smiles)
    assert row["name"] != wrong, row
    if row["tier"] != "abstain":
        if row["gate_outcome"].startswith("twin_verified:thioperoxol_"):
            assert _independent_reads_back(row["name"], smiles), row
        else:
            assert "-SO-thioperoxol" not in row["name"] and "-OS-thioperoxol" not in row["name"], row


@pytest.mark.parametrize("smiles,wrong", WRONG_BEFORE)
def test_best_effort_does_not_ship_the_wrong_name(smiles, wrong):
    row = name_best_effort(smiles)
    assert row["name"] != wrong, row
    if row["tier"] == "abstain":
        return
    if row["gate_outcome"].startswith("twin_verified:thioperoxol_"):
        assert _independent_reads_back(row["name"], smiles), row
    else:
        assert_full_rt(row["name"], smiles)    # a name OPSIN reads itself ('...dithioperoxol')


@pytest.mark.parametrize("smiles,wrong", CATIONS)
def test_a_cation_is_not_named_as_the_neutral_molecule(smiles, wrong):
    assert default_tier_row(smiles)["name"] != wrong
    assert name_best_effort(smiles)["name"] != wrong


@pytest.mark.parametrize("smiles,name", RING_CH2)
def test_the_carbon_between_a_ring_and_the_suffix_is_the_parent(smiles, name):
    row = default_tier_row(smiles)
    assert row["name"] == name, row
    assert row["tier"] == "pin_verified" and row["verified"] == "opsin_twin", row
    assert _independent_reads_back(name, smiles)
    best = name_best_effort(smiles)
    assert best["name"] == name and best["verified"] == "opsin_twin", best


@pytest.mark.parametrize("smiles,name", RING_CH2_DITHIO)
def test_the_dithioperoxol_of_a_ring_carbon_is_named(smiles, name):
    row = default_tier_row(smiles)
    assert row["name"] == name and row["tier"] == "pin_verified" and row["verified"] == "opsin", row
    assert_full_rt(name, smiles)


# a component of a mixture: the adduct name is read by OPSIN as a whole, so a component name OPSIN cannot read
# (the twin reads a whole name only) cannot be used; the general engine names the component as before
ADDUCTS = [
    ("CSO.CCO", "ethanol\u2014(hydroxysulfanyl)methane (1/1)"),
    ("CCCSO.c1ccccc1", "1-(hydroxysulfanyl)propane\u2014benzene (1/1)"),
]


@pytest.mark.parametrize("smiles,name", ADDUCTS)
def test_a_mixture_with_a_thioperoxol_component_is_still_named_at_best_effort(smiles, name):
    """ adducts (the Blue Book): each component is named alone. The best-effort tier named these
    mixtures before the twin shipped 'methane-SO-thioperoxol' at that tier; the component name no OPSIN reads
    would drop the whole mixture."""
    row = name_best_effort(smiles)
    assert row["name"] == name, row
    assert_full_rt(name, smiles)


# an isotopically labelled structure: the isotope decorator names the unlabelled skeleton and reads every decorated
# candidate with OPSIN, the Blue Book), which reads no thioperoxol suffix: it names the skeleton as before
ISOTOPES = [
    ("[3H]COS", "(sulfanyloxy)(3H1)methane"),
    ("[2H]C([2H])([2H])SO", "(hydroxysulfanyl)(2H3)methane"),
    ("[2H]OSC", "[(2H1)hydroxysulfanyl]methane"),
]


@pytest.mark.parametrize("smiles,name", ISOTOPES)
def test_an_isotopically_labelled_thioperoxol_is_still_named_at_best_effort(smiles, name):
    row = name_best_effort(smiles)
    assert row["name"] == name, row
    assert_full_rt(name, smiles)


def test_the_gate_ships_a_right_name_and_withholds_a_wrong_one():
    """The decision site: the same name for two structures. ``cyclopropane-1-SO-thioperoxol`` is the name
    of ``C1CC1SO`` and not of ``C1CC1CSO`` (the CH2 between the ring and the suffix)."""
    from orthonym.namer import _final_opsin_validity_gate
    from orthonym.validation.thioperoxol_twin import thioperoxol_twin_round_trip
    name = "cyclopropane-1-SO-thioperoxol"
    assert thioperoxol_twin_round_trip(name, "C1CC1SO") == "thioperoxol_sulfenic"
    assert thioperoxol_twin_round_trip(name, "C1CC1CSO") is None
    assert _final_opsin_validity_gate(name, "C1CC1SO") == name
    assert _final_opsin_validity_gate(name, "C1CC1CSO") != name


def test_the_peroxol_twin_names_the_group_that_carries_the_sulfur():
    """-OS-thioperoxol is peroxol with its second oxygen replaced by sulfur: the twin marks WHICH oxygen
    group carries the sulfur. The alcohol twin cannot: a hydroxy prefix and the suffix exchanged read to one
    diol ('2-hydroxypropane-1-OS-thioperoxol' is the name of CC(O)COS, not of CC(CO)OS)."""
    from orthonym.validation.thioperoxol_twin import thioperoxol_twin_round_trip
    name = "2-hydroxypropane-1-OS-thioperoxol"
    assert thioperoxol_twin_round_trip(name, "CC(O)COS") == "thioperoxol_peroxol"
    assert thioperoxol_twin_round_trip(name, "CC(CO)OS") is None
    assert _independent_reads_back(name, "CC(O)COS") and _independent_reads_back(name, "CC(CO)OS")  # its blind spot


def test_a_name_with_a_second_group_it_does_not_cite_is_not_read_back():
    """'2-aminoethane-1-OS-thioperoxol' for C(C(OS)OS)N: two -O-SH groups, one cited."""
    from orthonym.validation.thioperoxol_twin import thioperoxol_twin_round_trip
    assert thioperoxol_twin_round_trip("2-aminoethane-1-OS-thioperoxol", "C(C(OS)OS)N") is None
    assert thioperoxol_twin_round_trip("2-aminoethane-1-OS-thioperoxol", "C(COS)N") == "thioperoxol_peroxol"


def test_a_thioperoxol_name_does_not_ship_while_opsin_cannot_be_reached(monkeypatch):
    """A name OPSIN has no grammar for ships on its twin; with no OPSIN to read the twin nothing proves it, so
    it is withheld (the carve-out shipped it on its construction whatever OPSIN did)."""
    import orthonym.namer as namer
    monkeypatch.setattr(namer, "_validity_gate_status", lambda name: "unavailable")
    monkeypatch.setattr(namer, "_validity_gate_name_to_smiles", lambda name: None)
    assert namer._final_opsin_validity_gate("methane-SO-thioperoxol", "CSO") != "methane-SO-thioperoxol"
