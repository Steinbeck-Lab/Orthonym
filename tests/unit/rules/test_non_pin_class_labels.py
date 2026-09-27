"""Known non-PIN name classes ship unchanged but are never labelled pin_verified.

Each class below is a valid, round-trip-verified name whose PIN the engine does not
build yet. The name keeps shipping at the PIN tier (breadth holds), and the label says
what it is: a name-scoped provenance record (``record_non_pin_fragment``) moves it off
``pin_verified``. Controls in the same families stay ``pin_verified``.

- Von Baeyer names of fusion-nameable systems. "Five-membered ring
  requirement" (the Blue Book,:23710): "Fusion nomenclature gives preferred IUPAC
  names only to compounds having at least two rings of at least five or more members.
  [...] When fusion names are not allowed, unsaturated von Baeyer ring system names are
  preferred IUPAC names." (:19532) ranks fused and bridged fused systems above
  von Baeyer systems;:23883 "the bridged fused ring name is preferred to the von Baeyer
  name"; 'decahydronaphthalene (PIN) bicyclo[4.4.0]decane' (:24233).
- Von Baeyer names of the retained parents. (:9879,:9881): "The retained names
  adamantane and cubane are used in general nomenclature and as preferred IUPAC names".
- Retained peptide names. Controller ruling (TRIAGE.md 'Controller rulings'): "D-a does
  NOT extend to peptides... the PIN is the substitutive name"; (:50943).
- NAME_EXACT stereoparents shipped through the gate's carve-out ('germacrane'): OPSIN
  cannot parse them, so they are unverified, and (:50943) identifies no PIN.
- A stereoparent acid written as hydroxy + oxo ('26-hydroxy...-26-one'): (:18158;
  acids 7a:18172 before alcohols 17:18190) makes the acid the suffix.
- Whole-graph R/S on a steroid stereoparent: (:51045,:51047) "The name of a
  fundamental parent structure usually implies the absolute configuration of all
  chirality centers"; the configurations at C-8, 9, 10, 13 and 14 are implied, and
  'α/β' is the preferred way to state the rest:51053).
- Esters cited as acyloxy prefixes with a junior suffix (the path): esters (class
  9,:18182) outrank alcohols (class 17,:18190); (:31696,:31698).
- 'quinolizidine' / 'pyrrolizidine' / 'indolizidine' (0 Blue Book hits): the parents
  are quinolizine (:11582-11584), pyrrolizine (:11628) and indolizine (:11622), and
  saturated compounds take 'hydro' prefixes,:24221).
- A bare '(R)-'/'(S)-' on a substituent whose name carries locants:
  (:44624) "In preferred IUPAC names, stereodescriptors, preceded by a locant, must be
  cited"; only a group with no locants drops it (:45031); '[(1R)-1-chloropropyl]benzene
  (PIN)' (:44668).
"""
import re

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.jvm_budget import jvm_slots
from orthonym.rules.polycyclic import (
    fusion_nomenclature_applies,
    retained_von_baeyer_parent,
)
from tests.support.rt_assert import name_is_rt_exact


def _row(smiles):
    with jvm_slots(1, purpose="test-non-pin-class-labels"):
        return Orthonym(style="pin").name_tiered(smiles)


_DEMOTED = [
    # von Baeyer names of fusion-nameable systems (canary DK-VBFUSED; TRIAGE rows 89/90)
    ("CC1(C)C(O)C(O)CC2(C)C1CCC13CC(CCC21)C1(C)OC31",
     "5,5,9,14-tetramethyl-15-oxapentacyclo[11.3.1.0^1,10.0^4,9.0^14,16]heptadecane-6,7-diol"),
    ("CC1(C)CC=C[C@]2(C)OO[C@@H]3C[C@@]12CC[C@H]3O",
     "(1R,4S,9S,12R)-4,8,8-trimethyl-2,3-dioxatricyclo[7.3.1.0^4,9]tridec-5-en-12-ol"),
    # wp7 change-asserted-value: (:3448) 'dimethyl' keys at 'methyl', before 'oxo';
    # the primed locants are locants (:3442). Was "...-5,6'-dioxo-9',13'-dimethylspiro[...]".
    # OPSIN 2.9.0 full-InChIKey exact.
    ("CC12CCC(=O)C=C1C=CC1[C@@H]2CCC2(C)[C@H]1CCC21CCC(=O)O1",
     "(10'S,17'S)-9',13'-dimethyl-5,6'-dioxospiro[oxolane-2,14'-tetracyclo[8.7.0.0^4,9."
     "0^13,17]heptadeca-2,4-diene]"),
    # von Baeyer names of adamantane / cubane (canary DK-ADAM)
    ("OC1C2CC3CC1CC(O)(C3)C2", "tricyclo[3.3.1.1^3,7]decane-1,4-diol"),
    ("NC12CC3CC(CC(C3)C1)C2", "tricyclo[3.3.1.1^3,7]decan-1-amine"),
    ("OC(=O)C12C3C4C1C5C2C3C45", "pentacyclo[4.2.0.0^2,5.0^3,8.0^4,7]octane-1-carboxylic acid"),
    # retained peptide names, also inside an acyl prefix (canary DK-PEP)
    # (Pro-Ala-Ala left this list in j7: the tripeptide substitutive PIN is built,
    # '(2S)-2-{(2S)-2-[(2S)-pyrrolidine-2-carboxamido]propanamido}propanoic acid',
    # pinned in tests/integration/test_stereo_benchmark.py)
    ("N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CC(=O)O)C(=O)O",
     "(2S)-2-(phenylalanylglutamylamino)butanedioic acid"),
    # (stereoparent acid as hydroxy + oxo, canary DK-ACIDPFX: left this list in j7 --
    # the steroid producer cites the acid as the '-26-oic acid' suffix,
    # '(25S)-3-oxo-5α-cholestan-26-oic acid', the Blue Book), kept
    # at the PIN tier by the controller ruling on names; pinned in
    # tests/unit/rules/test_j7_defects_misc.py)
    # whole-graph R/S on a steroid stereoparent (canary DK-P101CIP)
    ("C[C@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(Cl)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO",
     "(8S,9R,10S,11S,13S,14S,16S,17R)-9-chloro-11,17,21-trihydroxy-16-methylpregna-1,4-"
     "diene-3,20-dione"),
    #: acyloxy prefixes on an alcohol (canary DK-GLYSUB)
    ("CCCCCCCC/C=C\\CCCCCCCC(=O)O[C@H](CO)COC(=O)CCCCCCCCCCCCCCCCC",
     "(2R)-1-(octadecanoyloxy)-2-[(9Z)-octadec-9-enoyloxy]propan-3-ol"),
    # Suite fix j4 (TRIAGE g3 C10b): polyol esters cited as acyloxy prefixes on the
    # bare hydride -- the decomposition weave (glycerol diester ether) and the
    # rules.esters.name_polyol_polyester fallback. (:31698);
    # (:31836) "Method (1) generates preferred IUPAC names".
    ("CCCCC/C=C\\C/C=C\\CCCCCCCC(=O)O[C@H](COCCCCCCCCCCCCCCCCCC)COC(=O)"
     "CCCCCCCCCCCCCCCCCCCCCCC",
     "(2R)-2-{[(9Z,12Z)-octadeca-9,12-dienoyl]oxy}-1-(octadecyloxy)-3-"
     "(tetracosanoyloxy)propane"),
    ("CCOCC(COC(C)=O)OC(C)=O", "1,2-bis(acetyloxy)-3-ethoxypropane"),
    ("CC(=O)OCC(C)OC(=O)CC", "1-(acetyloxy)-2-(propanoyloxy)propane"),
    # catalog names with no Blue Book standing
    ("C1CCN2CCCCC2C1", "quinolizidine"),
    ("C1CC2CCCN2C1", "pyrrolizidine"),
    ("C1CCN2CCCC2C1", "indolizidine"),
    ("OCC1CCCN2CCCCC12", "(quinolizidin-1-yl)methanol"),
    # (rest-of-suite s16, '4-[(S)-2-amino-2-carboxyethoxy]-4-oxobutanoic acid', left this
    # list in decision A part 2, 2026-09-27: the substituent's own numbering now reaches
    # the stereo emitter and the row ships its PIN '4-[(2S)-2-amino-2-carboxyethoxy]-4-
    # oxobutanoic acid' at pin_verified -- pinned in tests/integration/test_stereo_benchmark.py)
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", _DEMOTED)
def test_known_non_pin_class_ships_demoted(smiles, expected):
    row = _row(smiles)
    assert row["name"] == expected
    assert row["tier"] not in ("pin_verified", "abstain") and not row["is_pin"]
    assert name_is_rt_exact(expected, smiles)


@pytest.mark.opsin_gate
def test_carveout_stereoparent_is_not_labelled_verified():
    # 'germacrane' ships through the np_stereoparent carve-out: OPSIN cannot parse it.
    row = _row("CC(C)[C@@H]1CC[C@H](C)CCC[C@H](C)CC1")
    assert row["name"] == "germacrane"
    assert row["opsin"] == "unverified"
    assert row["tier"] != "pin_verified" and not row["is_pin"]


_CONTROLS = [
    ("C1CC2CCC1C2", "bicyclo[2.2.1]heptane"),                    # BB:2038
    ("C1CC2CC1C1CC21", "tricyclo[3.2.1.0^2,4]octane"),           # cf. BB:48763
    ("C12C3C1C1C2C31", "tetracyclo[2.2.0.0^2,6.0^3,5]hexane"),   # BB:9897
    ("CC1(C)C2CCC1(C)C(=O)C2", "1,7,7-trimethylbicyclo[2.2.1]heptan-2-one"),
    ("C1CCC2CCCCC2C1", "decahydronaphthalene"),                   # BB:24233
    ("C1CCC2CCCC2C1", "octahydro-1H-indene"),
    ("C12CC3CC(C1)CC(C3)C2", "adamantane"),                       # BB:9885
    ("C12C3C4C1C5C2C3C45", "cubane"),                             # BB:9889
    ("C[C@H](N)C(=O)N[C@@H](C)C(=O)O", "(2S)-2-[(2S)-2-aminopropanamido]propanoic acid"),
    ("C[C@]12CC[C@H]3[C@@H](CC[C@@H]4C[C@@H](O)CC[C@@]43C)[C@@H]1CC[C@@H]2O",
     "5β-androstane-3β,17β-diol"),
    ("CC(=O)OC[C@@H](O)COC(C)=O", "2-hydroxypropane-1,3-diyl diacetate"),
    ("CC(=O)OCC(COC(C)=O)OC(C)=O", "propane-1,2,3-triyl triacetate"),  # BB:31827
    ("CC[C@@H](Cl)c1ccccc1", "[(1R)-1-chloropropyl]benzene"),       # BB:44668
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", _CONTROLS)
def test_controls_stay_pin_verified(smiles, expected):
    row = _row(smiles)
    assert row["name"] == expected and row["tier"] == "pin_verified" and row["is_pin"]


# ---------------------------------------------------------------------------------
# The ring criterion itself, on skeletons rebuilt from von Baeyer descriptors
#: main ring, main bridge, then each secondary bridge 'n^i,j').
# ---------------------------------------------------------------------------------

def _skeleton(descriptor):
    parts = re.fullmatch(r"\w*?cyclo\[(.*)\]", descriptor).group(1).split(".")
    a, b, c = (int(x) for x in parts[:3])
    bonds = [(k, k + 1) for k in range(1, a + b + 2)] + [(a + b + 2, 1)]
    top, prev = a + b + 2, 1
    for _ in range(c):
        top += 1
        bonds.append((prev, top))
        prev = top
    bonds.append((prev, a + 2))
    for sec in parts[3:]:
        length, i, j = (int(x) for x in re.fullmatch(r"(\d+)\^(\d+),(\d+)", sec).groups())
        prev = i
        for _ in range(length):
            top += 1
            bonds.append((prev, top))
            prev = top
        bonds.append((prev, j))
    rw = Chem.RWMol()
    for _ in range(top):
        rw.AddAtom(Chem.Atom(6))
    for x, y in bonds:
        rw.AddBond(x - 1, y - 1, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


@pytest.mark.parametrize("descriptor,expected", [
    ("bicyclo[4.4.0]", True),             # decahydronaphthalene (PIN), BB:24233
    ("bicyclo[3.3.0]", True),             # octahydropentalene
    ("bicyclo[12.4.0]", True),            # dodecahydrobenzo[14]annulene (PIN), BB:23887
    ("tricyclo[12.3.1.0^5,10]", True),    # methanobenzo[13]annulene (PIN), BB:23875-23879
    ("tricyclo[5.2.1.0^2,6]", True),      # hexahydro-1H-4,7-methanoindene (PIN), BB:49311
    ("tricyclo[4.4.2.0^2,7]", True),      # 4,8-ethanopyrano[4,3-c]pyran (PIN), BB:32535
    ("bicyclo[4.1.0]", False),            # BB:23718 (one ring of five or more)
    ("bicyclo[4.2.0]", False),            # BB:23725
    ("bicyclo[3.2.0]", False),
    ("bicyclo[2.2.1]", False),
    ("tricyclo[3.3.1.1^3,7]", False),     # adamantane skeleton (bridged, not fused)
    ("tetracyclo[3.2.0.0^2,7.0^4,6]", False),   # BB:10824 (the 5-rings share 3 atoms)
    ("pentacyclo[4.2.0.0^2,5.0^3,8.0^4,7]", False),  # cubane, BB:9889
    ("tetracyclo[2.2.0.0^2,6.0^3,5]", False),        # BB:9897
    ("tricyclo[3.2.1.0^2,4]", False),                # BB:48763
])
def test_fusion_nomenclature_applies_on_bb_skeletons(descriptor, expected):
    mol = _skeleton(descriptor)
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms()}
    assert fusion_nomenclature_applies(mol, ring_atoms) is expected


def test_retained_von_baeyer_parent_is_all_carbon_only():
    ada = Chem.MolFromSmiles("C12CC3CC(C1)CC(C3)C2")
    assert retained_von_baeyer_parent(ada, set(range(ada.GetNumAtoms()))) == "adamantane"
    cub = Chem.MolFromSmiles("C12C3C4C1C5C2C3C45")
    assert retained_von_baeyer_parent(cub, set(range(cub.GetNumAtoms()))) == "cubane"
    aza = Chem.MolFromSmiles("C1C2CC3CC1NC(C2)C3")
    assert retained_von_baeyer_parent(aza, set(range(aza.GetNumAtoms()))) is None
