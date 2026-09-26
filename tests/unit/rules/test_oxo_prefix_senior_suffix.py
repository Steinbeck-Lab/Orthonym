"""A ring C=O beside a senior suffix group is an 'oxo' PREFIX on the indicated-hydrogen /
hydro parent -- never a second suffix glued on.

 fix a performance pass (wp4-ring-suffix), research group malformed-chromone (items 1-7) and
the open item '9-oxo-9H-xanthene acids/esters'. The fused assembler
(``fused_rings._assemble_fused_heterocycle_name``) glued '-4-one' and '-2-carboxylic
acid' into 'benzo[b]pyran-4-one2-carboxylic acid' at pin_verified (OPSIN parses it only
leniently; the NH quinolone form parsed to a different tautomer), put a senior suffix on
a ketone catalog core ('xanthone-2-carboxylic acid', unparseable), let an amine or a
ring -one take the suffix over a senior amide/nitrile/acid, and cited carboxamides and
nitriles on rings only as 'carbamoyl'/'cyano' prefixes. The monocyclic producer cited
'4-oxopyran-2-carboxylic acid' (no indicated hydrogen) and the polycyclic producer
dropped the 9-oxo of fluorenone acids.

Every expected name is an OPSIN 2.9.0 round trip to the input's full standard InChIKey,
checked here outside the engine (``tests/support/rt_assert.py``).

Rules (the Blue Book): "General compound classes listed in decreasing
order of seniority" (:18162): 7 acids (:18172) > 11 amides (:18184) > 14 nitriles
(:18187) > 15 aldehydes (:18188) > 16 "Ketones..., pseudoketones" (:18189) > 17 hydroxy
compounds (:18190) > 19 amines (:18192). "Prefix nomenclature" (:24862):
"After the introduction of indicated and 'added indicated hydrogen' atoms, all
substituent groups not expressed as suffixes are cited as prefixes" (:24864).
 (:24639): "in preferred IUPAC names indicated hydrogen must always be cited
when present in the corresponding structure". "NUMBERING" (:3219): "(b)
indicated hydrogen" (:3246, '2H-pyran-6-carboxylic acid (PIN)':3252) before "(c)
principal characteristic groups" before "(e) saturation" (hydro prefixes) before "(f)
detachable alphabetized prefixes"; '3,4-dihydro-2H-1-benzopyran (PIN) (not 2,3-dihydro-
4H-1-benzopyran' (:24673); '2,3-dihydro-1H-inden-2-yl (preferred prefix)' (:17374).
(PIN) examples of the pattern: '9,10-dioxo-9,10-dihydroanthracene-2-carboxylic acid'
(:29471), '5,8-dioxo-5,6,7,8-tetrahydronaphthalene-2-carboxylic acid' (:24890),
'5-oxo-2,5-dihydrofuran-2-carboxylic acid' (:29276), '2,2-dimethyl-1,3-dioxo-2,3-
dihydro-1H-isoindol-2-ium' (:41447). Parent spellings: '2H-1-benzopyran' is the PIN,
not chromene (Table 2.8 row (23),:11656); benzo names, not 'benzo[b]pyran',
'3-benzoxepine (PIN) benzo[d]oxepine':11821); '9H-xanthene' (Table 2.8 row (22));
naphthyridine isomers (Table 2.8 row (11),:11561-11563). Suffixes:
(:32669) "The suffix 'carboxamide' is always used to name amides with the -CO-NH2 group
attached to a ring"; (:34720) "The suffix 'carbonitrile' is always used to
name nitriles having the -CN group attached to a ring or ring system".
"""
import re

import pytest

from orthonym import Orthonym, name_compound
from tests.support.rt_assert import name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


PIN_ROWS = [
    # item 1 (sub-case A: an oxo substituent on the core; was '<core>-4-one2-carboxylic acid')
    ("OC(=O)c1cc(=O)c2ccccc2o1", "4-oxo-4H-1-benzopyran-2-carboxylic acid"),
    ("OC(=O)c1coc2ccccc2c1=O", "4-oxo-4H-1-benzopyran-3-carboxylic acid"),
    ("CCOC(=O)c1cc(=O)c2ccccc2o1", "ethyl 4-oxo-4H-1-benzopyran-2-carboxylate"),
    ("O=Cc1coc2ccccc2c1=O", "4-oxo-4H-1-benzopyran-3-carbaldehyde"),
    ("OC(=O)c1cc2ccccc2c(=O)o1", "1-oxo-1H-2-benzopyran-3-carboxylic acid"),
    ("CCn1cc(C(=O)O)c(=O)c2ccccc21", "1-ethyl-4-oxo-1,4-dihydroquinoline-3-carboxylic acid"),
    ("CCn1cc(C(=O)O)c(=O)c2ccc(C)nc21",
     "1-ethyl-7-methyl-4-oxo-1,4-dihydro-1,8-naphthyridine-3-carboxylic acid"),
    ("OC(=O)C1COc2ccccc2C1=O", "4-oxo-3,4-dihydro-2H-1-benzopyran-3-carboxylic acid"),
    ("OC(=O)C1Cc2ccccc2C1=O", "1-oxo-2,3-dihydro-1H-indene-2-carboxylic acid"),
    ("OC(=O)c1ccc2C(=O)N(C)C(=O)c2c1",
     "2-methyl-1,3-dioxo-2,3-dihydro-1H-isoindole-5-carboxylic acid"),
    ("COc1cc(O)c2c(c1C)[C@](C)(C(=O)O)OC2=O",
     "(1R)-4-hydroxy-6-methoxy-1,7-dimethyl-3-oxo-1,3-dihydro-2-benzofuran-1-carboxylic acid"),
    # the NH quinolone: the glued name parsed to a different tautomer (abstain before)
    ("OC(=O)c1c[nH]c2ccccc2c1=O", "4-oxo-1,4-dihydroquinoline-3-carboxylic acid"),
    # item 2 (sub-case B: a ketone catalog core; open item for the xanthene rows)
    ("OC(=O)c1cc2ccccc2oc1=O", "2-oxo-2H-1-benzopyran-3-carboxylic acid"),
    ("OC(=O)c1cc(=O)oc2ccccc12", "2-oxo-2H-1-benzopyran-4-carboxylic acid"),
    ("CCOC(=O)c1cc2ccccc2oc1=O", "ethyl 2-oxo-2H-1-benzopyran-3-carboxylate"),
    ("OC(=O)c1ccc2[nH]c(=O)c3ccccc3c2c1", "6-oxo-5,6-dihydrophenanthridine-2-carboxylic acid"),
    ("OC(=O)c1ccc2oc3ccccc3c(=O)c2c1", "9-oxo-9H-xanthene-2-carboxylic acid"),
    ("COC(=O)c1ccc2oc3ccccc3c(=O)c2c1", "methyl 9-oxo-9H-xanthene-2-carboxylate"),
    ("OC(=O)c1ccc2sc3ccccc3c(=O)c2c1", "9-oxo-9H-thioxanthene-2-carboxylic acid"),
    # item 5: the acridone position 4 (the reflected catalog map read it as 1)
    ("OC(=O)c1cccc2c(=O)c3ccccc3[nH]c12", "9-oxo-9,10-dihydroacridine-4-carboxylic acid"),
    ("OC(=O)c1ccc2[nH]c3ccccc3c(=O)c2c1", "9-oxo-9,10-dihydroacridine-2-carboxylic acid"),
    # item 3 (sub-case C: the senior group arrived as a prefix)
    ("NC(=O)c1cc(=O)c2ccccc2o1", "4-oxo-4H-1-benzopyran-2-carboxamide"),
    ("N#Cc1cc(=O)c2ccccc2o1", "4-oxo-4H-1-benzopyran-2-carbonitrile"),
    ("Nc1nc2ccccc2cc1C#N", "2-aminoquinoline-3-carbonitrile"),
    # item 4 (an amine beside an acid; 'quinolin-2-amine3-carboxylic acid' was unparseable)
    ("Nc1nc2ccccc2cc1C(=O)O", "2-aminoquinoline-3-carboxylic acid"),
    ("Nc1c(C(=O)O)cnc2ccccc12", "4-aminoquinoline-3-carboxylic acid"),
    ("Nc1ccc2[nH]ccc2c1C(=O)O", "5-amino-1H-indole-4-carboxylic acid"),
    # carboxamide / carbonitrile on a ring (were '3-cyanoquinoline', '5-carbamoyl-1H-indole',...)
    ("N#Cc1cnc2ccccc2c1", "quinoline-3-carbonitrile"),
    ("NC(=O)c1cnc2ccccc2c1", "quinoline-3-carboxamide"),
    ("N#Cc1ccc2[nH]ccc2c1", "1H-indole-5-carbonitrile"),
    ("NC(=O)c1ccc2[nH]ccc2c1", "1H-indole-5-carboxamide"),
    ("N#Cc1cc2ccccc2o1", "1-benzofuran-2-carbonitrile"),
    ("NC(=O)c1cc2ccccc2o1", "1-benzofuran-2-carboxamide"),
    # item 6 (naphthyridines; were 'pyrido[2,3-b]pyridine' / 'pyrido[4,3-c]pyridine')
    ("c1cnc2ncccc2c1", "1,8-naphthyridine"),
    ("c1cc2cnccc2cn1", "2,6-naphthyridine"),
    ("OC(=O)c1ccc2cccnc2n1", "1,8-naphthyridine-2-carboxylic acid"),
    # item 7 (a) monocyclic: the indicated hydrogen / hydro prefixes of the parent
    ("OC(=O)c1cc(=O)cco1", "4-oxo-4H-pyran-2-carboxylic acid"),
    ("OC(=O)c1cc(=O)c(O)co1", "5-hydroxy-4-oxo-4H-pyran-2-carboxylic acid"),
    ("OC(=O)c1ccc(=O)oc1", "2-oxo-2H-pyran-5-carboxylic acid"),
    ("OC(=O)c1cc(=O)ccs1", "4-oxo-4H-thiopyran-2-carboxylic acid"),
    ("N#Cc1cc(=O)cco1", "4-oxo-4H-pyran-2-carbonitrile"),
    ("OC(=O)c1ccc(=O)[nH]c1", "6-oxo-1,6-dihydropyridine-3-carboxylic acid"),
    ("OC(=O)c1ccc(=O)n(C)c1", "1-methyl-6-oxo-1,6-dihydropyridine-3-carboxylic acid"),
    ("OC(=O)c1cc(=O)[nH]cc1", "2-oxo-1,2-dihydropyridine-4-carboxylic acid"),
    ("OC(=O)C1=CC(=O)CCO1", "4-oxo-3,4-dihydro-2H-pyran-6-carboxylic acid"),
    # item 7 (b) polycyclic: the 9-oxo was dropped (a different molecule; abstain before)
    ("OC(=O)c1ccc2-c3ccccc3C(=O)c2c1", "9-oxo-9H-fluorene-2-carboxylic acid"),
    # found while mapping the class (same producers)
    ("OC(=O)c1ccc2c(=O)c3ccccc3c(=O)c2c1",  # the BB's own PIN,:29471 (abstain before)
     "9,10-dioxo-9,10-dihydroanthracene-2-carboxylic acid"),
    ("OC(=O)c1cc(=O)c2ccccc2[nH]1", "4-oxo-1,4-dihydroquinoline-2-carboxylic acid"),
    ("O=C(O)c1cc2ccccc2c(=O)[nH]1",  # was 'isoquinolin-1-one3-carboxylic acid'
     "1-oxo-1,2-dihydroisoquinoline-3-carboxylic acid"),
    ("OC(=O)c1c[nH]c(=O)[nH]c1=O",  # was '2,4-dioxo-1,3-diazine-5-carboxylic acid'
     "2,4-dioxo-1,2,3,4-tetrahydropyrimidine-5-carboxylic acid"),
    ("OC(=O)c1ccc(=O)[nH]n1",  # was '6-oxo-1H-1,2-diazine-3-carboxylic acid'
     "6-oxo-1,6-dihydropyridazine-3-carboxylic acid"),
    ("O=C(O)c1ccc(=O)n(-c2ccccc2)c1", "6-oxo-1-phenyl-1,6-dihydropyridine-3-carboxylic acid"),
    ("OC(=O)c1cc(=O)cc(-c2ccccc2)o1", "4-oxo-6-phenyl-4H-pyran-2-carboxylic acid"),
    ("COC(=O)c1ccc(=O)[nH]c1", "methyl 6-oxo-1,6-dihydropyridine-3-carboxylate"),
    ("CCOC(=O)c1cc(=O)cco1", "ethyl 4-oxo-4H-pyran-2-carboxylate"),
    ("NC(=O)c1ccc(=O)[nH]c1", "6-oxo-1,6-dihydropyridine-3-carboxamide"),
    #: hydroxy compounds (17) and nitriles (14) / acids (7) over amines (19);
    # amides (11) over aldehydes (15); nitriles (14) over hydroxy (17)
    ("Nc1ccc2cccc(O)c2n1", "2-aminoquinolin-8-ol"),  # was '8-hydroxyquinolin-2-amine'
    ("Nc1ncnc2[nH]c(C(=O)O)nc12", "6-amino-9H-purine-8-carboxylic acid"),
    ("N#Cc1nc2c(N)ncnc2[nH]1", "6-amino-9H-purine-8-carbonitrile"),
    ("O=Cc1cnc2ccccc2c1C(N)=O", "3-formylquinoline-4-carboxamide"),
    ("Oc1ccc2ncccc2c1C#N", "6-hydroxyquinoline-5-carbonitrile"),
    # controls: the ring C=O is the principal group (unchanged)
    ("Cc1cc(=O)c2ccccc2o1", "2-methyl-4H-1-benzopyran-4-one"),
    ("Oc1ccc2c(=O)cc(C)oc2c1", "7-hydroxy-2-methyl-4H-1-benzopyran-4-one"),
    ("O=c1ccc2ccc(O)cc2o1", "7-hydroxy-2H-1-benzopyran-2-one"),
]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS, ids=[r[0] for r in PIN_ROWS])
def test_oxo_prefix_pin(smiles, pin):
    name = name_compound(smiles)
    assert name == pin
    assert name_is_rt_exact(name, smiles), f"{name!r} does not round-trip to {smiles}"


# The research witnesses of the glued double suffix: whatever the PIN tier ships for
# them, a ring '-one' / '-dione' / '-amine' is never followed by a second suffix.
GLUED = re.compile(r"(?:one|dione|amine)\d|one-\d+-carb")


@pytest.mark.parametrize("smiles", [r[0] for r in PIN_ROWS[:27]])
def test_never_a_glued_second_suffix(smiles):
    res = Orthonym(style="pin").name_tiered(smiles)
    assert not GLUED.search(res.get("name") or ""), res


def test_catalog_acridone_and_xanthone_locants():
    """ (:12493) traditional acridine / xanthene numbering: C9 lies between
    C8a and C9a, the ring heteroatom (position 10) between C4a and C10a. The acridone
    map had both benzo rings reflected; the xanthone / thioxanthone maps called the ring
    chalcogen '10a' and its fusion neighbour '4b'."""
    from rdkit import Chem
    from orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
    for smiles in ("O=c1c2ccccc2[nH]c2ccccc12", "O=c1c2ccccc2oc2ccccc12",
                   "O=c1c2ccccc2sc2ccccc12"):
        locs = FUSED_HETEROCYCLE_DATA[smiles]["iupac_locants"]
        mol = Chem.MolFromSmiles(smiles)
        c9 = next(a for a, loc in locs.items() if loc == 9)
        x10 = next(a.GetIdx() for a in mol.GetAtoms()
                   if a.IsInRing() and a.GetSymbol() != "C")
        assert locs[x10] == 10, (smiles, locs)
        assert {locs[n.GetIdx()] for n in mol.GetAtomWithIdx(c9).GetNeighbors()
                if n.IsInRing()} == {"8a", "9a"}, (smiles, locs)
        assert {locs[n.GetIdx()] for n in mol.GetAtomWithIdx(x10).GetNeighbors()} \
            == {"4a", "10a"}, (smiles, locs)
        # C1 is the carbon next to C9a that is not a fusion atom
        c9a = next(a for a, loc in locs.items() if loc == "9a")
        assert 1 in {locs.get(n.GetIdx()) for n in mol.GetAtomWithIdx(c9a).GetNeighbors()}


# A group senior to the one the fused name expresses as its suffix, cited as a prefix
# or sitting in a substituent: the name still round-trips but is not a PIN
#,:18162;, so it ships below pin_verified. Each row: the
# SMILES and the name that was labelled pin_verified.
SENIOR_PREFIX_ROWS = [
    ("CNC(=O)c1cnc2ccccc2c1", "3-methylcarbamoylquinoline"),
    ("CNC(=O)c1cc(=O)c2ccccc2o1", "2-methylcarbamoylbenzo[b]pyran-4-one"),
    ("CC(C)(C)NC(=O)C1=CC2=C(C(=CC=C2)OC)OC1=O",
     "8-methoxy-3-(oxo(tert-butylamino)methyl)-2H-1-benzopyran-2-one"),
    ("CNS(=O)(=O)Cc1ccc2[nH]cc(CCN(C)C)c2c1",
     "3-(2-(dimethylamino)ethyl)-5-[(methylsulfamoyl)methyl]-1H-indole"),
]


@pytest.mark.parametrize("smiles,old", SENIOR_PREFIX_ROWS, ids=[r[0] for r in SENIOR_PREFIX_ROWS])
def test_senior_group_as_prefix_is_not_pin_verified(smiles, old):
    res = Orthonym(style="pin").name_tiered(smiles)
    assert res.get("tier") != "pin_verified", res
    assert res.get("name") and name_is_rt_exact(res["name"], smiles), res


def test_ring_pseudoketone_is_not_a_senior_group():
    """A lactone / lactam / cyclic imide inside the ring system is the '-one'
    pseudoketone class 16,:18189) even when perception labels it 'ester' /
    'tertiary_amide' / 'imide'; an exocyclic amide or acid is senior."""
    from rdkit import Chem
    from orthonym.rules.fused_rings import _senior_group_outside_ring
    from orthonym.rules.seniority import compare_seniority

    def senior(smiles):
        mol = Chem.MolFromSmiles(smiles)
        ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
        return _senior_group_outside_ring(mol, ring)

    for smiles in ("CN1C(=O)C(Cl)(Cl)C(=O)c2ccccc21", "O=C1OCc2ccccc21",
                   "CN1C(=O)c2ccccc2C1=O", "O=c1ccc2ccccc2o1"):
        got = senior(smiles)
        assert got is None or compare_seniority(got, "ketone") >= 0, (smiles, got)
    assert senior("NC(=O)c1cc(=O)c2ccccc2o1") == "primary_amide"
    assert senior("OC(=O)c1cc(=O)c2ccccc2o1") == "carboxylic_acid"
