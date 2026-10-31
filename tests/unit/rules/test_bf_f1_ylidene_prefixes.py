"""DT4 F1 task 1: a bare =O, =S or =NH on a ring atom of a fused ring system takes the prefix 'oxo',
'sulfanylidene' or 'imino' (``polycyclics.ring_ylidene_prefix``, used by ``bridged_fused_pin._assemble`` and
``_prefix_groups``), so the fused parent is named when its principal group is an acid, an aldehyde, an alcohol,...

Governing text (the Blue Book): "Ketones" (:29454) the prefix 'oxo' denotes a carbonyl group when a
group of higher seniority takes the suffix, '9,10-dioxo-9,10-dihydroanthracene-2-carboxylic acid (PIN)' (:29471);
 "Prefix nomenclature" (:24862) "After the introduction of indicated and 'added indicated hydrogen' atoms,
all substituent groups not expressed as suffixes are cited as prefixes" ('4-iminonaphthalen-1(4H)-one (PIN)'
:24870); (:29504) 'sulfanylidene' for =S ('3-amino-2-sulfanylidene-1,3-thiazolidin-4-one (PIN)':26312);
 (:26540) "The prefix 'imino' for =NH is used in presence of characteristic groups having seniority
over imines"; the
indicated hydrogen takes the lowest locant ('2,2-dimethyl-1,3-dioxo-2,3-dihydro-1H-isoindol-2-ium (PIN)':41447),
then the hydro prefixes:16872, detachable but not alphabetized), then the detachable prefixes.

Every expected name below was checked atom by atom against the input's numbering (the comments give the
locant of each substituent) and read back by OPSIN 2.9.0 to the input's full standard InChIKey outside the
engine (``name_is_rt_exact``)."""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.jvm_budget import jvm_slots
from orthonym.rules import polycyclics
from orthonym.rules.bridged_fused_pin import build_fused
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.unit

#: (smiles, PIN) the rows of the DT4 census that the ylidene prefixes name. Locants, by the atoms of the SMILES:
#: 1683: naphthalene C1 CHO, C2/C3 OH, C4 propan-2-yl, C5 and C8 the quinone carbonyls, C6 CH3;
#: 295: tetracene C2 COOH, C1 CH3, C3 OH, C12 OH, C6 and C11 the quinone carbonyls (ring B), C8 OCH3, C10 OH;
#: 765: 2-benzopyran O2, C3 and C4 and C5 CH3, C6 =O, C7 COOH, C8 OH, indicated hydrogen 3H (lowest of 3, 4, 6);
#: 861: cyclohepta[e]indene C1 propan-2-yl, C3 =O, C3a and C5a CH3, C6 and C10b OH, C8 CHO;
#: 1110: indole C2 COOH, C5 and C6 the carbonyls, indicated hydrogen 1H;
#: 564: indene C3a OH (the principal group: an alcohol is senior to an imine), C3 =NH, C6 tert-butyl;
#: 851: cyclopenta[d][1,2]oxazole C3 =O, C5 NH2 and COOH, indicated hydrogen 2H (total hydrogenation, no locants).
ROWS = [
    ("CC1=CC(=O)c2c(C=O)c(O)c(O)c(C(C)C)c2C1=O",
     "2,3-dihydroxy-6-methyl-5,8-dioxo-4-(propan-2-yl)-5,8-dihydronaphthalene-1-carbaldehyde"),
    ("COc1cc(O)c2c(c1)C(=O)c1cc3cc(O)c(C(=O)O)c(C)c3c(O)c1C2=O",
     "3,10,12-trihydroxy-8-methoxy-1-methyl-6,11-dioxo-6,11-dihydrotetracene-2-carboxylic acid"),
    ("CC1=C2C(=COC(C)C2C)C(O)=C(C(=O)O)C1=O",
     "8-hydroxy-3,4,5-trimethyl-6-oxo-4,6-dihydro-3H-2-benzopyran-7-carboxylic acid"),
    ("CC(C)C1=CC(=O)[C@]2(C)CC[C@]3(C)C(=CC=C(C=O)C[C@@H]3O)[C@]12O",
     "(3aR,5aR,6S,10bR)-6,10b-dihydroxy-3a,5a-dimethyl-3-oxo-1-(propan-2-yl)-3,3a,4,5,5a,6,7,10b-"
     "octahydrocyclohepta[e]indene-8-carbaldehyde"),
    ("O=C1C=C2C[C@H](C(=O)O)NC2=CC1=O",
     "(2R)-5,6-dioxo-2,3,5,6-tetrahydro-1H-indole-2-carboxylic acid"),
    ("CC(C)(C)C1CCC2(C(C1)CCC2=N)O", "6-tert-butyl-3-iminooctahydro-3aH-inden-3a-ol"),
    ("C1[C@@H]2[C@H](C[C@@]1(C(=O)O)N)ONC2=O",
     "(3aR,5S,6aS)-5-amino-3-oxohexahydro-2H-cyclopenta[d][1,2]oxazole-5-carboxylic acid"),
    # a one-ring-saturated fused carbocycle: C5 =O of naphthalene, C2 COOH (lowest, not C6 of 1-oxo-1,2,3,4-)
    ("O=C(O)c1ccc2c(c1)CCCC2=O", "5-oxo-5,6,7,8-tetrahydronaphthalene-2-carboxylic acid"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", ROWS)
def test_the_row_is_a_verified_pin(smiles, name):
    with jvm_slots(1, purpose="bf-f1-ylidene"):
        row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), (row["name"], row["tier"])
    assert name_is_rt_exact(name, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    # the book's PIN rows with an oxo or imino prefix on a ring: named before this task, they must not move
    ("O=C1c2ccccc2C(=O)c2cc(C(=O)O)ccc12", "9,10-dioxo-9,10-dihydroanthracene-2-carboxylic acid"),   #:29471
    ("N=C1C=CC(=O)c2ccccc12", "4-iminonaphthalen-1(4H)-one"),                                        #:24870
    ("O=C(O)c1cccc2ccc(N3C(=O)c4ccccc4C3=O)cc12",
     "7-(1,3-dioxo-1,3-dihydro-2H-isoindol-2-yl)naphthalene-1-carboxylic acid"),                      #:33879
    ("O=C1NC(=O)c2cc(C(=O)O)ccc12", "1,3-dioxo-2,3-dihydro-1H-isoindole-5-carboxylic acid"),
])
def test_the_protected_rows_do_not_move(smiles, name):
    with jvm_slots(1, purpose="bf-f1-ylidene"):
        row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), (row["name"], row["tier"])


@pytest.mark.opsin_gate
def test_the_cyclic_carbamate_is_not_named_with_an_alcohol_as_principal_group():
    # 670: the ring carbamate O-C(=O)-N is a pseudoketone, senior to the alcohols; the builder's name has
    # 'ol' as the suffix, which the spelling check rejects ("name expresses 'alcohol' as the principal class; the
    # structure has 'ketone'"). The row is not pin_verified: the ylidene prefix gets no wrong class through.
    smiles = "CS[C@H]1[C@H]2[C@@H]([C@H]([C@@H]1O)O)OC(=O)N2"
    with jvm_slots(1, purpose="bf-f1-ylidene"):
        row = Orthonym().name_tiered(smiles)
    assert row["tier"] == "abstain" and row["limit_code"] == "NO_VERIFIED_PIN", row["name"]


@pytest.mark.parametrize("smiles,idx,ring_idx,prefix", [
    ("O=C1CCCCC1", 0, 1, "oxo"),
    ("S=C1CCCCC1", 0, 1, "sulfanylidene"),
    ("N=C1CCCCC1", 0, 1, "imino"),
    ("CN=C1CCCCC1", 1, 2, None),           # a substituted imine: the fragment namer's
    ("O=C1CCCCC1", 1, 0, None),            # the ring atom is not the ylidene atom
    ("OC1CCCCC1", 0, 1, None),             # a single bond
    ("[18O]=C1CCCCC1", 0, 1, None),        # a labelled atom
    ("[NH2+]=C1CCCCC1", 0, 1, None),       # a charged atom
    ("N=C1CCCCC1", 1, 0, None),
])
def test_ring_ylidene_prefix(smiles, idx, ring_idx, prefix):
    mol = Chem.MolFromSmiles(smiles)
    if prefix is None and mol.GetBondBetweenAtoms(idx, ring_idx) is None:
        pytest.skip("not bonded")
    assert polycyclics.ring_ylidene_prefix(mol, idx, ring_idx) == prefix


def test_the_flag_is_off_for_the_other_caller():
    # ``name_substituted_polycyclic`` keeps the default: a bare =O is not spelled by the fragment namer
    mol = Chem.MolFromSmiles("O=C1c2ccccc2C(=O)c2ccccc12")
    a2l = {a: i + 1 for i, a in enumerate(a.GetIdx() for a in mol.GetAtoms() if a.IsInRing())}
    assert polycyclics._partial_sat_substituent_prefix(mol, a2l) is None
    assert polycyclics._partial_sat_substituent_prefix(mol, a2l, ylidene_prefixes=True) is not None


@pytest.mark.opsin_gate
def test_without_the_ylidene_prefixes_the_rows_decline(monkeypatch):
    # mutation: with no ylidene prefix (the state before the task) the builder declines the naphthoquinone
    # aldehyde, which it names with them
    mol = Chem.MolFromSmiles(ROWS[0][0])
    with jvm_slots(1, purpose="bf-f1-ylidene"):
        assert build_fused(Chem.Mol(mol)) is not None
        monkeypatch.setattr(polycyclics, "ring_ylidene_prefix", lambda *a, **k: None)
        assert build_fused(Chem.Mol(mol)) is None
