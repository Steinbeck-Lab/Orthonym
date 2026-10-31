"""DT4 F1 task 2: the '[1]benzofuro' / '[1]benzothieno' attached-component prefix
(``bridged_fused_pin.fusion_components.prefix_form``), decision D1 of `internal notes`.

Governing text (the Blue Book): "Heteromonocyclic components fused to a benzene ring" (:11813,
'1-benzofuran (PIN)':11827) the benzo unit is one component; (:13437) it "may be treated as a parent
component or an attached component depending on the order of seniority described in ";
(:11905) an attached name ends in 'o' "or by other means described in ", (:12047)
'furo (preferred prefix) (from furan, PIN)' and 'thieno', (:11907) "Locants that describe structural features of
components, such as positions of heteroatoms, are kept with the name of the component and are enclosed within
square brackets". PIN precedents of a benzo unit as attached component: '[1]benzopyrano[2,3-c]pyrrole (PIN)'
(:12157), '[1,2]benzoxazolo[6,5-g]quinoline (PIN)' (:13449). The prefix is certified only when four seniority
checks hold: (1) makes the benzo unit the attached component, (2) no multiparent name
('benzo[1,2-b:4,5-c']difuran (PIN) (not furo[3,4-f][1]benzofuran...)',:13453), (3) no retained name is
displaced (:13461), (4) no 'dibenzo' name ('6H-dibenzo[b,d]pyran (PIN) (not 6H-benzo[c][1]benzopyran',
:13463).

OPSIN 2.9.0 reads '[1]benzofuro' and '[1]benzofurano' to one structure, and reads the descriptor of any
symmetry-equivalent reading, so the read-back never decides the letters or the locants: they are pinned here by
the derivations below. The names were also read back by OPSIN to the input's full standard InChIKey
(``name_is_rt_exact``), outside the engine.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.jvm_budget import jvm_slots
from orthonym.rules.bridged_fused_pin import fusion_components as fc
from orthonym.rules.bridged_fused_pin import hetero_fusion, parents
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.unit


def _skeleton(smiles):
    return Chem.MolFromSmiles(parents._key_of(Chem.MolFromSmiles(smiles)))


# Pterocarpan skeleton, ring A = benzene (C1-C4, C4a, C11b), pyran (C4a, O5, C6, C6a, C11a, C11b), furan (C6a,
# C11a, O11, C10a, C6b), ring D = benzene (C6b, C7-C10, C10a).
# Derivation: the components are benzene, pyran, furan, benzene; the benzo units 1-benzopyran and
# 1-benzofuran tie on (a) O and (b) two rings; (c) the larger ring first, six before five: the parent is
# 1-benzopyran (O1, C2, C3, C4, C4a, C8a; the sides are lettered a 1-2, b 2-3, c 3-4). The furan is fused at
# C3-C4, side c. 1-Benzofuran numbers O1, C2, C3: the furan carbon C11a (next to O11) is its C2, C6a is its C3.
# Pyran C3 = C6a (benzofuran 3), pyran C4 = C11a (benzofuran 2); the locants follow the direction of lettering
# (C3 -> C4): '[3,2-c]':11911). The numbering of the whole system:12501): O5 and O11.
PTEROCARPAN = "COc1ccc2c(c1)OC1c3ccc(O)c(O)c3OCC21"
PTEROCARPAN_SKELETON = "c1ccc2c(c1)OC1c3ccccc3OCC21"
# The 270 / 762 skeleton: 1-benzopyran as the parent; the furan is fused at C7-C8 (side h: f 5-6, g 6-7, h 7-8,
# i 8-8a). OPSIN's structure of the name has the furan O on C7 and the furan carbon that is bonded to the
# benzo ring of the furan on C8: benzofuran C2 = pyran C7, benzofuran C3 = pyran C8; in the direction of
# lettering (C7 -> C8) the attached locants are '2,3': '[2,3-h]'. The letter is the lowest one the skeleton allows
# (1-benzopyran has no symmetry).
ANGULAR_SKELETON = "O1CC=CC2=C1C1=C(C=C2)OC2=C1C=CC=C2"


@pytest.mark.parametrize("smiles,name", [
    (PTEROCARPAN_SKELETON, "[1]benzofuro[3,2-c][1]benzopyran"),
    (ANGULAR_SKELETON, "[1]benzofuro[2,3-h][1]benzopyran"),
    # the thiophene analogue: O > S, so 1-benzofuran is the parent (a)); [1]benzothieno attached
    ("c1ccc2c(c1)oc1c2sc2ccccc12", "[1]benzothieno[3,2-b][1]benzofuran"),
    ("c1ccc2c(c1)sc1ncccc12", "[1]benzothieno[2,3-b]pyridine"),
    ("c1ccc2c(c1)sc1cccnc12", "[1]benzothieno[3,2-b]pyridine"),
])
def test_the_benzofuro_and_benzothieno_names_are_derived(smiles, name):
    assert hetero_fusion.rule_name(_skeleton(smiles)) == name


def test_the_book_benzo_unit_prefixes_are_unchanged():
    # '[1]benzopyrano[2,3-c]pyrrole (PIN)':12157 and '[1,2]benzoxazolo[6,5-g]quinoline (PIN)':13449
    assert hetero_fusion.rule_name(_skeleton("C1=C2CNC=C2Oc2ccccc21")) == "[1]benzopyrano[2,3-c]pyrrole"
    assert hetero_fusion.rule_name(
        _skeleton("c1cnc2cc3cc4cnoc4cc3cc2c1")) == "[1,2]benzoxazolo[6,5-g]quinoline"


def test_the_rejected_benzofuro_name_of_the_book_is_not_built():
    #:13461 '10H-furo[3',2':4,5]indeno[2,1-b]pyridine (PIN) (not 10H-[1]benzofuro[5',4':3,4]cyclopenta[1,2-b]
    # pyridine;... indene, as a retained name, must be used)': both names are this skeleton (OPSIN 2.9.0). The rule
    # builder gives it no name (a benzene that sits in the retained indene is never merged into a benzofuran unit,
    #, and the parent is the book's PIN from the table, never the rejected spelling.
    skeleton = _skeleton("C1=COC=2C1=C1CC3=NC=CC=C3C1=CC2")
    assert hetero_fusion.rule_name(skeleton) is None
    got = parents._parent_for_key(Chem.MolToSmiles(skeleton))
    assert got is not None and got[0] == "furo[3',2':4,5]indeno[2,1-b]pyridine" and got[2] == "bb_table"


def test_the_multiparent_name_is_kept_over_the_benzofuran_reading():
    #:13453 'benzo[1,2-b:4,5-c']difuran (PIN) (not furo[3,4-f][1]benzofuran...; a multiparent name is preferred
    # to a two-component fused name)'. The multiparent guard is load-bearing: with it off, the benzofuran reading
    # is built; with it on, the skeleton is left to the multiparent slice (declined here).
    skeleton = _skeleton("O1C=2C(C=C1)=CC1=COC=C1C2")
    assert hetero_fusion.rule_name(skeleton) is None
    real = hetero_fusion._multi
    hetero_fusion._multi = lambda *a, **k: False
    try:
        assert hetero_fusion.rule_name(skeleton) == "furo[3,4-f][1]benzofuran"
    finally:
        hetero_fusion._multi = real


def test_a_dibenzo_name_is_not_replaced_by_a_benzo_benzopyran_reading():
    #:13463 '6H-dibenzo[b,d]pyran (PIN) (not 6H-benzo[c][1]benzopyran;...)': two identical benzo components
    # fused to the parent are multiplied:13467), so no benzo unit reading is built
    assert hetero_fusion.rule_name(_skeleton("C1=CC=CC=2OCC3=C(C21)C=CC=C3")) is None


def test_the_parent_numbering_is_the_peripheral_one():
    # OPSIN 2.9.0's numbering of '6H-[1]benzofuro[3,2-c][1]benzopyran' is the pterocarpan numbering
    #:12501): O5 and O11 are the heteroatoms, C6a and C11a the saturable furan carbons; the engine's own
    # periphery check accepts it (fusion carbons lettered, order around the periphery)
    mol = Chem.MolFromSmiles(PTEROCARPAN_SKELETON)
    got = parents.fused_parent(mol, set(range(mol.GetNumAtoms())))
    assert got is not None and got.name == "[1]benzofuro[3,2-c][1]benzopyran"
    assert got.source == "hetero_fusion_name+opsin"
    for numbering in got.numberings:
        loc = {a: str(l) for a, l in numbering.items()}
        hetero = sorted(loc[a.GetIdx()] for a in mol.GetAtoms() if a.GetSymbol() == "O")
        assert hetero == ["11", "5"]
        assert fc.periphery_numbering_ok(mol, numbering)


@pytest.mark.parametrize("smiles,name", [
    # 6a,11a-dihydro-6H-[1]benzofuro[3,2-c][1]benzopyran is the pterocarpan skeleton (medicarpin-type rows)
    (PTEROCARPAN, "9-methoxy-6a,11a-dihydro-6H-[1]benzofuro[3,2-c][1]benzopyran-3,4-diol"),
    # a coumestan: the lactone C6=O is the principal group, a ketone-class suffix on the CH2 position
    ("COc1c(CC=C(C)C)c(O)cc2oc(=O)c3c4ccc(O)cc4oc3c12",
     "3,9-dihydroxy-1-methoxy-2-(3-methylbut-2-en-1-yl)-6H-[1]benzofuro[3,2-c][1]benzopyran-6-one"),
    ("COc1ccc2c(c1O)OCC1c3ccc(O)c(OC)c3OC21",
     "3,10-dimethoxy-6a,11a-dihydro-6H-[1]benzofuro[3,2-c][1]benzopyran-4,9-diol"),
])
@pytest.mark.opsin_gate
def test_the_rows_are_named_and_read_back(smiles, name):
    with jvm_slots(1, purpose="bf-f1-benzo"):
        row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified"), (row["name"], row["tier"])
    assert name_is_rt_exact(name, smiles)


def test_without_the_contraction_the_skeleton_is_declined(monkeypatch):
    # mutation: with the pre-D1 behaviour (no prefix for a benzofuran unit) the pterocarpan skeleton has no name
    real = fc.prefix_form

    def pre_d1(comp):
        loc, stem = fc.split_locants(comp.name)
        if comp.kind == "benzo" and stem.endswith(("furan", "thiophene")):
            return None
        return real(comp)

    monkeypatch.setattr(fc, "prefix_form", pre_d1)
    assert hetero_fusion.rule_name(_skeleton(PTEROCARPAN_SKELETON)) is None
