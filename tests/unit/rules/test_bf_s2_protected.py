"""Slice S2 protection: the 16 Blue Book bridged fused rows that MATCH at the S2 base stay
byte-identical at both tiers, and the boundary rows where a von Baeyer or phane name is the
PIN never get a bridged fused name from the builder (the design spec, sections 8 and 9)."""
from itertools import combinations

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.rules.bridged_fused import name_bridged_fused_pin
from orthonym.rules.bridged_fused_pin import build, selection

BB_MATCH_ROWS = [
    ("c1ccc2c3c4ccccc4c(c2c1)CC3", "9,10-ethanoanthracene"),                      #:14183
    ("c1ccc2c3ccc(o3)c2c1", "1,4-epoxynaphthalene"),                              #:14227
    ("c1c2cc3cc4c5ccc(o5)c4cc3c1C2", "5,8-epoxy-1,3-methanoanthracene"),          #:14229
    ("C1=C2CC(=C1)c1cc3c4ccc(c3cc12)CC4", "1,4-ethano-5,8-methanoanthracene"),    #:14235
    ("C1=C2CC(=C1)c1c2c2ccc1o2", "1,4-epoxy-5,8-methanonaphthalene"),             #:14237
    ("C1=CC23C=CC=CC2(C=C1)CC3", "4a,8a-ethanonaphthalene"),                      #:14249
    ("c1ccc2c(c1)C1CCN2c2ccccc21", "9H-9,10-ethanoacridine"),                     #:14251
    ("C1=CC2CCC1c1cc3ccccc3cc12", "1,4-dihydro-1,4-ethanoanthracene"),            #:14399
    ("c1ccc2c3ccc(c2c1)CC3", "1,4-ethanonaphthalene"),                            #:14407
    ("c1cc2cc3ccc4cc5ccc6cc7ccc8cc9ccc%10cc1c1cc%10c9cc8c7cc6c5cc4c3cc21",
     "12,19:13,18-di(metheno)dinaphtho[2,3-a:2',3'-o]pentaphene"),               #:14527
    ("C1=CC23CCC2(C=C1)c1ccc3o1", "1,4-epoxy-4a,8a-ethanonaphthalene"),           #:14587
    ("C1=C2CC(=C1)c1ccccc12", "1,4-methanonaphthalene"),                          #:19479
    ("c1ccc2c3cc(cc2c1)C3", "1,3-methanonaphthalene"),                            #:19863
    ("c1ccc2c(c1)C1CCC2C2CCCCC12", "1,2,3,4,4a,9,9a,10-octahydro-9,10-ethanoanthracene"),  #:19964
    ("C1=CC2C(C=C1)C1C=CC2C2CCCCC12",
     "1,2,3,4,4a,8a,9,9a,10,10a-decahydro-9,10-ethenoanthracene"),               #:19966
    ("c1cc2oc1-c1coc3occ-2c13", "2,3,9-trioxa-5,8-methanocyclopenta[cd]azulene"),  #:14501
]

#: boundary rows (the design spec, section 8) where the PIN is not a bridged fused name.
#: Two rings: a bridged fused ring system has at least three (a fused ring system of two or
#: more rings plus the rings its bridges create,, the Blue Book), so every
#: entry declines these at its ring-count guard, before any selection rule runs. The rules
#: named are the book's reasons for their PINs.
TWO_RING_BOUNDARY_ROWS = [
    "C12=CC=CC=C2C1",            #:23718 bicyclo[4.1.0]hepta-1,3,5-triene (PIN),:23710
    "C12=CC=CC=C2C=C1",          #:23725 bicyclo[4.2.0]octa-1,3,5,7-tetraene (PIN)
    "C1CCCCCCCCCCCCCCCc2cccc1c2",
    #:23847 1(1,3)-benzenacycloheptadecaphane (PIN; a phane name); OPSIN 2.9.0 reads the
    # book's von Baeyer name of the same compound,:23851 'bicyclo[16.3.1]docosa-1(22),18,20-
    # triene', to this SMILES's full InChIKey (it cannot read phane names)
]

#: boundary rows of three or more rings, which the selection reaches, with the reason it
#: declines each one
BOUNDARY_ROWS = [
    # (:23710) "Fusion nomenclature gives preferred IUPAC names only to compounds
    # having at least two rings of at least five or more members": both ortho-fused readings
    # (rings 3+6 and 3+5) fail it.:48763 '(1S,2R,4R,5S)-3,6,8-trioxatricyclo[3.2.1.0^2,4]
    # octane (PIN)'; OPSIN 2.9.0 reads that name to this SMILES
    ("[C@@H]12[C@H]3O[C@H]3[C@@H](OC1)O2", "five-membered ring rule"),
    # no reading leaves an ortho-fused ring system (:23863,:23873 "A fusion name is not
    # possible")::23867 tricyclo[9.3.1.1^4,8]hexadecane (PIN);:23857
    # 1(1,3)-benzena-4(1,3)-cyclohexanacyclohexaphane (PIN, a phane name), whose von Baeyer
    # name:23861 'tricyclo[9.3.1.1^4,8]hexadeca-1(15),11,13-triene' OPSIN 2.9.0 reads to
    # this SMILES
    ("C12CCC3CCCC(CCC(CCC1)C2)C3", "no fused reading"),
    ("C1=2CCC3CCCC(CCC(=CC=C1)C2)C3", "no fused reading"),
    #:23895 3,7-dithia-1(1,7),5(7,1)-dinaphthalenacyclooctaphane (PIN); the bridged fused
    # name (II) '5,7,14,16-tetrahydro-1,17:8,10-diethenodibenzo[c,j][1,8]dithiacyclo-
    # tetradecine' reads back but is not the PIN (:23897 "A phane name is senior to a
    # bridged fused ring name"). The builder computes no phane seniority: it declines
    # because that one fused reading's parent (rings 14, 6, 6) is not a PIN parent name
    # in its table, whose largest ring has eight members
    ("C12=CC=CC=3CSCC4=CC5=C(CSCC(=CC31)C=C2)C=CC=C5C=C4", "no parent name"),
]


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s2"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", BB_MATCH_ROWS)
def test_bb_match_rows_unchanged(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])


@pytest.mark.parametrize("smiles", TWO_RING_BOUNDARY_ROWS)
def test_two_ring_rows_are_declined_by_the_ring_count(smiles):
    mol = Chem.MolFromSmiles(smiles)
    system = selection.ring_system(mol)
    assert selection.cycle_rank(mol, system) == 2
    assert name_bridged_fused_pin(mol) is None
    assert selection.best_splits(mol, system) is None
    assert build(mol) is None


def _divalent_readings(mol, system):
    """(residual, its ortho-fused rings or None) for every set of one to three whole
    chains (the divalent bridge readings ``selection.best_splits`` ranks)."""
    chains = selection.ears(mol, system)
    for j in range(1, selection.MAX_BRIDGES + 1):
        for combo in combinations(chains, j):
            residual = set(system) - set().union(*combo)
            yield combo, residual, selection.fused_rings(mol, residual)


@pytest.mark.parametrize("smiles,reason", BOUNDARY_ROWS)
def test_boundary_rows_get_no_bridged_fused_name(smiles, reason):
    mol = Chem.MolFromSmiles(smiles)
    system = selection.ring_system(mol)
    assert selection.cycle_rank(mol, system) >= 3          # past the ring-count guard
    unsat = selection.ring_unsaturation(mol, system)
    fused = [(combo, residual, rings) for combo, residual, rings
             in _divalent_readings(mol, system) if rings is not None]
    if reason == "five-membered ring rule":
        assert fused and all(not selection._five_membered_rule(r) for _, _, r in fused)
        assert all(selection._candidate(mol, system, unsat, c) is None for c, _, _ in fused)
    elif reason == "no fused reading":
        assert not fused
    else:
        assert fused and all(selection._parent_name(mol, res) is None for _, res, _ in fused)
    assert selection.best_splits(mol, system) is None
    assert name_bridged_fused_pin(mol) is None
