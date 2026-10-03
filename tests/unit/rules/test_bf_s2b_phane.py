""" phane seniority inside the bridged fused builder (``selection.phane_reading``).

 (1) (the Blue Book): "cyclophanes are cyclic phane structures containing one
or more rings or ring systems, at least one ring or ring system of which must be a mancude
system attached to adjacent atoms or chains at nonadjacent ring positions";
(:23843): "cyclic phane systems > fused ring systems > bridged fused systems > non-fused bridged
systems";:23895 '(I) 3,7-dithia-1(1,7),5(7,1)-dinaphthalenacyclooctaphane (PIN; a phane name)
(II) 5,7,14,16-tetrahydro-1,17:8,10-diethenodibenzo[c,j][1,8]dithiacyclotetradecine (a bridged
fused ring name)',:23897 "A phane name is senior to a bridged fused ring name."

The builder declines a reading that cuts a mancude ring of the compound (an aromatic ring, or a
ring every atom of which that can carry a double bond carries a ring double bond, but for one
indicated hydrogen when their number is odd::3557) into a carbon bridge while the
ring system also holds atoms that lie in no mancude ring (the chains). The book's
bridged fused PINs that a literal reading of (1) would call cyclophanes keep their
names: their chains are the bridges (1,4-ethanonaphthalene:14407, 9,10-ethanoanthracene
:14183), their heteroatom bridge closes a furan (:14189,:14587), or every atom of the ring
system carries a ring double bond (:14277 (I); the gate's W2E-P5BR-2)."""
import pytest
from rdkit import Chem

import orthonym.rules.bridged_fused_pin as bfp
from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.rules.bridged_fused_pin import selection
from tests.support.rt_assert import name_is_rt_exact

#::23895: the one fused reading's parent, dibenzo[c,j][1,8]dithiacyclotetradecine, is printed
#: only in the non-PIN name (II), so no table names it; the test supplies a name to reach the
#: check
BOOK_PHANE = "C1SCc2ccc3cccc(CSCc4ccc5cccc1c5c4)c3c2"

#: a naphthalene joined at its 1,4-positions by a saturated chain: (b) (:14271)
#: picks the benzo[n]annulene parent with the naphthalene's C2=C3 as an etheno bridge, the
#: reading (II) of:23895 makes; the base shipped the first two pin_verified
CHAIN_LINKED = [
    ("c1ccc2c3ccc(c2c1)CCC3", "7,8-dihydro-6H-5,9-ethenobenzo[7]annulene"),
    ("c1ccc2c3ccc(c2c1)CCCC3", "6,7,8,9-tetrahydro-5,10-ethenobenzo[8]annulene"),
    ("c1ccc2c3ccc(c2c1)CCCCCCCCC3", None),
    ("c1ccc2c3ccc(c2c1)CCCCCCCCCC3", None),
]

#: the same shape on a mancude unit RDKit does not perceive as aromatic (4n pi electrons): a
#: heptalene joined at its 1,4-positions by a saturated chain, and the
#: eight-membered ring of benzo[8]annulene (:2662) joined across its 5,8-positions; the
#: builder read each with the unit's C=C as an etheno bridge before the mancude test, and the
#: S2b branch shipped the second column pin_verified
MANCUDE_NOT_AROMATIC = [
    ("C1=CC=C2C=C3C=CC(=C2C=C1)CCC3", "8,9-dihydro-7H-6,10-ethenocyclohepta[8]annulene"),
    ("C1=CC=C2C=C3C=CC(=C2C=C1)CCCC3", "7,8,9,10-tetrahydro-6,11-ethenocyclohepta[9]annulene"),
    ("C1=CC2=c3ccccc3=CC=C1CCC2", "7,8-dihydro-6H-5,9-ethenobenzo[9]annulene"),
    ("C1=CC2=c3ccccc3=CC=C1CCCC2", "6,7,8,9-tetrahydro-5,10-ethenobenzo[10]annulene"),
]

#: the same shape on a mancude unit with an indicated hydrogen: the seven-membered ring of
#: 5H-benzo[7]annulene joined at its 5,8-positions (the CH2 is the bridgehead) and the azepine
#: ring of 1H-1-benzazepine at its 1,4-positions. RDKit perceives neither ring as aromatic, and
#: the CH2 or N carries no ring double bond; the base shipped the first name pin_verified, the
#: S2b branch before this test all three
INDICATED_HYDROGEN_UNIT = [
    ("c1ccc2c(c1)C1C=CC(=C2)CCC1", "5,6,7,8-tetrahydro-5,9-ethenobenzo[8]annulene"),
    ("c1ccc2c(c1)C1C=CC(=C2)CCCC1", "6,7,8,9-tetrahydro-5H-5,10-ethenobenzo[9]annulene"),
    ("c1ccc2c(c1)N1C=CC(=C2)CCC1", "3,4-dihydro-2H-1,5-etheno-1-benzazocine"),
]

#: rows whose book name is a bridged fused PIN although a literal reading of (1)
#: makes them cyclophanes (a mancude system joined at nonadjacent positions), and the gate's
#: W2E-P5BR-2; the builder must name each, with its phane check silent
KEEP = [
    ("c1ccc2c3ccc(c2c1)CC3", "1,4-ethanonaphthalene"),                            #:14407
    ("c1ccc2c3c4ccccc4c(c2c1)CC3", "9,10-ethanoanthracene"),                      #:14183
    ("C1=C2CC(=C1)c1cc3c4ccc(c3cc12)CC4", "1,4-ethano-5,8-methanoanthracene"),    #:14235
    ("c1ccc2c3cc(cc2c1)C3", "1,3-methanonaphthalene"),                            #:19863
    ("c1cc2c3ccn2Cc1c3", "1,5-methanoindole"),                                    #:14253
    ("C1=CC2=C3C=c4ccc(o4)=CC3=C1C2", "6,9-epoxy-1,4-methanobenzo[8]annulene"),   #:14189
    ("C1=CC23CCC2(C=C1)c1ccc3o1", "1,4-epoxy-4a,8a-ethanonaphthalene"),           #:14587
    ("c1c2cc3cc4c5ccc(o5)c4cc3c1C2", "5,8-epoxy-1,3-methanoanthracene"),          #:14229
    ("c1ccc2c3ccc(o3)c2c1", "1,4-epoxynaphthalene"),                              #:14227
    ("C1=CC2=C3N=Cc4cc5ccc6cc5cc4C(=CC3=C1C2)CCCCC6",
     "1,4-methano-10,13-pentanonaphtho[2,3-c][1]benzazocine"),                   #:19904
    ("C12=CC=C(C3=CC=CC=C13)C=CC=C2", "5,10-ethenobenzo[8]annulene"),             # W2E-P5BR-2
]


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s2b"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


def test_the_book_phane_row_is_a_phane_reading(monkeypatch):
    mol = Chem.MolFromSmiles(BOOK_PHANE)
    monkeypatch.setattr(selection, "_DECLINED", {})
    monkeypatch.setattr(selection, "_parent_name", lambda m, residual: "dithiacyclotetradecine")
    splits = selection.best_splits(mol, selection.ring_system(mol))
    assert splits, "the reading (II) of :23895 is not found"
    assert all(selection.phane_reading(mol, sp) for sp in splits)


@pytest.mark.parametrize("smiles,base_name", CHAIN_LINKED)
def test_a_naphthalene_cut_into_an_etheno_bridge_beside_a_chain_is_a_phane_reading(smiles, base_name):
    mol = Chem.MolFromSmiles(smiles)
    splits = selection.best_splits(mol, selection.ring_system(mol))
    assert splits and all(selection.phane_reading(mol, sp) for sp in splits)
    assert bfp.build(mol) is None


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,base_name", CHAIN_LINKED)
def test_the_engine_ships_no_bridged_fused_name_for_a_cyclophane(smiles, base_name):
    pin = _row(smiles, "pin")
    assert pin["tier"] == "abstain", (pin.get("name"), pin["tier"])
    best = _row(smiles, "best-effort")
    assert best["tier"] in ("systematic_verified", "pin_unverified"), (best.get("name"), best["tier"])
    assert "etheno" not in (best.get("name") or "") and best.get("name") != base_name


@pytest.mark.parametrize("smiles,name", KEEP)
def test_bridged_fused_pins_keep_their_names(smiles, name):
    mol = Chem.MolFromSmiles(smiles)
    system = selection.ring_system(mol)
    options = bfp._options(mol, system)
    assert options, "declined"
    assert not any(selection.phane_reading(mol, o[0]) for o in options)
    got = bfp.build(mol)
    assert got is not None and got[0] == name, got


def test_only_carbon_bridge_atoms_count():
    # 1,4-epoxy-4a,8a-ethanonaphthalene (:14587): the epoxy oxygen lies in the furan ring of
    # the formal parent, and the ethano carbons carry no ring double bond; were the oxygen a
    # carbon atom, the check would fire
    mol = Chem.MolFromSmiles("C1=CC23CCC2(C=C1)c1ccc3o1")
    options = bfp._options(mol, selection.ring_system(mol))
    oxygen = next(a for o in options for chain in o[0].bridges for a in chain
                  if mol.GetAtomWithIdx(a).GetSymbol() == "O")
    assert mol.GetAtomWithIdx(oxygen).GetIsAromatic()
    assert not any(selection.phane_reading(mol, o[0]) for o in options)


def test_a_mancude_polycycle_without_a_chain_keeps_its_name():
    # W2E-P5BR-2: the etheno bridge is two aromatic carbon atoms, and every ring atom carries
    # a ring double bond (the clause that keeps:14277 (I) a bridged fused PIN)
    mol = Chem.MolFromSmiles("C12=CC=C(C3=CC=CC=C13)C=CC=C2")
    options = bfp._options(mol, selection.ring_system(mol))
    assert any(mol.GetAtomWithIdx(a).GetIsAromatic()
               for o in options for chain in o[0].bridges for a in chain)
    assert not any(selection.phane_reading(mol, o[0]) for o in options)


@pytest.mark.parametrize("smiles,earlier_name", MANCUDE_NOT_AROMATIC)
def test_a_mancude_ring_rdkit_does_not_call_aromatic_is_a_phane_unit_too(smiles, earlier_name):
    # (1) (:23828) asks for "a mancude system"; the bridge carbon atoms of the
    # reading are not aromatic for RDKit, and each lies in a ring of the input every atom of
    # which carries a ring double bond
    mol = Chem.MolFromSmiles(smiles)
    splits = selection.best_splits(mol, selection.ring_system(mol))
    assert splits
    for sp in splits:
        carbons = [a for chain in sp.bridges for a in chain
                   if mol.GetAtomWithIdx(a).GetAtomicNum() == 6]
        assert carbons and not any(mol.GetAtomWithIdx(a).GetIsAromatic() for a in carbons)
        assert any(selection._in_mancude_ring(mol, a, sp.unsaturated) for a in carbons)
        assert selection.phane_reading(mol, sp)
    assert bfp.build(mol) is None


def test_a_ring_with_a_saturated_atom_is_not_mancude():
    # 1,4-ethanonaphthalene (:14407): the ethano carbons lie only in rings that also hold the
    # bridgeheads' neighbours without a ring double bond (the ethano carbons themselves)
    mol = Chem.MolFromSmiles("c1ccc2c3ccc(c2c1)CC3")
    options = bfp._options(mol, selection.ring_system(mol))
    assert options
    for o in options:
        for chain in o[0].bridges:
            assert not any(selection._in_mancude_ring(mol, a, o[0].unsaturated) for a in chain)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,earlier_name", MANCUDE_NOT_AROMATIC)
def test_the_engine_ships_no_bridged_fused_name_for_a_non_aromatic_mancude_cyclophane(
        smiles, earlier_name):
    pin = _row(smiles, "pin")
    assert pin["tier"] == "abstain", (pin.get("name"), pin["tier"])
    best = _row(smiles, "best-effort")
    assert best["tier"] in ("systematic_verified", "pin_unverified"), (best.get("name"), best["tier"])
    assert "etheno" not in (best.get("name") or "") and best.get("name") != earlier_name
    assert name_is_rt_exact(best["name"], smiles), best["name"]


@pytest.mark.parametrize("smiles,earlier_name", INDICATED_HYDROGEN_UNIT)
def test_a_mancude_ring_with_an_indicated_hydrogen_is_a_phane_unit_too(smiles, earlier_name):
    # (:3557): a mancude ring may have "positions where no multiple bond is
    # attached"; the bridge carbon atoms carry a ring double bond and lie in a ring whose only
    # atom without one is that indicated hydrogen, and the chain lies in no mancude ring
    mol = Chem.MolFromSmiles(smiles)
    splits = selection.best_splits(mol, selection.ring_system(mol))
    assert splits
    for sp in splits:
        carbons = [a for chain in sp.bridges for a in chain
                   if mol.GetAtomWithIdx(a).GetAtomicNum() == 6]
        assert carbons and not any(mol.GetAtomWithIdx(a).GetIsAromatic() for a in carbons)
        assert all(a in sp.unsaturated for a in carbons)
        rings = [r for r in mol.GetRingInfo().AtomRings() if set(carbons) <= set(r)
                 and selection._mancude_ring(mol, r, sp.unsaturated)]
        assert rings and all(sum(1 for a in r if a not in sp.unsaturated) == 1 for r in rings)
        assert all(selection._in_mancude_ring(mol, a, sp.unsaturated) for a in carbons)
        assert selection.phane_reading(mol, sp)
    assert bfp.build(mol) is None


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,earlier_name", INDICATED_HYDROGEN_UNIT)
def test_the_engine_ships_no_bridged_fused_name_for_an_indicated_hydrogen_cyclophane(
        smiles, earlier_name):
    pin = _row(smiles, "pin")
    assert pin["tier"] == "abstain", (pin.get("name"), pin["tier"])
    best = _row(smiles, "best-effort")
    assert best["tier"] in ("systematic_verified", "pin_unverified"), (best.get("name"), best["tier"])
    assert "etheno" not in (best.get("name") or "") and best.get("name") != earlier_name
    assert name_is_rt_exact(best["name"], smiles), best["name"]


def test_an_indicated_hydrogen_bridge_atom_is_not_a_cut_ring():
    # '4,7-methanocyclopenta[a]indene (PIN)' (:19829): the two five-membered rings the methano
    # bridge closes are mancude with the methano carbon as their only atom without a ring
    # double bond, so that carbon is their indicated hydrogen, not part of a ring the reading
    # cuts; and no atom of the ring system lies outside a mancude ring
    mol = Chem.MolFromSmiles("C1=CC2=CC3=C4C=CC(=C3C2=C1)C4")
    options = bfp._options(mol, selection.ring_system(mol))
    assert options
    for o in options:
        (methano,) = o[0].bridges
        assert len(methano) == 1 and methano[0] not in o[0].unsaturated
        rings = [r for r in mol.GetRingInfo().AtomRings() if methano[0] in r]
        assert len(rings) == 2 and all(selection._mancude_ring(mol, r, o[0].unsaturated) for r in rings)
        assert not selection._in_mancude_ring(mol, methano[0], o[0].unsaturated)
        assert not selection.phane_reading(mol, o[0])
    got = bfp.build(mol)
    assert got is not None and got[0] == "4,7-methanocyclopenta[a]indene", got


def test_a_mancude_polycycle_with_one_indicated_hydrogen_has_no_chain():
    # a naphthalene joined at its 1,4-positions by -CH=CH-CH2-: every atom of the ring system
    # but the CH2 carries a ring double bond, and the CH2 is the indicated hydrogen of the two
    # seven-membered rings, so the ring system is one mancude polycycle with an odd atom count,
    # the case of W2E-P5BR-2 ('5,10-ethenobenzo[8]annulene', the four-carbon -CH=CH-CH=CH-);
    # the base named it 6H-5,9-ethenobenzo[7]annulene, which OPSIN reads back to the input
    smiles = "c1ccc2c3ccc(c2c1)C=CC3"
    mol = Chem.MolFromSmiles(smiles)
    options = bfp._options(mol, selection.ring_system(mol))
    assert options
    for o in options:
        system = set(o[0].residual).union(*(set(c) for c in o[0].bridges))
        assert len(system - o[0].unsaturated) == 1
        assert not selection.phane_reading(mol, o[0])
    got = bfp.build(mol)
    assert got is not None and got[0] == "6H-5,9-ethenobenzo[7]annulene", got
    assert name_is_rt_exact(got[0], smiles), got[0]
