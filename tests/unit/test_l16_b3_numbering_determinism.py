"""lane/l16 b3: the numbering key of a decorated heteromonocycle substituent, and the determinism of
a best-effort name under a reordering of the SMILES atoms."""

import pytest
from rdkit import Chem

import orthonym.rules.ring_substituents as rs
from orthonym.cli import _tiered_namer
from tests.support.rt_assert import assert_full_rt

pytestmark = pytest.mark.unit


# "Heteromonocyclic parent hydrides" (the Blue Book): "The locant '1' is given to a
# heteroatom that occurs first in the seniority sequence used for citation of the skeletal replacement
# ('a') prefixes. The numbering is then chosen to give lowest locants to heteroatoms considered as a set
# in ascending numerical order." The stem table of ``_decorated_heteroaryl_substituent_name`` holds no
# ring of two heteroatom elements (``identify_ring_system`` names none of the azoles of O or S), so the
# function is fed the ring name and the stem the way a table that held them would. ``Br`` marks the
# parent: the atom bonded to it holds the free valence and the rest is the substituent.
@pytest.mark.parametrize("smiles, ring_name, stem, expected", [
    # O1,C2,N3,N4,C5: the set {1,3,4} loses to {1,2,4} (N1,N2,C3,O4) unless the oxygen is placed at 1 first.
    ("Brc1nnc(C)o1", "1,3,4-oxadiazole", "1,3,4-oxadiazol", "5-methyl-1,3,4-oxadiazol-2-yl"),
    ("Brc1nnc(C)s1", "1,3,4-thiadiazole", "1,3,4-thiadiazol", "5-methyl-1,3,4-thiadiazol-2-yl"),
    # O1,N2,C3,C4,N5: the set {1,2,5} loses to {1,2,3} (N1,O2,N3).
    ("Brc1nonc1C", "1,2,5-oxadiazole", "1,2,5-oxadiazol", "4-methyl-1,2,5-oxadiazol-3-yl"),
    # S1,N2,C3,N4,C5: the first element and the set agree.
    ("Brc1nc(C)ns1", "1,2,4-thiadiazole", "1,2,4-thiadiazol", "3-methyl-1,2,4-thiadiazol-5-yl"),
])
def test_senior_heteroatom_takes_locant_one_before_the_lowest_set(monkeypatch, smiles, ring_name, stem,
                                                                  expected):
    """The decorated heteromonocycle substituent numbers the senior heteroatom 1, then the lowest set."""
    mol = Chem.MolFromSmiles(smiles)
    parent = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Br")
    attach = mol.GetAtomWithIdx(parent).GetNeighbors()[0].GetIdx()
    ring = tuple(mol.GetRingInfo().AtomRings()[0])
    frag = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() != parent]
    monkeypatch.setattr(rs, "identify_ring_system", lambda m, r: ring_name)
    monkeypatch.setitem(rs._PIN_HETEROARYL_STEMS, ring_name, stem)
    assert rs._decorated_heteroaryl_substituent_name(mol, frag, ring, attach) == expected


# --- the pseudoesters of one polyacid ---------------------------------------------------------------------------
#
# "Fully esterified acids derived from a single acid" (the Blue Book): "Fully esterified acids
# derived from a single acid are systematically named by placing the name(s) of the hydroxylic component... in front
# of the name of the acid component", 'dimethyl butanedioate (PIN)' (:31775). A pseudoester: the
# hydroxylic component an organyl of Group 13-15) is named as the ester: 'trimethylsilyl acetate'. The producer cut
# ONE group (the first match of the perception, so the atom order decided which) and left the other as an
# 'oxo[(...)oxy]methyl' prefix, which gave two names for one molecule; a bracket atom of a random SMILES (`[O]`)
# lost its hydrogen on the cut, and the acid fragment was unnameable, so the general engine named that spelling
# as a polyether of dioxo carbons.
ITACONATE = "CCCC[Sn](CCCC)(CCCC)OC(=O)CC(=C)C(=O)O[Sn](CCCC)(CCCC)CCCC"
_ITACONATE_NAME = "bis(tributylstannyl) methylidenebutanedioate"


def _spellings(smiles, count=8):
    mol = Chem.MolFromSmiles(smiles)
    return [smiles] + list(Chem.MolToRandomSmilesVect(mol, count, randomSeed=11))


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["best-effort"])
def test_a_polyacid_of_identical_pseudoester_groups_has_one_name_in_every_atom_order(tier):
    """Eight random spellings and the written one: one name, the ester of the whole acid."""
    namer = _tiered_namer("pin", tier)
    names = {smi: namer.name_tiered(smi)["name"] for smi in _spellings(ITACONATE)}
    assert set(names.values()) == {_ITACONATE_NAME}, names
    assert_full_rt(_ITACONATE_NAME, ITACONATE)


@pytest.mark.opsin_gate
def test_two_pseudoester_groups_of_different_organyls_are_cut_in_one_canonical_order():
    """The group cut first is the one of the lower canonical rank, not the first match of the atom order."""
    namer = _tiered_namer("pin", "best-effort")
    one = namer.name_tiered("C[Si](C)(C)OC(=O)CCC(=O)O[Sn](C)(C)C")["name"]
    other = namer.name_tiered("C[Sn](C)(C)OC(=O)CCC(=O)O[Si](C)(C)C")["name"]
    assert one == other


@pytest.mark.opsin_gate
def test_a_bracket_oxygen_of_a_pseudoester_takes_its_hydrogen_back_on_the_cut():
    """ (:32049) "the pseudoester is named as a traditional ester": the spelling RDKit writes
    for a tin ester (`[O]`, `[CH2]`) names the ester as the plain one does, not as '1-oxo-1-[(...)oxy]ethane'."""
    namer = _tiered_namer("pin", "best-effort")
    for smiles in ("CC(=O)[O][Sn]([CH2]CCC)([CH2]CCC)[CH2]CCC", "CC(=O)O[Sn](CCCC)(CCCC)CCCC"):
        assert namer.name_tiered(smiles)["name"] == "tributylstannyl acetate", smiles


# "Pseudoesters" (the Blue Book): the pseudoester R-CO-O-E "is named as a traditional ester, unless
# other names must be selected in accordance with the seniority order of classes, in decreasing order: salts > acids >
# anhydrides > esters"; the monobasic acids of P, As and (acyclic) B make anhydrides, (:35982, 'acetic
# dimethylarsinic anhydride (PIN)'). A group of such an E is therefore no ester of a fully esterified acid, and the
# polyacid keeps the name of one cut group, as before the polyacid name existed.
_ANHYDRIDE_E_DIESTERS = [
    # (smiles, best-effort name of the base)
    ("CP(C)OC(=O)CCC(=O)OP(C)C", "dimethylphosphanyl 4-[(dimethylphosphanyl)oxy]-4-oxobutanoate"),
    ("C[As](C)OC(=O)CCC(=O)O[As](C)C", "dimethylarsanyl 4-[(dimethylarsanyl)oxy]-4-oxobutanoate"),
    ("CB(C)OC(=O)CCC(=O)OB(C)C", "dimethylboranyl 4-[(dimethylboranyl)oxy]-4-oxobutanoate"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles, base_name", _ANHYDRIDE_E_DIESTERS)
def test_a_polyacid_of_anhydride_forming_pseudoester_groups_is_no_fully_esterified_acid(smiles, base_name):
    """The PIN tier abstains (no 'bis(...) butanedioate' pin_verified); best-effort keeps the one-group name."""
    pin = _tiered_namer("pin", "pin").name_tiered(smiles)
    assert pin["tier"] == "abstain" and "butanedioate" not in pin["name"], pin
    best = _tiered_namer("pin", "best-effort").name_tiered(smiles)
    assert (best["name"], best["tier"]) == (base_name, "systematic_verified"), best


@pytest.mark.opsin_gate
def test_five_pseudoester_groups_are_named_without_a_key_error():
    """The multiplier of five groups is in the table ('penta'), so the fully esterified acid is named and read back."""
    smiles = "C[Si](C)(C)OC(=O)c1cc(C(=O)O[Si](C)(C)C)c(C(=O)O[Si](C)(C)C)c(C(=O)O[Si](C)(C)C)c1C(=O)O[Si](C)(C)C"
    name = "pentakis(trimethylsilyl) benzene-1,2,3,4,5-pentacarboxylate"
    assert _tiered_namer("pin", "best-effort").name_tiered(smiles)["name"] == name
    assert_full_rt(name, smiles)


@pytest.mark.opsin_gate
def test_one_organyl_esterified_by_one_acid_has_one_name_in_every_spelling():
    """ "Polyesters formed from a single 'alcoholic' component" (the Blue Book): "When anions are
    identical functional class multiplicative nomenclature is used" ('ethane-1,2-diyl diacetate'); the one-cut name
    '(acetyloxy)dibutylstannyl acetate' is no name of that rule, so the ionic drawing, the bonded drawing and a bracket
    spelling of it keep one name."""
    namer = _tiered_namer("pin", "best-effort")
    names = {smi: namer.name_tiered(smi)["name"] for smi in (
        "CCCC[Sn+2]CCCC.CC(=O)[O-].CC(=O)[O-]", "CCCC[Sn](CCCC)(OC(C)=O)OC(C)=O",
        "CC(=O)[O][Sn]([CH2]CCC)([CH2]CCC)[O]C(C)=O")}
    assert set(names.values()) == {"1-{[(acetyloxy)di(butyl)stannyl]oxy}-1-oxoethane"}, names
