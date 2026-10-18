"""LO fix round A: a parent keeps its list name where the builder's name rests on a
numbering nothing but OPSIN gives, and the PINs the LO3 builder offer ships for four ketone / saturated
bicycle rows.

(1) Lycopodane (gold row P7A-LYCOPODANE, ``benchmarks/the gold set/packs/np_p7.json``). LO4b
(``) lets the bridged fused builder name a ring system with a tetravalent fusion atom
, the Blue Book 'naphtho[1,8a-b]azirine (PIN)'). The natural-product route offers the
builder's name before its own (``dispatch_table._systematic_pin_before_natural_product``, commit
``:,:50943, "Preferred IUPAC names (PINs) are not identified for the compounds in this
Chapter"), so 'lycopodane' became '(1S,8aR,9S,12aS)-dodecahydro-1,9-ethanopyrido[2,1-j]quinoline'. The
numbering of such a parent is OPSIN's alone (the engine's ``compute_fused_numbering`` has no grid for the
atom; ``periphery_numbering_ok`` only checks that the locants run round the periphery, LO4b record, Left 4),
and the list name is the book's own string "Semisystematic names of recommended parent
structures are listed in Table 10.1",:51375; Table 10.1 (a):51396 lists lycopodane). The book names no PIN
for either, so the route keeps the list name there. The exact-list parents named on a parent with no
such atom (kaurane, sparteine, trichothecane,...: gold rows corrected in `` and LO4c) are not
touched, and neither is a decorated parent.

(2) Four gold rows the LO3 offer now names with the Blue Book spelling (their gold strings were recorded as
not the PIN in ``default_tier_non_pin_rows.json``):

* '2H-1-benzopyran-3(4H)-one' (and its 6-bromo derivative): 'Added indicated hydrogen' (:24687),
  sentence:24689 "This method is preferred over the use of nondetachable hydro prefixes for
  preferred IUPAC names."; (:24872),:24876 "the use of nondetachable hydro prefixes is not used in
  preferred IUPAC names", example:24880 '1,4-dihydronaphthalen-1-one / naphthalen-1(4H)-one (PIN)'. The
  indicated hydrogen cannot sit on the ketone atom (3H-1-benzopyran is not a permissible mancude structure,
  checked below without any code of the engine; the book's wording:24860), so (:24841) ->
   (:24806) (1), (3): the lowest indicated hydrogen, 2H, and added indicated hydrogen for the
  3-one, (4H).
* '2-methylhexahydro-1H-pyrrolizine', '7-methyloctahydroindolizine': Table 2.8 (20) (:11622) 'indolizine
  (PIN)', (21) (:11628) 'pyrrolizine (1H-isomer shown; the PIN is 1H-pyrrolizine)'; (:17026)
  "Total hydrogenation is indicated by the appropriate multiplicative prefixes... but locants are omitted
  (see ", as '(3aR,7aS)-octahydro-1H-indole (PIN)' (:49612) and 'tetradecahydroacridine (PIN)'
  (:49652); 'pyrrolizidine' and 'indolizidine' are no parent of the book (0 hits).
"""
import itertools

import pytest
from rdkit import Chem

from orthonym.rules.bridged_fused_pin import has_unchecked_parent_numbering
from tests.support.pin_tiers import assert_pin_at_both_tiers, name_breadth, name_default
from tests.support.rt_assert import name_is_rt_exact

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

LYCOPODANE = "C1C[C@@H]2CC[C@@H]3CCC[C@@]24[C@@H]3CCCN4C1"
LYCOPODANE_BUILDER = "(1S,8aR,9S,12aS)-dodecahydro-1,9-ethanopyrido[2,1-j]quinoline"

# exact-list parents on a parent with no tetravalent fusion atom: the builder's name replaces the list name
SPARTEINE = "C1CCN2C[C@@H]3C[C@@H](CN4CCCC[C@@H]34)[C@H]2C1"
GRAYANOTOXANE = "C[C@@H]1[C@@H]2CCC(C)(C)[C@H]2CC[C@@]23C[C@@H](CC[C@@H]12)[C@@H](C)C3"
TRICHOTHECANE = "CC1CC[C@@]2(C)[C@@H](C1)O[C@@H]1CC[C@@]2(C)[C@@H]1C"

# a decorated lycopodane (13-one) is no list name: the inner path names it with the builder, as before
DECORATED = "C1C[C@@H]2CC(=O)[C@@H]3CCC[C@@]24[C@@H]3CCCN4C1"
DECORATED_PIN = "(1R,8aR,9R,12aS)-dodecahydro-1,9-ethanopyrido[2,1-j]quinolin-13-one"
# codeine: the natural-product route's bridged parent '[1]benzofuro[3,2-e]isoquinoline' has a tetravalent
# atom but a table numbering (source 'derived_table'), so the builder's name stands as before
CODEINE = "COc1ccc2c3c1O[C@H]1[C@@H](O)C=C[C@H]4[C@@H](C2)N(C)CC[C@@]341"


def _mol(smiles):
    return Chem.MolFromSmiles(smiles)


def test_the_unchecked_numbering_predicate_sees_only_the_tetravalent_parent():
    assert has_unchecked_parent_numbering(_mol(LYCOPODANE)) is True
    assert has_unchecked_parent_numbering(_mol(DECORATED)) is True
    for smiles in (SPARTEINE, GRAYANOTOXANE, TRICHOTHECANE, CODEINE, "C1CC2CCC1C2", "c1ccc2ccccc2c1", "CCO"):
        assert has_unchecked_parent_numbering(_mol(smiles)) is False, smiles
    assert has_unchecked_parent_numbering(None) is False


def test_a_table_10_1_parent_on_an_unchecked_numbering_keeps_its_list_name_at_the_default_tier():
    """The list name is emitted at the default tier and is no PIN: INTRODUCTION (the Blue Book;
    :50943) "Preferred IUPAC names (PINs) are not identified for the compounds in this Chapter." (lead decision
    2026-10-17): label systematic_verified, verified 'identity'. The name is the book's own string (Table 10.1
    (a),:51396), unchanged."""
    row = name_default(LYCOPODANE)
    assert row.get("name") == "lycopodane", row
    assert row.get("tier") == "systematic_verified" and row["verified"] == "identity", row
    assert row["is_pin"] is False, row


def test_the_breadth_tier_names_lycopodane_with_the_builder_name_which_reads_back():
    row = name_breadth(LYCOPODANE)
    assert row.get("name") == LYCOPODANE_BUILDER, row
    assert name_is_rt_exact(LYCOPODANE_BUILDER, LYCOPODANE)


@pytest.mark.parametrize("smiles", [SPARTEINE, GRAYANOTOXANE, TRICHOTHECANE])
def test_the_other_exact_list_parents_still_take_the_builder_name(smiles):
    row = name_default(smiles)
    assert row.get("tier") == "pin_verified", row
    assert row["name"] not in ("sparteine", "grayanotoxane", "trichothecane"), row
    assert name_is_rt_exact(row["name"], smiles), row


def test_a_decorated_parent_is_not_a_list_name_and_takes_the_builder_name():
    from orthonym.routing.dispatch_table import _keeps_list_parent_name
    assert _keeps_list_parent_name(_mol(LYCOPODANE), Chem.MolToSmiles(_mol(LYCOPODANE))) is True
    assert _keeps_list_parent_name(_mol(DECORATED), Chem.MolToSmiles(_mol(DECORATED))) is False
    assert _keeps_list_parent_name(_mol(CODEINE), Chem.MolToSmiles(_mol(CODEINE))) is False
    row = name_default(DECORATED)
    assert (row.get("name"), row.get("tier")) == (DECORATED_PIN, "pin_verified"), row
    assert name_is_rt_exact(DECORATED_PIN, DECORATED)


def test_codeine_keeps_the_bridged_fused_pin_of_the_natural_product_route():
    row = name_default(CODEINE)
    assert row.get("tier") == "pin_verified" and "4,12-methano[1]benzofuro[3,2-e]isoquinolin" in row["name"], row
    assert name_is_rt_exact(row["name"], CODEINE), row


# --- (2) the four rows --------------------------------------------------------------------------

def _permissible_indicated_hydrogen(atoms, bonds, hetero, position):
    """True when the ring carbon atoms left after one sp3 atom at ``position`` can all be paired by double
    bonds: the independent count (no engine code) behind 'nH-... is not a permissible structure'
    (the Blue Book)."""
    carbons = [a for a in atoms if a not in hetero]
    adj = {a: set() for a in carbons}
    for x, y in bonds:
        if x in adj and y in adj:
            adj[x].add(y)
            adj[y].add(x)

    def paired(nodes):
        nodes = set(nodes)
        if not nodes:
            return True
        first = sorted(nodes)[0]
        return any(paired(nodes - {first, other}) for other in adj[first] & nodes)

    return paired(set(carbons) - {position})


def test_3H_1_benzopyran_is_not_a_permissible_mancude_structure_but_2H_and_4H_are():
    atoms = ["O1", "C2", "C3", "C4", "C4a", "C5", "C6", "C7", "C8", "C8a"]
    # the bond list of 1-benzopyran: O1-C2-C3-C4-C4a-C5-C6-C7-C8-C8a-O1 and the fusion bond C4a-C8a
    bonds = [("O1", "C2"), ("C2", "C3"), ("C3", "C4"), ("C4", "C4a"), ("C4a", "C5"), ("C5", "C6"),
             ("C6", "C7"), ("C7", "C8"), ("C8", "C8a"), ("C8a", "O1"), ("C8a", "C4a")]
    ok = {c: _permissible_indicated_hydrogen(atoms, bonds, {"O1"}, c) for c in atoms if c != "O1"}
    assert ok["C2"] and ok["C4"], ok
    assert not ok["C3"], ok


def test_2H_pyrano_pyran_pattern_two_indicated_hydrogens_exist_where_the_count_says_so():
    # the 1H,5H pair of the ci-060 ring system (cyclopropa[c][1]benzofuran, OPSIN's numbering as in its
    # name): C7a has four ring bonds, so eight carbon atoms are paired; k = 2 is the least number of
    # indicated hydrogens, and no permissible set holds C4 (the 4-one needs added indicated hydrogen)
    bonds = [("C1", "C1a"), ("C1a", "C7a"), ("C7a", "C1"), ("C1a", "C2"), ("C2", "O3"), ("O3", "C3a"),
             ("C3a", "C7a"), ("C3a", "C4"), ("C4", "C5"), ("C5", "C6"), ("C6", "C7"), ("C7", "C7a")]
    carbons = ["C1", "C1a", "C2", "C3a", "C4", "C5", "C6", "C7"]
    adj = {a: set() for a in carbons}
    for x, y in bonds:
        if x in adj and y in adj:
            adj[x].add(y)
            adj[y].add(x)

    def paired(nodes):
        nodes = set(nodes)
        if not nodes:
            return True
        first = sorted(nodes)[0]
        return any(paired(nodes - {first, other}) for other in adj[first] & nodes)

    for k in (0, 1):
        assert not [c for c in itertools.combinations(carbons, k) if paired(set(carbons) - set(c))]
    sets = [c for c in itertools.combinations(carbons, 2) if paired(set(carbons) - set(c))]
    assert ("C1", "C5") in sets and not [c for c in sets if "C4" in c]


def test_cyclopenta_phenanthrene_takes_any_single_indicated_hydrogen_so_14H_accommodates_the_14_ol():
    # ci-061 (tests/integration/test_ci_benchmark.py): the steroid skeleton in its fixed numbering (a),
    # the Blue Book); an independent count with no engine code. Every one of the 17 carbon atoms can be the
    # one sp3 atom of the mancude parent, so 14H is permissible:24768, the 14-ol is on a fusion
    # atom that has no hydrogen in the parent), and with the double bond C8=C9 left, the other 14 atoms are the
    # hydro positions of '1,2,3,4,5,6,7,10,11,12,13,15,16,17-tetradecahydro-14H-'.
    edges = ("1-2 2-3 3-4 4-5 5-10 10-1 5-6 6-7 7-8 8-9 9-10 8-14 14-13 13-12 12-11 11-9 13-17 17-16 16-15 "
             "15-14").split()
    adj = {}
    for e in edges:
        a, b = map(int, e.split("-"))
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    def paired(nodes):
        nodes = set(nodes)
        if not nodes:
            return True
        first = min(nodes)
        return any(paired(nodes - {first, other}) for other in adj[first] & nodes)

    atoms = set(adj)
    assert len(atoms) == 17 and all(paired(atoms - {c}) for c in atoms)
    assert paired(atoms - {14, 8, 9})
    assert sorted(atoms - {14, 8, 9}) == [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 15, 16, 17]


@pytest.mark.parametrize("smiles,pin", [
    ("O1CC(CC2=CC=CC=C12)=O", "2H-1-benzopyran-3(4H)-one"),
    ("BrC=1C=C2CC(COC2=CC1)=O", "6-bromo-2H-1-benzopyran-3(4H)-one"),
    ("CC1CC2CCCN2C1", "2-methylhexahydro-1H-pyrrolizine"),
    ("CC1CCN2CCCC2C1", "7-methyloctahydroindolizine"),
])
def test_the_four_gold_rows_are_pins_at_both_tiers(smiles, pin):
    assert_pin_at_both_tiers(smiles, pin)
