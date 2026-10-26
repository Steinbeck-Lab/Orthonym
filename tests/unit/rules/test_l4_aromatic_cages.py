"""Cages with aromatic rings get a verified name at the best-effort tier (lane L4).

The default tier of ``name_compound`` is the best-effort tier. Thirty PubChem rows that hold an aromatic ring
inside a bridged or spiro system were left without a name by it, for five reasons, each a producer that gave
up where a name that reads back exists:

* RDKit does not perceive the same aromaticity over a SMILES round trip for some ring systems (porphyrin-type
  and bridged ones), so the canonical re-parse that ``analyze_cage_universal`` works in did not match the
  molecule and the analysis refused (``_graph_match``).
* The spiro ring system of a molecule that has other rings (a biaryl link, a morpholine, a boronate) was not
  one of two components, so ``name_spiro_vonbaeyer`` found no monospiro junction; and the senior ring system
  of such a molecule is one of those other rings, so the spiro cluster was never tried
  (``rules.ring_cluster``, ``name_general_spiro``).
* A stereocentre of the cage sits on a primed locant of the spiro name, which the fallback never stated; the
  block that states it is offered first and kept when the name reads back to the full key.
* A phosphorane or sulfurane that joins a monocycle to a bicycle it also belongs to has five or six ring
  bonds, not four.
* The substituent speller of the engine cites no added hydrogen where the producer does
  (``4-oxoquinazolin-3-yl`` for ``4-oxoquinazolin-3(4H)-yl``), so the cluster-scoped spiro attempt declines a
  molecule with a pyridone-type ring outside the cage and names it as before.
* The spiro cluster of a molecule that carries a monocycle of its own is a parent only where every rung that
  names the molecule from its senior ring declined (the last rung of the producer), so a morpholine or a
  piperidine stays the parent at the valid tier as it is at the default.
* The fallback DOES change some names the main branch ships (5 dev2000 rows): where a cluster-scoped spiro name
  or a floor spelling that reads back replaces a floor name that did not, the name is more complete, or cites the
  skeletal replacement ('a') prefix of a polycycle before 'spiro' and its locants in ascending order
  .
* A fused aromatic macrocycle of 16 to 22 rings, or of more than 100 atoms, is above the best-effort
  ceilings: a late offer, after every producer found no name, lifts them (``lifted_ring_ceiling_scope``).

The names are von Baeyer polyenes, "Five-membered ring requirement", the Blue Book: "When
fusion names are not allowed, unsaturated von Baeyer ring system names are preferred IUPAC names") or the
component-name spiro names of "Monospiro ring systems with different ring components at least one of
which is a polycyclic ring system",:10260) and "... with a nonstandard spiro atom"
,:11226). None is claimed as a PIN: the fused or bridged-fused name is the PIN where there is one,
and this tier does not build it. Every expected name is read back by a fresh OPSIN call that does not go
through the engine (``tests.support.rt_assert``), and the PIN tier is asserted to be unchanged.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym, name_compound
from orthonym.assembly import fragment_naming as fn
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.metrics.provenance import best_effort_request_ctx
from orthonym.rules import ring_cluster
from orthonym.rules import spiro as _spiro
from orthonym.rules import vonbaeyer_universal as vbu
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

#: m-bridged benzene, a tricycle of 9 atoms whose perceived aromatic atoms differ between the SMILES and its
#: canonical re-parse (``C1=CC2c3cccc2c31`` reads back as the non-aromatic Kekule structure)
META_BRIDGED = "C1=CC2=C3C=CC2C3=C1"
META_BRIDGED_NAME = "tricyclo[4.3.0.0^2,7]nona-1(6),2,4,8-tetraene"
#: the same molecule written with aromatic atoms: RDKit perceives another ring as aromatic
META_BRIDGED_AROMATIC = "c12c3c(ccc1)C2C=C3"

#: a phosphorus macrocycle (the aromaticity of its SMILES does not survive the round trip)
PHOSPHA_MACROCYCLE = "C1C2=CC=C(P2C3=CC=CC=C3)CC4=CC=C(N4)/C=C/5\\S/C(=C\\C6=CC=C1N6)/C=C5"
PHOSPHA_MACROCYCLE_NAME = (
    "(1Z,17Z)-23-phenyl-21-thia-22,24-diaza-23-phosphapentacyclo[16.2.1.1^3,6.1^8,11.1^13,16]"
    "tetracosa-1,3,5,8,10,13,15,17,19-nonaene")

#: a benzo-bridged bicyclo[4.2.1]nonane whose bridging atom is the spiro atom of a thiadiazolidine dioxide,
#: with a pyrazolyl substituent (a ring joined by a single bond) and no stereocentre
SPIRO_NO_STEREO = "CN1C(=CC(=N1)C2=CC3=C(CC4CCC(C3)C45CN(S(=O)(=O)N5)CC(F)(F)F)C=C2)C6=CC=CC=C6F"
SPIRO_NO_STEREO_NAME = (
    "5'-[5-(2-fluorophenyl)-1-methyl-1H-pyrazol-3-yl]-2,2-dioxo-1-(2,2,2-trifluoroethyl)"
    "spiro[[2,1,3]thiadiazolidine-4,13'-tricyclo[8.2.1.0^3,8]trideca-3(8),4,6-triene]")
#: the same cage with a morpholine of its own (the senior ring system) and three stereocentres, two on primed
#: locants of the spiro name
SPIRO_STEREO = "C1C[C@@H]2CC3=C(C[C@H]1[C@]24CN(S(=O)(=O)N4)CC(F)(F)F)C=CC(=C3)OCCN5CCOCC5"
SPIRO_STEREO_NAME = (
    "(1'R,4R,10'S)-5'-[2-(morpholin-4-yl)ethoxy]-2,2-dioxo-1-(2,2,2-trifluoroethyl)"
    "spiro[[2,1,3]thiadiazolidine-4,13'-tricyclo[8.2.1.0^3,8]trideca-3(8),4,6-triene]")
#: a boronate on the benzo ring: the dioxaborolane ring is joined to the cage by a single bond
SPIRO_BORONATE = "B1(OC(C(O1)(C)C)(C)C)C2=CC3=C(C[C@H]4CC[C@@H](C3)[C@@]45CN(S(=O)(=O)N5)CC(F)(F)F)C=C2"
SPIRO_BORONATE_NAME = (
    "(1'S,4R,10'R)-2,2-dioxo-5'-(4,4,5,5-tetramethyl-1,3,2-dioxaborolan-2-yl)-1-(2,2,2-trifluoroethyl)"
    "spiro[[2,1,3]thiadiazolidine-4,13'-tricyclo[8.2.1.0^3,8]trideca-3(8),4,6-triene]")

#: a phosphorane that is the spiro atom of a dioxaphospholane and a bicycle it also belongs to
LAMBDA5_SPIRO = "CC1CP23(CC4=CC=CC=C4C1(C2)C)OC(C(O3)(C)C)(C)C"
LAMBDA5_SPIRO_NAME = (
    "1,4',4',5',5',11-hexamethyl-2'λ5-spiro[9-phosphatricyclo[7.2.1.0^2,7]dodeca-2,4,6-triene-9,2'-"
    "[1,3,2]dioxaphospholane]")
LAMBDA6_SPIRO = "C1=CS23(C=N1)(OP(=O)(O2)O3)C(=O)O"
LAMBDA6_SPIRO_NAME = (
    "1-carboxy-3-oxo-1λ6-spiro[2,4,5-trioxa-1-thia-3-phosphabicyclo[1.1.1]pentane-1,1'-[1,3]thiazole]")

#: a fused aromatic macrocycle of 18 rings (above the best-effort ceiling of 15), one biconnected ring system
BEYOND_RINGS = ("C1=CC=C2C(=C1)C=CC3=C2C=CC4=C5C6=C7C8C9=CC=CC=C9C(C7=NC=C6N1C5=C(C=C34)C2=C3C4C5=CC=CC=C5C(C3="
                "NC=C21)C1=CC=CC=C41)C1=CC=CC=C81")
BEYOND_RINGS_NAME_START = "32,35,38-triazahexadecacyclo[38.6.6.6^23,30.1^4,20."
#: a ring system of 102 atoms (above the best-effort ceiling of 100)
BEYOND_ATOMS = ("CCOC(=O)C1=CC2=C(C(=C1)C3=CC=C(C=C3)C#CC4=CC=C(C=C4)OCCCCCCCCCCCCCCCCCCCCCCCC#CC5=CC=C(C=C5)C#CC#CC6="
                "CC=C(C=C6)C#CCCCCCCCCCCCCCCCCCCCCCCCOC7=CC=C(C=C7)C#CC8=CC=C2C=C8)OC")
BEYOND_ATOMS_NAME_START = "ethyl 102-methoxy-17,80-dioxaoctacyclo[85.2.2.2^7,10."

#: a spiro cage with a quinazolinone substituent (a dev split): the producer writes the added hydrogen of the
#: substituent, ``4-oxoquinazolin-3(4H)-yl`` "'Added indicated hydrogen'", the Blue Book), and the
#: engine's own speller does not, so the cluster-scoped spiro attempt declines it and names it as before
QUINAZOLINONE_SPIRO = "CC(C)[C@H](O)c1nc2ccccc2c(=O)n1[C@H]1C[C@]2(OC1=O)c1ccccc1N1C(=O)C(C)(C)N(S(C)(=O)=O)[C@H]12"
QUINAZOLINONE_SPIRO_NAME = (
    "(4'S,6S,7S)-4'-{2-[(1S)-1-hydroxy-2-methylpropyl]-4-oxoquinazolin-3(4H)-yl}-5-(methanesulfonyl)-4,4-dimethyl-"
    "3,5'-dioxospiro[2,5-diazatricyclo[6.4.0.0^2,6]dodeca-1(12),8,10-triene-7,2'-oxolane]")
#: a spiro system with an aglycone (milestone1500): the descriptors and the oxo locants are cited in ascending
#: order with the primed locant between the numbers "Lowest set of locants", the Blue Book: "Primed
#: locants are placed immediately after the corresponding unprimed locants in a set arranged in ascending order"),
#: and the skeletal replacement ('a') prefixes of the polycycle are cited before 'spiro' with the components in
#: alphanumerical order of their carbon names, oxane before tetracyclo,
GLYCOSIDE_SPIRO = "CC1OC(O[C@@H]2C(=O)C[C@@H](C)O[C@@]23O[C@@H]2CC(=O)O[C@@H]2C2=C3C(=O)c3c(O)cccc3C2=O)CCC1N(C)C"
GLYCOSIDE_SPIRO_NAME = (
    "(2S,3R,6R,11'R,15'R)-3-{[5-(dimethylamino)-6-methyloxan-2-yl]oxy}-4'-hydroxy-6-methyl-2',4,9',13'-tetraoxo-"
    "12',16'-dioxaspiro[oxane-2,17'-tetracyclo[8.7.0.0^3,8.0^11,15]heptadeca-1(10),3,5,7-tetraene]")

#: a spiro cluster beside a monocycle that holds a nitrogen atom: the senior ring is the monocycle
#: "General criteria for determining ring seniority", the Blue Book, "(b) has at least one nitrogen atom",
#::19412), so the monocycle is the parent and the spiro system a substituent, at the default and the valid tier
SENIOR_MONOCYCLE = [
    ("C1COCCN1CCC1CCC2(CC1)OCCO2", "4-[2-(1,4-dioxaspiro[4.5]decan-8-yl)ethyl]morpholine"),
    ("CN1CCN(CC1)CC1CCC2(CC1)OCCO2", "1-[(1,4-dioxaspiro[4.5]decan-8-yl)methyl]-4-methylpiperazine"),
    ("C1CCNCC1CC1CC2(C1)CCOCC2", "3-[(7-oxaspiro[3.5]nonan-2-yl)methyl]piperidine"),
]
#: a spiro system of a bicycle with three ring oxygens and an oxane (dev2000): the skeletal replacement ('a') prefix
#: is cited before 'spiro' and the components are put in alphanumerical order of their carbon names, bicyclo
#: before oxane, the Blue Book;,:10322), not ``spiro[oxane-2,8'-3,7,9-trioxabicyclo...]``
TRIOXABICYCLO_SPIRO = "NC1CC(N)C(O)C(OC2OC(CO)C(O)C3OC4(OC(C(N)CO)C(O)C(O)C4O)OC23)C1O"
TRIOXABICYCLO_SPIRO_NAME = (
    "6'-(1-amino-2-hydroxyethyl)-2-[(3,5-diamino-2,6-dihydroxycyclohexyl)oxy]-3',4',5,5'-tetrahydroxy-4-"
    "(hydroxymethyl)-3,7,9-trioxaspiro[bicyclo[4.3.0]nonane-8,2'-oxane]")
#: the stereodescriptors of a spiro name in ascending order of locants, primed ones after the unprimed of the same
#: number, the Blue Book; the example of at:49838 is ``(1S,1'S,2S,4S,4'R)``)
INTERLEAVED_BLOCK = "C[C@@H]1CC/C(=C2\\C(=O)C[C@@H]3O[C@@]4(C[C@H]23)C[C@H](O)[C@H](CC[C@@H](C)O)O4)O1"
INTERLEAVED_BLOCK_NAME = (
    "(1S,3R,4'S,5R,5'S,6E)-4'-hydroxy-5'-[(3R)-3-hydroxybutyl]-6-[(5R)-5-methyloxolan-2-ylidene]-7-oxospiro"
    "[2-oxabicyclo[3.3.0]octane-3,2'-oxolane]")

READ_BACK = [
    (META_BRIDGED, META_BRIDGED_NAME),
    (META_BRIDGED_AROMATIC, META_BRIDGED_NAME),
    (PHOSPHA_MACROCYCLE, PHOSPHA_MACROCYCLE_NAME),
    (SPIRO_NO_STEREO, SPIRO_NO_STEREO_NAME),
    (SPIRO_STEREO, SPIRO_STEREO_NAME),
    (SPIRO_BORONATE, SPIRO_BORONATE_NAME),
    (LAMBDA5_SPIRO, LAMBDA5_SPIRO_NAME),
    (LAMBDA6_SPIRO, LAMBDA6_SPIRO_NAME),
    (QUINAZOLINONE_SPIRO, QUINAZOLINONE_SPIRO_NAME),
    (GLYCOSIDE_SPIRO, GLYCOSIDE_SPIRO_NAME),
    (TRIOXABICYCLO_SPIRO, TRIOXABICYCLO_SPIRO_NAME),
    (INTERLEAVED_BLOCK, INTERLEAVED_BLOCK_NAME),
]


def _default(smiles):
    with jvm_slots(1, purpose="l4-aromatic-cages"):
        return name_compound(smiles)


@pytest.mark.parametrize("smiles, expected", READ_BACK)
def test_the_default_tier_names_the_cage_and_the_name_reads_back(smiles, expected):
    assert _default(smiles) == expected
    assert_full_rt(expected, smiles)


@pytest.mark.parametrize("smiles, expected", SENIOR_MONOCYCLE)
@pytest.mark.parametrize("tier", ["default", "valid"])
def test_a_spiro_cluster_beside_the_senior_monocycle_is_a_substituent_at_every_tier(smiles, expected, tier):
    with jvm_slots(1, purpose="l4-aromatic-cages"):
        name = name_compound(smiles) if tier == "default" else name_compound(smiles, emit_tier="valid")
    assert name == expected
    assert_full_rt(name, smiles)


@pytest.mark.parametrize("smiles", [META_BRIDGED, SPIRO_NO_STEREO, SPIRO_STEREO, LAMBDA5_SPIRO, LAMBDA6_SPIRO,
                                    BEYOND_RINGS])
def test_the_pin_tier_still_declines_these_cages(smiles):
    """The names above are not PINs and no PIN producer changed: the PIN tier gives what it gave."""
    with jvm_slots(1, purpose="l4-aromatic-cages"):
        assert name_compound(smiles, emit_tier="pin") == ""


def test_the_names_of_the_cages_are_labelled_below_the_pin():
    with jvm_slots(1, purpose="l4-aromatic-cages"):
        for smiles in (META_BRIDGED, SPIRO_STEREO, LAMBDA5_SPIRO):
            row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
            assert row["is_pin"] is False and row["tier"] in ("systematic_verified", "best_effort"), row


def test_a_ring_system_beyond_the_ceilings_is_named_by_the_late_offer():
    """16 to 30 rings, or 101 to 200 atoms, of one biconnected ring system: refused by the best-effort
    ceilings of the first attempt, analysed once the ceilings are lifted after every producer declined."""
    with jvm_slots(1, purpose="l4-aromatic-cages"):
        rings = name_compound(BEYOND_RINGS)
        atoms = name_compound(BEYOND_ATOMS)
    assert rings.startswith(BEYOND_RINGS_NAME_START), rings
    assert atoms.startswith(BEYOND_ATOMS_NAME_START), atoms
    assert_full_rt(rings, BEYOND_RINGS)
    assert_full_rt(atoms, BEYOND_ATOMS)


#: a ring of 20 benzene and pyridine rings joined by single bonds: one biconnected ring system of 20 rings whose
#: main-ring search ran 1,405 s with the lifted ceilings before the scan for the main bridge was charged
CYCLOPARAPHENYLENE = (
    "C1=CC2=CC=C1C3=CC=C(C=C3)C4=CC=C(C=C4)C5=CC=C(C=C5)C6=CN=C(C=C6)C7=NC=C(C=C7)C8=CC=C(C=C8)C9=CC=C(C=C9)"
    "C1=CC=C(C=C1)C1=CC=C(C=C1)C1=CC=C(C=C1)C1=CC=C(C=C1)C1=CC=C(C=C1)C1=CN=C(C=C1)C1=NC=C(C=C1)C1=CC=C(C=C1)"
    "C1=CC=C(C=C1)C1=CC=C2C=C1")


def test_a_ring_system_the_lifted_ceilings_cannot_analyse_in_time_is_declined_in_time():
    """The late offer is bounded by the hang budgets, which count the scan for the main bridge in it."""
    import time
    assert vbu.ring_system_beyond_ceiling(Chem.MolFromSmiles(CYCLOPARAPHENYLENE))
    start = time.monotonic()
    with jvm_slots(1, purpose="l4-aromatic-cages"):
        assert name_compound(CYCLOPARAPHENYLENE) == ""
    assert time.monotonic() - start < 60


# --- the pieces -------------------------------------------------------------------------------------------------


def test_the_canonical_reparse_of_an_aromatic_cage_is_matched_on_the_graph():
    """RDKit reads ``C1=CC2c3cccc2c31`` (the canonical SMILES of the molecule) as another perception of it, so
    no match that keeps the aromatic bonds exists; the graph, element, charge, isotope and hydrogen count of
    every atom still match."""
    mol = Chem.MolFromSmiles(META_BRIDGED)
    canon = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    assert len(mol.GetSubstructMatch(canon)) == 0
    match = vbu._graph_match(mol, canon)
    assert sorted(match) == list(range(mol.GetNumAtoms()))
    for c_idx, o_idx in enumerate(match):
        assert canon.GetAtomWithIdx(c_idx).GetTotalNumHs() == mol.GetAtomWithIdx(o_idx).GetTotalNumHs()
    assert vbu._graph_match(mol, Chem.MolFromSmiles("C1CCC2CCCC2C1")) == ()
    pair = vbu._canonical_copy_and_match(mol)
    assert pair is not None and len(pair[1]) == mol.GetNumAtoms()


def test_the_stereodescriptors_of_a_primed_locant_given_as_a_string_are_ordered_with_the_unprimed():
    from orthonym.rules.stereochemistry import _composite_locant_sort_key as key
    items = [("5'", "S"), (6, "E"), (3, "R"), ("4'", "S"), (5, "R"), (1, "S")]
    assert [f"{loc}{cip}" for loc, cip in sorted(items, key=key)] == ["1S", "3R", "4'S", "5R", "5'S", "6E"]
    assert key(((4, "'"), "S")) == key(("4'", "S"))
    assert key(("4''", "S")) > key(("4'", "S")) > key((4, "S"))
    assert key(("3a", "S")) > key((3, "S"))
    fused = [("8a'", "R"), ("5'", "R"), ("4a'", "S"), ("8'", "S"), ("4'", "S")]
    assert [f"{loc}{cip}" for loc, cip in sorted(fused, key=key)] == ["4'S", "4a'S", "5'R", "8'S", "8a'R"]


def test_the_graph_match_compares_radical_electrons():
    radical = Chem.MolFromSmiles("[CH2]C1=CC=CC=C1")
    other = Chem.MolFromSmiles("C[C]1=CC=CC=C1")
    assert vbu._graph_match(radical, other) == ()
    assert vbu._graph_match(radical, Chem.MolFromSmiles("[CH2]C1=CC=CC=C1")) != ()


def test_the_graph_match_does_not_give_up_the_chirality_of_a_stereocentre():
    """With defined stereo the match keeps the chirality or there is none: the plain match could map a centre
    onto its mirror image."""
    mol = Chem.MolFromSmiles("C[C@H](O)C1=CC=CC=C1")
    mirror = Chem.MolFromSmiles("C[C@@H](O)C1=CC=CC=C1")
    assert vbu._graph_match(mol, mirror) == ()
    assert vbu._graph_match(mol, Chem.MolFromSmiles("C[C@H](O)C1=CC=CC=C1")) != ()
    assert vbu._graph_match(Chem.MolFromSmiles("CC(O)C1=CC=CC=C1"), Chem.MolFromSmiles("C[C@@H](O)C1=CC=CC=C1")) == ()
    assert vbu._graph_match(Chem.MolFromSmiles("CC(O)C1=CC=CC=C1"), Chem.MolFromSmiles("OC(C)C1=CC=CC=C1")) != ()


def test_the_graph_match_keeps_the_label_of_a_stereo_double_bond():
    one = Chem.MolFromSmiles("C/C=C/C1CCCCC1/C=C\\C")
    swapped = Chem.MolFromSmiles("C/C=C\\C1CCCCC1/C=C/C")
    assert vbu._graph_match(one, swapped) == ()
    assert vbu._graph_match(one, Chem.MolFromSmiles("C/C=C/C1CCCCC1/C=C\\C")) != ()


def test_the_graph_match_does_not_map_atoms_of_other_hydrogen_counts_onto_each_other():
    """The same skeleton with a hydrogen on another atom is another molecule: no mapping."""
    one = Chem.MolFromSmiles("C1=CC=CC2=C1CC2")
    other = Chem.MolFromSmiles("C1=CC=CC2=C1C=C2")
    assert vbu._graph_match(one, other) == ()


def _whole_ring_system(mol):
    return {a for ring in mol.GetRingInfo().AtomRings() for a in ring}


def test_a_spiro_atom_with_five_ring_bonds_is_one_only_inside_the_scope():
    mol = Chem.MolFromSmiles(LAMBDA5_SPIRO)
    assert _spiro.find_monospiro_separation_atom(mol) is None
    with ring_cluster.ring_cluster_scope(_whole_ring_system(mol)):
        found = _spiro.find_monospiro_separation_atom(mol)
    assert found is not None
    atom, parts = found
    assert mol.GetAtomWithIdx(atom).GetSymbol() == "P"
    assert sorted(len(p) for p in parts) == [4, 11]
    # the graph outside the scope is what it was
    assert _spiro.find_monospiro_separation_atom(mol) is None


def _largest_ring_cluster(mol):
    clusters = []
    for ring in mol.GetRingInfo().AtomRings():
        merged = set(ring)
        for c in [c for c in clusters if c & merged]:
            clusters.remove(c)
            merged |= c
        clusters.append(merged)
    return max(clusters, key=len)


@pytest.mark.parametrize("smiles, expected", [
    (QUINAZOLINONE_SPIRO, True), (SPIRO_STEREO, False), (GLYCOSIDE_SPIRO, False), (SPIRO_NO_STEREO, False)])
def test_a_pyridone_type_ring_outside_the_cluster_is_one_the_engine_speller_cannot_spell(smiles, expected):
    from orthonym.assembly.general_engine import _ring_needs_added_hydrogen
    mol = Chem.MolFromSmiles(smiles)
    assert _ring_needs_added_hydrogen(mol, _largest_ring_cluster(mol)) is expected


def test_the_rings_of_the_molecule_outside_the_ring_system_are_not_in_the_way_inside_the_scope():
    mol = Chem.MolFromSmiles(SPIRO_STEREO)
    morpholine = next(set(r) for r in mol.GetRingInfo().AtomRings()
                      if any(mol.GetAtomWithIdx(a).GetSymbol() == "O" for a in r) and len(r) == 6)
    assert _spiro.find_monospiro_separation_atom(mol) is None
    with ring_cluster.ring_cluster_scope(_whole_ring_system(mol) - morpholine):
        found = _spiro.find_monospiro_separation_atom(mol)
    assert found is not None and sorted(len(p) for p in found[1]) == [4, 12]


def test_the_ring_graph_is_the_same_outside_the_scope():
    """A biaryl-type bond joins two ring atoms of different rings and is in the graph as it always was."""
    mol = Chem.MolFromSmiles("c1ccccc1-c1ccccc1")
    ring_atoms, adj = ring_cluster.ring_atom_graph(mol)
    assert len(ring_atoms) == 12 and sum(len(v) for v in adj.values()) == 2 * 13
    with ring_cluster.ring_cluster_scope(ring_atoms):
        _, scoped = ring_cluster.ring_atom_graph(mol)
    assert sum(len(v) for v in scoped.values()) == 2 * 12


def test_the_scan_for_the_main_bridge_is_charged_only_inside_the_lifted_scope():
    assert vbu.lifted_ceilings_active() is False
    with vbu.lifted_ring_ceiling_scope():
        assert vbu.lifted_ceilings_active() is True
    assert vbu.lifted_ceilings_active() is False


def test_the_lifted_ceilings_apply_to_the_best_effort_tier_only():
    outer = best_effort_request_ctx.set(True)
    try:
        assert vbu.cage_caps() == (vbu.BEST_EFFORT_MAX_CAGE_ATOMS, vbu.BEST_EFFORT_MAX_CAGE_RINGS)
        with vbu.lifted_ring_ceiling_scope():
            assert vbu.cage_caps() == (vbu.LIFTED_MAX_CAGE_ATOMS, vbu.LIFTED_MAX_CAGE_RINGS)
            with vbu.pin_ceilings_scope():
                assert vbu.cage_caps() == (vbu.MAX_CAGE_ATOMS, vbu.MAX_CAGE_RINGS)
        assert vbu.cage_caps() == (vbu.BEST_EFFORT_MAX_CAGE_ATOMS, vbu.BEST_EFFORT_MAX_CAGE_RINGS)
    finally:
        best_effort_request_ctx.reset(outer)
    with vbu.lifted_ring_ceiling_scope():
        assert vbu.cage_caps() == (vbu.MAX_CAGE_ATOMS, vbu.MAX_CAGE_RINGS)


@pytest.mark.parametrize("smiles, beyond", [
    (BEYOND_RINGS, True), (BEYOND_ATOMS, True), (META_BRIDGED, False),
    ("C1=CC=C2C(=C1)C=C3C=CC=C4C3=C2C56C4(C7=CC=CC8=C7C5=CC9=CC=CC=C89)C1=CC=CC2=C1C6=CC1=CC=CC=C21", False),
])
def test_only_a_ring_system_between_the_two_pairs_of_ceilings_is_offered_the_lift(smiles, beyond):
    assert vbu.ring_system_beyond_ceiling(Chem.MolFromSmiles(smiles)) is beyond


def test_an_offer_is_made_once_for_the_molecule_that_name_was_called_with():
    assert fn.first_whole_molecule_offer("x", "CCO") and fn.first_whole_molecule_offer("x", "CCO")  # no scope
    fn.enter_name_scope("CCO")
    try:
        assert fn.first_whole_molecule_offer("x", "CCO") is True
        assert fn.first_whole_molecule_offer("x", "CCO") is False         # the second recovery
        assert fn.first_whole_molecule_offer("y", "CCO") is True          # another offer
        assert fn.first_whole_molecule_offer("x", "CCN") is False         # a part, not the molecule
    finally:
        fn.exit_name_scope()
    fn.enter_name_scope("CCO")
    try:
        assert fn.first_whole_molecule_offer("x", "CCO") is True          # the next molecule starts afresh
    finally:
        fn.exit_name_scope()
