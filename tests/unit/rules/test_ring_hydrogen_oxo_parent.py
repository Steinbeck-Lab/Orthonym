"""The cyclic-oxo producer (``partial_saturation.name_cyclic_oxo_compound``) places hydrogen
only on ring atoms that can carry it, takes only mancude catalogue parents, and the
catalogue spells pyrrolizine with its indicated hydrogen; at both tiers.

- (the Blue Book) indicated hydrogen goes to a ring atom "connected to
  adjacent ring atoms by single bonds only, and carrying one or more hydrogen atoms";
   (:24693) added hydrogen to "a ring atom that is attached to adjacent ring atoms
  by single bonds only". A neutral nitrogen with three ring bonds carries neither, so
  'indolizin-3(2H)-one', not '...-3(2H,4H)-one'; 'imidazo[1,2-b]pyridazin-6(5H)-one', not
  '4,5-dihydroimidazo[1,2-b]pyridazin-6-one' (added hydrogen is "preferred over the use of
  nondetachable hydro prefixes",:24689). A boron, silicon or phosphorus ring atom with two
  ring bonds can carry it: 'borinin-2(1H)-one'.
- (:28410) a ketone is named from a MANCUDE parent: '4H-quinolizin-4-one', not
  the saturated catalogue entry's 'quinolizidin-4-one'.
- (:11628) "pyrrolizine (1H-isomer shown; the PIN is 1H-pyrrolizine)"; a fusion
  descriptor of that skeleton ('pyrrolo[1,2-a]pyrrole') is not the PIN, so a ketone the
  fusion producer names on it stays below the PIN label.
Every expected name reads back to the input's full InChIKey with OPSIN 2.9.0 (so do the old
spellings: only the rule tells them apart)."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots

PIN_ROWS = [
    ("O=C1CC=C2C=CC=CN12", "indolizin-3(2H)-one"),
    ("O=C1C=C2C=CC=CN2C1", "indolizin-2(3H)-one"),
    ("O=c1ccn2ccnc2[nH]1", "imidazo[1,2-a]pyrimidin-7(8H)-one"),
    ("O=c1ccc2nccn2[nH]1", "imidazo[1,2-b]pyridazin-6(5H)-one"),
    ("O=c1cc2ncccn2[nH]1", "pyrazolo[1,5-a]pyrimidin-2(1H)-one"),
    ("O=C1BC=CC=C1", "borinin-2(1H)-one"),
    ("O=c1cccc2ccccn12", "4H-quinolizin-4-one"),
    ("O=C1C=Cc2cccn21", "3H-pyrrolizin-3-one"),
    ("C1=Cn2cccc2C1", "1H-pyrrolizine"),                                          #:11628
    ("CC1=Cn2cccc2C1", "2-methyl-1H-pyrrolizine"),
]

#: names the producer already gave right
UNCHANGED = [
    ("O=C1NC(=O)c2ccccc21", "1H-isoindole-1,3(2H)-dione"),                       #:33853
    ("O=C1N(c2ccccc2)C(=O)c2ccccc21", "2-phenyl-1H-isoindole-1,3(2H)-dione"),
    ("O=c1ccoc2ccccc12", "4H-1-benzopyran-4-one"),
    ("O=c1cc[nH]c2ccccc12", "quinolin-4(1H)-one"),
    ("O=c1[nH]ccc2sccc12", "thieno[3,2-c]pyridin-4(5H)-one"),
    ("Cn1c(=O)c2c(ncn2C)n(C)c1=O", "1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione"),
]


def _row(smiles, tier):
    with jvm_slots(1, purpose="ring-hydrogen"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", PIN_ROWS + UNCHANGED)
def test_the_oxo_producer_places_hydrogen_only_where_it_can_sit(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])


@pytest.mark.opsin_gate
def test_a_fusion_descriptor_of_a_retained_skeleton_stays_below_the_pin():
    """'1H-pyrrolizin-2(3H)-one' is the rule's spelling; the fusion producer spells the
    parent 'pyrrolo[1,2-a]pyrrole', so the default tier declines and best-effort keeps it."""
    smiles = "O=C1Cc2cccn2C1"
    assert _row(smiles, "pin")["tier"] == "abstain"
    best = _row(smiles, "best-effort")
    assert (best.get("name"), best["tier"]) == ("1H-pyrrolo[1,2-a]pyrrol-2(3H)-one",
                                                "systematic_verified"), best


@pytest.mark.opsin_gate
def test_the_catalogue_mancude_test_is_computed_once_per_entry():
    """The oxo producer asks for every catalogue entry whether it is mancude, on every ring
    ketone; the answer depends on the entry alone, so a second ring ketone computes none of
    them again (caffeine took twice as long while each call recomputed about 200 entries)."""
    from orthonym.rules import partial_saturation
    caffeine = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
    assert _row(caffeine, "pin")["tier"] == "pin_verified"
    computed = partial_saturation._catalog_entry_is_mancude.cache_info().misses
    assert computed > 0
    assert _row(caffeine, "pin")["tier"] == "pin_verified"
    assert partial_saturation._catalog_entry_is_mancude.cache_info().misses == computed


def test_the_catalogue_mancude_test_runs_only_for_a_matching_entry(monkeypatch):
    """The mancude test of a catalogue entry runs only once the entry's skeleton has matched
    the ring system (two entries for 4H-quinolizin-4-one: '4H-quinolizine' and the
    saturated 'quinolizidine', which it rejects), not for all ~200 entries on the first
    ring ketone of a process (that cost about 80 ms of a 250 ms caffeine)."""
    from rdkit import Chem
    from orthonym.data.fused_heterocycles import FUSED_HETEROCYCLE_DATA
    from orthonym.rules import partial_saturation
    asked = []
    real = partial_saturation._catalog_entry_is_mancude
    monkeypatch.setattr(partial_saturation, "_catalog_entry_is_mancude",
                        lambda cs: asked.append(cs) or real(cs))
    mol = Chem.MolFromSmiles("O=c1cccc2ccccn12")
    ring = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    parents = partial_saturation._resolve_oxo_parent(mol, ring)
    assert [p[0] for p in parents] == ["4H-quinolizine"]
    assert sorted(FUSED_HETEROCYCLE_DATA[cs]["name"] for cs in asked) == [
        "4H-quinolizine", "quinolizidine"]


@pytest.mark.parametrize("smiles,mancude", [
    ("c1ccc2ccccc2c1", True),                 # naphthalene
    ("C1=Cn2cccc2C1", True),                  # 1H-pyrrolizine
    ("C1=CC=[N+]2C=CC=CC2=C1", True),         # quinolizinium: a charged atom in a double bond
    ("C1CCN2CCCCC2C1", False),                # quinolizidine
    ("O=c1c2ccccc2[nH]c2ccccc12", False),     # acridone: the ketone carbon has no ring double bond
    ("C1#CC=CC=CC=C1", False),                # a ring triple bond: undecidable, not a parent
])
def test_the_catalogue_mancude_test_is_the_engine_mancude_predicate(smiles, mancude):
    """``_catalog_entry_is_mancude`` is the engine's own predicate
    (``perception.mancude.is_mancude_ring_system`` over the ring atoms, at least one ring
    double bond): a charged ring atom and a ring triple bond are judged as everywhere else
    in the engine, the Blue Book, the maximum number of noncumulative
    double bonds)."""
    from orthonym.rules import partial_saturation
    assert partial_saturation._catalog_entry_is_mancude(smiles) is mancude


#: a descriptor of a skeleton the catalogue holds under a retained name, with no =X group
#: (a tautomer the catalogue does not hold): not the PIN:11628 "the PIN is
#: 1H-pyrrolizine"; the PIN is '3H-purine', read back FULL by OPSIN 2.9.0); the default tier
#: declines, best-effort keeps it.
RETAINED_SKELETON_DESCRIPTORS = [
    ("c1nc2cnc[nH]c2n1", "4H-imidazo[4,5-d]pyrimidine"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", RETAINED_SKELETON_DESCRIPTORS)
def test_a_descriptor_of_a_retained_skeleton_without_a_group_stays_below_the_pin(smiles, name):
    assert _row(smiles, "pin")["tier"] == "abstain"
    best = _row(smiles, "best-effort")
    assert (best.get("name"), best["tier"]) == (name, "systematic_verified"), best


#: the 3H tautomer of pyrrolizine is held by the catalogue ('3H-pyrrolizine', with OPSIN
#: 2.9.0's numbering): (the Blue Book) names by fusion the mancude systems
#: "that have no accepted retained or systematic name described in sections and
#: ", and Table 2.8 (:11628) retains pyrrolizine ("the PIN is 1H-pyrrolizine"), so
#: the PIN of the tautomer is the retained name with its indicated hydrogen,
#: not the descriptor '3H-pyrrolo[1,2-a]pyrrole'. OPSIN 2.9.0 reads both names below back to
#: the input's full InChIKey (an InChIKey, an InChIKey).
RETAINED_TAUTOMERS_THE_CATALOGUE_HOLDS = [
    ("C1=CC2=CC=CN2C1", "3H-pyrrolizine"),
    ("CC1=CC2=CC=CN2C1", "2-methyl-3H-pyrrolizine"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", RETAINED_TAUTOMERS_THE_CATALOGUE_HOLDS)
def test_a_tautomer_the_catalogue_holds_takes_the_retained_name(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), row


def test_the_polycomponent_producer_records_a_retained_skeleton(monkeypatch):
    """The polycomponent producer applies the same record as the two-component one. No
    retained skeleton of three or more rings reaches it today (every one has a carbocyclic
    ring, which it refuses), so the retained-name lookup is planted."""
    from rdkit import Chem
    from orthonym.metrics import provenance
    from orthonym.rules import fused_rings
    mol = Chem.MolFromSmiles("O1C=CC2=C1C=C1C(=N2)C=CO1")
    provenance.clear_provenance()
    assert fused_rings._try_polycomponent_fusion_name(mol) == "difuro[3,2-b:2',3'-e]pyridine"
    assert provenance.get_provenance().get("non_pin_fragments") == ()
    monkeypatch.setattr(fused_rings, "_retained_name_of_skeleton",
                        lambda mol, ring_atoms, descriptor: "planted-retained-name")
    provenance.clear_provenance()
    assert fused_rings._try_polycomponent_fusion_name(mol) == "difuro[3,2-b:2',3'-e]pyridine"
    assert provenance.get_provenance().get("non_pin_fragments") == (
        "difuro[3,2-b:2',3'-e]pyridin",)
