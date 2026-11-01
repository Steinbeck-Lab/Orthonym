"""LO4a class A5: a stereogenic unit outside the ring system no longer stops the bridged fused PIN builder.

``bridged_fused_pin.build_fused`` declined every molecule with a stereocentre or a stereogenic
double bond outside its ring system ("the package cites stereodescriptors on the ring-system
locants only"), so the pterin side chain of the fu4 record (``(1R,2S)-1,2-dihydroxypropyl``) left
three guanidine rows with no name at the default tier, and 594 corpus rows of the same shape with
them (``measurement lo4a``: 1,890 rows, default tier 950 -> 1,544 named, 0 lost, 0 tier drops,
0 wrong). The builder's prefix speller already cites a substituent's descriptors at the front of
its prefix; what was missing was the proof that it did so for EVERY unit, so the builder now
reads its own name back through OPSIN (full InChI, same canonical SMILES) and declines when the
name is not the molecule.

Blue Book (VERIFIED, ``the Blue Book Blue Book``):

* NAMING OF STEREOISOMERS (:44639),:44643: "In preferred IUPAC names, stereodescriptors
  are placed immediately at the front of the part of the name to which they relate. They are
  placed at the front of the complete name when related to the parent structure... When they
  relate to substituent groups, they are cited at the front of the corresponding prefix"
  ('[(1R)-1-chloropropyl]benzene (PIN)');:44674 "Multiple stereodescriptors are cited in the
  increasing order of their respective locants".
* INTRODUCTION (:50943): "Preferred IUPAC names (PINs) are not identified for the compounds
  in this Chapter [natural products]", so a steroid or a pterin is named by the systematic rules
  of to, here the fusion name with hydro prefixes and added hydrogen.
"""
import pytest
from rdkit import Chem

from tests.support.pin_tiers import assert_pin_at_both_tiers, name_breadth, name_default

pytestmark = pytest.mark.opsin_gate

# smiles, the PIN; each row has a stereogenic unit outside the ring system
ROWS = [
    # the three item rows (internal notes section 6, `_stereo_outside`): a pterin
    # whose 1,2-dihydroxypropyl side chain carries two stereocentres
    ("C[C@H](O)[C@H](O)[C@H]1CNC2=NC(N)=NC(=O)C2=N1",
     "(6R)-2-amino-6-[(1R,2S)-1,2-dihydroxypropyl]-7,8-dihydropteridin-4(6H)-one"),
    ("C[C@@H](O)[C@@H](O)[C@H]1CNC2=NC(N)=NC(=O)C2=N1",
     "(6R)-2-amino-6-[(1S,2R)-1,2-dihydroxypropyl]-7,8-dihydropteridin-4(6H)-one"),
    ("C[C@H](O)[C@H](O)[C@H]1CNC2=NC(N)=NC(=O)[C@]2(O)N1",
     "(4aS,6R)-2-amino-6-[(1R,2S)-1,2-dihydroxypropyl]-4a-hydroxy-5,6,7,8-tetrahydropteridin-4(4aH)-one"),
    # a stereocentre in the substituent, others in the ring system
    ("CC[C@@H](C)[C@H]1NC(=O)[C@H]2CCCN2C1=O",
     "(3R,8aR)-3-[(2R)-butan-2-yl]hexahydropyrrolo[1,2-a]pyrazine-1,4-dione"),
    # a stereocentre in the substituent only
    ("C[C@H](O)c1occ2c1C(=O)c1cc(O)cc(O)c1C2=O",
     "5,7-dihydroxy-1-[(1S)-1-hydroxyethyl]naphtho[2,3-c]furan-4,9-dione"),
    # a stereogenic double bond joined to the ring system (a ring locant), and one in the substituent
    ("O=C1/C(=C/c2ccccc2)Oc2cc(O)c3c(c21)OCO3",
     "(7Z)-7-benzylidene-4-hydroxy-2H-furo[3,2-e][1,3]benzodioxol-8(7H)-one"),
    ("CC/C=C1\\C[C@H]2CNc3ccccc3C(=O)N2C1",
     "(2E,11aS)-2-propylidene-1,2,3,10,11,11a-hexahydro-5H-pyrrolo[2,1-c][1,4]benzodiazepin-5-one"),
    ("CC(C)(O)/C=C/c1c2occc2cc2ccc(=O)oc12",
     "9-[(1E)-3-hydroxy-3-methylbut-1-en-1-yl]-7H-furo[3,2-g][1]benzopyran-7-one"),
    # a steroid skeleton with a side chain: the natural-product route does not name it, the fusion
    # name of is the systematic name
    ("CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C",
     None),
]
# the steroid row is named by the natural-product route ('cholestane'), not the builder: a control
CONTROL_ROWS = [row for row in ROWS if row[1] is None]
PIN_ROWS = [row for row in ROWS if row[1] is not None]


@pytest.mark.parametrize("smiles,pin", PIN_ROWS)
def test_stereo_outside_the_ring_system_is_cited_in_the_prefix(smiles, pin):
    """The PIN ships at both tiers (OPSIN reads it back to the full InChIKey of the input)."""
    assert_pin_at_both_tiers(smiles, pin)


def test_item_rows_producer_and_tiers():
    """The reproducers of the record: at the start of the item the default tier abstained and the
    best-effort tier named them `systematic_verified` through the general engine; both tiers now
    ship the builder's name as the PIN."""
    for smiles, pin in PIN_ROWS[:3]:
        d = name_default(smiles)
        assert (d.get("name"), d.get("tier"), d.get("producer")) == (
            pin, "pin_verified", "inner:tier_a_ring"), d
        b = name_breadth(smiles)
        assert (b.get("name"), b.get("tier")) == (pin, "pin_verified"), b


def test_natural_product_route_names_the_steroid_skeleton_systematically():
    # NPL 2026-11-01 (was: 'cholestane', pin_verified): INTRODUCTION (the Blue Book) "Preferred IUPAC names
    # (PINs) are not identified for the compounds in this Chapter", so the natural-product route offers the systematic
    # name of Chapters to (:50943); both tiers ship it as the PIN, OPSIN 2.9.0 reads it back to the full InChIKey.
    assert_pin_at_both_tiers(
        CONTROL_ROWS[0][0],
        "(8R,9S,10S,13R,14S,17R)-10,13-dimethyl-17-[(2R)-6-methylheptan-2-yl]hexadecahydro-1H-cyclopenta[a]phenanthrene")


def test_the_builder_names_the_pterin_with_every_descriptor():
    from orthonym.rules.bridged_fused_pin import build_fused
    got = build_fused(Chem.MolFromSmiles(PIN_ROWS[0][0]))
    assert got is not None and got[0] == PIN_ROWS[0][1], got


def test_a_name_that_does_not_read_back_is_declined(monkeypatch):
    """Fail closed: when OPSIN does not read the builder's name back to the molecule as drawn
    (or no OPSIN can be run), a molecule with a stereogenic unit outside the ring system is not
    named by the builder, so the older producers keep it."""
    import orthonym.rules.fused_pin_offer as fo
    from orthonym.rules.bridged_fused_pin import build_fused
    monkeypatch.setattr(fo, "_reads_back", lambda mol, name: False)
    assert build_fused(Chem.MolFromSmiles(PIN_ROWS[0][0])) is None


def test_a_molecule_without_outside_stereo_is_not_read_back(monkeypatch):
    """The proof is asked for only where a unit lies outside the ring system: a ring-only
    stereocentre is cited by the builder on the ring locants, as before."""
    import orthonym.rules.fused_pin_offer as fo
    from orthonym.rules.bridged_fused_pin import build_fused
    calls = []
    monkeypatch.setattr(fo, "_reads_back", lambda mol, name: calls.append(name) or False)
    got = build_fused(Chem.MolFromSmiles("C[C@@H]1CNC2=NC(N)=NC(=O)C2=N1"))
    assert got is not None and calls == [], (got, calls)


def test_stereo_outside_helper():
    from orthonym.rules.bridged_fused_pin import _stereo_outside, selection
    pterin = Chem.MolFromSmiles(PIN_ROWS[0][0])
    inside = Chem.MolFromSmiles("C[C@@H]1CNC2=NC(N)=NC(=O)C2=N1")
    for mol, expected in ((pterin, True), (inside, False)):
        system = selection.ring_system(mol)
        assert _stereo_outside(mol, {a: i for i, a in enumerate(sorted(system))}) is expected
