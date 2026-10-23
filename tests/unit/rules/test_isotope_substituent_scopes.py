"""Isotope descriptors inside a substituent prefix of a chain parent (PubChem 1M, lane S2: r546, r002).

 Names, (the Blue Book): "The name of an isotopically substituted compound is formed
by adding or inserting the nuclide symbol(s) enclosed in parentheses, preceded by any necessary
locant(s), letters, and/or numerals, before the part of the compound that is isotopically substituted.
Immediately after the parentheses there is neither space nor hyphen, except that when the name, or a part
of a name, includes a preceding locant, a hyphen is inserted."

* A benzo-fused heteroaryl prefix opens with its own locant set ('1,2-benzoxazol-3-yl'). Once
  ``496821116`` made the one-carbon sulfonamide the parent, a label on the benzo ring lived in a prefix
  of a chain parent and the descriptor had no slot there: the name before it was
  '3-(sulfamoylmethyl)(3a,4,5,6,7,7a-13C6)-1,2-benzisoxazole'. The slot is the opening mark of the
  prefix (``rules.isotopes._BRACKET_BENZO_LOCANT_RE``): '1-[(3a,4,5,6,7,7a-13C6)-1,2-benzoxazol-3-yl]
  methanesulfonamide'.
* Locants, Omission of locants (see also, (the Blue Book):
  "In preferred IUPAC names, locants are omitted if no locants are necessary in unmodified names. However, if
  isotopic modification requires a locant to specify its position, then all locants must be specified and
  none are omitted." The lone '(E)' of an aryl-vinyl chain prefix that has another double bond is read by
  OPSIN onto the labelled C1=C2 ('(1,2-2H2)' makes that bond stereogenic) unless it carries its locant:
  '(5E)'. The decorator names the skeleton as the unmodified name has it and re-runs it with the locant cited
  only when no descriptor reads back, so a labelled unit elsewhere ('(4-2H)aniline') leaves the prefix as it
  was ('[(E)-6-phenylhexa-1,5-dien-3-yl]'); a chain with one double bond keeps the bare descriptor.

Every expected name is read back by an independent OPSIN call to the input's full InChIKey, isotope
layer included (``tests.support.rt_assert.name_is_rt_exact``).
"""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

R546 = "[13CH]1=[13CH][13CH]=[13C]2[13C](=[13CH]1)C(=NO2)CS(=O)(=O)N"
R546_NAME = "1-[(3a,4,5,6,7,7a-13C6)-1,2-benzoxazol-3-yl]methanesulfonamide"
R002 = ("[2H]C=C([2H])C(C([2H])/C(=C/C1=C2C=CCCC2=C3CCC=CC3=C1)/[2H])NC4=CC5=C(C=C4)C6=CC=CC=C6C5"
        "(C7=CC=CC=C7)C8=CC=CC=C8")
R002_NAME = ("9,9-diphenyl-N-[(5E)-6-(3,4,5,6-tetrahydrophenanthren-9-yl)(1,2,4,5-2H4)hexa-1,5-dien-3-yl]"
             "-9H-fluoren-2-amine")


def _row(smiles, tier):
    namer = Orthonym(style="pin") if tier == "pin" else Orthonym(style="pin", **_emit_tier_flags(tier))
    return namer.name_tiered(smiles)


@pytest.mark.parametrize("tier", ["best-effort", "pin"])
@pytest.mark.parametrize("smiles,name", [
    (R546, R546_NAME),
    ("[13CH]1=[13CH][13CH]=[13C]2[13C](=[13CH]1)C(=NO2)CO", "[(3a,4,5,6,7,7a-13C6)-1,2-benzoxazol-3-yl]methanol"),
])
def test_a_labelled_benzo_heteroaryl_prefix_carries_its_descriptor(smiles, name, tier):
    res = _row(smiles, tier)
    assert res.get("name") == name, res.get("name")
    assert res.get("tier") == "pin_verified"
    assert name_is_rt_exact(name, smiles)


def test_the_labelled_parent_ring_keeps_its_descriptor_before_the_ring():
    # unchanged: the label on the parent ring itself, the slot after the hyphen of '-1,2-benzoxazole'
    smiles = "[13CH]1=[13CH][13CH]=[13C]2[13C](=[13CH]1)C(=NO2)C"
    res = _row(smiles, "best-effort")
    assert res.get("name") == "3-methyl(3a,4,5,6,7,7a-13C6)-1,2-benzoxazole"
    assert name_is_rt_exact(res["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    ("[2H]C=C([2H])C(C/C=C/c1ccccc1)Nc1ccccc1", "N-[(5E)-6-phenyl(1,2-2H2)hexa-1,5-dien-3-yl]aniline"),
    ("[2H]C=C([2H])C(C([2H])/C(=C/c1ccccc1)[2H])Nc1ccccc1",
     "N-[(5E)-6-phenyl(1,2,4,5-2H4)hexa-1,5-dien-3-yl]aniline"),
    (R002, R002_NAME),
])
def test_a_lone_stereo_bond_of_a_prefix_cites_its_locant_when_the_name_is_isotopic(smiles, name):
    res = _row(smiles, "best-effort")
    assert res.get("name") == name, res.get("name")
    assert name_is_rt_exact(name, smiles)
    # the bare '(E)' reads onto the labelled bond: not the input
    assert not name_is_rt_exact(name.replace("(5E)", "(E)"), smiles)


@pytest.mark.parametrize("smiles,name", [
    ("[2H]c1ccc(NC(C=C)C/C=C/c2ccccc2)cc1", "N-[(E)-6-phenylhexa-1,5-dien-3-yl](4-2H)aniline"),
    ("C=CC(C/C=C/c1ccccc1)Nc1ccc([13CH3])cc1", "4-(13C)methyl-N-[(E)-6-phenylhexa-1,5-dien-3-yl]aniline"),
])
def test_the_locant_is_cited_only_where_the_bare_descriptor_fails(smiles, name):
    # the label is on the aniline, the unit of the descriptor holds none: one descriptor per unit)
    res = _row(smiles, "best-effort")
    assert res.get("name") == name, res.get("name")
    assert name_is_rt_exact(name, smiles)


def test_the_unlabelled_twin_keeps_its_bare_descriptor():
    # the locant is cited only where the name is isotopic: the twin is unchanged
    res = _row("C=CC(C/C=C/c1ccccc1)Nc1ccccc1", "best-effort")
    assert res.get("name") == "N-[(E)-6-phenylhexa-1,5-dien-3-yl]aniline"


@pytest.mark.parametrize("smiles,name", [
    ("[2H]/C(=C(/[2H])\\c1ccccc1)/c1ccc(N)cc1", "4-[(E)-2-phenyl(2H2)ethenyl]aniline"),
    ("[2H]/C(=C(/[2H])\\c1ccccc1)/c1ccc(N(C)C)cc1", "N,N-dimethyl-4-[(E)-2-phenyl(2H2)ethenyl]aniline"),
])
def test_a_chain_with_one_double_bond_keeps_its_bare_descriptor_when_labelled(smiles, name):
    # nothing else in an ethenyl chain for a bare '(E)' to be read onto: unchanged
    res = _row(smiles, "best-effort")
    assert res.get("name") == name, res.get("name")
    assert name_is_rt_exact(name, smiles)


def test_a_benzo_fused_prefix_opens_a_slot_a_detachable_prefix_locant_does_not():
    from orthonym.rules.isotopes import _insertion_offsets
    for skel in ("1-(1,2-benzoxazol-3-yl)methanesulfonamide", "5-(1,3-benzodioxol-5-yl)pentan-2-one",
                 "2-(1,3-benzothiazol-2-yl)ethan-1-ol"):
        assert skel.index("(") + 1 in _insertion_offsets(skel), skel
    # the locants of '2,4-dimethyl' and '4-benzyl' belong to detachable prefixes, not to the ring
    for skel in ("1-(2,4-dimethylphenyl)ethan-1-one", "1-(4-benzylphenyl)ethan-1-one"):
        assert skel.index("(") + 1 not in _insertion_offsets(skel), skel
