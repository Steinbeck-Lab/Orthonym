"""Partially saturated two-ring fused systems: hydro prefixes on the mancude parent, with every
indicated hydrogen of the parent cited, and the spellings that are not the PIN labelled below it.

 (the Blue Book-14607): "In preferred IUPAC names, all indicated hydrogen atoms must
be cited" -- '4,5,6,7-tetrahydroimidazo[4,5-c]pyridine' omitted the 1H of its mancude parent (the
standard InChIKey cannot see which N carries the H; the FixedH InChI of OPSIN's read-back can).
 (:11813): a benzene ring fused to a heteromonocycle of five or more members is a benzo
name ('2-benzofuran (PIN)... benzo[c]furan'; the catalogue gives '1H-benzimidazole'), and
 (:17026, '6,7-dihydro-5H-benzo[7]annulene (PIN)':17032): saturation is given by
'hydro' prefixes, not by indicated hydrogen on every saturated position.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import _independent_parse, name_is_rt_exact

PIN_ROWS = [
    ("c1nc2c([nH]1)CCNC2", "4,5,6,7-tetrahydro-1H-imidazo[4,5-c]pyridine"),
    ("Cn1cnc2c1CCNC2", "1-methyl-4,5,6,7-tetrahydro-1H-imidazo[4,5-c]pyridine"),
    ("c1n[nH]c2c1CCNC2", "4,5,6,7-tetrahydro-1H-pyrazolo[3,4-c]pyridine"),
    ("C1Nc2ccccc2N1", "2,3-dihydro-1H-1,3-benzimidazole"),
    ("CC1Nc2ccccc2N1", "2-methyl-2,3-dihydro-1H-1,3-benzimidazole"),
    ("C1Nc2ccccc2N1C", "1-methyl-2,3-dihydro-1H-1,3-benzimidazole"),
    ("CC1CCc2c(C=O)cncc21", "7-methyl-6,7-dihydro-5H-cyclopenta[c]pyridine-4-carbaldehyde"),
    ("c1cc2c([nH]1)CCNC2", "4,5,6,7-tetrahydro-1H-pyrrolo[3,2-c]pyridine"),
    # benzo names, in the catalogue now (they were declined descriptor names)
    ("c1ccc2cscc2c1", "2-benzothiophene"),
    ("c1ccc2sncc2c1", "1,2-benzothiazole"),
    ("c1ccc2[se]ccc2c1", "1-benzoselenophene"),
    # a seven-membered ring: two rings in a row are numbered without the lattice (quick-wins
    # F-Q3a); was the non-PIN '5H,6H,7H,8H,9H-cyclohepta[b]pyridine' below
    ("C1CCCc2ncccc2C1", "6,7,8,9-tetrahydro-5H-cyclohepta[b]pyridine"),
    # a ketone on the saturated ring: the indicated hydrogen of the mancude parent sits on the
    # suffix position, hydro prefixes before it:24768,:16880; was the
    # declined non-PIN '4-(prop-2-en-1-yl)-6H,7H-cyclopenta[c]pyridin-5-one' below)
    ("C=CCc1cncc2c1C(=O)CC2", "4-(prop-2-en-1-yl)-6,7-dihydro-5H-cyclopenta[c]pyridin-5-one"),
    ("C1Cc2cc[nH]c2C1", "1,4,5,6-tetrahydrocyclopenta[b]pyrrole"),   # LO3, was 4H,5H,6H-cyclopenta[b]pyrrole
]
# LO3 (rules/fused_pin_offer.py): '4H,5H,6H-cyclopenta[b]pyrrole' (the fused namer's spelling, indicated
# hydrogen on every saturated position, recorded as not a PIN::17026) gave way to the
# builder's PIN, a row of PIN_ROWS now: the mancude parent 'cyclopenta[b]pyrrole' (OPSIN: N1=C2C(C=C1)=CC=C2)
# has no indicated hydrogen to cite, and the hydro prefixes '1,4,5,6-tetrahydro' put the hydrogen on N1 as the
# input has it (the FixedH check of ``test_hydro_name_cites...``).
NON_PIN_NAMES_NOW_REPLACED = [
    ("C1Cc2cc[nH]c2C1", "4H,5H,6H-cyclopenta[b]pyrrole"),
]


def _row(smiles, tier):
    with jvm_slots(1, purpose="fused-hydro-ih"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


def _fixed_h(smi):
    return inchi.MolToInchi(Chem.MolFromSmiles(smi), options="/FixedH")


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", PIN_ROWS)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_hydro_name_cites_the_mancude_parent_and_its_indicated_hydrogen(smiles, name, tier):
    row = _row(smiles, tier)
    assert row.get("name") == name and row["tier"] == "pin_verified", (row.get("name"), row["tier"])
    assert name_is_rt_exact(name, smiles)
    # the indicated hydrogen sits on the input's N: the FixedH InChI of the read-back agrees
    assert _fixed_h(_independent_parse(name, smiles)) == _fixed_h(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", NON_PIN_NAMES_NOW_REPLACED)
def test_the_old_spelling_is_still_right_but_no_longer_shipped(smiles, name):
    """The descriptor spelling reads back to the input (so the offer's replacement is a choice
    between two right names) and neither tier ships it any more."""
    assert name_is_rt_exact(name, smiles)
    for tier in ("pin", "best-effort"):
        assert _row(smiles, tier).get("name") != name


@pytest.mark.parametrize("smiles,count", [
    ("c1cc[nH]c1", 1), ("c1ccoc1", 0), ("c1ccn2cccc2c1", 0), ("C1Cc2cnccc2C1", 1),
    ("C1CCCc2ncccc2C1", 1), ("C1Nc2ccccc2N1", 1), ("C1Cc2cc[nH]c2C1", 0), ("c1ccc2ccccc2c1", 0),
])
def test_indicated_hydrogen_count_of_the_mancude_skeleton(smiles, count):
    from orthonym.rules.fused_rings import _mancude_indicated_h_count
    mol = Chem.MolFromSmiles(smiles)
    ring = set().union(*[set(r) for r in mol.GetRingInfo().AtomRings()])
    assert _mancude_indicated_h_count(mol, ring) == count
