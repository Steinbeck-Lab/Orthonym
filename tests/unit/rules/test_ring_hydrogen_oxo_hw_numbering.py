"""The cyclic-oxo producer (``partial_saturation.name_cyclic_oxo_compound``) numbers a
Hantzsch-Widman ring with two or more kinds of heteroatoms as its parent name does, so the
suffix and the added hydrogen carry the locants of that parent; at both tiers.

- (the Blue Book) "The locant '1' is given to a heteroatom that occurs
  first in the seniority sequence used for citation of the skeletal replacement ('a')
  prefixes. The numbering is then chosen to give lowest locants to heteroatoms considered as
  a set": '1,2,5-oxadiazole' (:14717), not a numbering with a nitrogen at 1 and the lower
  set {1,2,3}.
- (:24689) the ketone, thione or imine of a ring parent without indicated hydrogen
  takes 'added indicated hydrogen'; (:25041,:25042) the suffix (d) before the
  added hydrogen (e) for low locants: '1,3,4-oxadiazol-2(3H)-one', '1,2,5-oxadisilol-
  3(2H)-one'.
The producer's lowest heteroatom SET alone put a nitrogen or silicon at 1 and wrote the
locants of that numbering after the parent name ('1,3,4-oxadiazol-3(2H)-one', which OPSIN
cannot read, so the default tier declined; '1,2,5-oxadisilol-4-one' without its added
hydrogen, which OPSIN reads to the input). Every expected name reads back to the input's
full InChIKey with OPSIN 2.9.0."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots

ROWS = [
    ("O=c1[nH]nco1", "1,3,4-oxadiazol-2(3H)-one"),
    ("Cc1n[nH]c(=O)o1", "5-methyl-1,3,4-oxadiazol-2(3H)-one"),
    ("O=c1[nH]ncs1", "1,3,4-thiadiazol-2(3H)-one"),
    ("S=c1[nH]nco1", "1,3,4-oxadiazole-2(3H)-thione"),
    ("S=c1[nH]ncs1", "1,3,4-thiadiazole-2(3H)-thione"),
    ("O=c1cn[o][nH]1", "1,2,5-oxadiazol-3(2H)-one"),
    ("O=C1C=[SiH]O[SiH2]1", "1,2,5-oxadisilol-3(2H)-one"),
]

#: the senior heteroatom already took locant 1 under the lowest set
UNCHANGED = [
    ("O=c1[nH]cno1", "1,2,4-oxadiazol-5(4H)-one"),
    ("O=c1occo1", "2H-1,3-dioxol-2-one"),
    ("O=c1cccc[nH]1", "pyridin-2(1H)-one"),                                        #:21316
]


def _row(smiles, tier):
    with jvm_slots(1, purpose="ring-hydrogen"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", ROWS + UNCHANGED)
def test_a_heteromonocycle_ketone_is_numbered_from_its_senior_heteroatom(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])


def test_the_oxo_parent_numberings_put_the_senior_heteroatom_at_one():
    """Every numbering the producer offers for 1,3,4-oxadiazole has the oxygen at 1."""
    from rdkit import Chem

    from orthonym.rules.partial_saturation import _resolve_oxo_parent

    mol = Chem.MolFromSmiles("O=c1[nH]nco1")
    ring = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    oxygen = next(i for i in ring if mol.GetAtomWithIdx(i).GetSymbol() == "O")
    cands = _resolve_oxo_parent(mol, ring)
    assert cands and cands[0][0] == "1,3,4-oxadiazole", cands
    assert {loc[oxygen] for loc, _ in cands[0][1]} == {1}
