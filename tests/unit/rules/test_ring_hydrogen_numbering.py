"""A heteromonocycle with a =X group on a ring atom is numbered by the hydrogen of its mancude
parent (``heterocycles._ring_hydrogen_numbering``, consulted by the heterocycle assembly), at
both tiers.

 (the Blue Book-25044): "the starting point and direction of numbering of a
compound are chosen so as to give lowest locants to the structural features (if present)
considered successively in the order given": (b) heteroatoms, (c) indicated hydrogen, (d)
principal group named as suffix, (e) 'added indicated hydrogen', (f) 'hydro' prefixes, (g)
detachable prefixes. A parent with no indicated hydrogen takes 'added indicated hydrogen' for
a ketone,:24689;,:28410 "When no indicated hydrogen is present, the
methodology of 'added indicated hydrogen' is applied"), so the suffix locant (d) decides before
the added hydrogen (e): 'azet-3(2H)-one', not '2H-azet-3-one' (an indicated hydrogen azete does
not have; Note,:24691) and not 'azet-3(4H)-one'; '1,2-diazet-3(2H)-one', not
'1H-1,2-diazet-4-one'. The suffix (d) also decides before the hydro prefixes (f).
Every expected name reads back to the input's full InChIKey with OPSIN 2.9.0."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots

ROWS = [
    # the suffix (d) ranks before the hydro prefixes (f) and the added hydrogen (e) is cited
    # after it: the thione of these rings is the suffix class 16,
    # the Blue Book, before the class 19 amines:18192;:29504), so the
    # numbering is '2(1H)' with the amino group at 6, not '2,3-dihydro...-4-amine'.
    ("Nc1ccnc(=S)[nH]1", "6-aminopyrimidine-2(1H)-thione"),
    ("Nc1[nH]c(=S)ncc1F", "6-amino-5-fluoropyrimidine-2(1H)-thione"),
    ("O=C1C=NC1", "azet-3(2H)-one"),
    ("O=C1C=NN1", "1,2-diazet-3(2H)-one"),
    ("O=C1C=CC=NCC=C1", "azocin-5(2H)-one"),
]

#: names that were right before and stay right
UNCHANGED = [
    ("O=C1C=CC=NC1", "pyridin-3(2H)-one"),
    ("O=c1[nH]cccn1", "pyrimidin-2(1H)-one"),
    ("Nc1ccnc(=O)[nH]1", "6-aminopyrimidin-2(1H)-one"),
    ("O=c1cc[nH]c(=O)[nH]1", "pyrimidine-2,4(1H,3H)-dione"),
    ("Cn1cc(C(=O)O)c(=O)cc1", "1-methyl-4-oxo-1,4-dihydropyridine-3-carboxylic acid"),
    ("C=C1C=CC(N)=CN1", "6-methylidene-1,6-dihydropyridin-3-amine"),
    ("Cc1cc(=O)n(-c2ccccc2)[nH]1", "5-methyl-2-phenyl-1,2-dihydro-3H-pyrazol-3-one"),
    ("O=C1C=CC=COC1", "oxepin-3(2H)-one"),
    ("O=C1C=COCO1", "2H,4H-1,3-dioxin-4-one"),
]


def _row(smiles, tier):
    with jvm_slots(1, purpose="ring-hydrogen"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", ROWS + UNCHANGED)
def test_a_heteromonocycle_with_a_group_is_numbered_by_its_hydrogen(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])


def test_the_numbering_authority_leaves_a_ring_of_a_fused_system():
    """``_ring_hydrogen_numbering`` numbers a heteromonocycle only. The pyrimidine ring of
    caffeine is a ring of the purine system, whose hydrogen the fusion producers write
    ('1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione'); the authority leaves that ring to
    the cascade, as before the lane (it numbered the ring twelve ways on a purine ketone,
    about 16 ms of the first one in a process). A monocycle with the same groups is still
    numbered by it."""
    from rdkit import Chem
    from orthonym.rules import heterocycles
    caffeine = Chem.MolFromSmiles("Cn1cnc2c1c(=O)n(c(=O)n2C)C")
    six = next(set(r) for r in caffeine.GetRingInfo().AtomRings() if len(r) == 6)
    assert heterocycles._ring_hydrogen_numbering(caffeine, six) is None
    dimethyluracil = Chem.MolFromSmiles("Cn1ccc(=O)n(C)c1=O")
    ring = set(dimethyluracil.GetRingInfo().AtomRings()[0])
    assert heterocycles._ring_hydrogen_numbering(dimethyluracil, ring) is not None
