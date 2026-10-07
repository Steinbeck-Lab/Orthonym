"""Slice S2c-1 protection: names the slice must leave as they are, at both tiers.

Each is the Blue Book PIN spelling today and is reached by a code path S2c-1 changes: the
fusion numbering of systems with a five-membered ring (compute_fused_numbering), the
base-component choice of the two-ring fusion namer (fused_ring_selection), the catalogue and
its prefix-to-parent joins, and the bridged fused parent source (parents._parent_for_key).
Book lines (the Blue Book): '1,3-benzoxazole' and '1,3-benzothiazole':11815,
:43538), '2H-1-benzothiopyran' (:17006), 'pteridine', 'quinoline', 'isoquinoline', '1H-indole',
'9H-carbazole', 'phenanthridine' (Table 2.8), '9H-fluorene' (Table 2.7),
'1H-naphtho[2,3-d][1,2,3]triazole (PIN)' (:13980); the bridged rows are protection rows of the
S2 and S2b slices; 'pyrazino[2,3-d]pyridazine (PIN)' (:12403) is already the name for these two
SMILES spellings (the other spellings are S2c-1 targets)."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots

KEEP = [
    ('c1ccc2ocnc2c1', '1,3-benzoxazole'),
    ('c1ccc2scnc2c1', '1,3-benzothiazole'),
    ('Cc1nc2ccccc2s1', '2-methyl-1,3-benzothiazole'),
    ('c1ccc2nsnc2c1', '2,1,3-benzothiadiazole'),
    ('C1C=Cc2ccccc2S1', '2H-1-benzothiopyran'),
    ('C1COc2ccccc2N1', '3,4-dihydro-2H-1,4-benzoxazine'),
    ('c1ccc2c(c1)sc1ccsc12', 'thieno[3,2-b][1]benzothiophene'),
    ('c1cnc2ncnn2c1', '[1,2,4]triazolo[1,5-a]pyrimidine'),
    ('c1ncc2cocc2c1', 'furo[3,4-c]pyridine'),
    ('c1ccc2c(c1)ccc1occc12', 'naphtho[2,1-b]furan'),
    ('Cc1cc2c(ccc3ccccc32)o1', '2-methylnaphtho[2,1-b]furan'),
    ('c1ccc2c(c1)ccc1[nH]ccc12', '3H-benzo[e]indole'),
    ('c1ccc2cc3[nH]nnc3cc2c1', '1H-naphtho[2,3-d][1,2,3]triazole'),
    ('c1ccc2c(c1)[nH]c1ccccc12', '9H-carbazole'),
    ('C1c2ccccc2-c2ccccc21', '9H-fluorene'),
    ('c1ccc2c(c1)cnc1ccccc12', 'phenanthridine'),
    ('c1ccc2ncccc2c1', 'quinoline'),
    ('c1ccc2[nH]ccc2c1', '1H-indole'),
    ('c1ccc2occc2c1', '1-benzofuran'),
    ('c1ccc2cnccc2c1', 'isoquinoline'),
    ('c1cnc2ncncc2n1', 'pteridine'),
    ('C1CC2CC1c1ccoc12', '4,5,6,7-tetrahydro-4,7-methano-1-benzofuran'),
    ('C1CC2CC1c1cc[nH]c12', '4,5,6,7-tetrahydro-1H-4,7-methanoindole'),
    ('C1CC2CC1c1nc3ccccc3cc12', '1,2,3,4-tetrahydro-1,4-methanoacridine'),
    ('C1=CC2=CC3=C4C=CC(=C3C2=C1)C4', '4,7-methanocyclopenta[a]indene'),
    ('C12CC3CC(C1)CC2C3', 'octahydro-2,5-methanopentalene'),
    ('c12cnncc1nccn2', 'pyrazino[2,3-d]pyridazine'),
    ('c1nncc2nccnc12', 'pyrazino[2,3-d]pyridazine'),
    ('C1CC2CC1c1c2[nH]c2ccccc12', '2,3,4,9-tetrahydro-1H-1,4-methanocarbazole'),
]


def _row(smiles, tier):
    with jvm_slots(1, purpose="bf-s2c1"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,name", KEEP)
def test_names_s2c1_must_keep(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row.get("name"), row["tier"]) == (name, "pin_verified"), (row.get("name"), row["tier"])
