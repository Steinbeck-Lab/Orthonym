"""A suffix on a partially saturated fused system takes its low locant before the hydro prefixes.

 NUMBERING (the Blue Book-3221): low locants go to, in decreasing seniority, (b) indicated
hydrogen (:3246), (c) principal characteristic groups and free valences (suffixes) (:3256), (d) added
indicated hydrogen (:3270), (e) hydro/dehydro prefixes and 'ene'/'yne' endings (:3288-:3289), (f)
detachable prefixes (:3301). On a symmetric mancude parent the suffix decides before the hydro
prefixes: 'quinoxaline-2-carboxylic acid', not '-3-'. Each expected name is read back by OPSIN 2.9.0
to the input's full InChIKey and FixedH InChI.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from tests.support.rt_assert import _independent_parse, name_is_rt_exact

ROWS = [
    # the review rows (a suffix the old key numbered after the hydro prefixes)
    ("OC(=O)c1cnc2CCCCc2n1", "5,6,7,8-tetrahydroquinoxaline-2-carboxylic acid"),
    ("Nc1cnc2CCCCc2n1", "5,6,7,8-tetrahydroquinoxalin-2-amine"),
    ("Cc1nc2CCCCc2nc1C(=O)O", "3-methyl-5,6,7,8-tetrahydroquinoxaline-2-carboxylic acid"),
    ("OC(=O)c1nncc2CCCCc12", "5,6,7,8-tetrahydrophthalazine-1-carboxylic acid"),
    ("OC(=O)c1[nH]cc2CCCCc12", "4,5,6,7-tetrahydro-2H-isoindole-1-carboxylic acid"),
    # one row per parent family
    ("OC(=O)c1ccc2CCCNc2n1", "5,6,7,8-tetrahydro-1,8-naphthyridine-2-carboxylic acid"),
    ("OC(=O)c1occ2CCCCc12", "4,5,6,7-tetrahydro-2-benzofuran-1-carboxylic acid"),
    ("OC(=O)c1cc2CCCCc2[nH]1", "4,5,6,7-tetrahydro-1H-indole-2-carboxylic acid"),
    ("OC(=O)c1nccc2CCCCc12", "5,6,7,8-tetrahydroisoquinoline-1-carboxylic acid"),
    ("OC(=O)c1n[nH]c2CCCCc12", "4,5,6,7-tetrahydro-1H-indazole-3-carboxylic acid"),
    ("OC(=O)c1nc2CCCCc2[nH]1", "4,5,6,7-tetrahydro-1H-1,3-benzimidazole-2-carboxylic acid"),
    # prefixes only: (f) the set ({5,6} < {7,8}), then (g) (:3307) the prefix cited first
    ("BrC1CCc2nccnc2C1Cl", "6-bromo-5-chloro-5,6,7,8-tetrahydroquinoxaline"),
    ("Clc1nc2CCCCc2nc1Br", "2-bromo-3-chloro-5,6,7,8-tetrahydroquinoxaline"),
    # controls, right before the fix
    ("OC(=O)c1ncc2CCCCc2n1", "5,6,7,8-tetrahydroquinazoline-2-carboxylic acid"),
    ("OC(=O)c1cnc2CCCCc2c1", "5,6,7,8-tetrahydroquinoline-3-carboxylic acid"),
]


def _row(smiles, tier):
    with jvm_slots(1, purpose="fused-hydro-suffix"):
        if tier == "pin":
            return Orthonym().name_tiered(smiles)
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


def _fixed_h(smi):
    return inchi.MolToInchi(Chem.MolFromSmiles(smi), options="/FixedH")


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", ROWS)
@pytest.mark.parametrize("tier", ["pin", "best-effort"])
def test_suffix_locant_before_hydro_locants(smiles, name, tier):
    row = _row(smiles, tier)
    assert row.get("name") == name and row["tier"] == "pin_verified", (row.get("name"), row["tier"])
    assert name_is_rt_exact(name, smiles)
    assert _fixed_h(_independent_parse(name)) == _fixed_h(smiles)
