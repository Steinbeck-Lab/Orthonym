"""v24 W8 P2 Tasks 2.2/2.3 — biphenyl ring-assembly PCG parent (P-44.2.1.5).

A ring assembly of two benzene rings (2 rings) is senior to a single benzene ring
for parent selection (P-44.2.1.5 "greater number of rings"), so the PCG is a
suffix on the assembly: biphenyl-4-ol -> [1,1'-biphenyl]-4-ol (not 4-phenylphenol),
benzidine -> [1,1'-biphenyl]-4,4'-diamine (P-62.2.4.1.1: benzidine is Type-2
retained, general-nomenclature only). The ring-assembly builder already emits
these (biphenyl-2-ol/-3-ol/-carbaldehyde/-4,4'-diol/-4-amine all work); the only
blockers were two hand-curated whole-molecule retained entries ('4-phenylphenol',
'benzidine') short-circuiting via the direct path — demoted via pin_list.json
pin:false (the benzophenone/acetophenone precedent). Every PIN RT-verified.
"""
import pytest
from orthonym.namer import name_compound

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles,expected", [
    ("Oc1ccc(-c2ccccc2)cc1", "[1,1'-biphenyl]-4-ol"),          # 2.2: was 4-phenylphenol
    ("Nc1ccc(-c2ccc(N)cc2)cc1", "[1,1'-biphenyl]-4,4'-diamine"),  # 2.3: was benzidine
])
def test_biphenyl_pcg_assembly_parent(smiles, expected):
    assert name_compound(smiles, style="pin") == expected


@pytest.mark.parametrize("smiles,expected", [
    # Internal-consistency guards: these already route through the assembly path
    # and must stay correct (the 2.2 demotion must not disturb them).
    ("Oc1ccccc1-c1ccccc1", "[1,1'-biphenyl]-2-ol"),
    ("Oc1cccc(-c2ccccc2)c1", "[1,1'-biphenyl]-3-ol"),
    ("Oc1ccc(-c2ccc(O)cc2)cc1", "[1,1'-biphenyl]-4,4'-diol"),
    ("Nc1ccc(-c2ccccc2)cc1", "[1,1'-biphenyl]-4-amine"),
    ("O=Cc1ccc(-c2ccccc2)cc1", "[1,1'-biphenyl]-4-carbaldehyde"),
    ("c1ccc(-c2ccccc2)cc1", "1,1'-biphenyl"),  # bare assembly parent
])
def test_biphenyl_assembly_siblings_unregressed(smiles, expected):
    assert name_compound(smiles, style="pin") == expected
