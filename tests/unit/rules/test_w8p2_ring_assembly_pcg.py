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
from orthonym.namer import name_compound, Orthonym

pytestmark = pytest.mark.unit

# Gate-off raw namer: proves the emitter builds the exact PIN string itself
# (not an OPSIN self-consistency artifact). The RT-gate FAILS-OPEN with no Java,
# so a source-level assertion is mandatory for anything that can drop/mutate atoms.
RAW = Orthonym(_disable_opsin_validity_gate=True)


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


# ---------------------------------------------------------------------------
# Task 2.4 — mixed prefix+suffix ring-assembly builder (P-28.2.1 + P-66).
#
# When a suffix-expressible PCG (-COOH/-CHO/-CN/-OH/-NH2) coexists with OTHER
# substituents on a biaryl, the SENIOR PCG is the suffix on the enclosed
# assembly parent and every other substituent is a prefix (P-14.4 / P-41
# seniority, reused from rules/seniority.py). The PCG ring is UNPRIMED so the
# suffix gets the lowest locant (P-31.1.4.3.4). PIN form has NO hyphen between
# the prefix block and the opening bracket (BB: 6,6'-dinitro[1,1'-biphenyl]-
# 2,2'-dicarboxylic acid; 4'-cyano[1,1'-biphenyl]-4-yl). All 6 targets below
# were derived from the Blue Book and RT-verified against OPSIN 2.9 to the
# authoritative input structures.
#
# Before this builder: the wrong-PIN family (chloro+ol, bromo+amine, fluoro+ol)
# shipped RT-valid non-preferred names (hydroxy/amino as prefix, no bracket);
# the fail-closed family (cyano+COOH, methyl+CHO) was `unknown`; nitro+COOH was
# a gate-off LEAK ("...4-unknown organic compoundyl...", nitro un-nameable by
# the local prefix namer). All become the PIN.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    # chloro + ol -> ol is the PCG (suffix), unprimed ring, lowest locant.
    ("ClC1=CC=C(C=C1)C1=CC=C(C=C1)O", "4'-chloro[1,1'-biphenyl]-4-ol"),
    # bromo + amine -> amine suffix.
    ("BrC1=CC=C(C=C1)C1=CC=C(C=C1)N", "4'-bromo[1,1'-biphenyl]-4-amine"),
    # fluoro + ol on the SAME ring -> ol=4 (para), fluoro=3 (lowest set).
    ("FC=1C=C(C=CC1O)C1=CC=CC=C1", "3-fluoro[1,1'-biphenyl]-4-ol"),
    # cyano + COOH on the SAME ring -> COOH senior over nitrile: COOH=3 suffix,
    # cyano=4 prefix.
    ("C(#N)C1=C(C=C(C=C1)C1=CC=CC=C1)C(=O)O",
     "4-cyano[1,1'-biphenyl]-3-carboxylic acid"),
    # methyl + CHO -> carbaldehyde suffix (methyl always prefix).
    ("CC1=CC=C(C=C1)C1=CC=C(C=C1)C=O",
     "4'-methyl[1,1'-biphenyl]-4-carbaldehyde"),
    # nitro + COOH -> carboxylic acid suffix, nitro prefix (closes gate-off leak).
    ("[N+](=O)([O-])C1=CC=C(C=C1)C1=CC=C(C=C1)C(=O)O",
     "4'-nitro[1,1'-biphenyl]-4-carboxylic acid"),
])
def test_mixed_prefix_suffix_biaryl(smiles, expected):
    # Gated public API (confirms OPSIN accepts the PIN) ...
    assert name_compound(smiles, style="pin") == expected
    # ... and the gate-off raw emitter builds the exact string (leak-proof).
    assert RAW.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", [
    # No suffix-expressible PCG present -> builder must NOT fire; the plain
    # prefix-only path stays (non-bracketed, no suffix).
    ("Clc1ccc(-c2ccccc2)cc1", "4-chloro-1,1'-biphenyl"),
    ("Cc1ccc(-c2ccccc2)cc1", "4-methyl-1,1'-biphenyl"),
])
def test_mixed_builder_no_overreach_without_pcg(smiles, expected):
    assert name_compound(smiles, style="pin") == expected
    assert RAW.name(smiles) == expected
