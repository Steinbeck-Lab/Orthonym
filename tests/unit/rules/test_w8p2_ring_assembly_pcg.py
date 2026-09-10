""" W8 P2 Tasks 2.2/2.3 — biphenyl ring-assembly PCG parent.

A ring assembly of two benzene rings (2 rings) is senior to a single benzene ring
for parent selection "greater number of rings"), so the PCG is a
suffix on the assembly: biphenyl-4-ol -> [1,1'-biphenyl]-4-ol (not 4-phenylphenol),
benzidine -> [1,1'-biphenyl]-4,4'-diamine: benzidine is Type-2
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


# (c) (BB 7623/7625): the terminal 'a' of a numerical multiplying prefix
# is elided before a vowel-initial suffix -- 'tetra'+'ol' -> 'tetrol',
# 'tetra'+'amine' -> 'tetramine' ([1,1'-biphenyl]-3,3',4,4'-tetramine, PIN;
# benzenehexol, not benzenehexaol). 'di'/'tri' carry no terminal 'a', and
# '-carboxylic acid'/'-carbaldehyde'/'-carbonitrile' are consonant-initial, so
# those suffixes take NO elision. All RT-verified against OPSIN 2.9.
@pytest.mark.parametrize("smiles,expected", [
    # tetra -> tetrol / tetramine (the a-elision gains)
    ("Oc1ccc(-c2c(O)cc(O)cc2O)cc1", "[1,1'-biphenyl]-2,4,4',6-tetrol"),
    ("Nc1ccc(-c2ccc(N)c(N)c2)cc1N", "[1,1'-biphenyl]-3,3',4,4'-tetramine"),
    ("Oc1cccc(O)c1-c1c(O)cccc1O", "[1,1'-biphenyl]-2,2',6,6'-tetrol"),
])
def test_biphenyl_multiplier_a_elision(smiles, expected):
    assert name_compound(smiles, style="pin") == expected


@pytest.mark.parametrize("smiles,expected", [
    # Must NOT elide: di/tri (no terminal 'a') and consonant-initial suffixes.
    ("Oc1ccc(-c2ccc(O)cc2)cc1", "[1,1'-biphenyl]-4,4'-diol"),
    ("Nc1ccc(-c2ccc(N)cc2)cc1", "[1,1'-biphenyl]-4,4'-diamine"),
    ("OC(=O)c1ccc(-c2ccc(C(=O)O)cc2)cc1",
     "[1,1'-biphenyl]-4,4'-dicarboxylic acid"),
])
def test_biphenyl_multiplier_no_over_elision(smiles, expected):
    assert name_compound(smiles, style="pin") == expected


# ---------------------------------------------------------------------------
# Task 2.4 — mixed prefix+suffix ring-assembly builder +.
#
# When a suffix-expressible PCG (-COOH/-CHO/-CN/-OH/-NH2) coexists with OTHER
# substituents on a biaryl, the SENIOR PCG is the suffix on the enclosed
# assembly parent and every other substituent is a prefix /
# seniority, reused from rules/seniority.py). The PCG ring is UNPRIMED so the
# suffix gets the lowest locant. PIN form has NO hyphen between
# the prefix block and the opening bracket (the Blue Book,6'-dinitro[1,1'-biphenyl]-
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
    # Gated public API (confirms OPSIN accepts the PIN)...
    assert name_compound(smiles, style="pin") == expected
    #... and the gate-off raw emitter builds the exact string (leak-proof).
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


# ---------------------------------------------------------------------------
# gate regression (introduced b0ec23eb, fixed here).
#
# b0ec23eb hoisted ``if any(s['name'] is None...): return None`` to the top of
# ``name_ring_assembly``. But ``s['name']`` is ONE producer's opinion
# (``_name_substituent``); ``_mixed_ring_assembly_prefix_name`` names a whole
# further set. Nitro gets None from the first and 'nitro' from the second, so
# the blanket check killed the assembly before the producer that could spell it
# ever ran -- regressing the P2-RING-ASSEMBLY-MIXED gold target to
# 'unknown organic compound'. The abstention belongs at the veto loop,
# immediately before ``s['name']`` is spelled into the prefix string.
#
# "Nitro and nitroso compounds" (the Blue Book): "Compounds containing the
# -NO2 or -NO group are named by means of the prefixes 'nitro' and 'nitroso',
# respectively, unless these groups can be named on the basis of the parent
# structures nitric and nitrous acids, NO2-OH and NO-OH, respectively, or their
# corresponding esters, anhydrides, amides and hydrazides." A -NO2 on a ring
# assembly therefore always HAS a preferred prefix form; abstaining was never
# nomenclaturally correct.
#
# "Ring assemblies with a single bond junction" (the Blue Book): "Each
# cyclic system is numbered in the traditional way, one with unprimed locants,
# the other with primed locants. Lowest possible locants must be used to denote
# the positions of attachment. These locants must be cited in preferred IUPAC
# names" -- the [1,1'-biphenyl] parent and its primed locant set.
# ---------------------------------------------------------------------------
def test_name_substituent_none_does_not_veto_a_group_another_producer_names():
    """Pin the producer disagreement itself, not just the emitted name.

    A whole-molecule assertion alone would go green again if someone moved the
    blanket check somewhere else that still ran before the mixed prefix
    builder. Asserting both producers directly pins the actual invariant:
    ``_name_substituent`` returning None is NOT evidence the group is
    un-nameable.
    """
    from rdkit import Chem
    from orthonym.rules import ring_assemblies as RA

    mol = Chem.MolFromSmiles(
        "[N+](=O)([O-])C1=CC=C(C=C1)C1=CC=C(C=C1)C(=O)O")
    assert mol is not None
    match = mol.GetSubstructMatch(Chem.MolFromSmarts("[N+](=O)[O-]"))
    assert len(match) == 3, "nitro SMARTS must match exactly the -NO2 atoms"
    attach = match[0]  # the nitrogen carries the bond to the ring
    assert mol.GetAtomWithIdx(attach).GetSymbol() == 'N'

    # Producer A cannot name it...
    assert RA._name_substituent(mol, list(match), attach) is None
    #... producer B can, which is exactly why the assembly must not abort.
    assert RA._mixed_ring_assembly_prefix_name(
        mol, list(match), attach, None) == 'nitro'


# Every category ``_mixed_ring_assembly_prefix_name`` claims to support, each
# paired with a carboxylic-acid suffix so the group is forced down the PREFIX
# path. A guard that gates N categories must be probed on all N: the blanket
# check would have taken out any of these whose group makes _name_substituent
# return None, and only the nitro row was covered before.
@pytest.mark.parametrize("smiles,expected", [
    ("O=Cc1ccc(cc1)-c1ccc(cc1)C(=O)O",
     "4'-formyl[1,1'-biphenyl]-4-carboxylic acid"),
    ("N#Cc1ccc(cc1)-c1ccc(cc1)C(=O)O",
     "4'-cyano[1,1'-biphenyl]-4-carboxylic acid"),
    ("Nc1ccc(cc1)-c1ccc(cc1)C(=O)O",
     "4'-amino[1,1'-biphenyl]-4-carboxylic acid"),
    ("Oc1ccc(cc1)-c1ccc(cc1)C(=O)O",
     "4'-hydroxy[1,1'-biphenyl]-4-carboxylic acid"),
    ("Clc1ccc(cc1)-c1ccc(cc1)C(=O)O",
     "4'-chloro[1,1'-biphenyl]-4-carboxylic acid"),
    ("Cc1ccc(cc1)-c1ccc(cc1)C(=O)O",
     "4'-methyl[1,1'-biphenyl]-4-carboxylic acid"),
    ("COc1ccc(cc1)-c1ccc(cc1)C(=O)O",
     "4'-methoxy[1,1'-biphenyl]-4-carboxylic acid"),
    # Same-kind pair: routes through the single-suffix branch, which never
    # consults s['name'] at all -- the blanket check was wrong there too.
    ("OC(=O)c1ccc(cc1)-c1ccc(cc1)C(=O)O",
     "[1,1'-biphenyl]-4,4'-dicarboxylic acid"),
])
def test_mixed_prefix_supported_set_every_category(smiles, expected):
    assert name_compound(smiles, style="pin") == expected
    assert RAW.name(smiles) == expected
