"""v52 a phase — seniority / multiplicative bridges,,.

These pin the Blue-Book PIN for eight target rows the bb_conformance oracle marks (PIN).

State at Phase-6 start (recon 2026-09-23, clean HEAD 706fbe754, re-verified in-session):
all 7 multiplicative rows (A1,A2,B,C,D,E1,E2) currently ABSTAIN (``name_compound`` returns
``"unknown organic compound"``), NOT ``RIGHT_MOL_NONPIN`` — each build task moves a row from
ABSTAIN to MATCH, so it is 0-wrong-safe (nothing to regress). Row F emits the non-PIN spelling
``(ethylsilyl)methanoic acid`` today. Every expected string below was verified to round-trip
through OPSIN 2.9.0 to the target molecule before this suite was written.
"""

import pytest
from orthonym import name_compound

# BB "Preferred IUPAC multiplicative names" (three conditions) +
# "MULTIPLICATION OF IDENTICAL SENIOR PARENT STRUCTURES":
# multiplicative nomenclature is senior to substitutive).
MULTIPLICATIVE_CASES = [
    # A. composite oxa bridges over ring units, [Task 3]
    ("O=C(O)c1ccc(OCCOc2ccc(C(=O)O)cc2)cc1",
     "4,4'-[ethane-1,2-diylbis(oxy)]dibenzoic acid"),
    ("O=C(O)c1ccc(OCCOCCOc2ccc(C(=O)O)cc2)cc1",
     "4,4'-[oxybis(ethane-2,1-diyloxy)]dibenzoic acid"),
    # B. carbon-anchored long oxa-diyl bridge, [Task 4]
    ("O=C(O)c1ccccc1CCOCCOCCOCCOCCc1ccccc1C(=O)O",
     "2,2'-(3,6,9,12-tetraoxatetradecane-1,14-diyl)dibenzoic acid"),
    # C. acyclic polyvalent hub, tris arms, [Task 5]
    ("O=P(O)(O)CP(CP(=O)(O)O)CP(=O)(O)O",
     "[phosphanetriyltris(methylene)]tris(phosphonic acid)"),
    # D. group-14 bis(silane) composite bridge [Task 6]
    ("[SiH3]CC[SiH2]CC[SiH3]",
     "[silanediyldi(ethane-2,1-diyl)]bis(silane)"),
    # E. substituted-central / concatenated benzene bridges, [Task 7]
    ("c1ccc(CC(Cc2ccccc2)(Cc2ccccc2)Cc2ccccc2)cc1",
     "1,1'-(2,2-dibenzylpropane-1,3-diyl)dibenzene"),
    ("c1ccc(CSC(SCc2ccccc2)c2ccccc2)cc1",
     "1,1'-[(phenylmethylene)bis(sulfanediylmethylene)]dibenzene"),
]


@pytest.mark.parametrize("smiles,expected", MULTIPLICATIVE_CASES)
def test_multiplicative_pin(smiles, expected):
    assert name_compound(smiles) == expected


# BB "SENIORITY ORDER FOR PARENT STRUCTURES": the senior parent
# structure has the maximum number of principal-characteristic-group suffixes /
# senior parent hydride.: the carboxy group may be attached to any
# atom, C or heteroatom, of any parent hydride -> silane keeps the acid as a
# -carboxylic acid suffix instead of demoting silyl to a prefix on methanoic acid.
#
# BUILT (Task F, 2026-09-23): ``rules.mononuclear_hydrides.
# name_mononuclear_hydride_added_carbon`` now builds the added-carbon
# ``-carboxylic acid`` suffix on the whole Group-14/-15 hub family (silane /
# germane / stannane / plumbane / bismuthane; P/As/Sb bare/chain stay with
# ``rules.phosphorus.name_phosphane_carboxylic_acid`` at priority 47.65) and
# names organyl hub substituents as prefixes (ethyl -> ethylsilanecarboxylic
# acid,. The producer runs at dispatch priority 47.66 — AHEAD of
# ORGANOMETALLIC@50 and GENERAL — so per the silane parent-hydride
# candidate is OFFERED and PREFERRED over the C1 methanoic-acid + silyl form.
# Both targets OPSIN-round-trip to the input InChIKey.
P44_CASES = [
    ("CC[SiH2]C(=O)O", "ethylsilanecarboxylic acid"),
]


@pytest.mark.parametrize("smiles,expected", P44_CASES)
def test_senior_parent_hydride_suffix(smiles, expected):
    assert name_compound(smiles) == expected


# v52 follow-up (2026-09-23, Batch B) — breadth builds on top of the Phase-6
# multiplicative handlers. Each row ABSTAINED before the build and each expected
# string was verified to round-trip through OPSIN 2.9.0 to the input InChIKey
# (0-wrong: ABSTAIN -> MATCH), so nothing regressed.
V52_FOLLOWUP_MULTIPLICATIVE_CASES = [
    # B1. trivalent-N hub, tris(methylene) over phosphonic-acid units — the N
    # sibling of the P hub (ATMP, aminotris(methylenephosphonic acid)).
    # / /; central group 'nitrilo' (Table 15.1).
    ("O=P(O)(O)CN(CP(=O)(O)O)CP(=O)(O)O",
     "[nitrilotris(methylene)]tris(phosphonic acid)"),
    # B2. 4-oxygen composite ether bridge [2,2,2] over two benzoic-acid rings.
    # / (cf. the Blue Book the same 3-oxa family).
    ("O=C(O)c1ccc(OCCOCCOCCOc2ccc(C(=O)O)cc2)cc1",
     "4,4'-[ethane-1,2-diylbis(oxyethane-2,1-diyloxy)]dibenzoic acid"),
    # B3. silanediyl hub with 3-carbon (propane-3,1-diyl) arms — the n-carbon
    # generalisation of the ethane-2,1-diyl arm. Per (the Blue Book) the
    # arm's free valence NEAREST the multiplied parent takes the LOW locant '1'
    # and is cited LAST -> '{stem}ane-{n},1-diyl' (cf. 'oxydi(tetradecane-14,1-
    # diyl)' the Blue Book, 'silanediyldi(ethane-2,1-diyl)' the Blue Book).
    ("[SiH3]CCC[SiH2]CCC[SiH3]",
     "[silanediyldi(propane-3,1-diyl)]bis(silane)"),
    # B3b. same hub, 4-carbon (butane-4,1-diyl) arms — the {n},1 convention past
    # n=3.
    ("[SiH3]CCCC[SiH2]CCCC[SiH3]",
     "[silanediyldi(butane-4,1-diyl)]bis(silane)"),
    # B4. tertiary central carbon (3 CH2-phenyl arms + 1 H): two benzene parents,
    # one benzyl substituent. / (2 parents max, the Blue Book).
    ("c1ccc(CC(Cc2ccccc2)Cc2ccccc2)cc1",
     "1,1'-(2-benzylpropane-1,3-diyl)dibenzene"),
    # B5. trivalent Si hub, tris(ethane-2,1-diyl) over three silane parents.
    # /; central group 'silanetriyl': Si=Si).
    ("[SiH](CC[SiH3])(CC[SiH3])CC[SiH3]",
     "[silanetriyltri(ethane-2,1-diyl)]tris(silane)"),
    # B5b. same trivalent Si hub, 3-carbon (propane-3,1-diyl) arms — the {n},1
    # arm-locant convention on the triyl sibling, parent-side
    # locant '1' cited last).
    ("[SiH](CCC[SiH3])(CCC[SiH3])CCC[SiH3]",
     "[silanetriyltri(propane-3,1-diyl)]tris(silane)"),
]


@pytest.mark.parametrize("smiles,expected", V52_FOLLOWUP_MULTIPLICATIVE_CASES)
def test_v52_followup_multiplicative_pin(smiles, expected):
    assert name_compound(smiles) == expected
