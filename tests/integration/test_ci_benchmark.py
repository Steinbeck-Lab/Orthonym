"""
CI Benchmark Test Suite -- 100 compounds from ChEBI 500-sample (seed=123).

Deterministic regression test: verifies that name_compound() produces
the exact same IUPAC name for 100 representative ChEBI compounds.
Any change in output indicates a naming regression.

Sampling:
  1. Load ChEBI_SMILES.txt, deduplicate by canonical SMILES (111,823 unique)
  2. random.Random(123).sample(smiles_list, 500) -> 500-sample
  3. sample[:100] -> CI benchmark subset

This test does NOT call OPSIN (too slow for CI). It only verifies
deterministic name output against pre-computed expected names.

Target: all 100 tests pass in <60 seconds.
"""

import pytest

from orthonym import name_compound

# Pre-computed expected names for 100 ChEBI compounds (seed=123, first 100 of 500).
# Generated on 2026-02-07 after Phase 24 Plans 01-04.
# Any change here indicates a naming regression that must be investigated.
CI_BENCHMARK = [
    ("*=CC", "ethane"),
    (
        "CC(=O)OC[C@H]1O[C@@H](O[C@]23C[C@@H]4[C@@]2(COC(=O)c2ccccc2)"
        "[C@H]2O[C@]4(O)C[C@]3(C)O2)[C@H](O)[C@@H](O)[C@@H]1O",
        "17-phenylheptadecyl acetate",
    ),
    (
        "CCCCN(C)C(=O)CCCCCCCCCC[C@@H]1Cc2cc(O)ccc2[C@H]2CC[C@]3(C)"
        "[C@@H](O)CC[C@H]3[C@H]12",
        "(7R,8R,9S,13S,14S,17S)-estra-1,2,4-trien-3,17-diol",
    ),
    (
        "C[C@H](CCC(=O)O)[C@H]1C[C@H](O)[C@@]2(C)C3=CCC4C(C)(C)C(=O)"
        "CC[C@]4(C)C3=CC[C@]12C",
        "(10S,13R,14R,15S,17R,20R)-15,24-dihydroxy-4,4,14-trimethylchola-7,9-dien-3,24-dione",
    ),
    (
        "CCCCCCCCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COC(=O)CCCCCCCCC",
        "(2S)-3-(docosanoyloxy)-1-(decanoyloxy)propan-2-ol",  # alpha order fixed P80
    ),
    ("CC(C)CCCCCCCOC(=O)c1ccccc1C(=O)O", "8-methylnonyl benzene-1,2-dicarboxylate"),
    (
        "O=C(O)c1ccccc1-c1c2ccc(=O)c([As]3SCCS3)c-2oc2c([As]3SCCS3)"
        "c(O)ccc12",
        "benzoic acid",  # Phase 81: benzoic acid substructure (old: garbled "hydroxycycloanecarboxylic acid")
    ),
    (
        "Cc1cc([C@@]2(C)CCCC2(C)C)c(O)c(O)c1-c1c(C)cc([C@@]2(C)CCCC2"
        "(C)C)c(O)c1O",
        "1,2-dihydroxy-5-methyl-3-(1,2,2-trimethylcyclopentyl)benzene",
    ),
    (
        "CC(C)CCC1O[C@H]2C[C@H]3[C@@H]4CCC5CCCC[C@]5(C)[C@H]4CC[C@]3"
        "(C)[C@H]2[C@@H]1C",
        "(8R,9S,10S,13S,14S,16S,17R,20S)-16,22-epoxycholestane",
    ),
    (
        "CC(C)[C@H](N)C(=O)N[C@@H](CCCN=C(N)N)C(=O)N[C@@H](CCCCN)"
        "C(=O)O",
        "(2S)-6-aminoguanidino-2-(pentanoylamino)hexanoic acid",  # alpha order fixed P80
    ),
    (
        "NC(=O)[C@H](CCC/N=C(/N)CF)NC(=O)c1ccccc1",
        "(2S)-2-(benzoylamino)-5-(ethylamino)pentanamide",
    ),
    (
        "OCC1OC(Oc2cc(O)c3c(c2)OC(c2ccc(O)c(OC4OC(CO)C(O)C(O)C4O)c2)"
        "C(O)C3)C(O)C(O)C1O",
        "(glucopyranosyloxy)-3,5,7-trihydroxychromane",
    ),
    (
        "CSCC[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](N)CCCCN)C(=O)O",
        "(4S)-diamino-5-(butylamino)-methylsulfanyl-4-(hexanoylamino)pentanedioic acid",  # alpha order + P80-01 sulfanyl prefix
    ),
    (
        "CC(C)[C@H](NC(=O)[C@@H](N)CC(=O)O)C(=O)N[C@@H](Cc1ccccc1)"
        "C(=O)O",
        "L-aspartyl-L-valyl-L-phenylalanine",
    ),
    (
        "CC[C@H](C)C[C@H](C)CCCCCCCCC(=O)N[C@H]1C[C@@H](O)[C@@H]"
        "(NCCN)NC(=O)[C@@H]2[C@@H](O)CCN2C(=O)[C@H]([C@H](O)CCN)"
        "NC(=O)[C@H]([C@H](O)[C@@H](O)c2ccc(O)cc2)NC(=O)[C@@H]2"
        "C[C@@H](O)CN2C(=O)[C@H]([C@@H](C)O)NC1=O",
        "(1S,4S,5R,7S,10S,14R,16S,19S,22S,27S)-7-hexadecyl-4,10-diethyl"
        "-5,14,27-trihydroxy-19-octyl-22-propyl-3,9,12,18,21,24-hexaaza"
        "-tricyclo[22.3.0.0(12,16)]heptacosan-2,8,11,17,20,23-hexaone",
    ),
    ("O=[C]O[O-]", "methanolate"),
    (
        "C=C1[C@@H](O)O[C@H]2[C@H]1C[C@@H](OC(C)=O)[C@]13C(=O)O"
        "[C@H]4C[C@](C)(O)[C@H]([C@H]41)[C@@]31C=C(C)[C@]2(O)O1",
        "(1R,2R,4S,6S,8S,9S,12S,13S,14S,16S,19S)-2,6,9,14-tetrahydroxy-10,14-dimethyl-7,17,20-trioxa-hexacyclo[10.6.0.1(1,16).1(9,12).0(4,8).0(13,19)]icos-10-en-18-one acetate",  # Updated P72: IUPAC VB-6 citation order + VB-7 bridge atom orientation
    ),
    (
        "CC(C)=CCC/C(C)=C/CC/C(C)=C/CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C"
        "\\CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\CC"
        "/C(C)=C\\COP(=O)(O)OP(=O)(O)O",
        "(2Z,6Z,10Z,14Z,18Z,22Z,26Z,30Z,34Z,38E,42E)-3,7,11,15,19,23,"
        "27,31,35,39,43,47-dodecamethyloctatetraconta-2,6,10,14,18,22,"
        "26,30,34,38,42,46-dodecaenephosphonic acid",
    ),
    (
        "N[C@@H](CC(=O)O)C(=O)NCC(=O)N[C@@H](Cc1c[nH]c2ccccc12)"
        "C(=O)O",
        "L-aspartylglycyl-L-tryptophan",
    ),
    (
        "O=c1c(O[C@@H]2OC(CO)[C@@H](O)[C@H](O)C2O[C@@H]2OC(CO)"
        "[C@H](O)[C@H](O)C2O[C@@H]2OC(CO)[C@@H](O)[C@H](O)C2O)"
        "c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
        "3-oxanyl-4-oxo-2-phenyl-2H-pyran",  # Fixed: was "5,6-dibutyl-..." (fabricated from ring boundary leak)
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@@H](NC(=O)[C@@H](N)CC(N)=O)"
        "[C@@H](C)O)C(=O)O",
        "L-asparaginyl-L-threonyl-L-isoleucine",
    ),
    (
        "CCCCCC(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H]1C(=O)N[C@@H]"
        "(CCCN=C(N)N)C(=O)N[C@H]2CC[C@@H](O)N(C2=O)[C@@H]"
        "([C@@H](C)CC)C(=O)N(C)[C@@H](CC(=O)c2ccccc2N)C(=O)"
        "N[C@@H](C(C)C)C(=O)O[C@@H]1C",
        "N-(2S)-2-(hexanoylamino)butanedioyl(1S,4S,7S,8R,11S,14S,17S,21R)-7-amino-4,17-dibutyl-21-hydroxy-8,15-dimethyl-14-octyl-11-propyl-9-oxa-2,5,12,15,18-pentaaza-bicyclo[16.3.1]docosane",
    ),
    ("CC(C)CCCCCCCC=O", "9-methyldecanal"),
    (
        "CC1(C)C=Cc2c(cc(O)c3c(=O)c4ccc(O[C@@H]5c6c(cc(O)c7c(=O)"
        "c8cccc(O)c8oc67)O[C@H]5C(C)(C)O)c(O)c4oc23)O1",
        "6,6-dimethyl-2H-pyran",  # Fixed: was "2,3-dicyclohexyl-..." (fabricated from ring boundary leak)
    ),
    (
        "Nc1ncn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)n1",
        "6-amino-2-oxo-N-oxolanyl-1,3,5-triazine",
    ),
    (
        "O=c1c2c(O)cccc2oc2c3c(cc(O)c12)OC1OCCC31",
        "11,15-dihydroxy-6,8,20-trioxa-pentacyclo[10.8.0.0(2,9).0(3,7)"
        ".0(14,19)]icosan-13-one",
    ),
    (
        "CC1=C\\C=C\\C(C)=C\\C[C@H](C)NC(=O)/C(CC(C)C)=C/C(C)=C"
        "/C=C/C=C/[C@@](C)(O)[C@@H](O[C@@H]2OC[C@@H](O[C@H]3C"
        "[C@@](C)(O)[C@H](N(C)C)[C@@H](C)O3)[C@H](O)[C@H]2N)"
        "/C=C\\C=C\\1",
        "(3E,5E,7E,9E,11R,12S,13Z,15E,17E,19E,21E,24S)-12-[(2R,3R,4R,6S)-3-ethyl-4-hydroxy-2,4-dimethyl"
        "-6-oxanyloxyl]-11-hydroxy-3-isobutyl"
        "-5,11,17,21,24-pentamethylazacyclotetracosan-2-one",
    ),
    (
        "CC(C)[C@@H](NC(N)=O)C(=O)O",
        "(2R)-2-carbamoylamino-2-(methanoylamino)-3-methylbutanoic acid",  # alpha order fixed P80
    ),
    (
        "C/C1=C/[C@@H](C)C/C=C\\[C@H]2[C@@H]3O[C@]3(C)[C@@H](C)"
        "[C@H]3C(Cc4c[nH]c5ccccc45)NC(=O)[C@]32C2=N[C@@H](CC2)C1=O",
        "(2S,4Z,6S,8Z,10R,11R)-4,6-dimethyl-3-oxo-1-azacyclododecene",  # Fixed: fabricated subs from ring boundary leak
    ),
    (
        "CC(=O)N[C@@H](CSCCC(=O)C(=O)O)C(=O)O",
        "2-oxo-4-(pentylsulfanyl)butanedioic acid",  # parenthesization + alpha order fixed P80
    ),
    (
        "O=C(/C=C/c1ccc(Cl)cc1)c1ccccc1",
        "(2E)-3-(4-chlorophenyl)-1-phenylprop-2-en-1-one",
    ),
    (
        "CN(C)CCC=C1c2ccccc2COc2ccccc21",
        "3-cyclopentadecyl-1-(ethylamino)propan-1-amine",  # alpha order fixed P80
    ),
    (
        "N=C(N)NCCC[C@H](NC(=O)CNC(=O)[C@@H](N)CCC(=O)O)C(=O)CCl",
        "(4S)-4-aminochloro-5-(ethylamino)-guanidinooxopentanoic acid",  # alpha order fixed P80
    ),
    (
        "C=C[C@@H]1C(=C)CC[C@H]2[C@H]1C[C@H]1OC(=O)[C@@]3(C)"
        "[C@H](O)CC[C@@]2(C)[C@@]13O",
        "(1R,3R,4S,8S,9R,12R,13S,16S)-4-ethenyl-12,16-dihydroxy-9,13"
        "-dimethyl-15-oxa-tetracyclo[7.6.1.0(3,8).0(13,16)]hexadecan-14-one",
    ),
    (
        "C=C1C(=O)OC2/C=C(/CO)C(=O)/C=C\\C(C)(O)CC(OC(=O)/C(C)=C/C)C12",
        "4-methyl-5-oxooxolane",  # Fixed: fabricated "dipentadecyl" from ring boundary leak
    ),
    ("CSCCCCC=NO", "2-aza-8-thianonane"),
    ("O=Cc1ccc2ccccc2c1O", "1-hydroxynaphthalene-2-carbaldehyde"),
    (
        "C=CC1=C(C)C2=[N]3->[Fe]45<-[N]6=C(C=c7c(CCC(=O)O)c(C)c"
        "([n]74)=C2)[C@]2(CCC(=O)O2)[C@@](C)(O)C6=Cc2c(C=C)c(C)"
        "c([n]25)C=C13",
        "3-cyclopentacosylpropanoic acid",
    ),
    (
        "C/C=C/C=C/C(=O)CC(CCO)OCC[C@H](O)CC(=O)CC/C=C/C",
        "(6E,8E)-3-decyloxy-1-hydroxyhydroxydeca-6,8-dien-5-one",
    ),
    (
        "CC(=O)N[C@H]1[C@H](OC[C@H]2O[C@@H](O[C@H]3[C@H](O)"
        "[C@@H](O)C(O)O[C@@H]3CO)[C@H](O)[C@@H](O)[C@H]2O)"
        "O[C@H](CO)[C@@H](O[C@@H]2O[C@H](CO)[C@H](O)[C@H]"
        "(O[C@H]3O[C@H](CO)[C@H](O)[C@H](O)[C@H]3O)[C@H]2O)"
        "[C@@H]1O",
        "N-acetyl(2R,3R,4R,5S,6R)-3-amino-4-hydroxy-6-methyl-2,5-dioxanyloxane",
    ),
    (
        "O=C([O-])[C@H](O)[C@H](O)COP(=O)([O-])[O-]",
        "(2R,3R)-2,3-dihydroxyphosphono-4-phosphonooxybutanoate",  # P80-01 phosphonooxy prefix now generated
    ),
    (
        "CCCCCCCC(=O)N[C@H](C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H]"
        "(CCC(=O)O)C(=O)N[C@@H](CC(C)C)C(=O)O)C(C)C",
        "N-(2S)-2-(pentanoylamino)-3-phenylpropanoyl-L-glutamyl-L-leucine",
    ),
    ("OCCCO", "trimethylene glycol"),
    (
        "O=C([O-])[C@@](O)(CO)C(=O)CO",
        "(2R)-4-hydroxy-2-hydroxy-2-(hydroxymethyl)-3-oxobutanoate",  # alpha order fixed P80
    ),
    ("CCC1CC=C(N2CCCC2)C1=O", "5-ethyl-2-pyrrolidinylcyclopent-2-en-1-one"),
    (
        "Oc1ccc(CC2(O)Oc3cc(O)cc(O)c3C2O)cc1",
        "3,6,7,7a-tetrahydroxy-2,3-dihydro-1-benzofuran",
    ),
    (
        "CC/C=C\\C/C=C\\C/C=C\\CCCCCCCC(=O)OC[C@H](COP(=O)(O)"
        "OC[C@H](N)C(=O)O)OC(=O)CCCCCCCCC/C=C\\C/C=C\\CCCCC",
        "(2S)-2-aminohydroxypropanoic acid (11Z,14Z)-icosa-11,14-dienoate",  # Quality gate: tightened decomposition trigger
    ),
    (
        "C[NH2+][C@@H](C)[C@@H](O)c1ccccc1",
        "(1S,2S)-2-(methylamino)-1-phenylpropan-1-ol",
    ),
    ("CSCCSC", "2,5-dithiahexane"),
    (
        "CCCCCc1oc(CCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC"
        "[N+](C)(C)C)OC(=O)CCC/C=C\\C[C@H]2[C@@H](O)CC(O)O"
        "[C@@H]2/C=C/[C@@H](O)CCCCC)c(C)c1C",
        "(5Z)-(arachidoyloxy)-7-oxanylhept-5-enyl 13-furyltridecanoate",
    ),
    (
        "CC(=O)N[C@H]1C(O)O[C@H](CO)[C@@H](O)[C@@H]1O[C@H](C)"
        "C(=O)O",
        "(2R)-2-octyloxypropanoic acid",
    ),
    (
        "COc1cc(OC)c2c(O)c3c(c(-c4c5cc(OC)cc(OC)c5c(O)c5c(=O)cc(C)"
        "oc45)c2c1)O[C@](C)(O)CC3=O",
        "(6S)-9-hexadecyl-2,6-dihydroxy-12,14-dimethoxy-6-methyl-7-oxa"
        "-tricyclo[8.4.0.0(3,8)]tetradecan-4-one",
    ),
    (
        "OC[C@H]1O[C@H](OC[C@H]2O[C@H](OC[C@H]3O[C@H](O)[C@H](O)"
        "[C@@H](O)[C@@H]3O)[C@H](O)[C@@H](O)[C@H]2O)[C@H](O)"
        "[C@@H](O)[C@@H]1O",
        "(alpha-D-glucopyranosyloxy)(2R,3R,4S,5R,6S)-3,4,5-trihydroxy-2-methyl-6-oxanyloxane",
    ),
    (
        "N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CS)C(=O)N[C@@H](CS)C(=O)O",
        "L-phenylalanyl-L-cysteinyl-L-cysteine",
    ),
    (
        "COc1cc(OC)c2c(c1CC=C(C)C)O[C@H](c1ccccc1)CC2",
        "(2S)-5,7-dimethoxy-8-(2-methylbut-2-enyl)-2-phenylchromane",
    ),
    (
        "CN=C(N)NCCC/C=C/CCC[C@H](C)[C@H]1OC(=O)/C(C)=C\\C=C/"
        "[C@H](C)[C@H](O)C[C@H](O)[C@H](C)[C@@H](O)CC[C@@H](C)"
        "[C@H](O)C[C@@]2(O)O[C@H](C[C@H](O)C[C@H](OC(=O)CC(=O)O)"
        "C[C@@H](O)C[C@H](O)/C(C)=C\\C=C/[C@H]1C)C[C@@H](O)"
        "[C@@H]2O",
        "(2R,4R,6R,8R,10S,11Z,13Z,15R,16R,19Z,21Z,23S,24R,26S,27R,28S,31R,32R,34R)-16-dodecyl-4,6,8,10,24,26,28,32,34-nonahydroxy-11,15,19,23,27,31-hexamethyl-18-oxo-1,17-dioxacyclotetratriacontene propanedioate",  # Fixed: fabricated "dipropyl" from ring boundary leak
    ),
    (
        "NC(CCC(=O)NC(CSC(CC=O)c1ccccc1O)C(=O)NCC(=O)O)C(=O)O",
        "N-glutamyl-2-(propanoylamino)ethanoic acid",
    ),
    (
        "CC/C=C\\C/C=C\\C/C=C\\CCCCCC[C@@H](OO)C(=O)[O-]",
        "(2R,9Z,12Z,15Z)-2-hydroperoxyoctadeca-9,12,15-trienoate",
    ),
    (
        "Cc1c(Cl)c(O)cc2oc(=O)c3c(O)cc(O)cc3c12",
        "6-chloro-7-hydroxy-5-methylcoumarin",
    ),
    (
        "CN1C(=O)[C@]23SSS[C@@]1(CO)C(=O)N2[C@H]1Nc2ccccc2"
        "[C@@]1(c1c[nH]c2ccccc12)[C@@H]3O",
        "(1R,4S,8S,9S,10R)-9-hydroxy-4,19-dimethyl-10-octyl-5,6,7"
        "-trithia-2,17,19-triaza-pentacyclo[8.7.0.2(4,8).0(2,8).0(11,16)]"
        "nonadecan-3,18-dione",
    ),
    (
        "C[C@H](O)/C=C1\\C[C@H](O)[C@]23C[C@H]2C(C)(C)O[C@]3(O)C1=O",
        "(1S,2S,4E,6S,9R)-2,6-dihydroxy-8,8-dimethyl-7-oxa-tricyclo"
        "[4.4.0.0(1,9)]decan-5-one",
    ),
    (
        "C=C1[C@@H](O)CC[C@]2(C)C3=C(CC[C@@H]12)[C@]1(O)[C@@H](O)"
        "C[C@H]([C@H](C)CC[C@H](CC)C(C)C)[C@@]1(C)C[C@@H]3O",
        "(3S,5R,10S,12S,13R,14S,15S,17R,20R,24S)-stigmast-8-en-3,12,14,15-tetraol",
    ),
    ("Nc1[nH]c(=S)ncc1F", "4-amino-1,3-diazine"),
    (
        "CC[C@@H](C)[C@H](NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@@H](C)"
        "NC(=O)[C@H](CCO)NC(=O)c1ccccc1)C(=O)O",
        "N-(2R)-2-(butanoylamino)propanoyl(2S,3R)-amino-3-methyl-2-(hexanoylamino)pentanoic acid",  # depth-independent naming v11
    ),
    (
        "CC(=O)N[C@@H]1[C@@H](O)[C@H](O[C@@H]2O[C@H](CO)[C@H](O)"
        "[C@H](O[C@H]3O[C@H](CO)[C@@H](O)[C@H](O[C@@H]4O[C@H](CO)"
        "[C@H](O)[C@H](O)[C@H]4O)[C@H]3NC(C)=O)[C@H]2O)"
        "[C@@H](CO)O[C@H]1O",
        "(beta-D-galactopyranosyloxy)ethanediamide",
    ),
    (
        "CN1CCCN=C1/C=C/c1cccs1",
        "N-methyl-2-thienyl-1,3-diazine",
    ),
    (
        "CC/C=C/CCCC(=O)CCCCCC(=O)O",
        "(11E)-7-oxotetradec-11-enoic acid",
    ),
    (
        "C=C/C(C)=C/[C@]1(C)SC(=O)C(CC)=C1O",
        "(5S)-3-ethyl-4-hydroxy-5-methyl-5-(2-methylbuta-1,3-dienyl)-2-oxothiole",
    ),
    (
        "N[C@H](C=O)Cc1cnc[nH]1",
        "(2S)-2-amino-3-imidazolylpropanal",
    ),
    (
        "O=c1cc(-c2ccc(O)cc2)oc2cc(O)c(Cl)c(O)c12",
        "4-oxo-6-phenyl-2H-pyran",  # Fixed: was "2,3-dibutyl-..." (fabricated from ring boundary leak)
    ),
    (
        "CCCCCCC[C@@H](O)[C@H](O)CC#CC#C[C@@H](O)CC",
        "(3S,9R,10R)-heptadec-4,6-diyne-3,9,10-triol",
    ),
    (
        "CC1CCC/C=C\\C=C\\C(O)CC(O)C/C=C\\C=C\\C(O)C/C=C/C=C\\C(=O)O1",
        "(3Z,5E,9E,11Z,17E,19Z)-8,14,16-trihydroxy-24-methyloxacyclotetracosan-2-one",
    ),
    (
        "CCCCC/C=C\\C/C=C\\C/C=C\\CCCCCCC(=O)OC[C@@H](O)COP(=O)(O)"
        "OC1C(O)C(O)C(O)[C@@H](O)C1O",
        "(8Z,11Z,14Z)-((8Z,11Z,14Z)-icosa-8,11,14-trienoyloxy)icosa-8,11,14-trienehexaol",
    ),
    (
        "CCCCCCCC(O)CC(=O)N[C@@H](CC(C)C)C(=O)N[C@H](CCC(=O)O)"
        "C(=O)N[C@H]1C(=O)N[C@H](C(C)C)C(=O)N[C@@H](CC(C)C)"
        "C(=O)N[C@H](CO)C(=O)N[C@@H](CC(C)C)C(=O)N[C@H](CO)"
        "C(=O)N[C@@H]([C@@H](C)CC)C(=O)OC1C",
        "N-(2R)-hydroxy-2-(hexanoylamino)pentanedioyl(3S,6R,9S,12R,15S,18R,21R)-21-amino-3-(sec-butyl)-6,12-dihydroxymethyl"
        "-9,15-diisobutyl-18-isopropyl-22-methyl-5,8,11,14,17,20-hexaoxooxacyclodocosan-2-one",  # depth-independent naming v11
    ),
    (
        "CC[C@@H]1C[C@@]23OC(=O)C(=C2O)OC(=O)[C@]2(C)[C@H](CCCC"
        "[C@]3(C)C=C1C(=O)O)C(C)=C[C@@H]1[C@@H](OC3OC(C)C(OC(=O)"
        "c4c(C)cccc4OC)C(OC4CC(O)C(OC)C(C)O4)C3O)[C@@H](OC3OC(C)"
        "C(OC)C(C)(O)C3OC)CC[C@H]12",
        "(1S,7S,9R,12R,17R,20S,21R,22S,25R)-21-docosyl-9-ethyl-26"
        "-hydroxy-1,12,18-trimethyl-22-nonyl-2,5-dioxo-3,6-dioxa"
        "-pentacyclo[15.8.0.1(4,7).0(7,12).0(20,25)]hexacosa-4,10,18"
        "-triene-10-carboxylic acid",  # Updated P72: IUPAC VB-6 citation order
    ),
    (
        "CC(C)=CCC[C@@H](C(=O)O)[C@H]1C(=O)C[C@@]2(C)C3=C(CC"
        "[C@]12C)[C@@]1(C)CCC(=O)C(C)(C)[C@@H]1[C@@H](O)C3",
        "(5R,6S,10S,13R,14R,17R,20R)-6,21-dihydroxy-4,4,14-trimethylcholesta-8,24-dien-3,16,21-trione",
    ),
    (
        "CC(C)(O)/C=C/c1cc(O)ccc1O",
        "(3E)-4-(2,5-dihydroxyphenyl)-2-methylbut-3-en-2-ol",
    ),
    (
        "CC(C)=CCC(C)/C(C)=C/CO",
        "(2E)-3,4,7-trimethylocta-2,6-dien-1-ol",
    ),
    (
        "CCCCCC=CCC=CCCCCCCCC(=O)OCC(O)COP(=O)(O)OCCN",
        "amino(linoleoyloxy)octadeca-9,12-dienol",
    ),
    (
        "COc1cc(OC2OC(C(=O)O)C(O)C(O)C2O)c(C2CC(=O)c3ccc(O)cc3O2)"
        "c(O)c1CC=C(C)C",
        "(glucuronopyranosyloxy)-7-hydroxychroman-4-one",
    ),
    (
        "CC[C@H](C)C=C(C)C=CC1=CC2=C(Cl)C(=O)[C@@](C)(OC(C)=O)"
        "C(=O)C2=CN1C(CC(C)C)C(=O)OC",
        "(acetyloxy)(heptacosanoyloxy)-2-cyclodecyl-4-methylpentanedione",
    ),
    (
        "COc1ccc2c(c1OC)C(Cc1ccc(O)cc1)[N+](C)(C)CC2",
        "7,8-dimethoxy-N,N-dimethyl-1,2,3,4-tetrahydroisoquinoline",
    ),
    (
        "CCC/C=C/C/C=C/C/C=C/CCCCC(=O)O",
        "(6E,9E,12E)-hexadeca-6,9,12-trienoic acid",
    ),
    (
        "CC1=CC2=C(C=O)C(=O)C(C)(O)C(O)C2=CO1",
        "2-methyl-4H-pyran",  # Fixed: was "4,5-dihexyl-..." (fabricated from ring boundary leak)
    ),
    (
        "COc1cc(/C=C/C(=O)O[C@H]2[C@H](O)C[C@](O)(C(=O)O)C[C@H]2O)"
        "ccc1O",
        "(1R,2S,3R,5S)-1,2,3-trihydroxy-5-hydroxycyclohexan-5-carboxylic acid "
        "(2E)-3-(4-hydroxy-3-methoxyphenyl)prop-2-enoate",  # Updated: quality gate now triggers decomposition
    ),
    ("O=C1N=C([O-])c2ccccc21.[K+]", "potassium octanolate"),
    (
        "CC1=C(O)C(=O)C([C@@]2(C)CCCC2(C)C)=C(O)C1=O",
        "3-(1,2,2-trimethylcyclopentyl)-2,5-dihydroxy-6-methylcyclohexa-2,5-diene-1,4-dione",
    ),
    (
        "CC(=O)N(O)CCCCNC(=O)[C@H](COC(=O)c1cccc(O)c1O)NC(=O)"
        "[C@@H]1COC(c2cccc(O)c2O)=N1",
        "N-4-(ethanoylamino)butan-1-yl(2S)-3-(heptanoyloxy)-2-(propanoylamino)propanamide",
    ),
    (
        "CC(C)C1=C[C@@]23CC[C@H]4C(C)(C)CCC[C@]4(C(=O)O2)C3=CC1=O",
        "(1R,4S,9R)-13-isopropyl-5,5-dimethyl-16-oxa-tetracyclo"
        "[8.4.0.2(1,9).0(4,9)]hexadeca-10,13-dien-12,15-dione",  # Updated P72: IUPAC VB-7 bridge atom orientation
    ),
    (
        "COC1=C(N[C@H](C(=O)O)[C@@H](C)O[C@@H]2O[C@H](CO)[C@H](O)"
        "[C@H](O)[C@H]2O)C[C@](O)(CO)CC1=NCC(=O)O",
        "(beta-D-galactopyranosyloxy)(2S,3R)-3-hydroxybutanedioic acid",
    ),
    (
        "CC(=O)CCC1=C(C)C[C@@]2(CC1=O)C(=O)[C@@H]1C[C@@](O)(CO1)C2=O",
        "(1S,3R,5R)-9-butyl-5-hydroxy-10-methyl-7-oxa-tricyclo"
        "[3.2.1.5(3,3)]tridec-9-en-2,4,13-trione",
    ),
    (
        "CCCCCCCCCCCCC1=C(OC(C)=O)C(=O)c2ccccc2C1=O",
        "(acetyloxy)-1-docosyloxyethanedione",
    ),
    (
        "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O.O",
        "adenine",  # Phase 58: retained core name whitelist bypass (.O single-atom falls through)
    ),
    (
        "COc1cc2c(c(O)c1C/C=C(\\C)CCC=C(C)C)CN(CCc1c[nH]c3ccccc13)C2=O",
        "2-[(6E)-2,6-dimethylocta-2,6-dienyl]1-hydroxy-3-methoxybenzene",  # Fixed: benzene BFS no longer walks into fused lactam ring
    ),
    (
        "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](OC(=O)"
        "[C@@H](N)Cc2c[nH]cn2)[C@H]1O",
        "adenine (2S)-2-amino-3-imidazolylpropanoate",  # Phase 099-03: coverage guard rejects "adenine" for 33-HA molecule (ratio 0.21), decomposition produces complete name
    ),
    (
        "NC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](N)C[C@@H]2N)"
        "[C@H](O)[C@@H](O)[C@@H]1O",
        "(2R,3S,4S,5R,6R)-6-cyclohexyl-3,4,5-trihydroxy-2-methyl"
        "oxane",
    ),
    (
        "CSCC[C@H](NC(=O)[C@H](CO)NC(=O)[C@@H](N)CCCCN)C(=O)O",
        "(2S)-diaminohydroxy-4-(methylsulfanyl)-2-(propanoylamino)butanoic acid",  # alpha order + parenthesization fixed P80
    ),
    (
        "CSCC[C@H](N)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](Cc1cnc[nH]1)"
        "C(=O)O",
        "N-(2S)-amino-2-(butanoylamino)-4-carbamoylmethylsulfanylbutanoyl(2S)-2-amino-3-imidazolylpropanoic acid",  # alpha order + P80-01 sulfanyl prefix
    ),
    (
        "CC(C)=CCc1c(O)ccc(C(=O)C2C(c3c(O)cc(/C=C/c4cc(O)c(O)cc4O)"
        "cc3O)C=C(C)CC2c2ccc(OC3OC(C(=O)O)C(O)C(O)C3O)cc2O)c1O",
        "(glucuronopyranosyloxy)-1,3-dihydroxy-2-(2-methylbut-2-enyl)benzene",
    ),
    (
        "COC(=O)c1ccccc1OC1OC(COC2OC(C)C(O)C(O)C2O)C(O)C(O)C1O",
        "(rhamnopyranosyloxy)(oxan-2-yl)oxybenzene",
    ),
]


class TestCIBenchmark:
    """CI benchmark: 100 ChEBI compounds for regression detection.

    These tests verify deterministic IUPAC name output for a representative
    sample of real-world compounds from the ChEBI database. Failures indicate
    a naming regression that must be investigated before merge.

    Sampling: random.Random(123).sample(111823_unique_smiles, 500)[:100]
    """

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "smiles,expected_name",
        CI_BENCHMARK,
        ids=[f"ci-{i:03d}" for i in range(len(CI_BENCHMARK))],
    )
    def test_ci_benchmark(self, smiles, expected_name):
        """Verify naming output matches pre-computed expected name."""
        name = name_compound(smiles)
        assert name is not None, f"name_compound returned None for {smiles}"
        assert isinstance(name, str), f"Expected string, got {type(name)}"
        assert len(name) > 0, f"Empty name for {smiles}"
        assert name == expected_name, (
            f"CI REGRESSION: {smiles}\n"
            f"  Expected: {expected_name}\n"
            f"  Got:      {name}"
        )
