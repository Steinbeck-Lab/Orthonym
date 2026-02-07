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
        "(7R,8R,9S,13S,14S,17S)-estran-3,17-diol",
    ),
    (
        "C[C@H](CCC(=O)O)[C@H]1C[C@H](O)[C@@]2(C)C3=CCC4C(C)(C)C(=O)"
        "CC[C@]4(C)C3=CC[C@]12C",
        "(10S,13R,14R,15S,17R,20R)-15,24-dihydroxychol-7,9-dien-3,24-dione",
    ),
    (
        "CCCCCCCCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COC(=O)CCCCCCCCC",
        "(2S)-1-(decanoyloxy)-3-(docosanoyloxy)-2-hydroxypropane-1,3-dioate",
    ),
    ("CC(C)CCCCCCCOC(=O)c1ccccc1C(=O)O", "benzoic acid"),
    (
        "O=C(O)c1ccccc1-c1c2ccc(=O)c([As]3SCCS3)c-2oc2c([As]3SCCS3)"
        "c(O)ccc12",
        "benzoic acid",
    ),
    (
        "Cc1cc([C@@]2(C)CCCC2(C)C)c(O)c(O)c1-c1c(C)cc([C@@]2(C)CCCC2"
        "(C)C)c(O)c1O",
        "1,2-dihydroxy-5-methyl-3-octylbenzene",
    ),
    (
        "CC(C)CCC1O[C@H]2C[C@H]3[C@@H]4CCC5CCCC[C@]5(C)[C@H]4CC[C@]3"
        "(C)[C@H]2[C@@H]1C",
        "cholestane",
    ),
    (
        "CC(C)[C@H](N)C(=O)N[C@@H](CCCN=C(N)N)C(=O)N[C@@H](CCCCN)"
        "C(=O)O",
        "(2S)-2-(pentanoylamino)-6-diaminoguanidinohexanoic acid",
    ),
    (
        "NC(=O)[C@H](CCC/N=C(/N)CF)NC(=O)c1ccccc1",
        "(2S)-5-(ethylamino)-2-(heptanoylamino)pentanamide",
    ),
    (
        "OCC1OC(Oc2cc(O)c3c(c2)OC(c2ccc(O)c(OC4OC(CO)C(O)C(O)C4O)c2)"
        "C(O)C3)C(O)C(O)C1O",
        "2,4-dihydroxychromane",
    ),
    (
        "CSCC[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](N)CCCCN)C(=O)O",
        "(4S)-5-(butylamino)-4-(hexanoylamino)-diaminopentanedioic acid",
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
        "-tricyclo[22.3.0.012,16]heptacosan-2,8,11,17,20,23-hexaone",
    ),
    ("O=[C]O[O-]", "methanolate"),
    (
        "C=C1[C@@H](O)O[C@H]2[C@H]1C[C@@H](OC(C)=O)[C@]13C(=O)O"
        "[C@H]4C[C@](C)(O)[C@H]([C@H]41)[C@@]31C=C(C)[C@]2(O)O1",
        "icosyl acetate",
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
        "5,6-dibutyl-3-tetrahydropyranyl-4-oxo-2-phenyl-2H-pyran",
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
        "(3S)-4-(anilino)-3-(hexanoylamino)butanoic acid",
    ),
    ("CC(C)CCCCCCCC=O", "9-methyldecanal"),
    (
        "CC1(C)C=Cc2c(cc(O)c3c(=O)c4ccc(O[C@@H]5c6c(cc(O)c7c(=O)"
        "c8cccc(O)c8oc67)O[C@H]5C(C)(C)O)c(O)c4oc23)O1",
        "2,3-dicyclohexyl-6,6-dimethyl-2H-pyran",
    ),
    (
        "Nc1ncn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)n1",
        "6-amino-N-tetrahydrofuryl-2-oxo-1,3,5-triazine",
    ),
    (
        "O=c1c2c(O)cccc2oc2c3c(cc(O)c12)OC1OCCC31",
        "11,15-dihydroxy-6,8,20-trioxa-pentacyclo[10.8.0.02,9.03,7"
        ".014,19]icosan-13-one",
    ),
    (
        "CC1=C\\C=C\\C(C)=C\\C[C@H](C)NC(=O)/C(CC(C)C)=C/C(C)=C"
        "/C=C/C=C/[C@@](C)(O)[C@@H](O[C@@H]2OC[C@@H](O[C@H]3C"
        "[C@@](C)(O)[C@H](N(C)C)[C@@H](C)O3)[C@H](O)[C@H]2N)"
        "/C=C\\C=C\\1",
        "3-butyl-11-hydroxy-5,11,17,21,24-pentamethylazacyclotetracosan"
        "-2-one",
    ),
    (
        "CC(C)[C@@H](NC(N)=O)C(=O)O",
        "(2R)-2-(methanoylamino)-2-carbamoylamino-3-methylbutanoic acid",
    ),
    (
        "C/C1=C/[C@@H](C)C/C=C\\[C@H]2[C@@H]3O[C@]3(C)[C@@H](C)"
        "[C@H]3C(Cc4c[nH]c5ccccc45)NC(=O)[C@]32C2=N[C@@H](CC2)C1=O",
        "(2S,4Z,6S,8Z,10R,11R)-2,12-diethyl-4,6-dimethyl-10,11,11"
        "-trioxiranyl-3-oxo-1-azacyclododecene",
    ),
    (
        "CC(=O)N[C@@H](CSCCC(=O)C(=O)O)C(=O)O",
        "2-oxobutanedioic acid",
    ),
    (
        "O=C(/C=C/c1ccc(Cl)cc1)c1ccccc1",
        "(2E)-3-(4-chlorophenyl)1-phenylprop-2-en-1-one",
    ),
    (
        "CN(C)CCC=C1c2ccccc2COc2ccccc21",
        "1-(ethylamino)-3-cyclopentadecylpropan-1-amine",
    ),
    (
        "N=C(N)NCCC[C@H](NC(=O)CNC(=O)[C@@H](N)CCC(=O)O)C(=O)CCl",
        "(4S)-5-(ethylamino)-4-aminochloroguanidinooxopentanoic acid",
    ),
    (
        "C=C[C@@H]1C(=C)CC[C@H]2[C@H]1C[C@H]1OC(=O)[C@@]3(C)"
        "[C@H](O)CC[C@@]2(C)[C@@]13O",
        "(1R,3R,4S,8S,9R,12R,13S,16S)-4-ethyl-12,16-dihydroxy-9,13"
        "-dimethyl-15-oxa-tetracyclo[7.6.1.03,8.013,16]hexadecan-14-one",
    ),
    (
        "C=C1C(=O)OC2/C=C(/CO)C(=O)/C=C\\C(C)(O)CC(OC(=O)/C(C)=C/C)C12",
        "2,3-dipentadecyl-4-methyl-5-oxotetrahydrofuran",
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
        "ethanamide",
    ),
    (
        "O=C([O-])[C@H](O)[C@H](O)COP(=O)([O-])[O-]",
        "(2R,3R)-2,3-dihydroxyphosphonobutanoic acid",
    ),
    (
        "CCCCCCCC(=O)N[C@H](C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H]"
        "(CCC(=O)O)C(=O)N[C@@H](CC(C)C)C(=O)O)C(C)C",
        "(4S)-5-(hexylamino)-4-(nonanoylamino)pentanedioic acid",
    ),
    ("OCCCO", "trimethylene glycol"),
    (
        "O=C([O-])[C@@](O)(CO)C(=O)CO",
        "(2R)-2-(hydroxymethyl)-2,4-dihydroxy-2-hydroxy-3-oxobutanoate",
    ),
    ("CCC1CC=C(N2CCCC2)C1=O", "ethylcyclopent-1-en-4-one"),
    (
        "Oc1ccc(CC2(O)Oc3cc(O)cc(O)c3C2O)cc1",
        "3,6,7,7a-tetrahydroxy-2,3-dihydro-1-benzofuran",
    ),
    (
        "CC/C=C\\C/C=C\\C/C=C\\CCCCCCCC(=O)OC[C@H](COP(=O)(O)"
        "OC[C@H](N)C(=O)O)OC(=O)CCCCCCCCC/C=C\\C/C=C\\CCCCC",
        "2-aminotetratetracontanoic acid",
    ),
    (
        "C[NH2+][C@@H](C)[C@@H](O)c1ccccc1",
        "(1S,2S)-2-(methylamino)1-phenylpropan-1-ol",
    ),
    ("CSCCSC", "2,5-dithiahexane"),
    (
        "CCCCCc1oc(CCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC"
        "[N+](C)(C)C)OC(=O)CCC/C=C\\C[C@H]2[C@@H](O)CC(O)O"
        "[C@@H]2/C=C/[C@@H](O)CCCCC)c(C)c1C",
        "ammonium dopentacontanoate",
    ),
    (
        "CC(=O)N[C@H]1C(O)O[C@H](CO)[C@@H](O)[C@@H]1O[C@H](C)"
        "C(=O)O",
        "(2R)-2-octyloxypropanoic acid",
    ),
    (
        "COc1cc(OC)c2c(O)c3c(c(-c4c5cc(OC)cc(OC)c5c(O)c5c(=O)cc(C)"
        "oc45)c2c1)O[C@](C)(O)CC3=O",
        "(6S)-9-hexadecyl-2,6-dihydroxy-6,12,14-trimethyl-7-oxa"
        "-tricyclo[8.4.0.03,8]tetradecan-4-one",
    ),
    (
        "OC[C@H]1O[C@H](OC[C@H]2O[C@H](OC[C@H]3O[C@H](O)[C@H](O)"
        "[C@@H](O)[C@@H]3O)[C@H](O)[C@@H](O)[C@H]2O)[C@H](O)"
        "[C@@H](O)[C@@H]1O",
        "(2R,3S,4S,5R,6S)-6-tetrahydropyranyl-3,4,5-trihydroxy-2"
        "-methyltetrahydropyran",
    ),
    (
        "N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CS)C(=O)N[C@@H](CS)C(=O)O",
        "L-phenylalanyl-D-cysteinyl-D-cysteine",
    ),
    (
        "COc1cc(OC)c2c(c1CC=C(C)C)O[C@H](c1ccccc1)CC2",
        "1-hexyl-4,7-dimethoxy-8-pentylchromane",
    ),
    (
        "CN=C(N)NCCC/C=C/CCC[C@H](C)[C@H]1OC(=O)/C(C)=C\\C=C/"
        "[C@H](C)[C@H](O)C[C@H](O)[C@H](C)[C@@H](O)CC[C@@H](C)"
        "[C@H](O)C[C@@]2(O)O[C@H](C[C@H](O)C[C@H](OC(=O)CC(=O)O)"
        "C[C@@H](O)C[C@H](O)/C(C)=C\\C=C/[C@H]1C)C[C@@H](O)"
        "[C@@H]2O",
        "propanoic acid",
    ),
    (
        "NC(CCC(=O)NC(CSC(CC=O)c1ccccc1O)C(=O)NCC(=O)O)C(=O)O",
        "5-(anilino)-2-aminopentanedioic acid",
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
        "-trithia-2,17,19-triaza-pentacyclo[8.7.0.24,8.02,8.011,16]"
        "nonadecan-3,18-dione",
    ),
    (
        "C[C@H](O)/C=C1\\C[C@H](O)[C@]23C[C@H]2C(C)(C)O[C@]3(O)C1=O",
        "(1S,2S,6S,9R)-2,6-dihydroxy-8,8-dimethyl-7-oxa-tricyclo"
        "[4.4.0.01,9]decan-5-one",
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
        "(2S,3R)-2-(hexanoylamino)-3-methylpentanoic acid",
    ),
    (
        "CC(=O)N[C@@H]1[C@@H](O)[C@H](O[C@@H]2O[C@H](CO)[C@H](O)"
        "[C@H](O[C@H]3O[C@H](CO)[C@@H](O)[C@H](O[C@@H]4O[C@H](CO)"
        "[C@H](O)[C@H](O)[C@H]4O)[C@H]3NC(C)=O)[C@H]2O)"
        "[C@@H](CO)O[C@H]1O",
        "ethanediamide",
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
        "(5S)-3-ethyl-4-hydroxy-5-methyl-2-oxo-5-pentylthiole",
    ),
    (
        "N[C@H](C=O)Cc1cnc[nH]1",
        "(2S)-2-amino-3-imidazolylpropanal",
    ),
    (
        "O=c1cc(-c2ccc(O)cc2)oc2cc(O)c(Cl)c(O)c12",
        "2,3-dibutyl-4-oxo-6-phenyl-2H-pyran",
    ),
    (
        "CCCCCCC[C@@H](O)[C@H](O)CC#CC#C[C@@H](O)CC",
        "(3S,9R,10R)-heptadec-4,6-diyne-3,9,10-triol",
    ),
    (
        "CC1CCC/C=C\\C=C\\C(O)CC(O)C/C=C\\C=C\\C(O)C/C=C/C=C\\C(=O)O1",
        "8,14,16-trihydroxy-24-methyloxacyclotetracosan-2-one",
    ),
    (
        "CCCCC/C=C\\C/C=C\\C/C=C\\CCCCCCC(=O)OC[C@@H](O)COP(=O)(O)"
        "OC1C(O)C(O)C(O)[C@@H](O)C1O",
        "(8Z,11Z,14Z)-icosa-8,11,14-trien-1-oate",
    ),
    (
        "CCCCCCCC(O)CC(=O)N[C@@H](CC(C)C)C(=O)N[C@H](CCC(=O)O)"
        "C(=O)N[C@H]1C(=O)N[C@H](C(C)C)C(=O)N[C@@H](CC(C)C)"
        "C(=O)N[C@H](CO)C(=O)N[C@@H](CC(C)C)C(=O)N[C@H](CO)"
        "C(=O)N[C@@H]([C@@H](C)CC)C(=O)OC1C",
        "3,9,15-tributyl-22-methyl-18-propyloxacyclodocosan-2-one",
    ),
    (
        "CC[C@@H]1C[C@@]23OC(=O)C(=C2O)OC(=O)[C@]2(C)[C@H](CCCC"
        "[C@]3(C)C=C1C(=O)O)C(C)=C[C@@H]1[C@@H](OC3OC(C)C(OC(=O)"
        "c4c(C)cccc4OC)C(OC4CC(O)C(OC)C(C)O4)C3O)[C@@H](OC3OC(C)"
        "C(OC)C(C)(O)C3OC)CC[C@H]12",
        "(1S,7S,9R,12R,17R,20S,21R,22S,25R)-21-docosyl-9-ethyl-26"
        "-hydroxy-1,12,18-trimethyl-22-nonyl-2,5-dioxo-3,6-dioxa"
        "-pentacyclo[15.8.0.14,7.020,25.07,12]hexacosa-4,10,18"
        "-triene-10-carboxylic acid",
    ),
    (
        "CC(C)=CCC[C@@H](C(=O)O)[C@H]1C(=O)C[C@@]2(C)C3=C(CC"
        "[C@]12C)[C@@]1(C)CCC(=O)C(C)(C)[C@@H]1[C@@H](O)C3",
        "(5R,6S,10S,13R,14R,17R,20R)-6,21-dihydroxycholest-8,24-dien-3,16,21-trione",
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
        "aminohydroxyoctadeca-9,12-dien-1-oate",
    ),
    (
        "COc1cc(OC2OC(C(=O)O)C(O)C(O)C2O)c(C2CC(=O)c3ccc(O)cc3O2)"
        "c(O)c1CC=C(C)C",
        "1-17-carboxyheptadecyl-7-hydroxychroman-3-one",
    ),
    (
        "CC[C@H](C)C=C(C)C=CC1=CC2=C(Cl)C(=O)[C@@](C)(OC(C)=O)"
        "C(=O)C2=CN1C(CC(C)C)C(=O)OC",
        "2-cyclodecyl-4-methylpentan-1-oate",
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
        "4,5-dihexyl-2-methyl-4H-pyran",
    ),
    (
        "COc1cc(/C=C/C(=O)O[C@H]2[C@H](O)C[C@](O)(C(=O)O)C[C@H]2O)"
        "ccc1O",
        "4-ethenyl-1-hydroxy-2-methoxybenzene",
    ),
    ("O=C1N=C([O-])c2ccccc21.[K+]", "potassium octanolate"),
    (
        "CC1=C(O)C(=O)C([C@@]2(C)CCCC2(C)C)=C(O)C1=O",
        "5-dihydroxy-1-methyl-4-octylcyclohexa-1,4-diene-1,2-dione",
    ),
    (
        "CC(=O)N(O)CCCCNC(=O)[C@H](COC(=O)c1cccc(O)c1O)NC(=O)"
        "[C@@H]1COC(c2cccc(O)c2O)=N1",
        "13-phenyltridecyl benzoate",
    ),
    (
        "CC(C)C1=C[C@@]23CC[C@H]4C(C)(C)CCC[C@]4(C(=O)O2)C3=CC1=O",
        "(1R,4S,9R)-5,5-dimethyl-13-propyl-15-oxa-tetracyclo"
        "[8.4.0.21,9.04,9]hexadeca-10,13-dien-12,16-dione",
    ),
    (
        "COC1=C(N[C@H](C(=O)O)[C@@H](C)O[C@@H]2O[C@H](CO)[C@H](O)"
        "[C@H](O)[C@H]2O)C[C@](O)(CO)CC1=NCC(=O)O",
        "2-aminoicosanoic acid",
    ),
    (
        "CC(=O)CCC1=C(C)C[C@@]2(CC1=O)C(=O)[C@@H]1C[C@@](O)(CO1)C2=O",
        "(1S,3R,5R)-9-butyl-5-hydroxy-10-methyl-7-oxa-tricyclo"
        "[3.2.1.53,3]tridec-9-en-2,4,13-trione",
    ),
    (
        "CCCCCCCCCCCCC1=C(OC(C)=O)C(=O)c2ccccc2C1=O",
        "1-docosyloxyethan-1-oate",
    ),
    (
        "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O.O",
        "adenine",
    ),
    (
        "COc1cc2c(c(O)c1C/C=C(\\C)CCC=C(C)C)CN(CCc1c[nH]c3ccccc13)C2=O",
        "1H-indole",
    ),
    (
        "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](OC(=O)"
        "[C@@H](N)Cc2c[nH]cn2)[C@H]1O",
        "adenine",
    ),
    (
        "NC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](N)C[C@@H]2N)"
        "[C@H](O)[C@@H](O)[C@@H]1O",
        "(2R,3S,4S,5R,6R)-6-cyclohexyl-3,4,5-trihydroxy-2-methyl"
        "tetrahydropyran",
    ),
    (
        "CSCC[C@H](NC(=O)[C@H](CO)NC(=O)[C@@H](N)CCCCN)C(=O)O",
        "(2S)-2-(propanoylamino)-diaminohydroxybutanoic acid",
    ),
    (
        "CSCC[C@H](N)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](Cc1cnc[nH]1)"
        "C(=O)O",
        "(2S)-2-(butanoylamino)-3-imidazolylpropanoic acid",
    ),
    (
        "CC(C)=CCc1c(O)ccc(C(=O)C2C(c3c(O)cc(/C=C/c4cc(O)c(O)cc4O)"
        "cc3O)C=C(C)CC2c2ccc(OC3OC(C(=O)O)C(O)C(O)C3O)cc2O)c1O",
        "1,3-dihydroxy-2-pentylbenzene",
    ),
    (
        "COC(=O)c1ccccc1OC1OC(COC2OC(C)C(O)C(O)C2O)C(O)C(O)C1O",
        "(oxan-2-yl)oxybenzene",
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
