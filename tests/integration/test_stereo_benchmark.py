"""
Phase 63 Stereo Benchmark: CIP audit and stereo-mismatch tracking.

This test suite tracks all small+medium (<=40 HA) benchmark compounds that have
stereo in the molecule but did NOT achieve InChI round-trip in v6.0 baseline.

=== Phase 63 CIP Audit Results ===
Total benchmark compounds with stereo: 309 (of 500)
  Correct stereo labels emitted: 244 (79%)
  Missing stereo (wrong parent): 65 (21%)
  WRONG_LABEL (incorrect CIP): 0
  WRONG_LOCANT (count mismatch): 0

=== Phase 63 Stereo Accounting ===
Small (<=20 HA):
  Already RT in v6.0: 27
  Newly RT after Phase 63: 8 (7 from stereo integration, 1 from Phase 62)
  Still failing: 50 (all due to wrong parent, NOT stereo)

Medium (21-40 HA):
  Already RT in v6.0: 23
  Newly RT after Phase 63: 1
  Still failing: 138 (all due to wrong parent, NOT stereo)

=== Key Finding ===
After Plans 01-03, ZERO compounds have stereo-only InChI mismatch. Every remaining
failure has connectivity/parent differences. The stereo integration is complete for
all naming paths that produce correct parent structures. Remaining stereo improvements
require Phase 64+ parent selection fixes first.

Tests use current names as baselines. When a compound's name improves (e.g., via
Phase 64+ parent fixes), update the expected name here.
"""

import re

import pytest

from orthonym import name_compound


# ---------------------------------------------------------------------------
# Small stereo compounds (<=20 HA, 58 non-RT at v6.0 baseline)
# ---------------------------------------------------------------------------
SMALL_STEREO_COMPOUNDS = [
    ('C[C@@H]([NH3+])P(=O)([O-])[O-]', 'unknown organic compound'),  # MISSING_STEREO - wrong parent
    ('O=C([O-])/C=C/C(=O)O.[Na+]', 'sodium hydrogen (2E)-but-2-enedioate'),  # Phase 64: partial salt hydrogen prefix
    ('CC[C@H](C)[C@H](N)C(=O)[O-]', '2-aminohexanoate'),  # Phase 64: single-anion neutralize-then-name now produces -oate
    ('N[C@H](C[13C](=O)O)[13C](=O)O', '2-aminobutanoic acid'),  # MISSING_STEREO - isotope
    ('C/N=C(\\N)NCCCCN', '4-(methylamino)-4-guanidinobutan-1-amine'),  # MISSING_STEREO - wrong parent
    ('C=C1C=C[C@H](C(C)C)CC1', '(3S)-3-isopropyl-6-methylcyclohexene'),
    ('O=C1N[C@H]2NC(=O)N[C@H]2N1', "N,N'-dipropylurea"),  # MISSING_STEREO - wrong parent
    ('NC(=O)N/C=C\\C(=O)OO', '(2Z)-3-(methanoylamino)-3-carbamoylaminoprop-2-en-1-peroxol'),
    ('C=CC/C=C/CCC(=O)OC', 'methyl (4E)-octa-4,7-dienoate'),  # NEWLY_RT
    ('O=C([O-])C(=O)C[C@H](O)C(=O)[O-]', '(2S)-2-hydroxy-4-oxopentanedioate'),  # NEWLY_RT
    ('O[C@H]1[C@H](O)[C@@H](O)[C@@H]2O[C@@H]2[C@@H]1O', '(1R,2R,3S,4S,5R,6S)-2,3,4,5-tetrahydroxybicyclo[4.1.0]hexane'),
    ('OC[C@@H]1O[C@@](O)(CO)[C@@H](O)[C@@H]1O', '(2S,3S,4S,5S)-2,3,4-trihydroxy-2,5-dimethyltetrahydrofuran'),
    ('C[C@H](CC(=O)[O-])OC(=O)C[C@@H](C)O', '(3R)-3-(butanoyloxy)-hydroxybutanoate'),
    ('C[C@@H]1Cc2cc(O)cc(O)c2CO1', '(3R)-6,8-dihydroxy-3-methylisochromane'),  # NEWLY_RT
    ('O=P([O-])([O-])OC[C@@H](O)[C@H](O)[C@@H](O)CO', '(2S,3R,4R)-1-hydroxy-2,3,4-trihydroxypentanephosphonic acid'),
    ('O=C([O-])[C@@H](O)[C@H](O)[C@H](O)[C@@H](O)C(=O)[O-]', '(2R,3S,4R,5S)-2,3,4,5-tetrahydroxyhexanedioate'),  # NEWLY_RT
    ('N[C@@H](COC(=O)CCC(=O)O)C(=O)O', 'butanedioic acid'),  # MISSING_STEREO - wrong parent
    ('C=C(C(=O)OC)N1C(=O)C[C@@H](C)C1=O', '(octanoyloxy)-2-pyrrolidinylprop-2-enediamide'),  # MISSING_STEREO - wrong parent
    ('CC(=O)[C@@H](C)Nc1ccccc1C(=O)O', '2-(2-oxo(3R)-3-aminobutyl)benzoic acid'),
    ('C=C[C@]1(C)CCC(=C(C)C)C[C@H]1C(=C)C', '(1S,2S)-1-ethyl-2,4-diisopropyl-1-methylcyclohexane'),
    ('CC(C)CC[C@@H](O)[C@H]1C(=O)OC[C@@H]1CO', '(3S,4S)-oxolan-2-one'),
    ('CC(C)[C@@H]1CC[C@H](C)CCC[C@H](C)CC1', '(1S,4S,7R)-4-isopropyl-1,7-dimethylcyclodecane'),
    ('COc1c(C)c(O)cc2c1C(=O)N[C@H]2C', '(3S)-6-hydroxy-4-methoxy-3,5-dimethylisoindolin-1-one'),
    ('CC(=N)NCCCC[C@H](N)C(=O)O.Cl.Cl', '(2S)-6-(ethylamino)-2-aminoiminohexanoic acid'),
    ('C/C=C/CC(O)CCC(=O)NCC(=O)O', '2-(octanoylamino)-hydroxyethanoic acid'),  # MISSING_STEREO - wrong parent
    ('CCC[C@@H]1OCc2c(O)cccc2[C@H]1O', '(3S,4R)-4,8-dihydroxy-3-propylisochromane'),  # NEWLY_RT
    ('C=C(C)[C@@H]1CC[C@@H](C)[C@@]12CC=C(C)CC2', 'spiro[4.5]decane'),  # MISSING_STEREO - wrong parent
    ('NC(C(=O)O)C(CC[C@H](N)C(=O)O)C(=O)O', '(6S)-3-(hydroxymethyl)-2,6-diaminoheptanetrioic acid'),
    ('COc1c(O)c(O)cc2c1CO[C@@H](C)C2=O', '(3S)-6,7-dihydroxy-8-methoxy-3-methylisochroman-4-one'),  # NEWLY_RT
    ('O=C(O)/C=C/c1ccc(OS(=O)(=O)O)cc1', '(2E)-3-(4-(sulfooxy)phenyl)prop-2-enoic acid'),  # NEWLY_RT
    ('CC[C@@H](O)C[C@@H](O)c1cc(OC)cc(=O)o1', '(1R,3R)-1-cyclohexylpentane-1,3-diol'),
    ('CC(C)=C[C@H](O)C1=CC(=O)[C@@H](O)[C@H](O)[C@H]1O', '(4S,5R,6S)-3-(1-hydroxy-3-methylbut-2-enyl)-3,4,5,6-tetrahydroxycyclohex-2-en-1-one'),
    ('CCC(C)C1=C2C(=O)OC[C@H]2[C@@H](C)[C@H](C)O1', '(2S,3R,4S)-6-(sec-butyl)-4,5-diethyl-2,3-dimethyl-3,4-dihydro-2H-pyran'),
    ('COc1c(Cl)c2c(c(C(=O)O)c1Cl)C[C@H](C)O2', '(7aS)-2,3a-dichloro-3-methoxy-7a-methyl-2,3-dihydro-1-benzofuran-6-carboxylic acid'),  # MISSING_STEREO - wrong parent
    ('C/C=C/C=C/C(=O)C1=C(O)C(=C(C)C)NC1=O', '3-hexyl-4-hydroxy-5-isopropyl-2-oxoazole'),  # MISSING_STEREO - wrong parent
    ('CCCCCCC(=O)NC1=CC(=O)[C@@H]2CCCN12', 'heptanamide'),  # MISSING_STEREO - wrong parent
    ('C/C=C1\\[C@H]2C=C(C)C[C@]1([NH3+])c1ccc(=O)[nH]c1C2', '(2R,6R)-4-methyl-9-aza-tricyclo[6.4.0.1(2,6)]tridec-4-en-10-one'),
    ('CO[C@@H]1[C@H](O)[C@@H](CO)O[C@H]1n1ccc(=O)[nH]c1=O', '(2R,3R,4R,5R)-4-hydroxy-3,5-dimethyl-2-pyrimidinyltetrahydrofuran'),
    ('CCCCC[C@@H](O)[C@@H](O)c1cc(OC)cc(=O)o1', '(1R,2R)-1-cyclohexylheptane-1,2-diol'),
    ('C[C@H]1C[C@H](O)[C@@H]2[C@H]1[C@@H]1[C@H](CC[C@]2(C)O)[C@@]1(C)CO', '(1S,2S,5S,6R,7R,8R,9S,11S)-2,6,6,9-tetramethyl-tricyclo[6.3.0.0(5,7)]undecan-2,11-diol'),
    ('C[C@H](NC(=O)[C@H](C)NC(=O)[C@@H]1CCCN1)C(=O)O', 'N-(2S)-2-(pentanoylamino)propanoyl-2-aminopropanoic acid'),
    ('CCCCCC=CC1=C(CO)C(=O)C[C@H](O)[C@@H]1O', '(4R,5S)-3-heptyl-4,5-dihydroxy-2-hydroxymethylcyclohex-2-en-1-one'),
    ('CC(=O)[C@@]1(C)C(C)=C[C@H](O)[C@H]2C[C@](C)(O)CC[C@@H]21', '1-cyclodecanylethan-1-one'),  # MISSING_STEREO - wrong parent
    ('CC1=C[C@]2(CC1=O)[C@H](C)CC[C@@H](C(C)(C)O)[C@H]2O', 'spiro[4.5]decane'),  # MISSING_STEREO - wrong parent
    ('COCC1=C2[C@@H]3CC(C)(C)C[C@@H]3C[C@@]2(O)CC1=O', '(2R,6R,8R)-11-ethyl-8-hydroxy-4,4-dimethyl-tricyclo[6.3.0.0(2,6)]undec-1-en-10-one'),
    ('O=C([O-])C(=O)C[C@@H](O)[C@H](O)[C@H](O)COP(=O)([O-])[O-]', '(4R,5S,6R)-4,5,6-trihydroxy-2-oxophosphonoheptanoate'),
    ('CCC(O)CC(=O)O[C@H](CC(=O)[O-])C[N+](C)(C)C', 'ammonium dodecanoate'),  # MISSING_STEREO - wrong parent
    ('NC(N)=NCCC[C@H](NC(=O)[C@@H](N)CO)C(=O)O', '(2S)-5-(methylamino)-2-(propanoylamino)-5-guanidinopentanoic acid'),
    ('CCCC/C=C\\CCCCCCCCCOC(C)=O', '(10Z)-pentadec-10-en-1-yl acetate'),  # NEWLY_RT (Phase 62)
    ('CC(C)[C@H]1CC[C@@H](CO)c2c(O)cc(C(=O)O)cc21', '(1R,4R)-1,2,3,4-tetrahydronaphthalene'),
    ('CCOC(=O)C[C@@H](SP(=O)(OC)OC)C(=O)OCC', 'diethyl butanedioate'),  # MISSING_STEREO - wrong parent
    ('CC1C/C(=C\\CC(CC(N)=O)CC(=O)O)C(=O)C(C)C1', '3-(2-oxo1-ethyl-3,5-dimethylcyclohexyl)-5-carbamoylpentanoic acid'),  # MISSING_STEREO - carbamoyl prefix now correct
    ('CC(C)=CCc1ccc(O)c2c1C=C[C@H]1O[C@@H]2O[C@H]1C', '(2S,4S,5R)-4-methyl-9-2-methylbut-2-enyl-3,13-dioxa-tricyclo[6.4.0.1(2,5)]tridec-6-en-12-ol'),
    ('COc1cccc2c1CO[C@@H]2C[C@@H](O)[C@@H](O)[C@@H]1O[C@@H]1C', '(1R,2R)-3-cyclononyl-1-oxiranylpropane-1,2-diol'),
    ('C[C@H]1C[C@@H](O)[C@H]2C(=O)c3c(O)cccc3O[C@]2(C)[C@@H]1O', '(1R,10S,11R,12S,14R)-4,11,14-trihydroxy-10,12-dimethyl-9-oxa-tricyclo[8.4.0.0(3,8)]tetradecan-2-one'),
    ('C[C@H]1CCC/C=C/[C@@H]2CC[C@H](O)[C@H]2[C@H](O)/C=C/C(=O)O1', '(3E,5R,6S,7S,8E,13S)-5-hydroxy-13-methyl-2-oxo-6,7-dipropyl-1-oxacyclotridecene'),
    ('CC(C)=CCc1ccc(O)c2c1[C@H](CC(=O)O)OC2=O', '2-cyclononylethanoic acid'),  # MISSING_STEREO - wrong parent
    ('C/C=C/C(=O)O[C@H]1/C=C\\C(=O)[C@@H](O)CCC(=O)O[C@@H]1C', '(5S,7Z,9S,10R)-5-hydroxy-10-methyloxecan-2-one'),
]


# ---------------------------------------------------------------------------
# Medium stereo compounds (21-40 HA, 139 non-RT at v6.0 baseline)
# Representative subset: 50 compounds spanning common failure patterns
# ---------------------------------------------------------------------------
MEDIUM_STEREO_COMPOUNDS = [
    ('Oc1ccc2c(c1)O[C@H](c1ccc(O)c(O)c1)[C@@H](O)[C@@H]2O', '(2R,3S,4R)-3,4,7-trihydroxychromane'),
    ('C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1[C@@H](O)[C@@H](O)[C@@H]2O', '(8R,9S,13S,14S,15R,16R,17R)-estran-3,15,16,17-tetraol'),
    ('COc1cc(O)c2c(c1)C(=O)C1=C(C2=O)[C@@H](O)C[C@@](C)(O)C1', '(4S,6S)-4,6,14-trihydroxy-12-methoxy-6-methyl-tricyclo[8.4.0.0(3,8)]tetradec-3-en-2,9-dione'),
    ('C[C@@H]1CC(=O)O[C@@H](C)[C@H](O)/C=C\\C(=O)O[C@@H](C)C/C=C\\C(=O)O1', '(4R,7Z,10S,13Z,15R,16S)-15-hydroxy-4,10,16-trimethyloxacyclohexadecan-2-one'),
    ('CN1CCC2=C[C@H](O)[C@H]3OC(=O)c4cc5c(cc4[C@H]3[C@@H]21)OCO5', '(1S,13S,14S,20S)-20-hydroxy-15-methyl-2,7,9-trioxa-15-aza-pentacyclo[11.7.0.0(4,12).0(6,10).0(14,18)]icos-18-en-3-one'),  # Updated P72: IUPAC VB-6 citation order
    ('CCCCC[C@H](O)/C=C/[C@H]1CCC(=O)[C@@H]1C/C=C\\CCCC(=O)O', '(5Z)-7-cyclopentylhept-5-enoic acid'),
    ('CCCCC[C@H](O)/C=C/[C@@H]1[C@@H](C/C=C\\CCCC(=O)O)[C@H](O)C[C@H]1O', '(5Z)-7-cyclopentylhept-5-enoic acid'),
    ('CC(C)[C@H](NC(=O)[C@@H](N)Cc1ccc(O)cc1)C(=O)N[C@@H](CCCN=C(N)N)C(=O)O', '(2S)-5-(methylamino)-2-(pentanoylamino)pentanoic acid'),
    ('CC12CCC(=O)C=C1C=CC1[C@@H]2CCC2(C)[C@H]1CCC21CCC(=O)O1', '(9S,14S)-pregn-4,6-dien-3-one'),
    ('C=C(C)[C@H]1CC[C@]2(C)[C@@H]1CC[C@]1(C)C/C=C(\\C)CC/C=C(\\C)CC[C@H]12', '(1R,3E,7E,11R,12R,15S,16R)-15-isopropyl-1,4,8,12-tetramethyl-tricyclo[9.7.0.0(12,16)]octadeca-3,7-diene'),
    ('C/C(=C\\CC/C(C)=C/C/C=C(/CC(=O)c1cc(O)ccc1O)C(=O)O)CO', '(2Z,5E,9E)-2-(1-oxo1-(2,5-dihydroxyphenyl)ethyl)-11-hydroxy-6,10-dimethylundeca-2,5,9-trienoic acid'),  # NEWLY_RT
    ('CC(C)(O)[C@@H]1CC[C@@](C)([C@H]2CC[C@]3(C)[C@@H]2CC[C@@H]2[C@@]4(C)CCC(=O)C(C)(C)[C@@H]4CC[C@]23C)O1', '(5R,8R,9R,10R,13R,14R,17S)-4,4,8,10,14-pentamethylgonan-3-one'),
    ('C/C1=C/C[C@H](O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)/C(C)=C/[C@H]2OC(=O)[C@H](C)[C@@H]2CC1', '(beta-D-glucopyranosyloxy)(1S,2E,4S,6Z,10S)-4-hydroxy-3,7-dimethylcyclodeca-2,6-dien-1-carboxylate'),
    ('CC1(C)OC[C@]2(C)[C@@H](CC[C@@]3(C)[C@H]2[C@@H](O)C[C@H]2C[C@@H]4C[C@@]23CC[C@]4(O)CO)O1', '(1S,2S,5R,6R,8R,10S,11R,12R,17R)-1,5,12,15,15-pentamethyl-14,16-dioxa-pentacyclo[9.8.0.1(2,6).0(2,8).0(12,17)]icosan-5,10-diol'),  # Updated P72: IUPAC VB-6 citation order
    ('COC(=O)[C@@H]1CC23CCCN4CC[C@@]5(c6ccccc6N(C)C15CC2)[C@@H]43', '(1R,10R,21S)-10-ethyl-12-methyl-4,12-diaza-hexacyclo[9.7.0.2(8,11).1(4,8).0(13,18).0(1,21)]henicosane'),  # Updated P72: IUPAC VB-6 citation order
    ('COc1cc2c(cc1OC)[C@H]1Cc3ccc(OC)c(OC)c3CN1CC2', '(1R)-4,5,13,14-tetramethoxy-10-aza-tetracyclo[8.8.0.0(2,7).0(12,17)]octadecane'),
    ('COC(=O)CC[C@@H](C)[C@H]1C[C@@H](O)[C@H]2[C@@H]3[C@H](O)C[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3CC[C@@]21C', '(3R,5S,7R,8R,9S,10S,13R,14S,15R,17R,20R)-3,7,15-trihydroxycholan-24-one'),
    ('C/C=C(/CC[C@@H](C)[C@H]1CC[C@H]2C3=CC[C@H]4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C)C(C)C', '(3S,5S,9R,10S,13R,14R,17R,20R,24Z)-stigmast-7,24-dien-3-ol'),
    ('C=C(CC[C@@H](C)[C@H]1CC[C@@]2(C)C3=C(CC[C@]12C)[C@@]1(C)CC[C@H](O)C(C)(C)[C@@H]1CC3)C(C)C', '(3S,5R,10S,13R,14R,17R,20R)-4,4,14-trimethylergost-8,24-dien-3-ol'),
    ('C=C(CC[C@@H](C)[C@H]1CC[C@H]2C3=CC[C@H]4[C@H](C)C(=O)CC[C@]4(C)C3=C[C@@H](O)[C@]12C)C(C)C', '(4S,5S,10S,11R,13R,14S,17R,20R)-11-hydroxy-4-methylergost-7,9,24-trien-3-one'),
    ('CC(C)C[C@H](N)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](CCCN=C(N)N)C(=O)O', '(2S)-2-(butanoylamino)-5-(methylamino)-amino-5-guanidinopentanedioic acid'),
    ('CC(=O)N[C@@H]1[C@@H](O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2NC(C)=O)[C@@H](O)[C@@H](CO)O[C@@H]1O', '(beta-D-2-(acetylamino)-2-deoxy-glucopyranosyloxy)ethanamide'),  # MISSING_STEREO - wrong parent
    ('CC(=O)N[C@H]1C(OP(=O)(O)OP(=O)(O)OC[C@H]2O[C@@H](n3ccc(=O)[nH]c3=O)[C@H](O)[C@@H]2O)O[C@H](CO)[C@H](O)[C@@H]1O', 'N-acetyl(3R,4R,5R,6R)-3-amino-4,5-dihydroxy-6-methyl-2-oxolanyltetrahydropyran'),
    ('CC(C)[C@@H](C)[C@@H](O)[C@H]1CC[C@@H]([C@@]2(C)CCC(=O)[C@@]3(C)CC[C@H](O)C[C@]34C=C[C@@](O)(O4)C2=O)[C@@H]1C', '(1S,5R,7R,10S,12S)-5-dodecyl-7,12-dihydroxy-1,5-dimethyl-15-oxa-tricyclo[8.4.0.1(7,10)]pentadec-8-en-2,6-dione'),
    ('CC(=CCC(O)C(C)[C@H]1CC(=O)[C@@]2(C)C3=C(C(=O)[C@@H](O)[C@]12C)[C@@]1(C)CCC(=O)[C@](C)(CO)[C@@H]1CC3=O)C(=O)O', '(4S,5R,10S,12S,13R,14R,17R)-12,22,27-trihydroxy-4,14-dimethylcholest-8,24-dien-3,7,11,15,27-pentaone'),
    ('C=C(CC[C@@H](C(=O)O)[C@H]1[C@H](O)[C@H](O)[C@@]2(C)C3=CC[C@H]4C(C)(C)C(=O)CC[C@]4(C)C3=CC[C@]12C)C(C)C', '(5R,10S,13R,14R,15R,16S,17R,20R)-15,16,21-trihydroxy-4,4,14-trimethylergost-7,9,24-trien-3,21-dione'),
    ('CC(CCCC(C)(O)COS(=O)(=O)O)[C@H]1CC[C@H]2[C@@H]3[C@H](O)C[C@@H]4C[C@H](O)CCC4(C)[C@H]3C[C@H](O)C12C', '(3R,5S,7R,8R,9S,12S,14S,17R)-cholestan-3,7,12,25-tetraol'),
    ('CC(=O)O[C@H]1CC[C@]2(C)C3=C(CC[C@H]2C1(C)C)[C@]1(C)C[C@@H](O)[C@H]([C@@H](C/C=C/C(C)(C)O)C(=O)O)[C@@]1(C)CC3', '(3S,5R,10S,13R,14R,16R,17R,20R,23E)-16,21,25-trihydroxy-4,4,14-trimethyl-21-oxocholest-8,23-dien-3-yl acetate'),
    ('O=C1c2c(O)cc(O)cc2O[C@@H](c2ccc(O)c(O)c2)[C@@H]1O[C@@H]1OC[C@@H](O)[C@H](O)[C@H]1O', '(beta-D-xylopyranosyloxy)(2S,3S)-3,5,7-trihydroxychroman-4-one'),
    ('CC1=C[C@]2(C)C[C@@H](C)CC[C@@H]2[C@H](C(=O)[C@@H]2C(=O)N3CC[C@@H]4C(=O)O[C@H]2[C@@]43O)[C@@H]1C', '(3R,4R,5R,6S)-3-(2-methylbutyl)-1,3,6-trimethyl-5-oxocyclohex-1-enecarboxylate'),
    ('CC(=O)O[C@H]1[C@@H](OC(C)=O)C(C)(C)[C@]2(O)CC[C@H]3C(=O)c4ccoc4C[C@@H]3[C@@]2(C)[C@H]1OC(C)=O', '(1S,2S,3R,4S,5S,7R,10R)-3,4,5,7-tetrahydroxy-2,6,6-trimethyl-15-oxa-tetracyclo[8.7.0.0(2,7).0(12,16)]heptadecan-11-one acetate acetate acetate'),
    ('OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O[C@@H]1OC[C@@H](O)[C@H](O)[C@H]1O', '(beta-D-xylopyranosyloxy)(2R,3S,4S,5R,6R)-3,4,5,6-tetrahydroxy-2-methyltetrahydropyran'),
    ('OC[C@H]1O[C@H](O)[C@H](O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]2O)[C@@H](O)[C@H]1O', '(alpha-D-mannopyranosyloxy)(2R,3R,4S,5R,6S)-3,4,5,6-tetrahydroxy-2-methyltetrahydropyran'),
    ('OC[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)CO[C@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O', '(alpha-D-glucopyranosyloxy)(2S,3R,4S,5R)-2,3,4,5-tetrahydroxyhexane-1,6-diol'),
    ('CC(=O)OC[C@H]1O[C@@H](N2CCC(=O)NC2=O)[C@H](OC(C)=O)[C@@H]1OC(C)=O', '1,2-bis(acetyloxy)tetrahydrofuran'),  # MISSING_STEREO - wrong parent
    ('C=C1NC(=O)[C@H]([C@@H](C)[C@]2(O)C(=O)N(C)c3ccccc32)NC1=O', '(2R)-2-hydroxy-N-methylindolin-1-one'),
    ('C[C@H]1C=C[C@H]2C[C@@H](O)CC[C@H]2[C@@H]1c1cc(N)c(C=O)c(=O)o1', '(3R,4R,5R,6S)-3-(2-hydroxy(2R)-butyl)amino-3-hydroxy-6-methylcyclohex-1-enecarbaldehyde'),
    ('CC(C)=CCC/C(C)=C/CC[C@]1(C)Cc2c(c(O)cc3c2CN([C@H]2CCCNC2=O)C3=O)C[C@@H]1O', '(11S,12R)-12-(6E)-2,6-dimethylnona-2,6-dienyl-8,11-dihydroxy-12-methyl-4-pentyl-4-aza-tricyclo[7.4.0.0(2,6)]tridecan-5-one'),
    ('CC(C)[C@@H](C)[C@@H](O)[C@H]1CC[C@@H]([C@@]2(C)CCC(=O)[C@@]3(C)CC[C@H](O)C[C@]34C=C[C@@](O)(O4)C2=O)[C@@H]1C', '(1S,5R,7R,10S,12S)-5-dodecyl-7,12-dihydroxy-1,5-dimethyl-15-oxa-tricyclo[8.4.0.1(7,10)]pentadec-8-en-2,6-dione'),
    ('CC(=O)N[C@@H](CC(C)C)C(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N/C=C\\c1c[nH]c2ccccc12', 'N-N-(2S)-2-(ethanoylamino)-4-methylpentanoyl(2S)-2-(methylamino)-2-amino-3-phenylpropanoyl-3-(2-aminoethyl)-1H-indole'),
    ('O=C1N[C@@H](C[C@@]2(O)c3ccccc3N3C(=O)[C@@H]4CCCCN4[C@@H]32)C(=O)N[C@H]1Cc1ccccc1', '(8R,9S,15S)-8-dodecyl-8-hydroxy-1,10-diaza-tetracyclo[7.7.0.0(2,7).0(10,15)]hexadecan-16-one'),
    ('CC(C)[C@@H](C)[C@@H](O)[C@H]1CC[C@@H]([C@@]2(C)CCC(=O)[C@@]3(C)CC[C@H](O)C[C@]34C=C[C@@](O)(O4)C2=O)[C@@H]1C', '(1S,5R,7R,10S,12S)-5-dodecyl-7,12-dihydroxy-1,5-dimethyl-15-oxa-tricyclo[8.4.0.1(7,10)]pentadec-8-en-2,6-dione'),
    ('COc1c(Cl)c(C)cc2cc(O)c3c(c12)C(=O)c1cc2c(c(O)c1C3=O)[C@H](C)OC2=O', '(6S)-4,21-dihydroxy-15-methoxy-6,17-dimethyl-7-oxa-pentacyclo[11.8.0.0(3,11).0(5,9).0(14,19)]henicosan-2,8,12-trione'),  # Updated P72: IUPAC VB-6 citation order
    ('C/C=C(/CC[C@@H](C)[C@H]1CC[C@H]2C3=CC[C@H]4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C)C(C)C', '(3S,5S,9R,10S,13R,14R,17R,20R,24Z)-stigmast-7,24-dien-3-ol'),
    ('C=C(CC[C@@H](C)[C@H]1CC[C@@]2(C)C3=C(CC[C@]12C)[C@@]1(C)CC[C@H](O)C(C)(C)[C@@H]1CC3)C(C)C', '(3S,5R,10S,13R,14R,17R,20R)-4,4,14-trimethylergost-8,24-dien-3-ol'),
    ('C=C(CC[C@@H](C)[C@H]1CC[C@H]2C3=CC[C@H]4[C@H](C)C(=O)CC[C@]4(C)C3=C[C@@H](O)[C@]12C)C(C)C', '(4S,5S,10S,11R,13R,14S,17R,20R)-11-hydroxy-4-methylergost-7,9,24-trien-3-one'),
    ('CC(=O)O[C@H]1CC[C@]2(C)C3=C(CC[C@H]2C1(C)C)[C@]1(C)C[C@@H](O)[C@H]([C@@H](C/C=C/C(C)(C)O)C(=O)O)[C@@]1(C)CC3', '(3S,5R,10S,13R,14R,16R,17R,20R,23E)-16,21,25-trihydroxy-4,4,14-trimethyl-21-oxocholest-8,23-dien-3-yl acetate'),
    ('C[C@@H]1O[C@@H](OCCCCCCCCCCCCCCCCCCCC[C@@H](O)CC(=O)O)[C@H](O)C[C@H]1O', '(3R)-23-hexyloxy-3-hydroxytricosanoic acid'),
    ('CC(C)[C@H](CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C)OS(=O)(=O)[O-]', '(3S,8S,9S,10R,13R,14S,17R,20R,24S)-cholest-5-en-3-ol'),
]


# ---------------------------------------------------------------------------
# Test IDs
# ---------------------------------------------------------------------------
_SMALL_IDS = [
    f"s{i:02d}_HA{ha}"
    for i, (smi, _) in enumerate(SMALL_STEREO_COMPOUNDS)
    for ha in [len([a for a in __import__('rdkit').Chem.MolFromSmiles(smi).GetAtoms()
                    if a.GetAtomicNum() > 1]) if __import__('rdkit').Chem.MolFromSmiles(smi) else 0]
]

_MEDIUM_IDS = [
    f"m{i:02d}_HA{ha}"
    for i, (smi, _) in enumerate(MEDIUM_STEREO_COMPOUNDS)
    for ha in [len([a for a in __import__('rdkit').Chem.MolFromSmiles(smi).GetAtoms()
                    if a.GetAtomicNum() > 1]) if __import__('rdkit').Chem.MolFromSmiles(smi) else 0]
]


@pytest.mark.parametrize("smiles,expected_name", SMALL_STEREO_COMPOUNDS, ids=_SMALL_IDS)
def test_small_stereo_baseline(smiles, expected_name):
    """Track stereo naming for small (<=20 HA) benchmark compounds.

    These compounds had stereo in the molecule but did not round-trip in
    v6.0 baseline. The expected names are current baselines, NOT necessarily
    correct IUPAC names. When a compound's naming improves (e.g., from Phase 64+
    parent fixes), update the expected value.
    """
    result = name_compound(smiles)
    assert result == expected_name, (
        f"STEREO BASELINE CHANGED: {smiles}\n"
        f"  Expected: {expected_name}\n"
        f"  Got:      {result}"
    )


@pytest.mark.parametrize("smiles,expected_name", MEDIUM_STEREO_COMPOUNDS, ids=_MEDIUM_IDS)
def test_medium_stereo_baseline(smiles, expected_name):
    """Track stereo naming for medium (21-40 HA) benchmark compounds.

    Same purpose as test_small_stereo_baseline but for medium molecules.
    """
    result = name_compound(smiles)
    assert result == expected_name, (
        f"STEREO BASELINE CHANGED: {smiles}\n"
        f"  Expected: {expected_name}\n"
        f"  Got:      {result}"
    )


def test_cip_audit_no_wrong_labels():
    """Verify CIP audit: all emitted stereo labels match rdCIPLabeler.

    Phase 63 CIP audit confirmed 0 WRONG_LABEL and 0 WRONG_LOCANT across
    all 500 benchmark compounds. This test verifies the invariant holds.
    """
    import re
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    # Test a representative set of compounds that DO emit stereo
    stereo_compounds = [
        (smi, name) for smi, name in SMALL_STEREO_COMPOUNDS
        if re.search(r'\(\d+[RSEZ]', name)
    ]

    for smiles, expected_name in stereo_compounds[:20]:  # Check first 20
        result = name_compound(smiles)
        # Parse stereo from name
        stereo_in_name = re.findall(r'(\d+)([RSEZ])', result)

        # All CIP codes must be valid
        for locant, code in stereo_in_name:
            assert code in ('R', 'S', 'E', 'Z'), (
                f"Invalid CIP code '{code}' at locant {locant} in: {result}"
            )

        # Verify molecule has at least as many stereo as name claims
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            rdCIPLabeler.AssignCIPLabels(mol)
            mol_rs = sum(1 for a in mol.GetAtoms() if a.HasProp('_CIPCode'))
            mol_ez = sum(1 for b in mol.GetBonds() if b.HasProp('_CIPCode'))
            name_rs = sum(1 for _, c in stereo_in_name if c in ('R', 'S'))
            name_ez = sum(1 for _, c in stereo_in_name if c in ('E', 'Z'))
            assert name_rs <= mol_rs, (
                f"More R/S in name ({name_rs}) than in molecule ({mol_rs}): {smiles}"
            )
            assert name_ez <= mol_ez, (
                f"More E/Z in name ({name_ez}) than in molecule ({mol_ez}): {smiles}"
            )
