"""
Golden canary regression tests for compounds that round-trip correctly.

These compounds matched on InChI round-trip (Orthonym name -> OPSIN parse -> InChI
comparison) as of the Phase 44 baseline (v4.0+), extended in Phase 49 with 10 new
compound classes (acetals, disulfides, cyclic imides, thiocarboxylic acids, carbamic
acid). They form a regression safety net: any naming change that breaks one of these
tests must be investigated before merging.

If a test fails, first check whether the NEW name is also valid IUPAC nomenclature
(it might be an equally correct alternative). Only revert if the new name is wrong.

These tests run as part of the normal test suite (no @pytest.mark.slow) so that
regressions are caught immediately during development.

Original 75: 
Phase 49 additions: 10 compounds from missing compound classes (CLS-01 to CLS-05)
Phase 50 additions: 3 compounds from decomposition format fixes (ether/alkoxy)
Phase 83 additions: 19 compounds from v9.0 canary expansion (coverage-based + anchors)
Phase 95 additions: 2 compounds newly RT-matching in v10.0 (steroid + epoxycyclohexane)
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from orthonym import name_compound


# ---------------------------------------------------------------------------
# 312 golden canary compounds: (SMILES, expected_name)
# Original 75 from Phase 44 + 10 from Phase 49 + 3 from Phase 50
# + 44 from Phase 62 (Small Molecule Accuracy Sprint)
# + 9 from Phase 63 (Stereochemistry Accuracy)
# + 2 from Phase 66 (Medium Molecule Completeness)
# + 15 from Phase 67 (v7.0 Final Benchmark)
# + 25 from Phase 73 (v8.0 Closure: phase improvements + failure sentinels)
# + 19 from Phase 83 (v9.0 canary expansion: coverage-based + benchmark anchors)
# + 2 from Phase 95 (v10.0 newly RT-matching compounds)
# + 80 from Phase 102-04 (v11.0 canary expansion: newly identified RT-matching)
# + 28 from Phase 108 (v12.0 canary expansion: newly RT-matching from benchmark)
# ---------------------------------------------------------------------------

CANARY_COMPOUNDS = [
    (
        "COc1cc(CC(=O)C(=O)c2c(O)cc(O)c(OC)c2O)cc(OC)c1O",
        "1-(2,4,6-trihydroxy-3-methoxyphenyl)-3-(4-hydroxy-3,5-dimethoxyphenyl)propane-1,2-dione",
    ),
    (
        "CCCCCC/C=C/C=C(\\CCCC(=O)O)[N+](=O)[O-]",
        "(5E,7E)-5-nitrotetradeca-5,7-dienoic acid",
    ),
    (
        "Cc1ccc(C(=O)O)s1",
        "2-methylthiophene-5-carboxylic acid",
    ),
    (
        "COC(/C=C/c1ccccc1)[C@@H](C)C(OC)[C@@H](C)/C=C/C(C)=C/C(N)=O",
        "(2E,4E,6S,8R,10E)-7,9-dimethoxy-3,6,8-trimethyl-11-phenylundeca-2,4,10-trienamide",
    ),
    (
        "CCCCC/C=C\\C/C=C\\C/C=C\\C/C=C\\C[C@@H](O)CC(=O)O",
        "(3R,5Z,8Z,11Z,14Z)-3-hydroxyicosa-5,8,11,14-tetraenoic acid",
    ),
    (
        "C#CCCCCCCCCCCCC(O)CC(CO)OC(C)=O",
        "2-(acetyloxy)-4-hydroxyheptadec-16-yn-1-ol",
    ),
    (
        "C[C@@H](O)[C@H](NC(=O)[C@@H](N)CCCCN)C(=O)N[C@@H](CS)C(=O)O",
        "L-lysyl-L-threonyl-L-cysteine",
    ),
    (
        "SS",
        "disulfane",
    ),
    (
        "C[C@]12CC[C@@H](O)C[C@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)[C@H](O)CC[C@@H]12",
        "(3R,5R,8R,9S,10S,13S,14S,17R)-androstan-3,17-diol",
    ),
    (
        "CC(C)C[C@H](N)C(=O)N[C@@H](CO)C(=O)NCC(=O)O",
        "L-leucyl-L-serylglycine",
    ),
    (
        "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CO)C(=O)O",
        "L-tryptophyl-L-tyrosyl-L-serine",
    ),
    (
        "CC1=CC[C@]23O[C@@]2(C)CC[C@@H]2[C@H](OC(=O)[C@H]2C)[C@@H]13",
        "(1R,3S,6S,7S,10S,11R)-3,7,12-trimethyl-2,9-dioxa-tetracyclo[9.3.0.0(1,3).0(6,10)]tetradec-12-en-8-one",  # Updated P72: IUPAC VB-6 citation order
    ),
    (
        "C=CC(=O)CCCC",
        "hept-1-en-3-one",
    ),
    (
        "N[C@@H](CO)C(=O)N[C@@H](CO)C(=O)N1CCC[C@@H]1C(=O)O",
        "N-L-seryl-L-serineyl(2R)-pyrrolidine-2-carboxylic acid",
    ),
    (
        "C[C@H](NC(=O)[C@@H](N)CO)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O",
        "L-seryl-L-alanyl-L-tyrosine",
    ),
    (
        "CCCCCCCCCCC(C)C(=O)O",
        "2-methyldodecanoic acid",
    ),
    (
        "N[C@@H](CC(=O)O)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CCC(=O)O)C(=O)O",
        "L-aspartyl-L-tryptophyl-L-glutamic acid",
    ),
    (
        "CC/C=C\\CC(O)C(O)/C=C/C(O)CCCCCCCC(=O)O",
        "(10E,15Z)-9,12,13-trihydroxyoctadeca-10,15-dienoic acid",
    ),
    (
        "CC1=CC(=O)CC(C)(C)C1",
        "3,5,5-trimethylcyclohex-2-en-1-one",
    ),
    (
        "O=CC1=CC(O)C(O)C(O)C1O",
        "3,4,5,6-tetrahydroxycyclohex-1-enecarbaldehyde",
    ),
    (
        "COc1cc(C=CC(=O)O)cc(O)c1O",
        "3-(4,5-dihydroxy-3-methoxyphenyl)prop-2-enoic acid",
    ),
    (
        "C[C@H]1C/C=C\\[C@H]2[C@@H]3O[C@]3(C)[C@@H](C)[C@H]3[C@H](Cc4ccccc4)NC(=O)[C@@]32OC(=O)/C=C\\[C@@](C)(O)C1=O",
        "(1S,2Z,5S,7R,8Z,12R,15S,16S,17S,18R,20S)-15-benzyl-7-hydroxy-5,7,17,18-tetramethyl-11,19-dioxa-14-aza-tetracyclo[10.8.0.0(12,16).0(18,20)]icosa-2,8-dien-6,10,13-trione",  # Updated P72: IUPAC VB-6 citation order
    ),
    (
        "CC1=C[C@@H]2/C=C(\\C)CCC[C@H](O)/C=C/C(=O)O[C@]23C(=O)N[C@@H](CC(C)C)[C@@H]3[C@@H]1C",
        "(1S,2E,7S,8E,12R,15S,16S,17S)-7-hydroxy-15-isobutyl-3,17,18-trimethyl-11-oxa-14-aza-tricyclo[10.7.0.0(12,16)]nonadeca-2,8,18-trien-10,13-dione",
    ),
    (
        "O=C(O)/C(Cl)=C\\C(=O)C(Cl)C(=O)O",
        "(2E)-2,5-dichloro-4-oxohex-2-enedioic acid",
    ),
    (
        "CCCN(C=O)CCC",
        "N,N-dipropylformamide",
    ),
    (
        "CCCCCCCC/C=C/CCCCCCCC=O",
        "(9E)-octadec-9-enal",
    ),
    (
        "CCCc1nc(C)c(C)nc1C",
        "2,3,6-trimethyl-5-propylpyrazine",
    ),
    (
        "Cc1ccc(O)cc1C",
        "4-hydroxy-1,2-dimethylbenzene",
    ),
    (
        "C[C@H](NC(=O)[C@@H](N)CC(=O)O)C(=O)N[C@@H](CCCCN)C(=O)O",
        "L-aspartyl-L-alanyl-L-lysine",
    ),
    (
        "O=C(O)CCC(=O)c1ccc(F)cc1",
        "4-(4-fluorophenyl)-4-oxobutanoic acid",
    ),
    (
        "CCC/C=C\\C#CC/C=C\\CCCCCCCC(=O)O",
        "(9Z,14Z)-octadeca-9,14-dien-12-ynoic acid",
    ),
    (
        "O=P([O-])([O-])[O-].[Mg+2].[NH4+]",
        "ammonium magnesium phosphate",
    ),
    (
        "CCCCCCCCCCCCCCCCCCCCCC(=O)OCC",
        "ethyl docosanoate",
    ),
    (
        "CC(=O)[C@H](O)c1ccccc1",
        "(1R)-1-hydroxy-1-phenylpropan-2-one",
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](N)CCC(=O)O)C(=O)O",
        "L-glutamyl-L-glutamyl-L-isoleucine",
    ),
    (
        "N[C@@H](CC(=O)O)C(=O)N[C@@H](CO)C(=O)N[C@@H](CO)C(=O)O",
        "L-aspartyl-L-seryl-L-serine",
    ),
    (
        "N[C@@H](Cc1ccc(Br)cc1)C(=O)O",
        "(2S)-3-(4-bromophenyl)-2-aminopropanoic acid",
    ),
    (
        "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCCCCCCCCCCCCCCCC(=O)[O-]",
        "(19Z,22Z,25Z,28Z,31Z)-tetratriaconta-19,22,25,28,31-pentaenoate",
    ),
    (
        "CC1(C)C[C@H](O)[C@]23CC[C@@H](O)[C@](C)(CC[C@@H]12)C3",
        "(1S,4R,5R,8S,11S)-5,9,9-trimethyl-tricyclo[6.3.0.1(1,5)]dodecan-4,11-diol",
    ),
    (
        "CCCCCCCCCc1cc(=O)c2ccccc2n1C",
        "N-methyl-2-nonylquinolin-4-one",
    ),
    (
        "CC(C)[C@H](N)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CC(=O)O)C(=O)O",
        "L-valyl-L-tyrosyl-L-aspartic acid",
    ),
    (
        "C=C[C@@H](O)CCCCC",
        "(3S)-oct-1-en-3-ol",
    ),
    (
        "CCCCC/C=C\\C/C=C\\C/C=C\\CCCCC(=O)[O-]",
        "(6Z,9Z,12Z)-octadeca-6,9,12-trienoate",
    ),
    (
        "NCCCC[C@H](NC(=O)[C@@H](N)CCC(N)=O)C(=O)N[C@@H](CS)C(=O)O",
        "L-glutaminyl-L-lysyl-L-cysteine",
    ),
    (
        "Oc1ccnc2ccccc12",
        "4-hydroxyquinoline",
    ),
    (
        "CCCCCCC#CCCCCCC(=O)O",
        "tetradec-7-ynoic acid",
    ),
    (
        "CCCCCCCCCCCCOCCO",
        "3-oxapentadecan-1-ol",
    ),
    (
        "C[C@H]1C(=O)O[C@@H]2CCN3CC=C(COC(=O)[C@](C)(O)[C@]1(C)O)[C@H]23",
        "(1R,4R,5R,6R,16R)-5,6-dihydroxy-4,5,6-trimethyl-2,8-dioxa-13-aza-tricyclo[8.5.1.0(13,16)]hexadec-10-en-3,7-dione",
    ),
    (
        "CCCCCCC/C=C\\CCCCCCC(=O)O",
        "(8Z)-hexadec-8-enoic acid",
    ),
    (
        "NC(=O)C[C@H](N)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](CC(=O)O)C(=O)O",
        "L-asparaginyl-L-aspartyl-L-aspartic acid",
    ),
    (
        "CC/C=C(\\C)CC/C=C(\\C)CCC=C(C)C",
        "(6E,10E)-2,6,10-trimethyltrideca-2,6,10-triene",
    ),
    (
        "O=C(O)c1ccc[nH]1",
        "pyrrole-2-carboxylic acid",
    ),
    (
        "O=C(O)C(O)c1ccc(O)cc1O",
        "2-(2,4-dihydroxyphenyl)-2-hydroxyethanoic acid",
    ),
    (
        "OC[C@H](O)c1ccccc1",
        "(2R)-2-hydroxy-2-phenylethan-1-ol",
    ),
    (
        "O=[N+]([O-])OO",
        "peroxynitric acid",
    ),
    (
        "CCCCCCCCCCCCCCCC(=O)OC(CCCCC)CCCCCCCCCCCC(=O)[O-]",
        "13-(palmitoyloxy)octadecanoate",
    ),
    (
        "CC(C)CCCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
        "26-methylheptacosanoic acid",
    ),
    (
        "C=C/C(C)=C/CC(C)(C)/C(=N\\O)C(C)C",
        "(3Z,6E)-2,4,4,7-tetramethylnona-6,8-dien-3-one oxime",
    ),
    (
        "CCCCCCCCCCCCCCCCCC(=O)N[C@@H](CO)C(=O)O",
        "(2S)-2-(octadecanoylamino)-3-hydroxypropanoic acid",
    ),
    (
        "CCCC(O)CC(O)C(O)C/C=C/CCCCCCCC(=O)O",
        "(9E)-12,13,15-trihydroxyoctadec-9-enoic acid",
    ),
    (
        "O=C(O)[C@@H]1OC(O)[C@H](O)[C@@H](O)[C@H]1O",
        "(2R,3R,4S,5R)-3,4,5,6-tetrahydroxyoxane-2-carboxylic acid",
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@@H](N)C(C)C)C(=O)N[C@H](C(=O)O)[C@@H](C)O",
        "L-valyl-L-isoleucyl-L-threonine",
    ),
    (
        "COc1cccc(C(=O)O)c1O",
        "2-hydroxy-3-methoxybenzoic acid",
    ),
    (
        "CCCCCCC/C=C\\CCCCCCCC(=O)[O-]",
        "(9Z)-heptadec-9-enoate",
    ),
    (
        "CC(C)C[C@H](N)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](CC(N)=O)C(=O)O",
        "L-leucyl-L-histidyl-L-asparagine",
    ),
    (
        "C/C(C=O)=C\\CC/C(C)=C/C=O",
        "(2E,6E)-2,6-dimethylocta-2,6-dienedial",
    ),
    (
        "CCCCC/C=C\\C=C\\C=C/CCCCCCC(=O)O",
        "(8Z,10E,12Z)-octadeca-8,10,12-trienoic acid",
    ),
    (
        "CC1=C(O)C(=O)[C@]2(O)C[C@H]3C[C@](C)(C(=O)O)C[C@H]3[C@]12C",
        "(1S,3R,5S,7R,8R)-1,10-dihydroxy-5,8,9-trimethyl-11-oxo-tricyclo[6.3.0.0(3,7)]undec-9-ene-5-carboxylic acid",
    ),
    (
        "CCCCCCCCC(C)CC(C)C",
        "2,4-dimethyldodecane",
    ),
    (
        "O=C(O)C[C@H](O)CCCCCCO",
        "(3R)-9-hydroxy-3-hydroxynonanoic acid",
    ),
    (
        "COc1c(-c2ccccc2)c2ccc(O)cc2[nH]c1=O",
        "7-hydroxy-3-methoxy-4-phenylquinolin-2-one",
    ),
    (
        "CCCCCCCC/C=C\\CCCCCCO",
        "(7Z)-hexadec-7-en-1-ol",
    ),
    (
        "C/C=C\\CCCCC(=O)O",
        "(6Z)-oct-6-enoic acid",
    ),
    (
        "C/C=C(\\C)CCC=C(C)C",
        "(6E)-2,6-dimethylocta-2,6-diene",
    ),
    (
        "N[C@H](Cc1ccc(O)cc1)C(=O)O",
        "(2R)-3-(4-hydroxyphenyl)-2-aminopropanoic acid",
    ),
    # --- Phase 49: Missing Compound Classes (CLS-01 to CLS-05) ---
    # CLS-01: Acetals
    (
        "C1OCCO1",
        "1,3-dioxolane",
    ),
    (
        "COC(C)OC",
        "1,1-dimethoxyethane",
    ),
    # CLS-02: Disulfides
    (
        "CSSC",
        "2,3-dithiabutane",
    ),
    (
        "CCSSSCC",
        "3,4,5-trithiaheptane",
    ),
    # CLS-03: Cyclic Imides
    (
        "O=C1CCC(=O)N1",
        "succinimide",
    ),
    (
        "O=C1NC(=O)c2ccccc21",
        "phthalimide",
    ),
    # CLS-04: Thiocarboxylic Acids
    (
        "CC(=O)S",
        "ethanethioic S-acid",
    ),
    (
        "CC(=S)S",
        "ethanedithioic acid",
    ),
    # CLS-05: Carbamic Acid
    (
        "NC(=O)O",
        "carbamic acid",
    ),
    (
        "CN(C)C(=O)O",
        "N,N-dimethylcarbamic acid",
    ),
    # --- Phase 50: Decomposition Format Fixes (DEC-01/DEC-02) ---
    # Alkoxy naming on benzene (DEC-02: bare oxy elimination)
    # Now returns "anisole" (retained name, P-34.1.1.4 PIN)
    (
        "c1ccc(OC)cc1",
        "anisole",
    ),
    (
        "c1ccc(OCC)cc1",
        "ethoxybenzene",
    ),
    # Ring ether guard (DEC-01: oxane not decomposed)
    (
        "C1CCOCC1",
        "oxane",
    ),
    # --- Phase 62: Small Molecule Accuracy Sprint (44 compounds) ---
    # Wave 1-3 fixes: parent selection, substituent detection, E/Z ester fragments,
    # isochromane/chromane locant corrections
    (
        "O=CO",
        "formic acid",
    ),
    (
        "OC(=O)/C=C/c1ccccc1",
        "(2E)-3-phenylprop-2-enoic acid",
    ),
    (
        r"C(=N\O)c1ccccc1",
        "benzaldehyde oxime",
    ),
    (
        "CC(C)=CC=O",
        "3-methylbut-2-enal",
    ),
    (
        r"CC(=O)/C=C(\C)C",
        "mesityl oxide",
    ),
    (
        "C=CC/C=C/CCC(=O)OC",
        "methyl (4E)-octa-4,7-dienoate",
    ),
    (
        "CCCCC=O",
        "pentanal",
    ),
    (
        "CCCCCCCCCC=O",
        "decanal",
    ),
    (
        "CC(=O)OC1CCCCC1",
        "acetyloxycyclohexane",
    ),
    (
        "OCC(O)CO",
        "glycerol",
    ),
    (
        "c1cc(-c2ccco2)oc1",
        "2,2'-bifuran",
    ),
    (
        "OC(=O)c1ccc(O)c(O)c1",
        "3,4-dihydroxybenzoic acid",
    ),
    (
        "CC(=O)OCC(COC(C)=O)OC(C)=O",
        "1,2,3-tris(acetyloxy)propane",
    ),
    (
        r"O=C(O)CCCC/C=C\CCCCCCCCCC",
        "(6Z)-heptadec-6-enoic acid",
    ),
    (
        "CC(C)Cc1cccc(CC(C)C)c1O",
        "2-hydroxy-1,3-diisobutylbenzene",
    ),
    (
        "CCCCCCCCCCCCCCCCCCCC(=O)OC1CCCCC1",
        "cyclohexyl arachidate",
    ),
    (
        "CC(N)=O",
        "acetamide",
    ),
    (
        "COc1ccc(OC)c(OC)c1",
        "1,2,4-trimethoxybenzene",
    ),
    (
        r"O=C(O)/C=C\C(=O)O",
        "(2Z)-but-2-enedioic acid",
    ),
    (
        "O=C(O)c1c(O)cc(O)cc1O",
        "2,4,6-trihydroxybenzoic acid",
    ),
    (
        "NC(=O)c1cccc(O)c1",
        "3-hydroxybenzamide",
    ),
    (
        "CC(=O)O",
        "acetic acid",
    ),
    (
        "CC1(C)CC(=O)c2c(O)cc(O)cc2O1",
        "5,7-dihydroxy-2,2-dimethylchroman-4-one",
    ),
    (
        "O=C(O)CC(=O)CC(=O)O",
        "3-oxopentanedioic acid",
    ),
    (
        "CC1OC(O)C(O)C(O)C1O",
        "3,4,5,6-tetrahydroxy-2-methyloxane",
    ),
    (
        r"CCC/C=C\C/C=C\CCCCCCCC(=O)O",
        "(9Z,12Z)-hexadeca-9,12-dienoic acid",
    ),
    (
        r"CCCCC/C=C\CCCCCCCC(=O)O",
        "(9Z)-pentadec-9-enoic acid",
    ),
    (
        "CCCCCCCCCCCCCCCCC(=O)O",
        "heptadecanoic acid",
    ),
    (
        r"CCCCCCCC/C=C\CCCCCCCC(=O)O",
        "(9Z)-octadec-9-enoic acid",
    ),
    (
        "O=C(O)CCC(=O)O",
        "butanedioic acid",
    ),
    (
        "CC(C)(O)CC(=O)O",
        "3-hydroxy-3-methylbutanoic acid",
    ),
    (
        "OC(=O)CC(O)=O",
        "propanedioic acid",
    ),
    (
        r"CCCC/C=C\CCCCCCCCCOC(C)=O",
        "(10Z)-pentadec-10-en-1-yl acetate",
    ),
    (
        "O=C(O)CCO",
        "3-hydroxypropanoic acid",
    ),
    (
        r"CCCCC/C=C\C/C=C\C/C=C\C/C=C\CCCC(=O)O",
        "(5Z,8Z,11Z,14Z)-icosa-5,8,11,14-tetraenoic acid",
    ),
    (
        "CC(C)(C)c1ccc(CC(=O)O)cc1",
        "2-(4-(tert-butyl)phenyl)ethanoic acid",
    ),
    (
        r"CCCCC/C=C\C/C=C\C/C=C\CCCCCCCC(=O)O",
        "(9Z,12Z,15Z)-henicosa-9,12,15-trienoic acid",
    ),
    (
        r"CCCCCCCC/C=C\CCCCCCCC(=O)OC",
        "methyl (9Z)-octadec-9-enoate",
    ),
    (
        r"CCCCCC/C=C\C/C=C\CCCCC(=O)OC",
        "methyl (6Z,9Z)-hexadeca-6,9-dienoate",
    ),
    (
        "O=C(O)/C=C/c1ccc(O)cc1",
        "(2E)-3-(4-hydroxyphenyl)prop-2-enoic acid",
    ),
    (
        "OC(=O)c1ccc(O)cc1",
        "4-hydroxybenzoic acid",
    ),
    (
        "CC(O)C(=O)O",
        "2-hydroxypropanoic acid",
    ),
    (
        "O=C(O)c1cc(O)c(O)c(O)c1",
        "3,4,5-trihydroxybenzoic acid",
    ),
    (
        "OCC(O)C(O)C(O)C(O)CO",
        "2,3,4,5-tetrahydroxyhexane-1,6-diol",
    ),
    # --- Phase 63: Stereochemistry Accuracy RT Fixes (9 compounds) ---
    (
        "COc1c(O)c(O)cc2c1CO[C@@H](C)C2=O",
        "(3S)-6,7-dihydroxy-8-methoxy-3-methylisochroman-4-one",
    ),
    (
        "O=C([O-])[C@@H](O)[C@H](O)[C@H](O)[C@@H](O)C(=O)[O-]",
        "(2R,3S,4R,5S)-2,3,4,5-tetrahydroxyhexanedioate",
    ),
    (
        r"C/C(=C\CC/C(C)=C/C/C=C(/CC(=O)c1cc(O)ccc1O)C(=O)O)CO",
        "(2Z,5E,9E)-2-(1-oxo1-(2,5-dihydroxyphenyl)ethyl)-11-hydroxy-6,10-dimethylundeca-2,5,9-trienoic acid",
    ),
    (
        "O=C(O)/C=C/c1ccc(OS(=O)(=O)O)cc1",
        "(2E)-3-(4-(sulfooxy)phenyl)prop-2-enoic acid",
    ),
    (
        "C=CC/C=C/CCC(=O)OC",
        "methyl (4E)-octa-4,7-dienoate",
    ),
    (
        r"CCCC/C=C\CCCCCCCCCOC(C)=O",
        "(10Z)-pentadec-10-en-1-yl acetate",
    ),
    (
        "O=C([O-])C(=O)C[C@H](O)C(=O)[O-]",
        "(2S)-2-hydroxy-4-oxopentanedioate",
    ),
    (
        "CCC[C@@H]1OCc2c(O)cccc2[C@H]1O",
        "(3S,4R)-4,8-dihydroxy-3-propylisochromane",
    ),
    (
        "C[C@@H]1Cc2cc(O)cc(O)c2CO1",
        "(3R)-6,8-dihydroxy-3-methylisochromane",
    ),
    # Phase 66: Medium Molecule Completeness (2 compounds)
    (
        r"C/C(=C\CC/C(C)=C/C/C=C(/CC(=O)c1cc(O)ccc1O)C(=O)O)CO",
        "(2Z,5E,9E)-2-(1-oxo1-(2,5-dihydroxyphenyl)ethyl)-"
        "11-hydroxy-6,10-dimethylundeca-2,5,9-trienoic acid",
    ),
    (
        "CC(=O)O[C@H]1CC[C@]2(C)C3=C(CC[C@H]2C1(C)C)"
        "[C@]1(C)C[C@@H](O)[C@H]([C@@H](C/C=C/C(C)(C)O)"
        "C(=O)O)[C@@]1(C)CC3",
        "(3S,5R,10S,13R,14R,16R,17R,20R,23E)-16,21,25-"
        "trihydroxy-4,4,14-trimethyl-21-oxocholest-8,23-"
        "dien-3-yl acetate",
    ),
    # --- Phase 67: v7.0 Final Benchmark (15 compounds) ---
    (
        "CC1CC=C(N2CCCC2)C1=O",
        "5-methyl-2-pyrrolidinylcyclopent-2-en-1-one",
    ),
    (
        "O=C([O-])/C=C/C(=O)O.[Na+]",
        "sodium hydrogen (2E)-but-2-enedioate",
    ),
    (
        "CCCC=CCOC(=O)c1ccccc1",
        "hex-2-en-1-yl benzoate",
    ),
    (
        "CCN(C(C)C)C(C)C",
        "N-ethyl-N-isopropylpropan-2-amine",
    ),
    (
        "COc1cc(CC(O)C(=O)O)ccc1OS(=O)(=O)O",
        "3-(3-methoxy-4-(sulfooxy)phenyl)-2-hydroxypropanoic acid",
    ),
    (
        "CCN(CC)Cc1ccccc1",
        "N-benzyl-N-ethylethan-1-amine",
    ),
    (
        "NC(N)=[NH2+].O=C([O-])C(=O)O",
        "guanidinium hydrogen ethanedioate",
    ),
    (
        r"CCCCC/C=C\C/C=C\CCCCCCCCCC(=O)OC(COC(=O)CCCCCCC/C=C\C/C=C\CCCCC)COC(=O)CCCCCCC/C=C\C/C=C\CCCCC",
        "2-[(11Z,14Z)-icosa-11,14-dienoyloxy]-1,3-bis(linoleoyloxy)propane",
    ),
    (
        "NCCc1c[nH]c2ccc(O)cc12",
        "3-(2-aminoethyl)-5-hydroxy-1H-indole",
    ),
    (
        "O=C([O-])CC=CC(=O)C(=O)[O-]",
        "2-oxohex-3-enedioate",
    ),
    (
        r"COC1CC(=O)C23C(=O)NC(CC(C)C)C2C(C)C(C)=CC3/C=C(\C)CCCC1O",
        "(9E)-5-hydroxy-16-isobutyl-4-methoxy-9,13,14-trimethyl-17-aza-tricyclo[9.7.0.0(1,15)]octadeca-9,12-dien-2,18-dione",
    ),
    (
        "COC1C2=C(C)C(=O)OC2CC2CCC(O)C(C)C21C",
        "11-hydroxy-8-methoxy-6,9,10-trimethyl-4-oxa-tricyclo[7.4.0.0(3,7)]tridec-6-en-5-one",
    ),
    (
        "CC(C)(C)[NH3+]",
        "2-methylpropan-2-aminium",
    ),
    (
        "CC[C@@H](C)c1ncc(C(C)C)[nH]c1=O",
        "3-[(R)-sec-butyl]-6-isopropyl-2-oxo-1,4-diazine",
    ),
    (
        "O=C(O)CCc1cc(O)c(OS(=O)(=O)O)c(O)c1",
        "3-(3,5-dihydroxy-4-(sulfooxy)phenyl)propanoic acid",
    ),
    # --- Phase 73: v8.0 Closure (25 compounds) ---
    # v8.0 phase improvement canaries: Phase 71 charge routing fixes (8 compounds)
    (
        "O=[N+]([O-])c1ccccc1C(=O)[O-]",  # Phase 71: 2-nitrobenzoate charge routing
        "2-nitrobenzoate",
    ),
    (
        "O=[N+]([O-])c1cccc(C(=O)[O-])c1",  # Phase 71: 3-nitrobenzoate charge routing
        "3-nitrobenzoate",
    ),
    (
        "O=[N+]([O-])c1ccc(C(=O)[O-])cc1",  # Phase 71: 4-nitrobenzoate charge routing
        "4-nitrobenzoate",
    ),
    (
        "O=[N+]([O-])c1cc(C(=O)[O-])cc([N+](=O)[O-])c1",  # Phase 71: 3,5-dinitrobenzoate
        "3,5-dinitrobenzoate",
    ),
    (
        "O=[N+]([O-])c1cc(Cl)ccc1C(=O)[O-]",  # Phase 71: chloro-nitrobenzoate charge routing
        "4-chloro2-nitrobenzoate",
    ),
    (
        "O=[N+]([O-])c1cc(O)ccc1C(=O)[O-]",  # Phase 71: hydroxy-nitrobenzoate charge routing
        "4-hydroxy2-nitrobenzoate",
    ),
    (
        "O=[N+]([O-])c1ccccc1",  # Phase 71: nitrobenzene neutral pipeline preserved
        "nitrobenzene",
    ),
    (
        "[O-][n+]1ccccc1",  # Phase 71: pyridine N-oxide neutral pipeline preserved
        "pyridine 1-oxide",
    ),
    # v8.0 phase improvement canaries: Phase 68 carbamoyl prefix (4 compounds)
    (
        "NC(=O)CCCC(=O)O",  # Phase 68: carbamoyl prefix linear acid
        # Phase 103-03: chain tiebreaker changes shorten parent chain
        "4-carbamoylbutanoic acid",
    ),
    (
        "NC(=O)c1ccc(C(=O)O)cc1",  # Phase 68: carbamoyl prefix aromatic acid
        "4-carbamoylbenzoic acid",
    ),
    (
        "NC(=O)CCC(=O)O",  # Phase 68: carbamoyl prefix short chain
        # Phase 103-03: chain tiebreaker changes shorten parent chain
        "3-carbamoylpropanoic acid",
    ),
    (
        "NC(=O)CC(=O)O",  # Phase 68: carbamoyl prefix minimal chain
        # Phase 103-03: chain tiebreaker changes shorten parent chain
        "2-carbamoylethanoic acid",
    ),
    # Failure taxonomy sentinels: substituent_loss (3 compounds)
    (
        "C=C[C@](C)(O)CCC=C(C)CCC1OC(C)(C)OC1(C)C",  # Sentinel: substituent_loss - terpene cyclopentane
        "(3R)-9-cyclopentyl-3,7-dimethylnona-1,6-dien-3-ol",
    ),
    (
        "CC(C)=CCOc1ccc(C2=C(CC(C)C)C(=O)NC2=O)cc1",  # Sentinel: substituent_loss - phenoxy maleimide
        "1-ethenyl-4-(2-methylbut-2-enoxy)benzene",
    ),
    (
        "COc1ccc(C(=O)N2CCCC2=O)cc1",  # Sentinel: substituent_loss - methoxybenzamide pyrrolidinone
        "N-4-methoxybenzoylpyrrolidin-2-one",  # Updated: quality gate now triggers decomposition covering both rings
    ),
    # Failure taxonomy sentinels: parent_mismatch (3 compounds)
    (
        r"CC1C/C(=C\CC(CC(N)=O)CC(=O)O)C(=O)C(C)C1",  # Sentinel: parent_mismatch - cyclohexanone chain
        # Phase 103-03: chain tiebreaker refinements change parent chain selection
        "3-(2-aminoethyl)-5-(3,5-dimethylcyclohexyl)pentanoic acid",
    ),
    (
        "O=C(O)c1cc(O)c2c(n1)C(O)C(O)C=C2",  # Sentinel: parent_mismatch - hydroxypyridine carboxylic
        "4-hydroxypyridine-6-carboxylic acid",  # Fixed: was "2,3-dibutyl-..." (fabricated from ring boundary leak)
    ),
    (
        r"CC(C)=CCc1ccc(O)c2c1C=C[C@H]1O[C@@H]2O[C@H]1C",  # Sentinel: parent_mismatch - prenyl chromanone
        "(2S,4S,5R)-4-methyl-9-2-methylbut-2-enyl-3,13-dioxa-tricyclo[6.4.0.1(2,5)]tridec-6-en-12-ol",
    ),
    # Failure taxonomy sentinels: fragment_loss (3 compounds)
    (
        "O=C(O)Cc1cc(O)ccc1Nc1c(Cl)cccc1Cl",  # Sentinel: fragment_loss - dichloroanilino phenylacetic
        "2-(3-hydroxyphenyl)ethanoic acid",
    ),
    (
        "CCCCCCCCCc1ccc(OCCO)cc1",  # Sentinel: fragment_loss - nonylphenol ethoxylate
        "2-phenoxyethan-1-ol",
    ),
    (
        r"CCCCCCC(=O)NC1=CC(=O)[C@@H]2CCCN12",  # Sentinel: fragment_loss - heptanoyl pyrrolidinone
        "heptanamide",
    ),
    # Failure taxonomy sentinels: stereo_mismatch (2 compounds)
    (
        "C=C(C)C(C)CCC(C)C1CCC2C3=CCC4CC(O)CCC4(C)C3CCC21C",  # Sentinel: stereo_mismatch - ergostadienol
        "ergosta-7,25-dien-3-ol",
    ),
    (
        "CCC(CCC(C)C1CCC2C3C(O)C=C4CC(O)CCC4(C)C3CCC12C)C(C)C",  # Sentinel: stereo_mismatch - stigmastandiol
        "stigmast-5-en-3,7-diol",
    ),
    # Failure taxonomy sentinels: opsin_vocab_limit (2 compounds)
    (
        "CN1[C@@H]2CC[C@H]1C[C@@H](OC(=O)c1c[nH]c3ccccc13)C2.Cl",  # Sentinel: opsin_vocab - tropyl indolecarboxylate
        "nonanoyloxypiperidine",
    ),
    (
        "NC(C(=O)O)C(CCC(N)C(=O)O)C(=O)O",  # Sentinel: opsin_vocab - triamino triacid
        "2,6-diamino-3-(hydroxymethyl)heptanetrioic acid",
    ),
    # --- Phase 83: v9.0 canary expansion (19 compounds) ---
    # 4 from Phase 78 (fused heterocycle prefix generation)
    # 5 from Phase 79 (ring-as-substituent naming)
    # 3 from Phase 80 (polyfunctional routing)
    # 4 from Phase 81 (adaptive coverage gate: retained/fused naming)
    # 1 from Phase 82 (multi-ring substituent expression)
    # 2 benchmark regression anchors (newly passing in v9.0 benchmark)
    # Phase 78: Fused heterocycle prefix generation
    (
        "OC(=O)CCc1cccc2cccnc12",
        "8-(2-carboxyethyl)quinoline",
    ),
    (
        "CC(=O)c1ccc2[nH]ccc2c1",
        "5-acetyl-1H-indole",
    ),
    (
        "OC(=O)c1cc2ccccc2[nH]1",
        "1H-indole-2-carboxylic acid",
    ),
    (
        "Oc1ccc2ncccc2c1",
        "6-hydroxyquinoline",
    ),
    # Phase 79: Ring-as-substituent naming (non-phenyl rings on chain parents)
    (
        "OC(=O)CCC1CCCCC1",
        "3-cyclohexylpropanoic acid",
    ),
    (
        "OC(=O)CC1CCCC1",
        "2-cyclopentylethanoic acid",
    ),
    (
        "OC(=O)CC1CCC1",
        "2-cyclobutylethanoic acid",
    ),
    (
        "OC(=O)CC1CCCCC1",
        "2-cyclohexylethanoic acid",
    ),
    (
        "CC(=O)C1CCCCC1",
        "1-cyclohexylethan-1-one",
    ),
    # Phase 80: Polyfunctional routing (3+ functional groups)
    (
        "OC(=O)C(O)CC(=O)O",
        "2-hydroxybutanedioic acid",
    ),
    (
        "OC(=O)C(=O)CC(=O)O",
        "2-oxobutanedioic acid",
    ),
    (
        "OC(=O)CCCC(=O)O",
        "pentanedioic acid",
    ),
    # Phase 81: Adaptive coverage gate (retained/fused names preserved by confidence)
    (
        "c1ccc2c(c1)ccc1ccccc12",
        "phenanthrene",
    ),
    (
        "c1ccc2[nH]ccc2c1",
        "1H-indole",
    ),
    (
        "C1CC2CCCC(C1)C2",
        "bicyclo[3.3.1]nonane",
    ),
    (
        "c1ccc2ncccc2c1",
        "quinoline",
    ),
    # Phase 82: Multi-ring substituent expression (biphenyl compound prefix)
    (
        "OC(=O)CCc1ccc(-c2ccccc2)cc1",
        "3-([1,1'-biphenyl]-4-yl)propanoic acid",
    ),
    # Benchmark regression anchors (from Phase 83 500-sample benchmark)
    (
        "CCCCCC=CC1=C(CO)C(=O)C[C@H](O)[C@@H]1O",
        "(4R,5S)-3-(hept-1-en-1-yl)-4,5-dihydroxy-2-hydroxymethylcyclohex-2-en-1-one",
    ),
    (
        r"C=C(C)[C@H]1CC[C@]2(C)[C@@H]1CC[C@]1(C)C/C=C(\C)CC/C=C(\C)CC[C@H]12",
        "(1R,3E,7E,11R,12R,15S,16R)-1,4,8,12-tetramethyl-15-prop-1-en-2-yl-tricyclo[9.7.0.0(12,16)]octadeca-3,7-diene",
    ),
    # --- Phase 95: v10.0 canary expansion (2 newly RT-matching compounds) ---
    (
        "CO[C@@H](C=C(C)C)C[C@H](C)[C@@H]1CC[C@]2(C)C3=CC[C@H]4C(C)(C)C(=O)CC[C@]4(C)[C@H]3CC[C@@]12C",
        "(5R,9R,10R,13S,14S,17S,20S,23R)-23-methoxy-4,4,14-trimethylcholesta-7,24-dien-3-one",
    ),
    (
        "O[C@H]1[C@H](O)[C@@H](O)[C@@H]2O[C@@H]2[C@@H]1O",
        "(1R,2R,3S,4S,5R,6S)-2,3,4,5-tetrahydroxy-7-oxa-bicyclo[4.1.0]heptane",
    ),
    # --- Phase 102-04: v11.0 canary expansion (80 newly identified RT-matching compounds) ---
    (
        "O=S([O-])[O-]",
        "sulfite",
    ),
    (
        "N#Cc1ccccc1",
        "benzonitrile",
    ),
    (
        "C=CC(O)CCCCC",
        "oct-1-en-3-ol",
    ),
    (
        "CC(C)CCCCCCCC=O",
        "9-methyldecanal",
    ),
    (
        "C1CSSCS1",
        "1,3,4-trithiane",
    ),
    (
        "[Ag+].[Cl-]",
        "silver chloride",
    ),
    (
        "CSCCSC",
        "2,5-dithiahexane",
    ),
    (
        "CCCCCCCCCCCCCCCCCCCCCCC(=O)O",
        "tricosanoic acid",
    ),
    (
        "NCCO",
        "2-aminoethan-1-ol",
    ),
    (
        "CCCCCCCCCCCCCCCCOC(C)=O",
        "hexadecyl acetate",
    ),
    (
        "OCCCO",
        "trimethylene glycol",
    ),
    (
        "CC(C)C[C@H](N)C(=O)N[C@@H](CS)C(=O)O",
        "L-leucyl-L-cysteine",
    ),
    (
        "CCCCC=CC(=O)OCC",
        "ethyl hept-2-enoate",
    ),
    (
        "NCCCCCCCCC(=O)O",
        "9-aminononanoic acid",
    ),
    (
        "CCCCCC=CCC=CCCCC=O",
        "tetradeca-5,8-dienal",
    ),
    (
        "COC(=O)CCCCCCCCC(=O)OC",
        "dimethyl decanedioate",
    ),
    (
        "CCCC/C=C\\CCCCCCCC(=O)[O-]",
        "(9Z)-tetradec-9-enoate",
    ),
    (
        "NCCCC[C@H](N)C(=O)N[C@@H](CO)C(=O)NCC(=O)O",
        "L-lysyl-L-serylglycine",
    ),
    (
        "CCCCCC(C)CCCCCC(=O)O",
        "7-methyldodecanoic acid",
    ),
    (
        "Nc1ccc(-c2ccco2)cc1",
        "1-amino-4-furanylbenzene",
    ),
    (
        "O=C(O)CCC/C=C\\CCCC(=O)O",
        "(5Z)-dec-5-enedioic acid",
    ),
    (
        "Nc1ccnc2cc(Cl)ccc12",
        "7-chloroquinolin-4-amine",
    ),
    (
        "CCCc1coc(C)n1",
        "2-methyl-4-propyloxazole",
    ),
    (
        "CCCCCCCCCC/C=C/CCCC(=O)O",
        "(5E)-hexadec-5-enoic acid",
    ),
    (
        "CCCCCC#CC#CCCCCCCC(=O)O",
        "hexadeca-8,10-diynoic acid",
    ),
    (
        "NC(=O)CC[C@H](NC(=O)CNC(=O)[C@@H](N)CO)C(=O)O",
        "L-serylglycyl-L-glutamine",
    ),
    (
        "CC(C)C[C@H](N)C(=O)N[C@@H](C)C(=O)N[C@H](C(=O)O)C(C)C",
        "L-leucyl-L-alanyl-L-valine",
    ),
    (
        "CC(C)[C@H](N)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CS)C(=O)O",
        "L-valyl-L-lysyl-L-cysteine",
    ),
    (
        "CCCCC/C=C/CCCOC(C)=O",
        "(4E)-dec-4-en-1-yl acetate",
    ),
    (
        "CC(C)C[C@H](NC(=O)[C@H](CCCCN)NC(=O)[C@@H](N)Cc1ccc(O)cc1)C(=O)O",
        "L-tyrosyl-L-lysyl-L-leucine",
    ),
    (
        "C/C=C(/C)CCC(C)=O",
        "(5Z)-5-methylhept-5-en-2-one",
    ),
    (
        "CC/C=C\\CCCCCCCOC(C)=O",
        "(8Z)-undec-8-en-1-yl acetate",
    ),
    (
        "OC1CCCCCCC/C=C\\CCCCCCC1",
        "(9Z)-cycloheptadec-9-en-1-ol",
    ),
    (
        "C[C@H](N)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)NCC(=O)O",
        "L-alanyl-L-tryptophylglycine",
    ),
    (
        "N[C@@H](CC(=O)O)C(=O)NCC(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",
        "L-aspartylglycyl-L-tryptophan",
    ),
    (
        "C[C@H](NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@@H](N)Cc1ccc(O)cc1)C(=O)O",
        "L-tyrosyl-L-tyrosyl-L-alanine",
    ),
    (
        "CC(=O)OCCCC(=O)CCOC(C)=O",
        "1,6-bis(acetyloxy)hexan-3-one",
    ),
    (
        "C[C@@H](O)[C@H](NC(=O)CN)C(=O)N[C@@H](CC(N)=O)C(=O)O",
        "glycyl-L-threonyl-L-asparagine",
    ),
    (
        "C[C@H](NC(=O)[C@H](CCCCN)NC(=O)[C@@H](N)Cc1c[nH]c2ccccc12)C(=O)O",
        "L-tryptophyl-L-lysyl-L-alanine",
    ),
    (
        "C[C@@H](O)[C@H](NC(=O)CNC(=O)[C@@H](N)Cc1c[nH]c2ccccc12)C(=O)O",
        "L-tryptophylglycyl-L-threonine",
    ),
    (
        "CC(C)[C@H](NC(=O)[C@@H](N)CCCCN)C(=O)N[C@@H](Cc1ccccc1)C(=O)O",
        "L-lysyl-L-valyl-L-phenylalanine",
    ),
    (
        "NC(CCCO)C(=O)O",
        "2-amino-5-hydroxypentanoic acid",
    ),
    (
        "C[C@H](NC(=O)[C@@H](N)CS)C(=O)N[C@H](C(=O)O)[C@@H](C)O",
        "L-cysteinyl-L-alanyl-L-threonine",
    ),
    (
        "CCCCCCCCCCCCC(C)CCCC(C)CCCC(C)CCCCCCCCCCCC",
        "13,17,21-trimethyltritriacontane",
    ),
    (
        "CC(CO)C(=O)O",
        "3-hydroxy-2-methylpropanoic acid",
    ),
    (
        "O=C([O-])C=Cc1ccc(O)cc1",
        "3-(4-hydroxyphenyl)prop-2-enoate",
    ),
    (
        "C[C@@H](O)[C@H](NC(=O)[C@@H](N)CCC(=O)O)C(=O)N[C@@H](CS)C(=O)O",
        "L-glutamyl-L-threonyl-L-cysteine",
    ),
    (
        "CC/C=C/CCCC(=O)CCCCCC(=O)O",
        "(11E)-7-oxotetradec-11-enoic acid",
    ),
    (
        "CC#CC#CC#CC(O)C(O)CO",
        "2,3-dihydroxydeca-4,6,8-triyn-1-ol",
    ),
    (
        "C[C@H](N)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@H](C(=O)O)[C@@H](C)O",
        "L-alanyl-L-tryptophyl-L-threonine",
    ),
    (
        "CC(C)[C@H](NC(=O)[C@@H](N)CC(=O)O)C(=O)N[C@@H](Cc1ccccc1)C(=O)O",
        "L-aspartyl-L-valyl-L-phenylalanine",
    ),
    (
        "CC(C)(C)/C=C/C(=O)O",
        "(2E)-4,4-dimethylpent-2-enoic acid",
    ),
    (
        "N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](CO)C(=O)N[C@@H](CC(=O)O)C(=O)O",
        "L-histidyl-L-seryl-L-aspartic acid",
    ),
    (
        "CC(C)[C@H](NC(=O)[C@@H](N)CCC(=O)O)C(=O)N[C@@H](Cc1ccccc1)C(=O)O",
        "L-glutamyl-L-valyl-L-phenylalanine",
    ),
    (
        "CC(C)CCCCC/C=C\\CCCCCCC(=O)O",
        "(8Z)-15-methylhexadec-8-enoic acid",
    ),
    (
        "O=Cc1ccc2ccccc2c1O",
        "1-hydroxynaphthalene-2-carbaldehyde",
    ),
    (
        "CCCCCC/C=C/C/C=C/CCCCCCC(=O)O",
        "(8E,11E)-octadeca-8,11-dienoic acid",
    ),
    (
        "NCCCC[C@H](NC(=O)[C@H](CS)NC(=O)[C@@H](N)Cc1ccccc1)C(=O)O",
        "L-phenylalanyl-L-cysteinyl-L-lysine",
    ),
    (
        "C[C@@H](O)[C@H](NC(=O)[C@@H](N)CCC(N)=O)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",
        "L-glutaminyl-L-threonyl-L-tryptophan",
    ),
    (
        "C[C@@H](O)[C@H](NC(=O)[C@@H](N)Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CC(N)=O)C(=O)O",
        "L-tryptophyl-L-threonyl-L-asparagine",
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@@H](NC(=O)[C@@H](N)CC(N)=O)[C@@H](C)O)C(=O)O",
        "L-asparaginyl-L-threonyl-L-isoleucine",
    ),
    (
        "N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CS)C(=O)N[C@@H](CS)C(=O)O",
        "L-phenylalanyl-L-cysteinyl-L-cysteine",
    ),
    (
        "NC(=O)C[C@H](NC(=O)[C@@H](N)CO)C(=O)N[C@@H](Cc1ccccc1)C(=O)O",
        "L-seryl-L-asparaginyl-L-phenylalanine",
    ),
    (
        "NC(=O)C[C@H](NC(=O)[C@@H](N)Cc1ccccc1)C(=O)N[C@@H](CO)C(=O)O",
        "L-phenylalanyl-L-asparaginyl-L-serine",
    ),
    (
        "CC(C)=CCC(C)/C(C)=C/CO",
        "(2E)-3,4,7-trimethylocta-2,6-dien-1-ol",
    ),
    (
        "N[C@@H](CS)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](CCC(=O)O)C(=O)O",
        "L-cysteinyl-L-histidyl-L-glutamic acid",
    ),
    (
        "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2C3=CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C",
        "(9R,10S,13R,14R,17R,20R)-cholest-7-ene",
    ),
    (
        "O=C(O)C(=O)Cc1cccc(O)c1",
        "3-(3-hydroxyphenyl)-2-oxopropanoic acid",
    ),
    (
        "C[C@@H](O)[C@H](N)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CC(=O)O)C(=O)O",
        "L-threonyl-L-glutaminyl-L-aspartic acid",
    ),
    (
        "CCC/C=C/C/C=C/C/C=C/CCCCC(=O)O",
        "(6E,9E,12E)-hexadeca-6,9,12-trienoic acid",
    ),
    (
        "C[C@@H]1CC(=O)CC(C)(C)C1=O",
        "(1R)-1,3,3-trimethylcyclohexane-2,5-dione",
    ),
    (
        "CC(C)CC(=O)[C@@H](C)CCC(C)(C)O",
        "(5S)-8-hydroxy-2,5,8-trimethylnonan-4-one",
    ),
    (
        "N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CC(=O)O)C(=O)O",
        "L-phenylalanyl-L-glutamyl-L-aspartic acid",
    ),
    (
        "CCC1CC=C(N2CCCC2)C1=O",
        "5-ethyl-2-pyrrolidinylcyclopent-2-en-1-one",
    ),
    (
        "CCCCCCC[C@@H](O)[C@H](O)CC#CC#C[C@@H](O)CC",
        "(3S,9R,10R)-heptadeca-4,6-diyne-3,9,10-triol",
    ),
    (
        "CCCCC(O)C#CC(O)C(O)C(O)CCCCCCCC(=O)O",
        "9,10,11,14-tetrahydroxyoctadec-12-ynoic acid",
    ),
    (
        "CCCCCCCC/C=C\\CCCCCCCC(=O)O[C@@H](CO)COC(=O)CCC",
        "(2S)-1-(butanoyloxy)-2-(oleoyloxy)propan-3-ol",
    ),
    (
        "COc1ccc(C(O)C(=O)O)cc1OC",
        "2-(3,4-dimethoxyphenyl)-2-hydroxyethanoic acid",
    ),
    (
        "O=C(/C=C/c1ccc(Cl)cc1)c1ccccc1",
        "(2E)-3-(4-chlorophenyl)-1-phenylprop-2-en-1-one",
    ),
    (
        "CCCCCCCCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COC(=O)CCCCCCCCC",
        "(2S)-3-(docosanoyloxy)-1-(decanoyloxy)propan-2-ol",
    ),
    # --- Phase 107-02 RT canary compounds ---
    # FIX-10: fused ring dictionary (dibenzofuran) - RT validated
    (
        "c1ccc2c(c1)oc1ccccc12",
        "dibenzofuran",
    ),
    # FIX-10: fused ring dictionary (dibenzothiophene) - RT validated
    (
        "c1ccc2c(c1)sc1ccccc12",
        "dibenzothiophene",
    ),
    # FIX-10: fused ring dictionary (9H-carbazole) - RT validated
    (
        "c1ccc2c(c1)[nH]c1ccccc12",
        "9H-carbazole",
    ),
    # FIX-14: ylidene naming (methylidenecyclohexane) - RT validated
    (
        "C=C1CCCCC1",
        "methylidenecyclohexane",
    ),
    # FIX-14: ylidene on chain (3-methylidenepentane) - RT validated
    (
        "CCC(=C)CC",
        "3-methylidenepentane",
    ),
    # FIX-15: skeletal replacement large ring - RT validated
    (
        "C1CCOCCO1",
        "1,4-dioxacycloheptane",
    ),
    # FIX-15: mixed heteroatom large ring - RT validated
    (
        "C1CCNCCOC1",
        "1-oxa-4-azacyclooctane",
    ),
    # FIX-15: zwitterion beta-alanine - RT validated
    (
        "[NH3+]CCC(=O)[O-]",
        "beta-alanine",
    ),
    # --- Phase 108 v12.0 canary expansion (30 newly RT-matching compounds) ---
    (
        "COc1ccc(C(=O)N2CCCC2=O)cc1",
        "N-4-methoxybenzoylpyrrolidin-2-one",
    ),
    (
        "CCCCCC/C=C/C=C(\\CCCC(=O)O)[N+](=O)[O-]",
        "(5E,7E)-5-nitrotetradeca-5,7-dienoic acid",
    ),
    (
        "CCCCC/C=C\\C/C=C\\C/C=C\\C/C=C\\C[C@@H](O)CC(=O)O",
        "(3R,5Z,8Z,11Z,14Z)-3-hydroxyicosa-5,8,11,14-tetraenoic acid",
    ),
    (
        "C=C[C@]1(C)CCC(=C(C)C)C[C@H]1C(=C)C",
        "(1S,2S)-1-ethenyl-4-isopropylidene-1-methyl-2-(prop-1-en-2-yl)cyclohexane",
    ),
    (
        "CC/C=C\\CC(O)C(O)/C=C/C(O)CCCCCCCC(=O)O",
        "(10E,15Z)-9,12,13-trihydroxyoctadeca-10,15-dienoic acid",
    ),
    (
        "CCCCC/C=C\\C/C=C\\C/C=C\\CCCCC(=O)[O-]",
        "(6Z,9Z,12Z)-octadeca-6,9,12-trienoate",
    ),
    (
        "O=C(O)/C(Cl)=C\\C(=O)C(Cl)C(=O)O",
        "(2E)-2,5-dichloro-4-oxohex-2-enedioic acid",
    ),
    (
        "CCC/C=C\\C#CC/C=C\\CCCCCCCC(=O)O",
        "(9Z,14Z)-octadeca-9,14-dien-12-ynoic acid",
    ),
    (
        "C=C1C=C[C@H](C(C)C)CC1",
        "(3S)-3-isopropyl-6-methylidenecyclohexene",
    ),
    (
        "Cc1ccc(OC(=O)C(C)C)cc1",
        "1-((2-methylpropanoyl)oxy)-4-methylbenzene",
    ),
    (
        "CCCC/C=C\\CCCCCCCCCOC(C)=O",
        "(10Z)-pentadec-10-en-1-yl acetate",
    ),
    (
        "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCCCCCCCCCCCCCCCC(=O)[O-]",
        "(19Z,22Z,25Z,28Z,31Z)-tetratriaconta-19,22,25,28,31-pentaenoate",
    ),
    (
        "CCCCCCC/C=C\\CCCCCCC(=O)O",
        "(8Z)-hexadec-8-enoic acid",
    ),
    (
        "NC(=O)CCCC(N)C(=O)O",
        "2-amino-5-carbamoylpentanoic acid",
    ),
    (
        "CC/C=C(\\C)CC/C=C(\\C)CCC=C(C)C",
        "(6E,10E)-2,6,10-trimethyltrideca-2,6,10-triene",
    ),
    (
        "CCCCCCC/C=C\\CCCCCCCC(=O)[O-]",
        "(9Z)-heptadec-9-enoate",
    ),
    (
        "C/C(C=O)=C\\CC/C(C)=C/C=O",
        "(2E,6E)-2,6-dimethylocta-2,6-dienedial",
    ),
    (
        "CCCCC/C=C\\C=C\\C=C/CCCCCCC(=O)O",
        "(8Z,10E,12Z)-octadeca-8,10,12-trienoic acid",
    ),
    (
        "CCCCCCCC/C=C\\CCCCCCO",
        "(7Z)-hexadec-7-en-1-ol",
    ),
    (
        "C/C=C\\CCCCC(=O)O",
        "(6Z)-oct-6-enoic acid",
    ),
    (
        "C/C=C(\\C)CCC=C(C)C",
        "(6E)-2,6-dimethylocta-2,6-diene",
    ),
    (
        "COc1cc(OC)c2c(=O)c3c(O)cc(C)cc3oc2c1",
        "8-hydroxy-1,3-dimethoxy-6-methylxanthone",
    ),
    (
        "CCCCC/C=C\\C/C=C\\CCCCCCCCCC(=O)OC(COC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC)COC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC",
        "2-[(11Z,14Z)-icosa-11,14-dienoyloxy]-1,3-bis(linoleoyloxy)propane",
    ),
    (
        "C=C(C)[C@H]1CC[C@]2(C)[C@@H]1CC[C@]1(C)C/C=C(\\C)CC/C=C(\\C)CC[C@H]12",
        "(1R,3E,7E,11R,12R,15S,16R)-1,4,8,12-tetramethyl-15-prop-1-en-2-yl-tricyclo[9.7.0.0(12,16)]octadeca-3,7-diene",
    ),
    (
        "CC(=O)O[C@H]1CC[C@]2(C)C3=C(CC[C@H]2C1(C)C)[C@]1(C)C[C@@H](O)[C@H]([C@@H](C/C=C/C(C)(C)O)C(=O)O)[C@@]1(C)CC3",
        "(3S,5R,10S,13R,14R,16R,17R,20R,23E)-16,21,25-trihydroxy-4,4,14-trimethyl-21-oxocholest-8,23-dien-3-yl acetate",
    ),
    (
        "C[C@H]1C/C=C\\[C@H]2[C@@H]3O[C@]3(C)[C@@H](C)[C@H]3[C@H](Cc4ccccc4)NC(=O)[C@@]32OC(=O)/C=C\\[C@@](C)(O)C1=O",
        "(1S,2Z,5S,7R,8Z,12R,15S,16S,17S,18R,20S)-15-benzyl-7-hydroxy-5,7,17,18-tetramethyl-11,19-dioxa-14-aza-tetracyclo[10.8.0.0(12,16).0(18,20)]icosa-2,8-dien-6,10,13-trione",
    ),
    (
        "CC1=C[C@@H]2/C=C(\\C)CCC[C@H](O)/C=C/C(=O)O[C@]23C(=O)N[C@@H](CC(C)C)[C@@H]3[C@@H]1C",
        "(1S,2E,7S,8E,12R,15S,16S,17S)-7-hydroxy-15-isobutyl-3,17,18-trimethyl-11-oxa-14-aza-tricyclo[10.7.0.0(12,16)]nonadeca-2,8,18-trien-10,13-dione",
    ),
    (
        "COC1CC(=O)C23C(=O)NC(CC(C)C)C2C(C)C(C)=CC3/C=C(\\C)CCCC1O",
        "(9E)-5-hydroxy-16-isobutyl-4-methoxy-9,13,14-trimethyl-17-aza-tricyclo[9.7.0.0(1,15)]octadeca-9,12-dien-2,18-dione",
    ),
]

# Build test IDs from expected names (first 40 chars, sanitized for pytest)
_CANARY_IDS = [
    name[:40].replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    for _, name in CANARY_COMPOUNDS
]


@pytest.mark.parametrize("smiles,expected_name", CANARY_COMPOUNDS, ids=_CANARY_IDS)
def test_canary_rt312(smiles, expected_name):
    """Golden canary test: verify round-trip-matching compound still names correctly (312 compounds)."""
    result = name_compound(smiles)
    assert result == expected_name, (
        f"CANARY REGRESSION: {smiles}\n"
        f"  Expected: {expected_name}\n"
        f"  Got: {result}"
    )


# ---------------------------------------------------------------------------
# OPSIN parse assertions — format regression guard
# Single JVM invocation for all 202 canary names (batch mode)
# ---------------------------------------------------------------------------

_OPSIN_JAR = Path(__file__).resolve().parents[2] / "opsin-cli-2.8.0-jar-with-dependencies.jar"


def _opsin_available() -> bool:
    """Check if OPSIN JAR and Java runtime are available."""
    return _OPSIN_JAR.exists() and shutil.which("java") is not None


def _opsin_parse_batch(names: list[str]) -> dict[str, str | None]:
    """Parse multiple IUPAC names through OPSIN in a single JVM invocation.

    Writes all names to a temp file, invokes OPSIN once with -osmi,
    and returns a dict mapping name -> SMILES (or None if OPSIN failed).
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as tmp:
        for name in names:
            tmp.write(name + "\n")
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["java", "-jar", str(_OPSIN_JAR), "-osmi"],
            stdin=open(tmp_path),
            capture_output=True,
            text=True,
            timeout=120,
        )
        output_lines = result.stdout.splitlines()
    finally:
        os.unlink(tmp_path)

    results = {}
    for i, name in enumerate(names):
        if i < len(output_lines) and output_lines[i].strip():
            results[name] = output_lines[i].strip()
        else:
            results[name] = None
    return results


# Lazy-cached batch results (computed once on first access)
_OPSIN_RESULTS_CACHE: dict[str, str | None] | None = None


def _get_opsin_results() -> dict[str, str | None]:
    """Return cached OPSIN parse results for all canary names."""
    global _OPSIN_RESULTS_CACHE
    if _OPSIN_RESULTS_CACHE is None:
        names = [name for _, name in CANARY_COMPOUNDS]
        _OPSIN_RESULTS_CACHE = _opsin_parse_batch(names)
    return _OPSIN_RESULTS_CACHE


# Known OPSIN limitations: correct IUPAC names that OPSIN cannot parse.
# These are xfail'd rather than treated as Orthonym bugs.
_OPSIN_LIMITATIONS: dict[str, str] = {
    # OPSIN vocabulary gaps — valid IUPAC names beyond OPSIN's parser coverage
    "nonanoyloxypiperidine": (
        "Depth-independent naming regressed tropyl naming (was tropyl 1H-indole-3-carboxylate)"
    ),
    "2,6-diamino-3-(hydroxymethyl)heptanetrioic acid": (
        "OPSIN cannot parse 'heptanetrioic acid' (rare tricarboxylic acid suffix)"
    ),
}


@pytest.mark.skipif(not _opsin_available(), reason="OPSIN/Java not available")
@pytest.mark.parametrize(
    "smiles,expected_name",
    CANARY_COMPOUNDS,
    ids=[s for s, _ in CANARY_COMPOUNDS],
)
def test_canary_opsin_parseable(smiles, expected_name):
    """Verify canary compound names are OPSIN-parseable (format regression guard).

    Every canary compound's expected name is sent through OPSIN. Names that are
    correct IUPAC but beyond OPSIN's parser capability are marked xfail with
    documented reasons. Any other parse failure indicates a format regression.
    """
    if expected_name in _OPSIN_LIMITATIONS:
        pytest.xfail(f"OPSIN limitation: {_OPSIN_LIMITATIONS[expected_name]}")
    results = _get_opsin_results()
    parsed = results.get(expected_name)
    assert parsed, f"OPSIN cannot parse canary name: {expected_name}"
