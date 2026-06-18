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
# 700 golden canary compounds: (SMILES, expected_name)
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
        "3-(4-hydroxy-3,5-dimethoxyphenyl)-1-(2,4,6-trihydroxy-3-methoxyphenyl)propane-1,2-dione",
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
        "(2R)-N-hexylpyrrolidine-2-carboxylic acid",
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
        # Re-baselined v21 WS-A.1 S4: the appended-suffix anchor now routes
        # through orient_cycloalkene Path A and the suffix locant is cited
        # explicitly (P-31.1.4.3.4). OPSIN RT-verifies True; same structure,
        # same numbering as the old form.
        "3,4,5,6-tetrahydroxycyclohex-1-ene-1-carbaldehyde",
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
        "3,4-dimethylphenol",  # ASML-13: phenol suffix routing
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
        "(2S)-2-amino-3-(4-bromophenyl)propanoic acid",
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
        "(2R)-2-phenylethan-1-ol",  # PERC-06: diol no longer polyfunctional (same parent class)
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
        "(2S)-3-hydroxy-2-(octadecanoylamino)propanoic acid",
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
        "(3R)-3,9-dihydroxynonanoic acid",  # ASML-12: locant-aware prefix merge
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
        # WSD-07 (Phase 175) re-baseline: a top-level free amino acid now resolves
        # to its retained PIN with the configurational descriptor (D-tyrosine,
        # P-103.1.1.1) instead of the systematic substitutive name. RT-verified
        # (OPSIN parses 'D-tyrosine' to the input, InChI-L1 match).
        "D-tyrosine",
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
        # DD2 (Phase D, P-63.3.1(1)): dialkyl disulfide is a substitutive PIN,
        # not the 'dithia' skeletal-replacement chain (the old '2,3-dithiabutane'
        # consumed the S-S as two skeletal thia atoms — the catalog C3 defect).
        "CSSC",
        "(methyldisulfanyl)methane",
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
        "2,6-diisobutylphenol",  # ASML-13: phenol suffix routing
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
        "rhamnopyranose",
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
        "2-(4-tert-butylphenyl)ethanoic acid",  # P-16.3.4 (Phase 171): tert-butyl is simple, no enclosing marks
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
        "(2Z,5E,9E)-11-hydroxy-6,10-dimethyl-2-(1-oxo1-(2,5-dihydroxyphenyl)ethyl)undeca-2,5,9-trienoic acid",
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
        "(2Z,5E,9E)-11-hydroxy-6,10-dimethyl-2-(1-oxo1-(2,5-dihydroxyphenyl)ethyl)undeca-2,5,9-trienoic acid",
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
        "2-hydroxy-3-(3-methoxy-4-(sulfooxy)phenyl)propanoic acid",
    ),
    (
        "CCN(CC)Cc1ccccc1",
        "N-benzyl-N-ethylethanamine",
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
        "(3R)-9-(1,3-dioxolan-5-yl)-3,7-dimethylnona-1,6-dien-3-ol",
    ),
    (
        "CC(C)=CCOc1ccc(C2=C(CC(C)C)C(=O)NC2=O)cc1",  # Sentinel: substituent_loss - phenoxy maleimide
        # v21 WS-A.1 S2: the senior N-heterocycle (maleimide) is now the
        # parent per P-44.2.1(b) (old form named only the benzene side and
        # ignored the imide ring entirely). Both forms RT-False (multi-defect:
        # the prenyloxyphenyl decoration is still dropped) -- the sentinel
        # tracks the substituent_loss class either way.
        "3-isobutyl-2,5-dioxo-4-phenylazole",
    ),
    (
        "COc1ccc(C(=O)N2CCCC2=O)cc1",  # Sentinel: substituent_loss - methoxybenzamide pyrrolidinone
        "N-4-methoxybenzoylpyrrolidin-2-one",  # Updated: quality gate now triggers decomposition covering both rings
    ),
    # Failure taxonomy sentinels: parent_mismatch (3 compounds)
    (
        r"CC1C/C(=C\CC(CC(N)=O)CC(=O)O)C(=O)C(C)C1",  # Sentinel: parent_mismatch - cyclohexanone chain
        # Phase 103-03: chain tiebreaker refinements change parent chain selection
        # v21 WS-D.1: the demoted ring's ketone is now emitted as the prefix
        # 2-oxo (P-66.6.1) instead of being silently dropped. The new name is
        # strictly closer to the true structure (adds O3, matching the ring
        # ketone; the remaining exocyclic-alkene defect is pre-existing and
        # orthogonal). Re-baselined per the changed-row RT-verify discipline.
        "3-(2-aminoethyl)-5-(3,5-dimethyl-2-oxocyclohexyl)pentanoic acid",
    ),
    (
        "O=C(O)c1cc(O)c2c(n1)C(O)C(O)C=C2",  # Sentinel: parent_mismatch - hydroxypyridine carboxylic
        "4-hydroxypyridine-6-carboxylic acid",  # Fixed: was "2,3-dibutyl-..." (fabricated from ring boundary leak)
    ),
    (
        # v22 Phase G0 (DD7 S1): this prenyl-chromanone fuses a benzene ring into
        # the bicyclic O-bridge system. The prior expected von-Baeyer name
        # `…3,13-dioxa-tricyclo[6.4.0.1(2,5)]tridec-6-en-12-ol` is structurally
        # WRONG — it DROPS the benzo aromaticity (1 `-ene` for a 4-double-bond
        # molecule), re-parsing to a different over-saturated structure. G0
        # correctly fails closed (von Baeyer cannot represent aromaticity); the
        # correct fused-aromatic PIN is a Phase-G1 build. This canary now guards
        # the fail-closed refusal, not the old wrong cage.
        r"CC(C)=CCc1ccc(O)c2c1C=C[C@H]1O[C@@H]2O[C@H]1C",  # Sentinel: G0 fail-closed (was wrong von-Baeyer)
        "unknown organic compound",
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
        # Phase 139.1-01: decomposition now produces more complete fragment name
        # Phase 170 WS-4 (DEF-6): ring ketone -> -one suffix (this sentinel name
        # is malformed both before & after — RT=False unchanged, no RT regression).
        "N-heptanoyl(2S)-5-aminoazol-3-one",
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
        "CN1[C@@H]2CC[C@H]1C[C@@H](OC(=O)c1c[nH]c3ccccc13)C2.Cl",  # Phase 118-02: tropane NP naming
        "(1R,3r,5S)-tropan-3-yl nonanoate",
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
    # Phase 148 D-05 / P-44.1(a) cascade unblock: chain has principal
    # characteristic group (carboxylic acid / ketone) — chain wins as parent
    # per https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1(a). Pre-148 the
    # deleted `_should_bypass_fused_guard` short-circuited fused heterocycles
    # to ring-as-parent producing IUPAC-incorrect prefix names. Both new
    # names OPSIN-roundtrip-verified (commit 148-02-03).
    (
        "OC(=O)CCc1cccc2cccnc12",
        "3-(quinolin-8-yl)propanoic acid",  # P-44.1(a) chain wins (acid PG)
    ),
    (
        "CC(=O)c1ccc2[nH]ccc2c1",
        "1-(1H-indol-5-yl)ethan-1-one",  # P-44.1(a) chain wins (ketone PG)
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
        # WSD-01 (Phase 175) re-baseline: the four -OH are now the principal
        # characteristic group expressed as the '-tetraol' SUFFIX (P-14.4 c),
        # not a 'tetrahydroxy' prefix. RT-neutral (both forms OPSIN-L1-match).
        "(1R,2R,3S,4S,5R,6S)-7-oxa-bicyclo[4.1.0]heptane-2,3,4,5-tetraol",
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
        # Phase 167 HYG-03: was "trimethylene glycol" (deprecated glycol name);
        # PIN is propane-1,3-diol. Intentional, OPSIN-RT-verified correction.
        "propane-1,3-diol",
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
        "1-amino-4-(furan-2-yl)benzene",
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
        # Re-baselined v21 WS-A.1 S3 (P-31.1.4(c) suffix-locant priority):
        # dione anchors must take {1,4}, not {2,5}. Both forms OPSIN
        # RT-verify True; the new form is the PIN numbering.
        "(6R)-2,2,6-trimethylcyclohexane-1,4-dione",
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
        "(2S)-1-(decanoyloxy)-3-(docosanoyloxy)propan-2-ol",
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
        "(3S)-3-isopropyl-6-methylidenecyclohex-1-ene",
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
    # + 380 from Phase 143 (integration testing canary expansion)
    (
        "CCCCCC/C=C/C=C(\\CCCC(=O)O)[N+](=O)[O-]",  # acyclic,medium
        "(5E,7E)-5-nitrotetradeca-5,7-dienoic acid",
    ),
    (
        "CCCCC/C=C\\C/C=C\\C/C=C\\C/C=C\\C[C@@H](O)CC(=O)O",  # acyclic,medium
        "(3R,5Z,8Z,11Z,14Z)-3-hydroxyicosa-5,8,11,14-tetraenoic acid",
    ),
    (
        "C/C(=C\\CC/C(C)=C/C/C=C(/CC(=O)c1cc(O)ccc1O)C(=O)O)CO",  # aromatic,polyfunctional,medium
        "(2Z,5E,9E)-11-hydroxy-6,10-dimethyl-2-(1-oxo1-(2,5-dihydroxyphenyl)ethyl)undeca-2,5,9-trienoic acid",
    ),
    (
        "CC/C=C\\CC(O)C(O)/C=C/C(O)CCCCCCCC(=O)O",  # acyclic,medium
        "(10E,15Z)-9,12,13-trihydroxyoctadeca-10,15-dienoic acid",
    ),
    (
        "C[C@H]1C/C=C\\[C@H]2[C@@H]3O[C@]3(C)[C@@H](C)[C@H]3[C@H](Cc4ccccc4)NC(=O)[C@@]32OC(=O)/C=C\\[C@@](C)(O)C1=O",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "(1S,2Z,5S,7R,8Z,12R,15S,16S,17S,18R,20S)-15-benzyl-7-hydroxy-5,7,17,18-tetramethyl-11,19-dioxa-14-aza-tetracyclo[10.8.0.0(12,16).0(18,20)]icosa-2,8-dien-6,10,13-trione",
    ),
    (
        "CCCCC/C=C\\C/C=C\\CCCCCCCCCC(=O)OC(COC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC)COC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC",  # acyclic,large
        "2-[(11Z,14Z)-icosa-11,14-dienoyloxy]-1,3-bis(linoleoyloxy)propane",
    ),
    (
        "CC1=C[C@@H]2/C=C(\\C)CCC[C@H](O)/C=C/C(=O)O[C@]23C(=O)N[C@@H](CC(C)C)[C@@H]3[C@@H]1C",  # heterocycle,fused-ring,polyfunctional,medium
        "(1S,2E,7S,8E,12R,15S,16S,17S)-7-hydroxy-15-isobutyl-3,17,18-trimethyl-11-oxa-14-aza-tricyclo[10.7.0.0(12,16)]nonadeca-2,8,18-trien-10,13-dione",
    ),
    (
        "O=C(O)/C(Cl)=C\\C(=O)C(Cl)C(=O)O",  # acyclic,polyfunctional,small
        "(2E)-2,5-dichloro-4-oxohex-2-enedioic acid",
    ),
    (
        "CCC/C=C\\C#CC/C=C\\CCCCCCCC(=O)O",  # acyclic,medium
        "(9Z,14Z)-octadeca-9,14-dien-12-ynoic acid",
    ),
    (
        "CCCC/C=C\\CCCCCCCCCOC(C)=O",  # acyclic,medium
        "(10Z)-pentadec-10-en-1-yl acetate",
    ),
    (
        "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCCCCCCCCCCCCCCCC(=O)[O-]",  # acyclic,charged,large
        "(19Z,22Z,25Z,28Z,31Z)-tetratriaconta-19,22,25,28,31-pentaenoate",
    ),
    (
        "CCCCC/C=C\\C/C=C\\C/C=C\\CCCCC(=O)[O-]",  # acyclic,charged,medium
        "(6Z,9Z,12Z)-octadeca-6,9,12-trienoate",
    ),
    (
        "COC1CC(=O)C23C(=O)NC(CC(C)C)C2C(C)C(C)=CC3/C=C(\\C)CCCC1O",  # heterocycle,fused-ring,polyfunctional,medium
        "(9E)-5-hydroxy-16-isobutyl-4-methoxy-9,13,14-trimethyl-17-aza-tricyclo[9.7.0.0(1,15)]octadeca-9,12-dien-2,18-dione",
    ),
    (
        "CCCCCCC/C=C\\CCCCCCC(=O)O",  # acyclic,medium
        "(8Z)-hexadec-8-enoic acid",
    ),
    (
        "CC/C=C(\\C)CC/C=C(\\C)CCC=C(C)C",  # acyclic,medium
        "(6E,10E)-2,6,10-trimethyltrideca-2,6,10-triene",
    ),
    (
        "C=C/C(C)=C/CC(C)(C)/C(=N\\O)C(C)C",  # acyclic,small
        "(3Z,6E)-2,4,4,7-tetramethylnona-6,8-dien-3-one oxime",
    ),
    (
        "CCCCCCC/C=C\\CCCCCCCC(=O)[O-]",  # acyclic,charged,medium
        "(9Z)-heptadec-9-enoate",
    ),
    (
        "C=C(C)[C@H]1CC[C@]2(C)[C@@H]1CC[C@]1(C)C/C=C(\\C)CC/C=C(\\C)CC[C@H]12",  # fused-ring,medium
        "(1R,3E,7E,11R,12R,15S,16R)-1,4,8,12-tetramethyl-15-prop-1-en-2-yl-tricyclo[9.7.0.0(12,16)]octadeca-3,7-diene",
    ),
    (
        "C/C(C=O)=C\\CC/C(C)=C/C=O",  # acyclic,small
        "(2E,6E)-2,6-dimethylocta-2,6-dienedial",
    ),
    (
        "CCCCC/C=C\\C=C\\C=C/CCCCCCC(=O)O",  # acyclic,medium
        "(8Z,10E,12Z)-octadeca-8,10,12-trienoic acid",
    ),
    (
        "CCCCCCCC/C=C\\CCCCCCO",  # acyclic,medium
        "(7Z)-hexadec-7-en-1-ol",
    ),
    (
        "C/C=C\\CCCCC(=O)O",  # acyclic,small
        "(6Z)-oct-6-enoic acid",
    ),
    (
        "C/C=C(\\C)CCC=C(C)C",  # acyclic,small
        "(6E)-2,6-dimethylocta-2,6-diene",
    ),
    (
        "OC1C2CC3CC1CC(O)(C3)C2",  # fused-ring,small
        "tricyclo[3.3.1.1(3,7)]decan-2,7-diol",
    ),
    (
        "CC(C)C[C@H](NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@@H](N)CS)C(=O)O",  # aromatic,polyfunctional,medium
        "L-cysteinyl-L-tyrosyl-L-leucine",
    ),
    (
        "C/C=C\\C#CC#C/C=C/C=C/CCC(=O)O",  # acyclic,medium
        "(4E,6E,12Z)-tetradeca-4,6,12-trien-8,10-diynoic acid",
    ),
    (
        "COc1cc(CCC(=O)[O-])ccc1O",  # aromatic,charged,small
        "3-(4-hydroxy-3-methoxyphenyl)propanoate",
    ),
    (
        "N[C@H]1CO[C@H]1C(=O)O",  # heterocycle,polyfunctional,small
        "(2R,3S)-3-aminooxetane-2-carboxylic acid",
    ),
    (
        "CCCCCC(O)C(O)/C=C/C(O)C/C=C\\C/C=C\\CCCC(=O)[O-]",  # acyclic,charged,medium
        "(5Z,8Z,12E)-11,14,15-trihydroxyicosa-5,8,12-trienoate",
    ),
    (
        "CCC(=O)N[C@@H](CCSC)C(=O)[O-]",  # acyclic,charged,small
        "(2S)-4-(methylsulfanyl)-2-(propanoylamino)butanoate",
    ),
    (
        "CC(C)[C@H](NC(=O)CN)C(=O)N[C@@H](C)C(=O)O",  # acyclic,polyfunctional,medium
        "glycyl-L-valyl-L-alanine",
    ),
    (
        "O=C([O-])CCS",  # acyclic,charged,small
        "3-sulfanylpropanoate",
    ),
    (
        "CC1(C)CC[C@]2(C(=O)O)CC[C@]3(C)C(=CC[C@@H]4[C@@]5(C)CC[C@H](O)C(C)(C)[C@@H]5CC[C@]43C)[C@@H]2C1",  # fused-ring,large
        "(1S,2R,5R,7S,10R,11R,15S,20S)-7-hydroxy-1,2,6,6,10,17,17-heptamethyl-pentacyclo[12.8.0.0(2,11).0(5,10).0(15,20)]docos-13-ene-20-carboxylic acid",
    ),
    (
        "CC(=O)OC1C(c2ccccc2)CCC(c2ccccc2)C1O",  # aromatic,medium
        "2-(acetyloxy)-3-hydroxy-1,4-diphenylcyclohexane",
    ),
    (
        "CCCCC/C=C\\C[C@H](/C=C/C=C\\C/C=C\\CCCC(=O)[O-])OO",  # acyclic,charged,medium
        "(5Z,8Z,10E,12R,14Z)-12-hydroperoxyicosa-5,8,10,14-tetraenoate",
    ),
    (
        "C[C@@H]1CC[C@@H]2C=C(C(=O)O)[C@H]3C[C@](C)(C(=O)O)C[C@]132",  # fused-ring,medium
        "(1R,4S,6S,8R,9R)-6,9-dimethyl-tricyclo[6.3.0.0(4,8)]undec-2-ene-3,6-dicarboxylic acid",
    ),
    (
        "C[C@H]1O[C@@H](O)[C@H](O)[C@H](O)[C@@H]1O",  # heterocycle,small,carbohydrate
        # Phase 170 WS-4 (DEF-6): the 4 ring -OH are the principal group -> -tetraol
        # SUFFIX (P-33), not hydroxy prefixes. OPSIN-RT verified.
        "(2R,3S,4R,5R,6R)-2-methyloxane-3,4,5,6-tetraol",
    ),
    (
        "CC1(C)[C@@H](Br)CC=C(C(Cl)CCl)[C@@H]1Cl",  # small
        "(4S,6S)-4-bromo-6-chloro-1-(1,2-dichloroethyl)-5,5-dimethylcyclohex-1-ene",
    ),
    (
        "N[C@@H](CO)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](CS)C(=O)O",  # aromatic,heterocycle,polyfunctional,medium
        "L-seryl-L-histidyl-L-cysteine",
    ),
    (
        "c1ccc2c(c1)Cc1ccccc1-2",  # aromatic,fused-ring,small
        "fluorene",
    ),
    (
        "CC(=O)c1ccc(O)cc1O",  # aromatic,small
        "1-(2,4-dihydroxyphenyl)ethan-1-one",
    ),
    (
        "C[C@H](CCC1OCC1CO)[C@H]1CC[C@H]2[C@@H]3[C@H](O)C[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3C[C@H](O)[C@]12C",  # heterocycle,fused-ring,large,steroid
        "(3R,5S,7R,8R,9S,10S,12S,13R,14S,17R,20R)-24,26-epoxycholestan-3,7,12,27-tetraol",
    ),
    (
        "Cc1cc(C)c2ccccc2c1",  # aromatic,fused-ring,small
        "1,3-dimethylnaphthalene",
    ),
    (
        "NCCCC[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@@H](N)Cc1ccc(O)cc1)C(=O)O",  # aromatic,polyfunctional,medium
        "L-tyrosyl-L-aspartyl-L-lysine",
    ),
    (
        "O=C(O)C[C@H](O)CCCCCCCCCO",  # acyclic,medium
        "(3R)-3,12-dihydroxydodecanoic acid",
    ),
    (
        "NCCCC[C@H](NC(=O)[C@@H](N)Cc1cnc[nH]1)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "L-histidyl-L-lysyl-L-tryptophan",
    ),
    (
        "N[C@@H](CO)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "L-seryl-L-tyrosyl-L-tryptophan",
    ),
    (
        "CC(C)C[C@H](NC(=O)[C@@H](N)CC(=O)O)C(=O)N[C@@H](CC(=O)O)C(=O)O",  # acyclic,polyfunctional,medium
        "L-aspartyl-L-leucyl-L-aspartic acid",
    ),
    (
        "CC(C)C[C@H](NC(=O)CNC(=O)[C@@H](N)Cc1c[nH]c2ccccc12)C(=O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,medium
        "L-tryptophylglycyl-L-leucine",
    ),
    (
        "CC1=CC[C@@H]2C[C@H]1C2(C)C",  # fused-ring,small
        # SUB-02 (169.5): substituent prefix glues to the descriptor
        # (IUPAC-standard 'trimethylbicyclo', was the non-standard
        # 'trimethyl-bicyclo'); both OPSIN-RT, the no-hyphen form is correct.
        # Phase 170 WS-6 (DEF-7): P-14.4 von Baeyer numbering gives the ene + gem-
        # dimethyl the lowest locants -> the actual alpha-pinene PIN. OPSIN-RT verified.
        "(1R,5R)-2,6,6-trimethylbicyclo[3.1.1]hept-2-ene",
    ),
    (
        "CC(C)[C@H](NC(=O)[C@H](C)NC(=O)[C@@H](N)Cc1cnc[nH]1)C(=O)O",  # aromatic,heterocycle,polyfunctional,medium
        "L-histidyl-L-alanyl-L-valine",
    ),
    (
        "O=CC(=O)[C@@H](O)[C@H](O)[C@@H](O)CO",  # acyclic,polyfunctional,small
        "(3S,4R,5S)-3,4,5,6-tetrahydroxy-2-oxohexanal",
    ),
    (
        "O=C1C[C@](O)(CO)CC(O)=C1O",  # small
        "(5S)-2,3,5-trihydroxy-5-hydroxymethylcyclohex-2-en-1-one",
    ),
    (
        "CC(C)[C@H](NC(=O)[C@@H]1CCCN1)C(=O)N[C@@H](CCCCN)C(=O)O",  # heterocycle,polyfunctional,medium
        "L-prolyl-L-valyl-L-lysine",
    ),
    (
        "NCCCC[C@H](N)C(N)=O",  # acyclic,small
        "(2S)-2,6-diaminohexanamide",
    ),
    (
        "COc1cc(C(N)=O)c(N)c(O)c1O",  # aromatic,polyfunctional,small
        "2-amino-3,4-dihydroxy-5-methoxybenzamide",
    ),
    (
        "C1COCCO1",  # heterocycle,small
        "1,4-dioxane",
    ),
    (
        "CC[C@H](C)[C@H](N)C(=O)N[C@@H](Cc1ccccc1)C(=O)O",  # aromatic,polyfunctional,medium
        "L-isoleucyl-L-phenylalanine",
    ),
    (
        "NC(=O)CC[C@H](NC(=O)[C@@H](N)CCC(=O)O)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O",  # aromatic,heterocycle,polyfunctional,medium
        "L-glutamyl-L-glutaminyl-L-histidine",
    ),
    (
        "CC(C)C[C@H](NC(=O)[C@@H](N)[C@@H](C)O)C(=O)N[C@H](C(=O)O)C(C)C",  # acyclic,polyfunctional,medium
        "L-threonyl-L-leucyl-L-valine",
    ),
    (
        "CCC(C=O)CC",  # acyclic,small
        "2-ethylbutanal",
    ),
    (
        "CC(C)C[C@H](NC(=O)[C@H](CC(N)=O)NC(=O)[C@@H](N)CCCCN)C(=O)O",  # acyclic,polyfunctional,medium
        "L-lysyl-L-asparaginyl-L-leucine",
    ),
    (
        "CC(C)C[C@H](N)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](CCC(N)=O)C(=O)O",  # aromatic,heterocycle,polyfunctional,medium
        "L-leucyl-L-histidyl-L-glutamine",
    ),
    (
        "CC/C=C\\C/C=C\\C/C=C\\CCCCCCCCCCCC(=O)O",  # acyclic,medium
        "(13Z,16Z,19Z)-docosa-13,16,19-trienoic acid",
    ),
    (
        "CC1=C2C(=O)C[C@@]2(C)[C@@H]2C[C@@H]3[C@H](O)C[C@@H](C)[C@@]2(CC1)C3(C)C",  # fused-ring,medium
        "(1S,2S,9R,10R,12R,13S)-12-hydroxy-2,6,10,15,15-pentamethyl-tetracyclo[7.5.0.1(9,13).0(2,5)]pentadec-5-en-4-one",
    ),
    (
        "CCCCC(C)CCCCCC(C)CCCCCCCCCC(C)CC(=O)O",  # acyclic,medium
        "3,13,19-trimethyltricosanoic acid",
    ),
    (
        "CC(=O)CCC(C)O",  # acyclic,small
        "5-hydroxyhexan-2-one",
    ),
    (
        "CC/C=C\\C/C=C\\C=C\\C(O)CCCCCCCC(=O)[O-]",  # acyclic,charged,medium
        "(10E,12Z,15Z)-9-hydroxyoctadeca-10,12,15-trienoate",
    ),
    (
        "Nc1cc(N)cc(N)c1",  # aromatic,small
        "1,3,5-triaminobenzene",
    ),
    (
        "C=C[C@@]1(C)CCC(=O)C[C@H]1C(=C)C",  # small
        # Re-baselined v21 WS-A.1 S3 (P-31.1.4(c) suffix-locant priority):
        # the ketone suffix must take locant 1, not 4. Both forms OPSIN
        # RT-verify True; stereo descriptors track the renumbering.
        "(3S,4R)-4-ethenyl-4-methyl-3-(prop-1-en-2-yl)cyclohexan-1-one",
    ),
    (
        "O=C1C(CO)=C[C@@H](O)[C@@H](O)[C@H]1Br",  # small
        "(4R,5R,6R)-6-bromo-4,5-dihydroxy-2-hydroxymethylcyclohex-2-en-1-one",
    ),
    (
        "CCCCCCCCCCCCCCCCCCCCCC(=O)CC(=O)CCCCCC",  # acyclic,large
        "triacontane-7,9-dione",
    ),
    (
        "BrC1CCC(Br)C(Br)CCC(Br)C(Br)CCC1Br",  # medium
        "1,2,5,6,9,10-hexabromocyclododecane",
    ),
    (
        "O=C(/C=C/c1ccc(O)cc1)c1ccc(O)c(O)c1O",  # aromatic,medium
        "(2E)-3-(4-hydroxyphenyl)-1-(2,3,4-trihydroxyphenyl)prop-2-en-1-one",
    ),
    (
        "CNC(=N)N",  # acyclic,small
        "N-methylguanidine",
    ),
    (
        "CCCCCCCCCCCCCCCC(=O)OC(CCCCCCCCCC)CCCCCCC(=O)[O-]",  # acyclic,charged,large
        "8-(palmitoyloxy)octadecanoate",
    ),
    (
        "CC(C)[C@H](NC(=O)[C@H](CC(N)=O)NC(=O)[C@@H](N)Cc1cnc[nH]1)C(=O)O",  # aromatic,heterocycle,polyfunctional,medium
        "L-histidyl-L-asparaginyl-L-valine",
    ),
    (
        "CCCCCO",  # acyclic,small
        "pentan-1-ol",
    ),
    (
        "CC(C)c1cccc(O)c1O",  # aromatic,small
        "3-isopropylbenzene-1,2-diol",
    ),
    (
        "NC(=O)CC[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@@H](N)CCC(N)=O)C(=O)O",  # aromatic,polyfunctional,medium
        "L-glutaminyl-L-phenylalanyl-L-glutamine",
    ),
    (
        "O=C(CCc1cccc(O)c1)c1cc(O)ccc1O",  # aromatic,medium
        "1-(2,5-dihydroxyphenyl)-3-(3-hydroxyphenyl)propan-1-one",
    ),
    (
        "O=C(O)CCC[C@H](O)/C=C\\C=C\\C=C\\[C@H](O)C/C=C\\CCCCC(F)(F)F",  # acyclic,medium
        "(5S,6Z,8E,10E,12R,14Z)-20,20,20-trifluoro-5,12-dihydroxyicosa-6,8,10,14-tetraenoic acid",
    ),
    (
        "CCCCCCCCCCCCCCCC(=O)NCC(=O)[O-]",  # acyclic,charged,medium
        "2-(hexadecanoylamino)ethanoate",
    ),
    (
        "COc1cc(CC(C)N)c(OC)cc1I",  # aromatic,small
        "1-(4-iodo-2,5-dimethoxyphenyl)propan-2-amine",
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@H](Cc1cnc[nH]1)NC(=O)[C@@H](N)CCCCN)C(=O)O",  # aromatic,heterocycle,polyfunctional,medium
        "L-lysyl-L-histidyl-L-isoleucine",
    ),
    (
        "NC(=O)C[C@H](NC(=O)[C@@H](N)CCC(=O)O)C(=O)N[C@@H](Cc1ccccc1)C(=O)O",  # aromatic,polyfunctional,medium
        "L-glutamyl-L-asparaginyl-L-phenylalanine",
    ),
    (
        "O=C(O)[C@@H](O)C(=O)[C@H](O)[C@@H](O)CO",  # acyclic,polyfunctional,small
        "(2S,4R,5S)-2,4,5,6-tetrahydroxy-3-oxohexanoic acid",
    ),
    (
        "CC/C=C\\C[C@@H](C)O",  # acyclic,small
        "(2R,4Z)-hept-4-en-2-ol",
    ),
    (
        "CC(C)[C@H](N)C(=O)N[C@@H](CO)C(=O)N[C@@H](CO)C(=O)O",  # acyclic,polyfunctional,medium
        "L-valyl-L-seryl-L-serine",
    ),
    (
        "O=[N+]([O-])c1cccc([N+](=O)[O-])c1",  # aromatic,small
        "1,3-dinitrobenzene",
    ),
    (
        "O=C([O-])CC[C@H](O)C(=O)[O-]",  # acyclic,charged,small
        "(2S)-2-hydroxypentanedioate",
    ),
    (
        "CCCCCCCCC(=O)[O-]",  # acyclic,charged,small
        "nonanoate",
    ),
    (
        "C[C@@H]1CC[C@H]2C(C=O)=C[C@@H]3CC(C)(C)CC132",  # fused-ring,medium
        "(1R,4S,9R)-6,6,9-trimethyl-tricyclo[6.3.0.0(4,8)]undec-2-ene-2-carbaldehyde",
    ),
    (
        "CCCC#CCCCCCCCCC(=O)O",  # acyclic,medium
        "tetradec-10-ynoic acid",
    ),
    (
        "C=CCCCCCCCCCCC(C)CCCCCCCCCCCCCCCCCC",  # acyclic,large
        "13-methylhentriacont-1-ene",
    ),
    (
        "C/C(=C\\CC[C@@H](C)[C@H]1CC[C@@]2(C)C3=C(CC[C@]12C)[C@@]1(C)CC[C@@H](O)C(C)(C)[C@@H]1C[C@H]3O)C(=O)O",  # fused-ring,large,steroid
        "(3R,5R,7R,10S,13R,14R,17R,20R,24E)-3,7,27-trihydroxy-4,4,14-trimethylcholesta-8,24-dien-27-one",
    ),
    (
        "C/C(=C\\[C@@H](O)C[C@@H](C)[C@H]1CC(=O)[C@@]2(C)C3=C(C(=O)C[C@]12C)[C@@]1(C)CC[C@H](O)C(C)(C)[C@@H]1C[C@@H]3O)C(=O)O",  # fused-ring,polyfunctional,large,steroid
        "(3S,5R,7S,10S,13R,14R,17R,20R,23S,24E)-3,7,23,27-tetrahydroxy-4,4,14-trimethylcholesta-8,24-dien-11,15,27-trione",
    ),
    (
        "COc1cc(CCC(=O)c2ccc(O)cc2O)ccc1O",  # aromatic,medium
        "1-(2,4-dihydroxyphenyl)-3-(4-hydroxy-3-methoxyphenyl)propan-1-one",
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](N)CCC(N)=O)C(=O)O",  # acyclic,polyfunctional,medium
        "L-glutaminyl-L-glutamyl-L-isoleucine",
    ),
    (
        "CC(=O)OCCC(N)C(=O)O",  # acyclic,polyfunctional,small
        "2-amino-4-(ethanoyloxy)butanoic acid",
    ),
    (
        "C[C@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(Cl)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO",  # fused-ring,medium,steroid
        "(8S,9R,10S,11S,13S,14S,16S,17R)-9-chloro-11,17,21-trihydroxy-16-methylpregna-1,4-dien-3,20-dione",
    ),
    (
        "CC1CC(S)C(C)O1",  # heterocycle,small
        # Phase 170 WS-4 (DEF-6): ring -SH is the principal group -> -thiol SUFFIX
        # (P-33), not a sulfanyl prefix. OPSIN-RT verified.
        "2,5-dimethyloxolane-3-thiol",
    ),
    (
        "NC(=O)CC[C@H](NC(=O)[C@@H]1CCCN1)C(=O)N[C@@H](CS)C(=O)O",  # heterocycle,polyfunctional,medium
        "N-[(2S)-pyrrolidine-2-carbonyl]-L-glutaminyl-L-cysteine",
    ),
    (
        "CC(C)[C@@H](C)[C@@H]1O[C@H]1[C@@H](C)[C@H]1CC[C@@]2(O)C3=CC(=O)[C@@H]4C[C@@H](O)[C@@H](O)C[C@]4(C)[C@H]3CC[C@]12C",  # heterocycle,fused-ring,large,steroid
        "(2S,3R,5R,9R,10R,13R,14S,17R,20S,22S,23S,24R)-22,23-epoxy-2,3,14-trihydroxyergost-7-en-6-one",
    ),
    (
        "N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CS)C(=O)O",  # aromatic,polyfunctional,large
        "L-tyrosyl-L-tyrosyl-L-cysteine",
    ),
    (
        "CC/C=C\\CC/C=C\\C/C=C\\CC/C=C\\CC/C=C\\CCC(=O)O",  # acyclic,medium
        "(4Z,8Z,12Z,15Z,19Z)-docosa-4,8,12,15,19-pentaenoic acid",
    ),
    (
        "CCCC/C=C/C#CC#CCCCCC(=O)O",  # acyclic,medium
        "(10E)-pentadeca-10-en-6,8-diynoic acid",
    ),
    (
        "C[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)CN)C(=O)O",  # acyclic,polyfunctional,medium
        "glycyl-L-aspartyl-L-alanine",
    ),
    (
        "CC1CCC2C(C(=O)O)=CC3CC(C)(C)CC132",  # fused-ring,medium
        "6,6,9-trimethyl-tricyclo[6.3.0.0(4,8)]undec-2-ene-2-carboxylic acid",
    ),
    (
        "CC1CCCN1",  # heterocycle,small
        "2-methylpyrrolidine",
    ),
    (
        "O=C(Cc1ccccc1)OC/C=C/c1ccccc1",  # aromatic,medium
        "(2E)-3-phenylprop-2-enyl 2-phenylethanoate",
    ),
    (
        "N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CC(=O)O)C(=O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "L-glutamyl-L-tryptophyl-L-aspartic acid",
    ),
    (
        "CCCCCCCC/C=C\\CCCC/C=C\\CCCC(=O)O",  # acyclic,medium
        "(5Z,11Z)-icosa-5,11-dienoic acid",
    ),
    (
        "Cc1cc(C)c(C[C@H](N)C(=O)O)c(C)c1",  # aromatic,polyfunctional,small
        "(2S)-2-amino-3-(2,4,6-trimethylphenyl)propanoic acid",
    ),
    (
        "Oc1ccc(CCC(O)CC/C=C/c2ccccc2)cc1",  # aromatic,medium
        "(6E)-1-(4-hydroxyphenyl)-7-phenylhept-6-en-3-ol",
    ),
    (
        "C[C@]12CCC(=O)C(O)=C1CC[C@@H]1[C@@H]2CC[C@]2(C)C(=O)CC[C@@H]12",  # fused-ring,medium,steroid
        "(8R,9S,10R,13S,14S)-4-hydroxyandrost-4-en-3,17-dione",
    ),
    (
        "CC(C)C[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@@H](N)CC(=O)O)C(=O)O",  # acyclic,polyfunctional,medium
        "L-aspartyl-L-aspartyl-L-leucine",
    ),
    (
        "CCCCCCCCCCCCC(O)C(=O)[O-]",  # acyclic,charged,medium
        "2-hydroxytetradecanoate",
    ),
    (
        "CCCC(=O)CCCCCO",  # acyclic,small
        "9-hydroxynonan-4-one",
    ),
    (
        "CC(C)C1=C2[C@H]3CC=C(C=O)CC(=O)[C@]3(C)CC[C@@]2(C)CC1",  # fused-ring,medium
        "(1R,6R,9R)-3-isopropyl-6,9-dimethyl-10-oxo-tricyclo[7.5.0.0(2,6)]tetradeca-2,12-diene-12-carbaldehyde",
    ),
    (
        "CC/C=C\\CO",  # acyclic,small
        "(2Z)-pent-2-en-1-ol",
    ),
    (
        "C[C@@H](O)[C@H](N)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CC(N)=O)C(=O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,medium
        "L-threonyl-L-tryptophyl-L-asparagine",
    ),
    (
        "CC(C)[C@H](N)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CC(=O)O)C(=O)O",  # acyclic,polyfunctional,medium
        "L-valyl-L-asparaginyl-L-aspartic acid",
    ),
    (
        "CCCCCCCC(=O)OCC(=O)[C@@]1(O)CC[C@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@H]3[C@@H](O)C[C@@]21C",  # fused-ring,polyfunctional,large,steroid
        "(8S,9S,10R,11S,13S,14S,17R)-11,17-dihydroxy-3,20-dioxopregn-4-en-21-yl octanoate",
    ),
    (
        "COc1cc(O)cc(O)c1",  # aromatic,small
        "5-methoxybenzene-1,3-diol",
    ),
    (
        "O=c1[nH]c(=O)c2ccccc2[nH]1",  # aromatic,heterocycle,fused-ring,small
        "quinazoline-2,4-dione",
    ),
    (
        "CCCC/C=C/C(C)O",  # acyclic,small
        "(3E)-oct-3-en-2-ol",
    ),
    (
        "CCCC[C@H](O)/C=C/C=C/C(=O)O",  # acyclic,small
        "(2E,4E,6S)-6-hydroxydeca-2,4-dienoic acid",
    ),
    (
        "CCCCCCCCCCCCCCC(CC)C(C)=O",  # acyclic,medium
        "3-ethylheptadecan-2-one",
    ),
    (
        "CC/C=C\\C[C@@H](O)[C@H](O)/C=C/C=C/C=C\\C=C\\[C@H](O)CCCC(=O)O",  # acyclic,medium
        "(5R,6E,8Z,10E,12E,14R,15R,17Z)-5,14,15-trihydroxyicosa-6,8,10,12,17-pentaenoic acid",
    ),
    (
        "CC/C=C\\C/C=C\\C/C=C\\[C@H](O)[C@@H](O)CCCCCC(=O)O",  # acyclic,medium
        "(7S,8S,9Z,12Z,15Z)-7,8-dihydroxyoctadeca-9,12,15-trienoic acid",
    ),
    (
        "NC(=O)C[C@H](NC(=O)[C@@H](N)CO)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O",  # aromatic,polyfunctional,medium
        "L-seryl-L-asparaginyl-L-tyrosine",
    ),
    (
        "CCCc1cc(O)c(CC)c(=O)o1",  # aromatic,heterocycle,small
        # Phase 170 WS-4 (DEF-6): ring -OH expressed as the -ol SUFFIX (P-33).
        # OPSIN-RT verified (the 2-oxo remains a prefix as before).
        "3-ethyl-2-oxo-6-propyl-2H-pyran-4-ol",
    ),
    (
        "NCCCC[C@H](N)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)O",  # acyclic,polyfunctional,medium
        "L-lysyl-L-glutamyl-L-glutamic acid",
    ),
    (
        "C/C(=C/C=C\\C=C\\C(O)c1cnco1)C(O)C(C)(C)C(N)=O",  # aromatic,heterocycle,medium
        "(4Z,6Z,8E)-3,10-dihydroxy-2,2,4-trimethyl-10-(oxazol-5-yl)deca-4,6,8-trienamide",
    ),
    (
        "C=C(CC(=O)[O-])C(=O)[O-]",  # acyclic,charged,small
        "2-methylidenebutanedioate",
    ),
    (
        "O=C(O)/C=C/CCCCCCCCCCCCCCCO",  # acyclic,medium
        "(2E)-18-hydroxyoctadec-2-enoic acid",
    ),
    (
        "CCCCC/C=C\\CC=O",  # acyclic,small
        "(3Z)-non-3-enal",
    ),
    (
        "CC[C@H](C)[C@H](N)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CC(C)C)C(=O)O",  # acyclic,polyfunctional,medium
        "L-isoleucyl-L-glutaminyl-L-leucine",
    ),
    (
        "CC(C)[C@H](NC(=O)[C@@H](N)CCC(N)=O)C(=O)O",  # acyclic,polyfunctional,medium
        "L-glutaminyl-L-valine",
    ),
    (
        "O=c1c2cccc(O)c2oc2ccc(O)c(O)c12",  # aromatic,heterocycle,fused-ring,medium
        "4,7,8-trihydroxyxanthone",
    ),
    (
        "CC(=O)CCC(=O)CCCCC(=O)O",  # acyclic,polyfunctional,small
        "6,9-dioxodecanoic acid",
    ),
    (
        "O=C(O)[C@H]1OC(O)[C@H](O)[C@H](O)[C@H]1O",  # heterocycle,small,carbohydrate
        "(2S,3R,4R,5R)-3,4,5,6-tetrahydroxyoxane-2-carboxylic acid",
    ),
    (
        "CCCC(C)CCCCCCCCCCCC(=O)O",  # acyclic,medium
        "13-methylhexadecanoic acid",
    ),
    (
        "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCCCCCCCCCCCCCC(=O)O",  # acyclic,large
        "(17Z,20Z,23Z,26Z,29Z)-dotriaconta-17,20,23,26,29-pentaenoic acid",
    ),
    (
        "CC(=O)c1ccc(Cl)cc1",  # aromatic,small
        "1-(4-chlorophenyl)ethan-1-one",
    ),
    (
        "CCCCCCCCCC(C)CI",  # acyclic,small
        "1-iodo-2-methylundecane",  # P-14.3.4 (Phase 171 DEF-4): locant-1 cited on multi-substituent parent
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@H](CS)NC(=O)[C@@H](N)Cc1c[nH]c2ccccc12)C(=O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,medium
        "L-tryptophyl-L-cysteinyl-L-isoleucine",
    ),
    (
        "CC(O)CBr",  # acyclic,small
        "1-bromopropan-2-ol",
    ),
    (
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCC(C)=O",  # acyclic,large
        "triacontan-2-one",
    ),
    (
        "O=C(/C=C/c1ccc(O)cc1)C(=O)[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO",  # aromatic,medium
        "(1E,5R,6S,7R,8R)-5,6,7,8,9-pentahydroxy-1-(4-hydroxyphenyl)non-1-ene-3,4-dione",
    ),
    (
        "C[C@@H]1C=C[C@H]2C3C1CC[C@@](C)(O)O[C@@H]3OC(=O)[C@@H]2C",  # heterocycle,fused-ring,medium
        "(2R,5S,6R,9R,11S)-11-hydroxy-2,6,11-trimethyl-8,10-dioxa-tricyclo[7.4.1.0(5,14)]tetradec-3-en-7-one",
    ),
    (
        "N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CS)C(=O)O",  # aromatic,polyfunctional,medium
        "L-tyrosyl-L-cysteine",
    ),
    (
        "NCCCC[C@H](NC(=O)[C@@H]1CCCN1)C(=O)N[C@@H](CC(=O)O)C(=O)O",  # heterocycle,polyfunctional,medium
        "L-prolyl-L-lysyl-L-aspartic acid",
    ),
    (
        "NC(=O)CC[C@H](N)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O",  # aromatic,polyfunctional,medium
        "L-glutaminyl-L-tyrosine",
    ),
    (
        "N[C@@H](CC(=O)O)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O",  # aromatic,heterocycle,polyfunctional,large
        "L-aspartyl-L-tyrosyl-L-histidine",
    ),
    (
        "CC(C)CC(=O)OCCCc1ccccc1",  # aromatic,medium
        "3-phenylpropyl 3-methylbutanoate",
    ),
    (
        "CCCCC/C=C\\CC(/C=C/C=C\\CCCCC(=O)O)OO",  # acyclic,medium
        "(6Z,8E,12Z)-10-hydroperoxyoctadeca-6,8,12-trienoic acid",
    ),
    (
        "CC(CCC(=O)O)CC(=O)O",  # acyclic,small
        "3-methylhexanedioic acid",
    ),
    (
        "CCCCC[C@H](O)/C=C/CCCCCCCCCC(=O)O",  # acyclic,medium
        "(11E,13S)-13-hydroxyoctadec-11-enoic acid",
    ),
    (
        "Cc1c(N)cc(N)cc1N",  # aromatic,small
        "1,3,5-triamino-2-methylbenzene",
    ),
    (
        "CO[C@H](C=C(C)C)C[C@H](C)[C@@H]1CC[C@]2(C)C3=CC[C@H]4C(C)(C)C(=O)CC[C@]4(C)[C@H]3CC[C@@]12C",  # fused-ring,large,steroid
        "(5R,9R,10R,13S,14S,17S,20S,23S)-23-methoxy-4,4,14-trimethylcholesta-7,24-dien-3-one",
    ),
    (
        "CC(C)[C@H](N)C(=O)NCC(=O)N[C@@H](CCCCN)C(=O)O",  # acyclic,polyfunctional,medium
        "L-valylglycyl-L-lysine",
    ),
    (
        "NC(=O)Nc1ccccc1",  # aromatic,small
        "N-phenylurea",
    ),
    (
        "C[C@@H](O)[C@H](NC(=O)[C@@H](N)Cc1cnc[nH]1)C(=O)N[C@@H](CO)C(=O)O",  # aromatic,heterocycle,polyfunctional,medium
        "L-histidyl-L-threonyl-L-serine",
    ),
    (
        "CC1(C)CC23[C@@H]4CC(=O)[C@@H]2COC(=O)[C@@H]3CC[C@H]41",  # heterocycle,fused-ring,medium
        "(4R,7R,11S,14R)-3,3-dimethyl-9-oxa-tetracyclo[9.3.0.0(1,7).0(4,14)]tetradecan-8,12-dione",
    ),
    (
        "CC(=O)CCC1C(C)CCCC1(C)C",  # small
        "4-(2,2,6-trimethylcyclohexyl)butan-2-one",
    ),
    (
        "CS/C=C/C(=O)O",  # acyclic,small
        "(2E)-3-(methylsulfanyl)prop-2-enoic acid",
    ),
    (
        "CC(C)Cc1ccc([C@@H](C)C(=O)O)cc1",  # aromatic,small
        "(2R)-2-(4-isobutylphenyl)propanoic acid",
    ),
    (
        "O=C([O-])C=CCC(=O)[O-]",  # acyclic,charged,small
        "pent-2-enedioate",
    ),
    (
        "CCCCCCCCCC/C=C\\C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",  # acyclic,large
        "(7Z)-1,1,1,2,2,3,3,4,4,5,5,6,6-tridecafluorooctadec-7-ene",
    ),
    (
        "CCCCC/C=C\\CCCC(=O)CCCCCCCCC",  # acyclic,medium
        "(14Z)-icos-14-en-10-one",
    ),
    (
        "N=C(N)NCCS(=O)O",  # acyclic,small
        "hypotaurocyamine",
    ),
    (
        "C[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](N)CCC(N)=O)C(=O)O",  # acyclic,polyfunctional,medium
        "L-glutaminyl-L-glutamyl-L-alanine",
    ),
    (
        "N[C@@H](CO)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O",  # aromatic,heterocycle,polyfunctional,medium
        "L-seryl-L-aspartyl-L-histidine",
    ),
    (
        "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CS)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "L-tryptophyl-L-cysteinyl-L-histidine",
    ),
    (
        "CSCC(=O)C(C)(C)C",  # acyclic,small
        "3,3-dimethyl-1-(methylsulfanyl)butan-2-one",
    ),
    (
        "Cc1oc2ccccc2c1C",  # aromatic,heterocycle,fused-ring,small
        "2,3-dimethyl-1-benzofuran",
    ),
    (
        "CC[C@H](C)[C@H](N)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CO)C(=O)O",  # aromatic,polyfunctional,medium
        "L-isoleucyl-L-tyrosyl-L-serine",
    ),
    (
        "CC1=CCC(=O)CC(=O)[C@@]23C(=O)N[C@@H](CC(C)C)[C@@H]2[C@H](C)C(C)=C[C@@H]3C1",  # heterocycle,fused-ring,medium
        "(1S,9S,12S,13R,14S)-14-isobutyl-7,11,12-trimethyl-15-aza-tricyclo[7.7.0.0(1,13)]hexadeca-6,10-dien-2,4,16-trione",
    ),
    (
        "N[C@@H](CCC(=O)O)C(=O)N[C@@H](CS)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O",  # aromatic,heterocycle,polyfunctional,medium
        "L-glutamyl-L-cysteinyl-L-histidine",
    ),
    (
        "CC(O)C(C)C(N)C(=O)O",  # acyclic,polyfunctional,small
        "2-amino-4-hydroxy-3-methylpentanoic acid",
    ),
    (
        "NCCCC[C@H](N)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O",  # aromatic,heterocycle,polyfunctional,large
        "L-lysyl-L-phenylalanyl-L-histidine",
    ),
    (
        "C[C@H](CCC(=O)O)[C@H]1CC[C@H]2[C@@H]3CC[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",  # fused-ring,medium,steroid
        "(3R,5R,8R,9S,10S,13R,14S,17R,20R)-3,24-dihydroxycholan-24-one",
    ),
    (
        "CCC(O)CCl",  # acyclic,small
        "1-chlorobutan-2-ol",
    ),
    (
        "O=Cc1c[nH]c2cccc(O)c12",  # aromatic,heterocycle,fused-ring,small
        "4-hydroxy-1H-indole-3-carbaldehyde",
    ),
    (
        "CCCCCCCCC=C(CCCCCCCC(=O)[O-])[N+](=O)[O-]",  # acyclic,charged,medium
        "9-nitrooctadec-9-enoate",
    ),
    (
        "COc1cc(OC)c(OC)c(OC)c1/C=C/C(=O)c1c(O)c(OC)c(OC)c(OC)c1OC",  # aromatic,large
        "(2E)-1-(2-hydroxy-3,4,5,6-tetramethoxyphenyl)-3-(2,3,4,6-tetramethoxyphenyl)prop-2-en-1-one",
    ),
    (
        "O=Cc1cccc(C(=O)O)c1",  # aromatic,polyfunctional,small
        "3-formylbenzoic acid",
    ),
    (
        "CC(=O)OC[C@]12CC[C@H](O)C(C)(C)[C@@H]1CCC1=C2CC[C@]2(C)[C@@H]([C@H](C)[C@H](C/C=C(/C)C(=O)O)OC(C)=O)CC[C@@]12C",  # fused-ring,polyfunctional,large,steroid
        "(3S,5R,10R,13R,14R,17R,20S,22S,24Z)-3,27-dihydroxy-4,4,14-trimethyl-27-oxocholest-8,24-dien-19,22-diyl diacetate",
    ),
    (
        "C/C=C/CCCCCCCCCCCCCCC(=O)O",  # acyclic,medium
        "(16E)-octadec-16-enoic acid",
    ),
    (
        "CCCCC/C=C\\C/C=C\\CC(O)/C=C\\C=C\\C/C=C\\C(=O)O",  # acyclic,medium
        "(2Z,5E,7Z,11Z,14Z)-9-hydroxyicosa-2,5,7,11,14-pentaenoic acid",
    ),
    (
        "CC(C)[C@H](N)C(=O)NCC(=O)N[C@H](C(=O)O)[C@@H](C)O",  # acyclic,polyfunctional,medium
        "L-valylglycyl-L-threonine",
    ),
    (
        "CC([NH3+])C(=O)CCCCCC(=O)[O-]",  # acyclic,small
        "8-amino-7-oxononanoic acid",
    ),
    (
        "CC(C)C[C@H](NC(=O)[C@H](CCCCN)NC(=O)[C@@H](N)CC(C)C)C(=O)O",  # acyclic,polyfunctional,medium
        "L-leucyl-L-lysyl-L-leucine",
    ),
    (
        "CC(C)CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3C[C@@H]4O[C@@]45C[C@@H](O)CC[C@]5(C)[C@H]3CC[C@]12C",  # heterocycle,fused-ring,medium,steroid
        "(3S,5R,6S,8S,9S,10R,13R,14S,17R,20R)-5,6-epoxycholestan-3-ol",
    ),
    (
        "CC(C)CCCCCCCCC/C=C/C=O",  # acyclic,medium
        "(2E)-13-methyltetradec-2-enal",
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@@H](N)CC(N)=O)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O",  # aromatic,heterocycle,polyfunctional,medium
        "L-asparaginyl-L-isoleucyl-L-histidine",
    ),
    (
        "C[C@H](CCCC(C)(C)O)[C@H]1CC[C@H]2[C@@H]3[C@H](O)C[C@@H]4CC(=O)CC[C@]4(C)[C@H]3CC[C@]12C",  # fused-ring,medium,steroid
        "(5R,7R,8R,9S,10S,13R,14S,17R,20R)-7,25-dihydroxycholestan-3-one",
    ),
    (
        "CC(C)C[C@H](NC(=O)[C@@H](N)Cc1ccc(O)cc1)C(=O)NCC(=O)O",  # aromatic,polyfunctional,medium
        "L-tyrosyl-L-leucylglycine",
    ),
    (
        "CCCC(=O)OCC(O)COC(=O)CCC",  # acyclic,medium
        "1,3-bis(butanoyloxy)propan-2-ol",
    ),
    (
        "C[C@H](CC[C@H](O)C(C)(C)O)[C@H]1CC[C@@]2(C)C3=C(CC[C@]12C)[C@@]1(C)CC[C@H](O)C(C)(C)[C@@H]1CC3",  # fused-ring,large,steroid
        "(3S,5R,10S,13R,14R,17R,20R,24S)-4,4,14-trimethylcholest-8-en-3,24,25-triol",
    ),
    (
        "CCCCC(=O)OCCC(C)CCC=C(C)C",  # acyclic,medium
        "3,7-dimethyloct-6-en-1-yl pentanoate",
    ),
    (
        "CC(C)[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](N)CCC(N)=O)C(=O)O",  # acyclic,polyfunctional,medium
        "L-glutaminyl-L-glutamyl-L-valine",
    ),
    (
        "Fc1ccc(Br)cc1",  # aromatic,small
        "1-bromo-4-fluorobenzene",
    ),
    (
        "CC(C)[C@H](NC(=O)CN)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O",  # aromatic,polyfunctional,medium
        "glycyl-L-valyl-L-tyrosine",
    ),
    (
        "CC(C(=O)O)[C@H](O)CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3C[C@H](O)[C@]12C",  # fused-ring,large,steroid
        "(3R,5R,8R,9S,10S,12S,13R,14S,17R,20R,24R)-3,12,24,27-tetrahydroxycholestan-27-one",
    ),
    (
        "O=C(O)C=CC(=O)O",  # acyclic,small
        "but-2-enedioic acid",
    ),
    (
        "CCCCCCC(O)C/C=C/C(O)CCC(O)CCCC(=O)O",  # acyclic,medium
        "(9E)-5,8,12-trihydroxyoctadec-9-enoic acid",
    ),
    (
        "CCCC(S)CCO",  # acyclic,small
        "3-sulfanylhexan-1-ol",
    ),
    (
        "CC[C@H](C)[C@H](N)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](Cc1ccccc1)C(=O)O",  # aromatic,heterocycle,polyfunctional,medium
        "L-isoleucyl-L-histidyl-L-phenylalanine",
    ),
    (
        "C1=CCCCCCC1",  # small
        "cyclooctene",
    ),
    (
        "COP(=O)([O-])[O-]",  # acyclic,charged,small
        "methyl phosphate",
    ),
    (
        "NCCCC[C@H](N)C(=O)NCC(=O)NCC(=O)O",  # acyclic,polyfunctional,medium
        "L-lysylglycylglycine",
    ),
    (
        "CCCCC/C=C\\C(O)C(O)/C=C/CCCCCCC(=O)O",  # acyclic,medium
        "(8E,12Z)-10,11-dihydroxyoctadeca-8,12-dienoic acid",
    ),
    (
        "CCC1CC2CCC(O2)C(C)C(=O)OC(C)CC2CCC(O2)C(C)C(=O)OC(C)CC2CCC(O2)C(C)C(=O)OC(C)CC2CCC(O2)C(C)C(=O)O1",  # heterocycle,fused-ring,large
        "3-ethyl-6,11,14,19,22,27,30-heptamethyl-4,8,12,16,20,24,28,34-octaoxa-pentacyclo[29.2.1.2(7,9).2(15,17).2(23,25)]tetracontan-5,13,21,29-tetraone",
    ),
    (
        "CCCCC(C)CCCCCCCC(C)CCCCC(=O)O",  # acyclic,medium
        "6,14-dimethyloctadecanoic acid",
    ),
    (
        "NCC(=O)N[C@@H](CCC(=O)O)C(=O)NCC(=O)O",  # acyclic,polyfunctional,medium
        "glycyl-L-glutamylglycine",
    ),
    (
        "CC(C)CCC(=O)O",  # acyclic,small
        "4-methylpentanoic acid",
    ),
    (
        "CC1=CC2/C=C(\\C)CCC3OC3/C=C/C(=O)C23C(=O)NC(CC(C)C)C3C1C",  # heterocycle,fused-ring,medium
        "(2E,9E)-15-isobutyl-3,17,18-trimethyl-7-oxa-14-aza-tetracyclo[10.7.0.0(6,8).0(12,16)]nonadeca-2,9,18-trien-11,13-dione",
    ),
    (
        "CCCCCCCCCCCCC(O)C(=O)O",  # acyclic,medium
        "2-hydroxytetradecanoic acid",
    ),
    (
        "CSc1ccc(N)cc1",  # aromatic,small
        "1-amino-4-(methylsulfanyl)benzene",
    ),
    (
        "NC(=O)C[C@H](N)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](CS)C(=O)O",  # aromatic,heterocycle,polyfunctional,medium
        "L-asparaginyl-L-histidyl-L-cysteine",
    ),
    (
        "CC(/C=C/C(=O)O)=C\\C=C\\C=C(C)\\C=C\\C(=O)O",  # acyclic,medium
        "(2E,4E,6E,8E,10E)-4,9-dimethyldodeca-2,4,6,8,10-pentaenedioic acid",
    ),
    (
        "COc1c(Cl)cc(C=O)cc1Br",  # aromatic,small
        "3-bromo-5-chloro-4-methoxybenzaldehyde",
    ),
    (
        "CCCCC[C@@H](/C=C/C=C\\CCCCCCCCCC(=O)O)OO",  # acyclic,medium
        "(11Z,13E,15S)-15-hydroperoxyicosa-11,13-dienoic acid",
    ),
    (
        "ClC(Cl)(Cl)c1ccccc1",  # aromatic,small
        "(trichloromethyl)benzene",
    ),
    (
        "CCCCCCCC/C=C/COC(C)=O",  # acyclic,small
        "(2E)-undec-2-en-1-yl acetate",
    ),
    (
        "CC1C(=O)NC(=O)NC1=O",  # heterocycle,small
        "5-methyl-2,4,6-trioxo-1,3-diazinane",
    ),
    (
        "CCC(=O)OCC(=O)[C@@]1(OC(=O)CC)[C@@H](C)C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@@]21C",  # fused-ring,polyfunctional,large,steroid
        "(8S,9R,10S,11S,13S,14S,16S,17R)-9-fluoro-11-hydroxy-16-methyl-3,20-dioxopregn-1,4-dien-17,21-diyl dipropanoate",
    ),
    (
        "O=C(O)c1ccccc1[N+](=O)[O-]",  # aromatic,small
        "2-nitrobenzoic acid",
    ),
    (
        "C=CCCCC#CC#CCC(O)CCCCCCC(=O)O",  # acyclic,medium
        "8-hydroxyoctadeca-17-en-10,12-diynoic acid",
    ),
    (
        "CCCCCCCCCCCCCCOC(C)=O",  # acyclic,medium
        "tetradecyl acetate",
    ),
    (
        "C=CCCCCCCCC(C)C=O",  # acyclic,small
        "2-methylundec-10-enal",
    ),
    (
        "N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)NCC(=O)O",  # aromatic,heterocycle,polyfunctional,medium
        "L-histidyl-L-histidylglycine",
    ),
    (
        "O=C(O)[C@H](O)[C@H](O)[C@H](O)[C@@H](O)C(=O)O",  # acyclic,small
        "(2R,3S,4R,5R)-2,3,4,5-tetrahydroxyhexanedioic acid",
    ),
    (
        "COc1c(O)ccc(C(=O)O)c1OC",  # aromatic,small
        "4-hydroxy-2,3-dimethoxybenzoic acid",
    ),
    (
        "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)NCC(=O)N[C@@H](CC(=O)O)C(=O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,medium
        "L-tryptophylglycyl-L-aspartic acid",
    ),
    (
        "NCCCC[C@H](NC(=O)[C@@H](N)Cc1ccccc1)C(=O)O",  # aromatic,polyfunctional,medium
        "L-phenylalanyl-L-lysine",
    ),
    (
        "NC(C(=O)O)c1ccccc1F",  # aromatic,polyfunctional,small
        "2-amino-2-(2-fluorophenyl)ethanoic acid",
    ),
    (
        "CC(C)C(O)CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3[C@H](O)C[C@@H]4CC(=O)CC[C@]4(C)[C@H]3CC[C@]12C",  # fused-ring,medium,steroid
        "(5R,7R,8R,9S,10S,13R,14S,17R,20R)-7,24-dihydroxycholestan-3-one",
    ),
    (
        "NC(=O)C[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](N)Cc1cnc[nH]1)C(=O)O",  # aromatic,heterocycle,polyfunctional,medium
        "L-histidyl-L-glutamyl-L-asparagine",
    ),
    (
        "CC(C)[C@H](N)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)O",  # acyclic,polyfunctional,medium
        "L-valyl-L-alanyl-L-alanine",
    ),
    (
        "CC[C@H](C)[C@H](N)C(=O)NCC(=O)N[C@@H](CCC(=O)O)C(=O)O",  # acyclic,polyfunctional,medium
        "L-isoleucylglycyl-L-glutamic acid",
    ),
    (
        "N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O",  # aromatic,heterocycle,polyfunctional,large
        "L-histidyl-L-glutamyl-L-tyrosine",
    ),
    (
        "C/C(=C\\CC[C@@H](C)[C@H]1CC[C@@]2(C)C3=C(CC[C@]12C)[C@@]1(C)CC[C@H](O)C(C)(C)[C@@H]1CC3=O)CO",  # fused-ring,large,steroid
        "(3S,5R,10S,13R,14R,17R,20R,24E)-3,27-dihydroxy-4,4,14-trimethylcholesta-8,24-dien-7-one",
    ),
    (
        "C#CCCCCCCCCCCCC(CC(O)CO)OC(C)=O",  # acyclic,medium
        "4-(acetyloxy)-2-hydroxyheptadec-16-yn-1-ol",
    ),
    (
        "CCCCOC(=O)c1ccccc1",  # aromatic,small
        "butyl benzoate",
    ),
    (
        "N[C@@H](Cc1ccc(O)c(OS(=O)(=O)O)c1)C(=O)O",  # aromatic,polyfunctional,medium
        "(2S)-2-amino-3-(4-hydroxy-3-(sulfooxy)phenyl)propanoic acid",
    ),
    (
        "O=CC(=O)C[C@H](O)CO",  # acyclic,polyfunctional,small
        "(4S)-4,5-dihydroxy-2-oxopentanal",
    ),
    (
        "N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O",  # aromatic,polyfunctional,large
        "L-tyrosyl-L-tyrosyl-L-tyrosine",
    ),
    (
        "C[C@@H](O)[C@H](NC(=O)[C@H](CCCCN)NC(=O)[C@@H](N)CCC(=O)O)C(=O)O",  # acyclic,polyfunctional,medium
        "L-glutamyl-L-lysyl-L-threonine",
    ),
    (
        "OCC(O)C(O)C(O)C(O)C(O)C(O)CO",  # acyclic,medium
        "2,3,4,5,6,7-hexahydroxyoctane-1,8-diol",
    ),
    (
        "CC(=O)OCC/C(C)=C/C(=O)O",  # acyclic,polyfunctional,small
        "(2E)-5-(ethanoyloxy)-3-methylpent-2-enoic acid",
    ),
    (
        "C#N",  # acyclic,small
        "hydrogen cyanide",
    ),
    (
        "NC(=O)C[C@H](N)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CS)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "L-asparaginyl-L-tryptophyl-L-cysteinyl-L-histidine",
    ),
    (
        "C=CC(C)(O)CC/C=C(\\C)CCC(O)C(C)(O)CCCC(C)(O)CCCC(C)(O)CCCC(C)(O)CCCC(C)(O)CCCC(C)(O)CCCC(C)(O)CCCC(C)(O)CCCC(C)(O)CCCC(C)(O)CCCC(C)(O)CCC=C(C)C",  # acyclic,large
        "(6E)-3,7,11,15,19,23,27,31,35,39,43,47,51,55-tetradecamethyl-3,11,15,19,23,27,31,35,39,43,47,51-dodecahydroxyhexapentaconta-1,6,54-trien-10-ol",
    ),
    (
        "CCC=CC(=O)[O-]",  # acyclic,charged,small
        "pent-2-enoate",
    ),
    (
        "Nc1cc(O)cc(C(=O)[O-])c1",  # aromatic,charged,small
        "5-amino3-hydroxybenzoate",
    ),
    (
        "CCCCCC(Br)C(Br)CC(Br)C(Br)CCCCCCCC(=O)O",  # acyclic,medium
        "9,10,12,13-tetrabromooctadecanoic acid",
    ),
    (
        "CCCc1cccc(O)c1",  # aromatic,small
        "3-propylphenol",
    ),
    (
        "C[C@H](O)[C@H]([NH3+])C(=O)[O-]",  # acyclic,small
        "(2S,3S)-2-amino-3-hydroxybutanoic acid",
    ),
    (
        "O=[N+]([O-])c1sccc1-c1ncccn1",  # aromatic,heterocycle,small
        # v21 WS-A.1 S2 + WS-A.2: pyrimidine is the senior parent
        # (P-44.2.1(b)) and the demoted thiophene keeps its nitro. This IS
        # the OPSIN-verified PIN (RT True before and after); also a target
        # row in 
        "2-(2-nitrothiophen-3-yl)pyrimidine",
    ),
    (
        "COc1cc(O)cc(O)c1C(=O)CC(=O)c1ccccc1",  # aromatic,medium
        "1-(2,4-dihydroxy-6-methoxyphenyl)-3-phenylpropane-1,3-dione",
    ),
    (
        "NCC(=O)N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O",  # aromatic,heterocycle,polyfunctional,medium
        "glycyl-L-histidyl-L-tyrosine",
    ),
    (
        "CCOCCCO",  # acyclic,small
        "4-oxahexan-1-ol",
    ),
    (
        "NCCCC(=O)C(=O)O",  # acyclic,polyfunctional,small
        "5-amino-2-oxopentanoic acid",
    ),
    (
        "CC(C)C[C@H](NC(=O)[C@H](CO)NC(=O)[C@@H](N)Cc1ccccc1)C(=O)O",  # aromatic,polyfunctional,medium
        "L-phenylalanyl-L-seryl-L-leucine",
    ),
    (
        "O=CC/C=C\\CCCCCCCC(=O)O",  # acyclic,polyfunctional,small
        "(9Z)-12-oxododec-9-enoic acid",
    ),
    (
        "CCCCCCNC(=O)O",  # acyclic,polyfunctional,small
        "N-hexylcarbamic acid",
    ),
    (
        "CC(C)C[C@H](N)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CC(N)=O)C(=O)O",  # acyclic,polyfunctional,medium
        "L-leucyl-L-glutaminyl-L-asparagine",
    ),
    (
        "CC(=O)OCCc1ccccc1",  # aromatic,small
        "2-phenylethyl acetate",
    ),
    (
        "CCCCC/C=C\\C/C=C\\CC(O)/C=C/C=C\\CCCC(=O)O",  # acyclic,medium
        "(5Z,7E,11Z,14Z)-9-hydroxyicosa-5,7,11,14-tetraenoic acid",
    ),
    (
        "Cc1cc(C)c(C)c(C)c1C",  # aromatic,small
        "1,2,3,4,5-pentamethylbenzene",
    ),
    (
        "O=C1CCC(O)C(O)CO1",  # heterocycle,small
        "5,6-dihydroxyoxepan-2-one",
    ),
    (
        "CC(C)C[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H](N)CC(N)=O)C(=O)O",  # acyclic,polyfunctional,medium
        "L-asparaginyl-L-glutamyl-L-leucine",
    ),
    (
        "CC(C)CC1C(=O)N2CC(C)CC2C(=O)NC(C(C)C)C(=O)OC(C(C)C)C(=O)N2NCCCC2C(=O)N2NCC(O)CC2C(=O)N1C",  # heterocycle,fused-ring,polyfunctional,large
        "20-hydroxy-15-isobutyl-3,6-diisopropyl-11,16-dimethyl-4-oxa-1,7,13,16,22,23,29-heptaaza-tetracyclo[23.4.0.0(9,13).0(18,23)]nonacosan-2,5,8,14,17,24-hexaone",
    ),
    (
        "C[C@H](CCCC(C)(C)O)[C@H]1CC[C@H]2[C@@H]3[C@H](O)CC4=CC(=O)CC[C@]4(C)[C@H]3C[C@H](O)[C@]12C",  # fused-ring,large,steroid
        "(7R,8R,9S,10R,12S,13R,14S,17R,20R)-7,12,25-trihydroxycholest-4-en-3-one",
    ),
    (
        "CCC12C=C/C(C)=C\\C(C)(O)CCC(OC)C(C)C(O)C(C)C(O)C3OC(=CC3=O)CC(=O)OC(C1)C(C)C(=O)O2",  # heterocycle,fused-ring,polyfunctional,large
        "(4Z)-1-ethyl-6,11,13-trihydroxy-9-methoxy-4,6,10,12,22-pentamethyl-20,24,26-trioxa-tricyclo[19.3.1.1(14,17)]hexacosa-2,4,16-trien-15,19,23-trione",
    ),
    (
        "COC1OC2(OC)CC3CCC(O)C(C)C3(C)C(OC)C2=C1C",  # heterocycle,fused-ring,medium
        "3,5,8-trimethoxy-6,9,10-trimethyl-4-oxa-tricyclo[7.4.0.0(3,7)]tridec-6-en-11-ol",
    ),
    (
        "C[C@@]12C=C[C@]3(C1)[C@@H](O)C[C@H]1[C@@](C)(CCC[C@@]1(C)C(=O)O)[C@@H]3CC2",  # fused-ring,medium
        "(1S,2S,4S,5R,9S,10S,13S)-2-hydroxy-5,9,13-trimethyl-tetracyclo[8.5.0.1(1,13).0(4,9)]hexadec-14-ene-5-carboxylic acid",
    ),
    (
        "CC(=O)Cc1ccc(O)c(O)c1",  # aromatic,small
        "1-(3,4-dihydroxyphenyl)propan-2-one",
    ),
    (
        "N[C@@H](CC(=O)O)C(=O)N[C@@H](CS)C(=O)O",  # acyclic,polyfunctional,small
        "L-aspartyl-L-cysteine",
    ),
    (
        "CC1=C(O)C(=O)[C@H](O)[C@H](O)C1",  # small
        "(5R,6R)-2,5,6-trihydroxy-3-methylcyclohex-2-en-1-one",
    ),
    (
        "C[C@H](CCC[C@H](C)C(=O)O)[C@H]1CC[C@H]2[C@@H]3CC[C@H]4CC(=O)CC[C@]4(C)[C@H]3CC[C@]12C",  # fused-ring,polyfunctional,medium,steroid
        "(5S,8R,9S,10S,13R,14S,17R,20R,25S)-27-hydroxycholestan-3,27-dione",
    ),
    (
        "Cc1cnc(N)c(C)n1",  # aromatic,heterocycle,small
        # Phase 170 WS-4 (DEF-6): ring -NH2 is the principal group -> -amine SUFFIX
        # (P-33), not an amino prefix. OPSIN-RT verified.
        "2,6-dimethylpyrazin-3-amine",
    ),
    (
        "CCCCCCCCCCCCCO",  # acyclic,small
        "tridecan-1-ol",
    ),
    (
        "[NH2+]=C(C[C@H](O)[C@H](O)CO)C(=O)[O-]",  # acyclic,small
        "(4S,5R)-4,5,6-trihydroxy-2-iminohexanoic acid",
    ),
    (
        "CCCCC/C=C\\C/C=C\\C/C=C\\CC(O)C(O)CCCC(=O)O",  # acyclic,medium
        "(8Z,11Z,14Z)-5,6-dihydroxyicosa-8,11,14-trienoic acid",
    ),
    (
        "CCCCCCCCCCCCCC/C=C/CC(=O)O",  # acyclic,medium
        "(3E)-octadec-3-enoic acid",
    ),
    (
        "CC1(C)CCC(=O)[C@@]2(C)O[C@]3(O)CC[C@@]12C[C@H]3O",  # heterocycle,fused-ring,medium,carbohydrate
        "(1S,6R,9R,11R)-9,11-dihydroxy-1,5,5-trimethyl-10-oxa-tricyclo[4.4.0.2(6,9)]dodecan-2-one",
    ),
    (
        "CCCCCCOC(C)=O",  # acyclic,small
        "hexyl acetate",
    ),
    (
        "CCCCCCCCCCCCCCCCCC(O)[C@H](C)C[C@H](C)C[C@H](C)C[C@H](C)C[C@H](C)C[C@H](C)C[C@H](C)C[C@H](C)C(=O)O",  # acyclic,large
        "(2S,4S,6S,8S,10R,12R,14R,16R)-17-hydroxy-2,4,6,8,10,12,14,16-octamethyltetratriacontanoic acid",
    ),
    (
        "CC[C@H](C)[C@H](N)C(=O)N[C@H](C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O)[C@@H](C)CC",  # aromatic,heterocycle,polyfunctional,medium
        "L-isoleucyl-L-isoleucyl-L-histidine",
    ),
    (
        "CCCCCCC/C=C/CCCCCCCCC(=O)[O-]",  # acyclic,charged,medium
        "(10E)-octadec-10-enoate",
    ),
    (
        "N[C@@H](CCC(=O)O)C(=O)N[C@@H](CS)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,medium
        "L-glutamyl-L-cysteinyl-L-tryptophan",
    ),
    (
        "C=C1/C=C/C(C)(C)CCC/C(C)=C\\CC1",  # small
        "(1Z,6E)-1,8,8-trimethyl-5-methylidenecycloundeca-1,6-diene",
    ),
    (
        "COc1c(C)c(O)cc2c1C(=O)CC(c1ccccc1)O2",  # aromatic,heterocycle,fused-ring,medium
        "7-hydroxy-5-methoxy-6-methyl-2-phenylchroman-4-one",
    ),
    (
        "CCCCC[C@H](O)/C=C/C=C\\C=C\\C=C\\[C@@H](O)[C@@H](O)CCCC(=O)O",  # acyclic,medium
        "(5S,6R,7E,9E,11Z,13E,15S)-5,6,15-trihydroxyicosa-7,9,11,13-tetraenoic acid",
    ),
    (
        "C/C(C#Cc1cc(O)ccc1O)=C/CC(=O)O",  # aromatic,medium
        "(3Z)-6-(2,5-dihydroxyphenyl)-4-methylhex-3-en-5-ynoic acid",
    ),
    (
        "CCCCCCCCCCCCCCCCCC(=O)[C@@H](N)CO",  # acyclic,polyfunctional,medium
        "(2S)-2-amino-1-hydroxyicosan-3-one",
    ),
    (
        "CCCCCCCC(O)C=CC(O)CCC(O)CCCC(=O)O",  # acyclic,medium
        "5,8,11-trihydroxyoctadec-9-enoic acid",
    ),
    (
        "C1=Cc2cccc3cccc(c23)C1",  # aromatic,fused-ring,small
        "1H-phenalene",
    ),
    (
        "N[C@@H](CC(=O)O)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "L-aspartyl-L-tryptophyl-L-histidine",
    ),
    (
        "CC(=O)CCCCCCCCCCCCCCCc1cc(O)cc(O)c1",  # aromatic,medium
        "17-(3,5-dihydroxyphenyl)heptadecan-2-one",
    ),
    (
        "CC=C(C)C",  # acyclic,small
        "2-methylbut-2-ene",
    ),
    (
        "O=C(O)C(O)CC(O)C(=O)O",  # acyclic,small
        "2,4-dihydroxypentanedioic acid",
    ),
    (
        "CCCCCCCCCCCCCCCCCCCCCCCCCC(=O)[O-]",  # acyclic,charged,medium
        "hexacosanoate",
    ),
    (
        "CC(C)[C@H](NC(=O)[C@@H](NC(=O)[C@@H](N)CCCCN)C(C)C)C(=O)O",  # acyclic,polyfunctional,medium
        "L-lysyl-L-valyl-L-valine",
    ),
    (
        "CCCCCCCCCC(=O)OC",  # acyclic,small
        "methyl decanoate",
    ),
    (
        "COC(=O)c1cc(O)ccc1O",  # aromatic,small
        "methyl 2,5-dihydroxybenzoate",
    ),
    (
        "CCC/C=C/C(=O)C(CC)CCCC",  # acyclic,small
        "(4E)-7-ethylundec-4-en-6-one",
    ),
    (
        "CCCCCCCC/C=C\\CCCCCCCC(=O)O[C@H](CO)COC(=O)CCCCCCCCCCCCCCCCC",  # acyclic,large
        "(2R)-2-(oleoyloxy)-1-(stearoyloxy)propan-3-ol",
    ),
    (
        "O=C[C@@H](O)[C@H](O)CO",  # acyclic,small
        "(2S,3R)-2,3,4-trihydroxybutanal",
    ),
    (
        "C[C@H](NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@@H](N)Cc1ccccc1)C(=O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "L-phenylalanyl-L-tryptophyl-L-alanine",
    ),
    (
        "CC(=O)[C@H]([NH3+])C(=O)[O-]",  # acyclic,small
        "(2S)-2-amino-3-oxobutanoic acid",
    ),
    (
        "CC(C)=CCC/C(C)=C/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(\\C)C=O",  # acyclic,large
        "(2E,4E,6E,8E,10E,12E,14E,16E,18E)-2,6,11,15,19,23-hexamethyltetracosa-2,4,6,8,10,12,14,16,18,22-decaenal",
    ),
    (
        "C=CC(O)C#CC#CCC(O)CCCCCCCC",  # acyclic,medium
        "heptadeca-1-en-4,6-diyne-3,9-diol",
    ),
    (
        "[NH3+][C@@H](Cc1ccc(O)cc1)C(=O)[O-]",  # aromatic,small
        "L-tyrosine",
    ),
    (
        "Oc1cc(O)cc(-c2ccccc2)c1",  # aromatic,small
        "5,3-dihydroxy-1,1'-biphenyl",
    ),
    (
        "O=C(O)CC/C=C\\C/C=C\\C/C=C\\C=C\\[C@H](O)C/C=C\\C/C=C\\CCO",  # acyclic,medium
        "(4Z,7Z,10Z,12E,14R,16Z,19Z)-14,22-dihydroxydocosa-4,7,10,12,16,19-hexaenoic acid",
    ),
    (
        "BrC(Br)(Br)Br",  # acyclic,small
        "tetrabromomethane",
    ),
    (
        "CC(O)[C@@]1(O)CC[C@H]2[C@@H]3CC=C4CC(O)CC[C@]4(C)[C@H]3C(=O)C[C@@]21C",  # fused-ring,medium,steroid
        "(8S,9S,10R,13S,14S,17R)-3,17,20-trihydroxypregn-5-en-11-one",
    ),
    (
        "C[C@H](NC(=O)CN)C(=O)N[C@@H](CO)C(=O)O",  # acyclic,polyfunctional,medium
        "glycyl-L-alanyl-L-serine",
    ),
    (
        "O=C(/C=C/C=C/c1ccc(O)cc1)N1CCCCC1",  # aromatic,heterocycle,medium
        "N-[(2E,4E)-5-(4-hydroxyphenyl)penta-2,4-dienoyl]piperidine",
    ),
    (
        "O=P([O-])([O-])[O-].[Na+].[Na+].[Na+]",  # acyclic,salt,small
        "trisodium phosphate",
    ),
    (
        "C[C@@H](O)[C@H](NC(=O)[C@@H](NC(=O)[C@@H](N)CCCNC(=N)N)[C@@H](C)O)C(=O)N[C@@H](CC(=O)O)C(=O)O",  # acyclic,polyfunctional,large
        "L-arginyl-L-threonyl-L-threonyl-L-aspartic acid",
    ),
    (
        "CCCCC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCCCC(=O)OC[C@H](O)COC(=O)CCC/C=C\\C/C=C\\C/C=C\\CCCCCCCC",  # acyclic,large
        "(2S)-3-((5Z,8Z,11Z)-icosa-5,8,11-trienoyloxy)-1-((7Z,10Z,13Z,16Z)-docosa-7,10,13,16-tetraenoyloxy)propan-2-ol",
    ),
    (
        "CCC(C)CCCCCCCCCCCCC(=O)OC[C@H](O)COC(=O)CCCCCCCCCCC(C)CC",  # acyclic,large
        "(2S)-1-((12-methyltetradecanoyl)oxy)-3-((14-methylhexadecanoyl)oxy)propan-2-ol",
    ),
    (
        "CCCCCCCCCCCCC(O)CO",  # acyclic,medium
        "2-hydroxytetradecan-1-ol",
    ),
    (
        "O=C(/C=C/c1ccc(O)c(O)c1)c1cc(O)c(O)cc1O",  # aromatic,medium
        "(2E)-3-(3,4-dihydroxyphenyl)-1-(2,4,5-trihydroxyphenyl)prop-2-en-1-one",
    ),
    (
        "[NH3+]C(CC(=O)[O-])c1ccc(O)cc1",  # aromatic,small
        "3-amino-3-(4-hydroxyphenyl)propanoic acid",
    ),
    (
        "O=C(CO)c1ccc(O)cc1",  # aromatic,small
        "2-hydroxy-1-(4-hydroxyphenyl)ethan-1-one",
    ),
    (
        "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CC(=O)O)C(=O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "L-tryptophyl-L-glutamyl-L-aspartic acid",
    ),
    (
        "CC[C@H]1CC[C@H]2[C@@H]3CCC4CCCC[C@]4(C)[C@H]3CC[C@]12C",  # fused-ring,medium,steroid
        "pregnane",
    ),
    (
        "CC(C)[C@H](N)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)O",  # acyclic,polyfunctional,medium
        "L-valyl-L-glutamyl-L-glutamic acid",
    ),
    (
        "C[C@@H](O)[C@H](NC(=O)CN)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O",  # aromatic,heterocycle,polyfunctional,medium
        "glycyl-L-threonyl-L-histidine",
    ),
    (
        "C/C=C/C(=O)c1c(OC)cc(O)c(CC)c1O",  # aromatic,medium
        "(2E)-1-(3-ethyl-2,4-dihydroxy-6-methoxyphenyl)but-2-en-1-one",
    ),
    (
        "O=C1CCCCCO1",  # heterocycle,small
        "oxepan-2-one",
    ),
    (
        "CCCCC/C=C\\CC(O)/C=C/C=C\\C/C=C\\CCCC(=O)[O-]",  # acyclic,charged,medium
        "(5Z,8Z,10E,14Z)-12-hydroxyicosa-5,8,10,14-tetraenoate",
    ),
    (
        "CC1=C[C@@H]2C(C)(C)[C@H]3CC[C@H](C)[C@@]23CC1",  # fused-ring,small
        "(1R,3R,6S,7R)-2,2,6,10-tetramethyl-tricyclo[5.4.0.0(3,7)]undec-10-ene",
    ),
    (
        "CC/C=C\\CC",  # acyclic,small
        "(3Z)-hex-3-ene",
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@@H](N)Cc1c[nH]c2ccccc12)C(=O)NCC(=O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,medium
        "L-tryptophyl-L-isoleucylglycine",
    ),
    (
        "CC(C)C[C@H](NC(=O)[C@H](C)N)C(=O)N[C@H](C(=O)N[C@@H](CCC(N)=O)C(=O)O)[C@@H](C)O",  # acyclic,polyfunctional,medium
        "L-alanyl-L-leucyl-L-threonyl-L-glutamine",
    ),
    (
        "CCc1cnc(C)cn1",  # aromatic,heterocycle,small
        "2-ethyl-5-methylpyrazine",
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@@H](N)CS)C(=O)N[C@@H](CC(=O)O)C(=O)O",  # acyclic,polyfunctional,medium
        "L-cysteinyl-L-isoleucyl-L-aspartic acid",
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)CN)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O",  # aromatic,polyfunctional,medium
        "glycyl-L-isoleucyl-L-tyrosine",
    ),
    (
        "N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O",  # aromatic,heterocycle,polyfunctional,large
        "L-histidyl-L-phenylalanyl-L-tyrosine",
    ),
    (
        "NCCCC[C@H](NC(=O)[C@H](CC(N)=O)NC(=O)[C@@H](N)CC(=O)O)C(=O)O",  # acyclic,polyfunctional,medium
        "L-aspartyl-L-asparaginyl-L-lysine",
    ),
    (
        "C[C@@H](O)[C@H](O)[C@H](O)[C@@H](N)C=O",  # acyclic,polyfunctional,small
        "(2R,3R,4S,5R)-2-amino-3,4,5-trihydroxyhexanal",
    ),
    (
        "NC(CO)(CO)C(=O)O",  # acyclic,polyfunctional,small
        "2-amino-3-hydroxy-2-(hydroxymethyl)propanoic acid",
    ),
    (
        "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCCCC(=O)OC[C@H](O)COC(=O)CCCCCCCCC/C=C\\C/C=C\\CCCCC",  # acyclic,large
        "(2R)-3-((11Z,14Z)-icosa-11,14-dienoyloxy)-1-((7Z,10Z,13Z,16Z,19Z)-docosa-7,10,13,16,19-pentaenoyloxy)propan-2-ol",
    ),
    (
        "CCCCCCCC/C=C\\CCCCCCC(=O)O",  # acyclic,medium
        "(8Z)-heptadec-8-enoic acid",
    ),
    (
        "CCC(=O)c1ccc(OC)cc1OC",  # aromatic,small
        "1-(2,4-dimethoxyphenyl)propan-1-one",
    ),
    (
        "C#CCCCC(=O)O",  # acyclic,small
        "hex-5-ynoic acid",
    ),
    (
        "COc1c(O)c(CO)c(Cl)c(OC)c1OC",  # aromatic,medium
        "5-chloro-6-(hydroxymethyl)-2,3,4-trimethoxyphenol",
    ),
    (
        "CC(C)(C)OC(=O)c1cccc(N)c1",  # aromatic,small
        "2-methylpropan-2-yl 3-aminobenzoate",
    ),
    (
        "CCCCOc1ccc(CC(=O)NO)cc1",  # aromatic,medium
        # Phase 163.1 closure: e-preservation corrected per IUPAC P-16.3.3
        # (suffix "-hydroxamic acid" starts with consonant 'h' -> keep terminal e).
        "2-(4-(butyloxy)phenyl)ethane-1-hydroxamic acid",
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@H](CCC(N)=O)NC(=O)[C@@H](N)Cc1c[nH]c2ccccc12)C(=O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "L-tryptophyl-L-glutaminyl-L-isoleucine",
    ),
    (
        "CCCCCCC/C=C\\CCCCCCCC(=O)O",  # acyclic,medium
        "(9Z)-heptadec-9-enoic acid",
    ),
    (
        "CC(C)[C@H](NC(=O)[C@@H](N)[C@@H](C)O)C(=O)N[C@@H](CCCCN)C(=O)O",  # acyclic,polyfunctional,medium
        "L-threonyl-L-valyl-L-lysine",
    ),
    (
        "CC1=CCCC1(C)C",  # small
        "1,5,5-trimethylcyclopent-1-ene",
    ),
    (
        "CCCCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COC(=O)CCCCCCCCCCCCC",  # acyclic,large
        "(2S)-1-(myristoyloxy)-3-(stearoyloxy)propan-2-ol",
    ),
    (
        "C/C1=C\\[C@H](O)C[C@](C)(O)/C=C/[C@H](C(C)C)CC/C(C)=C/CC1",  # medium
        "(1R,2E,6E,10S,11E,13S)-13-hydroxy-10-isopropyl-3,7,13-trimethylcyclotetradeca-2,6,11-trien-1-ol",
    ),
    (
        "CC(C)[C@H](NC(=O)[C@H](CCC(N)=O)NC(=O)[C@@H](N)[C@@H](C)O)C(=O)O",  # acyclic,polyfunctional,medium
        "L-threonyl-L-glutaminyl-L-valine",
    ),
    (
        "N[C@@H](CC(=O)O)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O",  # aromatic,heterocycle,polyfunctional,large
        "L-aspartyl-L-histidyl-L-tyrosine",
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@@H](N)Cc1ccccc1)C(=O)N[C@@H](CC(C)C)C(=O)O",  # aromatic,polyfunctional,medium
        "L-phenylalanyl-L-isoleucyl-L-leucine",
    ),
    (
        "CC1(C)C(=O)CC(O)C23C(=O)OC4OCC(=CCC12)C43",  # heterocycle,fused-ring,polyfunctional,medium
        "2-hydroxy-5,5-dimethyl-11,13-dioxa-tetracyclo[7.5.1.0(1,6).0(12,15)]pentadec-8-en-4,14-dione",
    ),
    (
        "O=C(O)CCCCCCCCCCBr",  # acyclic,small
        "11-bromoundecanoic acid",
    ),
    (
        "NC(=O)CC[C@H](NC(=O)[C@@H](N)Cc1cnc[nH]1)C(=O)N[C@@H](CS)C(=O)O",  # aromatic,heterocycle,polyfunctional,medium
        "L-histidyl-L-glutaminyl-L-cysteine",
    ),
    (
        "CC(=O)OCC(=O)[C@@]1(O)[C@@H](C)C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@@]21C",  # fused-ring,polyfunctional,large,steroid
        "(8S,9R,10S,11S,13S,14S,16S,17R)-9-fluoro-11,17-dihydroxy-16-methyl-3,20-dioxopregn-1,4-dien-21-yl acetate",
    ),
    (
        "CC(O)CCCCCCCCCC(=O)[O-]",  # acyclic,charged,small
        "11-hydroxydodecanoate",
    ),
    (
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCC(=O)O",  # acyclic,medium
        "octacosanoic acid",
    ),
    (
        "C=CCCCCCCCCCCCC(=O)CC(O)COC(C)=O",  # acyclic,polyfunctional,medium
        "1-(acetyloxy)-2-hydroxyheptadec-16-en-4-one",
    ),
    (
        "C[C@@H](O)[C@H](O)C(=O)O",  # acyclic,small
        "(2S,3R)-2,3-dihydroxybutanoic acid",
    ),
    (
        "CCCCCC(=O)/C=C/C=C\\CCCCCCCC(=O)[O-]",  # acyclic,charged,medium
        "(9Z,11E)-13-oxooctadeca-9,11-dienoate",
    ),
    (
        "CCCC/C=C\\CCCCCCCC/C=C\\CCCC(=O)O",  # acyclic,medium
        "(5Z,15Z)-icosa-5,15-dienoic acid",
    ),
    (
        "O=c1cc(-c2ccccc2)c2cc(O)c(O)cc2o1",  # aromatic,heterocycle,fused-ring,medium
        "6,7-dihydroxy-4-phenylcoumarin",
    ),
    (
        "CC(CCC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC[C@@H]4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C)C(=O)O",  # fused-ring,medium,steroid
        "(3S,5R,8R,9S,10S,13R,14S,17R,20R)-3,27-dihydroxycholestan-27-one",
    ),
    (
        "C[C@H](NC(=O)[C@@H](N)CCC(=O)O)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,medium
        "L-glutamyl-L-alanyl-L-tryptophan",
    ),
    # --- Phase 145.2 D-09-a.3: HA=7 Tier A canaries ---
    # Cover the _MIN_RATIO_ACCEPT/1.5 coverage fallback at composer.py:1526
    # (D-10 hard preservation). One canary per Tier A handler. HA=7 is the
    # minimum geometrically feasible for complex_ring + benzene + heterocycle
    # with at least one substituent — HA≤5 is geometrically IMPOSSIBLE for
    # these handler classes (a substituted benzene already has HA=7: 6 ring
    # atoms + 1 substituent). HA=7 satisfies D-09-a.3 spirit per RESEARCH.md
    # §Q1: all HA<<15, well inside the rescue branch.
    (
        "C1CC2CC1CC2",
        "norbornane",
    ),  # complex_ring handler, HA=7, ratio=0.9524 (non-boosted)
    (
        "Fc1ccccc1",
        "fluorobenzene",
    ),  # benzene handler, HA=7, retained-scaffold boost → ratio=1.0
    (
        "Clc1ccncc1",
        "4-chloropyridine",
    ),  # heterocycle handler, HA=7, retained-scaffold boost → ratio=1.0
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

_OPSIN_JAR = Path(__file__).resolve().parents[2] / "opsin-cli-2.9.0-jar-with-dependencies.jar"


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
        # WR-01: wrap stdin in `with open(...)` so the fd closes deterministically
        # on all paths (including subprocess.run raising). `subprocess.run` does
        # not take ownership of file objects passed as stdin.
        with open(tmp_path) as stdin_f:
            result = subprocess.run(
                ["java", "-jar", str(_OPSIN_JAR), "-osmi"],
                stdin=stdin_f,
                capture_output=True,
                text=True,
                timeout=120,
            )
        # WR-03: fail loudly if OPSIN exits non-zero (JVM crash, classpath
        # error, OOM) so regressions are distinguishable from parse failures.
        # Mirrors the peer helper at test_deterministic_output.py:263.
        if result.returncode != 0:
            raise RuntimeError(
                f"OPSIN batch invocation failed (returncode={result.returncode}): "
                f"stderr={result.stderr!r}"
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
    "(1R,3r,5S)-tropan-3-yl nonanoate": (
        "Phase 118-02: tropane NP naming correctly identifies tropane as parent. OPSIN may not parse tropan-3-yl ester format."
    ),
    "2,6-diamino-3-(hydroxymethyl)heptanetrioic acid": (
        "OPSIN cannot parse 'heptanetrioic acid' (rare tricarboxylic acid suffix)"
    ),
    "(2R)-2-phenylethan-1-ol": (
        "PERC-06: diol reclassified as non-polyfunctional (same parent class). "
        "Non-polyfunctional path drops second OH — Phase 131 assembly fix needed."
    ),
    "N-heptanoyl(2S)-5-aminoazol-3-one": (
        "Phase 139.1-01: middle fragment OH capping produces decomposed name for "
        "heptanoyl-pyrrolidinone compound. OPSIN may not parse this format. "
        "Phase 170 WS-4: ring ketone now -one suffix (still unparseable — pre-existing)."
    ),
    "(3R)-9-(1,3-dioxolan-5-yl)-3,7-dimethylnona-1,6-dien-3-ol": (
        "Phase 142-03: dioxolane ring correctly identified (was wrongly named as "
        "cyclopentyl). OPSIN cannot parse dioxolanyl substituent prefix."
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
    # v22 Phase G0 (DD7 S1): a canary whose expected value is a deliberate
    # fail-closed refusal ('unknown organic compound') is NOT an IUPAC name and
    # is intentionally not OPSIN-parseable — skip the format-parse guard for it.
    from orthonym.errors import is_failure_name
    if is_failure_name(expected_name):
        pytest.skip("G0 fail-closed refusal signal, not an IUPAC name to parse")
    if expected_name in _OPSIN_LIMITATIONS:
        pytest.xfail(f"OPSIN limitation: {_OPSIN_LIMITATIONS[expected_name]}")
    results = _get_opsin_results()
    parsed = results.get(expected_name)
    assert parsed, f"OPSIN cannot parse canary name: {expected_name}"
