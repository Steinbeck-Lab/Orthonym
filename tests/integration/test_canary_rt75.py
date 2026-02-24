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
"""

import pytest

from orthonym import name_compound


# ---------------------------------------------------------------------------
# 158 golden canary compounds: (SMILES, expected_name)
# Original 75 from Phase 44 + 10 from Phase 49 + 3 from Phase 50
# + 44 from Phase 62 (Small Molecule Accuracy Sprint)
# + 9 from Phase 63 (Stereochemistry Accuracy)
# + 2 from Phase 66 (Medium Molecule Completeness)
# + 15 from Phase 67 (v7.0 Final Benchmark)
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
        "(2R,3R,4S,5R)-3,4,5,6-tetrahydroxytetrahydropyran-2-carboxylic acid",
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
        "isoindoline-1,3-dione",
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
    (
        "c1ccc(OC)cc1",
        "methoxybenzene",
    ),
    (
        "c1ccc(OCC)cc1",
        "ethoxybenzene",
    ),
    # Ring ether guard (DEC-01: tetrahydropyran not decomposed)
    (
        "C1CCOCC1",
        "tetrahydropyran",
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
        "3,4,5,6-tetrahydroxy-2-methyltetrahydropyran",
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
        "2-[(11z,14z)-icosa-11,14-dienoyloxy]-1,3-bis(linoleoyloxy)propane",
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
]

# Build test IDs from expected names (first 40 chars, sanitized for pytest)
_CANARY_IDS = [
    name[:40].replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    for _, name in CANARY_COMPOUNDS
]


@pytest.mark.parametrize("smiles,expected_name", CANARY_COMPOUNDS, ids=_CANARY_IDS)
def test_canary_rt158(smiles, expected_name):
    """Golden canary test: verify round-trip-matching compound still names correctly."""
    result = name_compound(smiles)
    assert result == expected_name, (
        f"CANARY REGRESSION: {smiles}\n"
        f"  Expected: {expected_name}\n"
        f"  Got: {result}"
    )
