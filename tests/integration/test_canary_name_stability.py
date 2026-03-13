"""
Name-stability canary: freezes exact generated names for ALL OPSIN-unparseable
benchmark compounds. These cannot be validated by OPSIN round-trip, so name-string
stability is the only regression protection.

If v11.0 intentionally changes a name, update the expected value.

Source: Phase 095 benchmark (500 ChEBI compounds, seed=123)
Tier: 3 of 3 (Tier 1 = RT-exact in test_canary_rt75.py, Tier 2 = connectivity)
Total: 120 compounds where OPSIN cannot parse the generated name
"""

import pytest

from orthonym import name_compound


# ---------------------------------------------------------------------------
# 120 name-stability canary compounds: (SMILES, expected_name)
# All OPSIN-unparseable benchmark compounds with frozen names
# Phase 095 v10.0 canary expansion
# ---------------------------------------------------------------------------

NAME_STABILITY_CANARY = [
    (
        "C=C(CC[C@@H](C)[C@H]1CC[C@H]2C3=CC[C@H]4[C@H](C)C(=O)CC[C@]4(C)C3=C[C@@H](O)[C@]12C)C(C)C",
        "(4S,5S,10S,11R,13R,14S,17R,20R)-11-hydroxy-4-methylergosta-7,9,24-trien-3-one",
    ),
    (
        "CC(=O)N[C@@H]1[C@@H](O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2NC(C)=O)[C@@H](O)[C@@H](CO)O[C@@H]1O",
        "(beta-D-2-(acetylamino)-2-deoxy-glucopyranosyloxy)ethanamide",
    ),
    (
        "[Cl-].[Cl-].[Cl-].[Yb+3]",
        "ytterbium compound (not supported) trichloride",
    ),
    (
        "c1ccc2cc3c(cc2c1)-c1cc2ccccc2cc1-c1cc2ccccc2cc1-c1cc2ccccc2cc1-3",
        "cycloane",
    ),
    (
        "NC(C(=O)O)C(CC[C@H](N)C(=O)O)C(=O)O",
        "(6S)-2,6-diamino-3-(hydroxymethyl)heptanetrioic acid",
    ),
    (
        "*C(=O)N[C@@H](CO[C@@H]1O[C@H](CO)[C@@H](O[C@@H]2O[C@H](CO)[C@H](O)[C@H](O[C@@H]3O[C@H](CO)[C@@H](O)[C@H](O[C@@H]4O[C@H](CO)[C@H](O)[C@H](O[C@H]5O[C@H](CO)[C@H](O)[C@H](O)[C@H]5NC(C)=O)[C@H]4O[C@@H]4O[C@@H](C)[C@@H](O)[C@@H](O)[C@@H]4O)[C@H]3NC(C)=O)[C@H]2O)[C@H](O)[C@H]1O)[C@H](O)/C=C/CCCCCCCCCCCCC",
        "(ethanediamide)(2S,3R,4E)-1-hydroxy-3-hydroxy-2-(methanoylamino)octadec-4-enamide",
    ),
    (
        "C[C@@H](O)[C@H](NC(=O)[C@@H]1CCCN1)C(=O)N[C@@H](Cc1ccccc1)C(=O)O",
        "N-(2S,3R)-3-hydroxy-2-(pentanoylamino)butanoyl(2S)-2-amino-3-phenylpropanoic acid",
    ),
    (
        "C=C1NC(=O)[C@H]([C@@H](C)[C@]2(O)C(=O)N(C)c3ccccc32)NC1=O",
        "(2R)-2-hydroxy-N-methylindolin-1-one",
    ),
    (
        "*N[C@@H](CC(=O)NC1O[C@H](CO)[C@@H](O[C@@H]2O[C@H](CO)[C@@H](O[C@@H]3O[C@H](CO[C@H]4O[C@H](CO[C@H]5O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]5O)[C@@H](O)[C@H](O[C@H]5O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]5O[C@H]5O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]5O)[C@@H]4O)[C@@H](O)[C@H](O[C@H]4O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]4O[C@H]4O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]4O[C@H]4O[C@H](CO)[C@@H](O)[C@H](O[C@H]5O[C@H](CO)[C@@H](O)[C@H](O)[C@H]5O)[C@@H]4O)[C@@H]3O)[C@H](O)[C@H]2NC(C)=O)[C@H](O)[C@H]1NC(C)=O)C(*)=O",
        "(3S)-butanetriamide",
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@@H](NC(=O)[C@H](C)NC(=O)[C@H](CCCCNC(=O)CCl)NC(=O)[C@H](CC(=O)O)NC(C)=O)[C@@H](C)O)C(=O)NCC(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@H](C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)O)C(C)C",
        "N-(3S)-chloro-3-(ethanoylamino)-4-(hexylamino)-hydroxybutanedioyl-L-phenylalanyl-L-valyl-L-glutamyl-L-glutaminyl-L-glutamyl-L-glutamic acid",
    ),
    (
        r"CC(=O)N[C@@H](CC(C)C)C(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N/C=C\c1c[nH]c2ccccc12",
        "N-(2S)-2-(hexanoylamino)-3-phenylpropanoyl-3-(2-aminoethyl)-1H-indole",
    ),
    (
        "CC(=O)OC[C@H]1O[C@@H](n2ccc(=O)[nH]c2=O)[C@H](OC(C)=O)[C@@H]1OC(C)=O",
        "1,2-bis(acetyloxy)oxolane",
    ),
    (
        "C/C1=C/C[C@H](O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)/C(C)=C/[C@H]2OC(=O)[C@H](C)[C@@H]2CC1",
        "(beta-D-glucopyranosyloxy)(1S,2E,4S,6Z,10S)-4-hydroxy-3,7-dimethylcyclodeca-2,6-dien-1-carboxylate",
    ),
    (
        "CC[C@@H](C(=O)[O-])C(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)([O-])OP(=O)([O-])OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)([O-])[O-]",
        "N-7-(10-carboxydecyl)adenineyl(2S)-amino-2-(aminomethyl)-2-(pentylsulfanyl)butanoate",
    ),
    (
        "CC1=C[C@]2(C)C[C@@H](C)CC[C@@H]2[C@H](C(=O)[C@@H]2C(=O)N3CC[C@@H]4C(=O)O[C@H]2[C@@]43O)[C@@H]1C",
        "(3R,4R,5R,6S)-3-(2-methylbutyl)-1,3,6-trimethyl-5-oxocyclohex-1-enecarboxylate",
    ),
    (
        r"CO[C@H]1C=C/C=C\C=C/C[C@H](OC(=O)[C@@H](C)NC(=O)C2=CCCCC2)[C@H](C)[C@@H](O)/C(C)=C\CCc2cc(O)cc(c2O)NC(=O)C1",
        "1-(cyclohexanecarbonyloxy)-3,9-dihydroxy-14-hydroxymethyl-2,4-dimethyl-12-oxo-8-propyl1-azacyclohenicosene",
    ),
    (
        "CC[C@@H]1OC(=O)C=C[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)C2O)CC[C@@H](C)C(=O)C=C[C@@H]2O[C@@H]2[C@]1(O)CO[C@@H]1OC(C)[C@H](O)[C@H](OC)[C@@H]1OC",
        "(2S,3S,4S,5S,9R,12S,13S)-2-ethyl-3-hydroxy-9,13-dimethyl-3,12-dioxanyl-8,16-dioxo-1-oxacyclohexadecene",
    ),
    (
        "C[C@H]1CN2[C@@H](O)[C@]34C[C@@]5(C(=O)Nc6c5ccc5c6C(=O)CC(C)(C)O5)C(C)(C)[C@@H]3C[C@@]2(C1)C(=O)N4C",
        "(5R,15R,19S,20S,23S,25S)-19-hydroxy-12,12,15,16,22,22-hexamethyl-13-oxa-7,16,17-triaza-heptacyclo[7.4.0.0(4,8).0(5,21).0(15,18).0(17,19)]hexacosan-6,10,14-trione",
    ),
    (
        "CCCCCCCCCCCCCCCC(=O)N1CCCC1",
        "1-pyrrolidinylN,N-dibutylhexadecanamide",
    ),
    (
        "[O]=[Sb]([O-])([O-])[OH]",
        "antimony compound (not supported)",
    ),
    (
        r"CCCCCC/C=C\CC(=O)N[C@@H](CO)[C@@H](O)CC(=O)N[C@H](C(=O)N[C@H](/C=C/C(=O)NCC(=O)c1c[nH]c2ccccc12)CO)C(C)C",
        "(2E,4R)-1-(3-acetyl-1H-indolyl)-5-hydroxy-4-(pentanoylamino)pent-2-enetetraamide",
    ),
    (
        "CCCCCCCCCCCCCCCCCCCCCC[C@H](O)C(=O)N[C@@H](COP(=O)(O)O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O)[C@H](O)CCCCCCCCCCCCCCC",
        "N-(2S)-2-hydroxytetracosanoyl(1S,2R,3S,4S,5R,6R)-aminocyclohexane-2,3,4,5,6-pentaol",
    ),
    (
        "CSCC[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@@H](N)Cc1ccc(O)cc1)C(=O)O",
        "(3S)-4-(butylamino)-3-(nonanoylamino)butanedioic acid",
    ),
    (
        "Oc1ccc2c(c1)O[C@H](c1ccc(O)c(O)c1)[C@@H](O)[C@@H]2O",
        "(2R,3S,4R)-3,4,7-trihydroxychromane",
    ),
    (
        "C#CCCC[C@@H]1OC(=O)[C@H](C)NC(=O)[C@H](Cc2ccc(OC)cc2)N(C)C(=O)[C@@H]2CCCN2C(=O)[C@H](Cc2ccccc2)N(C)C(=O)[C@H](C(C)C)NC(=O)C1(C)C",
        "(3S,6S,9S,12S,15S,19S)-15-isopropyl-N,N-dimethyl-3,18,18-trimethyl-2,5,8,11,14,17-hexaoxo-19-(pent-4-yn-1-yl)-6,12-diphenyl-1-oxa-4,7,10,13,16-pentaazacyclononadecane",
    ),
    (
        r"CC/C=C\C/C=C\C/C=C\CCCCCCCC(=O)OCC(COP(=O)(O)OCCNC)OC(=O)CCCCCCCCC/C=C\C/C=C\CCCCC",
        "2-((11Z,14Z)-icosa-11,14-dienoyloxy)-1-(linolenoyloxy)propanamine",
    ),
    (
        r"C=C1C(=O)O[C@@H]2C[C@@H](C)/C=C\C(=O)[C@@](C)(O)C[C@@H](OC(=O)CC(C)C)[C@@H]12",
        "(2R,3S)-4-methyl-5-oxooxolane",
    ),
    (
        "CC(C)=CCC/C(C)=C/CC[C@]1(C)Cc2c(c(O)cc3c2CN([C@H]2CCCNC2=O)C3=O)C[C@@H]1O",
        "(11S,12R)-12-(6E)-2,6-dimethylnona-2,6-dienyl-8,11-dihydroxy-12-methyl-4-pentyl-4-aza-tricyclo[7.4.0.0(2,6)]tridecan-5-one",
    ),
    (
        "CCCCCCCCCCCCCCCC(O)C(=O)N[C@@H](CO)[C@H](O)/C=C/CCCCCCCCCC(C)C",
        "1-(heptadecylamino)-hydroxy-2-hydroxyheptadecanamide",
    ),
    (
        "CN(C(=O)c1ccc2c(c1)OC(F)(F)O2)c1cccc(C(=O)Nc2c(Br)cc(C(F)(C(F)(F)F)C(F)(F)F)cc2OC(F)F)c1F",
        "N-methylbenzamideyl-2-amino-1-bromo-5-isopropyl-3-methoxybenzene",
    ),
    (
        "CC[C@H]1O[C@@H]2O[C@H](/C=C/C=C/C3C(c4oc(=O)cc(OC)c4C)C(/C=C/C=C/[C@H]4O[C@H]5O[C@H](CC)[C@](C)(O)[C@@]5(C)[C@H]4O)C3c3oc(=O)cc(OC)c3C)[C@H](O)[C@]2(C)[C@@]1(C)O",
        "(2S,3R,4R,5R)-5-ethyl-4-hydroxy-3,4-dimethyloxolane",
    ),
    (
        "CC(C)[C@@H]1CC[C@H](C)CCC[C@H](C)CC1",
        "(1S,4s,7R)-4-isopropyl-1,7-dimethylcyclodecane",
    ),
    (
        "C/C1=C/C[C@@H](/C(C)=C/c2csc(C)n2)OC(=O)C[C@H](O)C(C)(C)C(=O)[C@H](C)[C@@H](O)/C(C)=C/CC1",
        "(4S,7R,8R,9E,13Z,16S)-4,8-dihydroxy-5,5,7,9,13-pentamethyl-16-(2-methyl-4-(prop-1-en-1-yl)thiazolyl)-6-oxooxacyclohexadecan-2-one",
    ),
    (
        "N#CC(SC[C@H](NC(=O)CC[C@H]([NH3+])C(=O)[O-])C(=O)NCC(=O)[O-])c1c[nH]c2ccccc12",
        "(2R)-3-(decylsulfanyl)-1-(ethylamino)-2-(pentanoylamino)propanediamidate",
    ),
    (
        "C[C@H]1/C=C/C=C/C=C/C=C/C=C/[C@@H](O)[C@H](C(=O)O)[C@H](O)C[C@H](O)CCC[C@H](O)C[C@H](O)C[C@H](O)[C@@H](C)C(=O)O[C@@H]1C",
        "(3R,4S,6S,8S,12R,14R,15R,16R,17E,19E,21E,23E,25E,27S,28R)-15-formyl-4,6,8,12,14,16-hexahydroxy-3,27,28-trimethyloxacyclooctacosan-2-one",
    ),
    (
        "*B(*)*",
        "compound with wildcard atoms (not supported)",
    ),
    (
        "CC(=O)N[C@@H]1[C@@H](O[C@@H]2O[C@H](COS(=O)(=O)[O-])[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H](CO[C@@H]2O[C@H](CO)[C@H](O)[C@H](O[C@@H]3O[C@H](COS(=O)(=O)[O-])[C@@H](O)[C@H](O)[C@H]3O)[C@H]2NC(C)=O)O[C@H]1O.[Na+].[Na+]",
        "disodium (2R,3R,4R,5R,6R)-5-ethyl-3,6-dihydroxy-2,4-dioxanyloxane",
    ),
    (
        "CO[C@@H]1[C@H](OC(=O)CCC(=O)O)CC[C@](O)(CCl)[C@H]1[C@@]1(C)O[C@@H]1CC=C(C)C",
        # P-44.3 fix: no-PG path now selects chain over small oxirane ring in substituent
        "(1R,2S,3S,4R)-2-(3-methyl-1-oxiranylbut-2-enyl)-1-(chloromethyl)-1-hydroxy-3-methoxycyclohexyl butanedioate",
    ),
    (
        r"C=CCO/N=C(\C(=O)N[C@H]1CN2CC(S(C)(=O)=O)=C(C(=O)O)N2C1=O)c1csc(N)n1",
        "N-2-amino-2-cyclopentylethanoyl-4-methyl-1,2-diazole-5-carboxylic acid",
    ),
    (
        "C=CC(=O)Nc1ccc2ncnc(Nc3ccc(-c4ccccc4)cc3)c2c1",
        "4-(1,1'-biphenylamino)quinazoline",
    ),
    (
        "C/C=C/C1=CC(=O)[C@@]2(C(=O)c3c(OC)cc(OC)cc3C(=O)OC)O[C@H]2O1",
        "(2R,3S)-4-oxo-3-phenyl-6-(prop-1-en-1-yl)-3,4-dihydro-2H-pyran",
    ),
    (
        "C/C=C/C[C@@H]1NC(=O)[C@H](CC(C)C)N2C(=O)[C@H](C[C@H](C)[C@@H]2O)N(C)C(=O)[C@H](C)NC(=O)[C@H](Cc2ccc(O)c([N+](=O)[O-])c2)NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@H](Cc2cn(C(C)(C)[C@H]3CO3)c3ccccc23)NC1=O",
        "(3S,6S,9S,12S,15S,18S,21S)-21-(but-2-en-1-yl)-3,15-diisobutyl-N,N-dimethyl-9-methyl-2,5,8,11,14,17,20-heptaoxo-12-phenyl-18-pyrrolyl-1,4,7,10,13,16,19-heptaazacyclohenicosane",
    ),
    (
        r"CCCCCC/C=C\CC(=O)N[C@@H](CO)C(=O)N[C@H](C(=O)N[C@@H](CO)[C@@H](O)CC(=O)N[C@@H](CO)C(=O)N[C@H](C(=O)N[C@@H]1/C=C/C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H](Cc2ccc(O)cc2)C(=O)OC1)C(C)C)C(C)C",
        "N-(2S)-3-hydroxyhydroxy-2-(pentanoylamino)propanoyl(3S,6S,9E,11R)-3-benzyl-11-(4-carbamoylbutyl)-6-isopropyl-4-methyl-5,8-dioxooxacyclododecan-2-one",
    ),
    (
        "C/C=C(/CC[C@@H](C)[C@H]1CC[C@H]2C3=CC[C@H]4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C)C(C)C",
        "(3S,5S,9R,10S,13R,14R,17R,20R,24Z)-stigmasta-7,24-dien-3-ol",
    ),
    (
        "C[C@H]1C[C@H](O)[C@@H]2[C@H]1[C@@H]1[C@H](CC[C@]2(C)O)[C@@]1(C)CO",
        "(1S,2S,5S,6R,7R,8R,9S,11S)-2,6,6,9-tetramethyl-tricyclo[6.3.0.0(5,7)]undecan-2,11-diol",
    ),
    (
        "C[C@H]1CCC/C=C/[C@@H]2CC[C@H](O)[C@H]2[C@H](O)/C=C/C(=O)O1",
        "(3E,5R,6S,7S,8E,13S)-5-hydroxy-13-methyl-2-oxo-1-oxacyclotridecene",
    ),
    (
        "C=C1CC[C@@H](C/C=C2/CC[C@]3(OC2)O[C@@]2(O)CC[C@]3(C)OC2(C)C)C(C)(C)[C@H]1[C@@H](O)C=C1CCOC1=O",
        "(1S,2S,5S,9Z)-2,7,7-trimethyl-6,8,12-trioxa-tricyclo[3.1.0.2(2,5)]tridecan-5-ol",
    ),
    (
        "C=C1/C=C/C(=O)N(C)CC(=O)O[C@@H](CCCCCCCCCCCCCC)[C@H](C)C(=O)[C@](C)(O)C(=O)NCC(=O)N1",
        "(6E,14S,16S,17S)-17-tetradecyl-14-hydroxy-4,8,14,16-tetramethyl-5,10,13,15-tetraoxooxacycloheptadecan-2-one",
    ),
    (
        "CCCCC(C)/C=C(C)/C=C/C(=O)NC1=C[C@@](O)(/C=C/C=C/C=C/C(=O)NC2=C(O)CCC2=O)[C@H](O)CC1=O",
        "N-(2E,4E)-4,6-dimethyldeca-2,4-dienoyl(2E,4E,6E)-7-(5-amino-2-hydroxycyclohexyl)-1-(cyclopentylamino)-7-hydroxyhepta-2,4,6-trienamide",
    ),
    (
        "CC(C)(O)[C@@H]1CC[C@@](C)([C@H]2CC[C@]3(C)[C@@H]2CC[C@@H]2[C@@]4(C)CCC(=O)C(C)(C)[C@@H]4CC[C@]23C)O1",
        "(5R,8R,9R,10R,13R,14R,17S)-4,4,8,10,14-pentamethylgonan-3-one",
    ),
    (
        "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@H](COC1O[C@H](CO)[C@H](O)[C@H](OS(=O)(=O)[O-])[C@H]1O)NC(=O)C(O)CCCCCCCCCCCCCCCCCCCC",
        "N-2-hydroxydocosanoyltetracosanolate",
    ),
    (
        "C[C@H](NC(=O)[C@H](C)NC(=O)[C@@H]1CCCN1)C(=O)O",
        "N-(2S)-2-(pentanoylamino)propanoyl-2-aminopropanoic acid",
    ),
    (
        "CS[C@@]1(CO)C(=O)N2[C@H]3N(c4ccc5oc6cc(=O)c(N)c(C(=O)O)c-6nc5c4C(=O)O)c4ccccc4[C@@]3(c3c[nH]c4ccccc34)[C@H](O)[C@]2(SC)C(=O)N1C",
        "(1R,4S,7S,8S,9R)-16-tetradecyl-8-hydroxy-4,4,5,7-tetramethyl-9-octyl-2,5,16-triaza-tetracyclo[7.7.0.0(2,7).0(10,15)]hexadecan-3,6-dione",
    ),
    (
        "CC[C@@H](C)[C@H]1C(=O)N(C)[C@@H](Cc2ccc(OC)c(Br)c2)C(=O)N[C@@H]([C@@H](C)CC)C(=O)O[C@H](C)[C@H](NC(=O)[C@H](NC(=O)[C@@H](COS(=O)(=O)O)OC)C(C)C)C(=O)N[C@@H](CCCN=C(N)N)C(=O)N[C@H]2CC[C@@H](O)N1C2=O",
        "N-(2R)-methoxy-3-methyl-2-(propanoylamino)-sulfobutanoyl(1S,4S,7S,8R,11S,14S,17S,21R)-7-amino-4,11,17-tributyl-21-hydroxy-8,15-dimethyl-14-octyl-9-oxa-2,5,12,15,18-pentaaza-bicyclo[16.3.1]docosane",
    ),
    (
        "O.O.O.O.O.O.O.O=S(=O)([O-])[O-].[Ni+2]",
        "nickel compound (not supported)",
    ),
    (
        r"C/C1=C/C=C\C=C/C=C\C=C/C[C@@H]2C[C@H](O)C[C@](O)(C[C@H](O)C[C@@H](O)/C=C\C[C@@H](O)C[C@@H](O)C[C@H](O)C[C@H](O)[C@H](C)[C@H](C(C)C)OC1=O)O2",
        "(3Z,5Z,7Z,9Z,11Z,14R,16S,18R,20R,21Z,24R,26R,28S,30S,31S,32S)-16,18,20,24,26,28,30-heptahydroxy-32-isopropyl-3,31-dimethyl-2-oxo-1,15-dioxacyclodotriacontene",
    ),
    (
        "NCCCCCO[C@@H]1O[C@H](CO[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]2O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]2O)[C@@H](O)[C@H](O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]2O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]2O)[C@@H]1O",
        "(alpha-D-mannopyranosyloxy)(2R,3S,4S,5R,6R)-3,5-dihydroxy-4,6-dioxanyl-2-pentyloxane",
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@@H](NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H](N)CC(=O)O)C(C)C)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](Cc1c[nH]cn1)C(=O)O",
        "N-L-aspartyl-L-tyrosyl-L-valyl-L-arginyl-L-isoleucineyl(2S)-3-imidazolyl-2-(nonanoylamino)propanoic acid",
    ),
    (
        "CCOc1cc(C(=O)O)ccc1NC(=O)c1ccc(NC(=O)c2ccc(NC(=O)[C@@H](NC(=O)c3ccc(NC(=O)c4ccc([N+](=O)[O-])cc4)cc3)[C@@H](OC)C(N)=O)cc2)c(OC(C)C)c1O",
        # Phase 103-03: chain tiebreaker refinements detect aminomethyl substituent
        "N-(2S,3R)-2-(benzoylamino)-3-methoxypropanoyl-4-(16-carbamoylhexadecyl)-3-ethoxybenzoic acid",
    ),
    (
        "O=C1N[C@H]2NC(=O)N[C@H]2N1",
        "(4s,5s)-N,N'-dipropylurea",
    ),
    (
        r"C/C=C/C(=O)O[C@H]1/C=C\C(=O)[C@@H](O)CCC(=O)O[C@@H]1C",
        "(5S,7Z,9S,10R)-9-(3-carboxypropyl)-5-hydroxy-10-methyl-6-oxooxecan-2-one",
    ),
    (
        "CC(C)=CCC[C@](C)(O[C@@H]1O[C@H](CO[C@@H]2OC[C@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@H](O)[C@H]1O)[C@H]1CC[C@]2(C)[C@@H]1[C@H](O)C[C@@H]1[C@@]3(C)CC[C@H](O[C@@H]4O[C@H](CO)[C@@H](O)[C@H](O)[C@H]4O[C@@H]4O[C@H](CO)[C@@H](O)[C@H](O)[C@H]4O)C(C)(C)[C@@H]3CC[C@]12C",
        "(3S,5R,8R,9R,10R,12R,13R,14R,17S)-4,4,8,10,14-pentamethylgonan-12-ol",
    ),
    (
        "COC1=CC=C2[C@H]3Cc4ccc(OC)c5c4[C@@]2(C[C@@H](C2=C[C@@]4(O)[C@H]6Cc7ccc(O)c8c7[C@@]4(CCN6C)[C@@H](O8)C2=O)N3C)[C@H]1O5",
        "(5R,9R,13S,16S)-4,5-epoxy-3,6-dimethoxy-17-methylmorphina-6,8-diene",
    ),
    (
        "CC(=O)N[C@@H]1[C@@H](O)[C@H](O[C@@H]2O[C@H](CO)[C@@H](O[C@@H]3O[C@H](CO)[C@@H](O)[C@H](O[C@H]4O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]4O)[C@@H]3O)[C@H](O)[C@H]2NC(C)=O)[C@@H](CO)O[C@H]1O",
        "(alpha-D-mannopyranosyloxy)ethanediamide",
    ),
    (
        "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@H](CO[C@@H]1O[C@H](CO)[C@@H](O[C@@H]2O[C@H](CO)[C@H](O)[C@H](O)[C@H]2O)[C@H](O)C1O)NC(=O)CCCCCCCCCCC",
        "(beta-D-galactopyranosyloxy)(2R,4S,5S,6R)-2-triacontyl-3,4,5-trihydroxy-6-methyloxane",
    ),
    (
        "CC(C)[C@H]1CC[C@@H](CO)c2c(O)cc(C(=O)O)cc21",
        "(1R,4R)-1,2,3,4-tetrahydronaphthalene",
    ),
    (
        r"CC1=C[C@H]2OC3C[C@H]4OC(=O)/C=C\C=C/C(C(C)O)OCC/C(C)=C\C(=O)OC[C@@]2(CC1)C4(C)[C@]31CO1",
        "(1R,4Z,6Z,12Z,17R,22R,27S)-8-ethyl-12,20,26-trimethyl-2,9,15,23-tetraoxa-pentacyclo[15.8.1.1(24,26)]nonacosa-4,6,12,20-tetraen-3,14-dione",
    ),
    (
        "COCc1ccc(O)c(NC(C)=O)c1",
        "1-anilinoethanamide",
    ),
    (
        r"C/C(=C\CC[C@@H](C)[C@H]1CC[C@@]2(C)C3=CC[C@H]4C(C)(C)[C@@H](OC=O)CC[C@]4(C)C3=CC[C@]12C)CO",
        "(3S,5R,10S,13R,14R,17R,20R,24E)-27-hydroxy-4,4,14-trimethylcholest-7,9,24-trien-3-yl formate",
    ),
    (
        "CC(C)[C@@H]1NC(=O)c2csc(n2)[C@H](C(C)C)NC(=O)c2csc(n2)[C@H](C(C)C)NC(=O)[C@H]2N=C1O[C@@H]2C",
        "(4S,7R,8S,11S,17S)-4,11,17-triisopropyl-7-methyl-6-oxa-19,23-dithia-3,10,13,16,21,24-hexaaza-tetracyclo[16.2.1.2(12,14).1(5,8)]tetracos-5-en-2,9,15-trione",
    ),
    (
        r"CC/C=C\C/C=C\C/C=C\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCC)COP(=O)([O-])OC[C@H]([NH3+])C(=O)[O-]",
        "unknown organic compound",
    ),
    (
        "C[C@@H]([NH3+])P(=O)([O-])[O-]",
        "unknown organic compound",
    ),
    (
        "[I][Hg-2]([I])([I])[I]",
        "mercury compound (not supported)",
    ),
    (
        "CC(=O)OC[C@H]1O[C@@H](O[C@]2(COC(C)=O)O[C@H](COC(=O)/C=C/c3ccccc3)[C@@H](O)[C@@H]2OC(=O)/C=C/c2ccccc2)[C@H](OC(C)=O)[C@@H](O)[C@@H]1OC(C)=O",
        "1,1-bis(acetyloxy)-1-(benzoyloxy)-2-hydroxyoxolane",
    ),
    (
        "C[C@@H]1O[C@@H](O[C@@H]2C[C@H](c3ccc4c(c3O)C(=O)C3=C(C4=O)[C@@]4(O)C(=O)C[C@](C)(O)C[C@@]4(O)C=C3)O[C@H](C)[C@H]2O)CC[C@@H]1O[C@H]1C[C@@H](O)[C@H](O)[C@@H](C)O1",
        "(11S,14R,16R)-5-octadecyl-4,11,14,16-tetrahydroxy-14-methyl-tetracyclo[8.8.0.0(3,8).0(11,16)]octadeca-1,17-dien-2,9,12-trione",
    ),
    (
        "CSCC[C@H](NC(=O)[C@@H](N)Cc1ccccc1)C(=O)N1CCC[C@@H]1C(=O)O",
        "N-(2S)-2-amino-3-phenylpropanoyl(2R)-N-pentylpyrrolidine-2-carboxylic acid",
    ),
    (
        "C=C(CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3C(=O)C[C@H]4[C@](C)(C(=O)O)[C@@H](O)CC[C@]4(C)C3=C[C@@H](OC(C)=O)[C@]12C)C(C)C",
        "(3S,4S,5R,8S,10S,11R,13R,14S,17R,20R)-3-hydroxy-4-methyl-7-oxoergost-9,24-dien-11-yl acetate",
    ),
    (
        "CCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCC",
        "2-(N,N,dimethylmethan-1-aminiumyl)ethan-1-olium phosphonooxydecanephosphonic acid",
    ),
    (
        "CCN(CC)c1ccc2c(C=CC=CC=C3N(CCCCCC(=O)O)c4ccc(S(=O)(=O)[O-])cc4C3(C)C)cc(C(C)(C)C)[o+]c2c1",
        "unknown organic compound",
    ),
    (
        r"CCCCC/C=C\C/C=C\C/C=C\CC1OC1CCCC(=O)NCCO",
        "1-(ethylamino)-4-oxiranylbutanamide",
    ),
    (
        "CCCCCCCCC/C=C/[C@@H](O)[C@H](COP(=O)(O)OCCN)NC(=O)CCCCCCCCCCCCCCCCC",
        "amino-1-(tetradecylamino)-hydroxyoctadecanamide",
    ),
    (
        "CCCCCCC/C=C/C=C/[C@@H](O)[C@H](COP(=O)(O)OCCN)NC(=O)CCCCCCCCCCCCCCCCCCCC",
        "amino-1-(tetradecylamino)-hydroxyhenicosanamide",
    ),
    (
        "CC(C)C[C@H](N)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](CCCN=C(N)N)C(=O)O",
        "(2S)-amino-2-(butanoylamino)-5-guanidino-5-(methylamino)pentanedioic acid",
    ),
    (
        "C=C1CC23C=CC(=O)C(C)(CCCC(C)C(=O)NC(CCC(N)=O)C(=O)O)C2CC1CC3O",
        # Phase 103-01: chain exclusion + polycyclic parent changes name
        "N-glutyl-5-cyclododecanyl-2-methylpentanamide",
    ),
    (
        "Oc1cc(O)c2c(c1)O[C@H](c1ccc(O)c(O)c1)[C@H](O)[C@H]2c1c(O)cc(O)c2c1O[C@H](c1cc(O)c(O)c(O)c1)[C@H](O)C2",
        "(2R,3R,4R)-3,5,7-trihydroxychromane",
    ),
    (
        "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@H](COC1O[C@H](CO)[C@H](O)[C@H](OS(=O)(=O)[O-])[C@H]1O)NC(=O)CCCCCCCCCCCCCCCCCCCCC",
        "(hexanolate)-1-(octadecylamino)-dihydroxydocosanamide",
    ),
    (
        "CCCC(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)NC(CCC(=O)N[C@@H]1C(=O)N[C@@H](CCC(=O)O)C(=O)NC2CC[C@@H](O)N(C2=O)[C@@H](C(C)C)C(=O)N(C)[C@@H](Cc2ccccc2)C(=O)N[C@@H]([C@@H](C)CC)C(=O)O[C@@H]1C)C(=O)O",
        "N-2-(nonanoylamino)pentanedioyl-3-cyclodocosylpropanoic acid",
    ),
    (
        "CC(=O)OC1CC2OC3C=C(C)C(=O)[C@@H](O)[C@]3(C)[C@]1(C)[C@]21CO1",
        "(6S,7S,8S,12S)-5,8-dihydroxy-6,7,10-trimethyl-2-oxa-tetracyclo[5.4.0.1(3,6)]tetradec-10-en-9-one acetate",
    ),
    (
        "COCC1=C2[C@@H]3CC(C)(C)C[C@@H]3C[C@@]2(O)CC1=O",
        "(2R,6R,8R)-11-ethyl-8-hydroxy-4,4-dimethyl-tricyclo[6.3.0.0(2,6)]undec-1-en-10-one",
    ),
    (
        r"C[C@@H]1C[C@@H]2O[C@@H]3[C@@H](C)[C@H](O)[C@@H]4O[C@]5(C[C@H](O)CO5)[C@@H](C)[C@H](C)[C@H]4O[C@H]3C[C@H]2O[C@H]2C[C@H]3O[C@H]4C/C=C\C[C@H]5O[C@H]6C=C[C@H]7O[C@H]8[C@H](O)[C@H]9OCC=CC[C@@H]9O[C@@H]8C[C@@H]7O[C@@H]6C/C=C\[C@@H]5O[C@@H]4C[C@@H](O)[C@]3(C)O[C@@H]2C1",
        "(1S,3Z,6R,8S,11R,13S,14R,15R,21S,23R,25S,27R,31S,33R,35R,36S,38R,40R,42S,44R,45S,46S,47S,49R,50S,51S,52R,54S,56R,58S,60R,65S)-36,40,45,50,51-pentamethyl-7,12,16,22,26,32,37,43,48,53,57,61,63-tridecaoxa-tridecacyclo[31.28.0.0(6,31).0(8,27).0(36,60).0(38,58).0(42,56).0(44,54).0(47,52).0(49,63).0(49,64).0(62,65)]pentahexaconta-3,9,18,29-tetraen-14,35,46,65-tetraol",
    ),
    (
        r"CCCCC/C=C\C/C=C\CCCCCCCC(=O)O[C@H](COCCCCCCCCCCCCCCCCCC)COC(=O)CCCCCCCCCCCCCCCCCCCCCCC",
        "(2R)-1-(tetracosanoyloxy)-3-octadecyloxy-2-(linoleoyloxy)propane-1,2-dioate",
    ),
    (
        "C=C(CC[C@@H](C(=O)O)[C@H]1[C@H](O)[C@H](O)[C@@]2(C)C3=CC[C@H]4C(C)(C)C(=O)CC[C@]4(C)C3=CC[C@]12C)C(C)C",
        "(5R,10S,13R,14R,15R,16S,17R,20R)-15,16,21-trihydroxy-4,4,14-trimethylergosta-7,9,24-trien-3,21-dione",
    ),
    (
        "CCCCCCCCCCCCCCCCCCCCCCCC(O)C(O)C(=O)N[C@@H](COP(=O)([O-])O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1OC1O[C@H](COP(=O)([O-])O[C@@H]2[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@H](O)[C@@H]1O)[C@H](O)C(O)CCCCCCCCCCCCCC",
        "N-2,3-dihydroxyhexacosanoyl(1R,2R,3S,4R,5R,6R)-aminocyclohexane-3,4,5,6-tetraol",
    ),
    (
        "CC(=O)N[C@@H]1[C@@H](O[C@@H]2O[C@@H](C)[C@@H](O)[C@@H](O)[C@@H]2O)[C@H](O[C@@H]2O[C@H](CO)[C@H](O)[C@H](O[C@]3(C(=O)O)C[C@H](O)[C@@H](NC(=O)CO)[C@H]([C@H](O)[C@H](O)CO)O3)[C@H]2O)[C@@H](CO)O[C@H]1O",
        "(alpha-L-fucopyranosyloxy)(2R,3S,4R,5R,6R)-5-ethyl-4,6-dihydroxy-2-methyl-3-oxanyloxane",
    ),
    (
        "COc1cccc2c1[C@@H](OC)O[C@H]2c1c(O)ccc2c1C(=O)CC(C)(O)C2",
        "(methoxybenzene)methanol",
    ),
    (
        "NC(N)=NCCC[C@H](NC(=O)[C@H](CC(=O)O)NC(=O)[C@@H](N)Cc1cnc[nH]1)C(=O)O",
        "(2S)-2-(butanoylamino)-5-(methylamino)pentanedioic acid",
    ),
    (
        "CC1(C)CO[C@@](C)(CCCc2ccc(Cl)cc2)N1C(=O)n1ccnc1",
        "(2S)-N-imidazolyl-2,4,4-trimethyl-2-phenyloxazolane",
    ),
    (
        "[N-2][NH-]",
        "unknown organic compound",
    ),
    (
        "COc1cc(O)c2c(c1)C(=O)C1=C(C2=O)[C@@H](O)C[C@@](C)(O)C1",
        "(4S,6S)-4,6,14-trihydroxy-12-methoxy-6-methyl-tricyclo[8.4.0.0(3,8)]tetradec-3-en-2,9-dione",
    ),
    (
        "C[C@@H](O)[C@H](NC(=O)[C@@H]1CCCN1C(=O)[C@@H](N)Cc1c[nH]c2ccccc12)C(=O)O",
        "N-3-(2-carboxyethyl)-1H-indoleyl(2S,3R)-3-hydroxy-2-(pentanoylamino)butanoic acid",
    ),
    (
        r"C[C@@H]1CC(=O)O[C@@H](C)[C@H](O)/C=C\C(=O)O[C@@H](C)C/C=C\C(=O)O1",
        "(4R,7Z,10S,13Z,15R,16S)-15-hydroxy-4,10,16-trimethyl-6,12-dioxooxacyclohexadecan-2-one",
    ),
    (
        "O=C([O-])C(=O)C[C@@H](O)[C@H](O)[C@H](O)COP(=O)([O-])[O-]",
        "(4R,5S,6R)-4,5,6-trihydroxy-2-oxophosphono-7-phosphonooxyheptanoate",
    ),
    (
        "CSCC[C@H](N)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",
        "N-(2S)-amino-2-(butanoylamino)-methylsulfanylbutanedioyl-3-(2-carboxyethyl)-1H-indole",
    ),
    (
        "C/C(=C/C(=O)CC(C)C(=O)O)[C@H]1CC(=O)[C@@]2(C)C3=C(C(=O)C[C@]12C)[C@@]1(C)CC[C@H](O)C(C)(C)C1CC3=O",
        "(3S,10S,13R,14R,17R,20Z)-3,27-dihydroxy-4,4,14-trimethylcholesta-8,20-dien-7,11,15,23,27-pentaone",
    ),
    (
        r"CC/C=C\C/C=C\C/C=C\C/C=C\CCCCCCC(=O)OC[C@H](COC(=O)CCCCCCCCCCCCCCCCCCCCC)OC(=O)CCCCCCC/C=C\C/C=C\C/C=C\CC",
        "3-(arachidonoyloxy)-1-(docosanoyloxy)-2-(linolenoyloxy)propane",
    ),
    (
        "C#CCN1CC(=O)N(COC(=O)[C@@H]2[C@@H](C=C(C)C)C2(C)C)C1=O",
        "heptyl (2R,3R)-cyclopropanecarboxylate",
    ),
    (
        "CC(C)=C[C@H](O)C1=CC(=O)[C@@H](O)[C@H](O)[C@H]1O",
        "(4S,5R,6S)-3-(1-hydroxy-3-methylbut-2-enyl)-3,4,5,6-tetrahydroxycyclohex-2-en-1-one",
    ),
    (
        "CC1CCCC[C@H](O)[C@@H]2C[C@@H](O1)C1=C(O2)[C@H](O)CCC1=O",
        "(1S,2S,9R,14R)-2,14-dihydroxy-7-methyl-8,16-dioxa-tricyclo[7.7.1.0(10,15)]heptadec-10-en-11-one",
    ),
    (
        "COc1c2c(c(O)c3c4c(c(C)cc13)[C@@H]1O[C@@]3(C(OC)OC)O[C@@H]1[C@@](O[C@H]1CC(O)[C@@](O)(C(C)=O)C(C)O1)(O4)[C@@]3(O)Cn1cnc3nc(N)[nH]c(=O)c31)C(=O)C(O)CC2O[C@H]1CC(C)(O)[C@H](OC(C)=O)C(C)O1",
        "(31-phenylhentriacontyl acetate)-1-hydroxy-1-oxanylethan-1-one",
    ),
    (
        "CC(C)C(=O)OC[C@H]1O[C@H](O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@@H](O)[C@@H]1O",
        # Updated: branched acid naming fix correctly identifies 2-methylpropanoyl
        # (principal chain = 3C) instead of butanoyl (4C total carbon count).
        # IUPAC P-65.6.3.2.2: acyloxy prefix uses principal chain for acid stem.
        "(alpha-D-glucopyranosyloxy)((2-methylpropanoyl)oxy)-2-methylpropanetriol",
    ),
    (
        "[F][Au]([F])([F])([F])[F]",
        "gold compound (not supported)",
    ),
    (
        "CC(C)[C@@H]1NC(=O)[C@H](NC(=O)NC(Cc2c[nH]c3ccccc23)C(=O)O)CCCCNC(=O)[C@H](Cc2ccccc2)NC(=O)[C@H](C)N(C)C(=O)[C@H](CCc2ccc(O)cc2)NC1=O",
        "(3R,10S,13S,16S,19S)-3-(3-(3-carboxypropyl)-1H-indolyl)-10-benzyl-16-(1-ethyl-4-hydroxybenzenyl)-19-isopropyl-13-methyl-N-methyl-9,12,15,18-tetraoxoazacyclononadecan-2-one",
    ),
    (
        "CCC(C)C1=C2C(=O)OC[C@H]2[C@@H](C)[C@H](C)O1",
        "(2S,3R,4S)-6-(sec-butyl)-2,3-dimethyl-3,4-dihydro-2H-pyran",
    ),
    (
        "CC(C)[C@H](CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C)OS(=O)(=O)[O-]",
        "(3S,8S,9S,10R,13R,14S,17R,20R,24S)-cholest-5-en-3-ol",
    ),
    (
        "C[C@H]1C=C[C@H]2C[C@@H](O)CC[C@H]2[C@@H]1c1cc(N)c(C=O)c(=O)o1",
        "(3S,4R,5R,6R)-6-(2-hydroxy(2R)-butyl)amino-6-hydroxy-3-methylcyclohex-1-enecarbaldehyde",
    ),
    (
        "CCCCCCCCCCCCCCCCC[C@@H](O)[C@H](CO)NC(=O)C(O)CCCCCCCCCCCCCCCC",
        "(2S,3R)-2-(octadecanoylamino)-1-hydroxy-3-hydroxyicosanamide",
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@@H]1CCCN1)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O",
        "N-(2S,3S)-3-methyl-2-(pentanoylamino)pentanoyl(2S)-2-amino-3-imidazolylpropanoic acid",
    ),
    (
        "CC(=O)Oc1ccc(-c2c(O)c(O)c(-c3ccc(O)c(O)c3)c(OC(C)=O)c2OC(C)=O)cc1",
        "1,1,1-tris(acetyloxy)benzene",
    ),
    (
        "CC(NC(CCCN=C(N)N)C(=O)O)C(=O)O",
        "2-amino-5-guanidino-5-(methylamino)-2-(propylamino)pentanedioic acid",
    ),
    (
        "C=C(C(=O)OC)N1C(=O)C[C@@H](C)C1=O",
        "(octanoyloxy)-2-pyrrolidinylprop-2-enimide",
    ),
]

# Build test IDs from first 40 chars of SMILES (sanitized for pytest)
_CANARY_IDS = [
    smiles[:40].replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    for smiles, _ in NAME_STABILITY_CANARY
]


@pytest.mark.parametrize("smiles,expected_name", NAME_STABILITY_CANARY, ids=_CANARY_IDS)
def test_canary_name_stability(smiles, expected_name):
    """Name-stability canary test: verify name consistency for OPSIN-unparseable compounds (120 compounds).

    These compounds generate names that OPSIN cannot parse. Since round-trip
    validation is impossible, freezing the exact name string is the only way to
    detect regressions. Any name change should be investigated before merging.
    """
    result = name_compound(smiles)
    assert result == expected_name, (
        f"CANARY REGRESSION (name-stability tier): {smiles}\n"
        f"  Expected: {expected_name}\n"
        f"  Got: {result}"
    )


# ---------------------------------------------------------------------------
# P-44.3 canary compounds: 25 compounds frozen BEFORE Phase 104 parent
# selection changes. These track the impact of fixing the ring-vs-chain
# parent selection metric (IUPAC P-44.3(a)).
#
# Categories covered:
#   - Fused ring + chain (8): ring system total atoms > chain but individual
#     ring < chain, causing wrong parent selection
#   - NP backbone (5): steroids/alkaloids with long side chains
#   - Aromatic + FG on chain (5): single 6-membered ring where PG is on chain
#   - Hydrocarbon no-PG (3): ring + chain with no principal FG
#   - Multi-ring + branch (4): complex multi-ring systems with branching
#
# Phase 104 Plan 01: canary expansion. After Plan 02 changes, some of these
# names will change as the parent selection becomes IUPAC-compliant. Any name
# change must be verified as an intentional improvement, not a regression.
# ---------------------------------------------------------------------------

P44_3_CANARY = [
    # --- Fused ring + chain (8 compounds) ---
    # Row 12: anthranilic acid derivative; fused ring system should be parent
    # per P-44.3 but individual ring size (6) < chain causes chain selection
    (
        "CC(=O)[C@@H](C)Nc1ccccc1C(=O)O",
        "2-(2-oxo(3R)-3-aminobutyl)benzoic acid",
    ),
    # Row 45: imidazopyridine + tolyl; fused system (9 atoms) vs chain
    (
        "Cc1ccc(-c2nc3ccc(C)cn3c2CC(=O)N(C)C)cc1",
        "2-benzyl-3-(3-carbamoylpropyl)-6-methylimidazo[1,2-a]pyridine",
    ),
    # Row 47: pentacyclic anthraquinone; large fused system vs chain
    (
        "COc1c(Cl)c(C)cc2cc(O)c3c(c12)C(=O)c1cc2c(OC)cc(OC)c(O)c2c(O)c1C3=O",
        "4,6,22-trihydroxy-7,9,16-trimethoxy-18-methyl-pentacyclo[12.8.0.0(3,12).0(5,10).0(15,20)]docosan-2,13-dione",
    ),
    # Row 51: lactone with two phenyl groups; fused system vs chain
    (
        "O=C(O)C1=C(c2ccccc2)C(=Cc2ccccc2)C(=O)O1",
        "3-benzyl-5-formyl-4-phenyloxolan-2-one",
    ),
    # Row 54: tricyclic with methoxy; fused ring system vs chain
    (
        "COC(=O)[C@@]1(O)C(=O)C=C2c3cc(OC)cc(O)c3C(=O)CC21",
        "(5S)-5-ethyl-5,10-dihydroxy-12-methoxy-tricyclo[7.4.0.0(2,6)]tridec-2-en-4,8-dione",
    ),
    # Row 57: pentacyclic with methylenedioxy; fused system dominant
    (
        "COc1cc2c(cc1OC)C1C(CO2)Oc2c(ccc3occc23)C1",
        "18,19-dimethoxy-7,12,15-trioxa-pentacyclo[11.8.0.0(3,11).0(6,10).0(16,21)]henicosane",
    ),
    # Row 60: anthraquinone + acetic acid chain
    (
        "COc1cccc2c1C(=O)c1ccc3c(c1C2=O)C(=O)C[C@@H](CC(=O)O)C3",
        "2-cyclooctadecanylethanoic acid",
    ),
    # Row 82: tetracyclic stilbenoid; large fused system vs ethyl chain
    (
        "CC[C@@H]1Cc2cc(O)ccc2C2=C1c1ccc(O)cc1C[C@H]2O",
        "(9R,18R)-9-ethyl-tetracyclo[8.8.0.0(2,7).0(11,16)]octadec-1-en-5,14,18-triol",
    ),
    # --- NP backbone compounds (5 compounds) ---
    # Row 46: strychnine-type alkaloid; NP backbone should be ring parent
    (
        "COC(=O)[C@@H]1CC23CCCN4CC[C@@]5(c6ccccc6NC5=O)[C@H]4C[C@H]2[C@@H]1C[C@@H]3OC(C)=O",
        "(8R,9R,11S,12S,14S,15R)-14-ethoxy-15-ethyl-5,23-diaza-hexacyclo[9.3.0.2(1,12).0(5,9).0(17,22)]tetracosan-24-one",
    ),
    # Row 58: aspidosperma alkaloid skeleton; NP ring system vs chain
    (
        "CC[C@H]1[C@@H]2CC3[C@@H]4N(C)c5ccccc5[C@@]43CC[C@@H]2C[C@H]1C(=O)OC",
        "(3R,4S,5R,7R,10S,18S)-4,5-diethyl-17-methyl-17-aza-pentacyclo[8.8.0.0(3,7).0(10,18).0(11,16)]octadecane",
    ),
    # Row 59: long-chain amide with indole; NP ring vs C24 chain
    (
        "CCCCCCCCCCCCCCCCCCCCCCCC(=O)NCCc1c[nH]c2ccccc12",
        "N-tetracosanoyl-3-(2-aminoethyl)-1H-indole",
    ),
    # Row 69: chromanone NP derivative; tricyclic ring vs short chain
    (
        "C[C@H]1C[C@@H](O)[C@H]2C(=O)c3c(O)cccc3O[C@@H]2C1",
        "(1R,10R,12S,14R)-4,14-dihydroxy-12-methyl-9-oxa-tricyclo[8.4.0.0(3,8)]tetradecan-2-one",
    ),
    # Row 94: venlafaxine-type; cyclohexyl ring + methoxyphenyl
    (
        "COc1ccc(C(CN(C)C)C2(O)CCCCC2)cc1.[Cl-].[H+]",
        "1-(hydroxydecyl)-4-methoxybenzene hydrochloride",
    ),
    # --- Aromatic + FG on chain (5 compounds) ---
    # Row 4: cyclohexanone + chain with acid and amide
    (
        r"CC1C/C(=C\CC(CC(N)=O)CC(=O)O)C(=O)C(C)C1",
        "3-(2-aminoethyl)-5-(3,5-dimethylcyclohexyl)pentanoic acid",
    ),
    # Row 48: diaminotoluene + phenylpropyl chain
    (
        "Cc1ccc(NCCCc2ccccc2)c(N)c1",
        "2-amino-4-methyl-1-(N-propylbenzenylamino)benzene",
    ),
    # Row 42: cyanoacetamide with benzene; PG (acid) is on chain
    (
        "N#CC(NC(=O)CC(=O)O)c1ccccc1",
        "3-anilinopropanoic acid",
    ),
    # Row 74: quinoline thioether + ester chain
    (
        "COC(=O)CSc1cc(C)nc2ccccc12",
        "2-methylquinoline",
    ),
    # Row 83: malate ester derivative; chain vs ring parent
    (
        "C[C@H](CC(=O)[O-])OC(=O)C[C@@H](C)O",
        "(3R)-3-(butanoyloxy)-hydroxybutanoate",
    ),
    # --- Hydrocarbon no-PG (3 compounds) ---
    # Synthetic: cyclopropane + decane; P-44.3 fix: chain (10) > ring (3)
    (
        "C1CC1CCCCCCCCCC",
        "1-cyclopropyldecane",
    ),
    # Synthetic: cyclohexane + butyl; ring (6) > chain (4), ring correct
    (
        "C1CCCCC1CCCC",
        "butylcyclohexane",
    ),
    # Synthetic: cyclobutane + ethyl; ring (4) > chain (2), ring correct
    (
        "C1CCC1CC",
        "ethylcyclobutane",
    ),
    # --- Multi-ring + branch (4 compounds) ---
    # Row 22: oxazole with pyridyl and sulfonamide substituents
    (
        "CC(C)(C)c1nc(-c2cccc(NS(=O)(=O)c3c(F)cccc3F)c2)c(-c2ccncc2)o1",
        "2-(tert-butyl)-4-phenyl-5-pyridyloxazole",
    ),
    # Row 28: gallic acid derivative with multiple ester branches
    (
        "O=C(O)c1cc(O)c(O)c(OC(=O)c2cc(O)c(O)c(OC(=O)c3cc(O)c(O)c(O)c3)c2)c1",
        "3-tetradecoxy-4,5-dihydroxybenzoic acid",
    ),
    # Row 36: biphenyl with prenyl and methoxy groups
    (
        "COc1cc(-c2ccc(O)c(CC=C(C)C)c2)c(OC)c(O)c1O",
        "4',5,4-trihydroxy-3,2-dimethoxy-3'-2-methylbut-2-enyl-1,1'-biphenyl",
    ),
    # Row 38: isoflavone glycoside; flavone ring vs sugar chain
    (
        "O=c1c(-c2ccc(OC3OC(CO)C(O)C(O)C3O)cc2)coc2cc(O)cc(O)c12",
        "(glucopyranosyloxy)-4-oxo-5-phenyl-2H-pyran",
    ),
]

# Build test IDs for P-44.3 canary
_P44_3_IDS = [
    smiles[:40].replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    for smiles, _ in P44_3_CANARY
]


@pytest.mark.parametrize("smiles,expected_name", P44_3_CANARY, ids=_P44_3_IDS)
def test_canary_p44_3(smiles, expected_name):
    """P-44.3 canary test: freeze current names BEFORE parent selection fix (25 compounds).

    These compounds are known P-44.3 parent selection failures from FAILURE-TRACES.md.
    They are frozen at their CURRENT (pre-fix) names so that Phase 104 Plan 02 changes
    can be tracked. After the parent selection fix, some names will intentionally change
    as the ring-vs-chain metric becomes IUPAC-compliant (total ring atoms instead of
    individual ring size).

    Categories:
    - Fused ring + chain (8): individual ring < chain but total system >= chain
    - NP backbone (5): steroids/alkaloids with long side chains
    - Aromatic + FG on chain (5): single ring where PG is on chain
    - Hydrocarbon no-PG (3): ring + chain with no principal FG
    - Multi-ring + branch (4): complex multi-ring systems
    """
    result = name_compound(smiles)
    assert result == expected_name, (
        f"CANARY REGRESSION (P-44.3 tier): {smiles}\n"
        f"  Expected: {expected_name}\n"
        f"  Got: {result}"
    )
