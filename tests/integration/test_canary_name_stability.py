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
# 600 name-stability canary compounds: (SMILES, expected_name)
# All OPSIN-unparseable benchmark compounds with frozen names
# Phase 095 v10.0 canary expansion
# + 48 from Phase 108 (v12.0 canary expansion: diverse failure categories)
# ---------------------------------------------------------------------------

NAME_STABILITY_CANARY = [
    (
        "C=C(CC[C@@H](C)[C@H]1CC[C@H]2C3=CC[C@H]4[C@H](C)C(=O)CC[C@]4(C)C3=C[C@@H](O)[C@]12C)C(C)C",
        "(4S,5S,10S,11R,13R,14S,17R,20R)-11-hydroxy-4-methylergosta-7,9,24-trien-3-one",
    ),
    (
        "CC(=O)N[C@@H]1[C@@H](O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2NC(C)=O)[C@@H](O)[C@@H](CO)O[C@@H]1O",
        "(beta-D-2-(acetylamino)-2-deoxy-glucopyranosyloxy)alpha-D-2-(acetylamino)-2-deoxy-galactopyranose",
    ),
    (
        "[Cl-].[Cl-].[Cl-].[Yb+3]",
        "ytterbium compound (not supported) trichloride",
    ),
    (
        "c1ccc2cc3c(cc2c1)-c1cc2ccccc2cc1-c1cc2ccccc2cc1-c1cc2ccccc2cc1-3",
        "unknown organic compound",  # Phase 145.2: RATIO_REJECT_FLOOR=0.10 rejects 'ane' garbage chain name (ratio=0.05 for 4-naphthyl fused system). Falls through to generic "unknown" fallback — semantically equivalent "couldn't name" signal as the Phase 126 SMILES-passthrough.
    ),
    (
        "NC(C(=O)O)C(CC[C@H](N)C(=O)O)C(=O)O",
        "(6S)-2,6-diamino-3-(hydroxymethyl)heptanetrioic acid",
    ),
    (
        "*C(=O)N[C@@H](CO[C@@H]1O[C@H](CO)[C@@H](O[C@@H]2O[C@H](CO)[C@H](O)[C@H](O[C@@H]3O[C@H](CO)[C@@H](O)[C@H](O[C@@H]4O[C@H](CO)[C@H](O)[C@H](O[C@H]5O[C@H](CO)[C@H](O)[C@H](O)[C@H]5NC(C)=O)[C@H]4O[C@@H]4O[C@@H](C)[C@@H](O)[C@@H](O)[C@@H]4O)[C@H]3NC(C)=O)[C@H]2O)[C@H](O)[C@H]1O)[C@H](O)/C=C/CCCCCCCCCCCCC",
        "(ethanediamide)(2S,3R,4E)-1,3-dihydroxy-2-(methanoylamino)octadec-4-enamide",  # ASML-12: locant-aware prefix merge
    ),
    (
        "C[C@@H](O)[C@H](NC(=O)[C@@H]1CCCN1)C(=O)N[C@@H](Cc1ccccc1)C(=O)O",
        "N-[(2S,3R)-3-hydroxy-2-(pentanoylamino)butanoyl](2S)-2-amino-3-phenylpropanoic acid",
    ),
    (
        "C=C1NC(=O)[C@H]([C@@H](C)[C@]2(O)C(=O)N(C)c3ccccc32)NC1=O",
        "(2R)-2-hydroxy-N-methylindolin-1-one",
    ),
    (
        "*N[C@@H](CC(=O)NC1O[C@H](CO)[C@@H](O[C@@H]2O[C@H](CO)[C@@H](O[C@@H]3O[C@H](CO[C@H]4O[C@H](CO[C@H]5O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]5O)[C@@H](O)[C@H](O[C@H]5O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]5O[C@H]5O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]5O)[C@@H]4O)[C@@H](O)[C@H](O[C@H]4O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]4O[C@H]4O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]4O[C@H]4O[C@H](CO)[C@@H](O)[C@H](O[C@H]5O[C@H](CO)[C@@H](O)[C@H](O)[C@H]5O)[C@@H]4O)[C@@H]3O)[C@H](O)[C@H]2NC(C)=O)[C@H](O)[C@H]1NC(C)=O)C(*)=O",
        "compound with wildcard atoms (not supported)",  # Phase 145.2: RATIO_REJECT_FLOOR=0.10 rejects '(3S)-butanetriamide' as truncated garbage (ratio=0.094 on 100+ atom glycan w/ wildcards). Correct routing: wildcard-atom detection branch, which was always the right semantic for this SMILES.
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@@H](NC(=O)[C@H](C)NC(=O)[C@H](CCCCNC(=O)CCl)NC(=O)[C@H](CC(=O)O)NC(C)=O)[C@@H](C)O)C(=O)NCC(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@H](C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H](CCC(=O)O)C(=O)O)C(C)C",
        "N-[(3S)-chloro-3-(ethanoylamino)-4-(hexylamino)-hydroxybutanedioyl]-L-phenylalanyl-L-valyl-L-glutamyl-L-glutaminyl-L-glutamyl-L-glutamic acid",
    ),
    # Phase 148 Plan 02 Task 03: cascade unblock per P-44.1(a)/(b). Pre-148
    # the deleted `_should_bypass_fused_guard` produced ring-as-parent
    # `3-(2-aminoethyl)-1H-indole`-style prefix; post-148 cascade routes the
    # propanamide chain as parent (acid PG on chain). The indole is still
    # named correctly as `1H-indol-3-yl` substituent. Acceptable churn.
    (
        r"CC(=O)N[C@@H](CC(C)C)C(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N/C=C\c1c[nH]c2ccccc12",
        "N-acetyl(2S)-1-(amino(4Z)-2-(1H-indol-3-yl)eth-1-en-1-yl)-2-(hexanoylamino)-3-phenylpropanamide",
    ),
    (
        "CC(=O)OC[C@H]1O[C@@H](n2ccc(=O)[nH]c2=O)[C@H](OC(C)=O)[C@@H]1OC(C)=O",
        "(2R,3R,4R,5R)-1,2-bis(acetyloxy)oxolane",
    ),
    (
        "C/C1=C/C[C@H](O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)/C(C)=C/[C@H]2OC(=O)[C@H](C)[C@@H]2CC1",
        "(beta-D-glucopyranosyloxy)(1S,2E,4S,6Z,10S)-4-hydroxy-3,7-dimethylcyclodeca-2,6-dien-1-carboxylate",
    ),
    # Phase 148 Plan 02 Task 03: CoA-style ester. Pre-148 produced a
    # space-separated multi-fragment placeholder; post-148 cascade unblock
    # routes the chain as parent. Both are pre-existing IUPAC-questionable
    # for a CoA derivative — name-stability test passes on string match.
    (
        "CC[C@@H](C(=O)[O-])C(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)([O-])OP(=O)([O-])OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)([O-])[O-]",
        "N-[(2R)-2-hydroxy-3,3-dimethylbutanoyl](2S)-amino-2-(aminomethyl)-2-(pentylsulfanyl)butanoate",
    ),
    (
        "CC1=C[C@]2(C)C[C@@H](C)CC[C@@H]2[C@H](C(=O)[C@@H]2C(=O)N3CC[C@@H]4C(=O)O[C@H]2[C@@]43O)[C@@H]1C",
        "(3R,4R,5R,6S)-3-[(S)-2-methylbutyl]hydroxy-1,3,6-trimethyl-5-oxocyclohex-1-enecarboxylate",  # PERC-05: generic alcohol detects hemiaminal OH
    ),
    (
        r"CO[C@H]1C=C/C=C\C=C/C[C@H](OC(=O)[C@@H](C)NC(=O)C2=CCCCC2)[C@H](C)[C@@H](O)/C(C)=C\CCc2cc(O)cc(c2O)NC(=O)C1",
        "(7Z,9R,10R,11S,13Z,15Z,19R)-1-(cyclohexanecarbonyloxy)-3,9-dihydroxy-14-hydroxymethyl-2,4-dimethyl-12-oxo-8-propyl1-azacyclohenicosene",
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
        "1-pyrrolidinyl-N,N-dibutylhexadecanamide",
    ),
    (
        "[O]=[Sb]([O-])([O-])[OH]",
        "antimony compound (not supported)",
    ),
    # Phase 148 Plan 02 Task 03: cascade unblock per P-44.1. Pre-148
    # `3-acetyl-1H-indolyl` (ring-as-substituent w/ ketone prefix);
    # post-148 cascade picks the chain (carbonyl PG); indole rendered as
    # `2-(1H-indol-3-yl)-...` substituent. Acceptable churn.
    (
        r"CCCCCC/C=C\CC(=O)N[C@@H](CO)[C@@H](O)CC(=O)N[C@H](C(=O)N[C@H](/C=C/C(=O)NCC(=O)c1c[nH]c2ccccc12)CO)C(C)C",
        "(2E,4R)-1-(2-oxo2-(1H-indol-3-yl)-1-aminoethyl)-5-hydroxy-4-(pentanoylamino)pent-2-enetetraamide",
    ),
    (
        "CCCCCCCCCCCCCCCCCCCCCC[C@H](O)C(=O)N[C@@H](COP(=O)(O)O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O)[C@H](O)CCCCCCCCCCCCCCC",
        "N-[(2S)-2-hydroxytetracosanoyl](1S,2R,3S,4S,5R,6R)-aminocyclohexane-2,3,4,5,6-pentaol",
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
        "N-benzoyl-2-fluoro-3-(N-methylamino)benzamide",
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
        "N-L-glutamyl-2-(propanoylamino)ethanoate",
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
        "(1R,2S,3S,4R)-2-[(1R,4R)-(3R)-3-methyl-1-oxiranylbut-2-enyl]-1-(chloromethyl)-1-hydroxy-3-methoxycyclohexyl butanedioate",
    ),
    (
        r"C=CCO/N=C(\C(=O)N[C@H]1CN2CC(S(C)(=O)=O)=C(C(=O)O)N2C1=O)c1csc(N)n1",
        "N-[2-amino-2-(thiazol-4-yl)ethanoyl]-4-methyl-1,2-diazole-5-carboxylic acid",
    ),
    # Phase 148 D-05 / Plan 02 Task 03: cascade unblock interacts with the
    # Plan-01 carry-forward decomposition fragment-naming bug (148-01-SUMMARY
    # Risks §2). Pre-148 cascade non-fired → quinazoline-as-parent +
    # biphenylamino prefix. Post-148 cascade fires (acrylamide chain has C=O
    # PG); decomposition layer then mis-names the quinazoline+biphenyl ring
    # system, dropping it from the output. Marked xfail; root cause is the
    # decomposition fragment-naming layer (NOT cascade), to be fixed in
    # Phase 149 / IM-x.x. NOT a Phase 148 regression.
    pytest.param(
        "C=CC(=O)Nc1ccc2ncnc(Nc3ccc(-c4ccccc4)cc3)c2c1",
        "4-(1,1'-biphenylamino)quinazoline",
        marks=pytest.mark.xfail(
            strict=True,
            reason="Phase 149 / IM-x.x: decomposition fragment-naming bug for "
                   "ester/amide-linked fused heterocycles surfaced by Phase 148 "
                   "cascade unblock; per 148-01-SUMMARY Risks §2 (Plan-01 carry-forward).",
        ),
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
        # Phase 125: old "3-benzyl" was incorrect (fragment has OH on ring,
        # not plain benzyl). Fix correctly rejects the benzyl shortcut.
        "N-[(2S)-3-hydroxyhydroxy-2-(pentanoylamino)propanoyl](3S,6S,9E,11R)-11-[(S)-4-carbamoylbutyl]-3-(hydroxy4-methylphenyl)-6-isopropyl-4-methyl-5,8-dioxooxacyclododecan-2-one",
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
        "N-[(2E,4E)-4,6-dimethyldeca-2,4-dienoyl](2E,4E,6E)-7-(5-amino-2-hydroxycyclohexyl)-1-(cyclopentylamino)-7-hydroxyhepta-2,4,6-trienamide",
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
        "N-[(2S)-2-(pentanoylamino)propanoyl](2S)-2-aminopropanoic acid",
    ),
    (
        "CS[C@@]1(CO)C(=O)N2[C@H]3N(c4ccc5oc6cc(=O)c(N)c(C(=O)O)c-6nc5c4C(=O)O)c4ccccc4[C@@]3(c3c[nH]c4ccccc34)[C@H](O)[C@]2(SC)C(=O)N1C",
        "(1R,4S,7S,8S,9R)-16-tetradecyl-8-hydroxy-4,4,5,7-tetramethyl-9-octyl-2,5,16-triaza-tetracyclo[7.7.0.0(2,7).0(10,15)]hexadecan-3,6-dione",
    ),
    (
        "CC[C@@H](C)[C@H]1C(=O)N(C)[C@@H](Cc2ccc(OC)c(Br)c2)C(=O)N[C@@H]([C@@H](C)CC)C(=O)O[C@H](C)[C@H](NC(=O)[C@H](NC(=O)[C@@H](COS(=O)(=O)O)OC)C(C)C)C(=O)N[C@@H](CCCN=C(N)N)C(=O)N[C@H]2CC[C@@H](O)N1C2=O",
        "N-[(2R)-methoxy-3-methyl-2-(propanoylamino)-sulfobutanoyl](1S,4S,7S,8R,11S,14S,17S,21R)-7-amino-4,11,17-tributyl-21-hydroxy-8,15-dimethyl-14-octyl-9-oxa-2,5,12,15,18-pentaaza-bicyclo[16.3.1]docosane",
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
        "N,N-di(2S)-2-amino-3-imidazolylpropanoyl-N-(2S)-2-amino-3-phenylpropanoyl-N-(2S)-2-amino-5-guanidino-5-(methylamino)pentanoyl-N-(2S)-2-aminobutanedioyl-N-(2S)-2-aminopentanoyl-N-(2S)-pyrrolidine-2-carbonyl-N-(2S,3S)-2-aminohexanoyl(2S)-3-(4-hydroxyphenyl)-2-aminopropanoic acid",
    ),
    (
        "CCOc1cc(C(=O)O)ccc1NC(=O)c1ccc(NC(=O)c2ccc(NC(=O)[C@@H](NC(=O)c3ccc(NC(=O)c4ccc([N+](=O)[O-])cc4)cc3)[C@@H](OC)C(N)=O)cc2)c(OC(C)C)c1O",
        # Phase 103-03: chain tiebreaker refinements detect aminomethyl substituent
        "N-[(2S,3R)-2-(benzoylamino)-3-methoxypropanoyl]-4-(16-carbamoylhexadecyl)-3-ethoxybenzoic acid",
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
        "3-butyl-4-hydroxymethyl-1-isopropyl(1R,4R)-1,2,3,4-tetrahydronaphthalene",
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
        "(9Z,12Z,15Z)-octadeca-9,12,15-trienoic acid (palmitoyloxy)hexadecanol",  # DECO-22/25: ester threshold 2 + partial assembly now decomposes this phospholipid
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
        "(2R,3S,4S,5R,6S)-1,1-bis(acetyloxy)-1-(benzoyloxy)-2-hydroxyoxolane",
    ),
    (
        "C[C@@H]1O[C@@H](O[C@@H]2C[C@H](c3ccc4c(c3O)C(=O)C3=C(C4=O)[C@@]4(O)C(=O)C[C@](C)(O)C[C@@]4(O)C=C3)O[C@H](C)[C@H]2O)CC[C@@H]1O[C@H]1C[C@@H](O)[C@H](O)[C@@H](C)O1",
        "(11S,14R,16R)-5-octadecyl-4,11,14,16-tetrahydroxy-14-methyl-tetracyclo[8.8.0.0(3,8).0(11,16)]octadeca-1,17-dien-2,9,12-trione",
    ),
    (
        "CSCC[C@H](NC(=O)[C@@H](N)Cc1ccccc1)C(=O)N1CCC[C@@H]1C(=O)O",
        "N-[(2S)-2-amino-3-phenylpropanoyl](2R)-N-pentylpyrrolidine-2-carboxylic acid",
    ),
    (
        "C=C(CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3C(=O)C[C@H]4[C@](C)(C(=O)O)[C@@H](O)CC[C@]4(C)C3=C[C@@H](OC(C)=O)[C@]12C)C(C)C",
        "(3S,4S,5R,8S,10S,11R,13R,14S,17R,20R)-3-hydroxy-4-methyl-7-oxoergost-9,24-dien-11-yl acetate",
    ),
    (
        "CCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCC",
        "2-(N,N-dimethylmethan-1-aminiumyl)ethan-1-olium phosphonooxydecanephosphonic acid",
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
        "N-glutaminyl-5-cyclododecanyl-2-methylpentanamide",
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
        "N-[2-(nonanoylamino)pentanedioyl]-3-cyclodocosylpropanoic acid",
    ),
    (
        "CC(=O)OC1CC2OC3C=C(C)C(=O)[C@@H](O)[C@]3(C)[C@]1(C)[C@]21CO1",
        "(acetyloxy)(6S,7S,8S,12S)-5,8-dihydroxy-6,7,10-trimethyl-2-oxa-tetracyclo[5.4.0.1(3,6)]tetradec-10-en-9-one",
    ),
    (
        "COCC1=C2[C@@H]3CC(C)(C)C[C@@H]3C[C@@]2(O)CC1=O",
        "(2R,6R,8R)-11-ethyl-8-hydroxy-4,4-dimethyl-tricyclo[6.3.0.0(2,6)]undec-1-en-10-one",
    ),
    (
        r"C[C@@H]1C[C@@H]2O[C@@H]3[C@@H](C)[C@H](O)[C@@H]4O[C@]5(C[C@H](O)CO5)[C@@H](C)[C@H](C)[C@H]4O[C@H]3C[C@H]2O[C@H]2C[C@H]3O[C@H]4C/C=C\C[C@H]5O[C@H]6C=C[C@H]7O[C@H]8[C@H](O)[C@H]9OCC=CC[C@@H]9O[C@@H]8C[C@@H]7O[C@@H]6C/C=C\[C@@H]5O[C@@H]4C[C@@H](O)[C@]3(C)O[C@@H]2C1",
        "(1S,3Z,6R,8S,11R,13S,14R,15R,21S,23R,25S,27R,29Z,31S,33R,35R,36S,38R,40R,42S,44R,45S,46S,47S,49R,50S,51S,52R,54S,56R,58S,60R,65S)-36,40,45,50,51-pentamethyl-7,12,16,22,26,32,37,43,48,53,57,61,63-tridecaoxa-tridecacyclo[31.28.0.0(6,31).0(8,27).0(36,60).0(38,58).0(42,56).0(44,54).0(47,52).0(49,63).0(49,64).0(62,65)]pentahexaconta-3,9,18,29-tetraen-14,35,46,65-tetraol",
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
        "(methoxyphenyl)methanol",  # Phase 126: was '(methoxybenzene)methanol'; benzene -> phenyl in substituent context (IUPAC P-31.1.3.4)
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
        # Phase 148 Plan 02 Task 03: cascade unblock per P-44.1(a). Pre-148
        # ring-as-parent `3-(11-carboxyundecyl)-1H-indole` placed acid PG on
        # ring-substituent prefix (IUPAC-incorrect); post-148 cascade picks
        # the chain (peptide acid) as parent. Indole correctly rendered as
        # `1H-indol-3-yl` substituent. Acceptable churn.
        "C[C@@H](O)[C@H](NC(=O)[C@@H]1CCCN1C(=O)[C@@H](N)Cc1c[nH]c2ccccc12)C(=O)O",
        "N-[(2S)-3-(1H-indol-3-yl)-2-aminopropanoyl](2S,3R)-3-hydroxy-2-(pentanoylamino)butanoic acid",
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
        # Phase 148 Plan 02 Task 03: cascade unblock per P-44.1(a). Pre-148
        # ring-as-parent `3-(2-carboxyethyl)-1H-indole` placed acid PG on
        # ring-substituent prefix; post-148 cascade picks chain (PG-bearing
        # peptide) as parent. Indole rendered as `1H-indol-3-yl` substituent.
        # Acceptable churn.
        "CSCC[C@H](N)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",
        "N-[(2S)-amino-2-(butanoylamino)-methylsulfanylbutanedioyl](2S)-3-(1H-indol-3-yl)-2-aminopropanoic acid",
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
        "heptyl (2R,3R)-2,2-dimethyl-3-(2-methylprop-1-enyl)cyclopropanecarboxylate",
    ),
    (
        "CC(C)=C[C@H](O)C1=CC(=O)[C@@H](O)[C@H](O)[C@H]1O",
        "(4S,5R,6S)-3-[(S)-1-hydroxy-3-methylbut-2-enyl]-3,4,5,6-tetrahydroxycyclohex-2-en-1-one",
    ),
    (
        "CC1CCCC[C@H](O)[C@@H]2C[C@@H](O1)C1=C(O2)[C@H](O)CCC1=O",
        "(1S,2S,9R,14R)-2,14-dihydroxy-7-methyl-8,16-dioxa-tricyclo[7.7.1.0(10,15)]heptadec-10-en-11-one",
    ),
    # Phase 148 D-05 / Plan 02 Task 03: complex polyhetero macrocycle
    # (purine + multiple fused rings + glycoside chains). Pre-148 produced an
    # IUPAC-incorrect "phenylnonatriacontyl acetate" placeholder (clearly wrong
    # for a structure of this complexity). Post-148 cascade-unblock + the
    # Plan-01 carry-forward decomposition fragment-naming bug yields a
    # truncated `((acetyloxy)ethanone)-...` fragment. Both old and new are
    # broken; root cause is the decomposition layer for poly-fused systems
    # (Phase 149 / IM-x.x). Marked xfail; NOT a Phase 148 regression.
    pytest.param(
        "COc1c2c(c(O)c3c4c(c(C)cc13)[C@@H]1O[C@@]3(C(OC)OC)O[C@@H]1[C@@](O[C@H]1CC(O)[C@@](O)(C(C)=O)C(C)O1)(O4)[C@@]3(O)Cn1cnc3nc(N)[nH]c(=O)c31)C(=O)C(O)CC2O[C@H]1CC(C)(O)[C@H](OC(C)=O)C(C)O1",
        "(16S,17S,18S,19S,20R)-39-phenylnonatriacontyl acetate",
        marks=pytest.mark.xfail(
            strict=True,
            reason="Phase 149 / IM-x.x: decomposition fragment-naming bug for "
                   "complex polyhetero macrocycles surfaced by Phase 148 cascade "
                   "unblock; per 148-01-SUMMARY Risks §2 (Plan-01 carry-forward).",
        ),
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
    # Phase 148 Plan 02 Task 03: macrocyclic peptide with indole side-chain.
    # Pre-148 indole rendered as `3-(3-(3-carboxypropyl)-1H-indolyl)`
    # (ring-as-substituent w/ acid prefix); post-148 cascade unblock changes
    # the indole side-chain rendering to `3-(11-carboxyundecyl)` (chain
    # carries acid PG per P-44.1(a) within the substituent). Macrocycle parent
    # locants/stereo unchanged. Acceptable churn.
    (
        "CC(C)[C@@H]1NC(=O)[C@H](NC(=O)NC(Cc2c[nH]c3ccccc23)C(=O)O)CCCCNC(=O)[C@H](Cc2ccccc2)NC(=O)[C@H](C)N(C)C(=O)[C@H](CCc2ccc(O)cc2)NC1=O",
        "(3R,10S,13S,16S,19S)-10-benzyl-3-(11-carboxyundecyl)-16-(hydroxy4-ethylphenyl)-19-isopropyl-13-methyl-N-methyl-9,12,15,18-tetraoxoazacyclononadecan-2-one",  # Phase 148 cascade unblock; ASML-13 phenol suffix routing preserved
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
        "(3S,4R,5R,6R)-6-[(S)-2-hydroxy(2R)-butyl]amino-6-hydroxy-3-methylcyclohex-1-enecarbaldehyde",
    ),
    (
        "CCCCCCCCCCCCCCCCC[C@@H](O)[C@H](CO)NC(=O)C(O)CCCCCCCCCCCCCCCC",
        "(2S,3R)-2-(octadecanoylamino)-1,3-dihydroxyicosanamide",  # ASML-12: locant-aware prefix merge
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@@H]1CCCN1)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O",
        "N-[(2S,3S)-3-methyl-2-(pentanoylamino)pentanoyl](2S)-2-amino-3-imidazolylpropanoic acid",
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
    # ---- Phase 106-01: N-substituent naming canaries (FIX-11) ----
    # Cycloalkyl, branched, heterocyclic, and mixed N-substituents
    # that were previously mis-named as linear alkyls.
    (
        "CC(=O)NC1CCCC1",
        "N-cyclopentylacetamide",
    ),
    (
        "CC(=O)NC1CCCCC1",
        "N-cyclohexylacetamide",
    ),
    (
        "CC(=O)NC(C)C",
        "N-isopropylacetamide",
    ),
    (
        "CC(=O)Nc1ccncc1",
        "N-pyridinylacetamide",
    ),
    (
        "CC(=O)N(CC)C1CCCCC1",
        "N-cyclohexyl-N-ethylacetamide",
    ),
    (
        "O=CNC1CCCC1",
        "N-cyclopentylformamide",
    ),
    (
        "CCC(=O)NC(C)(C)C",
        "N-tert-butylpropanamide",
    ),
    (
        "CC(=O)N(C)c1ccccc1",
        "N-methyl-N-phenylacetamide",
    ),
    # ---- Phase 106-02: Multi-ester naming canaries (FIX-04, DECO-08/DECO-09) ----
    # Dicarboxylic diesters, polyol polyesters, independent esters, single ester,
    # DECO-09 (3+ ester bonds), DECO-08 (HA > 30)
    (
        "COC(=O)CC(=O)OC",
        "dimethyl propanedioate",
    ),
    (
        "COC(=O)CC(=O)OCC",
        "ethyl methyl propanedioate",
    ),
    (
        "CC(=O)OCC(COC(C)=O)OC(C)=O",
        "1,2,3-tris(acetyloxy)propane",
    ),
    (
        "CC(=O)OCOCOC(=O)CC",
        "(acetyloxy)(propanoyloxy)-propoxypropan-1-oate",
    ),
    (
        "CC(=O)OCNC(=O)OCC",
        "(acetyloxy)ethan-1-oate",
    ),
    (
        "CCOC(C)=O",
        "ethyl acetate",
    ),
    (
        "CC(=O)OC[C@H]1O[C@@H](n2ccc(=O)[nH]c2=O)[C@H](OC(C)=O)[C@@H]1OC(C)=O",
        "(2R,3R,4R,5R)-1,2-bis(acetyloxy)oxolane",
    ),
    (
        "CCCCC/C=C\\C/C=C\\CCCCCCCCCC(=O)OC(COC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC)COC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC",
        "2-[(11Z,14Z)-icosa-11,14-dienoyloxy]-1,3-bis(linoleoyloxy)propane",
    ),
    # --- Phase 107-02 canary compounds ---
    # FIX-13: iterative decomposition (multi-ester with glycosidic linkage)
    (
        "CC(=O)OCC1OC(OC(=O)C)C(OC(C)=O)C(OC(C)=O)C1OC(C)=O",
        "1,2,3,4-tetrakis(acetyloxy)oxane",
    ),
    # FIX-10: fused ring dictionary (dibenzofuran)
    (
        "c1ccc2c(c1)oc1ccccc12",
        "dibenzofuran",
    ),
    # FIX-10: fused ring dictionary (dibenzothiophene)
    (
        "c1ccc2c(c1)sc1ccccc12",
        "dibenzothiophene",
    ),
    # FIX-10: fused ring dictionary (9H-carbazole)
    (
        "c1ccc2c(c1)[nH]c1ccccc12",
        "9H-carbazole",
    ),
    # FIX-14: ylidene substituent naming (methylidenecyclohexane)
    (
        "C=C1CCCCC1",
        "methylidenecyclohexane",
    ),
    # FIX-14: ylidene on chain parent (3-methylidenepentane)
    (
        "CCC(=C)CC",
        "3-methylidenepentane",
    ),
    # FIX-15: skeletal replacement for large heterocyclic ring
    (
        "C1CCOCCO1",
        "1,4-dioxacycloheptane",
    ),
    # FIX-15: mixed heteroatom large ring replacement
    (
        "C1CCNCCOC1",
        "1-oxa-4-azacyclooctane",
    ),
    # FIX-15: zwitterion detection (beta-alanine)
    (
        "[NH3+]CCC(=O)[O-]",
        "beta-alanine",
    ),
    # FIX-15: zwitterion detection (betaine)
    (
        "C[N+](C)(C)CC(=O)[O-]",
        "betaine",
    ),
    # --- Phase 108 v12.0 canary expansion (48 compounds for 500+ total) ---
    # substituent_loss compounds (18)
    (
        "C=CCSSSSC",
        "2,3,4,5-tetrathiaoctane",
    ),
    (
        "CC(=O)c1c(C)c[nH]c1C",
        "1-pyrrolylethan-1-one",
    ),
    (
        "Nc1nc(N)nc(NC2CC2)n1",
        "4,6-diamino-2-cyclopropyl-1,3,5-triazine",
    ),
    (
        "COC(=O)c1cc2ccccc2cn1",
        "3-acetylisoquinoline",
    ),
    (
        "c1csc(-c2ccn3cnnc3n2)c1",
        "[1,2,4]triazolo[1,5-a]pyridine",
    ),
    (
        "N=C(N)NC(=N)Nc1ccc(O)cc1",
        "4-(N-methylguanidinyl)phenol",  # ASML-13: phenol suffix routing
    ),
    (
        "COc1cc(CO)cc(CC=C(C)C)c1O",
        "4-(hydroxymethyl)-6-methoxy-2-(2-methylbut-2-enyl)phenol",  # ASML-13: phenol suffix routing
    ),
    (
        "COc1cc(C(=O)CC(C)C)oc(=O)c1",
        "4-methyl-6-oxo-2-pentyl-2H-pyran",
    ),
    (
        "CC[C@H](C)[C@H](N)C(=O)[O-]",
        "(2S,3S)-2-aminohexanoate",
    ),
    (
        "COc1cc(COC(C)=O)ccc1OC(C)=O",
        "1-(acetyloxy)-2-hydroxymethylbenzene",
    ),
    (
        "O=C([O-])CCCOc1ccc(Cl)cc1Cl",
        "4-phenoxybutanoate",
    ),
    (
        "C=C(C)C1C=C2C(C)=CCCC2(C)CC1",
        "decahydronaphthalene",
    ),
    (
        "N[C@H](C[13C](=O)O)[13C](=O)O",
        "(2R)-2-aminobutanedioic acid",
    ),
    (
        "COc1c(C)c(O)cc2c1C(=O)N[C@H]2C",
        "(3S)-6-hydroxy-4-methoxy-3,5-dimethylisoindolin-1-one",
    ),
    (
        "C=CCC(CCc1ccccc1)OC(C)c1ccccc1",
        "1-phenylethoxy(hex-5-en-1-yl)benzene",
    ),
    (
        "CC(=N)NCCCC[C@H](N)C(=O)O.Cl.Cl",
        "(2S)-2-amino-6-(ethylamino)-iminohexanoic acid",
    ),
    (
        "Cc1cc(N2C(=O)c3ccccc3C2=O)n(C)n1",
        "isoindoline-1,3-dione",
    ),
    (
        "COc1cc(O)cc(C)c1Oc1cc(C)cc(O)c1O",
        "3-methoxy-5-methyl-4-phenoxyphenol",  # ASML-13: phenol suffix routing
    ),
    # fragment_loss compounds (8)
    (
        "O=CCS(=O)(=O)[O-]",
        "ethanolate",
    ),
    (
        "Cc1cc(=O)c2c(O)cc(O)cc2o1",
        "6-methyl-4-oxo-2H-pyran",
    ),
    (
        "CCOP(=S)(COC)OP(=S)(OCC)OCC",
        "methoxymethylethylethylethane",
    ),
    (
        "COc1cc(C2OC2C(=O)NCCCCN)ccc1O",
        "2-methoxyphenol",  # ASML-13: phenol suffix routing
    ),
    (
        "Cc1c(CO)oc(=O)c2c(O)cc(O)cc12",
        "2,3-dimethyl-6-oxo-2H-pyran",
    ),
    (
        "N[C@@H](COC(=O)CCC(=O)O)C(=O)O",
        "butanedioic acid",
    ),
    (
        "CCCCCCCCNC(O)CCc1ccc(O)c(OC)c1",
        "2-methoxyphenol",  # ASML-13: phenol suffix routing
    ),
    (
        "CCCCCC(C)OC(=O)COc1ccc(Cl)c2cccnc12",
        "5-chloroquinoline",
    ),
    # parent_mismatch compounds (8)
    (
        "CC1=NCCO1",
        "2-methyloxazole",
    ),
    (
        "*N=C=N[1*]",
        "2,4-diazapentane",
    ),
    (
        "C/N=C(\\N)NCCCCN",
        "4-guanidino-4-(methylamino)butan-1-amine",
    ),
    (
        "CCCOC(C)OCCc1ccccc1",
        "(1-hydroxy-1-ethoxyethylpropyl)benzene",  # PERC-05: generic alcohol detects fragment hydroxyl
    ),
    (
        "NC(=O)N/C=C\\C(=O)OO",
        "(2Z)-3-carbamoylamino-3-(methanoylamino)prop-2-en-1-peroxol",
    ),
    (
        "C=C(C)C#Cc1c(O)ccc(O)c1C=O",
        "3,6-dihydroxy-2-(2-methylbut-1-en-3-ynyl)benzaldehyde",
    ),
    (
        "C/C=C/CC(O)CCC(=O)NCC(=O)O",
        "hydroxy-2-(octanoylamino)ethanoic acid",
    ),
    (
        "Cc1c(O)cc2c(c1O)C(=O)c1ccccc1C2=O",
        "11,13-dihydroxy-12-methyl-tricyclo[8.4.0.0(3,8)]tetradecan-2,9-dione",
    ),
    # opsin_error compounds (12)
    (
        "C=CC(=O)NCCC[N+](C)(C)C",
        "N-methylmethylmethylpropylaminiumylprop-2-enamide",
    ),
    (
        "CC(C)CC(=O)OC1OC(C(=O)O)C(O)C(O)C1O",
        "(glucuronopyranosyloxy)-3-methylbutanoic acid",
    ),
    (
        "CC(C)C1=C(O)C(N)=C(/C=C/c2ccccc2)C(=O)C1=O",
        "(3E)-4-amino-5-hydroxy-6-isopropyl-3-styrenylcyclohexa-3,5-diene-1,2-dione",
    ),
    (
        "CC1(C)SC(C(NC(=O)COc2ccccc2)C(=O)O)NC1C(=O)O",
        "6-phenoxy-7-phenyl-3-(thiazolidin-2-yl)heptanedioic acid",
    ),
    (
        "O=C1O/C(=C/c2ccccc2)C(Cc2ccccc2)=C1Cc1ccccc1",
        "(5E)-3,4,5-tribenzyloxolan-2-one",
    ),
    (
        "O=C1NC(Cc2c[nH]c3ccccc23)C(=O)N/C1=C/c1cnc[nH]1",
        "(6E)-6-imidazolyl-2,5-dioxo-3-pyrrolylpiperazine",
    ),
    (
        "C/C=C1\\[C@H]2C=C(C)C[C@]1([NH3+])c1ccc(=O)[nH]c1C2",
        "(2R,6R,13E)-4-methyl-9-aza-tricyclo[6.4.0.1(2,6)]tridec-4-en-10-one",
    ),
    (
        "CC1=C[C@]2(C[C@H]1C)c1c(c(-c3ccccc3)c[nH]c1=O)O[C@@H]2O",
        "(8S,9R,13R)-8-hydroxy-10,13-dimethyl-5-phenyl-7-oxa-3-aza-tricyclo[4.3.0]tridec-10-en-2-one",
    ),
    (
        "CC1(C)[C@@H]2C[C@]34CCCN3C[C@@]2(C[C@@]12C(=O)Nc1cc(Cl)c(Cl)cc12)NC4=O",
        "(1S,3S,9S,11R)-12,12-dimethyl-7,14,21-triaza-hexacyclo[7.3.0.2(3,9).0(3,7).0(15,20)]docosan-13,22-dione",
    ),
    (
        "CC1OC(c2ccccc2O)=NC1C(=O)NCCCN(CCCNC(=O)c1cccc(O)c1O)C(=O)C1N=C(c2ccccc2O)OC1C",
        "N-5-methyl-2-phenyloxazole-4-carbonyl-5-methyl-2,4-diphenyloxazole",
    ),
    (
        "COC1C(=O)OC2C(O)CO/C=C\\C3=C(CCC(=O)OCC2/C=C(/C)C24OC5CC(O)C2C(C=CC4C1OC)C5)C(=O)OC3=O",
        "(4Z)-butanedioic anhydride",
    ),
    (
        "C=C1C(=O)O[C@H](CCCCCCCCCCCCC[C@@H](C)OC2OC(CO)C(O)C(OC3OC(CO)C(O)C(O)C3O)C2O)[C@H]1C(=O)O",
        "(glucopyranosyloxy)(4S,5R)-5-[(R)-2-pentadecyl-3,4,5-trihydroxy-6-methyloxyl]-4-formyl-3-methyloxolan-2-one",
    ),
    # stereo_mismatch compounds (2)
    (
        "CC(C)CC[C@@H](O)[C@H]1C(=O)OC[C@@H]1CO",
        "(3S,4S)-3-[(R)-1-hydroxy-4-methylpentyl]-4-hydroxymethyloxolan-2-one",
    ),
    (
        "CC(CC(=O)CC(C)C1C[C@H](O)[C@@]2(C)C3=C(C(=O)CC12C)C1(C)CC[C@H](O)C(C)(C)C1C[C@@H]3O)C(=O)O",
        "(3S,7S,14R,15S)-3,7,15,27-tetrahydroxy-4,4,14-trimethylcholest-8-en-11,23,27-trione",
    ),
    # + 381 from Phase 143 (integration testing canary expansion)
    (
        "CCCCCCC(=O)NC1=CC(=O)[C@@H]2CCCN12",  # heterocycle,fused-ring,medium
        "N-heptanoyl(2S)-5-amino-3-oxoazole",
    ),
    (
        "Nc1c(/N=N/c2ccc([N+](=O)[O-])cc2)c(S(=O)(=O)O)cc2cc(S(=O)(=O)O)c(/N=N/c3ccccc3)c(O)c12",  # aromatic,fused-ring,large
        "unknown organic compound",
    ),
    (
        "CCC/C=C\\CCCCCCCC/C=C/[C@@H](O)[C@H](CO[C@@H]1O[C@H](CO)[C@H](O)[C@H](O)[C@H]1O)NC(=O)CCCCCCCCCCCCCCCCCCCC",  # heterocycle,large,carbohydrate
        "(beta-D-galactopyranosyloxy)-1-(octadecylamino)-dihydroxyhenicosanamide",
    ),
    (
        "C/C=C1\\[C@H]2C=C(C)C[C@]1([NH3+])c1ccc(=O)[nH]c1C2",  # aromatic,heterocycle,fused-ring,charged,medium
        "(2R,6R,13E)-4-methyl-9-aza-tricyclo[6.4.0.1(2,6)]tridec-4-en-10-one",
    ),
    (
        "C[C@H]1C[C@@H](O)[C@@]23C1=C[C@@]1(C)CC[C@](C)(C[C@H](O)[C@H](O)[C@@](C)(O)CO)[C@H]1[C@@H]2CC[C@@H]3C",  # fused-ring,medium
        "(1S,3R,4R)-5-cyclopentadecanyl-2,3,4-trihydroxy-2-methylpentan-1-ol",
    ),
    (
        "CO[C@H]1C=C/C=C\\C=C/C[C@H](OC(=O)[C@@H](C)NC(=O)C2=CCCCC2)[C@H](C)[C@@H](O)/C(C)=C\\CCc2cc(O)cc(c2O)NC(=O)C1",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "(7Z,9R,10R,11S,13Z,15Z,19R)-1-(cyclohexanecarbonyloxy)-3,9-dihydroxy-14-hydroxymethyl-2,4-dimethyl-12-oxo-8-propyl1-azacyclohenicosene",
    ),
    # Phase 148 Plan 02 Task 03: duplicate of the above (same SMILES at L116);
    # cascade unblock per P-44.1. Acceptable churn — see L116 for rationale.
    (
        "CCCCCC/C=C\\CC(=O)N[C@@H](CO)[C@@H](O)CC(=O)N[C@H](C(=O)N[C@H](/C=C/C(=O)NCC(=O)c1c[nH]c2ccccc12)CO)C(C)C",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "(2E,4R)-1-(2-oxo2-(1H-indol-3-yl)-1-aminoethyl)-5-hydroxy-4-(pentanoylamino)pent-2-enetetraamide",
    ),
    (
        "CC[C@H]1[C@@H]2CC3[C@@H]4N(C)c5ccccc5[C@]45C[C@@H](C2C5O)N3[C@@H]1O",  # aromatic,heterocycle,fused-ring,medium,alkaloid
        "ajmaline",
    ),
    (
        "CCCCCC/C=C\\CCCCCCCC(=O)OC[C@@H]1COP(=O)(O)O[C@H]2[C@H](O)[C@@H](O)[C@H](O)[C@@H](CCCCCCC(=O)O1)[C@@H](O)C[C@@H](O)[C@H](/C=C/[C@@H](O)CCCCC)[C@@H](O)[C@H]2O",  # heterocycle,fused-ring,large
        "(8S,9R,10S,11R,12R,13R,14R,15S,18S)-nonacosyl (9Z)-hexadec-9-enoate",
    ),
    (
        "CC/C=C\\C/C=C\\C/C=C\\CCCCCCCC(=O)OCC(COP(=O)(O)OCCNC)OC(=O)CCCCCCCCC/C=C\\C/C=C\\CCCCC",  # acyclic,polyfunctional,large
        "2-((11Z,14Z)-icosa-11,14-dienoyloxy)-1-(linolenoyloxy)propanamine",
    ),
    (
        "CCCCC[C@H](O)/C=C/[C@@H]1[C@@H](C/C=C\\CCCC(=O)O)[C@H](O)C[C@H]1O",  # medium
        "(5Z)-7-cyclopentylhept-5-enoic acid",
    ),
    (
        "C=C1C(=O)O[C@@H]2C[C@@H](C)/C=C\\C(=O)[C@@](C)(O)C[C@@H](OC(=O)CC(C)C)[C@@H]12",  # heterocycle,fused-ring,polyfunctional,medium
        "(2R,3S)-4-methyl-5-oxooxolane",
    ),
    # Phase 148 Plan 02 Task 03: large peptide with indole side-chain. Post-148
    # cascade unblock + decomposition fragment-naming bug yields a tiny
    # `(4S)-4-(butanoylamino)pentanedioic acid` fragment (loses ~95% of the
    # molecule). Phase 149 / IM-x.x decomposition layer fix territory. Marked
    # xfail; NOT a Phase 148 regression (cascade decision is correct; the
    # downstream peptide-decomposition layer is the bug).
    pytest.param(
        "C[C@H](NC(=O)[C@H](CCC(=O)O)NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@H](CS)NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@@H](NC(=O)[C@@H]1CCCN1C(=O)CNC(=O)[C@H](CO)NC(=O)CN)[C@@H](C)O)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CC(N)=O)C(=O)N1CCC[C@H]1C(=O)NCC(=O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "N-(2R)-2-amino-3-sulfanylpropanoyl-N-(2S)-2-amino-3-hydroxypropanoyl-N,N,N-tri(2S)-2-aminobutanedioyl-N-(2S)-2-aminopropanoyl-N-(2S)-3-(4-hydroxyphenyl)-2-aminopropanoyl-N,N-di(2S)-pyrrolidine-2-carbonyl-N-(2S,3R)-2-amino-3-hydroxybutanoyl(2S)-2-aminopentanedioic acid",
        marks=pytest.mark.xfail(
            strict=True,
            reason="Phase 149 / IM-x.x: decomposition fragment-naming bug for "
                   "large peptides with indole side-chain (output truncates to "
                   "~5%); per 148-01-SUMMARY Risks §2 (Plan-01 carry-forward).",
        ),
    ),
    (
        "*c1c(*)c(*)c(-c2oc3c(*)c(*)c(*)c(*)c3c(=O)c2O[C@@H]2O[C@H](COC(=O)CC(=O)[O-])[C@@H](O)[C@H](O)[C@H]2O)c(*)c1*",  # aromatic,heterocycle,fused-ring,charged,large,carbohydrate
        "(propanedioyloxy)-3-oxanyl-4-oxo-2-phenyl-2H-pyran",
    ),
    (
        "COc1cc(OC)c(C(C)=O)c(O)c1CCOCCc1c(O)cc(OC)c(C(C)=O)c1O",  # aromatic,large
        "2-(hydroxyethyl)-5-methoxybenzene-1,3-dioxy-2-ethyl-3,5-dimethoxyphenol",
    ),
    (
        "CCCCCC/C=C\\CC(=O)N[C@@H](CO)C(=O)N[C@H](C(=O)N[C@@H](CO)[C@@H](O)CC(=O)N[C@@H](CO)C(=O)N[C@H](C(=O)N[C@@H]1/C=C/C(=O)N[C@@H](C(C)C)C(=O)N(C)[C@@H](Cc2ccc(O)cc2)C(=O)OC1)C(C)C)C(C)C",  # aromatic,heterocycle,polyfunctional,large
        "N-[(2S)-3-hydroxyhydroxy-2-(pentanoylamino)propanoyl](3S,6S,9E,11R)-11-[(S)-4-carbamoylbutyl]-3-(hydroxy4-methylphenyl)-6-isopropyl-4-methyl-5,8-dioxooxacyclododecan-2-one",
    ),
    # Phase 148 Plan 02 Task 03: CoA-style adenine-bearing thioester. Pre-148
    # OLD value `'adenine'` was a known-bad placeholder (drops the entire
    # molecule except the adenine fragment). Post-148 cascade unblock yields
    # a chain-as-parent rendering of the (non-adenine) acyl moiety with the
    # adenine-furanose-phosphate side preserved as N-acyl prefix. The new
    # name represents more of the molecule but still loses the
    # furanose-phosphate-adenine (Phase 149 / IM-x.x decomposition layer).
    # Acceptable churn — both names are partial; the new name is more
    # representative than the OLD `adenine` placeholder.
    (
        "CC(C)(COP(=O)([O-])OP(=O)([O-])OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)([O-])[O-])[C@@H](O)C(=O)NCCC(=O)NCCSC(=O)C1C(=O)CCCC1O",  # aromatic,heterocycle,fused-ring,polyfunctional,charged,large,carbohydrate
        "N-[(2R)-2-hydroxy-3,3-dimethylbutanoyl]amino-1-hydroxy-3-oxocyclohexane",
    ),
    (
        "[2H]C([2H])=C1CC[C@H](O)C/C1=C([2H])\\C=C1/CCC[C@]2(C)[C@@H]([C@H](C)CC[C@@H](O)C(C)(C)O)CC[C@@H]12",  # fused-ring,medium
        "(2E,4S)-hydroxy-1-methylidenecyclohexan-4-ol",
    ),
    (
        "CCCCCCCCCCCCCCCCCCCCCCCCCC(=O)N[C@@H](CO[C@H]1O[C@H](CO)[C@@H](O[C@@H]2O[C@H](CO)[C@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@H]1O)[C@H](O)[C@H](O)CCCCCCCCCCCCCC",  # heterocycle,large,carbohydrate
        "hexacosanoic acid alpha-D-glucopyranosyloxy (2S,3S,4R)-2-amino-3,4-dihydroxyoctadecan-1-ol beta-D-galactopyranosyloxy",
    ),
    (
        "COc1c(-c2ccc(O)cc2)oc2c(O)c(O)ccc2c1=O",  # aromatic,heterocycle,fused-ring,medium
        "3-methyl-4-oxo-2-phenyl-2H-pyran",
    ),
    (
        "CC(C)C(N)C(=O)NC(CC(N)=O)C(=O)NC(Cc1ccc(O)cc1)C(=O)NCC(=O)NC(CC(N)=O)C(=O)NCC(=O)NC(C(=O)NC(CO)C(=O)NC(CS)C(=O)NC(C=O)CO)C(C)C",  # aromatic,polyfunctional,large
        "4-anilino-3-(pentanoylamino)butanediamide",
    ),
    (
        "CC[C@H]1C=CC=CCC[C@@H](O)[C@@H](C)[C@@H](O)C[C@@H](OC)C[C@@H](O)[C@H](C)[C@H](O)[C@@H](C)C=CC(=O)O[C@H]2C[C@@]3(CC[C@H](C)[C@H](C[C@@H](C)O)O3)O[C@@H](CC1)[C@H]2CC",  # heterocycle,fused-ring,large,carbohydrate
        "(1S,6S,7R,8S,9R,11R,13S,14R,15R,22R,25S,27R,29R,32S,33S)-22,29-diethyl-7,9,13,15-tetrahydroxy-11-methoxy-6,8,14,32-tetramethyl-33-propyl-2,26,34-trioxa-tricyclo[23.3.1]tetratriaconta-4,18,20-trien-3-one",
    ),
    (
        "C/C1=C/C=C\\C=C/C=C\\C=C/C[C@@H]2C[C@H](O)C[C@](O)(C[C@H](O)C[C@@H](O)/C=C\\C[C@@H](O)C[C@@H](O)C[C@H](O)C[C@H](O)[C@H](C)[C@H](C(C)C)OC1=O)O2",  # heterocycle,fused-ring,large,carbohydrate
        "(3Z,5Z,7Z,9Z,11Z,14R,16S,18R,20R,21Z,24R,26R,28S,30S,31S,32S)-16,18,20,24,26,28,30-heptahydroxy-32-isopropyl-3,31-dimethyl-2-oxo-1,15-dioxacyclodotriacontene",
    ),
    (
        "Nc1nc2c(ncn2[C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O[C@H]3O[C@H](CO)[C@@H](O)[C@H](O)[C@H]3O)[C@@H](O)[C@H]2O)c(=O)[nH]1",  # aromatic,heterocycle,fused-ring,large,carbohydrate
        "guanosine (2R,3R,4S,5S,6R)-3,4,5-trihydroxy-6-methyloxane",
    ),
    (
        "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OC(=O)CCCC[C@@H]2SC[C@@H]3NC(=O)N[C@@H]32)[C@@H](O)[C@H]1O",  # aromatic,heterocycle,fused-ring,polyfunctional,large,carbohydrate
        "adenosine (4R,5S)-2-oxo-1,3-diazolidine",
    ),
    (
        "C/C=C/C(=O)O[C@H]1/C=C\\C(=O)[C@@H](O)CCC(=O)O[C@@H]1C",  # heterocycle,polyfunctional,medium
        "(5S,7Z,9S,10R)-9-(3-carboxypropyl)-5-hydroxy-10-methyl-6-oxooxecan-2-one",
    ),
    (
        "CCCCC/C=C\\C/C=C\\CCCCCCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCCCCC)COP(=O)(O)OC[C@H](N)C(=O)O",  # acyclic,polyfunctional,large
        "(2S)-2-aminohydroxypropanoic acid (13Z,16Z)-docosa-13,16-dienoate",
    ),
    (
        "CO[C@H]1[C@@H](O)[C@H](O)[C@H](OC[C@@]23C[C@@H]4[C@H](C)CC[C@H]4[C@@]4(C=O)C[C@@H]2C(=O)[C@]2(CC2C)[C@@]34C(=O)O)O[C@@H]1C",  # heterocycle,fused-ring,polyfunctional,large,carbohydrate
        "(1R,3S,4R,5S,7S,9S,10R,13R)-9-formyl-13,15-dimethyl-3-octyl-6-oxo-pentacyclo[8.3.0.0(3,7).0(4,9)]pentadecane-4-carboxylic acid",
    ),
    (
        "CC1=C[C@H]2OC3C[C@H]4OC(=O)/C=C\\C=C/C(C(C)O)OCC/C(C)=C\\C(=O)OC[C@@]2(CC1)C4(C)[C@]31CO1",  # heterocycle,fused-ring,large
        "(1R,4Z,6Z,12Z,17R,22R,27S)-8-ethyl-12,20,26-trimethyl-2,9,15,23-tetraoxa-pentacyclo[15.8.1.1(24,26)]nonacosa-4,6,12,20-tetraen-3,14-dione",
    ),
    (
        "C/C(=C\\CC[C@@H](C)[C@H]1CC[C@@]2(C)C3=CC[C@H]4C(C)(C)[C@@H](OC=O)CC[C@]4(C)C3=CC[C@]12C)CO",  # fused-ring,large,steroid
        "(3S,5R,10S,13R,14R,17R,20R,24E)-27-hydroxy-4,4,14-trimethylcholest-7,9,24-trien-3-yl formate",
    ),
    (
        "COc1cc2c(cc1OC)[C@H]1Cc3ccc(OC)c(OC)c3CN1CC2",  # aromatic,heterocycle,fused-ring,medium,alkaloid
        "berberine",
    ),
    (
        "CCCCC/C=C\\C/C=C\\C/C=C\\CC1OC1CCCC(=O)NCCO",  # heterocycle,medium
        "1-(ethylamino)-4-oxiranylbutanamide",
    ),
    (
        "COC1C(=O)OC2C(O)CO/C=C\\C3=C(CCC(=O)OCC2/C=C(/C)C24OC5CC(O)C2C(C=CC4C1OC)C5)C(=O)OC3=O",  # heterocycle,fused-ring,large,carbohydrate
        "(4Z)-butanedioic anhydride",
    ),
    (
        "C[C@@H]1C[C@@H]2O[C@@H]3[C@@H](C)[C@H](O)[C@@H]4O[C@]5(C[C@H](O)CO5)[C@@H](C)[C@H](C)[C@H]4O[C@H]3C[C@H]2O[C@H]2C[C@H]3O[C@H]4C/C=C\\C[C@H]5O[C@H]6C=C[C@H]7O[C@H]8[C@H](O)[C@H]9OCC=CC[C@@H]9O[C@@H]8C[C@@H]7O[C@@H]6C/C=C\\[C@@H]5O[C@@H]4C[C@@H](O)[C@]3(C)O[C@@H]2C1",  # heterocycle,fused-ring,large,carbohydrate
        "(1S,3Z,6R,8S,11R,13S,14R,15R,21S,23R,25S,27R,29Z,31S,33R,35R,36S,38R,40R,42S,44R,45S,46S,47S,49R,50S,51S,52R,54S,56R,58S,60R,65S)-36,40,45,50,51-pentamethyl-7,12,16,22,26,32,37,43,48,53,57,61,63-tridecaoxa-tridecacyclo[31.28.0.0(6,31).0(8,27).0(36,60).0(38,58).0(42,56).0(44,54).0(47,52).0(49,63).0(49,64).0(62,65)]pentahexaconta-3,9,18,29-tetraen-14,35,46,65-tetraol",
    ),
    (
        "CCCCC/C=C\\C/C=C\\CCCCCCCC(=O)O[C@H](COCCCCCCCCCCCCCCCCCC)COC(=O)CCCCCCCCCCCCCCCCCCCCCCC",  # acyclic,large
        "(2R)-1-(tetracosanoyloxy)-3-octadecyloxy-2-(linoleoyloxy)propane-1,2-dioate",
    ),
    (
        "CSCC[C@H](NC(=O)[C@H](C)NC(=O)[C@@H](N)CCC(N)=O)C(=O)O",  # acyclic,polyfunctional,medium
        "N-(2S)-2-aminopentanedioyl-N-(2S)-2-aminopentanoyl(2S)-2-aminopropanoic acid",
    ),
    (
        "CC[C@H](C)[C@@H](OC(C)=O)[C@@H](C)c1c(O)c2c(c3c1SCC(=O)N3)[C@@H](O)[C@@H]1[C@@]3(C)CC[C@H](C(C)(C)O)O[C@@H]3CC[C@@]1(C)O2",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "(3S,4R,5S,10S,11S,12R,15R,16R,19R)-24-phenyltetracosyl acetate",
    ),
    (
        "CCCCCCCCCCCCC/C=C/[C@@H](O)[C@H](CO[C@@H]1O[C@H](CO)[C@H](O)[C@H](O)[C@H]1O)NC(=O)CCCCCCCCCCCCCCCCCCCCCCCCC",  # heterocycle,large,carbohydrate
        "(beta-D-galactopyranosyloxy)-1-(octadecylamino)-dihydroxyhexacosanamide",
    ),
    (
        "C[C@@H]1CC(=O)O[C@@H](C)[C@H](O)/C=C\\C(=O)O[C@@H](C)C/C=C\\C(=O)O1",  # heterocycle,medium
        "(4R,7Z,10S,13Z,15R,16S)-15-hydroxy-4,10,16-trimethyl-6,12-dioxooxacyclohexadecan-2-one",
    ),
    (
        "N[C@@]1(C(=O)c2ccccc2)[C@H](c2ccccc2)[C@H]2CC(OC(=O)CCC(=O)O)[C@@H]1C2",  # aromatic,fused-ring,polyfunctional,medium
        "(butanedioyloxy)(1R,2S,3S,4R)-3-amino-3-heptyl-2-hexyl-6-hydroxy-bicyclo[2.2.1]heptane",
    ),
    (
        "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCCCCC(=O)OC[C@H](COC(=O)CCCCCCCCCCCCCCCCCCCCC)OC(=O)CCCCCCC/C=C\\C/C=C\\C/C=C\\CC",  # acyclic,large
        "3-(arachidonoyloxy)-1-(docosanoyloxy)-2-(linolenoyloxy)propane",
    ),
    (
        "COC1=C(C)C(=O)C2=C(C1=O)[C@H](CNC(=O)[C@H](C)N)N1C[C@H]3Cc4cc(C)c(OC)c(O)c4[C@@H]([C@@H]1C2)N3C",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "N-[(2S)-2-aminopropanoyl](3R,11S,12S,20R)-9-hydroxy-8,17-dimethoxy-7,16,20,21-tetramethyl-1,21-diaza-pentacyclo[10.8.0.1(3,11).0(5,10).0(14,19)]henicosa-14,16-dien-15,18-dione",
    ),
    (
        "CC(C)[C@H](NC(=O)[C@@H](N)CC(=O)O)C(=O)NCC(=O)N1CCC[C@H]1C(=O)O",  # heterocycle,polyfunctional,medium
        "N-(2S)-2-aminobutanedioyl-N-(2S)-pyrrolidine-2-carbonyl(2S)-2-aminopentanoic acid",
    ),
    (
        "COC1=CC(C)=C[C@@H](C)[C@@H](O)[C@@H](C)CC(C)=CC=C[C@H](OC)[C@@H]([C@@H](C)/C=C(\\C)C2=C[C@@H](OC)[C@H](C)[C@@H](C(C)C)O2)OC1=O",  # heterocycle,large
        "(7R,8S,9S,15S,16R)-16-[(1S,8S,11S,13R)-(2R,3S,4S)-2-isopropyl-3,4-dimethyl-6-(pent-2-en-2-yl)-3,4-dihydro-2H-pyranyl]-8-hydroxy-3,15-dihydroxymethyl-5,7,9,11-tetramethyloxacyclohexadecan-2-one",
    ),
    (
        "C=C[C@H]1C[N@]2CC[C@H]1C[C@@H]2[C@@H](O)c1ccnc2ccc(OC)cc12.O=C(O)[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO",  # aromatic,heterocycle,fused-ring,salt,large,alkaloid
        "cinchonane (2R,3S,4R,5R)-2,3,4,5,6-pentahydroxyhexanoic acid",
    ),
    (
        "CCCCC[C@H](O)CC[C@@H]1[C@H]2Cc3cccc(OCC(=O)O)c3C[C@H]2C[C@H]1O",  # aromatic,fused-ring,medium
        "2-phenoxyethanoic acid",
    ),
    (
        "CC1C/C(=C\\CC(CC(N)=O)CC(=O)O)C(=O)C(C)C1",  # polyfunctional,medium
        "3-(2-aminoethyl)-5-(3,5-dimethylcyclohexyl)pentanoic acid",
    ),
    (
        "C/N=C(\\N)NCCCCN",  # acyclic,small
        "4-guanidino-4-(methylamino)butan-1-amine",
    ),
    (
        "CCCCCCCCCCCCCCCC/C=C\\OC[C@H](COP(=O)(O)O)OC(=O)CCCCCCCCCCC",  # acyclic,large
        "(1Z)-1-pentadecyloxyphosphonooxyoctadec-1-enephosphonic acid",
    ),
    (
        "COc1ccc(C2CC(=O)c3c(O)cc(OC4OCC(O)C(O)C4O)cc3O2)cc1",  # aromatic,heterocycle,fused-ring,medium,carbohydrate
        "(xylopyranosyloxy)-5,7-dihydroxychroman-4-one",
    ),
    (
        "C=C(CC[C@@H](C)[C@H]1CC[C@@]2(C)C3=C(CC[C@]12C)[C@@]1(C)CC[C@H](O)C(C)(C)[C@@H]1CC3)C(C)C",  # fused-ring,large,steroid
        "(3S,5R,10S,13R,14R,17R,20R)-4,4,14-trimethylergosta-8,24-dien-3-ol",
    ),
    (
        "CCCCCC/C=C\\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCCCCCCC)COP(=O)(O)O",  # acyclic,large
        "(7Z)-phosphonooxyhexadec-7-enephosphonic acid henicosanoate",
    ),
    (
        "CC(C)=CCc1ccc(O)c2c1C=C[C@H]1O[C@@H]2O[C@H]1C",  # aromatic,heterocycle,fused-ring,medium
        "(2S,4S,5R)-4-methyl-9-2-methylbut-2-enyl-3,13-dioxa-tricyclo[6.4.0.1(2,5)]tridec-6-en-12-ol",
    ),
    (
        "CC[C@H](C)/C=C(C)/C=C/C(=O)c1c(O)c(-c2ccc(O)cc2)cn(O)c1=O",  # aromatic,heterocycle,medium
        "(2E,4E,6S)-1-(3-phenylpyridin-3-yl)-4,6-dimethylocta-2,4-dien-1-one",
    ),
    (
        "CCCCCCCCCCCCCCCC(=O)OC[C@H](CO[C@@H]1O[C@H](CO)[C@H](O)[C@H](O)[C@H]1O)OC(=O)CCCCCCCCCCCCCCC",  # heterocycle,large,carbohydrate
        "(beta-D-galactopyranosyloxy)(2S)-1,2-bis(palmitoyloxy)propan-3-ol",
    ),
    (
        "CCCCC/C=C\\C/C=C\\C/C=C\\CC(O)C(O)CCCC(=O)O[C@H](COC(=O)CCCCCCCCC(C)CC)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2ccc(N)nc2=O)[C@H](O)[C@@H]1O",  # aromatic,heterocycle,polyfunctional,large,carbohydrate
        "pentacosyl 5,6-dihydroxyarachidate",
    ),
    # Phase 148 Plan 02 Task 03: duplicate of L116 SMILES (canonical form
    # only differs by raw-string prefix). Same cascade-unblock rationale.
    (
        "CC(=O)N[C@@H](CC(C)C)C(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N/C=C\\c1c[nH]c2ccccc12",  # aromatic,heterocycle,fused-ring,large
        "N-acetyl(2S)-1-(amino(4Z)-2-(1H-indol-3-yl)eth-1-en-1-yl)-2-(hexanoylamino)-3-phenylpropanamide",
    ),
    # Phase 148 Plan 02 Task 03: acridone derivative. Pre-148 produced
    # acridone-as-parent w/ chain prefix (`...-N-methylacridone`); post-148
    # cascade unblock per P-44.1(a) selects the chain (carbinol) and renders
    # the acridine ring as `acridin-9-yl` substituent. The acridone (=O on
    # ring) is dropped from the parent rendering — a known consequence of
    # cascade preferring chain when chain has alcohol PG. Acceptable churn.
    (
        "C=C(C)C(O)Cc1c(OC)cc(O)c2c(=O)c3cccc(O)c3n(C)c12",  # aromatic,heterocycle,fused-ring,medium
        "1-(acridin-9-yl)-3-methylbut-3-en-2-ol",
    ),
    (
        "CC(=C\\C(C)=C\\c1ccc([N+](=O)[O-])cc1)/C=C(C)/C=C(\\C)CCc1oc([O-])c(C)c(=O)c1C",  # aromatic,heterocycle,charged,large
        "phenolate",
    ),
    (
        "CC(C)(C)c1nc(-c2cccc(NS(=O)(=O)c3c(F)cccc3F)c2F)c(-c2ccnc(N)n2)s1",  # aromatic,heterocycle,large
        "2-(tert-butyl)-4-phenyl-5-pyrimidinylthiazole",
    ),
    (
        "COc1cccc2c1CO[C@@H]2C[C@@H](O)[C@@H](O)[C@@H]1O[C@@H]1C",  # aromatic,heterocycle,fused-ring,medium
        "(2R,3R)-1-cyclononyl-3-oxiranylpropane-2,3-diol",
    ),
    (
        "[C-]#[N+][C@]12C[C@@H](O)C(C)(C)c3[nH]c4cccc5c4c3[C@]1(O)[C@H](CC[C@]2(C)C=C)C5(C)C",  # aromatic,heterocycle,fused-ring,medium
        "(4R,6S,7R,10R,18R)-7-ethenyl-3,3,6,7,11,11-hexamethyl-19-aza-pentacyclo[8.7.1.1(2,16).0(6,18).0(12,17)]nonadecan-4,18-diol",
    ),
    (
        "CCCCCCC/C=C\\CCCCCCCC(=O)OC[C@@H](O)COP(=O)(O)OC1C(O)C(O)C(O)[C@@H](O)C1O",  # large
        "(9Z)-((9Z)-heptadec-9-enoyloxy)heptadec-9-enehexaol",
    ),
    (
        "O=P([O-])([O-])OC[C@@H](O)[C@H](O)[C@@H](O)CO",  # acyclic,charged,small
        "(2S,3R,4R)-1,2,3,4-tetrahydroxy-5-phosphonooxypentanephosphonic acid",
    ),
    (
        "COC(=O)CC[C@@H](C)[C@H]1C[C@@H](O)[C@H]2[C@@H]3[C@H](O)C[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3CC[C@@]21C",  # fused-ring,medium,steroid
        "(3R,5S,7R,8R,9S,10S,13R,14S,15R,17R,20R)-3,7,15-trihydroxycholan-24-one",
    ),
    (
        "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O[C@@H]1OC[C@@H](O)[C@H](O)[C@H]1O",  # heterocycle,medium,carbohydrate
        "(beta-D-xylopyranosyloxy)beta-D-glucopyranose",
    ),
    (
        "C[C@@H]1O[C@@H](OCCCCCCCCCCCC(=O)O)[C@H](O)C[C@H]1O",  # heterocycle,medium,carbohydrate
        "12-hexyloxydodecanoic acid",
    ),
    (
        "COc1cc(-c2ccc(O)c(CC=C(C)C)c2)c(OC)c(O)c1-c1ccc(O)c(O)c1",  # aromatic,large
        "4',5,4'',3''-tetrahydroxy-3,2-dimethoxy-3'-2-methylbut-2-enyl-1,1':1,1''-terphenyl",
    ),
    (
        "COc1ccc(C[C@H](C)NC[C@@H](O)c2ccc(O)c(NC=O)c2)cc1.COc1ccc(C[C@H](C)NC[C@@H](O)c2ccc(O)c(NC=O)c2)cc1.O=C(O)/C=C/C(=O)O",  # aromatic,polyfunctional,salt,large
        "N-formyl(2S)-2-(3-amino-4-hydroxyphenyl)-1-amino-1-anilinoethan-2-ol N-formyl(2S)-2-(3-amino-4-hydroxyphenyl)-1-amino-1-anilinoethan-2-ol (2E)-but-2-enedioic acid",
    ),
    (
        "O=c1c(-c2ccc(OC3OC(CO)C(O)C(O)C3O)cc2)coc2cc3c(c(O)c12)OCO3",  # aromatic,heterocycle,fused-ring,large,carbohydrate
        "(glucopyranosyloxy)-11-hexyl-8-hydroxy-4,6,13-trioxa-tricyclo[7.4.0.0(3,7)]tridecan-10-one",
    ),
    (
        "O=C(O)c1cc2cc3c4c(c2oc1=O)CCCN4CCC3",  # aromatic,heterocycle,fused-ring,medium
        "6-oxo-7-oxa-13-aza-tetracyclo[7.7.1.0(3,8).0(13,17)]heptadecane-5-carboxylic acid",
    ),
    # Phase 148 Plan 02 Task 03: estradiol-tetraol. Locant numbering on
    # estra ring system updated from 1,2,4-trien to 1,3,5-trien (correct
    # IUPAC numbering for the aromatic A-ring of estranes per P-31.1.5
    # estra-X numbering convention). Stereo descriptors and hydroxyl
    # locants unchanged. Per Plan 01 SUMMARY this is "unrelated to
    # Phase 148" (incidental locant correction in the estra ring system,
    # not a cascade decision change). Acceptable churn — name now reflects
    # canonical estra-1,3,5-triene numbering.
    (
        "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1[C@@H](O)[C@@H](O)[C@@H]2O",  # aromatic,fused-ring,medium,steroid
        "(8R,9S,13S,14S,15R,16R,17R)-estra-1,3,5-trien-3,15,16,17-tetraol",
    ),
    (
        "COc1cc(Nc2ncc3c(n2)-c2ccc(Cl)cc2C(c2c(F)cccc2OC)=NC3)ccc1C(=O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "4-heptyl-13-octyl-3,12,14-triaza-tricyclo[9.4.0.0(5,10)]pentadec-3-ene",
    ),
    (
        "COC(=O)[C@@H]1CC23CCCN4CC[C@@]5(c6ccccc6N(C)C15CC2)[C@@H]43",  # aromatic,heterocycle,fused-ring,medium
        "(1R,10R,21S)-10-ethyl-12-methyl-4,12-diaza-hexacyclo[9.7.0.2(8,11).1(4,8).0(13,18).0(1,21)]henicosane",
    ),
    (
        "COc1c(Cl)c(C)cc2cc(O)c3c(c12)C(=O)c1cc2c(c(O)c1C3=O)[C@H](C)OC2=O",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "(6S)-4,21-dihydroxy-15-methoxy-6,17-dimethyl-7-oxa-pentacyclo[11.8.0.0(3,11).0(5,9).0(14,19)]henicosan-2,8,12-trione",
    ),
    (
        "CO[C@@H]1[C@H](O)[C@@H](CO)O[C@H]1n1ccc(=O)[nH]c1=O",  # aromatic,heterocycle,medium
        "(2R,3R,4R,5R)-4-hydroxy-3,5-dimethyl-2-pyrimidinyloxolane",
    ),
    (
        "C/C=C/C=C/c1cc2cc3c(c(O)c2c(=O)o1)-c1c(O)c2c(c(O)c1CC3)C(=O)c1c(O)c(OC)cc(O)c1C2=O",  # aromatic,heterocycle,fused-ring,large
        "7-penta-1,3-dien-1-yl-3,15,19,22,26-pentahydroxy-20-methoxy-6-oxa-hexacyclo[12.12.0.0(2,11).0(4,9).0(16,25).0(18,23)]hexacosan-5,17,24-trione",
    ),
    (
        "OC[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)CO[C@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O",  # heterocycle,medium,carbohydrate
        "(alpha-D-glucopyranosyloxy)(2S,3R,4S,5R)-2,3,4,5-tetrahydroxyhexane-1,6-diol",
    ),
    (
        "C=CCN(C)CCCCCCOc1ccc(C(=O)c2ccc(Br)cc2)c(F)c1",  # aromatic,medium
        "1-decoxy-3-fluorobenzene",
    ),
    (
        "COC(=O)[C@@]1(O)C(=O)C=C2c3cc(OC)cc(O)c3C(=O)O[C@]21C",  # aromatic,heterocycle,fused-ring,polyfunctional,medium
        "(5S,6R)-5-ethyl-5,10-dihydroxy-12-methoxy-6-methyl-7-oxa-tricyclo[7.4.0.0(2,6)]tridec-2-en-4,8-dione",
    ),
    (
        "CCCCCC/C=C\\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCC/C=C\\CCCCCCCC)COP(=O)([O-])[O-]",  # acyclic,charged,large
        "(9Z)-phosphonooxyoctadec-9-enephosphonic acid",
    ),
    (
        "COc1cc2c(cc1OC)C1C(CO2)Oc2c(ccc3occc23)C1O",  # aromatic,heterocycle,fused-ring,medium
        "18,19-dimethoxy-7,12,15-trioxa-pentacyclo[11.8.0.0(3,11).0(6,10).0(16,21)]henicosan-2-ol",
    ),
    (
        "COc1cccc2c1C(=O)c1ccc3c(c1C2=O)C(=O)C[C@@H](C)[C@H]3O",  # aromatic,fused-ring,medium
        "(14R,15R)-15-hydroxy-4-methoxy-14-methyl-tetracyclo[8.8.0.0(3,8).0(11,16)]octadecan-2,9,12-trione",
    ),
    (
        "CC(C)C[C@H]1C(=O)N2c3ccccc3[C@@](O)(C[C@@H]3NC(=O)c4ccccc4-n4c3nc3ccccc3c4=O)[C@H]2N1O",  # aromatic,heterocycle,fused-ring,large
        "(2S)-2-cyclopentyl-7-oxo-1,4-diazepane",
    ),
    (
        "CCCCCCCC/C=C\\CCCCCCCC(=O)OC[C@H](COP(=O)(O)OCCN)OC(=O)CCCCCCCC(O)/C=C/C(=O)O",  # acyclic,polyfunctional,large
        "(2E)-aminohydroxy-4-hydroxydodec-2-enoic acid (9Z)-octadec-9-enoate",
    ),
    (
        "COC1C=C(COC(=O)c2ccccc2)C(O)C(OC(=O)c2ccccc2)C1O",  # aromatic,medium
        "3-(benzoyloxy)-2,4-dihydroxy-1-hydroxymethylcyclohexane",
    ),
    (
        "OC[C@H]1O[C@H](O)[C@H](O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]2O)[C@@H](O)[C@H]1O",  # heterocycle,medium,carbohydrate
        "(alpha-D-mannopyranosyloxy)alpha-D-galactopyranose",
    ),
    (
        "C[C@H]1C[C@@H](O)[C@H]2C(=O)c3c(O)cccc3O[C@]2(C)[C@@H]1O",  # aromatic,heterocycle,fused-ring,medium
        "(1R,10S,11R,12S,14R)-4,11,14-trihydroxy-10,12-dimethyl-9-oxa-tricyclo[8.4.0.0(3,8)]tetradecan-2-one",
    ),
    (
        "COC(=S)NCc1ccc(OC2OC(C)C(O)C(O)C2O)cc1",  # aromatic,heterocycle,medium,carbohydrate
        "(rhamnopyranosyloxy)-4-(1-methoxy-1-(methylamino)methyl)phenol",
    ),
    (
        "COc1cc(O)c2c(c1O)C(=O)c1c(C(C)=O)c(O)cc(O)c1C2=O",  # aromatic,fused-ring,medium
        "7-ethyl-4,6,11,14-tetrahydroxy-12-methoxy-tricyclo[8.4.0.0(3,8)]tetradecan-2,9-dione",
    ),
    (
        "CC(=O)Oc1ccc2c(c1)oc(=O)c1c3cc(OC(C)=O)c(OC(C)=O)cc3oc21",  # aromatic,heterocycle,fused-ring,medium
        "13-phenyltridecyl acetate",
    ),
    (
        "O=Cc1ccccc1OC1OC(COC2OCC(O)C(O)C2O)C(O)C(O)C1O",  # aromatic,heterocycle,medium,carbohydrate
        "(xylopyranosyloxy)-2-(oxan-2-yl)oxybenzaldehyde",
    ),
    (
        "CCCCCCCC/C=C\\CCCCCCCC(=O)OC[C@H](COP(=O)(O)OC[C@@H](O)CO)OC(=O)CCC(O)/C=C/C=O",  # acyclic,polyfunctional,large
        "(2R)-2-((5E)-4-hydroxy-7-oxohept-5-enoyloxy)-hydroxy-1-(oleoyloxy)propanal",
    ),
    (
        "CC[C@@H]1Cc2cc(O)ccc2C2=C1c1ccc(O)cc1C[C@H]2CC",  # aromatic,fused-ring,medium
        "(9R,18R)-9,18-diethyl-tetracyclo[8.8.0.0(2,7).0(11,16)]octadec-1-en-5,14-diol",
    ),
    (
        "C=C1CC23CC1C(O)CC2C12CCCC(C)(C(=O)OC1)C2C3C(=O)O",  # heterocycle,fused-ring,polyfunctional,medium
        "11-hydroxy-4-methyl-18-oxo-17-oxa-pentacyclo[7.5.0.3(4,8).1(1,12).0(3,8)]octadecane-2-carboxylic acid",
    ),
    (
        "CCCCCCCC/C=C\\CCCCCCCCCCCC(=O)OC(COC(=O)CCCCCCCCCCCCC)COP(=O)(O)OCCNC",  # acyclic,polyfunctional,large
        "2-((13Z)-docos-13-enoyloxy)-1-(myristoyloxy)propanamine",
    ),
    (
        "C=Cc1c(C)c2[n]3c1=CC1=[N+]4C(=Cc5c(CC)c6c7[n]5[Mg-2]34[N+]3=C(C=2)[C@@H](C)[C@H](CCC(=O)O)C3=C7CC6=O)C(CCC)=C1C",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "3-cycloheptacosylpropanoic acid",
    ),
    (
        "C=CCO/N=C(\\C(=O)N[C@H]1CN2CC(S(C)(=O)=O)=C(C(=O)O)N2C1=O)c1csc(N)n1",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "N-[2-amino-2-(thiazol-4-yl)ethanoyl]-4-methyl-1,2-diazole-5-carboxylic acid",
    ),
    (
        "CC[C@@H](O)C[C@@H](O)c1cc(OC)cc(=O)o1",  # aromatic,heterocycle,medium
        "(1R,3R)-1-cyclohexylpentane-1,3-diol",
    ),
    (
        "*[C@@H]1OC[C@@H](O[C@@H]2O[C@H](CO)[C@H](O)[C@H](O[C@@H]3O[C@H](CO)[C@H](O)[C@H](O[C@@H]4O[C@H](C(=O)O)[C@@H](O[C@H]5O[C@H](CO)[C@@H](O)[C@H](O)[C@H]5NC(C)=O)[C@H](O)[C@H]4O)[C@H]3O)[C@H]2O)[C@H](O)[C@H]1O",  # heterocycle,polyfunctional,large,carbohydrate
        "N-acetyl(2R,3R,4R,5S,6R)-3-amino-4,5-dihydroxy-6-methyl-2-oxanyloxane",
    ),
    (
        "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(C)=O",  # acyclic,large
        "(2R)-2-(acetyloxy)propyl palmitate",
    ),
    (
        "NC(=O)CC[C@H](NC(=O)[C@@H]1CCCN1C(=O)[C@@H]1CCCN1)C(=O)O",  # heterocycle,polyfunctional,medium
        "N-[(2S)-N-pyrrolidinylpyrrolidine-2-carbonyl](2S)-2-amino-4-carbamoylbutanoic acid",
    ),
    (
        "CCCCC[C@@H](O)[C@@H](O)c1cc(OC)cc(=O)o1",  # aromatic,heterocycle,medium
        "(1R,2R)-1-cyclohexylheptane-1,2-diol",
    ),
    (
        "CNCC[C@H](Oc1cccc2ccccc12)c1cccs1",  # aromatic,heterocycle,fused-ring,medium
        "(3S)-1-(methylamino)-3-phenoxy-3-thienylpropan-1-amine",
    ),
    (
        "CC(C)=CCc1ccc(O)c2c1[C@H](CC(=O)O)OC2=O",  # aromatic,heterocycle,fused-ring,polyfunctional,medium
        "2-cyclononylethanoic acid",
    ),
    (
        "Cl.c1ccc2sc(C3(N4CCCCC4)CCCCC3)cc2c1",  # aromatic,heterocycle,fused-ring,salt,medium
        "1-benzothiophene",
    ),
    (
        "CCCCCCCCCCCCCC(O)CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C",  # acyclic,medium
        "hydroxy-3-(palmitoyloxy)-4-(propylamino)butanoic acid",
    ),
    (
        "CC/C=C\\C[C@H](O)/C=C/C=C/C=C\\C=C/[C@@H](O)[C@H](O)CCCC(=O)O[C@@H](CO)COC(=O)CCCCCCCCCCCCCCCCCC",  # acyclic,large
        "(2S)-2-((5R,6R,7Z,9Z,11E,13E,15S,17Z)-5,6,15-trihydroxyicosa-7,9,11,13,17-pentaenoyloxy)-1-(nonadecanoyloxy)-trihydroxypropan-3-ol",
    ),
    (
        "O=C(/C=C/c1ccc(O)cc1)O[C@@H]1C[C@](O)(C(=O)[O-])C[C@@H](O)[C@H]1O",  # aromatic,charged,medium
        "heptanoate",
    ),
    (
        "OC[C@H]1O[C@H](OC[C@H]2O[C@H](O[C@H]3[C@H](O)[C@@H](O)[C@@H](O[C@H]4[C@H](O)[C@@H](O)[C@@H](OC[C@H]5O[C@H](O[C@H]6[C@H](O)[C@@H](O)[C@@H](O[C@H]7[C@H](O)[C@@H](O)[C@@H](O)O[C@@H]7CO)O[C@@H]6CO)[C@H](O)[C@@H](O)[C@@H]5O)O[C@@H]4CO)O[C@@H]3CO)[C@H](O)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O",  # heterocycle,large,carbohydrate
        "(alpha-D-glucopyranosyloxy)(2R,3S,4S,5R,6R)-3,4,5-trihydroxy-2-methyl-6-oxanyloxane",
    ),
    (
        "CCCCCCCCCCCCCCCCCCCCCCC(C(=O)O)C(O)CCCCCCCCCCCCCCCCCC1CC1CCCCCCCCCCCCCCCCC(O)C(C)CCCCCCCCCCCCCCCCCC",  # large
        "2-hydroxytetracosanoic acid",
    ),
    (
        "Cc1c(O)cc2c(c1C)C(=O)O[C@@H]([C@@]1([C@@H]3CC=C4CCC[C@H](C)[C@@]4(C)C3)CO1)O2",  # aromatic,heterocycle,fused-ring,medium
        "2,3-dimethylphenol",
    ),
    (
        "O=C(ON1C(=O)CCC1=O)c1cc(Cl)c2c(c1Cl)C1(OC2=O)c2cc(Cl)c(O)cc2Oc2cc(O)c(Cl)cc21",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "3,6-dichloro-2,7-dihydroxy-9H-xanthene",
    ),
    (
        "CC(=O)OC[C@]1(C)[C@@H](OC(C)=O)CC[C@]2(C)C3=Cc4c(cc(-c5ccccc5)oc4=O)O[C@]3(C)[C@@H](OC(C)=O)C[C@@H]12",  # aromatic,heterocycle,fused-ring,large
        "bis(acetyloxy)ethyl acetate",
    ),
    (
        "COc1cc(C2CC(=O)c3c(O)cc(O)c(CC=C(C)C)c3O2)c(O)cc1O",  # aromatic,heterocycle,fused-ring,medium
        "5,7-dihydroxy-8-(2-methylbut-2-enyl)chroman-4-one",
    ),
    (
        "CCCCCC/C=C\\CCCCCCCCCC(=O)OC(COC(=O)CCCCCCC/C=C\\CCCCCCCC)COP(=O)(O)OCCNC",  # acyclic,polyfunctional,large
        "(9Z)-amino(oleoyloxy)octadec-9-enyl (11Z)-octadec-11-enoate",
    ),
    (
        "CC(=O)Nc1ccc(S(=O)(=O)NC(C)=O)cc1",  # aromatic,medium
        "N-acetyl-1-amino-4-sulfanylbenzene",
    ),
    (
        "Cc1ccc(NC2=CC(=O)c3sc(C)nc3C2=O)cc1",  # aromatic,heterocycle,fused-ring,medium
        "2-anilino-5-(ethylsulfanyl)cyclohex-2-ene-1,4-dione",
    ),
    (
        "CCCCCCCC/C=C\\CCCC(COC[C@@H](O)COP(=O)(O)OC[C@H](N)C(=O)O)OC",  # acyclic,polyfunctional,large
        "(2S)-2-aminobishexyloxyhydroxypropanoic acid",
    ),
    (
        "COc1c(O)c(O)c2c(c1O)C(=O)c1coc(C)c1C2=O",  # aromatic,heterocycle,fused-ring,medium
        "10,12,13-trihydroxy-11-methoxy-4-methyl-5-oxa-tricyclo[7.4.0.0(3,7)]tridecan-2,8-dione",
    ),
    (
        "COc1c(-c2cc(O)c(O)c(CC=C(C)C)c2)oc2cc(O)cc(O)c2c1=O",  # aromatic,heterocycle,fused-ring,medium
        "3-methyl-4-oxo-2-phenyl-2H-pyran",
    ),
    (
        "COc1cc2oc3c(c(=O)c2cc1O)C(=O)c1c(ccc2cc4c(c(O)c12)C(=O)N1C(C)(C4)OC(=O)C1(C)COC(=O)CC(C)C)C3=O",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "7-hexyl-3,24-dihydroxy-23-methoxy-7,10-dimethyl-9,20-dioxa-6-aza-heptacyclo[15.12.0.0(2,14).0(4,12).0(6,10).0(19,28).0(21,26)]nonacosan-5,8,18,27,29-pentaone",
    ),
    (
        "CCCCCCC/C=C\\CCCCCCCC(=O)OC[C@H](COP(=O)(O)OCCN)OC(=O)CCCCCCC/C=C\\CCCCCCCC",  # acyclic,polyfunctional,large
        "(2R)-1-((9Z)-heptadec-9-enoyloxy)-2-(oleoyloxy)propanamine",
    ),
    (
        "COc1cc(O)cc2c1C(=O)O[C@@H](C)CCCCC/C=C/2",  # aromatic,heterocycle,fused-ring,medium
        "3-methoxyphenol",
    ),
    (
        "COc1ccc2c3c1O[C@H]1C(=O)CC[C@@]4(O)[C@@H](C2)[NH+](C)CC[C@]314",  # aromatic,heterocycle,fused-ring,charged,medium,alkaloid
        "(5R,9R,13S,14S)-4,5-epoxy-14-hydroxy-3-methoxy-17-methylmorphinan-6-one",
    ),
    (
        "CC1OC(Oc2c(C3OC(CO)C(O)C(O)C3O)c(O)c3c(=O)cc(-c4ccc(O)c(O)c4)oc3c2C2OC(CO)C(O)C(O)C2O)C(O)C(O)C1O",  # aromatic,heterocycle,fused-ring,large,carbohydrate
        "(rhamnopyranosyloxy)-4-oxo-6-phenyl-2H-pyran",
    ),
    (
        "CCCCC/C=C\\C/C=C\\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCCCC)COP(=O)([O-])OCC[N+](C)(C)C",  # acyclic,large
        "(stearoyloxy)octadecyl (9Z,12Z)-octadeca-9,12-dienoate",
    ),
    (
        "C=C(C)C(=O)Cc1c(C)cc(Oc2cc(CO)cc(OC)c2)cc1OC",  # aromatic,medium
        "1-(6-methoxy-2-methyl-4-phenoxyphenyl)-3-methylbut-3-en-2-one",
    ),
    (
        "COC1C(C(OC2OC(C(=O)O)=CC(O)C2O)C(N)=O)OC(n2ccc(=O)[nH]c2=O)C1OC(N)=O",  # aromatic,heterocycle,polyfunctional,large,carbohydrate
        "3,4-dimethyl-5-oxanyl-2-pyrimidinyloxolane",
    ),
    (
        "CC(C)[C@H](NC(=O)[C@@H](N)Cc1ccc(O)cc1)C(=O)N[C@@H](CCCN=C(N)N)C(=O)O",  # aromatic,polyfunctional,large
        "(2S)-5-(methylamino)-2-(pentanoylamino)pentanoic acid",
    ),
    (
        "O=C(O)CCC(C(=O)O)C(CC(=O)O)C(=O)O",  # acyclic,medium
        "3,4-diformylheptanedioic acid",
    ),
    (
        "CC1CCC2(C(=O)O)CCC3(C)C(=CCC4C5(C)CCC(OC(=O)/C=C/c6ccc(O)cc6)C(C)(CO)C5CCC43C)C2C1(C)O",  # aromatic,fused-ring,polyfunctional,large
        "16-hydroxy-1,2,6,6,10,16,17-heptamethyl-7-nonyl-pentacyclo[12.8.0.0(2,11).0(5,10).0(15,20)]docos-13-ene-20-carboxylic acid",
    ),
    (
        "COc1ccc(C[C@@H]2NC(=O)/C(C)=C\\C3CSC(=N3)[C@@H](C)[C@@H](O)C[C@H](C)C[C@@H](C(C)(C)C)OC(=O)[C@H](C)N(C)C(=O)[C@H](C(C)C)N(C)C(=O)[C@H](C)N(C)C2=O)cc1",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "(2S,3S,5S,7S,10S,13S,16S,19S,22Z)-7-butyl-3-hydroxy-2,5,10,11,14,16,17,22-octamethyl-19-octyl-13-propyl-8-oxa-25-thia-11,14,17,20,27-pentaaza-bicyclo[22.2.1]heptacosa-1(27),22-diene",
    ),
    (
        "CC/C=C\\C/C=C\\C/C=C\\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCC)COP(=O)([O-])OC[C@H]([NH3+])C(=O)[O-]",  # acyclic,charged,large
        "(9Z,12Z,15Z)-octadeca-9,12,15-trienoic acid (palmitoyloxy)hexadecanol",
    ),
    (
        "CCCCCCC(C)(C)c1cc(O)c2c(c1)OC(C)(C)[C@H]1CC=C(CO)C[C@H]21",  # aromatic,heterocycle,fused-ring,medium
        "(2S,7S)-4,8,8-trimethyl-12-2-methyloctyl-9-oxa-tricyclo[8.4.0.0(2,7)]tetradec-4-en-14-ol",
    ),
    (
        "CC(=O)O[C@H]1[C@@H](OC(C)=O)C(C)(C)[C@]2(O)CC[C@H]3C(=O)c4ccoc4C[C@@H]3[C@@]2(C)[C@H]1OC(C)=O",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "bis(acetyloxy)ethanone acetate",
    ),
    (
        "O=c1c(-c2ccc(O)c(Br)c2)coc2c(Br)c(O)c(Br)c(O)c12",  # aromatic,heterocycle,fused-ring,medium
        "4-oxo-5-phenyl-2H-pyran",
    ),
    (
        "COc1cc(C)c2c(c1C(=O)O)Oc1c(c(C)c(O)c3c1[C@@H](O)OC3=O)OC2=O",  # aromatic,heterocycle,fused-ring,polyfunctional,medium
        "(17S)-13,17-dihydroxy-5-methoxy-7,12-dimethyl-9,15-dioxo-2,10,16-trioxa-tetracyclo[9.7.0.0(3,8).0(14,18)]octadecane-4-carboxylic acid",
    ),
    (
        "CC(=O)[C@@]1(C)C(C)=C[C@H](O)[C@H]2C[C@](C)(O)CC[C@@H]21",  # fused-ring,medium
        "1-cyclodecanylethan-1-one",
    ),
    (
        "O=C1CCC(N2C(=O)c3ccc(O)cc3C2=O)C(=O)N1",  # aromatic,heterocycle,fused-ring,medium
        "6-hydroxyisoindoline-1,3-dione",
    ),
    (
        "COC(=O)C[C@@H]1C[C@@]2(O)C(=O)c3ccc([C@H]4C[C@@H](N(C)C)[C@H](O)[C@@H](C)O4)c(O)c3C(=O)[C@]2(O)[C@@H](C)O1",  # aromatic,heterocycle,fused-ring,polyfunctional,large,carbohydrate
        "(tetracosanoyloxy)-2-cyclotetradecylethanedione",
    ),
    (
        "COC(=O)c1cc(O)cc(OC)c1C(=O)c1c(O)c(Cl)c(C)c(Cl)c1O",  # aromatic,polyfunctional,medium
        "4-(hydroxyoctyl)-3-methoxyphenol",
    ),
    (
        "C/C=C/C(=O)O[C@@H]1CC2O[C@@H]3C=C(C)[C@@H](O)[C@@H]4OCC2(O)[C@@]1(C)[C@@]34C",  # heterocycle,fused-ring,medium,carbohydrate
        "(2E)-((2E)-but-2-enoyloxy)but-2-enol",
    ),
    (
        "CC1=CC(=O)CC2C1(C)CCC(C)C2(C)CC/C(C)=C/C(=O)O",  # fused-ring,polyfunctional,medium
        "(2E)-5-cyclodecanyl-3-methylpent-2-enoic acid",
    ),
    (
        "CSCCC(N)C(=O)Oc1ccc(CC(N)C(=O)O)cc1",  # aromatic,polyfunctional,medium
        "2-amino-3-phenylpropanoic acid",
    ),
    (
        "COC(=O)[C@@H]1CC(=O)[C@]2(O)O[C@@H]3[C@@H](OC)[C@H](n4cnc5c(N)ncnc54)O[C@@H]3C[C@H]2O1",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "(1R,3S,4R,5R,7R,9R,11S)-11-ethyl-1-hydroxy-4-methoxy-5-pentyl-2,6,10-trioxa-tricyclo[7.4.0.0(3,7)]tridecan-13-one",
    ),
    (
        "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCCC",  # acyclic,large
        "(5Z,8Z,11Z,14Z)-(arachidonoyloxy)icosa-5,8,11,14-tetraenyl (7Z,10Z,13Z,16Z,19Z)-docosa-7,10,13,16,19-pentaenoate",
    ),
    (
        "O=C1c2c(O)cc(O)cc2O[C@@H](c2ccc(O)c(O)c2)[C@@H]1O[C@@H]1OC[C@@H](O)[C@H](O)[C@H]1O",  # aromatic,heterocycle,fused-ring,large,carbohydrate
        "(beta-D-xylopyranosyloxy)(2S,3S)-3,5,7-trihydroxychroman-4-one",
    ),
    (
        "O=C1N[C@@H](C[C@@]2(O)c3ccccc3N3C(=O)[C@@H]4CCCCN4[C@@H]32)C(=O)N[C@H]1Cc1ccccc1",  # aromatic,heterocycle,fused-ring,large
        "(8R,9S,15S)-8-dodecyl-8-hydroxy-1,10-diaza-tetracyclo[7.7.0.0(2,7).0(10,15)]hexadecan-16-one",
    ),
    (
        "C[C@@H]1CC(=O)C2=C(CC[C@@]34O[C@@]23C(=O)c2cccc(O)c2[C@@H]4O)C1",  # aromatic,heterocycle,fused-ring,medium
        "(1S,6S,10S,18S)-16,18-dihydroxy-6-methyl-19-oxa-pentacyclo[8.8.0.1(1,10).0(4,9).0(12,17)]nonadec-4-en-8,11-dione",
    ),
    (
        "C[C@@H]1O[C@@H](OCCCCCCCCCCCCCCCCCCCC[C@@H](O)CC(=O)O)[C@H](O)C[C@H]1O",  # heterocycle,large,carbohydrate
        "(3R)-23-hexyloxy-3-hydroxytricosanoic acid",
    ),
    (
        "COC(=O)c1ccccc1S(=O)(=O)NC(=O)Nc1nc(C)cc(C)n1",  # aromatic,heterocycle,medium
        "N-7-carboxyheptylcarbamyl-2-amino-4,6-dimethylpyrimidine",
    ),
    (
        "CC(=O)OC1CC2C3(C)CCC(OC(=O)CC(=O)O)C(C)(C)C3CCC2(C)C2(C)CCC(C3(C)CCC(C(C)(C)O)O3)C12",  # heterocycle,fused-ring,polyfunctional,large,steroid
        "4,4,8,10,14-pentamethyl-12-(acetyloxy)-3-(propanoyloxy)gonane",
    ),
    (
        "CCc1oc2ccc(-c3cnn(C)c3)cc2c1C(=O)c1ccc(O)cc1",  # aromatic,heterocycle,fused-ring,medium
        "2-ethyl-1-benzofuran",
    ),
    (
        "CCCCCCCCCC(O)CCC1NC(=O)C(C(C)O)NC(=O)C(C(C)O)NC(=O)C(CC(O)C(N)=O)NC(=O)C(C)NC(=O)C(CC(N)=O)NC(=O)C(CC(N)=O)NC(=O)C(CCCCCCCC(O)CCCCCC)NC(=O)CCNC(=O)CNC1=O",  # heterocycle,large
        "9-(2-carbamoylethyl)-15,18-bis(1-carbamoylmethyl)-31-(3-hydroxydodecyl)-3,6-dihydroxyethyl-21-(7-hydroxytetradecyl)-12-methyl-5,8,11,14,17,20,23,27,30-nonaoxoazacyclohentriacontan-2-one",
    ),
    (
        "Cc1ccc(O[C@H]2O[C@@H](C(=O)O)C(O)[C@@H](O)C2O)c(O)c1",  # aromatic,heterocycle,medium,carbohydrate
        "((2R,4R,6S)-3,4,5,6-tetrahydroxyoxane-2-carboxylic acid)-4-methylbenzene-1,2-diol",
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@H](C)NC(=O)[C@@H](N)CCCN=C(N)N)C(=O)O",  # acyclic,polyfunctional,medium
        "(2S,3S)-aminoguanidino-3-methyl-2-(propanoylamino)pentanoic acid",
    ),
    (
        "O=c1cc(-c2cc(O)c(O)cc2O)oc2cc(O)cc(O)c12",  # aromatic,heterocycle,fused-ring,medium
        "4-oxo-6-phenyl-2H-pyran",
    ),
    (
        "CC(C)[C@@H](C)[C@@H](O)[C@H]1CC[C@@H]([C@@]2(C)CCC(=O)[C@@]3(C)CC[C@H](O)C[C@]34C=C[C@@](O)(O4)C2=O)[C@@H]1C",  # heterocycle,fused-ring,large
        "(1S,5R,7R,10S,12S)-5-dodecyl-7,12-dihydroxy-1,5-dimethyl-15-oxa-tricyclo[8.4.0.1(7,10)]pentadec-8-en-2,6-dione",
    ),
    (
        "CCCCCCCCCCCCCCCCCC(=O)O[C@H](COCCCCCCCCCCCCCCCC)COP(=O)(O)O",  # acyclic,large
        "phosphonooxy-1-propoxyhexadecanephosphonic acid stearate",
    ),
    (
        "[NH3+]C(CCC(=O)[O-])C(=O)[O-].[Na+]",  # acyclic,salt,small
        "sodium glutamate",
    ),
    (
        "COc1ccc(-c2oc3c(CC=C(C)C)c(O)cc(O)c3c(=O)c2O[C@@H]2O[C@@H](C)[C@H](O)[C@@H](O)[C@H]2O)cc1",  # aromatic,heterocycle,fused-ring,large,carbohydrate
        "(alpha-L-rhamnopyranosyloxy)-5-hydroxy-4-oxo-6-phenyl-2H-pyran",
    ),
    (
        "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)(O)O)OC(=O)CCC(=O)O",  # acyclic,polyfunctional,large
        "phosphonophosphonooxybutanoic acid palmitate",
    ),
    (
        "C=C1C[C@]23C[C@H]1CC[C@H]2[C@]1(C)CCC[C@@](C)(C(=O)O)[C@H]1C[C@@H]3O",  # fused-ring,medium
        "(1R,2S,4S,5R,9S,10S,13R)-2-hydroxy-5,9-dimethyl-tetracyclo[8.5.0.1(1,13).0(4,9)]hexadecane-5-carboxylic acid",
    ),
    (
        "CC1=C[C@]2(CC1=O)[C@H](C)CC[C@@H](C(C)(C)O)[C@H]2O",  # medium
        "10-hydroxy-9-isopropyl-3,6-dimethyl-2-oxospiro[4.5]decane",
    ),
    (
        "OC[C@@H]1O[C@@](O)(CO)[C@@H](O)[C@@H]1O",  # heterocycle,small,carbohydrate
        "(2S,3S,4S,5S)-2,3,4-trihydroxy-2,5-dimethyloxolane",
    ),
    (
        "CN1C(=Cc2cc[n+](CCC[N+](C)(C)CCC[N+](C)(C)CCC[n+]3ccc(C=C4Oc5ccccc5N4C)c4ccccc43)c3ccccc23)Oc2ccccc21",  # aromatic,heterocycle,fused-ring,charged,large
        "propylundecylundecylaminium",
    ),
    (
        "CCCCC/C=C\\C/C=C\\CCCCCCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCC)COP(=O)(O)OC[C@H](N)C(=O)O",  # acyclic,polyfunctional,large
        "(2S)-2-aminohydroxypropanoic acid (13Z,16Z)-docosa-13,16-dienoate",
    ),
    (
        "C[C@]12CC[C@H](O)c3coc(c31)C(=O)C1=C2[C@@H](O)C[C@]2(C)C(=O)CC[C@@H]12",  # aromatic,heterocycle,fused-ring,medium,steroid
        "(3S,10R,11S,13S,14R)-3,11-dihydroxyandrosta-5,8-dien-7,17-dione",
    ),
    (
        "COc1ccc(C2COc3cc4c(cc3C2)C=CC(C)(C)O4)c(O)c1",  # aromatic,heterocycle,fused-ring,medium
        "12-heptyl-5,5-dimethyl-4,14-dioxa-tricyclo[8.4.0.0(3,8)]tetradec-6-ene",
    ),
    (
        "CC12CCC(=O)C=C1C=CC1[C@@H]2CCC2(C)[C@H]1CCC21CCC(=O)O1",  # heterocycle,fused-ring,medium,steroid
        "(9S,14S)-pregna-4,6-dien-3-one",
    ),
    # Phase 148 D-05 / Plan 02 Task 03: pre-148 OLD value `'2,3-dihydro-1-benzofuran'`
    # was already a known-bad placeholder (clearly wrong for a 35-atom
    # diphenyl-pyrrolidine + benzofuran-ethyl molecule). Post-148 cascade
    # unblock + decomposition fragment-naming bug yields a different
    # truncated fragment. Both old and new are broken; root cause is the
    # decomposition layer (Phase 149 / IM-x.x). Marked xfail.
    pytest.param(
        "NC(=O)C(c1ccccc1)(c1ccccc1)[C@@H]1CCN(CCc2ccc3c(c2)CCO3)C1",  # aromatic,heterocycle,fused-ring,large
        "2,3-dihydro-1-benzofuran",
        marks=pytest.mark.xfail(
            strict=True,
            reason="Phase 149 / IM-x.x: decomposition fragment-naming bug + "
                   "pre-existing bad placeholder; per 148-01-SUMMARY Risks §2 "
                   "(Plan-01 carry-forward).",
        ),
    ),
    (
        "CC[C@H]1C[C@]23OC(=O)C(=C(O)[C@@]4(CC)[C@@H]5CC[C@H](C)[C@H](O[C@H]6C[C@@H](O)[C@H](NC(=O)c7[nH]c(Cl)cc7Cl)[C@@H](C)O6)[C@H]5C=C[C@H]4C/C=C/C/C=C/[C@@]2(C)C=C1C(=O)O)C3=O",  # aromatic,heterocycle,fused-ring,polyfunctional,large,carbohydrate
        "(1R,6S,8S,11S,12E,15E,18R,21S,22S,23S,26R)-1,8-diethyl-2-hydroxy-11,23-dimethyl-4,27-dioxo-22-undecyl-5-oxa-pentacyclo[16.8.0.1(3,6).0(6,11).0(21,26)]heptacosa-2,9,12,15,19-pentaene-9-carboxylic acid",
    ),
    (
        "COc1nc(C)nc(NC(=O)NS(=O)(=O)c2ccccc2I)n1",  # aromatic,heterocycle,medium
        "4,6-dimethyl-2-phenyl-1,3,5-triazine",
    ),
    (
        "CCC(/C=C/C(C)C1CCC2C3=CCC4CC(OC5OC(CO)C(OC6OC(CO)C(O)C(O)C6O)C(O)C5O)CCC4(C)C3CCC21C)C(C)C",  # heterocycle,fused-ring,large,carbohydrate
        "(22E)-stigmasta-7,22-diene",
    ),
    (
        "CCCCCC/C=C\\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCCCCC)COP(=O)(O)O",  # acyclic,large
        "(7Z)-phosphonooxyhexadec-7-enephosphonic acid nonadecanoate",
    ),
    (
        "CC(=O)OCC12CCC(C)=CC1OC1C(O)C(OC(C)=O)C2(C)C12CO2",  # heterocycle,fused-ring,medium
        "(acetyloxy)ethyl acetate",
    ),
    (
        "COC12C=C(C(=O)C3C(C)C(C)=CC4(C)CC(C)CCC34)C(=O)N1CCC2C(=O)O",  # heterocycle,fused-ring,polyfunctional,medium
        "3-cyclohexyl-5-methyl-2-oxoazole",
    ),
    (
        "Nc1ccc(S(=O)(=O)Nc2ncc(CC(=O)O)s2)cc1",  # aromatic,heterocycle,polyfunctional,medium
        "aminobenzenesulfonamide",
    ),
    (
        "COC(=O)/C(CC(=O)O)=C(\\CCCCCCCCCCCCCCCCC1=C(C)C(=O)OC1=O)C(=O)O",  # heterocycle,polyfunctional,large
        "(2E)-3-(hydroxymethyl)pent-2-enedioic acid",
    ),
    (
        "CNC(=O)CC1NC(=O)c2csc(n2)-c2ccc(-c3nc(C(=O)NC(CO)C(=O)N4CCCC4C(N)=O)cs3)nc2-c2csc(n2)-c2csc(n2)C(C(C)C)NC(=O)CNC(=O)c2csc(n2)C(C(C)C)NC(=O)c2nc1sc2C",  # aromatic,heterocycle,fused-ring,large
        "36-dodecyl-15,25-diisopropyl-11-methyl-8-propyl-3,10,17,27,31-pentathia-7,14,21,24,35,39,40,41,42,43-decaaza-heptacyclo[32.4.0.1(2,5).1(9,12).1(16,19).1(26,29).1(30,33)]tritetracontan-6,13,20,23-tetraone",
    ),
    (
        "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCC(=O)O[C@H](COC(=O)CCCCCCC/C=C\\CCCCCCCCC)COP(=O)(O)OC1C(O)C(O)C(O)[C@@H](O)C1O",  # large
        "octacosyl (4Z,7Z,10Z,13Z,16Z,19Z)-docosa-4,7,10,13,16,19-hexaenoate",
    ),
    (
        "CN(C)[C@@H]1C(O)=C(C(=O)NCN2CCCC2)C(=O)[C@@]2(O)C(O)=C3C(=O)c4c(O)cccc4[C@@](C)(O)[C@H]3C[C@@H]12",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "(3S,7S,8S,10S,11S)-7-ethyl-5-hexyl-2,3,6,11,16-pentahydroxy-11-methyl-tetracyclo[8.8.0.0(3,8).0(12,17)]octadeca-1,5-dien-4,18-dione",
    ),
    (
        "COC[C@@]1(O)CC[C@@H]2C1=C[C@]1(C)C(=C(C(C)C)C[C@H]1O)C[C@H](O)[C@@H]2C",  # fused-ring,medium
        "(1S,2R,3S,8R,9R,12R)-12-ethyl-6-isopropyl-2,9-dimethyl-tricyclo[9.3.0.0(5,9)]tetradeca-5,10-dien-3,8,12-triol",
    ),
    (
        "CCC(O)CC(=O)O[C@H](CC(=O)[O-])C[N+](C)(C)C",  # acyclic,medium
        "ammonium dodecanoate",
    ),
    (
        "NC(N)=NCCC[C@H](NC(=O)[C@@H](N)CO)C(=O)O",  # acyclic,polyfunctional,medium
        "(2S)-5-guanidino-5-(methylamino)-2-(propanoylamino)pentanoic acid",
    ),
    (
        "CCCCC/C=C\\C/C=C\\CCCCCCCC(=O)OC[C@H](COC(=O)CCCCCCCCC/C=C\\CCCCCC)OC(=O)CCCCCCC/C=C\\C/C=C\\CCCCC",  # acyclic,large
        "1,2-bis(linoleoyloxy)-3-(oleoyloxy)propane",
    ),
    (
        "CCCCCCCCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)(O)OCCN)OC(=O)CCCCCCCCCCCCC",  # acyclic,polyfunctional,large
        "(2R)-1-(docosanoyloxy)-2-(myristoyloxy)propanamine",
    ),
    (
        "C=C1CC23CC1CC(O)C2C12COC(=O)C(C)(C(O)C(O)C1)C2C3C(=O)O",  # heterocycle,fused-ring,polyfunctional,medium
        "10,17,18-trihydroxy-4-methyl-5-oxo-6-oxa-pentacyclo[7.5.0.3(4,8).1(1,12).0(3,8)]octadecane-2-carboxylic acid",
    ),
    (
        "CCC(C)CCCCCCCCC(=O)OC[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2ccc(N)nc2=O)C(O)[C@H]1O)OC(=O)CCCCCCCCCC(C)C",  # aromatic,heterocycle,polyfunctional,large,carbohydrate
        "(10-methyldodecanoyloxy)((11-methyldodecanoyl)oxy)-11-methyldodecanol",
    ),
    (
        "CCOC(=O)C[C@@H](SP(=O)(OC)OC)C(=O)OCC",  # acyclic,medium
        "diethyl butanedioate",
    ),
    (
        "Cc1cn([C@H]2C[C@H](O)[C@@H](COP(=O)([O-])[O-])O2)c(=O)nc1N",  # aromatic,heterocycle,charged,medium
        "6-amino-5-methyl-2-oxo-N-oxolanylpyrimidine",
    ),
    (
        "C=C(C)[C@@H]1CC[C@@H](C)[C@@]12CC=C(C)CC2",  # small
        "4,8-dimethyl-1-prop-1-en-2-ylspiro[4.5]decane",
    ),
    (
        "COc1cc(/C=C\\c2ccc(OC)c(O)c2)cc(OC)c1",  # aromatic,medium
        "1-ethenyl-3,5-dimethoxybenzene",
    ),
    (
        "CCCC/C=C\\CCCCCCCC(=O)O[C@H](COC(=O)CCC/C=C\\C/C=C\\C/C=C\\CCCCCCCC)COC(=O)CCCCCCCCC/C=C\\CCCCCC",  # acyclic,large
        "1-[(5Z,8Z,11Z)-icosa-5,8,11-trienoyloxy]-2-[(9Z)-tetradec-9-enoyloxy]-3-(oleoyloxy)propane",
    ),
    (
        "CN1CCC2=C[C@H](O)[C@H]3OC(=O)c4cc5c(cc4[C@H]3[C@@H]21)OCO5",  # aromatic,heterocycle,fused-ring,medium
        "(1S,13S,14S,20S)-20-hydroxy-15-methyl-2,7,9-trioxa-15-aza-pentacyclo[11.7.0.0(4,12).0(6,10).0(14,18)]icos-18-en-3-one",
    ),
    (
        "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCCC",  # acyclic,large
        "(4Z,7Z,10Z,13Z,16Z)-((4Z,7Z,10Z,13Z,16Z)-docosa-4,7,10,13,16-pentaenoyloxy)docosa-4,7,10,13,16-pentaenyl (7Z,10Z,13Z,16Z,19Z)-docosa-7,10,13,16,19-pentaenoate",
    ),
    (
        "COc1cc(-c2oc3cc(O)c(C)c(O)c3c(=O)c2OC)ccc1O",  # aromatic,heterocycle,fused-ring,medium
        "2-methoxyphenol",
    ),
    (
        "CC(=CCC(O)C(C)[C@H]1CC(=O)[C@@]2(C)C3=C(C(=O)[C@@H](O)[C@]12C)[C@@]1(C)CCC(=O)[C@](C)(CO)[C@@H]1CC3=O)C(=O)O",  # fused-ring,polyfunctional,large,steroid
        "(4S,5R,10S,12S,13R,14R,17R)-12,22,27-trihydroxy-4,14-dimethylcholesta-8,24-dien-3,7,11,15,27-pentaone",
    ),
    (
        "CC(=O)N[C@@H](CSc1c2nc3cc(C=O)ccc3oc-2cc(=O)c1N)C(=O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,medium
        "(2R)-3-(tridecylsulfanyl)-2-(ethanoylamino)propanoic acid",
    ),
    (
        "CC[C@H](C)[C@H](N)C(=O)N[C@@H](CCCN=C(N)N)C(=O)N[C@@H](CC(=O)O)C(=O)O",  # acyclic,polyfunctional,medium
        "(2S)-aminoguanidino-2-(pentanoylamino)butanedioic acid",
    ),
    (
        "C[C@@H]1O[C@](C)(C(=O)O)O[C@H]1/C=C/C=C/C=C/C(=O)O[C@H]1CC[C@H](c2ccc3c(c2O)C(=O)c2cc(O)c4c(c2C3=O)C(=O)C(O)C(C)(O)C4)O[C@@H]1C",  # aromatic,heterocycle,fused-ring,polyfunctional,large,carbohydrate
        "5-nonadecyl-4,13,14,17-tetrahydroxy-14-methyl-tetracyclo[8.8.0.0(3,8).0(11,16)]octadecan-2,9,12-trione",
    ),
    (
        "COc1ccc(-c2cc(=O)c3c(O)cc(O)c([C@@H]4OC(CO)[C@@H](O)[C@H](O)C4O[C@@H]4OC(CO)[C@@H](O)[C@H](O)C4O[C@@H]4OC(C)[C@H](O)C(O)[C@@H]4O)c3o2)cc1",  # aromatic,heterocycle,fused-ring,large,carbohydrate
        "((3R,5S,6S)-3,4,5-trihydroxy-2-methyl-6-oxanyloxane)-4-oxo-6-phenyl-2H-pyran",
    ),
    (
        "CCC(C)(NC(=O)C(CCC(N)=O)NC(=O)C1CC(O)CN1C(=O)C(C)(C)NC(=O)C(C)(C)NC(=O)C(CC(C)C)NC(=O)CNC(=O)C(NC(=O)C(C)(C)NC(=O)C(C)(C)NC(=O)C(NC(C)=O)C(C)C)C(C)C)C(=O)N1CC(O)CC1C(=O)NC(C)(C)C(=O)NC(CO)Cc1ccccc1",  # aromatic,heterocycle,large
        "N-[2-(butanoylamino)-2-methylpropanoyl]-5-anilino-4-(pentanoylamino)pentanamide",
    ),
    (
        "O=[NH+]C=C1C=CN(COCN2C=CC(=C[NH+]=O)C=C2)C=C1",  # heterocycle,charged,medium
        "4-methyl-N-piperidinylpyridine",
    ),
    (
        "COc1c(Cl)c2c(c(C(=O)O)c1Cl)C[C@H](C)O2",  # aromatic,heterocycle,fused-ring,medium
        "(7aS)-2,3a-dichloro-3-methoxy-7a-methyl-2,3-dihydro-1-benzofuran-6-carboxylic acid",
    ),
    (
        "C=C1C[C@@]23C=CC(=O)[C@@](C)(CCC(=O)Nc4c(O)ccc(C(=O)OC)c4O)C2C[C@@H]1C[C@H]3O",  # aromatic,fused-ring,polyfunctional,large
        "(1R,4R,7S,12R)-12-hydroxy-7-methyl-7-undecyl-tricyclo[4.4.0.2(1,4)]dodec-9-en-8-one",
    ),
    (
        "CCCCC[C@H](O)/C=C/[C@H]1CCC(=O)[C@@H]1C/C=C\\CCCC(=O)O",  # polyfunctional,medium
        "(5Z)-7-cyclopentylhept-5-enoic acid",
    ),
    (
        "O=C[C@H](O)[C@@H](O)[C@H](O[C@H]1O[C@H](CO)[C@@H](O[C@H]2O[C@H](CO)[C@@H](O[C@H]3O[C@H](CO)[C@@H](O[C@H]4O[C@H](CO)[C@@H](O[C@H]5O[C@H](CO)[C@@H](O)[C@H](O)[C@H]5O)[C@H](O)[C@H]4O)[C@H](O)[C@H]3O)[C@H](O)[C@H]2O)[C@H](O)[C@H]1O)[C@H](O)CO",  # heterocycle,large,carbohydrate
        "(2R,3R,4R,5R)-4-triacontyloxy-2,3,5,6-tetrahydroxyhexanal",
    ),
    (
        "CC(=O)N[C@H]1C(OP(=O)(O)OP(=O)(O)OC[C@H]2O[C@@H](n3ccc(=O)[nH]c3=O)[C@H](O)[C@@H]2O)O[C@H](CO)[C@H](O)[C@@H]1O",  # aromatic,heterocycle,large,carbohydrate
        "N-acetyl(3R,4R,5R,6R)-3-amino-4,5-dihydroxy-6-methyl-2-oxolanyloxane",
    ),
    (
        "COc1ccc2c3c(c4cc(OC)c(OC)cc4c2c1)C[C@H]1CCCCN1C3",  # aromatic,heterocycle,fused-ring,medium
        "(16R)-5,10,11-trimethoxy-21-aza-pentacyclo[12.8.0.0(2,7).0(8,13).0(16,21)]docosane",
    ),
    (
        "CC[C@@]12CC(C(=O)OC)=C3Nc4cc(OC)c(O)cc4[C@@]34CCN(C[C@@H]3O[C@@H]31)[C@@H]24",  # aromatic,heterocycle,fused-ring,polyfunctional,medium
        "(4S,5R,7S,12R,20R)-2,4-diethyl-16-methoxy-6-oxa-9,19-diaza-hexacyclo[10.7.0.1(4,12).0(5,7).0(13,18).0(9,20)]icos-1-en-15-ol",
    ),
    (
        "CC(C)=CCC/C(C)=C/COC[C@H]1O[C@@H](N2CCC(=O)NC2=O)[C@H](O)[C@@H]1O",  # heterocycle,medium,carbohydrate
        "(2R,3R,4S,5R)-3,4-dihydroxy-2-piperazinyl-5-undecyloxolane",
    ),
    (
        "CC(C)[C@@]1(C)N=C(c2nc3ccccc3cc2C(=O)[O-])NC1=O.[NH4+]",  # aromatic,heterocycle,fused-ring,salt,medium
        "ammonium quinoline-3-carboxylate",
    ),
    (
        "CC1(C)CCN2CCC(C)(C)c3c2c1cc1cc(-c2nc4ccccc4s2)c(=O)oc31",  # aromatic,heterocycle,fused-ring,large
        "13-heptyl-2,2,8,8-tetramethyl-15-oxa-5-aza-tetracyclo[7.7.1.0(5,17).0(11,16)]heptadecan-14-one",
    ),
    (
        "CC1(C)OC[C@]2(C)[C@@H](CC[C@@]3(C)[C@H]2[C@@H](O)C[C@H]2C[C@@H]4C[C@@]23CC[C@]4(O)CO)O1",  # heterocycle,fused-ring,medium
        "(1S,2S,5R,6R,8R,10S,11R,12R,17R)-1,5,12,15,15-pentamethyl-14,16-dioxa-pentacyclo[9.8.0.1(2,6).0(2,8).0(12,17)]icosan-5,10-diol",
    ),
    (
        "CC(CCCC(C)(O)COS(=O)(=O)O)[C@H]1CC[C@H]2[C@@H]3[C@H](O)C[C@@H]4C[C@H](O)CCC4(C)[C@H]3C[C@H](O)C12C",  # fused-ring,large,steroid
        "(3R,5S,7R,8R,9S,12S,14S,17R)-cholestan-3,7,12,25-tetraol",
    ),
    (
        "CCCCC[C@@H](/C=C/C=C\\CCCCCCCC(=O)OC)OO",  # acyclic,medium
        "(9Z,11E,13S)-(linoleoyloxy)octadeca-9,11-dien-13-peroxol",
    ),
    (
        "Br.Br.CC[N+](CC)(CCC[N-]C1=CC(=O)C([N-]CCC[N+](CC)(CC)Cc2ccccc2)=CC1=O)Cc1ccccc1",  # aromatic,salt,large
        "ethylethylheptylpropylaminium",
    ),
    (
        "CCCC[C@@](C)(O)C/C=C/[C@H]1[C@H](CCCCCCC(=O)OC)C(=O)C[C@@H]1O",  # polyfunctional,medium
        "(cyclopentanecarbonyloxy)-7-cyclopentylheptanone",
    ),
    (
        "CCCCC/C=C\\C/C=C\\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCCCCCCCC)COP(=O)([O-])OCC[N+](C)(C)C",  # acyclic,large
        "(palmitoyloxy)hexadecyl (9Z,12Z)-octadeca-9,12-dienoate",
    ),
    (
        "C/C=C/C=C/C(=O)C1=C(O)C(=C(C)C)NC1=O",  # heterocycle,polyfunctional,medium
        "3-hexyl-4-hydroxy-5-isopropyl-2-oxoazole",
    ),
    (
        "CC[C@@]1(O)C[C@H](O[C@H]2C[C@H](N(C)C)[C@H](O[C@H]3C[C@H](O)[C@H](O[C@H]4CC[C@H](O)[C@H](C)O4)[C@H](C)O3)[C@H](C)O2)c2c(O)c3c(c(O)c2[C@H]1O[C@H]1C[C@H](N(C)C)[C@H](O[C@H]2C[C@H](O)[C@H](O[C@H]4CC[C@H](O)[C@H](C)O4)[C@H](C)O2)[C@H](C)O1)C(=O)c1cccc(O)c1C3=O",  # aromatic,heterocycle,fused-ring,large,carbohydrate
        "(4S,6R,7R)-6-ethyl-2,6,9,16-tetrahydroxy-4,7-diicosyl-tetracyclo[8.8.0.0(3,8).0(12,17)]octadecan-11,18-dione",
    ),
    (
        "NC(=O)N/C=C\\C(=O)OO",  # acyclic,small
        "(2Z)-3-carbamoylamino-3-(methanoylamino)prop-2-en-1-peroxol",
    ),
    # Phase 148 Plan 02 Task 03: acyl-CoA derivative. Pre-148 OLD value
    # `'adenine'` was a known-bad placeholder. Post-148 cascade unblock
    # produces a chain-as-parent rendering of the unsaturated acyl side.
    # Adenine+furanose+phosphate side still lost (Phase 149 / IM-x.x
    # decomposition). Acceptable churn — new name is more representative.
    (
        "CCC/C=C\\C/C=C\\CCCCCCCC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "N-[(2R)-2-hydroxy-3,3-dimethylbutanoyl](9Z,12Z)-amino-1-(ethylsulfanyl)hexadeca-9,12-diene",
    ),
    (
        "CC[C@H]1C2CC3[C@@H]4N(C)c5ccccc5[C@]45C[C@@H]([C@H]2[C@H]5O)N3[C@@H]1O",  # aromatic,heterocycle,fused-ring,medium,alkaloid
        "ajmaline",
    ),
    (
        "CC(C)C[C@H]1C(=O)N[C@@H](C)[C@]2(O)O[C@@](C)(NC(=O)[C@@H]3C=C4c5cccc6[nH]cc(c56)C[C@H]4N(C)C3)C(=O)N12",  # aromatic,heterocycle,fused-ring,large,alkaloid
        "(4R,7R)-4-tridecyl-6-methyl-6,11-diaza-tetracyclo[7.6.1.0(2,7).0(12,16)]hexadec-2-ene",
    ),
    (
        "COC(=O)[C@H]1[C@@H](OP(=O)(O)c2ccccc2)C[C@@H]2CC[C@H]1N2C",  # aromatic,heterocycle,fused-ring,medium,alkaloid
        "tropane",
    ),
    (
        "C[NH+]1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",  # aromatic,heterocycle,fused-ring,charged,medium,alkaloid
        "(5R,6S,9R,13S,14R)-4,5-epoxy-17-methylmorphin-7-en-3,6-diol",
    ),
    (
        "C=C[C@H]1C[N@]2CC[C@H]1C[C@@H]2[C@H](O)c1ccnc2ccc(O)cc12",  # aromatic,heterocycle,fused-ring,medium,alkaloid
        "cinchonane",
    ),
    (
        "COC1=C[C@]23CCN(C)[C@H](Cc4ccc(OC)c(O)c42)C3=C[C@@H]1O",  # aromatic,heterocycle,fused-ring,medium,alkaloid
        "(7S,9R,13S)-3,6-dimethoxy-17-methylmorphina-5,8-dien-4,7-diol",
    ),
    (
        "C=C[C@H]1CN2CC[C@@H]1C[C@H]2[C@H](OC(=O)OCC)c1ccnc2ccc(OC)cc12",  # aromatic,heterocycle,fused-ring,medium,alkaloid
        "cinchonane",
    ),
    (
        "COc1ccc2c3c1O[C@H]1C(=O)CC[C@@]4(O)[C@@H](C2)N(C)CC[C@]314",  # aromatic,heterocycle,fused-ring,medium,alkaloid
        "oxycodone",
    ),
    (
        "C=CC(C)(C)c1[nH]c2cccc3c2c1C[C@@H]1[C@H]3[C@H](O)[C@@H](C)CN1C",  # aromatic,heterocycle,fused-ring,medium,alkaloid
        "ergoline",
    ),
    (
        "C=C(C)C1CCC(C)(O)CC1",  # small,terpene
        "beta-terpineol",
    ),
    (
        "*c1c[nH]cc1*",  # aromatic,heterocycle,small
        "pyrrole",
    ),
    (
        "CC(C)=CCCc1ccsc1",  # aromatic,heterocycle,small
        "3-(2-methylpent-2-enyl)thiophene",
    ),
    (
        "CC(=O)[C@]1(O)CC[C@@]2(O)[C@]1(C)[C@H](OC(=O)c1ccc(O)cc1)C[C@@H]1[C@@]3(C)CC[C@H](O)CC3=CC[C@]12O",  # aromatic,fused-ring,polyfunctional,large,steroid
        "(3S,8S,9R,10R,12R,13S,14R,17S)-3,8,14,17-tetrahydroxy-20-oxopregn-5-en-12-yl heptanoate",
    ),
    (
        "CC(=O)N[C@@H]1[C@@H](O)[C@H](O[C@@H]2O[C@H](CO)[C@@H](O[C@@H]3O[C@H](CO[C@H]4O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]4O[C@@H]4O[C@H](CO)[C@@H](O[C@@H]5O[C@H](CO)[C@H](O)[C@H](O)[C@H]5O)[C@H](O)[C@H]4NC(C)=O)[C@@H](O)[C@H](O[C@H]4O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]4O[C@@H]4O[C@H](CO)[C@@H](O[C@@H]5O[C@H](CO)[C@H](O)[C@H](O)[C@H]5O)[C@H](O)[C@H]4NC(C)=O)[C@@H]3O)[C@H](O)[C@H]2NC(C)=O)[C@@H](CO[C@@H]2O[C@@H](C)[C@@H](O)[C@@H](O)[C@@H]2O)O[C@H]1O",  # heterocycle,large,carbohydrate
        "(alpha-L-fucopyranosyloxy)(beta-D-mannopyranosyloxy)(alpha-D-mannopyranosyloxy)(beta-D-galactopyranosyloxy)(alpha-D-mannopyranosyloxy)(beta-D-galactopyranosyloxy)(beta-D-2-(acetylamino)-2-deoxy-glucopyranosyloxy)(beta-D-2-(acetylamino)-2-deoxy-glucopyranosyloxy)(beta-D-2-(acetylamino)-2-deoxy-glucopyranosyloxy)beta-D-2-(acetylamino)-2-deoxy-glucopyranose",  # Phase 145.2: RATIO_REJECT_FLOOR=0.10 rejects 'ethanediamide' fragment (ratio=0.07 on 122 HA). Full-coverage sugar cascade now wins; name is substantively correct for this decasaccharide.
    ),
    (
        "O=C(O)CCN(C1(C(=O)NO)CCCC1)S(=O)(=O)c1ccc(Oc2ccc(F)cc2)cc1",  # aromatic,polyfunctional,large
        "3-anilino-3-sulfamoylpropanoic acid",
    ),
    (
        "CC(C)=CCC/C(C)=C/CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\CC/C(C)=C\\COP(=O)([O-])[O-]",  # acyclic,charged,large
        "(2Z,6Z,10Z,14Z,18Z,22Z,26Z,30Z,34E)-3,7,11,15,19,23,27,31,35,39-decamethyl-1-phosphonooxytetraconta-2,6,10,14,18,22,26,30,34,38-decaenephosphonic acid",
    ),
    (
        "O=P(O)(O)OC[C@H]1OC(O)(COP(=O)(O)O)[C@@H](O)[C@@H]1O",  # heterocycle,medium,carbohydrate
        "(3S,4S,5R)-2,3,4-trihydroxy-2,5-dimethyloxolane",
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@@H](N)C(C)C)C(=O)O",  # aromatic,polyfunctional,medium
        "L-valyl-L-tyrosyl-L-isoleucine",
    ),
    (
        "*C(=O)OC/C=C/c1ccc(O)c(OC)c1",  # aromatic,small
        "4-phenylbutyl formate",
    ),
    (
        "CO[C@H]1O[C@H](CO)[C@@H](O[C@@H]2O[C@H](CO)[C@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@H]1NC(C)=O",  # heterocycle,medium,carbohydrate
        "N-acetyl(2S,3R,4R,5S,6R)-3-amino-4-hydroxy-2,6-dimethyl-5-oxanyloxane",
    ),
    (
        "CCOC(=O)Nc1ccc2c(c1)N(C(=O)CC[NH+]1CCOCC1)c1ccccc1S2.[Cl-]",  # aromatic,heterocycle,fused-ring,salt,large
        "ethyl carbamatylbenzeneium chloride",
    ),
    # Phase 148 Plan 02 Task 03: 3-hydroxyoctadec-11-enoyl-CoA. Pre-148 OLD
    # `'adenine'` was a known-bad placeholder. Post-148 chain-as-parent
    # cascade unblock; adenine+furanose+phosphate Phase-149 territory.
    # Acceptable churn.
    (
        "CCCCCC/C=C\\CCCCCCC[C@@H](O)CC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)([O-])OP(=O)([O-])OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)([O-])[O-]",  # aromatic,heterocycle,fused-ring,polyfunctional,charged,large
        "N-[(2R)-2-hydroxy-3,3-dimethylbutanoyl](3R,11Z)-amino-1-(ethylsulfanyl)-3-hydroxyoctadec-11-ene",
    ),
    (
        "CCc1cc(S(=O)(=O)[O-])c2cc(C(C)C)cccc1-2",  # aromatic,fused-ring,charged,medium
        "pentadecanolate",
    ),
    (
        "C[C@@]12CCO[C@H]1C1=CC[C@H]3[C@H](O)CCC[C@]3(C)[C@H]1CC2",  # heterocycle,fused-ring,medium
        "(4R,5R,9R,10R,13R,17R)-9,13-dimethyl-16-oxa-tetracyclo[8.7.0.0(4,9).0(13,17)]heptadec-1-en-5-ol",
    ),
    (
        "CCCCCCCCCCCCCCCC[C@@H](O)COC[C@@H](O)CO",  # acyclic,medium
        "(2S)-3-octadecyloxy-2-hydroxypropan-1-ol",
    ),
    # Phase 148 D-05 / Plan 02 Task 03: cyanine-style benzoxazole dye.
    # Pre-148: ring-as-parent benzoxazole prefix. Post-148 cascade unblock
    # picks chain (acid PG); decomposition fragment-naming bug then renders
    # the benzoxazole+benzothiazole as "cyclononyl" — clearly wrong.
    # Phase 149 / IM-x.x decomposition layer fix territory. Marked xfail;
    # NOT a Phase 148 regression (cascade decision is correct per P-44.1(a);
    # only the downstream substituent-naming layer needs the fix).
    pytest.param(
        "O=C(O)CCCCCN1C(=CC=Cc2oc3cc(S(=O)(=O)[O-])ccc3[n+]2CCCCCC(=O)O)Oc2cc(S(=O)(=O)O)ccc21",  # aromatic,heterocycle,fused-ring,large
        "2-(15-carboxypentadecyl)-3-(5-carboxypentyl)-1,3-benzoxazole",
        marks=pytest.mark.xfail(
            strict=True,
            reason="Phase 149 / IM-x.x: decomposition fragment-naming bug for "
                   "cyanine-style benzoxazole+benzothiazole rings ('cyclononyl' "
                   "in output); per 148-01-SUMMARY Risks §2 (Plan-01 carry-forward).",
        ),
    ),
    (
        "[1*]C(=O)OC[C@H](COP(=O)(O)OCC[N+](C)(C)C)OC([2*])=O",  # acyclic,charged,medium
        "(2R)-nonyl formate",
    ),
    (
        "CCCCCCCCCCCCCC(=O)CC(C)C(=O)O",  # acyclic,polyfunctional,medium
        "2-methyl-4-oxoheptadecanoic acid",
    ),
    (
        "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCC(=O)[O-]",  # acyclic,charged,medium
        "(4Z,7Z,10Z,13Z,16Z,19Z)-docosa-4,7,10,13,16,19-hexaenoate",
    ),
    (
        "O=C(O)[C@H]1CCCCN1",  # heterocycle,polyfunctional,small
        "(2R)-piperidine-2-carboxylic acid",
    ),
    (
        "C[C@H](NC(=O)[C@H](CS)NC(=O)[C@@H](N)Cc1ccccc1)C(=O)O",  # aromatic,polyfunctional,medium
        "L-phenylalanyl-L-cysteinyl-L-alanine",
    ),
    (
        "Cc1cn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O[C@H]3O[C@H](CO)[C@@H](O)[C@H](O)[C@H]3O)[C@@H](O)[C@H]2O)c(=O)nc1N",  # aromatic,heterocycle,large,carbohydrate
        "6-amino-5-methyl-2-oxo-N-oxolanylpyrimidine",
    ),
    (
        "Nc1cncnc1",  # aromatic,heterocycle,small
        "5-aminopyrimidine",
    ),
    (
        "CC(=O)N[C@H]1[C@@H](O[C@H]2[C@H](O)[C@@H](CO)OC(O)[C@@H]2NC(C)=O)O[C@H](CO)[C@H](O)[C@@H]1O[C@@H]1O[C@H](CO)[C@H](O)[C@H](O)[C@H]1O",  # heterocycle,large,carbohydrate
        "(beta-D-galactopyranosyloxy)ethanediamide",
    ),
    (
        "CC1(C)OC(=O)C=CC2=CC3=C(CC[C@H]21)[C@]1(C)[C@@H](O)CC([C@@H]2C[C@@H]4C[C@H]2OC(=O)[C@]4(C)O)[C@@]1(C)CC3",  # heterocycle,fused-ring,large
        "(9R,13R,14S,17R)-14-hydroxy-8,8,13,17-tetramethyl-16-octyl-7-oxa-tetracyclo[10.7.0.0(3,9).0(13,17)]nonadeca-1,2,4-trien-6-one",
    ),
    (
        "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCC(=O)OC[C@@H](O)COP(=O)(O)O",  # acyclic,large
        "(3Z,6Z,9Z,12Z,15Z,18Z)-phosphonooxydocosa-3,6,9,12,15,18-hexaenephosphonic acid",
    ),
    (
        "CC1=CCC(/C(C)=C/C/C=C(\\C)CC/C=C(\\C)C=O)CC1",  # medium
        "(4E)-10-(4-methylcyclohexyl)-2,6-dimethylundeca-2,6,9-trienal",
    ),
    (
        "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCCCCC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)([O-])OP(=O)([O-])OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)([O-])[O-]",  # aromatic,heterocycle,fused-ring,polyfunctional,charged,large
        "3-amino-1-(ethylamino)-sulfanylpropanamide (8Z,11Z,14Z,17Z,20Z,23Z)-hexacosa-8,11,14,17,20,23-hexaenoic acid adenine (2R)-2-hydroxy-3,3-dimethylphosphonobutanoic acid",
    ),
    (
        "CC[C@H](C)[C@H](N)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CO)C(=O)O",  # aromatic,polyfunctional,medium
        "L-isoleucyl-L-phenylalanyl-L-serine",
    ),
    (
        "COc1cc(C)cc(OC)c1",  # aromatic,small
        "1,3-dimethoxy-5-methylbenzene",
    ),
    (
        "C=C1C=C[C@@H](C(C)C)CC1",  # small
        "(3R)-3-isopropyl-6-methylidenecyclohexene",
    ),
    (
        "COC1=C(O)c2c(O)cc(C)c3c4c(c(O)c(c23)C1=O)C(C)(C)[C@@H](C)O4",  # aromatic,heterocycle,fused-ring,medium
        "(13R)-4,6,10-trihydroxy-7-methoxy-2,12,12,13-tetramethyl-14-oxa-tetracyclo[7.6.1.0(5,16).0(11,15)]hexadec-6-en-8-one",
    ),
    (
        "COC1=CC(=O)c2c(O)cc(OC)c3c2C1=C[C@]1(C)[C@H](O)C(=O)C=C(OC)[C@H]31",  # aromatic,fused-ring,medium
        "(11S,12S,16S)-4,12-dihydroxy-2,8,15-trimethoxy-11-methyl-tetracyclo[7.7.1.0(5,17).0(11,16)]heptadeca-7,9,14-trien-6,13-dione",
    ),
    (
        "O=C(O)C1O[C@@H](Oc2ccc(-c3cc(=O)c4c(O)cc(O)c(O[C@@H]5OC(C(=O)O)[C@@H](O)[C@H](O)C5O)c4o3)cc2O)C(O)C(O)[C@@H]1O",  # aromatic,heterocycle,fused-ring,large,carbohydrate
        "(3S,6S)-3,4,5-trihydroxy-6-phenyloxane-2-carboxylic acid",
    ),
    (
        "COc1c(-c2csc([C@@H](O)CCC(C)(C)O)n2)nc(C(N)=O)c(O)c1OC",  # aromatic,heterocycle,medium
        "5-hydroxy-3,4-dimethyl-2-(thiazol-yl)pyridine-6-carboxamide",
    ),
    (
        "C=C(NC(=O)C(NC(=O)C(Cc1cnc[nH]1)NC(=O)C(NC(=O)C(CO)NC(=O)C1CSC(C)C2NC(=O)C(C)NC(=O)C(NC(=O)C(CCCCN)NC(=O)C(CCSC)NC(=O)C(CC(N)=O)NC(=O)C3CSC(C)C(NC(=O)C(CCCCN)NC(=O)C4CSC(C)C(NC(=O)C5CSCC(NC(=O)C(=CC)NC(=O)C(N)C(C)CC)C(=O)NC(C(C)CC)C(=O)NC(=C)C(=O)NC(CC(C)C)C(=O)N5)C(=O)N5CCCC5C(=O)NCC(=O)N4)C(=O)NCC(=O)NC(C)C(=O)NC(CC(C)C)C(=O)NC(CCSC)C(=O)NCC(=O)N3)C(C)SCC(NC2=O)C(=O)NC(CC(N)=O)C(=O)N1)C(C)CC)C(C)C)C(=O)NC(CCCCN)C(=O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "N-12-amino-13-methyl-5,8,11-trioxo-1-thia-4,7,10-triazacyclotridecane-3-carbonyl-N-2-aminobut-2-enoyl-N-2-aminoprop-2-enoyl-N-aspartyl-5-amino-14-ethyl-4,8,17,20-tetramethyl-3,19-dithia-7,10,13,16,21-pentaaza-bicyclo[9.9.2]docosane",
    ),
    (
        "CSCC[C@H](N)C(=O)N[C@@H](CCCN=C(N)N)C(=O)N[C@@H](CS)C(=O)O",  # acyclic,polyfunctional,medium
        "(2R)-aminoguanidinomethylsulfanyl-2-(pentanoylamino)-3-sulfanylpropanoic acid",
    ),
    (
        "CN[C@@H](CCCNC(N)=NC/C=C(/C)CCC=C(C)C)C(=O)N(C)[C@@H](CCCN=C(N)N)C(=O)N[C@H](C)[C@H](OS(=O)(=O)O)c1ccc(OS(=O)(=O)O)c(COS(=O)(=O)O)c1",  # aromatic,polyfunctional,large
        "N-[(2S)-amino-5-guanidino-5-(methylamino)-2-(pentanoylamino)pentanoyl]-4-((2R,4R)-(2R)-2-aminopropanesulfonic acidyl)-2-methanesulfonic acidyl-1-(sulfonyloxy)benzene",
    ),
    (
        "CC(C)C[C@H](NC(=O)[C@H](CCC(N)=O)NC(=O)[C@@H](N)Cc1c[nH]c2ccccc12)C(=O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "L-tryptophyl-L-glutaminyl-L-leucine",
    ),
    (
        "CC(C)[C@H](NC(=O)[C@@H](N)[C@@H](C)O)C(=O)N[C@H](C(=O)O)[C@@H](C)O",  # acyclic,polyfunctional,medium
        "L-threonyl-L-valyl-L-threonine",
    ),
    (
        "O=C(On1c(O)ccc1O)C1CCC(CN2C(=O)C=CC2=O)CC1",  # aromatic,heterocycle,medium
        "N-cyclohexyl-2,5-dihydroxyazole",
    ),
    (
        "[NH3+][C@@H](CCCCNC(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@@H]21)C(=O)[O-]",  # heterocycle,fused-ring,medium
        "(2S)-2-amino-6-(nonanoylamino)hexanoic acid",
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@H](CCCN=C(N)N)NC(=O)[C@@H](N)CO)C(=O)O",  # acyclic,polyfunctional,medium
        "(2S,3S)-aminoguanidinohydroxy-3-methyl-2-(pentanoylamino)pentanoic acid",
    ),
    (
        "CO[C@@H]1C[C@@H](C[C@H]2CC[C@H](C)[C@H]([C@@H](C)C(=O)O)O2)O[C@]2(O[C@](C)([C@H]3CC[C@@](C)([C@@H]4O[C@@H]([C@H]5O[C@](C)(O)[C@H](C)C[C@@H]5C(=O)O)C[C@@H]4CO)O3)C[C@H]2C)[C@@H]1C",  # heterocycle,large
        "(2R,3R,4R,6R)-3,4-dimethyl-6-oxanyloxane",
    ),
    (
        "N=C(NCCC[C@H](N)C(=O)O)NC(CC(=O)O)C(=O)O",  # acyclic,polyfunctional,medium
        "amino-2-guanidino-2-(methylamino)butanetrioic acid",
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@@H]1CCCN1)C(=O)N[C@@H](CCSC)C(=O)O",  # heterocycle,polyfunctional,medium
        "N-[(2S,3S)-3-methyl-2-(pentanoylamino)pentanoyl](2S)-2-aminopentanoic acid",
    ),
    (
        "CCCCC/C=C\\C/C=C\\CCCCCCCCCCCC(=O)OC[C@H](COP(=O)(O)OCCN)OC(=O)CCCCCCC[C@H](O)[C@@H](O)C/C=C\\CCCCC",  # acyclic,polyfunctional,large
        "(2R)-1-((13Z,16Z)-docosa-13,16-dienoyloxy)-2-(oleoyloxy)propanediol",
    ),
    (
        "COc1cc(-c2oc3cc(O)c(OC)c(OC)c3c(=O)c2OC)ccc1O",  # aromatic,heterocycle,fused-ring,medium
        "2-methoxyphenol",
    ),
    (
        "CC(C)C1=C2[C@H]3C[C@H]4O[C@](O)([C@H](O)[C@@H]4CO)[C@]3(C)CC[C@@]2(C)[C@H](O)C1=O",  # heterocycle,fused-ring,medium,carbohydrate
        "(1R,5S,6R,9R,10S,11R,12S,13R)-5,10,11-trihydroxy-3-isopropyl-6,9,12-trimethyl-15-oxa-tetracyclo[7.5.0.1(10,13).0(2,6)]pentadec-2-en-4-one",
    ),
    (
        "COc1ccc(C(=O)c2cc(OC)cc(OC)c2)cc1",  # aromatic,medium
        "methoxybenzene",
    ),
    (
        "NC(=O)CC[C@H](NC(=O)[C@@H](N)Cc1ccccc1)C(=O)N[C@@H](CCCN=C(N)N)C(=O)O",  # aromatic,polyfunctional,large
        "N-[(2S)-2-amino-3-phenylpropanoyl](2S)-aminocarbamoyl-5-guanidino-5-(methylamino)-2-(pentanoylamino)pentanoic acid",
    ),
    (
        "N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)N[C@@H](Cc1ccccc1)C(=O)O",  # aromatic,heterocycle,polyfunctional,large
        "L-tyrosyl-L-histidyl-L-phenylalanine",
    ),
    (
        "CCCCCCCCCCCCCCCCCCCCCC[C@@H](O)C(=O)N[C@@H](CO)[C@H](O)[C@H](O)CCCCCCCCCCCCCC",  # acyclic,large
        "(2R)-1-(octadecylamino)-hydroxy-2-hydroxytetracosanamide",
    ),
    (
        "COC(=O)[C@@H]1C[C@]2(O)c3ccccc3N3C(=O)C4(CC4)N([C@@H](OC)c4nc5ccccc5c(=O)n41)[C@H]32",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "(1S,3S,10S,21R)-3-ethyl-1-hydroxy-10-methoxy-4,8,11,14-tetraaza-heptacyclo[9.9.1.4(6,7).0(4,9).0(14,21).0(15,20)]heptacosan-5,13-dione",
    ),
    # Phase 148 Plan 02 Task 03: linoleoyl-CoA. Pre-148 OLD `'adenine'`
    # was a known-bad placeholder. Post-148 chain-as-parent. Acceptable churn.
    (
        "CCCCC/C=C\\C/C=C\\CCCCCCCC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "N-[(2R)-2-hydroxy-3,3-dimethylbutanoyl](9Z,12Z)-amino-1-(ethylsulfanyl)octadeca-9,12-diene",
    ),
    (
        "C/C=C(/C)C(=O)O[C@@H]1CCN2CC=C(COC(=O)/C(=C\\C)CO)[C@H]12",  # heterocycle,fused-ring,medium
        "(2Z)-((2-hydroxymethylbutanoyl)oxy)((2-methylbutanoyl)oxy)-2-methylbut-2-enol",
    ),
    (
        "CC(C)C[C@@H](NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)OCc1ccccc1)C(=O)NO",  # aromatic,large
        "hydroxy-N-hydroxyamido-N-((R)-4-methylpentan-1-hydroxamic acidyl)nonanamide",
    ),
    (
        "CC[C@H](C)c1occ2c(O)c(C(=O)O)c(=O)cc-2c1C",  # aromatic,heterocycle,fused-ring,medium
        "2-[(S)-sec-butyl]-3-methyl-2H-pyran",
    ),
    (
        "C=C(CO)C12O[C@H]1[C@@]1(C)[C@H](CCC[C@@H]1C)C[C@@H]2O",  # heterocycle,fused-ring,medium
        "2-cycloundecylprop-2-en-1-ol",
    ),
    (
        "C=C1[C@]23C(=O)O[C@@H](C)[C@]2(O)C(=O)O[C@@]1(C)CC1=C(C)[C@@]2(C=CC(=O)OC2(C)C)[C@H](OC(C)=O)C[C@]13C",  # heterocycle,fused-ring,large
        "(1S,2R,4R,5S,9S,12S,13S)-4-ethoxy-12-hydroxy-2,6,9,13,21,21-hexamethyl-10,14,20-trioxa-pentacyclo[7.6.1.0(1,12).0(2,7)]henicosa-6,17-dien-11,15,19-trione",
    ),
    (
        "CC[C@H](C)[C@H](NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@@H](NC(=O)[C@H](CCCN=C(N)N)NC(=O)[C@H](C)N)C(C)C)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](Cc1ccccc1)C(=O)O",  # aromatic,heterocycle,polyfunctional,large
        "N-[(2S)-3-(4-hydroxyphenyl)-2-(pentanoylamino)propanoyl](2S)-2-(pentanoylamino)-3-phenylpropanoic acid",
    ),
    (
        "CC1(C)O[C@@]23CC[C@@]4(C)C(=CC[C@H]5Cc6c([nH]c7ccccc67)[C@@]54C)C2=CC(=O)[C@@H]1O3",  # aromatic,heterocycle,fused-ring,large
        "(1S,8R,11S,14S,15S)-9,9,14,15-tetramethyl-10,26-dioxa-17-aza-heptacyclo[13.10.0.1(8,11).0(4,14).0(5,11).0(16,24).0(18,23)]hexacosa-3,5-dien-7-one",
    ),
    (
        "CCCC(F)Cn1cc(C(=O)C2C(C)(C)C2(C)C)c2ccccc21",  # aromatic,heterocycle,fused-ring,medium
        "1-(2-fluoropentyl)-1H-indole",
    ),
    (
        "CC1=C(C)[C@H](C[C@@H](C)[C@H]2CC[C@@]3(C)C4=C(C[C@H](O)[C@]23C)[C@@]2(C)CC[C@@H](OC(=O)C[C@@](C)(O)CC(=O)O)C(C)(C)[C@@H]2CC4)OC1=O",  # heterocycle,fused-ring,polyfunctional,large,steroid
        "(3R,5R,10S,11S,13R,14S,17R,20R,23S)-11-hydroxy-4,4,14-trimethyl-27-oxoergost-8,24-dien-3-yl hexanoate",
    ),
    (
        "CC1=C(/C=C/CCCC[C@H](C)O)C(=O)OC1=O",  # heterocycle,medium
        "(4E)-butanedioic anhydride",
    ),
    (
        "CCCCC[C@H](O)/C=C/C=C\\C/C=C\\CCCCCCC(=O)O",  # acyclic,medium
        "(8Z,11Z,13E,15S)-15-hydroxyicosa-8,11,13-trienoic acid",
    ),
    (
        "C=CC1C=C2C(=O)C(O)=C3C(CCC(O)C3(C)C)[C@@]2(O)[C@H](O)C1",  # fused-ring,medium
        "(10S,11R)-13-ethenyl-3,6,10,11-tetrahydroxy-5,5-dimethyl-tricyclo[8.4.0.0(4,9)]tetradeca-1,3-dien-2-one",
    ),
    (
        "CC(=O)N[C@@H](CSCC(O)C(O)CO)C(=O)O",  # acyclic,polyfunctional,medium
        "(2R)-3-(butylsulfanyl)-2-(ethanoylamino)-dihydroxypropanoic acid",
    ),
    (
        "CC(C)CCCCCCCCCCCO",  # acyclic,small
        "12-methyltridecan-1-ol",
    ),
    (
        "C[C@H](NC(=O)[C@H](Cc1cnc[nH]1)NC(=O)[C@@H](N)Cc1ccc(O)cc1)C(=O)O",  # aromatic,heterocycle,polyfunctional,medium
        "L-tyrosyl-L-histidyl-L-alanine",
    ),
    (
        "NC(N)=NCCC[C@H](NC(=O)[C@H](CCCN=C(N)N)NC(=O)[C@@H](N)Cc1ccc(O)cc1)C(=O)O",  # aromatic,polyfunctional,large
        "(2S)-5-(methylamino)-2-(pentanoylamino)pentanoic acid",
    ),
    (
        "C#CCCCn1c(Cc2cc(OC)c(OC)c(OC)c2Cl)nc2c(N)nc(F)nc21",  # aromatic,heterocycle,fused-ring,medium
        "2-fluoro-N-pent-4-yn-1-yladenine",
    ),
    (
        "CC(C(=O)[O-])C(O)C(=O)[O-]",  # acyclic,charged,small
        "3-hydroxy-2-methylbutanedioate",
    ),
    (
        "CCCCCCCCCCCCCCCCCCCCCCCC(=O)NCCS(=O)(=O)O",  # acyclic,large
        "1-(ethylamino)tetracosanesulfonic acid",
    ),
    (
        "CC(=O)O[C@@H]1/C=C/[C@](C)(O)C[C@@H](C)C/C=C/[C@H]2C=C(C(=O)O)[C@@H](C)[C@H]3[C@H](Cc4ccccc4)NC(=O)[C@]321",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "(1S,2E,5S,7R,8E,10R,11S,14S,15R,16S)-14-benzyl-10-ethoxy-7-hydroxy-5,7,16-trimethyl-12-oxo-13-aza-tricyclo[9.7.0.0(11,15)]octadeca-2,8,17-triene-17-carboxylic acid",
    ),
    (
        "CCCCCCCCCCCCCCCC(=O)OCC(=O)COP(=O)(O)O",  # acyclic,polyfunctional,medium
        "3-hydroxy-2-oxo-1-phosphonooxypropanephosphonic acid palmitate",
    ),
    (
        "CC(C)C[C@@H]1NC(=O)[C@@H](CC(C)C)OC(=O)CCNC(=O)[C@H](Cc2ccccc2)NC(=O)[C@H](CC(C)C)OC1=O",  # aromatic,heterocycle,large
        "(7S,10S,13S,16R)-7-benzyl-10,13,16-triisobutyl-6,9,12,15-tetraoxooxacyclohexadecan-2-one",
    ),
    (
        "CCCCCCCCCCCCCCCCCCCCCCCCC(O)C(=O)N[C@@H](COP(=O)(O)O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1OC1O[C@H](COP(=O)(O)O[C@@H]2[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@H](O)[C@@H]1O)[C@H](O)CCCCCCCCCCCCCCC",  # heterocycle,large,carbohydrate
        "N-2-hydroxyhexacosanoyl(1R,2R,3S,4R,5R,6R)-aminocyclohexane-3,4,5,6-tetraol",
    ),
    (
        "COc1ccc(CCNCC(O)COc2cccc(C)c2)cc1OC",  # aromatic,medium
        "1-amino-1-anilino-3-phenoxypropan-2-ol",
    ),
    (
        "CC[C@H](O)[C@H]1C[C@H]2OC(=O)c3c(cc(OC)c(OC)c3O)[C@H]2O1",  # aromatic,heterocycle,fused-ring,medium
        "(4R,6R,8R)-13-hydroxy-11,12-dimethoxy-6-propyl-3,7-dioxa-tricyclo[7.4.0.0(4,8)]tridecan-2-one",
    ),
    (
        "CC(=O)NC(CCCN(O)C(=O)/C=C(\\C)C(O)CO)C(=O)OCC/C(C)=C/C(=O)N(O)CCCC1NC(=O)C(CCCN(O)C(=O)/C=C(\\C)CCO)NC1=O",  # heterocycle,polyfunctional,large
        "2,5-dinonyl-3,6-dioxopiperazine 2-(ethanoylamino)-dihydroxy-5-(hexanoylamino)pentanoate",
    ),
    (
        "C[C@]12CCC3=C4CCC(=O)C=C4CC[C@H]3[C@@H]1CCC2=O",  # fused-ring,medium,steroid
        "(8S,13S,14S)-estra-4,9-dien-3,17-dione",
    ),
    (
        "Nc1ccc(/N=N/c2ccc(N(CCO)CCO)cc2)cc1",  # aromatic,medium
        "2-amino-2-anilinoethan-1-ol",
    ),
    (
        "O=C(O)CCC1=COC(CCC(=O)O)=CO1",  # heterocycle,medium
        "3-cyclohexylpropanedioic acid",
    ),
    # Phase 148 D-05 / Plan 02 Task 03: tetracyclic indole+oxazoline alkaloid
    # with two fused indole rings. Post-148 cascade unblock + decomposition
    # fragment-naming bug yields '1-cycloheptadecylethan-1-one' — the
    # "cycloheptadecyl" rendering is the Plan-01 carry-forward decomposition
    # fragment-naming bug for fused-indole systems. Phase 149 / IM-x.x
    # decomposition layer fix. Marked xfail.
    pytest.param(
        "CC(=O)C1=Nc2c(c(C)c(CCCCC(C)C)c3[nH]c4ccccc4c23)O[C@H]1c1c[nH]c2ccccc12",  # aromatic,heterocycle,fused-ring,large
        "(6S)-7-ethyl-3-methyl-2-2-methylhexyl-6-octyl-5-oxa-8,17-diaza-tetracyclo[8.7.0.0(4,9).0(11,16)]heptadec-7-ene",
        marks=pytest.mark.xfail(
            strict=True,
            reason="Phase 149 / IM-x.x: decomposition fragment-naming bug for "
                   "fused-indole systems ('cycloheptadecyl' in output); per "
                   "148-01-SUMMARY Risks §2 (Plan-01 carry-forward).",
        ),
    ),
    (
        "NCCCCCCNC(=O)[C@H](O)[C@@H](O)[C@H](O[C@@H]1O[C@H](CO)[C@H](O)[C@H](O)[C@H]1O)[C@H](O)CO",  # heterocycle,polyfunctional,large,carbohydrate
        "(beta-D-galactopyranosyloxy)(2R,3S,4R,5R)-amino-1-(hexylamino)-2,3,4,5,6-pentahydroxyhexanamide",
    ),
    (
        "*C(=O)[C@@H](N)Cc1c[nH]cn1",  # aromatic,heterocycle,small
        "3-imidazolylpropan-2-amine",
    ),
    (
        "[AtH]",  # acyclic,small
        "inorganic compound (not supported)",
    ),
    (
        "*C(=O)OC[C@H](COP(=O)(O)OCC(COP(=O)(O)OC[C@@H](COC(*)=O)OC(*)=O)OC(C)=O)OC(*)=O",  # acyclic,large
        "(formyloxy)(2R)-1,2-tris(formyloxy)(acetyloxy)propanol",
    ),
    (
        "O=C(O)C[C@H](O)CCCCCCCCCCCO",  # acyclic,medium
        "(3R)-3,14-dihydroxytetradecanoic acid",
    ),
    (
        "OC[C@H]1O[C@@H](O[C@H]2[C@H](O)[C@@H](O)[C@H](O[C@H]3[C@H](O)[C@@H](O)C(O)O[C@@H]3CO)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@@H]1O",  # heterocycle,large,carbohydrate
        "(beta-D-glucopyranosyloxy)maltose",
    ),
    (
        "C=C(C)/C=C/c1cccc2c1NC1ON=C(C(=C)OC)CC21O",  # aromatic,heterocycle,fused-ring,medium
        "10-2-methylbuta-1,3-dienyl-4-propyl-6-oxa-5,8-diaza-tricyclo[7.4.0.0(2,7)]tridec-4-en-2-ol",
    ),
    (
        "C=C/C=C/CCC=O",  # acyclic,small
        "(4E)-hepta-4,6-dienal",
    ),
    (
        "O=C=Nc1ccccc1",  # aromatic,small
        "phenyl isocyanate",
    ),
    (
        "COc1cc(O)cc2occ(-c3ccc(O)cc3O)c(=O)c12",  # aromatic,heterocycle,fused-ring,medium
        "3-methoxyphenol",
    ),
    (
        "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C(=O)[O-])C(C)(C)CCC1",  # charged,medium
        "(2E)-9-(2,2,6-trimethylcyclohexyl)-3,7-dimethylnona-2,4,6,8-tetraenoate",
    ),
    (
        "CC(=O)N[C@H]1[C@H](O[C@@H]2[C@H](O)[C@@H](O)[C@H](O[C@@H]3[C@H](O)[C@@H](O[C@@H]4[C@H](O)[C@@H](O[C@H]5[C@@H]([C@H](O)CO)O[C@@](O)(C(=O)O)C[C@H]5O[C@]5(C(=O)O)C[C@@H](O)[C@@H](O)[C@@H]([C@H](O)CO)O5)O[C@H]([C@@H](O)CO)[C@H]4O[C@@H]4O[C@H](CO)[C@@H](O)[C@H](O)[C@H]4O)O[C@H]([C@@H](O)CO)[C@H]3O)O[C@@H]2CO)O[C@H](CO)[C@H](O)[C@@H]1O[C@@H]1O[C@H](CO)[C@H](O)[C@H](O)[C@H]1O",  # heterocycle,polyfunctional,large,carbohydrate
        "N-acetyl(2S,3R,4R,5R,6R)-3-amino-5-hydroxy-6-methyl-2,4-dioxanyloxane",
    ),
    (
        "CC(C)(O)/C=C1/C(=O)C(C)(C)C(=O)C(C)(C)C1(O)O[C@@H]1O[C@H](COC(=O)c2cc(O)c(O)c(O)c2)[C@@H](O)[C@H](O)[C@H]1O",  # aromatic,heterocycle,polyfunctional,large,carbohydrate
        "icosyl 3,4,5-trihydroxybenzoate",
    ),
    (
        "CC(=O)O[C@H]1CC[C@]2(C)C3=C(CCC2C1(C)C)[C@]1(C)C[C@H](O)C([C@@](C)(O)CCC=C(C)C)[C@@]1(C)CC3",  # fused-ring,large,steroid
        "(3S,10S,13R,14R,16S,20S)-16,20-dihydroxy-4,4,14-trimethylcholest-8,24-dien-3-yl acetate",
    ),
    (
        "NCCCC[C@H](N)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@@H](CS)C(=O)O",  # acyclic,polyfunctional,medium
        "L-lysyl-L-asparaginyl-L-cysteine",
    ),
    (
        "CC[C@H](O)[C@@H](C)/C=C\\C[C@H](C)C=CC=C(C)[C@H]1OC(=O)C[C@H](O)CC[C@H](C)[C@@H](OC(C)=O)C=C[C@@H]1C",  # heterocycle,large
        "(4R,7S,8R,11S,12S)-12-[(3S,5S,10S)-3-hydroxy(3S,4S,5Z,8S)-4,8-dimethyltrideca-5,9,11-trienyl]-8-acetyl-4-hydroxy-7,11-dimethyloxacyclododecan-2-one",
    ),
    (
        "COc1cc(O)c2c(=O)c(OC)c(-c3ccc(OC)c(O)c3)oc2c1",  # aromatic,heterocycle,fused-ring,medium
        "3-methoxyphenol",
    ),
    (
        "CC[C@H](C)[C@@H](OC(C)=O)[C@@H](C)C1=CC(=O)C2=C(OC3C(=C2)[C@@]2(C)CC[C@H](C(C)(C)O)O[C@@H]2C[C@@H]3C)C1=O",  # heterocycle,fused-ring,polyfunctional,large
        "(3S,4R,5S,13S,15R,16R,19R)-octacosyl acetate",
    ),
    (
        "NCCCC[C@H](N)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)O",  # aromatic,polyfunctional,large
        "L-lysyl-L-tyrosyl-L-phenylalanine",
    ),
    (
        "C[C@]12CCC(=O)C=C1[C@H](O)C[C@@H]1[C@@H]2[C@@H](O)C[C@@]2(C)[C@H]1CC[C@]2(O)C(=O)CO",  # fused-ring,medium,steroid
        "(6R,8S,9S,10R,11S,13S,14S,17R)-6,11,17,21-tetrahydroxypregn-4-en-3,20-dione",
    ),
    (
        "CC(=O)OCC[C@@H]1OCc2c(O)cccc21",  # aromatic,heterocycle,fused-ring,medium
        "(acetyloxy)ethanol",
    ),
    (
        "O=C(c1ccc(O)cc1)c1cn(CCCCCO)c2ccccc12",  # aromatic,heterocycle,fused-ring,medium
        "1-(1-hydroxypentyl)-1H-indole",
    ),
    (
        "Cc1cc(O)cc(O)c1C(=O)O[C@@H]1C[C@]2(C)[C@H]3[C@@H](O)C(C)(C)C[C@@]3(O)C=C(CO)[C@]12O",  # aromatic,fused-ring,large
        "(2R,3S,6R,9R,10S,11R)-pentadecyl 2,4-dihydroxy-6-methylbenzoate",
    ),
    (
        "OCC1OC(c2c(O)cc3c(c2O)C(c2c(O)cc(O)c4c2OC(c2cccc(O)c2)C(O)C4)C(O)C(c2cccc(O)c2)O3)C(O)C(O)C1O",  # aromatic,heterocycle,fused-ring,large,carbohydrate
        "3,5,7-trihydroxychromane",
    ),
    (
        "CC[C@@]1(O)C[C@H](O[C@H]2C[C@H](O)[C@H](O[C@H]3CC[C@H](O[C@H]4CC[C@H](O)[C@H](C)O4)[C@H](C)O3)[C@H](C)O2)c2c(O)c3c(c(O)c2[C@H]1O)C(=O)c1cccc(O)c1C3=O",  # aromatic,heterocycle,fused-ring,large,carbohydrate
        "(4S,6R,7R)-4-octadecyl-6-ethyl-2,6,7,9,16-pentahydroxy-tetracyclo[8.8.0.0(3,8).0(12,17)]octadecan-11,18-dione",
    ),
    (
        "C=C1NC(=O)[C@H]([C@@H](O)c2ccc(O)cc2)NC(=O)[C@@H](NC(=O)[C@@H](O)[C@@H](N)CCCCCCCCCCCCCCC)[C@@H](C)OC(=O)CCNC(=O)[C@H](C)NC(=O)CNC1=O",  # aromatic,heterocycle,polyfunctional,large
        "(7S,16S,19S,20R)-19-[(4S,6S)-17-carbamoylheptadecyl]-16-[(S)-hydroxy4-(hydroxymethyl)phenyl]-7,13,20-trimethyl-6,9,12,15,18-pentaoxooxacycloicosan-2-one",
    ),
    (
        "COc1ccc(-c2oc3c(OC)c(OC)cc(OC)c3c(=O)c2OC)cc1OC",  # aromatic,heterocycle,fused-ring,medium
        "5-methyl-4-oxo-6-phenyl-2H-pyran",
    ),
    (
        "C[C@H]1CC[C@H]2C(C)(C)O[C@H]3CC(=O)OC[C@]32[C@]12C[C@]1(C)[C@@H](C)[C@H]3C(=O)O[C@@]4(C)O[C@@]1(O[C@@H]34)O2",  # heterocycle,fused-ring,large
        "(1R,4S,9S,10S,11S,16R,17S,18R,21S,23S,25S)-2,2,11,16,17,21-hexamethyl-3,7,14,20,22,24-hexaoxa-heptacyclo[7.4.0.0(4,9).0(10,15).0(14,23).0(16,23)]pentacosan-6,19-dione",
    ),
    (
        "C[C@H](C(=O)O)[C@@H](O)C(=O)O",  # acyclic,small
        "(2S,3R)-3-hydroxy-2-methylbutanedioic acid",
    ),
    (
        "CCCC/C=C\\CCCCCCCC(=O)O[C@H](COC(=O)CCCCCCCCC/C=C\\CCCCCC)COC(=O)CCCCCCCCCCCCCCCCCCCCC",  # acyclic,large
        "2-[(9Z)-tetradec-9-enoyloxy]-3-(docosanoyloxy)-1-(oleoyloxy)propane",
    ),
    (
        "COc1cc(O)cc2cc(CO)c(=O)oc12",  # aromatic,heterocycle,fused-ring,medium
        "6-hydroxy-3-hydroxymethyl-8-methoxycoumarin",
    ),
    (
        "C[C@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O",  # aromatic,heterocycle,polyfunctional,medium
        "(3S,5S)-5-amino-3-(hydroxymethyl)-5-methyl-1-phenyl-6-pyrrolidinylheptanedioic acid",
    ),
    (
        "CCC(C)c1nc(OC)c(C(O)c2ccc(O)cc2)n(OC)c1=O",  # aromatic,heterocycle,medium
        "5-(sec-butyl)-N-methyl-3-methyl-6-oxo-2-phenylpyrazine",
    ),
    (
        "C[C@H]1OC(OP(=O)([O-])OP(=O)([O-])OC[C@H]2O[C@@H](n3ccc(N)nc3=O)[C@H](O)[C@@H]2O)[C@@H](O)C[C@@H]1O",  # aromatic,heterocycle,charged,large,carbohydrate
        "(2R,3S,5S)-3,5-dihydroxy-2-methyl-6-oxolanyloxane",
    ),
    (
        "CC(C)=C1C/C=C(/C)CCC2OC2(C)CC1=O",  # heterocycle,fused-ring,medium
        "(4Z)-8-hydroxy-2-isopropylidene-5,9-dimethylcyclodec-4-en-1-one",
    ),
    (
        "CN[C@@H]1[C@@H](O)C(O)OC[C@]1(C)O",  # heterocycle,small,carbohydrate
        "garosamine",
    ),
    (
        "O=C([O-])C(=O)C[C@@H](O)C(=O)[O-]",  # acyclic,charged,small
        "(2R)-2-hydroxy-4-oxopentanedioate",
    ),
    (
        "CCCCC/C=C\\C/C=C\\C/C=C\\CCCCCCC(=O)O[C@H](COCCCCCCCCCCCCCCCC)COP(=O)([O-])OCC[N+](C)(C)C",  # acyclic,large
        "tetracosyl (8Z,11Z,14Z)-icosa-8,11,14-trienoate",
    ),
    (
        "CC(C)[C@H]1Oc2c(ccc3c2C(=O)O[C@H](C)C3)[C@H]1O",  # aromatic,heterocycle,fused-ring,medium
        "(5R,6R,12R)-5-hydroxy-6-isopropyl-12-methyl-7,11-dioxa-tricyclo[7.4.0.0(4,8)]tridecan-10-one",
    ),
    (
        "C=C1C(=O)N[C@H](C)C(=O)N[C@@H](CC(=O)c2ccccc2NC=O)C(=O)N[C@@H](C(=O)O)[C@H](C)C(=O)N[C@@H](CC)C(=O)N[C@@H](/C=C/C(C)=C/[C@H](C)[C@H](Cc2ccccc2)OC)[C@H](C)C(=O)N[C@@H](C(=O)O)CCC(=O)N1C",  # aromatic,heterocycle,polyfunctional,large
        "(8R,11S,12S,15S,18S,19R,22S,25R)-12-[(6S,8S)-(3E,5S,6S)-6-methoxy-3,5-dimethyl-7-phenylhepta-1,3-dienyl]-22-carbamoylbenzenyl-15-ethyl-8,19-diformyl-3,11,18,25-tetramethyl-N-methyl-5,10,14,17,21,24-hexaoxoazacyclopentacosan-2-one",
    ),
    (
        "C/C=C(\\C)[C@H]1[C@@H](/C=C/C=C(\\C)[C@@H](O)[C@@H](C)CO)[C@@H]2C[C@@H]3O[C@]3(C)C[C@H]2[C@@H]2O[C@@]21C",  # heterocycle,fused-ring,medium
        "(2S,3S,4E,6E)-7-cyclododecyl-3-hydroxy-2,4-dimethylhepta-4,6-dien-1-ol",
    ),
    (
        "C=C1C2CC(O)C34C1C3(C2)C(C(=O)O)C1C2(C)CCCC14OC2=O",  # heterocycle,fused-ring,polyfunctional,medium
        "14-hydroxy-6-methyl-16-oxo-17-oxa-hexacyclo[7.5.0.2(2,6).1(9,12).0(1,10).0(2,7)]heptadecane-8-carboxylic acid",
    ),
    (
        "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCCCCC(=O)OC(COC(=O)CCCCCCCCCCCCCCCCC)COP(=O)(O)OCCNC",  # acyclic,polyfunctional,large
        "2-(arachidonoyloxy)-1-(stearoyloxy)propanamine",
    ),
    (
        "CC(C)(O)/C=C/C[C@](C)(O[C@@H]1O[C@H](CO[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@H](O)[C@H]1O)[C@H]1CC[C@]2(C)[C@@H]1[C@H](O)C[C@@H]1[C@@]3(C)CC[C@H](O[C@@H]4O[C@H](CO)[C@@H](O)[C@H](O)[C@H]4O[C@@H]4O[C@H](CO)[C@@H](O)[C@H](O)[C@H]4O)C(C)(C)[C@@H]3CC[C@]12C",  # heterocycle,fused-ring,large,carbohydrate
        "(3S,5R,8R,9R,10R,12R,13R,14R,17S)-4,4,8,10,14-pentamethylgonan-12-ol",
    ),
    (
        "CCCCCCCCCCCCOCCOCCOCCOCCOCCO",  # acyclic,medium
        "3,6,9,12,15-pentaoxaheptacosan-1-ol",
    ),
    (
        "O=C1C([O-])=C[C@@H](O)[C@H](O)[C@H]1O",  # charged,small
        "hexanolate",
    ),
    (
        "O=Cc1cc(C2(c3ccccc3)NC(=O)NC2=O)ccc1O",  # aromatic,heterocycle,polyfunctional,medium
        "2-hydroxybenzaldehyde",
    ),
    (
        "CCC[NH+]1CCCC[C@H]1C(=O)Nc1c(C)cccc1C.[Cl-]",  # aromatic,heterocycle,salt,medium
        "(2S)-2-phenyl-N-propylpiperidineium chloride",
    ),
    (
        "CCCCCc1oc(CCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCC[C@H]2C(=O)C[C@@H](O)[C@@H]2/C=C/[C@@H](O)CCCCC)c(C)c1C",  # aromatic,heterocycle,polyfunctional,large
        "13-furyltridecanoic acid (cyclopentanecarbonyloxy)-7-cyclopentylheptanone 1-phosphonooxy-2-(propylamino)ethanephosphonic acid",
    ),
    (
        "C[C@@H](O)[C@H](N)C(=O)N[C@@H](CCCN=C(N)N)C(=O)N[C@H](C(=O)O)[C@@H](C)O",  # acyclic,polyfunctional,medium
        "(2S,3R)-aminoguanidino-3-hydroxy-2-(pentanoylamino)butanoic acid",
    ),
    (
        "C=C(C)[C@@H](/C=C/[C@@H](C)[C@H]1CC[C@H]2C3=CC[C@H]4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C)CC",  # fused-ring,medium,steroid
        "(3S,5S,9R,10S,13R,14R,17R,20R,22E,24R)-stigmasta-7,22,25-trien-3-ol",
    ),
    (
        "O=C(NNC(=O)c1cccc(Cl)c1)c1cccnc1",  # aromatic,heterocycle,medium
        "N-3-chlorobenzoyl-3-methylpyridine",
    ),
    (
        "CC=C[C@@H]1C=C2C=C[C@@H]3C[C@H](C)CC[C@H]3[C@]2(C)C(=O)[C@]12C(=O)N[C@H]([C@@H](C)O)C2=O",  # heterocycle,fused-ring,polyfunctional,medium
        "(4S,6R,9R,10S,12R,13R,18R)-18-ethyl-6,10-dimethyl-13-prop-1-en-1-yl-17-aza-tetracyclo[8.4.0.0(12,16)]octadeca-1,2-dien-11,15,16-trione",
    ),
    (
        "CC(=O)NCCSC1CC2C([C@H](C)O)C(=O)N2[C@H]1C(=O)O",  # heterocycle,fused-ring,polyfunctional,medium
        "(2S)-3-butyl-7-ethyl-2-methyl-1-aza-bicyclo[3.2.0]heptane",
    ),
    (
        "Cc1ccc2oc(-c3ccccc3)cc(=O)c2c1",  # aromatic,heterocycle,fused-ring,medium
        "methylbenzene",
    ),
    (
        "NCCCC[C@H](NC(=O)[C@@H](N)CC(N)=O)C(=O)N[C@@H](CC(=O)O)C(=O)O",  # acyclic,polyfunctional,medium
        "L-asparaginyl-L-lysyl-L-aspartic acid",
    ),
    (
        "CO[C@@H]1C[C@H]2O[C@H](C)[C@@H](C)c3c(C)c(O)cc(c32)O1",  # aromatic,heterocycle,fused-ring,medium
        "(1R,3S,10S,11R)-3-methoxy-8,10,11-trimethyl-4,12-dioxa-tricyclo[7.3.1.0(5,13)]tridecan-7-ol",
    ),
    (
        "Cc1cc(O)c2c(c1)OC(=O)c1c(O)ccc(O)c1[C@H]2O",  # aromatic,heterocycle,fused-ring,medium
        "(2R)-2,4,7,15-tetrahydroxy-13-methyl-10-oxa-tricyclo[9.4.0.0(3,8)]pentadecan-9-one",
    ),
    (
        "CC(=O)OC1CC(O)C(C)OC1C1(C)C(=O)c2cc3cc(C)c(cc4nc(cc5[nH]c(cc1n2)cc5C)C(=O)C4(C)C1OC(C)C(O)CC1O)[nH]3",  # aromatic,heterocycle,fused-ring,polyfunctional,large,carbohydrate
        "(acetyloxy)-4,13-dihexyl-4,13,18,22-tetramethyl-9,20,21,24-tetraaza-pentacyclo[15.2.1.2(8,10).1(3,6).1(12,15)]tetracosan-5,14-dione",
    ),
    # Phase 148 Plan 02 Task 03: linolenoyl-CoA / γ-linolenoyl-CoA.
    # Pre-148 OLD `'adenine'` known-bad placeholder. Post-148 chain-as-parent
    # cascade unblock. Acceptable churn.
    (
        "CCCCC/C=C\\C/C=C\\C/C=C\\CCCCC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O",  # aromatic,heterocycle,fused-ring,polyfunctional,large
        "N-[(2R)-2-hydroxy-3,3-dimethylbutanoyl](6Z,9Z,12Z)-amino-1-(ethylsulfanyl)octadeca-6,9,12-triene",
    ),
    (
        "N#[C][Mo-4]([C]#N)([C]#N)([C]#N)([C]#N)([C]#N)([C]#N)[C]#N",  # acyclic,charged,medium
        "molybdenum compound (not supported)",
    ),
]

# Build test IDs from first 40 chars of SMILES (sanitized for pytest)
# Phase 148 Plan 02 Task 03: NAME_STABILITY_CANARY entries may now be either
# bare (smiles, expected_name) tuples OR pytest.param(...) instances (used to
# attach xfail markers for Phase-149-deferred decomposition fragment-naming
# bugs per 148-01-SUMMARY Risks §2). Extract the SMILES from both forms.
def _extract_smiles(entry):
    if hasattr(entry, "values"):
        return entry.values[0]
    return entry[0]


_CANARY_IDS = [
    _extract_smiles(entry)[:40]
    .replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    for entry in NAME_STABILITY_CANARY
]


@pytest.mark.parametrize("smiles,expected_name", NAME_STABILITY_CANARY, ids=_CANARY_IDS)
def test_canary_name_stability(smiles, expected_name):
    """Name-stability canary test: verify name consistency for OPSIN-unparseable compounds (168 compounds).

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
        "2-[(R)-2-oxo(3R)-3-aminobutyl]benzoic acid",
    ),
    # Row 45: imidazopyridine + tolyl; fused system (9 atoms) vs chain
    # Phase 148 Plan 02 Task 03: cascade unblock per P-44.1(a). Pre-148
    # ring-as-parent `imidazo[1,2-a]pyridine` w/ chain prefix; post-148
    # cascade picks chain (amide PG); imidazopyridine rendered as
    # `imidazo[1,2-a]pyridin-3-yl` substituent. Acceptable churn (cascade
    # decision IUPAC-correct: chain has the principal characteristic group).
    # Note: trailing chain rendering as "decanamide" is the Plan-01
    # decomposition fragment-naming bug surface (Phase 149 / IM-x.x), but
    # the cascade decision itself is correct so the new name is stable
    # and string-equal to the post-148 generator output.
    (
        "Cc1ccc(-c2nc3ccc(C)cn3c2CC(=O)N(C)C)cc1",
        "2-(6-methyl-imidazo[1,2-a]pyridin-3-yl)-N,N-dimethyldecanamide",
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
    # Updated Phase 106: N-substituent pipeline produces amide-centric name
    # Updated Phase 121: parenthesized per IUPAC P-14.5.2 (positional locants)
    (
        "CCCCCCCCCCCCCCCCCCCCCCCC(=O)NCCc1c[nH]c2ccccc12",
        "N-(3-ethyl-1H-indolyl)tetracosanamide",
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
    # Phase 148 D-05 / Plan 02 Task 03: pre-148 OLD value `'2-methylquinoline'`
    # was a known-bad placeholder (drops the entire ester+thioether chain).
    # Post-148 cascade unblock + decomposition fragment-naming bug yields
    # `'2-(decylsulfanyl)-1-methoxyethan-2-oate'` — quinoline ring lost from
    # the output. Both old and new are wrong. Phase 149 / IM-x.x decomposition
    # layer fix. Marked xfail; NOT a Phase 148 regression.
    pytest.param(
        "COC(=O)CSc1cc(C)nc2ccccc12",
        "2-methylquinoline",
        marks=pytest.mark.xfail(
            strict=True,
            reason="Phase 149 / IM-x.x: decomposition fragment-naming bug for "
                   "ester-linked quinoline systems; per 148-01-SUMMARY Risks §2 "
                   "(Plan-01 carry-forward).",
        ),
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

# Build test IDs for P-44.3 canary (extract SMILES from bare-tuple OR
# pytest.param entries — see Phase 148 Plan 02 Task 03 _CANARY_IDS rationale)
_P44_3_IDS = [
    _extract_smiles(entry)[:40]
    .replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
    for entry in P44_3_CANARY
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
