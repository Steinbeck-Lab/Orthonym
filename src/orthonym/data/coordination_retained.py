"""
Retained-name table for N-coordinated metal-tetrapyrrole macrocycles.

 Milestone D1. These are Fe/Mg/Co/Ni tetrapyrrole coordination complexes
(heme, chlorophyll, cobalamin, siroheme, coenzyme F430). OPSIN 2.9.0 cannot
parse any of their names (measured: heme b / chlorophyll a / cyanocobalamin /
siroheme / F430 all -> opsin_parse None), so a round-trip oracle is impossible
for this class. The 0-wrong guarantee here is therefore CURATION PROVENANCE +
EXACT-InChIKey identity, exactly the contract already used by the amino-acid /
sugar / metallocene retained tables:

  * Each key is the standard InChIKey of one exact ChEBI structure.
  * Each value is that structure's ChEBI-accepted IUPAC/name annotation,
    taken VERBATIM from the ChEBI data (chebi_iupac_filtered.tsv / ChEBI id
    in the per-entry comment) -- never guessed, hand-derived, or approximated.
  * The lookup fires ONLY on an exact InChIKey match, and ONLY on the
    descriptive-fallback path (namer._descriptive_fallback), i.e. after every
    real namer has already declined. A miss stays abstaining (still 0-wrong);
    a molecule the normal namer names never reaches this table.

Charge / protonation states are enumerated as SEPARATE keys (ChEBI stores one
row per charge state; standard InChI does not renormalise formal charge), so
each key is an exact, unambiguous structural match.

Scope (-D1 plan IN-list): heme b/c/o, siroheme, chlorophyll/bacteriochlorophyll
family, the cobalamin group (cyano/adenosyl/methyl/aqua/hydroxo/nitrito/cob(I..III)),
and coenzyme F430. -D1 additionally added the 13 cobyrinic-acid-type corrinoid /
cobalt-precorrin rows (Co corrin/precorrin, no nucleotide loop) that each passed a strict
per-row provenance check (verbatim ChEBI name, InChIKey re-derived == key, distinct id,
opsin_parse == None); see internal notes SP5.1 addendum. Still OUT
(left abstaining): chlorophyllide/protochlorophyllide precursors, Zn/Cu porphyrin dyes,
and the general additive namer (D2). Rows whose ChEBI name is malformed
(unsigned charge, empty parentheses, stray double-hyphen) or whose structure has
no standard InChIKey (dative-bond SMILES) are also left out -- they stay abstaining.
"""

from typing import Dict

# InChIKey (standard, from Chem.MolToInchiKey) -> ChEBI-accepted name (verbatim).
COORDINATION_RETAINED: Dict[str, str] = {
    # === Cobalamin family (corrin core + 5,6-dimethylbenzimidazole nucleotide loop; vitamin B12 group) ===
    "OMAOKVYASDIYQG-DSRCUDDDSA-M": "cob(I)alamin",
    # ^ CHEBI:15982 -- cob(I)alamin
    "ASARMUCNOOHMLO-DSRCUDDDSA-L": "cob(II)alamin",
    # ^ CHEBI:16304 -- cob(II)alamin
    "NSLAUEAQDBERRV-DSRCUDDDSA-L": "cob(III)alamin",
    # ^ CHEBI:28911 -- cob(III)alamin
    "YOZNUFWCRFCGIH-WZHZPDAFSA-L": "Coalpha-[alpha-(5,6-dimethylbenzimidazolyl)]-Cobeta-aquacobamide",
    # ^ CHEBI:15852 -- aquacobalamin / vitamin B12a
    "YOZNUFWCRFCGIH-WZHZPDAFSA-K": "Coalpha-[alpha-(5,6-dimethylbenzimidazolyl)]-Cobeta-hydroxocobamide",
    # ^ CHEBI:27786 -- hydroxocobalamin / vitamin B12b
    "JEWJRMKHSMTXPP-WZHZPDAFSA-L": "Coalpha-[alpha-(5,6-dimethylbenzimidazolyl)]-Cobeta-methylcobamide",
    # ^ CHEBI:28115 -- methylcobalamin
    "RMRCNWBMXRMIRW-WZHZPDAFSA-L": "cyanocob(III)alamin",
    # ^ CHEBI:17439 -- cyanocobalamin / vitamin B12
    "UUWYBLVKLIHDAU-WZHZPDAFSA-K": "Coalpha-[alpha-(5,6-dimethylbenzimidazolyl)]-Cobeta-nitritocobamide",
    # ^ CHEBI:30529 -- nitritocobalamin
    "ZIHHMGTYZOSFRC-OUCXYWSSSA-L": "Coalpha-[alpha-(5,6-dimethylbenzimidazolyl)]-Cobeta-(5'-deoxy-5'-adenosyl)cobamide",
    # ^ CHEBI:18408 -- adenosylcobalamin / coenzyme B12

    # === Heme / siroheme family (Fe porphyrin / isobacteriochlorin macrocycles) ===
    "KABFMIBPWCXCRK-UHFFFAOYSA-L": "(protoporphyrinato)iron(II)",
    # ^ CHEBI:17627 -- heme b / protoheme (Fe(II) protoporphyrin IX)
    "DLKSSIHHLYNIKN-QIISWYHFSA-L": "[3,3',3'',3'''-[(7S,8S,12S,13S)-3,8,13,17-tetrakis(carboxymethyl)-8,13-dimethyl-7,8,12,13-tetrahydroporphyrin-2,7,12,18-tetrayl-kappaN(21),kappaN(22),kappaN(23),kappaN(24)]tetrapropanoato(2-)]iron",
    # ^ CHEBI:28599 -- siroheme (2- form)
    "DLKSSIHHLYNIKN-QIISWYHFSA-D": "[3,3',3'',3'''-[(7S,8S,12S,13S)-3,8,13,17-tetrakis(carboxymethyl)-8,13-dimethyl-7,8,12,13-tetrahydroporphyrin-2,7,12,18-tetrayl-kappaN(21),kappaN(22),kappaN(23),kappaN(24)]tetrapropanoato(10-)]ferrate(6-)",
    # ^ CHEBI:60052 -- siroheme (fully-deprotonated ferrate charge variant)
    "XSWPXBWSKQRBRZ-UHFFFAOYSA-L": "{3,3'-[3,7,12,17-tetramethyl-8,13-bis(1-sulfanylethyl)porphyrin-2,18-diyl-kappa(4)N(21),N(22),N(23),N(24)]dipropanoato(2-)}iron",
    # ^ CHEBI:60562 -- heme c (bis-thioether)
    "FISPASSVCDRERW-SCCPKUNWSA-J": "[3,3'-{7-ethenyl-12-[(1S,4E,8E)-1-hydroxy-5,9,13-trimethyltetradeca-4,8,12-trien-1-yl]-3,8,13,17-tetramethylporphyrin-2,18-diyl-kappa(4)N(21),N(22),N(23),N(24)}dipropanoato(4-)]iron",
    # ^ CHEBI:60530 -- heme o (hydroxyethylfarnesyl; tetraanion)

    # === Coenzyme F430 (Ni hydrocorphinoid; methyl-coenzyme M reductase cofactor) ===
    "XLFIRMYGVLUNOY-SXMZNAGASA-M": "{3,3',3''-[5-(2-amino-2-oxoethyl)-18,29-bis(carboxymethyl)-5,23-dimethyl-14,25-dioxo-9,26,27,28,30-pentaazaheptacyclo[19.5.1.1(3,6).1(8,11).1(16,19).0(1,23).0(10,15)]triaconta-6(30),9,15,19,21(27)-pentaene-4,17,22-triyl-kappa(4)N(9),N(27),N(28),N(30)]tripropanoato}nickel",
    # ^ CHEBI:28265 -- coenzyme F430 (Ni hydrocorphinoid)

    # === Chlorophyll / bacteriochlorophyll family (Mg porphyrin/chlorin macrocycles) ===
    "ATNHDLDRLWWWCB-AENOIHSZSA-M": "[(2E,7R,11R)-3,7,11,15-tetramethylhexadec-2-en-1-yl (2(2)R,17S,18S)-7-ethyl-2(1),2(2),17,18-tetrahydro-2(2)-(methoxycarbonyl)-3,8,13,17-tetramethyl-2(1)-oxo-12-ethenylcyclopenta[at]porphyrin-18-propanoato(2-)]magnesium",
    # ^ CHEBI:18230 -- chlorophyll a
    "DLGTYWBJNMHIPY-JYVCQAQSSA-M": "[methyl (3S,4S,21R)-14-ethyl-4,8,13,18-tetramethyl-20-oxo-3-(3-oxo-3-{[(2E,6E,10E)-3,7,11,15-tetramethylhexadeca-2,6,10,14-tetraen-1-yl]oxy}propyl)-9-vinylphorbine-21-carboxylatato(2-)-kappa(4)N(23),N(24),N(25),N(26)]magnesium",
    # ^ CHEBI:64668 -- chlorophyll b / geranylgeranyl chlorophyll family
    "ZNGSRZUYGXSJBD-ONWAGYJKSA-M": "[methyl (3S,4S)-4,8,13,18-tetramethyl-20-oxo-3-(3-oxo-3-{[(2E,7R,11R)-3,7,11,15-tetramethylhexadec-2-en-1-yl]oxy}propyl)-9,14-divinylphorbine-21-carboxylatato(3-)-kappa(4)N(23),N(24),N(25),N(26)]magnesate(1-)",
    # ^ CHEBI:73095 -- chlorophyll-family magnesate charge variant
    "COQGSKNKQZLZEJ-AENOIHSZSA-M": "[methyl (3S,4S,21R)-4,8,13,18-tetramethyl-20-oxo-3-(3-oxo-3-{[(2E,7R,11R)-3,7,11,15-tetramethylhexadec-2-en-1-yl]oxy}propyl)-9,14-divinylphorbine-21-carboxylatato(2-)-kappa(4)N(23),N(24),N(25),N(26)]magnesium",
    # ^ CHEBI:73113 -- chlorophyll-family (divinyl phytyl)
    "DSJXIQQMORJERS-AGGZHOMASA-M": "[methyl (3S,4S,13R,14R,21R)-9-acetyl-14-ethyl-4,8,13,18-tetramethyl-20-oxo-3-(3-oxo-3-{[(2E,7R,11R)-3,7,11,15-tetramethylhexadec-2-en-1-yl]oxy}propyl)-13,14-dihydrophorbine-21-carboxylatato(2-)-kappa4N(23),N(24),N(25),N(26)]magnesium",
    # ^ CHEBI:30033 -- bacteriochlorophyll a family
    "QOSUYSWYVJFCJO-VMLNTYRVSA-M": "[methyl (3S,4S,13R,14R)-9-acetyl-14-ethyl-4,8,13,18-tetramethyl-20-oxo-3-(3-oxo-3-{[(2E,7R,11R)-3,7,11,15-tetramethylhexadec-2-en-1-yl]oxy}propyl)-13,14-dihydrophorbine-21-carboxylatato(3-)-kappa4N(23),N(24),N(25),N(26)]magnesate(1-)",
    # ^ CHEBI:61720 -- bacteriochlorophyll-family magnesate charge variant
    "WMNTZEAJBUMETR-YKKLGNEQSA-M": "[methyl (3S,4S)-9-ethenyl-14-ethyl-13-formyl-4,8,18-trimethyl-20-oxo-3-(3-oxo-3-{[(2E,7R,11R)-3,7,11,15-tetramethylhexadec-2-en-1-yl]oxy}propyl)phorbine-21-carboxylatato(3-)-kappaN(23),kappa(4)N(24),kappaN(25),kappaN(26)]magnesate(1-)",
    # ^ CHEBI:61721 -- chlorophyll-family magnesate charge variant
    "XCSMCRBINIVJRP-YKKLGNEQSA-M": "[methyl (3S,4S)-13-formyl-4,8,18-trimethyl-20-oxo-3-(3-oxo-3-{[(2E,7R,11R)-3,7,11,15-tetramethylhexadec-2-en-1-yl]oxy}propyl)-9,14-divinylphorbine-21-carboxylatato(3-)-kappa(4)N(23),N(24),N(25),N(26)]magnesate(1-)",
    # ^ CHEBI:73096 -- chlorophyll-family magnesate charge variant
    "QGHLDVDZKNSEFX-VBYMZDBQSA-M": "[methyl (3S,4S,21R)-14-ethyl-13-(hydroxymethyl)-4,8,18-trimethyl-20-oxo-3-(3-oxo-3-{[(2E,7R,11R)-3,7,11,15-tetramethylhexadec-2-en-1-yl]oxy}propyl)-9-vinylphorbine-21-carboxylatato(2-)-kappa(4)N(23),N(24),N(25),N(26)]magnesium",
    # ^ CHEBI:76032 -- chlorophyll-family (hydroxymethyl)
    "SLHJCOLVGKWTAP-KHPXXVTNSA-M": "[methyl (3S,4S,13R,14R)-9-acetyl-14-ethyl-4,8,13,18-tetramethyl-20-oxo-3-(3-oxo-3-{[(2E,6E,10E)-3,7,11,15-tetramethylhexadeca-2,6,10,14-tetraen-1-yl]oxy}propyl)-13,14-dihydrophorbine-21-carboxylatato(3-)-kappa(4)N(23),N(24),N(25),N(26)]magnesate(1-)",
    # ^ CHEBI:90849 -- bacteriochlorophyll-family magnesate charge variant

    # === Cobyrinic-acid-type corrinoid precursors (Co corrin/precorrin, no nucleotide loop; -D1) ===
    "BKIWSQUNFCJSOI-LQRHGLAMSA-E": "{3,3',3'',3'''-[(7S,8S,12S,13S)-3,8,13,17-tetrakis(carboxymethyl)-8,13-dimethyl-3,7,8,12,13,20-hexahydroporphyrin-2,7,12,18-tetrayl-kappa(4)N(21),N(22),N(23),N(24)]tetrapropanoato(10-)}cobaltate(6-)",
    # ^ CHEBI:60053 -- cobalt-precorrin (hexahydroporphyrin, cobaltate 6- charge variant)
    "BKIWSQUNFCJSOI-LQRHGLAMSA-M": "{3,3',3'',3'''-[(7S,8S,12S,13S)-3,8,13,17-tetrakis(carboxymethyl)-8,13-dimethyl-3,7,8,12,13,20-hexahydroporphyrin-2,7,12,18-tetrayl-kappa(4)N(21),N(22),N(23),N(24)]tetrapropanoato(2-)}cobalt",
    # ^ CHEBI:3790 -- cobalt-precorrin (hexahydroporphyrin, cobalt form)
    "DFFFCFUPOVLDTP-IICGDJHVSA-M": "3,3',3'',3'''-{[(1R,2S,3S,7S,11S,17R,18R,19R)-2,7,12,18-tetrakis(carboxymethyl)-1,2,7,11,17-pentamethyl-18,19-didehydrocorrin-3,8,13,17-tetrayl-kappa(4)N(21),N(22),N(23),N(24)]tetrapropanoato(2-)}cobalt",
    # ^ CHEBI:3794 -- cobalt-dihydro-hexamethyl corrin (didehydrocorrin)
    "FKTVLCPLZMVWHD-QOMRAJGQSA-E": "{3,3',3'',3'''-[(7S,8S,12S,13S)-3,8,13,17-tetrakis(carboxymethyl)-8,13,15-trimethyl-3,7,8,12,13,20-hexahydroporphyrin-2,7,12,18-tetrayl-kappa(4)N(21),N(22),N(23),N(24)]tetrapropanoato(10-)}cobaltate(6-)",
    # ^ CHEBI:60060 -- cobalt-precorrin (15-methyl hexahydroporphyrin, cobaltate 6- variant)
    "FKTVLCPLZMVWHD-QOMRAJGQSA-M": "{3,3',3'',3'''-[(7S,8S,12S,13S)-3,8,13,17-tetrakis(carboxymethyl)-8,13,15-trimethyl-3,7,8,12,13,20-hexahydroporphyrin-2,7,12,18-tetrayl-kappa(4)N(21),N(22),N(23),N(24)]tetrapropanoato(2-)}cobalt",
    # ^ CHEBI:3791 -- cobalt-precorrin (15-methyl hexahydroporphyrin, cobalt form)
    "PWLXSVIETQPKMK-WTEINHRPSA-M": "3,3',3'',3'''-{[(1R,2S,3S,7S,11S,17R,18R,19R)-2,7,18-tris(carboxymethyl)-1,2,5,7,11,12,15,17-octamethylcorrin-3,8,13,17-tetrayl-kappa(4)N(21),N(22),N(23),N(24)]tetrapropanoato(2-)}cobalt",
    # ^ CHEBI:3795 -- cobalt octamethylcorrin (corrinoid precursor)
    "RFBIUXAOZAPWCC-RDKWKEIWSA-M": "3,3',3'',3'''-{[(1R,2S,3S,7S,11S,17R,18R,19R)-2,7,12,18-tetrakis(carboxymethyl)-1,2,7,11,17-pentamethylcorrin-3,8,13,17-tetrayl-kappa(4)N(21),N(22),N(23),N(24)]tetrapropanoato(2-)}cobalt",
    # ^ CHEBI:3789 -- cobalt pentamethylcorrin (cobyrinic-acid-type corrinoid)
    "RQHZZQLPDALGEQ-CYGMIEPJSA-D": "{3,3',3'',3'''-[(7S,8S,12S,13S)-3,8,13,17-tetrakis(carboxymethyl)-8,13,15-trimethyl-7,8,12,13-tetrahydroporphyrin-2,7,12,18-tetrayl-kappa(4)N(21),N(22),N(23),N(24)]tetrapropanoato(10-)}cobaltate(8-)",
    # ^ CHEBI:73299 -- cobalt-precorrin (15-methyl tetrahydroporphyrin, cobaltate 8- variant)
    "RQHZZQLPDALGEQ-CYGMIEPJSA-L": "{3,3',3'',3'''-[(7S,8S,12S,13S)-3,8,13,17-tetrakis(carboxymethyl)-8,13,15-trimethyl-7,8,12,13-tetrahydroporphyrin-2,7,12,18-tetrayl-kappa(4)N(21),N(22),N(23),N(24)]tetrapropanoato(2-)}cobalt",
    # ^ CHEBI:73373 -- cobalt-precorrin (15-methyl tetrahydroporphyrin, cobalt form)
    "VHHGJROBFDFFAE-AJSUAYSOSA-F": "3,3',3'',3'''-{[(2S,3S,7S,11S,17R)-1-(1-hydroxyethyl)-2,7,12,18-tetrakis(carboxymethyl)-2,7,17-trimethyl-18,19-didehydrocorrin-3,8,13,17-kappa(4)N(21),N(22),N(23),N(24)]tetrapropanoato(9-)}cobaltate(5-) 1(1),2(1)-delta-lactone",
    # ^ CHEBI:60061 -- cobalt hydroxyethyl-didehydrocorrin delta-lactone (cobaltate 5-)
    "XHLLHKIBBMZXKO-LWEGWLRDSA-L": "3,3',3'',3'''-{[(2S,3S,7S,11S,17R,18R)-2,7,12,18-tetrakis(carboxymethyl)-2,7,11,17-tetramethyl-1,19-didehydrocorrin-3,8,13,17-tetrayl-kappa(4)N(21),N(22),N(23),N(24)]tetrapropanoato(2-)}cobalt",
    # ^ CHEBI:52489 -- cobalt 1,19-didehydrocorrin (corrinoid precursor)
    "XZMXJYDTAININL-QIISWYHFSA-D": "{3,3',3'',3'''-[(7S,8S,12S,13S)-3,8,13,17-tetrakis(carboxymethyl)-8,13-dimethyl-7,8,12,13-tetrahydroporphyrin-2,7,12,18-tetrayl-kappa(4)N(21),N(22),N(23),N(24)]tetrapropanoato(10-)}cobalt(6-)",
    # ^ CHEBI:60049 -- cobalt-precorrin (tetrahydroporphyrin, cobalt 6- variant)
    "XZMXJYDTAININL-QIISWYHFSA-L": "{3,3',3'',3'''-[(7S,8S,12S,13S)-3,8,13,17-tetrakis(carboxymethyl)-8,13-dimethyl-7,8,12,13-tetrahydroporphyrin-2,7,12,18-tetrayl-kappa(4)N(21),N(22),N(23),N(24)]tetrapropanoato(2-)}cobalt",
    # ^ CHEBI:52491 -- cobalt-precorrin (tetrahydroporphyrin, cobalt form)
}

