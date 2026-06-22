"""
Retained (trivial) names that are preferred over systematic names.

These names are recognized by IUPAC as Preferred IUPAC Names (PINs).
ALWAYS check this lookup before applying systematic naming rules!

Keys are canonical SMILES, values are retained names.
"""

from typing import Optional

# Common retained names (canonical SMILES -> name)
# These take precedence over systematic names
RETAINED_NAMES = {
    # === SIMPLE ALKANES ===
    "C": "methane",
    "CC": "ethane",
    "CCC": "propane",
    "CCCC": "butane",
    
    # === AROMATIC HYDROCARBONS ===
    "c1ccccc1": "benzene",
    "Cc1ccccc1": "toluene",
    "CCc1ccccc1": "ethylbenzene",
    "C=Cc1ccccc1": "styrene",  # ethenylbenzene
    "CC(C)c1ccccc1": "cumene",  # isopropylbenzene
    # NOTE: xylene isomers are NOT retained names in IUPAC 2013 PIN
    # Use systematic: 1,2-dimethylbenzene, 1,3-dimethylbenzene, 1,4-dimethylbenzene
    "c1ccc2ccccc2c1": "naphthalene",
    "c1cc2ccc3cccc4ccc(c1)c2c34": "pyrene",
    "c1ccc2cc3ccccc3cc2c1": "anthracene",
    "c1ccc2c(c1)ccc1ccccc12": "phenanthrene",
    "c1ccc2c(c1)Cc1ccccc1-2": "fluorene",  # Has sp3 carbon (methylene bridge)
    "c1cc2c3c(cccc3c1)CC2": "acenaphthene",  # Has two sp3 carbons
    "C1=Cc2cccc3cccc1c23": "acenaphthylene",  # Fully aromatic
    "c1ccc2c(c1)ccc1c3ccccc3ccc21": "chrysene",
    "c1cc2cccc3c4cccc5cccc(c(c1)c23)c54": "perylene",
    "c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1": "pentacene",

    # === SIMPLE ALCOHOLS ===
    "CO": "methanol",
    "CCO": "ethanol",
    "OCC(O)CO": "glycerol",
    
    # === CARBOXYLIC ACIDS ===
    "O=CO": "formic acid",  # Canonical SMILES for HCOOH
    "CC(=O)O": "acetic acid",
    "CCC(=O)O": "propanoic acid",  # propionic acid is also accepted
    "CCCC(=O)O": "butanoic acid",  # butyric acid is also accepted
    "OC(=O)C(O)=O": "oxalic acid",
    "OC(=O)CC(=O)O": "malonic acid",
    "OC(=O)CCC(=O)O": "succinic acid",
    "OC(=O)CCCC(=O)O": "glutaric acid",
    "OC(=O)CCCCC(=O)O": "adipic acid",
    "O=C(O)c1ccccc1": "benzoic acid",  # Canonical SMILES (was OC(=O)c1ccccc1)
    "OC(=O)CC(O)(CC(=O)O)C(=O)O": "citric acid",
    
    # === ALDEHYDES ===
    "C=O": "formaldehyde",
    "CC=O": "acetaldehyde",
    "O=Cc1ccccc1": "benzaldehyde",
    
    # === KETONES ===
    # NOTE: acetone removed -- IUPAC 2013 P-31.1.3 PIN is "propan-2-one"
    "CC(=O)c1ccccc1": "acetophenone",
    "O=C(c1ccccc1)c1ccccc1": "benzophenone",
    
    # === AMINES ===
    # P-62.2.1.2 / P-15.2: methylamine / ethylamine / trimethylamine /
    # triethylamine are GENERAL-nomenclature functional-class names, NOT PINs.
    # The PINs are the substitutive forms (methanamine, ethanamine,
    # N,N-dimethylmethanamine, N,N-diethylethanamine), which the systematic
    # path already produces once these retained entries are absent. Removed from
    # the PIN-headline path (DD1 Fix 4 / H5). 'aniline' IS a retained PIN
    # (P-62.2.1.1.1) and stays.
    "Nc1ccccc1": "aniline",
    
    # === PHENOLS ===
    "Oc1ccccc1": "phenol",
    # NOTE: cresol isomers removed -- IUPAC 2013 PINs are "2-methylphenol",
    # "3-methylphenol", "4-methylphenol" (produced by systematic naming pipeline)
    "Oc1ccc(O)cc1": "hydroquinone",
    "Oc1cccc(O)c1": "resorcinol",
    "Oc1ccccc1O": "catechol",
    
    # === 5-MEMBERED AROMATIC HETEROCYCLES ===
    "c1ccoc1": "furan",
    "c1ccsc1": "thiophene",
    # Azoles carry a leading indicated hydrogen in the PIN (P-25.7.1.3): the
    # NH is the indicated-H position, cited as 1H-. Each SMILES key fixes a
    # specific tautomer, so the indicated-H locant is determined per key
    # (all OPSIN-2.9.0 round-trip verified, v23 IH-01).
    "c1cc[nH]c1": "1H-pyrrole",
    "c1c[nH]cn1": "1H-imidazole",
    "c1cnc[nH]1": "1H-imidazole",
    "c1cn[nH]c1": "1H-pyrazole",   # Canonical SMILES for pyrazole
    "c1cc[nH]n1": "1H-pyrazole",   # Alternate input form
    "c1cocn1": "oxazole",       # Canonical SMILES for oxazole
    "c1cnco1": "oxazole",       # Alternate input form
    "c1cnoc1": "isoxazole",     # Canonical SMILES for isoxazole
    "c1ccno1": "isoxazole",     # Alternate input form
    "c1cscn1": "thiazole",      # Canonical SMILES for thiazole
    "c1cncs1": "thiazole",      # Alternate input form
    "c1cnsc1": "isothiazole",   # Canonical SMILES for isothiazole
    "c1ccsn1": "isothiazole",   # Alternate input form
    "c1nnn[nH]1": "1H-tetrazole",  # this tautomer = 1H- (OPSIN-RT verified)
    
    # === 6-MEMBERED AROMATIC HETEROCYCLES ===
    "c1ccncc1": "pyridine",
    "c1ccnnc1": "pyridazine",
    "c1cncnc1": "pyrimidine",
    "c1cnccn1": "pyrazine",
    "c1nncnn1": "1,2,4-triazine",
    "c1ncncn1": "1,3,5-triazine",
    
    # === FUSED HETEROCYCLES ===
    # IUPAC 2013 PIN includes tautomer locant (indicated hydrogen) where applicable
    "c1ccc2ncccc2c1": "quinoline",
    "c1ccc2cnccc2c1": "isoquinoline",
    "c1ccc2[nH]ccc2c1": "1H-indole",  # 1H-indole is IUPAC 2013 PIN
    "c1ccc2[nH]cnc2c1": "1H-benzimidazole",  # 1H-benzimidazole is IUPAC 2013 PIN
    "c1ccc2occc2c1": "benzofuran",
    "c1ccc2sccc2c1": "benzothiophene",
    "c1cnc2ccccc2n1": "quinazoline",
    "c1ccc2nccnc2c1": "quinoxaline",
    "c1ncnc2[nH]cnc12": "7H-purine",  # 7H-purine is IUPAC 2013 PIN
    
    # === UNSATURATED 6-MEMBERED O-HETEROCYCLES (pyrans) ===
    # IUPAC 2013 prefers "2H-pyran" / "4H-pyran" over HW systematic "oxine"
    # OPSIN does not recognize "oxine"; these retained names ensure compatibility
    "C1=CCOC=C1": "2H-pyran",            # 2H-pyran (two C=C bonds)
    "C1=COC=CC1": "4H-pyran",            # 4H-pyran (two C=C bonds)
    "C1=COCCC1": "3,4-dihydro-2H-pyran", # dihydropyran (one C=C bond)
    "C1=CCOCC1": "3,6-dihydro-2H-pyran", # dihydropyran (one C=C bond)

    # === SATURATED HETEROCYCLES ===
    "C1CO1": "oxirane",
    "C1CN1": "aziridine",
    "C1CS1": "thiirane",
    "C1COC1": "oxetane",
    "C1CNC1": "azetidine",
    "C1CSC1": "thietane",
    "C1CCOC1": "oxolane",
    "C1CCNC1": "pyrrolidine",
    "C1CCSC1": "tetrahydrothiophene",
    "C1CCOCC1": "oxane",
    "C1CCNCC1": "piperidine",
    "C1CCSCC1": "thiane",
    "C1COCCN1": "morpholine",
    "C1CNCCN1": "piperazine",
    "O=C1CNCC(=O)N1": "piperazine-2,5-dione",  # Diketopiperazine (IUPAC P-31.1.2)

    # === CYCLOALKANES ===
    "C1CC1": "cyclopropane",
    "C1CCC1": "cyclobutane",
    "C1CCCC1": "cyclopentane",
    "C1CCCCC1": "cyclohexane",
    "C1CCCCCC1": "cycloheptane",
    "C1CCCCCCC1": "cyclooctane",
    
    # === AMIDES ===
    "NC=O": "formamide",
    "CC(=O)N": "acetamide",
    "CC(N)=O": "acetamide",  # canonical form of CC(=O)N

    # === COMMON SOLVENTS AND REAGENTS ===
    "ClCCl": "dichloromethane",
    "ClC(Cl)Cl": "chloroform",
    "ClC(Cl)(Cl)Cl": "carbon tetrachloride",
    "CCOC(C)=O": "ethyl acetate",
    "COC(C)=O": "methyl acetate",
    "CC#N": "acetonitrile",
    "CN(C)C=O": "N,N-dimethylformamide",

    # === SULFUR COMPOUNDS (Phase 10) ===
    # Disulfane (S-S bond, no carbon)
    "SS": "disulfane",
    # Thiols
    "CS": "methanethiol",
    "CCS": "ethanethiol",
    # Sulfides (thioethers)
    "CSC": "dimethyl sulfide",
    "CCSCC": "diethyl sulfide",
    # Sulfoxides (canonical SMILES form)
    "CS(C)=O": "dimethyl sulfoxide",  # DMSO
    # Sulfones (canonical SMILES form)
    "CS(C)(=O)=O": "dimethyl sulfone",
    # Sulfonic acids (canonical SMILES form)
    "CS(=O)(=O)O": "methanesulfonic acid",
    "CCS(=O)(=O)O": "ethanesulfonic acid",
    "O=S(=O)(O)c1ccccc1": "benzenesulfonic acid",

    # === PHOSPHORUS COMPOUNDS (Phase 11) ===
    # Phosphines (use IUPAC 2013 "phosphane" not "phosphine")
    "CP": "methylphosphane",
    "CCP": "ethylphosphane",
    "CP(C)C": "trimethylphosphane",
    "CCP(CC)CC": "triethylphosphane",
    "c1ccc(P(c2ccccc2)c2ccccc2)cc1": "triphenylphosphane",
    # Phosphine oxides
    "CP(C)(C)=O": "trimethylphosphane oxide",
    "CCP(=O)(CC)CC": "triethylphosphane oxide",
    "O=P(c1ccccc1)(c1ccccc1)c1ccccc1": "triphenylphosphane oxide",
    # Phosphonic acids
    "CP(=O)(O)O": "methanephosphonic acid",
    "CCP(=O)(O)O": "ethanephosphonic acid",
    "O=P(O)(O)c1ccccc1": "phenylphosphonic acid",
    # Phosphinic acids
    "CP(C)(=O)O": "dimethylphosphinic acid",
    "CCP(=O)(O)CC": "diethylphosphinic acid",
    # Phosphate esters (functional class naming)
    "COP(=O)(O)O": "methyl phosphate",
    "COP(=O)(O)OC": "dimethyl phosphate",
    "COP(=O)(OC)OC": "trimethyl phosphate",
    "CCOP(=O)(O)O": "ethyl phosphate",
    "CCOP(=O)(O)OCC": "diethyl phosphate",
    "CCOP(=O)(OCC)OCC": "triethyl phosphate",
    
    # === RETAINED NITROGEN COMPOUNDS ===
    "NC(N)=O": "urea",
    "N=C(N)N": "guanidine",

    # === OTHER COMMON COMPOUNDS ===
    "O": "water",
    "N": "ammonia",
    "O=C=O": "carbon dioxide",
    "C#N": "hydrogen cyanide",
    "O=S=O": "sulfur dioxide",
    "N#N": "dinitrogen",
    "O=O": "dioxygen",

    # === INORGANIC ACIDS (Phase 25 fix) ===
    "O=[N+]([O-])O": "nitric acid",
    "O=[N+]([O-])OO": "peroxynitric acid",

    # === NITRILES - AROMATIC (Phase 14.6 BUG-2 fix) ===
    "N#Cc1ccccc1": "benzonitrile",  # C6H5CN - PIN per P-66.1.1.1

    # === AMIDES - AROMATIC (Phase 20 suffix FG fix) ===
    "NC(=O)c1ccccc1": "benzamide",  # C6H5CONH2 - PIN per P-66.1.1.1

    # === THIAZOLIDINES (Phase 14.6 BUG-6 fix) ===
    # 1,3-thiazolidine: S at 1, N at 3 (not adjacent)
    "C1CSCN1": "thiazolidine",
    # 1,2-isothiazolidine: S at 1, N at 2 (adjacent)
    "C1CNSC1": "isothiazolidine",

    # === AMINO ACIDS (common) ===
    "NCC(=O)O": "glycine",
    "CC(N)C(=O)O": "alanine",
    "CC(C)C(N)C(=O)O": "valine",
    "CC(C)CC(N)C(=O)O": "leucine",
    "CCC(C)C(N)C(=O)O": "isoleucine",
    "OC(=O)C(N)Cc1ccccc1": "phenylalanine",
    "NC(Cc1c[nH]c2ccccc12)C(=O)O": "tryptophan",
    "NC(Cc1ccc(O)cc1)C(=O)O": "tyrosine",
    "CSCCC(N)C(=O)O": "methionine",
    "NC(CS)C(=O)O": "cysteine",
    "NC(CC(=O)O)C(=O)O": "aspartic acid",
    "NC(CCC(=O)O)C(=O)O": "glutamic acid",
    "NC(=O)CC(N)C(=O)O": "asparagine",
    "NC(=O)CCC(N)C(=O)O": "glutamine",
    "NCCCCC(N)C(=O)O": "lysine",
    "N=C(N)NCCCC(N)C(=O)O": "arginine",
    "NC(Cc1c[nH]cn1)C(=O)O": "histidine",
    "O=C(O)C1CCCN1": "proline",
    "NC(CO)C(=O)O": "serine",
    "CC(O)C(N)C(=O)O": "threonine",

    # === ADDITIONAL SATURATED HETEROCYCLES (Phase 8 expansion) ===
    "C1COCCO1": "1,4-dioxane",
    "C1CSCCO1": "thiomorpholine",
    "C1CN2CCC1CC2": "quinuclidine",
    "C1CCC2NCCCC2C1": "decahydroquinoline",
    "C1CCN2CCCCC2C1": "decahydroisoquinoline",

    # === FATTY ACIDS (Phase 8 expansion) ===
    "CCCCC(=O)O": "pentanoic acid",
    "CCCCCC(=O)O": "hexanoic acid",
    "CCCCCCCC(=O)O": "octanoic acid",
    "CCCCCCCCCC(=O)O": "decanoic acid",
    "CCCCCCCCCCCC(=O)O": "dodecanoic acid",
    "CCCCCCCCCCCCCC(=O)O": "tetradecanoic acid",
    "CCCCCCCCCCCCCCCC(=O)O": "hexadecanoic acid",
    "CCCCCCCCCCCCCCCCCC(=O)O": "octadecanoic acid",

    # === BRANCHED CARBOXYLIC ACIDS (Phase 8 expansion) ===
    # NOTE: isobutyric acid removed -- PIN is "2-methylpropanoic acid"
    # NOTE: isovaleric acid removed -- PIN is "3-methylbutanoic acid"
    # NOTE: pivalic acid removed -- PIN is "2,2-dimethylpropanoic acid"

    # === UNSATURATED ACIDS (Phase 8 expansion) ===
    "CC=CC(=O)O": "crotonic acid",
    "CC=CC=CC(=O)O": "sorbic acid",

    # === ALDEHYDES (Phase 8 expansion) ===
    # Note: butanal/pentanal preferred over butyraldehyde/valeraldehyde for consistency
    "CC=CC=O": "crotonaldehyde",

    # === KETONES (Phase 8 expansion) ===
    "CC(=O)CC(C)(C)C": "pinacolone",
    "CC(=O)C=C(C)C": "mesityl oxide",

    # === DIOLS AND POLYOLS (Phase 8 expansion) ===
    # Phase 167 HYG-03: ethylene/propylene/trimethylene glycol removed (deprecated
    # "glycol" names, not PINs). Systematic ethane-1,2-diol / propane-1,2-diol /
    # propane-1,3-diol now emitted; also enforced by _PIN_DENY (recurrence guard).
    # See  § "Phase 167".
    "OCCCCO": "butane-1,4-diol",

    # === UNSATURATED ALCOHOLS (Phase 8 expansion) ===
    "C=CCO": "allyl alcohol",
    "C#CCO": "propargyl alcohol",

    # === TERPENES (Phase 8 expansion) ===
    "C=C(C)C1CC=C(C)CC1": "limonene",
    "CC12CCC(CC1=O)C2(C)C": "camphor",

    # === NAPHTHOLS AND BIPHENYLS (Phase 8 expansion) ===
    "Oc1ccc2ccccc2c1": "2-naphthol",
    "Oc1ccc(-c2ccccc2)cc1": "4-phenylphenol",

    # === COMMON PHARMACEUTICALS (Phase 8 expansion) ===
    # Phase 167 HYG-03: "aspirin" (brand name) removed; PIN 2-acetyloxybenzoic
    # acid now emitted; also enforced by _PIN_DENY. See retained_name_conflicts.md.

    # === CYCLIC IMIDES (Phase 49, expanded Phase 91.1) ===
    "O=C1CCC(=O)N1": "succinimide",
    "O=C1C=CC(=O)N1": "maleimide",
    "O=C1CCCC(=O)N1": "glutarimide",
    "O=C1NC(=O)c2ccccc21": "phthalimide",

    # === COMMONLY ENCOUNTERED RETAINED NAMES (Phase 94) ===
    # Source: IUPAC 2013 Blue Book, various sections
    # OPSIN RT verified 2026-03-08
    "c1ccc(-c2ccccc2)cc1": "biphenyl",  # P-31.1.2.4 general nomenclature
    "C#C": "acetylene",  # P-31.1.2.1 PIN for unsubstituted ethyne
    "COc1ccccc1": "anisole",  # P-34.1.1.4 PIN
    "O=C1CCCCCN1": "caprolactam",  # P-31.1.4 retained lactam name

    # === NUCLEOSIDES (Phase 94) ===
    # Retained names per carbohydrate nomenclature conventions
    # Each nucleoside has fixed beta stereochemistry at anomeric position
    # OPSIN RT verified 2026-03-08 (all 8 pass)
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O": "adenosine",
    "Nc1nc2c(ncn2[C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1": "guanosine",
    "Nc1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)n1": "cytidine",
    "Cc1cn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)[nH]c1=O": "thymidine",
    "O=c1ccn([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c(=O)[nH]1": "uridine",
    "Nc1ncnc2c1ncn2[C@H]1C[C@H](O)[C@@H](CO)O1": "deoxyadenosine",
    "Nc1nc2c(ncn2[C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)[nH]1": "deoxyguanosine",
    "Nc1ccn([C@H]2C[C@H](O)[C@@H](CO)O2)c(=O)n1": "deoxycytidine",
    # v22 G2 COV-02: nucleotide retained names re-admitted on the hand-curated
    # side (the adenylic JSON deny entries carry hc_override). These are
    # correct-but-NON-PIN names (the PIN is the full systematic
    # adenosine/inosine 5'-(dihydrogen phosphate)); re-admitting them stops the
    # egregious atom-dropping mis-name AMP→'6-aminopyrimidine' /
    # IMP→'6-oxo-1,3-diazine'. A10 honest-fail-on-data: a correct retained name
    # beats a wrong systematic one when no PIN is computable. The inosinic KETO
    # tautomer is added explicitly (OPSIN normalises the name to the enol form,
    # which already promotes via the OPSIN import). OPSIN-RT verified.
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O": "5'-adenylic acid",
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1OP(=O)(O)O": "2'-adenylic acid",
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](OP(=O)(O)O)[C@H]1O": "3'-adenylic acid",
    "O=c1[nH]cnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O": "5'-inosinic acid",

    # === DISACCHARIDES (Phase 94) ===
    # NOTE: OPSIN cannot parse most disaccharide names -- InChI validation used
    # instead of OPSIN RT. Canonical SMILES from PubChem + RDKit canonicalization.
    "OC[C@@H]1O[C@@](CO)(O[C@H]2[C@H](O)[C@@H](O)[C@@H](O)O[C@@H]2CO)[C@@H](O)[C@H]1O": "sucrose",
    "OC[C@H]1O[C@@H](O[C@H]2[C@H](O)[C@@H](O)C(O)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@@H]1O": "maltose",
    "OC[C@H]1O[C@H](O[C@H]2[C@H](O)[C@@H](O)C(O)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@H]1O": "lactose",
    "OC[C@H]1O[C@@H](O[C@@H]2[C@@H](O)[C@H](O)[C@@H](O)O[C@@H]2CO)[C@H](O)[C@@H](O)[C@@H]1O": "cellobiose",
    "OC[C@H]1O[C@H](O[C@H]2O[C@H](CO)[C@H](O)[C@H](O)[C@H]2O)[C@H](O)[C@H](O)[C@H]1O": "trehalose",

    # === MODIFIED SUGARS (Phase 94) ===
    # N-acetylneuraminic acid (sialic acid / Neu5Ac)
    # Unique 9-carbon structure, not standard pyranose/furanose
    # OPSIN RT verified 2026-03-08
    "CC(=O)N[C@H]1[C@H]([C@H](O)[C@H](O)CO)OC(O)(C(=O)O)C[C@@H]1O": "N-acetylneuraminic acid",

    # === ADDITIONAL POLYCYCLIC AROMATICS (Phase 109 expansion) ===
    # All canonical SMILES verified via Chem.CanonSmiles + OPSIN RT 2026-03-16
    "c1ccc2c(c1)c1ccccc1c1ccccc21": "triphenylene",
    "c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61": "coronene",
    "c1ccc2c(c1)-c1cccc3cccc-2c13": "fluoranthene",

    # === BENZOIC ACID DERIVATIVES (Phase 109 expansion) ===
    # Note: salicylic acid and gallic acid omitted -- existing tests expect
    # systematic names (2-hydroxybenzoic acid, 3,4,5-trihydroxybenzoic acid)
    "O=C(O)c1cccc(C(=O)O)c1": "isophthalic acid",
    "Nc1ccccc1C(=O)O": "anthranilic acid",
    "Nc1ccc(C(=O)O)cc1": "4-aminobenzoic acid",
    "COc1cc(C(=O)O)ccc1O": "vanillic acid",

    # === ANHYDRIDES (Phase 109 expansion) ===
    # Note: acetic/succinic anhydride omitted -- existing tests expect
    # systematic names (ethanoic anhydride, butanedioic anhydride)
    "O=C1C=CC(=O)O1": "maleic anhydride",
    "O=C1OC(=O)c2ccccc21": "phthalic anhydride",

    # === HETEROCYCLE DERIVATIVES (Phase 109 expansion) ===
    "O=c1ccc2ccccc2o1": "coumarin",       # 2H-chromen-2-one
    "O=c1ccoc2ccccc12": "chromone",       # 4H-chromen-4-one
    "O=c1c2ccccc2oc2ccccc12": "xanthone",  # 9H-xanthen-9-one
    "O=C(O)c1cccnc1": "nicotinic acid",   # pyridine-3-carboxylic acid
    "O=C(O)c1ccncc1": "isonicotinic acid",  # pyridine-4-carboxylic acid
    "O=C(O)c1ccccn1": "picolinic acid",   # pyridine-2-carboxylic acid
    "NC(=O)c1cccnc1": "nicotinamide",     # pyridine-3-carboxamide
    "O=C1NS(=O)(=O)c2ccccc21": "saccharin",  # 1,1-dioxo-1,2-benzothiazol-3-one

    # === ADDITIONAL AMINES (Phase 109 expansion) ===
    "NCCCCN": "putrescine",    # butane-1,4-diamine
    "NCCCCCN": "cadaverine",   # pentane-1,5-diamine

    # === ADDITIONAL SOLVENTS (Phase 109 expansion) ===
    "C1COCO1": "1,3-dioxolane",

    # === NUCLEOBASES (Phase 109 expansion) ===
    "O=c1cc[nH]c(=O)[nH]1": "uracil",
    "Cc1c[nH]c(=O)[nH]c1=O": "thymine",
    "Nc1cc[nH]c(=O)n1": "cytosine",
    "Nc1ncnc2[nH]cnc12": "adenine",
    "Nc1nc2[nH]cnc2c(=O)[nH]1": "guanine",
    "Nc1nc(=O)c2[nH]cnc2[nH]1": "guanine",  # alternate tautomer (Phase 142)

    # === PURINE DERIVATIVES (Phase 142 expansion) ===
    "O=c1[nH]c(=O)c2[nH]cnc2[nH]1": "xanthine",  # 3,7-dihydro-1H-purine-2,6-dione
    "O=c1[nH]c(=O)c2nc[nH]c2[nH]1": "xanthine",  # alternate tautomer
    "O=c1[nH]cnc2[nH]cnc12": "hypoxanthine",  # 1,9-dihydro-6H-purine-6-one
    "O=c1[nH]cnc2nc[nH]c12": "hypoxanthine",  # alternate tautomer
    "O=C(O)c1cc(=O)[nH]c(=O)[nH]1": "orotic acid",  # pyrimidine-2,4(1H,3H)-dione-6-carboxylic acid
    "c1ccc2nc3ccccc3cc2c1": "acridine",  # dibenzo[b,e]pyridine

    # === LONG-CHAIN DIACIDS (Phase 109 expansion) ===
    "O=C(O)CCCCCCC(=O)O": "suberic acid",   # octanedioic acid
    "O=C(O)CCCCCCCC(=O)O": "azelaic acid",  # nonanedioic acid
    "O=C(O)CCCCCCCCC(=O)O": "sebacic acid",  # decanedioic acid

    # === AROMATIC DERIVATIVES (Phase 109 expansion) ===
    "c1ccc(Nc2ccccc2)cc1": "diphenylamine",

    # === BENZALDEHYDE DERIVATIVES (Phase 109 expansion) ===
    "COc1cc(C=O)ccc1O": "vanillin",           # 4-hydroxy-3-methoxybenzaldehyde
    "O=Cc1ccc(O)cc1": "4-hydroxybenzaldehyde",
    "COc1ccc(C=O)cc1": "anisaldehyde",        # 4-methoxybenzaldehyde
    "O=Cc1cccnc1": "nicotinaldehyde",         # pyridine-3-carbaldehyde
    "O=Cc1ccncc1": "isonicotinaldehyde",      # pyridine-4-carbaldehyde

    # === NAPHTHOL (Phase 109 expansion) ===
    "Oc1cccc2ccccc12": "1-naphthol",

    # === MISCELLANEOUS AROMATICS (Phase 109 expansion) ===
    "OC(c1ccccc1)c1ccccc1": "benzhydrol",             # diphenylmethanol
    "O=c1cc(-c2ccccc2)oc2ccccc12": "flavone",         # 2-phenyl-4H-chromen-4-one
    "Nc1ccc(N)cc1": "1,4-phenylenediamine",           # benzene-1,4-diamine
    "Oc1cccc(O)c1O": "pyrogallol",                    # benzene-1,2,3-triol
    "CC(=O)c1ccc(O)cc1": "4-hydroxyacetophenone",

    # === ADDITIONAL COMMON COMPOUNDS (Phase 109 expansion) ===
    # All canonical SMILES verified via Chem.CanonSmiles 2026-03-16
    "C1CCC2CCCCC2C1": "decahydronaphthalene",         # decalin
    "O=C(O)c1ccco1": "furan-2-carboxylic acid",       # furoic acid
    "c1ccc(-c2ccncc2)nc1": "2,2'-bipyridine",         # bipyridyl
    "Oc1cc(O)cc(O)c1": "phloroglucinol",              # benzene-1,3,5-triol
    "c1ccc2c(c1)ccc1cccnc12": "benzo[f]quinoline",
    "c1ccc2c(c1)ccc1ncccc12": "benzo[h]quinoline",
    "Oc1ccc2c(c1)OCO2": "sesamol",                    # 3,4-methylenedioxyphenol
    "O=Cc1ccc2c(c1)OCO2": "piperonal",                # 3,4-methylenedioxybenzaldehyde
    "C=CCc1ccc2c(c1)OCO2": "safrole",
    "CC(=O)c1ccco1": "2-acetylfuran",
    # Note: tetracene SMILES was actually benz[a]anthracene (angular) - removed
    "Cc1cc(C)c(O)c(C)c1": "mesitol",                  # 2,4,6-trimethylphenol
    "CC(C)(C)c1ccccc1": "tert-butylbenzene",
    "c1ccc(CCc2ccccc2)cc1": "1,2-diphenylethane",     # bibenzyl
    "C=Cc1ccc(C=C)cc1": "1,4-divinylbenzene",
    "Cc1cc(C)c(C)cc1C": "durene",                     # 1,2,4,5-tetramethylbenzene
    "c1ccc(Cc2ccccc2)cc1": "diphenylmethane",
    "c1ccc(C(c2ccccc2)c2ccccc2)cc1": "triphenylmethane",
    "C1=Cc2ccccc2C1": "1H-indene",
    "c1ccc(SSc2ccccc2)cc1": "diphenyl disulfide",
    "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]": "2,4,6-trinitrotoluene",
    "O=Cc1ccco1": "furfural",                         # furan-2-carbaldehyde
}


def get_retained_name(canonical_smiles: str) -> Optional[str]:
    """
    Get retained name for a canonical SMILES if one exists.

    Phase 150 D-05 + RESEARCH ADDITION 1 root-cause fix: consults the
    merged ALL_RETAINED_NAMES dict via late binding (avoids circular
    import at module load time). Resolves 2 function-import consumers
    in one edit: ring_assemblies.py:296, heterocycles.py:36.

    Args:
        canonical_smiles: Canonical SMILES string (must be canonicalized!)

    Returns:
        Retained name string, or None if not found
    """
    try:
        from orthonym.data import ALL_RETAINED_NAMES
        return ALL_RETAINED_NAMES.get(canonical_smiles)
    except ImportError:
        return RETAINED_NAMES.get(canonical_smiles)


def is_retained_name_compound(canonical_smiles: str) -> bool:
    """
    Check if compound has a retained name.

    Phase 150 D-05 + RESEARCH ADDITION 1 root-cause fix: consults
    merged ALL_RETAINED_NAMES via late binding.

    Args:
        canonical_smiles: Canonical SMILES string

    Returns:
        True if compound has a retained name
    """
    try:
        from orthonym.data import ALL_RETAINED_NAMES
        return canonical_smiles in ALL_RETAINED_NAMES
    except ImportError:
        return canonical_smiles in RETAINED_NAMES


def add_retained_name(canonical_smiles: str, name: str) -> None:
    """DEPRECATED: prefer ``orthonym.data.register_retained_name``.

    Phase 150 D-05 + REVIEW CR-01 root-cause fix: ``ALL_RETAINED_NAMES``
    is built once at import time (``data/__init__.py``); runtime mutators
    must keep the HC dict and the merged dict synchronised so the public
    ``get_retained_name`` lookup sees new entries.

    This wrapper updates BOTH the hand-curated ``RETAINED_NAMES`` dict
    (for legacy callers that read it directly) and delegates to
    ``register_retained_name`` (which mutates ``ALL_RETAINED_NAMES``).

    Args:
        canonical_smiles: Canonical SMILES string
        name: Retained IUPAC name
    """
    RETAINED_NAMES[canonical_smiles] = name
    try:
        from orthonym.data import register_retained_name
        register_retained_name(canonical_smiles, name)
    except ImportError:
        pass
