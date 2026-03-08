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
    "CC(C)=O": "acetone",
    "CC(=O)c1ccccc1": "acetophenone",
    "O=C(c1ccccc1)c1ccccc1": "benzophenone",
    
    # === AMINES ===
    "CN": "methylamine",
    "CCN": "ethylamine",
    "Nc1ccccc1": "aniline",
    "CN(C)C": "trimethylamine",
    "CCN(CC)CC": "triethylamine",
    
    # === PHENOLS ===
    "Oc1ccccc1": "phenol",
    "Cc1ccc(O)cc1": "4-cresol",
    "Cc1ccccc1O": "2-cresol",
    "Cc1cccc(O)c1": "3-cresol",
    "Oc1ccc(O)cc1": "hydroquinone",
    "Oc1cccc(O)c1": "resorcinol",
    "Oc1ccccc1O": "catechol",
    
    # === 5-MEMBERED AROMATIC HETEROCYCLES ===
    "c1ccoc1": "furan",
    "c1ccsc1": "thiophene",
    "c1cc[nH]c1": "pyrrole",
    "c1c[nH]cn1": "imidazole",
    "c1cnc[nH]1": "imidazole",
    "c1cn[nH]c1": "pyrazole",   # Canonical SMILES for pyrazole
    "c1cc[nH]n1": "pyrazole",   # Alternate input form
    "c1cocn1": "oxazole",       # Canonical SMILES for oxazole
    "c1cnco1": "oxazole",       # Alternate input form
    "c1cnoc1": "isoxazole",     # Canonical SMILES for isoxazole
    "c1ccno1": "isoxazole",     # Alternate input form
    "c1cscn1": "thiazole",      # Canonical SMILES for thiazole
    "c1cncs1": "thiazole",      # Alternate input form
    "c1cnsc1": "isothiazole",   # Canonical SMILES for isothiazole
    "c1ccsn1": "isothiazole",   # Alternate input form
    "c1nnn[nH]1": "tetrazole",
    
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
    "CC(C)C(=O)O": "isobutyric acid",
    "CC(C)CC(=O)O": "isovaleric acid",
    "CC(C)(C)C(=O)O": "pivalic acid",

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
    "OCCO": "ethylene glycol",
    "CC(O)CO": "propylene glycol",
    "OCCCO": "trimethylene glycol",
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
    "CC(=O)Oc1ccccc1C(=O)O": "aspirin",

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
}


def get_retained_name(canonical_smiles: str) -> Optional[str]:
    """
    Get retained name for a canonical SMILES if one exists.
    
    Args:
        canonical_smiles: Canonical SMILES string (must be canonicalized!)
        
    Returns:
        Retained name string, or None if not found
    """
    return RETAINED_NAMES.get(canonical_smiles)


def is_retained_name_compound(canonical_smiles: str) -> bool:
    """
    Check if compound has a retained name.
    
    Args:
        canonical_smiles: Canonical SMILES string
        
    Returns:
        True if compound has a retained name
    """
    return canonical_smiles in RETAINED_NAMES


def add_retained_name(canonical_smiles: str, name: str) -> None:
    """
    Add a new retained name to the lookup.
    
    Args:
        canonical_smiles: Canonical SMILES string
        name: Retained IUPAC name
    """
    RETAINED_NAMES[canonical_smiles] = name
