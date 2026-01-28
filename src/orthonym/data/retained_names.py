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
    "OC(=O)c1ccccc1": "benzoic acid",
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
    "c1ccc2ncccc2c1": "quinoline",
    "c1ccc2cnccc2c1": "isoquinoline",
    "c1ccc2[nH]ccc2c1": "indole",
    "c1ccc2[nH]cnc2c1": "benzimidazole",
    "c1ccc2occc2c1": "benzofuran",
    "c1ccc2sccc2c1": "benzothiophene",
    "c1cnc2ccccc2n1": "quinazoline",
    "c1ccc2nccnc2c1": "quinoxaline",
    "c1ncnc2[nH]cnc12": "purine",
    
    # === SATURATED HETEROCYCLES ===
    "C1CO1": "oxirane",
    "C1CN1": "aziridine",
    "C1CS1": "thiirane",
    "C1COC1": "oxetane",
    "C1CNC1": "azetidine",
    "C1CSC1": "thietane",
    "C1CCOC1": "tetrahydrofuran",
    "C1CCNC1": "pyrrolidine",
    "C1CCSC1": "tetrahydrothiophene",
    "C1CCOCC1": "tetrahydropyran",
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
    
    # === COMMON SOLVENTS AND REAGENTS ===
    "ClCCl": "dichloromethane",
    "ClC(Cl)Cl": "chloroform",
    "ClC(Cl)(Cl)Cl": "carbon tetrachloride",
    "CCOC(C)=O": "ethyl acetate",
    "COC(C)=O": "methyl acetate",
    "CC#N": "acetonitrile",
    "CS(C)=O": "dimethyl sulfoxide",
    "CN(C)C=O": "N,N-dimethylformamide",
    
    # === OTHER COMMON COMPOUNDS ===
    "O": "water",
    "N": "ammonia",
    "O=C=O": "carbon dioxide",
    "C#N": "hydrogen cyanide",
    "O=S=O": "sulfur dioxide",
    "N#N": "dinitrogen",
    "O=O": "dioxygen",
    
    # === AMINO ACIDS (common) ===
    "NCC(=O)O": "glycine",
    "CC(N)C(=O)O": "alanine",
    "CC(C)C(N)C(=O)O": "valine",
    "CC(C)CC(N)C(=O)O": "leucine",
    "CCC(C)C(N)C(=O)O": "isoleucine",
    "OC(=O)C(N)Cc1ccccc1": "phenylalanine",
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
