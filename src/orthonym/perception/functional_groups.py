"""
Functional group detection using SMARTS patterns.

Groups are ordered by IUPAC seniority (highest priority first).
The principal group (highest seniority) becomes the suffix;
all others become prefixes.
"""

from collections import defaultdict
from typing import Dict, List, Tuple
from rdkit import Chem


# SMARTS patterns ordered by IUPAC seniority (P-41 to P-43)
# First match = highest priority = principal group
FUNCTIONAL_GROUP_SMARTS = {
    # === ACIDS (highest priority) ===
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    # Thiocarboxylic acids (IUPAC P-65.3) -- rank just below carboxylic acid
    "thioic_S_acid": "[CX3](=O)[SX2H1]",    # R-C(=O)-SH -> thioic S-acid
    "thioic_O_acid": "[CX3](=S)[OX2H1]",    # R-C(=S)-OH -> thioic O-acid
    "dithioic_acid": "[CX3](=S)[SX2H1]",    # R-C(=S)-SH -> dithioic acid
    # Phase 163 Tier FRN-A: chalcogen-on-acid (P-65.3) -- additive per CONTEXT D-08
    "selenoic_Se_acid": "[CX3](=O)[SeX2H1]",     # R-C(=O)-SeH (P-65.3; AUDIT-FRN § 2)
    "selenoic_O_acid": "[CX3](=[SeX1])[OX2H1]",  # R-C(=Se)-OH (P-65.3; AUDIT-FRN § 2)
    "diselenoic_acid": "[CX3](=[SeX1])[SeX2H1]", # R-C(=Se)-SeH (P-65.3; AUDIT-FRN § 2)
    "telluroic_Te_acid": "[CX3](=O)[TeX2H1]",    # R-C(=O)-TeH (P-65.3 parallel; AUDIT-FRN § 2)
    "telluroic_O_acid": "[CX3](=[TeX1])[OX2H1]", # R-C(=Te)-OH (P-65.3 parallel; AUDIT-FRN § 2)
    "ditelluroic_acid": "[CX3](=[TeX1])[TeX2H1]",# R-C(=Te)-TeH (P-65.3 parallel; AUDIT-FRN § 2)
    # Carbamic acid (IUPAC P-65.2.3): N-C(=O)-OH (free acid, not ester)
    "carbamic_acid": "[NX3][CX3](=O)[OX2H1]",  # R2N-C(=O)-OH -> carbamic acid
    "sulfonic_acid": "[SX4](=O)(=O)[OX2H1]",
    "sulfinic_acid": "[SX3](=O)[OX2H1]",
    "sulfenic_acid": "[SX2]([OX2H])[#6]",  # DATA-05d: IUPAC P-65.3.1.4 R-S-OH
    "phosphonic_acid": "[PX4](=O)([OX2H1])[OX2H1]",
    # Phosphinic acid: R2P(=O)(OH) - two C attached to P
    "phosphinic_acid": "[PX4](=O)([OX2H1])([#6])[#6]",
    
    # === ACID DERIVATIVES ===
    "anhydride": "[CX3](=O)[OX2][CX3](=O)",
    "ester": "[CX3](=O)[OX2][#6]",
    # Phase 163 Tier FRN-D: iminoester / imidate (P-65.1.7) -- additive per CONTEXT D-08
    # AUDIT DECISION (AUDIT-FRN § 2.4): [NX2H1] only (=NH form); N-substituted iminoesters
    # (R-C(=NR')-O-R'') deferred to Phase 163.1 per Open Question 4. Free imidic acid form
    # (R-C(=NH)-OH) deferred per CONTEXT line 120. Cyclic imidates deferred per RESEARCH §5.4.
    "iminoester": "[CX3](=[NX2H1])[OX2][#6]",     # R-C(=NH)-O-R' (P-65.1.7; "alkyl alkanimidate")
    "thioester": "[CX3](=O)[SX2][#6]",
    "acid_chloride": "[CX3](=O)[Cl]",
    "acid_bromide": "[CX3](=O)[Br]",
    "acid_fluoride": "[CX3](=O)[F]",
    "acid_iodide": "[CX3](=O)[I]",  # DATA-04: IUPAC P-65.5.1

    # === NITROGEN ACID DERIVATIVES ===
    "primary_amide": "[CX3](=O)[NX3H2]",
    "secondary_amide": "[CX3](=O)[NX3H1][#6]",
    "tertiary_amide": "[CX3](=O)[NX3]([#6])[#6]",
    "hydrazide": "[CX3](=O)[NX3][NX3]",
    "imide": "[CX3](=O)[NX3][CX3](=O)",
    # Phase 163 Tier FRN-B: chalcogen-on-amide (P-66.1.4.1.1 + P-66.6.3) -- additive per CONTEXT D-08
    # AUDIT DECISION (AUDIT-FRN § 2.2): single-permissive [NX3] (NOT 3-way primary/secondary/tertiary
    # split) captures all 3 N-substitution levels per Open Question 2 + RESEARCH §3.2 line 265.
    "thioamide": "[CX3](=S)[NX3]",                # R-C(=S)-N(H,R) (P-66.1.4.1.1)
    "selenoamide": "[CX3](=[SeX1])[NX3]",         # R-C(=Se)-N(H,R) (P-66.6.3)
    "telluroamide": "[CX3](=[TeX1])[NX3]",        # R-C(=Te)-N(H,R) (P-66.6.3 parallel)

    # === SULFONAMIDES ===
    "primary_sulfonamide": "[SX4](=O)(=O)[NX3H2]",
    "secondary_sulfonamide": "[SX4](=O)(=O)[NX3H1][#6]",
    "tertiary_sulfonamide": "[SX4](=O)(=O)[NX3]([#6])[#6]",
    
    # === CARBAMATES (must check before esters -- N-C(=O)-O is more specific) ===
    "carbamate": "[NX3][CX3](=O)[OX2][#6]",

    # === UREA (must check before amides -- N-C(=O)-N is more specific) ===
    "urea": "[NX3][CX3](=O)[NX3]",

    # === GUANIDINE (must check before imines -- N-C(=N)-N is more specific) ===
    "guanidine": "[NX3][CX3](=[NX2])[NX3]",

    # === AMIDINE (IUPAC P-66.4.1: C(=NH)NH2, less specific than guanidine) ===
    "amidine": "[CX3](=[NX2H])[NX3H2]",  # DATA-03

    # === ISOCYANATES/ISOTHIOCYANATES (cumulated double bonds) ===
    "isocyanate": "[#6][NX2]=[CX2]=[OX1]",
    "isothiocyanate": "[#6][NX2]=[CX2]=[SX1]",

    # === N-OXIDES ===
    "n_oxide_aromatic": "[n+][O-]",
    "n_oxide_aliphatic": "[NX4+]([#6])([#6])([#6])[O-]",

    # === BORONIC ACIDS ===
    "boronic_acid": "[#6][BX3]([OX2H])([OX2H])",

    # === NITRILES ===
    "nitrile": "[CX2]#[NX1]",
    "isocyanide": "[#6][NX2]#[CX1]",
    
    # === CARBONYLS ===
    # Aldehyde: carbonyl C with 1 or 2 H (PERC-01: H2 added for formaldehyde;
    # collision resolution suppresses false positives on amides/acids)
    "aldehyde": "[CX3;H1,H2](=O)",
    "ketone": "[#6][CX3](=O)[#6]",
    "thioaldehyde": "[CX3H1](=S)",
    "thioketone": "[#6][CX3](=S)[#6]",
    # Phase 163 Tier FRN-C: chalcogen-on-aldehyde/ketone (P-66.6.3) -- additive per CONTEXT D-08
    "selenoaldehyde": "[CX3H1](=[SeX1])",         # R-C(=Se)H (P-66.6.3)
    "telluroaldehyde": "[CX3H1](=[TeX1])",        # R-C(=Te)H (P-66.6.3 parallel)
    "selenoketone": "[#6][CX3](=[SeX1])[#6]",     # R-C(=Se)-R' (P-66.6.3); selone PIN suffix form
    "telluroketone": "[#6][CX3](=[TeX1])[#6]",    # R-C(=Te)-R' (P-66.6.3 parallel); tellone PIN suffix form

    # === ALCOHOLS AND ANALOGS ===
    # Generic catch-all: any OH on sp3 carbon (IUPAC P-63.1)
    # Complements specific sub-type patterns below; ensures detection of
    # OH on carbons with non-carbon neighbors (halogens, nitrogen, sulfur)
    "alcohol": "[OX2H][CX4]",  # PERC-05: generic catch-all per D-01
    "primary_alcohol": "[OX2H][CX4H2]",
    "secondary_alcohol": "[OX2H][CX4H1]([#6])[#6]",
    "tertiary_alcohol": "[OX2H][CX4]([#6])([#6])[#6]",
    "phenol": "[OX2H][cX3]",
    "enol": "[OX2H][CX3]=[CX3]",
    "thiol": "[SX2H][#6]",
    "selenol": "[SeX2H]",
    
    # === HYDROPEROXIDES ===
    "hydroperoxide": "[OX2H][OX2][#6]",
    "peroxide": "[#6][OX2][OX2][#6]",
    
    # === AMINES ===
    "primary_amine": "[NX3;H2;!$([NX3][CX3]=O);!$([NX3][CX3]=[NX2])][#6]",  # PERC-02: sp2/sp3, excludes amide/urea/guanidine N
    "secondary_amine": "[NX3;H1;!$([NX3][CX3]=O);!$([NX3][CX3]=[NX2])]([CX4,cX3,$([CX3]=[CX3;!R])])[CX4,cX3,$([CX3]=[CX3;!R])]",  # DATA-02+PERC-07: sp3, aromatic, or acyclic vinyl C; exclude amides/guanidines
    "tertiary_amine": "[NX3;H0;!$([NX3][CX3]=O);!$([NX3][CX3]=[NX2])]([CX4,cX3,$([CX3]=[CX3;!R])])([CX4,cX3,$([CX3]=[CX3;!R])])[CX4,cX3,$([CX3]=[CX3;!R])]",  # DATA-02+PERC-07: sp3, aromatic, or acyclic vinyl C; exclude amides/guanidines
    "aromatic_amine": "[NX3H2][cX3]",
    
    # === IMINES ===
    "imine": "[CX3]=[NX2H]",
    "oxime": "[CX3]=[NX2][OX2H]",
    "hydrazone": "[CX3]=[NX2][NX3]",
    "hydrazine_fg": "[NX3;H1;!$([NX3][CX3]=O)][NX3H2]",  # DATA-05c: P-62.4 -NH-NH2, excludes hydrazides
    
    # === ETHERS (no suffix - substitutive naming) ===
    "ether": "[OX2]([CX4])[CX4]",
    "vinyl_ether": "[OX2]([#6])[CX3]=[CX3]",
    "aromatic_ether": "[OX2]([#6])[cX3]",
    "thioether": "[SX2]([#6])[#6]",
    "disulfide": "[#6][SX2][SX2][#6]",  # DATA-05b: P-63.6.2

    # === PHOSPHORUS COMPOUNDS (check more specific first) ===
    # Phosphate esters (C-O-P bonds, not C-P bonds) - check before phosphine oxide
    "phosphate_triester": "[PX4](=O)([OX2][#6])([OX2][#6])[OX2][#6]",
    "phosphate_diester": "[PX4](=O)([OX2][#6])([OX2][#6])[OX2H1]",
    "phosphate_monoester": "[PX4](=O)([OX2][#6])([OX2H1])[OX2H1]",
    # Phosphine oxide: R3P=O - three C attached to P(V)
    "phosphine_oxide": "[PX4](=O)([#6])([#6])[#6]",
    # Phosphines (P(III)) - check last as parent hydride
    "tertiary_phosphine": "[PX3]([#6])([#6])[#6]",
    "secondary_phosphine": "[PX3H1]([#6])[#6]",
    "primary_phosphine": "[PX3H2][#6]",

    # === SULFUR OXIDATION STATES (check more specific first) ===
    # Sulfone: S with 2 =O and 2 C neighbors (R-SO2-R')
    "sulfone": "[SX4](=[OX1])(=[OX1])([#6])[#6]",
    # Sulfoxide: S with 1 =O and 2 C neighbors (R-SO-R')
    "sulfoxide": "[SX3](=[OX1])([#6])[#6]",
    
    # === UNSATURATION ===
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    
    # === HALOGENS (always prefixes) ===
    "fluoro": "[FX1][#6]",
    "chloro": "[ClX1][#6]",
    "bromo": "[BrX1][#6]",
    "iodo": "[IX1][#6]",

    # === HYDROXAMIC ACIDS (PERC-03: P-65.3.3) ===
    "hydroxamic_acid": "[CX3](=O)[NX3;H1][OX2H]",  # R-C(=O)-NH-OH: free hydroxamic acid only

    # === CYANATES / THIOCYANATES (PERC-03: P-65.5) ===
    "cyanate": "[OX2][CX2]#[NX1]",
    "thiocyanate": "[SX2][CX2]#[NX1]",

    # === AZO (PERC-03: P-67.2) ===
    "azo": "[#6][NX2]=[NX2][#6]",

    # === OTHER ===
    "nitro": "[NX3+](=O)[O-]",
    "nitroso": "[NX2]=[OX1]",
    "azido": "[NX1]=[NX2+]=[NX1-]",
    "diazo": "[#6]=[NX2+]=[NX1-]",  # DATA-05a: P-61.5 diazo group
}

# Pre-compile all SMARTS patterns once at module load (avoid recompilation per molecule)
_COMPILED_FG_SMARTS = {}
for _fg_name, _smarts in FUNCTIONAL_GROUP_SMARTS.items():
    _pat = Chem.MolFromSmarts(_smarts)
    if _pat is not None:
        _COMPILED_FG_SMARTS[_fg_name] = _pat


def detect_functional_groups(mol) -> Dict[str, List[Tuple[int, ...]]]:
    """
    Detect all functional groups in a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        Dictionary mapping functional group names to lists of atom index tuples.
        Each tuple contains the indices of atoms in one instance of that group.

    Example:
        >>> mol = Chem.MolFromSmiles("CC(=O)O")  # acetic acid
        >>> groups = detect_functional_groups(mol)
        >>> "carboxylic_acid" in groups
        True
        >>> len(groups["carboxylic_acid"])
        1
    """
    results = defaultdict(list)

    for fg_name, pattern in _COMPILED_FG_SMARTS.items():
        matches = mol.GetSubstructMatches(pattern, uniquify=True)
        for match in matches:
            results[fg_name].append(match)
    
    # Post-processing: remove generic FG matches that overlap with more-specific FGs
    results = _resolve_fg_collisions(results)

    return dict(results)


def _resolve_fg_collisions(results):
    """Remove generic FG matches that overlap with more-specific FGs.

    Collision rules:
    - urea atoms should NOT also be detected as primary_amide/secondary_amide/tertiary_amide
    - guanidine atoms should NOT also be detected as imine
    - carbamate atoms should NOT also be detected as ester or amide (primary/secondary/tertiary)
    - isocyanate/isothiocyanate atoms should NOT also be detected as nitrile or primary_amide
    """
    for fg_specific, fg_generic_list in [
        ('urea', ['primary_amide', 'secondary_amide', 'tertiary_amide']),
        ('guanidine', ['imine']),
        ('carbamate', ['ester', 'primary_amide', 'secondary_amide', 'tertiary_amide']),
        ('isocyanate', ['nitrile', 'primary_amide']),
        ('isothiocyanate', ['nitrile', 'primary_amide']),
        # Carbamic acid: N-C(=O)-OH must NOT also match carboxylic_acid or amide
        ('carbamic_acid', ['carboxylic_acid', 'primary_amide', 'secondary_amide', 'tertiary_amide']),
        # Thiocarboxylic acids: SH in C(=O)SH or C(=S)SH must NOT match thiol
        # C(=O)SH must NOT match thioester either (C(=O)S is substructure of both)
        ('thioic_S_acid', ['thiol', 'thioester']),
        ('dithioic_acid', ['thiol', 'thioketone']),
        # C(=S)OH should not collide with carboxylic_acid (different SMARTS: =S vs =O)
        # but suppress thioketone matches on the C=S carbon
        ('thioic_O_acid', ['thioketone']),
        # Phase 163 Tier FRN-A: chalcogen-acid suppressions
        # (mirror thioic_S_acid -> thiol+thioester at line 228 above; AUDIT-FRN § 2.1)
        # Forward-reference note: selenoester/telluroester/tellurol added in
        # commits 163-02-03 (chalcogen-ketones) + 163-02-05 (chalcogen-esters);
        # unknown FG names are harmlessly no-op'd by the resolver loop below.
        ('selenoic_Se_acid', ['selenol', 'selenoester', 'thioester']),
        ('diselenoic_acid', ['selenol', 'selenoketone']),
        ('selenoic_O_acid', ['selenoketone', 'carboxylic_acid']),
        ('telluroic_Te_acid', ['tellurol', 'telluroester', 'thioester']),
        ('ditelluroic_acid', ['tellurol', 'telluroketone']),
        ('telluroic_O_acid', ['telluroketone', 'carboxylic_acid']),
        # Phase 163 Tier FRN-B: chalcogen-amide suppressions (AUDIT-FRN § 2.2 + RESEARCH §3.2).
        # Single-permissive [NX3] match captures =[S,Se,Te]-N(H,R) at all 3 N-degrees;
        # downstream N-degree inspection happens at assembly time.
        ('thioamide', ['thioketone', 'primary_amine', 'secondary_amine', 'tertiary_amine']),
        ('selenoamide', ['selenoketone', 'primary_amine', 'secondary_amine', 'tertiary_amine']),
        ('telluroamide', ['telluroketone', 'primary_amine', 'secondary_amine', 'tertiary_amine']),
        # Phase 163 Tier FRN-C: chalcogen-aldehyde/ketone suppressions (AUDIT-FRN § 2.3 + RESEARCH §3.3).
        # Defensive suppression mirrors existing thioaldehyde discipline; =O vs =Se/=Te should not
        # overlap structurally but the cascade preserves parallelism for downstream safety.
        ('selenoaldehyde', ['aldehyde', 'ketone']),
        ('telluroaldehyde', ['aldehyde', 'ketone']),
        ('selenoketone', ['selenoether', 'selenoester']),
        ('telluroketone', ['telluroether', 'telluroester']),
        # Phase 163 Tier FRN-D: iminoester suppressions (AUDIT-FRN § 2.4 + RESEARCH §3.4 + §9 Risk B).
        # Risk B mitigation: iminoester -> ester defensive suppression mirrors the existing thio*
        # cascade discipline; imine/primary_amine suppress the =NH from being double-claimed;
        # ether suppresses the -O-C portion from being double-claimed.
        ('iminoester', ['ester', 'imine', 'primary_amine', 'ether']),
        # Ester O-Ar bond should NOT also match aromatic_ether.
        # The ester oxygen in -C(=O)-O-Ar is part of the ester, not a separate
        # aromatic ether; without this, phenyl esters double-name as both
        # "phenoxy" and "phenoxycarbonyl".
        ('ester', ['aromatic_ether']),
        # PERC-01: broadened aldehyde [CX3;H1,H2](=O) now matches C=O in acid
        # derivatives (formates, formamides, etc.); suppress aldehyde on overlap
        ('carboxylic_acid', ['aldehyde']),
        ('ester', ['aldehyde']),
        ('anhydride', ['aldehyde']),
        ('acid_chloride', ['aldehyde']),
        ('acid_bromide', ['aldehyde']),
        ('acid_fluoride', ['aldehyde']),
        ('primary_amide', ['aldehyde']),
        ('secondary_amide', ['aldehyde']),
        ('tertiary_amide', ['aldehyde']),
        ('hydrazide', ['aldehyde']),
        ('imide', ['aldehyde']),
        ('thioic_S_acid', ['aldehyde']),
        ('carbamic_acid', ['aldehyde']),
        # USUB-11: imide suppresses overlapping amide matches to prevent
        # double-counting C=O groups (one as amide, one as ketone)
        ('imide', ['primary_amide', 'secondary_amide', 'tertiary_amide']),
        # PERC-02: broadened amine pattern overlaps with aromatic_amine on aromatic carbons
        ('aromatic_amine', ['primary_amine']),
        # PERC-03: hydroxamic acid suppresses amide + alcohol false positives
        ('hydroxamic_acid', ['primary_amide', 'secondary_amide', 'primary_alcohol',
                             'secondary_alcohol', 'tertiary_alcohol', 'carboxylic_acid']),
        # PERC-03: cyanate/thiocyanate suppress ether/thioether + nitrile
        ('cyanate', ['ether', 'nitrile']),
        ('thiocyanate', ['thioether', 'nitrile']),
        # PERC-03: azo suppresses imine
        ('azo', ['imine']),
        # DATA-03: guanidine suppresses amidine (guanidine is more specific)
        ('guanidine', ['amidine']),
        # DATA-03: amidine suppresses imine and primary_amine on its atoms
        ('amidine', ['imine', 'primary_amine']),
        # DATA-04: acid iodide suppresses aldehyde (parallel to other acid halides)
        ('acid_iodide', ['aldehyde']),
        # DATA-05b: disulfide suppresses thioether if S atoms overlap
        ('disulfide', ['thioether']),
        # DATA-05c: hydrazine suppresses primary_amine on its -NH2 nitrogen
        ('hydrazine_fg', ['primary_amine']),
        # DATA-05c: hydrazide is more specific than hydrazine_fg
        ('hydrazide', ['hydrazine_fg']),
        # PERC-05: specific alcohol subtypes suppress generic "alcohol" on same atoms
        ('primary_alcohol', ['alcohol']),
        ('secondary_alcohol', ['alcohol']),
        ('tertiary_alcohol', ['alcohol']),
        ('phenol', ['alcohol']),
        ('enol', ['alcohol']),
        # PERC-05: hydroxamic acid suppresses generic alcohol too
        ('hydroxamic_acid', ['alcohol']),
    ]:
        if fg_specific in results:
            specific_atoms = set()
            for match in results[fg_specific]:
                specific_atoms.update(match)

            for fg_generic in fg_generic_list:
                if fg_generic in results:
                    # Remove generic matches where ANY atom overlaps with specific
                    results[fg_generic] = [
                        m for m in results[fg_generic]
                        if not any(atom in specific_atoms for atom in m)
                    ]
                    # Clean up empty lists
                    if not results[fg_generic]:
                        del results[fg_generic]

    # Phosphate specificity: more-specific phosphate esters suppress
    # less-specific phosphate esters for the same P atom.
    # NOTE: phosphonic_acid is NOT suppressed -- it remains as principal group
    # candidate for molecules with C-O-P(=O)(OH)2 because the naming pipeline
    # uses its "phosphono" prefix. Full phosphate principal group naming is
    # deferred to a future phase.
    _phosphate_suppress = [
        ('phosphate_triester', ['phosphate_diester', 'phosphate_monoester']),
        ('phosphate_diester', ['phosphate_monoester']),
    ]
    for fg_specific, fg_generic_list in _phosphate_suppress:
        if fg_specific in results:
            # Collect P atom indices from specific matches (P is always first atom in SMARTS)
            specific_p_atoms = {m[0] for m in results[fg_specific]}
            for fg_generic in fg_generic_list:
                if fg_generic in results:
                    results[fg_generic] = [
                        m for m in results[fg_generic]
                        if m[0] not in specific_p_atoms
                    ]
                    if not results[fg_generic]:
                        del results[fg_generic]

    return results


def has_functional_group(mol, fg_name: str) -> bool:
    """
    Check if molecule contains a specific functional group.
    
    Args:
        mol: RDKit Mol object
        fg_name: Name of functional group (must be in FUNCTIONAL_GROUP_SMARTS)
        
    Returns:
        True if functional group is present
    """
    pattern = _COMPILED_FG_SMARTS.get(fg_name)
    if pattern is None:
        return False

    return mol.HasSubstructMatch(pattern)


def get_functional_group_atoms(mol, fg_name: str) -> List[Tuple[int, ...]]:
    """
    Get atom indices for all instances of a specific functional group.
    
    Args:
        mol: RDKit Mol object
        fg_name: Name of functional group
        
    Returns:
        List of tuples of atom indices
    """
    pattern = _COMPILED_FG_SMARTS.get(fg_name)
    if pattern is None:
        return []

    return list(mol.GetSubstructMatches(pattern, uniquify=True))


def count_functional_groups(mol) -> Dict[str, int]:
    """
    Count occurrences of each functional group.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Dictionary mapping functional group names to counts
    """
    groups = detect_functional_groups(mol)
    return {name: len(matches) for name, matches in groups.items()}
