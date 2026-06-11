"""
Functional group seniority rules for principal group selection.

Based on IUPAC 2013 Blue Book P-41 to P-43.
The principal group (highest seniority) becomes the suffix;
all other groups become prefixes.
"""

from typing import Optional, Tuple, List, Dict


# IM-01 (2026-04-25): per-FG SMARTS attachment atom override.
#
# Most functional-group SMARTS lead with the locant-bearing atom (the C of
# -COOH in [CX3](=O)[OX2H1]; the C of -OH in [CX4][OX2H1]). Parent-selection
# locant comparators historically used SMARTS-match index 0 as the canonical
# attachment for that FG instance.
#
# For PGs whose SMARTS leads with a flanking (non-locant-bearing) atom, this
# heuristic is wrong. The motivating case is ``disulfide`` (``[#6][SX2][SX2][#6]``):
# atom 0 is a flanking carbon, but IUPAC P-31.1.4 says the locant set for a
# multi-atom PG uses the heteroatoms (S, S). When a disulfide lives in a ring,
# using the flanking C's locant produces a wrong ring-vs-chain decision in
# the P-44.1(f) cascade, silently dropping the ring sulfurs from the name.
#
# Each entry maps FG name -> list of SMARTS-match indices to use as
# locant-bearing atoms. When the helper sees multiple indices, downstream
# comparators take ``min(locants)`` to match P-31.1.4. Default for any FG
# not present is ``[0]`` (preserves existing behaviour).
#
# Source: 
PG_ATTACHMENT_INDICES: Dict[str, List[int]] = {
    "disulfide": [1, 2],  # [#6][SX2][SX2][#6] -- the two S sulfurs
    # WS-A task 9: the ketone family SMARTS '[#6][CX3](=X)[#6]' leads with a
    # FLANKING carbon; the locant-bearing atom is the carbonyl/thiocarbonyl
    # carbon at match index 1. Without these entries every on-ring/on-chain/
    # locant computation used the flanking atom (an aryl ketone's "attachment"
    # was the benzene ring atom itself -> false pg_on_ring -> wrong parent).
    "ketone": [1],
    "thioketone": [1],
    "selenoketone": [1],
    "telluroketone": [1],
}

# Functional group seniority order (highest priority first)
# Groups in this list can be expressed as suffixes when principal
SENIORITY_ORDER = [
    # Acids (highest priority)
    "carboxylic_acid",
    # Thiocarboxylic acids (IUPAC P-65.3) -- just below carboxylic acid
    "thioic_S_acid",
    "thioic_O_acid",
    "dithioic_acid",
    # Phase 163 Tier FRN-A: chalcogen acids (P-65.3) — additive per CONTEXT D-08;
    # position locked by AUDIT-FRN § 5; parallel to thioic_S_acid/O_acid/dithioic
    "selenoic_Se_acid",
    "selenoic_O_acid",
    "diselenoic_acid",
    "telluroic_Te_acid",
    "telluroic_O_acid",
    "ditelluroic_acid",
    # Carbamic acid (IUPAC P-65.2.3) -- retained acid name, rank with carboxylic acids
    "carbamic_acid",
    "sulfonic_acid",
    "sulfinic_acid",
    "sulfenic_acid",   # IUPAC P-65.3.1.4: between sulfinic and phosphonic
    "phosphonic_acid",
    "phosphinic_acid",
    "boronic_acid",    # P-68.3 boron acid
    # BBR-PERC (169.7): free inorganic oxoacids — P-67 functional parents. Ranked in
    # the acid tier so they win PCG when perceived; NAMING is P-67 (downstream). Listed
    # so get_principal_group recognizes them (no KeyError) and they are not silently dropped.
    "phosphoric_acid",
    "sulfuric_acid",
    "nitric_acid",
    "carbonic_acid",

    # Acid derivatives
    "anhydride",
    "ester",
    # Phase 163 Tier FRN-D: iminoester at ester-tier per AUDIT-FRN § 5 LOCK
    # ("alkyl alkanimidate" functional-class parallel to "alkyl alkanoate" per P-65.1.7)
    "iminoester",
    "thioester",
    # Phase 163 Tier FRN-E: chalcogen-esters (P-65.6 extension) — additive per CONTEXT D-08
    "selenoester",
    "telluroester",
    "acid_chloride",
    "acid_bromide",
    "acid_fluoride",
    "acid_iodide",    # IUPAC P-65.5.1: parallel to other acid halides

    # Nitrogen acid derivatives
    "primary_amide",
    "secondary_amide",
    "tertiary_amide",
    # Phase 163 Tier FRN-B: chalcogen amides (P-66.1.4.1.1 + P-66.6.3) — additive per CONTEXT D-08
    # thioamide ranks below amide and above sulfonamide; selenoamide/telluroamide parallel
    "thioamide",
    "selenoamide",
    "telluroamide",
    "primary_sulfonamide",
    "secondary_sulfonamide",
    "tertiary_sulfonamide",
    "hydrazide",
    "hydroxamic_acid",  # IUPAC P-65.3.3: between hydrazides and imides
    "imide",
    "amidine",     # IUPAC P-66.4.1: between imide and nitrile

    # Nitriles
    "nitrile",
    "isocyanide",

    # Carbonyls
    "aldehyde",
    "ketone",
    "thioaldehyde",
    # Phase 163 Tier FRN-C aldehydes (P-66.6.3) — additive per CONTEXT D-08
    "selenoaldehyde",
    "telluroaldehyde",
    "thioketone",
    # Phase 163 Tier FRN-C ketones (P-66.6.3) — additive per CONTEXT D-08
    "selenoketone",
    "telluroketone",

    # Alcohols and analogs
    "primary_alcohol",
    "secondary_alcohol",
    "tertiary_alcohol",
    "phenol",
    "enol",
    "alcohol",         # PERC-05: generic catch-all, same seniority tier as other alcohols
    # Class 17 "Hydroxy compounds and chalcogen analogues" (alcohols/phenols/thiol/
    # selenol/tellurol) — all senior to class 18 hydroperoxide. BBR-HYG/DEF-D-09 (169.7):
    # hydroperoxide MOVED below the chalcogen-ols (was incorrectly above thiol with a
    # factually-wrong "Class 19" comment). Verified BlueBookV2 P-41 Table 4.1 lines
    # ~18190-18191: "17 Hydroxy compounds and chalcogen analogues" then "18 Hydroperoxides".
    "thiol",
    "selenol",
    "tellurol",        # BBR-PERC (169.7): Te analogue of -ol/-selenol (P-63.1.5)
    "hydroperoxide",   # P-41 Table 4.1 class 18 (peroxol -OOH); junior to class-17 hydroxy/thiol

    # Hydroxylamines (BBR-PERC, 169.7: P-68.3 class 21; ranked just above amines)
    "hydroxylamine",

    # Amines
    "primary_amine",
    "secondary_amine",
    "tertiary_amine",
    "aromatic_amine",

    # Imines
    "imine",
    "oxime",
    "hydrazone",

    # Pseudohalides and special groups (prefix-only in IUPAC 2013)
    # Placed here so they are recognized but always junior to suffix-capable groups
    "azido",        # P-65.5: prefix-only
    "azo",          # P-67.2: prefix-only
    "cyanate",      # P-65.5: prefix-only (pseudohalide)
    "thiocyanate",  # P-65.5: prefix-only (pseudohalide)
    "diazo",           # P-61.5: prefix-only
    "disulfide",       # P-63.6.2: prefix-only
    "hydrazine_fg",    # P-62.4: prefix-only

    # Sulfur oxidation states (functional class naming, lower seniority than amines)
    "sulfoxide",
    "sulfone",
    "thioether",  # Also called sulfide
    # BBR-PERC (169.7): Se/Te ether analogues, same tier as thioether (P-63.6)
    "selenoether",
    "telluroether",

    # Phosphorus compounds (functional class or substitutive naming)
    "phosphine_oxide",
    "phosphate_triester",
    "phosphate_diester",
    "phosphate_monoester",
    "tertiary_phosphine",
    "secondary_phosphine",
    "primary_phosphine",
]

# BBR-PERC/DEF-3 (Phase 169.7): groups present in SENIORITY_ORDER for ranking but
# which are ALWAYS detachable prefixes — never a principal (suffix) group (P-33).
# get_principal_group skips these so a prefix-only-only molecule is named with the
# group as a prefix (azidomethane) instead of dropping it. These mirror the
# "Pseudohalides and special groups (prefix-only)" block in SENIORITY_ORDER above.
_PREFIX_ONLY_PRINCIPAL = frozenset({
    "azido",         # P-65.5
    "azo",           # P-67.2
    "cyanate",       # P-65.5
    "thiocyanate",   # P-65.5
    "diazo",         # P-61.5
    "disulfide",     # P-63.6.2
    "hydrazine_fg",  # P-62.4
})

# ASML-18 / D-08: Map subtypes to canonical parent for seniority comparison.
# IUPAC P-65.1: All alcohol types have equal seniority; all amine types have equal seniority.
# Used only in get_principal_group() for equalization; SENIORITY_ORDER stays intact per D-09.
_SENIORITY_PARENT = {
    "primary_alcohol": "alcohol",
    "secondary_alcohol": "alcohol",
    "tertiary_alcohol": "alcohol",
    "phenol": "alcohol",
    "enol": "alcohol",
    "primary_amine": "amine",
    "secondary_amine": "amine",
    "tertiary_amine": "amine",
    "aromatic_amine": "amine",
}

# Suffix forms for principal groups
# Format: (chain_terminal_suffix, ring_attached_suffix), or None for functional-class-only
SUFFIX_FORMS = {
    "carboxylic_acid": ("oic acid", "carboxylic acid"),
    "thioic_S_acid": ("thioic S-acid", "carbothioic S-acid"),
    "thioic_O_acid": ("thioic O-acid", "carbothioic O-acid"),
    "dithioic_acid": ("dithioic acid", "carbodithioic acid"),
    # Phase 163 Tier FRN-A: chalcogen-acid SUFFIX_FORMS (parallel to thioic_*_acid) per AUDIT-FRN § 5
    "selenoic_Se_acid": ("selenoic Se-acid", "carboselenoic Se-acid"),
    "selenoic_O_acid": ("selenoic O-acid", "carboselenoic O-acid"),
    "diselenoic_acid": ("diselenoic acid", "carbodiselenoic acid"),
    "telluroic_Te_acid": ("telluroic Te-acid", "carbotelluroic Te-acid"),
    "telluroic_O_acid": ("telluroic O-acid", "carbotelluroic O-acid"),
    "ditelluroic_acid": ("ditelluroic acid", "carboditelluroic acid"),
    "carbamic_acid": ("carbamic acid", "carbamic acid"),  # Retained name, same for chain/ring
    "sulfonic_acid": ("sulfonic acid", "sulfonic acid"),
    "sulfinic_acid": ("sulfinic acid", "sulfinic acid"),
    "phosphonic_acid": ("phosphonic acid", "phosphonic acid"),
    "phosphinic_acid": ("phosphinic acid", "phosphinic acid"),
    "anhydride": ("oic anhydride", "carboxylic anhydride"),
    "ester": ("oate", "carboxylate"),
    "acid_chloride": ("oyl chloride", "carbonyl chloride"),
    "acid_bromide": ("oyl bromide", "carbonyl bromide"),
    "acid_fluoride": ("oyl fluoride", "carbonyl fluoride"),
    "primary_amide": ("amide", "carboxamide"),
    "secondary_amide": ("amide", "carboxamide"),
    "tertiary_amide": ("amide", "carboxamide"),
    # Phase 163 Tier FRN-B: chalcogen-amide SUFFIX_FORMS (P-66.1.4.1.1 + P-66.6.3 PIN)
    "thioamide": ("thioamide", "carbothioamide"),
    "selenoamide": ("selenoamide", "carboselenoamide"),
    "telluroamide": ("telluroamide", "carbotelluroamide"),
    "primary_sulfonamide": ("sulfonamide", "sulfonamide"),
    "secondary_sulfonamide": ("sulfonamide", "sulfonamide"),
    "tertiary_sulfonamide": ("sulfonamide", "sulfonamide"),
    "nitrile": ("nitrile", "carbonitrile"),
    "aldehyde": ("al", "carbaldehyde"),
    "ketone": ("one", "one"),
    "thioaldehyde": ("thial", "carbothialdehyde"),
    "thioketone": ("thione", "thione"),
    # Phase 163 Tier FRN-C: chalcogen-aldehyde/ketone SUFFIX_FORMS per AUDIT-FRN § 5 LOCK
    # PIN short form -selenal / -tellural (parallel to -thial per P-66.6.3)
    "selenoaldehyde": ("selenal", "carboselenoaldehyde"),
    "telluroaldehyde": ("tellural", "carbotelluroaldehyde"),
    # PIN suffix form -selone / -tellone (dialkyl-word form fails OPSIN per AUDIT § 5 LOCK + RESEARCH §7.2)
    "selenoketone": ("selone", "selone"),
    "telluroketone": ("tellone", "tellone"),
    "primary_alcohol": ("ol", "ol"),
    "secondary_alcohol": ("ol", "ol"),
    "tertiary_alcohol": ("ol", "ol"),
    "phenol": ("ol", "ol"),
    "enol": ("ol", "ol"),
    "alcohol": ("ol", "ol"),  # PERC-05: generic catch-all suffix form
    "thiol": ("thiol", "thiol"),
    "selenol": ("selenol", "selenol"),
    "tellurol": ("tellurol", "tellurol"),  # BBR-PERC (169.7): Te -ol analogue (P-63.1.5)
    "hydroperoxide": ("peroxol", "peroxol"),
    # BBR-PERC (169.7): hydroxylamine + free oxoacids + Se/Te ethers are named via
    # their handler / functional-parent (P-67/P-68.3) / functional-class (P-63.6) paths,
    # NOT as a chain suffix — None per the SUFFIX_FORMS contract (like thioether).
    "hydroxylamine": None,
    "phosphoric_acid": None,
    "sulfuric_acid": None,
    "nitric_acid": None,
    "carbonic_acid": None,
    "selenoether": None,
    "telluroether": None,
    "primary_amine": ("amine", "amine"),
    "secondary_amine": ("amine", "amine"),
    "tertiary_amine": ("amine", "amine"),
    "aromatic_amine": ("amine", "amine"),
    "imine": ("imine", "imine"),
    "oxime": ("oxime", "oxime"),           # functional class suffix
    "hydrazone": ("hydrazone", "hydrazone"),  # functional class suffix
    "boronic_acid": ("boronic acid", "boronic acid"),  # retained acid form
    "hydroxamic_acid": ("hydroxamic acid", "hydroxamic acid"),  # IUPAC P-65.3.3
    # --- Phase 93-01: close SENIORITY_ORDER suffix gaps ---
    # FGs with real suffix forms
    "hydrazide": ("ohydrazide", "carbohydrazide"),  # IUPAC P-66.3
    "imide": ("imide", "dicarboximide"),  # IUPAC P-66.2
    # FGs with functional class naming only (no substitutive suffix)
    "thioester": None,           # IUPAC P-65.3.1: functional class naming (S-alkyl alkanethioate)
    # Phase 163 Tier FRN-D + FRN-E: functional-class (handler-emitted) — SUFFIX_FORMS = None
    "iminoester": None,          # P-65.1.7: emitted via handlers/imidate.py per CONTEXT D-03
    "selenoester": None,         # P-65.6: functional-class "Se-alkyl alkaneselenoate"
    "telluroester": None,        # P-65.6: functional-class "Te-alkyl alkanetelluroate"
    "isocyanide": None,          # IUPAC 2013 P-66.5.3: prefix-only (isocyano)
    "sulfoxide": None,           # IUPAC P-63.6: functional class naming (dialkyl sulfoxide)
    "sulfone": None,             # IUPAC P-63.6: functional class naming (dialkyl sulfone)
    "thioether": None,           # IUPAC P-63.2: functional class naming (dialkyl sulfide)
    "phosphine_oxide": None,     # IUPAC P-68.3: functional class naming
    "phosphate_triester": None,  # IUPAC P-68: functional class naming
    "phosphate_diester": None,   # IUPAC P-68: functional class naming
    "phosphate_monoester": None, # IUPAC P-68: substitutive prefix only (phosphonooxy)
    "tertiary_phosphine": None,  # IUPAC P-68.3: parent hydride naming (phosphane)
    "secondary_phosphine": None, # IUPAC P-68.3: parent hydride naming
    "primary_phosphine": None,   # IUPAC P-68.3: parent hydride naming
    # Prefix-only FGs added to SENIORITY_ORDER (no suffix form)
    "azido": None,               # IUPAC P-65.5: prefix-only
    "azo": None,                 # IUPAC P-67.2: prefix-only
    "cyanate": None,             # IUPAC P-65.5: prefix-only (pseudohalide)
    "thiocyanate": None,         # IUPAC P-65.5: prefix-only (pseudohalide)
    # Phase 109: 6 new FG classes
    "acid_iodide": ("oyl iodide", "carbonyl iodide"),  # IUPAC P-65.5.1
    "amidine": ("imidamide", "carboximidamide"),        # IUPAC P-66.4.1
    "sulfenic_acid": ("sulfenic acid", "sulfenic acid"),  # IUPAC P-65.3.1.4
    "diazo": None,               # P-61.5: prefix-only
    "disulfide": None,           # P-63.6.2: prefix-only
    "hydrazine_fg": None,        # P-62.4: prefix-only
}

# Prefix forms for non-principal groups
# Audit (Phase 113-01): verified all SENIORITY_ORDER entries have a PREFIX_FORMS
# key. None entries are genuinely functional-class-only (no IUPAC prefix form).
PREFIX_FORMS = {
    "carboxylic_acid": "carboxy",
    "thioic_S_acid": "sulfanylcarbonyl",  # IUPAC P-65.1.1.4: S-acid prefix (-C(=O)SH)
    "thioic_O_acid": "carbothioyl",      # IUPAC P-65.1.1.4: O-acid prefix (-C(=S)OH)
    "dithioic_acid": "dithiocarboxy",     # IUPAC P-65.1.1.4: dithioic acid prefix (-C(=S)SH)
    # Phase 163 Tier FRN-A: chalcogen-acid PREFIX_FORMS (parallel to thioic_*_acid) per AUDIT-FRN § 5
    "selenoic_Se_acid": "selanylcarbonyl",
    "selenoic_O_acid": "carboselenoyl",
    "diselenoic_acid": "diselenocarboxy",
    "telluroic_Te_acid": "tellanylcarbonyl",
    "telluroic_O_acid": "carbotelluroyl",
    "ditelluroic_acid": "ditellurocarboxy",
    "carbamic_acid": "carbamoyloxy",  # When not principal group
    "sulfonic_acid": "sulfo",
    "sulfinic_acid": "sulfino",
    "aldehyde": "oxo",  # or "formyl" for terminal
    "ketone": "oxo",
    "thioketone": "sulfanylidene",  # P-63.1.5: =S as non-principal prefix
    # Phase 163 Tier FRN-C: =Se / =Te ketone non-principal prefix (parallel to sulfanylidene) per AUDIT-FRN § 5
    "selenoketone": "selanylidene",
    "telluroketone": "tellanylidene",
    "primary_alcohol": "hydroxy",
    "secondary_alcohol": "hydroxy",
    "tertiary_alcohol": "hydroxy",
    "phenol": "hydroxy",
    "enol": "hydroxy",
    "alcohol": "hydroxy",  # PERC-05: generic catch-all prefix form
    "thiol": "sulfanyl",
    "selenol": "selanyl",
    "tellurol": "tellanyl",   # BBR-PERC (169.7): Te analogue of sulfanyl/selanyl (P-63.1.5)
    "hydroperoxide": "hydroperoxy",
    # BBR-PERC (169.7): hydroxylamine + free oxoacids are functional parents
    # (P-67/P-68.3); not expressed as detachable prefixes (None, like thioether).
    "hydroxylamine": None,
    "phosphoric_acid": None,
    "sulfuric_acid": None,
    "nitric_acid": None,
    "carbonic_acid": None,
    "primary_amine": "amino",
    "secondary_amine": "amino",
    "tertiary_amine": "amino",
    "aromatic_amine": "amino",
    "imine": "imino",
    "oxime": "hydroxyimino",
    "hydrazone": "hydrazinylidene",
    "isocyanate": "isocyanato",
    "isothiocyanate": "isothiocyanato",
    "urea": "carbamoylamino",
    "guanidine": "guanidino",
    "carbamate": None,                  # functional class only
    "boronic_acid": "dihydroxyboranyl",
    "n_oxide_aromatic": None,           # functional class only
    "n_oxide_aliphatic": None,          # functional class only
    "primary_sulfonamide": "sulfamoyl",
    "secondary_sulfonamide": "sulfamoyl",
    "tertiary_sulfonamide": "sulfamoyl",
    "nitrile": "cyano",
    "isocyanide": "isocyano",
    # Amides as non-principal group prefix (IUPAC P-66.1.1.4 method 2)
    # Only primary_amide (-CONH2) gets carbamoyl here; secondary/tertiary amides
    # are already handled via the acylamino naming pathway in the pipeline.
    # Adding carbamoyl for sec/tert causes double-naming (e.g., "ethanoylamino" + "carbamoyl").
    "primary_amide": "carbamoyl",
    # Phase 163 Tier FRN-B: chalcogen-amide PREFIX_FORMS (parallel to primary_amide carbamoyl) per AUDIT-FRN § 5
    "thioamide": "carbamothioyl",
    "selenoamide": "carbamoselenoyl",
    "telluroamide": "carbamotelluroyl",
    # Halogens (always prefixes)
    "fluoro": "fluoro",
    "chloro": "chloro",
    "bromo": "bromo",
    "iodo": "iodo",
    # Other
    "nitro": "nitro",
    "nitroso": "nitroso",
    "azido": "azido",
    # Acid halides as non-principal group prefix (IUPAC P-65.5.1.4)
    "acid_chloride": "carbonochloridoyl",
    "acid_bromide": "bromocarbonyl",
    "acid_fluoride": "fluorocarbonyl",
    # Esters (handled specially in polyfunctional.py as acyloxy prefixes)
    "ester": None,  # Esters use alkoxycarbonyl prefix (generated in polyfunctional.py get_fg_prefix_form)
    # Ethers and thioethers
    "ether": None,  # Named by substitution: methoxy, ethoxy, etc.
    "thioether": None,  # IUPAC P-63.2: functional class naming (dialkyl sulfide)
    # BBR-PERC (169.7): Se/Te ethers use the dynamic (alkyl)selanyl/tellanyl generator
    # in substituent_prefix_forms.get_substituent_prefix_form (parallel to thioether).
    "selenoether": None,  # IUPAC P-63.6: (alkyl)selanyl substitutive prefix
    "telluroether": None,  # IUPAC P-63.6: (alkyl)tellanyl substitutive prefix
    # Sulfur oxidation states (IUPAC P-63.6)
    "sulfoxide": "sulfinyl",  # IUPAC P-63.6: bivalent prefix for -S(=O)-
    "sulfone": "sulfonyl",    # IUPAC P-63.6: bivalent prefix for -S(=O)2-
    # Phosphorus compounds
    "phosphonic_acid": "phosphono",
    "phosphinic_acid": "phosphino",
    # Phosphine oxide and phosphates use functional class naming
    "phosphine_oxide": None,
    "phosphate_triester": None,
    "phosphate_diester": None,
    "phosphate_monoester": "phosphonooxy",  # IUPAC P-67.1.3

    "tertiary_phosphine": None,
    "secondary_phosphine": None,
    "primary_phosphine": None,
    # PERC-04: New FG prefix forms
    "hydroxamic_acid": "N-hydroxyamido",  # IUPAC P-65.3.3
    "cyanate": "cyanato",                 # IUPAC P-65.5
    "thiocyanate": "thiocyanato",         # IUPAC P-65.5
    "azo": "diazenyl",                    # IUPAC P-67.2
    # --- Phase 93-01: close SENIORITY_ORDER prefix gaps ---
    "anhydride": None,            # IUPAC P-65.1: functional class naming only
    "secondary_amide": None,      # Named via acylamino pathway in universal pipeline
    "tertiary_amide": None,       # Named via acylamino pathway in universal pipeline
    "hydrazide": "hydrazinecarbonyl",  # IUPAC P-66.3.5
    "imide": None,                # Named as heterocyclic ring substituent
    "thioaldehyde": "thioxo",     # IUPAC P-63.1.5: =S as non-principal prefix (parallel to "oxo")
    # Phase 163 Tier FRN-C: =Se / =Te non-principal prefix (parallel to thioxo) per AUDIT-FRN § 5
    "selenoaldehyde": "selenoxo",
    "telluroaldehyde": "telluroxo",
    "thioester": None,            # Named via decomposition pathway
    # Phase 163 Tier FRN-D + FRN-E: functional-class — no prefix form
    "iminoester": None,
    "selenoester": None,
    "telluroester": None,
    # Phase 109: 6 new FG classes
    "acid_iodide": "iodocarbonyl",     # IUPAC P-65.5.1.4
    "amidine": "amidino",              # IUPAC P-66.4.1
    "sulfenic_acid": "sulfeno",        # IUPAC P-65.3.1.4
    "diazo": "diazo",                  # P-61.5
    "disulfide": "disulfanediyl",      # P-63.6.2
    "hydrazine_fg": "hydrazinyl",      # P-62.4
}


def get_principal_group(
    mol,
    functional_groups: Dict[str, List[tuple]]
) -> Tuple[Optional[str], List[tuple]]:
    """
    Determine the principal characteristic group.

    The principal group is the highest-seniority functional group
    that will be expressed as a suffix in the name.

    Args:
        mol: RDKit Mol object
        functional_groups: Dict from detect_functional_groups()

    Returns:
        Tuple of (group_name, list_of_atom_index_tuples)
        Returns (None, []) if no suffix-capable group found
    """
    for fg_name in SENIORITY_ORDER:
        if fg_name in functional_groups and functional_groups[fg_name]:
            # BBR-PERC/DEF-3 (169.7): a principal characteristic group MUST be
            # suffix-capable (P-33). The pseudohalide / special prefix-only groups
            # (azido/azo/cyanate/thiocyanate/diazo/disulfide/hydrazine) are kept in
            # SENIORITY_ORDER for RANKING but can never be the principal group — they
            # are always detachable prefixes (P-59/P-65.5/P-61/P-62.4/P-63.6.2/P-67.2).
            # Skipping them here makes a molecule whose ONLY group is prefix-only return
            # (None, []), so the FG is emitted as a prefix (e.g. azidomethane) instead
            # of being consumed-as-principal-then-dropped (-> bare 'methane'). Matches
            # this function's contract: "the group that will be expressed as a suffix".
            if fg_name in _PREFIX_ONLY_PRINCIPAL:
                continue
            return fg_name, functional_groups[fg_name]

    return None, []


def get_suffix(fg_name: str, is_ring: bool = False) -> Optional[str]:
    """
    Get the suffix form for a functional group.

    Args:
        fg_name: Name of functional group
        is_ring: True if the group is attached to a ring (not terminal on chain)

    Returns:
        Suffix string, or None if group has no suffix form
    """
    if fg_name not in SUFFIX_FORMS:
        return None

    entry = SUFFIX_FORMS[fg_name]
    if entry is None:
        return None  # Functional class naming only -- no substitutive suffix

    chain_suffix, ring_suffix = entry
    return ring_suffix if is_ring else chain_suffix


def get_prefix(fg_name: str) -> Optional[str]:
    """
    Get the prefix form for a functional group.

    Used when the group is not the principal group.

    Args:
        fg_name: Name of functional group

    Returns:
        Prefix string, or None if group has no standard prefix form
    """
    return PREFIX_FORMS.get(fg_name)


def is_suffix_group(fg_name: str) -> bool:
    """
    Check if a functional group can be expressed as a suffix.

    Args:
        fg_name: Name of functional group

    Returns:
        True if group can be a suffix (is in seniority order)
    """
    return fg_name in SENIORITY_ORDER


def compare_seniority(fg1: str, fg2: str) -> int:
    """
    Compare seniority of two functional groups.

    Args:
        fg1, fg2: Names of functional groups

    Returns:
        -1 if fg1 is higher seniority
         0 if equal seniority (or both not in list)
         1 if fg2 is higher seniority
    """
    try:
        idx1 = SENIORITY_ORDER.index(fg1)
    except ValueError:
        idx1 = len(SENIORITY_ORDER)  # Not in list = lowest priority

    try:
        idx2 = SENIORITY_ORDER.index(fg2)
    except ValueError:
        idx2 = len(SENIORITY_ORDER)

    if idx1 < idx2:
        return -1  # fg1 is higher (lower index = higher priority)
    elif idx1 > idx2:
        return 1   # fg2 is higher
    return 0       # Equal


def get_all_prefix_groups(
    functional_groups: Dict[str, List[tuple]],
    principal_group: Optional[str]
) -> Dict[str, List[tuple]]:
    """
    Get all functional groups that should be named as prefixes.

    This includes all groups except the principal group.

    Args:
        functional_groups: Dict from detect_functional_groups()
        principal_group: Name of the principal group (or None)

    Returns:
        Dict of functional group names to their atom indices,
        excluding the principal group
    """
    prefix_groups = {}

    for fg_name, matches in functional_groups.items():
        if fg_name == principal_group:
            continue
        if matches:
            prefix_groups[fg_name] = matches

    return prefix_groups
