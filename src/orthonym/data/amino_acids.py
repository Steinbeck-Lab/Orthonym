"""
Amino acid data for trivial name lookups.

Standard amino acids (the 20 proteinogenic amino acids) have trivial names
that are preferred as IUPAC PINs over systematic names.

Expanded in Phase 141-02 with 88 OPSIN simpleGroup amino acid entries.
These are complete molecules with canonical SMILES directly from OPSIN's
aminoAcids.xml vocabulary.

Stereochemistry note: Most natural amino acids are L-configured (S in CIP),
but IUPAC prefers R/S notation over D/L for PINs.
"""

import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Standard amino acids: canonical SMILES -> trivial name
# Note: These are for the L-enantiomers (naturally occurring form)
# SMILES without explicit stereo will match
STANDARD_AMINO_ACIDS: Dict[str, str] = {
    # Aliphatic
    "NCC(=O)O": "glycine",                    # Gly - achiral
    "CC(N)C(=O)O": "alanine",                 # Ala
    "CC(C)C(N)C(=O)O": "valine",              # Val
    "CC(C)CC(N)C(=O)O": "leucine",            # Leu
    "CCC(C)C(N)C(=O)O": "isoleucine",         # Ile

    # Hydroxyl-containing
    "CC(O)C(N)C(=O)O": "threonine",           # Thr
    "NC(CO)C(=O)O": "serine",                 # Ser
    "NC(C(=O)O)CO": "serine",                 # Ser - alternate canonical

    # Sulfur-containing
    # v23 Phase 5 data fix: the key was CSCC(N)C(=O)O — that is C4
    # (CH3-S-CH2-CH(NH2)-COOH) = S-methylcysteine, NOT methionine. True
    # methionine is C5 (CH3-S-CH2-CH2-CH(NH2)-COOH); a C4 input was therefore
    # named 'methionine' (a DIFFERENT molecule). Re-keyed to the correct C5
    # structure; the C4 S-methylcysteine moved to NON_STANDARD_AMINO_ACIDS.
    "CSCCC(N)C(=O)O": "methionine",           # Met (C5; OPSIN-RT verified)
    "NC(CS)C(=O)O": "cysteine",               # Cys

    # Acidic and amides
    "NC(CC(=O)O)C(=O)O": "aspartic acid",     # Asp
    "NC(CCC(=O)O)C(=O)O": "glutamic acid",    # Glu
    "NC(CC(N)=O)C(=O)O": "asparagine",        # Asn
    "NC(=O)CC(N)C(=O)O": "asparagine",        # Asn - alternate
    "NC(CCC(N)=O)C(=O)O": "glutamine",        # Gln
    "NC(=O)CCC(N)C(=O)O": "glutamine",        # Gln - alternate

    # Basic
    "NCCCCC(N)C(=O)O": "lysine",              # Lys
    "NC(CCCNC(N)=N)C(=O)O": "arginine",       # Arg (input form)
    "NC(=N)NCCCC(N)C(=O)O": "arginine",       # Arg - alternate
    "N=C(N)NCCCC(N)C(=O)O": "arginine",       # Arg - canonical form
    "NC(Cc1cnc[nH]1)C(=O)O": "histidine",     # His
    "NC(Cc1c[nH]cn1)C(=O)O": "histidine",     # His - alternate
    "NC(Cc1[nH]cnc1)C(=O)O": "histidine",     # His - alternate 2

    # Aromatic
    "NC(Cc1ccccc1)C(=O)O": "phenylalanine",   # Phe
    "NC(Cc1ccc(O)cc1)C(=O)O": "tyrosine",     # Tyr
    "NC(Cc1c[nH]c2ccccc12)C(=O)O": "tryptophan",  # Trp

    # Imino acid (proline)
    "OC(=O)C1CCCN1": "proline",               # Pro
    "O=C(O)C1CCCN1": "proline",               # Pro - alternate canonical
}

# Non-standard amino acids with trivial names
NON_STANDARD_AMINO_ACIDS: Dict[str, str] = {
    # v23 Phase 5: C4 CH3-S-CH2-CH(NH2)-COOH = S-methylcysteine (the S-methyl
    # derivative of cysteine), moved here from the STANDARD methionine slot it
    # had wrongly occupied. Name per Blue Book line 54569 (S-methyl-L-cysteine);
    # OPSIN parses 'S-methylcysteine' to this exact structure (RT verified).
    "CSCC(N)C(=O)O": "S-methylcysteine",
    "NCCCC(N)C(=O)O": "ornithine",            # Orn - not proteinogenic
    "NC(CCCN)C(=O)O": "ornithine",            # Orn - alternate
    "NCCC(N)C(=O)O": "2,4-diaminobutanoic acid",  # Dab
    "NCCCC(=O)O": "4-aminobutanoic acid",     # GABA - gamma-aminobutyric acid
    "CNCC(=O)O": "sarcosine",                 # N-methylglycine
}


def _integrate_opsin_simplegroup() -> Dict[str, str]:
    """Integrate OPSIN simpleGroup amino acid entries into NON_STANDARD_AMINO_ACIDS.

    These are complete molecules (full SMILES with all atoms) from OPSIN's
    aminoAcids.xml vocabulary. Their canonical SMILES can be directly used
    as lookup keys.

    Returns:
        Dict mapping canonical SMILES to trivial names for entries not already
        in STANDARD_AMINO_ACIDS or NON_STANDARD_AMINO_ACIDS.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return {}

    try:
        from .opsin_imports.amino_acids_opsin import OPSIN_AMINO_ACIDS
    except ImportError:
        return {}

    integrated = {}
    for _key, entry in OPSIN_AMINO_ACIDS.items():
        if entry.get('subType') != 'simpleGroup':
            continue

        smiles = entry.get('smiles', '')
        names = entry.get('names', [])
        if not smiles or not names:
            continue

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            can_smiles = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            continue

        # Hand-curated entries take precedence
        if can_smiles in STANDARD_AMINO_ACIDS:
            continue
        if can_smiles in NON_STANDARD_AMINO_ACIDS:
            continue
        if can_smiles in integrated:
            continue

        # Use the first name from OPSIN's name list
        integrated[can_smiles] = names[0]

    return integrated


def _build_acyl_names(all_amino_acids: Dict[str, str],
                      existing_acyl: Dict[str, str]) -> Dict[str, str]:
    """Build acyl (peptide linkage) names for all known amino acids.

    Rules per IUPAC 3AA-13:
    - Names ending in 'ine': replace with 'yl' (e.g., glycine -> glycyl)
    - Names ending in 'ic acid': replace with 'yl' (e.g., aspartic acid -> aspartyl)
    - Names ending in 'an': add 'yl' (e.g., tryptophan -> tryptophanyl)
    - Names ending in 'ate': replace with 'yl' (e.g., dimethyltaurate -> dimethyltauryl)
    - Default: add 'yl'

    Hand-curated entries in existing_acyl take precedence.

    Args:
        all_amino_acids: Dict of canonical SMILES -> trivial name for all amino acids.
        existing_acyl: Dict of existing hand-curated acyl names.

    Returns:
        Dict of generated trivial name -> acyl name (excludes hand-curated entries).
    """
    generated = {}
    for _smiles, name in all_amino_acids.items():
        if name in existing_acyl:
            # Hand-curated entry takes precedence
            continue
        if name in generated:
            continue

        # Generate acyl name based on suffix rules
        if name.endswith('ine'):
            acyl = name[:-3] + 'yl'
        elif name.endswith('ic acid'):
            acyl = name[:-7] + 'yl'
        elif name.endswith('an'):
            acyl = name + 'yl'
        elif name.endswith('ate'):
            acyl = name[:-3] + 'yl'
        elif name.endswith('ol'):
            acyl = name[:-2] + 'yl'
        elif name.endswith('amine'):
            acyl = name[:-5] + 'aminyl'
        else:
            acyl = name + 'yl'

        generated[name] = acyl

    return generated


# WSD-07 (Phase 175): L-form isomeric SMILES for standard amino acids with >1
# stereocentre, where the alpha-carbon L/D descriptor alone cannot distinguish the
# named (threo) diastereomer from its allo form. Full-stereo verification compares
# the input's isomeric canonical SMILES against the L-form and its enantiomer (D-);
# anything else (an allo diastereomer) DEFERS — never a wrong-config retained name.
_MULTI_STEREO_AA_LFORM: Dict[str, str] = {
    "isoleucine": "CC[C@H](C)[C@H](N)C(=O)O",
    "threonine": "C[C@@H](O)[C@H](N)C(=O)O",
}


def _enantiomer_canon(smiles: str) -> Optional[str]:
    """Canonical isomeric SMILES of the mirror image (all tetrahedral centres inverted)."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    for a in m.GetAtoms():
        t = a.GetChiralTag()
        if t == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        elif t == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return Chem.MolToSmiles(m, canonical=True)


def _aa_config_descriptor(mol, name: str) -> Optional[str]:
    """Recover the L/D configurational descriptor for a standard amino acid (WSD-07).

    Returns "" (achiral, or L which is implicit in the bare retained name),
    "D-" (explicit), or None to DEFER — the input is a diastereomer the bare
    retained name cannot represent (e.g. allo-isoleucine).
    """
    from rdkit import Chem
    lform = _MULTI_STEREO_AA_LFORM.get(name)
    if lform is not None:
        # Multi-stereocentre: emit only for a clean L or D enantiomer; an allo
        # diastereomer matches neither reference -> defer.
        input_canon = Chem.MolToSmiles(mol, canonical=True)  # isomeric (default)
        if input_canon == Chem.CanonSmiles(lform):
            return ""
        if input_canon == _enantiomer_canon(lform):
            return "D-"
        return None
    # Single-stereocentre: alpha-carbon CIP via the peptide descriptor logic
    # (lazy import avoids the data<->rules circular import at module load).
    from ..rules.peptides import _get_stereo_prefix
    return "D-" if _get_stereo_prefix(mol, name) == "D-" else ""


def get_amino_acid_name(
    canonical_smiles: str, mol=None, with_descriptor: bool = False,
) -> Optional[str]:
    """
    Get the trivial name for an amino acid if it's a standard one.

    Args:
        canonical_smiles: Canonical SMILES of the amino acid.
        mol: optional RDKit Mol (rebuilt from ``canonical_smiles`` if None) —
            needed for the stereo-strip fallback / descriptor recovery.
        with_descriptor: when True (free-amino-acid naming path, WSD-07), also
            recover and emit the L/D configurational descriptor (P-103.1.1.1; L
            implicit, D explicit) and DEFER (return None) for a diastereomer the
            bare retained name cannot represent. Default False preserves the
            bare-name contract used by ``name_peptide`` (which adds its own stereo).

    Returns:
        Trivial name (optionally with a D- descriptor) if found, None otherwise.
    """
    name = None
    is_standard = False
    # Exact-string lookup first (back-compat; the keys are stereo-free).
    if canonical_smiles in STANDARD_AMINO_ACIDS:
        name = STANDARD_AMINO_ACIDS[canonical_smiles]
        is_standard = True
    elif canonical_smiles in NON_STANDARD_AMINO_ACIDS:
        name = NON_STANDARD_AMINO_ACIDS[canonical_smiles]

    # WSD-07: stereo-strip fallback (lifted from rules/peptides.py) — a CIP-tagged
    # amino acid misses the stereo-free keys, so strip isomeric info and re-look-up.
    # GATED to the free-AA path (with_descriptor): the DEFAULT path stays
    # byte-identical (returns None on a stereo-tagged miss) so name_peptide — which
    # has its OWN strip fallback at peptides.py:320-326 and relies on the None to
    # drive residue/decomposition decisions — is completely unaffected. STANDARD
    # amino acids only: their L/D retained forms are OPSIN-parseable PINs, whereas a
    # non-standard 'D-<name>' (e.g. D-butyrine) is NOT and would be suppressed to
    # 'unknown' (worse than the systematic name) — non-standard AAs keep the
    # original behavior (defer to the systematic namer).
    if name is None and with_descriptor:
        from rdkit import Chem
        m = mol if mol is not None else Chem.MolFromSmiles(canonical_smiles)
        if m is not None:
            nostereo = Chem.MolToSmiles(m, isomericSmiles=False, canonical=True)
            nostereo_mol = Chem.MolFromSmiles(nostereo)
            if nostereo_mol is not None:
                ns = Chem.MolToSmiles(nostereo_mol, canonical=True)
                if ns in STANDARD_AMINO_ACIDS:
                    name = STANDARD_AMINO_ACIDS[ns]
                    is_standard = True

    if name is None or not with_descriptor:
        return name

    # WSD-07 descriptor path: STANDARD amino acids only (see note above). A
    # non-standard exact-match returns its bare name unchanged (original behavior).
    if not is_standard:
        return name

    # Recover the configurational descriptor + full-stereo verify.
    from rdkit import Chem
    m = mol if mol is not None else Chem.MolFromSmiles(canonical_smiles)
    if m is None:
        return name
    desc = _aa_config_descriptor(m, name)
    if desc is None:
        return None  # diastereomer -> defer to the systematic namer
    return f"D-{name}" if desc == "D-" else name


def is_standard_amino_acid(canonical_smiles: str) -> bool:
    """Check if SMILES matches a standard amino acid."""
    return canonical_smiles in STANDARD_AMINO_ACIDS


# Amino acid acyl (peptide linkage) names for all 20 standard amino acids.
# Used when assembling peptide names: N-terminal residues use the acyl form.
# IUPAC 3AA-13: the acyl form replaces the terminal -ine/-ic acid/-an with -yl.
AMINO_ACID_ACYL_NAMES: Dict[str, str] = {
    # Aliphatic
    "glycine": "glycyl",
    "alanine": "alanyl",
    "valine": "valyl",
    "leucine": "leucyl",
    "isoleucine": "isoleucyl",
    # Hydroxyl-containing
    "serine": "seryl",
    "threonine": "threonyl",
    # Sulfur-containing
    "cysteine": "cysteinyl",
    "methionine": "methionyl",
    # Acidic and amides
    "aspartic acid": "aspartyl",
    "glutamic acid": "glutamyl",
    "asparagine": "asparaginyl",
    "glutamine": "glutaminyl",
    # Basic
    "lysine": "lysyl",
    "arginine": "arginyl",
    "histidine": "histidyl",
    # Aromatic
    "phenylalanine": "phenylalanyl",
    "tyrosine": "tyrosyl",
    "tryptophan": "tryptophyl",
    # Imino acid
    "proline": "prolyl",
}


def get_amino_acid_acyl_name(trivial_name: str) -> Optional[str]:
    """
    Get the acyl (peptide linkage) form of an amino acid name.

    Args:
        trivial_name: Trivial name of the amino acid (e.g., "glycine")

    Returns:
        Acyl name (e.g., "glycyl") if found, None otherwise
    """
    return AMINO_ACID_ACYL_NAMES.get(trivial_name)


# --- Module-level integration (must run after all dicts are defined) ---
# Integrate OPSIN simpleGroup entries at import time
_opsin_simple = _integrate_opsin_simplegroup()
NON_STANDARD_AMINO_ACIDS.update(_opsin_simple)
logger.debug(
    "Integrated %d OPSIN simpleGroup amino acids (total non-standard: %d)",
    len(_opsin_simple), len(NON_STANDARD_AMINO_ACIDS),
)

# Build expanded acyl names from all amino acid entries
_all_aa = {}
_all_aa.update(STANDARD_AMINO_ACIDS)
_all_aa.update(NON_STANDARD_AMINO_ACIDS)
_generated_acyl = _build_acyl_names(_all_aa, AMINO_ACID_ACYL_NAMES)
AMINO_ACID_ACYL_NAMES.update(_generated_acyl)
logger.debug(
    "Expanded acyl names to %d entries (+%d generated)",
    len(AMINO_ACID_ACYL_NAMES), len(_generated_acyl),
)
