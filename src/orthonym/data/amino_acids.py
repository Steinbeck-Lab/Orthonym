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
    # v23 Phase 12 follow-on (audit F2): cystine is the disulfide dimer of two
    # cysteines. The natural L-cystine (2R,2'R) is already in the OPSIN-import
    # catalog (-> 'cystine'); the D-enantiomer (2S,2'S) was unrecognised. Keyed by
    # exact stereo SMILES (OPSIN parses 'D-cystine'; RT-verified). meso-cystine
    # (2R,2'S) has no simple retained name and stays fail-closed (unknown).
    "N[C@H](CSSC[C@@H](N)C(=O)O)C(=O)O": "D-cystine",
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


# v24 W8 P3 Task 3.1 (P-103.1.3.2.2): the C-3 epimer of L-threonine / L-isoleucine
# has the retained-name PIN `allo-<name>`. Base SMILES below are (L-form, L-allo
# form = C-3 inverted, CIP-verified): L-Thr (2S,3R) / L-allo-Thr (2S,3S);
# L-Ile (2S,3S) / L-allo-Ile (2S,3R). The full 4-stereoisomer descriptor map is
# built on demand (canonical-SMILES keys -> descriptor prefix) so the bare
# retained lookup can prepend allo-/D-allo- exactly like L(implicit)/D-.
_ALLO_AA_BASE: Dict[str, tuple] = {
    "threonine": ("C[C@@H](O)[C@H](N)C(=O)O", "C[C@H](O)[C@H](N)C(=O)O"),
    "isoleucine": ("CC[C@H](C)[C@H](N)C(=O)O", "CC[C@@H](C)[C@H](N)C(=O)O"),
}
_ALLO_AA_FORMS_CACHE: Optional[Dict[str, Dict[str, str]]] = None


def _allo_aa_forms() -> Dict[str, Dict[str, str]]:
    """Canonical-isomeric-SMILES -> descriptor-prefix map per allo-capable AA.
    Built lazily (RDKit at call time, not import time). Values: '' (L, implicit),
    'D-', 'allo-' (L-allo), 'D-allo-'."""
    global _ALLO_AA_FORMS_CACHE
    if _ALLO_AA_FORMS_CACHE is None:
        from rdkit import Chem
        forms: Dict[str, Dict[str, str]] = {}
        for nm, (l_smi, lallo_smi) in _ALLO_AA_BASE.items():
            forms[nm] = {
                Chem.CanonSmiles(l_smi): "",
                _enantiomer_canon(l_smi): "D-",
                Chem.CanonSmiles(lallo_smi): "allo-",
                _enantiomer_canon(lallo_smi): "D-allo-",
            }
        _ALLO_AA_FORMS_CACHE = forms
    return _ALLO_AA_FORMS_CACHE


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
    allo_forms = _allo_aa_forms().get(name)
    if allo_forms is not None:
        # Multi-stereocentre threonine/isoleucine: all four stereoisomers have a
        # retained-name PIN (L implicit, D-, allo- = C-3 epimer, D-allo-).
        # P-103.1.3.2.2. A SMILES matching none of the four (should not happen for
        # these 2-stereocentre AAs) -> None (defer to systematic).
        input_canon = Chem.MolToSmiles(mol, canonical=True)  # isomeric (default)
        return allo_forms.get(input_canon)
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
    # desc in {"", "D-", "allo-", "D-allo-"} (P-103.1.1.1 / P-103.1.3.2.2).
    return f"{desc}{name}"


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


# v24 W8 P3 (P-103.2.6, BB 54595-54608): the ester 'ate' stem for the
# single-alpha-stereocentre monocarboxylic standard amino acids + glycine.
# "Esters of amino acids ... are formed ... using the 'ate' ending obtained by
# replacing the 'ic acid' ending or the final letter 'e' of the retained name
# (or adding the ending 'ate' to the name tryptophan)". Diacid AAs (aspartic /
# glutamic -- need positional ester locants, e.g. `1-methyl L-aspartate`,
# BB 54606) and 2-stereocentre AAs (threonine / isoleucine -- allo descriptor
# entanglement) are deliberately OMITTED so the ester namer falls through to
# the pre-existing systematic path for them (no regression).
AMINO_ACID_ATE_STEMS: Dict[str, str] = {
    "glycine": "glycinate",
    "alanine": "alaninate",
    "valine": "valinate",
    "leucine": "leucinate",
    "serine": "serinate",
    "cysteine": "cysteinate",
    "methionine": "methioninate",
    "phenylalanine": "phenylalaninate",
    "tyrosine": "tyrosinate",
    "tryptophan": "tryptophanate",  # BB 54597: 'ate' ADDED to 'tryptophan'
    "proline": "prolinate",
    "arginine": "argininate",
    "lysine": "lysinate",
    "histidine": "histidinate",
    "asparagine": "asparaginate",
    "glutamine": "glutaminate",
}


def get_amino_acid_ate_stem(trivial_name: str) -> Optional[str]:
    """Get the ester 'ate' stem for an in-scope amino acid (P-103.2.6).

    Args:
        trivial_name: Bare (stereo-free) trivial name of the amino acid, e.g.
            "alanine" (NOT "L-alanine").

    Returns:
        The 'ate' stem (e.g. "alaninate") if in scope, None otherwise (defer
        to the pre-existing systematic ester namer -- diacid / 2-stereocentre
        AAs, and any non-standard amino acid, are never in this map).
    """
    return AMINO_ACID_ATE_STEMS.get(trivial_name)


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
