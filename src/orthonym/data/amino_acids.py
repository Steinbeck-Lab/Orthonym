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
    # catalog; the D-enantiomer (2S,2'S) was unrecognised, so it is keyed here by
    # exact stereo SMILES. meso-cystine (2R,2'S) has no retained name and stays
    # fail-closed (unknown).
    #
    # v29 P3-CLEANUP Item 1: the VALUE was `"D-cystine"` -- the descriptor written
    # straight into the table -- while the L form resolved through the OPSIN import
    # to a bare `"cystine"`. That is what made the two enantiomers asymmetric: D
    # kept a descriptor on EVERY path (including the deliberately bare
    # `with_descriptor=False` peptide/fragment contract) and L had one on none.
    # Both now carry the BARE retained name and the single descriptor path
    # (`_dimeric_aa_forms`) supplies L-/D-, so the pair cannot drift again.
    "N[C@H](CSSC[C@@H](N)C(=O)O)C(=O)O": "cystine",
    # v29 P3-CLEANUP Item 1: dopa is the OTHER Table 10.5 entry in this table that
    # was dropping its P-103.1.3.1 descriptor. The OPSIN import supplies only the
    # L (2S) key, so the D enantiomer had NO retained name at all while L shipped a
    # bare `dopa`. Hand-curated here (hand-curated wins: `_integrate_opsin_simplegroup`
    # skips keys already present) so the pair is symmetric, and the descriptor
    # itself comes from the shared path via `_DL_CAPABLE_NONSTANDARD`.
    # OPSIN-verified: `D-dopa` -> O=C(O)[C@H](N)CC1=CC=C(O)C(O)=C1, this structure.
    "N[C@H](Cc1ccc(O)c(O)c1)C(=O)O": "dopa",
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
# has the retained name `L-allo<name>` — `L-allothreonine`, FUSED, no hyphen after
# 'allo'. Base SMILES below are (L-form, L-allo form = C-3 inverted, CIP-verified):
# L-Thr (2S,3R) / L-alloThr (2S,3S); L-Ile (2S,3S) / L-alloIle (2S,3R). The full
# 4-stereoisomer descriptor map is built on demand (canonical-SMILES keys ->
# descriptor prefix) so the retained lookup can prepend `L-`, `D-`, `L-allo` or
# `D-allo`.
#
# v29 P3-CLEANUP MINOR 11: this comment wrote every allo descriptor with a
# TRAILING HYPHEN (`L-allo-<name>`, "prepend L-/D-/L-allo-/D-allo-") while the
# values are `"L-allo"` / `"D-allo"` and `_allo_aa_forms`'s own docstring says
# "*the last two carry NO trailing hyphen*" — one file contradicting itself about
# the exact spelling that `eba5d3bb` had just adjudicated. Corrected here rather
# than left as a second reading of the same rule.
#
# TWO CORRECTIONS to this comment, v29 P3-REGRESSION I12: (1) it said "PIN", but
# `### **P-100 INTRODUCTION**` (BlueBookV2.md:50939) states at :50943 that
# "*Preferred IUPAC names (PINs) are not identified for the compounds in this
# Chapter*" — these are prescribed retained names, not PINs; (2) the L was written
# as implicit, which is the P-103.3.4 PEPTIDE rule applied out of scope.
_ALLO_AA_BASE: Dict[str, tuple] = {
    "threonine": ("C[C@@H](O)[C@H](N)C(=O)O", "C[C@H](O)[C@H](N)C(=O)O"),
    "isoleucine": ("CC[C@H](C)[C@H](N)C(=O)O", "CC[C@@H](C)[C@H](N)C(=O)O"),
}
_ALLO_AA_FORMS_CACHE: Optional[Dict[str, Dict[str, str]]] = None

# v29 P3-CLEANUP Item 1 (P-103.1.1.2 + P-103.1.3.1): the Table 10.5 "less common"
# amino acids carry the SAME alpha-carbon D/L descriptor as the Table 10.4 ones.
# `## **P-103.1.3.1** The stereodescriptors 'D' and 'L'` (BlueBookV2.md:54291)
# scopes itself to "*the alpha-amino carboxylic acids*" -- NOT to Table 10.4 --
# and at :54301 names cystine EXPLICITLY: "*The 'L' configuration corresponds to
# the 'S' configuration of the CIP system, except that cysteine has the 'R'
# configuration (and also cystine, see P-103.1.1.2)*". Cystine is a Table 10.5
# entry (`P-103.1.1.2 Retained names of 'less common' amino acids`, :54233/:54238,
# systematic equivalent `3,3'-disulfanediyldialanine`). That is the SAME sentence
# I12 quoted to license `cysteine -> L-cysteine`, so leaving cystine bare was an
# inconsistency in the I12 class fix, not a separate policy.
#
# Cystine is a SYMMETRIC DIMER: BOTH stereocentres are alpha-carbons, so ONE
# descriptor designates both (2R,2'R = L-cystine, 2S,2'S = D-cystine) -- exactly
# what the pre-existing hand-curated `D-cystine` entry already assumed for the D
# side while the L side shipped bare.
#
# The MESO diastereomer (2R,2'S) is deliberately ABSENT from the map. It is a
# single compound, not the equimolar D/L MIXTURE that P-103.1.3.1 (:54305)
# reserves 'DL' for ("*A mixture of equimolar amounts of 'D' and 'L' compounds is
# termed a 'racemate' and is designated by the stereodescriptor 'DL'*"), so it has
# no retained form here: `.get()` returns None -> defer to the systematic namer
# (fail closed), preserving the behaviour claimed at the NON_STANDARD entry above.
_DIMERIC_AA_BASE: Dict[str, str] = {
    # retained name -> the L form; the D form is derived as its exact enantiomer,
    # so the two descriptors CANNOT drift apart the way the hand-curated pair did.
    "cystine": "N[C@@H](CSSC[C@H](N)C(=O)O)C(=O)O",
}
_DIMERIC_AA_FORMS_CACHE: Optional[Dict[str, Dict[str, str]]] = None

# The NON-standard retained names whose alpha-carbon configuration IS designated
# D/L, i.e. the ones the `is_standard` gate in `get_amino_acid_name` must not
# silence. Membership requires a Blue Book Table 10.4/10.5 entry -- NOT mere
# presence in the merged OPSIN-import vocabulary.
#
# Why so short: an audit of all 49 stereo-specified non-standard entries against
# `BlueBookV2.md` found NO occurrence of statine, lanthionine, diaminopimelic,
# selenocystine, tellurocystine, carnitine, lysopine, pantetheine or homocystine
# (positive controls in the same chapter: cysteine 6, homocysteine 2, citrulline
# 1, allysine 1, cysteic 1, so the search does reach Table 10.5; the OCR-tolerant
# pattern returned the same counts as the literal one). Prepending `L-` to those
# would invent a name the Blue Book does not give and that OPSIN cannot parse --
# which the validity gate would then suppress to `unknown organic compound`,
# trading a spelling nit for a coverage loss (invariant 11).
#
#   cystine : Table 10.5 (:54238); named EXPLICITLY by P-103.1.3.1 at :54301.
#             Descriptor via `_dimeric_aa_forms` (two alpha-carbons).
#   dopa    : Table 10.5 (:54241, systematic `3-hydroxytyrosine`). One
#             alpha-carbon, standard L=S rule, so `_get_stereo_prefix` judges it.
#             OPSIN-verified both ways: `L-dopa`/`D-dopa` parse back to the exact
#             (S)/(R) input structures.
_DL_CAPABLE_NONSTANDARD = frozenset({"cystine", "dopa"})


def _allo_aa_forms() -> Dict[str, Dict[str, str]]:
    """Canonical-isomeric-SMILES -> descriptor-prefix map per allo-capable AA.
    Built lazily (RDKit at call time, not import time). Values: 'L-', 'D-',
    'L-allo', 'D-allo' (the last two carry NO trailing hyphen: the Blue Book fuses
    'allo' into the retained name, giving `L-allothreonine`).

    THE 'allo' IS NOT HYPHENATED for an amino acid. `## **P-103.1.3.2.2** Use of the
    prefix 'allo'` (BlueBookV2.md:54320) writes `L-allothreonine`,
    `L-alloisoleucine` (:54326-54330), and `allothreonine`/`alloisoleucine` at
    :54226-54228 and inside a peptide at :54721 (`L-allothreonyl`) — 5 unhyphenated
    occurrences and ZERO hyphenated ones (searched both the literal `allo-threonine`
    and the OCR form `alloGthreonine`, with a positive control proving the `G`=hyphen
    pattern finds matches in this file). The hyphenated, ITALIC `*allo*-` does occur
    — but only as a CARBOHYDRATE/cyclitol configurational prefix (:53011, :53021,
    :54890, e.g. `D-*allo*-non-3-ulose`, `*allo*-inositol`). v29 P3-REGRESSION
    applied the sugar convention to amino acids; corrected here. This OVERTURNS the
    v24 W8 P3 spelling (recorded in baseline_targets `rebaselined_v28_aa_stereo_gold`),
    which cited P-103.1.3.2.2 — the section that spells it fused.

    v29 P3-REGRESSION I12: the L forms used to map to ``''`` and ``'allo-'``, i.e.
    the L was DROPPED. The suppression cited `P-103.1.3.2.2`, and that section
    refutes it — `## **P-103.1.3.2.2** Use of the prefix 'allo'`
    (BlueBookV2.md:54320) writes all four out at :54324-54330 WITH the descriptor
    (`L-isoleucine`, `L-alloisoleucine`, `L-threonine`, `L-allothreonine`), each
    alongside a fully-configured systematic alternative. The omission licence is
    `### **P-103.3.4** Indication of configuration in peptides` (:54715) and is
    scoped to peptides."""
    global _ALLO_AA_FORMS_CACHE
    if _ALLO_AA_FORMS_CACHE is None:
        from rdkit import Chem
        forms: Dict[str, Dict[str, str]] = {}
        for nm, (l_smi, lallo_smi) in _ALLO_AA_BASE.items():
            forms[nm] = {
                Chem.CanonSmiles(l_smi): "L-",
                _enantiomer_canon(l_smi): "D-",
                # No hyphen after 'allo': the Blue Book FUSES it into the retained
                # amino-acid name -- see the docstring below.
                Chem.CanonSmiles(lallo_smi): "L-allo",
                _enantiomer_canon(lallo_smi): "D-allo",
            }
        _ALLO_AA_FORMS_CACHE = forms
    return _ALLO_AA_FORMS_CACHE


def _dimeric_aa_forms() -> Dict[str, Dict[str, str]]:
    """Canonical-isomeric-SMILES -> descriptor-prefix map per symmetric-dimer AA.

    Built lazily (RDKit at call time, not import time), mirroring
    ``_allo_aa_forms``. Values are ``'L-'`` / ``'D-'``; every other stereoisomer
    (for cystine, the meso form) is ABSENT by construction, so the caller's
    ``.get()`` yields None = "defer to the systematic namer". See the
    ``_DIMERIC_AA_BASE`` comment for the Blue Book derivation."""
    global _DIMERIC_AA_FORMS_CACHE
    if _DIMERIC_AA_FORMS_CACHE is None:
        from rdkit import Chem
        forms: Dict[str, Dict[str, str]] = {}
        for nm, l_smi in _DIMERIC_AA_BASE.items():
            forms[nm] = {
                Chem.CanonSmiles(l_smi): "L-",
                _enantiomer_canon(l_smi): "D-",
            }
        _DIMERIC_AA_FORMS_CACHE = forms
    return _DIMERIC_AA_FORMS_CACHE


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

    Returns "L-" or "D-" for a resolved α-carbon, "" when there is nothing to
    designate (glycine is achiral; an unresolvable centre also yields ""), or None
    to DEFER — the input is a diastereomer the retained name cannot represent.

    v29 P3-REGRESSION I12 — this used to end with

        return "D-" if _get_stereo_prefix(mol, name) == "D-" else ""

    which computed the configuration correctly and then threw the L away, so
    L-alanine and a configuration-free name were spelled identically. That is a
    stereo LOSS, and it was invisible to every gate: OPSIN resolves bare `alanine`
    to the L structure, so the round-trip compares EQUAL, and the gold rows pinned
    the bare names. Only Orthonym's own stereo backstop objected.

    The suppression was `### **P-103.3.4** Indication of configuration in peptides`
    (BlueBookV2.md:54715) applied outside its scope; verbatim at :54717: "*The
    stereodescriptor 'L' is not indicated in the names nor in the symbolic
    representation of peptides composed of amino acids listed in Table 10.4.*"  A
    free amino acid is not a peptide — which `rules/peptides.py` already stated in
    its own docstring ("*the L-omission is a display rule applied here, so a
    standalone amino acid (P-103.1) still shows L*"). This module violated that
    invariant; `rules/esters.py` never did, which is why `methyl L-alaninate`
    (BB :54601) was always right.

    NOT a PIN claim: `### **P-100 INTRODUCTION**` (:50939) at :50943 — "*Preferred
    IUPAC names (PINs) are not identified for the compounds in this Chapter.*"
    `L-alanine` is the Blue Book's prescribed retained name, not a PIN.
    """
    from rdkit import Chem
    # Symmetric-dimer AAs (cystine): BOTH stereocentres are alpha-carbons, so the
    # single-alpha-carbon `_get_stereo_prefix` below cannot judge them. Explicit
    # stereoisomer map; meso is absent -> None -> defer (P-103.1.1.2 / P-103.1.3.1,
    # see `_DIMERIC_AA_BASE`).
    dimeric_forms = _dimeric_aa_forms().get(name)
    if dimeric_forms is not None:
        return dimeric_forms.get(Chem.MolToSmiles(mol, canonical=True))
    allo_forms = _allo_aa_forms().get(name)
    if allo_forms is not None:
        # Multi-stereocentre threonine/isoleucine: all four stereoisomers have a
        # retained name (L-, D-, L-allo- = C-3 epimer, D-allo-), per
        # `## **P-103.1.3.2.2** Use of the prefix 'allo'` (:54320), examples at
        # :54324-54330. A SMILES matching none of the four (should not happen for
        # these 2-stereocentre AAs) -> None (defer to systematic).
        input_canon = Chem.MolToSmiles(mol, canonical=True)  # isomeric (default)
        return allo_forms.get(input_canon)
    # Single-stereocentre: alpha-carbon CIP via the peptide descriptor logic, which
    # already returns exactly "L-"/"D-"/"" and handles the cysteine CIP inversion
    # (L = R for the S/Se side chains). Its verdict is now USED, not filtered.
    # (lazy import avoids the data<->rules circular import at module load).
    from ..rules.peptides import _get_stereo_prefix
    return _get_stereo_prefix(mol, name)


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

    # WSD-07 descriptor path: STANDARD amino acids, PLUS the Blue-Book-attested
    # non-standard retained names that P-103.1.3.1 designates a D/L for.
    #
    # v29 P3-CLEANUP Item 1: this used to be a blanket `if not is_standard`. The
    # docstring's justification for that gate is real but NARROWER than the gate --
    # a non-standard `D-<name>` invented for an OPSIN-vocabulary entry (D-butyrine,
    # D-statine, ...) is not OPSIN-parseable and would be SUPPRESSED to 'unknown',
    # i.e. a coverage loss traded for a spelling (invariant 11). That argument does
    # not reach a name the Blue Book itself prescribes the descriptor for and OPSIN
    # parses: `L-cystine` -> the exact input structure (RT-verified). Using
    # `is_standard` as the proxy therefore dropped the descriptor on the one
    # non-standard AA the BB names EXPLICITLY at :54301, while the D side kept one
    # because it had been hand-written into the table VALUE -- the asymmetry.
    #
    # The registry is keyed by RETAINED NAME and is deliberately tiny: membership
    # requires a Blue Book Table 10.4/10.5 entry, not mere presence in the merged
    # OPSIN-import vocabulary. A BB-absence audit over all 49 stereo-specified
    # non-standard entries (positive controls: cysteine 6, homocysteine 2,
    # citrulline 1, allysine 1, cysteic 1 -- so the search reaches Table 10.5) found
    # NO Blue Book occurrence of statine / lanthionine / diaminopimelic /
    # selenocystine / tellurocystine / carnitine / homocystine, so those correctly
    # keep the bare OPSIN vocabulary name rather than acquiring a fabricated one.
    if not is_standard and name not in _DL_CAPABLE_NONSTANDARD:
        return name

    # Recover the configurational descriptor + full-stereo verify.
    from rdkit import Chem
    m = mol if mol is not None else Chem.MolFromSmiles(canonical_smiles)
    if m is None:
        return name
    desc = _aa_config_descriptor(m, name)
    if desc is None:
        return None  # diastereomer -> defer to the systematic namer
    # desc in {"", "L-", "D-", "L-allo-", "D-allo-"} — "" only when there is nothing
    # to designate (achiral glycine, or an unresolvable centre). P-103.1.3.1 /
    # P-103.1.3.2.2.
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


# --- Task E: PIN gate over the amino-acid tables -----------------------------
# Until now this surface never consulted the curated PIN authority
# (data/iupac_2013_pin_list.json), which is why `CNCC(=O)O` emitted `sarcosine`
# on the DEFAULT --style pin path. `sarcosine` appears nowhere in the Blue Book
# (0 hits; grep methodology validated against `glycine` = 26 hits), and
# P-103.1.1.3 "Systematic substitutive names" (BlueBookV2.md:54247), sentence
# :54251, directs exactly this case elsewhere: "When not denoted by a retained
# name, amino acids receive systematic substitutive names constructed by
# applying the principles, rules and conventions of substitutive nomenclature."
# The Blue Book states the precedent itself at :54253 -- norvaline and
# norleucine are likewise non-retained and "are not recommended".
#
# Note the framing: P-100 "INTRODUCTION" (:50939), sentence :50943, says
# "Preferred IUPAC names (PINs) are not identified for the compounds in this
# Chapter", so NO Chapter-10 name is Blue-Book-designated (PIN) -- not
# `sarcosine` and not `glycine`. The P-103 tables give prescribed RETAINED
# names. The defect is therefore not "sarcosine is not the PIN" but "sarcosine
# is a name no Blue Book rule licenses".
#
# Rows are DEMOTED, never deleted: a denied name moves to
# GENERAL_ONLY_AMINO_ACIDS so general nomenclature and --trivial keep it. This
# mirrors the ALL_RETAINED_NAMES / GENERAL_RETAINED_NAMES split exactly.
#
# ⚠ The gate is the curated deny-list, NOT the OPSIN `is_pin` field.
#  hard-codes `"is_pin": False` (lines 268/742/847),
# so all 232 entries in amino_acids_opsin.py carry False; gating on it would
# withdraw all 88 names this module integrates, `cystine` (Blue Book Table 10.5,
# P-103.1.1.2) among them. Each deny is adjudicated per row, with a citation and
# a verified systematic replacement -- the granularity data/__init__.py:263
# already identifies as the correct one.
from .pin_policy import is_pin_denied as _is_pin_denied  # noqa: E402

#: Canonical SMILES -> trivial name for amino acids withdrawn from the PIN path.
GENERAL_ONLY_AMINO_ACIDS: Dict[str, str] = {
    smi: name
    for smi, name in {**STANDARD_AMINO_ACIDS, **NON_STANDARD_AMINO_ACIDS}.items()
    if _is_pin_denied(name)
}

for _smi in GENERAL_ONLY_AMINO_ACIDS:
    STANDARD_AMINO_ACIDS.pop(_smi, None)
    NON_STANDARD_AMINO_ACIDS.pop(_smi, None)

if GENERAL_ONLY_AMINO_ACIDS:
    logger.debug(
        "PIN gate withdrew %d amino-acid keys (%d distinct names) to "
        "GENERAL_ONLY_AMINO_ACIDS",
        len(GENERAL_ONLY_AMINO_ACIDS),
        len(set(GENERAL_ONLY_AMINO_ACIDS.values())),
    )
