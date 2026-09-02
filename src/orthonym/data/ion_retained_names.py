"""
Retained names for common ions used in IUPAC nomenclature.

This module provides lookup tables for common organic and inorganic ions
that have retained (trivial) names preferred over systematic names.

For salts, the naming convention is: cation name + anion name
Example: sodium acetate, potassium chloride

Keys are canonical SMILES (verified with RDKit canonicalization).
"""

from typing import Optional
from rdkit import Chem


# === ORGANIC ANIONS (carboxylates, alkoxides, etc.) ===
# Named with -ate, -ide suffixes
# All keys are in RDKit canonical SMILES form
RETAINED_ANIONS = {
    # Carboxylate anions
    'CC(=O)[O-]': 'acetate',
    'O=C[O-]': 'formate',
    'CCC(=O)[O-]': 'propanoate',
    'CCCC(=O)[O-]': 'butanoate',
    'O=C([O-])c1ccccc1': 'benzoate',
    'O=C([O-])C(=O)[O-]': 'oxalate',
    'O=C([O-])CC(=O)[O-]': 'malonate',
    'O=C([O-])CCC(=O)[O-]': 'succinate',

    # Alkoxide anions — Wave2 T2d, BB P-63.8.1 VERBATIM: "The traditional
    # names methoxide, ethoxide, propoxide, butoxide, phenoxide, and
    # aminoxide... are retained as preferred IUPAC names and may be
    # substituted in the same way as the corresponding alcohols. The
    # traditional name tert-butoxide... is also retained as a preferred
    # IUPAC name but cannot be substituted. The traditional name
    # isopropoxide... is retained for general nomenclature" (its PIN is
    # propan-2-olate: "potassium propan-2-olate (PIN) potassium
    # isopropoxide") — so isopropoxide is deliberately NOT in this table.
    'C[O-]': 'methoxide',
    'CC[O-]': 'ethoxide',
    'CCC[O-]': 'propoxide',
    'CCCC[O-]': 'butoxide',
    'CC(C)(C)[O-]': 'tert-butoxide',
    '[O-]c1ccccc1': 'phenoxide',  # Also called phenolate

    # Carbanions
    # (Plan 184-01): the '[CH3-]' -> 'methanide' entry was SUBSUMED by the
    # systematic emit_parent_hydride_cumulative_suffix primitive (ions.py). The
    # equivalence was proven byte-identical (the primitive emits 'methanide' for
    # the single-carbon chain, locant omitted) AND route_charged is confirmed to
    # reach the carbanion branch for [CH3-], so the primitive is now the single
    # source of truth (no per-molecule retained band-aid). [c-]1ccccc1 (phenide)
    # stays — it is a ring carbanion outside the acyclic parent-hydride path.
    '[c-]1ccccc1': 'phenide',  # Also called benzenide

    # Azanide (NH2-) — the conjugate base of azane (NH3). F-T6 (P-72.2.2.2):
    # the preselected name of the bare nitrogen-hydride anion. RETAINED_ANIONS is
    # consulted before INORGANIC_ANIONS, so this shadows the legacy inorganic
    # 'amide' (a deprecated name) — azanide is the IUPAC 2013 PIN and OPSIN
    # round-trips it to [NH2-]. (Bare [NH2-] previously emitted 'amide', which the
    # OPSIN validity gate suppressed to 'unknown organic compound'.)
    '[NH2-]': 'azanide',

    # Alkynide anions
    '[C-]#C': 'ethynide',  # Terminal alkynide

    # charged-species fix, 169.6 caveats (.7): retained charged-species names recovered
    # from the 169.6 route_charged regression (neutralize-first produced OPSIN-
    # unparseable forms -> suppressed). All RT-verified; sanctioned by P-72/P-74.
    # HOO-: BB P-72.2.2.2.2 (line 41031) "The retained names hydroxide, for HO-,
    # and hydroperoxide, for HOO-, are preselected names but cannot be substituted."
    # So the bare dioxidane anion is the preselected name 'hydroperoxide' (the
    # systematic 'dioxidanide' is the alternative). HEAD dropped the charge to the
    # neutral 'dioxidane'; this restores the correct anion word. Substituted
    # peroxol anions (CH3-O-O-) keep the systematic -peroxolate/-dioxidanide path.
    '[O-]O': 'hydroperoxide',
    'N[O-]': 'aminoxide',  # H2N-O- conjugate base of hydroxylamine (P-74); was -> unknown
    'O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F': 'bistriflimide',  # was -> 'triflimidic acid'
}

# === ORGANIC CATIONS (ammonium, carbocations) ===
# Named with -ium, -ylium suffixes
# All keys are in RDKit canonical SMILES form
RETAINED_CATIONS = {
    # Ammonium cations
    '[NH4+]': 'ammonium',
    'C[NH3+]': 'methylammonium',
    'CC[NH3+]': 'ethylammonium',
    'CCC[NH3+]': 'propylammonium',
    'C[NH2+]C': 'dimethylammonium',
    'CC[NH2+]CC': 'diethylammonium',
    'C[NH+](C)C': 'trimethylammonium',
    'C[N+](C)(C)C': 'tetramethylammonium',
    'CC[N+](CC)(CC)CC': 'tetraethylammonium',

    # Guanidinium cation (P-73, Table 7.3)
    'NC(N)=[NH2+]': 'guanidinium',

    # Uronium — protonated urea (P-73.1.2.2; BB 1717/41492/41496: the parent
    # cation 'uronium', NO numerical locants in the PIN). Canonical key is the
    # O-protonated tautomer RDKit picks for NC(=[OH+])N.
    'NC(N)=[OH+]': 'uronium',

    # H2N+ nitrenium (P-73.2.2.1 / BB 41563/42381): the PRESELECTED name is
    # 'azanylium'; 'aminylium'/'nitrenium' are alternatives only. (HEAD emitted
    # 'aminylium' via the general retained-name path — this row wins at
    # CATION_RETAINED@500, ahead of retained_name@1300.)
    '[NH2+]': 'azanylium',

    # Group-14 ylium cations (P-73.2.2.1; BB 41557 lists methylium/propylium/
    # cyclobutylium AND (C6H5)3Si+ 'triphenylsilylium' as PINs — the ylium of the
    # parent hydride silane/germane). Bare [SiH3+]/[GeH3+] preselected forms.
    '[SiH3+]': 'silylium',
    '[GeH3+]': 'germylium',

    # Chalcogen pyrylium cations (BB 41738-41740: thiopyrylium / selenopyrylium /
    # telluropyrylium are ALL (PIN) — retained aromatic-cation parents).
    'c1cc[s+]cc1': 'thiopyrylium',
    'c1cc[se+]cc1': 'selenopyrylium',
    'c1cc[te+]cc1': 'telluropyrylium',

    # === Element-hydride onium cations (P-73.1.1.1) ===
    # The '-onium' names in the BB 41345-41349 table are RETAINED/traditional
    # forms; the PRESELECTED/PIN is the parent-hydride stem + '-ium' ('-anium'),
    # documented verbatim in the BB "preselected name" column
    # (41376/41378/41380/41382, plus 41362 diphenyliodanium PIN and 45996
    # "Arsanium, stibanium... treated... as phosphorus centered cations").
    # We emit the PIN; the retained -onium alternative is noted in-line.
    # H2Cl+ -> chloranium (41382) [alt chloronium]
    # H2Br+ -> bromanium [alt bromonium] (halogen -anium PIN)
    # H2I+ -> iodanium (41362) [alt iodonium]
    # H3Se+ -> selanium (42314 selaniumyl* preselected) [alt selenonium]
    # H3Te+ -> tellanium [alt telluronium] (chalcogen -anium PIN)
    # H4As+ -> arsanium (45996) [alt arsonium]
    # H4Sb+ -> stibanium (45996) [alt stibonium]
    '[ClH2+]': 'chloranium',
    '[BrH2+]': 'bromanium',
    '[IH2+]': 'iodanium',
    '[SeH3+]': 'selanium',
    '[TeH3+]': 'tellanium',
    '[AsH4+]': 'arsanium',
    '[SbH4+]': 'stibanium',

    # Carbocations (carbonium/carbenium ions)
    '[CH3+]': 'methylium',
    '[CH2+]C': 'ethylium',
    '[CH2+]CC': 'propylium',
    '[CH2+]C(C)C': 'isobutylium',
    # PA1 sweep: the row `'C[C+](C)C': 'isopropylium'` was DELETED. WRONG
    # STRUCTURE, as its own former comment ("tert-butyl cation") conceded:
    # C[C+](C)C is the 2-methylpropan-2-ylium / tert-butyl cation, C4H9+, a
    # TERTIARY cation. isopropylium is the propan-2-yl cation C[CH+]C, C3H7+.
    # The formulae differ, so this is not an isomer mix-up but a different
    # compound, and it WAS live: C[C+](C)C emitted "isopropylium". Deleted rather
    # than re-keyed, because 'isopropylium' is itself a non-PIN form (the
    # substitutive PIN is propan-2-ylium, P-73), so re-keying would trade a
    # wrong-structure emission for a non-PIN one. Now falls through to the
    # systematic cation namer.
    '[CH2+]C(C)(C)C': 'neopentylium',  # Primary carbocation adjacent to tert-butyl

    # Aromatic cations
    '[C+]1=CC=CC=C1': 'phenylium',

    # Oxonium cations
    '[OH3+]': 'oxonium',
    'C[OH2+]': 'methyloxonium',
    'C[OH+]C': 'dimethyloxonium',

    # Sulfonium cations
    '[SH3+]': 'sulfonium',  # charged-species fix, 169.6 caveat: parent sulfonium (P-73.1.1.1 Table 7.3); RT-verified
    'C[SH2+]': 'methylsulfonium',
    'C[SH+]C': 'dimethylsulfonium',
    'C[S+](C)C': 'trimethylsulfonium',

    # Phosphonium cations
    # H4P+ PIN is 'phosphanium' (BB 41378/42393: "phosphanium (preselected
    # name) phosphonium"; 41356/42120 confirm...phosphanium (PIN)). Traditional
    # 'phosphonium' is the alternative only.
    '[PH4+]': 'phosphanium',
    'C[PH3+]': 'methylphosphonium',
    'C[P+](C)(C)C': 'tetramethylphosphonium',
}

# === INORGANIC CATIONS (metal ions) ===
# Used for naming salts: sodium chloride, calcium acetate
INORGANIC_CATIONS = {
    # Alkali metals (Group 1)
    '[Li+]': 'lithium',
    '[Na+]': 'sodium',
    '[K+]': 'potassium',
    '[Rb+]': 'rubidium',
    '[Cs+]': 'cesium',

    # Alkaline earth metals (Group 2)
    '[Mg+2]': 'magnesium',
    '[Ca+2]': 'calcium',
    '[Sr+2]': 'strontium',
    '[Ba+2]': 'barium',

    # Transition metals (common oxidation states)
    '[Fe+2]': 'iron(II)',
    '[Fe+3]': 'iron(III)',
    '[Cu+]': 'copper(I)',
    '[Cu+2]': 'copper(II)',
    '[Zn+2]': 'zinc',
    '[Ag+]': 'silver',
    '[Au+]': 'gold(I)',
    '[Au+3]': 'gold(III)',
    '[Mn+2]': 'manganese(II)',
    '[Co+2]': 'cobalt(II)',
    '[Ni+2]': 'nickel(II)',
    '[Cr+3]': 'chromium(III)',

    # Other metals
    '[Al+3]': 'aluminium',
    '[Pb+2]': 'lead(II)',
    '[Sn+2]': 'tin(II)',
    '[Sn+4]': 'tin(IV)',
}

# === INORGANIC ANIONS (halides, hydroxide, etc.) ===
# Used for naming salts
# All keys are in RDKit canonical SMILES form
INORGANIC_ANIONS = {
    # Halides
    '[F-]': 'fluoride',
    '[Cl-]': 'chloride',
    '[Br-]': 'bromide',
    '[I-]': 'iodide',

    # Hydroxide and oxide
    '[OH-]': 'hydroxide',
    '[O-2]': 'oxide',

    # Chalcogenides
    '[S-2]': 'sulfide',
    '[S-]': 'hydrosulfide',  # HS- as [S-] (H implicit)
    '[Se-2]': 'selenide',

    # Nitrogen anions
    '[N-3]': 'nitride',
    '[NH2-]': 'amide',  # Metal amides (not organic amides)
    '[N-]=[N+]=[N-]': 'azide',

    # Carbon anions
    '[C-4]': 'carbide',
    '[C-]#N': 'cyanide',

    # Oxygen-containing anions (canonical forms)
    'O=[N+]([O-])[O-]': 'nitrate',
    # B3 data-bug fix: the old key 'O=[N+][O-]' puts a formal + charge on
    # N with only 3 total bond-order units (RDKit accepts it with an "unusual
    # valence" warning), which is NOT the real nitrite anion -- verified its
    # InChIKey differs from both real nitrite and from OPSIN's own parse of
    # "nitrite" (canonical N(=O)[O-]). The genuine nitrite anion has the
    # negative charge on the terminal oxygen with a NEUTRAL, normal-valence
    # N (O=N[O-]); that is what a real nitrite input canonicalizes to, so the
    # old key was dead (never matched by any real input) AND wrong. Verified
    # RT: opsin_roundtrip_check('O=N[O-]', 'nitrite') passes.
    'O=N[O-]': 'nitrite',
    # PA1 sweep: the row `'O=[SH](=O)[O-]': 'sulfate'` was DELETED. WRONG
    # STRUCTURE: that key is HO3S- (hydrogensulfite / bisulfite, HSO3-), not
    # sulfate (O4S2-). The formulae differ by a whole oxygen. The WS-E.3 fix
    # already diagnosed this key as broken -- see the _canonicalize_anion_table
    # comment below, which names it as "the WS-E.3 sulfate-key bug" -- and added
    # the correct dianion key immediately after, but left the wrong key in place.
    # It was NOT merely dead: 'O=[SH](=O)[O-]' is RDKit-canonical and did match,
    # so bisulfite emitted "sulfate". Removed; the correct sulfate key below, and
    # the distinct 'sulfite'/'hydrogensulfate' rows, are unaffected.
    # SO4(2-) bare dianion of sulfuric acid (P-12.2 / BlueBookV2.md:35449); the
    # legacy 'O=[SH](=O)[O-]' key above was a -1 [SH] form that never matched the
    # real fully-deprotonated dianion '[O-]S(=O)(=O)[O-]' (canonical O=S(=O)([O-])[O-]).
    'O=S(=O)([O-])[O-]': 'sulfate',
    # HSO4- mono-anion (P-12.2 / BlueBookV2.md:7150); MUST stay distinct from
    # 'sulfate' (different protonation state). OPSIN round-trips 'hydrogensulfate'
    # -> S(=O)(=O)(O)[O-] (the correct mono-anion).
    'O=S(=O)([O-])O': 'hydrogensulfate',
    'O=S([O-])[O-]': 'sulfite',
    # B3 (e): genuine missing row -- the real bisulfite/hydrogensulfite
    # mono-anion (HSO3-, the O-protonated tautomer of sulfurous acid's
    # conjugate base). Verified RT: opsin_roundtrip_check('O=S([O-])O',
    # 'hydrogensulfite') passes (OPSIN parses to the same InChIKey). Named
    # 'hydrogensulfite' to match the existing 'hydrogensulfate' row's style
    # rather than the informal 'bisulfite'.
    'O=S([O-])O': 'hydrogensulfite',
    'O=P([O-])([O-])[O-]': 'phosphate',
    'O=P([O-])([O-])O': 'hydrogen phosphate',
    'O=P([O-])(O)O': 'dihydrogen phosphate',
    # B3 (e): genuine missing row -- the fully-deprotonated dianion of
    # phosphonic acid's P-H tautomer (HPO3(2-)). Verified RT:
    # opsin_roundtrip_check('O=[PH]([O-])[O-]', 'phosphonate') passes.
    'O=[PH]([O-])[O-]': 'phosphonate',
    'O=C([O-])[O-]': 'carbonate',
    'O=C([O-])O': 'hydrogen carbonate',
    # B3 (d) DATA BUG FIX: this SMILES is ClO2- (2 oxygens) = chlorite,
    # NOT chlorate (ClO3-, 3 oxygens). Verified: OPSIN round-trip of
    # 'chlorate' against this SMILES FAILS (wrong InChIKey); 'chlorite'
    # PASSES. The wrong name was previously caught downstream by the SELF-01
    # full round-trip gate (0-wrong held), surfacing only as the
    # 'unknown organic compound' sentinel instead of the correct word.
    '[O-][Cl+][O-]': 'chlorite',  # RDKit canonical form
}


def _canonicalize_anion_table(table):
    # Re-key an anion lookup table by RDKit-canonical SMILES so a hand-written
    # non-canonical key can never silently fail to match a canonicalized input
    # (the WS-E.3 sulfate-key bug: 'O=[SH](=O)[O-]' never matched the bare -2
    # dianion 'O=S(=O)([O-])[O-]'). Keys that fail to parse are kept verbatim.
    out = {}
    for smi, word in table.items():
        m = Chem.MolFromSmiles(smi)
        out[Chem.MolToSmiles(m) if m is not None else smi] = word
    return out


INORGANIC_ANIONS = _canonicalize_anion_table(INORGANIC_ANIONS)
RETAINED_ANIONS = _canonicalize_anion_table(RETAINED_ANIONS)


# P-65.6.2.1 / P-65.6.1.1: dicarboxylate anion names that are RETAINED FOR GENERAL
# NOMENCLATURE ONLY — their preferred IUPAC name is the SYSTEMATIC '-dioate'
# (malonate -> propanedioate, BB:29761 'propanedioic acid (PIN)'; succinate ->
# butanedioate, BB:29793/31575 'potassium sodium butanedioate (PIN)'). In PIN
# style get_anion_name(pin=True) returns None for these, so the caller falls
# through to the systematic neutralize->name path (which yields '-dioate').
# Retained anions that ARE PINs — acetate/formate/benzoate (retained-PIN acids),
# oxalate (BB:29723 'oxalic acid (PIN)'), carbonate (BB 'disodium carbonate
# (PIN)') — are deliberately NOT here and keep their retained name in every style.
_GENERAL_ONLY_ANIONS = frozenset(
    Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in (
        'O=C([O-])CC(=O)[O-]',    # malonate -> propanedioate (PIN)
        'O=C([O-])CCC(=O)[O-]',   # succinate -> butanedioate (PIN)
    )
)


def get_anion_name(smiles: str, pin: bool = False) -> Optional[str]:
    """
    Look up retained name for an anion by its SMILES.

    The input SMILES is canonicalized before lookup to ensure
    consistent matching regardless of input format.

    Args:
        smiles: SMILES string (will be canonicalized)

    Returns:
        Retained name if found, None otherwise

    Example:
        >>> get_anion_name('CC(=O)[O-]')
        'acetate'
        >>> get_anion_name('[Cl-]')
        'chloride'
    """
    # Canonicalize input SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    canonical = Chem.MolToSmiles(mol)

    # PIN style: a dicarboxylate anion that is retained FOR GENERAL NOMENCLATURE
    # ONLY (malonate/succinate) has no retained PIN -> signal fall-through to the
    # systematic '-dioate' path (P-65.6.1.1). Retained-PIN anions are unaffected.
    if pin and canonical in _GENERAL_ONLY_ANIONS:
        return None

    # Check organic anions first
    if canonical in RETAINED_ANIONS:
        return RETAINED_ANIONS[canonical]

    # Check inorganic anions
    if canonical in INORGANIC_ANIONS:
        return INORGANIC_ANIONS[canonical]

    return None


# Cluster C (P-73.1.2.1): the alkyl/dialkyl/trialkyl(tetraalkyl)-ammonium
# retained names are GENERAL nomenclature only, NOT preferred IUPAC names -- the
# PIN is the substitutive aminium (methylammonium -> methanaminium; dimethyl-
# ammonium -> N-methylmethanaminium; tetramethylammonium -> N,N,N-trimethyl-
# methanaminium). BB: line 26672 'methanaminium chloride (PIN)' over 'methyl-
# azanium'; line 41354 '(CH3)4N+... N,N,N-trimethylmethanaminium (PIN)' over
# 'tetramethylammonium'. On the PIN path these keys are denied so the systematic
# route_charged aminium name wins; they remain available for general/common style.
# 'ammonium' (NH4+) IS a genuine retained PIN and is deliberately NOT listed.
PIN_NONPREFERRED_CATIONS = frozenset({
    'C[NH3+]', 'CC[NH3+]', 'CCC[NH3+]', 'C[NH2+]C', 'CC[NH2+]CC',
    'C[NH+](C)C', 'C[N+](C)(C)C', 'CC[N+](CC)(CC)CC',
})


def get_cation_name(smiles: str) -> Optional[str]:
    """
    Look up retained name for a cation by its SMILES.

    The input SMILES is canonicalized before lookup to ensure
    consistent matching regardless of input format.

    Args:
        smiles: SMILES string (will be canonicalized)

    Returns:
        Retained name if found, None otherwise

    Example:
        >>> get_cation_name('[NH4+]')
        'ammonium'
        >>> get_cation_name('[Na+]')
        'sodium'
    """
    # Canonicalize input SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    canonical = Chem.MolToSmiles(mol)

    # Check organic cations first
    if canonical in RETAINED_CATIONS:
        return RETAINED_CATIONS[canonical]

    # Check inorganic cations
    if canonical in INORGANIC_CATIONS:
        return INORGANIC_CATIONS[canonical]

    return None


def get_ion_name(smiles: str) -> Optional[str]:
    """
    Look up retained name for any ion (cation or anion).

    Convenience function that checks both cation and anion lookups.

    Args:
        smiles: SMILES string (will be canonicalized)

    Returns:
        Retained name if found, None otherwise

    Example:
        >>> get_ion_name('[Na+]')
        'sodium'
        >>> get_ion_name('[Cl-]')
        'chloride'
    """
    # Try cation lookup
    name = get_cation_name(smiles)
    if name:
        return name

    # Try anion lookup
    return get_anion_name(smiles)
