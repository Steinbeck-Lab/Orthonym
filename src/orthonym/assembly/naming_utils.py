"""
Name assembly utility functions for IUPAC name generation.

Pure functions for:
- Alkyl substituent naming (methyl through decyl)
- Substituent prefix formatting with locants and multipliers
- Alphabetization sort keys (IUPAC rules for prefix ordering)
- Complex substituent detection and multiplier selection
- Vowel elision (terminal 'e' removal before vowel suffixes)
- Suffix with locants formatting (PIN infix style)

All functions are pure: they accept locants as integers (not atom indices).
The mapping from atom indices to locants is handled elsewhere.
"""

import re
from typing import List


# ============================================================================
# Pre-compiled patterns and constants (hoisted from function bodies)
# ============================================================================

# Used by is_complex_substituent() — compound multiplier+substituent patterns
_MULT_SUBSTITUENT_RE = re.compile(
    r'^(?:di|tri|tetra|penta|hexa)'
    r'(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|'
    r'phenyl|naphthyl|cyclopentyl|cyclohexyl|benzyl|vinyl|allyl|'
    r'fluoro|chloro|bromo|iodo)'
)

# Halogen + alkyl compound substituent patterns (e.g., fluoromethyl,
# trifluoromethyl, chloroethyl). These are compound substituents per
# IUPAC P-31.1.2.3 and require enclosing marks.
_HALOALKYL_RE = re.compile(
    r'^(?:(?:di|tri|tetra|penta|hexa)?(?:fluoro|chloro|bromo|iodo))'
    r'(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|nonyl|decyl|'
    r'undecyl|dodecyl|tridecyl|tetradecyl|pentadecyl|'
    r'hexadecyl|heptadecyl|octadecyl|nonadecyl|icosyl)$'
)

# Alkyl + functional group compound substituent patterns (e.g., methylamino,
# ethylamino, propylamino). These are compound substituents per IUPAC P-31.1.2
# and require enclosing marks.
_ALKYLAMINO_RE = re.compile(
    r'^(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|'
    r'nonyl|decyl|undecyl|dodecyl|tridecyl|tetradecyl|pentadecyl|'
    r'hexadecyl|heptadecyl|octadecyl|nonadecyl|icosyl|'
    r'phenyl|benzyl|cyclopentyl|cyclohexyl)'
    r'(?:amino|imino)$'
)

# Phase 4 SUBST-01: a substituted substituent whose carrier is itself a '-yl'
# group bearing a terminal alkyl-yl tail ('cyclohexylmethyl', 'cyclopentylethyl',
# 'piperidinylmethyl') is a compound prefix requiring enclosing marks
# ('(cyclohexylmethyl)benzene', P-16.3.3). The internal 'yl' before the terminal
# alkyl marks the compound boundary; simple alkyls ('methyl') lack it, and FG
# heads ('hydroxymethyl') do not end the head in 'yl'.
_RINGYL_ALKYL_RE = re.compile(
    r'yl(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl)$'
)

_COMPOUND_OXY_PREFIXES = frozenset((
    'sulfooxy', 'sulfonyloxy', 'phosphonooxy', 'phosphonatoxy', 'carbonyloxy',
))

# BUG-B guard: FG types that substituent naming demonstrably handles on 1-3C branches.
# Each entry verified empirically (Phase 105-01 + Phase 113-01) to produce correct
# prefix via name_substituent() on small branches in BOTH polyfunctional.py and
# composer.py code paths. FG types NOT in this set stay in the polyfunctional
# prefix list (IUPAC P-59.1).
#
# NOTE (W2F-P3): 'nitro' is now IN this set. The historical exclusion (drops
# sulfanyl/nitro/amino/azido on small branches) was caused by
# composer.py's carbon-count terminal fallback returning alkyl names; with that
# fallback deleted (composer.py:8704-8711 -> None) the Tier-4 enumerator names
# nitro branches faithfully ('nitromethyl' / '1,2-dinitropropyl', verified). thiol,
# azido, secondary_amine, tertiary_amine stay OUT (their branch-path behaviour is
# unverified — separate follow-on). FG types NOT in this set stay in the
# polyfunctional prefix list (IUPAC P-59.1).
BRANCH_HANDLED_FGS: frozenset = frozenset({
    'primary_alcohol',    # -> "hydroxymethyl", "2-hydroxypropyl"
    'secondary_alcohol',  # -> "hydroxy" included in branch name
    'primary_amine',      # -> "aminomethyl", "2-aminoethyl"
    'fluoro', 'chloro', 'bromo', 'iodo',  # -> "fluoromethyl" etc.
    'nitro',              # W2F-P3 -> "nitromethyl", "1,2-dinitropropyl"
})

# Shared C1-C20 alkyl roots used by needs_brackets(), is_complex_substituent(),
# and regex patterns. Extends coverage beyond the original C1-C10 lists.
_ALKYL_ROOTS_FULL = (
    'methyl', 'ethyl', 'propyl', 'butyl', 'pentyl',
    'hexyl', 'heptyl', 'octyl', 'nonyl', 'decyl',
    'undecyl', 'dodecyl', 'tridecyl', 'tetradecyl', 'pentadecyl',
    'hexadecyl', 'heptadecyl', 'octadecyl', 'nonadecyl', 'icosyl',
)

# Chalcogen compound-substituent suffixes that, fused to an alkyl root, form a
# COMPLEX (compound) substituent requiring enclosing marks per IUPAC P-16.3.3 /
# P-63.6 — '(methylsulfanyl)', '(methylselanyl)', '(methyltellanyl)'. Phase 171
# BBR-ASM (DEF-8): the Se/Te analogues (selanyl/tellanyl + the oxidized
# seleninyl/selenonyl/tellurinyl/telluronyl) were missing, so 'methylselanyl' was
# wrongly classed simple -> '1-methylselanylpropane' instead of PIN
# '1-(methylselanyl)propane'.
_COMPOUND_S_SUFFIXES_COMPLEX = (
    'sulfinyl', 'sulfonyl', 'sulfanyl',
    'seleninyl', 'selenonyl', 'selanyl',
    'tellurinyl', 'telluronyl', 'tellanyl',
    # DD2 Fix B (Phase D, P-63.3.1(1)): (alkyl)disulfanyl is a compound
    # substituent — '(methyldisulfanyl)methane'. Bare 'disulfanyl' (terminal
    # -S-SH, no alkyl root) is simple and is not matched by the root loop.
    'disulfanyl',
)

# Functional group prefixes that, when fused with alkyl roots, form compound
# substituents requiring enclosing marks per IUPAC P-14.5.2.
# Hoisted to module level for performance (was recreated inside needs_brackets).
_COMPOUND_FG_PREFIXES = (
    'hydroxy', 'carboxy', 'amino', 'oxo', 'formyl', 'cyano',
    'nitro', 'mercapto', 'sulfanyl', 'phospho',
    # Wave-2 C2 (P-16.3.3): '(isothiocyanatomethyl)benzene' BB verbatim
    'isothiocyanato', 'isocyanato',
    'fluoro', 'chloro', 'bromo', 'iodo',
    'difluoro', 'trifluoro', 'dichloro', 'trichloro',
    'dibromo', 'tribromo',
)

# Wave2 T2b: heteroatom-substituted amino/oxy preselected prefixes
# (P-68.3.1.1.1.5 / P-35.3.1) — compound substituents that take enclosing
# marks per P-16.3.3. BB PINs: '4-(hydroxyamino)phenol',
# '2-(aminooxy)ethan-1-amine', '8-(chloroamino)octanoic acid'.
_COMPOUND_HETEROATOM_AMINO_PREFIXES = frozenset({
    'hydroxyamino', 'aminooxy',
    'fluoroamino', 'chloroamino', 'bromoamino', 'iodoamino',
})

# Wave2 T3b: parent-hydride stems used as roots of the PIN acid-stem
# chalcogen-oxide prefixes ('methanesulfinyl' / 'benzenesulfonyl' /
# 'cyclohexanesulfinyl', P-63.6). Consumed by needs_brackets alongside
# _ALKYL_ROOTS_FULL so those compound prefixes take enclosing marks.
_HYDRIDE_STEM_ROOTS = tuple(
    f"{p}ane" for p in (
        'meth', 'eth', 'prop', 'but', 'pent', 'hex', 'hept', 'oct', 'non',
        'dec', 'undec', 'dodec', 'tridec', 'tetradec', 'pentadec', 'hexadec',
        'heptadec', 'octadec', 'nonadec', 'icos',
    )
) + ('benzene',) + tuple(
    f"cyclo{p}ane" for p in ('prop', 'but', 'pent', 'hex', 'hept', 'oct')
)

# Used by alpha_sort_key() — pre-compiled regex patterns
_LOCANT_PREFIX_RE = re.compile(r'^[\d,]+-')
_N_LOCANT_PREFIX_RE = re.compile(r'^[nN],?[nN]?-')
# Leading indicated-hydrogen descriptor (e.g. '1H-', '2H-'). Like a locant, the
# italic indicated H is IGNORED for alphabetization (IUPAC P-14.5.2), so
# '(1H-imidazol-5-yl)' must sort at 'i', not at the leading digit '1'.
_INDICATED_H_PREFIX_RE = re.compile(r'^\d+[hH]-')


# ============================================================================
# Terminal Functional Group Types (IUPAC P-14.3.4.1)
# ============================================================================

# Terminal functional groups that always occupy position 1 by definition.
# The chain is numbered FROM the terminal group, so locant-1 is implicit.
# Canonical source: IUPAC P-14.3.4.1. Used by should_omit_locant_one().
def unbranched_alkylidene_name(mol, c_idx, exclude_idx):
    """Ylidene name for an UNBRANCHED all-carbon H-saturated chain rooted at
    the double-bonded carbon *c_idx* (walking away from *exclude_idx*):
    methylidene / ethylidene / propylidene (Wave-2 completion C; shared by the
    sulfine P-64.4.2 and azinic-acid P-61.5.3 namers). Fail-closed None on
    branching, heteroatoms, rings, charges, or further unsaturation."""
    from rdkit import Chem as _Chem
    from ..data.chain_names import get_chain_prefix as _gcp
    seen = {c_idx}
    prev, cur, length = exclude_idx, c_idx, 1
    while True:
        atom = mol.GetAtomWithIdx(cur)
        if (atom.GetAtomicNum() != 6 or atom.GetFormalCharge() != 0
                or atom.IsInRing()):
            return None
        for b in atom.GetBonds():
            other = b.GetOtherAtomIdx(cur)
            if other == prev and cur == c_idx:
                continue  # the defining double bond
            if b.GetBondType() != _Chem.BondType.SINGLE and other != prev:
                return None
        nxts = [n.GetIdx() for n in atom.GetNeighbors()
                if n.GetAtomicNum() > 1 and n.GetIdx() != prev
                and n.GetIdx() not in seen]
        if len(nxts) > 1:
            return None
        if not nxts:
            break
        seen.add(nxts[0])
        prev, cur = cur, nxts[0]
        length += 1
    prefix = _gcp(length)
    return f"{prefix}ylidene" if prefix else None


TERMINAL_FG_TYPES = frozenset({
    "carboxylic_acid",  # Always at chain end (locant 1)
    "peroxy_acid",      # Always at chain end (P-43.1: propaneperoxoic acid)
    "imidic_acid",      # Always at chain end (P-65.1.3.1: ethanimidic acid)
    "hydrazonic_acid",  # Always at chain end (P-65.1.3.2: methanehydrazonic acid)
    "aldehyde",         # Always at chain end (locant 1)
    "nitrile",          # Always at chain end (locant 1)
    "primary_amide",    # Always at chain end (locant 1)
    "secondary_amide",  # Always at chain end (locant 1)
    "tertiary_amide",   # Always at chain end (locant 1)
    "acid_chloride",    # Always at chain end (locant 1)
    "acid_bromide",     # Always at chain end (locant 1)
    "acid_fluoride",    # Always at chain end (locant 1)
    "thioic_S_acid",    # Always at chain end (locant 1)
    "thioic_O_acid",    # Always at chain end (locant 1)
    "dithioic_acid",    # Always at chain end (locant 1)
    "carbamic_acid",    # Retained name, terminal (locant 1)
    # Phase 163 Tier FRN-A chalcogen acids (P-66.6.3 functional replacement; IUPAC PIN
    # 'propaneselenoic Se-acid' / 'propanetelluroic Te-acid' — never carries locant-1)
    "selenoic_Se_acid", "selenoic_O_acid", "diselenoic_acid",
    "telluroic_Te_acid", "telluroic_O_acid", "ditelluroic_acid",
    # Phase 163 Tier FRN-B chalcogen amides (P-66.1.4.1.1 functional replacement; IUPAC PIN
    # 'propanethioamide' / 'propaneselenoamide' — never carries locant-1)
    "thioamide", "selenoamide", "telluroamide",
    # Phase 163 Tier FRN-C chalcogen aldehydes (P-66.6.3 — "propanethial" / "propaneselenal" /
    # "propanetellural"; suffix is "-thial"/"-selenal"/"-tellural" per seniority.py SUFFIX_FORMS,
    # but FG identifier is the long form. Always terminal — never carries locant-1.)
    "thioaldehyde", "selenoaldehyde", "telluroaldehyde",
    # D-FOLLOWON item 8 (P-66.4.1): the amidine (imidamide / carboximidamide)
    # characteristic carbon is ALWAYS chain-terminal (C1), exactly like
    # nitrile / amide / aldehyde — so its suffix locant-1 is implicit and elided:
    # 'propanimidamide' not 'propan-1-imidamide'.
    "amidine",
    # R8a (P-66.3.1.1): the hydrazide characteristic carbon is ALWAYS chain-terminal
    # (C1), exactly like amide / nitrile / aldehyde — so its suffix locant-1 is
    # implicit and elided: 'pentanehydrazide' not 'pentane-1-hydrazide'.
    "hydrazide",
    # Wave2 T3d: the amidrazone (hydrazonamide) / hydrazidine (hydrazonohydrazide)
    # / thiohydrazide characteristic carbon is likewise ALWAYS chain-terminal —
    # 'ethanehydrazonamide' / 'ethanehydrazonohydrazide' / 'ethanethiohydrazide',
    # never a '-1-' locant (parallel to amidine/hydrazide).
    "hydrazonamide",
    "hydrazidine",
    "thiohydrazide",
})


# ============================================================================
# Centralized Locant-1 Elision (IUPAC P-14.3.4)
# ============================================================================


# Amine-family suffix FG classes whose locant elides on a symmetric 2-carbon
# (ethane) parent (P-14.3.4.4): 'ethanamine' not 'ethan-1-amine'. Scoped to the
# amine family ON PURPOSE — it is the only suffix that routes through the general
# _generate_suffix path AND needs this elision (DD1 Fix 4 masking pair). -ol /
# -thiol have dedicated handlers that already elide the bare case ('ethanol',
# 'ethanethiol'); including them here changed the SUBSTITUTED-alcohol locant
# (2-phenylethan-1-ol -> 2-phenylethanol), a separate unsettled question outside
# Phase B. Carbonyl-type suffixes (-one/-al) always cite the locant in PINs
# (P-31.1.4, 1-phenylethan-1-one) and are likewise excluded.
_ETHANE_SUFFIX_ELIDE_FGS = frozenset({
    "primary_amine", "secondary_amine", "tertiary_amine",
    # DD2 (Phase D, P-56.1 / P-63.4): the peroxol family is a single monovalent
    # chalcogen suffix on a terminal carbon — exactly like -amine — so on a
    # monosubstituted ethane the two carbons are equivalent and the locant is
    # omitted: 'ethaneperoxol' (P-56.1 verbatim), 'ethanedithioperoxol', not
    # 'ethane-1-peroxol'. These route through _generate_suffix (no dedicated
    # handler), so the elision must be declared here.
    "hydroperoxide", "so_thioperoxol", "os_thioperoxol", "dithioperoxol",
    # D-FOLLOWON item 1 (P-14.3.4.4 / P-65.3.1): the S/Se/Te oxoacid suffixes
    # (-sulfinic/-sulfonic/-selenonic/-seleninic/-telluronic/-tellurinic acid)
    # attach to a chain carbon whose locant is now computed (the _pick_locant_atom
    # neighbor-walk fix). On the symmetric 2-carbon (ethane) parent the two carbons
    # are equivalent and the suffix sits on a terminal carbon, so the locant is
    # omitted: 'ethanesulfinic acid' / 'ethaneselenonic acid' (PIN), NOT
    # 'ethane-1-sulfinic acid'. methane (chain_length==1) is already elided by
    # Rule 1; a chain of 3+ carbons (butane-2-sulfinic acid) keeps its locant.
    "sulfinic_acid", "sulfonic_acid",
    "selenonic_acid", "seleninic_acid", "telluronic_acid", "tellurinic_acid",
    # C1 (P-14.3.4.4 / P-65.3.1): sulfonohydrazide is the N-analogue of sulfonic
    # acid and attaches to a chain carbon exactly the same way, so it elides its
    # locant on a symmetric 2-carbon (ethane) parent: 'ethanesulfonohydrazide'
    # (PIN), not 'ethane-1-sulfonohydrazide'. methane is covered by Rule 1; a
    # 3+ carbon chain keeps its locant (parallels sulfonohydrazide above).
    "sulfonohydrazide",
    # Wave2 T1b (P-66.1.1.2 / P-14.3.4.4): the sulfonamide characteristic S attaches
    # to a chain carbon exactly like sulfonic_acid, so its suffix locant elides on the
    # symmetric 2-carbon (ethane) parent -> 'ethanesulfonamide' (PIN, OPSIN-RT), not
    # 'ethane-1-sulfonamide'; N-substituents don't perturb the ethane symmetry
    # (N-methylethanesulfonamide). A 3+ carbon chain keeps its locant
    # (propane-1-sulfonamide). Sulfinamide (no FG defined yet) deferred.
    "primary_sulfonamide", "secondary_sulfonamide", "tertiary_sulfonamide",
    # Wave2 T3d (P-66.1.1 / P-14.3.4.4): sulfonimidamide attaches to a chain
    # carbon exactly like sulfonamide -> 'ethanesulfonimidamide' (OPSIN-RT),
    # not 'ethane-1-sulfonimidamide'. propane-1- keeps its locant.
    "sulfonimidamide",
    # Wave2 T3d (P-66.1.1 item 25): sulfinimidamide, S-suffix parallel ->
    # 'ethanesulfinimidamide' (propane-1- keeps its locant).
    "sulfinimidamide",
    # Wave-2 P1AM (P-66.4.1.1): Se imidamide suffixes attach to a chain carbon
    # like the S siblings -> 'methaneseleninimidamide' / 'methaneselenonimidamide'.
    "seleninimidamide",
    "selenonimidamide",
    # Wave-2 P1AM Task 7 (P-66.4.2.1 / P-14.3.4.1, BB 2877): amidrazones are
    # cited without terminal locants -> 'methanimidohydrazide' (not
    # 'methan-1-imidohydrazide').
    "imidohydrazide",
    # Wave2 T2a (P-62.3.1.1 / P-14.3.4.4): the imine =NH sits on a chain carbon;
    # on the symmetric 2-carbon (ethane) parent the locant elides -> 'ethanimine'
    # (BB VERBATIM 'N-methylethanimine (PIN)'), not 'ethan-1-imine'. A 3+ carbon
    # chain keeps its locant (BB VERBATIM 'N-hydroxypropan-1-imine (PIN)').
    "imine",
    # Wave2 T1b (P-63.1.5 / P-14.3.4.4): selenol/tellurol are monovalent-chalcogen
    # suffixes exactly like -ol/-thiol, so the locant elides on the symmetric 2-carbon
    # (ethane) parent -> 'ethaneselenol'/'ethanetellurol' (PIN, OPSIN-RT), not
    # 'ethane-1-selenol'. Unlike -ol/-thiol (which have dedicated handlers that already
    # elide), selenol/tellurol route through the generic SUFFIX_FORMS path and so must
    # be declared here (Rule 3b's docstring already anticipates them). propan-1-selenol
    # (3-carbon) and substituted 2-X-ethaneselenol (is_monosubstituted False) keep the locant.
    "selenol", "tellurol",
})


def should_omit_locant_one(
    *,
    context: str,
    chain_length: int = 0,
    is_ring: bool = False,
    is_heterocyclic: bool = False,
    is_monosubstituted: bool = False,
    fg_type: str = "",
) -> bool:
    """Determine whether locant-1 should be omitted per IUPAC P-14.3.4.

    Centralized decision point for ALL locant-1 elision in the pipeline.
    Every call site that decides whether to omit locant-1 must use this
    function rather than reimplementing the logic inline.

    Args:
        context: One of "suffix", "prefix", or "bond".
        chain_length: Length of the parent chain (0 for rings).
        is_ring: True if the parent is a ring system.
        is_heterocyclic: True if the ring contains heteroatoms.
        is_monosubstituted: True if only one substituent/FG is present.
        fg_type: Functional group type string (e.g., "carboxylic_acid").

    Returns:
        True if locant-1 should be omitted from the name.
    """
    # Rule 1: Methane derivatives (chain_length=1): always omit
    # Only position exists, so locant is always trivially 1.
    if chain_length == 1:
        return True

    # Rule 2: di-/trinuclear bond locants (P-14.3.4.2(d)): omit bond locant.
    # ethene/ethyne (chain_length 2, only one bond position) always omit;
    # propene/propyne (chain_length 3) omit ONLY when the caller passes
    # chain_length=3, which it does solely for an UNSUBSTITUTED trinuclear
    # parent (Wave2 T6a — a substituent makes the position distinctive:
    # '3-chloroprop-1-ene' keeps its locant).
    if context == "bond" and chain_length in (2, 3):
        return True

    # Rule 3: Terminal groups: suffix locant-1 is implicit
    # Chain is numbered from the terminal group (acid, aldehyde, nitrile, etc.)
    if context == "suffix" and fg_type in TERMINAL_FG_TYPES:
        return True

    # Rule 3b: A single monovalent-heteroatom suffix (-amine/-ol/-thiol/-selenol/
    # -tellurol) on a 2-carbon (ethane) parent. The two carbons of ethane are
    # equivalent and the group sits on a terminal carbon, so the locant is always
    # 1 and is omitted (P-14.3.4.4): 'ethanamine'/'ethanol'/'ethanethiol', not
    # 'ethan-1-amine'. This is FG-class-gated on purpose: carbonyl-type suffixes
    # (-one/-al) ALWAYS cite their locant in PINs (P-31.1.4 — '1-phenylethan-1-one',
    # 'propan-2-one'), so a ketone on ethane must NOT elide. methane
    # (chain_length==1) is covered unconditionally by Rule 1; multi-instance
    # suffixes (ethane-1,2-diamine) keep locants (is_monosubstituted is False).
    if (context == "suffix" and not is_ring
            and chain_length == 2 and is_monosubstituted
            and fg_type in _ETHANE_SUFFIX_ELIDE_FGS):
        return True

    # Rule 4: Monosubstituted rings. WSD-06: the ring callers now pass
    # `is_monosubstituted` = `is_only_one_substitutable_position(parent_hydride)`
    # (the real topological-symmetry predicate below) AND-ed with the count, so a
    # substituted ASYMMETRIC ring (e.g. cyclohexene) correctly KEEPS the locant.
    if context == "prefix" and is_ring and is_monosubstituted:
        if is_heterocyclic:
            return False  # Position matters in heterocycles
        return True  # Symmetric carbocyclic: omit

    # Rule 5: Monosubstituted hydrocarbon chain at position 1.
    # P-14.3.4 (PIN): the locant '1' is omitted ONLY where the position is
    # unambiguous — methane (Rule 1, chain_length==1) and ETHANE (chain_length==2,
    # the two carbons are equivalent so there is a single monosubstitution product
    # -> 'chloroethane'). For propane and longer the terminal substituent is
    # distinguishable from interior positions ('1-chloropropane' != '2-chloropropane',
    # '1-chloropentane' is the PIN), so the locant MUST be cited. Phase 171 DEF-4
    # (BlueBookV2 P-14.3.4 @2869; gold 'ClCCCCC' -> '1-chloropentane'). Previously this
    # used `chain_length > 1`, which wrongly elided the locant for all chains.
    if context == "prefix" and not is_ring and is_monosubstituted and chain_length == 2:
        return True

    # Rule 6: Mono-cycloalkene bond locant
    # Single double bond in a ring: locant omitted (cyclohexene, not cyclohex-1-ene)
    if context == "bond" and is_ring and is_monosubstituted:
        return True

    return False


def is_only_one_substitutable_position(parent_mol) -> bool:
    """WSD-06 (Phase 175): topological-symmetry predicate for locant-1 elision.

    Returns True iff every substitutable ring position of ``parent_mol`` is
    equivalent — i.e. all H-bearing ring carbons share ONE RDKit canonical rank
    (``CanonicalRankAtoms(breakTies=False)`` = symmetry equivalence classes). This
    is the real Blue Book condition (P-14.3.4.3 "only one kind of substitutable
    hydrogen" / P-14.3.4.4 "no isomer generated by moving") that replaces the
    ``num_double==1`` / ring-monosubstituted COUNT PROXIES, which conflate "one
    feature" with "ring is symmetric".

    ``parent_mol`` MUST be the PARENT HYDRIDE for the rule being decided, NOT the
    decorated molecule:
      * substituent-prefix locant (P-14.3.4.3): ring + unsaturation, substituents
        stripped (cyclohexane -> True -> methylcyclohexane; cyclohexene -> False ->
        3-bromocyclohex-1-ene; benzene -> True -> toluene).
      * bond locant (P-14.3.4.2(d)/4.4): the SATURATED ring WITH substituents (so
        an unsubstituted cyclohexene -> cyclohexane -> True -> omit -> 'cyclohexene';
        bromocyclohexene -> bromocyclohexane -> False -> keep the ene-locant).
    """
    if parent_mol is None:
        return False
    try:
        from rdkit import Chem
        ranks = list(Chem.CanonicalRankAtoms(parent_mol, breakTies=False))
    except Exception:
        return False
    ring_atoms = set()
    for ring in parent_mol.GetRingInfo().AtomRings():
        ring_atoms.update(ring)
    subpos = [
        i for i in ring_atoms
        if parent_mol.GetAtomWithIdx(i).GetSymbol() == 'C'
        and parent_mol.GetAtomWithIdx(i).GetTotalNumHs() >= 1
    ]
    if not subpos:
        return False
    return len({ranks[i] for i in subpos}) == 1


# ============================================================================
# Alkyl Substituent Names
# ============================================================================

ALKYL_NAMES = {
    1: "methyl",
    2: "ethyl",
    3: "propyl",
    4: "butyl",
    5: "pentyl",
    6: "hexyl",
    7: "heptyl",
    8: "octyl",
    9: "nonyl",
    10: "decyl",
}


def get_alkyl_name(carbon_count: int) -> str:
    """Get alkyl substituent name for a given carbon count.

    Returns 'methyl' for 1, 'ethyl' for 2, ..., 'decyl' for 10.
    For carbon counts > 10, delegates to the centralized chain_names module
    which supports up to 999 carbons using IUPAC compositional naming.

    Args:
        carbon_count: Number of carbons in the alkyl chain (1-999).

    Returns:
        Alkyl substituent name string.

    Raises:
        ValueError: If carbon_count is outside supported range.
    """
    if carbon_count in ALKYL_NAMES:
        return ALKYL_NAMES[carbon_count]
    # Delegate to centralized module for chains > 10
    from ..data.chain_names import get_alkyl_name as _get_alkyl_name
    return _get_alkyl_name(carbon_count)


# ============================================================================
# Multiplier Prefixes
# ============================================================================

# Simple multiplicative prefixes (for simple substituent names)
SIMPLE_MULTIPLIERS = {
    2: "di",
    3: "tri",
    4: "tetra",
    5: "penta",
    6: "hexa",
    7: "hepta",
    8: "octa",
    9: "nona",
    10: "deca",
    11: "undeca",
    12: "dodeca",
    13: "trideca",
    14: "tetradeca",
    15: "pentadeca",
    16: "hexadeca",
    17: "heptadeca",
    18: "octadeca",
    19: "nonadeca",
    20: "icosa",
}

# Complex multiplicative prefixes (for substituents with locants/hyphens)
COMPLEX_MULTIPLIERS = {
    2: "bis",
    3: "tris",
    4: "tetrakis",
    5: "pentakis",
    6: "hexakis",
    7: "heptakis",
    8: "octakis",
    9: "nonakis",
    10: "decakis",
    11: "undecakis",
    12: "dodecakis",
    13: "tridecakis",
    14: "tetradecakis",
    15: "pentadecakis",
    16: "hexadecakis",
    17: "heptadecakis",
    18: "octadecakis",
    19: "nonadecakis",
    20: "icosakis",
}


def needs_brackets(name: str) -> bool:
    """Determine if a substituent name is a compound substituent needing parentheses.

    Per IUPAC P-14.5.2, compound substituents (those that contain locants,
    hyphens, or functional group prefixes fused with alkyl names) must be
    enclosed in parentheses when used as prefixes on ring parents.

    Simple substituents (single-word names like methyl, chloro, hydroxy)
    do NOT need parentheses.

    Already-bracketed names (starting with '(' or '[') are left alone.

    Args:
        name: The substituent name (e.g., 'methyl', 'hydroxymethyl',
              '2-methylpropyl', '(N,N-dimethylamino)').

    Returns:
        True if the substituent needs enclosing parentheses, False otherwise.

    Examples:
        >>> needs_brackets("methyl")
        False
        >>> needs_brackets("chloro")
        False
        >>> needs_brackets("hydroxy")
        False
        >>> needs_brackets("hydroxymethyl")
        True
        >>> needs_brackets("carboxymethyl")
        True
        >>> needs_brackets("aminoethyl")
        True
        >>> needs_brackets("2-methylpropyl")
        True
        >>> needs_brackets("(N,N-dimethylamino)")
        False
        >>> needs_brackets("methoxy")
        False
    """
    if not name:
        return False

    # Already wrapped in parentheses or square brackets -- skip
    if (name.startswith('(') and name.endswith(')')) or \
       (name.startswith('[') and name.endswith(']')):
        return False

    # Contains a digit (has locants): definitely compound
    if any(ch.isdigit() for ch in name):
        return True

    # Contains a hyphen (compound substituent): definitely compound
    if '-' in name:
        return True

    # v23 Phase 12 follow-on: acyl-substituted amino prefixes (-oylamino) are
    # compound substituents (a substituted amino) and take enclosing marks per
    # P-16.3.3 — carbamoylamino (the urea FG prefix), ethanoylamino,
    # benzoylamino, sulfamoylamino, etc. The '-oyl' acyl ending distinguishes
    # them from simple alkylamino (methylamino ends in 'ylamino', not
    # 'oylamino'). Already-bracketed forms returned False above, so this only
    # promotes the unbracketed urea prefix to '(carbamoylamino)'.
    if name.lower().endswith('oylamino'):
        return True

    # Wave2 T2b: heteroatom-substituted amino/oxy preselected prefixes are
    # compound substituents and take enclosing marks (P-16.3.3) — BB PINs
    # write 4-(hydroxyamino)phenol, 2-(aminooxy)ethan-1-amine and P-35.3.1
    # 8-(chloroamino)octanoic acid. Simple preselected prefixes (diazenyl,
    # amino, isocyanato) stay bare.
    if name.lower() in _COMPOUND_HETEROATOM_AMINO_PREFIXES:
        return True

    # Functional group prefixes fused with alkyl names are compound substituents.
    # Examples: hydroxymethyl, carboxymethyl, aminoethyl, oxoethyl, formylmethyl
    # But NOT: methoxy, ethoxy (these are simple ether prefixes, single concept)
    # Uses module-level _COMPOUND_FG_PREFIXES and _ALKYL_ROOTS_FULL (C1-C20).
    name_lower = name.lower()
    for fg in _COMPOUND_FG_PREFIXES:
        if name_lower.startswith(fg):
            remainder = name_lower[len(fg):]
            # Check if the remainder is an alkyl root
            for alkyl in _ALKYL_ROOTS_FULL:
                if remainder == alkyl:
                    return True

    # Compound sulfur/selenium/tellurium prefixes: alkyl + sulfinyl/sulfonyl/sulfanyl
    # Per IUPAC P-16.3.3, "methylsulfinyl" = methyl + sulfinyl = compound substituent
    # requiring parentheses: "2-(methylsulfinyl)ethanoic acid"
    # But NOT bare "sulfinyl", "sulfonyl", "sulfanyl" (simple, no alkyl prefix)
    # Wave2 T3b: the PIN acid-stem forms build on the PARENT HYDRIDE
    # ('methanesulfinyl', 'benzenesulfonyl', 'cyclohexanesulfinyl' — BB
    # '2-(methanesulfonyl)ethan-1-ol' verbatim), so hydride stems are roots too.
    for alkyl in _ALKYL_ROOTS_FULL + _HYDRIDE_STEM_ROOTS:
        if name_lower.startswith(alkyl):
            remainder = name_lower[len(alkyl):]
            for s_suffix in _COMPOUND_S_SUFFIXES_COMPLEX:
                if remainder == s_suffix:
                    return True

    # W2F-P2 (P-16.3.3): ACYL-substituted chalcogen prefixes — an acyl group on
    # a sulfanyl/selanyl/tellanyl (acetylsulfanyl, formylsulfanyl,
    # propanoylsulfanyl, benzoylsulfanyl, ...) — are compound substituents and
    # take enclosing marks: '9-(acetylsulfanyl)-9-oxononanoic acid' (P-35.5.1
    # BB 18128). The acyl stem is retained 'acetyl'/'formyl' or ends in '-oyl'.
    for s_suffix in _COMPOUND_S_SUFFIXES_COMPLEX:
        if name_lower.endswith(s_suffix):
            acyl = name_lower[: -len(s_suffix)]
            if acyl and (acyl.endswith("oyl") or acyl in ("acetyl", "formyl")):
                return True

    # W2F-P2 (P-16.3.3): benzyl-based oxy/chalcogen prefixes (benzyloxy,
    # benzylsulfanyl, ...) are compound substituents (benzyl = substituted
    # methyl) and take enclosing marks: '9-(benzyloxy)-9-oxononanoic acid'
    # (P-35.4.2 / P-35.3.2:18097). Bare 'benzyl' and 'benzyloxymethyl' (remainder
    # not a bare oxy/chalcogen suffix) are unaffected.
    if name_lower.startswith("benzyl"):
        rest = name_lower[len("benzyl"):]
        if rest == "oxy" or rest in _COMPOUND_S_SUFFIXES_COMPLEX:
            return True

    return False


# ============================================================================
# IUPAC Enclosing Marks (P-16.3.3)
# ============================================================================


def get_bracket_depth(name: str) -> int:
    """Determine the current bracket nesting depth of a name.

    Returns 0 if no brackets, 1 if contains (), 2 if contains [], etc.
    Used to determine what enclosing marks to use at the next level.

    Args:
        name: A substituent or compound name.

    Returns:
        Integer nesting depth (0-3).

    Examples:
        >>> get_bracket_depth("methyl")
        0
        >>> get_bracket_depth("2-methylpropyl")
        0
        >>> get_bracket_depth("(2-methylpropyl)")
        1
        >>> get_bracket_depth("2-[(1-methylethyl)]propyl")
        2
    """
    if '{' in name:
        return 3
    if '[' in name:
        return 2
    if '(' in name:
        return 1
    return 0


# Pre-compiled patterns for compute_nesting_depth (P-16.5.4.1)
# P-16.5.4.1.1: Indicated hydrogen -- (1H), (3H), (9aH) etc.
_INDICATED_H_RE = re.compile(r'\(\d+[a-z]?H\)')
# P-16.5.4.1.2: Fusion/spiro/ring-assembly/von Baeyer brackets -- [2,3-b],
# [4.5], [2.2.1], [1,1'-biphenyl] (ring-assembly enclosures carry primes)
_FUSION_BRACKET_RE = re.compile(r"\[[0-9a-z,.'\-]+\]")
# P-16.5.4.1.3: Stereo descriptors -- (R), (S), (E), (Z), (1R,2S), etc.
_STEREO_PAREN_RE = re.compile(r'\((?:\d+[a-z]?,)*[RSEZ](?:,\d+[a-z]?[RSEZ]?)*\)')


def compute_nesting_depth(name: str) -> int:
    """Compute effective bracket nesting depth per P-16.5.4.1 (Dec 2025).

    Analyzes a name string and returns the effective nesting depth by
    counting only nesting-relevant brackets, per the following subsections:

    - P-16.5.4.1.1: Ignore indicated hydrogen parentheses, e.g., (1H), (3H)
    - P-16.5.4.1.2: Ignore fusion/spiro/ring assembly/von Baeyer brackets
    - P-16.5.4.1.3: Count stereo descriptor and compound locant parentheses
    - P-16.5.4.1.4: Escalate if consecutive same-level marks would result
    - P-16.5.4.1.5: Isotopic labeling convention (not applicable -- not implemented)

    Args:
        name: The chemical name string to analyze.

    Returns:
        Effective nesting depth (0 = no relevant brackets, 1 = has relevant
        parentheses, 2 = has relevant square brackets, etc.).
    """
    # Start with the original name and strip out non-nesting brackets
    # by replacing them with placeholder text that contains no brackets.
    working = name

    # Remove indicated hydrogen parentheses (P-16.5.4.1.1)
    working = _INDICATED_H_RE.sub('__IH__', working)

    # Remove fusion/spiro/von Baeyer square brackets (P-16.5.4.1.2)
    working = _FUSION_BRACKET_RE.sub('__FB__', working)

    # Now count remaining bracket types to determine depth
    max_depth = 0
    if '(' in working:
        max_depth = 1
    if '[' in working:
        max_depth = 2
    if '{' in working:
        max_depth = 3

    return max_depth


def apply_enclosing_marks(name: str, depth: int = 0) -> str:
    """Apply IUPAC P-16.3.3 enclosing marks at the correct nesting depth.

    Nesting order: ( ) -> [ ] -> { } -> ( ) again
    Depth 0: parentheses
    Depth 1: square brackets (name already contains parentheses)
    Depth 2: braces (name already contains brackets)

    When depth=-1 (sentinel for auto-detect), calls compute_nesting_depth()
    on the input name to determine the effective starting depth from the
    name's existing brackets per P-16.5.4.1 (Dec 2025 errata). This makes
    all callers that use the default depth automatically benefit from
    subsection-aware nesting.

    Args:
        name: The compound substituent name (without outer brackets).
        depth: Nesting depth (0 = outermost, -1 = auto-detect from name).

    Returns:
        Name enclosed in the appropriate bracket type.

    Examples:
        >>> apply_enclosing_marks("2-methylpropyl", 0)
        '(2-methylpropyl)'
        >>> apply_enclosing_marks("2-methylpropyl", 1)
        '[2-methylpropyl]'
        >>> apply_enclosing_marks("2-methylpropyl", 2)
        '{2-methylpropyl}'
        >>> apply_enclosing_marks("(R)-butan-2-yl", -1)
        '[(R)-butan-2-yl]'
    """
    MARKS = [('(', ')'), ('[', ']'), ('{', '}')]

    if depth == -1:
        # Auto-detect: compute effective depth from name content
        depth = compute_nesting_depth(name)

    # P-16.5.4.1.4: Check for consecutive same-level marks.
    # If the name starts with the same type of bracket we'd add, escalate --
    # BUT only if the leading bracket is nesting-relevant (not indicated H,
    # not fusion/spiro brackets).
    open_mark, close_mark = MARKS[depth % 3]
    if name.startswith(open_mark):
        # Check if the leading bracket is non-nesting (indicated H or fusion)
        leading_is_nesting = True
        if open_mark == '(':
            # Check for indicated hydrogen: (1H), (3H), (9aH)
            if _INDICATED_H_RE.match(name):
                leading_is_nesting = False
            # Check for stereo: (R), (S), (E), (Z), (1R,2S) -- these ARE nesting
            elif _STEREO_PAREN_RE.match(name):
                leading_is_nesting = True
        elif open_mark == '[':
            # Check for fusion/spiro brackets: [2,3-b], [4.5]
            if _FUSION_BRACKET_RE.match(name):
                leading_is_nesting = False

        if leading_is_nesting:
            depth += 1
            open_mark, close_mark = MARKS[depth % 3]

    return f"{open_mark}{name}{close_mark}"


def is_complex_substituent(name: str) -> bool:
    """Determine if a substituent name is complex.

    A substituent is considered complex if its name contains digits,
    hyphens, is enclosed in parentheses, or contains embedded substituent
    multiplier prefixes (e.g., diphenyl, trimethyl within a compound name).
    Complex substituents require bis/tris/tetrakis multipliers instead of
    di/tri/tetra, and are enclosed in parentheses in the final name.

    Note: Modification prefixes like "tetrahydro-" or "dihydro-" are NOT
    multipliers and do NOT make a name complex.

    Args:
        name: The substituent name (e.g., 'methyl', '1-methylethyl').

    Returns:
        True if the substituent is complex, False otherwise.

    Examples:
        >>> is_complex_substituent("methyl")
        False
        >>> is_complex_substituent("ethyl")
        False
        >>> is_complex_substituent("1-methylethyl")
        True
        >>> is_complex_substituent("2-propyl")
        True
        >>> is_complex_substituent("diphenylphosphanyl")
        True
        >>> is_complex_substituent("tetrahydropyranyl")
        False
    """
    # Check for digits (indicates locants within the substituent name)
    if any(ch.isdigit() for ch in name):
        return True
    # Check for hyphens (indicates compound substituent) — EXCEPT a leading
    # italicized 'sec-'/'tert-' detachable prefix on an otherwise-simple retained
    # name (tert-butyl, sec-butyl). IUPAC P-16.3.4 treats these as SIMPLE for
    # multiplication (di-tert-butyl, NOT bis(tert-butyl)) and does not enclose them
    # in marks (N-tert-butyl, NOT N-(tert-butyl)). Phase 171 BBR-ASM: without this,
    # coupling the paren/bis decision to is_complex_substituent over-parenthesised
    # tert-butyl. The leading-digit case above still catches genuine compounds.
    _hyphen_probe = name.lower()
    for _retained_prefix in ("sec-", "tert-"):
        if _hyphen_probe.startswith(_retained_prefix):
            _hyphen_probe = _hyphen_probe[len(_retained_prefix):]
            break
    if "-" in _hyphen_probe:
        return True
    # Check for embedded multiplier + substituent name patterns (IUPAC P-14.5.2)
    if _MULT_SUBSTITUENT_RE.match(name):
        return True
    # Compound oxy-prefixes require brackets (OPSIN/IUPAC parenthesization)
    if name in _COMPOUND_OXY_PREFIXES:
        return True
    # Compound sulfur prefixes per IUPAC P-16.3.3
    name_lower = name.lower()
    for alkyl in _ALKYL_ROOTS_FULL:
        if name_lower.startswith(alkyl):
            remainder = name_lower[len(alkyl):]
            for s_suffix in _COMPOUND_S_SUFFIXES_COMPLEX:
                if remainder == s_suffix:
                    return True
    # DD2 Fix B (Phase D, P-63.3.1(1)): (alkyl)peroxy is a compound substituent
    # requiring enclosing marks — '(methylperoxy)ethane', '(ethylperoxy)benzene'.
    # The simple 'peroxy' bridge and 'hydroperoxy' prefix are NOT compound.
    if name_lower.endswith('peroxy') and name_lower not in ('peroxy', 'hydroperoxy'):
        return True
    # Acyloxy compound prefixes per IUPAC P-16.3.3:
    # "acetyloxy", "benzoyloxy", "propanoyloxy" etc. are compound prefixes
    # (acyl + oxy) that require complex multipliers (bis/tris) and parenthesization.
    if name.endswith('yloxy') and len(name) > 5:
        # Matches: acetyloxy, benzoyloxy, propanoyloxy, butanoyloxy, etc.
        # Does NOT match: methoxy, ethoxy (simple alkoxy, no 'yl' before 'oxy')
        return True
    # Haloalkyl compound substituents per IUPAC P-31.1.2.3:
    # "fluoromethyl", "trifluoromethyl", "chloroethyl" etc.
    if _HALOALKYL_RE.match(name):
        return True
    # Alkyl+amino compound substituents per IUPAC P-31.1.2:
    # "methylamino", "ethylamino", "phenylamino" etc.
    if _ALKYLAMINO_RE.match(name):
        return True
    # Acylamino compound substituents: "ethanoylamino", "propanoylamino" etc.
    if name.endswith('amino') and 'oyl' in name:
        return True
    # (R-oxy)alkyl compound substituents per IUPAC P-16.3.3 / P-63.2.2.2 (v22
    # C-T2 / V-3): an alkoxy/aryloxy unit ('<R>oxy') fused to a terminal alkyl-yl
    # stem is a compound prefix taking enclosing marks — '(phenoxymethyl)benzene',
    # '(methoxymethyl)benzene'. A bare terminal alkoxy ('methoxy', 'phenoxy')
    # ends in 'oxy' and stays simple; 'hydroxy'/'carboxy' heads are NOT ethers.
    for _oxy_stem in ('oxymethyl', 'oxyethyl', 'oxypropyl', 'oxybutyl',
                      'oxypentyl', 'oxyhexyl'):
        if name_lower.endswith(_oxy_stem):
            _head = name_lower[:-len(_oxy_stem)]
            if _head and not _head.endswith(('hydr', 'carb')):
                return True
            break
    # Phase 4 SUBST-01: compound (ring/substituent)-yl + alkyl-yl
    # ('cyclohexylmethyl', 'piperidinylmethyl') — enclosing marks per P-16.3.3.
    if _RINGYL_ALKYL_RE.search(name_lower):
        return True
    # v23 Phase 8 (P-16.3.3): a Group-14 silyl/germyl substituent carrying its OWN
    # prefixes ('trihydroxysilyl', 'hydroxydimethylsilyl', 'aminodimethylsilyl') is
    # a compound prefix taking enclosing marks. The bare 'silyl'/'germyl' stems
    # stay simple; 'trimethylsilyl'/'triethylsilyl' are already caught by the
    # alkyl-multiplier regex above. (These prefixed silyl names are all new in v23
    # Phase 8, so this rule adds no regression to pre-existing substituents.)
    if (name_lower.endswith(('silyl', 'germyl'))
            and name_lower not in ('silyl', 'germyl')):
        return True
    return False


# P-35.1 (BlueBookV2.md:17954): simple prefixes whose 'di'/'tri' concatenation
# collides with a P-29.3.1 catenated-hydride prefix (disulfanyl = -SSH, NOT
# two -SH). These take the derived multipliers bis/tris/... even though the
# name itself is simple; P-16.5.1.10 then parenthesizes the multiplied term.
CATENATION_AMBIGUOUS_PREFIXES = frozenset({
    "sulfanyl",   # disulfanyl -SSH          (P-35.1 verbatim example)
    "selanyl",    # diselanyl -SeSeH         (P-35.1: 'diselanyl, -SeSeH')
    "tellanyl",   # ditellanyl -TeTeH
    "phosphanyl", # diphosphanyl (P-45.3.1 examples)
    "arsanyl",    # diarsanyl (diarsane P-21.1.2)
    "stibanyl",   # distibanyl (distibane P-21.1.2)
    "azanyl",     # diazanyl = hydrazinyl -NH-NH2
    "oxidanyl",   # dioxidanyl -OOH
})


def get_multiplier_prefix(count: int, substituent_name: str) -> str:
    """Get the appropriate multiplier prefix for a count of substituents.

    For count=1, returns empty string (no multiplier needed).
    For simple substituent names (no digits/hyphens), uses di/tri/tetra.
    For complex substituent names, uses bis/tris/tetrakis.

    Args:
        count: Number of identical substituents.
        substituent_name: The substituent name to determine simple vs complex.

    Returns:
        Multiplier prefix string, or empty string for count=1.

    Examples:
        >>> get_multiplier_prefix(1, "methyl")
        ''
        >>> get_multiplier_prefix(2, "methyl")
        'di'
        >>> get_multiplier_prefix(3, "methyl")
        'tri'
        >>> get_multiplier_prefix(2, "1-methylethyl")
        'bis'
        >>> get_multiplier_prefix(3, "1-methylethyl")
        'tris'
    """
    if count <= 1:
        return ""

    if is_complex_substituent(substituent_name) \
            or substituent_name in CATENATION_AMBIGUOUS_PREFIXES:
        if count in COMPLEX_MULTIPLIERS:
            return COMPLEX_MULTIPLIERS[count]
        # For counts > 20, build compositional multiplier using chain_names
        from ..data.chain_names import get_chain_prefix
        prefix = get_chain_prefix(count)
        # Complex multipliers use "-akis" suffix (e.g., "henicosakis")
        if not prefix.endswith("a"):
            prefix += "a"
        return prefix + "kis"
    else:
        if count in SIMPLE_MULTIPLIERS:
            return SIMPLE_MULTIPLIERS[count]
        # For counts > 20, build compositional multiplier using chain_names
        from ..data.chain_names import get_chain_prefix
        prefix = get_chain_prefix(count)
        # Simple multipliers use trailing 'a' (e.g., "henicosa", "docosa")
        if not prefix.endswith("a"):
            prefix += "a"
        return prefix


# ============================================================================
# Substituent Prefix Formatting
# ============================================================================


def _has_stereo_prefix(name: str) -> bool:
    """Check if a substituent name starts with a CIP stereo descriptor prefix.

    Detects patterns like "(R)-", "(S)-", "(1R)-", "(2S,3R)-", "(11z,14z)-"
    at the start. This distinguishes stereo-prefixed names from compound
    substituent names that happen to start with parentheses (e.g.,
    "(2-methylphenyl)").

    A stereo prefix is: '(' + optional digits/comma + single letter R/S/E/Z + ')' + '-'
    Case-insensitive for E/Z since generated names may use lowercase e/z.
    """
    if not name.startswith('('):
        return False
    # Match: (R)-, (S)-, (1R)-, (2S,3R)-, (11z,14z)- etc.
    return bool(re.match(r'^\((?:\d+[RSEZrsez],)*\d*[RSEZrsez]\)-', name))


def _wrap_n_substituent(name: str) -> str:
    """Apply IUPAC P-16.3.3 bracket escalation to an N-substituent name.

    When an N-substituent already contains parentheses (from stereo
    descriptors or compound substituent names), the outer enclosure must
    use square brackets to maintain unambiguous nesting.

    Simple names (no parentheses) are returned unchanged -- they do not
    need brackets around them in N-prefix context.

    Names that already have balanced outer enclosing marks -- starting
    with '(' or '[' and ending with the matching close, where the opening
    mark at position 0 is balanced at the final position -- are returned
    unchanged (already properly enclosed).

    Args:
        name: The bare N-substituent name (without N- prefix).

    Returns:
        The name, optionally wrapped in square brackets.

    Examples:
        >>> _wrap_n_substituent("methyl")
        'methyl'
        >>> _wrap_n_substituent("dimethyl")
        'dimethyl'
        >>> _wrap_n_substituent("(2S)-2-(pentanoylamino)propanoyl")
        '[(2S)-2-(pentanoylamino)propanoyl]'
        >>> _wrap_n_substituent("(2R,3S)-3-hydroxy-2-(benzoylamino)butanoyl")
        '[(2R,3S)-3-hydroxy-2-(benzoylamino)butanoyl]'
        >>> _wrap_n_substituent("[already-bracketed]")
        '[already-bracketed]'
        >>> _wrap_n_substituent("(3-ethyl-1H-indolyl)")
        '(3-ethyl-1H-indolyl)'
    """
    # Already has outer square brackets -- no double-wrapping
    if name.startswith('[') and name.endswith(']'):
        return name
    # Already has balanced outer parentheses as enclosing marks --
    # this means a previous step already enclosed the name. Check that
    # the opening paren at position 0 is balanced at the final position.
    if name.startswith('(') and name.endswith(')'):
        depth = 0
        for i, ch in enumerate(name):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth == 0 and i == len(name) - 1:
                # The opening paren matches the closing paren at the end
                return name
    # Contains parentheses (stereo prefix, compound sub-substituent, etc.)
    # -> escalate to square brackets per P-16.3.3
    if '(' in name:
        return f"[{name}]"
    return name


def _is_fully_enclosed(name: str) -> bool:
    """True when ``name`` is a single fully-bracketed token: it opens with a
    bracket whose MATCHING close is the final character (e.g. '(2-methylpropyl)',
    '[bis(sulfanyl)]'). False when trailing text lies outside the leading
    bracket ('(oxan-2-yl)oxy', '(benzylsulfanyl)methyl') — those still need an
    OUTER enclosing mark. Used by format_substituent_prefix's P-16.5 escalation.
    """
    if not name or name[0] not in '([{':
        return False
    pairs = {'(': ')', '[': ']', '{': '}'}
    close = pairs[name[0]]
    depth = 0
    for i, ch in enumerate(name):
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
            if depth == 0:
                # Matched the opening bracket; fully enclosed only if this is
                # the last character AND the close matches the open type.
                return i == len(name) - 1 and ch == close
    return False


def format_substituent_prefix(name: str, locants: List[int], count: int) -> str:
    """Format a substituent with locants and multiplier prefix.

    BBR-HYG(e) / DEF-8 inventory (Phase 169.7 — begin; consolidation lands in
    BBR-ASM / Phase 171). This is the ONE correct substituent-prefix + needs-parens
    reference (P-16.3.5 / P-16.3.3). Phase 171 consolidates the divergent
    needs-parens / enclosing-mark / prefix-assembly predicates onto THIS function;
    do NOT add a 4th. The divergent predicates as of 169.7:
      - rules/amides.py:32           _has_positional_locants(name)   (digit-only locant test)
      - assembly/naming_utils.py:525 is_complex_substituent(name)    (complex-substituent test)
      - assembly/naming_utils.py:461 apply_enclosing_marks(name, depth)
      - rules/ortho_fused.py:505     format_substituent_prefix(substituents)  (different signature)
    They DISAGREE today (e.g. on 'trifluoromethyl'/'tert-butyl': is_complex_substituent
    True but _has_positional_locants False) — see
    tests/unit/assembly/test_needs_parens_consolidation.py (xfail-strict tripwire).

    Produces a formatted substituent prefix ready for insertion into an
    IUPAC name. Handles simple and complex substituents differently:
    - Simple: locants + multiplier + name (e.g., '2,2-dimethyl')
    - Complex: locants + multiplier + (name) (e.g., '3,5-bis(1-methylethyl)')

    Args:
        name: Base substituent name (e.g., 'methyl', '1-methylethyl').
        locants: List of locant positions for this substituent.
        count: Number of identical substituents.

    Returns:
        Formatted prefix string.

    Examples:
        >>> format_substituent_prefix("methyl", [2], 1)
        '2-methyl'
        >>> format_substituent_prefix("methyl", [2, 2], 2)
        '2,2-dimethyl'
        >>> format_substituent_prefix("ethyl", [3, 5], 2)
        '3,5-diethyl'
        >>> format_substituent_prefix("1-methylethyl", [4, 7], 2)
        '4,7-bis(1-methylethyl)'
    """
    # Format locants as comma-separated string
    locant_str = ",".join(str(loc) for loc in locants)

    # Get the multiplier prefix
    multiplier = get_multiplier_prefix(count, name)

    # P-16.5.1.10: any term modified by a DERIVED multiplier (bis/tris/
    # tetrakis/...kis) is enclosed in parentheses, even when the base name
    # is simple (bis(sulfanyl), P-35.1).
    derived_multiplier = bool(multiplier) and (
        multiplier in COMPLEX_MULTIPLIERS.values() or multiplier.endswith("kis")
    )

    # Complex substituents get parentheses around the name.
    # Per IUPAC P-14.5.2, compound substituent names containing numeric locants
    # need enclosing marks to avoid ambiguity:
    # - count > 1: always wrap for bis/tris multiplier (e.g., "3,5-bis(2-methylpropyl)")
    # - count == 1: wrap if name has bare digits at the start (e.g., "3-(2-methylpropyl)")
    #   but NOT for names that already have their own parenthesization (e.g., "(oxan-2-yl)oxy")
    #   and NOT for names that are just hyphenated (e.g., "3-sec-butyl" is fine)
    complex = is_complex_substituent(name)
    if _has_stereo_prefix(name):
        # Name has a CIP stereo descriptor prefix (e.g., "(R)-sec-butyl"):
        # use square brackets per IUPAC P-16.3.3 nesting rules
        formatted_name = f"[{name}]"
    elif ('(' in name or '[' in name) and not _is_fully_enclosed(name):
        # P-16.5.4.1 (W2E-P1FC Task 8, generalized by w2f p1 per
        # P-16.5.2.4/P-16.5.3.1): the name CARRIES an enclosing mark
        # anywhere — leading '(benzylsulfanyl)methyl', interior
        # 'bromo(phenyl)methyl' (P-29.6.2.1), or trailing-stem
        # '[(4-methoxyphenyl)methoxy]methyl' — but is not itself fully
        # wrapped. Such a name is a compound substituent by construction
        # (regardless of is_complex_substituent, which keys on digits/
        # hyphens and misses these shapes). Citing it BARE drops structure
        # (OPSIN parses 'bromo(phenyl)methylbenzene' as bromo + benzyl,
        # TWO substituents) and plain parens double the mark
        # ('((...)methyl)', non-PIN). Escalate to the next bracket level.
        # Fusion/von-Baeyer/ring-assembly brackets remain nesting-IGNORED
        # inside apply_enclosing_marks (P-16.5.4.1.2), so
        # 'furo[3,2-b]pyridin-2-yl' still takes plain parentheses.
        formatted_name = apply_enclosing_marks(name, -1)
    elif (complex or derived_multiplier) and not name.startswith('(') \
            and not name.startswith('['):
        # Complex name (or a simple name taking a derived bis/tris/...kis
        # multiplier, P-16.5.1.10) without existing enclosing marks: wrap.
        formatted_name = f"({name})"
    else:
        formatted_name = name

    # Wave2 T5a: a simple multiplier joining an italicized-prefix-led name
    # keeps the hyphen boundary: '1,2-di-tert-butylbenzene' (PIN, BB
    # P-25.7.1.x example list), never 'ditert-butyl'.  Parenthesized names
    # never start with 'tert-'/'sec-', so the bis/tris path is unaffected.
    if multiplier and formatted_name.startswith(("tert-", "sec-")):
        multiplier = f"{multiplier}-"

    # Assemble: locants-multiplier+name. An empty locant list (elided per
    # P-14.3.4, e.g. a mononuclear parent: phenylmethanol) takes no hyphen.
    if not locant_str:
        return f"{multiplier}{formatted_name}"
    return f"{locant_str}-{multiplier}{formatted_name}"


# ============================================================================
# Alphabetization Sort Key
# ============================================================================

# Prefixes to IGNORE for IUPAC alphabetization
# These are multiplicative prefixes and sec-/tert- detachable prefixes
IGNORE_FOR_ALPHA = {
    "di", "tri", "tetra", "penta", "hexa", "hepta", "octa", "nona", "deca",
    "undeca", "dodeca",
    "bis", "tris", "tetrakis", "pentakis", "hexakis", "heptakis",
    "octakis", "nonakis", "decakis",
    "sec", "tert",
}

# Note: iso-, neo-, cyclo- are INCLUDED in alphabetization
# (they are non-detachable prefixes that affect alphabetical order)

# Pre-sorted prefix list for alpha_sort_key() — longest first to avoid partial matches
_SORTED_ALPHA_PREFIXES = sorted(IGNORE_FOR_ALPHA - {"sec", "tert"},
                                key=len, reverse=True)


def alpha_sort_key(substituent_name: str) -> str:
    """Generate an alphabetization sort key for IUPAC prefix ordering.

    According to IUPAC 2013 rules:
    - Multiplicative prefixes (di-, tri-, tetra-, bis-, tris-, tetrakis-)
      are IGNORED for alphabetization.
    - sec- and tert- are IGNORED for alphabetization (treated as detachable).
    - iso-, neo-, cyclo- are INCLUDED (non-detachable, affect sort order).

    The input is the base substituent name. The function strips any
    leading multiplicative prefix or sec-/tert- to produce a sort key.

    Args:
        substituent_name: The substituent name (e.g., 'methyl', 'dimethyl',
                         'isopropyl', 'tert-butyl').

    Returns:
        Lowercase sort key string.

    Examples:
        >>> alpha_sort_key("methyl")
        'methyl'
        >>> alpha_sort_key("dimethyl")
        'methyl'
        >>> alpha_sort_key("triethyl")
        'ethyl'
        >>> alpha_sort_key("isopropyl")
        'isopropyl'
        >>> alpha_sort_key("neopentyl")
        'neopentyl'
        >>> alpha_sort_key("cyclopropyl")
        'cyclopropyl'
        >>> alpha_sort_key("tert-butyl")
        'butyl'
    """
    text = substituent_name.lower()

    # Leading positional descriptors are ALWAYS ignored for alphabetization
    # (P-14.5.2): a parent locant ('3-'), a compound locant set ('2,4-'), and
    # the italic indicated-hydrogen descriptor ('1H-'). Strip them up front so
    # the enclosed-vs-simple decision below sees the bare substituent name —
    # this is what makes '3-(1H-imidazol-5-yl)' and '(1H-imidazol-5-yl)' yield
    # the same key, and '2-(2,4-dihydroxyphenyl)' route to the complete-name
    # branch just like the bare '(2,4-dihydroxyphenyl)'.
    text = _LOCANT_PREFIX_RE.sub('', text)
    text = _INDICATED_H_PREFIX_RE.sub('', text)

    # P-14.5.2 (Phase 171 BBR-ASM, DEF-8): a COMPOUND substituent cited as a
    # fully-enclosed unit is alphabetized on the first letter of its COMPLETE
    # name — its INTERNAL multiplying prefix (di/tri…) is part of the name and
    # is NOT ignored ('(2,4-dimethylpentyl)' sorts at 'd', before 'ethyl';
    # '2-(2,4-dihydroxyphenyl)' sorts at 'd', before 'hydroxy'). Contrast
    # P-14.5.1: a bare 'di'/'tri' multiplying SEPARATE simple prefixes on the
    # parent IS ignored ('2,2-dimethyl' sorts at 'm'). The enclosing marks
    # disambiguate the two rules, so only fully-enclosed input takes the
    # complete-name path.
    if (text.startswith('(') and text.endswith(')')) or (text.startswith('[') and text.endswith(']')):
        inner = text[1:-1]
        inner = _LOCANT_PREFIX_RE.sub('', inner)       # drop the inner positional locant
        inner = _INDICATED_H_PREFIX_RE.sub('', inner)  # and a leading '1H-' descriptor
        if (inner.startswith('(') and inner.endswith(')')) or (inner.startswith('[') and inner.endswith(']')):
            inner = inner[1:-1]
        return inner  # complete name; internal multiplying prefix NOT stripped

    # ---- Simple (non-enclosed) prefix: P-14.5.1 ----
    # Strip N-locant prefixes (N- or N,N-)
    # e.g., "N,N-dimethylamino" -> "dimethylamino" -> "amino" (after multi-prefix strip)
    text = _N_LOCANT_PREFIX_RE.sub('', text)

    # Handle hyphenated detachable prefixes: sec- and tert-
    for prefix in ("sec-", "tert-"):
        if text.startswith(prefix):
            return text[len(prefix):]

    # Handle non-hyphenated multiplicative prefixes (di-, tri-, ...): for a
    # SIMPLE prefix these are ignored (P-14.5.1). Sort by longest prefix first
    # to avoid partial matches (e.g., 'tetra' before 'tri'). The result is a
    # sort key only, never reconstructed.
    for prefix in _SORTED_ALPHA_PREFIXES:
        if text.startswith(prefix):
            remainder = text[len(prefix):]
            # Only strip if there is a remainder (avoid stripping entire word)
            if remainder:
                # Wave2 T1c: do NOT strip when the "multiplier" is really the
                # start of a numeric CHAIN STEM — there it is part of the name
                # and alphabetizes (P-14.5.2): tridecyl/octadecyl ('tri'/'octa'
                # + 'dec...'), pentacosyl ('cos...'), triacontyl ('acont...'),
                # pentanamido/octadecanamido ('penta'/'octa' + 'nam...'),
                # decanoyloxy ('deca' + 'noyl...'). A genuine multiplied
                # prefix never continues with these stems.
                if remainder.startswith(
                        ('dec', 'cos', 'cont', 'acont', 'nam', 'noyl')):
                    return text
                # Wave2 T2b: 'diazenyl' (HN=N-, P-35.2.2) and 'diazo'
                # (P-61.5) — the leading 'di' is structural (the two
                # nitrogens of diazene/diazo), never a multiplier; the BB
                # multiplies them with bis() ('bis(diazenyl)', not
                # 'didiazenyl'). Exact-remainder match so 'diazido'
                # (a genuine 2x azido) still strips to 'azido'.
                if remainder in ('azenyl', 'azo'):
                    return text
                return remainder

    return text


def prefix_citation_sort_key(prefix: str) -> tuple:
    """P-14.5.2 alpha key + P-14.5.4 lowest-locant tie-break for citation.

    BB P-14.5.4 (BlueBookV2.md:3517): 'When two or more prefixes consist of
    identical Roman letters, priority for order of citation is given to the
    group that contains the lowest locant(s) at the first point of
    difference.' The prefix's LEADING parent-locant set is stripped (those
    locants are assigned BY citation order, they may not decide it); the
    remaining locant tokens are compared in order of appearance via
    locant_sort_key. Drop-in replacement key for alpha_sort_key at
    citation-sort sites: identical alpha keys now resolve by locants
    instead of stable-sort input order.
    """
    from .name_comparison import locant_sort_key, _LOCANT_TOKEN_FINDER
    # Tier 1 is the Roman LETTERS of the complete-name alpha key (P-14.5.2):
    # digits/hyphens are dropped so identical-letter prefixes (pentan-2-yl vs
    # pentan-3-yl) collide here and the locant tier below decides, per
    # P-14.5.4. (alpha_sort_key already strips leading positional locants and
    # multiplicative prefixes; here we additionally strip the *internal*
    # locant digits so the tier is letters-only.)
    alpha_letters = ''.join(ch for ch in alpha_sort_key(prefix) if ch.isalpha())
    core = _LOCANT_PREFIX_RE.sub('', prefix)
    locs = tuple(locant_sort_key(t)
                 for t in _LOCANT_TOKEN_FINDER.findall(core))
    return (alpha_letters, locs)


# ============================================================================
# Vowel Elision
# ============================================================================

# Vowels that trigger elision of terminal 'e' in the parent stem
_ELISION_VOWELS = frozenset("aiouy")
# Note: 'e' is excluded from triggering elision per IUPAC convention;
# terminal 'e' is only elided before a, i, o, u, y


def apply_vowel_elision(parent_stem: str, suffix: str) -> str:
    """Apply IUPAC vowel elision rules when joining parent stem and suffix.

    The terminal 'e' in a parent stem is elided (removed) when the suffix
    begins with 'a', 'i', 'o', 'u', or 'y'. The 'e' is NOT elided before
    consonants or before another 'e'.

    Args:
        parent_stem: Parent name stem (e.g., 'propane', 'butane', 'propan').
        suffix: Suffix to append (e.g., 'ol', 'al', 'one', 'amine', 'diol').

    Returns:
        Combined string with elision applied if appropriate.

    Examples:
        >>> apply_vowel_elision("propane", "ol")
        'propanol'
        >>> apply_vowel_elision("propane", "al")
        'propanal'
        >>> apply_vowel_elision("propane", "one")
        'propanone'
        >>> apply_vowel_elision("propane", "amine")
        'propanamine'
        >>> apply_vowel_elision("butane", "diol")
        'butanediol'
        >>> apply_vowel_elision("ethane", "oic acid")
        'ethanoic acid'
        >>> apply_vowel_elision("propan", "ol")
        'propanol'
    """
    if not parent_stem or not suffix:
        return parent_stem + suffix

    # Check if parent stem ends in 'e' and suffix starts with a vowel
    # that triggers elision
    if parent_stem[-1] == "e" and suffix[0] in _ELISION_VOWELS:
        return parent_stem[:-1] + suffix

    return parent_stem + suffix


def _join_multiplied_suffix(suffix_multiplier: str, suffix: str) -> str:
    """Join a numerical multiplier to a characteristic-group suffix with the
    IUPAC P-63.1.2 / P-16.3.3 vowel elision applied.

    The final letter 'a' of a multiplying prefix ('tetra', 'penta', 'hexa', …)
    is elided before the suffix '-ol' (tetra+ol → tetrol, hexa+ol → hexol;
    Blue Book P-63.1.2: ``benzenehexol``, ``cyclohexane-1,2,3,4-tetrol``) and
    before the suffix '-amine' (tetra+amine → tetramine; **Blue Book P-62.2.4.1.2
    states this verbatim**: "The terminal letter 'a' of a numerical prefix is
    elided before the suffix amine, i.e., 'tetramine', not 'tetraamine'",
    e.g. ``[1,1'-biphenyl]-3,3',4,4'-tetramine``, ``silanetetramine``).

    Also elided before the ketone suffix '-one' (tetra+one → tetrone, penta+one →
    pentone): **Blue Book P-64.2.2.1(1) states this verbatim** — "the final letter
    'a' of a numerical multiplying prefix is elided before the suffix '-one', for
    example, 'tetrone'", with PINs ``pentacosane-7,9,17,19-tetrone`` (BB 28363) and
    ``pyrene-1,3,6,8(2H,7H)-tetrone`` (BB 28932). (Both '-tetraone' and '-tetrone'
    happen to round-trip through OPSIN, so RT alone could not decide the PIN — the
    Blue Book does.)

    The elision fires only before a VOWEL-initial suffix, so it is scoped to the
    explicit set {'ol', 'amine', 'one'}: the chalcogen ketone suffixes '-thione'/
    '-selone'/'-tellone' begin with a consonant and take NO elision (tetrathione),
    and 'di'/'tri' carry no terminal 'a', so 'diol'/'triol'/'dione'/'trione' are
    unaffected.
    """
    if suffix in ("ol", "amine", "one") and suffix_multiplier.endswith("a"):
        return suffix_multiplier[:-1] + suffix
    return suffix_multiplier + suffix


# ============================================================================
# Suffix with Locants Formatting (PIN Infix Style)
# ============================================================================

def format_suffix_with_locants(
    parent_stem: str,
    unsaturation: str,
    suffix: str,
    suffix_locants: List[int],
    suffix_multiplier: str = "",
) -> str:
    """Assemble a parent name with infix locants in IUPAC 2013 PIN style.

    Constructs the parent name by combining stem + unsaturation infix,
    then attaching the suffix with locants using hyphens. Applies vowel
    elision rules where appropriate.

    Args:
        parent_stem: The chain prefix (e.g., 'prop', 'but', 'pent').
        unsaturation: Unsaturation infix (e.g., 'an', 'en', 'yn', '').
        suffix: The functional group suffix (e.g., 'ol', 'one', 'oic acid', 'al').
        suffix_locants: Locant positions for the suffix group(s).
        suffix_multiplier: Multiplier for multiple suffix groups (e.g., 'di', 'tri').

    Returns:
        Assembled parent name string.

    Examples:
        >>> format_suffix_with_locants("prop", "an", "ol", [1])
        'propan-1-ol'
        >>> format_suffix_with_locants("but", "an", "one", [2])
        'butan-2-one'
        >>> format_suffix_with_locants("prop", "an", "ol", [1, 2], "di")
        'propane-1,2-diol'
        >>> format_suffix_with_locants("pent", "an", "oic acid", [])
        'pentanoic acid'
        >>> format_suffix_with_locants("prop", "an", "al", [])
        'propanal'
    """
    # Build the base: stem + unsaturation (e.g., 'propan', 'buten')
    base = parent_stem + unsaturation

    if suffix_locants:
        # Format locants as comma-separated
        locant_str = ",".join(str(loc) for loc in suffix_locants)

        # Build the suffix part: multiplier + suffix (e.g., 'diol', 'tetrol').
        # P-63.1.2 elides multiplier-final 'a' before '-ol' (tetra+ol → tetrol).
        full_suffix = _join_multiplied_suffix(suffix_multiplier, suffix)

        # When there's a multiplier (e.g., 'di'), the base keeps terminal 'e'
        # because 'diol' starts with 'd' (consonant), so no elision.
        # When there's no multiplier and suffix starts with vowel, apply elision.
        if suffix_multiplier:
            # With multiplier: base + 'e' + '-locants-' + multiplier + suffix
            # e.g., 'propane-1,2-diol'
            # The 'e' is added because the next character is '-' (or the
            # multiplier starts with a consonant like 'd')
            base_with_e = apply_vowel_elision(base + "e", full_suffix)
            # But we need to insert locants between base and suffix
            # So: base+'e' + '-locants-' + full_suffix
            return f"{base}e-{locant_str}-{full_suffix}"
        else:
            # Without multiplier: base + '-locants-' + suffix
            # IUPAC P-16.3.3 vowel elision: drop terminal 'e' of the parent stem
            # only when the suffix begins with a, i, o, u, or y. Before consonant-
            # leading suffixes (-thione, -selone, -tellone, etc.) preserve the 'e'.
            #
            # Examples:
            #   propan + 1 + ol     -> 'propan-1-ol'     (elide e — 'o' is vowel)
            #   butan  + 2 + one    -> 'butan-2-one'     (elide e — 'o' is vowel)
            #   propan + 2 + thione -> 'propane-2-thione' (keep e — 't' is consonant)
            #   propan + 2 + selone -> 'propane-2-selone' (keep e — 's' is consonant)
            #   ethan  + 1 + thione -> 'ethane-1-thione' (P-66.1.4.3 example PIN)
            if suffix and suffix[0] in _ELISION_VOWELS:
                return f"{base}-{locant_str}-{suffix}"
            else:
                return f"{base}e-{locant_str}-{suffix}"
    else:
        # No locants: combine base and suffix with elision
        # e.g., 'pentanoic acid', 'propanal'
        if suffix_multiplier:
            # Multiple terminal groups without locants (e.g., diacids):
            # 'butanedioic acid', 'pentanedioic acid'
            full_suffix = _join_multiplied_suffix(suffix_multiplier, suffix)
            return apply_vowel_elision(base + "e", full_suffix)
        # DD2 (Phase D, P-56.2): a suffix led by an italic chalcogen-pair
        # descriptor ('SO-thioperoxol' / 'OS-thioperoxol') needs a separating
        # hyphen even with no numeric locant — 'methane-SO-thioperoxol', not
        # 'methaneSO-thioperoxol'. Matched explicitly (not a bare isupper() test)
        # so a future uppercase-led suffix cannot silently inherit this path.
        if suffix.startswith(("SO-", "OS-")):
            return f"{base}e-{suffix}"
        return apply_vowel_elision(base + "e", suffix)
