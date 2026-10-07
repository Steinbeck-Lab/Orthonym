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
from typing import List, Optional, Tuple

# ============================================================================
# Pre-compiled patterns and constants (hoisted from function bodies)
# ============================================================================

# Used by is_complex_substituent — compound multiplier+substituent patterns
_MULT_SUBSTITUENT_RE = re.compile(
    r'^(?:di|tri|tetra|penta|hexa)'
    r'(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|'
    r'phenyl|naphthyl|cyclopentyl|cyclohexyl|benzyl|vinyl|allyl|'
    r'fluoro|chloro|bromo|iodo)'
)

# Every numerical multiplying-prefix syllable uses. Kept adjacent to the
# regexes that need it because those compile at import time, ahead of
# ``SIMPLE_MULTIPLIERS``; ``tests/unit/assembly/test_naming_utils.py`` asserts the two
# stay in step, so this is a forward declaration and NOT a second source of truth.
# Longest-first so the alternation reads unambiguously (``trideca`` before ``tri``).
_MULTIPLIER_SYLLABLES = (
    'pentadeca', 'tetradeca', 'heptadeca', 'hexadeca', 'octadeca', 'nonadeca',
    'trideca', 'undeca', 'dodeca', 'tetra', 'penta', 'hexa', 'hepta', 'octa',
    'nona', 'deca', 'icosa', 'di', 'tri',
)
_MULT_ALT = '|'.join(_MULTIPLIER_SYLLABLES)

# Halogen + alkyl compound substituent patterns (e.g., fluoromethyl,
# trifluoromethyl, chloroethyl). These are compound substituents per
# IUPAC and require enclosing marks.
#
# ⚠ Phase C Task 5a: the multiplier alternation used to stop at ``hexa``, which
# was invisible for as long as every fully-halogenated alkyl carried locants --
# ``is_complex_substituent`` returns True on the FIRST digit it sees, so this regex
# was never the load-bearing gate. (``:3007``) removes those digits, and
# the truncation surfaced immediately as a LOST enclosing mark:
# ``(1,1,2,2,3,3,3-heptafluoropropyl)benzene`` became ``heptafluoropropylbenzene``
# (hepta is not in the old list), while ``(pentafluoroethyl)benzene`` was fine.
# ``:3023`` prints ``1-chloro-2-(pentafluoroethyl)benzene (PIN)`` WITH the marks, and
# a bare ``heptafluoropropylbenzene`` also reads as ``heptafluoro`` + ``propylbenzene``.
# The multiplier set is now complete, because the substituted-alkyl class is OPEN.
#
# ⚠ RELATIONSHIP TO ``_MULT_SUBSTITUENT_RE`` (raised by review, 2026-07-29). The two
# patterns OVERLAP on ``di``..``hexa`` + halogen + alkyl -- ``difluoromethyl`` matches
# both, and ``_MULT_SUBSTITUENT_RE`` is tested first (``is_complex_substituent``).
# The overlap is deliberate REDUNDANCY, not a division of labour, and neither is a
# subset of the other: ``_MULT_SUBSTITUENT_RE`` is unanchored and also covers
# non-halogen stems (``triphenyl``), while this pattern is ``$``-anchored and carries
# the full multiplier set (``hepta`` upward). So a future narrowing of either
# one does not silently open a hole in the halogen case -- but do not *rely* on that:
# if you touch either, re-check the hepta+ case, which is the one that was broken and
# which no test exercised before Task 5a because the names still carried digits.
_HALOALKYL_RE = re.compile(
    r'^(?:(?:' + _MULT_ALT + r')?(?:fluoro|chloro|bromo|iodo))'
    r'(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|nonyl|decyl|'
    r'undecyl|dodecyl|tridecyl|tetradecyl|pentadecyl|'
    r'hexadecyl|heptadecyl|octadecyl|nonadecyl|icosyl)$'
)

# Halogen-substituted simple alkoxy compound prefix): a halogen prefix
# fused onto a bare alkoxy stem -> takes enclosing marks, '(chloromethoxy)'. Bare
# 'methoxy'/'ethoxy' are simple and NOT matched. BB '1-(chloromethoxy)-4-nitrobenzene'.
_HALO_ALKOXY_RE = re.compile(
    r'^(?:(?:' + _MULT_ALT + r')?(?:fluoro|chloro|bromo|iodo))'
    r'(?:meth|eth|prop|but|pent|hex|hept|oct|non|dec|'
    r'undec|dodec|tridec|tetradec|pentadec|hexadec|heptadec|octadec|nonadec|icos)'
    r'oxy$'
)

# (the Blue Book): the retained contracted R-O- prefixes 'methoxy',
# 'ethoxy', 'propoxy', 'butoxy' and 'phenoxy' "are fully substitutable (with the exception
# of tert-butoxy) and are considered as simple prefixes" (:17958 'methoxy' is a simple
# prefix). One of them carrying a substituent prefix is a compound prefix,
#:15762, "a simple substituent group... to which is attached one or more simple
# substituent groups"), which takes parentheses,:7232): '1-(chloromethoxy)-4-
# nitrobenzene (PIN)' (:27711), '[2-(carboxymethoxy)ethoxy]acetic acid' (:23112),
# '1-methoxy-2-(2-methoxyethoxy)ethane (PIN)' (:27756). Without a locant the digit test
# cannot see it ('methoxymethoxy', 'chloromethoxy', 'phenylmethoxy'). Every stem that
# ends the name is tried, and a split whose root is a substituent prefix decides
# ('methoxymethoxy' = 'methoxy' + 'methoxy'; 'methoxym' + 'ethoxy' has no prefix root).
# 'isopropoxy' (general nomenclature only, no substitution), the italic
# 'tert-'/'sec-' forms (removed before this test) and a bare multiplier ('dimethoxy') stay
# simple. The substituent prefix in front of the stem may itself be enclosed: the prefix
# producers pass '(methylsulfanyl)methoxy' and '(trifluoromethyl)methoxy', a root that
# ends in a closing mark, and the compound prefix takes the next mark around it
# ('[(4-methylphenyl)methoxy]benzene',,:21618).
#
# ONE rule. "Substituted" is the question is_substituted_substituent answers (c),
#:7035, "any component which is substituted automatically requires use of the
# multiplicative forms 'bis', 'tris', etc."), and the same answer decides the enclosing marks
# here (needs_brackets, is_complex_substituent) and in enclose_if_compound /
# format_substituent_prefix. So is_substituted_contracted_alkoxy is that predicate
# restricted to the tokens that end in a contracted stem, and the root reading below is
# one arm OF that predicate: its vocabulary split does not know every substituent prefix
# ('acetylmethoxy', 'ethenylmethoxy', 'bis(trifluoromethyl)methoxy',
# 'carboxy(phenyl)methoxy'), and the root reading does not know every root the split
# knows ('methylidenemethoxy'); the enclosure and the multiplier now agree on all of them.
_CONTRACTED_ALKOXY_STEMS = ("methoxy", "ethoxy", "propoxy", "butoxy", "phenoxy")
_NOT_A_PREFIX_ROOT = frozenset(("iso", "cyclo", "mono"))
_SUBSTITUENT_ROOT_ENDINGS = ("o", "oxy", "yl", ")", "]", "}")


def is_substituted_contracted_alkoxy(name_lower: str) -> bool:
    """True for a retained contracted alkoxy prefix that carries a substituent prefix
    ('methoxymethoxy', 'chloromethoxy', 'hydroxyethoxy', 'chlorophenoxy', and with an
    enclosed substituent prefix '(methylsulfanyl)methoxy'): a compound prefix
    that takes enclosing marks. False for the simple prefixes themselves and
    for a multiplied simple prefix ('dimethoxy').

    Decided by is_substituted_substituent (the one 'is this prefix substituted' rule),
    restricted to the tokens that end in a contracted stem; that predicate reads the root
    in front of the stem with _contracted_alkoxy_root_is_a_prefix."""
    return (name_lower.endswith(_CONTRACTED_ALKOXY_STEMS)
            and is_substituted_substituent(name_lower))


def _contracted_alkoxy_root_is_a_prefix(name_lower: str) -> bool:
    """The root reading of a contracted alkoxy token: a stem that ends the token, preceded by
    a root that ends like a substituent prefix ('chloro', 'hydroxy', 'phenyl', an enclosed
    '(methylsulfanyl)'), and is not 'iso', 'cyclo' or 'mono'. An arm of
    is_substituted_substituent (see the comment above _CONTRACTED_ALKOXY_STEMS).

    A root that ends in an enclosed DESCRIPTOR is not a substituent prefix: an isotope
    descriptor modifies the simple prefix it precedes, which keeps 'di'
    ('1,2-di[(13C)methyl]benzene (PIN. ', the Blue Book), so '(2H3)methoxy'
    is simple. The descriptor test is the one _enclosed_leader_is_a_substituent uses."""
    for stem in _CONTRACTED_ALKOXY_STEMS:
        if name_lower.endswith(stem):
            root = name_lower[: -len(stem)]
            if (root and root not in _NOT_A_PREFIX_ROOT
                    and root.endswith(_SUBSTITUENT_ROOT_ENDINGS)
                    and not _ends_in_an_enclosed_descriptor(root)):
                return True
    return False


def _ends_in_an_enclosed_descriptor(root: str) -> bool:
    """Does ``root`` end in an enclosed group that names nothing ('(2h3)', '(13c)', '[1,2]')?
    False for a root that does not end in a closing mark."""
    pairs = {')': '(', ']': '[', '}': '{'}
    closer = root[-1:]
    if closer not in pairs:
        return False
    opener, depth = pairs[closer], 0
    for i in range(len(root) - 1, -1, -1):
        if root[i] == closer:
            depth += 1
        elif root[i] == opener:
            depth -= 1
            if depth == 0:
                return not _enclosure_names_a_group(root[i + 1:-1])
    return False


def _enclosure_names_a_group(inner: str) -> bool:
    """An enclosed group with a real name in it (two letters in a row: '4-methylphenyl'), as
    opposed to a descriptor of only digits, commas, dots, primes, hyphens and a bare element
    or italic letter ('1,2,4', '3.2.1', '1,5-a', '4-2H', '2H3', '13C')."""
    return any(ch.isalpha() and inner[j + 1:j + 2].isalpha() for j, ch in enumerate(inner))


# Alkyl + functional group compound substituent patterns (e.g., methylamino,
# ethylamino, propylamino). These are compound substituents per IUPAC
# and require enclosing marks.
# Breadth job 1 (M03): every cycloalkyl ring size, not only C5/C6 -- a compound
# prefix takes enclosing marks, the Blue Book), so '2-cyclopropyl
# amino-2-oxo...' is '2-(cyclopropylamino)-2-oxo...'.
_ALKYLAMINO_RE = re.compile(
    r'^(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl|'
    r'nonyl|decyl|undecyl|dodecyl|tridecyl|tetradecyl|pentadecyl|'
    r'hexadecyl|heptadecyl|octadecyl|nonadecyl|icosyl|'
    r'phenyl|benzyl|'
    r'cyclo(?:prop|but|pent|hex|hept|oct|non|dec|undec|dodec)yl)'
    r'(?:amino|imino)$'
)

# a phase SUBST-01: a substituted substituent whose carrier is itself a '-yl'
# group bearing a terminal alkyl-yl tail ('cyclohexylmethyl', 'cyclopentylethyl',
# 'piperidinylmethyl') is a compound prefix requiring enclosing marks
# ('(cyclohexylmethyl)benzene',. The internal 'yl' before the terminal
# alkyl marks the compound boundary; simple alkyls ('methyl') lack it, and FG
# heads ('hydroxymethyl') do not end the head in 'yl'.
_RINGYL_ALKYL_RE = re.compile(
    r'yl(?:methyl|ethyl|propyl|butyl|pentyl|hexyl|heptyl|octyl)$'
)

_COMPOUND_OXY_PREFIXES = frozenset((
    'sulfooxy', 'sulfonyloxy', 'phosphonooxy', 'phosphonatoxy', 'carbonyloxy',
    # (the Blue Book): an acid's own acyl prefix joined to
    # 'oxy' or 'amino' by concatenation is a compound prefix, enclosed as
    # '3-(sulfooxy)propanoic acid (PIN)' (:36488) and '3-(sulfamoyloxy)propanoic
    # acid (PIN)' (:36494); 'sulfino' is the prefix of -SO-OH,:17971).
    'sulfinooxy', 'sulfoamino', 'sulfinoamino',
))

# BUG-B guard: FG types that substituent naming demonstrably handles on 1-3C branches.
# Each entry verified empirically (a phase-01 + a phase-01) to produce correct
# prefix via name_substituent on small branches in BOTH polyfunctional.py and
# composer.py code paths. FG types NOT in this set stay in the polyfunctional
# prefix list (IUPAC.
#
# NOTE (W2F-P3): 'nitro' is now IN this set. The historical exclusion (drops
# sulfanyl/nitro/amino/azido on small branches) was caused by
# composer.py's carbon-count terminal fallback returning alkyl names; with that
# fallback deleted (composer.py:8704-8711 -> None) the Tier-4 enumerator names
# nitro branches faithfully ('nitromethyl' / '1,2-dinitropropyl', verified). thiol,
# azido, secondary_amine, tertiary_amine stay OUT (their branch-path behaviour is
# unverified — separate follow-on). FG types NOT in this set stay in the
# polyfunctional prefix list (IUPAC.
BRANCH_HANDLED_FGS: frozenset = frozenset({
    'primary_alcohol',    # -> "hydroxymethyl", "2-hydroxypropyl"
    'secondary_alcohol',  # -> "hydroxy" included in branch name
    'primary_amine',      # -> "aminomethyl", "2-aminoethyl"
    'fluoro', 'chloro', 'bromo', 'iodo',  # -> "fluoromethyl" etc.
    'nitro',              # W2F-P3 -> "nitromethyl", "1,2-dinitropropyl"
    'azido',              # -> "azidomethyl", "3-azidopropyl" (substituent_naming
                          # Pass 1c3;, BB '(2-azidoethyl)benzene (PIN)')
})
# a phase (E3 Task 6) tried adding 'aldehyde' here and REVERTED it: it
# broke `tests/unit/rules/test_bugb_guard.py::test_no_dangerous_entries`
# (aldehyde/ketone/nitrile/... are deliberately excluded because this whole
# set means "trust the branch namer blindly"), and doing so UNMASKED a
# separate, pre-existing, genuinely dangerous bug: a `-CH2-CHO` branch on
# `OC(=O)C(CC=O)CCC` is mis-named "(2-hydroxyethyl)" (a different molecule,
# the aldehyde read as a hydroxyl) by the compound-substituent namer, and with
# 'aldehyde' in this set that wrong branch name would have been TRUSTED
# (skipping the parent-level "oxo" that used to at least keep the atom count
# honest, even if duplicated). still caught the resulting name in
# both directions (RT mismatch), so 0-wrong held either way, but "safe
# either way" is not "safe to ship" per this set's own documented contract.
# The real Task 6 fix (`C/C=C(\\C=O)C(CC(=O)O)CC(=O)O`'s duplicate unlocated
# "oxo") is instead scoped narrowly in `_handler_shared.py`'s FG-prefix loop
# to the UNLOCATABLE case specifically (see the comment there), leaving this
# frozenset and its dangerous-FG contract untouched.

# ---------------------------------------------------------------------------
# (b)/(d) /: the italicized STRUCTURAL prefixes that are written with a
# hyphen and are nevertheless part of a SIMPLE retained prefix name.
#
# 'tert-butyl' is a retained *preferred* prefix, BB 16270) and the Blue
# Book cites it BARE, hyphen and all: BB 16286 ***tert*-butyldi(methyl)phosphane
# (PIN)** -- no enclosing marks on the first cited group. BB 16282 adds that the
# retained name "has never been recommended for further substitution... Acceptable
# locants have never been adopted for this name", so the hyphen can never be a
# locant boundary. Two rules state the consequence, and NEITHER is:
# `****` clause (b) (BB 7070) carries the verbatim example `di-*tert*-butyl`,
# making the SIMPLE multiplier correct (not 'bis(tert-butyl)'); `****`
# clause (d) (BB 6964) supplies the hyphen. The BARE citation is inferred from
# BB 3465 `4-butyl-4-*tert*-butylcyclohexan-1-ol` (PIN) -- bare directly after a
# locant -- and BB 16286 `*tert*-butyldi(methyl)phosphane` (PIN).
# NOTE: 'N-tert-butyl' does NOT occur in the Blue Book (0 hits in 4 encodings,
# against 30 for `*tert*-butyl` as a positive control), so it must not be quoted
# as an example. is the PARENTHESES rule and its six-item list does not
# mention italicized prefixes at all.
# 'sec-' behaves identically ('sec-butyl', general nomenclature).
#
# THE ONE definition of this carve-out. Do not re-derive it inline: before
# it existed as three divergent copies (is_complex_substituent's inline strip,
# format_substituent_prefix's startswith open-code, organometallics'
# `'-' in name`) and was MISSING from needs_brackets, so the predicates disagreed
# and the union of them over-enclosed 'tert-butyl' -> '(tert-butyl)arsonic acid'.
_ITALICIZED_STRUCTURAL_PREFIXES = ("sec-", "tert-")


def strip_italicized_structural_prefix(name: str) -> Tuple[str, bool]:
    """Split a leading italicized structural prefix (``sec-`` / ``tert-``) off a
    substituent name.

    Returns ``(remainder, had_prefix)``. The hyphen of such a prefix is part of a
    SIMPLE retained name /, BB 16282/16286) and must therefore
    not be read as evidence that the name is a compound substituent.

    The remainder is what decides: ``tert-butyl`` reduces to the simple ``butyl``
    and is cited bare, while ``tert-butylsulfanyl`` reduces to the still-compound
    ``butylsulfanyl`` and keeps its enclosing marks.

    Examples:
        >>> strip_italicized_structural_prefix("tert-butyl")
        ('butyl', True)
        >>> strip_italicized_structural_prefix("sec-butyl")
        ('butyl', True)
        >>> strip_italicized_structural_prefix("2-methylpropyl")
        ('2-methylpropyl', False)
    """
    if not name:
        return name, False
    lowered = name.lower()
    for prefix in _ITALICIZED_STRUCTURAL_PREFIXES:
        if lowered.startswith(prefix):
            return name[len(prefix):], True
    return name, False


def has_structural_hyphen(name: str) -> bool:
    """Does ``name`` carry a hyphen that makes it a COMPOUND substituent?

    True for every hyphen except a leading italicized structural prefix
    (b)/(d): ``tert-butyl`` / ``sec-butyl`` are simple; ``2-methylpropyl`` and
    ``tert-butyl-dimethylsilyl`` are compound).

    Examples:
        >>> has_structural_hyphen("tert-butyl")
        False
        >>> has_structural_hyphen("2-methylpropyl")
        True
        >>> has_structural_hyphen("methyl")
        False
    """
    remainder, _ = strip_italicized_structural_prefix(name)
    return "-" in remainder


def multiplier_needs_hyphen(name: str) -> bool:
    """Does a SIMPLE multiplier joined to ``name`` keep a hyphen boundary?

    (b)/(d): ``di-tert-butyl`` (never ``ditert-butyl``), ``di-sec-butyl``. This
    is the SECOND leg of the italicized-prefix rule — the first
    (``has_structural_hyphen``) withholds enclosing marks, this one inserts the
    hyphen — and it shares the one detection primitive so the two legs can never
    disagree about what an italicized prefix is.

    Examples:
        >>> multiplier_needs_hyphen("tert-butyl")
        True
        >>> multiplier_needs_hyphen("methyl")
        False
    """
    _, had_prefix = strip_italicized_structural_prefix(name)
    return had_prefix


def italicized_prefix_is_bare(name: str) -> bool:
    """Must ``name`` be cited BARE because its only hyphen is an italicized one?

    ``True`` only when ``name`` leads with an italicized structural prefix AND
    the remainder is itself simple — ``tert-butyl`` / ``sec-butyl`` (BB 16286
    cites ``*tert*-butyldi(methyl)phosphane`` (PIN) with the group bare).
    ``False`` for everything else, including a compound remainder
    (``tert-butylsulfanyl`` keeps its marks) and any name that does not
    lead with such a prefix.

    This is the form a site with its OWN compound test should call: because it
    returns ``False`` for every non-italicized name, dropping it in front of an
    existing raw ``'-' in name`` test changes that site's behaviour for
    italicized-led names ONLY, and never widens or narrows it for anything else.

    The remainder is judged by the SAME union ``enclose_if_compound`` uses
    (``needs_brackets`` OR ``is_complex_substituent``), because neither is
    complete alone: ``needs_brackets`` misses the bare two-prefix compound
    ``butylamino``, so using it by itself made this predicate call
    ``tert-butylamino`` bare while the canonical path enclosed it. Erring toward
    "compound" is the safe direction — it can only keep marks, never drop them.

    Examples:
        >>> italicized_prefix_is_bare("tert-butyl")
        True
        >>> italicized_prefix_is_bare("tert-butylsulfanyl")
        False
        >>> italicized_prefix_is_bare("tert-butylamino")
        False
        >>> italicized_prefix_is_bare("2-methylpropyl")
        False
        >>> italicized_prefix_is_bare("methyl")
        False
    """
    remainder, had_prefix = strip_italicized_structural_prefix(name)
    if not had_prefix:
        return False
    if has_structural_hyphen(remainder) or any(ch.isdigit() for ch in remainder):
        return False
    return not (needs_brackets(remainder) or is_complex_substituent(remainder))


# Shared C1-C20 alkyl roots used by needs_brackets, is_complex_substituent,
# and regex patterns. Extends coverage beyond the original C1-C10 lists.
_ALKYL_ROOTS_FULL = (
    'methyl', 'ethyl', 'propyl', 'butyl', 'pentyl',
    'hexyl', 'heptyl', 'octyl', 'nonyl', 'decyl',
    'undecyl', 'dodecyl', 'tridecyl', 'tetradecyl', 'pentadecyl',
    'hexadecyl', 'heptadecyl', 'octadecyl', 'nonadecyl', 'icosyl',
)

# Chalcogen compound-substituent suffixes that, fused to an alkyl root, form a
# COMPLEX (compound) substituent requiring enclosing marks per IUPAC /
# — '(methylsulfanyl)', '(methylselanyl)', '(methyltellanyl)'. a phase
# assembly/parenthesisation fix : the Se/Te analogues (selanyl/tellanyl + the oxidized
# seleninyl/selenonyl/tellurinyl/telluronyl) were missing, so 'methylselanyl' was
# wrongly classed simple -> '1-methylselanylpropane' instead of PIN
# '1-(methylselanyl)propane'.
_COMPOUND_S_SUFFIXES_COMPLEX = (
    'sulfinyl', 'sulfonyl', 'sulfanyl',
    'seleninyl', 'selenonyl', 'selanyl',
    'tellurinyl', 'telluronyl', 'tellanyl',
    # DD2 Fix B (Phase D, (1)): (alkyl)disulfanyl is a compound
    # substituent — '(methyldisulfanyl)methane'. Bare 'disulfanyl' (terminal
    # -S-SH, no alkyl root) is simple and is not matched by the root loop.
    'disulfanyl',
)

# C4-C: roots that are NOT a substituent group — a chalcogen suffix carrying
# only one of these is still a SIMPLE prefix and must stay bare. 'disulfanyl' is
# 'di' + 'sulfanyl' (terminal -S-SH), 'trisulfanyl' is 'tri' + 'sulfanyl', and
# so on: the leading morpheme multiplies the chalcogen chain rather than naming a
# substituent on it. Without this carve-out the general rule below would enclose
# a bare '(disulfanyl)'.
_MULTIPLYING_PREFIX_ROOTS = frozenset({
    'di', 'tri', 'tetra', 'penta', 'hexa', 'hepta', 'octa',
    'bis', 'tris', 'tetrakis',
})

# Functional group prefixes that, when fused with alkyl roots, form compound
# substituents requiring enclosing marks per IUPAC.
# Hoisted to module level for performance (was recreated inside needs_brackets).
_COMPOUND_FG_PREFIXES = (
    'hydroxy', 'carboxy', 'amino', 'oxo', 'formyl', 'cyano',
    'nitro', 'mercapto', 'sulfanyl', 'phospho',
    # Wave-2 C2: '(isothiocyanatomethyl)benzene' BB verbatim
    'isothiocyanato', 'isocyanato',
    #: 'azidomethyl' is azido + methyl, a compound prefix; BB
    # '(2-azidoethyl)benzene (PIN)' (:25995). Reached once the polyfunctional
    # substituent namer consumes the azido group (Pass 1c3, TRIAGE g7 C14).
    'azido',
    'fluoro', 'chloro', 'bromo', 'iodo',
    'difluoro', 'trifluoro', 'dichloro', 'trichloro',
    'dibromo', 'tribromo',
)

# Wave2 T2b: heteroatom-substituted amino/oxy preselected prefixes
# / — compound substituents that take enclosing
# marks per. BB PINs: '4-(hydroxyamino)phenol',
# '2-(aminooxy)ethan-1-amine', '8-(chloroamino)octanoic acid'.
_COMPOUND_HETEROATOM_AMINO_PREFIXES = frozenset({
    'hydroxyamino', 'aminooxy',
    'fluoroamino', 'chloroamino', 'bromoamino', 'iodoamino',
    # W3-P11: -O-NO2 nitrooxy is a compound (acyl-on-oxy)
    # preselected prefix -> enclosing marks: '3-(nitrooxy)propanoic acid'.
    'nitrooxy',
    #: the oxime prefix =N-OH 'hydroxyimino' is a compound
    # substituent (hydroxy on imino) and takes enclosing marks. The
    # polyfunctional acyclic path already wrapped it via format_fg_prefix, but the
    # ring/seniority composer path emitted it BARE ('4-hydroxyimino-...' instead
    # of '4-(hydroxyimino)-...'). Centralising it here fixes both paths.
    'hydroxyimino',
})

#: parent-hydride stems that, when carrying an acyl-suffix-family
# ending, form a compound prefix INCLUDING a parent-hydride name (so it needs
# enclosing marks: 'cyclohexanecarbonyl' -> '(cyclohexanecarbonyl)'). Curated on
# purpose -- a simple substituent root such as 'hydroxy' (hydroxycarbonimidoyl) is
# NOT a parent hydride and must stay bare (see needs_brackets).
_ACYL_PARENT_HYDRIDE_STEMS = (
    'hydrazine',
    'cyclopropane', 'cyclobutane', 'cyclopentane', 'cyclohexane',
    'cycloheptane', 'cyclooctane',
    'benzene', 'naphthalene',
    'furan', 'thiophene', 'pyrrole', 'pyridine', 'pyrimidine', 'pyrazine',
    'imidazole', 'thiazole', 'oxazole', 'oxane', 'thiane', 'piperidine',
    'azetidine', 'oxetane', 'aziridine', 'oxirane',
)
# acyl-suffix family expressed as a substituent prefix.
_ACYL_SUFFIX_FAMILY = frozenset({
    'carbonyl', 'carbothioyl', 'carboselenoyl', 'carbotelluroyl',
    'carbohydrazonoyl', 'carbohydrazido', 'carboximidoyl', 'carbonimidoyl',
})

# Wave2 T3b: parent-hydride stems used as roots of the PIN acid-stem
# chalcogen-oxide prefixes ('methanesulfinyl' / 'benzenesulfonyl' /
# 'cyclohexanesulfinyl',. Consumed by needs_brackets alongside
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

# Used by alpha_sort_key — pre-compiled regex patterns
# A leading locant set: arabic numerals, each optionally with a letter (4a) and
# primes (3', 2'') -- the primed locants of a spiro / ring-assembly parent are
# locants like any other and never letters preamble, the Blue Book:
# letters are compared "unless used as locants"). Without the primes, the
# rendered prefix "3',4,4'-trihydroxy" kept its locants and its multiplier, and
# keyed at "t" (or, as a raw string, at the digit "3").
_LOCANT_PREFIX_RE = re.compile(r"^\d+[a-z]?['\u2032\u2033]*(?:,\d+[a-z]?['\u2032\u2033]*)*-")
_N_LOCANT_PREFIX_RE = re.compile(r'^[nN],?[nN]?-')
# Leading indicated-hydrogen descriptor (e.g. '1H-', '2H-'). Like a locant, the
# italic indicated H is IGNORED for alphabetization (IUPAC, so
# '(1H-imidazol-5-yl)' must sort at 'i', not at the leading digit '1'.
_INDICATED_H_PREFIX_RE = re.compile(r'^\d+[hH]-')

# ---------------------------------------------------------------------------
# alphanumerical-order NOISE: what may never enter the PRIMARY key.
#
# `### **** ALPHANUMERICAL ORDER` closes its preamble (the Blue Book)
# with, verbatim:
#
# In these subsections the principles of alphanumerical order do not include
# Greek letters (except in conjunctive names) or isotopic or stereochemical
# descriptors.
#
# and:44601 says the same from the other side: capitalized CIP stereodescriptors
# "are written in italics to indicate that they are not involved in the primary
# stage of alphanumerical order". The preamble (:3442) also excludes Roman
# letters "used as locants... or in an isotopic descriptor".
#
# So `[(E)-2-phenylethenyl]` alphabetizes at `p`, NOT at `e`. This matters twice
# over: since wired (g) ("lowest locants to the substituent
# cited first as a prefix",:3307) the key decides LOCANTS, not merely citation
# order, so a wrong key is a wrong name rather than a cosmetic one.
#
# ⚠ THIS CANNOT SHIP WITHOUT ``cip_descriptor_rank_key``. Before that tier
# existed, the descriptor sitting in the primary key was silently breaking ties
# that forbids it to break -- gold row W2F-P8-01 pits `(1R)-1-chloroethyl`
# against `(1S)-1-chloroethyl`, whose Roman letters are identical. Stripping the
# descriptor alone made them tie and the order fell through to the engineering
# fallback, flipping the PIN and failing the gate at 1649/1651. (j) is what
# is supposed to decide there. The two changes are one change.
#
# A "descriptor run" is one or more descriptor tokens joined by hyphens and
# terminated by one, anchored either at the start of the prefix or immediately
# after an opening enclosing mark -- which is where a substituent's own
# configuration sits (`[(1s,4s)-4-methylcyclohexyl]`, `{(1S,2S)-3-oxo-...}`).
#
# `d`/`l` (the italic small-capital configurational prefixes of `beta-D-gluco-
# pyranosyl`) are admitted ONLY as a NON-INITIAL token of a run. A bare leading
# `d-` is far more likely to be an ordinary name than a descriptor, and this
# regex must never corrupt one: anything that does not match keeps the previous
# behaviour exactly.
#
# ⚠ Deliberately NOT `rules.stereochemistry.strip_stereo`, though the two overlap.
# That one answers "does the CONSTITUTIONAL form of a whole NAME parse?" for the
# OPSIN gate probe; this one answers "which characters may enter a sort
# key for one PREFIX?" and is therefore broader (Greek + isotopic + D/L) and
# applied globally rather than only leading + nested. Unifying them would break
# one of the two callers -- different questions with overlapping syntax.
_ALPHA_NOISE_HEAD = (
    r"(?:"
    r"\((?:[0-9]+[a-z]?[rsez*]|[rsez]|rs|sr|[,\-+ ])+\)"   # (R) (2S) (1E,3E) (RS)
    r"|rel|rac|cis|trans|\(±\)"                        # relative / racemic / geometric
    r"|alpha|beta|gamma|delta|λ\d+|[α-ω]"    # Greek, spelled/literal; λ-convention
    r"|\[\d+[a-z]{1,2}\]"                                   # isotopic: [2H], [13C]
    r")"
)
_ALPHA_NOISE_TAIL = r"(?:" + _ALPHA_NOISE_HEAD + r"|[dl])"
_ALPHA_NOISE_RUN_RE = re.compile(
    r"(?:^|(?<=[\[({]))" + _ALPHA_NOISE_HEAD + r"-(?:" + _ALPHA_NOISE_TAIL + r"-)*",
    re.IGNORECASE,
)

# An isotopic descriptor is BRACKETED and JUXTAPOSED -- no hyphen joins it to the
# name it modifies -- so the hyphen-terminated run above can never reach it.
# (f)'s own example list prints `di([4-2H]benzoyl)` (:7104), i.e. the
# descriptor carries its own locant inside the brackets. Requiring a following
# letter keeps this off ordinary enclosing marks: `[pent-2-en-1-yl]` cannot match
# (it does not end in digits-then-element), and neither can a bare `[3]`.
_ALPHA_ISOTOPE_RE = re.compile(r"\[[\d,\-]*\d+[a-z]{1,2}\](?=[a-z])", re.IGNORECASE)


def strip_alphanumerical_noise(text: str) -> str:
    """Remove everything (``the Blue Book``) bars from a sort key.

    Greek letters, isotopic descriptors and stereochemical descriptors are not
    part of alphanumerical order. THE single definition of that exclusion, so the
    ~140 ``alpha_sort_key`` call sites cannot each grow their own -- one already
    had (``rules/ring_substituents.py`` open-coded ``_alpha_sort_key(_strip_stereo(nm))``).

    What is removed here does not vanish from the ordering: it reappears at
    ``prefix_citation_sort_key`` tier 3 via:func:`cip_descriptor_rank_key`,
    which is where (j) says configuration belongs.

    Examples:
        >>> strip_alphanumerical_noise('(e)-3-phenylprop-2-en-1-yl')
        '3-phenylprop-2-en-1-yl'
        >>> strip_alphanumerical_noise('beta-d-glucopyranosyloxy')
        'glucopyranosyloxy'
        >>> strip_alphanumerical_noise('[(2z)-pent-2-en-1-yl]')
        '[pent-2-en-1-yl]'
        >>> strip_alphanumerical_noise('[4-2h]benzoyl')
        'benzoyl'
        >>> strip_alphanumerical_noise('decyl') # not a descriptor
        'decyl'
    """
    return _ALPHA_ISOTOPE_RE.sub('', _ALPHA_NOISE_RUN_RE.sub('', text))


# ============================================================================
# Terminal Functional Group Types (IUPAC
# ============================================================================

# Terminal functional groups that always occupy position 1 by definition.
# The chain is numbered FROM the terminal group, so locant-1 is implicit.
# Canonical source: IUPAC. Used by should_omit_locant_one.
def unbranched_alkylidene_name(mol, c_idx, exclude_idx):
    """Ylidene name for an UNBRANCHED all-carbon H-saturated chain rooted at
    the double-bonded carbon *c_idx* (walking away from *exclude_idx*):
    methylidene / ethylidene / propylidene (Wave-2 completion C; shared by the
    sulfine and azinic-acid namers). Fail-closed None on
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
    "peroxy_acid",      # Always at chain end: propaneperoxoic acid)
    "imidic_acid",      # Always at chain end: ethanimidic acid)
    "hydrazonic_acid",  # Always at chain end: methanehydrazonic acid)
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
    # a phase Tier FRN-A chalcogen acids functional replacement; IUPAC PIN
    # 'propaneselenoic Se-acid' / 'propanetelluroic Te-acid' — never carries locant-1)
    "selenoic_Se_acid", "selenoic_O_acid", "diselenoic_acid",
    "telluroic_Te_acid", "telluroic_O_acid", "ditelluroic_acid",
    # a phase Tier FRN-B chalcogen amides functional replacement; IUPAC PIN
    # 'propanethioamide' / 'propaneselenoamide' — never carries locant-1)
    "thioamide", "selenoamide", "telluroamide",
    # a phase Tier FRN-C chalcogen aldehydes — "propanethial" / "propaneselenal" /
    # "propanetellural"; suffix is "-thial"/"-selenal"/"-tellural" per seniority.py SUFFIX_FORMS,
    # but FG identifier is the long form. Always terminal — never carries locant-1.)
    "thioaldehyde", "selenoaldehyde", "telluroaldehyde",
    # D-FOLLOWON item 8: the amidine (imidamide / carboximidamide)
    # characteristic carbon is ALWAYS chain-terminal (C1), exactly like
    # nitrile / amide / aldehyde — so its suffix locant-1 is implicit and elided:
    # 'propanimidamide' not 'propan-1-imidamide'.
    "amidine",
    # R8a: the hydrazide characteristic carbon is ALWAYS chain-terminal
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
    # a phase, the Blue Book): the imidohydrazide (amidrazone)
    # characteristic carbon -C(=NH)-NH-NH2 is ALWAYS chain-terminal, exactly like
    # its siblings hydrazide / amidine / hydrazonamide above, so its suffix
    # locant-1 is implicit and elided -- 'cyclohexanecarboximidohydrazide' and
    # 'ethanediimidohydrazide', never 'cyclohexane-1-carboximidohydrazide' /
    # 'ethane-1,2-diimidohydrazide'. (It was already in _ETHANE_SUFFIX_ELIDE_FGS,
    # which only covered the symmetric-ethane MONO case; the ring appended-carbon
    # and the chain DI cases bypassed that and kept the locant.)
    "imidohydrazide",
})


# ============================================================================
# Centralized Locant-1 Elision (IUPAC
# ============================================================================


# Amine-family suffix FG classes whose locant elides on a symmetric 2-carbon
# (ethane) parent: 'ethanamine' not 'ethan-1-amine'. Scoped to the
# amine family ON PURPOSE — it is the only suffix that routes through the general
# _generate_suffix path AND needs this elision (DD1 Fix 4 masking pair). -ol /
# -thiol have dedicated handlers that already elide the bare case ('ethanol',
# 'ethanethiol'); including them here changed the SUBSTITUTED-alcohol locant
# (2-phenylethan-1-ol -> 2-phenylethanol), a separate unsettled question outside
# Phase B. Carbonyl-type suffixes (-one/-al) always cite the locant in PINs
#, 1-phenylethan-1-one) and are likewise excluded.
_ETHANE_SUFFIX_ELIDE_FGS = frozenset({
    "primary_amine", "secondary_amine", "tertiary_amine",
    # DD2 (Phase D, /: the peroxol family is a single monovalent
    # chalcogen suffix on a terminal carbon — exactly like -amine — so on a
    # monosubstituted ethane the two carbons are equivalent and the locant is
    # omitted: 'ethaneperoxol' verbatim), 'ethanedithioperoxol', not
    # 'ethane-1-peroxol'. These route through _generate_suffix (no dedicated
    # handler), so the elision must be declared here.
    "hydroperoxide", "so_thioperoxol", "os_thioperoxol", "dithioperoxol",
    # D-FOLLOWON item 1 /: the S/Se/Te oxoacid suffixes
    # (-sulfinic/-sulfonic/-selenonic/-seleninic/-telluronic/-tellurinic acid)
    # attach to a chain carbon whose locant is now computed (the _pick_locant_atom
    # neighbor-walk fix). On the symmetric 2-carbon (ethane) parent the two carbons
    # are equivalent and the suffix sits on a terminal carbon, so the locant is
    # omitted: 'ethanesulfinic acid' / 'ethaneselenonic acid' (PIN), NOT
    # 'ethane-1-sulfinic acid'. methane (chain_length==1) is already elided by
    # Rule 1; a chain of 3+ carbons (butane-2-sulfinic acid) keeps its locant.
    "sulfinic_acid", "sulfonic_acid",
    # W3-P04 /: the FRN-modified sulfur-oxo-acid suffixes
    # attach to a chain carbon exactly like sulfonic/sulfinic, so their locant
    # elides on the symmetric 2-carbon (ethane) parent (ethanesulfonothioic
    # S-acid, PIN); methane is covered by Rule 1; C3+ keeps the locant.
    "sulfonoperoxoic_acid", "sulfonothioic_S_acid",
    "sulfinimidic_acid", "sulfonimidic_acid",
    "selenonic_acid", "seleninic_acid", "telluronic_acid", "tellurinic_acid",
    # C1 /: sulfonohydrazide is the N-analogue of sulfonic
    # acid and attaches to a chain carbon exactly the same way, so it elides its
    # locant on a symmetric 2-carbon (ethane) parent: 'ethanesulfonohydrazide'
    # (PIN), not 'ethane-1-sulfonohydrazide'. methane is covered by Rule 1; a
    # 3+ carbon chain keeps its locant (parallels sulfonohydrazide above).
    "sulfonohydrazide",
    # Wave2 /: the sulfonamide characteristic S attaches
    # to a chain carbon exactly like sulfonic_acid, so its suffix locant elides on the
    # symmetric 2-carbon (ethane) parent -> 'ethanesulfonamide' (PIN, OPSIN-RT), not
    # 'ethane-1-sulfonamide'; N-substituents don't perturb the ethane symmetry
    # (N-methylethanesulfonamide). A 3+ carbon chain keeps its locant
    # (propane-1-sulfonamide).
    "primary_sulfonamide", "secondary_sulfonamide", "tertiary_sulfonamide",
    # Task Y /: sulfinamide's characteristic S attaches to
    # a chain carbon exactly like sulfonamide's, so the suffix locant elides on the
    # symmetric 2-carbon parent -> 'ethanesulfinamide', not 'ethane-1-sulfinamide'.
    # A 3+ carbon chain KEEPS its locant, which the Blue Book's own worked PINs
    # require: `butane-2-sulfinamide`  and
    # `N-hydroxypropane-1-sulfinamide` . (This closes the TODO that stood
    # here reading "Sulfinamide (no FG defined yet) deferred".)
    "primary_sulfinamide", "secondary_sulfinamide", "tertiary_sulfinamide",
    # Wave2 /: sulfonimidamide attaches to a chain
    # carbon exactly like sulfonamide -> 'ethanesulfonimidamide' (OPSIN-RT),
    # not 'ethane-1-sulfonimidamide'. propane-1- keeps its locant.
    "sulfonimidamide",
    # Wave2 item 25): sulfinimidamide, S-suffix parallel ->
    # 'ethanesulfinimidamide' (propane-1- keeps its locant).
    "sulfinimidamide",
    # Wave-2 P1AM: Se imidamide suffixes attach to a chain carbon
    # like the S siblings -> 'methaneseleninimidamide' / 'methaneselenonimidamide'.
    "seleninimidamide",
    "selenonimidamide",
    # Wave-2 P1AM Task 7 /, BB 2877): amidrazones are
    # cited without terminal locants -> 'methanimidohydrazide' (not
    # 'methan-1-imidohydrazide').
    "imidohydrazide",
    # Wave2 /: the imine =NH sits on a chain carbon;
    # on the symmetric 2-carbon (ethane) parent the locant elides -> 'ethanimine'
    # (BB VERBATIM 'N-methylethanimine (PIN)'), not 'ethan-1-imine'. A 3+ carbon
    # chain keeps its locant (BB VERBATIM 'N-hydroxypropan-1-imine (PIN)').
    "imine",
    # Wave2 /: selenol/tellurol are monovalent-chalcogen
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
    """Determine whether locant-1 should be omitted per IUPAC.

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

    # Rule 2: di-/trinuclear bond locants (d)): omit bond locant.
    # ethene/ethyne (chain_length 2, only one bond position) always omit;
    # propene/propyne (chain_length 3) omit ONLY when the caller passes
    # chain_length=3, which it does solely for an UNSUBSTITUTED trinuclear
    # parent (Wave2 — a substituent makes the position distinctive:
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
    # 1 and is omitted: 'ethanamine'/'ethanol'/'ethanethiol', not
    # 'ethan-1-amine'. This is FG-class-gated on purpose: carbonyl-type suffixes
    # (-one/-al) ALWAYS cite their locant in PINs — '1-phenylethan-1-one',
    # 'propan-2-one'), so a ketone on ethane must NOT elide. methane
    # (chain_length==1) is covered unconditionally by Rule 1; multi-instance
    # suffixes (ethane-1,2-diamine) keep locants (is_monosubstituted is False).
    #
    # (the Blue Book, example:44186 '(2-13C)ethan-1-ol [not (2-13C)ethanol]'):
    # when an isotopic modification forces a locant in the current scope, ALL
    # locants must be specified and none omitted — so the ethane suffix locant is
    # restored (C[13CH2][15NH2] -> (1-13C,15N)ethan-1-amine, not...ethanamine).
    # This is the chain-suffix analogue of the ring guard at
    # handlers/_handler_shared.py:440. Keyed on locants_are_forced (NOT the
    # broader scope_has_isotopic_modification) so a single-atom label whose
    # descriptor is itself locant-free — (15N)ethanamine — still elides.
    if (context == "suffix" and not is_ring
            and chain_length == 2 and is_monosubstituted
            and fg_type in _ETHANE_SUFFIX_ELIDE_FGS):
        from .locant_omission import locants_are_forced  # local: avoid import cycle
        if not locants_are_forced():
            return True

    # Rule 4: Monosubstituted rings. -06: the ring callers now pass
    # `is_monosubstituted` = `is_only_one_substitutable_position(parent_hydride)`
    # (the real topological-symmetry predicate below) AND-ed with the count, so a
    # substituted ASYMMETRIC ring (e.g. cyclohexene) correctly KEEPS the locant.
    if context == "prefix" and is_ring and is_monosubstituted:
        if is_heterocyclic:
            return False  # Position matters in heterocycles
        # (the Blue Book, "Locants are not omitted when there is a possibility
        # of isomers"): a positional isotopic label ON THE RING breaks the symmetry
        # that licensed the monosubstituted omission, so the substituent locant is
        # restored -- the Blue Book '1-(79Br)bromo(2-13C)benzene', not
        # '(79Br)bromo(2-13C)benzene'. The label is stripped before this ring is named,
        # so the fact is carried down by the ambient parent-positional isotope scope
        # rules/isotopes.py enters ONLY when a labelled atom sits on the PARENT ring
        # (loc_rank >= 1 AND parent-labelled). It is DELIBERATELY the narrow
        # parent-positional flag, NOT locants_are_forced: a positional locant a label
        # needs INSIDE A SUBSTITUENT -- '(1,1,2,2,2-pentafluoro(1-13C)ethyl)benzene',
        # the ¹³C on the ethyl -- leaves the ring symmetric and must NOT restore a ring
        # locant, and locant-free descriptors ('(2H3)methoxybenzene', '(2H6)benzene')
        # still elide. This is the ring-prefix analogue of Rule 3b above.
        from .locant_omission import (  # local: avoid import cycle
            parent_scope_has_positional_isotope)
        if parent_scope_has_positional_isotope():
            return False
        return True  # Symmetric carbocyclic: omit

    # Rule 5: Monosubstituted hydrocarbon chain at position 1.
    #: the locant '1' is omitted ONLY where the position is
    # unambiguous — methane (Rule 1, chain_length==1) and ETHANE (chain_length==2,
    # the two carbons are equivalent so there is a single monosubstitution product
    # -> 'chloroethane'). For propane and longer the terminal substituent is
    # distinguishable from interior positions ('1-chloropropane' != '2-chloropropane',
    # '1-chloropentane' is the PIN), so the locant MUST be cited. a phase
    # (the Blue Book; gold 'ClCCCCC' -> '1-chloropentane'). Previously this
    # used `chain_length > 1`, which wrongly elided the locant for all chains.
    if context == "prefix" and not is_ring and is_monosubstituted and chain_length == 2:
        return True

    # Rule 6: Mono-cycloalkene bond locant
    # Single double bond in a ring: locant omitted (cyclohexene, not cyclohex-1-ene)
    if context == "bond" and is_ring and is_monosubstituted:
        return True

    return False


def is_only_one_substitutable_position(parent_mol) -> bool:
    """-06 (a phase): topological-symmetry predicate for locant-1 elision.

    Returns True iff every substitutable ring position of ``parent_mol`` is
    equivalent — i.e. all H-bearing ring carbons share ONE RDKit canonical rank
    (``CanonicalRankAtoms(breakTies=False)`` = symmetry equivalence classes). This
    is the real Blue Book condition "only one kind of substitutable
    hydrogen" / "no isomer generated by moving") that replaces the
    ``num_double==1`` / ring-monosubstituted COUNT PROXIES, which conflate "one
    feature" with "ring is symmetric".

    ``parent_mol`` MUST be the PARENT HYDRIDE for the rule being decided, NOT the
    decorated molecule:
      * substituent-prefix locant: ring + unsaturation, substituents
        stripped (cyclohexane -> True -> methylcyclohexane; cyclohexene -> False ->
        3-bromocyclohex-1-ene; benzene -> True -> toluene).
      * bond locant (d)/4.4): the SATURATED ring WITH substituents (so
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

    Returns 'methyl' for 1, 'ethyl' for 2,..., 'decyl' for 10.
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


def simple_multiplier_word(n: int) -> Optional[str]:
    """The simple multiplying prefix for ``n``, or ``None`` when no
    word can be formed.

    ``SIMPLE_MULTIPLIERS`` is only the BASIC-term table (Table 1.4,,
    ``the Blue Book``) and stops at ``20: icosa``. Everything above 20 is
    not tabulated but COMPOSED, per (``the Blue Book``): the
    basic terms are cited "in the order opposite to that of the constituent
    digits in the arabic numbers", joined without hyphens -- 21 ``henicosa``,
    22 ``docosa``, 31 ``hentriaconta`` (verbatim examples at ``:2820``-``:2821``).
    ``data.chain_names.get_chain_prefix`` already implements exactly that
    composition for chain stems, so the multiplier is its stem plus the
    terminal ``a``; this function is the single place that states the rule.

    Returns ``None`` -- never a bare integer -- for any ``n`` the composition
    cannot spell (``n < 2``, or outside ``get_chain_prefix``'s 1-9999 range),
    so callers FAIL CLOSED instead of emitting a non-word such as ``"21oxa"``
    or ``"21cyclo"``. Note ``n == 1`` has no multiplying prefix at all (the
    unmultiplied form is used), which is why it is ``None`` and not ``""``:
    a caller that wants the empty string must say so itself.
    """
    if n < 2:
        return None
    if n in SIMPLE_MULTIPLIERS:
        return SIMPLE_MULTIPLIERS[n]
    from ..data.chain_names import get_chain_prefix
    try:
        prefix = get_chain_prefix(n)
    except ValueError:
        # Outside the composition's supported range -> no word exists to emit.
        return None
    return prefix if prefix.endswith("a") else prefix + "a"


def needs_brackets(name: str) -> bool:
    """Determine if a substituent name is a compound substituent needing parentheses.

    Per IUPAC, compound substituents (those that contain locants,
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
        >>> needs_brackets("tert-butyl")
        False
        >>> needs_brackets("tert-butylsulfanyl")
        True
    """
    if not name:
        return False

    # Already wrapped in parentheses or square brackets -- skip
    if (name.startswith('(') and name.endswith(')')) or \
       (name.startswith('[') and name.endswith(']')):
        return False

    # / (BB 16286 '*tert*-butyldi(methyl)phosphane' (PIN), cited
    # BARE): a leading italicized structural prefix is part of a SIMPLE retained
    # name, so it must not trip the hyphen rule below. Decide on the REMAINDER --
    # 'tert-butyl' -> 'butyl' (simple, bare) but 'tert-butylsulfanyl' ->
    # 'butylsulfanyl' (still compound by the chalcogen rule, keeps its marks).
    # THE single carve-out site; see strip_italicized_structural_prefix.
    _remainder, _had_italicized = strip_italicized_structural_prefix(name)
    if _had_italicized:
        return needs_brackets(_remainder)

    # Contains a digit (has locants): definitely compound
    if any(ch.isdigit() for ch in name):
        return True

    # Contains a hyphen (compound substituent): definitely compound
    if '-' in name:
        return True

    # a phase follow-on: acyl-substituted amino prefixes (-oylamino) are
    # compound substituents (a substituted amino) and take enclosing marks per
    # — carbamoylamino (the urea FG prefix), ethanoylamino,
    # benzoylamino, sulfamoylamino, etc. The '-oyl' acyl ending distinguishes
    # them from simple alkylamino (methylamino ends in 'ylamino', not
    # 'oylamino'). Already-bracketed forms returned False above, so this only
    # promotes the unbracketed urea prefix to '(carbamoylamino)'.
    if name.lower().endswith('oylamino'):
        return True

    # Wave2 T2b: heteroatom-substituted amino/oxy preselected prefixes are
    # compound substituents and take enclosing marks — BB PINs
    # write 4-(hydroxyamino)phenol, 2-(aminooxy)ethan-1-amine and
    # 8-(chloroamino)octanoic acid. Simple preselected prefixes (diazenyl,
    # amino, isocyanato) stay bare.
    if name.lower() in _COMPOUND_HETEROATOM_AMINO_PREFIXES:
        return True

    name_lower = name.lower()

    # Halogen-substituted alkoxy is a COMPOUND substituent: a halogen
    # prefix fused onto a simple alkoxy -> '(chloromethoxy)', not bare. BB PIN
    # '1-(chloromethoxy)-4-nitrobenzene'. Bare 'methoxy'/'ethoxy' stay
    # simple; only the halo-alkoxy compound is caught here.
    if _HALO_ALKOXY_RE.match(name_lower):
        return True
    # Any other substituent prefix on a retained contracted alkoxy is compound too
    #:27667,,: '(methoxymethoxy)', '(phenylmethoxy)'.
    if is_substituted_contracted_alkoxy(name_lower):
        return True

    # Functional group prefixes fused with alkyl names are compound substituents.
    # Examples: hydroxymethyl, carboxymethyl, aminoethyl, oxoethyl, formylmethyl
    # But NOT: methoxy, ethoxy (these are simple ether prefixes, single concept)
    # Uses module-level _COMPOUND_FG_PREFIXES and _ALKYL_ROOTS_FULL (C1-C20).
    # The FG prefix may carry a basic multiplying prefix: 'dihydroxymethyl' is
    # two hydroxy on methyl, as compound a prefix as 'hydroxymethyl'
    # "Parentheses are used around compound (see and complex...
    # prefixes", the Blue Book; '(chloromethyl)silane (PIN)':7236). Left
    # bare, '2-dihydroxymethylphenol' was shipped for 2-(dihydroxymethyl)phenol.
    # A multiplied ALKYL ('dimethyl') is no FG prefix and stays simple.
    _fg_heads = [name_lower]
    for _mult in ('di', 'tri', 'tetra'):
        if name_lower.startswith(_mult):
            _fg_heads.append(name_lower[len(_mult):])
    for fg in _COMPOUND_FG_PREFIXES:
        for _head in _fg_heads:
            if _head.startswith(fg):
                remainder = _head[len(fg):]
                # Check if the remainder is an alkyl root
                for alkyl in _ALKYL_ROOTS_FULL:
                    if remainder == alkyl:
                        return True

    # Compound sulfur/selenium/tellurium prefixes: alkyl + sulfinyl/sulfonyl/sulfanyl
    # Per IUPAC, "methylsulfinyl" = methyl + sulfinyl = compound substituent
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

    # W2F-P2: ACYL-substituted chalcogen prefixes — an acyl group on
    # a sulfanyl/selanyl/tellanyl (acetylsulfanyl, formylsulfanyl,
    # propanoylsulfanyl, benzoylsulfanyl,...) — are compound substituents and
    # take enclosing marks: '9-(acetylsulfanyl)-9-oxononanoic acid'
    # BB 18128). The acyl stem is retained 'acetyl'/'formyl' or ends in '-oyl'.
    for s_suffix in _COMPOUND_S_SUFFIXES_COMPLEX:
        if name_lower.endswith(s_suffix):
            acyl = name_lower[: -len(s_suffix)]
            if acyl and (acyl.endswith("oyl") or acyl in ("acetyl", "formyl")):
                return True

    # C4-C: ANY substituent root carrying a chalcogen suffix is a
    # compound prefix, not just the alkyl/hydride-stem roots enumerated above.
    # The root loop at:792 is keyed on _ALKYL_ROOTS_FULL + _HYDRIDE_STEM_ROOTS,
    # so aryl ('phenyl'), cycloalkyl ('cyclohexyl') and alkoxy ('methoxy',
    # 'tert-butoxy') roots all fell through and shipped UNENCLOSED. That is a
    # wrong-structure defect, not a style one: '2-phenyldisulfanylethan-1-ol'
    # reparses as 2-phenyl + disulfanyl (OPSIN -> CC(O)SSc1ccccc1, a DIFFERENT
    # molecule), which is why suppressed it.
    # The Blue Book requires the marks for exactly these roots:
    # '4-(phenylsulfanyl)piperidine (PIN)' BB 27832
    # '(methoxysulfanyl)cyclohexane (PIN)' BB 27914
    # '[(methoxysulfanyl)oxy]methane (PIN)' BB 39479
    # 'methyl 4-[(phenylsulfanyl)sulfonyl]naphthalene-1-carboxylate (PIN)' BB 31717
    # Stated as a RULE over the root rather than a longer list, because every
    # extension of that list has leaked the next root (this is the third such
    # gap found in this predicate family). A bare chalcogen prefix keeps its
    # simple status: the root must be non-empty and must not be a mere
    # multiplying prefix ('disulfanyl' itself, 'trisulfanyl').
    for s_suffix in _COMPOUND_S_SUFFIXES_COMPLEX:
        if name_lower.endswith(s_suffix):
            root = name_lower[: -len(s_suffix)]
            if root and root not in _MULTIPLYING_PREFIX_ROOTS:
                return True

    # W2F-P2: benzyl-based oxy/chalcogen prefixes (benzyloxy,
    # benzylsulfanyl,...) are compound substituents (benzyl = substituted
    # methyl) and take enclosing marks: '9-(benzyloxy)-9-oxononanoic acid'
    # /:18097). Bare 'benzyl' and 'benzyloxymethyl' (remainder
    # not a bare oxy/chalcogen suffix) are unaffected.
    if name_lower.startswith("benzyl"):
        rest = name_lower[len("benzyl"):]
        if rest == "oxy" or rest in _COMPOUND_S_SUFFIXES_COMPLEX:
            return True

    #: the alkoxycarbonyl / alkoxycarbonimidoyl compound-acyl
    # family is a SUBSTITUTED acyl prefix (an alkoxy on -CO-/-C(=NH)-) and takes
    # enclosing marks — '4-(methoxycarbonyl)benzoic acid',
    # '4-(methoxycarbonimidoyl)benzoic acid'. This mirrors composer's local
    # `_is_compound_prefix` `endswith('oxycarbonyl')` test and extends it to the
    # `carbonimidoyl` analogue so the imidate prefix is enclosed by
    # the SAME rule the ester prefix uses. The '(benzyloxy)carbon(imidoyl)' forms
    # already carry an inner mark and are enclosed by `enclose_if_compound`.
    if name_lower.endswith("oxycarbonyl") or name_lower.endswith("oxycarbonimidoyl"):
        return True

    #: a SUBSTITUTED alkyl whose substituent is itself a '-yl'
    # group — a cyclic-substituent-on-alkyl such as 'cyclohexylmethyl' (=
    # cyclohexyl + methyl) or 'phenylmethyl' — is a compound substituent and
    # takes enclosing marks. Without this the digit/hyphen/FG-prefix tests above
    # all miss it (a bare cyclic root carries no digit), so it shipped
    # UNENCLOSED: '3-(cyclohexylmethylamino)propanenitrile' reads ambiguously as
    # cyclohexyl + methylamino, where the PIN is '3-[(cyclohexylmethyl)amino]...'.
    # Detect it as <root ending in 'yl'> + <alkyl root>; the remainder must be a
    # genuine substituent ('yl'-terminated) and not itself a bare alkyl root
    # (guards against a non-name like 'ethylmethyl').
    for alkyl in _ALKYL_ROOTS_FULL:
        if name_lower.endswith(alkyl):
            remainder = name_lower[: -len(alkyl)]
            if (remainder and remainder.endswith("yl")
                    and remainder not in _ALKYL_ROOTS_FULL):
                return True

    #: the thiocyanato pseudohalide prefix (-S-C#N) is parenthesized
    # in PINs — BB '3-(thiocyanato)propanoic acid (PIN)':31013, 'S-ethyl
    # 3-(thiocyanato)propanethioate (PIN)':32041. Its N-anchored sibling
    # isothiocyanato is NOT enclosed (isothiocyanatobenzene, isothiocyanatosilane),
    # so match the bare word EXACTLY, never endswith (which would also catch
    # 'isothiocyanato').
    if name_lower == "thiocyanato":
        return True

    #: a substituent prefix that INCLUDES the name of a parent hydride
    # (an acyl-suffix-family prefix built on a ring or hydrazine parent) is
    # parenthesised so two parent hydrides are not read as one -- BB verbatim
    # '4-(cyclohexanecarbonyl)benzene-1-carbothioic acid (PIN)' (:7322),
    # '2-(hydrazinecarbonyl)benzene-1-sulfonic acid (PIN)' (:34012),
    # '3-(hydrazinecarbohydrazonoyl)benzoic acid (PIN)' (:34638). Gated on a
    # CURATED parent-hydride stem list (not a generic root match): the root
    # 'hydroxy' in 'hydroxycarbonimidoyl' is a simple substituent on the acyl
    # carbon, NOT a second parent hydride, and must stay bare (2 baseline MATCH
    # rows). A leading ring locant ('furan-2-carbonyl') is stripped before the
    # suffix check.
    for _stem in _ACYL_PARENT_HYDRIDE_STEMS:
        if name_lower.startswith(_stem):
            _rem = name_lower[len(_stem):].lstrip("-0123456789")
            if _rem in _ACYL_SUFFIX_FAMILY:
                return True

    return False


# ============================================================================
# IUPAC Enclosing Marks
# ============================================================================


def get_bracket_depth(name: str) -> int:
    """Determine the current bracket nesting depth of a name.

    Returns 0 if no brackets, 1 if contains , 2 if contains , etc.
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


# Pre-compiled patterns for compute_nesting_depth
#: Indicated hydrogen -- (1H), (3H), (9aH) etc.
_INDICATED_H_RE = re.compile(r'\(\d+[a-z]?H\)')
#: Fusion/spiro/ring-assembly/von Baeyer brackets -- [2,3-b],
# [4.5], [2.2.1], [1,1'-biphenyl] (ring-assembly enclosures carry primes).
# (the Blue Book) "the square brackets of ring fusion, spiro
# fusion, ring assembly, or the extended von Baeyer names are ignored": that
# includes descriptors with superscript locants, written '^n' ('[3.2.3^7.2^4]',
# '[3.2.1.0^2,4]'); 'tris(tetracyclo[3.2.0.0^2,7.0^4,6]heptane)' (:10824).
_FUSION_BRACKET_RE = re.compile(r"\[[0-9a-z,.'^\-]+\]")
# (the Blue Book) "the square brackets of... ring assembly
#... names are ignored", with (heading:7444) "the presence of square
# brackets and/or parentheses that are an integral part of the name of a parent
# structure does not affect the nesting order" (:7446). _FUSION_BRACKET_RE's
# character class cannot see three spellings of a ring-assembly enclosure the
# builders emit: the component parentheses
# ('[1,1'-bi(cyclohexan)]', also around a von Baeyer component
# '[2,2'-bi(bicyclo[2.2.1]heptan)]'), colon-separated locant sets
# ('[1,1':4',1''-terphenyl]') and replicated indicated hydrogen
# ('[1H,1'H-2,2'-biindol]'). Counting those marks put a ring-assembly prefix one
# or two levels too deep: '{[1,1'-bi(cyclohexan)]-4-yl}benzene', where the BB
# writes '4-[4-([1,1'-bi(cyclohexan)]-4-yl)phenyl]-...' (:17102) and
# '3,5-di([1,1':3',1''-terphenyl]-3-yl)pyridine' (:23917). The match is anchored
# on the assembly form itself -- a locant set of >= 2 locants, '-', a
# multiplying prefix (ASSEMBLY_MULTIPLIERS in rules/ring_assemblies.py) and a
# component that closes the bracket -- so a nesting bracket such as
# '[3,5-bis(trifluoromethyl)phenyl]' ('bis', a multiplier) never
# matches. (name_comparison's alphanumerical keys keep _FUSION_BRACKET_RE.)
_RA_LOCANT = r"\d+[a-z]?['′″]*"
_RING_ASSEMBLY_BRACKET_RE = re.compile(
    r"\[(?:" + _RA_LOCANT + r"H,)*(?:" + _RA_LOCANT + r"H-)?"
    + _RA_LOCANT + r"(?:[,:]" + _RA_LOCANT + r")+-"
    + r"(?:bi|ter|quater|quinque|sexi|septi|octi|novi|deci|undeci|dodeci)"
    + r"(?:[a-z]+|\((?:[^()\[\]{}]|\[[^\[\]{}]*\])+\))\]"
)
#: Stereo descriptors -- (R), (S), (E), (Z), (1R,2S), etc.
_STEREO_PAREN_RE = re.compile(r'\((?:\d+[a-z]?,)*[RSEZ](?:,\d+[a-z]?[RSEZ]?)*\)')


def compute_nesting_depth(name: str) -> int:
    """Compute effective bracket nesting depth per (Dec 2025).

    Analyzes a name string and returns the effective nesting depth by
    counting only nesting-relevant brackets, per the following subsections:

    -: Ignore indicated hydrogen parentheses, e.g., (1H), (3H)
    -: Ignore fusion/spiro/ring assembly/von Baeyer brackets
    -: Count stereo descriptor and compound locant parentheses
    -: Escalate if consecutive same-level marks would result
    -: Isotopic labeling convention (not applicable -- not implemented)

    Args:
        name: The chemical name string to analyze.

    Returns:
        Effective nesting depth (0 = no relevant brackets, 1 = has relevant
        parentheses, 2 = has relevant square brackets, etc.).
    """
    # Start with the original name and strip out non-nesting brackets
    # by replacing them with placeholder text that contains no brackets.
    working = name

    # Remove indicated hydrogen parentheses
    working = _INDICATED_H_RE.sub('__IH__', working)

    # Remove ring-assembly enclosures, with their integral component marks
    # +, then fusion/spiro/von Baeyer square brackets
    #
    working = _RING_ASSEMBLY_BRACKET_RE.sub('__RA__', working)
    working = _FUSION_BRACKET_RE.sub('__FB__', working)

    # Now measure how deep the remaining (nesting-relevant) marks actually nest.
    # (the Blue Book) cycles "{[({})]}": after { } comes
    #  again, then , so the depth is the NESTING depth, not "which mark
    # types occur". The old type test capped at 3, so a name already four levels
    # deep ('2-fluoro-4-({6-[methyl(prop-2-en-1-yl)amino]hexyl}oxy)phenyl') was
    # enclosed in '(' again -- '(' directly around '(' -- instead of '['.
    # For names up to three levels this equals the old type test (the marks are
    # added innermost-first in the ([ { order).
    #
    # TRIAGE j12 finding 8: the depth is counted in MARK LEVELS, not in raw
    # bracket levels. A group whose mark was escalated by (:7509,
    # "consecutive enclosing marks of the same level, the next level of enclosing
    # mark is used": '[(2S)-2-{...}propanoyl]' where '(' would stand next to
    # '(2S)') sits one mark level above its raw depth, and everything around it
    # must continue the order from its actual mark. Counting raw levels wrapped
    # '(2S)-2-[(2S)-2-{2-[(2R)-...]acetamido}propanamido]-3-methylbutanamido' in
    # '[' -- a bracket directly around a bracket -- instead of '{'. Each existing
    # group's level is the smallest level at or above its natural one (one above
    # its highest inner group) whose mark is the mark it actually carries.
    levels = _mark_levels(working)
    if levels is None:
        level = 0
        max_depth = 0
        for ch in working:
            if ch in '([{':
                level += 1
                max_depth = max(max_depth, level)
            elif ch in ')]}':
                level = max(0, level - 1)
        return max_depth
    return 0 if not levels else 1 + max(levels)


def _mark_levels(text: str):
    """The mark levels (0 = '(', 1 = '[', 2 = '{', 3 = '(' again,...)
    of the top-level enclosed groups of ``text``, or None when the marks are
    unbalanced."""
    opens = {'(': 0, '[': 1, '{': 2}
    marks = '([{'
    closes = {')': '(', ']': '[', '}': '{'}
    stack = []            # [open_char, max_child_level, first_content_char]
    top: list = []
    for i, ch in enumerate(text):
        if ch in opens:
            stack.append([ch, -1, text[i + 1] if i + 1 < len(text) else ''])
        elif ch in closes:
            if not stack or stack[-1][0] != closes[ch]:
                return None
            open_ch, child_max, first = stack.pop()
            lvl = child_max + 1
            # the one mark above the natural one is read as the
            # escalation only when it is what that rule produces (the content
            # starts with the natural mark); any other mismatch is an integral
            # bracket this scan does not recognise, counted at its natural level
            # as before
            if (opens[open_ch] != lvl % 3 and opens[open_ch] == (lvl + 1) % 3
                    and first == marks[lvl % 3]):
                lvl += 1
            if stack:
                stack[-1][1] = max(stack[-1][1], lvl)
            else:
                top.append(lvl)
    if stack:
        return None
    return top


def apply_enclosing_marks(name: str, depth: int = 0) -> str:
    """Apply the IUPAC enclosing marks at the nesting depth.

    Nesting order:  ->  -> { } ->  again
    Depth 0: parentheses
    Depth 1: square brackets (name already contains parentheses)
    Depth 2: braces (name already contains brackets)

    When depth=-1 (sentinel for auto-detect), calls compute_nesting_depth
    on the input name to determine the effective starting depth from the
    name's existing brackets per (Dec 2025 errata). This makes
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

    #: Check for consecutive same-level marks.
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
            # Check for fusion/spiro brackets: [2,3-b], [4.5], and a leading
            # ring-assembly enclosure: [1,1'-bi(cyclohexan)]
            if (_FUSION_BRACKET_RE.match(name)
                    or _RING_ASSEMBLY_BRACKET_RE.match(name)):
                leading_is_nesting = False

        if leading_is_nesting:
            depth += 1
            open_mark, close_mark = MARKS[depth % 3]

    return f"{open_mark}{name}{close_mark}"


def renest_group_and_ancestors(name: str, open_idx: int) -> str:
    """Re-derive the enclosing mark of the group opening at ``open_idx``
    and of every group that encloses it, innermost first.

    For a producer that SPLICES a substituent into an already-marked prefix: the
    spliced group gets deeper, and every enclosing group must move up the
    "{[({})]}" order (the Blue Book) with it. Re-marking only the
    spliced group left '3-[[3-(alpha-L-rhamnopyranosyloxy)decanoyl]oxy]' with a
    bracket directly inside a bracket. Each group's mark is what
    ``apply_enclosing_marks(content, -1)`` gives its content, the same rule every
    other producer uses, including the consecutive-mark step.
    Groups outside the chain are left untouched.
    """
    pairs = {'(': ')', '[': ']', '{': '}'}
    i = open_idx
    while i is not None and 0 <= i < len(name) and name[i] in pairs:
        depth = 0
        close = None
        for j in range(i, len(name)):
            if name[j] in '([{':
                depth += 1
            elif name[j] in ')]}':
                depth -= 1
                if depth == 0:
                    close = j
                    break
        if close is None:
            return name
        content = name[i + 1:close]
        marked = apply_enclosing_marks(content, -1)
        name = name[:i] + marked[0] + content + marked[-1] + name[close + 1:]
        # the enclosing opener of position i
        depth = 0
        parent = None
        for j in range(i - 1, -1, -1):
            if name[j] in ')]}':
                depth += 1
            elif name[j] in '([{':
                if depth == 0:
                    parent = j
                    break
                depth -= 1
        i = parent
    return name


def enclose_if_compound(name: str) -> str:
    """Enclose a substituent prefix in marks iff it is compound/complex.

    A SIMPLE prefix (``methyl``, ``phenyl``, ``cyclohexyl``, and the retained
    ``benzyl``) is cited bare — BB ``2-benzylpyridine`` (PIN),. A
    COMPOUND or COMPLEX prefix takes enclosing marks, escalating (-> [ -> {
    when the name already carries brackets — BB
    ``2-[(4-bromophenyl)methyl]pyridine`` (PIN), /.

    The compound test is the union of the two existing predicates because
    neither is complete on its own: ``needs_brackets`` catches a locant, a
    hyphen and a fused functional prefix (``bromomethyl``) but not a bare
    two-prefix compound; ``is_complex_substituent`` catches the two-prefix
    compound (``cyclohexylmethyl``) but not the fused functional prefix.

    A prefix that is NOT fully enclosed yet CONTAINS an inner enclosing mark is
    also compound and takes an OUTER (escalating) mark — ``cyclohexyl(methyl)amino``
    (an N,N-disubstituted amino core, -> ``[cyclohexyl(methyl)amino]``.
    Neither ``needs_brackets`` nor ``is_complex_substituent`` sees this when the
    core carries no locant or hyphen, so the inner-mark test is the third arm of
    the union. (The same gap was patched inline at one call site with
    ``'(' in name or '[' in name``; this centralises it — after the
    ``_is_fully_enclosed`` early return above, any residual ``([{`` is a genuine
    inner mark, and every leading-mark exception — indicated H ``(1H)``, stereo
    ``(R)-``, fusion ``[2,3-b]`` — already carries a digit and is caught by
    ``is_complex_substituent``, so this arm only newly fires on marks-without-locant.)

    Idempotent: an already fully-enclosed token is returned untouched.

    Examples:
        >>> enclose_if_compound("benzyl")
        'benzyl'
        >>> enclose_if_compound("cyclohexylmethyl")
        '(cyclohexylmethyl)'
        >>> enclose_if_compound("(4-bromophenyl)methyl")
        '[(4-bromophenyl)methyl]'
        >>> enclose_if_compound("(2-chloroethyl)")
        '(2-chloroethyl)'
        >>> enclose_if_compound("cyclohexyl(methyl)amino")
        '[cyclohexyl(methyl)amino]'
    """
    if not name or _is_fully_enclosed(name):
        return name
    # Lane L2: the writer's record decides the marks when it has decided them
    # (``assembly.prefix_derivation``;, the Blue Book)
    from .prefix_derivation import carried, derivation_of
    _rec = derivation_of(name)
    if _rec is not None and _rec.enclosed is not None:
        return carried(apply_enclosing_marks(name, -1), like=name) if _rec.enclosed else name
    if (needs_brackets(name) or is_complex_substituent(name)
            or any(mark in name for mark in '([{')):
        # Review F1 (properfix a performance pass): ``apply_enclosing_marks`` always
        # returns a plain ``str``, which used to drop the record of a
        # recorded ``name`` right here -- the single most-called enclosure
        # site in the engine (~150 callers). ``carried`` keeps it (the
        # mark itself carries no new fact; (c) is
        # about the CORE prefix, enclosed or not).
        return carried(apply_enclosing_marks(name, -1), like=name)
    # A compound prefix without a locant -- a substituent prefix in front of its
    # parent prefix, as is_substituted_substituent reads it ('silylamino',
    # 'cyanoamino', 'chlorophenyl', 'methylcarbamoyl', 'nitrosooxy') -- is enclosed
    # too: (the Blue Book) "Parentheses are used around compound
    #... and complex... prefixes"; '(silylamino)silyl' (:26322),
    # '4-(disilylamino)cyclohexane-1-carbonitrile (PIN)' (:38198). The contracted
    # amide prefixes ('acetamido', 'benzamido') are cited bare ('2-acetamido...',
    # '4-benzamido...' in the Blue Book) although they take 'bis'.
    if not name.lower().endswith('amido') and is_substituted_substituent(name):
        return carried(apply_enclosing_marks(name, -1), like=name)
    return name


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
    # (b)/(d) /: decide on the REMAINDER after an italicized structural
    # prefix, the way needs_brackets already does. Using the carve-out only as an
    # early-exit gate (which is what has_structural_hyphen below is) left every
    # suffix rule further down anchored to the RAW name, so the predicate
    # disagreed with itself: 'butylamino' was complex but 'tert-butylamino' was
    # not, because _ALKYLAMINO_RE is anchored with '^' and 'tert-' pushed the
    # alkyl root off the anchor. Recursing makes the whole function decide on
    # 'butylamino', so the italicized prefix changes nothing but the spelling.
    _remainder, _had_italicized = strip_italicized_structural_prefix(name)
    if _had_italicized:
        return is_complex_substituent(_remainder)
    # Check for hyphens (indicates compound substituent) — EXCEPT a leading
    # italicized 'sec-'/'tert-' detachable prefix on an otherwise-simple retained
    # name (tert-butyl, sec-butyl). IUPAC (b) (BB 7070, verbatim
    # `di-*tert*-butyl`) treats these as SIMPLE for multiplication (di-tert-butyl,
    # NOT bis(tert-butyl)) and BB 3465 cites the prefix bare straight after a
    # locant. (NOT, the parentheses rule -- verbatim at BB 7085,
    # "*Parentheses (round brackets)... are used to enclose multiplied components
    # that are: (a) simple substituent prefixes having locants*"; and
    # 'N-tert-butyl' is not a Blue Book example -- verified absent.)
    # a phase assembly/parenthesisation fix: without this,
    # coupling the paren/bis decision to is_complex_substituent over-parenthesised
    # tert-butyl. The leading-digit case above still catches genuine compounds.
    #: the strip is the SHARED primitive, not an inline copy.
    if has_structural_hyphen(name):
        return True
    # Check for embedded multiplier + substituent name patterns (IUPAC
    if _MULT_SUBSTITUENT_RE.match(name):
        return True
    # Compound oxy-prefixes require brackets (OPSIN/IUPAC parenthesization)
    if name in _COMPOUND_OXY_PREFIXES:
        return True
    # Compound sulfur prefixes per IUPAC
    name_lower = name.lower()
    for alkyl in _ALKYL_ROOTS_FULL:
        if name_lower.startswith(alkyl):
            remainder = name_lower[len(alkyl):]
            for s_suffix in _COMPOUND_S_SUFFIXES_COMPLEX:
                if remainder == s_suffix:
                    return True
    # DD2 Fix B (Phase D, (1)): (alkyl)peroxy is a compound substituent
    # requiring enclosing marks — '(methylperoxy)ethane', '(ethylperoxy)benzene'.
    # The simple 'peroxy' bridge and 'hydroperoxy' prefix are NOT compound.
    if name_lower.endswith('peroxy') and name_lower not in ('peroxy', 'hydroperoxy'):
        return True
    # Acyloxy compound prefixes per IUPAC:
    # "acetyloxy", "benzoyloxy", "propanoyloxy" etc. are compound prefixes
    # (acyl + oxy) that require complex multipliers (bis/tris) and parenthesization.
    if name.endswith('yloxy') and len(name) > 5:
        # Matches: acetyloxy, benzoyloxy, propanoyloxy, butanoyloxy, etc.
        # Does NOT match: methoxy, ethoxy (simple alkoxy, no 'yl' before 'oxy')
        return True
    # Haloalkyl compound substituents per IUPAC:
    # "fluoromethyl", "trifluoromethyl", "chloroethyl" etc.
    if _HALOALKYL_RE.match(name):
        return True
    # Compound FG-prefix + alkyl substituents per IUPAC: a simple
    # functional prefix fused to an alkyl-yl root is a SUBSTITUTED (compound)
    # substituent and takes enclosing marks — 'hydroxymethyl' -> '(hydroxymethyl)',
    # 'cyanomethyl' -> '(cyanomethyl)', 'carboxymethyl', 'aminoethyl', 'oxoethyl'.
    # BB PINs: '2-(hydroxymethyl)benzene-1,4-diol' (:6802),
    # '6-(hydroxymethyl)oxane-2,3,4-triol' (:2688), '(cyanomethyl)' (:33087),
    # '4-chloro-2-(hydroxymethyl)-5-oxohexyl' (:25293). This is the SAME set that
    # needs_brackets already recognises (~:961); is_complex_substituent was the
    # divergent predicate that missed it, so an N-substituent 'cyanomethyl' or a
    # ring sub-substituent 'hydroxymethyl' was cited BARE ('N-cyanomethylacetamide',
    # 'N-(4-hydroxymethylphenyl)acetamide'). The di/bis MULTIPLIER is decided
    # elsewhere (get_multiplier_prefix uses a graph test, a trace measured zero calls
    # here), so this promotes only the ENCLOSURE decision.
    for _fg in _COMPOUND_FG_PREFIXES:
        if name_lower.startswith(_fg) and name_lower[len(_fg):] in _ALKYL_ROOTS_FULL:
            return True
    # A substituent prefix on a retained contracted alkoxy, the Blue Book)
    # is a compound prefix and takes enclosing marks,:7232):
    # '1-(chloromethoxy)-4-nitrobenzene (PIN)' (:27711). Without this the bridged fused
    # and partially hydrogenated fused names and the cyclohexane prefixes cited
    # 'methoxymethoxy' and 'chloromethoxy' bare. The same test as needs_brackets.
    if is_substituted_contracted_alkoxy(name_lower):
        return True
    # Alkyl+amino compound substituents per IUPAC:
    # "methylamino", "ethylamino", "phenylamino" etc.
    if _ALKYLAMINO_RE.match(name):
        return True
    # Breadth job 1 (M03): an organyl on a diazenyl / hydrazinyl group is a compound
    # prefix and takes enclosing marks, the Blue Book):
    # '(phenyldiazenyl)', 'N'-[1-(tert-butyldiazenyl)cyclohexyl]' -- never
    # '1-tert-butyldiazenylcyclohexyl'. The bare 'diazenyl' / 'hydrazinyl' stay simple.
    if (name_lower.endswith(('yldiazenyl', 'ylhydrazinyl'))
            and name_lower not in ('diazenyl', 'hydrazinyl')):
        return True
    # Acylamino compound substituents: "ethanoylamino", "propanoylamino" etc.
    if name.endswith('amino') and 'oyl' in name:
        return True
    # S-amido compound substituents per IUPAC: a `<parent-hydride>sulfonamido`
    # / `sulfinamido` prefix (methanesulfonamido, benzenesulfonamido, ethanesulfonamido)
    # is a compound substituent (its stem is a sulfonyl/sulfinyl acyl) and is enclosed
    # -> the gold-oracle PIN `3-(methanesulfonamido)propanoic acid`,
    # `3-(benzenesulfonamido)propanoic acid`. The bare retained `sulfonamido`/`sulfinamido`
    # (no stem) and the acyl-amido family (`acetamido`, `benzamido`) stay SIMPLE.
    # #29: without this the heterocycle _format_c_substituent path emitted the
    # non-PIN bare `2-methanesulfonamido-1,3-thiazole-5-carboxylic acid`.
    if (name_lower.endswith(('sulfonamido', 'sulfinamido'))
            and name_lower not in ('sulfonamido', 'sulfinamido')):
        return True
    # W3-P10: a SUBSTITUTED pnictogen-yl prefix (dihydroxyarsanyl /
    # dimethylarsanyl /...stibanyl /...bismuthanyl) is a compound substituent
    # requiring enclosing marks -> '4-(dihydroxyarsanyl)benzoic acid'. The bare
    # parent-hydride-yl form ('arsanyl', 'stibanyl') stays simple.
    for _pnictyl in ('arsanyl', 'stibanyl', 'bismuthanyl'):
        if name.endswith(_pnictyl) and name != _pnictyl:
            return True
    # The same for phosphanyl, the Blue Book):
    # '3-(hydroxyphosphanyl)propanoic acid (PIN)' (:27272),
    # '4-[ethyl(methyl)phosphanyl]-1H-imidazole'. The bare 'phosphanyl' and the
    # catenated parent-hydride prefixes 'diphosphanyl' / 'triphosphanyl' stay
    # simple.
    if name_lower.endswith('phosphanyl'):
        _p_head = name_lower[:-len('phosphanyl')]
        if _p_head and _p_head not in ('di', 'tri', 'tetra', 'penta', 'hexa'):
            return True
    # A nitrogen-ylidene prefix carrying its own prefix is a compound prefix too:
    # '4-(phenylhydrazinylidene)cyclohexane-1-carboxylic acid (PIN)'
    # (the Blue Book), '4-[(dimethylcarbamoyl)hydrazinylidene]heptanoic acid
    # (PIN)' (:38678), 'methyl [(methylimino)silyl]acetate (PIN)' (:26578). The
    # bare 'imino' / 'hydrazinylidene' (and their multiplied 'diimino', (b)
    #:7072) stay simple.
    for _n_ylidene in ('imino', 'hydrazinylidene'):
        if name_lower.endswith(_n_ylidene):
            _ny_head = name_lower[:-len(_n_ylidene)]
            if _ny_head and _ny_head not in ('di', 'tri', 'tetra', 'penta', 'hexa'):
                return True
    # An N-substituted carbamoyl / carbamothioyl is a compound prefix:
    # '2-[(methylcarbamoyl)amino]naphthalene-1-carboxylic acid (PIN)' (:33354).
    for _carbamoyl in ('carbamoyl', 'carbamothioyl'):
        if name_lower.endswith(_carbamoyl) and name_lower != _carbamoyl:
            return True
    # (R-oxy)alkyl compound substituents per IUPAC / (
    # C- / V-3): an alkoxy/aryloxy unit ('<R>oxy') fused to a terminal alkyl-yl
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
    # a phase SUBST-01: compound (ring/substituent)-yl + alkyl-yl
    # ('cyclohexylmethyl', 'piperidinylmethyl') — enclosing marks per.
    if _RINGYL_ALKYL_RE.search(name_lower):
        return True
    # a phase: a Group-14 silyl/germyl substituent carrying its OWN
    # prefixes ('trihydroxysilyl', 'hydroxydimethylsilyl', 'aminodimethylsilyl') is
    # a compound prefix taking enclosing marks. The bare 'silyl'/'germyl' stems
    # stay simple; 'trimethylsilyl'/'triethylsilyl' are already caught by the
    # alkyl-multiplier regex above. (These prefixed silyl names are all new in
    # a phase, so this rule adds no regression to pre-existing substituents.)
    if (name_lower.endswith(('silyl', 'germyl'))
            and name_lower not in ('silyl', 'germyl')):
        return True
    return False


# ===========================================================================
# (c) / (a): is the prefix SUBSTITUTED? (the MULTIPLIER
# question, which is NOT the enclosure question above)
# ===========================================================================
#
# `is_complex_substituent` decides ENCLOSURE and its operative test
# is "contains a digit / hyphen / a known compound morpheme". Using it to pick
# the multiplier conflated two different rules and was wrong in BOTH
# directions:
#
# * `propan-2-yl`, `naphthalen-2-yl`, `bicyclo[3.2.1]octan-3-yl` are SIMPLE
# prefixes that merely carry a locant. (a) parenthesises them and
# (a) multiplies them with the SIMPLE `di` —
# `di(propanG2Gyl)!(preferred!prefix)` at the Blue Book,
# `tetra(naphthalenG2Gyl)` at:7089, `di(bicyclo[3.2.1]octanG3Gyl)` under
# clause (f), and `1,4-di(propan-2-yl)cyclohexane (PIN)` at:25721.
# The digit test made all of these `bis`.
# * `hydroxymethyl`, `carboxymethyl`, `oxomethyl`, `silylamino`,
# `phenyldiazenyl` are SUBSTITUTED prefixes with no digit anywhere, so the
# digit test missed them and emitted `di` where (a) requires `bis`.
#
# The Blue Book's own matched pair states the real rule in one molecule:
# `1,4-di(propan-2-yl)cyclohexane (PIN)` (:25721)
# `1,4-bis(2-chloropropan-2-yl)benzene (PIN)` (:25793)
# one `chloro` apart, and the multiplier flips. `**P"16.3.5**` (:7104) clause
# (a) names the criterion outright: "*compound or complex (i.e. SUBSTITUTED)
# prefixes*"; `### **P"16.3.2** General!methodology.` (:7031) clause (c): "*any
# component which is substituted automatically requires use of the
# multiplicative forms 'bis', 'tris', etc.*", against clause (a)'s
# "*UNSUBSTITUTED prefixes, such as ethyl or tert-butyl... are multiplied by
# the multiplicative prefixes 'di', 'tri', etc.*"
#
# WHAT THE GRAPH CAN AND CANNOT DECIDE. The rule is a property of the SKELETON
# ("does the prefix's own parent hydride bear a substituent?"), so the graph is
# the primary authority and `_fragment_bears_substituent` below implements it.
# But the graph alone is REFUTED by the Blue Book in two independent ways, and
# both refutations are load-bearing, so the graph is consulted only when a caller
# can supply it and the two name-level carve-outs are applied on top:
#
# 1. RETAINED SPELLINGS. `tert-butyl` is branched in the graph, yet
# (a) (`:7033`) lists it verbatim among the "*unsubstituted
# prefixes, such as ethyl or tert-butyl*" taking `di` — `di-*tert*-butyl`
# (`:7070`), `1,2-di-*tert*-butylbenzene (PIN)` (`:25717`). One connection
# table therefore has two valid names on OPPOSITE sides of the rule: the
# retained `tert-butyl` takes `di-`, its systematic synonym
# `2-methylpropan-2-yl` takes `bis`. `benzyl` is the same case
# (`### **** Retained prefixes that are preferred prefixes`,
# `:16272`; `dibenzyl` inside a PIN at `:23258`) and is even simpler,
# because `:24420` forbids substituting it at all, so (a) can never
# fire on the spelling.
# 2. WHICH ATOM IS THE PARENT. The same O-CH3 fragment is `methoxy` (a simple
# retained contracted prefix, `di`) when the oxygen is the attachment, and
# the `methoxy` half of `methoxymethyl` (compound, `bis`) when a carbon is.
# Nothing in the connection table of the WHOLE prefix says which parse the
# producer chose.
#
# So: graph when available, and a WHOLE-TOKEN vocabulary reconstruction of the
# producer's own derivation otherwise. The reconstruction differs from the
# morpheme-peel it replaces in the two ways that made the peel wrong in both
# directions:
#
# * a split requires BOTH sides to be RECOGNISED units. The old tail test
# accepted any remainder "ending in a group-shaped letter", so `nitroso`
# split as `nitro` + `so` (bare `'o'` was an accepted ending) and
# `oxolan-2-yl` split as `oxo` + `lan-2-yl`. Both shipped a wrong `bis`.
# * the vocabulary is DERIVED FROM THE PRODUCERS' OWN TABLES rather than
# hand-listed, so it cannot be wrong on its complement the way the closed
# hand-list was: `methoxymethyl` was simply absent from it and silently read
# as simple.

# --- vocabulary 1: characteristic-group prefixes, straight from the producer ---
# `rules/seniority.PREFIX_FORMS` is the table the namer itself uses to spell a
# characteristic group as a prefix. Cycle-free: `rules/seniority` imports
# nothing from `orthonym`, and `rules/__init__` imports only from it.
#
# `**P"16.3.3**`(b) (`:7067`, continuing at `:7094`) puts these on the BASIC
# multiplier: "*simple substituent prefixes, including parent hydrides with 'ene'
# and 'yne' endings (without locants), and characteristic groups*" — verbatim
# `diimino` (`:7072`), `dibromo` (`:7073`), and `1,4-dinitrosobenzene (PIN)`
# under `## **** Nitro and nitroso compounds` (`:25943`).
#
# Some entries in the table are themselves CONCATENATED compound prefixes
# (`phosphonooxy`, `carbamoyloxy`, `nitrooxy`, `aminooxy`, `sulfanyloxy`,
# `carbamoylamino`, `hydroxyimino`, `sulfanylcarbonyl`,...). They are NOT
# filtered by hand: each one decomposes into two units of this same vocabulary,
# so the split test below classifies them correctly on its own.
from ..rules.seniority import PREFIX_FORMS as _SENIORITY_PREFIX_FORMS  # noqa: E402

_FG_PREFIX_SPELLINGS: frozenset = frozenset(
    v.lower() for v in _SENIORITY_PREFIX_FORMS.values() if isinstance(v, str) and v
)

# --- vocabulary 2: the SIX retained contracted alkoxy prefixes ------------
# `### **** Retained names` (`:27665`), verbatim at `:27667`: "*Some
# contracted names are retained for R-O– substituent groups... and are
# considered as simple prefixes requiring the numerical prefixes 'di', 'tri',
# etc. They are:*" — and the list that follows is exactly these six, each
# "(preferred prefix)". Confirmed in a PIN: `1,2-dimethoxybenzene (PIN)`
# (`:27705`), `1,1-dimethoxypropane (PIN)` (`:5098`).
#
# The list is CLOSED. Every OTHER `<R>oxy` prefix is formed by concatenation and
# takes the DERIVED multiplier — `****` (`:27633`), verbatim:
# "*Substituent prefix names for R′-O– groups are formed by concatenation, i.e.,
# by adding the prefix 'oxy' to the substituent prefix name for the group R′.
# These compound prefixes require the numerical multiplying prefixes 'bis',
# 'tris', etc.*", first example `pentyloxy (preferred prefix)` (`:27637`).
#
# So a predicate that keys on the SHAPE "ends in -oxy" is wrong whichever answer
# it gives; only membership decides. `tert-butoxy` reaches this set through the
# italicized-prefix carve-out, which strips `tert-` first.
_RETAINED_CONTRACTED_ALKOXY: frozenset = frozenset({
    'methoxy', 'ethoxy', 'propoxy', 'butoxy', 'phenoxy', 'butoxy',
})

# --- vocabulary 3: retained hydrocarbyl and divalent prefixes -------------
# `### **** Retained prefixes that are preferred prefixes` (`:16270`),
# `:16272`: "*The traditional prefixes benzyl, benzylidene, benzylidyne are
# retained preferred prefixes, but are not to be substituted*"; `:24414`
# `C6H5-CH2– benzyl (preferred prefix)`. `dibenzyl` appears inside a PIN at
# `:23258`, `dibenzylphosphinite (PIN)` at `:40975`.
#
# `methylene`/`ethylene` are here because they must NOT be classed substituted:
# their `bis` comes from the separate (b) ambiguity rule (`:7162`
# `bis(methylene)` "*whereas dimethylene might be used to define the –CH2-CH2–
# group*"), which is `CATENATION_AMBIGUOUS_PREFIXES`, not substitution.
_RETAINED_HYDROCARBYL: frozenset = frozenset({
    'phenyl', 'benzyl', 'benzylidene', 'benzylidyne', 'trityl',
    'tolyl', 'xylyl', 'naphthyl', 'mesityl', 'styryl', 'cinnamyl',
    'vinyl', 'allyl', 'phenylene', 'methylene', 'ethylene', 'propylene',
    'methylidene', 'ethylidene', 'methylidyne', 'ethylidyne',
    'carbonyl', 'oxalyl', 'malonyl', 'succinyl', 'glutaryl', 'adipoyl',
    'phthaloyl', 'isophthaloyl', 'terephthaloyl',
})

# --- vocabulary 4: mononuclear hydride -yl / -ylidene carriers ------------
# The `-yl` forms of the preselected mononuclear parent hydrides /
#. Bare, these are simple prefixes; several additionally take `bis`
# via (a) (`:7140` `bis(sulfanyl) (preferred prefix)` "*whereas
# disulfanyl defines the –SSH group*"), which is the separate ambiguity leg.
_HYDRIDE_YL_CARRIERS: frozenset = frozenset({
    'silyl', 'germyl', 'stannyl', 'plumbyl', 'boryl', 'alumanyl',
    'oxidanyl', 'sulfanyl', 'selanyl', 'tellanyl', 'polonanyl',
    'azanyl', 'phosphanyl', 'arsanyl', 'stibanyl', 'bismuthanyl',
    'azanylidene', 'sulfanylidene', 'selanylidene', 'tellanylidene',
    'diazenyl', 'sulfinyl', 'sulfonyl', 'phosphoryl', 'phosphanylidene',
    'oxy', 'thio', 'seleno', 'telluro', 'imino', 'amino', 'hydrazino',
    'silylidene', 'germylidene', 'stannylidene',
})

# --- vocabulary 5: divalent BRIDGE prefixes ---------------------
# `di(metheno)` appears inside a PIN: `12,19:13,18-di(metheno)dinaphtho[...]
# pentaphene (PIN)` (`:14527`, `:23818`).
_BRIDGE_PREFIXES: frozenset = frozenset({
    'metheno', 'etheno', 'ethano', 'propano', 'butano', 'benzeno',
    'epoxy', 'epithio', 'epimino', 'episeleno', 'epitelluro',
    'nitrilo', 'furano', 'naphtho', 'phospho',
})

# The union: a WHOLE token that is one of these is ONE component, hence simple.
_SIMPLE_PREFIX_VOCABULARY: frozenset = (
    frozenset(_ALKYL_ROOTS_FULL)
    | frozenset(f'cyclo{p}yl' for p in ('prop', 'but', 'pent', 'hex', 'hept',
                                        'oct', 'non', 'dec'))
    | _FG_PREFIX_SPELLINGS
    | _RETAINED_CONTRACTED_ALKOXY
    | _RETAINED_HYDROCARBYL
    | _HYDRIDE_YL_CARRIERS
    | _BRIDGE_PREFIXES
)

# The units that may be DETACHED from the FRONT of a compound prefix. This is
# the same vocabulary: a compound prefix is built by concatenating units, so any
# unit can lead. Sorted longest-first so `carbamoyloxy` peels `carbamoyl`, not
# a shorter accidental match.
_FRONT_PREFIX_UNITS: Tuple[str, ...] = tuple(
    sorted(_SIMPLE_PREFIX_VOCABULARY, key=len, reverse=True)
)

# --- vocabulary 6: parent-hydride STEMS ----------------------------------
# Used ONLY as a negative filter: a stem that splits into
# `<front unit> + <recognised stem>` is substituted (`chloropropan-2-yl`), and
# one that does not is a bare hydride (`oxolan-2-yl` -> `oxo` + `lan`, and `lan`
# is no stem). Incompleteness here is fail-soft toward "simple", i.e. toward the
# pre-existing behaviour, never toward a newly-wrong `bis`.
_CHAIN_STEM_ROOTS: Tuple[str, ...] = (
    'meth', 'eth', 'prop', 'but', 'pent', 'hex', 'hept', 'oct', 'non',
    'dec', 'undec', 'dodec', 'tridec', 'tetradec', 'pentadec', 'hexadec',
    'heptadec', 'octadec', 'nonadec', 'icos',
)


def _build_hydride_stems() -> frozenset:
    """Parent-hydride stems, derived from the producers' own name tables."""
    stems = set()
    for root in _CHAIN_STEM_ROOTS:
        stems.add(root)
        for tail in ('an', 'en', 'yn', 'ane', 'ene', 'yne'):
            stems.add(root + tail)
        stems.add('cyclo' + root)
        for tail in ('an', 'en', 'ane', 'ene'):
            stems.add('cyclo' + root + tail)
    # Hantzsch-Widman ring stems, from the producers' own two tables: the
    # heteroatom prefix (`oxa`, `thia`, `aza`,...) loses its terminal 'a' before
    # a vowel-initial HW stem, giving `oxolan`, `oxan`, `oxocan`, `thiolan`,
    # `azolidin`, and the multiplied forms `dioxolan`, `trioxan`. Needed because
    # a stem is the NEGATIVE filter on a split: without `oxolan` here,
    # `oxolan-2-ylmethyl` cannot be recognised as ring-yl + alkyl.
    try:
        from ..data.hw_heteroatoms import HW_PREFIXES
        from ..data.hw_stems import HW_STEMS
        hetero = sorted({p.lower() for p in HW_PREFIXES.values() if p})
        hw_tails = set()
        for entry in HW_STEMS.values():
            if isinstance(entry, dict):
                hw_tails.update(str(v).lower() for v in entry.values() if v)
            elif entry:
                hw_tails.add(str(entry).lower())
        for pre in hetero:
            base = pre[:-1] if pre.endswith('a') else pre
            for tail in hw_tails:
                if not tail:
                    continue
                joined = (base if tail[0] in 'aeiouy' else pre) + tail
                stems.add(joined)
                if joined.endswith('e'):
                    stems.add(joined[:-1])
                for mult in ('di', 'tri', 'tetra'):
                    stems.add(mult + joined)
                    if joined.endswith('e'):
                        stems.add(mult + joined[:-1])
    except Exception:  # pragma: no cover - tables are always importable in-tree
        pass
    try:
        from ..data.fused_heterocycles import FUSED_HETEROCYCLE_PREFIX_STEMS
        for value in FUSED_HETEROCYCLE_PREFIX_STEMS.values():
            if isinstance(value, str) and value:
                stems.add(value.strip().lower())
    except Exception:  # pragma: no cover
        pass
    try:
        from ..data.retained_names import RETAINED_NAMES
        for value in RETAINED_NAMES.values():
            if not isinstance(value, str):
                continue
            token = value.strip().lower()
            if not token or not token.replace('-', '').isalpha():
                continue
            stems.add(token)
            # the substituent-prefix stem drops a trailing 'e' ('benzene' ->
            # 'benzen-1-yl', 'naphthalene' -> 'naphthalen-2-yl')
            if token.endswith('e'):
                stems.add(token[:-1])
    except Exception:  # pragma: no cover - table is always importable in-tree
        pass
    return frozenset(stems)


_HYDRIDE_STEMS: frozenset = _build_hydride_stems()

# Yl-type endings that terminate a parent-hydride-derived substituent prefix.
_YL_TYPE_ENDINGS: Tuple[str, ...] = (
    'ylidenes', 'ylidene', 'ylidyne', 'ylium', 'ylidyn', 'yliden',
    'ylo', 'yl', 'ide', 'uide', 'ium', 'ylia',
)

# `<stem>` then any number of `-<locants>-<infix>` groups then `-<locants>-<yl>`:
# `propan-2-yl`, `prop-1-en-2-yl`, `bicyclo[3.2.1]octan-3-yl`,
# `[1,2,4]triazolo[1,5-a]pyrimidin-2-yl`, `1H-imidazol-1-yl`.
_HYDRIDE_YL_SHAPE_RE = re.compile(
    r"^(?P<head>[^-]*?[a-z])"          # stem, possibly bracket-bearing
    r"(?:-\d[\d,'′]*-[a-z]{2,7})*"  # -1-en- / -2-yn- unsaturation infixes
    r"-\d[\d,'′]*-"                # the attachment locant set
    r"(?P<end>[a-z]+)$"                 # the yl-type ending
)

# A multiplied-unsaturation infix (`-1,3-dien`, `-2,4,6-trien`, `-1,3-diyn`) — the
# marker that a euphonic 'a' was inserted after the chain root. Used by
# `_is_bare_hydride_yl` to confirm a `<root>a` head is genuinely a multiplied-ene
# stem before dropping the 'a', never a coincidental trailing 'a'.
_MULTIPLIED_UNSAT_MARKER_RE = re.compile(r"-\d[\d,'′]*-(?:di|tri|tetra|penta)(?:en|yn)")

# Bracketed descriptors that belong to a STEM, not to a substituent: fusion
# locants `[1,5-a]`, von Baeyer/spiro descriptors `[3.2.1]`, isotope descriptors
# `[4-2H]`, and leading heteroatom-position sets `[1,2,4]`. clause (f)
# is verbatim that these keep the BASIC multiplier: `di(bicyclo[3.2.1]octanG3Gyl)
# !(preferred!prefix)`, `di([4G2H]benzoyl)!(preferred!prefix)`,
# `8,8′Goxydi(spiro[4.5]decane)!(PIN)`.
_STEM_BRACKET_RE = re.compile(r"\[[^\[\]]*\]")
# A leading indicated-hydrogen descriptor: `1H-`, `2H-`, `4H-`.
# NB: distinct from the module-level ``_INDICATED_H_RE`` (the ADDED-indicated-H
# PAREN form ``(2H)`` used by compute_nesting_depth / apply_enclosing_marks).
# Renamed to end a name-collision: this leading-prefix pattern used to shadow the
# paren form as ``_INDICATED_H_RE``, so compute_nesting_depth silently counted
# ``(2H)`` as a nesting level and escalated `(...)` enclosures to `[...]`
#: added-indicated-H parentheses are IGNORED).
_LEADING_IH_PREFIX_RE = re.compile(r"^\d+[hH]-")

# clause (a) (the Blue Book): "*The basic numerical prefixes 'di',
# 'tri', 'tetra', etc. are used to indicate a multiplicity of: (a) functional and
# cumulative suffixes*" -- `diol`, `dicarboxylic acid`, `disulfonic acid`.
#
# `get_multiplier_prefix` is called for SUFFIXES as well as for substituent
# prefixes, and a suffix word is NEVER decomposed for substituted-ness. Without
# this guard the morpheme scan read `carboxylic acid` as `carboxy` + `lic acid`
# and `sulfonic acid` as `sulfo` + `nic acid`, emitting `benzene-1,4-bis-
# carboxylic acid` and `4-methylbenzene-1,3-bissulfonic acid` -- 10 gold
# regressions, caught by the pre-commit flip measurement.
_FUNCTIONAL_SUFFIX_TAILS = (
    'ic acid', 'oic acid', 'carboxylic acid', 'sulfonic acid', 'sulfinic acid',
    'phosphonic acid', 'carbothioic acid', 'peroxoic acid',
    'carboxamide', 'sulfonamide', 'amide', 'carbonitrile', 'nitrile',
    'carbaldehyde', 'carboxylate', 'sulfonate', 'oate', 'carboxy',
)

# clause (b) (`:7104`), verbatim: "*'thioic acid' and 'dithioic acid'
# suffixes, and their Se and Te analogues, as exceptions to suffixes described in
# and *" -- `bis(thioic!acid)![multiple!preferred!suffix...
# whereas!dithioic!acid!describes!a!–CSSH!suffix]` (`:7106`) and
# `bis(dithioic!acid)... not!didithoic!acid` (`:7108`).
#
# ⚠ MATCHED AS A WHOLE TOKEN, never with `endswith`. `'carbodithioic
# acid'.endswith('dithioic acid')` is True, and the Blue Book puts the two on
# OPPOSITE sides four lines apart, both PINs, under `### ****
# Functional replacement in systematic names of carboxylic acids`:
# `benzene-1,2-dicarbodithioic acid (PIN) (not tetrathiophthalic acid)`:30313
# `ethanebis(dithioic acid) (PIN) (not tetrathiooxalic acid)`:30317
# The `endswith` form shipped `cyclohexane-1,2-biscarbodithioic acid`.
_BIS_SUFFIX_EXCEPTION_TOKENS: frozenset = frozenset({
    'thioic acid', 'dithioic acid',
    'selenoic acid', 'diselenoic acid',
    'telluroic acid', 'ditelluroic acid',
})

# Backwards-compatible alias: several sibling modules and tests import this name.
_BIS_SUFFIX_EXCEPTIONS = tuple(sorted(_BIS_SUFFIX_EXCEPTION_TOKENS))

# `### ****` (`:30215`) spells the same suffixes with an ITALIC
# CHALCOGEN LOCANT (`thioic O-acid`, `dithioic S-acid`, `selenoic Se-acid`) and
# notes the locants are "normally... omitted". Keying the exception on the
# locant-free spelling ALONE made the whole `<alkane>bis(thioic O-acid)` family
# abstain: `OC(=S)CCC(O)=S` fell from `butanebisthioic O-acid` to
# `unknown organic compound`, and the candidate the build then offered was
# `butanedithioic S-acid` -- a single -CSSH, which is precisely the collision
# (b) exists to prevent. So the infix is stripped before the lookup.
_CHALCOGEN_LOCANT_INFIX_RE = re.compile(
    r"\b(?:Se|Te|As|[OSNP])(?:'|′)*-(?=acid\b)", re.IGNORECASE)


def _normalise_suffix_token(suffix: str) -> str:
    """Lower-case a suffix token and drop any italic chalcogen locant infix.

    ``'thioic O-acid'`` -> ``'thioic acid'``; ``'dithioic S-acid'`` ->
    ``'dithioic acid'``. Leaves everything else untouched.
    """
    token = ' '.join(suffix.strip().split())
    token = _CHALCOGEN_LOCANT_INFIX_RE.sub('', token)
    return ' '.join(token.split()).lower()


def suffix_takes_derived_multiplier(suffix_name: str) -> bool:
    """(a) vs (b): does this SUFFIX take bis/tris, not di/tri?

    `**P"16.3.3**` (`:7038`): "*The basic numerical prefixes 'di', 'tri',
    'tetra', etc. are used to indicate a multiplicity of: (a) functional and
    cumulative suffixes, basic or modified by functional replacement, **with the
    exception of 'thioic acid' and 'dithioic acid' described in (b)***".

    So the answer is False for every suffix except that one closed family. A
    suffix is NEVER decomposed for substituted-ness -- doing so read
    `carboxylic acid` as `carboxy` + `lic acid` and broke 10 gold rows.

    Examples:
        >>> suffix_takes_derived_multiplier("carboxylic acid")
        False
        >>> suffix_takes_derived_multiplier("carbodithioic acid")
        False
        >>> suffix_takes_derived_multiplier("dithioic acid")
        True
        >>> suffix_takes_derived_multiplier("dithioic O-acid")
        True
    """
    if not suffix_name:
        return False
    return _normalise_suffix_token(suffix_name) in _BIS_SUFFIX_EXCEPTION_TOKENS


def get_suffix_multiplier_prefix(count: int, suffix_name: str) -> str:
    """The multiplier for a FUNCTIONAL or CUMULATIVE SUFFIX (a)).

    A separate entry point from:func:`get_multiplier_prefix` because the two
    callers are asking different questions and the substituent answer is not the
    suffix answer. Sharing one function meant a suffix could be handed to a
    substituent decomposition -- the mechanism behind
    `benzene-1,4-biscarboxylic acid`.
    """
    if count <= 1:
        return ""
    if suffix_takes_derived_multiplier(suffix_name):
        if count in COMPLEX_MULTIPLIERS:
            return COMPLEX_MULTIPLIERS[count]
        from ..data.chain_names import get_chain_prefix
        prefix = get_chain_prefix(count)
        if not prefix.endswith("a"):
            prefix += "a"
        return prefix + "kis"
    word = simple_multiplier_word(count)
    if word is None:
        raise ValueError(
            f"no simple multiplying prefix can be formed for {count}")
    return word


# A leading locant set on a substituent prefix: `2-`, `2,2-`, `1,3-`, `2'-`, and
# -- widened in -FINAL -- the ITALIC LETTER locants `N-`, `N,N'-`, `O-`,
# `S-`, `N2-`. Without the letter forms, `N-pentylcarbamoyl` never had its
# locant peeled, so the prefix unit behind it was unreachable and the whole
# `N-`-substituted-carbamoyl class read as simple.
_LEADING_LOCANT_RE = re.compile(
    r"^(?:\d[\d,'′]*|[NOSP](?:\d+)?(?:['′])*(?:,[NOSP](?:\d+)?(?:['′])*)*)-",
    re.IGNORECASE)

# Multiplicative syllables that may precede a repeated substituent morpheme
# (`trimethylsilyl`, `difluoromethyl`). Only ever peeled when a real morpheme
# follows, so a chain root that merely begins with one (`tridecyl`) is safe.
_LEADING_MULTIPLIER_SYLLABLES = (
    'tetrakis', 'pentakis', 'hexakis', 'tris', 'bis',
    'tetra', 'penta', 'hexa', 'hepta', 'octa', 'nona', 'deca',
    'di', 'tri',
)


def _strip_stem_decorations(stem: str) -> str:
    """Reduce a hydride stem to its bare alphabetic run.

    Drops bracketed fusion/von-Baeyer/isotope descriptors, a leading
    indicated-hydrogen descriptor and any leading locant set, so
    ``[1,2,4]triazolo[1,5-a]pyrimidin`` -> ``triazolopyrimidin`` and
    ``1H-imidazol`` -> ``imidazol``.
    """
    out = _STEM_BRACKET_RE.sub('', stem)
    out = _LEADING_IH_PREFIX_RE.sub('', out)
    m = _LEADING_LOCANT_RE.match(out)
    if m:
        out = out[m.end():]
    return out


def _stem_splits_into_prefix_plus_stem(stem: str) -> bool:
    """Does an alphabetic hydride stem decompose as `<prefix unit><stem>`?

    ``chloropropan`` -> ``chloro`` + ``propan`` (a recognised stem) -> True, so
    the prefix is substituted. ``oxolan`` -> ``oxo`` + ``lan``, and ``lan`` is
    no stem, so False and the prefix stays simple. This is the exact test the
    old code lacked: it accepted any remainder "ending in a group-shaped letter",
    which `lan-2-yl` satisfies.
    """
    for mult in ('',) + _LEADING_MULTIPLIER_SYLLABLES:
        if mult and not stem.startswith(mult):
            continue
        rest = stem[len(mult):]
        if not rest:
            continue
        for unit in _FRONT_PREFIX_UNITS:
            if not rest.startswith(unit) or len(rest) == len(unit):
                continue
            tail = rest[len(unit):].lstrip('-')
            if tail in _HYDRIDE_STEMS or tail in _SIMPLE_PREFIX_VOCABULARY:
                return True
            # Euphonic-'a' multiplied-unsaturation stem root (`hydroxypenta` ->
            # `hydroxy` + `penta`, where `penta` = `pent` + euphonic 'a' before a
            # `dien`/`triyn` ending the flat stem table does not carry). Symmetric
            # with the same accept in `_is_bare_hydride_yl`; only fires on the
            # euphonic shape the polyene substituent newly emits (a review-polyene
            # BLOCKER 1). Additive -- no front-prefix + `<root>a` head occurs in
            # any pre-existing emitted name (gold-diff 0/1888).
            if tail.endswith('a') and tail[:-1] in _CHAIN_STEM_ROOTS:
                return True
    return False


def _is_bare_hydride_yl(token: str, require_known_stem: bool = False) -> bool:
    """Is ``token`` an UNSUBSTITUTED parent hydride carrying only its own
    attachment (and unsaturation) locants?

    These are simple by (a) and parenthesised by (a) -- two
    different rules, which is the whole point:
    `di(propanG2Gyl)!(preferred!prefix)` (`:7087`),
    `tetra(naphthalenG2Gyl)` (`:7090`), and clause (f)
    `di(bicyclo[3.2.1]octanG3Gyl)`. Verbatim PIN witnesses for the ring case:
    `1,2-di(furan-2-yl)-2-hydroxyethan-1-one (PIN)` (`:29677`),
    `di(1*H*-imidazol-1-yl)methanethione (PIN)` (`:29544`),
    `di(naphthalen-2-yl)ethanedione (PIN)` (`:28380`),
    `2,6-di(tetraphen-1-yl)pyridine (PIN)` (`:25762`).
    """
    m = _HYDRIDE_YL_SHAPE_RE.match(token)
    if m is None:
        return False
    if m.group('end') not in _YL_TYPE_ENDINGS:
        return False
    stem = _strip_stem_decorations(m.group('head'))
    if not stem or not stem.isalpha():
        return False
    if require_known_stem:
        # STRICTER role: as one HALF of a split, the stem must be a stem we
        # actually know. Without this asymmetry `oxolan-2-yl` splits into `oxo`
        # + `lan-2-yl`, because `lan` is alphabetic and does not itself split --
        # exactly the I2 defect, reintroduced from the other side. Being strict
        # here makes an incomplete stem table fail toward "SIMPLE", i.e. toward
        # the pre-existing behaviour, never toward a newly-wrong `bis`.
        if stem in _HYDRIDE_STEMS:
            return True
        # Euphonic-'a' multiplied-unsaturation stem: `penta`(-1,3-dien),
        # `hexa`(-2,4-dien), `buta`(-1,3-diyn). The 'a' is inserted before a
        # consonant-initial multiplied 'diene'/'triyne' ending, so the flat table
        # (built from `pent`/`penten`/... in _build_hydride_stems) lacks it.
        # Accept when dropping the euphonic 'a' yields a known chain-stem root AND
        # the ORIGINAL token actually carries the `-<locs>-di|tri|tetraen/yn`
        # marker that produced the 'a' -- so this only ever fires on the shape the
        # polyene substituent newly emits (a review-polyene BLOCKER 1: a
        # SUBSTITUTED polyene prefix must take `bis`, not `di`, per (a);
        # the UNSUBSTITUTED form stays `di` via the loose/_is_single_simple_unit
        # path and (b)). Additive by construction.
        if (stem.endswith('a') and stem[:-1] in _CHAIN_STEM_ROOTS
                and _MULTIPLIED_UNSAT_MARKER_RE.search(token)):
            return True
        return False
    return not _stem_splits_into_prefix_plus_stem(stem)


def _is_single_simple_unit(token: str) -> bool:
    """Is ``token`` ONE simple nomenclatural component (a))?"""
    if not token:
        return False
    if token in _SIMPLE_PREFIX_VOCABULARY:
        return True
    return _is_bare_hydride_yl(token)


def _enclosed_leader_is_a_substituent(work: str) -> bool:
    """Does ``work`` open with an enclosed SUBSTITUENT followed by more name?

    `(4-methylphenyl)methyl` -> yes, a detached prefix by construction.
    `[1,2,4]triazolo[...]pyrimidin-2-yl` -> NO: `[1,2,4]` is a locant list, and
    (f) is verbatim that bracket-bearing simple components keep the BASIC
    multiplier (`di([4G2H]benzoyl)!(preferred!prefix)`). Reading a leading
    fusion/isotope bracket as a substituent is what made every bracket-leading
    fused heterocyclyl take `bis`.
    """
    if work[:1] not in '([{':
        return False
    pairs = {'(': ')', '[': ']', '{': '}'}
    opener = work[0]
    closer = pairs[opener]
    depth = 0
    for i, ch in enumerate(work):
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                inner = work[1:i]
                trailing = work[i + 1:]
                if not trailing:
                    return False
                # A descriptor, not a substituent: only digits, commas, dots,
                # primes, hyphens and a bare element/italic letter (`1,2,4`,
                # `3.2.1`, `1,5-a`, `4-2H`). Anything with a real name in it
                # (`4-methylphenyl`) is a detached prefix.
                return _enclosure_names_a_group(inner)
    return False


def is_substituted_substituent(
    name: Optional[str],
    *,
    mol=None,
    atoms=None,
    attachment: Optional[int] = None,
) -> bool:
    """Is this substituent prefix SUBSTITUTED, per (c) / (a)?

    ``True`` -> the derived multipliers ``bis``/``tris``/``tetrakis``.
    ``False`` -> the basic multipliers ``di``/``tri``/``tetra``.

    This answers ONLY the multiplier question. Enclosing marks are decided
    separately by:func:`is_complex_substituent` /:func:`needs_brackets`
    , and the two genuinely disagree: ``propan-2-yl`` is enclosed AND
    simply multiplied — ``di(propan-2-yl)``.

    A prefix is substituted when a DETACHABLE substituent prefix sits in front
    of its own parent core. A prefix that is nothing but an (unsubstituted)
    parent hydride with attachment/unsaturation locants is simple, however many
    digits it carries.

    Examples:
        >>> is_substituted_substituent("propan-2-yl")
        False
        >>> is_substituted_substituent("naphthalen-2-yl")
        False
        >>> is_substituted_substituent("tridecyl")
        False
        >>> is_substituted_substituent("hydroxymethyl")
        True
        >>> is_substituted_substituent("2-methylbutyl")
        True
        >>> is_substituted_substituent("bromomethyl")
        True
    """
    if not name:
        return False
    # Lane L2 (R2): the writer that built the prefix recorded whether it attached a
    # substituent prefix (``assembly.prefix_derivation``), (c)
    # (the Blue Book); that record decides. A name without one is read below.
    from .prefix_derivation import derivation_of
    _rec = derivation_of(name)
    if _rec is not None:
        return _rec.substituted

    # (a) names `tert-butyl` among the UNSUBSTITUTED prefixes, and
    # `sec-` behaves identically. Decide on the remainder, via THE shared
    # primitive, so this leg cannot drift from the enclosure legs. This runs
    # FIRST, before the graph, because it is precisely the case where the
    # SPELLING overrides the connection table.
    remainder, had_italicized = strip_italicized_structural_prefix(name)
    if had_italicized:
        # `tert-butyl` -> `butyl`, simple. `tert-butylsulfanyl` ->
        # `butylsulfanyl`, still a compound prefix.
        return is_substituted_substituent(remainder, mol=mol, atoms=atoms,
                                          attachment=attachment)

    work = name.strip()

    # A fully enclosed token is a single component: decide on its interior
    # (`(2-chloroethyl)` is substituted, `(propan-2-yl)` is not).
    if _is_fully_enclosed(work):
        return is_substituted_substituent(work[1:-1], mol=mol, atoms=atoms,
                                          attachment=attachment)

    # A leading stereodescriptor block ('(1s,4s)-', '(2R)-') is not a detachable
    # substituent: it describes the prefix it precedes, so the multiplier
    # is decided on that prefix. Without this peel every stereo-bearing prefix
    # read as simple and a substituted one took 'di': '1,3-di[(1s,4s)-4-methyl
    # cyclohexyl]propan-2-ol' against (a) (the Blue Book, 'bis' for
    # "compound or complex (i.e. substituted) prefixes"; TRIAGE g7 C16).
    # '(2S)-butan-2-yl' stays simple ('di[(2S)-butan-2-yl]', (a)).
    _stereo_block = re.match(r'^\((?:\d+[RSEZrsez],)*\d*[RSEZrsez]\)-', work)
    if _stereo_block and len(work) > _stereo_block.end():
        return is_substituted_substituent(work[_stereo_block.end():], mol=mol,
                                          atoms=atoms, attachment=attachment)

    lowered = work.lower()

    # GRAPH-PRIMARY. When a caller supplies the fragment, (c) is a
    # question about the skeleton and the graph answers it directly; the name is
    # then consulted only for the two carve-outs above and below, which are
    # spelling facts the graph cannot see.
    if mol is not None and atoms is not None:
        graph_verdict = _fragment_bears_substituent(mol, atoms, attachment)
        if graph_verdict is not None:
            if lowered in _SIMPLE_PREFIX_VOCABULARY:
                return False
            return graph_verdict

    # A SUFFIX, not a substituent prefix. Suffix tokens are multi-word
    # (`carboxylic acid`, `dithioic O-acid`); a substituent prefix never is.
    # The routed callers use `get_suffix_multiplier_prefix`, and this is the
    # backstop for any that were missed -- without it, the fail-safe below would
    # turn every unrecognised suffix into `bis`.
    if ' ' in lowered:
        # `work`, not `lowered`: the italic chalcogen locant is upper-case
        # (`thioic O-acid`), and the normaliser strips it case-insensitively.
        return suffix_takes_derived_multiplier(work)
    if lowered.endswith(_FUNCTIONAL_SUFFIX_TAILS):
        return False

    # A CONCATENATED unit is substituted even when the producers' table carries
    # it as one spelling. `PREFIX_FORMS` contains both kinds -- `nitroso` and
    # `hydroperoxy` are single units, while `phosphonooxy`, `nitrooxy`,
    # `carbamoyloxy`, `aminooxy`, `sulfanyloxy` and `hydroxyimino` are
    # concatenations of two -- so this split is tested BEFORE whole-token
    # membership, or the table's own compound entries would short-circuit to
    # "simple". The Blue Book files them under sections literally titled
    # `**** Compound and complex substituent groups` (`:36327`,
    # which Appendix 2's row `:56614` cites for `phosphonooxy*`) and
    # `### **** Substituent groups formed by concatenation` (`:31276`,
    # for `sulfooxy`), and `****` (`:36484`) says outright they are
    # "*formed by concatenation or substitution (see *".
    #
    # NO multiplier syllable is peeled here -- that is what keeps `disulfanyl`
    # (-S-SH, a single preselected prefix) from splitting into `di` + `sulfanyl`.
    if _splits_at_a_unit_boundary(lowered, allow_multiplier=False):
        return True

    # ONE recognised component -> simple (a)). Tested on the WHOLE
    # token before any multiplier peeling, which is what makes `nitroso`,
    # `tridecyl`, `disulfanyl` and `hydroperoxy` safe: each is itself a listed
    # prefix, so it is never decomposed into a prefix plus a syllable.
    if _is_single_simple_unit(lowered):
        return False

    # An acyloxy prefix is a compound (acyl + oxy) prefix: BB `bis(acetyloxy)`,
    # `tetrakis(acetyloxy)`, and this repo's `bis(hexadecanoyloxy)` gold rows.
    # A bare alkoxy has no `yl` before the `oxy` -- and the six RETAINED
    # contracted alkoxy prefixes were already accepted as single
    # units above, so this cannot misfire on them.
    if lowered.endswith('yloxy') and len(lowered) > 5:
        return True

    # `acetamido`, `benzamido`, `formamido` -- the contracted spelling of
    # `<acyl>ylamino`, i.e. a concatenation of two units, so `bis` per
    # (a). Cf. `### **** Substituents of the types
    # -NH-CO-R and -NH-SO2-R` (`:32991`). The bare `amido` has an empty head and
    # is not caught.
    if lowered.endswith('amido') and len(lowered) > 5:
        return True

    # An enclosed SUBSTITUENT leader followed by more name is a detached prefix
    # by construction -- but a leading bracketed DESCRIPTOR is not.
    if _enclosed_leader_is_a_substituent(work):
        return True

    # Peel an optional leading locant set -- the locants of a DETACHED prefix
    # (`2-methylbutyl`, `N-pentylcarbamoyl`). A locant belonging to the core's
    # own attachment or unsaturation sits AFTER the stem (`propan-2-yl`) and was
    # already handled by `_is_single_simple_unit`.
    #
    # ORDER MATTERS, and getting it wrong shipped a regression: the enclosed-
    # leader test used to run on the RAW name, so a leading locant blocked it
    # permanently and `3-(4-methylphenyl)propyl` read as simple while
    # `(4-methylphenyl)propyl` read as substituted. Hence the peel is repeated
    # here and the leader test re-run on the peeled body.
    body = lowered
    m = _LEADING_LOCANT_RE.match(body)
    if m:
        body = body[m.end():]
        if _enclosed_leader_is_a_substituent(body):
            return True
        if _is_single_simple_unit(body):
            return False
        # The detached prefix after the locant set may itself begin with an
        # italicized structural prefix, alone or multiplied: '4-tert-butylphenyl',
        # '2,4-di-tert-butylphenyl', 'N-tert-butylcarbamoyl'. Its italic letters are
        # part of that simple prefix ('tert-butyl', (a), the Blue Book)
        # and are set aside here as they are at the front of a name above; what
        # follows decides ('butylphenyl' is two units, so the whole prefix is
        # substituted and takes 'bis', (c):7035, (a):7104,
        # 'bis({3,5-di-tert-butyl-2-...phenyl}methylidene)':37980).
        unitalic, had_italic = _set_aside_italicized_prefix(body)
        if had_italic:
            return is_substituted_substituent(unitalic)

    # A retained contracted alkoxy prefix, the Blue Book) with a
    # substituent prefix in front of its stem, read from the root's ending where the
    # vocabulary split does not know the root ('acetylmethoxy', 'ethenylmethoxy',
    # 'bis(trifluoromethyl)methoxy'): substituted, so 'bis' (c),:7035) and
    # enclosing marks,:7232). The enclosure predicates ask this same
    # predicate through is_substituted_contracted_alkoxy.
    return _splits_into_two_units(body) or _contracted_alkoxy_root_is_a_prefix(body)


def _set_aside_italicized_prefix(body: str) -> Tuple[str, bool]:
    """``body`` without a leading italicized structural prefix that may follow a
    multiplying syllable: 'tert-butylphenyl' -> ('butylphenyl', True),
    'di-tert-butylphenyl' and 'ditert-butylphenyl' -> ('dibutylphenyl', True),
    'methylphenyl' -> ('methylphenyl', False)."""
    for mult in ('',) + _LEADING_MULTIPLIER_SYLLABLES:
        if mult and not body.startswith(mult):
            continue
        rest = body[len(mult):]
        if mult and rest.startswith('-'):
            rest = rest[1:]
        remainder, had = strip_italicized_structural_prefix(rest)
        if had and remainder:
            return mult + remainder, True
    return body, False


def _is_recognised_prefix_component(token: str) -> bool:
    """One complete substituent-prefix component of any recognised kind.

    A vocabulary unit, a bare parent-hydride-yl, or a concatenated acyl form
    (`acetamido`, `hexadecanoyloxy`). A bare parent-hydride STEM (`benzene`,
    `propan`) is deliberately NOT a component: a stem is not a prefix, and
    accepting one would split the simple `benzenesulfinyl` -- (a)'s
    "*functionalized parent hydrides*" class, verbatim
    `di(benzenesulfinyl)acetic!acid!(PIN)` (`:7324`) -- into two units.
    """
    if not token:
        return False
    if token in _SIMPLE_PREFIX_VOCABULARY:
        return True
    if _is_bare_hydride_yl(token, require_known_stem=True):
        return True
    if len(token) > 5 and token.endswith('yloxy'):
        return True
    # `acetamido`, `benzamido`, `formamido`: an acyl stem concatenated with
    # `amido`, i.e. the contracted spelling of `<acyl>ylamino`. Compound by
    # (a); cf. `### **** Substituents of the types
    # -NH-CO-R and -NH-SO2-R` (`:32991`) and its `bis(cyanomethyl)oxamide (PIN)`
    # (`:33087`).
    if len(token) > 5 and token.endswith('amido'):
        return True
    return _is_replacement_named_hydride_yl(token)


# Skeletal replacement ('a') prefixes, each optionally multiplied and
# optionally followed by the locant set of the next one: 'oxa', 'dioxa',
# 'oxa-2-sila', 'dioxa-2,4-diphospha'.
_A_PREFIX_RUN_RE = re.compile(
    r'^(?:(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca)?'
    r'(?:fluora|chlora|broma|ioda|oxa|thia|selena|tellura|aza|phospha|arsa|'
    r'stiba|bisma|sila|germa|stanna|plumba|bora|alumina|galla|inda|thalla)'
    r'(?:-\d+(?:,\d+)*-)?)+')


def _is_replacement_named_hydride_yl(token: str) -> bool:
    """Is ``token`` ONE parent-hydride prefix named by skeletal replacement,
    such as 'oxabutyl', 'dioxahexyl' or 'oxa-2-silabutyl'?

    "Nondetachable prefixes describe structural modifications to the parent
    structure creating new parent structures, for example, replacement prefixes,
    which can be either skeletal replacement ('a') prefixes (see or..."
    , the Blue Book). The 'a' prefixes are part of the parent
    hydride, so the whole token is one component. As the tail of a split this
    makes '2-ethyl-1-oxabutyl' two units ('ethyl' + '1-oxabutyl'), a substituted
    prefix that takes 'bis' (a), the Blue Book); '1-oxabutyl'
    alone stays a single unit."""
    m = _A_PREFIX_RUN_RE.match(token)
    if not m or m.end() >= len(token):
        return False
    core = token[m.end():]
    return (core in _SIMPLE_PREFIX_VOCABULARY
            or _is_bare_hydride_yl(core, require_known_stem=True))


def _splits_at_a_unit_boundary(body: str, allow_multiplier: bool = True,
                               _depth: int = 0) -> bool:
    """Does ``body`` need TWO OR MORE units of the producers' vocabulary?

    That is the operative form of (c) / (a): a compound prefix is
    one built by concatenating units, and **both** sides of the split must be
    recognised. Requiring only the front to be recognised and testing the tail
    by SHAPE is what produced `bis(nitroso)` (`nitro` + `so`, where `so` merely
    ended in a group-shaped letter) and `bis(oxolan-2-yl)` (`oxo` + `lan-2-yl`).

    Split is attempted from both ends, because either side may be the
    multi-character one: `methoxy|methyl` is found from the front,
    `pyridin-2-yl|methyl` and `acetamido|phenyl` only from the back.
    """
    if _depth > 4 or not body:
        return False

    mults = ('',) + _LEADING_MULTIPLIER_SYLLABLES if allow_multiplier else ('',)

    # front-anchored: <optional multiplier><unit><recognised remainder>
    for mult in mults:
        if mult and not body.startswith(mult):
            continue
        rest = body[len(mult):]
        if not rest:
            continue
        for unit in _FRONT_PREFIX_UNITS:
            if not rest.startswith(unit) or len(rest) == len(unit):
                continue
            tail = rest[len(unit):].lstrip('-')
            if not tail:
                continue
            m = _LEADING_LOCANT_RE.match(tail)
            if m:
                tail = tail[m.end():]
            if not tail:
                continue
            if _is_recognised_prefix_component(tail):
                return True
            # A located prefix after the unit that opens with an enclosed
            # substituent is a second unit too: 'hydroxy-3-[(2-methylprop-2-
            # enoyl)oxy]propoxy' is 'hydroxy' + '[(...)oxy]propoxy', a substituted
            # prefix that takes 'tris' (a), the Blue Book).
            if m and _enclosed_leader_is_a_substituent(tail):
                return True
            if _splits_at_a_unit_boundary(tail, allow_multiplier, _depth + 1):
                return True

    # back-anchored: <recognised head><unit>
    for unit in _FRONT_PREFIX_UNITS:
        if not body.endswith(unit) or len(body) == len(unit):
            continue
        head = body[:-len(unit)].rstrip('-')
        if not head:
            continue
        m = _LEADING_LOCANT_RE.match(head)
        if m:
            head = head[m.end():]
        if head and _is_recognised_prefix_component(head):
            return True
    return False


# Kept as the historical name used by the front-anchored call site above.
_splits_into_two_units = _splits_at_a_unit_boundary


def _fragment_bears_substituent(mol, atoms, attachment: Optional[int]):
    """GRAPH form of (c): does the fragment's own parent hydride bear a
    substituent?

    Returns ``True``/``False``, or ``None`` when the graph cannot decide and the
    caller must fall back to the name (an empty/mis-specified fragment, or a
    fragment whose parent-hydride choice is genuinely name-dependent).

    The test: walk the skeleton reachable from ``attachment`` through atoms of the
    attachment atom's own element, plus any ring system it belongs to. Anything
    hanging off that skeleton is a substituent.

    NOTE the deliberate asymmetry with the name path. This is consulted only
    when a caller supplies a fragment, and today NO caller can -- every one of
    the ~88 `get_multiplier_prefix` call sites has already reduced its fragment
    to a string by the time the multiplier is chosen. That is reported as a
    finding rather than papered over: threading the graph to all of them is a
    separate, larger refactor, and the name path above is not a fallback but the
    primary implementation, because is decided by the DERIVATION the
    producer chose (see the `tert-butyl` / `2-methylpropan-2-yl` and
    `methoxy` / `methoxymethyl` refutations above).
    """
    try:
        idxs = set(int(a) for a in atoms)
    except Exception:
        return None
    if not idxs:
        return None
    if attachment is None or int(attachment) not in idxs:
        return None
    anchor = int(attachment)
    try:
        anchor_atom = mol.GetAtomWithIdx(anchor)
        ring_info = mol.GetRingInfo()
    except Exception:
        return None
    element = anchor_atom.GetAtomicNum()

    # The fragment's own parent hydride: atoms reachable from the anchor through
    # same-element bonds, unioned with every ring the anchor sits in.
    skeleton = {anchor}
    frontier = [anchor]
    while frontier:
        cur = frontier.pop()
        for nbr in mol.GetAtomWithIdx(cur).GetNeighbors():
            j = nbr.GetIdx()
            if j not in idxs or j in skeleton:
                continue
            same_ring = any(cur in r and j in r for r in ring_info.AtomRings())
            if nbr.GetAtomicNum() == element or same_ring:
                skeleton.add(j)
                frontier.append(j)
    return bool(idxs - skeleton)


# W3-P03-7 (c)/(d), BB 38222/L7104): unbranched primary-alkyl prefixes
# whose NAME begins with a numerical-multiplier syllable take ENCLOSING MARKS
# when multiplied (count > 1), so the reader cannot fold the multiplier into the
# alkyl stem — 'di(dodecyl)silane' (PIN), 'di(decyl)', 'di(tridecyl)', 'tri(decyl)'
# — while KEEPING the BASIC di/tri multiplier (NOT the derived bis/tris; the alkyl
# name is not otherwise complex). The affected stems are decyl (C10; 'dec-'=deca)
# and dodecyl..nonadecyl (C12-C19; 'do-'/'tri-'/'tetra-'/…/'nona-' + 'decyl').
# EXCLUDED: methyl..nonyl (C1-C9, no leading multiplier syllable), undecyl (C11,
# 'un-' is not a multiplier), and icosyl+ (C20+, 'icos-' is not a leading di/tri
# syllable). This is orthogonal to is_complex_substituent (which would wrongly
# switch the multiplier to bis) and to needs_brackets (which never flags them).
_P1634_MARK_ALKYL_STEMS = frozenset({
    "decyl",        # C10 (dec- = deca)
    "dodecyl",      # C12 (do- = di in dodeca)
    "tridecyl",     # C13
    "tetradecyl",   # C14
    "pentadecyl",   # C15
    "hexadecyl",    # C16
    "heptadecyl",   # C17
    "octadecyl",    # C18
    "nonadecyl",    # C19
})


def needs_p1634_marks(name: str) -> bool:
    """(c)/(d): does this alkyl prefix take enclosing marks when multiplied?

    Returns True for unbranched primary-alkyl names beginning with a numerical-
    multiplier syllable (decyl C10; dodecyl..nonadecyl C12-C19). Such names are
    parenthesised when count > 1 — 'di(dodecyl)silane' — but keep the basic di/tri
    multiplier. undecyl (C11) and icosyl+ (C20+) are excluded (no leading
    multiplier syllable), as are C1-C9. The caller applies the count > 1 gate.

    Examples:
        >>> needs_p1634_marks("dodecyl")
        True
        >>> needs_p1634_marks("undecyl")
        False
        >>> needs_p1634_marks("methyl")
        False
    """
    return name.lower() in _P1634_MARK_ALKYL_STEMS


# (the Blue Book): simple prefixes whose 'di'/'tri' concatenation
# collides with a catenated-hydride prefix (disulfanyl = -SSH, NOT
# two -SH). These take the derived multipliers bis/tris/... even though the
# name itself is simple; then parenthesizes the multiplied term.
CATENATION_AMBIGUOUS_PREFIXES = frozenset({
    "sulfanyl",   # disulfanyl -SSH verbatim example)
    "selanyl",    # diselanyl -SeSeH: 'diselanyl, -SeSeH')
    "tellanyl",   # ditellanyl -TeTeH
    "phosphanyl", # diphosphanyl examples)
    "arsanyl",    # diarsanyl (diarsane
    "stibanyl",   # distibanyl (distibane
    "azanyl",     # diazanyl = hydrazinyl -NH-NH2
    "oxidanyl",   # dioxidanyl -OOH
    # The catenated-chain prefixes themselves are ALSO ambiguous when multiplied:
    # two -S-S-H groups are 'bis(disulfanyl)', because 'didisulfanyl' would read as
    # a catenation ('di' + 'disulfanyl' = a 4-sulfur -SSSS- chain). BB
    # verbatim: '3,4-bis(disulfanyl)benzamide (PIN)' (the Blue Book) and
    # 'bis(disulfanyl)... (not didisulfanyl)' (the Blue Book). The tri- forms
    # (-S-S-S-H = trisulfanyl) collide the same way ('ditrisulfanyl' -> a -SSSSSS-
    # chain), so they take bis/tris too.
    "disulfanyl",  # -SSH -> bis(disulfanyl) (the Blue Book, the Blue Book)
    "diselanyl",   # -SeSeH -> bis(diselanyl)
    "ditellanyl",  # -TeTeH -> bis(ditellanyl)
    "trisulfanyl", # -SSSH -> bis(trisulfanyl)
    "triselanyl",  # -SeSeSeH
    "tritellanyl", # -TeTeTeH
    # (e) (the Blue Book) 'bis', 'tris'... "before names beginning
    # with a multiplicative prefix 'di'": 'bis(diazenyl) (preselected prefix...)
    # (not didiazenyl)' (:7194). The other catenated dinuclear hydride prefixes.
    "diazenyl",    # HN=N- -> bis(diazenyl) (:7194)
    "diazanyl",    # H2N-NH- -> bis(diazanyl)
    "disilanyl", "digermanyl", "distannanyl", "diplumbanyl", "diboranyl",
    "diphosphanyl", "diarsanyl", "distibanyl", "dibismuthanyl", "dioxidanyl",
})


# (c): the skeletal replacement ('a') prefixes at the front of a
# substituent prefix, in full ('oxa', 'dioxa', 'thia', 'aza', 'sila': '1-oxaethyl',
# '1,2-oxazol-3-yl', 'oxan-2-yl', '2-azabicyclo[...]') or with the final 'a' elided
# before a Hantzsch-Widman stem ('oxolan-2-yl', 'oxiran-2-yl', 'azetidin-1-yl',
# '1,2,4-triazol-1-yl'). The halogen 'a' prefixes are left out (their spellings
# begin retained names, 'fluoranthenyl'), and the Group 13 stems out of the elided
# form ('indolyl').
_REPLACEMENT_FRONT_RE = re.compile(
    r"^(?!(?:thi|ox|selen|tellur)anthren|(?:phosph|ars|stib|bism)in(?:ol|dol)"
    r"|(?:di|tri|tetra)?(?:az|phosph|ars|stib|bism|sil|germ|stann|plumb|bor)ane$)"
    r"(?:(?:di|tri|tetra|penta|hexa)?"
    r"(?:oxa|thia|selena|tellura|aza|phospha|arsa|stiba|"
    r"bisma|sila|germa|stanna|plumba|bora)"
    r"|(?:di|tri|tetr|tetra)?(?:ox|thi|selen|tellur|az|phosph|ars|stib|bism|sil|germ|"
    r"stann|plumb|bor)(?=ir|et|ol|in|an|ep|oc|on|ec))")
# (The retained fusion names 'thianthrene', 'oxanthrene', 'selenanthrene',
# 'phosphinoline', 'arsinoline', 'phosphindole',... open with an 'a' stem but are
# not replacement names, 'thianthrene (PIN)':11757; 'indazole' is retained too.
# A mononuclear or catenated hydride parent ('silane', 'disilane', 'phosphane') is
# not a replacement name either; its own multiplier rule is (a).)

#: The endings of a parent hydride name that a multiplicative name multiplies
#: ('azetidine', 'azepane', '1,3-dioxolane', 'silane'): (c) applies to the
#: parent join of multiplicative nomenclature as to a substituent prefix
#: ('bis(azacyclododecane) (PIN)', the Blue Book).
_PARENT_HYDRIDE_ENDINGS = ("ane", "ene", "ine", "ole", "ane)", "ine)")


def opens_with_replacement_prefix(name: Optional[str]) -> bool:
    """ (c) (the Blue Book,:7174-7178): does this substituent prefix open,
    after its locants and indicated hydrogen, with a skeletal replacement ('a')
    prefix, so that 'di' in front of it could be read as part of the replacement
    count ('di(1,2-oxazol-3-yl)' as a 'dioxazole')? Substituent prefixes (ending
    in 'yl', 'ylidene', 'ylidyne') and the parent hydrides a multiplicative name
    multiplies ('azepane' in "1,1'-methylenebis(azepane)", not the
    'diazepane' ring) are considered; the acyl prefixes of oxalic and oxamic
    acid and the oxoacid prefixes ('phosphono') are not replacement names."""
    if not name:
        return False
    core = name.strip()
    while _is_fully_enclosed(core):
        core = core[1:-1]
    core = re.sub(r"^(?:[0-9,]+[a-z]?H?-|\d*H-|[0-9,']+-)+", "", core)
    if not core.endswith(("yl", "ylidene", "ylidyne") + _PARENT_HYDRIDE_ENDINGS):
        return False
    if core.startswith(("oxal", "oxam")):
        return False
    return bool(_REPLACEMENT_FRONT_RE.match(core))


#: A 'hydro' prefix as it stands in a name: its locant set, the hyphen, a basic
#: multiplying prefix and 'hydro' ('2,3-dihydro', '1,2,3,4-tetrahydro',
#: '4a,8a-dihydro'), not the 'hydroxy' / 'hydroperoxy' / 'hydroseleno' /
#: 'hydrotelluro' / 'hydrosulf...' prefixes or 'hydrogen'. A hydro prefix on a
#: selenophene or tellurophene ring ('4,5-dihydroselenophen-2-yl',
#: '2,3-dihydroselenopheno[2,3-b]pyridine') still counts; '4-hydroselenophenyl'
#: (a 'hydroseleno' prefix on 'phenyl') does not. Indicated hydrogen ('1H-') is
#: not a hydro prefix.
_HYDRO_PREFIX_RE = re.compile(
    r"(?<![a-z])\d+[a-z]?'*(?:,\d+[a-z]?'*)*-"
    r"(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca|undeca|dodeca|tetradeca"
    r"|hexadeca|octadeca|icosa)?hydro"
    r"(?!xy|peroxy|gen|sulf|(?:selen|tellur)(?!ophen(?:e|o|-)))")


def carries_hydro_prefix(name: Optional[str]) -> bool:
    """Does this substituent prefix carry 'hydro' prefixes of its own
    ('2,3-dihydro-1H-indol-1-yl', '4,5-dihydrothiophen-2-yl')? Indicated hydrogen
    alone ('1H-imidazol-1-yl') does not count."""
    return bool(name) and _HYDRO_PREFIX_RE.search(name) is not None


def get_multiplier_prefix(count: int, substituent_name: str) -> str:
    """Get the appropriate multiplier prefix for a count of substituents.

    For count=1, returns empty string (no multiplier needed).

    ⚠ **Docstring corrected 2026-07-30.** It said "For simple substituent names (no
    digits/hyphens), uses di/tri/tetra. For complex substituent names, uses
    bis/tris/tetrakis." That is **not what this function does**, and the difference is
    load-bearing. The predicate is ``is_substituted_substituent`` (plus
    ``CATENATION_AMBIGUOUS_PREFIXES``) — **not** ``is_complex_substituent``, which a trace
    measured at **zero** calls from here. "Has digits/hyphens" is the *complex* test, and
    the two disagree:

        1,2-xylene complex=True substituted=False -> 'di'
        propan-2-yl complex=True substituted=False -> 'di'
        bromomethyl complex=True substituted=True -> 'bis'
        sulfanyl complex=False substituted=False -> 'bis' (catenation-ambiguous)

    The governing rule is ** / **, not (which is *alphanumerical
    order* — a section this project has twice cited by mistake for di/bis). The
    discriminator is a *contracted* name: ``:5098`` ``1,1-dimethoxypropane (PIN)`` takes
    ``di`` because ``methoxy`` is contracted (``:17958``), while ``:35344``
    ``1,1-bis(methylsulfanyl)pentane (PIN)`` takes ``bis`` for the uncontracted form.

    ⛔ **Do not "fix" this by re-pointing at ``is_complex_substituent``.** Commit
    `` did exactly that and regressed in both directions —
    ``bis(propan-2-yl)`` against the PIN at ``:25719``, and ``dihydroxymethyl`` against
    (a). Preserve the ordering of the checks.

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

    # -FIX Item 4: the hyphen belongs to FORMING the multiplied token, so it
    # is produced here -- the one place that already knows both the count and the
    # name -- rather than at each of the ~40 call sites. Three of those sites had
    # open-coded around it and shipped `ditert-butyl` / `N,N-ditert-butyl...`,
    # which appears ZERO times in the Blue Book, while sibling producers spelled
    # the same fragment `di-tert-butyl`.
    #
    # `**** Hyphens` -> `**** Hyphens are used in substitutive
    # names:` clause `(d) to separate italic letters from Roman letters`
    # (the Blue Book) with the verbatim example `di-tert-butyl `
    # (:6964); `### **** Cyclic hydrocarbons` has `1,2-di-tert-butyl-
    # benzene (PIN)` (:25717) and `tri-tert-butylphenyl` at:37495. The multiplier
    # is the SIMPLE `di`/`tri` because `**** General methodology` clause
    # (a) (:7033) lists "unsubstituted prefixes, such as ethyl or tert-butyl"
    # among the simple components "multiplied by the multiplicative prefixes
    # 'di', 'tri', etc." -- so `bis(tert-butyl)` and `di(tert-butyl)` are both
    # wrong, and neither occurs in the text.
    #
    # -CLOSEOUT Item A, ORDERING CORRECTION. This hyphen leg used to run
    # FIRST, on the asserted ground that "only ever fires for an italicized-
    # prefix-led name, which is simple by (a) and therefore never takes
    # bis/tris, so the derived-multiplier branch below is unreachable with a
    # hyphen". That assertion is FALSE: `tert-butylsulfanyl` is italicized-
    # prefix-led and SUBSTITUTED (it reduces to the compound `butylsulfanyl`),
    # so it takes `bis` per (a) — and running the hyphen leg first
    # emitted `di-tert-butylsulfanyl`. The substituted test therefore has to
    # come first, and the two rules DO collide, exactly as (`:6968`)
    # anticipates: "*No hyphen is placed after a numerical prefix cited in front
    # of a compound substituent enclosed by parentheses*" — so a substituted
    # italicized-led prefix takes `bis(...)` with NO hyphen, while a simple one
    # (`tert-butyl`) keeps the (d) hyphen. Both legs now reachable.
    #
    # The MULTIPLIER question is (c) /
    # (a) -- "is the component SUBSTITUTED?" -- and NOT the enclosure
    # question that `is_complex_substituent` answers. Consulting the
    # enclosure predicate here was wrong in both directions: it made every
    # locant-bearing SIMPLE prefix `bis` (`bis(propan-2-yl)` against
    # `1,4-di(propan-2-yl)cyclohexane (PIN)`, the Blue Book) and left every
    # digit-free SUBSTITUTED prefix `di` (`dihydroxymethyl` against
    # (a)'s `bis(bromomethyl)` class). `is_complex_substituent` keeps
    # its own job unchanged; only this decision moved.
    #
    # The second leg is (d) (a simple component that would be AMBIGUOUS
    # if multiplied by 'di' -- `disulfanyl` is -S-SH, so two -SH must be
    # `bis(sulfanyl)`), and it is preserved exactly as it was.
    # The third leg is (c): a component that opens with a skeletal
    # replacement ('a') prefix, of an 'a' chain or a Hantzsch-Widman ring, takes
    # the derived multiplier "to describe clearly the number of replacement atoms"
    # ('bis(1,2-oxazol-3-yl)... whereas di(1,2-oxazol-3-yl) might be interpreted as
    # a dioxazole ring system', the Blue Book; 'bis(azacyclododecane)':7176).
    # The fourth leg is a prefix that carries 'hydro' prefixes of its own. Hydro
    # prefixes are detachable prefixes that are "not included in the category of
    # alphabetized detachable prefixes which describe substitution" (the summary
    # of changes, item 5, the Blue Book), so the rule text of /
    # (a) (:7085,:7104) does not settle the multiplier; the Blue Book's
    # one multiplied hydro-prefixed prefix takes 'bis':
    # 'bis(4,5-dihydrothiophen-2-yl)di(methyl)germane (PIN)',
    #:38232), and it has no 'di(' before such a prefix. Indicated hydrogen alone
    # keeps 'di': 'di(1H-imidazol-1-yl)methanethione (PIN)',:29544).
    if is_substituted_substituent(substituent_name) \
            or substituent_name in CATENATION_AMBIGUOUS_PREFIXES \
            or opens_with_replacement_prefix(substituent_name) \
            or carries_hydro_prefix(substituent_name):
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
        # Single source of the / simple-multiplier rule.
        # ``count <= 1`` already returned above, so ``None`` here can only mean
        # "outside the composition's range" -- the same ValueError the inlined
        # ``get_chain_prefix`` call used to raise.
        word = simple_multiplier_word(count)
        if word is None:
            raise ValueError(
                f"no simple multiplying prefix can be formed for {count}")
        # (d): a SIMPLE prefix led by an italicized structural prefix
        # keeps a hyphen between the multiplier and the italic letters --
        # `di-tert-butyl`, never `ditert-butyl`. Reached only on the simple leg,
        # so a substituted `tert-`-led prefix correctly gets `bis(...)` above
        # with no hyphen.
        if multiplier_needs_hyphen(substituent_name):
            return word + "-"
        return word


def multiplied_component(count: int, name: str, marked: str) -> str:
    """Join the multiplier to a component whose enclosure is already decided.

    ``name`` is the BARE prefix name (what decides the multiplier); ``marked`` is
    the same prefix after the caller has applied whatever enclosing marks its own
    .x context requires. Returns the finished multiplied token.

    THE one place the multiplier and the hyphen meet, so the ~7 producers that
    used to keep a private ``{1:'', 2:'di', 3:'tri'}`` table -- and therefore
    could never emit ``bis``/``tris`` at all -- cannot drift again.

    Two rules interact here and both are in:

    * `****` clause (d) (``the Blue Book``, verbatim
      ``di-*tert*-butyl``): a hyphen separates the multiplier from a BARE
      italicized prefix.
    * `****` (``:6968``): "*No hyphen is placed after a numerical prefix
      cited in front of a compound substituent enclosed by parentheses, even if
      that substituent begins with locants*" -- so once the caller has enclosed
      the component, the hyphen is dropped.

    Examples:
        >>> multiplied_component(2, 'methyl', 'methyl')
        'dimethyl'
        >>> multiplied_component(2, 'tert-butyl', 'tert-butyl')
        'di-tert-butyl'
        >>> multiplied_component(2, 'propan-2-yl', '(propan-2-yl)')
        'di(propan-2-yl)'
        >>> multiplied_component(2, 'cyclohexylmethyl', '(cyclohexylmethyl)')
        'bis(cyclohexylmethyl)'
        >>> multiplied_component(2, 'dodecyl', 'dodecyl')
        'di(dodecyl)'
    """
    if count > 1 and marked == name and enclose_if_compound(name) != name:
        # (a) (the Blue Book-7087, "simple substituent prefixes
        # having locants", 'di(propan-2-yl)') and (:7232, compound and
        # complex prefixes): a multiplied component a caller passes bare takes the
        # marks its name needs, so no producer can write 'dipropan-2-yl'.
        marked = enclose_if_compound(name)
    elif count > 1 and marked == name and needs_p1634_marks(name):
        # (the Blue Book, "Parentheses... are used to enclose
        # multiplied components that are:") (c) "simple substituent prefixes...
        # beginning with a multiplicative prefix" ('di(dodecyl)') and (d) those
        # "beginning with 'dec'" ('di(decyl)', 'tri(decyl)') (:7104); the
        # multiplier stays the basic one. 'di(dodecyl)silane (PIN)' (:38222).
        marked = f"({name})"
    mult = get_multiplier_prefix(count, name)
    if mult.endswith('-') and marked != name:
        #: the component is enclosed, so the hyphen goes.
        mult = mult[:-1]
    if ((mult in COMPLEX_MULTIPLIERS.values() or mult.endswith('kis'))
            and not _is_fully_enclosed(marked)):
        # (the Blue Book): "Parentheses are used to enclose
        # terms modified by the numerical prefixes 'bis', 'tris', 'tetrakis',
        # etc." -- 'bis(sulfanyl)', never 'bissulfanyl'.
        marked = apply_enclosing_marks(marked, -1)
    # Review F1 (properfix a performance pass): keep ``name``'s record on the finished,
    # multiplied token -- a caller that merges THIS output again (nested
    # multiplication, or a second dedup pass) must still be able to read it.
    from .prefix_derivation import carried
    return carried(f"{mult}{marked}", like=name)


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
    """Apply IUPAC bracket escalation to an N-substituent name.

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
    # Already one fully enclosed token -- '(...)', '[...]' or '{...}' -- no
    # double-wrapping. The brace case was missing: enclose_if_compound gives a
    # name holding '(' and '[' its outer '{...}', which the escalation below
    # re-wrapped as '({...})' -- the same fragment enclosed twice, against
    # (the Blue Book) one level per enclosed fragment:
    # 'N-({(3S)-3-(1-methyl-1H-indol-3-yl)-3-[3-(trifluoromethyl)phenyl]
    # propanoyl})pyrrolidine' (TRIAGE g7 C10).
    if _is_fully_enclosed(name):
        return name
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
    # Contains an inner enclosing mark (stereo prefix, compound sub-substituent,
    # an already-bracketed sub-fragment, etc.) -> escalate (-> [ -> { per
    # via the shared primitive, which auto-detects the nesting depth
    # from the name's existing marks. The old `'(' in name -> f"[{name}]"` capped
    # at one level and produced a double `[[...]...]` when the name already held a
    # square bracket (e.g. an N-[(biphenylyl)methyl] arm) -- a phase (B).
    if any(mark in name for mark in '(['):
        return apply_enclosing_marks(name, -1)
    return name


def _is_fully_enclosed(name: str) -> bool:
    """True when ``name`` is a single fully-bracketed token: it opens with a
    bracket whose MATCHING close is the final character (e.g. '(2-methylpropyl)',
    '[bis(sulfanyl)]'). False when trailing text lies outside the leading
    bracket ('(oxan-2-yl)oxy', '(benzylsulfanyl)methyl') — those still need an
    OUTER enclosing mark. Used by format_substituent_prefix's escalation.
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


_ITALIC_LOCANT_LETTERS = frozenset({
    "N", "O", "S", "P", "C", "B", "Si", "Se", "Te", "As", "Sb", "Bi", "Al",
})


def _is_locant_token(token: str) -> bool:
    """True iff ``token`` is a single locant: ``4``, ``N``, ``N'``, ``N1``."""
    text = token.strip().rstrip("'′″")
    if not text:
        return False
    if text.isdigit():
        return True
    # An italic heteroatom locant, optionally with a superscripted index (N1, N3).
    head = text.rstrip("0123456789")
    return bool(head) and head in _ITALIC_LOCANT_LETTERS


#: The italic designators that begin a retained ring parent name: 's-indacene
#: (PIN)', 'as-indacene (PIN)', the Blue Book).
_ITALIC_PARENT_DESIGNATOR_RE = re.compile(r"^(?:as|s)-(?=[a-z])")


def begins_with_italic_designator(name: str) -> bool:
    """True iff ``name`` opens with the italic designator of a retained parent
    ('s-indacene', 'as-indacen-1-yl'). A preceding prefix that ends in a Roman
    letter is then joined with a hyphen: (the Blue Book, "Hyphens
    are used in substitutive names:") (d) "to separate italic letters from Roman
    letters" (:6960), example '*as*-indacene' (:6966); '...cyclopenta[*cd*]-*s*-
    indacene (PIN)' (:14505). So '3-methyl-s-indacene', not '3-methyls-indacene'."""
    return bool(name) and bool(_ITALIC_PARENT_DESIGNATOR_RE.match(name))


def starts_with_locant(prefix: str) -> bool:
    """True iff a formatted substituent prefix opens with a locant set.

    ``'4-chloro'`` and ``'N-methyl'`` and ``'N,4-dimethyl'`` -> True;
    ``'chloro'`` and ``'methyl'`` -> False.

    WHY: a locant is always separated from the preceding prefix by a hyphen, and
    the benzene joiner used to test only ``first_char.isdigit``. An ITALIC
    locant starts with a letter, so a merged prefix list glued together as
    ``3-chloroN-methylbenzamide`` -- which OPSIN parsed to the correct InChIKey,
    the documented failure mode where a structural oracle cannot see a spelling
    defect. Deciding a separator during assembly is not a post-processor; nothing
    here rewrites a finished name.

    Fails SAFE: an unrecognised head returns False, i.e. the previous behaviour.
    """
    head, sep, _rest = prefix.partition("-")
    if not sep or not head:
        return False
    return all(_is_locant_token(tok) for tok in head.split(","))


_FUSION_LOCANT_RE = re.compile(r"(\d+)([a-z]+)")


def locant_sort_key(locant) -> Tuple[int, int, str]:
    """Order one locant within a substituent group's locant set.

    ITALIC LETTER locants (``N``, ``N'``, ``O``, ``S``) sort BEFORE arabic
    numerals; numerals then sort numerically, and same-letter italics
    lexically. The Blue Book's own mixed sets are the authority:

    * ``:42213`` ``*N*,*N*,*N*,1-tetramethylquinolin-1-ium-3-aminium (PIN)``
    * ``:42460`` ``*N*,1,4-triphenyl-1*H*-1,2,4-triazol-4-ium-3-aminide (PIN)``

    Both show the italic letters leading and ONE multiplying prefix spanning the
    whole set -- which is (``:7038``) clause (b) (``:7067``), "the basic
    numerical prefixes 'di', 'tri', 'tetra', etc. are used to indicate a
    multiplicity of:... simple substituent prefixes", whose own example list
    prints ``dimethyl``. Multiplicity is a property of the substituent NAME, not
    of which atom carries it, so an ``N``-methyl and a ring/chain ``methyl`` are
    ONE group of two. "Citation of locants" (``:2869``) is then
    deny-by-default and requires the whole locant set to be cited.

    WHY THIS EXISTS AS A SHARED HELPER: a merged group holds ``int`` and ``str``
    locants together, and ``sorted(["N", 4])`` raises TypeError. Every producer
    that merges an italic bucket into a numeric one needs exactly this key, so it
    lives here rather than being re-derived per producer.

    ``format_substituent_prefix`` already renders a merged set correctly once it
    is ordered -- it reproduces both Blue Book examples byte-exactly -- so NO new
    prefix formatter is introduced. (That function's own docstring records "do
    NOT add a 4th" predicate; this helper deliberately obeys that.)
    """
    if isinstance(locant, bool):  # bool is an int subclass; never a locant
        raise TypeError(f"bad locant {locant!r}")
    if isinstance(locant, int):
        return (1, locant, "")
    text = str(locant)
    if text.isdigit():
        return (1, int(text), "")
    # A fusion locant is a numeral plus a lower-case letter ('4a', '8a'), and
    # (the Blue Book) places it "immediately after the
    # corresponding numeric locant" ('4, 4a, 5'), not in the italic-letter class.
    # Sorting a fused ring's '4a' against an int raised TypeError in
    # name_substituted_heterocycle (breadth job 3, M35).
    _fusion = _FUSION_LOCANT_RE.fullmatch(text)
    if _fusion:
        return (1, int(_fusion.group(1)), _fusion.group(2))
    return (0, 0, text)


def format_substituent_prefix(name: str, locants: List[int], count: int) -> str:
    """Format a substituent with locants and multiplier prefix.

    name-hygiene fix, item (e) / inventory (a phase — begin; consolidation lands in
    assembly/parenthesisation fix / a phase). This is the ONE correct substituent-prefix + needs-parens
    reference /. a phase consolidates the divergent
    needs-parens / enclosing-mark / prefix-assembly predicates onto THIS function;
    do NOT add a 4th. The divergent predicates as of 169.7:
      - rules/amides.py _has_positional_locants(name)
      - assembly/naming_utils.py is_complex_substituent(name)
      - assembly/naming_utils.py apply_enclosing_marks(name, depth)
      - rules/ortho_fused.py format_substituent_prefix(substituents) (different signature)

    -FIX Item 9 — this paragraph used to assert three things that are all
    FALSE today, and a false comment is how the next session acquires a wrong
    belief, so it is corrected rather than trimmed:

    * it claimed the predicates "DISAGREE today... on 'trifluoromethyl'/
      'tert-butyl': is_complex_substituent True but _has_positional_locants
      False". `is_complex_substituent('tert-butyl')` is **False**;
    * it claimed they are independent. `_has_positional_locants` now DELEGATES
      (`rules/amides.py`: `return is_complex_substituent(name)`), so the two
      cannot disagree at all;
    * it called the consolidation test an "xfail-strict tripwire". That marker is
      gone — `tests/unit/assembly/test_needs_parens_consolidation.py` now plainly
      asserts the equality and PASSES.

    Line numbers are deliberately omitted above: the previous ones (525, 461, 32)
    had all drifted, which is what made the inventory unfollowable. The
    consolidation onto THIS function still stands; do NOT add a fourth predicate.

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

    #: any term modified by a DERIVED multiplier (bis/tris/
    # tetrakis/...kis) is enclosed in parentheses, even when the base name
    # is simple (bis(sulfanyl),.
    derived_multiplier = bool(multiplier) and (
        multiplier in COMPLEX_MULTIPLIERS.values() or multiplier.endswith("kis")
    )

    # Complex substituents get parentheses around the name.
    # Per IUPAC, compound substituent names containing numeric locants
    # need enclosing marks to avoid ambiguity:
    # - count > 1: always wrap for bis/tris multiplier (e.g., "3,5-bis(2-methylpropyl)")
    # - count == 1: wrap if name has bare digits at the start (e.g., "3-(2-methylpropyl)")
    # but NOT for names that already have their own parenthesization (e.g., "(oxan-2-yl)oxy")
    # and NOT for names that are just hyphenated (e.g., "3-sec-butyl" is fine)
    complex = is_complex_substituent(name)
    # W3-P03-1: FG-fused-alkyl prefixes (carboxymethyl, hydroxymethyl,
    # aminomethyl,...) are digit-less/hyphen-less COMPOUND substituents that
    # is_complex_substituent misses but needs_brackets flags (BB '3-(carboxymethyl)
    # heptanedioic acid'). Fold needs_brackets in here — the ONE canonical
    # enclosing-mark predicate the docstring mandates.
    #
    #: this used to carry a FOURTH copy of the (b)/(d) tert-/sec- carve-out
    # (`and not name.lower.startswith(("tert-", "sec-"))`), because needs_brackets
    # lacked it. needs_brackets now owns the carve-out (via
    # strip_italicized_structural_prefix), so the open-code is gone. It was also
    # strictly BROADER than the rule: it suppressed the marks on 'tert-butyl' (right,
    # (b)/(d)) but equally on 'tert-butylsulfanyl' (wrong — a compound chalcogen
    # prefix takes marks under, exactly as '(methylsulfanyl)' does).
    if needs_brackets(name):
        complex = True
    # A compound prefix without a locant ('silylamino', 'cyanoamino',
    # 'methylcarbamoyl', 'nitrosooxy'): the same arm as enclose_if_compound
    #, the Blue Book; '4-(disilylamino)cyclohexane-1-carbonitrile
    # (PIN)',:38198). The contracted amide prefixes stay bare ('4-benzamido...').
    if (not complex and not name.lower().endswith('amido')
            and is_substituted_substituent(name)):
        complex = True
    # W3-P03-7 (c)/(d)): a multiplied alkyl name beginning with a numeric-
    # multiplier syllable (decyl / dodecyl..nonadecyl) takes enclosing marks —
    # 'di(dodecyl)silane' — but the multiplier stays basic di/tri (get_multiplier_
    # prefix, computed above from is_complex_substituent, is untouched, so it is
    # NOT switched to bis). Gated on count > 1: a single 'dodecylsilane' is bare.
    if count > 1 and needs_p1634_marks(name):
        complex = True
    if _has_stereo_prefix(name):
        # Name has a CIP stereo descriptor prefix (e.g., "(R)-sec-butyl"):
        # the descriptor's parentheses count, the Blue Book),
        # so the next mark of the order (BB 7444) -- '[' for a name no
        # deeper than the descriptor, '{' once it carries its own '[...]':
        # '{(2S)-1-[4-(5-carboxypentyl)phenyl]-...-2-ylamino}', not '[(2S)-1-[...'.
        formatted_name = apply_enclosing_marks(name, -1)
    elif ('(' in name or '[' in name) and not _is_fully_enclosed(name):
        # (W2E-P1FC Task 8, generalized by w2f p1 per
        # /: the name CARRIES an enclosing mark
        # anywhere — leading '(benzylsulfanyl)methyl', interior
        # 'bromo(phenyl)methyl', or trailing-stem
        # '[(4-methoxyphenyl)methoxy]methyl' — but is not itself fully
        # wrapped. Such a name is a compound substituent by construction
        # (regardless of is_complex_substituent, which keys on digits/
        # hyphens and misses these shapes). Citing it BARE drops structure
        # (OPSIN parses 'bromo(phenyl)methylbenzene' as bromo + benzyl,
        # TWO substituents) and plain parens double the mark
        # ('((...)methyl)', non-PIN). Escalate to the next bracket level.
        # Fusion/von-Baeyer/ring-assembly brackets remain nesting-IGNORED
        # inside apply_enclosing_marks, so
        # 'furo[3,2-b]pyridin-2-yl' still takes plain parentheses.
        formatted_name = apply_enclosing_marks(name, -1)
    elif (complex or derived_multiplier) and not name.startswith(('(', '[', '{')):
        # Complex name (or a simple name taking a derived bis/tris/...kis
        # multiplier, without existing enclosing marks: wrap.
        # Decision A part 2 (lysopine): a name that arrives already enclosed in
        # braces ('{[(1R)-1-carboxyethyl]amino}', the third mark of the
        # order, the Blue Book) is fully enclosed -- the elif above has
        # already taken every name with an inner mark that is NOT fully enclosed
        # -- so wrapping it again gave the extra level
        # '2-({[(1R)-1-carboxyethyl]amino})'.
        formatted_name = f"({name})"
    else:
        formatted_name = name

    # (the Blue Book, nesting order "{[({})]}"): a name its
    # producer already wrapped in plain parentheses (the benzene amido branch
    # returns '((2R)-2-chloropropanamido)') reached the else branch above
    # unchanged, so '(' sat directly around the descriptor's '(' --
    # '4-((2S,3R)-...-4-oxobutanamido)benzamido' (TRIAGE g3 C10c). The outer
    # mark is re-chosen from the content, as for every other compound prefix;
    # a content at depth 0 keeps its '(' (no change).
    if formatted_name is name and name.startswith('(') and _is_fully_enclosed(name):
        _remarked = apply_enclosing_marks(name[1:-1], -1)
        if _remarked[0] != '(':
            formatted_name = _remarked

    # Wave2 T5a: a simple multiplier joining an italicized-prefix-led name
    # keeps the hyphen boundary: '1,2-di-tert-butylbenzene' (PIN, BB
    #.x example list), never 'ditert-butyl'. Parenthesized names
    # never start with 'tert-'/'sec-', so the bis/tris path is unaffected.
    #: a DIFFERENT rule from the enclosure carve-out (this one inserts a
    # hyphen, that one withholds marks) but the same DETECTION, so it shares the
    # one primitive rather than open-coding a fifth startswith. The leg now has
    # its own named primitive (multiplier_needs_hyphen) because a SECOND producer
    # -- organometallics._ligand_token -- needed it and was shipping the
    # malformed 'ditert-butyl' for want of it.
    # -FIX Item 4: the hyphen now comes from `get_multiplier_prefix` itself
    # (the one place that has both the count and the name), so this second copy
    # is gone -- keeping it would emit `di--tert-butyl`. The assertion below is
    # the tripwire that the primitive really did it.
    if multiplier and multiplier_needs_hyphen(formatted_name):
        assert multiplier.endswith('-'), (
            "get_multiplier_prefix must supply the P-16.2.4.1(d) hyphen: "
            "%r + %r" % (multiplier, formatted_name))

    # Assemble: locants-multiplier+name. An empty locant list (elided per
    #, e.g. a mononuclear parent: phenylmethanol) takes no hyphen.
    #
    # Lane L2 proper-fix a performance pass (review finding F1): every return above this
    # point is a plain ``str`` (f-string / ``apply_enclosing_marks``), which
    # drops the writer's ``assembly.prefix_derivation`` record even when
    # ``name`` carried one on entry (c)
    # substitution). A caller that groups several occurrences of the SAME
    # substituent AFTER this function has formatted each one individually
    # (``composer._merge_duplicate_prefixes``) then re-derives the multiplier
    # from the finished text with no record left to read -- the mechanism
    # ``carried`` exists for exactly this ("a writer step that adds marks or
    # a descriptor to a recorded name keeps the record",
    # ``assembly/prefix_derivation.py``). Thread the INPUT's record onto the
    # fully-formatted output so it survives past this writer.
    from .prefix_derivation import carried
    if not locant_str:
        return carried(f"{multiplier}{formatted_name}", like=name)
    return carried(f"{locant_str}-{multiplier}{formatted_name}", like=name)


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

# Pre-sorted prefix list for alpha_sort_key — longest first to avoid partial matches
_SORTED_ALPHA_PREFIXES = sorted(IGNORE_FOR_ALPHA - {"sec", "tert"},
                                key=len, reverse=True)


def _numeral_chain_stems() -> frozenset:
    """Chain stems that a multiplicative prefix in ``IGNORE_FOR_ALPHA`` shadows.

    A numeral prefix like ``penta`` is the chain stem ``pent`` plus the linking
    vowel ``a``, so ``pentan-2-yl`` starts with ``penta`` by coincidence. Derived
    from ``data.chain_names`` rather than hardcoded, so a change to the stem table
    cannot silently desynchronise this guard.
    """
    from ..data.chain_names import get_chain_prefix
    stems = set()
    for n in range(1, 41):
        try:
            stems.add(get_chain_prefix(n))
        except Exception:                      # noqa: BLE001 — table bound
            break
    return frozenset(s for s in stems
                     if s + 'a' in IGNORE_FOR_ALPHA)


_NUMERAL_CHAIN_STEMS = _numeral_chain_stems()


def _strip_leading_marks_and_locants(text: str) -> str:
    """Drop every leading enclosing mark, locant and indicated-hydrogen token of a
    composed prefix, so the key starts at the first LETTER of its complete name
    : '[(2-aminoethoxy)hydroxyphosphoryl]oxy' -> 'aminoethoxy)...',
    '(3,4,5-trihydroxybenzoyl)oxy' -> 'trihydroxybenzoyl)oxy'. The remaining marks
    are removed by alpha_sort_key."""
    while True:
        new = text.lstrip('([{')
        new = _INDICATED_H_PREFIX_RE.sub('', _LOCANT_PREFIX_RE.sub('', new))
        if new == text:
            return text
        text = new


def _alpha_sort_key_core(substituent_name: str) -> str:
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

    # preamble, last sentence (the Blue Book). Applied FIRST and
    # GLOBALLY, before any other stripping, for two reasons:
    # * a descriptor sits AHEAD of the locant it qualifies ('(1E,3E,5E)-hepta-
    # 1,3,5-trien-1-yl'), so the locant strip below cannot see the locant
    # until the descriptor is gone;
    # * `prefix_citation_sort_key` builds its tier-1 key from EVERY alpha
    # character this function returns, so an INTERNAL descriptor
    # ('2-[(2Z)-pent-2-en-1-yl]cyclopentyl') contributes a stray 'z' unless it
    # is removed wherever it occurs, not merely at the front.
    # Conjunctive names are the rule's own carve-out; this engine does not emit
    # them is not a PIN construction here), so the exception cannot fire.
    text = strip_alphanumerical_noise(text)

    # Leading positional descriptors are ALWAYS ignored for alphabetization
    #, under `### **** ALPHANUMERICAL ORDER` at BB 3436:
    # "*The name of a prefix for a substituent is considered to begin with the
    # first letter of its complete name*", BB 3477, example
    # `7-(1,2-difluorobutyl)-5-ethyltridecane (PIN)`; a locant is not a letter.
    # NOT, whose heading is `## **** Lowest set of locants`
    # (BB 3189) and which carries no alphabetization content at all -- the one
    # site in src/ where a citation sweep misfired on the word "locant". Line
    # 2018 below, in this same function, always cited for the same
    # subject.): a parent locant ('3-'), a compound locant set ('2,4-'), and
    # the italic indicated-hydrogen descriptor ('1H-'). Strip them up front so
    # the enclosed-vs-simple decision below sees the bare substituent name —
    # this is what makes '3-(1H-imidazol-5-yl)' and '(1H-imidazol-5-yl)' yield
    # the same key, and '2-(2,4-dihydroxyphenyl)' route to the complete-name
    # branch just like the bare '(2,4-dihydroxyphenyl)'.
    text = _LOCANT_PREFIX_RE.sub('', text)
    text = _INDICATED_H_PREFIX_RE.sub('', text)

    # (a phase assembly/parenthesisation fix,): a COMPOUND substituent cited as a
    # fully-enclosed unit is alphabetized on the first letter of its COMPLETE
    # name — its INTERNAL multiplying prefix (di/tri…) is part of the name and
    # is NOT ignored ('(2,4-dimethylpentyl)' sorts at 'd', before 'ethyl';
    # '2-(2,4-dihydroxyphenyl)' sorts at 'd', before 'hydroxy'). Contrast
    #: a bare 'di'/'tri' multiplying SEPARATE simple prefixes on the
    # parent IS ignored ('2,2-dimethyl' sorts at 'm'). The enclosing marks
    # disambiguate the two rules, so only fully-enclosed input takes the
    # complete-name path.
    if (text.startswith('(') and text.endswith(')')) or (text.startswith('[') and text.endswith(']')):
        inner = text[1:-1]
        inner = _LOCANT_PREFIX_RE.sub('', inner)       # drop the inner positional locant
        inner = _INDICATED_H_PREFIX_RE.sub('', inner)  # and a leading '1H-' descriptor
        if (inner.startswith('(') and inner.endswith(')')) or (inner.startswith('[') and inner.endswith(']')):
            inner = inner[1:-1]
        elif inner[:1] in '([{':
            # an enclosed COMPOSED prefix, '[(2-chloroethyl)amino]': see below
            inner = _strip_leading_marks_and_locants(inner)
        return inner  # complete name; internal multiplying prefix NOT stripped

    # again, for a COMPOSED prefix whose organyl sits in its own marks
    # with the composing suffix outside them -- '(3,4,5-trihydroxybenzoyl)oxy',
    # '(6-sulfanylhexyl)oxy', '(2-chloroethyl)sulfanyl'. It too
    # "is considered to begin with the first letter of its complete name"
    # (the Blue Book): 't', 's', 'c' -- never the inner locant, which used to lead the
    # key once the marks were dropped, sorting '3,4,5-trihydroxybenzoyloxy'
    # ahead of every lettered prefix ('hydroxy'). Complete name: the internal
    # multiplying prefix is kept.
    if text[:1] in '([{':
        return _strip_leading_marks_and_locants(text)

    # ---- Simple (non-enclosed) prefix: ----
    # Strip N-locant prefixes (N- or N,N-)
    # e.g., "N,N-dimethylamino" -> "dimethylamino" -> "amino" (after multi-prefix strip)
    text = _N_LOCANT_PREFIX_RE.sub('', text)

    # Handle hyphenated detachable prefixes: sec- and tert- — the
    # italicized prefix is ignored, so 'tert-butyl' sorts at 'b'). A DIFFERENT
    # rule from the enclosure carve-out, but the same DETECTION, so it calls the
    # one primitive instead of re-deriving the literal tuple 1400 lines below the
    # constant (this was the copy the tripwire's regex shape could not see).
    _text_remainder, _text_had_italicized = strip_italicized_structural_prefix(text)
    if _text_had_italicized:
        return _text_remainder

    # Handle non-hyphenated multiplicative prefixes (di-, tri-,...): for a
    # SIMPLE prefix these are ignored. Sort by longest prefix first
    # to avoid partial matches (e.g., 'tetra' before 'tri'). The result is a
    # sort key only, never reconstructed.
    for prefix in _SORTED_ALPHA_PREFIXES:
        if text.startswith(prefix):
            remainder = text[len(prefix):]
            # Only strip if there is a remainder (avoid stripping entire word)
            if remainder:
                # 'di-tert-butyl': a multiplier on an italicized-prefix name. Both
                # are ignored -- (:3448) the multiplier, /
                # (:3477/:3495) the italic 'tert' -- so it keys at
                # 'butyl', like the bare 'tert-butyl' above ('4-butyl-4-tert-
                # butylcyclohexan-1-ol (PIN)',:3465). The remainder '-tert-butyl'
                # used to be returned as the key.
                if remainder.startswith('-'):
                    _it_rest, _it_had = strip_italicized_structural_prefix(remainder[1:])
                    if _it_had:
                        return _it_rest
                # 'di(prop-1-en-2-yl)', 'bis(2-chloroethyl)': the multiplier
                # multiplies the WHOLE enclosed unit, so (:3448)
                # ignores it and the unit keys by its own complete name
                #,:3477) -- 'propenyl', 'chloroethyl'. The '(' guard
                # below is for a multiplier carved out of the MIDDLE of a
                # compound name; a fully enclosed remainder is not that.
                if _is_fully_enclosed(remainder):
                    return _alpha_sort_key_core(remainder)
                # -FIX Item 7: a numeral that is part of the substituent's
                # own STEM is not a multiplicative prefix. `pentan-2-yl` is
                # `pent` + `an-2-yl`, not `penta` + `n-2-yl`; stripping gave the
                # key `n-2-yl`, which sorted it before `octyl` and (because
                # `prefix_citation_sort_key` delegates here, and (g) gives
                # the lowest locant to the prefix cited FIRST) also inverted the
                # locants. `### **** ALPHANUMERICAL ORDER` -> `****`
                # (the Blue Book) only lets a MULTIPLICATIVE prefix be
                # ignored; `****` (:3477) says a prefix "is considered to
                # begin with the first letter of its COMPLETE name", and the
                # complete name here begins `p`.
                #
                # Two structural tells, neither a name list:
                # (a) a multiplicative prefix multiplies a substituent NAME, so
                # it is never followed by `-<digit>` -- that hyphen-locant
                # belongs to the stem's own unsaturation
                # (`penta-1,3-dien-1-yl` is pent + a + `-1,3-dien-1-yl`).
                # `di-tert-butyl` is unaffected: its hyphen precedes a
                # LETTER, not a digit.
                # (b) the candidate is `<chain stem>` + the linking `a`, and the
                # name continues with the alkane/alkene/alkyne morphology
                # `an`/`en`/`yn` -- `pentan-`, `hexane-`, `decan-`. A genuine
                # multiplied prefix never continues that way.
                if re.match(r'-\d', remainder):
                    return text
                if prefix.endswith('a') and prefix[:-1] in _NUMERAL_CHAIN_STEMS \
                        and text[len(prefix) - 1:].startswith(('an', 'en', 'yn')):
                    return text
                # Wave2 T1c: do NOT strip when the "multiplier" is really the
                # start of a numeric CHAIN STEM — there it is part of the name
                # and alphabetizes: tridecyl/octadecyl ('tri'/'octa'
                # + 'dec...'), pentacosyl ('cos...'), triacontyl ('acont...'),
                # pentanamido/octadecanamido ('penta'/'octa' + 'nam...'),
                # decanoyloxy ('deca' + 'noyl...'). A genuine multiplied
                # prefix never continues with these stems.
                if remainder.startswith(
                        ('dec', 'cos', 'cont', 'acont', 'nam', 'noyl')):
                    return text
                # Wave2 T2b: 'diazenyl' (HN=N-, and 'diazo'
                # — the leading 'di' is structural (the two
                # nitrogens of diazene/diazo), never a multiplier; the BB
                # multiplies them with bis ('bis(diazenyl)', not
                # 'didiazenyl'). Exact-remainder match so 'diazido'
                # (a genuine 2x azido) still strips to 'azido'.
                if remainder in ('azenyl', 'azo'):
                    return text
                # vs, decided STRUCTURALLY rather than by
                # whether the caller happened to wrap the string.
                #
                # ignores a multiplicative prefix that multiplies
                # SEPARATE simple prefixes on the parent ('2,2-dimethyl' -> 'm').
                # counts one INTERNAL to a single compound substituent,
                # which "is considered to begin with the first letter of its
                # complete name" (:3477) -- the BB's own
                # `7-(1,2-difluorobutyl)-5-ethyltridecane (PIN)` alphabetizes the
                # compound prefix at 'd', ahead of 'ethyl'.
                #
                # The enclosure test above is only a PROXY for that distinction
                # and it fails whenever a caller passes the bare name -- which
                # they do. `5,9-dimethyl-2-(propan-2-ylidene)deca-4,8-dien-1-yl`
                # keyed at 'm' instead of 'd' and was cited last instead of
                # first. It had been masked: the prefix also carried a `(4E)`
                # whose '(' sorted ahead of every letter and accidentally
                # restored the right order, so removing the descriptor per:3446
                # exposed it. Invariant 9, an unmasked generator.
                #
                # Strip ONLY when the remainder is a BARE SIMPLE PREFIX. Two
                # independent tells are needed; neither alone separates the
                # cases actually observed:
                # * `is_substituted_substituent` -- the predicate that already
                # answers "is this prefix substituted?" for 's bis/di
                # choice. Catches 'hydroxypentyl' (from '4,5-dihydroxypentyl'),
                # whose own locants the leading-locant strip already consumed.
                # * residual LOCANTS or enclosing marks -- what is left when a
                # multiplier is carved out of the middle of a compound name.
                # Catches 'methyl-2-(propan-2-ylidene)deca-4,8-dien-1-yl',
                # which the predicate reads as unsubstituted because it is a
                # fragment rather than a well-formed name.
                # A leading-locant test cannot serve as a third: '2,2-dimethyl'
                # and '4,5-dihydroxypentyl' both carry one and go opposite ways.
                if is_substituted_substituent(remainder) \
                        or any(c.isdigit() for c in remainder) \
                        or '(' in remainder or '[' in remainder:
                    return text
                return remainder

    return text


# Enclosing marks are TYPOGRAPHY, not alphanumerical content.
# orders "nonitalic Roman letters" and then numerals (:3442); a bracket is
# neither, so it may not enter a sort key.
_ENCLOSING_MARKS_RE = re.compile(r'[()\[\]{}]')


# Letters that are not tier-1 letters: the italic structural prefixes
# 'sec'/'tert' /,:3477/:3495, '1-(butan-2-yl)-3-tert-
# butylbenzene (PIN)') and an ISOLATED single letter -- a letter locant or
# italic element ('N', '4a', '1H', 's-indacene'), which the preamble
# (:3442) excludes: letters are considered first "unless used as locants or
# part of a compound or composite locant". No word of a substituent name is a
# single letter, so an isolated letter is always one of these.
_NON_TIER1_LETTERS_RE = re.compile(
    r"(?<![a-z])(?:sec|tert)(?=-)|(?<![a-z])[a-z](?![a-z])", re.IGNORECASE)


def _key_letters(text: str) -> str:
    """The Roman LETTERS of a sort-key text, in order tier 1)."""
    return ''.join(ch for ch in _NON_TIER1_LETTERS_RE.sub('', text) if ch.isalpha())


class AlphaKey(str):
    """The text of a sort key, ORDERED letters first.

    ``### **** ALPHANUMERICAL ORDER`` (``the Blue Book``):
    "*Nonitalic Roman letters are considered first... When all the Roman
    letters are identical, the set of locants... are compared*"; ``****``
    (``:3477``): "*The name of a prefix for a substituent is considered to begin
    with the first letter of its complete name*". A plain string comparison of
    the key text lets a hyphen or a digit decide against a letter, because both
    sort below every lowercase letter in ASCII: ``'prop-1-en-2-yl'`` <
    ``'propan-2-ylidene'`` (``'-'`` < ``'a'``), although the letters say
    ``propanylidene`` < ``propenyl``; ``'hydroxy-4-methylpentyl'`` <
    ``'hydroxymethyl'``, although ``hydroxymethyl`` is an initial segment of
    ``hydroxymethylpentyl`` and comes first (``:21663``,
    ``4-methyl-3-methylidenehexanoic acid (PIN)``). The Blue Book's own
    ``5-(butan-2-yl)-5-butylhentriacontane (PIN)`` (``:3461``) is the
    letter-by-letter reading.

    The value IS the key text (so ``alpha_sort_key('dimethyl') == 'methyl'``
    and every inspection of the text is unchanged); only the ORDER differs from
    ``str``, tier by tier:

    1. the Roman letters (``_key_letters``) -- (:3442), (:3477);
    2. the italic structural prefixes of the ORIGINAL name, absence first --
       ``****`` (:3495) "When... Roman letters do not permit a
       decision..., italicized letters are considered": '4-butyl-4-tert-
       butylcyclohexan-1-ol (PIN)' (:3465). The key text of 'butyl' and
       'tert-butyl' is 'butyl' for both, so without this tier ``sorted``
       kept the input order and one molecule got both spellings;
    3. the key text, which carries the locants when the letters tie
       ,:3517);
    4. the original name -- not a rule, it only keeps the order total.

    ``prefix_citation_sort_key`` tier 1 is the same letters string. Python gives
    a subclass's reflected comparison priority, so a plain-string sentinel on
    the left (``'zzzzz' < key``) is ordered the same way. Equality with a plain
    string is string equality (``alpha_sort_key('tert-butyl') == 'butyl'``);
    two keys are equal only when every tier is.
    """

    def __new__(cls, text: str, original: str = ''):
        obj = super().__new__(cls, text)
        obj._original = original
        return obj

    def _order(self):
        orig = self._original
        return (_key_letters(self), _italic_prefixes(orig), str(self), orig)

    @staticmethod
    def _other_order(other):
        if isinstance(other, AlphaKey):
            return other._order()
        if isinstance(other, str):
            return (_key_letters(other), (), other, other)
        return None

    def __lt__(self, other):
        o = self._other_order(other)
        return NotImplemented if o is None else self._order() < o

    def __le__(self, other):
        o = self._other_order(other)
        return NotImplemented if o is None else self._order() <= o

    def __gt__(self, other):
        o = self._other_order(other)
        return NotImplemented if o is None else self._order() > o

    def __ge__(self, other):
        o = self._other_order(other)
        return NotImplemented if o is None else self._order() >= o

    def __eq__(self, other):
        if isinstance(other, AlphaKey):
            return self._order() == other._order()
        return str.__eq__(self, other)

    def __ne__(self, other):
        eq = self.__eq__(other)
        return eq if eq is NotImplemented else not eq

    __hash__ = str.__hash__


_ITALIC_PREFIX_RE = re.compile(r"(?<![a-z])(sec|tert)-")


def _italic_prefixes(name: str) -> tuple:
    """The italic structural prefixes ('sec', 'tert') of a substituent name, in
    order of appearance,:3495)."""
    return tuple(_ITALIC_PREFIX_RE.findall(name.lower())) if name else ()


def alpha_sort_key(substituent_name: str) -> str:
    """ alphanumerical sort key for a substituent prefix.

    Thin wrapper over:func:`_alpha_sort_key_core` that removes enclosing marks
    from the finished key.

    ⚠ **This strip is load-bearing, not cosmetic.** The core returns the key
    with any marks the branch logic left in place, and a leading mark decides
    comparisons outright because ``(`` is ASCII 40 and ``[`` is 91 -- both below
    every lowercase letter. ``alpha_sort_key('4-[(1R)-1-chloroethyl]phenoxy')``
    used to yield ``'[1-chloroethyl]phenoxy'``, which sorts ahead of
    ``'chloroethyl'`` and, through (g), took locant 1 -- the wrong
    numbering for gold row W2F-P8-01. The defect was masked while
    stereodescriptors were still in the key, since their ``(`` sorted earlier
    still. ``tests/unit/rules/test_anilino_preferred_prefix.py`` had already
    recorded the same leak from the ``(`` side.

    The key is only ever COMPARED, never reconstructed into a name, so dropping
    marks cannot affect any emitted string.

    The result is an:class:`AlphaKey`: its text is the key, its ORDER is
    's (letters first, then the text), so ``'propan-2-ylidene'`` sorts
    before ``'prop-1-en-2-yl'`` and ``'hydroxymethyl'`` before
    ``'1-hydroxy-4-methylpentyl'``.

    Examples:
        >>> alpha_sort_key('4-[(1R)-1-chloroethyl]phenoxy')
        'chloroethylphenoxy'
        >>> alpha_sort_key('(2-chloroethyl)')
        'chloroethyl'
        >>> alpha_sort_key('methyl')
        'methyl'
        >>> alpha_sort_key('propan-2-ylidene') < alpha_sort_key('prop-1-en-2-yl')
        True
    """
    return AlphaKey(_ENCLOSING_MARKS_RE.sub('', _alpha_sort_key_core(substituent_name)),
                    strip_alphanumerical_noise(substituent_name.lower()))


# ---------------------------------------------------------------------------
# (j) / -- the CIP tie-break.
#
# `****`'s numbering cascade ends with clause (j) (the Blue Book),
# verbatim:
#
# (j) When there is a choice for lower locants related to the presence of
# stereogenic centers or stereoisomers, the lower locant is assigned to CIP
# stereodescriptors *Z*, *R*, *M*, and *r* (pseudoasymmetry) that are
# preferred to *E*, *S*, *P*, and *s*, respectively, which are preferred to
# the non-CIP stereodescriptors *cis*, *trans*, or *r* (reference), *c*, and
# *t*
#
# and `****` (:22606) states the citation-order half: "When names based
# on alphanumerical order and isotopic descriptors are the same, further choice
# depends on the alphabetic order of the stereochemical descriptors 'R' and 'S'."
#
#:22589 specifies the tier ORDER outright -- "since the alphabetic characters
# and locants (ignoring the configuration symbols) are identical the
# configurational symbols are compared and 'R' precedes 'S'" -- i.e. letters,
# then locants, THEN configuration. That is why this is tier 3 and the
# full-string engineering fallback moves to tier 4.
#
# ⚠ THREE of the four pairs happen to be alphabetical, but `Z` before `E` is
# NOT::45363 says "'Z' is senior to 'E', **irrespective of the alphabetical
# order**". A string comparison silently gets that one backwards, which is why
# this is an explicit rank table and not `sorted`.
#
# Worked examples this reproduces, both printed as (PIN):
#:22609 1-[(1R)-1-bromoethyl]-1-[(1S)-1-bromoethyl]cyclopentane
#:22597 4-{3,4-bis[(1R)-1-chloroethyl]phenoxy}-1,2-bis[(1S)-1-chloroethyl]benzene
#
# (i) puts isotopic modifications AHEAD of (j) for low locants. Their
# locants already reach tier 2, so no separate tier is added here; a case that
# needs one would be an isotope-vs-isotope choice, which this engine has never
# emitted.
_CIP_DESCRIPTOR_RANK = {
    # CIP, preferred member of each pair -> 0
    'z': 0, 'r': 0, 'm': 0,
    'seqcis': 0,
    # CIP, other member -> 1
    'e': 1, 's': 1, 'p': 1,
    'seqtrans': 1,
    # non-CIP -> 2, after every CIP descriptor
    'cis': 2, 'trans': 2, 'c': 2, 't': 2,
}

# A configuration block: '(1R)', '(2S,3R)', '(E)', '(1r,4r)', '(2Z,6E)' -- OR a
# bare hyphenated non-CIP descriptor, which carries no enclosing marks at all
# ('cis-4-methylcyclohexyl'). Both forms are found by ONE pattern so the ranks
# come back in order of appearance even when a name mixes them.
_CIP_BLOCK_RE = re.compile(
    r"\((?:[0-9]+[a-z]?[rsezmp*]|[rsezmp]|seqcis|seqtrans|rs|sr|[,\-+ ])+\)"
    r"|(?:^|(?<=[\[({-]))(?:cis|trans)(?=-)",
    re.IGNORECASE,
)
_CIP_TOKEN_RE = re.compile(r"seqcis|seqtrans|cis|trans|[rsezmp]", re.IGNORECASE)


def cip_descriptor_rank_key(prefix: str) -> tuple:
    """(j) rank of a prefix's configurational symbols, in order.

    Lower sorts first, i.e. is cited first and takes the lower locant. Returns
    an empty tuple when the prefix carries no configuration, so a prefix WITH a
    descriptor never outranks one without on this tier alone -- absence ties
    with absence and the comparison falls through.

    Examples:
        >>> cip_descriptor_rank_key('(1R)-1-chloroethyl')
        (0,)
        >>> cip_descriptor_rank_key('(1S)-1-chloroethyl')
        (1,)
        >>> cip_descriptor_rank_key('(2Z)-pent-2-en-1-yl')
        (0,)
        >>> cip_descriptor_rank_key('(2E)-pent-2-en-1-yl')
        (1,)
        >>> cip_descriptor_rank_key('methyl')
        
    """
    ranks = []
    for block in _CIP_BLOCK_RE.findall(prefix or ''):
        for tok in _CIP_TOKEN_RE.findall(block):
            rank = _CIP_DESCRIPTOR_RANK.get(tok.lower())
            if rank is not None:
                ranks.append(rank)
    return tuple(ranks)


def cip_locant_rank_key(items) -> tuple:
    """(j) NUMBERING key of ONE candidate numbering: descriptors WITH locants.

    ``items`` is an iterable of ``(locant, code)`` pairs, one per stereodescriptor
    the name would cite under that numbering (atom ``_CIPCode`` values R/S/r/s/M/P,
    bond ``_CIPCode`` values Z/E). ``locant`` may be any sortable value, as long as
    one call site passes one kind (ints for rings and chains, ``(int, str)`` for
    natural-product locants such as ``3a`` or ``4'``). Lower key = preferred
    numbering.

    Why a second key next to:func:`cip_descriptor_rank_key`: that one ranks the
    CODES of one prefix in order of appearance -- right for citation order
    , where the locants are already equal. As a NUMBERING key it throws
    the locants away, so a lone descriptor ties across every numbering and the
    input atom order decides (``(1r)-``, ``(3r)-`` and ``(5r)-1,3,5-trimethyl-
    cyclohexane`` for one molecule, one per SMILES spelling).

    ``### **** NUMBERING`` clause (j) (``the Blue Book``): "*the lower
    locant is assigned to CIP stereodescriptors Z, R, M, and r (pseudoasymmetry)
    that are preferred to E, S, P, and s, respectively*". Two tiers:

    1. the ``(locant, rank)`` pairs in locant order, compared at the first point of
       difference, so a descriptor at a lower locant wins and, at an equal locant,
       the preferred member of its pair wins. The Blue Book's own reading of "first
       point of difference": ``(2Z,4S,8R,9E)-undeca-2,9-diene-4,8-diol (PIN)``
       (``:3403``) "*the choice is between 'E' and 'Z' for position '2', not
       between 'R' and 'S' for position '4'*". With equal locant sets this orders
       exactly as the rank tuple of:func:`cip_descriptor_rank_key` did.
    2. only when tier 1 ties exactly: chiral (upper-case) before pseudoasymmetric
       (lower-case) at the first point of difference -- Sequence Rule 4a,
       ``****`` (``:45459``) "*Chiral stereogenic units precede
       pseudoasymmetric stereogenic units*"; this reproduces hexachlorocyclohexane
       isomer 2, ``(1R,2R,3r,4S,5S,6s)`` (``:48131``), not ``(1r,2R,3R,4s,5S,6S)``.

    Returns ```` when no item carries a known descriptor, so stereo-free
    numberings tie exactly as before.

    Examples:
        >>> cip_locant_rank_key([(1, 'r')]) < cip_locant_rank_key([(3, 'r')])
        True
        >>> cip_locant_rank_key([(2, 'Z'), (4, 'S'), (8, 'R'), (9, 'E')]) < \
cip_locant_rank_key([(2, 'E'), (4, 'R'), (8, 'S'), (9, 'Z')])
        True
        >>> cip_locant_rank_key()
        
    """
    rows = sorted(
        (loc, _CIP_DESCRIPTOR_RANK[code.lower()], 0 if code[:1].isupper() else 1)
        for loc, code in items
        if code and code.lower() in _CIP_DESCRIPTOR_RANK
    )
    if not rows:
        return ()
    return (tuple((loc, rank) for loc, rank, _kind in rows),
            tuple(kind for _loc, _rank, kind in rows))


def prefix_citation_sort_key(prefix: str, *,
                             parent_locants: bool = False) -> tuple:
    """ alphanumerical order for citation, as a TOTAL order.

    Three tiers, consulted in turn:

    1. the Roman LETTERS only. ``### **** ALPHANUMERICAL ORDER``'s preamble
       (``the Blue Book``) puts "*Nonitalic Roman letters... first*", and
       ``****`` (``:3448``) makes multiplicative prefixes not alter the
       order already established. Digits and hyphens are dropped, so
       identical-letter prefixes (``pentan-2-yl`` vs ``pentan-3-yl``) tie here on
       purpose and tier 2 decides.
    2. the prefix's OWN locants, in ORDER OF APPEARANCE. ``****``
       (``:3517``): "*When two or more prefixes consist of identical Roman
       letters, priority for order of citation is given to the group that
       contains the lowest locant(s) at the first point of difference*", whose
       first example is this very pair — ``4-(2-methylbutyl)-N-(3-methylbutyl)-
       aniline (PIN)``, "*for ordering the substituents '2' is lower than '3'*"
       (``:3521``). Order of appearance rather than a sorted set, because
       ``:3533`` prefers ``1-(2-methylpentan-3-yl)-1-(3-methylpentan-2-yl)cyclo-
       pentane (PIN)`` on the ground that "*the locant set '2,3' is lower than
       '3,2'*". Compared via ``locant_sort_key``, so locant 2 precedes locant 10.
    3. the CONFIGURATIONAL symbols, in order of appearance, ranked by
       ``****`` clause (j) (``:3346``) / ``****`` (``:22606``).
       ``:22589`` puts this tier exactly here: "*since the alphabetic characters
       and locants (ignoring the configuration symbols) are identical the
       configurational symbols are compared and 'R' precedes 'S'*". Before this
       tier existed the descriptor was doing this job from INSIDE tier 1, which
        (``:3446``) forbids -- and:44601 says outright that capitalized
       CIP descriptors "are written in italics to indicate that they are not
       involved in the primary stage of alphanumerical order".
    4. the full string. NOT a nomenclature rule -- an engineering requirement.
       Tiers 1-2 are not injective (``cyclohexylmethyl`` vs a differently-spelled
       prefix with the same letters and no locants), and a tie in a ``sorted``
       whose input order came from a ``Counter`` over RDKit neighbours resolves to
       HOW THE SMILES WAS WRITTEN. That is how one molecule got two names:
       ``CCC(C)CSSSCCC(C)C`` and its own re-spelling gave
       ``1-(2-methylbutyl)-3-(3-methylbutyl)trisulfane`` and
       ``1-(3-methylbutyl)-3-(2-methylbutyl)trisulfane``. This tier can only ever
       be reached once has been exhausted, so it decides nothing the Blue
       Book decides -- it only stops atom order from deciding.

    ``parent_locants`` selects which of the TWO input conventions this function is
    being handed, because the callers genuinely differ and the leading locant means
    opposite things in them:

    * ``False`` (default) -- a BARE substituent name (``2-methylbutyl``,
      ``pentan-2-yl``). Its leading locant is the substituent's OWN and is exactly
      what compares.
    * ``True`` -- an already-RENDERED prefix string (``3,5-dichloro``,
      ``2-methyl``) as produced by ``format_substituent_prefix``. Its leading
      locants are PARENT locants, which are assigned BY citation order and
      therefore may not decide it; they are stripped before tier 2.

    Passing the wrong convention was the whole defect: one strip served both, so
    the bare names lost the locant needs and keyed to ``('methylbutyl',
    )`` apiece.
    """
    from .name_comparison import _LOCANT_TOKEN_FINDER, locant_sort_key
    alpha_letters = _key_letters(alpha_sort_key(prefix))
    body = _LOCANT_PREFIX_RE.sub('', prefix) if parent_locants else prefix
    locs = tuple(locant_sort_key(t)
                 for t in _LOCANT_TOKEN_FINDER.findall(body))
    return (alpha_letters, locs, cip_descriptor_rank_key(prefix), prefix)


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
    IUPAC / (a) vowel elision applied.

    The final letter 'a' of a multiplying prefix ('tetra', 'penta', 'hexa', …)
    is elided before the suffix '-ol' (tetra+ol → tetrol, hexa+ol → hexol;
    Blue Book: ``benzenehexol``, ``cyclohexane-1,2,3,4-tetrol``) and
    before the suffix '-amine' (tetra+amine → tetramine; **Blue Book
    states this verbatim**: "The terminal letter 'a' of a numerical prefix is
    elided before the suffix amine, i.e., 'tetramine', not 'tetraamine'",
    e.g. ``[1,1'-biphenyl]-3,3',4,4'-tetramine``, ``silanetetramine``).

    Also elided before the ketone suffix '-one' (tetra+one → tetrone, penta+one →
    pentone): **Blue Book (1) states this verbatim** — "the final letter
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
        >>> format_suffix_with_locants("pent", "an", "oic acid", )
        'pentanoic acid'
        >>> format_suffix_with_locants("prop", "an", "al", )
        'propanal'
    """
    # Build the base: stem + unsaturation (e.g., 'propan', 'buten')
    base = parent_stem + unsaturation

    # (b) (`:7104`): the 'thioic acid'/'dithioic acid' suffixes and their
    # Se/Te analogues are multiplied with the DERIVED multiplier ('bis'/'tris'/…)
    # AND enclosed in parentheses, and the italic chalcogen locant is OMITTED
    # inside the enclosure — `hexanebis(thioic acid)` (`:30231`),
    # `ethanebis(dithioic acid)` (`:30317`), NOT `hexanebisthioic O-acid`.
    # Gated on the same closed exception family that `get_suffix_multiplier_prefix`
    # already routed to 'bis' (`suffix_takes_derived_multiplier`), so it fires for
    # exactly those suffixes and nothing else (a) 'dioic acid' still
    # takes 'di', unenclosed). The italic locant is dropped with the shared
    # infix regex so the drawn tautomer collapses to the "normally omitted"
    # acid spelling `:30215`; OPSIN round-trips both to one structure).
    if suffix_multiplier and suffix_takes_derived_multiplier(suffix):
        inner = ' '.join(_CHALCOGEN_LOCANT_INFIX_RE.sub('', suffix).split())
        if suffix_locants:
            locant_str = ",".join(str(loc) for loc in suffix_locants)
            return f"{base}e-{locant_str}-{suffix_multiplier}({inner})"
        return f"{base}e{suffix_multiplier}({inner})"

    # /: even at count 1 (no multiplier) 'dithioic acid' and
    # 'dithioate' are ENCLOSED so the 'di' of 'dithio-' is not misread as a
    # multiplicative 'di' on the suffix -- BB '...prop-2-ene(dithioic acid) (PIN)'
    # (:30275), 'sodium propane(dithioate) (PIN)' (:31569). Plain 'thioic acid' at
    # count 1 stays BARE ('hexanethioic O-acid',:30224), so the count=1 wrap is
    # scoped to EXACTLY these two 'di'-chalcogen tokens, never the wider
    # bis-exception family. The italic chalcogen locant is dropped with the shared
    # infix regex (the "normally omitted" acid spelling,:30215).
    if not suffix_multiplier:
        _di_norm = ' '.join(_CHALCOGEN_LOCANT_INFIX_RE.sub('', suffix).split())
        if _di_norm in ("dithioic acid", "dithioate"):
            if suffix_locants:
                locant_str = ",".join(str(loc) for loc in suffix_locants)
                return f"{base}e-{locant_str}-({_di_norm})"
            return f"{base}e({_di_norm})"

    if suffix_locants:
        # Format locants as comma-separated
        locant_str = ",".join(str(loc) for loc in suffix_locants)

        # Build the suffix part: multiplier + suffix (e.g., 'diol', 'tetrol').
        # elides multiplier-final 'a' before '-ol' (tetra+ol → tetrol).
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
            # IUPAC (a) vowel elision: drop terminal 'e' of the parent stem
            # only when the suffix begins with a, i, o, u, or y. Before consonant-
            # leading suffixes (-thione, -selone, -tellone, etc.) preserve the 'e'.
            #
            # Examples:
            # propan + 1 + ol -> 'propan-1-ol' (elide e — 'o' is vowel)
            # butan + 2 + one -> 'butan-2-one' (elide e — 'o' is vowel)
            # propan + 2 + thione -> 'propane-2-thione' (keep e — 't' is consonant)
            # propan + 2 + selone -> 'propane-2-selone' (keep e — 's' is consonant)
            # ethan + 1 + thione -> 'ethane-1-thione' example PIN)
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
        # DD2 (Phase D,: a suffix led by an italic chalcogen-pair
        # descriptor ('SO-thioperoxol' / 'OS-thioperoxol') needs a separating
        # hyphen even with no numeric locant — 'methane-SO-thioperoxol', not
        # 'methaneSO-thioperoxol'. Matched explicitly (not a bare isupper test)
        # so a future uppercase-led suffix cannot silently inherit this path.
        if suffix.startswith(("SO-", "OS-")):
            return f"{base}e-{suffix}"
        return apply_vowel_elision(base + "e", suffix)
