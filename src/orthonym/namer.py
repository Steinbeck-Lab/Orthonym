"""
Main IUPAC nomenclature generator.

Architecture:
    SMILES → Perception → Classification → Assembly → IUPAC Name

This is the inverse of OPSIN's pipeline:
    OPSIN: Name → Tokenize → Parse → Build Structure
    Orthonym: Structure → Perceive → Classify → Assemble Name
"""

import contextvars
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem

from .data import (
    RETAINED_NAMES,  # noqa: F401 re-exported: imported FROM this module by tests/unit/data/test_opsin_merge_layer.py
)
from .diagnostics import strict_mode as _strict_mode
from .errors import (
    _DESCRIPTIVE_FALLBACK_NAMES,
    _METAL_NAMES,  # noqa: F401 re-exported: imported FROM this module by data/cation_words.py, rules/salts.py
    _ORGANIC_ELEMENTS,  # noqa: F401 re-exported: imported FROM this module by tests/unit/test_hyg02_error_catalog.py
    OrthonymLimitError,
    classify_failure_limit,
    classify_scope_limit,
    is_failure_name,
    is_refusal_sentinel,
    no_verified_pin,
)

logger = logging.getLogger(__name__)

# a phase: env-var override for the triviality controller. Read at import
# time so ORTHONYM_ENABLE_TRIVIALITY_CONTROLLER=1/true/yes/on flips the default for
# all Orthonym instances (mirrors the a phase env-gate pattern).
_TRIV_ENV = os.environ.get("ORTHONYM_ENABLE_TRIVIALITY_CONTROLLER", "").strip().lower()
_DEFAULT_TRIV = _TRIV_ENV in ("1", "true", "yes", "on")

# a phase: env-var override for group-splitting. Read at import time so
# ORTHONYM_ENABLE_GROUP_SPLITTING=1/true/yes/on flips the default for all instances.
_GS_ENV = os.environ.get("ORTHONYM_ENABLE_GROUP_SPLITTING", "").strip().lower()
_DEFAULT_GS = _GS_ENV in ("1", "true", "yes", "on")

# part B + W7: per-call NamingResult capture slot for name_with_tree.
# ContextVar provides thread-local AND asyncio-task-local isolation per PEP 567;
# safer than a module-global dict against concurrent Orthonym.name_with_tree
# calls. The slot is installed by Orthonym.name_with_tree before invoking
# self.name; composer._assemble_name_impl writes the inner-dispatch
# NamingResult into the slot when present (default=None makes the regular
# Orthonym.name path a no-op).
_name_with_tree_capture: "contextvars.ContextVar[Optional[Dict[str, Any]]]" = (
    contextvars.ContextVar("name_with_tree_capture", default=None)
)

from .assembly.composer import assemble_name
from .assembly.offer_pool import (
    ABSTAIN,
    BEST_EFFORT,
    PIN_UNVERIFIED,
    PIN_VERIFIED,
    SYSTEMATIC_VERIFIED,
)
from .perception.chains import find_principal_chain
from .perception.functional_groups import detect_functional_groups
from .perception.ions import detect_species_type, get_ion_sites, get_radical_sites
from .perception.rings import (
    classify_ring,
    get_complete_ring_atom_set,
    get_ring_info,
    get_ring_systems,
    is_aromatic_ring,
)
from .perception.stereo import (
    assign_stereochemistry,
    get_double_bond_stereo,
    get_stereocenters,
)

# a phase NEW: routing substrate. `StoutClass` + `ClassDispatchResult`
# are imported eagerly at module-load time so `_name_impl`'s
# cascade-continuation + GENERAL fallback can reference them by name. The
# `routing` sub-package is callable-only (no callbacks back into namer)
# so this is NOT a circular import — the predicate factories + handler
# shims inside `routing/dispatch_table.py` use lazy imports for
# `orthonym.rules.*` and `orthonym.namer.Orthonym` per
# PATTERNS + namer.py:853 lazy-import precedent.
from .routing.dispatch_table import ClassDispatchResult, StoutClass
from .rules.locants import build_atom_to_locant, orient_chain
from .rules.seniority import get_principal_group

# ---------------------------------------------------------------------------
# Universal stereo backstop (a phase,)
# ---------------------------------------------------------------------------


def _final_stereo_check(
    mol,
    name: str,
    handler: str = 'unknown',
    atom_to_locant: Optional[Dict[int, int]] = None,
    is_phenol_benzene: Optional[bool] = None,
) -> str:
    """Universal stereo backstop: inject (or log) missing stereodescriptors.

    Runs AFTER all handler-specific stereo injection. Only activates when
    a handler missed stereo (predicate `needs_stereo_injection` True).

    a phase -01 (//): the backstop is flipped from detect-only
    to REAL injection on the only cohort carrying an authoritative parent
    locant map — `chain` and NON-phenol `benzene` (threaded via the per-call
    confidence dict as a POST-HOC CandidateName field). When the (now D/L-aware,
    Plan 01 Pattern D) predicate fires AND the handler is allow-listed AND a
    valid `atom_to_locant` is threaded, it calls the existing
    `inject_stereo_from_locant_map` (whose guard rejects None/empty/non-
    positive maps, degrading to a no-op).

    For `complex_ring`, `polycyclic`, `heterocycle`, `unknown`, phenol benzene,
    or no map -> it stays LOG-ONLY (the pre-177 detect-only behavior): no
    authoritative numbering exists for those, and a fabricated locant is worse
    than a missing one (/: missing beats wrong). This is the path the
    E-stilbene `unknown` case keeps (descriptor-less output is EXPECTED per the
    -01 LOG-ONLY PROTECT, not a regression — the lipid/steroid reservoir flip
    is -gated,).

    Args:
        mol: RDKit Mol object (with stereo info from original SMILES)
        name: Generated IUPAC name (may or may not contain stereo)
        handler: Name of the handler that produced this name (for attribution)
        atom_to_locant: Authoritative {atom_idx: 1-indexed locant} for the parent
            (threaded POST-HOC from the winning candidate). None for non-allow-
            listed handlers -> log-only.
        is_phenol_benzene: True when a benzene parent is a phenol (excluded from
            the inject allowlist per).

    Returns:
        The name with injected stereodescriptors (allow-listed cohort) or the
        original name unchanged (log-only cohort).
    """
    from .rules.stereochemistry import (
        inject_stereo_from_locant_map,
        needs_stereo_injection,
    )

    if not needs_stereo_injection(mol, name):
        return name

    # -01 : inject allowlist = chain + NON-phenol benzene ONLY, and
    # only when an authoritative atom_to_locant has been threaded.
    _allow_inject = (
        atom_to_locant
        and (
            handler == 'chain'
            or (handler in ('benzene', 'direct') and not is_phenol_benzene)
        )
    )
    if _allow_inject:
        injected = inject_stereo_from_locant_map(name, mol, atom_to_locant)
        if injected and injected != name:
            return injected
        # The guard inside the injector rejected the map (None/empty/
        # non-positive) or there was nothing to inject -> fall through to the
        # log-only branch (a missing descriptor beats a wrong one).

    # W4-S1 / /: a SINGLE stereogenic unit on a
    # NON-CARBON centre (sulfoxide S, quaternary N+, silane Si, chiral P,...)
    # that the parent atom_to_locant never covers -> the descriptor is dropped
    # by the allow-list path above (the centre is a substituent atom, a
    # characteristic-group heteroatom, or a mononuclear hetero parent, none of
    # which appear in the carbon-parent locant map). Such a lone hetero
    # stereocentre carries NO locant (a single unambiguous stereogenic unit;
    # the same locant omission as the STEREO-06 mononuclear-methane case
    # `(R)-bromo(chloro)(fluoro)methane`), so emit the bare "(R)-"/"(S)-".
    # ACCURACY GATES (missing beats wrong): fire ONLY on EXACTLY ONE R/S atom
    # with ZERO E/Z bonds (never drops a second descriptor), and ONLY when that
    # centre is a non-carbon atom, so this can never emit a locant-less
    # descriptor for a ring/chain carbon that requires a numeric locant.
    # DETERMINISTIC: the CIP code is engine-assigned (order-independent) and
    # there is no locant to compute.
    _rs_atoms = [a for a in atoms_of(mol) if a.HasProp('_CIPCode')]
    _bond_stereo = sum(1 for b in bonds_of(mol) if b.HasProp('_CIPCode'))
    if (
        len(_rs_atoms) == 1
        and _bond_stereo == 0
        and _rs_atoms[0].GetSymbol() != 'C'
    ):
        _cip = _rs_atoms[0].GetProp('_CIPCode')
        if _cip in ('R', 'S', 'r', 's'):
            return f"({_cip})-{name}"

    # Predicate said True but no authoritative injection happened -> log gap.
    n_atom_stereo = sum(1 for a in atoms_of(mol) if a.HasProp('_CIPCode'))
    n_bond_stereo = sum(1 for b in bonds_of(mol) if b.HasProp('_CIPCode'))
    # -CLOSEOUT Item D:...unless the name expresses its configuration
    # through a descriptor channel `needs_stereo_injection` cannot see, in which
    # case this warning is a FALSE POSITIVE and the name is already complete.
    # 1147 of the gate's warnings across 122 distinct names were this.
    if _stereo_is_implied_by_name(name):
        return name
    logger.warning(
        "Stereo backstop: '%s' (handler: %s) has %d R/S + %d E/Z but name lacks "
        "descriptors. Fix handler to include stereo natively.",
        name, handler, n_atom_stereo, n_bond_stereo
    )
    return name


# ---------------------------------------------------------------------------
# -CLOSEOUT Item D: the descriptor channels the stereo backstop is blind to
# ---------------------------------------------------------------------------
#
# `needs_stereo_injection`'s descriptor vocabulary is `{R,S,r,s,E,Z}` inside
# parentheses, plus `alpha|beta-D/L-` (the ANOMERIC sugar form) and a bare
# `D-`/`L-` token. The Blue Book's PIN descriptor vocabulary is wider.
# `## **** Stereodescriptors used in the nomenclature of natural
# products` (the Blue Book) legitimises, at:44628-44632, "*(i) The
# descriptors 'D' and 'L'... for carbohydrates, amino acids and peptides, and
# cyclitols; (ii) '*erythro*' and '*threo*'...; (iii) The stereodescriptors
# 'alpha', 'beta' are used in the nomenclature of natural products to describe
# the absolute configuration of alkaloids, terpenes and terpenoids, steroids*".
#
# So a name can be configurationally COMPLETE through a channel this predicate
# cannot read, and the backstop then warns about a correct name. That was 1147
# of the gate's warnings, over 122 distinct names.
#
# WHY THIS IS A DIAGNOSTIC-ONLY EXEMPTION AND NOT A WIDENING OF
# `needs_stereo_injection`. That predicate gates BOTH this log AND the
# injection path above it, and `rules/stereochemistry.py:456` carries an
# explicit "*Per, do NOT broaden*". Widening it would stop injection
# firing on names that currently receive a correct descriptor block — trading a
# noisy log for real stereo loss (session a project rule: removing a wrong output
# can unmask a worse one). This check therefore sits at the LOG site only and
# changes no emitted name.
#
# NOT exempted, deliberately: the free amino acids (`alanine`, `serine`,
# `cysteine`, `cystine`, `threonine`, `isoleucine`, `allo-threonine`,
# `allo-isoleucine`). There the warning is TRUE — `## **** The
# stereodescriptors 'D' and 'L'` (:54291) requires the configuration at the
# alpha-carbon to be designated, `### **** Indication of configuration
# in peptides` (:54715) scopes L-omission to PEPTIDES only, and Table 10.4 pairs
# each retained name with a `rel-` (RELATIVE) systematic equivalent (:54204,
#:54211), so the bare name is not enantiospecific. Left warning on purpose.

# `### **** Stereochemical configuration of parent structures`
# (:51045): "*The stereodescriptors 'alpha', 'beta', and 'xi'... are cited
# before the name of the fundamental parent structure*", and `****`
# (:51051) "*Each chirality center is described by the stereodescriptor 'alpha',
# 'beta', or 'xi'*". A locant-prefixed alpha/beta IS a stereodescriptor.
_ALPHA_BETA_DESCRIPTOR_RE = re.compile(r"\d+(?:alpha|beta|xi|α|β|ξ)\b")

# `## **** RETAINED NAMES OF NUCLEOSIDES` (:54943) retains exactly these
# seven, whose full ribofuranosyl configuration is implied by the name; the
# nucleotide stems come from `## **** RETAINED NAMES` (:55003). BB's own
# examples are descriptor-free, e.g. `uridine 5'-(tetrahydrogen triphosphate)`
# (:55029) and `2',3',5'-tri-O-acetyladenosine` (:54981).
#
# -FINAL I10: DERIVED from the producer table, not copied from it. The
# nucleoside half was a hand-written duplicate of `rules/nucleosides.py`'s
# `_NUCLEOSIDE_STEM` values, so the two could drift, and the docstring below
# asserted they could not. Deriving it also removes the surface a mutation used:
# appending `'itol'`/`'neuraminic'` to a literal tuple silenced `xylitol` and the
# neuraminic acids -- the exact three names this predicate must leave VISIBLE --
# and survived all 92 tests. There is no longer a literal tuple to append to.
def _nucleoside_stems() -> Tuple[str, ...]:
    """Retained nucleoside stems + nucleotide stems."""
    stems = set()
    try:
        from .rules.nucleosides import _NUCLEOSIDE_STEM
        for value in _NUCLEOSIDE_STEM.values():
            if isinstance(value, str) and value:
                # `2'-deoxyadenosine` contributes the bare `adenosine` too, so a
                # derivative of either spelling is covered.
                stems.add(value.lower())
                stems.add(value.lower().split('-')[-1])
    except Exception:  # pragma: no cover - table is always importable in-tree
        pass
    # `## **** RETAINED NAMES` (:55003) -- the nucleotide `-ylic acid`
    # stems, which have no constructor table of their own in this repo. Kept
    # explicit and separately cited rather than folded into the derived set.
    # `thymidine` is retained by `## **** RETAINED NAMES OF NUCLEOSIDES`
    # (:54943) but is not reachable through `_NUCLEOSIDE_STEM`, which maps
    # (uracil, deoxy) to `2'-deoxyuridine`; kept so the exemption covers what the
    # Blue Book retains rather than only what today's constructor emits.
    stems.update({
        'thymidine',
        'adenylic', 'guanylic', 'inosinic', 'xanthylic', 'cytidylic',
        'thymidylic', 'uridylic',
    })
    return tuple(sorted(s for s in stems if len(s) > 5))


_IMPLIED_STEREO_NUCLEOSIDE_STEMS: Tuple[str, ...] = _nucleoside_stems()

# The stem must end at a word boundary, so an acyl/alkyl PREFIX is still covered
# (`5'-O-(2-methylbutanoyl)adenosine`, `2',3',5'-tri-O-acetyladenosine` -- both
# verbatim-BB shapes) while an unrelated longer word is not silently swallowed.
_NUCLEOSIDE_STEM_RE = re.compile(
    "(?:%s)\\b" % "|".join(re.escape(s) for s in _IMPLIED_STEREO_NUCLEOSIDE_STEMS)
) if _IMPLIED_STEREO_NUCLEOSIDE_STEMS else None


def _steroid_stereoparent_stems() -> frozenset:
    """Stems of the stereoparents that imply an alpha/beta skeleton.

    Derived from the same three natural-product tables the Family-5 leg keys off,
    so the alpha/beta leg is anchored to a PARENT rather than firing on any name
    that merely contains `3beta`.
    """
    stems = set()
    try:
        from .data.natural_products import (
            NAME_EXACT_NP_PARENTS,
            NATURAL_PRODUCT_DERIVATIVES,
            NATURAL_PRODUCT_SCAFFOLDS,
        )
        names = set(NAME_EXACT_NP_PARENTS)
        names |= {v for v in NATURAL_PRODUCT_DERIVATIVES.values()
                  if isinstance(v, str)}
        names |= {e['name'] for e in NATURAL_PRODUCT_SCAFFOLDS.values()
                  if isinstance(e, dict) and isinstance(e.get('name'), str)}
    except Exception:  # pragma: no cover
        return frozenset()
    for raw in names:
        low = raw.lower()
        stems.add(low)
        for suf in ('anediol', 'enediol', 'anediyl', 'anol', 'enol', 'anone',
                    'enone', 'ane', 'ene', 'one', 'ol', 'ine', 'an', 'en'):
            if low.endswith(suf) and len(low) - len(suf) >= 4:
                stems.add(low[:-len(suf)])
    return frozenset(s for s in stems if len(s) >= 4)


_STEROID_STEREOPARENT_STEMS: frozenset = _steroid_stereoparent_stems()

# A SHORT stem must start at a token boundary; a long one need not.
#
# Neither rule alone works, and both failure modes were measured:
#
# * a plain substring test is too loose -- `tropane` contributes the
# 4-character stem `trop`, which matches inside
# `1,6-anhydro-beta-D-al-trop-yranose`, a carbohydrate with no steroid in it
# (the single spurious hit across 1963 gold + pack + table names);
# * requiring a boundary for EVERY stem is too strict -- substitutive names
# concatenate the prefix straight onto the stem, so `androst` in
# `17beta-hydroxy-androst-4-en-3-one` (spelled `hydroxyandrost...`) is
# preceded by a letter and would be missed. An existing test caught this.
#
# Six characters is the cut: at that length an accidental interior match is not
# observed anywhere in the corpus, while every genuine sterane/terpenoid stem
# that appears mid-word (`androst`, `cholest`, `pregnan`, `cholan`) is longer.
_LONG_STEROID_STEMS = sorted(
    (s for s in _STEROID_STEREOPARENT_STEMS if len(s) >= 6), key=len, reverse=True)
_SHORT_STEROID_STEMS = sorted(
    (s for s in _STEROID_STEREOPARENT_STEMS if len(s) < 6), key=len, reverse=True)


def _compile_alt(stems, prefix=""):
    if not stems:
        return None
    return re.compile(prefix + "(?:%s)" % "|".join(re.escape(s) for s in stems))


_STEROID_STEM_LONG_RE = _compile_alt(_LONG_STEROID_STEMS)
_STEROID_STEM_SHORT_RE = _compile_alt(_SHORT_STEROID_STEMS, r"(?<![a-z])")


def _has_steroid_stereoparent_stem(low: str) -> bool:
    if _STEROID_STEM_LONG_RE is not None and _STEROID_STEM_LONG_RE.search(low):
        return True
    return (_STEROID_STEM_SHORT_RE is not None
            and _STEROID_STEM_SHORT_RE.search(low))

# `### ****` (:55227) retains `sphinganine` "*for the aliphatic amino
# alcohol HAVING THE DESCRIBED ABSOLUTE CONFIGURATION*", and (:55231) gives
# `(4E)-sphing-4-enine` under `### **** Glycosphingolipids`.
#
# -FINAL I10: this was a SIX-CHARACTER substring test, `'sphing' in name`,
# whose first clause (`'sphinganine' in name`) was dead by subsumption. It
# matched anything containing `sphing`, and a mutation to `'sph'` -- which
# matches every `phosph...` name in the corpus -- survived all 92 tests. Now an
# anchored match against the two retained spellings the Blue Book names.
_SPHINGOID_RETAINED_RE = re.compile(r"\bsphing(?:anine|-?\d*-?enine)\b")


def _stereo_is_implied_by_name(name: str) -> bool:
    """Does ``name`` already express its configuration, by a channel the
    R/S + E/Z counter cannot see?

    Diagnostic predicate only — it suppresses a false-positive WARNING and never
    changes an emitted name.

    Every leg is anchored to a table that PRODUCES these names, so it cannot
    drift from the producers. -FINAL corrected this sentence, which was
    FALSE for three of the five legs when written: the steroid leg was a bare
    `\\d+(?:alpha|beta|xi)` regex with no parent requirement, the nucleoside leg
    was an unbounded substring scan over a hand-copied duplicate of
    `rules/nucleosides.py`'s table, and the sphingoid leg was the six-character
    `'sphing' in name`. A false in-code assertion is the class this range was
    written to delete, so it is repaired rather than trimmed.

    It also cannot see COMPLETENESS, and the citation it rests on says
    completeness is required. `### ****` (:51045) in full: the name of a
    fundamental parent structure "*implies the absolute configuration of all
    chirality centers **and the configuration of double bonds, when applicable**,
    without further specification. **All chirality must be defined**...*" The
    ellipsis in the previous version of this comment removed both the double-bond
    clause and the completeness obligation. A partially-described name in an
    exempted family would therefore be silenced; no producer reachable today
    emits one (~20 in-family constructions with an extra undescribed element all
    fail closed or fall back to a fully systematic name), but nothing structurally
    prevents it. The primitive that could check --
    `stereochemistry.count_expressed_stereo_descriptors` -- counts only
    parenthesised R/S/E/Z tokens and so returns 0 for `5alpha-cholestan-3beta-ol`,
    `myo-inositol` and `adenosine`, i.e. it cannot read the alpha/beta or
    implied-parent channels at all. Closing this needs a family-implied-count
    oracle (implied + expressed == defined), not that primitive.
    """
    if not name:
        return False

    _low = name.lower()

    # Family 6 — `chalcone` . `****` (:28297), under the
    # heading `### **** Retained names`, ends: "*Chalcone refers only to
    # the trans- or (E)- stereoisomer.*" The retained name therefore IS the
    # descriptor for its one E/Z unit — exactly the "channel the R/S + E/Z
    # counter cannot see" this predicate exists to cover. Without this leg every
    # chalcone emission logs "has 0 R/S + 1 E/Z but name lacks descriptors",
    # which is a FALSE POSITIVE: the name is configurationally complete, and
    # 's completeness obligation is met because the molecule has exactly
    # one stereogenic unit and the name implies it.
    #
    # EXACT equality, not a substring: it is anchored to the sole producer, the
    # `"O=C(/C=C/c1ccccc1)c1ccccc1": "chalcone"` row in `data/retained_names.py`,
    # which is keyed on the whole-molecule isomeric canonical SMILES and so can
    # only ever emit this one bare name. A substring test would wrongly exempt a
    # substituted `...chalcone...` name if that class is ever built — and those
    # names may legitimately need descriptors for ADDITIONAL stereogenic units.
    if _low == "chalcone":
        return True

    # Family 4 — steroid/terpenoid alpha/beta: the name DOES carry descriptors
    # (`cholest-5-en-3beta-yl hydrogen sulfate`, `5alpha-cholestan-3beta-ol`).
    # Produced by `rules.steroid_stereo.collect_steroid_alpha_beta`, whose own
    # module docstring cites.
    #
    # BOTH conditions are required. The locanted descriptor alone exempted any
    # name containing `3beta`, with no steroid parent anywhere in it; anchoring it
    # to a stereoparent stem is what makes the leg keyed off the producer.
    if (_ALPHA_BETA_DESCRIPTOR_RE.search(name)
            and _has_steroid_stereoparent_stem(_low)):
        return True

    # Family 1 — inositols. `## **** DEFINITIONS` (:54821): "*Inositols
    # have retained names and... employ the stereodescriptors 'D' and 'L' to
    # describe configurations*"; `****` (:54831): "*Stereoisomeric
    # inositols are described by adding italicized prefixes at the front of the
    # name 'inositol'.... Names denoted by the prefixes are preferred.*" The
    # prefix IS the descriptor for the five meso forms. Same table as the
    # OPSIN-validity carve-out below.
    try:
        from .rules.inositols import INOSITOL_NAMES
        if name in INOSITOL_NAMES:
            return True
    except Exception:
        pass

    # Family 3 — retained nucleosides/nucleotides and their derivatives.
    # Anchored: the stem must END at a word boundary, so `2',3',5'-tri-O-
    # acetyladenosine` (BB:54981) still matches while an unrelated longer word
    # cannot be swallowed whole.
    if _NUCLEOSIDE_STEM_RE is not None and _NUCLEOSIDE_STEM_RE.search(_low):
        return True

    # Family 5 — stereoparent hydrides. `****` (:50991) defines a
    # stereoparent as a parent that "*should include as much configuration as
    # possible*"; `### ****` (:51045) is the governing sentence: "*The
    # name of a fundamental parent structure usually implies the absolute
    # configuration of all chirality centers **and the configuration of double
    # bonds, when applicable**, without further specification. **All chirality
    # must be defined**...*"
    #
    # -FINAL: a third citation, `### ****` (:52106,
    # "*Stereochemistry implied by the name of the stereoparent structure remains
    # the same, unless otherwise specified*"), is REMOVED as out of scope. The
    # sentence is verbatim, but its section heading is "*Removal of a terminal
    # ring.*" and the same paragraph ends "*This use of 'des' is restricted to
    # steroids*" -- it governs `des`-prefixed steroid names, not stereoparent
    # implication in general. already carries the point.
    #
    # Membership is `****` Table 10.1 (:51377), approached through the
    # THREE tables that emit these names (see the note below), so no new
    # hardcoded list. NB `NATURAL_PRODUCT_DERIVATIVES` is WIDER than Table 10.1
    # -- it carries derivative and trivial names that are not fundamental parent
    # hydrides -- so this leg is broader than alone would license.
    # Recorded rather than narrowed: it is diagnostic-only, and narrowing it
    # would restore warnings on names that do carry implied configuration.
    try:
        from .data.natural_products import (
            NAME_EXACT_NP_PARENTS,
            NATURAL_PRODUCT_DERIVATIVES,
            NATURAL_PRODUCT_SCAFFOLDS,
        )
        if name in NAME_EXACT_NP_PARENTS:
            return True
        if name in set(NATURAL_PRODUCT_DERIVATIVES.values()):
            return True
        # All three are needed: `NAME_EXACT_NP_PARENTS` deliberately excludes the
        # parents that DO round-trip in OPSIN (`tropane`, `prostane`,
        # `thromboxane`), and `tropane`/`stigmastane` live only in the scaffold
        # table. Keying off all three covers Table 10.1 without a new list.
        if name in {
            _e['name'] for _e in NATURAL_PRODUCT_SCAFFOLDS.values()
            if isinstance(_e, dict) and 'name' in _e
        }:
            return True
    except Exception:
        pass

    # `### ****` (:55227): "*the retained name 'sphinganine' for the
    # aliphatic amino alcohol HAVING THE DESCRIBED ABSOLUTE CONFIGURATION...
    # is preferred to the systematic name (2S,3R)-2-aminooctadecane-1,3-diol*",
    # and (:55231) `(4E)-sphing-4-enine`.
    if _SPHINGOID_RETAINED_RE.search(_low):
        return True

    return False


# ---------------------------------------------------------------------------
# Universal OPSIN-grammar backstop (a phase)
# ---------------------------------------------------------------------------


def _final_grammar_check(name: str, smiles: Optional[str], handler: str,
                         grammar, stats: Dict[str, int]) -> str:
    """Universal OPSIN-grammar backstop: validate + (round-trip-gated) repair.

    Mirrors the `_final_stereo_check` shape (a phase pattern).
    Single chokepoint per internal notes — never per-handler-exit
    . On unrepairable failure, log WARNING and return ORIGINAL
    name (,). Never silently mutate.

    Args:
        name: The assembled IUPAC name (post stereo backstop).
        smiles: Original SMILES the name was generated from. Forwarded
            to `OpsinGrammar.suggest_fix` for the round-trip gate per
            internal notes /. May be None on cold paths; the
            grammar layer documents the degraded-path contract.
        handler: Handler attribution string (per a phase
            pattern); used in WARNING logs only.
        grammar: An `OpsinGrammar` instance (or None when the
            `_disable_grammar_validation` kwarg was passed at
            construction time per).
        stats: The per-instance counter dict (/) shared by
            reference with the grammar instance. Mutated in-place.

    Returns:
        Either the validated name (happy path), the round-trip-gated
        repair (on validate-fail + successful repair), or the
        original name (on validate-fail + no repair). Never raises.
    """
    if grammar is None or not name:
        return name

    if grammar.validate(name):
        stats["validate_passed"] = stats.get("validate_passed", 0) + 1
        return name

    # validate rejected — try suggest_fix (round-trip-gated).
    # LOCKED signature: name FIRST, source_smiles SECOND.
    repaired, repair_class = grammar.suggest_fix(name, source_smiles=smiles)

    if repaired is not None and repaired != name:
        # The grammar layer already incremented the matching
        # `repair_succeeded_<class>` bucket internally per.
        logger.warning(
            "OPSIN grammar repair: handler=%s class=%s original=%r repaired=%r",
            handler, repair_class, name[:50], repaired[:50],
        )
        return repaired

    # validate rejected and no class produced a round-trip-passing
    # candidate. Log unrepairable WARNING and fall back to ORIGINAL
    # name per + (never silently mutate).
    logger.warning(
        "OPSIN grammar validation failed: handler=%s name=%r",
        handler, name[:50],
    )
    return name


# ============================================================================
# a phase: pre-emission OPSIN-parse validity gate (..)
# ============================================================================
# A real-OPSIN sibling to the always-on _final_grammar_check: a generated
# production name that OPSIN CANNOT PARSE is suppressed -> the EXISTING
# _descriptive_fallback STRING (never a shipped invalid IUPAC string). Catches
# ONLY opsin_cant_parse (~818) malformed strings, NOT semantically-wrong-but-
# parseable names (heptanolate/camphor's cyclopentanone both PARSE; C4).
#
# Default-ON in production (the OPPOSITE of the 168/169 default-OFF RT-mover
# flags) — shipping an invalid string IS the bug. Env escape hatch for
# raw-output tests (the test suite disables it via a conftest autouse fixture;
# the gate's own tests re-enable it). FAIL-OPEN on no-JAR .
_VG_ENV = os.environ.get("ORTHONYM_DISABLE_OPSIN_VALIDITY_GATE", "").strip().lower()
_DISABLE_VALIDITY_GATE = _VG_ENV in ("1", "true", "yes", "on")

# Breadth Job 1: set while ``Orthonym._name_with_pin_promotion`` runs its two
# ``name`` calls, so neither re-enters the wrapper.
_PIN_PROMOTION_WRAPPED = contextvars.ContextVar("orthonym_pin_promotion_wrapped",
                                                default=False)

# Set by ``Orthonym._strict_pin_twin_name`` while the strict twin names: the twin
# reports the strict path's own name to ``name_tiered``, so the default tier's
# emission rule (``Orthonym._default_tier_decision``) is not applied to it.
_DEFAULT_TIER_POLICY_OFF = contextvars.ContextVar("orthonym_default_tier_policy_off",
                                                  default=False)

# (a phase): the constitutional self-consistency gate. After the
# parseability gate confirms OPSIN ACCEPTS a name, re-perceive it: if OPSIN parses
# the name to a CONSTITUTIONALLY DIFFERENT molecule than the input, the name is
# wrong (names a different compound) and is suppressed to the honest fallback.
# * "warn" — log + count would-suppressions WITHOUT changing output (the mandatory
# warn-only validation phase: 0 protect-row false-positives before flip).
# * "on" — suppress on a verified constitutional mismatch (the accuracy-first state).
# * "off" — disabled.
# Comparison is STEREO-INSENSITIVE and TAUTOMER-/AROMATICITY-TOLERANT (standard
# InChIKey skeleton block, -07): a stereo-only or mobile-H-tautomer difference
# must NEVER suppress a constitutionally-correct name. Fail-OPEN on every inconclusive
# outcome (OPSIN can't be consulted, either structure unparseable by RDKit).
# Flipped "warn" -> "on" 2026-06-22 after warn-only validation: the constitutional gate
# would-suppress 0/79 protect-set rows (zero false-positives), while correctly flagging
# the wrong-molecule TARGET cases (C[Ti](Cl)(Cl)Cl->"methane", [O-]C(=O)CN->"acetate",...).
# Accuracy-first: a name for a different molecule is suppressed to the honest fallback.
_SC_DEFAULT = "on"
_SC_MODE = os.environ.get("ORTHONYM_SELF_CONSISTENCY_GATE", _SC_DEFAULT).strip().lower()
if _SC_MODE not in ("off", "warn", "on"):
    _SC_MODE = _SC_DEFAULT

# a phase L0/L1: producer-agnostic coverage AUDIT at the `_finish` choke.
# "off" disables it entirely; "shadow" (the default) records a verdict but
# NEVER changes the returned name; "veto" (shipped in L1,) is a
# REAL abstain -- it downgrades an incomplete winner (per `audit_coverage`,
# except the fail-open `method="unavailable"` case) to the honest descriptive
# fallback instead of shipping it. Read fresh on every call (not cached at
# import like `_SC_MODE`/`_DISABLE_VALIDITY_GATE` above) so a measurement
# script can flip it per-process without import-order games, and so
# `monkeypatch.setenv` works per-test without a module reload.
_COVERAGE_AUDIT_MODES = ("off", "shadow", "veto")


def _coverage_audit_mode() -> str:
    mode = os.environ.get("ORTHONYM_COVERAGE_AUDIT", "shadow").strip().lower()
    return mode if mode in _COVERAGE_AUDIT_MODES else "shadow"


def _self01_lookup(name: str) -> Tuple[Optional[bool], bool, str]:
    """The verdict `_final_opsin_validity_gate` ALREADY computed for
    THIS EXACT `name` -- reused by the L0/L1 coverage audit instead of a
    second OPSIN call (CARRIED RULING, `task-L0-brief.md`) -- PLUS an explicit
    "is a fresh re-anchor even worth trying" decision (a phase L0 review
    fix, findings C1/C2: a deny-by-default `None` alone let the bare-str path
    fall through to `validate_atom_coverage` -- a REAL `java -jar opsin`
    subprocess -- for cases where that call is GUARANTEED to teach us
    nothing, which is exactly the "second unconditional OPSIN call per name"
    the carried ruling forbids. Measured: 0.9-2.7s per carve-out emission,
    and every unit test that names a molecule without the `opsin_gate`
    fixture, because the suite's autouse fixture disables the gate by
    default -> `gate_disabled` -> old code's bare `None` -> reanchor).

    Returns:
        ``(complete, skip_reanchor, detail)``:

        * ``complete``: ``True``/``False`` when ALREADY answered for
          this exact string -- a REAL reused verdict (deny-by-default via
          `resolve_gate_outcome`: an outcome recorded for a DIFFERENT string
          -- e.g. `_apply_trivial_fallback` swapped the name after the gate
          ran -- never counts). `self01_verified` /
          `self01_verified_constitution_only` -> ``True`` (constitution
          proven -- the constitution-only carve-out still proves
          constitution, all atom-coverage cares about). `self01_warn_mismatch`
          -> ``False`` (a PROVEN mismatch shipped anyway under
          `_SC_MODE == "warn"`). **``None`` in EVERY other case, including
          when ``skip_reanchor`` is ``True``** -- deliberately, because
          `audit_coverage` checks ``self01_complete is not None`` FIRST: a
          skip decision must reach ITS OWN ``skip_reanchor`` branch there, not
          be swallowed by the self01 branch as if it were a real verdict.
        * ``skip_reanchor``: ``True`` when a fresh `validate_atom_coverage`
          re-anchor must NOT be attempted at all -- a `carveout:*` outcome (a
          BY-DESIGN OPSIN-unparseable PIN: thioperoxol / inositol /
          np_stereoparent / dianhydride / chalcogen_dianhydride /
          polyol_polyester / organometallic_additive / phane / halogen_uide
          -- the parse is guaranteed to fail), or `gate_disabled` /
          `unavailable` / `not_run` (no real gate decision exists at all to
          reuse OR to usefully repeat: disabled/unavailable already mean "no
          jar was consulted", so a reanchor would just make the SAME
          jar-absence discovery via a second code path; not_run means this
          bare-str winner never went through the gate machinery in the first
          place). `audit_coverage` always answers ``complete=True`` for this
          case (fail-OPEN, never blocks in SHADOW) -- but THAT answer comes
          from `audit_coverage`'s own `skip_reanchor` branch, not from this
          function's `complete` slot (see above).
        * ``detail``: the message to use verbatim as the `CoverageVerdict`
          detail when ``skip_reanchor`` is ``True``; ``""`` otherwise.

        The remaining rare jar-present-but-non-terminal outcomes
        (`suppressed`, `inconclusive`, `bypassed`, `self01_skipped`,
        `descriptive_fallback`) return ``(None, False, "")`` -- "no verdict
        available, but a reanchor might still learn something" -- exactly the
        prior behaviour for those.
    """
    from .metrics import provenance as _pv
    prov = _pv.get_provenance()
    resolved = _pv.resolve_gate_outcome(
        prov["gate_outcome"], prov["gate_outcome_name"], name)
    if resolved in (_pv.GATE_OUTCOME_SELF01,
                    _pv.GATE_OUTCOME_SELF01_CONSTITUTION_ONLY,
                    _pv.GATE_OUTCOME_STEREO_RECOMPOSED):
        # STEREO_RECOMPOSED is full-InChIKey verified (a review -P2 F3) — a
        # stronger proof than, so it takes the same "verdict ok, no
        # reanchor needed" path (the name is already the reanchored, verified form).
        return True, False, ""
    if resolved == _pv.GATE_OUTCOME_SELF01_WARN_MISMATCH:
        return False, False, ""
    if (resolved.startswith(_pv.GATE_OUTCOME_CARVEOUT_PREFIX)
            or resolved in (_pv.GATE_OUTCOME_DISABLED,
                            _pv.GATE_OUTCOME_UNAVAILABLE,
                            _pv.GATE_OUTCOME_NOT_RUN)):
        # `complete=None` (NOT True) is deliberate -- see the docstring above.
        return None, True, f"{resolved}: no self01, reanchor skipped"
    return None, False, ""


# Module-level singleton so the OPSIN parse cache is shared across all name
# calls in a process (lazy-init on first use; see _validity_oracle).
_VALIDITY_ORACLE = None
# The oracle _validity_oracle built last (an oracle put in _VALIDITY_ORACLE from
# outside, e.g. a test's stub, is not it and is used as it is).
_VALIDITY_ORACLE_BUILT = None


def _validity_oracle():
    """The shared validity-gate OpsinOracle, for the jar ``_find_opsin_jar`` names now.

    Built at first use, and built again when the jar resolution has changed since
    (``orthonym.jars.fetch_all`` in a process that started in reduced mode, or a
    test that simulates a missing jar). An oracle built without a jar answers
    'unavailable' to every parse, and the gate fails closed on that, so keeping it
    would withhold every later name of the process ('unknown organic compound';
    TRIAGE 'Unit suite -- the unknown-organic-compound order leak').
    """
    global _VALIDITY_ORACLE, _VALIDITY_ORACLE_BUILT
    from .validation.opsin_roundtrip import _find_opsin_jar
    jar = _find_opsin_jar()
    oracle = _VALIDITY_ORACLE
    if oracle is None or (oracle is _VALIDITY_ORACLE_BUILT and oracle._jar != jar):
        from .assembly.retained_substitution import OpsinOracle
        oracle = OpsinOracle(opsin_jar=jar)
        _VALIDITY_ORACLE = _VALIDITY_ORACLE_BUILT = oracle
    return oracle


# W3-P06: dianhydride/polyanhydride PIN word-form — >=2 acid-like
# words then a NUMERICALLY-multiplied '...anhydride' ('diacetic butanedioic
# dianhydride'). Used ONLY by the validity gate to carve out these OPSIN-
# unparseable-but-correct-by-construction names (see _final_opsin_validity_gate).
_DIANHYDRIDE_PIN_RE = re.compile(
    r"^(?:[a-z0-9][a-z0-9,()'\-]* ){2,}(?:di|tri|tetra|penta|hexa)anhydride$"
)

# W8-P4: the chalcogen di-/poly-anhydride sibling of the above —
# >=2 acid-like words then a 'bis(thioanhydride)'/'tris(selenoanhydride)'/...
# class word (BB 32446 'diacetic butanedioic bis(thioanhydride) (PIN)').
# OPSIN-2.9 has no grammar for the multiplied-and-parenthesised chalcogen-
# anhydride class word (exactly the plain dianhydride gap above). Emitted ONLY
# by the hard-gated rules.anhydrides._name_chalcogen_dianhydride (residue walk
# reused from _name_dianhydride) -- correct by construction.
_CHALCOGEN_DIANHYDRIDE_PIN_RE = re.compile(
    r"^(?:[a-z0-9][a-z0-9,()'\-]* ){2,}"
    r"(?:bis|tris|tetrakis)\((?:thio|seleno|telluro)anhydride\)$"
)

# W3-P07 method (1)): the functional-class polyol polyester PIN —
# a multivalent parent-group descriptor ('propane-1,2,3-triyl') followed by >=2
# space-separated, locant-prefixed anion words each ending in '...ate'
# ('propane-1,2,3-triyl 1,3-diacetate 2-propanoate'). Emitted ONLY by the
# hard-gated rules.lipids._assemble_glyceride mixed-acyl branch (correct-by-
# construction numbering), but OPSIN cannot parse the multi-anion functional-class
# syntax. Used ONLY by the validity gate to carve out this OPSIN-unparseable-but-
# correct PIN (exactly the inositol / dianhydride situation).
_POLYOL_POLYESTER_PIN_RE = re.compile(
    r"^[a-z]+ane-[0-9,]+-(?:di|tri|tetra|penta)yl"
    r"(?: [0-9,]+-[A-Za-z0-9()\[\],'*-]*ate){2,}$"
)

# W3-P10: di-/polynuclear noncarbon-oxoacid PREFIX-derivative PINs —
# locant-prefixed class groups on a di/hypodi/tri parent oxoacid
# ('1,3-diamidodiphosphoric acid'). These are correct-by-construction (emitted
# ONLY by the hard-gated rules.inorganic_acids.name_polyacid_derivative off an
# exact backbone), but OPSIN has a locant-assignment BUG for prefix substituents
# on these polyacid parents: it stacks '1,3-diamido' both on (verified — OPSIN
# parses the analogous '1,3-diimido...' correctly, but '1,3-diamido'/'1,3-dichloro'
# to the wrong constitution). The Blue Book is the sole PIN authority; OPSIN's
# re-parse bug must not gate our correctness (same rationale as the dianhydride /
# inositol carve-outs, applied to the constitutional self-consistency check
# rather than the unparseable path). Used ONLY by the validity gate.
_POLYACID_PREFIX_DERIVATIVE_PIN_RE = re.compile(
    r"^(?:\d+(?:,\d+)*-(?:di|tri|tetra|penta|hexa)?"
    r"(?:amido|imido|hydrazido|nitrido|chloro|bromo|fluoro|iodo|cyanato|azido|"
    r"thio|seleno|telluro))+"
    r"(?:hypo)?(?:di|tri|tetra)"
    r"(?:phosphoric|phosphonic|sulfuric|selenic|telluric|arsoric) acid$"
)

# W8-P9 /.4/.6): additive/coordination organometallic PINs contain
# eta/kappa/mu multicentre descriptors (or '-ido' anionic-ligand words) that
# OPSIN-2.9's generation grammar cannot parse. These are emitted ONLY by the
# hard-gated organometallic assembler (rules.organometallics) -> correct by
# construction. Same rationale as the inositol/dianhydride/thioperoxol carve-outs.
# CR guard (Task 9.1 open question): tightened so it never un-suppresses an
# unrelated malformed name -- requires EITHER an eta/kappa/mu descriptor OR a
# '-ido' coordination-ligand token, AND the name must end in a known
# metal stem (optionally followed by a Stock/Ewens-Bassett suffix).
_ORGANOMETALLIC_ADDITIVE_PIN_RE = re.compile(
    r"^(?=.*(?:[ηκμ]|(?:chlorido|bromido|fluorido|iodido|hydrido)))"
    r".*(?:titanium|chromium|iron|nickel|molybdenum|tungsten|manganese|"
    r"platinum|iridium|mercury|osmium|ruthenium|cobalt|vanadium|rhodium)"
    r"(?:\([0-9IVX+\-]+\))?$"
)

# W8-P8 /.3): the phane simplified-skeletal PIN
# ('1,4(1,4)-dibenzenacyclohexaphane') -- superatom locants, a parenthesized
# attachment-locant set, an amplification prefix ('...ena'/'...ina' etc,
# possibly di/tri/bis-multiplied), and the 'phane' suffix. OPSIN 2.9.0 has NO
# phane grammar whatsoever (verified 2026-07-16: neither this PIN form nor
# the legacy bracket-prefix '[2.2]paracyclophane' parses), so there is no RT
# oracle for ANY phane name. This PIN is emitted ONLY by the hard-gated
# `rules.phane.build_phane_pin` (monocyclic all-benzene-homophane class,
# correct by construction) and is guarded by a source-level formula-
# conservation veto (`phane._phane_formula_veto`) since the RT-gate would
# otherwise fail OPEN with no Java -- exactly the inositol/dianhydride/
# organometallic carve-out precedent above.
_PHANE_PIN_RE = re.compile(
    r"^\d+(?:,\d+)*\([\d,]+\)-"
    r"(?:[a-z]+|(?:di|tri|tetra|penta|hexa)[a-z]+|"
    r"(?:bis|tris|tetrakis)\([a-z0-9\[\],.]+\))"
    r"(?:cyclo)?[a-z]*phane$"
)

# W8-P5 Task 2 /, BB 41102-41110/41303): a SUBSTITUTED
# halogen-uide anion ('diphenyliodanuide (PIN)' BB 41110, the Ph2I- hydride-
# addition ate-complex) is correct-by-construction (emitted ONLY by the
# hard-gated `rules.ions._emit_group13_uide` halogen branch, generalized
# alongside the already-RT-clean silanuide/boranuide/phosphanuide siblings —
# see `_UIDE_STEMS`/`_UIDE_STD_VALENCE` in `rules/ions.py`) but OPSIN 2.9
# REJECTS it with "unphysical valency state" whenever the implicit-bonding-
# number 'iodanuide'/'bromanuide'/'chloranuide' token is SUBSTITUTED
# (verified 2026-07-18: bare 'iodanuide' parses fine to [IH2-]; the explicit-
# lambda BB-cited alternative 'diphenyl-lambda3-iodanide' also parses fine to
# the IDENTICAL structure [I-] with 2 phenyls -- the BB text itself states
# this equivalence at 41110; 'methyliodanuide'/'diphenylbromanuide'/
# 'diphenylchloranuide' error the same way -- an OPSIN grammar limitation on
# SUBSTITUTED halogen-uide tokens, not a naming defect). Exactly the
# inositol/dianhydride/phane OPSIN-coverage-gap situation. Scoped to the
# HALOGEN uide stems only (iodan/broman/chloran) -- Si/B/P/Ge/Sn/Pb-uide
# forms already round-trip cleanly and need no carve-out. This 'uide' suffix
# string is emitted by no other code path (grep-verified: the only emitter
# is `rules.ions._emit_group13_uide`, which carries its own source-level
# atom-conservation veto since the RT-gate would otherwise fail OPEN).
_HALOGEN_UIDE_PIN_RE = re.compile(
    r"^[a-z][a-z0-9(),'-]*(?:iodan|broman|chloran)uide$"
)

# W8-P4 / atom-drop safety floor): source-level structural motifs
# reproduced (2026-07-18) to make the RAW (no-Java, gate-OFF) namer silently
# DROP atoms or emit a WRONG constitution, because no handler in this cycle's
# cascade claims them fully. Each is the EXACT verified-open shape from the
# W8-P4 ledger reproduction -- never a general "any P/S/N present" heuristic.
# See _p4_oxoacid_anhydride_leak_motif below for the full per-motif citation.
_P4_THIOPEROXY_ACID_PAT = Chem.MolFromSmarts("[CX3](=O)[OX2][SX2,SeX2]")
_P4_NITRAMIDO_PAT = Chem.MolFromSmarts("[#6][NX3;H1][N+](=O)[O-]")
# Ring-excluded: a CYCLIC 'naked carbonate' carbon (e.g. 1,3-dioxan-2-one) is
# already correctly named via ring nomenclature -- only the ACYCLIC case
# (embedded in a linear ester/anhydride-like chain) hits the mis-walk bug.
_P4_NAKED_CARBONATE_PAT = Chem.MolFromSmarts("[CX3;!R](=O)([OX2])[OX2]")


def _p4_oxoacid_anhydride_leak_motif(mol) -> bool:
    """W8-P4 atom-drop safety floor (accuracy-critical phase; RT-gate fails
    OPEN with no Java, so this check is INDEPENDENT of OPSIN/Java).

    Each branch is a reproduced, gate-OFF WRONG/DROP leak with NO owning
    handler built this cycle:
      * acyl-O-[S,Se]H thio/seleno-peroxy acid -- no peroxy-acid
        base suffix exists this cycle ; 'CC(=O)OS' raw -> 'ethane'
        (drops S+O2).
      * -NH-NO2 nitramido on carbon -- substituent-prefix build
        deferred; 'O=[N+]([O-])NCC(=O)O' raw -> 'ethanoic acid' (drops
        -NH-NO2).
      * an ACYCLIC 'naked carbonate' carbon -O-C(=O)-O- with NO carbon
        substituent, embedded in a larger chain -- the substitutive
        nested-prefix engine ('{[(acetyloxy)carbonyl]oxy}formic acid') is not
        built this cycle; 'CC(=O)OC(=O)OC(=O)O' raw -> a WRONG constitution
        (parses to a smaller, different molecule -- confirmed via OPSIN RT of
        the shipped string). Ring atoms excluded so a genuinely-handled cyclic
        carbonate (1,3-dioxan-2-one) never fires.
      * an unclaimed di-nuclear noncarbon-oxoacid (P/S/Se) backbone
        partial ester / substituent-prefix on a senior acid) -- both
        design-contract items deferred this cycle. Reuses the existing
        perception-only ``_find_polyacid_backbone``, then confirms NO builder
        (inorganic-acid table/FRN OR any of the sulfonic/chalcogen/peroxy
        anhydride emitters) already claims the WHOLE molecule -- so a
        genuinely-handled case (benzenesulfonic anhydride; a nucleoside
        di/triphosphate, which never reaches GENERAL at all) is never
        mis-vetoed.

    The caller gates this on ``result.class_id == StoutClass.GENERAL``: every
    dedicated handler that DOES own one of these shapes (nucleoside, the
    inorganic-acid table, the anhydride bridge-variant emitters) runs at an
    earlier CFR priority and never reaches GENERAL, so this never fires on an
    already-correct name.
    """
    if (_P4_THIOPEROXY_ACID_PAT is not None
            and mol.HasSubstructMatch(_P4_THIOPEROXY_ACID_PAT)):
        return True
    if (_P4_NITRAMIDO_PAT is not None
            and mol.HasSubstructMatch(_P4_NITRAMIDO_PAT)):
        return True
    if (_P4_NAKED_CARBONATE_PAT is not None
            and mol.HasSubstructMatch(_P4_NAKED_CARBONATE_PAT)):
        return True
    from .rules.inorganic_acids import _find_polyacid_backbone, name_inorganic_acid
    if _find_polyacid_backbone(mol) is not None and name_inorganic_acid(mol) is None:
        from .rules.anhydrides import (
            _name_chalcogen_anhydride,
            _name_peroxy_anhydride,
            _name_sulfonic_anhydride,
        )
        if (_name_sulfonic_anhydride(mol) is None
                and _name_chalcogen_anhydride(mol) is None
                and _name_peroxy_anhydride(mol) is None):
            return True
    return False


def _validity_gate_jar_present() -> bool:
    """JAR-presence PROBE for the fail-OPEN guard .

    Independent of parse: OpsinOracle.name_to_smiles returns None for BOTH
    "JAR absent" and "parse failed", so the gate cannot infer JAR-absence from
    a parse result. This probe MUST be checked FIRST — else a no-Java env would
    suppress EVERY name (the OPPOSITE of OpsinOracle.rt_safe's fail-CLOSED).
    Monkeypatched in the gate's unit tests.
    """
    from .validation.opsin_roundtrip import _find_opsin_jar
    return _find_opsin_jar() is not None


def _validity_gate_status(name: str) -> str:
    """Cached 3-valued OPSIN parse outcome for the gate .

    Returns ``"parsed"`` | ``"rejected"`` | ``"unavailable"``. Delegates to
    ``OpsinOracle.parse_status`` via a module-level singleton so the parse cache
    is shared process-wide. ``"unavailable"`` (subprocess timeout / OSError /
    no-JAR) is not a rejection, but it verifies nothing: the gate fails CLOSED on
    it (TRIAGE g7 C01), after the oracle's own retry ladder. Monkeypatched in
    tests.
    """
    return _validity_oracle().parse_status(name)


def _validity_gate_name_to_smiles(name: str) -> Optional[str]:
    """OPSIN canonical SMILES for ``name`` via the shared singleton oracle (cached),
    or None if OPSIN rejected it / could not be consulted. Used by the
    constitutional self-consistency gate to re-perceive the emitted name."""
    return _validity_oracle().name_to_smiles(name)


def _validity_gate_name_to_opsin_smiles(name: str) -> Optional[str]:
    """OPSIN's own SMILES for ``name``, as OPSIN wrote it (not RDKit's canonical
    form of it), via the shared singleton oracle (the same cached parse), or None
    if OPSIN rejected it / could not be consulted. Used by
    :func:`_lone_pair_configuration_verified`, which must read the name's
    structure with another toolkit than RDKit."""
    return _validity_oracle().name_to_opsin_smiles(name)


def _same_canonical_smiles(smiles_a: str, smiles_b: str) -> bool:
    """True iff both SMILES parse and have the same RDKit canonical isomeric SMILES
    (the same molecular graph). False when either does not parse."""
    try:
        ma = Chem.MolFromSmiles(smiles_a)
        mb = Chem.MolFromSmiles(smiles_b)
        if ma is None or mb is None:
            return False
        return Chem.MolToSmiles(ma) == Chem.MolToSmiles(mb)
    except Exception:
        return False


def _self_consistency_skeleton(smiles: str) -> Optional[str]:
    """Constitutional skeleton key = the first (skeleton) block of the standard
    InChIKey. It encodes formula + connectivity + mobile-H-normalized H layer, and
    EXCLUDES stereochemistry (the /t,/b layers live in the second block). So it is
    stereo-insensitive and tautomer-tolerant by construction (-07). Returns
    None if RDKit cannot parse/hash the SMILES (-> the caller fails OPEN)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        from rdkit.Chem import inchi
        ik = inchikey_of(mol)
        return ik.split("-")[0] if ik else None
    except Exception:
        return None


def _self_consistency_full_key(smiles: str) -> Optional[str]:
    """The FULL standard InChIKey (skeleton block + stereo block + mobile-H /
    charge-normalized proton layer) of ``smiles`` — None if RDKit cannot parse it.

    An EQUAL full key between the input structure and a name's OPSIN re-perception
    is the strongest possible identity signal: it is exactly the project's headline
    round-trip metric, so two structures with one full InChIKey ARE the same
    molecule (same constitution, charge and stereo, protonation normalized).
    uses this to short-circuit to "ok" and thereby stops over-rejecting a salt whose
    input is written ionically (drug.[H+].[Cl-]) while its correct name OPSIN
    re-perceives as the neutral molecular form (drug.Cl): the two share one InChIKey
    but differ in the RegistrationHash TAUTOMER layer, so the stereo fallback below
    used to call them a mismatch. Equal full keys can only turn a spurious mismatch
    into ok, never the reverse (0-wrong preserved)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        from rdkit.Chem import inchi
        return inchikey_of(mol) or None
    except Exception:
        return None


#: One descriptor inside a stereodescriptor block: an optional locant (digits,
#: an optional letter like '3a', primes) and the CIP code.
_STEREO_BLOCK_TOKEN_RE = re.compile(r"^(\d+[a-z]?['′]*)?([RSrsEZMP])$")
_STEREO_BLOCK_RE = re.compile(r"\(([^()]*)\)(-?)")


def _drop_pseudoasymmetric_descriptors(name: str):
    """``name`` with every lowercase r/s (pseudoasymmetric) descriptor removed from
    its stereodescriptor blocks, and the removed codes; ``(name, )`` if none.

    '(1R,4s,7S)-1,7-dimethyl-4-(propan-2-yl)cyclodecane' ->
    ('(1R,7S)-1,7-dimethyl-4-(propan-2-yl)cyclodecane', ['s']). A block left
    empty is removed with its hyphen. Only blocks whose every comma-separated
    token is a descriptor are touched, so a substituent in parentheses is never
    edited.
    """
    reduced, tokens = _drop_pseudoasymmetric_tokens(name)
    return reduced, [code for _loc, code in tokens]


def _drop_pseudoasymmetric_tokens(name: str):
    """As:func:`_drop_pseudoasymmetric_descriptors`, but each removed descriptor
    comes with its locant: ``[('4', 's')]``; the locant is ``None`` for a bare
    ``(s)``."""
    removed = []

    def _edit(m):
        tokens = [t.strip() for t in m.group(1).split(',')]
        parsed = [_STEREO_BLOCK_TOKEN_RE.match(t) for t in tokens]
        if not tokens or not all(parsed):
            return m.group(0)
        kept = []
        for t, p in zip(tokens, parsed):
            if p.group(2) in ('r', 's'):
                removed.append((p.group(1), p.group(2)))
            else:
                kept.append(t)
        if len(kept) == len(tokens):
            return m.group(0)
        if not kept:
            return ''
        return '(' + ','.join(kept) + ')' + m.group(2)

    return _STEREO_BLOCK_RE.sub(_edit, name), removed


def _pseudoasymmetric_name_verified(name: str, smiles: Optional[str]) -> bool:
    """Verify a name that OPSIN rejects only for its pseudoasymmetric descriptors.

    OPSIN 2.9.0 parses NO name with a lowercase r/s (pseudoasymmetric) CIP
    descriptor, so '(1R,4s,7S)-1,7-dimethyl-4-(propan-2-yl)cyclodecane' can
    never round-trip, although it is the systematic name (the Blue Book's own
    '13-norgermacrane (1R,4s,7S)-4-ethyl-1,7-dimethylcyclodecane', the Blue Book:
    51471). It is verified in two independent parts:

    1. OPSIN round trip of the name WITHOUT its pseudoasymmetric descriptors,
       compared by FULL InChIKey with the input whose pseudoasymmetric centres are
       made unspecified: constitution, every true stereocentre and every E/Z are
       confirmed by OPSIN.
    2. The removed codes against the canonical CIP labeller
       (``perception.stereo.assign_stereochemistry``, the centres engine): the
       input must have exactly as many pseudoasymmetric centres as the name has
       codes, all with ONE and the same code. With (1) fixing every other centre,
       each pseudoasymmetric centre has exactly two configurations, r and s, so
       equal codes on both sides fix it too. Mixed r/s codes would need the
       locant of each centre, so they are NOT verified here (fail closed).
    3. WHERE each removed descriptor sits (``_pseudo_locants_map_to_centres``):
       read in the parsed name's own numbering, every removed token's locant must
       land on a distinct pseudoasymmetric centre of the input with that code.
       Without it the code multiset alone let '(1R,2s,7S)-' and '(1R,7S,9s)-'
       (a descriptor on a CH2) pass for the (1R,4s,7S) isomer, and
       '(1s,3s)-cyclohexane-1,4-diol' pass for the cis-1,4-diol. A token with no
       locant fails closed.
    """
    if not name or not smiles:
        return False
    reduced, tokens = _drop_pseudoasymmetric_tokens(name)
    codes = [code for _loc, code in tokens]
    if not codes or reduced == name:
        return False
    if any(not loc for loc, _code in tokens):
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        from .perception.stereo import assign_stereochemistry
        assign_stereochemistry(mol)
        pseudo = [a for a in mol.GetAtoms()
                  if a.HasProp('_CIPCode') and a.GetProp('_CIPCode') in ('r', 's')]
        input_codes = sorted(a.GetProp('_CIPCode') for a in pseudo)
        if input_codes != sorted(codes) or len(set(codes)) != 1:
            return False
        pseudo_codes = {a.GetIdx(): a.GetProp('_CIPCode') for a in pseudo}
        for a in pseudo:
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        in_key = _self_consistency_full_key(Chem.MolToSmiles(mol))
        opsin_smiles = _validity_gate_name_to_smiles(reduced)
        if in_key is None or opsin_smiles is None:
            return False
        if _self_consistency_full_key(opsin_smiles) != in_key:
            return False
        return _pseudo_locants_map_to_centres(reduced, mol, tokens, pseudo_codes)
    except Exception:
        return False


def _pseudo_locants_map_to_centres(reduced: str, mol, tokens, pseudo_codes) -> bool:
    """True iff some reading of ``reduced`` onto ``mol`` puts every removed
    ``(locant, code)`` token on a distinct pseudoasymmetric centre with that code.

    OPSIN parses the REDUCED name (no lowercase descriptor left), and its
    ``-o extendedsmi`` output labels every atom with the locant it read
    (``$_AV``). Substituent atoms reuse plain numbers (an ethyl's '1', '2'), and
    the constitutional isomorphism onto ``mol`` need not be unique, so the check
    is existential over the isomorphisms and requires the input CIP code at the
    mapped atom: a locant on a CH2 or out of range finds no such atom. Primes are
    stripped on both sides so a descriptor inside a substituent prefix is judged
    by the same rule.
    """
    from .validation.opsin_roundtrip import _AV_LINE_RE, _extended_smiles
    ext = _extended_smiles(reduced)
    if not ext:
        return False
    m = _AV_LINE_RE.match(ext.strip())
    if not m:
        return False
    labels = [t.strip() for t in m.group(2).split(';')]
    omol = Chem.MolFromSmiles(m.group(1))
    if omol is None or omol.GetNumAtoms() != len(labels):
        return False

    def _norm(loc) -> str:
        return str(loc).replace("'", '').replace('′', '')

    need: Dict[Tuple[str, str], int] = {}
    for loc, code in tokens:
        need[(_norm(loc), code)] = need.get((_norm(loc), code), 0) + 1
    matches = mol.GetSubstructMatches(omol, useChirality=False, uniquify=False,
                                      maxMatches=10_000)
    for match in matches:
        if len(match) != omol.GetNumAtoms():
            continue
        have: Dict[Tuple[str, str], int] = {}
        for opsin_idx, mol_idx in enumerate(match):
            code = pseudo_codes.get(mol_idx)
            if code is not None:
                k = (_norm(labels[opsin_idx]), code)
                have[k] = have.get(k, 0) + 1
        if all(have.get(k, 0) >= n for k, n in need.items()):
            return True
    return False


#: One CIP descriptor of a stereodescriptor block: an optional locant (digits, an
#: optional letter like '4a', primes) and the code. The locant is absent only
#: where the stereogenic unit has none: "preceded by a numerical or letter
#: locant... when such locants are present"), e.g. the sulfur of
#: 'ethyl (R)-4-nitrobenzene-1-sulfinate'.
_CIP_DESCRIPTOR_TOKEN_RE = re.compile(r"^(\d+[a-z]?['′]*)?([RSrsEZ])$")


def _cip_descriptor_tokens(blocks) -> Optional[list]:
    """Every ``(locant, code)`` of the descriptor ``blocks`` that
    ``rules.stereochemistry.strip_stereo_blocks`` removed, primes dropped from the
    locant (``None`` for a locant-free code); ``None`` when a block holds anything
    else (a relative-configuration word such as 'rel' / 'cis', 'R*', 'RS',...):
    such a descriptor does not assert one configuration of one stereogenic unit,
    so it cannot be checked."""
    tokens = []
    for block in blocks:
        if not (block.startswith("(") and block.endswith(")")):
            return None
        for tok in block[1:-1].split(","):
            m = _CIP_DESCRIPTOR_TOKEN_RE.match(tok.strip())
            if not m:
                return None
            loc = m.group(1)
            tokens.append((loc.replace("'", "").replace("′", "") if loc else None,
                           m.group(2)))
    return tokens


def _bare_skeleton(mol):
    """A copy of ``mol`` with the same atom and bond indices, every bond single and
    every atom neutral, non-aromatic and hydrogen-free: the graph two tautomers,
    two Kekulé structures or two charge forms of one constitution share."""
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        a.SetIsAromatic(False)
        a.SetFormalCharge(0)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for b in rw.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
        b.SetStereo(Chem.BondStereo.STEREONONE)
    rw.UpdatePropertyCache(strict=False)
    return rw


def _stereo_descriptors_verified(name: str, smiles: Optional[str],
                                 reduced: str) -> bool:
    """critic-1 / R63 (claims conformance, 2026-09-27): is every stereodescriptor
    of ``name`` right, for a name OPSIN rejects only for its stereo layer?

    The stereo-layer carve-out of the validity gate ships such a name after
    judged the stereo-STRIPPED parse (``reduced``,:func:`strip_stereo`), so the
    constitution is OPSIN-checked and the descriptors are not. They were shipped
    unchecked, and some were wrong: '(7R,8S)-7-chloro-1-(3,5-dihydroxyphenyl)-2-
    methylundecyl acetate' for an input whose stereocentres are the name's C1 and C2
    (C7 is unspecified, C8 is a CH2), and '(2S,3R)-3-oxoestr-4-en-17β-yl
    3-phenylpropanoate', whose C2 is a CH2 and C3 the ketone carbon.
    "NAMING OF STEREOISOMERS" (the Blue Book): "They are preceded by a
    numerical or letter locant to describe the position of the stereogenic unit
    when such locants are present" -- a descriptor asserts the configuration of the
    unit at its locant, so a descriptor on a CH2 is a wrong name, not a narrower
    OPSIN grammar.

    Generalises:func:`_pseudoasymmetric_name_verified` (r/s only) to every CIP
    descriptor. True iff, for some reading of ``reduced`` onto the input:

    1. every descriptor block is a list of CIP code tokens (R, S, r, s, E, Z),
       each with its locant where the unit has one; anything else (``rel``,
       ``cis``, ``RS``, ``R*``) is not verifiable;
    2. every token lands on a DISTINCT stereogenic unit of the input whose CIP
       label (``perception.stereo.assign_stereochemistry``, the centres engine) is
       that code -- a stereocentre for R/S/r/s, a double bond at that atom for
       E/Z. A located token is read in OPSIN's own numbering of ``reduced`` (the
       ``$_AV`` atom labels of ``-o extendedsmi``), so a locant on a CH2, a
       heteroatom or out of range finds no such unit; a locant-free token may be
       any unit with its code (step 3 still has to hold);
    3. the input with exactly those units made unspecified has the FULL InChIKey
       of ``reduced``'s OPSIN parse -- so every OTHER stereo unit of the input is
       exactly what the rest of the name says (the retained parent's implied
       configuration, a 17β,...), and none is omitted or contradicted. A token
       that restates a unit the parent already fixes (tropane's 1R,5S) is kept
       specified on both sides.

    With (3) fixing every other unit, the code fixes each token's unit, so the name
    denotes the input. Fails closed (False) on anything it cannot establish.
    """
    if not name or not smiles or not reduced:
        return False
    try:
        import itertools
        from .rules.stereochemistry import strip_stereo_blocks
        stripped, blocks = strip_stereo_blocks(name)
        if stripped != reduced:
            return False
        tokens = _cip_descriptor_tokens(blocks)
        if not tokens:
            return False
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        from .perception.stereo import assign_stereochemistry
        assign_stereochemistry(mol)
        atom_code = {a.GetIdx(): a.GetProp('_CIPCode') for a in mol.GetAtoms()
                     if a.HasProp('_CIPCode')
                     and a.GetProp('_CIPCode') in ('R', 'S', 'r', 's')}
        bond_code = {b.GetIdx(): b.GetProp('_CIPCode') for b in mol.GetBonds()
                     if b.GetBondType() == Chem.BondType.DOUBLE
                     and b.HasProp('_CIPCode') and b.GetProp('_CIPCode') in ('E', 'Z')}
        from .validation.opsin_roundtrip import _AV_LINE_RE, _extended_smiles
        ext = _extended_smiles(reduced)
        if not ext:
            return False
        m = _AV_LINE_RE.match(ext.strip())
        if not m:
            return False
        labels = [t.strip().replace("'", "").replace('′', '')
                  for t in m.group(2).split(';')]
        omol = Chem.MolFromSmiles(m.group(1))
        if (omol is None or omol.GetNumAtoms() != len(labels)
                or omol.GetNumAtoms() != mol.GetNumAtoms()):
            return False
        reduced_key = _self_consistency_full_key(m.group(1))
        if not reduced_key:
            return False
        n = omol.GetNumAtoms()
        matches = [mt for mt in mol.GetSubstructMatches(
            omol, useChirality=False, uniquify=False, maxMatches=10_000)
            if len(mt) == n]
        if not matches:
            # A tautomer, Kekulé or charge-form difference between the parse and
            # the input (the constitution itself is already verified): map the
            # bare graphs, which keep the atom indices.
            matches = [mt for mt in _bare_skeleton(mol).GetSubstructMatches(
                _bare_skeleton(omol), useChirality=False, uniquify=False,
                maxMatches=10_000) if len(mt) == n]
        by_label: Dict[str, List[int]] = {}
        for oi, lab in enumerate(labels):
            by_label.setdefault(lab, []).append(oi)
        tried: Dict[frozenset, bool] = {}
        for mt in matches:
            inv = {mi: oi for oi, mi in enumerate(mt)}
            cands = []
            for loc, code in tokens:
                c = []
                for oi in (by_label.get(loc, ()) if loc is not None
                           else range(n)):
                    mi = mt[oi]
                    if code in ('E', 'Z'):
                        c.extend(('b', b.GetIdx())
                                 for b in mol.GetAtomWithIdx(mi).GetBonds()
                                 if bond_code.get(b.GetIdx()) == code)
                    elif atom_code.get(mi) == code:
                        c.append(('a', mi))
                if not c:
                    break
                cands.append(c)
            else:
                for combo in itertools.islice(itertools.product(*cands), 64):
                    if len(set(combo)) != len(combo):
                        continue
                    unspecify = []
                    for kind, idx in combo:
                        if kind == 'a':
                            if (omol.GetAtomWithIdx(inv[idx]).GetChiralTag()
                                    == Chem.ChiralType.CHI_UNSPECIFIED):
                                unspecify.append((kind, idx))
                        else:
                            b = mol.GetBondWithIdx(idx)
                            ob = omol.GetBondBetweenAtoms(
                                inv[b.GetBeginAtomIdx()], inv[b.GetEndAtomIdx()])
                            if ob is None or ob.GetStereo() == Chem.BondStereo.STEREONONE:
                                unspecify.append((kind, idx))
                    key = frozenset(unspecify)
                    if key not in tried:
                        if len(tried) >= 128:
                            return False
                        rw = Chem.RWMol(mol)
                        for kind, idx in unspecify:
                            if kind == 'a':
                                rw.GetAtomWithIdx(idx).SetChiralTag(
                                    Chem.ChiralType.CHI_UNSPECIFIED)
                            else:
                                rw.GetBondWithIdx(idx).SetStereo(
                                    Chem.BondStereo.STEREONONE)
                        tried[key] = (_self_consistency_full_key(
                            Chem.MolToSmiles(rw)) == reduced_key)
                    if tried[key]:
                        return True
        return False
    except Exception:
        return False


def _cip_labels_match_input_as_written(smiles: Optional[str]) -> bool:
    """D3 guard: are the engine's CIP labels of ``smiles`` those of the input string
    AS WRITTEN?

    The engine's labels (``perception.stereo.assign_stereochemistry``) come from
    ``centres`` run on RDKit's canonical SMILES of RDKit's reading of the input.
    RDKit reads a three-coordinate (lone-pair) stereocentre that carries a
    ring-closure digit with the opposite configuration from the OpenSMILES reading
    CDK and OPSIN use, and writes such a centre back the same way; so where the
    input and RDKit's canonical SMILES differ in whether that centre carries a
    ring-closure digit, the engine's label is the other stereoisomer's. Measured:
    'C1(=CC=CC=C1)N1[P@@](N2CCC[C@H]2C1)OC(C)C' (P is R for CDK and for RDKit's
    own labeller) gets the engine label S (canonical 'CC(C)O[P@@]1N(...)...N21').
    Here ``centres`` labels the ORIGINAL string (its 1-based positions are the
    string's atom order, RDKit's atom indices of the same string), and the two
    maps of tetrahedral labels (R/S/r/s) must be equal and non-empty. E/Z labels are
    not compared (a double bond carries no implicit neighbour). False when centres
    is unavailable or anything fails (fail-closed)."""
    if not smiles:
        return False
    try:
        from .perception.centres_bridge import centres_label_batch
        from .perception.stereo import assign_stereochemistry
        batch = centres_label_batch([smiles])
        if batch is None:
            return False
        tetra = ('R', 'S', 'r', 's')
        as_written = {pos - 1: d for pos, d in (batch.get(smiles) or {}).items()
                      if d in tetra}
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        assign_stereochemistry(mol)
        engine = {a.GetIdx(): a.GetProp('_CIPCode') for a in mol.GetAtoms()
                  if a.HasProp('_CIPCode') and a.GetProp('_CIPCode') in tetra}
        return bool(engine) and engine == as_written
    except Exception:
        return False


#: CIP tetrahedral labels as atom-map codes for the lone-pair check (9: a compared
#: centre that centres leaves unlabelled).
_LP_CIP_CODE = {'R': 1, 'S': 2, 'r': 3, 's': 4}
_LP_UNLABELLED = 9
_LP_TETRAHEDRAL = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)


def _lone_pair_compared_atom(atom) -> bool:
    """A tetrahedral stereocentre of RDKit's reading that the lone-pair check
    compares: every one except a nitrogen outside a three-membered ring. RDKit
    perceives such an N (a bridgehead shared by at least three rings) beyond the
    InChI stereo rules, so no InChIKey round trip sees its configuration and OPSIN
    writes none for it (measured: the cinchona 1-azabicyclo[2.2.2]octane N)."""
    return (atom.GetChiralTag() in _LP_TETRAHEDRAL
            and not (atom.GetAtomicNum() == 7 and not atom.IsInRingSize(3)))


def _has_lone_pair_stereocentre(smiles: Optional[str]) -> bool:
    """True iff RDKit's reading of ``smiles`` keeps a compared tetrahedral
    stereocentre with three ligands in all (heavy atoms and H), the fourth position
    being the lone pair: P(III), As(III), S(IV) and Se(IV) (sulfoxides, sulfinyl,
    sulfinates, sulfonium), N in a three-membered ring."""
    if not smiles or '@' not in smiles:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return False
    return mol is not None and any(
        _lone_pair_compared_atom(a) and a.GetTotalDegree() == 3 for a in mol.GetAtoms())


def _lp_heavy_mol(smiles: str):
    """RDKit's reading of ``smiles`` with explicit hydrogens removed, each atom
    carrying ``_lp_pos``: its 1-based position in the string, which is the position
    ``centres`` labels (a written [H] is an atom of the string for both)."""
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    mol = Chem.MolFromSmiles(smiles, ps)
    if mol is None:
        return None
    for a in mol.GetAtoms():
        a.SetIntProp('_lp_pos', a.GetIdx() + 1)
    return Chem.RemoveHs(mol)


def _lp_label_code(atom, labels: Dict[int, str]) -> int:
    if not _lone_pair_compared_atom(atom):
        return 0
    return _LP_CIP_CODE.get(labels.get(atom.GetIntProp('_lp_pos')), _LP_UNLABELLED)


def _lp_annotated_smiles(mol, labels: Dict[int, str]) -> str:
    """Canonical SMILES of ``mol`` with its stereo removed and each compared centre's
    CIP label (from ``labels``) as its atom-map number: two such strings are equal
    iff some isomorphism of the two graphs (elements, bonds, charges, hydrogens)
    carries every compared centre onto a centre with the same label."""
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        a.SetAtomMapNum(_lp_label_code(a, labels))
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for b in rw.GetBonds():
        b.SetStereo(Chem.BondStereo.STEREONONE)
        b.SetBondDir(Chem.BondDir.NONE)
    return Chem.MolToSmiles(rw)


def _lp_inchi_correspondence(mol):
    """(standard InChI, {canonical position: atom}) from RDKit's InChI of ``mol`` and
    its AuxInfo '/N:' layer, or None. Two readings with one InChI correspond atom by
    atom through their canonical positions, whatever their tautomer or charge form."""
    from rdkit.Chem import inchi as _inchi_mod
    std, aux = _inchi_mod.MolToInchiAndAuxInfo(mol)
    found = re.search(r'/N:([0-9,;]+)', aux or '')
    if not std or not found:
        return None
    out = {}
    for comp, part in enumerate(found.group(1).split(';')):
        for k, tok in enumerate(part.split(',')):
            out[(comp, k)] = mol.GetAtomWithIdx(int(tok) - 1)
    return std, out


def _lone_pair_configuration_verified(name: str, smiles: Optional[str]) -> bool:
    """Lone-pair stereocentres (TRIAGE.md 'Lone-pair stereocentres -- RDKit
    reading'): does ``name`` denote the configuration of ``smiles`` as a reader that
    treats the lone pair as an implicit hydrogen reads the string?

    Every round trip here compares RDKit InChIKeys of the input and of OPSIN's
    SMILES of the name. RDKit does not read a lone pair the way it reads an implicit
    hydrogen in every position of the string: it reads '[C@@H](C)(O)c1ccccc1' as
    'C[C@H](O)c1ccccc1' (the implicit H of a first atom is the 'from' atom, the
    Daylight rule) but '[S@@](C)(=O)c1ccccc1' as 'C[S@@](=O)c1ccccc1' -- the other
    enantiomer -- and some ring-closure positions at P(III) and at an aziridine N
    likewise; the RDKit Book says of missing ligands that "in SMILES they are treated
    the same as implicit hydrogens". CDK and OPSIN's SMILES writer treat the lone
    pair as an implicit hydrogen everywhere (measured: CDK's reading of OPSIN's SMILES
    gives OPSIN's own StdInChIKey on every row). When RDKit misreads the input and
    OPSIN's SMILES of the name alike, a round trip passes for a name of the other
    stereoisomer: before this check '[S@@](C)(=O)c1ccccc1' shipped
    '(R)-(methanesulfinyl)benzene' pin_verified at the best-effort and PIN tiers,
    while the input, read with the lone pair as an implicit hydrogen, is the (S)
    enantiomer, '(S)-(methanesulfinyl)benzene (PIN)', the Blue Book).

    The check, for an input with a lone-pair stereocentre (``_has_lone_pair_
    stereocentre``) only: ``centres`` (CDK) labels the input string as written and
    OPSIN's own SMILES of the name (``_validity_gate_name_to_opsin_smiles``, not
    RDKit's canonical form); the labels of every compared centre must agree under an
    isomorphism of the two structures -- the annotated canonical SMILES
    (``_lp_annotated_smiles``) are equal, or, for a name whose structure OPSIN writes
    as another tautomer or charge form, the centres agree position by position
    through their shared standard InChI (``_lp_inchi_correspondence``). False when
    OPSIN or centres is unavailable or anything fails (fail-closed); True without a
    lone-pair stereocentre (nothing to check)."""
    if not _has_lone_pair_stereocentre(smiles):
        return True
    if not name:
        return False
    try:
        opsin_smiles = _validity_gate_name_to_opsin_smiles(name)
        if not opsin_smiles:
            return False
        from .perception.centres_bridge import centres_label_batch
        batch = centres_label_batch([smiles, opsin_smiles])
        if batch is None or smiles not in batch or opsin_smiles not in batch:
            return False
        lab_in, lab_out = batch[smiles], batch[opsin_smiles]
        mol_in, mol_out = _lp_heavy_mol(smiles), _lp_heavy_mol(opsin_smiles)
        if mol_in is None or mol_out is None:
            return False
        if _lp_annotated_smiles(mol_in, lab_in) == _lp_annotated_smiles(mol_out, lab_out):
            return True
        corr_in, corr_out = _lp_inchi_correspondence(mol_in), _lp_inchi_correspondence(mol_out)
        if (corr_in is None or corr_out is None or corr_in[0] != corr_out[0]
                or set(corr_in[1]) != set(corr_out[1])):
            return False
        for pos, a in corr_in[1].items():
            b = corr_out[1][pos]
            if _lone_pair_compared_atom(a) != _lone_pair_compared_atom(b):
                return False
            if _lp_label_code(a, lab_in) != _lp_label_code(b, lab_out):
                return False
        return True
    except Exception:
        return False


def _lp_labels_by_atom(labels: Dict[int, str], order: List[int]) -> Dict[int, str]:
    """centres labels of a SMILES RDKit wrote from a mol, keyed by that mol's atom
    indices (``order``: the ``_smilesAtomOutputOrder`` of the same write)."""
    return {order[pos - 1]: lab for pos, lab in labels.items() if 0 < pos <= len(order)}


#: Bound on RDKit's CIP labeller for the lone-pair reading check (about a second, the
#: RDKit documentation's figure for 1,250,000 iterations); exceeding it keeps the input.
_LP_CIP_MAX_ITERATIONS = 1_250_000


def _lone_pair_standard_spelling_and_order(
        smiles: Optional[str]) -> Tuple[Optional[str], Optional[List[int]]]:
    """(the string to name for ``smiles``, the atom order) -- see
    :func:`_lone_pair_standard_spelling`. The order is None when ``smiles`` is
    returned; otherwise ``order[k]`` is the atom of RDKit's reading of ``smiles``
    (hydrogens removed) that is atom ``k`` of the returned string."""
    if not _has_lone_pair_stereocentre(smiles):
        return smiles, None
    try:
        from rdkit.Chem import rdCIPLabeler

        from .perception.centres_bridge import _smiles_output_order, centres_label_batch
        mol = _lp_heavy_mol(smiles)
        if mol is None:
            return smiles, None
        batch = centres_label_batch([smiles])
        if batch is None or smiles not in batch:
            return smiles, None
        written = batch[smiles]
        as_written = {a.GetIdx(): written[a.GetIntProp('_lp_pos')]
                      for a in mol.GetAtoms() if a.GetIntProp('_lp_pos') in written}
        centres = [a.GetIdx() for a in mol.GetAtoms()
                   if _lone_pair_compared_atom(a) and a.GetTotalDegree() == 3]
        rdkit_read = Chem.Mol(mol)
        rdCIPLabeler.AssignCIPLabels(rdkit_read, atomsToLabel=centres, bondsToLabel=[],
                                     maxRecursiveIterations=_LP_CIP_MAX_ITERATIONS)
        misread = []
        for idx in centres:
            atom = rdkit_read.GetAtomWithIdx(idx)
            if (idx in as_written and atom.HasProp('_CIPCode')
                    and atom.GetProp('_CIPCode') != as_written[idx]):
                misread.append(idx)
        if not misread:
            return smiles, None
        standard = Chem.RWMol(mol)
        for idx in misread:
            standard.GetAtomWithIdx(idx).InvertChirality()
        standard = standard.GetMol()
        spelled = Chem.MolToSmiles(standard)
        spelled_order = _smiles_output_order(standard)
        batch = centres_label_batch([spelled])
        if spelled_order is None or batch is None or spelled not in batch:
            return smiles, None
        if _lp_labels_by_atom(batch[spelled], spelled_order) != as_written:
            return smiles, None
        reread = Chem.MolFromSmiles(spelled)
        if reread is None or Chem.MolToSmiles(reread) != spelled:
            return smiles, None
        return spelled, spelled_order
    except Exception:
        return smiles, None


def _lone_pair_standard_spelling(smiles: Optional[str]) -> Optional[str]:
    """TRIAGE.md 'Lone-pair centre written first -- naming and the gate probe': the
    string the engine names for the caller's ``smiles``.

    RDKit reads a lone-pair stereocentre written first (and at some ring-closure
    positions) unlike an implicit hydrogen, while OpenSMILES, CDK (``centres``) and
    OPSIN treat the lone pair like an implicit hydrogen (TRIAGE.md 'Lone-pair
    stereocentres -- RDKit reading'). Measured: RDKit reads '[S@](=O)(C)CC' as
    'CC[S@@](C)=O', the (R) enantiomer; CDK reads it as (S), as it reads
    'CC[S@](C)=O', and the name of the input is '(S)-(methanesulfinyl)ethane', the
    PIN form of (the Blue Book). The exit check
    (:func:`_lone_pair_configuration_verified`) withdrew the (R) name, so such a
    string was not named at all.

    RDKit reads ``smiles`` unlike the standard reading at a lone-pair stereocentre
    when RDKit's CIP label of its reading there (``rdCIPLabeler``) differs from
    CDK's label of the string as written. With no such centre ``smiles`` is
    returned unchanged. Otherwise those centres are inverted in RDKit's reading and
    the result is RDKit's canonical SMILES of it, returned only when CDK's labels of
    that string equal CDK's labels of ``smiles`` as written at every atom
    (tetrahedral and double-bond labels alike) and RDKit writes it back unchanged.
    The engine's CIP labels are CDK's labels of RDKit's canonical SMILES of its
    reading (``centres_label_mol``), which for that string is the string itself, so
    the engine then names the molecule the standard reading of ``smiles`` denotes.
    When RDKit's canonical SMILES repeats a ring-closure misreading (the P(III) and
    aziridine rows of test_lone_pair_configuration.py) that check fails and
    ``smiles`` is returned: the two misreadings cancel there and those rows are
    named right as they are. Anything that cannot be shown returns ``smiles`` (the
    exit check then decides, as before). The exit check still compares CDK's
    reading of the caller's string as written with OPSIN's own SMILES of the name
    (:func:`_lone_pair_as_written`), so a name ships only when it denotes that
    reading."""
    return _lone_pair_standard_spelling_and_order(smiles)[0]


#: The lone-pair input scope of this thread's outermost public naming call
#: (``_lone_pair_input_enter``): ``open``, and ``rewrite`` = (the string named, the
#: caller's string as written, the atom order) when the two differ, else None.
_LP_INPUT = threading.local()


def _lone_pair_input_enter(smiles: Optional[str]) -> Tuple[Optional[str], bool]:
    """Open the lone-pair input scope of a public naming call: (the string to name,
    whether this call opened the scope). Only the outermost call rewrites
    (:func:`_lone_pair_standard_spelling`); a nested public call (``name_tiered`` ->
    ``name``, ``name_with_tree`` -> ``name``, the strict PIN twin, a fragment's
    own ``name``) names the string it is given. Close with
    :func:`_lone_pair_input_exit`."""
    if getattr(_LP_INPUT, 'open', False):
        return smiles, False
    spelled, order = _lone_pair_standard_spelling_and_order(smiles)
    _LP_INPUT.open = True
    _LP_INPUT.rewrite = (spelled, smiles, order) if spelled != smiles else None
    return spelled, True


def _lone_pair_input_exit(opened: bool) -> None:
    if opened:
        _LP_INPUT.open = False
        _LP_INPUT.rewrite = None


def _lone_pair_as_written(smiles: Optional[str]) -> Optional[str]:
    """The caller's string as written when ``smiles`` is the standard spelling this
    thread's public call named in its place (:func:`_lone_pair_input_enter`), else
    ``smiles``."""
    rewrite = getattr(_LP_INPUT, 'rewrite', None)
    if rewrite is not None and smiles == rewrite[0]:
        return rewrite[1]
    return smiles


def _lone_pair_caller_atoms(hint):
    """An atom-index map of the string named, re-keyed to the atoms of the caller's
    string (RDKit's reading, hydrogens removed) when the public call named a standard
    spelling in its place; None when it cannot be re-keyed; unchanged otherwise."""
    rewrite = getattr(_LP_INPUT, 'rewrite', None)
    if rewrite is None or not hint:
        return hint
    order = rewrite[2]
    try:
        return {order[k]: v for k, v in hint.items()}
    except Exception:
        return None


def _stereo_descriptor_repair(name: str, smiles: str, reduced: str,
                              stats: Optional[Dict[str, int]]) -> str:
    """critic-1 / R63: what the stereo-layer carve-out ships when a descriptor of
    ``name`` fails:func:`_stereo_descriptors_verified`, all 0-wrong:

    1. the input's stereo composed onto ``reduced`` in OPSIN's own numbering
       (:func:`_try_compose_input_stereo`), when the composed name reads back to
       the input's FULL InChIKey ('(1R,2S)-7-chloro-...-2-methylundecyl acetate');
    2. else ``reduced`` itself when IT reads back to the full key (the retained
       parent and its α/β descriptors carry all the stereo: '3-oxoestr-4-en-17β-yl
       3-phenylpropanoate'); it is not what the PIN path built, so it is recorded
       as a non-PIN name (labelled below pin);
    3. else nothing: the name is withdrawn to the descriptive fallback.
    """
    from .metrics import provenance as _pv
    if stats is not None:
        stats["gate_stereo_descriptor_mismatch"] = (
            stats.get("gate_stereo_descriptor_mismatch", 0) + 1)
    in_key = _self_consistency_full_key(smiles)
    if in_key is not None:
        composed = _try_compose_input_stereo(reduced, smiles, in_key)
        if (composed is not None
                and _shipped_name_round_trip(composed, smiles) == "verified"):
            _record_gate_outcome(_pv.GATE_OUTCOME_STEREO_RECOMPOSED, composed)
            return composed
        if _shipped_name_round_trip(reduced, smiles) == "verified":
            _pv.record_non_pin_fragment(reduced)
            _record_gate_outcome(_pv.GATE_OUTCOME_FULL_KEY_VERIFIED, reduced)
            return reduced
    logger.warning("OPSIN validity gate: a stereodescriptor does not match the "
                   "input's CIP labels, suppressed: %r", name[:80])
    from .metrics.abstention import AbstentionCode, record_suppression
    record_suppression(AbstentionCode.GATE_SUPPRESSED,
                       detail='stereo_descriptor_unverified', candidate=name)
    _suppressed_to = _descriptive_fallback(smiles)
    _record_gate_outcome(_pv.GATE_OUTCOME_SUPPRESSED, _suppressed_to)
    return _suppressed_to


def _self_consistency_net_charge(smiles: str) -> Optional[int]:
    """Net formal charge of ``smiles`` (None if unparseable). The InChIKey skeleton
    block used by _self_consistency_skeleton EXCLUDES the charge/protonation layer
    (/q,/p), so a name that silently drops or adds a charge (e.g. hydroperoxide
    anion [O-]O -> neutral 'dioxidane' OO) shares the skeleton and would pass. Net
    charge is a genuine constitutional difference and must be checked separately."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return sum(a.GetFormalCharge() for a in mol.GetAtoms())
    except Exception:
        return None


#: Phase-1 reclaim (Adv1 guard): a bare RELATIVE stereo descriptor is 50/50-
#: ambiguous on a true enantiomeric pair — OPSIN picks one enantiomer, so BOTH the
#: full-InChIKey and the exact-CanonSmiles compare false-confirm it. Match only
#: descriptor positions (start / after ([ { space tab -). Absolute (nR)/(nZ) blocks
#: and D-/L-/alpha-/beta- config tokens are deliberately NOT matched.
_RELATIVE_STEREO_RE = re.compile(
    r"(?:^|[-([{ \t])(?:cis|trans|rel|rac|syn|anti)-"
    r"|\brel-|[±]|R\*|S\*|\((?:RS|SR|±)\)", re.IGNORECASE)


def _has_relative_stereo_token(name: str) -> bool:
    return bool(name) and _RELATIVE_STEREO_RE.search(name) is not None


#: a phase kill-switch for the general-fallback stereo-omission reclaim path
#: (default ON). ``ORTHONYM_STEREO_OMISSION_RECLAIM=0`` restores the pre-
#: suppress-always behaviour — used for A/B measurement and as a safety valve.
_STEREO_OMISSION_RECLAIM = os.environ.get(
    "ORTHONYM_STEREO_OMISSION_RECLAIM", "1") != "0"


def _try_compose_input_stereo(
    flat_name: str, smiles: str, in_key: str
) -> Optional[str]:
    """ a phase — stereo-OMISSION reclaim on the general-fallback tier.

    A general-fallback candidate whose CONSTITUTION round-trips (skeleton InChIKey
    equal) but whose FULL key differs from the input is, in the stereo-only case, a
    name that dropped the input's stereochemistry. Compose the input's stereo back
    onto ``flat_name`` with 's RT-gated re-anchor
    (:func:`inject_stereo_reanchored_rt_gated`), passing ``builder_map=None`` so the
    injector re-anchors to OPSIN's OWN locant numbering for ``flat_name`` — the
    general-fallback tier has no handler builder-map to offer here, and OPSIN's map
    is exactly the numbering the composed name is read back with.

    Returns the composed name ONLY when it recomputes to ``in_key`` (the input's full
    InChIKey) — the identical 0-wrong bar this gate applies to every other RT-verified
    emission. A genuine stereo-CONFLICT (wrong stereo) never composes to ``in_key``,
    so it returns None and the caller suppresses as before. Returns None (never
    raises) on any OPSIN/RDKit unavailability -> the caller suppresses (fail-closed).

    Cost (a review -P2 M1): the compose makes ~4 in-process OPSIN parses (the two
    inside ``inject_stereo_reanchored_rt_gated``'s RT-gate, the OPSIN re-anchor of
    ``flat_name``, and the final re-verify parse of the composed name) — each ~1 ms,
    all in-process. The caller's skeleton prefilter keeps this off the giant
    wrong-constitution suppressions, so it adds no new hang class.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        from .rules.stereochemistry import inject_stereo_reanchored_rt_gated
        composed = inject_stereo_reanchored_rt_gated(
            flat_name, mol, None, input_smiles=smiles)
        if not composed or composed == flat_name:
            return None  # nothing composed (conflict / no OPSIN locant map)
        composed_smiles = _validity_gate_name_to_smiles(composed)
        if composed_smiles is None:
            return None
        if _self_consistency_full_key(composed_smiles) == in_key:
            # the key does not place a hydron (validation/protonation_identity.py)
            from .validation.protonation_identity import protonation_site_verdict
            if protonation_site_verdict(smiles, composed_smiles, composed) == "mismatch":
                return None
            # nor the radical graph (validation/radical_identity.py)
            from .validation.radical_identity import radical_identity_verdict
            if radical_identity_verdict(smiles, composed_smiles) == "mismatch":
                return None
            return composed  # stereo composed AND full-key-verified == input
    except Exception as _exc:
        # Fail-closed (0-wrong): any OPSIN/RDKit unavailability -> the caller
        # suppresses. Log it (a review -P2 M3) so a silent self-disable from a
        # regression (a renamed import, a signature change) is visible in a debug
        # trace rather than masquerading as a plain abstain.
        logger.debug("stereo-omission reclaim skipped (%s): %s",
                     type(_exc).__name__, _exc)
        return None
    return None


# C6: RegistrationHash-backed stereo layer for. The InChIKey skeleton
# block used by _self_consistency_skeleton EXCLUDES stereo by construction
# (-07), so a name that encodes the WRONG stereoisomer (right constitution)
# passes the skeleton compare. RegistrationHash.GetMolLayers gives a stereo-bearing
# canonical layer that distinguishes enantiomers/diastereomers. Prep is normalize-only
# (no Cleanup, no tautomer/charge standardization) per spec; NO stereo-assignment
# call — GetMolLayers reads the parsed mol's stereo directly (assign is inert for it).
from rdkit.Chem import RegistrationHash as _RegistrationHash
from rdkit.Chem.MolStandardize import rdMolStandardize as _rdMolStandardize

_RH_NORMALIZER = _rdMolStandardize.Normalizer()


def _rh_prep(mol):
    m = Chem.Mol(mol)
    Chem.SanitizeMol(m)
    m = _RH_NORMALIZER.normalize(m)   # normalize ONLY — no Cleanup / tautomer / charge std
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return m


def _registration_stereo_layer(smiles: str):
    """The stereo-bearing, TAUTOMER-CANONICAL RegistrationHash layer for ``smiles``
    (None if unparseable). Uses ``TAUTOMER_HASH`` — NOT ``CANONICAL_SMILES`` — so a
    pure TAUTOMER difference (e.g. the 6-oxo vs 6-hydroxy purine of ``5'-inosinic
    acid``, whose stereo is identical) is normalized away and only a genuine
    STEREO difference remains. Distinguishes stereoisomers the InChIKey skeleton
    block cannot, while staying tautomer-tolerant like that block."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        layers = _RegistrationHash.GetMolLayers(_rh_prep(mol))
        return layers.get(_RegistrationHash.HashLayer.TAUTOMER_HASH)
    except Exception:
        return None


# Al, Ga, In, Tl: the Blue Book /:2062 -- their compounds' names carry no PIN
# status (the organic-vs-inorganic decision "has not yet been reached").
_NO_PIN_STATUS_ELEMENTS = frozenset({13, 31, 49, 81})


def _has_no_pin_status_element(smiles: str) -> bool:
    """True iff the structure contains aluminium, gallium, indium or thallium."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return mol is not None and any(
        a.GetAtomicNum() in _NO_PIN_STATUS_ELEMENTS for a in mol.GetAtoms())


# "Ocenes" (the Blue Book): the ocene parents are named
# substitutively and carry PINs ('2-(osmocen-1-yl)ethanol (PIN)',:40135).
_OCENE_STEMS = ('ferrocen', 'ruthenocen', 'osmocen', 'nickelocen', 'chromocen',
                'cobaltocen', 'vanadocen')


def _is_p69_organometallic_without_pin(smiles: str, name: Optional[str]) -> bool:
    """True iff the structure is an organometallic compound of a Group 1-12 metal
    (or a lanthanide/actinide), whose names the Blue Book gives without PIN status.

     INTRODUCTION (the Blue Book): "Preferred IUPAC names are indicated
    for organometallic compounds named substitutively for the metals in Groups 14-
    16. However, neither preferred IUPAC names or preselected names (see
    for organometallic compounds involving the transition elements (including the
    Group 3 elements) and Groups 1 and 2 elements, except for 'ocene' compounds,
    are noted." (:40157) accordingly gives 'methyllithium',
    'methylmagnesium iodide' with no '(PIN)'. "Organometallic" is structural,
     (:39721): "at least one bond between one metal atom and one carbon
    atom" -- ``perception.metals.has_covalent_metal_carbon_bond`` (true metals
    only; the Group 13-16 metalloids have their own names, and a salt of an
    organic ion has no metal-carbon bond). An ocene name keeps its PIN status.
    Label only: the name is unchanged."""
    if not smiles:
        return False
    if name and any(stem in name for stem in _OCENE_STEMS):
        return False
    try:
        from .perception.metals import has_covalent_metal_carbon_bond
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None and has_covalent_metal_carbon_bond(mol)
    except Exception:
        return False


def _is_metal_adduct_without_pin(smiles: str, name: Optional[str]) -> bool:
    """True iff ``name`` is a adduct name (em-dash notation) of a
    disconnected structure with a metal atom in it.

     "Mixed organic - inorganic adducts" (the Blue Book): "preferred
    IUPAC names cannot be assigned to mixed adducts because preferred IUPAC names
    have not yet been determined for inorganic components"; (:39735) notes
    no PIN for the Group 1-12 metals either. Label only: the name is unchanged."""
    if not smiles or not name or "\u2014" not in name or "." not in smiles:
        return False
    try:
        from .rules.adducts import _METAL_ELEMENT_NAMES
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None and any(
            a.GetSymbol() in _METAL_ELEMENT_NAMES for a in mol.GetAtoms())
    except Exception:
        return False


def _is_one_atom_ion_salt(smiles: str) -> bool:
    """True iff the input is two or more components and every component is a
    one-heavy-atom ion ('[Li+].[Li+].[Li+].[N-3]', '[Fr+].[NH2-]'): a salt of
    element ions. A salt is named by "citing the name of the cation(s) followed by
    the name of the anion... as a separate word", the Blue Book;
     :43590 "binary names formed by citing the name of the cation followed
    by that of the anion"); the em-dash notation of (:3765) is the one of
    adducts, "formed by direct combination of separate molecular entities"
    (:3761)."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return False
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return len(frags) >= 2 and all(
        f.GetNumHeavyAtoms() == 1 and Chem.GetFormalCharge(f) != 0 for f in frags)


def _has_metal_atom(smiles: str) -> bool:
    """True iff the structure holds an atom of a metal element (the element set of
    ``rules.adducts._METAL_ELEMENT_NAMES``)."""
    try:
        from .rules.adducts import _METAL_ELEMENT_NAMES
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        return mol is not None and any(
            a.GetSymbol() in _METAL_ELEMENT_NAMES for a in mol.GetAtoms())
    except Exception:
        return False


def _is_carbon_free_compound(smiles: str) -> bool:
    """True iff the structure has at least one atom and no carbon atom: a
    carbon-free compound, whose names are preselected names at most, never PINs
    ('sulfuric acid', 'ammonia', 'tetrachlorosilane', 'hexafluoro-λ5-phosphanuide',
    'iron(III) trichloride', 'potassium hydrogen sulfate').

     PREFERRED IUPAC NAMES (the Blue Book): "the label 'PIN' is added to
    the names of compounds whose parent hydride contains at least one of the
    following elements: B, Si, Ge, Sn, Pb, N, P, As, Sb, Bi, O, S, Se, Te, Po, F, Cl,
    Br, I, At, and that also contain at least one carbon atom in their structure";
    :2058 "Rules for the selection of preferred IUPAC names (PINs) for compounds
    containing Al, Ga, In, Tl, as well as for compounds containing B, Si,... At,
    and that do not contain carbon... will be discussed in a further publication.
    ... the label 'preselected name' is added to appropriate names."
    PRESELECTED NAMES (:2062): preselected names are chosen "for noncarbon-
    containing (inorganic) parents to be used as the basis for preferred IUPAC
    names for organic derivatives"; examples '(HO)3PO phosphoric acid (preselected
    name)' (:2076), 'sulfuric acid (preselected name)' (:35449), 'tetrachlorosilane
    (preselected name)' (:35758). The Blue Book labels no carbon-free structure
    '(PIN)': of its (PIN) names, the three whose extracted structure has no carbon
    are fragments of carbon-containing names ('... carbonazidimidoyl fluoride
    (PIN)':31484, 'bis(methanaminium) sulfate (PIN)':43564, '...-pentaphenoxy-
    λ5-phosphane (PIN)':46100). For the metal compounds also (:4667)
    "preferred IUPAC names have not yet been determined for inorganic components"
    and (:4712). A carbon-containing salt keeps its PIN status ('sodium
    hydrogen carbonate (PIN)',:31623; 'N,N-diethylethanaminium hydrogen sulfate
    (PIN)',:43566). Label only: the name is unchanged."""
    if not smiles:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return False
        return not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms())
    except Exception:
        return False


def _specified_stereo_count(mol) -> int:
    """Count of EXPLICITLY-specified stereo features (assigned chiral centres +
    directional double bonds). Used to tell a stereo CONFLICT (same count, different
    canonical) from stereo OMISSION (one side under-specifies)."""
    try:
        n = len(Chem.FindMolChiralCenters(
            mol, includeUnassigned=False, useLegacyImplementation=False))
    except TypeError:  # older RDKit signature
        n = len(Chem.FindMolChiralCenters(mol, includeUnassigned=False))
    for b in mol.GetBonds():
        if b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
                             Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS):
            n += 1
    return n


def _name_omits_input_stereo(input_smiles: str, opsin_smiles: str) -> bool:
    """True iff the name's OPSIN re-perception (``opsin_smiles``) specifies FEWER
    stereo features than the input structure — a stereo OMISSION, i.e. the name
    describes a less-specific (WRONG) molecule. Structural (counts on the parsed
    mols), never a string scan of the name. Returns False when the two are equal,
    when the name over-specifies, or when either is unparseable (fail-OPEN: an
    inconclusive comparison never rejects)."""
    mi = Chem.MolFromSmiles(input_smiles)
    mo = Chem.MolFromSmiles(opsin_smiles)
    if mi is None or mo is None:
        return False
    return _specified_stereo_count(mo) < _specified_stereo_count(mi)


def _self_consistency_verdict(input_smiles: str, opsin_smiles: str,
                              ignore_stereo: bool = False,
                              name: Optional[str] = None) -> str:
    """``"ok"`` | ``"mismatch"`` | ``"inconclusive"`` — does the OPSIN re-perception of
    the emitted name encode the SAME molecule as the input structure?

    Constitution = InChIKey skeleton block (formula + connectivity + mobile-H;
    stereo- and charge-insensitive) PLUS net formal charge PLUS the protonation
    flag of the full key (the input's charge form, for every input). On the stereo-strict
    primary path this ALSO fails a stereo CONFLICT (same amount of specified stereo,
    different RegistrationHash canonical layer). ``ignore_stereo=True`` (the OPSIN-validity
    stereo carve-out, which judges a stereo-STRIPPED OPSIN parse) compares constitution
    only — a stereo-strict compare there would suppress every stereo-bearing input by
    construction (see the OPSIN-validity stereo carve-out block below)."""
    # Strongest identity signal first: an EQUAL FULL InChIKey (skeleton + stereo +
    # normalized proton layer) means the name denotes the SAME molecule as the input
    # by the headline round-trip metric — accept immediately. This is 0-wrong-safe
    # (it only turns a spurious mismatch into ok) and closes a stereo-salt blind
    # spot: a stereo-bearing ionic salt (drug.[H+].[Cl-]) and the neutral molecular
    # form its correct name re-perceives to (drug.Cl) share one full key but differ
    # in the RegistrationHash TAUTOMER layer, so the stereo fallback below wrongly
    # returned "mismatch" (na>0 skips the na==0 short-circuit). Applies to the
    # ignore_stereo carve-out too: equal keys are the same molecule there as well.
    # Radical identity BEFORE the full-key shortcut: the full InChIKey encodes
    # neither radical electrons nor bond order, so '[CH2][CH2]' and 'C=C' (or an
    # aminoxyl and its charge form) share one key. When either side carries a
    # chemical radical, the parse must be the same radical graph
    # (validation/radical_identity.py).
    from .validation.radical_identity import radical_identity_verdict
    if radical_identity_verdict(input_smiles, opsin_smiles) == "mismatch":
        return "mismatch"
    # Protonation site BEFORE the full-key shortcut too: the standard InChI moves
    # an onium atom's hydrons into one mobile /p layer, so a name that puts the
    # '-ium' on the wrong nitrogen ('N-methyl-3-(methylamino)propanamidium' for
    # C[NH2+]CCC(=O)NC) shares the input's full key
    # (validation/protonation_identity.py).
    from .validation.protonation_identity import protonation_site_verdict
    if protonation_site_verdict(input_smiles, opsin_smiles, name) == "mismatch":
        return "mismatch"
    ka = _self_consistency_full_key(input_smiles)
    kb = _self_consistency_full_key(opsin_smiles)
    if ka is not None and kb is not None and ka == kb:
        return "ok"
    if ka is None and kb is None and _same_canonical_smiles(input_smiles, opsin_smiles):
        # Neither side has a standard InChIKey -- RDKit generates none above ~1000
        # heavy atoms (C1001 has one, C1024 has none) -- so both keys below are None
        # and the verdict used to be "inconclusive", shipped unverified at the
        # pin_verified label. Equal canonical isomeric SMILES are the same molecular
        # graph (element, charge, isotope, H count, stereo), so this is an identity,
        # not an approximation. Unequal SMILES stay "inconclusive" as before.
        # (TRIAGE j3-long-alkanes: 'hectakiliane' for C1100.)
        return "ok"
    a = _self_consistency_skeleton(input_smiles)
    b = _self_consistency_skeleton(opsin_smiles)
    if a is None or b is None:
        return "inconclusive"
    if a != b:
        return "mismatch"
    # Skeleton matches -- the charge form must match too. The skeleton block
    # excludes the charge (/q) and protonation (/p) layers, so a name that drops,
    # adds or moves a charge or a hydron shares it with the input:
    # * the hydroperoxide anion [O-]O (-1) named 'dioxidane' (OO, 0);
    # * 'trisodium 5-hydroxybenzene-1,3-disulfonate', whose parse is three Na+
    # and a dianion (net +1), for the trianion salt (net 0) --,
    # the Blue Book: "Neutral salts of acids are named by citing the name
    # of the cation(s) followed by the name of the anion";
    # * 'methyl phosphate', whose parse is the dianion, for the neutral ester
    # COP(=O)(O)O. That ester is 'methyl dihydrogen phosphate',
    # the Blue Book: "Partial acid esters of polybasic acids are named by
    # citing alkyl groups... followed by the word 'hydrogen'... and the name
    # of the appropriate anion"; example:35940), so a name without the
    # 'hydrogen' denotes the anion, a different species. Neutral inputs used
    # to be exempt from this check; they are not.
    # So the net charges must be equal, and so must the protonation flag (the last
    # character of the full key, from the /p layer). A charge form that IS the same
    # species (an ionic salt drug.[H+].[Cl-] read back as drug.Cl, an amino-acid
    # zwitterion read back as the neutral acid) shares the whole full key and was
    # accepted above.
    ca = _self_consistency_net_charge(input_smiles)
    cb = _self_consistency_net_charge(opsin_smiles)
    if ca is not None and cb is not None and ca != cb:
        return "mismatch"
    if ka is not None and kb is not None and ka[-1] != kb[-1]:
        return "mismatch"
    if ignore_stereo:
        return "ok"
    # C6: constitution + charge match. Catch a name whose OPSIN re-perception
    # encodes a DIFFERENT stereoisomer. Stereo OMISSION (one side under-specifies) is
    # not a constitutional error and is tolerated; only a CONFLICT (same AMOUNT of
    # specified stereo, different configuration) suppresses. Gold PINs round-trip
    # their stereo exactly, so this never fires on them.
    mi = Chem.MolFromSmiles(input_smiles)
    mo = Chem.MolFromSmiles(opsin_smiles)
    if mi is None or mo is None:
        return "ok"
    na = _specified_stereo_count(mi)
    nb = _specified_stereo_count(mo)
    # a phase (C3): a name that OVER-specifies stereo (nb > na) FABRICATES
    # configuration the input does not define -- "CIP Priority and Sequence
    # Rules" / "Configuration Specification" (the Blue Book):
    # a stereodescriptor asserts a specific configuration, so it may only be used
    # when that configuration is actually defined; an undefined centre is left
    # undescribed, never assigned an arbitrary one -> reject. na == 0 alone
    # is no longer a blanket pass: only the truly achiral case (na == 0 AND nb == 0)
    # stays ok. Producer guards (C2) prevent most fabrication reaching here; this is
    # the OPSIN-parseable backstop.
    if nb > na:
        return "mismatch"
    if na == 0:
        return "ok"
    # a phase: a name that specifies FEWER stereo features than the input OMITS
    # defined stereo -> it describes a less-specific WRONG molecule. Reject it (0-wrong);
    # the tier degrades (abstain at default; best-effort/later phases re-name with full
    # stereo). Structural count, never a name-string scan.
    if _name_omits_input_stereo(input_smiles, opsin_smiles):
        return "mismatch"
    # Equal specified-stereo count. When the two are the EXACT same constitution
    # (isomeric-stripped canonical equal — NOT merely mobile-H/tautomer-equivalent),
    # decide by a SOUND per-element atom-mapped compare: the name's asserted structure
    # as a chiral substructure query into the input. This covers tetrahedral R/S AND
    # double-bond E/Z. The prior RegistrationHash TAUTOMER_HASH layer was E/Z-BLIND on
    # conjugated systems (fumarate vs maleate hashed EQUAL), so it silently shipped a
    # wrong geometric isomer. Probes: a temp dir/risk3_stereo_probe*.py; audit and
    # derivation: internal notes (RISK 3 / C6).
    _mi2, _mo2 = Chem.Mol(mi), Chem.Mol(mo)
    Chem.RemoveStereochemistry(_mi2)
    Chem.RemoveStereochemistry(_mo2)
    if Chem.MolToSmiles(_mi2) == Chem.MolToSmiles(_mo2):
        return "ok" if mi.HasSubstructMatch(mo, useChirality=True) else "mismatch"
    # Different EXACT constitution but the InChIKey skeleton already matched => a
    # tautomer / mobile-H difference. Stay tautomer-tolerant: fall back to the
    # TAUTOMER_HASH stereo layer (it normalizes the tautomer, leaving only a genuine
    # tetrahedral stereo difference to catch on this rare same-count path).
    sa = _registration_stereo_layer(input_smiles)
    sb = _registration_stereo_layer(opsin_smiles)
    if sa is None or sb is None or sa == sb:
        return "ok"
    return "mismatch"


def _record_gate_outcome(outcome: str, name: Optional[str]) -> None:
    """ T1: publish what the validity gate DID for ``name``.

    Observation-only (two ContextVar writes) — no gate BEHAVIOUR depends on
    it, and no emitted name changes. `name_tiered` derives `opsin` /
    `gates_passed` from this instead of from jar presence. ``name`` is the
    string the caller is about to RETURN, so a suppression records the
    fallback it shipped rather than the candidate it rejected.

    Threaded through a contextvar rather than the gate's return value on
    purpose: `_final_opsin_validity_gate` returns a `str` that four call sites
    treat as the name, and widening that contract to a tuple would touch every
    one of them.
    """
    from .metrics.provenance import record_gate_outcome
    record_gate_outcome(outcome, name)


def _self_consistency_decision(name: str, smiles: Optional[str], opsin_smiles: str,
                               stats: Optional[Dict[str, int]],
                               ignore_stereo: bool = False) -> str:
    """: OPSIN already PARSED ``name``; verify it parses to the SAME molecule.

    Suppresses (mode "on") only on a VERIFIED constitutional (or, on the stereo-strict
    primary path, stereo-CONFLICT) mismatch — never on a stereo-omission / tautomer
    difference, never when the comparison is inconclusive (fail-OPEN). In mode "warn"
    it logs + counts the would-suppression but ships the name unchanged; in mode "off"
    it is a no-op. ``ignore_stereo=True`` is used by the OPSIN-validity stereo carve-out,
    which judges a stereo-STRIPPED parse and must stay stereo-insensitive."""
    from .metrics import provenance as _pv
    if _SC_MODE == "off" or not smiles:
        _record_gate_outcome(_pv.GATE_OUTCOME_SELF01_SKIPPED, name)
        return name
    verdict = _self_consistency_verdict(smiles, opsin_smiles, ignore_stereo=ignore_stereo,
                                        name=name)
    if verdict == "inconclusive" and _SC_MODE == "on":
        # R19 (claims conformance, 2026-09-27): "inconclusive" means a skeleton
        # key could not be computed for one side, so nothing was compared, and
        # this used to ship the name (fail-OPEN). The claim is a round trip for
        # every name: run the FULL-key one (no fail-open branch) and ship only on
        # a pass; otherwise the name is withdrawn like any unverified name.
        # (The warn configuration keeps its documented meaning: it ships.)
        if _shipped_name_round_trip(name, smiles) == "verified":
            _record_gate_outcome(_pv.GATE_OUTCOME_FULL_KEY_VERIFIED, name)
            return name
        if stats is not None:
            stats["self_consistency_inconclusive_suppressed"] = (
                stats.get("self_consistency_inconclusive_suppressed", 0) + 1)
        logger.warning("self_consistency inconclusive and no full-key round trip, "
                       "suppressed: %r (opsin=%s)", name[:80], opsin_smiles)
        from .metrics.abstention import AbstentionCode, record_suppression
        record_suppression(AbstentionCode.GATE_SUPPRESSED,
                           detail='self01_inconclusive', candidate=name)
        _suppressed_to = _descriptive_fallback(smiles)
        _record_gate_outcome(_pv.GATE_OUTCOME_SUPPRESSED, _suppressed_to)
        return _suppressed_to
    if verdict != "mismatch":
        # T1: "ok" is the ONE state that earns. "inconclusive"
        # ships by failing OPEN — the comparison could not be made, so nothing
        # was verified and it must not claim to have been.
        _record_gate_outcome(
            _pv.GATE_OUTCOME_SELF01 if verdict == "ok"
            else _pv.GATE_OUTCOME_SELF01_INCONCLUSIVE, name)
        return name  # "ok" ships; "inconclusive" fails OPEN
    if stats is not None:
        stats["self_consistency_mismatch"] = stats.get("self_consistency_mismatch", 0) + 1
    if _SC_MODE == "warn":
        logger.warning(
            "SELF-01 would-suppress (warn-only): %r names a constitutionally DIFFERENT "
            "molecule (input=%s opsin=%s)", name[:80], smiles, opsin_smiles)
        _record_gate_outcome(_pv.GATE_OUTCOME_SELF01_WARN_MISMATCH, name)
        return name
    # mode "on": suppress to the honest descriptive fallback.
    if stats is not None:
        stats["self_consistency_suppressed"] = stats.get("self_consistency_suppressed", 0) + 1
    logger.warning("self_consistency rejected (different molecule): %r (opsin=%s)",
                   name[:80], opsin_smiles)
    from .metrics.abstention import AbstentionCode, record_suppression
    record_suppression(AbstentionCode.GATE_SUPPRESSED, detail='self01_mismatch',
                       candidate=name)
    _suppressed_to = _descriptive_fallback(smiles)
    _record_gate_outcome(_pv.GATE_OUTCOME_SUPPRESSED, _suppressed_to)
    return _suppressed_to


def _grammar_gap_carveout(name: str) -> Optional[str]:
    """The carve-out slug of a construction-verified name whose CLASS OPSIN 2.9.0
    has no grammar for at all, or None.

    For these classes a healthy OPSIN call can only answer 'rejected' (each
    docstring above records the verification: no thioperoxol suffix family, no
    inositol or Table 10.1 stereoparent name, no 'di'-collapsed or chalcogen
    anhydride class word, no multi-anion functional-class ester, no phane grammar,
    no substituted halogen-uide), so the name ships on its construction alone and
    OPSIN's answer adds nothing. The validity gate therefore consults this both
    after a 'rejected' answer and on 'unavailable': the output of these classes does
    not depend on whether OPSIN could be reached (TRIAGE j12 finding 1).

    NOT here: the organometallic additive carve-out (OPSIN parses some '-ido'
    ligand names, and the class has a known atom-drop hole, so a healthy run's
     on a parsed name is a real check), the stereo-layer carve-out (its
    licence is a parse of the stereo-stripped form) and the polyacid carve-out
    (reached only after a parse)."""
    if "thioperoxol" in name and not re.search(r"[A-Za-z](SO|OS)-thioperoxol", name):
        return "thioperoxol"
    try:
        from .rules.inositols import INOSITOL_NAMES
        if name in INOSITOL_NAMES:
            return "inositol"
    except Exception:
        pass
    try:
        from .data.natural_products import NAME_EXACT_NP_PARENTS
        if name in NAME_EXACT_NP_PARENTS:
            return "np_stereoparent"
    except Exception:
        pass
    if _DIANHYDRIDE_PIN_RE.match(name):
        return "dianhydride"
    if _CHALCOGEN_DIANHYDRIDE_PIN_RE.match(name):
        return "chalcogen_dianhydride"
    if _POLYOL_POLYESTER_PIN_RE.match(name):
        return "polyol_polyester"
    if _PHANE_PIN_RE.match(name):
        return "phane"
    if _HALOGEN_UIDE_PIN_RE.match(name):
        return "halogen_uide"
    return None


def _ship_grammar_gap_carveout(name: str, slug: str) -> str:
    """Record the carve-out outcome of ``slug`` and return ``name`` unchanged."""
    from .metrics import provenance as _pv
    _record_gate_outcome(_pv.carveout_outcome(slug), name)
    if slug == "np_stereoparent":
        # Shipped by design (controller ruling 2026-09-25, data/natural_products
        #.get_natural_product_name), but OPSIN cannot parse it, so it is not
        # verified, and (the Blue Book) identifies no PIN for these
        # parents: the name must not be LABELLED pin_verified. Name-scoped record;
        # the name itself is unchanged.
        _pv.record_non_pin_fragment(name)
    return name


def _final_opsin_validity_gate(name: str, smiles: Optional[str],
                               stats: Optional[Dict[str, int]] = None,
                               *, besteffort_unverified: bool = False,
                               general_fallback_tier: bool = False) -> str:
    """ real-OPSIN validity gate (//).

    Suppress an OPSIN-unparseable production name to the EXISTING
    _descriptive_fallback STRING (never None / never a shipped invalid string).
    Runs AFTER stereo + grammar repair, as the last transform before the value
    leaves name/name_with_confidence; inside the is_top_level_naming guard
    so it never fires on decomposition fragments.

    ``besteffort_unverified`` (task-JAR-ABSENT, a review 0-wrong hole): the four
    name/name_with_confidence call sites pass ``self._general_fallback_unverified``.
    When True AND the OPSIN jar is GENUINELY absent (a no-Java deployment), the
     jar-absent fail-OPEN below is replaced by a fail-CLOSED suppression:
    at best-effort tier we cannot constitutionally verify ANY name without OPSIN
    (the Wave-0 reconstructor has no NameFacts extractor for an arbitrary emitted
    name yet -- ``verify_or_none(name, smiles, name_facts=None)`` is provably None
    on every jar-absent call), and a review proved this branch ships WRONG-molecule
    names at best-effort in a no-Java env (``COS(=O)(=O)O`` -> ``methane``). 0-wrong
    is ABSOLUTE, so an unverifiable best-effort emission must abstain, not ship.
    Default False keeps the historical jar-absent fail-OPEN for the PIN/default
    tier and for every direct unit-test caller (2-/3-positional-arg callers are
    unaffected). See task-jar-absent-report.md for why the default/PIN tier is a
    separate, larger no-Java-PIN policy decision left as a scoped follow-up.
    """
    from .metrics import provenance as _pv
    if not name:
        # The ONE return that records nothing, deliberately: there is no name
        # to have an outcome about, and writing one here would OVERWRITE the
        # outcome of a real earlier gate call in the same naming session (the
        # retry cascade calls this gate more than once). The contextvar's
        # NOT_RUN default already fails closed.
        return name
    if _DISABLE_VALIDITY_GATE:
        _record_gate_outcome(_pv.GATE_OUTCOME_DISABLED, name)
        return name
    # Don't re-gate an already-descriptive fallback (it won't OPSIN-parse, and
    # re-suppressing is idempotent anyway) — skip the wasted OPSIN subprocess.
    try:
        from .errors import _DESCRIPTIVE_FALLBACK_NAMES
        if name in _DESCRIPTIVE_FALLBACK_NAMES:
            _record_gate_outcome(_pv.GATE_OUTCOME_DESCRIPTIVE_FALLBACK, name)
            return name
    except Exception:
        pass
    # fail-OPEN: probe the JAR FIRST.
    if not _validity_gate_jar_present():
        # task-JAR-ABSENT (a review 0-wrong hole): a GENUINELY absent jar means no
        # constitutional verification is possible for this name -- no OPSIN, and
        # the Wave-0 reconstructor needs a NameFacts extractor that does not yet
        # exist for an arbitrary emitted name (verify_or_none(name_facts=None) is
        # provably None on every jar-absent call). At BEST-EFFORT tier
        # (general_fallback_unverified) that unverified branch is exactly the
        # class a review showed ships WRONG-molecule names in a no-Java deployment
        # (COS(=O)(=O)O -> methane; ClP(Cl)(=O)OC1=CC=CC=C1 ->
        # (phosphonooxy)benzene, Cl2 silently swapped for (OH)2). 0-wrong is
        # ABSOLUTE, so fail CLOSED to the honest fallback rather than ship it.
        # The default/PIN tier (besteffort_unverified=False) keeps the historical
        # fail-OPEN so this cannot touch the 1652 gate's correct-by-
        # construction OPSIN-unparseable carve-out classes (inositol,
        # np_stereoparent, thioperoxol, dianhydride, phane,...) -- a separate
        # no-Java-PIN policy question left as a scoped follow-up (see report).
        if besteffort_unverified:
            from .metrics.abstention import AbstentionCode, record_suppression
            record_suppression(
                AbstentionCode.GATE_SUPPRESSED,
                detail='jar_absent_besteffort_unverified', candidate=name)
            _suppressed_to = _descriptive_fallback(smiles)
            _record_gate_outcome(_pv.GATE_OUTCOME_SUPPRESSED, _suppressed_to)
            return _suppressed_to
        _record_gate_outcome(_pv.GATE_OUTCOME_UNAVAILABLE, name)
        return name
    # (code review 2026-06-02): 'rejected' and 'unavailable' are different
    # facts. The old `parse-or-None` check conflated them. 'unavailable'
    # (subprocess timeout / OSError / JVM crash mid-run on a loaded host) is no
    # verdict at all; it used to fail OPEN and now fails CLOSED (TRIAGE g7 C01,
    # below), after OpsinOracle._one_shot has retried it with a longer budget.
    #
    # (a phase): re-perceive the name via OPSIN. A non-None canonical
    # SMILES means OPSIN PARSED it -> run the constitutional self-consistency check
    # (suppress if the name encodes a DIFFERENT molecule). None means rejected or
    # unavailable; only then consult parse_status to distinguish them. Using
    # name_to_smiles as the primary probe also avoids a second OPSIN subprocess on
    # the common parsed path.
    opsin_smiles = _validity_gate_name_to_smiles(name)
    if opsin_smiles is not None:
        # BE-STRICT full-RT (2026-08-30, user-directed): at the general-fallback
        # tiers (best-effort / complete) the contract is a FULL round-trip
        # (constitution AND stereo AND charge), so a name that OPSIN parses to a
        # DIFFERENT full InChIKey than the input — a wrong molecule, an
        # OPSIN-valence artifact (e.g. an organo-tin `stanna`-replacement name
        # whose re-parsed Sn valence differs), OR a stereo-INCOMPLETE name (a
        # stereo-stripped/omitted descriptor is a WRONG name, not a lesser one) —
        # MUST abstain, never ship. The PIN/default tier keeps the skeleton-based
        # below (stereo-tolerant by design for the gold PINs), so the 1656
        # gate is byte-identical (this branch is skipped when general_fallback_tier
        # is False). Fail-OPEN only when the input key is uncomputable.
        if general_fallback_tier and smiles is not None:
            _in_key = _self_consistency_full_key(smiles)
            if _in_key is not None:
                _out_key = _self_consistency_full_key(opsin_smiles)
                # abstain when the round-trip CANNOT be confirmed equal: either the
                # OPSIN output is itself RDKit-unparseable (an organo-metal valence
                # artifact — _out_key None) OR it computes to a DIFFERENT full key
                # (wrong molecule / stereo-incomplete). Only a computable, EQUAL key
                # ships at the RT-verified tier. An equal key is not enough when a
                # radical is involved (validation/radical_identity.py).
                from .validation.radical_identity import radical_identity_verdict
                _radical_bad = radical_identity_verdict(smiles, opsin_smiles) == "mismatch"
                if _out_key is None or _out_key != _in_key or _radical_bad:
                    # a phase stereo-OMISSION reclaim (0-wrong ABSOLUTE). When the
                    # difference is stereo-ONLY — the CONSTITUTION round-trips (skeleton
                    # InChIKey equal) but the flat name dropped the input's stereo — try
                    # composing that stereo back on (RT-gated, OPSIN's own numbering) and
                    # ship it ONLY if it recomputes to the input's full key. The skeleton
                    # prefilter keeps the ~wrong-CONSTITUTION suppressions off this
                    # (2-OPSIN-call) compose path, so no added hang surface on the giants;
                    # a genuine stereo-CONFLICT never composes to _in_key and still
                    # suppresses. Same 0-wrong bar as the RT-verified tier.
                    _in_skel = _self_consistency_skeleton(smiles)
                    if (_STEREO_OMISSION_RECLAIM and not _radical_bad
                            and _out_key is not None and _in_skel is not None
                            and _in_skel == _self_consistency_skeleton(opsin_smiles)):
                        _composed = _try_compose_input_stereo(name, smiles, _in_key)
                        if _composed is not None:
                            # Full-InChIKey verified (a review -P2 F3): record a
                            # VERIFIED outcome, not a carve-out. The composed name
                            # was proven to recompute to the input's full key —
                            # stronger than — so the public provenance must
                            # report it "verified", not "unverified".
                            _record_gate_outcome(
                                _pv.GATE_OUTCOME_STEREO_RECOMPOSED, _composed)
                            return _composed
                    from .metrics.abstention import AbstentionCode, record_suppression
                    record_suppression(
                        AbstentionCode.GATE_SUPPRESSED,
                        detail='full_rt_unconfirmed_general_fallback', candidate=name)
                    _suppressed_to = _descriptive_fallback(smiles)
                    _record_gate_outcome(_pv.GATE_OUTCOME_SUPPRESSED, _suppressed_to)
                    return _suppressed_to
        # W3-P10: correct-by-construction polyacid prefix-derivative PIN that OPSIN
        # RE-PARSES to the wrong constitution (locant-assignment bug). Ship it —
        # the BB is the sole PIN authority and OPSIN's bug must not gate us.
        if _POLYACID_PREFIX_DERIVATIVE_PIN_RE.match(name):
            # Review fixes: the carve-out exists for OPSIN's wrong re-parse only; a
            # round trip that verifies the name ('2-thiodisulfuric acid') is recorded
            # as such, not pre-empted by the carve-out.
            if (_SC_MODE == "on" and smiles is not None
                    and _self_consistency_verdict(smiles, opsin_smiles, name=name) == "ok"):
                _record_gate_outcome(_pv.GATE_OUTCOME_SELF01, name)
                return name
            _record_gate_outcome(
                _pv.carveout_outcome("polyacid_prefix_derivative"), name)
            return name
        # _self_consistency_decision records its own outcome (one per verdict).
        return _self_consistency_decision(name, smiles, opsin_smiles, stats)
    # opsin_smiles is None -> either OPSIN was UNAVAILABLE (transient: timeout /
    # OSError / JVM crash, the jar being present), OR OPSIN PARSED the name but
    # emitted a SMILES that RDKit cannot canonicalise (an impossible valence — e.g.
    # the lambda6 candidate '1,2lambda6,3-dioxathiolane' produced for a
    # lambda6-multiring-spiro that the spiro engine fail-closes). A
    # 'parsed'-but-uncanonicalisable output is UNVERIFIABLE: we cannot confirm the
    # name describes the input structure, so it must fall through to the whitelist
    # carve-outs below and, absent a whitelist hit, suppress to the honest fallback
    # (fail-closed, accuracy #1). Conflating the two here shipped the wrong name.
    #
    # 'unavailable' fails CLOSED too (TRIAGE g7 C01, 2026-09-27; it used to fail
    # OPEN,). It is not evidence against the name, but it is no evidence FOR
    # it either, and 0-wrong is absolute: every emission verified, else abstain.
    # Measured in a fresh process with both probes stubbed 'unavailable' (the state
    # an OPSIN timeout on a loaded host leaves): the quinoxaline-quinoline
    # c1ccc2nc3ccc4ncccc4c3nc2c1 shipped as 'benzene', the bromo-indolyl imidazolyl
    # methanone lost its Br, the thiourea w18 lost its thioxo -- each a different
    # molecule at the pin_unverified label. 's worry (a valid name lost to a
    # hiccup) is met where it arises: OpsinOracle._one_shot retries a failed
    # OPSIN call with a longer budget before it reports 'unavailable'. The
    # carve-outs whose licence needs an OPSIN observation are NOT consulted: the
    # stereo-layer carve-out needs a parse of the stripped form, and the
    # organometallic additive one a 'rejected' (OPSIN parses some of its names,
    # so is a real check there). The grammar-gap classes ARE consulted
    # (TRIAGE j12 finding 1): OPSIN has no grammar for them, a healthy call can
    # only reject them, and they ship on their construction alone, so the gate's
    # verdict on them must not depend on whether OPSIN was reachable
    # (_grammar_gap_carveout). The same tier rule applies as after a 'rejected'
    # answer: at the general-fallback tiers only the exact NP stereoparents ship.
    # A producer that assembles such a name from fragment names the gate must
    # verify (the glyceride polyester reads 'acetic acid' / 'propanoic acid'
    # through this gate) still cannot build it while OPSIN is unreachable: that
    # dependence is the fragments', and they fail closed like any other name.
    # The jar-ABSENT reduced mode (, above) is a separate, opt-in
    # configuration and keeps its own policy.
    if _validity_gate_status(name) == "unavailable":
        _gap_slug = _grammar_gap_carveout(name)
        if _gap_slug is not None and (not general_fallback_tier
                                      or _gap_slug == "np_stereoparent"):
            return _ship_grammar_gap_carveout(name, _gap_slug)
        if stats is not None:
            stats["opsin_unavailable_suppressed"] = (
                stats.get("opsin_unavailable_suppressed", 0) + 1)
        logger.warning("OPSIN validity gate: OPSIN unavailable, suppressed "
                       "unverified name: %r", name[:60])
        from .metrics.abstention import AbstentionCode, record_suppression
        record_suppression(AbstentionCode.GATE_SUPPRESSED,
                           detail='opsin_unavailable', candidate=name)
        _suppressed_to = _descriptive_fallback(smiles)
        _record_gate_outcome(_pv.GATE_OUTCOME_SUPPRESSED, _suppressed_to)
        return _suppressed_to
    # BE-STRICT (2026-08-30, user-directed): at the general-fallback tiers
    # (best-effort / complete — ``general_fallback_tier`` True), the contract is
    # an OPSIN-ROUND-TRIP-VERIFIED name, so an OPSIN-DEFINITIVELY-REJECTED name
    # cannot be verified and MUST abstain rather than ship. This makes the
    # "ship-an-OPSIN-unparseable name" carve-outs below (thioperoxol / inositol /
    # np_stereoparent / dianhydride / phane / halogen-uide / the stereo-layer
    # carve-out) PIN/DEFAULT-TIER ONLY: those names are the correct, gold-validated
    # PIN and MUST still ship at the PIN tier (``general_fallback_tier`` False, so
    # this branch is skipped and the 1656 PIN gold gate is byte-identical). At the
    # RT-verified tiers a correct-but-OPSIN-unparseable name is an honest abstain,
    # never a stereo-stripped or unparseable emission (accuracy #1: a name that
    # does not round-trip is not a shippable name). Closes the precision leak that
    # let 5/1435 OPSIN-unparseable names ship at best-effort (a review + the 1500-mol
    # head-to-head). PIN path untouched.
    if general_fallback_tier:
        #: EXEMPT the construction-verified NP stereoparent carve-out
        # (germacrane, corynoxan,... -- the retained parent
        # hydrides). These are 0-wrong by EXACT canonical-SMILES table membership
        # (NAME_EXACT_NP_PARENTS) -- a stronger guarantee than OPSIN-RT -- and
        # OPSIN 2.9.0 simply lacks the generation grammar for them
        # (name_to_smiles -> None). Suppressing a curated, exact-match PIN here is a
        # defect (best-effort abstains where a verified name exists, a project rule):
        # `germacrane`'s systematic form does not round-trip, so best-effort was
        # dropping the retained PIN to the abstention sentinel instead of keeping it.
        # They ship via the ``np_stereoparent`` carve-out below. This is NARROW ON
        # PURPOSE: the other OPSIN-unparseable carve-outs stay best-effort-suppressed
        # (the ``organometallic_additive`` one has a known atom-drop hole, and the
        # stereo-layer carve-out is exactly the precision class BE-STRICT targets).
        try:
            from .data.natural_products import NAME_EXACT_NP_PARENTS as _NP_EXACT
            _np_exact_hit = name in _NP_EXACT
        except Exception:
            _np_exact_hit = False
        if not _np_exact_hit:
            from .metrics.abstention import AbstentionCode, record_suppression
            record_suppression(
                AbstentionCode.GATE_SUPPRESSED,
                detail='opsin_unparseable_general_fallback', candidate=name)
            _suppressed_to = _descriptive_fallback(smiles)
            _record_gate_outcome(_pv.GATE_OUTCOME_SUPPRESSED, _suppressed_to)
            return _suppressed_to
        # else: fall through to the np_stereoparent carve-out (ships the retained PIN).
    # The grammar-gap carve-outs (TRIAGE j12 finding 1 put them in one helper, used
    # here and on 'unavailable' above), in their historical order:
    # - DD2 / Phase D thioperoxol: OPSIN's generation grammar does not recognise the
    # chalcogen-peroxol suffix family ('-SO-thioperoxol', '-OS-thioperoxol',
    # '-dithioperoxol'), so it REJECTS these correct PINs verbatim:
    # `CH3-S-OH -> methane-SO-thioperoxol (PIN)`). guard: only a WELL-FORMED
    # thioperoxol name; the italic 'SO-'/'OS-' descriptor must be separator-led
    # (a multiplied form glued onto the descriptor, 'ethane-1,2-bisSO-thioperoxol',
    # is a real formatting defect and stays suppressed).
    # - a phase inositol retained names, OPSIN-UNPARSEABLE (verified
    # name_to_smiles -> None for every one), produced ONLY by the hard-gated
    # InChIKey recogniser (rules.inositols.name_inositol).
    # - a phase a/c) terpene/alkaloid stereoparents,
    # OPSIN-UNPARSEABLE, produced ONLY by the exact-canonical-SMILES
    # NATURAL_PRODUCT_DERIVATIVES lookup; recorded non-PIN:50943).
    # - W3-P06 di-/polyanhydride PINs: OPSIN's anhydride word-rule
    # cannot parse the 'di-'-COLLAPSED acid words ("Unexpected number of words in
    # anhydride"); emitted ONLY by the hard-gated rules.anhydrides._name_dianhydride.
    # - W8-P4 chalcogen di-/poly-anhydride PINs (see the regex).
    # - W3-P07 method (1)) functional-class polyol polyester PIN
    # ('propane-1,2,3-triyl 1,3-diacetate 2-propanoate') from _assemble_glyceride;
    # OPSIN cannot parse the multi-anion syntax.
    # - W8-P8 Task 8.12 /.3) phane PINs from the hard-gated
    # rules.phane.build_phane_pin (formula-conservation-vetoed); OPSIN has no phane
    # grammar at all.
    # - W8-P5 Task 2 / substituted halogen-uide PINs
    # ('diphenyliodanuide') from the hard-gated rules.ions._emit_group13_uide
    # halogen branch (atom-conservation-vetoed).
    _gap_slug = _grammar_gap_carveout(name)
    if _gap_slug is not None:
        return _ship_grammar_gap_carveout(name, _gap_slug)
    # W8-P9 Task 9.1 /.4/.6): additive/coordination organometallic
    # PIN carve-out -- see _ORGANOMETALLIC_ADDITIVE_PIN_RE docstring above. Not a
    # grammar-gap class (OPSIN parses some of its names), so it is reached only
    # after a 'rejected' answer. It is mutually exclusive with the phane and
    # halogen-uide patterns (different endings), so checking it after them keeps
    # every outcome.
    if _ORGANOMETALLIC_ADDITIVE_PIN_RE.match(name):
        _record_gate_outcome(
            _pv.carveout_outcome("organometallic_additive"), name)
        return name
    # OPSIN-validity stereo carve-out / (a phase): decide on WHERE OPSIN fails. If the name is
    # rejected ONLY because of its stereo layer — i.e. the stereo-STRIPPED
    # constitutional form parses — then OPSIN's narrower generation-side stereo
    # grammar must NOT gate Orthonym correctness (audit Dim-08 §C; the verbatim
    # Blue Book PIN `(1s,4s)-cyclohexane-1,4-diol` was being suppressed to
    # `unknown`). The constitutional gate stays STRICT: a name whose stereo-stripped
    # form ALSO fails to parse is still suppressed. strip_stereo is read-only — the
    # shipped name keeps its stereo descriptors.
    #
    # (2026-07-27): this branch used to `return name` outright, skipping
    #, so ANY constitutional defect rode out free as long as the name
    # happened to carry a stereo prefix OPSIN rejects — verification was strongest
    # on well-formed names and ABSENT on malformed ones, exactly backwards. Measured
    # on benchmarks/pubchem_2000.csv with the gates ON: 107 names shipped through
    # here and 98 of them named a DIFFERENT molecule (witness:
    # COC(=O)NCC[C@@H]1CC[C@H]2[C@@H]1C2(Br)Br shipped as `(1S,4S,5R)-methyl
    # N-octylcarbamate` — ring + both Br silently dropped); 0 of the 1641 gold
    # targets were affected.
    #
    # The carve-out's licence is narrow — OPSIN need not parse the stereo LAYER —
    # and it was never a licence to ship an UNVERIFIED CONSTITUTION. The comparison
    # was available all along: _self_consistency_verdict compares the InChIKey
    # SKELETON block, which EXCLUDES stereochemistry by construction (-07),
    # so it can judge the stereo-STRIPPED parse with no loss of validity. Route it
    # through the SAME decision the parsed path uses: a verified
    # constitutional mismatch suppresses, while `ok`/`inconclusive` still ship the
    # FULL stereo name (fail-OPEN on any comparison that could not be made).
    from .rules.stereochemistry import strip_stereo
    _stripped = strip_stereo(name)
    if _stripped != name:
        _stripped_status = _validity_gate_status(_stripped)
        # -FOLLOWUP: _validity_gate_status is THREE-valued, and the stripped
        # probe is a second, independent OPSIN consultation that can hiccup on its
        # own. 'unavailable' (subprocess timeout / OSError on a loaded host) means
        # OPSIN was never consulted: no evidence against the name, and none for its
        # CONSTITUTION either -- which is the one thing this carve-out must verify
        # (: it was never a licence to ship an unverified constitution). So it
        # fails CLOSED, exactly as the primary probe does (TRIAGE g7 C01,
        # 2026-09-27; it used to ship the name here with nothing checked).
        # OpsinOracle._one_shot's retry ladder keeps a slow OPSIN from costing the
        # names this carve-out exists to rescue (the verbatim BB PIN
        # `(1s,4s)-cyclohexane-1,4-diol`). A jar that is genuinely missing is the
        # _validity_gate_jar_present reduced mode above, not this branch.
        if _stripped_status == "unavailable":
            if stats is not None:
                stats["gate_stereo_unavailable"] = (
                    stats.get("gate_stereo_unavailable", 0) + 1)
            logger.warning("OPSIN validity gate: OPSIN unavailable for the "
                           "stereo-stripped probe, suppressed: %r", name[:60])
            from .metrics.abstention import AbstentionCode, record_suppression
            record_suppression(AbstentionCode.GATE_SUPPRESSED,
                               detail='opsin_unavailable_stereo_probe',
                               candidate=name)
            _suppressed_to = _descriptive_fallback(smiles)
            _record_gate_outcome(_pv.GATE_OUTCOME_SUPPRESSED, _suppressed_to)
            return _suppressed_to
        # 'rejected' is unchanged: a stripped form OPSIN DEFINITIVELY rejects is a
        # genuine constitutional defect and falls through to the fallback below.
        if _stripped_status == "parsed":
            _stripped_smiles = _validity_gate_name_to_smiles(_stripped)
            if _stripped_smiles is not None:
                # ignore_stereo: this carve-out judges the stereo-STRIPPED parse, so
                # the verdict MUST stay stereo-insensitive (a stereo-strict compare
                # would suppress every stereo-bearing input here by construction).
                _decided = _self_consistency_decision(name, smiles, _stripped_smiles,
                                                      stats, ignore_stereo=True)
                if _decided != name:
                    # Own counter: _self_consistency_decision bumps the SHARED
                    # self_consistency_{mismatch,suppressed} counters for both the
                    # primary parsed path and this carve-out, so telemetry cannot
                    # otherwise tell the two apart.
                    if stats is not None:
                        stats["gate_stereo_mismatch"] = (
                            stats.get("gate_stereo_mismatch", 0) + 1)
                    # `_decided` is the fallback; _self_consistency_decision
                    # already recorded GATE_OUTCOME_SUPPRESSED against it.
                    return _decided  # PROVED a different molecule -> suppress
                if stats is not None:
                    stats["gate_stereo_kept"] = stats.get("gate_stereo_kept", 0) + 1
                # T1: judged the stereo-STRIPPED parse, so an
                # "ok" here verifies the CONSTITUTION ONLY — the stereo layer
                # was never checked (that is the whole point of this
                # carve-out). Downgrade the inner verdict to say so. Any other
                # inner outcome (inconclusive / skipped / warn-mismatch) is
                # already non-verifying and is left exactly as recorded.
                if (_pv.get_provenance()["gate_outcome"]
                        == _pv.GATE_OUTCOME_SELF01):
                    # critic-1 / R63 (claims conformance, 2026-09-27): the
                    # carve-out's licence is that OPSIN's stereo grammar is
                    # narrower than the Blue Book's (lowercase r/s), never that
                    # the descriptors need no check. Each one is checked against
                    # the input's CIP labels at its locant; a name with a wrong
                    # one is repaired to a full-key round-tripping name or
                    # withdrawn (_stereo_descriptor_repair).
                    if not _stereo_descriptors_verified(name, smiles, _stripped):
                        return _stereo_descriptor_repair(
                            name, smiles, _stripped, stats)
                    _record_gate_outcome(
                        _pv.GATE_OUTCOME_SELF01_CONSTITUTION_ONLY, name)
                return name  # only the stereo layer is OPSIN-narrow -> ship it whole
            # 'parsed' but NO usable SMILES: OPSIN accepted the stripped name and
            # then emitted a structure RDKit cannot canonicalise (an impossible
            # valence) — the identical situation the non-stereo path already settled
            # above, where 'parsed'-but-uncanonicalisable is UNVERIFIABLE and
            # suppressed. It is
            # not merely an unmade measurement: it is positive evidence that the name
            # denotes a structure RDKit rejects as impossible. A stereo layer must
            # not lower the burden of proof, so fall through to the honest fallback
            # exactly as that path does. Measured: 1 corpus row, 0 gold rows, and
            # that row's name is independently wrong.
            if stats is not None:
                stats["gate_stereo_unverifiable"] = (
                    stats.get("gate_stereo_unverifiable", 0) + 1)
    # NOTE (, resolved): radical names now ship normally. The validity gate's
    # primary probe runs the OpsinOracle WITH `-r` (retained_substitution.py), so
    # a well-formed radical name parses -> constitutional compare -> ships.
    # The non-terminal-locant fix landed long ago (single sites:
    # C[CH]C -> `propan-2-yl` via emit_parent_hydride_cumulative_suffix); multi-site
    # free valences: ethane-1,2-diyl / propane-1,2,3-triyl /
    # ethan-1-yl-2-ylidene) landed in w2f p5. The suppression below is NOT
    # radical-specific: it is the general fail-closed backstop for names OPSIN
    # rejects EVEN WITH `-r` (constitutional defects) — no validity-gate edit is
    # needed to emit radicals.
    # Definitively unparseable (constitutional defect) -> suppress to the honest fallback.
    if stats is not None:
        stats["opsin_suppressed"] = stats.get("opsin_suppressed", 0) + 1
    logger.warning("OPSIN validity gate suppressed unparseable name: %r", name[:60])
    from .metrics.abstention import AbstentionCode, record_suppression
    record_suppression(AbstentionCode.GATE_SUPPRESSED, detail='opsin_unparseable',
                       candidate=name)
    _suppressed_to = _descriptive_fallback(smiles)
    _record_gate_outcome(_pv.GATE_OUTCOME_SUPPRESSED, _suppressed_to)
    return _suppressed_to


def _full_inchikey_offer_match(input_smiles: str, opsin_smiles: str,
                               name: Optional[str] = None) -> bool:
    """ a phase L3-1: True iff ``input_smiles`` and ``opsin_smiles`` share
    the SAME FULL (all-layer) InChIKey -- constitution AND stereo AND
    protonation/mobile-H-tautomer state all identical.

    Strictly stronger than `_self_consistency_skeleton`'s stereo-/charge-
    insensitive FIRST block (which is what the ordinary gate proves):
    the L3 CHARACTERIZATION found the discard-gap primaries are constitution-
    CORRECT (they pass that tolerant compare -- including its deliberate
    ``na == 0`` licence, "a stereo-UNSPECIFIED input named by a stereo-
    implying retained name... tolerated here") but full-InChIKey-FAIL (33/34
    differ in the SECOND block: a fabricated/omitted/conflicting stereo
    assignment the constitutional gate lets through by design). An offer
    competing against a systematic floor needs the STRICTER bar, not the
    PIN-emission gate's tolerant one.

    Fail-OPEN (``True``) on any parse/hash failure -- an inconclusive compare
    must never reject an offer.
    """
    try:
        mi = Chem.MolFromSmiles(input_smiles)
        mo = Chem.MolFromSmiles(opsin_smiles)
        if mi is None or mo is None:
            return True
        from rdkit.Chem import inchi
        ik_in = inchi.MolToInchiKey(mi)
        ik_out = inchi.MolToInchiKey(mo)
        if not ik_in or not ik_out:
            return True
        if ik_in != ik_out:
            return False
        # An equal key does not place a hydron (validation/protonation_identity.py).
        from .validation.protonation_identity import protonation_site_verdict
        return protonation_site_verdict(input_smiles, opsin_smiles, name) != "mismatch"
    except Exception:
        return True


def _offer_rt_ok(name: str, input_smiles: Optional[str]) -> bool:
    """ a phase L4-core/L3-1: the RT/PIN-gate PASS/FAIL predicate for ONE
    candidate ``name`` offered as a whole-molecule winner (the ``rt_ok``
    ``offer_pool.select_rt_passing`` is injected with, built at `_finish`).

     a phase L3-1 CHANGE: this now requires a FULL-InChIKey match
    (`_full_inchikey_offer_match`), not merely the constitution-only
    verdict the L4-core version reused outright. The L3 CHARACTERIZATION
    measured that the constitution-only bar lets a stereo-wrong (or stereo-
    fabricated) primary through, which is exactly the gap the systematic
    floor exists to close -- so a primary that only proves constitution must
    NOT out-rank a floor offer that proves the full molecule.

    NO double-OPSIN in the common case: `_validity_gate_name_to_smiles` is
    MEMOIZED (`OpsinOracle._name_cache`), and the ordinary gate
    (`_final_opsin_validity_gate`) already called it on this EXACT string
    before recording whatever verdict `_self01_lookup` reads back -- so the
    call below is a cache HIT, not a new JVM invocation, for every name the
    gate already processed. Only a string with NO recorded verdict at all
    (e.g. a name `_apply_trivial_fallback` swapped in AFTER the gate ran, or
    a fresh floor/alternative offer the gate never saw) triggers a genuinely
    NEW OPSIN call here.

    Fail-OPEN throughout, mirroring L1's `method == "unavailable"` handling:
    jar-absent, a transient 'unavailable' OPSIN outcome, and an inconclusive
    compare all return True. This predicate must never be the reason
    `select_rt_passing` empties the whole offer pool.

     no-abstain Phase A fix-a performance pass (Findings 1+2, a review adversarial
    review): `skip_reanchor` bundles FOUR distinct outcomes as though they all
    meant "no check is possible or worth attempting" -- but `unavailable` /
    `not_run` mean only "no gate call was recorded for THIS EXACT STRING
    (yet)", which is a LIVE, checkable question right now whenever a jar is
    present -- unlike a genuine `carveout:*` (a by-design OPSIN-unparseable
    PIN class -- thioperoxol/inositol/...) or `gate_disabled` (the whole
    verification layer is deliberately off). Fail-opening all four alike let a
    stubbed producer ship a WRONG-MOLECULE name via a `t4_floor` offer with
    ZERO gates firing end-to-end: `a temp dir/probe_whitebox_floor.py` stubbed
    `t4_coverage.name_t4_complete` to return `"ethanol"` for an unrelated
    abstainer (`CC(=O)C1=C(C)S[C@@H](C)CC1=O`) and the genuine
    `_finish`/`_maybe_append_t4_floor_offer`/`_offer_rt_ok`/`select_rt_passing`
    chain shipped it, `source=t4_floor`, `gate_outcome=not_run`. Deny-by-
    default: a never-gated string (`unavailable`/`not_run`) must EARN a fresh
    positive `verify_or_none` verdict before it may win, whenever a jar is
    present; only a genuine carveout / gate-disabled / no-jar case keeps the
    advisory fail-open (that reasoning is legitimately unaffected -- none of
    those three ever had a check available at all).
    """
    self01_complete, skip_reanchor, _detail = _self01_lookup(name)
    if self01_complete is False:
        # self01_warn_mismatch -- a PROVEN constitutional mismatch shipped
        # anyway under _SC_MODE=="warn". A real fail for RT-gating purposes;
        # the full-InChIKey bar can only be at least as strict, never rescue it.
        return False
    if skip_reanchor:
        # Findings 1+2 fix: split the FOUR bundled outcomes by reading the
        # `_detail` string `_self01_lookup` already returns (it is always
        # formatted `f"{resolved}:..."`) rather than re-deriving the outcome
        # from provenance directly -- so a test (or caller) that mocks
        # `_self01_lookup` as a black box still drives this branch correctly,
        # and no second, redundant contextvar read is needed.
        from .metrics import provenance as _pv
        if (_detail.startswith(_pv.GATE_OUTCOME_CARVEOUT_PREFIX)
                or _detail.startswith(_pv.GATE_OUTCOME_DISABLED + ":")):
            # by-design OPSIN-unparseable PIN, or the whole gate deliberately
            # off -- no check was ever possible or intended. Advisory
            # fail-open stands, UNCHANGED from before this fix.
            return True
        # `unavailable` or `not_run`: this string has NEVER been checked.
        if not _validity_gate_jar_present():
            return True  # genuinely no way to verify right now -- fail-OPEN
        if not input_smiles:
            return True  # nothing to compare against -- inconclusive, fail-OPEN
        # A fresh, real verdict is possible -- require it. verify_or_none's
        # OPSIN branch fails closed (returns None) on a transient
        # 'unavailable' outcome too (it only succeeds on a genuine
        # `method == "parse_back"` match), so this single call also closes
        # Finding 2's transient-unavailable fail-open for this branch.
        from .validation.reconstruct import verify_or_none
        return verify_or_none(name, input_smiles) is not None
    if not _validity_gate_jar_present():
        return True  # fail-OPEN: no jar to consult
    # NOTE: deliberately NOT short-circuited on `self01_complete is True`
    # (L3-1's whole point) -- proceed to the full-InChIKey compare below,
    # reusing the memoized name_to_smiles lookup regardless of what the
    # constitution-only verdict said.
    opsin_smiles = _validity_gate_name_to_smiles(name)
    if opsin_smiles is None:
        # OPSIN could not re-perceive this string. Fail OPEN on a transient
        # 'unavailable' ONLY when this EXACT string ALREADY earned a real
        # positive verdict (`self01_complete is True`) -- re-looking up a
        # previously-VERIFIED name and hitting a transient blip must not lose it.
        #
        # fix a performance pass (coordinator CRITICAL, 0-wrong): for a string with NO
        # positive verdict (`self01_complete is None` -- `bypassed` / `suppressed`
        # / `inconclusive`, e.g. a FRESH floor/alternative offer whose exact
        # string the gate NEVER verified), a None here means the offer is
        # UNVERIFIED, so it must fail CLOSED. `bypassed` reaches THIS branch
        # (not the deny-by-default `verify_or_none` branch above, which
        # `_self01_lookup` only routes `unavailable`/`not_run` into), so without
        # this a transient OPSIN failure shipped an ungated floor offer
        # (`F[B-](F)(F)F` -> the unverified floor `2,2-difluoro-2-borapropan-2-uide`;
        # `[Se-]CC` -> the OPSIN-unparseable `1-selanidoethane`). This makes the
        # offers lane symmetric with the recovery-lane (`_cand_from_t4`), which
        # already fails CLOSED on the same blip. A definitive 'rejected' fails
        # closed for every string regardless.
        return (self01_complete is True
                and _validity_gate_status(name) == "unavailable")
    if not input_smiles:
        return True  # nothing to compare against -- inconclusive, fail-OPEN
    return _full_inchikey_offer_match(input_smiles, opsin_smiles, name)


def _shipped_name_round_trip(name: str, input_smiles: Optional[str]) -> str:
    """Claims conformance (2026-09-27): the round trip of a name that is about to
    ship at a general tier, with NO fail-open branch (unlike `_offer_rt_ok`).

    Returns ``"verified"`` when OPSIN reads ``name`` and the FULL standard
    InChIKey of what it read equals the input's (for a radical input the radical
    graph must agree too, as the key encodes no radical electrons); equal full
    keys mean the name denotes the whole input, so every atom is covered.
    ``"unavailable"`` when OPSIN could not be consulted after the oracle's retry
    ladder; ``"failed"`` otherwise (OPSIN rejected the name, a key differs, or a
    key could not be computed). The OPSIN call is memoized, so a name the gate or
    the offer pool already read costs no new parse.
    """
    if not name or not input_smiles:
        return "failed"
    opsin_smiles = _validity_gate_name_to_smiles(name)
    if opsin_smiles is None:
        return ("unavailable" if _validity_gate_status(name) == "unavailable"
                else "failed")
    in_key = _self_consistency_full_key(input_smiles)
    if not in_key or _self_consistency_full_key(opsin_smiles) != in_key:
        return "failed"
    try:
        from .validation.radical_identity import radical_identity_verdict
        if radical_identity_verdict(input_smiles, opsin_smiles) == "mismatch":
            return "failed"
        # Nor does the key place a hydron: the partial salts of one polybasic
        # acid, and the cations of one base protonated on different atoms, share
        # it (validation/protonation_identity.py).
        from .validation.protonation_identity import protonation_site_verdict
        if protonation_site_verdict(input_smiles, opsin_smiles, name) == "mismatch":
            return "failed"
    except Exception:
        return "failed"
    return "verified"


def _np_exact_list_name(smiles: Optional[str]) -> Optional[str]:
    """The exact-match natural-product parent name of exactly this input
    (``data.natural_products.NAME_EXACT_NP_PARENTS``, looked up by the input's
    canonical SMILES in ``NATURAL_PRODUCT_DERIVATIVES``), or None. Unlike
    ``get_natural_product_name`` it does not give way at the general tiers."""
    if not smiles:
        return None
    try:
        from .data.natural_products import (
            GENERAL_ONLY_NATURAL_PRODUCTS,
            NAME_EXACT_NP_PARENTS,
            NATURAL_PRODUCT_DERIVATIVES,
        )
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if canonical in GENERAL_ONLY_NATURAL_PRODUCTS:
            return None
        name = NATURAL_PRODUCT_DERIVATIVES.get(canonical)
        return name if name in NAME_EXACT_NP_PARENTS else None
    except Exception:
        return None


def _coordination_list_name(smiles: Optional[str], name: Optional[str]) -> bool:
    """True iff ``name`` is the exact-match coordination list name of exactly this
    input (D1 table, matched by the input's exact InChIKey)."""
    if not smiles or not name:
        return False
    try:
        return _coordination_retained_name(smiles) == name
    except Exception:
        return False


def _is_exact_match_list_name(smiles: Optional[str], name: Optional[str]) -> bool:
    """True iff ``name`` is one of the exact-match list names the general tiers
    may ship without an OPSIN round trip: a NAME_EXACT natural-product parent
    ('germacrane', matched on the whole structure) or the coordination retained
    name of exactly this input."""
    if not name:
        return False
    try:
        from .data.natural_products import NAME_EXACT_NP_PARENTS
        if name in NAME_EXACT_NP_PARENTS:
            return True
    except Exception:
        pass
    try:
        return bool(smiles) and _coordination_retained_name(smiles) == name
    except Exception:
        return False


def _default_tier_emits(row: dict, pin_form: bool, smiles: Optional[str],
                        trivial_fallback: bool = False) -> bool:
    """Whether the default tier emits the name of ``row`` (paper conformance, user
    decision 2026-09-30; the submitted text, Methods). True for:

    * a pin_verified name ("the strict PIN path built the name and verified it");
    * an exact-match list name, natural-product parent or metal complex ("The only
      deliberate exceptions are names derived from the natural-product and
      metal-complex lists... emitted based solely on their exact structural
      match");
    * a name of the formats OPSIN has no grammar for, shipped on their construction
      (gate outcome ``carveout:<class>``; "At the default tier, ten further name
      formats absent from OPSIN's grammar... are emitted without parse-back
      validation because their construction rules are exact");
    * a strict-path PIN whose stereodescriptors OPSIN cannot read, verified by the
      constitution and the descriptor check (``self_consistency_constitution_only``
      with ``pin_form``; "At the default tier, it compares only the constitutional
      part of the key");
    * with ``trivial_fallback`` (``--trivial``), a name from the table of retained
      trivial names, which that option asks for.

    Every other name -- one the engine records as not the PIN, a name only a
    breadth producer built, a general-engine or retained-table name, a name of a
    class without PIN status -- is declined (NO_VERIFIED_PIN)."""
    from .metrics import provenance as _pv
    name = row.get("name")
    if not name or is_failure_name(name) or row.get("tier") == ABSTAIN:
        return True
    if row.get("tier") == PIN_VERIFIED:
        return True
    if _is_exact_match_list_name(smiles, name):
        return True
    gate = row.get("gate_outcome") or ""
    if gate.startswith(_pv.GATE_OUTCOME_CARVEOUT_PREFIX):
        return True
    if gate == _pv.GATE_OUTCOME_SELF01_CONSTITUTION_ONLY and pin_form:
        return True
    # the opt-in reduced mode (no OPSIN jar): a strict-path name in PIN form ships
    # unverified, as the reduced mode documents; any other name is declined
    if gate == _pv.GATE_OUTCOME_UNAVAILABLE and pin_form:
        return True
    if trivial_fallback and row.get("source") == "trivial_retained":
        return True
    return False


def _default_tier_declined_row(smiles: str):
    """(label, row) of a default-tier decline (NO_VERIFIED_PIN): the label the plain
    call returns and the ``name_tiered`` row.

    The label is the NO_VERIFIED_PIN catalog text, 'unknown organic compound', never the
    UNSUPPORTED_ELEMENT text '<metal> compound (not supported)': the engine built a
    name for the structure (its elements are supported), only not a verified PIN
    ('ethenylsodium' is declined as 'unknown organic compound', NO_VERIFIED_PIN)."""
    from .errors import LIMIT_CATALOG
    from .metrics import provenance as _pv
    label, formula = LIMIT_CATALOG["NO_VERIFIED_PIN"]["message"], None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            from rdkit.Chem import rdMolDescriptors
            formula = rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        pass
    return label, {"name": label, "tier": ABSTAIN, "is_pin": False,
                   "source": "abstain", "opsin": "n/a", "gates_passed": [],
                   "gate_outcome": _pv.GATE_OUTCOME_SUPPRESSED,
                   "formula": formula, "limit_code": "NO_VERIFIED_PIN",
                   "stereo_unexpressed": False, "suffix_free_prefix_name": False,
                   "prefix_order_fallback": False, "verified": "unverified"}


# ---------------------------------------------------------------------------
# Compound class pre-routing (a phase, CLASS-06)
# ---------------------------------------------------------------------------

# Pre-compiled SMARTS for sugar ring detection
_PYRANOSE_SMARTS = None
_FURANOSE_SMARTS = None
_RING_OH_SMARTS = None


def _get_sugar_smarts():
    """Lazy-initialize SMARTS patterns for sugar ring detection."""
    global _PYRANOSE_SMARTS, _FURANOSE_SMARTS, _RING_OH_SMARTS
    if _PYRANOSE_SMARTS is None:
        # 6-membered ring with 1 O and 5 C (pyranose)
        _PYRANOSE_SMARTS = _compiled_smarts("[OX2;r6]1[CX4][CX4][CX4][CX4][CX4]1")
        # 5-membered ring with 1 O and 4 C (furanose)
        _FURANOSE_SMARTS = _compiled_smarts("[OX2;r5]1[CX4][CX4][CX4][CX4]1")
        # OH group on a ring carbon
        _RING_OH_SMARTS = _compiled_smarts("[C;r]([OX2H])")
    return _PYRANOSE_SMARTS, _FURANOSE_SMARTS, _RING_OH_SMARTS


def _has_sugar_ring_pattern(mol) -> bool:
    """Detect pyranose/furanose ring with minimum 2 OH groups.

    Uses SMARTS matching to identify sugar-like rings:
    - Pyranose: 6-membered ring with 1 O and 5 C
    - Furanose: 5-membered ring with 1 O and 4 C
    - Requires at least 2 hydroxyl groups on ring carbons
      (2 not 3, to handle deoxy sugars)

    Args:
        mol: RDKit Mol object.

    Returns:
        True if molecule has a sugar ring pattern.
    """
    if mol is None:
        return False

    pyranose, furanose, ring_oh = _get_sugar_smarts()

    has_sugar_ring = False
    if pyranose is not None and mol.HasSubstructMatch(pyranose):
        has_sugar_ring = True
    elif furanose is not None and mol.HasSubstructMatch(furanose):
        has_sugar_ring = True

    if not has_sugar_ring:
        return False

    # Count OH groups on ring carbons
    if ring_oh is not None:
        oh_matches = mol.GetSubstructMatches(ring_oh)
        if len(oh_matches) >= 2:
            return True

    return False


def classify_compound_class(mol, canonical_smiles: str) -> Optional[str]:
    """Classify molecule into compound class for pre-routing.

    Returns class label or None for general routing.
    Classification order per:
        steroid -> alkaloid -> terpene -> peptide -> amino_acid -> carbohydrate -> general

    Args:
        mol: RDKit Mol object.
        canonical_smiles: Canonical SMILES string.

    Returns:
        One of 'steroid', 'alkaloid', 'terpene', 'carbohydrate', or None.
    """
    if mol is None:
        return None

    # Sugar detection: check sugar lookup table first
    from .data.sugar_names import lookup_sugar
    if lookup_sugar(canonical_smiles) is not None:
        return "carbohydrate"

    # Pyranose/furanose SMARTS for sugars not in lookup
    if _has_sugar_ring_pattern(mol):
        return "carbohydrate"

    # NP detection handles steroid/alkaloid/terpene via scaffold matching
    from .perception.natural_products import detect_natural_product
    np_info = detect_natural_product(mol)
    if np_info is not None:
        return np_info.get("scaffold_class")  # "steroid", "alkaloid", "terpene"

    # Check exact derivative lookup -- some derivatives (e.g. alpha-pinene)
    # don't match a scaffold via substructure but ARE in the derivatives dict.
    # Infer class from the derivative name patterns.
    from .data.natural_products import get_natural_product_name
    deriv_name = get_natural_product_name(canonical_smiles)
    if deriv_name is not None:
        return _infer_class_from_derivative_name(deriv_name)

    return None  # General routing


# Terpene-related name patterns for class inference from derivative names
_TERPENE_KEYWORDS = frozenset([
    "pinene", "pinane", "bornane", "camphor", "limonene",
    "terpineol", "terpinene", "carotene", "lycopene", "menthane",
    "thujane", "pinanol", "borneol", "fenchone", "prostane",
])

# Steroid-related name patterns
_STEROID_KEYWORDS = frozenset([
    "cholesterol", "testosterone", "progesterone", "estradiol",
    "androstane", "pregnane", "cholestane", "estrane", "gonane",
    "campestanol", "ergostane", "stigmastane",
    "androstenedione", "androstanediol", "androstenediol",
    "androstenol", "androstenone", "estratetraenol",
    "androstadienone", "cardenolide", "cardanolide",
    "bufanolide", "bufadienolide",
])

# Alkaloid-related name patterns
_ALKALOID_KEYWORDS = frozenset([
    "morphine", "codeine", "diamorphine", "hydromorphone",
    "hydrocodone", "oxycodone", "lysergic", "lysergamide",
    "lysergol", "tropane", "morphinan", "aporphine",
    "dihydromorphine", "dihydrocodeine", "codeinone",
    "morphinone",
])

# Flavonoid / other
_FLAVONOID_KEYWORDS = frozenset([
    "flavone", "flavanone", "isoflavone", "chromanone", "chromone",
])


def _infer_class_from_derivative_name(name: str) -> Optional[str]:
    """Infer compound class from a derivative's trivial name.

    Args:
        name: Trivial/retained name of the derivative.

    Returns:
        Class label or None if class cannot be inferred.
    """
    name_lower = name.lower().replace("-", "")
    # Check terpene keywords
    for kw in _TERPENE_KEYWORDS:
        if kw in name_lower:
            return "terpene"
    # Check steroid keywords
    for kw in _STEROID_KEYWORDS:
        if kw in name_lower:
            return "steroid"
    # Check alkaloid keywords
    for kw in _ALKALOID_KEYWORDS:
        if kw in name_lower:
            return "alkaloid"
    # Beta-lactam and flavonoid are not routed to specific class handlers
    return None


@dataclass
class MolecularFeatures:
    """Container for perceived molecular features."""

    mol: Any  # RDKit Mol object
    smiles: str = ""
    canonical_smiles: str = ""

    # Functional group information
    functional_groups: Dict[str, List[tuple]] = field(default_factory=dict)
    principal_group: Optional[str] = None
    principal_group_atoms: List[tuple] = field(default_factory=list)

    # Chain information
    principal_chain: List[int] = field(default_factory=list)
    atom_to_locant: Dict[int, int] = field(default_factory=dict)
    substituents: Dict[int, List[List[int]]] = field(default_factory=dict)

    # Ring information
    ring_systems: List[set] = field(default_factory=list)
    all_ring_atoms: frozenset = field(default_factory=frozenset)  # All atoms in all ring systems (fused/bridged/spiro merged)
    is_cyclic: bool = False
    is_aromatic: bool = False
    ring_type: Optional[str] = None  # 'cycloalkane', 'cycloalkene', 'aromatic', 'heterocyclic_aromatic', 'heterocyclic_saturated'
    principal_ring: Optional[tuple] = None  # Atom indices of the principal ring
    senior_ring_system: Optional[tuple] = None  # most senior ring system (atom indices)
    oriented_ring: Optional[List[int]] = None  # Ring atoms reordered for naming
    ring_substituents: Dict[int, List[List[int]]] = field(default_factory=dict)  # Substituents on ring
    ring_double_bonds: List[tuple] = field(default_factory=list)  # Double bonds in ring
    ring_double_bond_locants: List[int] = field(default_factory=list)  # Locants for ring double bonds

    # Ring assembly information (biphenyl, bipyridine, etc.)
    ring_assembly_info: Optional[Dict] = None

    # Benzene-specific information
    is_benzene: bool = False  # True if principal ring is benzene
    benzene_ring: Optional[tuple] = None  # Atom indices of the benzene ring
    benzene_substituents: Dict[int, List[Dict]] = field(default_factory=dict)  # From get_benzene_substituents

    # Polycyclic aromatic information
    polycyclic_name: Optional[str] = None  # Name of PAH parent (naphthalene, etc.)
    polycyclic_substituents: Dict[int, List[Dict]] = field(default_factory=dict)  # From get_polycyclic_substituents

    # Heterocycle-specific information
    heterocycle_info: Optional[Dict] = None  # From classify_heterocycle
    oriented_heterocycle: Optional[List[int]] = None  # Ring atoms in IUPAC numbering order
    heterocycle_atom_to_locant: Optional[Dict[int, int]] = None  # Atom idx -> locant mapping
    heterocycle_substituents: Dict[int, List[Dict]] = field(default_factory=dict)  # From get_heterocycle_substituents

    # Stereochemistry
    stereocenters: List[dict] = field(default_factory=list)
    double_bond_stereo: List[dict] = field(default_factory=list)

    # Multiple bonds
    double_bonds: List[tuple] = field(default_factory=list)
    triple_bonds: List[tuple] = field(default_factory=list)

    # Polyfunctional compound information
    is_polyfunctional: bool = False  # True if molecule has 2+ distinct functional groups
    non_principal_groups: Dict[str, List[tuple]] = field(default_factory=dict)  # FGs other than principal

    # Ester-specific information
    ester_match: Optional[tuple] = None  # SMARTS match for principal ester group

    # Amide-specific information
    amide_type: Optional[str] = None  # "primary", "secondary", or "tertiary"
    n_substituents: List[Dict] = field(default_factory=list)  # N-substituents from get_n_substituents

    # Ring-as-substituent information (when chain is parent per IUPAC
    ring_substituents_as_groups: List[tuple] = field(default_factory=list)  # Rings that become substituents
    chain_is_parent: bool = False  # True when parent selection chose chain over ring

    # Ion/radical species information
    species_type: str = 'neutral'  # 'neutral', 'ion', 'zwitterion', 'salt', 'radical'
    ion_sites: Dict[str, List[Dict]] = field(default_factory=dict)  # From get_ion_sites
    radical_sites: List[Dict] = field(default_factory=list)  # From get_radical_sites
    total_charge: int = 0  # Net formal charge of the molecule

    # a phase / V18 Appendix A.5: parent-selection result for downstream
    # coverage_scoring readers. Populated by _classify when select_parent
    # runs for a cyclic+chain molecule (chain_len >= 2). a phase /
    # will recalibrate FACTOR_WEIGHTS_V18['parent_correctness'] against the
    # discrimination this slot provides.
    # Source: V18_MILESTONE_PLAN Appendix A.5.
    parent_selection_result: Optional[Any] = None


def compute_features(mol, smiles: Optional[str] = None) -> MolecularFeatures:
    """a phase: thin module-level wrapper around Orthonym._perceive.

    Provides a public-API perception entry for tests and downstream
    callers. Mirrors what name_compound does internally before naming.

    Args:
        mol: RDKit Mol object.
        smiles: Optional input SMILES; if None, derived via Chem.MolToSmiles(mol).

    Returns:
        MolecularFeatures populated by Orthonym._perceive.

    Source: a phase Plan 02 fix (public-API perception entry).
    """
    if smiles is None:
        smiles = Chem.MolToSmiles(mol)
    canonical_smiles = Chem.CanonSmiles(smiles)
    return Orthonym()._perceive(mol, smiles, canonical_smiles)


def _collect_ring_substituent_positions(features, ring_atoms):
    """Set of ring atom indices that bear an off-ring (non-H) substituent.

    Used by a phase dispatch helper branch 4 (simple heterocycle) to feed
    ``orient_heterocycle_with_substituents``.
    """
    ring_set = set(ring_atoms)
    positions = set()
    for atom_idx in ring_atoms:
        atom = features.mol.GetAtomWithIdx(atom_idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() not in ring_set and nbr.GetSymbol() != 'H':
                positions.add(atom_idx)
                break
    return positions


def _demote_offring_principal_group_matches(features, ring_atoms) -> None:
    """ (BB 25207): the principal group appears in BOTH the ring and
    a chain; the portion with the GREATER number of the group is the parent
    (tie -> ring,. When the ring wins, an off-ring principal-group
    match must NOT be expressed as a second ring suffix — it stays on its
    demoted chain and is named as a substituent prefix (e.g. the pendant
    butan-2-one on a cyclopentane-1,2-dione -> '4-(2-oxobutyl)').

    This filters ``features.principal_group_atoms`` in place to the ring-anchored
    matches only, WHEN:
      * the ring is the chosen parent (this helper is called only then);
      * the principal group is a NON-terminal, skeletal-carbon carbonyl-type
        group (ketone family) whose exocyclic form can be expressed as an
        oxo-substituent prefix — terminal/appended groups (-carbaldehyde,
        -carboxylic acid,...) legitimately anchor exocyclically and are
        untouched (they route through the ring_anchored_pg_atoms path);
      * the ring's on-ring match count is STRICTLY GREATER than the off-ring
        count (ring wins outright) OR they TIE ring default) — but
        only demote when there is at least one on-ring match to keep as the
        ring suffix (else the ring cannot express the group and we must not
        strip it; fail closed by leaving matches unchanged).

    The demoted matches remain in ``features.functional_groups`` so the pendant
    chain is picked up by the ring-substituent enumerator and named honestly.
    Fail-closed: any ambiguity leaves ``principal_group_atoms`` unchanged.
    """
    pg = features.principal_group
    matches = features.principal_group_atoms
    if not pg or not matches or len(matches) < 2:
        return
    # Only the ketone family (non-terminal, exocyclic-expressible as 'oxo').
    # These are exactly the PGs whose off-ring form has an oxo-substituent
    # prefix; expanding beyond this set risks stripping a group the ring
    # cannot re-express, so gate tightly (accuracy-first).
    _KETONE_FAMILY = {"ketone", "thioketone", "selenoketone", "telluroketone"}
    if pg not in _KETONE_FAMILY:
        return
    ring_set = set(ring_atoms)
    on_ring, off_ring = [], []
    for match in matches:
        # A match is 'on-ring' when its carbonyl carbon is a ring atom.
        # For a ketone SMARTS the carbonyl C is the skeletal carbon; require a
        # ring CARBON in the match bonded to the FG heteroatom in the match.
        match_set = set(match)
        is_on_ring = False
        for atom_idx in match:
            if atom_idx not in ring_set:
                continue
            atom = features.mol.GetAtomWithIdx(atom_idx)
            if atom.GetSymbol() != 'C':
                continue
            if any(
                nbr.GetIdx() in match_set
                and nbr.GetIdx() not in ring_set
                and nbr.GetSymbol() != 'C'
                for nbr in atom.GetNeighbors()
            ):
                is_on_ring = True
                break
        (on_ring if is_on_ring else off_ring).append(match)
    #: ring is parent only when it has the GREATER count, or ties
    # (ring default). Require >=1 on-ring match to keep as the suffix and
    # >=1 off-ring match to demote; otherwise nothing to do / fail closed.
    if not on_ring or not off_ring:
        return
    if len(on_ring) < len(off_ring):
        # Chain has more of the group -> ring should NOT be parent for this
        # group. Leave unchanged and let the normal path decide (fail closed
        # rather than force a wrong ring-parent split).
        return
    features.principal_group_atoms = on_ring


def _build_ring_info_for_parent_selection(features):
    """a phase: dispatch on ring type and produce authoritative IUPAC locants.

    7-branch cascade (order per 147-internal notes with W-1 fix):
      1. Fused-heterocycle (match_fused_heterocycle_core) — preserve
         byte-identical path; runs first.
      2. Polycyclic aromatic (identify_polycyclic + get_polycyclic_iupac_locants)
         — runs for ANY ring system with a PAH match, NOT gated on fused_type
         (W-1 fix: pyrene is ortho-peri-fused but must still flow here).
      3. Benzene-only single ring (orient_benzene with canonical
         get_benzene_substituents helper — fix).
      4. Simple heterocycle single ring (orient_heterocycle_with_substituents).
      5. Spiro detection -> {"iupac_locants": None} stub (a phase fills).
      6. Bridged/VB detection -> {"iupac_locants": None} stub (a phase fills).
      7. Else -> None (carbocyclic monocycle / hydrocarbon polycycle:
         sorted fallback in _build_ring_pos preserves back-compat per).

    Returns None for acyclic molecules.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html
    Source: a phase internal notes,,,; Plan 02 W-1, W-2,.
    """
    if not features.is_cyclic:
        return None

    # Match namer.py top-level relative-import style (W-2 fix).
    from .data.fused_heterocycles import match_fused_heterocycle_core
    from .rules.benzene import (
        get_benzene_substituents,
        molecule_principal_group_and_fgs,
        orient_benzene,
        principal_group_ring_atoms,
    )
    from .rules.bridged_fused import is_bridged_fused
    from .rules.fused_rings import classify_fused_system
    from .rules.heterocycles import orient_heterocycle_with_substituents
    from .rules.polycyclics import (
        get_polycyclic_iupac_locants,
        identify_polycyclic,
    )

    mol = features.mol

    # a phase-02: skip Branch 1 (fused-heterocycle catalog) when
    # the input is mixed-spiro/fused — Branch 5b owns that dispatch.
    # Without this guard, the catalog returns a PARTIAL locant map
    # (only the fused component's atoms; missing the spiro side ring),
    # which violates the cascade-step-6 coverage invariant downstream.
    from .rules.spiro import is_mixed_spiro_fused as _phase151_is_mixed_spiro_fused
    _is_mixed_spiro_fused_input = _phase151_is_mixed_spiro_fused(mol)

    # Branch 1: fused-heterocycle (preserve byte-identical path for
    # ACTUAL heterocycles — indole/quinoline/etc.). Skip when the matched
    # core has no heteroatoms (e.g., pyrene also lives in
    # FUSED_HETEROCYCLES registry incidentally; PAHs must flow to branch 2
    # so their tuple-locant numbering is used per W-1 fix).
    fused_type = classify_fused_system(mol)
    if not _is_mixed_spiro_fused_input and fused_type in ('ortho-fused', 'ortho-peri-fused'):
        het_match = match_fused_heterocycle_core(mol)
        if het_match is not None:
            _, atom_mapping, _ = het_match
            core_atom_indices = list(atom_mapping.keys())
            has_heteroatom_in_core = any(
                mol.GetAtomWithIdx(idx).GetSymbol() not in ('C', 'H')
                for idx in core_atom_indices
            )
            if has_heteroatom_in_core:
                return {"iupac_locants": atom_mapping}

    # Branch 2: PAH (W-1 fix — runs for ANY ring system with a PAH match,
    # not gated on the fused-only block above). Pyrene/anthracene/phenanthrene
    # are ortho-peri-fused; naphthalene is ortho-fused. All four flow through
    # here when not a fused-heterocycle.
    pah_name = identify_polycyclic(mol)
    if pah_name is not None:
        pah_locants = get_polycyclic_iupac_locants(mol, pah_name)
        if pah_locants is not None:
            return {"iupac_locants": pah_locants}

    ring_systems = features.ring_systems
    if len(ring_systems) == 1:
        ring_atoms = tuple(ring_systems[0])
        ring_mol_atoms = [mol.GetAtomWithIdx(i) for i in ring_atoms]

        # Branch 3: benzene-only (single 6-aromatic-C ring).
        # fix: use the canonical get_benzene_substituents helper
        # (returns Dict[int, List[Dict]] with real substituent entries),
        # NOT a {idx: } placeholder — the latter would yield arbitrary
        # orientation because orient_benzene checks ``if atom_idx in
        # substituents:`` and every dict key matches an empty-list value.
        if (len(ring_atoms) == 6
                and all(a.GetIsAromatic() and a.GetSymbol() == 'C'
                        for a in ring_mol_atoms)):
            try:
                substituents = get_benzene_substituents(mol, ring_atoms)
                # E1/DD4 (c)): anchor the principal characteristic group
                # to the lowest locant before detachable substituents.
                # Phase C Task 9: this was derived from the is_suffix marker
                # alone, which made it EMPTY for phenols (a ring -OH is spelled as
                # the 'hydroxy' PREFIX here and promoted to '-ol' downstream) --
                # so this locant HINT disagreed with the emitted NAME. It now uses
                # the same shared authority as the composer, so the two agree.
                #
                # ⚠ Phase C Task 9b: reading ``features.principal_group`` alone
                # made the whole anchor a NO-OP here. ``compute_features`` runs
                # ``_perceive`` only, so that attribute is None for all four external
                # callers (rules/ring_chalcogen_oxide.py:277,
                # rules/multiplicative.py:1795 and:2454, rules/ions.py:2240) --
                # i.e. every caller but namer.py:3900. Fall back to the shared
                # molecule-level authority so the hint is computed, not skipped.
                _pg = getattr(features, 'principal_group', None)
                _fgs = getattr(features, 'functional_groups', None)
                if _pg is None:
                    _pg, _fallback_fgs = molecule_principal_group_and_fgs(mol)
                    if not _fgs:
                        _fgs = _fallback_fgs
                pcg_positions = principal_group_ring_atoms(
                    mol, ring_atoms, substituents,
                    principal_group=_pg,
                    detected_fgs=_fgs,
                )
                oriented = orient_benzene(
                    mol, ring_atoms, substituents,
                    principal_group_positions=pcg_positions or None,
                )
                atom_to_locant = {
                    atom_idx: i + 1
                    for i, atom_idx in enumerate(oriented)
                }
                return {"iupac_locants": atom_to_locant}
            except Exception:
                pass  # Defensive: fall through on handler edge case.

        # Branch 4: simple heterocycle (single ring with >=1 heteroatom).
        has_heteroatom = any(
            a.GetSymbol() not in ('C', 'H') for a in ring_mol_atoms
        )
        if has_heteroatom:
            substituent_positions = _collect_ring_substituent_positions(
                features, ring_atoms,
            )
            try:
                _, atom_to_locant = orient_heterocycle_with_substituents(
                    mol, ring_atoms, substituent_positions,
                )
                return {"iupac_locants": atom_to_locant}
            except Exception:
                pass

    # Branch 5 (a phase-02): pure spiro + mixed spiro/fused
    # cascade-step-6 suppliers.
    #
    # Pure spiro: is_spiro_system True iff n_rings == n_spiro + 1 (lock).
    # Wraps existing get_spiro_numbering / _get_polyspiro_numbering with
    # the cascade-step-6 coverage invariant (Pitfall 7).
    #
    # Mixed spiro/fused: is_mixed_spiro_fused True iff ≥1 spiro atom AND
    # n_rings > n_spiro + 1 AND detect_natural_product is None (Pitfall 3
    # false-positive guard) AND the topology is AUTONOM separable AND
    # the fused part has a catalog name ((b) canary-stability guard).
    #
    # If neither supplier returns full coverage (None), fall through to
    # Branch 6 (a phase-01 VB) and onwards.
    #
    # Source: 151-internal notes / /; internal notes-B.md verdict
    # MIXED_SPIRO_FUSED_MISSING + Q-05 NESTED_FORM_PARSEABLE.
    from .rules.spiro import (
        get_mixed_spiro_fused_iupac_locants,
        get_spiro_iupac_locants,
        is_mixed_spiro_fused,
        is_spiro_system,
    )
    if is_spiro_system(mol):
        sl = get_spiro_iupac_locants(mol)
        if sl is not None:
            return {"iupac_locants": sl}
        # Else: spiro system but supplier declined coverage — fall through
        # to the existing get_spiro_atoms-True stub return so other
        # downstream branches don't attempt to take over.
        return {"iupac_locants": None}
    if is_mixed_spiro_fused(mol):
        msfl = get_mixed_spiro_fused_iupac_locants(mol)
        if msfl is not None:
            return {"iupac_locants": msfl}
        # Mixed-spiro/fused but supplier declined — emit None so the
        # cascade-step-6 gate falls through to the sorted-int proxy
        # rather than mis-routing to Branch 6 (VB).
        return {"iupac_locants": None}

    # Branch 6 (a phase-01): Von Baeyer ≥4-ring authoritative locants.
    # Routes tetracyclic / pentacyclic / higher non-cataloged bridged systems
    # to the new polycyclic_von_baeyer module. Anti-canary lock inside
    # is_higher_polycyclo guarantees bicyclo / tricyclo / aromatic / steroid
    # / mixed cases skip this branch and fall through to is_bridged_fused
    # below or downstream branches.
    #
    # Source: 151-internal notes / /; internal notes-A.md verdict
    # THIN_WRAPPER; a phase cascade-step-6 gate (candidate_pool.py:634).
    from .rules.polycyclic_von_baeyer import (
        get_higher_polycyclo_iupac_locants,
        is_higher_polycyclo,
    )
    if is_higher_polycyclo(mol):
        vbl = get_higher_polycyclo_iupac_locants(mol)
        if vbl is not None:
            return {"iupac_locants": vbl}

    # Branch 6 (existing): residual bridged / Von Baeyer stub for cases the
    # new module did NOT handle (e.g., bicyclic / tricyclic / aromatic
    # bridged systems caught by is_bridged_fused). a phase-02 / 03 wire
    # spiro and ring-assembly suppliers here.
    if is_bridged_fused(mol):
        return {"iupac_locants": None}

    # Branch 7 (a phase-03): ring assembly size 3+ supplier.
    # Wires get_ring_assembly_iupac_locants per internal notes-C.md verdict
    # SUPPLIER_MISSING + internal notes Pattern S-3. Path-topology check
    # added to detect_ring_assembly per rejects branched arrangements
    # (1,3,5-triphenylbenzene) so Branch 7 only fires on linear chains.
    # 2-system bi- assemblies are NOT routed here (covered by existing
    # naming pipeline + the >=3 gate keeps blast-radius minimal).
    #
    # Source: 151-internal notes / / /; internal notes-C.md
    # composite verdict; Pitfall 7 (full coverage or None).
    if hasattr(features, 'ring_systems') and len(features.ring_systems) >= 3:
        from .rules.ring_assemblies import (
            detect_ring_assembly as _phase151_detect_ring_assembly,
        )
        from .rules.ring_assemblies import (
            get_ring_assembly_iupac_locants as _phase151_get_ra_locants,
        )
        info = _phase151_detect_ring_assembly(mol, features.ring_systems)
        if info is not None and info.get("count", 0) >= 3:
            ral = _phase151_get_ra_locants(mol)
            if ral is not None:
                return {"iupac_locants": ral}

    # Branch 6.5 (a phase): non-cataloged fused systems.
    # Cataloged compounds reach Branches 1 (fused-heterocycle catalog) and
    # 2 (PAH) first. If they didn't, but the system is ortho-fused or
    # ortho-peri-fused with EXACTLY 2 SSSR components, route base-component
    # selection through.3.
    #
    # CRITICAL: emits `base_component_atoms` (NEW key), NOT `iupac_locants`.
    # Cascade step 6 in candidate_pool._has_iupac_locants checks
    # specifically for `iupac_locants` — Branch 6.5's new key is invisible
    # to that gate, so cascade step 6 stays GATED for non-cataloged fused
    # systems (a phase + a phase lock).
    #
    # SCOPE LIMIT (a phase Plan 02 triage): restricted to 2-component
    # fused systems where.3 base selection is reliable. 3+ component
    # systems (steroids, complex polycycles) fall through to Branch 7 to
    # avoid propagating partial base-atoms that disrupt downstream parent
    # selection for systems whose IUPAC name requires the full ring system
    # as parent. 3+ component systematic-name assembly is deferred to
    # a phase.x or a phase per trade-off.
    #
    # Source: https://iupac.qmul.ac.uk/fusedring/FR23.html
    # Source: 149-internal notes;;; trade-off.
    # Source: 147-internal notes (cascade step 6 gate carry-forward).
    if fused_type in ('ortho-fused', 'ortho-peri-fused'):
        # Branch 1 (catalog) and Branch 2 (PAH) already missed (control
        # flow reached this point — Branch 1 returns earlier on
        # heterocycle match; Branch 2 returns earlier on pah_locants).
        from .rules.fused_ring_selection import (
            _enumerate_components,
            select_base_component,
        )
        components = _enumerate_components(mol)
        # Scope guard: 2-component fused systems only (Plan 02 triage).
        if len(components) == 2:
            try:
                base_atoms, _ = select_base_component(mol, components)
                return {"base_component_atoms": frozenset(base_atoms)}
            except (ValueError, Exception):
                pass  # Fall through to Branch 7

    # Branch 7: else -> sorted fallback in _build_ring_pos.
    return None


# Confidence threshold below which the quality gate rejects a name as
# truncated/incomplete and falls back to decomposition naming.
# Calibrated against 500-compound ChEBI benchmark: 0.30 catches
# catastrophically incomplete names without false positives on correct names.
_TRUNCATION_CONFIDENCE_THRESHOLD = 0.30

#:: the accepted values of the binding-proof flag.
BINDING_PROOF_MODES = ("off", "audit", "enforce")


def validate_binding_proof(value: str) -> str:
    """Single source of truth for the binding-proof mode check.

    Shared by ``Orthonym.__init__`` and every surface that accepts the flag
    (notably the CLI's ``--batch`` path, which builds its namers lazily inside
    a per-row ``except Exception`` and would otherwise degrade a typo into a
    row-level "ERROR:" line). Validate eagerly and identically everywhere: a
    misspelled mode that silently degraded to "off" would present as a clean
    run with the proof never executing -- the one failure mode an audit flag
    must not have.
    """
    if value not in BINDING_PROOF_MODES:
        raise ValueError(
            f"binding_proof must be 'off', 'audit' or 'enforce', "
            f"got {value!r}")
    return value


import functools as _functools
from .perception.molcache import atoms_of, bonds_of, inchikey_of
from .perception.smarts_cache import compiled as _compiled_smarts

#: The SMILES string given to the outermost ``name`` of this thread (set and cleared
#: by ``_budget_scope`` at depth 1), None outside any ``name``.
_CALLER_INPUT = threading.local()


def _caller_input() -> Optional[str]:
    return getattr(_CALLER_INPUT, 'smiles', None)


def _budget_scope(fn):
    """ giant-molecule hang fix: bracket a top-level ``name`` with the
    per-top-level fragment work budget.

    ``enter_name_scope``/``exit_name_scope`` maintain a raw ``name`` call-stack
    counter on the fragment thread-local; the TRUE outermost call arms the budget
    and every nested re-entry (the recursion through ``name_compound`` and the
    isolated producer) shares it. Crucially, ``isolated_naming_session`` resets
    the session state but NOT this counter, so the budget survives the nested
    recovery entries a giant molecule triggers -- guaranteeing termination.

    A decorator (not inline enter/exit) so the budget is released on EVERY exit
    path -- ``name`` has several early ``return`` sites that each call
    ``end_naming_session`` -- with no leak that could contaminate the next
    molecule.
    """
    @_functools.wraps(fn)
    def _wrapper(self, *args, **kwargs):
        from .assembly.fragment_naming import (
            PerfBudgetExceeded,
            disarm_hang_budgets,
            enter_name_scope,
            exit_name_scope,
        )
        from .metrics.provenance import best_effort_request_ctx
        depth = enter_name_scope()
        _lp_opened = False
        _policy = False
        _request_token = None
        try:
            if depth == 1:
                # The tier of the request (``provenance.best_effort_request_ctx``)
                # for the whole outermost call, the last-resort rescue below
                # included; reset in the finally below.
                _request_token = best_effort_request_ctx.set(
                    bool(getattr(self, '_general_fallback_unverified', False)))
                # Lone-pair stereocentres: the string the caller gave, which the exit
                # check reads as written (_exit_lone_pair_check); nested name calls
                # name strings RDKit wrote. A lone-pair centre RDKit reads unlike the
                # standard reading is named as the standard spelling of the caller's
                # string (_lone_pair_standard_spelling), and the exit check still
                # reads the caller's string as written (_lone_pair_as_written).
                _given = args[0] if args else kwargs.get('smiles')
                _spelled, _lp_opened = _lone_pair_input_enter(_given)
                if _spelled != _given:
                    if args:
                        args = (_spelled,) + tuple(args[1:])
                    else:
                        kwargs['smiles'] = _spelled
                _CALLER_INPUT.smiles = _spelled
                # The default tier's emission rule applies to the caller's call
                # only; the provenance it reads is this call's own.
                _policy = self._default_tier_policy_applies()
                if _policy:
                    self._default_tier_row = None
                    from .metrics.provenance import clear_provenance
                    clear_provenance()
            _result = fn(self, *args, **kwargs)
            if _policy:
                _result = self._default_tier_decision(
                    _result, args[0] if args else kwargs.get('smiles'),
                    kwargs.get('raise_on_limit', False))
            return _result
        except PerfBudgetExceeded:
            # M2.5: the per-top-level OPERATION budget was exhausted deep inside
            # a combinatorial ring analysis (a genuinely explosive symmetric
            # metallo-macrocycle / large cyclic peptide). PerfBudgetExceeded is
            # a BaseException, so it unwound PAST every broad ``except Exception``
            # on the recursive path to here. Convert it to the SAME clean abstain
            # the engine emits for any unnameable input -- ONLY at the true
            # outermost name (depth == 1); a nested scope re-raises so the
            # signal keeps unwinding. inv 9: this is a clean abstain (the
            # descriptive/coordination fallback), never a partial or wrong name.
            if depth != 1:
                raise
            # Disarm both budgets FIRST: the descriptive/coordination fallback
            # below re-enters the fused matcher + von-Baeyer via _classify, and
            # with a still-exhausted budget that would re-raise and escape this
            # boundary (the exit_name_scope in finally runs only after).
            disarm_hang_budgets()
            smiles = args[0] if args else kwargs.get('smiles')
            # M0: before the clean abstain, offer ONE bounded, strictly
            # OPSIN-RT-gated whole-molecule name. The budget fired because the
            # MAIN path's combinatorial ring analysis exploded, but the
            # coverage-by-construction producer often names the same molecule
            # cheaply AND exactly (measured: a 100-atom 10-ring fused peptide is
            # named at 497 chars, full-InChIKey RT). `_try_perf_budget_t4_rescue`
            # re-arms a FRESH budget (so a genuine re-explosion re-raises and
            # abstains, never hangs) and ships only an OPSIN-round-tripping name
            # (so an unparseable coordination name is rejected -- 0-wrong).
            # Best-effort only + only on this already-abstaining boundary, so
            # PIN/complete output is byte-identical (offer-not-return, inv 18).
            _rescued = self._try_perf_budget_t4_rescue(smiles)
            if _rescued is not None and not is_failure_name(_rescued):
                _result = self._finish(_rescued, smiles)
                if _policy:
                    _result = self._default_tier_decision(_result, smiles, False)
                return _result
            logger.warning(
                "PERF BUDGET exhausted: abstaining on smiles=%s (macrocycle-hang guard)",
                (smiles or '')[:60])
            try:
                from .metrics.abstention import AbstentionCode, record_abstention
                record_abstention(AbstentionCode.BRANCH_UNNAMEABLE,
                                  detail='perf_op_budget')
            except Exception:
                pass
            # Route the abstain through name's single audited exit (_finish),
            # but SUPPRESS the floor-offer: the budget fired precisely because
            # this molecule's naming EXPLODES, so re-running the full producer
            # here (a) is now unbudgeted and could re-hang, and (b) for these
            # metallo-macrocycles ships an OPSIN-UNPARSEABLE coordination name
            # its RT gate wrongly accepts (a pre-existing spelling-layer gap this
            # hang guard must not unmask -- inv 9). The clean sentinel / curated
            # coordination-retained name is the correct abstain.
            self._suppress_floor_offer = True
            try:
                _result = self._finish(_descriptive_fallback(smiles), smiles)
            finally:
                self._suppress_floor_offer = False
            if _policy:
                _result = self._default_tier_decision(_result, smiles, False)
            return _result
        finally:
            exit_name_scope()
            if depth == 1:
                _CALLER_INPUT.smiles = None
            if _request_token is not None:
                best_effort_request_ctx.reset(_request_token)
            _lone_pair_input_exit(_lp_opened)
    return _wrapper


class _FailureRescues:
    """The optional rescues the outermost ``name`` runs once its main path has
    produced a failure it is about to ship: the late general-engine recovery, the
    demote-senior-group rescue, the decomposition retry, the alternate-parent
    rescue, the natural-product downgrade and the clean general fall-through.

    They charge the per-molecule hang budgets the outermost ``name`` armed for
    the whole molecule (``fragment_naming._PERF_BUDGET`` / ``_ANALYSIS_CALL_BUDGET``,
    memo hits replaying their recorded charge). A rescue that exhausted one used to
    raise ``PerfBudgetExceeded`` all the way to the ``_budget_scope`` boundary,
    which abstains with the floor offer suppressed -- so a molecule whose main path
    had FINISHED lost the round-trip-gated floor name that ``_finish`` offers every
    other failure. Measured on the row
    'COC(=O)c1ccc2[nH]c(=N[C@H]3C[C@@H](NC(=O)[C@H]4CCOC4)C3)sc2c1' at the best-
    effort tier: its naming charges 410 analysis calls at 171c54d5e and 518-520
    from 818c36360 on, where ``rules.amides.name_amide`` declines the oxolane ring
    it used to spell as cyclopentane (a 0-wrong decline; with only that decline
    reverted in a scratch copy the count is 412) and the rescues do more work; the
    ceiling is 500, the trip falls inside the clean general fall-through, and the
    molecule abstained where the floor names it round-trip exact.

    ``run`` calls one rescue; when the rescue exhausts a budget it declines
    (``fallback``) and every later rescue is skipped. ``name`` then offers the
    bounded rescue the boundary would have offered, re-arms a fresh budget and
    leaves through its main exit as for any other failure (floor offer included).
    Only the frame that owns the budgets (``name_scope_depth == 1``) does this;
    a nested ``name`` re-raises, so the signal still unwinds to that frame. A
    trip in the main path (the hang class, e.g. the Ni-corrin of
    ``tests/unit/rules/test_m25_workbudget.py``) reaches the boundary exactly as
    before. Work stays bounded: one budget for the main path and its rescues, one
    for the rescue, one for the exit; a re-explosion at the exit raises to the
    boundary.
    """

    __slots__ = ("_owns_budget", "exhausted")

    def __init__(self):
        from .assembly.fragment_naming import name_scope_depth
        self._owns_budget = name_scope_depth() == 1
        self.exhausted = False

    def run(self, rescue, *args, fallback=None, **kwargs):
        if self.exhausted:
            return fallback
        from .assembly.fragment_naming import PerfBudgetExceeded
        try:
            return rescue(*args, **kwargs)
        except PerfBudgetExceeded:
            if not self._owns_budget:
                raise
            self.exhausted = True
            logger.info("hang budget exhausted inside the failure rescue %s; "
                        "later rescues skipped, the main exit runs on a fresh budget",
                        getattr(rescue, "__name__", rescue))
            return fallback


#: cross-instance recursion guard for the parent-offer retry. The
# retry spawns FRESH inner instances whose own best-effort ship-failure hooks
# (`_try_demote_senior_group_rescue`, decomposition,...) can be top-level
# (`is_top_level_naming` keys off the fragment `visited` set, not session
# depth), so without this a demote-spawned inner could re-enter the parent
# offer. One flag per thread; set for the whole duration of an outer offer.
_ALT_PARENT_RESCUE = threading.local()



#: The fewest naming passes the clean-first offer (``_prefer_clean_over_floor_substitute``)
#: may make, whatever the main path made: one clean recovery of a small molecule.
_OFFER_MIN_PASSES = 64

class Orthonym:
    """The naming engine, with every option.

    Build one instance and name as many molecules with it as you like; the
    options apply to every call.:func:`name_compound` builds a fresh instance for
    each call.

    Creating an instance checks that the OPSIN and centres jars are present and
    stops with an error if they are not (run ``orthonym --fetch-jars``). Set
    ``ORTHONYM_ALLOW_REDUCED=1`` to name without them, with no OPSIN check.

    Parameters
    ----------
    style: {"pin", "general", "cas"}, default "pin"
        Naming style, as for:func:`name_compound`.
    enable_triviality_controller, enable_group_splitting, trivial_fallback: bool
        As for:func:`name_compound`. All default to False.
    general_fallback, general_fallback_unverified, allow_aromatic_general, full_coverage: bool
        The switches behind the command line's ``--emit-tier``. All default to
        False, which is the default tier. ``valid`` sets ``general_fallback``;
        ``complete`` adds ``allow_aromatic_general``; ``best-effort`` adds
        ``general_fallback_unverified``; ``full-coverage`` adds ``full_coverage``.
    binding_proof: {"off", "audit", "enforce"}, default "off"
        As for:func:`name_compound`.

    Notes
    -----
    The parameters whose names start with an underscore are for the engine's own
    use and tests; leave them at their defaults.

    Examples
    --------
    >>> from orthonym import Orthonym
    >>> namer = Orthonym
    >>> namer.name("CCO")
    'ethanol'
    >>> namer.name("CC(=O)O")
    'acetic acid'
    """

    def __init__(self, style: str = "pin", *,
                 _disable_grammar_validation: bool = False,
                 _disable_opsin_validity_gate: bool = False,
                 enable_triviality_controller: bool = False,
                 enable_group_splitting: bool = False,
                 trivial_fallback: bool = False,
                 general_fallback: bool = False,
                 general_fallback_unverified: bool = False,
                 allow_aromatic_general: bool = False,
                 full_coverage: bool = False,
                 _principal_group_override: Optional[str] = None,
                 _forced_parent_rank: int = 0,
                 _seed_excluded_dispatch_classes: frozenset = frozenset(),
                 binding_proof: str = "off"):
        """Set up the engine; see the class description for the parameters.

        Raises
        ------
        orthonym.jars.JarUnavailable
            If the OPSIN or centres jar is missing and ``ORTHONYM_ALLOW_REDUCED`` is
            not set.
        ValueError
            If ``binding_proof`` is not ``"off"``, ``"audit"`` or ``"enforce"``.
        """
        # Front door for the pinned OPSIN / centres jars: resolved once per process.
        # Missing -> JarUnavailable here, before any naming path can quietly fall back
        # to a weaker mode (no OPSIN check, a different CIP labeller). Opt out with
        # ORTHONYM_ALLOW_REDUCED=1.
        from .jars import require_all
        require_all()
        # Perf lever A1/A11 (2026-09-13): opt-in process tuning for batch workers
        # (ORTHONYM_GC_TUNE=on). A no-op unless the variable is set; see runtime_tuning.py.
        from .runtime_tuning import maybe_tune_from_env
        maybe_tune_from_env()
        #: validate eagerly. A misspelled mode that silently degraded to
        # "off" would present as a clean run with the proof never executing --
        # the one failure mode an audit flag must not have.
        self._binding_proof: str = validate_binding_proof(binding_proof)
        self.style = style
        # Task 1.9 (PIN-policy): fallback-only opt-in. When True the name path
        # substitutes a general-only retained name for the "unknown organic
        # compound" failure signal; it never overrides a derived PIN.
        self._trivial_fallback: bool = trivial_fallback
        #: opt-in general-engine inline fallback (decision 3:
        # PIN-strict default unchanged; lower tiers opt-in). When True, a
        # GENERAL-class abstention is retried through the binding-carrying
        # general engine (assembly/general_engine.py), gated by the E1
        # certificate and the existing downstream moat. Default False ->
        # default output byte-identical.
        self._general_fallback: bool = general_fallback
        #: opt-in — ship an E1-passed engine name when OPSIN cannot
        # verify it (absent jar / rejected / transient). A PARSED-but-
        # MISMATCHED name is never shipped at any tier.
        self._general_fallback_unverified: bool = general_fallback_unverified
        #: plumbing-only opt-in (inert until P1/P2 consume it). When
        # True, threaded into general_engine.name_general -> name_general_ring
        # -> vonbaeyer_universal.analyze_cage_universal(allow_mancude=...).
        # Default False -> default output byte-identical.
        self._allow_aromatic_general: bool = allow_aromatic_general
        #: the full-coverage opt-in (``--emit-tier full-coverage``).
        # The SINGLE bit that arms the D2 general coordination-additive
        # namer, which sits strictly ABOVE best-effort (full-coverage is
        # best-effort's production superset PLUS this marker). Default False ->
        # D2 dispatch is never reached and EVERY tier's output is
        # byte-identical (the SP5.4 isolation property). Published as
        # ``full_coverage_ctx`` for the recursive re-entry, torn down in the
        # ``name`` finally with its three siblings.
        self._full_coverage: bool = full_coverage
        # (169.5): per-instance bypass for the OPSIN validity gate, used
        # by neutralize-recurse / fragment intermediate naming (those produce an
        # INTERMEDIATE name that is transformed downstream, not a final output,
        # so they must not be gated). Final top-level name calls leave this
        # False -> the gate applies.
        self._disable_opsin_validity_gate: bool = _disable_opsin_validity_gate
        # a phase (verification as a filter over the dispatch cascade).
        # `_excluded_dispatch_classes` holds the classes already tried and
        # GATE-REJECTED for the molecule currently being named; the cascade skips
        # them so the next entry gets its turn. `_last_dispatch_class` is the
        # class that produced the name now under gate. Both are per-molecule
        # scratch: name clears them on entry and in a finally, so nothing
        # leaks between molecules (that would make naming order-dependent).
        # B2: an OPTIONAL seed of dispatch classes to exclude for EVERY
        # molecule named by this instance (default empty -> byte-identical). Used
        # by the aminium suffix-correctness re-derivation in charged_router to
        # skip the RETAINED_NAME handler (a retained diol-principal name such as
        # 'trometamol' cannot take a valid '-ium' suffix; excluding it lets the
        # systematic amine-principal parent be derived instead). It SURVIVES the
        # per-molecule reset below (that reset restores this seed, not the empty
        # set) so the exclusion is honoured on the first pass, not only on a
        # gate-rejection retry.
        self._seed_excluded_dispatch_classes: frozenset = frozenset(
            _seed_excluded_dispatch_classes)
        self._excluded_dispatch_classes: frozenset = self._seed_excluded_dispatch_classes
        self._last_dispatch_class = None
        # a phase T3: scoped principal-group override (FG name). Set ONLY by the
        # charged chokepoint's S/P-oxoacid-anion re-entry (_reenter_forced) so the
        # anionic group is forced as the principal characteristic group per /.
        # None for every normal name call -> byte-identical production behaviour.
        self._principal_group_override: Optional[str] = _principal_group_override
        # (offer-not-return, a project rule): force the k-th-ranked
        # parent instead of ``ranked[0]``. 0 for every normal name call ->
        # byte-identical. Set >0 ONLY by ``_try_alternate_parent_rescue`` on a
        # FRESH inner instance, after ``ranked[0]`` has already abstained, and
        # every candidate it yields is RT-gated before adoption (0-wrong).
        self._forced_parent_rank: int = _forced_parent_rank
        # Size of the top-level parent pool from the most recent top-level
        # classification (written by _classify). Lets the offer-retry bound its
        # loop at the real pool length. Reset per top-level name below.
        self._top_parent_pool_size: int = 1
        # a phase +: per-instance counter dict, NEVER
        # module-global. Pre-seed all seven buckets so callers see a
        # complete histogram even before any name invocation.
        from .validation.opsin_grammar import OpsinGrammar
        self._grammar_stats: Dict[str, int] = {
            k: 0 for k in OpsinGrammar.STAT_KEYS
        }
        if _disable_grammar_validation:
            self._grammar = None
        else:
            # Share the dict by reference per: the grammar
            # instance increments the same dict the Orthonym instance
            # exposes via `get_validation_stats`.
            self._grammar = OpsinGrammar(stats=self._grammar_stats)

        # a phase NEW: instantiate CFR router (default-ON; no opt-out
        # flag per internal notes + +). Per the audit
        # fresh-instance invariant: each Orthonym carries its OWN router
        # with empty dispatch_stats; recursive `Orthonym(...)` calls produce
        # fresh routers, and the byte-identical contract is on NAME OUTPUT
        # only, NOT on per-call dispatch_stats.
        from .routing import ClassFirstRouter
        self._cfr_router = ClassFirstRouter()

        # a phase option (b) NEW: skip-decomposition flag threading.
        # `_name_impl(_skip_decomposition=True)` (called by
        # `name_pipeline_only`) sets this on entry; the
        # DECOMPOSITION_PRE_GENERAL predicate factory reads from kwargs
        # passed by `dispatch`.
        self._skip_decomposition: bool = False

        # a phase: triviality-controller opt-in (default OFF = Stage A SACRED
        # byte-identical canary invariant). Env override via ORTHONYM_ENABLE_TRIVIALITY_CONTROLLER.
        # Task 1.9 (context resolution 3): trivial_fallback=True implies the
        # triviality controller (one user intent: "allow non-PIN trivial output").
        self._enable_triviality_controller: bool = (
            enable_triviality_controller or _DEFAULT_TRIV or trivial_fallback
        )
        # WARNING #9 fix (CRITICAL for runtime per internal notes): instantiate the OpsinOracle
        # when the flag is ON so the RT-safety gate can run. Without it, OpsinOracle.rt_safe
        # degrades to the permissive always-True fallback and runtime enforcement is silently
        # disabled. Default OFF keeps the oracle None (zero cost, byte-identical Stage A).
        self._triv_oracle = None
        if self._enable_triviality_controller:
            try:
                from .assembly.retained_substitution import OpsinOracle
                from .validation.opsin_roundtrip import _find_opsin_jar
                _jar = _find_opsin_jar()
                self._triv_oracle = OpsinOracle(opsin_jar=_jar)
            except Exception as exc:
                logger.warning(
                    "Phase 168 OpsinOracle instantiation failed: %s; "
                    "T2 RT-safety degrades to permissive fallback", exc,
                )
                self._triv_oracle = None

        # a phase: group-splitting flag (default OFF -> byte-identical Stage A).
        # Env override ORTHONYM_ENABLE_GROUP_SPLITTING. The OpsinOracle is instantiated
        # only when the flag is ON so the FAIL-CLOSED per-split RT gate  can run;
        # default OFF keeps it None (zero cost, byte-identical Stage A).
        self._enable_group_splitting: bool = enable_group_splitting or _DEFAULT_GS
        self._split_oracle = None
        if self._enable_group_splitting:
            try:
                from .assembly.retained_substitution import OpsinOracle
                from .validation.opsin_roundtrip import _find_opsin_jar
                _gs_jar = _find_opsin_jar()
                self._split_oracle = OpsinOracle(opsin_jar=_gs_jar)
            except Exception as exc:
                logger.warning(
                    "Phase 169 group-split OpsinOracle instantiation failed: %s; "
                    "per-split RT gate FAIL-CLOSED (splits rejected)", exc,
                )
                self._split_oracle = None

    def get_dispatch_stats(self) -> Dict[Any, int]:
        """Return how often each compound-class route was taken by this instance.

        The engine tries compound classes in a fixed order (salts, ions,
        carbohydrates, natural products, the general rules and more) and counts
        which class named each molecule.

        Returns
        -------
        dict
            Compound class to count. The keys are members of
            ``orthonym.routing.StoutClass``.
        """
        return self._cfr_router.get_dispatch_stats()

    def reset_dispatch_stats(self) -> None:
        """Set the route counters back to zero.

        Resets this instance's class counters and the finer counters of
        :meth:`get_inner_dispatch_stats`. The finer counters are shared by every
        instance in the process, so this resets them for all instances.
        """
        self._cfr_router.reset_dispatch_stats()
        # a phase: also reset the inner-dispatch counter.
        from .assembly.inner_dispatch import reset_inner_dispatch_stats
        reset_inner_dispatch_stats()

    def get_inner_dispatch_stats(self) -> Dict[str, int]:
        """Return how often each handler inside the general class was used.

        Within the general compound class, a second step picks one of several
        handlers (acids, esters, amines and so on). These counters are shared by
        every instance in the process;:meth:`reset_dispatch_stats` clears them.

        Returns
        -------
        dict of str to int
            Handler id to count; empty before the first molecule.
        """
        from .assembly.inner_dispatch import get_inner_dispatch_stats as _stats
        return _stats()

    def name_with_tree(self, smiles: str):
        """Name one molecule and return the parts of the name.

        Parameters
        ----------
        smiles: str
            The structure, as a SMILES string.

        Returns
        -------
        NamingResult
            ``name`` is the same string:meth:`name` returns; ``tree`` is the
            :class:`NameTreeNode` of its parts (a single coarse node when the part of
            the engine that built the name records no finer structure);
            ``atom_to_locant_hint`` maps atom indices to locants where one was
            recorded, else ``None``.

        Raises
        ------
        ValueError
            If RDKit cannot read the SMILES.

        Examples
        --------
        >>> from orthonym import Orthonym
        >>> Orthonym.name_with_tree("OC1CCCCC1").name
        'cyclohexanol'
        """
        # Lone-pair centre written first: the standard spelling of the input is named
        # when RDKit reads a lone-pair centre of it unlike the standard reading
        # (_lone_pair_standard_spelling); the atom-to-locant hint is then re-keyed to
        # the atoms of the caller's string. Any other input is unchanged.
        spelled, _lp_opened = _lone_pair_input_enter(smiles)
        try:
            result = self._name_with_tree_impl(spelled)
            if spelled != smiles and result.atom_to_locant_hint:
                result = result._replace(
                    atom_to_locant_hint=_lone_pair_caller_atoms(result.atom_to_locant_hint))
            return result
        finally:
            _lone_pair_input_exit(_lp_opened)

    def _name_with_tree_impl(self, smiles: str):
        """The body of:meth:`name_with_tree`, for the string it names."""
        # Lazy import to avoid composer.py -> name_tree -> namer.py cycle
        # at module-load time.
        from .assembly.name_tree import NamingResult
        # part B + W7: install a per-call capture slot via
        # contextvars.ContextVar (PEP 567) so composer._assemble_name_impl
        # can write the inner-dispatch NamingResult into it without
        # changing the public assemble_name return type. ContextVar is
        # thread-local AND asyncio-task-local — safe under concurrent
        # invocation from multiple threads / tasks.
        token = _name_with_tree_capture.set({"naming": None})
        try:
            try:
                name = self.name(smiles)
            except ValueError:
                raise  # invalid SMILES — propagate (matches name contract)
            except (TypeError, KeyError, IndexError, AttributeError) as _e:
                # robustness: name_with_tree must be as resilient as the
                # module-level name_compound. The inner handlers' "pool.best.name
                # raises on None" fall-through contract surfaces here because
                # self.name lacks name_compound's broad except. Recover with the
                # same descriptive fallback so name_with_tree never crashes and its
                # name matches the canary's name_compound output.
                logger.debug(
                    "name_with_tree naming error for %s: %s: %s",
                    smiles, type(_e).__name__, _e,
                )
                name = _descriptive_fallback(smiles)
            slot = _name_with_tree_capture.get()
            captured = slot["naming"] if slot else None
            tree = captured.tree if captured is not None else None
            hint = captured.atom_to_locant_hint if captured is not None else None
        finally:
            _name_with_tree_capture.reset(token)
        if name:
            # a phase + boundary guarantee. Two failure modes:
            # (a) tree is None — salts/ions/radicals/retained names are produced
            # by paths BELOW dispatch_inner that never write the capture slot.
            # (b) STALE tree — the inner handler wrote the slot, but downstream
            # _name_impl processing (decomposition engine, coverage gate,
            # stereo backstop) OVERRODE the final name afterward, so the
            # captured tree no longer round-trips to it.
            # In BOTH cases synthesize the sanctioned coarse node (, counted)
            # so name_tree_to_string(tree) == name holds for EVERY SMILES
            # and --dump-tree works universally . The str fragment_legacy
            # round-trips verbatim. class_id="coarse_fallback" distinguishes this
            # boundary node from real handler trees in the coarse-bucket report.
            from .assembly.name_tree import NameTreeNode
            from .assembly.name_tree_to_string import (
                NameTreeSerializerError,
                name_tree_to_string,
            )
            #: this staleness check is the defence point against bad trees,
            # so it must not itself crash on one. name_tree_to_string raises
            # NameTreeSerializerError (a ValueError subclass) on a malformed node
            # (empty parent_stem + non-str fragment_legacy). An uncaught raise
            # here would escape as ValueError, which this method's docstring maps
            # to "invalid SMILES" — mis-surfacing a malformed captured tree on a
            # perfectly valid input. Treat a malformed/unserializable captured
            # tree exactly like a stale one: synthesize the sanctioned coarse
            # fallback. This keeps name output byte-identical (only the tree path
            # is affected).
            needs_fallback = tree is None
            if not needs_fallback:
                try:
                    needs_fallback = name_tree_to_string(tree) != name
                except NameTreeSerializerError:
                    needs_fallback = True
            if needs_fallback:
                tree = NameTreeNode(
                    parent_stem=name, class_id="coarse_fallback",
                    iupac_section_cite="P-73", fragment_legacy=name,
                )
        return NamingResult(name=name, tree=tree, atom_to_locant_hint=hint)

    def get_validation_stats(self) -> Dict[str, int]:
        """Return how often the OPSIN grammar pre-check passed or repaired a name.

        The engine checks a candidate against OPSIN's grammar before the full round
        trip, and can fix brackets, stereodescriptors or hyphens. The counters
        belong to this instance and start at zero.

        Returns
        -------
        dict of str to int
            One counter per outcome, for example ``validate_passed`` and
            ``repair_succeeded_bracket``.
        """
        return dict(self._grammar_stats)

    # a phase. Bounded deliberately: each retry is a full re-name plus an
    # OPSIN round-trip, and the cascade's useful alternatives for one molecule
    # are few. 3 was chosen as the smallest bound that lets a molecule try the
    # GENERAL engine after two specialised classes have been rejected.
    _PHASE4_MAX_GATE_RETRIES = 3

    def _retry_cascade_on_gate_rejection(self, smiles, pre_gate, gated):
        """ a phase: a GATE REJECTION re-enters the dispatch cascade.

        The cascade in ``_name_impl`` already falls through when a handler
        returns ``None``, but the gates run out here in ``name`` -- AFTER the
        cascade has exited -- so a handler that produced a name the gate then
        rejected aborted the whole molecule with no retry. That is the defect
        this phase exists to close: "keep the first that clears both gates, fall
        through instead of aborting the molecule."

        ⚠ The roadmap framed this as filtering over ``CandidatePool``. It is not:
        the pool holds ONE candidate in 158 of 158 traced calls. The real
        generator of alternatives is the DISPATCH CASCADE, and this works with
        it directly.

        Only ever converts an ABSTENTION into a name: it runs solely when the
        gate replaced a real name with the descriptive fallback, and it returns
        that same fallback unless a later class produces a name that PASSES the
        gate. A name that already shipped cannot be changed by this path, so
        0-wrong is preserved by construction -- every replacement has cleared
        the same OPSIN round-trip the original failed.
        """
        from .errors import _DESCRIPTIVE_FALLBACK_NAMES
        # Did the gate actually suppress a REAL name? If the handler already
        # abstained, there is nothing to retry past.
        if (gated == pre_gate
                or gated not in _DESCRIPTIVE_FALLBACK_NAMES
                or not pre_gate
                or pre_gate in _DESCRIPTIVE_FALLBACK_NAMES):
            return gated

        for _ in range(self._PHASE4_MAX_GATE_RETRIES):
            cls = self._last_dispatch_class
            if cls is None or cls in self._excluded_dispatch_classes:
                break
            self._excluded_dispatch_classes |= {cls}
            try:
                cand = self._name_impl(smiles)
            except Exception:                                   # noqa: BLE001
                # A later class raising is a decline, not a failure of the
                # molecule -- the original fallback still ships.
                break
            if not cand or cand in _DESCRIPTIVE_FALLBACK_NAMES:
                continue
            _mol = Chem.MolFromSmiles(smiles)
            if _mol is None:
                break
            from .assembly.coverage_scoring import retrieve_confidence
            _conf = retrieve_confidence()
            cand = _final_stereo_check(
                _mol, cand, handler=_conf.get('handler', 'unknown'),
                atom_to_locant=_conf.get('atom_to_locant'),
                is_phenol_benzene=_conf.get('is_phenol_benzene'),
            )
            cand = _final_grammar_check(
                cand, smiles, _conf.get('handler', 'unknown'),
                self._grammar, self._grammar_stats,
            )
            # (a review F1): capture the gate's OUTPUT and return THAT, not the
            # pre-gate `cand`. The gate can now return a name DIFFERENT from its
            # input — the stereo-omission reclaim composes the input's dropped
            # stereo back on — so returning `cand` here would ship the flat,
            # stereo-INCOMPLETE name at the RT-verified tier (which the BE-STRICT
            # contract forbids). Before that reclaim the gate only ever returned
            # `cand` or a fallback, so `return cand` was equivalent; it is not now.
            _gated_cand = _final_opsin_validity_gate(
                cand, smiles, self._grammar_stats,
                besteffort_unverified=self._general_fallback_unverified,
                general_fallback_tier=self._general_fallback)
            if _gated_cand not in _DESCRIPTIVE_FALLBACK_NAMES:
                return _gated_cand
        return gated

    @_budget_scope
    def name(self, smiles: str, *, raise_on_limit: bool = False) -> str:
        """Name one molecule with this instance's options.

        Parameters
        ----------
        smiles: str
            The structure, as a SMILES string.
        raise_on_limit: bool, default False
            Raise:class:`OrthonymLimitError` for a structure the engine cannot
            handle, instead of returning a label.

        Returns
        -------
        str
            The name, or a label that says why no name was given (see
            :func:`orthonym.errors.is_failure_name`).

        Raises
        ------
        ValueError
            If RDKit cannot read the SMILES.
        OrthonymLimitError
            If ``raise_on_limit`` is true and the structure is out of scope. For a
            ring system the engine cannot name yet the code is
            ``UNSUPPORTED_RING_SYSTEM``; when that happens inside a part of the
            molecule, the code reported can be the more general ``UNNAMEABLE``.

        Examples
        --------
        >>> from orthonym import Orthonym
        >>> Orthonym.name("C/C=C/C")
        '(2E)-but-2-ene'
        """
        if not _PIN_PROMOTION_WRAPPED.get() and self._pin_promotion_eligible():
            return self._name_with_pin_promotion(smiles, raise_on_limit=raise_on_limit)
        # Start runtime fragment cache session (only at top-level depth)
        from .assembly.fragment_naming import (
            end_naming_session,
            is_top_level_naming,
            start_naming_session,
        )
        # An unparseable SMILES raises ValueError (the documented contract,
        # "Raises: ValueError: If SMILES is invalid"), checked BEFORE the session
        # opens and before the producer try below. That try's catch-all exists so
        # a VALID input never crashes (Phase1 B5); it turned _name_impl's own
        # "Invalid SMILES" ValueError into the descriptive fallback 'unknown', and
        # the CLI's --dump-tree then exited 0 on garbage. Top level only: a
        # recursive fragment call keeps its fail-soft return. The empty string
        # parses to an empty molecule and keeps its 'unknown'.
        if smiles and is_top_level_naming() and Chem.MolFromSmiles(smiles) is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        start_naming_session()
        # (ContextVar, Token) pairs THIS frame publishes below; every exit resets
        # them LIFO via _reset_published_ctx (a frame-local list, see there).
        _ctx_tokens: list = []
        # Wave2 (cross-molecule stereo-contamination fix): the confidence /
        # candidate-pool / parent-correctness thread-locals are PER-MOLECULE, but
        # they persist across name calls (documented at the post-dispatch gate
        # below). The pytest conftest clears them per-test; production (and the
        # phase-gate / determinism loops, which name 1000+ molecules through one
        # instance) never did — so a molecule whose handler leaves an
        # allow-injecting {handler, atom_to_locant} entry (e.g. a benzene
        # ring-substituent) contaminated the NEXT molecule's _final_stereo_check,
        # injecting a spurious order-dependent descriptor ('(3R,5R)-stigmastane').
        # Reset at the TOP of each top-level session so every molecule starts
        # clean — the same invariant conftest enforces for the suite.
        if is_top_level_naming():
            try:
                from .assembly.coverage_scoring import clear_confidence
                clear_confidence()
            except Exception:
                pass
            try:
                from .assembly.candidate_pool import clear_pool
                clear_pool()
            except Exception:
                pass
            # a phase: same invariant as the clears around it. The
            # gate-rejection exclusion set is PER-MOLECULE; leaking it would
            # make a molecule's name depend on what was named before it, which
            # is precisely the order-dependence this block exists to stop.
            self._excluded_dispatch_classes = self._seed_excluded_dispatch_classes
            self._last_dispatch_class = None
            #: same per-top-level invariant. The pool size is stale
            # until _classify reruns; default 1 = "no alternative parent" so a
            # molecule that never reaches parent selection cannot trigger
            # the offer-retry on a previous molecule's pool.
            self._top_parent_pool_size = 1
            # Task 0.1: reset the typed-abstention telemetry slot per
            # top-level molecule (same invariant as the confidence/pool
            # clears above). Side-effect-only — never changes a name.
            try:
                from .metrics.abstention import clear_abstention
                clear_abstention()
            except Exception:
                pass
            #: same per-top-level-molecule invariant as the clears above.
            # A batch consumer names hundreds of inputs in one process; without
            # this reset, molecule N's audit would carry molecule N-1's discarded
            # candidates and manufacture selection failures that never happened.
            # Stays ENABLED across the reset (clear_ledger, not disable).
            try:
                from .metrics.candidate_ledger import (
                    clear_ledger as _clear_cand_ledger,
                )
                from .metrics.candidate_ledger import (
                    is_enabled as _cand_ledger_on,
                )
                if _cand_ledger_on():
                    _clear_cand_ledger()
            except Exception:
                pass
            #: reset the binding-proof ledger per top-level molecule,
            # for the same reason as the three clears above. Without it a
            # molecule that records no spine would finalize against the
            # PREVIOUS molecule's record and report its verdict.
            try:
                from .validation.proof_ledger import clear_ledger
                clear_ledger()
            except Exception:
                pass
            # a phase L0: same per-top-level-molecule invariant as the
            # clears above. `_last_ger_result` is stashed by
            # `_record_binding_proof` for the SHADOW coverage audit at
            # `_finish`; without this reset a molecule with NO general-engine
            # candidate at all would inherit the PREVIOUS molecule's result
            # object (the `.name == name` staleness guard in `_finish` makes a
            # false-positive match unlikely but not impossible on a name-string
            # coincidence, and leaving stale state around is the wrong default
            # regardless). `_last_coverage_verdict` is reset for the same
            # reason -- a molecule the audit never reaches (mode "off", or an
            # exception) must not report a PREVIOUS molecule's verdict.
            self._last_ger_result = None
            self._last_coverage_verdict = None
            # a phase L2: same per-top-level-molecule invariant as the two
            # resets above -- `self._offers` is rebuilt at `_finish` from
            # THIS molecule's winner; without this reset a molecule whose
            # audit never reaches the offer-building code (mode "off", an
            # exception, or a non-top-level recursive call) would report the
            # PREVIOUS molecule's offer pool instead of an empty one.
            self._offers = []
            # a phase L3-1: same per-top-level-molecule invariant --
            # `_t4_floor_candidate` is a telemetry stash (mirroring
            # `_last_ger_result`) of the systematic-floor name the LAST
            # `_maybe_append_t4_floor_offer` call computed, if any.
            self._t4_floor_candidate = None
            # a phase cleanup T1: same per-top-level-molecule invariant --
            # `_last_selected_offer` stashes the WINNING `Offer`
            # `_select_rt_passing_offer_name` picked (or `None` if it fell
            # back to `current_name` unchanged), so `name_tiered` can derive
            # tier/is_pin/source from the offer that actually won rather than
            # the process-wide provenance contextvar, which still describes
            # whichever producer ran LAST (the losing primary on a floor win).
            self._last_selected_offer = None
            # D3: the name `_maybe_prefix_order_fallback` shipped for THIS
            # molecule, if any (read by `name_tiered`), reset for the same reason.
            self._prefix_order_fallback_name = None
            #: publish the engine flag so fragment/component recursion
            # (name_compound builds FRESH namers) inherits it. Top-level only;
            # reset in the finally below.
            try:
                from .metrics.provenance import general_fallback_ctx
                _ctx_tokens.append((general_fallback_ctx, general_fallback_ctx.set(
                    self._general_fallback)))
            except Exception:
                pass
            #: publish the BEST-EFFORT discriminator for the same reason and
            # with the same lifetime -- read by
            # `composer._integrate_universal_prefixes` to pick the substituent
            # VOCABULARY, which it cannot do from its own arguments because its
            # callers are tier-unaware PIN handlers.
            try:
                from .metrics.provenance import best_effort_ctx
                _ctx_tokens.append((best_effort_ctx, best_effort_ctx.set(
                    self._general_fallback_unverified)))
            except Exception:
                pass
            #: publish allow_aromatic_general with the same lifetime so
            # the recursive re-entry (name_compound builds a FRESH namer) gives a
            # recursively named fragment the SAME tier the top-level call ran at.
            # Without this a top-level allow_aromatic_general=True was silently lost
            # on recursion. Top-level only; reset in the finally below.
            try:
                from .metrics.provenance import allow_aromatic_general_ctx
                _ctx_tokens.append((allow_aromatic_general_ctx, allow_aromatic_general_ctx.set(
                    self._allow_aromatic_general)))
            except Exception:
                pass
            #: publish the full-coverage opt-in with the same lifetime
            # so the recursive re-entry (name_compound builds a FRESH namer)
            # inherits it. Top-level only; reset in the finally below. Default
            # False keeps D2 unreachable and every tier byte-identical.
            try:
                from .metrics.provenance import full_coverage_ctx
                _ctx_tokens.append((full_coverage_ctx, full_coverage_ctx.set(
                    self._full_coverage)))
            except Exception:
                pass
        # Prelude guard (TRIAGE C2, 2026-09-27): an exit BEFORE the main try below --
        # the isotope decorator's two returns, or an exception anywhere in this
        # prelude -- tears down exactly what the main finally tears down: the four
        # ctx tokens published above and the naming session. The isotope exits used
        # to end only the session, so one best-effort naming of an isotope-labelled
        # molecule left general_fallback_ctx / best_effort_ctx /
        # allow_aromatic_general_ctx True for every LATER naming in the process, and
        # a default PIN call then shipped a best-effort-tier name
        # ('(2S)-2-(hydroxyamino)butane' for ON[C@@H](C)CC, which the PIN tier
        # declines). A name must never depend on what the process named before.
        _prelude_done = False
        try:
            # --- Wave-2 P2: isotopic substitution decorator / ---
            # RDKit skeleton perception ignores GetIsotope, so an isotope-labeled
            # mol would name as the UNLABELED skeleton (wrong PIN). Route it to the
            # fail-closed decorator BEFORE _name_impl strips the label. Clean
            # pass-through when the mol carries no isotope (has_isotopes gate) —
            # zero cost + byte-identical on the entire unlabeled corpus.
            if is_top_level_naming():
                _iso_probe = Chem.MolFromSmiles(smiles)
                if _iso_probe is not None:
                    from .rules.isotopes import decorate_isotopic_name, has_isotopes
                    if has_isotopes(_iso_probe):
                        _iso_name = decorate_isotopic_name(smiles, self.style, self)
                        if _iso_name is not None:
                            # (fix wave 1): every exit of name goes
                            # through _finish. This one lives OUTSIDE the main
                            # try/finally below; the prelude guard's finally owns
                            # its teardown, in the same order as the hooked exits
                            # (proof asserted while the session is still live,
                            # session ended before the value is handed back).
                            return self._finish(_iso_name, smiles)
                        # Decorator failed closed on an isotope-labeled molecule: REFUSE.
                        # Falling through would emit the UNLABELED skeleton name (label
                        # silently dropped) — a wrong name the OPSIN validity gate cannot
                        # catch (it parses to the unlabeled structure; ignores
                        # isotopes). Accuracy-first: never emit a label-dropping name.
                        from .metrics.abstention import AbstentionCode, record_abstention
                        record_abstention(AbstentionCode.OTHER,
                                          detail='isotope_decorator_failed')
                        return self._finish(_descriptive_fallback(smiles), smiles)
            # The input is named as drawn: nothing here rewrites ``smiles`` before the
            # producers and the gates see it, because every gate compares the name
            # with THIS structure. A cation beside a NEUTRAL inorganic oxoacid
            # ([Na+].OC(=O)O, net charge +1) used to be deprotonated to the balanced
            # salt first; the salt name then verified against the rewritten
            # structure, so 'sodium hydrogen carbonate' (NaHCO3, net 0, full
            # InChIKey...-M) shipped as verified for an input whose key ends in -N.
            # A different protonation state is a different species. Such an input is
            # a mixed adduct of the cation and the acid: (the Blue Book:
            # 4667) "names are formed by citing the names of individual compounds,
            # connected by long (em) dashes" -> 'carbonic acid—sodium(1+) (1/1)' at
            # the general tiers, with no PIN ("preferred IUPAC names cannot be
            # assigned to mixed adducts",:4667), so the PIN tier abstains.
            # M1 (Levers C1/E): open the scoped-per-call memo so name_substituent /
            # name_pipeline_only memos live for exactly this naming call and are torn
            # down in the finally below (bounded, determinism-safe). push_scope returns
            # None on a nested re-entry (the inner call shares the outer cache), so only
            # the outermost frame tears it down. See assembly/memo.py.
            from .assembly.memo import pop_scope as _memo_pop
            from .assembly.memo import push_scope as _memo_push
            _memo_scope_token = _memo_push()
            _prelude_done = True
        finally:
            if not _prelude_done:
                self._reset_published_ctx(_ctx_tokens)
                end_naming_session()
        try:
            # + Wave-0 D1: structural scope pre-check, now UNCONDITIONAL.
            # A wildcard atom makes the input InChIKey uncomputable, so the
            # round-trip oracle fails OPEN and a wildcard input ships a
            # WRONG molecule (CC* -> "ethane"). Refuse it before any producer OR
            # the general-engine recovery runs — recovery would re-run producers
            # on the wildcard mol and re-open the hole, so this must NOT route
            # through the OrthonymLimitError handler's recovery path below.
            # Perf: zero-false-negative pre-filter -- an RDKit dummy/wildcard
            # atom (atomic number 0) is spelled ONLY as `*`/`[*]` (bare `*`,
            # `[*]`, `[1*]`,...) or as `[#0]`/decorated (`[13#0]`, `[#0-]`,
            #...) -- both spellings are checked, so the filter has no false
            # negative, and skips the extra RDKit parse on the common
            # (non-wildcard) path.
            if '*' in smiles or '#0' in smiles:
                _probe = Chem.MolFromSmiles(smiles)
                if _probe is not None:
                    _scope = classify_scope_limit(_probe)
                    if _scope is not None:
                        _scope.smiles = smiles
                        if raise_on_limit:
                            raise _scope
                        # Default path: honest descriptive fallback, byte-identical
                        # to the current output for a wildcard whose producer
                        # already fails (idx3), via the SAME _apply_trivial_fallback
                        # the limit handler uses — but WITHOUT recovery.
                        from .metrics.abstention import AbstentionCode, record_abstention
                        record_abstention(AbstentionCode.OTHER, detail=_scope.code)
                        return self._finish(
                            self._apply_trivial_fallback(_scope.message, smiles),
                            smiles)
            try:
                result = self._name_impl(smiles)
            except OrthonymLimitError as _limit:
                # G0 fail-closed (DD7 S1): a ring subsystem refused to emit a
                # structurally-wrong name (de-aromatised von-Baeyer cage /
                # phantom ring-as-substituent). Default path -> the descriptive
                # fallback string ('unknown organic compound'); opt-in path ->
                # re-raise the named limit. This is the single catch point; the
                # signal propagated un-wrapped from the assembly layer.
                if _limit.smiles is None:
                    _limit.smiles = smiles
                # Task 0.1: a top-level ring-system refusal means the
                # PARENT structure itself was declined (NO_PARENT); the same
                # signal from a nested fragment frame is a BRANCH failure of
                # the outer molecule. Other limit codes stay in the residual.
                from .metrics.abstention import AbstentionCode, record_abstention
                if _limit.code == 'UNSUPPORTED_RING_SYSTEM':
                    record_abstention(
                        AbstentionCode.NO_PARENT if is_top_level_naming()
                        else AbstentionCode.BRANCH_UNNAMEABLE,
                        detail=_limit.code,
                    )
                else:
                    record_abstention(AbstentionCode.OTHER, detail=_limit.code)
                if raise_on_limit and is_top_level_naming():
                    raise
                # (opt-in): a raised limit bypassed the GENERAL-block
                # engine wiring; give the engine its late, fully-gated shot.
                _rec = self._try_general_engine_recovery(smiles)
                if _rec is not None:
                    #: an early exit that bypasses even
                    # _apply_trivial_fallback, and the LATE-RECOVERY path --
                    # precisely the one carrying the known stale-certificate
                    # hazard (_retained_structural_preference can replace the
                    # whole name after verify_certificate passed). Leaving it
                    # unhooked would hide the finding this audit exists to see.
                    return self._finish(_rec, smiles)
                # Task 1.9: PIN fails closed; --trivial falls back to a
                # general-only retained name when no PIN could be derived.
                # (fix wave 1): --trivial is a real opt-in and this
                # swap ships a GENUINE retained name (is_failure_name False),
                # so the exit must be audited like any other.
                return self._finish(
                    self._apply_trivial_fallback(_limit.message, smiles),
                    smiles)
            except Exception as _exc:  # noqa: BLE001
                # Phase1 B5: a producer raised an UNEXPECTED exception. A
                # namer must never crash on a valid input -- degrade to the late
                # general-engine recovery (RT-gated -> 0-wrong-safe), and if that
                # too declines, to an honest abstention. Measured: a ring-ketone
                # producer (name_cyclic_oxo_compound) raises on some fused cages
                # the systematic path names completely; before this the
                # exception escaped name and every caller without its own
                # try/except crashed (only the eval harness's worker caught it).
                # The strict opt-in still re-raises so a real bug is not masked
                # in that mode.
                logger.info("name_impl raised %s; routing to recovery: %s",
                            type(_exc).__name__, _exc,
                            exc_info=logger.isEnabledFor(logging.DEBUG))
                # ORTHONYM_STRICT=1 (audit 2026-09-03, R4): the environment
                # form of raise_on_limit, so a batch run can fail loudly on a
                # programming error instead of recording an abstention.
                if (raise_on_limit or _strict_mode()) and is_top_level_naming():
                    raise
                try:
                    _rec = self._try_general_engine_recovery(smiles)
                except Exception:  # noqa: BLE001 - recovery must not re-crash
                    _rec = None
                if _rec is not None:
                    return self._finish(_rec, smiles)
                return self._finish(_descriptive_fallback(smiles), smiles)
            # Universal stereo backstop (a phase,)
            # Only apply at top level -- decomposition fragments handle stereo
            # through their own naming paths.
            if is_top_level_naming():
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    from .assembly.coverage_scoring import retrieve_confidence
                    _conf = retrieve_confidence()
                    handler = _conf.get('handler', 'unknown')
                    # a phase -01 : thread the authoritative parent
                    # map + phenol flag to the backstop.
                    result = _final_stereo_check(
                        mol, result, handler=handler,
                        atom_to_locant=_conf.get('atom_to_locant'),
                        is_phenol_benzene=_conf.get('is_phenol_benzene'),
                    )
                    # Universal OPSIN-grammar backstop (a phase,).
                    # Order: stereo-backstop -> grammar-backstop. Stereo
                    # may have repositioned descriptors that grammar
                    # then re-validates.
                    result = _final_grammar_check(
                        result, smiles, handler,
                        self._grammar, self._grammar_stats,
                    )
                    # (169.5): real-OPSIN validity gate, the last
                    # transform before the name leaves name. Default-ON,
                    # fail-OPEN on no-JAR. Skipped for neutralize-recurse /
                    # fragment intermediates (self._disable_opsin_validity_gate).
                    if not self._disable_opsin_validity_gate:
                        _pre_gate = result
                        result = _final_opsin_validity_gate(
                            result, smiles, self._grammar_stats,
                            besteffort_unverified=self._general_fallback_unverified, general_fallback_tier=self._general_fallback,
                        )
                        # a phase: the gate is now a FILTER over the
                        # cascade, not a terminal abort. Only fires when a real
                        # name was suppressed; returns the same fallback unless
                        # a later class clears the gate.
                        result = self._retry_cascade_on_gate_rejection(
                            smiles, _pre_gate, result,
                        )
            # DETERMINISM (w2f p11): abstention must have ONE canonical sentinel.
            # Some fail-closed paths (a handler that DECLINES without producing a
            # candidate — e.g. lambda-multiring-spiro unsupported) leave result ==
            # '', while paths that produce a candidate the validity gate then
            # suppresses go through _descriptive_fallback -> 'unknown organic
            # compound'. For a molecule whose winning path is atom-ordering-
            # dependent, name then returns '' on some SMILES spellings and
            # 'unknown organic compound' on others — both mean "no name", but they
            # are different strings, so the determinism eval (raw-string compare)
            # flags it. An empty string is never a valid IUPAC name; normalise it
            # to the canonical fallback so abstention is a single deterministic
            # string regardless of which internal path abstained. is_unknown
            # already treats '' and the fallback as equivalent, so gold matching
            # is unchanged.
            if not (result and result.strip()):
                result = _descriptive_fallback(smiles)
            #: a WELDED refusal sentinel is not a name.
            #
            # `is_failure_name` matches the 'unknown…' and '… (not supported)'
            # families by SUBSTRING, so a decorated occurrence of those is already
            # caught here. The substituent cascade's placeholder was not: it
            # arrives with a locant or italic element prefix attached
            # ('N-substituentformamide'), so the slot-level equality checks let it
            # through and the welded string shipped as a name with a real tier.
            # Measured over 10,000 rows: 310 such emissions, and 0 of the 3,616
            # round-tripping names contain the substring.
            #
            # Normalised to the descriptive fallback HERE, *above* the recovery
            # attempts below, precisely so the general-engine and decomposition
            # fallbacks still get their shot — a project rule: removing a wrong
            # output must never be the last step, because it can unmask a worse
            # generator or a silent atom drop. So this can only turn a welded
            # non-name into either a REAL name (recovery succeeds) or an honest
            # labelled abstention, never into silence.
            #
            # 2026-09-25 (pre-existing-failures plan, Task 3): "already caught"
            # above was true of the recovery TRIGGERS only. Every rescue below
            # keys on `is_failure_name`, so a welded 'unknown…' did get its
            # rescues -- but when they all declined, the welded string itself
            # was what this method returned: '(2E)-2-methyl-5-unknownpent-2-enoic
            # acid' (CHEBI:131506) shipped from name and name_tiered whenever
            # the OPSIN validity gate was off. The producer that built it now
            # voids its candidate (errors.UnnameableSubstituentError); this is
            # the exit-side half of the same invariant, for every family: a
            # result that carries a sentinel but is not a WHOLE descriptive
            # fallback is not a name. Bare 'unknown' is the legacy whole
            # UNNAMEABLE message and is left as it is. The rescues' triggers are
            # unchanged (both strings are failures), and the four breadth
            # baselines of the plan hold 0 such rows (checked 2026-09-25).
            if (result and is_refusal_sentinel(result)
                    and result not in _DESCRIPTIVE_FALLBACK_NAMES
                    and result.strip().lower() != 'unknown'):
                result = _descriptive_fallback(smiles)
            # Breadth -- ZINC loss: every step from here to the main exit is an
            # optional rescue of the failure the main path produced. A rescue
            # that exhausts the molecule's hang budget declines instead of
            # abstaining the whole molecule, and the main exit below still runs
            # (see ``_FailureRescues``).
            _rescues = _FailureRescues()
            # Breadth job 3 review fixes: a salt of element ions that the default
            # tier names by its whole-compound name ('lithium nitride': its
            # stoichiometric name 'trilithium nitride' is suppressed, and the
            # default last resort below supplies the name) keeps that name at the
            # best-effort tier. The best-effort rescues would otherwise join the
            # ions in the adduct notation of ('lithium(1+)--azanetriide
            # (3/1)'), which is the format of adducts, not of a salt,
            # ``_is_one_atom_ion_salt``). Only a name that reads back to the
            # input's full key is taken; otherwise the rescues run as before.
            if (is_failure_name(result) and self._general_fallback_unverified
                    and not self._disable_opsin_validity_gate
                    and is_top_level_naming() and _is_one_atom_ion_salt(smiles)):
                from .data import get_general_retained_name, get_unverified_retained_name
                _canon_salt = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)
                _salt_word = ((get_general_retained_name(_canon_salt)
                               if self._trivial_fallback else None)
                              or get_unverified_retained_name(_canon_salt))
                if (_salt_word
                        and _shipped_name_round_trip(_salt_word, smiles) == "verified"):
                    from .metrics.provenance import record_source
                    record_source("trivial_retained")
                    result = _salt_word
            # (opt-in): a candidate suppressed by a downstream gate
            # (/ vetoes) left only the failure sentinel; give the
            # engine its late, fully-gated shot before the failure ships.
            if is_failure_name(result):
                _rec = _rescues.run(self._try_general_engine_recovery, smiles)
                if _rec is not None and getattr(
                        self, "_last_recovery_floor_substitute", False):
                    _rec = _rescues.run(self._prefer_clean_over_floor_substitute,
                                        smiles, _rec, fallback=_rec)
                if _rec is not None:
                    result = _rec
            # tail #7: best-effort DEMOTE-SENIOR-GROUP rescue. A ring system
            # that names cleanly can be blocked because an ACYCLIC senior group
            # (an ester/acid on a substituent) is selected as the principal group
            # and its handler linearises or declines the ring parent (the
            # spiro-quinone alkaloid #7: a pendant acetate makes pg='ester', which
            # diverts the whole molecule from the spiro ring parent that would name
            # it as '...spiro[...]' with the acetate demoted to (acetyloxy)methyl).
            # Re-pick the principal group with the acyclic senior groups excluded
            # and re-name via the _principal_group_override seam; RT-verified, so
            # 0-wrong holds. Runs ONLY when we are about to ship a failure.
            # tier-policy: gated on `_general_fallback` (COMPLETE tier), not
            # `_general_fallback_unverified` (best-effort). This rescue POSITIVELY
            # RT-verifies its candidate internally (`_stereo_emit_decision` +
            # `_rt_match`), and at the COMPLETE tier `_stereo_emit_decision`
            # returns `(True, False)` only for a FULL-stereo name (else abstain),
            # so the RT gate is a FULL-InChIKey compare -- an unverified candidate
            # still returns None and the molecule still abstains. Best-effort is
            # unchanged (gfu=True implies gf=True); the default/PIN tier never
            # runs it (`_general_fallback` False) so PIN output is byte-identical.
            if (is_top_level_naming() and is_failure_name(result)
                    and self._general_fallback
                    and not self._disable_opsin_validity_gate):
                _dr = _rescues.run(self._try_demote_senior_group_rescue, smiles)
                if _dr is not None:
                    result = _dr
            # (-02): last-resort DECOMPOSITION before abstaining. A
            # NON-GENERAL handler (e.g. natural_products on a tropane scaffold)
            # can claim a molecule and emit a structure-DROPPING name (`tropane`,
            # dropping an ester) that then suppresses to the failure
            # sentinel — bypassing the GENERAL-only decomposition fallback inside
            # _name_impl. Give the decomposition engine a final, FULLY-GATED shot
            # here. This is purely additive: it runs ONLY when we are about to
            # ship a failure, so it can turn an 'unknown' into a valid name but
            # can NEVER alter a currently-passing name (0 gold regression by
            # construction). The candidate is routed through the SAME
            # validity gate (incl. the stereo-strip carve-out for names whose
            # only OPSIN-narrow layer is stereo, /, so 0-wrong holds:
            # a decomposition name whose constitution does not reproduce the
            # input is suppressed right back to the failure sentinel.
            if (is_top_level_naming() and is_failure_name(result)
                    and not self._disable_opsin_validity_gate):
                _dprobe = Chem.MolFromSmiles(smiles)
                if _dprobe is not None:
                    from .decomposition import try_decompose
                    _dname = _rescues.run(try_decompose, _dprobe, style=self.style)
                    if _dname and not is_failure_name(_dname):
                        _dgated = _rescues.run(
                            _final_opsin_validity_gate,
                            _dname, smiles, self._grammar_stats,
                            besteffort_unverified=self._general_fallback_unverified,
                            general_fallback_tier=self._general_fallback)
                        if not is_failure_name(_dgated):
                            result = _dgated
            # (offer-not-return, a project rule): a molecule whose SENIOR
            # (ranked[0]) parent dead-ends can be rescued by a JUNIOR pool
            # member. Runs ONLY on the ship-a-failure path (so no PIN-nameable
            # molecule is re-rooted -- they succeed at ranked[0] and never reach
            # here), and every junior candidate is RT-gated before adoption
            # (0-wrong). PIN/default output is byte-identical.
            # tier-policy: gated on `_general_fallback` (COMPLETE tier) -- the
            # candidate is FULL-InChIKey RT-verified at the complete tier exactly
            # as the demote-senior rescue above; the method's own internal guard
            # (below) is loosened to match. Default/PIN tier unaffected.
            if (is_top_level_naming() and is_failure_name(result)
                    and self._general_fallback
                    and not self._disable_opsin_validity_gate):
                _ap = _rescues.run(self._try_alternate_parent_rescue, smiles)
                if _ap is not None:
                    result = _ap
            # giants Engine 3: a retained natural-product PARENT HYDRIDE
            # (`ursane`, `hopane`, `cevane`,... -- the
            # stereoparents) IS the PIN, and `_final_opsin_validity_gate`
            # whitelists it (`np_stereoparent` carve-out) precisely because
            # OPSIN 2.9.0 cannot parse it. That is right for the PIN tiers, but
            # on the BEST-EFFORT path the whitelist costs a round-trip the
            # general engine can win. Offer its von-Baeyer systematic name and
            # adopt it ONLY when the full-InChI round-trip verifies. Measured
            # over all 53 such skeletons: 52 convert, 1 keeps the retained name
            # (`germacrane`, whose systematic form is a NON-RT
            # `...-1,7-dimethyl-4-(propan-2-yl)cyclodecane`) -- so an unconditional
            # downgrade would REGRESS it, which is why the RT gate is
            # load-bearing rather than belt-and-braces. (: the construction-
            # verified NAME_EXACT_NP_PARENTS are also exempted from the
            # BE-STRICT-unparseable suppression in _final_opsin_validity_gate, so a
            # KEPT retained parent like `germacrane` ships at best-effort instead of
            # abstaining.) Best-effort only
            # (`_general_fallback_unverified`), so PIN/default output is
            # byte-identical.
            if (is_top_level_naming() and self._general_fallback_unverified
                    and not self._disable_opsin_validity_gate
                    and not is_failure_name(result)):
                _np_sys = _rescues.run(self._try_np_systematic_downgrade, smiles, result)
                if _np_sys is not None:
                    result = _np_sys
            # CQ5 Task A (offer-not-return, a project rule): the committed primary
            # was voided by the RT/validity gate and every rescue above still
            # ships a failure. The general engine DID build a whole-graph
            # systematic candidate that OPSIN round-trips, but it was never
            # consulted: the late `_try_general_engine_recovery` at the top of
            # this failure block re-ran the engine INSIDE this name session,
            # where the best-effort contextvars are set + session_depth is
            # elevated, so its substituent recursion emits the SAME RT-failing
            # form the primary did (measured: witnesses in
            # internal notes + task-A-report.md; the good name
            # is produced only when the recursion runs in a clean, depth-0
            # context). Re-invoke the SAME RT-gated recovery in that clean
            # context and adopt its result IFF it is a real (RT-verified) name;
            # else keep the abstain. 0-wrong holds by the recovery's own OPSIN
            # round-trip gate (a project rule — a candidate that does not round-trip
            # returns None here). Best-effort only (`_general_fallback_unverified`)
            # and on the ship-a-failure path only, so no currently-shipping name
            # can change and PIN/complete output is byte-identical.
            if (is_top_level_naming() and is_failure_name(result)
                    and self._general_fallback_unverified
                    and not self._disable_opsin_validity_gate):
                _ft = _rescues.run(self._try_besteffort_clean_general_fallthrough, smiles)
                if _ft is not None and not is_failure_name(_ft):
                    result = _ft
            if _rescues.exhausted:
                # A rescue ran out of the hang budget (``_FailureRescues``). The
                # trip no longer reaches the ``_budget_scope`` boundary, so its last
                # resort is offered here, in the boundary's order: the bounded T4
                # rescue first (it arms and then disarms a budget of its own),
                # then the main exit below on a fresh budget -- the floor offer of
                # ``_finish`` included, as for any other failure.
                from .assembly.fragment_naming import rearm_hang_budgets
                if is_failure_name(result):
                    _pb = self._try_perf_budget_t4_rescue(smiles)
                    if _pb is not None and not is_failure_name(_pb):
                        result = _pb
                rearm_hang_budgets()
            # Task 1.9: PIN fails closed; --trivial falls back to a general-only
            # retained name when the systematic pipeline derived no PIN (and,
            # since wp7, an OPSIN-import trivial name without PIN evidence is
            # the DEFAULT last resort). It runs BEFORE the check below, so
            # that check sees what the caller would receive: '[H][H]' ships
            # 'molecular hydrogen' by default, and raise_on_limit must not raise
            # on it (TRIAGE C24; the raise used to precede this fallback).
            result = self._apply_trivial_fallback(result, smiles)
            #: post-failure limit (opt-in). If naming produced no real
            # name, map the failure to a named code. Keyed off an actual failure
            # so it can never fire on a successfully-named compound.
            if raise_on_limit and is_top_level_naming() and is_failure_name(result):
                _probe = Chem.MolFromSmiles(smiles)
                if _probe is not None:
                    raise classify_failure_limit(_probe, smiles=smiles)
            #: the MAIN exit. Everything that can rewrite the producer's
            # string -- stereo backstop, grammar repair, the OPSIN validity
            # gate, decomposition retry, trivial fallback -- has now run, so
            # this is the first point at which the proof can be asserted on
            # what the caller actually receives.
            return self._finish(result, smiles)
        finally:
            # M1: tear down the memo scope opened before this try (only the
            # outermost frame holds a real token; nested re-entries got None).
            _memo_pop(_memo_scope_token)
            # / /: unwind the ctx vars THIS frame published.
            self._reset_published_ctx(_ctx_tokens)
            end_naming_session()

    @staticmethod
    def _reset_published_ctx(tokens: list) -> None:
        """Reset, last-in first-out, the provenance ctx vars one ``name`` frame
        published (general_fallback_ctx, best_effort_ctx, allow_aromatic_general_ctx,
        full_coverage_ctx), and empty ``tokens``. Idempotent.

        The tokens live in the FRAME (a local list), not on the instance: a
        same-instance re-entry that also runs as a top-level naming (the isotope
        decorator names the stripped skeleton with ``namer.name(...)``) used to
        overwrite ``self._gf_ctx_token`` & co. with its own tokens, reset those, and
        set the attributes to None -- so the outer frame's tokens were never reset
        and the ctx vars stayed at the outer tier for every LATER naming in the
        process (TRIAGE C2). A leaked best-effort / allow_aromatic_general /
        full_coverage flag puts general-tier vocabulary (or arms D2) on a later
        default-tier naming -- an order dependence no single-molecule test shows."""
        while tokens:
            var, tok = tokens.pop()
            try:
                var.reset(tok)
            except Exception:
                pass

    def _strict_pin_twin_name(self, smiles: str) -> Optional[str]:
        """: name ``smiles`` with a STRICT sibling engine -- every construction
        param identical to this instance EXCEPT the four breadth flags, which are
        forced OFF. Used only by ``name_tiered`` to decide honest PIN status: a
        pin_verified name the strict path does not itself produce is not a
        certified PIN (see the demotion at its call site).

        The twin is built once and cached (deterministic, so a per-instance reuse
        is safe). It cannot recurse into ``name_tiered`` -- it calls ``name`` -- so
        there is no re-entrancy. It inherits the OPSIN-gate setting so its
        acceptance bar matches this instance's (finding F4).
        """
        tw = getattr(self, "_pin_twin", None)
        if tw is None:
            tw = Orthonym(
                style=self.style,
                _disable_grammar_validation=(self._grammar is None),
                _disable_opsin_validity_gate=self._disable_opsin_validity_gate,
                enable_triviality_controller=self._enable_triviality_controller,
                enable_group_splitting=self._enable_group_splitting,
                trivial_fallback=self._trivial_fallback,
                general_fallback=False,
                general_fallback_unverified=False,
                allow_aromatic_general=False,
                full_coverage=False,
                _seed_excluded_dispatch_classes=self._seed_excluded_dispatch_classes,
                binding_proof=self._binding_proof,
            )
            self._pin_twin = tw
        # Branch review fixes: the twin is the STRICT path, so it never runs the
        # PIN tier's promotion re-run (``_name_with_pin_promotion``). The twin is a
        # default-configuration instance at name-scope depth 1, so without this
        # guard it re-ran with the promoted best-effort producers and reproduced
        # the breadth name it exists to tell apart ('...-(2,6-dioxo-1H-pyrimidin-3-
        # yl)propanoic acid' came out pin_verified at best-effort).
        _wrapped = _PIN_PROMOTION_WRAPPED.set(True)
        _policy_off = _DEFAULT_TIER_POLICY_OFF.set(True)
        try:
            return tw.name(smiles)
        except Exception:
            # A twin failure must never break tier reporting; treat as "strict
            # did not produce this name" (conservative -> demote), never a crash.
            return None
        finally:
            _DEFAULT_TIER_POLICY_OFF.reset(_policy_off)
            _PIN_PROMOTION_WRAPPED.reset(_wrapped)

    def name_tiered(self, smiles: str) -> dict:
        """Name one molecule and say how the name was made and checked.

        This is the row the command line prints with ``--provenance``.

        Parameters
        ----------
        smiles: str
            The structure, as a SMILES string.

        Returns
        -------
        dict
            ``name``
                The name, or a label when there is none.
            ``tier``
                How the name was built: ``pin_verified`` (the strict path for the
                Preferred IUPAC Name built it, certified it and OPSIN read it back),
                ``pin_unverified`` (a name in preferred-name form that OPSIN read
                back, but whose preferred status is not certified: a producer outside
                the strict path built it or a part of it),
                ``systematic_verified`` (a checked systematic name that is not the
                preferred name, from the general engine, a table of retained names,
                or the strict path when the name contains a part the engine records
                as not the preferred form), ``best_effort`` (the last-resort
                producers, or a name whose own string no round trip confirmed) or
                ``abstain`` (no name).
            ``is_pin``
                True only for a certified Preferred IUPAC Name.
            ``source``
                Which part of the engine produced the name, for example
                ``pin_path``, ``general_engine``, ``trivial_retained`` or
                ``abstain``.
            ``opsin``
                What the OPSIN check found: ``verified``,
                ``verified_constitution_only``, ``unverified`` or ``n/a``.
            ``gates_passed``
                The checks this name passed, for example ``self_consistency``
                (OPSIN read the name back to your structure) and ``atom_coverage``.
            ``gate_outcome``
                What the final OPSIN check did for this name, for example
                ``self_consistency_verified``, ``suppressed`` or ``not_run``, or
                ``carveout:<class>`` for a class OPSIN cannot read.
            ``formula``
                The molecular formula, given when there is no name.
            ``limit_code``
                The reason code when there is no name, for example
                ``UNSUPPORTED_ELEMENT``, or ``NO_VERIFIED_PIN`` when the default
                tier built a name that is not a verified PIN (a wider tier returns
                it).
            ``stereo_unexpressed``
                True when a stereocentre of the input is not stated in the name.
            ``suffix_free_prefix_name``
                True when the name states the principal characteristic group as a
                prefix with no suffix.
            ``prefix_order_fallback``
                True when the name cites its substituent prefixes out of the
                alphanumerical order because the round trip of the ordered
                spelling fails at the stereo layer only (a stereocentre read
                differently by the SMILES hand-off, not by the name); such a name
                is best_effort, never a PIN.
            ``verified``
                ``opsin`` (OPSIN read the whole name back to the same molecule),
                ``opsin_constitution`` (read back with the same constitution; the
                stereodescriptors were not confirmed by OPSIN), ``identity`` (a name
                from an exact-match list: a metal-complex name found by the input's
                exact InChIKey, or a natural-product parent name found by its exact
                structure; OPSIN cannot read these names) or ``unverified`` (no
                read-back recorded).

        Examples
        --------
        >>> from orthonym import Orthonym
        >>> row = Orthonym.name_tiered("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
        >>> row["name"], row["tier"], row["verified"]
        ('1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione', 'pin_verified', 'opsin')
        """
        # B2: open ONE memo scope around BOTH the primary name and the strict PIN
        # twin (_strict_pin_twin_name -> tw.name), so the twin reuses this run's
        # breadth-INDEPENDENT memo entries instead of recomputing them. The two
        # breadth-SENSITIVE namespaces (fused_core, name_substituent) carry the four
        # breadth flags in their keys (a lever), so the twin's flags-OFF lookups miss
        # the primary's flags-ON entries and recompute at the correct configuration.
        # push_scope returns None inside name's own push (a nested no-op), so both
        # engines share THIS scope; only this frame tears it down.
        from .assembly.memo import pop_scope as _memo_pop, push_scope as _memo_push
        # Lone-pair centre written first: the row names (and derives its tier and
        # formula from) the standard spelling of the input when RDKit reads a
        # lone-pair centre of it unlike the standard reading; see
        # _lone_pair_standard_spelling. Any other input is unchanged.
        smiles, _lp_opened = _lone_pair_input_enter(smiles)
        try:
            _tier_scope = _memo_push()
            try:
                return self._name_tiered_impl(smiles)
            finally:
                _memo_pop(_tier_scope)
        finally:
            _lone_pair_input_exit(_lp_opened)

    def _name_tiered_impl(self, smiles: str) -> dict:
        from .metrics.provenance import clear_provenance, get_provenance
        clear_provenance()
        self._default_tier_row = None
        name = self.name(smiles)
        # The default tier's emission rule already derived this call's row (the
        # emitted name's, or the decline's): the same row, not a second derivation.
        stash = getattr(self, "_default_tier_row", None)
        self._default_tier_row = None
        if stash is not None and stash[0] == name:
            return dict(stash[1])
        return self._tier_row_and_pin_form(smiles, name, get_provenance())[0]

    def _tier_row_and_pin_form(self, smiles: str, name, prov: dict):
        """The provenance row of ``name`` (see:meth:`name_tiered`), and whether the
        name is in strict-path PIN form: its tier is ``pin_verified`` before the
        label is lowered for a missing full OPSIN read-back (a grammar carve-out,
        the constitution-only stereo branch). The default tier's emission rule
        reads the second value (:func:`_default_tier_emits`)."""
        from .metrics import provenance as _pv
        source = prov["source"] or "pin_path"
        # T1: what the gate DID for THIS name, not whether a jar exists.
        # The old `gate_active = (not disabled and jar_present)` stamped
        # `opsin="verified"` / `` on every PIN-path emission on a
        # machine with a jar, including the gate's ten by-design carve-outs and
        # its stereo-stripped branch — three OPSIN-UNPARSEABLE names reported
        # (FINDINGS.md). `resolve_gate_outcome` additionally denies
        # an outcome recorded for a DIFFERENT string than the one shipped.
        if self._disable_opsin_validity_gate:
            # The instance flag short-circuits the gate at its call sites, so
            # nothing is recorded; the honest label is that it was disabled.
            gate_outcome = _pv.GATE_OUTCOME_DISABLED
        else:
            gate_outcome = _pv.resolve_gate_outcome(
                prov["gate_outcome"], prov["gate_outcome_name"], name)
        gate_opsin_label = _pv.opsin_label_for_gate_outcome(gate_outcome)
        formula = None
        limit_code = None
        # T6.4: only surface stereo_unexpressed when a general-engine
        # name actually shipped (the contextvar could be set by an emission the
        # downstream constitutional gate later suppressed -> failure row).
        stereo_unexpressed = False
        #: same guard, same reason -- the contextvar can be set by an
        # assembly-tier candidate that the constitutional gate then SUPPRESSED,
        # and tagging an abstention row as an ill-formed emission would corrupt
        # the very census this field exists to make possible.
        suffix_free_prefix_name = False
        # Tier labels -- paper semantics: the tier a name earns once a round trip
        # of THIS string has passed (set by the branches whose label depends on
        # verification: the general engine -> systematic_verified, a PIN-path name
        # carrying a non-PIN part -> pin_unverified). The shipped-name check below
        # can verify a name after its label was derived; the label then follows.
        # None where the label does not depend on a round trip (a PIN, a trivial
        # table name, the last-resort floor).
        _tier_if_verified = None
        # A name of the exact-match metal-complex list (D1 table, the input's
        # exact InChIKey) is a ChEBI name taken verbatim: labelled by the path that
        # returned it, as in the paper's measured run (user decision 2026-09-30),
        # so the name-scoped non-PIN records, which read the spelling of names the
        # engine builds, are not read for it ('kappa4' is no non-PIN token); the
        # whole-call records are.
        _is_coord_list = bool(name) and not is_failure_name(name) and \
            _coordination_list_name(smiles, name)
        _non_pin_scope = None if _is_coord_list else name
        # A name of the exact-match natural-product parent list (NAME_EXACT_NP_PARENTS,
        # the input's exact canonical SMILES; 'germacrane') likewise: the paper's
        # measured run labelled it pin_verified (user decision 2026-09-30, (c) covers
        # both list classes). No non-PIN record applies to a verbatim list name, the
        # whole-call ones neither (they describe the candidates of the call, e.g. the
        # systematic name the general tiers tried first).
        _is_np_list = bool(name) and not is_failure_name(name) and \
            _np_exact_list_name(smiles) == name
        if _is_np_list:
            _non_pin_scope = None
            prov = dict(prov, general_ring_prefix=False, pin_promotion_rerun=False)
        if not name or is_failure_name(name):
            tier, is_pin, opsin = ABSTAIN, False, "n/a"
            source = prov["source"] or "abstain"
            #: descriptive last-resort row — the abstain residual is
            # LABELED (formula + classified reason), never a bare unknown.
            try:
                _mol = Chem.MolFromSmiles(smiles)
                if _mol is not None:
                    from rdkit.Chem import rdMolDescriptors
                    formula = rdMolDescriptors.CalcMolFormula(_mol)
                    limit_code = classify_failure_limit(
                        _mol, smiles=smiles).code
            except Exception:
                pass
            # Composer1 Task 5: best-effort clean-abstain contract. In the
            # general-fallback tier the honest outcome for a molecule neither the
            # PIN path nor the general engine can name is a CLEAN abstain — no
            # name. Surfacing the PIN always-emit descriptive residual here (e.g.
            # a partial '…-unknown-…' string, or 'unknown organic compound') would
            # masquerade a non-name as a name for an abstain row that is already fully
            # LABELED by (tier=abstain, source=abstain, limit_code, formula). Scoped to
            # ``self._general_fallback`` so the PIN-default path (and its always-
            # emit ``name``) is byte-identical; ``name`` itself is untouched.
            if self._general_fallback:
                name = None
        elif source == "general_engine":
            # The general engine runs its OWN OPSIN round-trip against the
            # input structure (`_rt_match`,:2921) and publishes the verdict as
            # prov["opsin"], which is always a non-empty string — so this
            # branch never consulted jar presence and is unchanged.
            # The inline engine lane records no engine verdict (prov["opsin"]
            # None); the validity gate's outcome for this string then decides.
            opsin = prov["opsin"] or gate_opsin_label
            tier = SYSTEMATIC_VERIFIED if opsin == "verified" else BEST_EFFORT
            _tier_if_verified = SYSTEMATIC_VERIFIED
            is_pin = False
            stereo_unexpressed = bool(prov.get("stereo_unexpressed"))
            suffix_free_prefix_name = bool(prov.get("suffix_free_prefix_name"))
        elif source == "trivial_retained":
            # R68 / critic-2: 'systematic_verified' only when a round trip passed;
            # the honesty rule at the end of this method demotes an unverified one
            # to best_effort (after the general tiers' own check has run).
            tier, is_pin = SYSTEMATIC_VERIFIED, False
            opsin = gate_opsin_label
        elif _pv.name_carries_non_pin_part(prov, _non_pin_scope):
            # -T1c: a composer name carrying a ring substituent prefix only
            # the GENERAL tier could build (a systematic replacement / von Baeyer
            # substituent form). Valid, but not PREFERRED -- the ring PIN may be a
            # retained name -- so it must not ship as pin_verified/is_pin. The same
            # demotion applies to a name that contains a recorded non-PIN fragment
            # (``record_non_pin_fragment``, e.g. a carbon-substituted
            # '...azaniumyl' prefix, a peptide, a von Baeyer name of an
            # ortho-fused system).
            # Tier labels -- paper semantics (Methods, "Tiers"; user decision
            # 2026-09-30): a name that contains a name-scoped non-PIN record (a
            # recorded non-PIN fragment or label, or non-PIN vocabulary) is one the
            # code knows is not the PIN, so a verified one is "a correct systematic
            # name that is not the PIN": systematic_verified. A name that carries
            # only a whole-call record -- a ring substituent prefix only the general
            # tier could build (``general_ring_prefix``) or the PIN tier's promotion
            # re-run (``pin_promotion_rerun``) -- is "a name in PIN form that only a
            # breadth producer built, so its preferred status is not certified":
            # pin_unverified. Without a verified round trip either is best_effort.
            opsin = gate_opsin_label
            _known_non_pin = _pv.name_carries_non_pin_part(
                dict(prov, pin_promotion_rerun=False, general_ring_prefix=False,
                     uncertified_pin_names=()),
                _non_pin_scope)
            _tier_if_verified = (SYSTEMATIC_VERIFIED if _known_non_pin
                                 else PIN_UNVERIFIED)
            tier = _tier_if_verified if opsin == "verified" else BEST_EFFORT
            is_pin = False
        else:
            tier, is_pin = PIN_VERIFIED, True
            opsin = gate_opsin_label
        # a phase cleanup T1: the branches above derive tier/is_pin/
        # source from the provenance CONTEXTVAR, which reflects whichever
        # producer ran LAST -- correct when `self._offers` holds a single
        # primary offer (the contextvar and that offer are built from the
        # identical snapshot, so nothing changes below), but WRONG when
        # `_select_rt_passing_offer_name` picked a DIFFERENT, later-appended
        # offer over that primary (e.g. the systematic floor beating a
        # stereo-wrong primary): the contextvar still describes the LOSING
        # primary. Re-derive from the WINNING `Offer` itself whenever one won
        # and its `.name` is exactly what shipped (`_last_selected_offer` is
        # `None` when `select_rt_passing` found no complete+RT-passing offer,
        # in which case `current_name` ships unchanged and there is nothing
        # to override). Guarded off the abstain branch too, though that is
        # belt-and-braces: a winning offer can never be a failure-name
        # sentinel (both offer-construction sites in `_finish` already
        # refuse to append one).
        #
        # Tier labels -- paper semantics (2026-09-28): when the PRIMARY offer won
        # (same source as the contextvar, i.e. the snapshot the branches above
        # read), its labels are NOT copied. The primary Offer's tier is a ranking
        # key built in `_finish` from the general engine's own round-trip value
        # (prov["opsin"]) alone, which the PIN path and the inline engine lane
        # never record, so it read best_effort for every verified general-engine
        # name of the inline lane (e.g. '12-(bicyclo[2.2.2]octan-1-yl)-6,9-
        # dimethyltricyclo[7.4.0.0^2,6]tridecane', gate verified) and for
        # every verified PIN-path name carrying a non-PIN part. The branches above
        # derive the same source/is_pin from the same snapshot, and the tier from
        # the verification of the shipped string. A DIFFERENT winner (the
        # last-resort floor, `source` 't4_floor', tier best_effort) still
        # supplies its own labels.
        _offer_winner = getattr(self, "_last_selected_offer", None)
        if (_offer_winner is not None and _offer_winner.name == name
                and tier != ABSTAIN and _offer_winner.source != source):
            source = _offer_winner.source
            is_pin = _offer_winner.is_pin
            tier = _offer_winner.tier
            _tier_if_verified = None
        #: honest-PIN demotion (the β-carotene "not a verified PIN" fix).
        # A pin_verified name that the STRICT PIN path (no breadth flags) does NOT
        # itself produce is not a certified PIN -- it exists only because a breadth
        # flag (general_fallback / general_fallback_unverified / allow_aromatic_
        # general / full_coverage) enabled a best-effort producer (e.g. the
        # allow_mancude ring-substituent path that renders the β-carotene polyene;
        # the strict path abstains there). Its constitution round-trips, but its
        # PIN PREFERENCE is uncertified -> demote to PIN_UNVERIFIED, is_pin=False.
        #
        # The discriminator is a strict TWIN engine (identical construction minus
        # the breadth flags), because the enabling gate is best_effort_ctx /
        # general_fallback_ctx read at MANY sites through the substituent recursion
        # -- no single producer frame captures it (verified: the universal-namer
        # and allow_mancude sites are both off-path or non-discriminating for this
        # molecule). Gated on the FINAL tier AFTER the offer override (finding F7:
        # an Offer default is PIN_VERIFIED, so the else-branch alone would miss the
        # offer-winner path -- the measured case). Runs ONLY on a breadth-flagged
        # instance, so the default/PIN engine is byte-identical and pays nothing.
        # The twin inherits this instance's OPSIN-gate setting (finding F4: a
        # gate-mismatched twin could false-keep). Necessary-not-sufficient: catches
        # the strict-vs-breadth divergence class, not the ~517 strict-path
        # non-PINs (a separate axis). See CAROTENOID-B-DESIGN-2026-09-09.md.
        if (tier == PIN_VERIFIED
                and (self._general_fallback or self._general_fallback_unverified
                     or self._allow_aromatic_general or self._full_coverage)):
            _strict_name = self._strict_pin_twin_name(smiles)
            if _strict_name != name:
                tier = PIN_UNVERIFIED
                is_pin = False
        # "Preferred IUPAC multiplicative names" (the Blue Book):
        # identical parent structures, identically substituted at identical locants,
        # take the multiplicative name as the PIN (:23180-23184); '4,4′-methylene
        # diphenol (PIN) (1) 4-[(4-hydroxyphenyl)methyl]phenol' (:26874). When
        # ``multiplicative_pin_expected`` holds and the name did not come from the
        # multiplicative producer (dispatch class MULTIPLICATIVE), the name is a
        # correct systematic name that is not the PIN, on every route that built it
        # (the first run included): systematic_verified, is_pin False. The default
        # tier then declines it; the wider tiers keep it.
        # The routes (carbohydrates, natural products, peptides, lipids) name
        # by their own nomenclature, not by parent selection; their labels are
        # the natural-product decision's (plan Task 32), so they are not read here.
        if tier == PIN_VERIFIED and name and not is_failure_name(name):
            _mult_cls = getattr(self, "_last_dispatch_class", None)
            if getattr(_mult_cls, "name", None) not in (
                    "MULTIPLICATIVE", "CARBOHYDRATE_LOOKUP", "NATURAL_PRODUCT",
                    "PEPTIDE", "LIPID"):
                from .rules.pin_vocabulary import multiplicative_pin_expected
                _mol_mult = Chem.MolFromSmiles(smiles)
                if _mol_mult is not None and multiplicative_pin_expected(_mol_mult):
                    tier = SYSTEMATIC_VERIFIED
                    is_pin = False
        # wp7 (verification panel RISK 2): no oracle ran for this name. The
        # gate records UNAVAILABLE when it cannot consult OPSIN -- the opt-in
        # reduced mode without a jar (ORTHONYM_ALLOW_REDUCED, jars.py); a
        # transient OPSIN failure with the jar present now suppresses (TRIAGE g7
        # C01) -- and the name then ships unverified. A pin_verified label claims a verification that did not
        # happen ('methanaminium' for the radical cation C[NH2+] shipped so), so
        # such a name is at most pin_unverified. The PIN tier's names and the
        # breadth are unchanged; only the label is honest.
        _unavailable_demoted = False
        if tier == PIN_VERIFIED and gate_outcome == _pv.GATE_OUTCOME_UNAVAILABLE:
            tier = PIN_UNVERIFIED
            is_pin = False
            _unavailable_demoted = True
        # Claims conformance (2026-09-27): at the general tiers every shown name
        # has passed the round trip, apart from the exact-match list names
        # (NAME_EXACT natural-product parents, coordination retained names). The
        # labels above come from the gate outcome recorded for THIS string, and
        # some winners never had one: an offer that `_select_rt_passing_offer_name`
        # picked after the gate had suppressed another string (its `_offer_rt_ok`
        # full-key check is not recorded, so the name read 'bypassed' or
        # 'not_run'), or a general-engine name whose own check was recorded as
        # 'unverified' although the string round-trips. Such a name is checked
        # here, without any fail-open branch: when OPSIN reads it back to the
        # input's full InChIKey it is labelled verified; otherwise it is withdrawn
        # and the row abstains with its limit code (a name OPSIN cannot read, such
        # as one with a lowercase r/s descriptor, is never shown at these tiers).
        # Reduced mode (no jar, ORTHONYM_ALLOW_REDUCED) and a disabled gate keep
        # their documented labels. The default (PIN) tier is unchanged.
        if (tier != ABSTAIN and opsin != "verified" and name
                and self._general_fallback
                and not self._disable_opsin_validity_gate
                and not _DISABLE_VALIDITY_GATE
                and _validity_gate_jar_present()
                and not _is_exact_match_list_name(smiles, name)):
            if _shipped_name_round_trip(name, smiles) == "verified":
                opsin = "verified"
                gate_outcome = _pv.GATE_OUTCOME_FULL_KEY_VERIFIED
            else:
                logger.info("name_tiered withdrew %r: no full-key round trip", name)
                name = None
                tier, is_pin, opsin = ABSTAIN, False, "n/a"
                source = "abstain"
                gate_outcome = _pv.GATE_OUTCOME_SUPPRESSED
                stereo_unexpressed = False
                suffix_free_prefix_name = False
                try:
                    _mol = Chem.MolFromSmiles(smiles)
                    if _mol is not None:
                        from rdkit.Chem import rdMolDescriptors
                        formula = rdMolDescriptors.CalcMolFormula(_mol)
                        limit_code = classify_failure_limit(
                            _mol, smiles=smiles).code
                except Exception:
                    pass
        # Tier labels -- paper semantics (2026-09-28): the check above verified a
        # name whose label was derived before it ran (an offer winner the gate
        # never judged, a general-engine name whose own check read 'unverified').
        # A verified name keeps the tier it earns verified -- systematic_verified
        # for the general engine, pin_unverified for a PIN-path name with a
        # non-PIN part -- instead of the best_effort it read while unverified.
        if (tier == BEST_EFFORT and opsin == "verified"
                and _tier_if_verified is not None):
            tier = _tier_if_verified
        # (The four no-PIN-status demotions below run after the shipped-name check
        # so that they read the final verification of the shipped string.)
        # wp7 (verification panel NIT): the Blue Book gives no PIN for these
        # elements' compounds. the Blue Book "Names of organic compounds based
        # on aluminium, gallium, indium, and thallium are not followed by the
        # parenthetical abbreviation (PIN), because the decision to choose between
        # a name based on organic or inorganic principles has not yet been
        # reached";:2062 names based on alumane, gallane, indigane and thallane
        # "currently do not have PIN status". 'dimethylaluminum' for C[Al]C shipped
        # pin_verified. Such a name is a verified systematic name at most.
        if tier in (PIN_VERIFIED, PIN_UNVERIFIED) and _has_no_pin_status_element(smiles):
            tier = SYSTEMATIC_VERIFIED if opsin == "verified" else BEST_EFFORT
            is_pin = False
        # TRIAGE j12 finding 6: likewise (the Blue Book) notes no PIN
        # for organometallic compounds of the Group 1-12 metals, except the ocenes
        # ('tert-butyllithium', 'methylmagnesium bromide', 'diethylzinc' shipped
        # pin_verified). Same demotion; the name is unchanged.
        # Not for a name of the exact-match metal-complex list (heme, chlorophyll,
        # cobalamin, siroheme, coenzyme F430): its label is the one the naming
        # path gives it (user decision 2026-09-30, the paper's measured labels).
        if (tier in (PIN_VERIFIED, PIN_UNVERIFIED)
                and _is_p69_organometallic_without_pin(smiles, name)
                and not _is_coord_list):
            tier = SYSTEMATIC_VERIFIED if opsin == "verified" else BEST_EFFORT
            is_pin = False
        # Breadth Job 2: a mixed organic-inorganic adduct of a metal has no PIN
        # either, the Blue Book). Same demotion; the name is unchanged.
        if (tier in (PIN_VERIFIED, PIN_UNVERIFIED)
                and _is_metal_adduct_without_pin(smiles, name)):
            tier = SYSTEMATIC_VERIFIED if opsin == "verified" else BEST_EFFORT
            is_pin = False
        # A carbon-free compound ('sodium chloride', 'sulfuric acid', 'ammonia')
        # keeps the label its naming path gives it (user decision 2026-09-30: the
        # labels of the paper's measured run, 'sodium chloride' pin_verified); the
        # elements above keep their own demotion.
        # Review fixes (a performance pass): a salt of element ions with a metal cation and a
        # carbon anion ('dimagnesium methanetetraide', a metal carbide):
        # (the Blue Book) "preferred IUPAC names have not yet been
        # determined for inorganic components"; (:39735) gives no PIN for
        # the compounds of the Group 1-12 metals. Same demotion; the name is
        # unchanged. A carbon-free salt of element ions keeps its label as above.
        if (tier in (PIN_VERIFIED, PIN_UNVERIFIED)
                and _is_one_atom_ion_salt(smiles) and _has_metal_atom(smiles)
                and not _is_carbon_free_compound(smiles)):
            tier = SYSTEMATIC_VERIFIED if opsin == "verified" else BEST_EFFORT
            is_pin = False
        gates = []
        if source == "general_engine":
            gates.append("atom_coverage")
            # E1-path verification is the engine's OWN round-trip check
            # (prov["opsin"]), not a gate outcome, so its token is unchanged.
            if opsin == "verified":
                gates.append("self_consistency")
        elif opsin != "n/a":
            # pin_verified / systematic_verified-retained: the token is
            # whatever the gate actually earned — `` only for a
            # full-name "ok",
            # `(constitution)` for the stereo-stripped verdict, and
            # nothing at all for a carve-out, a bypass or a fail-OPEN.
            _token = _pv.gate_token_for_gate_outcome(gate_outcome)
            if _token is not None:
                gates.append(_token)
        #: honest verification-provenance label for the full-coverage
        # tier. Derived from what ALREADY shipped, deny-by-default "unverified":
        # "identity" -> a D1 exact-InChIKey coordination retained-name hit
        # (the input matches a curated table entry exactly; no
        # OPSIN read-back)
        # "opsin" -> OPSIN read the FULL name back to the input (,
        # the engine's own RT, a full-key recomposition or the
        # full-key round trip): opsin == "verified"
        # "opsin_constitution" -> OPSIN read back only the stereo-STRIPPED name
        # (opsin == "verified_constitution_only"): the constitution
        # is confirmed by OPSIN, the descriptors by the CIP check
        # of _stereo_descriptors_verified (R68: this used to say
        # "opsin" although OPSIN rejects the full name)
        # "reconstructor" -> (SP5.6, if built) structurally reconstructed
        # "unverified" -> emitted with no passing oracle (/ D2 additive,
        # a grammar carve-out), or an abstain row. Never
        # claims a verification the emission did not earn
        # (mirrors the allowlist discipline of
        # ``opsin_label_for_gate_outcome``).
        verified = "unverified"
        if name and not is_failure_name(name):
            if _is_coord_list or _is_np_list:
                verified = "identity"
            elif opsin == "verified":
                verified = "opsin"
            elif opsin == "verified_constitution_only":
                verified = "opsin_constitution"
        # A name from the exact-match coordination list (D1: heme, chlorophyll,
        # cobalamin, siroheme, coenzyme F430; checked by exact InChIKey identity,
        # 'identity') keeps the label its naming path gives it (user decision
        # 2026-09-30, the paper's measured labels: 35 pin_verified and 3
        # best_effort on ChEBI).
        # (a strict-path name lowered only because OPSIN could not be consulted, the
        # opt-in reduced mode, is still in PIN form)
        _pin_form = tier == PIN_VERIFIED or (tier == PIN_UNVERIFIED and _unavailable_demoted)
        # R63 / R68 (claims conformance, 2026-09-27): a tier whose name says
        # 'verified' is earned only by a full round trip (verified 'opsin') or the
        # documented identity table. The grammar carve-outs (OPSIN has no grammar
        # for their class), the constitution-only stereo branch, and a name no
        # check covers keep shipping where they shipped, one label down:
        # pin_verified -> pin_unverified (not a certified PIN, as for the other
        # pin_unverified demotions), systematic_verified -> best_effort. Labels
        # only: the name and the gate outcome are unchanged.
        if tier == PIN_VERIFIED and verified not in ("opsin", "identity"):
            tier, is_pin = PIN_UNVERIFIED, False
        elif tier == SYSTEMATIC_VERIFIED and verified not in ("opsin", "identity"):
            tier = BEST_EFFORT
        # D3: the best-effort last resort cited the parent's prefixes out of the
        # order (``_maybe_prefix_order_fallback``): never a PIN, best_effort.
        prefix_order_fallback = bool(
            name and not is_failure_name(name)
            and getattr(self, "_prefix_order_fallback_name", None) == name)
        if prefix_order_fallback:
            tier, is_pin = BEST_EFFORT, False
            _pin_form = False
        return ({"name": name, "tier": tier, "is_pin": is_pin,
                 "source": source, "opsin": opsin, "gates_passed": gates,
                 "gate_outcome": gate_outcome,
                 "formula": formula, "limit_code": limit_code,
                 "stereo_unexpressed": stereo_unexpressed,
                 "suffix_free_prefix_name": suffix_free_prefix_name,
                 "prefix_order_fallback": prefix_order_fallback,
                 "verified": verified}, _pin_form)

    def _retained_structural_preference(self, mol) -> Optional[str]:
        """ (AutoNom A3): retained/fusion structural-recognizer
        preference -- a PIN-QUALITY GUARDRAIL, not a naming path.

        Runs the EXISTING retained-name recognizers on the WHOLE molecule
        GRAPH (never string surgery, never a new catalog) so the general
        engine's late-recovery can prefer a clean retained/fused-heterocycle
        name over an ugly von-Baeyer/replacement emission, when one applies.
        Tried in order (first hit wins; each is an exact whole-molecule
        match -- none of these know about substituents, so this can only
        ever match a BARE ring system):

          1. ``data.fused_heterocycles.get_fused_heterocycle_name`` -- the
             curated fused-ring catalog (quinazoline, indole, purine,...).
          2. ``data.retained_names.get_retained_name`` on the whole-molecule
             canonical SMILES -- simple retained trivial names.
          3. The ``heterocycles.py`` monocyclic retained lookup (reuses
             ``get_ring_canonical_smiles`` + ``get_retained_name``),
             restricted to a molecule that IS a single ring covering every
             heavy atom (no exocyclic substituents) so the name recovered
             cannot silently omit part of the structure.

        (1) is tried before (2) deliberately: a reproduce-first data
        audit found >=2 canonical-SMILES keys shared between the two tables
        where ``get_retained_name`` carries a WRONG (non-isomeric) name for
        a fused ring system that ``FUSED_HETEROCYCLE_DATA`` already has
        correct -- e.g. benzo[f]quinoline's canonical SMILES also keys
        ``ALL_RETAINED_NAMES`` to ``benzo[h]quinoline`` (a different
        connectivity; OPSIN-verified NOT to round-trip), and quinolizidine's
        key resolves to ``decahydroisoquinoline`` (also a different
        connectivity). Checking (1) first means the caller's gate
        never even has to catch these -- and it still would, since this
        method never itself verifies its answer.

        Returns the candidate name string or ``None`` (no recognizer
        matched). The caller re-verifies whatever this returns through the
        SAME OPSIN gate as the general name and keeps the general
        name if the retained candidate does not independently round-trip
        (fail-closed; e.g. a lambda-convention ring name that is only valid
        embedded in a larger spiro PIN, not standalone).
        """
        try:
            from .data.fused_heterocycles import get_fused_heterocycle_name
            fused = get_fused_heterocycle_name(mol)
            if fused and fused[0]:
                return fused[0]
            from .data.retained_names import get_retained_name
            canonical = Chem.MolToSmiles(mol, canonical=True)
            retained = get_retained_name(canonical)
            if retained:
                return retained
            ring_info = mol.GetRingInfo()
            atom_rings = ring_info.AtomRings()
            if len(atom_rings) == 1 and len(atom_rings[0]) == mol.GetNumAtoms():
                from .rules.heterocycles import get_ring_canonical_smiles
                ring_smiles = get_ring_canonical_smiles(mol, atom_rings[0])
                mono_retained = get_retained_name(ring_smiles)
                if mono_retained:
                    return mono_retained
        except Exception as e:  # fail-closed: a recognizer bug must never
            # turn a preference lookup into a crash; the caller keeps
            # whatever general-engine candidate it already had.
            logger.info(
                "retained structural preference error (kept general): %s", e)
        return None

    def _stereo_emit_decision(self, mol, cand, smiles):
        """ T6.2 / Phase-1 reclaim: ONE shared stereo-policy decision for
        the general-engine emission sites (the inline G1 fallback in ``_name_impl``
        and the recovery/rescue ladder in ``_try_general_engine_recovery`` & the
        alternate/demote/ rescues).

        Returns the 3-tuple ``(permitted, flagged, name_out)``, where ``name_out``
        is the name the caller must EMIT and RT-verify (possibly the input's stereo
        COMPOSED onto ``cand``):
          * ``(True, False, cand)`` — count matches -> offer AS-IS, unflagged.
          * ``(True, False, composed)`` — Phase-1 reclaim: the count-mismatch
            candidate is either already-complete-but-miscounted (symmetric ``bis``/
            multiplied names) or a stereo-OMISSION whose descriptor was composed
            back on via the RT-gated re-anchor. Emitted UNFLAGGED so the
            caller's exact-isomeric ``_rt_match(flagged=False)`` (CanonSmiles ==
            CanonSmiles AND full-InChIKey, InChI-STRONGER than a bare InChIKey) is
            the emission authority. 0-wrong ABSOLUTE.
          * ``(True, True, cand)`` — jar-dead measurement escape hatch only
            (``ORTHONYM_BE_STRIP_STEREO=1``): ship CONSTITUTION-ONLY, flagged.
          * ``(False, False, None)`` — ABSTAIN: a genuine stereo CONFLICT, an
            unexpressible element (axial), a bare RELATIVE descriptor (Adv1), a
            wrong-constitution candidate, or a jar-dead reclaim it cannot verify.

        Phase-1 replaces the raw count identity as the ABSTAIN authority. A count
        ``count_expressed == count_defined`` fails in BOTH directions (under-counts
        symmetric/multiplied names; and it can never be TRUE for a name that dropped
        a descriptor) and is blind to CORRECTNESS. So the count survives only as a
        cheap FAST-ACCEPT (STEP 1); the completeness-and-correctness authority moves
        to the exact-isomeric ``_rt_match`` at each caller — mirroring v47
        ``_ring_cistrans_is_complete`` replacing a proxy count with an exact
        predicate. Governing rule: (the Blue Book) — "In
        preferred IUPAC names, stereodescriptors, preceded by a locant, must be
        cited to specify each stereogenic unit"; an exocyclic ylidene C=C is not a
         omission class, so its descriptor is REQUIRED.
        """
        from .rules.stereochemistry import general_engine_stereo_complete

        # STEP 0 — RELATIVE-DESCRIPTOR GUARD (Adv1). A bare relative token
        # (cis/trans/rel/rac/syn/anti/R*/S*/(RS)/±) is 50/50-ambiguous on a true
        # enantiomeric pair: OPSIN picks one enantiomer, so BOTH the full-InChIKey
        # and the exact-CanonSmiles compare false-confirm it. Never permit or
        # compose such a candidate; abstain. NON-REGRESSING: the count grammar
        # (_STEREO_EMBEDDED_RE = [RSEZrsez] only) already scores cis/trans as 0, so
        # today's count veto already abstains these — the guard only makes the
        # invariant explicit and closes a latent rel-(R,S) hole for ~0 breadth.
        if _has_relative_stereo_token(cand):
            return (False, False, None)

        # STEP 1 — CHEAP FAST-ACCEPT. Count identity -> offer as-is, UNFLAGGED. The
        # caller's exact-isomeric _rt_match(flagged=False) still verifies it, so a
        # conflict that happens to match the count is rejected downstream (unchanged
        # from today). Every currently-passing emission takes this branch.
        if general_engine_stereo_complete(mol, cand):
            return (True, False, cand)

        # STEP 2 — RECLAIM REQUIRES A LIVE OPSIN VERIFIER. Without it we cannot
        # verify a composed/as-is completion, so keep the conservative count-based
        # behaviour (never ship a stripped/omitted name unverified). The
        # ORTHONYM_BE_STRIP_STEREO measurement escape hatch is preserved verbatim
        # for the jar-dead path only (user directive 2026-08-31;).
        if self._disable_opsin_validity_gate or not _validity_gate_jar_present():
            import os as _os
            if (self._general_fallback_unverified
                    and _os.environ.get("ORTHONYM_BE_STRIP_STEREO") == "1"):
                return (True, True, cand)      # legacy ship-stripped
            return (False, False, None)

        # STEP 3 — CONSTITUTION PRECHECK. Only complete stereo onto a base whose
        # constitution already matches the input (skeleton InChIKey equal). A
        # wrong-constitution candidate is a wrong molecule -> abstain (composition
        # never reached). The skeleton prefilter also keeps compose off the giant
        # wrong-constitution suppressions (no new hang surface).
        in_skel = _self_consistency_skeleton(smiles)
        cand_smi = _validity_gate_name_to_smiles(cand)
        cand_skel = _self_consistency_skeleton(cand_smi) if cand_smi else None
        if in_skel is None or cand_skel != in_skel:
            return (False, False, None)

        # STEP 4a — AS-IS COMPLETE-BUT-MISCOUNTED (symmetric bis-/multiplied names).
        # cand may already express every element; the count merely under-counted it
        # (the bis(...) multiplier doubles the DEFINED units on mol). Verify with the
        # EXACT-ISOMERIC gate (NOT the count, NOT a bare InChIKey): _rt_match(
        # flagged=False), whose CanonSmiles == CanonSmiles branch is InChI-STRONGER
        # (catches C=N/ylidene the InChIKey is blind to — Adv3).
        if cand_smi is not None and self._rt_match(smiles, cand_smi, False):
            return (True, False, cand)         # e.g. CID 101644479

        # STEP 4b — OMISSION -> COMPOSE the input's stereo onto the flat/partial base
        # via the EXISTING RT-gated re-anchor. Returns the composed name only if
        # it recomputes to the input's full key; returns None on a genuine conflict /
        # unexpressible element. Emitted UNFLAGGED so the caller's exact-isomeric
        # _rt_match is the authority (, Adv3). Gated by the kill switch.
        in_key = _self_consistency_full_key(smiles)
        composed = (_try_compose_input_stereo(cand, smiles, in_key)
                    if (in_key and _STEREO_OMISSION_RECLAIM) else None)
        if composed:
            return (True, False, composed)     # e.g. CID 122493921 -> (1Z)-...

        # STEP 5 — genuine CONFLICT / unexpressible (axial, InChI-invisible tail that
        # composition cannot express) -> ABSTAIN all-or-nothing).
        return (False, False, None)

    @staticmethod
    def _rt_match(input_smiles: str, opsin_smiles: str,
                  stereo_flagged: bool) -> bool:
        """ T6.3: round-trip compare at the granularity the name
        asserts.

        For a ``stereo_unexpressed``-flagged emission this is a PER-ELEMENT
        atom-mapped stereo compare (RISK 3, closed here). ``_stereo_emit_decision``
        flags a name whenever it is stereo-INCOMPLETE, which INCLUDES a name that
        asserts PARTIAL stereo — so the compare must verify every element the name
        DOES assert while tolerating the ones it omits. It does this in two steps:

          1. a constitution guard — same molecular graph modulo stereo. Both sides
             are copied, ``RemoveStereochemistry``'d, and compared as ISOMERIC
             canonical SMILES. Removing only stereo (not ``isomericSmiles=False``,
             which also erases isotope labels) keeps BOTH charge AND isotope, so
             neutral != anion and a ``13C``-labelled input != its plain name;
             the reverted ca6bb3de loosening dropped charge — we must not reintroduce
             it, and the old ``isomericSmiles=False`` guard silently dropped isotope);
          2. a CHIRAL SUBSTRUCTURE match with the name's asserted structure
             (``opsin_smiles``) as the QUERY and the input as the TARGET. A query
             element WITHOUT stereo matches any target element (tolerate OMISSION);
             a query element WITH stereo forces the target to agree (catch CONFLICT
             and FABRICATION). ``useChirality=True`` enforces BOTH tetrahedral R/S
             and double-bond E/Z on the pinned RDKit (verified — the whole-molecule
             RegistrationHash of the reverted ca6bb3de was E/Z-blind and could not).
             Existence over valid mappings is the correct criterion, so a symmetric
             molecule (meso / C2 / enantiomer pair) is not spuriously rescued — no
             automorphism maps a wrong configuration onto the input.

        Because step 1 forces an atom-count bijection (same constitution), the
        step-2 match is a full isomorphism, not a fragment embedding. Soundness
        is scoped to the stereo classes RDKit's mol representation RETAINS from
        SMILES — tetrahedral R/S (incl. ring cis/trans, epoxide, S=O/P=O/N+) and
        double-bond E/Z. AXIAL stereo (allene/cumulene/atropisomer/helicene) is
        dropped by ``MolFromSmiles`` on BOTH sides, so it is invisible here — but
        also unreachable, as no producer can assert a descriptor RDKit will not
        carry (if that ever changes this silently un-closes). Enhanced-stereo
        CXSMILES groups (``|&1:..|``) are likewise ignored by ``MolFromSmiles``
        (as they are on the unflagged path). The compare is a strict tightening of
        the old constitution-only guard (NEW-accepts ⊆ OLD-accepts, verified), so
        it can never introduce a wrong-acceptance — only reject more. Derivation
        and probes: `internal notes` (RISK 3),
        ``a temp dir/risk3_stereo_probe*.py``.

        For every other emission use the EXACT isomeric comparison (byte-
        identical to the pre-P6 ``Chem.CanonSmiles`` == ``Chem.CanonSmiles``).
        Any exception -> False (fail-closed; matches the prior ladder).
        """
        try:
            if stereo_flagged:
                _o = Chem.MolFromSmiles(opsin_smiles)
                _i = Chem.MolFromSmiles(input_smiles)
                if _o is None or _i is None:
                    return False
                # constitution guard on COPIES (RemoveStereochemistry mutates; the
                # originals must keep their stereo for the step-2 chiral match).
                _oc, _ic = Chem.Mol(_o), Chem.Mol(_i)
                Chem.RemoveStereochemistry(_oc)
                Chem.RemoveStereochemistry(_ic)
                if Chem.MolToSmiles(_oc) != Chem.MolToSmiles(_ic):
                    return False
                # 0-wrong: RDKit CanonSmiles can call a charged / aromatic
                # multi-component pair equal while InChI distinguishes them, so back
                # the CanonSmiles constitution guard with the InChIKey SKELETON
                # (connectivity) the 0-wrong oracle uses. Strictly tightening (reject
                # more, never accept new); only rejects when the skeletons provably
                # disagree, else falls through to the chiral match as before.
                _sk_i = _self_consistency_skeleton(input_smiles)
                _sk_o = _self_consistency_skeleton(opsin_smiles)
                if _sk_i is not None and _sk_o is not None and _sk_i != _sk_o:
                    return False
                return _i.HasSubstructMatch(_o, useChirality=True)
            # Non-flagged: EXACT identity. CanonSmiles ALONE is too weak for
            # charged / aromatic multi-component species -- it called a 3-component
            # cyanine dye (frozen sample idx=3094) equal while InChI gave a DIFFERENT
            # key (WGHRUVOOYOJZCP != KXGYMDQLSGIIJG), so a wrong multi-fragment
            # `(1/1/1)` name shipped as verified. Require the FULL InChIKey the
            # 0-wrong oracle (reclaim_measure / _self_consistency_full_key) uses --
            # the call site already DOCUMENTS `_rt_match` as "a FULL-InChIKey
            # compare"; this makes the code match. Fall back to the CanonSmiles match
            # only when InChI is genuinely unavailable for a side (no regression).
            # Strictly tightening (NEW-accepts subset of OLD-accepts).
            if Chem.CanonSmiles(opsin_smiles) != Chem.CanonSmiles(input_smiles):
                return False
            _ik_i = _self_consistency_full_key(input_smiles)
            _ik_o = _self_consistency_full_key(opsin_smiles)
            if _ik_i is not None and _ik_o is not None:
                return _ik_i == _ik_o
            return True
        except Exception:
            return False

    def _try_general_engine_recovery(self, smiles: str) -> Optional[str]:
        """The late engine recovery (``_try_general_engine_recovery_impl``).

        Also records, in ``self._last_recovery_floor_substitute``, whether the
        name it returns is the universal floor standing in for a rung candidate
        that failed its round trip (``t4_coverage.last_t4_was_floor_substitute``),
        so the caller can let a clean-context recovery offer the engine's own name
        first (``_prefer_clean_over_floor_substitute``)."""
        origin: dict = {}
        rec = self._try_general_engine_recovery_impl(smiles, origin)
        self._last_recovery_floor_substitute = bool(
            rec is not None and origin.get("floor_substitute"))
        # P1: the name is the final rung's universal floor (every engine rung
        # declined), see ``_prefer_clean_over_floor_substitute``.
        self._last_recovery_final_floor = bool(
            rec is not None and origin.get("final_floor"))
        return rec

    def _engine_recovery_spelling(self, smiles: str) -> Optional[str]:
        """A second spelling of ``smiles`` for a caller whose use of the first one
        failed: the name the late general-engine recovery gives it
        (``_try_general_engine_recovery``), run in an isolated depth-0 naming
        session inside the caller's own naming (the tier context and the fragment
        memo are the caller's), without the main path's handlers.

        This is the spelling the engine ships when its main path declines the
        molecule, so it is the one an adduct component had before a main-path
        producer could name it ('...-7-carboxamide' where the main path now builds
        the demoted N-acyl float '...-7-carbonyl)-3-chloro-4-fluoroaniline'). The
        recovery round-trip gates its own name; the caller checks again what it
        builds from it. Best-effort tier only (None otherwise). The provenance the
        recovery records is dropped: the caller's own gates decide what ships.
        Never raises an ``Exception`` (a hang budget trip still unwinds to the
        frame that owns the budget)."""
        if not self._general_fallback_unverified:
            return None
        from .assembly.fragment_naming import isolated_naming_session
        from .metrics.provenance import get_provenance, restore_provenance
        snap = get_provenance()
        try:
            with isolated_naming_session():
                rec = self._try_general_engine_recovery(smiles)
        except Exception as exc:  # noqa: BLE001 - a second spelling is optional
            logger.info("engine recovery spelling failed for %s: %s", smiles, exc)
            rec = None
        finally:
            restore_provenance(snap)
        if rec is None or is_failure_name(rec):
            return None
        return rec

    def _prefer_clean_over_floor_substitute(self, smiles: str,
                                            floor_name: str) -> str:
        """``floor_name`` came from the in-``name`` recovery, where the rung
        candidate failed its round trip and the universal floor's verified
        spelling stood in for it. That failing rung is the degraded-context
        symptom ``_try_besteffort_clean_general_fallthrough`` exists for (the
        engine inside ``name`` builds a worse candidate than at depth 0 with a
        fresh memo), and the clean fall-through only runs when nothing shipped --
        so a verified floor spelling used to pre-empt the engine's own verified
        name, e.g. a 1,3-thiazole spelled '3-thia-1-azacyclopenta-1,4-diene' and
        its carboxylic acid as '12-hydroxy-...-13-oxa-...-12-en'. Offer-not-return
        (a project rule): run the clean recovery now and keep its name when it is
        not itself a floor stand-in; otherwise keep ``floor_name`` (and the
        provenance of the run that built it). Every candidate here passed the
        recovery's own full round trip. Best-effort, top level, gate on only --
        the same conditions as the clean fall-through.

        Only for a molecule with a ring. There the parent is a ring
        (1), the Blue Book, "Within the same class, a ring or ring system
        has seniority over a chain"), which the engine's rungs choose by ring
        seniority and spell by their Hantzsch-Widman / retained names, while the
        floor may pick a junior ring (benzene over pyridine) and spells every
        ring by replacement ('2-azacyclohexa-2,4,6-trien-1-yl'). In an acyclic
        molecule the floor's principal chain is the one (the Blue Book:
        20922, "(a) contains the greater number of heteroatoms of any kind; (b)
        has the greater number of skeletal atoms"), while the engine's ring-first
        rung, with no ring to choose, falls back to a short chain ('1-oxoethane'
        for CC(=O)NCN(C)N=O against the floor's '3-methyl-6-oxo-1-oxa-2,3,5-
        triazahept-1-ene'), so the floor spelling stays.

        Only in the outermost ``name`` (``name_scope_depth == 1``). The
        fragment-recursion depth that ``is_top_level_naming`` reads is 0 while
        any ``name`` dispatches its whole molecule (and ``isolated_naming_session``
        resets it to 0), so a nested best-effort ``name`` -- the ``name_compound``
        instances that name the acid of an acyloxy prefix or a glycoside cap,
        which inherit the best-effort tier from the context -- whose own recovery
        produced a floor stand-in ran this offer for its sub-structure (ChEBI
        PEG-glycopeptide, 209 heavy atoms: 12 nested offers, 43.9 s, each ending
        in the same floor string). A nested name is round-trip gated again by the
        frame that uses it, and the names this offer contributes come from the
        outermost frame: on the 13 dev-set rows whose stand-in it replaces, all 13
        at ``name_scope_depth == 1``, each from a clean engine rung."""
        from .assembly.fragment_naming import is_top_level_naming, name_scope_depth
        if not (name_scope_depth() == 1 and is_top_level_naming()
                and self._general_fallback_unverified
                and not self._disable_opsin_validity_gate):
            return floor_name
        _m = Chem.MolFromSmiles(smiles)
        if _m is None or _m.GetRingInfo().NumRings() == 0:
            return floor_name
        from .metrics.provenance import get_provenance, restore_provenance
        _snap = get_provenance()
        # Two clean contexts, the cheap one first. The in-``name`` recovery is
        # degraded three ways (``_try_besteffort_clean_general_fallthrough``): the
        # best-effort context variables, the elevated session depth and the
        # molecule's fragment memo. The first clean run is a probe: it resets the
        # first two and READS the memos (fragment memo and scope memo) without
        # writing them (``speculative_fragment_naming`` + ``push_sandbox``: its
        # writes and its budget spend are dropped on exit), so every fragment the
        # main path already named costs nothing. The memo holds names the main
        # path built in the best-effort context, so an engine rung built on them
        # is not the clean-context name (dev2000 'O=S(=O)(O)c1cccc(N=Nc2ccc(Nc3cc
        # ccc3)cc2)c1': '...-3-(2-hydroxy-1,3-dioxa-2-thiapropa-1,2-dienyl)benzene'
        # from the probe, '...-3-(1,1-dioxo-2-oxa-1λ6-thiaethyl)benzene' fresh), and
        # the probe never ships one. It ships only the final rung's universal
        # floor, reached when every engine rung declined even with the molecule's
        # fragment names at hand; otherwise the recovery runs on a fresh memo
        # exactly as before (the probe left no trace in any memo or budget). On the
        # 36 outermost offers of the a dev split / milestone1500 / dev2000 best-effort
        # evals, the 11 probes that ended in the final rung gave the fresh run's
        # name and the fresh run ended there too. A fresh memo re-derives every
        # fragment of the molecule: for two ChEBI peptides (119 and 209 heavy
        # atoms) 12.3 s and 43.8 s against the recipe's 90 s stall, where the
        # probe took 0.14 s and 2.5 s and gave the same string.
        from .assembly.fragment_naming import (
            PerfBudgetExceeded,
            speculative_fragment_naming,
        )
        from .assembly.memo import pop_sandbox, push_sandbox
        # Branch review fixes: the fresh-memo run names the CANONICAL spelling of the
        # molecule (RDKit's canonical isomeric SMILES, the same string for every
        # atom order), so its name no longer depends on how the input was written
        # (dev2000 'O=S(=O)(O)c1cccc(N=Nc2ccc(Nc3ccccc3)cc2)c1' came out three ways
        # over five spellings, each replacing the same floor string). It is the same
        # molecule, so its round trip is the input's; the probe keeps the input
        # spelling because it reads the molecule's memos.
        _canonical_smiles = Chem.MolToSmiles(_m) or smiles
        # PubChem 1M -- branch losses: the offer re-names a molecule whose verified
        # stand-in ships either way, so it is an optional improvement, bounded by
        # what the main path cost: at most as many naming passes
        # (``Orthonym._name_impl`` calls, ``fragment_naming.naming_passes``) as the
        # main path made before it, and never fewer than ``_OFFER_MIN_PASSES`` (one
        # clean recovery of a small molecule already takes several: 52 for a dev2000
        # row whose main path made 5). A clean context misses every memo entry the
        # best-effort main path made (the fragment and sort-key memos are keyed by
        # the naming context), so for a big peptide the offer re-derived the whole
        # molecule: PubChem-1M rows at 82 s (1,519 passes against the main path's
        # 5), 64 s (964 against 143) and more than 600 s (the fragment work budget
        # exhausted), each ending with the stand-in's own string. The 39 outermost
        # offers of the a dev split / milestone1500 / dev2000 best-effort rows made at
        # most 52 passes (census at d6d6187ac), so none of them meets the cap. The
        # count depends on the molecule alone, so the outcome is the same in every
        # process. An offer that reaches the cap keeps the stand-in (the name the
        # engine shipped before the offer existed).
        from .assembly.fragment_naming import (
            OptionalNamingCapExceeded,
            naming_pass_cap,
            naming_passes,
        )
        try:
            with naming_pass_cap(max(naming_passes(), _OFFER_MIN_PASSES)):
                for _fresh_memo in (False, True):
                    self._last_recovery_final_floor = False
                    if _fresh_memo:
                        clean = self._try_besteffort_clean_general_fallthrough(
                            _canonical_smiles)
                    else:
                        _sandbox = push_sandbox()
                        try:
                            with speculative_fragment_naming():
                                clean = self._try_besteffort_clean_general_fallthrough(
                                    smiles, reset_cache=False)
                        except PerfBudgetExceeded:
                            clean = None  # not charged (speculative); the fresh run decides
                        finally:
                            pop_sandbox(_sandbox)
                    if (clean is not None and not is_failure_name(clean)
                            and not getattr(self, "_last_recovery_floor_substitute", False)
                            and (_fresh_memo
                                 or getattr(self, "_last_recovery_final_floor", False))):
                        return clean
                    restore_provenance(_snap)
        except OptionalNamingCapExceeded:
            logger.info("clean-first offer stopped at its naming-pass cap; "
                        "the stand-in is kept")
            restore_provenance(_snap)
        self._last_recovery_floor_substitute = True
        return floor_name

    def _try_general_engine_recovery_impl(self, smiles: str,
                                          origin: dict) -> Optional[str]:
        """ (opt-in): late engine recovery for abstentions.

        Covers the two flows the in-``_name_impl`` wiring cannot see:
        (a) a handler raised a NamingLimit (e.g. UNSUPPORTED_RING_SYSTEM)
        that bypassed the GENERAL block, and (b) a candidate name was
        SUPPRESSED by a downstream gate (/ vetoes) after
        ``_name_impl`` returned. Re-perceives, runs the general engine,
        and re-applies the SAME E1 + gates to the emission.
        Returns a verified name or None; never raises (fail-closed).
        Default OFF (``general_fallback=False``) -> byte-identical.
        """
        if not self._general_fallback:
            return None
        from .assembly.fragment_naming import is_top_level_naming
        if not is_top_level_naming():
            return None
        #: before the von-Baeyer engine, prefer a retained fused-heterocycle
        # whole-molecule name (the `...9H-purin-6-amine` parent of an acyl-CoA)
        # when it round-trips. A giant purine-containing molecule is not itself a
        # bare fused ring, so the normal dispatch never routes it to
        # `name_fused_heterocycle`; without this it is named on a von-Baeyer
        # `...tetraazabicyclo[4.3.0]...` polyene here. RT-gated (adopted only when
        # verified), so a molecule whose fused form is not yet RT-exact still
        # falls through to the valid von-Baeyer name below -- no regression,
        # 0-wrong. Best-effort only, so PIN/default output is unchanged.
        if self._general_fallback_unverified:
            _mol_up = Chem.MolFromSmiles(smiles)
            if _mol_up is not None:
                _fused_up = self._try_retained_fused_upgrade(_mol_up, smiles)
                if _fused_up is not None:
                    return _fused_up
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            from .assembly.general_engine import name_general
            from .validation.coverage_gate import certify_general_result
            canonical = Chem.MolToSmiles(mol, canonical=True)
            # FINAL-REVIEW FIX 1: mark whether `cand` came from the
            # aggressive producer, so the shared ladder below can require it
            # to POSITIVELY round-trip (never ship a name OPSIN cannot parse).
            # Stays False for the multi-fragment and engine-own-name paths, so
            # their `general_fallback_unverified` semantics are byte-identical.
            _cand_from_t4 = False
            # B5 general lever: retained so the RT-mismatch rescue below can
            # re-run the cascade. Stays None on the multi-fragment path
            # (which never perceives single-component features), so the rescue
            # is single-component-only by construction.
            feats = None
            if len(Chem.GetMolFrags(mol)) > 1:
                #: multi-fragment split-name-join, complete tier ONLY.
                # `name_general` (and every single-component handler) refuses a
                # multi-fragment mol, so the recovery would otherwise abstain.
                # Under `complete` (allow_aromatic_general), name each NEUTRAL
                # component through the complete-tier single-component pipeline
                # and re-assemble via the existing adduct namer
                # (order/proportion/water-last preserved). Fail-closed if ANY
                # component is unnameable OR charged (charged -> P5 scope). The
                # whole-string gate below round-trips the joined name.
                cand = self._name_multifragment_complete(mol, canonical)
                if cand is None:
                    return None
            else:
                feats = self._perceive(mol, smiles, canonical)
                self._classify(feats)
                eng = name_general(
                    mol, feats,
                    allow_aromatic_general=self._allow_aromatic_general,
                    allow_suffix_free=self._general_fallback_unverified)
                #: charge is lifted only under complete
                # (allow_aromatic_general); the E1 cert must accept the charged
                # partition there too (the charge is a suffix on a bound atom).
                # a phase B4: the shared best-effort certification gate (E1 +
                # structural binding-spine axes: atom/bond/charge/stereo). A
                # structural void routes to the producer below exactly like an
                # E1 failure -- catching a bad eng result EARLY so gets a
                # chance (the measured +1 breadth-routing gain). structural_only
                # keeps the P4/P5/P6 name-spelling FPs advisory (covers
                # name well-formedness downstream).
                if eng is None or not certify_general_result(
                        mol, eng,
                        allow_charged=self._allow_aromatic_general,
                        structural_only=True):
                    # (best-effort tier ONLY): the engine's own
                    # flag-gated attempt just declined (name_general returned
                    # None) or produced a non-atom-complete partition. Hand off
                    # to the universal coverage-by-construction producer, which
                    # re-runs name_general at the T4-PERMISSIVE flags
                    # (allow_aromatic_general / allow_suffix_free) INDEPENDENT of
                    # this instance's flags and E1-audits the result internally,
                    # returning an atom-complete name or None (clean abstain).
                    # PIN-isolated: this whole recovery is already
                    # `_general_fallback`-gated (early `return None` at the top).
                    # The name is NOT shipped raw: it falls through to the SAME
                    # stereo-emit decision + OPSIN round-trip ladder below that the
                    # engine's own name gets, so a name whose constitution does
                    # not round-trip is suppressed to abstain .
                    # Task 4 (coverage-by-construction breadth).
                    #
                    # tier-policy: this is the MEASURED demotion gate for the
                    # ~88/150 ring+stereo abstainers that ship a FULL-InChIKey
                    # RT-verified name at best-effort but abstained at COMPLETE
                    # (trace: internal notes CORESTEREO
                    # addendum + task-tierpolicy-report.md). It was
                    # `_general_fallback_unverified` (best-effort only); re-gated
                    # to `_general_fallback` so a name also ships at the COMPLETE
                    # tier -- but ONLY through the RT ladder below. At the complete
                    # tier `_stereo_emit_decision` returns `(True, False)` only for
                    # a FULL-stereo name (else `(False, False)` -> abstain), so the
                    # `_rt_match` is a FULL-InChIKey compare; and a name OPSIN
                    # cannot parse hits `elif... or _cand_from_t4: return None`
                    # (below) regardless of tier. An unverified candidate is
                    # therefore never shipped at COMPLETE -- 0-wrong preserved.
                    # `name_t4_complete` still requires a live jar (below), and the
                    # default/PIN tier never reaches here (`_general_fallback`
                    # False), so PIN output is byte-identical.
                    if not self._general_fallback:
                        return None
                    # runs the engine at allow_aromatic_general=True
                    # unconditionally, which OPENS the von-Baeyer / mancude-cage
                    # path even for instances that deliberately kept
                    # allow_aromatic_general OFF. E1 proves atom COVERAGE but not
                    # ring-numbering validity / PIN-preference, so an aggressive
                    # von-Baeyer name opens can be caught ONLY by the OPSIN
                    # round-trip ladder below -- and with no jar that gate fails
                    # OPEN , so would ship an UNVERIFIED (possibly
                    # non-PIN or valence-illegal) name. Keep the abstention
                    # without a jar, mirroring the inline site's FIX 1
                    # abstain-without-Java contract. (Internal/test mode via
                    # _disable_opsin_validity_gate is a controlled context and
                    # keeps the ladder's own unverified-ship path.) This leaves
                    # the engine's own-name path below byte-identical; it only
                    # constrains the NEW producer to a live.
                    if not (self._disable_opsin_validity_gate
                            or _validity_gate_jar_present()):
                        return None
                    from .assembly.fragment_naming import isolated_naming_session
                    from .assembly.t4_coverage import name_t4_complete
                    # Phase1 B5: is a fresh whole-molecule naming, but the
                    # recovery lane invokes it mid-name at session_depth >= 1, so
                    # its recursion hits MAX_NAMING_DEPTH prematurely and DEGRADES
                    # to an abstention (measured root cause). Run it in an isolated
                    # depth-0 session so it gets the full recursion budget a
                    # standalone call gets.
                    from .assembly.t4_coverage import (
                        last_t4_was_final_floor,
                        last_t4_was_floor_substitute,
                        reset_t4_floor_substitute,
                    )
                    reset_t4_floor_substitute()
                    with isolated_naming_session():
                        cand = name_t4_complete(mol, feats)
                    if cand is None:
                        return None
                    origin["floor_substitute"] = last_t4_was_floor_substitute()
                    origin["final_floor"] = last_t4_was_final_floor()
                    # FINAL-REVIEW FIX 1: this name must POSITIVELY
                    # round-trip or the path ABSTAINS. The shared ladder below
                    # ships a `general_fallback_unverified` emission even when
                    # OPSIN CANNOT PARSE it (its `elif not
                    # self._general_fallback_unverified` is False for the T4
                    # opt-in) -- correct for the engine's OWN best-effort name,
                    # but NOT for the aggressive producer, which opens
                    # von-Baeyer / mancude paths whose ONLY validity check is
                    # this round-trip. E1 proves atom COVERAGE (so never a wrong
                    # MOLECULE) but a name OPSIN cannot parse is malformed. Flag
                    # the origin so the ladder rejects an unparseable name
                    # instead of shipping it unverified; the parse-but-mismatch
                    # case is already rejected there for every tier. T4-SCOPED:
                    # the flag is False everywhere else, so the engine-own and
                    # existing-tier paths are untouched.
                    _cand_from_t4 = True
                else:
                    cand = eng.name
                    # (audit-only): record the spine the certificate just
                    # accepted, so the exit of name can re-assert it on the
                    # FINAL string. Deliberately placed BEFORE the retained-name
                    # preference below: when that swap fires, `eng.bindings`
                    # describe a name that is no longer shipped and the re-anchor
                    # is SUPPOSED to report TOKEN_ABSENT. That is a true positive
                    # about a real gap in today's proof coverage (the swap is
                    # independently OPSIN-RT-gated, so it is not a wrongness bug
                    # today; what is missing is bindings for the substituted
                    # name), and it must not be papered over by recording later.
                    self._record_binding_proof(
                        mol, eng, stage="general_engine_recovery")
                    #: retained-name preference (PIN-quality guardrail,
                    # `complete` tier only). A structural recognizer is about to
                    # be OVERRULED by an ugly von-Baeyer/replacement name the
                    # general engine just built -- before that ships, see
                    # whether a retained/fused-heterocycle name applies to the
                    # WHOLE input molecule and prefer it, but ONLY if it
                    # independently round-trips. This is a preference over an
                    # emission that already exists, never a precondition to
                    # naming: if the recognizer finds nothing, or its candidate
                    # fails to verify, `cand` is untouched and the shared
                    # verification ladder below runs on the general name exactly
                    # as it did before this phase.
                    if self._allow_aromatic_general:
                        _retained_cand = self._retained_structural_preference(mol)
                        if _retained_cand and _retained_cand != cand:
                            if self._disable_opsin_validity_gate:
                                cand = _retained_cand  # test/internal mode
                            elif _validity_gate_jar_present():
                                _r_smi = _validity_gate_name_to_smiles(_retained_cand)
                                _r_ok = False
                                if _r_smi is not None:
                                    try:
                                        _r_ok = (Chem.CanonSmiles(_r_smi)
                                                 == Chem.CanonSmiles(smiles))
                                    except Exception:
                                        _r_ok = False
                                if _r_ok:
                                    cand = _retained_cand
                            # jar missing: fail-closed -- keep the general `cand`
                            # rather than ship an unverified retained guess.
            # FIX 2: fail closed on DROPPED stereo (general-engine path).
            # The ladder below is CONSTITUTIONAL (atoms+bonds+charge,
            # stereo-blind), so a general emission that omits E/Z or R/S the
            # input carries would pass and ship a WRONG stereoisomer
            # specification. The universal stereo backstop is LOG-ONLY for the
            # handler='unknown' cohort that general emissions arrive as (no
            # authoritative parent locant map), so it cannot suppress this. Any
            # stereo the engine DOES express (parent-scope _stereo_prefix or
            # substituent-internal name_substituent) makes needs_stereo_injection
            # False, so genuinely-expressed cases still ship WITH stereo -- only
            # genuinely-dropped-stereo emissions abstain here.
            # T6.2: shared stereo-emit policy (was an inline
            # needs_stereo_injection fail-close). complete/valid abstain on
            # dropped stereo; best-effort ships a constitution-only name flagged
            # stereo_unexpressed (0-wrong: names a superset, never a wrong
            # stereoisomer).
            _stereo_flagged = False
            if cand:
                _permitted, _stereo_flagged, _name_out = self._stereo_emit_decision(
                    mol, cand, smiles)
                if not _permitted:
                    return None
                cand = _name_out          # adopt composed/as-is reclaim
            #: explicit verification ladder. verified = OPSIN parsed
            # the name AND it round-trips to the input structure. A parsed-
            # but-MISMATCHED name is NEVER shipped, at any tier. task-JAR-ABSENT
            # (a review 0-wrong hole): the no-jar best-effort branch below no
            # longer ships unverified -- the old claim that "no-jar transience
            # ships ONLY behind the opt-in... a genuine environment gap,
            # not a proof gap" mischaracterized the hole (a review's witnesses were
            # demonstrably WRONG-molecule, not merely-unprovable-but-correct),
            # so the branch now routes through `verify_or_none` and abstains.
            # no-abstain Phase A: when the jar IS present but OPSIN
            # rejects `cand` outright, the WIRED oracle below decides --
            # see the `else` branch after the `_opsin_smi is not None` check.
            opsin_status = "unverified"
            if self._disable_opsin_validity_gate:
                pass  # test/internal mode: ship unverified (G2 contract)
            elif not _validity_gate_jar_present():
                if not self._general_fallback_unverified:
                    return None
                # no-abstain / task-JAR-ABSENT (a review 0-wrong hole): a
                # jar-absent best-effort emission must NOT ship `cand`
                # unverified. Before this fix control fell straight through
                # this whole if/elif/else to the emit at the bottom with
                # opsin_status="unverified" and ZERO verification of any kind
                # -- a review measured 7/7 WRONG-molecule ships in a no-Java
                # deployment (`COS(=O)(=O)O`->`methane`;
                # `ClP(Cl)(=O)OC1=CC=CC=C1`->`(phosphonooxy)benzene` with Cl2
                # silently swapped for (OH)2, a wrong constitution). Route
                # through the Wave-0 OPSIN-free reconstructor oracle instead,
                # exactly as the jar-PRESENT rejected branch does at the `else`
                # below (shared seam). `name_facts=None` here (there is no
                # name->NameFacts extractor for an arbitrary general-engine
                # name yet -- see the `else` branch note), so with NO jar
                # `verify_or_none`'s OPSIN branch is "unavailable" and its
                # reconstructor branch (which requires name_facts) is skipped:
                # it is PROVABLY None on this path today -- i.e. a clean
                # fail-closed ABSTAIN, while remaining future-proof (a later
                # NameFacts extractor lights up real jar-absent verification
                # here for free, with zero further change to this call site).
                # NEVER ship unverified. Label matches the `else` branch:
                # with name_facts=None the ONLY reachable non-None is a
                # genuine OPSIN full-InChIKey confirm, so "verified" is honest.
                from .validation.reconstruct import verify_or_none
                if verify_or_none(cand, smiles, name_facts=None) is None:
                    return None
                opsin_status = "verified"
            else:
                _opsin_smi = _validity_gate_name_to_smiles(cand)
                if _opsin_smi is not None:
                    # T6.3: compare at the granularity the name asserts
                    # -- constitution-only for a flagged emission, exact
                    # isomeric otherwise.
                    if not self._rt_match(smiles, _opsin_smi, _stereo_flagged):
                        # Phase1 B5 general lever: the recovery lane's own
                        # (rung-0) general name denotes a DIFFERENT molecule, but
                        # the PG-suppressed cascade may build a complete,
                        # correct one (measured: 9/45 in-scope suppressed rows).
                        # Try it ONCE -- best-effort + live-jar only, positively
                        # RT-verified -- rather than abstain. Never fires for a
                        # name that was already (`_cand_from_t4`) or on the
                        # multi-fragment path (`feats is None`).
                        _rescue = (None if (_cand_from_t4 or feats is None)
                                   else self._try_t4_rescue(mol, feats, smiles))
                        if _rescue is None:
                            return None  # wrong name: never ship
                        cand, _stereo_flagged = _rescue
                    opsin_status = "verified"
                elif not self._general_fallback_unverified or _cand_from_t4:
                    # rejected/transient: no unverified opt-in, OR a name
                    # that must POSITIVELY round-trip (FINAL-REVIEW FIX 1). For
                    # a non- name `_cand_from_t4` is False, so this reduces to
                    # the pre-existing `not general_fallback_unverified` guard --
                    # byte-identical for the engine-own and multi-fragment paths.
                    return None
                else:
                    # no-abstain Phase A (the `verify_or_none` keystone):
                    # OPSIN could not parse `cand` at ALL (jar present;
                    # rejected or a transient in-process failure -- the two
                    # are indistinguishable from a single None here, exactly
                    # as the pre-existing branches above already treat them).
                    # Before this SHIPPED completely unverified -- `opsin_status`
                    # stayed "unverified" with literally no proof, the exact gap
                    # measured live on a dev split (3 witnesses: a spiro-fused
                    # dioxatricyclic ketone, a dioxa-aza-cyclododecanone, and a
                    # spiro-decahydronaphthalene isoindolinone -- task-A-report.md
                    # A.1). Route it through the Wave-0 OPSIN-free reconstructor
                    # oracle instead of shipping bare: `verify_or_none` ships
                    # `cand` only if OPSIN parses it after all (re-checked
                    # inside the oracle) OR the reconstructor CONFIRMS it from
                    # `name_facts`.
                    #
                    # `name_facts=None` here: `cand` at this rung is an
                    # assembled name STRING from the general engine's own
                    # `name_general` result (`eng`, not a `_Candidate` with a
                    # `result_obj`), and there is no name->NameFacts
                    # extractor for an arbitrary general-engine name yet (a
                    # separate, larger task -- see task-A-report.md A.3). With
                    # `name_facts=None` only the OPSIN branch could confirm,
                    # and it already failed to parse one line above, so this
                    # call ALWAYS abstains today -- that is the CORRECT, sound
                    # answer (never a false CONFIRM): a name neither OPSIN nor
                    # the reconstructor can check must not ship unverified.
                    # This TIGHTENS the tier (measured before/after in
                    # task-A-report.md); it does not add a stereo-omission
                    # degrade path.
                    # fix-a performance pass Finding 3: with `name_facts=None`, the ONLY
                    # reachable success inside `verify_or_none` is its OPSIN
                    # full-InChIKey branch (the reconstructor branch requires
                    # `name_facts is not None`, unreachable here) -- so a
                    # non-None return here is a genuine OPSIN verification,
                    # not a reconstructor one. Label it "verified" (accurate),
                    # never "verified_reconstructor" (a future task that
                    # supplies real `name_facts` and reaches the reconstructor
                    # branch may introduce its own distinct label then).
                    from .validation.reconstruct import verify_or_none
                    if verify_or_none(cand, smiles, name_facts=None) is None:
                        return None
                    opsin_status = "verified"
            if cand and not is_failure_name(cand):
                from .metrics.provenance import record_source, record_stereo_unexpressed
                record_source("general_engine", opsin=opsin_status)
                record_stereo_unexpressed(_stereo_flagged)
                return cand
        except Exception as e:  # fail-closed: keep the abstention
            logger.info(
                "general engine late recovery error (kept abstention): %s", e)
        return None

    def _try_besteffort_clean_general_fallthrough(
            self, smiles: str, *, reset_cache: bool = True) -> Optional[str]:
        """CQ5 Task A: the RT-failure fall-through (best-effort tier only).

        Re-invoke the RT-gated ``_try_general_engine_recovery`` in a CLEAN naming
        context so the general engine reaches the RT-verifying systematic /
        replacement-nomenclature candidate that the in-``name`` recovery misses.

        THREE things break the in-``name`` recovery for the converting witnesses
        (trace-confirmed, a project rule — `internal notes` +
        ``task-A-report.md`` + ``task1-report.md``):

        * The best-effort contextvars (``best_effort_ctx`` etc.) are SET, so the
          substituent recursion (which re-enters through ``name_compound`` and
          reads them) uses the best-effort substituent path, which emits the SAME
          RT-failing substituent form the primary did rather than the systematic
          replacement-nomenclature form that round-trips.
        * ``session_depth`` is already >= 1, so the substituent recursion consumes
          ``MAX_NAMING_DEPTH`` from an elevated floor and a deep substituent hits
          the cap prematurely, degrading to a worse name (exactly the reason the
           sub-branch already wraps its call in ``isolated_naming_session``).
        * CQ5 Task 1 (RISK 4): the whole-molecule fragment MEMO CACHE is still the
          PRIMARY pass's, and it holds ``recursion_depth_fallback`` SKIP entries for
          the deep substituents the primary could not name (poisoned by the same
          elevated ``session_depth``). ``isolated_naming_session`` deliberately keeps
          that cache live (the giant-hang fix), so a depth-0 reset alone reads the
          poisoned skips back and still misses the good name — the
          ``COP(=O)(C=C(F)F)C=C(F)F`` drop. ``reset_cache=True`` installs a fresh
          empty cache for the isolated body, so the deep substituents are re-derived
          from the depth-0 budget; it is restored on exit.

        Resetting the four propagation contextvars to their defaults and running in
        a depth-0 isolated session WITH a fresh memo cache reproduces the state a
        fresh top-level call gets, under which ``name_general`` produces the
        RT-verifying candidate. The recovery's OWN OPSIN round-trip gate still
        decides what ships, so a candidate that does not round-trip returns ``None``
        (0-wrong; a project rule — verify what SHIPS, not just that the bad path
        stopped). Never raises.

        Gated by the caller on ``_general_fallback_unverified`` + the
        ship-a-failure path, so PIN/complete output is byte-identical and no
        currently-shipping name can change (offer-not-return; a project rule).

        ``reset_cache=False`` keeps the fragment memo the caller has installed
        (context variables and session depth are still reset): the memo-reading
        first attempt of ``_prefer_clean_over_floor_substitute``, which runs it
        inside ``speculative_fragment_naming`` so that memo is a throwaway copy.
        The fall-through itself always runs on a fresh memo.
        """
        if not self._general_fallback_unverified:
            return None
        from .assembly.fragment_naming import isolated_naming_session
        from .metrics.provenance import (
            allow_aromatic_general_ctx,
            best_effort_ctx,
            full_coverage_ctx,
            general_fallback_ctx,
        )
        _toks = None
        try:
            _toks = (
                general_fallback_ctx.set(False),
                best_effort_ctx.set(False),
                allow_aromatic_general_ctx.set(False),
                full_coverage_ctx.set(False),
            )
            with isolated_naming_session(reset_cache=reset_cache):
                return self._try_general_engine_recovery(smiles)
        except Exception as e:  # fail-closed: keep the abstention
            logger.info(
                "best-effort clean fall-through error (kept abstention): %s", e)
            return None
        finally:
            if _toks is not None:
                for _var, _tok in zip(
                        (general_fallback_ctx, best_effort_ctx,
                         allow_aromatic_general_ctx, full_coverage_ctx),
                        _toks):
                    try:
                        _var.reset(_tok)
                    except Exception:
                        pass

    def _try_perf_budget_t4_rescue(self, smiles: str) -> Optional[str]:
        """ M0: last-resort, BOUNDED, strictly-RT-gated whole-molecule name
        for a molecule whose MAIN naming path exhausted the macrocycle-hang budget.

        The op budget fires when the main path's combinatorial ring analysis
        explodes, but the coverage-by-construction producer
        (``t4_coverage.name_t4_complete``) frequently names the same molecule
        CHEAPLY and EXACTLY -- a 100-heavy-atom 10-ring fused peptide is named at
        497 chars with a FULL-InChIKey round-trip (measured 2026-09-08). Before
        this the outermost ``PerfBudgetExceeded`` boundary abstained
        unconditionally AND suppressed the floor offer (to dodge a re-hang and the
        metallo-macrocycle OPSIN-unparseable coordination name its lenient floor RT
        wrongly accepts). This offers that name back under two guards that
        neutralise both hazards:

        * ``rearm_hang_budgets`` installs a FRESH budget around the single T4
          call, so a genuinely explosive molecule re-raises ``PerfBudgetExceeded``
          (caught here -> clean abstain) rather than hanging.
        * STRICT verification: OPSIN must PARSE the name AND it must round-trip
          (``_rt_match``), so an unparseable coordination name is rejected
          (0-wrong; inv 9 -- verify what SHIPS, not just that the bad path stopped).

        Best-effort tier only (``_general_fallback_unverified``) and reached only on
        a molecule that ALREADY abstains at this boundary -- or, since the ZINC-loss
        fix, from ``name`` itself when one of its failure rescues exhausted the
        budget and the molecule is about to ship a failure (``_FailureRescues``) --
        so no currently-shipping name can change and PIN/complete output is
        byte-identical (offer-not-return, inv 18). Never raises.
        """
        if not self._general_fallback_unverified:
            return None
        # Mirror the sub-branch's live-jar contract (FIX 1): with no jar
        # the OPSIN round-trip below cannot verify, and an unverified name
        # (von-Baeyer / mancude paths) must never ship -- abstain cleanly instead.
        if not (self._disable_opsin_validity_gate or _validity_gate_jar_present()):
            return None
        from .assembly.fragment_naming import (
            PerfBudgetExceeded, disarm_hang_budgets, isolated_naming_session,
            rearm_hang_budgets)
        from .assembly.t4_coverage import name_t4_complete
        try:
            rearm_hang_budgets()  # BOUND the rescue: a re-explosion re-raises
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            # `name_t4_complete(mol, None)` perceives internally and, run in a
            # depth-0 isolated session with a FRESH memo cache (the CQ5
            # clean-fallthrough state), names the giant in ~0.2 s (feats=None is
            # sufficient here, unlike the elevated-depth linear catch M0 tried).
            # Deliberately NOT re-running `self._classify` first: that is the
            # SAME combinatorial parent analysis that just exploded, and it
            # re-explodes on a genuine hang witness (measured: vancomycin's
            # rescue was 22 s of re-classify, name_t4 itself 0 s). The producer's
            # own internal perception is cheap and bailed early for every hang
            # witness (measured: 0-2200 inner ops), so the rescue is near-free.
            with isolated_naming_session(reset_cache=True):
                cand = name_t4_complete(mol, None)
            if not cand or is_failure_name(cand):
                return None
            _permitted, _flagged, _name_out = self._stereo_emit_decision(
                mol, cand, smiles)
            if not _permitted:
                return None
            cand = _name_out          # adopt composed/as-is reclaim
            if self._disable_opsin_validity_gate:
                return cand
            _smi = _validity_gate_name_to_smiles(cand)
            if _smi is not None and self._rt_match(smiles, _smi, _flagged):
                return cand
            return None
        except PerfBudgetExceeded:
            return None  # rescue itself exploded -> clean abstain
        except Exception as e:  # noqa: BLE001 - fail-closed
            logger.info("perf-budget T4 rescue error (kept abstention): %s", e)
            return None
        finally:
            disarm_hang_budgets()  # keep the descriptive fallback below safe

    def _try_demote_senior_group_rescue(self, smiles: str):
        """ tail #7: best-effort rescue that re-picks the principal group with
        ACYCLIC senior characteristic groups (ester / carboxylic acid / their
        chalcogen analogues) excluded, then re-names via the
        ``_principal_group_override`` seam so a ring parent that was blocked by an
        off-ring senior group can name (with that group demoted to a detachable
        prefix). Returns a POSITIVELY round-tripping name or ``None`` (fail-closed).

        Guarded: best-effort opt-in + live OPSIN jar (the caller already checks
        both); only fires on the ship-a-failure path so it can never alter a
        passing name. The re-perception works in two steps --
        cut the ester link atom out of parent selection, recompute seniority (see
        known parent-selection findings) -- but here
        via the existing override rather than a new parent-selection path.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            from .perception.functional_groups import detect_functional_groups
            from .rules.seniority import get_principal_group
            fgs = detect_functional_groups(mol)
            # The acyclic senior groups that most often mis-own a ring parent.
            _DEMOTE = ('ester', 'carboxylic_acid', 'thioester', 'thiocarboxylic_acid',
                       'carbothioic_acid')
            if not any(fgs.get(g) for g in _DEMOTE):
                return None
            _reduced = {k: v for k, v in fgs.items() if k not in _DEMOTE}
            new_pg, _ = get_principal_group(mol, _reduced)
            if new_pg is None:
                return None  # nothing else to anchor on -> leave the abstention
            # Re-name with the demoted principal group forced. A FRESH namer at
            # the PIN style keeps the ring assembler on its normal path; the
            # override is the same seam rules/ions.py / charged_router.py use.
            inner = Orthonym(style=self.style,
                              _principal_group_override=new_pg)
            cand = inner.name(smiles)
            if not cand or is_failure_name(cand):
                return None
            _permitted, _flagged, _name_out = self._stereo_emit_decision(
                mol, cand, smiles)
            if not _permitted:
                return None
            cand = _name_out          # adopt composed/as-is reclaim
            if self._disable_opsin_validity_gate:
                return cand
            smi = _validity_gate_name_to_smiles(cand)
            if smi is not None and self._rt_match(smiles, smi, _flagged):
                return cand
            return None
        except Exception as e:  # fail-closed: keep the abstention
            logger.info("demote-senior-group rescue error (kept abstention): %s", e)
            return None

    def _try_alternate_parent_rescue(self, smiles: str):
        """ (offer-not-return, a project rule): best-effort rescue that,
        when ``ranked[0]`` (the PIN-correct senior parent) leads to an
        un-nameable remainder, RETRIES the JUNIOR members of the same pool
        and adopts the FIRST whose full name round-trips. Returns a POSITIVELY
        round-tripping name or ``None`` (fail-closed).

        This realises the OFFER the pool always had: ``p44_scorer.pool_candidates``
        builds >=2 ranked parent candidates for these molecules but
        ``select_parent_unified`` historically hard-committed ``ranked[0]`` and
        discarded ``ranked[1:]`` (p44_scorer.py:543-545), and the gate-rejection
        retry (`_retry_cascade_on_gate_rejection`) only swaps the DISPATCH CLASS
        for the SAME parent -- so a wrong-parent dead-end was terminal. Here we
        re-name on a FRESH inner instance with ``_forced_parent_rank=k``, which
        threads ``_offer_rank=k`` into that instance's every ``select_parent``
        call (top-level AND its own internal gate-reject retries), so the junior
        parent stays selected while the dispatch cascade finds a handler for it.

        PIN-safety (a project rule + 8): the ranking is NEVER reordered and
        ``ranked[0]`` is never changed -- this runs ONLY on the ship-a-failure
        path at the best-effort tier (the caller checks
        ``is_top_level_naming and is_failure_name(result) and
        self._general_fallback_unverified and not
        self._disable_opsin_validity_gate``), so every PIN-nameable molecule has
        already succeeded at ``ranked[0]`` and never reaches here. The RT gate
        (`_rt_match`) makes it 0-wrong: a junior candidate is adopted only if its
        full name reproduces the input's full structure -- never an unverified
        lower-ranked name. Bounded by the real pool length.
        """
        # Never nest: a forced-rank inner instance, or an inner spawned by
        # another best-effort rescue, must not re-enter the offer (its own
        # ship-failure hooks can be top-level because is_top_level_naming keys
        # off the fragment `visited` set, not session depth).
        if self._forced_parent_rank != 0:
            return None
        if getattr(_ALT_PARENT_RESCUE, 'active', False):
            return None
        # tier-policy: run at the COMPLETE tier (`_general_fallback`), not
        # best-effort-only. The inner candidate is FULL-InChIKey RT-gated below
        # (`_stereo_emit_decision` -> `(True, False)` only for full stereo at the
        # complete tier, then `_rt_match`), so an unverified junior parent is
        # never adopted -- the molecule still abstains. Caller matches this gate.
        if not self._general_fallback:
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            # Cheap prune: an alternative parent can only exist when a ring AND
            # an acyclic carbon chain compete. Molecules without both take a
            # pre-empt / single-candidate pool (parent_pool_size == 1) and have
            # nothing to offer -- skip them so the failure path spawns no inner.
            ri = mol.GetRingInfo()
            if ri.NumRings() < 1:
                return None
            acyclic_c = sum(
                1 for a in atoms_of(mol)
                if a.GetAtomicNum() == 6 and not a.IsInRing())
            if acyclic_c < 2:
                return None

            _ALT_PARENT_RESCUE.active = True
            try:
                pool_size = None
                rank = 1
                # Hard ceiling in case pool_size is somehow never populated;
                # real pools are 2-4, so this never truncates a genuine pool.
                while rank < 8:
                    inner = Orthonym(
                        style=self.style,
                        general_fallback=True,
                        general_fallback_unverified=True,
                        allow_aromatic_general=self._allow_aromatic_general,
                        _forced_parent_rank=rank,
                    )
                    cand = inner.name(smiles)
                    if pool_size is None:
                        pool_size = getattr(inner, '_top_parent_pool_size', 1)
                    if cand and not is_failure_name(cand):
                        # RT gate (load-bearing, 0-wrong) -- identical shape to
                        # the demote-senior-group rescue above.
                        _permitted, _flagged, _name_out = self._stereo_emit_decision(
                            mol, cand, smiles)
                        if _permitted:
                            cand = _name_out          # adopt composed/as-is reclaim
                            smi = _validity_gate_name_to_smiles(cand)
                            if smi is not None and self._rt_match(smiles, smi, _flagged):
                                return cand
                    rank += 1
                    if pool_size is not None and rank >= pool_size:
                        break
                return None
            finally:
                _ALT_PARENT_RESCUE.active = False
        except Exception as e:  # fail-closed: keep the abstention
            logger.info("alternate-parent rescue error (kept abstention): %s", e)
            _ALT_PARENT_RESCUE.active = False
            return None

    def _try_t4_rescue(self, mol, feats, smiles):
        """ Phase1 B5 general lever: build a PG-suppressed cascade name when
        the recovery lane's own rung-0 general name denoted a DIFFERENT molecule
        (RT-mismatch). Returns ``(name, stereo_flagged)`` for a POSITIVELY
        round-tripping name, or ``None``.

        Guarded exactly like the handoff above -- best-effort opt-in
        (``_general_fallback_unverified``) and a live OPSIN jar (or the test-mode
        gate bypass) -- so PIN/complete/valid tiers and jarless runs never reach
        it (the caller only invokes it on the best-effort RT-mismatch abstention
        path). Cannot recurse: ``name_t4_complete`` is a direct producer call,
        never ``name_tiered``. The name is E1+spine certified inside
        ``name_t4_complete`` AND re-verified by the positive RT check here, so a
        rescue can only ever turn an abstention into a right-molecule emission.
        """
        if not self._general_fallback_unverified:
            return None
        if not (self._disable_opsin_validity_gate or _validity_gate_jar_present()):
            return None
        try:
            from .assembly.fragment_naming import isolated_naming_session
            from .assembly.t4_coverage import name_t4_complete
            # Phase1 B5: isolated depth-0 session (see the handoff site) so
            # gets the full recursion budget a standalone call gets, rather
            # than the elevated session_depth of the enclosing name.
            with isolated_naming_session():
                cand = name_t4_complete(mol, feats)
        except Exception as e:  # fail-closed: a producer bug keeps the abstention
            logger.info("t4 rescue error (kept abstention): %s", e)
            return None
        if not cand or is_failure_name(cand):
            return None
        _permitted, _flagged, _name_out = self._stereo_emit_decision(
            mol, cand, smiles)
        if not _permitted:
            return None
        cand = _name_out          # adopt composed/as-is reclaim
        if self._disable_opsin_validity_gate:
            return (cand, _flagged)  # test/internal mode
        smi = _validity_gate_name_to_smiles(cand)
        if smi is not None and self._rt_match(smiles, smi, _flagged):
            return (cand, _flagged)
        return None

    def _name_multifragment_complete(self, mol, canonical_smiles: str) -> Optional[str]:
        """: name an all-neutral multi-fragment input under `complete`.

        Delegates to the adduct assembler (``rules.adducts.name_adduct``)
        with the per-component namer switched to the ``complete`` tier
        (``general_fallback`` + ``allow_aromatic_general``). name_adduct handles
        component ordering seniority / water-last), proportion notation,
        and — crucially — fails closed (returns None) if ANY component is
        unnameable OR charged (charged multi-fragment is the salt/ion router's
        and, when it too abstains, P5's scope). Returns the assembled name or
        None; never raises. Fires ONLY under `complete` (the caller gates on a
        multi-fragment mol + `allow_aromatic_general`)."""
        if not self._allow_aromatic_general:
            return None  # multi-fragment recovery is complete-tier only
        try:
            from .rules.adducts import name_adduct
            return name_adduct(
                mol, canonical_smiles, style=self.style,
                general_fallback=True, allow_aromatic_general=True,
                general_fallback_unverified=self._general_fallback_unverified)
        except Exception as e:
            logger.info("P4 multi-fragment recovery error (kept abstention): %s", e)
            return None

    def _finish(self, name: str, smiles: str) -> str:
        """ (fix wave 1): THE single exit of:meth:`name`.

        ``name`` has five ``return`` statements -- the isotope decorator
        exit, the isotope refusal, the late-recovery exit, the limit /
        ``--trivial`` fallback exit and the main exit. Hooking the
        binding-proof step at individual call sites left three of them
        unaudited (two of which ship a real, non-failure name), so every
        future exit would silently reopen the same hole. They all funnel
        through here instead, and
        ``tests/unit/validation/test_name_exits_are_audited.py`` parses the
        AST to keep it that way.

        Contract:

        * ``off`` mode is a byte-for-byte pass-through, established by the
          first branch below -- not by careful matching.
        * ``audit`` mode never changes the returned string (that is
          ``_apply_binding_proof``'s contract, unit-pinned).
        * This method NEVER raises. A bug in the audit must not turn a
          successful naming into a crash, exactly as ``_apply_binding_proof``
          and ``proof_ledger.finalize`` already guarantee internally; the
          belt-and-braces guard here also covers the ``is_top_level_naming``
          import, which the isotope exit reaches after its session ended.

        It deliberately depends on NOTHING the ``try:`` block in ``name``
        sets up -- only on ``self._binding_proof``, the naming depth and the
        two arguments -- so the isotope exit (which sits before that block)
        can route through it unchanged.
        """
        # D1 (coordination retained table): an N-coordinated metal tetrapyrrole
        # (heme/chlorophyll/cobalamin/siroheme/F430) has no OPSIN-parseable name, so
        # no real namer produces a verified name for it -- control lands on the metal
        # sentinel (via _descriptive_fallback, already hooked above) OR, when the
        # OPSIN validity gate is degraded, on a WRONG best-effort fragment that
        # bypasses _descriptive_fallback entirely. Override with the exact-InChIKey
        # ChEBI retained name HERE too, at name's single audited exit, so the table
        # wins deterministically for exactly these structures regardless of gate
        # state. The exact-InChIKey match (25 curated metal-macrocycles, none of them
        # PIN-nameable) means no correctly-named / PIN molecule is ever touched.
        # Placed before the emitted-record below so telemetry logs the final name.
        try:
            from .assembly.fragment_naming import is_top_level_naming as _d1_top
            if _d1_top():
                _d1_name = _coordination_retained_name(smiles)
                if _d1_name is not None:
                    name = _d1_name
        except Exception:  # pragma: no cover - a lookup must never break naming
            pass
        #: record the name actually returned, in its OWN try/except and
        # BEFORE the ``off`` short-circuit below -- otherwise the emitted record
        # would be silently absent in the default binding-proof mode, which is the
        # mode every measurement runs in.
        #
        # Recorded explicitly rather than inferred by looking the string up among
        # the candidates, because measured (PE1 plan, C10): the pool held
        # '(2R)-piperidine-2-carboxylic acid' while naming emitted
        # '(2R)-piperidine-2-carboxylate'. A lookup would report "emitted name not
        # among candidates" for every anion row.
        try:
            from .metrics import candidate_ledger as _cl
            if _cl.is_enabled():
                _cl.record_candidate("namer._finish", _cl.Stage.EMITTED, name)
        except Exception:  # pragma: no cover - telemetry must never break naming
            pass
        # a phase L0: producer-agnostic coverage AUDIT, SHADOW mode
        # (telemetry only -- NEVER changes `name`, only stores a verdict on
        # `self._last_coverage_verdict` and logs one line). Gated to the
        # top-level naming call, the same scope the gate itself uses
        # (`name`'s main exit calls `_final_opsin_validity_gate` only inside
        # its own `is_top_level_naming` block), so `_self01_lookup`'s
        # provenance read is never asked about a recursive fragment's state.
        # Skipped for a failure-name winner (an abstention makes no coverage
        # claim to audit, and its sentinel string would just waste an OPSIN
        # re-anchor call on the bare-str fallback path for nothing).
        try:
            _audit_mode = _coverage_audit_mode()
            if _audit_mode != "off":
                from .assembly.fragment_naming import is_top_level_naming as _cov_is_top_level
                if _cov_is_top_level() and not is_failure_name(name):
                    _cov_mol = Chem.MolFromSmiles(smiles) if smiles else None
                    if _cov_mol is not None:
                        from .assembly.coverage_audit import audit_coverage
                        _ger = getattr(self, "_last_ger_result", None)
                        # Staleness guard mirroring
                        # `provenance.resolve_gate_outcome`: only use the
                        # stashed GeneralEngineResult if it is FOR this exact
                        # name (a later rewrite -- e.g. the retained-name
                        # preference swap -- must not attach someone else's
                        # bindings to the string that actually ships).
                        _result_obj = (
                            _ger if _ger is not None
                            and getattr(_ger, "name", None) == name
                            else None)
                        if _result_obj is not None:
                            # GER path never needs -- E1 is Java-free
                            # and strictly more informative.
                            _self01, _skip_reanchor, _skip_detail = None, False, ""
                        else:
                            _self01, _skip_reanchor, _skip_detail = (
                                _self01_lookup(name))
                        verdict = audit_coverage(
                            _cov_mol, name, _result_obj,
                            self01_complete=_self01,
                            skip_reanchor=_skip_reanchor,
                            skip_detail=_skip_detail)
                        self._last_coverage_verdict = verdict
                        logger.info(
                            "coverage_audit mode=%s method=%s complete=%s "
                            "heavy_atoms=%d name=%r",
                            _audit_mode, verdict.method, verdict.complete,
                            _cov_mol.GetNumHeavyAtoms(), name[:80])
                        # a phase L1: turn the audit into a real VETO.
                        # Reject ONLY an incomplete winner whose method is
                        # NOT "unavailable" -- that method is the fail-OPEN
                        # signal (OPSIN absent, or a skip-reanchor carve-out /
                        # gate-disabled outcome per `_self01_lookup`), and
                        # vetoing there would abstain names that are
                        # otherwise fine (the CARRIED CONSTRAINT from L0).
                        # `shadow`/`off` never reach here with a veto effect:
                        # `off` short-circuited above (`_audit_mode != "off"`
                        # guards this whole block) and `shadow` simply never
                        # takes this branch.
                        if (_audit_mode == "veto" and not verdict.complete
                                and verdict.method != "unavailable"):
                            logger.info(
                                "coverage_audit VETO: method=%s detail=%r -- "
                                "abstaining instead of shipping %r",
                                verdict.method, verdict.detail, name[:80])
                            name = _descriptive_fallback(smiles)
                        # a phase L2/L4-core: wrap the current winner
                        # (post-veto, so its `.name` is exactly whatever
                        # `name` holds at this point -- the fallback string
                        # if L1 just fired, the original winner otherwise) as
                        # ONE `Offer` and run it through
                        # `select_rt_passing` -- the RT-GATE-OVER-OFFERS
                        # selection primitive (L4-core): the first offer, in
                        # rank order, that is both `.complete` AND passes the
                        # `_offer_rt_ok` predicate below. With a single offer
                        # this is a provable IDENTITY WHENEVER that offer's
                        # own `rt_ok` is True (the common case -- the primary
                        # winner was already gated upstream, so `_offer_rt_ok`
                        # reuses that recorded verdict via
                        # `_self01_lookup`, zero fresh OPSIN calls); if
                        # `select_rt_passing` returns `None` (no offer both
                        # complete and rt_ok), `_select_rt_passing_offer_name`
                        # below falls back to the CURRENT `name` unchanged --
                        # L4-core must never newly-abstain a name that ships
                        # today, only add the selection MECHANISM. This is
                        # the safety net L3-1's floor offer (and later
                        # capability offers) rely on: an added offer can
                        # never win un-RT-gated.
                        #
                        # `result_obj`/`is_pin`/`tier`/`source` are read from
                        # the SAME provenance `name_tiered` derives its labels
                        # from (`namer.py` `name_tiered`, ~:2984-3097) --
                        # re-evaluated fresh against the (possibly post-veto)
                        # `name` rather than reusing `_result_obj`/`_self01`
                        # above, which were computed against the PRE-veto
                        # name. `is_pin`/`tier` deliberately use the COARSE
                        # form the brief sanctions for L2 (a `True`/"pin_verified"
                        # default with the 3 known demotions --
                        # general_engine / trivial_retained /
                        # general_ring_prefix -- to `False`), since with one
                        # offer the ranking can never change the result
                        # regardless of how precisely tier is graded; L3/L4
                        # can sharpen this if a real second offer ever needs
                        # to out-rank it on tier.
                        # Tier labels -- paper semantics (2026-09-28): this tier is a
                        # ranking key only; `name_tiered` does not copy it when this
                        # primary offer wins (it reads prov["opsin"] alone, which the
                        # PIN path and the inline engine lane never record).
                        _offer_ger = getattr(self, "_last_ger_result", None)
                        _offer_result_obj = (
                            _offer_ger if _offer_ger is not None
                            and getattr(_offer_ger, "name", None) == name
                            else None)
                        from .metrics.provenance import get_provenance as _offer_get_prov
                        from .metrics.provenance import (
                            name_carries_non_pin_part as _offer_name_carries_non_pin_part)
                        _offer_prov = _offer_get_prov()
                        _offer_source = _offer_prov.get("source") or "pin_path"
                        if _offer_source == "general_engine":
                            _offer_is_pin = False
                            _offer_tier = (
                                SYSTEMATIC_VERIFIED
                                if _offer_prov.get("opsin") == "verified"
                                else BEST_EFFORT)
                        elif _offer_source == "trivial_retained":
                            _offer_is_pin, _offer_tier = False, SYSTEMATIC_VERIFIED
                        elif _offer_name_carries_non_pin_part(_offer_prov, name):
                            _offer_is_pin = False
                            _offer_tier = (
                                SYSTEMATIC_VERIFIED
                                if _offer_prov.get("opsin") == "verified"
                                else BEST_EFFORT)
                        else:
                            _offer_is_pin, _offer_tier = True, PIN_VERIFIED
                        from .assembly.offer_pool import Offer
                        self._offers = [Offer(
                            name=name, result_obj=_offer_result_obj,
                            is_pin=_offer_is_pin, tier=_offer_tier,
                            source=_offer_source, complete=verdict.complete)]
                        # a phase L3-1: the systematic FLOOR as a 2nd,
                        # RT-gated offer -- best-effort ONLY (structurally
                        # impossible under --emit-tier pin/complete: this
                        # whole call is gated behind
                        # `_general_fallback_unverified`, the SAME flag
                        # `name_t4_complete`'s pre-existing caller requires).
                        # See `_maybe_append_t4_floor_offer`'s docstring for
                        # the cost guard (skips the floor entirely when the
                        # primary already full-RT-passes) and the PIN-first
                        # byte-identity argument (a pin-strict run never
                        # reaches this line at all).
                        if (self._general_fallback_unverified
                                and not getattr(self, '_suppress_floor_offer', False)):
                            self._maybe_append_t4_floor_offer(
                                _cov_mol, smiles, name,
                                primary_is_failure=False)
                        name = self._select_rt_passing_offer_name(name, smiles)
                elif (_cov_is_top_level() and is_failure_name(name)
                        and self._general_fallback_unverified
                        and not getattr(self, '_suppress_floor_offer', False)):
                    # a phase L3-1: the CRITICAL wiring wrinkle the task
                    # brief flags -- the block above never runs for a
                    # failure-name sentinel (its own guard,
                    # `not is_failure_name(name)`, is false here: auditing a
                    # sentinel's "coverage" or vetoing it would be
                    # meaningless). This narrower sibling path seeds
                    # `self._offers` with ONLY the floor candidate (a
                    # sentinel is not itself an offer) and lets
                    # `_select_rt_passing_offer_name` ship the floor when it
                    # full-RT-passes, or fall back to the UNCHANGED sentinel
                    # otherwise -- WITHOUT the meaningless `audit_coverage`
                    # call on the sentinel string.
                    _cov_mol = Chem.MolFromSmiles(smiles) if smiles else None
                    if _cov_mol is not None:
                        self._offers = []
                        self._maybe_append_t4_floor_offer(
                            _cov_mol, smiles, name, primary_is_failure=True)
                        if self._offers:
                            name = self._select_rt_passing_offer_name(
                                name, smiles)
        except Exception as _cae:  # pragma: no cover - defensive
            # SHADOW must never break a name regardless of what goes wrong
            # inside the audit; a VETO must fail OPEN on its own internal
            # error too; and (a phase L2) a failure while building/
            # selecting the offer must never change or crash the name either.
            # `name` is only ever reassigned above, inside this same try
            # block, and each reassignment (veto, then offer-selection) fully
            # completes before the next statement runs -- so an exception
            # anywhere in this block leaves `name` at whichever of those
            # values it last finished assigning (its original winner if
            # nothing fired yet, the L1 fallback if veto fired and the
            # exception came after, etc.), never a half-applied one.
            logger.info("coverage audit failed (shadow, ignored): %s", _cae)
        # Claims conformance part 2 (2026-09-27): every name exit returns only a
        # name whose own round trip passed (exact-match list names and the
        # documented default-tier carve-outs excepted); see _exit_round_trip_check.
        # After the offer selection, so an offer that won on an advisory fail-open
        # branch of _offer_rt_ok is checked too.
        try:
            name = self._exit_round_trip_check(name, smiles)
        except Exception as _xe:  # pragma: no cover - defensive, fails closed
            logger.info("exit round-trip check failed (withdrawn): %s", _xe)
            name = _descriptive_fallback(smiles)
        # D3 (user decision 2026-09-28): the best-effort tier's last resort for a
        # molecule it is about to leave unnamed -- after every other path and the
        # exit check, so it can only turn an abstention into a name, never change
        # a name that ships. See _maybe_prefix_order_fallback.
        try:
            name = self._maybe_prefix_order_fallback(name, smiles)
        except Exception as _pe:  # pragma: no cover - defensive, keeps the abstention
            logger.info("prefix-order fallback failed (abstention kept): %s", _pe)
        # Review fixes (the paper, L74: "At the best-effort tier, a name that OPSIN
        # rejects is not emitted; the only exceptions are the exact-match list names
        # described below"): at the general tiers the natural-product lookup gives
        # way to the systematic name, which the gate verifies; when no
        # verified name was found, the exact-match natural-product parent name of the
        # whole input ('germacrane') is the last resort before a decline, as the
        # metal-complex list name is (above). Only an abstention turns into a name.
        try:
            from .assembly.fragment_naming import is_top_level_naming as _np_top
            from .assembly.fragment_naming import name_scope_depth as _np_depth
            if (self._general_fallback and is_failure_name(name) and _np_top()
                    and _np_depth() == 1):
                _np_name = _np_exact_list_name(smiles)
                if _np_name is not None:
                    from .metrics import provenance as _np_pv
                    _np_pv.record_source("pin_path")
                    _record_gate_outcome(
                        _np_pv.carveout_outcome("np_stereoparent"), _np_name)
                    name = _np_name
        except Exception as _npe:  # pragma: no cover - defensive, keeps the abstention
            logger.info("natural-product list fallback failed (abstention kept): %s", _npe)
        try:
            if self._binding_proof == "off":
                return name
            from .assembly.fragment_naming import is_top_level_naming
            if not is_top_level_naming():
                return name
            return self._apply_binding_proof(name, smiles)
        except Exception as _fe:      # pragma: no cover - defensive
            logger.info("binding-proof finish failed (ignored): %s", _fe)
            return name

    def _select_rt_passing_offer_name(self, current_name: str,
                                       smiles: str) -> str:
        """ a phase L4-core: run `select_rt_passing` over `self._offers`
        (already built by the caller) with an `_offer_rt_ok`-backed predicate,
        and return the winner's name -- or `current_name` UNCHANGED if no
        offer is both `.complete` and `rt_ok` (never newly-abstain a name
        that ships today; that is L4-core's whole contract, not this
        function's choice to make).

        Factored out of `_finish` as its own method (rather than inlined)
        so it is independently testable with a hand-built `self._offers`
        pool and a monkeypatched `_offer_rt_ok`, without needing a second
        real offer to exist yet (L3-1 is what will add one).
        """
        from .assembly.offer_pool import select_rt_passing

        def _rt_ok(offer):
            return _offer_rt_ok(offer.name, smiles)

        selected = select_rt_passing(self._offers, _rt_ok)
        logger.info(
            "select_rt_passing considered %d offer(s); winner=%r",
            len(self._offers), selected.name if selected is not None else None)
        # a phase cleanup T1: stash the winning Offer (or None when no
        # offer both completed and RT-passed, in which case `current_name`
        # ships unchanged and there is no "winner" to report) so `name_tiered`
        # can read its tier/is_pin/source instead of the stale contextvar.
        self._last_selected_offer = selected
        return selected.name if selected is not None else current_name

    def _maybe_append_t4_floor_offer(self, mol, smiles: str,
                                      primary_name: str,
                                      primary_is_failure: bool) -> None:
        """ a phase L3-1: append the systematic floor
        (`assembly.t4_coverage.name_t4_complete`) to `self._offers` as a 2ND
        `Offer`, IF the primary winner does not already full-RT-pass.

        THE breadth lever this task exists for: the L3 SIZING measured that
        the floor already builds a correct, full-round-tripping name for
        36/86 single-fragment-noncharged gap rows the pipeline otherwise
        emits WRONG (34) or abstains on (2) -- reaching emission for only 1.
        Appending it here (ranked and RT-gated by `_select_rt_passing_offer_name`
        right after this call returns) is what lets it WIN instead of just
        existing unreachably.

        Best-effort ONLY BY CONSTRUCTION: both call sites in `_finish` only
        invoke this method inside an `if self._general_fallback_unverified`
        guard -- the SAME flag `name_t4_complete`'s pre-existing caller
        (`_try_general_engine_recovery`, ~:3414) requires, so a PIN/complete/
        valid-tier run can never reach `name_t4_complete` from here either.

        Cost guard: `name_t4_complete` recurses the full naming machinery
        (perception + classification + the producer cascade), so it is
        only computed when it might actually be NEEDED --

        * ``primary_is_failure=False`` (the ordinary "primary emitted a real
          name" call site): `_offer_rt_ok(primary_name, smiles)` is checked
          FIRST -- a cheap, MEMOIZED lookup (reuses the gate's own
          OPSIN call for `primary_name`, no new JVM invocation) -- and the
          floor is skipped entirely when the primary already full-RT-passes.
          This is the common case, and exactly what keeps a PIN winner that
          already round-trips exact untouched (the "PIN-first preserved"
          guard).
        * ``primary_is_failure=True`` (the primary is already a failure-name
          sentinel, e.g. 'unknown organic compound'): skips straight to
          computing the floor without probing `_offer_rt_ok` on the sentinel
          first -- a sentinel can never itself round-trip, and probing it
          would be a WASTED fresh OPSIN call the sentinel was never gated
          against in the first place (`_final_opsin_validity_gate` returns
          early on a known descriptive-fallback string, `namer.py`
          ~:1194-1198, without ever calling `_validity_gate_name_to_smiles`
          on it -- so there is no cached verdict to reuse here either).

        `name_t4_complete` is run inside `isolated_naming_session` (mirrors
        `_try_general_engine_recovery`'s existing call site, ~:3442-3443):
        called mid-`name` at `session_depth >= 1` it would hit
        `MAX_NAMING_DEPTH` prematurely and degrade to an abstention
        (documented root cause, `t4_coverage.py` ~:292-299 / RE-CONFIRMED at
        `_try_general_engine_recovery` above) -- the isolated session gives
        it the full recursion budget a standalone call gets.

        `complete` is derived via `audit_coverage` exactly like the primary
        offer above (bare-str path, `result_obj=None`) -- NOT assumed True
        from `name_t4_complete`'s own internal E1 certification, so a floor
        candidate is held to the SAME audited bar as everything else in the
        pool. The REAL 0-wrong safety net is `_offer_rt_ok` (full-InChIKey)
        at selection time, not this flag: a floor that is `complete` but
        fails `rt_ok` still cannot win (`select_rt_passing` requires both).

        Never raises -- a producer bug here must leave `self._offers` exactly
        as it already was (fail-closed), matching this whole `_finish` block's
        contract that a coverage-audit failure must never change or crash the
        name.
        """
        if mol is None:
            return
        # Require an ACTUAL, LIVE way to verify a brand-new candidate against
        # the primary before ever computing one. Two distinct guards, both
        # measured necessary (a caught regression each):
        #
        # * `self._disable_opsin_validity_gate` (instance flag, "test/internal
        # mode"): when set, `_final_opsin_validity_gate` is never called for
        # the PRIMARY on its own naming path either (namer.py ~:4296's own
        # `if not self._disable_opsin_validity_gate:` guard), so no gate
        # outcome is recorded for it and `_self01_lookup` cannot distinguish
        # "never checked because gate is off" from "checked and belongs to
        # a stale, unrelated string" (`resolve_gate_outcome`'s ``bypassed``
        # case, deny-by-default but NOT in `_self01_lookup`'s skip_reanchor
        # bucket) -- so `_offer_rt_ok(primary_name,...)` could go either
        # way depending on provenance left over from a PRIOR, unrelated
        # `.name` call in the same process (measured:
        # `tests/unit/validation/test_binding_proof_wiring.py::
        # test_a_retained_name_swap_after_the_certificate_is_reported`,
        # order-dependent without this guard). Skipping the floor mechanism
        # entirely when the gate is explicitly off keeps that mode's
        # pre-L3-1 behaviour unchanged -- exactly what "test/internal mode"
        # should mean for a NEW OPSIN-gated capability.
        # * A missing jar: mirrors the EXISTING producer's own requirement
        # (`_try_general_engine_recovery` ~:3431-3433, `_try_t4_rescue`
        # ~:3646 -- "Keep the abstention without a jar") -- a name must
        # never ship UNVERIFIED. Without this, `_offer_rt_ok` fails OPEN
        # (True) for a candidate NO ONE has ever checked, which would let a
        # von-Baeyer name that must NEVER ship win purely because there was
        # nothing to disprove it with (measured:
        # `tests/unit/test_general_fallback_wiring.py::
        # test_mancude_refused_regardless_of_optin_no_jar`).
        if self._disable_opsin_validity_gate or not _validity_gate_jar_present():
            return
        if not primary_is_failure and _offer_rt_ok(primary_name, smiles):
            return  # primary already full-RT-passes -- the floor is not needed
        try:
            from .assembly.fragment_naming import isolated_naming_session
            from .assembly.t4_coverage import name_t4_complete
            canonical = Chem.MolToSmiles(mol, canonical=True)
            with isolated_naming_session():
                feats = self._perceive(mol, smiles, canonical)
                self._classify(feats)
                floor_name = name_t4_complete(mol, feats)
            if not floor_name or is_failure_name(floor_name):
                return
            from .assembly.coverage_audit import audit_coverage
            _self01, _skip_reanchor, _skip_detail = _self01_lookup(floor_name)
            verdict = audit_coverage(
                mol, floor_name, None, self01_complete=_self01,
                skip_reanchor=_skip_reanchor, skip_detail=_skip_detail)
            from .assembly.offer_pool import Offer
            self._offers.append(Offer(
                name=floor_name, result_obj=None, is_pin=False, tier=BEST_EFFORT,
                source="t4_floor", complete=verdict.complete))
            self._t4_floor_candidate = floor_name
            logger.info(
                "t4 floor offer: name=%r complete=%s (method=%s)",
                floor_name[:80], verdict.complete, verdict.method)
        except Exception as _fe:  # pragma: no cover - defensive
            logger.info(
                "t4 floor offer computation failed (kept existing offers): %s",
                _fe)

    def _maybe_prefix_order_fallback(self, name: str, smiles: str) -> str:
        """D3 (user decision 2026-09-28): the best-effort tier's last resort for a
        molecule whose round trip depends on the citation order of its prefixes.

        The round trip (OPSIN's SMILES of the name, read by RDKit, full InChIKey)
        can depend on the prefix order although the name's meaning does not
        , the Blue Book, "Alphanumerical order is used to establish the
        order of citation of detachable substituent prefixes";:3446, the order
        does not involve stereochemical descriptors). Measured on the P(III) centre
        of PubChem-500k row
        ``Cc1cn([C@H]2CC(O[P@]3O[C@](C)[C@@H]4CCCN43)[C@@H](CO)O2)c(=O)[nH]c1=O``:
        OPSIN's own StdInChIKey is the same for both orders, but its SMILES carries
        a ring-closure digit on the three-coordinate P in one order and not in the
        other, and RDKit reads a lone-pair stereocentre that carries a ring-closure
        digit with the opposite configuration from the OpenSMILES reading that CDK
        (the ``centres`` labeller) and OPSIN use -- as it reads the input, which
        carries one too. So the -ordered name never round-trips and the
        molecule had no name. ``t4_coverage.name_prefix_order_fallback`` builds the
        universal floor's name with the parent's prefixes in another order when the
         spelling fails its round trip at the stereo layer only.

        Runs only when ``name`` is a failure name at the end of the outermost
        best-effort ``name`` (after the exit check), so no name that ships
        changes and the PIN, valid and complete tiers never reach it. The name
        ships only when it passes every check a shipped best-effort name passes --
        the coverage audit, the full-InChIKey round trip of this string
        (``_shipped_name_round_trip``), the stereodescriptor check
        (``_stereo_descriptors_verified``: every descriptor is the engine's CIP
        label at its unit) and the exit check -- AND
        ``_cip_labels_match_input_as_written``: the engine's CIP labels equal an
        independent OpenSMILES-conformant reading of the input string (CDK). A
        round trip that passes through the RDKit reading on one side only (an
        input without a ring-closure digit at the centre, read as written, and a
        reordered name whose SMILES has one) would otherwise ship the other
        stereoisomer's descriptors. It breaks on purpose, so it is never a
        PIN: the winning offer is the last-resort floor's (source ``t4_floor``,
        tier best_effort, ``is_pin`` False), and ``name_tiered`` reports
        ``prefix_order_fallback`` True for it. Otherwise returns ``name``
        unchanged. Never raises out of the caller's guard."""
        if name and not is_failure_name(name):
            return name
        if (not self._general_fallback_unverified
                or getattr(self, '_suppress_floor_offer', False)
                or self._disable_opsin_validity_gate or _DISABLE_VALIDITY_GATE
                or not smiles or not _validity_gate_jar_present()):
            return name
        from .assembly.fragment_naming import (
            is_top_level_naming, isolated_naming_session, name_scope_depth)
        if not (name_scope_depth() == 1 and is_top_level_naming()):
            return name
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not (
                any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
                    for a in mol.GetAtoms())
                or any(b.GetStereo() != Chem.BondStereo.STEREONONE
                       for b in mol.GetBonds())):
            return name  # no stereo unit: the fallback cannot apply
        from .assembly.fragment_naming import PerfBudgetExceeded
        from .assembly.t4_coverage import name_prefix_order_fallback
        try:
            with isolated_naming_session():
                reordered = name_prefix_order_fallback(mol)
        except PerfBudgetExceeded:
            # the molecule's hang budget ran out inside the floor (a BaseException,
            # so the floor's own guard does not catch it): the abstention stands.
            return name
        if not reordered or is_failure_name(reordered):
            return name
        if _shipped_name_round_trip(reordered, smiles) != "verified":
            return name
        from .rules.stereochemistry import strip_stereo_blocks
        if not _stereo_descriptors_verified(
                reordered, smiles, strip_stereo_blocks(reordered)[0]):
            return name
        if not _cip_labels_match_input_as_written(smiles):
            return name
        from .assembly.coverage_audit import audit_coverage
        _self01, _skip_reanchor, _skip_detail = _self01_lookup(reordered)
        verdict = audit_coverage(
            mol, reordered, None, self01_complete=_self01,
            skip_reanchor=_skip_reanchor, skip_detail=_skip_detail)
        if not verdict.complete:
            return name
        if self._exit_round_trip_check(reordered, smiles) != reordered:
            return name
        from .assembly.offer_pool import Offer
        self._last_selected_offer = Offer(
            name=reordered, result_obj=None, is_pin=False, tier=BEST_EFFORT,
            source="t4_floor", complete=True)
        self._prefix_order_fallback_name = reordered
        logger.info("prefix-order fallback named %r", reordered[:80])
        return reordered

    def _record_binding_proof(self, mol, eng, stage: str) -> None:
        """: record a certified general-engine spine on the ledger.

        Shared by the two producer sites (the inline general-engine fallback
        and the late recovery) so the record block exists once. No-op unless
        the proof flag is on, and every failure is swallowed: recording is
        observation, and a bug in it must never break naming.

        The imports stay lazy and inside the body on purpose --
        ``tests/unit/test_stereo_tier_policy.py`` monkeypatches the
        ``general_engine`` / ``e1_certificate`` modules and the surrounding
        call sites only tolerate that because nothing here is imported at
        module scope.
        """
        # a phase L0: stash the raw GeneralEngineResult for the SHADOW
        # coverage audit at `_finish`, INDEPENDENT of `self._binding_proof`
        # below -- that flag gates a separate (older) ledger feature, and the
        # audit needs `eng` itself (for `certify_general_result`), not the
        # ledger's derived proof. `_finish` re-checks `eng.name == name`
        # before using it, so a later rewrite of the string (e.g. the
        # retained-name preference swap right after this call) is not
        # mistaken for a certified GeneralEngineResult winner.
        self._last_ger_result = eng
        if self._binding_proof == "off":
            return
        try:
            from .validation.binding_spine import BindingSpine
            from .validation.proof_ledger import record_spine
            record_spine(
                mol, BindingSpine.from_token_bindings(
                    eng.bindings,
                    stereo_atom_to_locant=getattr(
                        eng, 'stereo_atom_to_locant', None)),
                stage=stage,
                name_at_record=eng.name,
                allow_charged=self._allow_aromatic_general)
        except Exception as _pe:
            logger.info("binding-proof record failed (ignored): %s", _pe)

    def _apply_binding_proof(self, name: str, smiles: str) -> str:
        """ a phase: assert the recorded binding spine on the FINAL name.

        audit -> observe and log only (the name is returned unchanged).
        enforce-> a failed proof abstains (fail closed). No default path sets
                  enforce in a phase.

        ``finalize`` is always called with ``mode="audit"``, including under
        enforce: strict mode escalates the three UNPROVEN-not-disproven codes
        (CHARGE_UNVERIFIED / UNBOUND_MORPHEME / PROOF_UNSUBSTANTIATED) to
        errors, and abstaining on missing evidence rather than on a proved
        disagreement would refuse correct names wholesale.

        Every failure mode here returns ``name`` untouched: a bug in the audit
        must never turn a successful naming into a crash or an abstention.

        An ABSTENTION is not finalized at all. When a producer recorded a
        spine but a downstream gate then suppressed the emission, the string
        reaching this point is the failure sentinel, not a name -- and
        asserting a spine against 'unknown organic compound' reports
        TOKEN_ABSENT for every token while saying nothing about proof
        coverage. The proof answers "does the SHIPPED NAME spell the graph?";
        with nothing shipped the question does not arise. Measured on 250
        corpus rows: this artefact was 9 of 15 recorded spines, i.e. it would
        have been the majority of the Task 7 census. The record is left on the
        ledger unfinalized (``ok is None``), which is the honest state --
        "recorded, never shipped" -- and is distinguishable from a real
        failure (``ok is False``).
        """
        if is_failure_name(name):
            return name
        try:
            from .validation.proof_ledger import finalize
            proof = finalize(name, mode="audit")
        except Exception as _pe:
            logger.info("binding-proof finalize failed (ignored): %s", _pe)
            return name
        if proof is None:
            return name
        if not proof.ok:
            logger.info("binding-proof findings for %s: %s", smiles,
                        proof.codes())
            if self._binding_proof == "enforce":
                return _descriptive_fallback(smiles)
        return name

    def _default_tier_policy_applies(self) -> bool:
        """The default tier's emission rule applies to this call: no breadth flag
        (``--emit-tier pin``), the round-trip gate on with its jar, not the strict
        twin of a wider tier, and the caller's own call (read by ``_budget_scope``
        at name-scope depth 1; a nested ``name`` names a part of a molecule)."""
        # Review fixes: the paper's default configuration is the PIN style
        # ('--style general' keeps its general forms), and the rule does not depend on
        # the OPSIN jar (in the opt-in reduced mode a strict-path name in PIN form is
        # emitted unverified, every other name declined). Only the raw-output
        # switches that turn the validity gate off (the private constructor flag and
        # ORTHONYM_DISABLE_OPSIN_VALIDITY_GATE) leave it off.
        if (self.style != "pin"
                or self._general_fallback or self._general_fallback_unverified
                or self._allow_aromatic_general or self._full_coverage
                or self._disable_opsin_validity_gate or _DISABLE_VALIDITY_GATE
                or _DEFAULT_TIER_POLICY_OFF.get()):
            return False
        from .assembly.fragment_naming import is_top_level_naming
        return is_top_level_naming()

    def _default_tier_decision(self, name, smiles: Optional[str],
                               raise_on_limit: bool = False):
        """Paper conformance (user decision 2026-09-30): the default tier emits a
        name only when it is pin_verified -- "The label pin_verified means the
        strict PIN path built the name and verified it" -- or one of the default
        tier's exceptions (:func:`_default_tier_emits`); "The default configuration
        emits a name only when the pipeline can build the preferred IUPAC name
        (PIN); otherwise, it declines" (the submitted text, Methods, "Tiers"). Any
        other name is declined: the call returns the structure's decline label
        (raises ``OrthonymLimitError`` NO_VERIFIED_PIN with ``raise_on_limit``),
        and ``name_tiered`` reports tier ``abstain`` with limit code
        NO_VERIFIED_PIN. The wider tiers keep every name. The row derived here is
        kept for ``name_tiered`` (``_default_tier_row``)."""
        if not name or is_failure_name(name) or not smiles:
            return name
        from .metrics.provenance import get_provenance
        row, pin_form = self._tier_row_and_pin_form(smiles, name, get_provenance())
        if _default_tier_emits(row, pin_form, smiles, self._trivial_fallback):
            self._default_tier_row = (name, row)
            return name
        label, declined = _default_tier_declined_row(smiles)
        self._default_tier_row = (label, declined)
        logger.info("default tier declined %r (tier %s): not a verified PIN",
                    name, row.get("tier"))
        if raise_on_limit:
            raise no_verified_pin(label, smiles=smiles)
        return label

    def _pin_promotion_eligible(self) -> bool:
        """The default (PIN) tier in its production configuration, at top level:
        no breadth flag, the round-trip gate on with its jar, the PIN style, and
        this ``name`` is the outermost one (``name_scope_depth == 1``: its own
        ``_budget_scope`` has entered, no enclosing ``name`` runs).

        ``is_top_level_naming`` alone is not "the caller's molecule": it reads
        the fragment-recursion depth, which is 0 while ANY ``name`` dispatches
        its whole molecule and which ``isolated_naming_session`` resets to 0 for
        the recovery lane, the rescues and the clean fall-through. So a helper
        instance in the default configuration, created by a producer of an outer
        call at any tier, read as top level and named what its first run could
        not name a second time with the promoted producers: the peptide handler's
        substitutive sub-namer (``dispatch_table._handle_peptide``) and the
        default-tier ``name_compound`` instances of the glycoside aglycone namer
        and of the clean fall-through. Measured on a 119-heavy-atom ChEBI
        glycopeptide at the best-effort tier: 52 wrapper calls in nested
        ``name`` frames (32 from the peptide sub-namer, 1 from the peptide-acid
        namer of an amido prefix, 19 from the aglycone namer); the nine ChEBI
        peptides the paper recipe lost to its 90 s stall took 102-152 s alone
        against 60-79 s before the re-run existed. At the PIN tier those helpers
        never re-ran (``_PIN_PROMOTION_WRAPPED`` is set for the whole outer
        call), so the PIN tier is unchanged: the re-run is the PIN tier's policy
        for the caller's call, not for a helper of another tier. (Letting a helper
        that names the whole molecule keep it -- the peptide sub-namer -- gave one
        dev-set row its PIN at best-effort, but put 3-11 s back on the ChEBI
        peptides, 116-209 heavy atoms, in the recipe; not done.)"""
        if (self._general_fallback or self._general_fallback_unverified
                or self._allow_aromatic_general or self._full_coverage
                or self.style != "pin" or self._principal_group_override is not None
                or self._disable_opsin_validity_gate or _DISABLE_VALIDITY_GATE):
            return False
        from .assembly.fragment_naming import is_top_level_naming, name_scope_depth
        return (name_scope_depth() == 1 and is_top_level_naming()
                and _validity_gate_jar_present())

    def _name_with_pin_promotion(self, smiles: str, *, raise_on_limit: bool) -> str:
        """Breadth Job 1 (M01): the PIN tier names the molecule as before; only when
        that run returns no name does it name it again with the best-effort ring-
        substituent producers admitted, each name kept only when its vocabulary is
        all PIN vocabulary (``rules.pin_vocabulary.promote_at_pin_tier``) -- e.g.
        '[2-(thiophen-2-yl)-1,3-thiazol-4-yl]methanamine',
        the Blue Book), '(4-methylcyclohex-3-en-1-yl)' (e),:3288).

        A molecule the first run names keeps that name byte for byte, so a route
        that becomes reachable only through a promoted prefix can never pre-empt
        the route that names it today. The second run goes through every gate the
        first does (the round trip included) and its memo entries are keyed apart
        (``assembly.memo.pin_promotion_var``). When it also returns no name, the
        first run's result and provenance are what the caller sees.

        A kept second-run name is labelled pin_unverified with ``is_pin`` False by
        ``name_tiered`` (``metrics.provenance.record_pin_promotion_rerun``): a
        breadth producer built it, so it is not certified as the PIN.
        """
        from .assembly.memo import pin_promotion_var
        from .metrics import provenance as _pv
        wrapped = _PIN_PROMOTION_WRAPPED.set(True)
        try:
            from .assembly.fragment_naming import PerfBudgetExceeded
            from .rules.pin_vocabulary import promotion_could_change, promotion_site_probe
            limit_error = None
            with promotion_site_probe() as _sites:
                try:
                    first = self.name(smiles, raise_on_limit=raise_on_limit)
                except OrthonymLimitError as _le:
                    # raise_on_limit: the first run found no name. The re-run gets
                    # its chance; the limit is raised only when it finds none either.
                    first, limit_error = None, _le
            if first and not is_failure_name(first):
                return first
            if not promotion_could_change(_sites):
                # Branch review fixes (perf): the re-run would repeat the first run
                # and find no name either (``rules.pin_vocabulary.
                # promotion_could_change``): the first run reached no site whose
                # behaviour the re-run changes -- e.g. an input with a wildcard atom,
                # which is declined before any producer runs -- or only promotion
                # calls whose promoted producers, previewed in the context each call
                # ran in, keep no name.
                if limit_error is not None:
                    raise limit_error
                return first
            snap = _pv.get_provenance()
            promoted = pin_promotion_var.set(True)
            second = None
            try:
                _pv.clear_provenance()
                second = self.name(smiles)
            except PerfBudgetExceeded:
                # Branch review fixes: the re-run shares the hang budgets the first
                # run armed (this wrapper runs in the outermost name, the frame
                # that owns them), so it can exhaust what the first run left. That
                # is "no second name": the first run's result, provenance and limit
                # stand, so ``raise_on_limit=True`` still raises OrthonymLimitError
                # (the trip used to unwind to the outermost boundary, which returned
                # the failure label instead).
                logger.info("PIN promotion re-run exhausted the hang budget; first "
                            "result kept")
                second = None
            except Exception as _pe:  # noqa: BLE001 - the re-run must never break naming
                logger.info("PIN promotion re-run failed (first result kept): %s", _pe)
                second = None
            finally:
                pin_promotion_var.reset(promoted)
            if second and not is_failure_name(second):
                # Kept only in PIN form: a round trip passed for THIS string, and the
                # name carries no fragment or token that labels it below the PIN
                # (``name_carries_non_pin_part``: 'ethanamide', '1-oxacyclopentan-
                # 2-yl',...), and the molecule is no multiplicative candidate. A
                # re-run name that fails any of these is dropped and the molecule
                # stays as the first run left it.
                _prov = _pv.get_provenance()
                _outcome = _pv.resolve_gate_outcome(
                    _prov["gate_outcome"], _prov["gate_outcome_name"], second)
                from .rules.pin_vocabulary import multiplicative_candidate
                _mol_mc = Chem.MolFromSmiles(smiles)
                if (_pv.opsin_label_for_gate_outcome(_outcome) == "verified"
                        and (_prov["source"] or "pin_path") == "pin_path"
                        and not _pv.name_carries_non_pin_part(
                            _prov, second, label_forms=False)
                        and _mol_mc is not None
                        and not multiplicative_candidate(_mol_mc)):
                    # Branch review fixes: a kept re-run name is NOT certified as the
                    # PIN. The promoted producers are the best-effort composition
                    # branches (``allow_mancude``), and the vocabulary guard is a
                    # closed list, so a re-run name can be a right-molecule name that
                    # the Blue Book does not prefer: 'methyl...-3-(methoxycarbonyl)
                    # pent-3-enoate' for 'dimethyl...butanedioate',
                    # the Blue Book), an oxolan-2-one parent beside a senior
                    # 1-benzopyran-4-one,:29628), '(1-sulfanylideneethyl)'
                    # for 'ethanethioyl',:30246). The paper's definition
                    # (tier labels): "pin_unverified means a name in PIN form that
                    # only a breadth producer built". So the name ships labelled
                    # pin_unverified, is_pin False (``name_carries_non_pin_part``).
                    _pv.record_pin_promotion_rerun()
                    return second
                logger.info("PIN promotion re-run name %r is not a verified PIN form; "
                            "first result kept", second)
            _pv.restore_provenance(snap)
            if limit_error is not None:
                raise limit_error
            return first
        finally:
            _PIN_PROMOTION_WRAPPED.reset(wrapped)

    def _exit_round_trip_check(self, name: str, smiles: str) -> str:
        """The check every public naming entry point applies to the name it is about
        to return: the round trip (:meth:`_exit_round_trip_check_rt`), then, for an
        input with a lone-pair stereocentre, the configuration check that does not go
        through RDKit's reading (:meth:`_exit_lone_pair_check`)."""
        return self._exit_lone_pair_check(
            self._exit_round_trip_check_rt(name, smiles), smiles)

    def _exit_lone_pair_check(self, name: str, smiles: str) -> str:
        """Lone-pair stereocentres (TRIAGE.md 'Lone-pair stereocentres -- RDKit
        reading'): withdraw a name that passed its round trip but names the other
        configuration at a lone-pair stereocentre than the input as written
        (:func:`_lone_pair_configuration_verified`). The round trip compares RDKit's
        readings of the input and of OPSIN's SMILES, so an RDKit misreading shared by
        both sides passes it; this check reads both strings with CDK instead.

        Top level only, on a name the round trip kept, at every tier. Passes
        unchanged: a failure name, an input without a lone-pair stereocentre, a
        disabled gate (the documented raw-output configuration), an exact-match list
        name (identity by the input's InChIKey; OPSIN cannot read it) and the
        no-jar reduced mode (no OPSIN to read the name; such names are labelled
        unverified). Anything else that fails or cannot be checked is withdrawn."""
        from .assembly.fragment_naming import is_top_level_naming
        if not name or is_failure_name(name) or not is_top_level_naming():
            return name
        # Only the string the caller gave (``_caller_input``; outside any name a
        # public entry point's own argument). A nested name of another string --
        # the acid of an ester, a salt's component -- names a SMILES RDKit wrote from
        # its own reading, so its configuration is RDKit's by construction; the whole
        # name is checked against the caller's string. Measured: naming
        # 'CCCCCCC#CCO[S@](=O)c1ccc(C)cc1' runs a nested name of
        # 'Cc1ccc([S@@](=O)O)cc1' whose name '(S)-4-methylbenzene-1-sulfinic acid'
        # OPSIN does not read; a check there withdrew it, and with it the ester's
        # right name 'non-2-yn-1-yl (S)-4-methylbenzene-1-sulfinate' (pin_verified,
        # the same configuration for CDK as the input).
        _caller = _caller_input()
        if _caller is not None and smiles != _caller:
            return name
        # The caller's string as written: when a lone-pair centre of it is read by
        # RDKit unlike the standard reading, the public call named its standard
        # spelling instead (_lone_pair_standard_spelling), and the name is checked
        # against the string the caller gave (TRIAGE.md 'Lone-pair centre written
        # first -- naming and the gate probe').
        _written = _lone_pair_as_written(smiles)
        if not _has_lone_pair_stereocentre(_written):
            return name
        if self._disable_opsin_validity_gate or _DISABLE_VALIDITY_GATE:
            return name
        try:
            if _is_exact_match_list_name(smiles, name):
                return name
            if not _validity_gate_jar_present():
                return name
            if _lone_pair_configuration_verified(name, _written):
                return name
            return self._withdraw_at_exit(name, smiles, "exit_lone_pair_configuration")
        except Exception as _le:  # pragma: no cover - defensive, fails closed
            logger.info("lone-pair configuration check failed (withdrawn): %s", _le)
            return _descriptive_fallback(smiles)

    def _exit_round_trip_check_rt(self, name: str, smiles: str) -> str:
        """Claims conformance part 2 (2026-09-27): the check every public naming
        entry point applies to the name it is about to return, with no fail-open
        branch where a check is possible.

        Some names reach an exit without a round trip that passed for THIS string:
        the trivial last resort (`_apply_trivial_fallback`) returns a name after the
        gate suppressed another string and never gates it; an offer can win on one
        of `_offer_rt_ok`'s advisory fail-open branches (a carve-out outcome, no
        jar, no input, an inconclusive key compare); and ships an
        'inconclusive' compare. With OPSIN unavailable (a timeout on a loaded host,
        the jar present) the trivial last resort returned 'benzol' for benzene at
        the default AND best-effort tiers of ``name``, ``name_compound`` and
        ``name_with_confidence``, with no check at all.

        Top level only. Returns ``name`` when it passed, else the descriptive
        fallback (a failure name), recording the outcome for the string returned:

        * a failure name, a disabled gate (instance flag or env, the documented
          raw-output configuration) and an exact-match list name
          (`_is_exact_match_list_name`) pass unchanged;
        * no jar (the opt-in reduced mode, ``ORTHONYM_ALLOW_REDUCED``): the
          default / valid / complete tiers keep their documented behaviour (no
          check possible, labelled unverified); best-effort fails closed, as the
          gate does there;
        * a general tier (``general_fallback``): the FULL-key round trip
          (`_shipped_name_round_trip`, memoized, so a string the gate or an offer
          already read costs no new parse) unless a full-key proof is recorded
          for this string; pass -> labelled verified, else withdrawn;
        * the default tier: a verified outcome recorded for this string (,
          the documented constitution-only stereo branch, the full-key ones) or
          one of the documented grammar carve-outs passes; a string the gate never
          judged (``bypassed`` / ``not_run``) goes through the default-tier gate
          now; anything else (an 'inconclusive' compare, 'unavailable' with the
          jar present,...) is withdrawn. The ``off`` / ``warn``
          configurations keep their documented meaning.
        """
        from .assembly.fragment_naming import is_top_level_naming
        if not name or is_failure_name(name) or not is_top_level_naming():
            return name
        if self._disable_opsin_validity_gate or _DISABLE_VALIDITY_GATE:
            return name
        try:
            if _is_exact_match_list_name(smiles, name):
                return name
            from .metrics import provenance as _pv
            prov = _pv.get_provenance()
            resolved = _pv.resolve_gate_outcome(
                prov["gate_outcome"], prov["gate_outcome_name"], name)
            if not _validity_gate_jar_present():
                if not self._general_fallback_unverified:
                    return name
                return self._withdraw_at_exit(
                    name, smiles, "jar_absent_besteffort_unverified")
            if self._general_fallback:
                if resolved in (_pv.GATE_OUTCOME_STEREO_RECOMPOSED,
                                _pv.GATE_OUTCOME_FULL_KEY_VERIFIED):
                    return name
                if _shipped_name_round_trip(name, smiles) == "verified":
                    if resolved != _pv.GATE_OUTCOME_SELF01:
                        _record_gate_outcome(_pv.GATE_OUTCOME_FULL_KEY_VERIFIED, name)
                    return name
                return self._withdraw_at_exit(name, smiles, "exit_full_key_round_trip")

            def _passes_default_tier(outcome: str) -> bool:
                return (outcome in (_pv.GATE_OUTCOME_SELF01,
                                    _pv.GATE_OUTCOME_SELF01_CONSTITUTION_ONLY,
                                    _pv.GATE_OUTCOME_STEREO_RECOMPOSED,
                                    _pv.GATE_OUTCOME_FULL_KEY_VERIFIED)
                        or outcome.startswith(_pv.GATE_OUTCOME_CARVEOUT_PREFIX)
                        or (outcome == _pv.GATE_OUTCOME_SELF01_SKIPPED
                            and _SC_MODE == "off")
                        or (outcome == _pv.GATE_OUTCOME_SELF01_WARN_MISMATCH
                            and _SC_MODE == "warn"))

            if _passes_default_tier(resolved):
                return name
            if resolved in (_pv.GATE_OUTCOME_BYPASSED, _pv.GATE_OUTCOME_NOT_RUN):
                gated = _final_opsin_validity_gate(
                    name, smiles, self._grammar_stats,
                    besteffort_unverified=self._general_fallback_unverified,
                    general_fallback_tier=False)
                if is_failure_name(gated):
                    return gated
                prov = _pv.get_provenance()
                if _passes_default_tier(_pv.resolve_gate_outcome(
                        prov["gate_outcome"], prov["gate_outcome_name"], gated)):
                    return gated
                return self._withdraw_at_exit(gated, smiles, "exit_gate_unverified")
            return self._withdraw_at_exit(name, smiles, "exit_" + resolved)
        except Exception as _xe:  # pragma: no cover - defensive, fails closed
            logger.info("exit round-trip check failed (withdrawn): %s", _xe)
            return _descriptive_fallback(smiles)

    @staticmethod
    def _withdraw_at_exit(name: str, smiles: str, detail: str) -> str:
        """Withdraw ``name`` at a public exit (see `_exit_round_trip_check`)."""
        from .metrics import provenance as _pv
        from .metrics.abstention import AbstentionCode, record_suppression
        logger.info("exit round-trip check withdrew %r (%s)", name[:80], detail)
        record_suppression(AbstentionCode.GATE_SUPPRESSED, detail=detail,
                           candidate=name)
        _suppressed_to = _descriptive_fallback(smiles)
        _record_gate_outcome(_pv.GATE_OUTCOME_SUPPRESSED, _suppressed_to)
        return _suppressed_to

    def _apply_trivial_fallback(self, result: str, smiles: str) -> str:
        """Task 1.9 (PIN-policy --trivial fallback).

        Returns ``result`` unchanged unless ALL of these hold:
          - naming produced only the failure signal (``is_failure_name``), AND
          - a general-only (PIN-denied) retained name exists for the molecule and
            ``self._trivial_fallback`` is set (opt-in), OR an OPSIN-import trivial
            name without PIN evidence exists (default, wp7).
        In that case the trivial name is returned. This is
        a fallback for an underivable PIN, never a downgrade of a derived PIN
        (a real name is never a failure name, so this can never fire on one).
        Applied only at the top level so recursive fragment calls are unaffected.
        """
        # wp7: a promoted OPSIN-import trivial name without Blue Book PIN
        # evidence (data.OPSIN_UNVERIFIED_RETAINED_NAMES: 'lepidine', 'nicotine',
        #...) is no longer a PIN-tier lookup; it is this DEFAULT last resort, so a
        # molecule the systematic pipeline cannot name keeps the name it had, now
        # labelled non-PIN (source 'trivial_retained'). The general-only
        # (PIN-denied) names stay behind the opt-in flag.
        from .assembly.fragment_naming import is_top_level_naming
        if not is_top_level_naming():
            return result
        if not is_failure_name(result):
            return result
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return result
        from .data import get_general_retained_name, get_unverified_retained_name
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        trivial = (get_general_retained_name(canonical_smiles)
                   if self._trivial_fallback else None)
        if not trivial:
            trivial = get_unverified_retained_name(canonical_smiles)
        if trivial:
            from .metrics.provenance import record_source
            record_source("trivial_retained")
            return trivial
        return result

    def name_with_confidence(self, smiles: str) -> dict:
        """Name one molecule and return a coverage score with it.

        Returns
        -------
        dict
            ``name`` (the name), ``confidence`` (a score from 0 to 1, or ``None``
            when no measurement was taken), ``verification``, ``factors`` (the parts
            of the score), ``handler`` (the part of the engine that built the name),
            and further keys for the atom-to-locant map and the reason for a decline
            (``limit``, ``abstention``).

        Raises
        ------
        ValueError
            If RDKit cannot read the SMILES.

        Examples
        --------
        >>> from orthonym import Orthonym
        >>> Orthonym.name_with_confidence("CCO")["name"]
        'ethanol'
        """
        # Lone-pair centre written first: the standard spelling of the input is named
        # when RDKit reads a lone-pair centre of it unlike the standard reading
        # (_lone_pair_standard_spelling); the atom-to-locant map is then re-keyed to
        # the atoms of the caller's string. Any other input is unchanged.
        spelled, _lp_opened = _lone_pair_input_enter(smiles)
        # The default tier's emission rule (see ``_default_tier_decision``) is
        # applied once, to the name this call returns: this path is no ``name``
        # scope, so the rule is switched off for the names it builds on the way.
        from .assembly.fragment_naming import name_scope_depth
        _policy = name_scope_depth() == 0 and self._default_tier_policy_applies()
        if _policy:
            from .metrics.provenance import clear_provenance
            clear_provenance()
        _policy_off = _DEFAULT_TIER_POLICY_OFF.set(True) if _policy else None
        try:
            try:
                metadata = self._name_with_confidence_impl(spelled)
            finally:
                if _policy_off is not None:
                    _DEFAULT_TIER_POLICY_OFF.reset(_policy_off)
            if _policy and isinstance(metadata, dict):
                metadata = self._default_tier_confidence(metadata, spelled)
            if spelled != smiles and isinstance(metadata, dict) and metadata.get('atom_to_locant'):
                metadata['atom_to_locant'] = _lone_pair_caller_atoms(metadata['atom_to_locant'])
            return metadata
        finally:
            _lone_pair_input_exit(_lp_opened)

    def _default_tier_confidence(self, metadata: dict, smiles: str) -> dict:
        """``name_with_confidence`` at the default tier: the emission rule of
        ``_default_tier_decision`` for the name it returns; a declined name gives the
        record of a decline with ``limit`` NO_VERIFIED_PIN."""
        name = metadata.get('name')
        if not name or is_failure_name(name):
            return metadata
        from .metrics.provenance import get_provenance
        row, pin_form = self._tier_row_and_pin_form(smiles, name, get_provenance())
        if _default_tier_emits(row, pin_form, smiles, self._trivial_fallback):
            return metadata
        from .assembly.coverage_scoring import unmeasured_confidence
        from .metrics.abstention import abstention_code_for
        label, _ = _default_tier_declined_row(smiles)
        md = unmeasured_confidence(name=label, handler='fallback')
        md['limit'] = no_verified_pin(label, smiles=smiles).as_dict()
        _abst = abstention_code_for(label)
        md['abstention'] = _abst.value if _abst else None
        return md

    def _name_with_confidence_impl(self, smiles: str) -> dict:
        """The body of:meth:`name_with_confidence`, for the string it names."""
        from .assembly.coverage_scoring import (
            clear_confidence,
            retrieve_confidence,
            unmeasured_confidence,
        )
        from .assembly.fragment_naming import (
            end_naming_session,
            is_top_level_naming,
            start_naming_session,
        )
        from .metrics.abstention import (
            AbstentionCode,
            abstention_code_for,
            clear_abstention,
            record_abstention,
        )
        from .metrics.provenance import best_effort_request_ctx
        start_naming_session()
        clear_confidence()
        # Task 0.1: reset the typed-abstention slot (twin of name).
        if is_top_level_naming():
            clear_abstention()
        # The tier of the request (``provenance.best_effort_request_ctx``), as
        # ``_budget_scope`` sets it for ``name``: this entry point calls
        # ``_name_impl`` without that scope. Set only outside a request, so a call
        # made inside a ``name`` keeps that request's tier; reset below.
        _request_token = None
        if best_effort_request_ctx.get() is None:
            _request_token = best_effort_request_ctx.set(
                bool(getattr(self, '_general_fallback_unverified', False)))
        try:
            # Wave-0 D1: same wildcard pre-check as name. name_with_confidence
            # returns a metadata dict, so mirror the limit handler's fallback dict
            # (lines ~4405–4419) rather than name's string return.
            # Perf: zero-false-negative pre-filter -- an RDKit dummy/wildcard
            # atom (atomic number 0) is spelled ONLY as `*`/`[*]` (bare `*`,
            # `[*]`, `[1*]`,...) or as `[#0]`/decorated (`[13#0]`, `[#0-]`,
            #...) -- both spellings are checked, so the filter has no false
            # negative, and skips the extra RDKit parse on the common
            # (non-wildcard) path.
            if '*' in smiles or '#0' in smiles:
                _probe = Chem.MolFromSmiles(smiles)
                if _probe is not None:
                    _scope = classify_scope_limit(_probe)
                    if _scope is not None:
                        _scope.smiles = smiles
                        record_abstention(AbstentionCode.OTHER, detail=_scope.code)
                        _fallback_name = self._exit_round_trip_check(
                            self._apply_trivial_fallback(_scope.message, smiles),
                            smiles)
                        _abst = abstention_code_for(_fallback_name)
                        _md = unmeasured_confidence(
                            name=_fallback_name, handler='fallback')
                        _md['limit'] = _scope.as_dict()
                        _md['abstention'] = _abst.value if _abst else None
                        return _md
            # --- Wave-2 P2: isotopic substitution decorator / ---
            # Twin of name's hook (:2938). name_with_confidence bypassed it and
            # called _name_impl directly, so an isotope-labeled mol named as the
            # UNLABELED skeleton (label silently dropped = wrong molecule; the OPSIN
            # validity gate cannot catch it — it parses to the unlabeled structure,
            # and 's skeleton compare ignores isotopes). Route to the
            # fail-closed decorator BEFORE _name_impl strips the label. has_isotopes
            # gate => zero cost + byte-identical on the entire unlabeled corpus.
            # Nothing scored coverage on this early-return path, so both exits use
            # the honest unmeasured record (C4). We are already INSIDE the try
            # whose finally ends the session, so just return — no second
            # end_naming_session (unlike name, whose hook sat outside its try).
            if is_top_level_naming():
                _iso_probe = Chem.MolFromSmiles(smiles)
                if _iso_probe is not None:
                    from .rules.isotopes import decorate_isotopic_name, has_isotopes
                    if has_isotopes(_iso_probe):
                        _iso_name = decorate_isotopic_name(smiles, self.style, self)
                        if _iso_name is not None:
                            # Claims conformance part 2: same exit check as name.
                            _iso_name = self._exit_round_trip_check(_iso_name, smiles)
                            _md = unmeasured_confidence(
                                name=_iso_name,
                                handler=('fallback' if is_failure_name(_iso_name)
                                         else 'isotope'))
                            _md['limit'] = None
                            _md['abstention'] = None
                            return _md
                        # Decorator failed closed on an isotope-labeled molecule:
                        # REFUSE. Falling through to _name_impl would emit the
                        # UNLABELED skeleton name (label silently dropped). Abstain
                        # to the descriptive fallback, exactly as name does (:2967).
                        record_abstention(AbstentionCode.OTHER,
                                          detail='isotope_decorator_failed')
                        _fallback_name = _descriptive_fallback(smiles)
                        _abst = abstention_code_for(_fallback_name)
                        _md = unmeasured_confidence(
                            name=_fallback_name, handler='fallback')
                        _md['limit'] = None
                        _md['abstention'] = _abst.value if _abst else None
                        return _md
            try:
                name = self._name_impl(smiles)
            except OrthonymLimitError as _limit:
                # G0 fail-closed (DD7 S1): a ring subsystem refused. Surface the
                # descriptive-fallback name + the named limit code (no crash).
                if _limit.smiles is None:
                    _limit.smiles = smiles
                # Task 0.1: mirror name's classification of the
                # limit signal (top-level ring refusal = NO_PARENT).
                if _limit.code == 'UNSUPPORTED_RING_SYSTEM':
                    record_abstention(
                        AbstentionCode.NO_PARENT if is_top_level_naming()
                        else AbstentionCode.BRANCH_UNNAMEABLE,
                        detail=_limit.code,
                    )
                else:
                    record_abstention(AbstentionCode.OTHER, detail=_limit.code)
                # Task 1.9 (C8): apply trivial fallback here, mirroring name.
                # Claims conformance part 2: then the same exit check as name.
                _fallback_name = self._exit_round_trip_check(
                    self._apply_trivial_fallback(_limit.message, smiles), smiles)
                _abst = abstention_code_for(_fallback_name)
                # C4: nothing scored this either, so report the shared
                # unmeasured record rather than a fabricated 0.0 (a verdict of
                # FAIL is as unfounded as a verdict of PASS when no measurement
                # happened). The 'limit'/'abstention' keys already carry the
                # out-of-scope signal a consumer needs.
                _md = unmeasured_confidence(name=_fallback_name,
                                            handler='fallback')
                _md['limit'] = _limit.as_dict()
                _md['abstention'] = _abst.value if _abst else None
                return _md
            # Universal stereo backstop (a phase,)
            if is_top_level_naming():
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    _conf = retrieve_confidence()
                    handler = _conf.get('handler', 'unknown')
                    # a phase -01 : thread the authoritative parent
                    # map + phenol flag to the backstop.
                    name = _final_stereo_check(
                        mol, name, handler=handler,
                        atom_to_locant=_conf.get('atom_to_locant'),
                        is_phenol_benzene=_conf.get('is_phenol_benzene'),
                    )
                    # Universal OPSIN-grammar backstop (a phase,).
                    name = _final_grammar_check(
                        name, smiles, handler,
                        self._grammar, self._grammar_stats,
                    )
                    # (169.5): real-OPSIN validity gate (twin of name).
                    if not self._disable_opsin_validity_gate:
                        name = _final_opsin_validity_gate(
                            name, smiles, self._grammar_stats,
                            besteffort_unverified=self._general_fallback_unverified, general_fallback_tier=self._general_fallback,
                        )
            metadata = retrieve_confidence()
            # C4: no candidate was scored (early-return path). This used to
            # fabricate confidence=1.0 with all four factors at 1.0 and
            # handler='direct' -- "Early return paths are high confidence" --
            # so the PUBLIC API reported PERFECT confidence and PERFECT atom
            # coverage for a name that nothing had scored. Demonstrated defect:
            # CC(C)(C)OOCCO (9 heavy atoms) emits 'ethan-1-ol' (3 heavy atoms)
            # and this path certified it at atom_coverage=1.0.
            #
            # "Early return" does not mean "high confidence"; it means NOTHING
            # MEASURED. The honest record says exactly that: confidence=None,
            # verification='unverified', factors={} (so no consumer can read an
            # atom_coverage that was never computed), handler='unmeasured'.
            # Shared with the name path via one constructor so the two can no
            # longer disagree.
            if not metadata['name']:
                metadata = unmeasured_confidence(name=name)
            else:
                # Ensure name matches (the stored candidate should match
                # what was returned)
                metadata['name'] = name

            # (a phase): informational limit annotation (additive — does
            # not change 'name'). None for in-scope inputs; otherwise the named
            # out-of-scope code (scope pre-check, else failure mapping).
            _limit = None
            _probe = Chem.MolFromSmiles(smiles)
            if _probe is not None:
                _lim = classify_scope_limit(_probe)
                if _lim is None and is_failure_name(metadata.get('name')):
                    _lim = classify_failure_limit(_probe, smiles=smiles)
                if _lim is not None:
                    _limit = _lim.as_dict()
            metadata['limit'] = _limit

            # Task 1.9 (C8): apply trivial fallback, mirroring name line 1483.
            # _apply_trivial_fallback is a no-op when trivial_fallback=False,
            # when called recursively, or when a real PIN was derived — safe.
            metadata['name'] = self._apply_trivial_fallback(metadata['name'], smiles)
            # Claims conformance part 2 (2026-09-27): the same exit check as name
            # (_exit_round_trip_check): the trivial last resort above is never
            # gated, and no name leaves without a round trip that passed.
            _checked = self._exit_round_trip_check(metadata['name'], smiles)
            if _checked != metadata['name']:
                metadata = unmeasured_confidence(name=_checked, handler='fallback')
                _lim2 = None
                if _probe is not None:
                    _lim2 = classify_scope_limit(_probe)
                    if _lim2 is None and is_failure_name(_checked):
                        _lim2 = classify_failure_limit(_probe, smiles=smiles)
                metadata['limit'] = _lim2.as_dict() if _lim2 is not None else None

            # Task 0.1: typed abstention code (additive key; None for
            # a successfully named molecule).
            _abst = abstention_code_for(metadata['name'])
            metadata['abstention'] = _abst.value if _abst else None

            return metadata
        finally:
            if _request_token is not None:
                best_effort_request_ctx.reset(_request_token)
            end_naming_session()
            clear_confidence()

    def _try_retained_fused_upgrade(self, mol, smiles: str) -> Optional[str]:
        """: a retained fused-heterocycle whole-molecule name (currently the
        purine ring system) when it round-trips, else None.

        A giant purine-containing molecule (an acyl-CoA) is not itself a bare
        fused ring, so the normal dispatch never routes it to
        ``name_fused_heterocycle``; it falls to the von-Baeyer general-engine
        fallback and is named on a ``...tetraazabicyclo[4.3.0]...`` polyene parent
        instead of its PIN-quality ``...9H-purin-6-amine``. This offers the fused
        name as a preferred candidate, adopted by the caller ONLY when its OPSIN
        round-trip verifies -- so a molecule whose fused form is not yet RT-exact
        still falls through to the (valid) von-Baeyer name (no regression; 0-wrong
        via the RT probe + the downstream gate). ``best_effort_ctx`` is
        already set for this session (the caller gates on
        ``_general_fallback_unverified``), so ``name_substituted_purine`` recurses
        the giant arm under best-effort."""
        try:
            from .rules.purine import _PURINE_CORE
        except Exception:
            return None
        if _PURINE_CORE is None or not mol.HasSubstructMatch(_PURINE_CORE):
            return None
        try:
            from .rules.fused_rings import name_fused_heterocycle
            res = name_fused_heterocycle(mol)
        except Exception:
            return None
        if not res or not res[0]:
            return None
        cand = res[0]
        try:
            from .validation import opsin_roundtrip_check
            rt = opsin_roundtrip_check(smiles, cand)
        except Exception:
            return None
        return cand if rt and rt.get("passed") else None

    def _try_np_systematic_downgrade(
            self, smiles: str, retained_name: str) -> Optional[str]:
        """ giants Engine 3: an OPSIN-round-trippable name for a retained
        natural-product parent hydride, when the general-engine recovery can
        produce one -- else None (keep the retained name).

        The mirror of:meth:`_try_retained_fused_upgrade`: that one trades a
        von-Baeyer name for a retained one, this one trades a retained name for
        a recovered one. Both are best-effort-only, RT-gated, whole-molecule
        swaps that leave the PIN default byte-identical.

        `ursane`, `hopane`, `cevane`,... are valid PINs
        that OPSIN 2.9.0 cannot parse, so they are whitelisted past the
        gate and ship unverified. On the best-effort path a round-trippable name
        is worth more than the retained spelling, so offer it here -- but adopt
        it ONLY on a verified full-InChI round-trip. Most rows recover a
        von-Baeyer systematic parent; the recovery's own retained-name
        preference can also return a trivial name (e.g. `(4E)-sphing-4-enine` ->
        `sphingosine`, RT-verified), which is acceptable on the breadth tier
        (the PIN default keeps the sphingoid parent). The gate is load-bearing,
        not defensive: `germacrane` recovers NO round-tripping name (its
        systematic cyclodecane-parent form does not round-trip), so an
        unconditional downgrade regresses it — the RT gate keeps the retained
        PIN. (`corynoxan` DID join this class historically, but its stereo-composed
        systematic spiro name now round-trips at 0-wrong, so it converts.)

        Never raises: any failure keeps the retained name.
        """
        if not self._general_fallback_unverified:
            return None
        try:
            from .data.natural_products import NAME_EXACT_NP_PARENTS
        except Exception:
            return None
        if retained_name not in NAME_EXACT_NP_PARENTS:
            return None
        # `_try_general_engine_recovery` stamps `source="general_engine"` (and an
        # `opsin` status) via `record_source` BEFORE we can decline its
        # candidate. Snapshot the pre-probe provenance so a KEPT retained name
        # (RT-reject) is labelled by its own producer rather than the discarded
        # general-engine attempt -- otherwise the cohort/census attribution the
        # project ranks levers by is skewed on every kept row.
        from .metrics.provenance import get_provenance, restore_provenance
        _snap = get_provenance()
        try:
            cand = self._try_general_engine_recovery(smiles)
            if cand and not is_failure_name(cand) and cand != retained_name:
                from .validation import opsin_roundtrip_check
                rt = opsin_roundtrip_check(smiles, cand)
                if rt and rt.get("passed"):
                    return cand  # adopt: general_engine provenance is correct
        except Exception:  # noqa: BLE001 - any failure keeps the retained PIN
            pass
        # RT-reject / no candidate / crash: restore provenance, keep retained.
        restore_provenance(_snap)
        return None

    def _name_impl(self, smiles: str, _skip_decomposition: bool = False) -> str:
        """Internal naming implementation (wrapped by session management).

        a phase substrate: the implicit cascade at lines 853-1115 is
        REPLACED by a single ``self._cfr_router.dispatch(...)`` call. Per
        the audit option (c) the zwitterion-character in-place
        mutation (lines 1000-1034) lives INLINE here BEFORE the dispatch
        call — preserves predicate-purity invariant cleanly. Per
        Task 158-02-01 design choice (a) the GENERAL handler shim returns
        None to signal that ``_name_impl`` runs the legacy ``_perceive ->
        _classify -> assemble_name`` pipeline INLINE — keeps ``routing/``
        decoupled from ``composer.py`` (boundary). Quality-gate
        post-checks at lines 1129-1207 are PRESERVED VERBATIM after
        the dispatch call.
        """
        # a phase option (b): thread `_skip_decomposition` through
        # the instance attribute so the DECOMPOSITION_PRE_GENERAL predicate
        # factory + handler shim can read it from kwargs forwarded by
        # `dispatch`.
        self._skip_decomposition = _skip_decomposition
        # One naming pass of the outermost name (the unit that bounds the
        # clean-first offer, ``_prefer_clean_over_floor_substitute``).
        from .assembly.fragment_naming import count_naming_pass
        count_naming_pass()

        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Get canonical SMILES for consistent processing
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

        # ============================================================
        # Pre-dispatch: zwitterion-character in-place mutation ((c))
        # ============================================================
        # a phase the audit option (c) lock: the cascade at
        # namer.py:1000-1034 MUTATED `mol` and `canonical_smiles` in-place
        # for downstream cascade consumption. CFR's predicate-handler model
        # does NOT support "mutate state, continue cascade" patterns
        # natively. Per the audit's UNANIMOUS decision, the mutation lives
        # INLINE here BEFORE CFR.dispatch — preserves hard invariant
        # (every entry's `side_effect_inventory == `) cleanly.
        #
        # CRITICAL byte-identical gate: in the cascade the mutation at
        # line 1000-1034 was only REACHED when ALL of the prior branches
        # (salt/radical/zwitterion/ion at 852-956 + multi-component-neutral
        # at 967-998) had already declined to handle the molecule. Those
        # branches return early on success, so the mutation effectively
        # only ran on `species_type == 'neutral'` molecules that were not
        # multi-component cocrystals. To preserve byte-identical behavior
        # we MUST gate the inline pre-dispatch mutation on
        # `species_type == 'neutral'`; otherwise salts like `[Ag+].[Cl-]`
        # (which DO satisfy `_has_true_zwitterion_character` because they
        # have both + and - atoms) get mutated before SALT routing fires
        # and salt detection fails downstream.
        # [Rule 1 - Bug] Found during Task 158-02-07 canary investigation.
        #
        # CHARGE NEUTRALIZATION for large zwitterions reclassified as
        # 'neutral' by the HA>20 + quaternary-N guard in
        # `detect_species_type`. These molecules still carry formal
        # charges (P-O-, N+) that prevent FG detection SMARTS from matching
        # (e.g., [OX2H] for COOH). Neutralize O- to OH; leave quaternary
        # N+ (no H) charged to preserve valid valence. ONLY applies to
        # molecules that have true zwitterion character -- NOT to molecules
        # with internal charges (nitro [N+](=O)[O-], azide, N-oxide) which
        # are normal functional groups.
        _zwitter_species_type = detect_species_type(mol)
        if _zwitter_species_type == 'neutral' and Chem.GetFormalCharge(mol) == 0:
            from .perception.ions import _has_true_zwitterion_character
            if _has_true_zwitterion_character(mol):
                try:
                    from rdkit.Chem import RWMol
                    rwmol = RWMol(mol)
                    for atom in rwmol.GetAtoms():
                        charge = atom.GetFormalCharge()
                        if charge < 0:
                            # O- -> OH (add H for each negative charge)
                            atom.SetFormalCharge(0)
                            cur_h = atom.GetNumExplicitHs()
                            atom.SetNumExplicitHs(cur_h + abs(charge))
                        elif charge > 0:
                            total_h = atom.GetTotalNumHs()
                            if total_h >= charge:
                                # Protonated amine: remove H to neutralize
                                cur_h = atom.GetNumExplicitHs()
                                atom.SetFormalCharge(0)
                                atom.SetNumExplicitHs(max(0, cur_h - charge))
                            # else: quaternary N+ (no H) -- leave charged
                            # to preserve valid valence
                    Chem.SanitizeMol(rwmol)
                    mol = rwmol.GetMol()
                    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                except Exception:
                    pass  # If neutralization fails, continue with original mol

        # ============================================================
        # CFR dispatch — replaces cascade lines 853-1115
        # ============================================================
        # a phase single chokepoint integration: CFR routes input
        # mol; backstops (a phase + 156) wrap output name at
        # `Orthonym.name` line 654. The CFR dispatcher walks
        # DISPATCH_TABLE in priority order; the first predicate that
        # matches wins; the result's handler is invoked here.
        #
        # The the audit DISPATCH_TABLE has 17 outer-cascade rows + GENERAL
        # catch-all = 18 entries. Several handler shims (ANION_SMALL,
        # POLY_ANION, MULTI_COMPONENT_NEUTRAL, DECOMPOSITION_PRE_GENERAL)
        # may return None to signal "no match — continue cascade", in
        # which case we re-dispatch with the matched class excluded so the
        # next priority entry runs. This mirrors the cascade's
        # fall-through behavior (e.g., namer.py:913-914 + 955-956).
        #
        # `_skip_decomposition` is threaded via kwargs per option (b);
        # `style` is threaded via kwargs so the RETAINED_NAME and
        # AMINO_ACID predicates can read it without binding `self`.
        result = self._cfr_router.dispatch(
            mol, smiles, canonical_smiles,
            _skip_decomposition=self._skip_decomposition,
            _style=self.style,
        )
        # a phase: a class already tried and REJECTED BY A GATE on this
        # molecule is skipped, so the cascade below moves to the next entry.
        # Treated exactly like a handler that returned None, which is the
        # fall-through the cascade already implements.
        if result.class_id in self._excluded_dispatch_classes:
            self._last_dispatch_class = result.class_id
            name = None
        else:
            name = result.handler(
                mol, smiles, canonical_smiles,
                features=None,
                style=self.style,
                _skip_decomposition=self._skip_decomposition,
                general_fallback=self._general_fallback,
                allow_aromatic_general=self._allow_aromatic_general,
                general_fallback_unverified=self._general_fallback_unverified,
            )

        # Cascade-continuation when a handler returns None (the audit row
        # notes for ANION_SMALL / POLY_ANION / MULTI_COMPONENT_NEUTRAL /
        # DECOMPOSITION_PRE_GENERAL). The cascade falls through to the
        # next sibling; mirror that here by re-running dispatch with
        # progressively higher priority floors.
        if name is None and result.class_id != StoutClass.GENERAL:
            from .routing.dispatch_table import DISPATCH_TABLE
            current_priority = DISPATCH_TABLE[result.class_id].priority
            for entry in sorted(DISPATCH_TABLE.values(), key=lambda e: e.priority):
                if entry.priority <= current_priority:
                    continue
                if entry.class_id in self._excluded_dispatch_classes:
                    continue  # a phase: already gate-rejected on this molecule
                # Predicate signature: (mol, smiles, canonical_smiles, features, **kwargs)
                try:
                    matched = entry.predicate(
                        mol, smiles, canonical_smiles, None,
                        _skip_decomposition=self._skip_decomposition,
                        _style=self.style,
                    )
                except TypeError:
                    matched = entry.predicate(mol, smiles, canonical_smiles, None)
                if not matched:
                    continue
                self._cfr_router._dispatch_stats[entry.class_id] += 1
                name = entry.handler(
                    mol, smiles, canonical_smiles,
                    features=None,
                    style=self.style,
                    _skip_decomposition=self._skip_decomposition,
                    general_fallback=self._general_fallback,
                    allow_aromatic_general=self._allow_aromatic_general,
                    general_fallback_unverified=self._general_fallback_unverified,
                )
                result = ClassDispatchResult(
                    class_id=entry.class_id,
                    handler=entry.handler,
                    audit_record={},
                    tier=entry.tier,
                )
                if name is not None or entry.class_id == StoutClass.GENERAL:
                    break
        # a phase: whichever class actually produced this name, so a gate
        # rejection in name can exclude it and re-enter the cascade.
        self._last_dispatch_class = result.class_id

        # ============================================================
        # GENERAL pipeline fallback — Task 158-02-01 design choice (a)
        # ============================================================
        # The GENERAL handler shim returns None to signal that
        # `_name_impl` runs the legacy `_perceive -> _classify ->
        # assemble_name` pipeline inline. This avoids coupling
        # `routing/` to `composer.py` (boundary). The check below
        # mirrors the cascade's GENERAL pipeline at namer.py:1124-1131.
        if name is None and result.class_id == StoutClass.GENERAL:
            features = self._perceive(mol, smiles, canonical_smiles)
            self._classify(features)
            assembled = assemble_name(features, style=self.style)
            name = assembled

            # a phase -01 (/): thread the authoritative chain
            # atom_to_locant to the backstop. The general/chain pipeline does
            # NOT call store_confidence (only the ring path does), so the
            # backstop would otherwise see handler='unknown' and stay log-only.
            # When the parent is a CHAIN we publish a minimal POST-HOC candidate
            # carrying handler='chain' + the authoritative map so the backstop
            # can inject a missed stereodescriptor. The map is built in
            # _perceive at the chain-orientation site (features.atom_to_locant);
            # it is never recomputed and never passed into compute_confidence
            # (byte-identity Risk 1).
            if (getattr(features, 'chain_is_parent', False)
                    and getattr(features, 'atom_to_locant', None)):
                from .assembly.coverage_scoring import (
                    CandidateName,
                    retrieve_confidence,
                    store_confidence,
                )
                _existing = retrieve_confidence()
                # Only publish when no richer candidate was already stored for
                # this call (e.g. a ring/Tier-A candidate) — never clobber it.
                if not _existing.get('name'):
                    # C4: this candidate exists ONLY to publish
                    # atom_to_locant to the stereo backstop -- nothing scored
                    # it, and it carries no factors. It previously claimed
                    # confidence=1.0, which (a) is a fabricated pass and
                    # (b) fed the coverage gate below a handler that is
                    # neither 'unknown' nor 'retained_name' together with an
                    # EMPTY factors dict, so the gate's missing-measurement
                    # default silently read as perfect coverage. confidence
                    # is None: the gate's `is not None` guard then skips it
                    # for the same reason it skipped 1.0, but honestly.
                    store_confidence(CandidateName(
                        name=name,
                        handler='chain',
                        confidence=None,
                        atom_to_locant=dict(features.atom_to_locant),
                    ))

            # ============================================================
            #: general-engine inline fallback (OPT-IN, default OFF).
            # Fires ONLY when the legacy GENERAL pipeline abstained, so the
            # default PIN path stays byte-identical (decision 3). An
            # engine emission still re-enters the SAME downstream moat:
            # the >15-HA coverage gate, the P10 source-vetoes, and the
            # OPSIN-RT gate.
            # ============================================================
            #: prefer a retained fused-heterocycle whole-molecule name (e.g.
            # the `...9H-purin-6-amine` parent of an acyl-CoA) over the general
            # engine's von-Baeyer polyene, WHEN it round-trips. A giant
            # purine-containing molecule is not itself a bare fused ring, so the
            # normal dispatch never routes it to `name_fused_heterocycle` -- it
            # falls straight to the von-Baeyer fallback below and loses the
            # PIN-quality retained parent (`...tetraazabicyclo[4.3.0]...`). Try the
            # fused name first; adopt it ONLY when its OPSIN round-trip verifies,
            # so a molecule whose fused form is not yet RT-exact still falls
            # through to the (valid) von-Baeyer name -- never a regression, 0-wrong
            # via the RT probe + the downstream gate. Best-effort only
            # (`_general_fallback_unverified`), so PIN/default output is unchanged.
            if (self._general_fallback_unverified
                    and (not name or is_failure_name(name))):
                _fused_up = self._try_retained_fused_upgrade(mol, smiles)
                if _fused_up is not None:
                    name = _fused_up

            if self._general_fallback and (not name or is_failure_name(name)):
                from .assembly.general_engine import name_general
                from .validation.coverage_gate import certify_general_result
                try:
                    _eng = name_general(
                        mol, features,
                        allow_aromatic_general=self._allow_aromatic_general,
                        allow_suffix_free=self._general_fallback_unverified)
                    # a phase B4: shared best-effort certification gate (E1 +
                    # structural binding-spine axes), replacing the E1-only
                    # check so the inline G1 lane cannot ship a name that
                    # re-fragments a ring or swaps a stereo feature.
                    if _eng is not None and certify_general_result(
                            mol, _eng,
                            allow_charged=self._allow_aromatic_general,
                            structural_only=True):
                        # (audit-only): record the certified spine here,
                        # at the inline site, for re-assertion at the exit of
                        # name. Everything between this point and the return
                        # -- the stereo backstop, the grammar repair backstop
                        # and the OPSIN validity gate -- may rewrite the string
                        # the certificate was computed on.
                        self._record_binding_proof(
                            mol, _eng, stage="general_engine")
                        # FIX 1: complete-tier abstain-without-Java
                        # contract. E1 does NOT verify ring numbering/locants,
                        # and the downstream _final_opsin_validity_gate FAILS
                        # OPEN with no OPSIN jar -- so under `complete`
                        # (allow_aromatic_general) accepting this E1-only name
                        # with no jar would ship an UNVERIFIED name from the NEW
                        # aggressive producers (P1/P2/P5). Mirror the
                        # late-recovery jar guard: keep the abstention. Scoped to
                        # allow_aromatic_general so valid/best-effort/pre-
                        # behavior is unchanged. (best-effort opts in via
                        # _general_fallback_unverified; test/internal mode via
                        # _disable_opsin_validity_gate.)
                        _no_jar_abstain = (
                            self._allow_aromatic_general
                            and not self._general_fallback_unverified
                            and not self._disable_opsin_validity_gate
                            and not _validity_gate_jar_present()
                        )
                        # FIX 2: fail closed on DROPPED stereo. is
                        # constitutional (stereo-blind) and the universal stereo
                        # backstop is LOG-ONLY for the handler='unknown' cohort
                        # general emissions arrive as -- so a general name that
                        # omits E/Z or R/S the input carries would otherwise
                        # ship a WRONG stereoisomer. Expressed stereo (parent
                        # _stereo_prefix / substituent-internal name_substituent)
                        # makes the predicate False -> no over-abstention.
                        # T6.2: shared stereo-emit policy (same helper as
                        # the late-recovery site). complete/valid abstain on
                        # dropped stereo; best-effort ships a constitution-only
                        # name flagged stereo_unexpressed.
                        #
                        # no-abstain Phase A fix-a performance pass, Finding 4 (HIGH,
                        # a review adversarial review): the OLD comment here claimed
                        # the downstream `_final_opsin_validity_gate` "verifies at
                        # the granularity [a flagged emission] asserts" -- WRONG.
                        # That gate is CONSTITUTIONAL ONLY (InChIKey skeleton +
                        # charge; it never inspects the stereo layer at all), so
                        # a flagged emission that OMITS one stereo descriptor
                        # while asserting a WRONG value for another (partial
                        # omission + partial CONFLICT -- e.g. a 2-centre input
                        # where the name cites only 1 centre and gets it
                        # negated) has the SAME skeleton and would ship as a
                        # WRONG STEREOISOMER unverified. Reproduced directly
                        # (`a temp dir/probe_finding4_g1_stereo.py`):
                        # `C[C@@H](O)[C@@H](N)C` + candidate
                        # `(3R)-3-aminobutan-2-ol` -- skeleton matches, full
                        # InChIKey differs, and this lane shipped it with no
                        # check at all. Route a FLAGGED emission through the
                        # SAME `_rt_match` superset gate the late-recovery site
                        # (`_try_general_engine_recovery`) already uses for this
                        # exact purpose -- proven (a review's Q1,
                        # `a temp dir/probe_rtmatch_semantics.py`, 13/14 probed
                        # cases) to ACCEPT a genuine omission while REJECTING a
                        # conflict/fabrication/wrong-enantiomer. An unflagged
                        # emission (`_stereo_flagged=False`, already fully
                        # stereo-complete) is UNCHANGED -- this only tightens the
                        # flagged (best-effort-only) case. (This gate is
                        # jar-PRESENT-only; the jar-ABSENT best-effort case no
                        # longer fail-opens -- see the task-JAR-ABSENT
                        # verify_or_none gate just before the emit below, which
                        # closed the a review 0-wrong hole here.)
                        _permitted, _stereo_flagged, _g1_name = self._stereo_emit_decision(
                            mol, _eng.name, smiles)
                        if (_permitted and _stereo_flagged
                                and not self._disable_opsin_validity_gate
                                and _validity_gate_jar_present()):
                            _g1_opsin_smi = _validity_gate_name_to_smiles(
                                _eng.name)
                            if (_g1_opsin_smi is None
                                    or not self._rt_match(
                                        smiles, _g1_opsin_smi, True)):
                                _permitted = False
                        # Phase-1 reclaim: a COMPOSED name (_g1_name != _eng.name) is
                        # UNFLAGGED and so skips the flagged-True block above. Unlike
                        # the late-recovery ladder sites, this inline G1 lane has no
                        # exact _rt_match(flagged=False) before the emit -- the
                        # downstream _final_opsin_validity_gate is stereo-TOLERANT at
                        # some tiers (the contributor guide live defect: BBR-GATE ships a name
                        # whose stereo-stripped form parses). So verify a composed
                        # reclaim EXACT-ISOMERIC inline before adopting. Scoped to
                        # composed names only => every currently-passing emission
                        # (fast-accept: _g1_name == _eng.name) stays byte-identical.
                        if (_permitted and _g1_name != _eng.name
                                and not self._disable_opsin_validity_gate
                                and _validity_gate_jar_present()):
                            _g1_c_smi = _validity_gate_name_to_smiles(_g1_name)
                            if (_g1_c_smi is None
                                    or not self._rt_match(smiles, _g1_c_smi, False)):
                                _permitted = False
                        # task-JAR-ABSENT (a review 0-wrong hole, inline-G1
                        # sibling of the late-recovery ladder site above):
                        # `_no_jar_abstain` only fires for the COMPLETE tier
                        # (`allow_aromatic_general and not
                        # general_fallback_unverified`), so a jar-absent
                        # BEST-EFFORT emission fell through and shipped
                        # `_eng.name` with NO verification of any kind -- the
                        # downstream `_final_opsin_validity_gate` fails OPEN
                        # with no jar, so nothing caught a wrong molecule. Same
                        # defect class as the primary hole. Mirror the primary
                        # fix: route a jar-absent best-effort emission through
                        # the OPSIN-free reconstructor oracle before shipping.
                        # `name_facts=None` => provably None with no jar today
                        # (a clean fail-closed ABSTAIN; future-proof for a
                        # later NameFacts extractor). Untouched when the jar is
                        # PRESENT (the stereo gate above already covers that)
                        # and in disable-gate test mode (G2 ship-unverified
                        # contract, matching the primary site's `if
                        # self._disable_opsin_validity_gate: pass`).
                        if (_permitted
                                and self._general_fallback_unverified
                                and not self._disable_opsin_validity_gate
                                and not _validity_gate_jar_present()):
                            from .validation.reconstruct import verify_or_none
                            if verify_or_none(
                                    _eng.name, smiles,
                                    name_facts=None) is None:
                                _permitted = False
                        if not _no_jar_abstain and _permitted:
                            name = _g1_name          # adopt composed/as-is reclaim
                            #: observation-only provenance (the emission
                            # still flows through the normal downstream gates).
                            from .metrics.provenance import record_source, record_stereo_unexpressed
                            record_source("general_engine")
                            record_stereo_unexpressed(_stereo_flagged)
                except Exception as _e:  # fail-closed: an engine bug must
                    # never turn an abstention into a crash or a wrong name.
                    logger.info("general engine error (kept abstention): %s",
                                _e)

        # ============================================================
        # Post-dispatch quality gates — preserved VERBATIM from
        # lines 1129-1207. CFR is the dispatch substrate; these gates
        # run on the GENERAL pipeline's assembled name AFTER dispatch,
        # NOT on outputs from class-specific handlers.
        #
        # [Rule 1 - Bug] Found during Task 158-02 verification: gating
        # on ANY non-None name (the original implementation) caused
        # PEPTIDE / SALT / etc. handler outputs to be re-checked
        # against the GENERAL pipeline's confidence store, which is
        # populated by `assemble_name` and may still hold stale data
        # from the PREVIOUS call (the `_confidence_store` is thread-
        # local and persists across `Orthonym.name` invocations; see
        # `assembly/coverage_scoring.py:_confidence_store`). Pre-CFR
        # the gate only ran on `assembled = assemble_name(...)` output
        # at the very end of the cascade — handler early-returns
        # (e.g. peptide / salt / amino_acid) bypassed it entirely.
        # Restrict the gate to GENERAL-pipeline outputs to preserve
        # byte-identical canary. Honest-fail-on-data per
        # internal notes: the fix is upstream (gate scope), NOT a
        # threshold relaxation or postprocessor band-aid.
        # ============================================================
        # C4 -- the `> 15` size term was REMOVED from this entry test, so
        # small molecules are no longer excluded from the quality gates
        # wholesale. Previously a 9-heavy-atom molecule could not reach any
        # gate at all: the reference defect CC(C)(C)OOCCO -> 'ethan-1-ol'
        # (six atoms silently dropped) was excluded here, several lines
        # BEFORE the `conf_handler != 'unknown'` check people assumed was
        # responsible.
        #
        # MEASURED before changing (scripts/../a temp dir widen probe, the
        # COMPLETE affected population -- all 714 rows of
        # benchmarks/pubchem_2000.csv with heavy_atoms <= 15, 0 timeouts):
        # 593 molecules are newly admitted (class GENERAL, HA <= 15)
        # garbled token: 0 fire
        # len(name) < 6: 0 fire
        # confidence < 0.30: 0 fire
        # atom_coverage < 0.55: 0 fire
        # => 0 emitted names change. Confirmed by a full A/B naming run
        # over the same 714 rows (0 diffs).
        #
        # So this widening is behaviour-neutral TODAY and is a reachability
        # fix only. It does NOT catch the reference defect, and no threshold
        # tweak here could: nothing stores a candidate for it
        # (handler='unknown' fails both gates) and the atom_coverage number
        # is a name-length proxy scoring 'ethan-1-ol' at 0.74, above the 0.55
        # cut-off, where its REAL coverage is 3/9 = 0.33. The gates become
        # useful for small molecules only once producer-side atom-coverage
        # records exist; this removes the size exclusion that would otherwise
        # still hide them at that point.
        if (not _skip_decomposition
                and name
                and result.class_id == StoutClass.GENERAL):
            assembled = name  # local alias for -byte-identical body
            _heavy = mol.GetNumHeavyAtoms()
            _GARBLED_TOKENS = ('cycloane', 'anedicarboxamide', 'aneyl')
            assembled_lower = assembled.lower()
            is_garbled = any(tok in assembled_lower for tok in _GARBLED_TOKENS)

            # Also detect stub-only names: when the parent text is empty,
            # the assembly may produce just a bare suffix like "ane" or "ol".
            # A name shorter than 6 chars for a 15+ atom molecule is garbled.
            #
            # C4: the `_heavy > 15` term is retained HERE deliberately. The
            # justification for this heuristic is explicitly size-based (a
            # 5-char name is damning for a 15+ atom molecule, unremarkable for
            # a small one), so it is kept inside its stated domain rather than
            # extended past it. Legitimate short PINs do exist -- 'furan' (5),
            # 'urea' (4), 'oxane' (5) -- and although those happen to route as
            # StoutClass.RETAINED_NAME today and so never reach this block,
            # relying on that would make a size-justified heuristic depend on
            # an unrelated routing accident. Measured: 0 of the 593 newly
            # admitted molecules have len(name) < 6 either way.
            if not is_garbled and _heavy > 15 and len(assembled) < 6:
                is_garbled = True

            if is_garbled:
                # Task 0.1: the downgrade machinery engaged — the
                # assembled GENERAL name is being rejected (P2.1 bucket).
                from .metrics.abstention import AbstentionCode, record_suppression
                record_suppression(AbstentionCode.COVERAGE_DOWNGRADE,
                                   detail='garbled', candidate=assembled)
                from .decomposition import try_decompose
                decomp_name = try_decompose(mol, style=self.style)
                if decomp_name and decomp_name != assembled:
                    return decomp_name
                # Decomposition also failed: return the SMILES-based
                # canonical SMILES as an honest fallback rather than a
                # garbled pseudo-IUPAC name that could mislead.
                return canonical_smiles

            # Confidence-based rejection: if the coverage scoring system
            # indicates the assembled name is catastrophically incomplete
            # (confidence < threshold), attempt decomposition fallback.
            # This is NOT a postprocessor -- the confidence score reflects
            # genuine structural coverage analysis computed during naming.
            from .assembly.coverage_scoring import retrieve_confidence
            conf_data = retrieve_confidence()
            conf_score = conf_data.get('confidence', None)
            conf_handler = conf_data.get('handler', 'unknown')
            # Only gate on confidence when it was actually computed during
            # assemble_name -- handler='unknown' means no scoring happened
            # (e.g., decomposition path, retained names, etc.).
            if (conf_score is not None
                    and conf_handler != 'unknown'
                    and conf_score < _TRUNCATION_CONFIDENCE_THRESHOLD):
                # Task 0.1: downgrade machinery engaged (P2.1 bucket).
                from .metrics.abstention import AbstentionCode, record_suppression
                record_suppression(AbstentionCode.COVERAGE_DOWNGRADE,
                                   detail='low_confidence', candidate=assembled)
                from .decomposition import try_decompose
                decomp_name = try_decompose(mol, style=self.style)
                if decomp_name and decomp_name != assembled:
                    logger.info(
                        "Quality gate: rejecting low-confidence name "
                        "(%.4f < %.2f), using decomposition fallback",
                        conf_score, _TRUNCATION_CONFIDENCE_THRESHOLD,
                    )
                    return decomp_name

            #: atom_coverage secondary gate -- catches quality-gate false
            # positives where retained ring names inflate character-based
            # confidence but atom coverage reveals only partial molecule
            # description.
            #
            # C4 -- READ BEFORE TRUSTING THIS GATE. Two facts:
            #
            # (1) `atom_cov` here is NOT a coverage measurement. No production
            # caller passes parent_atom_indices into compute_confidence
            # (candidate_pool.add's "Risk 1" byte-identical mitigation),
            # so this value is the name-LENGTH proxy -- numerically equal
            # to factors['ratio'] -- or a retained-name 1.0. Its
            # provenance is published as conf_data['coverage_provenance'].
            # The gate is retained AS IS (removing it would un-suppress
            # names it currently rejects, which is the unsafe direction),
            # but it must not be described as verified coverage, and it
            # cannot be relied on to catch atom drops: the reference
            # defect CC(C)(C)OOCCO -> 'ethan-1-ol' scores 10/9/1.5 = 0.74
            # on this proxy and so passes, while its REAL coverage is
            # 3/9 = 0.33. A sound gate needs producer-side atom records.
            #
            # (2) the missing-measurement default used to be 1.0, i.e. "no
            # measurement" read as "perfect coverage" -- fail-open. It is
            # LIVE, not latent: the chain-parent path above publishes a
            # candidate with handler='chain' and an EMPTY factors dict,
            # which clears both guards on this branch and then defaults to
            # 1.0. Missing is now None and handled explicitly. The gate
            # abstains on None (same control flow as the old 1.0, so no
            # emitted name changes) because it has nothing sound to gate
            # on -- an abstention, not a pass. Making it fail CLOSED here
            # would be a suppression change and needs measuring first.
            if (conf_score is not None
                    and conf_handler != 'unknown'
                    and conf_handler != 'retained_name'):
                _factors = conf_data.get('factors') or {}
                atom_cov = _factors.get('atom_coverage')
                if atom_cov is None:
                    logger.info(
                        "Atom coverage gate: ABSTAIN (unverified) -- no "
                        "atom_coverage was measured for handler=%s; not "
                        "reading a missing measurement as perfect coverage",
                        conf_handler,
                    )
                elif atom_cov < 0.55:
                    # Verify molecule has cleavable bonds before triggering
                    from .decomposition.bond_cleavage import find_cleavable_bonds
                    if find_cleavable_bonds(mol):
                        # Task 0.1: downgrade machinery engaged
                        # (P2.1 bucket).
                        from .metrics.abstention import (
                            AbstentionCode,
                            record_suppression,
                        )
                        record_suppression(AbstentionCode.COVERAGE_DOWNGRADE,
                                           detail='low_atom_coverage',
                                           candidate=assembled)
                        from .decomposition import try_decompose
                        decomp_name = try_decompose(mol, style=self.style)
                        if decomp_name and decomp_name != assembled:
                            logger.info(
                                "Atom coverage gate: rejecting low-coverage "
                                "name (atom_cov=%.4f < 0.55), using "
                                "decomposition fallback",
                                atom_cov,
                            )
                            return decomp_name

        # ============================================================
        # W8-P9 Task 9.2: organometallic structure-loss veto (ACCURACY-
        # CRITICAL, mandatory regardless of which other tasks land).
        # ============================================================
        # A covalent bond between a TRUE metal (Groups 1/2/3-12 -- see
        # has_covalent_metal_carbon_bond) and a carbon atom means the
        # compound IS a organometallic. If the ORGANOMETALLIC handler
        # did not produce the winning name (result.class_id != ORGANOMETALLIC),
        # the CFR cascade fell through to a class that has NO legitimate way
        # to name a true-metal-carbon compound -- it can only have "won" by
        # silently naming a DECOMPOSED SUB-FRAGMENT that drops the metal
        # (`C[Ti](Cl)(Cl)Cl` -> 'methane'; `Cl[Pt]1(Cl)...` -> '...ole', both
        # confirmed structure-loss leaks in the no-Java gate-off path, where
        # the OPSIN validity/ gate fails OPEN). Source-level veto: does
        # NOT depend on OPSIN/Java. Never fires for a dot-separated IONIC
        # metallocene/salt (has_covalent_metal_carbon_bond is False -- no bond
        # crosses fragments) or for Group 13-16 metalloids (excluded from the
        # predicate's scope; MONONUCLEAR_HYDRIDE / -04 heterocycle defer /
        # etc. are legitimate alternate paths for those, not leaks).
        if name and result.class_id != StoutClass.ORGANOMETALLIC:
            from .perception.metals import (
                has_covalent_metal_carbon_bond,
                has_metal_coordination_bond,
            )
            #: extend the veto to DATIVE-bonded coordination compounds
            # (Gd-DOTA/Mo-N2/Fe-siderophore) — a true metal that coordinates via
            # O/N/P dative bonds, which has_covalent_metal_carbon_bond misses
            # (no M-C bond). Same true-metal scope (metalloids excluded) so
            # boronic acids / silanes are untouched, and dative-only so ionic
            # and covalently-drawn salts stay nameable. Without this the
            # decomposition names a leftover ligand fragment and drops the metal
            # (a silent wrong-molecule the RT/E1 gates do not catch).
            if (has_covalent_metal_carbon_bond(mol)
                    or has_metal_coordination_bond(mol)):
                from .metrics.abstention import AbstentionCode, record_suppression
                record_suppression(AbstentionCode.GATE_SUPPRESSED,
                                   detail='organometallic_veto', candidate=name)
                return _descriptive_fallback(smiles)

        # ============================================================
        # W8-P4: oxoacid/anhydride atom-drop safety floor /.
        # ============================================================
        # Mirrors the organometallic veto above: when a WRONG/DROP structural
        # motif (see _p4_oxoacid_anhydride_leak_motif docstring) is present and
        # NO dedicated handler claimed the molecule (class_id fell all the way
        # through to GENERAL), the GENERAL/legacy pipeline can only have
        # "won" by silently naming a fragment that drops the motif's atoms
        # ('ethane' for CC(=O)OS) or mis-perceiving the chain constitution
        # ('1-(propanoyloxy)methanoic acid' for CC(=O)OC(=O)OC(=O)O). Source-
        # level: independent of OPSIN/Java (the RT-gate fails OPEN with no
        # Java). Scoped to class_id == GENERAL so a molecule any earlier CFR
        # entry (nucleoside, inorganic-acid table, anhydride bridge variants)
        # legitimately owns is never mis-vetoed.
        if name and result.class_id == StoutClass.GENERAL:
            if _p4_oxoacid_anhydride_leak_motif(mol):
                from .metrics.abstention import AbstentionCode, record_suppression
                record_suppression(AbstentionCode.GATE_SUPPRESSED,
                                   detail='oxoacid_anhydride_motif',
                                   candidate=name)
                return _descriptive_fallback(smiles)

        # ============================================================
        # W8-P10: Tier-B structure-conservation vetoes (Java-free; the "E1
        # atom-coverage certificate" from the cross-tool audit). Mirrors the
        # organometallic/oxoacid vetoes above -- runs UNCONDITIONALLY (never
        # gated by self._disable_opsin_validity_gate), so the raw/no-Java
        # path gets the same protection the OPSIN gate gives the
        # Java path (which fails OPEN without a JAR). See
        # perception/structure_conservation.py for the full design record
        # and the false-positive investigation behind each check's scope.
        # ============================================================
        # Scoped to TOP-LEVEL naming only (never a recursive substituent/
        # fragment naming call -- e.g. name_fragment_recursively names an
        # isolated charged fragment such as the boron-anion piece of
        # "trimethylboryl" as a standalone molecule up the call stack; that
        # inner name is not the final shipped name and must not be
        # suppressed here, or the fragment's "unknown organic compound"
        # replacement gets spliced into the middle of the parent name).
        from .assembly.fragment_naming import is_top_level_naming
        if name and not is_failure_name(name) and is_top_level_naming():
            from .perception.structure_conservation import (
                charge_dropped,
                fused_ring_atom_drop,
                partial_sat_sp3_substituent_drop,
            )
            # Charge-conservation veto (P5-family leak: a substituted/
            # hypervalent-at-the-charge-centre anion whose charge-aware
            # route fails internally and falls through to a charge-blind
            # path, e.g. `[I-](CCO)c1ccccc1`, `[B-](CCO)(C)(C)C`,
            # `C[Si-](C)(C)[H]`). NOT gated on result.class_id -- a live
            # probe showed CORRECT charged names ship through every class_id
            # (including GENERAL/ORGANOMETALLIC), so class-based gating would
            # over-veto; the charge is verified from the SHIPPED NAME's
            # suffix shape instead (see charge_dropped docstring).
            if charge_dropped(mol, name):
                from .metrics.abstention import AbstentionCode, record_suppression
                record_suppression(AbstentionCode.GATE_SUPPRESSED,
                                   detail='charge_dropped', candidate=name)
                return _descriptive_fallback(smiles)
            # R12-spillover veto: sp3-ring substituent silently dropped by
            # the partially-saturated fused-carbocycle emitter's downstream
            # enrichment hand-off (2,6-dimethyl-/2-amino-/2-hydroxy-
            # tetrahydronaphthalene-class leaks). Scoped internally to the
            # exact tetralin-class shape it is proven safe for (see
            # partial_sat_sp3_substituent_drop docstring) -- no class_id
            # gate needed since the check is a no-op unless
            # name_partially_saturated_carbocycle(mol) itself fires.
            if partial_sat_sp3_substituent_drop(mol, name):
                from .metrics.abstention import AbstentionCode, record_suppression
                record_suppression(AbstentionCode.GATE_SUPPRESSED,
                                   detail='partial_sat_sp3_substituent_drop',
                                   candidate=name)
                return _descriptive_fallback(smiles)
            # W8-P6 Task 6.0: bare-monocyclic-name atom-drop veto. A fused/
            # bridged/spiro polycyclic (2+ SSSR rings sharing an atom) can
            # never be correctly named by a bare "cyclo<stem>ane" parent
            # (e.g. a piperidine-fused cyclohexane silently named
            # '(3R,4R)-cyclohexane', dropping the whole N-ring). See
            # fused_ring_atom_drop docstring.
            if fused_ring_atom_drop(mol, name):
                from .metrics.abstention import AbstentionCode, record_suppression
                record_suppression(AbstentionCode.GATE_SUPPRESSED,
                                   detail='fused_ring_atom_drop',
                                   candidate=name)
                return _descriptive_fallback(smiles)
            # -A: Java-free valence guard — a von-Baeyer name that cites a
            # ring carbon as BOTH a double-bond terminus AND an oxo/-one carbon
            # is a 5-bond carbon (the caffeine oxo/ene class). catches it
            # WITH a jar; this source-level check catches it WITHOUT one, so
            # best-effort (T4) can never ship a valence-illegal name. Runs
            # unconditionally (never gated on the jar), like the other P10 vetoes.
            from .perception.structure_conservation import oxo_ene_valence_illegal
            if oxo_ene_valence_illegal(name):
                from .metrics.abstention import AbstentionCode, record_suppression
                record_suppression(AbstentionCode.GATE_SUPPRESSED,
                                   detail='oxo_ene_valence_illegal',
                                   candidate=name)
                return _descriptive_fallback(smiles)

        # Orthonym is deterministic-rules-only (-01,): there is no
        # ML fallback. The rule-based pipeline output is the final name.
        return name

    def _perceive(self, mol, smiles: str, canonical_smiles: str) -> MolecularFeatures:
        """
        Extract molecular features using RDKit.

        This is the perception layer - converts structure to features.
        """
        features = MolecularFeatures(
            mol=mol,
            smiles=smiles,
            canonical_smiles=canonical_smiles
        )

        # Add species type detection for ionic/radical compounds
        features.species_type = detect_species_type(mol)
        features.total_charge = Chem.GetFormalCharge(mol)

        if features.species_type in ('ion', 'zwitterion', 'salt'):
            features.ion_sites = get_ion_sites(mol)
        if features.species_type == 'radical':
            features.radical_sites = get_radical_sites(mol)

        # Detect functional groups using SMARTS patterns
        features.functional_groups = detect_functional_groups(mol)

        # Filter consumed atoms to prevent double-counting
        # (e.g., acid halide Cl should not also appear as "chloro" prefix)
        features.functional_groups = _filter_consumed_fg_atoms(features.functional_groups)

        # Detect ring systems
        features.ring_systems = get_ring_systems(mol)
        features.all_ring_atoms = get_complete_ring_atom_set(mol)
        features.is_cyclic = len(features.ring_systems) > 0

        # Check aromaticity
        for ring_system in features.ring_systems:
            if is_aromatic_ring(mol, ring_system):
                features.is_aromatic = True
                break

        # Detect multiple bonds
        features.double_bonds = self._find_double_bonds(mol)
        features.triple_bonds = self._find_triple_bonds(mol)

        # Assign CIP stereochemistry labels BEFORE extracting stereo info.
        # -03 (, Pitfall 2): route the SECOND CIP chokepoint through the
        # single source-of-truth assign_stereochemistry so this path can never
        # disagree with perception/stereo.py under ORTHONYM_USE_CENTRES_CIP=1.
        # assign_stereochemistry sets _CIPCode (rdCIPLabeler by default; centres
        # when gated ON + available) AND the _CIP_ASSIGNED_PROP idempotent marker
        # itself, so the explicit SetProp is no longer needed here.
        assign_stereochemistry(mol)

        # Extract stereochemistry (now depends on _CIPCode being set)
        features.stereocenters = get_stereocenters(mol)
        features.double_bond_stereo = get_double_bond_stereo(mol)

        return features

    def _classify(self, features: MolecularFeatures) -> None:
        """
        Apply IUPAC classification rules.

        Determines:
        - Principal characteristic group (highest seniority)
        - Principal chain/ring (following IUPAC 2013 criteria)
        - Substituent positions and identities
        """
        # Determine principal functional group
        pg_name, pg_atoms = get_principal_group(
            features.mol,
            features.functional_groups
        )
        # a phase T3: scoped principal-group override /. When the
        # charged chokepoint re-enters an S/P-oxoacid anion that coexists with a
        # SENIOR neutral acid (carboxylic), it forces the anion's acid group
        # (sulfonic/sulfinic/phosphonic) as principal so it becomes the suffix and
        # the carboxylic acid is demoted to a 'carboxy' prefix. Default None.
        if (self._principal_group_override
                and features.functional_groups.get(self._principal_group_override)):
            pg_name = self._principal_group_override
            pg_atoms = features.functional_groups[self._principal_group_override]
        features.principal_group = pg_name
        features.principal_group_atoms = pg_atoms

        # a phase: stash the controller flag + oracle onto features (the
        # features._ring_info transient-attribute pattern at:1532) so CandidatePool.add
        # can lift them onto the pool (BLOCKER #9 fix). WARNING #9: the oracle is required
        # for the runtime RT-safety gate. Both default to OFF/None (Stage A).
        features._enable_triviality_controller = self._enable_triviality_controller
        features._triv_oracle = self._triv_oracle
        # a phase: plumb the group-splitting flag + flag-ON-only oracle onto
        # features so the substituent_no_prefix_form tier-3 fallback in polyfunctional.py can read them.
        features._enable_group_splitting = self._enable_group_splitting
        features._split_oracle = self._split_oracle

        # Store ester match(es) if principal group is ester
        if pg_name == "ester" and pg_atoms:
            features.ester_match = pg_atoms[0]  # First ester match (backward compat)
            features.all_ester_matches = pg_atoms  # All ester matches

        # Detect polyfunctional compounds (multiple distinct FGs)
        from .rules.polyfunctional import detect_polyfunctional, get_non_principal_groups
        features.is_polyfunctional = detect_polyfunctional(
            features.mol, features.functional_groups
        )
        if features.is_polyfunctional:
            features.non_principal_groups = get_non_principal_groups(
                features.functional_groups, features.principal_group
            )

        # Parent selection for ALL cyclic molecules (a phase: no fused-heterocycle bypass).
        # cascade runs whenever a meaningful chain exists; cascade itself
        # enforces NP override (parent_selection.py:688-700) and
        # ring-on-tie tiebreaker (parent_selection.py:862-869).
        # Source: https://iupac.qmul.ac.uk/BlueBook/P4.html
        # Source: https://iupac.qmul.ac.uk/BlueBook/P5.html
        # Source: AUTONOM 1990 (full seniority cascade on ALL structures).
        if features.is_cyclic:
            from .rules.parent_selection import select_parent

            # Get ring atoms to exclude when finding chain
            all_ring_atoms = set()
            for ring in features.ring_systems:
                all_ring_atoms.update(ring)

            # Find potential principal chain (excluding ring atoms)
            # IMPORTANT: namer.py does the chain finding, then passes result to select_parent
            potential_chain = find_principal_chain(
                features.mol,
                features.functional_groups,
                features.principal_group,
                exclude_atoms=all_ring_atoms
            )

            # Only do parent selection if we found a meaningful chain (>= 2
            # carbons) OR (C- / V-4, a 1-carbon chain whose
            # principal-characteristic-group carbon is NOT ring-attached: the
            # ring then cannot express the PCG as a ring suffix, so the 1-carbon
            # parent (formamide / formic acid / methanal) must be parent-eligible
            # (else an N-aryl formamide falls to the ring -> 'carbamoylbenzene').
            # select_parent applies the same ring-neighbour test, so a
            # ring-attached single PCG carbon (benzaldehyde/benzamide) is
            # unaffected.
            _run_parent_selection = bool(potential_chain) and len(potential_chain) >= 2
            if (potential_chain and len(potential_chain) == 1
                    and features.principal_group and features.principal_group_atoms):
                _single = potential_chain[0]
                if not any(
                    nbr.GetIdx() in all_ring_atoms
                    for nbr in features.mol.GetAtomWithIdx(_single).GetNeighbors()
                ):
                    _run_parent_selection = True
                else:
                    # Wave2 T3a: skeletal suffixes (-ol/-amine/-thiol/-imine,...)
                    # have no exocyclic-carbon form, so a ring-attached single
                    # PCG carbon still cannot put the suffix on the ring.
                    # Parent selection is eligible when the single carbon IS a
                    # PG attachment atom and NO attachment atom is a ring atom
                    # (else the ring expresses the suffix itself — e.g. a
                    # sugar's ring -OH union keeps the ring parent).
                    from .rules.parent_selection import (
                        SKELETAL_SUFFIX_PGS,
                        _pg_attachment_atoms,
                    )
                    if features.principal_group in SKELETAL_SUFFIX_PGS:
                        _att: set = set()
                        for _m in features.principal_group_atoms:
                            _att.update(_pg_attachment_atoms(
                                features.principal_group, _m))
                        if _single in _att and not (_att & all_ring_atoms):
                            _run_parent_selection = True
            if _run_parent_selection:
                # a phase: delegate ring-type dispatch to helper.
                # Replaces the prior inline fused-hetero-only block; new
                # helper covers fused-hetero / PAH / benzene / simple-
                # hetero / spiro-stub / VB-stub / else->None per.
                _ring_info = _build_ring_info_for_parent_selection(features)
                # a phase: stash on features so downstream pool.add
                # call sites can read it without a signature change at
                # every composer.py call site (transient runtime
                # attribute; not a MolecularFeatures dataclass field
                # per; safe because the dataclass is not frozen).
                features._ring_info = _ring_info

                # Pass pre-computed chain to select_parent
                selection = select_parent(
                    mol=features.mol,
                    ring_systems=features.ring_systems,
                    principal_chain=potential_chain,
                    principal_group=features.principal_group,
                    principal_group_atoms=features.principal_group_atoms,
                    ring_info=_ring_info,
                    #: 0 for every normal call (byte-identical);
                    # >0 only on a best-effort offer-retry inner instance.
                    _offer_rank=self._forced_parent_rank,
                )
                features.parent_selection_result = selection  # a phase (V18 Appendix A.5)
                #: publish the top-level pool size so the offer-retry
                # can bound its loop at the real pool length. Guarded to the true
                # top-level molecule (visited set empty) so a substituent's own
                # parent selection never clobbers it.
                try:
                    from .assembly.fragment_naming import is_top_level_naming as _sp13_top
                    if _sp13_top():
                        self._top_parent_pool_size = getattr(
                            selection, 'parent_pool_size', 1)
                except Exception:
                    pass

                if selection.parent_type == 'chain':
                    features.chain_is_parent = True
                    # is_cyclic stays True -- ring data needed for ring-as-substituent naming (a phase)
                    features.principal_chain = selection.parent_atoms
                    features.ring_substituents_as_groups = selection.substituent_rings

        # For cyclic molecules, identify principal ring and its type
        if features.is_cyclic:
            # Check for ring assemblies FIRST (identical disconnected ring systems)
            # Must come before fused/polycyclic classification because ring assemblies
            # have 2+ separate ring systems that would otherwise be misrouted
            if len(features.ring_systems) >= 2:
                from .rules.ring_assemblies import detect_ring_assembly
                assembly_info = detect_ring_assembly(features.mol, features.ring_systems)
                if assembly_info:
                    features.ring_assembly_info = assembly_info
                    if not features.chain_is_parent:
                        return  # Skip other ring classification for assemblies

            # (a phase) chokepoint consolidation: compute the senior ring
            # system ONCE via select_principal_ring_system) and carry it
            # on the ParentSelectionResult, so the PAH guard, the among-rings
            # selector, and the derived senior_ring_system / principal_ring all
            # read ONE authoritative value instead of recomputing the identical
            # pure call. Behavior-preserving by construction (same args, same
            # pure function); this only de-duplicates two identical calls into one.
            from .rules.ring_selection import select_principal_ring_system
            _principal_ring_system = (
                select_principal_ring_system(features.mol, features.ring_systems)
                if features.ring_systems else None
            )
            # Carry it on the one authoritative result. `selection` is the same
            # ParentSelectionResult set by the select_parent call above (None when
            # there was no chain candidate, so select_parent never ran).
            selection = features.parent_selection_result
            if selection is not None:
                selection.principal_ring_system = _principal_ring_system

            # Check for polycyclic aromatics FIRST (naphthalene, anthracene, etc.)
            # These take precedence over single-ring classification
            from .rules.polycyclics import get_polycyclic_substituents, identify_polycyclic
            pah_name = identify_polycyclic(features.mol)
            if pah_name and len(features.ring_systems) >= 2:
                # task 9 (A-i): the PAH early-return must not preempt the
                # parent decision. With >=2 ring systems, take the PAH
                # route ONLY when the PAH system survives:
                # (a) — a principal characteristic group on a
                # DIFFERENT ring system makes that system the parent
                # (2-(naphthalen-2-yl)cyclohexan-1-ol, not naphthalene);
                # (b) — the among-rings winner must BE the PAH
                # system (2-(naphthalen-2-yl)furan: furan is senior).
                from .rules.parent_selection import is_principal_group_on_ring
                from .rules.polycyclics import get_polycyclic_core_atoms
                _core = get_polycyclic_core_atoms(features.mol, pah_name)
                _core_set = set(_core) if _core else set()
                if _core_set:
                    # Default False so the among-rings veto below runs
                    # unchanged when there is NO principal characteristic group
                    # (no PCG => none can sit on the PAH core). Assigned only
                    # inside the principal_group_atoms branch otherwise.
                    _pg_on_pah = False
                    if features.principal_group_atoms:
                        _pg_on_pah = is_principal_group_on_ring(
                            features.mol, _core_set,
                            features.principal_group_atoms,
                            features.principal_group,
                        )
                        _pg_on_other = any(
                            is_principal_group_on_ring(
                                features.mol, rs,
                                features.principal_group_atoms,
                                features.principal_group,
                            )
                            for rs in features.ring_systems
                            if not (set(rs) & _core_set)
                        )
                        if _pg_on_other and not _pg_on_pah:
                            pah_name = None
                    if pah_name:
                        #: reuse the one authoritative computation
                        _senior = _principal_ring_system
                        # / build precedes: when
                        # the principal characteristic group sits ON the PAH
                        # core, a senior heterocyclic SUBSTITUENT ring must not
                        # steal the parent (5-(1,3-dioxo-...-isoindol-2-yl)-
                        # naphthalene-1-carboxylic acid). Only the among-rings
                        # tie-break is skipped; already ran.
                        if (_senior and not (set(_senior) & _core_set)
                                and not _pg_on_pah):
                            pah_name = None
            if pah_name:
                features.polycyclic_name = pah_name
                features.polycyclic_substituents = get_polycyclic_substituents(
                    features.mol, pah_name,
                    principal_group=features.principal_group,
                )
                # Set ring type for consistency
                features.ring_type = 'aromatic'
                if not features.chain_is_parent:
                    return  # Skip other ring classification for PAHs

            ring_info = get_ring_info(features.mol)
            atom_rings = ring_info['atom_rings']

            # Wave2 determinism): among monocyclic candidate rings,
            # atom_rings[0] (first SSSR ring) is SMILES-order-dependent. When the
            # principal characteristic group sits on EXACTLY ONE ring, makes
            # that ring senior (it precedes the seniority tiebreak) — so move
            # it to the front deterministically. Without this, a molecule with two
            # equal rings where only one bears the PCG (e.g. 4-styrylbenzoic acid:
            # the COOH benzene vs the styryl phenyl) flipped parent by spelling,
            # dropping the acid to 'ethenylbenzene' for some orderings. Fused/PAH
            # systems returned earlier; chain-parent cases are unaffected (the PCG
            # then isn't on a ring). Tie (PCG on >1 ring) keeps SSSR order.
            if (atom_rings and len(atom_rings) >= 2
                    and features.principal_group_atoms
                    and not features.chain_is_parent):
                from .rules.parent_selection import is_principal_group_on_ring
                _pcg_rings = [
                    r for r in atom_rings
                    if is_principal_group_on_ring(
                        features.mol, set(r),
                        features.principal_group_atoms,
                        features.principal_group)
                ]
                if len(_pcg_rings) == 1 and _pcg_rings[0] is not atom_rings[0]:
                    atom_rings = [_pcg_rings[0]] + [
                        r for r in atom_rings if r is not _pcg_rings[0]
                    ]

            if atom_rings:
                # Select the most senior ring system per IUPAC
                # and store it for downstream use (e.g., ring-vs-chain
                # comparison in the composer). The monocyclic dispatch
                # path below uses atom_rings[0] which preserves SSSR
                # cyclic traversal order needed by orientation functions.
                # Complex multi-ring (fused/bridged) systems are handled
                # by _classify_complex_ring in the composer.
                #: reuse the one authoritative computation (see top of block)
                principal = _principal_ring_system
                features.senior_ring_system = principal if principal else atom_rings[0]

                # For multi-ring-system molecules, use a SSSR ring from
                # the senior system as principal_ring when the senior
                # system is strictly LARGER than the default ring's system.
                # This ensures fused/bridged senior systems (imidazopyridine,
                # xanthene) take precedence over small monocyclic rings
                # (benzene) per IUPAC.
                #
                #.1 S2: select_principal_ring_system, with the S1
                # unsaturation tiebreak) is now AUTHORITATIVE among ring
                # systems. The legacy ``size_diff >= 3`` heuristic that DISCARDED
                # the winner for equal/near-equal senior systems is deleted
                # (it produced (furan-2-yl)benzene instead of 2-phenylfuran and
                # pyridinylcyclohexane instead of 4-cyclohexylpyridine). Gated only
                # by the two correctness guards below; verified on the among-rings
                # PIN gold corpus (benchmarks/the gold set/among_rings_gold.json).
                if principal and len(features.ring_systems) >= 2:
                    senior_set = set(principal)
                    # Find the ring system that contains the default ring
                    default_system = set(atom_rings[0])
                    for rs in features.ring_systems:
                        if set(atom_rings[0]).issubset(rs):
                            default_system = rs
                            break

                    # Guard (a): the principal characteristic group must be
                    # IN the parent. If the PG sits on the default ring system,
                    # that ring stays parent regardless of ring seniority.
                    from .rules.parent_selection import is_principal_group_on_ring
                    pg_on_default = False
                    pg_on_senior = False
                    if features.principal_group_atoms:
                        pg_on_default = is_principal_group_on_ring(
                            features.mol, default_system,
                            features.principal_group_atoms,
                            features.principal_group,
                        )
                        # C4 +: a BRIDGING characteristic group
                        # (e.g. a secondary amine N bonded to a carbon of EACH
                        # ring — diaryl/aryl-heteroaryl amines) satisfies
                        # for BOTH ring systems. When it also sits on the SENIOR
                        # ring, no longer disqualifies the senior ring, so
                        # the seniority tiebreak decides -> the senior ring
                        # is the parent (pyridine over benzene ->
                        # N-phenylpyridin-4-amine, not (aminopyridinyl)benzene).
                        pg_on_senior = is_principal_group_on_ring(
                            features.mol, senior_set,
                            features.principal_group_atoms,
                            features.principal_group,
                        )

                    # Guard (b) Phase-171 / CHEBI:59269: when the CHAIN is the
                    # parent, both rings are mere substituents; reassigning
                    # principal_ring here perturbs stereodescriptor emission on the
                    # chain handler (the documented +1 self-test regressor of the
                    # 171 trial). Among-ring seniority does not decide a chain
                    # parent — keep the default ring. (See test_among_rings_gold.py
                    # TestPhase171StereoGuard.)
                    if ((not pg_on_default or pg_on_senior)
                            and not features.chain_is_parent
                            and senior_set != default_system):
                        best_ring = atom_rings[0]
                        best_overlap = 0
                        for ring in atom_rings:
                            overlap = len(set(ring) & senior_set)
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_ring = ring
                        features.principal_ring = best_ring
                    else:
                        # Wave-2 C2: when the PG sits on TWO OR
                        # MORE benzene rings (the dicyano-diaryl-ether BB
                        # example), atom_rings[0] is SMILES-order-dependent
                        # — the parent flipped with the spelling (a genuine
                        # NEW-NONDET gate catch). Break the tie with the
                        # deterministic substituent-count + canonical-rank
                        # selector; everything else keeps atom_rings[0].
                        _pg_ring_count = 0
                        if features.principal_group_atoms:
                            for rs in features.ring_systems:
                                if is_principal_group_on_ring(
                                        features.mol, set(rs),
                                        features.principal_group_atoms,
                                        features.principal_group):
                                    _pg_ring_count += 1
                        _det_ring = None
                        if _pg_ring_count >= 2:
                            from .rules.benzene import (
                                _select_benzene_parent_ring as _sbpr,
                            )
                            from .rules.benzene import (
                                is_benzene_ring as _ibr,
                            )
                            if all(_ibr(features.mol, r) for r in atom_rings):
                                #: the ring with more principal-group
                                # instances wins before any substituent count.
                                _det_ring = _sbpr(features.mol,
                                                  features.principal_group_atoms,
                                                  features.principal_group)
                        features.principal_ring = (
                            _det_ring if _det_ring is not None
                            else atom_rings[0]
                        )
                else:
                    features.principal_ring = atom_rings[0]
                features.ring_type = classify_ring(features.mol, features.principal_ring)

                # (BB 25207): the principal ketone-family group sits
                # in BOTH the ring and a pendant chain; the ring won parent
                # selection (more of the group, or ring default on tie), so an
                # off-ring match must NOT become a second ring '-one' suffix —
                # demote it to its chain, which the ring-substituent enumerator
                # then names as an oxo-alkyl prefix (2-oxobutyl). Fail-closed:
                # leaves principal_group_atoms unchanged on any ambiguity.
                if not getattr(features, 'chain_is_parent', False):
                    _demote_offring_principal_group_matches(
                        features, features.principal_ring
                    )

                # (b) /: a mancude monocyclic hydrocarbon
                # (all-carbon, RDKit-aromatic, NOT benzene, n>=7) is named as the
                # cyclo-polyene (cyclodeca-1,3,5,7,9-pentaene), never as an
                # [n]annulene component prefix and never dropped as "unknown".
                # RDKit marks it aromatic (10-pi Huckel), routing it to the
                # aromatic decline; kekulise it so it takes the existing
                # cycloalkene (polyene) path exactly like cyclooctatetraene.
                if features.ring_type == 'aromatic':
                    from .rules.cycloalkanes import (
                        is_mancude_monocyclic_hydrocarbon as _is_mancude_ring,
                    )
                    if _is_mancude_ring(features.mol, features.principal_ring):
                        _km = Chem.Mol(features.mol)
                        try:
                            Chem.Kekulize(_km, clearAromaticFlags=True)
                            features.mol = _km
                            features.ring_type = 'cycloalkene'
                        except Exception:
                            pass  # fail closed: leave as aromatic (declines)

                # Check if this is a benzene ring
                from .rules.benzene import get_benzene_substituents, is_benzene_ring
                if is_benzene_ring(features.mol, features.principal_ring):
                    features.is_benzene = True
                    # C- (V-3): with >1 benzene ring and no principal
                    # characteristic group (aralkyl/diaryl ethers, e.g. benzyl
                    # phenyl ether), `atom_rings[0]` is SMILES-order-dependent, so
                    # the parent flipped with the input spelling -> non-determinism
                    # ('(phenoxymethyl)benzene' vs 'benzoxybenzene'). Pick the
                    # parent ring deterministically (carbon-linked preference +
                    # canonical rank). Single-benzene and PCG-bearing rings are
                    # left to the existing principal-ring selection.
                    from .rules.benzene import (
                        _preferred_benzene_parent_ring,
                        _select_benzene_parent_ring,
                    )
                    _benzene_rings = [
                        r for r in features.mol.GetRingInfo().AtomRings()
                        if is_benzene_ring(features.mol, r)
                    ]
                    if len(_benzene_rings) > 1 and not features.principal_group:
                        # / (diaryl-linked-by-heteroatom):
                        # pick the parent whose COMPLETE name is preferred by
                        # alphanumerical order (R<S tie-break), naming each tied
                        # candidate. Falls back to the structure-only selector
                        # when the name-based choice is unavailable.
                        _pref = _preferred_benzene_parent_ring(features.mol)
                        if _pref is not None:
                            features.principal_ring = _pref[0]
                        else:
                            _bz = _select_benzene_parent_ring(features.mol)
                            if _bz is not None:
                                features.principal_ring = _bz
                    features.benzene_ring = features.principal_ring
                    features.benzene_substituents = get_benzene_substituents(
                        features.mol, features.principal_ring
                    )
                elif features.ring_type and features.ring_type.startswith('heterocyclic'):
                    # Heterocyclic ring: classify, detect substituents, and orient
                    from .rules.heterocycles import (
                        classify_heterocycle,
                        get_heterocycle_substituents,
                        orient_heterocycle_with_substituents,
                        ring_principal_suffix_atoms,
                    )

                    features.heterocycle_info = classify_heterocycle(
                        features.mol, features.principal_ring
                    )

                    # Detect substituent positions for proper orientation
                    ring_set = set(features.principal_ring)
                    sub_positions = set()
                    for idx in features.principal_ring:
                        atom = features.mol.GetAtomWithIdx(idx)
                        for neighbor in atom.GetNeighbors():
                            if neighbor.GetIdx() not in ring_set:
                                sub_positions.add(idx)
                                break

                    # / ring-suffix fix : identify the ring atoms bearing the
                    # principal characteristic group so the suffix gets the lowest
                    # locant (c)), after the heteroatom. Same extraction as
                    # the cycloalkene branch: a ring C whose FG-match heteroatom is
                    # exocyclic (C=O ketone, C-OH alcohol/phenol, C-NH2 amine,...).
                    pg_ring_atoms = set()
                    if features.principal_group:
                        from .rules.seniority import get_prefix as _pg_get_prefix
                        _pg_prefix = _pg_get_prefix(features.principal_group)
                        from .rules.seniority import SUFFIX_FORMS as _PG_SUFFIX_FORMS
                        # the ring suffix of the principal group is an N-substituted
                        # amide suffix ('-carboxamide'): the heterocycle assembler
                        # cites it. An ester or anhydride class is named by
                        # functional class nomenclature (a lactone ring by 'oxo'),
                        # so it cites no ring suffix to number first.
                        _pg_ring_suffix = (_PG_SUFFIX_FORMS.get(features.principal_group)
                                           or ("", ""))[1]
                        _pg_has_suffix_form = _pg_ring_suffix.endswith("carboxamide")
                        # The principal characteristic group is a CLASS, not a
                        # single FG label. A ring bearing an exocyclic primary
                        # alcohol (-CH2OH) plus secondary ring alcohols (ring -OH)
                        # has principal_group == primary_alcohol, but the RING
                        # carbons that carry the -ol SUFFIX are the secondary ones
                        # (the assembly already unions them via the shared 'hydroxy'
                        # prefix). Collect every same-prefix FG class so orient
                        # gives the suffix-bearing ring carbons the lowest locants
                        # (c)) DETERMINISTICALLY (alcohol-class union);
                        # without this the suffix-locant tie is broken by SMILES
                        # atom order -> non-deterministic numbering .
                        for _fg, _matches in features.functional_groups.items():
                            # A principal group with no prefix form but an amide ring
                            # suffix (an N-substituted amide: 'secondary_amide',
                            # 'tertiary_amide') still anchors its own matches: its suffix
                            # ('-carboxamide') takes the lowest locant before the
                            # prefixes, (c) (the Blue Book) --
                            # 'N,N,6-trimethylpyridine-3-carboxamide', not 'N,N,2-
                            # trimethylpyridine-5-carboxamide' ('2-chloropyridine-3-
                            # carboxamide (PIN)',:32895).
                            if (_fg != features.principal_group
                                    or _pg_prefix is not None
                                    or not _pg_has_suffix_form) and (
                                    _pg_prefix is None
                                    or _pg_get_prefix(_fg) != _pg_prefix):
                                continue
                            for match in _matches:
                                match_set = set(match)
                                if not (match_set & ring_set):
                                    # CASE A (twin of the cycloalkane branch
                                    # below, ~3627): a WHOLLY-EXOCYCLIC appended
                                    # suffix (-carboxylic acid, -carbaldehyde,
                                    # -carbonitrile, -carboxamide,...). The
                                    # entire FG match lies OFF the ring, so the
                                    # CASE-B ring-C-bonded-heteroatom test never
                                    # fires and pg_ring_atoms stayed empty --
                                    # the ring-direction tie for a symmetric-
                                    # heteroatom ring (furan/thiophene/pyrrole)
                                    # was then broken arbitrarily by canonical
                                    # rank, giving the suffix the HIGHER locant
                                    # (2-methylfuran-5-carboxylic acid). The
                                    # expressed-suffix ANCHOR is the ring atom
                                    # bonded to the match carbon; per (c)
                                    # it takes the lowest locant BEFORE the
                                    # detachable prefix (5-methylfuran-2-
                                    # carboxylic acid). Same-prefix
                                    # scoping is inherited from the enclosing
                                    # loop, so only principal-group-class
                                    # suffixes are anchored. (A shared helper
                                    # unifying this with the cycloalkane twin +
                                    # general_engine + parent-selection is a
                                    # tracked follow-up.)
                                    # NARROW (protect W6B-LOCK): anchor ONLY a
                                    # genuine C-APPENDED suffix -- the ring-bonded
                                    # match carbon must carry a characteristic
                                    # MULTIPLE bond to a non-carbon match atom
                                    # (C=O aldehyde/acid/amide, C#N nitrile). The
                                    # same-prefix union also admits an
                                    # exocyclic -CH2OH / -CH2NH2 whose ring-bonded
                                    # carbon is sp3 single-bonded: that is a
                                    # (hydroxymethyl)/(aminomethyl) PREFIX, not a
                                    # ring-appended -ol/-amine suffix, so it must
                                    # NOT be anchored (else the oxepane pentol
                                    # loses the lowest locants to the hydroxymethyl
                                    # carbon -- protect W6B-LOCK regression).
                                    for atom_idx in match:
                                        atom = features.mol.GetAtomWithIdx(atom_idx)
                                        if atom.GetSymbol() != 'C':
                                            continue
                                        _is_suffix_c = any(
                                            b.GetBondTypeAsDouble() > 1.0
                                            and b.GetOtherAtom(atom).GetIdx() in match_set
                                            and b.GetOtherAtom(atom).GetSymbol() != 'C'
                                            for b in atom.GetBonds()
                                        )
                                        if not _is_suffix_c:
                                            continue
                                        for nbr in atom.GetNeighbors():
                                            if nbr.GetIdx() in ring_set:
                                                pg_ring_atoms.add(nbr.GetIdx())
                                                break
                                    continue
                                for atom_idx in match:
                                    if atom_idx not in ring_set:
                                        continue
                                    atom = features.mol.GetAtomWithIdx(atom_idx)
                                    if atom.GetSymbol() != 'C':
                                        continue
                                    for nbr in atom.GetNeighbors():
                                        nbr_idx = nbr.GetIdx()
                                        if (nbr_idx not in ring_set
                                                and nbr_idx in match_set
                                                and nbr.GetSymbol() != 'C'):
                                            pg_ring_atoms.add(atom_idx)
                                            break

                    # (c) NUMBERING (the Blue Book): a principal
                    # characteristic group expressed as a SUFFIX takes the lowest
                    # locant BEFORE a detachable prefix (f). The same-prefix loop
                    # above anchors only a CARBON-centred appended suffix, so it
                    # MISSES two ring-suffix families whose suffix decision lives
                    # in get_heterocycle_substituents: (A) the pseudoketone
                    # -one/-dione/-thione ring carbons (principal group is a
                    # lactam-declined amide / cyclic imide / lactone-declined
                    # ester / thioamide, whose get_prefix is None) and (B) a ring
                    # atom bearing an exocyclic SULFUR-oxoacid suffix (S is not a
                    # carbon). Without these the (f) substituent set decided
                    # and put the SUBSTITUTED ring atom at locant 1
                    # (1-methylimidazolidine-2,5-dione, 1-methylpiperazine-4-
                    # sulfonic acid). Feed the suffix atoms exactly as the carboxy
                    # path does so (c) governs.
                    pg_ring_atoms |= ring_principal_suffix_atoms(
                        features.mol, features.principal_ring,
                        features.principal_group, features.functional_groups
                    )

                    # Orient considering heteroatoms, the principal group, then
                    # other substituents for lowest locants order).
                    oriented, atom_to_locant = orient_heterocycle_with_substituents(
                        features.mol, features.principal_ring, sub_positions,
                        principal_group_atoms=pg_ring_atoms if pg_ring_atoms else None
                    )
                    features.oriented_heterocycle = oriented
                    features.heterocycle_atom_to_locant = atom_to_locant

                    # Get substituent details for naming; pass the principal group so
                    # a senior FG on the ring is emitted as a SUFFIX, not a prefix.
                    features.heterocycle_substituents = get_heterocycle_substituents(
                        features.mol,
                        features.principal_ring,
                        oriented,
                        atom_to_locant,
                        principal_group=features.principal_group
                    )
                else:
                    # Non-benzene, non-heterocyclic ring: detect substituents and orient
                    from .rules.cycloalkanes import (
                        get_ring_double_bonds,
                        get_ring_substituents,
                        orient_cycloalkane,
                        orient_cycloalkene,
                    )

                    # Get ring substituents
                    features.ring_substituents = get_ring_substituents(
                        features.mol, features.principal_ring
                    )

                    # Get ring double bonds
                    features.ring_double_bonds = get_ring_double_bonds(
                        features.mol, features.principal_ring
                    )

                    # Determine principal group atoms on the ring
                    # IUPAC /: principal group gets lowest locant
                    # We identify ring C atoms that directly bear the principal
                    # group's characteristic heteroatom (e.g., C=O for ketone,
                    # C-OH for alcohol). The characteristic heteroatom must be
                    # bonded DIRECTLY to a ring carbon (not via an exocyclic C).
                    pg_ring_atoms = set()
                    if features.principal_group and features.principal_group in features.functional_groups:
                        from .rules.parent_selection import (
                            SKELETAL_SUFFIX_PGS as _SKEL_SFX_RING)
                        ring_set = set(features.principal_ring)
                        # Breadth job 3 review fixes: the principal characteristic
                        # group is the WHOLE equal-seniority class the suffix cites
                        # (DD5,, as on the chain path below -- the
                        # principal SUBTYPE's matches alone left a secondary or
                        # tertiary ring -OH out of the anchor set when a primary
                        # -CH2OH decided the subtype. And an -ol/-amine/-thiol/-one
                        # decorates a skeletal atom: it has no exocyclic-carbon
                        # suffix form (SKELETAL_SUFFIX_PGS), so a wholly exocyclic
                        # match (the -CH2OH of 'hydroxymethyl') is a substituent and
                        # anchors nothing. (c) (the Blue Book) numbers
                        # the suffixes first: '4-(hydroxymethyl)cyclohexan-1-ol',
                        # not '1-(hydroxymethyl)cyclohexan-4-ol'.
                        _pg_ring_matches = (features.principal_group_atoms
                                            or features.functional_groups[features.principal_group])
                        for match in _pg_ring_matches:
                            match_set = set(match)
                            if not (match_set & ring_set):
                                if features.principal_group in _SKEL_SFX_RING:
                                    continue
                                #.1 S4: wholly-exocyclic match = an
                                # APPENDED suffix (-carbaldehyde, -carboxylic
                                # acid, -carbonitrile). Its expressed-suffix
                                # ANCHOR (the ring atom bonded to the match
                                # carbon) takes the lowest locant per
                                # (2-methylcyclohexane-1-
                                # carbaldehyde, never 1-methyl-2-). Matches
                                # with no ring contact at all (a CHO at the
                                # end of a demoted chain) contribute nothing.
                                for atom_idx in match:
                                    atom = features.mol.GetAtomWithIdx(atom_idx)
                                    if atom.GetSymbol() != 'C':
                                        continue
                                    for nbr in atom.GetNeighbors():
                                        if nbr.GetIdx() in ring_set:
                                            pg_ring_atoms.add(nbr.GetIdx())
                                            break
                                continue
                            for atom_idx in match:
                                if atom_idx in ring_set:
                                    atom = features.mol.GetAtomWithIdx(atom_idx)
                                    if atom.GetSymbol() != 'C':
                                        continue
                                    # Check if this ring C is bonded to a non-ring
                                    # HETEROATOM that is in the FG match
                                    for nbr in atom.GetNeighbors():
                                        nbr_idx = nbr.GetIdx()
                                        if (nbr_idx not in ring_set
                                                and nbr_idx in match_set
                                                and nbr.GetSymbol() != 'C'):
                                            pg_ring_atoms.add(atom_idx)
                                            break

                    # v52 P1 Task D3 (c), the Blue Book): the loops
                    # above only anchor a CARBON-rooted appended suffix
                    # (-carboxylic acid, -carbaldehyde,...) -- ``atom.GetSymbol
                    # != 'C': continue`` skips every atom of a sulfonic_acid /
                    # sulfinic_acid match, whose root is S, not C. So a ring
                    # bearing -SO3H / -SO2H never got its suffix atom anchored
                    # and (f) (the substituent set) decided numbering
                    # instead, giving 'cyclohex-1-ene-4-sulfonic acid' where
                    # (c) requires the suffix at locant 1
                    # ('cyclohex-3-ene-1-sulfonic acid'). Reuse the SAME
                    # ring_principal_suffix_atoms anchor the heterocycle branch
                    # above already feeds for this exact family (its case B) --
                    # not a new mechanism, the shared one applied to the
                    # non-benzene/non-heterocyclic ring path too.
                    from .rules.heterocycles import ring_principal_suffix_atoms
                    pg_ring_atoms |= ring_principal_suffix_atoms(
                        features.mol, features.principal_ring,
                        features.principal_group, features.functional_groups
                    )

                    # Orient the ring based on type
                    if features.ring_type == 'cycloalkane':
                        features.oriented_ring = orient_cycloalkane(
                            features.mol,
                            features.principal_ring,
                            features.ring_substituents,
                            principal_group_atoms=pg_ring_atoms if pg_ring_atoms else None
                        )
                    elif features.ring_type == 'cycloalkene':
                        features.oriented_ring = orient_cycloalkene(
                            features.mol,
                            features.principal_ring,
                            features.ring_double_bonds,
                            features.ring_substituents,
                            principal_group_atoms=pg_ring_atoms if pg_ring_atoms else None
                        )
                        # Calculate ring double bond locants (wrap-aware:
                        # the closure bond positions (0, n-1) is locant n)
                        if features.oriented_ring and features.ring_double_bonds:
                            from .rules.cycloalkanes import ring_double_bond_locant
                            oriented = features.oriented_ring
                            n_ring = len(oriented)
                            locants = []
                            for a1, a2 in features.ring_double_bonds:
                                pos1 = oriented.index(a1)
                                pos2 = oriented.index(a2)
                                locants.append(
                                    ring_double_bond_locant(pos1, pos2, n_ring)
                                )
                            features.ring_double_bond_locants = sorted(locants)

        # Find principal chain (for acyclic molecules or chain-is-parent cyclic molecules)
        # Skip if chain was already set by parent selection (chain_is_parent = True)
        if not features.is_cyclic or features.chain_is_parent:
            if not features.chain_is_parent:
                # Normal acyclic molecule - find principal chain
                # Guard: exclude ring atoms even in "acyclic" path (defensive)
                _ring_exclude = set()
                ri = features.mol.GetRingInfo()
                for ring in ri.AtomRings():
                    _ring_exclude.update(ring)
                features.principal_chain = find_principal_chain(
                    features.mol,
                    features.functional_groups,
                    features.principal_group,
                    exclude_atoms=_ring_exclude if _ring_exclude else None
                )

            if features.principal_chain:
                # Collect principal group atom indices for orientation
                pg_atom_set: set = set()
                if features.principal_group and features.principal_group in features.functional_groups:
                    from .rules.parent_selection import (
                        SKELETAL_SUFFIX_PGS,
                        _pg_attachment_atoms,
                    )
                    # (c) (the Blue Book): the principal characteristic
                    # group gets the lowest locants, and that group is the WHOLE
                    # equal-seniority class the suffix cites -- DD5
                    # already unions primary/secondary/tertiary -ol (and the amine
                    # classes) into features.principal_group_atoms, which the suffix
                    # locants are counted from. Orienting on the principal SUBTYPE's
                    # matches alone left a secondary -OH out of criterion (a), so
                    # OCC=CCC(O)CCO tied {1,7}/{1,7} and fell to the ene locant:
                    # 'hept-2-ene-1,5,7-triol' (PIN 'hept-5-ene-1,3,7-triol',
                    # {1,3,7} < {1,5,7}; TRIAGE row 95 rt-14). Non-union groups have
                    # principal_group_atoms == their own matches (unchanged).
                    _pg_matches = (features.principal_group_atoms
                                   or features.functional_groups[features.principal_group])
                    _chain_set = set(features.principal_chain)
                    for match in _pg_matches:
                        if features.principal_group in SKELETAL_SUFFIX_PGS:
                            # task 9: the -one family's locant atom is
                            # the carbonyl carbon (PG_ATTACHMENT_INDICES);
                            # feeding the whole match (incl. the FLANKING
                            # carbon) made orientation criterion (a) tie at
                            # {1,2} both ways for 2-carbon ketone chains and
                            # fall through to alphabetics ('...ethan-2-one').
                            _match_atoms = set(_pg_attachment_atoms(
                                features.principal_group, match))
                        else:
                            _match_atoms = set(match)
                        pg_atom_set.update(_match_atoms)
                        # normalizes a match to (heteroatom, bearing C), and
                        # for an N-substituted amine that bearing C can be an
                        # N-alkyl carbon OFF the chain, leaving the instance
                        # invisible to criterion (a): CCN(CC)CCCC(C)N shipped
                        # 'N5,N5-diethylpentane-2,5-diamine' (PIN 'N1,N1-diethyl-
                        # pentane-1,4-diamine', cf. BB:4675). The suffix locant of
                        # such an instance is the chain atom bonded to its
                        # heteroatom, so anchor there. Only the heteroatom's own
                        # chain neighbours count (an -OH on a side-branch carbon
                        # is a substituent, never anchored).
                        if not (_match_atoms | set(match)) & _chain_set:
                            for _a in match:
                                _atom = features.mol.GetAtomWithIdx(_a)
                                if _atom.GetAtomicNum() in (1, 6):
                                    continue
                                pg_atom_set.update(
                                    nb.GetIdx() for nb in _atom.GetNeighbors()
                                    if nb.GetIdx() in _chain_set)

                # Get initial substituents for orientation criterion (d)
                # This is needed BEFORE orientation to apply lowest-locant rule
                initial_subs = self._find_substituents_by_atom(
                    features.mol,
                    features.principal_chain
                )

                # Orient chain using IUPAC 2013 criteria
                features.principal_chain = orient_chain(
                    chain=features.principal_chain,
                    mol=features.mol,
                    principal_group_atoms=pg_atom_set,
                    double_bonds=features.double_bonds,
                    triple_bonds=features.triple_bonds,
                    substituent_positions=initial_subs,
                )

                # Build atom-to-locant mapping from oriented chain
                features.atom_to_locant = build_atom_to_locant(features.principal_chain)

                # Get substituents with final locant-based positions
                features.substituents = self._find_substituents(
                    features.mol,
                    features.principal_chain
                )

        # ATOM-PARTITION invariant: a principal-group occurrence is a PARENT SUFFIX, not a
        # substituent's group. get_principal_group (:4123) gathers every same-prefix-class
        # occurrence molecule-wide (the alcohol-class union), so the tertiary OH of a
        # C(C)(C)O arm ON A RING is otherwise emitted as a second ring -ol, double-assigning its
        # atom (it is ALSO in ring_substituents) -> an atom-short, wrong parent
        # (cyclohexane-1,2-diol for a mono-ol ring). The suffix-vs-
        # prefix choice follows atom<->parent membership; the off-parent group's atoms
        # already flow to the
        # substituent enumerator and are named there.
        # internal notes.
        #
        # ⚠ POSITIVE EVIDENCE, not parent-absence. Applied AFTER parent selection, BEFORE
        # assemble_name reads principal_group_atoms (:3623). An earlier "strip if locant NOT in
        # my computed parent set" over-stripped legitimate suffixes whenever the parent-set
        # computation was incomplete (fused/spiro/orientation): it regressed 6 then 2 PIN gold
        # rows. This instead removes an occurrence ONLY when its locant carbon is provably inside
        # an ENUMERATED SUBSTITUENT -- a legitimate parent suffix carbon never is, so this cannot
        # regress a correct name. Scoped to SKELETAL suffixes (-ol/-amine/-thiol/-imine/-one),
        # whose locant IS a skeletal atom (parent_selection.py:45-52); appended suffixes
        # (-carboxylic acid/-carbaldehyde/...) have an exocyclic locant by design and are left
        # alone. Never empties the suffix set. E1 disjointness stays the backstop.
        # ⚠ Excludes amines/imine: N-suffixes carry special multi-N locant handling,
        # N1/N2 locants) and the composer already runs its OWN diamine principal_group_atoms
        # filter (composer.py:6461-6493). A generic strip here runs first and breaks it -- it
        # regressed exactly two amine PIN gold rows to abstention (NCCNCN, CN(C)CCN(C)CCN;
        # target_passes 1650). Alcohols/thiols/ketones have no such special handling. Measured.
        from .rules.parent_selection import SKELETAL_SUFFIX_PGS as _SKEL_SFX
        _PARTITION_SAFE = _SKEL_SFX - {
            "primary_amine", "secondary_amine", "tertiary_amine", "imine"}
        if (not os.environ.get("ORTHONYM_DISABLE_PARTITION_FILTER")
                and features.principal_group in _PARTITION_SAFE
                and features.principal_group_atoms):
            from .rules.parent_selection import _pg_attachment_atoms as _pgaa
            _sub_atoms: set = set()
            for _subdict in (getattr(features, 'ring_substituents', None),
                             getattr(features, 'substituents', None)):
                if not _subdict:
                    continue
                for _sublist in _subdict.values():
                    for _sub in _sublist:  # each _sub is a list of atom indices
                        _sub_atoms.update(_sub)
            if _sub_atoms:
                _kept = [
                    _m for _m in features.principal_group_atoms
                    if not (set(_pgaa(features.principal_group, tuple(_m))) <= _sub_atoms)
                ]
                if _kept and len(_kept) < len(features.principal_group_atoms):
                    features.principal_group_atoms = _kept

        #: Naming decision trace
        if logger.isEnabledFor(logging.DEBUG):
            parent_type = 'chain' if getattr(features, 'chain_is_parent', False) else 'ring'
            parent_atoms = (features.principal_chain if parent_type == 'chain'
                            else features.principal_ring or [])
            sub_count = (sum(len(v) for v in features.substituents.values())
                         if features.substituents else 0)
            logger.debug(
                "NAMING_DECISION: smiles=%s parent_type=%s parent_size=%d "
                "principal_group=%s fg_count=%d sub_count=%d is_cyclic=%s",
                features.canonical_smiles,
                parent_type,
                len(parent_atoms) if parent_atoms else 0,
                features.principal_group,
                len(features.functional_groups),
                sub_count,
                features.is_cyclic,
            )

    def _find_double_bonds(self, mol) -> List[tuple]:
        """Find all C=C double bonds."""
        double_bonds = []
        for bond in bonds_of(mol):
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                begin = bond.GetBeginAtom()
                end = bond.GetEndAtom()
                # Only C=C double bonds (not C=O, etc.)
                if begin.GetSymbol() == 'C' and end.GetSymbol() == 'C':
                    double_bonds.append((
                        bond.GetBeginAtomIdx(),
                        bond.GetEndAtomIdx()
                    ))
        return double_bonds

    def _find_triple_bonds(self, mol) -> List[tuple]:
        """Find all C≡C triple bonds."""
        triple_bonds = []
        for bond in bonds_of(mol):
            if bond.GetBondType() == Chem.BondType.TRIPLE:
                begin = bond.GetBeginAtom()
                end = bond.GetEndAtom()
                # Only C≡C triple bonds (not C≡N)
                if begin.GetSymbol() == 'C' and end.GetSymbol() == 'C':
                    triple_bonds.append((
                        bond.GetBeginAtomIdx(),
                        bond.GetEndAtomIdx()
                    ))
        return triple_bonds

    def _find_substituents(self, mol, chain: List[int]) -> Dict[int, List[List[int]]]:
        """Find substituents attached to the principal chain (keyed by position)."""
        from .perception.chains import get_substituents
        return get_substituents(mol, chain)

    def _find_substituents_by_atom(self, mol, chain: List[int]) -> Dict[int, List[List[int]]]:
        """
        Find substituents attached to the principal chain, keyed by atom index.

        This is used for orient_chain which expects substituent_positions
        keyed by atom indices on the chain, not by position numbers.

        Args:
            mol: RDKit Mol object
            chain: List of atom indices in the principal chain

        Returns:
            Dict mapping chain atom index to list of substituent atom lists
        """
        from .perception.chains import _bfs_substituent

        chain_set = set(chain)
        substituents = {}

        for chain_idx in chain:
            chain_atom = mol.GetAtomWithIdx(chain_idx)
            position_subs = []

            for neighbor in chain_atom.GetNeighbors():
                nbr_idx = neighbor.GetIdx()

                # Skip atoms that are part of the main chain
                if nbr_idx in chain_set:
                    continue

                # BFS to find full substituent
                sub_atoms = _bfs_substituent(mol, nbr_idx, chain_set)
                position_subs.append(sub_atoms)

            if position_subs:
                substituents[chain_idx] = position_subs

        return substituents


def _filter_consumed_fg_atoms(functional_groups: dict) -> dict:
    """Filter out functional group matches whose atoms are consumed by higher-priority groups.

    This prevents double-counting. For example, the Cl in an acid chloride
    (-C(=O)Cl) matches both the acid_chloride SMARTS and the chloro SMARTS.
    Without filtering, the Cl would appear as both "oyl chloride" (suffix) and
    "chloro" (prefix), producing incorrect names like "1-chloroethanoyl chloride"
    instead of "acetyl chloride".

    Rules:
        1. Acid halides consume halogens: remove chloro/bromo/fluoro matches
           where the halogen atom is part of an acid halide group.
        2. Anhydrides consume esters: remove ester matches where atoms overlap
           with an anhydride group.

    Args:
        functional_groups: Dict from detect_functional_groups.

    Returns:
        Filtered copy of functional_groups with consumed matches removed.
    """
    from .rules.acid_halides import get_acid_halide_consumed_atoms
    from .rules.anhydrides import get_anhydride_consumed_atoms

    fg = dict(functional_groups)  # shallow copy

    # --- Rule 1: Acid halides consume halogens ---
    halide_consumed = get_acid_halide_consumed_atoms(fg)
    if halide_consumed:
        for halogen_key in ("chloro", "bromo", "fluoro"):
            if halogen_key in fg:
                filtered = []
                for match in fg[halogen_key]:
                    # match = (halogen_idx, C_idx) from SMARTS [X][#6]
                    halogen_atom = match[0]
                    if halogen_atom not in halide_consumed:
                        filtered.append(match)
                if filtered:
                    fg[halogen_key] = filtered
                else:
                    del fg[halogen_key]

    # --- Rule 2: Anhydrides consume esters ---
    anhydride_consumed = get_anhydride_consumed_atoms(fg)
    if anhydride_consumed:
        if "ester" in fg:
            filtered = []
            for match in fg["ester"]:
                match_set = set(match)
                # If ANY atom in the ester match overlaps with anhydride, remove it
                if not match_set & anhydride_consumed:
                    filtered.append(match)
            if filtered:
                fg["ester"] = filtered
            else:
                del fg["ester"]

    return fg



def name_with_tree(smiles: str, style: str = "pin"):
    """Name one molecule and return the parts of the name.

    A shortcut for ``Orthonym(style=style).name_with_tree(smiles)``.

    Parameters
    ----------
    smiles: str
        The structure, as a SMILES string.
    style: {"pin", "general", "cas"}, default "pin"
        Naming style, as for:func:`name_compound`.

    Returns
    -------
    NamingResult
        The name (the same string:func:`name_compound` returns), the tree of its
        parts as a:class:`NameTreeNode`, and a map from atom index to locant
        where one was recorded.

    Raises
    ------
    ValueError
        If RDKit cannot read the SMILES.

    Examples
    --------
    >>> from orthonym import name_with_tree
    >>> result = name_with_tree("OC1CCCCC1")
    >>> result.name
    'cyclohexanol'
    >>> result.tree.parent_stem
    'cyclohex'
    >>> result.tree.suffix
    'ol'
    """
    return Orthonym(style=style).name_with_tree(smiles)


def name_compound(smiles: str, style: str = "pin",
                   include_confidence: bool = False,
                   *,
                   enable_triviality_controller: bool = False,
                   enable_group_splitting: bool = False,
                   trivial_fallback: bool = False,
                   general_fallback: Optional[bool] = None,
                   general_fallback_unverified: Optional[bool] = None,
                   allow_aromatic_general: Optional[bool] = None,
                   full_coverage: Optional[bool] = None,
                   raise_on_limit: bool = False,
                   binding_proof: str = "off"):
    """Name one molecule.

    Reads a structure written as SMILES and returns its IUPAC name as a string.
    At the default settings a name is returned only when the strict path for the
    Preferred IUPAC Name built it and verified it, or when it is one of the default
    tier's exceptions (the exact-match list names, the few name formats OPSIN cannot
    read, a PIN whose stereodescriptors OPSIN cannot read); otherwise, and where no
    name passes, it is a label such as ``'unknown organic compound'`` or
    ``'inorganic compound (not supported)'``. The wider tiers (``general_fallback``
    and the options below) also return names that are not the preferred name. Use
    :func:`orthonym.errors.is_failure_name` to tell a label from a name, and
    :meth:`Orthonym.name_tiered` to learn which tier a name earned.

    Parameters
    ----------
    smiles: str
        The structure, as a SMILES string.
    style: {"pin", "general", "cas"}, default "pin"
        Naming style. ``"pin"`` aims at the Preferred IUPAC Name. ``"general"``
        allows a few general IUPAC forms where the recommendations offer one (for
        example some adduct names and axial stereodescriptors). ``"cas"`` is
        accepted and at present gives the same names as ``"pin"``.
    include_confidence: bool, default False
        Return a dictionary instead of a string. Its ``"name"`` key holds the
        name, ``"handler"`` the part of the engine that built it, and
        ``"confidence"`` and ``"factors"`` a coverage score and its parts.
        ``"confidence"`` is ``None`` when no measurement was taken.
    enable_triviality_controller: bool, default False
        Where the recommendations prefer a retained parent name to the systematic
        one, use the retained name. Each change is checked by an OPSIN round trip.
        The environment setting ``ORTHONYM_ENABLE_TRIVIALITY_CONTROLLER=1`` turns
        this on as well.
    enable_group_splitting: bool, default False
        When an ester or thioester group inside a larger molecule has no prefix
        form, write it as its parts (``oxo`` plus ``ethoxy``, for example) instead
        of giving up. Each split name must pass an OPSIN round trip. The
        environment setting ``ORTHONYM_ENABLE_GROUP_SPLITTING=1`` turns this on
        as well.
    trivial_fallback: bool, default False
        When no preferred name can be built, also allow a retained trivial name
        that is not a preferred name. A molecule whose preferred name can be built
        keeps it. Turns on ``enable_triviality_controller``.
    general_fallback: bool or None, default None
        When the strict path declines, try the general naming engine (the
        ``valid`` tier of the command line). Its names must pass an atom-coverage
        certificate and a full-InChIKey round trip. ``None`` means "off" for a
        call you make; the engine uses it to pass the setting on when it names
        parts of a molecule.
    general_fallback_unverified: bool or None, default None
        Also switch on the ``best-effort`` tier: the last-resort producers, von
        Baeyer and spiro names for ring systems of up to 100 skeletal atoms and 11
        rings (the other tiers build these names for ring systems of up to 40
        skeletal atoms and 8 rings), and adducts with a one-atom ion. Their names still have to pass the full-InChIKey round trip. ``None`` as
        for ``general_fallback``.
    allow_aromatic_general: bool or None, default None
        Let the general engine also name aromatic and heterocyclic ring systems
        (the ``complete`` tier). ``None`` as for ``general_fallback``.
    full_coverage: bool or None, default None
        Also try the coordination-name builder for metal tetrapyrrole and corrin
        complexes (the ``full-coverage`` tier). It builds a name or declines.
        ``None`` as for ``general_fallback``.
    raise_on_limit: bool, default False
        Raise:class:`OrthonymLimitError` for a structure the engine cannot
        handle, instead of returning a label.
    binding_proof: {"off", "audit", "enforce"}, default "off"
        An extra check that every part of a general-engine name still maps onto
        the atoms it names in the final string. ``"audit"`` records the result
        and never changes the name; ``"enforce"`` also declines when the check
        fails.

    Returns
    -------
    str or dict
        The name, or a label that says why no name was given. A dictionary when
        ``include_confidence`` is true.

    Raises
    ------
    ValueError
        If RDKit cannot read the SMILES, or ``binding_proof`` is not one of the
        three values.
    OrthonymLimitError
        If ``raise_on_limit`` is true and the structure is out of scope.

    Examples
    --------
    >>> from orthonym import name_compound
    >>> name_compound("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O")
    '(2R)-2-[4-(2-methylpropyl)phenyl]propanoic acid'
    >>> name_compound("O=[U](=O)=O")
    'inorganic compound (not supported)'
    """
    if not include_confidence:
        from .assembly.fragment_naming import is_top_level_naming
        if not is_top_level_naming():
            from .assembly.nested_memo import cached_nested_call
            from .decomposition.engine import nacyl_float_refusing
            from .metrics.provenance import (allow_aromatic_general_ctx, best_effort_ctx,
                                             full_coverage_ctx, general_fallback_ctx)
            # nacyl_float_refusing: decision A's N-acyl float scope changes the name
            # (routing.dispatch_table._handle_peptide opens it), so it is a key part.
            _key = (smiles, style, enable_triviality_controller, enable_group_splitting,
                    trivial_fallback, general_fallback, general_fallback_unverified,
                    allow_aromatic_general, full_coverage, raise_on_limit, binding_proof,
                    general_fallback_ctx.get(), best_effort_ctx.get(),
                    allow_aromatic_general_ctx.get(), full_coverage_ctx.get(),
                    nacyl_float_refusing())
            return cached_nested_call("name_compound_nested", _key, lambda: _name_compound_impl(
                smiles, style, include_confidence,
                enable_triviality_controller=enable_triviality_controller,
                enable_group_splitting=enable_group_splitting, trivial_fallback=trivial_fallback,
                general_fallback=general_fallback, general_fallback_unverified=general_fallback_unverified,
                allow_aromatic_general=allow_aromatic_general, full_coverage=full_coverage,
                raise_on_limit=raise_on_limit, binding_proof=binding_proof))
    return _name_compound_impl(
        smiles, style, include_confidence,
        enable_triviality_controller=enable_triviality_controller,
        enable_group_splitting=enable_group_splitting, trivial_fallback=trivial_fallback,
        general_fallback=general_fallback, general_fallback_unverified=general_fallback_unverified,
        allow_aromatic_general=allow_aromatic_general, full_coverage=full_coverage,
        raise_on_limit=raise_on_limit, binding_proof=binding_proof)


def _name_compound_impl(smiles: str, style: str = "pin",
                   include_confidence: bool = False,
                   *,
                   enable_triviality_controller: bool = False,
                   enable_group_splitting: bool = False,
                   trivial_fallback: bool = False,
                   general_fallback: Optional[bool] = None,
                   general_fallback_unverified: Optional[bool] = None,
                   allow_aromatic_general: Optional[bool] = None,
                   full_coverage: Optional[bool] = None,
                   raise_on_limit: bool = False,
                   binding_proof: str = "off"):
    """
    Convenience function to generate IUPAC name from SMILES.

    Args:
        smiles: SMILES string
        style: Naming style
            - "pin": Preferred IUPAC Names (default, uses retained names when available)
            - "systematic": Always generate systematic name (bypass retained names)
            - "general": General IUPAC (more flexible)
            - "cas": CAS-style naming
        include_confidence: If True, return dict with confidence metadata
            instead of plain str
        binding_proof: proof mode forwarded to ``Orthonym`` -- see
            that constructor. Default "off" keeps every existing caller
            byte-identical; an invalid value raises ValueError there.

    Returns:
        str: IUPAC systematic name (default)
        dict: {'name': str, 'confidence': float, 'factors': dict, 'handler': str}
              when include_confidence=True

    Example:
        >>> name_compound("CCO")
        'ethanol'
        >>> name_compound("CC(=O)O")
        'acetic acid'
        >>> name_compound("c1ccccc1")
        'benzene'
        >>> name_compound("C=CCO")
        'prop-2-en-1-ol'
        >>> name_compound("C=CCO", trivial_fallback=True)
        'prop-2-en-1-ol'
        >>> name_compound("CCO", include_confidence=True)
        {'name': 'ethanol', 'confidence': 1.0,...}
    """
    #: inherit the engine flag from the propagation ctx when the
    # caller didn't say — fragment/component recursion re-enters through
    # here with a FRESH namer, and the top-level opt-in must carry through.
    if general_fallback is None:
        try:
            from .metrics.provenance import general_fallback_ctx
            general_fallback = general_fallback_ctx.get()
        except Exception:
            general_fallback = False
    #: inherit the REST of the best-effort tier the same way. Before
    # this, only `general_fallback` carried through the recursive re-entry;
    # `general_fallback_unverified` (published as `best_effort_ctx`) and
    # `allow_aromatic_general` (published as `allow_aromatic_general_ctx`) were
    # silently downgraded, so a compound substituent nameable ONLY at the
    # unverified/aromatic-general tier abstained when reached recursively even
    # though the whole molecule was named at that tier. An explicit True/False
    # from the caller still wins; None means "inherit the top-level tier".
    if general_fallback_unverified is None:
        try:
            from .metrics.provenance import best_effort_ctx
            general_fallback_unverified = best_effort_ctx.get()
        except Exception:
            general_fallback_unverified = False
    if allow_aromatic_general is None:
        try:
            from .metrics.provenance import allow_aromatic_general_ctx
            allow_aromatic_general = allow_aromatic_general_ctx.get()
        except Exception:
            allow_aromatic_general = False
    #: inherit the full-coverage opt-in through the recursive
    # re-entry the same way (a FRESH namer would otherwise lose a top-level
    # full_coverage=True on fragment recursion). An explicit True/False from
    # the caller still wins; None means "inherit the top-level tier".
    if full_coverage is None:
        try:
            from .metrics.provenance import full_coverage_ctx
            full_coverage = full_coverage_ctx.get()
        except Exception:
            full_coverage = False

    namer = Orthonym(
        style=style,
        enable_triviality_controller=enable_triviality_controller,
        enable_group_splitting=enable_group_splitting,
        trivial_fallback=trivial_fallback,
        general_fallback=general_fallback,
        general_fallback_unverified=general_fallback_unverified,
        allow_aromatic_general=allow_aromatic_general,
        full_coverage=full_coverage,
        binding_proof=binding_proof,
    )

    if include_confidence:
        try:
            return namer.name_with_confidence(smiles)
        except ValueError:
            raise
        except Exception as e:
            logger.warning("name_with_confidence failed: %s", e)
            # C4: naming raised, so nothing measured anything. Use the one
            # shared unmeasured record (confidence=None,
            # verification='unverified') instead of a fabricated 0.0, and
            # include the 'limit'/'abstention' keys the happy path always sets
            # so consumers see one stable shape.
            from .assembly.coverage_scoring import unmeasured_confidence
            _md = unmeasured_confidence(name=_descriptive_fallback(smiles),
                                        handler='fallback')
            _md['limit'] = None
            _md['abstention'] = None
            return _md

    try:
        result = namer.name(smiles, raise_on_limit=raise_on_limit)
        if result:
            # Check if result contains 'unknown' as a component (partial failure)
            if 'unknown' in result.lower():
                # A default-tier decline (NO_VERIFIED_PIN) keeps its own label: the
                # structure's elements are supported, so the UNSUPPORTED_ELEMENT
                # text of the descriptive fallback would misstate the reason.
                _stash = getattr(namer, "_default_tier_row", None)
                if (_stash is not None and _stash[0] == result
                        and _stash[1].get("limit_code") == "NO_VERIFIED_PIN"):
                    return result
                return _descriptive_fallback(smiles)
            return result
        # If name returned empty/None, generate descriptive fallback
        return _descriptive_fallback(smiles)
    except ValueError:
        raise  # Re-raise ValueError (invalid SMILES) for caller to handle
    except (TypeError, KeyError, IndexError, AttributeError) as e:
        # Graceful fallback for unexpected errors in the naming pipeline.
        # Log the error type for debugging but return a fallback name rather
        # than crashing or returning None.
        logger.debug(
            "Naming error for %s: %s: %s", smiles, type(e).__name__, e
        )
        return _descriptive_fallback(smiles)


def name_pipeline_only(smiles: str, style: str = "pin"):
    """Name a molecule using only the systematic pipeline, skipping decomposition.

    This provides the non-decomposition name without consuming any depth budget
    on decomposition probes. Used by the decomposition engine to compare its
    result against what the systematic pipeline would produce.

    Returns:
        IUPAC name string, or None if naming fails.
    """
    # M1 (Lever E): memoize the whole pipeline call, ctx-keyed. The
    # decomposition engine calls this re-entrantly on the same (smiles, style)
    # within one molecule (~9 dup calls / 0.85 s marginal on the perf sample). A
    # naive (smiles, style) key is PROVEN output-changing -- the result depends on
    # the ambient tier contextvars (1 same-key-different-result observed), so those
    # four vars are IN the key. push_scope creates a scope iff none is open (a
    # standalone call), else shares the outer name scope. Fail-open + verify-mode
    # checked; plan internal notes Step 2.
    from .assembly.memo import cache_or_compute, pop_scope, push_scope
    from .metrics.provenance import (
        allow_aromatic_general_ctx,
        best_effort_ctx,
        full_coverage_ctx,
        general_fallback_ctx,
    )

    def _body():
        try:
            namer = Orthonym(style=style)
            return namer._name_impl(smiles, _skip_decomposition=True)
        except OrthonymLimitError:
            # G0 (DD7 S1,): a fragment-level fail-closed refusal is "no
            # name", not a crash — propagate as None so the decomposition engine
            # treats the fragment as out-of-scope and tries another strategy.
            # Caught explicitly (before the broad except) so the intent is
            # documented and a genuine fragment bug is not silently conflated with
            # a legitimate refusal.
            return None
        except Exception:
            return None

    _memo_token = push_scope()
    try:
        from .decomposition.engine import nacyl_float_refusing
        # nacyl_float_refusing: decision A's N-acyl float scope (see name's key).
        _key = (smiles, style, general_fallback_ctx.get(), best_effort_ctx.get(),
                allow_aromatic_general_ctx.get(), full_coverage_ctx.get(),
                nacyl_float_refusing())
        return cache_or_compute("name_pipeline_only", _key, _body)
    finally:
        pop_scope(_memo_token)


# Metals/inorganic elements and the metal-name map now live in orthonym.errors
# (single source of truth) and are re-exported via the module-top import so
# data.cation_words keeps importing them from orthonym.namer.
# _ORGANIC_ELEMENTS / _METAL_NAMES are bound at module top.


def _coordination_retained_name(smiles: str) -> Optional[str]:
    """ Milestone D1: exact-InChIKey retained name for an N-coordinated metal
    tetrapyrrole (heme / chlorophyll / cobalamin / siroheme / coenzyme F430), or
    ``None`` when the input is not one of the curated structures.

    OPSIN 2.9.0 cannot parse ANY of these coordination-complex names, so no real
    namer can ever produce an OPSIN-verified name for them; control otherwise lands
    on the honest ``"<metal> compound (not supported)"`` sentinel (or, when the OPSIN
    validity gate is degraded, a WRONG best-effort fragment such as
    ``"...hexadec-2-en-1-yl propanoate"`` for chlorophyll a). The only sound 0-wrong
    oracle for the class is therefore exact-structure identity: each table key is the
    standard InChIKey of one exact ChEBI structure and each value is that structure's
    ChEBI-accepted name (verbatim) -- the same contract as ``RETAINED_METALLOCENES``
    and the amino-acid/sugar retained tables. A hit is an exact structural match; a
    miss returns ``None`` and the caller keeps its own (abstaining) result, so this
    can never emit a wrong name.

    A cheap bracketed-metal-token pre-check keeps the InChIKey cost off the
    ~everything-else naming path (``[Fe`` / ``[Mg`` / ``[Co`` / ``[Ni`` uniquely mark
    an Fe/Mg/Co/Ni atom in a SMILES; no organic bracket atom begins with those).
    """
    if not smiles:
        return None
    if not ("[Fe" in smiles or "[Mg" in smiles or "[Co" in smiles or "[Ni" in smiles):
        return None
    try:
        from .data.coordination_retained import COORDINATION_RETAINED
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return COORDINATION_RETAINED.get(inchikey_of(mol))
    except Exception:  # pragma: no cover - a lookup must never break naming
        return None


def _descriptive_fallback(smiles: str) -> str:
    """Generate a descriptive fallback message instead of bare 'unknown'.

    For inorganic/metallic compounds, returns a descriptive message like
    'gold compound (not supported)'. For organic molecules that failed
    naming, returns 'unknown organic compound'.

     (a phase): this now delegates to ``errors.classify_failure_limit``
    — the single classifier that also backs the named ``OrthonymLimitError``
    catalog. The returned ``.message`` is byte-identical to the strings this
    function returned before (so the default always-emit output is unchanged);
    the structured code/ref it carries is surfaced only via the opt-in
    ``raise_on_limit`` / ``classify_limit`` paths.

    Args:
        smiles: The SMILES string that could not be named.

    Returns:
        A descriptive string (never bare 'unknown' for a parseable molecule).
    """
    # D1: before emitting the honest metal sentinel, consult the exact-InChIKey
    # coordination retained-name table. A hit (heme/chlorophyll/cobalamin/siroheme/
    # F430) returns the ChEBI-accepted name; a miss falls through to the sentinel,
    # exactly as before. Reached only after every real namer declined, so a molecule
    # the normal namer names is never touched here.
    _coord = _coordination_retained_name(smiles)
    if _coord is not None:
        return _coord
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return "unknown"
    return classify_failure_limit(mol, smiles=smiles).message


def classify_limit(smiles: str) -> Optional[OrthonymLimitError]:
    """Say whether a structure is out of scope, without raising.

    A wildcard atom (``*``) is refused at once. Any other structure is named with
    the default settings; if that gives a label instead of a name, the reason is
    returned.

    Parameters
    ----------
    smiles: str
        The structure, as a SMILES string.

    Returns
    -------
    OrthonymLimitError or None
        The reason, with its code (for example ``UNSUPPORTED_ELEMENT``,
        ``WILDCARD_ATOMS``, or ``NO_VERIFIED_PIN`` when the default settings built
        a name that is not a verified preferred IUPAC name) and message. ``None``
        when the structure gets a name, and also when RDKit cannot read the SMILES
        at all (that is a reading error, not a scope limit).

    Examples
    --------
    >>> from orthonym import classify_limit
    >>> classify_limit("O=[U](=O)=O").code
    'UNSUPPORTED_ELEMENT'
    >>> classify_limit("CC*").code
    'WILDCARD_ATOMS'
    >>> classify_limit("CCO") is None
    True
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None
    if mol is None:
        return None
    scope = classify_scope_limit(mol)
    if scope is not None:
        scope.smiles = smiles
        return scope
    namer = Orthonym()
    try:
        produced = namer.name(smiles)
    except Exception:
        produced = None
    if is_failure_name(produced):
        # The default tier built a name but declined it (not a verified PIN).
        stash = getattr(namer, "_default_tier_row", None)
        if (stash is not None and stash[0] == produced
                and stash[1].get("limit_code") == "NO_VERIFIED_PIN"):
            return no_verified_pin(produced, smiles=smiles)
        return classify_failure_limit(mol, smiles=smiles)
    return None
