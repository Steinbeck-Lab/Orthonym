"""
Main IUPAC nomenclature generator.

Architecture:
    SMILES → Perception → Classification → Assembly → IUPAC Name

This is the inverse of OPSIN's pipeline:
    OPSIN:     Name → Tokenize → Parse → Build Structure
    Orthonym: Structure → Perceive → Classify → Assemble Name
"""

import contextvars
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from .errors import (
    OrthonymLimitError,
    classify_scope_limit,
    classify_failure_limit,
    is_failure_name,
    is_refusal_sentinel,
    _ORGANIC_ELEMENTS,
    _METAL_NAMES,
)

logger = logging.getLogger(__name__)

# Phase 168 D-08: env-var override for the triviality controller. Read at import
# time so ORTHONYM_ENABLE_TRIVIALITY_CONTROLLER=1/true/yes/on flips the default for
# all Orthonym instances (mirrors the Phase 162 env-gate pattern).
_TRIV_ENV = os.environ.get("ORTHONYM_ENABLE_TRIVIALITY_CONTROLLER", "").strip().lower()
_DEFAULT_TRIV = _TRIV_ENV in ("1", "true", "yes", "on")

# Phase 169 D-05: env-var override for group-splitting. Read at import time so
# ORTHONYM_ENABLE_GROUP_SPLITTING=1/true/yes/on flips the default for all instances.
_GS_ENV = os.environ.get("ORTHONYM_ENABLE_GROUP_SPLITTING", "").strip().lower()
_DEFAULT_GS = _GS_ENV in ("1", "true", "yes", "on")

# CR-04 part B + W7: per-call NamingResult capture slot for name_with_tree.
# ContextVar provides thread-local AND asyncio-task-local isolation per PEP 567;
# safer than a module-global dict against concurrent Orthonym().name_with_tree()
# calls. The slot is installed by Orthonym.name_with_tree() before invoking
# self.name(); composer._assemble_name_impl writes the inner-dispatch
# NamingResult into the slot when present (default=None makes the regular
# Orthonym.name() path a no-op).
_name_with_tree_capture: "contextvars.ContextVar[Optional[Dict[str, Any]]]" = (
    contextvars.ContextVar("name_with_tree_capture", default=None)
)

from .perception.ions import detect_species_type, get_ion_sites, get_radical_sites
from .perception.functional_groups import detect_functional_groups
from .perception.chains import find_principal_chain
from .perception.rings import get_ring_systems, get_ring_info, is_aromatic_ring, classify_ring, get_complete_ring_atom_set
from .perception.stereo import get_stereocenters, get_double_bond_stereo, _CIP_ASSIGNED_PROP, assign_stereochemistry
from .rules.seniority import get_principal_group
from .rules.locants import orient_chain, build_atom_to_locant
from .rules.stereochemistry import collect_stereodescriptors, format_stereodescriptor_string
from .assembly.composer import assemble_name
from .assembly.offer_pool import (
    PIN_VERIFIED, SYSTEMATIC_VERIFIED, BEST_EFFORT, ABSTAIN,
)
from .data import ALL_RETAINED_NAMES as RETAINED_NAMES

# Phase 158 NEW: routing substrate. `StoutClass` + `ClassDispatchResult`
# are imported eagerly at module-load time so `_name_impl`'s
# cascade-continuation + GENERAL fallback can reference them by name. The
# `routing` sub-package is callable-only (no callbacks back into namer)
# so this is NOT a circular import — the predicate factories + handler
# shims inside `routing/dispatch_table.py` use lazy imports for
# `orthonym.rules.*` and `orthonym.namer.Orthonym` per
# PATTERNS § 3 + namer.py:853 lazy-import precedent.
from .routing.dispatch_table import StoutClass, ClassDispatchResult


# ---------------------------------------------------------------------------
# Universal stereo backstop (Phase 140, STER-16)
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

    Phase 177 WSB-01 (D-04/D-05/D-06): the backstop is flipped from detect-only
    to REAL injection on the only cohort carrying an authoritative parent
    locant map — `chain` and NON-phenol `benzene` (threaded via the per-call
    confidence dict as a POST-HOC CandidateName field). When the (now D/L-aware,
    Plan 01 Pattern D) predicate fires AND the handler is allow-listed AND a
    valid `atom_to_locant` is threaded, it calls the existing
    `inject_stereo_from_locant_map` (whose D-09 guard rejects None/empty/non-
    positive maps, degrading to a no-op).

    For `complex_ring`, `polycyclic`, `heterocycle`, `unknown`, phenol benzene,
    or no map -> it stays LOG-ONLY (the pre-177 detect-only behavior): no
    authoritative numbering exists for those, and a fabricated locant is worse
    than a missing one (D-06/D-09: missing beats wrong). This is the path the
    E-stilbene `unknown` case keeps (descriptor-less output is EXPECTED per the
    WSB-01 LOG-ONLY PROTECT, not a regression — the lipid/steroid reservoir flip
    is WS-C-gated, D-16).

    Args:
        mol: RDKit Mol object (with stereo info from original SMILES)
        name: Generated IUPAC name (may or may not contain stereo)
        handler: Name of the handler that produced this name (for attribution)
        atom_to_locant: Authoritative {atom_idx: 1-indexed locant} for the parent
            (threaded POST-HOC from the winning candidate). None for non-allow-
            listed handlers -> log-only.
        is_phenol_benzene: True when a benzene parent is a phenol (excluded from
            the inject allowlist per D-05).

    Returns:
        The name with injected stereodescriptors (allow-listed cohort) or the
        original name unchanged (log-only cohort).
    """
    from .rules.stereochemistry import (
        needs_stereo_injection, inject_stereo_from_locant_map,
    )

    if not needs_stereo_injection(mol, name):
        return name

    # WSB-01 (D-05): inject allowlist = chain + NON-phenol benzene ONLY, and
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
        # The D-09 guard inside the injector rejected the map (None/empty/
        # non-positive) or there was nothing to inject -> fall through to the
        # log-only branch (a missing descriptor beats a wrong one).

    # W4-S1 (P-93.2.2 / P-93.2.3 / P-93.2.6): a SINGLE stereogenic unit on a
    # NON-CARBON centre (sulfoxide S, quaternary N+, silane Si, chiral P, ...)
    # that the parent atom_to_locant never covers -> the descriptor is dropped
    # by the allow-list path above (the centre is a substituent atom, a
    # characteristic-group heteroatom, or a mononuclear hetero parent, none of
    # which appear in the carbon-parent locant map). Such a lone hetero
    # stereocentre carries NO locant (a single unambiguous stereogenic unit;
    # the same P-91 locant omission as the STEREO-06 mononuclear-methane case
    # `(R)-bromo(chloro)(fluoro)methane`), so emit the bare "(R)-"/"(S)-".
    # ACCURACY GATES (missing beats wrong): fire ONLY on EXACTLY ONE R/S atom
    # with ZERO E/Z bonds (never drops a second descriptor), and ONLY when that
    # centre is a non-carbon atom, so this can never emit a locant-less
    # descriptor for a ring/chain carbon that requires a numeric locant.
    # DETERMINISTIC: the CIP code is engine-assigned (order-independent) and
    # there is no locant to compute.
    _rs_atoms = [a for a in mol.GetAtoms() if a.HasProp('_CIPCode')]
    _bond_stereo = sum(1 for b in mol.GetBonds() if b.HasProp('_CIPCode'))
    if (
        len(_rs_atoms) == 1
        and _bond_stereo == 0
        and _rs_atoms[0].GetSymbol() != 'C'
    ):
        _cip = _rs_atoms[0].GetProp('_CIPCode')
        if _cip in ('R', 'S', 'r', 's'):
            return f"({_cip})-{name}"

    # Predicate said True but no authoritative injection happened -> log gap.
    n_atom_stereo = sum(1 for a in mol.GetAtoms() if a.HasProp('_CIPCode'))
    n_bond_stereo = sum(1 for b in mol.GetBonds() if b.HasProp('_CIPCode'))
    # v29 P3-CLOSEOUT Item D: ...unless the name expresses its configuration
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
# v29 P3-CLOSEOUT Item D: the descriptor channels the stereo backstop is blind to
# ---------------------------------------------------------------------------
#
# `needs_stereo_injection`'s descriptor vocabulary is `{R,S,r,s,E,Z}` inside
# parentheses, plus `alpha|beta-D/L-` (the ANOMERIC sugar form) and a bare
# `D-`/`L-` token.  The Blue Book's PIN descriptor vocabulary is wider.
# `## **P-91.2.1.2.2** Stereodescriptors used in the nomenclature of natural
# products` (BlueBookV2.md:44626) legitimises, at :44628-44632, "*(i) The
# descriptors 'D' and 'L' ... for carbohydrates, amino acids and peptides, and
# cyclitols; (ii) '*erythro*' and '*threo*' ...; (iii) The stereodescriptors
# 'alpha', 'beta' are used in the nomenclature of natural products to describe
# the absolute configuration of alkaloids, terpenes and terpenoids, steroids*".
#
# So a name can be configurationally COMPLETE through a channel this predicate
# cannot read, and the backstop then warns about a correct name.  That was 1147
# of the gate's warnings, over 122 distinct names.
#
# WHY THIS IS A DIAGNOSTIC-ONLY EXEMPTION AND NOT A WIDENING OF
# `needs_stereo_injection`.  That predicate gates BOTH this log AND the
# injection path above it, and `rules/stereochemistry.py:456` carries an
# explicit "*Per D-20, do NOT broaden*".  Widening it would stop injection
# firing on names that currently receive a correct descriptor block — trading a
# noisy log for real stereo loss (session invariant 11: removing a wrong output
# can unmask a worse one).  This check therefore sits at the LOG site only and
# changes no emitted name.
#
# NOT exempted, deliberately: the free amino acids (`alanine`, `serine`,
# `cysteine`, `cystine`, `threonine`, `isoleucine`, `allo-threonine`,
# `allo-isoleucine`).  There the warning is TRUE — `## **P-103.1.3.1** The
# stereodescriptors 'D' and 'L'` (:54291) requires the configuration at the
# alpha-carbon to be designated, `### **P-103.3.4** Indication of configuration
# in peptides` (:54715) scopes L-omission to PEPTIDES only, and Table 10.4 pairs
# each retained name with a `rel-` (RELATIVE) systematic equivalent (:54204,
# :54211), so the bare name is not enantiospecific.  Left warning on purpose.

# `### **P-101.2.6** Stereochemical configuration of parent structures`
# (:51045): "*The stereodescriptors 'alpha', 'beta', and 'xi' ... are cited
# before the name of the fundamental parent structure*", and `**P-101.2.6.1**`
# (:51051) "*Each chirality center is described by the stereodescriptor 'alpha',
# 'beta', or 'xi'*".  A locant-prefixed alpha/beta IS a stereodescriptor.
_ALPHA_BETA_DESCRIPTOR_RE = re.compile(r"\d+(?:alpha|beta|xi)\b")

# `## **P-105.1** RETAINED NAMES OF NUCLEOSIDES` (:54943) retains exactly these
# seven, whose full ribofuranosyl configuration is implied by the name; the
# nucleotide stems come from `## **P-106.1** RETAINED NAMES` (:55003).  BB's own
# examples are descriptor-free, e.g. `uridine 5'-(tetrahydrogen triphosphate)`
# (:55029) and `2',3',5'-tri-O-acetyladenosine` (:54981).
#
# v29 P3-FINAL I10: DERIVED from the producer table, not copied from it.  The
# nucleoside half was a hand-written duplicate of `rules/nucleosides.py`'s
# `_NUCLEOSIDE_STEM` values, so the two could drift, and the docstring below
# asserted they could not.  Deriving it also removes the surface a mutation used:
# appending `'itol'`/`'neuraminic'` to a literal tuple silenced `xylitol` and the
# neuraminic acids -- the exact three names this predicate must leave VISIBLE --
# and survived all 92 tests.  There is no longer a literal tuple to append to.
def _nucleoside_stems() -> Tuple[str, ...]:
    """Retained nucleoside stems (P-105.1) + nucleotide stems (P-106.1)."""
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
    # `## **P-106.1** RETAINED NAMES` (:55003) -- the nucleotide `-ylic acid`
    # stems, which have no constructor table of their own in this repo.  Kept
    # explicit and separately cited rather than folded into the derived set.
    # `thymidine` is retained by `## **P-105.1** RETAINED NAMES OF NUCLEOSIDES`
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
            NAME_EXACT_NP_PARENTS, NATURAL_PRODUCT_DERIVATIVES,
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
#   * a plain substring test is too loose -- `tropane` contributes the
#     4-character stem `trop`, which matches inside
#     `1,6-anhydro-beta-D-al-trop-yranose`, a carbohydrate with no steroid in it
#     (the single spurious hit across 1963 gold + pack + table names);
#   * requiring a boundary for EVERY stem is too strict -- substitutive names
#     concatenate the prefix straight onto the stem, so `androst` in
#     `17beta-hydroxy-androst-4-en-3-one` (spelled `hydroxyandrost...`) is
#     preceded by a letter and would be missed.  An existing test caught this.
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

# `### **P-107.4.3.1**` (:55227) retains `sphinganine` "*for the aliphatic amino
# alcohol HAVING THE DESCRIBED ABSOLUTE CONFIGURATION*", and (:55231) gives
# `(4E)-sphing-4-enine` under `### **P-107.4.3** Glycosphingolipids`.
#
# v29 P3-FINAL I10: this was a SIX-CHARACTER substring test, `'sphing' in name`,
# whose first clause (`'sphinganine' in name`) was dead by subsumption.  It
# matched anything containing `sphing`, and a mutation to `'sph'` -- which
# matches every `phosph...` name in the corpus -- survived all 92 tests.  Now an
# anchored match against the two retained spellings the Blue Book names.
_SPHINGOID_RETAINED_RE = re.compile(r"\bsphing(?:anine|-?\d*-?enine)\b")


def _stereo_is_implied_by_name(name: str) -> bool:
    """Does ``name`` already express its configuration, by a channel the
    R/S + E/Z counter cannot see?

    Diagnostic predicate only — it suppresses a false-positive WARNING and never
    changes an emitted name.

    Every leg is anchored to a table that PRODUCES these names, so it cannot
    drift from the producers.  v29 P3-FINAL corrected this sentence, which was
    FALSE for three of the five legs when written: the steroid leg was a bare
    `\\d+(?:alpha|beta|xi)` regex with no parent requirement, the nucleoside leg
    was an unbounded substring scan over a hand-copied duplicate of
    `rules/nucleosides.py`'s table, and the sphingoid leg was the six-character
    `'sphing' in name`.  A false in-code assertion is the class this range was
    written to delete, so it is repaired rather than trimmed.

    It also cannot see COMPLETENESS, and the citation it rests on says
    completeness is required.  `### **P-101.2.6**` (:51045) in full: the name of a
    fundamental parent structure "*implies the absolute configuration of all
    chirality centers **and the configuration of double bonds, when applicable**,
    without further specification.  **All chirality must be defined**...*"  The
    ellipsis in the previous version of this comment removed both the double-bond
    clause and the completeness obligation.  A partially-described name in an
    exempted family would therefore be silenced; no producer reachable today
    emits one (~20 in-family constructions with an extra undescribed element all
    fail closed or fall back to a fully systematic name), but nothing structurally
    prevents it.  The primitive that could check --
    `stereochemistry.count_expressed_stereo_descriptors` -- counts only
    parenthesised R/S/E/Z tokens and so returns 0 for `5alpha-cholestan-3beta-ol`,
    `myo-inositol` and `adenosine`, i.e. it cannot read the alpha/beta or
    implied-parent channels at all.  Closing this needs a family-implied-count
    oracle (implied + expressed == defined), not that primitive.
    """
    if not name:
        return False

    _low = name.lower()

    # Family 6 — `chalcone` (v29 Task L).  `**P-64.2.1.1**` (:28297), under the
    # heading `### **P-64.2.1** Retained names`, ends: "*Chalcone refers only to
    # the trans- or (E)- stereoisomer.*"  The retained name therefore IS the
    # descriptor for its one E/Z unit — exactly the "channel the R/S + E/Z
    # counter cannot see" this predicate exists to cover.  Without this leg every
    # chalcone emission logs "has 0 R/S + 1 E/Z but name lacks descriptors",
    # which is a FALSE POSITIVE: the name is configurationally complete, and
    # P-101.2.6's completeness obligation is met because the molecule has exactly
    # one stereogenic unit and the name implies it.
    #
    # EXACT equality, not a substring: it is anchored to the sole producer, the
    # `"O=C(/C=C/c1ccccc1)c1ccccc1": "chalcone"` row in `data/retained_names.py`,
    # which is keyed on the whole-molecule isomeric canonical SMILES and so can
    # only ever emit this one bare name.  A substring test would wrongly exempt a
    # substituted `...chalcone...` name if that class is ever built — and those
    # names may legitimately need descriptors for ADDITIONAL stereogenic units.
    if _low == "chalcone":
        return True

    # Family 4 — steroid/terpenoid alpha/beta: the name DOES carry descriptors
    # (`cholest-5-en-3beta-yl hydrogen sulfate`, `5alpha-cholestan-3beta-ol`).
    # Produced by `rules.steroid_stereo.collect_steroid_alpha_beta`, whose own
    # module docstring cites P-101.2.6.
    #
    # BOTH conditions are required.  The locanted descriptor alone exempted any
    # name containing `3beta`, with no steroid parent anywhere in it; anchoring it
    # to a stereoparent stem is what makes the leg keyed off the producer.
    if (_ALPHA_BETA_DESCRIPTOR_RE.search(name)
            and _has_steroid_stereoparent_stem(_low)):
        return True

    # Family 1 — inositols.  `## **P-104.1** DEFINITIONS` (:54821): "*Inositols
    # have retained names and ... employ the stereodescriptors 'D' and 'L' to
    # describe configurations*"; `**P-104.2.1**` (:54831): "*Stereoisomeric
    # inositols are described by adding italicized prefixes at the front of the
    # name 'inositol'. ... Names denoted by the prefixes are preferred.*"  The
    # prefix IS the descriptor for the five meso forms.  Same table as the
    # OPSIN-validity carve-out below.
    try:
        from .rules.inositols import INOSITOL_NAMES
        if name in INOSITOL_NAMES:
            return True
    except Exception:
        pass

    # Family 3 — retained nucleosides/nucleotides and their derivatives.
    # Anchored: the stem must END at a word boundary, so `2',3',5'-tri-O-
    # acetyladenosine` (BB :54981) still matches while an unrelated longer word
    # cannot be swallowed whole.
    if _NUCLEOSIDE_STEM_RE is not None and _NUCLEOSIDE_STEM_RE.search(_low):
        return True

    # Family 5 — stereoparent hydrides.  `**P-101.2.1.3**` (:50991) defines a
    # stereoparent as a parent that "*should include as much configuration as
    # possible*"; `### **P-101.2.6**` (:51045) is the governing sentence: "*The
    # name of a fundamental parent structure usually implies the absolute
    # configuration of all chirality centers **and the configuration of double
    # bonds, when applicable**, without further specification.  **All chirality
    # must be defined**...*"
    #
    # v29 P3-FINAL: a third citation, `### **P-101.3.6**` (:52106,
    # "*Stereochemistry implied by the name of the stereoparent structure remains
    # the same, unless otherwise specified*"), is REMOVED as out of scope.  The
    # sentence is verbatim, but its section heading is "*Removal of a terminal
    # ring.*" and the same paragraph ends "*This use of 'des' is restricted to
    # steroids*" -- it governs `des`-prefixed steroid names, not stereoparent
    # implication in general.  P-101.2.6 already carries the point.
    #
    # Membership is `**P-101.2.7**` Table 10.1 (:51377), approached through the
    # THREE tables that emit these names (see the note below), so no new
    # hardcoded list.  NB `NATURAL_PRODUCT_DERIVATIVES` is WIDER than Table 10.1
    # -- it carries derivative and trivial names that are not fundamental parent
    # hydrides -- so this leg is broader than P-101.2.7 alone would license.
    # Recorded rather than narrowed: it is diagnostic-only, and narrowing it
    # would restore warnings on names that do carry implied configuration.
    try:
        from .data.natural_products import (
            NAME_EXACT_NP_PARENTS, NATURAL_PRODUCT_DERIVATIVES,
            NATURAL_PRODUCT_SCAFFOLDS,
        )
        if name in NAME_EXACT_NP_PARENTS:
            return True
        if name in set(NATURAL_PRODUCT_DERIVATIVES.values()):
            return True
        # All three are needed: `NAME_EXACT_NP_PARENTS` deliberately excludes the
        # parents that DO round-trip in OPSIN (`tropane`, `prostane`,
        # `thromboxane`), and `tropane`/`stigmastane` live only in the scaffold
        # table.  Keying off all three covers Table 10.1 without a new list.
        if name in {
            _e['name'] for _e in NATURAL_PRODUCT_SCAFFOLDS.values()
            if isinstance(_e, dict) and 'name' in _e
        }:
            return True
    except Exception:
        pass

    # `### **P-107.4.3.1**` (:55227): "*the retained name 'sphinganine' for the
    # aliphatic amino alcohol HAVING THE DESCRIBED ABSOLUTE CONFIGURATION ...
    # is preferred to the systematic name (2S,3R)-2-aminooctadecane-1,3-diol*",
    # and (:55231) `(4E)-sphing-4-enine`.
    if _SPHINGOID_RETAINED_RE.search(_low):
        return True

    return False


# ---------------------------------------------------------------------------
# Universal OPSIN-grammar backstop (Phase 156)
# ---------------------------------------------------------------------------


def _final_grammar_check(name: str, smiles: Optional[str], handler: str,
                         grammar, stats: Dict[str, int]) -> str:
    """Universal OPSIN-grammar backstop: validate + (round-trip-gated) repair.

    Mirrors the `_final_stereo_check` shape (Phase 152 D-04 pattern).
    Single chokepoint per CONTEXT.md D-13 — never per-handler-exit
    (AP-6). On unrepairable failure, log WARNING and return ORIGINAL
    name (D-11, D-15). Never silently mutate.

    Args:
        name: The assembled IUPAC name (post stereo backstop).
        smiles: Original SMILES the name was generated from. Forwarded
            to `OpsinGrammar.suggest_fix` for the round-trip gate per
            CONTEXT.md D-09 / D-10. May be None on cold paths; the
            grammar layer documents the degraded-path contract.
        handler: Handler attribution string (per Phase 152 D-04
            pattern); used in WARNING logs only.
        grammar: An `OpsinGrammar` instance (or None when the
            `_disable_grammar_validation` kwarg was passed at
            construction time per D-14).
        stats: The per-instance counter dict (D-17 / AP-19) shared by
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

    # validate() rejected — try suggest_fix (D-09 round-trip-gated).
    # D-10 LOCKED signature: name FIRST, source_smiles SECOND.
    repaired, repair_class = grammar.suggest_fix(name, source_smiles=smiles)

    if repaired is not None and repaired != name:
        # The grammar layer already incremented the matching
        # `repair_succeeded_<class>` bucket internally per D-17.
        logger.warning(
            "OPSIN grammar repair: handler=%s class=%s original=%r repaired=%r",
            handler, repair_class, name[:50], repaired[:50],
        )
        return repaired

    # validate() rejected and no class produced a round-trip-passing
    # candidate. Log unrepairable WARNING and fall back to ORIGINAL
    # name per D-11 + D-15 (never silently mutate).
    logger.warning(
        "OPSIN grammar validation failed: handler=%s name=%r",
        handler, name[:50],
    )
    return name


# ============================================================================
# Phase 169.5 SUB-03: pre-emission OPSIN-parse validity gate (D-11..D-14)
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
# the gate's own tests re-enable it). FAIL-OPEN on no-JAR (D-13).
_VG_ENV = os.environ.get("ORTHONYM_DISABLE_OPSIN_VALIDITY_GATE", "").strip().lower()
_DISABLE_VALIDITY_GATE = _VG_ENV in ("1", "true", "yes", "on")

# SELF-01 (v23 Phase 0): the constitutional self-consistency gate. After the
# parseability gate confirms OPSIN ACCEPTS a name, re-perceive it: if OPSIN parses
# the name to a CONSTITUTIONALLY DIFFERENT molecule than the input, the name is
# wrong (names a different compound) and is suppressed to the honest fallback.
#   * "warn" — log + count would-suppressions WITHOUT changing output (the mandatory
#              warn-only validation phase: 0 protect-row false-positives before flip).
#   * "on"   — suppress on a verified constitutional mismatch (the accuracy-first state).
#   * "off"  — disabled.
# Comparison is STEREO-INSENSITIVE and TAUTOMER-/AROMATICITY-TOLERANT (standard
# InChIKey skeleton block, ADR-18-07): a stereo-only or mobile-H-tautomer difference
# must NEVER suppress a constitutionally-correct name. Fail-OPEN on every inconclusive
# outcome (OPSIN can't be consulted, either structure unparseable by RDKit).
# Flipped "warn" -> "on" 2026-06-22 after warn-only validation: the constitutional gate
# would-suppress 0/79 protect-set rows (zero false-positives), while correctly flagging
# the wrong-molecule TARGET cases (C[Ti](Cl)(Cl)Cl->"methane", [O-]C(=O)CN->"acetate", ...).
# Accuracy-first: a name for a different molecule is suppressed to the honest fallback.
_SC_DEFAULT = "on"
_SC_MODE = os.environ.get("ORTHONYM_SELF_CONSISTENCY_GATE", _SC_DEFAULT).strip().lower()
if _SC_MODE not in ("off", "warn", "on"):
    _SC_MODE = _SC_DEFAULT

# v33 Phase 0 L0/L1: producer-agnostic coverage AUDIT at the `_finish` choke.
# "off" disables it entirely; "shadow" (the default) records a verdict but
# NEVER changes the returned name; "veto" (shipped in L1, `6e495eb1`) is a
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
    """The SELF-01 verdict `_final_opsin_validity_gate` ALREADY computed for
    THIS EXACT `name` -- reused by the L0/L1 coverage audit instead of a
    second OPSIN call (CARRIED RULING, `task-L0-brief.md`) -- PLUS an explicit
    "is a fresh re-anchor even worth trying" decision (v33 Phase 0 L0 review
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

        * ``complete``: ``True``/``False`` when SELF-01 ALREADY answered for
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
                    _pv.GATE_OUTCOME_SELF01_CONSTITUTION_ONLY):
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


# Module-level singleton so the OPSIN parse cache is shared across all name()
# calls in a process (lazy-init on first use).
_VALIDITY_ORACLE = None

# W3-P06 (P-65.7.6.1): dianhydride/polyanhydride PIN word-form — >=2 acid-like
# words then a NUMERICALLY-multiplied '...anhydride' ('diacetic butanedioic
# dianhydride'). Used ONLY by the validity gate to carve out these OPSIN-
# unparseable-but-correct-by-construction names (see _final_opsin_validity_gate).
_DIANHYDRIDE_PIN_RE = re.compile(
    r"^(?:[a-z0-9][a-z0-9,()'\-]* ){2,}(?:di|tri|tetra|penta|hexa)anhydride$"
)

# W8-P4 (P-65.7.6.4.1): the chalcogen di-/poly-anhydride sibling of the above —
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

# W3-P07 (P-65.6.3.3.3.2 method (1)): the functional-class polyol polyester PIN —
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

# W3-P10 (P-67.2.2.2): di-/polynuclear noncarbon-oxoacid PREFIX-derivative PINs —
# locant-prefixed class groups on a di/hypodi/tri parent oxoacid
# ('1,3-diamidodiphosphoric acid'). These are correct-by-construction (emitted
# ONLY by the hard-gated rules.inorganic_acids.name_polyacid_derivative off an
# exact backbone), but OPSIN has a locant-assignment BUG for prefix substituents
# on these polyacid parents: it stacks '1,3-diamido' both on P-1 (verified — OPSIN
# parses the analogous '1,3-diimido...' correctly, but '1,3-diamido'/'1,3-dichloro'
# to the wrong constitution). The Blue Book is the sole PIN authority; OPSIN's
# re-parse bug must not gate our correctness (same rationale as the dianhydride /
# inositol carve-outs, applied to the SELF-01 constitutional self-consistency check
# rather than the unparseable path). Used ONLY by the validity gate.
_POLYACID_PREFIX_DERIVATIVE_PIN_RE = re.compile(
    r"^(?:\d+(?:,\d+)*-(?:di|tri|tetra|penta|hexa)?"
    r"(?:amido|imido|hydrazido|nitrido|chloro|bromo|fluoro|iodo|cyanato|azido|"
    r"thio|seleno|telluro))+"
    r"(?:hypo)?(?:di|tri|tetra)"
    r"(?:phosphoric|phosphonic|sulfuric|selenic|telluric|arsoric) acid$"
)

# W8-P9 (P-69.2.3/.4/.6): additive/coordination organometallic PINs contain
# eta/kappa/mu multicentre descriptors (or '-ido' anionic-ligand words) that
# OPSIN-2.9's generation grammar cannot parse. These are emitted ONLY by the
# hard-gated organometallic assembler (rules.organometallics) -> correct by
# construction. Same rationale as the inositol/dianhydride/thioperoxol carve-outs.
# CR guard (Task 9.1 open question): tightened so it never un-suppresses an
# unrelated malformed name -- requires EITHER an eta/kappa/mu descriptor OR a
# '-ido' coordination-ligand token, AND the name must end in a known P-69
# metal stem (optionally followed by a Stock/Ewens-Bassett suffix).
_ORGANOMETALLIC_ADDITIVE_PIN_RE = re.compile(
    r"^(?=.*(?:[ηκμ]|(?:chlorido|bromido|fluorido|iodido|hydrido)))"
    r".*(?:titanium|chromium|iron|nickel|molybdenum|tungsten|manganese|"
    r"platinum|iridium|mercury|osmium|ruthenium|cobalt|vanadium|rhodium)"
    r"(?:\([0-9IVX+\-]+\))?$"
)

# W8-P8 (P-26.2/.3): the phane simplified-skeletal PIN
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

# W8-P5 Task 2 (P-72.3/P-72.8.1, BB 41102-41110/41303): a SUBSTITUTED
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

# W8-P4 (P-65.7/P-67 atom-drop safety floor): source-level structural motifs
# reproduced (2026-07-18) to make the RAW (no-Java, gate-OFF) namer silently
# DROP atoms or emit a WRONG constitution, because no handler in this cycle's
# cascade claims them fully. Each is the EXACT verified-open shape from the
# W8-P4 ledger reproduction -- never a general "any P/S/N present" heuristic.
# See _p4_oxoacid_anhydride_leak_motif() below for the full per-motif citation.
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
      * acyl-O-[S,Se]H thio/seleno-peroxy acid (P-65.1.5.3) -- no peroxy-acid
        base suffix exists this cycle (OQ-5); 'CC(=O)OS' raw -> 'ethane'
        (drops S+O2).
      * -NH-NO2 nitramido on carbon (P-67.1.4.3.2) -- substituent-prefix build
        deferred; 'O=[N+]([O-])NCC(=O)O' raw -> 'ethanoic acid' (drops
        -NH-NO2).
      * an ACYCLIC 'naked carbonate' carbon -O-C(=O)-O- with NO carbon
        substituent, embedded in a larger chain (P-67.3.1) -- the substitutive
        nested-prefix engine ('{[(acetyloxy)carbonyl]oxy}formic acid') is not
        built this cycle; 'CC(=O)OC(=O)OC(=O)O' raw -> a WRONG constitution
        (parses to a smaller, different molecule -- confirmed via OPSIN RT of
        the shipped string). Ring atoms excluded so a genuinely-handled cyclic
        carbonate (1,3-dioxan-2-one) never fires.
      * an unclaimed di-nuclear noncarbon-oxoacid (P/S/Se) backbone (P-67.2.5.2
        partial ester / P-67.2.6 substituent-prefix on a senior acid) -- both
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
            _name_sulfonic_anhydride, _name_chalcogen_anhydride,
            _name_peroxy_anhydride,
        )
        if (_name_sulfonic_anhydride(mol) is None
                and _name_chalcogen_anhydride(mol) is None
                and _name_peroxy_anhydride(mol) is None):
            return True
    return False


def _validity_gate_jar_present() -> bool:
    """JAR-presence PROBE for the SUB-03 fail-OPEN guard (D-13).

    Independent of parse: OpsinOracle.name_to_smiles returns None for BOTH
    "JAR absent" and "parse failed", so the gate cannot infer JAR-absence from
    a parse result. This probe MUST be checked FIRST — else a no-Java env would
    suppress EVERY name (the OPPOSITE of OpsinOracle.rt_safe's fail-CLOSED).
    Monkeypatched in the gate's unit tests.
    """
    from .validation.opsin_roundtrip import _find_opsin_jar
    return _find_opsin_jar() is not None


def _validity_gate_status(name: str) -> str:
    """Cached 3-valued OPSIN parse outcome for the SUB-03 gate (CR-01).

    Returns ``"parsed"`` | ``"rejected"`` | ``"unavailable"``. Delegates to
    ``OpsinOracle.parse_status`` via a module-level singleton so the parse cache
    is shared process-wide. ``"unavailable"`` (subprocess timeout / OSError /
    no-JAR) MUST be treated as fail-OPEN by the caller — a transient OPSIN
    failure must never suppress a valid name. Monkeypatched in tests.
    """
    global _VALIDITY_ORACLE
    if _VALIDITY_ORACLE is None:
        from .assembly.retained_substitution import OpsinOracle
        from .validation.opsin_roundtrip import _find_opsin_jar
        _VALIDITY_ORACLE = OpsinOracle(opsin_jar=_find_opsin_jar())
    return _VALIDITY_ORACLE.parse_status(name)


def _validity_gate_name_to_smiles(name: str) -> Optional[str]:
    """OPSIN canonical SMILES for ``name`` via the shared singleton oracle (cached),
    or None if OPSIN rejected it / could not be consulted. Used by the SELF-01
    constitutional self-consistency gate to re-perceive the emitted name."""
    global _VALIDITY_ORACLE
    if _VALIDITY_ORACLE is None:
        from .assembly.retained_substitution import OpsinOracle
        from .validation.opsin_roundtrip import _find_opsin_jar
        _VALIDITY_ORACLE = OpsinOracle(opsin_jar=_find_opsin_jar())
    return _VALIDITY_ORACLE.name_to_smiles(name)


def _self_consistency_skeleton(smiles: str) -> Optional[str]:
    """Constitutional skeleton key = the first (skeleton) block of the standard
    InChIKey. It encodes formula + connectivity + mobile-H-normalized H layer, and
    EXCLUDES stereochemistry (the /t,/b layers live in the second block). So it is
    stereo-insensitive and tautomer-tolerant by construction (ADR-18-07). Returns
    None if RDKit cannot parse/hash the SMILES (-> the caller fails OPEN)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        from rdkit.Chem import inchi
        ik = inchi.MolToInchiKey(mol)
        return ik.split("-")[0] if ik else None
    except Exception:
        return None


def _self_consistency_full_key(smiles: str) -> Optional[str]:
    """The FULL standard InChIKey (skeleton block + stereo block + mobile-H /
    charge-normalized proton layer) of ``smiles`` — None if RDKit cannot parse it.

    An EQUAL full key between the input structure and a name's OPSIN re-perception
    is the strongest possible identity signal: it is exactly the project's headline
    round-trip metric, so two structures with one full InChIKey ARE the same
    molecule (same constitution, charge and stereo, protonation normalized). SELF-01
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
        return inchi.MolToInchiKey(mol) or None
    except Exception:
        return None


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


# v30 C6: RegistrationHash-backed stereo layer for SELF-01. The InChIKey skeleton
# block used by _self_consistency_skeleton EXCLUDES stereo by construction
# (ADR-18-07), so a name that encodes the WRONG stereoisomer (right constitution)
# passes the skeleton compare. RegistrationHash.GetMolLayers gives a stereo-bearing
# canonical layer that distinguishes enantiomers/diastereomers. Prep is normalize-only
# (no Cleanup, no tautomer/charge standardization) per spec §4; NO stereo-assignment
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
                              ignore_stereo: bool = False) -> str:
    """``"ok"`` | ``"mismatch"`` | ``"inconclusive"`` — does the OPSIN re-perception of
    the emitted name encode the SAME molecule as the input structure?

    Constitution = InChIKey skeleton block (formula + connectivity + mobile-H;
    stereo- and charge-insensitive) PLUS net formal charge. On the stereo-strict
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
    ka = _self_consistency_full_key(input_smiles)
    kb = _self_consistency_full_key(opsin_smiles)
    if ka is not None and kb is not None and ka == kb:
        return "ok"
    a = _self_consistency_skeleton(input_smiles)
    b = _self_consistency_skeleton(opsin_smiles)
    if a is None or b is None:
        return "inconclusive"
    if a != b:
        return "mismatch"
    # Skeleton matches — guard against a name that fails to preserve a CHARGED
    # input's net charge (the skeleton block excludes the charge layer).
    ca = _self_consistency_net_charge(input_smiles)
    cb = _self_consistency_net_charge(opsin_smiles)
    # Only a genuinely CHARGED input whose charge the name drops/changes is a leak
    # (e.g. [O-]O, -1, named 'dioxidane', 0). A NEUTRAL input is exempt: acid/ester
    # names are protonation-ambiguous in OPSIN (e.g. 'methyl phosphate' round-trips
    # to the -2 phosphate dianion) — that is not a wrong-molecule error.
    if ca is not None and cb is not None and ca != 0 and ca != cb:
        return "mismatch"
    if ignore_stereo:
        return "ok"
    # v30 C6: constitution + charge match. Catch a name whose OPSIN re-perception
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
    # v33 Phase 1 (C3): a name that OVER-specifies stereo (nb > na) FABRICATES
    # configuration the input does not define -- P-92 "CIP Priority and Sequence
    # Rules" / P-93 "Configuration Specification" (BlueBookV2.md:44525/:44527):
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
    # v27 Phase 0: a name that specifies FEWER stereo features than the input OMITS
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
    # wrong geometric isomer. Probes: *.py; audit and
    # derivation:  (RISK 3 / C6).
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
    """v29 P7 T1: publish what the validity gate DID for ``name``.

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
    """SELF-01: OPSIN already PARSED ``name``; verify it parses to the SAME molecule.

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
    verdict = _self_consistency_verdict(smiles, opsin_smiles, ignore_stereo=ignore_stereo)
    if verdict != "mismatch":
        # v29 P7 T1: "ok" is the ONE state that earns SELF-01. "inconclusive"
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


def _final_opsin_validity_gate(name: str, smiles: Optional[str],
                               stats: Optional[Dict[str, int]] = None,
                               *, besteffort_unverified: bool = False) -> str:
    """SUB-03 real-OPSIN validity gate (D-11/D-12/D-13).

    Suppress an OPSIN-unparseable production name to the EXISTING
    _descriptive_fallback STRING (never None / never a shipped invalid string).
    Runs AFTER stereo + grammar repair, as the last transform before the value
    leaves name()/name_with_confidence(); inside the is_top_level_naming guard
    so it never fires on decomposition fragments.

    ``besteffort_unverified`` (task-JAR-ABSENT, FABLE 0-wrong hole): the four
    name()/name_with_confidence() call sites pass ``self._general_fallback_unverified``.
    When True AND the OPSIN jar is GENUINELY absent (a no-Java deployment), the
    D-13 jar-absent fail-OPEN below is replaced by a fail-CLOSED suppression:
    at best-effort tier we cannot constitutionally verify ANY name without OPSIN
    (the Wave-0 reconstructor has no NameFacts extractor for an arbitrary emitted
    name yet -- ``verify_or_none(name, smiles, name_facts=None)`` is provably None
    on every jar-absent call), and FABLE proved this branch ships WRONG-molecule
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
    # D-13 fail-OPEN: probe the JAR FIRST.
    if not _validity_gate_jar_present():
        # task-JAR-ABSENT (FABLE 0-wrong hole): a GENUINELY absent jar means no
        # constitutional verification is possible for this name -- no OPSIN, and
        # the Wave-0 reconstructor needs a NameFacts extractor that does not yet
        # exist for an arbitrary emitted name (verify_or_none(name_facts=None) is
        # provably None on every jar-absent call). At BEST-EFFORT tier
        # (general_fallback_unverified) that unverified branch is exactly the
        # class FABLE showed ships WRONG-molecule names in a no-Java deployment
        # (COS(=O)(=O)O -> methane; ClP(Cl)(=O)OC1=CC=CC=C1 ->
        # (phosphonooxy)benzene, Cl2 silently swapped for (OH)2). 0-wrong is
        # ABSOLUTE, so fail CLOSED to the honest fallback rather than ship it.
        # The default/PIN tier (besteffort_unverified=False) keeps the historical
        # D-13 fail-OPEN so this cannot touch the 1652 gate's correct-by-
        # construction OPSIN-unparseable carve-out classes (inositol,
        # np_stereoparent, thioperoxol, dianhydride, phane, ...) -- a separate
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
    # CR-01 (code review 2026-06-02): suppress ONLY on a DEFINITIVE OPSIN
    # rejection. 'parsed' ships as-is; 'unavailable' (subprocess timeout / OSError
    # mid-run on a loaded host) fails OPEN — a transient OPSIN failure must never
    # turn a valid, round-trip-passing name into a descriptive fallback. The old
    # `parse-or-None` check conflated 'rejected' with 'unavailable', breaking the
    # phase's "0-regression by construction" guarantee.
    #
    # SELF-01 (v23 Phase 0): re-perceive the name via OPSIN. A non-None canonical
    # SMILES means OPSIN PARSED it -> run the constitutional self-consistency check
    # (suppress if the name encodes a DIFFERENT molecule). None means rejected or
    # unavailable; only then consult parse_status to distinguish them (fail-OPEN on
    # unavailable). Using name_to_smiles as the primary probe also avoids a second
    # OPSIN subprocess on the common parsed path.
    opsin_smiles = _validity_gate_name_to_smiles(name)
    if opsin_smiles is not None:
        # W3-P10: correct-by-construction polyacid prefix-derivative PIN that OPSIN
        # RE-PARSES to the wrong constitution (locant-assignment bug). Ship it —
        # the BB is the sole PIN authority and OPSIN's bug must not gate us.
        if _POLYACID_PREFIX_DERIVATIVE_PIN_RE.match(name):
            _record_gate_outcome(
                _pv.carveout_outcome("polyacid_prefix_derivative"), name)
            return name
        # _self_consistency_decision records its own outcome (one per verdict).
        return _self_consistency_decision(name, smiles, opsin_smiles, stats)
    # opsin_smiles is None -> either OPSIN was UNAVAILABLE (transient: no jar /
    # timeout / OSError), OR OPSIN PARSED the name but emitted a SMILES that RDKit
    # cannot canonicalise (an impossible valence — e.g. the lambda6 candidate
    # '1,2lambda6,3-dioxathiolane' produced for a lambda6-multiring-spiro that the
    # spiro engine fail-closes). ONLY 'unavailable' fails OPEN (a transient hiccup
    # must never suppress a valid, round-trip-passing name — CR-01). A
    # 'parsed'-but-uncanonicalisable output is UNVERIFIABLE: we cannot confirm the
    # name describes the input structure, so it must fall through to the whitelist
    # carve-outs below and, absent a whitelist hit, suppress to the honest fallback
    # (fail-closed, accuracy #1). Conflating the two here shipped the wrong name.
    if _validity_gate_status(name) == "unavailable":
        _record_gate_outcome(_pv.GATE_OUTCOME_UNAVAILABLE, name)
        return name  # transient -> fail-OPEN (never suppress)
    # DD2 / OPSIN-validity grammar carve-out (Phase D): OPSIN's generation grammar does not recognise the
    # P-63.4.2 chalcogen-peroxol suffix family ('-SO-thioperoxol', '-OS-thioperoxol',
    # '-dithioperoxol'), so it REJECTS these correct PINs (P-56.2 verbatim:
    # `CH3-S-OH -> methane-SO-thioperoxol (PIN)`). Like the stereo-grammar carve-out
    # below, an OPSIN coverage gap on a valid IUPAC suffix must not gate Orthonym
    # correctness — gold is name-exact-match, not RT. CR-03 guard: only un-suppress a
    # WELL-FORMED thioperoxol name. The italic 'SO-'/'OS-' descriptor must be
    # separator-led (hyphen/paren); a multiplied form glued straight onto the
    # descriptor ('ethane-1,2-bisSO-thioperoxol') is a real formatting defect and
    # stays suppressed to the honest fallback rather than shipping malformed.
    if "thioperoxol" in name and not re.search(r"[A-Za-z](SO|OS)-thioperoxol", name):
        _record_gate_outcome(_pv.carveout_outcome("thioperoxol"), name)
        return name
    # v23 Phase 12 follow-on (P-104.2.1): the inositol retained names
    # (myo-/scyllo-/cis-/epi-/neo-/allo-/muco-/D-chiro-/L-chiro-inositol) are the
    # PIN but are OPSIN-UNPARSEABLE (no generation-grammar support — verified
    # name_to_smiles -> None for every one), exactly the thioperoxol situation
    # above. They are produced ONLY by the hard-gated InChIKey recogniser
    # (rules.inositols.name_inositol) — correct by construction — and validated
    # name-exact, so OPSIN's coverage gap must not suppress them.
    try:
        from .rules.inositols import INOSITOL_NAMES
        if name in INOSITOL_NAMES:
            _record_gate_outcome(_pv.carveout_outcome("inositol"), name)
            return name
    except Exception:
        pass
    # v23 Phase 14 (P-101.2.7 Table 10.1 a/c): the terpene/alkaloid stereoparent
    # retained names (abietane/kaurane/.../yohimban/sparteine/...) are the recommended
    # semisystematic parent names but are OPSIN-UNPARSEABLE (verified name_to_smiles ->
    # None for each), exactly the inositol/thioperoxol situation.  They are produced ONLY
    # by the exact-canonical-SMILES NATURAL_PRODUCT_DERIVATIVES lookup (correct by
    # construction; structures ChEBI/Wikidata cross-verified), validated name-exact, so
    # OPSIN's coverage gap must not suppress them.
    try:
        from .data.natural_products import NAME_EXACT_NP_PARENTS
        if name in NAME_EXACT_NP_PARENTS:
            _record_gate_outcome(_pv.carveout_outcome("np_stereoparent"), name)
            return name
    except Exception:
        pass
    # W3-P06 (P-65.7.6.1): di-/polyanhydride PINs ('diacetic butanedioic
    # dianhydride') are correct-by-construction (emitted ONLY by the hard-gated
    # rules.anhydrides._name_dianhydride) but OPSIN's anhydride word-rule cannot
    # parse the 'di-'-COLLAPSED acid words ("Unexpected number of words in
    # anhydride" — it accepts only the un-collapsed 'acetic butanedioic
    # dianhydride', which round-trips to the same structure). Exactly the
    # thioperoxol/inositol OPSIN generation-grammar gap. The regex guard keeps the
    # carve-out tight: >=2 acid-like words followed by a multiplied '...anhydride'.
    if _DIANHYDRIDE_PIN_RE.match(name):
        _record_gate_outcome(_pv.carveout_outcome("dianhydride"), name)
        return name
    # W8-P4 (P-65.7.6.4.1): chalcogen di-/poly-anhydride PIN carve-out — see
    # _CHALCOGEN_DIANHYDRIDE_PIN_RE docstring above.
    if _CHALCOGEN_DIANHYDRIDE_PIN_RE.match(name):
        _record_gate_outcome(
            _pv.carveout_outcome("chalcogen_dianhydride"), name)
        return name
    # W3-P07 (P-65.6.3.3.3.2 method (1)): functional-class polyol polyester PIN
    # ('propane-1,2,3-triyl 1,3-diacetate 2-propanoate') — correct-by-construction
    # from _assemble_glyceride but OPSIN cannot parse the multi-anion syntax
    # (exactly the inositol/dianhydride generation-grammar gap).
    if _POLYOL_POLYESTER_PIN_RE.match(name):
        _record_gate_outcome(_pv.carveout_outcome("polyol_polyester"), name)
        return name
    # W8-P9 Task 9.1 (P-69.2.3/.4/.6): additive/coordination organometallic
    # PIN carve-out — see _ORGANOMETALLIC_ADDITIVE_PIN_RE docstring above.
    if _ORGANOMETALLIC_ADDITIVE_PIN_RE.match(name):
        _record_gate_outcome(
            _pv.carveout_outcome("organometallic_additive"), name)
        return name
    # W8-P8 Task 8.12 (P-26.2/.3): phane simplified-skeletal PIN carve-out —
    # see _PHANE_PIN_RE docstring above. Emitted ONLY by the hard-gated
    # rules.phane.build_phane_pin (formula-conservation-vetoed), so OPSIN's
    # total lack of phane grammar must not suppress it.
    if _PHANE_PIN_RE.match(name):
        _record_gate_outcome(_pv.carveout_outcome("phane"), name)
        return name
    # W8-P5 Task 2 (P-72.3/P-72.8.1): substituted halogen-uide PIN carve-out
    # ('diphenyliodanuide') — see _HALOGEN_UIDE_PIN_RE docstring above.
    # Emitted ONLY by the hard-gated rules.ions._emit_group13_uide halogen
    # branch (atom-conservation-vetoed), so OPSIN's substituted-uide grammar
    # limitation must not suppress it.
    if _HALOGEN_UIDE_PIN_RE.match(name):
        _record_gate_outcome(_pv.carveout_outcome("halogen_uide"), name)
        return name
    # OPSIN-validity stereo carve-out / DEF-9 (Phase 169.7): decide on WHERE OPSIN fails. If the name is
    # rejected ONLY because of its stereo layer — i.e. the stereo-STRIPPED
    # constitutional form parses — then OPSIN's narrower generation-side stereo
    # grammar must NOT gate Orthonym correctness (audit Dim-08 §C; the verbatim
    # Blue Book PIN `(1s,4s)-cyclohexane-1,4-diol` was being suppressed to
    # `unknown`). The constitutional gate stays STRICT: a name whose stereo-stripped
    # form ALSO fails to parse is still suppressed. strip_stereo is read-only — the
    # shipped name keeps its stereo descriptors.
    #
    # ARCH-2 (2026-07-27): this branch used to `return name` outright, skipping
    # SELF-01, so ANY constitutional defect rode out free as long as the name
    # happened to carry a stereo prefix OPSIN rejects — verification was strongest
    # on well-formed names and ABSENT on malformed ones, exactly backwards. Measured
    # on  with the gates ON: 107 names shipped through
    # here and 98 of them named a DIFFERENT molecule (witness:
    # COC(=O)NCC[C@@H]1CC[C@H]2[C@@H]1C2(Br)Br shipped as `(1S,4S,5R)-methyl
    # N-octylcarbamate` — ring + both Br silently dropped); 0 of the 1641 gold
    # targets were affected.
    #
    # The carve-out's licence is narrow — OPSIN need not parse the stereo LAYER —
    # and it was never a licence to ship an UNVERIFIED CONSTITUTION. The comparison
    # was available all along: _self_consistency_verdict compares the InChIKey
    # SKELETON block, which EXCLUDES stereochemistry by construction (ADR-18-07),
    # so it can judge the stereo-STRIPPED parse with no loss of validity. Route it
    # through the SAME SELF-01 decision the parsed path uses: a verified
    # constitutional mismatch suppresses, while `ok`/`inconclusive` still ship the
    # FULL stereo name (fail-OPEN on any comparison that could not be made).
    from .rules.stereochemistry import strip_stereo
    _stripped = strip_stereo(name)
    if _stripped != name:
        _stripped_status = _validity_gate_status(_stripped)
        # ARCH-2-FOLLOWUP: _validity_gate_status is THREE-valued, and the stripped
        # probe is a second, independent OPSIN consultation that can hiccup on its
        # own. 'unavailable' (subprocess timeout / OSError on a loaded host) means
        # OPSIN was never consulted — it is NOT evidence against the name — so it
        # must fail OPEN, exactly as the primary probe already does at CR-01 above.
        # Testing `== "parsed"` alone lumped it with 'rejected' and let a transient
        # hiccup suppress the very names this carve-out exists to rescue (the
        # verbatim BB PIN `(1s,4s)-cyclohexane-1,4-diol` -> `unknown`), contradicting
        # the promise made three paragraphs up. Note this is NOT a blanket escape
        # hatch: a PERSISTENTLY unavailable OPSIN (JAR genuinely missing) is already
        # caught by the _validity_gate_jar_present() probe above, which fails the
        # whole gate open — so the two behaviours agree rather than compete.
        if _stripped_status == "unavailable":
            if stats is not None:
                stats["gate_stereo_unavailable"] = (
                    stats.get("gate_stereo_unavailable", 0) + 1)
            _record_gate_outcome(_pv.GATE_OUTCOME_UNAVAILABLE, name)
            return name  # transient -> fail-OPEN, with the stereo layer intact
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
                    return _decided  # SELF-01 PROVED a different molecule -> suppress
                if stats is not None:
                    stats["gate_stereo_kept"] = stats.get("gate_stereo_kept", 0) + 1
                # v29 P7 T1: SELF-01 judged the stereo-STRIPPED parse, so an
                # "ok" here verifies the CONSTITUTION ONLY — the stereo layer
                # was never checked (that is the whole point of this
                # carve-out). Downgrade the inner verdict to say so. Any other
                # inner outcome (inconclusive / skipped / warn-mismatch) is
                # already non-verifying and is left exactly as recorded.
                if (_pv.get_provenance()["gate_outcome"]
                        == _pv.GATE_OUTCOME_SELF01):
                    _record_gate_outcome(
                        _pv.GATE_OUTCOME_SELF01_CONSTITUTION_ONLY, name)
                return name  # only the stereo layer is OPSIN-narrow -> ship it whole
            # 'parsed' but NO usable SMILES: OPSIN accepted the stripped name and
            # then emitted a structure RDKit cannot canonicalise (an impossible
            # valence) — the identical situation the non-stereo path already settled
            # above, where only 'unavailable' fails OPEN and
            # 'parsed'-but-uncanonicalisable is UNVERIFIABLE and suppressed. It is
            # not merely an unmade measurement: it is positive evidence that the name
            # denotes a structure RDKit rejects as impossible. A stereo layer must
            # not lower the burden of proof, so fall through to the honest fallback
            # exactly as that path does. Measured: 1 corpus row, 0 gold rows, and
            # that row's name is independently wrong.
            if stats is not None:
                stats["gate_stereo_unverifiable"] = (
                    stats.get("gate_stereo_unverifiable", 0) + 1)
    # NOTE (D-06, resolved): radical names now ship normally. The validity gate's
    # primary probe runs the OpsinOracle WITH `-r` (retained_substitution.py), so
    # a well-formed radical name parses -> SELF-01 constitutional compare -> ships.
    # The P-71 non-terminal-locant fix landed long ago (single sites:
    # C[CH]C -> `propan-2-yl` via emit_parent_hydride_cumulative_suffix); multi-site
    # free valences (P-71.2.3: ethane-1,2-diyl / propane-1,2,3-triyl /
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


def _full_inchikey_offer_match(input_smiles: str, opsin_smiles: str) -> bool:
    """v33 Phase 0 L3-1: True iff ``input_smiles`` and ``opsin_smiles`` share
    the SAME FULL (all-layer) InChIKey -- constitution AND stereo AND
    protonation/mobile-H-tautomer state all identical.

    Strictly stronger than `_self_consistency_skeleton`'s stereo-/charge-
    insensitive FIRST block (which is what the ordinary SELF-01 gate proves):
    the L3 CHARACTERIZATION found the discard-gap primaries are constitution-
    CORRECT (they pass that tolerant compare -- including its deliberate
    ``na == 0`` licence, "a stereo-UNSPECIFIED input named by a stereo-
    implying retained name ... tolerated here") but full-InChIKey-FAIL (33/34
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
        return ik_in == ik_out
    except Exception:
        return True


def _offer_rt_ok(name: str, input_smiles: Optional[str]) -> bool:
    """v33 Phase 0 L4-core/L3-1: the RT/PIN-gate PASS/FAIL predicate for ONE
    candidate ``name`` offered as a whole-molecule winner (the ``rt_ok``
    ``offer_pool.select_rt_passing`` is injected with, built at `_finish`).

    v33 Phase 0 L3-1 CHANGE: this now requires a FULL-InChIKey match
    (`_full_inchikey_offer_match`), not merely the constitution-only SELF-01
    verdict the L4-core version reused outright. The L3 CHARACTERIZATION
    measured that the constitution-only bar lets a stereo-wrong (or stereo-
    fabricated) primary through, which is exactly the gap the systematic
    floor exists to close -- so a primary that only proves constitution must
    NOT out-rank a floor offer that proves the full molecule.

    NO double-OPSIN in the common case: `_validity_gate_name_to_smiles` is
    MEMOIZED (`OpsinOracle._name_cache`), and the ordinary SELF-01 gate
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

    v33 no-abstain Phase A fix-round-1 (Findings 1+2, FABLE adversarial
    review): `skip_reanchor` bundles FOUR distinct outcomes as though they all
    meant "no check is possible or worth attempting" -- but `unavailable` /
    `not_run` mean only "no gate call was recorded for THIS EXACT STRING
    (yet)", which is a LIVE, checkable question right now whenever a jar is
    present -- unlike a genuine `carveout:*` (a by-design OPSIN-unparseable
    PIN class -- thioperoxol/inositol/...) or `gate_disabled` (the whole
    verification layer is deliberately off). Fail-opening all four alike let a
    stubbed T4 producer ship a WRONG-MOLECULE name via a `t4_floor` offer with
    ZERO gates firing end-to-end:  stubbed
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
        # formatted `f"{resolved}: ..."`) rather than re-deriving the outcome
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
    # constitution-only SELF-01 verdict said.
    opsin_smiles = _validity_gate_name_to_smiles(name)
    if opsin_smiles is None:
        # OPSIN could not re-perceive this string. Fail OPEN on a transient
        # 'unavailable' ONLY when this EXACT string ALREADY earned a real
        # positive SELF-01 verdict (`self01_complete is True`) -- re-looking up a
        # previously-VERIFIED name and hitting a transient blip must not lose it.
        #
        # WS7 fix round 1 (coordinator CRITICAL, 0-wrong): for a string with NO
        # positive verdict (`self01_complete is None` -- `bypassed` / `suppressed`
        # / `inconclusive`, e.g. a FRESH floor/alternative offer whose exact
        # string the gate NEVER verified), a None here means the offer is
        # UNVERIFIED, so it must fail CLOSED. `bypassed` reaches THIS branch
        # (not the deny-by-default `verify_or_none` branch above, which
        # `_self01_lookup` only routes `unavailable`/`not_run` into), so without
        # this a transient OPSIN failure shipped an ungated floor offer
        # (`F[B-](F)(F)F` -> the unverified floor `2,2-difluoro-2-borapropan-2-uide`;
        # `[Se-]CC` -> the OPSIN-unparseable `1-selanidoethane`). This makes the
        # offers lane symmetric with the recovery-lane T4 (`_cand_from_t4`), which
        # already fails CLOSED on the same blip. A definitive 'rejected' fails
        # closed for every string regardless.
        return (self01_complete is True
                and _validity_gate_status(name) == "unavailable")
    if not input_smiles:
        return True  # nothing to compare against -- inconclusive, fail-OPEN
    return _full_inchikey_offer_match(input_smiles, opsin_smiles)


# ---------------------------------------------------------------------------
# Compound class pre-routing (Phase 141, CLASS-06)
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
        _PYRANOSE_SMARTS = Chem.MolFromSmarts("[OX2;r6]1[CX4][CX4][CX4][CX4][CX4]1")
        # 5-membered ring with 1 O and 4 C (furanose)
        _FURANOSE_SMARTS = Chem.MolFromSmarts("[OX2;r5]1[CX4][CX4][CX4][CX4]1")
        # OH group on a ring carbon
        _RING_OH_SMARTS = Chem.MolFromSmarts("[C;r]([OX2H])")
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
    Classification order per D-01:
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
    senior_ring_system: Optional[tuple] = None  # P-44.2 most senior ring system (atom indices)
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
    n_substituents: List[Dict] = field(default_factory=list)  # N-substituents from get_n_substituents()

    # Ring-as-substituent information (when chain is parent per IUPAC P-44.1)
    ring_substituents_as_groups: List[tuple] = field(default_factory=list)  # Rings that become substituents
    chain_is_parent: bool = False  # True when parent selection chose chain over ring

    # Ion/radical species information
    species_type: str = 'neutral'  # 'neutral', 'ion', 'zwitterion', 'salt', 'radical'
    ion_sites: Dict[str, List[Dict]] = field(default_factory=dict)  # From get_ion_sites()
    radical_sites: List[Dict] = field(default_factory=list)  # From get_radical_sites()
    total_charge: int = 0  # Net formal charge of the molecule

    # Phase 148 D-09 / V18 Appendix A.5: parent-selection result for downstream
    # coverage_scoring readers. Populated by _classify when select_parent()
    # runs for a cyclic+chain molecule (chain_len >= 2). Phase 149 / IM-11
    # will recalibrate FACTOR_WEIGHTS_V18['parent_correctness'] against the
    # discrimination this slot provides.
    # Source: V18_MILESTONE_PLAN Appendix A.5.
    parent_selection_result: Optional[Any] = None


def compute_features(mol, smiles: Optional[str] = None) -> MolecularFeatures:
    """Phase 147 BL-1: thin module-level wrapper around Orthonym()._perceive.

    Provides a public-API perception entry for tests and downstream
    callers. Mirrors what name_compound() does internally before naming.

    Args:
        mol: RDKit Mol object.
        smiles: Optional input SMILES; if None, derived via Chem.MolToSmiles(mol).

    Returns:
        MolecularFeatures populated by Orthonym()._perceive.

    Source: Phase 147 Plan 02 BL-1 fix (public-API perception entry).
    """
    if smiles is None:
        smiles = Chem.MolToSmiles(mol)
    canonical_smiles = Chem.CanonSmiles(smiles)
    return Orthonym()._perceive(mol, smiles, canonical_smiles)


def _collect_ring_substituent_positions(features, ring_atoms):
    """Set of ring atom indices that bear an off-ring (non-H) substituent.

    Used by Phase 147 dispatch helper branch 4 (simple heterocycle) to feed
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
    """P-59.2.1.6 (BB 25207): the principal group appears in BOTH the ring and
    a chain; the portion with the GREATER number of the group is the parent
    (tie -> ring, P-52.2.8). When the ring wins, an off-ring principal-group
    match must NOT be expressed as a second ring suffix — it stays on its
    demoted chain and is named as a substituent prefix (e.g. the pendant
    butan-2-one on a cyclopentane-1,2-dione -> '4-(2-oxobutyl)').

    This filters ``features.principal_group_atoms`` in place to the ring-anchored
    matches only, WHEN:
      * the ring is the chosen parent (this helper is called only then);
      * the principal group is a NON-terminal, skeletal-carbon carbonyl-type
        group (ketone family) whose exocyclic form can be expressed as an
        oxo-substituent prefix — terminal/appended groups (-carbaldehyde,
        -carboxylic acid, ...) legitimately anchor exocyclically and are
        untouched (they route through the ring_anchored_pg_atoms path);
      * the ring's on-ring match count is STRICTLY GREATER than the off-ring
        count (ring wins outright) OR they TIE (P-52.2.8 ring default) — but
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
    # P-59.2.1.6: ring is parent only when it has the GREATER count, or ties
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
    """Phase 147 D-03: dispatch on ring type and produce authoritative IUPAC locants.

    7-branch cascade (order per 147-CONTEXT.md D-03 with W-1 fix):
      1. Fused-heterocycle (match_fused_heterocycle_core) — preserve D-09
         byte-identical path; runs first.
      2. Polycyclic aromatic (identify_polycyclic + get_polycyclic_iupac_locants)
         — runs for ANY ring system with a PAH match, NOT gated on fused_type
         (W-1 fix: pyrene is ortho-peri-fused but must still flow here).
      3. Benzene-only single ring (orient_benzene with canonical
         get_benzene_substituents helper — BL-2 fix).
      4. Simple heterocycle single ring (orient_heterocycle_with_substituents).
      5. Spiro detection -> {"iupac_locants": None} stub (Phase 151 fills).
      6. Bridged/VB detection -> {"iupac_locants": None} stub (Phase 151 fills).
      7. Else -> None (carbocyclic monocycle / hydrocarbon polycycle:
         sorted fallback in _build_ring_pos preserves back-compat per D-09).

    Returns None for acyclic molecules.

    Source: https://iupac.qmul.ac.uk/BlueBook/P4.html P-44.1
    Source: Phase 147 CONTEXT D-03, D-04, D-06, D-09; Plan 02 W-1, W-2, BL-2.
    """
    if not features.is_cyclic:
        return None

    # Match namer.py top-level relative-import style (W-2 fix).
    from .rules.fused_rings import classify_fused_system
    from .data.fused_heterocycles import match_fused_heterocycle_core
    from .rules.polycyclics import (
        identify_polycyclic,
        get_polycyclic_iupac_locants,
    )
    from .rules.benzene import (
        orient_benzene, get_benzene_substituents, principal_group_ring_atoms,
        molecule_principal_group_and_fgs,
    )
    from .rules.heterocycles import orient_heterocycle_with_substituents
    from .perception.rings import get_spiro_atoms
    from .rules.bridged_fused import is_bridged_fused

    mol = features.mol

    # Phase 151-02 D-09: skip Branch 1 (fused-heterocycle catalog) when
    # the input is mixed-spiro/fused — Branch 5b owns that dispatch.
    # Without this guard, the catalog returns a PARTIAL locant map
    # (only the fused component's atoms; missing the spiro side ring),
    # which violates the cascade-step-6 coverage invariant downstream.
    from .rules.spiro import is_mixed_spiro_fused as _phase151_is_mixed_spiro_fused
    _is_mixed_spiro_fused_input = _phase151_is_mixed_spiro_fused(mol)

    # Branch 1: fused-heterocycle (preserve D-09 byte-identical path for
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
        # BL-2 fix: use the canonical get_benzene_substituents helper
        # (returns Dict[int, List[Dict]] with real substituent entries),
        # NOT a {idx: []} placeholder — the latter would yield arbitrary
        # orientation because orient_benzene checks ``if atom_idx in
        # substituents:`` and every dict key matches an empty-list value.
        if (len(ring_atoms) == 6
                and all(a.GetIsAromatic() and a.GetSymbol() == 'C'
                        for a in ring_mol_atoms)):
            try:
                substituents = get_benzene_substituents(mol, ring_atoms)
                # E1/DD4 (P-14.4(c)): anchor the principal characteristic group
                # to the lowest locant before detachable substituents.
                # v29 Phase C Task 9: this was derived from the is_suffix marker
                # alone, which made it EMPTY for phenols (a ring -OH is spelled as
                # the 'hydroxy' PREFIX here and promoted to '-ol' downstream) --
                # so this locant HINT disagreed with the emitted NAME. It now uses
                # the same shared authority as the composer, so the two agree.
                #
                # ⚠ v29 Phase C Task 9b: reading ``features.principal_group`` alone
                # made the whole anchor a NO-OP here. ``compute_features()`` runs
                # ``_perceive`` only, so that attribute is None for all four external
                # callers (rules/ring_chalcogen_oxide.py:277,
                # rules/multiplicative.py:1795 and :2454, rules/ions.py:2240) --
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

    # Branch 5 (Phase 151-02 D-21): pure spiro + mixed spiro/fused
    # cascade-step-6 suppliers.
    #
    # Pure spiro: is_spiro_system True iff n_rings == n_spiro + 1 (D-09 lock).
    # Wraps existing get_spiro_numbering / _get_polyspiro_numbering with
    # the cascade-step-6 coverage invariant (Pitfall 7).
    #
    # Mixed spiro/fused: is_mixed_spiro_fused True iff ≥1 spiro atom AND
    # n_rings > n_spiro + 1 AND detect_natural_product is None (Pitfall 3
    # false-positive guard) AND the topology is HERITAGE §4 separable AND
    # the fused part has a catalog name (D-22(b) canary-stability guard).
    #
    # If neither supplier returns full coverage (None), fall through to
    # Branch 6 (Phase 151-01 VB) and onwards.
    #
    # Source: 151-CONTEXT.md D-09 / D-13 / D-21; 151-AUDIT-B.md verdict
    # MIXED_SPIRO_FUSED_MISSING + Q-05 NESTED_FORM_PARSEABLE.
    from .rules.spiro import (
        is_spiro_system,
        is_mixed_spiro_fused,
        get_spiro_iupac_locants,
        get_mixed_spiro_fused_iupac_locants,
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

    # Branch 6 (Phase 151-01 D-21): Von Baeyer ≥4-ring authoritative locants.
    # Routes tetracyclic / pentacyclic / higher non-cataloged bridged systems
    # to the new polycyclic_von_baeyer module. Anti-canary lock D-04 inside
    # is_higher_polycyclo guarantees bicyclo / tricyclo / aromatic / steroid
    # / mixed cases skip this branch and fall through to is_bridged_fused
    # below or downstream branches.
    #
    # Source: 151-CONTEXT.md D-04 / D-06 / D-21; 151-AUDIT-A.md verdict
    # THIN_WRAPPER; Phase 147 cascade-step-6 gate (candidate_pool.py:634).
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
    # bridged systems caught by is_bridged_fused). Phase 151-02 / 03 wire
    # spiro and ring-assembly suppliers here.
    if is_bridged_fused(mol):
        return {"iupac_locants": None}

    # Branch 7 (Phase 151-03 D-21): ring assembly size 3+ supplier.
    # Wires get_ring_assembly_iupac_locants per 151-AUDIT-C.md verdict
    # SUPPLIER_MISSING + 151-PATTERNS.md Pattern S-3. Path-topology check
    # added to detect_ring_assembly per D-15 rejects branched arrangements
    # (1,3,5-triphenylbenzene) so Branch 7 only fires on linear chains.
    # 2-system bi- assemblies are NOT routed here (covered by existing
    # naming pipeline + the >=3 gate keeps blast-radius minimal).
    #
    # Source: 151-CONTEXT.md D-15 / D-18 / D-19 / D-21; 151-AUDIT-C.md
    # composite verdict; Pitfall 7 (full coverage or None).
    if hasattr(features, 'ring_systems') and len(features.ring_systems) >= 3:
        from .rules.ring_assemblies import (
            detect_ring_assembly as _phase151_detect_ring_assembly,
            get_ring_assembly_iupac_locants as _phase151_get_ra_locants,
        )
        info = _phase151_detect_ring_assembly(mol, features.ring_systems)
        if info is not None and info.get("count", 0) >= 3:
            ral = _phase151_get_ra_locants(mol)
            if ral is not None:
                return {"iupac_locants": ral}

    # Branch 6.5 (Phase 149 D-09): non-cataloged fused systems.
    # Cataloged compounds reach Branches 1 (fused-heterocycle catalog) and
    # 2 (PAH) first. If they didn't, but the system is ortho-fused or
    # ortho-peri-fused with EXACTLY 2 SSSR components, route base-component
    # selection through FR-2.3.
    #
    # CRITICAL: emits `base_component_atoms` (NEW key), NOT `iupac_locants`.
    # Cascade step 6 in candidate_pool._has_iupac_locants checks
    # specifically for `iupac_locants` — Branch 6.5's new key is invisible
    # to that gate, so cascade step 6 stays GATED for non-cataloged fused
    # systems (Phase 147 D-06 + Phase 149 SC-7 lock).
    #
    # SCOPE LIMIT (Phase 149 Plan 02 triage): restricted to 2-component
    # fused systems where FR-2.3 base selection is reliable. 3+ component
    # systems (steroids, complex polycycles) fall through to Branch 7 to
    # avoid propagating partial base-atoms that disrupt downstream parent
    # selection for systems whose IUPAC name requires the full ring system
    # as parent. 3+ component systematic-name assembly is deferred to
    # Phase 149.x or Phase 155 per D-08 trade-off.
    #
    # Source: https://iupac.qmul.ac.uk/fusedring/FR23.html
    # Source: 149-CONTEXT.md D-09; SC-2; SC-7; D-08 trade-off.
    # Source: 147-CONTEXT.md D-06 (cascade step 6 gate carry-forward).
    if fused_type in ('ortho-fused', 'ortho-peri-fused'):
        # Branch 1 (catalog) and Branch 2 (PAH) already missed (control
        # flow reached this point — Branch 1 returns earlier on
        # heterocycle match; Branch 2 returns earlier on pah_locants).
        from .rules.fused_ring_selection import (
            select_base_component,
            _enumerate_components,
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

#: v29 P1: the accepted values of the binding-proof flag.
BINDING_PROOF_MODES = ("off", "audit", "enforce")


def validate_binding_proof(value: str) -> str:
    """Single source of truth for the v29 P1 binding-proof mode check.

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


def _budget_scope(fn):
    """v33 giant-molecule hang fix: bracket a top-level ``name()`` with the
    per-top-level fragment work budget.

    ``enter_name_scope``/``exit_name_scope`` maintain a raw ``name()`` call-stack
    counter on the fragment thread-local; the TRUE outermost call arms the budget
    and every nested re-entry (the recursion through ``name_compound`` and the
    isolated T4 producer) shares it. Crucially, ``isolated_naming_session`` resets
    the session state but NOT this counter, so the budget survives the nested
    recovery entries a giant molecule triggers -- guaranteeing termination.

    A decorator (not inline enter/exit) so the budget is released on EVERY exit
    path -- ``name()`` has several early ``return`` sites that each call
    ``end_naming_session`` -- with no leak that could contaminate the next
    molecule.
    """
    @_functools.wraps(fn)
    def _wrapper(self, *args, **kwargs):
        from .assembly.fragment_naming import enter_name_scope, exit_name_scope
        enter_name_scope()
        try:
            return fn(self, *args, **kwargs)
        finally:
            exit_name_scope()
    return _wrapper


# v37 SP1.3: cross-instance recursion guard for the parent-offer retry. The
# retry spawns FRESH inner instances whose own best-effort ship-failure hooks
# (`_try_demote_senior_group_rescue`, decomposition, ...) can be top-level
# (`is_top_level_naming()` keys off the fragment `visited` set, not session
# depth), so without this a demote-spawned inner could re-enter the parent
# offer. One flag per thread; set for the whole duration of an outer offer.
_ALT_PARENT_RESCUE = threading.local()


class Orthonym:
    """
    IUPAC nomenclature generator.
    
    Implements the structure-to-name workflow:
    1. Perception: Extract molecular features using RDKit
    2. Classification: Apply IUPAC seniority rules
    3. Assembly: Build name from fragments
    
    Example:
        >>> namer = Orthonym()
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
        """
        Initialize namer.

        Args:
            style: Naming style
                - "pin": Preferred IUPAC Names (IUPAC 2013)
                - "general": General IUPAC (more flexible)
                - "cas": CAS-style naming
            _disable_grammar_validation: Phase 156 escape hatch (D-14).
                When True, the OPSIN grammar pre-validation layer is
                disabled (`self._grammar` is None). ON by default in
                production; OFF only for unit tests inspecting raw
                handler output.
            enable_triviality_controller: Phase 168 D-08 opt-in flag.
                Default False (Stage A SACRED byte-identical canary
                invariant). When True, the triviality controller
                (assembly/retained_substitution.py) swaps systematic
                PIN-eligible parents to retained PIN forms at the
                name-tree IR layer per IUPAC P-15.1.8.1..3, and an
                OpsinOracle is instantiated for the TRIV-03 T2 RT-safety
                gate. Env override:
                ORTHONYM_ENABLE_TRIVIALITY_CONTROLLER=1/true/yes/on.
            trivial_fallback: PIN-policy fallback flag (CLI ``--trivial``).
                Default False (PIN fails closed). When True, a general-only
                (PIN-denied) retained name is returned ONLY when the default
                pipeline could not derive a PIN — a FALLBACK, never a
                downgrade of a derivable PIN. Per the PIN-policy contract
                (context resolution 3) this ALSO implies
                ``enable_triviality_controller=True`` (one user intent:
                "allow non-PIN trivial output").
            general_fallback: v25 G1 opt-in. When True, a GENERAL-class
                abstention is retried through the binding-carrying general
                engine (assembly/general_engine.py), gated by the E1
                certificate (validation/e1_certificate.py) and the existing
                downstream moat (>15-HA gate, P10 vetoes, SELF-01). Default
                False — default output byte-identical for existing callers.
            allow_aromatic_general: v26 P0 opt-in (plumbing only; inert
                until P1/P2 land). When True, threaded down into
                ``general_engine.name_general`` -> ``name_general_ring`` ->
                ``vonbaeyer_universal.analyze_cage_universal`` as
                ``allow_mancude``. Backs ``--emit-tier complete``. Default
                False — default output byte-identical for existing callers.
            binding_proof: v29 Phase 1 opt-in. One of:
                - "off" (default): no proof work at all. The only cost on the
                  default path is one string comparison, so PIN output is
                  byte-identical BY CONSTRUCTION, not by careful matching.
                - "audit": record the general engine's binding spine and
                  re-assert it (validation/proof_ledger.py) against the string
                  ``name()`` actually returns. OBSERVE-ONLY — the name is never
                  altered, findings go to ``logger.info`` and the ledger.
                - "enforce": additionally ABSTAIN (fail closed) when the
                  re-anchored proof fails. Implemented and unit-tested here but
                  set by NO default path in this phase; it exists for the later
                  phases that produce bindings for post-processed names.
                Anything else raises ValueError: a typo must not silently
                disable the proof.
        """
        # v29 P1: validate eagerly. A misspelled mode that silently degraded to
        # "off" would present as a clean run with the proof never executing --
        # the one failure mode an audit flag must not have.
        self._binding_proof: str = validate_binding_proof(binding_proof)
        self.style = style
        # Task 1.9 (PIN-policy): fallback-only opt-in. When True the name path
        # substitutes a general-only retained name for the "unknown organic
        # compound" failure signal; it never overrides a derived PIN.
        self._trivial_fallback: bool = trivial_fallback
        # v25 G1: opt-in general-engine inline fallback (decision 3:
        # PIN-strict default unchanged; lower tiers opt-in). When True, a
        # GENERAL-class abstention is retried through the binding-carrying
        # general engine (assembly/general_engine.py), gated by the E1
        # certificate and the existing downstream moat. Default False ->
        # default output byte-identical.
        self._general_fallback: bool = general_fallback
        # v25 G3: T4 opt-in — ship an E1-passed engine name when OPSIN cannot
        # verify it (absent jar / rejected / transient). A PARSED-but-
        # MISMATCHED name is never shipped at any tier.
        self._general_fallback_unverified: bool = general_fallback_unverified
        # v26 P0: plumbing-only opt-in (inert until P1/P2 consume it). When
        # True, threaded into general_engine.name_general -> name_general_ring
        # -> vonbaeyer_universal.analyze_cage_universal(allow_mancude=...).
        # Default False -> default output byte-identical.
        self._allow_aromatic_general: bool = allow_aromatic_general
        # v37 SP5.4: the full-coverage opt-in (``--emit-tier full-coverage``).
        # The SINGLE bit that arms the D2 general P-69 coordination-additive
        # namer, which sits strictly ABOVE best-effort (full-coverage is
        # best-effort's production superset PLUS this marker). Default False ->
        # D2 dispatch is never reached and EVERY tier's output is
        # byte-identical (the SP5.4 isolation property). Published as
        # ``full_coverage_ctx`` for the recursive re-entry, torn down in the
        # ``name()`` finally with its three siblings.
        self._full_coverage: bool = full_coverage
        # SUB-03 (169.5): per-instance bypass for the OPSIN validity gate, used
        # by neutralize-recurse / fragment intermediate naming (those produce an
        # INTERMEDIATE name that is transformed downstream, not a final output,
        # so they must not be gated). Final top-level name() calls leave this
        # False -> the gate applies.
        self._disable_opsin_validity_gate: bool = _disable_opsin_validity_gate
        # v29 Phase 4 (verification as a filter over the dispatch cascade).
        # `_excluded_dispatch_classes` holds the classes already tried and
        # GATE-REJECTED for the molecule currently being named; the cascade skips
        # them so the next entry gets its turn. `_last_dispatch_class` is the
        # class that produced the name now under gate. Both are per-molecule
        # scratch: name() clears them on entry and in a finally, so nothing
        # leaks between molecules (that would make naming order-dependent).
        # v36 B2: an OPTIONAL seed of dispatch classes to exclude for EVERY
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
        # Phase 173.6 T3: scoped principal-group override (FG name). Set ONLY by the
        # charged chokepoint's S/P-oxoacid-anion re-entry (_reenter_forced) so the
        # anionic group is forced as the principal characteristic group per P-72/P-74.
        # None for every normal name() call -> byte-identical production behaviour.
        self._principal_group_override: Optional[str] = _principal_group_override
        # v37 SP1.3 (offer-not-return, invariant 18): force the k-th-ranked P-44
        # parent instead of ``ranked[0]``. 0 for every normal name() call ->
        # byte-identical. Set >0 ONLY by ``_try_alternate_parent_rescue`` on a
        # FRESH inner instance, after ``ranked[0]`` has already abstained, and
        # every candidate it yields is RT-gated before adoption (0-wrong).
        self._forced_parent_rank: int = _forced_parent_rank
        # Size of the top-level P-44 parent pool from the most recent top-level
        # classification (written by _classify). Lets the offer-retry bound its
        # loop at the real pool length. Reset per top-level name() below.
        self._top_parent_pool_size: int = 1
        # Phase 156 D-17 + AP-19: per-instance counter dict, NEVER
        # module-global. Pre-seed all seven buckets so callers see a
        # complete histogram even before any name() invocation.
        from .validation.opsin_grammar import OpsinGrammar
        self._grammar_stats: Dict[str, int] = {
            k: 0 for k in OpsinGrammar.STAT_KEYS
        }
        if _disable_grammar_validation:
            self._grammar = None
        else:
            # Share the dict by reference per D-17: the grammar
            # instance increments the same dict the Orthonym instance
            # exposes via `get_validation_stats()`.
            self._grammar = OpsinGrammar(stats=self._grammar_stats)

        # Phase 158 D-14 NEW: instantiate CFR router (default-ON; no opt-out
        # flag per CONTEXT D-14 + AP-2 + AP-19). Per audit § 3.6 RL-4
        # fresh-instance invariant: each Orthonym() carries its OWN router
        # with empty dispatch_stats; recursive `Orthonym(...)` calls produce
        # fresh routers, and the byte-identical contract is on NAME OUTPUT
        # only, NOT on per-call dispatch_stats.
        from .routing import ClassFirstRouter
        self._cfr_router = ClassFirstRouter()

        # Phase 158 RL-7 option (b) NEW: skip-decomposition flag threading.
        # `_name_impl(_skip_decomposition=True)` (called by
        # `name_pipeline_only()`) sets this on entry; the
        # DECOMPOSITION_PRE_GENERAL predicate factory reads from kwargs
        # passed by `dispatch()`.
        self._skip_decomposition: bool = False

        # Phase 168 D-08: triviality-controller opt-in (default OFF = Stage A SACRED
        # byte-identical canary invariant). Env override via ORTHONYM_ENABLE_TRIVIALITY_CONTROLLER.
        # Task 1.9 (context resolution 3): trivial_fallback=True implies the
        # triviality controller (one user intent: "allow non-PIN trivial output").
        self._enable_triviality_controller: bool = (
            enable_triviality_controller or _DEFAULT_TRIV or trivial_fallback
        )
        # WARNING #9 fix (CRITICAL for TRIV-03 runtime per CONTEXT D-07): instantiate the OpsinOracle
        # when the flag is ON so the T2 RT-safety gate can run. Without it, OpsinOracle.rt_safe
        # degrades to the permissive always-True fallback and TRIV-03 runtime enforcement is silently
        # disabled. Default OFF keeps the oracle None (zero cost, byte-identical Stage A).
        self._triv_oracle = None
        if self._enable_triviality_controller:
            try:
                from .assembly.retained_substitution import OpsinOracle
                import sys
                from pathlib import Path
                _scripts = str(Path(__file__).resolve().parent.parent.parent / "scripts")
                if _scripts not in sys.path:
                    sys.path.insert(0, _scripts)
                try:
                    from validate_retained_names import find_opsin_jar
                    _jar = find_opsin_jar() or None
                except ImportError:
                    _jar = None
                self._triv_oracle = OpsinOracle(opsin_jar=_jar)
            except Exception as exc:
                logger.warning(
                    "Phase 168 OpsinOracle instantiation failed: %s; "
                    "T2 RT-safety degrades to permissive fallback", exc,
                )
                self._triv_oracle = None

        # Phase 169 D-05: group-splitting flag (default OFF -> byte-identical Stage A).
        # Env override ORTHONYM_ENABLE_GROUP_SPLITTING. The OpsinOracle is instantiated
        # only when the flag is ON so the FAIL-CLOSED per-split RT gate (D-05) can run;
        # default OFF keeps it None (zero cost, byte-identical Stage A).
        self._enable_group_splitting: bool = enable_group_splitting or _DEFAULT_GS
        self._split_oracle = None
        if self._enable_group_splitting:
            try:
                from .assembly.retained_substitution import OpsinOracle
                import sys
                from pathlib import Path
                _gs_scripts = str(Path(__file__).resolve().parent.parent.parent / "scripts")
                if _gs_scripts not in sys.path:
                    sys.path.insert(0, _gs_scripts)
                try:
                    from validate_retained_names import find_opsin_jar
                    _gs_jar = find_opsin_jar() or None
                except ImportError:
                    _gs_jar = None
                self._split_oracle = OpsinOracle(opsin_jar=_gs_jar)
            except Exception as exc:
                logger.warning(
                    "Phase 169 group-split OpsinOracle instantiation failed: %s; "
                    "per-split RT gate FAIL-CLOSED (splits rejected)", exc,
                )
                self._split_oracle = None

    def get_dispatch_stats(self) -> Dict[Any, int]:
        """Phase 158 D-16: per-instance CFR dispatch histogram.

        Returns a defensive copy of the (StoutClass -> int) histogram
        recorded by the CFR router on every dispatch. Per CONTEXT D-16 +
        AP-6 the counter lives on the Orthonym instance via the CFR
        router; reset on demand via `reset_dispatch_stats()`.

        The return type is `Dict[Any, int]` (not `Dict[StoutClass, int]`)
        to avoid eager `from .routing import StoutClass` at module-load
        time, which would create a circular import. Callers that need
        the StoutClass type can import it directly from `orthonym.routing`.
        """
        return self._cfr_router.get_dispatch_stats()

    def reset_dispatch_stats(self) -> None:
        """Phase 158 D-16 + Phase 160 D-18: explicit reset for batch-run boundaries.

        Resets BOTH the Phase 158 outer-CFR (per-instance) counter AND the
        Phase 160 inner-dispatch (module-level) counter. The inner-dispatch
        counter is module-level today (see assembly/inner_dispatch.py
        :_INNER_DISPATCH_STATS) which means it is shared across Orthonym
        instances — calling ``reset_dispatch_stats()`` on one instance
        resets the shared inner counter visible to all instances.
        """
        self._cfr_router.reset_dispatch_stats()
        # Phase 160 D-18: also reset the inner-dispatch counter.
        from .assembly.inner_dispatch import reset_inner_dispatch_stats
        reset_inner_dispatch_stats()

    def get_inner_dispatch_stats(self) -> Dict[str, int]:
        """Phase 160 D-18: inner-dispatch per-handler-id counters.

        Returns a defensive copy of the (handler_id -> int) histogram
        recorded by ``assembly/inner_dispatch.dispatch_inner`` on every
        match. Inner-dispatch is the second stage of the Phase 158 +
        Phase 160 dispatch pipeline: outer CFR routes to a StoutClass;
        for the GENERAL class, inner-dispatch then routes to one of the
        30 handlers in ``INNER_DISPATCH_TABLE``.

        Per CONTEXT D-18: companion to ``get_dispatch_stats()``; CLI
        ``--dispatch-stats`` flag prints both together. Per AP-160-13 the
        underlying counter is module-level (shared across instances)
        because inner-dispatch is a pure-function call site — adding
        per-instance threading would require touching every handler entry
        point. The counter is reset via ``reset_dispatch_stats()`` (which
        clears BOTH outer + inner counters).

        Returns:
            Dict mapping handler_id (str) to dispatch count (int). Empty
            dict if no inner-dispatch calls have happened yet.
        """
        from .assembly.inner_dispatch import get_inner_dispatch_stats as _stats
        return _stats()

    def name_with_tree(self, smiles: str):
        """Phase 160 DECOMP-02 public API: return NamingResult(name, tree, hint).

        Phase 160 ships the NameTreeNode IR substrate alongside the legacy
        ``assemble_name`` path; first-wave handlers (Plans 02-03 ship)
        return ``NamingResult(name=<final string>, tree=None, ...)`` per
        CONTEXT D-05 incremental migration. The ``tree`` field is None for
        the 30 currently-extracted handlers; tree population is a v19+
        follow-up phase. The ``name`` field is byte-identical to
        ``Orthonym.name(smiles)``.

        For tree-emitting handlers (v19+1 onwards), this method returns a
        NamingResult whose ``tree`` is a NameTreeNode and where
        ``name_tree_to_string(tree)`` round-trips to ``name`` byte-for-byte.

        Args:
            smiles: SMILES string to convert to IUPAC name.

        Returns:
            NamingResult NamedTuple with fields:
              - name: str (byte-identical to Orthonym.name(smiles))
              - tree: Optional[NameTreeNode] (None for first-wave handlers)
              - atom_to_locant_hint: Optional[Dict[int, int]]

        Raises:
            ValueError: If SMILES is invalid.
        """
        # Lazy import to avoid composer.py -> name_tree -> namer.py cycle
        # at module-load time.
        from .assembly.name_tree import NamingResult
        # CR-04 part B + W7: install a per-call capture slot via
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
                raise  # invalid SMILES — propagate (matches name() contract)
            except (TypeError, KeyError, IndexError, AttributeError) as _e:
                # SC-3 robustness: name_with_tree must be as resilient as the
                # module-level name_compound. The inner handlers' "pool.best().name
                # raises on None" fall-through contract surfaces here because
                # self.name() lacks name_compound's broad except. Recover with the
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
            # Phase 165 SC-1 + SC-3 boundary guarantee. Two failure modes:
            #  (a) tree is None — salts/ions/radicals/retained names are produced
            #      by paths BELOW dispatch_inner that never write the capture slot.
            #  (b) STALE tree — the inner handler wrote the slot, but downstream
            #      _name_impl processing (decomposition engine, coverage gate,
            #      stereo backstop) OVERRODE the final name afterward, so the
            #      captured tree no longer round-trips to it.
            # In BOTH cases synthesize the sanctioned coarse node (D-03, counted)
            # so name_tree_to_string(tree) == name holds for EVERY SMILES (SC-1)
            # and --dump-tree works universally (SC-3). The str fragment_legacy
            # round-trips verbatim. class_id="coarse_fallback" distinguishes this
            # boundary node from real handler trees in the coarse-bucket report.
            from .assembly.name_tree import NameTreeNode
            from .assembly.name_tree_to_string import (
                NameTreeSerializerError,
                name_tree_to_string,
            )
            # WR-3: this staleness check is the defence point against bad trees,
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
        """Return a defensive copy of the per-instance grammar counters.

        Phase 156 D-17 telemetry accessor. Buckets are pre-seeded in
        `__init__`; counters are mutated in-place by the underlying
        `OpsinGrammar` instance via the shared-by-reference dict.
        """
        return dict(self._grammar_stats)

    # v29 Phase 4. Bounded deliberately: each retry is a full re-name plus an
    # OPSIN round-trip, and the cascade's useful alternatives for one molecule
    # are few. 3 was chosen as the smallest bound that lets a molecule try the
    # GENERAL engine after two specialised classes have been rejected.
    _PHASE4_MAX_GATE_RETRIES = 3

    def _retry_cascade_on_gate_rejection(self, smiles, pre_gate, gated):
        """v29 Phase 4: a GATE REJECTION re-enters the dispatch cascade.

        The cascade in ``_name_impl`` already falls through when a handler
        returns ``None``, but the gates run out here in ``name()`` -- AFTER the
        cascade has exited -- so a handler that produced a name the gate then
        rejected aborted the whole molecule with no retry. That is the defect
        this phase exists to close: "keep the first that clears both gates, fall
        through instead of aborting the molecule."

        ⚠ The roadmap framed this as filtering over ``CandidatePool``. It is not:
        the pool holds ONE candidate in 158 of 158 spied calls. The real
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
            if _final_opsin_validity_gate(
                    cand, smiles, self._grammar_stats,
                    besteffort_unverified=self._general_fallback_unverified) \
                    not in _DESCRIPTIVE_FALLBACK_NAMES:
                return cand
        return gated

    @_budget_scope
    def name(self, smiles: str, *, raise_on_limit: bool = False) -> str:
        """
        Generate IUPAC name from SMILES.

        Args:
            smiles: SMILES string representing the molecule
            raise_on_limit: HYG-02 (Phase 173) opt-in. When True, a provably
                out-of-scope input raises ``OrthonymLimitError(code, message)``
                instead of returning a plausible-but-wrong / descriptive string
                — letting a caller distinguish "can't handle" from "got it
                wrong". Default False preserves the always-emit behaviour
                byte-for-byte (no limit is ever substituted into the result).

                G0 (DD7 S1) note / WR-04: a top-level ring refusal carries the
                specific ``UNSUPPORTED_RING_SYSTEM`` code. When the refusal
                originates inside a RECURSIVE (substituent/fragment) naming call,
                that inner frame returns its ``.message`` and the top-level
                re-derives a generic ``UNNAMEABLE``/``UNSUPPORTED_ELEMENT`` code
                via the post-failure classifier — i.e. the specific ring code
                can be lost for those (rare) nested cases. The molecule still
                fails closed; only the code precision is reduced.

        Returns:
            IUPAC systematic name

        Raises:
            ValueError: If SMILES is invalid
            OrthonymLimitError: If raise_on_limit and the input is out of scope
        """
        # Start runtime fragment cache session (only at top-level depth)
        from .assembly.fragment_naming import start_naming_session, end_naming_session, is_top_level_naming
        start_naming_session()
        # Wave2 T3 (cross-molecule stereo-contamination fix): the confidence /
        # candidate-pool / parent-correctness thread-locals are PER-MOLECULE, but
        # they persist across name() calls (documented at the post-dispatch gate
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
            # v29 Phase 4: same invariant as the clears around it. The
            # gate-rejection exclusion set is PER-MOLECULE; leaking it would
            # make a molecule's name depend on what was named before it, which
            # is precisely the order-dependence this block exists to stop.
            self._excluded_dispatch_classes = self._seed_excluded_dispatch_classes
            self._last_dispatch_class = None
            # v37 SP1.3: same per-top-level invariant. The pool size is stale
            # until _classify reruns; default 1 = "no alternative parent" so a
            # molecule that never reaches P-44 parent selection cannot trigger
            # the offer-retry on a previous molecule's pool.
            self._top_parent_pool_size = 1
            # v25 P0 Task 0.1: reset the typed-abstention telemetry slot per
            # top-level molecule (same invariant as the confidence/pool
            # clears above). Side-effect-only — never changes a name.
            try:
                from .metrics.abstention import clear_abstention
                clear_abstention()
            except Exception:
                pass
            # v30 PE-1: same per-top-level-molecule invariant as the clears above.
            # A batch consumer names hundreds of inputs in one process; without
            # this reset, molecule N's audit would carry molecule N-1's discarded
            # candidates and manufacture selection failures that never happened.
            # Stays ENABLED across the reset (clear_ledger, not disable).
            try:
                from .metrics.candidate_ledger import (
                    clear_ledger as _clear_cand_ledger,
                    is_enabled as _cand_ledger_on,
                )
                if _cand_ledger_on():
                    _clear_cand_ledger()
            except Exception:
                pass
            # v29 P1: reset the binding-proof ledger per top-level molecule,
            # for the same reason as the three clears above. Without it a
            # molecule that records no spine would finalize against the
            # PREVIOUS molecule's record and report its verdict.
            try:
                from .validation.proof_ledger import clear_ledger
                clear_ledger()
            except Exception:
                pass
            # v33 Phase 0 L0: same per-top-level-molecule invariant as the
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
            # v33 Phase 0 L2: same per-top-level-molecule invariant as the two
            # resets above -- `self._offers` is rebuilt at `_finish` from
            # THIS molecule's winner; without this reset a molecule whose
            # audit never reaches the offer-building code (mode "off", an
            # exception, or a non-top-level recursive call) would report the
            # PREVIOUS molecule's offer pool instead of an empty one.
            self._offers = []
            # v33 Phase 0 L3-1: same per-top-level-molecule invariant --
            # `_t4_floor_candidate` is a telemetry stash (mirroring
            # `_last_ger_result`) of the systematic-floor name the LAST
            # `_maybe_append_t4_floor_offer` call computed, if any.
            self._t4_floor_candidate = None
            # v33 Phase 0 cleanup T1: same per-top-level-molecule invariant --
            # `_last_selected_offer` stashes the WINNING `Offer`
            # `_select_rt_passing_offer_name` picked (or `None` if it fell
            # back to `current_name` unchanged), so `name_tiered` can derive
            # tier/is_pin/source from the offer that actually won rather than
            # the process-wide provenance contextvar, which still describes
            # whichever producer ran LAST (the losing primary on a floor win).
            self._last_selected_offer = None
            # v25 G3: publish the engine flag so fragment/component recursion
            # (name_compound builds FRESH namers) inherits it. Top-level only;
            # reset in the finally below.
            try:
                from .metrics.provenance import general_fallback_ctx
                self._gf_ctx_token = general_fallback_ctx.set(
                    self._general_fallback)
            except Exception:
                self._gf_ctx_token = None
            # v30: publish the BEST-EFFORT discriminator for the same reason and
            # with the same lifetime -- read by
            # `composer._integrate_universal_prefixes` to pick the substituent
            # VOCABULARY, which it cannot do from its own arguments because its
            # callers are tier-unaware PIN handlers.
            try:
                from .metrics.provenance import best_effort_ctx
                self._be_ctx_token = best_effort_ctx.set(
                    self._general_fallback_unverified)
            except Exception:
                self._be_ctx_token = None
            # v37 SP1.1b: publish allow_aromatic_general with the same lifetime so
            # the recursive re-entry (name_compound builds a FRESH namer) gives a
            # recursively named fragment the SAME tier the top-level call ran at.
            # Without this a top-level allow_aromatic_general=True was silently lost
            # on recursion. Top-level only; reset in the finally below.
            try:
                from .metrics.provenance import allow_aromatic_general_ctx
                self._aag_ctx_token = allow_aromatic_general_ctx.set(
                    self._allow_aromatic_general)
            except Exception:
                self._aag_ctx_token = None
            # v37 SP5.4: publish the full-coverage opt-in with the same lifetime
            # so the recursive re-entry (name_compound builds a FRESH namer)
            # inherits it. Top-level only; reset in the finally below. Default
            # False keeps D2 unreachable and every tier byte-identical.
            try:
                from .metrics.provenance import full_coverage_ctx
                self._fc_ctx_token = full_coverage_ctx.set(
                    self._full_coverage)
            except Exception:
                self._fc_ctx_token = None
        # --- Wave-2 P2: isotopic substitution decorator (P-82.2.1 / P-45.4) ---
        # RDKit skeleton perception ignores GetIsotope, so an isotope-labeled
        # mol would name as the UNLABELED skeleton (wrong PIN). Route it to the
        # fail-closed decorator BEFORE _name_impl strips the label. Clean
        # pass-through when the mol carries no isotope (has_isotopes gate) —
        # zero cost + byte-identical on the entire unlabeled corpus.
        if is_top_level_naming():
            _iso_probe = Chem.MolFromSmiles(smiles)
            if _iso_probe is not None:
                from .rules.isotopes import has_isotopes, decorate_isotopic_name
                if has_isotopes(_iso_probe):
                    _iso_name = decorate_isotopic_name(smiles, self.style, self)
                    if _iso_name is not None:
                        # v29 P1 (fix wave 1): every exit of name() goes
                        # through _finish. This one lives OUTSIDE the
                        # try/finally below, so it owns its own session
                        # teardown -- try/finally keeps the ordering
                        # identical to the hooked exits (proof asserted
                        # while the session is still live, session ended
                        # before the value is handed back).
                        try:
                            return self._finish(_iso_name, smiles)
                        finally:
                            end_naming_session()
                    # Decorator failed closed on an isotope-labeled molecule: REFUSE.
                    # Falling through would emit the UNLABELED skeleton name (label
                    # silently dropped) — a wrong name the OPSIN validity gate cannot
                    # catch (it parses to the unlabeled structure; SELF-01 ignores
                    # isotopes). Accuracy-first: never emit a label-dropping name.
                    from .metrics.abstention import AbstentionCode, record_abstention
                    record_abstention(AbstentionCode.OTHER,
                                      detail='isotope_decorator_failed')
                    try:
                        return self._finish(_descriptive_fallback(smiles),
                                            smiles)
                    finally:
                        end_naming_session()
        # --- W3-P09 (P-65.6.2.3.2): normalize a charge-imbalanced acid-salt
        # notation. A metal cation + a NEUTRAL polybasic inorganic oxoacid written
        # without the balancing deprotonation ([Na+].OC(=O)O = NaHCO3, net +1) is a
        # valid acid salt in a malformed charge representation. Rewrite it to the
        # balanced salt so species classification -> name_salt (method 2 PIN) ->
        # the self-consistency gate all see the chemically-valid net-0 structure.
        # Fail-closed (returns None) for every other shape -> smiles unchanged.
        # Cheap gate: only a multi-fragment ('.') input carrying a cation ('+')
        # can be this shape, so single-fragment / anion-only inputs skip the parse.
        if is_top_level_naming() and '.' in smiles and '+' in smiles:
            from .rules.salts import normalize_imbalanced_acid_salt
            _bal = normalize_imbalanced_acid_salt(smiles)
            if _bal is not None:
                smiles = _bal
        try:
            # HYG-02 + Wave-0 D1: structural scope pre-check, now UNCONDITIONAL.
            # A wildcard atom makes the input InChIKey uncomputable, so the
            # SELF-01 round-trip oracle fails OPEN and a wildcard input ships a
            # WRONG molecule (CC* -> "ethane"). Refuse it before any producer OR
            # the general-engine recovery runs — recovery would re-run producers
            # on the wildcard mol and re-open the hole, so this must NOT route
            # through the OrthonymLimitError handler's recovery path below.
            # Perf: zero-false-negative pre-filter -- an RDKit dummy/wildcard
            # atom (atomic number 0) is spelled ONLY as `*`/`[*]` (bare `*`,
            # `[*]`, `[1*]`, ...) or as `[#0]`/decorated (`[13#0]`, `[#0-]`,
            # ...) -- both spellings are checked, so the filter has no false
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
                        from .metrics.abstention import (
                            AbstentionCode, record_abstention)
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
                # v25 P0 Task 0.1: a top-level ring-system refusal means the
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
                # v25 G2 (opt-in): a raised limit bypassed the GENERAL-block
                # engine wiring; give the engine its late, fully-gated shot.
                _rec = self._try_general_engine_recovery(smiles)
                if _rec is not None:
                    # v29 P1: an early exit that bypasses even
                    # _apply_trivial_fallback, and the LATE-RECOVERY path --
                    # precisely the one carrying the known stale-certificate
                    # hazard (_retained_structural_preference can replace the
                    # whole name after verify_certificate passed). Leaving it
                    # unhooked would hide the finding this audit exists to see.
                    return self._finish(_rec, smiles)
                # Task 1.9: PIN fails closed; --trivial falls back to a
                # general-only retained name when no PIN could be derived.
                # v29 P1 (fix wave 1): --trivial is a real opt-in and this
                # swap ships a GENUINE retained name (is_failure_name False),
                # so the exit must be audited like any other.
                return self._finish(
                    self._apply_trivial_fallback(_limit.message, smiles),
                    smiles)
            except Exception as _exc:  # noqa: BLE001
                # v30 Phase1 B5: a producer raised an UNEXPECTED exception. A
                # namer must never crash on a valid input -- degrade to the late
                # general-engine recovery (RT-gated -> 0-wrong-safe), and if that
                # too declines, to an honest abstention. Measured: a ring-ketone
                # producer (name_cyclic_oxo_compound) raises on some fused cages
                # the T4 systematic path names completely; before this the
                # exception escaped name() and every caller without its own
                # try/except crashed (only the eval harness's worker caught it).
                # The strict opt-in still re-raises so a real bug is not masked
                # in that mode.
                logger.info("name_impl raised %s; routing to recovery: %s",
                            type(_exc).__name__, _exc)
                if raise_on_limit and is_top_level_naming():
                    raise
                try:
                    _rec = self._try_general_engine_recovery(smiles)
                except Exception:  # noqa: BLE001 - recovery must not re-crash
                    _rec = None
                if _rec is not None:
                    return self._finish(_rec, smiles)
                return self._finish(_descriptive_fallback(smiles), smiles)
            # Universal stereo backstop (Phase 140, STER-16)
            # Only apply at top level -- decomposition fragments handle stereo
            # through their own naming paths.
            if is_top_level_naming():
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    from .assembly.coverage_scoring import retrieve_confidence
                    _conf = retrieve_confidence()
                    handler = _conf.get('handler', 'unknown')
                    # Phase 177 WSB-01 (D-04): thread the authoritative parent
                    # map + phenol flag to the backstop.
                    result = _final_stereo_check(
                        mol, result, handler=handler,
                        atom_to_locant=_conf.get('atom_to_locant'),
                        is_phenol_benzene=_conf.get('is_phenol_benzene'),
                    )
                    # Universal OPSIN-grammar backstop (Phase 156, D-13).
                    # Order: stereo-backstop -> grammar-backstop. Stereo
                    # may have repositioned descriptors that grammar
                    # then re-validates.
                    result = _final_grammar_check(
                        result, smiles, handler,
                        self._grammar, self._grammar_stats,
                    )
                    # SUB-03 (169.5): real-OPSIN validity gate, the last
                    # transform before the name leaves name(). Default-ON,
                    # fail-OPEN on no-JAR. Skipped for neutralize-recurse /
                    # fragment intermediates (self._disable_opsin_validity_gate).
                    if not self._disable_opsin_validity_gate:
                        _pre_gate = result
                        result = _final_opsin_validity_gate(
                            result, smiles, self._grammar_stats,
                            besteffort_unverified=self._general_fallback_unverified,
                        )
                        # v29 Phase 4: the gate is now a FILTER over the
                        # cascade, not a terminal abort. Only fires when a real
                        # name was suppressed; returns the same fallback unless
                        # a later class clears the gate.
                        result = self._retry_cascade_on_gate_rejection(
                            smiles, _pre_gate, result,
                        )
            # DETERMINISM (w2f p11): abstention must have ONE canonical sentinel.
            # Some fail-closed paths (a handler that DECLINES without producing a
            # candidate — e.g. lambda-multiring-spiro unsupported) leave result ==
            # '' , while paths that produce a candidate the validity gate then
            # suppresses go through _descriptive_fallback -> 'unknown organic
            # compound'. For a molecule whose winning path is atom-ordering-
            # dependent, name() then returns '' on some SMILES spellings and
            # 'unknown organic compound' on others — both mean "no name", but they
            # are different strings, so the determinism eval (raw-string compare)
            # flags it. An empty string is never a valid IUPAC name; normalise it
            # to the canonical fallback so abstention is a single deterministic
            # string regardless of which internal path abstained. is_unknown()
            # already treats '' and the fallback as equivalent, so gold matching
            # is unchanged.
            if not (result and result.strip()):
                result = _descriptive_fallback(smiles)
            # v31: a WELDED refusal sentinel is not a name.
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
            # fallbacks still get their shot — invariant 9: removing a wrong
            # output must never be the last step, because it can unmask a worse
            # generator or a silent atom drop. So this can only turn a welded
            # non-name into either a REAL name (recovery succeeds) or an honest
            # labelled abstention, never into silence.
            if result and is_refusal_sentinel(result) and not is_failure_name(result):
                result = _descriptive_fallback(smiles)
            # v25 G2 (opt-in): a candidate suppressed by a downstream gate
            # (SELF-01 / vetoes) left only the failure sentinel; give the
            # engine its late, fully-gated shot before the failure ships.
            if is_failure_name(result):
                _rec = self._try_general_engine_recovery(smiles)
                if _rec is not None:
                    result = _rec
            # v30 tail #7: best-effort DEMOTE-SENIOR-GROUP rescue. A ring system
            # that names cleanly can be blocked because an ACYCLIC senior group
            # (an ester/acid on a substituent) is selected as the principal group
            # and its handler linearises or declines the ring parent (the
            # spiro-quinone alkaloid #7: a pendant acetate makes pg='ester', which
            # diverts the whole molecule from the spiro ring parent that would name
            # it as '...spiro[...]' with the acetate demoted to (acetyloxy)methyl).
            # Re-pick the principal group with the acyclic senior groups excluded
            # and re-name via the _principal_group_override seam; RT-verified, so
            # 0-wrong holds. Runs ONLY when we are about to ship a failure.
            # v37 tier-policy: gated on `_general_fallback` (COMPLETE tier), not
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
                _dr = self._try_demote_senior_group_rescue(smiles)
                if _dr is not None:
                    result = _dr
            # v28 (WSC-02): last-resort DECOMPOSITION before abstaining. A
            # NON-GENERAL handler (e.g. natural_products on a tropane scaffold)
            # can claim a molecule and emit a structure-DROPPING name (`tropane`,
            # dropping an ester) that SELF-01 then suppresses to the failure
            # sentinel — bypassing the GENERAL-only decomposition fallback inside
            # _name_impl. Give the decomposition engine a final, FULLY-GATED shot
            # here. This is purely additive: it runs ONLY when we are about to
            # ship a failure, so it can turn an 'unknown' into a valid name but
            # can NEVER alter a currently-passing name (0 gold regression by
            # construction). The candidate is routed through the SAME SELF-01
            # validity gate (incl. the stereo-strip carve-out for names whose
            # only OPSIN-narrow layer is stereo, P-91/P-93), so 0-wrong holds:
            # a decomposition name whose constitution does not reproduce the
            # input is suppressed right back to the failure sentinel.
            if (is_top_level_naming() and is_failure_name(result)
                    and not self._disable_opsin_validity_gate):
                _dprobe = Chem.MolFromSmiles(smiles)
                if _dprobe is not None:
                    from .decomposition import try_decompose
                    _dname = try_decompose(_dprobe, style=self.style)
                    if _dname and not is_failure_name(_dname):
                        _dgated = _final_opsin_validity_gate(
                            _dname, smiles, self._grammar_stats,
                            besteffort_unverified=self._general_fallback_unverified)
                        if not is_failure_name(_dgated):
                            result = _dgated
            # v37 SP1.3 (offer-not-return, invariant 18): a molecule whose SENIOR
            # (ranked[0]) P-44 parent dead-ends can be rescued by a JUNIOR pool
            # member. Runs ONLY on the ship-a-failure path (so no PIN-nameable
            # molecule is re-rooted -- they succeed at ranked[0] and never reach
            # here), and every junior candidate is RT-gated before adoption
            # (0-wrong). PIN/default output is byte-identical.
            # v37 tier-policy: gated on `_general_fallback` (COMPLETE tier) -- the
            # candidate is FULL-InChIKey RT-verified at the complete tier exactly
            # as the demote-senior rescue above; the method's own internal guard
            # (below) is loosened to match. Default/PIN tier unaffected.
            if (is_top_level_naming() and is_failure_name(result)
                    and self._general_fallback
                    and not self._disable_opsin_validity_gate):
                _ap = self._try_alternate_parent_rescue(smiles)
                if _ap is not None:
                    result = _ap
            # v33 giants Engine 3: a retained natural-product PARENT HYDRIDE
            # (`ursane`, `hopane`, `cevane`, ... -- the P-101.2.7 Table 10.1
            # stereoparents) IS the PIN, and `_final_opsin_validity_gate`
            # whitelists it (`np_stereoparent` carve-out) precisely because
            # OPSIN 2.9.0 cannot parse it. That is right for the PIN tiers, but
            # on the BEST-EFFORT path the whitelist costs a round-trip the
            # general engine can win. Offer its von-Baeyer systematic name and
            # adopt it ONLY when the full-InChI round-trip verifies. Measured
            # over all 53 such skeletons: 51 convert, 2 keep the retained name
            # (`germacrane`, whose systematic form is a NON-RT
            # `...-1,7-dimethyl-4-(propan-2-yl)cyclodecane`, and `corynoxan`,
            # for which the engine produces nothing) -- so an unconditional
            # downgrade would REGRESS those two, which is why the RT gate is
            # load-bearing rather than belt-and-braces. Best-effort only
            # (`_general_fallback_unverified`), so PIN/default output is
            # byte-identical.
            if (is_top_level_naming() and self._general_fallback_unverified
                    and not self._disable_opsin_validity_gate
                    and not is_failure_name(result)):
                _np_sys = self._try_np_systematic_downgrade(smiles, result)
                if _np_sys is not None:
                    result = _np_sys
            # HYG-02: post-failure limit (opt-in). If naming produced no real
            # name, map the failure to a named code. Keyed off an actual failure
            # so it can never fire on a successfully-named compound.
            if raise_on_limit and is_top_level_naming() and is_failure_name(result):
                _probe = Chem.MolFromSmiles(smiles)
                if _probe is not None:
                    raise classify_failure_limit(_probe, smiles=smiles)
            # Task 1.9: PIN fails closed; --trivial falls back to a general-only
            # retained name when the systematic pipeline derived no PIN.
            #
            # v29 P1: the MAIN exit. Everything that can rewrite the producer's
            # string -- stereo backstop, grammar repair, the OPSIN validity
            # gate, decomposition retry, trivial fallback -- has now run, so
            # this is the first point at which the proof can be asserted on
            # what the caller actually receives.
            return self._finish(
                self._apply_trivial_fallback(result, smiles), smiles)
        finally:
            # v25 G3: unwind the propagation ctx published above (top-level
            # sessions only set a token; nested calls leave it None).
            _tok = getattr(self, '_gf_ctx_token', None)
            if _tok is not None:
                try:
                    from .metrics.provenance import general_fallback_ctx
                    general_fallback_ctx.reset(_tok)
                except Exception:
                    pass
                self._gf_ctx_token = None
            # v30: torn down with its sibling. A leaked best-effort flag would
            # put best-effort vocabulary on a LATER pin naming in the same
            # process -- an H3 break that no single-molecule test would show.
            _be_tok = getattr(self, '_be_ctx_token', None)
            if _be_tok is not None:
                try:
                    from .metrics.provenance import best_effort_ctx
                    best_effort_ctx.reset(_be_tok)
                except Exception:
                    pass
                self._be_ctx_token = None
            # v37 SP1.1b: torn down with its siblings. A leaked
            # allow_aromatic_general would put general-tier vocabulary on a LATER
            # PIN naming in the same process -- an H3 break no single-molecule test
            # would show (mirrors the best_effort_ctx reasoning above).
            _aag_tok = getattr(self, '_aag_ctx_token', None)
            if _aag_tok is not None:
                try:
                    from .metrics.provenance import allow_aromatic_general_ctx
                    allow_aromatic_general_ctx.reset(_aag_tok)
                except Exception:
                    pass
                self._aag_ctx_token = None
            # v37 SP5.4: torn down with its three siblings. A leaked
            # full_coverage flag would arm D2 on a LATER default-tier naming in
            # the same process -- the exact isolation break SP5.4 exists to
            # prevent (mirrors the best_effort_ctx / allow_aromatic_general_ctx
            # reasoning above).
            _fc_tok = getattr(self, '_fc_ctx_token', None)
            if _fc_tok is not None:
                try:
                    from .metrics.provenance import full_coverage_ctx
                    full_coverage_ctx.reset(_fc_tok)
                except Exception:
                    pass
                self._fc_ctx_token = None
            end_naming_session()

    def name_tiered(self, smiles: str) -> dict:
        """v25 G3: name + honest tier/provenance labels (observation-only).

        Returns ``{name, tier, is_pin, source, opsin, gates_passed}`` where
        tier is ``pin_verified`` (PIN path) / ``systematic_verified`` (engine
        or trivial-retained, RT-verified) / ``best_effort`` (engine,
        E1-only, unverified — requires the ``general_fallback_unverified``
        opt-in) / ``abstain``. ``pin_unverified`` (systematic-PIN
        certification) is reserved. Default construction (no flags) keeps
        today's PIN-or-abstain behavior byte-identically.
        """
        from .metrics import provenance as _pv
        from .metrics.provenance import clear_provenance, get_provenance
        clear_provenance()
        name = self.name(smiles)
        prov = get_provenance()
        source = prov["source"] or "pin_path"
        # v29 P7 T1: what the gate DID for THIS name, not whether a jar exists.
        # The old `gate_active = (not disabled and jar_present())` stamped
        # `opsin="verified"` / `SELF-01` on every PIN-path emission on a
        # machine with a jar, including the gate's ten by-design carve-outs and
        # its stereo-stripped branch — three OPSIN-UNPARSEABLE names reported
        # SELF-01 (FINDINGS.md §2). `resolve_gate_outcome` additionally denies
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
        # v27 P6 T6.4: only surface stereo_unexpressed when a general-engine
        # name actually shipped (the contextvar could be set by an emission the
        # downstream constitutional gate later suppressed -> failure row).
        stereo_unexpressed = False
        # v30: same guard, same reason -- the contextvar can be set by an
        # assembly-tier candidate that the constitutional gate then SUPPRESSED,
        # and tagging an abstention row as an ill-formed emission would corrupt
        # the very census this field exists to make possible.
        suffix_free_prefix_name = False
        if not name or is_failure_name(name):
            tier, is_pin, opsin = ABSTAIN, False, "n/a"
            source = prov["source"] or "abstain"
            # v25 G4: descriptive last-resort row — the abstain residual is
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
            # v28 Composer1 Task 5: best-effort clean-abstain contract. In the
            # general-fallback tier the honest outcome for a molecule neither the
            # PIN path nor the general engine can name is a CLEAN abstain — no
            # name. Surfacing the PIN always-emit descriptive residual here (e.g.
            # a partial '…-unknown-…' string, or 'unknown organic compound') would
            # masquerade a non-name as a name for an abstain row that is already fully
            # LABELED by (tier=abstain, source=abstain, limit_code, formula). Scoped to
            # ``self._general_fallback`` so the PIN-default path (and its always-
            # emit ``name()``) is byte-identical; ``name()`` itself is untouched.
            if self._general_fallback:
                name = None
        elif source == "general_engine":
            # The general engine runs its OWN OPSIN round-trip against the
            # input structure (`_rt_match`, :2921) and publishes the verdict as
            # prov["opsin"], which is always a non-empty string — so this
            # branch never consulted jar presence and is unchanged.
            opsin = prov["opsin"] or gate_opsin_label
            tier = SYSTEMATIC_VERIFIED if opsin == "verified" else BEST_EFFORT
            is_pin = False
            stereo_unexpressed = bool(prov.get("stereo_unexpressed"))
            suffix_free_prefix_name = bool(prov.get("suffix_free_prefix_name"))
        elif source == "trivial_retained":
            tier, is_pin = SYSTEMATIC_VERIFIED, False
            opsin = gate_opsin_label
        elif prov.get("general_ring_prefix"):
            # v30 P3-T1c: a composer name carrying a ring substituent prefix only
            # the GENERAL tier could build (a systematic replacement / von Baeyer
            # substituent form). Valid, but not PREFERRED -- the ring PIN may be a
            # retained name -- so it must not ship as pin_verified/is_pin. Demoted on
            # the same verified/unverified split the general engine uses, so the tier
            # still means what it means everywhere else.
            opsin = gate_opsin_label
            tier = SYSTEMATIC_VERIFIED if opsin == "verified" else BEST_EFFORT
            is_pin = False
        else:
            tier, is_pin = PIN_VERIFIED, True
            opsin = gate_opsin_label
        # v33 Phase 0 cleanup T1: the branches above derive tier/is_pin/
        # source from the provenance CONTEXTVAR, which reflects whichever
        # producer ran LAST -- correct when `self._offers` holds a single
        # primary offer (the contextvar and that offer are built from the
        # identical snapshot, so nothing changes below), but WRONG when
        # `_select_rt_passing_offer_name` picked a DIFFERENT, later-appended
        # offer over that primary (e.g. the T4 systematic floor beating a
        # stereo-wrong primary): the contextvar still describes the LOSING
        # primary. Re-derive from the WINNING `Offer` itself whenever one won
        # and its `.name` is exactly what shipped (`_last_selected_offer` is
        # `None` when `select_rt_passing` found no complete+RT-passing offer,
        # in which case `current_name` ships unchanged and there is nothing
        # to override). Guarded off the abstain branch too, though that is
        # belt-and-braces: a winning offer can never be a failure-name
        # sentinel (both offer-construction sites in `_finish` already
        # refuse to append one).
        _offer_winner = getattr(self, "_last_selected_offer", None)
        if (_offer_winner is not None and _offer_winner.name == name
                and tier != ABSTAIN):
            source = _offer_winner.source
            is_pin = _offer_winner.is_pin
            tier = _offer_winner.tier
        gates = []
        if source == "general_engine":
            gates.append("atom_coverage")
            # E1-path verification is the engine's OWN round-trip check
            # (prov["opsin"]), not a gate outcome, so its token is unchanged.
            if opsin == "verified":
                gates.append("self_consistency")
        elif opsin != "n/a":
            # pin_verified / systematic_verified-retained: the token is
            # whatever the gate actually earned — `SELF-01` only for a
            # full-name SELF-01 "ok",
            # `SELF-01(constitution)` for the stereo-stripped verdict, and
            # nothing at all for a carve-out, a bypass or a fail-OPEN.
            _token = _pv.gate_token_for_gate_outcome(gate_outcome)
            if _token is not None:
                gates.append(_token)
        # v37 SP5.4: honest verification-provenance label for the full-coverage
        # tier. Derived from what ALREADY shipped, deny-by-default "unverified":
        #   "identity" -> a D1 exact-InChIKey coordination retained-name hit
        #   "opsin"    -> the name round-trips through OPSIN (SELF-01 / engine RT)
        #   "reconstructor" -> (SP5.6, if built) structurally reconstructed
        #   "unverified"    -> emitted with no passing oracle (T4 / D2 additive),
        #                      or an abstain row. Never claims a verification the
        #                      emission did not earn (mirrors the allowlist
        #                      discipline of ``opsin_label_for_gate_outcome``).
        verified = "unverified"
        if name and not is_failure_name(name):
            _is_d1_identity = False
            try:
                _is_d1_identity = _coordination_retained_name(smiles) == name
            except Exception:
                _is_d1_identity = False
            if _is_d1_identity:
                verified = "identity"
            elif opsin in ("verified", "verified_constitution_only"):
                verified = "opsin"
        return {"name": name, "tier": tier, "is_pin": is_pin,
                "source": source, "opsin": opsin, "gates_passed": gates,
                "gate_outcome": gate_outcome,
                "formula": formula, "limit_code": limit_code,
                "stereo_unexpressed": stereo_unexpressed,
                "suffix_free_prefix_name": suffix_free_prefix_name,
                "verified": verified}

    def _retained_structural_preference(self, mol) -> Optional[str]:
        """v26 P6 (Heritage A3): retained/fusion structural-recognizer
        preference -- a PIN-QUALITY GUARDRAIL, not a naming path.

        Runs the EXISTING retained-name recognizers on the WHOLE molecule
        GRAPH (never string surgery, never a new catalog) so the general
        engine's late-recovery can prefer a clean retained/fused-heterocycle
        name over an ugly von-Baeyer/replacement emission, when one applies.
        Tried in order (first hit wins; each is an exact whole-molecule
        match -- none of these know about substituents, so this can only
        ever match a BARE ring system):

          1. ``data.fused_heterocycles.get_fused_heterocycle_name`` -- the
             curated fused-ring catalog (quinazoline, indole, purine, ...).
          2. ``data.retained_names.get_retained_name`` on the whole-molecule
             canonical SMILES -- simple retained trivial names.
          3. The ``heterocycles.py`` monocyclic retained lookup (reuses
             ``get_ring_canonical_smiles`` + ``get_retained_name``),
             restricted to a molecule that IS a single ring covering every
             heavy atom (no exocyclic substituents) so the name recovered
             cannot silently omit part of the structure.

        (1) is tried before (2) deliberately: a v26 P6 reproduce-first data
        audit found >=2 canonical-SMILES keys shared between the two tables
        where ``get_retained_name`` carries a WRONG (non-isomeric) name for
        a fused ring system that ``FUSED_HETEROCYCLE_DATA`` already has
        correct -- e.g. benzo[f]quinoline's canonical SMILES also keys
        ``ALL_RETAINED_NAMES`` to ``benzo[h]quinoline`` (a different
        connectivity; OPSIN-verified NOT to round-trip), and quinolizidine's
        key resolves to ``decahydroisoquinoline`` (also a different
        connectivity). Checking (1) first means the caller's SELF-01 gate
        never even has to catch these -- and it still would, since this
        method never itself verifies its answer.

        Returns the candidate name string or ``None`` (no recognizer
        matched). The caller re-verifies whatever this returns through the
        SAME OPSIN SELF-01 gate as the general name and keeps the general
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

    def _stereo_emit_decision(self, mol, cand):
        """v27 P6 T6.2: ONE shared stereo-policy decision for BOTH general-
        engine emission sites (the inline G1 fallback in ``_name_impl`` and the
        late-recovery ladder in ``_try_general_engine_recovery``).

        Returns ``(permitted, flagged)``:
          * ``(True, False)``  — the name already expresses every stereo element
            the input carries (or the input has none) -> ship as-is, no flag.
          * ``(True, True)``   — best-effort (``_general_fallback_unverified``):
            ship a CONSTITUTION-ONLY name and mark it ``stereo_unexpressed``.
            This is honest and NOT a wrong stereoisomer — it names a superset
            (BB P-91.2.2 sanctions omitting stereodescriptors for exactly the
            von-Baeyer/spiro/fused polycyclic classes best-effort emits;
            P-91.3: descriptors are an additive layer over a complete name).
          * ``(False, False)`` — pin/valid/complete: ABSTAIN on dropped stereo.
            SELF-01 is stereo-blind, so a stereo-dropped name would pass it and
            could ship a WRONG stereoisomer under the PIN contract (P-91.2.1:
            PINs must specify every stereogenic unit). Fail-closed is correct.

        Root-cause: replaces the two inline ``needs_stereo_injection`` branches
        that previously fail-closed at both sites, so best-effort no longer
        inherits the complete-tier fail-close (roadmap Phase 6 stereo policy).

        v27 Phase S Task 1 (accuracy keystone): uses the per-element
        ``general_engine_stereo_complete`` predicate — NOT the coarse
        ``needs_stereo_injection`` name-side boolean — so a PARTIAL-stereo name
        (some elements expressed, one dropped) is treated as INCOMPLETE and
        never ships in complete (all-or-nothing; P-91.2.1). This closes the
        verified partial-stereo wrong-ship hole (the old boolean returned False
        on any name with a stereo block, shipping partials past SELF-01).
        """
        from .rules.stereochemistry import general_engine_stereo_complete
        if general_engine_stereo_complete(mol, cand):
            return (True, False)
        if self._general_fallback_unverified:
            return (True, True)
        return (False, False)

    @staticmethod
    def _rt_match(input_smiles: str, opsin_smiles: str,
                  stereo_flagged: bool) -> bool:
        """v27 P6 T6.3: SELF-01 round-trip compare at the granularity the name
        asserts.

        For a ``stereo_unexpressed``-flagged emission this is a PER-ELEMENT
        atom-mapped stereo compare (v30 RISK 3, closed here). ``_stereo_emit_decision``
        flags a name whenever it is stereo-INCOMPLETE, which INCLUDES a name that
        asserts PARTIAL stereo — so the compare must verify every element the name
        DOES assert while tolerating the ones it omits. It does this in two steps:

          1. a constitution guard — same molecular graph modulo stereo. Both sides
             are copied, ``RemoveStereochemistry``'d, and compared as ISOMERIC
             canonical SMILES. Removing only stereo (not ``isomericSmiles=False``,
             which also erases isotope labels) keeps BOTH charge AND isotope, so
             neutral != anion and a ``13C``-labelled input != its plain name (P-82;
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
        and probes: `` (RISK 3),
        `*.py``.

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
                return _i.HasSubstructMatch(_o, useChirality=True)
            return Chem.CanonSmiles(opsin_smiles) == Chem.CanonSmiles(input_smiles)
        except Exception:
            return False

    def _try_general_engine_recovery(self, smiles: str) -> Optional[str]:
        """v25 G2 (opt-in): late engine recovery for abstentions.

        Covers the two flows the in-``_name_impl`` wiring cannot see:
        (a) a handler raised a NamingLimit (e.g. UNSUPPORTED_RING_SYSTEM)
        that bypassed the GENERAL block, and (b) a candidate name was
        SUPPRESSED by a downstream gate (SELF-01 / vetoes) after
        ``_name_impl`` returned. Re-perceives, runs the general engine,
        and re-applies the SAME E1 + SELF-01 gates to the emission.
        Returns a verified name or None; never raises (fail-closed).
        Default OFF (``general_fallback=False``) -> byte-identical.
        """
        if not self._general_fallback:
            return None
        from .assembly.fragment_naming import is_top_level_naming
        if not is_top_level_naming():
            return None
        # v33: before the von-Baeyer engine, prefer a retained fused-heterocycle
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
            # v31 T4 FINAL-REVIEW FIX 1: mark whether `cand` came from the
            # aggressive T4 producer, so the shared ladder below can require it
            # to POSITIVELY round-trip (never ship a T4 name OPSIN cannot parse).
            # Stays False for the multi-fragment and engine-own-name paths, so
            # their `general_fallback_unverified` semantics are byte-identical.
            _cand_from_t4 = False
            # B5 general lever: retained so the RT-mismatch rescue below can
            # re-run the T4 cascade. Stays None on the multi-fragment path
            # (which never perceives single-component features), so the rescue
            # is single-component-only by construction.
            feats = None
            if len(Chem.GetMolFrags(mol)) > 1:
                # v26 P4: multi-fragment split-name-join, complete tier ONLY.
                # `name_general` (and every single-component handler) refuses a
                # multi-fragment mol, so the recovery would otherwise abstain.
                # Under `complete` (allow_aromatic_general), name each NEUTRAL
                # component through the complete-tier single-component pipeline
                # and re-assemble via the existing P-14.8.1 adduct namer
                # (order/proportion/water-last preserved). Fail-closed if ANY
                # component is unnameable OR charged (charged -> P5 scope). The
                # whole-string SELF-01 gate below round-trips the joined name.
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
                # v26 P5: charge is lifted only under complete
                # (allow_aromatic_general); the E1 cert must accept the charged
                # partition there too (the charge is a suffix on a bound atom).
                # Phase 1 B4: the shared best-effort certification gate (E1 +
                # structural binding-spine axes: atom/bond/charge/stereo). A
                # structural void routes to the T4 producer below exactly like an
                # E1 failure -- catching a bad eng result EARLY so T4 gets a
                # chance (the measured +1 breadth-routing gain). structural_only
                # keeps the P4/P5/P6 name-spelling FPs advisory (SELF-01 covers
                # name well-formedness downstream).
                if eng is None or not certify_general_result(
                        mol, eng,
                        allow_charged=self._allow_aromatic_general,
                        structural_only=True):
                    # v31 T4 (best-effort tier ONLY): the engine's own
                    # flag-gated attempt just declined (name_general returned
                    # None) or produced a non-atom-complete partition. Hand off
                    # to the universal coverage-by-construction producer, which
                    # re-runs name_general at the T4-PERMISSIVE flags
                    # (allow_aromatic_general / allow_suffix_free) INDEPENDENT of
                    # this instance's flags and E1-audits the result internally,
                    # returning an atom-complete name or None (clean abstain).
                    # PIN-isolated: this whole recovery is already
                    # `_general_fallback`-gated (early `return None` at the top).
                    # The T4 name is NOT shipped raw: it falls through to the SAME
                    # stereo-emit decision + OPSIN round-trip ladder below that the
                    # engine's own name gets, so a T4 name whose constitution does
                    # not round-trip is suppressed to abstain (SELF-01).
                    # Task 4 (coverage-by-construction breadth).
                    #
                    # v37 tier-policy: this is the MEASURED demotion gate for the
                    # ~88/150 ring+stereo abstainers that ship a FULL-InChIKey
                    # RT-verified name at best-effort but abstained at COMPLETE
                    # (spy:  CORESTEREO
                    # addendum + task-tierpolicy-report.md). It was
                    # `_general_fallback_unverified` (best-effort only); re-gated
                    # to `_general_fallback` so a T4 name also ships at the COMPLETE
                    # tier -- but ONLY through the RT ladder below. At the complete
                    # tier `_stereo_emit_decision` returns `(True, False)` only for
                    # a FULL-stereo name (else `(False, False)` -> abstain), so the
                    # `_rt_match` is a FULL-InChIKey compare; and a T4 name OPSIN
                    # cannot parse hits `elif ... or _cand_from_t4: return None`
                    # (below) regardless of tier. An unverified T4 candidate is
                    # therefore never shipped at COMPLETE -- 0-wrong preserved.
                    # `name_t4_complete` still requires a live jar (below), and the
                    # default/PIN tier never reaches here (`_general_fallback`
                    # False), so PIN output is byte-identical.
                    if not self._general_fallback:
                        return None
                    # T4 runs the engine at allow_aromatic_general=True
                    # unconditionally, which OPENS the von-Baeyer / mancude-cage
                    # path even for instances that deliberately kept
                    # allow_aromatic_general OFF. E1 proves atom COVERAGE but not
                    # ring-numbering validity / PIN-preference, so an aggressive
                    # von-Baeyer name T4 opens can be caught ONLY by the OPSIN
                    # round-trip ladder below -- and with no jar that gate fails
                    # OPEN (D-13), so T4 would ship an UNVERIFIED (possibly
                    # non-PIN or valence-illegal) name. Keep the abstention
                    # without a jar, mirroring the inline site's v26 P7 FIX 1
                    # abstain-without-Java contract. (Internal/test mode via
                    # _disable_opsin_validity_gate is a controlled context and
                    # keeps the ladder's own unverified-ship path.) This leaves
                    # the engine's own-name path below byte-identical; it only
                    # constrains the NEW T4 producer to a live SELF-01.
                    if not (self._disable_opsin_validity_gate
                            or _validity_gate_jar_present()):
                        return None
                    from .assembly.t4_coverage import name_t4_complete
                    from .assembly.fragment_naming import isolated_naming_session
                    # v30 Phase1 B5: T4 is a fresh whole-molecule naming, but the
                    # recovery lane invokes it mid-name() at session_depth >= 1, so
                    # its recursion hits MAX_NAMING_DEPTH prematurely and DEGRADES
                    # to an abstention (measured root cause). Run it in an isolated
                    # depth-0 session so it gets the full recursion budget a
                    # standalone call gets.
                    with isolated_naming_session():
                        cand = name_t4_complete(mol, feats)
                    if cand is None:
                        return None
                    # v31 T4 FINAL-REVIEW FIX 1: this T4 name must POSITIVELY
                    # round-trip or the path ABSTAINS. The shared ladder below
                    # ships a `general_fallback_unverified` emission even when
                    # OPSIN CANNOT PARSE it (its `elif not
                    # self._general_fallback_unverified` is False for the T4
                    # opt-in) -- correct for the engine's OWN best-effort name,
                    # but NOT for the aggressive T4 producer, which opens
                    # von-Baeyer / mancude paths whose ONLY validity check is
                    # this round-trip. E1 proves atom COVERAGE (so never a wrong
                    # MOLECULE) but a name OPSIN cannot parse is malformed. Flag
                    # the T4 origin so the ladder rejects an unparseable T4 name
                    # instead of shipping it unverified; the parse-but-mismatch
                    # case is already rejected there for every tier. T4-SCOPED:
                    # the flag is False everywhere else, so the engine-own and
                    # existing-tier paths are untouched.
                    _cand_from_t4 = True
                else:
                    cand = eng.name
                    # v29 P1 (audit-only): record the spine the certificate just
                    # accepted, so the exit of name() can re-assert it on the
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
                    # v26 P6: retained-name preference (PIN-quality guardrail,
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
            # v26 P7 FIX 2: fail closed on DROPPED stereo (general-engine path).
            # The SELF-01 ladder below is CONSTITUTIONAL (atoms+bonds+charge,
            # stereo-blind), so a general emission that omits E/Z or R/S the
            # input carries would pass SELF-01 and ship a WRONG stereoisomer
            # specification. The universal stereo backstop is LOG-ONLY for the
            # handler='unknown' cohort that general emissions arrive as (no
            # authoritative parent locant map), so it cannot suppress this. Any
            # stereo the engine DOES express (parent-scope _stereo_prefix or
            # substituent-internal name_substituent) makes needs_stereo_injection
            # False, so genuinely-expressed cases still ship WITH stereo -- only
            # genuinely-dropped-stereo emissions abstain here.
            # v27 P6 T6.2: shared stereo-emit policy (was an inline
            # needs_stereo_injection fail-close). complete/valid abstain on
            # dropped stereo; best-effort ships a constitution-only name flagged
            # stereo_unexpressed (0-wrong: names a superset, never a wrong
            # stereoisomer).
            _stereo_flagged = False
            if cand:
                _permitted, _stereo_flagged = self._stereo_emit_decision(
                    mol, cand)
                if not _permitted:
                    return None
            # v25 G3: explicit verification ladder. verified = OPSIN parsed
            # the name AND it round-trips to the input structure. A parsed-
            # but-MISMATCHED name is NEVER shipped, at any tier. task-JAR-ABSENT
            # (FABLE 0-wrong hole): the no-jar best-effort branch below no
            # longer ships unverified -- the old claim that "no-jar transience
            # ships ONLY behind the T4 opt-in ... a genuine environment gap,
            # not a proof gap" mischaracterized the hole (FABLE's witnesses were
            # demonstrably WRONG-molecule, not merely-unprovable-but-correct),
            # so the branch now routes through `verify_or_none` and abstains.
            # v33 no-abstain Phase A: when the jar IS present but OPSIN
            # rejects `cand` outright, the WIRED oracle below decides --
            # see the `else` branch after the `_opsin_smi is not None` check.
            opsin_status = "unverified"
            if self._disable_opsin_validity_gate:
                pass  # test/internal mode: ship unverified (G2 contract)
            elif not _validity_gate_jar_present():
                if not self._general_fallback_unverified:
                    return None
                # v33 no-abstain / task-JAR-ABSENT (FABLE 0-wrong hole): a
                # jar-absent best-effort emission must NOT ship `cand`
                # unverified. Before this fix control fell straight through
                # this whole if/elif/else to the emit at the bottom with
                # opsin_status="unverified" and ZERO verification of any kind
                # -- FABLE measured 7/7 WRONG-molecule ships in a no-Java
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
                    # v27 P6 T6.3: compare at the granularity the name asserts
                    # -- constitution-only for a flagged emission, exact
                    # isomeric otherwise.
                    if not self._rt_match(smiles, _opsin_smi, _stereo_flagged):
                        # v30 Phase1 B5 general lever: the recovery lane's own
                        # (rung-0) general name denotes a DIFFERENT molecule, but
                        # the T4 PG-suppressed cascade may build a complete,
                        # correct one (measured: 9/45 in-scope suppressed rows).
                        # Try it ONCE -- best-effort + live-jar only, positively
                        # RT-verified -- rather than abstain. Never fires for a
                        # name that was already T4 (`_cand_from_t4`) or on the
                        # multi-fragment path (`feats is None`).
                        _rescue = (None if (_cand_from_t4 or feats is None)
                                   else self._try_t4_rescue(mol, feats, smiles))
                        if _rescue is None:
                            return None  # wrong name: never ship
                        cand, _stereo_flagged = _rescue
                    opsin_status = "verified"
                elif not self._general_fallback_unverified or _cand_from_t4:
                    # rejected/transient: no unverified opt-in, OR a T4 name
                    # that must POSITIVELY round-trip (FINAL-REVIEW FIX 1). For
                    # a non-T4 name `_cand_from_t4` is False, so this reduces to
                    # the pre-existing `not general_fallback_unverified` guard --
                    # byte-identical for the engine-own and multi-fragment paths.
                    return None
                else:
                    # v33 no-abstain Phase A (the `verify_or_none` keystone):
                    # OPSIN could not parse `cand` at ALL (jar present;
                    # rejected or a transient in-process failure -- the two
                    # are indistinguishable from a single None here, exactly
                    # as the pre-existing branches above already treat them).
                    # Before this SHIPPED completely unverified -- `opsin_status`
                    # stayed "unverified" with literally no proof, the exact gap
                    # measured live on dev500 (3 witnesses: a spiro-fused
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
                    # T4 `result_obj`), and there is no name->NameFacts
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
                    # fix-round-1 Finding 3: with `name_facts=None`, the ONLY
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
                from .metrics.provenance import (
                    record_source, record_stereo_unexpressed)
                record_source("general_engine", opsin=opsin_status)
                record_stereo_unexpressed(_stereo_flagged)
                return cand
        except Exception as e:  # fail-closed: keep the abstention
            logger.info(
                "general engine late recovery error (kept abstention): %s", e)
        return None

    def _try_demote_senior_group_rescue(self, smiles: str):
        """v30 tail #7: best-effort rescue that re-picks the principal group with
        ACYCLIC senior characteristic groups (ester / carboxylic acid / their
        chalcogen analogues) excluded, then re-names via the
        ``_principal_group_override`` seam so a ring parent that was blocked by an
        off-ring senior group can name (with that group demoted to a detachable
        prefix). Returns a POSITIVELY round-tripping name or ``None`` (fail-closed).

        Guarded: best-effort opt-in + live OPSIN jar (the caller already checks
        both); only fires on the ship-a-failure path so it can never alter a
        passing name. The re-perception mirrors what do --
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
            _permitted, _flagged = self._stereo_emit_decision(mol, cand)
            if not _permitted:
                return None
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
        """v37 SP1.3 (offer-not-return, invariant 18): best-effort rescue that,
        when ``ranked[0]`` (the PIN-correct senior parent) leads to an
        un-nameable remainder, RETRIES the JUNIOR members of the same P-44 pool
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

        PIN-safety (invariant 1 + 8): the P-44 ranking is NEVER reordered and
        ``ranked[0]`` is never changed -- this runs ONLY on the ship-a-failure
        path at the best-effort tier (the caller checks
        ``is_top_level_naming() and is_failure_name(result) and
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
        # v37 tier-policy: run at the COMPLETE tier (`_general_fallback`), not
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
                1 for a in mol.GetAtoms()
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
                        _permitted, _flagged = self._stereo_emit_decision(mol, cand)
                        if _permitted:
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
        """v30 Phase1 B5 general lever: build a T4 PG-suppressed cascade name when
        the recovery lane's own rung-0 general name denoted a DIFFERENT molecule
        (SELF-01 RT-mismatch). Returns ``(name, stereo_flagged)`` for a POSITIVELY
        round-tripping T4 name, or ``None``.

        Guarded exactly like the T4 handoff above -- best-effort opt-in
        (``_general_fallback_unverified``) and a live OPSIN jar (or the test-mode
        gate bypass) -- so PIN/complete/valid tiers and jarless runs never reach
        it (the caller only invokes it on the best-effort RT-mismatch abstention
        path). Cannot recurse: ``name_t4_complete`` is a direct producer call,
        never ``name_tiered``. The T4 name is E1+spine certified inside
        ``name_t4_complete`` AND re-verified by the positive RT check here, so a
        rescue can only ever turn an abstention into a right-molecule emission.
        """
        if not self._general_fallback_unverified:
            return None
        if not (self._disable_opsin_validity_gate or _validity_gate_jar_present()):
            return None
        try:
            from .assembly.t4_coverage import name_t4_complete
            from .assembly.fragment_naming import isolated_naming_session
            # v30 Phase1 B5: isolated depth-0 session (see the handoff site) so
            # T4 gets the full recursion budget a standalone call gets, rather
            # than the elevated session_depth of the enclosing name().
            with isolated_naming_session():
                cand = name_t4_complete(mol, feats)
        except Exception as e:  # fail-closed: a producer bug keeps the abstention
            logger.info("t4 rescue error (kept abstention): %s", e)
            return None
        if not cand or is_failure_name(cand):
            return None
        _permitted, _flagged = self._stereo_emit_decision(mol, cand)
        if not _permitted:
            return None
        if self._disable_opsin_validity_gate:
            return (cand, _flagged)  # test/internal mode
        smi = _validity_gate_name_to_smiles(cand)
        if smi is not None and self._rt_match(smiles, smi, _flagged):
            return (cand, _flagged)
        return None

    def _name_multifragment_complete(self, mol, canonical_smiles: str) -> Optional[str]:
        """v26 P4: name an all-neutral multi-fragment input under `complete`.

        Delegates to the P-14.8.1 adduct assembler (``rules.adducts.name_adduct``)
        with the per-component namer switched to the ``complete`` tier
        (``general_fallback`` + ``allow_aromatic_general``). name_adduct handles
        component ordering (P-14.8 seniority / water-last), proportion notation,
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
        """v29 P1 (fix wave 1): THE single exit of :meth:`name`.

        ``name()`` has five ``return`` statements -- the isotope decorator
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

        It deliberately depends on NOTHING the ``try:`` block in ``name()``
        sets up -- only on ``self._binding_proof``, the naming depth and the
        two arguments -- so the isotope exit (which sits before that block)
        can route through it unchanged.
        """
        # v36 D1 (coordination retained table): an N-coordinated metal tetrapyrrole
        # (heme/chlorophyll/cobalamin/siroheme/F430) has no OPSIN-parseable name, so
        # no real namer produces a verified name for it -- control lands on the metal
        # sentinel (via _descriptive_fallback, already hooked above) OR, when the
        # OPSIN validity gate is degraded, on a WRONG best-effort fragment that
        # bypasses _descriptive_fallback entirely. Override with the exact-InChIKey
        # ChEBI retained name HERE too, at name()'s single audited exit, so the table
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
        # v30 PE-1: record the name actually returned, in its OWN try/except and
        # BEFORE the ``off`` short-circuit below -- otherwise the emitted record
        # would be silently absent in the default binding-proof mode, which is the
        # mode every measurement runs in.
        #
        # Recorded explicitly rather than inferred by looking the string up among
        # the candidates, because measured (PE1 plan §1, C10): the pool held
        # '(2R)-piperidine-2-carboxylic acid' while naming emitted
        # '(2R)-piperidine-2-carboxylate'. A lookup would report "emitted name not
        # among candidates" for every anion row.
        try:
            from .metrics import candidate_ledger as _cl
            if _cl.is_enabled():
                _cl.record_candidate("namer._finish", _cl.Stage.EMITTED, name)
        except Exception:  # pragma: no cover - telemetry must never break naming
            pass
        # v33 Phase 0 L0: producer-agnostic coverage AUDIT, SHADOW mode
        # (telemetry only -- NEVER changes `name`, only stores a verdict on
        # `self._last_coverage_verdict` and logs one line). Gated to the
        # top-level naming call, the same scope the SELF-01 gate itself uses
        # (`name()`'s main exit calls `_final_opsin_validity_gate` only inside
        # its own `is_top_level_naming()` block), so `_self01_lookup`'s
        # provenance read is never asked about a recursive fragment's state.
        # Skipped for a failure-name winner (an abstention makes no coverage
        # claim to audit, and its sentinel string would just waste an OPSIN
        # re-anchor call on the bare-str fallback path for nothing).
        try:
            _audit_mode = _coverage_audit_mode()
            if _audit_mode != "off":
                from .assembly.fragment_naming import (
                    is_top_level_naming as _cov_is_top_level)
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
                            # GER path never needs SELF-01 -- E1 is Java-free
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
                        # v33 Phase 0 L1: turn the audit into a real VETO.
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
                        # v33 Phase 0 L2/L4-core: wrap the current winner
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
                        # reuses that recorded SELF-01 verdict via
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
                        _offer_ger = getattr(self, "_last_ger_result", None)
                        _offer_result_obj = (
                            _offer_ger if _offer_ger is not None
                            and getattr(_offer_ger, "name", None) == name
                            else None)
                        from .metrics.provenance import (
                            get_provenance as _offer_get_prov)
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
                        elif _offer_prov.get("general_ring_prefix"):
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
                        # v33 Phase 0 L3-1: the systematic FLOOR as a 2nd,
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
                        if self._general_fallback_unverified:
                            self._maybe_append_t4_floor_offer(
                                _cov_mol, smiles, name,
                                primary_is_failure=False)
                        name = self._select_rt_passing_offer_name(name, smiles)
                elif (_cov_is_top_level() and is_failure_name(name)
                        and self._general_fallback_unverified):
                    # v33 Phase 0 L3-1: the CRITICAL wiring wrinkle the task
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
            # error too; and (v33 Phase 0 L2) a failure while building/
            # selecting the offer must never change or crash the name either.
            # `name` is only ever reassigned above, inside this same try
            # block, and each reassignment (veto, then offer-selection) fully
            # completes before the next statement runs -- so an exception
            # anywhere in this block leaves `name` at whichever of those
            # values it last finished assigning (its original winner if
            # nothing fired yet, the L1 fallback if veto fired and the
            # exception came after, etc.), never a half-applied one.
            logger.info("coverage audit failed (shadow, ignored): %s", _cae)
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
        """v33 Phase 0 L4-core: run `select_rt_passing` over `self._offers`
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
        # v33 Phase 0 cleanup T1: stash the winning Offer (or None when no
        # offer both completed and RT-passed, in which case `current_name`
        # ships unchanged and there is no "winner" to report) so `name_tiered`
        # can read its tier/is_pin/source instead of the stale contextvar.
        self._last_selected_offer = selected
        return selected.name if selected is not None else current_name

    def _maybe_append_t4_floor_offer(self, mol, smiles: str,
                                      primary_name: str,
                                      primary_is_failure: bool) -> None:
        """v33 Phase 0 L3-1: append the systematic floor
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
        (perception + classification + the T4 producer cascade), so it is
        only computed when it might actually be NEEDED --

        * ``primary_is_failure=False`` (the ordinary "primary emitted a real
          name" call site): `_offer_rt_ok(primary_name, smiles)` is checked
          FIRST -- a cheap, MEMOIZED lookup (reuses the SELF-01 gate's own
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

        `name_t4_complete` is run inside `isolated_naming_session()` (mirrors
        `_try_general_engine_recovery`'s existing T4 call site, ~:3442-3443):
        called mid-`name()` at `session_depth >= 1` it would hit
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
        #   mode"): when set, `_final_opsin_validity_gate` is never called for
        #   the PRIMARY on its own naming path either (namer.py ~:4296's own
        #   `if not self._disable_opsin_validity_gate:` guard), so no gate
        #   outcome is recorded for it and `_self01_lookup` cannot distinguish
        #   "never checked because gate is off" from "checked and belongs to
        #   a stale, unrelated string" (`resolve_gate_outcome`'s ``bypassed``
        #   case, deny-by-default but NOT in `_self01_lookup`'s skip_reanchor
        #   bucket) -- so `_offer_rt_ok(primary_name, ...)` could go either
        #   way depending on provenance left over from a PRIOR, unrelated
        #   `.name()` call in the same process (measured:
        #   `tests/unit/validation/test_binding_proof_wiring.py::
        #   test_a_retained_name_swap_after_the_certificate_is_reported`,
        #   order-dependent without this guard). Skipping the floor mechanism
        #   entirely when the gate is explicitly off keeps that mode's
        #   pre-L3-1 behaviour unchanged -- exactly what "test/internal mode"
        #   should mean for a NEW OPSIN-gated capability.
        # * A missing jar: mirrors the EXISTING T4 producer's own requirement
        #   (`_try_general_engine_recovery` ~:3431-3433, `_try_t4_rescue`
        #   ~:3646 -- "Keep the abstention without a jar") -- a T4 name must
        #   never ship UNVERIFIED. Without this, `_offer_rt_ok` fails OPEN
        #   (True) for a candidate NO ONE has ever checked, which would let a
        #   von-Baeyer name that must NEVER ship win purely because there was
        #   nothing to disprove it with (measured:
        #   `tests/unit/test_general_fallback_wiring.py::
        #   test_mancude_refused_regardless_of_optin_no_jar`).
        if self._disable_opsin_validity_gate or not _validity_gate_jar_present():
            return
        if not primary_is_failure and _offer_rt_ok(primary_name, smiles):
            return  # primary already full-RT-passes -- the floor is not needed
        try:
            from .assembly.t4_coverage import name_t4_complete
            from .assembly.fragment_naming import isolated_naming_session
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

    def _record_binding_proof(self, mol, eng, stage: str) -> None:
        """v29 P1: record a certified general-engine spine on the ledger.

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
        # v33 Phase 0 L0: stash the raw GeneralEngineResult for the SHADOW
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
        """v29 Phase 1: assert the recorded binding spine on the FINAL name.

        audit  -> observe and log only (the name is returned unchanged).
        enforce-> a failed proof abstains (fail closed). No default path sets
                  enforce in v29 Phase 1.

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

    def _apply_trivial_fallback(self, result: str, smiles: str) -> str:
        """Task 1.9 (PIN-policy --trivial fallback).

        Returns ``result`` unchanged unless ALL of these hold:
          - ``self._trivial_fallback`` is set (opt-in), AND
          - naming produced only the failure signal (``is_failure_name``), AND
          - a general-only (PIN-denied) retained name exists for the molecule.
        In that single case the general-only trivial name is returned. This is
        a fallback for an underivable PIN, never a downgrade of a derived PIN
        (a real name is never a failure name, so this can never fire on one).
        Applied only at the top level so recursive fragment calls are unaffected.
        """
        if not self._trivial_fallback:
            return result
        from .assembly.fragment_naming import is_top_level_naming
        if not is_top_level_naming():
            return result
        if not is_failure_name(result):
            return result
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return result
        from .data import get_general_retained_name
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        trivial = get_general_retained_name(canonical_smiles)
        if trivial:
            from .metrics.provenance import record_source
            record_source("trivial_retained")
            return trivial
        return result

    def name_with_confidence(self, smiles: str) -> dict:
        """Generate IUPAC name with confidence metadata.

        Returns:
            dict with keys:
              - 'name' (str): The IUPAC name
              - 'confidence' (float): 0.0-1.0 aggregate confidence score
              - 'factors' (dict): Individual factor scores
                  {'ratio': float, 'atom_coverage': float,
                   'fg_recognition': float, 'substituent_completeness': float}
              - 'handler' (str): Which handler produced the name

        Raises:
            ValueError: If SMILES is invalid
        """
        from .assembly.fragment_naming import start_naming_session, end_naming_session, is_top_level_naming
        from .assembly.coverage_scoring import (
            clear_confidence, retrieve_confidence, unmeasured_confidence,
        )
        from .metrics.abstention import (
            AbstentionCode, abstention_code_for, clear_abstention,
            record_abstention,
        )
        start_naming_session()
        clear_confidence()
        # v25 P0 Task 0.1: reset the typed-abstention slot (twin of name()).
        if is_top_level_naming():
            clear_abstention()
        try:
            # Wave-0 D1: same wildcard pre-check as name(). name_with_confidence
            # returns a metadata dict, so mirror the limit handler's fallback dict
            # (lines ~4405–4419) rather than name()'s string return.
            # Perf: zero-false-negative pre-filter -- an RDKit dummy/wildcard
            # atom (atomic number 0) is spelled ONLY as `*`/`[*]` (bare `*`,
            # `[*]`, `[1*]`, ...) or as `[#0]`/decorated (`[13#0]`, `[#0-]`,
            # ...) -- both spellings are checked, so the filter has no false
            # negative, and skips the extra RDKit parse on the common
            # (non-wildcard) path.
            if '*' in smiles or '#0' in smiles:
                _probe = Chem.MolFromSmiles(smiles)
                if _probe is not None:
                    _scope = classify_scope_limit(_probe)
                    if _scope is not None:
                        _scope.smiles = smiles
                        record_abstention(AbstentionCode.OTHER, detail=_scope.code)
                        _fallback_name = self._apply_trivial_fallback(
                            _scope.message, smiles)
                        _abst = abstention_code_for(_fallback_name)
                        _md = unmeasured_confidence(
                            name=_fallback_name, handler='fallback')
                        _md['limit'] = _scope.as_dict()
                        _md['abstention'] = _abst.value if _abst else None
                        return _md
            # --- Wave-2 P2: isotopic substitution decorator (P-82.2.1 / P-45.4) ---
            # Twin of name()'s hook (:2938). name_with_confidence bypassed it and
            # called _name_impl directly, so an isotope-labeled mol named as the
            # UNLABELED skeleton (label silently dropped = wrong molecule; the OPSIN
            # validity gate cannot catch it — it parses to the unlabeled structure,
            # and SELF-01's skeleton compare ignores isotopes). Route to the
            # fail-closed decorator BEFORE _name_impl strips the label. has_isotopes
            # gate => zero cost + byte-identical on the entire unlabeled corpus.
            # Nothing scored coverage on this early-return path, so both exits use
            # the honest unmeasured record (v29 C4). We are already INSIDE the try
            # whose finally ends the session, so just return — no second
            # end_naming_session (unlike name(), whose hook sat outside its try).
            if is_top_level_naming():
                _iso_probe = Chem.MolFromSmiles(smiles)
                if _iso_probe is not None:
                    from .rules.isotopes import has_isotopes, decorate_isotopic_name
                    if has_isotopes(_iso_probe):
                        _iso_name = decorate_isotopic_name(smiles, self.style, self)
                        if _iso_name is not None:
                            _md = unmeasured_confidence(
                                name=_iso_name, handler='isotope')
                            _md['limit'] = None
                            _md['abstention'] = None
                            return _md
                        # Decorator failed closed on an isotope-labeled molecule:
                        # REFUSE. Falling through to _name_impl would emit the
                        # UNLABELED skeleton name (label silently dropped). Abstain
                        # to the descriptive fallback, exactly as name() does (:2967).
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
                # v25 P0 Task 0.1: mirror name()'s classification of the
                # limit signal (top-level ring refusal = NO_PARENT).
                if _limit.code == 'UNSUPPORTED_RING_SYSTEM':
                    record_abstention(
                        AbstentionCode.NO_PARENT if is_top_level_naming()
                        else AbstentionCode.BRANCH_UNNAMEABLE,
                        detail=_limit.code,
                    )
                else:
                    record_abstention(AbstentionCode.OTHER, detail=_limit.code)
                # Task 1.9 (C8): apply trivial fallback here, mirroring name().
                _fallback_name = self._apply_trivial_fallback(_limit.message, smiles)
                _abst = abstention_code_for(_fallback_name)
                # v29 C4: nothing scored this either, so report the shared
                # unmeasured record rather than a fabricated 0.0 (a verdict of
                # FAIL is as unfounded as a verdict of PASS when no measurement
                # happened). The 'limit'/'abstention' keys already carry the
                # out-of-scope signal a consumer needs.
                _md = unmeasured_confidence(name=_fallback_name,
                                            handler='fallback')
                _md['limit'] = _limit.as_dict()
                _md['abstention'] = _abst.value if _abst else None
                return _md
            # Universal stereo backstop (Phase 140, STER-16)
            if is_top_level_naming():
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    _conf = retrieve_confidence()
                    handler = _conf.get('handler', 'unknown')
                    # Phase 177 WSB-01 (D-04): thread the authoritative parent
                    # map + phenol flag to the backstop.
                    name = _final_stereo_check(
                        mol, name, handler=handler,
                        atom_to_locant=_conf.get('atom_to_locant'),
                        is_phenol_benzene=_conf.get('is_phenol_benzene'),
                    )
                    # Universal OPSIN-grammar backstop (Phase 156, D-13).
                    name = _final_grammar_check(
                        name, smiles, handler,
                        self._grammar, self._grammar_stats,
                    )
                    # SUB-03 (169.5): real-OPSIN validity gate (twin of name()).
                    if not self._disable_opsin_validity_gate:
                        name = _final_opsin_validity_gate(
                            name, smiles, self._grammar_stats,
                            besteffort_unverified=self._general_fallback_unverified,
                        )
            metadata = retrieve_confidence()
            # v29 C4: no candidate was scored (early-return path). This used to
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
            # Shared with the name() path via one constructor so the two can no
            # longer disagree.
            if not metadata['name']:
                metadata = unmeasured_confidence(name=name)
            else:
                # Ensure name matches (the stored candidate should match
                # what was returned)
                metadata['name'] = name

            # HYG-02 (Phase 173): informational limit annotation (additive — does
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

            # Task 1.9 (C8): apply trivial fallback, mirroring name() line 1483.
            # _apply_trivial_fallback is a no-op when trivial_fallback=False,
            # when called recursively, or when a real PIN was derived — safe.
            metadata['name'] = self._apply_trivial_fallback(metadata['name'], smiles)

            # v25 P0 Task 0.1: typed abstention code (additive key; None for
            # a successfully named molecule).
            _abst = abstention_code_for(metadata['name'])
            metadata['abstention'] = _abst.value if _abst else None

            return metadata
        finally:
            end_naming_session()
            clear_confidence()

    def _try_retained_fused_upgrade(self, mol, smiles: str) -> Optional[str]:
        """v33: a retained fused-heterocycle whole-molecule name (currently the
        purine ring system) when it round-trips, else None.

        A giant purine-containing molecule (an acyl-CoA) is not itself a bare
        fused ring, so the normal dispatch never routes it to
        ``name_fused_heterocycle``; it falls to the von-Baeyer general-engine
        fallback and is named on a ``...tetraazabicyclo[4.3.0]...`` polyene parent
        instead of its PIN-quality ``...9H-purin-6-amine``. This offers the fused
        name as a preferred candidate, adopted by the caller ONLY when its OPSIN
        round-trip verifies -- so a molecule whose fused form is not yet RT-exact
        still falls through to the (valid) von-Baeyer name (no regression; 0-wrong
        via the RT probe + the downstream SELF-01 gate). ``best_effort_ctx`` is
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
        """v33 giants Engine 3: an OPSIN-round-trippable name for a retained
        natural-product parent hydride, when the general-engine recovery can
        produce one -- else None (keep the retained name).

        The mirror of :meth:`_try_retained_fused_upgrade`: that one trades a
        von-Baeyer name for a retained one, this one trades a retained name for
        a recovered one. Both are best-effort-only, RT-gated, whole-molecule
        swaps that leave the PIN default byte-identical.

        `ursane`, `hopane`, `cevane`, ... are valid PINs (P-101.2.7 Table 10.1)
        that OPSIN 2.9.0 cannot parse, so they are whitelisted past the SELF-01
        gate and ship unverified. On the best-effort path a round-trippable name
        is worth more than the retained spelling, so offer it here -- but adopt
        it ONLY on a verified full-InChI round-trip. Most rows recover a
        von-Baeyer systematic parent; the recovery's own retained-name
        preference can also return a trivial name (e.g. `(4E)-sphing-4-enine` ->
        `sphingosine`, RT-verified), which is acceptable on the breadth tier
        (the PIN default keeps the sphingoid parent). The gate is load-bearing,
        not defensive: 2 of the 53 skeletons (`germacrane`, `corynoxan`) recover
        NO round-tripping name, and an unconditional downgrade regresses them
        (to a non-RT cyclodecane-parent name and to the abstention sentinel
        respectively).

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

        Phase 158 substrate: the v18 implicit cascade at lines 853-1115 is
        REPLACED by a single ``self._cfr_router.dispatch(...)`` call. Per
        audit § 3.5 RL-3 option (c) the zwitterion-character in-place
        mutation (v18 lines 1000-1034) lives INLINE here BEFORE the dispatch
        call — preserves D-26 predicate-purity invariant cleanly. Per
        Task 158-02-01 design choice (a) the GENERAL handler shim returns
        None to signal that ``_name_impl`` runs the legacy ``_perceive ->
        _classify -> assemble_name`` pipeline INLINE — keeps ``routing/``
        decoupled from ``composer.py`` (D-19 boundary). Quality-gate
        post-checks at v18 lines 1129-1207 are PRESERVED VERBATIM after
        the dispatch call.
        """
        # Phase 158 RL-7 option (b): thread `_skip_decomposition` through
        # the instance attribute so the DECOMPOSITION_PRE_GENERAL predicate
        # factory + handler shim can read it from kwargs forwarded by
        # `dispatch()`.
        self._skip_decomposition = _skip_decomposition

        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Get canonical SMILES for consistent processing
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

        # ============================================================
        # Pre-dispatch: zwitterion-character in-place mutation (RL-3 (c))
        # ============================================================
        # Phase 158 audit § 3.5 RL-3 option (c) lock: the v18 cascade at
        # namer.py:1000-1034 MUTATED `mol` and `canonical_smiles` in-place
        # for downstream cascade consumption. CFR's predicate-handler model
        # does NOT support "mutate state, continue cascade" patterns
        # natively. Per the audit's UNANIMOUS decision, the mutation lives
        # INLINE here BEFORE CFR.dispatch — preserves D-26 hard invariant
        # (every entry's `side_effect_inventory == ()`) cleanly.
        #
        # CRITICAL byte-identical gate: in the v18 cascade the mutation at
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
        # `detect_species_type()`. These molecules still carry formal
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
        # CFR dispatch — replaces v18 cascade lines 853-1115
        # ============================================================
        # Phase 158 D-13 single chokepoint integration: CFR routes input
        # mol; backstops (Phase 152 + 156) wrap output name at
        # `Orthonym.name()` line 654. The CFR dispatcher walks
        # DISPATCH_TABLE in priority order; the first predicate that
        # matches wins; the result's handler is invoked here.
        #
        # The audit § 1 DISPATCH_TABLE has 17 outer-cascade rows + GENERAL
        # catch-all = 18 entries. Several handler shims (ANION_SMALL,
        # POLY_ANION, MULTI_COMPONENT_NEUTRAL, DECOMPOSITION_PRE_GENERAL)
        # may return None to signal "no match — continue cascade", in
        # which case we re-dispatch with the matched class excluded so the
        # next priority entry runs. This mirrors the v18 cascade's
        # fall-through behavior (e.g., namer.py:913-914 + 955-956).
        #
        # `_skip_decomposition` is threaded via kwargs per RL-7 option (b);
        # `style` is threaded via kwargs so the RETAINED_NAME and
        # AMINO_ACID predicates can read it without binding `self`.
        result = self._cfr_router.dispatch(
            mol, smiles, canonical_smiles,
            _skip_decomposition=self._skip_decomposition,
            _style=self.style,
        )
        # v29 Phase 4: a class already tried and REJECTED BY A GATE on this
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
            )

        # Cascade-continuation when a handler returns None (audit § 1 row
        # notes for ANION_SMALL / POLY_ANION / MULTI_COMPONENT_NEUTRAL /
        # DECOMPOSITION_PRE_GENERAL). The v18 cascade falls through to the
        # next sibling; mirror that here by re-running dispatch with
        # progressively higher priority floors.
        if name is None and result.class_id != StoutClass.GENERAL:
            from .routing.dispatch_table import DISPATCH_TABLE
            current_priority = DISPATCH_TABLE[result.class_id].priority
            for entry in sorted(DISPATCH_TABLE.values(), key=lambda e: e.priority):
                if entry.priority <= current_priority:
                    continue
                if entry.class_id in self._excluded_dispatch_classes:
                    continue  # v29 Phase 4: already gate-rejected on this molecule
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
                )
                result = ClassDispatchResult(
                    class_id=entry.class_id,
                    handler=entry.handler,
                    audit_record={},
                    tier=entry.tier,
                )
                if name is not None or entry.class_id == StoutClass.GENERAL:
                    break
        # v29 Phase 4: whichever class actually produced this name, so a gate
        # rejection in name() can exclude it and re-enter the cascade.
        self._last_dispatch_class = result.class_id

        # ============================================================
        # GENERAL pipeline fallback — Task 158-02-01 design choice (a)
        # ============================================================
        # The GENERAL handler shim returns None to signal that
        # `_name_impl` runs the legacy `_perceive -> _classify ->
        # assemble_name` pipeline inline. This avoids coupling
        # `routing/` to `composer.py` (D-19 boundary). The check below
        # mirrors the v18 cascade's GENERAL pipeline at namer.py:1124-1131.
        if name is None and result.class_id == StoutClass.GENERAL:
            features = self._perceive(mol, smiles, canonical_smiles)
            self._classify(features)
            assembled = assemble_name(features, style=self.style)
            name = assembled

            # Phase 177 WSB-01 (D-04/D-05): thread the authoritative chain
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
                    CandidateName, store_confidence, retrieve_confidence,
                )
                _existing = retrieve_confidence()
                # Only publish when no richer candidate was already stored for
                # this call (e.g. a ring/Tier-A candidate) — never clobber it.
                if not _existing.get('name'):
                    # v29 C4: this candidate exists ONLY to publish
                    # atom_to_locant to the stereo backstop -- nothing scored
                    # it, and it carries no factors. It previously claimed
                    # confidence=1.0, which (a) is a fabricated pass and
                    # (b) fed the D-01 coverage gate below a handler that is
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
            # v25 G1: general-engine inline fallback (OPT-IN, default OFF).
            # Fires ONLY when the legacy GENERAL pipeline abstained, so the
            # default PIN path stays byte-identical (v25 decision 3). An
            # engine emission still re-enters the SAME downstream moat:
            # the >15-HA coverage gate, the P10 source-vetoes, and the
            # SELF-01 OPSIN-RT gate.
            # ============================================================
            # v33: prefer a retained fused-heterocycle whole-molecule name (e.g.
            # the `...9H-purin-6-amine` parent of an acyl-CoA) over the general
            # engine's von-Baeyer polyene, WHEN it round-trips. A giant
            # purine-containing molecule is not itself a bare fused ring, so the
            # normal dispatch never routes it to `name_fused_heterocycle` -- it
            # falls straight to the von-Baeyer fallback below and loses the
            # PIN-quality retained parent (`...tetraazabicyclo[4.3.0]...`). Try the
            # fused name first; adopt it ONLY when its OPSIN round-trip verifies,
            # so a molecule whose fused form is not yet RT-exact still falls
            # through to the (valid) von-Baeyer name -- never a regression, 0-wrong
            # via the RT probe + the downstream SELF-01 gate. Best-effort only
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
                    # Phase 1 B4: shared best-effort certification gate (E1 +
                    # structural binding-spine axes), replacing the E1-only
                    # check so the inline G1 lane cannot ship a name that
                    # re-fragments a ring or swaps a stereo feature.
                    if _eng is not None and certify_general_result(
                            mol, _eng,
                            allow_charged=self._allow_aromatic_general,
                            structural_only=True):
                        # v29 P1 (audit-only): record the certified spine here,
                        # at the inline site, for re-assertion at the exit of
                        # name(). Everything between this point and the return
                        # -- the stereo backstop, the grammar repair backstop
                        # and the OPSIN validity gate -- may rewrite the string
                        # the certificate was computed on.
                        self._record_binding_proof(
                            mol, _eng, stage="general_engine")
                        # v26 P7 FIX 1: complete-tier abstain-without-Java
                        # contract. E1 does NOT verify ring numbering/locants,
                        # and the downstream _final_opsin_validity_gate FAILS
                        # OPEN with no OPSIN jar -- so under `complete`
                        # (allow_aromatic_general) accepting this E1-only name
                        # with no jar would ship an UNVERIFIED name from the NEW
                        # aggressive producers (P1/P2/P5). Mirror the
                        # late-recovery jar guard: keep the abstention. Scoped to
                        # allow_aromatic_general so valid/best-effort/pre-v26
                        # behavior is unchanged. (best-effort opts in via
                        # _general_fallback_unverified; test/internal mode via
                        # _disable_opsin_validity_gate.)
                        _no_jar_abstain = (
                            self._allow_aromatic_general
                            and not self._general_fallback_unverified
                            and not self._disable_opsin_validity_gate
                            and not _validity_gate_jar_present()
                        )
                        # v26 P7 FIX 2: fail closed on DROPPED stereo. SELF-01 is
                        # constitutional (stereo-blind) and the universal stereo
                        # backstop is LOG-ONLY for the handler='unknown' cohort
                        # general emissions arrive as -- so a general name that
                        # omits E/Z or R/S the input carries would otherwise
                        # ship a WRONG stereoisomer. Expressed stereo (parent
                        # _stereo_prefix / substituent-internal name_substituent)
                        # makes the predicate False -> no over-abstention.
                        # v27 P6 T6.2: shared stereo-emit policy (same helper as
                        # the late-recovery site). complete/valid abstain on
                        # dropped stereo; best-effort ships a constitution-only
                        # name flagged stereo_unexpressed.
                        #
                        # v33 no-abstain Phase A fix-round-1, Finding 4 (HIGH,
                        # FABLE adversarial review): the OLD comment here claimed
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
                        # ():
                        # `C[C@@H](O)[C@@H](N)C` + candidate
                        # `(3R)-3-aminobutan-2-ol` -- skeleton matches, full
                        # InChIKey differs, and this lane shipped it with no
                        # check at all. Route a FLAGGED emission through the
                        # SAME `_rt_match` superset gate the late-recovery site
                        # (`_try_general_engine_recovery`) already uses for this
                        # exact purpose -- proven (FABLE's Q1,
                        # , 13/14 probed
                        # cases) to ACCEPT a genuine omission while REJECTING a
                        # conflict/fabrication/wrong-enantiomer. An unflagged
                        # emission (`_stereo_flagged=False`, already fully
                        # stereo-complete) is UNCHANGED -- this only tightens the
                        # flagged (best-effort-only) case. (This gate is
                        # jar-PRESENT-only; the jar-ABSENT best-effort case no
                        # longer fail-opens -- see the task-JAR-ABSENT
                        # verify_or_none gate just before the emit below, which
                        # closed the FABLE 0-wrong hole here.)
                        _permitted, _stereo_flagged = self._stereo_emit_decision(
                            mol, _eng.name)
                        if (_permitted and _stereo_flagged
                                and not self._disable_opsin_validity_gate
                                and _validity_gate_jar_present()):
                            _g1_opsin_smi = _validity_gate_name_to_smiles(
                                _eng.name)
                            if (_g1_opsin_smi is None
                                    or not self._rt_match(
                                        smiles, _g1_opsin_smi, True)):
                                _permitted = False
                        # task-JAR-ABSENT (FABLE 0-wrong hole, inline-G1
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
                            name = _eng.name
                            # v25 G3: observation-only provenance (the emission
                            # still flows through the normal downstream gates).
                            from .metrics.provenance import (
                                record_source, record_stereo_unexpressed)
                            record_source("general_engine")
                            record_stereo_unexpressed(_stereo_flagged)
                except Exception as _e:  # fail-closed: an engine bug must
                    # never turn an abstention into a crash or a wrong name.
                    logger.info("general engine error (kept abstention): %s",
                                _e)

        # ============================================================
        # Post-dispatch quality gates — preserved VERBATIM from v18
        # lines 1129-1207. CFR is the dispatch substrate; these gates
        # run on the GENERAL pipeline's assembled name AFTER dispatch,
        # NOT on outputs from class-specific handlers.
        #
        # [Rule 1 - Bug] Found during Task 158-02 verification: gating
        # on ANY non-None name (the original implementation) caused
        # PEPTIDE / SALT / etc. handler outputs to be re-checked
        # against the GENERAL pipeline's confidence store, which is
        # populated by `assemble_name()` and may still hold stale data
        # from the PREVIOUS call (the `_confidence_store` is thread-
        # local and persists across `Orthonym.name()` invocations; see
        # `assembly/coverage_scoring.py:_confidence_store`). Pre-CFR
        # the gate only ran on `assembled = assemble_name(...)` output
        # at the very end of the cascade — handler early-returns
        # (e.g. peptide / salt / amino_acid) bypassed it entirely.
        # Restrict the gate to GENERAL-pipeline outputs to preserve
        # CFR-04 byte-identical canary. Honest-fail-on-data per
        # CONTEXT D-29: the fix is upstream (gate scope), NOT a
        # threshold relaxation or postprocessor band-aid.
        # ============================================================
        # v29 C4 -- the `> 15` size term was REMOVED from this entry test, so
        # small molecules are no longer excluded from the quality gates
        # wholesale. Previously a 9-heavy-atom molecule could not reach any
        # gate at all: the reference defect CC(C)(C)OOCCO -> 'ethan-1-ol'
        # (six atoms silently dropped) was excluded here, several lines
        # BEFORE the `conf_handler != 'unknown'` check people assumed was
        # responsible.
        #
        # MEASURED before changing ( widen probe, the
        # COMPLETE affected population -- all 714 rows of
        #  with heavy_atoms <= 15, 0 timeouts):
        #   593 molecules are newly admitted (class GENERAL, HA <= 15)
        #   garbled token         : 0 fire
        #   len(name) < 6         : 0 fire
        #   confidence < 0.30     : 0 fire
        #   atom_coverage < 0.55  : 0 fire
        #   => 0 emitted names change. Confirmed by a full A/B naming run
        #      over the same 714 rows (0 diffs).
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
            assembled = name  # local alias for v18-byte-identical body
            _heavy = mol.GetNumHeavyAtoms()
            _GARBLED_TOKENS = ('cycloane', 'anedicarboxamide', 'aneyl')
            assembled_lower = assembled.lower()
            is_garbled = any(tok in assembled_lower for tok in _GARBLED_TOKENS)

            # Also detect stub-only names: when the parent text is empty,
            # the assembly may produce just a bare suffix like "ane" or "ol".
            # A name shorter than 6 chars for a 15+ atom molecule is garbled.
            #
            # v29 C4: the `_heavy > 15` term is retained HERE deliberately. The
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
                # v25 P0 Task 0.1: the downgrade machinery engaged — the
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
            # assemble_name() -- handler='unknown' means no scoring happened
            # (e.g., decomposition path, retained names, etc.).
            if (conf_score is not None
                    and conf_handler != 'unknown'
                    and conf_score < _TRUNCATION_CONFIDENCE_THRESHOLD):
                # v25 P0 Task 0.1: downgrade machinery engaged (P2.1 bucket).
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

            # D-01: atom_coverage secondary gate -- catches quality-gate false
            # positives where retained ring names inflate character-based
            # confidence but atom coverage reveals only partial molecule
            # description.
            #
            # v29 C4 -- READ BEFORE TRUSTING THIS GATE. Two facts:
            #
            # (1) `atom_cov` here is NOT a coverage measurement. No production
            #     caller passes parent_atom_indices into compute_confidence
            #     (candidate_pool.add()'s "Risk 1" byte-identical mitigation),
            #     so this value is the name-LENGTH proxy -- numerically equal
            #     to factors['ratio'] -- or a retained-name 1.0. Its
            #     provenance is published as conf_data['coverage_provenance'].
            #     The gate is retained AS IS (removing it would un-suppress
            #     names it currently rejects, which is the unsafe direction),
            #     but it must not be described as verified coverage, and it
            #     cannot be relied on to catch atom drops: the reference
            #     defect CC(C)(C)OOCCO -> 'ethan-1-ol' scores 10/9/1.5 = 0.74
            #     on this proxy and so passes, while its REAL coverage is
            #     3/9 = 0.33. A sound gate needs producer-side atom records.
            #
            # (2) the missing-measurement default used to be 1.0, i.e. "no
            #     measurement" read as "perfect coverage" -- fail-open. It is
            #     LIVE, not latent: the chain-parent path above publishes a
            #     candidate with handler='chain' and an EMPTY factors dict,
            #     which clears both guards on this branch and then defaults to
            #     1.0. Missing is now None and handled explicitly. The gate
            #     abstains on None (same control flow as the old 1.0, so no
            #     emitted name changes) because it has nothing sound to gate
            #     on -- an abstention, not a pass. Making it fail CLOSED here
            #     would be a suppression change and needs measuring first.
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
                        # v25 P0 Task 0.1: downgrade machinery engaged
                        # (P2.1 bucket).
                        from .metrics.abstention import (
                            AbstentionCode, record_suppression,
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
        # CRITICAL, mandatory regardless of which other P-9 tasks land).
        # ============================================================
        # A covalent bond between a TRUE metal (Groups 1/2/3-12 -- see
        # has_covalent_metal_carbon_bond) and a carbon atom means the
        # compound IS a P-69 organometallic. If the ORGANOMETALLIC handler
        # did not produce the winning name (result.class_id != ORGANOMETALLIC),
        # the CFR cascade fell through to a class that has NO legitimate way
        # to name a true-metal-carbon compound -- it can only have "won" by
        # silently naming a DECOMPOSED SUB-FRAGMENT that drops the metal
        # (`C[Ti](Cl)(Cl)Cl` -> 'methane'; `Cl[Pt]1(Cl)...` -> '...ole', both
        # confirmed structure-loss leaks in the no-Java gate-off path, where
        # the OPSIN validity/SELF-01 gate fails OPEN). Source-level veto: does
        # NOT depend on OPSIN/Java. Never fires for a dot-separated IONIC
        # metallocene/salt (has_covalent_metal_carbon_bond is False -- no bond
        # crosses fragments) or for Group 13-16 metalloids (excluded from the
        # predicate's scope; MONONUCLEAR_HYDRIDE / WSD-04 heterocycle defer /
        # etc. are legitimate alternate paths for those, not leaks).
        if name and result.class_id != StoutClass.ORGANOMETALLIC:
            from .perception.metals import (
                has_covalent_metal_carbon_bond, has_metal_coordination_bond,
            )
            # v38: extend the veto to DATIVE-bonded coordination compounds
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
        # W8-P4: oxoacid/anhydride atom-drop safety floor (P-65.7/P-67).
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
        # path gets the same protection the OPSIN SELF-01 gate gives the
        # Java path (which fails OPEN without a JAR). See
        # perception/structure_conservation.py for the full design record
        # and the false-positive investigation behind each check's scope.
        # ============================================================
        # Scoped to TOP-LEVEL naming only (never a recursive substituent/
        # fragment naming call -- e.g. name_fragment_recursively() names an
        # isolated charged fragment such as the boron-anion piece of
        # "trimethylboryl" as a standalone molecule up the call stack; that
        # inner name is not the final shipped name and must not be
        # suppressed here, or the fragment's "unknown organic compound"
        # replacement gets spliced into the middle of the parent name).
        from .assembly.fragment_naming import is_top_level_naming
        if name and not is_failure_name(name) and is_top_level_naming():
            from .perception.structure_conservation import (
                charge_dropped, partial_sat_sp3_substituent_drop,
                fused_ring_atom_drop,
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
            # v25 G5-A: Java-free valence guard — a von-Baeyer name that cites a
            # ring carbon as BOTH a double-bond terminus AND an oxo/-one carbon
            # is a 5-bond carbon (the caffeine oxo/ene class). SELF-01 catches it
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

        # Orthonym is deterministic-rules-only (ADR-21-01, v21): there is no
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
        # WSB-03 (D-13, Pitfall 2): route the SECOND CIP chokepoint through the
        # single source-of-truth assign_stereochemistry() so this path can never
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
        # Phase 173.6 T3: scoped principal-group override (P-72/P-74). When the
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

        # Phase 168 D-08: stash the controller flag + oracle onto features (the
        # features._ring_info transient-attribute pattern at :1532) so CandidatePool.add()
        # can lift them onto the pool (BLOCKER #9 fix). WARNING #9: the oracle is required
        # for the TRIV-03 T2 runtime RT-safety gate. Both default to OFF/None (Stage A).
        features._enable_triviality_controller = self._enable_triviality_controller
        features._triv_oracle = self._triv_oracle
        # Phase 169 D-05: plumb the group-splitting flag + flag-ON-only oracle onto
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

        # Parent selection for ALL cyclic molecules (Phase 148: no fused-heterocycle bypass).
        # P-44.1 cascade runs whenever a meaningful chain exists; cascade itself
        # enforces P-31.1.3.4 NP override (parent_selection.py:688-700) and
        # P-52.2.8 ring-on-tie tiebreaker (parent_selection.py:862-869).
        # Source: https://iupac.qmul.ac.uk/BlueBook/P4.html  P-44.1
        # Source: https://iupac.qmul.ac.uk/BlueBook/P5.html  P-52.2.8
        # Source: HERITAGE 1990 §4 (full seniority cascade on ALL structures).
        if features.is_cyclic:
            from .rules.parent_selection import select_parent

            # Get ring atoms to exclude when finding chain
            all_ring_atoms = set()
            for ring in features.ring_systems:
                all_ring_atoms.update(ring)

            # Find potential principal chain (excluding ring atoms)
            # IMPORTANT: namer.py does the chain finding, then passes result to select_parent()
            potential_chain = find_principal_chain(
                features.mol,
                features.functional_groups,
                features.principal_group,
                exclude_atoms=all_ring_atoms
            )

            # Only do parent selection if we found a meaningful chain (>= 2
            # carbons) OR (v22 C-T2 / V-4, P-44.1.1) a 1-carbon chain whose
            # principal-characteristic-group carbon is NOT ring-attached: the
            # ring then cannot express the PCG as a ring suffix, so the 1-carbon
            # parent (formamide / formic acid / methanal) must be parent-eligible
            # (else an N-aryl formamide falls to the ring -> 'carbamoylbenzene').
            # select_parent() applies the same ring-neighbour test, so a
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
                    # Wave2 T3a: skeletal suffixes (-ol/-amine/-thiol/-imine, ...)
                    # have no exocyclic-carbon form, so a ring-attached single
                    # PCG carbon still cannot put the suffix on the ring.
                    # Parent selection is eligible when the single carbon IS a
                    # PG attachment atom and NO attachment atom is a ring atom
                    # (else the ring expresses the suffix itself — e.g. a
                    # sugar's ring -OH union keeps the ring parent).
                    from .rules.parent_selection import (
                        SKELETAL_SUFFIX_PGS, _pg_attachment_atoms,
                    )
                    if features.principal_group in SKELETAL_SUFFIX_PGS:
                        _att: set = set()
                        for _m in features.principal_group_atoms:
                            _att.update(_pg_attachment_atoms(
                                features.principal_group, _m))
                        if _single in _att and not (_att & all_ring_atoms):
                            _run_parent_selection = True
            if _run_parent_selection:
                # Phase 147 D-03: delegate ring-type dispatch to helper.
                # Replaces the prior inline fused-hetero-only block; new
                # helper covers fused-hetero / PAH / benzene / simple-
                # hetero / spiro-stub / VB-stub / else->None per D-03.
                _ring_info = _build_ring_info_for_parent_selection(features)
                # Phase 147: stash on features so downstream pool.add()
                # call sites can read it without a signature change at
                # every composer.py call site (transient runtime
                # attribute; not a MolecularFeatures dataclass field
                # per D-08; safe because the dataclass is not frozen).
                features._ring_info = _ring_info

                # Pass pre-computed chain to select_parent
                selection = select_parent(
                    mol=features.mol,
                    ring_systems=features.ring_systems,
                    principal_chain=potential_chain,
                    principal_group=features.principal_group,
                    principal_group_atoms=features.principal_group_atoms,
                    ring_info=_ring_info,
                    # v37 SP1.3: 0 for every normal call (byte-identical);
                    # >0 only on a best-effort offer-retry inner instance.
                    _offer_rank=self._forced_parent_rank,
                )
                features.parent_selection_result = selection  # Phase 148 D-02 (V18 Appendix A.5)
                # v37 SP1.3: publish the top-level pool size so the offer-retry
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
                    # is_cyclic stays True -- ring data needed for ring-as-substituent naming (Phase 139 ARCH-01)
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

            # D-03 (Phase 178) chokepoint consolidation: compute the senior ring
            # system ONCE (P-44.2 via select_principal_ring_system) and carry it
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
            from .rules.polycyclics import identify_polycyclic, get_polycyclic_substituents
            pah_name = identify_polycyclic(features.mol)
            if pah_name and len(features.ring_systems) >= 2:
                # WS-A task 9 (A-i): the PAH early-return must not preempt the
                # P-44 parent decision. With >=2 ring systems, take the PAH
                # route ONLY when the PAH system survives:
                #   (a) P-44.1 — a principal characteristic group on a
                #       DIFFERENT ring system makes that system the parent
                #       (2-(naphthalen-2-yl)cyclohexan-1-ol, not naphthalene);
                #   (b) P-44.2 — the among-rings winner must BE the PAH
                #       system (2-(naphthalen-2-yl)furan: furan is senior).
                from .rules.polycyclics import get_polycyclic_core_atoms
                from .rules.parent_selection import is_principal_group_on_ring
                _core = get_polycyclic_core_atoms(features.mol, pah_name)
                _core_set = set(_core) if _core else set()
                if _core_set:
                    # Default False so the P-44.2 among-rings veto below runs
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
                        # D-03: reuse the one authoritative computation
                        _senior = _principal_ring_system
                        # FR-4 / P-66.2.2 build (P-44.1 precedes P-44.2): when
                        # the principal characteristic group sits ON the PAH
                        # core, a senior heterocyclic SUBSTITUENT ring must not
                        # steal the parent (5-(1,3-dioxo-...-isoindol-2-yl)-
                        # naphthalene-1-carboxylic acid). Only the among-rings
                        # tie-break (P-44.2) is skipped; P-44.1 already ran.
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

            # Wave2 T3 (P-44.1 determinism): among monocyclic candidate rings,
            # atom_rings[0] (first SSSR ring) is SMILES-order-dependent. When the
            # principal characteristic group sits on EXACTLY ONE ring, P-44.1 makes
            # that ring senior (it precedes the P-44.2 seniority tiebreak) — so move
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
                # Select the most senior ring system per IUPAC P-44.2
                # and store it for downstream use (e.g., ring-vs-chain
                # comparison in the composer). The monocyclic dispatch
                # path below uses atom_rings[0] which preserves SSSR
                # cyclic traversal order needed by orientation functions.
                # Complex multi-ring (fused/bridged) systems are handled
                # by _classify_complex_ring() in the composer.
                # D-03: reuse the one authoritative computation (see top of block)
                principal = _principal_ring_system
                features.senior_ring_system = principal if principal else atom_rings[0]

                # For multi-ring-system molecules, use a SSSR ring from
                # the senior system as principal_ring when the senior
                # system is strictly LARGER than the default ring's system.
                # This ensures fused/bridged senior systems (imidazopyridine,
                # xanthene) take precedence over small monocyclic rings
                # (benzene) per IUPAC P-44.2.
                #
                # WS-A.1 S2: select_principal_ring_system (P-44.2, with the S1
                # P-44.4.1 unsaturation tiebreak) is now AUTHORITATIVE among ring
                # systems. The legacy ``size_diff >= 3`` heuristic that DISCARDED
                # the P-44.2 winner for equal/near-equal senior systems is deleted
                # (it produced (furan-2-yl)benzene instead of 2-phenylfuran and
                # pyridinylcyclohexane instead of 4-cyclohexylpyridine). Gated only
                # by the two correctness guards below; verified on the among-rings
                # PIN gold corpus ().
                if principal and len(features.ring_systems) >= 2:
                    senior_set = set(principal)
                    # Find the ring system that contains the default ring
                    default_system = set(atom_rings[0])
                    for rs in features.ring_systems:
                        if set(atom_rings[0]).issubset(rs):
                            default_system = rs
                            break

                    # Guard (a) P-44.1: the principal characteristic group must be
                    # IN the parent. If the PG sits on the default ring system,
                    # that ring stays parent regardless of P-44.2 ring seniority.
                    from .rules.parent_selection import is_principal_group_on_ring
                    pg_on_default = False
                    pg_on_senior = False
                    if features.principal_group_atoms:
                        pg_on_default = is_principal_group_on_ring(
                            features.mol, default_system,
                            features.principal_group_atoms,
                            features.principal_group,
                        )
                        # C4 (P-44.1 + P-44.2): a BRIDGING characteristic group
                        # (e.g. a secondary amine N bonded to a carbon of EACH
                        # ring — diaryl/aryl-heteroaryl amines) satisfies P-44.1
                        # for BOTH ring systems. When it also sits on the SENIOR
                        # ring, P-44.1 no longer disqualifies the senior ring, so
                        # the P-44.2 seniority tiebreak decides -> the senior ring
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
                    # 171 P-1 trial). Among-ring seniority does not decide a chain
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
                        # Wave-2 C2 (P-45.2.1): when the PG sits on TWO OR
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
                                is_benzene_ring as _ibr,
                            )
                            if all(_ibr(features.mol, r) for r in atom_rings):
                                _det_ring = _sbpr(features.mol)
                        features.principal_ring = (
                            _det_ring if _det_ring is not None
                            else atom_rings[0]
                        )
                else:
                    features.principal_ring = atom_rings[0]
                features.ring_type = classify_ring(features.mol, features.principal_ring)

                # P-59.2.1.6 (BB 25207): the principal ketone-family group sits
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

                # P-22.1.2(b) / P-25.3.2.1.1: a mancude monocyclic hydrocarbon
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
                from .rules.benzene import is_benzene_ring, get_benzene_substituents
                if is_benzene_ring(features.mol, features.principal_ring):
                    features.is_benzene = True
                    # v22 C-T2 (V-3): with >1 benzene ring and no principal
                    # characteristic group (aralkyl/diaryl ethers, e.g. benzyl
                    # phenyl ether), `atom_rings[0]` is SMILES-order-dependent, so
                    # the parent flipped with the input spelling -> non-determinism
                    # ('(phenoxymethyl)benzene' vs 'benzoxybenzene'). Pick the
                    # parent ring deterministically (carbon-linked preference +
                    # canonical rank). Single-benzene and PCG-bearing rings are
                    # left to the existing principal-ring selection.
                    from .rules.benzene import (
                        _select_benzene_parent_ring, _preferred_benzene_parent_ring,
                    )
                    _benzene_rings = [
                        r for r in features.mol.GetRingInfo().AtomRings()
                        if is_benzene_ring(features.mol, r)
                    ]
                    if len(_benzene_rings) > 1 and not features.principal_group:
                        # P-45.5.1 / P-45.6.3 (diaryl-linked-by-heteroatom):
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
                        orient_heterocycle_with_substituents,
                        get_heterocycle_substituents,
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

                    # WS-4 / ring-suffix fix (DEF-6): identify the ring atoms bearing the
                    # principal characteristic group so the suffix gets the lowest
                    # locant (P-14.4(c)), after the heteroatom. Same extraction as
                    # the cycloalkene branch: a ring C whose FG-match heteroatom is
                    # exocyclic (C=O ketone, C-OH alcohol/phenol, C-NH2 amine, ...).
                    pg_ring_atoms = set()
                    if features.principal_group:
                        from .rules.seniority import get_prefix as _pg_get_prefix
                        _pg_prefix = _pg_get_prefix(features.principal_group)
                        # The principal characteristic group is a CLASS, not a
                        # single FG label. A ring bearing an exocyclic primary
                        # alcohol (-CH2OH) plus secondary ring alcohols (ring -OH)
                        # has principal_group == primary_alcohol, but the RING
                        # carbons that carry the -ol SUFFIX are the secondary ones
                        # (the assembly already unions them via the shared 'hydroxy'
                        # prefix). Collect every same-prefix FG class so orient
                        # gives the suffix-bearing ring carbons the lowest locants
                        # (P-14.4(c)) DETERMINISTICALLY (SEN-03 alcohol-class union);
                        # without this the suffix-locant tie is broken by SMILES
                        # atom order -> non-deterministic numbering (CARB-01).
                        for _fg, _matches in features.functional_groups.items():
                            if _pg_prefix is None or _pg_get_prefix(_fg) != _pg_prefix:
                                continue
                            for match in _matches:
                                match_set = set(match)
                                if not (match_set & ring_set):
                                    # CASE A (twin of the cycloalkane branch
                                    # below, ~3627): a WHOLLY-EXOCYCLIC appended
                                    # suffix (-carboxylic acid, -carbaldehyde,
                                    # -carbonitrile, -carboxamide, ...). The
                                    # entire FG match lies OFF the ring, so the
                                    # CASE-B ring-C-bonded-heteroatom test never
                                    # fires and pg_ring_atoms stayed empty --
                                    # the ring-direction tie for a symmetric-
                                    # heteroatom ring (furan/thiophene/pyrrole)
                                    # was then broken arbitrarily by canonical
                                    # rank, giving the suffix the HIGHER locant
                                    # (2-methylfuran-5-carboxylic acid). The
                                    # expressed-suffix ANCHOR is the ring atom
                                    # bonded to the match carbon; per P-14.4(c)
                                    # it takes the lowest locant BEFORE the
                                    # detachable prefix (5-methylfuran-2-
                                    # carboxylic acid). Same-prefix (SEN-03)
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
                                    # SEN-03 same-prefix union also admits an
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

                    # Orient considering heteroatoms, the principal group, then
                    # other substituents for lowest locants (P-14.4 order).
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
                        get_ring_substituents, get_ring_double_bonds,
                        orient_cycloalkane, orient_cycloalkene
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
                    # IUPAC P-31.1.3.4 / P-31.1.4: principal group gets lowest locant
                    # We identify ring C atoms that directly bear the principal
                    # group's characteristic heteroatom (e.g., C=O for ketone,
                    # C-OH for alcohol). The characteristic heteroatom must be
                    # bonded DIRECTLY to a ring carbon (not via an exocyclic C).
                    pg_ring_atoms = set()
                    if features.principal_group and features.principal_group in features.functional_groups:
                        ring_set = set(features.principal_ring)
                        for match in features.functional_groups[features.principal_group]:
                            match_set = set(match)
                            if not (match_set & ring_set):
                                # v21 WS-A.1 S4: wholly-exocyclic match = an
                                # APPENDED suffix (-carbaldehyde, -carboxylic
                                # acid, -carbonitrile). Its expressed-suffix
                                # ANCHOR (the ring atom bonded to the match
                                # carbon) takes the lowest locant per
                                # P-31.1.4.2.4 (2-methylcyclohexane-1-
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
                        SKELETAL_SUFFIX_PGS, _pg_attachment_atoms,
                    )
                    for match in features.functional_groups[features.principal_group]:
                        if features.principal_group in SKELETAL_SUFFIX_PGS:
                            # WS-A task 9: the -one family's locant atom is
                            # the carbonyl carbon (PG_ATTACHMENT_INDICES);
                            # feeding the whole match (incl. the FLANKING
                            # carbon) made orientation criterion (a) tie at
                            # {1,2} both ways for 2-carbon ketone chains and
                            # fall through to alphabetics ('...ethan-2-one').
                            pg_atom_set.update(
                                _pg_attachment_atoms(
                                    features.principal_group, match
                                )
                            )
                        else:
                            pg_atom_set.update(match)

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

        # v30 ATOM-PARTITION invariant: a principal-group occurrence is a PARENT SUFFIX, not a
        # substituent's group. get_principal_group() (:4123) gathers every same-prefix-class
        # occurrence molecule-wide (the SEN-03 alcohol-class union), so the tertiary OH of a
        # C(C)(C)O arm ON A RING is otherwise emitted as a second ring -ol, double-assigning its
        # atom (it is ALSO in ring_substituents) -> an atom-short, wrong parent
        # (cyclohexane-1,2-diol for a mono-ol ring). decide suffix-vs-
        # prefix by atom<->parent membership; the off-parent group's atoms
        # already flow to the
        # substituent enumerator and are named there.
        # .
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
        # ⚠ Excludes amines/imine: N-suffixes carry special multi-N locant handling (P-62,
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

        # INST-05: Naming decision trace
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
        for bond in mol.GetBonds():
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
        for bond in mol.GetBonds():
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

        This is used for orient_chain() which expects substituent_positions
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
        functional_groups: Dict from detect_functional_groups().

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
    """Module-level convenience wrapper for Orthonym(style).name_with_tree(smiles).

    CR-04 part A (BLOCKER): downstream consumers expect
    ``from orthonym import name_with_tree`` to work analogously to
    ``name_compound``.

    Args:
        smiles: SMILES string to name.
        style: 'pin' (preferred) | 'systematic' | 'cas'. Default 'pin'.

    Returns:
        NamingResult(name, tree, atom_to_locant_hint).
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
        binding_proof: v29 P1 proof mode forwarded to ``Orthonym`` -- see
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
        {'name': 'ethanol', 'confidence': 1.0, ...}
    """
    # v25 G3: inherit the engine flag from the propagation ctx when the
    # caller didn't say — fragment/component recursion re-enters through
    # here with a FRESH namer, and the top-level opt-in must carry through.
    if general_fallback is None:
        try:
            from .metrics.provenance import general_fallback_ctx
            general_fallback = general_fallback_ctx.get()
        except Exception:
            general_fallback = False
    # v37 SP1.1b: inherit the REST of the best-effort tier the same way. Before
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
    # v37 SP5.4: inherit the full-coverage opt-in through the recursive
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
            # v29 C4: naming raised, so nothing measured anything. Use the one
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
                return _descriptive_fallback(smiles)
            return result
        # If name() returned empty/None, generate descriptive fallback
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
    try:
        namer = Orthonym(style=style)
        return namer._name_impl(smiles, _skip_decomposition=True)
    except OrthonymLimitError:
        # G0 (DD7 S1, WR-03): a fragment-level fail-closed refusal is "no name",
        # not a crash — propagate as None so the decomposition engine treats the
        # fragment as out-of-scope and tries another strategy. Caught explicitly
        # (before the broad except) so the intent is documented and a genuine
        # fragment bug is not silently conflated with a legitimate refusal.
        return None
    except Exception:
        return None


# Metals/inorganic elements and the metal-name map now live in orthonym.errors
# (single source of truth) and are re-exported via the module-top import so
# data.cation_words keeps importing them from orthonym.namer.
# _ORGANIC_ELEMENTS / _METAL_NAMES are bound at module top.


def _coordination_retained_name(smiles: str) -> Optional[str]:
    """v36 Milestone D1: exact-InChIKey retained name for an N-coordinated metal
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
        return COORDINATION_RETAINED.get(Chem.MolToInchiKey(mol))
    except Exception:  # pragma: no cover - a lookup must never break naming
        return None


def _descriptive_fallback(smiles: str) -> str:
    """Generate a descriptive fallback message instead of bare 'unknown'.

    For inorganic/metallic compounds, returns a descriptive message like
    'gold compound (not supported)'. For organic molecules that failed
    naming, returns 'unknown organic compound'.

    HYG-02 (Phase 173): this now delegates to ``errors.classify_failure_limit``
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
    # v36 D1: before emitting the honest metal sentinel, consult the exact-InChIKey
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
    """Diagnostic: return the OrthonymLimitError for an out-of-scope input, else None.

    Lets a caller probe "can Orthonym handle this?" without triggering an
    exception. Returns a scope limit for structurally-refused inputs (wildcards);
    otherwise runs the default namer and, if it fails to produce a real name,
    returns the failure-mapped limit. Returns None for in-scope inputs and for
    unparseable SMILES (which are a parse error, not a scope limit).
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
    try:
        produced = Orthonym().name(smiles)
    except Exception:
        produced = None
    if is_failure_name(produced):
        return classify_failure_limit(mol, smiles=smiles)
    return None
