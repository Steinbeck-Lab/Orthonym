"""
Data module - Lookup tables and reference data.

Contains hand-curated data (primary) merged with auto-imported OPSIN data.
Hand-curated entries always take precedence over OPSIN imports on conflict.
"""

import logging
import os
from typing import Dict, Optional

from .retained_names import RETAINED_NAMES as _HAND_CURATED_NAMES
from .retained_names import get_retained_name as _hand_curated_get

logger = logging.getLogger(__name__)

# Try to import OPSIN data (may not exist if import script hasn't run)
try:
    from .opsin_imports import OPSIN_RETAINED_NAMES as _OPSIN_NAMES_RAW
except ImportError:
    _OPSIN_NAMES_RAW = {}
    logger.debug("OPSIN imports not available (run scripts/import_opsin_xml.py)")


# --- Stem filtering ---
# OPSIN stores many names as stems (e.g., "norbornen" instead of "norbornene",
# "norborn" instead of "norbornane"). These stems are OPSIN parser internals
# meant to have suffixes appended. Using them directly as retained names causes
# wrong output. Filter to only include names that look like complete words.

# Suffixes that indicate a complete chemical name
_COMPLETE_NAME_ENDINGS = (
    # Hydrocarbons
    "ane", "ene", "yne", "ylene", "adiene", "atriene",
    # Heterocycles and rings (complete forms)
    "ole", "ine", "oline", "azine", "azole",
    "idine", "idine",
    "furan", "pyran", "thiophene", "pyrrole", "pyridine",
    "oxetane", "azetidine", "thiirane", "oxirane", "aziridine",
    # Saturated heterocycles (complete -an forms that are NOT stems)
    "acridan", "indan", "thiinan", "dithian",
    # Functional groups
    "ol", "al", "one", "ide", "ate", "ite",
    "amine", "amide", "nitrile", "oxide", "ether",
    # Cations / ions
    "ium", "ion",
    # Acids
    "acid",
    # Sugars
    "ose",
    # Steroids / NPs
    "sterol", "sterone", "stenone", "stadiene",
    # Natural products with complete names
    "curcumin", "capsaicin", "caffeine", "morphine", "codeine",
    "nicotine", "quinine", "strychnine", "colchicine",
    "resveratrol", "melatonin", "serotonin", "dopamine",
    # Generic complete name endings
    "iol", "diol", "triol",
    # Aromatic retained names
    "benzene", "toluene", "xylene", "styrene",
    "naphthalene", "anthracene", "phenanthrene",
    "indole", "purine", "pteridine", "quinoline", "isoquinoline",
    "catechol", "resorcinol", "hydroquinone",
    # Amino acids
    "glycine", "alanine", "valine", "leucine",
    "proline", "serine", "threonine", "cysteine", "tyrosine",
    "histidine", "tryptophan", "phenylalanine",
    # Nucleobases
    "guanine", "adenine", "cytosine", "thymine", "uracil",
    # Steroids
    "cholesterol", "cortisone", "testosterone", "estradiol",
    # Multi-word endings
    "anhydride",
    # Phenols and specific compound class endings
    "phenol",
    # Thiin family (complete names, not stems)
    "thiin", "oxathiin", "dithiin",
    # Specific complete retained names that don't fit patterns
    "urea", "thiourea",
    # a phase RESEARCH section 4.2: complete-form endings produced by
    # _STEM_TO_COMPLETE_SUFFIX in scripts/import_opsin_xml.py
    "ocene", "hrene",
    # a phase Plan 02 deviation (Signal 1 extension per internal notes
    # "Extended with additional complete-form endings discovered during
    # round-trip validation"): additional IUPAC retained-name endings
    # surfaced by the validator's pass_smiles set. Each entry below
    # corresponds to a family of names that round-trip cleanly via OPSIN
    # (Signal 3) and is therefore an empirically-verified complete-form
    # retained name. The (S1 OR S2) AND S3 promotion rule still applies;
    # extending S1 here is a refinement, not a relaxation.
    "illin",     # vanillin, isovanillin, homovanillin, o-vanillin family retained)
    "genin",     # saligenin, naringenin, apigenin, rhapontigenin (flavonoid retained)
    "pterin",    # pterin, methanopterin, dihydropterin, tetrahydropterin natural product)
    # NOTE: 'oform' / 'fluoroform' / 'iodoform' / 'bromoform' deliberately NOT
    # added — `chloroform` is already in HC (HC line ~), and adding the OPSIN
    # entries for FC(F)F / IC(I)I / BrC(Br)Br causes the substituent renderer
    # to emit `(fluoroform-yl)` instead of `(trifluoromethyl)` for compounds
    # like FC(F)CC. a phase Plan 02 leaves these to systematic naming;
    # a phase substituent-pipeline work owns the trihalomethane substituent
    # form. internal notes (out-of-scope guard for substituent pipeline).
    # NOTE: 'thranil' / 'anthranil' deliberately NOT added — anthranil is OPSIN
    # data for c1ccc2nocc2c1, which a phase byte-identical lock requires
    # to render as '1,2-benzisoxazole'. Adding the ending would violate the lock.
    "phthalid",  # phthalid, isophthalid retained lactone)
)


def _is_complete_name(name: str) -> bool:
    """Check if an OPSIN name is a complete chemical name (not a stem).

    OPSIN stores many entries as stems (e.g., 'norbornen', 'stilben', 'corrin')
    that need suffix appending by OPSIN's parser. These should not be used
    directly as retained names.

    Returns True if the name appears to be a complete, usable name.
    """
    name_lower = name.lower().rstrip()

    # Names containing spaces are typically complete (e.g., "acetic acid")
    if " " in name_lower:
        return True

    # Names with locant prefixes and hyphens often indicate complete names
    # but stems can also have locants (e.g., "isoflavan-3-en")

    # Check against known complete endings
    return any(name_lower.endswith(ending) for ending in _COMPLETE_NAME_ENDINGS)


# a phase: load IUPAC 2013 PIN allow-list (single source of truth).
# Single canonical store: src/orthonym/data/iupac_2013_pin_list.json.
# Replaces the previous hard-coded _OPSIN_NON_PIN_EXCLUSIONS frozenset
# (promotion to JSON allow-list with citation per entry).
import json
from pathlib import Path

_PIN_LIST_PATH = Path(__file__).parent / "iupac_2013_pin_list.json"
# a phase REVIEW: standardise on a broad except clause for both
# JSON loaders. Previously the PIN list loader caught only
# FileNotFoundError, while the round-trip cache loader caught
# (json.JSONDecodeError, KeyError) -- same JSON, two different failure
# modes for the same kind of corruption (and TypeError from
# frozenset-of-non-hashables was uncaught in either path). Standardise
# on (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) so
# any malformed file degrades gracefully rather than crashing import.
# Source: internal notes.
# Task E: the loader moved to data/pin_policy.py so the amino-acid,
# natural-product and trivial-acid surfaces read the SAME adjudicated sets
# instead of not reading them at all. Values are unchanged; the try/except
# degradation (empty sets + warning on a malformed file) now lives there.
from .pin_policy import (  # noqa: E402
    PIN_ALLOW as _PIN_ALLOW,
)
from .pin_policy import (
    PIN_DENY as _PIN_DENY,
)
from .pin_policy import (
    PIN_DENY_HC as _PIN_DENY_HC,
)

# a phase: load round-trip cache (Plan 02 produces this)
_ROUNDTRIP_CACHE_PATH = (
    Path(__file__).parent / "opsin_imports" / "_phase150_validation.json"
)
_PROVISIONAL_MODE = not _ROUNDTRIP_CACHE_PATH.exists()
if not _PROVISIONAL_MODE:
    try:
        with open(_ROUNDTRIP_CACHE_PATH) as f:
            _ROUNDTRIP_DATA = json.load(f)
        _ROUNDTRIP_PASS = frozenset(_ROUNDTRIP_DATA.get("pass_smiles", []))
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("Failed to load _phase150_validation.json: %s", e)
        _ROUNDTRIP_PASS = frozenset()
        _PROVISIONAL_MODE = True
else:
    _ROUNDTRIP_PASS = frozenset()
    logger.info(
        "Phase 150 round-trip cache absent; classifier in PROVISIONAL mode "
        "(Signal 3 deferred until Plan 02 validator runs)"
    )


def _is_promotable(smiles: str, name: str) -> bool:
    """3-signal AND gate per a phase internal notes - pure function.

    Signal 1: _is_complete_name heuristic.
    Signal 2: data/iupac_2013_pin_list.json allow-list (PIN authority).
    Signal 3: per-entry OPSIN round-trip via InChI L1 (Plan 02).

    Promotion rule: (S1 OR S2) AND S3.
    Special case: explicit DENY in Signal 2 always REJECTS (overrides S1, S3).
    Provisional mode (Plan 01 before validator runs): (S1 OR S2) only.

    Source: 150-internal notes +.
    Source: internal notes section 4.1.
    """
    name_lower = name.lower().strip()
    if name_lower in _PIN_DENY:
        return False
    s1 = _is_complete_name(name)
    s2 = name_lower in _PIN_ALLOW
    if _PROVISIONAL_MODE:
        return s1 or s2
    s3 = smiles in _ROUNDTRIP_PASS
    return (s1 or s2) and s3


# a phase: _OPSIN_NON_PIN_EXCLUSIONS DERIVED from JSON allow-list
# (single source of truth). Preserved as frozenset alias for backward-compat
# with downstream readers.
_OPSIN_NON_PIN_EXCLUSIONS = _PIN_DENY


# --- PA1 R5: governing the SECOND retained-name surface -----------------------
# `opsin_imports/simple_groups.py` carries 423 entries that EVERY ONE tag with
# ``'is_pin': False``. That flag is a uniform generator default (see
# scripts/import_opsin_xml.py, which hard-codes False) and therefore carries zero
# per-entry information -- and nothing reads it. Two existing tests
# (tests/unit/data/test_opsin_imports.py, test_opsin_merge_layer.py) actually
# ASSERT the uniformity, which confirms it is a structural default rather than a
# curated judgement. The consequence measured by the PA1 audit: 222 of 420
# nameable entries emit their own trivial name as the headline, with no PIN
# authority consulted, and 11 of the 12 known non-PIN emissions came from here.
#
# The literal reading of ``is_pin: False`` -- "no entry on this surface may be a
# headline PIN unless the curated allow-list says so" -- is implemented below and
# gated behind ORTHONYM_GOVERN_OPSIN_SIMPLE_GROUPS=1.
#
# MEASURED BLAST RADIUS (why it ships OFF).
# * Withdraws 272 headline keys: ALL_RETAINED_NAMES 867 -> 595, with all 272
# demoted to --trivial (GENERAL_RETAINED_NAMES 117 -> 403), not deleted.
# * benchmarks/pubchem_2000.csv, exact whole-molecule hits: 0 of 2000. No
# molecule in that corpus IS one of the withdrawn entries.
# * A/B naming run, 300-molecule random sample (seed 1234) of the same corpus:
# 1 of 300 emitted names changed (0.33%), 0 became "unknown".
#
# The blast radius is therefore SMALL -- but the single change is a REGRESSION,
# which is the actual reason this is off:
# cid 5313963, withdrawing the trivial "choline"
# OFF: choline 3-[(hexadec-1-en-1-yl)oxy]1-phosphonooxypropan-2-yl oleate
# ON: 3-[(hexadec-1-en-1-yl)oxy]propane-1,2-diyl oleate ethyl phosphatium
# The ON name silently loses the trimethylammonium group entirely.
#
# That is the same failure mode the PA1 a lever tranche measured 8 times over:
# withdrawing a trivial name only helps when the systematic engine derives the
# correct PIN in its place, and deny-only produced "methane" for N=C=N,
# "methanamine" for thiuram monosulfide, and a 1,2,3-triol for pentaerythritol.
# So the evidence is consistent across both experiments: flipping 423 entries
# blind trades a bounded set of non-PIN names for an unbounded set of wrong ones.
# The individually-adjudicated denies live in iupac_2013_pin_list.json, which is
# the correct granularity: each carries a Blue Book citation AND a verified
# replacement. This switch exists so the class can be measured and so the
# remaining entries can be adjudicated in batches behind a real gate.
_GOVERN_OPSIN_SIMPLE = (
    os.environ.get("ORTHONYM_GOVERN_OPSIN_SIMPLE_GROUPS", "").strip() == "1"
)

try:
    from .opsin_imports.simple_groups import OPSIN_SIMPLE_GROUPS as _SIMPLE_RAW
except ImportError:
    _SIMPLE_RAW = {}

# SMILES keys whose ONLY provenance is the ungoverned simple_groups surface.
_SIMPLE_GROUP_KEYS = frozenset(_SIMPLE_RAW)


def _governed_out(smiles: str, name: str) -> bool:
    """True iff R5 governance withdraws this OPSIN candidate as a headline name.

    Applies only to the ``simple_groups`` surface, and only to names the curated
    PIN allow-list does not positively affirm (``pin: true``). Entries already
    carrying an explicit ``pin: false`` deny are handled by ``_is_promotable``
    and are unaffected by this switch.
    """
    if not _GOVERN_OPSIN_SIMPLE:
        return False
    if smiles not in _SIMPLE_GROUP_KEYS:
        return False
    return name.lower().strip() not in _PIN_ALLOW


# a phase: REFACTORED _OPSIN_NAMES filter (single-signal -> 3-signal AND).
_OPSIN_NAMES: Dict[str, str] = {
    smi: name for smi, name in _OPSIN_NAMES_RAW.items()
    if _is_promotable(smi, name) and not _governed_out(smi, name)
}

_stem_count = len(_OPSIN_NAMES_RAW) - len(_OPSIN_NAMES)
if _stem_count > 0:
    logger.debug(
        "OPSIN imports: %d entries promoted, %d stems/non-PINs filtered "
        "(provisional_mode=%s)",
        len(_OPSIN_NAMES), _stem_count, _PROVISIONAL_MODE
    )

# Merge: OPSIN first, then hand-curated overwrites (: hand-curated wins)
# a phase: gate the hand-curated dict against _PIN_DENY_HC (the unified
# deny set, hc_override-exempt) BEFORE the merge — eliminating the two-path
# asymmetry where HC was merged RAW while OPSIN imports passed _is_promotable.
# This is DENY-based exclusion, NOT the full (S1 OR S2) AND S3 promotion gate:
# the a phase A1 audit proved full-gate routing of the curated dict drops 217
# genuine names whose canonical SMILES are absent from the OPSIN round-trip cache
# (Signal 3). _OPSIN_NAMES already applied _is_promotable (deny included); this
# extends the same explicit-DENY enforcement to the hand-curated side.
_HAND_CURATED_GATED: Dict[str, str] = {
    smi: name for smi, name in _HAND_CURATED_NAMES.items()
    if name.lower().strip() not in _PIN_DENY_HC
}
ALL_RETAINED_NAMES: Dict[str, str] = {**_OPSIN_NAMES, **_HAND_CURATED_GATED}

# Backward-compatible alias
RETAINED_NAMES = ALL_RETAINED_NAMES


# Task 1.9 (PIN-policy --trivial fallback): the general-only retained names —
# exactly the entries the PIN deny gate EXCLUDES from ALL_RETAINED_NAMES. The
# default (PIN) pipeline never consults this dict (fail-closed); the opt-in
# ``Orthonym(trivial_fallback=True)`` / CLI ``--trivial`` path falls back to it
# by canonical SMILES ONLY when the systematic pipeline produced no derivable
# PIN. Two contributing sources, mirroring the two deny surfaces:
# (a) hand-curated entries filtered out by ``_PIN_DENY_HC`` (e.g. glycerol,
# allyl alcohol, chloroform, catechol, nicotinic acid), and
# (b) OPSIN-import candidates blocked SOLELY by ``_PIN_DENY`` membership —
# i.e. names that WOULD have promoted (S1 or S2, and S3 in non-provisional
# mode) but for the explicit deny (e.g. dihydroxalate, dihydrotartrate,
# glyoxal, phosgene). Candidates that fail promotion for OTHER reasons
# (stem heuristic, absent round-trip) are NOT re-admitted here.
# Hand-curated wins key collisions (same precedence as ALL_RETAINED_NAMES).
def _opsin_blocked_solely_by_deny(smiles: str, name: str) -> bool:
    """True iff the OPSIN candidate would have promoted but for the deny gate.

    Reconstructs ``_is_promotable`` WITHOUT its ``_PIN_DENY`` short-circuit:
    the candidate must satisfy the real promotion rule ((S1 OR S2) AND S3, or
    (S1 OR S2) in provisional mode) AND be explicitly denied. This excludes
    candidates that fail promotion for any reason other than the deny.
    """
    name_lower = name.lower().strip()
    if name_lower not in _PIN_DENY:
        return False
    s1 = _is_complete_name(name)
    s2 = name_lower in _PIN_ALLOW
    if _PROVISIONAL_MODE:
        return s1 or s2
    s3 = smiles in _ROUNDTRIP_PASS
    return (s1 or s2) and s3


_GENERAL_OPSIN: Dict[str, str] = {
    smi: name for smi, name in _OPSIN_NAMES_RAW.items()
    if _opsin_blocked_solely_by_deny(smi, name)
    # PA1 R5: a name withdrawn from the PIN headline by the governance switch is
    # still a legitimate general-nomenclature name, so it must remain reachable
    # via --trivial -- exactly as the explicit pin:false denies are. Without this
    # the switch would delete names outright instead of demoting them.
    or (_governed_out(smi, name) and _is_promotable(smi, name))
}
_GENERAL_HAND_CURATED: Dict[str, str] = {
    smi: name for smi, name in _HAND_CURATED_NAMES.items()
    if name.lower().strip() in _PIN_DENY_HC
}
GENERAL_RETAINED_NAMES: Dict[str, str] = {
    **_GENERAL_OPSIN, **_GENERAL_HAND_CURATED
}


def get_general_retained_name(canonical_smiles: str) -> Optional[str]:
    """Look up a general-only (PIN-denied) retained name by canonical SMILES.

    Returns None when no such name exists. Consulted ONLY by the ``--trivial``
    fallback path (Task 1.9); the default PIN pipeline never calls this.
    """
    return GENERAL_RETAINED_NAMES.get(canonical_smiles)


def get_retained_name(canonical_smiles: str) -> Optional[str]:
    """Look up retained name from merged dictionary."""
    return ALL_RETAINED_NAMES.get(canonical_smiles)


def has_retained_name(canonical_smiles: str) -> bool:
    """Check if a retained name exists for this SMILES."""
    return canonical_smiles in ALL_RETAINED_NAMES


def register_retained_name(canonical_smiles: str, name: str):
    """Register a retained name at runtime (for testing/dynamic use).

    a phase REVIEW: this is the CANONICAL runtime registration
    API. The legacy ``data.retained_names.add_retained_name`` is a
    deprecated alias that now delegates here (per fix) so both
    APIs keep ``ALL_RETAINED_NAMES`` and the HC dict in sync.

    Source: internal notes +.
    """
    ALL_RETAINED_NAMES[canonical_smiles] = name


# Log merge stats at import time
_conflict_count = sum(1 for k in _OPSIN_NAMES if k in _HAND_CURATED_NAMES)
if _conflict_count > 0:
    logger.info(
        "Retained names merged: %d hand-curated + %d OPSIN imports "
        "(%d conflicts, hand-curated wins; provisional_mode=%s)",
        len(_HAND_CURATED_NAMES), len(_OPSIN_NAMES), _conflict_count,
        _PROVISIONAL_MODE
    )

__all__ = ["RETAINED_NAMES", "ALL_RETAINED_NAMES", "get_retained_name",
           "has_retained_name", "register_retained_name",
           "GENERAL_RETAINED_NAMES", "get_general_retained_name"]
