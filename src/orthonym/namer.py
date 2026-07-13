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
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from .errors import (
    OrthonymLimitError,
    classify_scope_limit,
    classify_failure_limit,
    is_failure_name,
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

    # Predicate said True but no authoritative injection happened -> log gap.
    n_atom_stereo = sum(1 for a in mol.GetAtoms() if a.HasProp('_CIPCode'))
    n_bond_stereo = sum(1 for b in mol.GetBonds() if b.HasProp('_CIPCode'))
    logger.warning(
        "Stereo backstop: '%s' (handler: %s) has %d R/S + %d E/Z but name lacks "
        "descriptors. Fix handler to include stereo natively.",
        name[:50], handler, n_atom_stereo, n_bond_stereo
    )
    return name


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


def _self_consistency_verdict(input_smiles: str, opsin_smiles: str) -> str:
    """``"ok"`` | ``"mismatch"`` | ``"inconclusive"`` — does the OPSIN re-perception of
    the emitted name encode the SAME constitution as the input structure?

    Constitution = InChIKey skeleton block (formula + connectivity + mobile-H;
    stereo- and charge-insensitive) PLUS net formal charge (the skeleton block
    excludes the charge layer, so a charge-dropping name would otherwise pass)."""
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
    if ca is None or cb is None:
        return "ok"  # cannot compare charge -> trust the skeleton (fail-OPEN on charge)
    # Only a genuinely CHARGED input whose charge the name drops/changes is a leak
    # (e.g. [O-]O, -1, named 'dioxidane', 0). A NEUTRAL input is exempt: acid/ester
    # names are protonation-ambiguous in OPSIN (e.g. 'methyl phosphate' round-trips
    # to the -2 phosphate dianion) — that is not a wrong-molecule error, and firing
    # on it wrongly suppresses correct names.
    if ca != 0 and ca != cb:
        return "mismatch"
    return "ok"


def _self_consistency_decision(name: str, smiles: Optional[str], opsin_smiles: str,
                               stats: Optional[Dict[str, int]]) -> str:
    """SELF-01: OPSIN already PARSED ``name``; verify it parses to the SAME molecule.

    Suppresses (mode "on") only on a VERIFIED constitutional mismatch — never on a
    stereo-only / tautomer difference, never when the comparison is inconclusive
    (fail-OPEN). In mode "warn" it logs + counts the would-suppression but ships the
    name unchanged (behaviour-neutral); in mode "off" it is a no-op."""
    if _SC_MODE == "off" or not smiles:
        return name
    verdict = _self_consistency_verdict(smiles, opsin_smiles)
    if verdict != "mismatch":
        return name  # "ok" ships; "inconclusive" fails OPEN
    if stats is not None:
        stats["self_consistency_mismatch"] = stats.get("self_consistency_mismatch", 0) + 1
    if _SC_MODE == "warn":
        logger.warning(
            "SELF-01 would-suppress (warn-only): %r names a constitutionally DIFFERENT "
            "molecule (input=%s opsin=%s)", name[:80], smiles, opsin_smiles)
        return name
    # mode "on": suppress to the honest descriptive fallback.
    if stats is not None:
        stats["self_consistency_suppressed"] = stats.get("self_consistency_suppressed", 0) + 1
    logger.warning("SELF-01 suppressed (different molecule): %r (opsin=%s)",
                   name[:80], opsin_smiles)
    return _descriptive_fallback(smiles)


def _final_opsin_validity_gate(name: str, smiles: Optional[str],
                               stats: Optional[Dict[str, int]] = None) -> str:
    """SUB-03 real-OPSIN validity gate (D-11/D-12/D-13).

    Suppress an OPSIN-unparseable production name to the EXISTING
    _descriptive_fallback STRING (never None / never a shipped invalid string).
    Runs AFTER stereo + grammar repair, as the last transform before the value
    leaves name()/name_with_confidence(); inside the is_top_level_naming guard
    so it never fires on decomposition fragments.
    """
    if not name or _DISABLE_VALIDITY_GATE:
        return name
    # Don't re-gate an already-descriptive fallback (it won't OPSIN-parse, and
    # re-suppressing is idempotent anyway) — skip the wasted OPSIN subprocess.
    try:
        from .errors import _DESCRIPTIVE_FALLBACK_NAMES
        if name in _DESCRIPTIVE_FALLBACK_NAMES:
            return name
    except Exception:
        pass
    # D-13 fail-OPEN: probe the JAR FIRST.
    if not _validity_gate_jar_present():
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
        return name  # transient -> fail-OPEN (never suppress)
    # DD2 / BBR-GATE (Phase D): OPSIN's generation grammar does not recognise the
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
        return name
    # W3-P07 (P-65.6.3.3.3.2 method (1)): functional-class polyol polyester PIN
    # ('propane-1,2,3-triyl 1,3-diacetate 2-propanoate') — correct-by-construction
    # from _assemble_glyceride but OPSIN cannot parse the multi-anion syntax
    # (exactly the inositol/dianhydride generation-grammar gap).
    if _POLYOL_POLYESTER_PIN_RE.match(name):
        return name
    # BBR-GATE / DEF-9 (Phase 169.7): decide on WHERE OPSIN fails. If the name is
    # rejected ONLY because of its stereo layer — i.e. the stereo-STRIPPED
    # constitutional form parses — then the name is correct by construction
    # (P-91/P-93) and OPSIN's narrower generation-side stereo grammar must NOT gate
    # Orthonym correctness (audit Dim-08 §C; the verbatim Blue Book PIN
    # `(1s,4s)-cyclohexane-1,4-diol` was being suppressed to `unknown`). The
    # constitutional gate stays STRICT: a name whose stereo-stripped form ALSO fails
    # to parse is still suppressed. strip_stereo is read-only — the shipped name keeps
    # its stereo descriptors.
    from .rules.stereochemistry import strip_stereo
    _stripped = strip_stereo(name)
    if _stripped != name and _validity_gate_status(_stripped) == "parsed":
        if stats is not None:
            stats["gate_stereo_kept"] = stats.get("gate_stereo_kept", 0) + 1
        return name  # ship the full stereo name — only the stereo layer is OPSIN-narrow
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
    return _descriptive_fallback(smiles)


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
    from .rules.benzene import orient_benzene, get_benzene_substituents
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
                # (the ring atoms bearing a senior suffix group) to the lowest
                # locant before detachable substituents. Derived from the
                # is_suffix marker already set by get_benzene_substituents
                # (acids/aldehydes/amides); phenols are anchored downstream by
                # _renumber_relative_to, so an empty set here is a no-op.
                pcg_positions = {
                    atom_idx for atom_idx, subs in substituents.items()
                    if any(s.get("is_suffix") for s in subs)
                }
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
                 _principal_group_override: Optional[str] = None):
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
        """
        self.style = style
        # Task 1.9 (PIN-policy): fallback-only opt-in. When True the name path
        # substitutes a general-only retained name for the "unknown organic
        # compound" failure signal; it never overrides a derived PIN.
        self._trivial_fallback: bool = trivial_fallback
        # SUB-03 (169.5): per-instance bypass for the OPSIN validity gate, used
        # by neutralize-recurse / fragment intermediate naming (those produce an
        # INTERMEDIATE name that is transformed downstream, not a final output,
        # so they must not be gated). Final top-level name() calls leave this
        # False -> the gate applies.
        self._disable_opsin_validity_gate: bool = _disable_opsin_validity_gate
        # Phase 173.6 T3: scoped principal-group override (FG name). Set ONLY by the
        # charged chokepoint's S/P-oxoacid-anion re-entry (_reenter_forced) so the
        # anionic group is forced as the principal characteristic group per P-72/P-74.
        # None for every normal name() call -> byte-identical production behaviour.
        self._principal_group_override: Optional[str] = _principal_group_override
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
                        end_naming_session()
                        return _iso_name
                    # Decorator failed closed on an isotope-labeled molecule: REFUSE.
                    # Falling through would emit the UNLABELED skeleton name (label
                    # silently dropped) — a wrong name the OPSIN validity gate cannot
                    # catch (it parses to the unlabeled structure; SELF-01 ignores
                    # isotopes). Accuracy-first: never emit a label-dropping name.
                    end_naming_session()
                    return _descriptive_fallback(smiles)
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
            # HYG-02: structural scope pre-check (opt-in). Only the wildcard
            # class is refused here — the one class with zero in-scope risk.
            if raise_on_limit:
                _probe = Chem.MolFromSmiles(smiles)
                if _probe is not None:
                    _scope = classify_scope_limit(_probe)
                    if _scope is not None:
                        _scope.smiles = smiles
                        raise _scope
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
                if raise_on_limit and is_top_level_naming():
                    raise
                # Task 1.9: PIN fails closed; --trivial falls back to a
                # general-only retained name when no PIN could be derived.
                return self._apply_trivial_fallback(_limit.message, smiles)
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
                        result = _final_opsin_validity_gate(
                            result, smiles, self._grammar_stats,
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
            # HYG-02: post-failure limit (opt-in). If naming produced no real
            # name, map the failure to a named code. Keyed off an actual failure
            # so it can never fire on a successfully-named compound.
            if raise_on_limit and is_top_level_naming() and is_failure_name(result):
                _probe = Chem.MolFromSmiles(smiles)
                if _probe is not None:
                    raise classify_failure_limit(_probe, smiles=smiles)
            # Task 1.9: PIN fails closed; --trivial falls back to a general-only
            # retained name when the systematic pipeline derived no PIN.
            return self._apply_trivial_fallback(result, smiles)
        finally:
            end_naming_session()

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
        return trivial if trivial else result

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
        from .assembly.coverage_scoring import retrieve_confidence, clear_confidence
        start_naming_session()
        clear_confidence()
        try:
            try:
                name = self._name_impl(smiles)
            except OrthonymLimitError as _limit:
                # G0 fail-closed (DD7 S1): a ring subsystem refused. Surface the
                # descriptive-fallback name + the named limit code (no crash).
                if _limit.smiles is None:
                    _limit.smiles = smiles
                # Task 1.9 (C8): apply trivial fallback here, mirroring name().
                _fallback_name = self._apply_trivial_fallback(_limit.message, smiles)
                return {
                    'name': _fallback_name,
                    'confidence': 0.0,
                    'factors': {},
                    'handler': 'fallback',
                    'limit': _limit.as_dict(),
                }
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
                        )
            metadata = retrieve_confidence()
            # If no candidate was scored (early return path), build minimal metadata
            if not metadata['name']:
                metadata = {
                    'name': name,
                    'confidence': 1.0,  # Early return paths are high confidence
                    'factors': {'ratio': 1.0, 'atom_coverage': 1.0,
                                'fg_recognition': 1.0,
                                'substituent_completeness': 1.0},
                    'handler': 'direct',
                }
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

            return metadata
        finally:
            end_naming_session()
            clear_confidence()

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
                    store_confidence(CandidateName(
                        name=name,
                        handler='chain',
                        confidence=1.0,
                        atom_to_locant=dict(features.atom_to_locant),
                    ))

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
        if (not _skip_decomposition
                and name
                and result.class_id == StoutClass.GENERAL
                and mol.GetNumHeavyAtoms() > 15):
            assembled = name  # local alias for v18-byte-identical body
            _GARBLED_TOKENS = ('cycloane', 'anedicarboxamide', 'aneyl')
            assembled_lower = assembled.lower()
            is_garbled = any(tok in assembled_lower for tok in _GARBLED_TOKENS)

            # Also detect stub-only names: when the parent text is empty,
            # the assembly may produce just a bare suffix like "ane" or "ol".
            # A name shorter than 6 chars for a 15+ atom molecule is garbled.
            if not is_garbled and len(assembled) < 6:
                is_garbled = True

            if is_garbled:
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
            if (conf_score is not None
                    and conf_handler != 'unknown'
                    and conf_handler != 'retained_name'):
                atom_cov = conf_data.get('factors', {}).get('atom_coverage', 1.0)
                if atom_cov < 0.55:
                    # Verify molecule has cleavable bonds before triggering
                    from .decomposition.bond_cleavage import find_cleavable_bonds
                    if find_cleavable_bonds(mol):
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
        # features so the DROP-23 tier-3 fallback in polyfunctional.py can read them.
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
                    ring_info=_ring_info
                )
                features.parent_selection_result = selection  # Phase 148 D-02 (V18 Appendix A.5)

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
                    features.mol, pah_name
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

                    # WS-4 / BBR-RSFX (DEF-6): identify the ring atoms bearing the
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
                   raise_on_limit: bool = False):
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
    namer = Orthonym(
        style=style,
        enable_triviality_controller=enable_triviality_controller,
        enable_group_splitting=enable_group_splitting,
        trivial_fallback=trivial_fallback,
    )

    if include_confidence:
        try:
            return namer.name_with_confidence(smiles)
        except ValueError:
            raise
        except Exception as e:
            logger.warning("name_with_confidence failed: %s", e)
            return {
                'name': _descriptive_fallback(smiles),
                'confidence': 0.0,
                'factors': {},
                'handler': 'fallback',
            }

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
