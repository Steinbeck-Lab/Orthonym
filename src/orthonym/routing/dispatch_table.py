"""Phase 158 dispatch table for class-first routing (CFR-02).

Authoritative source for v18 outer-cascade dispatch decisions. The DISPATCH_TABLE
is built at module-import time by 19 explicit ``_register_dispatch(...)`` calls
(18 outer-cascade entries from 158-AUDIT-CFR.md § 1 + 1 GENERAL catch-all per
CONTEXT D-08); post-import the table is FROZEN (RuntimeError on further
registration via the ``_REGISTRATION_FROZEN`` sentinel).

Architecture (158-CONTEXT.md):
- StoutClass(StrEnum) — D-11 explicit type-safe enum; one member per dispatch
  class. v19 sibling-phase slots are commented out per D-20/D-21/D-22 (NOT
  registered in DISPATCH_TABLE; reserved for Phase 161 / 162 / 163).
- ClassDispatchEntry — D-05 frozen dataclass; immutable; serializable for the
  D-08 audit log.
- DISPATCH_TABLE: OrderedDict[StoutClass, ClassDispatchEntry] — D-05 + D-06
  priority-ordered; OrderedDict makes priority order explicit even though
  Python 3.7+ dict preserves insertion order.
- Per-class predicate factories — 1-3-line wrappers around existing detection
  helpers (D-26 pure; AP-4 line-count discipline).
- Per-class handler shims — 1-line wrappers around existing handler functions
  in src/orthonym/rules/ (AP-5 no-logic-in-shim discipline).
- Lazy imports inside predicate / handler bodies (PATTERNS § 3 + namer.py:853
  pattern) to avoid circular ``routing -> rules -> routing`` imports at
  module-import time.

Anti-pattern hygiene (158-AUDIT-CFR.md AP-block):
- AP-1: silent fallthrough -> GENERAL @ priority 99999 with ``lambda *_: True``.
- AP-3: invent-as-you-go entries -> audit § 1 is the locked spec; every entry
  here traces 1:1 to a § 1 row.
- AP-4: predicate body > 3 lines -> all bodies stay 1-3 lines.
- AP-5: handler logic in shim -> all handler shims are 1-line wrappers (the
  ANION_SMALL / POLY_ANION / MULTI_COMPONENT_NEUTRAL shims preserve the v18
  byte-identical neutralize-recurse + try/except logic verbatim per audit
  § 3 + RESEARCH § "Code Examples").
- AP-8: predicate side effects -> ``side_effect_inventory`` MUST be ``()``
  for every entry (D-26 hard invariant).
- AP-15: padding § 1 with inner ``_classify`` classes -> Path-(b) reconciliation
  per RESEARCH § 8.4; inner classes are Phase 160 / D-23 territory.
- AP-21: predicate that mutates ``mol`` / ``MolecularFeatures`` / global state
  -> the integrity test ``test_side_effect_inventory_is_empty`` (Plan-03)
  parametrizes over ``list(StoutClass)`` and asserts ``()`` for every entry.

IUPAC P-section cites for each StoutClass live in the per-row docstring
(audit § 1 column "iupac_section").
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# NOTE: NO eager import of ``orthonym.rules.*`` or ``orthonym.namer``;
# predicate factories + handler shims use lazy imports inside their bodies
# per ``namer.py:853`` precedent to avoid the circular
# ``orthonym.routing -> orthonym.rules -> orthonym.namer -> orthonym.routing``
# load order. ``rdkit.Chem`` is a third-party module so eager import is safe.
from rdkit import Chem  # noqa: F401  -- type annotation only


# ---------------------------------------------------------------------------
# Section 2: StoutClass(StrEnum)
#
# One member per audit § 1 row (Path-(b) reconciliation: 18 outer-cascade
# entries + GENERAL catch-all = 19 active members). v19 sibling-phase slots
# are commented out per CONTEXT D-20 / D-21 / D-22 — they are NOT registered
# in DISPATCH_TABLE in Phase 158.
# ---------------------------------------------------------------------------


# Python 3.10 compatibility shim: ``StrEnum`` is stdlib only from 3.11+. Build
# an equivalent ``str + Enum`` mixin so the enum members behave like strings
# for serialization (audit log JSON, telemetry dict keys) AND like Enum
# members for exhaustive iteration / mypy checking.
class _StrEnumBase(str, Enum):
    """``str + Enum`` mixin; equivalent to ``enum.StrEnum`` from Python 3.11+."""

    def __str__(self) -> str:  # mirror ``StrEnum.__str__`` value-only repr
        return str(self.value)


class StoutClass(_StrEnumBase):
    """Phase 158 dispatch class identifiers (CONTEXT D-11).

    Per CONTEXT D-11 + audit § 1 Path-(b) reconciliation: 18 outer-cascade
    entries (rows 1-18 of audit § 1) + 1 GENERAL catch-all = 19 enum members.
    v19 sibling-phase slots are commented out (audit § 4) and inserted by
    Phase 161 / 162 / 163 as 1-line ``_register_dispatch(...)`` additions.
    """

    # --- Tier-1 charged-species + dot-disconnected (audit § 1 rows 1-8) ---
    SALT = "salt"                             # row 1; namer.py:852-854; P-65.6.2.1/P-63.8.1/P-77
    RADICAL = "radical"                       # row 2; namer.py:855-857; P-15.7
    ZWITTERION = "zwitterion"                 # row 3; namer.py:858-860; P-74
    ANION_RETAINED = "anion_retained"         # row 4; namer.py:861-871; P-72
    CATION_RETAINED = "cation_retained"       # row 5; namer.py:872-875; P-73
    CATION_QUATERNARY = "cation_quaternary"   # Phase 184 WS-E.1 (P-73.1.2.1 quaternary aminium; priority 480, tier 1 — below ZWITTERION@300/LIPID@250, above CATION_RETAINED@500)
    ANION_SMALL = "anion_small"               # row 6; namer.py:877-914; P-72.2.1
    POLY_ANION = "poly_anion"                 # row 7; namer.py:916-956; P-72.2.1
    MULTI_COMPONENT_NEUTRAL = "multi_component_neutral"  # row 8; namer.py:967-998

    # --- Tier-2 multiplicative + class-routed handlers (audit § 1 rows 9-16) ---
    MULTIPLICATIVE = "multiplicative"         # row 9;  namer.py:1040-1043; P-51.3
    CARBOHYDRATE_LOOKUP = "carbohydrate_lookup"  # row 10; namer.py:1051-1062; P-10
    NATURAL_PRODUCT = "natural_product"       # row 11; namer.py:1070-1073; P-10
    PEPTIDE = "peptide"                       # row 12; namer.py:1078-1083; P-66.6.3
    RETAINED_NAME = "retained_name"           # row 13; namer.py:1085-1088; P-22+P-25
    AMINO_ACID = "amino_acid"                 # row 14; namer.py:1090-1094; P-66
    SKELETAL_REPLACEMENT = "skeletal_replacement"  # row 15; namer.py:1099-1102; P-15.4
    CYCLOPHANE = "cyclophane"                 # row 16; namer.py:1110-1113; P-26.4

    # --- Decomposition + general catch-all (audit § 1 rows 17-18 + GENERAL) ---
    DECOMPOSITION_PRE_GENERAL = "decomposition_pre_general"  # row 17; namer.py:1118-1122
    GENERAL = "general"                       # row 18; namer.py:1124-1131; CFR-02 D-08

    # --- v19 sibling-phase reservations (audit § 4); commented-out -> NOT registered ---
    ORGANOMETALLIC = "organometallic"             # Phase 161 (P-69; priority 50)
    LIPID = "lipid"                               # Phase 180 (P-107 lipid backbone; priority 250, tier 1 — before ZWITTERION@300, RESOLVED A1)
    # Phase 163 FRN attaches via SENIORITY_ORDER extension (for chalcogen
    # acid/amide/aldehyde/ketone analogs that route through GENERAL@99999) +
    # INNER_DISPATCH entry for imidate (functional-class naming for
    # iminoesters). NO new OUTER CFR rows. See */
    # 163-AUDIT-FRN.md` § 8 for the full intercept analysis. Phase 161 D-02
    # priority-placeholder-correction precedent.


# ---------------------------------------------------------------------------
# Section 3: ClassDispatchEntry + ClassDispatchResult dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassDispatchEntry:
    """Phase 158 D-05 + D-26: frozen dataclass row of DISPATCH_TABLE.

    ``side_effect_inventory`` MUST be ``()`` per the D-26 hard invariant; the
    integrity test ``test_side_effect_inventory_is_empty`` (Plan-03) asserts
    this for every entry. AP-21 explicitly bans any predicate that mutates
    ``mol``, ``MolecularFeatures``, module-global state, or thread-local state.
    """

    class_id: StoutClass
    priority: int                                                  # D-06: spaced in 100s for v19 insertability
    tier: int                                                      # 1 = pre-perception (mol-only); 2 = post-perception
    predicate: Callable[..., bool]
    handler: Callable[..., Optional[str]]                          # may return None to signal cascade-continuation
    iupac_section: str
    description: str
    side_effect_inventory: Tuple[str, ...] = ()                    # D-26 hard invariant


@dataclass(frozen=True)
class ClassDispatchResult:
    """Phase 158 D-03: returned by ``ClassFirstRouter.dispatch()``.

    Caller invokes ``result.handler(...)`` to get the name string. Frozen so
    a single dispatch result can be safely passed across recursive call
    boundaries (RL-4 fresh-instance pattern).
    """

    class_id: StoutClass
    handler: Callable[..., Optional[str]]
    audit_record: dict
    tier: int


# ---------------------------------------------------------------------------
# Section 4: DISPATCH_TABLE + _register_dispatch helper (D-05 + D-06)
# ---------------------------------------------------------------------------

DISPATCH_TABLE: "OrderedDict[StoutClass, ClassDispatchEntry]" = OrderedDict()
_REGISTRATION_FROZEN: bool = False


def _register_dispatch(
    class_id: StoutClass,
    priority: int,
    tier: int,
    predicate: Callable[..., bool],
    handler: Callable[..., Optional[str]],
    iupac_section: str,
    description: str,
    side_effect_inventory: Tuple[str, ...] = (),
) -> None:
    """Phase 158 D-05: private helper; module-import time only.

    Raises:
        RuntimeError: if called after ``_REGISTRATION_FROZEN`` is set;
            if the same ``class_id`` is registered twice;
            if the same ``priority`` is registered twice.
    """
    if _REGISTRATION_FROZEN:
        raise RuntimeError(
            f"DISPATCH_TABLE is frozen post-import; cannot register {class_id}."
        )
    if class_id in DISPATCH_TABLE:
        raise RuntimeError(f"Duplicate registration for {class_id}.")
    if any(e.priority == priority for e in DISPATCH_TABLE.values()):
        raise RuntimeError(
            f"Priority {priority} is already assigned; pick a different value "
            f"(priorities are spaced in 100s per CONTEXT D-06)."
        )
    DISPATCH_TABLE[class_id] = ClassDispatchEntry(
        class_id=class_id,
        priority=priority,
        tier=tier,
        predicate=predicate,
        handler=handler,
        iupac_section=iupac_section,
        description=description,
        side_effect_inventory=side_effect_inventory,
    )


# ---------------------------------------------------------------------------
# Section 5: Per-class predicate factories (audit § 1 + § 2; D-26 pure)
#
# Every predicate body is 1-3 lines per CONTEXT line 169 + AP-4. Lazy imports
# live INSIDE bodies per PATTERNS § 3 to avoid circular load order. The
# ``**kwargs`` tail absorbs ``_skip_decomposition`` (and any future kwargs)
# threaded by ``ClassFirstRouter.dispatch()``; no predicate reads it except
# ``_is_decomposition_pre_general`` (RL-7 option (b)).
# ---------------------------------------------------------------------------


def _is_salt(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """Tier-1; mirrors namer.py:852 (audit § 1 row 1; § 2.1 purity proof)."""
    from orthonym.perception.ions import detect_species_type
    return detect_species_type(mol) == 'salt'


def _is_radical(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """Tier-1; mirrors namer.py:855 (audit § 1 row 2; § 2.2 purity proof)."""
    from orthonym.perception.ions import detect_species_type
    return detect_species_type(mol) == 'radical'


def _is_zwitterion(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """Tier-1; mirrors namer.py:858 (audit § 1 row 3; § 2.3 purity proof).

    Note: this matches FORMAL-CHARGE-NEUTRAL-FALSE zwitterions (species_type
    returns ``'zwitterion'``). The FORMAL-CHARGE-ZERO + true-zwitterion-character
    case (namer.py:1000-1034 in-place mutation) is INLINE in ``_name_impl``
    BEFORE CFR.dispatch per RL-3 option (c); it is NOT a CFR entry.
    """
    from orthonym.perception.ions import detect_species_type
    return detect_species_type(mol) == 'zwitterion'


def _is_anion_retained(mol, smiles, canonical_smiles, features=None, *, _style: str = "pin", **kwargs) -> bool:
    """Tier-1; mirrors namer.py:861-871 (audit § 1 row 4; § 2.4 purity proof).

    Predicate IS handler pattern: name_anion(retained_only=True) returns the
    retained name when the lookup hits, else None. The handler shim re-runs
    the same call to obtain the name string.
    """
    from orthonym.perception.ions import detect_species_type, get_ion_sites
    if detect_species_type(mol) != 'ion':
        return False
    sites = get_ion_sites(mol)
    if not (sites['anions'] and not sites['cations']):
        return False
    from orthonym.rules.ions import name_anion
    return name_anion(mol, style=_style, retained_only=True) is not None


def _is_cation_retained(mol, smiles, canonical_smiles, features=None, *, _style: str = "pin", **kwargs) -> bool:
    """Tier-1; mirrors namer.py:872-875 (audit § 1 row 5; § 2.5 purity proof)."""
    from orthonym.perception.ions import detect_species_type, get_ion_sites
    if detect_species_type(mol) != 'ion':
        return False
    sites = get_ion_sites(mol)
    if not (sites['cations'] and not sites['anions']):
        return False
    from orthonym.rules.ions import name_cation
    return name_cation(mol, style=_style, retained_only=True) is not None


def _is_cation_quaternary(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """Phase 184 WS-E.1 (P-73.1.2.1): narrow predicate for a STANDALONE quaternary
    ammonium cation that should get the systematic ``-aminium`` PIN.

    Fires ONLY for: exactly one cation site, NO anion sites (a mixed-sign
    zwitterion is owned by ZWITTERION@300), a single fragment (a dot-disconnected
    salt is out of scope), and the cation N satisfies the quaternary shape
    (symbol N, formal charge +1, 0 H, degree >= 4, via ``classify_cation ==
    'quaternary'``). Pure read-only (AP-21): does NOT mutate ``mol`` / ``features``.

    Registered BELOW LIPID@250 / ZWITTERION@300 (so phosphatidylcholine + betaine
    are reached first — RESEARCH Pitfall 3) and ABOVE CATION_RETAINED@500 (so it
    beats the non-PIN ``tetramethylammonium`` / ``choline`` trivials).
    """
    from orthonym.perception.ions import detect_species_type, get_ion_sites
    if detect_species_type(mol) != 'ion':
        return False
    if '.' in (canonical_smiles or ''):
        return False  # multi-fragment salt -> out of scope (salts.py owns it)
    sites = get_ion_sites(mol)
    if len(sites['cations']) != 1 or sites['anions']:
        return False
    from orthonym.rules.ions import classify_cation
    return classify_cation(mol, sites['cations'][0]) == 'quaternary'


def _is_anion_small(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """Tier-1; mirrors namer.py:877-914 (audit § 1 row 6; § 2.6 purity proof).

    169.6-04 (Task 3): the ``<= 25 HA`` size-cutoff band-aid was REMOVED — it
    excluded large single anions from this handler so their charge was dropped
    downstream. route_charged (reached via the handler -> name_anion delegation)
    now names large anions structurally (neutralize -> re-enter -> -oate/-olate/
    -sulfonate), so the predicate matches ANY single anion (no upper HA bound).
    The class name ``anion_small`` is retained (it is the dispatch enum key).
    """
    from orthonym.perception.ions import detect_species_type, get_ion_sites
    if detect_species_type(mol) != 'ion':
        return False
    sites = get_ion_sites(mol)
    return (
        len(sites['anions']) == 1
        and not sites['cations']
    )


def _is_poly_anion(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """Tier-1; mirrors namer.py:916-956 (audit § 1 row 7; § 2.7 purity proof)."""
    from orthonym.perception.ions import detect_species_type, get_ion_sites
    if detect_species_type(mol) != 'ion':
        return False
    sites = get_ion_sites(mol)
    return len(sites['anions']) >= 2 and not sites['cations']


def _is_multi_component_neutral(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """Tier-1; mirrors namer.py:967-998 (audit § 1 row 8; § 2.8 purity proof)."""
    from orthonym.perception.ions import detect_species_type
    if not ('.' in canonical_smiles and detect_species_type(mol) == 'neutral'):
        return False
    frags = canonical_smiles.split('.')
    if len(frags) < 2:
        return False
    multi_atom_count = 0
    for frag_smi in frags:
        frag_mol = Chem.MolFromSmiles(frag_smi)
        if frag_mol is not None and frag_mol.GetNumHeavyAtoms() >= 2:
            multi_atom_count += 1
    return multi_atom_count >= 2


def _is_multiplicative(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """Tier-2; predicate IS handler (audit § 1 row 9; § 2.9 purity proof).

    ``name_multiplicative`` is invoked once here (predicate) and once in the
    handler shim — the v18 cascade at namer.py:1040-1043 also calls it twice
    in effect (once via the truthy check, once via assigning the return). The
    cost is paid by Tier-2 perception amortization per CONTEXT D-07.
    """
    from orthonym.rules.multiplicative import name_multiplicative
    return name_multiplicative(mol) is not None


def _is_carbohydrate_lookup(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """Tier-2; mirrors namer.py:1051-1062 (audit § 1 row 10; § 2.10 purity proof).

    Phase 183 (D-03/D-13): the carbohydrate tier (priority 1000, above the
    oxane/heterocycle handler) now fires for THREE recognized-sugar cases, gated
    narrowly so it fires ONLY where a sugar is recognized:

    1. cataloged sugars (``lookup_sugar`` is not None — unchanged fast-path);
    2. a clean/decorated SINGLE sugar ring the systematic-mono engine can name
       (``name_monosaccharide_systematic`` is not None — deoxy/amino/uronic, D-01);
    3. >= 2 LINKED recognized sugar rings the disaccharide assembler can name
       (``oligosaccharides.name_disaccharide`` via ``_classify_units`` is not None,
       D-02).

    Every cataloged sugar, Phase-176 simple glycoside, cataloged amino sugar, and
    non-sugar molecule stays byte-identical: clauses 2/3 only ADD firing on
    structures the catalog misses; the handler still falls through (None ->
    cascade-continue) when no engine produces a name. ``classify_compound_class``
    is the cheap pre-gate (sugar-ring SMARTS) that bounds the cost of the
    structural recognizers.
    """
    from orthonym.namer import classify_compound_class
    if classify_compound_class(mol, canonical_smiles) != 'carbohydrate':
        return False
    from orthonym.data.sugar_names import (
        lookup_sugar,
        name_monosaccharide_systematic,
    )
    if lookup_sugar(canonical_smiles) is not None:
        return True
    if name_monosaccharide_systematic(mol) is not None:
        return True
    from orthonym.rules.oligosaccharides import _classify_units
    return _classify_units(mol) is not None


def _is_lipid(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """Tier-1; predicate IS handler (Phase 180 P-107). PURE: no mol mutation beyond
    the idempotent CIP prop that the deriver owns (D-26). RESOLVED A1: registered
    Tier-1 @250 so phosphatidylcholine (zwitterion) is reached before ZWITTERION@300,
    which returns '' (not None) and would otherwise terminate the cascade."""
    if mol is None:
        return False
    from orthonym.perception.lipids import detect_lipid_backbone
    return detect_lipid_backbone(mol) is not None


def _handle_lipid(mol, smiles, canonical_smiles, features=None, *,
                  style: str = "pin", **kwargs) -> Optional[str]:
    """Routes a clean lipid backbone to rules.lipids.name_lipid; None → cascade-continue."""
    from orthonym.rules.lipids import name_lipid
    return name_lipid(mol, style=style)


def _is_natural_product(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """Tier-2; predicate IS handler (audit § 1 row 11; § 2.11 purity proof)."""
    from orthonym.rules.natural_products import name_natural_product
    return name_natural_product(mol) is not None


def _is_peptide(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """Tier-2; mirrors namer.py:1078-1083 (audit § 1 row 12; § 2.12 purity proof)."""
    from orthonym.rules.amino_acids import is_peptide
    return is_peptide(mol)


def _is_retained_name(mol, smiles, canonical_smiles, features=None, *, _style: str = "pin", **kwargs) -> bool:
    """Tier-1 (dict lookup); mirrors namer.py:1085-1088 (audit § 1 row 13; § 2.13 proof).

    Style is threaded by the dispatcher (caller passes ``_style=self.style``
    through ``dispatch(...)`` kwargs). The predicate returns False when the
    instance is in ``"systematic"`` style — preserving the v18 gate at
    namer.py:1086.
    """
    if _style == "systematic":
        return False
    from orthonym.data import ALL_RETAINED_NAMES
    return canonical_smiles in ALL_RETAINED_NAMES


def _is_amino_acid(mol, smiles, canonical_smiles, features=None, *, _style: str = "pin", **kwargs) -> bool:
    """Tier-2; predicate IS handler (audit § 1 row 14; § 2.14 purity proof)."""
    if _style == "systematic":
        return False
    from orthonym.rules.amino_acids import name_amino_acid
    return name_amino_acid(mol, canonical_smiles) is not None


def _is_skeletal_replacement(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """Tier-2; predicate IS handler (audit § 1 row 15; § 2.15 purity proof)."""
    from orthonym.rules.skeletal_replacement import try_skeletal_replacement_name
    return try_skeletal_replacement_name(mol) is not None


def _is_cyclophane(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """Tier-2; predicate IS handler (audit § 1 row 16; § 2.16 purity proof)."""
    from orthonym.rules.phane import name_cyclophane
    return name_cyclophane(mol) is not None


def _is_decomposition_pre_general(mol, smiles, canonical_smiles, features=None, *,
                                  _skip_decomposition: bool = False, **kwargs) -> bool:
    """Tier-2; mirrors namer.py:1118-1122 (audit § 1 row 17; § 2.17 purity proof).

    RL-7 option (b) flag-threading: ``_skip_decomposition`` is forwarded by
    the dispatcher as a kwarg; the predicate reads it directly rather than
    binding ``self`` via ``functools.partial``. The truthy-decompose check
    runs in the HANDLER (not predicate) so the predicate stays cheap-pure
    per AP-4 line-count discipline; on a True predicate the handler may
    still return None (e.g., decomposition produced no result), at which
    point ``_name_impl`` falls through to GENERAL via the next dispatch
    iteration in the v18 cascade equivalent.
    """
    if _skip_decomposition:
        return False
    # AP-4 hygiene: do not run try_decompose() in the predicate body. The
    # handler shim runs it once and returns its result; if None, the GENERAL
    # entry's catch-all fires. This is byte-identical to the v18 cascade at
    # namer.py:1118-1122 because in v18 the truthy-check ALSO ran the
    # decomposition once and returned its result; the cascade fell through
    # only on None.
    return True


def _is_general(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """Tier-2 (catch-all); always True per CONTEXT D-08 (audit § 1 row 18; § 2.18 proof).

    Per AP-1 + CFR-02: explicit catch-all at priority 99999 means NO silent
    fallthrough is possible by construction.
    """
    return True


def _is_organometallic(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """Tier-1; CFR priority 50; ORGM-03 + D-12 purity (audit § 2.20).

    PURE per CONTEXT D-12 + Phase 158 D-26 hard invariant:
    - mol is None guard early-exit.
    - Lazy-import detect_metal_complex from perception.metals.
    - Returns True iff detect_metal_complex(mol) is not None.
    - NO mol mutation; NO features mutation; NO module-global state R/W;
      NO exception swallowing (RDKit exceptions propagate).
    """
    if mol is None:
        return False
    from orthonym.perception.metals import detect_metal_complex
    return detect_metal_complex(mol) is not None


# ---------------------------------------------------------------------------
# Section 6: Per-class handler shims (audit § 1; AP-5 1-line-wrapper discipline)
#
# The ANION_SMALL / POLY_ANION / MULTI_COMPONENT_NEUTRAL / DECOMPOSITION shims
# preserve the v18 byte-identical neutralize-recurse + try/except logic
# verbatim per audit § 3 (RL-3 / RL-4 invariants) — the "logic" they contain
# is byte-identical replicated v18 code, NOT new invented logic. AP-5 forbids
# inventing naming logic in shims; preserving v18 verbatim is allowed.
# ---------------------------------------------------------------------------


def _handle_salt(mol, smiles, canonical_smiles, features=None, *,
                 style: str = "pin", **kwargs) -> Optional[str]:
    """Mirrors namer.py:853-854."""
    from orthonym.rules.salts import name_salt
    return name_salt(mol, style=style)


def _handle_radical(mol, smiles, canonical_smiles, features=None, *,
                    style: str = "pin", **kwargs) -> Optional[str]:
    """Mirrors namer.py:856-857.

    169.6-03 (CHOKE-01): delegate to the route_charged chokepoint FIRST (it owns
    the alkyl/-ylidene/-ylidyne radical naming now that radicals.py carbon-counting
    is deleted); fall through to name_radical (which keeps the acyl/oxyl/aryl
    structured helpers + the retained-name lookup) on ''.
    """
    from orthonym.rules.charged_router import route_charged
    routed = route_charged(mol, style)
    if routed:
        return routed
    from orthonym.rules.radicals import name_radical
    return name_radical(mol, style=style)


def _handle_zwitterion(mol, smiles, canonical_smiles, features=None, *,
                       style: str = "pin", **kwargs) -> Optional[str]:
    """Mirrors namer.py:859-860."""
    from orthonym.rules.salts import name_zwitterion
    return name_zwitterion(mol, style=style)


def _handle_anion_retained(mol, smiles, canonical_smiles, features=None, *,
                           style: str = "pin", **kwargs) -> Optional[str]:
    """Mirrors namer.py:868-871."""
    from orthonym.rules.ions import name_anion
    return name_anion(mol, style=style, retained_only=True)


def _handle_cation_retained(mol, smiles, canonical_smiles, features=None, *,
                            style: str = "pin", **kwargs) -> Optional[str]:
    """Mirrors namer.py:872-875."""
    from orthonym.rules.ions import name_cation
    return name_cation(mol, style=style, retained_only=True)


def _handle_cation_quaternary(mol, smiles, canonical_smiles, features=None, *,
                              style: str = "pin", **kwargs) -> Optional[str]:
    """Phase 184 WS-E.1 thin shim: the parent decision + ``-aminium`` assembly is
    owned by the ``route_charged`` chokepoint (which dispatches the 'quaternary'
    cation kind to ``ions.name_quaternary_aminium`` and applies the mono-cation
    OPSIN RT-gate backstop). Returns None on no-match so the dispatcher falls
    through to CATION_RETAINED@500 (the legacy trivial)."""
    from orthonym.rules.charged_router import route_charged
    return route_charged(mol, style) or None


def _handle_anion_small(mol, smiles, canonical_smiles, features=None, *,
                        style: str = "pin", **kwargs) -> Optional[str]:
    """Mirrors namer.py:884-913 — dual-path with neutralize-recurse fallback.

    Per audit § 3.1 RL-4 fresh-instance pattern: the recursive
    ``Orthonym(style).name(neutral_smi)`` call hits a FRESH router with
    empty dispatch_stats. Returns None on no-match; the dispatcher falls
    through to the next entry (v18 byte-identical to namer.py:913-914).

    169.6-03 (CHOKE-01): the parent decision is owned by the route_charged
    chokepoint (reached via name_anion's single-anion delegation below); this
    handler stays a thin shim over name_anion + the v18 neutralize-recurse
    fallback.
    """
    from orthonym.rules.charged_router import route_charged
    routed = route_charged(mol, style)
    if routed:
        return routed
    from orthonym.rules.ions import name_anion
    ion_result = name_anion(mol, style=style)
    if ion_result:
        return ion_result
    # Ion pipeline returned empty -- try neutralize-then-name (audit § 3.1)
    try:
        from rdkit.Chem import RWMol
        from orthonym.rules.ions import classify_anion, _acid_name_to_carboxylate
        from orthonym.perception.ions import _get_internal_charge_atoms, get_ion_sites
        sites = get_ion_sites(mol)
        anion_type = classify_anion(mol, sites['anions'][0])
        if anion_type == 'carboxylate':
            internal_charge_atoms = _get_internal_charge_atoms(mol)
            rwmol = RWMol(mol)
            for atom in rwmol.GetAtoms():
                if (atom.GetSymbol() == 'O'
                        and atom.GetFormalCharge() == -1
                        and atom.GetIdx() not in internal_charge_atoms):
                    atom.SetFormalCharge(0)
                    atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
            neutral_mol = rwmol.GetMol()
            neutral_smi = Chem.MolToSmiles(neutral_mol, canonical=True)
            from orthonym.namer import Orthonym
            # SUB-03 (169.5): neutral name is an INTERMEDIATE (ionized below) —
            # bypass the validity gate so a malformed intermediate isn't
            # suppressed to a descriptive string before the ionize step.
            neutral_namer = Orthonym(style=style, _disable_opsin_validity_gate=True)
            neutral_name = neutral_namer.name(neutral_smi)
            if neutral_name:
                anion_name = _acid_name_to_carboxylate(neutral_name, 1)
                if anion_name:
                    return anion_name
    except (ValueError, RuntimeError, KeyError, IndexError, RecursionError) as exc:
        # Expected operational fall-through (edge molecule / recursion limit),
        # matching ions.py's explicit lists. Logged, not silent.
        logger.debug("anion_small handler fell through: %s: %s",
                     type(exc).__name__, exc)
    except Exception as exc:
        # WR-07 (code review 2026-06-02): an UNEXPECTED exception here used to be
        # silently swallowed (bare ``except Exception: pass``), masking real logic
        # bugs (e.g. an AttributeError or a malformed SuffixInfo from the SUB-01
        # routing change) as a quiet cascade-to-GENERAL with zero telemetry.
        # Surface it LOUDLY (WARNING + stack) but still fall through, because
        # name() has no top-level safety net and the v18 contract is no-crash.
        logger.warning("anion_small handler UNEXPECTED error (possible bug): %s: %s",
                       type(exc).__name__, exc, exc_info=True)
    return None  # signal dispatcher to continue cascade


def _handle_poly_anion(mol, smiles, canonical_smiles, features=None, *,
                       style: str = "pin", **kwargs) -> Optional[str]:
    """Mirrors namer.py:921-955 — dianion neutralize-then-name.

    Per audit § 3.2 RL-4 fresh-instance pattern. Returns None on no-match;
    the dispatcher falls through to the next entry (v18 byte-identical to
    namer.py:955-956).

    169.6-03 (CHOKE-01): try the route_charged chokepoint FIRST (it neutralizes
    ALL same-sign centers -> the dicarboxylate/dianion parent bears every ionic
    suffix, GUARD 2); fall through to the v18 oxoacid + carboxylate
    neutralize-recurse body on ''.
    """
    from orthonym.rules.charged_router import route_charged
    routed = route_charged(mol, style)
    if routed:
        return routed
    try:
        from rdkit.Chem import RWMol
        from orthonym.rules.ions import classify_anion, _acid_name_to_carboxylate
        from orthonym.perception.ions import _get_internal_charge_atoms, get_ion_sites
        sites = get_ion_sites(mol)
        # CR-02 (code review 2026-06-02): a fully-deprotonated S/P-oxoacid dianion
        # (e.g. CP(=O)([O-])[O-], -2) reaches this poly-anion handler and MUST NOT
        # ship the neutral acid name (the carboxylate_count check below is 0 for a
        # phosphonate, so it would `return neutral_name` uncharged). Mirror
        # name_anion's multi-anion oxoacid routing: _name_oxoacid_anion neutralizes
        # ALL non-internal [O-] and applies the canonical ionic suffix
        # ("methanephosphonic acid" -> "methanephosphonate", which OPSIN
        # round-trips to the -2 dianion). Routed only when EVERY anion is an
        # S/P-oxoacid site; carboxylate / mixed cases fall through unchanged.
        anion_sites = sites.get('anions', [])
        if anion_sites and all(
            classify_anion(mol, a) in ('sulfonate', 'sulfinate', 'phosphonate')
            for a in anion_sites
        ):
            from orthonym.rules.ions import _name_oxoacid_anion, _validate_anion_name
            oxo_name = _name_oxoacid_anion(mol, style)
            if oxo_name:
                return _validate_anion_name(mol, oxo_name)
        internal_charge_atoms = _get_internal_charge_atoms(mol)
        rwmol = RWMol(mol)
        neutralized = False
        for atom in rwmol.GetAtoms():
            if (atom.GetSymbol() == 'O'
                    and atom.GetFormalCharge() == -1
                    and atom.GetIdx() not in internal_charge_atoms):
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
                neutralized = True
        if neutralized:
            neutral_mol = rwmol.GetMol()
            neutral_smi = Chem.MolToSmiles(neutral_mol, canonical=True)
            from orthonym.namer import Orthonym
            # SUB-03 (169.5): neutral name is an INTERMEDIATE (ionized below) —
            # bypass the validity gate so a malformed intermediate isn't
            # suppressed to a descriptive string before the ionize step.
            neutral_namer = Orthonym(style=style, _disable_opsin_validity_gate=True)
            neutral_name = neutral_namer.name(neutral_smi)
            if neutral_name:
                # IUPAC P-72.2.1: acid suffix -> carboxylate for deprotonated sites
                carboxylate_count = sum(
                    1 for a in sites['anions']
                    if classify_anion(mol, a) == 'carboxylate'
                )
                if carboxylate_count > 0:
                    anion_name = _acid_name_to_carboxylate(
                        neutral_name, carboxylate_count
                    )
                    if anion_name:
                        return anion_name
                return neutral_name
    except (ValueError, RuntimeError, KeyError, IndexError, RecursionError) as exc:
        # Expected operational fall-through, matching ions.py's explicit lists.
        logger.debug("poly_anion handler fell through: %s: %s",
                     type(exc).__name__, exc)
    except Exception as exc:
        # WR-07 (code review 2026-06-02): surface a genuinely-unexpected exception
        # loudly instead of the old silent ``except Exception: pass`` that masked
        # SUB-01-routing logic bugs as a quiet cascade-to-GENERAL. Still falls
        # through (no top-level safety net in name(); v18 contract is no-crash).
        logger.warning("poly_anion handler UNEXPECTED error (possible bug): %s: %s",
                       type(exc).__name__, exc, exc_info=True)
    return None


def _handle_multi_component_neutral(mol, smiles, canonical_smiles, features=None, *,
                                    style: str = "pin", **kwargs) -> Optional[str]:
    """Mirrors namer.py:967-996 — per-component recursive naming.

    Per audit § 3.3 RL-4 fresh-instance pattern: each fragment's
    ``Orthonym(style).name(frag_smi)`` call hits a FRESH router. Returns
    None when no fragment names produce content (v18 falls through to
    normal pipeline).
    """
    frags = canonical_smiles.split('.')
    frag_mols = []
    for frag_smi in frags:
        frag_mol = Chem.MolFromSmiles(frag_smi)
        if frag_mol:
            ha = frag_mol.GetNumHeavyAtoms()
            frag_mols.append((frag_smi, ha))
    multi_atom_frags = [(s, ha) for s, ha in frag_mols if ha >= 2]
    if len(multi_atom_frags) < 2:
        return None
    # Sort by descending heavy atom count for consistent output (v18 line 984)
    frag_mols.sort(key=lambda x: -x[1])
    component_names = []
    for frag_smi, _ in frag_mols:
        try:
            from orthonym.namer import Orthonym
            frag_namer = Orthonym(style=style)
            frag_name = frag_namer.name(frag_smi)
            if frag_name and frag_name != "unknown":
                component_names.append(frag_name)
        except Exception:
            pass  # Skip unnamed fragments (v18 line 992-993)
    if component_names:
        return ' '.join(component_names)
    return None


def _handle_multiplicative(mol, smiles, canonical_smiles, features=None, *,
                           style: str = "pin", **kwargs) -> Optional[str]:
    """Mirrors namer.py:1041-1043."""
    from orthonym.rules.multiplicative import name_multiplicative
    return name_multiplicative(mol)


def _handle_carbohydrate_lookup(mol, smiles, canonical_smiles, features=None, *,
                                style: str = "pin", **kwargs) -> Optional[str]:
    """Catalog-first carbohydrate cascade (Phase 183 D-03 + D-10).

    The v18 cascade at namer.py:1051-1062 inlined a plain lookup-then-join. This
    handler keeps that fast-path FIRST (D-05) and extends it with two new
    engines, behind the broadened predicate :func:`_is_carbohydrate_lookup`:

    1. **Catalog (``lookup_sugar``) FIRST.** If it returns ``(anomer, config,
       base_name)``:

       * **D-10 uronic interception.** The free uronic acid SMILES is itself a
         catalog key (``URONIC_ACID_NAMES`` -> ``ALL_SUGAR_NAMES``), so
         ``lookup_sugar`` returns ``base_name == "glucuronopyranose"``. The plain
         ``"-".join`` would emit the OPSIN-UNPARSEABLE ``beta-D-glucuronopyranose``
         (the validity gate then suppresses it to ``unknown``). So when
         ``"urono" in base_name`` we route to
         :func:`~orthonym.data.sugar_names.uronic_free_acid_name`, which emits
         the P-102.5.6.6.4 free-acid form ``beta-D-glucopyranuronic acid``.
         A ``None`` from it cascade-continues (fail-closed, D-11).
       * **Otherwise** the existing plain ``"-".join(parts)`` is UNCHANGED, so
         every non-uronic cataloged sugar (incl. simple glycosides handled
         elsewhere and the cataloged amino sugars) stays byte-identical.

    2. **Systematic monosaccharide (D-01/D-04).** On a catalog miss, try
       :func:`~orthonym.data.sugar_names.name_monosaccharide_systematic` so a
       free deoxy/amino/uronic single sugar ring names systematically
       (``6-deoxy-beta-D-glucopyranose``) rather than as a substituted oxane.

    3. **Disaccharide / oligosaccharide (D-02).** On a further miss, try
       :func:`~orthonym.rules.oligosaccharides.name_disaccharide` for the
       P-102.7 glycosyl-glycoside / glycosylglycose form.

    The first non-None wins; otherwise ``None`` (cascade-continue -> the existing
    pipeline names it byte-identically). All imports are lazy/in-function.
    """
    from orthonym.data.sugar_names import lookup_sugar
    sugar_info = lookup_sugar(canonical_smiles)
    if sugar_info is not None:
        anomer, config, base_name = sugar_info
        # D-10: a cataloged uronic base must emit the OPSIN-parseable free-acid
        # form, not the unparseable "...uronopyranose" join.
        if "urono" in base_name:
            from orthonym.data.sugar_names import uronic_free_acid_name
            free_acid = uronic_free_acid_name(anomer, config, base_name)
            if free_acid is not None:
                return free_acid
            # uronic_free_acid_name fail-closed -> cascade-continue (D-11).
            return None
        parts = []
        if anomer:
            parts.append(anomer)
        if config:
            parts.append(config)
        parts.append(base_name)
        return "-".join(parts)

    # Catalog miss -> systematic monosaccharide (D-01), then disaccharide (D-02).
    from orthonym.data.sugar_names import name_monosaccharide_systematic
    mono = name_monosaccharide_systematic(mol)
    if mono is not None:
        return mono
    from orthonym.rules.oligosaccharides import name_disaccharide
    return name_disaccharide(mol)


def _handle_natural_product(mol, smiles, canonical_smiles, features=None, *,
                            style: str = "pin", **kwargs) -> Optional[str]:
    """Mirrors namer.py:1071-1073."""
    from orthonym.rules.natural_products import name_natural_product
    return name_natural_product(mol)


def _handle_peptide(mol, smiles, canonical_smiles, features=None, *,
                    style: str = "pin", **kwargs) -> Optional[str]:
    """Mirrors namer.py:1080-1083."""
    from orthonym.rules.peptides import name_peptide
    return name_peptide(mol)


def _handle_retained_name(mol, smiles, canonical_smiles, features=None, *,
                          style: str = "pin", **kwargs) -> Optional[str]:
    """Mirrors namer.py:1086-1088 — direct dict lookup."""
    from orthonym.data import ALL_RETAINED_NAMES
    return ALL_RETAINED_NAMES.get(canonical_smiles)


def _handle_amino_acid(mol, smiles, canonical_smiles, features=None, *,
                       style: str = "pin", **kwargs) -> Optional[str]:
    """Mirrors namer.py:1091-1094."""
    from orthonym.rules.amino_acids import name_amino_acid
    return name_amino_acid(mol, canonical_smiles)


def _handle_skeletal_replacement(mol, smiles, canonical_smiles, features=None, *,
                                 style: str = "pin", **kwargs) -> Optional[str]:
    """Mirrors namer.py:1100-1102."""
    from orthonym.rules.skeletal_replacement import try_skeletal_replacement_name
    return try_skeletal_replacement_name(mol)


def _handle_cyclophane(mol, smiles, canonical_smiles, features=None, *,
                       style: str = "pin", **kwargs) -> Optional[str]:
    """Mirrors namer.py:1111-1113."""
    from orthonym.rules.phane import name_cyclophane
    return name_cyclophane(mol)


def _handle_decomposition_pre_general(mol, smiles, canonical_smiles, features=None, *,
                                       style: str = "pin",
                                       _skip_decomposition: bool = False,
                                       **kwargs) -> Optional[str]:
    """Mirrors namer.py:1118-1122 — bond-cleavage decomposition.

    Returns None on decomposition-failed; the dispatcher falls through to
    GENERAL (v18 byte-identical to namer.py:1118-1122 — the v18 cascade
    falls through to the perceive/classify/assemble path on None).
    """
    if _skip_decomposition:
        return None
    from orthonym.decomposition import try_decompose
    return try_decompose(mol, style=style)


def _handle_general(mol, smiles, canonical_smiles, features=None, *,
                    style: str = "pin", **kwargs) -> Optional[str]:
    """GENERAL pipeline catch-all per CONTEXT D-08.

    Returns None to signal that ``_name_impl`` runs the legacy
    ``_perceive -> _classify -> assemble_name`` pipeline INLINE (Task
    158-02-01 design choice (a)). This decouples ``routing/`` from
    ``composer.py`` (D-19 boundary respect). The ``_name_impl`` body
    detects ``result.class_id == StoutClass.GENERAL and name is None``
    and runs the legacy pipeline directly.
    """
    return None


def _handle_organometallic(mol, smiles, canonical_smiles, features=None, *,
                           style: str = "pin", **kwargs) -> Optional[str]:
    """Mirrors handlers.organometallic.name_organometallic (audit § 1 row ORGM).

    Cascade-continuation on None per CONTEXT D-02: if name_organometallic
    returns None (compound not actually ORGM or cannot be named), the CFR
    cascade falls through to SALT@100 → ... → GENERAL@99999.
    """
    from orthonym.assembly.handlers.organometallic import name_organometallic
    result = name_organometallic(features, mol, style=style)
    return result.name if result is not None else None


# ---------------------------------------------------------------------------
# Section 7: _register_dispatch(...) calls — audit § 1 1:1 translation
#
# Priorities are spaced in 100s per CONTEXT D-06 to leave room for v19
# sibling-phase insertions. The cascade order mirrors namer.py:852-1131
# byte-identical.
# ---------------------------------------------------------------------------

# --- Phase 161: ORGANOMETALLIC at priority 50 (intercepts BEFORE SALT@100) ---
# Per CONTEXT D-02: ferrocene/ruthenocene/all sandwich complexes are
# dot-separated [M+n].[ligand-]...[ligand-] SMILES that would otherwise
# route to SALT@100 and produce nonsense (verified empirically:
# Orthonym().name('[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1') returns
# 'iron(II) dipentanide' today; Orthonym().name('C[Li]') returns
# 'methane'; etc. per RESEARCH executive summary line 89).
# Priority 50 sits BELOW the prior CFR minimum (SALT@100); free per Phase 158
# audit log. Cascade-continuation on None preserved per D-02.
_register_dispatch(
    class_id=StoutClass.ORGANOMETALLIC, priority=50, tier=1,
    predicate=_is_organometallic, handler=_handle_organometallic,
    iupac_section="Blue Book P-69 + IR-10 + Salzer 1999",
    description="Organometallic complex (metal-carbon direct bond or sandwich/half-sandwich); routes to handlers.organometallic.name_organometallic",
    side_effect_inventory=(),
)

# --- Tier-1 charged-species + dot-disconnected (audit § 1 rows 1-8) ---
_register_dispatch(
    class_id=StoutClass.SALT, priority=100, tier=1,
    predicate=_is_salt, handler=_handle_salt,
    iupac_section="impl routing — salt construction P-65.6.2.1/P-63.8.1/P-77 (cation word(s) + anion); ion names P-72/P-73 (D-09: corrected 169.6-04 from the wrong conjunctive-nomenclature cite)",
    description="Multi-component salt; routes to rules.salts.name_salt",
    side_effect_inventory=(),
)
_register_dispatch(
    class_id=StoutClass.RADICAL, priority=200, tier=1,
    predicate=_is_radical, handler=_handle_radical,
    iupac_section="impl routing — radical-electron detection per Blue Book P-15.7",
    description="Single radical species; routes to rules.radicals.name_radical",
    side_effect_inventory=(),
)
_register_dispatch(
    class_id=StoutClass.LIPID, priority=250, tier=1,
    predicate=_is_lipid, handler=_handle_lipid,
    iupac_section="Blue Book P-107 lipids (glycerides P-107.2 / phosphatidic acids P-107.3 / glycolipids P-107.4)",
    description="Lipid backbone (glycerol / glycero-phospho-X / sphingoid); routes to rules.lipids.name_lipid",
    side_effect_inventory=(),
)
_register_dispatch(
    class_id=StoutClass.ZWITTERION, priority=300, tier=1,
    predicate=_is_zwitterion, handler=_handle_zwitterion,
    iupac_section="impl routing — formal-charge>0 zwitterion detection per Blue Book P-74",
    description="Formal-charge zwitterion; routes to rules.salts.name_zwitterion",
    side_effect_inventory=(),
)
_register_dispatch(
    class_id=StoutClass.ANION_RETAINED, priority=400, tier=1,
    predicate=_is_anion_retained, handler=_handle_anion_retained,
    iupac_section="Blue Book P-72 retained anion names (acetate, benzoate, etc.)",
    description="Retained anion name lookup; routes to rules.ions.name_anion(retained_only=True)",
    side_effect_inventory=(),
)
_register_dispatch(
    class_id=StoutClass.CATION_QUATERNARY, priority=480, tier=1,
    predicate=_is_cation_quaternary, handler=_handle_cation_quaternary,
    iupac_section="Blue Book P-73.1.2.1 + Table 7.4 quaternary ammonium -> systematic -aminium PIN",
    description="Standalone quaternary cation (single N+, 0 H, degree>=4, no anion, single fragment); routes to rules.charged_router.route_charged",
    side_effect_inventory=(),
)
_register_dispatch(
    class_id=StoutClass.CATION_RETAINED, priority=500, tier=1,
    predicate=_is_cation_retained, handler=_handle_cation_retained,
    iupac_section="Blue Book P-73 retained cation names",
    description="Retained cation name lookup; routes to rules.ions.name_cation(retained_only=True)",
    side_effect_inventory=(),
)
_register_dispatch(
    class_id=StoutClass.ANION_SMALL, priority=600, tier=1,
    predicate=_is_anion_small, handler=_handle_anion_small,
    iupac_section="Blue Book P-72.2.1 carboxylate -oate suffix",
    description="Single-anion ≤25 HA; ion pipeline + neutralize-recurse fallback",
    side_effect_inventory=(),
)
_register_dispatch(
    class_id=StoutClass.POLY_ANION, priority=700, tier=1,
    predicate=_is_poly_anion, handler=_handle_poly_anion,
    iupac_section="Blue Book P-72.2.1 polycarboxylate -oate suffix",
    description="Poly-anion (≥2 anionic sites); RWMol-neutralize + recursive name",
    side_effect_inventory=(),
)
_register_dispatch(
    class_id=StoutClass.MULTI_COMPONENT_NEUTRAL, priority=800, tier=1,
    predicate=_is_multi_component_neutral, handler=_handle_multi_component_neutral,
    iupac_section="impl routing — dot-disconnected neutral SMILES with ≥2 multi-atom fragments",
    description="Cocrystal/solvate; per-component recursive naming; ' '.join",
    side_effect_inventory=(),
)

# --- Tier-2 multiplicative + class-routed handlers (audit § 1 rows 9-16) ---
_register_dispatch(
    class_id=StoutClass.MULTIPLICATIVE, priority=900, tier=2,
    predicate=_is_multiplicative, handler=_handle_multiplicative,
    iupac_section="Blue Book P-51.3 multiplicative nomenclature",
    description="Symmetric bridged dimers; routes to rules.multiplicative.name_multiplicative",
    side_effect_inventory=(),
)
_register_dispatch(
    class_id=StoutClass.CARBOHYDRATE_LOOKUP, priority=1000, tier=2,
    predicate=_is_carbohydrate_lookup, handler=_handle_carbohydrate_lookup,
    iupac_section="Blue Book P-10 retained carbohydrate names",
    description="Carbohydrate lookup via data.sugar_names.lookup_sugar; assembles anomer/config/base",
    side_effect_inventory=(),
)
_register_dispatch(
    class_id=StoutClass.NATURAL_PRODUCT, priority=1100, tier=2,
    predicate=_is_natural_product, handler=_handle_natural_product,
    iupac_section="Blue Book P-10 natural product retained names",
    description="Natural product (steroid/alkaloid/terpene); routes to rules.natural_products.name_natural_product",
    side_effect_inventory=(),
)
_register_dispatch(
    class_id=StoutClass.PEPTIDE, priority=1200, tier=2,
    predicate=_is_peptide, handler=_handle_peptide,
    iupac_section="Blue Book P-66.6.3 peptide grammar",
    description="Peptide; routes to rules.peptides.name_peptide",
    side_effect_inventory=(),
)
_register_dispatch(
    class_id=StoutClass.RETAINED_NAME, priority=1300, tier=1,
    predicate=_is_retained_name, handler=_handle_retained_name,
    iupac_section="Blue Book retained PIN per P-22 + P-25 retained names catalog",
    description="Retained-name dict lookup; gated on style != 'systematic'",
    side_effect_inventory=(),
)
_register_dispatch(
    class_id=StoutClass.AMINO_ACID, priority=1400, tier=2,
    predicate=_is_amino_acid, handler=_handle_amino_acid,
    iupac_section="Blue Book P-66 amino acid grammar",
    description="Amino acid lookup; routes to rules.amino_acids.name_amino_acid; gated on style != 'systematic'",
    side_effect_inventory=(),
)
_register_dispatch(
    class_id=StoutClass.SKELETAL_REPLACEMENT, priority=1500, tier=2,
    predicate=_is_skeletal_replacement, handler=_handle_skeletal_replacement,
    iupac_section="Blue Book P-15.4 skeletal replacement nomenclature",
    description="Skeletal replacement; routes to rules.skeletal_replacement.try_skeletal_replacement_name",
    side_effect_inventory=(),
)
_register_dispatch(
    class_id=StoutClass.CYCLOPHANE, priority=1600, tier=2,
    predicate=_is_cyclophane, handler=_handle_cyclophane,
    iupac_section="Blue Book P-26.4 phane nomenclature",
    description="Cyclophane topology; routes to rules.phane.name_cyclophane",
    side_effect_inventory=(),
)

# --- Decomposition + GENERAL catch-all (audit § 1 rows 17-18) ---
_register_dispatch(
    class_id=StoutClass.DECOMPOSITION_PRE_GENERAL, priority=99000, tier=2,
    predicate=_is_decomposition_pre_general, handler=_handle_decomposition_pre_general,
    iupac_section="impl routing — bond cleavage; v4.0 Phase 39 decomposition engine",
    description="Decomposition fallback; gated on _skip_decomposition (RL-7 option (b))",
    side_effect_inventory=(),
)
_register_dispatch(
    class_id=StoutClass.GENERAL, priority=99999, tier=2,
    predicate=_is_general, handler=_handle_general,
    iupac_section="impl routing — GENERAL pipeline (composer.assemble_name); D-08 catch-all",
    description="Catch-all per D-08; handler returns None to signal _name_impl runs legacy pipeline",
    side_effect_inventory=(),
)

# --- Lock the table at module-import end (D-05; AP-21 mutation prevention) ---
_REGISTRATION_FROZEN = True
