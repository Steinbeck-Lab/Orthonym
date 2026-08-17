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
    INORGANIC_ACID = "inorganic_acid"             # v22 G2 COV-02 (P-67/P-65.2.1; priority 40, before ORGANOMETALLIC@50)
    ORGANOMETALLIC = "organometallic"             # Phase 161 (P-69; priority 50)
    LIPID = "lipid"                               # Phase 180 (P-107 lipid backbone; priority 250, tier 1 — before ZWITTERION@300, RESOLVED A1)
    MONONUCLEAR_HYDRIDE = "mononuclear_hydride"    # v23 Phase 6/7 (P-68 / P-31.1.4.2 λ-convention + Group-15 As/Sb/Bi; priority 45, between INORGANIC_ACID@40 and ORGANOMETALLIC@50)
    CHALCOGEN_CHAIN = "chalcogen_chain"           # v23 Phase 7 (P-21.2.2 homogeneous chalcogen-chain parent hydrides: trisulfane/trioxidane; priority 46)
    POLYAZANE = "polyazane"                       # v23 Phase 7 (P-68.3.1.1/.3 hydrazine/diazene/triazane/azo; priority 47)
    FREE_HOMONUCLEAR_G14_HYDRIDE = "free_homonuclear_g14_hydride"  # Wave-3 (P-21.2.3/P-68.2.3 disilane/trisilane/digermene; priority 47.7)
    DINUCLEAR_HYDRIDE = "dinuclear_hydride"        # v23 Phase 10 (P-69.5.3 Group-14/Group-15 catenated hydride: germylstibane; priority 48)
    KETENE = "ketene"                               # Wave-2 completion (P-64.2.2.4 ethenone/dibromoethenone; priority 49)
    RING_CHALCOGEN_OXIDE = "ring_chalcogen_oxide"   # Wave-2 completion (P-25.6/P-74.3.1.3 dibenzothiophene 5-oxide/5,5-dioxide; priority 49.5)
    HYDRO_FUSED_PEROXOL = "hydro_fused_peroxol"     # W2E-P1FG (P-63.4.1 1,2,3,4-tetrahydronaphthalene-1-peroxol; priority 49.6)
    THIOIMIDE = "thioimide"                          # W2E-D3 (P-66.1.4.2 acyclic N-H thioimide R-C(=S)-NH-C(=S)-R' -> N-(ethanethioyl)ethanethioamide; priority 49.7 — after HYDRO_FUSED_PEROXOL@49.6, before ORGM@50)
    CYCLIC_POLYESTER = "cyclic_polyester"            # W3-P08 (P-65.6.3.5.3 lactide / cyclic di-/polyester -> 1,4-dioxane-2,5-dione; priority 49.8 — after THIOIMIDE@49.7, before ORGM@50)
    AZINIC_DERIVATIVE = "azinic_derivative"          # Wave-2 completion C (P-61.5.3 ethylideneazinic acid; priority 48.3)
    HETERONE = "heterone"                            # Wave-2 completion C (P-64.4.1 dimethylsilanone/phosphanone; priority 48.4)
    SULFINE = "sulfine"                              # Wave-2 completion C (P-64.4.2 propylidene-lambda4-sulfanone; priority 48.5)
    CUMULATIVE_ZWITTERION = "cumulative_zwitterion"  # W4-I4 (P-74.1.1 same-parent -ium-...-ide on a homogeneous heteroatom chain: 1,2,2,2-tetramethylhydrazin-2-ium-1-ide / triaz-2-en-2-ium-1-ide / dioxidan-2-ium-1-ide; priority 48.35 — before HETERONE@48.4)
    YLIDE = "ylide"                                  # W4-I4 (P-74.2.1.1 onium cation + adjacent carbanion: 2-(trimethylazaniumyl)propan-2-ide / ...phosphaniumyl / dimethyloxidaniumyl / dimethylsulfaniumyl; priority 48.36 — before HETERONE@48.4)
    PSEUDOKETONE_HETERO = "pseudoketone_hetero"      # Wave-2 completion C (P-64.1.2.1(b)/P-64.5.2.2 1-silylethan-1-one; priority 48.6)
    ACYL_CHALCOGENCHAIN_PSEUDOKETONE = "acyl_chalcogenchain_pseudoketone"  # W3-P14 (P-68.4.1.3 acyl on a homogeneous >=3-chalcogen chain: CH3CH2-CO-O-O-OH -> 1-trioxidanylpropan-1-one; priority 48.65 — after PSEUDOKETONE_HETERO@48.6, before HETEROIMINE@48.7)
    LAMBDA5_PHOSPHANIMINE = "lambda5_phosphanimine"  # W4-I4 (P-74.2.1.5 phosphine imide R3X=N-R at lambda5: N-ethyl-P,P,P-triphenyl-lambda5-phosphanimine; priority 48.66 — before HETEROIMINE@48.7)
    HETEROIMINE = "heteroimine"                      # W2E-P1FG (P-62.3.1.3 X=NH -> 1-methylphosphanimine; priority 48.7)
    LAMBDA_SULFANE_IMINE_OXIDE = "lambda_sulfane_imine_oxide"  # W3-P13 (P-68.4.3.3-.8 mononuclear S/Se/Te imine/oxide: sulfimide/sulfoximide/sulfonediimine/sulfur di-/tri-imide; priority 48.75 — after HETEROIMINE@48.7, before KETENE@49)
    POLYCHALCOGEN_OXIDE = "polychalcogen_oxide"      # W3-P13 (P-68.4.3.2 di-/polysulfoxide-sulfone: CH3-S(=O)-S(=O)-CH3 -> 1,2-dimethyl-1lambda4,2lambda4-disulfane-1,2-dione; priority 46.5 — after CHALCOGEN_CHAIN@46, before POLYAZANE@47)
    CATENATED_HYDRIDE = "catenated_hydride"          # Wave-2 completion (P-21.2.3/P-52.1.3 disiloxane/trisiloxane/disilazane; priority 47.5 — after CHALCOGEN_CHAIN@46/POLYAZANE@47, before ORGM@50)
    HETEROCHALCOGEN_ABA = "heterochalcogen_aba"      # W3-P14 (P-68.4.2.1/P-21.2.3.1 pure-chalcogen a[ba]n parent hydride: HS-O-SH -> dithioxane, CH3-S-O-SH -> methyldithioxane; priority 47.55 — after CATENATED_HYDRIDE@47.5, BEFORE SKELETAL_REPLACEMENT@1500 so the preselected dithioxane parent pre-empts the '3-oxa-2,4-dithiapentane' skeletal name)
    HOMONUCLEAR_PNICTOGEN_CHAIN = "homonuclear_pnictogen_chain"  # W3-P14 (P-68.3.2.2 homonuclear Group-15 catenated hydride: PP -> diphosphane, pentaarsane, dibismuthane; priority 47.6 — pnictogen analogue of POLYAZANE@47)
    PNICTOGEN_CARBOXYLIC_ACID = "pnictogen_carboxylic_acid"  # W3-P14 (P-68.3.2.3.1 added-carbon -carboxylic acid on a P/As/Sb parent hydride: H2P-COOH -> phosphanecarboxylic acid; priority 47.65 — ahead of the generic acid namer in GENERAL)
    INOSITOL = "inositol"                          # v23 Phase 12 follow-on (P-104.2.1 cyclitol retained names myo-/scyllo-/.../chiro-inositol; name-exact, OPSIN-unparseable; priority 1700 — above CYCLOPHANE@1600, below DECOMP_PRE_GENERAL@99000; no free dense slot)
    NUCLEOSIDE = "nucleoside"                       # v23 Phase 14 (P-105.2/P-106 decorated nucleosides/nucleotides: 5'-mono/di/tri-phosphate + O-acyl ester; priority 1800 — after RETAINED@1300 so bare nucleosides + AMP/adenylic stay retained; before DECOMP_PRE_GENERAL@99000; strip-and-recognise, OPSIN-RT, fail-closed)
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
    """Tier-1; mirrors namer.py:967-998 (audit § 1 row 8; § 2.8 purity proof).

    Wave-2 P0A (P-14.8): broadened so single-multi-atom + recognized-
    inorganic-single-atom sets (oxalic acid + water; nicotine + HCl) also
    dispatch here. The legacy >=2-multi-atom condition is kept verbatim as
    the first accept. Bare-metal single atoms ([Ni], [Fe]) are NOT
    recognized components -> those inputs keep their current routing
    (ORGANOMETALLIC@50 fires earlier anyway). Charged input never reaches
    here (detect_species_type != 'neutral').
    """
    from orthonym.perception.ions import detect_species_type
    if not ('.' in canonical_smiles and detect_species_type(mol) == 'neutral'):
        return False
    frags = canonical_smiles.split('.')
    if len(frags) < 2:
        return False
    from orthonym.rules.adducts import SINGLE_ATOM_COMPONENT_NAMES
    multi_atom_count = 0
    single_atoms_recognized = True
    for frag_smi in frags:
        frag_mol = Chem.MolFromSmiles(frag_smi)
        if frag_mol is None:
            return False
        if frag_mol.GetNumHeavyAtoms() >= 2:
            multi_atom_count += 1
        elif (Chem.MolToSmiles(frag_mol, canonical=True)
              not in SINGLE_ATOM_COMPONENT_NAMES):
            single_atoms_recognized = False
    if multi_atom_count >= 2:
        return True  # legacy condition — byte-identical accept set
    # P0A extension: exactly the P-14.8 solvate/hydrate/hydracid shape
    return (multi_atom_count >= 1 and single_atoms_recognized
            and len(set(frags)) >= 2)


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
    is_carb = classify_compound_class(mol, canonical_smiles) == 'carbohydrate'
    from orthonym.data.sugar_names import (
        lookup_sugar,
        name_monosaccharide_systematic,
    )
    if is_carb:
        if lookup_sugar(canonical_smiles) is not None:
            return True
        if name_monosaccharide_systematic(mol) is not None:
            return True
    # W6 structural detectors (cheap RDKit ring-walks, NO OPSIN — the handler
    # RT-gates every name).  These run EVEN when the sugar-ring SMARTS pre-gate
    # misses: a fully-O-methylated ring has no free ring-OH for the SMARTS, yet is
    # a bona-fide O-methyl sugar.  Each fast-fails on a non-sugar ring shape.
    from orthonym.data.sugar_names import (
        _find_sugar_oxoacid_ester,     # W6-P1 phosphate/sulfate ester
        _find_sugar_acyl_esters,       # W6B-T2 O-acyl ester (acetate/benzoate/...)
        _has_anomeric_hetero_sugar,    # W6-P2 glycosylamine / glycosyl halide
        _has_o_methyl_sugar,           # W6-P3 O-methyl ether
    )
    if _find_sugar_oxoacid_ester(mol) is not None:
        return True
    if _find_sugar_acyl_esters(mol) is not None:
        return True
    # W8-P7b.4 C-substituted sugar (n-C-R / n-deoxy-n-R): a ring carbon bears an
    # extra C/halogen substituent, so the clean-sugar SMARTS pre-gate misses it.
    from orthonym.data.sugar_names import _is_c_substituted_sugar_shape
    if _is_c_substituted_sugar_shape(mol):
        return True
    # W6B-T7 fail-closed veto: claim an uncataloged 2-ulosonic acid so the handler
    # can refuse it (the general ring-carboxylic namer drops its side-chain stereo).
    from orthonym.data.sugar_names import _is_uncataloged_ulosonic_acid
    if _is_uncataloged_ulosonic_acid(mol, canonical_smiles):
        return True
    # W6B-T8 open-chain aldonate/aldarate ester (acyclic sugar-acid ester).
    from orthonym.data.sugar_names import _has_open_chain_acid_ester
    if _has_open_chain_acid_ester(mol):
        return True
    # W6B-T9 N-alkylamino open-chain aldose.
    from orthonym.data.sugar_names import _has_open_chain_amino_aldose
    if _has_open_chain_amino_aldose(mol):
        return True
    # W6B-T11 linear reducing oligosaccharide (3+ units / 1->6 links the binary
    # disaccharide SMARTS misses) -> the handler's name_disaccharide falls back to
    # name_linear_oligosaccharide (RT-gated).
    from orthonym.rules.oligosaccharides import _has_oligo_chain, _has_extended_oligo
    if _has_oligo_chain(mol):
        return True
    # v33 glyco slices 1-2: NON-REDUCING (raffinose) and BRANCHED oligosaccharides
    # -> the handler's name_disaccharide routes to name_nonreducing/branched (RT-gated).
    if _has_extended_oligo(mol):
        return True
    # W6B-T12 glycosyloxy on a senior aglycone (sugar O-linked to a non-sugar
    # aglycone bearing a group senior to hydroxy).
    from orthonym.data.sugar_names import _has_glycosyloxy_aglycone
    if _has_glycosyloxy_aglycone(mol):
        return True
    # W6B-T13 C-glycosyl on a senior aglycone (sugar C-C linked).
    from orthonym.data.sugar_names import _has_c_glycosyl_aglycone
    if _has_c_glycosyl_aglycone(mol):
        return True
    # W6B-T13 fail-closed veto: a sugar with a non-methyl/non-acyl ring-O ether
    # (the n-O-yl shape) whose sugar the general namer would drop.
    from orthonym.data.sugar_names import _is_sugar_o_ether_leak
    if _is_sugar_o_ether_leak(mol):
        return True
    if _has_anomeric_hetero_sugar(mol):
        return True
    if _has_o_methyl_sugar(mol):
        return True
    if is_carb:
        from orthonym.rules.oligosaccharides import _classify_units
        return _classify_units(mol) is not None
    return False


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


def _is_inositol(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """v23 Phase 12 follow-on; hard-gated cyclitol recognizer (P-104.2.1). Fires
    for the seven meso inositols (named) AND the chiral chiro pair (refused — see
    _handle_inositol); both are detected order-stably from the cyclohexanehexol
    skeleton + stereo layer. The undefined-stereo hexol does NOT match (keeps the
    systematic name)."""
    from orthonym.rules.inositols import name_inositol, is_chiral_inositol
    return name_inositol(mol) is not None or is_chiral_inositol(mol)


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
    # W3-P07 (P-65.6.3.1.2 / P-65.6.3.4): a PSEUDOESTER R-CO-O-Z (Z a Group-13/14/15
    # organyl, e.g. CH3-CO-O-Si(CH3)3) is named as an ESTER ('trimethylsilyl
    # acetate'), NOT as a P-69 organometallic parent hydride. The ester is the
    # senior characteristic group; ORGM@50 would otherwise claim the Si FIRST and
    # emit a garbage silane ('(acetaldehydoxy)tri(methyl)silane', OPSIN-suppressed
    # to 'unknown'). Decline so the pseudoester principal-group routes to the
    # noncarbon_ester handler. Read-only (D-12/D-26 purity preserved).
    from orthonym.perception.functional_groups import detect_functional_groups
    if detect_functional_groups(mol).get('pseudoester'):
        return False
    # F-T6 (DD3): a single-anion ion centred on a Group-13/14 metalloid is a
    # charged parent-hydride anion owned by the ANION_SMALL -> route_charged
    # emitter path, NOT the neutral P-69 organometallic handler (priority 50) which
    # would otherwise claim it FIRST and DROP the charge (`C[Si-](C)C` ->
    # 'trimethylsilane' instead of 'trimethylsilanide'; `C[B-](C)(C)C` -> 'unknown'
    # instead of 'tetramethylboranuide'). Decline here so the charged path wins:
    #   - 'heteroatom_hydride_anion' (P/As/Sb/Si/Ge/Sn/Pb) -> the -ide path;
    #   - 'uide_anion'               (B/Si/P/… ate-complex) -> the -uide path.
    # PURE: only read-only perception/classification calls, no mutation (D-12/D-26).
    from orthonym.perception.ions import detect_species_type, get_ion_sites
    if detect_species_type(mol) == 'ion':
        _sites = get_ion_sites(mol)
        if len(_sites.get('anions', [])) == 1 and not _sites.get('cations'):
            from orthonym.rules.ions import classify_anion
            if classify_anion(mol, _sites['anions'][0]) in (
                    'heteroatom_hydride_anion', 'uide_anion'):
                return False
    # v23 Phase 7 (7a): a neutral mononuclear Group-15 (As/Sb/Bi) parent hydride
    # — trimethylarsane / triphenylarsane / arsane / trichloroarsane — is named
    # SUBSTITUTIVELY (P-68.3), NOT by P-69 organometallic nomenclature. The
    # MONONUCLEAR_HYDRIDE handler@45 already intercepts these before ORGM@50, but
    # decline here too so the classification is semantically honest and the
    # gate-OFF path is correct (ORGM otherwise mis-opens C[As](C)C to the carbon
    # chain '(methylmethyl)methane'). The hydride namer fail-closes on anything
    # that is not a clean parent hydride (=O / ring / charge / dot-disconnect), so
    # a genuine As/Sb/Bi coordination complex is unaffected. Cheap pure RDKit
    # graph walk (no OPSIN); mirrors the single-anion decline above.
    from orthonym.rules.mononuclear_hydrides import name_mononuclear_hydride
    if name_mononuclear_hydride(mol) is not None:
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
                        # P-72.6: junior anionic groups -> anionic prefix
                        # (carboxylato/oxido), not neutral carboxy/hydroxy.
                        from orthonym.rules.ions import (
                            _apply_anionic_substituent_prefixes,
                        )
                        anion_name = _apply_anionic_substituent_prefixes(
                            mol, sites['anions'], anion_name
                        )
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
    """P-14.8 adduct assembly first; frozen space-join for identical-only
    sets; otherwise fail-closed (None -> cascade-continue).

    Wave-2 P0A replaces the legacy per-component ' '.join: distinct
    component sets now emit the P-14.8.1 em-dash + (n/m) proportion form
    via rules.adducts.name_adduct (each fragment named by a FRESH
    Orthonym(style) — audit § 3.3 RL-4 pattern — so the per-fragment
    OPSIN validity gate stays on). The ONLY preserved legacy behavior is
    the all-identical-fragments space-join ('CCO.OCC' -> 'ethanol
    ethanol', the Plan-01 byte-identical representative): identical
    entities are not an adduct of SEPARATE molecular entities (P-14.8.1
    definition). The old silent skip of unnameable fragments (a
    structure-dropping hazard) is removed — partial sets refuse.
    """
    from orthonym.rules.adducts import name_adduct
    adduct_name = name_adduct(mol, canonical_smiles, style=style)
    if adduct_name is not None:
        return adduct_name
    # Frozen legacy path: ALL fragments constitutionally identical.
    frags = canonical_smiles.split('.')
    frag_mols = [Chem.MolFromSmiles(f) for f in frags]
    if any(fm is None for fm in frag_mols):
        return None
    if len({Chem.MolToSmiles(fm, canonical=True) for fm in frag_mols}) != 1:
        return None  # distinct set already refused by name_adduct
    if sum(1 for fm in frag_mols if fm.GetNumHeavyAtoms() >= 2) < 2:
        return None
    try:
        from orthonym.namer import Orthonym
        frag_name = Orthonym(style=style).name(frags[0])
    except Exception:
        return None
    if not frag_name or frag_name.startswith("unknown"):
        return None
    return ' '.join([frag_name] * len(frags))


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
    # Catalog-join (D-05/D-10/F-CATALOG-JOIN) then systematic monosaccharide
    # (D-01) — factored into sugar_names.name_free_sugar so the free-sugar-ester
    # path names its residual through the SAME cascade (DRY). Byte-identical.
    from orthonym.data.sugar_names import name_free_sugar
    free = name_free_sugar(mol, canonical_smiles)
    if free is not None:
        return free

    # W6-P1: free-sugar mono-phosphate / sulfate ester (BB P-102.5.6.1.2/.1.3).
    # Fires only on a single sugar ring bearing exactly one O-phosphate/O-sulfate;
    # fail-closed (None) otherwise -> disaccharide, then cascade-continue.
    from orthonym.data.sugar_names import name_sugar_ester
    ester = name_sugar_ester(mol, canonical_smiles)
    if ester is not None:
        return ester

    # W6-P2: glycosylamine (anomeric -NH2, P-102.6.1.3) and glycosyl halide
    # (anomeric halogen, P-102.6.1.5). Each fail-closed on any non-exact shape.
    from orthonym.data.sugar_names import name_glycosylamine, name_glycosyl_halide
    for _fn in (name_glycosylamine, name_glycosyl_halide):
        _nm = _fn(mol, canonical_smiles)
        if _nm is not None:
            return _nm

    # W6-P3: O-methyl ether sugar (P-102.5.6.1). Fail-closed on non-O-methyl.
    from orthonym.data.sugar_names import name_sugar_o_methyl
    ome = name_sugar_o_methyl(mol, canonical_smiles)
    if ome is not None:
        return ome

    # W6B-T8: open-chain aldonate / aldarate(partial) ester (P-102.5.6.6.2.1/.5.3).
    from orthonym.data.sugar_names import name_aldonate_ester
    aldonate = name_aldonate_ester(mol, canonical_smiles)
    if aldonate is not None:
        return aldonate

    # W6B-T9: N-alkylamino open-chain aldose (P-102.5.4.1.2).
    from orthonym.data.sugar_names import name_amino_deoxy_open_sugar
    amino_open = name_amino_deoxy_open_sugar(mol, canonical_smiles)
    if amino_open is not None:
        return amino_open

    # W6B-T12: glycosyloxy on a senior aglycone (P-102.6.1.2), ABOVE the glycoside
    # decomposition path so the aglycone-as-parent form wins over the legacy
    # (glycosyloxy)aglycone form.
    from orthonym.data.sugar_names import name_glycosyloxy_aglycone
    glyoxy = name_glycosyloxy_aglycone(mol, canonical_smiles)
    if glyoxy is not None:
        return glyoxy

    # W6B-T13: C-glycosyl on a senior aglycone (P-102.6.1.4).
    from orthonym.data.sugar_names import name_c_glycosyl_aglycone
    cgly = name_c_glycosyl_aglycone(mol, canonical_smiles)
    if cgly is not None:
        return cgly

    # W8-P7b.5: glycosyloxy n-O-yl on a senior parent (P-102.6.2) — a sugar bonded
    # via a NON-anomeric ring O -> (<anomer>-<config>-glycopyranos-n-O-yl)<parent>.
    # Fires ABOVE the deferral veto below; fail-closed (None) on any non-exact shape.
    from orthonym.data.sugar_names import name_glycosyloxy_yl_parent
    noyl = name_glycosyloxy_yl_parent(mol, canonical_smiles)
    if noyl is not None:
        return noyl

    # W8-P7b.4: C-substituted monosaccharide (P-102.5.6.3.1/.3.2) -> <n>-C-<sub>- or
    # <n>-deoxy-<n>-<sub>-<sugar>. Fail-closed (None) on any non-exact shape; the
    # substituted-centre config + locant are RT-verified (fail-closed w/o Java).
    from orthonym.data.sugar_names import name_c_substituted_sugar
    csug = name_c_substituted_sugar(mol, canonical_smiles)
    if csug is not None:
        return csug

    # W6B-T13 fail-closed veto (n-O-yl, P-102.6.2 remainder): a sugar-O-ether the
    # namers cannot handle would otherwise drop the sugar (a wrong name) via the
    # general chain namer -> refuse ('' -> unknown), never ship a sugar-dropping name.
    from orthonym.data.sugar_names import _is_sugar_o_ether_leak
    if _is_sugar_o_ether_leak(mol):
        return ''

    # Disaccharide / oligosaccharide (D-02).
    from orthonym.rules.oligosaccharides import name_disaccharide
    disacc = name_disaccharide(mol)
    if disacc is not None:
        return disacc

    # W6B-T7 fail-closed veto: an uncataloged 2-ulosonic acid must NOT fall through
    # to the general ring-carboxylic namer (which drops side-chain stereo -> a
    # wrong, RT-failing name).  Refuse ('' terminates the cascade -> 'unknown').
    from orthonym.data.sugar_names import _is_uncataloged_ulosonic_acid
    if _is_uncataloged_ulosonic_acid(mol, canonical_smiles):
        return ''
    return None


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
    """Mirrors namer.py:1111-1113.

    Wave-8 P8 (Task 8.12): OPSIN 2.9.0 still cannot parse ANY phane name, so
    there is no RT oracle -- but `rules.phane.build_phane_pin` now builds a
    BB-name-exact, correct-by-construction P-26.2/.3 simplified-skeletal PIN
    for the monocyclic all-benzene-homophane class (verified against the Blue
    Book directly; guarded by `_phane_formula_veto`, a source-level atom-
    conservation check, since the RT-gate would otherwise fail OPEN with no
    Java). `_PHANE_PIN_RE` (namer.py) carves this PIN grammar out of the
    OPSIN validity gate so it isn't suppressed to a descriptive fallback.

    `name_cyclophane` falls back to the legacy semi-systematic bracket-prefix
    composer (`[m.n]paracyclophane`) for `is_cyclophane`-positive topologies
    `build_phane_pin` doesn't (yet) cover (heteroatom bridges, fused/hetero
    amplificants, von Baeyer/spiro skeletons, mixed amplificants, substituted
    phanes) -- that legacy form is UNVERIFIABLE (no BB-name-exact fixture, no
    RT oracle), so it is still withheld here: raise the G0
    UNSUPPORTED_RING_SYSTEM signal (jar-independent 'unknown') rather than
    emit it. Only a genuine `build_phane_pin` PIN is emitted.
    """
    from orthonym.rules.phane import build_phane_pin, name_cyclophane
    pin = build_phane_pin(mol)
    if pin is not None:
        return pin
    if name_cyclophane(mol) is not None:
        from orthonym.errors import unsupported_ring_system
        raise unsupported_ring_system()
    return None


def _handle_inositol(mol, smiles, canonical_smiles, features=None, *,
                     style: str = "pin", **kwargs) -> Optional[str]:
    """v23 Phase 12 follow-on: retained inositol PIN (P-104.2.1) for the seven
    meso inositols; the chiral chiro pair is REFUSED to a deterministic
    descriptive fallback (RDKit perceives its absolute config non-deterministically,
    so a systematic CIP name would flip the enantiomer by atom order)."""
    from orthonym.rules.inositols import name_inositol, is_chiral_inositol
    nm = name_inositol(mol)
    if nm is not None:
        return nm
    if is_chiral_inositol(mol):
        from orthonym.namer import _descriptive_fallback
        return _descriptive_fallback(smiles)
    return None


def _is_nucleoside(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """v23 Phase 14 — decorated nucleoside / nucleotide recogniser (P-105.2/P-106).
    Fires only for a furanose-N-glycoside bearing a recognised sugar decoration
    (5'-phosphate chain or O-acyl ester) whose bare nucleoside is a retained
    name; bare nucleosides + the retained adenylic/inosinic monophosphates are
    NOT claimed here (they hit RETAINED_NAME@1300 first). Fail-closed."""
    from orthonym.rules.nucleosides import name_nucleoside
    return name_nucleoside(mol) is not None


def _handle_nucleoside(mol, smiles, canonical_smiles, features=None, *,
                       style: str = "pin", **kwargs) -> Optional[str]:
    """v23 Phase 14 — decorated nucleoside/nucleotide PIN (P-105.2/P-106)."""
    from orthonym.rules.nucleosides import name_nucleoside
    return name_nucleoside(mol)


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


def _is_inorganic_acid(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """v22 G2 COV-02; priority 40 (before ORGANOMETALLIC@50 so silicic acid is
    not claimed as a Si organometallic).

    PURE: exact full-molecule canonical-SMILES match only (zero false positives;
    charged conjugate bases / esters / derivatives never match). No mutation.
    """
    if mol is None:
        return False
    from orthonym.rules.inorganic_acids import name_inorganic_acid
    return name_inorganic_acid(mol) is not None


def _handle_inorganic_acid(mol, smiles, canonical_smiles, features=None, **kwargs) -> Optional[str]:
    """Return the retained free-inorganic-oxoacid PIN (P-67/P-65.2.1), else None
    (cascade-continuation per CONTEXT D-02)."""
    from orthonym.rules.inorganic_acids import name_inorganic_acid
    return name_inorganic_acid(mol)


def _is_mononuclear_hydride(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """v23 Phase 6/7 (P-68 / P-21.1 / P-31.1.4.2); priority 45 (after
    INORGANIC_ACID@40 so an oxoacid is never claimed here, before ORGANOMETALLIC@50
    so a Group-15 As/Sb/Bi parent hydride is named substitutively, not as ORGM).

    PURE graph classifier (Phase 7 generalised Phase 6): a mononuclear parent
    hydride — all-halogen hub (SF6/PF5 -> λ; PCl3/SF2/AsCl3 -> no λ) OR an
    organyl/bare Group-15 As/Sb/Bi hub (trimethylarsane / triphenylarsane /
    arsane). Fail-closed; no mutation.
    """
    if mol is None:
        return False
    from orthonym.rules.mononuclear_hydrides import name_mononuclear_hydride
    return name_mononuclear_hydride(mol) is not None


def _handle_mononuclear_hydride(mol, smiles, canonical_smiles, features=None, **kwargs) -> Optional[str]:
    """Return the λ-convention mononuclear-hydride PIN (P-68 / P-31.1.4.2), else
    None (cascade-continuation per CONTEXT D-02)."""
    from orthonym.rules.mononuclear_hydrides import name_mononuclear_hydride
    return name_mononuclear_hydride(mol)


def _is_chalcogen_chain(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """v23 Phase 7 (P-21.2.2); priority 46. A homogeneous O/S/Se/Te chain parent
    hydride (trisulfane/trioxidane). PURE graph classifier, fail-closed."""
    if mol is None:
        return False
    from orthonym.rules.polychalcogen import name_chalcogen_chain
    return name_chalcogen_chain(mol) is not None


def _handle_chalcogen_chain(mol, smiles, canonical_smiles, features=None, **kwargs) -> Optional[str]:
    """Return the chalcogen-chain PIN (P-21.2.2), else None (cascade-continuation)."""
    from orthonym.rules.polychalcogen import name_chalcogen_chain
    return name_chalcogen_chain(mol)


def _is_polychalcogen_oxide(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """W3-P13 (P-68.4.3.2); priority 46.5. A di-/polysulfoxide-sulfone: a chain
    of >=2 identical S/Se/Te (each lambda4/lambda6) every one bearing >=1 =O
    (CH3-S(=O)-S(=O)-CH3 -> ...disulfane-1,2-dione). PURE graph classifier,
    fail-closed."""
    if mol is None:
        return False
    from orthonym.rules.polychalcogen import name_polysulfoxide_sulfone
    return name_polysulfoxide_sulfone(mol) is not None


def _handle_polychalcogen_oxide(mol, smiles, canonical_smiles, features=None, **kwargs) -> Optional[str]:
    """Return the di-/polysulfoxide-sulfone PIN (P-68.4.3.2), else None
    (cascade-continuation)."""
    from orthonym.rules.polychalcogen import name_polysulfoxide_sulfone
    return name_polysulfoxide_sulfone(mol)


def _is_polyazane(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """v23 Phase 7 (P-68.3.1.1/.3); priority 47. A polyazane-family parent hydride
    (hydrazine/diazene/triazane/azo). PURE graph classifier, fail-closed."""
    if mol is None:
        return False
    from orthonym.rules.polyazane import name_polyazane
    return name_polyazane(mol) is not None


def _handle_polyazane(mol, smiles, canonical_smiles, features=None, **kwargs) -> Optional[str]:
    """Return the polyazane-family PIN (P-68.3.1.1/.3), else None (cascade-continuation)."""
    from orthonym.rules.polyazane import name_polyazane
    return name_polyazane(mol)


def _is_free_homonuclear_g14_hydride(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """Wave-3 (P-21.2.3 / P-68.2.3); priority 47.7. A free homonuclear Group-14
    catenated parent hydride (disilane / trisilane / digermene). PURE graph
    classifier, fail-closed."""
    if mol is None:
        return False
    from orthonym.rules.multiplicative import name_free_homonuclear_group14_hydride
    return name_free_homonuclear_group14_hydride(mol) is not None


def _handle_free_homonuclear_g14_hydride(mol, smiles, canonical_smiles, features=None, **kwargs) -> Optional[str]:
    """Return the free homonuclear Group-14 catenated-hydride PIN (P-21.2.3 /
    P-68.2.3), else None (cascade-continuation per CONTEXT D-02)."""
    from orthonym.rules.multiplicative import name_free_homonuclear_group14_hydride
    return name_free_homonuclear_group14_hydride(mol)


def _is_dinuclear_hydride(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """v23 Phase 10 (P-69.5.3); priority 48. A two-atom Group-14/Group-15 catenated
    parent hydride (germylstibane). PURE graph classifier, fail-closed."""
    if mol is None:
        return False
    from orthonym.rules.mononuclear_hydrides import name_dinuclear_hydride
    return name_dinuclear_hydride(mol) is not None


def _handle_dinuclear_hydride(mol, smiles, canonical_smiles, features=None, **kwargs) -> Optional[str]:
    """Return the di-nuclear catenated-hydride PIN (P-69.5.3), else None (cascade-continuation)."""
    from orthonym.rules.mononuclear_hydrides import name_dinuclear_hydride
    return name_dinuclear_hydride(mol)


def _is_ketene(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """Wave-2 completion (P-64.2.2.4); priority 49. The exact (halo)ketene
    heterocumulene O=C=C(H/X)2 (ethenone parent). PURE graph classifier,
    fail-closed."""
    if mol is None:
        return False
    from orthonym.rules.ketenes import name_ketene
    return name_ketene(mol) is not None


def _is_azinic_derivative(mol, smiles, canonical_smiles, features=None,
                          **kwargs) -> bool:
    """Wave-2 completion C (P-61.5.3); priority 48.3. Ylidene azinic acid
    (aci-nitro) parents. PURE graph classifier, fail-closed."""
    if mol is None:
        return False
    from orthonym.rules.inorganic_acids import name_azinic_derivative
    return name_azinic_derivative(mol) is not None


def _handle_azinic_derivative(mol, smiles, canonical_smiles, features=None,
                              **kwargs) -> Optional[str]:
    from orthonym.rules.inorganic_acids import name_azinic_derivative
    return name_azinic_derivative(mol)


def _is_cumulative_zwitterion(mol, smiles, canonical_smiles, features=None,
                              **kwargs) -> bool:
    """W4-I4 (P-74.1.1); priority 48.35. Same-parent '-ium-...-ide' zwitterion on
    a homogeneous heteroatom chain parent hydride (hydrazine / triazane->triazene
    / dioxidane). PURE graph classifier, fail-closed."""
    if mol is None:
        return False
    from orthonym.rules.ions import emit_cumulative_ium_ide
    return emit_cumulative_ium_ide(mol) is not None


def _handle_cumulative_zwitterion(mol, smiles, canonical_smiles, features=None,
                                  **kwargs) -> Optional[str]:
    from orthonym.rules.ions import emit_cumulative_ium_ide
    return emit_cumulative_ium_ide(mol)


def _is_ylide(mol, smiles, canonical_smiles, features=None,
              **kwargs) -> bool:
    """W4-I4 (P-74.2.1.1); priority 48.36. Onium cation (N/P/O/S+, no free H)
    bonded to an adjacent carbanion -> the carbanion '-ide' parent with the onium
    '-aniumyl' prefix. PURE graph classifier, fail-closed."""
    if mol is None:
        return False
    from orthonym.rules.ions import emit_ylide
    return emit_ylide(mol) is not None


def _handle_ylide(mol, smiles, canonical_smiles, features=None,
                  **kwargs) -> Optional[str]:
    from orthonym.rules.ions import emit_ylide
    return emit_ylide(mol)


def _is_heterone(mol, smiles, canonical_smiles, features=None,
                 **kwargs) -> bool:
    """Wave-2 completion C (P-64.1.2.2/P-64.4.1); priority 48.4. Si/Ge/P/As
    =O heterone parents (dimethylsilanone). PURE graph classifier."""
    if mol is None:
        return False
    from orthonym.rules.mononuclear_hydrides import name_heterone
    return name_heterone(mol) is not None


def _handle_heterone(mol, smiles, canonical_smiles, features=None,
                     **kwargs) -> Optional[str]:
    from orthonym.rules.mononuclear_hydrides import name_heterone
    return name_heterone(mol)


def _is_sulfine(mol, smiles, canonical_smiles, features=None,
                **kwargs) -> bool:
    """Wave-2 completion C (P-64.4.2); priority 48.5. Acyclic thiocarbonyl
    S-oxides (propylidene-lambda4-sulfanone). PURE graph classifier."""
    if mol is None:
        return False
    from orthonym.rules.mononuclear_hydrides import name_sulfine
    return name_sulfine(mol) is not None


def _handle_sulfine(mol, smiles, canonical_smiles, features=None,
                    **kwargs) -> Optional[str]:
    from orthonym.rules.mononuclear_hydrides import name_sulfine
    return name_sulfine(mol)


def _is_pseudoketone_hetero(mol, smiles, canonical_smiles, features=None,
                            **kwargs) -> bool:
    """Wave-2 completion C (P-64.1.2.1(b)/P-64.5.2.2); priority 48.6. Acyl
    on Si/Ge/P/As hub (1-silylethan-1-one). PURE graph classifier."""
    if mol is None:
        return False
    from orthonym.rules.pseudoketones import name_acyl_hetero_pseudoketone
    return name_acyl_hetero_pseudoketone(mol) is not None


def _handle_pseudoketone_hetero(mol, smiles, canonical_smiles, features=None,
                                **kwargs) -> Optional[str]:
    from orthonym.rules.pseudoketones import name_acyl_hetero_pseudoketone
    return name_acyl_hetero_pseudoketone(mol)


def _is_acyl_chalcogenchain_pseudoketone(mol, smiles, canonical_smiles,
                                         features=None, **kwargs) -> bool:
    """W3-P14 (P-68.4.1.3); priority 48.65. An acyl on a homogeneous >=3-chalcogen
    chain (CH3CH2-CO-O-O-OH -> 1-trioxidanylpropan-1-one). PURE graph classifier,
    fail-closed (declines ordinary esters/thioesters)."""
    if mol is None:
        return False
    from orthonym.rules.pseudoketones import name_acyl_chalcogenchain_pseudoketone
    return name_acyl_chalcogenchain_pseudoketone(mol) is not None


def _handle_acyl_chalcogenchain_pseudoketone(mol, smiles, canonical_smiles,
                                             features=None, **kwargs) -> Optional[str]:
    """Return the acyl-chalcogenchain pseudoketone PIN (P-68.4.1.3), else None
    (cascade-continuation)."""
    from orthonym.rules.pseudoketones import name_acyl_chalcogenchain_pseudoketone
    return name_acyl_chalcogenchain_pseudoketone(mol)


def _is_lambda5_phosphanimine(mol, smiles, canonical_smiles, features=None,
                              **kwargs) -> bool:
    """W4-I4 (P-74.2.1.5); priority 48.66. Phosphine imide R3X=N-R (X=P/As/Sb) at
    the lambda5 bonding number -> lambda5-phosphanimine PIN. PURE, fail-closed."""
    if mol is None:
        return False
    from orthonym.rules.mononuclear_hydrides import name_lambda5_phosphanimine
    return name_lambda5_phosphanimine(mol) is not None


def _handle_lambda5_phosphanimine(mol, smiles, canonical_smiles, features=None,
                                  **kwargs) -> Optional[str]:
    from orthonym.rules.mononuclear_hydrides import name_lambda5_phosphanimine
    return name_lambda5_phosphanimine(mol)


def _is_heteroimine(mol, smiles, canonical_smiles, features=None,
                    **kwargs) -> bool:
    """W2E-P1FG (P-62.3.1.3); priority 48.7. X=NH where X is a mononuclear
    P/As/Si hub (CH3-P=NH -> 1-methylphosphanimine). PURE graph classifier."""
    if mol is None:
        return False
    from orthonym.rules.mononuclear_hydrides import name_heteroimine
    return name_heteroimine(mol) is not None


def _handle_heteroimine(mol, smiles, canonical_smiles, features=None,
                        **kwargs) -> Optional[str]:
    from orthonym.rules.mononuclear_hydrides import name_heteroimine
    return name_heteroimine(mol)


def _is_lambda_sulfane_imine_oxide(mol, smiles, canonical_smiles, features=None,
                                   **kwargs) -> bool:
    """W3-P13 (P-68.4.3.3-.8); priority 48.75. Mononuclear S/Se/Te hub of
    non-standard valence bearing >=1 imine (=N-H/=N-R) plus optional =O and
    organyls: sulfimide/sulfoximide/sulfonediimine/sulfur di-/tri-imide
    (S,S-diethyl-N-phenyl-lambda4-sulfanimine). PURE graph classifier."""
    if mol is None:
        return False
    from orthonym.rules.mononuclear_hydrides import name_lambda_sulfane_imine_oxide
    return name_lambda_sulfane_imine_oxide(mol) is not None


def _handle_lambda_sulfane_imine_oxide(mol, smiles, canonical_smiles, features=None,
                                       **kwargs) -> Optional[str]:
    from orthonym.rules.mononuclear_hydrides import name_lambda_sulfane_imine_oxide
    return name_lambda_sulfane_imine_oxide(mol)


def _is_ring_chalcogen_oxide(mol, smiles, canonical_smiles, features=None,
                             **kwargs) -> bool:
    """Wave-2 completion (P-25.6/P-74.3.1.3); priority 49.5. A neutral ring
    S/Se/Te bearing 1-2 exocyclic =O on an otherwise-bare nameable ring system
    (dibenzothiophene 5-oxide / 5,5-dioxide). Fail-closed classifier+namer."""
    if mol is None:
        return False
    from orthonym.rules.ring_chalcogen_oxide import name_ring_chalcogen_oxide
    return name_ring_chalcogen_oxide(mol) is not None


def _handle_ring_chalcogen_oxide(mol, smiles, canonical_smiles, features=None,
                                 **kwargs) -> Optional[str]:
    """Return the additive ring-chalcogen oxide name (P-25.6), else None
    (cascade-continuation)."""
    from orthonym.rules.ring_chalcogen_oxide import name_ring_chalcogen_oxide
    return name_ring_chalcogen_oxide(mol)


def _is_cyclic_polyester(mol, smiles, canonical_smiles, features=None,
                         **kwargs) -> bool:
    """W3-P08 (P-65.6.3.5.3); priority 49.8. A saturated monocyclic C/O ring
    with >=2 ester carbonyls (lactide) -> 1,4-dioxane-2,5-dione. Fail-closed
    graph classifier (declines single lactone / carbonate / anhydride)."""
    if mol is None:
        return False
    from orthonym.rules.lactones import name_cyclic_polyester
    return name_cyclic_polyester(mol) is not None


def _handle_cyclic_polyester(mol, smiles, canonical_smiles, features=None,
                             **kwargs) -> Optional[str]:
    """Return the lactide / cyclic-polyester dione PIN (P-65.6.3.5.3), else
    None (cascade-continuation)."""
    from orthonym.rules.lactones import name_cyclic_polyester
    return name_cyclic_polyester(mol)


def _is_hydro_fused_peroxol(mol, smiles, canonical_smiles, features=None,
                            **kwargs) -> bool:
    """W2E-P1FG (P-63.4.1); priority 49.6. -OOH on an sp3 carbon of a
    partially saturated fused carbocycle (1,2,3,4-tetrahydronaphthalene-1-
    peroxol). Fail-closed classifier+namer."""
    if mol is None:
        return False
    from orthonym.rules.partial_saturation import name_hydro_fused_chalcogen_suffix
    return name_hydro_fused_chalcogen_suffix(mol) is not None


def _handle_hydro_fused_peroxol(mol, smiles, canonical_smiles, features=None,
                                **kwargs) -> Optional[str]:
    """Return the ring peroxol PIN (P-63.4.1), else None (cascade-continuation)."""
    from orthonym.rules.partial_saturation import name_hydro_fused_chalcogen_suffix
    return name_hydro_fused_chalcogen_suffix(mol)


def _handle_ketene(mol, smiles, canonical_smiles, features=None, **kwargs) -> Optional[str]:
    """Return the ethenone-parent ketene PIN (P-64.2.2.4), else None (cascade-continuation)."""
    from orthonym.rules.ketenes import name_ketene
    return name_ketene(mol)


def _is_thioimide(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """W2E-D3 (P-66.1.4.2); priority 49.7. The exact acyclic N-H thioimide
    R-C(=S)-NH-C(=S)-R' with plain alkanethioyl branches. The decomposition
    engine never reaches this class (its amide-bond SMARTS requires C(=O)), so
    this dedicated graph classifier fills the gap. PURE graph classifier,
    fail-closed."""
    if mol is None:
        return False
    from orthonym.rules.thioimides import name_thioimide
    return name_thioimide(mol) is not None


def _handle_thioimide(mol, smiles, canonical_smiles, features=None, **kwargs) -> Optional[str]:
    """Return the N-(alkanethioyl)alkanethioamide PIN (P-66.1.4.2), else None."""
    from orthonym.rules.thioimides import name_thioimide
    return name_thioimide(mol)


def _is_catenated_hydride(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """Wave-2 completion (P-21.2.3/P-52.1.3); priority 47.5. An alternating
    homonuclear Group-14/bridge catenated hydride (disiloxane/trisiloxane/
    disilazane). PURE graph classifier, fail-closed."""
    if mol is None:
        return False
    from orthonym.rules.catenated_hydrides import name_catenated_hydride
    return name_catenated_hydride(mol) is not None


def _handle_catenated_hydride(mol, smiles, canonical_smiles, features=None, **kwargs) -> Optional[str]:
    """Return the catenated Group-14/bridge hydride PIN (P-21.2.3), else None."""
    from orthonym.rules.catenated_hydrides import name_catenated_hydride
    return name_catenated_hydride(mol)


def _is_heterochalcogen_aba(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """W3-P14 (P-68.4.2.1/P-21.2.3.1); priority 47.55. A pure-chalcogen a[ba]n
    parent hydride (HS-O-SH -> dithioxane, CH3-S-O-S-CH3 -> dimethyldithioxane).
    PURE graph classifier, fail-closed; pre-empts SKELETAL_REPLACEMENT@1500."""
    if mol is None:
        return False
    from orthonym.rules.catenated_hydrides import name_heterochalcogen_aba
    return name_heterochalcogen_aba(mol) is not None


def _handle_heterochalcogen_aba(mol, smiles, canonical_smiles, features=None, **kwargs) -> Optional[str]:
    """Return the pure-chalcogen a[ba]n PIN (P-68.4.2.1 / P-21.2.3.1: dithioxane),
    else None (cascade-continuation)."""
    from orthonym.rules.catenated_hydrides import name_heterochalcogen_aba
    return name_heterochalcogen_aba(mol)


def _is_homonuclear_pnictogen_chain(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """W3-P14 (P-68.3.2.2); priority 47.6. A homonuclear Group-15 catenated parent
    hydride (PP -> diphosphane, pentaarsane, dibismuthane). PURE graph classifier,
    fail-closed."""
    if mol is None:
        return False
    from orthonym.rules.catenated_hydrides import name_homonuclear_pnictogen_chain
    return name_homonuclear_pnictogen_chain(mol) is not None


def _handle_homonuclear_pnictogen_chain(mol, smiles, canonical_smiles, features=None, **kwargs) -> Optional[str]:
    """Return the homonuclear Group-15 catenated-hydride PIN (P-68.3.2.2), else
    None (cascade-continuation)."""
    from orthonym.rules.catenated_hydrides import name_homonuclear_pnictogen_chain
    return name_homonuclear_pnictogen_chain(mol)


def _is_pnictogen_carboxylic_acid(mol, smiles, canonical_smiles, features=None, **kwargs) -> bool:
    """W3-P14 (P-68.3.2.3.1); priority 47.65. An added-carbon -carboxylic acid on
    a P/As/Sb parent hydride (H2P-COOH -> phosphanecarboxylic acid). PURE graph
    classifier, fail-closed; runs ahead of the generic acid namer (GENERAL)."""
    if mol is None:
        return False
    from orthonym.rules.phosphorus import name_phosphane_carboxylic_acid
    return name_phosphane_carboxylic_acid(mol) is not None


def _handle_pnictogen_carboxylic_acid(mol, smiles, canonical_smiles, features=None, **kwargs) -> Optional[str]:
    """Return the pnictogen -carboxylic acid PIN (P-68.3.2.3.1), else None
    (cascade-continuation)."""
    from orthonym.rules.phosphorus import name_phosphane_carboxylic_acid
    return name_phosphane_carboxylic_acid(mol)


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

# --- v22 G2 COV-02: INORGANIC_ACID at priority 40 (intercepts BEFORE ---
# ORGANOMETALLIC@50 so silicic acid O[Si](O)(O)O is named, not claimed as a Si
# organometallic; and before GENERAL so free phosphoric/sulfuric/carbonic acids
# stop returning 'trihydrophosphate'/'unknown'/'methane'). Exact canonical-SMILES
# match -> zero false positives; cascade-continuation on None per D-02. ---
_register_dispatch(
    class_id=StoutClass.INORGANIC_ACID, priority=40, tier=1,
    predicate=_is_inorganic_acid, handler=_handle_inorganic_acid,
    iupac_section="Blue Book P-67 / P-65.2.1 / P-68.2",
    description="Free mononuclear/simple-polynuclear inorganic oxoacid with a "
                "retained/preselected PIN (phosphoric/sulfuric/carbonic/silicic/"
                "nitric/diphosphoric/disulfuric acid); exact canonical-SMILES lookup",
)


# --- v23 Phase 6/7: MONONUCLEAR_HYDRIDE at priority 45 (intercepts BETWEEN ---
# INORGANIC_ACID@40 and ORGANOMETALLIC@50). A mononuclear parent hydride (P-68 /
# P-21.1 / P-31.1.4.2): an all-halogen hub on any of S/Se/Te/P/As/Sb/Bi/I (SF6 ->
# hexafluoro-lambda6-sulfane, PF5 -> pentafluoro-λ5; standard valence PCl3 ->
# trichlorophosphane, SF2 -> difluorosulfane, AsCl3 -> trichloroarsane), OR an
# organyl/bare Group-15 As/Sb/Bi hub (C[As](C)C -> trimethylarsane, [AsH3] ->
# arsane) — all 'unknown' / ORGM-mangled before these phases. Graph classifier,
# fail-closed; cascade-continuation on None per D-02. Phase 7 dropped Phase 6's
# "λ REQUIRED" gate and added the As/Sb/Bi stems + the organyl regime. ---
_register_dispatch(
    class_id=StoutClass.MONONUCLEAR_HYDRIDE, priority=45, tier=1,
    predicate=_is_mononuclear_hydride, handler=_handle_mononuclear_hydride,
    iupac_section="Blue Book P-68 / P-21.1 / P-31.1.4.2 (λ-convention)",
    description="Mononuclear parent hydride: all-halogen hub "
                "(S/Se/Te/P/As/Sb/Bi/I) with optional λ-convention "
                "(hexafluoro-lambda6-sulfane / trichlorophosphane / trichloroarsane) "
                "or an organyl/bare Group-15 As/Sb/Bi hub (trimethylarsane / arsane); "
                "graph classifier, fail-closed",
)


# --- v23 Phase 7: CHALCOGEN_CHAIN at priority 46 (after MONONUCLEAR_HYDRIDE@45). ---
# A homogeneous O/S/Se/Te chain parent hydride (P-21.2.2): SS -> disulfane,
# OOO -> trioxidane, SSSS -> tetrasulfane, CSSS -> 1-methyltrisulfane (all
# 'unknown' before this phase). Carbon substitution admitted only for >=3
# chalcogens (avoids the sulfide/disulfide functional classes). Graph classifier,
# fail-closed; cascade-continuation on None per D-02. ---
_register_dispatch(
    class_id=StoutClass.CHALCOGEN_CHAIN, priority=46, tier=1,
    predicate=_is_chalcogen_chain, handler=_handle_chalcogen_chain,
    iupac_section="Blue Book P-21.2.2 / P-68.3",
    description="Homogeneous chalcogen-chain parent hydride (disulfane / trioxidane "
                "/ tetrasulfane / 1-methyltrisulfane); graph classifier, fail-closed",
)


# --- W3-P13: POLYCHALCOGEN_OXIDE at priority 46.5 (after CHALCOGEN_CHAIN@46, ---
# before POLYAZANE@47). A di-/polysulfoxide-sulfone (P-68.4.3.2): a chain of >=2
# identical S/Se/Te (each lambda4/lambda6) every one bearing >=1 =O, named by
# adding '-one' to the lambda-<multiplier>sulfane parent — CH3-S(=O)-S(=O)-CH3 ->
# 1,2-dimethyl-1lambda4,2lambda4-disulfane-1,2-dione ('unknown' before this).
# CHALCOGEN_CHAIN@46 already declines it (the =O makes O a non-chain heteroatom),
# and the single-S sulfoxide/sulfone (n=1) stays with the P-63.6 sulfur handler.
# Graph classifier, fail-closed; cascade-continuation on None per D-02. ---
_register_dispatch(
    class_id=StoutClass.POLYCHALCOGEN_OXIDE, priority=46.5, tier=1,
    predicate=_is_polychalcogen_oxide, handler=_handle_polychalcogen_oxide,
    iupac_section="Blue Book P-68.4.3.2",
    description="Di-/polysulfoxide-sulfone chain named on the lambda-disulfane "
                "parent (1,2-dimethyl-1lambda4,2lambda4-disulfane-1,2-dione); "
                "graph classifier, fail-closed",
)


# --- v23 Phase 7: POLYAZANE at priority 47 (after CHALCOGEN_CHAIN@46). ---
# The polyazane parent-hydride family (P-68.3.1.1 / P-68.3.1.3): NN -> hydrazine,
# N=N -> diazene, NNN -> triazane, N=NN -> triaz-1-ene, CN=NC -> 1,2-dimethyldiazene,
# PhN=NPh -> 1,2-diphenyldiazene (all 'unknown' before this phase). Graph classifier
# over N-N bonds; fail-closed on amines/diamines/hydroxylamine/hydrazones/azides;
# cascade-continuation on None per D-02. ---
_register_dispatch(
    class_id=StoutClass.POLYAZANE, priority=47, tier=1,
    predicate=_is_polyazane, handler=_handle_polyazane,
    iupac_section="Blue Book P-68.3.1.1 / P-68.3.1.3 / P-21.2.2",
    description="Polyazane-family parent hydride (hydrazine / diazene / triazane / "
                "triaz-1-ene / 1,2-diphenyldiazene); graph classifier, fail-closed",
)


# --- Wave-2 completion: CATENATED_HYDRIDE at priority 47.5 (after POLYAZANE@47, ---
# before DINUCLEAR_HYDRIDE@48 / ORGM@50). Alternating homonuclear Group-14/bridge
# catenated parent hydride (P-21.2.3/P-52.1.3): [SiH3]O[SiH3] -> disiloxane,
# [SiH3]O[SiH2]O[SiH3] -> trisiloxane, [SiH3]N[SiH3] -> disilazane, [SnH3]O[SnH3]
# -> distannoxane. These are exactly what skeletal_replacement Gate 3b declines
# (terminal Group-14) to avoid the dimethoxysilane structure-loss. Graph
# classifier, fail-closed; cascade-continuation on None. ---
_register_dispatch(
    class_id=StoutClass.CATENATED_HYDRIDE, priority=47.5, tier=1,
    predicate=_is_catenated_hydride, handler=_handle_catenated_hydride,
    iupac_section="Blue Book P-21.2.3 / P-52.1.3",
    description="Alternating Group-14/bridge catenated parent hydride "
                "(disiloxane / trisiloxane / disilazane / distannoxane); "
                "graph classifier, fail-closed",
)


# --- W3-P14: HETEROCHALCOGEN_ABA at priority 47.55 (after CATENATED_HYDRIDE@47.5, ---
# before HOMONUCLEAR_PNICTOGEN_CHAIN@47.6). A pure-chalcogen a[ba]n parent hydride
# (P-68.4.2.1 / P-21.2.3.1): HS-O-SH -> dithioxane, CH3-S-O-SH -> methyldithioxane,
# CH3-S-O-S-CH3 -> dimethyldithioxane. These preselected parent hydrides receive
# the PIN and MUST pre-empt SKELETAL_REPLACEMENT@1500 (CH3-S-O-S-CH3 would else be
# the valid-but-non-PIN '3-oxa-2,4-dithiapentane'). NARROW (every backbone atom a
# chalcogen, both termini the same JUNIOR chalcogen) so carbon-in-chain skeletal
# names (2,4,7,10-tetraoxaundecane, methoxymethane) are untouched. Graph
# classifier, fail-closed; cascade-continuation on None. ---
_register_dispatch(
    class_id=StoutClass.HETEROCHALCOGEN_ABA, priority=47.55, tier=1,
    predicate=_is_heterochalcogen_aba, handler=_handle_heterochalcogen_aba,
    iupac_section="Blue Book P-68.4.2.1 / P-21.2.3.1",
    description="Pure-chalcogen a[ba]n parent hydride "
                "(dithioxane / methyldithioxane / dimethyldithioxane); "
                "graph classifier, fail-closed",
)


# --- W3-P14: HOMONUCLEAR_PNICTOGEN_CHAIN at priority 47.6 (after ---
# CATENATED_HYDRIDE@47.5, before FREE_HOMONUCLEAR_G14_HYDRIDE@47.7 / DINUCLEAR@48).
# The Group-15 pnictogen analogue of POLYAZANE@47 (N -> polyazane): a homonuclear
# unbranched H-saturated chain of >=2 identical P/As/Sb/Bi atoms (P-68.3.2.2):
# PP -> diphosphane, pentaarsane, dibismuthane. All 'unknown organic compound'
# before (MONONUCLEAR@45 declines a >=2-atom chain, the heteronuclear DINUCLEAR@48
# declines a homonuclear one, ORGM@50 declines a nonmetal). Graph classifier,
# fail-closed; cascade-continuation on None. ---
_register_dispatch(
    class_id=StoutClass.HOMONUCLEAR_PNICTOGEN_CHAIN, priority=47.6, tier=1,
    predicate=_is_homonuclear_pnictogen_chain,
    handler=_handle_homonuclear_pnictogen_chain,
    iupac_section="Blue Book P-68.3.2.2 / P-21.2.2",
    description="Homonuclear Group-15 catenated parent hydride "
                "(diphosphane / pentaarsane / dibismuthane); graph classifier, "
                "fail-closed",
)


# --- W3-P14: PNICTOGEN_CARBOXYLIC_ACID at priority 47.65 (after ---
# HOMONUCLEAR_PNICTOGEN_CHAIN@47.6, before FREE_HOMONUCLEAR_G14_HYDRIDE@47.7).
# An added-carbon -carboxylic acid on a Group-15 P/As/Sb parent hydride
# (P-68.3.2.3.1): H2P-COOH -> phosphanecarboxylic acid. Runs AHEAD of the generic
# acid namer (GENERAL@99999), which otherwise picks the carboxyl C as parent and
# emits the non-PIN '1-phosphanylmethanoic acid'. Graph classifier, fail-closed
# (a P=O phosphonic/phosphinic acid keeps its retained acid); cascade-continuation
# on None. ---
_register_dispatch(
    class_id=StoutClass.PNICTOGEN_CARBOXYLIC_ACID, priority=47.65, tier=1,
    predicate=_is_pnictogen_carboxylic_acid,
    handler=_handle_pnictogen_carboxylic_acid,
    iupac_section="Blue Book P-68.3.2.3.1",
    description="Added-carbon -carboxylic acid on a P/As/Sb parent hydride "
                "(phosphanecarboxylic acid); graph classifier, fail-closed",
)


# --- Wave-3: FREE_HOMONUCLEAR_G14_HYDRIDE at priority 47.7 (after ---
# CATENATED_HYDRIDE@47.5, before DINUCLEAR_HYDRIDE@48 / ORGM@50). A free-molecule
# homonuclear Group-14 catenated parent hydride (P-21.2.3 / P-68.2.3): disilane,
# trisilane, digermane, and the P-68.2.3 unsaturated digermene / disilene. These
# reached the terminal 'inorganic compound (not supported)' fallback before
# (detect_metal_complex declines the multimetal single fragment, ORGM@50 never
# engages). Graph classifier, fail-closed; cascade-continuation on None. ---
_register_dispatch(
    class_id=StoutClass.FREE_HOMONUCLEAR_G14_HYDRIDE, priority=47.7, tier=1,
    predicate=_is_free_homonuclear_g14_hydride,
    handler=_handle_free_homonuclear_g14_hydride,
    iupac_section="Blue Book P-21.2.3 / P-68.2.3",
    description="Free homonuclear Group-14 catenated parent hydride "
                "(disilane / trisilane / digermene); graph classifier, fail-closed",
)


# --- v23 Phase 10: DINUCLEAR_HYDRIDE at priority 48 (after POLYAZANE@47, before ---
# ORGANOMETALLIC@50). A two-atom Group-14/Group-15 catenated parent hydride
# (P-69.5.3 two-class-2-metal substitutive): [GeH3][SbH2] -> germylstibane,
# [SiH3][AsH2] -> silylarsane, [PbH3][BiH2] -> plumbylbismuthane (Group-15 is the
# P-41-senior parent; Group-14 is the -yl substituent). All 'X compound (not
# supported)' before this phase (detect_metal_complex returns None for the
# multimetal single fragment, so ORGM@50 never engages). Graph classifier,
# fail-closed (exactly one Group-14 + one Group-15 hub, H-saturated, single bond);
# cascade-continuation on None per D-02. ---
_register_dispatch(
    class_id=StoutClass.DINUCLEAR_HYDRIDE, priority=48, tier=1,
    predicate=_is_dinuclear_hydride, handler=_handle_dinuclear_hydride,
    iupac_section="Blue Book P-69.5.3 / P-68 / P-41",
    description="Di-nuclear Group-14/Group-15 catenated parent hydride "
                "(germylstibane / silylarsane / plumbylbismuthane); graph "
                "classifier, fail-closed",
)


_register_dispatch(
    class_id=StoutClass.AZINIC_DERIVATIVE, priority=48.3, tier=1,
    predicate=_is_azinic_derivative, handler=_handle_azinic_derivative,
    iupac_section="Blue Book P-61.5.3",
    description="Ylidene azinic acid (aci-nitro) parents "
                "(ethylideneazinic acid); graph classifier, fail-closed",
)
_register_dispatch(
    class_id=StoutClass.CUMULATIVE_ZWITTERION, priority=48.35, tier=1,
    predicate=_is_cumulative_zwitterion, handler=_handle_cumulative_zwitterion,
    iupac_section="Blue Book P-74.1.1 / P-74.2.1.3 / P-74.2.2.1.1 / .2 / .6",
    description="Same-parent -ium-...-ide zwitterion on a homogeneous heteroatom "
                "chain (1,2,2,2-tetramethylhydrazin-2-ium-1-ide / triaz-2-en-2-"
                "ium-1-ide / dioxidan-2-ium-1-ide); graph classifier, fail-closed",
)
_register_dispatch(
    class_id=StoutClass.YLIDE, priority=48.36, tier=1,
    predicate=_is_ylide, handler=_handle_ylide,
    iupac_section="Blue Book P-74.2.1.1",
    description="Ylide: onium cation + adjacent carbanion -> carbanion -ide parent "
                "with onium -aniumyl prefix (2-(trimethylazaniumyl)propan-2-ide); "
                "graph classifier, fail-closed",
)
_register_dispatch(
    class_id=StoutClass.HETERONE, priority=48.4, tier=1,
    predicate=_is_heterone, handler=_handle_heterone,
    iupac_section="Blue Book P-64.1.2.2 / P-64.4.1",
    description="Si/Ge/P/As heterone parents (dimethylsilanone / "
                "methylsilanone); graph classifier, fail-closed",
)
_register_dispatch(
    class_id=StoutClass.SULFINE, priority=48.5, tier=1,
    predicate=_is_sulfine, handler=_handle_sulfine,
    iupac_section="Blue Book P-64.4.2",
    description="Acyclic thiocarbonyl S-oxides (propylidene-lambda4-"
                "sulfanone); graph classifier, fail-closed",
)
_register_dispatch(
    class_id=StoutClass.PSEUDOKETONE_HETERO, priority=48.6, tier=1,
    predicate=_is_pseudoketone_hetero, handler=_handle_pseudoketone_hetero,
    iupac_section="Blue Book P-64.1.2.1 (b) / P-64.5.2.2",
    description="Acyl on Si/Ge/P/As hub (1-silylethan-1-one / "
                "1-phosphanylbutan-1-one); graph classifier, fail-closed",
)
# --- W3-P14: ACYL_CHALCOGENCHAIN_PSEUDOKETONE at priority 48.65 (after ---
# PSEUDOKETONE_HETERO@48.6, before HETEROIMINE@48.7). An acyl group terminating a
# homogeneous chain of >=3 identical chalcogens (P-68.4.1.3): CH3CH2-CO-O-O-OH ->
# 1-trioxidanylpropan-1-one (the chalcogen chain is the 'trioxidanyl' substituent
# on the carbonyl parent). The >=3-chalcogen gate keeps ordinary esters (R-CO-O-C),
# thioesters (R-CO-S-C), carboxylic acids (-CO-OH) and peroxy acids (-CO-O-OH) out.
# Was 'unknown organic compound'. Graph classifier, fail-closed. ---
_register_dispatch(
    class_id=StoutClass.ACYL_CHALCOGENCHAIN_PSEUDOKETONE, priority=48.65, tier=1,
    predicate=_is_acyl_chalcogenchain_pseudoketone,
    handler=_handle_acyl_chalcogenchain_pseudoketone,
    iupac_section="Blue Book P-68.4.1.3",
    description="Acyl on a homogeneous >=3-chalcogen chain "
                "(1-trioxidanylpropan-1-one); graph classifier, fail-closed",
)
_register_dispatch(
    class_id=StoutClass.LAMBDA5_PHOSPHANIMINE, priority=48.66, tier=1,
    predicate=_is_lambda5_phosphanimine, handler=_handle_lambda5_phosphanimine,
    iupac_section="Blue Book P-74.2.1.5",
    description="Phosphine imide R3X=N-R at lambda5 (N-ethyl-P,P,P-triphenyl-"
                "lambda5-phosphanimine); graph classifier, fail-closed",
)
_register_dispatch(
    class_id=StoutClass.HETEROIMINE, priority=48.7, tier=1,
    predicate=_is_heteroimine, handler=_handle_heteroimine,
    iupac_section="Blue Book P-62.3.1.3",
    description="Heteroatom imine X=NH on a P/As/Si mononuclear hub "
                "(1-methylphosphanimine); graph classifier, fail-closed",
)
# --- W3-P13: LAMBDA_SULFANE_IMINE_OXIDE at priority 48.75 (after HETEROIMINE@48.7,
# before KETENE@49). The mononuclear lambda-sulfane imine/oxide family
# (P-68.4.3.3-.8): a single non-ring S/Se/Te hub of non-standard valence bearing
# >=1 imine (=N-H/=N-R) plus optional =O and organyls — sulfimide / sulfonediimine
# / sulfoximide / sulfur di-/tri-imide. =O outranks =N (P-41): oxo -> -one/-dione
# suffix + (R-imino) prefix; imine-only -> -imine/-diimine/-triimine suffix. All
# 'unknown' before this; the earlier chalcogen/imine handlers (heterone@48.4,
# sulfine@48.5, heteroimine@48.7) all decline these (no S hub / wrong degree).
# Graph classifier, fail-closed; cascade-continuation on None per D-02. ---
_register_dispatch(
    class_id=StoutClass.LAMBDA_SULFANE_IMINE_OXIDE, priority=48.75, tier=1,
    predicate=_is_lambda_sulfane_imine_oxide,
    handler=_handle_lambda_sulfane_imine_oxide,
    iupac_section="Blue Book P-68.4.3.3 / .4 / .5 / .6 / .7 / .8",
    description="Mononuclear lambda-sulfane imine/oxide (S,S-diethyl-N-phenyl-"
                "lambda4-sulfanimine / diphenyl-lambda6-sulfanediimine / "
                "dimethyl(phenylimino)-lambda6-sulfanone); graph classifier, "
                "fail-closed",
)

# --- Wave-2 completion: KETENE at priority 49 (after DINUCLEAR_HYDRIDE@48, ---
# before ORGANOMETALLIC@50). The exact (halo)ketene heterocumulene named on the
# ethenone parent (P-64.2.2.4): C=C=O -> ethenone, BrC(Br)=C=O ->
# dibromoethenone (both BB verbatim; 'unknown' before this). Alkyl/aryl and
# ylidene ketenes fail the classifier and cascade (general ketone principles,
# not built). Graph classifier, fail-closed; cascade-continuation on None. ---
_register_dispatch(
    class_id=StoutClass.KETENE, priority=49, tier=1,
    predicate=_is_ketene, handler=_handle_ketene,
    iupac_section="Blue Book P-64.2.2.4",
    description="(Halo)ketene heterocumulene on the ethenone parent "
                "(ethenone / dibromoethenone); graph classifier, fail-closed",
)


# --- Wave-2 completion: RING_CHALCOGEN_OXIDE at priority 49.5 (after
# KETENE@49, before ORGANOMETALLIC@50). A neutral ring S/Se/Te with 1-2
# exocyclic =O on an otherwise-bare ring system, named additively on the
# intact ring parent (P-25.6 / P-74.3.1.3): dibenzo[b,d]thiophene 5-oxide /
# 5,5-dioxide (the ring-S sibling of pyridine 1-oxide). The de-oxidised base
# must resolve both a name AND an authoritative chalcogen locant; anything
# else cascades. Fail-closed; cascade-continuation on None. ---
_register_dispatch(
    class_id=StoutClass.RING_CHALCOGEN_OXIDE, priority=49.5, tier=1,
    predicate=_is_ring_chalcogen_oxide, handler=_handle_ring_chalcogen_oxide,
    iupac_section="Blue Book P-25.6 / P-74.3.1.3",
    description="Ring-chalcogen oxide named additively on the ring parent "
                "(dibenzo[b,d]thiophene 5-oxide / 5,5-dioxide); fail-closed",
)
# --- W2E-P1FG: HYDRO_FUSED_PEROXOL at priority 49.6 (before ORGM@50). The
# -OOH suffix on an sp3 carbon of a partially saturated fused carbocycle
# (1,2,3,4-tetrahydronaphthalene-1-peroxol, BB verbatim P-63.4.1) — the
# fallback ring-parent path silently DROPPED the -OOH before this. Fail-
# closed classifier+namer; cascade-continuation on None. ---
_register_dispatch(
    class_id=StoutClass.HYDRO_FUSED_PEROXOL, priority=49.6, tier=1,
    predicate=_is_hydro_fused_peroxol, handler=_handle_hydro_fused_peroxol,
    iupac_section="Blue Book P-63.4.1",
    description="-OOH suffix on a partially saturated fused carbocycle "
                "(1,2,3,4-tetrahydronaphthalene-1-peroxol); fail-closed",
)
# --- W2E-D3: THIOIMIDE at priority 49.7 (after HYDRO_FUSED_PEROXOL@49.6,
# before ORGM@50). The acyclic N-H thioimide R-C(=S)-NH-C(=S)-R' (BB verbatim
# P-66.1.4.2: CH3-CS-NH-CS-CH3 -> N-(ethanethioyl)ethanethioamide). The
# O-imide analogue is named by the decomposition engine, which never reaches
# the thio case because its amide-bond SMARTS requires C(=O); this dedicated
# graph classifier + namer fills exactly that gap. Fail-closed; cascade-
# continuation on None (N-substituted / branched / aryl / mixed-O forms). ---
_register_dispatch(
    class_id=StoutClass.THIOIMIDE, priority=49.7, tier=1,
    predicate=_is_thioimide, handler=_handle_thioimide,
    iupac_section="Blue Book P-66.1.4.2",
    description="Acyclic N-H thioimide R-C(=S)-NH-C(=S)-R' on the "
                "N-(alkanethioyl)alkanethioamide PIN; fail-closed",
)
# --- W3-P08: CYCLIC_POLYESTER at priority 49.8 (after THIOIMIDE@49.7, before
# ORGM@50). A lactide / cyclic di-/polyester -- a saturated monocyclic C/O ring
# with >=2 ester carbonyls -- named as a Hantzsch-Widman heterocycle with the
# acyl carbons as a -dione/-trione suffix (P-65.6.3.5.3): glycolide
# O=C1COC(=O)CO1 -> 1,4-dioxane-2,5-dione ('unknown' before this; the single-
# carbonyl lactone namer fails closed on the 2nd ring ester O). Fail-closed
# classifier+namer (declines single lactone / carbonate / anhydride); cascade-
# continuation on None. ---
_register_dispatch(
    class_id=StoutClass.CYCLIC_POLYESTER, priority=49.8, tier=1,
    predicate=_is_cyclic_polyester, handler=_handle_cyclic_polyester,
    iupac_section="Blue Book P-65.6.3.5.3",
    description="Lactide / cyclic di-/polyester named as an HW heterocycle "
                "-dione (glycolide -> 1,4-dioxane-2,5-dione); fail-closed",
)


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

# --- v23 Phase 12 follow-on: INOSITOL at priority 1700 (after CYCLOPHANE@1600, ---
# before DECOMPOSITION_PRE_GENERAL@99000 / GENERAL@99999). The nine inositol
# (cyclohexane-1,2,3,4,5,6-hexol) retained names are the PIN (P-104.2.1) and are
# OPSIN-UNPARSEABLE, so this is a NAME-EXACT recogniser. Hard-gated InChIKey
# lookup over the cyclohexanehexol skeleton -> fail-closed (only one of the nine
# fully-stereodefined inositols matches; an undefined-stereo or substituted hexol
# keeps the systematic name). Placed at 1700 because the dense 100-1600 region has
# no free hundreds-slot; it sits ABOVE the retained/skeletal/cyclophane block but
# still intercepts before the systematic cyclohexanehexol naming in GENERAL.
_register_dispatch(
    class_id=StoutClass.INOSITOL, priority=1700, tier=2,
    predicate=_is_inositol, handler=_handle_inositol,
    iupac_section="Blue Book P-104.2.1 (cyclitols / inositol retained names)",
    description="Inositol cyclitol retained PIN (myo-/scyllo-/cis-/epi-/neo-/allo-/"
                "muco-/D-chiro-/L-chiro-inositol); name-exact (OPSIN-unparseable), "
                "hard-gated InChIKey lookup, fail-closed; routes to rules.inositols.name_inositol",
    side_effect_inventory=(),
)

# --- v23 Phase 14: NUCLEOSIDE/NUCLEOTIDE decoration at priority 1800 ---
# After RETAINED_NAME@1300 (so bare nucleosides + the retained adenylic/inosinic
# monophosphates keep their catalog names) and INOSITOL@1700, before
# DECOMPOSITION_PRE_GENERAL@99000 (which would otherwise mis-decompose ATP/ADP
# into 'adenosine diphosphoric acid'). strip-and-recognise engine; OPSIN-RT;
# fail-closed (declines any molecule it cannot fully account for).
_register_dispatch(
    class_id=StoutClass.NUCLEOSIDE, priority=1800, tier=2,
    predicate=_is_nucleoside, handler=_handle_nucleoside,
    iupac_section="Blue Book P-105.2 / P-106 (nucleoside/nucleotide decoration)",
    description="Decorated nucleoside/nucleotide (5'-mono/di/tri-phosphate + O-acyl "
                "ester); strip-and-recognise; routes to rules.nucleosides.name_nucleoside",
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
