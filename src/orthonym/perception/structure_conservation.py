"""Tier-B structure-conservation checks (Wave-8 P10 -- the "E1 atom-coverage
certificate" from the cross-tool audit).

Java-free, source-level structural checks that close leaks where the winning
naming path emits a name for a DIFFERENT molecule than the input (atom-drop,
charge-drop) and only OPSIN's SELF-01 constitutional gate
(``namer._final_opsin_validity_gate`` / ``_self_consistency_decision``) would
otherwise have caught it. SELF-01 FAILS OPEN when no JAR is present
(``_validity_gate_jar_present()`` returns False), so in a no-Java deployment
these leaks ship as confidently-wrong names. The functions here are called
UNCONDITIONALLY from ``namer.py::Orthonym._name_impl`` -- i.e. NOT gated by
``self._disable_opsin_validity_gate`` -- so they protect the raw path too.

Design principle (accuracy-first, precision-over-recall, per
``./skills/fix-methodology.md``): every check here is scoped as
NARROWLY as the evidence supports. A check that cannot be made precise
(false-veto risk on a currently-correct name) is deliberately left
unimplemented rather than shipped -- see the docstring on
``partial_sat_sp3_substituent_drop`` for a worked example of the
investigation this required (an initial, broader version of that check
false-vetoed 17/25 real-corpus hexahydro-/octahydronaphthalene names before
being narrowed to the exact shape it is proven safe for).
"""

from typing import Optional, Tuple

import re

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


# ---------------------------------------------------------------------------
# Java-free structural invariants of the INPUT molecule
# ---------------------------------------------------------------------------

def input_invariants(mol: Chem.Mol) -> Tuple[int, int, int]:
    """Java-free structural invariants of the INPUT molecule.

    Returns ``(heavy_atom_count, dou, bond_order_sum)``:

    - ``heavy_atom_count``: ``mol.GetNumHeavyAtoms()`` (authoritative).
    - ``dou``: rings + sum(bond_order - 1) over heavy-heavy bonds. A pure
      GRAPH invariant, deliberately NOT the classical H-count
      degree-of-unsaturation formula (which conflates formal charges/
      radicals and would misfire on ions/radicals). Using the SUM of
      (order - 1) rather than a discrete count of "multiple bonds" is what
      makes this exact for aromatic rings too: RDKit reports an aromatic
      bond's order as 1.5 (``GetBondTypeAsDouble()``), so each contributes
      0.5 -- benzene's 6 aromatic C-C bonds contribute 6*0.5=3, plus 1 ring,
      for the expected DoU of 4 (matching the Kekule double-bond count).
    - ``bond_order_sum``: rounded sum of heavy-heavy bond orders -- a coarse
      diagnostic signal, not consumed by any veto below.

    Pure RDKit. No OPSIN import, no subprocess -- safe to call unconditionally
    on the no-Java path.
    """
    heavy = mol.GetNumHeavyAtoms()
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    unsaturation = 0.0
    bond_order_sum = 0.0
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if a1.GetAtomicNum() <= 1 or a2.GetAtomicNum() <= 1:
            continue  # heavy-heavy bonds only
        order = bond.GetBondTypeAsDouble()
        bond_order_sum += order
        if order > 1.0:
            unsaturation += order - 1.0
    dou = n_rings + int(round(unsaturation))
    return heavy, dou, int(round(bond_order_sum))


def net_formal_charge(mol: Chem.Mol) -> int:
    """Net formal charge of the molecule (Java-free, always available)."""
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())


# ---------------------------------------------------------------------------
# Charge-conservation veto
# ---------------------------------------------------------------------------
# Closes the P5-family leak class: a substituted / hypervalent-at-the-charged-
# atom anion whose charge-bearing route fails internally and falls through to
# a charge-blind naming path, silently dropping the formal charge:
#   [I-](CCO)c1ccccc1  -> raw 'iodobenzenylethan-1-ol' (neutral; charge on a
#                          substituted lambda3-iodanuide iodine, degree 2)
#   C[Si-](C)(C)[H]    -> raw 'trimethylsilane' (neutral; charge on silicon)
#   [B-](CCO)(C)(C)C   -> raw '2-(trimethylboryl)ethan-1-olate' (charge on
#                          boron dropped, BUT the name coincidentally ends in a
#                          real anion suffix '-olate' belonging to the terminal
#                          CH2-OH -- so the NAME-shape check cannot catch it; it
#                          stays a DOCUMENTED, intentional gap. See the phase
#                          report and the unit test named
#                          test_charge_dropped_known_limitation_boron_olate.)
#
# TWO independent guards, both required to fire (precision over recall):
#
# 1. NAME-shape: the shipped name lacks any ionic suffix. This alone is a
#    NAME check (NOT a dispatch-class check) because a live probe showed
#    CORRECT charged names ship through EVERY class_id -- 'benzenediazonium',
#    'ethylsulfanylium', 'ethane-1,2-bis(aminium)' are all class_id=GENERAL --
#    so class-based gating would over-veto them.
#
# 2. CHARGE-CENTRE atom scope: the veto fires only when a charge-bearing atom
#    is a p-block MAIN-GROUP metalloid / heavy-p-block element (Groups 13-15
#    excluding C and N) OR a HYPERVALENT/substituted halogen (degree >= 2).
#    These are exactly the centres whose charged routing Orthonym does not yet
#    fully cover, so a fall-through silently neutralizes them. The routine
#    ionic centres are DELIBERATELY excluded, because they have dedicated
#    charged routes that emit a correct ionic-suffixed name:
#      - N+  (ammonium / choline / carnitine / aminium): would over-veto
#        every quaternary-ammonium functional-class name -- CONFIRMED risk on
#        a corpus probe (choline phospholipids named '...enol', charge on N+).
#      - O-  (carboxylate / alkoxide / phenolate), C- (carbanide -> '...ide'),
#        S/Se/Te (thiolate / sulfonium): all routine, all correctly suffixed.
#    A correct main-group charged name (methylsilanuide, tetramethyl-
#    phosphanuide, silylium, phosphanium, ...) always carries its ionic suffix,
#    so guard 1 already spares it even though its centre is in-scope -- the two
#    guards together fire ONLY on a genuine main-group charge drop.
_IONIC_SUFFIX_RE = re.compile(r'(?:ate|ide|ium)\)?$', re.IGNORECASE)

# Charge-centre elements whose charged routing Orthonym does not fully cover.
# Group 13 (B, Al, Ga, In, Tl), Group 14 minus C (Si, Ge, Sn, Pb), Group 15
# minus N (P, As, Sb, Bi). C/N/O/S/Se/Te and simple (degree<=1) halide anions
# are EXCLUDED -- see the block comment above.
_MAIN_GROUP_CHARGE_CENTRES = frozenset({
    5, 13, 31, 49, 81,   # B, Al, Ga, In, Tl
    14, 32, 50, 82,      # Si, Ge, Sn, Pb
    15, 33, 51, 83,      # P, As, Sb, Bi
})
_HALOGENS = frozenset({9, 17, 35, 53, 85})  # F, Cl, Br, I, At


def looks_like_ionic_name(name: str) -> bool:
    """Coarse recognizer: does ``name`` textually encode ionic character via
    a recognised P-72/P-73 charge suffix?

    Deliberately LENIENT (broad match), because a false NEGATIVE here (a
    correctly-charged name this regex fails to recognise) only means a leak
    is missed -- safe, per precision-over-recall. Every currently-shipping
    net-charged PIN this task could enumerate -- the ~45 net-charged rows of
    `` (azanylium, silylium, ...anium,
    ...ide, ...ate, bis(...ide)/bis(...ate)/bis(...ium) forms) -- matches this
    pattern.

    Cation suffixes ('-ium', '-ylium', '-anium', '-onium') all end in the
    literal substring "ium". Anion suffixes ('-ide', '-uide', '-ate',
    '-oate', '-olate', '-thiolate', '-aminide', ...) all end in "ide" or
    "ate". A trailing ``)`` is allowed for a multiplied bis(...)/tris(...)
    form (e.g. ``ethane-1,2-bis(aminium)``).
    """
    if not name:
        return False
    return bool(_IONIC_SUFFIX_RE.search(name))


def _has_main_group_charge_centre(mol: Chem.Mol) -> bool:
    """True iff any charge-bearing atom is a HYPERVALENT / substituted
    (degree >= 2) halogen -- the one charge-centre class proven to fall through
    to a genuinely WRONG-CONNECTIVITY neutral name (``[I-](CCO)c1ccccc1`` ->
    '2-iodobenzenylethan-1-ol', a different molecule).

    NARROWED (W8-P10 controller 2026-07-18): the group-13/14/15 main-group
    metalloids (B/Al/Ga/In/Tl, Si/Ge/Sn/Pb, P/As/Sb/Bi) were originally in
    scope on the assumption that a correct charged name always carries an ionic
    suffix (so guard 1 would spare it). That assumption is FALSE for the
    established over-coordinated-metalloid class, which is DELIBERATELY named as
    its neutral parent hydride/organyl (``C[Si-](C)(C)C`` -> 'tetramethyl-
    silane', ``C[Sn-](C)(C)C`` -> 'tetramethylstannane';
    tests/unit/rules/test_charged_suffixes_ft6.py) -- a charge NORMALIZATION of
    the same skeleton, not a wrong-connectivity drop. Including those groups
    over-vetoed those correct names (3 regressions on the full rules HEAD-A/B).
    They are excluded here; any residual main-group charge-drop stays a
    gated-safe no-Java residual (precision-over-recall). ``_MAIN_GROUP_CHARGE_
    CENTRES`` is retained for documentation only.
    """
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() == 0:
            continue
        z = atom.GetAtomicNum()
        if z in _HALOGENS and atom.GetDegree() >= 2:
            return True
    return False


def charge_dropped(mol: Chem.Mol, name: str) -> bool:
    """True iff the input carries a nonzero NET formal charge, a charge-bearing
    atom is a main-group centre Orthonym's charged routing does not fully
    cover (see ``_has_main_group_charge_centre``), AND ``name`` does not
    textually encode ionic character (see ``looks_like_ionic_name``).

    Scope: NET charge only -- a genuinely charged single ion (or an ionic
    assembly whose charges do not cancel). A net-ZERO zwitterion (nitro,
    N-oxide, amino-acid zwitterion, sulfonium/ammonium ylide, ...) never
    reaches this check (``net_formal_charge(mol) == 0`` short-circuits), so
    the entire (large) neutral-by-charge-separation naming surface is
    untouched by construction. The main-group-centre guard further excludes
    every routine ionic centre (ammonium/choline N+, carboxylate/alkoxide O-,
    carbanide C-, thiolate/sulfonium S), which have dedicated charged routes
    -- so this fires ONLY on a genuine main-group charge drop.

    Callers MUST exclude an already-descriptive-fallback ``name`` (see
    ``errors.is_failure_name``) before calling this -- an honest abstention
    ('unknown organic compound', '<metal> compound (not supported)', ...) is
    not a "dropped charge" and must not be double-suppressed.
    """
    if not name:
        return False
    if net_formal_charge(mol) == 0:
        return False
    if not _has_main_group_charge_centre(mol):
        return False
    return not looks_like_ionic_name(name)


# ---------------------------------------------------------------------------
# R12-spillover veto: sp3-ring substituent drop on the partially-saturated
# fused-carbocycle emitter
# ---------------------------------------------------------------------------

def partial_sat_sp3_substituent_drop(mol: Chem.Mol, name: str) -> bool:
    """R12 spillover (2026-07-17, W8-P1/R12 adversarial-verify workflow;
    closed here per W8-P10).

    ``rules.polycyclics.name_partially_saturated_carbocycle`` (the
    "tetralin"-class emitter: one fully-aromatic ring fused to one fully-sp3
    ring, e.g. 1,2,3,4-tetrahydronaphthalene) NEVER bakes a saturated
    (sp3)-ring substituent into its own bare parent name -- by construction,
    its ``_partial_sat_substituent_prefix`` helper explicitly skips every
    atom in ``sat_set`` (``if idx in sat_set: continue``) and its own
    docstring records the design: "the composer's normal decoration path
    already carries substituents on the SATURATED (sp3) ring" / "It must NOT
    add sp3-ring substituents (the composer does that)". The ONLY way an
    sp3-ring substituent (the -CH3/-NH2/-OH/... at e.g. tetralin C2) reaches
    the shipped name is the downstream composer enrichment pass
    (``_enrich_handler_name`` -> ``_integrate_universal_prefixes``).

    That hand-off is where the leak lives: enrichment can silently return an
    EMPTY prefix (``_integrate_universal_prefixes`` catches any discovery
    exception/None and returns ``""``) when its ``oriented_ring``/
    ``exclude_atoms`` plumbing for the ``partial_sat`` branch does not match
    the parent atom set it is given -- confirmed live for a plain second
    ring-methyl (``oriented_ring=None``) and for a heteroatom substituent
    whose own principal-group match incorrectly gets folded into
    ``exclude_atoms`` even though the bare name never represents it.

    Detection (structural, not a heuristic): because ``bare`` provably NEVER
    encodes an sp3-ring substituent, ``name == bare`` in the presence of a
    structurally-confirmed sp3-ring off-ring substituent is a DETERMINISTIC
    proof of drop, not a guess -- there is no other route by which that
    substituent could be missing from the string while a distinct heavy atom
    hangs off an sp3 ring position.

    Scope guard (precision-over-recall): restricted to the exact
    "fully-aromatic-ring + fully-saturated-ring" (tetralin-class) shape --
    every ring atom must be EITHER aromatic OR in ``sat_set``. An initial,
    broader version of this check (using ``sat_set`` alone, without this
    guard) FALSE-VETOED 17 of 25 real hits on a corpus sweep of
    ``benchmark_multi_corpus_results.csv`` -- all of them mixed alkene/hydro
    systems (hexahydro-/octahydronaphthalene, hydroazulene, ...) whose
    substituents are, in fact, already correctly enriched; those go through a
    materially different mechanism (``_partial_sat_substituent_prefix``
    itself bakes in substituents on any NON-sat_set ring atom, aromatic or
    not) that this function must not second-guess. Narrowing to the clean
    two-ring aromatic/saturated split eliminated every false positive found
    (0/91 on the combined synthetic + corpus validation set) while still
    catching the 3 documented target leaks plus 2 additional real-corpus
    leaks the narrower check newly surfaced.

    IUPAC cite: P-25.3.1 / P-58.2.2.3 (partially saturated carbocycles;
    detachable-prefix placement); mirrors the existing R12 fusion-atom veto
    inside ``name_partially_saturated_carbocycle`` itself (ring-FUSION atom
    substituents), which this function does NOT duplicate (that path already
    returns ``None`` -- ``bare`` -- for those cases, so this function is a
    no-op on them).
    """
    if not name:
        return False
    from ..rules.polycyclics import (
        name_partially_saturated_carbocycle, _offring_substituent_atoms,
    )
    from ..rules.partial_saturation import detect_carbocyclic_partial_saturation

    bare = name_partially_saturated_carbocycle(mol)
    if not bare:
        return False  # not this emitter's molecule at all
    if name != bare:
        return False  # enrichment added something -- not (this) leak

    ring_atoms = set()
    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms.update(ring)
    sat_info = detect_carbocyclic_partial_saturation(mol, ring_atoms)
    if sat_info is None:
        return False
    atom_to_locant = sat_info.get('atom_to_locant') or {}
    sat_set = set(sat_info.get('saturated_indices') or [])
    if not atom_to_locant or not sat_set:
        return False

    # Tetralin-class shape guard -- see docstring.
    leftover = [
        idx for idx in ring_atoms
        if idx not in sat_set and not mol.GetAtomWithIdx(idx).GetIsAromatic()
    ]
    if leftover:
        return False

    offring = _offring_substituent_atoms(mol, atom_to_locant, sat_set)
    return bool(offring)


# ---------------------------------------------------------------------------
# Fused/bridged/spiro ring-system atom-drop veto (Wave-8 P6 Task 6.0)
# ---------------------------------------------------------------------------
# Closes the leak where a molecule whose ring atoms span TWO OR MORE SSSR
# rings sharing at least one atom (ortho-fused, bridged, or spiro -- any
# ring-junction topology) gets named by the "chain" handler as a BARE,
# unsubstituted monocyclic cycloalkane/cycloalkene (optionally stereo-
# prefixed), e.g. `C1CC[C@@H]2CCCN[C@@H]2C1` (a piperidine-fused
# cyclohexane) -> raw `(3R,4R)-cyclohexane`. That name is structurally
# impossible: a bare "cyclo<stem>ane" parent, by IUPAC construction, names
# exactly ONE ring, so it can never simultaneously account for a second
# ring's atoms. No name-shape/ring-size arithmetic is needed to prove the
# drop -- the mere existence of >=2 ring-sharing SSSR rings on a molecule
# whose shipped name has this exact bare shape is deterministic proof that
# atoms are unaccounted for (there is no legitimate bare-monocyclic name for
# any fused/bridged/spiro polycyclic).
#
# The bare-name regex intentionally matches ONLY the no-substituent shape
# (an optional leading stereo-descriptor parenthetical, then
# "cyclo<letters>e"): any real substituent or suffix token (a locant-
# prefixed substituent, a principal-group suffix like "-ol"/"-oic acid",
# ring-fusion terms like "bicyclo"/"spiro"/"decahydro") breaks the match, so
# this never fires on a correctly-decorated or correctly fused/bridged/spiro
# name -- only on the literal drop shape.
_BARE_MONOCYCLIC_RE = re.compile(r'^(\([0-9A-Za-z,]+\)-)?cyclo[a-z]+e$')


def fused_ring_atom_drop(mol: Chem.Mol, name: str) -> bool:
    """True iff `name` is a bare (unsubstituted, optionally stereo-prefixed)
    monocyclic cycloalkane/cycloalkene name for a molecule whose rings span
    2+ SSSR rings sharing at least one atom (fused, bridged, or spiro).

    Such a name is provably wrong: a single "cyclo<stem>ane" parent cannot
    represent a second ring's atoms, regardless of what those atoms are
    (carbocyclic or heteroatom-containing). See module-level comment above
    for the full design record.
    """
    if not name:
        return False
    if not _BARE_MONOCYCLIC_RE.match(name):
        return False
    atom_rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    if len(atom_rings) < 2:
        return False
    for i in range(len(atom_rings)):
        for j in range(i + 1, len(atom_rings)):
            if atom_rings[i] & atom_rings[j]:
                return True
    return False


__all__ = [
    "input_invariants",
    "net_formal_charge",
    "looks_like_ionic_name",
    "_has_main_group_charge_centre",
    "charge_dropped",
    "partial_sat_sp3_substituent_drop",
    "fused_ring_atom_drop",
]
