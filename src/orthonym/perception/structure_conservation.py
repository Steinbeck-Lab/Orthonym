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
# [I-](CCO)c1ccccc1 -> raw 'iodobenzenylethan-1-ol' (neutral; charge on a
# substituted lambda3-iodanuide iodine, degree 2)
# C[Si-](C)(C)[H] -> raw 'trimethylsilane' (neutral; charge on silicon)
# [B-](CCO)(C)(C)C -> raw '2-(trimethylboryl)ethan-1-olate' (charge on
# boron dropped, BUT the name coincidentally ends in a
# real anion suffix '-olate' belonging to the terminal
# CH2-OH -- so the NAME-shape check cannot catch it; it
# stays a DOCUMENTED, intentional gap. See the phase
# report and the unit test named
# test_charge_dropped_known_limitation_boron_olate.)
#
# TWO independent guards, both required to fire (precision over recall):
#
# 1. NAME-shape: the shipped name lacks any ionic suffix. This alone is a
# NAME check (NOT a dispatch-class check) because a live probe showed
# CORRECT charged names ship through EVERY class_id -- 'benzenediazonium',
# 'ethylsulfanylium', 'ethane-1,2-bis(aminium)' are all class_id=GENERAL --
# so class-based gating would over-veto them.
#
# 2. CHARGE-CENTRE atom scope: the veto fires only when a charge-bearing atom
# is a p-block MAIN-GROUP metalloid / heavy-p-block element (Groups 13-15
# excluding C and N) OR a HYPERVALENT/substituted halogen (degree >= 2).
# These are exactly the centres whose charged routing Orthonym does not yet
# fully cover, so a fall-through silently neutralizes them. The routine
# ionic centres are DELIBERATELY excluded, because they have dedicated
# charged routes that emit a correct ionic-suffixed name:
# - N+ (ammonium / choline / carnitine / aminium): would over-veto
# every quaternary-ammonium functional-class name -- CONFIRMED risk on
# a corpus probe (choline phospholipids named '...enol', charge on N+).
# - O- (carboxylate / alkoxide / phenolate), C- (carbanide -> '...ide'),
# S/Se/Te (thiolate / sulfonium): all routine, all correctly suffixed.
# A correct main-group charged name (methylsilanuide, tetramethyl-
# phosphanuide, silylium, phosphanium,...) always carries its ionic suffix,
# so guard 1 already spares it even though its centre is in-scope -- the two
# guards together fire ONLY on a genuine main-group charge drop.
# B3: added 'ite' (P-72.2.2/P-65.3.1's OTHER acid-anion suffix pair,
# '-ous acid' -> '-ite', sibling to '-ic acid' -> '-ate' -- chlorite,
# nitrite, sulfite, phosphite, hypochlorite,...). Missing it meant a
# genuinely-ionic HALOGEN-centred retained name ending in '-ite' (chlorite,
# for the hypervalent-Cl+ shape `_has_main_group_charge_centre` already
# flags in-scope) read as "does not look ionic" and was wrongly suppressed
# by `charge_dropped` as if the charge had been silently dropped, when it
# had not.
_IONIC_SUFFIX_RE = re.compile(r'(?:ate|ite|ide|ium)\)?$', re.IGNORECASE)

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
    `` (azanylium, silylium,...anium,
    ...ide,...ate, bis(...ide)/bis(...ate)/bis(...ium) forms) -- matches this
    pattern.

    Cation suffixes ('-ium', '-ylium', '-anium', '-onium') all end in the
    literal substring "ium". Anion suffixes ('-ide', '-uide', '-ate',
    '-oate', '-olate', '-thiolate', '-aminide',...) all end in "ide" or
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
    N-oxide, amino-acid zwitterion, sulfonium/ammonium ylide,...) never
    reaches this check (``net_formal_charge(mol) == 0`` short-circuits), so
    the entire (large) neutral-by-charge-separation naming surface is
    untouched by construction. The main-group-centre guard further excludes
    every routine ionic centre (ammonium/choline N+, carboxylate/alkoxide O-,
    carbanide C-, thiolate/sulfonium S), which have dedicated charged routes
    -- so this fires ONLY on a genuine main-group charge drop.

    Callers MUST exclude an already-descriptive-fallback ``name`` (see
    ``errors.is_failure_name``) before calling this -- an honest abstention
    ('unknown organic compound', '<metal> compound (not supported)',...) is
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

    ⚠ **This veto's ORIGINAL premise no longer holds, and the rewrite is the
    point.** It used to reason: ``rules.polycyclics.name_partially_saturated_-
    carbocycle`` "NEVER bakes a saturated -ring substituent into its own
    bare parent name -- by construction, its ``_partial_sat_substituent_prefix``
    helper explicitly skips every atom in ``sat_set``", so the ONLY route for an
    sp3-ring substituent was the composer's enrichment pass, and
    ``name == bare`` alongside an sp3-ring off-ring substituent was therefore a
    deterministic proof of drop.

    The hand-off that premise described was itself the defect. It derived a
    SECOND numbering (placing tetralin's 2-methyl at locant 1) and it split one
    substituent set across two prefix formatters (``2-methyl-6-methyl-`` where
    P-16.3.3 requires ``2,6-dimethyl-``). Both faults are fixed by making the
    producer the single speller of every ring substituent prefix, so ``bare``
    NOW encodes sp3-ring substituents, and the old test would false-veto every
    correct name in the class.

    Detection, restated for the current construction and still structural
    rather than heuristic: the producer publishes the off-ring atoms its name
    actually spells (``PartialSatName.spelled_offring_atoms`` -- prefixes plus
    the exocyclic atoms of any principal-characteristic-group suffix such as
    ``-2-carboxylic acid`` / ``-2-ol``). An off-ring heavy atom on an sp3 ring
    position that is in NEITHER that set NOR the enrichment the shipped name
    added is unaccounted for, and that is a drop. Subtracting the published set
    is not a loosening: before it was subtracted, this veto suppressed the very
    names that carry those atoms -- every ring-COOH member of the class emitted
    ``unknown organic compound`` while ``bare`` already held the correct
    ``1,2,3,4-tetrahydronaphthalene-2-carboxylic acid``.

    Scope guard (precision-over-recall): restricted to the exact
    "fully-aromatic-ring + fully-saturated-ring" (tetralin-class) shape --
    every ring atom must be EITHER aromatic OR in ``sat_set``. An initial,
    broader version of this check (using ``sat_set`` alone, without this
    guard) FALSE-VETOED 17 of 25 real hits on a corpus sweep of
    ``benchmark_multi_corpus_results.csv`` -- all of them mixed alkene/hydro
    systems (hexahydro-/octahydronaphthalene, hydroazulene,...) whose
    substituents are, in fact, already correctly spelled by
    ``_partial_sat_substituent_prefix`` and which this function must not
    second-guess. Narrowing to the clean
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
        name_partially_saturated_carbocycle_with_locants,
        _offring_substituent_atoms,
    )
    from ..rules.partial_saturation import detect_carbocyclic_partial_saturation

    produced = name_partially_saturated_carbocycle_with_locants(mol)

    ring_atoms = set()
    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms.update(ring)
    sat_info = detect_carbocyclic_partial_saturation(mol, ring_atoms)
    if sat_info is None:
        return False  # not this emitter's class at all
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

    # Ring-FUSION atom substituents belong to the producer's own R12 fusion
    # veto, and this function has always been a no-op on them (the producer
    # declines, so there was no `bare` to compare against). Keep it that way
    # explicitly now that the None case below is no longer an early exit.
    ri = mol.GetRingInfo()
    for idx in atom_to_locant:
        if ri.NumAtomRings(idx) < 2:
            continue
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetIdx() not in atom_to_locant and nb.GetAtomicNum() > 1:
                return False

    offring = _offring_substituent_atoms(mol, atom_to_locant, sat_set)
    if not offring:
        return False  # nothing on the saturated ring that could be dropped

    if produced is None:
        # The tetralin-class producer REFUSED. It fails closed exactly when it
        # cannot account for a ring substituent, so nothing downstream has a
        # verified hydro-fused parent to hang this substituent on.
        #
        # This branch is NOT optional. Before the producer learned to spell
        # sp3-ring substituents it emitted its bare parent here, and the veto
        # caught the drop by string equality against it. Once the producer
        # refuses instead, an early `return False` turns the refusal into a
        # LICENCE for a worse generator: measured Java-free on
        # `NC1CCc2ccccc2C1` -> `4-butylcyclohexan-1-amine`, a wrong molecule
        # (the aromatic half of the tetralin re-spelled as a butyl chain on a
        # monocycle) where the answer had been a clean abstention.
        # Removing a wrong output must never unmask a worse one.
        return True

    if name != produced.name:
        return False  # something else composed this name -- not (this) leak

    # Atoms the parent name itself already spells are not dropped -- the
    # producer publishes exactly which ones those are.
    offring -= produced.spelled_offring_atoms
    return bool(offring)


# ---------------------------------------------------------------------------
# Fused/bridged/spiro ring-system atom-drop veto (Wave-8 P6.0)
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


_ENE_RE = re.compile(r'-([0-9,()]+)-(?:di|tri|tetra|penta|hexa)?ene\b')
# 'oxo' is a detachable prefix glued to the stem ('2-oxobicyclo…', '2,4-dioxo…'),
# so there is NO word boundary after it -- match the multiplier + 'oxo', no \b.
_OXO_PREFIX_RE = re.compile(r'\b([0-9,]+)-(?:di|tri|tetra|penta|hexa)?oxo')
_ONE_SUFFIX_RE = re.compile(r'-([0-9,]+)-(?:di|tri|tetra|penta|hexa)?one\b')


def _ene_termini(name: str) -> set:
    """VB locants that carry a ring double bond in `name`. A plain locant ``n``
    means the bond n=(n+1) -> termini {n, n+1}; a compound ``n(m)`` -> {n, m}."""
    out: set = set()
    for block in _ENE_RE.findall(name):
        # block like "1,7" or "1(6),3" -- split on commas OUTSIDE parens
        for tok in re.findall(r'\d+\(\d+\)|\d+', block):
            m = re.match(r'(\d+)\((\d+)\)', tok)
            if m:
                out.update((int(m.group(1)), int(m.group(2))))
            else:
                n = int(tok)
                out.update((n, n + 1))
    return out


def _oxo_locants(name: str) -> set:
    out: set = set()
    for block in _OXO_PREFIX_RE.findall(name) + _ONE_SUFFIX_RE.findall(name):
        out.update(int(t) for t in re.findall(r'\d+', block))
    return out


def oxo_ene_valence_illegal(name: str) -> bool:
    """ G5-A source-level (Java-free) valence guard.

    True iff a ring carbon is cited as BOTH a ring double-bond terminus AND an
    oxo/-one carbon -- a five-bond carbon (the caffeine-class
    ``2,4-dioxo-...-1,7-diene`` invalid name). Pure string parse of the emitted
    von-Baeyer locants; no RDKit/OPSIN. Runs unconditionally downstream so
    best-effort is safe even when the OPSIN RT gate is unavailable (no jar).
    Fail-closed on the cumulated-carbon edge (an oxo carbon truly cannot also
    hold a ring double bond).
    """
    if not name or 'ene' not in name or ('oxo' not in name and 'one' not in name):
        return False
    return bool(_ene_termini(name) & _oxo_locants(name))


__all__ = [
    "input_invariants",
    "net_formal_charge",
    "looks_like_ionic_name",
    "_has_main_group_charge_centre",
    "charge_dropped",
    "partial_sat_sp3_substituent_drop",
    "fused_ring_atom_drop",
    "oxo_ene_valence_illegal",
]
