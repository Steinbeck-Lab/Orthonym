"""v25 G1: general substitutive chain namer with atom->token bindings.

NEW code (design: , G1).
Unlike the legacy composer path, every emission carries a TokenBinding
partition over ALL heavy atoms, so the E1 certificate
(validation/e1_certificate.py) can verify no atom was silently dropped --
Java-free. Output is OPT-IN (namer general_fallback flag); it re-enters the
existing moat (>15-HA gate, P10 vetoes, SELF-01 OPSIN-RT) downstream.

G1 scope: neutral, single-fragment, chain-parented molecules with an
all-carbon parent chain of length >= 2 and a suffix from the supported set
(or none). Ring parents -> G2 (general ring fallback). Charged -> G3.
Anything outside scope REFUSES (returns None): fail-closed, never partial.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

from ..errors import is_refusal_sentinel

logger = logging.getLogger(__name__)

# Suffix support keyed by the SUFFIX STRING (robust to fg-name spelling):
# 'terminal' suffixes sit on chain terminus C1 (locants omitted / dioic);
# 'locant' suffixes carry explicit locants.
_SUPPORTED_SUFFIX_STYLES = {
    'oic acid': 'terminal', 'al': 'terminal', 'nitrile': 'terminal',
    'amide': 'terminal',
    'one': 'locant', 'ol': 'locant', 'amine': 'locant', 'thiol': 'locant',
}

# v29 P7 Task 5.  An INLINE suffix ('one', 'ol', 'amine', 'thiol', 'imine')
# adds no skeletal atom: it CONVERTS one parent-hydride atom into the
# characteristic group, so its locant designates THAT atom and no other.
#
#   P-64.2.2.2 "Cyclic ketones" (BlueBookV2/BlueBookV2.md:28384 heading;
#   sentence at :28386) -- "Names of cyclic ketones are formed substitutively
#   by using the suffix 'one'.  As the formation of ketones is achieved by the
#   conversion of a methylene, >CH2, group into a >C=O group, the suffix 'one'
#   with appropriate locants can be added to the name of parent hydrides having
#   such groups."   :28390 -- "Ketones resulting from the substitution of >CH2
#   groups are named substitutively using the suffix 'one' to designate the
#   principal characteristic group."   The section's own example is
#   `bicyclo[3.2.1]octan-2-one (PIN)` (:28396) -- a von Baeyer ketone whose
#   locant is the carbonyl carbon, not a neighbour of it.
#
# The FG SMARTS match is NOT a proxy for that atom.  The shipped ketone pattern
# is `[#6][CX3](=O)[#6]` (perception/functional_groups.py:342), so a match also
# contains BOTH flanking carbons; the previous `min(atom_to_locant[i] for i in
# on_cage)` therefore cited whichever neighbour happened to number lowest.
_INLINE_SUFFIX_CORES = frozenset({'one', 'ol', 'amine', 'thiol', 'imine'})


def _inline_suffix_locant(pg, match, parent_set, atom_to_locant):
    """Locant of the parent atom an inline suffix converts, or ``None``.

    Delegates atom selection to ``rules.parent_selection._pg_attachment_atoms``
    -- the primitive that already owns the "which atom of this SMARTS match
    bears the locant" question, driven by ``seniority.PG_ATTACHMENT_INDICES``
    (``ketone`` -> ``[1]``, the alcohol/thiol/amine family -> the carbon(s),
    ``imine`` -> the SMARTS-leading carbon by default).  ``min()`` over the
    survivors matches that primitive's documented contract for multi-position
    groups such as ``secondary_amine`` -> ``[1, 2]``.

    ``None`` means the characteristic atom is NOT on this parent -- e.g. an
    acetyl carbon hanging off the chosen chain -- and the caller must refuse.
    Citing the attachment atom's locant instead would both misplace the suffix
    and swallow the acyl carbons into ``suffix_atoms``, emitting a name for a
    strictly smaller molecule.
    """
    from ..rules.parent_selection import _pg_attachment_atoms

    anchors = [i for i in _pg_attachment_atoms(pg, tuple(match))
               if i in parent_set]
    if not anchors:
        return None
    return min(atom_to_locant[i] for i in anchors)


# Functional groups that reach the monocycle ring-suffix path
# (``_RING_SUFFIX_STYLES``) with NO entry in ``seniority.PG_ATTACHMENT_INDICES``
# and whose SMARTS have been audited to lead with the locant-bearing atom, so
# ``_pg_attachment_atoms``' index-0 default is correct for them:
#
#   carboxylic_acid        [CX3](=O)[OX2H1]                -> 0 is the acyl C
#   ester                  [CX3](=O)[OX2][#6]              -> 0 is the acyl C
#   primary/secondary/
#     tertiary_amide       [CX3](=O)[NX3...]               -> 0 is the acyl C
#   primary/secondary/
#     tertiary_sulfonamide [SX4](=O)(=O)[NX3...]           -> 0 is the S
#   amidine                [CX3](=[NX2])[NX3;...]          -> 0 is the amidine C
#   nitrile                [CX2]#[NX1]                     -> 0 is the nitrile C
#   aldehyde               [CX3;H1,H2](=O)                 -> 0 is the carbonyl C
#
# This list is NOT decoration.  ``_pg_attachment_atoms`` silently falls back to
# SMARTS index 0 for any unknown FG, and for the ``phenol``/``aromatic_amine``/
# ``enol`` SMARTS index 0 is the HETEROATOM -- a gap that became a live
# regression once before (see the v29 P7 C2 block in seniority.py).  Anything
# that reaches this path undeclared must be audited, not assumed; the invariant
# is enforced by tests/unit/assembly/test_general_monocycle_pg_anchor.py.
_LEADING_ANCHOR_RING_PGS = frozenset({
    "carboxylic_acid", "ester",
    "primary_amide", "secondary_amide", "tertiary_amide",
    "primary_sulfonamide", "secondary_sulfonamide", "tertiary_sulfonamide",
    "amidine", "nitrile", "aldehyde",
})


def _pg_bearing_ring_atoms(mol, pg, match, ring_set):
    """Ring atoms that BEAR the principal characteristic group (P-14.4).

    The raw SMARTS match is not the answer.  ``[#6][CX3](=O)[#6]`` matches a
    ring ketone's carbonyl carbon *and both of its ring neighbours*, so
    intersecting the whole match with the ring hands the numbering comparator
    three atoms for a monoketone and all six for a para-dione -- identical for
    every candidate orientation, which silently neuters the principal-group
    criterion and lets ring unsaturation win instead.  P-64.2.1.2
    (``BlueBookV2.md:28307``) settles the intended outcome at ``:28320``:
    ``1,4-benzoquinone   cyclohexa-2,5-diene-1,4-dione (PIN)`` -- the dione
    takes 1,4, the diene takes 2,5.

    So ask the one question both ring-suffix styles share: *which ring atom
    carries the group?*

    * inline suffix (``one``/``ol``/``amine``/``thiol``/``imine``) -- the
      characteristic atom is itself a ring atom; that is the answer.
    * appended suffix (``carboxylic acid``/``carbaldehyde``/``carbonitrile``/
      ``carboxamide``/``sulfonamide``/...) -- the anchor hangs off the ring, so
      the locant belongs to the ring atom it is attached to.  Intersecting the
      raw match with the ring returns *nothing* for these, which is why an
      unsubstituted ring acid numbered its own attachment carbon 4.

    Atom selection is delegated to ``parent_selection._pg_attachment_atoms``
    (driven by ``seniority.PG_ATTACHMENT_INDICES``) -- the primitive that
    already owns "which atom of this SMARTS match bears the locant".  An FG with
    neither a table entry nor a place on ``_LEADING_ANCHOR_RING_PGS`` is treated
    as undeclared and falls back to the historical whole-match behaviour rather
    than silently trusting SMARTS index 0.
    """
    from ..rules.parent_selection import _pg_attachment_atoms
    from ..rules.seniority import PG_ATTACHMENT_INDICES

    whole_match = [i for i in match if i in ring_set]
    if PG_ATTACHMENT_INDICES.get(pg) is None and pg not in _LEADING_ANCHOR_RING_PGS:
        return whole_match

    anchors = _pg_attachment_atoms(pg, tuple(match))
    on_ring = [i for i in anchors if i in ring_set]
    if on_ring:
        return on_ring

    attached = []
    for a in anchors:
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            j = nb.GetIdx()
            if j in ring_set and j not in attached:
                attached.append(j)
    return attached or whole_match


_MULT_SIMPLE = {2: 'di', 3: 'tri', 4: 'tetra', 5: 'penta', 6: 'hexa',
                7: 'hepta', 8: 'octa', 9: 'nona', 10: 'deca'}
_MULT_COMPLEX = {2: 'bis', 3: 'tris', 4: 'tetrakis', 5: 'pentakis',
                 6: 'hexakis'}
# Complex prefix (needs bis/tris + enclosure): contains locants, hyphens,
# brackets, or an internal multiplying prefix.
_COMPLEX_PREFIX_RE = re.compile(r"[0-9\-\(\)\[\]]")


@dataclass(frozen=True)
class TokenBinding:
    """Atoms expressed by one emitted name token."""
    atom_ids: Tuple[int, ...]
    token: str
    # 'parent' | 'prefix' | 'suffix' | 'replacement'. Every value here must be a
    # key of ``binding_spine._LEGACY_ROLE_KINDS``, or the flat adapter coerces it
    # to PREFIX and records it in ``legacy_role_coerced``.
    role: str


@dataclass(frozen=True)
class GeneralEngineResult:
    name: str
    bindings: Tuple[TokenBinding, ...]


def _refuse(reason: str) -> None:
    logger.info("general_engine refused: %s", reason)
    return None


def _common_refusal(mol, allow_charged: bool = False) -> Optional[str]:
    """Engine-wide scope refusals shared by the chain and ring paths.

    v26 P5: ``allow_charged`` (set only under ``complete`` /
    ``allow_aromatic_general``) lifts the net-charge refusal so the charged
    general path can emit a ``-ylium``/``-ide``/``-uide``/``-ium`` suffix on the
    numbered parent (``_charge_suffix_text`` + fail-closed). The multi-fragment,
    radical (radical-cation) and isotope refusals STAY even when charge is
    allowed -- those remain out of scope for ``complete``.
    """
    if mol is None:
        return "no mol"
    if not allow_charged and Chem.GetFormalCharge(mol) != 0:
        return "net charge (G3 scope)"
    if len(Chem.GetMolFrags(mol)) > 1:
        return "multi-fragment (G3 scope)"
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return "radical"
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        return "isotope"
    return None


def _charge_suffix_text(mol, atom_to_locant) -> Optional[str]:
    """v26 P5 (BB P-73 cations / P-74 anions): the charge-suffix string for
    skeletal charge(s) on the ALREADY-NUMBERED general parent --
    ``-1-ium`` / ``-2-ylium`` / ``-1-ide`` / ``-1-uide`` (or the multiplied
    ``-1,4-diium`` / ``-1,2-diylium`` / ``-1,4-diide`` forms).

    The locant is the ACTUAL charged atom's parent locant, so the emitted
    descriptor is structurally faithful on its own (SELF-01 is only the
    backstop, which fails OPEN without Java). Reuses the proven ion perception
    (``get_ion_sites``) + classifiers (``classify_cation`` / ``classify_anion``)
    rather than reinventing charge typing.

    FAIL CLOSED (return None -> the caller abstains, never a wrong/neutral name)
    on any charge that cannot be faithfully expressed as a suffix on THIS parent:
      * a charge on a SUBSTITUENT atom (not in the parent numbering);
      * an FG-anchored anion (alkoxide/thiolate/carboxylate/sulfonate/...) whose
        charge sits on an off-parent oxygen -- the PIN charged path owns those;
      * diazonium / acylium cations -- the PIN path owns those;
      * a multiply-charged single atom, mixed sign centres, mixed suffix kinds,
        an internal-only (P-59 nitro/azide/N-oxide/diazo) charge, or a
        multiplicity beyond the simple table.
    """
    from ..perception.ions import get_ion_sites
    from ..rules.ions import classify_cation, classify_anion

    sites = get_ion_sites(mol)  # excludes internal P-59 charges
    anions = list(sites.get('anions') or [])
    cations = list(sites.get('cations') or [])
    if bool(anions) == bool(cations):
        # neither (internal-only net charge) or BOTH (mixed-sign) -> out of scope
        return None
    charged = anions or cations
    negative = bool(anions)
    parent_atoms = set(atom_to_locant)

    per_locant_base: List[Tuple[int, str]] = []
    for site in charged:
        idx = site['atom_idx']
        if idx not in parent_atoms:
            return None  # charge on a substituent -> not a parent suffix
        if abs(int(site.get('charge', 0))) != 1:
            return None  # multiply-charged single atom -> tight scope
        if negative:
            acls = classify_anion(mol, site)
            if acls in ('carbanion', 'heteroatom_hydride_anion'):
                base = 'ide'          # P-72.2.2.1: loss of H+ from a skeletal atom
            elif acls == 'uide_anion':
                base = 'uide'         # P-72.3: hydride ADDED to a skeletal atom
            else:
                return None           # FG anion -> PIN path owns it
        else:
            ccls = classify_cation(mol, site)
            if ccls == 'ylium':
                base = 'ylium'        # P-73.2.2.1.1: loss of H- from a skeletal C
            elif ccls in ('aminium', 'onium', 'quaternary'):
                base = 'ium'          # P-73.1: protonated / substituted skeletal heteroatom
            else:
                return None           # diazonium / acylium -> PIN path owns it
        per_locant_base.append((atom_to_locant[idx], base))

    bases = {b for _, b in per_locant_base}
    if len(bases) != 1:
        return None                   # mixed suffix kinds on one parent
    base = next(iter(bases))
    locants = sorted(loc for loc, _ in per_locant_base)
    n = len(locants)
    if n == 1:
        mult = ''
    else:
        mult = _MULT_SIMPLE.get(n)
        if mult is None:
            return None
    return '-' + ','.join(map(str, locants)) + '-' + mult + base


def _append_charge_suffix(name: str, mol, atom_to_locant,
                          has_fg_suffix: bool) -> Optional[str]:
    """Splice the P5 charge suffix onto an assembled parent ``name`` (pre-stereo).

    Elides a single trailing parent 'e' only when the charge-suffix text
    begins with a vowel (``cyclohexane``->``cyclohexan-1-ide``,
    ``1-methylpyridine``->``1-methylpyridin-1-ium``, ``...pentaene``->
    ``...pentaen-4-ium``): the single-charge bases (``-ium``/``-ylium``/
    ``-ide``/``-uide``) start with i/y/u. The MULTIPLIED forms
    (``-1,4-diium``/``-1,4-diide``/...) begin with the consonant of the
    multiplier (di/tri/...), so the terminal 'e' must be RETAINED
    (P-16.7.1(a) elides only before a vowel or 'y'), e.g.
    ``1,4-diazine``->``1,4-diazine-1,4-diium`` (NOT
    ``diazin-1,4-diium``). Returns the charged name, or None to FAIL CLOSED
    (a charge that is not expressible, or a co-occurring FG suffix -- the
    cumulative FG+charge construction is out of P5 scope)."""
    if has_fg_suffix:
        return None  # FG suffix + skeletal charge (cumulative) -> out of P5 scope
    cs = _charge_suffix_text(mol, atom_to_locant)
    if cs is None:
        return None
    return _elide_before_ionic_suffix(name, cs)


def _elide_before_ionic_suffix(name: str, cs: str) -> str:
    """P-16.7.1(a) / P-74.1.1 (:42417) parent-'e' elision before an ionic suffix.

    The governing rule is **P-16.7 "ELISION OF VOWELS"** (:7591), clause
    **P-16.7.1(a)** (:7595): *"the terminal letter 'e' in names of parent
    hydrides or endings 'ene' and 'yne' when followed by a suffix or 'en' ending
    beginning with 'a', 'e', 'i', 'o', 'u', or 'y'"*.  **P-74.1.1 "Ionic centers
    in the same parent structure"** (:42417) restates it for ionic suffixes
    specifically: *"The final letter 'e' of the name of a parent hydride, or of
    an 'ide' or 'uide' suffix, is elided before the letter 'i' or 'y', or before
    a cumulative suffix beginning with a vowel."*  (NOT P-16.3.3, which is
    "The basic numerical prefixes 'di', 'tri', 'tetra', etc." -- multiplication,
    not elision.)  The single-charge bases
    (``ium``/``ylium``/``ide``/``uide``/``olate``) start with a vowel or 'y' ->
    elide (``...pentaene`` -> ``...pentaen-6-ium``). The MULTIPLIED forms
    (``-1,4-diium``) begin with the multiplier's consonant -> the 'e' is
    RETAINED (``1,4-diazine-1,4-diium``). Extracted from ``_append_charge_suffix``
    so the zwitterion (P-74.1.1) path elides by the SAME rule, not a copy."""
    first_alpha = next((c for c in cs if c.isalpha()), '')
    stem = name[:-1] if name.endswith('e') and first_alpha in 'aeiouy' else name
    return stem + cs


def _genuine_ion_sites(mol):
    """The molecule's genuine ionic centres, as ``(cations, anions)``.

    This used to subtract semipolar ``[X+]-[O-]`` oxide pairs itself, from a
    hand-written list of cation elements, because ``get_ion_sites`` reported
    such a pair as a genuine ionic centre and that suppressed the correct name
    ``4-methyl-2-oxo-1,2-oxaphospholane`` for ``CC1CO[PH+](C1)[O-]``.

    That subtraction has moved to its root cause,
    ``perception/ions.py::_semipolar_chalcogenide_atoms`` (P-74.2.1.2 /
    P-74.2.1.4), so ``get_ion_sites`` no longer reports the pair at all and
    every other caller is fixed too, not just this one.

    The local version is not merely redundant, it was WRONG, and it is worth
    recording why so it is not reintroduced. Keying on a cation element list
    cannot distinguish a semipolar oxide from a genuine oxoanion: measured over
    the three corpora plus 307 synthetic probes, the old code still fired on 14
    molecules after the perception fix landed, and the two of those that are
    real corpus rows -- ``[O-][I+]([O-])(O)(O)(O)O`` and
    ``[O-][I+]([O-])([O-])(O)(O)O`` -- are genuine iodine oxoanions whose
    charges it wrongly erased. The replacement proves the bonding pattern
    instead: the positive charge must be balanced locally by its terminal
    chalcogenide anions, and the uncharged multiple-bond depiction must both
    sanitise and carry the same standard InChIKey.
    """
    from ..perception.ions import get_ion_sites

    sites = get_ion_sites(mol)
    return (list(sites.get('cations') or []), list(sites.get('anions') or []))


def _has_ionic_centres(mol) -> bool:
    """True when the molecule carries a genuine (non-semipolar) ionic centre.

    Deliberately reuses the SAME perception the suffix builders use, so the
    fail-closed guard and the emitters can never disagree."""
    cations, anions = _genuine_ion_sites(mol)
    return bool(cations) or bool(anions)


# P-73.1 / P-73.2.2.1.1 cationic suffix bases, keyed by ``classify_cation``.
_ZWIT_CATION_BASES = {
    'aminium': 'ium', 'onium': 'ium', 'quaternary': 'ium', 'ylium': 'ylium',
}
# P-72.2.2.1 / P-72.3 SKELETAL anionic suffix bases, keyed by ``classify_anion``.
# This is the P-74.1.1 branch (BOTH ionic centres skeletal to the parent hydride)
# and it is REACHED -- not a lookup table nobody visits (the contributor guide #10). Entry
# point: ``name_general_ring``/``name_general_spiro`` -> ``_emit_ring_from_analysis``.
# Five cage zwitterions exercise it, every emitted name OPSIN-round-tripped to the
# input InChIKey; they are the parametrised cases in
# ``tests/unit/assembly/test_general_engine_zwitterion.py::SKELETAL_ZWITTERIONS``.
# NB ``classify_anion`` returns 'aminide' for a ring N(-), which is deliberately
# NOT in this table -- the Blue Book's own aminide zwitterion (:42460,
# ``N,1,4-triphenyl-1H-1,2,4-triazol-4-ium-3-aminide``) needs an N-substituted
# aminide the cage producer cannot build, so it fails closed instead.
_ZWIT_SKELETAL_ANION_BASES = {
    'carbanion': 'ide', 'heteroatom_hydride_anion': 'ide', 'uide_anion': 'uide',
}
# P-72.2.2 ``-ol`` -> ``-olate``: an anionic oxygen hanging off ONE parent atom.
_ZWIT_OLATE_ANION_CLASSES = frozenset({'alkoxide', 'phenolate'})


def _zwitterion_suffix_plan(mol, atom_to_locant, allow_fg_anion: bool = True):
    """Cumulative ionic suffixes for a NET-NEUTRAL zwitterion whose ionic
    centres lie in THIS numbered parent.

    **The governing case is P-74.1.2** "Zwitterionic compounds with at least one
    ionic center on a characteristic group" (heading :42445).  Its sentence
    (:42447): *"Zwitterionic compounds with at least one ionic center on a
    characteristic group may be named by adding the appropriate ionic suffix to
    the name of the ionic parent hydride.  In names, cationic suffixes are cited
    before anionic suffixes."*  Worked (PIN) example (:42456)
    ``1-methyl-4,6-diphenylpyridin-1-ium-2-carboxylate`` — a SKELETAL ring
    ``-ium`` plus a characteristic-group-derived anionic suffix, which is exactly
    the shape of the R7 target's ``-6-ium-2-olate`` (the ``-olate`` branch below).

    **P-74.1.1** "Ionic centers in the same parent structure" (heading :42415)
    governs the other branch — both centres skeletal to the parent hydride — and
    supplies the construction and elision used by BOTH branches.  :42419: *"For
    nomenclature purposes, zwitterionic compounds having the ionic centers in the
    same parent structure are not considered as neutral compounds."*  :42417:
    *"…may be named by combining appropriate cumulative suffixes at the end of
    the name of a parent hydride in the order 'ium', 'ylium', 'ide', 'uide'.  …
    In either case anionic suffixes are cited after cationic suffixes in the
    name…  The final letter 'e' … is elided before the letter 'i' or 'y'…"*

    This is the ZWITTERION widening of the v26-P5 charge-suffix layer. That
    layer (``_charge_suffix_text``) is gated on NET molecular charge and
    explicitly declines the mixed-sign case, so a zwitterion (net 0) reached
    neither the suffix nor the refusal and shipped a NEUTRAL name -- the error
    the Blue Book itself calls out at :42439,
    ``2-methyl-4-oxo-3,4-dihydro-1H-2-benzoselenopyran-2-ium-3-ide (PIN)``
    *(not ``…-2-ium-3-id-4-one``)*.

    LOCANTS.  P-74.1.2 (:42447) closes with *"For assignment of lower locants,
    ionic centers on skeletal atoms of the parent hydride are preferred to the
    locants for positions of attachment of characteristic groups denoted by
    ionic suffixes."*  This function does NOT re-number: it consumes the
    ``atom_to_locant`` the ring/chain engine already fixed.  That is sound
    because the ionic criterion never gets a choice here — for a heterocyclic
    von Baeyer parent, P-23.3.1 (:9765) fixes the numbering from the hydrocarbon
    system and P-23.3.2.1 (:9777) then assigns *"low locants … to the
    heteroatoms considered together as a set"*, which is decided strictly BEFORE
    any ionic-suffix locant.  For the R7 target the four numberings the
    bicyclo[4.4.0] descriptor permits give heteroatom sets {1,2,4}, {1,8,10},
    {3,5,6} and {6,7,9} — all distinct, so P-23.3.2.1 decides alone and the
    P-74.1.2 locant sentence is satisfied VACUOUSLY (zero remaining freedom).
    See ``test_p74_1_2_ionic_locant_rule_has_no_freedom_here`` and the xfail
    ``test_von_baeyer_heteroatom_locants_should_be_lowest_set`` for the separate,
    PRE-EXISTING P-23.3.2.1 defect that the cage engine picks {3,5,6}.

    Returns ``(held_out_atoms, suffix_text)`` where ``held_out_atoms`` must be
    withheld from substituent discovery so the anion is not ALSO spelled as a
    neutral prefix (the ``2-oxo`` double-count), or ``None`` to FAIL CLOSED.

    Scope (tight, mirroring ``charged_router._route_zwitterion``'s D-06 scope):
    exactly one cationic and one anionic centre, each singly charged, the cation
    skeletal to this parent. The anion is either skeletal (``-ide``/``-uide``) or
    a single-bonded, H-free oxygen on a parent atom (``-olate``). Anything else
    -- an off-parent cation, a multiply-charged atom, a carboxylate/sulfonate FG
    anion (the PIN charged path owns those), a net-charged species -- declines.
    """
    from ..rules.ions import classify_cation, classify_anion

    if mol is None or Chem.GetFormalCharge(mol) != 0:
        return None  # net-charged -> the existing _charge_suffix_text path owns it

    # Semipolar [X+]-[O-] oxide pairs are internal charges, not ionic centres.
    cations, anions = _genuine_ion_sites(mol)
    if len(cations) != 1 or len(anions) != 1:
        return None
    cation, anion = cations[0], anions[0]
    if abs(int(cation.get('charge', 0))) != 1:
        return None
    if abs(int(anion.get('charge', 0))) != 1:
        return None

    # --- cationic centre: must be a numbered atom of THIS parent (P-74.1.1) ---
    cation_idx = cation['atom_idx']
    if cation_idx not in atom_to_locant:
        return None
    cation_base = _ZWIT_CATION_BASES.get(classify_cation(mol, cation))
    if cation_base is None:
        return None  # diazonium / acylium -> the PIN charged path owns those
    cation_locant = atom_to_locant[cation_idx]

    # --- anionic centre ---
    anion_idx = anion['atom_idx']
    held: set = set()
    if anion_idx in atom_to_locant:
        anion_base = _ZWIT_SKELETAL_ANION_BASES.get(classify_anion(mol, anion))
        if anion_base is None:
            return None
        anion_locant = atom_to_locant[anion_idx]
    else:
        if not allow_fg_anion:
            return None  # caller cannot hold the atom out -> would double-count
        atom = mol.GetAtomWithIdx(anion_idx)
        if (atom.GetSymbol() != 'O' or atom.GetDegree() != 1
                or atom.GetTotalNumHs()):
            return None
        if classify_anion(mol, anion) not in _ZWIT_OLATE_ANION_CLASSES:
            return None  # carboxylate/sulfonate/... -> PIN charged path
        neighbour = atom.GetNeighbors()[0]
        if neighbour.GetIdx() not in atom_to_locant:
            return None
        anion_base = 'olate'
        anion_locant = atom_to_locant[neighbour.GetIdx()]
        held.add(anion_idx)

    # P-74.1.2 (:42447): cationic suffix first, then the anionic suffix.
    text = '-%s-%s-%s-%s' % (cation_locant, cation_base,
                            anion_locant, anion_base)
    return frozenset(held), text


def name_general_chain(
    mol, features, allow_charged: bool = False, allow_mancude: bool = False,
) -> Optional[GeneralEngineResult]:
    """Name a chain-parented molecule with a full atom->token partition.

    Returns None on ANY condition outside the verified G1 scope.

    v26 P5: ``allow_charged`` (only under ``complete``) lifts the net-charge
    refusal and emits a ``-ide``/``-ylium``/``-ium``/``-uide`` suffix on the
    numbered chain parent when the charge sits on a chain skeletal atom
    (fail-closed otherwise).

    v27 P1: ``allow_mancude`` (complete/best-effort tier only) lets a multi-ring
    cage SUBSTITUENT on the chain be named via the universal von-Baeyer engine
    (parent<->substituent symmetry). Default False -> PIN path byte-identical.
    """
    reason = _common_refusal(mol, allow_charged=allow_charged)
    if reason:
        return _refuse(reason)

    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    chain = list(getattr(features, 'principal_chain', None) or ())
    if ring_atoms and not getattr(features, 'chain_is_parent', False):
        return _refuse("ring parent (ring path owns it)")
    if len(chain) < 2:
        return _refuse("chain too short")
    if any(mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in chain):
        return _refuse("hetero parent chain (replacement nomenclature)")

    part = _partition(mol, features, chain)
    if part is None:
        return None
    return _assemble(mol, features, chain, part, allow_charged=allow_charged,
                     allow_mancude=allow_mancude)


def _partition(mol, features, chain) -> Optional[dict]:
    """Split heavy atoms into chain / suffix / substituent fragments.

    Suffix atoms (PG-match atoms off the chain) are folded into the
    blocked set BEFORE substituent discovery so a suffix oxygen is never
    double-expressed as a 'hydroxy'/'oxo' prefix.
    """
    from .substituent_enumerator import discover_substituents
    from ..rules.seniority import get_suffix

    chain_set = set(chain)
    pg = getattr(features, 'principal_group', None)
    pg_matches = list(getattr(features, 'principal_group_atoms', None) or [])

    atom_to_locant = dict(getattr(features, 'atom_to_locant', None)
                          or {a: i + 1 for i, a in enumerate(chain)})
    if set(chain) - set(atom_to_locant):
        return _refuse("chain atom missing from atom_to_locant")

    suffix_core = None
    suffix_atoms: set = set()
    pg_chain_locants: List[int] = []
    if pg:
        suffix_core = get_suffix(pg, is_ring=False)
        if suffix_core not in _SUPPORTED_SUFFIX_STYLES:
            return _refuse(f"unsupported suffix for pg={pg!r}")
        seen = set()
        for match in pg_matches:
            key = tuple(sorted(match))
            if key in seen:
                continue
            seen.add(key)
            on_chain = [i for i in match if i in chain_set]
            if not on_chain:
                return _refuse("PG instance not on parent chain")
            if suffix_core in _INLINE_SUFFIX_CORES:
                # P-64.2.2.2 (:28386): an inline suffix CONVERTS a parent atom,
                # so its locant is that atom's -- never a flanking atom of the
                # SMARTS match, and never an attachment atom standing in for a
                # characteristic carbon that is off the chain.
                loc = _inline_suffix_locant(pg, match, chain_set,
                                            atom_to_locant)
                if loc is None:
                    return _refuse(
                        "inline suffix characteristic atom off parent chain")
            else:
                loc = min(atom_to_locant[i] for i in on_chain)
            suffix_atoms.update(i for i in match
                                if i not in chain_set
                                and mol.GetAtomWithIdx(i).GetAtomicNum() > 1)
            pg_chain_locants.append(loc)

    try:
        subs = discover_substituents(
            mol, chain_set | suffix_atoms, parent_type='chain',
            principal_chain=chain, atom_to_locant=atom_to_locant,
            general_fallback=True)
    except AssertionError as e:
        return _refuse(f"partition incomplete: {e}")
    if subs is None:
        # v28 Composer1 Task 5: substituent off a suffix/FG atom the chain walk
        # cannot reach (e.g. the N-aryl ring of an amide anilide). Fail closed —
        # never drop it (wrong constitution).
        return _refuse("partition incomplete: unassigned atoms off suffix/FG")

    return {
        'suffix_core': suffix_core,
        'suffix_atoms': frozenset(suffix_atoms),
        'pg_locants': sorted(pg_chain_locants),
        'substituents': subs,
        'atom_to_locant': atom_to_locant,
    }


def _alpha_key(prefix: str) -> str:
    """Alphabetization key: letters only; sec-/tert- excluded, iso/neo/cyclo
    included (IUPAC P-14.5.2).

    v29: the italicized-prefix strip is the shared primitive, not a re-derived
    copy of the literal tuple (this loop form is one of the two the v29 P3
    tripwire's `startswith((...))` regex could not see).
    """
    from .naming_utils import strip_italicized_structural_prefix
    remainder, _ = strip_italicized_structural_prefix(prefix)
    return re.sub(r"[^a-z]", "", remainder.lower())


def _is_complex_prefix(name: str) -> bool:
    """Does ``name`` need bis/tris + enclosure (as opposed to di/tri, bare)?

    ``_COMPLEX_PREFIX_RE`` includes a hyphen in its character class, which made
    this a copy of the compound predicate carrying no P-16.3.4 carve-out — and
    because ONE regex drives both the enclosure choice and the di-vs-bis choice
    here, a 'tert-butyl' got BOTH wrong at once ('bis(tert-butyl)' where the Blue
    Book writes 'di-tert-butyl'). The carve-out is the shared primitive.
    """
    from .naming_utils import italicized_prefix_is_bare
    if italicized_prefix_is_bare(name):
        return False
    return bool(_COMPLEX_PREFIX_RE.search(name))


def _mult_prefix(n: int, name: str) -> Optional[str]:
    """'2,2-' + this -> 'dimethyl' / 'bis(2-chloroethyl)'. None if n too big."""
    if n == 1:
        return f"({name})" if _is_complex_prefix(name) else name
    table = _MULT_COMPLEX if _is_complex_prefix(name) else _MULT_SIMPLE
    if n not in table:
        return None
    if table is _MULT_COMPLEX:
        return f"{table[n]}({name})"
    # P-16.3.3(b)/P-16.2.4.1(d) second leg: 'di-tert-butyl', never 'ditert-butyl'.
    from .naming_utils import multiplier_needs_hyphen
    return f"{table[n]}{'-' if multiplier_needs_hyphen(name) else ''}{name}"


def _stereo_prefix(mol, atom_to_locant) -> str:
    """Parent-scope stereodescriptor block from STRUCTURE (rdCIPLabeler via
    collect_stereodescriptors' idempotent guard). '' when achiral."""
    from ..rules.stereochemistry import (
        collect_stereodescriptors, format_stereodescriptor_string,
    )
    return format_stereodescriptor_string(
        collect_stereodescriptors(mol, atom_to_locant))


def _stem_block(mol, chain, atom_to_locant) -> Optional[Tuple[str, str]]:
    """(parent_token, hydride_block) e.g. ('but', 'but-2-ene') or
    ('hex', 'hexane'). None on an unsupported bond pattern."""
    from ..data.chain_names import get_chain_prefix

    base = get_chain_prefix(len(chain))
    if not base:
        return None
    ene, yne = [], []
    pos = {a: atom_to_locant[a] for a in chain}
    for i in range(len(chain) - 1):
        bond = mol.GetBondBetweenAtoms(chain[i], chain[i + 1])
        if bond is None:
            return None
        order = bond.GetBondTypeAsDouble()
        loc = min(pos[chain[i]], pos[chain[i + 1]])
        if order == 2.0:
            ene.append(loc)
        elif order == 3.0:
            yne.append(loc)
        elif order != 1.0:
            return None  # aromatic/dative chain bond: out of scope
    ene.sort()
    yne.sort()
    if not ene and not yne:
        return base, base + "ane"
    block = base
    if ene:
        mult = _MULT_SIMPLE.get(len(ene), '') if len(ene) > 1 else ''
        if len(ene) > 1:
            block += 'a'
        block += '-' + ','.join(map(str, ene)) + '-' + mult + 'ene'
    if yne:
        if ene:
            block = block[:-1]  # 'ene' -> 'en' before '-N-yne'
        mult = _MULT_SIMPLE.get(len(yne), '') if len(yne) > 1 else ''
        if len(yne) > 1 and not ene:
            block += 'a'
        block += '-' + ','.join(map(str, yne)) + '-' + mult + 'yne'
    return base, block


def _suffix_block(style: str, core: str, locants: List[int]) -> Optional[str]:
    """Suffix text WITHOUT the elision decision: '-1-ol', '-1,2-diol',
    'oic acid', 'dial'. None if unsupported multiplicity/placement."""
    n = len(locants)
    if n == 0:
        return None
    if style == 'terminal':
        if n == 1:
            if locants != [1]:
                return None  # terminal suffix must sit on C1
            return core
        if n == 2:
            return 'di' + core  # e.g. 'dioic acid', 'dial' (locants implicit)
        return None
    mult = '' if n == 1 else _MULT_SIMPLE.get(n)
    if n > 1 and mult is None:
        return None
    return '-' + ','.join(map(str, sorted(locants))) + '-' + (mult or '') + core


def _assemble(mol, features, chain, part,
              allow_charged: bool = False,
              allow_mancude: bool = False) -> Optional[GeneralEngineResult]:
    from .substituent_enumerator import name_substituent

    atom_to_locant = part['atom_to_locant']

    stem = _stem_block(mol, chain, atom_to_locant)
    if stem is None:
        return _refuse("unsupported chain bond pattern")
    parent_token, hydride = stem

    # --- substituent prefixes (recursion reuse) ---
    groups: Dict[str, List[int]] = {}
    frag_bindings: List[TokenBinding] = []
    for sub in part['substituents']:
        frag = set(sub.frag_atoms)
        attach_nbrs = [n.GetIdx() for n in
                       mol.GetAtomWithIdx(sub.attach_mol_idx).GetNeighbors()
                       if n.GetIdx() in frag]
        if not attach_nbrs:
            return _refuse("substituent without chain attachment")
        # v27 P1: complete-tier cage substituent recursion (see name_general_ring).
        prefix = name_substituent(mol, frag, attach_nbrs[0],
                                  allow_mancude=allow_mancude)
        if is_refusal_sentinel(prefix):
            return _refuse("branch unnameable (tier-5 fallback)")
        groups.setdefault(prefix, []).append(sub.locant)
        frag_bindings.append(TokenBinding(tuple(sorted(frag)), prefix,
                                          'prefix'))

    prefix_parts = []
    for prefix in sorted(groups, key=_alpha_key):
        locs = sorted(groups[prefix])
        text = _mult_prefix(len(locs), prefix)
        if text is None:
            return _refuse("multiplicity beyond table")
        prefix_parts.append(','.join(map(str, locs)) + '-' + text)

    # --- suffix ---
    bindings = frag_bindings
    suffix_text = ''
    if part['suffix_core']:
        style = _SUPPORTED_SUFFIX_STYLES[part['suffix_core']]
        suffix_text = _suffix_block(style, part['suffix_core'],
                                    part['pg_locants'])
        if suffix_text is None:
            return _refuse("unsupported suffix placement/multiplicity")
        if part['suffix_atoms']:
            bindings.append(TokenBinding(tuple(sorted(part['suffix_atoms'])),
                                         part['suffix_core'], 'suffix'))

    # --- elision: 'ane' + vowel-initial suffix -> 'an' + suffix ---
    body = hydride
    if suffix_text:
        first_alpha = next((c for c in suffix_text if c.isalpha()), '')
        if body.endswith('e') and first_alpha in 'aeiouy':
            body = body[:-1]
        body += suffix_text

    # Prefix text abuts the stem directly ('3-ethyl-2,2-dimethylhexane').
    name = ('-'.join(prefix_parts) + body) if prefix_parts else body
    # v26 P5: charge suffix on a chain skeletal atom (fail closed otherwise).
    if allow_charged and Chem.GetFormalCharge(mol) != 0:
        name = _append_charge_suffix(
            name, mol, {a: atom_to_locant[a] for a in chain},
            has_fg_suffix=bool(part['suffix_core']))
        if name is None:
            return _refuse("charge not expressible as a chain-parent suffix")
    elif _has_ionic_centres(mol):
        # P-74.1.1 (:42419): a zwitterion is "not considered as a neutral
        # compound". The chain producer holds no atom out of discovery, so it
        # cannot build the cumulative suffix without double-counting an FG
        # anion -> fail closed rather than ship a neutral (wrong) structure.
        return _refuse("ionic centres not expressible as a chain-parent suffix")
    # v25 G4: parent-scope stereo from structure (substituent-internal
    # stereo is already handled inside name_substituent's stereo route).
    name = _stereo_prefix(
        mol, {a: atom_to_locant[a] for a in chain}) + name
    bindings.append(TokenBinding(tuple(chain), parent_token, 'parent'))
    return GeneralEngineResult(name=name, bindings=tuple(bindings))


# Ring suffix forms (get_suffix(pg, is_ring=True) values) -- all carry locants
# on a ring parent; 'appended' matches _build_parent_with_unsaturation types.
#
# v27 P2: widened for the high-enrichment linker/suffix groups that previously
# forced the ring engine to abstain. All are gated on the engine tier
# (allow_aromatic_general) via name_general_ring/_monocycle, so the PIN default
# is byte-identical; each emission is SELF-01-verified downstream.
#   * 'sulfonamide' / 'carboximidamide' (amidine) attach directly to the ring
#     carbon (like -carboxamide) -> 'appended' (P-65.3.1 / P-66.4.1).
#   * 'imine' is the aza-'-one' (P-66.3) -> 'inline'; it shares the ketone
#     valence guard (see the suffix_core in ('one','imine') check below).
_RING_SUFFIX_STYLES = {
    'ol': 'inline', 'one': 'inline', 'amine': 'inline', 'thiol': 'inline',
    'imine': 'inline',
    'carboxylic acid': 'appended', 'carbaldehyde': 'appended',
    'carbonitrile': 'appended', 'carboxamide': 'appended',
    'sulfonamide': 'appended', 'carboximidamide': 'appended',
    # v27 P2 (P-65.6.3.2.1): ester ring PIN is the functional-class TWO-WORD
    # `<R-yl> <ring>carboxylate`. The parent block is built with the 'appended'
    # carboxylate suffix on the acid core; the alcoholic `R-yl ` word is
    # prepended by name_general_ring (see _extract_ring_ester). Only the clean
    # acyclic mono-ester is built here; lactones / aryl / poly-esters fail closed.
    'carboxylate': 'appended',
}


def _detect_ring_lactone(mol, pg_matches, ring_set):
    """A LACTONE (cyclic ester) on THIS ring, or ``None``.

    v30 RISK 5 Class 1. Returns ``(carbonyl_c, exo_o)`` -- the ring carbonyl carbon
    and its exocyclic double-bonded O -- when an ``ester`` match's carbonyl carbon
    AND its single-bonded (ester) O are BOTH in ``ring_set`` (the ester is
    ring-internal). The double-bonded O must be exocyclic. Fail-closed (``None``)
    for an acyclic/aryl ester (ester O off-ring), a thioester, or a malformed match
    -- those keep the existing ester/carboxylate handling. Mirrors the carbonyl-C /
    ester-O identification in ``_extract_ring_ester`` (:876-893), applied per match.
    """
    for match in pg_matches:
        carbonyl_c = dbl_o = single_o = None
        for i in match:
            a = mol.GetAtomWithIdx(i)
            if a.GetSymbol() != 'C':
                continue
            _d = _s = None
            for b in a.GetBonds():
                o = b.GetOtherAtom(a)
                if o.GetSymbol() != 'O':
                    continue
                if b.GetBondType() == Chem.BondType.DOUBLE:
                    _d = o.GetIdx()
                elif b.GetBondType() == Chem.BondType.SINGLE:
                    _s = o.GetIdx()
            if _d is not None and _s is not None:
                carbonyl_c, dbl_o, single_o = i, _d, _s
                break
        if carbonyl_c is None:
            continue
        # lactone <=> the carbonyl C and the ESTER (single-bond) O are both in-ring,
        # and the carbonyl (=O) O is exocyclic.
        if (carbonyl_c in ring_set and single_o in ring_set
                and dbl_o not in ring_set):
            return (carbonyl_c, dbl_o)
    return None


def _extract_ring_ester(mol, pg_matches, ring_set, allow_mancude):
    """v27 P2 (P-65.6.3.2.1): decompose a ring carboxylic-acid ESTER for the
    functional-class two-word PIN ``<R-yl> <ring>carboxylate``.

    Returns ``(r_word, r_frag_atoms, acid_core_atoms, ring_attach_atom)`` for
    the single clean RING-ACID mono-ester with an ACYCLIC alcohol, or ``None``
    (fail closed) for any of: more than one ester (polyester), a reverse/aryl
    ester (the ring is on the alcohol side, so the acid C is not bonded to the
    ring), a lactone / ring-bearing R, or an R the substituent namer declines.
    Never guesses the acid side.
    """
    from .substituent_enumerator import name_substituent

    seen: set = set()
    matches = []
    for m in pg_matches:
        k = tuple(sorted(m))
        if k in seen:
            continue
        seen.add(k)
        matches.append(set(m))
    if len(matches) != 1:
        return None  # mono-ester only; polyester fails closed (Phase 4+)
    match = matches[0]

    # Identify the acid core: a match C bearing one =O and one single-bond O.
    carbonyl_c = carbonyl_o = ester_o = None
    for i in match:
        a = mol.GetAtomWithIdx(i)
        if a.GetSymbol() != 'C':
            continue
        dbl_o = single_o = None
        for b in a.GetBonds():
            o = b.GetOtherAtom(a)
            if o.GetSymbol() != 'O':
                continue
            if b.GetBondType() == Chem.BondType.DOUBLE:
                dbl_o = o.GetIdx()
            elif b.GetBondType() == Chem.BondType.SINGLE:
                single_o = o.GetIdx()
        if dbl_o is not None and single_o is not None:
            carbonyl_c, carbonyl_o, ester_o = i, dbl_o, single_o
            break
    if carbonyl_c is None:
        return None

    # The acid carbon must attach to the ring (else the ring is the alcohol side
    # -> reverse/aryl ester, deferred). Exactly one ring neighbour.
    ring_nbrs = [n.GetIdx()
                 for n in mol.GetAtomWithIdx(carbonyl_c).GetNeighbors()
                 if n.GetIdx() in ring_set]
    if len(ring_nbrs) != 1:
        return None
    ring_attach = ring_nbrs[0]

    # R (alcoholic component): the ester-O neighbour that is NOT the acid C.
    r_starts = [n.GetIdx()
                for n in mol.GetAtomWithIdx(ester_o).GetNeighbors()
                if n.GetIdx() != carbonyl_c]
    if len(r_starts) != 1:
        return None
    r_start = r_starts[0]
    if mol.GetAtomWithIdx(r_start).IsInRing():
        return None  # lactone / aryl ester -> fail closed (acyclic R only)

    core = {carbonyl_c, carbonyl_o, ester_o}
    r_frag: set = set()
    stack = [r_start]
    while stack:
        x = stack.pop()
        if x in r_frag or x in core:
            continue
        r_frag.add(x)
        for n in mol.GetAtomWithIdx(x).GetNeighbors():
            if n.GetIdx() not in r_frag and n.GetIdx() not in core:
                stack.append(n.GetIdx())
    if not r_frag or (r_frag & ring_set):
        return None

    r_word = name_substituent(mol, r_frag, r_start, allow_mancude=allow_mancude)
    if is_refusal_sentinel(r_word):
        return None
    return r_word, r_frag, core, ring_attach


def _ring_amide_n_prefixes(mol, pg_matches, allow_mancude):
    """v27 P2 (P-66.1.1.3.1 / P-65.3.1): collect and format N-substituents on a
    ring carboxamide / sulfonamide.

    The amide/sulfonamide nitrogen's non-H neighbours (other than the C=O / S
    anchor) are N-substituents -- named by ``name_substituent`` and cited with
    the italic ``N``/``N,N`` locant. Returns
    ``(n_sub_atoms, prefix_entries, bindings)`` where ``n_sub_atoms`` is the set
    of N-substituent fragment atoms to hold out of the ring-substituent set and
    the appended-suffix atoms, and ``prefix_entries`` is a list of
    ``(alpha_key, formatted_token)`` merged into the ring-substituent ordering by
    the caller. Returns ``None`` (fail closed) if any N-substituent is
    unnameable -- never drop it (that would ship a different constitution).
    """
    from .substituent_enumerator import name_substituent
    from .naming_utils import (
        _wrap_n_substituent, SIMPLE_MULTIPLIERS, is_complex_substituent,
    )

    n_sub_atoms: set = set()
    name_counts: Dict[str, int] = {}
    bindings: List[TokenBinding] = []
    seen: set = set()
    for match in pg_matches:
        key = tuple(sorted(match))
        if key in seen:
            continue
        seen.add(key)
        match_set = set(match)
        n_idxs = [i for i in match_set
                  if mol.GetAtomWithIdx(i).GetSymbol() == 'N']
        if len(n_idxs) != 1:
            # >1 N (amidine / imide) -- N-substitution not handled here; the
            # bare-suffix path (no N-subs) still applies if there are none.
            continue
        n_idx = n_idxs[0]
        # The FG anchor is the sulfonamide S or the carbonyl C (=O) bonded to N
        # -- NOT just any C neighbour (an N-alkyl carbon is also a C neighbour).
        anchor = None
        for nb in mol.GetAtomWithIdx(n_idx).GetNeighbors():
            if nb.GetIdx() not in match_set:
                continue
            if nb.GetSymbol() == 'S':
                anchor = nb.GetIdx()
                break
            if nb.GetSymbol() == 'C' and any(
                    b.GetBondType() == Chem.BondType.DOUBLE
                    and b.GetOtherAtom(nb).GetSymbol() == 'O'
                    for b in nb.GetBonds()):
                anchor = nb.GetIdx()
                break
        if anchor is None:
            return None  # can't locate the amide/sulfonamide anchor -> fail closed
        exclude = {n_idx, anchor}
        for nb in mol.GetAtomWithIdx(n_idx).GetNeighbors():
            j = nb.GetIdx()
            if j in exclude or nb.GetAtomicNum() <= 1:
                continue
            frag: set = set()
            stack = [j]
            while stack:
                x = stack.pop()
                if x in frag or x in exclude:
                    continue
                frag.add(x)
                for n2 in mol.GetAtomWithIdx(x).GetNeighbors():
                    if n2.GetIdx() not in frag and n2.GetIdx() not in exclude:
                        stack.append(n2.GetIdx())
            nm = name_substituent(mol, frag, j, allow_mancude=allow_mancude)
            if is_refusal_sentinel(nm):
                return None  # unnameable N-substituent -> fail closed
            n_sub_atoms |= frag
            name_counts[nm] = name_counts.get(nm, 0) + 1
            bindings.append(TokenBinding(tuple(sorted(frag)), nm, 'prefix'))

    entries: List = []
    for nm, cnt in name_counts.items():
        if is_complex_substituent(nm):
            disp = _wrap_n_substituent(nm) if '(' in nm else f"({nm})"
        else:
            disp = _wrap_n_substituent(nm)
        if cnt == 1:
            token = f"N-{disp}"
        else:
            mult = SIMPLE_MULTIPLIERS.get(cnt)
            if mult is None:
                return None
            token = f"{','.join(['N'] * cnt)}-{mult}{disp}"
        entries.append((_alpha_key(nm), token))
    return n_sub_atoms, entries, bindings


def name_general_ring(
    mol, features, allow_aromatic_general: bool = False,
    allow_charged: bool = False,
) -> Optional[GeneralEngineResult]:
    """v25 G2: universal von-Baeyer ring-parent path (opt-in engine only).

    v26 P0: ``allow_aromatic_general`` is threaded to
    ``analyze_cage_universal(..., allow_mancude=...)`` (plumbing only; the
    mancude refusal there still fires unconditionally until P2).

    v26 P5: ``allow_charged`` (only under ``complete``) lifts the net-charge
    refusal and emits a charge suffix on a von-Baeyer cage skeletal atom
    (``...pentaen-4-ium``); fail-closed otherwise.
    """
    from ..rules.vonbaeyer_universal import analyze_cage_universal
    from ..rules.polycyclic import _build_parent_with_unsaturation
    from ..rules.ring_selection import select_principal_ring_system
    from ..rules.seniority import get_suffix
    from .substituent_enumerator import discover_substituents, name_substituent

    reason = _common_refusal(mol, allow_charged=allow_charged)
    if reason:
        return _refuse(reason)
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    if not ring_atoms:
        return _refuse("acyclic (chain path owns it)")
    if getattr(features, 'chain_is_parent', False):
        return _refuse("chain parent (chain path owns it)")

    ring_systems = list(getattr(features, 'ring_systems', None) or [])
    if ring_systems:
        # v30 vB-engine Piece 1 (P-44.1): under best-effort, honour the
        # principal-group-bearing ring system in selection so the true parent is
        # not orphaned into an unnameable substituent. Passed ONLY when
        # allow_aromatic_general (best-effort) -> PIN gets no hint -> byte-identical.
        pg_hint = (list(getattr(features, 'principal_group_atoms', None) or [])
                   if allow_aromatic_general else None)
        senior = select_principal_ring_system(
            mol, ring_systems, principal_group_atoms=pg_hint)
        cage_seed = set(senior) if senior else None
    else:
        cage_seed = None

    cage = analyze_cage_universal(
        mol, cage_atoms=cage_seed, allow_mancude=allow_aromatic_general)
    if cage is None:
        return _refuse("cage unanalyzable (monocycle/spiro/caps/kekulize)")

    return _emit_ring_from_analysis(
        mol, features, cage, allow_aromatic_general, allow_charged)


def _emit_ring_from_analysis(
    mol, features, cage, allow_aromatic_general: bool, allow_charged: bool,
) -> Optional[GeneralEngineResult]:
    """v27 P3: shared ring-emission tail — ring suffix (P-6x) + substituent
    recursion (P-29.2) + charge suffix + parent-scope stereo — for a
    ``UniversalCage`` OR ``SpiroSystem`` analysis -- two forms of the ONE field
    contract declared in ``rules/vonbaeyer_universal.RingAnalysis`` (they inherit
    it and add nothing, so a field added there reaches both by construction).

    Extracted verbatim from ``name_general_ring`` so the spiro parent producer
    (``name_general_spiro``) reuses the whole tail — ``_RING_SUFFIX_STYLES``,
    the ester functional-class two-word, amide/sulfonamide N-substituents, the
    enone valence guard, substituent recursion, charge and stereo — with zero
    duplication. Keys only on ``cage.{cage_atoms,atom_to_locant,unsaturation,
    total_atoms,hetero_prefix,descriptor}`` so it is analysis-form-agnostic."""
    from ..rules.polycyclic import _build_parent_with_unsaturation
    from ..rules.seniority import get_suffix
    from .substituent_enumerator import discover_substituents, name_substituent

    cage_set = set(cage.cage_atoms)
    atom_to_locant = dict(cage.atom_to_locant)

    # v27 P5 (P-25.3): PIN-quality fusion upgrade over the VB polyene for a BARE
    # mancude FUSED parent (no suffix, no substituents). Fail-closed to the VB
    # polyene: name_fusion_parent only returns a fusion word that is a vetted
    # catalog exact-match (Java-free) or AFFIRMATIVE-RT-verified (rejects the
    # orientation slips + any stereo the bare word cannot express -> 0-wrong in
    # every environment). Skipped for spiro (not a fusion system).
    if (allow_aromatic_general and getattr(cage, 'is_mancude', False)
            and getattr(features, 'principal_group', None) is None
            and not str(cage.descriptor).startswith(
                ('spiro', 'dispiro', 'trispiro', 'tetraspiro', 'pentaspiro'))):
        from .general_fusion import name_fusion_parent
        from ..rules.stereochemistry import general_engine_stereo_complete
        _fusion_word = name_fusion_parent(mol, cage.cage_atoms)
        # v27 Phase S Task 2: the bare fusion word carries NO stereo block, so
        # take this early-return ONLY when the parent has no defined stereo
        # element (all-or-nothing, PS-1). If a mancude fused parent DID carry a
        # ring stereocentre / ring-bond E/Z, fall through to the VB polyene tail
        # (line ~893) whose _stereo_prefix expresses it — never ship a stereo-
        # dropping fusion word past the stereo-blind SELF-01. For the common
        # achiral aromatic parent (the P5 win) this is byte-identical (nd == 0).
        if _fusion_word and general_engine_stereo_complete(mol, _fusion_word):
            return GeneralEngineResult(
                name=_fusion_word,
                bindings=(TokenBinding(tuple(cage.cage_atoms),
                                       _fusion_word, 'parent'),))

    # --- suffix (ring forms; every PG instance must touch the cage) ---
    pg = getattr(features, 'principal_group', None)
    pg_matches = list(getattr(features, 'principal_group_atoms', None) or [])
    suffix_core = None
    suffix_atoms: set = set()
    pg_locants: List[int] = []
    ester_r_word = None          # v27 P2: alcoholic `R-yl ` word (functional class)
    ester_r_frag: set = set()    # v27 P2: R-alkyl atoms held out of discovery
    n_sub_atoms: set = set()     # v27 P2: amide/sulfonamide N-substituent atoms
    n_sub_entries: List = []     # v27 P2: (alpha_key, 'N-…' token)
    n_sub_bindings: List[TokenBinding] = []
    if pg:
        suffix_core = get_suffix(pg, is_ring=True)
        if suffix_core not in _RING_SUFFIX_STYLES:
            return _refuse(f"unsupported ring suffix for pg={pg!r}")
        if suffix_core == 'carboxylate':
            # v27 P2 (P-65.6.3.2.1): ring ester -> functional-class two-word.
            # The acid core (C=O, ester O) becomes the appended 'carboxylate'
            # suffix on the ring; the alcoholic R is prepended as a word.
            er = _extract_ring_ester(
                mol, pg_matches, cage_set, allow_aromatic_general)
            if er is None:
                return _refuse("ester not a clean acyclic ring-acid mono-ester")
            ester_r_word, ester_r_frag, _acid_core, _ring_attach = er
            suffix_atoms.update(_acid_core)
            pg_locants.append(atom_to_locant[_ring_attach])
        else:
            seen = set()
            for match in pg_matches:
                key = tuple(sorted(match))
                if key in seen:
                    continue
                seen.add(key)
                on_cage = [i for i in match if i in cage_set]
                if suffix_core in _INLINE_SUFFIX_CORES:
                    # P-64.2.2.2 (:28386) -- see _inline_suffix_locant.
                    loc = _inline_suffix_locant(pg, match, cage_set,
                                                atom_to_locant)
                    if loc is None:
                        return _refuse(
                            "inline suffix characteristic atom off cage")
                elif on_cage:
                    loc = min(atom_to_locant[i] for i in on_cage)
                else:
                    # appended suffix (e.g. -carboxylic acid): match sits fully
                    # off-cage; locant = the cage neighbor of any match atom.
                    nbrs = [n.GetIdx()
                            for i in match
                            for n in mol.GetAtomWithIdx(i).GetNeighbors()
                            if n.GetIdx() in cage_set]
                    if not nbrs:
                        return _refuse("PG instance not attached to cage")
                    loc = min(atom_to_locant[n] for n in nbrs)
                pg_locants.append(loc)
                suffix_atoms.update(i for i in match
                                    if i not in cage_set
                                    and mol.GetAtomWithIdx(i).GetAtomicNum() > 1)

        # v27 P2: N-substituents on a ring carboxamide / sulfonamide. Held out
        # of both the appended-suffix atoms and the ring-substituent set, cited
        # with the italic N-locant. Fail closed if any is unnameable (never drop
        # an N-substituent -> that would ship a different constitution).
        if suffix_core in ('carboxamide', 'sulfonamide'):
            n_res = _ring_amide_n_prefixes(
                mol, pg_matches, allow_aromatic_general)
            if n_res is None:
                return _refuse("ring amide/sulfonamide N-substituent unnameable")
            n_sub_atoms, n_sub_entries, n_sub_bindings = n_res
            suffix_atoms -= n_sub_atoms

    # v25 G5-A defense-in-depth: a ring '-one'/'-imine' locant must never
    # coincide with a ring double-bond locant -- that carbon would be both =ring
    # and =O/=N (the 5-bond-carbon class). v26 P2 lifted the aromatic-cage refusal
    # in analyze_cage_universal behind ``allow_mancude`` (--emit-tier complete),
    # so mancude cages reach this guard too; the guard keys on
    # ``double_bond_pairs`` (populated for both the kekulized mancude polyene and
    # the isolated ene). v27 P2 reproduce-first finding: RELAXING this guard to a
    # true-valence test does NOT unlock RT-valid fused enones (anthrone /
    # inden-1-one / fused dienones emit von-Baeyer names that do NOT round-trip --
    # the real blocker is the cage kekulization/numbering, a deeper Phase-4 fix),
    # and would add a jar-absent best-effort wrongness risk, so the guard is kept
    # conservative. v27 P2 EXTENDS it to '-imine' (the new inline aza-'-one'
    # suffix shares the same valence constraint). Fail-closed.
    if suffix_core in ('one', 'imine'):
        # both endpoints of every ring double bond are termini a =O/=N cannot share
        _ene_termini = {loc for pair in cage.unsaturation.get('double_bond_pairs', ())
                        for loc in pair}
        if _ene_termini & set(pg_locants):
            return _refuse("ring ketone/imine locant coincides with ring double "
                           "bond (valence)")

    # P-74.1.2 (BlueBookV2.md:42445, sentence :42447) is the governing case for
    # the R7 shape -- a skeletal ring cation plus a characteristic-group-derived
    # anionic suffix. P-74.1.1 (:42419) supplies the construction: a zwitterion
    # whose ionic centres lie in THIS parent is "not considered as a neutral
    # compound" -> cumulative ionic suffixes (cationic before anionic, P-74.1.2
    # :42447). Planned BEFORE discovery because the anionic oxygen of an -olate
    # must be HELD OUT of the substituent partition -- otherwise it is also
    # spelled as a neutral 'oxo'/'hydroxy' prefix and the atom is counted twice.
    # Declined when an FG suffix is already present: the cumulative FG+charge
    # construction is out of scope (same rule as _append_charge_suffix).
    zwit_plan = (_zwitterion_suffix_plan(mol, atom_to_locant)
                 if (allow_charged and not suffix_core) else None)
    zwit_held: set = set(zwit_plan[0]) if zwit_plan else set()

    # --- substituent prefixes (generic ordered-atom discovery + recursion) ---
    ordered_cage = sorted(cage_set, key=lambda i: atom_to_locant[i])
    try:
        subs = discover_substituents(
            mol, cage_set | suffix_atoms | ester_r_frag | n_sub_atoms | zwit_held,
            parent_type='chain',
            principal_chain=ordered_cage, atom_to_locant=atom_to_locant,
            general_fallback=True)
    except AssertionError as e:
        return _refuse(f"partition incomplete: {e}")
    if subs is None:
        # v28 Composer1 Task 5: unassigned atom off a suffix/FG atom -> fail
        # closed (never drop it).
        return _refuse("partition incomplete: unassigned atoms off suffix/FG")

    groups: Dict[str, List[int]] = {}
    frag_bindings: List[TokenBinding] = []
    for sub in subs:
        frag = set(sub.frag_atoms)
        attach_nbrs = [n.GetIdx() for n in
                       mol.GetAtomWithIdx(sub.attach_mol_idx).GetNeighbors()
                       if n.GetIdx() in frag]
        if not attach_nbrs:
            return _refuse("substituent without cage attachment")
        # v27 P1: under the complete/best-effort tier (allow_aromatic_general),
        # a multi-ring cage substituent is named via the universal von-Baeyer
        # engine (parent<->substituent symmetry). PIN default (flag off) is
        # byte-identical — name_substituent's allow_mancude defaults False.
        prefix = name_substituent(mol, frag, attach_nbrs[0],
                                  allow_mancude=allow_aromatic_general)
        if is_refusal_sentinel(prefix):
            return _refuse("branch unnameable (tier-5 fallback)")
        groups.setdefault(prefix, []).append(sub.locant)
        frag_bindings.append(TokenBinding(tuple(sorted(frag)), prefix,
                                          'prefix'))

    # Ring substituents and N-substituents are detachable prefixes cited in one
    # alphanumeric order (P-14.5.2); N-substituent tokens carry the italic N
    # locant instead of a numeral but sort by the same substituent-name key.
    prefix_entries: List = list(n_sub_entries)
    for prefix in groups:
        locs = sorted(groups[prefix])
        text = _mult_prefix(len(locs), prefix)
        if text is None:
            return _refuse("multiplicity beyond table")
        prefix_entries.append(
            (_alpha_key(prefix), ','.join(map(str, locs)) + '-' + text))
    prefix_entries.sort(key=lambda t: t[0])
    prefix_parts = [tok for _, tok in prefix_entries]

    # --- parent block: hetero-prefix + descriptor + parent(+ene)(+suffix) ---
    fg_suffix = None
    bindings = frag_bindings + n_sub_bindings
    if suffix_core:
        fg_suffix = {'suffix': suffix_core, 'locants': sorted(pg_locants),
                     'type': _RING_SUFFIX_STYLES[suffix_core]}
        if suffix_atoms:
            bindings.append(TokenBinding(tuple(sorted(suffix_atoms)),
                                         suffix_core, 'suffix'))
    parent_block = _build_parent_with_unsaturation(
        cage.total_atoms, cage.unsaturation, fg_suffix=fg_suffix)

    # Hyphen glue (mirrors polycyclic.py:2830): a substituent prefix keeps its
    # joining '-' only before a locant-initial hetero prefix.
    core = cage.hetero_prefix + cage.descriptor + parent_block
    if prefix_parts:
        joined = '-'.join(prefix_parts)
        head = cage.hetero_prefix or cage.descriptor
        name = joined + ('-' if head[:1].isdigit() else '') + core
    else:
        name = core

    # P-74.1.1: cumulative ionic suffixes for a net-neutral zwitterion, then the
    # v26 P5 net-charge suffix, then the fail-closed backstop. A molecule with a
    # genuine ionic centre must NEVER receive a neutral name -- that is a
    # different (uncharged) species, not an approximation.
    if zwit_plan is not None:
        name = _elide_before_ionic_suffix(name, zwit_plan[1])
        if zwit_held:
            bindings.append(TokenBinding(tuple(sorted(zwit_held)),
                                         zwit_plan[1].lstrip('-'), 'suffix'))
    elif allow_charged and Chem.GetFormalCharge(mol) != 0:
        name = _append_charge_suffix(name, mol, atom_to_locant,
                                     has_fg_suffix=bool(suffix_core))
        if name is None:
            return _refuse("charge not expressible as a cage-parent suffix")
    elif _has_ionic_centres(mol):
        return _refuse("ionic centres not expressible as a cage-parent suffix")

    # v25 G4: parent-scope stereo from structure (VB locants).
    name = _stereo_prefix(mol, atom_to_locant) + name

    # v27 P2 (P-65.6.3.2.1): prepend the alcoholic component as a separate word
    # -> `ethyl <ring>carboxylate` (functional-class ester two-word PIN).
    if ester_r_word is not None:
        name = ester_r_word + ' ' + name
        bindings.append(TokenBinding(tuple(sorted(ester_r_frag)),
                                     ester_r_word, 'prefix'))

    bindings.extend(_ring_parent_bindings(cage, name, parent_block))
    return GeneralEngineResult(name=name, bindings=tuple(bindings))


def _ring_parent_bindings(cage, name: str, parent_block: str
                          ) -> List[TokenBinding]:
    """v29 Phase 2 T5: bind the ring parent as the tokens that SPELL it.

    The spelled parent word is ``hetero_prefix + descriptor + stem +
    (ene/yne block) (+ suffix)``. This used to be a single binding whose token
    was ``cage.descriptor`` ALONE -- ``'bicyclo[2.2.1]'`` -- claiming every cage
    atom. Two things were wrong with that, and both showed up in audit mode:

    * the token carried no stem, so the arity oracle could not decide what it
      spells (``ARITY_UNVERIFIED``, and with no other token on the molecule,
      ``PROOF_UNSUBSTANTIATED``: nothing about the name was corroborated); and
    * the replacement morphemes were bound to nothing, so P5 reported the
      leftover ``'xa'`` of ``7-oxa`` as name text no binding accounts for.

    Who claims the heteroatom is decided by the oracle, not by preference.
    ``token_arity('bicyclo[2.2.1]hept')`` is confidently **7** -- the descriptor
    arithmetic and the stem independently agree on 7 SKELETAL POSITIONS, and a
    replacement prefix does not change that count. The oracle's verdict on the
    morpheme itself is the other half: ``'oxa'`` "qualifies no skeleton, so its
    ZERO atoms replace nothing". A replacement prefix says which ELEMENT sits at
    a position the stem has already counted; it contributes no atom of its own.

    So the parent keeps the WHOLE cage (anything less is an ``ARITY_MISMATCH``
    error against a correct name -- the exact false alarm this task exists to
    remove) and each replacement morpheme binds ZERO atoms. That satisfies
    EXCLUSIVE CLAIM trivially, and P-24.2 agrees: the prefix replaces, the stem
    counts.

    One binding per HETEROATOM, not per distinct morpheme. ``'7,8-diaza…'``
    writes ``aza`` once and expresses the count with the multiplier ``di``, but
    P4 counts a multiplied morpheme as spelled once PER multiplicand, so it
    reads that name as spelling ``aza`` twice. Collapsing the two nitrogens into
    a single ``'aza'`` binding therefore trades the old over-claim for the
    mirror-image ``MULTIPLICITY_MISMATCH`` ("spelled 2 times but 1 binding
    claims it") -- measured, not predicted.

    The unsaturation block is deliberately left out of every token here: it
    spells BONDS, not atoms, so it binds nothing and the parent token stops at
    the stem rather than swallowing it.
    """
    cage_atoms = tuple(cage.cage_atoms)
    descriptor = str(cage.descriptor)
    single = [TokenBinding(cage_atoms, descriptor, 'parent')]

    from ..data.chain_names import get_chain_prefix
    try:
        stem = get_chain_prefix(cage.total_atoms)
    except ValueError:
        return single
    parent_token = descriptor + stem
    # Only claim a token the FINAL name really spells: P4 anchors tokens to
    # spans, so an unlocatable token would be a fresh TOKEN_ABSENT. The
    # ``parent_block`` check keeps this honest about the stem specifically
    # (spiro/other analysis shapes may not open with this stem at all).
    if not parent_block.startswith(stem) or parent_token not in name:
        return single

    bindings = [TokenBinding(cage_atoms, parent_token, 'parent')]
    hetero_per_atom = tuple(getattr(cage, 'hetero_per_atom', ()) or ())
    if cage.hetero_prefix and not hetero_per_atom:
        # Defence-in-depth, not a hard access: ``RingAnalysis`` makes this
        # unreachable for the two forms that exist today (both inherit the
        # field, neither can omit it), so this is deliberately a warning, not
        # a raise -- a future third analysis form that inherits the field but
        # never populates it degrades to the pre-af0d7262 audit finding
        # (UNBOUND_MORPHEME) rather than a wrong name, and a crash on a naming
        # path is worse than a silent audit gap. This turns that gap noisy.
        logger.warning(
            "_ring_parent_bindings: hetero_prefix=%r but hetero_per_atom is "
            "empty on %s; no replacement binding will be emitted for this "
            "morpheme (falls back to the P5 UNBOUND_MORPHEME audit finding)",
            cage.hetero_prefix, type(cage).__name__)
    for _atom_idx, morpheme in hetero_per_atom:
        if morpheme not in name:
            continue
        bindings.append(TokenBinding((), morpheme, 'replacement'))
    return bindings


def _ring_suffix_text(core: str, locants: List[int]) -> Optional[str]:
    """Ring suffix WITHOUT the parent-elision decision: '-3-ol', '-2,5-diol',
    '-2-carboxylic acid', '-1,3-dicarboxylic acid'. None on unsupported
    multiplicity. The parent-'e' elision is applied by the caller from the
    first alphabetic char of this string (mirrors chain ``_assemble``)."""
    n = len(locants)
    if n == 0:
        return None
    mult = '' if n == 1 else _MULT_SIMPLE.get(n)
    if n > 1 and mult is None:
        return None
    return '-' + ','.join(map(str, sorted(locants))) + '-' + (mult or '') + core


def _bond_ring_locant(la: int, lb: int, n: int) -> int:
    """Ring-bond locant for a bond between ring positions ``la`` and ``lb``
    (1..n): the lower endpoint, except the wraparound bond {n, 1} which is
    cited as ``n`` (P-31.1.4.3 lowest-locant numbering already handled by the
    orienter; this only formats the chosen orientation)."""
    if {la, lb} == {1, n}:
        return n
    return min(la, lb)


def _orient_carbocycle(mol, ring_order, sub_positions, pg_ring_atoms):
    """Number an all-carbon monocycle by lowest locants to, in order:
    principal group -> ring unsaturation -> substituents -> canonical rank
    (P-14.4 / P-31.1.4). Returns (oriented_ring, atom_to_locant).

    The canonical-rank final tier is a deterministic symmetry-breaker; among
    orientations that tie on every IUPAC criterion the choice is round-trip
    equivalent (SELF-01 is the downstream authority for the emitted string)."""
    ring_list = list(ring_order)
    n = len(ring_list)
    canon = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    best_key = None
    best_oriented = None
    for start in range(n):
        for direction in (1, -1):
            oriented = [ring_list[(start + direction * k) % n] for k in range(n)]
            loc = {a: i + 1 for i, a in enumerate(oriented)}
            pg = sorted(loc[i] for i in pg_ring_atoms if i in loc)
            uns = []
            for i in range(n):
                a, b = oriented[i], oriented[(i + 1) % n]
                bond = mol.GetBondBetweenAtoms(a, b)
                if bond is not None and bond.GetBondTypeAsDouble() in (2.0, 3.0):
                    uns.append(_bond_ring_locant(loc[a], loc[b], n))
            uns.sort()
            sub = sorted(loc[i] for i in sub_positions if i in loc)
            key = (pg, uns, sub, [canon[a] for a in oriented])
            if best_key is None or key < best_key:
                best_key = key
                best_oriented = oriented
    return best_oriented, {a: i + 1 for i, a in enumerate(best_oriented)}


def _carbocycle_parent_name(mol, oriented, atom_to_locant) -> Optional[str]:
    """Parent name for an all-carbon monocycle: 'benzene' (6-membered
    aromatic), 'cyclohexane' (saturated), or 'cyclohex-1-ene' style
    (unsaturated). None -> fail closed (aromatic non-6, unnameable stem)."""
    from ..data.chain_names import get_chain_prefix
    n = len(oriented)
    ring_set = set(oriented)
    if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in oriented):
        return 'benzene' if n == 6 else None
    base = get_chain_prefix(n)
    if not base:
        return None
    ene, yne = [], []
    for i in range(n):
        a, b = oriented[i], oriented[(i + 1) % n]
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None:
            return None
        order = bond.GetBondTypeAsDouble()
        if order == 1.0:
            continue
        loc = _bond_ring_locant(atom_to_locant[a], atom_to_locant[b], n)
        if order == 2.0:
            ene.append(loc)
        elif order == 3.0:
            yne.append(loc)
        else:
            return None  # aromatic/dative bond in a non-aromatic ring: bail
    ene.sort()
    yne.sort()
    if not ene and not yne:
        return 'cyclo' + base + 'ane'
    stem = 'cyclo' + base
    if ene:
        mult = _MULT_SIMPLE.get(len(ene), '') if len(ene) > 1 else ''
        if len(ene) > 1:
            stem += 'a'
        stem += '-' + ','.join(map(str, ene)) + '-' + mult + 'ene'
    if yne:
        if ene:
            stem = stem[:-1]  # 'ene' -> 'en' before '-N-yne'
        mult = _MULT_SIMPLE.get(len(yne), '') if len(yne) > 1 else ''
        if len(yne) > 1 and not ene:
            stem += 'a'
        stem += '-' + ','.join(map(str, yne)) + '-' + mult + 'yne'
    return stem


def name_general_monocycle(
    mol, features, allow_aromatic_general: bool = False,
    allow_charged: bool = False,
) -> Optional[GeneralEngineResult]:
    """v26 P1: general LONE-monocycle ring-parent path (opt-in engine only).

    Names a molecule whose SENIOR ring system is a single (non-fused) ring
    -- benzene, pyridine, thiophene, imidazole, ... -- with its substituents
    coming from the never-None universal recursion
    (``substituent_enumerator.name_substituent``) instead of the default
    composer's finite per-class vocabulary. That is the root-cause fix for
    "bare ring names, substituted form abstains."

    The parent ring name + numbering come from the EXISTING lowest-locant
    machinery (``name_heterocycle`` + ``orient_heterocycle_with_substituents``
    for heterocycles; ``benzene``/cycloalkane for all-carbon). Fail-closed
    (returns None) on anything outside this scope; the E1 atom-partition and
    SELF-01 OPSIN round-trip are the downstream authorities.

    Gated behind ``allow_aromatic_general`` -- inert (returns None) when the
    flag is False, so the PIN/default path stays byte-identical.
    """
    if not allow_aromatic_general:
        return None

    from ..rules.ring_selection import select_principal_ring_system
    from ..rules.seniority import get_suffix
    from ..rules.heterocycles import (
        name_heterocycle, orient_heterocycle_with_substituents,
    )
    from .substituent_enumerator import discover_substituents, name_substituent

    reason = _common_refusal(mol, allow_charged=allow_charged)
    if reason:
        return _refuse(reason)
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() == 0:
        return _refuse("acyclic (chain path owns it)")
    if getattr(features, 'chain_is_parent', False):
        return _refuse("chain parent (chain path owns it)")

    # --- identify the parent monocycle: the SENIOR ring system must be a
    #     single, non-fused ring (substituents MAY carry their own rings; the
    #     universal recursion names them). ---
    ring_systems = list(getattr(features, 'ring_systems', None) or [])
    if ring_systems:
        senior = select_principal_ring_system(mol, ring_systems)
        senior_set = set(senior) if senior else None
    else:
        senior_set = None
    atom_rings = [tuple(r) for r in ring_info.AtomRings()]
    parent_ring = None
    if senior_set is not None:
        subrings = [r for r in atom_rings if set(r) <= senior_set]
        if len(subrings) == 1 and set(subrings[0]) == senior_set:
            parent_ring = subrings[0]
    elif len(atom_rings) == 1:
        parent_ring = atom_rings[0]
    if parent_ring is None:
        return _refuse("no lone monocycle parent (fused/cage or ambiguous)")

    ring_set = set(parent_ring)

    # --- suffix (ring forms; mirror name_general_ring) ---
    pg = getattr(features, 'principal_group', None)
    pg_matches = list(getattr(features, 'principal_group_atoms', None) or [])
    suffix_core = None
    suffix_atoms: set = set()
    pg_locants: List[int] = []
    pg_ring_atoms: set = set()
    # v30 RISK 5 Class 1 (P-66.6.1): a ring ester (LACTONE) -- carbonyl C AND the
    # single-bonded ester O both in THIS ring -- is the oxa-heterocycle bearing a
    # ring '-one', never a '<ring>-carboxylate' (the ester producer emits an
    # impossible ring-O locant + an anion suffix on a neutral: '2,5-dihydrofuran-1-
    # carboxylate'). Reclassify to a ring KETONE: the ring O stays the parent's
    # heteroatom (named by `name_heterocycle`), the exocyclic =O is the '-one'.
    # Both references decompose the ring ester at perception the same way (refR5).
    # The synthetic match `(exo_o, carbonyl_c)` puts the carbonyl C at index 1 for
    # `_pg_attachment_atoms('ketone', ...)` and leaves the exocyclic O as the sole
    # off-ring suffix atom. Verified RT: O=C1OCC=C1 -> 2,5-dihydrofuran-2-one.
    if pg == 'ester' and pg_matches:
        _lac = _detect_ring_lactone(mol, pg_matches, ring_set)
        if _lac is not None:
            _carbonyl_c, _exo_o = _lac
            pg = 'ketone'
            pg_matches = [(_exo_o, _carbonyl_c)]
    if pg:
        suffix_core = get_suffix(pg, is_ring=True)
        if suffix_core not in _RING_SUFFIX_STYLES:
            return _refuse(f"unsupported ring suffix for pg={pg!r}")

    # --- numbering (heteroatom/pg/substituent lowest-locant) + parent name ---
    sub_positions: set = set()  # filled after suffix_atoms known; provisional below
    # First pass: collect off-ring pg (suffix) atoms so they are excluded from
    # the substituent set that drives numbering.
    if pg:
        seen = set()
        for match in pg_matches:
            key = tuple(sorted(match))
            if key in seen:
                continue
            seen.add(key)
            suffix_atoms.update(i for i in match
                                if i not in ring_set
                                and mol.GetAtomWithIdx(i).GetAtomicNum() > 1)
            pg_ring_atoms.update(
                _pg_bearing_ring_atoms(mol, pg, match, ring_set))
    for i in ring_set:
        for nb in mol.GetAtomWithIdx(i).GetNeighbors():
            j = nb.GetIdx()
            if (j not in ring_set and j not in suffix_atoms
                    and nb.GetAtomicNum() > 1):
                sub_positions.add(i)
                break

    has_hetero = any(mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in ring_set)
    if has_hetero:
        oriented, atom_to_locant = orient_heterocycle_with_substituents(
            mol, parent_ring, sub_positions, pg_ring_atoms)
        parent_name = name_heterocycle(mol, parent_ring)
    else:
        oriented, atom_to_locant = _orient_carbocycle(
            mol, parent_ring, sub_positions, pg_ring_atoms)
        parent_name = _carbocycle_parent_name(mol, oriented, atom_to_locant)
    if not parent_name or 'unknown' in parent_name.lower():
        return _refuse("monocycle parent name underivable")
    if any(i not in atom_to_locant for i in ring_set):
        return _refuse("ring atom missing from numbering")

    # --- pg locants under the chosen numbering (mirror name_general_ring) ---
    if pg:
        seen = set()
        for match in pg_matches:
            key = tuple(sorted(match))
            if key in seen:
                continue
            seen.add(key)
            on_ring = [i for i in match if i in ring_set]
            if suffix_core in _INLINE_SUFFIX_CORES:
                # P-64.2.2.2 (:28386) -- see _inline_suffix_locant.
                loc = _inline_suffix_locant(pg, match, ring_set,
                                            atom_to_locant)
                if loc is None:
                    return _refuse(
                        "inline suffix characteristic atom off ring")
            elif on_ring:
                loc = min(atom_to_locant[i] for i in on_ring)
            else:
                nbrs = [n.GetIdx()
                        for i in match
                        for n in mol.GetAtomWithIdx(i).GetNeighbors()
                        if n.GetIdx() in ring_set]
                if not nbrs:
                    return _refuse("PG instance not attached to ring")
                loc = min(atom_to_locant[n] for n in nbrs)
            pg_locants.append(loc)

    # A '-one' on an aromatic ring carbon needs added-hydrogen machinery this
    # path does not build (pyridin-2(1H)-one); fail closed rather than emit an
    # aromatic-carbon-with-=O impossibility.
    if suffix_core == 'one' and any(
            mol.GetAtomWithIdx(i).GetIsAromatic()
            for m in pg_matches for i in m if i in ring_set):
        return _refuse("ring ketone on aromatic carbon (added-H not built)")

    # --- substituent prefixes (universal recursion; mirror name_general_ring) ---
    ordered_ring = sorted(ring_set, key=lambda i: atom_to_locant[i])
    try:
        subs = discover_substituents(
            mol, ring_set | suffix_atoms, parent_type='chain',
            principal_chain=ordered_ring, atom_to_locant=atom_to_locant,
            general_fallback=True)
    except AssertionError as e:
        return _refuse(f"partition incomplete: {e}")
    if subs is None:
        # v28 Composer1 Task 5: unassigned atom off a suffix/FG atom -> fail
        # closed (never drop it).
        return _refuse("partition incomplete: unassigned atoms off suffix/FG")

    groups: Dict[str, List[int]] = {}
    frag_bindings: List[TokenBinding] = []
    for sub in subs:
        frag = set(sub.frag_atoms)
        attach_nbrs = [n.GetIdx() for n in
                       mol.GetAtomWithIdx(sub.attach_mol_idx).GetNeighbors()
                       if n.GetIdx() in frag]
        if not attach_nbrs:
            return _refuse("substituent without ring attachment")
        if len(attach_nbrs) != 1:
            return _refuse(
                "substituent attaches to parent ring at >1 point "
                "(spiro/fused/bridge)")
        # v27 P1: complete-tier cage substituent recursion (see name_general_ring).
        prefix = name_substituent(mol, frag, attach_nbrs[0],
                                  allow_mancude=allow_aromatic_general)
        if is_refusal_sentinel(prefix):
            return _refuse("branch unnameable (tier-5 fallback)")
        groups.setdefault(prefix, []).append(sub.locant)
        frag_bindings.append(TokenBinding(tuple(sorted(frag)), prefix,
                                          'prefix'))

    prefix_parts = []
    for prefix in sorted(groups, key=_alpha_key):
        locs = sorted(groups[prefix])
        text = _mult_prefix(len(locs), prefix)
        if text is None:
            return _refuse("multiplicity beyond table")
        prefix_parts.append(','.join(map(str, locs)) + '-' + text)

    # --- suffix text + parent-'e' elision ---
    bindings = frag_bindings
    core = parent_name
    if suffix_core:
        suffix_text = _ring_suffix_text(suffix_core, pg_locants)
        if suffix_text is None:
            return _refuse("unsupported suffix multiplicity/placement")
        first_alpha = next((c for c in suffix_text if c.isalpha()), '')
        if core.endswith('e') and first_alpha in 'aeiouy':
            core = core[:-1]
        core = core + suffix_text
        if suffix_atoms:
            bindings.append(TokenBinding(tuple(sorted(suffix_atoms)),
                                         suffix_core, 'suffix'))

    # --- glue prefixes + core (mirror name_general_ring hyphen rule) ---
    if prefix_parts:
        joined = '-'.join(prefix_parts)
        name = joined + ('-' if core[:1].isdigit() else '') + core
    else:
        name = core

    # v26 P5: charge suffix on a ring skeletal atom (fail closed otherwise).
    if allow_charged and Chem.GetFormalCharge(mol) != 0:
        name = _append_charge_suffix(name, mol, atom_to_locant,
                                     has_fg_suffix=bool(suffix_core))
        if name is None:
            return _refuse("charge not expressible as a monocycle-parent suffix")
    elif _has_ionic_centres(mol):
        # P-74.1.1 (:42419) fail-closed backstop -- see the chain producer.
        return _refuse("ionic centres not expressible as a monocycle-parent "
                       "suffix")

    # v25 G4: parent-scope stereo from structure (ring locants).
    name = _stereo_prefix(mol, atom_to_locant) + name

    # E1 parent binding: a stem guaranteed to survive suffix elision.
    parent_token = parent_name[:-1] if parent_name.endswith('e') else parent_name
    bindings.append(TokenBinding(tuple(sorted(ring_set)), parent_token, 'parent'))
    return GeneralEngineResult(name=name, bindings=tuple(bindings))


def name_general_spiro(
    mol, features, allow_aromatic_general: bool = False,
    allow_charged: bool = False,
) -> Optional[GeneralEngineResult]:
    """v27 P3 (P-24.2): general SPIRO ring-parent path (opt-in engine only).

    The von-Baeyer cage engine refuses spiro (``<2 bridgeheads``), so a spiro
    ring system that the default per-class handlers abstain on (functionalized /
    mancude / suffix-form PIN) dies at ``name_general_ring``'s cage refusal.
    This sibling routes the senior spiro ring system through
    ``analyze_spiro_universal`` (audited) and reuses the SAME
    ``_emit_ring_from_analysis`` tail as the cage path. Fail-closed (None) on any
    non-spiro / unaudited system; the SELF-01 round-trip is the downstream gate.
    """
    from ..rules.vonbaeyer_universal import analyze_spiro_universal
    from ..rules.ring_selection import select_principal_ring_system

    reason = _common_refusal(mol, allow_charged=allow_charged)
    if reason:
        return _refuse(reason)
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    if not ring_atoms:
        return _refuse("acyclic (chain path owns it)")
    if getattr(features, 'chain_is_parent', False):
        return _refuse("chain parent (chain path owns it)")

    # A spiro system is perceived as SEPARATE ring systems (its rings share only
    # a single atom, not a fused edge), so the senior component is just one ring.
    # The spiro PARENT is the whole cluster of rings joined through spiro atoms:
    # grow the senior component across shared-atom junctions, then hand the full
    # cluster to analyze_spiro_universal (which fail-closes if it is not pure
    # spiro — e.g. a fused/bridged cluster).
    ring_systems = list(getattr(features, 'ring_systems', None) or [])
    if not ring_systems:
        return _refuse("no ring systems perceived")
    senior = select_principal_ring_system(mol, ring_systems)
    if not senior:
        return _refuse("no senior ring system")
    clusters = [set(s) for s in ring_systems]
    _merged = True
    while _merged:
        _merged = False
        for _i in range(len(clusters)):
            for _j in range(_i + 1, len(clusters)):
                if clusters[_i] & clusters[_j]:
                    clusters[_i] |= clusters[_j]
                    clusters.pop(_j)
                    _merged = True
                    break
            if _merged:
                break
    cage_seed = next((c for c in clusters if set(senior) <= c), None)
    if cage_seed is None:
        return _refuse("senior ring system not in any cluster")

    spiro = analyze_spiro_universal(
        mol, cage_atoms=cage_seed, allow_mancude=allow_aromatic_general)
    if spiro is None:
        return _refuse("not an analyzable spiro ring system")

    return _emit_ring_from_analysis(
        mol, features, spiro, allow_aromatic_general, allow_charged)


#: Ledger site id for the parent-hydride tier below. Exported so a test can
#: prove the tier EXECUTED rather than infer it from the emitted string --
#: the emitted string cannot distinguish "this tier built it" from "an existing
#: producer happened to build the same words".
TERMINAL_RING_PARENT_SITE = "general_engine._name_terminal_ring_parent"


def _name_terminal_ring_parent(
    mol, features, allow_aromatic_general: bool = False,
) -> Optional[GeneralEngineResult]:
    """v30: the LAST-RESORT whole-molecule PARENT-HYDRIDE tier.

    ``rules.terminal_ring.terminal_ring_name(mol, ring, free_valence_atom=None)``
    returns an audited von Baeyer / spiro / P-22.2.3-replacement parent hydride
    and has done since v30 PB -- but it was reachable ONLY as a ``-yl``
    substituent namer, so a bare ring system no catalog covers abstained even
    though the generator could name it. This is the parent-side sibling of the
    substituent-side wiring in
    ``substituent_enumerator._terminal_bare_ring_substituent`` (cited by symbol,
    not line: that file is edited often and a line number goes stale), and the
    same three constraints apply, each for a measured reason:

    * **Best-effort branch only** (``allow_aromatic_general``, True for
      ``complete``/``best-effort`` and False for ``pin`` --
      ``cli._emit_tier_flags``). These are valid **non-preferred** names:
      ``bicyclo[4.4.0]deca-1,3,5,7,9-pentaene`` is not the PIN for naphthalene,
      so putting them on the PIN return would break PIN byte-identity.
    * **LAST resort.** Called only after ``name_general_ring``,
      ``name_general_spiro`` and ``name_general_monocycle`` have ALL declined,
      i.e. where the molecule would otherwise abstain. v30 PB measured that
      running its generator ahead of the existing fallback destroyed two correct
      names while ``structure_wrong`` stayed 0 in both runs, so only a
      row-by-row paired diff caught it. Order is load-bearing.
    * **Atom conservation, asserted here and not delegated.** The generator
      names a RING, so a decorated molecule named by its bare ring parent would
      DROP the decoration -- a wrong constitution, which is strictly worse than
      an abstention. E1 (``verify_certificate``) would also catch it downstream,
      but E1 is measured to be reached on a small fraction of rows, so the
      invariant is stated at the producer.

    ``allow_mancude`` is not a parameter of ``terminal_ring_name``: it delegates
    to ``analyze_cage_universal``/``analyze_spiro_universal`` with
    ``allow_mancude=True`` unconditionally (``terminal_ring.py:645``, ``:650``),
    which is what lets it express fused aromatics. No existing default changes.

    Every guard inside ``terminal_ring`` stays in force -- the reconstruction
    audit, the charged-skeletal-atom (P-73) refusal, ``MAX_CAGE_ATOMS`` and
    ``MAX_CAGE_RINGS``. A refused ring system abstains, and that is correct.
    """
    if not allow_aromatic_general:
        return None
    if mol is None:
        return _refuse("no mol")
    # This tier states its own scope rather than reusing ``_common_refusal``,
    # which was written for the chain/ring producers. Three of that function's
    # four refusals are ALREADY enforced more strictly below or inside
    # ``terminal_ring``, and the fourth is wrong for this tier:
    #
    # * multi-fragment -- impossible here: the ``ring == heavy`` test plus the
    #   connectivity walk below admit only ONE connected ring system, so a second
    #   fragment refuses either as a non-ring heavy atom or as a disjoint system.
    # * net/atom charge -- ``terminal_ring_name`` refuses ANY charged skeletal
    #   ring atom (P-73 is a different naming class), and every heavy atom here
    #   IS a skeletal ring atom, so its guard is the stricter of the two.
    # * isotope -- kept, explicitly: a parent hydride expresses no isotope, so
    #   naming an isotopologue with it would denote a different species.
    # * radical -- DELIBERATELY NOT refused, and this is the whole reason the
    #   tier could not fire without saying so. RDKit reports a skeletal atom in a
    #   non-standard valence state as carrying radical electrons: in
    #   ``C1CC[Al]CC1`` the ring aluminium has two bonds against a default
    #   valence of three, so ``GetNumRadicalElectrons()`` is 1. That is not a
    #   radical, it is exactly the case the lambda convention exists to express
    #   (P-15.4.1.3), and ``build_replacement_prefix`` already emits it --
    #   ``1lambda2-aluminacyclohexane``, which round-trips through OPSIN to the
    #   input. A radical on a skeletal CARBON is a genuine radical and belongs to
    #   P-71 radical nomenclature, which this tier does not build, so it refuses.
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        return _refuse("isotope (a parent hydride expresses none)")
    if any(a.GetNumRadicalElectrons() and a.GetAtomicNum() == 6
           for a in mol.GetAtoms()):
        return _refuse("radical on a skeletal carbon (P-71, not this tier)")

    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    ring_info = mol.GetRingInfo()
    ring = {i for i in heavy if ring_info.NumAtomRings(i) > 0}
    if not ring:
        return _refuse("acyclic (parent-hydride tier names a ring)")
    if ring != heavy:
        return _refuse("decorated ring (parent-hydride tier names the ring "
                       "only; naming it would drop the decoration)")
    # ONE connected ring system: two ring systems joined by nothing have no
    # single von Baeyer / replacement descriptor between them.
    seed = min(ring)
    seen = {seed}
    stack = [seed]
    while stack:
        for nb in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
            j = nb.GetIdx()
            if j in ring and j not in seen:
                seen.add(j)
                stack.append(j)
    if seen != ring:
        return _refuse("two disjoint ring systems (no single descriptor)")
    # A principal characteristic group needs a SUFFIX on the parent, which this
    # tier does not build; the ring is bare by the ``ring != heavy`` test above,
    # so a PG here can only be a skeletal one the suffix machinery owns.
    if getattr(features, 'principal_group', None):
        return _refuse("principal characteristic group needs a parent suffix")

    from ..rules.terminal_ring import terminal_ring_name
    try:
        result = terminal_ring_name(mol, sorted(ring), None)
    except Exception as e:  # noqa: BLE001 - a generator bug must degrade
        logger.info("terminal-ring parent raised %s; refuse", type(e).__name__)
        return None
    if result is None:
        return _refuse("terminal ring namer declined (out of scope or the "
                       "reconstruction audit failed)")

    from ..metrics.candidate_ledger import (
        Scope as _LScope, Stage as _LStage, record_candidate as _lrecord)
    _lrecord(TERMINAL_RING_PARENT_SITE, _LStage.PRODUCED, result.name,
             scope=_LScope.MOLECULE, detail=f"basis:{result.basis}")
    return GeneralEngineResult(
        name=result.name,
        bindings=(TokenBinding(tuple(sorted(ring)), result.name, 'parent'),))


#: Ledger site id for the assembly tier below.
TERMINAL_RING_ASSEMBLY_SITE = "general_engine._name_terminal_ring_assembly"


def _name_terminal_ring_assembly(
    mol, features, allow_aromatic_general: bool = False,
    allow_suffix_free: bool = False,
) -> Optional[GeneralEngineResult]:
    """v30: the LAST-RESORT ASSEMBLY tier -- a nameable von Baeyer ring parent
    PLUS its substituents, with every functional group cited as a PREFIX.

    Why this exists, measured rather than assumed. Of the abstaining dev500
    rows, **0** are bare ring systems (so the parent-hydride tier above can
    never fire on them) but **71** have a nameable ring system AND every
    substituent nameable. Those rows need no new generator: the ring half and
    the decoration half both already work and nothing joins them. What blocks
    them is the SUFFIX logic in the ring producer -- `ester not a clean acyclic
    ring-acid mono-ester` (41), `PG instance not attached to cage` (21),
    `unsupported suffix for pg='ester'` (19), `inline suffix characteristic atom
    off cage` (13). Suppress the principal group and every one of those
    functional groups becomes a detachable prefix, which the existing tail can
    already spell.

    **So this reuses ``_emit_ring_from_analysis`` verbatim rather than composing
    a name.** That is not tidiness, it is the difference between working and
    not: a hand-rolled join of the same parents and prefixes was measured at
    **5 RT_EXACT of 71**, and routing the identical inputs through the existing
    tail gives **44 of 71**. The tail owns the three things a hand join gets
    wrong -- ``_mult_prefix`` (enclosing marks and ``bis``/``tris``, so
    ``15-(2-methylpropyl)`` rather than ``15-2-methylpropyl``), ``_alpha_key``
    (P-14.5.2 order), and ``_stereo_prefix``. The last one is why the gain
    lands in **rt_exact and not merely rt_constitutional**.

    ⚠ **Conformance debt, deliberately incurred and recorded.** A PG-suppressed
    name cites a ketone as ``3-oxo-…`` with NO suffix. P-33 requires the
    principal characteristic group to be the suffix, so these are valid
    descriptions of the right structure that are **not well-formed PINs**. That
    is licit on T4 -- invariant 1: "a table miss must degrade to an uglier name,
    never to a refusal" -- and it is confined to the best-effort branch, but it
    is invisible to round-trip, SELF-01 and E1, which is exactly the spelling
    layer the BB-conformance audit sized. It is v31's axis, not a defect here.

    Order and gating are identical to the parent tier above and load-bearing for
    the same measured reasons: best-effort only, strictly last, and every
    ``terminal_ring`` guard in force -- ``terminal_ring_name`` is called as the
    GATE precisely so its emission-point re-proof of
    ``audit_von_baeyer_descriptor`` (plus the charge, ``MAX_CAGE_ATOMS`` and
    ``MAX_CAGE_RINGS`` refusals) decides whether the ring may be spelled at all.
    """
    if not allow_aromatic_general or mol is None:
        return None
    # T4 ONLY for a suffix-free name. `allow_suffix_free` is the best-effort
    # discriminator (`general_fallback_unverified`), NOT `allow_aromatic_general`
    # -- the latter is True for `complete` as well, and a suffix-free name on
    # `complete` would assert RT-VERIFIED status for an ill-formed name.
    #
    # Scoped to the case that is actually ill-formed: with no principal
    # characteristic group there is no required suffix to omit (P-41 only
    # mandates a suffix when such a group is present), so that name is
    # well-formed and may run wherever this tier runs. Suppressing a PG that
    # DOES exist is the ill-formed construction, and that needs T4.
    _pg = getattr(features, 'principal_group', None)
    if _pg and not allow_suffix_free:
        return _refuse("suffix-free prefix name is T4-only (P-41 requires the "
                       "principal characteristic group as suffix)")
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        return _refuse("isotope (assembly tier expresses none)")
    if any(a.GetNumRadicalElectrons() and a.GetAtomicNum() == 6
           for a in mol.GetAtoms()):
        return _refuse("radical on a skeletal carbon (P-71, not this tier)")
    if len(Chem.GetMolFrags(mol)) > 1:
        return _refuse("multi-fragment (adduct namer's scope)")
    if getattr(features, 'chain_is_parent', False):
        return _refuse("chain parent (chain path owns it)")

    from ..rules.ring_selection import select_principal_ring_system
    from ..rules.terminal_ring import terminal_ring_name
    from ..rules.vonbaeyer_universal import analyze_cage_universal

    ring_systems = list(getattr(features, 'ring_systems', None) or [])
    if not ring_systems:
        return _refuse("no ring system (assembly tier needs a ring parent)")
    senior = select_principal_ring_system(mol, ring_systems)
    if not senior:
        return _refuse("no senior ring system")
    senior = sorted(set(senior))

    # GATE. terminal_ring_name is the authority on whether this ring system may
    # be spelled at all: it kekulizes, refuses a charged skeletal atom (P-73),
    # enforces MAX_CAGE_ATOMS / MAX_CAGE_RINGS, and RE-PROVES the von Baeyer
    # reconstruction audit at the emission point. Its answer is not merely a
    # hint -- if it declines, this tier declines.
    try:
        gate = terminal_ring_name(mol, senior, None)
    except Exception as e:  # noqa: BLE001 - a generator bug must degrade
        logger.info("assembly gate raised %s; refuse", type(e).__name__)
        return None
    if gate is None:
        return _refuse("terminal ring namer declined the parent ring system")

    cage = analyze_cage_universal(mol, cage_atoms=set(senior),
                                  allow_mancude=True)
    if cage is None:
        # rank 1: von Baeyer starts at two rings, so the cage analyzer has no
        # descriptor for a monocycle and the tail cannot spell one. The
        # monocycle parent is name_general_monocycle's job, which has already
        # declined by the time we get here.
        return _refuse("monocycle (no von Baeyer descriptor for one ring)")

    # The invariant that makes the reuse legitimate: the map the parent was
    # AUDITED under and the map the substituent locants are spelled from must be
    # the same map. They are the same object today (terminal_ring_name delegates
    # to this very analyzer for rank >= 2), and if that ever stops being true a
    # substituent locant would be placed against a different numbering than the
    # parent was proven under -- a wrong name, not an ugly one. So it is checked
    # rather than trusted.
    if dict(gate.numbering) != dict(cage.atom_to_locant):
        return _refuse("parent audit numbering != spelling numbering")

    # Every functional group becomes a detachable prefix. Copied, never mutated
    # in place: the caller owns `features` and reuses it.
    import copy as _copy
    prefix_only = _copy.copy(features)
    try:
        prefix_only.principal_group = None
        prefix_only.principal_group_atoms = []
    except (AttributeError, TypeError) as e:
        logger.info("assembly tier: features not clonable (%s); refuse",
                    type(e).__name__)
        return None

    result = _emit_ring_from_analysis(
        mol, prefix_only, cage, allow_aromatic_general, allow_aromatic_general)
    if result is None:
        return None  # the tail already logged its own refusal reason

    from ..metrics.candidate_ledger import (
        Scope as _LScope, Stage as _LStage, record_candidate as _lrecord)
    _lrecord(TERMINAL_RING_ASSEMBLY_SITE, _LStage.PRODUCED, result.name,
             scope=_LScope.MOLECULE,
             detail=f"basis:{gate.basis} suffix_free:{bool(_pg)}")
    if _pg:
        # TAG, do not merely count: v31 must be able to ENUMERATE these rows.
        from ..metrics.provenance import record_suffix_free_prefix_name
        record_suffix_free_prefix_name(True)
    return result


def name_general(
    mol, features, allow_aromatic_general: bool = False,
    allow_suffix_free: bool = False,
) -> Optional[GeneralEngineResult]:
    """v25 engine dispatcher: chain parent -> G1 path, ring parent -> G2 path.

    v26 P0/P1: ``allow_aromatic_general`` widens the ring producer -- it is
    threaded to ``name_general_ring`` -> ``analyze_cage_universal`` (aromatic
    cages) and enables the general lone-monocycle path
    (``name_general_monocycle``). Default False -> byte-identical to pre-P0.

    v26 P5: net charge is lifted ONLY under ``complete`` (``allow_charged`` ==
    ``allow_aromatic_general``); the charge becomes a ``-ylium``/``-ide``/
    ``-uide``/``-ium`` suffix on the numbered parent (fail-closed). Under
    ``valid`` / the PIN default (``allow_aromatic_general`` False) charge is
    still refused -> byte-identical to pre-P5.
    """
    allow_charged = allow_aromatic_general
    ring_atoms = any(a.IsInRing() for a in mol.GetAtoms()) if mol else False
    if not ring_atoms or getattr(features, 'chain_is_parent', False):
        return name_general_chain(mol, features, allow_charged=allow_charged,
                                  allow_mancude=allow_aromatic_general)
    cage_result = name_general_ring(
        mol, features, allow_aromatic_general=allow_aromatic_general,
        allow_charged=allow_charged)
    if cage_result is not None:
        return cage_result
    # v27 P3: spiro producer sibling — the cage engine refuses spiro, so this
    # runs before the lone-monocycle fallback. Inert unless the senior ring
    # system is an analyzable spiro (else fail-closed None -> monocycle path).
    spiro_result = name_general_spiro(
        mol, features, allow_aromatic_general=allow_aromatic_general,
        allow_charged=allow_charged)
    if spiro_result is not None:
        return spiro_result
    mono_result = name_general_monocycle(
        mol, features, allow_aromatic_general=allow_aromatic_general,
        allow_charged=allow_charged)
    if mono_result is not None:
        return mono_result
    # v30: LAST resort, and the two must stay last -- see the docstrings on
    # _name_terminal_ring_parent / _name_terminal_ring_assembly. Both are inert
    # under the PIN default (allow_aromatic_general False) -> byte-identical.
    #
    # Parent-hydride FIRST, then assembly. Not interchangeable: the parent tier
    # only accepts a molecule that IS one bare ring system, and for such a
    # molecule the assembly tier would produce the identical string by a longer
    # route (no substituents to place). Trying assembly first would therefore
    # spend the cage analysis and the whole emission tail to reach the same
    # answer, and would route a bare ring through substituent discovery for no
    # reason.
    parent_result = _name_terminal_ring_parent(
        mol, features, allow_aromatic_general=allow_aromatic_general)
    if parent_result is not None:
        return parent_result
    return _name_terminal_ring_assembly(
        mol, features, allow_aromatic_general=allow_aromatic_general,
        allow_suffix_free=allow_suffix_free)
