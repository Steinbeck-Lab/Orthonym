""" G2: universal von-Baeyer cage analysis.

Names ANY bridged/fused polycyclic cage — including AROMATIC cages — by
kekulizing a canonical copy and expressing every former-aromatic bond as an
explicit ene locant (P-23 unsaturation). The emitted ``...-polyene`` cage
re-parses (OPSIN) to a kekule structure whose canonicalization re-aromatizes
to the SAME molecule, so structural fidelity is preserved; this is the
universal T3 ring fallback for the opt-in general engine ONLY. The default
PIN path's aromatic-cage refusal (polycyclic.py:2789) is deliberately
untouched.

Determinism: kekulization is atom-order dependent, so we ALWAYS canonical-
reparse first and work in canonical indices; results map back to original
indices via GetSubstructMatch.
"""
from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from rdkit import Chem

logger = logging.getLogger(__name__)


class _Malformed(Exception):
    """Internal: a descriptor whose numbered path is degenerate (an edge from a
    locant to itself). Raised inside the reconstructor, never propagated."""

#: Implementation ceilings on the cage this module will analyse. The Blue Book
#: sets NO upper size limit on von Baeyer nomenclature (P-23 is construction
#: rules only), so both numbers are ours, not nomenclature's. They guard only
#: this module -- the default PIN path (``polycyclic.name_polycyclic_complete``)
#: caps nothing above ``ring_count < 2``.
#:
# : T3b measured what raising them buys, over the 1203 ring molecules
#: of ``, and the answer is NOTHING: with both caps
#: set to 200 the emitted name is byte-identical for every molecule the caps
#: touch. So the VALUES stay and the fix went to what was being COUNTED --
#: ``MAX_CAGE_RINGS`` was compared against RDKit's symmetrized ring-set
#: cardinality instead of the P-23.1.9 ring number (see
#: ``polycyclic.von_baeyer_ring_count``), which made this cap refuse *heptacyclo*
#: cages for being "more than 8 rings".
#:
#: Do not raise ``MAX_CAGE_RINGS`` without first fixing P-23.2.4 main-bridge
#: selection: ``_find_main_ring`` only ever offers a 0- or 1-atom main bridge, so
#: 7 of 19 Blue Book von Baeyer PIN descriptors come back with a non-preferred
#: main bridge (locked in ``tests/unit/rules/test_v29_p2_vb_ring_count.py``).
#: Raising the cap would broaden that class, not add PIN coverage.
MAX_CAGE_ATOMS = 40
MAX_CAGE_RINGS = 8


@dataclass(frozen=True)
class RingAnalysis:
    """The ONE field contract ``_emit_ring_from_analysis`` consumes.

    Why this base exists
    -----------------------------------------------
    ``UniversalCage`` (von Baeyer) and ``SpiroSystem`` (spiro) are two ring
    analysis forms feeding ONE shared emission tail, so the tail is written
    against a field shape rather than a class. That shape used to be stated
    TWICE -- ``SpiroSystem`` re-declared ``UniversalCage``'s fields by hand --
    and the two copies drifted every time either producer was touched:

    * ``: the skeletal-replacement TOTALITY gate was added to the
      spiro analyzer only, so a mercury von-Baeyer ring shipped as
      ``bicyclo[3.3.0]octane`` -- a hydrocarbon name;
    * ``: ``hetero_per_atom`` was added to the cage only, so the
      spiro sibling kept the ``UNBOUND_MORPHEME`` finding that commit existed
      to remove.

    Opposite directions, same cause: a contract maintained in parallel. It is
    declared here ONCE. The subclasses add NO fields -- they exist for their
    names (repr, ``isinstance``, per-form docs), not to extend the shape -- so
    a field added to the emission contract reaches BOTH producers by
    construction. Declaring a field is not populating it, so the other half of
    the guarantee is a test asserting each form actually fills it
    (``tests/unit/validation/test_v29_p2_ring_parent_bindings.py``).
    """

    descriptor: str                 # e.g. "bicyclo[2.2.1]" / "spiro[4.5]"
    total_atoms: int
    hetero_prefix: str              # "" | "7-oxa" | "2,5-diaza"...
    unsaturation: dict              # {'double_bonds': [...], 'triple_bonds': [...]}
    cage_atoms: Tuple[int, ...]     # ORIGINAL (input-mol) indices
    atom_to_locant: Dict[int, int]  # ORIGINAL idx -> ring locant
    canon_match: Tuple[int, ...]    # index map back to ORIGINAL indices
    is_mancude: bool = False        # aromatic/mancude ring system; True only
                                    # under the opt-in complete tier
                                    # (allow_mancude), where it emits as a
                                    # kekulized polyene
    # T5: which ORIGINAL atom each morpheme of ``hetero_prefix``
    # spells -- (orig idx, morpheme), e.g. ((6, 'oxa')). Carried so a consumer
    # can bind one token per replacement morpheme instead of letting the parent
    # token over-claim the heteroatoms it does not spell. MUST come from the
    # same builder that spelled ``hetero_prefix`` (never a second, differently
    # -spelling one) so the decomposition agrees with the string by
    # construction. Defaults to so a hand-built or foreign-shaped analysis
    # stays valid; means "not reported" and consumers must fall back to
    # whole-parent attribution, NOT assume the ring is all-carbon.
    hetero_per_atom: Tuple[Tuple[int, str], ...] = ()


@dataclass(frozen=True)
class UniversalCage(RingAnalysis):
    """von Baeyer polycyclic analysis. Adds no field to ``RingAnalysis``.

    ``canon_match`` here is the canonical-copy index map: canon idx -> orig idx.
    """


# : ``bicyclo[`` / ``tricyclo[`` /... head, capturing the bracket body.
_VB_HEAD_RE = re.compile(r'^[a-z]+cyclo\[(.+)\]$')
#: one secondary-bridge term: ``2^3,7`` (PIN typography) or ``2(3,7)`` (the
#: older parenthesis form ``_build_descriptor`` also emits; OPSIN parses both).
_VB_SEC_RE = re.compile(r'^(\d+)(?:\^(\d+),(\d+)|\((\d+),(\d+)\))$')

#: Cost bound on the tie-break search in ``reconstruct_von_baeyer_skeleton``
#: (see the ``secondary_order`` note there). 6! = 720 orderings is the ceiling.
_MAX_SECONDARY_PERMUTED = 6


def parse_von_baeyer_descriptor(descriptor: str):
    """``'tricyclo[3.3.1.1^3,7]'`` -> ``([3, 3, 1], [(1, 3, 7)])``; None if
    the string is not a well-formed von Baeyer descriptor.

    The three leading integers are the bicyclic system (two main-ring branches
    then the main bridge, P-23.2.2 "cited in descending numerical order"); each
    remaining term is ``length^low,high`` for one secondary bridge.
    """
    if not descriptor:
        return None
    m = _VB_HEAD_RE.match(descriptor.strip())
    if not m:
        return None
    toks = m.group(1).split('.')
    if len(toks) < 3:
        return None
    try:
        primary = [int(t) for t in toks[:3]]
    except ValueError:
        return None
    secondary = []
    for tok in toks[3:]:
        sm = _VB_SEC_RE.match(tok)
        if not sm:
            return None
        d = int(sm.group(1))
        lo_s, hi_s = (sm.group(2), sm.group(3)) if sm.group(2) else (
            sm.group(4), sm.group(5))
        lo, hi = int(lo_s), int(hi_s)
        secondary.append((d, min(lo, hi), max(lo, hi)))
    return primary, secondary


def reconstruct_von_baeyer_skeleton(descriptor: str, secondary_order=None):
    """Rebuild the ring skeleton a von Baeyer descriptor STRING denotes.

    Returns ``(total_atoms, frozenset{(low_locant, high_locant)})`` -- the exact
    bond set the emitted name encodes, in locant space -- or ``None`` when the
    string is malformed.

    The numbering is not a convention of ours; it is read straight off the Blue
    Book, which is what makes this a proof rather than a heuristic:

    * **P-23.2.3 "Numbering bicyclic alicyclic hydrocarbons"**
      (``BlueBookV2/BlueBookV2.md:9589``): *"The bicyclic ring system is numbered
      starting with one of the bridgeheads and proceeding first along the longer
      segment of the main ring to the second bridgehead, then back to the first
      bridgehead along the unnumbered segment of the main ring. Numbering is
      completed by numbering the main bridge beginning with the atom next to the
      first bridgehead."*
    * **P-23.2.5.2 "Numbering the secondary bridge"** (``:9623``): *"After the
      main ring and main bridge have been numbered, the independent secondary
      bridge is numbered continuing from the higher numbered bridgehead of the
      main ring."*
    * **P-23.2.6.3 "Numbering of secondary bridges"** (``:9711``): *"the
      numbering continues from the highest number of the main ring and main
      bridge. Each secondary bridge is numbered in turn starting with the
      independent secondary bridge linked to the highest numbered bridgehead
      atom of the main ring, then the independent secondary bridge linked to the
      next highest bridgehead atom, and so on. Each atom of a secondary bridge
      is numbered starting with the atom next to the higher numbered
      bridgehead."*

    ``secondary_order`` overrides the order in which the secondary bridges
    consume locants. P-23.2.6.3 fixes that order by descending attachment
    bridgehead, but leaves ties to P-23.2.6.4; the caller uses this to try the
    remaining orderings rather than false-reject a cage whose only deviation is
    a tie-break, which is a *preference* question and not a legality one.
    """
    parsed = parse_von_baeyer_descriptor(descriptor)
    if parsed is None:
        return None
    (a, b, c), secondary = parsed
    if a < 0 or b < 0 or c < 0:
        return None
    # P-23.2.2: the bicyclic numbers are cited in descending order, and the
    # numbering rule above ("first along the LONGER segment") depends on it.
    if a < b or b < c:
        return None
    n_main = a + b + c + 2
    bh1, bh2 = 1, a + 2
    edges = set()

    def walk(path):
        for x, y in zip(path, path[1:]):
            if x == y:
                raise _Malformed()
            edges.add((min(x, y), max(x, y)))

    try:
        # longer main-ring segment: 1 -> 2..a+1 -> a+2
        walk([bh1] + list(range(2, a + 2)) + [bh2])
        # back along the unnumbered segment: a+2 -> a+3..a+b+2 -> 1
        walk([bh2] + list(range(a + 3, a + b + 3)) + [bh1])
        # main bridge, "beginning with the atom next to the first bridgehead"
        walk([bh1] + list(range(a + b + 3, a + b + c + 3)) + [bh2])
    except _Malformed:
        return None

    if secondary_order is None:
        # independent bridges before dependent (P-23.2.6.1.3); within each,
        # the one linked to the highest numbered bridgehead first.
        secondary_order = sorted(
            range(len(secondary)),
            key=lambda i: (1 if secondary[i][2] > n_main else 0,
                           -secondary[i][2], -secondary[i][1]))
    elif sorted(secondary_order) != list(range(len(secondary))):
        return None

    nxt = n_main + 1
    try:
        for i in secondary_order:
            d, lo, hi = secondary[i]
            segment = list(range(nxt, nxt + d))
            nxt += d
            # "numbered starting with the atom next to the higher numbered
            # bridgehead" -- hi first, then the new atoms, ending at lo.
            walk([hi] + segment + [lo])
    except _Malformed:
        return None
    total = nxt - 1
    for d, lo, hi in secondary:
        if lo < 1 or hi > total:
            return None
    return total, frozenset(edges)


def _cage_edges_in_locant_space(mol, cage_atoms, numbering: Dict[int, int]):
    """The molecule's OWN cage bond set, relabelled through ``numbering``.
    None when a cage atom is unnumbered or two bonded atoms share a locant."""
    cage = set(cage_atoms)
    out = set()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in cage and j in cage:
            li, lj = numbering.get(i), numbering.get(j)
            if li is None or lj is None or li == lj:
                return None
            out.add((min(li, lj), max(li, lj)))
    return frozenset(out)


def audit_von_baeyer_descriptor(
    mol, cage_atoms, numbering: Dict[int, int], descriptor: str,
) -> bool:
    """Java-free structural correctness floor for a von-Baeyer descriptor.

    Rebuild the ring skeleton the EMITTED DESCRIPTOR STRING denotes (Blue Book
    numbering rules, ``reconstruct_von_baeyer_skeleton``) and require SET
    EQUALITY with the input's actual cage bond set in locant space. Any mismatch
    -- a bridge the descriptor fails to cite, an atom count the brackets do not
    account for, or a numbering inconsistent with the string -- returns
    ``False`` so the caller fails closed.

    Why the string and not our own bridge list
    ------------------------------------------
    This audit used to reconstruct from ``PolycyclicDescriptor.bridge_info_list``
    -- internal bookkeeping -- rather than from the descriptor that actually gets
    spelled. Measured over 8,201 enumerated cages (N<=12) and 1,623 corpus cages,
    that was wrong in both directions at once:

    * **False rejections.** ``BridgeInfo.atoms`` for a secondary bridge is stored
      in the opposite order to the bond path (numbering runs from the *higher*
      numbered bridgehead, P-23.2.6.3), so the walk crossed a non-bonded pair and
      the audit refused **556/8,201** enumerated and **14/1,623** corpus cages
      whose emitted name was perfectly correct.
    * **False acceptances.** The bridge list can agree with itself while the
      spelled string does not describe the cage at all -- 6 corpus cages passed
      the old audit with a descriptor denoting a different skeleton.

    Set-equality against the string catches both, and is what the name is
    actually judged on. It is deliberately independent of OPSIN: both downstream
    gates (the validity gate and SELF-01) are documented FAIL-OPEN when Java is
    unavailable (``namer.py:588``, ``:608``), and with them off **30** enumerated
    cages ship a descriptor whose brackets account for fewer atoms than its own
    stem counts (e.g. ``tetracyclo[5.1.1.2^3,6]dodecane`` -- 11 bracketed atoms,
    ``dodecane`` = 12). This floor refuses those without asking Java.

    This proves LEGALITY (the name denotes this molecule under this numbering),
    never PREFERENCE (that it is the PIN among legal alternatives).
    """
    rebuilt = reconstruct_von_baeyer_skeleton(descriptor)
    if rebuilt is None:
        return False
    cage = set(cage_atoms)
    total, asserted = rebuilt
    if total != len(cage):
        return False
    # P-23.1.9, under P-23.1 "DEFINITIONS AND TERMINOLOGY"
    # (``BlueBookV2/BlueBookV2.md:9558``): *"A 'polycyclic system' contains a
    # number of rings equal to the minimum number of scissions required to
    # convert the system into an acyclic skeleton. The number of rings is
    # indicated by the nondetachable prefix 'bicyclo' (not dicyclo), 'tricyclo',
    # 'tetracyclo', etc."* The word is part of the name, so a descriptor whose
    # brackets are right but whose ring-count word is wrong still denotes the
    # wrong system -- ``tetracyclo[3.3.1.1^3,7]`` for adamantane passed every
    # other clause of this audit.
    from .polycyclic import cyclo_ring_count_word
    circuit_rank = len(asserted) - total + 1
    head = descriptor.strip().split('[', 1)[0]
    if head != cyclo_ring_count_word(circuit_rank):
        return False
    # the numbering must be a bijection of the cage onto 1..N; anything else
    # cannot be read against the descriptor's locants at all.
    locants = [numbering.get(i) for i in cage]
    if any(l is None for l in locants):
        return False
    if sorted(locants) != list(range(1, total + 1)):
        return False
    actual = _cage_edges_in_locant_space(mol, cage, numbering)
    if actual is None:
        return False
    if actual == asserted:
        return True
    # P-23.2.6.4 leaves a tie-break when two secondary bridges attach to the
    # same bridgehead. A cage that matches under some other ordering IS legally
    # described by this string; only the numbering preference differs, so
    # refusing it would lose a correct name (the failure mode this audit had).
    parsed = parse_von_baeyer_descriptor(descriptor)
    if parsed is None:
        return False
    n_secondary = len(parsed[1])
    if 2 <= n_secondary <= _MAX_SECONDARY_PERMUTED:
        for order in itertools.permutations(range(n_secondary)):
            alt = reconstruct_von_baeyer_skeleton(descriptor,
                                                  secondary_order=list(order))
            if alt is not None and alt[0] == total and alt[1] == actual:
                return True
    return False


def analyze_cage_universal(
    mol, cage_atoms=None, allow_mancude: bool = False,
    spiro_atom: Optional[int] = None,
) -> Optional[UniversalCage]:
    """Deterministic universal cage analysis; None on any refusal.

    ``spiro_atom`` (default None, original-mol atom index) is the spiro-junction
    atom when this cage is a COMPONENT of a spiro ring system. When passed, the
    von-Baeyer numbering selection gives that atom the lowest locant (P-24.5.2),
    above the heteroatom criteria; it is mapped into the canonical index space
    used internally before being threaded to ``VonBaeyerAnalyzer.analyze``. Every
    whole-molecule caller leaves it None, and the result is byte-identical.

     P2: when ``allow_mancude`` is True the aromatic/mancude-cage refusal
    below is LIFTED -- the cage is kekulized (already done above) and every
    former-aromatic bond is emitted as an explicit von-Baeyer polyene ene
    locant (P-23 unsaturation). When False (the default / PIN path) the
    refusal fires exactly as before, so that path is byte-identical.

    Every returned descriptor is put through ``audit_von_baeyer_descriptor``
    (a Java-free skeleton edge-audit): the descriptor+numbering must assert
    exactly the molecular-graph ring/bridge bond set or the cage is discarded
    (fail-closed). This floor holds for the saturated/valid tier too and is
    what makes the mancude polyene emission trustworthy independent of OPSIN.
    """
    from .polycyclic import (
        VonBaeyerAnalyzer, _get_largest_connected_ring_component,
        von_baeyer_ring_count,
    )
    from .ring_replacement import build_replacement_prefix
    from .ring_unsaturation import render_ring_unsaturation

    if mol is None:
        return None

    # The self-consistency check below round-trips `mol` through a canonical
    # SMILES and re-parses it, purely to obtain a canonical atom-order
    # mapping (`orig_to_canon`). Try it UNCHANGED first -- this is the
    # original algorithm, byte-identical for every existing caller (all of
    # which pass a mol whose own aromaticity/Kekulization state already
    # round-trips cleanly through this).
    mol_for_match = mol
    canon_smiles = Chem.MolToSmiles(mol_for_match, canonical=True)
    canon = Chem.MolFromSmiles(canon_smiles)
    match = mol_for_match.GetSubstructMatch(canon) if canon is not None else ()

    if canon is None or len(match) != mol.GetNumAtoms():
        # WS-NOABSTAIN fallback (only reached when the original check above
        # already failed, so this can only ADD coverage, never change an
        # already-working case). If the caller passed an already-Kekulized
        # mol (aromatic flags cleared -- e.g. `universal_substituent.py`'s
        # `_build_ctx`, which Kekulizes its whole working context up front
        # for its own bond-order bookkeeping), the freshly reparsed `canon`
        # above comes back RE-AROMATIZED while `mol` still has explicit
        # Kekule bonds for the exact same graph -- `GetSubstructMatch` is
        # bond-type-strict by default, so it reports NO match at all for a
        # perfectly legal cage (naphthalene, any mancude ring). Retry by
        # re-aromatizing a COPY of `mol` so the match compares aromatic-to-
        # aromatic representations. `Chem.SetAromaticity` on an
        # already-aromatic mol is a no-op, so this retry is only ever
        # useful for the Kekulized-caller case the first attempt above
        # cannot handle.
        mol_for_match = Chem.Mol(mol)
        try:
            Chem.SetAromaticity(mol_for_match)
            # A prior ``Chem.Kekulize(..., clearAromaticFlags=True)`` demotes
            # a bracket-explicit H -- e.g. the pyrrole-type ``[nH]`` in an
            # indole/purine/imidazole-fused system -- from EXPLICIT to
            # IMPLICIT bookkeeping (``GetTotalNumHs()`` stays correct
            # throughout, but ``GetNumExplicitHs()`` drops to 0).
            # ``SetAromaticity`` above restores the aromatic FLAG but not
            # that explicit-H bookkeeping, and ``MolToSmiles`` decides
            # whether to write ``[nH]`` vs. bare ``n`` from the
            # explicit/no-implicit state, not from ``GetTotalNumHs()`` alone
            # -- so the canonical SMILES below would silently lose the ring
            # N-H, and re-parsing it would then fail to kekulize entirely
            # (measured on plain pyrrole and on this task's protonated-
            # purine witness). Re-pin every atom's already-correct total H
            # count as EXPLICIT before serializing, so this retry is
            # lossless regardless of what the caller's mol's Kekulize state
            # did to the explicit/implicit split.
            for a in mol_for_match.GetAtoms():
                total_h = a.GetTotalNumHs()
                a.SetNoImplicit(True)
                a.SetNumExplicitHs(total_h)
        except Exception:
            return None

        canon_smiles = Chem.MolToSmiles(mol_for_match, canonical=True)
        canon = Chem.MolFromSmiles(canon_smiles)
        if canon is None:
            return None
        match = mol_for_match.GetSubstructMatch(canon)
        if len(match) != mol.GetNumAtoms():
            logger.info("vonbaeyer_universal: no whole-mol canon match; refuse")
            return None

    orig_to_canon = {orig: c for c, orig in enumerate(match)}

    kek = Chem.RWMol(canon)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception as e:  # kekulization failure -> fail closed
        logger.info("vonbaeyer_universal: kekulize failed: %s", e)
        return None

    ri = kek.GetRingInfo()
    ring_atoms = set()
    for ring in ri.AtomRings():
        ring_atoms.update(ring)
    if not ring_atoms:
        return None

    if cage_atoms is not None:
        cage_canon = {orig_to_canon[i] for i in cage_atoms}
        if not cage_canon <= ring_atoms:
            return None
    else:
        cage_canon = _get_largest_connected_ring_component(kek, ring_atoms)

    if len(cage_canon) > MAX_CAGE_ATOMS:
        return None
    # T3b: count rings the way P-23.1.9 defines them (minimum
    # scissions to reach an acyclic skeleton = circuit rank), NOT the count of
    # RDKit's symmetrized ring set. The symmetrized count over-counts symmetric
    # cages (adamantane 4 vs 3, cubane 6 vs 5), so this cap used to refuse
    # heptacyclo cages as "> 8 rings". See von_baeyer_ring_count.
    n_rings = von_baeyer_ring_count(kek, cage_canon)
    if n_rings is None or n_rings < 2 or n_rings > MAX_CAGE_RINGS:
        return None

    analyzer = VonBaeyerAnalyzer()
    if len(analyzer._find_all_bridgeheads(kek, cage_canon)) < 2:
        return None  # spiro / degenerate: out of G2 scope
    # Map the spiro-junction atom into the canonical index space the analyzer
    # works in (P-24.5.2 low-locant preference); None -> spiro-blind, byte-
    # identical to every whole-molecule caller.
    spiro_canon = (
        orig_to_canon.get(spiro_atom) if spiro_atom is not None else None
    )
    try:
        desc = analyzer.analyze(kek, cage_canon, spiro_atom=spiro_canon)
    except Exception as e:
        logger.info("vonbaeyer_universal: analyze failed: %s", e)
        return None
    if not desc or not desc.numbering:
        return None
    # AUDIT: the numbering must cover the exact cage (no silent shrink).
    if set(desc.numbering.keys()) != set(cage_canon):
        logger.info("vonbaeyer_universal: numbering!=cage; refuse")
        return None
    # AUDIT: descriptor+numbering must assert the exact molecular ring/bridge
    # bond set (Java-free skeleton floor; SELF-01 fails open without Java).
    if not audit_von_baeyer_descriptor(
            kek, cage_canon, desc.numbering, desc.descriptor_string):
        logger.info("vonbaeyer_universal: descriptor edge-audit failed; refuse")
        return None

    # T2a: skeletal-replacement TOTALITY. The prefix builder can only
    # spell the elements in its table; every other skeletal ring atom used to be
    # skipped while the stem kept counting it, so ``C1CC2CC[Hg]C2C1`` named as
    # ``bicyclo[3.3.0]octane`` — a hydrocarbon name for a mercury ring. SELF-01
    # fails OPEN with no OPSIN jar (a supported mode), so nothing downstream
    # caught it. Refuse whenever a skeletal atom is left unexpressed: never a
    # ring stem that counts an atom no morpheme in the name spells.
    replacement = build_replacement_prefix(kek, desc.numbering, cage_canon)
    if replacement.unexpressed:
        logger.info(
            "vonbaeyer_universal: %d skeletal atom(s) unexpressible by the "
            "replacement-prefix table; refuse", len(replacement.unexpressed))
        return None
    hetero = replacement.prefix

    # G5-A / T1: cite each ring double bond with the von-Baeyer
    # COMPOUND locant n(m) when its two atoms are NOT consecutively numbered (a
    # fusion/bridge ene, e.g. octalin 1(6)); plain n when m == n+1. The bare
    # min(n,m) model mislabels non-consecutive enes (and, adjacent to an oxo,
    # fabricates the 5-bond-carbon valence clash). ``double_bond_pairs`` (raw
    # (low,high) VB locants) is retained for the engine's valence guard.
    #
    # T1 moved this to ``rules/ring_unsaturation.py`` so it covers BOTH bond
    # orders. Before T1 the triple-bond locants came straight from
    # ``get_polycyclic_unsaturation``'s ``min(loc1, loc2)`` — a bare locant with
    # no composite form and NO GUARD, i.e. a wrong-bond citation waiting for a
    # non-consecutively-numbered yne. The primitive fails closed on that state
    # instead (it is unreachable for standard bonding numbers, so reaching it
    # means the NUMBERING is wrong). Scoped to THIS cage payload — the default
    # polycyclic path is untouched (it uses ints directly).
    ring_unsat = render_ring_unsaturation(kek, desc.numbering)
    if ring_unsat is None:
        logger.info("vonbaeyer_universal: ring unsaturation not citable; refuse")
        return None
    unsat = {
        'double_bonds': list(ring_unsat.double_locants),
        'double_bond_pairs': list(ring_unsat.double_pairs),
        'triple_bonds': list(ring_unsat.triple_locants),
    }

    cage_orig = tuple(sorted(match[c] for c in cage_canon))
    atom_to_locant = {match[c]: loc for c, loc in desc.numbering.items()}

    # G5-A / P2: a cage carrying an AROMATIC ring atom (original-mol
    # perception) is mancude. On the DEFAULT / PIN path (allow_mancude=False)
    # its PIN is a fused/retained parent (P-25) + added/indicated H (P-58.2.2),
    # NOT a von-Baeyer polyene, so we still refuse (fail-closed, byte-identical
    # to pre-P2). Under the opt-in complete tier (allow_mancude=True) the cage
    # was kekulized above and every former-aromatic bond is already captured in
    # ``unsat`` as an explicit ene locant (P-23) -- express it as the kekulized
    # von-Baeyer polyene instead of refusing. Isolated ring double bonds
    # (norbornadiene) and saturated hetero cages (quinuclidine) are NOT aromatic
    # -> named on both paths. The oxo/ene valence guard in the engine
    # (general_engine.name_general_ring, suffix_core=='one') still fires.
    # WS-NOABSTAIN: read aromaticity off `mol_for_match` (the re-aromatized
    # copy above), never the original `mol` -- an already-Kekulized caller's
    # `mol` has every `GetIsAromatic()` flag cleared, which used to make this
    # ALWAYS read False (never mancude) regardless of true aromaticity. That
    # was latent/inert only because the match check above voided the whole
    # call first for every genuinely aromatic cage; now that the match is
    # fixed, this must be fixed in the same commit or a real mancude cage
    # would silently fall through the default (`allow_mancude=False`) PIN
    # path as if it were saturated.
    is_mancude = any(mol_for_match.GetAtomWithIdx(i).GetIsAromatic() for i in cage_orig)
    if is_mancude and not allow_mancude:
        logger.info("vonbaeyer_universal: mancude/aromatic cage -> refuse (default path)")
        return None

    return UniversalCage(
        descriptor=desc.descriptor_string,
        total_atoms=desc.total_atoms,
        hetero_prefix=hetero or "",
        unsaturation=unsat,
        cage_atoms=cage_orig,
        atom_to_locant=atom_to_locant,
        canon_match=tuple(match),
        is_mancude=is_mancude,
        # ``build_replacement_prefix`` reports per_atom in the CANONICAL indices
        # it was handed (``kek``/``cage_canon``); ``match`` maps those back to
        # original indices, the space every other field on this dataclass uses.
        # These morphemes come from the SAME builder that produced
        # ``hetero_prefix`` above, so the decomposition is consistent with the
        # string by construction -- not re-derived and hoped to agree.
        hetero_per_atom=tuple(sorted(
            (match[c], morpheme) for c, morpheme in replacement.per_atom)),
    )


# =====================================================================
# P3 (P-24.2): general SPIRO analysis — a sibling to the von-Baeyer
# cage engine above. ``analyze_cage_universal`` deliberately refuses spiro
# (``<2 bridgeheads`` at:156); this analyzer names monospiro / linear &
# branched polyspiro / heterocyclic spiro ring SYSTEMS, returning the SAME
# dataclass shape (``SpiroSystem`` mirrors ``UniversalCage``) so the general
# engine's suffix+substituent+stereo emission tail consumes it unchanged.
# It composes the audited building blocks in ``rules/spiro.py`` (do not
# re-derive) and runs every result through ``audit_spiro_descriptor`` — the
# mandatory Java-free structural floor (SELF-01 fails OPEN without Java).
# =====================================================================


@dataclass(frozen=True)
class SpiroSystem(RingAnalysis):
    """Spiro ring-system analysis. Adds no field to ``RingAnalysis``.

    The field shape is INHERITED, not re-declared: this class used to restate
    ``UniversalCage``'s fields by hand and the two lists drifted twice (see
    ``RingAnalysis``). ``descriptor`` is ``"spiro[4.5]"`` / ``"dispiro[3.2.3.2]"``
    and ``canon_match`` is the sorted original cage-atom tuple (the analysis
    works on a ring-only submol, so there is no separate canonical copy to map).
    """


def audit_spiro_descriptor(
    mol, cage_atoms, numbering: Dict[int, int], spiro_atoms, descriptor: str,
) -> bool:
    """Java-free structural correctness floor for a spiro descriptor+numbering.

    Mirrors ``audit_von_baeyer_descriptor``'s set-equality contract, adapted to
    spiro topology. Fail-closed (``False``) on any of:
      (bijection) numbering is not a 1-1 map of the cage atoms onto {1..N}
                   (rejects a numbering that maps two atoms to the same locant);
      (coverage) the SSSR rings contained in the cage do not union to exactly
                   the cage atom set (a ring atom silently dropped);
      (pure-spiro) two cage rings share >1 atom (fused/bridged mis-routed here),
                   a shared atom is not a declared spiro atom, a spiro atom is
                   not in exactly 2 cage rings, a non-spiro atom is not in
                   exactly 1, or n_rings != n_spiro + 1;
      (simple-cycle) any cage atom has != 2 ring-neighbours within one of its
                   rings (a real bond the descriptor's cycle would assert is
                   missing, or a bridge edge is present);
      (arithmetic) the descriptor's segment integers + n_spiro != N (a wrong
                   ring-size descriptor for the audited skeleton).
    """
    import re
    cage = set(cage_atoms)
    n = len(cage)
    if n == 0:
        return False
    # (bijection)
    if set(numbering.keys()) != cage:
        return False
    if sorted(numbering.values()) != list(range(1, n + 1)):
        return False
    ri = mol.GetRingInfo()
    cage_rings = [set(r) for r in ri.AtomRings() if set(r) <= cage]
    if not cage_rings:
        return False
    # (coverage)
    union: set = set()
    for r in cage_rings:
        union |= r
    if union != cage:
        return False
    spiro_set = set(spiro_atoms)
    # (pure-spiro) pairwise ring intersections are single spiro atoms
    for i in range(len(cage_rings)):
        for j in range(i + 1, len(cage_rings)):
            inter = cage_rings[i] & cage_rings[j]
            if len(inter) > 1:
                return False
            if len(inter) == 1 and next(iter(inter)) not in spiro_set:
                return False
    ring_count: Dict[int, int] = {a: 0 for a in cage}
    for r in cage_rings:
        for a in r:
            ring_count[a] += 1
    for a in cage:
        if a in spiro_set:
            if ring_count[a] != 2:
                return False
        elif ring_count[a] != 1:
            return False
    if len(cage_rings) != len(spiro_set) + 1:
        return False
    # (simple-cycle) every atom has exactly 2 ring-neighbours inside each of
    # its rings — the cycle edges the descriptor asserts are all real bonds.
    for r in cage_rings:
        for a in r:
            nb = [x.GetIdx() for x in mol.GetAtomWithIdx(a).GetNeighbors()
                  if x.GetIdx() in r]
            if len(nb) != 2:
                return False
    # (arithmetic) descriptor segment sum + n_spiro == N
    try:
        body = descriptor[descriptor.index('[') + 1:descriptor.index(']')]
    except ValueError:
        return False
    segs = []
    for tok in body.split('.'):
        m = re.match(r'^(\d+)', tok)
        if not m:
            return False
        segs.append(int(m.group(1)))
    if sum(segs) + len(spiro_set) != n:
        return False
    return True


def _extract_spiro_submol(mol, cage_set):
    """Build a standalone submol of exactly ``cage_set`` (ring atoms), bonds
    among them preserved, broken parent/substituent bonds left as implicit H.
    Returns ``(submol, submol_to_mol)`` where ``submol_to_mol[k]`` is the
    original index of submol atom ``k`` — or ``(None, None)`` on failure.
    Atoms are added in sorted-index order for determinism."""
    order = sorted(cage_set)
    mol_to_sub = {m: k for k, m in enumerate(order)}
    rw = Chem.RWMol()
    for m in order:
        rw.AddAtom(Chem.Atom(mol.GetAtomWithIdx(m).GetAtomicNum()))
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in mol_to_sub and j in mol_to_sub:
            rw.AddBond(mol_to_sub[i], mol_to_sub[j], b.GetBondType())
    sub = rw.GetMol()
    try:
        Chem.SanitizeMol(sub)
    except Exception:
        return None, None
    return sub, order


def analyze_spiro_universal(
    mol, cage_atoms=None, allow_mancude: bool = False, free_valence_atoms=None,
) -> Optional[SpiroSystem]:
    """Deterministic universal spiro analysis; None on any refusal.

    ``cage_atoms`` selects the spiro ring system inside ``mol`` (defaults to all
    ring atoms). ``free_valence_atoms`` (original indices) biases the monospiro
    numbering to give the free valence the lowest locant (P-29.3, substituent
    use). ``allow_mancude`` lifts the aromatic-spiro refusal, emitting the
    kekulized ene-locant (P-23) form; when False an aromatic spiro fails closed
    (deferring to the retained-ring-name PIN path). Every result is put through
    ``audit_spiro_descriptor`` (fail-closed on any structural mismatch).
    """
    from ..perception.rings import get_spiro_atoms
    from .spiro import (
        generate_spiro_descriptor, get_spiro_numbering,
        _get_polyspiro_numbering, _build_hetero_prefix,
    )
    from .polycyclic import (  # noqa: F401 (_get_alkane_name = parity import)
        _get_alkane_name, von_baeyer_ring_count,
    )

    if mol is None:
        return None

    ring_atoms_mol = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    if not ring_atoms_mol:
        return None
    cage_set = set(cage_atoms) if cage_atoms is not None else set(ring_atoms_mol)
    if not cage_set <= ring_atoms_mol:
        return None
    if len(cage_set) > MAX_CAGE_ATOMS:
        return None

    # cage-only submol (reuses the spiro.py blocks unchanged, isolated from any
    # substituent rings on the parent).
    sub, sub_to_mol = _extract_spiro_submol(mol, cage_set)
    if sub is None:
        return None
    mol_to_sub = {m: k for k, m in enumerate(sub_to_mol)}

    ri = sub.GetRingInfo()
    n_rings = ri.NumRings()
    # T3b: the CAP is on the P-23.1.9 ring count (circuit rank), not
    # on the symmetrized ring-set cardinality -- same fix as the cage sibling.
    # ``n_rings`` itself is left as the symmetrized count because the pure-spiro
    # identity check below (n_rings == n_spiro + 1) is written against that
    # enumeration; for a pure spiro system the two agree anyway (no bridge means
    # no symmetry-degenerate smallest rings), so this only stops a bridged
    # candidate from being rejected for the wrong reason before that check runs.
    vb_rings = von_baeyer_ring_count(sub, set(range(sub.GetNumAtoms())))
    if vb_rings is None or vb_rings < 2 or vb_rings > MAX_CAGE_RINGS:
        return None
    spiro_sub = get_spiro_atoms(sub)
    if not spiro_sub:
        return None
    # pure spiro: n_spiro + 1 rings, and every ring-ring share is a spiro atom
    if n_rings != len(spiro_sub) + 1:
        return None

    descriptor = generate_spiro_descriptor(sub)
    if not descriptor:
        return None

    fv_sub = None
    if free_valence_atoms:
        fv_sub = {mol_to_sub[a] for a in free_valence_atoms if a in mol_to_sub}

    if len(spiro_sub) == 1:
        center = next(iter(spiro_sub))
        numbering = get_spiro_numbering(
            sub, center, suffix_ring_atoms=fv_sub)
    else:
        numbering = _get_polyspiro_numbering(
            sub, spiro_sub, suffix_ring_atoms=fv_sub)
    # numbering must cover exactly the (ring-only) submol atoms
    if not numbering or set(numbering.keys()) != set(range(sub.GetNumAtoms())):
        return None

    if not audit_spiro_descriptor(
            sub, set(range(sub.GetNumAtoms())), numbering, spiro_sub, descriptor):
        logger.info("analyze_spiro_universal: descriptor audit failed; refuse")
        return None

    # --- unsaturation (ene/yne locants on the fixed spiro numbering) ---
    is_mancude = any(sub.GetAtomWithIdx(i).GetIsAromatic()
                     for i in range(sub.GetNumAtoms()))
    if is_mancude and not allow_mancude:
        return None
    kek = Chem.RWMol(sub)
    if is_mancude:
        try:
            Chem.Kekulize(kek, clearAromaticFlags=True)
        except Exception:
            return None
    # T1: one shared producer for both bond orders (this block and the
    # cage sibling's computed the same thing twice, and only this copy guarded the
    # yne). The blanket non-consecutive-yne refusal that used to live here is now
    # the primitive's, so both paths refuse identically.
    from .ring_unsaturation import render_ring_unsaturation
    ring_unsat = render_ring_unsaturation(kek, numbering)
    if ring_unsat is None:
        logger.info("analyze_spiro_universal: ring unsaturation not citable; "
                    "refuse")
        return None
    unsat = {
        'double_bond_pairs': list(ring_unsat.double_pairs),
        'double_bonds': list(ring_unsat.double_locants),
        'triple_bonds': list(ring_unsat.triple_locants),
    }

    # --- heteroatom skeletal-replacement prefix (P-24.2.4) ---
    hetero_prefix = ""
    hetero_per_atom: Tuple[Tuple[int, str], ...] = ()
    if any(sub.GetAtomWithIdx(i).GetSymbol() != 'C'
           for i in range(sub.GetNumAtoms())):
        # T2a: the SAME skeletal-replacement totality rule the cage
        # sibling applies. ``build_replacement_prefix`` is consulted for
        # ``.unexpressed`` ONLY — the spelling stays with ``_build_hetero_prefix``
        # so this path's strings are byte-identical for the in-table elements (the
        # two builders differ in their λ source and their >20-multiplier handling).
        #
        # ``_build_hetero_prefix`` now applies this same gate INTERNALLY, so this
        # block is no longer the only thing standing between an off-table element
        # and an invented morpheme (it previously was, which is why the OTHER
        # caller — the PIN ``name_spiro_system`` — shipped ``3-alaspiro[…]``).
        # It is kept deliberately: the gate here runs against the numbering THIS
        # function computed (which may carry a free-valence bias via
        # ``suffix_ring_atoms``), whereas the builder re-derives its own numbering
        # unbiased. The two agree on element membership but not necessarily on
        # which atoms the numbering reaches, so keeping both makes the refusal
        # independent of that difference.
        from .ring_replacement import build_replacement_prefix
        totality = build_replacement_prefix(
            sub, numbering, set(range(sub.GetNumAtoms())))
        if totality.unexpressed:
            logger.info(
                "analyze_spiro_universal: %d skeletal atom(s) unexpressible by "
                "the replacement-prefix table; refuse",
                len(totality.unexpressed))
            return None
        hp = _build_hetero_prefix(
            sub, set(spiro_sub), set(range(sub.GetNumAtoms())))
        if hp is None:
            return None  # hetero present but prefix underivable -> fail closed
        hetero_prefix = hp.prefix
        # T5 (sibling completion): the per-morpheme decomposition of
        # the prefix, in ORIGINAL indices -- the same field, from the same
        # builder-that-spelled-it discipline, as the cage sibling above. Without
        # it the shared emission tail binds no token to ``oxa``/``thia`` and P5
        # reports the morpheme as name text nothing accounts for, which is the
        # exact finding T5 removed on the cage while leaving it live here.
        hetero_per_atom = tuple(sorted(
            (sub_to_mol[k], morpheme) for k, morpheme in hp.per_atom))

    total_atoms = sub.GetNumAtoms()
    cage_orig = tuple(sorted(cage_set))
    atom_to_locant = {sub_to_mol[k]: loc for k, loc in numbering.items()}

    return SpiroSystem(
        descriptor=descriptor,
        total_atoms=total_atoms,
        hetero_prefix=hetero_prefix,
        unsaturation=unsat,
        cage_atoms=cage_orig,
        atom_to_locant=atom_to_locant,
        canon_match=tuple(cage_orig),
        is_mancude=is_mancude,
        hetero_per_atom=hetero_per_atom,
    )
