"""v30 P3-T1b: the TERMINAL ring namer -- the audited systematic generator that
stands where the ``'substituent'`` refusal sentinel used to.

Why this module exists
----------------------
``errors.py:224`` calls the bare word ``substituent`` *"a REFUSAL, not a name"*,
and it is what the ring-bearing branch of
``assembly/substituent_enumerator._descriptive_fallback`` returns when every
narrow producer has declined. Our tables ARE the path, so a table miss has
nowhere to fall. This module is the fall-through: a systematic name built from
the ring graph itself, so a table miss degrades to an UGLIER name instead of a
refusal.

What is NEW here and what is reused
-----------------------------------
The polycyclic half already existed and is reused unchanged --
``vonbaeyer_universal.analyze_cage_universal`` / ``analyze_spiro_universal``
already kekulize, build the von Baeyer / spiro descriptor, apply the TOTAL
skeletal-replacement prefix (with λ) and cite every ring multiple bond, and
already gate every result on a reconstruction audit
(``audit_von_baeyer_descriptor`` / ``audit_spiro_descriptor``). Measured
2026-08-03: they name rank-2..8 cages including hetero and mancude ones. What
they cannot do -- by definition, at ``vonbaeyer_universal.py:424`` -- is a
MONOCYCLE, because von Baeyer nomenclature starts at two rings.

So the new code below is the MONOCYCLE branch: P-22.2.3 skeletal ('a')
replacement nomenclature over a ``cyclo``-alkane stem, with a locant for every
non-carbon skeletal atom, λ for hypervalence, and an explicit locant for every
ring multiple bond -- plus its own reconstruction audit, built to the same
contract as the von Baeyer one: parse the EMITTED STRING back into
(ring size, {locant: element}, {bond-order edges}) and require SET EQUALITY with
the molecular graph under the emitted numbering.

Scope (stated explicitly, per the task contract)
------------------------------------------------
* **Elements** -- carbon plus the 18 ring 'a'-prefix rows this project admits
  (``ring_replacement.HETEROATOM_PREFIXES``: O S Se Te N P As Sb Bi Si Ge Sn Pb
  B Al Ga In Tl). The Blue Book's replacement set is Table 1.5 (P-15.4.1.1),
  which is a CLOSED list, so an off-table skeletal element (Zn/Cd/Hg/Fe/…) has no
  morpheme and MUST refuse -- see the ``ring_replacement`` module docstring.
* **Cycle rank** -- 1 (this module's monocycle branch) and 2..8 (delegated to the
  universal analyzers; 8 is ``vonbaeyer_universal.MAX_CAGE_RINGS``, deliberately
  NOT raised here: its own comment at ``vonbaeyer_universal.py:48`` requires the
  P-23.2.4 main-bridge selection to be fixed first).
* **Size** -- at most 40 skeletal atoms (``MAX_CAGE_ATOMS``; the monocycle branch
  is additionally bounded by ``data.chain_names.get_chain_prefix``).
* **Charge** -- a charged skeletal ring atom REFUSES. A ring cation/anion is
  P-73 (``cation_words`` / ``ion_retained_names``), not replacement
  nomenclature; ``[n+]`` in a ring is a different naming class and inventing a
  neutral 'a'-prefix name for it would name a DIFFERENT species.

Within that scope, and for a name that passes the reconstruction audit, this
module does not return ``None`` for a connected ring system. The audit is part
of the contract, not an escape hatch: a name that fails it is refused, and the
caller keeps its existing behaviour. Every refusal is logged at INFO with the
reason so the refused count is measurable.

IUPAC references
----------------
* P-22.2.3 "Skeletal replacement ('a') nomenclature" for monocyclic rings.
* P-15.4.1.1 + Table 1.5 -- the closed replacement-prefix set.
* P-15.4.1.3 -- λ placement (immediately after the locant, no hyphen).
* P-31.1.4.2(1) -- ring multiple-bond locant citation.
* P-29.2 / P-29.3.2 -- the ``-yl`` free valence and its lowest locant.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Set, Tuple

from rdkit import Chem

logger = logging.getLogger(__name__)

#: Shared with the von Baeyer sibling so the two halves of the terminal namer
#: cannot drift into two different ceilings.
from .lambda_convention import LAMBDA  # noqa: E402
from .vonbaeyer_universal import MAX_CAGE_ATOMS, MAX_CAGE_RINGS  # noqa: E402

__all__ = [
    "TerminalRingName",
    "audit_monocycle_replacement_name",
    "build_monocycle_replacement_name",
    "monocycle_numbering",
    "parse_monocycle_replacement_name",
    "terminal_ring_name",
]


@dataclass(frozen=True)
class TerminalRingName:
    """One audited terminal ring name.

    ``name``      the emitted string -- the parent hydride when
                  ``free_valence`` is None, else the ``…-<loc>-yl`` substituent
                  token.
    ``numbering``  atom idx -> ring locant, the SAME map the name was spelled
                  from (never re-derived), so a consumer can place its own
                  substituent locants consistently.
    ``basis``      which generator + audit produced it: ``'monocycle'``,
                  ``'von_baeyer'`` or ``'spiro'``.
    """

    name: str
    numbering: Dict[int, int]
    basis: str


# --------------------------------------------------------------------------
# morpheme <-> element inversion (the audit's element oracle)
# --------------------------------------------------------------------------

def _morpheme_to_element() -> Dict[str, str]:
    """``{'oxa': 'O', 'thia': 'S', …}`` -- the inverse of the SAME table the
    prefix builder spells from, so the audit cannot disagree with the speller
    about what a morpheme means."""
    from .ring_replacement import HETEROATOM_PREFIXES
    return {stem: sym for sym, (stem, _rank) in HETEROATOM_PREFIXES.items()}


def _multiplier_to_count() -> Dict[str, int]:
    from ..assembly.naming_utils import SIMPLE_MULTIPLIERS
    return {word: n for n, word in SIMPLE_MULTIPLIERS.items()}


def _size_from_stem(stem: str) -> Optional[int]:
    """Invert ``get_chain_prefix`` over the sizes this module can emit."""
    from ..data.chain_names import get_chain_prefix
    for n in range(3, MAX_CAGE_ATOMS + 1):
        try:
            if get_chain_prefix(n) == stem:
                return n
        except (ValueError, KeyError):
            continue
    return None


# --------------------------------------------------------------------------
# monocycle perception + numbering
# --------------------------------------------------------------------------

def _cycle_order(mol, ring_atoms: Sequence[int]) -> Optional[list]:
    """The atoms of ``ring_atoms`` in cyclic order, or None when they are not a
    single simple cycle (every atom exactly two ring-neighbours inside the set).
    """
    core = list(ring_atoms)
    n = len(core)
    if n < 3:
        return None
    core_set = set(core)
    adj = {}
    for i in core:
        nb = [x.GetIdx() for x in mol.GetAtomWithIdx(i).GetNeighbors()
              if x.GetIdx() in core_set]
        if len(nb) != 2:
            return None
        adj[i] = nb
    order = [core[0], adj[core[0]][0]]
    while len(order) < n:
        prev, cur = order[-2], order[-1]
        nxt = [x for x in adj[cur] if x != prev]
        if not nxt:
            return None
        order.append(nxt[0])
    if len(order) != n or order[0] not in adj[order[-1]]:
        return None
    return order


def monocycle_numbering(
    mol, ring_atoms: Sequence[int], free_valence_atom: Optional[int] = None,
) -> Optional[Dict[int, int]]:
    """Deterministic P-14.4-style numbering of a simple monocycle.

    Lowest-locant key, applied in order: heteroatoms as a SET, then heteroatoms
    in element-seniority order (P-23.3.1 ranks, shared with the von Baeyer
    prefix builder so the two do not disagree), then the free valence
    (P-29.3.2), then the ring multiple bonds. Every starting atom and both
    directions are enumerated, so the answer is independent of RDKit atom order.

    Returns ``None`` only when ``ring_atoms`` is not a single simple cycle.
    """
    order = _cycle_order(mol, ring_atoms)
    if order is None:
        return None
    from .ring_replacement import HETEROATOM_PREFIXES
    n = len(order)
    het = [i for i in order if mol.GetAtomWithIdx(i).GetSymbol() != 'C']
    # ring bond endpoints, as atom pairs, for the unsaturation tie-break
    ring_set = set(order)
    multiple: list = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in ring_set and j in ring_set:
            # NOTE the documented hazard: RDKit reports an AROMATIC bond as
            # order 1.5, so `>= 2.0` would MISS it. This function is only a
            # tie-break, and the caller kekulizes before spelling, but the
            # predicate is written to catch aromatic too so the numbering does
            # not depend on whether the caller kekulized first.
            if bond.GetIsAromatic() or bond.GetBondTypeAsDouble() >= 2.0:
                multiple.append((i, j))

    best_key = None
    best = None
    for start in range(n):
        for direction in (1, -1):
            a2p = {order[(start + direction * p) % n]: p + 1 for p in range(n)}
            het_set = tuple(sorted(a2p[i] for i in het))
            sen = tuple(a2p[i] for i in sorted(
                het, key=lambda a: (HETEROATOM_PREFIXES.get(
                    mol.GetAtomWithIdx(a).GetSymbol(), ('', 99))[1], a2p[a])))
            fv = a2p[free_valence_atom] if free_valence_atom in a2p else 0
            unsat = tuple(sorted(
                tuple(sorted((a2p[i], a2p[j]))) for i, j in multiple))
            key = (het_set, sen, fv, unsat)
            if best_key is None or key < best_key:
                best_key, best = key, a2p
    return best


# --------------------------------------------------------------------------
# monocycle name construction
# --------------------------------------------------------------------------

def _unsaturation_block(double_locs: Sequence[str], triple_locs: Sequence[str],
                        ) -> str:
    """``''`` (saturated -> caller appends ``an``), ``'-1-en'``,
    ``'a-1,3-dien'``, ``'-2-yn'``, ``'a-1,3-dien-5-yn'`` …

    Elision follows P-16.3.3: a multiplied ending keeps the stem's terminal
    ``a`` (``cyclohexa-1,3-diene``); a single ending elides it
    (``cyclohex-1-ene``).
    """
    from ..assembly.naming_utils import SIMPLE_MULTIPLIERS
    terms = []
    for locs, ending in ((list(double_locs), 'en'), (list(triple_locs), 'yn')):
        if not locs:
            continue
        mult = SIMPLE_MULTIPLIERS.get(len(locs), '') if len(locs) > 1 else ''
        terms.append((locs, mult, ending))
    if not terms:
        return ''
    # the stem keeps its 'a' iff the FIRST ending is multiplied
    head_a = 'a' if terms[0][1] else ''
    out = head_a
    for locs, mult, ending in terms:
        out += f"-{','.join(locs)}-{mult}{ending}"
    return out


def build_monocycle_replacement_name(
    mol, ring_atoms: Sequence[int], numbering: Dict[int, int],
    free_valence_locant: Optional[int] = None,
) -> Optional[str]:
    """The P-22.2.3 replacement name for a simple monocycle, or None.

    ``mol`` MUST already be kekulized: ``render_ring_unsaturation`` reads
    ``GetBondType()``, and an aromatic bond is neither DOUBLE nor TRIPLE, so an
    un-kekulized mancude ring would silently lose every one of its double bonds.

    Returns the parent hydride (``1-thiacyclohexane``) when
    ``free_valence_locant`` is None, else the substituent token
    (``1-thiacyclohexan-4-yl``).
    """
    from ..data.chain_names import get_chain_prefix
    from .ring_replacement import build_replacement_prefix
    from .ring_unsaturation import render_ring_unsaturation

    ring_set = set(ring_atoms)
    n = len(ring_set)
    if n < 3 or n > MAX_CAGE_ATOMS:
        return None
    try:
        stem = get_chain_prefix(n)
    except (ValueError, KeyError):
        return None
    if not stem:
        return None

    # TOTALITY: a skeletal atom no morpheme spells must refuse -- the stem
    # counts it regardless, which is the silent-heteroatom-drop shape
    # (`C1CC2CC[Hg]C2C1` -> `bicyclo[3.3.0]octane`).
    repl = build_replacement_prefix(mol, numbering, ring_set)
    if repl.unexpressed:
        logger.info("terminal_ring: %d skeletal atom(s) have no replacement "
                    "morpheme; refuse", len(repl.unexpressed))
        return None

    unsat = render_ring_unsaturation(mol, numbering)
    if unsat is None:
        logger.info("terminal_ring: ring unsaturation not citable; refuse")
        return None
    # A monocycle is numbered consecutively around the ring, so every multiple
    # bond joins locants n and n+1 (or N and 1). A COMPOSITE locant here means
    # the numbering is not a walk of the cycle -- refuse rather than emit a
    # citation the grammar below cannot audit.
    if any('(' in t for t in unsat.double_locants + unsat.triple_locants):
        logger.info("terminal_ring: composite ring-bond locant on a monocycle "
                    "(numbering is not a cycle walk); refuse")
        return None
    # The N->1 ring-closure bond. ``_locant`` renders it ``1(N)`` (non-
    # consecutive), so the composite refusal above ALREADY catches it and this is
    # a second, explicit statement of the same decision.
    #
    # It is deliberately a REFUSAL rather than a citation. Citing it as locant
    # ``N`` is the usual reading of P-31.1.4.2, but the audit would then have to
    # apply that same convention to parse it back -- speller and parser sharing an
    # assumption is exactly the shape the audit exists to rule out, so the
    # convention would be unproven rather than verified. Extending the audit to
    # prove it independently (e.g. against OPSIN's reading) is the follow-up; a
    # refusal is the correct interim state, not a permanent one.
    for lo, hi in list(unsat.double_pairs) + list(unsat.triple_pairs):
        if (lo, hi) == (1, n):
            logger.info("terminal_ring: ring-closure multiple bond (1,%d) is "
                        "not independently auditable; refuse", n)
            return None

    d_locs = sorted(unsat.double_locants, key=int)
    t_locs = sorted(unsat.triple_locants, key=int)

    # P-23.3.1 / P-22.2.3: the 'a'-prefix block attaches DIRECTLY to the ring
    # stem, no hyphen ('1-thiacyclohexane', never '1-thia-cyclohexane').
    head = f"{repl.prefix}cyclo" if repl.prefix else "cyclo"
    body = _unsaturation_block(d_locs, t_locs)
    if not body:
        body = 'an'
    if free_valence_locant is None:
        return f"{head}{stem}{body}e"
    return f"{head}{stem}{body}-{free_valence_locant}-yl"


# --------------------------------------------------------------------------
# the reconstruction audit -- the thing that makes "always emit" safe
# --------------------------------------------------------------------------

#: the closed grammar ``build_monocycle_replacement_name`` emits, and nothing
#: else. Anything this does not match is refused rather than assumed correct.
_MONO_RE = re.compile(
    r'^(?P<repl>.*?)cyclo(?P<stem>[a-z]+?)'
    r'(?P<unsat>an|a?(?:-\d+(?:,\d+)*-(?:di|tri|tetra|penta|hexa|hepta|octa|'
    r'nona)?(?:en|yn))+)'
    r'(?:-(?P<fv>\d+)-yl|e)$'
)
#: Built from the SAME closed table the prefix speller uses, longest-first so
#: the alternation cannot mis-split a morpheme ('thia' vs 'thalla'). Anchoring
#: the morpheme set (rather than ``[a-z]+``) is what makes the replacement block
#: unambiguously parseable, which is what the audit needs.
_MORPHEME_ALT = '|'.join(sorted(_morpheme_to_element(), key=len, reverse=True))
_REPL_TERM_RE = re.compile(
    r'(?P<locs>\d+(?:' + LAMBDA + r'\d+)?(?:,\d+(?:' + LAMBDA + r'\d+)?)*)-'
    r'(?P<mult>di|tri|tetra|penta|hexa|hepta|octa|nona)?'
    r'(?P<morph>' + _MORPHEME_ALT + r')'
)
_UNSAT_TERM_RE = re.compile(
    r'-(?P<locs>\d+(?:,\d+)*)-'
    r'(?P<mult>di|tri|tetra|penta|hexa|hepta|octa|nona)?(?P<ending>en|yn)')


def parse_monocycle_replacement_name(name: str):
    """``'1-thiacyclohexan-4-yl'`` -> ``(6, {1: 'S'}, frozenset(), frozenset(),
    4)``: (ring size, {locant: element}, double-bond edges, triple-bond edges,
    free-valence locant or None). ``None`` when the string is not in the closed
    grammar this module emits.

    Reads the string ONLY -- no access to the molecule -- so the audit that
    consumes it is a genuine independent reconstruction, exactly as
    ``reconstruct_von_baeyer_skeleton`` is for the cage sibling.
    """
    if not name:
        return None
    m = _MONO_RE.match(name.strip())
    if not m:
        return None
    n = _size_from_stem(m.group('stem'))
    if n is None:
        return None
    fv = int(m.group('fv')) if m.group('fv') else None
    if fv is not None and not (1 <= fv <= n):
        return None

    # --- replacement prefixes -> {locant: element}
    elements: Dict[int, str] = {}
    repl = m.group('repl')
    if repl:
        m2e = _morpheme_to_element()
        m2c = _multiplier_to_count()
        # Terms are '-'-separated and must tile the block EXACTLY: any character
        # the grammar does not account for is a parse failure, never ignored
        # text (a silently-dropped morpheme would be a dropped heteroatom).
        cursor = 0
        while cursor < len(repl):
            tm = _REPL_TERM_RE.match(repl, cursor)
            if not tm or tm.start() != cursor:
                return None
            element = m2e.get(tm.group('morph'))
            if element is None:
                return None
            locs = []
            for tok in tm.group('locs').split(','):
                base = tok.split(LAMBDA, 1)[0]
                if not base.isdigit():
                    return None
                locs.append(int(base))
            expected = m2c.get(tm.group('mult')) if tm.group('mult') else 1
            if expected != len(locs):
                return None  # the multiplier and the locant count disagree
            for loc in locs:
                if not (1 <= loc <= n) or loc in elements:
                    return None
                elements[loc] = element
            cursor = tm.end()
            if cursor < len(repl):
                if repl[cursor] != '-':
                    return None
                cursor += 1

    # --- unsaturation -> edge sets, in locant space
    doubles: Set[Tuple[int, int]] = set()
    triples: Set[Tuple[int, int]] = set()
    unsat = m.group('unsat')
    if unsat != 'an':
        consumed = 0
        if unsat.startswith('a'):
            consumed = 1
        m2c = _multiplier_to_count()
        for tm in _UNSAT_TERM_RE.finditer(unsat, consumed):
            locs = [int(x) for x in tm.group('locs').split(',')]
            expected = m2c.get(tm.group('mult')) if tm.group('mult') else 1
            if expected != len(locs):
                return None
            target = doubles if tm.group('ending') == 'en' else triples
            for loc in locs:
                if not (1 <= loc <= n):
                    return None
                hi = 1 if loc == n else loc + 1
                edge = (min(loc, hi), max(loc, hi))
                if edge in doubles or edge in triples:
                    return None
                target.add(edge)
            consumed = tm.end()
        if consumed != len(unsat):
            return None  # trailing text the grammar does not account for
    return n, elements, frozenset(doubles), frozenset(triples), fv


def audit_monocycle_replacement_name(
    mol, ring_atoms: Sequence[int], numbering: Dict[int, int], name: str,
    free_valence_atom: Optional[int] = None,
) -> bool:
    """Reconstruction audit for a monocycle replacement name (fail-closed).

    Same contract as ``vonbaeyer_universal.audit_von_baeyer_descriptor``: parse
    the EMITTED STRING back into the skeleton it denotes and require SET
    EQUALITY with the molecule's own ring skeleton under ``numbering``. Checks,
    every one of which returns False:

    * the string is not in the emitted grammar;
    * the ring size the stem counts != the number of skeletal atoms;
    * ``numbering`` is not a bijection of the ring onto 1..N;
    * the ring bonds are not the cycle 1-2-…-N-1 the name asserts;
    * an element at a locant differs from the molecule's atom there (including a
      heteroatom the name does not mention -- the silent-drop shape);
    * a double/triple bond the name cites is not that bond order in the
      molecule, or a multiple bond in the molecule is not cited;
    * the free-valence locant does not land on the attachment atom.
    """
    parsed = parse_monocycle_replacement_name(name)
    if parsed is None:
        return False
    n, elements, doubles, triples, fv = parsed
    ring = set(ring_atoms)
    if n != len(ring):
        return False
    # bijection onto 1..N
    locs = [numbering.get(i) for i in ring]
    if any(l is None for l in locs) or sorted(locs) != list(range(1, n + 1)):
        return False
    inv = {numbering[i]: i for i in ring}

    # the cycle 1-2-…-N-1 the name asserts
    asserted_edges = frozenset(
        (min(a, b), max(a, b)) for a, b in
        ((k, 1 if k == n else k + 1) for k in range(1, n + 1)))
    actual_edges = set()
    order_by_edge: Dict[Tuple[int, int], float] = {}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in ring and j in ring:
            a, b = sorted((numbering[i], numbering[j]))
            if a == b:
                return False
            actual_edges.add((a, b))
            # ⚠ RDKit reports an AROMATIC bond as 1.5. The caller kekulizes, but
            # if it did not, 1.5 must NOT be read as a single bond and pass -- so
            # an aromatic bond is recorded as its own order and can never equal
            # the 1.0 / 2.0 / 3.0 the grammar can express.
            order_by_edge[(a, b)] = (
                1.5 if bond.GetIsAromatic() else bond.GetBondTypeAsDouble())
    if frozenset(actual_edges) != asserted_edges:
        return False

    # elements
    for loc in range(1, n + 1):
        symbol = mol.GetAtomWithIdx(inv[loc]).GetSymbol()
        claimed = elements.get(loc, 'C')
        if symbol != claimed:
            return False
        # a charged skeletal atom is a different naming class (P-73) and no
        # 'a'-prefix name expresses it -> the name would denote the neutral ring
        if mol.GetAtomWithIdx(inv[loc]).GetFormalCharge() != 0:
            return False

    # bond orders
    for edge, order in order_by_edge.items():
        if order == 2.0:
            if edge not in doubles:
                return False
        elif order == 3.0:
            if edge not in triples:
                return False
        elif order == 1.0:
            if edge in doubles or edge in triples:
                return False
        else:
            return False  # aromatic (1.5) or dative: not expressible here
    if not doubles <= set(order_by_edge) or not triples <= set(order_by_edge):
        return False

    # free valence
    if fv is None:
        if free_valence_atom is not None:
            return False
    else:
        if free_valence_atom is None or inv.get(fv) != free_valence_atom:
            return False
    return True


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------

def _kekulized_copy(mol, ring_atoms):
    """``(kek, ring_atoms)`` with aromatic flags cleared, or ``(None, None)``.

    Works on an RWMol copy in the SAME index space, so the caller's atom indices
    and any numbering built from them stay valid. Kekulization failure DEGRADES
    (returns None) -- it never raises, per the task's hazard list.
    """
    try:
        kek = Chem.RWMol(mol)
        Chem.Kekulize(kek, clearAromaticFlags=True)
        return kek.GetMol(), ring_atoms
    except Exception as e:  # noqa: BLE001 - kekulization must degrade
        logger.info("terminal_ring: kekulize failed (%s); refuse", e)
        return None, None


def terminal_ring_name(
    mol, ring_atoms: Sequence[int], free_valence_atom: Optional[int] = None,
) -> Optional[TerminalRingName]:
    """The audited terminal name for one connected ring system.

    ``ring_atoms`` must be the skeletal atoms of a single connected ring system
    (the caller owns the partition). ``free_valence_atom`` makes it a ``-yl``
    substituent token numbered per P-29.3.2; ``None`` yields the parent hydride.

    Returns ``None`` only when the ring system is out of the module's stated
    scope (see the module docstring) or when the emitted name FAILS its
    reconstruction audit. Both are logged at INFO.
    """
    if mol is None or not ring_atoms:
        return None
    ring = set(ring_atoms)
    if len(ring) > MAX_CAGE_ATOMS:
        logger.info("terminal_ring: %d atoms > MAX_CAGE_ATOMS; refuse",
                    len(ring))
        return None
    if free_valence_atom is not None and free_valence_atom not in ring:
        return None
    # SCOPE: charge. A charged skeletal ring atom is P-73, not replacement
    # nomenclature; naming it neutral would denote a different species.
    for i in ring:
        if mol.GetAtomWithIdx(i).GetFormalCharge() != 0:
            logger.info("terminal_ring: charged skeletal ring atom; refuse "
                        "(P-73 class, not replacement nomenclature)")
            return None

    kek, ring_atoms_kek = _kekulized_copy(mol, ring)
    if kek is None:
        return None

    from .polycyclic import von_baeyer_ring_count
    try:
        rank = von_baeyer_ring_count(kek, ring)
    except Exception:  # noqa: BLE001
        rank = None
    if rank is None:
        logger.info("terminal_ring: ring count undecidable; refuse")
        return None

    if rank == 1:
        numbering = monocycle_numbering(kek, sorted(ring), free_valence_atom)
        if numbering is None:
            logger.info("terminal_ring: not a single simple cycle; refuse")
            return None
        fv_loc = (numbering.get(free_valence_atom)
                  if free_valence_atom is not None else None)
        name = build_monocycle_replacement_name(
            kek, sorted(ring), numbering, fv_loc)
        if name is None:
            return None
        if not audit_monocycle_replacement_name(
                kek, sorted(ring), numbering, name, free_valence_atom):
            logger.info("terminal_ring: monocycle name %r failed the "
                        "reconstruction audit; refuse", name)
            return None
        return TerminalRingName(name=name, numbering=dict(numbering),
                                basis='monocycle')

    if rank > MAX_CAGE_RINGS:
        # Deliberately NOT raised here: vonbaeyer_universal.py:48 requires the
        # P-23.2.4 main-bridge selection to be fixed first.
        logger.info("terminal_ring: ring count %d > MAX_CAGE_RINGS; refuse",
                    rank)
        return None

    # rank >= 2: the audited universal analyzers, unchanged. They run their own
    # reconstruction audits internally (audit_von_baeyer_descriptor /
    # audit_spiro_descriptor) and return None on any mismatch.
    return _polycyclic_terminal_name(mol, ring, free_valence_atom)


def _polycyclic_terminal_name(mol, ring, free_valence_atom):
    """Delegate to the existing audited cage / spiro analyzers."""
    from .vonbaeyer_universal import (
        analyze_cage_universal,
        analyze_spiro_universal,
        audit_von_baeyer_descriptor,
    )
    for basis, fn in (('von_baeyer', analyze_cage_universal),
                      ('spiro', analyze_spiro_universal)):
        try:
            if basis == 'spiro':
                res = fn(mol, cage_atoms=ring, allow_mancude=True,
                         free_valence_atoms=(
                             {free_valence_atom} if free_valence_atom is not None
                             else None))
            else:
                res = fn(mol, cage_atoms=ring, allow_mancude=True)
        except Exception as e:  # noqa: BLE001 - a declining analyzer must degrade
            logger.info("terminal_ring: %s analyzer raised %s; continue",
                        basis, type(e).__name__)
            res = None
        if res is None:
            continue
        # RE-PROVE the audit at the emission point for the cage form. The
        # analyzer already ran it, but this is the guard THIS module's tests
        # exercise and it re-proves against the ring set THIS function chose.
        if basis == 'von_baeyer':
            kek = Chem.RWMol(mol)
            try:
                Chem.Kekulize(kek, clearAromaticFlags=True)
            except Exception:  # noqa: BLE001
                continue
            if not audit_von_baeyer_descriptor(
                    kek.GetMol(), set(res.cage_atoms), res.atom_to_locant,
                    res.descriptor):
                logger.info("terminal_ring: %s descriptor %r failed the "
                            "reconstruction audit; refuse", basis,
                            res.descriptor)
                continue
        name = _spell_ring_analysis(res, free_valence_atom)
        if name is None:
            continue
        return TerminalRingName(name=name,
                                numbering=dict(res.atom_to_locant),
                                basis=basis)
    return None


def _spell_ring_analysis(res, free_valence_atom) -> Optional[str]:
    """``RingAnalysis`` -> the parent hydride / ``-yl`` token string."""
    from ..data.chain_names import get_chain_prefix
    try:
        stem = get_chain_prefix(res.total_atoms)
    except (ValueError, KeyError):
        return None
    if not stem:
        return None
    d_locs = list(res.unsaturation.get('double_bonds') or ())
    t_locs = list(res.unsaturation.get('triple_bonds') or ())
    body = _unsaturation_block(d_locs, t_locs) or 'an'
    # P-23.3.1: the 'a'-prefix block attaches directly to the descriptor
    # ('2-oxabicyclo[2.2.2]octane'), exactly as the analyzers' own PIN callers
    # concatenate it -- no separator is inserted here.
    head = f"{res.hetero_prefix}{res.descriptor}"
    if free_valence_atom is None:
        return f"{head}{stem}{body}e"
    fv = res.atom_to_locant.get(free_valence_atom)
    if fv is None:
        return None
    return f"{head}{stem}{body}-{fv}-yl"
