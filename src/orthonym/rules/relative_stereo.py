"""cis/trans names for ring pairs at the best-effort tier.

A ring pair that the Blue Book names with lowercase ``r``/``s`` descriptors has a second,
general name with ``cis`` or ``trans`` in front:

    (1r,4r)-cyclohexane-1,4-diol (PIN) trans-cyclohexane-1,4-diol (the Blue Book)
    bis[(1r,4r)-4-methylcyclohexyl]phosphane (PIN)
                                         bis(trans-4-methylcyclohexyl)phosphane (:48117)

 "Achiral cyclic compounds" (``:48103``) gives the lowercase descriptors the PIN
status; ``### Relative configuration: the stereodescriptors 'cis' and 'trans'``
(``:48155``) gives the general name: "When one ligand and one hydrogen atom are attached at each
of two positions of a monocycle, the steric relation (the relative configuration) of the two
ligands is expressed as 'cis' or 'trans', followed by a hyphen and placed before the name of the
compound. No locants are required before the stereodescriptors."

The best-effort tier ships only a name that OPSIN reads back to the input's full InChIKey, and
OPSIN reads no lowercase ``r``/``s``, so the strict path's name is withdrawn there. This module
turns such a withdrawn name into the general name. It works from what the descriptor check
accepted (``validation.stereo_descriptors.stereo_block_units``: the span of each descriptor block,
its multiplicity and the stereo units the reading binds to it), never from a locant read out of the string, and the
caller audits every candidate: the completeness predicates below, then the full-key round trip.

``note_withdrawn_name``/``withdrawn_names`` are the slot that hands the withdrawn names from the
validity gate to the offer builder,:func:`append_relative_stereo_offer`
(``Orthonym._maybe_append_relative_stereo_offer`` calls it).
"""
import itertools
import logging
import re
import threading
from typing import Callable, List, Optional, Sequence, Tuple

from rdkit import Chem

from .stereochemistry import (
    _ring_cistrans_is_complete,
    _rings_jointly_complete,
    locate_stereo_blocks,
    strip_stereo_blocks,
)

logger = logging.getLogger(__name__)

#: One CIP descriptor token of a block, a locant (primes allowed) and a code: ``1r``, ``3'R``, ``2E``.
_CIP_TOKEN_RE = re.compile(r"^(?:\d+[a-z]?['′]*)?([RSrsEZ])$")

#: A name carries at most this many blocks to convert (two senses each: four candidates).
_MAX_BLOCKS = 2

#: The slot keeps the first names withdrawn for the input, in the order withdrawn. The bound is
#: only a memory guard: a naming withdraws one to seven such names (RS0 fixture, 810 rows).
_MAX_NAMES = 64

_slot = threading.local()


def _block_codes(text: str) -> List[str]:
    """The CIP code letters of one descriptor block (``'(1R,4r)'`` gives ``['R', 'r']``);
    empty for a block that is not a list of CIP tokens (``'cis'``, ``'rel'``)."""
    if not (text.startswith("(") and text.endswith(")")):
        return []
    matches = [_CIP_TOKEN_RE.match(t.strip()) for t in text[1:-1].split(",")]
    if not matches or not all(matches):
        return []
    return [m.group(1) for m in matches]


def _holds_pseudoasymmetric_code(text: str) -> bool:
    """Does the descriptor block ``text`` cite a lowercase ``r`` or ``s``?"""
    return any(code in "rs" for code in _block_codes(text))


def holds_pseudoasymmetric_block(name: str) -> bool:
    """Does ``name`` carry a descriptor block with a lowercase ``r`` or ``s``, a block that starts a later word of an
    ester included ('ethyl (1r,4r)-4-hydroxycyclohexane-1-carboxylate')?"""
    return any(_holds_pseudoasymmetric_code(text) for _s, _e, text in locate_stereo_blocks(name, words=True))


# --- skeletal replacement chains -----------------------------------------------------------------

#: The skeletal replacement ('a') prefixes of (``oxa``, ``aza``,...).
_A_PREFIXES = ("oxa", "thia", "selena", "tellura", "aza", "phospha", "arsa", "stiba", "bisma",
               "sila", "germa", "stanna", "plumba", "bora")

_replacement_chain_re = None


def _replacement_chain_pattern():
    """``(?<![a-z])`` + a run of 'a' prefixes (each optionally multiplied and followed by the
    locants of the next one) + an acyclic chain stem or ``cyclo``: ``2-oxaeth-1-en-1-yl``,
    ``4-oxa-2-azabut-3-en-1-yl``, ``1-oxacyclopentan-3-yl``, ``3-azahexanamide``. A ring written
    with a Hantzsch-Widman stem (``oxolan``, ``1,3-oxazol``, ``1,4-oxazepan``), a von Baeyer or
    spiro name with a replacement prefix (``2-azaspiro[4.5]decane``) and the letters inside a
    word (``carboxamide``) do not match. The chain stems are the ones of ``data.chain_names``."""
    global _replacement_chain_re
    if _replacement_chain_re is None:
        from ..data.chain_names import get_chain_prefix
        stems = set()
        for n in range(1, 41):
            try:
                stems.add(get_chain_prefix(n))
            except Exception:  # noqa: BLE001 - the stem table is bounded
                break
        stems_alt = "|".join(sorted(stems, key=len, reverse=True))
        run = "(?:(?:di|tri|tetra|penta|hexa)?(?:%s)(?:-\\d+(?:,\\d+)*-)?)+" % "|".join(_A_PREFIXES)
        _replacement_chain_re = re.compile(r"(?<![a-z])%s(?:cyclo|%s)" % (run, stems_alt))
    return _replacement_chain_re


def names_a_replacement_chain(name: str) -> bool:
    """Does ``name`` spell an acyclic chain or a monocycle by skeletal replacement,
    such as ``...-2-oxaeth-1-en-1-yl...`` or ``1-oxacyclopentan-3-yl``? A name of this class is
    not a general name of the Blue Book (the heteroatom has a functional-group or
    Hantzsch-Widman name), so it is not the start of a relative-stereo offer. READ-ONLY: it
    classifies a candidate; it never rewrites one."""
    return bool(name and _replacement_chain_pattern().search(name))


# --- the slot ------------------------------------------------------------------------------

def clear_withdrawn_names() -> None:
    """Empty the slot. Called at the start and at the end of every outermost ``name`` so that
    a naming reads only the names its own run withdrew."""
    _slot.input = None
    _slot.names = []


def note_withdrawn_name(name: str, smiles: str, caller_input: Optional[str]) -> None:
    """Record that the validity gate withdrew ``name`` for ``smiles`` at the best-effort tier.

    Only a WHOLE-MOLECULE call is recorded: ``smiles`` is the string the outermost ``name``
    was given (``caller_input``). The cascade also gates the names of fragments of the
    molecule, and the names of other molecules a producer probes; their strings differ and
    they are ignored. Only a name with a lowercase ``r``/``s`` block is kept (the name no
    best-effort name can be: OPSIN reads no such descriptor), once, in the order withdrawn.
    Never raises: the slot is read by an offer builder that must not break naming."""
    try:
        if not name or not smiles or smiles != caller_input:
            return
        if not holds_pseudoasymmetric_block(name):
            return
        names = getattr(_slot, "names", None)
        if names is None or getattr(_slot, "input", None) != smiles:
            names = _slot.names = []
            _slot.input = smiles
        if name not in names and len(names) < _MAX_NAMES:
            names.append(name)
    except Exception:  # pragma: no cover - a slot write must never break naming
        pass


def withdrawn_names(smiles: str) -> Tuple[str, ...]:
    """The names with a lowercase ``r``/``s`` block that the gate withdrew for the whole
    molecule ``smiles`` since the slot was cleared, in the order withdrawn."""
    if getattr(_slot, "input", None) != smiles:
        return ()
    return tuple(getattr(_slot, "names", None) or ())


# --- the conversion ---------------------------------------------------------------------------

def _pair_ring(mol, ring_info, record):
    """``(a, b, ring_atoms)`` when the descriptor block of ``record`` names the two positions
    of one monocycle that expresses with ``cis``/``trans``; ``None`` otherwise.

    "One ligand and one hydrogen atom are attached at each of two positions of a monocycle"
    (the Blue Book): the block binds exactly two stereocentres; they lie in one
    saturated, non-aromatic 4- or 6-membered ring none of whose atoms belongs to another ring
    (the only rings in which two positions can both be pseudoasymmetric); each of the two
    carries one hydrogen; and they are the only stereocentres of that ring (a third would
    make the word under-specified). The block is cited once (``multiplicity`` 1): a block
    under a multiplier (``bis[(1r,4r)-...]``, ``:48117``) is not converted."""
    if record.multiplicity != 1 or len(record.units) != 2:
        return None
    if any(kind != "a" for kind, _idx in record.units):
        return None
    a, b = (idx for _kind, idx in record.units)
    rings = [r for r in ring_info.AtomRings() if a in r and b in r]
    if len(rings) != 1:
        return None
    ring = rings[0]
    if len(ring) not in (4, 6):
        return None
    if any(ring_info.NumAtomRings(i) != 1 for i in ring):
        return None
    atoms = {i: mol.GetAtomWithIdx(i) for i in ring}
    if any(at.GetIsAromatic() for at in atoms.values()):
        return None
    for i in range(len(ring)):
        bond = mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % len(ring)])
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            return None
    if any(atoms[i].GetTotalNumHs(includeNeighbors=True) != 1 for i in (a, b)):
        return None
    if {i for i, at in atoms.items() if at.HasProp("_CIPCode")} != {a, b}:
        return None
    return a, b, frozenset(ring)


def convert_pair_blocks(name: str, mol, records: Sequence) -> List[str]:
    """The ``cis``/``trans`` names of ``name``, a name with lowercase ``r``/``s`` descriptor
    blocks, for the input ``mol``; ```` when the blocks cannot all be converted.

    ``records`` are the:class:`validation.stereo_descriptors.StereoBlockUnits` of ``name`` (``text``, ``span``,
    ``multiplicity``, ``units``), the output of the descriptor check that accepted ``name``
    for this input. Every block with a lowercase code must be a ring pair
    (:func:`_pair_ring`) that is complete for a relative word
    (``_ring_cistrans_is_complete``: flipping both centres gives the same molecule; with two
    blocks also ``_rings_jointly_complete``), and there are at most two such blocks, each in
    its own ring; a block with only uppercase codes (a stereocentre beside the pair, a meso
    pair) keeps its absolute descriptors. Every block is replaced at its span by ``cis-``
    or ``trans-``, in both senses, rightmost first so that the spans of the blocks to its
    left still hold; the enclosing marks of the groups that held a block, and of
    every group around them, are re-derived after each replacement (the stereo parentheses
    count toward the nesting order,, the Blue Book, and the word has none:
    ``[(1r,4r)-4-methylcyclohexyl]acetic acid`` becomes ``(trans-4-methylcyclohexyl)acetic
    acid``, not ``[trans-4-methylcyclohexyl]acetic acid``).

    The candidates are in a fixed order (the senses of the leftmost block vary slowest,
    ``cis`` first). A candidate names the right molecule only if OPSIN reads it back to the
    input's full InChIKey; the caller checks that, which also picks the sense.

    Each candidate is a string built from ``name``, so the producer that built ``name`` for the
    whole molecule is the one that built the candidate: the label of ``name`` in the producer
    record of the call (``metrics.producer_trail``) passes to every candidate, which is what makes
    the row of a molecule that ships a ``cis``/``trans`` name say which code built it."""
    from ..assembly.naming_utils import enclosing_opener, renest_group_and_ancestors
    from ..metrics.producer_trail import carried
    from ..perception.stereo import assign_stereochemistry

    blocks = [r for r in records if _holds_pseudoasymmetric_code(r.text)]
    if not blocks or len(blocks) > _MAX_BLOCKS:
        return []
    m = Chem.Mol(mol)
    assign_stereochemistry(m)
    ring_info = m.GetRingInfo()
    pairs = []
    for record in blocks:
        pair = _pair_ring(m, ring_info, record)
        if pair is None or not _ring_cistrans_is_complete(m, pair[0], pair[1]):
            return []
        pairs.append(pair)
    rings = [ring for _a, _b, ring in pairs]
    if len(set(rings)) != len(rings):
        return []
    if len(rings) == 2 and not _rings_jointly_complete(m, rings):
        return []
    # the spans are those of the name the records were read from
    if any(name[r.span[0]:r.span[1]] != r.text + "-" for r in blocks):
        return []
    ordered = sorted(blocks, key=lambda r: r.span[0])
    out: List[str] = []
    for senses in itertools.product(("cis", "trans"), repeat=len(ordered)):
        candidate = name
        for record, sense in sorted(zip(ordered, senses), key=lambda t: -t[0].span[0]):
            start, end = record.span
            opener = enclosing_opener(candidate, start)
            candidate = candidate[:start] + sense + "-" + candidate[end:]
            if opener is not None:
                candidate = renest_group_and_ancestors(candidate, opener)
        if candidate not in out:
            out.append(carried(name, candidate))
    return out


# --- the offer ----------------------------------------------------------------------------------

def append_relative_stereo_offer(offers: list, mol, smiles: str, primary_name: str,
                                 primary_is_failure: bool, *,
                                 rt_ok: Callable[[str, Optional[str]], bool],
                                 self01_lookup: Callable) -> None:
    """: append the ``cis``/``trans`` name of a ring pair to ``offers`` as an ``Offer``
    above the systematic floor, IF the primary winner does not already full-RT-pass.

    The best-effort tier ships only a name that OPSIN reads, and OPSIN 2.9.0 reads no
    lowercase ``r``/``s`` (BE-STRICT, ``_final_opsin_validity_gate``), so the name of a ring pair
    that cites them -- ``(1r,4r)-cyclohexane-1,4-diol``, the Blue Book
    "Achiral cyclic compounds") -- is withdrawn there and the molecule is left with the floor's
    ``trans-1,4-dihydroxycyclohexane``. The Blue Book gives the same pair a general name,
    ``trans-cyclohexane-1,4-diol`` "Relative configuration: the stereodescriptors
    'cis' and 'trans'", ``:48155``).

    The names the gate withdrew for this whole molecule are in:func:`withdrawn_names`, in the
    order they were withdrawn; the cascade builds them (the strict path is one source of them,
    not the only one). A withdrawn name is ELIGIBLE when

    * it does not spell a chain or a monocycle by skeletal replacement
      (:func:`names_a_replacement_chain`);
    * the descriptor check (``validation.stereo_descriptors.stereo_block_units``, the check the
      default tier's stereo-layer carve-out applies) accepts it for this input: every lowercase
      block is one it located, at the front of the name or of a prefix, never in the middle of
      the name, and its r/s descriptors are proven right;
    *:func:`convert_pair_blocks` converts every lowercase block (a ring pair that both
      completeness predicates make complete for a relative word) and re-derives the enclosing
      marks; and
    * OPSIN reads a converted name back to the input's full InChIKey (``rt_ok``, which has no
      fail-open branch for a name no gate has seen).

    The first candidate of the first eligible name is appended: ``tier`` systematic_verified (a
    correct name that is not the preferred one), never a PIN. The floor offer stays in the pool,
    so a name that does not read back leaves today's floor name.

    ``rt_ok(name, smiles)`` is the full-InChIKey round-trip predicate of the offer pool
    (``namer._offer_rt_ok``) and ``self01_lookup(name)`` the verdict the coverage audit
    reuses (``namer._self01_lookup``); the caller passes them so that they are resolved in the
    namer module at call time.

    Best-effort ONLY, like the floor: both call sites in ``Orthonym._finish`` sit inside the
    ``_general_fallback_unverified`` guard. Nothing is built when the slot holds no name for this
    input, which is every input without a lowercase r/s block.

    Never raises past the candidate loop -- a failure there must leave ``offers`` as it was."""
    withdrawn = withdrawn_names(smiles)
    if not withdrawn:
        return
    if not primary_is_failure and rt_ok(primary_name, smiles):
        return  # primary already full-RT-passes -- no relative name is needed
    try:
        from ..assembly.coverage_audit import audit_coverage
        from ..assembly.offer_pool import SYSTEMATIC_VERIFIED, Offer
        from ..validation.stereo_descriptors import stereo_block_units
        for strict_name in withdrawn:
            if names_a_replacement_chain(strict_name):
                continue
            records = stereo_block_units(
                strict_name, smiles, strip_stereo_blocks(strict_name)[0])
            if records is None and strip_stereo_blocks(strict_name, words=True)[0] != strip_stereo_blocks(strict_name)[0]:
                # a block that starts a later word ('ethyl (1r,4r)-4-hydroxycyclohexane-1-carboxylate'): read in word
                # mode. The block's word is the producer's: OPSIN reads a relative word in the wrong word to the same key
                # ('trans-phenyl 4-methyl...carboxylate'), so the read-back below checks the pair, not the word (2026-11-20)
                records = stereo_block_units(
                    strict_name, smiles, strip_stereo_blocks(strict_name, words=True)[0], (), True)
            if records is None:
                continue
            for candidate in convert_pair_blocks(strict_name, mol, records):
                if not rt_ok(candidate, smiles):
                    continue
                _self01, _skip_reanchor, _skip_detail = self01_lookup(candidate)
                verdict = audit_coverage(
                    mol, candidate, None, self01_complete=_self01,
                    skip_reanchor=_skip_reanchor, skip_detail=_skip_detail)
                offers.append(Offer(
                    name=candidate, result_obj=None, is_pin=False,
                    tier=SYSTEMATIC_VERIFIED, source="pin_path",
                    complete=verdict.complete))
                # the winner's labels come from the name-scoped record when the primary
                # offer has the same source (``pin_path``): ``name_tiered`` reads the
                # labels of the offer only when its source differs from the provenance's
                from ..metrics.provenance import record_non_pin_label
                record_non_pin_label(
                    candidate, rule="P-93.5.1.2",
                    detail="cis/trans is the general name of a ring pair; the PIN "
                           "cites its lowercase r/s descriptors (P-93.5.1.1.2)")
                logger.info("relative stereo offer: name=%r complete=%s (method=%s)",
                            candidate[:80], verdict.complete, verdict.method)
                return
    except Exception as _re:  # pragma: no cover - defensive
        logger.info("relative stereo offer computation failed (kept existing offers): %s", _re)
