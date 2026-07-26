"""Total skeletal ('a') replacement-prefix construction for RING systems.

P-23.3 / P-31.1.4.3.4 (von Baeyer) and P-24.2.4 (spiro): a skeletal non-carbon
ring atom is expressed by a replacement ('a') prefix carrying its locant --
``3-oxabicyclo[2.2.1]heptane``, ``2-oxa-6-thiaspiro[4.5]decane``.

Why this module exists
----------------------
The builder that used to live inline in ``polycyclic.py`` was **partial**, and
partiality here is a wrong-structure bug rather than a coverage gap. It gated
each ring atom on ``symbol in HETEROATOM_PREFIXES`` and skipped everything else,
so an off-table skeletal element contributed **no** morpheme -- while the ring
stem still counted it (``total_atoms = len(ring_atoms)``). ``C1CC2CC[Hg]C2C1``
therefore came back as ``bicyclo[3.3.0]octane``: a hydrocarbon name for a
mercury ring. SELF-01 (name -> structure round trip) fails OPEN when the OPSIN
jar is absent -- a supported mode -- so nothing downstream caught it.

The primitive below returns the prefix *together with* the information a caller
needs to fail closed:

    ``unexpressed`` -- every skeletal non-carbon ring atom the emitted prefix
    does not correctly express: an off-table element, an atom the numbering does
    not reach (no locant to cite), or an occurrence count past the multiplying-
    prefix table (which spelled the bare integer, ``"22sila"``).

**Caller contract: refuse when ``unexpressed`` is non-empty.** Never emit a name
whose ring stem counts an atom that no morpheme in it spells. Both universal
analyzers (``vonbaeyer_universal.analyze_cage_universal`` and
``analyze_spiro_universal``) apply exactly this rule, so the two siblings no
longer disagree: before this module, the cage path *dropped* the atom and the
spiro path *invented* a morpheme for it (``polycyclic_bridged``'s
``get_heteroatom_prefix`` falls back to ``symbol.lower() + 'a'``, spelling
``3-znaspiro[5.5]undecane``).

The table is CLOSED, so fail-closed is the only sound design
-----------------------------------------------------------
The Blue Book's replacement-prefix set (P-22.2.1 / Table 2.8) is a fixed list, not
a generative rule: there is no way to derive a prefix for an arbitrary element, so
"element-blind" is genuinely unachievable and refusing off-table elements is the
correct end state rather than a temporary limitation.

In particular **Zn, Cd and Hg appear in NO replacement table** -- mercury was
explicitly DELETED by P-22.2.2 -- and such rings are named by P-69 organometallic
nomenclature instead. So the mercury cage refusing here is right; do NOT add a
``mercura`` prefix to make it name. (``data/hw_heteroatoms.HW_PREFIXES`` does carry
``Hg: 'mercura'``, which is how the spiro sibling used to spell it. That path now
refuses via this table.)

``HETEROATOM_PREFIXES`` is deliberately left at the 14 entries it had; the
remaining genuinely-cited Table 2.8 rows are added against their citations as a
separate change, because a wrong table entry ships a wrong NAME. Extension is
data-only -- add ``symbol: (stem, seniority_key)`` rows and nothing else in this
module moves.

Relationship to ``rules/skeletal_replacement.py``: that module is the ACYCLIC
P-15.4 chain namer (``2,5,8-trioxanonane``). Same nomenclature family, different
parent class and a different numbering source; they share no state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .lambda_convention import format_lambda_token, nonstandard_bonding_number


# Skeletal-replacement ('a') prefixes for von Baeyer / spiro ring systems, in
# IUPAC 2013 Table 2.8 / P-31.1.4.3.4 seniority order (the order heteroatoms are
# CITED in the name and the order they receive lowest locants):
# F>Cl>Br>I>O>S>Se>Te>N>P>As>Sb>Bi>Si>Ge>Sn>Pb>B. The integer is a relative sort
# key only; the strings are the replacement stems and are OPSIN-round-trip
# verified (e.g. heptasilabicyclo[2.2.1]heptane, heptagermabicyclo[2.2.1]heptane).
# Group-14/15 + B added in v23 Phase 5 so an all-heteroatom von Baeyer system is
# named via its hydride-replacement stem instead of silently dropping the
# heteroatoms (structure-loss safety).
#
# Halogens are absent on purpose: a standard-valence halogen cannot be a skeletal
# ring atom, and the lambda-convention cases (P-22.2.7.1 ``1lambda3-iodinane``)
# are monocycles handled by the Hantzsch-Widman path, not by this builder.
HETEROATOM_PREFIXES: Dict[str, Tuple[str, int]] = {
    'O': ('oxa', 1),
    'S': ('thia', 2),
    'Se': ('selena', 3),
    'Te': ('tellura', 4),
    'N': ('aza', 5),
    'P': ('phospha', 6),
    'As': ('arsa', 7),
    'Sb': ('stiba', 8),
    'Bi': ('bisma', 9),
    'Si': ('sila', 10),
    'Ge': ('germa', 11),
    'Sn': ('stanna', 12),
    'Pb': ('plumba', 13),
    'B': ('bora', 14),
}


@dataclass(frozen=True)
class ReplacementPrefix:
    """Result of ``build_replacement_prefix``.

    ``prefix``      the replacement block exactly as it is concatenated onto the
                    ring descriptor -- ``"3-oxa"``, ``"2,4-dioxa"``,
                    ``"5-oxa-3-sila"``, ``""`` when nothing is expressed. NO
                    trailing hyphen (P-23.3.1: the 'a'-prefix attaches directly
                    to the descriptor, ``2-oxabicyclo[2.2.2]octane``). Always
                    byte-identical to the legacy inline builder, INCLUDING its
                    malformed >20 multiplier fallback -- see ``unexpressed``.
    ``per_atom``    ``(atom_idx, morpheme)`` for every heteroatom the prefix
                    spells CORRECTLY, ascending by atom index. One entry per
                    atom, so a consumer can bind each spelled morpheme to the
                    single atom it claims.
    ``unexpressed`` skeletal non-carbon ring atoms ``prefix`` does not correctly
                    express, ascending. **Non-empty means the caller must
                    refuse.**

    ``per_atom`` indices and ``unexpressed`` are disjoint and together cover
    every non-carbon ring atom.
    """
    prefix: str
    per_atom: Tuple[Tuple[int, str], ...]
    unexpressed: Tuple[int, ...]


def vb_lambda_for_atom(mol, atom_idx: int) -> Optional[int]:
    """λ bonding number (P-14.1.2 / P-14.1.3; placement P-15.4.1.3; per-topology
    P-21.2.4 / P-22.2.7 / P-23.6 / P-24.8) for a ring 'a'-prefix atom, or None.

    Correction: earlier comments in this fix's history cited P-31.1.4.2 for the
    λ-convention itself. That section heading is "If there is a choice of names
    and numbering..." (BB:16633) -- it governs CHOICE, not the λ symbol. The
    nonstandard-bonding-number concept is P-14.1.2 (standard) / P-14.1.3
    (nonstandard); the symbol's placement (immediately after the locant, no
    hyphen) is P-15.4.1.3; and each parent-hydride topology has its own
    governing subsection: P-21.2.4 (acyclic), P-22.2.7 (monocyclic), P-23.6
    (von Baeyer ring 'a'-prefix -- the context THIS function serves), P-24.8
    (spiro).

    Refines the shared ``nonstandard_bonding_number`` with the SKELETAL-DEGREE
    rule that governs ring replacement nomenclature: when an 'a'-replacement name
    is parsed, each skeletal atom takes the LOWEST valid valence >= its skeletal
    degree. So a high-degree (bridgehead) heteroatom whose actual valence EQUALS
    that connectivity-forced value needs NO λ -- it is inferred, and an explicit λ
    there is redundant AND rejected (e.g. the bridgehead Te in
    heptatellurabicyclo[2.2.1]heptane has degree 3 and valence 4 = the lowest Te
    valence >= 3, so it must NOT carry λ4). λ IS cited only when the valence
    EXCEEDS the connectivity-forced value (the heteroatom carries extra hydrogen),
    e.g. a degree-2 ring S(IV): without the λ4 it would read as S(II) -- a
    different molecule. This is intentionally LOCAL to the ring 'a'-prefix path;
    the shared ``nonstandard_bonding_number`` stays standard-valence-relative for
    the parent-hydride/chain namers (SF6 -> lambda6-sulfane needs λ even though
    its degree forces valence 6).
    """
    lam = nonstandard_bonding_number(mol, atom_idx)
    if lam is None:
        return None
    from rdkit.Chem import GetPeriodicTable
    atom = mol.GetAtomWithIdx(atom_idx)
    valences = [v for v in GetPeriodicTable().GetValenceList(atom.GetAtomicNum())
                if v > 0]
    degree = atom.GetDegree()
    forced = min((v for v in valences if v >= degree), default=None)
    if forced is not None and lam == forced:
        return None  # connectivity-forced valence; inferred (no spurious λ)
    return lam


def build_replacement_prefix(
    mol, numbering: Dict[int, int], ring_atoms: Set[int],
) -> ReplacementPrefix:
    """Build the ring skeletal-replacement prefix, TOTALLY.

    Args:
        mol: RDKit Mol (may be a kekulized copy or a ring-only submol; indices
            must agree with ``numbering`` and ``ring_atoms``).
        numbering: atom index -> ring locant (1-based).
        ring_atoms: the skeletal ring atoms of the system being named.

    Returns:
        ``ReplacementPrefix``. ``prefix`` is byte-identical to the string the
        inline builder in ``polycyclic.py`` produced -- for EVERY input, not only
        the well-formed ones, because three PIN callers concatenate it and
        withholding a part there would turn a malformed name into a
        heteroatom-dropping (wrong-structure) one. ``unexpressed`` names the
        atoms that string does not correctly express.
    """
    # (locant, lambda, atom_idx) per element. Iterating a SORTED atom list makes
    # the traversal deterministic; the per-element sort below then makes the
    # output independent of it either way.
    by_element: Dict[str, List[Tuple[int, Optional[int], int]]] = {}
    unexpressed: List[int] = []

    for atom_idx in sorted(ring_atoms):
        symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
        if symbol == 'C':
            continue
        if symbol not in HETEROATOM_PREFIXES:
            # Off-table skeletal element: no morpheme exists to spell it, and the
            # ring stem counts it regardless -> the caller must refuse.
            unexpressed.append(atom_idx)
            continue
        if atom_idx not in numbering:
            # In-table element the numbering does not reach: no locant can be
            # cited, so it cannot be expressed either. Same silent-drop shape.
            unexpressed.append(atom_idx)
            continue
        by_element.setdefault(symbol, []).append(
            (numbering[atom_idx], vb_lambda_for_atom(mol, atom_idx), atom_idx))

    if not by_element:
        return ReplacementPrefix(prefix="", per_atom=(),
                                 unexpressed=tuple(sorted(unexpressed)))

    from ..assembly.naming_utils import SIMPLE_MULTIPLIERS

    parts: List[str] = []
    per_atom: List[Tuple[int, str]] = []
    # Cite elements in Table-2.8 seniority order, NOT ascending locant order
    # (``5-oxa-3-sila``: oxa precedes sila although sila holds the lower locant).
    for element in sorted(by_element, key=lambda s: HETEROATOM_PREFIXES[s][1]):
        entries = sorted(by_element[element], key=lambda e: e[0])
        stem = HETEROATOM_PREFIXES[element][0]
        locant_str = ','.join(
            format_lambda_token(loc, lam) for loc, lam, _idx in entries)
        count = len(entries)
        if count > 1:
            # ``SIMPLE_MULTIPLIERS`` stops at 20 and the legacy fallback spelled
            # the bare integer -- ``"22sila"``, not a word. That IS reachable
            # today: MAX_CAGE_ATOMS is 40, and a 22-Si bicyclo[10.9.1] currently
            # yields ``1,…,22-22sila``. ``prefix`` keeps the legacy string (the
            # PIN callers must not silently lose 22 skeletal atoms instead), but
            # the atoms are reported unexpressed so the general tier refuses.
            mult = SIMPLE_MULTIPLIERS.get(count)
            parts.append(f"{locant_str}-{mult or str(count)}{stem}")
            if mult is None:
                unexpressed.extend(idx for _loc, _lam, idx in entries)
                continue
        else:
            parts.append(f"{locant_str}-{stem}")
        per_atom.extend((idx, stem) for _loc, _lam, idx in entries)

    # The internal '-' between multiple replacement terms is kept by the join;
    # emit NO trailing '-' (callers concatenate prefix + descriptor directly).
    return ReplacementPrefix(
        prefix='-'.join(parts),
        per_atom=tuple(sorted(per_atom)),
        unexpressed=tuple(sorted(unexpressed)),
    )
