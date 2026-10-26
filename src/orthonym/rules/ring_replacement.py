"""Total skeletal ('a') replacement-prefix construction for RING systems.

 (von Baeyer) and (spiro): a skeletal non-carbon
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
mercury ring. (name -> structure round trip) fails OPEN when the OPSIN
jar is absent -- a supported mode -- so nothing downstream caught it.

The primitive below returns the prefix *together with* the information a caller
needs to fail closed:

    ``unexpressed`` -- every skeletal non-carbon ring atom the emitted prefix
    does not correctly express: an off-table element or an atom the numbering does
    not reach (no locant to cite).

**Caller contract: refuse when ``unexpressed`` is non-empty.** Never emit a name
whose ring stem counts an atom that no morpheme in it spells. Both universal
analyzers (``vonbaeyer_universal.analyze_cage_universal`` and
``analyze_spiro_universal``) apply exactly this rule, so the two siblings no
longer disagree: before this module, the cage path *dropped* the atom and the
spiro path *invented* a morpheme for it (``polycyclic_bridged``'s
``get_heteroatom_prefix`` falls back to ``symbol.lower + 'a'``, spelling
``3-znaspiro[5.5]undecane``).

The table is CLOSED, so fail-closed is the only sound design
-----------------------------------------------------------
The Blue Book's replacement-prefix set is Table 1.5, [BBv2:6436-6443])
and it is a fixed list, not a generative rule. verbatim: *"Nondetachable
prefixes, called 'a' prefixes, are used to designate the replacing skeletal atoms
with their standard bonding number. Those related to these recommendations are
listed in Table 1.5."* The wording is restrictive, so there is no way to derive a
prefix for an arbitrary element: "element-blind" replacement is genuinely
unachievable and refusing off-table elements is the correct end state rather than a
temporary limitation.

(Earlier revisions of this docstring cited " / Table 2.8" for the
replacement set. Table 2.8 is *"Retained names of heterocyclic parent ring
components"* [BBv2:11511] -- an unrelated table. Corrected against the book.)

In particular **Zn, Cd and Hg appear in NO replacement table** -- mercury was
explicitly DELETED by -- and such rings are named by organometallic
nomenclature instead. So the mercury cage refusing here is right; do NOT add a
``mercura`` prefix to make it name. (``data/organometallics.METALLACYCLE_A_PREFIX``
is the legitimate home of Hg/Zn/Cd.)

Three orders, three different element SETS -- do not borrow across them
----------------------------------------------------------------------
This is the trap that governs which of Table 1.5's 25 rows this module may emit.
The Blue Book gives three seniority orders over 'a'-prefix elements and they do
NOT cover the same elements:

=============== =========================================== ======== ==========
rule governs elements citation
=============== =========================================== ======== ==========
       general / chains, "naming and numbering" **25** [BBv2:6446]
         citation order INSIDE a von Baeyer name **22** [BBv2:9765]
       numbering seniority when there is a CHOICE **18** [BBv2:9789]
=============== =========================================== ======== ==========

 drops ``At``, ``Po`` and ``C``; drops those three **and** the
four halogens. Emitting a replacement prefix needs BOTH a citation position and a
numbering rank, so the set this module may spell is the intersection --
's 18 elements -- and the ring halogens admitted by
``lambda_convention.FUSED_RING_LAMBDA_HALOGENS`` (iodine, since lane L3: its citation
position is 's, its numbering rank the one of that order, ahead of oxygen -- the order the
book itself gives for a numbering choice among heteroatoms with the halogens first, (b)
(the Blue Book: "low locants are assigned in accord with the following decreasing seniority order
of heteroatoms: F > Cl > Br > I > O > S >...") and (:8806: "the locant '1' is given to the
heteroatom first cited in the order of seniority given above" -- and a name that carries it is recorded
as not the PIN, since itself gives the halogens no rank), which together are exactly
``HETEROATOM_PREFIXES`` below. The remaining rows of Table 1.5 (six) stay in
``TABLE_1_5`` (the book's table is recorded whole) but are listed in ``VB_INADMISSIBLE``
with the reason, and they fail closed through the ordinary ``unexpressed`` contract.
Borrowing a position for an element the von Baeyer rules never rank would be
inventing a rule; a halogen is admitted by a measurement (a panel whose names read back),
not by a reading.

Relationship to ``rules/skeletal_replacement.py``: that module is the ACYCLIC
 chain namer (``2,5,8-trioxanonane``). Same nomenclature family, different
parent class and a different numbering source; they share no state. Its own
element table is deliberately NOT extended alongside this one: whether skeletal
replacement or the substitutive parent hydride (``alumane`` / ``gallane`` /
``indigane`` / ``thallane``, Table 2.1 [BBv2:7924-7930]) is the PIN for a
Group-13 atom embedded in a CHAIN is a selection question this module does
not answer, and the chain path fails closed until it is answered.

Relationship to ``data/hw_heteroatoms.py``: that is **Table 2.4**
, the Hantzsch-Widman monocycle context, which spells two of the
same elements DIFFERENTLY on purpose -- ``aluma`` "(not alumina)" [BBv2:8245] and
``indiga`` "(not inda)", under a footnote reading "Compare with Table 1.5"
[BBv2:8250]. A prefix is therefore a function of *(element, nomenclature
context)*, never of the element alone. Do not unify the two tables.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..data.chain_names import multiplying_prefix
from .lambda_convention import (
    FUSED_RING_LAMBDA_HALOGENS, format_lambda_token, nonstandard_bonding_number,
    ring_halogen_ion, ring_halogen_ion_lambda,
)

# ---------------------------------------------------------------------------
# Blue Book Table 1.5, COMPLETE -- element -> (prefix, STANDARD bonding number).
#
# Transcribed verbatim from [BBv2:6436-6443]. The table is printed as a 5x5 grid
# whose column headers ARE the standard bonding numbers, so each row below carries
# the number from its own column -- the second value is Blue Book data, never a
# periodic-table default.: 'a' prefixes designate the replacing
# skeletal atoms "with their standard bonding number", which is what makes an
# atom whose actual bonding number differs need the lambda convention
#, e.g. the PIN ``6lambda5-phosphaspiro[4.5]decane`` [BBv2:6452]).
#
# This constant is the BOOK'S TABLE, not the set this module may emit -- see
# ``HETEROATOM_PREFIXES``. It exists so every one of the 25 rows has an explicit,
# cited disposition instead of being silently absent.
TABLE_1_5: Dict[str, Tuple[str, int]] = {
    # bonding number 1
    'F': ('fluora', 1), 'Cl': ('chlora', 1), 'Br': ('broma', 1),
    'I': ('ioda', 1), 'At': ('astata', 1),
    # bonding number 2
    'O': ('oxa', 2), 'S': ('thia', 2), 'Se': ('selena', 2),
    'Te': ('tellura', 2), 'Po': ('polona', 2),
    # bonding number 3
    'N': ('aza', 3), 'P': ('phospha', 3), 'As': ('arsa', 3),
    'Sb': ('stiba', 3), 'Bi': ('bisma', 3),
    # bonding number 4
    'C': ('carba', 4), 'Si': ('sila', 4), 'Ge': ('germa', 4),
    'Sn': ('stanna', 4), 'Pb': ('plumba', 4),
    # bonding number 3
    'B': ('bora', 3), 'Al': ('alumina', 3), 'Ga': ('galla', 3),
    'In': ('inda', 3), 'Tl': ('thalla', 3),
}

# [BBv2:9765], verbatim: the replacement prefixes are "cited in the
# order: F > Cl > Br > I > O > S > Se > Te > N > P > As > Sb > Bi > Si > Ge > Sn >
# Pb > B > Al > Ga > In > Tl". TWENTY-TWO elements -- At, Po and C have no von
# Baeyer citation position at all. (Cross-check: Table 2.4, the Hantzsch-Widman
# table [BBv2:8236-8248], is "in decreasing order of seniority" over the SAME 22
# elements in the SAME sequence, which is why the HW-ordered spiro sites agree
# with this one for every element either can reach.)
VB_CITATION_ORDER: Tuple[str, ...] = (
    'F', 'Cl', 'Br', 'I',
    'O', 'S', 'Se', 'Te',
    'N', 'P', 'As', 'Sb', 'Bi',
    'Si', 'Ge', 'Sn', 'Pb',
    'B', 'Al', 'Ga', 'In', 'Tl',
)

# [BBv2:9789], verbatim: "If there is still a choice, low locants are
# assigned in accord with the decreasing seniority order of heteroatoms O > S >
# Se > Te > N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In > Tl."
# EIGHTEEN elements: the four halogens are absent here even though ranks
# them for citation, so a halogen replacement has no sanctioned numbering rank.
# This sequence is ``VB_CITATION_ORDER`` minus the halogens, so ONE integer key
# can serve both the citation order and the numbering rank.
VB_NUMBERING_SENIORITY: Tuple[str, ...] = VB_CITATION_ORDER[4:]

# Why the other seven Table 1.5 rows are NOT emitted in a ring context. Each is a
# missing Blue Book rule, not an unimplemented feature: spelling one would mean
# choosing a citation position or a numbering rank the book does not give.
VB_INADMISSIBLE: Dict[str, str] = {
    'At': "no P-23.3.1 citation position and no P-23.3.2.2 numbering rank; "
          "'astata' appears only in Table 1.5, Appendix 1 and a mononuclear-"
          "hydride aside [BBv2:7915] -- there is no ring example in the book",
    'Po': "no P-23.3.1 citation position and no P-23.3.2.2 numbering rank; "
          "'polona' likewise has no ring example anywhere in the book",
    'C': "no P-23.3.1 citation position and no P-23.3.2.2 numbering rank, and "
         "P-21.1.1.1 [BBv2:7915] rules the morpheme out directly: \"The name "
         "'carbane' has never been used in place of methane; it is not "
         "recommended.\" Replacing a skeletal carbon by carbon is also a no-op "
         "in a hydrocarbon ring; 'carba' belongs to carbaborane nomenclature",
    'F': "P-23.3.1 gives a citation position but P-23.3.2.2 gives NO numbering "
         "rank, and this module has not been measured for the element: a halogen "
         "is admitted by listing it in ``lambda_convention.FUSED_RING_LAMBDA_HALOGENS`` "
         "(iodine is), after a panel showing its names read back. A skeletal RING "
         "halogen always has a nonstandard bonding number (standard is 1, ring "
         "degree is >= 2), so P-15.4.1.3 makes its lambda mandatory and "
         "``vb_lambda_for_atom`` cites it for an admitted halogen",
    'Cl': "see 'F'", 'Br': "see 'F'", 'I': "see 'F'",
}
# iodine is admitted (HETEROATOM_PREFIXES below), so it has no refusal to explain
VB_INADMISSIBLE = {element: reason for element, reason in VB_INADMISSIBLE.items()
                   if element not in FUSED_RING_LAMBDA_HALOGENS}

# The halogens that are not admitted: ranks them for citation, gives no numbering rank
# (an admitted halogen takes the rank of the citation order, below), the book prints no ring replacement
# name with one (its halogen ring examples are Hantzsch-Widman and fusion names: ``1H-1lambda3-iodole``,
# ``3H-3lambda3,2,4-benziodadioxepine`` (PIN) [BBv2:14557], ``lambda3-iodane`` [BBv2:7968]), and a
# panel has to show the element's names read back before it is listed.

# The elements this module MAY spell: (citation) INTERSECT
# (numbering) = 's eighteen. The integer is a relative sort key that
# doubles as both orders (see ``VB_NUMBERING_SENIORITY``); the strings come from
# ``TABLE_1_5`` so a spelling cannot drift from the book. The O..B rows keep the
# keys 1..14 they have always had, so extending the table did not renumber -- and
# could not reorder -- any citation that already shipped. Stems are OPSIN
# round-trip verified (heptasilabicyclo[2.2.1]heptane,
# heptagermabicyclo[2.2.1]heptane, 2-aluminaspiro[5.5]undecane).
#
# THE ADMITTED RING HALOGENS (iodine; ``lambda_convention.FUSED_RING_LAMBDA_HALOGENS`` is the one home
# of the admission, read by the benzoheterocycle producer and the mancude-parent valence as well). The
# book gives a halogen its citation position, "Heterogeneous heterocyclic von Baeyer systems"
# (the Blue Book): the 'a' prefixes are "cited in the order: F > Cl > Br > I > O > S >...", and
# the same sequence heads Table 2.4 (:8236) and the heteroatom order of (:12139). Its
# numbering tie-break, (:9789), lists eighteen elements and no halogen, so an admitted
# halogen takes the rank of the citation order: ahead of oxygen, the keys -4 (F) to -1 (I) that
# ``hw_heteroatoms.HETEROATOM_PRIORITY`` gives the halogens, so no key of the other elements moves.
# The rank decides a numbering only where the locant SET of ties, and a name that rests on
# it is recorded as not the PIN (``build_replacement_prefix``): the book prints no von Baeyer, spiro or
# monocyclic replacement name with a halogen.
_HALOGEN_RANK_OFFSET = len(VB_CITATION_ORDER) - len(VB_NUMBERING_SENIORITY)
HETEROATOM_PREFIXES: Dict[str, Tuple[str, int]] = {
    **{element: (TABLE_1_5[element][0], VB_CITATION_ORDER.index(element) - _HALOGEN_RANK_OFFSET)
       for element in VB_CITATION_ORDER[:_HALOGEN_RANK_OFFSET]
       if element in FUSED_RING_LAMBDA_HALOGENS},
    **{element: (TABLE_1_5[element][0], rank)
       for rank, element in enumerate(VB_NUMBERING_SENIORITY, start=1)},
}


@dataclass(frozen=True)
class ReplacementPrefix:
    """Result of ``build_replacement_prefix``.

    ``prefix`` the replacement block exactly as it is concatenated onto the
                    ring descriptor -- ``"3-oxa"``, ``"2,4-dioxa"``,
                    ``"5-oxa-3-sila"``, ``""`` when nothing is expressed. NO
                    trailing hyphen: the 'a'-prefix attaches directly
                    to the descriptor, ``2-oxabicyclo[2.2.2]octane``). Always
                    byte-identical to the legacy inline builder, except that a count
                    past 20 is the composed multiplying prefix (``docosasila``),
                    never the digits it used to spell (``22sila``).
    ``per_atom`` ``(atom_idx, morpheme)`` for every heteroatom the prefix
                    spells CORRECTLY, ascending by atom index. One entry per
                    atom, so a consumer can bind each spelled morpheme to the
                    single atom it claims.
    ``unexpressed`` skeletal non-carbon ring atoms ``prefix`` does not correctly
                    express, ascending. **Non-empty means the caller must
                    refuse.**

    ``preselected_only``
                    True when the skeleton is a homogeneous noncarbon one (every
                    skeletal atom the same noncarbon element) with more than 20 atoms.
                    The 'a' name ``prefix`` gives is then a valid name for general
                    nomenclature and never the preferred one:
                    "Heterocyclic noncarbon von Baeyer and spiro compounds"
                    (the Blue Book) "Preselected names for von Baeyer compounds
                    and for spiro compounds having only monocyclic components and
                    consisting entirely of heteroatoms of the same kind are names
                    formed by citing the appropriate prefix, such as 'bicyclo',
                    'spiro', etc., and descriptor... followed by a numerical prefix
                    for the total number of heteroatoms and the name of the
                    mononuclear parent hydride; alternative names for use in general
                    nomenclature are formed by skeletal replacement ('a')
                    nomenclature", e.g. ``bicyclo[4.2.1]nonasilane`` (:9810). That
                    name is not built here, so **a caller that claims a PIN must
                    refuse when this is set**, as it refused before the count past 20
                    had a spelled word. (The same-kind cages of 20 atoms or fewer keep
                    the 'a' name at the PIN callers: a separate, recorded lead.)

    ``per_atom`` indices and ``unexpressed`` are disjoint and together cover
    every non-carbon ring atom.
    """
    prefix: str
    per_atom: Tuple[Tuple[int, str], ...]
    unexpressed: Tuple[int, ...]
    preselected_only: bool = False


def vb_lambda_for_atom(mol, atom_idx: int) -> Optional[int]:
    """λ bonding number /; placement; per-topology
     / / / for a ring 'a'-prefix atom, or None.

    Correction: earlier comments in this fix's history cited for the
    λ-convention itself. That section heading is "If there is a choice of names
    and numbering..." (the Blue Book) -- it governs CHOICE, not the λ symbol. The
    nonstandard-bonding-number concept is (standard) /
    (nonstandard); the symbol's placement (immediately after the locant, no
    hyphen) is; and each parent-hydride topology has its own
    governing subsection: (acyclic), (monocyclic),
    (von Baeyer ring 'a'-prefix -- the context THIS function serves),
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
    if ring_halogen_ion(mol, atom_idx) is not None:
        # an ion is named from the hydride whose ring atom has one hydrogen more than it has bonds, the
        # anion 'ide', the cation 'ylium': the lambda of that hydride, 3 for two ring bonds
        # (``lambda_convention.ring_halogen_ion_lambda``)
        return ring_halogen_ion_lambda(mol, atom_idx)
    lam = nonstandard_bonding_number(mol, atom_idx)
    if lam is None:
        return None
    from rdkit.Chem import GetPeriodicTable
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetSymbol() in FUSED_RING_LAMBDA_HALOGENS:
        # a ring halogen has the standard bonding number 1 and a ring degree of two or more, so its
        # bonding number is never the standard one and (the Blue Book: "The symbol
        # lambda-n is used to describe heteroatoms having nonstandard bonding numbers... placed
        # immediately after the locant") has it cited whatever the degree forces: '1lambda3-iodane'
        # (:7968), '1lambda3-iodinane (PIN)',:9158), '2lambda3-iodabicyclo[2.2.1]heptane'
        # read by OPSIN as the name without the symbol, the same structure
        return lam
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
        if symbol in FUSED_RING_LAMBDA_HALOGENS and not mol.GetAtomWithIdx(atom_idx).IsInRing():
            # a halogen is a RING 'a' atom here (the module is the ring one): in a chain its
            # hypervalent state is the parent hydride 'lambda3-iodane' of (the Blue Book),
            # which ``assembly/universal_halane`` builds
            unexpressed.append(atom_idx)
            continue
        if (symbol in FUSED_RING_LAMBDA_HALOGENS and mol.GetAtomWithIdx(atom_idx).GetFormalCharge()
                and ring_halogen_ion(mol, atom_idx) is None):
            # a ring halogen ion that is not the simple iodonium or iodanuide centre of
            # ``ring_halogen_ion`` has no name this builder can spell; the neutral 'ioda' name would
            # denote a different species. The simple ions are spelled here as their ring atom, with the
            # lambda of their hydride, and carry their charge in the ending of the caller
            # ('ylium', 'ide'
            unexpressed.append(atom_idx)
            continue
        if symbol not in HETEROATOM_PREFIXES:
            # Not spellable in a ring context: either genuinely off Table 1.5 (no
            # morpheme exists at all -- Zn/Cd/Hg/Fe/...) or one of the seven
            # Table 1.5 rows in ``VB_INADMISSIBLE``, which have a morpheme but no
            # von Baeyer citation position and/or no numbering rank. Both are the
            # same failure from the caller's side: the ring stem counts the atom
            # regardless, so the caller must refuse.
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

    parts: List[str] = []
    per_atom: List[Tuple[int, str]] = []
    # Cite elements in seniority order, NOT ascending locant order
    # (``5-oxa-3-sila``: oxa precedes sila although sila holds the lower locant).
    for element in sorted(by_element, key=lambda s: HETEROATOM_PREFIXES[s][1]):
        entries = sorted(by_element[element], key=lambda e: e[0])
        stem = HETEROATOM_PREFIXES[element][0]
        locant_str = ','.join(
            format_lambda_token(loc, lam) for loc, lam, _idx in entries)
        count = len(entries)
        if count > 1:
            # Table 1.4 stops at 20; past that the multiplying prefix is composed
            #, 'Derivation of basic numerical terms', the Blue Book),
            # so a 22-Si bicyclo[10.9.1] (MAX_CAGE_ATOMS is 40) is ``docosasila``,
            # never the digits ``22sila``.
            parts.append(f"{locant_str}-{multiplying_prefix(count)}{stem}")
        else:
            parts.append(f"{locant_str}-{stem}")
        per_atom.extend((idx, stem) for _loc, _lam, idx in entries)

    if any(element in FUSED_RING_LAMBDA_HALOGENS for element in by_element):
        # valid, read back by the callers, never labelled the PIN: the numbering rank of a halogen is
        # the book's citation order and not a rule of, and the book prints no replacement
        # name of a von Baeyer, spiro or monocyclic system with a halogen (name-scoped record)
        from ..metrics.provenance import record_non_pin_fragment
        record_non_pin_fragment('-'.join(parts))

    # (the Blue Book): a skeleton of one noncarbon element has a
    # preselected name this module does not build; see ``preselected_only``.
    homogeneous_noncarbon = (
        len(by_element) == 1 and not unexpressed and len(ring_atoms) > 20
        and all(mol.GetAtomWithIdx(i).GetSymbol() != 'C' for i in ring_atoms))

    # The internal '-' between multiple replacement terms is kept by the join;
    # emit NO trailing '-' (callers concatenate prefix + descriptor directly).
    return ReplacementPrefix(
        prefix='-'.join(parts),
        per_atom=tuple(sorted(per_atom)),
        unexpressed=tuple(sorted(unexpressed)),
        preselected_only=homogeneous_noncarbon,
    )
