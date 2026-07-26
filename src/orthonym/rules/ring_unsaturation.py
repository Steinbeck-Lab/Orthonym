"""Ring-unsaturation locant rendering for von Baeyer / spiro parents.

One producer for BOTH bond orders, replacing two inline blocks that disagreed.

Why this module exists
----------------------
``vonbaeyer_universal.analyze_cage_universal`` recomputed the double-bond locants
into the von-Baeyer COMPOUND form -- ``f"{lo}({hi})"`` when the bond's two atoms
are not consecutively numbered, e.g. octalin's ``1(6)`` -- but left the
triple-bond locants exactly as ``polycyclic.get_polycyclic_unsaturation`` produced
them: ``min(loc1, loc2)``, a bare lower locant with no compound form and **no
guard**. A cage carrying a non-consecutively-numbered triple bond would therefore
have emitted a ``-yne`` locant denoting a DIFFERENT bond than the one present.
The spiro sibling refused that case; the cage path was silent. Same computation,
two implementations, one of them unsound -- so it becomes one function used by
both.

``polycyclic.get_polycyclic_unsaturation`` is deliberately NOT changed: it is
shared with the PIN von-Baeyer stack, whose emitted strings are gold-locked, and
that stack consumes bare ints. The soundness fix belongs where the general tier
consumes the locants.

The compound locant is a DOUBLE-BOND rule (P-31.1.4.2(1))
---------------------------------------------------------
P-31.1.4.2(1) is worded for double bonds only: "A compound locant is used for *a
double bond* if the locants of the atoms at each end of the bond do not differ by
a value of one. When a compound locant is required, the higher locant is cited in
parentheses." The same double-bond-only wording appears in P-31.1.6.1(3) and
P-31.1.7.2. Every ``-yne`` example in the Blue Book carries a PLAIN locant --
``bicyclo[14.3.1]icosa-11,13,18-trien-2-yne``,
``bicyclo[11.3.1]heptadec-2-en-11-yne`` -- with the parentheses in those very
names appearing only on the ``-ene`` component. So the bare lower locant is
CORRECT for a triple bond on every structure the Blue Book covers, and there is
no sanctioned ``x(y)`` form to fall back on.

Why the non-consecutive branch refuses instead of citing the bare locant
------------------------------------------------------------------------
A non-consecutively-numbered triple bond is *provably unreachable* for standard
bonding numbers: P-23.1.2 (a bridge connects two bridgeheads) makes von Baeyer
numbering a concatenation of runs each ending at a bridgehead, so every
non-consecutively-numbered bond is incident to a bridgehead; P-23.1.1 defines a
bridgehead as having >= 3 skeletal neighbours; a C(triple)C carbon has exactly 2
sigma bonds. A bridgehead therefore can never be a triple-bond terminus.

Reaching that state consequently means the UPSTREAM NUMBERING is wrong -- and
emitting the bare lower locant there would be a genuine wrong-structure emission,
because a reader parses ``-8-yne`` as the 8-9 bond. So the branch fails closed:
it is an assertion against a numbering bug, not a nomenclature fallback. The
structural claim is tested (``test_ring_unsaturation.py``) rather than assumed.

One documented, NOT ESTABLISHED hole: a lambda-n heteroatom could in principle be
3-connected AND triply bonded, which would make a non-consecutive yne structurally
possible. The Blue Book gives no rule for citing that bond, so the fail-closed
branch is what covers it -- correctly, by abstaining. ``_YNE_COMPOSITE_ALLOWED``
is the one-line flip should such a rule ever be cited; it must stay ``False``
until then, since no compound ``-yne`` locant exists in the literature.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rdkit import Chem


#: Whether a non-consecutively-numbered triple bond may be cited with the compound
#: ``x(y)`` locant. Must stay False: P-31.1.4.2(1) grants the compound locant to
#: DOUBLE bonds only and no compound ``-yne`` locant exists in the Blue Book, so
#: emitting one would be the first such name in existence. Kept as a named flag
#: only because it is the single line to change if the NOT-ESTABLISHED lambda-n
#: case (see the module docstring) is ever given a rule.
_YNE_COMPOSITE_ALLOWED = False


@dataclass(frozen=True)
class RingUnsaturation:
    """Ring double/triple bonds as both raw locant pairs and display strings.

    ``double_pairs`` / ``triple_pairs``
        sorted ``(low, high)`` locant pairs. The general engine's oxo/ene valence
        guard consumes the double pairs in exactly this form: both endpoints of a
        ring double bond are termini a ``=O``/``=N`` cannot share.
    ``double_locants`` / ``triple_locants``
        display strings in the same order -- ``str(lo)`` when ``hi == lo + 1``,
        else the compound ``f"{lo}({hi})"``. ``triple_locants`` is always plain:
        a non-consecutive triple bond refuses before this is built.
    """
    double_pairs: Tuple[Tuple[int, int], ...]
    triple_pairs: Tuple[Tuple[int, int], ...]
    double_locants: Tuple[str, ...]
    triple_locants: Tuple[str, ...]


def _locant(lo: int, hi: int) -> str:
    """P-31.1.4.2(1): a ring DOUBLE bond whose end locants do not differ by one is
    cited with the compound locant ``lo(hi)`` -- the higher locant in parentheses
    (octalin ``1(6)``); otherwise the lower locant alone."""
    return str(lo) if hi == lo + 1 else f"{lo}({hi})"


def render_ring_unsaturation(
    mol, numbering: Dict[int, int],
) -> Optional[RingUnsaturation]:
    """Render every ring multiple bond of ``mol`` on the locants in ``numbering``.

    Only bonds whose BOTH atoms carry a locant are considered -- an exocyclic or
    substituent multiple bond is not ring unsaturation and is expressed elsewhere.

    Args:
        mol: RDKit Mol, already kekulized if the ring system is mancude (an
            aromatic bond is neither DOUBLE nor TRIPLE and would be skipped).
        numbering: atom index -> ring locant (1-based).

    Returns:
        ``RingUnsaturation``, or ``None`` to refuse -- currently only for a
        non-consecutively-numbered triple bond (see the module docstring).
    """
    double_pairs: List[Tuple[int, int]] = []
    triple_pairs: List[Tuple[int, int]] = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i not in numbering or j not in numbering:
            continue
        lo, hi = sorted((numbering[i], numbering[j]))
        bt = bond.GetBondType()
        if bt == Chem.BondType.DOUBLE:
            double_pairs.append((lo, hi))
        elif bt == Chem.BondType.TRIPLE:
            if hi != lo + 1 and not _YNE_COMPOSITE_ALLOWED:
                return None
            triple_pairs.append((lo, hi))

    double_pairs.sort()
    triple_pairs.sort()
    return RingUnsaturation(
        double_pairs=tuple(double_pairs),
        triple_pairs=tuple(triple_pairs),
        double_locants=tuple(_locant(lo, hi) for lo, hi in double_pairs),
        triple_locants=tuple(_locant(lo, hi) for lo, hi in triple_pairs),
    )
