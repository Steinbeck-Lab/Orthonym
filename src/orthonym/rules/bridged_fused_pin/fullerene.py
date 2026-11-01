"""The fused parent of a (C60-Ih)[5,6]fullerene with at most one addend on a [6,6] bond (lane J1).

 Systematic names (the Blue Book): "(C60-Ih)[5,6]fullerene (PIN)"; NUMBERING OF
FULLERENES (:15336): the systematic numbering is the CAS one, the table of ``data/fullerene_c60``.
 Fullerenes and modified fullerenes ortho fused to organic rings or ring systems (:15464-15480):
"Organic ring systems, including monocyclic rings... are always cited as prefixes to the name of the
fullerene or modified fullerene which is always the parent component. Each system retains its own name
and numbering, both for indicating fusion sites and for indicating positions of substitution. The
fullerene or modified fullerene locants are always unprimed, and primes are added to the fused organic ring
or ring systems... The fusion is described by citing the primed locants of the organic ring or ring
system component and the unprimed locants of the fullerene or modified fullerene in that order, enclosed
in brackets and separated by a colon. Locants for monocyclic hydrocarbons are omitted in preferred IUPAC
names." The book's names: '3'H-cyclopropa[1,9](C60-Ih)[5,6]fullerene (PIN)', '[1,3]dioxolo[4',5':1,9]
(C60-Ih)[5,6]fullerene (PIN)', 'tris(oxireno)[2',3':1,9;2'',3'':2,12;2''',3'':7,8](C60-Ih)[5,6]fullerene
(PIN)'. "After fusion, nonfullerene components other than alicyclic bi- and polycyclic ring systems have the
maximum number of noncumulative double bonds and indicated hydrogen is cited as needed" and "the fusion
bond is always a single bond and the fusion atoms cannot accept an 'exo' double bond": the two cage atoms of
the fusion bond have four ring bonds and carry no double bond (``hydro.hydro_state`` finds them
ineligible), so the existing indicated-hydrogen, hydro-prefix and suffix machinery names the rest.

The skeleton is a fused parent for ``selection.fused_split`` (and ``parents.fused_parent`` for its
numberings) at the best-effort tier only: the name is read back by ``validation.fullerene_read_back``, the
engine's own reader, because OPSIN reads no fullerene name, and ``record_non_pin_fragment`` keeps it from
every PIN tier. The numberings are the template numbering composed with the symmetries of the cage that put the
addend's bond on 1,9 then chooses among them), the attached ring's numbering is its own
(``fusion_components.mono_numberings`` for a heteromonocycle; a monocyclic hydrocarbon puts its fusion atoms
on 1',2'), and the descriptor cites the attached locants in the order of the fullerene locants 1,9, as low as
the numbering allows, the Blue Book: "These numbers are chosen to be as low as is
consistent with the numbering of the compound").
"""
from functools import lru_cache
from typing import Dict, Optional, Set, Tuple

from rdkit import Chem

from ...data.fullerene_c60 import C60_NAME
from ...perception.fullerene import C60Adduct, c60_adduct
from . import fusion_components as fc
from .parents import FusedParent
from .selection import Split, _standard_bonding, ring_unsaturation

#: source of a FusedParent made here
SOURCE = "fullerene_template"


def tier_open() -> bool:
    """True in a best-effort request (``provenance.best_effort_request_ctx``, the discriminator the von Baeyer
    ceilings read) and nowhere else: a PIN, valid or complete request leaves the cage to the fast decline."""
    from ...metrics.provenance import best_effort_request_ctx
    return best_effort_request_ctx.get() is True


@lru_cache(maxsize=512)
def _component(elements: Tuple[int, ...]):
    """``(prefix, numberings)`` of the attached monocycle whose atoms, from one cage atom round to the other,
    have the atomic numbers ``elements`` (both ends are the carbon atoms of the fusion bond), or None.

    A heteromonocycle is a Hantzsch-Widman or retained component (``fusion_components``) numbered by its own
    rule, its prefix the component's attached form ('pyrrolo', '[1,3]dioxolo'); a monocyclic
    hydrocarbon is 'benzo' or 'cyclo...a', the Blue Book) and numbers its fusion atoms
    1',2'. Each numbering is a tuple of locant numbers by position."""
    n = len(elements)
    if not 3 <= n <= 7:
        return None
    rw = Chem.RWMol()
    for z in elements:
        rw.AddAtom(Chem.Atom(z))
    for i in range(n):
        rw.AddBond(i, (i + 1) % n, Chem.BondType.SINGLE)
    km = rw.GetMol()
    km.UpdatePropertyCache(strict=False)
    ring = frozenset(range(n))
    if all(z == 6 for z in elements):
        comp = fc.Component("carbo", f"cyclo{n}", frozenset([0]), ring, ())
        prefix = fc.prefix_form(comp)
        if prefix is None:
            return None
        first = [0] * n                      # fusion atoms 1', 2', then the rest in ring order from 2'
        first[0], first[n - 1] = 1, 2
        second = [0] * n
        second[n - 1], second[0] = 1, 2
        for j in range(1, n - 1):
            first[n - 1 - j] = 2 + j
            second[j] = 2 + j
        return prefix, (tuple(first), tuple(second))
    nums = fc.mono_numberings(km, ring, hw=True)
    names = {fc.hetero_mono_name(km, ring, num) for num in nums}
    if not nums or len(names) != 1 or None in names:
        return None
    comp = fc.Component("hetero", names.pop(), frozenset([0]), ring, ())
    prefix = fc.prefix_form(comp)
    if prefix is None:
        return None
    return prefix, tuple(tuple(num[i] for i in range(n)) for num in nums)


class _Reading:
    """The parent of one C60 adduct: its name and every numbering (cage locants are ints, the attached ring's
    are primed strings)."""
    __slots__ = ("name", "numberings", "adduct")

    def __init__(self, name: str, numberings: Tuple[Dict[int, object], ...], adduct: C60Adduct):
        self.name, self.numberings, self.adduct = name, numberings, adduct


def _reading(mol, residual: Set[int]) -> Optional[_Reading]:
    """The parent reading of the ring system ``residual`` of ``mol``, or None when it is not a C60 adduct of the
    scope (``perception.fullerene.c60_adduct``) or its attached ring has no component name here."""
    adduct = c60_adduct(mol)
    if adduct is None or set(residual) != set(adduct.cage) | set(adduct.ring):
        return None
    if not adduct.ring:
        return _Reading(C60_NAME, tuple(dict(m) for m in adduct.locants), adduct)
    n = len(adduct.ring) + 2
    around = (adduct.sp3[0],) + adduct.ring + (adduct.sp3[1],)
    got = _component(tuple(mol.GetAtomWithIdx(a).GetAtomicNum() for a in around))
    if got is None:
        return None
    prefix, rhos = got
    carbocycle = all(mol.GetAtomWithIdx(a).GetAtomicNum() == 6 for a in around)
    options = []
    for phi in adduct.locants:
        for rho in rhos:
            # the attached locants in the order of the cage locants 1, 9
            pos1 = 0 if phi[adduct.sp3[0]] == 1 else n - 1
            key = (rho[pos1], rho[n - 1 - pos1])
            options.append((key, phi, rho))
    if not carbocycle:
        best = min(k for k, _, _ in options)
        options = [o for o in options if o[0] == best]
        descriptor = f"[{best[0]}',{best[1]}':1,9]"
    else:
        descriptor = "[1,9]"
    numberings = []
    for _key, phi, rho in options:
        a2l: Dict[int, object] = dict(phi)
        for i, atom in enumerate(adduct.ring, start=1):
            a2l[atom] = f"{rho[i]}'"
        numberings.append(a2l)
    return _Reading(f"{prefix}{descriptor}{C60_NAME}", tuple(numberings), adduct)


def fullerene_split(mol, system: Set[int]) -> Optional[Split]:
    """The one-split reading of ``system`` as a C60 fused parent with no bridge, or None
    (``selection.fused_split`` asks first). Best-effort requests only; every call is None elsewhere."""
    if len(system) < 60 or not tier_open():
        return None
    system = set(system)
    reading = _reading(mol, system)
    if reading is None or not _standard_bonding(mol, system):
        return None
    unsat = ring_unsaturation(mol, system)
    if unsat is None:
        return None
    from ...metrics.provenance import record_non_pin_fragment
    # no fullerene name is a PIN (the book's names are 'PIN' parents; this lane ships names its own reader
    # verified at the best-effort tier, never at a PIN tier)
    record_non_pin_fragment("(C60-Ih)[5,6]fulleren")
    return Split(frozenset(system), (), (), reading.name, len(set(system) & unsat) // 2, unsat)


def fused_parent(mol, residual) -> Optional[FusedParent]:
    """The name and every numbering of the C60 parent on ``residual`` (``parents.fused_parent`` asks first)."""
    if len(residual) < 60 or not tier_open():
        return None
    reading = _reading(mol, set(residual))
    if reading is None:
        return None
    return FusedParent(reading.name, reading.numberings, SOURCE)


def fullerene_in_scope(mol) -> bool:
    """True when ``mol`` is a C60 adduct of the scope (``perception.fullerene.c60_adduct``) whose parent this module
    names (``_reading``: the attached ring has a component name) in a request that names it, the best-effort tier."""
    if not tier_open() or mol is None or mol.GetNumAtoms() < 60:
        return False
    adduct = c60_adduct(mol)
    return adduct is not None and _reading(mol, set(adduct.cage) | set(adduct.ring)) is not None
