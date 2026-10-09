"""The book's names for a single ring of ten or fewer members, with their numbering.

The best-effort writers (``terminal_ring``, ``terminal_fragment``, the floor in
``assembly/universal_substituent``) used to spell every ring of ten or fewer members by
skeletal replacement ('1-azacyclobutan-1-yl', '1,3-diazacyclopenta-2,4-dien-1-yl') and a
benzene ring as 'cyclohexa-1,3,5-trien-1-yl'. Neither is a book spelling:

* (the Blue Book): "Benzene is the retained name for C6H6";
  (:16290): 'phenyl' is a preferred prefix, fully substitutable.
* (:8482): heteromonocycles with up to and including ten ring members are named
  by the extended Hantzsch-Widman system; (:23682) makes those names (and the
  retained names of Tables 2.2 and 2.3) the PINs. 'azetidine (PIN)' (:8402).
* (1) (:15813) and (c) (:2913): 'cyclopropyl', never 'cyclopropan-1-yl'.

``monocycle_form`` returns the book's parent name of one ring and the numbering it is
spelled in, so a writer can place its substituent locants on it. The numbering follows
 (:3219) and (:8284): the most senior heteroatom (O > S > Se > Te > N >
P > As > Sb > Bi > Si > Ge > Sn > Pb > B) gets locant 1, then lowest locants to the
heteroatoms as a set and in seniority order, then to indicated hydrogen (b),
 :8318), then to the free valence (c)), then to the prefixes
(f)); a canonical-rank tie-break makes the result independent of input atom order.

The parent name of a heteromonocycle is built from the ring skeleton and this form's own
numbering: the retained name Tables 2.2, 2.3), when one applies and its fixed
numbering matches this ring's, else the Hantzsch-Widman name (``heterocycles.build_hw_name``)
on this form's own heteroatom locants. No namer's output is parsed. Out of scope (``None``): charged
(an 'ium' nitrogen only for a caller that passes ``allow_cation`` and expresses the charge)
or isotopically labelled ring atoms, rings with partial unsaturation other than one
indicated-hydrogen atom of a mancude ring (hydro prefixes are the ring-hydrogen lane's
build; a mancude ring whose atoms carry an exocyclic double bond takes them for the ring
atoms without a ring double bond), ring heteroatoms with a nonstandard bonding number, and elements outside the
Hantzsch-Widman table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

from rdkit import Chem

__all__ = ["MonocycleForm", "aromatic_view", "branch_carriers", "monocycle_form"]

#: (:8284) seniority, Hantzsch-Widman elements of Table 2.4 that this module
#: names (Al, Ga, In, Tl are left to the caller).
_SENIORITY = ("O", "S", "Se", "Te", "N", "P", "As", "Sb", "Bi", "Si", "Ge", "Sn", "Pb", "B")
_RANK = {el: i for i, el in enumerate(_SENIORITY)}
#: standard bonding number of each ring heteroatom (Table 2.4); another valence needs the
#: lambda convention, which this module does not spell
_BONDING = {"O": 2, "S": 2, "Se": 2, "Te": 2, "N": 3, "P": 3, "As": 3, "Sb": 3, "Bi": 3,
            "Si": 4, "Ge": 4, "Sn": 4, "Pb": 4, "B": 3}

#: retained names Tables 2.2, 2.3), and the Hantzsch-Widman names whose locants
#: omits, that cite no heteroatom locant: locant -> element of their fixed
#: numbering. A name outside this table and without a
#: locant list must hold exactly one heteroatom:8468, locant '1').
_RETAINED_HETERO_LOCANTS = {
    "pyridine": {1: "N"}, "pyrimidine": {1: "N", 3: "N"}, "pyrazine": {1: "N", 4: "N"},
    "pyridazine": {1: "N", 2: "N"}, "pyrrole": {1: "N"}, "imidazole": {1: "N", 3: "N"},
    "pyrazole": {1: "N", 2: "N"}, "furan": {1: "O"}, "thiophene": {1: "S"},
    "selenophene": {1: "Se"}, "tellurophene": {1: "Te"}, "piperidine": {1: "N"},
    "piperazine": {1: "N", 4: "N"}, "morpholine": {1: "O", 4: "N"},
    "thiomorpholine": {1: "S", 4: "N"}, "selenomorpholine": {1: "Se", 4: "N"},
    "telluromorpholine": {1: "Te", 4: "N"}, "pyrrolidine": {1: "N"},
    "imidazolidine": {1: "N", 3: "N"}, "pyrazolidine": {1: "N", 2: "N"},
    "pyran": {1: "O"}, "thiopyran": {1: "S"}, "selenopyran": {1: "Se"},
    "telluropyran": {1: "Te"},
    # Hantzsch-Widman names whose heteroatom locants (the Blue Book)
    # omits, no isomer being possible: '1H-tetrazole (PIN) (not 1H-1,2,3,4-tetrazole)'
    # (:2989)
    "tetrazole": {1: "N", 2: "N", 3: "N", 4: "N"},
    "pentazole": {1: "N", 2: "N", 3: "N", 4: "N", 5: "N"},
}
@dataclass(frozen=True)
class MonocycleForm:
    """One ring spelled by its book name.

    ``parent`` the parent hydride name ('benzene', '1H-imidazole', 'piperidine').
    ``numbering`` ring atom index -> locant, the numbering ``parent`` is spelled in.
    ``kind`` 'benzene', 'cycloalkane' or 'heteromonocycle'.
    """

    parent: str
    numbering: Dict[int, int]
    kind: str
    parent_cites_locant: bool = False
    #: the free valence sits on a hydro-prefix atom that is not the indicated-hydrogen atom
    #: ('2,4-dioxo-1,2,3,4-tetrahydropyrimidin-1-yl'); ``fv_locant`` is its locant,
    #: ``fv_has_hydrogen`` says the atom is a ring carbon (it keeps a hydrogen to substitute)
    fv_hydro_locant: Optional[int] = None
    fv_has_hydrogen: bool = True

    def cites_locant(self, free_valence_locant: Optional[int] = None) -> bool:
        """The parent or its prefix at ``free_valence_locant`` cites a locant,
        the Blue Book): a heteromonocycle whose parent cites one (indicated hydrogen,
        a hydro prefix, or heteroatom locants that are not omitted,, or any
        prefix with a free valence off locant 1."""
        if self.kind != "heteromonocycle":
            return False
        return self.parent_cites_locant or free_valence_locant is not None

    def prefix(self, free_valence_atom: int, bond_order: int = 1) -> Optional[str]:
        """The substituent prefix with the free valence at ``free_valence_atom``.

        'phenyl':16290), 'cyclopropyl' / 'cyclohexylidene' (1)
        :15813), 'pyridin-2-yl' / 'piperidin-4-ylidene' / '1H-imidazol-1-yl' (2)
        :15814: the final 'e' of the parent is elided before 'y', the locant is cited)."""
        return self.prefix_at_locant(self.numbering.get(free_valence_atom), bond_order)

    def prefix_at_locant(self, loc: Optional[int], bond_order: int = 1) -> Optional[str]:
        """``prefix`` for the free valence at ring locant ``loc``."""
        if loc is None or bond_order not in (1, 2):
            return None
        if self.kind == "benzene":
            return "phenyl" if bond_order == 1 and loc == 1 else None
        if self.kind == "cycloalkane":
            if loc != 1:
                return None
            return self.parent[:-3] + ("yl" if bond_order == 1 else "ylidene")
        stem = self.parent[:-1] if self.parent.endswith("e") else self.parent
        out = f"{stem}-{loc}-{'yl' if bond_order == 1 else 'ylidene'}"
        if loc == self.fv_hydro_locant and (bond_order == 2 or not self.fv_has_hydrogen):
            # (the Blue Book): "When no hydrogen atoms are present or when
            # an 'ylidene' type substituent group is needed, it is necessary to use 'added
            # [indicated] hydrogen'"; 'pyridin-1(4H)-yl (preferred prefix)' (:16079),
            # (:24703). This spelling uses hydro prefixes instead: valid, not the PIN.
            from ..metrics.provenance import record_non_pin_label
            record_non_pin_label(out, rule="P-58.2.2.2",
                                 detail="hydro prefixes where added hydrogen is required")
        return out


def _cycle_order(mol, ring: Sequence[int]) -> Optional[list]:
    ring_set = set(ring)
    adj = {}
    for a in ring:
        nb = [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() in ring_set]
        if len(nb) != 2:
            return None
        adj[a] = nb
    start = min(ring)
    order, prev, cur = [start], None, start
    while True:
        nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
        if nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
        if len(order) > len(ring):
            return None
    return order if len(order) == len(ring) else None


def _ring_double_bond_atoms(mol, ring_set) -> Optional[set]:
    """Ring atoms that carry a ring double bond in a Kekulé structure of ``mol``."""
    try:
        kek = Chem.RWMol(mol)
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:  # noqa: BLE001 - an unkekulizable ring is out of scope
        return None
    out = set()
    for b in kek.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in ring_set and j in ring_set and b.GetBondType() == Chem.BondType.DOUBLE:
            out |= {i, j}
        if i in ring_set and j in ring_set and b.GetBondType() == Chem.BondType.TRIPLE:
            return None
    return out


def _exocyclic_multiple(mol, a: int, ring_set) -> bool:
    for b in mol.GetAtomWithIdx(a).GetBonds():
        o = b.GetOtherAtomIdx(a)
        if o not in ring_set and b.GetBondTypeAsDouble() >= 2.0:
            return True
    return False


def _canonical_ranks(mol):
    return list(Chem.CanonicalRankAtoms(mol, breakTies=True))


def _split_hydro(locants, ih_locant=None):
    """Indicated hydrogen and hydro prefixes for the sp3 positions ``locants`` of a ring
    whose mancude parent the name cites: an odd count leaves one indicated hydrogen, at
    the lowest locant (b), the Blue Book), or at ``ih_locant``, the atom of
    a free valence that takes it (``free_valence_carries_indicated_hydrogen``); the rest
    pair up as hydro prefixes / (e))."""
    s = sorted(locants)
    if len(s) % 2:
        if ih_locant is not None and ih_locant in s:
            return (ih_locant,), tuple(x for x in s if x != ih_locant)
        return (s[0],), tuple(s[1:])
    return (), tuple(s)


def free_valence_carries_indicated_hydrogen(state, free_valence_atom) -> bool:
    """ (the Blue Book): "When there are an equal number of indicated
    hydrogen atoms and principal characteristic groups or free valences to be
    accommodated, the indicated hydrogen atoms are placed at peripheral atoms that will
    accommodate these principal characteristic groups or free valences" -- the free
    valence is put on the mancude parent before the hydro prefixes Example 5,
    :25462-:25477, 'Parent hydride with free valence 2H-isoindol-2-yl' ->
    '1,3-dioxo-1,3-dihydro-2H-isoindol-2-yl (preferred prefix)'; '2-(1,3,4,5-tetrahydro-
    2H-2-benzazepin-2-yl)ethan-1-ol (PIN)',:24774). True when the atom of the free valence
    has no hydrogen in the mancude parent (a ring nitrogen with two ring bonds,
    ``bridged_fused_pin.hydro.lacks_hydrogen``) and the parent allows indicated hydrogen
    there (``state`` is a ``ring_hydrogen`` state). A ring carbon keeps a hydrogen in the
    mancude parent and needs none: '2,3-dihydro-1H-inden-2-yl (preferred prefix)'
    (:17374), '3,4-dihydro-1H-2-benzopyran-3-yl' (:17412)."""
    from .bridged_fused_pin import hydro
    return (state is not None and free_valence_atom is not None
            and hydro.lacks_hydrogen(state, free_valence_atom, 1)
            and any(free_valence_atom in s for s in state.ih_sets))


def _numbering(mol, order, hetero, sp3_atoms, free_valence_atom, branch_atoms,
               ih_atom=None, cationic=()) -> Dict[int, int]:
    n = len(order)
    ranks = _canonical_ranks(mol)
    senior = min((_RANK[mol.GetAtomWithIdx(a).GetSymbol()] for a in hetero), default=None)
    best_key, best = None, None
    for start in range(n):
        for direction in (1, -1):
            a2p = {order[(start + direction * p) % n]: p + 1 for p in range(n)}
            first = order[start]
            #: locant 1 goes to a heteroatom of the most senior element
            first_ok = 0 if senior is None or (
                first in hetero and _RANK[mol.GetAtomWithIdx(first).GetSymbol()] == senior) else 1
            het_set = tuple(sorted(a2p[a] for a in hetero))
            sen = tuple(a2p[a] for a in sorted(
                hetero, key=lambda x: (_RANK[mol.GetAtomWithIdx(x).GetSymbol()], a2p[x])))
            ih, hydro = _split_hydro((a2p[a] for a in sp3_atoms),
                                     a2p.get(ih_atom) if ih_atom is not None else None)
            # "CATIONIC PREFIX NAMES" (1) (the Blue Book, method (1) leads to
            # preferred names): "Where there is a choice for numbering, free valences receive
            # lowest possible locants"; '4,4-dimethylpiperazin-4-ium-1-ylium (PIN)' (:42205).
            # Then (:42219): low locants for the skeletal cationic centres.
            chg = tuple(sorted(a2p[a] for a in cationic))
            fv = a2p[free_valence_atom] if free_valence_atom in a2p else 0
            br = tuple(sorted(a2p[a] for a in branch_atoms if a in a2p))
            tie = tuple(ranks[order[(start + direction * p) % n]] for p in range(n))
            # (a) heteroatoms, (b) indicated hydrogen, (c) free valence,
            # (e) hydro prefixes, (f) detachable prefixes
            key = (first_ok, het_set, sen, ih, fv, chg, hydro, br, tie)
            if best_key is None or key < best_key:
                best_key, best = key, a2p
    return best


_HYDRO_MULT = {2: "di", 4: "tetra", 6: "hexa", 8: "octa"}


def monocycle_form(mol, ring_atoms: Iterable[int], free_valence_atom: Optional[int] = None,
                   branch_atoms: Sequence[int] = (),
                   allow_cation: bool = False) -> Optional[MonocycleForm]:
    """The book's name and numbering of one ring (``_monocycle_form``), recorded with
    ``assembly.book_prefixes.note_book_form`` when one is given."""
    form = _monocycle_form(mol, ring_atoms, free_valence_atom, branch_atoms, allow_cation)
    if form is not None:
        from ..assembly.book_prefixes import note_book_form
        note_book_form("ring")
    return form


def _monocycle_form(mol, ring_atoms: Iterable[int], free_valence_atom: Optional[int] = None,
                    branch_atoms: Sequence[int] = (),
                    allow_cation: bool = False) -> Optional[MonocycleForm]:
    """The book's name and numbering of the single ring ``ring_atoms``, or ``None``.

    ``branch_atoms`` lists the ring atoms that carry a substituent prefix, once per prefix
     (f) lowest locants). ``free_valence_atom`` is the ring atom of the free valence
    of a substituent prefix, ``None`` for a parent."""
    ring = list(ring_atoms)
    ring_set = set(ring)
    if not 3 <= len(ring) <= 10:
        return None
    if free_valence_atom is not None and free_valence_atom not in ring_set:
        return None
    ri = mol.GetRingInfo()
    if any(ri.NumAtomRings(a) != 1 for a in ring):
        return None
    order = _cycle_order(mol, ring)
    if order is None:
        return None
    cationic = []
    for a in ring:
        atom = mol.GetAtomWithIdx(a)
        if atom.GetIsotope() or atom.GetNumRadicalElectrons():
            return None
        if atom.GetFormalCharge():
            # an 'ium' ring nitrogen: only for a caller that expresses the charge
            if not (allow_cation and atom.GetSymbol() == "N" and atom.GetFormalCharge() == 1):
                return None
            cationic.append(a)
    hetero = [a for a in ring if mol.GetAtomWithIdx(a).GetSymbol() != "C"]
    if any(mol.GetAtomWithIdx(a).GetSymbol() not in _RANK for a in hetero):
        return None
    for a in hetero:
        atom = mol.GetAtomWithIdx(a)
        if atom.GetTotalValence() != _BONDING[atom.GetSymbol()] + (a in cationic):
            return None
    aromatic = all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring)
    if not aromatic and any(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
        return None
    # a ring C=O (or C=S, C=N) of a mancude ring leaves that atom without a ring double bond:
    # it takes indicated hydrogen or a hydro prefix and the caller cites the 'oxo' prefix
    # ('2-oxo-2H-pyran-5-yl', '6-oxo-1,6-dihydropyridin-3-yl'; the Blue Book,
    #:43483)
    exo_multiple = aromatic and any(_exocyclic_multiple(mol, a, ring_set) for a in ring)
    ring_bonds = [mol.GetBondBetweenAtoms(order[i], order[(i + 1) % len(order)])
                  for i in range(len(order))]
    saturated = (not aromatic) and all(
        b.GetBondType() == Chem.BondType.SINGLE for b in ring_bonds)

    if not hetero:
        if exo_multiple:
            return None  # no book name spells a carbocycle that has a ring C=X on a mancude ring
        if aromatic and len(ring) == 6:
            kind, parent = "benzene", "benzene"
        elif saturated:
            from ..data.chain_names import get_chain_prefix
            kind, parent = "cycloalkane", f"cyclo{get_chain_prefix(len(ring))}ane"
        else:
            return None  # 'cyclohex-2-en-1-yl' etc.: the caller's spelling is the book's
        numbering = _numbering(mol, order, [], [], free_valence_atom, list(branch_atoms))
        if free_valence_atom is not None and numbering.get(free_valence_atom) != 1:
            return None
        return MonocycleForm(parent=parent, numbering=numbering, kind=kind)

    dbl = _ring_double_bond_atoms(mol, ring_set)
    if dbl is None:
        return None
    # (:8318): after the mancude double bonds are placed, a ring atom of
    # bonding number three or more joined to its ring neighbours by single bonds only is
    # an indicated-hydrogen position; in a partly hydrogenated ring the further such atoms
    # take hydro prefixes. A saturated ring has its own Hantzsch-Widman
    # name ('oxolane', and none of these.
    sp3 = [] if saturated else [
        a for a in ring if a not in dbl
        and _BONDING.get(mol.GetAtomWithIdx(a).GetSymbol(), 4) >= 3]
    if aromatic and len(sp3) > 1 and not exo_multiple:
        return None
    if any(a in sp3 for a in cationic):
        return None  # an 'ium' nitrogen with hydro prefixes or indicated hydrogen is not built
    ih_atom = _free_valence_ih_atom(mol, ring, order, sp3, free_valence_atom)
    numbering = _numbering(mol, order, hetero, sp3, free_valence_atom, list(branch_atoms),
                           ih_atom, cationic)
    het = {numbering[a]: mol.GetAtomWithIdx(a).GetSymbol() for a in hetero}
    from .heterocycles import (_apply_retained_stem, _mancude_retained_stem,
                               _saturated_ring_retained_name, build_hw_name)
    try:
        # (Tables 2.2, 2.3): a retained name, looked up by the ring skeleton
        retained = (_saturated_ring_retained_name(mol, ring) if saturated
                    else _mancude_retained_stem(mol, ring))
        if retained in _RETAINED_HETERO_LOCANTS:
            stem = retained
        else:
            # (with:8470) on this form's own locants
            stem = build_hw_name(sorted(het.items()), len(ring), saturated, not saturated)
            if stem and not saturated:
                stem = _apply_retained_stem(stem)
    except Exception:  # noqa: BLE001 - a builder that raises is a decline
        return None
    if not stem:
        return None
    if stem in _RETAINED_HETERO_LOCANTS and het != _RETAINED_HETERO_LOCANTS[stem]:
        # the retained name's fixed numbering is not this ring's numbering
        return None
    ih, hydro = _split_hydro((numbering[a] for a in sp3),
                             numbering.get(ih_atom) if ih_atom is not None else None)
    if hydro and len(hydro) not in _HYDRO_MULT:
        return None
    parent = stem
    if ih:
        parent = f"{ih[0]}H-{parent}"
    if hydro:
        #: a hyphen before a locant, none before a letter
        glue = "-" if parent[:1].isdigit() else ""
        parent = (f"{','.join(str(x) for x in hydro)}-{_HYDRO_MULT[len(hydro)]}hydro"
                  f"{glue}{parent}")
    # (the Blue Book): the parent cites a locant when it carries indicated
    # hydrogen or a hydro prefix, or when its heteroatom locants are not the ones
    # omits (a retained name's fixed numbering, Tables 2.2/2.3, cites none)
    parent_cites_locant = bool(ih) or bool(hydro) or (
        stem not in _RETAINED_HETERO_LOCANTS and len(het) > 1
        and not (len(set(het.values())) == 1 and len(het) >= len(ring) - 1))
    fv_hydro = (numbering[free_valence_atom] if free_valence_atom in sp3
                and numbering[free_valence_atom] not in ih else None)
    return MonocycleForm(parent=parent, numbering=numbering, kind="heteromonocycle",
                         parent_cites_locant=parent_cites_locant, fv_hydro_locant=fv_hydro,
                         fv_has_hydrogen=(free_valence_atom is None or
                                          mol.GetAtomWithIdx(free_valence_atom).GetSymbol() == "C"))


def _free_valence_ih_atom(mol, ring, order, sp3, free_valence_atom):
    """The atom of the free valence when it takes the ring's one indicated hydrogen
    (``free_valence_carries_indicated_hydrogen``: '1,5-dihydro-4H-1,2,4-triazol-4-yl',
    not '4,5-dihydro-1H-1,2,4-triazol-4-yl'), else None (the lowest locant takes it)."""
    if free_valence_atom not in sp3 or len(sp3) < 3 or len(sp3) % 2 == 0:
        return None
    from . import ring_hydrogen as rh_rule
    try:
        rh = rh_rule.ring_hydrogen(mol, ring, (), {a: i + 1 for i, a in enumerate(order)})
    except Exception:  # noqa: BLE001 - a structure the rule cannot read keeps the lowest
        return None
    if rh is None or not free_valence_carries_indicated_hydrogen(rh.state, free_valence_atom):
        return None
    return free_valence_atom


def aromatic_view(mol):
    """A copy of ``mol`` (same atom indices) with aromaticity perceived, for a caller
    that works on a kekulized copy without aromatic flags; ``None`` when RDKit cannot
    perceive it."""
    try:
        view = Chem.Mol(mol)
        Chem.SanitizeMol(view, Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
        return view
    except Exception:  # noqa: BLE001 - no view: the caller keeps its own spelling
        return None


def branch_carriers(mol, ring_atoms, free_valence_atom=None, within=None):
    """One entry per substituent prefix on each ring atom (f)): every heavy
    neighbour outside the ring (inside ``within`` when given), less the parent bond of
    the free-valence atom when ``within`` is not given."""
    ring = set(ring_atoms)
    out = []
    for a in sorted(ring):
        n = 0
        for nb in mol.GetAtomWithIdx(a).GetNeighbors():
            j = nb.GetIdx()
            if j in ring or nb.GetAtomicNum() <= 1:
                continue
            if within is not None and j not in within:
                continue
            n += 1
        if within is None and a == free_valence_atom:
            n -= 1
        out.extend([a] * max(n, 0))
    return out
