"""Charge-only resonance forms of a conjugated cation, found by matching its pi bonds.

A cation whose +1 can sit on several nitrogen atoms is one species whatever SMILES draws the charge (the contributor guide core
principle 4 "Determinism"), and the hydron that names it sits on one atom of the delocalised system:
"General rule for systematically naming cationic centers in parent hydrides" (the Blue Book),:41368: "A cation
derived formally by adding one or more hydrons to any position of a neutral parent hydride... is named by replacing the
final letter 'e' of the parent hydride name, if any, by the suffix 'ium'".

Give each atom of a Kekule structure p, its number of pi bonds (a double bond counts one, a triple bond two). A shift of
the +1 lowers p by one on the nitrogen that loses it and raises p by one on the neutral three-connected nitrogen that
takes it. The shifted structure is a perfect matching of the pi bonds in which atom i has p'(i) pi bonds: replace atom i
by p'(i) copies, join the copies of bonded atoms, and ask Edmonds' blossom algorithm for a perfect matching. The search
is polynomial, it counts its own work (configurations and matching nodes) and reads no clock, so one molecule gives one
answer on every machine.

The +1 does not move in a pi system that holds a carbanion centre, a carbon with a negative charge: an N-heterocyclic
carbene drawn as the zwitterion ``[N+]=[C-]`` (the +1 would swap between the two nitrogen atoms of N-C(-)=N+), an
N-ylide, an isocyanide or a dicyanomethanide conjugated with the cation. RDKit's enumeration writes a carbene, an
N-ylide and an isocyanide as one structure. For a dicyanomethanide it writes more, among them the +1 on the other
nitrogen, but only with the charge of the carbanion moved onto a nitrile nitrogen, and:func:`accepted_charge_forms`
rejects that drawing because only nitrogen atoms may change their charge. So the forms RDKit gives such a group are the
drawn one alone, and a form that RDKit's filtered enumeration does not give is not offered here either
(:meth:`_Drawing.in_carbanion_system`). Measured on the corpus rows that RDKit finished and that hold a carbanion in a
pi system with a mobile nitrogen cation: 867 rows, RDKit gives a second form from that system for none of them (two
rows get theirs from another pi system), and the helper's forms equal RDKit's on 844. A carbanion in another pi system,
or cut off from the cation by sp3 atoms, does not stop a shift.

The contract of every function that answers a question about forms is: an empty answer is a proof (no form, no site),
and ``None`` is "undecided" with one reason code in ``why``. A caller treats ``None`` as unknown, never as empty.
Reason codes:
  ``metal`` a metal bonded in a fragment that holds a mobile nitrogen cation
  ``radical`` an unpaired electron in such a fragment, for the questions that read structures, not forms
  ``kekulize`` no Kekule structure
  ``configurations`` more lost and donor choices or Kekule alternatives than the count allows
  ``nodes`` more matching nodes than the count allows
  ``bond_order`` a matching that no bond order draws
  ``inchi`` no standard InChIKey for the drawing, so no form can be checked against it
  ``anion`` an anionic pi atom beside a +1 that cannot reach a site without moving the anion too

Two counts of the graph of pi-capable atoms (:func:`pi_capable`) choose between RDKit's resonance enumeration and this
module::func:`pi_atoms` for the enumeration with ``UNCONSTRAINED_CATIONS`` (:func:`charge_drawings`) and
:func:`pi_work` for the one without a flag. The graph is the same for every drawing of the charge, so one species takes
one route.
"""
import itertools
from typing import Dict, List, Mapping, Optional, Tuple

from rdkit import Chem

#: pi atoms of the largest pi system up to which RDKit's enumeration with ``UNCONSTRAINED_CATIONS`` is asked. RDKit
#: builds every bond-order arrangement of a conjugated group, about x3 to x8 per linked ring.
RDKIT_RESONANCE_MAX_CONJUGATED = 32
#::func:`pi_work` up to which RDKit's enumeration without a flag is asked. Its cost follows the acyclic groups that
#: carry charge (azide, diazonium, azo, nitroso: about x7 to x13 each), not the rings.
RDKIT_RESONANCE0_MAX_WORK = 28
#: lost and donor choices, and Kekule alternatives, tried in one call before the answer is undecided
MAX_CONFIGURATIONS = 400
#: copies of atoms in one matching before the answer is undecided
MAX_PI_NODES = 600

#: atomic numbers that make a bonded fragment undecided: the alkali and alkaline-earth metals, the d and f blocks, and
#: Al, Ga, In, Tl, Sn, Pb, Sb and Bi. B, Si, Ge, As, Se and Te are ordinary pi atoms here.
METALS = frozenset(
    {3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88}
    | set(range(21, 31)) | set(range(39, 49)) | set(range(57, 81)) | set(range(89, 113))
    | {13, 31, 49, 81, 50, 82, 51, 83})

#: N, O, P, S, As, Se and Te: an atom with at most three neighbours that holds a lone pair for a pi system
_LONE_PAIR_Z = frozenset({7, 8, 15, 16, 33, 34, 52})

#: a cationic nitrogen that can hand its charge on: an N+ with a double or aromatic bond that is no nitro group or
#: N-oxide (the charge of those sits on the oxygen)
MOBILE_N_CATION = Chem.MolFromSmarts('[#7+;!$([#7+]~[#8-]);!$([#7+]=[#8]);$([#7+]=*),$([#7+]:*)]')

#: the atoms of an azido group in each of its three drawings: ``-N=[N+]=[N-]``, its charge-separated twin
#: ``-[N-]-[N+]#N`` and, on a nitrogen host only, ``R2[N+]=N-N=[N-]`` (the +1 on the host; the host is part of the group
#: there, as it is the atom that holds the +1). The group stands on a heavy atom (D2): hydrazoic acid itself, HN3, has no
#: host for the prefix 'azido', so its zwitterionic name is the only name it has
_AZIDE_DRAWINGS = tuple(Chem.MolFromSmarts(p) for p in (
    '[#7;X2;D2;+0]=[#7+;X2]=[#7-;X1]',
    '[#7;X2;D2;-1]-[#7+;X2]#[#7;X1;+0]',
    '[#7+;X3;!R]=[#7;X2;+0]-[#7;X2;+0]=[#7-;X1]'))

_BOND_OF_ORDER = {0: Chem.BondType.SINGLE, 1: Chem.BondType.DOUBLE, 2: Chem.BondType.TRIPLE}
_PI_BONDS = {Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}


# ---------------------------------------------------------------------------------------------------------------------
# counts of the pi-capable graph
# ---------------------------------------------------------------------------------------------------------------------

def pi_capable(atom) -> bool:
    """True when the atom can sit in a pi system in every drawing of its species: it is aromatic, carries a multiple
    bond, is an N, O, P, S, As, Se or Te with at most three neighbours (hydrogens counted), or is a charged atom with at
    most three neighbours (the carbocation or carbanion centre of a triarylmethyl dye). The nitrogen that holds the +1
    in one drawing and the nitrogen that holds a lone pair in another are both pi-capable, so the set does not depend
    on where the charge is drawn."""
    if atom.GetIsAromatic():
        return True
    if any(bond.GetBondTypeAsDouble() > 1.0 for bond in atom.GetBonds()):
        return True
    if atom.GetAtomicNum() in _LONE_PAIR_Z and atom.GetTotalDegree() <= 3:
        return True
    return atom.GetFormalCharge() != 0 and atom.GetTotalDegree() <= 3


def _pi_components(mol) -> List[List[int]]:
    """The connected components of two or more pi-capable atoms joined by any bond, each as a sorted list of atom
    indices, ordered by their lowest index."""
    count = mol.GetNumAtoms()
    capable = [pi_capable(atom) for atom in mol.GetAtoms()]
    parent = list(range(count))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if capable[i] and capable[j]:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    groups: Dict[int, List[int]] = {}
    for i in range(count):
        if capable[i]:
            groups.setdefault(find(i), []).append(i)
    return [group for _root, group in sorted(groups.items()) if len(group) > 1]


def _charged(mol, atoms) -> int:
    return sum(1 for i in atoms if mol.GetAtomWithIdx(i).GetFormalCharge() != 0)


def pi_system(mol) -> Tuple[int, int]:
    """(pi atoms, charged pi atoms) of the largest pi component of two or more atoms; (0, 0) when there is none. A lone
    atom is no system. Of two components of one size the one with more charged atoms is meant."""
    best = (0, 0)
    for component in _pi_components(mol):
        best = max(best, (len(component), _charged(mol, component)))
    return best


def pi_atoms(mol) -> int:
    """The pi atoms of the largest pi system: the count that decides whether RDKit's enumeration with
    ``UNCONSTRAINED_CATIONS`` is asked."""
    return pi_system(mol)[0]


def pi_work(mol) -> int:
    """W0: the largest, over the pi components, of pi atoms + 2 x formally charged pi atoms. RDKit's enumeration without
    a flag costs what its worst group costs, and a small charge-dense group can sit beside a larger neutral one."""
    return max((len(component) + 2 * _charged(mol, component) for component in _pi_components(mol)), default=0)


def rdkit_flags8_affordable(mol) -> bool:
    """True when RDKit's enumeration with ``UNCONSTRAINED_CATIONS`` is asked for ``mol``."""
    return pi_atoms(mol) <= RDKIT_RESONANCE_MAX_CONJUGATED


def rdkit_flags0_affordable(mol) -> bool:
    """True when RDKit's enumeration without a flag is asked for ``mol``."""
    return pi_work(mol) <= RDKIT_RESONANCE0_MAX_WORK


def charge_drawings(mol, rdkit_source, *, why=None) -> Optional[List[Chem.Mol]]:
    """The drawings of ``mol`` that the charge-only forms of its species are read from: ``rdkit_source(mol)`` (RDKit's
    resonance structures, with ``UNCONSTRAINED_CATIONS``) when:func:`rdkit_flags8_affordable`, else
    :func:`charge_shift_forms`. RDKit builds every bond-order arrangement of a conjugated group and its C++ call cannot
    be interrupted, so a molecule above the bound never reaches ``rdkit_source``. ``None`` is undecided (``why`` receives
    the reason code): the caller names the input as drawn, it does not ask RDKit."""
    if rdkit_flags8_affordable(mol):
        return rdkit_source(mol)
    return charge_shift_forms(mol, why=why)


def accepted_charge_forms(mol, smiles: str, candidates) -> Dict[str, Tuple[str, Optional[Tuple[int, ...]]]]:
    """The charge-only forms of ``mol`` among ``candidates`` (drawings of the same species, as ``Mol`` or ``None``), as
    {canonical SMILES: (string, atom order or None)}: the drawing ``smiles`` itself first (order None), then each
    candidate that is a charge-only form, in the order the candidates come. The candidates are RDKit's resonance
    structures or the output of:func:`charge_shift_forms`; one filter decides for both. A candidate is a form when every
    atom keeps its hydrogen count, no atom has an unpaired electron, only nitrogen atoms change their charge, the +1 only
    changes places and goes onto a neutral nitrogen with three connections (a lone pair to give: amidinium, guanidinium,
    azolium), and the number of double bonds between an aromatic and a non-aromatic atom is the drawing's. It is read
    back from its written string and kept only when the standard InChIKey is the drawing's, so a form cannot describe
    another molecule. The bounds on the number of forms are those of ``namer._charge_forms``, not this function's.

    The +1 of an azido group does not move (:func:`azide_atoms`): a form in which an atom of the group in either
    drawing loses or gains the +1 is no form. ``namer._charge_forms`` says it: "A cumulated or dipolar nitrogen cation (an
    azide, a diazo compound, a nitrilium) moves a -1, not a +1, between its resonance structures... such an input is named
    as it is drawn". Without the rule RDKit's enumeration of ``CN(C)N=[N+]=[N-]`` offered ``C[N+](C)=NN=[N-]`` (the +1 on
    the amine nitrogen, a neutral nitrogen with three connections) and the form choice shipped the name of that drawing,
    read by OPSIN to other atoms than the azide the input draws."""
    from .centres_bridge import _smiles_output_order
    key = Chem.MolToInchiKey(mol)
    base_h = [a.GetTotalNumHs() for a in mol.GetAtoms()]
    base_charge = [a.GetFormalCharge() for a in mol.GetAtoms()]
    is_n = [a.GetSymbol() == 'N' for a in mol.GetAtoms()]
    n_charges = sorted(c for k, c in enumerate(base_charge) if is_n[k])
    drawn = Chem.MolToSmiles(mol)
    exocyclic = exocyclic_double_bonds_to_aromatic(mol)
    azide = azide_atoms(mol)
    found: Dict[str, Tuple[str, Optional[Tuple[int, ...]]]] = {drawn: (smiles, None)}
    for form in candidates:
        if form is None:
            continue
        try:
            form = Chem.Mol(form)
            Chem.SanitizeMol(form)
        except Exception:
            continue
        charge = [a.GetFormalCharge() for a in form.GetAtoms()]
        lost = {k for k in range(len(charge)) if base_charge[k] == 1 and charge[k] == 0}
        gained = {k for k in range(len(charge)) if base_charge[k] == 0 and charge[k] == 1}
        if ([a.GetTotalNumHs() for a in form.GetAtoms()] != base_h
                or any(a.GetNumRadicalElectrons() for a in form.GetAtoms())
                or any(not is_n[k] and charge[k] != base_charge[k]
                       for k in range(len(charge)))
                or sorted(c for k, c in enumerate(charge) if is_n[k]) != n_charges
                or len(lost) != len(gained)
                or any(charge[k] != base_charge[k] and k not in lost and k not in gained
                       for k in range(len(charge)))
                or any(mol.GetAtomWithIdx(k).GetTotalDegree() != 3 for k in gained)
                or exocyclic_double_bonds_to_aromatic(form) != exocyclic
                or (lost | gained) & (azide | azide_atoms(form))):
            continue
        written = Chem.MolToSmiles(form, canonical=False)
        order = _smiles_output_order(form)
        back = Chem.MolFromSmiles(written)
        if (back is None or order is None or back.GetNumAtoms() != mol.GetNumAtoms()
                or len(order) != mol.GetNumAtoms() or Chem.MolToInchiKey(back) != key):
            continue
        found.setdefault(Chem.MolToSmiles(back), (written, tuple(order)))
    return found


def azide_atoms(mol) -> frozenset:
    """The atoms of the azido groups of ``mol``, in whichever of the three drawings of:data:`_AZIDE_DRAWINGS` it
    has them. "AZIDES" (the Blue Book): the group -N3 (-N=N+=N-) is named by the prefix 'azido'; its
    charges are the bonding of that prefix, not ionic centres that move."""
    if not any(a.GetFormalCharge() == 1 for a in mol.GetAtoms()):
        return frozenset()
    return frozenset(i for drawing in _AZIDE_DRAWINGS for match in mol.GetSubstructMatches(drawing) for i in match)


def charge_places(forms) -> int:
    """How many different places the +1 takes among ``forms`` (the values of:func:`accepted_charge_forms`):
    the set of atoms of the input that carry a +1 in the form, read from its string through its atom order. Two
    forms with the +1 on the same atoms are Kekule structures of one placement, so a count of forms exceeds the
    count of placements wherever a conjugated system that RDKit does not aromatise has more than one."""
    places = set()
    for string, order in forms:
        form = Chem.MolFromSmiles(string)
        if form is None:
            places.add(string)
            continue
        places.add(frozenset(order[a.GetIdx()] if order else a.GetIdx()
                             for a in form.GetAtoms() if a.GetFormalCharge() > 0))
    return len(places)


_CONFIGURED_BOND = frozenset({Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
                              Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS})
_CIS_BOND = frozenset({Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS})
_CONFIGURED_ATOM = frozenset({Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW})


def stereo_elements(mol, order: Optional[Tuple[int, ...]] = None) -> Dict[tuple, bool]:
    """The configured stereo elements of ``mol``: each double bond that carries an E/Z (cis/trans) configuration
    and each atom that carries a tetrahedral chiral tag, keyed by its atoms in the numbering of the INPUT
    (``order[k]`` is the input's atom of atom k of ``mol``, as in:func:`accepted_charge_forms`; None is the
    identity), and valued by the configuration itself in a way that does not depend on how ``mol`` numbers or
    writes its atoms. A double bond: ``('=', {begin, end})`` is True when the lowest-numbered neighbours at the two
    ends lie on the same side (RDKit's reference atoms of the bond, re-expressed on those neighbours). A chiral atom:
    ``('@', atom)`` is the sense of its tag taken over its neighbours in the order of their numbers in the input.
    Two molecules of one graph (the forms of one species share their atoms and their connections and differ in bond
    orders and charges) can then be compared element by element."""
    to_input = (lambda k: order[k]) if order else (lambda k: k)
    found: Dict[tuple, bool] = {}
    for bond in mol.GetBonds():
        if bond.GetStereo() not in _CONFIGURED_BOND:
            continue
        ends = (bond.GetBeginAtom(), bond.GetEndAtom())
        reference = list(bond.GetStereoAtoms())
        if len(reference) != 2:
            continue
        same_side = bond.GetStereo() in _CIS_BOND
        for atom, other in (ends, ends[::-1]):
            neighbours = [to_input(n.GetIdx()) for n in atom.GetNeighbors() if n.GetIdx() != other.GetIdx()]
            drawn = next((to_input(k) for k in reference if mol.GetBondBetweenAtoms(atom.GetIdx(), k)), None)
            if drawn is not None and neighbours and drawn != min(neighbours):
                same_side = not same_side
        found[('=', frozenset(to_input(a.GetIdx()) for a in ends))] = same_side
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() not in _CONFIGURED_ATOM:
            continue
        around = [to_input(b.GetOtherAtomIdx(atom.GetIdx())) for b in atom.GetBonds()]
        swaps = sum(1 for i in range(len(around)) for j in range(i + 1, len(around)) if around[i] > around[j])
        found[('@', to_input(atom.GetIdx()))] = (atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW) != bool(swaps % 2)
    return found


def kept_stereo(mol, forms) -> Optional[Tuple[int, ...]]:
    """For each form of ``forms`` (the values of:func:`accepted_charge_forms` for ``mol``, in that order), how many of
    the configured stereo elements of ``mol`` (:func:`stereo_elements`) the form's own string carries with the same
    configuration. A form is written from a Kekule structure whose double bonds are not the drawing's, and a bond that
    is single in it carries no E/Z: the standard InChIKey that admits a form (:func:`accepted_charge_forms`) does not
    see the bonds of a delocalised polymethine, so the key cannot say that a form has lost them. None where a form's
    string cannot be read (the caller then does not compare the forms by this count)."""
    wanted = stereo_elements(mol)
    counts = []
    for string, order in forms:
        form = Chem.MolFromSmiles(string)
        if form is None:
            return None
        mine = stereo_elements(form, order)
        counts.append(sum(1 for key, value in wanted.items() if key in mine and mine[key] == value))
    return tuple(counts)


# ---------------------------------------------------------------------------------------------------------------------
# Edmonds' blossom algorithm
# ---------------------------------------------------------------------------------------------------------------------

def perfect_matching(nodes, adj: Mapping) -> Optional[Dict[int, int]]:
    """A perfect matching of the graph on ``nodes`` with the neighbour lists ``adj`` (a neighbour outside ``nodes`` is
    ignored), as {node: its mate}, or None when there is none. Edmonds' blossom algorithm: odd cycles (pyrrole) need
    it. Nodes and neighbours are taken in sorted order, so the matching does not depend on the order of ``adj``."""
    order = sorted(nodes)
    count = len(order)
    if count % 2:
        return None
    index = {node: i for i, node in enumerate(order)}
    graph = [sorted({index[w] for w in adj.get(node, ()) if w in index and w != node}) for node in order]
    mate = [-1] * count
    parent = [-1] * count
    base = list(range(count))
    used = [False] * count
    blossom = [False] * count

    def lowest_common_ancestor(a, b):
        seen = [False] * count
        while True:
            a = base[a]
            seen[a] = True
            if mate[a] == -1:
                break
            a = parent[mate[a]]
        while True:
            b = base[b]
            if seen[b]:
                return b
            b = parent[mate[b]]

    def mark_path(v, b, child):
        while base[v] != b:
            blossom[base[v]] = blossom[base[mate[v]]] = True
            parent[v] = child
            child = mate[v]
            v = parent[mate[v]]

    def find_path(root):
        for i in range(count):
            used[i] = False
            parent[i] = -1
            base[i] = i
        used[root] = True
        queue = [root]
        head = 0
        while head < len(queue):
            v = queue[head]
            head += 1
            for to in graph[v]:
                if base[v] == base[to] or mate[v] == to:
                    continue
                if to == root or (mate[to] != -1 and parent[mate[to]] != -1):
                    current = lowest_common_ancestor(v, to)
                    for i in range(count):
                        blossom[i] = False
                    mark_path(v, current, to)
                    mark_path(to, current, v)
                    for i in range(count):
                        if blossom[base[i]]:
                            base[i] = current
                            if not used[i]:
                                used[i] = True
                                queue.append(i)
                elif parent[to] == -1:
                    parent[to] = v
                    if mate[to] == -1:
                        return to
                    nxt = mate[to]
                    used[nxt] = True
                    queue.append(nxt)
        return -1

    for i in range(count):
        if mate[i] != -1:
            continue
        v = find_path(i)
        if v == -1:
            return None      # no augmenting path from i: a maximum matching leaves i out, so none is perfect
        while v != -1:
            previous = parent[v]
            nxt = mate[previous]
            mate[v] = previous
            mate[previous] = v
            v = nxt
    return {order[i]: order[mate[i]] for i in range(count)}


# ---------------------------------------------------------------------------------------------------------------------
# the drawn molecule as a Kekule structure, and the shifts of its pi bonds
# ---------------------------------------------------------------------------------------------------------------------

class _UndecidedError(Exception):
    """The matching cannot answer; ``reason`` is the code the caller reads from ``why``."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _Budget:
    """The count of configurations tried in one call."""

    def __init__(self, limit: int):
        self.limit = limit
        self.spent = 0

    def spend(self) -> None:
        self.spent += 1
        if self.spent > self.limit:
            raise _UndecidedError("configurations")


def exocyclic_double_bonds_to_aromatic(mol) -> int:
    """The number of double bonds between an aromatic atom and an atom that is not."""
    return sum(1 for bond in mol.GetBonds()
               if bond.GetBondType() == Chem.BondType.DOUBLE
               and bond.GetBeginAtom().GetIsAromatic() != bond.GetEndAtom().GetIsAromatic())


def _report(why, reason: str) -> None:
    if why is not None:
        why.append(reason)


def _require_decidable(mol, mobile, *, radicals: bool) -> None:
    """Raise ``_UndecidedError`` for a metal, and with ``radicals`` for an unpaired electron, in a fragment that holds a
    mobile nitrogen cation: the pi-bond count of such a fragment does not give its structures (a bonded metal moves
    charge and bond orders in ways this matching does not draw, and an unpaired electron can move along the pi
    system)."""
    holders = set(mobile)
    for fragment in Chem.GetMolFrags(mol):
        if not holders.intersection(fragment):
            continue
        atoms = [mol.GetAtomWithIdx(i) for i in fragment]
        if any(a.GetAtomicNum() in METALS and a.GetDegree() > 0 for a in atoms):
            raise _UndecidedError("metal")
        if radicals and any(a.GetNumRadicalElectrons() for a in atoms):
            raise _UndecidedError("radical")


class _Drawing:
    """``mol`` as one Kekule structure, with what every shift of its pi bonds reads: p per atom, neighbours, hydrogen
    counts and charges, and the pi-capable component of each atom."""

    def __init__(self, mol):
        self.mol = mol
        kekule = Chem.Mol(mol)
        try:
            Chem.Kekulize(kekule, clearAromaticFlags=True)
        except Exception:
            raise _UndecidedError("kekulize")
        self.kekule = kekule
        self.count = mol.GetNumAtoms()
        self.p = [0] * self.count
        for bond in kekule.GetBonds():
            pi = _PI_BONDS.get(bond.GetBondType(), 0)
            self.p[bond.GetBeginAtomIdx()] += pi
            self.p[bond.GetEndAtomIdx()] += pi
        self.adj = [sorted(nb.GetIdx() for nb in atom.GetNeighbors()) for atom in kekule.GetAtoms()]
        self.base_h = [atom.GetTotalNumHs() for atom in mol.GetAtoms()]
        self.base_q = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
        self.exocyclic = exocyclic_double_bonds_to_aromatic(mol)
        self.component_of: Dict[int, int] = {}
        self.components = _pi_components(mol)
        for k, component in enumerate(self.components):
            for i in component:
                self.component_of[i] = k
        self.carbanion_components = {
            k for k, component in enumerate(self.components)
            if any(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 and self.base_q[i] < 0 for i in component)}
        self._key = None

    @property
    def key(self) -> str:
        if self._key is None:
            self._key = Chem.MolToInchiKey(self.mol)
        return self._key

    def in_carbanion_system(self, atom: int) -> bool:
        """True when ``atom`` lies in a pi-capable component that holds a carbon with a negative charge. The +1 of a
        cation in such a system does not move (see the module docstring): RDKit's enumeration gives the group no second
        drawing, and the matching, which would draw one (the +1 across the carbene carbon of ``N-C(-)=N+``), offers
        none."""
        return self.component_of.get(atom) in self.carbanion_components

    def cations(self, restrict=None) -> List[int]:
        """The nitrogen cations that carry a pi bond and can give the +1 away, in index order: not those in a pi
        system that holds a carbanion centre (:meth:`in_carbanion_system`)."""
        return [i for i, atom in enumerate(self.mol.GetAtoms())
                if atom.GetAtomicNum() == 7 and self.base_q[i] == 1 and self.p[i] >= 1
                and not self.in_carbanion_system(i)
                and (restrict is None or i in restrict)]

    def donors(self, restrict=None) -> List[int]:
        """The neutral three-connected nitrogen atoms without a pi bond (a lone pair to give), in index order."""
        return [i for i, atom in enumerate(self.mol.GetAtoms())
                if atom.GetAtomicNum() == 7 and self.base_q[i] == 0 and self.p[i] == 0
                and atom.GetTotalDegree() == 3 and not atom.GetNumRadicalElectrons()
                and (restrict is None or i in restrict)]

    def donors_beside(self, donors, lost) -> List[int]:
        """The donors in the pi-capable component of some atom of ``lost``: a shift needs a path of pi atoms."""
        components = {self.component_of[i] for i in lost if i in self.component_of}
        return [d for d in donors if self.component_of.get(d) in components]

    def _flood(self, interest, p) -> set:
        """The atoms with a pi bond reachable from ``interest`` through such atoms."""
        reached = set()
        stack = sorted(interest)
        while stack:
            u = stack.pop()
            if u in reached or p[u] < 1:
                continue
            reached.add(u)
            stack.extend(v for v in self.adj[u] if p[v] >= 1 and v not in reached)
        return reached

    def _match(self, atoms, p) -> Optional[Dict[Tuple[int, int], int]]:
        """The pi bonds of ``atoms`` when atom i has p[i] of them, as {(i, j): bonds between i and j}, or None."""
        order = sorted(a for a in atoms if p[a] >= 1)
        start: Dict[int, int] = {}
        owner: List[int] = []
        for atom in order:
            start[atom] = len(owner)
            owner.extend([atom] * p[atom])
        if len(owner) > MAX_PI_NODES:
            raise _UndecidedError("nodes")
        adj: Dict[int, List[int]] = {i: [] for i in range(len(owner))}
        for u in order:
            for v in self.adj[u]:
                if v in start:
                    for a in range(p[u]):
                        adj[start[u] + a].extend(start[v] + b for b in range(p[v]))
        matching = perfect_matching(range(len(owner)), adj)
        if matching is None:
            return None
        bonds: Dict[Tuple[int, int], int] = {}
        for i, j in matching.items():
            if i < j:
                u, v = owner[i], owner[j]
                pair = (u, v) if u < v else (v, u)
                bonds[pair] = bonds.get(pair, 0) + 1
        return bonds

    def _build(self, bonds, atoms, lost, gained):
        """The sanitized molecule with the pi bonds ``bonds`` over ``atoms``, the +1 off ``lost`` and on ``gained``;
        None when it does not sanitize."""
        rw = Chem.RWMol(self.kekule)
        for bond in rw.GetBonds():
            a, c = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if (a in atoms and c in atoms) or a in lost or c in lost:
                order = bonds.get((a, c) if a < c else (c, a), 0)
                if order > 2:
                    raise _UndecidedError("bond_order")
                bond.SetBondType(_BOND_OF_ORDER[order])
        for i in tuple(lost) + tuple(gained):
            atom = rw.GetAtomWithIdx(i)
            atom.SetFormalCharge(0 if i in lost else 1)
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(self.base_h[i])
        built = rw.GetMol()
        try:
            Chem.SanitizeMol(built)
        except Exception:
            return None
        return built

    def shifted(self, lost, gained):
        """The structure with the +1 of each atom of ``lost`` moved onto the atoms of ``gained`` (equal numbers), or
        None when the pi bonds cannot be re-matched."""
        p = list(self.p)
        for i in lost:
            p[i] -= 1
        for i in gained:
            p[i] += 1
        interest = set(gained)
        for i in lost:
            if p[i] >= 1:
                interest.add(i)
            interest.update(nb for nb in self.adj[i] if p[nb] >= 1)
        atoms = self._flood(interest, p)
        bonds = self._match(atoms, p)
        if bonds is None:
            return None
        return self._build(bonds, atoms, set(lost), set(gained))

    def kekule_components(self) -> List[set]:
        """The connected sets of atoms that carry a pi bond, ordered by their lowest atom."""
        seen = set()
        out = []
        for i in range(self.count):
            if self.p[i] >= 1 and i not in seen:
                atoms = self._flood({i}, self.p)
                seen |= atoms
                out.append(atoms)
        return out

    def alternative(self, atoms, u, v):
        """The structure with the single bond u-v made a pi bond and the other pi bonds of ``atoms`` re-matched (same
        charges), or None when no matching exists."""
        p = list(self.p)
        p[u] -= 1
        p[v] -= 1
        bonds = self._match(atoms, p)
        if bonds is None:
            return None
        pair = (u, v) if u < v else (v, u)
        bonds[pair] = bonds.get(pair, 0) + 1
        return self._build(bonds, atoms, set(), set())

    def single_pi_bonds(self, atoms) -> List[Tuple[int, int]]:
        """The non-aromatic single bonds between atoms of ``atoms``, in bond order: the bonds that a Kekule
        alternative can make double."""
        out = []
        for bond in self.kekule.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if (u in atoms and v in atoms and bond.GetBondType() == Chem.BondType.SINGLE
                    and not self.mol.GetBondBetweenAtoms(u, v).GetIsAromatic()):
                out.append((u, v) if u < v else (v, u))
        return out


def _is_form(drawing: _Drawing, structure) -> Optional[str]:
    """The canonical SMILES of ``structure`` when it is a charge-only form of the drawing, else None: no unpaired
    electron, every hydrogen count and the number of exocyclic double bonds to an aromatic atom as drawn, and the
    written string reads back as a molecule with the drawing's standard InChIKey, so a form cannot describe another
    molecule."""
    if not drawing.key:
        raise _UndecidedError("inchi")
    if any(atom.GetNumRadicalElectrons() for atom in structure.GetAtoms()):
        return None
    if [atom.GetTotalNumHs() for atom in structure.GetAtoms()] != drawing.base_h:
        return None
    if exocyclic_double_bonds_to_aromatic(structure) != drawing.exocyclic:
        return None
    back = Chem.MolFromSmiles(Chem.MolToSmiles(structure, canonical=False))
    if back is None or back.GetNumAtoms() != drawing.count or Chem.MolToInchiKey(back) != drawing.key:
        return None
    return Chem.MolToSmiles(back)


def _mobile_cations(mol) -> List[int]:
    return [match[0] for match in mol.GetSubstructMatches(MOBILE_N_CATION)]


def _shifts(drawing: _Drawing, lost_atoms, donors, max_moves: int, budget: _Budget):
    """(lost, gained, structure) for every set of up to ``max_moves`` simultaneous shifts of the +1 from ``lost_atoms``
    onto ``donors`` that the pi bonds allow, in a fixed order."""
    for moves in range(1, max_moves + 1):
        if len(lost_atoms) < moves or len(donors) < moves:
            break
        for lost in itertools.combinations(lost_atoms, moves):
            budget.spend()
            beside = drawing.donors_beside(donors, lost)
            for gained in itertools.combinations(beside, moves):
                budget.spend()
                structure = drawing.shifted(lost, gained)
                if structure is not None:
                    yield lost, gained, structure


def charge_shift_forms(mol, *, lost=None, gained=None, max_moves=3, max_configurations=MAX_CONFIGURATIONS,
                       why=None) -> Optional[List[Chem.Mol]]:
    """The charge-only resonance forms of ``mol`` as molecules (the drawn form excluded), under the filter chain of
    ``namer._charge_forms``: the +1 moves between nitrogen atoms along pi bonds (not in a pi system that holds a
    carbanion centre, see the module docstring), every hydrogen count is the drawing's, the standard InChIKey of the
    written form is the drawing's, and the canonical SMILES differs from the drawing's.
    Up to ``max_moves`` shifts happen together. Without ``lost`` and ``gained`` the other Kekule structures of a
    conjugated system (another arrangement of the double bonds of a ring that is not aromatic), of the drawing and of
    every shifted structure whether or not that one is a form, are forms too when their canonical SMILES differs; so a
    placement of the charge gives every one of its Kekule structures.

    ``lost`` restricts the nitrogen cations that give the charge away and ``gained`` the atoms that take it (collections
    of atom indices).  means proven: no form. A molecule with no mobile nitrogen cation has none, and neither has one
    with an unpaired electron (every structure of it keeps the electron, and a structure with one is no form). None
    means undecided; ``why`` (a list) receives one reason code. The forms come in a fixed order and each has its own
    canonical SMILES; they carry the atom indices of ``mol``."""
    try:
        return _charge_shift_forms(mol, lost, gained, max_moves, max_configurations)
    except _UndecidedError as stop:
        _report(why, stop.reason)
        return None


def _charge_shift_forms(mol, lost, gained, max_moves, max_configurations) -> List[Chem.Mol]:
    mobile = _mobile_cations(mol)
    if not mobile:
        return []
    _require_decidable(mol, mobile, radicals=False)
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        return []        # every structure keeps the unpaired electron, and a form with one is no form
    drawing = _Drawing(mol)
    budget = _Budget(max_configurations)
    drawn = Chem.MolToSmiles(mol)
    forms: Dict[str, Chem.Mol] = {}

    def keep(structure) -> None:
        canonical = _is_form(drawing, structure)
        if canonical is not None and canonical != drawn:
            forms.setdefault(canonical, structure)

    def keep_alternatives(source: _Drawing) -> None:
        for atoms in source.kekule_components():
            for u, v in source.single_pi_bonds(atoms):
                budget.spend()
                structure = source.alternative(atoms, u, v)
                if structure is not None:
                    keep(structure)

    donors = drawing.donors(gained)
    shifted = []
    for _lost, _gained, structure in _shifts(drawing, drawing.cations(lost), donors, max_moves, budget):
        keep(structure)
        shifted.append(structure)
    if lost is None and gained is None:
        keep_alternatives(drawing)
        for structure in shifted:
            keep_alternatives(_Drawing(structure))
    return list(forms.values())


def hydron_sites(mol, centre, ring_system, *, why=None,
                 max_configurations=MAX_CONFIGURATIONS) -> Optional[Dict[int, Chem.Mol]]:
    """Site-2 acceptance: {ring atom X: a structure in which the centre is neutral and X, an atom that is not H or C and
    carries a hydrogen, bears +1}, for the ring atoms X of ``ring_system`` (atom indices) other than ``centre``, the
    +1 of the centre moved onto one neutral three-connected nitrogen. It has no exocyclic-double-bond filter and no
    InChIKey filter: the acceptance is RDKit's own for the site. The structures are sanitized and carry the atom
    indices of ``mol``.

    {} means proven: no site by moving the +1 of the centre onto a nitrogen, and none for a centre in a pi system that
    holds a carbanion centre (the module docstring: RDKit finds no site there either). None means undecided; ``why``
    receives one reason code. ``anion`` is reported when no site is found and an anionic atom sits in the pi system of
    the centre: moving the +1 may need the -1 to move too (a zwitterion), which this matching does not draw. Sites
    found are returned whether or not an anion is present."""
    try:
        return _hydron_sites(mol, centre, ring_system, max_configurations)
    except _UndecidedError as stop:
        _report(why, stop.reason)
        return None


def _component_of_atom(drawing: _Drawing, atom: int) -> List[int]:
    """The pi-capable component that holds ``atom``;  when it holds none."""
    k = drawing.component_of.get(atom)
    return [] if k is None else drawing.components[k]


def _hydron_sites(mol, centre, ring_system, max_configurations) -> Dict[int, Chem.Mol]:
    if mol.GetAtomWithIdx(centre).GetFormalCharge() != 1:
        return {}
    _require_decidable(mol, [centre], radicals=True)
    drawing = _Drawing(mol)
    if drawing.p[centre] < 1 or drawing.in_carbanion_system(centre):
        return {}
    budget = _Budget(max_configurations)
    found: Dict[int, Chem.Mol] = {}
    for donor in drawing.donors_beside(drawing.donors(), (centre,)):
        budget.spend()
        structure = drawing.shifted((centre,), (donor,))
        if structure is None:
            continue
        for x in sorted(ring_system):
            atom = structure.GetAtomWithIdx(x)
            if (x != centre and atom.GetAtomicNum() not in (1, 6) and atom.GetFormalCharge() == 1
                    and drawing.base_h[x] >= 1):
                found.setdefault(x, structure)
    if not found and any(mol.GetAtomWithIdx(i).GetFormalCharge() < 0 for i in _component_of_atom(drawing, centre)):
        raise _UndecidedError("anion")
    return found


def charge_placements(mol, *, why=None, max_moves=3,
                      max_configurations=MAX_CONFIGURATIONS) -> Optional[List[Chem.Mol]]:
    """The structures of ``mol`` with the +1 of each mobile nitrogen cation moved onto a neutral three-connected
    nitrogen (up to ``max_moves`` at once), the drawn structure excluded: the other places the charge of one species
    can be drawn, for the neutral analogues of ``validation.protonation_identity``. The structures are sanitized and
    carry the atom indices of ``mol``; no filter beyond a valid structure is applied, as the analogue of a structure is
    read, not named.

     means proven: no placement of a nitrogen +1. None means undecided; ``why`` receives one reason code. Only the +1
    of a nitrogen moves: an anion, an oxonium or a sulfonium that RDKit's enumeration moves too stays where it is
    drawn, so the set of placements can be narrower than RDKit's. The +1 of a pi system that holds a carbanion centre
    stays where it is drawn (the module docstring), as it does in RDKit's enumeration."""
    try:
        return _charge_placements(mol, max_moves, max_configurations)
    except _UndecidedError as stop:
        _report(why, stop.reason)
        return None


def _charge_placements(mol, max_moves, max_configurations) -> List[Chem.Mol]:
    mobile = _mobile_cations(mol)
    if not mobile:
        return []
    _require_decidable(mol, mobile, radicals=True)
    drawing = _Drawing(mol)
    budget = _Budget(max_configurations)
    return [structure for _lost, _gained, structure
            in _shifts(drawing, drawing.cations(), drawing.donors(), max_moves, budget)]
