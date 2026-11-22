"""Spelling check: an acyclic chain of four or more heterounits has a skeletal replacement ('a') PIN
(registered with:func:`orthonym.validation.pin_spelling.register`).

``## **** Skeletal replacement ('a') nomenclature in acyclic chains`` (the Blue Book),
(:23348): "Skeletal replacement ('a') nomenclature rather than substitutive or multiplicative names must be used to
generate preferred IUPAC names for acyclic structures when four or more heterounits are present in a unbranched
chain containing at least one carbon atom and when none of the heteroatoms constitute all or part of the principal
characteristic group of the compound." The paragraph that follows (:23350) defines the unit: "A heterounit is a set
of heteroatoms having a name of its own such as, -SS-, disulfanediyl; -SiH2-O-SiH2-, disiloxane-1,3-diyl;
-SOS-, dithioxanediyl (not -OSiH2O- nor -OSO- that correspond to three consecutive units 'oxysilanediyloxy' and
'oxysulfanediyloxy', respectively)." (:23405): "The same number of characteristic groups that would be
expressed as suffixes in substitutive names must be present in skeletal replacement ('a') names." An ester is named
by functional class nomenclature in a PIN,:31663), so the 'a' group stands in its alkyl word
('dimethyl 2,4,6,8-tetrasilanonan-1-yl phosphate (PIN)',:23401): the reading below looks for the 'a' block
anywhere in the name, enclosures included.

The structure decides whether the rule applies (:func:`a_chain_applies`) and the name decides whether it is met
(:func:`names_an_a_chain`); a name that does not meet it for a molecule the rule applies to is a systematic name,
not the PIN. The reading is conservative on purpose: it applies only to a shape it can read with certainty, so a
molecule it cannot read (a branch that carries heteroatoms, a principal group that may sit off the chain, a
heteroatom with a double bond, a ring, an ion, several fragments) is never a failure. It reads the RDKit
structure and the name only: no naming code, no OPSIN.

The chain (S1 of the lane): a simple path of neutral carbons and plain heteroatoms whose ends are carbon or one of
B, Si, P, Ge; a heterounit counts per run of consecutive heteroatoms; at least four units. Every other part of the
molecule must hang on the path in one of the few ways that leave the chain the one would choose; and a
nitrogen in the chain is a skeletal atom only when it is not the principal amine (S2:, classes 17
and 19,:18162; Polyamines,:26324, '3,6,9,12-tetraazatetradecane-1,14-diamine (PIN)',:26400).
"""
from __future__ import annotations

import re
from typing import Optional

from rdkit import Chem

from ..pin_spelling import SpellingFailure, register
from .units import STEM_RE

RULE = "P-51.4.1.1"

#: the cap on the molecule the reading handles (heavy atoms)
_MAX_HEAVY_ATOMS = 120
#: elements of a plain heteroatom of a chain, and the ones that may end it,:23348,:23350)
_HETERO = frozenset({"B", "N", "O", "Si", "P", "S", "Ge", "Se", "Te"})
_END_HETERO = frozenset({"B", "Si", "P", "Ge"})
_CHALCOGEN = frozenset({"O", "S", "Se", "Te"})
_HALOGEN = frozenset({"F", "Cl", "Br", "I"})
#: a heterounit of three atoms that is one unit (:23350, 'disiloxane-1,3-diyl', 'dithioxanediyl')
_ONE_UNIT_OF_THREE = frozenset({("Si", "O", "Si"), ("S", "O", "S")})
#: the groups that stand as a suffix in front of the amines (Table 4.1:18162, classes 17 and 19; the acyl and oxo
#: groups, the acids and their derivatives, are senior to all three)
_SENIOR = frozenset({"OH", "=O", "SH", "acyl"})

_SINGLE = Chem.BondType.SINGLE
_DOUBLE = Chem.BondType.DOUBLE
_TRIPLE = Chem.BondType.TRIPLE


# --- the name -------------------------------------------------------------------------------------------------

#: The block of replacement ('a') prefixes of an acyclic parent: locants (with an optional lambda convention), an
#: optional multiplier, the prefix of a skeletal heteroatom; then an acyclic stem and the beginning of an ending
#: ('ane', 'a-' of 'hexadeca-3,5-diene', '-2-ene', 'en', 'yn', 'yl'). The reader keeps its own copy of the pattern the
#: writer's vocabulary uses (``rules/pin_vocabulary.py::_ACYCLIC_A_NAME``): a check must not import the writer. The
#: copy is never narrower than the original (a test holds it so on every name of the Blue Book) and reads what the
#: original does not: the multiplying prefixes of eleven and more and, the Blue Book,
#::2811: 'undeca', 'icosa', 'docosa', 'triaconta'; "The letter 'i' in 'icosa' is elided after a vowel") and the
#: stems of forty-one carbon atoms and more ('hexahexacontane', 'triheptacontan-73-oic acid'), which an 'a' name of a
#: long polyether has. A name the pattern does not read would be a failure, so it errs toward reading more.
_A_PREFIX = r"(?:oxa|thia|selena|tellura|aza|phospha|arsa|stiba|bisma|sila|germa|stanna|plumba|bora)"
_UNIT_TERM = r"(?:hen|do|tri|tetra|penta|hexa|hepta|octa|nona)"
_TEN_TERM = r"(?:tetra|penta|hexa|hepta|octa|nona)"
_MULTIPLIER = (r"(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca|undeca|dodeca|trideca|tetradeca|pentadeca|hexadeca"
               r"|heptadeca|octadeca|nonadeca|(?:hen)?icosa|(?:do|tri|tetra|penta|hexa|hepta|octa|nona)cosa"
               r"|" + _UNIT_TERM + r"?(?:tria|" + _TEN_TERM + r")conta)?")
_A_SEGMENT = r"\d+(?:λ\d+)?(?:,\d+(?:λ\d+)?)*-" + _MULTIPLIER + _A_PREFIX + r"-?"
_HIGH_STEM = (r"(?:" + _UNIT_TERM + r"?" + _TEN_TERM + r"cont|(?:" + _UNIT_TERM
              + r"|deca|undeca|dodeca|trideca|tetradeca|pentadeca|hexadeca|heptadeca|octadeca|nonadeca|icosa)?hect)")
_A_NAME = re.compile(
    r"(?<![A-Za-z])(?:" + _A_SEGMENT + r")+(?:" + STEM_RE + r"|" + _HIGH_STEM + r")"
    r"(?=a?n|a-|a\d|-\d|en|yn|yl|a,)")


def names_an_a_chain(name: str) -> bool:
    """True when ``name`` has a block of skeletal replacement ('a') prefixes on an acyclic stem anywhere in it
    ('2,5,8,11-tetraoxadodecane', '3,6,9,12-tetraoxatetradecanedioic acid', 'dimethyl 2,4,6,8-tetrasilanonan-1-yl
    phosphate'); a name of a ring ('1,4,7,10-tetraoxacyclopentadecane') has none."""
    return bool(name) and _A_NAME.search(name) is not None


# --- the structure --------------------------------------------------------------------------------------------

def _oxo_like(atom) -> bool:
    """An atom with a double bond to N, O, S or Se, or a triple bond to N."""
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom).GetSymbol()
        kind = bond.GetBondType()
        if (kind == _DOUBLE and other in ("N", "O", "S", "Se")) or (kind == _TRIPLE and other == "N"):
            return True
    return False


def _plain(atom) -> bool:
    """A heteroatom that is a skeletal atom of an acyclic chain: neutral, saturated, without a radical, with
    two bonds when it is a chalcogen, and neither it nor a neighbour carries an oxo-like double or triple bond
    (so no amide or ester atom, no sulfone, no imine)."""
    symbol = atom.GetSymbol()
    if symbol not in _HETERO or atom.GetFormalCharge() or atom.GetIsAromatic() or atom.GetNumRadicalElectrons():
        return False
    if any(b.GetBondType() != _SINGLE for b in atom.GetBonds()):
        return False
    if symbol in _CHALCOGEN and atom.GetDegree() != 2:
        return False
    return not _oxo_like(atom) and not any(_oxo_like(n) for n in atom.GetNeighbors())


#: a carboxylic acid, its noncyclic ester, acid halide or anhydride: a group senior to an amide,
#::18162), so no amide group is the principal characteristic group; a lactone or cyclic anhydride is a pseudoketone,
#: below the amides (class 9 "lactones and other cyclic esters are named as heterocycles; see 16 below",:18186)
_SENIOR_TO_AMIDE = Chem.MolFromSmarts("[CX3;!R;$(C[#6])](=O)!@[OX2,F,Cl,Br,I]")


def _amide_nitrogen(atom) -> bool:
    """The nitrogen of a plain amide: neutral, acyclic, every bond single, every neighbour a
    carbon, and exactly one of them a C=O carbon whose other heavy neighbour is one carbon (no urea, carbamate,
    imide, hydrazide or hydroxamic acid)."""
    if (atom.GetSymbol() != "N" or atom.GetFormalCharge() or atom.GetIsAromatic() or atom.IsInRing()
            or atom.GetNumRadicalElectrons() or any(b.GetBondType() != _SINGLE for b in atom.GetBonds())
            or any(n.GetSymbol() != "C" for n in atom.GetNeighbors())):
        return False
    acyl = []
    for n in atom.GetNeighbors():
        oxo = [b for b in n.GetBonds() if b.GetBondType() == _DOUBLE and b.GetOtherAtom(n).GetSymbol() == "O"]
        if oxo:
            others = [x for x in n.GetNeighbors() if x.GetIdx() != atom.GetIdx()
                      and x.GetIdx() != oxo[0].GetOtherAtom(n).GetIdx()]
            if len(others) != 1 or others[0].GetSymbol() != "C" or n.GetDegree() != 3:
                return False
            acyl.append(n)
        elif _oxo_like(n):
            return False
    return len(acyl) == 1


def _run_units(run) -> Optional[int]:
    """The heterounits of a run of consecutive heteroatoms (the element symbols), or None for a run that is not
    read: one atom is one unit; two are one when they are the same element ('disulfanediyl') and two when not;
    three are one for Si-O-Si and S-O-S (:23350) and three for X-Y-X ('oxysilanediyloxy'); the others are not read."""
    if len(run) == 1:
        return 1
    if len(run) == 2:
        return 1 if run[0] == run[1] else 2
    if len(run) == 3:
        if tuple(run) in _ONE_UNIT_OF_THREE:
            return 1
        return 3 if run[0] == run[2] and run[1] != run[0] else None
    return None


def _heterounits(mol, path) -> Optional[int]:
    """The number of heterounits of ``path``, or None when a run of it is not read."""
    total, run = 0, []
    for idx in list(path) + [None]:
        symbol = mol.GetAtomWithIdx(idx).GetSymbol() if idx is not None else "C"
        if symbol != "C":
            run.append(symbol)
            continue
        if run:
            units = _run_units(run)
            if units is None:
                return None
            total += units
            run = []
    return total


def _paths(mol, allowed, ends):
    """Every simple path of the acyclic molecule between two of ``ends`` through ``allowed`` atoms, as lists of
    atom indices (the acyclic atoms form a forest: one path per pair)."""
    for start in ends:
        parent = {start: None}
        stack = [start]
        while stack:
            here = stack.pop()
            for nb in mol.GetAtomWithIdx(here).GetNeighbors():
                j = nb.GetIdx()
                if j in allowed and j not in parent:
                    parent[j] = here
                    stack.append(j)
        for end in ends:
            if end > start and end in parent:
                path, at = [], end
                while at is not None:
                    path.append(at)
                    at = parent[at]
                yield path


def _component(mol, root, path_set):
    """The atom indices of the part of the molecule that hangs from ``root`` once the path is removed."""
    seen, stack = {root}, [root]
    while stack:
        for nb in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
            j = nb.GetIdx()
            if j not in seen and j not in path_set:
                seen.add(j)
                stack.append(j)
    return seen


def _clean(atom) -> bool:
    return atom.GetFormalCharge() == 0 and atom.GetNumRadicalElectrons() == 0 and not atom.GetIsAromatic()


def _acyl_group(mol, comp) -> bool:
    """A component that is an acyl group: only C, N, O and halogens; exactly one oxo-like carbon; every N and O
    of the group is bonded to that carbon, which has at most two of them (an acid, an ester, an acid halide, an
    amide, a nitrile, an acylamino or acyloxy group; not a carbonate, a carbamate, a urea or a nitro group)."""
    atoms = [mol.GetAtomWithIdx(i) for i in comp]
    if not all(_clean(a) and a.GetSymbol() in ({"C", "N", "O"} | _HALOGEN) for a in atoms):
        return False
    oxo_carbons = [a for a in atoms if a.GetSymbol() == "C" and _oxo_like(a)]
    if len(oxo_carbons) != 1:
        return False
    carbon = oxo_carbons[0]
    near = {n.GetIdx() for n in carbon.GetNeighbors() if n.GetSymbol() in ("N", "O")}
    return len(near) <= 2 and all(a.GetIdx() in near for a in atoms if a.GetSymbol() in ("N", "O"))


def _guanidine_nitrogen(atom) -> bool:
    """The N of an Arg-like side chain between a carbon and a guanidine carbon (a neutral acyclic carbon with three
    nitrogen neighbours and one C=N bond), in either tautomer. With a group senior to amides it is a skeletal atom:
     (the Blue Book) '9-imino-2,8,10,16-tetraazaheptadecanediimidamide (PIN)' puts the nitrogens beside
    a C=NH carbon in the chain, so a chain through an Arg side chain counts it,:20916)."""
    if (atom.GetSymbol() != "N" or atom.GetDegree() != 2 or atom.IsInRing() or atom.GetFormalCharge()
            or atom.GetNumRadicalElectrons()):
        return False
    for n in atom.GetNeighbors():
        if (n.GetSymbol() == "C" and n.GetDegree() == 3 and not n.IsInRing() and not n.GetFormalCharge()
                and all(x.GetSymbol() == "N" for x in n.GetNeighbors())
                and any(b.GetBondType() == _DOUBLE for b in n.GetBonds())):
            return True
    return False


def _kind(mol, path, host, root, amides_skeletal: bool = False):
    """What hangs from the path atom ``host`` at ``root`` (the neighbour that is not on the path): one of
    'halogen', 'OH', '=O', 'SH', 'NH2', 'alkyl' (carbon and halogen atoms only), 'NHR' (an NHR or NR2 amine with
    saturated carbon parts), 'acyl', or None for anything else. The leaves stand on a path carbon, an acyl group on an end atom of the path."""
    path_set = set(path)
    comp = _component(mol, root, path_set)
    bond = mol.GetBondBetweenAtoms(host, root)
    first = mol.GetAtomWithIdx(root)
    on_carbon = mol.GetAtomWithIdx(host).GetSymbol() == "C"
    symbol, kind = first.GetSymbol(), bond.GetBondType()
    if len(comp) == 1 and _clean(first) and on_carbon:
        if symbol in _HALOGEN and kind == _SINGLE:
            return "halogen"
        if symbol == "O" and kind in (_SINGLE, _DOUBLE):
            return "OH" if kind == _SINGLE else "=O"
        if symbol == "S" and kind == _SINGLE:
            return "SH"
        if symbol == "N" and kind == _SINGLE:
            return "NH2"
        if symbol == "N" and kind == _DOUBLE and amides_skeletal and host in (path[0], path[-1]):
            return "=NH"                # the guanidine carbon that ends a chain through an Arg side chain
    if kind != _SINGLE:
        return None
    atoms = [mol.GetAtomWithIdx(i) for i in comp]
    if all(_clean(a) and a.GetSymbol() in ({"C"} | _HALOGEN) for a in atoms):
        return "alkyl"
    # the principal amine NHR or NR2 on a path carbon: the nitrogen and saturated carbons only, every bond of the
    # nitrogen single to a carbon ('4-ethoxy-N,N,N-trimethyl-3,5,8-trioxa-4-phosphadecan-10-aminium (PIN)',:23379)
    if (on_carbon and symbol == "N" and _clean(first)
            and all(n.GetSymbol() == "C" for n in first.GetNeighbors())
            and all(b.GetBondType() == _SINGLE for b in first.GetBonds())
            and all(_clean(a) and a.GetSymbol() == "C" for a in atoms if a.GetIdx() != root)
            and all(b.GetBondType() == _SINGLE for i in comp for b in mol.GetAtomWithIdx(i).GetBonds())):
        return "NHR"
    if host in (path[0], path[-1]) and _acyl_group(mol, comp):
        return "acyl"
    # an acylamino group on a path carbon when an amide is no principal group ('7,14,21,28-tetraacetamido',:23428)
    if amides_skeletal and on_carbon and _amide_nitrogen(first) and _acyl_group(mol, comp - {root}):
        return "acylamino"
    if amides_skeletal:
        # any other branch of a chain whose amide nitrogens are skeletal (a side chain of a peptide, a ring): it
        # holds no group senior to an amide (``_senior_groups_at_ends``), so it is a prefix of the chain, and a
        # ring as a prefix leaves the 'a' PIN of the chain ('4-cyclohexyl-8-methyl-4-oxo-3,5-dioxa-8-thia-
        # 4λ5-phosphatetradecan-8-ium iodide (PIN)',, the Blue Book)
        return "other"
    return None


def _n_policy_ok(mol, path, hung) -> bool:
    """S2: a nitrogen in the chain is a skeletal atom only when the principal group is another one (a senior
    group hangs on the chain), or, for a polyamine, when the two NH2 groups sit one on each end atom and no
    chain nitrogen carries a substituent,:26324; ':26400')."""
    nitrogens = [i for i in path if mol.GetAtomWithIdx(i).GetSymbol() == "N"]
    if any(kind == "NHR" for _host, kind in hung):
        # the substituted amine is the principal group only without a senior group or another amine
        return not nitrogens and all(kind not in _SENIOR and kind not in ("NH2", "NHR")
                                     for _host, kind in hung if kind != "NHR") and \
            sum(1 for _host, kind in hung if kind == "NHR") == 1
    if not nitrogens or any(kind in _SENIOR for _host, kind in hung):
        return True
    amino_hosts = sorted(host for host, kind in hung if kind == "NH2")
    return (amino_hosts == sorted((path[0], path[-1]))
            and all(mol.GetAtomWithIdx(i).GetDegree() == 2 for i in nitrogens))


def _senior_groups_at_ends(mol, path, senior) -> bool:
    """Every group senior to amides (the carbonyl carbons ``senior``) stands at an end of ``path``, one at most per
    end: on the end atom itself or in an acyl group hanging from it. A second one at an end, or one elsewhere, makes
    a parent with more principal groups senior, the Blue Book, 'The senior parent structure has the
    maximum number of substituents corresponding to the principal characteristic group (suffix)'): the diacid of an
    N-acyl aspartic acid, whose junction nitrogen or oxygen is no chain atom:23430)."""
    covered = 0
    path_set = set(path)
    for end in {path[0], path[-1]}:
        here = 1 if end in senior else 0
        for nb in mol.GetAtomWithIdx(end).GetNeighbors():
            if nb.GetIdx() not in path_set:
                comp = _component(mol, nb.GetIdx(), path_set)
                if comp & senior:
                    if not _acyl_group(mol, comp) or any(mol.GetAtomWithIdx(i).IsInRing() for i in comp):
                        return False    # a senior group deeper in a branch (on a ring): not read
                    here += len(comp & senior)
        if here > 1:
            return False
        covered += here
    return covered == len(senior)


def _is_a_chain(mol, path, amides_skeletal: bool = False, senior=frozenset()) -> bool:
    units = _heterounits(mol, path)
    if units is None or units < 4:
        return False
    path_set = set(path)
    hung = []
    for host in path:
        for nb in mol.GetAtomWithIdx(host).GetNeighbors():
            if nb.GetIdx() in path_set:
                continue
            kind = _kind(mol, path, host, nb.GetIdx(), amides_skeletal)
            if kind is None:
                return False
            hung.append((host, kind))
    if any(kind == "other" for _host, kind in hung) and not any(_amide_nitrogen(mol.GetAtomWithIdx(i)) for i in path):
        return False            # an unread branch is read only on a chain of skeletal amide nitrogens
    return _senior_groups_at_ends(mol, path, senior) and _n_policy_ok(mol, path, hung)


def a_chain_applies(mol) -> bool:
    """True when gives the acyclic structure ``mol`` a skeletal replacement ('a') PIN, as far as the
    reading above can tell: a single fragment without a ring, of at most 120 heavy atoms, in which some chain of
    four or more heterounits carries nothing the reading does not know, and whose heteroatoms are no part of
    the principal characteristic group."""
    if mol is None or mol.GetNumHeavyAtoms() > _MAX_HEAVY_ATOMS:
        return False
    if len(Chem.GetMolFrags(mol)) != 1:
        return False
    # (:23405): with a group senior to amides present, an amide nitrogen of the chain is a skeletal atom
    # and its C=O an 'oxo' prefix: 'methyl 7,14,21,28-tetraacetamido-2,9,16,23-tetraoxo-3,10,17,24-
    # tetraazatriacontan-30-oate (PIN; an ester is senior to an amide or a ketone)' (:23428)
    senior = frozenset(m[0] for m in mol.GetSubstructMatches(_SENIOR_TO_AMIDE))
    amides_skeletal = bool(senior)
    cyclic = mol.GetNumBonds() != mol.GetNumAtoms() - 1
    if cyclic and not amides_skeletal:
        return False            # a ring is read only beside a chain of skeletal amide nitrogens (below)
    plain = {a.GetIdx() for a in mol.GetAtoms() if not a.IsInRing()
             and (_plain(a) or (amides_skeletal and (_amide_nitrogen(a) or _guanidine_nitrogen(a))))}
    if len(plain) < 4 or (cyclic and not any(_amide_nitrogen(mol.GetAtomWithIdx(i)) for i in plain)):
        return False
    carbons = {a.GetIdx() for a in mol.GetAtoms()
               if a.GetSymbol() == "C" and a.GetFormalCharge() == 0 and not a.IsInRing()}
    allowed = carbons | plain
    ends = sorted(i for i in allowed
                  if i in carbons or mol.GetAtomWithIdx(i).GetSymbol() in _END_HETERO)
    return any(_is_a_chain(mol, path, amides_skeletal, senior) for path in _paths(mol, allowed, ends))


def _hanging_path(mol, ring_atoms, root):
    """The atoms of the part of the molecule that hangs from the ring atom ``root`` by one bond, in order from the atom bonded
    to the ring, when that part is an unbranched path of neutral atoms (no ring, no other attachment); else None."""
    part, previous = [], None
    here = root
    while True:
        part.append(here)
        onward = [n.GetIdx() for n in mol.GetAtomWithIdx(here).GetNeighbors()
                  if n.GetIdx() != previous and n.GetIdx() not in ring_atoms]
        ring_nb = [n.GetIdx() for n in mol.GetAtomWithIdx(here).GetNeighbors() if n.GetIdx() in ring_atoms]
        if len(onward) > 1 or (len(part) > 1 and ring_nb) or len(part) > _MAX_HEAVY_ATOMS:
            return None
        if not onward:
            return part
        previous, here = here, onward[0]


def a_substituent_chain_applies(mol) -> bool:
    """True when a chain that hangs from a ring by one bond has four or more heterounits between its carbon ends.

     (the Blue Book) gives the 'a' name to "four or more heterounits... in a unbranched chain
    containing at least one carbon atom", (:23430) ends the chain in carbon, so the atom bonded to the ring is
    outside the chain when it is a heteroatom ('tetraoxatridecan-13-yl' of a '(...)oxy' group), and the substituent
    group of the book's own example, 'N-(14-amino-3,6,9,12-tetraazatetradecan-1-yl)' (:23415, "since four heteroatoms are
    also present in the N-substituent group, it must also be named by skeletal replacement ('a') nomenclature"),
    is named the same way. The reading is conservative as the whole-chain one is: one neutral fragment, the hanging part an
    unbranched path of saturated atoms with a carbon at the free end, nothing else."""
    if mol is None or mol.GetNumHeavyAtoms() > _MAX_HEAVY_ATOMS or len(Chem.GetMolFrags(mol)) != 1:
        return False
    ring_atoms = {i for ring in mol.GetRingInfo().AtomRings() for i in ring}
    if not ring_atoms:
        return False
    for root in sorted({n.GetIdx() for i in ring_atoms for n in mol.GetAtomWithIdx(i).GetNeighbors()} - ring_atoms):
        if sum(1 for n in mol.GetAtomWithIdx(root).GetNeighbors() if n.GetIdx() in ring_atoms) != 1:
            continue
        part = _hanging_path(mol, ring_atoms, root)
        if part is None:
            continue
        # a terminal -OH / -SH is a suffix group on the chain's carbon end, not a unit of the chain ('...tetraoxatetradecan-1-ol')
        if (len(part) > 2 and mol.GetAtomWithIdx(part[-1]).GetSymbol() in ("O", "S")
                and mol.GetAtomWithIdx(part[-1]).GetDegree() == 1):
            part = part[:-1]
        atoms = [mol.GetAtomWithIdx(i) for i in part]
        if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetIsAromatic() for a in atoms):
            continue
        symbols = [a.GetSymbol() for a in atoms]
        if symbols[-1] != "C" or any(sym != "C" and sym not in _HETERO for sym in symbols):
            continue
        if any(b.GetBondType() != _SINGLE for a in atoms for b in a.GetBonds()):
            continue
        chain = part[1:] if symbols[0] != "C" else part
        # a nitrogen of the chain is read only for a whole chain (``_n_policy_ok``): here the units are carbon and chalcogens
        if chain and all(mol.GetAtomWithIdx(i).GetSymbol() == "C"
                         or (mol.GetAtomWithIdx(i).GetSymbol() in _CHALCOGEN and _plain(mol.GetAtomWithIdx(i)))
                         for i in chain):
            units = _heterounits(mol, chain)
            if units is not None and units >= 4 and mol.GetAtomWithIdx(chain[0]).GetSymbol() == "C":
                return True
    return False


@register(RULE)
def a_substituent_chain_name_check(mol, name) -> Optional[SpellingFailure]:
    """A failure when a chain of four or more heterounits hangs from a ring of ``mol`` and ``name`` has no 'a' block."""
    if not name or not a_substituent_chain_applies(mol) or names_an_a_chain(name):
        return None
    return SpellingFailure(
        RULE, "skeletal replacement 'a' name required: a chain hanging from a ring has four or more heterounits "
              "between its carbon ends (P-51.4.1.1, P-51.4.1.4), and the name has no 'a' prefix block on an acyclic stem")


#: the end of the word after an 'a' stem when the 'a' chain is a parent: the suffix of an acid, ester, acyl halide,
#: anhydride or peroxy acid ('...tetraazatriacontan-30-oate',, the Blue Book; '...pentadec-2-enoic
#: acid'), or the alcohol word of an ester ('3,6,9,12,15,18,21,24,27-nonaoxahentriacontan-1-yl acetate', '...-1,14-diyl
#: diacetate',. The word runs to a space or the end without an enclosing mark: a prefix ('...-1-amido}',
#: '...-1-yl)amino') never matches.
_ACID_SUFFIX_AFTER_STEM = re.compile(r"[a-z0-9,\-]*?(?:o(?:ic acid|ate|yl |ic anhydride|peroxoic acid)|yl )")


def names_an_a_acid_parent(name: str) -> bool:
    """True when an 'a' block of ``name`` carries the acid (ester, acyl halide) suffix itself, or is the alcohol word of
    an ester: the 'a' chain is a parent, not a prefix ('...-tetraazapentadecan-1-amido}...propanoic acid' has the block on a prefix only)."""
    return bool(name) and any(_ACID_SUFFIX_AFTER_STEM.match(name, m.end()) for m in _A_NAME.finditer(name))


@register(RULE)
def a_chain_name_check(mol, name) -> Optional[SpellingFailure]:
    """A failure when ``mol`` is a chain that gives an 'a' PIN and ``name`` has no 'a' block; None when
    the rule does not apply (or is not read) or the name has the block. With a group senior to amides (an acid, its
    ester or acyl halide) the block must carry that group's suffix: (:18875) puts the principal group on the
    parent, so an 'a' block on a prefix leaves the chain of the structure unnamed as the 'a' parent."""
    if not name or not a_chain_applies(mol):
        return None
    if mol.HasSubstructMatch(_SENIOR_TO_AMIDE):
        if names_an_a_acid_parent(name):
            return None
    elif names_an_a_chain(name):
        return None
    return SpellingFailure(
        RULE, "skeletal replacement 'a' name required: the acyclic chain has four or more heterounits, none "
              "of them part of the principal characteristic group (P-51.4.1.1), and the name has no 'a' prefix "
              "block on an acyclic stem")


def _principal_amine_with_a_branch(mol) -> bool:
    """A neutral acyclic amine whose one nitrogen (single bonds to carbons only, no OH or SH in the molecule) carries
    a branch that is an 'a' chain: four or more separate O or S atoms on a carbon path from the nitrogen's carbon."""
    if mol is None or mol.GetRingInfo().NumRings() or len(Chem.GetMolFrags(mol)) != 1:
        return False
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return False
    nitrogens = [a for a in mol.GetAtoms() if a.GetSymbol() == "N"]
    if len(nitrogens) != 1 or any(n.GetSymbol() != "C" for n in nitrogens[0].GetNeighbors()):
        return False
    if any(a.GetSymbol() in ("O", "S") and a.GetDegree() == 1 for a in mol.GetAtoms()):
        return False
    n_idx = nitrogens[0].GetIdx()
    for nb in nitrogens[0].GetNeighbors():
        branch, stack = set(), [nb.GetIdx()]
        while stack:
            a = stack.pop()
            if a in branch or a == n_idx:
                continue
            branch.add(a)
            stack.extend(x.GetIdx() for x in mol.GetAtomWithIdx(a).GetNeighbors())
        if sum(1 for i in branch if mol.GetAtomWithIdx(i).GetSymbol() in ("O", "S")) >= 4:
            return True
    return False


_A_AMINE_PARENT = re.compile(_A_NAME.pattern + r"an-\d+-amine$")


@register(RULE)
def a_chain_amine_parent_check(mol, name) -> Optional[SpellingFailure]:
    """A failure when the principal amine of ``mol`` has an 'a' chain on its nitrogen and the parent of ``name`` is
    not an 'a' chain: makes the 'a' chain the parent ('N-methyl-N-(2,5,8,11-tetraoxatridecan-13-yl)-
    2,5,8,11-tetraoxatridecan-13-amine'); an 'a' block only inside a prefix of a substitutive parent is not the PIN."""
    if not name or not _principal_amine_with_a_branch(mol) or _A_AMINE_PARENT.search(name):
        return None
    return SpellingFailure(
        RULE, "the parent of a principal amine with an 'a' chain on its nitrogen is that 'a' chain (P-51.4.1.1)")


# --- cations ------------------------------------------------------------------------------------------

#: an onium centre that counts as one skeletal heteroatom: element -> its number of carbon neighbours,:41752)
_ONIUM_CENTRES = {"S": 3, "Se": 3, "Te": 3, "P": 4, "As": 4}
_ONIUM_ATOMS = frozenset({"C", "O", "S", "Se", "Te", "P", "As", "Si", "Ge", "B"})


def _onium_centre(atom) -> bool:
    return (atom.GetFormalCharge() == 1 and _ONIUM_CENTRES.get(atom.GetSymbol()) == atom.GetDegree()
            and all(n.GetSymbol() == "C" for n in atom.GetNeighbors()))


def onium_a_chain_applies(mol) -> bool:
    """True when the cation of ``mol`` (the whole molecule, or the one carbon part of a salt whose other parts are
    carbon-free anions) has an 'a' PIN by method (1) (``### **** SKELETAL REPLACEMENT ('a') NOMENCLATURE
    FOR CATIONS``, the Blue Book): an acyclic cation without nitrogen (an aminium nitrogen is the principal
    group,:41775), every atom of this reading, single bonds at the heteroatoms, and a path between two chain ends
    (carbon, or B, Si, P, Ge) with four heteroatoms, no two adjacent, an onium centre among them
    ('3,12-dimethyl-6,9-dioxa-3,12-dithiatetradecane-3,12-diium (PIN)',:41785)."""
    if mol is None:
        return False
    parts = Chem.GetMolFrags(mol, asMols=True)
    organic = [p for p in parts if any(a.GetSymbol() == "C" for a in p.GetAtoms())]
    if len(organic) != 1 or any(Chem.GetFormalCharge(p) >= 0 for p in parts if p is not organic[0]):
        return False
    cation = organic[0]
    if cation.GetNumHeavyAtoms() > 60 or cation.GetNumBonds() != cation.GetNumAtoms() - 1:
        return False
    centres = [a for a in cation.GetAtoms() if _onium_centre(a)]
    if not centres or any(
            (a.GetFormalCharge() and not _onium_centre(a)) or a.GetSymbol() not in _ONIUM_ATOMS
            or a.GetNumRadicalElectrons() or (a.GetSymbol() != "C" and any(b.GetBondType() != _SINGLE
                                                                              for b in a.GetBonds()))
            for a in cation.GetAtoms()):
        return False
    ends = [a.GetIdx() for a in cation.GetAtoms() if a.GetSymbol() in {"C"} | _END_HETERO]
    for i, start in enumerate(ends):
        for end in ends[i + 1:]:
            path = Chem.GetShortestPath(cation, start, end)
            hetero = [k for k, idx in enumerate(path) if cation.GetAtomWithIdx(idx).GetSymbol() != "C"]
            if (len(hetero) >= 4 and all(b - a > 1 for a, b in zip(hetero, hetero[1:]))
                    and any(_onium_centre(cation.GetAtomWithIdx(path[k])) for k in hetero)):
                return True
    return False


@register("P-73.4")
def onium_a_chain_name_check(mol, name) -> Optional[SpellingFailure]:
    """A failure when the cation of ``mol`` has an 'a' PIN (:func:`onium_a_chain_applies`) and ``name`` has no 'a'
    block: the substitutive '{2-[2-(2-methoxyethoxy)ethoxy]ethyl}di(methyl)sulfanium iodide' is not the PIN."""
    if not name or names_an_a_chain(name) or not onium_a_chain_applies(mol):
        return None
    return SpellingFailure(
        "P-73.4", "skeletal replacement 'a' name required: the cation has four heteroatoms on a chain with an onium "
                  "centre among them (P-73.4 method (1)), and the name has no 'a' prefix block")
