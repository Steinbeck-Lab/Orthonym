"""Substituent prefixes of nitrogen and oxygen groups the book composes from their parts
(roadmap item 12a).

The writers of the best-effort tier (the universal floor, ``universal_substituent``, and
the terminal-fragment writer, ``rules.terminal_fragment``) spell a branch that no leaf
names as a skeletal replacement ('a') chain: ``=N-OH`` as '2-oxa-1-azaethan-1-ylidene',
``-N=N-Ph`` as '2-phenyl-1,2-diazaeth-1-en-1-yl', ``-OOH`` as '1,2-dioxaethan-1-yl'. An
'a' chain "must be terminated by a C atom or one of the following heteroatoms: P, As,
Sb, Bi, Si, Ge, Sn, Pb, B, Al, Ga, In, or Tl", the Blue Book, section
 General rules), so none of these is a book name. The book names each one by a
substituent prefix:

* ``=N-OH``, ``=N-OR``, ``=N-R`` 'hydroxyimino', '(methoxyimino)', '(methylimino)'
  , oximes: ':38485' "4-[(ethoxyimino)methyl]benzene-1-sulfonic acid
  (PIN)",:38460 "3-(hydroxyimino)butanal (PIN)"; ``imino`` is "used only to designate
  =NH as a substituent", Changes from the 1979 edition 7(b),:1695, so a substituted
  imine nitrogen is cited as the prefix of its substituent plus 'imino');
* ``=N-NR2`` '(R2)hydrazinylidene': "'hydrazinylidene' for H2N-N=" (Changes from the
  1979 edition 7(j),:1703), "(dimethylcarbamoyl)hydrazinylidene (preferred prefix)"
  ,:24611);
* ``-N=N-R`` '(R)diazenyl': "The prefix 'diazenyl' is a preselected prefix"
  Substitution of diazene,:38732), "(methyldiazenyl)acetic acid (PIN)" (:38740);
* ``-NR-NR2`` '(locants)hydrazin-1-yl': 'hydrazinyl (preselected prefix)',:16037,
  :2911); a substituent on a hydrazine unit cites its locant and the free valence's
  ,:2869; '4-(1-acetyl-2-ethylhydrazin-1-yl)benzoic acid (PIN)',
  "Substituent groups derived from hydrazides",:34042; '2-nitrohydrazin-1-yl (preselected
  prefix)', "Substituent groups derived from the hydrazides of nitric acid and
  nitrous acid",:36437), and the locants are left out only where no isomer can be generated
  ,:2953: 'ethylidenehydrazinyl (preferred prefix) (not 2-ethylidenehydrazin-1-yl)',
  :2959;,:3007, a unit in which every position holds the same substituent);
* ``-OO-R`` '(R)peroxy' and ``-OOH`` 'hydroperoxy': "The prefix 'peroxy', not 'dioxy',
  is retained for the group -OO-" and "The prefix 'hydroperoxy' is formed by
  concatenation to describe the group -OOH as a substituent" Hydroperoxides,
  :27944); "the prefixes 'R'-peroxy' (not R'-dioxy)" Peroxides, disulfides,
  diselenides, and ditellurides,:27858), "(methylperoxy)ethane (PIN)" (:27870);
* ``-N=CR2`` with a defined C=N geometry '[(E)-R2-methylidene]amino' ? the
  amino prefix with its ylidene substituent, as ``universal_substituent._hetero_root_leaf``
  builds it for the geometry-free case; the descriptor is cited at the front of the
  ylidene prefix,,:44643);
* ``-NR2(+)-O(-)`` '(R2)(oxido)azaniumyl' Amine oxides, imine oxides, and
  chalcogen analogues,:26648 "[dimethyl(oxido)azaniumyl]methyl");
* ``=N(+)(R)-O(-)`` '(R)(oxido)azaniumylidene',:43386);
* ``-N=O`` 'nitroso' Nitro and nitroso compounds: "the prefixes 'nitro' and
  'nitroso'",:25935);
* ``-O-N<`` '(R-amino)oxy', an oxime ether '[(R-ylidene)amino]oxy',
  :27633 'R-oxy'; the nitrogen prefix of an oxime is the amino prefix with its ylidene
  substituent as above);
* ``=[OR]+`` '(R)oxidaniumylidene',:42350 "(methyloxidaniumylidene)");
* ``=[NR2]+`` '(R2)azaniumylidene': "(1) all prefix names are formed by adding the
  suffixes 'yl', 'ylidene', etc. to the cation name" Cationic prefix names,
  :42298), 'azaniumyl (preselected prefix)' (:42303). The preferred prefix of =N(CH3)2+
  is 'N-methylmethanaminiumylidene' (:42322); the composed form built here is valid,
  never a PIN, and is recorded as a non-PIN fragment.

``group_shape`` reads the structure and returns the parts the host must name (each by its
own recursion, so stereo, charges and nested groups are the host's business);
``compose_group`` joins the parts' names. Both are pure: the same input always gives the
same output. A shape the module does not build, any charge it does not spell and any
isotope give ``None``, and the host keeps its previous spelling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Sequence, Tuple

from rdkit import Chem

_SINGLE = Chem.BondType.SINGLE

#: kinds a writer builds only after its own heteroatom-rooted composer declined (the
#: composer spells '-O-NH-R' itself where it can)
LATE_KINDS = frozenset({"aminooxy"})
_DOUBLE = Chem.BondType.DOUBLE


@dataclass(frozen=True)
class GroupPart:
    """One substituent hanging on the group: ``attach`` is its atom bonded to the group's
    ``host`` atom, ``atoms`` the part's atoms, ``locant`` the hydrazine-unit locant of the
    host atom (1 or 2) or ``None``."""
    attach: int
    atoms: FrozenSet[int]
    host: int
    locant: Optional[int] = None
    order: int = 1                  # bond order host-attach: 1 for a -yl part, 2 for -ylidene


@dataclass(frozen=True)
class GroupShape:
    kind: str                       # imino | hydrazinylidene | diazenyl | hydrazinyl | peroxy
    # | hydroperoxy | azaniumylidene | oxidaniumylidene | iminyl
    # | nitroso | oxidoazaniumyl | oxidoazaniumylidene | aminooxy
    parts: Tuple[GroupPart, ...]
    unit: FrozenSet[int]            # the atoms of the group itself (not of any part)
    charged: FrozenSet[int] = frozenset()  # atoms whose (genuine ionic) charge the prefix spells
    internal: FrozenSet[int] = frozenset()  # atoms of a semipolar bond (N+-O-) the prefix spells
    stereo_bond: Optional[Tuple[int, int]] = None  # an internal double bond to cite (diazenyl)
    acylhydrazide: bool = False     # hydrazinyl: R-CO- or R-SO2- on the far nitrogen


def _plain(atom, charge: int = 0) -> bool:
    return (atom.GetFormalCharge() == charge and not atom.GetIsotope()
            and not atom.GetNumRadicalElectrons() and not atom.IsInRing())


def _side(mol, start: int, group: FrozenSet[int], blocked: FrozenSet[int]) -> FrozenSet[int]:
    """The atoms of ``group`` reachable from ``start`` without crossing ``blocked``."""
    seen, stack = set(), [start]
    while stack:
        cur = stack.pop()
        if cur in seen or cur in blocked:
            continue
        seen.add(cur)
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            k = nb.GetIdx()
            if k in group and k not in seen and k not in blocked:
                stack.append(k)
    return frozenset(seen)


def _parts_off(mol, host: int, group: FrozenSet[int], unit: FrozenSet[int],
               locant: Optional[int] = None,
               allow_ylidene: bool = False) -> Optional[List[GroupPart]]:
    """The substituent parts of ``host`` (every neighbour in ``group`` outside ``unit``),
    each joined by a single bond (or, with ``allow_ylidene``, one carbon part joined by a
    double bond); ``None`` when a part is joined otherwise or two parts share atoms (a ring
    through the unit)."""
    parts: List[GroupPart] = []
    claimed: set = set()
    for nb in mol.GetAtomWithIdx(host).GetNeighbors():
        j = nb.GetIdx()
        if j not in group or j in unit:
            continue
        bt = mol.GetBondBetweenAtoms(host, j).GetBondType()
        order = 1
        if bt == _DOUBLE and allow_ylidene and nb.GetSymbol() == "C":
            order = 2
        elif bt != _SINGLE:
            return None
        atoms = _side(mol, j, group, frozenset(unit))
        if atoms & claimed:
            return None
        claimed |= atoms
        parts.append(GroupPart(attach=j, atoms=atoms, host=host, locant=locant, order=order))
    return parts


def _hydrazide_acyl(mol, part: GroupPart) -> bool:
    """True when ``part`` is an acyl group joined to a nitrogen of a hydrazine unit, so that the
    group has a hydrazido prefix, which is the PIN form: R-CO-NH-NH- and R-SO2-NH-NH-
     "When a group having priority for citation as a principal characteristic group is
    present, a hydrazide group of the type R-CO-NH-NH- or R-SO2-NH-NH- (or the analogous selenium
    or tellurium group) is named: (1)... 'acetohydrazido'... (2) as an acylhydrazinyl substituent
    group... Method (1) generates preferred IUPAC names", the Blue Book, under
    "Substituent groups derived from hydrazides"), R-C(=NH)-NH-NH-,:34548:
    '3-(methanimidohydrazido)propanoic acid (PIN)',:34559), and the groups whose second atom on
    the carbonyl is a hetero atom other than an amino nitrogen: '3-(hydrazinecarbohydrazido)
    propanoic acid (PIN)' (:38712), 'diazenecarbohydrazido (preferred prefix)' (:38912), an
    ester oxygen, '4-{[(hydrazinecarbonyl)oxy]formohydrazido}butanoic acid (PIN)' (:34129,
    's analogy for the alkoxycarbonyl group, ASSUMED). The acylhydrazinyl form is valid,
    never the PIN. The one acyl group whose hydrazinyl form is the PIN is carbamoyl, an amino
    nitrogen with no N-N or N=N bond on the carbonyl: 'semicarbazido... 2-carbamoylhydrazin-1-yl
    (preferred prefix)' "Semicarbazides",:38640; '3-(2-carbamoylhydrazin-1-yl)
    propanoic acid (PIN)',:38644)."""
    atom = mol.GetAtomWithIdx(part.attach)
    if atom.GetSymbol() not in ("C", "S", "Se", "Te"):
        return False
    marks = [nb.GetIdx() for nb in atom.GetNeighbors()
             if nb.GetSymbol() in ("O", "S", "Se", "Te", "N")
             and mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx()).GetBondType() == _DOUBLE]
    if not marks:
        return False
    rest = [nb for nb in atom.GetNeighbors() if nb.GetIdx() != part.host and nb.GetIdx() not in marks]
    return not any(nb.GetSymbol() == "N" and not any(
        x.GetSymbol() == "N" for x in nb.GetNeighbors() if x.GetIdx() != atom.GetIdx())
        for nb in rest)


def group_shape(mol, root: int, group: FrozenSet[int], parent: int,
                unstereo_iminyl: bool = False) -> Optional[GroupShape]:
    """The shape of the group ``group`` attached to ``parent`` through ``root``, or ``None``
    (not a group this module composes). ``parent`` is the one atom outside ``group`` that
    ``root`` is bonded to. ``unstereo_iminyl``: also build '[(R-ylidene)amino]' for an N=C
    bond with no defined geometry (the floor's ``_hetero_root_leaf`` builds that one)."""
    group = frozenset(group)
    if root not in group or parent in group:
        return None
    atom = mol.GetAtomWithIdx(root)
    if atom.GetIsotope() or atom.GetNumRadicalElectrons() or atom.IsInRing():
        return None
    bond = mol.GetBondBetweenAtoms(root, parent)
    if bond is None or bond.GetBondType() not in (_SINGLE, _DOUBLE):
        return None
    order = 2 if bond.GetBondType() == _DOUBLE else 1
    sym = atom.GetSymbol()
    inner = [nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetIdx() in group]

    if sym == "N" and atom.GetFormalCharge() == 0:
        if order == 2:
            # =N-X: the nitrogen is full (the double bond and one single bond)
            if len(inner) != 1:
                return None
            x = inner[0]
            if mol.GetBondBetweenAtoms(root, x).GetBondType() != _SINGLE:
                return None
            xa = mol.GetAtomWithIdx(x)
            if xa.GetSymbol() == "N" and _plain(xa):
                # =N-N<: the far nitrogen carries every substituent, single bonds only
                parts = _parts_off(mol, x, group, frozenset({root, x}), locant=2,
                                   allow_ylidene=True)
                if parts is None or len(parts) > 2:
                    return None
                if any(p.order == 2 for p in parts) and len(parts) != 1:
                    return None  # an azine: =N-N=C< takes both valences of the far nitrogen
                if sum(len(p.atoms) for p in parts) + 2 != len(group):
                    return None
                return GroupShape("hydrazinylidene", tuple(parts), frozenset({root, x}))
            # any other X (a ring nitrogen, a nitro group, a carbon, O, S...): '(X-yl)imino'
            atoms = _side(mol, x, group, frozenset({root}))
            if atoms != group - {root}:
                return None
            return GroupShape("imino", (GroupPart(x, atoms, root),), frozenset({root}))
        # order == 1: -N=C< with a defined geometry is iminyl, -N(=N-R) is diazenyl,
        # -N(-N<) is hydrazinyl
        if len(inner) == 1:
            x = inner[0]
            bx = mol.GetBondBetweenAtoms(root, x)
            if (bx.GetBondType() == _DOUBLE
                    and (unstereo_iminyl or bx.GetStereo() != Chem.BondStereo.STEREONONE)
                    and mol.GetAtomWithIdx(x).GetSymbol() == "C"):
                atoms = _side(mol, x, group, frozenset({root}))
                if atoms != group - {root}:
                    return None
                return GroupShape("iminyl", (GroupPart(x, atoms, root, order=2),),
                                  frozenset({root}))
        if len(inner) == 1 and len(group) == 2:
            o = mol.GetAtomWithIdx(inner[0])
            if (o.GetSymbol() == "O" and o.GetDegree() == 1 and _plain(o)
                    and mol.GetBondBetweenAtoms(root, inner[0]).GetBondType() == _DOUBLE):
                return GroupShape("nitroso", (), frozenset(group))
        n2s = [j for j in inner if mol.GetAtomWithIdx(j).GetSymbol() == "N"]
        if len(n2s) != 1 or not _plain(mol.GetAtomWithIdx(n2s[0])):
            return None
        n2 = n2s[0]
        b12 = mol.GetBondBetweenAtoms(root, n2).GetBondType()
        unit = frozenset({root, n2})
        if b12 == _DOUBLE:
            if len(inner) != 1:
                return None
            parts = _parts_off(mol, n2, group, unit)
            if parts is None or len(parts) > 1:
                return None
            if sum(len(p.atoms) for p in parts) + 2 != len(group):
                return None
            return GroupShape("diazenyl", tuple(parts), unit, stereo_bond=(root, n2))
        if b12 != _SINGLE:
            return None
        p1 = _parts_off(mol, root, group, unit, locant=1)
        p2 = _parts_off(mol, n2, group, unit, locant=2, allow_ylidene=True)
        if p1 is None or p2 is None or len(p1) > 1 or len(p2) > 2:
            return None
        if any(p.order == 2 for p in p2) and len(p2) != 1:
            return None  # =C< takes both free valences of the far nitrogen
        parts = p1 + p2
        if sum(len(p.atoms) for p in parts) + 2 != len(group):
            return None
        return GroupShape("hydrazinyl", tuple(parts), unit,
                          acylhydrazide=any(_hydrazide_acyl(mol, p) for p in p2))

    if sym == "N" and atom.GetFormalCharge() == 1 and order == 1 and atom.GetTotalValence() == 4:
        # R2N+(-O-)-: 'dimethyl(oxido)azaniumyl', the Blue Book)
        oxides = [j for j in inner if mol.GetAtomWithIdx(j).GetSymbol() == "O"
                  and mol.GetAtomWithIdx(j).GetFormalCharge() == -1
                  and mol.GetAtomWithIdx(j).GetDegree() == 1
                  and not mol.GetAtomWithIdx(j).GetIsotope()]
        if len(oxides) != 1 or any(b.GetBondType() != _SINGLE for b in atom.GetBonds()):
            return None
        unit = frozenset({root, oxides[0]})
        parts = _parts_off(mol, root, group, unit)
        if parts is None or len(parts) > 2:
            return None
        if sum(len(p.atoms) for p in parts) + 2 != len(group):
            return None
        return GroupShape("oxidoazaniumyl", tuple(parts), unit, internal=unit)

    if sym == "N" and atom.GetFormalCharge() == 1 and order == 2:
        # =[NH2+], =[NHR+], =[NR2+]: valence four, one double bond; =[N+](R)-O(-) is the
        # nitrone group, '(R)(oxido)azaniumylidene' Dipolar substituent groups,
        # the Blue Book '2-(oxidoazaniumylidyne)ethyl')
        if atom.GetTotalValence() != 4 or len(inner) > 2:
            return None
        oxides = [j for j in inner if mol.GetAtomWithIdx(j).GetSymbol() == "O"
                  and mol.GetAtomWithIdx(j).GetFormalCharge() == -1
                  and mol.GetAtomWithIdx(j).GetDegree() == 1
                  and not mol.GetAtomWithIdx(j).GetIsotope()]
        if oxides:
            if len(oxides) != 1:
                return None
            unit = frozenset({root, oxides[0]})
            parts = _parts_off(mol, root, group, unit)
            if parts is None or len(parts) > 1:
                return None
            if sum(len(p.atoms) for p in parts) + 2 != len(group):
                return None
            return GroupShape("oxidoazaniumylidene", tuple(parts), unit, internal=unit)
        parts = _parts_off(mol, root, group, frozenset({root}))
        if parts is None or sum(len(p.atoms) for p in parts) + 1 != len(group):
            return None
        return GroupShape("azaniumylidene", tuple(parts), frozenset({root}),
                          charged=frozenset({root}))

    if sym == "O" and atom.GetFormalCharge() == 0 and order == 1 and len(inner) == 1:
        o2 = inner[0]
        a2 = mol.GetAtomWithIdx(o2)
        if (a2.GetSymbol() == "N" and _plain(a2) and a2.GetFormalCharge() == 0
                and mol.GetBondBetweenAtoms(root, o2).GetBondType() == _SINGLE):
            # -O-N<: '(R-amino)oxy' (an oxime ether is '[(R-ylidene)amino]oxy'); the nitrogen
            # prefix is named by the host (``amino`` with its substituents)
            atoms = _side(mol, o2, group, frozenset({root}))
            if atoms != group - {root}:
                return None
            return GroupShape("aminooxy", (GroupPart(o2, atoms, root),), frozenset({root}))
        if (a2.GetSymbol() != "O" or not _plain(a2)
                or mol.GetBondBetweenAtoms(root, o2).GetBondType() != _SINGLE):
            return None
        unit = frozenset({root, o2})
        parts = _parts_off(mol, o2, group, unit)
        if parts is None or len(parts) > 1:
            return None
        if sum(len(p.atoms) for p in parts) + 2 != len(group):
            return None
        return GroupShape("peroxy" if parts else "hydroperoxy", tuple(parts), unit)

    if sym == "O" and atom.GetFormalCharge() == 1 and order == 2 and atom.GetTotalValence() == 3:
        parts = _parts_off(mol, root, group, frozenset({root}))
        if parts is None or len(parts) > 1 or sum(len(p.atoms) for p in parts) + 1 != len(group):
            return None
        return GroupShape("oxidaniumylidene", tuple(parts), frozenset({root}),
                          charged=frozenset({root}))
    return None


def compose_group(shape: GroupShape, names: Sequence[str]) -> Optional[str]:
    """The prefix of ``shape`` given the host's name of each of its parts (in the order of
    ``shape.parts``: a ``-yl`` prefix for each part), or ``None``. A prefix with parts is
    recorded as substituted and enclosed (``prefix_derivation.built``), so it is cited with
    enclosing marks wherever it stands, whatever its first letters are ('tert-butyl(oxido)
    azaniumylidene' is compound although a name beginning 'tert-butyl' reads as simple)."""
    if len(names) != len(shape.parts) or any(not n for n in names):
        return None
    from .prefix_derivation import built
    if shape.kind == "hydroperoxy":
        return "hydroperoxy"
    if shape.kind == "nitroso":
        return "nitroso"
    if shape.kind == "aminooxy":
        from .substituent_enumerator import cite_organyl_in_composed_prefix
        cited = cite_organyl_in_composed_prefix(names[0])
        return built(f"{cited}oxy", substituted=True, enclosed=True) if cited else None
    if shape.kind == "hydrazinyl":
        token = _hydrazinyl(shape, names)
        if token and shape.acylhydrazide:
            # (:34020): the hydrazido prefix is the PIN form of R-CO-NH-NH-, this
            # acylhydrazinyl one is the book's second, non-preferred method
            from ..metrics.provenance import record_non_pin_fragment
            record_non_pin_fragment(token)
        return built(token, substituted=bool(names), enclosed=bool(names)) if token else None
    from .composer import _assemble_decorated_amino_prefix
    head = {"imino": "imino", "hydrazinylidene": "hydrazinylidene", "diazenyl": "diazenyl",
            "peroxy": "peroxy", "azaniumylidene": "azaniumylidene",
            "oxidaniumylidene": "oxidaniumylidene", "iminyl": "amino",
            "oxidoazaniumyl": "azaniumyl",
            "oxidoazaniumylidene": "azaniumylidene"}[shape.kind]
    if shape.kind in ("oxidoazaniumyl", "oxidoazaniumylidene"):
        names = list(names) + ["oxido"]
    if not names:
        return head
    token = _assemble_decorated_amino_prefix([(n, False) for n in names], enclose=False,
                                             head=head)
    return built(token, substituted=True, enclosed=True)


def _hydrazinyl(shape: GroupShape, names: Sequence[str]) -> Optional[str]:
    """'2-phenylhydrazin-1-yl', '1,2-dimethylhydrazin-1-yl', 'ethylidenehydrazinyl',
    'hydrazinyl'. "Citation of locants" (the Blue Book): "if any locants are
    essential for defining the structure of the parent structure or of a unit of structure as
    defined by its appropriate enclosing marks, then all locants must be cited", so a unit
    that cites one locant cites the free valence's too: '4-(1-acetyl-2-ethylhydrazin-1-yl)
    benzoic acid (PIN)' (:34042), '3-(2-carbamoylhydrazin-1-yl)propanoic acid (PIN)' (:38644),
    '2-nitrohydrazin-1-yl (preselected prefix)' (:36437). (b) "The locant '1' is
    omitted... in monosubstituted homogeneous chains consisting of only two identical atoms"
    (:2903) is the unsubstituted unit alone, 'hydrazinyl (not hydrazin-1-yl)' (:2911): the
    free valence is the one substituent. "Omission of locants" (:2871) leaves the
    locants out in two cases here. "Locants are omitted when no isomer can be
    generated by moving suffixes and/or prefixes (if any) from their position to another or
    by interchanging them between two different positions" (:2953): an ylidene can stand on
    the far nitrogen only, 'ethylidenehydrazinyl (preferred prefix) (not 2-ethylidenehydrazin-
    1-yl)' (:2959). "All locants are omitted in compounds or substituent groups in
    which all substitutable positions are completely substituted... in the same way" (:3007):
    three identical substituents, 'trimethylhydrazinyl'."""
    if not names:
        return "hydrazinyl"
    ylidene = any(part.order == 2 for part in shape.parts)
    if ylidene and len(shape.parts) > 1:
        # An ylidene on the far nitrogen with a substituent on the first leaves no isomer, so the
        # book spelling cites no locant, 'ethylidene(methyl)hydrazinyl'. OPSIN 2.9.0 reads that
        # as ethylidene joined to the methyl carbon, a name that does not read back to the
        # group, so no prefix is built and the host keeps its other spelling.
        return None
    if ylidene or (len(shape.parts) == 3 and len(set(names)) == 1):
        from .composer import _assemble_decorated_amino_prefix
        return _assemble_decorated_amino_prefix([(n, False) for n in names], enclose=False,
                                                head="hydrazinyl")
    return hydrazine_substituents(
        [(part.locant, name) for part, name in zip(shape.parts, names)]) + "hydrazin-1-yl"


def hydrazine_substituents(cited: Sequence[Tuple[Optional[int], str]]) -> str:
    """The substituent prefixes of a hydrazine unit, each with its locant,
    the Blue Book: deny by default), in alphanumerical order and with identical ones
    multiplied: '2,2-dimethyl', '1,2-dimethyl', '2-phenyl'. ``cited`` holds one
    ``(locant, '-yl' prefix name)`` per substituent; the head ('hydrazinyl',
    'hydrazine-1-sulfonyl') is the caller's."""
    from .naming_utils import alpha_sort_key, format_substituent_prefix
    grouped: dict = {}
    for locant, name in cited:
        grouped.setdefault(name, []).append(locant)
    pieces = []
    for name, locs in sorted(grouped.items(), key=lambda kv: alpha_sort_key(kv[0])):
        locs = sorted(locs)
        pieces.append(format_substituent_prefix(name, locs, len(locs)))
    return "-".join(pieces)
