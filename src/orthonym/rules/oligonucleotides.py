"""Oligonucleotide names (Blue Book and.

 "DERIVATIVES OF NUCLEOTIDES" (the Blue Book, the paragraph at:55065): "Oligonucleotides are named
using the prefix names derived from the traditional names for the nucleotides." The example is
`2'-deoxyguanylyl-(3'->5')-2'-deoxyuridylyl-(3'->5')-2'-deoxyguanosine`: one prefix per unit from the 5' end, each
joined to the next by the locants of the phosphodiester, and the nucleoside name of the 3'-terminal unit.
 (:55055): the prefix names come from the nucleotides by replacing 'ic acid' with 'yl' ('adenylyl',
'cytidylyl'); "the substituent prefix name from inosinic acid is an exception; it is named inosinylyl".
 (:55071): "When a phosphorothioic acid is used (HS-P in place of HO-P), the prefix P-thio is added at the front
of the name of the nucleotide": `2'-deoxy-P-thioguanylyl-(3'->5')-2'-deoxy-P-thiouridylyl-(3'->5')-2'-deoxyguanosine`,
'deoxy' and 'thio' in alphanumerical order ('deoxy' is detachable and alphabetized,,:53025;.
 INTRODUCTION (:50939): no PIN is identified for these compounds, so the name is a systematic one, never a PIN.

The producer names what these names state and refuses all else (returns None, the caller keeps its other name):
a linear chain of units joined 3' -> 5' by neutral phosphodiesters (P-thio allowed, without a stated P
configuration), every unit an unsubstituted standard base (adenine, guanine, hypoxanthine, xanthine, cytosine, uracil,
and thymine in the 2'-deoxy series) on a beta-D-ribofuranose or 2'-deoxy-beta-D-ribofuranose with every sugar centre
defined, a free 5'-OH on the first unit and a free 3'-OH on the last, and every atom of the molecule in a unit or a
link. The units are perceived with the producer of the nucleoside names (``rules/nucleosides.py``): the same core,
base and configuration tests, one per unit.

The tests overlap on purpose: the atom-coverage test, the free 5'-OH / 3'-OH end test, the 2'-OH test, the 3'/5'
pairing and same-unit tests of ``_links`` and the 2'-O / deoxy consistency test are backstops. On every input a
refusal test names, another of these tests refuses it too (a review mutated each alone, 2026-11-01), so a name is
built only when all of them pass.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from rdkit import Chem

from .nucleosides import (
    _base_numbering,
    _classify_base,
    _iter_cores,
    _sugar_is_plain,
    _sugar_labels_ok,
    nucleoside_names_off,
)

#: nucleoside stem -> the prefix name of its nucleotide,:55055: 'ic acid' -> 'yl'; inosinylyl is the
#: stated exception). The stem of the 2'-deoxy series is the same, '2'-deoxy' is a prefix,:54963).
_PREFIX_WORD = {
    "adenosine": "adenylyl", "guanosine": "guanylyl", "inosine": "inosinylyl", "xanthosine": "xanthylyl",
    "cytidine": "cytidylyl", "uridine": "uridylyl", "thymidine": "thymidylyl",
}

#: the locants of the phosphodiester between a unit and the next, as the book writes them (:55065)
_LINK = "-(3'→5')-"

_ELEMENTS = frozenset({"C", "N", "O", "P", "S"})


class _Unit:
    """A nucleoside unit: its core, its stem and sugar (``thymidine`` is the 2'-deoxy series' own stem), the atoms
    it owns and its three sugar oxygens (None where the sugar has none)."""
    __slots__ = ("core", "stem", "sugar", "atoms", "o2", "o3", "o5")


def _is_terminal(atom: Chem.Atom, elements: Tuple[str, ...]) -> bool:
    """A neutral OH or SH: one heavy neighbour (the caller's), one hydrogen."""
    return (atom.GetSymbol() in elements and atom.GetDegree() == 1 and atom.GetTotalNumHs() == 1
            and not atom.GetFormalCharge())


def _exocyclic_oxygen(mol: Chem.Mol, core, prime: str) -> Optional[int]:
    """The oxygen on the sugar carbon ``prime`` outside the ring, None when it has none."""
    for nb in mol.GetAtomWithIdx(core.idx_for(prime)).GetNeighbors():
        if nb.GetSymbol() == "O" and nb.GetIdx() not in core.ring_atoms:
            return nb.GetIdx()
    return None


def _unit_of(mol: Chem.Mol, core) -> Optional[_Unit]:
    """The unit of one furanose core, or None when its name is not one of the tables of this module: a sugar
    with more than hydroxy groups, a base other than the standard ones or one that carries a group, a 2'-OH that
    is not free. The configuration of the sugar is read last (:func:`name_oligonucleotide`)."""
    if not _sugar_is_plain(mol, core):
        return None
    base = _classify_base(mol, core)
    if base is None:
        return None
    stem, sugar, subs, strip = base
    if subs:
        # the one base group a prefix name has: the 5-methyl of thymine, in the 2'-deoxy series ('thymidylyl')
        if subs != [("5", "methyl")] or stem != "uridine" or sugar != "deoxy":
            return None
        stem = "thymidine"
    unit = _Unit()
    unit.core, unit.stem, unit.sugar = core, stem, sugar
    unit.o2 = _exocyclic_oxygen(mol, core, "2'")
    unit.o3, unit.o5 = _exocyclic_oxygen(mol, core, "3'"), _exocyclic_oxygen(mol, core, "5'")
    if unit.o3 is None or unit.o5 is None or (unit.o2 is None) != (sugar == "deoxy"):
        return None
    if unit.o2 is not None and not _is_terminal(mol.GetAtomWithIdx(unit.o2), ("O",)):
        return None                               # a 2'-ester or 2'-phosphate: another compound
    num = _base_numbering(mol, core.base_n)
    ring = set(num)
    atoms = set(core.ring_atoms) | {core.idx_for("5'"), unit.o3, unit.o5} | ring | set(strip)
    if unit.o2 is not None:
        atoms.add(unit.o2)
    for a in ring:                                # the exocyclic oxygen and nitrogen of the base (no group on them)
        atoms |= {nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors()
                  if nb.GetIdx() not in ring and nb.GetIdx() != core.anomeric}
    unit.atoms = atoms
    return unit


def _links(mol: Chem.Mol, units: List[_Unit]) -> Optional[Tuple[Dict[int, Tuple[int, bool]], set]]:
    """{unit -> (next unit, P-thio)} of the phosphodiesters, and the atoms they own besides the sugar oxygens
    of the units, or None when a phosphorus is not a neutral diester of the 3'-O of one unit and the 5'-O of
    another."""
    position = {}                                  # sugar oxygen -> (unit, "3'" or "5'")
    for i, unit in enumerate(units):
        position[unit.o3], position[unit.o5] = (i, "3'"), (i, "5'")
    follows: Dict[int, Tuple[int, bool]] = {}
    owned: set = set()
    for p in (a for a in mol.GetAtoms() if a.GetSymbol() == "P"):
        if p.GetDegree() != 4 or p.GetFormalCharge():
            return None
        double, terminal, bridging = [], [], []
        for bond in p.GetBonds():
            nb = bond.GetOtherAtom(p)
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                double.append(nb)
            elif bond.GetBondType() != Chem.BondType.SINGLE:
                return None
            elif nb.GetDegree() == 1:
                terminal.append(nb)
            else:
                bridging.append(nb)
        if len(double) != 1 or len(terminal) != 1 or len(bridging) != 2 or double[0].GetDegree() != 1 \
                or double[0].GetSymbol() not in ("O", "S") or double[0].GetFormalCharge() \
                or double[0].GetTotalNumHs():
            return None
        if not _is_terminal(terminal[0], ("O", "S")):
            return None
        thio = [double[0].GetSymbol(), terminal[0].GetSymbol()].count("S")
        if thio > 1:
            return None                           # a dithio acid is not the P-thio of the book
        if thio and p.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            return None                           # the name states no P configuration has none)
        ends = sorted((position.get(o.GetIdx(), (-1, "")) for o in bridging), key=lambda t: t[1])
        if ends[0][1] != "3'" or ends[1][1] != "5'" or ends[0][0] == ends[1][0] \
                or any(o.GetSymbol() != "O" or o.GetFormalCharge() for o in bridging):
            return None                           # a 5'-5' or 3'-3' link, or a cyclic 3',5'-phosphate
        if ends[0][0] in follows:
            return None
        follows[ends[0][0]] = (ends[1][0], bool(thio))
        owned |= {p.GetIdx(), double[0].GetIdx(), terminal[0].GetIdx()}      # the bridging oxygens are the units'
    return follows, owned


def _chain(units: List[_Unit], follows: Dict[int, Tuple[int, bool]]) -> Optional[List[int]]:
    """The units from the 5' end, or None unless the links make one linear chain through all of them."""
    followed = {nxt for nxt, _ in follows.values()}
    starts = [i for i in range(len(units)) if i not in followed]
    if len(starts) != 1 or len(follows) != len(units) - 1 or len(followed) != len(follows):
        return None
    order = [starts[0]]
    while order[-1] in follows:
        order.append(follows[order[-1]][0])
        if len(order) > len(units):
            return None
    return order if len(order) == len(units) else None


def _prefixes(unit: _Unit, thio: bool) -> str:
    """'2'-deoxy' and 'P-thio' in alphanumerical order in front of the stem,:55071;."""
    parts = []
    if unit.sugar == "deoxy" and unit.stem != "thymidine":
        parts.append(("deoxy", "2'-deoxy"))
    if thio:
        parts.append(("thio", "P-thio"))
    return "-".join(text for _, text in sorted(parts))


def _names_the_input(mol: Chem.Mol) -> bool:
    """Whether ``mol`` is the molecule the running ``name`` was asked about, and carries no isotope label.

    The name states no isotopic descriptor, and a nested naming of the isotope-stripped skeleton
    (``rules/isotopes.py`` names it before it places the descriptors) would take this name for it: a prefix name
    takes no descriptor, so the labelled molecule, named before by its systematic skeleton, would have no name.
    The name states the whole molecule either, not a capped fragment or a part of an adduct that the outer name is
    built from (``assembly/universal_halane.py::names_the_input``)."""
    from ..assembly.fragment_naming import outermost_name_input
    from ..assembly.universal_halane import names_the_input
    from .isotopes import has_isotopes
    outer = outermost_name_input()
    if outer is None:
        return True                               # a direct call, outside any naming
    given = Chem.MolFromSmiles(outer)
    return given is not None and not has_isotopes(given) and names_the_input(mol)


def name_oligonucleotide(mol: Chem.Mol) -> Optional[str]:
    """The name of a linear 3'->5' oligonucleotide, or None (the caller keeps another name)."""
    if mol is None or nucleoside_names_off() or not any(a.GetSymbol() == "P" for a in mol.GetAtoms()):
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    if any(a.GetSymbol() not in _ELEMENTS or a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetIsotope()
           for a in mol.GetAtoms()):
        return None
    units: List[_Unit] = []
    for core in _iter_cores(mol):
        unit = _unit_of(mol, core)
        if unit is None:
            return None
        units.append(unit)
    if len(units) < 2:
        return None
    linked = _links(mol, units)
    if linked is None:
        return None
    follows, owned = linked
    order = _chain(units, follows)
    if order is None:
        return None
    covered = set(owned)
    for unit in units:
        if covered & unit.atoms:
            return None
        covered |= unit.atoms
    if len(covered) != mol.GetNumAtoms():
        return None                               # an atom that is in no unit and no link is not named
    first, last = units[order[0]], units[order[-1]]
    if not _is_terminal(mol.GetAtomWithIdx(first.o5), ("O",)) or not _is_terminal(mol.GetAtomWithIdx(last.o3), ("O",)):
        return None                               # a 5'- or 3'-terminal group other than OH (a terminal phosphate)
    # the configuration last, so that no other phosphorus compound is labelled for it. The labels are those of the
    # free hydroxy groups (a substituted 2'-O can turn the label of C3'), which every unit has here.
    from ..perception.stereo import assign_stereochemistry
    assign_stereochemistry(mol)
    if not all(_sugar_labels_ok(mol, u.core, u.sugar) for u in units):
        return None                               # an other-than-beta-D sugar or an undefined centre
    if not _names_the_input(mol):
        return None
    words = [_prefixes(units[i], follows[i][1]) + _PREFIX_WORD[units[i].stem] for i in order[:-1]]
    return _LINK.join(words + [_prefixes(last, False) + last.stem])
