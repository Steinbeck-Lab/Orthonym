"""UN lane H -- a hypervalent halogen atom (Cl, Br, I) in the universal floor, as a one-atom parent
hydride or substituent root with the lambda convention.

The floor (``universal_substituent``) builds a name from a spine of skeletal atoms (carbon and the
'a'-replacement heteroatoms) and cites everything else as a prefix. A halogen with a bonding number
other than its standard one is neither: it is not an 'a' atom (no ``REPLACEMENT_TERMS`` morpheme), and a
spine that threaded it would count it as a carbon of the chain and drop it from the name (a phantom
``methane``, ``universal_substituent._chain_spine_from_path``). That guard stays; this module gives the
atom a way to be named that does not go through a spine at all: it is its own one-atom parent, or the
root of one substituent, and every neighbour is a prefix named by the floor's own recursion.

* "Nomenclature based on halogen parent hydrides" (the Blue Book): "For cyclic and
  acyclic parent hydrides the lambda convention is used to indicate nonstandard bonding numbers."
  'CH3-ICl2 dichloro(methyl)-lambda3-iodane (PIN)', 'C6H5-I(OH)2 phenyl-lambda3-iodanediol (PIN)',
  '(CH3-CO-O)2I- bis(acetyloxy)-lambda3-iodanyl (preferred prefix)'. The token comes from
  ``rules.lambda_convention`` (``LAMBDA``, ``nonstandard_bonding_number``), the one home of the
  lambda convention.
* "Seniority order for classes", Table 4.1 (the Blue Book): class 39 "lambda7, lambda5 and
  lambda3 halogen compounds in the order F > Cl > Br > I" (:18215) is senior to class 40 "Carbon
  compounds: rings, chains". That is the only seniority applied here: ``halane_parent`` makes a
  hypervalent halogen the parent of the whole molecule whatever else it holds, and a ring or chain a
  prefix of it. The classes above 39 are NOT checked: the principal characteristic groups (classes
  1-20: an acid, an ester, an alcohol, an amine... would take the suffix and the parent) and the
  heteroatom hydrides N > P > As > Sb > Bi > Si > Ge > Sn > Pb > B > Al > Ga > In > Tl > O > S > Se > Te
  (classes 21-38,,:18917). The floor writes no suffix and ranks no class anywhere, so its
  names are systematic, never a PIN; applying to a hypervalent halogen is a lead, not built. A
  second hypervalent halogen of the molecule is the root of a substituent of the first.
* "The locant '1' is omitted: (a) in substituted mononuclear parent hydrides"
  (the Blue Book): the prefixes of the one-atom parent carry no locant, the second and further
  enclosed, (:7272: "The second and further substituents are each enclosed with
  parentheses even for simple substituents"), by the shared assembler
  ``composer._assemble_decorated_amino_prefix`` that spells 'dimethylphosphanyl' and 'hydroxy(methyl)'
  for the other mononuclear parents.
* "Substituent prefixes derived from mononuclear parent hydrides" (:15856): "Other mononuclear
  parent hydrides, i.e., those derived from O, F, Cl, Br, I, S,... are named by method (2) in ",
  the '-yl' of the hydride name with its final 'e' elided (iodane -> iodanyl, iodanylidene), and "Substituent
  groups derived from mononuclear parent hydrides modified by the lambda-convention are discussed... in
   for elements of Group 17".
* A chlorine or bromine of bonding number 3, 5 or 7 is drawn charge-separated by RDKit
  (``Cl(=O)(=O)=O`` is ``[Cl+3]([O-])([O-])[O-]``), a 1,2-dipolar drawing "1,2-Dipolar
  compounds",:42497) of the same oxo groups; each of the ``q`` terminal ``[O-]`` of a ``+q`` halogen is
  read as the ``oxo`` of the uncharged drawing, whose name is the lambda-convention one of, and
  the halogen and those oxygens are the charges the name accounts for.

Off by default (``_Ctx.halane_forms``): ``universal_substituent.name_universal_substitutive`` asks for it
only after every other floor spelling has declined, so the floor's own names are not changed; it is a
last rung of the floor, which comes before the fallbacks after it (the neutral-form naming of a
charge-separated chalcogenide, and the producers the namer tries when the floor gives nothing), so a
molecule one of those named can now get the halane name instead. A shape this module does not name (a
halogen in a ring, a stereogenic halogen, a halogen with a charge it cannot account for, a halogen with
a bond other than single, double or triple -- a dative bond -- or an attachment that is not a single or
double bond) returns ``None`` and the floor voids as it always did.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

from rdkit import Chem

from ..rules.lambda_convention import LAMBDA, nonstandard_bonding_number

#: the parent hydride of each halogen that carries the lambda convention
HALANE_STEM: Dict[str, str] = {"Cl": "chlorane", "Br": "bromane", "I": "iodane"}

#: class 39 "in the order F > Cl > Br > I": the senior halogen is the parent
_CLASS_39_ORDER = {"Cl": 0, "Br": 1, "I": 2}

#: the bond types that are valence bonds of a lambda-convention name; a dative (or any other) bond is not
_PLAIN_BONDS = frozenset({Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE})

#: the ending of the group a mononuclear parent hydride gives by the order of its attachment bond,
#: / method (2); the bond order of the free valence is the bond to the host
_YL_ENDING = {1: "yl", 2: "ylidene", 3: "ylidyne"}


@dataclass(frozen=True)
class HalaneUnit:
    """One halogen atom the floor can name as a mononuclear parent hydride."""

    atom: int
    stem: str                        # 'iodane'
    lam: Optional[int]               # the bonding number cited after the lambda, None when standard
    ending: str = "e"                # 'e' hydride, 'ium' cation, 'ide' anion method (2))
    ionic: bool = False              # a genuine +/-1 centre: the charge is accounted for by the name
    oxides: FrozenSet[int] = frozenset()  # terminal [O-] of a charge-separated drawing, cited 'oxo'


def has_halane_candidate(mol) -> bool:
    """True when *mol* has a Cl, Br or I that is not a plain monovalent halide: bonded to two or more
    heavy atoms, or charged (the cheap test ahead of ``halane_unit``)."""
    return any(a.GetSymbol() in HALANE_STEM and a.GetDegree() > 0
               and (a.GetDegree() > 1 or a.GetFormalCharge()) for a in mol.GetAtoms())


def names_the_input(mol) -> bool:
    """True unless a ``name`` call is running and *mol* is not the molecule it was asked to name:
    a nested frame names structures it derives from the input (the neutral skeleton of a cation, a
    capped fragment), and the outer name is built from theirs, so their provenance is the outer name's.
    The molecule is the same when its molecular formula (atoms, hydrogens and charge) is: a drawing the
    namer normalised keeps it, a skeleton with another hydron or a part of the molecule does not."""
    from rdkit.Chem import rdMolDescriptors

    from .fragment_naming import outermost_name_input
    outer = outermost_name_input()
    if outer is None:
        return True
    try:
        given = Chem.MolFromSmiles(outer)
        return given is None or (rdMolDescriptors.CalcMolFormula(given)
                                 == rdMolDescriptors.CalcMolFormula(mol))
    except Exception:  # noqa: BLE001 - an unreadable input is no reason to hold the rung back
        return True


def _terminal_oxide(atom) -> bool:
    """A terminal ``[O-]``: charge -1, no hydrogen, one bond, which is single."""
    return (atom.GetSymbol() == "O" and atom.GetFormalCharge() == -1 and atom.GetDegree() == 1
            and atom.GetTotalNumHs() == 0 and atom.GetIsotope() == 0
            and atom.GetBonds()[0].GetBondType() == Chem.BondType.SINGLE)


def halane_unit(mol, idx: int) -> Optional[HalaneUnit]:
    """The:class:`HalaneUnit` of atom *idx*, or ``None`` when the floor cannot name it as one.

    * a neutral halogen of non-standard bonding number (3, 5, 7: ``nonstandard_bonding_number``, the
      lambda home): ``-lambda3-iodane``;
    * a ``+q`` halogen with exactly ``q`` terminal ``[O-]`` neighbours, the charge-separated drawing of
      a halogen oxide: bonding number ``valence + q`` of the uncharged drawing, the oxides cited ``oxo``;
    * a ``+1`` halogen of bonding number 2, ``-ium`` "General rule for systematically naming
      cationic centers in parent hydrides", the Blue Book: "replacing the final letter 'e' of the
      parent hydride name, if any, by the suffix 'ium'"; 'diphenyliodanium (PIN)',:41362), and a ``-1`` one, the anion of "ANIONS FORMED BY ADDITION OF HYDRIDE
      IONS" (:41089) in its method (2) spelling, ``-lambda3-iodanide`` ("(2) by adding the suffix 'ide' to
      a parent hydride in which the bonding number at the anionic site is higher than the standard bonding
      number and is expressed by the lambda-convention",:41094; 'diphenyliodanuide (PIN)
      diphenyl-lambda3-iodanide',:41110). OPSIN 2.9.0 has no grammar for the substituted method (1)
      '-uide' token (``validation.uide_twin``); the method (2) spelling reads, so the floor's own
      round trip verifies it. The number is the one the twin of the default tier cites
      (``rules.ions._UIDE_STD_VALENCE`` + 2).

    ``None`` for a halogen in a ring, with an isotope, a radical or a chiral tag (the descriptor of a
    stereogenic hypervalent atom is not cited), with a bond that is not single, double or triple (a dative
    bond counts in the bonding number of ``GetTotalValence`` but is no valence bond of the lambda name:
    ``N->I(C)C`` would be named with one hydrogen fewer than the input), or with a charge none of these
    accounts for. A stereo double
    bond at the halogen is not declined: no descriptor is cited for it, and the name ships only if its
    read-back reaches the input's full InChIKey, which decides whether the key sees that bond."""
    atom = mol.GetAtomWithIdx(idx)
    stem = HALANE_STEM.get(atom.GetSymbol())
    if (stem is None or atom.IsInRing() or atom.GetIsotope() or atom.GetNumRadicalElectrons()
            or atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
            or any(b.GetBondType() not in _PLAIN_BONDS for b in atom.GetBonds())):
        return None
    charge = atom.GetFormalCharge()
    heavy = [nb for nb in atom.GetNeighbors() if nb.GetAtomicNum() > 1]
    if charge == 0:
        lam = nonstandard_bonding_number(mol, idx)
        if lam is None or lam % 2 == 0 or not heavy:
            return None
        return HalaneUnit(idx, stem, lam)
    if charge > 0:
        oxides = frozenset(nb.GetIdx() for nb in heavy if _terminal_oxide(nb))
        if len(oxides) == charge:
            lam = atom.GetTotalValence() + charge
            if lam >= 3 and lam % 2 == 1:
                return HalaneUnit(idx, stem, lam, oxides=oxides)
    if abs(charge) == 1 and atom.GetTotalValence() == 2 and len(heavy) >= 1:
        if charge == 1:
            return HalaneUnit(idx, stem, None, "ium", ionic=True)
        from ..rules.ions import _UIDE_STD_VALENCE
        return HalaneUnit(idx, stem, _UIDE_STD_VALENCE[atom.GetSymbol()] + 2, "ide", ionic=True)
    return None


def halane_parent(ctx, component: FrozenSet[int]) -> Optional[HalaneUnit]:
    """The halogen that is the parent of the whole molecule: an ionic centre first (its charge suffix names
    it on a parent only), then the senior halogen ``F > Cl > Br > I`` class 39), then the
    canonical rank (numbering-invariant, never the atom index). A hypervalent halogen is the parent
    whatever else the molecule holds: classes 1-38, which rank above class 39, are not applied here
    (see the module docstring)."""
    units = [u for u in (halane_unit(ctx.mol, a) for a in component
                         if ctx.mol.GetAtomWithIdx(a).GetSymbol() in HALANE_STEM) if u is not None]
    if not units:
        return None
    return min(units, key=lambda u: (not u.ionic, _CLASS_39_ORDER[ctx.mol.GetAtomWithIdx(u.atom)
                                                                  .GetSymbol()],
                                     ctx.canon_rank[u.atom]))


def _pieces(mol, x: int, component: FrozenSet[int]) -> Optional[List[Tuple[int, FrozenSet[int]]]]:
    """``[(neighbour, its atoms)]``: the pieces of *component* that stay once atom *x* is taken away.
    A halogen in no ring is a cut vertex, so the pieces are disjoint and cover ``component - {x}``;
    ``None`` when they do not (never expected: the guard is for a ring the caller did not see)."""
    out: List[Tuple[int, FrozenSet[int]]] = []
    claimed = {x}
    for nb in mol.GetAtomWithIdx(x).GetNeighbors():
        j = nb.GetIdx()
        if nb.GetAtomicNum() <= 1 or j not in component:
            continue
        if j in claimed:
            return None
        piece, stack = set(), [j]
        while stack:
            cur = stack.pop()
            if cur in piece:
                continue
            piece.add(cur)
            stack.extend(n.GetIdx() for n in mol.GetAtomWithIdx(cur).GetNeighbors()
                         if n.GetAtomicNum() > 1 and n.GetIdx() in component
                         and n.GetIdx() not in piece and n.GetIdx() != x)
        if piece & claimed:
            return None
        claimed |= piece
        out.append((j, frozenset(piece)))
    return out if claimed == set(component) else None


def name_halane_component(ctx, component: FrozenSet[int], attach_hint: Optional[int], is_top: bool,
                          emit_branch_stereo: bool = False, relative_ring_override=None):
    """A ``_ComponentResult`` that names *component* on its halogen, or ``None``.

    ``is_top``: the halogen is the parent hydride of the whole molecule, ``<prefixes>-lambda3-iodane``
    (``halane_parent``). Otherwise the component is a branch attached through its halogen
    (``attach_hint``) and the result is the substituent prefix ``<prefixes>-lambda3-iodanyl``, bound to
    the order of its attachment bond. The prefixes are the neighbours, each named by the floor's own
    recursion on its piece (its stereo block and charges carried as in the branch loop of
    ``_name_component``), without locants and in the order."""
    from . import universal_substituent as us
    from .composer import _assemble_decorated_amino_prefix
    mol = ctx.mol
    unit = (halane_parent(ctx, component) if is_top
            else halane_unit(mol, attach_hint) if attach_hint is not None else None)
    if unit is None or unit.atom not in component:
        return None
    x = unit.atom
    order = None
    if not is_top:
        outside = [nb.GetIdx() for nb in mol.GetAtomWithIdx(x).GetNeighbors()
                   if nb.GetAtomicNum() > 1 and nb.GetIdx() not in component]
        if len(outside) != 1 or unit.ionic:
            return None
        order = round(mol.GetBondBetweenAtoms(x, outside[0]).GetBondTypeAsDouble())
        if order not in _YL_ENDING:
            return None
    pieces = _pieces(mol, x, component)
    if pieces is None:
        return None

    if unit.ionic and not any(site["atom_idx"] == x for site in ctx.cation_sites + ctx.anion_sites):
        return None  # a charge the ion perception does not own (a semipolar centre): not named here
    core = unit.stem[:-1] + (_YL_ENDING[order] if order is not None else unit.ending)
    if unit.lam:
        core = f"{LAMBDA}{unit.lam}-{core}"
    bindings: List[Tuple[str, FrozenSet[int]]] = [(core, frozenset({x}))]
    names: List[Tuple[str, bool]] = []
    charged: set = set()
    internal: set = set()
    if unit.oxides:
        # the charge-separated drawing is the name's own ``oxo`` groups on a lambda halogen: those
        # charges are consumed here, as ion-site charges when the ion perception lists them and as
        # internal (semipolar) charges when it does not -- either way the top-level guards see them
        sites = {site["atom_idx"] for site in ctx.cation_sites + ctx.anion_sites}
        owned = {x} | set(unit.oxides)
        charged |= owned & sites
        internal |= owned - sites
    for j, atoms in pieces:
        if j in unit.oxides:
            names.append(("oxo", False))
            bindings.append(("oxo", atoms))
            continue
        bond = mol.GetBondBetweenAtoms(x, j)
        piece_order = round(bond.GetBondTypeAsDouble())
        if piece_order not in (1, 2):
            return None
        sub = us._name_component(ctx, atoms, attach_hint=j, is_top=False,
                                 emit_branch_stereo=emit_branch_stereo,
                                 relative_ring_override=relative_ring_override)
        if sub is None:
            return None
        rendered = us._render_as_substituent(sub, piece_order)
        if emit_branch_stereo and sub.spine_atom_to_locant:
            rel = (relative_ring_override.get(frozenset(sub.spine_atom_to_locant))
                   if relative_ring_override else None)
            block = (f"{rel}-" if rel is not None else
                     us._branch_stereo_block(mol, sub.spine_atom_to_locant, x, j, ctx.ring_ylidene))
            if block:
                from .prefix_derivation import carried
                rendered = carried(block + rendered, like=rendered)
        names.append((rendered, False))
        bindings.append((rendered, sub.covers))
        charged |= sub.charged
        internal |= sub.internal_atoms

    head = f"-{core}" if names and unit.lam else core
    token = _assemble_decorated_amino_prefix(names, enclose=False, head=head)
    if not token:
        return None
    if unit.ionic:
        charged.add(x)
    return us._ComponentResult(
        name=token, bindings=bindings, covers=frozenset(component), attach_locant=None,
        is_prefix_ready=not is_top, charged=frozenset(charged), internal_atoms=frozenset(internal))
