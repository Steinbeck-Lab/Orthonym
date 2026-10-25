"""Amides of the mononuclear nitrogen and halogen oxoacids: nitramides, nitrous amides and the
amides of the halogen acids,,.

BLUE BOOK AUTHORITY
-------------------
 "SENIORITY ORDER FOR CLASSES" (the Blue Book) ranks the amides (class 11) above the
amines (class 19), so a nitrogen that carries a nitro, nitroso or halogen group is the nitrogen of
an amide of the acid that group belongs to, and its substituents are cited as prefixes.

 "N-SUBSTITUTION OF AMINES AND IMINES BY HETEROATOMS" (:26588-:26610): "This is a change. In
accordance with the seniority of classes (see, compounds such as R-NH-Cl, R-NH-NO, and
R-NH-NO2 are now named as derivatives of amides (see." Its examples (PIN):
'ethylhypochlorous amide' (:26604), 'methylnitrous amide' (:26606), 'methyl(nitro)nitramide'
(:26608), 'methylbromous amide' (:26610).

 "Amides and hydrazides of nitric and nitrous acids" (:35866): "Preferred IUPAC names
for amides and hydrazides of nitric and nitrous acids are now systematically based on nitric or
nitrous amide and hydrazide, in accordance with the seniority order of classes rather than as
nitro and nitroso amines" -- 'dipropylnitrous amide', 'butyl(ethyl)nitrous amide' (:35882,
:35884), '(chloromethyl)(methyl)nitramide' (:35886).

 "Amides of halogen acids" (:39700): "The seniority order of classes is applied to
give preferred IUPAC names for amides of halogen acids." -- 'methylbromic amide' (:39704),
'ethylhypochlorous amide' (:39706); (:25880) 'methylhypochlorous amide'.

The acids are those of "CLASS 7e. OTHER MONOBASIC 'OXO' ACIDS USED AS FUNCTIONAL PARENTS"
(:18453): nitric, nitrous, perfluoric... hypofluorous, perchloric... hypochlorous, perbromic...
hypobromous, periodic... hypoiodous. Their seniority is "(a) the central atom first in the list
N > F > Cl > Br > I; (b) highest oxidation number for the central atom", which chooses the parent
when the nitrogen carries two such groups ('methyl(nitro)nitramide': one nitro group is the
amide, the other a prefix).

The amide nitrogen is the ONLY substitutable position of the amide, so its prefixes are cited
without the locant N in the (:7272) form of a mononuclear parent:
'methyl(nitro)nitramide', 'butyl(ethyl)nitrous amide', 'methylbromic amide'.

SCOPE (fails closed on everything else, never a per-molecule case)
------------------------------------------------------------------
1. **One acyclic amide nitrogen** with 1-2 acid groups (nitro, nitroso, a halogen with 0-3 oxygen
   atoms) and 0-2 carbon substituents, the rest of the molecule being exactly those substituents
   (so that every atom is accounted for). A nitrogen in a ring is a heterocycle N-substituted by
   the group ('1-chloropiperidine'), which the book's cyclic-amide rule (class 11, "cyclic amides
   are named as heterocycles") leaves to the ring names.
2. **Nothing in the molecule senior to this amide**:18162: the ions, classes 4-6, the acids,
   anhydrides, esters, acid halides and the amides of the senior acids, classes 7 to 11 -- a carboxylic or sulfonic
   amide, a urea, carbamate, amidine, hydroxamic acid). The check fails closed: the molecule is named here only
   when every group in it is on the explicit JUNIOR list (``_JUNIOR_GROUPS``: hydrazides, nitriles, aldehydes,
   ketones, hydroxy compounds, amines, ethers, sulfides, the prefix-only groups...), the structure has no
   charge but the nitro and azido groups of a carbon, and no carbonyl carbon with a heteroatom neighbour but the
   hydrazide's. Where it fails, that group is the principal one and the groups of this module are N-prefixes of
   its name ('N-methyl-N-nitrosourea (PIN)',:25965): ``rules/n_hetero_amines.py``. The nitramide is held
   to the same check: 'acetylnitramide' and '(carboxymethyl)nitramide' are not names of the amide of nitric acid
   when the acetamide or the acid is the class.
3. **``N,N'-dinitromethanediamine``**: exactly two nitro-bearing amide nitrogens, each with NO
   other substituent, both attached to the SAME bridging methylene (the one shape with a real
   witness); a numeric-locant diamine would need superscripted ``N^1``/``N^3`` locants.

Plain ``nitramide`` (H2N-NO2) stays with the retained-name table; the other bare amides
('nitrous amide', 'hypochlorous amide'...) are named here, "Amides and hydrazides"
(:35760): "Amides and hydrazides are named by functional class nomenclature by replacing the term
'acid' in the name of the corresponding acid by 'amide'".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional

from rdkit import Chem

_NITRO_SMARTS = Chem.MolFromSmarts('[NX3+](=O)[O-]')

# (:18453) acids and Table 5.1 (:24980) prefixes of the halogen groups, by the central
# atom (atomic number) and the number of oxygen atoms on it (0, 1, 2, 3).
_HALOGEN_ACIDS = {
    9: ('hypofluorous', 'fluorous', 'fluoric', 'perfluoric'),
    17: ('hypochlorous', 'chlorous', 'chloric', 'perchloric'),
    35: ('hypobromous', 'bromous', 'bromic', 'perbromic'),
    53: ('hypoiodous', 'iodous', 'iodic', 'periodic'),
}
_HALOGEN_PREFIXES = {
    9: ('fluoro', 'fluorosyl', 'fluoryl', 'perfluoryl'),
    17: ('chloro', 'chlorosyl', 'chloryl', 'perchloryl'),
    35: ('bromo', 'bromosyl', 'bromyl', 'perbromyl'),
    53: ('iodo', 'iodosyl', 'iodyl', 'periodyl'),
}
# (a): "the central atom first in the list N > F > Cl > Br > I".
_ELEMENT_RANK = {7: 0, 9: 1, 17: 2, 35: 3, 53: 4}


@dataclass(frozen=True)
class _AcidGroup:
    """A nitro, nitroso or halogen(oxide) group on the amide nitrogen: the central atom and its
    terminal oxygens (which are the group; the nitrogen it is on is not)."""
    atoms: FrozenSet[int]
    element: int
    n_oxygen: int

    def parent_word(self) -> str:
        """The class name of the amide of this group's acid,,."""
        if self.element == 7:
            return 'nitramide' if self.n_oxygen == 2 else 'nitrous amide'
        return f'{_HALOGEN_ACIDS[self.element][self.n_oxygen]} amide'

    def prefix_name(self) -> str:
        """The prefix of the group (Table 5.1): nitro, nitroso, chloro, chlorosyl, chloryl,
        perchloryl..."""
        if self.element == 7:
            return 'nitro' if self.n_oxygen == 2 else 'nitroso'
        return _HALOGEN_PREFIXES[self.element][self.n_oxygen]

    def seniority_key(self):
        """: the central atom N > F > Cl > Br > I, then the highest oxidation number."""
        return (_ELEMENT_RANK[self.element], -self.n_oxygen)


def _acid_group(mol, n_atom, z_atom) -> Optional[_AcidGroup]:
    """The acid group whose central atom ``z_atom`` is bonded to the amide nitrogen, or None.

    The central atom is a nitrogen (nitro: two oxygens, nitroso: one) or a halogen with up to
    three terminal oxygens, every oxygen terminal, the group neutral as a whole (the charge
    separated and the hypervalent drawings are one group), and bonded to the amide nitrogen
    and to nothing else."""
    z = z_atom.GetAtomicNum()
    if z != 7 and z not in _HALOGEN_ACIDS:
        return None
    if z_atom.GetIsAromatic() or z_atom.IsInRing() or z_atom.GetTotalNumHs():
        return None
    bond = mol.GetBondBetweenAtoms(n_atom.GetIdx(), z_atom.GetIdx())
    if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
        return None
    oxygens = [nb for nb in z_atom.GetNeighbors() if nb.GetIdx() != n_atom.GetIdx()]
    for o in oxygens:
        if o.GetAtomicNum() != 8 or o.GetDegree() != 1 or o.GetTotalNumHs():
            return None
    if z_atom.GetFormalCharge() + sum(o.GetFormalCharge() for o in oxygens) != 0:
        return None
    if z_atom.GetNumRadicalElectrons() or any(o.GetNumRadicalElectrons() for o in oxygens):
        return None
    if any(a.GetIsotope() for a in [z_atom] + oxygens):
        return None
    if z == 7 and len(oxygens) not in (1, 2):
        return None
    if z != 7 and len(oxygens) > 3:
        return None
    return _AcidGroup(
        atoms=frozenset([z_atom.GetIdx()] + [o.GetIdx() for o in oxygens]),
        element=z, n_oxygen=len(oxygens))


def _is_amide_nitrogen_candidate(atom) -> bool:
    """A neutral, acyclic, non-aromatic nitrogen with three single bonds (H counted)."""
    if atom.GetAtomicNum() != 7 or atom.GetFormalCharge() or atom.GetIsAromatic() or atom.IsInRing():
        return False
    if atom.GetNumRadicalElectrons() or atom.GetIsotope():
        return False
    if any(b.GetBondType() != Chem.BondType.SINGLE for b in atom.GetBonds()):
        return False
    return atom.GetDegree() + atom.GetTotalNumHs() == 3


def _nitro_amide_units(mol) -> List[Dict[str, int]]:
    """Every ``R2N-[N+](=O)[O-]`` unit: a nitro group whose N is directly
    bonded to an otherwise-neutral NITROGEN (the "amide" nitrogen this
    module is about) rather than to carbon (ordinary nitroalkane/nitroarene
    -- a completely different, already-working class this must never
    touch). Returns one ``{'nitro_n': idx, 'amide_n': idx}`` dict per
    independent unit.
    """
    if mol is None or _NITRO_SMARTS is None:
        return []
    units: List[Dict[str, int]] = []
    for match in mol.GetSubstructMatches(_NITRO_SMARTS):
        n_no2 = match[0]
        atom = mol.GetAtomWithIdx(n_no2)
        others = [nb for nb in atom.GetNeighbors() if nb.GetIdx() not in match]
        if len(others) != 1:
            continue
        amide = others[0]
        if amide.GetAtomicNum() != 7 or amide.GetFormalCharge() != 0:
            continue
        units.append({'nitro_n': n_no2, 'amide_n': amide.GetIdx()})
    return units


def _amide_unit(mol):
    """The one acyclic amide nitrogen of the molecule that carries acid groups, as
    ``(n_idx, groups, carbon_roots)``, or None when there is not exactly one, when a neighbour
    of it is neither an acid group nor a carbon, or when it has no group."""
    found = None
    for atom in mol.GetAtoms():
        if not _is_amide_nitrogen_candidate(atom):
            continue
        groups: List[_AcidGroup] = []
        roots: List[int] = []
        for nb in atom.GetNeighbors():
            group = _acid_group(mol, atom, nb)
            if group is not None:
                groups.append(group)
            elif nb.GetAtomicNum() == 6:
                roots.append(nb.GetIdx())
            else:
                groups = []
                break
        if not groups:
            continue
        if found is not None:
            return None  # a second amide nitrogen with a group: not this shape
        found = (atom.GetIdx(), groups, roots)
    return found


# The groups of the classes JUNIOR to the amides of the class 7e acids, Table 4.1, the Blue Book-:18197
# and the prefix-only groups of Table 5.1,:24980): hydrazides (class 12), nitriles (14), aldehydes (15), ketones
# (16), hydroxy compounds (17), hydroperoxides (18), amines (19), imines (20), the groups of the senior atom
# (21 and below: hydrazines, hydroxylamines, ethers, sulfides, sulfoxides, sulfones, peroxides), and the groups
# cited only as prefixes (halogens, nitro, nitroso, azido, diazo, azo). Every other perceived group is, or may be,
# senior (an acid, ester, acid halide or amide of the classes 7 to 11, an ion, urea, carbamate, amidine,
# hydroxamic acid...): the check fails closed.
_JUNIOR_GROUPS = frozenset({
    'hydrazide', 'thiohydrazide', 'sulfonohydrazide',
    'nitrile', 'isocyanide',
    'aldehyde', 'ketone', 'thioaldehyde', 'thioketone',
    'alcohol', 'primary_alcohol', 'secondary_alcohol', 'tertiary_alcohol', 'phenol', 'enol',
    'thiol', 'hydroperoxide', 'so_thioperoxol', 'os_thioperoxol', 'dithioperoxol',
    'primary_amine', 'secondary_amine', 'tertiary_amine', 'aromatic_amine',
    'imine', 'oxime', 'hydrazone', 'hydroxylamine', 'hydrazine_fg', 'aminooxy',
    'ether', 'aromatic_ether', 'vinyl_ether', 'thioether', 'selenoether', 'disulfide', 'peroxide', 'sulfoxide', 'sulfone',
    'primary_phosphine', 'secondary_phosphine', 'tertiary_phosphine',
    'alkene', 'alkyne',
    'fluoro', 'chloro', 'bromo', 'iodo', 'n_fluoroamine', 'n_chloroamine', 'n_bromoamine', 'n_iodoamine',
    'nitro', 'nitroso', 'azido', 'azo', 'diazo', 'diazenyl',
})
# carbon, nitrogen, oxygen, the halogens and sulfur, and the elements whose compounds are classes 22 and below of
# (phosphorus 22, silicon 26, boron 30, selenium 37), junior to every amide: phosphanes, silanes,
# boranes, selenides. Their acids and esters (a phosphorus acid group, a P=O bonded to O or N) are the senior ones:
# the atom layer refuses them.
_ELEMENTS_OF_JUNIOR_GROUPS = frozenset({5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53})
_HALOGEN_Z = frozenset({9, 17, 35, 53})
_C_NITRO = Chem.MolFromSmarts('[#6][N+](=O)[O-]')
_C_AZIDO = Chem.MolFromSmarts('[#6]N=[N+]=[N-]')


def _bonds_to(atom, kinds):
    return [b for b in atom.GetBonds() if b.GetBondType() in kinds]


def _carbon_is_junior(atom) -> bool:
    """A carbon that is no carbonyl, imidoyl or thiocarbonyl of an acid, ester, amide, amidine, urea...: with a
    double or triple bond to a heteroatom (ketone, aldehyde, imine, nitrile) its other neighbours are carbon, or
    -- the hydrazide, class 12 -- one nitrogen that has a nitrogen neighbour of its own."""
    multi = [b for b in _bonds_to(atom, (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE))
             if b.GetOtherAtom(atom).GetAtomicNum() != 6]
    if not multi:
        return True
    if len(multi) > 1:
        return False
    partner = multi[0].GetOtherAtom(atom)
    others = [b for b in atom.GetBonds()
              if b.GetIdx() != multi[0].GetIdx() and b.GetOtherAtom(atom).GetAtomicNum() != 6]
    if not others:
        return True
    if len(others) == 1 and partner.GetAtomicNum() == 8 and others[0].GetBondType() == Chem.BondType.SINGLE:
        amine = others[0].GetOtherAtom(atom)
        if amine.GetAtomicNum() == 7 and not amine.GetIsAromatic() and not amine.GetFormalCharge():
            return any(nn.GetAtomicNum() == 7 and nn.GetIdx() != atom.GetIdx() and not nn.GetFormalCharge()
                       and not _bonds_to(nn, (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE))
                       for nn in amine.GetNeighbors())
    return False


def _sulfur_is_junior(atom) -> bool:
    """Thioether, thiol, disulfide, sulfoxide, sulfone, or a ring sulfur: never a sulfonic, sulfinic... acid
    group, which has a heteroatom besides the oxygens on the sulfur."""
    if atom.GetFormalCharge():
        return False
    if atom.GetIsAromatic():
        return True
    oxo = [nb for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 8 and nb.GetDegree() == 1]
    oxo_idx = {nb.GetIdx() for nb in oxo}
    rest = [nb for nb in atom.GetNeighbors() if nb.GetIdx() not in oxo_idx]
    if oxo:
        return len(oxo) in (1, 2) and len(rest) == 2 and all(nb.GetAtomicNum() == 6 for nb in rest)
    if _bonds_to(atom, (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)):
        return False
    return all(nb.GetAtomicNum() in (6, 16) for nb in rest)


def _oxygen_is_junior(atom) -> bool:
    """Hydroxy, ether, peroxide, carbonyl or sulfinyl/sulfonyl oxygen, a nitro or nitroso oxygen; never an
    ester oxygen of a nitric, nitrous, sulfuric... acid (a neighbour N with a double bond to oxygen, a
    sulfur, a halogen) nor the oxygen of a carbonyl that the carbon check refuses."""
    if atom.GetIsAromatic():
        return True
    for nb in atom.GetNeighbors():
        if nb.GetAtomicNum() not in (6, 7, 8, 16):
            return False
        if atom.GetDegree() == 2 and nb.GetAtomicNum() in (7, 16):
            if nb.GetAtomicNum() == 16 or any(
                    b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(nb).GetAtomicNum() == 8
                    for b in nb.GetBonds()):
                return False
    return True


def _nitrogen_is_junior(atom) -> bool:
    """An amine, hydrazine, hydroxylamine or ring nitrogen, an imine, oxime, hydrazone, azo or nitrile nitrogen, a
    C-nitroso nitrogen; a charged nitrogen only as the nitro or azido group of a carbon (checked by the caller)."""
    if atom.GetIsAromatic() or atom.GetFormalCharge():
        return True
    nbs = list(atom.GetNeighbors())
    oxo = [nb for nb in nbs if nb.GetAtomicNum() == 8 and nb.GetDegree() == 1
           and atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx()).GetBondType()
           == Chem.BondType.DOUBLE]
    if oxo:                                 # nitroso: one carbon besides the oxygen
        return len(nbs) == 2 and all(nb.GetAtomicNum() in (6, 8) for nb in nbs)
    return all(nb.GetAtomicNum() in (6, 7, 8) for nb in nbs)


def _heavier_element_is_junior(atom) -> bool:
    """A phosphane, silane, borane or selenide classes 22, 26, 30, 37): neutral, single bonds,
    only carbon as neighbours (a selenide has two, a selenol is not taken: the substituent namer has no name for
    its group). A phosphorus, silicon, boron or selenium atom with an oxygen, nitrogen or halogen neighbour or a
    double bond is a group of an acid, an ester or an amide of that element: refused."""
    if atom.GetFormalCharge() or _bonds_to(atom, (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)):
        return False
    if atom.GetAtomicNum() == 34 and atom.GetDegree() != 2:
        return False
    return all(nb.GetAtomicNum() == 6 for nb in atom.GetNeighbors())


def _atoms_are_junior(mol, atoms) -> bool:
    """True when every atom of ``atoms`` belongs to a group of a class junior to the amides of the class 7e acids:
    only carbon, nitrogen, oxygen, sulfur and the halogens; no charge but the nitro and azido groups of a carbon;
    no carbonyl or imidoyl carbon with a heteroatom neighbour (acid, ester, amide, amidine, urea, carbamate,
    hydroxamic acid: the hydrazide is the one exception); no sulfonic, sulfinic... group; no ester of an
    oxoacid; no halogen on a heteroatom."""
    charged_ok = set()
    for patt in (_C_NITRO, _C_AZIDO):
        for match in mol.GetSubstructMatches(patt):
            charged_ok.update(match[1:])
    for idx in atoms:
        atom = mol.GetAtomWithIdx(idx)
        z = atom.GetAtomicNum()
        if z not in _ELEMENTS_OF_JUNIOR_GROUPS or atom.GetNumRadicalElectrons():
            return False
        if atom.GetFormalCharge() and idx not in charged_ok:
            return False
        if z == 6:
            ok = _carbon_is_junior(atom)
        elif z == 16:
            ok = _sulfur_is_junior(atom)
        elif z == 8:
            ok = _oxygen_is_junior(atom)
        elif z == 7:
            ok = _nitrogen_is_junior(atom)
        elif z in _HALOGEN_Z:
            ok = atom.GetDegree() == 1 and atom.GetNeighbors()[0].GetAtomicNum() == 6
        else:                                   # P, Si, B, Se: a phosphane, silane, borane, selenide
            ok = _heavier_element_is_junior(atom)
        if not ok:
            return False
    return True


def _branches_are_junior(mol, branches: List[List[int]]) -> bool:
    """The check of that the amide of a class 7e acid is the class of the molecule: every atom outside the
    amide nitrogen and its acid groups is in a group junior to it (:func:`_atoms_are_junior`) and every group
    the perception finds is on the junior list (``_JUNIOR_GROUPS``). Fails closed: a group in neither is
    refused, and the group that is the principal one is then the class of the molecule (its name, and the
    groups of this module as N-prefixes of it: ``rules/n_hetero_amines.py``)."""
    from ..perception.functional_groups import detect_functional_groups

    if not _atoms_are_junior(mol, [i for branch in branches for i in branch]):
        return False
    return all(name in _JUNIOR_GROUPS for name, found in detect_functional_groups(mol).items() if found)


def _name_oxoacid_amide(mol, amide_n: int, groups: List[_AcidGroup], roots: List[int],
                        style: str = 'pin') -> Optional[str]:
    """The amide of the senior acid of the groups on ``amide_n``, with every other group and the
    carbon substituents as prefixes: ``(chloromethyl)(methyl)nitramide``, ``dipropylnitrous
    amide``, ``methyl(nitro)nitramide``, ``ethylhypochlorous amide``."""
    from .sulfonamides import _branch_atoms

    # One group, or two on a nitrogen that has a carbon substituent: the more senior acid is the amide
    # and the other group its prefix, as in the book's 'methyl(nitro)nitramide' (:26608).
    if len(groups) > 2 or (len(groups) == 2 and not roots):
        return None
    if len(roots) + len(groups) > 3:
        return None
    parent = min(groups, key=lambda g: g.seniority_key())
    others = [g for g in groups if g is not parent]

    # The branches of an acyclic nitrogen are disjoint; the amide N, its groups and its carbon
    # branches must be the WHOLE molecule (a second, independent unit would leave atoms out).
    branches = [_branch_atoms(mol, root, amide_n) for root in roots]
    covered = {amide_n}
    for grp in groups:
        covered |= grp.atoms
    for branch in branches:
        covered.update(branch)
    if len(covered) != mol.GetNumAtoms():
        return None

    #: the amide of a class 7e acid is the class only where nothing in the molecule is senior to it
    # (an acid, ester, acid halide, an amide of a carboxylic or sulfonic acid, a cation or anion...);
    # that group is the principal one then, and the groups of this module are prefixes of its name.
    if not _branches_are_junior(mol, branches):
        return None

    from ..errors import is_refusal_sentinel
    from .amides import _name_n_substituent

    names: List[str] = []
    for branch in branches:
        name = _name_n_substituent(mol, branch, len(branch))
        if not name or is_refusal_sentinel(name):
            return None  # a refusal sentinel must never be woven into the emitted name
        names.append(name)
    names.extend(g.prefix_name() for g in others)

    word = parent.parent_word()
    if not names:
        # bare amide: plain nitramide is the retained-name table's
        return None if word == 'nitramide' else word

    # (the Blue Book, "(chloromethyl)(methyl)nitramide (PIN)"): the amide N of the functional
    # parent is its ONLY substitutable position, so the substituent locants are OMITTED
    # -- the PIN is '(chloromethyl)(methyl)nitramide', NOT
    # 'N-(chloromethyl)-N-methylnitramide' (that spelling is the alternative *methanamine*-parent
    # name, where 'N' locants ARE needed). The prefixes are cited without the 'N-' locant, in
    # alphanumerical order, enclosed per the mononuclear single-attachment rule
    # (the Blue Book): the first cited substituent is bare (unless compound/complex,
    #, the second and further are each parenthesised even when simple ->
    # 'methyl(nitro)nitramide'. Multiplied identical simple substituents keep the multiplier
    # outside the marks.
    from ..assembly.prefix_marks import mononuclear_substituent_block
    prefix = mononuclear_substituent_block(names)
    if not prefix:
        return None
    return f"{prefix}{word}"


def _name_dinitro_diamine(mol, units: List[Dict[str, int]], style: str) -> Optional[str]:
    """Shape 3: ``N,N'-dinitro<diamine>`` for exactly two nitro-amide units
    sharing one plain methylene bridge (the only verified real-witness
    shape). Fails closed on anything wider."""
    amide_ns = [u['amide_n'] for u in units]
    nitro_ns = [u['nitro_n'] for u in units]

    bridges = set()
    for amide_n in amide_ns:
        atom = mol.GetAtomWithIdx(amide_n)
        others = [nb.GetIdx() for nb in atom.GetNeighbors()
                  if nb.GetIdx() not in nitro_ns]
        if len(others) != 1:
            return None  # an amide N with any OTHER substituent is out of scope
        bridges.add(others[0])
    if len(bridges) != 1:
        return None  # the two amide N's must share the SAME bridging atom

    bridge_idx = next(iter(bridges))
    bridge_atom = mol.GetAtomWithIdx(bridge_idx)
    if bridge_atom.GetAtomicNum() != 6:
        return None
    heavy_neighbors = {nb.GetIdx() for nb in bridge_atom.GetNeighbors()}
    if heavy_neighbors != set(amide_ns):
        return None  # the bridge carbon must bear ONLY the two amide N's
    if bridge_atom.GetIsAromatic() or bridge_atom.IsInRing():
        return None

    # This structural shape (a carbon whose ONLY heavy-atom connections are
    # the two qualifying amide nitrogens) has exactly one possible parent
    # hydride: a single carbon carrying two implicit H's plus the two amine
    # groups, i.e. methanediamine. This is NOT a per-molecule string special
    # case -- it is the unique, graph-forced consequence of the structural
    # guards just above (degree-2 heavy connectivity, both neighbours being
    # the two amide N's, carbon, non-aromatic, ring-free) -- and it sidesteps
    # a SEPARATE, pre-existing, out-of-scope gap: `name_compound('NCN')`
    # (bare methanediamine, no nitro at all) itself currently mis-names to
    # 'methanamine' and is suppressed by to the sentinel, because
    # this tree's general amine producer does not yet build a gem-diamine
    # (two -NH2 on the SAME carbon) parent. Fixing that general gap is out
    # of scope for a nitro/nitramide task; the graph proof above lets this
    # narrow branch stay correct without depending on it.
    return "N,N'-dinitromethanediamine"


def name_substituted_nitramide(mol, style: str = 'pin') -> Optional[str]:
    """Return the PIN of an amide of a mononuclear nitrogen or halogen oxoacid, else None.

    ``(hydroxymethyl)nitramide``, ``bis(hydroxymethyl)nitramide``, ``dipropylnitrous amide``,
    ``methyl(nitro)nitramide``, ``ethylhypochlorous amide``, ``methylbromic amide``,
    ``N,N'-dinitromethanediamine``. Fails closed on every shape outside the three documented in
    this module's docstring."""
    if mol is None:
        return None
    # Only the name of a whole molecule. A fragment named inside a substituent's recursion is asked for a parent
    # name that the prefix converter turns into the prefix of the group ('[chloro(methyl)amino]'); a
    # functional class name ('methylhypochlorous amide') has no such conversion, and the composed prefix that
    # the general engine builds for the group is lost (see ``n_hetero_amines.name_n_hetero_substituted``).
    from ..assembly.fragment_naming import is_top_level_naming

    if not is_top_level_naming():
        return None
    units = _nitro_amide_units(mol)
    if len(units) == 2 and units[0]['amide_n'] != units[1]['amide_n']:
        return _name_dinitro_diamine(mol, units, style)
    found = _amide_unit(mol)
    if found is None:
        return None
    amide_n, groups, roots = found
    return _name_oxoacid_amide(mol, amide_n, groups, roots, style)
