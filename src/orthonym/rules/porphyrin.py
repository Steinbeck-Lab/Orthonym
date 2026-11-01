"""Substituted free-base porphyrins named on the retained parent ``porphyrin``.

``porphyrin`` is the semisystematic parent name of Blue Book (Table 10.1 "Names of
fundamental stereoparent structures (nonlimiting)", (d) Miscellanous). Its numbering and its implied
indicated hydrogen come from the tetrapyrrole recommendations that cites (reference 17,
<https://iupac.qmul.ac.uk/tetrapyrrole/TP1.html>):

*.1 "... the name "porhyrin" [sic] implies that the saturated nitrogen atoms are at positions 21
  and 23 unless specifically indicated otherwise.";
*.2 "The 1-24 numbering system... is adopted for the porphyrin nucleus": ring A is N21 with C1-C4,
  C5 is a meso carbon, ring B is N22 with C6-C9, C10 meso, ring C is N23 with C11-C14, C15 meso, ring D
  is N24 with C16-C19, C20 meso;
*.7 "Systematic Names for Substituted Porphyrins": when there is a choice, the lowest locants go to
  (a) indicated hydrogen, (b) principal groups named as suffix, (c) substituents named as prefixes and
  hydro prefixes, all considered together, (d) the substituent named as a prefix that is cited first.
  Its examples are spelled without indicated hydrogen: ``2,7,12,17-tetraethyl-3,8,13,18-tetramethylporphyrin``.

The name is built the way the other retained fused parents are built (``rules/purine.py``): an
atom-mapped skeleton query whose map numbers ARE the locants, and the shared fused-parent assembler
(``rules/fused_rings.py``) for multipliers, enclosing marks, alphanumerical order and substituent
prefixes. The producer returns ``None`` (the cascade goes on with today's names) for every molecule that
is not a free-base, prefix-only, mancude porphyrin; ``decline_reason`` names the class.

LABEL: Blue Book INTRODUCTION says "Preferred IUPAC names (PINs) are not identified for the
compounds in this Chapter" (Chapter, which holds. The default tier labels the names of this
module ``pin_verified`` because the natural-product route that calls it does so (the route policy of
``routing/dispatch_table.py::_handle_natural_product``), exactly as for the bare ``porphyrin`` and the
steroid names. That label is the route's policy, not a Blue Book PIN ruling, and the.. systematic
name of a substituted porphyrin is not derived here.

ANION: ``name_porphyrin_dianion`` names the dianion that a metal ion leaves behind, the free base
minus the two hydrons of its implied N-H (.1: N21 and N23). Anions derived from parent
hydrides and their derivatives: "An anion derived formally by the removal of one or more hydrons from
any position of a neutral parent hydride is preferably named by using the suffix '-ide'... Numerical
prefixes 'di', 'tri', etc. are used to denote multiplicity; locants identify positions of the negative
charges." The parent hydride is ``porphyrin`` (21H,23H), the hydrons come from 21 and 23, so the
parent of the anion is ``porphyrin-21,23-diide`` and the substituent prefixes are cited before it, as
before a parent with a suffix. The anion has no PIN: the compounds of carry no PIN label
 INTRODUCTION, quoted above), so the caller labels the name a verified systematic name.
"""
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from rdkit import Chem

PARENT_NAME = "porphyrin"
#: the parent of the anion that loses the two hydrons of the implied N-H (N21 and N23)
ANION_PARENT_NAME = PARENT_NAME + "-21,23-diide"

#.2, Fig. 2, written out. alpha: 1, 4, 6, 9, 11, 14, 16, 19; beta: 2, 3, 7, 8, 12, 13, 17, 18;
# meso: 5, 10, 15, 20; N21-N24 close rings A-D. 28 bonds.
_CORE_EDGES: Tuple[Tuple[int, int], ...] = (
    (1, 2), (2, 3), (3, 4), (21, 1), (21, 4),          # ring A
    (6, 7), (7, 8), (8, 9), (22, 6), (22, 9),          # ring B
    (11, 12), (12, 13), (13, 14), (23, 11), (23, 14),  # ring C
    (16, 17), (17, 18), (18, 19), (24, 16), (24, 19),  # ring D
    (4, 5), (5, 6), (9, 10), (10, 11), (14, 15), (15, 16), (19, 20), (20, 1),   # meso links
)
_NITROGEN_LOCANTS = (21, 22, 23, 24)
# the retained name implies the two hydrogens on N21 and N23 (.1)
_IMPLIED_NH = (21, 23)


@lru_cache(maxsize=1)
def _core_query() -> Chem.Mol:
    """The 24-atom core as a query: element C for 1-20, N for 21-24, any bond (Kekule and aromatic drawings
    match alike). Each query atom's map number is its locant. ``MolToSmarts`` followed by a parse reorders
    the atoms, so a locant is always read from ``GetAtomMapNum``, never from an atom index."""
    skeleton = Chem.RWMol()
    for locant in range(1, 25):
        atom = Chem.Atom(7 if locant in _NITROGEN_LOCANTS else 6)
        atom.SetAtomMapNum(locant)
        skeleton.AddAtom(atom)
    for first, second in _CORE_EDGES:
        skeleton.AddBond(first - 1, second - 1, Chem.BondType.SINGLE)
    return Chem.MolFromSmarts(Chem.MolToSmarts(skeleton).replace("-", "~"))


def _core_matches(mol) -> list:
    """Every way the 24-atom core matches the graph (the D4 symmetry gives 8 for a bare core). The cheap
    counts come first: a core needs 24 heavy atoms and 4 nitrogens."""
    if mol is None or mol.GetNumHeavyAtoms() < 24:
        return []
    if sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7) < 4:
        return []
    from ..perception.fullerene import cage_atoms
    cage = cage_atoms(mol)
    if cage:
        # a fullerene cage (60 carbon atoms of three ring bonds each) holds no part of the core, and matching the core
        # inside it costs seconds a call: the atoms left are matched and the matches mapped back to the molecule
        keep = [i for i in range(mol.GetNumAtoms()) if i not in cage]
        rest = Chem.RWMol(mol)
        for i in sorted(cage, reverse=True):
            rest.RemoveAtom(i)
        rest = rest.GetMol()
        rest.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(rest)
        return [tuple(keep[j] for j in match) for match in rest.GetSubstructMatches(_core_query(), uniquify=False)]
    return list(mol.GetSubstructMatches(_core_query(), uniquify=False))


def _locant_map(match) -> Dict[int, int]:
    """atom index -> locant for one match tuple (query atom i matched ``match[i]``)."""
    query = _core_query()
    return {atom_index: query.GetAtomWithIdx(i).GetAtomMapNum() for i, atom_index in enumerate(match)}


def _drawn_nh(mol, locants: Dict[int, int]) -> Tuple[int, ...]:
    """The locants of the core nitrogens that carry a hydrogen in the drawing."""
    return tuple(sorted(loc for index, loc in locants.items()
                        if loc in _NITROGEN_LOCANTS and mol.GetAtomWithIdx(index).GetTotalNumHs() == 1))


def core_numberings(mol) -> List[Dict[int, int]]:
    """The numberings of the molecule's porphyrin core that are names of it: the graph matches whose drawn
    N-H sit on N21 and N23 (.1). A trans drawing has 4 of them, a cis drawing none."""
    return [locants for locants in (_locant_map(m) for m in _core_matches(mol))
            if _drawn_nh(mol, locants) == _IMPLIED_NH]


def _is_charged(mol, core) -> bool:
    """A net formal charge, or a charge on a core atom. A charge-separated group outside the core whose
    charges cancel (nitro, azide) is not a charge."""
    return (sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) != 0
            or any(mol.GetAtomWithIdx(i).GetFormalCharge() for i in core))


def _is_mancude_core(mol, locants: Dict[int, int]) -> bool:
    """Every core carbon is three-connected counting hydrogens and has no exocyclic multiple bond."""
    core = set(locants)
    for index, loc in locants.items():
        if loc in _NITROGEN_LOCANTS:
            continue
        atom = mol.GetAtomWithIdx(index)
        if atom.GetTotalNumHs() + atom.GetDegree() != 3:
            return False
        for bond in atom.GetBonds():
            if (bond.GetOtherAtomIdx(index) not in core and bond.GetBondType() != Chem.BondType.SINGLE
                    and not bond.GetIsAromatic()):
                return False
    return True


def _has_stereo(mol) -> bool:
    """A chiral tag on any atom or a specified double-bond geometry on any bond."""
    if any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms()):
        return True
    unspecified = (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)
    return any(bond.GetStereo() not in unspecified for bond in mol.GetBonds())


def decline_reason(mol, *, bare_ok: bool = False) -> Optional[str]:
    """None when the molecule is a free-base, prefix-only, mancude porphyrin this module names; otherwise
    the reason code of the first guard that declines it. Pure: no naming. The order is the order of the
    guards, so a molecule with several reasons reports the first (the free-ion and N-metal rows keep
    ``fragments`` and ``n_substituted`` before the broad ``metal`` guard). ``bare_ok`` lets the bare
    24-atom parent pass: the exact lookup owns the free base, the anion has no entry of its own."""
    matches = _core_matches(mol)
    if not matches:
        return "no_core"
    if len(Chem.GetMolFrags(mol)) > 1:
        return "fragments"
    if len({frozenset(match) for match in matches}) > 1:
        return "multiple_cores"
    locants = _locant_map(matches[0])
    core = set(locants)
    if _is_charged(mol, core):
        return "charged"
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        return "radical"
    if any(atom.GetIsotope() for atom in mol.GetAtoms()):
        return "isotope"
    if any(mol.GetAtomWithIdx(i).GetDegree() != 2 for i, loc in locants.items() if loc in _NITROGEN_LOCANTS):
        return "n_substituted"
    from ..perception.metals import is_metal_element
    if any(is_metal_element(atom.GetSymbol()) for atom in mol.GetAtoms()):
        return "metal"      # a metal, or a semimetal, anywhere: is out of scope
    if not _is_mancude_core(mol, locants):
        return "not_mancude"
    if not core_numberings(mol):
        return "nh_pattern"
    from ..perception.rings import get_ring_systems
    if {i for system in get_ring_systems(mol) if system & core for i in system} != core:
        return "fused_ring"
    from ..perception.functional_groups import detect_functional_groups
    from .seniority import get_principal_group
    if get_principal_group(mol, detect_functional_groups(mol))[0] is not None:
        return "suffix_group"   # another parent is senior,: a different name
    if _has_stereo(mol):
        return "stereo"
    if mol.GetNumHeavyAtoms() == 24 and not bare_ok:
        return "bare"           # the exact lookup in data/natural_products.py owns the bare parent
    from .fused_rings import _exocyclic_atoms_accounted
    if not _exocyclic_atoms_accounted(mol, core):
        return "unaccounted"
    return None


def _name_on_parent(mol, parent_name: str) -> Optional[str]:
    """The name of a prefix-only porphyrin ``mol`` (a molecule ``decline_reason`` accepts) on the parent
    ``parent_name``. Among the numberings that put the drawn N-H on N21 and N23 (.1) the lowest
    locants decide (.7 (c), (d) = (c)/(f), (g)): the locants of the detachable prefixes
    together, then of the prefix cited first in alphanumerical order. (a) indicated hydrogen and (b)
    suffixes are equal for every candidate (no suffix group is in scope). The name is the shared
    assembler's; no string operation is applied to it."""
    from .fused_rings import (
        _assemble_fused_heterocycle_name,
        _first_cited_prefix_key,
        _fused_substituent_citation_key,
        get_fused_heterocycle_substituents,
    )
    rank = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    best = None
    for locants in core_numberings(mol):
        subs = get_fused_heterocycle_substituents(mol, locants)
        signature = tuple(locants[a] for a in sorted(locants, key=lambda a: (rank[a], a)))
        key = (_fused_substituent_citation_key(subs), _first_cited_prefix_key(subs), signature)
        if best is None or key < best[0]:
            best = (key, locants, subs)
    if best is None:
        return None
    _, locants, subs = best
    return _assemble_fused_heterocycle_name(mol, parent_name, subs, locants)


def name_substituted_porphyrin(mol) -> Optional[str]:
    """``5,10,15,20-tetraphenylporphyrin`` for a substituted free-base porphyrin, else None."""
    if decline_reason(mol) is not None:
        return None
    return _name_on_parent(mol, PARENT_NAME)


def _conjugate_acid_of_core_dianion(mol):
    """The free base that the dianion ``mol`` comes from, or None: ``mol`` is a net -2 ion whose only
    charges on the core are -1 on two nitrogens without hydrogen, and every other charge is part of a
    group whose charges cancel. The two nitrogens take a hydrogen each; the result is sanitised."""
    matches = _core_matches(mol)
    if not matches or len(Chem.GetMolFrags(mol)) > 1:
        return None
    core = set(matches[0])
    anionic = [i for i in core if mol.GetAtomWithIdx(i).GetAtomicNum() == 7
               and mol.GetAtomWithIdx(i).GetFormalCharge() == -1]
    if len(anionic) != 2 or any(mol.GetAtomWithIdx(i).GetTotalNumHs() for i in anionic):
        return None
    if sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) != -2:
        return None
    acid = Chem.RWMol(mol)
    for index in anionic:
        atom = acid.GetAtomWithIdx(index)
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(1)
    try:
        Chem.SanitizeMol(acid)
    except (ValueError, RuntimeError):
        return None
    return acid.GetMol()


def name_porphyrin_dianion(mol) -> Optional[str]:
    """``5,10,15,20-tetraphenylporphyrin-21,23-diide`` for the dianion of a prefix-only free-base
    porphyrin, quoted in the module docstring), else None.

    The anion is named as its free base is, on the parent ``porphyrin-21,23-diide``: the hydrons that
    leave are the two of the implied N-H (N21 and N23,.1), so the locants of the diide are the
    same for every numbering and the numbering is the free base's. A dianion whose negative nitrogens
    are neighbours (the free base would carry its N-H on N21 and N22) has no numbering that puts the
    drawn N-H on N21 and N23 and is declined; so is every ion with another net charge, a metal, a
    suffix group or a stereo descriptor, as for the free base."""
    acid = _conjugate_acid_of_core_dianion(mol)
    if acid is None or decline_reason(acid, bare_ok=True) is not None:
        return None
    return _name_on_parent(acid, ANION_PARENT_NAME)
