"""The heteroatom ligand of a P/As heterone: ``-O-P=O`` -> ``methoxyphosphanone``.

``mononuclear_hydrides.name_heterone`` names a hub X=O (X = P, As,...) whose other ligands are
hydrocarbyl or halogen organyls (``ethylphosphanone``, ``phenylphosphanone``). A ligand bonded to the
hub through O, S or N (``CH3-O-P=O``) was not an organyl, so the producer declined the whole molecule
and the molecule fell to a substitutive name built on the prefix 'oxophosphanyl'. That prefix is for
a heterone in the presence of a more senior group; here the heterone is the principal group.

Governing text (the Blue Book, quoted with its section heading):

* "HETERONES" (:25977): "Compounds containing the -PO, -PO2, -AsO or -AsO2 are called
  heterones (see,. In the presence of a more senior characteristic group they are
  described by the compound prefixes oxophosphanyl, dioxo-lambda5-phosphanyl, oxoarsanyl, and
  dioxo-lambda5-arsanyl." The class is -PO and -AsO in one sentence.
* example (:25985): ``[(2-methylpropyl)amino]arsanone (PIN)`` -- an arsanone whose only
  ligand is the heteroatom group 'amino' substituted by an alkyl, cited as a prefix of the heterone
  and not as a suffix of an amine. ``N-(2-methylpropyl)-1-oxoarsanamine`` is listed beside it without
  the (PIN) label.
* "Acyclic heterones" (:29390): the suffix 'one' "leads to preferred IUPAC names"
  ("Method (1)",:29396).
*, under " POLYFUNCTIONAL KETONES, PSEUDOKETONES AND HETERONES" (:29585):
  "Ketones, pseudoketones and heterones, and their chalcogen analogues in the order =O > =S > =Se
  > =Te, are senior to hydroxy compounds and their chalcogen analogues, amines, and imines in the
  seniority order of classes." An alkoxy, alkylsulfanyl or amino ligand therefore never outranks the
  heterone, and a ligand that holds any other group is not built here.
* / (:27633,:27649): 'methoxy', 'phenoxy', '(propan-2-yl)oxy' and
  'methylsulfanyl' are the preferred prefixes of the O-R and S-R groups; (:7272) cites
  the one ligand of a mononuclear hub with enclosing marks when it is a compound prefix.

The step from the arsanone amino example (:25985) to ``-O-P=O`` is by ANALOGY: the Blue Book prints no
phosphanone with an alkoxy ligand. The class, -PO and -AsO), the suffix rule and the
seniority are the same for P and As, and for O and N ligands. Nor does the book make ``-O-P=O``
an acid group of its own: "Names of mononuclear noncarbon oxoacids" (:35409, the list at:35431)
holds phosphinic, phosphinous, phosphonic, phosphonous, phosphoric and phosphorous acid and no HO-P=O, and
metaphosphorous acid, [HPO2]n = (-P(OH)O-)n, is polymeric and "only for general nomenclature" (:36677).
The hub is admitted only at bonding number 3 with one ligand, the shape of ``phenylphosphanone`` (:28291).
The label of the name is decided by that analogy: see the lane report.

What is built (fail closed in every other case): P or As hub, one terminal =O (or the =S, =Se, =Te
analogue that ``name_heterone`` already builds for organyl ligands: ``ethylphosphanethione``,
:29585 "and their chalcogen analogues in the order =O > =S > =Se > =Te"), ONE single-bonded ligand (no
hydrogen on the hub), the ligand an acyclic neutral O, S or N atom whose organyl part holds only
carbon and halogen atoms. A ligand holding any other group (hydroxy, a carbonyl, an ester, a nitro...)
needs a seniority comparison against the heterone and is left to the other producers.

The route is not offered while a fragment is being named for a prefix (``in_fragment_recursion``): the
recursive substituent namer turns the name of the H-capped fragment into a prefix by suffix surgery,
and a heterone has no '-yl' form (its prefix is 'oxophosphanyl',.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from rdkit import Chem

logger = logging.getLogger(__name__)

#: (:25977): the hubs of the heterone class that carry the ligand route (-PO and -AsO)
_HUBS = frozenset({'P', 'As'})
#: the ligand atoms: ether O, thioether S, amino N
_LINKS = frozenset({'O', 'S', 'N'})
#: the atoms of the organyl part of the ligand: carbon, and the halogens (prefix-only,
_ORGANYL = frozenset({'C', 'F', 'Cl', 'Br', 'I'})


def _ligand_atoms(mol, link_idx: int, hub_idx: int) -> Optional[List[int]]:
    """The ligand: the link atom and the subtree beyond it, every atom of the subtree a neutral,
    non-radical carbon or halogen; None otherwise (a second hetero atom is another class)."""
    atoms = [link_idx]
    seen = {hub_idx, link_idx}
    stack = [n.GetIdx() for n in mol.GetAtomWithIdx(link_idx).GetNeighbors()
             if n.GetIdx() != hub_idx and n.GetAtomicNum() > 1]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        atom = mol.GetAtomWithIdx(i)
        if (atom.GetSymbol() not in _ORGANYL or atom.GetFormalCharge()
                or atom.GetNumRadicalElectrons()):
            return None
        atoms.append(i)
        stack.extend(n.GetIdx() for n in atom.GetNeighbors()
                     if n.GetIdx() not in seen and n.GetAtomicNum() > 1)
    return atoms


def _amino_ligand(mol, frag: List[int], link_idx: int) -> Optional[str]:
    """The prefix of a substituted amino ligand, spelled by the shared chokepoint, or None. Only a
    single-bonded N carrying at least one carbon ('[(2-methylpropyl)amino]', 'dimethylamino') is
    accepted, and only a name that ends in 'amino' or 'anilino'."""
    link = mol.GetAtomWithIdx(link_idx)
    if link.GetDegree() < 2 or any(b.GetBondType() != Chem.BondType.SINGLE for b in link.GetBonds()):
        return None
    from ..assembly.substituent_enumerator import name_substituent
    name = name_substituent(mol, sorted(frag), link_idx, allow_mancude=False)
    if not name or not name.endswith(('amino', 'anilino')):
        return None
    return name


def _chalcogen_ligand(mol, frag: List[int], link_idx: int, hub_idx: int) -> Optional[str]:
    """The prefix of an R-O- or R-S- ligand. O: the organyl is named from its own carbon (the strict
    organyl chokepoint, hydrocarbyl and halogen only) and concatenated by the shared alkoxy
    morphology, which contracts a substituted phenyl to 'phenoxy' ('4-chlorophenoxy',,
    :27633). S: the shared primitive of the mono-chalcogen group, '(methylsulfanyl)',
    (:27649). The general cascade is not used: handed a chalcogen attachment it re-roots the fragment
    at a carbon and names a different constitution."""
    link = mol.GetAtomWithIdx(link_idx)
    if link.GetSymbol() == 'S':
        from ..assembly.substituent_enumerator import composed_chalcogen_group_prefix
        return composed_chalcogen_group_prefix(mol, frag, link_idx, {hub_idx})
    carbon = next(n.GetIdx() for n in link.GetNeighbors() if n.GetIdx() != hub_idx)
    from .substituent_purity import organyl_prefix_name
    organyl = organyl_prefix_name(mol, carbon, link_idx)
    if not organyl:
        return None
    from ..assembly.substituent_enumerator import alkoxy_prefix_from_substituent
    return alkoxy_prefix_from_substituent(organyl)


def hetero_ligand_prefix(mol, hub_idx: int, link_idx: int) -> Optional[str]:
    """The prefix name of the one ligand ``link_idx`` of the heterone hub ``hub_idx`` (P or As, one
    terminal =O/=S/=Se/=Te, bonding number 3), for ``CH3-O-P=O`` 'methoxy', ``C6H5-O-P=O`` 'phenoxy',
    ``(CH3)2CH-CH2-NH-As=O`` '(2-methylpropyl)amino'; None when the ligand is not in the class above.
    Pure: no mol mutation."""
    # The name of an isolated fragment (the H-capped substituent that the recursive substituent namer
    # names as a molecule and turns into a prefix by suffix surgery) is not offered: a heterone has no
    # '-yl' form, its prefix is 'oxophosphanyl',:25977). Offering '(ethenyloxy)phosphanone'
    # there made the fragment of an enol ether, C=C(R)-O-P=O under a senior group, a source of
    # '(ethenyloxy)phosphanonyl'; the structured prefix '[(oxophosphanyl)oxy]' was then lost.
    from ..assembly.fragment_naming import in_fragment_recursion
    if in_fragment_recursion():
        return None
    hub = mol.GetAtomWithIdx(hub_idx)
    if (hub.GetSymbol() not in _HUBS or hub.GetDegree() != 2 or hub.GetTotalNumHs()
            or hub.GetTotalValence() != 3):
        return None
    bond = mol.GetBondBetweenAtoms(hub_idx, link_idx)
    link = mol.GetAtomWithIdx(link_idx)
    sym = link.GetSymbol()
    if (bond is None or bond.GetBondType() != Chem.BondType.SINGLE or sym not in _LINKS
            or link.IsInRing() or link.GetFormalCharge() or link.GetNumRadicalElectrons()):
        return None
    frag = _ligand_atoms(mol, link_idx, hub_idx)
    if frag is None:
        return None
    if sym == 'N':
        name = _amino_ligand(mol, frag, link_idx)
    elif link.GetDegree() == 2 and not link.GetTotalNumHs():
        name = _chalcogen_ligand(mol, frag, link_idx, hub_idx)
    else:
        return None
    from ..errors import is_refusal_sentinel
    if not name or ' ' in name or is_refusal_sentinel(name):
        return None
    # The stereo-expression obligation of organyl_prefix_name: a prefix that does not express every
    # defined stereo element of its fragment names another compound (a bond or a centre the
    # substituent emitter cannot locate is dropped, and no later check compares stereo).
    from .stereochemistry import count_defined_stereo_in_fragment, count_expressed_stereo_descriptors
    defined = count_defined_stereo_in_fragment(mol, frag)
    if defined and count_expressed_stereo_descriptors(name) != defined:
        logger.debug("hetero_ligand_prefix: fail-closed on unexpressed stereo in %r", name)
        return None
    return name


__all__ = ["hetero_ligand_prefix"]
