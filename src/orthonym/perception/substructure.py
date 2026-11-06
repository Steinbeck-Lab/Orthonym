"""Substructure matches of a query whose atom order is not the caller's choice.

RDKit's matcher extends a partial match in the atom order of the query. A molecule parsed from an OPSIN name comes
in OPSIN's order, which can start at a leaf of a highly symmetric molecule (a methyl of one of fifteen tert-butyl
groups); the matcher then tries the automorphisms of every leaf group before it reaches an atom that fails, and on
such a molecule it does not return (measured: a 134-atom boron-carbazole emitter, no return in 130 s). The same
query renumbered into its canonical SMILES order, which walks the molecule from a canonical start along bonds,
matches in 0.08 s with the same matches. ``ordered_matches`` searches that renumbered query and returns each match
indexed by the query's own atoms, so callers read it as before.
"""
from typing import TYPE_CHECKING, List, Tuple

from rdkit import Chem

if TYPE_CHECKING:  # numpy loads only when a match is searched: ``import orthonym`` starts no BLAS threads
    import numpy as np


def ordered_matches(target: Chem.Mol, query: Chem.Mol, *, use_chirality: bool = False, uniquify: bool = True,
                    max_matches: int = 1000) -> List[Tuple[int, ...]]:
    """:func:`ordered_match_array` as a list of tuples."""
    return [tuple(row) for row in ordered_match_array(target, query, use_chirality=use_chirality,
                                                       uniquify=uniquify, max_matches=max_matches).tolist()]


def ordered_match_array(target: Chem.Mol, query: Chem.Mol, *, use_chirality: bool = False, uniquify: bool = True,
                        max_matches: int = 1000) -> "np.ndarray":
    """``target.GetSubstructMatches(query,...)`` searched in the canonical atom order of ``query``.

    One row per match, indexed by the atoms of ``query`` (as RDKit returns a match); an array, because a caller
    that ranks 100,000 matches of a large molecule spent seconds per call on the tuples alone. Which matches are
    found first can
    differ from a search in the query's own order: uncapped and with ``uniquify=False`` the set of matches is the
    same; with ``uniquify=True`` the representative of a set of atoms can differ, and under ``max_matches`` the
    subset can.
    """
    import numpy as np

    work = Chem.Mol(query)
    try:
        Chem.MolToSmiles(work)
        order = [int(i) for i in work.GetPropsAsDict(True, True).get("_smilesAtomOutputOrder", ())]
    except Exception:  # noqa: BLE001 -- no canonical order: the query is searched as given
        order = []
    if sorted(order) != list(range(query.GetNumAtoms())):
        found = target.GetSubstructMatches(query, useChirality=use_chirality, uniquify=uniquify,
                                           maxMatches=max_matches)
        return np.asarray(found, dtype=np.intp).reshape(-1, query.GetNumAtoms())
    renumbered = Chem.RenumberAtoms(query, order)
    pair = same_graph_pair(target, renumbered)       # a mixture of many components: see same_graph_pair
    if pair is None:
        found = target.GetSubstructMatches(renumbered, useChirality=use_chirality, uniquify=uniquify,
                                           maxMatches=max_matches)
    else:
        target, renumbered, params = pair
        params.uniquify, params.maxMatches, params.useChirality = uniquify, max_matches, use_chirality
        found = target.GetSubstructMatches(renumbered, params)
    if not found:
        return np.empty((0, len(order)), dtype=np.intp)
    # match[new] is the target atom of query atom order[new]: index the columns by the inverse order
    inverse = np.empty(len(order), dtype=np.intp)
    inverse[np.asarray(order, dtype=np.intp)] = np.arange(len(order), dtype=np.intp)
    return np.asarray(found, dtype=np.intp)[:, inverse]


#: atom property that:func:`same_graph_pair` sets on copies of both graphs
_COMPONENT_SIZE = "_orthonym_component_shape"


def same_graph_pair(target: Chem.Mol, query: Chem.Mol):
    """``(target, query, parameters)`` to match ``query`` onto ``target`` when they can be the same graph, else None.

    A mixture drawn as one graph (a molecule with many identical or similar components) is a disconnected query,
    and RDKit's matcher takes its atoms in the query's order: it lets a one-atom component take a carbon of an
    ethane component of the target, finds that the ethane components do not fit only after the last component, and
    backtracks through every way of choosing which carbons the one-atom components take (measured: a 33-atom mixture
    of 19 components, no return in 30 s, the call cannot be interrupted). When the query and the target have the
    same number of atoms and of bonds and more than one component, every match is an isomorphism, and it maps each
    component onto a component of the same atom and bond count. Each atom of a copy of both graphs carries the atom
    and bond count of its component, and the matcher compares it (``SubstructMatchParameters.atomProperties``): the
    branches that it cuts hold no complete match, so the matches that remain are the matches of the plain call, in
    the order of the plain call, and so are the first ``n`` of them.

    A connected pair of the same sizes is tagged with the local environment and the symmetry class of each atom
    (:func:`_connected_pair`) for the same reason: a molecule with many equivalent groups (a cyclodextrin with silyl
    ethers on its ring units, 142 atoms) sends the matcher into dead ends from which it does not come back, in the
    query's own order and in its canonical order (measured: no return in 40 s for the first 100 matches; with the
    classes, 0.0 s), and so does a query of the input's size that is another molecule (a name read with one bond
    order otherwise), which holds no match at all. None for any other pair: the plain call is made."""
    if query.GetNumAtoms() != target.GetNumAtoms() or query.GetNumBonds() != target.GetNumBonds():
        return None
    if len(Chem.GetMolFrags(query)) < 2:
        return _connected_pair(target, query)
    tagged = []
    for graph in (target, query):
        copy = Chem.Mol(graph)
        for frag in Chem.GetMolFrags(copy):
            members = set(frag)
            bonds = sum(1 for b in copy.GetBonds() if b.GetBeginAtomIdx() in members)
            for idx in frag:
                copy.GetAtomWithIdx(idx).SetProp(_COMPONENT_SIZE, f"{len(frag)}:{bonds}")
        tagged.append(copy)
    params = Chem.SubstructMatchParameters()
    params.atomProperties = [_COMPONENT_SIZE]
    return tagged[0], tagged[1], params


#: atom properties that:func:`_connected_pair` sets on copies of both graphs
_SYMMETRY_CLASS = "_orthonym_symmetry_class"
_ENVIRONMENT = "_orthonym_local_environment"


def _symmetry_classes(graph: Chem.Mol) -> List[int]:
    """The canonical symmetry class of each atom of ``graph`` read as a graph of element labels: bond orders,
    aromaticity, charges, isotopes, hydrogen counts and stereo are left out, so the classes are those every match
    preserves (a match keeps the elements and the bonds; a bijection of equal counts keeps all of them). The classes
    are invariant under a renumbering of the atoms, so atoms that a match pairs have the same class in both graphs."""
    work = Chem.RWMol(graph)
    for atom in work.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsotope(0)
        atom.SetNumRadicalElectrons(0)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
        atom.SetIsAromatic(False)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in work.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    return [int(r) for r in Chem.CanonicalRankAtoms(work, breakTies=False, includeChirality=False,
                                                    includeIsotopes=False, includeAtomMaps=False)]


def _local_environments(graph: Chem.Mol) -> List[str]:
    """The element of each atom with the bond types and elements of its bonds, in a fixed order. A match of equal
    counts keeps each of them (:func:`Bond.Match` compares the bond types, the matcher the elements), so the atoms a
    match pairs have one environment, and the two graphs have the same atoms by environment."""
    out = []
    for atom in graph.GetAtoms():
        bonds = sorted(f"{int(b.GetBondType())}{b.GetOtherAtom(atom).GetAtomicNum()}" for b in atom.GetBonds())
        out.append(f"{atom.GetAtomicNum()}:{','.join(bonds)}")
    return out


def _connected_pair(target: Chem.Mol, query: Chem.Mol):
    """``same_graph_pair`` for a connected query: both graphs tagged with the local environment and the symmetry class
    of each atom, or None (the plain call) for a dummy atom, which the matcher lets stand for any element, a query atom
    that has a query of its own, or a graph whose classes cannot be computed. A pair whose atoms differ by environment
    holds no match (a bond order that the name reads otherwise): its tags are such that the matcher stops at once."""
    if any(a.GetAtomicNum() == 0 or a.HasQuery() for graph in (target, query) for a in graph.GetAtoms()):
        return None
    try:
        classes = [_symmetry_classes(graph) for graph in (target, query)]
    except Exception:  # noqa: BLE001 -- no classes: the plain call, as before
        return None
    environments = [_local_environments(graph) for graph in (target, query)]
    if sorted(environments[0]) != sorted(environments[1]):
        environments = [["target"] * target.GetNumAtoms(), ["query"] * query.GetNumAtoms()]
    tagged = []
    for graph, ranks, envs in zip((target, query), classes, environments):
        copy = Chem.Mol(graph)
        for atom, rank, env in zip(copy.GetAtoms(), ranks, envs):
            atom.SetIntProp(_SYMMETRY_CLASS, rank)
            atom.SetProp(_ENVIRONMENT, env)
        tagged.append(copy)
    params = Chem.SubstructMatchParameters()
    params.atomProperties = [_SYMMETRY_CLASS, _ENVIRONMENT]
    return tagged[0], tagged[1], params


def same_graph_matches(target: Chem.Mol, query: Chem.Mol, *, use_chirality: bool = False,
                       max_matches: int = 64) -> List[Tuple[int, ...]]:
    """``target.GetSubstructMatches(query, uniquify=False, maxMatches=max_matches, useChirality=use_chirality)``,
    the same matches in the same order, for a query and a target that are the same graph up to atom order
    (:func:`same_graph_pair` says why the call returns for a mixture of many components)."""
    pair = same_graph_pair(target, query)
    if pair is None:
        return list(target.GetSubstructMatches(query, uniquify=False, maxMatches=max_matches,
                                               useChirality=use_chirality))
    target, query, params = pair
    params.uniquify = False
    params.maxMatches = max_matches
    params.useChirality = use_chirality
    return list(target.GetSubstructMatches(query, params))
