"""A radical centre that is also an ionic centre: the one-atom parent of a radical ion.

A molecule whose only radical centre is an atom that carries a formal charge (a boranuide with its hydrogens
removed, an iodanuide, a silylium) is a radical ion of the parent hydride of that one atom. The floor builds nothing for
it (its charge machinery and its free valences are not combined on one atom), and the radical offer
(``assembly/radical_offer.py``) leaves it to this candidate.

``### **** RADICAL IONS DERIVED FROM PARENT HYDRIDES`` (the Blue Book): "The order of seniority, radicals >
anions > cations, is reflected in a preferred IUPAC name. Suffixes assigned to anionic and/or cationic centers are placed
first after the name of the parent structure..., followed by suffixes attributed to radical centers." and (:43422) "A
radical ion derived formally by the removal of one or more hydrogen atoms from a single skeleton atom... of an ionic or
zwitterionic parent hydride is named by adding to its name the suffixes 'yl' or 'ylidene' with appropriate multiplying
prefixes before 'yl' or 'ylidene', with elision of the final letter 'e' of the name of the ionic parent hydride."
The examples are '(CH3)3B•- trimethylboranuidyl (PIN)' (:43430), 'H2C•+ methyliumyl (PIN)' (:43443) and, under
```` (:43463), '(N,N-diethylethanaminiumyl)boranuidyl (PIN)', a boranuide whose substituent is an ammonium. The
suffixes of a radical of several hydrogens removed from one atom are '-ylidene' and '-ylidyne' (``### ****
Specific method and retained names``:40474; ```` General method: '-ylidene' or '-diyl' and '-ylidyne' or
'-triyl'), which this candidate cites first, '-diyl' and '-triyl' after them.

The one home of the ionic parent hydride of an atom is ``rules.radicals.ionic_parent_hydride`` (the producer of the PIN
tiers, ``rules.radicals._name_mononuclear_radical_ion``, names the centres of ``_PIN_RADICAL_ION_CENTRES`` with it and leaves the
empty-orbital cations of alone, ``radical_ion_is_empty_orbital_cation``); this module names the same ions at the
offer, for the centres and substituents the producer does not take (a boron holding three radical electrons, an iodine, a
silylium, substituents that are not carbon groups: every neighbour of the atom is a prefix named by the floor for its
branch). The name is ``<prefixes><stem><ionic suffix><radical suffix>``. The row is ``systematic_verified``: it ships only
when OPSIN ``-r`` reads it back to the input's full InChIKey and radical graph (``assembly/radical_offer.py``), which a
spelling of the ionic suffix that OPSIN does not place the way this one does never passes.
"""
from typing import Dict, Iterator, List, Optional

from rdkit import Chem

#: the halogens, whose ionic parent hydride is cited with its bonding number as a second spelling
_HALOGENS = frozenset({17, 35, 53})
#: the suffix of the radical that is multiplied ('-diyl', '-triyl') for the number of hydrogens removed from the atom
_MULTIPLIED_SUFFIX = {2: "diyl", 3: "triyl"}


def _branches(mol, centre: int):
    """``[(neighbour, atoms of its branch, bond order)]`` of the heavy neighbours of ``centre``, or None when two of
    them are joined without ``centre`` (the atom is in a ring)."""
    found, seen = [], {centre}
    for nbr in mol.GetAtomWithIdx(centre).GetNeighbors():
        if nbr.GetAtomicNum() < 2:
            continue
        if nbr.GetIdx() in seen:
            return None
        atoms, stack = {nbr.GetIdx()}, [nbr.GetIdx()]
        while stack:
            for onward in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
                j = onward.GetIdx()
                if j == centre:
                    continue
                if j not in atoms:
                    atoms.add(j)
                    stack.append(j)
        if atoms & seen:
            return None
        seen |= atoms
        order = round(mol.GetBondBetweenAtoms(centre, nbr.GetIdx()).GetBondTypeAsDouble())
        found.append((nbr.GetIdx(), frozenset(atoms), order))
    return found


def _prefix_block(names: List[str]) -> str:
    """The substituent prefixes of a one-atom parent in the order of, identical ones multiplied, each
    enclosed when it is compound; a one-atom parent has no locant."""
    from .naming_utils import alpha_sort_key, enclose_if_compound, multiplied_component
    counts: Dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    parts = []
    for name in sorted(counts, key=alpha_sort_key):
        parts.append(multiplied_component(counts[name], name, enclose_if_compound(name)))
    return "".join(parts)


def one_atom_prefix_block(mol, centre: int, acyloxy_leaf: bool = True) -> Optional[str]:
    """The prefixes of the one-atom parent ``centre`` (every heavy neighbour is a prefix, named by the floor for its
    branch, a double or triple bond cited as '-ylidene' or '-ylidyne'), in the order of; '' for no neighbour;
    None when the atom is in a ring or a branch has no name."""
    from .universal_substituent import name_universal_substituent_prefix
    branches = None if mol.GetAtomWithIdx(centre).IsInRing() else _branches(mol, centre)
    if branches is None:
        return None
    names = []
    for neighbour, atoms, order in branches:
        named = name_universal_substituent_prefix(mol, atoms, neighbour, bond_order=order, acyloxy_leaf=acyloxy_leaf)
        if not named:
            return None
        names.append(named)
    return _prefix_block(names)


def radical_ion_names(mol, acyloxy_leaf: bool = True) -> Iterator[str]:
    """The names of ``mol`` when its one radical centre is a charged atom outside every ring, or nothing."""
    from ..rules.radicals import (
        _RADICAL_SUFFIX_BY_NE, ionic_parent_hydride, radical_ion_is_empty_orbital_cation)
    centres = [a for a in mol.GetAtoms() if a.GetNumRadicalElectrons()]
    if len(centres) != 1 or len(Chem.GetMolFrags(mol)) != 1 or any(a.GetIsotope() for a in mol.GetAtoms()):
        return
    atom = centres[0]
    radicals = atom.GetNumRadicalElectrons()
    if (radicals not in _RADICAL_SUFFIX_BY_NE or not atom.GetFormalCharge()
            or radical_ion_is_empty_orbital_cation(atom)):
        return
    bonding = sum(round(b.GetBondTypeAsDouble()) for b in atom.GetBonds()) + atom.GetTotalNumHs() + radicals
    parent = ionic_parent_hydride(atom, bonding)
    prefixes = one_atom_prefix_block(mol, atom.GetIdx(), acyloxy_leaf) if parent else None
    if prefixes is None:
        return
    # the final 'e' of an 'ide' or 'uide' parent is elided before 'y',:43422), kept before '-diyl' and '-triyl'
    elided = parent[:-1] if parent.endswith("e") else parent
    forms = [f"{elided}{_RADICAL_SUFFIX_BY_NE[radicals]}"]
    if radicals > 1:
        forms.append(f"{parent}{_MULTIPLIED_SUFFIX[radicals]}")
    for form in forms:
        yield f"{prefixes}{form}"
    if atom.GetAtomicNum() in _HALOGENS and parent.endswith("uide"):
        # a halogen: the bonding number of the atom is cited, the Blue Book) after the prefixes, which
        # OPSIN 2.9.0 needs to place the free valence on 'iodanuide' ('pentyl-λ1-iodanuidyl')
        standard = Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())
        for form in forms:
            yield f"{prefixes}-λ{standard}-{form}" if prefixes else f"λ{standard}-{form}"
