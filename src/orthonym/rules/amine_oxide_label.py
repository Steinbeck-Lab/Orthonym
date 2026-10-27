"""The label of the names an amine oxide is given: the oxide is the principal characteristic group.

 AMINE OXIDES, IMINE OXIDES, AND CHALCOGEN ANALOGUES (the Blue Book): amine oxides "are placed with
zwitterions in the order of compound classes (see. Thus, amine and imine oxides are named by method (1) and
all other amino groups, if present, are named as substituent groups by using the prefix 'amino'. Method (2) is
used when the oxide is on a nitrogen atom of a substituent group". The examples (:26630-26636):
'3,5-bis(dimethylamino)-N,N-dimethylpentan-1-amine N-oxide (PIN) (not N1,N1,N3,N3,N5,N5-hexamethylpentane-1,3,5-
triamine N1-oxide; the N-oxide is classified as a zwitterion and it is the preferred characteristic group)' and
'2-(3-carbamoyl-5-carboxyphenyl)ethan-1-amine N-oxide (PIN)' against '3-(2-aminoethyl)-5-carbamoylbenzoic acid
N3-oxide'. The zwitterions (class 5,,:18158) are senior to the acids (class 7), the amides and
every other class but the radicals, the radical ions and the anions (classes 1 to 4).

So a name that cites the oxide beside another amine as one suffix ('...-triamine N-oxide'), that makes another group
the suffix ('...benzoic acid N-oxide'), or that cites the oxide of a neutral molecule as a prefix of method (2)
('2-(oxo-λ5-azanyl)ethyl') is a valid name and not the PIN: its label is lowered here, at the producers that build
such a name, and the name is kept for the wider tiers.
"""
from __future__ import annotations

from typing import Optional

#: the principal characteristic groups that are amines class 19); the oxide of one of them is the suffix
_AMINES = frozenset({'primary_amine', 'secondary_amine', 'tertiary_amine', 'aromatic_amine'})


def oxide_is_the_only_amine_suffix(base_mol) -> bool:
    """True when the amine ``base_mol`` (the amine oxide without its oxide atom) has an amine as its principal
    characteristic group and holds one amine group only: the oxidised nitrogen. Then 'amine N-oxide' names the
    molecule on the oxide as its one suffix method (1))."""
    try:
        from ..perception.functional_groups import detect_functional_groups
        from .seniority import get_principal_group
        group, matches = get_principal_group(base_mol, detect_functional_groups(base_mol))
    except Exception:  # noqa: BLE001 - an unreadable molecule is not shown to be the sole amine
        return False
    return group in _AMINES and len(matches) == 1


def record_unless_sole_amine_suffix(name: Optional[str], base_mol) -> Optional[str]:
    """``name`` of an amine oxide, recorded as not the PIN unless the oxide is the one amine suffix of it."""
    if name and not oxide_is_the_only_amine_suffix(base_mol):
        from ..metrics.provenance import record_non_pin_label
        record_non_pin_label(name, rule="P-62.5",
                             detail="the N-oxide is a zwitterion and the preferred characteristic group: the other "
                                    "amino groups are cited as prefixes, not as suffixes of one parent")
    return name


def record_neutral_method_two_prefix(mol, prefix: Optional[str]) -> Optional[str]:
    """``prefix`` (an amine oxide cited as a substituent group of method (2)) recorded as not the PIN when ``mol``
    has no radical, no anion and no cation beside the oxide (the molecule drawn with the oxide as the zwitterion N+-O-):
    in a neutral molecule the oxide is the principal characteristic group (class 5 of P-41), and its name is built on it
    by method (1). A substituent group that is a radical, '2-(oxo-λ5-azanyl)ethyl (PIN)' (:26642), keeps its name."""
    if not prefix:
        return prefix
    ionic, drawn_as_zwitterion = False, False
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons():
            ionic = True
        elif atom.GetFormalCharge() and not _is_oxide_atom(atom):
            ionic = True
        elif atom.GetFormalCharge():
            drawn_as_zwitterion = True
    if drawn_as_zwitterion and not ionic:
        from ..metrics.provenance import record_non_pin_label
        record_non_pin_label(prefix, rule="P-62.5",
                             detail="method (2) names an oxide on a nitrogen atom of a substituent group; in a neutral "
                                    "molecule the N-oxide is the principal characteristic group")
    return prefix


def _is_oxide_atom(atom) -> bool:
    """A nitrogen cation with a terminal oxide anion on it, or that oxide anion."""
    if atom.GetSymbol() == 'N' and atom.GetFormalCharge() == 1:
        return any(n.GetSymbol() == 'O' and n.GetFormalCharge() == -1 and n.GetDegree() == 1
                   for n in atom.GetNeighbors())
    if atom.GetSymbol() == 'O' and atom.GetFormalCharge() == -1 and atom.GetDegree() == 1:
        return any(n.GetSymbol() == 'N' and n.GetFormalCharge() == 1 for n in atom.GetNeighbors())
    return False
