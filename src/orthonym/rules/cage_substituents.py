"""Names for the non-ring branches of a von Baeyer cage (large-polycycle plan, Task 3a).

``polycyclic.get_polycyclic_substituents`` spells the branches of a tricyclo-and-larger cage on
the PIN path. A branch that holds a heteroatom (a halogen, a nitro group, a haloalkyl, an
alkylsulfanyl) is not an alkyl group, and naming it by its carbon count describes a different
molecule ('1-methyl...' for a bromomethyl branch, a bare parent for a bromo branch). The cage
collector therefore names such a branch through the shared substituent cascade, or declines the
cage, and never skips or approximates it.

Only a branch whose heteroatoms are all in groups that are NEVER the principal characteristic
group is built here: halogens, nitro, ether oxygen and thioether sulfur. A branch with a group
that can be expressed as a suffix (an acid, ester, amide, nitrile, aldehyde, ketone, alcohol,
hydroperoxide, amine or imine: classes 1 through 20 of "SENIORITY ORDER FOR CLASSES",
the Blue Book, Table 4.1) is left to the producers that choose the principal group. A prefix
form of such a group on a cage that has no senior group (a 'carbamoyl' branch on a parent hydride)
is not the preferred name, so for those branches the collector declines the cage, as before.

 (the Blue Book, section " HALOGEN COMPOUNDS"): "named in two ways: (1) by
substitutive nomenclature, using the prefixes 'bromo', 'chloro', 'fluoro', and 'iodo'...".
 (:25935, section " NITRO AND NITROSO COMPOUNDS"): "Compounds containing the -NO2
or -NO group are named by means of the prefixes 'nitro' and 'nitroso'...".
"""
import logging
from typing import Iterable

logger = logging.getLogger(__name__)

#: returned by:func:`cage_branch_token` for a branch with no honest name: the cage is declined
DECLINE = object()

_HALOGENS = frozenset({9, 17, 35, 53})


def _heavy_neighbors(atom):
    return [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]


def _is_prefix_only_heteroatom(atom) -> bool:
    """True when the non-carbon, non-hydrogen ``atom`` of a branch is in a group that can only
    be cited as a prefix: a halogen, a nitro group, an ether oxygen or a thioether sulfur."""
    number = atom.GetAtomicNum()
    heavy = _heavy_neighbors(atom)
    if number in _HALOGENS:
        return (atom.GetFormalCharge() == 0 and len(heavy) == 1
                and atom.GetBonds()[0].GetBondTypeAsDouble() == 1.0)
    if number in (8, 16):
        # O or S bonded to two carbons by single bonds, nothing else: R-O-R, R-S-R
        return (atom.GetFormalCharge() == 0 and atom.GetTotalNumHs() == 0 and len(heavy) == 2
                and all(n.GetAtomicNum() == 6 for n in heavy)
                and all(b.GetBondTypeAsDouble() == 1.0 for b in atom.GetBonds()))
    if number == 7:
        # nitro: N on one carbon and on two oxygens that carry nothing else
        carbons = [n for n in heavy if n.GetAtomicNum() == 6]
        oxygens = [n for n in heavy if n.GetAtomicNum() == 8 and len(_heavy_neighbors(n)) == 1]
        return (len(heavy) == 3 and len(carbons) == 1 and len(oxygens) == 2
                and atom.GetTotalNumHs() == 0)
    return False


def _is_acyloxy_or_peroxy(first, sub_set) -> bool:
    """True for an O-attached acyloxy (-O-C(=O)-) or peroxy (-O-O-) branch."""
    if first.GetSymbol() != 'O':
        return False
    for neighbor in first.GetNeighbors():
        if neighbor.GetIdx() not in sub_set:
            continue
        if neighbor.GetSymbol() == 'O':
            return True
        if neighbor.GetSymbol() == 'C' and any(
                bond.GetBondTypeAsDouble() == 2.0
                and bond.GetOtherAtom(neighbor).GetSymbol() in ('O', 'S', 'N')
                for bond in neighbor.GetBonds()):
            return True
    return False


def _cascade_token(mol, nbr_idx, sub_atoms):
    from ..assembly.substituent_enumerator import name_substituent
    try:
        # allow_mancude=False (the signature default, stated): the PIN-path cascade never
        # spells a fused ring as a von Baeyer prefix. At the best-effort tier the cascade
        # reads the ambient tier itself.
        token = name_substituent(mol, sub_atoms, nbr_idx, allow_mancude=False)
    except Exception:  # noqa: BLE001 -- a cascade decline is not a crash
        logger.debug("cage branch at atom %s: the cascade raised", nbr_idx, exc_info=True)
        return None
    if not token or token == 'substituent' or token.startswith('C'):
        return None
    return token


def cage_branch_token(mol, nbr_idx: int, sub_atoms: Iterable[int]):
    """The prefix of a heteroatom-bearing non-ring branch of a cage, from the shared cascade.

    ``nbr_idx`` is the branch atom bonded to the cage, ``sub_atoms`` every atom of the branch.
    Returns the token; ``None`` when the collector's own paths decide the branch (it holds no
    heteroatom, it is a plain alkoxy group, or it holds a group that can be the principal
    characteristic group, which those paths decline); or:data:`DECLINE` when the cascade cannot
    name a branch of prefix-only groups, so that the cage is declined.

    An O-attached acyloxy or peroxy branch (the macrocycle F3 route) is named through the
    cascade at the best-effort tier only: at the PIN tier a substituent-prefix form of an ester
    is not the PIN.
    """
    from ..metrics.provenance import best_effort_ctx
    sub_atoms = list(sub_atoms)
    sub_set = set(sub_atoms)
    first = mol.GetAtomWithIdx(nbr_idx)
    hetero = [mol.GetAtomWithIdx(i) for i in sub_atoms
              if mol.GetAtomWithIdx(i).GetAtomicNum() not in (1, 6)]
    if not hetero:
        return None
    if bool(best_effort_ctx.get()) and _is_acyloxy_or_peroxy(first, sub_set):
        return _cascade_token(mol, nbr_idx, sub_atoms)      # None falls through to the legacy paths
    if (first.GetSymbol() == 'O' and [a.GetIdx() for a in hetero] == [nbr_idx]
            and _is_prefix_only_heteroatom(first)):
        return None                        # -O-alkyl: the alkoxy table of the collector
    nitro_oxygens = {o.GetIdx() for a in hetero
                     if a.GetAtomicNum() == 7 and _is_prefix_only_heteroatom(a)
                     for o in _heavy_neighbors(a) if o.GetAtomicNum() == 8}
    if not all(_is_prefix_only_heteroatom(a) for a in hetero if a.GetIdx() not in nitro_oxygens):
        return None
    token = _cascade_token(mol, nbr_idx, sub_atoms)
    return DECLINE if token is None else token
