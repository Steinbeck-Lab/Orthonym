"""The organyl words of a diester whose two organyl groups differ and whose two acid positions are not equivalent.

 "Fully esterified acids derived from a single acid" (the Blue Book): the organyl groups are cited
as separate words before the anion, in alphanumerical order, with locants where needed: '1,1-diethyl 3-methyl
butane-1,1,3-tricarboxylate (PIN)' (:31783). Where the two acid positions are equivalent no locant is needed: 'ethyl
methyl benzene-1,3-dicarboxylate (PIN)' (:5068), 'ethyl methyl butanedioate (PIN)' (:31779). Without the locants
'ethyl methyl 4-methylbenzene-1,3-dicarboxylate' names two molecules.

The anion's own name gives the two acid locants; which organyl group takes which locant is read back: both assignments
are built and OPSIN keeps the one whose structure is the input (the other names the other isomer). With no reading
(no jar) the diester is declined.
"""
import re
from typing import Callable, List, Optional, Sequence

from rdkit import Chem

_LOCANTS = re.compile(r"-(\d+[a-z]?'*(?:,\d+[a-z]?'*)+)-(?:di|tri|tetra)(?:carboxylate|oate)$")


def positions_equivalent(diacid, carbonyls: Sequence[int]) -> bool:
    """True when the two acid carbons ``carbonyls`` of ``diacid`` are one symmetry class."""
    ranks = list(Chem.CanonicalRankAtoms(diacid, breakTies=False))
    return ranks[carbonyls[0]] == ranks[carbonyls[1]]


def located_organyl_words(mol, organyls: List[str], ate: str, sort_key: Callable[[str], object],
                          read: Callable[[str], Optional[str]], chain_length: int = 0) -> Optional[str]:
    """'1-ethyl 3-methyl 4-methylbenzene-1,3-dicarboxylate', '1,1-diethyl 3-methyl butane-1,1,3-tricarboxylate' for the
    organyl groups ``organyls`` (one per ester) of ``mol``, or None when the anion ``ate`` gives no locant per ester or
    not exactly one assignment of the locants reads back to ``mol``. A chain '-dioate' cites no terminal locant
    , the Blue Book): its acid positions are 1 and ``chain_length``."""
    from itertools import permutations

    from ..assembly.naming_utils import enclose_if_compound, multiplied_component, render_locants
    m = _LOCANTS.search(ate)
    locants = m.group(1).split(",") if m else (["1", str(chain_length)] if chain_length > 1 and ate.endswith("dioate")
                                                 else [])
    if len(locants) != len(organyls) or not 2 <= len(organyls) <= 4 or _has_unlocanted_prefix(ate):
        return None
    target = Chem.MolToSmiles(mol, isomericSmiles=False)
    names = set()
    for perm in set(permutations(locants)):
        held = {}
        for org, loc in zip(organyls, perm):
            held.setdefault(org, []).append(loc)
        words = []
        for org in sorted(held, key=sort_key):
            locs = sorted(held[org], key=lambda x: (int(re.match(r"\d+", x).group(0)), x))
            # a compound organyl behind its locant is enclosed, the Blue Book): '1-(2-methylbutyl)'
            marked = enclose_if_compound(org)
            words.append(f"{render_locants(locs)}-" + (multiplied_component(len(locs), org, marked) if len(locs) > 1
                                                       else marked))
        names.add(" ".join(words) + f" {ate}")
    found = []
    for name in sorted(names):
        smiles = read(name)
        back = Chem.MolFromSmiles(smiles) if smiles else None
        if back is not None and Chem.MolToSmiles(back, isomericSmiles=False) == target:
            found.append(name)
    return found[0] if len(found) == 1 else None


def _has_unlocanted_prefix(ate: str) -> bool:
    """True when a prefix of the anion has no locant, or the anion cannot be read: the acid dropped it because its
    two acid positions were alike ('chlorobutanedioic acid (PIN)',, the Blue Book); the
    organyl locants tell those positions apart, so the prefix needs its locant again ('4-ethyl 1-methyl
    2-methylbutanedioate'), which this does not build."""
    from ..validation.spelling.models import read_units
    parsed = read_units(ate)
    if parsed is None:
        return True
    return any(not x.locs for p in parsed[2] for x in p.prefixes)
