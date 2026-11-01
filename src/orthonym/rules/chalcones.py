"""The retained name 'chalcone' with the substituents of its two benzene rings.

 KETONES, "Retained names", (the Blue Book): "The name 'chalcone' is the only
retained name as a preferred IUPAC name and is limited to ring substitution only by characteristic groups lower
than 'ketone'. Chalcone refers only to the trans- or (E)- stereoisomer." Its examples: 'chalcone (PIN)
(2E)-1,3-diphenylprop-2-en-1-one' (:28299) and '2',4'-dihydroxy-3,3'-dimethoxychalcone (PIN)
(2E)-1-(2,4-dihydroxy-3-methoxyphenyl)-3-(3-methoxyphenyl)prop-2-en-1-one' (:28303). The ring of the benzoyl half
(on C=O) takes the primed locants, the ring of the styryl half (on C-3) the unprimed ones, as in the second
example: 3 and 3' are methoxy, 2' and 4' hydroxy, 3' bears the methoxy next to the benzoyl carbon.

The structure test is ``validation.spelling.checks_chalcone.chalcone_skeleton``, the one the spelling check uses
to refuse a chalcone built on the systematic parent; ``name_substituted_chalcone`` is the producer.

The substituents named here are whole characteristic groups (glossary "Characteristic group", the Blue Book:
"A single heteroatom, such as -Cl, and =O; a heteroatom bearing... hydrogen atoms or other heteroatoms, such as
-NH2, -OH,..."): halo, nitro, nitroso, azido and isocyanato (Table 5.1, prefixes only,:24980), hydroxy, sulfanyl,
amino and an amino or alkoxy group whose carbon part is a plain saturated alkyl chain. A hydrocarbyl group (methyl)
is no characteristic group, and a group that carries a ring, another heteroatom or a substituted chain (morpholinyl,
2-chloroethoxy, trimethylsilyloxy) is more than one: the producer declines and the molecule keeps its systematic name.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from rdkit import Chem


def _digit(locant: str) -> Tuple[int, int]:
    """Sort key of a locant such as ``3`` or ``3'``: the number, then the primes."""
    return int(locant.rstrip("'")), len(locant) - len(locant.rstrip("'"))


def _ring_in_order(mol, ring, attach: int, reverse: bool) -> List[int]:
    """The ring atoms in the order of the numbering that starts at ``attach`` (position 1) and runs one way
    round the ring."""
    members = set(ring)
    order = [attach]
    prev, cur = None, attach
    while len(order) < len(ring):
        nxt = [n.GetIdx() for n in mol.GetAtomWithIdx(cur).GetNeighbors()
               if n.GetIdx() in members and n.GetIdx() != prev and n.GetIdx() not in order]
        nxt.sort(reverse=reverse)
        pick = nxt[0]
        order.append(pick)
        prev, cur = cur, pick
    return order


def _plain_alkyl(mol, atoms: set) -> bool:
    """True when ``atoms`` are saturated acyclic carbons with no charge, radical or isotope label (an alkyl part)."""
    return bool(atoms) and all(
        mol.GetAtomWithIdx(i).GetSymbol() == 'C' and not mol.GetAtomWithIdx(i).IsInRing()
        and not mol.GetAtomWithIdx(i).GetIsotope() and mol.GetAtomWithIdx(i).GetFormalCharge() == 0
        and not mol.GetAtomWithIdx(i).GetNumRadicalElectrons()
        and all(b.GetBondTypeAsDouble() == 1.0 for b in mol.GetAtomWithIdx(i).GetBonds()) for i in atoms)


def _is_characteristic_group(mol, atoms: set, root: int) -> bool:
    """True when the substituent ``atoms`` (bonded to the ring at ``root``) are one characteristic group of the
    chalcone rule (see the module docstring): halo; nitro; nitroso; azido; isocyanato; hydroxy, sulfanyl, amino;
    amino or alkoxy with a plain alkyl part."""
    r = mol.GetAtomWithIdx(root)
    if r.GetIsotope() or r.GetNumRadicalElectrons():
        return False
    sym = r.GetSymbol()
    if sym in ('F', 'Cl', 'Br', 'I'):
        return atoms == {root}
    rest = atoms - {root}
    if sym in ('O', 'S') and r.GetFormalCharge() == 0:
        return not rest or (_plain_alkyl(mol, rest) and r.GetDegree() == 2)   # alkoxy, alkylsulfanyl
    if sym != 'N':
        return False
    heavy = sorted(mol.GetAtomWithIdx(i).GetSymbol() for i in rest)
    if heavy in (['O', 'O'], ['O']) and all(mol.GetAtomWithIdx(i).GetDegree() == 1 for i in rest):
        return True                                   # nitro, nitroso
    if heavy == ['N', 'N'] or heavy == ['C', 'O']:
        chain = [mol.GetAtomWithIdx(i) for i in rest]
        return all(x.GetDegree() <= 2 for x in chain)  # azido, isocyanato
    return r.GetFormalCharge() == 0 and (not rest or _plain_alkyl(mol, rest))


def _in_nitro(atom) -> bool:
    """True when the charged ``atom`` is the N+ or the O- of a nitro group drawn charge-separated (-[N+](=O)[O-])."""
    n = atom if atom.GetSymbol() == 'N' else next((x for x in atom.GetNeighbors() if x.GetSymbol() == 'N'), None)
    if n is None or n.GetFormalCharge() != 1:
        return False
    oxygens = [x for x in n.GetNeighbors() if x.GetSymbol() == 'O' and x.GetDegree() == 1]
    return len(oxygens) == 2 and sorted(o.GetFormalCharge() for o in oxygens) == [-1, 0]


def _in_azido(atom) -> bool:
    """True when the charged ``atom`` is the N+ or the terminal N- of an azido group (-N=[N+]=[N-])."""
    if atom.GetSymbol() != 'N':
        return False
    mid = atom if atom.GetFormalCharge() == 1 else next(iter(atom.GetNeighbors()), None)
    if mid is None or mid.GetSymbol() != 'N' or mid.GetFormalCharge() != 1 or mid.GetDegree() != 2:
        return False
    return sorted(x.GetFormalCharge() for x in mid.GetNeighbors() if x.GetSymbol() == 'N') == [-1, 0]


def _substituents(mol, ring, attach: int) -> Optional[Dict[int, Tuple[str, int]]]:
    """``{ring atom: (prefix name, attachment atom)}`` of the ring's substituents, or None when one is not named
    here (not rooted at O, S or N as an amine, ether or thiol, or the namer declines it)."""
    from ..assembly.substituent_enumerator import name_substituent
    out: Dict[int, Tuple[str, int]] = {}
    members = set(ring)
    for a in ring:
        if a == attach:
            continue
        atom = mol.GetAtomWithIdx(a)
        if atom.GetIsotope():
            return None
        for nb in atom.GetNeighbors():
            if nb.GetIdx() in members:
                continue
            if mol.GetBondBetweenAtoms(a, nb.GetIdx()).GetBondTypeAsDouble() != 1.0:
                return None
            # the atoms of the substituent: everything reached from the root without crossing the ring
            seen, stack = {nb.GetIdx()}, [nb.GetIdx()]
            while stack:
                for n2 in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
                    if n2.GetIdx() not in seen and n2.GetIdx() not in members:
                        seen.add(n2.GetIdx())
                        stack.append(n2.GetIdx())
            if not _is_characteristic_group(mol, seen, nb.GetIdx()):
                return None
            name = name_substituent(mol, seen, nb.GetIdx())
            if not name or name == 'substituent':
                return None
            out[a] = (name, nb.GetIdx())
    return out


def name_substituted_chalcone(mol) -> Optional[str]:
    """The name of an (E)-chalcone with ring substituents of the classes named above, '<prefixes>chalcone',
    or None (not this class, no ring substituent -- the retained table names the bare chalcone -- or a
    substituent this producer does not name). The one stereogenic unit is the (E) double bond, which the
    name 'chalcone' implies: a molecule with any other stereo element is declined."""
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    from ..validation.spelling.checks_chalcone import chalcone_skeleton
    skeleton = chalcone_skeleton(mol)
    if skeleton is None:
        return None
    ring_a_c, c1, o1, c2, c3, ring_b_c, ring_a, ring_b = skeleton
    if any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED or (a.GetFormalCharge() and not (_in_nitro(a) or _in_azido(a)))
           or a.GetIsotope() or a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return None
    if any(b.GetStereo() not in (Chem.BondStereo.STEREONONE,)
           for b in mol.GetBonds() if {b.GetBeginAtomIdx(), b.GetEndAtomIdx()} != {c2, c3}):
        return None
    from ..assembly.naming_utils import alpha_sort_key, format_substituent_prefix
    best = None
    subs_a, subs_b = _substituents(mol, ring_a, ring_a_c), _substituents(mol, ring_b, ring_b_c)
    if subs_a is None or subs_b is None or not (subs_a or subs_b):
        return None
    options = []
    for ring, attach, subs, primed in ((ring_a, ring_a_c, subs_a, "'"), (ring_b, ring_b_c, subs_b, "")):
        per_direction = []
        for reverse in (False, True):
            order = _ring_in_order(mol, ring, attach, reverse)
            locants = {atom: f"{pos + 1}{primed}" for pos, atom in enumerate(order)}
            per_direction.append([(subs[a][0], locants[a]) for a in subs])
        options.append(per_direction)
    for dir_a in options[0]:
        for dir_b in options[1]:
            items = dir_a + dir_b
            # lowest locants to all the prefixes together (unprimed before primed at one number), then to the
            # prefix cited first in alphanumerical order
            key = (sorted(_digit(loc) for _n, loc in items),
                   [_digit(loc) for _n, loc in sorted(items, key=lambda it: (alpha_sort_key(it[0]), _digit(it[1])))])
            if best is None or key < best[0]:
                best = (key, items)
    groups: Dict[str, List[str]] = {}
    for name, loc in best[1]:
        groups.setdefault(name, []).append(loc)
    parts = []
    for name in sorted(groups, key=alpha_sort_key):
        locs = sorted(groups[name], key=_digit)
        parts.append(format_substituent_prefix(name, locs, len(locs)))
    return "-".join(parts) + "chalcone"
