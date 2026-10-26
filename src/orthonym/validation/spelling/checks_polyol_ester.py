"""Spelling check of the esters of a single polyol: functional class multiplicative nomenclature, not acyloxy prefixes.

 "Polyesters formed from a single 'alcoholic' component" (the Blue Book ff.): "Esters derived
from a single 'polyhydroxylic' component are named by placing the name of the 'polyhydroxylic' component denoted
by a multivalent organyl group... in front of the name(s) of the acid component". (:31819): "When
anions are identical functional class multiplicative nomenclature is used" ('propane-1,3-diyl bis(chloroacetate)
(PIN)',:31825). "When anions are different, two methods are used:" (:31831): "Method (1) generates
preferred IUPAC names but names formed by using method (2) are acceptable in general nomenclature"; method (2)
cites "all other ester groups... as prefixes in the name of the organyl group" ('(1) propane-1,2,3-triyl
2-acetate 1-hexadecanoate 3-[(9Z)-octadec-9-enoate] (PIN) (2) 2-(acetyloxy)-3-(hexadecanoyloxy)propyl
(9Z)-octadec-9-enoate',:31846; '(1) 1,4-phenylene acetate dichloroacetate (PIN) (2) 4-(acetyloxy)phenyl
dichloroacetate').

The rule is decided here only where it is the rule of the molecule, and no further:

* the ester groups are the principal characteristic groups (no group senior to the esters, "SENIORITY ORDER FOR
  CLASSES", Table 4.1,:18162: an ion, a radical, an acid, an anhydride);
* the molecule has **one** alcoholic component, which carries two or more ester groups: once the acyl-oxygen bonds of
  the ester groups are cut, one fragment holds the oxygen of every ester group of the molecule. Otherwise the
  polyester has several alcoholic components and is named by (:31860 ff.), where the book's PINs cite
  acyloxy prefixes ('2-(acetyloxy)ethyl methyl butanedioate (PIN)', 'phenyl 3-(benzoyloxy)benzoate (PIN)');
* the polyol is a multivalent organyl group of one parent: the carbons that carry the ester oxygens are all ring
  atoms of one ring, or all acyclic carbons on one chain. Oxygens on a ring and on its side chain, or on two rings,
  have no multivalent group of one parent ('2-[2-(acetyloxy)ethyl]phenyl acetate (PIN)', '6-[4-(acetyloxy)phenyl]
  pyridin-3-yl acetate (PIN)',:31878-31897).

No naming code, no OPSIN: the structure and the words of the name.
"""
from __future__ import annotations

import re

from rdkit import Chem

from ..pin_spelling import ABSTAIN, PASS, SpellingFailure, register
from .models import P10

#: a carboxylic ester group R-C(=O)-O-C (R a carbon or the hydrogen of a formate); a carbonate or a carbamate
#: has no carbon on its carbonyl carbon and is not one
_ESTER = Chem.MolFromSmarts("[CX3;H1,$([CX3][#6])](=[OX1])[OX2][#6]")

#: groups senior to the esters, Table 4.1): an ion (the nitro group and the N-oxide are no ion of the class), a
#: carboxylic, sulfur, phosphorus or boron acid group, an anhydride
_SENIOR = tuple(Chem.MolFromSmarts(s) for s in (
    "[+,-;!$([N+]~[O-]);!$([O-]~[N+])]",
    "[CX3,SX4,SX3,PX4,PX3,BX3,AsX4,AsX3](=[O,S])[OX2H1,SX2H1]",
    "[CX3](=O)[OX2][CX3,SX4,PX4]=O"))

#: an acyloxy prefix as the reader of the name sees it: 'acetyloxy', 'hexadecanoyloxy', '[(9Z)-octadec-9-enoyl]oxy',
#: '(2-methylpropanoyl)oxy', 'benzoyloxy', 'formyloxy'
_ACYLOXY = re.compile(r"(?:oyl|acetyl|formyl)[\]\)\}]*oxy")


def _words(name):
    """The words of ``name`` split at the spaces outside any enclosing mark."""
    out, depth, cur = [], 0, []
    for ch in name:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == " " and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [w for w in out if w]


def _on_one_parent(mol, carbons) -> bool:
    """True when the carbons that carry the ester oxygens are the ring atoms of one ring, or acyclic carbons that
    lie on one chain: the minimal subtree of the acyclic carbon skeleton that joins them is a path."""
    carbons = set(carbons)
    in_ring = {a for a in carbons if mol.GetAtomWithIdx(a).IsInRing()}
    if in_ring:
        return in_ring == carbons and any(carbons <= set(ring) for ring in mol.GetRingInfo().AtomRings())
    skeleton = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and not a.IsInRing()}
    nbrs = {a: [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() in skeleton]
            for a in skeleton}
    tree = set()
    stack = [next(iter(carbons))]
    while stack:                                # the component of the skeleton that holds the carbons
        a = stack.pop()
        if a in tree:
            continue
        tree.add(a)
        stack.extend(nbrs[a])
    if not carbons <= tree:
        return False
    leaves = [a for a in tree if len([n for n in nbrs[a] if n in tree]) <= 1 and a not in carbons]
    while leaves:                               # prune the branches that carry no ester oxygen
        a = leaves.pop()
        tree.discard(a)
        for n in nbrs[a]:
            if n in tree and n not in carbons and len([m for m in nbrs[n] if m in tree]) <= 1:
                leaves.append(n)
    return all(len([n for n in nbrs[a] if n in tree]) <= 2 for a in tree)


def single_polyol_ester(mol):
    """``(state, count)``: how the ester groups of ``mol`` sit on alcoholic components.

    ``"none"`` -- no ester group, or each alcoholic fragment holds one (monoesters, nothing to decide);
    ``"single"`` -- one alcoholic component of one parent carries every ester group (``count`` >= 2);
    ``"other"`` -- an alcoholic fragment holds two or more ester groups and the molecule is not that case
    (several alcoholic components, a polyol of more than one parent): the rule of the polyester is another one;
    ``"ring"`` -- an ester linkage lies in a ring (a lactone): not read."""
    matches = mol.GetSubstructMatches(_ESTER)
    if not matches:
        return "none", 0
    bonds = []
    for acyl, _o_dbl, o_ester, alcohol in matches:
        if mol.GetBondBetweenAtoms(acyl, o_ester).IsInRing():
            return "ring", 0
        bonds.append((acyl, o_ester, alcohol))
    rw = Chem.RWMol(mol)
    for acyl, o_ester, _alcohol in bonds:
        rw.RemoveBond(acyl, o_ester)
    owner = {a: k for k, frag in enumerate(Chem.GetMolFrags(rw)) for a in frag}
    per_fragment = {}
    for _acyl, o_ester, alcohol in bonds:
        per_fragment.setdefault(owner[o_ester], []).append(alcohol)
    if all(len(v) == 1 for v in per_fragment.values()):
        return "none", 0
    if len(per_fragment) == 1:
        carbons = next(iter(per_fragment.values()))
        if mol.GetAtomWithIdx(carbons[0]).GetAtomicNum() == 6 and _on_one_parent(mol, carbons):
            return "single", len(carbons)
    return "other", 0


@register("P-65.6.3.3.3")
def polyol_esters_by_acyloxy_prefixes(mol, name):
    """Esters of a single polyol named by prefixes,:31819 ff.): the PIN names the polyol as a
    multivalent organyl group before the anion names (functional class multiplicative nomenclature, for
    identical anions, method (1) of for different anions), and the other ester groups are no acyloxy
    prefixes of the alkyl word. PASS when the molecule has no single polyol ester or the name cites no acyloxy prefix
    before its last word; ABSTAIN on a name of the classes, on a polyester of another kind (see
    :func:`single_polyol_ester`), on a molecule with a group senior to the esters, and on a name whose last word is not
    an ester anion."""
    if P10.search(name):
        return ABSTAIN
    state, count = single_polyol_ester(mol)
    if state == "none":
        return PASS
    if state != "single":
        return ABSTAIN
    if (any(mol.HasSubstructMatch(patt) for patt in _SENIOR)
            or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())):
        return ABSTAIN
    words = _words(name)
    if len(words) < 2 or not words[-1].rstrip("])}").endswith("ate"):
        return ABSTAIN
    cited = [w for w in words[:-1] if _ACYLOXY.search(w)]
    if not cited:
        return PASS
    return SpellingFailure(
        "P-65.6.3.3.3",
        f"{count} ester groups of one polyol: the alkyl word cites the others as acyloxy prefixes ('{cited[0]}'); the "
        f"PIN is the functional class multiplicative name (P-65.6.3.3.3.1, P-65.6.3.3.3.2 method (1))")
