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
  atoms of one ring system (a steroid ring system is one parent, '5α-cholestane-3β,6α-diyl diacetate',:31979), or
  all acyclic carbons on one chain. Oxygens on a ring and on its side chain, or on two rings, have no multivalent
  group of one parent ('2-[2-(acetyloxy)ethyl]phenyl acetate (PIN)', '6-[4-(acetyloxy)phenyl]pyridin-3-yl acetate
  (PIN)',:31878-31897) -- unless the polyol is symmetrical (below).
* the polyol of two or more parents is a multiplicative organyl group when its arms are identical and the anions are
  too (or the polyol is symmetrical, so that no anion needs a locant of the group to say where it stands): one unit of
  the polyol (a ring system, a ring assembly, a chain of carbon atoms, a hetero atom) is the central group, and what
  hangs on it toward each ester oxygen is the same group in every branch. (:31815) "a multivalent
  organyl group (alkyl, aryl, etc.)", "Preferred IUPAC multiplicative names" (:23178) conditions (1)-(3)
  (identical linking bonds, symmetrically substituted arms, identical locants), (:5208) "Substitution,
  expressed by prefixes or implied, is allowed on simple multiplicative groups, symmetrical or unsymmetrical" (the
  central group may carry a chloro or methyl group of its own: '(4-chloro-1,3-phenylene)bis(methylene)'),
  "Introduction" (2) (:5141) "A subset of functional class nomenclature in which the identical structural units are
  linked by a di- or polyvalent nonacidic residue (this subset is only used to name esters...)":
  '1,2-phenylenedi(propan-3,1-yl) diacetate [PIN; see (2) and ]' (:31829),
  '2-[3-(formyloxy)propyl]cyclohexane-1,1-diyldi(propane-3,1-diyl) diacetate (PIN)' (:31878: three arms of one
  constitution on a ring, two acetates and a formate, which no locant of a multiplicative group places). When the arms
  or the anions differ so the group cannot say where each anion stands, the polyester is named by substitutive
  nomenclature, (:31856): '2-{2-[(acetyloxy)methyl]phenyl}ethyl acetate (PIN)' (:31899).

No naming code, no OPSIN: the structure and the words of the name.
"""
from __future__ import annotations

import re

from rdkit import Chem

from ..pin_spelling import ABSTAIN, PASS, SpellingFailure, register
from .models import P10

#: an ester group (acid centre, =O, ester O, alcohol C): a carboxylic ester R-C(=O)-O-C (R a carbon or the hydrogen of
#: a formate; a carbonate or a carbamate has no carbon on its carbonyl carbon and is not one), or the ester of a
#: sulfonic or sulfinic acid R-S(=O)x-O-C. names the esters of both alike,:31696: "an ester
#: with the general structure R-CO-O-R′ or R-S(O)x-O-R′";,:31771: "This rule applies equally to
#: carboxylic, sulfonic, sulfinic, etc. acids"); a sulfonate matches once per =O, read once per ester O. The ester of
#: the carboxylic acid on a ring nitrogen ('piperidine-1-carboxylate') is a carboxylic ester too (:24910
#: 'tetraazane-1-carboxylic acid (PIN) [... the carboxylic acid is senior to the carbonic acid derivative]')
_ESTER = Chem.MolFromSmarts(
    "[$([CX3;H1,$([CX3][#6])]),$([CX3;!R][N;R;X3]),$([SX4]([#6])(=[OX1])=[OX1]),$([SX3]([#6])=[OX1])]"
    "(=[OX1])[OX2][#6]")

#: groups senior to the esters, Table 4.1): an ion (the nitro group and the N-oxide are no ion of the class), a
#: carboxylic, sulfur, phosphorus or boron acid group, an anhydride
_SENIOR = tuple(Chem.MolFromSmarts(s) for s in (
    "[+,-;!$([N+]~[O-]);!$([O-]~[N+])]",
    "[CX3,SX4,SX3,PX4,PX3,BX3,AsX4,AsX3](=[O,S])[OX2H1,SX2H1]",
    "[CX3](=O)[OX2][CX3,SX4,PX4]=O"))

#: an acyloxy prefix as the reader of the name sees it: 'acetyloxy', 'hexadecanoyloxy', '[(9Z)-octadec-9-enoyl]oxy',
#: '(2-methylpropanoyl)oxy', 'benzoyloxy', 'formyloxy', '(methanesulfonyl)oxy'
_ACYLOXY = re.compile(r"(?:oyl|acetyl|formyl|carbonyl|sulfonyl|sulfinyl)[\]\)\}]*oxy")


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


def _ring_systems(mol):
    """The ring systems of ``mol`` as atom sets: rings that share an atom (fused, spiro, bridged) are one system.
    Rings joined by a bond, or by an atom between them, are two,, name each system as one
    parent hydride; names the rings joined by single bonds as an assembly)."""
    systems = []
    for ring in mol.GetRingInfo().AtomRings():
        merged = set(ring)
        rest = []
        for system in systems:
            if system & merged:
                merged |= system
            else:
                rest.append(system)
        systems = rest + [merged]
    return systems


def _on_one_parent(mol, carbons) -> bool:
    """True when the carbons that carry the ester oxygens are ring atoms of one ring system, or acyclic carbons that
    lie on one chain: the minimal subtree of the acyclic carbon skeleton that joins them is a path."""
    carbons = set(carbons)
    in_ring = {a for a in carbons if mol.GetAtomWithIdx(a).IsInRing()}
    if in_ring:
        return in_ring == carbons and any(carbons <= system for system in _ring_systems(mol))
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
    ``"linked"`` -- one alcoholic component of more than one parent carries every ester group, and the carbons that
    carry them are one symmetry class of it (ring, hetero atom or chain between identical arms: a multiplicative
    organyl group, '1,2-phenylenebis(methylene)', 'oxydi(ethane-2,1-diyl)'; ``count`` >= 2);
    ``"other"`` -- an alcoholic fragment holds two or more ester groups and the molecule is not that case
    (several alcoholic components, a polyol of unequal arms): the rule of the polyester is another one.
    An ester linkage in a ring (a lactone) is left out: SENIORITY ORDER FOR CLASSES, Table 4.1 (:18182), "9
    Esters (functional class names are given to noncyclic esters; lactones and other cyclic esters are named as
    heterocycles; see 16 below)"; the open-chain esters beside it are read as in any other molecule."""
    matches = mol.GetSubstructMatches(_ESTER)
    if not matches:
        return "none", 0
    bonds = []
    for acyl, _o_dbl, o_ester, alcohol in matches:
        if not mol.GetBondBetweenAtoms(acyl, o_ester).IsInRing() and (acyl, o_ester, alcohol) not in bonds:
            bonds.append((acyl, o_ester, alcohol))
    if not bonds:
        return "none", 0
    rw = Chem.RWMol(mol)
    for acyl, o_ester, _alcohol in bonds:
        rw.RemoveBond(acyl, o_ester)
    frags = [set(frag) for frag in Chem.GetMolFrags(rw)]
    owner = {a: k for k, frag in enumerate(frags) for a in frag}
    per_fragment = {}
    oxygens = {}
    for _acyl, o_ester, alcohol in bonds:
        per_fragment.setdefault(owner[o_ester], []).append(alcohol)
        oxygens.setdefault(owner[o_ester], []).append(o_ester)
    if all(len(v) == 1 for v in per_fragment.values()):
        return "none", 0
    if len(per_fragment) == 1:
        frag_id = next(iter(per_fragment))
        carbons = per_fragment[frag_id]
        if mol.GetAtomWithIdx(carbons[0]).GetAtomicNum() == 6:
            anions = {Chem.MolFragmentToSmiles(rw, atomsToUse=sorted(frags[owner[acyl]]), isomericSmiles=True)
                      for acyl, _o, _c in bonds}
            kind = polyol_group_kind(mol, rw, frags[frag_id], carbons, oxygens[frag_id], len(anions) == 1)
            if kind != "other":
                return kind, len(carbons)
    return "other", 0


def polyol_group_kind(mol, cut_mol, polyol, carbons, oxygens, anions_identical) -> str:
    """How the polyol of an alcoholic component is a multivalent organyl group "Polyesters formed from a
    single 'alcoholic' component", the Blue Book): ``"single"`` -- the carbons that carry the ester oxygens lie on
    one parent (a ring system, or one chain); ``"linked"`` -- the polyol is of more than one parent and is a
    multiplicative organyl group (one symmetry class of carbons, or identical arms under identical anions);
    ``"other"`` -- it is neither, and the polyester is named by substitutive nomenclature, (:31856).

    ``mol`` is the molecule, ``cut_mol`` the molecule with the acid-oxygen bond of every ester group cut, ``polyol`` the
    atoms of the alcoholic fragment (the ester oxygens included), ``carbons`` the carbons that carry the ester oxygens,
    ``oxygens`` those ester oxygens, ``anions_identical`` whether every ester group has the same anion. The one reading
    of the rule for ``single_polyol_ester`` (the spelling check) and the label of the method (2) names
    (``rules.esters.polyol_is_one_multivalent_group``)."""
    if _on_one_parent(mol, carbons):
        return "single"
    if _one_symmetry_class(cut_mol, carbons) or (anions_identical and _identical_arms(cut_mol, polyol, oxygens)):
        return "linked"
    return "other"


def _one_symmetry_class(cut_mol, atoms) -> bool:
    """True when ``atoms`` (two or more) are one symmetry class of ``cut_mol`` (the molecule with the acyl-oxygen
    bonds of its ester groups cut), the constitution alone: identical arms, whatever acid each ester group has and
    whatever its stereo layer says."""
    work = Chem.Mol(cut_mol)
    work.UpdatePropertyCache(strict=False)
    ranks = list(Chem.CanonicalRankAtoms(work, breakTies=False, includeChirality=False))
    return len(atoms) >= 2 and len({ranks[a] for a in atoms}) == 1


def _units_of(cut_mol, polyol):
    """The units of the polyol of a polyester, as the list of their atom sets: a ring system, a ring
    assembly (ring systems of one skeleton bonded directly by single bonds,, an unbranched chain of
    acyclic carbon atoms, a run of one hetero element, or one other atom. The simple groups of a multiplicative
    group,:5208) and the components of a concatenated one,:5238). None when a
    chain of carbon atoms branches: a branched skeleton has no single choice of central group here."""
    cut_mol = Chem.Mol(cut_mol)
    cut_mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(cut_mol)           # the rings of the cut molecule, which an edit of the bonds left unread
    systems = []
    for ring in cut_mol.GetRingInfo().AtomRings():
        if not set(ring) <= polyol:
            continue
        merged = set(ring)
        rest = []
        for system in systems:
            if system & merged:
                merged |= system
            else:
                rest.append(system)
        systems = rest + [merged]
    keys = [Chem.MolFragmentToSmiles(cut_mol, atomsToUse=sorted(s), isomericSmiles=False) for s in systems]
    parent = list(range(len(systems)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for bd in cut_mol.GetBonds():
        a, b = bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()
        ia = next((i for i, s in enumerate(systems) if a in s), None)
        ib = next((i for i, s in enumerate(systems) if b in s), None)
        if ia is not None and ib is not None and ia != ib and bd.GetBondTypeAsDouble() == 1.0 \
                and keys[ia] == keys[ib]:
            parent[root(ia)] = root(ib)
    groups = {}
    for i, s in enumerate(systems):
        groups.setdefault(root(i), set()).update(s)
    nodes = [frozenset(g) for g in groups.values()]
    in_ring = set().union(*nodes) if nodes else set()
    acyclic = polyol - in_ring
    seen = set()
    for a in sorted(acyclic):
        if a in seen:
            continue
        z = cut_mol.GetAtomWithIdx(a).GetAtomicNum()
        run = {a}
        stack = [a]
        while stack:
            x = stack.pop()
            for nb in cut_mol.GetAtomWithIdx(x).GetNeighbors():
                j = nb.GetIdx()
                if j in acyclic and j not in run and nb.GetAtomicNum() == z:
                    run.add(j)
                    stack.append(j)
        seen |= run
        if z == 6 and any(len([n for n in cut_mol.GetAtomWithIdx(x).GetNeighbors() if n.GetIdx() in run]) > 2
                          for x in run):
            return None
        nodes.append(frozenset(run))
    return nodes


def _arm_key(cut_mol, atoms, hook):
    """The canonical constitution of an arm (``atoms``) with the atom of the central unit it hangs on (``hook``) as a
    dummy atom: the linking bond and everything beyond it, nothing of the central unit."""
    rw = Chem.RWMol(cut_mol)
    for atom in rw.GetAtoms():                  # the nuclides of the arms count, the configuration does not
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in rw.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    at = rw.GetAtomWithIdx(hook)
    at.SetAtomicNum(0)
    at.SetFormalCharge(0)
    at.SetIsotope(0)
    at.SetNoImplicit(True)
    at.SetNumExplicitHs(0)
    at.SetIsAromatic(False)
    rw.UpdatePropertyCache(strict=False)
    return Chem.MolFragmentToSmiles(rw, atomsToUse=sorted(set(atoms) | {hook}), isomericSmiles=True,
                                    allBondsExplicit=True)


def _identical_arms(cut_mol, polyol, oxygens) -> bool:
    """True when one unit of the polyol (``_units_of``) is a central group whose branches toward the ester oxygens
    are identical: removing it leaves one piece per ester oxygen, each holding that one oxygen, all of one
    constitution (the linking bond and everything beyond it); pieces without an ester oxygen are substituents of the
    central group, which (:5208) allows. The stereo layer is not read, as in
    :func:`_one_symmetry_class`: 'identical' is the constitution of the arms."""
    nodes = _units_of(cut_mol, set(polyol))
    if nodes is None or len(oxygens) < 2:
        return False
    leaf = set(oxygens)
    for centre in nodes:
        if centre & leaf:
            continue
        pieces = []
        seen = set(centre)
        for start in sorted(polyol - set(centre)):
            if start in seen:
                continue
            piece = {start}
            stack = [start]
            while stack:
                x = stack.pop()
                for nb in cut_mol.GetAtomWithIdx(x).GetNeighbors():
                    j = nb.GetIdx()
                    if j in polyol and j not in piece and j not in centre:
                        piece.add(j)
                        stack.append(j)
            seen |= piece
            hooks = {(nb.GetIdx(), x) for x in piece for nb in cut_mol.GetAtomWithIdx(x).GetNeighbors()
                     if nb.GetIdx() in centre}
            pieces.append((piece, hooks))
        holding = [(p, h) for p, h in pieces if p & leaf]
        if len(holding) != len(oxygens) or any(len(p & leaf) != 1 or len(h) != 1 for p, h in holding):
            continue
        keys = {_arm_key(cut_mol, p, next(iter(h))[0]) for p, h in holding}
        if len(keys) == 1:
            return True
    return False


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
    if state not in ("single", "linked"):
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
