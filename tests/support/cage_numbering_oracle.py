"""An oracle for the numbering of a von Baeyer cage that carries detachable prefixes and a suffix.

It does not use the numbering choice of ``VonBaeyerAnalyzer``: every numbering of a cage is the numbering of
its bare skeleton composed with an automorphism of that skeleton (an element-blind graph automorphism), so the
oracle takes, over all of them, the lowest (c) suffix locants, then (f) all prefix locants together, then (g)
the locants in the order the prefixes are cited, the Blue Book; (f):3300, (g):3307;
alphanumerical order). The prefixes it knows are the few the tests use: halogens, methoxy, methyl and ethyl; an
``-OH`` is the suffix ``ol``.
"""
import re

from rdkit import Chem

from orthonym.perception.automorphisms import skeleton_automorphisms
from orthonym.rules.polycyclic import VonBaeyerAnalyzer

_NAMES = {("Cl", 1): "chloro", ("Br", 1): "bromo", ("F", 1): "fluoro", ("I", 1): "iodo",
          ("O", 2): "methoxy", ("C", 1): "methyl", ("C", 2): "ethyl"}
_MULTIPLIER = {1: "", 2: "di", 3: "tri"}


def _branches(mol, ring):
    """``[(ring atom, 'ol' or the prefix name)]`` for every branch of the cage."""
    out = []
    for atom_idx in sorted(ring):
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            if nbr.GetIdx() in ring:
                continue
            branch, stack = {nbr.GetIdx()}, [nbr.GetIdx()]
            while stack:
                for nxt in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
                    if nxt.GetIdx() not in ring and nxt.GetIdx() not in branch:
                        branch.add(nxt.GetIdx())
                        stack.append(nxt.GetIdx())
            if nbr.GetSymbol() == "O" and len(branch) == 1:
                out.append((atom_idx, "ol"))
            else:
                out.append((atom_idx, _NAMES[(nbr.GetSymbol(), len(branch))]))
    return out


def oracle(smiles):
    """``(prefix part of the name, suffix locants)`` of the lowest numbering over the skeleton's automorphisms.

    The family of numberings is the engine's numbering of the molecule (any member will do: the
    descriptor fixes the labelled skeleton, so the others are its images under the automorphisms)
    composed with every automorphism of the cage's ring graph. Which descriptor that is, is not what is
    tested here: chooses it before the numbering criteria apply.
    """
    mol = Chem.MolFromSmiles(smiles)
    ring = {i for r in mol.GetRingInfo().AtomRings() for i in r}
    branches = _branches(mol, ring)
    base = VonBaeyerAnalyzer().analyze(mol, ring).numbering
    automorphisms, exhaustive = skeleton_automorphisms(mol, ring, element_blind=True, cap=5000)
    assert exhaustive
    best = None
    for sigma in automorphisms:
        locant = {a: base[sigma[a]] for a in ring}
        suffix = tuple(sorted(locant[a] for a, kind in branches if kind == "ol"))
        placed = sorted((kind, locant[a]) for a, kind in branches if kind != "ol")
        key = (suffix, tuple(sorted(loc for _, loc in placed)), tuple(placed))
        if best is None or key < best[0]:
            best = (key, suffix, placed)
    _, suffix, placed = best
    groups = {}
    for kind, loc in placed:
        groups.setdefault(kind, []).append(loc)
    parts = [",".join(map(str, sorted(locs))) + "-" + _MULTIPLIER[len(locs)] + kind
             for kind, locs in sorted(groups.items())]
    return "-".join(parts), suffix


_CAGE = re.compile(r"^(?P<prefix>.*?)(?:bi|tri|tetra|penta|hexa|hepta|octa|nona)cyclo\[")
_SUFFIX = re.compile(r"(\d+(?:,\d+)*)-(?:di|tri)?ol$")


def check(smiles, name):
    prefix, suffix = oracle(smiles)
    m = _CAGE.match(name)
    assert m, name
    got = m.group("prefix").rstrip("-")
    assert got == prefix, (name, "expected prefixes", prefix)
    s = _SUFFIX.search(name)
    assert (tuple(int(x) for x in s.group(1).split(",")) if s else ()) == suffix, (name, suffix)


