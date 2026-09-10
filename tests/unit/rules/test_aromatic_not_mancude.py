"""RDKit aromaticity is not evidence that a ring is MANCUDE (Task AA4).

``GetIsAromatic`` answers "does this ring satisfy RDKit's aromaticity
model?". Three sites in ``rules/heterocycles.py`` read it as "this ring
carries the maximum number of noncumulative double bonds", and the two come
apart whenever a ring atom contributes a LONE PAIR to the pi system instead of
a double bond. Two or more pyrrole-type heteroatoms do exactly that, so

    c1c[nH][nH]1 is RDKit-aromatic with 1 ring double bond, max 2
    c1cc[nH]cc[nH]c1 is RDKit-aromatic with 3 ring double bonds, max 4

and both used to be named as their mancude parent -- ``1,2-diazete`` and
``1,4-diazocine`` -- which (``the Blue Book``) defines as a
DIFFERENT molecule: "Unsaturated compounds are those having the maximum number
of noncumulative double bonds (mancude compounds) and at least one double
bond."

The governing rules, each opened at write time with its section heading:

* **** (``:8222``) -- the definition quoted above.
* ** "Hantzsch-Widman heteromonocycles"** (``:24169``), sentence
  ``:24171`` -- "'Hydro' prefixes added to names of fully unsaturated
  Hantzsch-Widman rings lead to preferred IUPAC names for partially
  unsaturated rings."
* ** "General methodology"** (``:16878``), sentence ``:16880`` --
  "Indicated hydrogen atoms have priority over 'hydro' prefixes for low
  locants. If indicated hydrogen atoms are present in a name, the 'hydro'
  prefixes precede them."
* ** "Names of saturated heteromonocyclic compounds"** (``:16926``),
  sentence ``:16928`` -- "Preferred IUPAC names of saturated heteromonocyclic
  compounds are either Hantzsch-Widman names described in or
  retained names described in Table 2.3."
* **** (``:8284``) -- the heteroatom numbering cascade and the
  citation sequence F, Cl, Br, I, O, S, Se, Te, N, P, As,...

Measured over 397,371 enumerated bare heteromonocycles (sizes 3-10, up to
three heteroatoms from N/O/S/P/Se/Te/As, every independent ring-edge set):
133 wrong molecules of the hydro kind, 56 saturated rings spelled with an
unsaturated stem, 0 elsewhere.
"""

import itertools

import pytest
from rdkit import Chem

from orthonym.rules.heterocycles import (
    _kekulized_if_aromatic,
    _mancude_max_matching,
    _ring_double_bond_count,
    name_heterocycle,
)


def _name(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return name_heterocycle(mol, mol.GetRingInfo().AtomRings()[0])


# --------------------------------------------------------------------------
# Family A -- RDKit-aromatic HYDRO forms named as their mancude parent.
# The three rows reported (unfixed) by Task AA2, plus one from each ring size
# and one carrying an indicated hydrogen.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    # the three Task AA2 reported as producer-level wrong molecules
    ("c1c[nH][nH]1", "1,2-dihydro-1,2-diazete"),
    ("c1cc[nH]cc[nH]c1", "1,4-dihydro-1,4-diazocine"),
    ("c1ccc[nH][nH]cc1", "1,2-dihydro-1,2-diazocine"),
    # four-membered, mixed heteroatoms
    ("c1c[pH][nH]1", "1,2-dihydro-1,2-azaphosphete"),
    ("c1c[pH][pH]1", "1,2-dihydro-1,2-diphosphete"),
    # seven-membered: ih_count == 1, so an indicated hydrogen is REQUIRED and
    # (:16880) puts the hydro prefixes in front of it
    ("c1c[nH][nH]cc[nH]1", "2,5-dihydro-1H-1,2,5-triazepine"),
    ("c1cc[nH][nH][nH]c1", "2,3-dihydro-1H-1,2,3-triazepine"),
    # seven-membered with a divalent chalcogen: n_eligible is 6, so
    # ih_count == 0 and NO indicated hydrogen may be cited
    ("c1cc[nH]o[nH]c1", "2,7-dihydro-1,2,7-oxadiazepine"),
    ("c1c[nH]scc[nH]1", "2,5-dihydro-1,2,5-thiadiazepine"),
    # eight-membered
    ("c1cc[pH]cc[pH]c1", "1,4-dihydro-1,4-diphosphocine"),
])
def test_aromatic_hydro_form_gets_hydro_prefixes(smiles, expected):
    assert _name(smiles) == expected


# --------------------------------------------------------------------------
# Family B -- fully SATURATED rings that RDKit calls aromatic. These take the
# saturated Hantzsch-Widman stem,:16928), never the unsaturated
# one and never a hydro prefix.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    ("N1NN1", "triaziridine"),            # plain single-bond input, still aromatic to RDKit
    ("[nH]1[nH][nH]1", "triaziridine"),
    ("N1NO1", "1,2,3-oxadiaziridine"),
    ("[nH]1[nH]s1", "1,2,3-thiadiaziridine"),
    ("[pH]1[pH][pH]1", "triphosphirane"),
    ("[nH]1o[pH]1", "1,2,3-oxazaphosphiridine"),
    # all-divalent three-membered rings: no ring double bond is possible at
    # all, so the unsaturated stem can never apply
    ("s1ss1", "trithiirane"),
    ("o1ss1", "1,2,3-oxadithiirane"),
    ("[nH]1ss1", "1,3,2-dithiaziridine"),
    ("[nH]1oo1", "1,3,2-dioxaziridine"),
])
def test_saturated_ring_gets_the_saturated_stem(smiles, expected):
    assert _name(smiles) == expected


# --------------------------------------------------------------------------
# Nothing that was already right may move.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    ("c1ccncc1", "pyridine"),
    ("c1cc[nH]c1", "1H-pyrrole"),
    ("c1ccoc1", "furan"),
    ("c1ccsc1", "thiophene"),
    ("c1cnc[nH]1", "1H-imidazole"),
    ("c1cc[nH]n1", "1H-pyrazole"),
    ("c1ccnnc1", "pyridazine"),
    ("C1=COCC1", "2,3-dihydrofuran"),
    ("C1C=CC=CN1", "1,2-dihydropyridine"),
    ("C1CCOC1", "oxolane"),
    ("C1CCNCC1", "piperidine"),
    # AA2's lambda class must be byte-identical: the lambda branch refuses
    # RDKit-aromatic rings before any of this is reachable.
    ("C1=CC[SH]=C1", "2H-1λ4-thiophene"),
    ("C1CC[SH]=C1", "3,4-dihydro-2H-1λ4-thiophene"),
])
def test_correct_names_are_unchanged(smiles, expected):
    assert _name(smiles) == expected


# --------------------------------------------------------------------------
# The helpers themselves.
# --------------------------------------------------------------------------
def test_kekulized_helper_is_identity_for_non_aromatic_rings():
    """The non-aromatic half of every caller set is byte-identical BY
    CONSTRUCTION -- the helper hands back the very object it was given."""
    for smiles in ("C1=COCC1", "C1CCOC1", "C1C=CC=CN1", "C1CC[SH]=C1"):
        mol = Chem.MolFromSmiles(smiles)
        ring = set(mol.GetRingInfo().AtomRings()[0])
        assert _kekulized_if_aromatic(mol, ring) is mol, smiles


def test_kekulized_helper_exposes_double_bonds_of_an_aromatic_ring():
    mol = Chem.MolFromSmiles("c1cc[nH]cc[nH]c1")
    ring = set(mol.GetRingInfo().AtomRings()[0])
    assert _ring_double_bond_count(mol, ring) == 3          # not 0, as written
    kek = _kekulized_if_aromatic(mol, ring)
    assert kek is not mol
    assert not any(kek.GetAtomWithIdx(i).GetIsAromatic() for i in ring)


@pytest.mark.parametrize("smiles,expected_d", [
    ("c1ccccc1", 3),
    ("c1ccncc1", 3),
    ("c1cc[nH]c1", 2),
    ("c1c[nH][nH]1", 1),
    ("N1NN1", 0),
    ("C1CCOC1", 0),
    ("C1=COCC1", 1),
])
def test_ring_double_bond_count(smiles, expected_d):
    mol = Chem.MolFromSmiles(smiles)
    ring = set(mol.GetRingInfo().AtomRings()[0])
    assert _ring_double_bond_count(mol, ring) == expected_d


# --------------------------------------------------------------------------
# The invariant, as a property over an enumerated sample rather than a list of
# molecules: an emitted Hantzsch-Widman name may assert full unsaturation only
# when the ring really carries the mancude maximum. Fail closed otherwise --
# an abstention is recoverable, a wrong molecule is not.
# --------------------------------------------------------------------------
def _sample_rings():
    """Bare heteromonocycles of sizes 4-8, one or two N/O/S heteroatoms, every
    independent ring-edge set. Small enough to stay a unit test."""
    out = {}
    for n in range(4, 9):
        edge_sets = []
        for r in range(0, n // 2 + 1):
            for combo in itertools.combinations(range(n), r):
                used = set()
                if all(not (p in used or (p + 1) % n in used) and
                       (used.add(p) or used.add((p + 1) % n) or True)
                       for p in combo):
                    edge_sets.append(frozenset(combo))
        for k in (1, 2):
            for posns in itertools.combinations(range(n), k):
                for hets in itertools.product(("N", "O", "S"), repeat=k):
                    pat = ["C"] * n
                    for p, h in zip(posns, hets):
                        pat[p] = h
                    for edges in edge_sets:
                        toks = []
                        for i in range(n):
                            if i == 0:
                                toks.append(pat[0] + ("=1" if (n - 1) in edges else "1"))
                            else:
                                toks.append(("=" if (i - 1) in edges else "") + pat[i])
                        smi = "".join(toks) + "1"
                        mol = Chem.MolFromSmiles(smi)
                        if mol is None or mol.GetRingInfo().NumRings() != 1:
                            continue
                        if mol.GetNumHeavyAtoms() != n:
                            continue
                        out[Chem.MolToSmiles(mol)] = None
    return sorted(out)


def test_no_ring_is_named_as_a_mancude_parent_it_is_not():
    rings = _sample_rings()
    assert len(rings) > 500, f"sample too small to be meaningful: {len(rings)}"
    offenders = []
    for smi in rings:
        mol = Chem.MolFromSmiles(smi)
        ring = mol.GetRingInfo().AtomRings()[0]
        name = name_heterocycle(mol, ring)
        if not name:
            continue
        d = _ring_double_bond_count(mol, set(ring))
        if d is None or d == 0:
            continue
        max_match, _ = _mancude_max_matching(mol, list(ring))
        if d < max_match and "hydro" not in name:
            offenders.append((smi, name, d, max_match))
    assert not offenders, (
        f"{len(offenders)} ring(s) named as a mancude parent they are not: "
        f"{offenders[:10]}"
    )
