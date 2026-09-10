"""An exocyclic double bond is not hydrogenation — a pyridinone is not a hydro form.

THE REGRESSION THIS PINS
------------------------
 made `name_heterocycle` refuse any ring holding fewer ring double
bonds than its mancude parent, on the correct grounds that spelling the mancude
stem for such a ring names a *different* molecule (that fix removed 149 wrong
molecules and must not be undone).

But `_mancude_bond_eligible` computed the mancude maximum from the **element**
alone. It never noticed that a ring atom bearing an **exocyclic double bond** has
already spent the valence a ring double bond would need. So in
``pyridin-2(1H)-one`` the C-2 carbon — 2 ring sigma bonds plus ``C=O`` — was
counted as able to hold a ring double bond, the ring's maximum came out as
pyridine's **3** against an actual **2**, and the ring read as a HYDRO FORM and
was refused.

Measured cost: the correct ``5-(3-fluorophenyl)-1H-pyridin-2-one`` became
``unknown organic compound``. Found by a 400-molecule corpus A/B, **not** by the
phase gate (no such row in the PIN oracle) and **not** by the 397,371-ring
enumeration behind (which enumerated *bare* rings, so a ring bearing
an aryl substituent was outside its universe).

THE RULE
--------
The compound is a **pseudoketone**, named on the numbered mancude ring with an
added suffix — not a hydro form:

* **** "'Hidden' amides" (``the Blue Book``) — naming an acyl group
  as a substituent on a heterocyclic ring nitrogen "is allowed but only in
  general nomenclature", because "preferred IUPAC names are constructed" as
  pseudoketones.
* **** "Lactams and lactims" (``:33224``) — of its two methods,
  "(1) as heterocyclic pseudoketones" is the one that "generates preferred IUPAC
  names".

⚠ Deliberately narrow: only exocyclic bonds of order **>= 2** count, and only to
atoms outside the ring. Counting single bonds would veto an N-methyl ring
nitrogen; counting RDKit's aromatic order 1.5 would veto naphthalene's fusion
carbons. Both are pinned below as controls, because both were wrong in a first
draft of the fix and only a dry-run comparison against the old rule caught them.
"""
from __future__ import annotations

import pytest
from rdkit import Chem

from orthonym.rules.heterocycles import _mancude_bond_eligible


def _ring(smiles: str, *, want_hetero: bool = True):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"bad test SMILES {smiles}"
    for ring in mol.GetRingInfo().AtomRings():
        has_het = any(mol.GetAtomWithIdx(i).GetSymbol() != "C" for i in ring)
        if has_het == want_hetero:
            return mol, list(ring)
    raise AssertionError(f"no matching ring in {smiles}")


# ---------------------------------------------------------------------------
# 1. The eligibility rule itself
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_exocyclic_double_bond_carbon_cannot_hold_a_ring_double_bond():
    """C-2 of pyridin-2(1H)-one: 2 ring sigma + C=O leaves no valence."""
    mol, ring = _ring("O=c1cccc[nH]1")
    eligible = _mancude_bond_eligible(mol, ring)
    ineligible = [
        i for i, ok in zip(ring, eligible)
        if not ok
    ]
    assert len(ineligible) == 1, f"expected exactly one ineligible atom, got {ineligible}"
    atom = mol.GetAtomWithIdx(ineligible[0])
    assert atom.GetSymbol() == "C"
    assert any(
        b.GetBondTypeAsDouble() >= 2 and b.GetOtherAtomIdx(atom.GetIdx()) not in set(ring)
        for b in atom.GetBonds()
    ), "the ineligible atom must be the one bearing the exocyclic double bond"


@pytest.mark.unit
@pytest.mark.parametrize("smiles,n_eligible", [
    ("c1ccncc1", 6),        # pyridine — every atom eligible, unchanged
    ("c1cc[nH]c1", 5),      # 1H-pyrrole — unchanged
    ("c1ccoc1", 4),         # furan — the divalent O is the only ineligible one
    ("c1ccsc1", 4),         # thiophene
    ("O=c1cccc[nH]1", 5),   # pyridin-2(1H)-one — the C=O carbon drops out
    ("C1=CC(=O)NC=C1", 5),  # same ring written kekule
])
def test_eligible_counts(smiles, n_eligible):
    mol, ring = _ring(smiles)
    assert sum(_mancude_bond_eligible(mol, ring)) == n_eligible


# ---------------------------------------------------------------------------
# 2. The two controls that a first draft of the fix got WRONG
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_aromatic_fusion_bond_is_not_counted_as_exocyclic():
    """Naphthalene's fusion carbons must stay eligible.

    RDKit reports an aromatic bond as order 1.5. A draft that counted every
    exocyclic bond order took naphthalene's per-ring maximum from 3 to 2.
    """
    mol, ring = _ring("c1ccc2ccccc2c1", want_hetero=False)
    assert sum(_mancude_bond_eligible(mol, ring)) == 6


@pytest.mark.unit
def test_single_exocyclic_bond_does_not_veto_a_ring_nitrogen():
    """An N-methyl ring nitrogen keeps its eligibility.

    A single bond spends one valence, not two, so N (bonding number 3) still has
    room. A draft that counted single bonds broke `1-methylpyridin-2(1H)-one`.
    """
    mol, ring = _ring("O=C1C=CC=CN1C")
    eligible = dict(zip(ring, _mancude_bond_eligible(mol, ring)))
    n_idx = [i for i in ring if mol.GetAtomWithIdx(i).GetSymbol() == "N"][0]
    assert eligible[n_idx] is True


# ---------------------------------------------------------------------------
# 3. End-to-end: the lost name is back, and 65a2206e's fixes still hold
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    # the exact molecule the corpus A/B found lost
    ("C1=CC(=CC(=C1)F)C2=CNC(=O)C=C2", "5-(3-fluorophenyl)-1H-pyridin-2-one"),
    ("c1ccc(-c2ccc(=O)[nH]c2)cc1", "5-phenyl-1H-pyridin-2-one"),
    #... and the unsubstituted / alkyl / N-substituted members must not move
    ("O=c1cccc[nH]1", "pyridin-2(1H)-one"),
    ("CC1=CC(=O)NC=C1", "4-methylpyridin-2(1H)-one"),
    ("O=C1C=CC=CN1C", "1-methylpyridin-2(1H)-one"),
])
def test_pseudoketone_ring_is_named(smiles, expected):
    from orthonym.namer import name_compound
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize("smiles,expected", [
    ("N1NN1", "triaziridine"),                                  # not 'triazirine'
    ("C1=CN=N1", "1,2-diazete"),
    ("N1NC=CC=CCC1", "1,2,3,4-tetrahydro-1,2-diazocine"),
    ("N1C=CNCCCC1", "1,4,5,6,7,8-hexahydro-1,4-diazocine"),
])
def test_65a2206e_fixes_survive(smiles, expected):
    """The 149-wrong-molecule fix must not be weakened by the exocyclic veto.

    None of these rings carries an exocyclic multiple bond, so the veto returns
    True immediately for every atom and the eligibility expression is the one
     shipped, verbatim.
    """
    from orthonym.namer import name_compound
    assert name_compound(smiles) == expected
