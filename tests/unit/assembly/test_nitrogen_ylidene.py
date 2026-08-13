"""Phase 1 B1: the P-29.2 / BB(:1703) nitrogen ylidene substituent form.

A nitrogen attached to its parent by a DOUBLE bond is an ``ylidene`` free
valence, not a ``-yl``: ``=N-N<`` -> ``hydrazin-1-ylidene`` (the deprecated
``hydrazono`` is never emitted, BB :1703), ``=N-R`` -> ``{R}imino``. Before this
the cascade built ``...hydrazinyl`` and the P-29.2 guard refused it, so the
whole branch abstained -- the measured decorated-steroid / hydrazone class.

Asserted at the SUBSTITUENT-PREFIX level (deterministic, OPSIN-free). The
whole-molecule round-trip is verified in a fresh subprocess by
`` (the in-pytest SELF-01/OPSIN path is unreliable
across process warm-up -- the reason that harness uses subprocesses); each
prefix below is confirmed to round-trip embedded in a parent, e.g.
``(2,2-dimethylhydrazin-1-ylidene)cyclohexane`` -> ``CN(C)N=C1CCCCC1``.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import name_substituent

pytestmark = pytest.mark.unit


def _frag_and_attach(smi):
    """The =N-... substituent fragment (excl. the parent) + its =N atom."""
    mol = Chem.MolFromSmiles(smi)
    na = next(a.GetIdx() for a in mol.GetAtoms()
              if a.GetSymbol() == 'N'
              and any(b.GetBondType() == Chem.BondType.DOUBLE
                      for b in a.GetBonds()))
    parent = next(b.GetOtherAtom(mol.GetAtomWithIdx(na)).GetIdx()
                  for b in mol.GetAtomWithIdx(na).GetBonds()
                  if b.GetBondType() == Chem.BondType.DOUBLE)
    frag, stack = {na}, [na]
    while stack:
        c = stack.pop()
        for nb in mol.GetAtomWithIdx(c).GetNeighbors():
            k = nb.GetIdx()
            if k != parent and k not in frag:
                frag.add(k)
                stack.append(k)
    return mol, sorted(frag), na


@pytest.mark.parametrize("smi,expected", [
    ("C1CCC(=NN(C)C)CC1", "2,2-dimethylhydrazin-1-ylidene"),  # N,N-dimethylhydrazone
    ("C1CCC(=NN)CC1",     "hydrazin-1-ylidene"),              # bare hydrazone
    ("C1CCC(=NNC)CC1",    "2-methylhydrazin-1-ylidene"),      # methylhydrazone
    ("C1CCC(=NC)CC1",     "methylimino"),                     # N-methylimine
    ("C1CCC(=N)CC1",      "imino"),                           # imine =NH
])
def test_nitrogen_ylidene_prefix(smi, expected):
    mol, frag, na = _frag_and_attach(smi)
    assert name_substituent(mol, frag, na) == expected


def test_deprecated_hydrazono_never_emitted():
    """BB :1703 discontinued 'hydrazono' even for general nomenclature; the
    systematic 'hydrazinylidene' is emitted instead."""
    mol, frag, na = _frag_and_attach("C1CCC(=NN)CC1")
    assert "hydrazono" not in name_substituent(mol, frag, na)
