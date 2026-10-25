import pytest

from orthonym.rules.phane_pin.amplificants import amplificant_hydride

from ._helpers import mol


def _ring_atoms(m):
    return frozenset(i for r in m.GetRingInfo().AtomRings() for i in r)


@pytest.mark.parametrize("smiles,name,prefix,n_numberings", [
    ("Cc1ccccc1", "benzene", "benzena", 12),
    ("Cc1ccncc1", "pyridine", "pyridina", 2),
    ("Cc1ccoc1", "furan", "furana", 2),
    ("CC1CCNCC1", "piperidine", "piperidina", 2),
    ("CC1CCCCC1", "cyclohexane", "cyclohexana", 12),
    ("Cc1cnoc1", "1,2-oxazole", "[1,2]oxazola", 1),      # (:14794): brackets round heteroatom locants
    ("Cc1ccc2ccccc2c1", "naphthalene", "naphthalena", 4),
    ("Cc1cnccn1", "pyrazine", "pyrazina", 4),
    ("Cc1cc[nH]c1", "pyrrole", "pyrrola", 2),            # (:15173): the indicated hydrogen is cited apart
])
def test_amplificant_hydride(smiles, name, prefix, n_numberings):
    # the number of numberings is the order of the automorphism group of the hydride (RDKit, VERIFIED 2026-10-13)
    m = mol(smiles)
    got = amplificant_hydride(m, _ring_atoms(m))
    assert got.name == name and got.prefix == prefix and len(got.numberings) == n_numberings


@pytest.mark.parametrize("smiles", [
    "CC1=CCCCC1",             # a cycloalkene: the engine names it with an 'ene' ending, not as a hydro derivative of benzene
    "O=C1CCCCC1",             # an exocyclic double bond on the ring
    "C1=CC2=CC=CC3=C2C(=C1)C=C3",   # peri-fused: no cata-fused numbering
])
def test_amplificants_that_decline(smiles):
    m = mol(smiles)
    assert amplificant_hydride(m, _ring_atoms(m)) is None


@pytest.mark.parametrize("smiles,needs", [
    ("c1cc2c(cn1)ncc1cc[nH]c12", True),     # a fused hydride with a pyrrole-type NH
    ("c1cc[nH]c1", True),
    ("C1=CC=CNC=C1", True),                 # 1H-azepine
    ("C1=CCOC=C1", True),                   # 2H-pyran
    ("c1cc[siH]cc1", False),                # silabenzene: the Si-H has a valence left for a pi bond
    ("c1ccc2ncccc2c1", False),
    ("C1=CC=COC=C1", False),
])
def test_a_hydride_with_a_saturated_ring_atom_needs_indicated_hydrogen(smiles, needs):
    # (:3557): read from the structure, whatever the name says
    from orthonym.rules.phane_pin.amplificants import _needs_indicated_hydrogen
    assert _needs_indicated_hydrogen(mol(smiles)) is needs
