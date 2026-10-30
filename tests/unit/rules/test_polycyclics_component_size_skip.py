"""identify_polycyclic skips a PAH pattern whose size no fused ring component has.

The coverage veto accepts a match only when the fused ring component holding it is exactly
the matched core, so such a pattern can never be accepted. The skip must leave every result
unchanged (checked against the unskipped search below) and must keep a fullerene derivative
with a hetero atom (the all-carbon cage guard does not see it) from searching its cage with
every pattern, which took minutes per call (PubChem CID 71528211).
"""
import subprocess
import sys
import textwrap

import pytest
from rdkit import Chem

from orthonym.rules import polycyclics as P

CAGE_CID_71528211 = "CC1=NC2=CC=CC=C2C(=O)N1C3=CC=C(C=C3)C(=O)N/N=C/C4=CC=C(C=C4)C5C67C8=C9C1=C2C3=C4C%10=C%11C%12=C%13C%14=C%10C%10=C4C4=C%15C%16=C%17C%18=C%19C%20=C%21C%22=C%23C%24=C%25C%26=C(C1=C3C%11=C%26C%12=C%24C%22=C%13C%20=C%14C%18=C%10%16)C1=C%25C3=C(C6=C19)C1=C6C7(C[N+]5(C)C)C(=C%15C8=C24)C%17=C6C%19=C%21C1=C%233"


def _unskipped(mol):
    """The table search as it was before the skip: every pattern, then the coverage veto."""
    components = []
    for ring in mol.GetRingInfo().AtomRings():
        rs = set(ring)
        merged = [c for c in components if c & rs]
        for c in merged:
            components.remove(c)
        components.append(rs.union(*merged) if merged else rs)
    for name, data, pattern in P._COMPILED_PAH:
        matches = mol.GetSubstructMatches(pattern, maxMatches=1, uniquify=True)
        if matches and len(matches[0]) == data['num_atoms']:
            core = set(matches[0])
            covered = set().union(*[c for c in components if c & core])
            if covered == core:
                return name
    return None


@pytest.mark.parametrize("smiles", [
    "Cc1ccc2ccccc2c1",                      # 2-methylnaphthalene
    "c1ccc(-c2cccc3ccccc23)cc1",            # 1-phenylnaphthalene: two components
    "C1=CC2C=CC1O2",                        # no PAH
    "c1ccc2c(c1)C1OC2C=C1",                 # 1,4-epoxy: extra ring atom, vetoed
    "Oc1ccc2ccc3cccc4ccc1c2c34",            # pyrenol
    "Cc1ccc2c(c1)Cc1ccccc1-2",              # 2-methylfluorene
    "c1ccc2cc3ccccc3cc2c1",                 # anthracene
    "Nc1ccc2c(c1)ccc1ccccc12",              # phenanthrenamine
    "c1ccc2c(c1)ccc1c2ccc2c1ccc1ccccc12",   # a larger cata-fused PAH
    "C1CCc2c(C1)ccc1ccccc21",               # tetrahydrophenanthrene
    "c1ccc2c(c1)-c1cccc3cccc-2c13",         # fluoranthene
    "O=C1c2ccccc2C(=O)c2ccccc12",           # anthraquinone
    "c1ccc2c(c1)C1(c3ccccc3-2)c2ccccc2-c2ccccc21",  # spiro-bifluorene: one spiro-joined component
    "c1ccc2c(c1)oc1ccc3ccccc3c12",          # a furan fused onto naphthalene
])
def test_skip_leaves_the_table_result_unchanged(smiles, monkeypatch):
    # only the table search: no exact-SMILES lookup, no open-series reading
    monkeypatch.setattr(P, "is_polycyclic_aromatic", lambda smi: False)
    monkeypatch.setattr(P, "_identify_series", lambda mol: None)
    mol = Chem.MolFromSmiles(smiles)
    assert P._identify_polycyclic_impl(mol) == _unskipped(mol)


def test_fullerene_derivative_with_a_hetero_atom_is_fast():
    code = textwrap.dedent(f"""
        from rdkit import Chem
        from orthonym.rules.polycyclics import identify_polycyclic
        print(identify_polycyclic(Chem.MolFromSmiles({CAGE_CID_71528211!r})))
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "None"
