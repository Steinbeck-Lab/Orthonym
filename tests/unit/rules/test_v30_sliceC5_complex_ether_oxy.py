"""v30 Slice C.5 — recursive (R)oxy for ring/heteroatom ether substituents.

`_name_alkoxy_branch` used to `return None` for any ether -O-R where R bears a ring or
heteroatom, dropping the whole substituent (SELF-01 -> abstain). The ring-capable
`name_substituent` already names such an R as a `-yl`; C.5 recurses on R and wraps as
`(R-yl)oxy`, so the glycosyloxy / heterocyclyloxy class composes and round-trips.

0-wrong is preserved by SELF-01 + the `.endswith('yl')` fail-closed guard: a non-`-yl`
recursion returns None (abstain), never a wrong molecule.
"""
from rdkit import Chem
from rdkit.Chem import inchi
import pytest

from orthonym.assembly.substituent_enumerator import name_substituent


def _ik(smi):
    m = Chem.MolFromSmiles(smi)
    return inchi.MolToInchiKey(m).split("-")[0] if m else None


def _rt(smi, name):
    from orthonym import namer as N
    osmi = N._validity_gate_name_to_smiles(name)
    return osmi is not None and _ik(smi) == _ik(osmi)


@pytest.mark.parametrize("smi", [
    "OC(=O)COc1cccnc1",   # (pyridin-3-yloxy)acetic acid  -- heteroaryl ether R
    "OC(=O)COC1CCCCC1",   # (cyclohexyloxy)acetic acid     -- carbocyclyl ether R
    "OC(=O)COC1CCOCC1",   # (oxan-4-yl)oxy...              -- saturated heterocyclyl ether R
])
def test_ring_ether_oxy_class_round_trips(smi):
    # Whole-molecule: -O-(ring R) used to drop the whole substituent (SELF-01 -> abstain).
    # C.5 recurses on R and composes (R)oxy; the emitted name must denote the SAME molecule.
    from orthonym import Orthonym
    out = Orthonym(general_fallback=True, general_fallback_unverified=True,
                    allow_aromatic_general=True).name(smi)
    assert out and "unknown" not in out, out
    assert _rt(smi, out), f"wrong molecule: {out!r}"


def test_alkoxy_branch_direct_handles_saturated_ring():
    # unit-level: the fixed handler returns a (R-yl)oxy prefix, not None.
    from orthonym.assembly.substituent_enumerator import _name_alkoxy_branch
    m = Chem.MolFromSmiles("OC(=O)COC1OC(CO)C(O)C(O)C1O")
    r = _name_alkoxy_branch(m, list(range(4, 16)), 4, set())
    assert r and "oxy" in r and "oxan" in r, r


def test_simple_alkoxy_unchanged():
    # Pure-alkyl ether path must be untouched: anisole still names as methoxybenzene.
    from orthonym import Orthonym
    out = Orthonym().name("COc1ccccc1")
    assert "methoxy" in out and "benzene" in out, out


def test_pyridinyloxy_names_recursively():
    # -O-(pyridin-3-yl): heteroaromatic ether R, previously declined.
    m = Chem.MolFromSmiles("OC(=O)COc1cccnc1")   # (pyridin-3-yloxy)acetic acid
    o = None
    for a in m.GetAtoms():
        if a.GetSymbol() == "O" and a.GetDegree() == 2:
            o = a.GetIdx()
    assert o is not None
    ring_nbr = [n.GetIdx() for n in m.GetAtomWithIdx(o).GetNeighbors()
                if m.GetAtomWithIdx(n.GetIdx()).GetIsAromatic()][0]
    frag = [o] + [a.GetIdx() for a in m.GetAtoms() if a.GetIsAromatic()]
    r = name_substituent(m, frag, o)
    assert r not in (None, "", "substituent"), f"declined: {r!r}"
    assert "oxy" in r and "pyridin" in r, r
