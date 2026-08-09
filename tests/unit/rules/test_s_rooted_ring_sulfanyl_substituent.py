"""v30 sub-lever A: S-rooted thioether substituent with a RING-bearing R.

The chalcogen-rooted sulfanyl cascade tier declines a saturated-ring / ring-on-chain
R, so -S-cyclohexyl / -S-CH2Ar fell through to the ugly replacement name. Under the
best-effort tier recurse name_substituent on R and wrap 'sulfanyl' -- mirroring the
O-rooted alkoxy (cyclohexyloxy) and N-rooted amino (cyclohexylamino) ring paths. PIN
default byte-identical (best-effort only). Gate-independent probe re-anchor rejects a
yl-less ring-assembly R (the ed52fa98 / 8afa533c F1 class).
"""
import pytest
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import name_substituent


def _sulfanyl_frag(mol_smi):
    """Build (mol, frag_atoms, attach_S) for a -S-R substituent from a WHOLE
    molecule whose S bridges a REAL parent atom and R. A `[*]` dummy parent breaks
    the helper's CH3-S-R probe (RDKit valence artifact at the wildcard bond), so
    the substituent path must be exercised with a real parent atom."""
    m = Chem.MolFromSmiles(mol_smi)
    assert m is not None, mol_smi
    # the divalent bridging S: neutral, degree 2, both single bonds
    S = next(a.GetIdx() for a in m.GetAtoms()
             if a.GetSymbol() == 'S' and a.GetDegree() == 2
             and a.GetFormalCharge() == 0)
    # R side = the S neighbour that is NOT the aromatic-parent ring anchor; grab
    # the whole substituent fragment by BFS from S excluding the parent side.
    nbrs = [n.GetIdx() for n in m.GetAtomWithIdx(S).GetNeighbors()]
    # parent side = the neighbour in the largest aromatic ring (the tolyl anchor)
    ri = m.GetRingInfo()
    parent = next(n for n in nbrs if ri.NumAtomRings(n) > 0
                  and m.GetAtomWithIdx(n).GetIsAromatic())
    rside = next(n for n in nbrs if n != parent)
    frag = {S}
    stack = [rside]
    while stack:
        a = stack.pop()
        if a in frag:
            continue
        frag.add(a)
        for nb in m.GetAtomWithIdx(a).GetNeighbors():
            ni = nb.GetIdx()
            if ni != S and ni not in frag and ni != parent:
                stack.append(ni)
    return m, sorted(frag), S


# whole-molecule SMILES (S bridges a tolyl parent and R), expected -S-R prefix.
BEST_EFFORT = [
    ("Cc1ccc(SC2CCCCC2)cc1", "cyclohexylsulfanyl"),
    ("Cc1ccc(SCc2ccccc2)cc1", "benzylsulfanyl"),
    ("Cc1ccc(Sc2ccccc2)cc1", "phenylsulfanyl"),
    ("Cc1ccc(SC2CCCC2)cc1", "cyclopentylsulfanyl"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("mol_smi,expected", BEST_EFFORT)
def test_best_effort_names_s_rooted_ring_sulfanyl(mol_smi, expected):
    m, frag, S = _sulfanyl_frag(mol_smi)
    assert name_substituent(m, frag, S, allow_mancude=True) == expected


@pytest.mark.parametrize("mol_smi,_expected", BEST_EFFORT)
def test_pin_default_byte_identical(mol_smi, _expected):
    # PIN default keeps the historical sentinel (best-effort-only scope).
    m, frag, S = _sulfanyl_frag(mol_smi)
    assert name_substituent(m, frag, S, allow_mancude=False) == "substituent"


# a yl-less ring-ASSEMBLY R (biphenyl) must fail closed in THIS producer -- the
# probe re-anchor rejects it (the wrong-molecule leak stays out of the helper).
@pytest.mark.opsin_gate
def test_ring_assembly_R_fails_closed_in_this_producer():
    import orthonym.assembly.substituent_enumerator as SE
    m, frag, S = _sulfanyl_frag("Cc1ccc(SCc2ccc(-c3ccccc3)cc2)cc1")
    assert SE._name_thio_ring_branch(m, set(frag), S) is None

