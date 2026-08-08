"""v30 sub-lever A: N-rooted secondary-amine substituent with a RING-bearing R.

`_name_amino_branch` names alkyl (methylamino) and aryl (anilino) R, but declined a
saturated-ring or ring-on-chain R, so `-NH-cyclohexyl` / `-NH-CH2Ar` fell through to
the ugly replacement name. Under the best-effort tier, recurse `name_substituent` on R
and wrap 'amino' — mirroring the O-rooted alkoxy path (`cyclohexyloxy`). PIN default is
byte-identical (the intercept gates on allow_mancude and only fires after
`_name_amino_branch` declines).
"""
import pytest
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import name_substituent


def _frag(smi):
    m = Chem.MolFromSmiles(smi)
    assert m is not None, smi
    dummy = next(a for a in m.GetAtoms() if a.GetAtomicNum() == 0)
    fv = dummy.GetNeighbors()[0].GetIdx()
    frag = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() != 0]
    return m, frag, fv


BEST_EFFORT = [
    ("[*]NC1CCCCC1", "cyclohexylamino"),
    ("[*]NCc1ccccc1", "benzylamino"),
    ("[*]NCc1ccc(F)cc1", "[(4-fluorophenyl)methyl]amino"),
    ("[*]NC1CCC(O)CC1", "(4-hydroxycyclohexyl)amino"),
    ("[*]NC1CCCC1", "cyclopentylamino"),
]


@pytest.mark.parametrize("smi,expected", BEST_EFFORT)
def test_best_effort_names_n_rooted_ring_amino(smi, expected):
    m, frag, fv = _frag(smi)
    assert name_substituent(m, frag, fv, allow_mancude=True) == expected


@pytest.mark.parametrize("smi,_expected", BEST_EFFORT)
def test_pin_default_byte_identical(smi, _expected):
    m, frag, fv = _frag(smi)
    assert name_substituent(m, frag, fv, allow_mancude=False) == "substituent"


# unchanged at BOTH tiers (existing alkyl/aryl amino path owns these)
UNCHANGED = [
    ("[*]Nc1ccccc1", "anilino"),
    ("[*]NC", "methylamino"),
]


@pytest.mark.parametrize("smi,expected", UNCHANGED)
@pytest.mark.parametrize("am", [True, False])
def test_existing_amino_unchanged(smi, expected, am):
    m, frag, fv = _frag(smi)
    assert name_substituent(m, frag, fv, allow_mancude=am) == expected


# safely deferred (must NOT fabricate a partial/wrong name): a disubstituted N and
# an acyl branch both fall through to the sentinel, never a dropped-atom name.
@pytest.mark.parametrize("smi", ["[*]N(C)C1CCCCC1", "[*]NC(=O)C1CCCCC1"])
def test_deferred_shapes_fail_closed(smi):
    m, frag, fv = _frag(smi)
    assert name_substituent(m, frag, fv, allow_mancude=True) is None


# Fable review of ff00bf1f: a ring-ASSEMBLY R (biphenyl) makes name_substituent
# return the yl-LESS parent hydride "1,1'-biphenyl"; wrapping it shipped an
# OPSIN-unparseable T4 name on previously-abstaining molecules (the 8afa533c F1
# class). The gate-independent probe re-anchor must reject it -> fail closed.
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", [
    "[*]Nc1ccc(-c2ccccc2)cc1",     # -NH-(biphenyl-4-yl)
    "[*]NCc1ccc(-c2ccccc2)cc1",    # -NH-CH2-(biphenyl-4-yl)
])
def test_ring_assembly_R_fails_closed_not_malformed(smi):
    m, frag, fv = _frag(smi)
    assert name_substituent(m, frag, fv, allow_mancude=True) is None
