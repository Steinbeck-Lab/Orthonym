""" sub-lever A: N-rooted secondary-amine substituent with a RING-bearing R.

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


# DISUBSTITUTED N with a ring-bearing R — the v2 extension (was v1-deferred). The
# ONE amino-prefix assembler orders + marks + multiplies the two R
# names; each ring R is per-branch re-anchor-verified. OPSIN RT-exact on real
# molecules (3-(cyclohexyl(methyl)amino)propanoic acid etc.).
DISUBSTITUTED = [
    ("[*]N(C)C1CCCCC1", "cyclohexyl(methyl)amino"),
    ("[*]N(C1CCCCC1)C1CCCCC1", "dicyclohexylamino"),
    ("[*]N(Cc1ccccc1)C1CCCCC1", "benzyl(cyclohexyl)amino"),
]


@pytest.mark.parametrize("smi,expected", DISUBSTITUTED)
def test_disubstituted_n_ring_amino(smi, expected):
    m, frag, fv = _frag(smi)
    assert name_substituent(m, frag, fv, allow_mancude=True) == expected


# STILL deferred (must NOT fabricate a partial/wrong name): an acyl branch is the
# amido family; a HETEROATOM-rooted branch is hydrazine/nitroso/hydroxylamine
# (a review RISK 5 — name_substituent gives OPSIN-lenient INVALID replacement names
# like '2-oxa-1-azaeth-1-en-1-yl' for -N=O that the re-anchor accepts). Both must
# fall through to their own producers, never 'amino' over them.
@pytest.mark.parametrize("smi", [
    "[*]NC(=O)C1CCCCC1",       # acyl -> amido family
    "[*]N(N=O)C1CCCCC1",       # -N=O nitroso branch
    "[*]N(N)C1CCCCC1",         # -NH2 hydrazine branch
    "[*]N(OC)C1CCCCC1",        # -O-CH3 hydroxylamine branch
])
def test_deferred_shapes_fail_closed(smi):
    m, frag, fv = _frag(smi)
    assert name_substituent(m, frag, fv, allow_mancude=True) is None


# Ring-ASSEMBLY R (biphenyl): the blocker fix routes the assembly
# substituent through name_ring_assembly_prefix, so instead of the yl-LESS parent
# hydride "1,1'-biphenyl" (OPSIN-unparseable — the 8afa533c F1 class it used to
# fail closed on) it now emits the CORRECT bracketed free-valence form.
# OPSIN RT-exact: 3-(([1,1'-biphenyl]-4-yl)amino)propanoic acid etc.
@pytest.mark.parametrize("smi,expected", [
    ("[*]Nc1ccc(-c2ccccc2)cc1", "([1,1'-biphenyl]-4-yl)amino"),
    ("[*]NCc1ccc(-c2ccccc2)cc1", "[([1,1'-biphenyl]-4-yl)methyl]amino"),
])
def test_ring_assembly_R_named_not_malformed(smi, expected):
    m, frag, fv = _frag(smi)
    out = name_substituent(m, frag, fv, allow_mancude=True)
    assert out == expected
    # the whole point of the old fail-closed guard: NEVER the yl-less hydride
    assert out != "(1,1'-biphenyl)amino"
