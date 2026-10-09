""" sub-lever A: N-rooted secondary-amine substituent with a RING-bearing R.

`_name_amino_branch` names alkyl (methylamino) and aryl (anilino) R, but declined a
saturated-ring or ring-on-chain R, so `-NH-cyclohexyl` / `-NH-CH2Ar` fell through to
the ugly replacement name. Under the best-effort tier, recurse `name_substituent` on R
and wrap 'amino' — mirroring the O-rooted alkoxy path (`cyclohexyloxy`). That intercept
gates on allow_mancude and only fires after `_name_amino_branch` declines; at the PIN
tier the -NH- of an undecorated ring-yl is the bare connective of `ring_substituents`
('cyclohexylamino',, and every other row still declines there.
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


#: An -NH- joining an undecorated ring-yl to the parent is the compound prefix '(R)amino' at the
#: PIN tier too, the Blue Book "Preferred IUPAC names for prefixes
#: corresponding to -NHR, -NRR', or -NR2 are formed by prefixing the names of the groups R and
#: R' to the prefix 'amino'"; the bare connective of ``ring_substituents``). The other rows keep
#: the PIN-tier decline: a carbon carrier, or a ring that carries a group of its own.
PIN_TIER_NAMED = {"[*]NC1CCCCC1", "[*]NC1CCCC1"}


@pytest.mark.parametrize("smi,expected", BEST_EFFORT)
def test_pin_default_tier(smi, expected):
    m, frag, fv = _frag(smi)
    want = expected if smi in PIN_TIER_NAMED else "substituent"
    assert name_substituent(m, frag, fv, allow_mancude=False) == want


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
# j7 (TRIAGE g7 C09): the three heteroatom-rooted branches come back as skeletal-
# replacement chains that END on N or O ('1,2-diazaethyl', '3-oxa-1,2-diazaprop-2-
# en-1-yl', '2-oxa-1-azapropyl'), which (the Blue Book, "The chain
# must be terminated by a C atom or one of... P, As, Sb, Bi, Si, Ge, Sn, Pb, B, Al,
# Ga, In, or Tl") does not allow. Strict xfail until the producers cover them.
_C09_REASON = ("TRIAGE g7 C09 (j7, measured): the last-resort rules.terminal_fragment names heteroatom-rooted / heteroatom-terminated branches as skeletal-replacement chains that end on N/O/S ('1,2-diazaethyl', '2-oxa-1-azapropyl', '2-oxaethyl', '6-oxahex-5-en-1-yl'), which P-15.4.3.1 (BlueBookV2.md:6465) does not allow. Refusing them there alone (tried: both ends, trimmed far end, free-valence end only) reroutes 44-71 m1500 best-effort names through the universal floor with worse spellings and drops 2 PIN-tier names; the fix needs the chain composer / amino-oxy-hydrazinyl producers to cover these branches first. TRIAGE.md 'Suite fix -- j7-defects-misc', remaining.")


@pytest.mark.parametrize("smi", [
    pytest.param("[*]N(N=O)C1CCCCC1", marks=pytest.mark.xfail(strict=True, reason=_C09_REASON)),
    pytest.param("[*]N(N)C1CCCCC1", marks=pytest.mark.xfail(strict=True, reason=_C09_REASON)),
    pytest.param("[*]N(OC)C1CCCCC1", marks=pytest.mark.xfail(strict=True, reason=_C09_REASON)),
])
def test_deferred_shapes_fail_closed(smi):
    m, frag, fv = _frag(smi)
    assert name_substituent(m, frag, fv, allow_mancude=True) is None


def test_acyl_branch_takes_the_amido_prefix():
    """The acyl branch (formerly in the deferred list) is named by the amido
    family's own producer: 'cyclohexanecarboxamido' method (1),
    the Blue Book "changing the suffixes 'amide' and 'carboxamide' into
    'amido' and 'carboxamido'"); '3-(cyclohexanecarboxamido)propanoic acid' is
    OPSIN full-InChIKey exact. j7, TRIAGE g7 C09 (stale expectation)."""
    m, frag, fv = _frag("[*]NC(=O)C1CCCCC1")
    assert name_substituent(m, frag, fv, allow_mancude=True) == "cyclohexanecarboxamido"


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
