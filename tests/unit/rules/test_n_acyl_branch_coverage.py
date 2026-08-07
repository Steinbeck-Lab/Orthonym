"""fable-review of 9bb3a532 — Finding 1: `_name_amino_branch`'s legacy acyl COUNT
fallback (`_pure_linear_alkyl_len` -> `{stem}anoylamino`) had NO fragment-coverage
check, unlike its strict sibling `linear_acyl_amido_prefix`. So an N carrying a
SECOND substituent besides the acyl had that substituent silently DROPPED:

  -N(CH3)C(=O)CH3 -> 'ethanoylamino'   (drops N-methyl -> denotes -NH-C(=O)CH3)
  -N(CH3)CHO      -> 'methanoylamino'  (drops N-methyl)
  -N(C(=O)CH3)2   -> 'ethanoylamino'   (drops an ENTIRE acetyl group)

The N-rooted intercept (9bb3a532) routed these fragments to `_name_amino_branch`,
making that wrong token the `name_substituent` output (previously they fell to the
cascade). Fix: the count fallback fails closed unless the N is mono-substituted
(the acyl branch is its sole heavy non-parent neighbour) -- so the builder never
mints a name for a fragment it cannot fully describe. It returns None for the
multi-substituted N; the molecule then abstains or degrades, never the dropped-atom
name. (0-wrong is enforced by the producer being honest, not by SELF-01 alone.)
"""
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import _name_amino_branch


def _amino(smi, frag, attach, parent):
    m = Chem.MolFromSmiles(smi)
    return _name_amino_branch(m, set(frag), attach, set(parent))


def test_n_methyl_acetamido_no_dropped_prefix():
    # CN(C(C)=O)c1ccccc1 : C0-N1(-C2(=O3)-C? ...)-ring ; frag {0,1,2,3,4}, attach N1
    m = Chem.MolFromSmiles("CN(C(C)=O)c1ccccc1")
    parent = {5, 6, 7, 8, 9, 10}
    assert _name_amino_branch(m, {0, 1, 2, 3, 4}, 1, parent) is None


def test_n_methyl_formamido_no_dropped_prefix():
    m = Chem.MolFromSmiles("CN(C=O)c1ccccc1")  # C0-N1(-C2=O3)-ring
    parent = {4, 5, 6, 7, 8, 9}
    assert _name_amino_branch(m, {0, 1, 2, 3}, 1, parent) is None


def test_diacyl_imide_no_dropped_prefix():
    m = Chem.MolFromSmiles("CC(=O)N(C(C)=O)c1ccccc1")  # (CH3CO)2N- imide
    parent = {6, 7, 8, 9, 10, 11}
    assert _name_amino_branch(m, {0, 1, 2, 3, 4, 5}, 3, parent) is None


def test_plain_acetamido_still_named():
    # a genuine mono-substituted -NH-C(=O)CH3 must still name (regression guard).
    m = Chem.MolFromSmiles("CC(=O)Nc1ccccc1")  # C0-C1(=O2)-N3-ring
    assert _name_amino_branch(m, {0, 1, 2, 3}, 3, {4, 5, 6, 7, 8, 9}) == "acetamido"


def test_longer_linear_acylamino_still_named():
    # -NH-CO-C3H7 (butanamido) via the count path, mono-substituted -> still named.
    m = Chem.MolFromSmiles("CCCC(=O)Nc1ccccc1")  # butanoyl-NH-phenyl
    got = _name_amino_branch(m, {0, 1, 2, 3, 4, 5}, 5, {6, 7, 8, 9, 10, 11})
    assert got == "butanamido"
