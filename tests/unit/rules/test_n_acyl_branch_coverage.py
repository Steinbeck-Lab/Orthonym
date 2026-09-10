"""a review-review of 9bb3a532 — Finding 1: `_name_amino_branch`'s legacy acyl COUNT
fallback (`_pure_linear_alkyl_len` -> `{stem}anoylamino`) had NO fragment-coverage
check, unlike its strict sibling `linear_acyl_amido_prefix`. So an N carrying a
SECOND substituent besides the acyl had that substituent silently DROPPED:

  -N(CH3)C(=O)CH3 -> 'ethanoylamino' (drops N-methyl -> denotes -NH-C(=O)CH3)
  -N(CH3)CHO -> 'methanoylamino' (drops N-methyl)
  -N(C(=O)CH3)2 -> 'ethanoylamino' (drops an ENTIRE acetyl group)

The N-rooted intercept routed these fragments to `_name_amino_branch`,
making that wrong token the `name_substituent` output (previously they fell to the
cascade). Fix: the count fallback fails closed unless the N is mono-substituted
(the acyl branch is its sole heavy non-parent neighbour) -- so the builder never
mints a name for a fragment it cannot fully describe. It returns None for the
multi-substituted N; the molecule then abstains or degrades, never the dropped-atom
name. (0-wrong is enforced by the producer being honest, not by alone.)

⚙ F-amido UPDATE: the mono-acyl N-substituted case (`-N(R')-C(=O)-R`) is no
longer fail-closed -- `n_substituted_acyl_amido_prefix` now builds the CORRECT
 method-(1) prefix (`N-methylacetamido`, `N-methylformamido`), so the
N-substituent is CITED, never dropped. The two single-acyl assertions below are
flipped from `is None` to the built name (change-asserted-value; artifacts:
(1) the Blue Book `2-(N-methylpropanamido)benzene-1-sulfonic acid (PIN)`; (2) OPSIN RT
InChIKey-exact on `3-(N-methylacetamido)propanoic acid` / `3-(N-methylformamido)-
propanoic acid`; (3) mutation: reverting the F-amido builder makes `_name_amino_branch`
return None again, failing these). The DIACYL IMIDE (two acyls) STILL fails closed --
that is not the N-substituted-amido class (the builder requires exactly one acyl).
NB these are helper-level names: the full molecules `CN(C(C)=O)c1ccccc1` /
`CN(C=O)c1ccccc1` still name as the amide SUFFIX PIN (`N-methyl-N-phenylacetamide` /
`...formamide`) because the amide is their principal group (the Blue Book) -- the helper's
prefix value is only USED when a senior parent is present.
"""
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import _name_amino_branch


def _amino(smi, frag, attach, parent):
    m = Chem.MolFromSmiles(smi)
    return _name_amino_branch(m, set(frag), attach, set(parent))


def test_n_methyl_acetamido_no_dropped_prefix():
    # CN(C(C)=O)c1ccccc1: C0-N1(-C2(=O3)-C?...)-ring; frag {0,1,2,3,4}, attach N1.
    # F-amido: the N-methyl is now CITED (not dropped) -> the correct prefix.
    m = Chem.MolFromSmiles("CN(C(C)=O)c1ccccc1")
    parent = {5, 6, 7, 8, 9, 10}
    assert _name_amino_branch(m, {0, 1, 2, 3, 4}, 1, parent) == "N-methylacetamido"


def test_n_methyl_formamido_no_dropped_prefix():
    m = Chem.MolFromSmiles("CN(C=O)c1ccccc1")  # C0-N1(-C2=O3)-ring
    parent = {4, 5, 6, 7, 8, 9}
    assert _name_amino_branch(m, {0, 1, 2, 3}, 1, parent) == "N-methylformamido"


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
