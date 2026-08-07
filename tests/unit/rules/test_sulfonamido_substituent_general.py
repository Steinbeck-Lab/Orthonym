"""v30 breadth (task #25 remainder) — the general-engine substituent namer must name an
N-rooted `-NH-SO2-R` branch as the P-66.1.1.4.3 `{R}sulfonamido` prefix
(methanesulfonamido / ethanesulfonamido / benzenesulfonamido / cyclohexanesulfonamido),
NOT the cascade's carbon-rooted `carbamoyl` misroot (which SWAPS S->C and DROPS S,2xO,CH3 —
a different molecule; SELF-01 suppresses it, so the whole molecule abstains).

Root cause (hetsweep, same class as the alkoxy fix ffffdab4 and the N-rooted fix 9bb3a532):
`name_substituent` normalises the attach atom to the N; `_name_amino_branch` (the builder the
N-rooted intercept delegates to) had no sulfonamido branch and returned None, so `-NH-SO2R`
fell through to the cascade -> `carbamoyl`.

Fix: a shared primitive `sulfonamido_prefix_from_n_branch` reusing the acid-stem sulfonyl
builder (`_acid_stem_oxide_prefix(..., 'sulfonyl')`) with the suffix rewrite sulfonyl->sulfonamido,
wired into `_name_amino_branch`. Fails closed (None) for substituted-arene / CF3 / N,N-disubstituted
R -> no new wrong emission (those keep abstaining). BB authority P-66.1.1.4.3, BlueBookV2.md:32995.
"""
from rdkit import Chem

from orthonym.assembly.substituent_enumerator import name_substituent
from orthonym.assembly.substituent_naming import sulfonamido_prefix_from_n_branch


# ---- positive: end-to-end through name_substituent (proves intercept -> amino_branch -> helper)

def test_methanesulfonamido():
    m = Chem.MolFromSmiles("CS(=O)(=O)Nc1ccccc1")  # C0-S1(=O2)(=O3)-N4-ring
    assert name_substituent(m, {0, 1, 2, 3, 4}, 4, allow_mancude=True) == "methanesulfonamido"


def test_ethanesulfonamido():
    m = Chem.MolFromSmiles("CCS(=O)(=O)Nc1ccccc1")
    assert name_substituent(m, {0, 1, 2, 3, 4, 5}, 5, allow_mancude=True) == "ethanesulfonamido"


def test_cyclohexanesulfonamido():
    m = Chem.MolFromSmiles("O=S(=O)(Nc1ccccc1)C2CCCCC2")  # O0=S1(=O2)(-N3)-cyclohexyl
    assert name_substituent(
        m, {0, 1, 2, 3, 10, 11, 12, 13, 14, 15}, 3, allow_mancude=True
    ) == "cyclohexanesulfonamido"


def test_benzenesulfonamido():
    m = Chem.MolFromSmiles("O=S(=O)(Nc1ccccc1)c2ccccc2")
    assert name_substituent(
        m, {0, 1, 2, 3, 10, 11, 12, 13, 14, 15}, 3, allow_mancude=True
    ) == "benzenesulfonamido"


def test_never_carbamoyl_for_methanesulfonamido():
    # the wrong-molecule / atom-drop token must never come back for -NH-SO2CH3.
    m = Chem.MolFromSmiles("CS(=O)(=O)Nc1ccccc1")
    assert name_substituent(m, {0, 1, 2, 3, 4}, 4, allow_mancude=True) != "carbamoyl"


# ---- helper-level: fail-closed guards (no new wrong emission)

def test_helper_methanesulfonamido():
    m = Chem.MolFromSmiles("CS(=O)(=O)Nc1ccccc1")
    parent = {5, 6, 7, 8, 9, 10}
    assert sulfonamido_prefix_from_n_branch(m, 4, {0, 1, 2, 3, 4}, parent) == "methanesulfonamido"


def test_helper_substituted_arene_fails_closed():
    # 4-methylbenzenesulfonyl (tosyl) R -> the acid-stem primitive returns None (locanted
    # substituted arene not built) -> fail closed, NOT a mis-named or atom-dropped prefix.
    m = Chem.MolFromSmiles("O=S(=O)(Nc1ccccc1)c2ccc(C)cc2")
    parent = {4, 5, 6, 7, 8, 9}
    frag = {0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 16}
    assert sulfonamido_prefix_from_n_branch(m, 3, frag, parent) is None


def test_helper_nn_disubstituted_fails_closed():
    # -N(CH3)-SO2CH3: N carries a second heavy substituent -> not a bare -NH-SO2R -> None.
    m = Chem.MolFromSmiles("CN(S(C)(=O)=O)c1ccccc1")  # C0-N1(-S2(-C3)(=O4)(=O5))-ring
    parent = {6, 7, 8, 9, 10, 11}
    frag = {0, 1, 2, 3, 4, 5}
    assert sulfonamido_prefix_from_n_branch(m, 1, frag, parent) is None


# ---- fable-review (b5e4d3da) self-guard holes: a SHARED primitive must fail closed on
# every non-'-NH-' attachment, since one caller guards only GetSymbol()=='N'.

def test_helper_double_bonded_n_fails_closed():
    # =N-SO2R (sulfonimidoyl-like attach): different bond order / H count from -NH-.
    m = Chem.MolFromSmiles("CS(=O)(=O)N=C1CCCCC1")  # C0-S1(=O2)(=O3)-N4=C5(ring)
    assert sulfonamido_prefix_from_n_branch(m, 4, {0, 1, 2, 3, 4}, {5, 6, 7, 8, 9, 10}) is None


def test_helper_ring_member_n_fails_closed():
    # a RING nitrogen is not an exocyclic pendant; naming it as one cuts the ring.
    m = Chem.MolFromSmiles("CS(=O)(=O)N1CCCCC1")  # C0-S1(=O2)(=O3)-N4(ring)
    assert sulfonamido_prefix_from_n_branch(m, 4, {0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}) is None


def test_helper_bridging_n_fails_closed():
    # N with TWO parent attachments is divalent -> not a monovalent prefix.
    m = Chem.MolFromSmiles("CS(=O)(=O)N(c1ccccc1)c1ccccc1")
    n_idx = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == 'N')
    frag = {0, 1, 2, 3, n_idx}
    parent = set(range(m.GetNumAtoms())) - frag
    assert sulfonamido_prefix_from_n_branch(m, n_idx, frag, parent) is None


def test_helper_charged_and_isotopic_r_fail_closed():
    # a charged / isotopically-labelled R would be spelled as the plain stem and
    # drop the label -> a different molecule.
    m1 = Chem.MolFromSmiles("[CH2-]S(=O)(=O)Nc1ccccc1")
    assert sulfonamido_prefix_from_n_branch(m1, 4, {0, 1, 2, 3, 4}, {5, 6, 7, 8, 9, 10}) is None
    m2 = Chem.MolFromSmiles("[13CH3]S(=O)(=O)Nc1ccccc1")
    assert sulfonamido_prefix_from_n_branch(m2, 4, {0, 1, 2, 3, 4}, {5, 6, 7, 8, 9, 10}) is None


# ---- regression: pre-existing N-branch names unchanged by the new branch

def test_acetamido_unchanged():
    m = Chem.MolFromSmiles("CC(=O)Nc1ccccc1")  # -NH-C(=O)CH3
    assert name_substituent(m, {0, 1, 2, 3}, 3, allow_mancude=True) == "acetamido"


def test_methylamino_unchanged():
    m = Chem.MolFromSmiles("CNc1ccccc1")
    assert name_substituent(m, {0, 1}, 1, allow_mancude=True) == "methylamino"
