""" breadth (task #25 remainder) — the general-engine substituent namer must name an
N-rooted `-NH-SO2-R` branch as the `{R}sulfonamido` prefix
(methanesulfonamido / ethanesulfonamido / benzenesulfonamido / cyclohexanesulfonamido),
NOT the cascade's carbon-rooted `carbamoyl` misroot (which SWAPS S->C and DROPS S,2xO,CH3 —
a different molecule; suppresses it, so the whole molecule abstains).

Root cause (hetsweep, same class as the alkoxy fix ffffdab4 and the N-rooted fix 9bb3a532):
`name_substituent` normalises the attach atom to the N; `_name_amino_branch` (the builder the
N-rooted intercept delegates to) had no sulfonamido branch and returned None, so `-NH-SO2R`
fell through to the cascade -> `carbamoyl`.

Fix: a shared primitive `sulfonamido_prefix_from_n_branch` reusing the acid-stem sulfonyl
builder (`_acid_stem_oxide_prefix(..., 'sulfonyl')`) with the suffix rewrite sulfonyl->sulfonamido,
wired into `_name_amino_branch`. Fails closed (None) for substituted-arene / CF3 / N,N-disubstituted
R -> no new wrong emission (those keep abstaining). BB authority, the Blue Book.
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


def test_helper_substituted_arene_now_named():
    # ASSERTION FLIP (was None / fail-closed, task #28): a SUBSTITUTED arene R is now
    # named `{substituted-arene}-<n>-sulfonamido` -- this is the sulfa-drug scaffold and
    # a BB verbatim PIN class, the Blue Book
    # `2-(4-aminobenzene-1-sulfonamido)-1,3-thiazole-5-carboxylic acid (PIN)`).
    # The old None was a LIMITATION (`_acid_stem_unsaturated_oxide_prefix` detached only a
    # CARBON parent-side of S; a sulfonamide's other side is the N), not a correctness fact.
    # New value independently confirmed: `4-(4-methylbenzene-1-sulfonamido)benzoic acid`
    # round-trips OPSIN InChIKey-EXACT to the input (OPSIN = independent inverse engine).
    m = Chem.MolFromSmiles("O=S(=O)(Nc1ccccc1)c2ccc(C)cc2")  # -NH-SO2-(4-methylphenyl)
    parent = {4, 5, 6, 7, 8, 9}
    frag = {0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 16}
    assert sulfonamido_prefix_from_n_branch(m, 3, frag, parent) == "4-methylbenzene-1-sulfonamido"


def test_helper_sulfanilamide_arene_named():
    # the canonical sulfa-drug scaffold: R = 4-aminophenyl -> `4-aminobenzene-1-sulfonamido`
    # (BB verbatim PIN prefix, the Blue Book). RT-EXACT verified end-to-end.
    m = Chem.MolFromSmiles("Nc1ccc(S(=O)(=O)Nc2ccccc2)cc1")
    assert sulfonamido_prefix_from_n_branch(
        m, 8, {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16}, {9, 10, 11, 12, 13, 14}
    ) == "4-aminobenzene-1-sulfonamido"


def test_helper_naphthalene_arene_named():
    # a fused-arene R names via the acid-stem engine reuse -> `naphthalene-1-sulfonamido`.
    m = Chem.MolFromSmiles("O=S(=O)(Nc1ccccc1)c2cccc3ccccc23")
    n_idx = next(a.GetIdx() for a in m.GetAtoms()
                 if a.GetSymbol() == 'N')
    # parent = the plain N-phenyl; frag = everything else
    ri = m.GetRingInfo()
    benz = next(set(r) for r in ri.AtomRings()
                if len(r) == 6
                and all(m.GetAtomWithIdx(i).GetIsAromatic() for i in r)
                and any(nb.GetIdx() in r for nb in m.GetAtomWithIdx(n_idx).GetNeighbors())
                and not any(len(rr) == 6 and set(rr) != set(r) and set(rr) & set(r)
                            for rr in ri.AtomRings()))
    frag = set(range(m.GetNumAtoms())) - benz
    assert sulfonamido_prefix_from_n_branch(m, n_idx, frag, benz) == "naphthalene-1-sulfonamido"


def test_helper_competing_sulfonic_acid_R_fails_closed():
    # a review review of 7c621b84: R itself carrying a 2nd -SO3H makes the capped acid a
    # MULTIPLIED name (`ethane-1,2-disulfonic acid`); the suffix-strip then corrupts to
    # `ethane-1,2-disulfonamido` (OPSIN-unparseable wrong constitution). Guard 1 (exactly
    # one S-oxo-acid in the capped fragment) must fail closed -> the cascade names the
    # correct skeletal `...dithia-1-azahexyl` form instead.
    m = Chem.MolFromSmiles("O=S(=O)(Nc1ccccc1)CCS(O)(=O)=O")  # -NH-SO2-CH2CH2-SO3H
    n_idx = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == 'N'
                 and any(nb.GetSymbol() == 'S' for nb in a.GetNeighbors()))
    benz = next(set(r) for r in m.GetRingInfo().AtomRings()
                if len(r) == 6 and all(m.GetAtomWithIdx(i).GetIsAromatic() for i in r)
                and any(nb.GetIdx() in r for nb in m.GetAtomWithIdx(n_idx).GetNeighbors()))
    frag = set(range(m.GetNumAtoms())) - benz
    assert sulfonamido_prefix_from_n_branch(m, n_idx, frag, benz) is None


def test_helper_acidnamer_migration_R_fails_closed():
    # a review review: R = cyclohexylmethyl -> the gate-disabled acid namer mis-placed the
    # SO3H onto the ring (`methylcyclohexanesulfonic acid`), a wrong constitution. Guard 2
    # (run the acid sub-namer gate-ON) makes its own reject it -> fail closed.
    m = Chem.MolFromSmiles("O=S(=O)(Nc1ccccc1)CC2CCCCC2")  # -NH-SO2-CH2-cyclohexyl
    n_idx = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == 'N'
                 and any(nb.GetSymbol() == 'S' for nb in a.GetNeighbors()))
    benz = next(set(r) for r in m.GetRingInfo().AtomRings()
                if len(r) == 6 and all(m.GetAtomWithIdx(i).GetIsAromatic() for i in r)
                and any(nb.GetIdx() in r for nb in m.GetAtomWithIdx(n_idx).GetNeighbors()))
    frag = set(range(m.GetNumAtoms())) - benz
    assert sulfonamido_prefix_from_n_branch(m, n_idx, frag, benz) is None


def test_helper_acidnamer_drop_R_fails_closed():
    # (The method keeps its historical name so the node id is stable.) a review review: R =
    # acetamidomethyl -> the gate-disabled acid namer DROPPED the acetamido
    # (`ethanesulfonic acid`) and guard 2 (gate-ON) rejected it -> fail closed (None).
    # That drop is gone: the helper now builds the prefix with the acetamido kept,
    # 'acetamidomethanesulfonamido', the BB pattern of "Substituents of the
    # types -NH-CO-R and -NH-SO2-R" (the Blue Book; examples:33019 '3-
    # (methanesulfonamido)propanoic acid (PIN)',:33024 '(1-cyclohexylmethanesulfonamido)
    # acetic acid'). change-asserted-value: the new value is checked independently below
    # (OPSIN reads '(acetamidomethanesulfonamido)benzene' back to the exact input, so
    # nothing is dropped -- the very property the old guard enforced), a capability gain
    # like test_helper_hetarene_now_names_after_v51_r2 below, not a relaxation.
    m = Chem.MolFromSmiles("O=S(=O)(Nc1ccccc1)CNC(C)=O")  # -NH-SO2-CH2-NHC(=O)CH3
    n_idx = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == 'N'
                 and any(nb.GetSymbol() == 'S' for nb in a.GetNeighbors()))
    benz = next(set(r) for r in m.GetRingInfo().AtomRings()
                if len(r) == 6 and all(m.GetAtomWithIdx(i).GetIsAromatic() for i in r)
                and any(nb.GetIdx() in r for nb in m.GetAtomWithIdx(n_idx).GetNeighbors()))
    frag = set(range(m.GetNumAtoms())) - benz
    prefix = sulfonamido_prefix_from_n_branch(m, n_idx, frag, benz)
    assert prefix == "acetamidomethanesulfonamido"
    from rdkit.Chem import inchi
    from orthonym.validation.opsin_roundtrip import opsin_parse
    parsed = opsin_parse(f"({prefix})benzene")
    assert parsed and (inchi.MolToInchiKey(Chem.MolFromSmiles(parsed))
                       == inchi.MolToInchiKey(m))


def test_helper_hetarene_now_names_after_v51_r2():
    # v51 R2, Table 6.2): the acid namer used to return `unknown organic
    # compound` for `pyridine-3-sulfonic acid`, so this stem builder failed closed
    # (-> None). R2 wired the sulfonic-acid suffix onto a heterocyclic ring parent,
    # so `pyridine-3-sulfonic acid` now names and the helper correctly builds the
    # `pyridine-3-sulfonamido` prefix. change-asserted-value: the new value is
    # OPSIN-verified below to describe the exact frag (no atom drop / wrong
    # constitution), so this is a capability gain, not a stale relaxation.
    m = Chem.MolFromSmiles("O=S(=O)(Nc1ccccc1)c2cccnc2")  # -NH-SO2-(pyridin-3-yl)
    n_idx = next(a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == 'N'
                 and any(nb.GetSymbol() == 'S' for nb in a.GetNeighbors()))
    ri = m.GetRingInfo()
    benz = next(set(r) for r in ri.AtomRings()
                if len(r) == 6
                and all(m.GetAtomWithIdx(i).GetSymbol() == 'C' for i in r)
                and any(nb.GetIdx() in r for nb in m.GetAtomWithIdx(n_idx).GetNeighbors()))
    frag = set(range(m.GetNumAtoms())) - benz
    prefix = sulfonamido_prefix_from_n_branch(m, n_idx, frag, benz)
    assert prefix == "pyridine-3-sulfonamido"
    # Independent check (OPSIN, not the code under test): the prefix on benzene
    # parses back to the exact input structure (0-wrong).
    from rdkit.Chem import inchi
    from orthonym.validation.opsin_roundtrip import opsin_parse
    parsed = opsin_parse(f"({prefix})benzene")
    assert parsed and (inchi.MolToInchiKey(Chem.MolFromSmiles(parsed))
                       == inchi.MolToInchiKey(m))


def test_helper_nn_disubstituted_fails_closed():
    # -N(CH3)-SO2CH3: N carries a second heavy substituent -> not a bare -NH-SO2R -> None.
    m = Chem.MolFromSmiles("CN(S(C)(=O)=O)c1ccccc1")  # C0-N1(-S2(-C3)(=O4)(=O5))-ring
    parent = {6, 7, 8, 9, 10, 11}
    frag = {0, 1, 2, 3, 4, 5}
    assert sulfonamido_prefix_from_n_branch(m, 1, frag, parent) is None


# ---- a review-review self-guard holes: a SHARED primitive must fail closed on
# every non-'-NH-' attachment, since one caller guards only GetSymbol=='N'.

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
