""" a phase (E2) Tasks 7-11: `name_substituent_fragment` CAPABILITY fixes.

All five sites live in `assembly/substituent_naming.py::name_substituent_fragment`
(+ its helpers), per internal notes. Each was a trace-verified
directly against `name_substituent_fragment` before any code changed (a project rule:
"trace before you code"). Witness SMILES for the real CHEBI ids were re-derived from
`benchmarks/chebi_5000.csv` (the coordinator's own quoted witnesses for the sulfonamide
and ring-branch tasks did not match their targets) and each target InChIKey14 was
independently verified against RDKit before use.

- Task 7 (E2d, attachment locant): a non-terminal free valence on an ether-bearing
  chain (`1,1-dimethoxypropan-2-yl`) used to be numbered as if it were terminal
  (`1,1-dimethoxypropyl`, a locant-omission bug -- the locant distinguishes
  propan-2-yl from propan-1-yl). Fixed in `_name_ether_substituted_chain`: the
  terminal-only restriction is lifted, the backbone is now oriented from whichever
  end gives the free valence the lowest locant, mirroring
  `_located_acyclic_alkyl_name._orient`), and `_ether_chain_locants_omitted` takes
  the real attach locant (forcing citation whenever it is not 1, matching the
  existing sibling licence `_l5_substituent_prefix`'s k>=2 reading).
- Task 8 (E2b, amide inside a substituent): a secondary/tertiary amide
  `-C(=O)-N(H)(R)-` IN A CHAIN branch used to hard-decline the WHOLE fragment
  (: Pass 2's plain-N branch requires a bare -NH2). New Pass 1e in
  `_name_polyfunctional_acyclic_substituent` consumes the amide N + its own R
  branch, leaving the oxo unchanged for Pass 2, and assembles
  '{R}amino'/'[(R)amino]' via the shared `composer._assemble_decorated_amino_prefix`.
  Pantetheine itself (the coordinator's witness) is a SEPARATE, pre-existing
  composer bug away from RT-correct (see the docstring on
  `test_pantetheine_still_abstains_safely` below) -- documented as a partial per
  this batch's explicit licence ("FALL THROUGH + document it as partial").
- Task 9 (E2c, sulfonamide inside a substituent, WRONG-MOLECULE RISK): a
  `-CH2-SO2-NHCH3` arm was mis-read by `parent_to_prefix`'s generic '-amide'
  regex (which matches the literal substring 'amide' in 'methanesulfonamide' too)
  and collapsed to 'carbamoylmethyl' -- a DIFFERENT constitution (the sulfonyl
  silently became a carbonyl). Fixed two ways: (a) new Pass 1f in
  `_name_polyfunctional_acyclic_substituent` builds the correct
  '(R-sulfamoyl)'/'sulfamoyl' prefix structurally, reusing
  `rules.benzene._build_n_substituted_sulfamoyl_prefix`; (b) a hard fail-closed
  guard in `parent_to_prefix` refuses the generic '-amide' transform for any
  name ending 'sulfonamide'/'sulfinamide', so even an UNCOVERED sulfonyl shape
  can never fall back to the wrong 'carbamoyl' spelling.
- Task 10 (E2a, ring-branch ownership,): a substituent whose only
  free-standing structure is a ring reachable through a chalcogen link
  (`-S-Ar`) was declined by the generic ring chokepoint (attach_idx is the S,
  not a ring atom) and fell through to the fail-closed guard. Fixed by
  widening Step 1b's chalcogen-ether handler from `('Se', 'Te')` to include
  plain `'S'` -- `get_sulfanyl_prefix` already recurses the R side through
  `name_substituent_fragment` itself (the SAME trustworthy ring chokepoint this
  function's own Step 1c uses), so a ring-bearing R is now named correctly.
  A second, related bug surfaced and was fixed in the same edit: `parent_chain`
  is `` at one call site (`substituent_enumerator._name_substituent_cascade`
  Tier 4), which left `get_sulfanyl_prefix`'s "neither side is on the chain"
  same-size tie-break to guess -- and it guessed WRONG for `-S-CH2-C6H5` vs a
  tolyl parent (7 heavy atoms each side, 'benzylsulfanyl' vs the WRONG
  '(4-methylphenyl)sulfanyl', a different molecule). Fixed by folding the
  fragment's own `sub_atoms` boundary into the chain hint, which is always
  unambiguous by construction.
- Task 11 (E2e, N-nitroso hetero-substituent): CHEBI:138933's N(5)-substituent
  is a carbamimidoyl group `-C(=NH)-NH-N=O` attached to the PARENT via one of
  its own amino nitrogens. The whole-molecule view has THREE nitrogens on the
  amidine carbon (guanidino-shaped), but the substituent fragment (excluding
  the parent's own nitrogen) only ever holds the OTHER two -- the bare imino N
  plus the remaining amino N's own R. The recursive Tier-4 path silently
  DROPPED the nitroso and reported 'methanimidamide' (2 atoms short of the
  5-atom fragment), which 's atom-count mismatch already caught (0-wrong
  held; only breadth was missing). Fixed with two additions: a general 'nitroso'
  fragment detector (a bare -N=O substituent, -- was previously
  unsupported inside `name_substituent_fragment` even though the sibling
  `name_substituent` wrapper already had it), and a new carbamimidoyl-N-
  substituent detector (W3-P02-8) that builds 'N-{R}carbamimidoyl' structurally.
  The full ornithine `.name` integration is blocked by a SEPARATE,
  pre-existing "guanidino FG-prefix-loop double-counts a decorated amidine
  branch" bug in `rules/polyfunctional.py` (confirmed pre-existing: the SAME
  duplication appears on a plain dimethylamino-substituted analogue that never
  reaches any of this batch's new code) -- continues to catch it and
  safely abstain (0-wrong preserved). Documented as a partial per this batch's
  explicit licence.

Ordinary-substituent regression: `name_substituent_fragment` is one of the most
heavily shared functions in the tree, so a batch of unrelated, previously-correct
substituents is asserted unchanged below (methyl/ethyl/phenyl/2-hydroxyethyl/
cyclohexyl/(4-chlorophenyl)methyl).

Pre-existing failures NOT touched by this batch (verified via `scripts/an A/B check`
against `substituent_naming.py` at HEAD -- identical failures with and
without this batch's changes, so they predate this session and are out of scope):
`test_p14_3_4_task3b_baked_locants.py::test_defect_c_guards[...aspartic acid]`,
`test_sulfonamido_substituent_general.py::test_helper_acidnamer_drop_R_fails_closed`,
`test_n_substituted_amido_prefix.py::test_helper_sentinel_R_fails_closed`.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import Orthonym
from orthonym.assembly.substituent_naming import name_substituent_fragment
from orthonym.validation.opsin_roundtrip import opsin_parse


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def _full_rt(smiles: str, name: str) -> bool:
    o = opsin_parse(name)
    if not o:
        return False
    m1 = Chem.MolFromSmiles(smiles)
    m2 = Chem.MolFromSmiles(o)
    if m1 is None or m2 is None:
        return False
    return inchi.MolToInchiKey(m1) == inchi.MolToInchiKey(m2)


def _key14(smiles: str) -> str:
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))[:14]


# ===========================================================================
# Task 7 (E2d) -- non-terminal attachment locant
# ===========================================================================

def test_task7_fragment_level_attach_locant():
    """CHEBI:169513 (COC(OC)C(C)c1ccccc1, verified InChIKey14 UFOUDYPOSJJEDJ).
    The fragment attached at the middle carbon of the 3-carbon chain must be
    numbered 'propan-2-yl', not the terminal-only 'propyl'."""
    smi = "COC(OC)C(C)c1ccccc1"
    assert _key14(smi) == "UFOUDYPOSJJEDJ"
    mol = Chem.MolFromSmiles(smi)
    sub_atoms = [0, 1, 2, 3, 4, 5, 6]
    attach_idx = 5
    parent_chain = [7, 8, 9, 10, 11, 12]
    result = name_substituent_fragment(mol, sub_atoms, attach_idx, parent_chain)
    assert result == "1,1-dimethoxypropan-2-yl", result


@pytest.mark.opsin_gate
def test_task7_name_integration_and_rt(namer):
    smi = "COC(OC)C(C)c1ccccc1"
    name = namer.name(smi)
    assert name == "(1,1-dimethoxypropan-2-yl)benzene", name
    assert _full_rt(smi, name), name


# ===========================================================================
# Task 8 (E2b) -- amide inside a substituent
# ===========================================================================

def test_task8_fragment_level_pantetheine_arm():
    """Pantetheine CHEBI:16753
    (CC(C)(CO)[C@@H](O)C(=O)NCCC(=O)NCCS, verified InChIKey14 ZNXZGRMVNNHPCA).
    The amide's own N-substituent arm -CH2CH2C(=O)NHCH2CH2SH must name (not
    drop to None): the amide carbon stays on the chain (oxo at its locant) and
    the -NH-(2-sulfanylethyl) branch is the compound '[(2-sulfanylethyl)amino]'
    prefix at the SAME locant. (Alphanumerical citation ORDER of 'oxo' vs the
    bracketed compound amino prefix is a separate, pre-existing
    `naming_utils.alpha_sort_key` limitation -- it does not strip a LOCANT
    nested one bracket-level deep -- out of this file's scope; the
    CONSTITUTION asserted here is what matters and is RT-provably correct,
    see the two integration tests below.)"""
    smi = "CC(C)(CO)[C@@H](O)C(=O)NCCC(=O)NCCS"
    assert _key14(smi) == "ZNXZGRMVNNHPCA"
    mol = Chem.MolFromSmiles(smi)
    sub_atoms = [10, 11, 12, 13, 14, 15, 16, 17]
    attach_idx = 10
    parent_chain = [9, 7, 5, 1, 3]
    result = name_substituent_fragment(mol, sub_atoms, attach_idx, parent_chain)
    assert result is not None, "amide-in-substituent arm must not drop to None"
    assert "sulfanylethyl" in result and "amino" in result and "oxo" in result, result


@pytest.mark.opsin_gate
def test_task8_name_integration_and_rt(namer):
    """CHOKE-POINT-OFF-PATH guard: a witness where the SAME amide-in-substituent
    shape is reached end-to-end via `.name` (benzoic acid is senior to the
    amide, so the amide arm MUST be expressed as a chain substituent, exactly
    exercising the new Pass 1e -- unlike pantetheine, where a separate,
    pre-existing composer bug intervenes before this capability's output is
    used, see the test below)."""
    smi = "OC(=O)c1ccc(CCC(=O)NCCS)cc1"
    assert _key14(smi) == "CTUBSZMLRVDFKF"
    name = namer.name(smi)
    # 2026-09-25 (pre-existing-failures plan, Task 5) change-asserted-value: a prefix "is considered to begin with the first letter of its complete name" (the Blue Book): 'oxo' (o) is cited before
    # '[(2-sulfanylethyl)amino]' (s), and (g) (:3307) then gives oxo the lower
    # locant -- here both sit on C3. OPSIN RT exact.
    assert name == "4-{3-oxo-3-[(2-sulfanylethyl)amino]propyl}benzoic acid", name
    assert _full_rt(smi, name), name


@pytest.mark.opsin_gate
def test_pantetheine_still_abstains_safely(namer):
    """0-wrong guard for the coordinator's own witness. Pantetheine's TWO
    secondary-amide groups are bridged through a nitrogen (not a continuous
    diamide chain), so ONE must be the suffix and the other a substituent
    prefix -- this exposes a SEPARATE, pre-existing composer bug
    (`rules/polyfunctional.py`'s principal-group selection for two
    disconnected `secondary_amide` instances) that is reproducible on
    molecules with NO amide-in-substituent shape at all
    (e.g. 'CC(=O)NCCC(=O)NC' -> also 'unknown organic compound' at HEAD,
    before any of this batch's edits). FALL THROUGH here is correct per this
    batch's explicit licence: 0-wrong (safe abstention) over breadth."""
    smi = "CC(C)(CO)[C@@H](O)C(=O)NCCC(=O)NCCS"
    name = namer.name(smi)
    # Never a WRONG molecule: either it safely abstains, or (if some future
    # fix to the separate composer bug changes this) it must RT-match.
    assert name == "unknown organic compound" or _full_rt(smi, name), name


# ===========================================================================
# Task 9 (E2c) -- sulfonamide inside a substituent (WRONG-MOLECULE RISK)
# ===========================================================================

def test_task9_fragment_level_no_carbamoyl():
    """Indole CHEBI:10650 (CNS(=O)(=O)Cc1ccc2[nH]cc(CCN(C)C)c2c1, verified
    InChIKey14 KQKPFRSPSRPDEB). The -CH2-SO2-NHCH3 arm must NEVER collapse the
    sulfonyl to 'carbamoyl' (a different constitution, WRONG-MOLECULE risk)."""
    smi = "CNS(=O)(=O)Cc1ccc2[nH]cc(CCN(C)C)c2c1"
    assert _key14(smi) == "KQKPFRSPSRPDEB"
    mol = Chem.MolFromSmiles(smi)
    sub_atoms = [0, 1, 2, 3, 4, 5]
    attach_idx = 5
    parent_chain = [6, 7, 8, 9, 10, 11, 12, 18, 19]
    result = name_substituent_fragment(mol, sub_atoms, attach_idx, parent_chain)
    assert result is not None
    assert "carbamoyl" not in result, result
    assert result == "(methylsulfamoyl)methyl", result


@pytest.mark.opsin_gate
def test_task9_name_integration_no_carbamoyl_and_rt(namer):
    smi = "CNS(=O)(=O)Cc1ccc2[nH]cc(CCN(C)C)c2c1"
    name = namer.name(smi)
    assert "carbamoyl" not in name, name
    assert _full_rt(smi, name), name


# ===========================================================================
# Task 10 (E2a) -- ring-branch ownership
# ===========================================================================

def test_task10_fragment_level_ring_branch():
    """A SOLE- witness (constructed + self-verified, no CHEBI id given
    in the brief for this task): -S-(2,4-dihydroxyphenyl) attached to an
    ethyl chain. Before this fix the ring chokepoint declined (attach_idx is
    the S, not a ring atom) and the blanket ring-fragment guard
    fired -> None."""
    smi = "CCSc1ccc(O)cc1O"
    mol = Chem.MolFromSmiles(smi)
    sub_atoms = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    attach_idx = 2
    parent_chain = [1, 0]
    result = name_substituent_fragment(mol, sub_atoms, attach_idx, parent_chain)
    assert result == "(2,4-dihydroxyphenyl)sulfanyl", result


@pytest.mark.opsin_gate
def test_task10_name_integration_and_rt(namer):
    """The acid is senior to the thioether, forcing the ring-bearing
    sulfanyl arm to be a chain substituent -- exercising ring-branch
    ownership end-to-end via `.name`."""
    smi = "OC(=O)CCSc1ccc(O)cc1O"
    assert _key14(smi) == "RWVORULKTLJCIE"
    name = namer.name(smi)
    assert name == "3-[(2,4-dihydroxyphenyl)sulfanyl]propanoic acid", name
    assert _full_rt(smi, name), name


# ===========================================================================
# Task 11 (E2e) -- N-nitroso hetero-substituent
# ===========================================================================

def test_task11_fragment_level_n_nitrosocarbamimidoyl():
    """CHEBI:138933 (N=C(NCCC[C@H](N)C(=O)O)NN=O, verified InChIKey14
    DFSJTMFCAJNYBY). The N(5)-substituent arm -C(=NH)-NH-N=O, attached to the
    parent via its own amino nitrogen, must name 'N-nitrosocarbamimidoyl' --
    not drop the nitroso (the pre-fix recursive path silently lost it)."""
    smi = "N=C(NCCC[C@H](N)C(=O)O)NN=O"
    assert _key14(smi) == "DFSJTMFCAJNYBY"
    mol = Chem.MolFromSmiles(smi)
    sub_atoms = [0, 1, 11, 12, 13]
    attach_idx = 1
    parent_chain = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = name_substituent_fragment(mol, sub_atoms, attach_idx, parent_chain)
    assert result == "N-nitrosocarbamimidoyl", result


def test_task11_bare_nitroso_fragment():
    """The new general nitroso-fragment detector: a bare -N=O substituent
     fed directly to `name_substituent_fragment` (not via the sibling
    `name_substituent` wrapper, which already had this -- see
    `assembly/test_nitroso_substituent.py`)."""
    mol = Chem.MolFromSmiles("N=C(NCCC[C@H](N)C(=O)O)NN=O")
    assert name_substituent_fragment(mol, [12, 13], 12, [1, 11]) == "nitroso"


@pytest.mark.opsin_gate
def test_task11_ornithine_still_abstains_safely(namer):
    """0-wrong guard for the real ornithine witness. The full `.name`
    integration is blocked by a SEPARATE, pre-existing composer bug: the
    'guanidino' FG-prefix loop in `rules/polyfunctional.py` double-counts a
    decorated amidine branch (duplicating 'guanidino' alongside the correct
    '[(N-nitrosocarbamimidoyl)amino]' fragment this batch's fix produces).
    Reproducible on a plain dimethylamino analogue too
    (`OC(=O)CCCCNC(=N)N(C)C` -> also abstains at HEAD, unrelated to nitroso).
    FALL THROUGH here is correct per this batch's explicit licence."""
    smi = "N=C(NCCC[C@H](N)C(=O)O)NN=O"
    name = namer.name(smi)
    assert name == "unknown organic compound" or _full_rt(smi, name), name


def test_task11_e4_fail_closed_fallthrough():
    """E4: a SYNTHETIC un-nameable arm -- a tertiary (N,N-disubstituted) amino
    nitrogen on the amidine carbon -- makes the new W3-P02-8 detector decline
    (>1 amino-N substituent is out of its envelope) and `name_substituent_fragment`
    returns None for the arm. The FULL molecule then falls through with NO
    wrong emission (0-wrong preserved)."""
    mol = Chem.MolFromSmiles("CCNC(=N)N(C)C")
    sub_atoms = [3, 4, 5, 6, 7]
    attach_idx = 3
    parent_chain = [2, 1, 0]
    assert name_substituent_fragment(mol, sub_atoms, attach_idx, parent_chain) is None


@pytest.mark.opsin_gate
def test_task11_e4_whole_molecule_no_wrong_emission(namer):
    smi = "OC(=O)CCCCNC(=N)N(C)C"
    name = namer.name(smi)
    assert name == "unknown organic compound" or _full_rt(smi, name), name


# ===========================================================================
# Regression: name_substituent_fragment is heavily shared -- ordinary
# substituents must be byte-identical after all five fixes.
# ===========================================================================

@pytest.mark.parametrize(
    "smi,sub_atoms,attach_idx,parent_chain,expected",
    [
        ("CC", [1], 1, [0], "methyl"),
        ("CCC", [1, 2], 1, [0], "ethyl"),
        ("c1ccccc1C", [0, 1, 2, 3, 4, 5], 0, [6], "phenyl"),
        ("OCCC", [0, 1, 2], 2, [3], "2-hydroxyethyl"),
        ("C1CCCCC1C", [0, 1, 2, 3, 4, 5], 0, [6], "cyclohexyl"),
        ("Clc1ccc(CBr)cc1", [0, 1, 2, 3, 4, 5, 7, 8], 5, [6],
         "(4-chlorophenyl)methyl"),
    ],
)
def test_ordinary_substituents_unchanged(smi, sub_atoms, attach_idx,
                                          parent_chain, expected):
    mol = Chem.MolFromSmiles(smi)
    result = name_substituent_fragment(mol, sub_atoms, attach_idx, parent_chain)
    assert result == expected, result
