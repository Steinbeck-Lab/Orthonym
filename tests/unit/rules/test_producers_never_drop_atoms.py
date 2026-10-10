"""A PIN-path producer never offers a name that drops or misreads atoms.

Each class below was found by naming a real molecule with the OPSIN validity gate OFF
(the suite default, so a producer's raw output is what the test sees) and reading the
name back through OPSIN: the producer shipped a name for a DIFFERENT molecule, which in
production only the round-trip gate stopped. Each test states the producer-level
invariant (a producer returns the right name or declines), not a spelling, and checks the
invariant on the molecules the class was measured on plus controls it must keep naming.

  * ``name_polyfunctional`` -- (a) a suffix is cited only for principal-group instances on
    the principal chain, the Blue Book, under "SENIORITY ORDER FOR
    PARENT STRUCTURES"); (b) a substituent branch is skipped as "named by the functional-
    group loop" only when that loop emitted a prefix for it; (c) a carbon-free branch no
    functional group covers refuses the handler instead of vanishing; (d) the same-class
    alcohols of a re-selected principal group leave the prefix loop.
  * ``composer._check_for_acylamino`` -- an acyl whose amide N carries a further substituent
    is not spelled from the acyl alone.
  * ``composer._amide_acyl_parent_locants`` -- a ring carbon is never an acyl-chain atom.
  * ``polycyclic.get_polycyclic_substituents`` -- a branch is named by its carbon count only
    when it is an unbranched saturated alkyl.
  * ``composer._merge_duplicate_prefixes`` -- the merged prefix carries its atoms.
"""
import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.errors import is_failure_name
from tests.support.rt_assert import name_is_rt_exact


def _ships_only_exact(smiles):
    """name_compound (gate off, as the suite runs it) is a decline or round-trips."""
    name = name_compound(smiles)
    assert is_failure_name(name) or name_is_rt_exact(name, smiles), (
        f"{smiles}: {name!r} is not the input molecule")
    return name


# ---------------------------------------------------------------------------
# name_polyfunctional
# ---------------------------------------------------------------------------

#: molecules on which name_polyfunctional (called from name_ester_family) used to return
#: a name for a different molecule: an '-ol' suffix for OH groups on no chain atom
#: (phosphatidylinositol, histidyl adenylate), a thioether / ether / ester branch skipped
#: although no prefix named it (acetylcysteine thioether, N-acetylmuramic acid, a malonate
#: half-ester of a macrolide), an acyl sulfate whose =O no group covers, a secondary OH
#: cited both as 'hydroxy' and in '-1,4-diol'
POLYFUNCTIONAL_WITNESSES = [
    # the six OH are the inositol's, on no atom of the C20 acyl chain ('...trienehexol')
    "CCCCC/C=C\\C/C=C\\C/C=C\\CCCCCCC(=O)OC[C@@H](O)COP(=O)(O)OC1C(O)C(O)C(O)[C@@H](O)C1O",
    # the '-ol' of 'propanol' is the ester O of the adenosine phosphate
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](OC(=O)[C@@H](N)Cc2c[nH]cn2)[C@H]1O",
    # '2-oxobutanoic acid': the S-(2-acetamido-2-carboxyethyl) branch lost
    "CC(=O)N[C@@H](CSCCC(=O)C(=O)O)C(=O)O",
    # '(2R)-propanoic acid': the glucosaminyl ether branch lost
    "CC(=O)N[C@H]1C(O)O[C@H](CO)[C@@H](O)[C@@H]1O[C@H](C)C(=O)O",
    # 'propanoic acid': the macrolide O-substituent of the malonate half-ester lost
    ("CN=C(N)NCCC/C=C/CCC[C@H](C)[C@H]1OC(=O)/C(C)=C\\C=C/[C@H](C)[C@H](O)C[C@H](O)"
     "[C@H](C)[C@@H](O)CC[C@@H](C)[C@H](O)C[C@@]2(O)O[C@H](C[C@H](O)C[C@H](OC(=O)CC(=O)O)"
     "C[C@@H](O)C[C@H](O)/C(C)=C\\C=C/[C@H]1C)C[C@@H](O)[C@@H]2O"),
    # '2-(sulfooxy)ethan-1-ol': the acyl C=O is covered by no group
    "OCC(=O)OS(=O)(=O)O",
    # '2-(acetyloxy)-4-hydroxyheptadec-16-yne-1,4-diol': the 4-OH cited twice
    "C#CCCCCCCCCCCCC(O)CC(CO)OC(C)=O",
    # '3-hydroxy-4-sulfonylbutanoic acid' / '...-4-sulfinylbutanoic acid': the table form of a
    # sulfone / sulfoxide prefix names the S=O core only, the carboxymethyl on the S is lost
    "OC(=O)CC(O)CS(=O)(=O)CC(=O)O",
    "OC(=O)CC(O)CS(=O)CC(=O)O",
]

#: controls name_polyfunctional must keep naming exactly (no stereo, so the producer's
#: own name is the whole name)
POLYFUNCTIONAL_CONTROLS = {
    "OC(=O)CC(O)CSC(C)(C)C": "4-(tert-butylsulfanyl)-3-hydroxybutanoic acid",
    "OC(=O)CC(O)COC(C)(C)C": "4-tert-butoxy-3-hydroxybutanoic acid",
    "OC(=O)CC(O)CCl": "4-chloro-3-hydroxybutanoic acid",
    "OC(=O)CC(O)CSCC(=O)O": "4-[(carboxymethyl)sulfanyl]-3-hydroxybutanoic acid",
    "OC(=O)CC(O)COc1ccccc1": "3-hydroxy-4-phenoxybutanoic acid",
    # the suffix group is -P(O)O: its match holds no chain atom, it is bonded to one
    # (the 'suffix groups sit on the parent' decline must not read it as off-chain)
    "C(C(=O)NCP(O)O)N": "(2-aminoacetamido)methanephosphonous acid",
    # a sulfone whose substituent the dispatcher names keeps its prefix
    "OC(=O)CC(O)CS(=O)(=O)C": "3-hydroxy-4-(methanesulfonyl)butanoic acid",
}


@pytest.fixture
def poly_calls(monkeypatch):
    """Record every (input SMILES, returned name) of rules.polyfunctional.name_polyfunctional
    (the callers import it inside their function bodies, so the patched object is seen)."""
    import orthonym.rules.polyfunctional as P
    calls = []
    orig = P.name_polyfunctional

    def spy(features):
        out = orig(features)
        calls.append((features.canonical_smiles, out))
        return out

    monkeypatch.setattr(P, "name_polyfunctional", spy)
    return calls


@pytest.mark.parametrize("smiles", POLYFUNCTIONAL_WITNESSES)
def test_polyfunctional_never_returns_a_name_for_another_molecule(poly_calls, smiles):
    # drive the pipeline; what is asserted is the PRODUCER's return, not the final name
    # (a later producer may still be wrong with the gate off: that is its own defect)
    name_compound(smiles)
    named = [(smi, nm) for smi, nm in poly_calls if nm]
    for call_smiles, name in named:
        assert name_is_rt_exact(name, call_smiles), (
            f"name_polyfunctional({call_smiles}) returned {name!r}, which is not that molecule")


@pytest.mark.parametrize("smiles,expected", sorted(POLYFUNCTIONAL_CONTROLS.items()))
def test_polyfunctional_controls_still_named(poly_calls, smiles, expected):
    assert name_compound(smiles) == expected
    assert any(n == expected for _s, n in poly_calls), (
        "the name no longer comes from name_polyfunctional")


def test_polyfunctional_union_class_alcohol_not_cited_twice():
    """ (the Blue Book): both OH of the diol are the suffix; the ester-demoted
    path used to cite the secondary OH again as '4-hydroxy' (a gem-diol to OPSIN)."""
    smiles = "C#CCCCCCCCCCCCC(O)CC(CO)OC(C)=O"
    name = name_compound(smiles)
    assert "hydroxy" not in name
    assert name == "2-(acetyloxy)heptadec-16-yne-1,4-diol"
    assert name_is_rt_exact(name, smiles)


# ---------------------------------------------------------------------------
# composer._check_for_acylamino
# ---------------------------------------------------------------------------

def _acylamino(smiles, branch, chain):
    from orthonym.assembly.composer import _check_for_acylamino
    mol = Chem.MolFromSmiles(smiles)
    return _check_for_acylamino(mol, branch, chain)


def test_acylamino_keeps_nh_acetamido():
    # CC(=O)N[C@@H](C)C(=O)O: atoms 0 C, 1 C, 2 O, 3 N | 4 CH, 5 C, 6 C, 7 O, 8 O
    assert _acylamino("CC(=O)NC(C)C(=O)O", [0, 1, 2, 3], [4, 5, 6]) == "acetamido"


def test_acylamino_spells_a_substituted_nitrogen_as_method_1_amido():
    # CC(=O)N(C)C(C)C(=O)O: the N-methyl (atom 4) is in no acyl atom; 'ethanoylamino' lost it.
    # (the Blue Book) method (1): 'N-methylacetamido', never the acyl alone.
    assert _acylamino("CC(=O)N(C)C(C)C(=O)O", [0, 1, 2, 3, 4], [5, 6, 7]) == "(N-methylacetamido)"


@pytest.mark.parametrize("smiles", [
    "CC(=O)N(C)[C@@H](C)C(=O)O",
    "CC(C)C[C@H](N(C)C(=O)[C@@H](C)NC(=O)c1ccccc1)C(=O)O",
    "CCC(=O)N(CC)CC(=O)O",
    "O=C(c1ccccc1)N(C)CC(=O)O",
])
def test_n_substituted_acylamino_molecules_ship_only_exact_names(smiles):
    name = _ships_only_exact(smiles)
    assert not is_failure_name(name)  # the unmasked producers name these exactly


# ---------------------------------------------------------------------------
# composer._amide_acyl_parent_locants / _merge_duplicate_prefixes
# ---------------------------------------------------------------------------

def test_acyl_parent_locants_never_number_a_ring_carbon():
    from orthonym.assembly.composer import _amide_acyl_parent_locants
    smiles = ("CCCCN(C)C(=O)CCCCCCCCCC[C@@H]1Cc2cc(O)ccc2[C@H]2CC[C@]3(C)[C@@H](O)CC[C@H]3"
              "[C@H]12")
    mol = Chem.MolFromSmiles(smiles)
    amide = mol.GetSubstructMatch(Chem.MolFromSmarts("[CX3](=O)N"))
    locants = _amide_acyl_parent_locants(mol, tuple(amide))
    assert locants, "the acyl chain has locants"
    assert not any(mol.GetAtomWithIdx(a).IsInRing() for a in locants)
    assert sorted(locants.values()) == list(range(1, 12))  # the 11 carbons of undecanamide


def test_amide_with_ring_on_the_chain_is_not_named_from_the_chain_alone():
    smiles = ("CCCCN(C)C(=O)CCCCCCCCCC[C@@H]1Cc2cc(O)ccc2[C@H]2CC[C@]3(C)[C@@H](O)CC[C@H]3"
              "[C@H]12")
    name = _ships_only_exact(smiles)
    assert name != "(12R,13R,14S,14S,15S,16S)-N-butyl-16,16-dihydroxy-N-methylundecanamide"


def test_merged_duplicate_prefix_carries_the_union_of_its_atoms():
    from orthonym.assembly.composer import NameFragment, _merge_duplicate_prefixes
    a = NameFragment(text="1-hydroxy", locants=(1,), fragment_type="prefix",
                     atoms=frozenset({1, 2}))
    b = NameFragment(text="4-hydroxy", locants=(4,), fragment_type="prefix",
                     atoms=frozenset({7, 8}))
    merged = _merge_duplicate_prefixes([a, b])
    assert len(merged) == 1 and merged[0].atoms == frozenset({1, 2, 7, 8})
    c = NameFragment(text="4-hydroxy", locants=(4,), fragment_type="prefix", atoms=None)
    merged = _merge_duplicate_prefixes([a, c])
    assert len(merged) == 1 and merged[0].atoms is None  # unreported stays unreported


def test_unnameable_branch_voids_the_candidate_not_the_branch():
    """The composer used to skip a branch it could not name and ship the rest."""
    smiles = "C/C=C/C=C/C(=O)CC(CCO)OCC[C@H](O)CC(=O)CC/C=C/C"
    name = _ships_only_exact(smiles)
    assert name != "(6E,8E)-1-hydroxydeca-6,8-dien-5-one"


# ---------------------------------------------------------------------------
# polycyclic.get_polycyclic_substituents
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles", [
    "CC(=O)OC12CC3CC(CC(C3)C1)C2",        # acetyloxy was 'ethoxy' (C=O lost)
    "CC(C)OC12CC3CC(CC(C3)C1)C2",         # propan-2-yloxy was 'propoxy'
    "OOC12CC3CC(CC(C3)C1)C2",             # hydroperoxy vanished
    "ClC12CC3CC(CC(C3)C1)C2",             # chloro vanished
    "O=[N+]([O-])C12CC3CC(CC(C3)C1)C2",   # nitro vanished
])
def test_cage_producer_declines_a_branch_it_cannot_spell(smiles):
    from orthonym.rules.polycyclic import name_polycyclic_complete
    assert name_polycyclic_complete(Chem.MolFromSmiles(smiles)) is None


@pytest.mark.parametrize("smiles,expected", [
    ("CCOC12CC3CC(CC(C3)C1)C2", "1-ethoxytricyclo[3.3.1.1^3,7]decane"),
    ("NC12CC3CC(CC(C3)C1)C2", "tricyclo[3.3.1.1^3,7]decan-1-amine"),
    ("C=CC12CC3CC(CC(C3)C1)C2", "1-ethenyltricyclo[3.3.1.1^3,7]decane"),
])
def test_cage_producer_keeps_naming_the_branches_it_can_spell(smiles, expected):
    from orthonym.rules.polycyclic import name_polycyclic_complete
    result = name_polycyclic_complete(Chem.MolFromSmiles(smiles))
    assert result is not None and result[0] == expected
    assert name_is_rt_exact(expected, smiles)


# ---------------------------------------------------------------------------
# composer._check_for_acylamino: method (1) amido prefix with an N-substituent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("CC(=O)N(C)CC(=O)O", "(N-methylacetamido)acetic acid"),
    ("CC(=O)N(C)C(C)C(=O)O", "2-(N-methylacetamido)propanoic acid"),
    ("CCC(=O)N(CC)CC(=O)O", "(N-ethylpropanamido)acetic acid"),
    ("CC(=O)N(CCO)CC(=O)O", "[N-(2-hydroxyethyl)acetamido]acetic acid"),
    ("O=CN(C)CC(=O)O", "(N-methylformamido)acetic acid"),
])
def test_n_substituted_acylamino_prefix_is_the_method_1_amido(smiles, expected):
    """ 'Substituents of the types -NH-CO-R and -NH-SO2-R' (the Blue Book):
    'Method (1) generates preferred IUPAC names.' (:32997); '2-(N-methylpropanamido)benzene-1-
    sulfonic acid (PIN)' (:33039). Never the method-(2) '(acetyl)(methyl)amino' spelling."""
    assert name_compound(smiles) == expected
    assert name_is_rt_exact(expected, smiles)


# the default tier (gate ON, emission rule applied) for an N-substituted acylamino branch:
# the method (1) name is the PIN, and what the builder cannot spell is declined, never a
# method-(2) spelling labelled PIN (the reviewer's B1: '[(acetyl)(methyl)amino]acetic acid'
# shipped as pin_verified when this producer merely declined the acyl-only spelling)
N_SUBSTITUTED_ACYLAMINO_PIN = {
    "CC(=O)N(C)CC(=O)O": "(N-methylacetamido)acetic acid",
    "CC(=O)N(C)C(C)C(=O)O": "2-(N-methylacetamido)propanoic acid",
    "CC(C)CC(N(C)C(=O)C)C(=O)O": "4-methyl-2-(N-methylacetamido)pentanoic acid",
    "CC(=O)N(C)CCC(=O)O": "3-(N-methylacetamido)propanoic acid",
}


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", sorted(N_SUBSTITUTED_ACYLAMINO_PIN.items()))
def test_default_tier_ships_the_method_1_pin_for_an_n_substituted_acylamino(smiles, expected):
    """ (the Blue Book) 'Method (1) generates preferred IUPAC names.' (:32997),
    example '2-(N-methylpropanamido)benzene-1-sulfonic acid (PIN)' (:33039)."""
    from tests.support.default_tier import default_tier_row
    row = default_tier_row(smiles)
    assert row["is_pin"] is True and row["name"] == expected, row
    assert "(acetyl)" not in row["name"]          # never the method-(2) acylamino spelling
    assert name_is_rt_exact(expected, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", [
    "CC(C)C(=O)N(C)CC(=O)O",      # branched acyl
    "O=C(c1ccccc1)N(C)CC(=O)O",   # ring acyl
    "C=CC(=O)N(C)CC(=O)O",        # unsaturated acyl
])
def test_default_tier_declines_an_n_substituted_acyl_it_cannot_spell_as_method_1(smiles):
    """The strict amido builder spells only an unbranched saturated acyl; for the rest the
    default tier declines (NO_VERIFIED_PIN) and best-effort names the molecule exactly."""
    from tests.support.default_tier import assert_default_tier_declines
    from tests.support.rt_assert import name_best_effort
    assert_default_tier_declines(smiles)
    be = name_best_effort(smiles)
    assert not is_failure_name(be["name"]) and name_is_rt_exact(be["name"], smiles), be


# ---------------------------------------------------------------------------
# open: S-conjugate branch named inside name_polyfunctional (reviewer MAJOR 2)
# ---------------------------------------------------------------------------

# change-asserted-value (leads program L6, item 43a, merged in 89b7abd72, b80c73f52 and the
# 263d27eaa acylsulfanyl hunk): this was a strict xfail. The cause recorded there (the
# S-branch 'CC(=O)N[C@@H](C[*])C(=O)O' named as '' inside name_polyfunctional, so
# get_sulfanyl_prefix returned None and the handler was refused) is gone: the decomposition
# fallback now cites an acylated non-suffix nitrogen as the acylamino prefix instead of gluing
# 'N-acetyl<systematic name>', and a substituted acyl on sulfur is an enclosed compound prefix.
# (the Blue Book) 'Method (1) generates preferred IUPAC names.' (:32998)
# -> the 'acetamido' prefix; (:7232) "Parentheses are used around compound...
# prefixes" -> '[(2,3-dihydroxypropanoyl)sulfanyl]'; (:17758,:17764) 'acyl groups,
# full substitution allowed'.
# Names now (both read back by OPSIN 2.9.0 to the input's full InChIKey, independently of the
# engine): 'CC(=O)N[C@@H](CSC(=O)C(O)CO)C(=O)O' is
# '(2R)-2-acetamido-3-[(2,3-dihydroxypropanoyl)sulfanyl]propanoic acid' (pin_verified at both
# tiers, tests/unit/assembly/test_leads_l6_acylsulfanyl.py) and 'CC(=O)N[C@@H](CSCCC(=O)C(=O)O)C(=O)O'
# is '4-{[(2R)-2-acetamido-2-carboxyethyl]sulfanyl}-2-oxobutanoic acid' (best-effort,
# systematic_verified). Mutation: with the three L6 changes reverted both rows give the glued
# 'N-acetyl' name again and this test fails.
@pytest.mark.parametrize("smiles", [
    "CC(=O)N[C@@H](CSCCC(=O)C(=O)O)C(=O)O",
    "CC(=O)N[C@@H](CSC(=O)C(O)CO)C(=O)O",
])
def test_mercapturate_best_effort_is_a_substitutive_name(smiles):
    from tests.support.rt_assert import name_best_effort
    name = name_best_effort(smiles)["name"]
    assert name_is_rt_exact(name, smiles)
    assert not name.startswith("N-acetyl"), name   # the glue 'N-acetyl<systematic name>' form
