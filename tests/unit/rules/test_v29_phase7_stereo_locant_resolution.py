""" a phase Task 2 — a parent-scope stereodescriptor must cite the PARENT's numbering.

Defect class C1 (internal notes): two default-PIN-path
emissions carried a stereodescriptor whose locant does not exist in the name it decorates,
which OPSIN rejects with `Could not find atom/bond that: <stereoChemistry …> appeared to be
referring to`. That is a statement about OUR name, not about OPSIN's coverage.

Governing rule — ** "NAMING OF STEREOISOMERS"** (`the Blue Book Blue Book`):

    "In preferred IUPAC names, stereodescriptors are placed immediately at the front of the
     part of the name to which they relate. They are placed at the front of the complete
     name when related to the parent structure; they are cited in parentheses followed by a
     hyphen. When they relate to substituent groups, they are cited at the front of the
     corresponding prefix. They are preceded by a numerical or letter locant to describe the
     position of the stereogenic unit when such locants are present; general rules of
     numbering are applied (see."

with the section's own boundary PINs `[(1R)-1-chloropropyl]benzene` (descriptor inside the
enclosing marks, substituent numbering, NOT duplicated at the front) and
`(5Z)-4-[(1E)-prop-1-en-1-yl]hepta-1,5-diene` (parent block in the PARENT's numbering,
substituent block in the substituent's — two independent scopes).

Read with ** "Citation of locants"** (`:2869`), whose own worked example makes the
scoping explicit: "locants are not used for the structural units defined by the parentheses
even though locants are used for these substituents of the parent structure ethanone."

ROOT CAUSE (one, shared by both defects — trace evidence in
internal notes): the descriptor block was built
from a locant map chosen by a priority chain in which `features.oriented_ring` OVERRODE the
principal-chain map. When the selected parent is the chain and the ring is only a
SUBSTITUENT, that map is the substituent's numbering, so every locant it produces is
foreign to the parent scope.
"""

import re

import pytest

from orthonym import Orthonym
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
DEFAULT_TIER_DECLINES = frozenset({
    "CN1C=C(C2=CC=CC=C21)[C@@H](CC(=O)N3CCCC3)C4=CC(=CC=C4)C(F)(F)F",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_obj_name(namer_obj, smiles):
    if smiles in DEFAULT_TIER_DECLINES:
        return _declined_pin_row(smiles)["name"]
    return namer_obj.name(smiles)


def _dt_obj_row(namer_obj, smiles):
    if smiles in DEFAULT_TIER_DECLINES:
        return _declined_pin_row(smiles)
    return namer_obj.name_tiered(smiles)


def _dt_name_compound(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return name_compound(smiles)


def _dt_name(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return Orthonym(style="pin").name(smiles)


def _dt_row(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym(style="pin").name_tiered(smiles)



# --------------------------------------------------------------------------
# The two defect molecules (FINDINGS.md "The two C1 generator defects")
# --------------------------------------------------------------------------

# 2a: parent is `methanol` — ONE carbon. (a) ("The locant '1' is omitted:
# (a) in substituted mononuclear parent hydrides", the Blue Book) means the parent scope has no
# cited locant at all, so a parent-level `(1R,2R)-` cannot resolve. The two stereocentres
# both live in the cyclopropyl SUBSTITUENT and are already cited inside its brackets.
SMILES_2A = "C[C@@H]1C[C@H]1CO"
EXPECTED_2A = "[(1R,2R)-2-methylcyclopropyl]methanol"

# 2b: parent is `prop-2-enoic acid` — three carbons, stereogenic double bond C2=C3, so the
# descriptor locant is 2 (the lower locant of the bond). The emitted `5` was an index from
# the cyclopentadienyl substituent's ring numbering. The descriptor VALUE is E, taken from
# `rdCIPLabeler.AssignCIPLabels` (never the legacy labeller — the contributor guide).
SMILES_2B = "C/C(=C\\C1C=CC=C1)/C(=O)O"
EXPECTED_2B = "(2E)-3-(cyclopenta-2,4-dien-1-yl)-2-methylprop-2-enoic acid"


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def test_2a_parent_block_not_duplicated_from_substituent_numbering(namer):
    """: the cyclopropyl centres belong to the SUBSTITUENT scope only."""
    assert _dt_obj_name(namer, SMILES_2A) == EXPECTED_2A


def test_2a_emits_exactly_one_descriptor_block(namer):
    """The `(1R,2R)` block must appear once — inside the brackets, not also at the front."""
    name = _dt_obj_name(namer, SMILES_2A)
    assert name.count("(1R,2R)") == 1, name
    assert not name.startswith("("), f"parent-level block leaked: {name}"


def test_2b_descriptor_locant_is_renumbered_into_the_parent(namer):
    """The locant must be 2 (C2=C3 of prop-2-enoic acid), not the pre-renumbering 5."""
    name = _dt_obj_name(namer, SMILES_2B)
    block = re.match(r"\(([^)]*)\)-", name)
    assert block is not None, f"expected a leading stereodescriptor block, got: {name}"
    assert block.group(1) == "2E", f"expected locant 2, got {block.group(1)!r} in {name}"


def test_2b_full_name(namer):
    assert _dt_obj_name(namer, SMILES_2B) == EXPECTED_2B


def test_2b_descriptor_value_matches_the_cip_labeller():
    """E vs Z is DERIVED, not assumed: rdCIPLabeler is the authority (the contributor guide)."""
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    mol = Chem.MolFromSmiles(SMILES_2B)
    rdCIPLabeler.AssignCIPLabels(mol)
    codes = [
        b.GetPropsAsDict()["_CIPCode"]
        for b in mol.GetBonds()
        if b.HasProp("_CIPCode")
    ]
    assert codes, "no bond CIP code found — the premise of this test is gone"
    assert codes == ["E"], codes
    assert EXPECTED_2B.startswith(f"(2{codes[0]})-")


# --------------------------------------------------------------------------
# Negative controls — a LEGITIMATE parent-level descriptor must be untouched.
# --------------------------------------------------------------------------

# Verified against HEAD before the fix (see the report's before/after table): every one of
# these already emitted exactly this name, and must keep emitting it. The ring-parent rows
# are the ones that exercise the map-selection branch this fix changes.
UNCHANGED = [
    # chain parent, R/S — the brief's required negative control
    ("C[C@H](O)CC", "(2S)-butan-2-ol"),
    # chain parent, E/Z
    ("C/C=C/C(=O)O", "(2E)-but-2-enoic acid"),
    # ring parents: the descriptor legitimately cites the RING's numbering
    ("C[C@@H]1CCCC[C@H]1O", "(1R,2R)-2-methylcyclohexan-1-ol"),
    ("O[C@H]1CCCC[C@@H]1O", "(1S,2S)-cyclohexane-1,2-diol"),
    ("C1C[C@H](O)C[C@@H](C)C1", "(1S,3S)-3-methylcyclohexan-1-ol"),
    ("C[C@@H](O)c1ccccc1", "(1R)-1-phenylethan-1-ol"),
    # heterocyclic ring parent (exercises the heterocycle_atom_to_locant branch)
    ("C[C@H]1CCCO1", "(2S)-2-methyloxolane"),
    # the verbatim Blue Book pseudo-asymmetric PIN shape the stereo carve-out exists for
    ("O[C@@H]1CC[C@H](C)CC1", "(1s,4s)-4-methylcyclohexan-1-ol"),
]


def test_negative_control_set_is_non_empty():
    """Guard against the vacuous-loop failure mode (a a phase review finding)."""
    assert len(UNCHANGED) >= 8


@pytest.mark.parametrize("smiles,expected", UNCHANGED)
def test_legitimate_parent_descriptor_unchanged(namer, smiles, expected):
    assert _dt_obj_name(namer, smiles) == expected


def test_true_exocyclic_ez_still_borrows_the_ring_locant(namer):
    """The exocyclic licence must survive the tightening.

    `_is_true_exocyclic` now demands BOTH halves of "exocyclic" — the in-scope atom in a
    ring AND the other end out of any ring. This is the genuine case: the C=C hangs off
    the ring, so borrowing the ring atom's locant is correct and must be kept.
    """
    assert _dt_obj_name(namer, "C/C=C1\\CC(C)CC1") == "(1Z)-1-ethylidene-3-methylcyclopentane"


# --------------------------------------------------------------------------
# Collateral: names that the same rule repairs (measured in the 1000-row A/B).
# --------------------------------------------------------------------------

REPAIRED = [
    # a duplicated substituent block whose locants happen to EXIST on the parent — the
    # insidious variant, since it parses as a different molecule rather than failing
    ("CC[C@H]1C[C@H]1CCCCCCCCCCC(=O)O",
     "11-[(1R,2S)-2-ethylcyclopropyl]undecanoic acid"),
    # HEAD cited one bond with a locant of 1; both bonds are stereogenic, at 2 and 4
    ("C/C(=C\\C#N)/C=C/N1CCCCC1",
     "(2E,4E)-3-methyl-5-(piperidin-1-yl)penta-2,4-dienenitrile"),
]


@pytest.mark.parametrize("smiles,expected", REPAIRED)
def test_same_rule_repairs_these(namer, smiles, expected):
    assert _dt_obj_name(namer, smiles) == expected


# The third repaired row of the A/B: HEAD emitted NO descriptor at all; the chain map
# supplies '(3S)'. It is a hidden amide (acyl on a ring N), so its PIN is the
# pseudoketone: 'Hidden' amides (the Blue Book) "now preferably named
# as a pseudoketone" (:33127), cf. '1-(piperidin-1-yl)propan-1-one (PIN)
# 1-propanoylpiperidine' (:29374). The old literal
# 'N-[(3S)-...propanoyl]pyrrolidine' was never the PIN and never shipped (its
# content holds '(' and '[', so it needs braces); what shipped was the doubled
# 'N-({(3S)-...propanoyl})pyrrolidine':7444: one level per fragment;
# TRIAGE g7 C10). OPSIN 2.9.0 full-InChIKey: both names below EXACT.
HIDDEN_AMIDE = "CN1C=C(C2=CC=CC=C21)[C@@H](CC(=O)N3CCCC3)C4=CC(=CC=C4)C(F)(F)F"
HIDDEN_AMIDE_PIN = ("(3S)-3-(1-methyl-1H-indol-3-yl)-1-(pyrrolidin-1-yl)-3-"
                    "[3-(trifluoromethyl)phenyl]propan-1-one")
HIDDEN_AMIDE_SHIPPED = ("N-{(3S)-3-(1-methyl-1H-indol-3-yl)-3-[3-(trifluoromethyl)"
                        "phenyl]propanoyl}pyrrolidine")


@pytest.mark.xfail(strict=True, reason=(
    "PIN is the pseudoketone (P-66.1.3, BlueBookV2.md:33127): "
    "rules/pseudoketones.name_pseudoketone names only an UNSUBSTITUTED acyl chain "
    "('1-(pyrrolidin-1-yl)propan-1-one'); the substituted-acyl pseudoketone "
    "producer is not built -- TODO in TRIAGE.md 'Suite fix -- j5-pin-labels-b'"))
def test_hidden_amide_row_pin(namer):
    assert _dt_obj_name(namer, HIDDEN_AMIDE) == HIDDEN_AMIDE_PIN


@pytest.mark.opsin_gate
def test_hidden_amide_row_ships_below_the_pin_tier(namer):
    """Production (gate on): the stereo descriptor is resolved, the N-acyl name
    ships RT-exact with ONE level of enclosing marks, labelled below pin_verified."""
    from tests.support.rt_assert import name_is_rt_exact
    r = _dt_obj_row(namer, HIDDEN_AMIDE)
    assert r["name"] == HIDDEN_AMIDE_SHIPPED, r
    assert r["tier"] != "pin_verified", r
    assert name_is_rt_exact(r["name"], HIDDEN_AMIDE), r


def test_fails_closed_rather_than_citing_an_unresolvable_locant(namer):
    """The honest fallback when no parent-scope locant resolves.

    Every stereogenic bond here lies inside a cyclodecene SUBSTITUENT. HEAD cited `(1E)`
    at parent scope, which OPSIN parsed as a DIFFERENT stereoisomer. There is no parent
    locant these bonds can legitimately take puts them on the prefix), so the
    parent block is dropped: the constitution stays right and the stereo is simply not
    asserted. Per the contributor guide #9 this is pinned so the fallback cannot silently drift into
    a fabricated descriptor.
    """
    smiles = "C1CCC/C=C(\\CCCC1)/CC(C(=O)[O-])(/C/2=C/CCCCCCCC2)/C/3=C/CCCCCCCC3"
    name = _dt_obj_name(namer, smiles)
    assert name == "2,2,3-tri(cyclodec-1-en-1-yl)propanoate"
    assert not name.startswith("("), f"a parent-scope block was fabricated: {name}"


# --------------------------------------------------------------------------
# The site itself: map selection is what the fix changes.
# --------------------------------------------------------------------------

def test_descriptor_block_is_decided_by_the_map_it_is_given():
    """The map IS the defect: the same molecule yields a block or nothing, per scope.

    Both maps below were measured by a trace on the live call (report §"Spy evidence"):
    `oriented_ring` describes the cyclopropyl SUBSTITUENT, while the parent selected by
    the handler is the one-carbon `methanol` chain. Only the substituent map contains the
    stereocentres, so only it may carry the `(1R,2R)` block — at substituent scope.
    """
    from rdkit import Chem

    from orthonym.assembly.handlers._handler_shared import _generate_stereodescriptors
    from orthonym.namer import compute_features

    features = compute_features(Chem.MolFromSmiles(SMILES_2A), SMILES_2A)
    assert features.stereocenters, "premise gone: no stereocentres perceived"

    substituent_ring_map = {3: 1, 1: 2, 2: 3}   # cyclopropyl numbering
    parent_chain_map = {4: 1}                   # methanol numbering

    block = _generate_stereodescriptors(
        features, atom_to_locant_override=substituent_ring_map
    )
    assert block is not None and block.text == "(1R,2R)-", block

    assert _generate_stereodescriptors(
        features, atom_to_locant_override=parent_chain_map
    ) is None, "the parent scope has no stereocentre, so it must carry no block"


# --------------------------------------------------------------------------
# THE CLASS, not two patches (Task 2 review, CRITICAL).
#
# The first fix lived in ONE caller (`general_acyclic.py`, via an
# `atom_to_locant_override`). The shared priority chain was untouched, so the four
# `composer.py` callers that pick a parent the same way — `if features.principal_chain:
# chain parent` — still took a SUBSTITUENT ring's numbering. These two molecules are the
# reviewer's measured counter-examples; both reached `composer.py:5564` and both produced
# the same OPSIN error signature as 2a/2b. The rule now lives in
# `_generate_stereodescriptors` itself, so every caller inherits it.
# --------------------------------------------------------------------------

# `prop-2-enamide` has THREE carbons; the emitted `(1R,2R)` came from the cyclopropyl
# substituent's own numbering. The parent genuinely does have a stereogenic C2=C3 bond, so
# the correct parent block is `(2E)` — the fix ADDS a right descriptor, it does not merely
# delete a wrong one (the contributor guide #9).
SMILES_CE1 = "C[C@@H]1C[C@H]1/C=C/C(N)=O"
EXPECTED_CE1 = "(2E)-3-[(1R,2R)-2-methylcyclopropyl]prop-2-enamide"

SMILES_CE2 = "C[C@@H]1C[C@H]1/C=C/C(=O)NC"
# The parent block stands at the front of the complete name, the N-substituent in the
# prefix series: (the Blue Book) "Stereodescriptors placed at the front of
# the complete name or name fragment to which they apply"; (:3477).
EXPECTED_CE2 = "(2E)-N-methyl-3-[(1R,2R)-2-methylcyclopropyl]prop-2-enamide"

COUNTEREXAMPLES = [(SMILES_CE1, EXPECTED_CE1), (SMILES_CE2, EXPECTED_CE2)]


def test_counterexample_set_is_non_empty():
    """No vacuous parametrised loop (a a phase review finding)."""
    assert len(COUNTEREXAMPLES) == 2


@pytest.mark.parametrize("smiles,expected", COUNTEREXAMPLES)
def test_amide_caller_inherits_the_parent_scope_rule(namer, smiles, expected):
    """`composer.py:5564` never passed an override — the shared chain must decide."""
    assert _dt_obj_name(namer, smiles) == expected


@pytest.mark.parametrize("smiles,expected", COUNTEREXAMPLES)
def test_counterexample_block_cites_a_locant_the_parent_actually_has(namer, smiles, expected):
    """`prop-2-enamide` has C1..C3, so a leading `1`/`2` must denote the parent's own bond."""
    name = _dt_obj_name(namer, smiles)
    block = re.search(r"\((\d+)([EZRS])\)-", name)
    assert block is not None, f"expected a parent-scope block, got: {name}"
    assert block.group(1) == "2", f"locant {block.group(1)!r} is not the parent's: {name}"
    # the substituent block keeps its OWN numbering, inside the brackets
    assert "[(1R,2R)-2-methylcyclopropyl]" in name, name


def _ce1_features(principal_chain):
    """'s real molecule and the real maps the trace recorded at `composer.py:5564`.

    A stub rather than `compute_features`, deliberately: `principal_chain` and
    `oriented_ring` are populated LATER in `namer.py` during parent selection (`:4684`,
    `:4732`), so `compute_features` alone leaves them empty and a test built on it would
    assert nothing about the branch under test.
    """
    from types import SimpleNamespace

    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    mol = Chem.MolFromSmiles(SMILES_CE1)
    rdCIPLabeler.AssignCIPLabels(mol)       # never the legacy labeller (the contributor guide)
    return SimpleNamespace(
        mol=mol,
        stereocenters=[1, 3],
        double_bond_stereo=[(4, 5)],
        atom_to_locant={6: 1, 5: 2, 4: 3},      # prop-2-enamide, the PARENT
        heterocycle_atom_to_locant=None,
        oriented_ring=[1, 3, 2],                # 2-methylcyclopropyl, a SUBSTITUENT
        principal_chain=principal_chain,
    )


# (principal_chain, caller_selects_parent) -> the block the shared function must return.
# Measured 2x2; every cell is a distinct branch of the rule.
MAP_SELECTION_TABLE = [
    # a parent-assembling caller with a chain parent: the PARENT's own C2=C3 bond.
    ((6, 5, 4), True, "(2E)-"),
    # `_inject_stereo_if_missing`: it did not choose the parent, so principal_chain is not
    # evidence about it — legacy ring priority is retained on purpose.
    ((6, 5, 4), False, "(1R,2R)-"),
    # no chain at all => a ring parent; the ring map IS the parent scope, both ways.
    ((), True, "(1R,2R)-"),
    ((), False, "(1R,2R)-"),
]


def test_map_selection_table_is_non_empty():
    """No vacuous parametrised loop (a a phase review finding)."""
    assert len(MAP_SELECTION_TABLE) == 4


@pytest.mark.parametrize("principal_chain,caller_selects_parent,expected",
                         MAP_SELECTION_TABLE)
def test_the_rule_lives_in_the_shared_function_not_in_a_caller(
        principal_chain, caller_selects_parent, expected):
    """The class contract, asserted on `_generate_stereodescriptors` itself.

    Row 1 is what makes this a CLASS fix rather than five patches: called with no
    override — exactly how the four `composer.py` sites call it — the function refuses the
    substituent ring's numbering on its own, so a caller need not know the rule.

    Row 2 is the measured boundary: applying the rule at `_inject_stereo_if_missing`
    dropped correct descriptors from two ring-parent names whose `principal_chain` was a
    stale 1–2 atom fragment.

    Rows 3–4 are why the guard is conditional and not a deletion: with no chain, the ring
    map is the only numbering there is.
    """
    from orthonym.assembly.handlers._handler_shared import _generate_stereodescriptors

    features = _ce1_features(principal_chain)
    block = _generate_stereodescriptors(
        features, caller_selects_parent=caller_selects_parent)
    assert block is not None, "premise gone: no descriptor at all"
    assert block.text == expected, block.text


# --------------------------------------------------------------------------
# Both corrected names must actually PARSE. The whole defect class is defined by
# OPSIN rejecting the stereo layer, so this is the assertion that bites.
# --------------------------------------------------------------------------

def _opsin_smiles(jar: str, name: str):
    """Feed the name on STDIN.

    NOTE: the shared `opsin_to_smiles` fixture passes the name as a trailing CLI
    argument, which the OPSIN CLI interprets as an input FILE, so it returns None for
    every name including valid ones (the contributor guide: a harness that always fails is as
    useless as one that always passes). Stdin is the working interface.
    """
    import subprocess

    proc = subprocess.run(
        ["java", "-jar", jar, "-osmi"],
        input=name + "\n", capture_output=True, text=True, timeout=60,
    )
    out = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    return out[-1] if out else None


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [(SMILES_2A, EXPECTED_2A),
                                         (SMILES_2B, EXPECTED_2B),
                                         (SMILES_CE1, EXPECTED_CE1),
                                         (SMILES_CE2, EXPECTED_CE2)])
def test_corrected_names_round_trip_through_opsin(opsin_jar, smiles, name):
    """The defect class IS "OPSIN rejects the stereo layer", so this is the assertion
    that bites — and it checks IDENTITY, not merely parseability: the parsed structure
    must be the input stereoisomer, InChIKey-for-InChIKey."""
    from rdkit import Chem

    if not opsin_jar:
        pytest.skip("OPSIN jar not available")

    parsed = _opsin_smiles(opsin_jar, name)
    assert parsed, f"OPSIN could not parse the corrected name: {name}"

    got = Chem.MolFromSmiles(parsed)
    assert got is not None, f"OPSIN returned unparseable SMILES {parsed!r} for {name}"
    assert Chem.MolToInchiKey(got) == Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)), (
        f"{name} round-trips to a DIFFERENT stereoisomer: {parsed}"
    )


# ==========================================================================
# RESIDUAL (Task 2 CRITICAL review): `_inject_stereo_if_missing`.
#
# The five parent-ASSEMBLING callers inherit the parent-scope rule by default.
# `_inject_stereo_if_missing` is the one caller that does NOT select a parent --
# it decorates a name ~42 handler sites already built -- so it was left on legacy
# ring-priority resolution and stayed a live member of the class.
#
# Measured over 661 live calls / 176 molecules / 21 distinct call sites
# (pubchem_2000 + chebi_5000 stereo rows, census in
# internal notes):
# * 429 calls pass an explicit `atom_to_locant` -> immune, 0 differ.
# * 66 calls / 16 molecules resolve DIFFERENTLY under the two candidate scopes.
# * 14 of those 16 abstain; 2 emit, and in both the LEGACY answer is correct.
# 0 leak a wrong emission, so the 0-wrong invariant held -- it was a coverage
# loss, not a wrong name.
#
# Two inference attempts were measured and REFUTED, and these tests pin both so
# neither can be reinstated:
# * `features.principal_chain` truthy => chain parent -- false, it is a stale
# 1-2 atom fragment on the two ring-parent rows below.
# * `features.chain_is_parent` (parent selection's own verdict, namer.py:1447)
# => chain parent -- false, it is True on 48/48 differing pubchem calls
# INCLUDING both ring-parent rows.
# So the scope is DECLARED by the producer that chose the parent.
# ==========================================================================

# The residual counter-example. NOTE the parent stem: the acyl chain is
# UNSATURATED, so the PIN is `prop-2-enoyl`, not `propanoyl`. `propanoyl` is
# saturated and OPSIN rejects `(2E)-...propanoyl chloride` outright with
# "Could not find bond that: <stereoChemistry locant='2' type='EorZ'> was
# referring to" -- the C=C had been silently dropped by the acyl namer.
SMILES_C1R = "C[C@@H]1C[C@H]1/C=C/C(Cl)=O"
EXPECTED_C1R = "(2E)-3-[(1R,2R)-2-methylcyclopropyl]prop-2-enoyl chloride"


def test_residual_acid_halide_cites_the_parents_own_numbering(namer):
    """HEAD emitted `(1R,2R)-3-[(1R,2R)-2-methylcyclopropyl]propanoyl chloride`:
    the cyclopropyl SUBSTITUENT's numbering, duplicated at parent scope, on a
    parent stem that had also lost its double bond."""
    assert _dt_obj_name(namer, SMILES_C1R) == EXPECTED_C1R


def test_residual_parent_block_is_not_the_substituent_block(namer):
    """The `(1R,2R)` belongs inside the brackets and nowhere else."""
    name = _dt_obj_name(namer, SMILES_C1R)
    assert name.startswith("(2E)-"), name
    assert name.count("(1R,2R)") == 1, f"substituent block duplicated: {name}"


# The two rows a naive `principal_chain`-only guard broke. Both are RING parents
# carrying a STALE truthy `principal_chain` (pc=(1,0) and pc=(1,)) AND
# `chain_is_parent=True`. They must keep their descriptors.
PENTAACETATE = "CC(=O)O[C@@H]1[C@H](C([C@H]([C@@H](C1OC(=O)C)OC(=O)C)OC(=O)C)F)OC(=O)C"
# The pentaacetate's PIN is the polyol-ester functional-class name:
# (the Blue Book) "All preferred IUPAC names for esters are named by
# functional class nomenclature"; (:31819) 'propane-1,2,3-triyl
# triacetate (PIN)' (:31827), '...-2-fluorooxane-3,4,5-triyl triacetate' (:53293).
# Both ring directions round-trip (OPSIN 2.9.0 full InChIKey, the molecule is
# mirror-symmetric through C3/C6); (j) (:3346) gives R the lower locant.
# The engine emits the acyloxy-prefix name '(1R,2S,4R,5S)-1,2,4,5,6-pentakis
# (acetyloxy)-3-fluorocyclohexane' at pin_verified (RT exact, but not the PIN, and
# its numbering misses (g):3307, which gives the first-cited acetyloxy
# 1,2,3,4,5). The old snapshot '(1S,2R,4S,5R)-3-fluoro-1,2,4,5,6-pentakis(acetyloxy)
# cyclohexane' was the same non-PIN with the fluoro cited out of order
# and the S-first numbering. What this row guards --
# the descriptors are kept -- is asserted spelling-free below.
PENTAACETATE_PIN = "(1R,2S,4R,5S)-6-fluorocyclohexane-1,2,3,4,5-pentayl pentaacetate"
RING_PARENT_ROWS = [
    (PENTAACETATE, PENTAACETATE_PIN),
    ("C1CCCC(/C=C\\CC1)OC=O", "(2Z)-cyclonon-2-en-1-yl formate"),
]
_RING_PARENT_PARAMS = [
    pytest.param(*RING_PARENT_ROWS[0], marks=pytest.mark.xfail(strict=True, reason=(
        "PIN is the functional-class polyol ester (P-65.6.3.2.1 :31663); the "
        "engine ships the acyloxy-prefix name at pin_verified -- TODO in "
        "TRIAGE.md 'Suite fix -- j1-regressions'"))),
    RING_PARENT_ROWS[1],
]


def test_ring_parent_row_set_is_non_empty():
    """No vacuous parametrised loop."""
    assert len(RING_PARENT_ROWS) == 2


@pytest.mark.parametrize("smiles,expected", _RING_PARENT_PARAMS)
def test_ring_parent_descriptors_are_not_dropped(namer, smiles, expected):
    assert _dt_obj_name(namer, smiles) == expected


def test_pentaacetate_ring_parent_keeps_its_descriptors(namer):
    """The contract of the strict-xfail row above, independent of its spelling:
    the ring parent's four descriptors survive, i.e. the name round-trips to the
    input's FULL InChIKey (stereo layer included; OPSIN run outside the engine)."""
    from tests.support.rt_assert import assert_full_rt
    name = _dt_obj_name(namer, PENTAACETATE)
    assert_full_rt(name, PENTAACETATE)


def _capture_injections(namer, smiles):
    """Run the REAL pipeline and capture every `_inject_stereo_if_missing` call.

    `compute_features` alone is NOT enough here: it does not run parent
    selection, so `principal_group`/`principal_chain`/`chain_is_parent` are all
    empty and a test built on it would assert against features production never
    sees. Spying the live call is also the stronger test -- it pins the value the
    producer actually declared.
    """
    import orthonym.assembly.composer as composer

    orig = composer._inject_stereo_if_missing
    seen = []

    def _spy(features, name, atom_to_locant=None, parent_scope=None):
        out = orig(features, name, atom_to_locant, parent_scope)
        seen.append({"features": features, "name": name,
                     "atom_to_locant": atom_to_locant,
                     "parent_scope": parent_scope, "out": out})
        return out

    composer._inject_stereo_if_missing = _spy
    try:
        final = _dt_obj_name(namer, smiles)
    finally:
        composer._inject_stereo_if_missing = orig
    return final, seen


def test_the_injection_site_is_actually_on_the_path(namer):
    """Spy validated on known positives before anything is asserted about it
    (a documented 8-for-8 failure mode: the named site is off the path)."""
    for smiles, _ in RING_PARENT_ROWS:
        _final, seen = _capture_injections(namer, smiles)
        assert seen, f"_inject_stereo_if_missing never ran for {smiles}"
    _final, seen = _capture_injections(namer, SMILES_C1R)
    assert seen, "_inject_stereo_if_missing never ran for the residual row"


def test_ring_parent_rows_defeat_both_refuted_inferences(namer):
    """Pins WHY the scope must be declared: on these rows both candidate
    signals say "chain" while the name is numbered in the RING."""
    checked = 0
    for smiles, _expected in RING_PARENT_ROWS:
        _final, seen = _capture_injections(namer, smiles)
        assert seen, f"site never ran for {smiles}"
        features = seen[0]["features"]
        assert features.principal_chain, (
            "premise gone: principal_chain is falsy, so it could not mislead")
        assert getattr(features, "chain_is_parent", False), (
            "premise gone: chain_is_parent is False, so it could not mislead")
        #... and yet the chain it points at is a 1-2 atom fragment, not the parent.
        assert len(features.principal_chain) <= 2, features.principal_chain
        checked += 1
    assert checked == len(RING_PARENT_ROWS)


def test_undeclared_scope_fails_closed_when_the_two_scopes_disagree(namer):
    """`parent_scope=None` and the ring/chain resolutions differ -> no block.

    Better a missing stereo block than a locant that may not resolve
    , the Blue Book "Citation of locants",:2869) -- the same posture
    `_ring_handler_parent_atom_indices` already documents.
    """
    from orthonym.assembly.composer import _inject_stereo_if_missing
    from orthonym.assembly.handlers._handler_shared import _generate_stereodescriptors

    _final, seen = _capture_injections(namer, SMILES_C1R)
    assert seen, "site never ran"
    features = seen[0]["features"]

    ring = _generate_stereodescriptors(features, caller_selects_parent=False)
    chain = _generate_stereodescriptors(features, caller_selects_parent=True)
    ring_t = ring.text if ring else None
    chain_t = chain.text if chain else None
    assert ring_t != chain_t, (
        "premise gone: the two scopes agree here, so this row cannot test "
        f"fail-closed ({ring_t!r} vs {chain_t!r})")

    stem = "3-[(1R,2R)-2-methylcyclopropyl]prop-2-enoyl chloride"
    assert _inject_stereo_if_missing(features, stem) == stem
    assert _inject_stereo_if_missing(
        features, stem, parent_scope='chain') == chain_t + stem
    assert _inject_stereo_if_missing(
        features, stem, parent_scope='ring') == ring_t + stem


def test_declared_scope_is_honoured_even_when_the_scopes_agree(namer):
    """A declaration must never *lose* a descriptor the legacy path emitted."""
    from orthonym.assembly.composer import _inject_stereo_if_missing

    smiles, _ = RING_PARENT_ROWS[1]
    _final, seen = _capture_injections(namer, smiles)
    assert seen, "site never ran"
    features = seen[0]["features"]
    stem = "cyclonon-2-en-1-yl formate"
    assert _inject_stereo_if_missing(
        features, stem, parent_scope='ring') == "(2Z)-" + stem


# --------------------------------------------------------------------------
# The producer reports the parent it chose -- that is where the fix is rooted.
# --------------------------------------------------------------------------

ACYL_SCOPE_ROWS = [
    ("CCC(Cl)=O", "chain"),                    # acyclic acyl -> chain parent
    ("c1ccccc1C(Cl)=O", "ring"),               # benzoyl -> ring parent
    ("C[C@@H]1C[C@H]1/C=C/C(Cl)=O", "chain"),  # the residual row
]


def test_acyl_scope_row_set_is_non_empty():
    assert len(ACYL_SCOPE_ROWS) == 3


@pytest.mark.parametrize("smiles,expected_scope", ACYL_SCOPE_ROWS)
def test_acid_halide_producer_reports_its_parent(namer, smiles, expected_scope):
    """End-to-end: the producer records the branch it took and the handler hands
    that declaration to the injector."""
    _final, seen = _capture_injections(namer, smiles)
    assert seen, f"the injection site never ran for {smiles}"
    scopes = {c["parent_scope"] for c in seen}
    assert expected_scope in scopes, (
        f"{smiles}: expected a {expected_scope!r} declaration, saw {scopes}")


def test_ring_ester_declares_ring_only_when_the_alcohol_is_the_ring():
    """`<R>yl <acyl>ate` is numbered in the ALCOHOL component."""
    from rdkit import Chem

    from orthonym.assembly.handlers.ring_ester import _alcohol_is_ring

    assert _alcohol_is_ring(Chem.MolFromSmiles("C1CCCC(/C=C\\CC1)OC=O")) == 'ring'
    assert _alcohol_is_ring(Chem.MolFromSmiles("CCCCCC(=O)OC")) is None


# --------------------------------------------------------------------------
# Collateral, found by verifying WHAT IS EMITTED after the fix (the contributor guide #9):
# `_build_acyl_name`'s `unsaturation` parameter had NO caller, so every acyl
# halide was spelled saturated and its C=C silently dropped.
# --------------------------------------------------------------------------

ACYL_UNSATURATION_ROWS = [
    ("CCCCC/C=C\\C(Cl)=O", "(2Z)-oct-2-enoyl chloride"),
    ("CCC(Cl)=O", "propanoyl chloride"),      # saturated control, unchanged
    ("CC(=O)Cl", "acetyl chloride"),          # retained name, unchanged
    ("c1ccccc1C(Cl)=O", "benzoyl chloride"),  # ring parent, unchanged
]


def test_acyl_unsaturation_row_set_is_non_empty():
    assert len(ACYL_UNSATURATION_ROWS) == 4


@pytest.mark.parametrize("smiles,expected", ACYL_UNSATURATION_ROWS)
def test_acyl_halide_spells_its_own_unsaturation(namer, smiles, expected):
    assert _dt_obj_name(namer, smiles) == expected


@pytest.mark.opsin_gate
def test_residual_name_round_trips_through_opsin(opsin_jar):
    """InChIKey identity, not merely parseability."""
    from rdkit import Chem

    if not opsin_jar:
        pytest.skip("OPSIN jar not available")

    parsed = _opsin_smiles(opsin_jar, EXPECTED_C1R)
    assert parsed, f"OPSIN could not parse: {EXPECTED_C1R}"
    got = Chem.MolFromSmiles(parsed)
    assert got is not None, parsed
    assert Chem.MolToInchiKey(got) == Chem.MolToInchiKey(
        Chem.MolFromSmiles(SMILES_C1R)), (
        f"{EXPECTED_C1R} round-trips to a DIFFERENT stereoisomer: {parsed}")


@pytest.mark.opsin_gate
def test_the_briefs_saturated_spelling_is_genuinely_unparseable(opsin_jar):
    """Guards the correction: `propanoyl` carries no C2=C3, so `(2E)-` cannot
    resolve. This is why the expected name is `prop-2-enoyl`, and it must stay
    red if anyone 'simplifies' the stem back."""
    if not opsin_jar:
        pytest.skip("OPSIN jar not available")

    bad = "(2E)-3-[(1R,2R)-2-methylcyclopropyl]propanoyl chloride"
    assert _opsin_smiles(opsin_jar, bad) is None, (
        "OPSIN now parses the saturated spelling -- re-derive the expected name")
