"""v29 Phase 7 Task 2 — a parent-scope stereodescriptor must cite the PARENT's numbering.

Defect class C1 ( §3): two default-PIN-path
emissions carried a stereodescriptor whose locant does not exist in the name it decorates,
which OPSIN rejects with `Could not find atom/bond that: <stereoChemistry …> appeared to be
referring to`. That is a statement about OUR name, not about OPSIN's coverage.

Governing rule — **P-91.3 "NAMING OF STEREOISOMERS"** (`BlueBookV2/BlueBookV2.md:44639`):

    "In preferred IUPAC names, stereodescriptors are placed immediately at the front of the
     part of the name to which they relate. They are placed at the front of the complete
     name when related to the parent structure; they are cited in parentheses followed by a
     hyphen. When they relate to substituent groups, they are cited at the front of the
     corresponding prefix. They are preceded by a numerical or letter locant to describe the
     position of the stereogenic unit when such locants are present; general rules of
     numbering are applied (see P-14.4)."

with the section's own boundary PINs `[(1R)-1-chloropropyl]benzene` (descriptor inside the
enclosing marks, substituent numbering, NOT duplicated at the front) and
`(5Z)-4-[(1E)-prop-1-en-1-yl]hepta-1,5-diene` (parent block in the PARENT's numbering,
substituent block in the substituent's — two independent scopes).

Read with **P-14.3.3 "Citation of locants"** (`:2869`), whose own worked example makes the
scoping explicit: "locants are not used for the structural units defined by the parentheses
even though locants are used for these substituents of the parent structure ethanone."

ROOT CAUSE (one, shared by both defects — spy evidence in
): the descriptor block was built
from a locant map chosen by a priority chain in which `features.oriented_ring` OVERRODE the
principal-chain map. When the selected parent is the chain and the ring is only a
SUBSTITUENT, that map is the substituent's numbering, so every locant it produces is
foreign to the parent scope.
"""

import re

import pytest

from orthonym import Orthonym


# --------------------------------------------------------------------------
# The two defect molecules (FINDINGS.md §3 "The two C1 generator defects")
# --------------------------------------------------------------------------

# 2a: parent is `methanol` — ONE carbon. P-14.3.4.2(a) ("The locant '1' is omitted:
# (a) in substituted mononuclear parent hydrides", BB:2895) means the parent scope has no
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
    """P-91.3: the cyclopropyl centres belong to the SUBSTITUENT scope only."""
    assert namer.name(SMILES_2A) == EXPECTED_2A


def test_2a_emits_exactly_one_descriptor_block(namer):
    """The `(1R,2R)` block must appear once — inside the brackets, not also at the front."""
    name = namer.name(SMILES_2A)
    assert name.count("(1R,2R)") == 1, name
    assert not name.startswith("("), f"parent-level block leaked: {name}"


def test_2b_descriptor_locant_is_renumbered_into_the_parent(namer):
    """The locant must be 2 (C2=C3 of prop-2-enoic acid), not the pre-renumbering 5."""
    name = namer.name(SMILES_2B)
    block = re.match(r"\(([^)]*)\)-", name)
    assert block is not None, f"expected a leading stereodescriptor block, got: {name}"
    assert block.group(1) == "2E", f"expected locant 2, got {block.group(1)!r} in {name}"


def test_2b_full_name(namer):
    assert namer.name(SMILES_2B) == EXPECTED_2B


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
    """Guard against the vacuous-loop failure mode (a Phase 6 review finding)."""
    assert len(UNCHANGED) >= 8


@pytest.mark.parametrize("smiles,expected", UNCHANGED)
def test_legitimate_parent_descriptor_unchanged(namer, smiles, expected):
    assert namer.name(smiles) == expected


def test_true_exocyclic_ez_still_borrows_the_ring_locant(namer):
    """The exocyclic licence must survive the tightening.

    `_is_true_exocyclic` now demands BOTH halves of "exocyclic" — the in-scope atom in a
    ring AND the other end out of any ring. This is the genuine case: the C=C hangs off
    the ring, so borrowing the ring atom's locant is correct and must be kept.
    """
    assert namer.name("C/C=C1\\CC(C)CC1") == "(1Z)-1-ethylidene-3-methylcyclopentane"


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
    # HEAD emitted NO descriptor at all; the chain map supplies the correct one
    ("CN1C=C(C2=CC=CC=C21)[C@@H](CC(=O)N3CCCC3)C4=CC(=CC=C4)C(F)(F)F",
     "N-[(3S)-3-(1-methyl-1H-indol-3-yl)-3-[3-(trifluoromethyl)phenyl]propanoyl]pyrrolidine"),
]


@pytest.mark.parametrize("smiles,expected", REPAIRED)
def test_same_rule_repairs_these(namer, smiles, expected):
    assert namer.name(smiles) == expected


def test_fails_closed_rather_than_citing_an_unresolvable_locant(namer):
    """The honest fallback when no parent-scope locant resolves.

    Every stereogenic bond here lies inside a cyclodecene SUBSTITUENT. HEAD cited `(1E)`
    at parent scope, which OPSIN parsed as a DIFFERENT stereoisomer. There is no parent
    locant these bonds can legitimately take (P-91.3 puts them on the prefix), so the
    parent block is dropped: the constitution stays right and the stereo is simply not
    asserted. Per the contributor guide #9 this is pinned so the fallback cannot silently drift into
    a fabricated descriptor.
    """
    smiles = "C1CCC/C=C(\\CCCC1)/CC(C(=O)[O-])(/C/2=C/CCCCCCCC2)/C/3=C/CCCCCCCC3"
    name = namer.name(smiles)
    assert name == "2,2,3-tri(cyclodec-1-en-1-yl)propanoate"
    assert not name.startswith("("), f"a parent-scope block was fabricated: {name}"


# --------------------------------------------------------------------------
# The site itself: map selection is what the fix changes.
# --------------------------------------------------------------------------

def test_descriptor_block_is_decided_by_the_map_it_is_given():
    """The map IS the defect: the same molecule yields a block or nothing, per scope.

    Both maps below were measured by a spy on the live call (report §"Spy evidence"):
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
EXPECTED_CE2 = "N-methyl(2E)-3-[(1R,2R)-2-methylcyclopropyl]prop-2-enamide"

COUNTEREXAMPLES = [(SMILES_CE1, EXPECTED_CE1), (SMILES_CE2, EXPECTED_CE2)]


def test_counterexample_set_is_non_empty():
    """No vacuous parametrised loop (a Phase 6 review finding)."""
    assert len(COUNTEREXAMPLES) == 2


@pytest.mark.parametrize("smiles,expected", COUNTEREXAMPLES)
def test_amide_caller_inherits_the_parent_scope_rule(namer, smiles, expected):
    """`composer.py:5564` never passed an override — the shared chain must decide."""
    assert namer.name(smiles) == expected


@pytest.mark.parametrize("smiles,expected", COUNTEREXAMPLES)
def test_counterexample_block_cites_a_locant_the_parent_actually_has(namer, smiles, expected):
    """`prop-2-enamide` has C1..C3, so a leading `1`/`2` must denote the parent's own bond."""
    name = namer.name(smiles)
    block = re.search(r"\((\d+)([EZRS])\)-", name)
    assert block is not None, f"expected a parent-scope block, got: {name}"
    assert block.group(1) == "2", f"locant {block.group(1)!r} is not the parent's: {name}"
    # the substituent block keeps its OWN numbering, inside the brackets (P-91.3)
    assert "[(1R,2R)-2-methylcyclopropyl]" in name, name


def _ce1_features(principal_chain):
    """CE-1's real molecule and the real maps the spy recorded at `composer.py:5564`.

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
    """No vacuous parametrised loop (a Phase 6 review finding)."""
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
