"""(c) outranks (f): the principal characteristic group takes the low locant.

§** "NUMBERING"** (``the Blue Book Blue Book``) verbatim:

    When several structural features appear in cyclic and acyclic compounds, low
    locants are assigned to them in the following decreasing order of seniority:

and the ordered criteria, each on its own line:

    :3227 (a) fixed numbering in chains, rings, or ring systems
    :3246 (b) indicated hydrogen for unsubstituted compounds
    :3256 (c) principal characteristic groups and free valences (suffixes);
    :3270 (d) 'added indicated hydrogen'
    :3288 (e) saturation/unsaturation
    :3301 (f) detachable alphabetized prefixes, all considered together in a
           series of increasing numerical order;
    :3307 (g) lowest locants for the substituent cited first as a prefix
    :3320 (h) nonstandard valence state

Benzene has no fixed numbering (a) and no indicated hydrogen (b), so **(c)
decides**, four places above (f).

The defect these tests pin: ``orient_benzene``'s (c) tier was reached but
handed an EMPTY set, because the caller derived the principal-characteristic-group
ring atoms from the ``is_suffix`` marker alone. A phenolic -OH is still spelled as
the *prefix* ``hydroxy`` at orientation time and is promoted to the ``-ol`` suffix
only later, so criterion (c) never ran and the chloro prefix took locant 1.

★ The critical negative is ``Cc1c(C)c(C)c(C)c(C)c1Cl`` ->
``1-chloro-2,3,4,5,6-pentamethylbenzene``: the same six-substituent shape as the
defect but with **no principal characteristic group at all**, so (c) is vacuous and
(f)+(g) legitimately govern. A "fix" that also moved this row would have
implemented "lowest locant to the ring's first substituent", not (c).
"""

import pytest

from orthonym import name_compound
from orthonym.rules import benzene as benzene_rules
from orthonym.rules.benzene import (
    get_benzene_substituents,
    principal_group_ring_atoms,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Whole-name assertions. Session a project rule: "the locant moved" is NOT a pass
# condition -- a change that stops a bad path can emit something worse -- so
# every row below asserts the COMPLETE emitted name.
# --------------------------------------------------------------------------

# The suffix set {1,2,3,4,5} beats {2,3,4,5,6} at the first point of difference.
P14_4_C_TARGETS = [
    ("Oc1c(O)c(O)c(O)c(O)c1Cl", "6-chlorobenzene-1,2,3,4,5-pentol"),
    ("Oc1c(O)c(O)c(O)c(O)c1F", "6-fluorobenzene-1,2,3,4,5-pentol"),
    ("Oc1c(O)c(O)c(O)c(O)c1C", "6-methylbenzene-1,2,3,4,5-pentol"),
    # Suffix set {1,2,4} (two adjacent OH, third OH four positions on) beats the
    # emitted {2,4,5} at the FIRST term. Chloro then takes what is left, 5.
    ("Oc1cc(O)c(Cl)cc1O", "5-chlorobenzene-1,2,4-triol"),
    # Same class, found by widening the census beyond the reported rows.
    # Two adjacent OH -> {1,2}; the chloro is adjacent to one of them -> 3.
    ("Oc1c(Cl)cccc1O", "3-chlorobenzene-1,2-diol"),
    # The SAME molecule spelled differently -- both must land on one name.
    ("Oc1c(O)cccc1Cl", "3-chlorobenzene-1,2-diol"),
    # OH at {1,2,4} beats {2,4,5}; the two chloro take 3 and 5.
    ("Oc1cc(Cl)c(O)c(Cl)c1O", "3,5-dichlorobenzene-1,2,4-triol"),
    ("Oc1c(O)c(O)c(O)c(O)c1N=[N+]=[N-]", "6-azidobenzene-1,2,3,4,5-pentol"),
    # ★ The row that proves the defect was NOT "the alphabet picked the wrong
    # substituent for position 1". Here 'hydroxy' < 'iodo', so the pre-fix numberer
    # already gave locant 1 to a hydroxy -- and was STILL wrong, because nothing
    # minimised the suffix SET: it emitted '2-iodobenzene-1,3,4,5,6-pentol', suffix
    # {1,3,4,5,6}. (c) requires the whole set to be lowest, {1,2,3,4,5}.
    ("Oc1c(O)c(O)c(O)c(O)c1I", "6-iodobenzene-1,2,3,4,5-pentol"),
]


@pytest.mark.parametrize("smiles,expected", P14_4_C_TARGETS)
def test_pcg_takes_the_low_locant(smiles, expected):
    """(c): the suffix locant set is minimised BEFORE the prefixes."""
    assert name_compound(smiles) == expected


# --------------------------------------------------------------------------
# MANDATORY NEGATIVES -- each measured correct before the change; none may move.
# --------------------------------------------------------------------------

P14_4_NEGATIVES = [
    # ★★ THE critical guard. Six substituents, no principal characteristic group,
    # so criterion (c) is vacuous: (f) ties every orientation at {1,2,3,4,5,6} and
    # (g) gives locant 1 to the alphabetically first prefix, chloro.
    ("Cc1c(C)c(C)c(C)c(C)c1Cl", "1-chloro-2,3,4,5,6-pentamethylbenzene"),
    # No suffix at all -> (f) legitimately governs.
    ("Brc1ccc(Br)cc1", "1,4-dibromobenzene"),
    ("Cc1c(C)c(C)c(C)c(C)c1C", "hexamethylbenzene"),
    ("Fc1c(F)c(F)c(F)c(F)c1F", "hexafluorobenzene"),
    ("Clc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl", "hexachlorobenzene"),
    # Already correct -- the tier must not be double-applied.
    ("Nc1c(N)c(N)c(N)c(N)c1Cl", "6-chlorobenzene-1,2,3,4,5-pentamine"),
    ("Sc1c(S)c(S)c(S)c(S)c1Cl", "6-chlorobenzene-1,2,3,4,5-pentathiol"),
    ("Nc1c(N)c(N)c(N)c(N)c1C", "6-methylbenzene-1,2,3,4,5-pentamine"),
    ("Sc1c(S)c(S)c(S)c(S)c1C", "6-methylbenzene-1,2,3,4,5-pentathiol"),
    # (c) fixes O at 1, then (f) minimises Cl.
    ("Oc1ccccc1Cl", "2-chlorophenol"),
    ("Oc1ccccc1O", "benzene-1,2-diol"),
    # Suffix set {1,4} is already lowest; the prefix tier then picks Cl at 2.
    ("Oc1ccc(O)c(Cl)c1", "2-chlorobenzene-1,4-diol"),
    # Three contiguous OH -> {1,2,3} already lowest.
    ("Oc1c(O)c(O)cc(Cl)c1", "5-chlorobenzene-1,2,3-triol"),
    ("Oc1cc(Cl)cc(O)c1O", "5-chlorobenzene-1,2,3-triol"),
    ("Oc1c(C)c(O)c(C)c(O)c1C", "2,4,6-trimethylbenzene-1,3,5-triol"),
    #: -ol outranks -thiol, so the SH stays a sulfanyl prefix. The numbering
    # hint must agree with that (it did not before: the tier anchored the THIOL).
    ("Oc1ccccc1S", "2-sulfanylphenol"),
    #: thiol outranks amine.
    ("Nc1ccccc1S", "2-aminobenzenethiol"),
    ("Sc1ccc(Cl)cc1", "4-chlorobenzenethiol"),
    # A suffix SENIOR to -ol is present, so the OH stays a hydroxy prefix.
    ("Oc1ccc(C(=O)O)cc1", "4-hydroxybenzoic acid"),
    ("Oc1ccc(N)cc1", "4-aminophenol"),
    ("Oc1ccc(Cl)c(Cl)c1", "3,4-dichlorophenol"),
    # Uniform complete substitution: licence, no locants at all.
    ("Oc1c(O)c(O)c(O)c(O)c1O", "benzenehexol"),
]


@pytest.mark.parametrize("smiles,expected", P14_4_NEGATIVES)
def test_negatives_do_not_move(smiles, expected):
    """Rows correct before the (c) fix and still correct after it."""
    assert name_compound(smiles) == expected


# --------------------------------------------------------------------------
# The 17 gold-derived benzene names exposed to a numbering change.
# 10 carry a benzene -ol/phenol suffix plus another locanted prefix; 7 already
# exercise the working is_suffix tier. All are predicted unchanged.
# --------------------------------------------------------------------------

GOLD_DERIVED = [
    ("CC(C)(C)c1cccc(C(C)(C)C)c1O", "2,6-di-tert-butylphenol"),
    ("Cc1cc(COCc2cc(C)c(O)cc2)ccc1O",
     "4,4'-[oxybis(methylene)]bis(2-methylphenol)"),
    ("Oc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]",
     "2,4,6-trinitrophenol"),
    ("Cc1cc(C)cc(O)c1", "3,5-dimethylphenol"),
    ("Cc1ccccc1O", "2-methylphenol"),
    ("Oc1ccccc1C", "2-methylphenol"),
    ("Oc1ccccc1Br", "2-bromophenol"),
    ("Cc1cccc(O)c1C", "2,3-dimethylphenol"),
    ("COc1cccc(O)c1", "3-methoxyphenol"),
    ("[Se]=Cc1ccc(C)cc1", "4-methylbenzene-1-carboselenaldehyde"),
    ("O=C(O)c1ccc(S(=O)(=O)O)cc1C(=O)O",
     "4-sulfobenzene-1,2-dicarboxylic acid"),
    ("Cc1ccc(S(=O)(=O)O)cc1S(=O)(=O)O", "4-methylbenzene-1,3-disulfonic acid"),
    ("Nc1ccc(S(=O)(=O)O)cc1", "4-aminobenzene-1-sulfonic acid"),
    ("CCc1ccc(cc1)S(=O)(=O)OC", "methyl 4-ethylbenzene-1-sulfonate"),
    ("Cc1ccc(cc1)S(=O)(=O)O", "4-methylbenzene-1-sulfonic acid"),
    # ⚠ KNOWN-DEFECT SNAPSHOT, not a gold PIN. The gold row is
    # 'N1-(4-aminophenyl)-N4-phenylbenzene-1,4-diamine'; the emitted form drops the
    # '1' from the first italic-N locant. The ASSIGNMENT is what Task 9b is
    # responsible for and it is correct -- '(4-aminophenyl)' is on the LOWER
    # nitrogen, as (g) requires ('aminoanilino' is cited before 'anilino') --
    # so this row is here to catch the assignment flipping, which it did once
    # during Task 9b before the italic-N prefixes were fed to the (g) tier.
    ("Nc1ccc(Nc2ccc(Nc3ccccc3)cc2)cc1",
     "N-(4-aminophenyl)-N4-phenylbenzene-1,4-diamine"),
    # ⚠ The 17th mandated row, absent from the Task 9 file. Its actual current
    # output is a REFUSAL, which is a real tripwire: the day it becomes nameable
    # this row must be re-derived rather than silently accepting whatever appears.
    # Asserting 'not 4-[(4-hydroxyanilino)methyl]phenol' would be vacuously green
    # against the refusal string -- see feedback_harness_that_reports_success.
    ("Oc1ccc(CNc2ccc(O)cc2)cc1", "unknown organic compound"),
]


@pytest.mark.parametrize("smiles,expected", GOLD_DERIVED)
def test_gold_derived_names_unchanged(smiles, expected):
    """Every gold-derived benzene name exposed to the numbering change."""
    assert name_compound(smiles) == expected


# --------------------------------------------------------------------------
# The re-anchor branch, through the NAME. Task 9b: the branch had
# name-visible consequences that no test asserted -- 'Oc1ccccc1S' ->
# '2-sulfanylphenol' passes with the feature fully reverted, so it was not a
# witness for anything.
# --------------------------------------------------------------------------

P41_REANCHOR_NAMES = [
    # -ol outranks -thiol, so the SH is a 'sulfanyl' prefix and the OH takes the
    # suffix AND locant 1.
    ("Oc1ccccc1S", "2-sulfanylphenol"),
    ("Oc1ccc(S)cc1", "4-sulfanylphenol"),
    ("Oc1cccc(S)c1", "3-sulfanylphenol"),
    # ★ Three groups, so the numbering is genuinely contested: the -ol anchors at
    # 1 and (g) then orders the two prefixes (chloro before sulfanyl).
    ("Oc1cc(S)cc(Cl)c1", "3-chloro-5-sulfanylphenol"),
    ("Oc1c(S)cccc1Cl", "2-chloro-6-sulfanylphenol"),
    # thiol outranks amine, so here the SH takes the suffix.
    ("Nc1ccccc1S", "2-aminobenzenethiol"),
    ("Nc1ccc(S)cc1", "4-aminobenzenethiol"),
]


@pytest.mark.parametrize("smiles,expected", P41_REANCHOR_NAMES)
def test_p41_reanchor_is_name_visible(smiles, expected):
    """ seniority decides which group is the suffix -- asserted as a NAME."""
    assert name_compound(smiles) == expected


# --------------------------------------------------------------------------
# The DISCRIMINATOR itself, pinned directly -- not merely its consequence.
# --------------------------------------------------------------------------

def _pcg_atoms(smiles):
    """The (c) ring-atom set the production wiring actually computes."""
    from rdkit import Chem
    from orthonym.perception.functional_groups import detect_functional_groups
    from orthonym.rules.seniority import get_principal_group

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    ring_atoms = next(
        r for r in mol.GetRingInfo().AtomRings()
        if len(r) == 6 and all(
            mol.GetAtomWithIdx(i).GetIsAromatic()
            and mol.GetAtomWithIdx(i).GetSymbol() == "C" for i in r
        )
    )
    detected = detect_functional_groups(mol)
    pg, _ = get_principal_group(mol, detected)
    subs = get_benzene_substituents(mol, ring_atoms)
    return principal_group_ring_atoms(
        mol, ring_atoms, subs, principal_group=pg, detected_fgs=detected,
    )


def test_discriminator_non_empty_for_the_pentol():
    """The promoted -ol is seen as the PCG: five ring atoms, all hydroxy-bearing."""
    atoms = _pcg_atoms("Oc1c(O)c(O)c(O)c(O)c1Cl")
    assert len(atoms) == 5, atoms


def test_discriminator_empty_for_the_pentamethyl_guard():
    """★ No principal characteristic group at all -> criterion (c) is vacuous."""
    assert _pcg_atoms("Cc1c(C)c(C)c(C)c(C)c1Cl") == set()


def test_discriminator_empty_when_no_suffix_and_no_promotable_group():
    assert _pcg_atoms("Brc1ccc(Br)cc1") == set()


def test_discriminator_picks_only_the_senior_group_p41():
    """: with -ol and -thiol both present only the -ol is the PCG.

    Before this change the set was the THIOL's ring atom (the only ``is_suffix``
    one), i.e. the numbering hint contradicted the emitted ``2-sulfanylphenol``.
    """
    from rdkit import Chem
    atoms = _pcg_atoms("Oc1ccccc1S")
    mol = Chem.MolFromSmiles("Oc1ccccc1S")
    assert len(atoms) == 1, atoms
    # The one atom must be the OH-bearing ring carbon, not the SH-bearing one.
    (idx,) = tuple(atoms)
    assert any(
        n.GetSymbol() == "O" for n in mol.GetAtomWithIdx(idx).GetNeighbors()
    ), f"PCG atom {idx} does not bear the hydroxy group"


def test_discriminator_ignores_hydroxy_when_a_senior_suffix_is_present():
    """A carboxylic acid outranks -ol, so the PCG is the acid's ring atom only.

    ⚠ Task 9b repair: this used to assert only ``len(atoms) == 1``, which is TRUE
    of the wrong answer too -- the hydroxy-bearing atom is also a single atom -- so
    it could not detect the very error it is named for. It now asserts the atom's
    IDENTITY.
    """
    from rdkit import Chem

    smiles = "Oc1ccc(C(=O)O)cc1"
    atoms = _pcg_atoms(smiles)
    mol = Chem.MolFromSmiles(smiles)
    assert len(atoms) == 1, atoms
    (idx,) = tuple(atoms)
    neighbours = list(mol.GetAtomWithIdx(idx).GetNeighbors())
    # The PCG atom must bear the -COOH carbon, never the hydroxy oxygen.
    assert not any(n.GetSymbol() == "O" for n in neighbours), (
        "PCG atom %d bears the hydroxy group, not the senior carboxylic acid" % idx
    )
    assert any(
        n.GetSymbol() == "C" and any(
            b.GetBondTypeAsDouble() == 2.0 for b in n.GetBonds()
        )
        for n in neighbours
    ), "PCG atom %d does not bear the carboxylic acid carbon" % idx
    # And the whole name, so the discriminator's consequence is pinned too.
    assert name_compound(smiles) == "4-hydroxybenzoic acid"


def test_discriminator_excludes_a_co_occurring_junior_SUFFIX_p41():
    """★: with -carboxylic acid AND -sulfonic acid both already ``is_suffix``,
    only the SENIOR one is the principal characteristic group.

    The pre-existing callers passed the UNION of every ``is_suffix`` ring atom, so
    the numberer minimised a set that included a group the name does not spell as
    the suffix. No currently emitted name differs (``3,4-disulfobenzoic acid``
    re-anchors the acid to position 1 downstream regardless), so this contract is
    pinned HERE rather than through a name -- otherwise the filter would be
    silently unprotected and any future junior ring suffix could reintroduce the
    union behaviour undetected.
    """
    from rdkit import Chem
    smiles = "OC(=O)c1ccc(S(=O)(=O)O)c(S(=O)(=O)O)c1"
    atoms = _pcg_atoms(smiles)
    mol = Chem.MolFromSmiles(smiles)
    assert len(atoms) == 1, f"expected the carboxylic acid ring atom only, got {atoms}"
    (idx,) = tuple(atoms)
    # The one atom must carry the -COOH, never one of the two -SO3H.
    assert not any(
        n.GetSymbol() == "S" for n in mol.GetAtomWithIdx(idx).GetNeighbors()
    ), f"PCG atom {idx} bears a sulfonic acid, not the senior carboxylic acid"


def test_discriminator_falls_back_to_legacy_without_a_principal_group():
    """Fail toward current behaviour: no principal group -> the legacy is_suffix set.

    ⚠ Task 9b repair: this used to run on the PENTOL, whose legacy answer is ALSO
    ``set``, so ``== set`` was tautological -- it passed whether the fallback
    returned the legacy union or nothing at all. It now runs on a molecule whose
    legacy ``is_suffix`` union is NON-empty and asserts that exact set, so a
    fallback that returned ``set`` instead would fail.
    """
    from rdkit import Chem
    from orthonym.perception.functional_groups import detect_functional_groups

    smiles = "Cc1ccc(S(=O)(=O)O)cc1"          # 4-methylbenzene-1-sulfonic acid
    mol = Chem.MolFromSmiles(smiles)
    ring_atoms = next(
        r for r in mol.GetRingInfo().AtomRings()
        if len(r) == 6 and all(
            mol.GetAtomWithIdx(i).GetIsAromatic()
            and mol.GetAtomWithIdx(i).GetSymbol() == "C" for i in r
        )
    )
    subs = get_benzene_substituents(mol, ring_atoms)
    detected = detect_functional_groups(mol)

    legacy = {
        atom_idx for atom_idx, entries in subs.items()
        if any(e.get("is_suffix") for e in entries)
    }
    assert len(legacy) == 1, (
        "witness is only meaningful with a NON-empty legacy union: %s" % legacy
    )

    assert principal_group_ring_atoms(
        mol, ring_atoms, subs,
        principal_group=None, detected_fgs=detected,
    ) == legacy

    # Guard the guard: with the principal group supplied the answer is the same
    # single atom, so the row above is testing the FALLBACK and not a coincidence
    # of this molecule having no other candidate.
    assert principal_group_ring_atoms(
        mol, ring_atoms, subs,
        principal_group="sulfonic_acid", detected_fgs=detected,
    ) == legacy


def test_discriminator_respects_the_functional_class_guard():
    """The anchor makes the SAME call as the promotion -- even when that call is wrong.

    ⚠ Task 9b: this test used to present the emitted
    ``1,2,3,4,5-pentahydroxy-6-isocyanatobenzene`` as CORRECT, while the same file
    asserted the opposite treatment for the structurally parallel azide (green only
    because ``'azide'`` was a dead key the detector could never emit). It is now
    labelled for what it is: a snapshot of a KNOWN DEFECT.

    Derived at source, not guessed: ``the Blue Book Blue Book`` item (p)
    records that the -N=C=O group "and its chalcogen analogues... have been added
    to the list of characteristic groups that are ALWAYS cited as prefixes in
    substitutive nomenclature"; ``:26003`` repeats it for isocyanates; and
    ``:26014`` gives ``4-isocyanatobenzene-1-sulfonyl chloride (PIN)`` --
    isocyanato as a detachable prefix WITH a suffix on the parent. A group that can
    never be a suffix cannot outrank -ol, so the PIN is
    ``6-isocyanatobenzene-1,2,3,4,5-pentol``. The PIN is pinned xfail-strict in
    ``test_p14_4_g_total_order.py``; what this test guards is only that the anchor
    and the promotion agree, which is the invariant Task 9b is responsible for.
    """
    assert _pcg_atoms("Oc1c(O)c(O)c(O)c(O)c1N=C=O") == set()
    # The anchor's answer and the NAME must be consistent: hydroxy stays a prefix.
    assert (
        name_compound("Oc1c(O)c(O)c(O)c(O)c1N=C=O")
        == "1,2,3,4,5-pentahydroxy-6-isocyanatobenzene"
    )


def test_the_azide_twin_is_treated_the_OTHER_way_and_that_is_correct():
    """★ The row the dead ``'azide'`` key made accidentally green.

    §** "AZIDES"** (``the Blue Book Blue Book``): azides "are named
    using substitutive nomenclature and the prefix 'azido'. This method gives
    preferred IUPAC names rather than names based on the class name 'azido' in
    functional class nomenclature", with ``:25997``
    ``3-azidonaphthalene-2-sulfonic acid (PIN)`` -- azido cited as a prefix while a
    SUFFIX governs. So azido must NOT block the promotion, the anchor MUST fire,
    and the answer is genuinely right rather than right by accident.
    """
    atoms = _pcg_atoms("Oc1c(O)c(O)c(O)c(O)c1N=[N+]=[N-]")
    assert len(atoms) == 5, atoms
    assert (
        name_compound("Oc1c(O)c(O)c(O)c(O)c1N=[N+]=[N-]")
        == "6-azidobenzene-1,2,3,4,5-pentol"
    )


def _parent_selection_locant_hint(smiles):
    """The benzene locant HINT that namer.py Branch 3 publishes for the ring.

    ⚠ Task 9b repair: this helper used to ASSIGN ``features.principal_group``
    itself, which concealed the fact that the whole anchor was a NO-OP for every
    external caller -- ``compute_features`` runs ``_perceive`` only, so the
    attribute is ``None`` there. Nothing is assigned now; the production fallback
    inside ``_build_ring_info_for_parent_selection`` has to do the work.
    """
    from rdkit import Chem
    from orthonym.namer import (
        compute_features, _build_ring_info_for_parent_selection,
    )

    features = compute_features(Chem.MolFromSmiles(smiles), smiles)
    assert getattr(features, "principal_group", None) is None, (
        "compute_features now populates principal_group -- this helper's premise "
        "is stale and the production fallback it exercises may be dead"
    )
    info = _build_ring_info_for_parent_selection(features)
    assert info and info.get("iupac_locants"), f"no locant hint produced for {smiles}"
    return features.mol, info["iupac_locants"]


def _hint_locants_by_neighbour(mol, hint, symbol):
    return {
        locant for atom_idx, locant in hint.items()
        if any(n.GetSymbol() == symbol
               for n in mol.GetAtomWithIdx(atom_idx).GetNeighbors())
    }


def test_locant_hint_agrees_with_the_emitted_name():
    """The parent-selection locant HINT and the emitted NAME must agree.

    namer.py Branch 3 publishes an independent benzene numbering used for parent
    selection and stereo locants. It derived its (c) anchor from ``is_suffix``
    too, so for a phenol it disagreed with the name the composer emitted. Both now
    consult ``principal_group_ring_atoms``. Without this the two could drift apart
    silently -- no name-level test can see it.
    """
    # The five hydroxy-bearing ring atoms must hold locants 1-5, chloro gets 6,
    # matching '6-chlorobenzene-1,2,3,4,5-pentol'.
    mol, hint = _parent_selection_locant_hint("Oc1c(O)c(O)c(O)c(O)c1Cl")
    assert _hint_locants_by_neighbour(mol, hint, "O") == {1, 2, 3, 4, 5}, hint
    assert _hint_locants_by_neighbour(mol, hint, "Cl") == {6}, hint

    # ★ The guard: no principal characteristic group, so (g) gives locant 1 to
    # chloro -- matching '1-chloro-2,3,4,5,6-pentamethylbenzene'.
    mol, hint = _parent_selection_locant_hint("Cc1c(C)c(C)c(C)c(C)c1Cl")
    assert _hint_locants_by_neighbour(mol, hint, "Cl") == {1}, hint


# ⚠ Task 9b repair: the Task 9 version of the test below used two molecules for
# which production never computes this hint at all, so it proved nothing about the
# unified anchor. These two DO route through it, and each asserts the hint AND the
# whole emitted name, so a hint that silently disagreed with the name would fail.
HINT_AND_NAME = [
    # A mono-hydroxy phenol with two bulky prefixes: the ONE oxygen-bearing ring
    # atom must hold locant 1, and the two quaternary carbons 2 and 6.
    ("CC(C)(C)c1cccc(C(C)(C)C)c1O", "2,6-di-tert-butylphenol", "O", {1}),
    # Two oxygen-bearing ring atoms, only ONE of which is the promoted -ol. The
    # anchor must put the phenol OH at 1 and leave the methoxy at 3.
    ("COc1cccc(O)c1", "3-methoxyphenol", "O", {1, 3}),
]


@pytest.mark.parametrize("smiles,expected,symbol,locants", HINT_AND_NAME)
def test_locant_hint_is_computed_for_molecules_production_actually_uses(
    smiles, expected, symbol, locants
):
    """The hint exists, is anchored, and agrees with the emitted whole name."""
    mol, hint = _parent_selection_locant_hint(smiles)
    assert _hint_locants_by_neighbour(mol, hint, symbol) == locants, hint
    assert name_compound(smiles) == expected


def test_discriminator_is_wired_into_the_live_orientation_call(monkeypatch):
    """The set reaches ``orient_benzene`` -- the tier is fed, not just computed.

    Guards the failure mode of session a project rule: a helper that is correct but
    never called. Validated against a known positive (the pentol) AND a known
    negative (the pentamethyl guard) in the same run, so a trace that recorded
    nothing could not pass.
    """
    seen = []
    original = benzene_rules.orient_benzene

    def recording(mol, ring_atoms, substituents, principal_group_positions=None):
        seen.append(principal_group_positions)
        return original(
            mol, ring_atoms, substituents,
            principal_group_positions=principal_group_positions,
        )

    monkeypatch.setattr(benzene_rules, "orient_benzene", recording)

    name_compound("Oc1c(O)c(O)c(O)c(O)c1Cl")
    assert seen, "orient_benzene was never called -- the spy proved nothing"
    assert any(p and len(p) == 5 for p in seen), seen

    seen.clear()
    name_compound("Cc1c(C)c(C)c(C)c(C)c1Cl")
    assert seen, "orient_benzene was never called for the guard molecule"
    assert all(not p for p in seen), seen
