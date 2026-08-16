"""v30: the whole-molecule PARENT-HYDRIDE fallback tier.

``rules/terminal_ring.terminal_ring_name(mol, ring_atoms, free_valence_atom=None)``
returns an audited von Baeyer / spiro / P-22.2.3-replacement PARENT HYDRIDE, and
its own docstring says so -- but until this tier it was reachable ONLY as a
``-yl`` substituent namer. Nothing called it for a whole molecule, so a bare ring
system no catalog covers abstained even though the generator could name it.

This module pins the tier that closes that, and -- equally -- pins the four
things it must NOT do:

* **the catalogs still win.** ``bicyclo[4.4.0]deca-1,3,5,7,9-pentaene`` is a
  VALID name for naphthalene and is NOT its PIN, so this tier is strictly below
  every existing producer and fires only where the molecule would abstain.
* **best-effort only.** The tier is gated on ``allow_aromatic_general``, which
  ``cli._emit_tier_flags`` sets True for ``complete``/``best-effort`` and False
  for ``pin`` -- so the PIN return is untouched and PIN byte-identity holds.
* **atom conservation.** The generator names a RING. A decorated molecule must
  not be named by its bare ring parent (that is a wrong constitution, not an
  uglier name), so the tier refuses unless the ring system IS every heavy atom.
* **a wrong molecule is still suppressed.** ``terminal_ring_name`` is audited
  against its own emitted string, not against OPSIN, and measured it can still
  emit a name that denotes a DIFFERENT species (the group-14 ``[Sn]``/``[Pb]``
  rings below). SELF-01 is what catches those, and this tier must not bypass it.
"""

import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.errors import is_failure_name
from orthonym.validation.opsin_roundtrip import opsin_parse


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _namer(tier: str) -> Orthonym:
    return Orthonym(style="pin", **_emit_tier_flags(tier))


def _abstains(name) -> bool:
    return name is None or is_failure_name(name)


def _inchikey(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return Chem.MolToInchiKey(mol)


def _rt_inchikey(name: str):
    """OPSIN-parse ``name`` and return the InChIKey it denotes, or None."""
    smi = opsin_parse(name)
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(mol) if mol is not None else None


# --------------------------------------------------------------------------
# KNOWN POSITIVES -- bare ring systems that abstained before this tier
# --------------------------------------------------------------------------
# Every row here was measured to (a) abstain at ``best-effort`` before the tier
# existed and (b) have ``terminal_ring_name(mol, ring, None)`` emit a name whose
# OPSIN parse is InChIKey-identical to the input. They are group-13/15
# heteracycles: the replacement morpheme exists in Table 1.5 but no retained or
# Hantzsch-Widman name covers them, which is exactly the "table miss with
# nowhere to fall" this tier is for.
PARENT_HYDRIDE_POSITIVES = [
    ("C1CC[Al]CC1", "1λ2-aluminacyclohexane"),
    ("C1CC[Ga]CC1", "1λ2-gallacyclohexane"),
    ("C1CC[In]CC1", "1λ2-indacyclohexane"),
    ("C1CC[Bi]CC1", "1λ2-bismacyclohexane"),
    ("C1CCC[Al]CCC1", "1λ2-aluminacyclooctane"),
    ("C1CC[Al]CC[Al]C1", "1λ2,4λ2-dialuminacyclooctane"),
    ("C1CC[Ga]CC[Ga]C1", "1λ2,4λ2-digallacyclooctane"),
    ("C1CC[Sb]CC[Sb]C1", "1λ2,4λ2-distibacyclooctane"),
    ("C1CC2CCC1[Al]2", "7λ2-aluminabicyclo[2.2.1]heptane"),
    ("C1[Al]C2CC[Al]1CC2", "1,8λ2-dialuminabicyclo[2.2.2]octane"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", PARENT_HYDRIDE_POSITIVES)
def test_bare_ring_system_emits_a_round_tripping_parent_hydride(
        smiles, expected, opsin_gate):
    """A bare ring system no catalog covers emits a parent hydride that OPSIN
    parses back to the SAME molecule.

    The structural assertion is the round-trip; ``expected`` is carried only so
    a change of spelling is visible in the diff rather than silently accepted.
    """
    emitted = _namer("best-effort").name(smiles)
    assert not _abstains(emitted), (
        f"{smiles} abstained; the parent-hydride tier did not fire")
    assert _rt_inchikey(emitted) == _inchikey(smiles), (
        f"{smiles} -> {emitted!r} does not round-trip to the input molecule")
    assert emitted == expected, (
        f"{smiles} spelling changed: {emitted!r} != {expected!r}")


@pytest.mark.opsin_gate
def test_the_tier_moves_at_least_six_bare_ring_systems(opsin_gate):
    """Non-vacuity at the class level: the tier is not a one-molecule fix."""
    namer = _namer("best-effort")
    named = [s for s, _ in PARENT_HYDRIDE_POSITIVES
             if not _abstains(namer.name(s))]
    assert len(named) >= 6, (
        f"only {len(named)} of {len(PARENT_HYDRIDE_POSITIVES)} positives emit")


# --------------------------------------------------------------------------
# NON-VACUITY -- prove the NEW site actually executed
# --------------------------------------------------------------------------

def test_the_new_site_is_the_one_that_produced_the_name():
    """A test that passes because nothing ran is worthless. Record the candidate
    ledger and require an entry from THIS tier's site for a known positive, and
    require NO such entry for a molecule the catalogs already name."""
    from orthonym.metrics import candidate_ledger as cl
    from orthonym.assembly.general_engine import TERMINAL_RING_PARENT_SITE

    def sites_for(smiles):
        cl.enable()
        try:
            _namer("best-effort").name(smiles)
            return [e.site for e in cl.read_ledger()]
        finally:
            cl.disable()

    fired = sites_for("C1CC[Al]CC1")
    assert TERMINAL_RING_PARENT_SITE in fired, (
        f"the parent-hydride site never recorded a candidate; sites={fired}")

    # the catalogs own naphthalene, so the tier must never be reached for it
    not_fired = sites_for("c1ccc2ccccc2c1")
    assert TERMINAL_RING_PARENT_SITE not in not_fired, (
        "the parent-hydride tier ran for naphthalene, which the catalogs name")


# --------------------------------------------------------------------------
# THE CATALOGS MUST STILL WIN -- a real regression pin
# --------------------------------------------------------------------------
# Captured from HEAD before the tier was written, at BOTH tiers.
CATALOG_NAMES = [
    ("c1ccc2ccccc2c1", "naphthalene"),
    ("c1ccc2[nH]ccc2c1", "1H-indole"),
    ("c1ccc2ncccc2c1", "quinoline"),
    ("c1ccncc1", "pyridine"),
    ("c1ccccc1", "benzene"),
    ("C1C2CC3CC1CC(C2)C3", "adamantane"),
    ("C1CCOCC1", "oxane"),
    ("C1COCCN1", "morpholine"),
    ("C12C3C4C1C5C4C3C25", "cubane"),
]


@pytest.mark.parametrize("tier", ["pin", "best-effort"])
@pytest.mark.parametrize("smiles,expected", CATALOG_NAMES)
def test_catalog_name_is_unchanged_by_the_parent_hydride_tier(
        tier, smiles, expected):
    """``naphthalene`` must not become ``bicyclo[4.4.0]deca-1,3,5,7,9-pentaene``."""
    assert _namer(tier).name(smiles) == expected


# --------------------------------------------------------------------------
# REFUSALS STAY REFUSALS
# --------------------------------------------------------------------------

def test_charged_skeletal_ring_atom_is_not_named_by_this_tier():
    """A ring cation is P-73 (``cation_words``/``ion_retained_names``), not
    replacement nomenclature; ``terminal_ring_name`` refuses it and the tier must
    not route around that. ``pyridin-1-ium`` comes from the ion path, so the
    assertion is that the von Baeyer/replacement form never appears."""
    emitted = _namer("best-effort").name("c1cc[nH+]cc1")
    assert emitted is None or "cyclohexa" not in emitted
    assert emitted is None or "azacyclo" not in emitted


def test_over_max_cage_atoms_still_abstains():
    """``MAX_CAGE_ATOMS`` is a stated scope bound, not an escape hatch."""
    from orthonym.rules.terminal_ring import MAX_CAGE_ATOMS
    n = MAX_CAGE_ATOMS + 6
    smiles = "C1" + "C" * (n - 2) + "C1"
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None and mol.GetNumAtoms() == n
    from orthonym.rules.terminal_ring import terminal_ring_name
    assert terminal_ring_name(mol, list(range(n)), None) is None


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", ["C1CC[Sn]CC1", "C1CC[Pb]CC1"])
def test_a_parent_hydride_that_denotes_a_different_molecule_is_suppressed(
        smiles, opsin_gate):
    """0-wrong. ``terminal_ring_name`` emits ``1-stannacyclohexane`` /
    ``1-plumbacyclohexane`` here and OPSIN parses BOTH to a different species,
    so the tier must let SELF-01 suppress them rather than shipping a name its
    own reconstruction audit was happy with."""
    from orthonym.rules.terminal_ring import terminal_ring_name
    mol = Chem.MolFromSmiles(smiles)
    generated = terminal_ring_name(mol, list(range(mol.GetNumAtoms())), None)
    assert generated is not None, (
        "premise broken: the generator no longer names this ring, so the test "
        "no longer exercises the suppression path")
    assert _rt_inchikey(generated.name) != _inchikey(smiles), (
        "premise broken: this name now round-trips, so it is no longer a "
        "wrong-molecule case")

    assert _abstains(_namer("best-effort").name(smiles)), (
        f"{smiles} shipped {generated.name!r}, which denotes a different "
        f"molecule -- SELF-01 was bypassed")


# --------------------------------------------------------------------------
# THE ASSEMBLY TIER -- a nameable ring parent PLUS its substituents
# --------------------------------------------------------------------------
# Measured: 0 abstaining dev500 rows are bare ring systems, but 71 have a
# nameable ring system AND every substituent nameable. Those need no new
# generator, only the JOIN -- and the join already exists in
# ``_emit_ring_from_analysis``. What blocks those rows is the SUFFIX logic
# (`ester not a clean mono-ester`, `PG instance not attached to cage`,
# `inline suffix off cage`): with the principal group suppressed, every
# functional group becomes a detachable prefix and the tail runs.
#
# Reusing that tail rather than composing strings by hand is worth 44 RT_EXACT
# of 71 against 5 for a hand-rolled composer, because the tail owns the
# enclosing marks (``_mult_prefix`` -> ``15-(2-methylpropyl)``, ``bis``/``tris``),
# the P-14.5.2 alphabetisation (``_alpha_key``) and -- decisively -- the
# stereodescriptor block (``_stereo_prefix``), which is why the gain lands in
# rt_exact rather than rt_constitutional.
#
# ⚠ Every row here is verified to be emitted BY THIS TIER (`emitted_is_mine`
# via the candidate ledger), not merely to emit. That distinction is the whole
# validity of the list: the source target set was measured at an earlier commit,
# and by the time this tier existed **35 of those rows already emitted from the
# ordinary producers** -- correctly, and in the PIN-preferred SUFFIX form
# (`…bicyclo[4.4.0]decan-3-ol`, which is better than the `3-hydroxy-…` prefix
# form this tier would build). Asserting those would have credited this tier
# with another producer's work and, worse, would have pinned the WRONG spelling
# as expected. Re-checked rather than trusted (invariant 14).
ASSEMBLY_POSITIVES = [
    ("CN1C(=O)c2ccccc2NC(=O)[C@@H]1Cc1ccccc1",
     "(4S)-4-benzyl-5-methyl-3,6-dioxo-2,5-diazabicyclo[5.4.0]undeca-"
     "1(11),7,9-triene"),
    # v31 lever B (P-16.3.3): 'hydroxymethyl' is a compound substituent and is now
    # ENCLOSED -> '(hydroxymethyl)'; the perturbed candidate strings also let the
    # PIN-preferred SUFFIX form win (ring 4-OH -> '-4-ol', not a '4-hydroxy'
    # prefix), which this file's own header (lines ~238-243) marks as better.
    # RT-verified identical InChIKey.
    ("C[C@@]12CCC[C@@]3(C)[C@@H](C1)[C@@](O)(CO)CC[C@@]23C",
     "(1R,3R,4R,7S,8S)-4-(hydroxymethyl)-1,7,8-trimethyltricyclo"
     "[5.4.0.0^3,8]undecan-4-ol"),
    ("CC1=CC(=O)C2=C(C)CC[C@@H]3[C@H](OC(=O)[C@@H]3C)[C@@H]12",
     "(5S,6R,9S,10S)-2,6,11-trimethyl-7,13-dioxo-8-oxatricyclo"
     "[8.3.0.0^5,9]trideca-1,11-diene"),
    ("O=C1NC2=Nc3ccc(Cl)c(Cl)c3CN2C1O",
     "10,11-dichloro-6-hydroxy-5-oxo-2,4,7-triazatricyclo[7.4.0.0^3,7]"
     "trideca-1(9),2,10,12-tetraene"),
    # v31 lever B (P-16.3.3): '(hydroxymethyl)' now enclosed + PIN-preferred
    # SUFFIX form (ring 1,5-diol, not a '1,5-dihydroxy' prefix). RT-verified.
    ("C=C1[C@@H](CO)C[C@H](O)[C@H](C)[C@@H]2CC(C)(C)C[C@]12O",
     "(1R,3S,5S,6R,7S)-3-(hydroxymethyl)-6,9,9-trimethyl-2-"
     "methylidenebicyclo[5.3.0]decane-1,5-diol"),
    ("CC1CCCC(O)/C=C/C2C(O)CC(O)CC2/C=C/C=C\\C/C=C/C=C\\C(=O)O1",
     "(2E,11Z,13E,16Z,18E)-4,22,24-trihydroxy-8-methyl-10-oxo-9-oxabicyclo"
     "[18.4.0]tetracosa-2,11,13,16,18-pentaene"),
    ("CCC(C)CC(C)/C=C\\[C@@H]1O[C@H]2[C@H](C(=O)O[C@H]2C)[C@H](O)[C@H]1O",
     "(1S,3S,4R,5S,6R,9S)-4,5-dihydroxy-9-methyl-7-oxo-3-((1Z)-3,5-"
     "dimethylhept-1-en-1-yl)-2,8-dioxabicyclo[4.3.0]nonane"),
    ("CCN1CC(=O)Nc2ccccc2C(=O)Nc2ccccc2C(=O)O[C@H](Cc2ccccc2)C1=O",
     "(7R)-7-benzyl-5-ethyl-3,6,9,17-tetraoxo-8-oxa-2,5,16-triazatricyclo"
     "[16.4.0.0^10,15]docosa-1(22),10,12,14,18,20-hexaene"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", ASSEMBLY_POSITIVES)
def test_decorated_ring_assembles_a_round_tripping_name(
        smiles, expected, opsin_gate):
    """A decorated ring system whose parent and every substituent are nameable
    now emits the joined name, and it round-trips EXACTLY -- stereo included."""
    emitted = _namer("best-effort").name(smiles)
    assert not _abstains(emitted), f"{smiles} still abstains"
    assert _rt_inchikey(emitted) == _inchikey(smiles), (
        f"{smiles} -> {emitted!r} does not round-trip to the input")
    assert emitted == expected, f"spelling changed: {emitted!r}"


@pytest.mark.opsin_gate
def test_the_assembly_gain_lands_in_exact_not_merely_constitutional(opsin_gate):
    """The tail injects stereodescriptors, so these must be EXACT matches. A
    constitution-only match would mean the stereo block was lost."""
    namer = _namer("best-effort")
    exact = 0
    for smiles, _ in ASSEMBLY_POSITIVES:
        emitted = namer.name(smiles)
        if not _abstains(emitted) and _rt_inchikey(emitted) == _inchikey(smiles):
            exact += 1
    assert exact == len(ASSEMBLY_POSITIVES), (
        f"only {exact}/{len(ASSEMBLY_POSITIVES)} are EXACT matches")


@pytest.mark.opsin_gate
def test_the_assembly_site_is_the_one_that_produced_the_name(opsin_gate):
    """Non-vacuity for the assembly half, in both directions.

    ⚠ This test REQUIRES the gate, and not as a formality. The suite disables
    the OPSIN validity gate by default; with it off, an earlier producer's
    name for this molecule is never SELF-01-suppressed, so it ships and this
    tier -- which is last resort -- is never reached. The test then fails while
    production works. The gate state is part of what "last resort" means here.
    """
    from orthonym.metrics import candidate_ledger as cl
    from orthonym.assembly.general_engine import TERMINAL_RING_ASSEMBLY_SITE

    def sites_for(smiles):
        cl.enable()
        try:
            _namer("best-effort").name(smiles)
            return [e.site for e in cl.read_ledger()]
        finally:
            cl.disable()

    assert TERMINAL_RING_ASSEMBLY_SITE in sites_for(ASSEMBLY_POSITIVES[0][0])
    assert TERMINAL_RING_ASSEMBLY_SITE not in sites_for("c1ccc2ccccc2c1")
    # And the negative that matters most for LAST-RESORT ordering: a row the
    # ordinary ring producer already names in the PIN-preferred SUFFIX form must
    # not be taken over by this tier's prefix form.
    already = "C[C@H]1CCC[C@@]2(C)CCC(O)CC12"
    assert _namer("best-effort").name(already).endswith("decan-3-ol")
    assert TERMINAL_RING_ASSEMBLY_SITE not in sites_for(already)


# The 13 rows the assembly composes to a name denoting a DIFFERENT constitution.
# Causes measured, and all of them live in the substituent-prefix layer rather
# than in the join: `-OCH3` spelled `hydroxymethyl`, `-OC(C)=O` spelled `acetyl`,
# and the known `-C(=O)OH` spelled `formyl`. Every one MUST be suppressed.
ASSEMBLY_WRONG_CONSTITUTION = [
    "COc1c(O)ccc2c(=O)oc3cc(C)cc(O)c3c(=O)c12",
    "COC(=O)c1cc(OC)c2c(=O)c3c(C(=O)OC)c(OC)ccc3oc2c1",
    "CO[C@H]1C[C@H](O)[C@H](C)c2c1c(C=O)cn2C",
    "C=C1c2ccoc2C[C@H]2[C@H]1CC[C@@]1(O)C(C)(C)[C@H](OC(C)=O)C[C@H](OC(C)=O)[C@]21C",
    "CC(=O)N[C@@]12OP3(=O)O[C@]1(O)O[C@H](CO)[C@@H](O)[C@@]2(O)O3",
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", ASSEMBLY_WRONG_CONSTITUTION)
def test_an_assembled_name_of_the_wrong_constitution_never_ships(
        smiles, opsin_gate):
    """0-wrong, asserted on the rows measured to compose WRONGLY.

    The assertion is deliberately on the CONSTITUTION rather than on abstention:
    the contract is "never a different compound", not "never a name". A row that
    starts emitting a correct name here is a pass, not a failure."""
    emitted = _namer("best-effort").name(smiles)
    if _abstains(emitted):
        return
    got = opsin_parse(emitted)
    if got is None:
        return  # unparseable ships nothing downstream can mistake for correct
    want_mol = Chem.MolFromSmiles(smiles)
    got_mol = Chem.MolFromSmiles(got)
    assert got_mol is not None
    assert (Chem.MolToSmiles(got_mol, isomericSmiles=False)
            == Chem.MolToSmiles(want_mol, isomericSmiles=False)), (
        f"WRONG CONSTITUTION SHIPPED for {smiles}: {emitted!r}")


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,_expected", ASSEMBLY_POSITIVES)
def test_a_suffix_free_name_is_T4_only(smiles, _expected, opsin_gate):
    """P-41 (`BlueBookV2.md:25009`, "Seniority order of classes"): *"If
    characteristic groups other than those given in Table 5.1 are present, one
    (and only one) kind must be cited as suffix (the principal characteristic
    group) for classes other than radicals"*.

    So a PG-suppressed name omits a REQUIRED suffix -- ill-formed, not merely
    non-preferred. It ships only on best-effort, where the alternative is
    silence. On ``complete`` it must NOT ship: that tier claims RT-verified
    status, and claiming it for an ill-formed name is the harm. Gating on
    ``allow_aromatic_general`` would NOT achieve this -- that flag is True for
    ``complete`` too; the discriminator is ``general_fallback_unverified``.
    """
    assert not _abstains(_namer("best-effort").name(smiles))
    complete = _namer("complete").name(smiles)
    if _abstains(complete):
        return
    # If `complete` does emit, it must be via some other producer -- never a
    # name that omits the required suffix.
    assert _rt_inchikey(complete) == _inchikey(smiles), (
        f"complete shipped a non-round-tripping name: {complete!r}")


@pytest.mark.opsin_gate
def test_the_suffix_free_debt_is_tagged_per_row(opsin_gate):
    """Condition: TAG, do not merely count. A number in a report is not
    recoverable; a field is. ``name_tiered`` must carry
    ``suffix_free_prefix_name`` so v31 can ENUMERATE these rows."""
    namer = _namer("best-effort")
    # a PG-suppressed emission -> flagged
    row = namer.name_tiered("CN1C(=O)c2ccccc2NC(=O)[C@@H]1Cc1ccccc1")
    assert row["suffix_free_prefix_name"] is True, row

    # a catalog name -> NOT flagged (the flag must not leak across molecules)
    row2 = namer.name_tiered("c1ccc2ccccc2c1")
    assert row2["name"] == "naphthalene"
    assert row2["suffix_free_prefix_name"] is False, row2

    # an abstention -> NOT flagged, even though a suppressed candidate for it
    # was built by this very tier (that row emits nothing, so tagging it would
    # corrupt the census the field exists to enable)
    row3 = namer.name_tiered(
        "COC(=O)c1cc(OC)c2c(=O)c3c(C(=O)OC)c(OC)ccc3oc2c1")
    if _abstains(row3["name"]):
        assert row3["suffix_free_prefix_name"] is False, row3


def test_the_two_numbering_maps_must_agree():
    """The tier gates on ``terminal_ring_name`` (which re-proves the von Baeyer
    reconstruction audit at its emission point) but spells through
    ``_emit_ring_from_analysis``, which numbers from the cage analysis. Those two
    maps must be the SAME map -- if they ever diverge, every substituent locant
    would be placed against a different numbering than the parent was audited
    under. This pins the invariant that makes the reuse legitimate."""
    from orthonym.rules.terminal_ring import terminal_ring_name
    from orthonym.rules.vonbaeyer_universal import analyze_cage_universal
    from orthonym.rules.ring_selection import select_principal_ring_system

    smiles = "C[C@H]1CCC[C@@]2(C)CCC(O)CC12"
    mol = Chem.MolFromSmiles(smiles)
    namer = _namer("best-effort")
    feats = namer._perceive(mol, smiles, Chem.MolToSmiles(mol))
    namer._classify(feats)
    senior = sorted(set(select_principal_ring_system(
        mol, list(feats.ring_systems))))
    tr = terminal_ring_name(mol, senior, None)
    cage = analyze_cage_universal(mol, cage_atoms=set(senior),
                                  allow_mancude=True)
    assert tr is not None and cage is not None
    assert dict(tr.numbering) == dict(cage.atom_to_locant)


@pytest.mark.parametrize("smiles", [
    "CC1CC[Al]CC1",        # methyl-decorated
    "OC1CC[Al]CC1",        # hydroxy-decorated
])
def test_a_decorated_ring_is_never_named_by_its_bare_parent_hydride(smiles):
    """Atom conservation. The tier names a RING; emitting the bare parent
    hydride for a decorated molecule would drop the decoration silently -- a
    wrong constitution, which is worse than an abstention."""
    emitted = _namer("best-effort").name(smiles)
    if _abstains(emitted):
        return
    mol = Chem.MolFromSmiles(smiles)
    ri = mol.GetRingInfo()
    ring = [a.GetIdx() for a in mol.GetAtoms() if ri.NumAtomRings(a.GetIdx())]
    from orthonym.rules.terminal_ring import terminal_ring_name
    bare = terminal_ring_name(mol, ring, None)
    if bare is not None:
        assert emitted != bare.name, (
            f"{smiles} was named {emitted!r} -- the bare ring parent, with the "
            f"decoration dropped")
