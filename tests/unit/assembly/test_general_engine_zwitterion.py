"""P-74.1.2 / P-74.1.1 — zwitterionic ionic centres in the general-engine parent.

R7 (v29 residue). A mesoionic zwitterion was emitted by the best-effort tier as a
NEUTRAL name: the ring ``[N+]`` was spelled ``aza`` like any neutral ring N, and
the ``[O-]`` was spelled as a neutral ``oxo`` prefix. The result described a
molecule that cannot exist (OPSIN: "Atom is in unphysical valency state!
Element: C valency: 5") and was one of the remaining T6 emitted-but-unparseable
rows.

Blue Book, ``BlueBookV2/BlueBookV2.md`` (every line pointer below re-verified
with ``sed -n '<N>p'``):

* **P-74.1.2 "Zwitterionic compounds with at least one ionic center on a
  characteristic group"** (heading ``:42445``) is the GOVERNING case for R7,
  whose shape is a skeletal ring ``-ium`` plus a characteristic-group-derived
  ``-olate``. Sentence ``:42447``: *"Zwitterionic compounds with at least one
  ionic center on a characteristic group may be named by adding the appropriate
  ionic suffix to the name of the ionic parent hydride.  In names, cationic
  suffixes are cited before anionic suffixes.  For assignment of lower locants,
  ionic centers on skeletal atoms of the parent hydride are preferred to the
  locants for positions of attachment of characteristic groups denoted by ionic
  suffixes."*  Worked (PIN) example ``:42456``:
  ``1-methyl-4,6-diphenylpyridin-1-ium-2-carboxylate``.
* **P-74.1.1 "Ionic centers in the same parent structure"** (heading ``:42415``)
  governs the other branch — both centres skeletal — and supplies the
  construction and elision used by both. ``:42419``: *"For nomenclature
  purposes, zwitterionic compounds having the ionic centers in the same parent
  structure are not considered as neutral compounds."*  Construction ``:42417``:
  *"…may be named by combining appropriate cumulative suffixes at the end of the
  name of a parent hydride in the order 'ium', 'ylium', 'ide', 'uide'. … anionic
  suffixes are cited after cationic suffixes … The final letter 'e' of the name
  of a parent hydride … is elided before the letter 'i' or 'y', or before a
  cumulative suffix beginning with a vowel."*
* The Blue Book's own worked example of exactly this defect, ``:42439``:
  ``2-methyl-4-oxo-3,4-dihydro-1H-2-benzoselenopyran-2-ium-3-ide (PIN)``
  *(not ``2-methyl-3,4-dihydro-1H-2-benzoselenopyran-2-ium-3-id-4-one``)*.
* Elision is **P-16.7 "ELISION OF VOWELS"** (``:7591``), clause
  **P-16.7.1(a)** (``:7595``) — not P-16.3.3, which is the multiplying-prefix
  rule ("The basic numerical prefixes 'di', 'tri', 'tetra', etc.").

The fix widens the EXISTING v26-P5 charge-suffix layer
(``general_engine._charge_suffix_text`` / ``_append_charge_suffix``), which was
scoped to NET molecular charge and therefore never saw a zwitterion (net 0).
"""

import pytest
from rdkit import Chem, RDLogger

from orthonym.assembly import general_engine as ge
from orthonym.namer import Orthonym

RDLogger.DisableLog("rdApp.*")


# The R7 row. Mesoionic: ring [N+] and an [O-] on a ring carbon.
MESOIONIC = "CC1=N[N+]2=CC=CC=C2C(=N1)[O-]"

# Coordinator-verified, re-verified in this session: fed to opsin-cli-2.9.0 the
# returned structure has InChIKey YZYLEZPSMVVLED-UHFFFAOYSA-N, identical to
# MESOIONIC's InChIKey.
TARGET = ("4-methyl-3,5,6-triazabicyclo[4.4.0]deca-1(10),2,4,6,8-"
          "pentaen-6-ium-2-olate")


def _name(smiles, tier="best-effort"):
    """Name ``smiles`` at ``tier`` using the CLI's own tier->flag contract."""
    from orthonym.cli import _emit_tier_flags

    return Orthonym(style="pin", **_emit_tier_flags(tier)).name(smiles)


# --------------------------------------------------------------------------
# The target row
# --------------------------------------------------------------------------

def test_mesoionic_zwitterion_emits_cumulative_ium_olate():
    """P-74.1.1: the cumulative ``-6-ium-2-olate``, not a neutral ``2-oxo``."""
    assert _name(MESOIONIC) == TARGET


def test_mesoionic_zwitterion_no_longer_emits_neutral_oxo():
    """The specific wrong rendering must be gone (P-74.1.1 :42419)."""
    name = _name(MESOIONIC)
    assert "oxo" not in name
    assert name.endswith("-olate")


def test_mesoionic_cationic_suffix_precedes_anionic():
    """P-74.1.2 :42447 — cationic suffixes are cited before anionic ones."""
    name = _name(MESOIONIC)
    assert name.index("-ium") < name.index("-olate")


def test_mesoionic_parent_hydride_e_is_elided():
    """P-74.1.1 :42417 — final 'e' elided before a suffix beginning 'i'."""
    name = _name(MESOIONIC)
    assert "pentaen-6-ium" in name
    assert "pentaene-6-ium" not in name


def test_mesoionic_pin_tier_unchanged():
    """No PIN byte may move: the PIN tier abstained before and must still."""
    assert _name(MESOIONIC, "pin") == "unknown organic compound"


# --------------------------------------------------------------------------
# The plan primitive, directly
# --------------------------------------------------------------------------

def _plan_for(smiles):
    """Build the parent numbering the ring tail would use, then plan."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return mol


def test_plan_declines_when_cation_is_off_parent():
    """A cation that is not a numbered parent atom is not a parent suffix.

    ISOLATION (code review): the previous input (betaine with a 2-atom
    ``atom_to_locant``) also tripped the later, independent anion-class check
    ('carboxylate' is not an olate class), so it did not prove the line its
    docstring claims. This input is MESOIONIC with the SAME numbering the ring
    tail builds, minus the cationic ring N -- measured to trip the
    cation-off-parent check and NOTHING else, so deleting that check makes this
    test fail.
    """
    mol = Chem.MolFromSmiles(MESOIONIC)
    # atom 3 = the ring [N+]; atom 11 = the anionic O (never a numbered atom).
    full = {i: i for i in range(mol.GetNumAtoms()) if i != 11}
    assert ge._zwitterion_suffix_plan(mol, full) is not None, (
        "control: with the cation numbered, the plan must build -- otherwise "
        "this test would pass for the wrong reason")
    off_parent = {i: i for i in full if i != 3}
    assert ge._zwitterion_suffix_plan(mol, off_parent) is None


def test_plan_declines_on_net_charged_molecule():
    """Net-charged species belong to the existing net-charge suffix path.

    ISOLATION (code review): no real SMILES can trip the net-charge guard
    ALONE. Proof: the guard is only redundant-free if every later check passes,
    which requires exactly one genuine cation of +1 and one genuine anion of -1
    (they sum to 0); ``get_ion_sites`` lists EVERY non-zero-charge atom and
    subtracts only the P-59 internal-charge groups (nitro / N-oxide / azide /
    diazo), and ``_genuine_ion_sites`` subtracts only semipolar ``[X+]-[O-]``
    pairs -- every one of those subtracted sets is itself net 0. So the
    molecular net charge is 0 whenever the later checks pass. The old input
    ``C[N+](C)(C)C`` therefore also tripped ``len(anions) != 1``.

    The check is isolated here by supplying a valid one-cation/one-anion site
    pair for a net-charged molecule, which is exactly the state the guard
    exists to reject.
    """
    mol = Chem.MolFromSmiles("C[N+]12[CH-]CC(CC1)CC2")   # net 0, plan builds
    a2l = {a.GetIdx(): a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() != 0}
    assert ge._zwitterion_suffix_plan(mol, a2l) is not None, "control"

    charged = Chem.MolFromSmiles("C[N+]12CCC(CC1)CC2")   # same cage, net +1
    assert Chem.GetFormalCharge(charged) == 1
    sites = ge._genuine_ion_sites(mol)                    # the neutral one's sites
    ge_sites = ge._genuine_ion_sites
    try:
        ge._genuine_ion_sites = lambda _m: sites
        assert ge._zwitterion_suffix_plan(charged, a2l) is None
    finally:
        ge._genuine_ion_sites = ge_sites


def test_plan_declines_on_internal_p59_charges():
    """P-59 nitro/azide/N-oxide charges are not ionic centres (get_ion_sites)."""
    for smi in ("C[N+](=O)[O-]", "[N-]=[N+]=NC", "C[N+]([O-])(C)C"):
        mol = Chem.MolFromSmiles(smi)
        a2l = {a.GetIdx(): a.GetIdx() + 1 for a in mol.GetAtoms()}
        assert ge._zwitterion_suffix_plan(mol, a2l) is None, smi


def test_plan_holds_out_the_anionic_oxygen():
    """The olate oxygen must be held out of substituent discovery so it is not
    ALSO emitted as a neutral ``oxo``/``hydroxy`` prefix (no double-count)."""
    mol = Chem.MolFromSmiles(MESOIONIC)
    # atom 3 = ring [N+], atom 9 = the ring C bearing the [O-] (atom 11).
    atom_to_locant = {i: i for i in range(len(mol.GetAtoms())) if i != 11}
    plan = ge._zwitterion_suffix_plan(mol, atom_to_locant)
    assert plan is not None
    held, text = plan
    assert 11 in held, "the anionic oxygen must be held out of discovery"
    assert text.endswith("olate")
    assert "ium" in text


# --------------------------------------------------------------------------
# The SKELETAL -ide / -uide branch (P-74.1.1: BOTH centres in the parent hydride)
#
# Code review finding: ``_ZWIT_SKELETAL_ANION_BASES`` shipped exercised by ZERO
# tests -- the "presence in a lookup table is not evidence the table is reached"
# pattern (the contributor guide #10, 8-for-8). It IS reached: entry point
# ``name_general_ring`` -> ``_emit_ring_from_analysis``. Each row below was
# constructed for this review, run through the engine, and the emitted name fed
# to ``opsin-cli-2.9.0 -r -osmi``; the parsed structure's RDKit InChIKey equals
# the input SMILES's InChIKey in all five cases (5/5, plus MESOIONIC = 6/6).
# --------------------------------------------------------------------------

SKELETAL_ZWITTERIONS = [
    # N-methylquinuclidinium 2-ylide: ring [N+] bridgehead + ring carbanion.
    ("C[N+]12[CH-]CC(CC1)CC2",
     "1-methyl-1-azabicyclo[2.2.2]octan-1-ium-2-ide",
     "RXUJURWSIKMHCP-UHFFFAOYSA-N"),
    # phosphonium analogue -- the cation need not be nitrogen.
    ("C[P+]12[CH-]CC(CC1)CC2",
     "1-methyl-1-phosphabicyclo[2.2.2]octan-1-ium-2-ide",
     "AYAHDEYJWAFOET-UHFFFAOYSA-N"),
    # decalin-shaped cage: a ring carbanion away from the bridgehead.
    ("C[N+]12CCCCC1[CH-]CCC2",
     "1-methyl-1-azabicyclo[4.4.0]decan-1-ium-5-ide",
     "VUOBCZVBBZNFTE-UHFFFAOYSA-N"),
    # P-72.3 -uide (hydride ADDITION): a skeletal boranuide.
    ("C[N+]12CC[BH-](CC1)CC2",
     "1-methyl-1-aza-4-borabicyclo[2.2.2]octan-1-ium-4-uide",
     "SKEWLPNITGSBFR-UHFFFAOYSA-N"),
    # -uide on a Group-14 metalloid: a skeletal silanuide.
    ("C[N+]12CC[SiH2-](CC1)CC2",
     "1-methyl-1-aza-4-silabicyclo[2.2.2]octan-1-ium-4-uide",
     "NOSIBMGQTKJOQG-UHFFFAOYSA-N"),
]


def _ring_producer(smiles):
    """Run the cage producer directly (Java-free), as ``test_general_engine``
    does. This is the site the review found untested; going through the namer
    alone cannot prove it, because a competing producer's neutral name can win
    whenever the OPSIN validity gate is off (which it is, suite-wide)."""
    from orthonym.assembly.general_engine import name_general_ring

    nm = Orthonym(_disable_opsin_validity_gate=True)
    mol = Chem.MolFromSmiles(smiles)
    feats = nm._perceive(mol, smiles, Chem.MolToSmiles(mol, canonical=True))
    nm._classify(feats)
    return name_general_ring(mol, feats, allow_aromatic_general=True,
                             allow_charged=True)


@pytest.mark.parametrize("smiles,expected,inchikey", SKELETAL_ZWITTERIONS)
def test_skeletal_zwitterion_producer_emits_cumulative_ium_ide(
        smiles, expected, inchikey):
    """P-74.1.1 :42417 — cumulative suffixes when BOTH centres are skeletal."""
    res = _ring_producer(smiles)
    assert res is not None, smiles
    assert res.name == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected,inchikey", SKELETAL_ZWITTERIONS)
def test_skeletal_zwitterion_emits_cumulative_ium_ide(smiles, expected, inchikey):
    """End-to-end, with the production OPSIN validity gate ENABLED.

    The gate matters here: for the two bicyclo[2.2.2] carbanion rows a
    competing producer offers the charge-DROPPING neutral
    ``1-methyl-1-azabicyclo[2.2.2]octane``, and it is the gate (that name is
    OPSIN-unparseable — N1 would need five bonds) that rejects it. Suite-wide
    the gate is off, so this assertion is only meaningful under the marker.
    """
    assert _name(smiles) == expected


@pytest.mark.parametrize("smiles,expected,inchikey", SKELETAL_ZWITTERIONS)
def test_skeletal_zwitterion_input_inchikey_is_the_recorded_one(
        smiles, expected, inchikey):
    """Pins the structure each recorded OPSIN round-trip was checked against.

    The OPSIN leg itself (name -> structure) is not re-run here: this suite is
    the Java-free unit layer. What this locks is that the SMILES has not drifted
    from the molecule whose round-trip was verified.
    """
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)) == inchikey


@pytest.mark.parametrize("smiles,expected,inchikey", SKELETAL_ZWITTERIONS)
def test_skeletal_zwitterion_takes_the_skeletal_branch(smiles, expected, inchikey):
    """The plan must hold NO atom out: a skeletal anion is a numbered parent
    atom, so nothing is withheld from substituent discovery (unlike -olate)."""
    mol = Chem.MolFromSmiles(smiles)
    a2l = {a.GetIdx(): a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    plan = ge._zwitterion_suffix_plan(mol, a2l)
    assert plan is not None, smiles
    held, text = plan
    assert held == frozenset(), "a skeletal anion needs no hold-out"
    assert text.endswith(("ide", "uide"))
    assert "ium" in text


def test_skeletal_branch_declines_a_ring_aminide():
    """``classify_anion`` returns 'aminide' for a ring N(-), deliberately absent
    from ``_ZWIT_SKELETAL_ANION_BASES`` -- the Blue Book's aminide zwitterion
    (:42460) needs an N-substituted aminide this producer cannot build, so it
    must fail closed rather than invent a suffix."""
    mol = Chem.MolFromSmiles("C[N+]12[N-]CC(CC1)CC2")
    from orthonym.rules.ions import classify_anion
    from orthonym.perception.ions import get_ion_sites
    assert classify_anion(mol, get_ion_sites(mol)["anions"][0]) == "aminide"
    a2l = {a.GetIdx(): a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    assert ge._zwitterion_suffix_plan(mol, a2l) is None


# --------------------------------------------------------------------------
# P-74.1.2's locant sentence (:42447), and the numbering it depends on
# --------------------------------------------------------------------------

def _legal_von_baeyer_numberings(smiles):
    """The heteroatom locant SETS of every numbering the bicyclo[4.4.0]
    descriptor permits. P-23.2.3 "Numbering bicyclic alicyclic hydrocarbons"
    (heading :9589), sentence :9591: *"The bicyclic ring system is numbered
    starting with one of the bridgeheads and proceeding first along the longer
    segment of the main ring to the second bridgehead, then back to the first
    bridgehead along the unnumbered segment."*  Both segments are 4 atoms here,
    so all four numberings satisfy it."""
    from orthonym.rules.vonbaeyer_universal import analyze_cage_universal

    mol = Chem.MolFromSmiles(smiles)
    cage = analyze_cage_universal(mol, allow_mancude=True)
    ring = sorted(cage.cage_atoms)
    bh = [i for i in ring
          if sum(1 for n in mol.GetAtomWithIdx(i).GetNeighbors()
                 if n.GetIdx() in cage.cage_atoms) == 3]

    def segments(a, b):
        out = []

        def walk(cur, prev, path):
            for n in mol.GetAtomWithIdx(cur).GetNeighbors():
                i = n.GetIdx()
                if i == prev or i not in cage.cage_atoms or i == a:
                    continue
                if i == b:
                    if path:
                        out.append(list(path))
                elif i not in path:
                    walk(i, cur, path + [i])
        walk(a, None, [])
        return out

    sets = set()
    for start, other in ((bh[0], bh[1]), (bh[1], bh[0])):
        segs = segments(start, other)
        for seg in segs:
            rest = [s for s in segs if s != seg]
            if not rest:
                continue
            order = [start] + seg + [other] + list(reversed(rest[0]))
            if len(order) != len(ring):
                continue
            a2l = {idx: n + 1 for n, idx in enumerate(order)}
            sets.add(tuple(sorted(a2l[i] for i in ring
                                  if mol.GetAtomWithIdx(i).GetSymbol() != "C")))
    return mol, cage, sets


def test_p74_1_2_ionic_locant_rule_has_no_freedom_here():
    """P-74.1.2 :42447: *"For assignment of lower locants, ionic centers on
    skeletal atoms of the parent hydride are preferred to the locants for
    positions of attachment of characteristic groups denoted by ionic
    suffixes."*  The shipped target does NOT violate it -- the rule never gets a
    choice.

    P-23.3.1 (:9765) *"Numbering is determined first by the fixed numbering of
    the hydrocarbon system"*, then P-23.3.2.1 (:9777) *"Low locants are assigned
    to the heteroatoms considered together as a set compared in increasing
    numerical order."*  Heteroatom locants outrank ionic-suffix locants, and the
    four legal numberings give four DISTINCT heteroatom sets, so P-23.3.2.1
    decides alone and leaves zero freedom for the ionic criterion. Satisfied
    vacuously.
    """
    _mol, _cage, sets = _legal_von_baeyer_numberings(MESOIONIC)
    assert len(sets) == 4, sets
    assert sorted(sets) == [(1, 2, 4), (1, 8, 10), (3, 5, 6), (6, 7, 9)]
    # distinct => P-23.3.2.1 is decisive on its own; no tie reaches P-74.1.2.
    assert len(set(sets)) == len(sets)


@pytest.mark.xfail(strict=True, reason=(
    "PRE-EXISTING von Baeyer defect, NOT introduced by the zwitterion fix "
    "(the neutral name it replaced was also '3,5,6-triaza...'). P-23.3.2.1 "
    "(:9777) requires the LOWEST heteroatom set; the cage engine picks "
    "(3,5,6) where (1,2,4) is legal and lower. Reported, not fixed: "
    "re-numbering the von Baeyer engine has repo-wide blast radius."))
def test_von_baeyer_heteroatom_locants_should_be_lowest_set():
    """P-23.3.2.1 (:9777) — *"Low locants are assigned to the heteroatoms
    considered together as a set compared in increasing numerical order. The
    preferred numbering is the lowest set at the first point of difference."*"""
    mol, cage, sets = _legal_von_baeyer_numberings(MESOIONIC)
    chosen = tuple(sorted(cage.atom_to_locant[i] for i in sorted(cage.cage_atoms)
                          if mol.GetAtomWithIdx(i).GetSymbol() != "C"))
    assert chosen == min(sets), f"engine chose {chosen}, lowest legal is {min(sets)}"


# --------------------------------------------------------------------------
# Requirement 4 — probe EVERY category the new guard gates.
# A guard that silences a correct name is a regression even if R7 improves.
# --------------------------------------------------------------------------

# (smiles, expected name) captured at d193c723 BEFORE the fix, on both tiers.
UNCHANGED = [
    ("C[N+](C)(C)C", "N,N,N-trimethylmethanaminium"),   # quaternary ammonium
    ("CC(=O)[O-]", "acetate"),                          # carboxylate
    ("C[N+](=O)[O-]", "nitromethane"),                  # P-59 nitro
    ("C[N+]([O-])(C)C", "N,N-dimethylmethanamine N-oxide"),   # P-59 N-oxide
    ("[N-]=[N+]=NC", "azidomethane"),                   # P-59 azide
    ("c1ccccc1[N+]#N", "benzenediazonium"),             # diazonium
    ("CC(=O)O", "acetic acid"),
    ("c1ccccc1", "benzene"),
    ("CCO", "ethanol"),
    ("[NH3+]CC(=O)[O-]", "glycine"),                    # amino-acid zwitterion
    ("C[N+](C)(C)CC(=O)[O-]", "(trimethylazaniumyl)acetate"),      # betaine
    ("[O-]C(=O)c1ccc[nH+]c1", "pyridin-1-ium-3-carboxylate"),      # P-74.1.2
]


@pytest.mark.parametrize("smiles,expected", UNCHANGED)
def test_charged_species_unchanged_pin_tier(smiles, expected):
    assert _name(smiles, "pin") == expected


@pytest.mark.parametrize("smiles,expected", UNCHANGED)
def test_charged_species_unchanged_best_effort_tier(smiles, expected):
    assert _name(smiles, "best-effort") == expected


# --------------------------------------------------------------------------
# Fail closed: a charged molecule must never receive a NEUTRAL name.
# --------------------------------------------------------------------------

def test_semipolar_phosphoryl_oxide_keeps_its_neutral_name():
    """the contributor guide #9 regression guard, caught by measurement during R7.

    ``[PH+]...[O-]`` is the charge-separated depiction of a NEUTRAL ``P=O``.
    RDKit/InChI agree: ``CC1CO[PH+](C1)[O-]`` and OPSIN's parse of
    ``4-methyl-2-oxo-1,2-oxaphospholane`` (``CC1CP(OC1)=O``) share InChIKey
    ``DWXZAVJWXATXAG-UHFFFAOYSA-N``. A first cut of the fail-closed guard
    treated the pair as a zwitterion and destroyed this correct,
    round-tripping name.
    """
    assert _name("CC1CO[PH+](C1)[O-]") == "4-methyl-2-oxo-1,2-oxaphospholane"


def test_semipolar_pairs_are_not_ionic_centres():
    """The guard's own predicate, directly."""
    for smi in ("CC1CO[PH+](C1)[O-]", "C[N+](=O)[O-]", "C[N+]([O-])(C)C",
                "[O-][n+]1ccccc1", "CS(=O)C"):
        mol = Chem.MolFromSmiles(smi)
        assert ge._has_ionic_centres(mol) is False, smi


def test_genuine_zwitterion_is_an_ionic_centre():
    """The anion is on a ring CARBON, not on the cation -> not semipolar."""
    assert ge._has_ionic_centres(Chem.MolFromSmiles(MESOIONIC)) is True
    assert ge._has_ionic_centres(Chem.MolFromSmiles("[NH3+]CC(=O)[O-]")) is True


def test_unexpressible_zwitterion_is_refused_not_neutralised():
    """the contributor guide #9 — verify what is EMITTED, not just that the bad path stopped.

    An ionic centre the parent cannot carry must abstain, never ship a neutral
    name that describes a different (uncharged) species.
    """
    # A sulfonium ylide: cation S+, anion carbanion, both off any ring parent.
    name = _name("C[S+](C)[CH2-]", "best-effort")
    assert "oxo" not in name
    # Whatever is emitted must not be the plain neutral hydride name.
    assert name not in ("dimethyl(methyl)sulfane", "trimethylsulfane")
