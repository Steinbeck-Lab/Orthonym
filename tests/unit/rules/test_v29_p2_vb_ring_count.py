"""v29 Phase 2 T3b: the von Baeyer ring COUNT, and what the cage caps bound.

Two things are locked here.

1. ``von_baeyer_ring_count`` computes the number P-23.1.9 defines and
   ``cyclo_ring_count_word`` spells. It is NOT RDKit's ring-set cardinality:
   RDKit returns the *symmetrized* SSSR, which keeps extra symmetry-equivalent
   smallest rings and therefore over-counts precisely the symmetric cages von
   Baeyer nomenclature exists for (adamantane 4 vs 3, cubane 6 vs 5). Before
   T3b ``MAX_CAGE_RINGS`` was compared against that over-count, so a cap meant
   to bound "8 rings" refused *heptacyclo* (7-ring) cages.

2. P-23.2.4 main-bridge selection (v41 M4 subpart #1, FIXED). ``_find_main_ring``
   used to offer only a 0-atom (direct bond) or 1-atom (common neighbour) main
   bridge, so a bridgehead pair joined by a 2+-atom bridge was never a candidate
   and the ``main_bridge_len`` slot of its score could hold only 0 or 1 -- the
   main RING size came out right every time and the main BRIDGE did not. Case 3
   in ``_find_main_ring`` (a 2+-atom main bridge) plus the largest-bridge
   selection in ``_find_main_bridge`` now produce the preferred main bicycle;
   ``BB_MAIN_BRIDGE_PIN_CASES`` reach the exact Blue Book PIN, while
   ``BB_MAIN_BRIDGE_SUBPART2_CASES`` still need consistent secondary/dependent
   bridge numbering (subpart #2) and DEGRADE cleanly until then.

Ground truth is the Blue Book itself, cited by ``BlueBookV2/BlueBookV2.md`` line.
The structures were obtained by parsing each cited Blue Book name to a structure
(name -> structure is a settled direction, and no claim of PIN *preference* is
made from it); each one is independently re-checked here against the descriptor's
own arithmetic -- P-23.2.6.1.4, ":9651": total ring atoms == sum of the bracket
numbers + 2 -- and against its ring-count prefix, so a mis-parse cannot pass.
"""
import pytest
from rdkit import Chem

from orthonym.rules.polycyclic import (
    VonBaeyerAnalyzer,
    _get_largest_connected_ring_component,
    cyclo_ring_count_word,
    von_baeyer_ring_count,
)
from orthonym.rules.vonbaeyer_universal import (
    MAX_CAGE_RINGS,
    analyze_cage_universal,
)


def _cage_of(smiles):
    """(mol, cage_atoms) the way the cage analyzers pick the cage."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"test datum is not valid SMILES: {smiles!r}"
    ring_atoms = set()
    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms.update(ring)
    return mol, _get_largest_connected_ring_component(mol, ring_atoms)


# --------------------------------------------------------------------------
# 1. The count itself, against Blue Book ring-count prefixes.
#    (bb_line, label, smiles, bb_ring_count, bb_cyclo_word)
# --------------------------------------------------------------------------
BB_RING_COUNTS = [
    (9840, "adamantane = tricyclo[3.3.1.1^3,7]decane (PIN)",
     "C1C2CC3CC1CC(C2)C3", 3, "tricyclo"),
    (9889, "cubane = pentacyclo[4.2.0.0^2,5.0^3,8.0^4,7]octane (PIN)",
     "C1C2C3C4C1C1C4C3C21", 5, "pentacyclo"),
    (9897, "prismane = tetracyclo[2.2.0.0^2,6.0^3,5]hexane (PIN)",
     "C1C2C3C1C1C3C21", 4, "tetracyclo"),
    (9583, "norbornane = bicyclo[2.2.1]heptane", "C1CC2CCC1C2", 2, "bicyclo"),
    (9571, "bicyclo[2.2.2]octane", "C1CC2CCC1CC2", 2, "bicyclo"),
    (9635, "tricyclo[4.2.2.2^2,5]dodecane (PIN)",
     "C1CC2CCC1C1CCC(CC1)CC2", 3, "tricyclo"),
]


@pytest.mark.unit
@pytest.mark.parametrize("bb_line,label,smiles,count,word", BB_RING_COUNTS)
def test_ring_count_is_the_blue_book_number(bb_line, label, smiles, count, word):
    """P-23.1.9 (``:9558``) / P-23.2.6.1.1 (``:9645``): the ring number is the
    minimum number of scissions that makes the skeleton acyclic. It must equal
    the ``...cyclo`` prefix of the cited Blue Book name."""
    mol, cage = _cage_of(smiles)
    got = von_baeyer_ring_count(mol, cage)
    assert got == count, (
        f"BlueBookV2.md:{bb_line} {label}: P-23.1.9 ring count should be "
        f"{count}, got {got}")
    assert cyclo_ring_count_word(got) == word


@pytest.mark.unit
@pytest.mark.parametrize("bb_line,label,smiles,count,word", BB_RING_COUNTS)
def test_symmetrized_ring_set_is_a_different_quantity(
        bb_line, label, smiles, count, word):
    """The reason the primitive exists. RDKit's ring-set cardinality is >= the
    P-23.1.9 count and is strictly greater on the symmetric cages, so it must
    never be used where the Blue Book number is meant."""
    mol, cage = _cage_of(smiles)
    symmetrized = sum(1 for r in mol.GetRingInfo().AtomRings()
                      if set(r) <= cage)
    assert symmetrized >= count, (
        "the symmetrized ring set can never be SMALLER than the ring number")
    assert von_baeyer_ring_count(mol, cage) == count


@pytest.mark.unit
def test_symmetrized_count_really_overcounts_the_canonical_cages():
    """Not hypothetical: the two most-cited von Baeyer PINs in the book are
    over-counted, which is what made the ring cap refuse legal cages."""
    for smiles, bb_count, symm_count in (
            ("C1C2CC3CC1CC(C2)C3", 3, 4),        # adamantane, ":9840"
            ("C1C2C3C4C1C1C4C3C21", 5, 6),       # cubane, ":9889"
    ):
        mol, cage = _cage_of(smiles)
        assert von_baeyer_ring_count(mol, cage) == bb_count
        assert mol.GetRingInfo().NumRings() == symm_count


@pytest.mark.unit
def test_ring_count_fails_closed_on_an_empty_cage():
    mol = Chem.MolFromSmiles("CCO")
    assert von_baeyer_ring_count(mol, set()) is None


@pytest.mark.unit
def test_ring_count_counts_components_separately():
    """``E - V + C``: two independent monocycles are 2 scissions, not 1 -- so a
    disconnected atom set can never masquerade as a single 1-ring system."""
    mol = Chem.MolFromSmiles("C1CC1.C1CCC1")
    ring_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()}
    assert von_baeyer_ring_count(mol, ring_atoms) == 2


# --------------------------------------------------------------------------
# 2. The cap now bounds that count (the T3b behaviour change).
# --------------------------------------------------------------------------
#: 12-atom bridged cage; reference name ``10,11-dimethylheptacyclo[...]dodecane``
#: -> 7 rings, well inside ``MAX_CAGE_RINGS = 8``. RDKit's symmetrized ring set
#: has ELEVEN rings for it, so the pre-T3b cap refused it as "> 8 rings".
HEPTACYCLO_CAGE = "C1CC2C(CC1C)C3C45C36C47C56C72"


@pytest.mark.unit
def test_cap_admits_a_heptacyclo_cage_the_symmetrized_count_refused():
    mol, cage = _cage_of(HEPTACYCLO_CAGE)
    assert von_baeyer_ring_count(mol, cage) == 7
    assert sum(1 for r in mol.GetRingInfo().AtomRings() if set(r) <= cage) == 11
    assert 7 <= MAX_CAGE_RINGS < 11, (
        "this test is only meaningful while the cap sits between the two counts")
    cage_analysis = analyze_cage_universal(mol)
    assert cage_analysis is not None, (
        "a 7-ring cage must not be refused by an 8-ring cap")
    assert cage_analysis.descriptor.startswith("heptacyclo["), (
        f"expected a heptacyclo descriptor, got {cage_analysis.descriptor!r}")


@pytest.mark.unit
def test_cap_still_fails_closed_above_the_limit(monkeypatch):
    """The backstop above the new limit. Squeezing the cap below the cage's own
    P-23.1.9 count must still refuse -- the cap is live, not decorative."""
    import orthonym.rules.vonbaeyer_universal as vbu
    mol, cage = _cage_of(HEPTACYCLO_CAGE)
    assert von_baeyer_ring_count(mol, cage) == 7
    monkeypatch.setattr(vbu, "MAX_CAGE_RINGS", 6)
    assert analyze_cage_universal(mol) is None
    monkeypatch.setattr(vbu, "MAX_CAGE_RINGS", 7)
    assert analyze_cage_universal(mol) is not None


@pytest.mark.unit
def test_ovalene_stays_refused():
    """Ovalene is 34 cage atoms / 10 rings. It is above the RING cap (10 > 8)
    but well INSIDE the atom cap, and it also fails the descriptor edge-audit,
    so it refuses at any cap value -- see
    ``tests/unit/rules/test_v26_p2_aromatic_vonbaeyer.py`` for the re-derived
    reason. Locked here because T3b only ever LOWERS a ring count, so a
    once-refused system must not become nameable by accident."""
    ovalene = "c1cc2ccc3ccc4ccc5ccc6ccc7ccc8ccc1c1c2c3c4c2c5c6c7c8c12"
    mol, cage = _cage_of(ovalene)
    assert len(cage) == 34
    assert von_baeyer_ring_count(mol, cage) == 10
    assert analyze_cage_universal(mol, allow_mancude=True) is None


# --------------------------------------------------------------------------
# 3. P-23.2.4 main-bridge selection (v41 M4 subpart #1).
#    (bb_line, bb_descriptor, smiles, bb_atoms, bb_ring_count,
#     bb_main_ring_atoms, bb_main_bridge_len, bb_balance)
#
# ``_find_main_ring`` used to offer only a 0-atom (direct bond) or 1-atom
# (common neighbour) main bridge, so a bridgehead pair joined by a 2+-atom bridge
# was never a candidate and the ``main_bridge_len`` slot of its score could hold
# only 0 or 1. v41 M4 subpart #1 adds Case 3 (a 2+-atom main bridge) and makes
# ``_find_main_bridge`` select the LARGEST bridge (P-23.2.4), so the preferred
# main bicycle (P-23.2.4 largest main bridge, P-23.2.6.2.1 symmetric division) is
# now produced.
#
# PIN_CASES: subpart #1 alone reaches the exact Blue Book PIN end to end.
# SUBPART2_CASES: subpart #1 makes the main BICYCLE preferred, but the FULL
# descriptor additionally needs consistent numbering of the secondary / dependent
# bridges (subpart #2, the dependent-bridge discovery-fixpoint). Their PIN
# positions 25/28 exceed the main-bicycle atom count -> a dependent secondary
# bridge (9739, 9731), or a secondary-bridge numbering the current machinery
# cannot make self-consistent (9749, 9753). The engine DEGRADES cleanly there:
# the preferred main bicycle whose descriptor fails the legality audit is
# replaced by the legacy 0/1-atom-main-bridge result (a valid, RT-correct but
# non-preferred name) or an abstain -- never a wrong name (0-wrong holds).
BB_MAIN_BRIDGE_PIN_CASES = [
    (9631, "tricyclo[9.3.3.1^1,11]",
     "C123CCCCCCCCCC(CCC1)(CCC2)C3", 18, 3, 14, 3, 3),
    (9721, "tetracyclo[4.4.2.2^2,5.2^7,10]",
     "C12C3CCC(C(C4CCC1CC4)CC2)CC3", 16, 4, 10, 2, 4),
    (9761, "tetracyclo[7.4.3.2^3,7.1^3,7]",
     "C12CC34CCCC(CC(CCCC1)CCC2)(CC3)C4", 19, 4, 13, 3, 4),
]

# v41 M4#2 splits the subpart-2 cases by the defect that blocked each (see
# .planning/audit-v41/M4-2-CODEMAP.md): Finding B = main-bridge numbering direction
# (fixed in Task 1), Finding A = branched-component/dependent-bridge discovery (Task 2).
# Fix B (main-bridge orientation) alone reaches the PIN for all three of these --
# including 9731, whose descriptor DOES carry a dependent bridge (0^11,25): the
# existing discovery + Step-6 resolver number that dependent bridge correctly once
# the main bridge is oriented right, so 9731 needs no discovery change.
BB_SUBPART2_FINDING_B_CASES = [
    (9749, "tetracyclo[6.3.3.2^3,6.1^2,6]",
     "C12C3C4CCC(CC(CCC1)CCC2)(CC4)C3", 17, 4, 11, 3, 3),
    (9753, "tetracyclo[6.3.3.2^2,6.1^3,6]",
     "C12C3C4CCC(CC(CCC1)CCC2)(C4)CC3", 17, 4, 11, 3, 3),
    (9731, "hexacyclo[15.3.2.2^3,7.1^2,12.0^13,21.0^11,25]",
     "C12C3C4CCCC5CCCC(C(C6CCCC(CCC1)CC26)C3)C4C5", 25, 6, 20, 2, 3),
]
# Only 9739 needs Fix A: a branched (>=3-endpoint) component whose collapse to one
# 2-endpoint bridge drops an atom (26), so the dependent bridge 1^13,28 is never built.
BB_SUBPART2_FINDING_A_CASES = [
    (9739, "pentacyclo[13.7.4.3^3,8.0^18,20.1^13,28]",
     "C12CC3CCCCC4CCCCC(CC(CCC5CC5CC1)CCCC2)CC(C4)C3", 30, 5, 22, 4, 7),
]
BB_MAIN_BRIDGE_SUBPART2_CASES = (
    BB_SUBPART2_FINDING_B_CASES + BB_SUBPART2_FINDING_A_CASES)

BB_MAIN_BRIDGE_CASES = BB_MAIN_BRIDGE_PIN_CASES + BB_MAIN_BRIDGE_SUBPART2_CASES


def _our_descriptor(smiles):
    mol, cage = _cage_of(smiles)
    desc = VonBaeyerAnalyzer().analyze(mol, cage)
    return desc.descriptor_string if desc else None


def _split(descriptor):
    """(main_ring_atoms, main_bridge_len, balance) from a von Baeyer descriptor.

    The first three bracket numbers are the main bicycle: two main-ring branches
    then the main bridge (P-23.2.6.1.2, ``:9647``). Main ring = branch1 +
    branch2 + the 2 main bridgeheads; balance = the smaller branch.
    """
    body = descriptor[descriptor.index("[") + 1:descriptor.rindex("]")]
    nums = []
    for part in body.split("."):
        head = part.split("^")[0]
        nums.append(int(head))
    b1, b2, mb = nums[0], nums[1], nums[2]
    return b1 + b2 + 2, mb, min(b1, b2)


@pytest.mark.unit
@pytest.mark.parametrize(
    "bb_line,bb_desc,smiles,atoms,rings,mr,mb,bal", BB_MAIN_BRIDGE_CASES)
def test_blue_book_datum_is_self_consistent(
        bb_line, bb_desc, smiles, atoms, rings, mr, mb, bal):
    """Validates the ground truth independently of how the structure was
    obtained: P-23.2.6.1.4 (``:9651``) atom arithmetic, the P-23.1.9 ring count
    against the ``...cyclo`` prefix, and our reading of the bracket."""
    mol, cage = _cage_of(smiles)
    assert len(cage) == atoms, f"BlueBookV2.md:{bb_line}: cage atom count"
    assert von_baeyer_ring_count(mol, cage) == rings
    assert cyclo_ring_count_word(rings) == bb_desc.split("[")[0]
    got_mr, got_mb, got_bal = _split(bb_desc)
    assert (got_mr, got_mb, got_bal) == (mr, mb, bal)
    # P-23.2.6.1.4: sum of the bracket numbers + 2 == total ring atoms
    body = bb_desc[bb_desc.index("[") + 1:bb_desc.rindex("]")]
    total = sum(int(p.split("^")[0]) for p in body.split("."))
    assert total + 2 == atoms


@pytest.mark.unit
@pytest.mark.parametrize(
    "bb_line,bb_desc,smiles,atoms,rings,mr,mb,bal", BB_MAIN_BRIDGE_PIN_CASES)
def test_subpart1_reaches_blue_book_pin(
        bb_line, bb_desc, smiles, atoms, rings, mr, mb, bal):
    """v41 M4 subpart #1 (P-23.2.4 / P-23.2.6.2.1): the preferred main bicycle
    is now produced AND the full descriptor equals the Blue Book PIN.

    P-23.2.1 (main ring = as many skeletal atoms as possible) was always
    satisfied -- our main ring has exactly the Blue Book's atom count. What used
    to fail was the next criterion, P-23.2.4 (``:9603``, the main bridge
    "includes as many of the atoms as possible that are not included in the main
    ring") and P-23.2.6.2.1 (``:9661``, "the main ring must be divided as
    symmetrically as possible by the main bridge"). ``_find_main_ring`` Case 3
    (a 2+-atom main bridge) plus ``_find_main_bridge``'s largest-bridge selection
    fix both: the main-ring size, main-bridge length AND balance now match the
    Blue Book, and so does the whole descriptor string.
    """
    ours = _our_descriptor(smiles)
    assert ours is not None, f"BlueBookV2.md:{bb_line}: analyzer returned None"
    # Main bicycle: main-ring size, main-bridge length, and balance all preferred.
    assert _split(ours) == (mr, mb, bal), (
        f"BlueBookV2.md:{bb_line}: main bicycle should be preferred "
        f"(P-23.2.4/P-23.2.6.2.1); BB (mr={mr}, mb={mb}, bal={bal}), "
        f"ours {ours!r} -> {_split(ours)}")
    # And the whole descriptor is the Blue Book PIN.
    assert ours == bb_desc, (
        f"BlueBookV2.md:{bb_line}: descriptor should equal the Blue Book PIN "
        f"{bb_desc!r}, got {ours!r}")


@pytest.mark.unit
@pytest.mark.parametrize(
    "bb_line,bb_desc,smiles,atoms,rings,mr,mb,bal", BB_SUBPART2_FINDING_B_CASES)
def test_subpart2_finding_b_main_bridge_direction(
        bb_line, bb_desc, smiles, atoms, rings, mr, mb, bal):
    """v41 M4#2 Fix B (P-23.2.6.3 main-bridge numbering direction): the engine
    already builds the byte-correct BB descriptor string, but Step 2 of
    ``_order_and_number_secondary_bridges`` numbered the main bridge in stored order
    with no direction check (unlike Step 5), so after ``_select_pin_orientation``
    swaps the bridgehead pair the numbering ran backwards and the reconstruction
    audit correctly rejected it. ``_orient_main_bridge`` fixes the direction to match
    the audit, so the PIN is emitted legally."""
    mol, cage = _cage_of(smiles)
    desc = VonBaeyerAnalyzer().analyze(mol, cage)
    assert desc is not None and desc.legality is True, (
        f"BlueBookV2.md:{bb_line}: expected legal PIN, got "
        f"{getattr(desc, 'descriptor_string', None)!r} "
        f"legality={getattr(desc, 'legality', None)}")
    assert desc.descriptor_string == bb_desc, (
        f"BlueBookV2.md:{bb_line}: {bb_desc!r} vs {desc.descriptor_string!r}")


@pytest.mark.unit
@pytest.mark.xfail(strict=True, reason=(
    "v41 M4#2 Fix A pending: these two need the branched-component / dependent-bridge "
    "discovery fix (Finding A). Their PIN attachment positions exceed the main-bicycle "
    "atom count -> a genuine dependent bridge. Until Fix A the engine DEGRADES (legacy "
    "non-preferred name, or abstain) rather than emit these PINs -- never a wrong name "
    "(0-wrong). Fix B (main-bridge direction) is already shipped."))
@pytest.mark.parametrize(
    "bb_line,bb_desc,smiles,atoms,rings,mr,mb,bal", BB_SUBPART2_FINDING_A_CASES)
def test_descriptor_equals_blue_book_pending_subpart2(
        bb_line, bb_desc, smiles, atoms, rings, mr, mb, bal):
    assert _our_descriptor(smiles) == bb_desc
