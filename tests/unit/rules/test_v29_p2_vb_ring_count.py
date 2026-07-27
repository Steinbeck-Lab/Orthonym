"""v29 Phase 2 T3b: the von Baeyer ring COUNT, and what the cage caps bound.

Two things are locked here.

1. ``von_baeyer_ring_count`` computes the number P-23.1.9 defines and
   ``cyclo_ring_count_word`` spells. It is NOT RDKit's ring-set cardinality:
   RDKit returns the *symmetrized* SSSR, which keeps extra symmetry-equivalent
   smallest rings and therefore over-counts precisely the symmetric cages von
   Baeyer nomenclature exists for (adamantane 4 vs 3, cubane 6 vs 5). Before
   T3b ``MAX_CAGE_RINGS`` was compared against that over-count, so a cap meant
   to bound "8 rings" refused *heptacyclo* (7-ring) cages.

2. The P-23.2.4 main-bridge defect that is the real reason not to raise
   ``MAX_CAGE_RINGS``. ``_find_main_ring`` only ever offers a 0-atom (direct
   bond) or 1-atom (common neighbour) main bridge, so a bridgehead pair joined
   by a 2+-atom bridge is never even a candidate. Its score tuple carries a
   ``main_bridge_len`` comment citing P-23.2.4, but the value it scores can only
   ever be 0 or 1. Blue Book ground truth below shows the consequence: the main
   RING size comes out right every time and the main BRIDGE does not.

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
# 3. The P-23.2.4 main-bridge defect (why the cap VALUE must not rise yet).
#    (bb_line, bb_descriptor, smiles, bb_atoms, bb_ring_count,
#     bb_main_ring_atoms, bb_main_bridge_len, bb_balance)
# --------------------------------------------------------------------------
BB_MAIN_BRIDGE_CASES = [
    (9631, "tricyclo[9.3.3.1^1,11]",
     "C123CCCCCCCCCC(CCC1)(CCC2)C3", 18, 3, 14, 3, 3),
    (9721, "tetracyclo[4.4.2.2^2,5.2^7,10]",
     "C12C3CCC(C(C4CCC1CC4)CC2)CC3", 16, 4, 10, 2, 4),
    (9749, "tetracyclo[6.3.3.2^3,6.1^2,6]",
     "C12C3C4CCC(CC(CCC1)CCC2)(CC4)C3", 17, 4, 11, 3, 3),
    (9753, "tetracyclo[6.3.3.2^2,6.1^3,6]",
     "C12C3C4CCC(CC(CCC1)CCC2)(C4)CC3", 17, 4, 11, 3, 3),
    (9761, "tetracyclo[7.4.3.2^3,7.1^3,7]",
     "C12CC34CCCC(CC(CCCC1)CCC2)(CC3)C4", 19, 4, 13, 3, 4),
    (9739, "pentacyclo[13.7.4.3^3,8.0^18,20.1^13,28]",
     "C12CC3CCCCC4CCCCC(CC(CCC5CC5CC1)CCCC2)CC(C4)C3", 30, 5, 22, 4, 7),
    (9731, "hexacyclo[15.3.2.2^3,7.1^2,12.0^13,21.0^11,25]",
     "C12C3C4CCCC5CCCC(C(C6CCCC(CCC1)CC26)C3)C4C5", 25, 6, 20, 2, 3),
]


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
    "bb_line,bb_desc,smiles,atoms,rings,mr,mb,bal", BB_MAIN_BRIDGE_CASES)
def test_main_ring_right_main_bridge_wrong(
        bb_line, bb_desc, smiles, atoms, rings, mr, mb, bal):
    """The defect, characterised positively so it cannot rot into a vague xfail.

    P-23.2.1 (main ring = as many skeletal atoms as possible) is satisfied on
    every one of these: our main ring has exactly the Blue Book's atom count.
    What fails is the next criterion -- P-23.2.4 (``:9603``, the main bridge
    "includes as many of the atoms as possible that are not included in the main
    ring") and P-23.2.6.2.1 (``:9661``, "the main ring must be divided as
    symmetrically as possible by the main bridge"): our main bridge is shorter
    than the Blue Book's, or (case ``:9721``) equal but dividing the main ring
    less symmetrically.

    Root cause: ``_find_main_ring`` enumerates main-bridgehead pairs only via a
    direct bond (0-atom bridge) or a common neighbour (1-atom bridge), so a pair
    joined by a 2+-atom bridge is never a candidate and the ``main_bridge_len``
    slot of its score tuple can only hold 0 or 1.
    """
    ours = _our_descriptor(smiles)
    assert ours is not None, f"BlueBookV2.md:{bb_line}: analyzer returned None"
    our_mr, our_mb, our_bal = _split(ours)
    assert our_mr == mr, (
        f"BlueBookV2.md:{bb_line}: main-ring size should still be right "
        f"(P-23.2.1); BB {mr}, ours {our_mr} from {ours!r}")
    assert our_mb < mb or (our_mb == mb and our_bal < bal), (
        f"BlueBookV2.md:{bb_line}: expected the KNOWN P-23.2.4 shortfall "
        f"(BB main bridge {mb}, balance {bal}); ours {ours!r} gives main "
        f"bridge {our_mb}, balance {our_bal}. If this now matches the Blue "
        f"Book, the defect is fixed -- delete this test and unmark the strict "
        f"xfail below.")


@pytest.mark.unit
@pytest.mark.xfail(strict=True, reason=(
    "P-23.2.4 / P-23.2.6.2.1 not implemented: _find_main_ring can only offer a "
    "0- or 1-atom main bridge, so these Blue Book PIN descriptors come back "
    "with a non-preferred main bicycle. Structurally valid, NOT preferred. "
    "Fixing this is the precondition for raising MAX_CAGE_RINGS."))
@pytest.mark.parametrize(
    "bb_line,bb_desc,smiles,atoms,rings,mr,mb,bal", BB_MAIN_BRIDGE_CASES)
def test_descriptor_equals_blue_book(
        bb_line, bb_desc, smiles, atoms, rings, mr, mb, bal):
    assert _our_descriptor(smiles) == bb_desc
