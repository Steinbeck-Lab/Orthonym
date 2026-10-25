"""The numbering of a von Baeyer cage that the prefix locants and (j) leave tied: branch stereo, branches
too large to name, and the order of the atoms (large-polycycle plan, Task 3a, fix a performance pass).

 NUMBERING (the Blue Book), (j) (:3346): "the lower locant is assigned to CIP stereodescriptors
Z, R, M, and r (pseudoasymmetry) that are preferred to E, S, P, and s". The Blue Book's own examples apply
it to the stereodescriptors of substituent groups: ``1-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]benzene (PIN)``
(:3399) and ``1-[(1r,4r)-4-methylcyclohexyl]-2-[(1s,4s)-4-methylcyclohexyl]ethane-1,1,2,2-tetracarbonitrile
(PIN)`` (:3409, "the substituent denoted by the 'r' stereodescriptor receives the lowest locant").

The criteria stopped at the stereodescriptors of the cage itself, so a cage with two branches that differ
only by stereo took the numbering the input atom order produced: ``5-[(2R)-butan-2-yl]-3-[(2S)-butan-2-yl]...``
or ``3-[(2R)-butan-2-yl]-5-[(2S)-butan-2-yl]...`` for one molecule, both ``pin_verified``.
"""
import pytest
from rdkit import Chem

from tests.support.cage_numbering_oracle import check
from tests.support.default_tier import assert_default_tier_declines, strict_path_row
from tests.support.large_polycycles import be, spellings
from tests.support.rt_assert import assert_full_rt

pytestmark = pytest.mark.opsin_gate

SPELLINGS = 12

# the tricyclo[2.2.1.0^2,6]heptane (nortricyclene) skeleton, 3 and 5 are its two equivalent positions
RS = "C[C@H](CC)C1C2C3CC1C([C@@H](C)CC)C32"          # (2R)- and (2S)-butan-2-yl
RR = "C[C@H](CC)C1C2C3CC1C([C@H](C)CC)C32"
EZ = "C/C=C/C1C2C3CC1C(/C=C\\C)C32"                  # (1E)- and (1Z)-prop-1-en-1-yl
BIG = "C" * 31 + "C1C2C3CC1C(" + "C" * 32 + ")C32"    # branches of 31 and 32 atoms: not named for (g)


def _canonical_ring(mol):
    return {i for ring in mol.GetRingInfo().AtomRings() for i in ring}


@pytest.mark.unit
@pytest.mark.parametrize("smiles, expected", [
    (RS, "3-[(2R)-butan-2-yl]-5-[(2S)-butan-2-yl]tricyclo[2.2.1.0^2,6]heptane"),
    (RR, "3,5-di[(2R)-butan-2-yl]tricyclo[2.2.1.0^2,6]heptane"),
    (EZ, "3-[(1Z)-prop-1-en-1-yl]-5-[(1E)-prop-1-en-1-yl]tricyclo[2.2.1.0^2,6]heptane"),
])
def test_the_descriptors_of_a_branch_take_the_preferred_one_to_the_lower_locant(smiles, expected):
    """The R prefix takes 3 and the S one 5 (:3399), the Z propenyl 3 and the E one 5 (:3346): at both
    tiers, for every spelling, and the name reads back to the input (OPSIN)."""
    for spelling in [smiles] + spellings(smiles, SPELLINGS):
        row = strict_path_row(spelling)
        assert row["name"] == expected and row["tier"] == "pin_verified", (spelling, row)
        assert be(spelling)["name"] == expected, spelling
    assert_full_rt(expected, smiles)


@pytest.mark.unit
def test_the_signature_of_a_branch_holds_its_stereo():
    """Two branches that differ only by stereo are two branches (they were one: the signature was a
    non-isomeric SMILES), and the signature is the same for every spelling of the molecule."""
    from orthonym.rules.cage_numbering import aromatic_view, cage_branches

    def signatures(smiles):
        mol = aromatic_view(Chem.MolFromSmiles(smiles))
        branches = cage_branches(mol, _canonical_ring(mol), set())
        return sorted(sig for entries in branches.values() for sig, _atoms, _first in entries)

    rs, rr, ez = signatures(RS), signatures(RR), signatures(EZ)
    assert rs[0] != rs[1] and rr[0] == rr[1] and ez[0] != ez[1]
    assert rs[0][:2] == rs[1][:2] and rs[0][2] and rs[1][2]       # the same branch, its stereo apart
    plain = signatures("CC(C)C1C2C3CC1C(C(C)C)C32")
    assert plain[0][2] == "" and plain[0] == plain[1]             # no stereo, no stereo part
    for smiles in (RS, RR, EZ):
        assert {tuple(signatures(s)) for s in spellings(smiles, 12)} == {tuple(signatures(smiles))}


@pytest.mark.unit
def test_the_codes_of_a_prefix_are_read_in_the_order_the_name_cites_them():
    from orthonym.assembly.naming_utils import cip_descriptor_codes, cip_descriptor_rank_key
    assert cip_descriptor_codes("(2R,3S)-3-hydroxybutan-2-yl") == ("R", "S")
    assert cip_descriptor_codes("[(1Z)-prop-1-en-1-yl]") == ("Z",)
    assert cip_descriptor_codes("methyl") == ()
    assert cip_descriptor_rank_key("(2R,3S)-3-hydroxybutan-2-yl") == (0, 1)


@pytest.mark.unit
def test_prefixes_that_differ_only_by_stereo_are_cited_by_their_descriptors_whatever_the_atom_order():
    """ and (j): ``(1Z)-prop-1-en-1-yl`` before ``(1E)-prop-1-en-1-yl``. The prefixes were
    sorted on a key that ignores the descriptors, so the order was the order the cage's atoms were met in."""
    from orthonym.rules.polycyclic import _assemble_substituent_prefix
    z = {"locant": 3, "name": "(1Z)-prop-1-en-1-yl", "atom_indices": []}
    e = {"locant": 5, "name": "(1E)-prop-1-en-1-yl", "atom_indices": []}
    forward, backward = _assemble_substituent_prefix([z, e]), _assemble_substituent_prefix([e, z])
    assert forward == backward
    assert forward.index("Z") < forward.index("E"), forward


@pytest.mark.unit
def test_a_branch_too_large_to_name_leaves_the_name_without_the_pin_label():
    """(g) needs the names of the branches (the one cited first takes the lower locant). A branch of more than
    ``MAX_NAMED_BRANCH_ATOMS`` atoms is not named, so the choice between 3,5 and 5,3 is not the Blue Book's: the
    default tier declines (NO_VERIFIED_PIN), the best-effort tier names the cage with the label
    ``systematic_verified`` and one name for every spelling, and the name reads back."""
    assert_default_tier_declines(BIG)
    names = set()
    for spelling in [BIG] + spellings(BIG, SPELLINGS):
        row = be(spelling)
        assert row["tier"] == "systematic_verified", (spelling, row)
        names.add(row["name"])
    assert len(names) == 1, sorted(names)
    name = names.pop()
    assert name.endswith("tricyclo[2.2.1.0^2,6]heptane") and "dotriacontyl" in name, name
    assert_full_rt(name, BIG)


@pytest.mark.unit
def test_branches_of_30_atoms_or_fewer_are_named_and_decide_the_numbering():
    """The cap is 30 atoms: branches of 29 and 30 atoms are named and (g) cites the one whose name is first
    in alphanumerical order, ``nonacosyl`` (n) before ``triacontyl`` (t), so it takes the lower
    locant, and the name keeps the PIN label."""
    smiles = "C" * 29 + "C1C2C3CC1C(" + "C" * 30 + ")C32"
    expected = "3-nonacosyl-5-triacontyltricyclo[2.2.1.0^2,6]heptane"
    for spelling in [smiles] + spellings(smiles, 6):
        row = strict_path_row(spelling)
        assert row["name"] == expected and row["tier"] == "pin_verified", (spelling, row)


@pytest.mark.unit
def test_a_numbering_chosen_without_ranking_every_candidate_is_not_certified(monkeypatch):
    """More tied decompositions than ``_MAX_LOCANT_CANDIDATES`` are not ranked: the incumbent numbering
    stands, and it is marked, so that the name that carries its descriptor is not a PIN."""
    from orthonym.rules.polycyclic import VonBaeyerAnalyzer
    mol = Chem.MolFromSmiles("ClC1C2C(Br)C3C1C3C2")
    ring = _canonical_ring(mol)
    assert VonBaeyerAnalyzer().analyze(mol, ring).numbering_certified is True
    monkeypatch.setattr(VonBaeyerAnalyzer, "_MAX_LOCANT_CANDIDATES", 1)
    assert VonBaeyerAnalyzer().analyze(mol, ring).numbering_certified is False


@pytest.mark.unit
@pytest.mark.parametrize("smiles", [
    "C12(Cl)C3C4C5C6C7(Br)C8C9C%10C7C7C(Cl)(C1C4C67)C%10C2C9C3C85",     # a dodecahedrane with three prefixes
    "BrC12C3C4(Cl)C1C1C2C3C41F",                                         # a cubane with three prefixes
])
def test_a_cage_with_more_tied_numberings_than_are_ranked_has_one_name_for_every_spelling(smiles):
    """The numberings that nothing ranks keep the choice the analysis made, and the analysis runs on the
    molecule in the order of its canonical SMILES, the bonds of every atom in the order of their atoms: the
    same molecule is the same graph for every spelling. (Both cages are non-PIN at the default tier.)"""
    names = {be(s)["name"] for s in [smiles] + spellings(smiles, SPELLINGS)}
    assert len(names) == 1, sorted(names)


@pytest.mark.unit
@pytest.mark.parametrize("smiles", [RS, EZ, "C[C@H](CC)C12CCC(CC1)([C@@H](C)CC)CC2", "ClC1C2C(Br)C3C1C3C2"])
def test_the_canonical_copy_is_one_graph_for_every_spelling(smiles):
    """``VonBaeyerAnalyzer._canonical_copy``: the same molecule has the same atoms, bonds, neighbour order,
    chirality and double-bond stereo whichever way it was written, and writes the canonical SMILES of the
    molecule it was made from."""
    from orthonym.rules.polycyclic import VonBaeyerAnalyzer
    forms = set()
    for spelling in spellings(smiles, 20):
        mol = Chem.MolFromSmiles(spelling)
        order = VonBaeyerAnalyzer._canonical_atom_order(mol)
        copy = VonBaeyerAnalyzer._canonical_copy(mol, order)
        assert copy is not None and Chem.MolToSmiles(copy) == Chem.MolToSmiles(mol), spelling
        forms.add((
            tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType()), str(b.GetStereo()),
                   tuple(b.GetStereoAtoms())) for b in copy.GetBonds()),
            tuple((a.GetAtomicNum(), str(a.GetChiralTag()), tuple(n.GetIdx() for n in a.GetNeighbors()))
                  for a in copy.GetAtoms())))
    assert len(forms) == 1


@pytest.mark.unit
@pytest.mark.parametrize("smiles", [
    "CCC12C3C4C5C3C1(I)C5C42Br",            # ethyl, iodo, bromo on a cube
    "CCC12C3C4C5C3C1(C)C5(Br)C42",
    "CC12C3C4C5C3C1(Cl)C5(I)C42",
    "C12(C3C4C1C1(C4C4C5C3C2C5C14)Br)Cl",   # a heptacyclo[6.4.0.0^2,7.0^3,6.0^4,11.0^5,10.0^9,12]dodecane
])
def test_the_numberings_a_decomposition_does_not_reach_are_ranked_too(smiles):
    """The decompositions of a cube give 24 of the 48 numberings of ``pentacyclo[4.2.0.0^2,5.0^3,8.0^4,7]octane``;
    the other 24 are the same cage numbered through an automorphism of its skeleton, and the lowest prefix locants
     (f), (g), the Blue Book,:3307) can be in them. The name is the lowest numbering over the
    automorphisms (the oracle), the same for every spelling, and reads back. (These cages are not PINs: the
    retained name ``cubane`` is preferred, and the fourth has no PIN; the best-effort tier names them.)"""
    names = set()
    for spelling in [smiles] + spellings(smiles, SPELLINGS):
        row = be(spelling)
        assert row["name"] and row["tier"] != "abstain", (spelling, row)
        names.add(row["name"])
    assert len(names) == 1, sorted(names)
    name = names.pop()
    check(smiles, name)
    assert_full_rt(name, smiles)


TMS_ETHYNYL_PORPHYRIN = ("C[Si](C)(C)C#Cc1c2nc(c(-c3ccccc3)c3ccc([nH]3)c(-c3ccccc3)c3nc(c(-c4ccccc4)c4ccc1[nH]4)C=C3)C=C2")
_PORPHYRIN_TAIL = ("-21,22,23,24-tetraazapentacyclo[16.2.1.1^3,6.1^8,11.1^13,16]tetracosa-1(20),2,4,6(24),7,9,11,13(22),14,16,18-"
                   "undecaene")


@pytest.mark.unit
def test_a_prefix_cited_after_phenyl_takes_the_higher_locant_of_the_tied_numberings():
    """ (the Blue Book, section `` ALPHANUMERICAL ORDER``): "The name of a prefix for a substituent is
    considered to begin with the first letter of its complete name", so ``[(trimethylsilyl)ethynyl]`` is cited under
    ``t``, after ``phenyl``; (g) (:3307, section `` NUMBERING``): "lowest locants for the substituent cited first as
    a prefix in the name". The two numberings below carry the same descriptor, the same hetero locants, the same ``ene``
    locants and the same prefix locant set {2,7,12,17} (both read back to the input, OPSIN), so (c) to (f) tie and (g)
    decides: phenyl is cited first and ``2,7,17`` is lower than ``2,12,17`` at its second locant. The older name
    ``2,12,17-triphenyl-7-[...]`` was ranked with the branch named ``ethynyltri(methyl)silyl`` (cited under ``e``), which the
    composed name does not use."""
    lower = "2,7,17-triphenyl-12-[(trimethylsilyl)ethynyl]" + _PORPHYRIN_TAIL
    higher = "2,12,17-triphenyl-7-[(trimethylsilyl)ethynyl]" + _PORPHYRIN_TAIL
    assert_full_rt(lower, TMS_ETHYNYL_PORPHYRIN)
    assert_full_rt(higher, TMS_ETHYNYL_PORPHYRIN)
    phenyl = lambda name: [int(x) for x in name.split("-triphenyl")[0].split(",")]
    assert phenyl(lower) < phenyl(higher) and sorted(phenyl(lower) + [12]) == sorted(phenyl(higher) + [7])
    names = {be(s)["name"] for s in [TMS_ETHYNYL_PORPHYRIN] + spellings(TMS_ETHYNYL_PORPHYRIN, 6)}
    assert names == {lower}, names
