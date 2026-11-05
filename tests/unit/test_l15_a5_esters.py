"""A chain ester whose acid side holds another ester is not named with that ester's C=O dropped (``rules.esters``).

 Esters cited as prefixes: an ester group is cited as a prefix ('alkoxy...oxo', 'alkoxycarbonyl') only
"when another group is present that has priority for citation as the principal group". Systems composed of
rings and chains: the ring is senior to the chain, so for a chain ester and a ring ester the ring ester is the parent
('methyl 4-(3-methoxy-3-oxopropyl)benzoate'), and the chain ester is cited as '3-methoxy-3-oxopropyl'.

``name_ester`` took every acid-side C=O off the acid word, another ester's included: the prefix pipeline then named the
O-less fragment ('4-(methoxymethyl)phenyl'), the wrong name passed the probe, and the decomposition that gives the PIN was
skipped. The chain-ester match now declines, as the ring-acid path already did for its own carbonyl.
"""
import pytest
from rdkit import Chem

from orthonym.cli import _tiered_namer
from orthonym.rules.esters import name_ester
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

_TIERS = ["valid", "complete", "best-effort"]

_DIESTER = "COC(=O)CCc1ccc(cc1)C(=O)OC"
_ROW16 = ("COC(=O)[C@H](C)CNC(=O)[C@H](Cc1ccc(OC)c(Cl)c1)NC(=O)/C=C/C[C@H](OC(=O)[C@@H](O)CC(C)C)"
          "[C@H](C)[C@H]1O[C@@H]1c1ccccc1")
# The alcohol group runs through its free valence: "THE PRINCIPAL SUBSTITUENT CHAIN" (the Blue Book), (b)
# "the greater number of skeletal atoms, i.e., the longest chain", before (d) the multiple bonds (:22639). The chain
# through the attachment carbon has seven atoms (hept-5-en-3-yl); the five atoms of the chain counted from the free
# valence ('...-5-oxo-1-{1-[...oxiranyl]ethyl}pent-3-en-1-yl', this row's earlier name) are the shorter chain.
_ROW16_PIN = ("(2S,3S,5E)-7-{[(2S)-3-(3-chloro-4-methoxyphenyl)-1-{[(2R)-3-methoxy-2-methyl-3-oxopropyl]amino}-1-oxopropan-2-yl]"
              "amino}-7-oxo-2-[(2R,3R)-3-phenyloxiran-2-yl]hept-5-en-3-yl (2S)-2-hydroxy-4-methylpentanoate")

_PIN_CASES = [
    (_DIESTER, "methyl 4-(3-methoxy-3-oxopropyl)benzoate"),
    (_ROW16, _ROW16_PIN),
]


def _row(smiles, tier):
    return _tiered_namer("pin", tier).name_tiered(smiles)


@pytest.mark.parametrize("smiles,name", _PIN_CASES)
def test_second_ester_is_not_dropped(smiles, name):
    """Best-effort named both with the ester as a prefix (the benzoate's 'methoxycarbonyl' on a benzene)."""
    row = _row(smiles, "best-effort")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row
    assert_full_rt(row["name"], smiles)


def test_second_ester_is_not_dropped_by_the_valid_and_complete_tiers():
    """The decomposition names the diester in these tiers with 'benzene-1-carboxylate' (a systematic spelling: the
    acid half 'benzoic acid' has no producer there); both ester groups stay in the name and the read-back is exact.
    This is a known non-PIN (the book's name is 'methyl 4-(3-methoxy-3-oxopropyl)benzoate', retained
    'benzoic acid'): the row is a status pin of the tier label, not a fix proof; it passed before this task as well
    (open item (b) of the lane ledger)."""
    for tier in ("valid", "complete"):
        row = _row(_DIESTER, tier)
        assert (row["name"], row["tier"]) == (
            "methyl 4-(3-methoxy-3-oxopropyl)benzene-1-carboxylate", "systematic_verified"), row
        assert_full_rt(row["name"], _DIESTER)


@pytest.mark.parametrize("tier", _TIERS)
def test_row16_gives_the_pin(tier):
    row = _row(_ROW16, tier)
    assert (row["name"], row["tier"]) == (_ROW16_PIN, "pin_verified"), row
    assert_full_rt(row["name"], _ROW16)


@pytest.mark.parametrize("smiles,name", _PIN_CASES)
def test_the_pin_tier_name_is_unchanged(smiles, name):
    row = _row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified"), row


def test_chain_ester_match_declines_when_the_acid_side_holds_another_ester():
    """The match (2, 3, 1, 0) is the chain ester (C=O, O, ester O, methyl); the benzoate C=O lies off its chain. The
    wrong name 'methyl 3-[4-(methoxymethyl)phenyl]propanoate' came only at best-effort, so the producer is read there."""
    from orthonym.metrics import provenance
    mol = Chem.MolFromSmiles(_DIESTER)
    assert [mol.GetAtomWithIdx(i).GetSymbol() for i in (0, 1, 2, 3)] == ["C", "O", "C", "O"]
    token = provenance.best_effort_ctx.set(True)
    try:
        assert name_ester(mol, (2, 3, 1, 0)) is None
    finally:
        provenance.best_effort_ctx.reset(token)


# Controls: the names of molecules with an acid-side C=O that the guard must leave alone, same at every tier.
_CONTROLS = [
    # the amide C=O is on the acid chain
    ("CCOC(=O)C(C)CC(N)=O", "ethyl 4-amino-2-methyl-4-oxobutanoate", "pin_verified"),
]


@pytest.mark.parametrize("tier", ["pin"] + _TIERS)
@pytest.mark.parametrize("smiles,name,label", _CONTROLS)
def test_controls_are_unchanged(smiles, name, label, tier):
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, label), row
    assert_full_rt(row["name"], smiles)


# Controls the PIN tier does not name (it abstains), so the guard did not open a new route to a name that drops an atom: the
# wider tiers name them as esters, in the form of a preferred name that the PIN tier did not certify (the label is
# ``pin_unverified``, not ``pin_verified``; ``is_pin`` is False). "General methodology" (the Blue Book):
# "All preferred IUPAC names for esters are named by functional class nomenclature"; the 'acyloxy' of "Esters
# cited as prefixes" (:31696) is for an ester beside a group senior to it, and here the other group is an ester of its own
# acid (the 'methyl' of the acetoxy acid's ester is the alcohol word). lane/l17 item 2: the wider tiers named these two with
# the ester only in a prefix ('1-(acetyloxy)-2-methoxy-2-oxoethane', a name of a lower class, Table 4.1 class 9 esters above
# class 17 hydroxy compounds); the ester name is read back by OPSIN to the input (``assert_full_rt``).
_NON_PIN_CONTROLS = [
    ("COC(=O)COC(C)=O", "methyl (acetyloxy)acetate"),
    ("CC(=O)OC(C)C(=O)OC", "methyl 2-(acetyloxy)propanoate"),
]


@pytest.mark.parametrize("tier", _TIERS)
@pytest.mark.parametrize("smiles,name", _NON_PIN_CONTROLS)
def test_non_pin_controls_name_the_ester(smiles, name, tier):
    row = _row(smiles, tier)
    assert (row["name"], row["tier"]) == (name, "pin_unverified"), row
    assert_full_rt(row["name"], smiles)
