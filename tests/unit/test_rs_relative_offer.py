"""cis/trans names for ring pairs at the best-effort tier.

The best-effort tier ships a name only if OPSIN reads it fully, and OPSIN 2.9.0 reads no
lowercase r/s. A ring pair that only r/s expresses at the default tier is therefore named
there with the general name of the Blue Book, ``trans-cyclohexane-1,4-diol``
(the Blue Book, "Achiral cyclic compounds"), where one of its words,
``cis`` or ``trans``, is complete for the pair and OPSIN reads the name back to the input's
full InChIKey. OPSIN reads every expected cis/trans name below (checked with a fresh call). A molecule that has
no such word (three stereo units, a geminal pair) is left without a name by the conversion and gets, since
2026-10-20, the row the default tier gives it, a lowercase-descriptor name OPSIN cannot read
(``assembly/default_tier_offer.py``; the last two tests).
"""
import pytest
from rdkit import Chem

from tests.support.default_tier import default_tier_row
from tests.support.rt_assert import assert_full_rt, name_best_effort

pytestmark = pytest.mark.opsin_gate


@pytest.mark.parametrize("smiles,name", [
    ("O[C@H]1CC[C@H](O)CC1", "trans-cyclohexane-1,4-diol"),                               # the Blue Book
    ("O[C@H]1CC[C@@H](O)CC1", "cis-cyclohexane-1,4-diol"),                                #:48111
    ("N[C@H]1CC[C@H](C(=O)O)CC1", "trans-4-aminocyclohexane-1-carboxylic acid"),
    ("C[C@H]1CC[C@H](O)CC1", "trans-4-methylcyclohexan-1-ol"),
    ("C[C@H]1CC[C@H](OC(C)=O)CC1", "trans-4-methylcyclohexyl acetate"),
    ("C[C@H](O)CC[C@H]1CC[C@H](O)CC1", "trans-4-[(3S)-3-hydroxybutyl]cyclohexan-1-ol"),   # a true stereocentre beside the pair
])
def test_relative_name_replaces_the_floor_name(smiles, name):
    # (:48155,:48163); (:48103)
    row = name_best_effort(smiles)
    assert row["name"] == name and row["is_pin"] is False, row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    # with (the Blue Book,:7478): the stereo parentheses are gone,
    # so the group that held the block and every group around it move down one level
    ("OC(=O)C[C@H]1CC[C@H](C)CC1", "(trans-4-methylcyclohexyl)acetic acid"),
    ("C[C@H]1CC[C@H](N2CCCCC2)CC1", "1-(trans-4-methylcyclohexyl)piperidine"),
    ("C[C@H]1CC[C@H](NC(N)=O)CC1", "(trans-4-methylcyclohexyl)urea"),
    ("C[C@H]1CC[C@H](CC(=O)OCC)CC1", "ethyl (trans-4-methylcyclohexyl)acetate"),
    ("C[C@H]1CC[C@H](CN2CCCCC2)CC1", "1-[(trans-4-methylcyclohexyl)methyl]piperidine"),   # two levels: {} becomes
    ("C[C@H]1CC[C@@H](CN2CCCCC2)CC1", "1-[(cis-4-methylcyclohexyl)methyl]piperidine"),
    ("C[C@H]1CC[C@H](COCC(=O)O)CC1", "[(trans-4-methylcyclohexyl)methoxy]acetic acid"),
])
def test_converted_prefix_marks(smiles, name):
    row = name_best_effort(smiles)
    assert row["name"] == name and row["source"] != "t4_floor", row
    assert_full_rt(row["name"], smiles)


def _stripped_skeleton_is_the_inputs(name, smiles):
    """OPSIN (a fresh call, not the engine's oracle) reads the name without its stereo layer to the
    input's constitution: the one check of a name OPSIN 2.9.0 cannot read in full (no lowercase r/s)."""
    from rdkit.Chem import inchi
    from orthonym.rules.stereochemistry import strip_stereo
    from tests.support.rt_assert import _independent_parse
    assert _independent_parse(name, smiles) is None                  # OPSIN reads no r/s: no full read-back
    read = _independent_parse(strip_stereo(name), smiles)
    key = lambda s: inchi.MolToInchiKey(Chem.MolFromSmiles(s)).split("-")[0]
    return read is not None and key(read) == key(smiles)


# Changed asserted values (user decision 2026-10-20, "ship it too": the best-effort tier does not name less than
# the default tier names, ``assembly/default_tier_offer.py``). Before: tier 'abstain', because OPSIN reads no
# lowercase r/s and the cis/trans conversion has no word for these molecules. After: the row the
# default tier gives, name and label (pin_unverified). Evidence: the Blue Book, ``### Achiral
# cyclic compounds``, '(1r,4r)-4-chloro-4-methylcyclohexan-1-ol (PIN)' verbatim for the geminal pair; the Blue
# Book is silent on the (1s,4s) pair and on 2,6-dimethyloxan-4-ol (UNVERIFIED by the source, the same rule
# (a) with. Independent check (not the engine's CIP labeller): RDKit's rdCIPLabeler gives r,r /
# s,s / (S, r, R) for the three inputs, and OPSIN reads the stereo-stripped names to each input's constitution.
# Not independently checked: which of 2 and 6 takes R and which S in '(2R,4r,6S)'.
def test_three_centres_get_the_default_tier_row_at_best_effort():
    smiles = "C[C@H]1C[C@@H](O)C[C@@H](C)O1"
    row = name_best_effort(smiles)
    assert row["name"] == "(2R,4r,6S)-2,6-dimethyloxan-4-ol" == default_tier_row(smiles)["name"], row
    assert row["tier"] == "pin_unverified" and row["is_pin"] is False, row
    assert _stripped_skeleton_is_the_inputs(row["name"], smiles)


@pytest.mark.parametrize("smiles,name", [
    ("C[C@]1(Cl)CC[C@H](O)CC1", "(1r,4r)-4-chloro-4-methylcyclohexan-1-ol"),    #:48115 (PIN), verbatim
    ("C[C@@]1(Cl)CC[C@H](O)CC1", "(1s,4s)-4-chloro-4-methylcyclohexan-1-ol"),
])
def test_geminal_pair_ring_gets_the_default_tier_row_at_best_effort(smiles, name):
    # the cis/trans name has no OPSIN reading ('trans-4-chloro-4-methylcyclohexan-1-ol',:48115)
    row = name_best_effort(smiles)
    assert row["name"] == name == default_tier_row(smiles)["name"], row
    assert row["tier"] == "pin_unverified" and row["is_pin"] is False, row
    assert _stripped_skeleton_is_the_inputs(row["name"], smiles)


def test_chiral_pair_gets_no_relative_name():
    # a relative word is 50/50 on a chiral pair (namer.py:2104-2108); the absolute name stays
    row = name_best_effort("O[C@H]1CCCC[C@@H]1O")
    assert row["name"] == "(1S,2S)-cyclohexane-1,2-diol" and row["tier"] == "pin_verified", row


def test_fused_pair_unchanged():
    assert name_best_effort("C1CC[C@H]2CCCC[C@H]2C1")["name"] == "cis-decahydronaphthalene"


def test_handoff_is_inert_without_rs():
    row = name_best_effort("C[C@H](O)CC")                  # R/S only: no lowercase block, no offer
    assert not row["name"].startswith(("cis-", "trans-")), row
    assert_full_rt(row["name"], "C[C@H](O)CC")


def test_two_ring_pairs_each_take_their_own_marks():
    # two blocks in one name: the enclosing marks of each block's chain are re-derived, rightmost
    # block first, so the middle ring is written in  inside
    smiles = "O[C@H]1CC[C@H](CC[C@H]2CC[C@H](O)CC2)CC1"
    row = name_best_effort(smiles)
    assert row["name"] == "trans-4-[2-(trans-4-hydroxycyclohexyl)ethyl]cyclohexan-1-ol", row
    assert_full_rt(row["name"], smiles)


def test_two_ring_pairs_of_one_molecule_in_either_sense_read_back():
    # the molecule is symmetric under exchange of its two rings, so cis/trans and trans/cis name one
    # structure; the first sense that reads back to the input's full key is the one cited
    smiles = "O[C@H]1CC[C@H](CC[C@@H]2CC[C@H](O)CC2)CC1"
    row = name_best_effort(smiles)
    assert row["name"] == "cis-4-[2-(trans-4-hydroxycyclohexyl)ethyl]cyclohexan-1-ol", row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles", [
    "O[C@H]1CC[C@H](O)CC1",
    "O[C@H]1CC[C@@H](O)CC1",
    "OC(=O)C[C@H]1CC[C@H](C)CC1",
    "C[C@H]1CC[C@H](N2CCCCC2)CC1",
])
def test_labels_of_a_relative_name(smiles):
    # (the Blue Book, "Relative configuration: the stereodescriptors 'cis'
    # and 'trans'"): the general name is a correct systematic name that is not the preferred
    # one (the PIN cites the lowercase r/s descriptors,, and OPSIN reads it in full
    row = name_best_effort(smiles)
    assert row["tier"] == "systematic_verified", row
    assert row["is_pin"] is False and row["verified"] == "opsin", row
    assert row["gate_outcome"] == "full_key_round_trip_verified", row
    assert row["source"] != "t4_floor", row


@pytest.mark.parametrize("smiles,name,producer", [
    ("O[C@H]1CC[C@@H](O)CC1", "cis-cyclohexane-1,4-diol", "inner:general_acyclic"),
    ("O[C@H]1CC[C@H](O)CC1", "trans-cyclohexane-1,4-diol", "inner:general_acyclic"),
    ("N[C@H]1CC[C@@H](C(=O)O)CC1", "cis-4-aminocyclohexane-1-carboxylic acid", "inner:ester_family"),
])
def test_a_relative_name_names_the_producer_of_the_name_it_was_made_from(smiles, name, producer):
    # (the Blue Book): the cis/trans name is the conversion of the name with lowercase
    # r/s that a producer built for the whole molecule,:48103), so the row names that
    # producer (the per-call record, metrics/producer_trail). The default tier's row for the same
    # molecule names the same producer: only the spelling of the stereodescriptor differs.
    row = name_best_effort(smiles)
    assert (row["name"], row["tier"], row["producer"]) == (name, "systematic_verified", producer), row
    assert default_tier_row(smiles)["producer"] == producer


def test_convert_pair_blocks_passes_the_producer_label_on():
    # the conversion turns a name that a seam returned into another string: the label follows to
    # every candidate (the senses of the block), and a name no seam returned gives no label
    from orthonym.metrics import producer_trail as pt
    from orthonym.rules.relative_stereo import convert_pair_blocks
    name = "(1r,4r)-cyclohexane-1,4-diol"
    smiles = "O[C@H]1CC[C@H](O)CC1"
    mol = Chem.MolFromSmiles(smiles)
    records = _records(name, smiles)
    token = pt.open_trail(smiles)
    try:
        pt.noted("inner:general_acyclic", name, mol)
        out = convert_pair_blocks(name, mol, records)
        assert out == ["cis-cyclohexane-1,4-diol", "trans-cyclohexane-1,4-diol"], out
        assert [pt.lookup(c) for c in out] == ["inner:general_acyclic"] * 2
    finally:
        pt.close_trail(token)
    token = pt.open_trail(smiles)
    try:
        out = convert_pair_blocks(name, mol, records)              # no seam returned `name`
        assert out and [pt.lookup(c) for c in out] == [None] * len(out)
    finally:
        pt.close_trail(token)


def test_default_tier_keeps_its_r_s_name_and_label():
    # the default tier's emission rule is unchanged by the best-effort offer
    row = default_tier_row("O[C@H]1CC[C@H](O)CC1")
    assert row["name"] == "(1r,4r)-cyclohexane-1,4-diol", row
    assert row["tier"] == "pin_unverified" and row["is_pin"] is False, row
    assert row["gate_outcome"] == "self_consistency_constitution_only", row


# --- the pieces, without a whole naming -------------------------------------------------------

def _records(name, smiles):
    from orthonym.namer import _stereo_block_units
    from orthonym.rules.stereochemistry import strip_stereo_blocks
    records = _stereo_block_units(name, smiles, strip_stereo_blocks(name)[0])
    assert records is not None, (name, smiles)
    return records


def test_convert_pair_blocks_rederives_marks():
    from orthonym.rules.relative_stereo import convert_pair_blocks
    name = "1-{[(1r,4r)-4-methylcyclohexyl]methyl}piperidine"
    smiles = "C[C@H]1CC[C@H](CN2CCCCC2)CC1"
    out = convert_pair_blocks(name, Chem.MolFromSmiles(smiles), _records(name, smiles))
    assert "1-[(trans-4-methylcyclohexyl)methyl]piperidine" in out, out
    assert "1-[(cis-4-methylcyclohexyl)methyl]piperidine" in out, out
    for cand in out:
        # a bracket that holds no parenthesis breaks the nesting order {[({})]}
        assert not ("[" in cand and "(" not in cand[cand.index("["):]), cand
        assert "{" not in cand, cand


def test_convert_pair_blocks_candidates_are_the_two_senses():
    from orthonym.rules.relative_stereo import convert_pair_blocks
    name = "(1r,4r)-cyclohexane-1,4-diol"
    smiles = "O[C@H]1CC[C@H](O)CC1"
    out = convert_pair_blocks(name, Chem.MolFromSmiles(smiles), _records(name, smiles))
    assert out == ["cis-cyclohexane-1,4-diol", "trans-cyclohexane-1,4-diol"], out


@pytest.mark.parametrize("name,smiles", [
    # a block cited under a multiplier is not converted (the Blue Book)
    ("bis[(1r,4r)-4-methylcyclohexyl]phosphane", "C[C@H]1CC[C@H](P[C@H]2CC[C@H](C)CC2)CC1"),
    # a ring atom with no hydrogen: OPSIN reads no cis/trans name (the Blue Book)
    ("(1r,4r)-4-chloro-4-methylcyclohexan-1-ol", "C[C@]1(Cl)CC[C@H](O)CC1"),
    # two rings share atoms
    ("(4as,8as)-decahydronaphthalene", "C1CC[C@H]2CCCC[C@H]2C1"),
])
def test_convert_pair_blocks_refuses(name, smiles):
    from orthonym.namer import _stereo_block_units
    from orthonym.rules.relative_stereo import convert_pair_blocks
    from orthonym.rules.stereochemistry import strip_stereo_blocks
    records = _stereo_block_units(name, smiles, strip_stereo_blocks(name)[0])
    if records is None:        # the verifier already declines it: nothing to convert
        return
    assert convert_pair_blocks(name, Chem.MolFromSmiles(smiles), records) == []


def test_convert_pair_blocks_leaves_an_absolute_block_alone():
    from orthonym.rules.relative_stereo import convert_pair_blocks
    # a meso pair: no lowercase token, the absolute block stays
    name = "(1R,2S)-cyclohexane-1,2-diol"
    smiles = "O[C@H]1CCCC[C@H]1O"
    assert convert_pair_blocks(name, Chem.MolFromSmiles(smiles), _records(name, smiles)) == []


def test_withdrawn_slot_keeps_only_whole_molecule_names_with_a_lowercase_block():
    from orthonym.rules import relative_stereo as rs
    rs.clear_withdrawn_names()
    smiles = "O[C@H]1CC[C@H](O)CC1"
    rs.note_withdrawn_name("(2S)-butan-2-ol", smiles, smiles)                      # no lowercase code
    rs.note_withdrawn_name("cyclohexane-1,4-diol", smiles, smiles)                 # no block
    rs.note_withdrawn_name("(1r,4r)-cyclohexane-1,4-diol", smiles, smiles)
    rs.note_withdrawn_name("(1r,4r)-cyclohexane-1,4-diol", smiles, smiles)         # once
    rs.note_withdrawn_name("(1r,4r)-4-aminocyclohexan-1-ol", smiles, smiles)       # a second name, in order
    # a gate call on a fragment of the molecule (its string differs from the caller's input)
    rs.note_withdrawn_name("(1R,3r,5S)-tropan-3-ol", "C1CC1", smiles)
    # no outermost naming, no caller input
    rs.note_withdrawn_name("(1s,4s)-cyclohexane-1,4-diol", smiles, None)
    assert rs.withdrawn_names(smiles) == ("(1r,4r)-cyclohexane-1,4-diol",
                                          "(1r,4r)-4-aminocyclohexan-1-ol")
    assert rs.withdrawn_names("C1CC1") == ()
    assert rs.withdrawn_names("not noted") == ()
    rs.clear_withdrawn_names()
    assert rs.withdrawn_names(smiles) == ()


def test_the_gate_hands_over_a_withdrawn_lowercase_name_at_best_effort_only():
    import orthonym.namer as namer
    from orthonym.rules import relative_stereo as rs
    smiles = "O[C@H]1CC[C@H](O)CC1"
    name = "(1r,4r)-cyclohexane-1,4-diol"
    rs.clear_withdrawn_names()
    namer._CALLER_INPUT.smiles = smiles            # the outermost name sets this
    try:
        # default tier: the stereo-layer carve-out ships the name, nothing is handed over
        shipped = namer._final_opsin_validity_gate(name, smiles, None, general_fallback_tier=False)
        assert shipped == name and rs.withdrawn_names(smiles) == ()
        # best-effort tier: BE-STRICT withdraws it, and the withdrawn name is handed over
        gate_out = namer._final_opsin_validity_gate(
            name, smiles, None, besteffort_unverified=True, general_fallback_tier=True)
        assert gate_out != name
        assert rs.withdrawn_names(smiles) == (name,)
        # a gate call on another string (a fragment of the molecule) hands nothing over
        rs.clear_withdrawn_names()
        namer._final_opsin_validity_gate(
            name, "C1CCCCC1", None, besteffort_unverified=True, general_fallback_tier=True)
        assert rs.withdrawn_names(smiles) == () and rs.withdrawn_names("C1CCCCC1") == ()
        # a name without a lowercase block that the gate lets through hands nothing over
        rs.clear_withdrawn_names()
        namer._CALLER_INPUT.smiles = "OC1CCC(O)CC1"
        namer._final_opsin_validity_gate("cyclohexane-1,4-diol", "OC1CCC(O)CC1", None,
                                         besteffort_unverified=True, general_fallback_tier=True)
        assert rs.withdrawn_names("OC1CCC(O)CC1") == ()
    finally:
        namer._CALLER_INPUT.smiles = None
        rs.clear_withdrawn_names()


def test_the_slot_is_empty_after_a_naming():
    from orthonym.rules import relative_stereo as rs
    name_best_effort("O[C@H]1CC[C@H](O)CC1")
    assert rs.withdrawn_names("O[C@H]1CC[C@H](O)CC1") == ()


def test_the_offer_is_not_built_when_the_primary_round_trips():
    from orthonym.cli import _emit_tier_flags
    from orthonym.namer import Orthonym
    from orthonym.rules import relative_stereo as rs
    smiles = "O[C@H]1CC[C@H](O)CC1"
    mol = Chem.MolFromSmiles(smiles)
    ns = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    rs.clear_withdrawn_names()
    rs.note_withdrawn_name("(1r,4r)-cyclohexane-1,4-diol", smiles, smiles)
    ns._offers = []
    ns._maybe_append_relative_stereo_offer(mol, smiles, "trans-1,4-dihydroxycyclohexane",
                                           primary_is_failure=False)
    assert ns._offers == []                      # the primary reads back: nothing is offered
    ns._maybe_append_relative_stereo_offer(mol, smiles, "unknown organic compound",
                                           primary_is_failure=True)
    assert [o.name for o in ns._offers] == ["trans-cyclohexane-1,4-diol"], ns._offers
    offer = ns._offers[0]
    assert (offer.tier, offer.is_pin, offer.complete) == ("systematic_verified", False, True), offer
    rs.clear_withdrawn_names()


def test_the_offer_is_built_for_no_other_input():
    from orthonym.cli import _emit_tier_flags
    from orthonym.namer import Orthonym
    from orthonym.rules import relative_stereo as rs
    smiles = "O[C@H]1CC[C@H](O)CC1"
    ns = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    rs.clear_withdrawn_names()
    rs.note_withdrawn_name("(1r,4r)-cyclohexane-1,4-diol", "OC1CCC(O)CC1", "OC1CCC(O)CC1")   # another input's name
    ns._offers = []
    ns._maybe_append_relative_stereo_offer(Chem.MolFromSmiles(smiles), smiles,
                                           "unknown organic compound", primary_is_failure=True)
    assert ns._offers == []
    rs.clear_withdrawn_names()


# --- which withdrawn name is offered ---------------------------------------------------------------

#: A cyclobutane pair beside a chiral side chain. The cascade withdrew three names for it: two spell
#: the isoxazole acyl group as a skeletal replacement chain (``2-oxaeth-1-en-1-yl``,
#: ``1-oxa-2-azacyclopenta-2,4-dien-4-yl``), the third is the name of the Blue Book's kind.
_CHAIN_ROW = "Cc1nocc1C(=O)N1CCCO[C@@H](CNC(=O)[C@H]2C[C@@H](c3ccccc3)C2)C1"
_CHAIN_NAMES = (
    "(1s,3R)-N-({(2S)-4-[1-(3-methyl-1,2-oxazol-4-yl)-2-oxaeth-1-en-1-yl]-1,4-oxazepan-2-yl}"
    "methyl)-3-phenylcyclobutane-1-carboxamide",
    "(1s,3R)-N-({(2S)-4-[1-(3-methyl-1-oxa-2-azacyclopenta-2,4-dien-4-yl)-2-oxaeth-1-en-1-yl]-"
    "1,4-oxazepan-2-yl}methyl)-3-phenylcyclobutane-1-carboxamide",
    "(1s,3R)-N-{[(2S)-4-(3-methyl-1,2-oxazole-4-carbonyl)-1,4-oxazepan-2-yl]methyl}-"
    "3-phenylcyclobutane-1-carboxamide",
)


def _offer_for(smiles, withdrawn):
    from orthonym.cli import _emit_tier_flags
    from orthonym.namer import Orthonym
    from orthonym.rules import relative_stereo as rs
    ns = Orthonym(style="pin", **_emit_tier_flags("best-effort"))
    rs.clear_withdrawn_names()
    for name in withdrawn:
        rs.note_withdrawn_name(name, smiles, smiles)
    ns._offers = []
    ns._maybe_append_relative_stereo_offer(Chem.MolFromSmiles(smiles), smiles,
                                           "unknown organic compound", primary_is_failure=True)
    rs.clear_withdrawn_names()
    return [o.name for o in ns._offers]


def test_the_first_eligible_withdrawn_name_is_offered():
    assert _offer_for(_CHAIN_ROW, _CHAIN_NAMES) == [
        "cis-N-{[(2S)-4-(3-methyl-1,2-oxazole-4-carbonyl)-1,4-oxazepan-2-yl]methyl}-"
        "3-phenylcyclobutane-1-carboxamide"]


@pytest.mark.parametrize("names", [_CHAIN_NAMES[:2], _CHAIN_NAMES[:1]])
def test_a_replacement_chain_name_is_not_converted(names):
    # the cascade's own spelling of the acyl group is a chain by skeletal replacement;
    # a converted name of that kind is not offered
    assert _offer_for(_CHAIN_ROW, names) == []


def test_the_replacement_chain_filter_alone_withholds_the_name(monkeypatch):
    # with the classifier off the same withdrawn name converts and reads back to the input
    import orthonym.rules.relative_stereo as rs
    monkeypatch.setattr(rs, "names_a_replacement_chain", lambda name: False)
    offered = _offer_for(_CHAIN_ROW, _CHAIN_NAMES[:1])
    assert [n.startswith("cis-N-({(2S)-4-[1-(3-methyl-1,2-oxazol-4-yl)-2-oxaeth-1-en-1-yl]")
            for n in offered] == [True], offered


def test_a_name_the_descriptor_check_refuses_is_skipped_for_the_next_one():
    # (1s,4s) cites the other sense of the pair: the check refuses it, the next name is used
    smiles = "O[C@H]1CC[C@H](O)CC1"
    assert _offer_for(smiles, ["(1s,4s)-cyclohexane-1,4-diol",
                               "(1r,4r)-cyclohexane-1,4-diol"]) == ["trans-cyclohexane-1,4-diol"]
    assert _offer_for(smiles, ["(1s,4s)-cyclohexane-1,4-diol"]) == []


def test_a_lowercase_block_in_the_middle_of_a_name_is_not_converted():
    # (the Blue Book): a descriptor block is cited at the front of the name or of a
    # prefix. A block after a closing mark is not one the descriptor check locates, so the
    # reduced name still holds it, OPSIN reads no lowercase r/s, and there is no reading to convert.
    from orthonym.namer import _stereo_block_units
    from orthonym.rules.stereochemistry import strip_stereo_blocks
    smiles = "CC(C)CNC(=O)[C@H]1C[C@H](F)C1"
    front = "(1r,3r)-3-fluoro-N-(2-methylpropyl)cyclobutane-1-carboxamide"
    middle = "N-(2-methylpropyl)(1r,3r)-3-fluorocyclobutane-1-carboxamide"
    assert _stereo_block_units(front, smiles, strip_stereo_blocks(front)[0]) is not None
    assert _stereo_block_units(middle, smiles, strip_stereo_blocks(middle)[0]) is None
    from orthonym.rules import relative_stereo as rs
    rs.clear_withdrawn_names()
    rs.note_withdrawn_name(middle, smiles, smiles)       # no located lowercase block: not stored
    assert rs.withdrawn_names(smiles) == ()
    rs.clear_withdrawn_names()
    assert _offer_for(smiles, [middle]) == []
    assert _offer_for(smiles, [front]) == ["trans-3-fluoro-N-(2-methylpropyl)cyclobutane-1-carboxamide"]


@pytest.mark.parametrize("name", [
    "2-imino-5-[({[(1S,4r)-4-{3-[(3S)-3-(1-oxaethyl)oxolan-3-yl]-4-oxa-2-azabut-3-en-1-yl}cyclohexyl]methyl}amino)-oxomethyl]-2,3-dihydro-1,3-oxazole",
    "(1s,3R)-N-({(2S)-4-[1-(3-methyl-1,2-oxazol-4-yl)-2-oxaeth-1-en-1-yl]-1,4-oxazepan-2-yl}methyl)-3-phenylcyclobutane-1-carboxamide",
    "2-[(1r,3R)-3-(methylsulfanyl)cyclobutyl]-N-{(1S)-1-(2-oxaethyl)-2-[(3S)-3,6,6-trimethyl-1-oxa-4-azacyclohexan-4-yl]-3-oxaprop-2-en-1-yl}acetamide",
    "(1r,4R)-N-{[(7R)-5-(3,3-dimethyl-1-oxo-2-oxabutyl)-8-oxa-5-azaspiro[3.5]nonan-7-yl]methyl}-4-ethylcyclohexane-1-carboxamide",
    "(1S,3s)-3-{2-[(1R,4R)-4-(3,3-difluoro-2-oxo-1-azahexyl)cycloheptan-1-yl]-1-methyl-3-oxa-1-azaprop-2-en-1-yl}cyclobutane-1-carboxamide",
    "(1s,3R)-3-(dimethylamino)-N-{(1S,3S)-5-[1-(1-oxa-3-azacyclopenta-2,4-dien-5-yl)-2-oxaeth-1-en-1-yl]-5-azaspiro[2.4]heptan-1-yl}cyclobutane-1-carboxamide",
    "1-oxacyclopentan-3-yl", "3-azahexanamide", "2,3-dioxa-1,4-diazapentane", "1,3,5-triazadecane",
])
def test_names_a_replacement_chain(name):
    from orthonym.rules.relative_stereo import names_a_replacement_chain
    assert names_a_replacement_chain(name)


@pytest.mark.parametrize("name", [
    "(1r,4r)-cyclohexane-1,4-diol",
    "(1s,3R)-N-{[(2S)-4-(3-methyl-1,2-oxazole-4-carbonyl)-1,4-oxazepan-2-yl]methyl}-3-phenylcyclobutane-1-carboxamide",
    "N-{(5S,6S)-2-azaspiro[4.5]decan-6-yl}cyclobutane-1-carboxamide",       # spiro with a replacement prefix
    "8-oxa-5-azaspiro[3.5]nonane", "1-azabicyclo[2.2.2]octane", "2,3-diazabicyclo[4.3.0]nona-1(6),2,4-triene",
    "oxolan-2-yl", "2,2-dimethyloxane", "1,4-oxazepane", "1,3-oxazole", "1,2,4-oxadiazol-3-yl", "thiazol-2-yl",
    "azepane", "azetidin-1-yl", "azane", "silane", "phosphane", "oxane-2-carboxamide", "oxamic acid",
    "N-methylcarboxamide", "dithiane", "thianthrene",
])
def test_does_not_name_a_replacement_chain(name):
    from orthonym.rules.relative_stereo import names_a_replacement_chain
    assert not names_a_replacement_chain(name)
