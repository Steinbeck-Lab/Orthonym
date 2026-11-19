"""Names whose stereodescriptors OPSIN cannot read, at the best-effort tier (user decision 2026-11-19), and the
arsenic and antimony oxoacid anions.

OPSIN 2.9.0 and OPSIN main read no lowercase r/s "Achiral cyclic compounds", the Blue Book;
dan2097/opsin#351) and assign no CIP labels on some polyol rings. The best-effort tier ships such a name when OPSIN reads
the stereo-stripped name to the input's constitution, every descriptor matches the input's CIP labels at its unit (the
stereo-layer carve-out), and RDKit's rdCIPLabeler gives every cited unit the code the centres engine gives it
(``validation.second_cip_labeller``); the label is ``best_effort``, never the PIN.

 "Names of mononuclear noncarbon oxoacids" (the Blue Book) names arsonic, arsinic and stibonic acids as
the phosphorus acids are named, and their anions take the same 'ic acid' -> 'ate', 'ous acid' -> 'ite' swap."""
import pytest

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.metrics import provenance as pv
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

#: CHEBI:1090 (a 2-deoxystreptamine phosphate); CHEBI:11369 (an inositol pentakisphosphate)
AMINOCYCLITOL = "N[C@@H]1C[C@H](N)[C@@H](O)[C@H](OP(=O)(O)O)[C@H]1O"
INOSITOL_PHOSPHATE = "O=P(O)(O)O[C@H]1[C@@H](OP(=O)(O)O)[C@H](OP(=O)(O)O)[C@@H](O)[C@H](OP(=O)(O)O)[C@H]1OP(=O)(O)O"


def _row(smiles, tier):
    with jvm_slots(1, purpose="unread-stereo-offer"):
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


@pytest.mark.parametrize("smiles,name", [
    (AMINOCYCLITOL, "(1R,2r,3S,4R,6S)-4,6-diamino-2-(phosphonooxy)cyclohexane-1,3-diol"),
    (INOSITOL_PHOSPHATE, "(1s,2R,3S,4s,5R,6S)-2,3,4,5,6-pentakis(phosphonooxy)cyclohexan-1-ol"),
])
def test_unread_stereo_name_ships_at_best_effort(smiles, name):
    row = _row(smiles, "best-effort")
    assert (row["name"], row["tier"], row["is_pin"]) == (name, "best_effort", False)
    assert row["gate_outcome"] == pv.GATE_OUTCOME_SELF01_CONSTITUTION_ONLY


def test_default_tier_still_declines():
    assert _row(AMINOCYCLITOL, "pin").get("tier") == "abstain"


def test_labellers_that_disagree_decline(monkeypatch):
    import orthonym.validation.second_cip_labeller as second
    monkeypatch.setattr(second, "cited_codes_agree", lambda *a: False)
    assert _row(AMINOCYCLITOL, "best-effort").get("tier") == "abstain"


@pytest.mark.parametrize("smiles,name", [
    ("C[As]([O-])[O-]", "methylarsonite"),
    ("C[As](=O)([O-])[O-]", "methylarsonate"),
    ("C[As](C)(=O)[O-]", "dimethylarsinate"),
    ("c1ccccc1[As](=O)([O-])[O-]", "phenylarsonate"),
    ("C[Sb](=O)([O-])[O-]", "methylstibonate"),
])
def test_arsenic_and_antimony_oxoacid_anions(smiles, name):
    row = _row(smiles, "pin")
    assert (row["name"], row["tier"]) == (name, "pin_verified")
    assert name_is_rt_exact(name, smiles)


def test_unread_blocks_sharing_a_locant_are_not_offered():
    """Two blocks that give one locant different codes could be read swapped onto each other's groups (review
    2026-11-19). OPSIN reads neither r/s block of this urea, so nothing places them: neither name is offered."""
    from orthonym.rules.stereochemistry import strip_stereo_blocks
    from orthonym.validation.second_cip_labeller import cited_codes_agree
    smiles = "C[C@H]1CC[C@H](NC(=O)N[C@H]2CC[C@@H](CC)CC2)CC1"
    swapped = "1-[(1r,4r)-4-ethylcyclohexyl]-3-[(1s,4s)-4-methylcyclohexyl]urea"
    with jvm_slots(1, purpose="unread-stereo-offer"):
        assert not cited_codes_agree(swapped, smiles, strip_stereo_blocks(swapped)[0])


#: a phosphatidylinositol (ChEBI panel row): the glycerol block (2R) and the inositol block share locant 2 with
#: different codes; OPSIN reads (2R) and cannot assign CIP labels on the inositol ring
PI_SMILES = "CCCCCCCCCCCCCCCCCC(=O)OC[C@@H](O)COP(=O)(O)O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O"
PI_NAME = ("(2R)-2-hydroxy-3-[(hydroxy{[(1S,2R,3R,4S,5S,6R)-2,3,4,5,6-pentahydroxycyclohexyl]oxy}phosphoryl)oxy]propyl"
           " octadecanoate")


#: locant 1 is R in the glycerol block and S in the inositol block (swap-open): only OPSIN's placement reads it
PIE_SMILES = "CCCCCC/C=C\\CCCCCCCC(=O)O[C@H](CO)COP(=O)(O)O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O"
PIE_NAME = ("(1R)-1-(hydroxymethyl)-2-[(hydroxy{[(1S,2R,3R,4S,5S,6R)-2,3,4,5,6-pentahydroxycyclohexyl]oxy}phosphoryl)"
            "oxy]ethyl (9Z)-hexadec-9-enoate")


@pytest.mark.parametrize("name,smiles,agrees", [
    (PIE_NAME, PIE_SMILES, True),
    (PIE_NAME.replace("(1R)-", "(1S)-"), PIE_SMILES, False),
    (PI_NAME, PI_SMILES, True),
    # the glycerol centre flipped in the name: OPSIN places (2S) on the glycerol and the full-key check fails
    (PI_NAME.replace("(2R)-", "(2S)-"), PI_SMILES, False),
    # an inositol centre flipped in the name: the searched block no longer matches the input's labels
    (PI_NAME.replace("(1S,2R,3R,", "(1S,2S,3R,"), PI_SMILES, False),
    ("(3S,4R)-1-[(1s,4R)-4-cyano-4-(4-fluorophenyl)cyclohexyl]-3-methyl-4-phenylpiperidine-4-carboxylic acid",
     "[H][C@@]1(C)CN([C@]2([H])CC[C@@](C#N)(c3ccc(F)cc3)CC2)CC[C@]1(C(=O)O)c1ccccc1", True),
])
def test_opsin_places_the_blocks_it_reads(name, smiles, agrees):
    """C2b (2026-11-19): when blocks give one locant different codes, OPSIN places every block it reads on its own
    group (the stereo-free name with that block alone), and only the unread block is searched by locant; blocks that
    give a shared locant one code keep the plain reading."""
    from orthonym.rules.stereochemistry import strip_stereo_blocks
    from orthonym.validation.second_cip_labeller import cited_codes_agree
    with jvm_slots(1, purpose="unread-stereo-offer"):
        assert cited_codes_agree(name, smiles, strip_stereo_blocks(name)[0]) is agrees


def test_phosphatidylinositol_ships_at_best_effort():
    row = _row(PI_SMILES, "best-effort")
    assert (row["name"], row["tier"], row["is_pin"]) == (PI_NAME, "best_effort", False)


def test_same_code_at_a_shared_locant_is_not_swap_open():
    """Blocks that give one locant the same code cannot be read swapped into a wrong name: the old reading stands."""
    from orthonym.validation.stereo_descriptors import swap_open
    assert not swap_open([[("9", "Z"), ("12", "Z")], [("9", "Z")]])
    assert swap_open([[("1", "r"), ("4", "r")], [("1", "s"), ("4", "s")]])


def test_equal_groups_at_unequal_places_still_need_a_placement():
    """The two 4-methylcyclohexyl groups of this pyridine are the same group at places that are not equivalent, so
    the swapped name is a different compound: the descriptor check refuses the swap. The same groups on the
    phosphane of "Achiral cyclic compounds" are equivalent, so both readings name one compound."""
    from orthonym.rules.stereochemistry import strip_stereo_blocks
    from orthonym.validation.stereo_descriptors import stereo_descriptors_verified
    smiles = "C[C@H]1CC[C@H](c2ccc([C@H]3CC[C@@H](C)CC3)cn2)CC1"   # the 2-ring (1r,4r), the 5-ring (1s,4s)
    swapped = "2-[(1s,4s)-4-methylcyclohexyl]-5-[(1r,4r)-4-methylcyclohexyl]pyridine"
    phosphane = "C[C@H]1CC[C@H](P[C@H]2CC[C@@H](C)CC2)CC1"
    with jvm_slots(1, purpose="unread-stereo-offer"):
        assert not stereo_descriptors_verified(swapped, smiles, strip_stereo_blocks(swapped)[0])
        for name in ("[(1r,4r)-4-methylcyclohexyl][(1s,4s)-4-methylcyclohexyl]phosphane",
                     "[(1s,4s)-4-methylcyclohexyl][(1r,4r)-4-methylcyclohexyl]phosphane"):
            assert stereo_descriptors_verified(name, phosphane, strip_stereo_blocks(name)[0])


def test_the_offer_does_not_pair_half_blocks_across_equivalent_groups():
    """Review 2026-11-19: checked one locant at a time, two equivalent groups can each take half of two blocks; the
    best-effort offer of a name OPSIN cannot read does not use that reading (the default tier keeps the reading it
    had)."""
    from orthonym.rules.stereochemistry import strip_stereo_blocks
    from orthonym.validation.second_cip_labeller import cited_codes_agree
    smiles = "C[C@H]1[C@@H](CCCC1)NC(=O)N[C@@H]1[C@H](CCCC1)C"
    wrong = "1-[(1R,2S)-2-methylcyclohexyl]-3-[(1S,2R)-2-methylcyclohexyl]urea"
    with jvm_slots(1, purpose="unread-stereo-offer"):
        assert not cited_codes_agree(wrong, smiles, strip_stereo_blocks(wrong)[0])


@pytest.mark.parametrize("smiles,right", [
    ("C[C@]1(O)C[C@@](c2cccc(OC(F)(F)F)c2)([C@H](O)c2ccccn2)C1", True),     # ring C1 R, methyl C S: the name's
    ("C[C@]1(O)C[C@@](c2cccc(OC(F)(F)F)c2)([C@@H](O)c2ccccn2)C1", False),   # ring C1 S, methyl C R
])
def test_a_locant_free_token_is_not_read_on_another_block_s_centre(smiles, right):
    """A locant-free descriptor may name any unit with its code, so '(S)' of the side chain and '1R' of the ring could
    be read swapped: '1R' on the side chain's methyl C1 and '(S)' on the ring C1 (ChEMBL 37, 12 of 950 names OPSIN
    cannot read, 2026-11-19). "NAMING OF STEREOISOMERS": a descriptor states the unit at its locant. OPSIN now
    places the '(S)' block it reads, and '1R' is read on the ring only."""
    from orthonym.rules.stereochemistry import strip_stereo_blocks
    from orthonym.validation.stereo_descriptors import checked_block_units, swap_open
    name = ("(1R,3s)-3-[(S)-hydroxy(pyridin-2-yl)methyl]-1-methyl-3-[3-(trifluoromethoxy)phenyl]"
            "cyclobutan-1-ol")
    assert swap_open([[("1", "R"), ("3", "s")], [(None, "S")]])
    assert not swap_open([[("1", "R")], [(None, "R")]])
    assert not swap_open([[("1", "R")], [(None, "E")]])
    with jvm_slots(1, purpose="unread-stereo-offer"):
        for symmetric in (True, False):
            reading = checked_block_units(name, smiles, strip_stereo_blocks(name)[0], symmetric=symmetric)
            assert (reading is not None) is right


@pytest.mark.parametrize("name,smiles,right", [
    # review 2026-11-19: '(s)' faces '1r' of another unread block; OPSIN places neither, and the swapped isomer does
    # not read the name, so the right name keeps its reading and the wrong code still fails
    ("(s)-[(1R,2R)-2-methylcyclohexyl][(1S,2S)-2-methylcyclohexyl][(1r,4r)-4-methylcyclohexyl]methanol",
     "O[C@]([C@@H]1CCCC[C@H]1C)([C@H]1CCCC[C@@H]1C)[C@H]1CC[C@H](C)CC1", True),
    ("(r)-[(1R,2R)-2-methylcyclohexyl][(1S,2S)-2-methylcyclohexyl][(1r,4r)-4-methylcyclohexyl]methanol",
     "O[C@]([C@@H]1CCCC[C@H]1C)([C@H]1CCCC[C@@H]1C)[C@H]1CC[C@H](C)CC1", False),
    # a locant-free (Z) and the (2E) of the parent passed swapped onto each other's double bonds
    ("(2E)-3-[(1r,4r)-4-[(Z)-2-phenylethenyl]cyclohexyl]prop-2-enoic acid",
     "OC(=O)/C=C\\[C@H]1CC[C@H](/C=C/c2ccccc2)CC1", False),
    ("(2E)-3-[(1r,4r)-4-[(Z)-2-phenylethenyl]cyclohexyl]prop-2-enoic acid",
     "OC(=O)/C=C/[C@H]1CC[C@H](/C=C\\c2ccccc2)CC1", True),
])
def test_a_locant_free_token_passes_only_when_the_swapped_isomer_does_not(name, smiles, right):
    """When OPSIN cannot place the blocks, a reading with a locant-free token that faces its opposite code is kept only
    if the isomer of the swapped reading (both units inverted) does not read the name too."""
    from orthonym.rules.stereochemistry import strip_stereo_blocks
    from orthonym.validation.stereo_descriptors import checked_block_units
    with jvm_slots(1, purpose="unread-stereo-offer"):
        for symmetric in (True, False):
            assert (checked_block_units(name, smiles, strip_stereo_blocks(name)[0],
                                        symmetric=symmetric) is not None) is right


def test_the_swapped_isomer_is_found_when_opsin_cannot_place_the_blocks(monkeypatch):
    """Without OPSIN's placement the cyclobutanol name has a plain reading for both isomers; the swapped-isomer test
    sees that and the check declines both, where the plain reading accepted the wrong one."""
    from orthonym.rules.stereochemistry import strip_stereo_blocks
    from orthonym.validation import stereo_descriptors as sd
    name = ("(1R,3s)-3-[(S)-hydroxy(pyridin-2-yl)methyl]-1-methyl-3-[3-(trifluoromethoxy)phenyl]"
            "cyclobutan-1-ol")
    right = "C[C@]1(O)C[C@@](c2cccc(OC(F)(F)F)c2)([C@H](O)c2ccccn2)C1"
    wrong = "C[C@]1(O)C[C@@](c2cccc(OC(F)(F)F)c2)([C@@H](O)c2ccccn2)C1"
    reduced = strip_stereo_blocks(name)[0]
    by_block = sd._cip_descriptor_tokens(strip_stereo_blocks(name)[1])
    monkeypatch.setattr(sd, "_opsin_placed_blocks", lambda *a, **k: None)
    with jvm_slots(1, purpose="unread-stereo-offer"):
        for smiles in (right, wrong):
            reading = sd.stereo_block_units(name, smiles, reduced)
            assert reading is not None                       # the plain reading cannot tell the two apart
            assert sd._swapped_reading_passes(name, smiles, reduced, (), reading, by_block)
            assert sd.checked_block_units(name, smiles, reduced) is None
