from rdkit import Chem
from orthonym.validation.reconstruct import (
    Verdict, NameFacts, reconstruct_and_verify, verify_or_none)

def _mol(smi):
    return Chem.MolFromSmiles(smi)

def test_confirms_simple_alkanol():
    facts = NameFacts(parent_kind="chain", parent_length=3,
                      principal_group=("ol", (1,)))
    assert reconstruct_and_verify(facts, _mol("CCCO")).verdict == Verdict.CONFIRMED

def test_mismatch_on_wrong_chain_length():
    facts = NameFacts(parent_kind="chain", parent_length=4,
                      principal_group=("ol", (1,)))
    assert reconstruct_and_verify(facts, _mol("CCCO")).verdict == Verdict.MISMATCH

def test_abstains_on_unmodeled_net_charge():
    facts = NameFacts(parent_kind="chain", parent_length=3, net_charge=-1)
    assert reconstruct_and_verify(facts, _mol("CCC(=O)[O-]")).verdict == Verdict.ABSTAINED

def test_zwitterion_input_neutral_facts_is_not_confirmed():
    # net-0 but charge-separated input vs a neutral rebuild -> MISMATCH, never CONFIRM.
    facts = NameFacts(parent_kind="chain", parent_length=3,
                      principal_group=("ol", (1,)))
    v = reconstruct_and_verify(facts, _mol("[O-]C(=O)CC[N+](C)(C)C")).verdict
    assert v in (Verdict.MISMATCH, Verdict.ABSTAINED)

def test_abstains_on_populated_indicated_h():
    facts = NameFacts(parent_kind="chain", parent_length=3,
                      principal_group=("ol", (1,)), indicated_h=(1,))
    assert reconstruct_and_verify(facts, _mol("CCCO")).verdict == Verdict.ABSTAINED

def test_never_raises_returns_error_or_abstain_on_garbage():
    facts = NameFacts(parent_kind="nonsense", parent_length=-1)
    v = reconstruct_and_verify(facts, _mol("CCCO")).verdict
    assert v in (Verdict.ABSTAINED, Verdict.ERROR)  # never CONFIRMED, never raise

def test_confirms_oxa_replacement():
    # 2-oxapropane == dimethyl ether CH3-O-CH3: chain 3, position 2 -> O.
    facts = NameFacts(parent_kind="chain", parent_length=3, replacements=((2, "O"),))
    assert reconstruct_and_verify(facts, _mol("COC")).verdict == Verdict.CONFIRMED

def test_mismatch_on_wrong_replacement_element():
    facts = NameFacts(parent_kind="chain", parent_length=3, replacements=((2, "N"),))
    assert reconstruct_and_verify(facts, _mol("COC")).verdict == Verdict.MISMATCH

def test_confirms_unsaturation():
    facts = NameFacts(parent_kind="chain", parent_length=3, unsaturations=((1, 2),))
    assert reconstruct_and_verify(facts, _mol("C=CC")).verdict == Verdict.CONFIRMED

def test_confirms_named_substituent():
    # 2-methylpropan-1-ol: chain 3, -ol at 1, methyl at 2.
    facts = NameFacts(parent_kind="chain", parent_length=3,
                      principal_group=("ol", (1,)), substituents=(("methyl", 2),))
    assert reconstruct_and_verify(facts, _mol("CC(C)CO")).verdict == Verdict.CONFIRMED

def test_confirms_chloro_substituent():
    # 2-chloropropan-1-ol.
    facts = NameFacts(parent_kind="chain", parent_length=3,
                      principal_group=("ol", (1,)), substituents=(("chloro", 2),))
    assert reconstruct_and_verify(facts, _mol("CC(Cl)CO")).verdict == Verdict.CONFIRMED

def test_dropped_substituent_is_mismatch():
    facts = NameFacts(parent_kind="chain", parent_length=3, principal_group=("ol", (1,)))
    assert reconstruct_and_verify(facts, _mol("CC(C)CO")).verdict == Verdict.MISMATCH

def test_unknown_substituent_name_abstains():
    facts = NameFacts(parent_kind="chain", parent_length=3,
                      substituents=(("some-metal-ligand", 2),))
    assert reconstruct_and_verify(facts, _mol("CC(C)C")).verdict == Verdict.ABSTAINED

def test_confirms_nitro_substituent():
    # 2-nitropropane.
    facts = NameFacts(parent_kind="chain", parent_length=3,
                      substituents=(("nitro", 2),))
    assert reconstruct_and_verify(facts, _mol("CC([N+](=O)[O-])C")).verdict == Verdict.CONFIRMED

def test_confirms_cyano_substituent():
    # 2-methylpropanenitrile analog kept simple: propane-2-carbonitrile via table
    # graft (cyano fragment C#N grafted at locant 2 of propane) == CC(C#N)C.
    facts = NameFacts(parent_kind="chain", parent_length=3,
                      substituents=(("cyano", 2),))
    assert reconstruct_and_verify(facts, _mol("CC(C#N)C")).verdict == Verdict.CONFIRMED

def test_sanitize_failure_is_error_not_raise():
    # A chain of length 1 with five -ol groups piled on the same atom (5 single
    # bonds on one carbon) drives RDKit's SanitizeMol into an
    # AtomValenceException. Verify this is caught by the outer handler in
    # reconstruct_and_verify as ERROR, never a raise and never a false
    # CONFIRMED.
    facts = NameFacts(parent_kind="chain", parent_length=1,
                      principal_group=("ol", (1, 1, 1, 1, 1)))
    result = reconstruct_and_verify(facts, _mol("CO"))
    assert result.verdict == Verdict.ERROR


def test_verify_or_none_opsin_parseable_correct():
    assert verify_or_none("ethanol", "CCO") == "ethanol"

def test_verify_or_none_opsin_parseable_wrong_constitution():
    assert verify_or_none("ethane", "CCO") is None

def test_verify_or_none_rejects_charge_wrong_name():
    # BLOCKER 1: skeleton block matches but full key differs -> must be None.
    assert verify_or_none("propanoic acid", "CCC(=O)[O-]") is None

def test_verify_or_none_rejects_wrong_enantiomer():
    # (R)-name vs (S)-input: skeleton matches, full key differs -> None.
    assert verify_or_none("(2R)-butan-2-ol", "C[C@H](O)CC") is None

def test_verify_or_none_unparseable_reconstructor_confirms():
    facts = NameFacts(parent_kind="chain", parent_length=3, principal_group=("ol", (1,)))
    assert verify_or_none("nonopsin-propan-1-ol", "CCCO", name_facts=facts) \
        == "nonopsin-propan-1-ol"

def test_verify_or_none_unparseable_no_facts_is_none():
    assert verify_or_none("some-nonopsin-name", "CCCO") is None

def test_verify_or_none_wildcard_input_is_none():
    assert verify_or_none("ethane", "CC*") is None

def test_verify_or_none_reconstructor_abstains_on_stereo_input():
    # Wave-0 NameFacts is constitution-only (no stereo fields, and
    # reconstruct_and_verify's _normalize strips stereo both sides), so the
    # raw reconstructor CONFIRMS a chain-4/ol-at-2 rebuild against EITHER the
    # achiral or the chiral butan-2-ol SMILES -- it cannot tell them apart.
    # verify_or_none must not let that ship: an OPSIN-unparseable name paired
    # with a stereo-bearing input must ABSTAIN (None), never CONFIRM a name
    # whose stereo claim was never checked. The achiral witness (same facts,
    # no defined stereocentre) must still CONFIRM -- the guard is scoped to
    # inputs that actually assert stereo, not a blanket abstain.
    facts = NameFacts(parent_kind="chain", parent_length=4,
                       principal_group=("ol", (2,)))
    assert reconstruct_and_verify(facts, _mol("C[C@H](O)CC")).verdict == Verdict.CONFIRMED
    assert verify_or_none("nonopsin-butan-2-ol", "CC(O)CC", name_facts=facts) \
        == "nonopsin-butan-2-ol"
    assert verify_or_none("nonopsin-butan-2-ol", "C[C@H](O)CC", name_facts=facts) is None
