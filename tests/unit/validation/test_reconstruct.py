from rdkit import Chem
from orthonym.validation.reconstruct import (
    Verdict, NameFacts, reconstruct_and_verify)

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
