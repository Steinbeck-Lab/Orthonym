"""A double-bond configuration the standard InChIKey leaves out (an amidine C=N) must agree between the input and
the OPSIN parse of its name (``validation.key_blind_double_bonds``); a configuration the key encodes, or a parse
drawn as another tautomer, is left to the key."""
import pytest
from rdkit import Chem

from orthonym.validation.key_blind_double_bonds import _key_blind_bonds, key_blind_double_bond_verdict


def test_the_amidine_configuration_is_invisible_to_the_key():
    assert len(_key_blind_bonds(Chem.MolFromSmiles("C/C(N)=N/C"))) == 1
    assert _key_blind_bonds(Chem.MolFromSmiles("C/C=C/C")) == []


def test_a_wrong_configuration_is_a_mismatch_and_the_right_one_a_match():
    assert key_blind_double_bond_verdict("C/C(N)=N/C", "C/C(N)=N\\C") == "mismatch"
    assert key_blind_double_bond_verdict("C/C(N)=N/C", "C/C(N)=N/C") == "match"


def test_the_key_and_other_tautomers_are_not_second_guessed():
    assert key_blind_double_bond_verdict("C/C=C/C", "C/C=C\\C") == "n/a"     # the key sees it
    assert key_blind_double_bond_verdict("C/C(N)=N/C", "C/C(=N)NC") == "n/a"  # another tautomer: never rejected here
    assert key_blind_double_bond_verdict("CC(N)=NC", "CC(N)=NC") == "n/a"     # nothing configured


def test_a_parse_without_the_configuration_is_an_omission():
    """A name that leaves the descriptor out reads back with no configuration: an omission, not another molecule."""
    assert key_blind_double_bond_verdict("C/C(N)=N/C", "CC(N)=NC") == "omission"


@pytest.mark.opsin_gate
def test_the_pin_path_refuses_the_opposite_configuration():
    """The PIN path's read-back (``namer._self_consistency_verdict``) refuses a parse with the other configuration of a
    key-blind C=N and accepts the same one."""
    from orthonym.namer import _self_consistency_verdict
    assert _self_consistency_verdict("C/C(N)=N/C", "C/C(N)=N\\C") == "mismatch"
    assert _self_consistency_verdict("C/C(N)=N/C", "C/C(N)=N/C") == "ok"


@pytest.mark.opsin_gate
def test_a_salt_whose_name_omits_the_configuration_keeps_its_name():
    """Review fix: refusing an omission lost this salt's only name at best-effort; it keeps the name it had."""
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    smiles = "CN/C(NCc1ccccc1)=[NH+]\\C.CN/C(NCc1ccccc1)=[NH+]\\C.O=S(=O)([O-])[O-]"
    row = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    assert row["name"] == "bis(N-benzyl-N',N''-dimethylguanidinium) sulfate", row
