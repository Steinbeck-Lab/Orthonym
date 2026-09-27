import pytest
from orthonym import Orthonym
from orthonym.rules.charged_router import _parent_has_chain_locants


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


# Direct unit tests for _parent_has_chain_locants (JVM-free)
def test_parent_has_chain_locants_saturated():
    """Saturated chain anions with chain locants."""
    assert _parent_has_chain_locants("propanoate") is True


def test_parent_has_chain_locants_unsaturated_enoate():
    """Unsaturated chain anions ending in -enoate."""
    assert _parent_has_chain_locants("(2E)-but-2-enoate") is True


def test_parent_has_chain_locants_dioate():
    """Dicarboxylic chain anions ending in -dioate."""
    assert _parent_has_chain_locants("butanedioate") is True


def test_parent_has_chain_locants_ynone():
    """Triple-bond chain anions ending in -ynoate."""
    assert _parent_has_chain_locants("prop-2-ynoate") is True


def test_parent_has_chain_locants_retained_acetate():
    """Retained anion names without chain locants."""
    assert _parent_has_chain_locants("acetate") is False


def test_parent_has_chain_locants_retained_benzoate():
    """Retained aromatic anion names."""
    assert _parent_has_chain_locants("benzoate") is False


def test_parent_has_chain_locants_retained_2_naphthoate():
    """Retained naphthoate (positional, not chain-numbered)."""
    assert _parent_has_chain_locants("2-naphthoate") is False


# Integration tests with Orthonym.name — require OPSIN gate ON
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # the unsaturated betaine (the fix)
    ("C[N+](C)(C)C/C=C/C(=O)[O-]", "4-(trimethylazaniumyl)(2E)-but-2-enoate"),
    # saturated betaine (regression check)
    ("C[N+](C)(C)CCC(=O)[O-]", "3-(trimethylazaniumyl)propanoate"),
])
def test_unsaturated_betaine_integration(namer, smi, expected):
    """Integration tests for unsaturated and saturated betaine zwitterions."""
    assert namer.name(smi) == expected


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    # longer saturated betaine (regression)
    ("C[N+](C)(C)CCCC(=O)[O-]", "4-(trimethylazaniumyl)butanoate"),
    # carnitine (regression). Name unchanged; labelled below pin_verified since
    # decision A part 1 (test_decision_a_n_substituted_amino_acids.py).
    ("C[N+](C)(C)C[C@H](O)CC(=O)[O-]", "L-carnitine"),
])
def test_betaine_regressions(namer, smi, expected):
    """Ensure saturated cases still work correctly."""
    assert namer.name(smi) == expected
