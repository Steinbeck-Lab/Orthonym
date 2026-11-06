import pytest
from tests.support.read_back import same_key


@pytest.mark.opsin_gate
def test_brace_escalation_oxirane(namer):
    smi = "CCC(C)OCC1CO1"
    name = namer.name(smi)
    assert name == "{[(butan-2-yl)oxy]methyl}oxirane", name
    assert "[[" not in name and "]]" not in name, name
    assert same_key(smi, name), name
