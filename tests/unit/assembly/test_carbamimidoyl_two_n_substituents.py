"""An amidine group with two substituents on its amino nitrogen is 'N,N-...carbamimidoyl'.

; the Blue Book '4-[methyl(N-methyl-N-phenylcarbamimidoyl)amino]benzoic acid (PIN)', not
'4-({imino[methyl(phenyl)amino]methyl}methylamino)benzoic acid'. assembly.amidine_prefix.carbamimidoyl_on_n names none,
one or two substituents (the branch fell to '[(dimethylamino)(imino)methyl]', labelled pin_unverified, with two).
"""
import pytest

from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name,tier", [
    ("CN(C(=N)N(C)c1ccccc1)c1ccc(C(=O)O)cc1", "4-[methyl(N-methyl-N-phenylcarbamimidoyl)amino]benzoic acid",
     "best-effort"),
    ("CN(C)C(=N)Nc1ccc(C(=O)O)cc1", "4-[(N,N-dimethylcarbamimidoyl)amino]benzoic acid", "pin"),
    ("CNC(=N)Nc1ccc(C(=O)O)cc1", "4-[(N-methylcarbamimidoyl)amino]benzoic acid", "pin"),
    ("NC(=N)Nc1ccc(C(=O)O)cc1", "4-(carbamimidoylamino)benzoic acid", "pin"),
    ("CC(C)C1=CC=C(C=C1)N2CCC(=CC3=CC(=CC=C3)O)C2=NC(=N)N(C)CCOC",
     "3-[(3-hydroxyphenyl)methylidene]-2-{[N-(2-methoxyethyl)-N-methylcarbamimidoyl]imino}-1-[4-(propan-2-yl)phenyl]"
     "pyrrolidine", "best-effort"),
])
def test_carbamimidoyl_n_substituents(smiles, name, tier):
    row = tier_row(smiles, tier)
    assert row["name"] == name, row
    assert_full_rt(row["name"], smiles)


def test_a_heteroatom_second_substituent_is_not_carbamimidoyl():
    # an N-amino second substituent: no N,N-disubstituted carbamimidoyl group
    row = tier_row("CC1=C(C(=NO1)C2=C(C=CC=C2Cl)Cl)C(=N)N(C)NC3=CC=CC=C3", "best-effort")
    assert "carbamimidoyl" not in row["name"], row
