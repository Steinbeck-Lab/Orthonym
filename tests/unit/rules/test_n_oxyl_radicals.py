"""N-oxyl (aminoxyl) radicals R2N-O. --.

the Blue Book (:40677-40705): the radical is named "(1) additively,
using the term 'oxyl'", and "Method (1) generates preferred IUPAC names";
'aminoxyl' is retained (:40681,:40699) and its substituted form is the PIN
'(ClCH2)2N-O. bis(chloromethyl)aminoxyl (PIN)' (:40703).

Every expected name is checked two ways: the exact string, and a strict OPSIN
-r round trip (canonical SMILES with the radical dot), because the full
InChIKey alone cannot separate a radical from a closed-shell twin.
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.jvm_bridge import opsin_stdout
from tests.support.jars import jar_or_skip


def _strict_rt(name: str, smiles: str) -> bool:
    jar = jar_or_skip()
    txt, _ = opsin_stdout(name, allow_radicals=True, jar_path=jar)
    parsed = Chem.MolFromSmiles((txt or "").strip())
    return parsed is not None and Chem.MolToSmiles(parsed) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def _name(smiles: str, tier: str = "pin") -> dict:
    with jvm_slots(1, purpose="test-n-oxyl"):
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


N_OXYL = [
    ("N[O]", "aminoxyl"),                                    # BB:40699 (preselected)
    ("ClCN(CCl)[O]", "bis(chloromethyl)aminoxyl"),           # BB:40703 (PIN)
    ("CN(C)[O]", "dimethylaminoxyl"),
    ("CN[O]", "methylaminoxyl"),
    ("CC(C)(C)N([O])C(C)(C)C", "di-tert-butylaminoxyl"),
    ("[O]N(c1ccccc1)c1ccccc1", "diphenylaminoxyl"),
    ("CN([O])c1ccccc1", "methyl(phenyl)aminoxyl"),
    ("CC1(C)CCCC(C)(C)N1[O]", "(2,2,6,6-tetramethylpiperidin-1-yl)oxyl"),
    ("CC1(C)CC(O)CC(C)(C)N1[O]", "(4-hydroxy-2,2,6,6-tetramethylpiperidin-1-yl)oxyl"),
    ("CC1(C)CC(=O)CC(C)(C)N1[O]", "(2,2,6,6-tetramethyl-4-oxopiperidin-1-yl)oxyl"),
    ("CC1(C)CC(N)(C(=O)O)CC(C)(C)N1[O]", "(4-amino-4-carboxy-2,2,6,6-tetramethylpiperidin-1-yl)oxyl"),
    ("O=C1c2ccccc2C(=O)N1[O]", "(1,3-dioxo-1,3-dihydro-2H-isoindol-2-yl)oxyl"),
    ("O=C1CCC(=O)N1[O]", "(2,5-dioxopyrrolidin-1-yl)oxyl"),
    ("[O]N1CCOCC1", "(morpholin-4-yl)oxyl"),
    ("[O]N1CCCCC1", "(piperidin-1-yl)oxyl"),
    ("O=C1CCCN1[O]", "(2-oxopyrrolidin-1-yl)oxyl"),
]


@pytest.mark.parametrize("smiles,expected", N_OXYL)
def test_n_oxyl_named_at_pin_tier(smiles, expected):
    row = _name(smiles, "pin")
    assert row["name"] == expected
    assert row["tier"] != "abstain"
    assert _strict_rt(expected, smiles)


@pytest.mark.parametrize("smiles,expected", N_OXYL)
def test_n_oxyl_best_effort_same_name(smiles, expected):
    assert _name(smiles, "best-effort")["name"] == expected


# The closed-shell and charge-separated twins keep their own names: the new
# producer runs only on a radical oxygen.
TWINS = [
    ("CC1(C)CCCC(C)(C)N1O", "2,2,6,6-tetramethylpiperidin-1-ol"),
    ("C[N+](C)(C)[O-]", "N,N-dimethylmethanamine N-oxide"),
]


@pytest.mark.parametrize("smiles,expected", TWINS)
def test_twins_unchanged(smiles, expected):
    assert _name(smiles, "pin")["name"] == expected


@pytest.mark.opsin_gate
def test_acyl_n_oxyl_is_not_built_here():
    # A hydroxamic-acid radical needs the amide rules; the producer declines.
    # With the OPSIN gate on (every real deployment) the PIN tier abstains
    # rather than guessing, and the best-effort tier still names it.
    assert _name("CC(=O)N([O])C", "pin")["tier"] == "abstain"
    be = _name("CC(=O)N([O])C", "best-effort")
    assert be["tier"] != "abstain" and _strict_rt(be["name"], "CC(=O)N([O])C")


@pytest.mark.parametrize("prefixes,expected", [
    (["methyl", "methyl"], "dimethyl"),
    (["chloromethyl", "chloromethyl"], "bis(chloromethyl)"),
    (["tert-butyl", "tert-butyl"], "di-tert-butyl"),
    (["phenyl", "methyl"], "methyl(phenyl)"),
    (["methyl", "chloromethyl"], "(chloromethyl)(methyl)"),
    (["methyl"], "methyl"),
])
def test_aminoxyl_prefix_string(prefixes, expected):
    from orthonym.rules.radicals import _aminoxyl_prefix_string
    assert _aminoxyl_prefix_string(prefixes) == expected
