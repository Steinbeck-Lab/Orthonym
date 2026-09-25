"""Chalcogen radicals and ring-N / aminyl radicals,.

Each name must match exactly and pass a strict OPSIN -r round trip (canonical
SMILES, radical dots included).
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
    with jvm_slots(1, purpose="test-radical-b3b4"):
        return Orthonym(style="pin", **_emit_tier_flags(tier)).name_tiered(smiles)


PIN_ROWS = [
    # (the Blue Book): "named on the basis of preselected parent radical
    # names, such as 'sulfanyl', 'selanyl', 'disulfanyl'".
    ("[S]c1ccccc1", "phenylsulfanyl"),                  # (PIN):40727
    ("C[Se]", "methylselanyl"),                          # (PIN):40733
    ("CC(C)(C)S[S]", "tert-butyldisulfanyl"),            # (PIN):40737
    ("ClCC(=S)[S]", "(chloroethanethioyl)sulfanyl"),     # BB example
    ("C[S]", "methylsulfanyl"),
    ("CS[S]", "methyldisulfanyl"),
    ("[S]c1ccc[nH]1", "(1H-pyrrol-2-yl)sulfanyl"),
    # (:40632): 'benzenaminyl (PIN)... (not anilino)'.
    ("[NH]c1ccccc1", "benzenaminyl"),
    ("c1ccc(cc1)[N]c1ccccc1", "N-phenylbenzenaminyl"),
    #: a ring-N radical is the ring's N-yl group; '2,5-dioxopyrrolidin-1-yl (PIN)' (:40645).
    ("O=C1CCC(=O)[N]1", "2,5-dioxopyrrolidin-1-yl"),
    ("C1CC[N]CC1", "piperidin-1-yl"),
    ("C1CC[N]C1", "pyrrolidin-1-yl"),
    ("O=C1c2ccccc2C(=O)[N]1", "1,3-dioxo-1,3-dihydro-2H-isoindol-2-yl"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", PIN_ROWS)
def test_pin_tier(smiles, expected):
    row = _name(smiles)
    assert row["name"] == expected and row["tier"] != "abstain"
    assert _strict_rt(expected, smiles)


@pytest.mark.opsin_gate
def test_prefix_without_indicated_hydrogen_is_not_a_pin():
    # The shared prefix namer gives 'pyrrol-1-yl'; the PIN form keeps indicated
    # hydrogen ('(1H-indol-1-yl)acetic acid (PIN)'). The PIN tier abstains and the
    # best-effort tier ships it with a non-PIN label.
    assert _name("[N]1C=CC=C1")["tier"] == "abstain"
    be = _name("[N]1C=CC=C1", "best-effort")
    assert be["tier"] not in ("abstain", "pin_verified")
    assert _strict_rt(be["name"], "[N]1C=CC=C1")
