import pytest
from rdkit import Chem
from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.jvm_bridge import opsin_stdout
from orthonym.validation.opsin_roundtrip import _find_opsin_jar


def _radical_rt(name: str, target_smiles: str) -> bool:
    """OPSIN round-trip WITH allowRadicals (opsin_roundtrip_check is radical-blind)."""
    jar = _find_opsin_jar("2.9.0")
    txt, _ = opsin_stdout(name, allow_radicals=True, jar_path=jar)
    if not txt:
        return False
    smi = txt.strip().split("\n")[-1].strip()
    if not smi or "could not" in smi.lower():
        return False
    m, t = Chem.MolFromSmiles(smi), Chem.MolFromSmiles(target_smiles)
    if m is None or t is None:
        return False
    return Chem.InchiToInchiKey(Chem.MolToInchi(m)) == Chem.InchiToInchiKey(Chem.MolToInchi(t))


def _name_be(smiles: str):
    with jvm_slots(1, purpose="test-radical"):
        r = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    return r.get("name")


# distinct substitution patterns -> generalisation, not a special case
SUBSTITUTED_ARYLOXYL = [
    "[O]c1ccc(O)cc1",      # 4-hydroxyphenoxyl
    "[O]c1ccc(Cl)cc1",     # 4-chlorophenoxyl
    "[O]c1ccccc1C",        # 2-methylphenoxyl
]


@pytest.mark.parametrize("smiles", SUBSTITUTED_ARYLOXYL)
def test_substituted_aryloxyl_named_and_rt(smiles):
    name = _name_be(smiles)
    assert name and name not in ("unknown organic compound", None), f"abstained on {smiles}"
    assert _radical_rt(name, smiles), f"{name!r} did not -r round-trip for {smiles}"


def test_unsubstituted_phenoxyl_pin_unchanged():
    assert _name_be("[O]c1ccccc1") == "phenoxyl"


def test_unsubstituted_methoxyl_pin_unchanged():
    assert _name_be("C[O]") == "methoxyl"
