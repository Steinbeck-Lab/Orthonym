"""Anions of the inorganic oxoacids named from their acid (ChEBI C1, 2026-11-19).

: the anion of an oxoacid is named by changing 'ic acid' to 'ate' and 'ous acid' to 'ite'; the hydroxy
hydrogen atoms that remain are cited as 'hydrogen' ('dihydrogen phosphate (preselected name)'). Orthoperiodic and
orthotelluric acid are IUPAC Red Book names (, Table.1) that OPSIN 2.9.0 reads. Every name is carbon-free:
it ships at the best-effort tier as systematic_verified, and the PIN tier declines it."""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.cli import _emit_tier_flags
from orthonym.jvm_budget import jvm_slots
from orthonym.rules.oxoacid_anions import name_oxoacid_anion
from tests.support.rt_assert import name_is_rt_exact

pytestmark = pytest.mark.opsin_gate

CASES = [
    ("[O-][As](O)O", "dihydrogen arsorite"),
    ("[O-][As]([O-])O", "hydrogen arsorite"),
    ("[H][As]([O-])[O-]", "arsonite"),              # the As-H hydrogen is not an acidic one
    ("[H][As]([O-])O", "hydrogen arsonite"),
    ("[O-][Sb]([O-])[O-]", "stiborite"),
    ("O=I([O-])([O-])([O-])([O-])[O-]", "orthoperiodate"),
    ("O=I([O-])(O)(O)(O)O", "tetrahydrogen orthoperiodate"),
    ("[O-][Te]([O-])([O-])([O-])([O-])[O-]", "orthotellurate"),
    ("O[Te]([O-])([O-])([O-])([O-])[O-]", "hydrogen orthotellurate"),
    ("O[Te](O)(O)(O)(O)O", "orthotelluric acid"),
    ("[H]P(=O)([O-])OP([H])(=O)[O-]", "diphosphonate"),
]


def _mol(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)       # RDKit refuses the heptavalent iodine when it sanitizes
    mol.UpdatePropertyCache(strict=False)
    return mol


@pytest.mark.parametrize("smiles,name", CASES + [
    ("OP(O)(=O)[O-]", "dihydrogen phosphate"),            # 'phosphor' shortens: not 'phosphorate'
    ("OS(=O)(=O)[O-]", "hydrogen sulfate"),
    ("[O-]P([O-])O", "hydrogen phosphite"),
    ("[O-][I+]([O-])(O)(O)(O)O", "tetrahydrogen orthoperiodate"),   # the charge-separated drawing of I=O
])
def test_oxoacid_anion_names(smiles, name):
    assert name_oxoacid_anion(_mol(smiles)) == name


@pytest.mark.parametrize("smiles,name", CASES)
def test_oxoacid_anion_names_ship_at_best_effort_only(smiles, name):
    with jvm_slots(1, purpose="oxoacid-anions"):
        be = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
        pin = Orthonym(style="pin", **_emit_tier_flags("pin")).name_tiered(smiles)
        assert (be["name"], be["tier"]) == (name, "systematic_verified")
        assert pin.get("tier") == "abstain"
        assert name_is_rt_exact(name, smiles)


def test_orthoperiodic_acid_keeps_its_name():
    """The neutral H5IO6 is named as it was ('pentahydroxy(oxo)-λ7-iodane'); the Red Book acid name serves its
    anions only."""
    assert name_oxoacid_anion(_mol("OI(O)(O)(O)(O)=O")) is None


@pytest.mark.parametrize("smiles", ["CC(=O)[O-]", "[O-]S(=O)(=O)C", "O=S([O-])S(=O)(=O)[O-]", "[O-][N+](=O)[O-]C"])
def test_out_of_class_declines(smiles):
    assert name_oxoacid_anion(_mol(smiles)) is None


def test_a_red_book_anion_keeps_a_salt_below_the_pin():
    """Review 2026-11-19: the salt of an organic cation and orthoperiodate was labelled pin_unverified at the valid
    tier, a PIN form the Blue Book never defines; the Red Book anion is a non-PIN part."""
    smiles = "CCCC[N+](CCCC)(CCCC)CCCC.O=I([O-])(O)(O)(O)O"
    with jvm_slots(1, purpose="oxoacid-anions"):
        row = Orthonym(style="pin", **_emit_tier_flags("valid")).name_tiered(smiles)
    assert "tetrahydrogen orthoperiodate" in (row["name"] or "")
    assert row["tier"] not in ("pin_verified", "pin_unverified") and row["is_pin"] is False
