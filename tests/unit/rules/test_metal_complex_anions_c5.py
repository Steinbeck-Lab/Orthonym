"""Metal complexes of the ChEBI lane C5 (2026-11-20): cyano complex anions, Group 2 covalent salts, and the
metals admitted to bonded names that had no metal name.

- A complex anion names its ligands as any bonded complex does, with the charge number of the complex
  ('tetracyanonickel(2-)', as 'trimethylzinc(1-)' in ``test_group12_inorganic_ligands``): Definitions of
  terms (the Blue Book), "names for substituent groups as defined in Section can also be used as ligand
  names", and 'pentaammine(ethyl)osmium(1+)' for the charge number. OPSIN 2.9.0 reads neither 'cyanido'
  nor an '-ate' name of a transition metal, so this form is the one that reads back. The metal's oxidation number
  as drawn must be one of its charges in ``data/metal_scope.py``: tetracyanonickel(4-) (nickel(0)) declines.
- Calcium, strontium, barium and beryllium are admitted to ``DISCONNECT``, as Group 1 and magnesium were: a covalent
  drawing of a salt is named through its ionic drawing ('calcium dipropanoate'), labelled below the PIN because the
  drawing as written is the structure to name.
- Silver, gold, niobium and tantalum get their metal names ('methylgold', 'dicyanogold(1-)').

No name here is a PIN: "Coordination nomenclature is the primary nomenclature method used to name organometallic
compounds containing elements of Groups 3 through 12", and gives the PIN label only to the
elements it lists. Every name is read back by a fresh OPSIN call, to the full standard InChIKey of the input.
"""
import pytest

from orthonym.data.metal_scope import DISCONNECT, has_mechanism
from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

NAMED = [
    # complex anions with cyano ligands (ChEBI: CHEBI:49928, CHEBI:30331, CHEBI:30639, CHEBI:49491, CHEBI:30058)
    ("N#[C][Ni-2]([C]#N)([C]#N)[C]#N", "tetracyanonickel(2-)"),
    ("N#[C][Co-3]([C]#N)([C]#N)([C]#N)([C]#N)[C]#N", "hexacyanocobalt(3-)"),
    ("N#[C][Mo-3]([C]#N)([C]#N)([C]#N)([C]#N)[C]#N", "hexacyanomolybdenum(3-)"),
    ("N#[C][Au-][C]#N", "dicyanogold(1-)"),
    ("N#[C][Au-][C]#N.[Na+]", "dicyanogold(1-)—sodium(1+) (1/1)"),
    # covalent drawings of Group 2 salts (ChEBI: CHEBI:81716)
    ("CCC(=O)[O][Ca][O]C(=O)CC", "calcium dipropanoate"),
    ("CC(=O)O[Sr]OC(C)=O", "strontium diacetate"),
    ("CC(=O)O[Ba]OC(C)=O", "barium diacetate"),
    # organyls of gold and silver
    ("C[Au](C)C", "trimethylgold"),
    ("C[Ag]", "methylsilver"),
]


@pytest.mark.parametrize("smiles,name", NAMED)
def test_named_below_the_pin(smiles, name):
    row = tier_row(smiles, "best-effort")
    assert row["name"] == name, row
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row
    assert_full_rt(row["name"], smiles)
    assert tier_row(smiles, "pin")["tier"] == "abstain"


def test_an_oxidation_number_outside_the_table_declines():
    """[Ni(CN)4]4- is nickel(0); the table gives nickel the charges 2, 3 and 4, and a standard InChIKey does not bound
    the charge of a metal, so no complex name is built."""
    assert tier_row("N#[C][Ni-4]([C]#N)([C]#N)[C]#N", "best-effort")["tier"] == "abstain"


def test_group_2_disconnect_admissions():
    assert all(has_mechanism(s, DISCONNECT) for s in ("Be", "Mg", "Ca", "Sr", "Ba", "Ra"))


# ChEBI lane C1 (2026-11-20): carbon-free rows with a name OPSIN reads. Every name is carbon-free, so a preselected
# name at most,, never a PIN.
CARBON_FREE = [
    # the oxide ion as a ligand: "an anion name ending in 'ide' ends in 'ido' as a ligand"
    ("[O]=[V](=[O])[O-]", "oxidodioxovanadium"),                     # CHEBI:75224
    ("[O]=[Fe][O-]", "oxido(oxo)iron"),                              # CHEBI:78619
    ("[Na+].[Na+].[Na+].[O]=[V]([O-])([O-])[O-]", "trisodium trioxido(oxo)vanadium"),     # CHEBI:35607
    ("[Na+].[Na+].[O]=[Mo](=[O])([O-])[O-]", "disodium dioxidodioxomolybdenum"),          # CHEBI:75215
    # radium joins DISCONNECT; gold(V), silver(III), chromium(V) join the table of charges
    ("[Cl][Ra][Cl]", "radium dichloride"),                           # CHEBI:754767
    ("[F][Au]([F])([F])([F])[F]", "gold(V) fluoride"),               # CHEBI:30080
    ("[F][Cr]([F])([F])([F])[F]", "chromium(V) fluoride"),           # CHEBI:33041
    # a complex anion with a name of its own keeps it: no ligand name is built for it
    ("[O-][Mn](=O)(=O)=O", "permanganate"),
    ("O=[Cr](=O)([O-])[O-]", "chromate"),
    ("N#C[Fe-4](C#N)(C#N)(C#N)(C#N)C#N", "ferrocyanide"),
]


@pytest.mark.parametrize("smiles,name", CARBON_FREE)
def test_carbon_free_rows(smiles, name):
    row = tier_row(smiles, "best-effort")
    assert row["name"] == name, row
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row
    assert_full_rt(row["name"], smiles)
