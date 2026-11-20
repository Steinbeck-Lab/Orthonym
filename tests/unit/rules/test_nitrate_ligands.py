"""Esters of nitric and nitrous acid as ligands and prefixes (2026-11-20).

 "Substituent groups derived from esters of nitric acid and nitrous acid" (the Blue Book): they "are
named by concatenation by adding the acyl groups 'nitro' for -NO2 and 'nitroso' for -NO to the prefix 'oxy'";
'-O-NO2 nitrooxy (preselected prefix)' (:36405). The substituent namer spelled the -O-NO2 branch 'nitroyloxy', which no
reader reads, and inside a naming call its capped piece got no name at all, so ChEBI's phenylmercury nitrate had no
name. The ligand names are no PIN: coordination nomenclature for Groups 3 through 12.
"""
import pytest
from rdkit import Chem

from orthonym.assembly.sulfur_acid_ester_prefixes import nitrogen_acid_oxy_prefix
from tests.support.pin_tiers import tier_row
from tests.support.rt_assert import assert_full_rt

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]


@pytest.mark.parametrize("smiles,name", [
    ("O=[N+]([O-])O[Hg]c1ccccc1", "(nitrooxy)(phenyl)mercury"),     # CHEBI:136021
    ("O=[N+]([O-])O[Hg]C", "methyl(nitrooxy)mercury"),
    ("O=NO[Hg]C", "methyl(nitrosooxy)mercury"),
])
def test_nitrate_and_nitrite_ligands(smiles, name):
    row = tier_row(smiles, "best-effort")
    assert row["name"] == name, row
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row
    assert_full_rt(row["name"], smiles)


@pytest.mark.parametrize("smiles,frag,attach,prefix", [
    ("O=[N+]([O-])OCCC", {0, 1, 2, 3}, 3, "nitrooxy"),
    ("O=NOCCC", {0, 1, 2}, 2, "nitrosooxy"),
    ("O=[N+]([O-])SCCC", {0, 1, 2, 3}, 3, None),        # the S analogue is another prefix ('nitrosulfanyl')
    ("ON(O)OCCC", {0, 1, 2, 3}, 3, None),                # no N=O: no ester of nitric or nitrous acid
])
def test_the_structural_rule(smiles, frag, attach, prefix):
    assert nitrogen_acid_oxy_prefix(Chem.MolFromSmiles(smiles), frag, attach) == prefix


def test_a_nitro_group_in_a_sigma_ligand():
    """A nitro group, always drawn N+(=O)O-, is a neutral part of a sigma ligand (rules.organometallics._in_nitro_group).
    The 'chlorido' name is read back through its 'chloro' twin (validation.halido_twin), so the label is the check."""
    row = tier_row("Cl[Ti](Cl)(Cl)c1ccc([N+](=O)[O-])cc1", "best-effort")
    assert row["name"] == "trichlorido(4-nitrophenyl)titanium", row
    assert row["tier"] == "systematic_verified" and row["is_pin"] is False, row

