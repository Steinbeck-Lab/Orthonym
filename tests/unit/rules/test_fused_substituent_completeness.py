"""A fused-heterocycle catalog name never drops a substituent it cannot name
(pre-existing-failures plan, Task 4, TRIAGE row 88).

`rules/fused_rings.py::get_fused_heterocycle_substituents` skips a branch that
`_identify_fused_substituent` cannot name (`sub_info is None -> continue`). The
completeness check `_exocyclic_atoms_accounted` used to run only at the general
tiers, so with the OPSIN validity gate off the PIN tier shipped the bare core:
'5-hydroxy-2,3-dihydro-1H-isoindole-1,3-dione' (C8H5NO3) for
5-hydroxythalidomide (C13H10N2O5), and 'quinoline' for 6-(methanesulfonyl)- and
6-silylquinoline (the check's own documented witnesses).

 "SUBSTITUTIVE NOMENCLATURE" (the Blue Book): the parent's
substitutable hydrogen atoms "are substituted by nomenclaturally significant
structural fragments represented either by prefixes and/or suffixes"; a branch
with no prefix is not in the name. The PIN tier now fails closed; best-effort
names all three RT-exact (full InChIKey). These run with the gate OFF (the raw
producer).
"""
import pytest

from tests.support.rt_assert import assert_tier_contract

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("smiles", [
    "O=C1CCC(N2C(=O)c3ccc(O)cc3C2=O)C(=O)N1",   # row 88
    "CS(=O)(=O)c1ccc2ncccc2c1",
    "[SiH3]c1ccc2ncccc2c1",
])
def test_fused_catalog_name_keeps_every_substituent(smiles):
    assert_tier_contract(smiles)


@pytest.mark.parametrize("smiles,expected", [
    # controls: every branch nameable at the PIN tier -> unchanged PIN names
    # 2026-09-25 (pre-existing-failures plan, Task 5) change-asserted-value: R21: (the Blue Book) "Cyclic imides are preferably named as heterocyclic pseudoketones"; "2-phenyl-1H-isoindole-1,3(2H)-dione (PIN)... N-phenylphthalimide" (:33853); (:24689) added indicated hydrogen is preferred over hydro prefixes for PINs.
    ("O=C1c2ccc(O)cc2C(=O)N1C", "5-hydroxy-2-methyl-1H-isoindole-1,3(2H)-dione"),
    ("OB(O)c1ccc2ncccc2c1", "(quinolin-6-yl)boronic acid"),
    ("Cc1ccc2[nH]ccc2c1", "5-methyl-1H-indole"),
])
def test_nameable_branches_unchanged(smiles, expected):
    from orthonym import name_compound
    assert name_compound(smiles) == expected
