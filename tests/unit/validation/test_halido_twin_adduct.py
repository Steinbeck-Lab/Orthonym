"""The 'chloro' twin of an organometallic component of an adduct name (``validation.halido_twin``).

 (the Blue Book) spells the halide ligand 'chlorido' ('trichlorido(methyl)titanium',:39789); OPSIN reads
only the long-established 'chloro',:39767). A drawing of a complex beside its separate ligands is an adduct
; each organometallic component is read through its twin, and the read must reproduce the drawn charges.
"""
import pytest

from orthonym.validation.halido_twin import halido_twin, note_bonded_halido_name
from tests.support.rt_assert import name_best_effort

pytestmark = pytest.mark.opsin_gate


def test_the_twin_rewrites_each_organometallic_component():
    note_bonded_halido_name("benzylidenedichloridoruthenium")
    assert (halido_twin("tricyclohexylphosphane—benzylidenedichloridoruthenium (2/1)")
            == "tricyclohexylphosphane—benzylidenedichlororuthenium (2/1)")
    assert halido_twin("sodium chloride") == "sodium chloride"


@pytest.mark.parametrize("smiles,name", [
    ("C1CCC(CC1)P(C1CCCCC1)C1CCCCC1.C1CCC(CC1)P(C1CCCCC1)C1CCCCC1.Cl[Ru](Cl)=Cc1ccccc1",
     "tricyclohexylphosphane—benzylidenedichloridoruthenium (2/1)"),
    # the ionic drawing beside a solvent gets the name of the bonded drawing, as the same drawing alone does
    # ('trichlorido(methyl)titanium'; a review ruling 2026-11-23: one name for both drawings of one compound)
    ("C[Ti+3].[Cl-].[Cl-].[Cl-].C1CCOC1", "oxolane—trichlorido(methyl)titanium (1/1)"),
    ("C[Ti](Cl)(Cl)Cl.C1CCOC1", "oxolane—trichlorido(methyl)titanium (1/1)"),
])
def test_a_complex_beside_its_ligands_is_named_below_the_pin(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "systematic_verified" and not r["is_pin"], r
