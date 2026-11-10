"""Salts of the anions of the noncarbon oxoacids of / and of phosphinic, phosphorous and arsorous acid,
and of the oxoanions of Mn, Cr and Re (program 22, 2026-11-10).

 'CLASS 7e. OTHER MONOBASIC 'OXO' ACIDS USED AS FUNCTIONAL PARENTS': "All these names are preselected names"
(the Blue Book), 'hypochlorous acid ClOH', 'bromous acid Br(O)OH' (:18460);: 'boric acid B(OH)3',
'disulfurous acid (HO)S(O)-O-S(O)(OH)', 'dithionous acid (hypodisulfurous acid) (HO)(O)S-S(O)(OH)' (:18427-18439).
The anion of an '-ous acid' ends in '-ite', of an '-ic acid' in '-ate' Anions derived from acids,
:40953). A name without carbon is a preselected name at most, never a PIN (the contributor guide, scope); a metal-centred anion
gives no PIN either.
"""
import pytest

from orthonym import Orthonym

pytestmark = pytest.mark.opsin_gate


@pytest.fixture(scope="module")
def best_effort():
    return Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)


@pytest.mark.parametrize("smiles, name", [
    ("[Na+].[O-]Cl", "sodium hypochlorite"),
    ("[Ca+2].[O-]Cl.[O-]Cl", "calcium dihypochlorite"),
    ("[Na+].[O-]Br", "sodium hypobromite"),
    ("[K+].[O-]Br=O", "potassium bromite"),
    ("[Na+].O=[PH2][O-]", "sodium phosphinate"),
    ("[Na+].[Na+].O=S([O-])OS(=O)[O-]", "disodium disulfite"),
    ("[Na+].[Na+].[Na+].[O-]B([O-])[O-]", "trisodium borate"),
    ("[K+].[O-][Mn](=O)(=O)=O", "potassium permanganate"),
    ("[K+].[K+].O=[Cr](=O)([O-])O[Cr](=O)(=O)[O-]", "dipotassium dichromate"),
    ("[K+].[O-][Re](=O)(=O)=O", "potassium perrhenate"),
    ("C[N+](C)(C)C.[O-][Mn](=O)(=O)=O", "N,N,N-trimethylmethanaminium permanganate"),
])
def test_oxoanion_salt_named_without_pin_label(best_effort, smiles, name):
    row = best_effort.name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "systematic_verified")


@pytest.mark.parametrize("smiles, name", [
    # an organic cation: the same treatment as the perchlorate already in the table
    ("C[N+](C)(C)C.[O-]Cl", "N,N,N-trimethylmethanaminium hypochlorite"),
    ("CC[NH3+].O=[PH2][O-]", "ethanaminium phosphinate"),
])
def test_organic_cation_salt_of_a_nonmetal_oxoanion(smiles, name):
    row = Orthonym().name_tiered(smiles)
    assert (row["name"], row["tier"]) == (name, "pin_verified")


@pytest.mark.parametrize("smiles", [
    "[Na+].[Na+].O=S([O-])S(=O)(=O)[O-]",   # the S-S drawing: OPSIN reads 'disulfite' as the S-O-S acid of the book
    "[K+].[O-][Tc](=O)(=O)=O",              # technetium is not admitted to salts (data/metal_scope.py)
])
def test_not_named_from_the_table(best_effort, smiles):
    row = best_effort.name_tiered(smiles)
    assert "disulfite" not in (row["name"] or "") and "pertechnetate" not in (row["name"] or ""), row
