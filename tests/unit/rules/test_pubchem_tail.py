"""The PubChem tail lane: λ names of radical components, bracketed 'azanidyl' prefixes, hydrido ligands.

* "The λ-convention" (the Blue Book): a divalent or trivalent radical centre "may be described by the
  λ-convention... only for general nomenclature" ('dichloro-λ2-methane'). OPSIN 2.9.0 reads no radical component of
  an adduct name, and reads its λ name as an ordinary one (``rules.lambda_component``).
* (:7232): a compound prefix is enclosed, '(tert-butylazanidyl)' as '(tert-butylazaniumyl)'.
* A hydrogen on the metal of a halide component is the 'hydrido' ligand ('chlorohydridoruthenium').
Adduct names are no PIN,:4667).
"""
import pytest
from rdkit import Chem

from tests.support.rt_assert import name_best_effort, name_is_rt_exact


def _radical_adducts_read():
    from orthonym.opsin_capabilities import radical_adduct_enabled
    return radical_adduct_enabled()


@pytest.mark.parametrize("smiles,name", [
    ("C[Si]C", "dimethyl-λ2-silane"),
    ("[SnH2]", "λ2-stannane"),
    ("C[B]", "methyl-λ1-borane"),
    ("C[Si]c1ccccc1", "methyl(phenyl)-λ2-silane"),
    ("O=[Si]", "oxo-λ2-silane"),
    ("Cl[Sn]Cl", "dichloro-λ2-stannane"),
    ("C[CH2]", None),        # a monovalent radical: no λ
    ("[Pb]", None),          # a bare atom is the element
])
def test_the_lambda_name_of_a_radical_component(smiles, name):
    from orthonym.metrics.provenance import best_effort_ctx
    from orthonym.rules.lambda_component import lambda_component_name
    token = best_effort_ctx.set(True)
    try:
        assert lambda_component_name(smiles) == name
    finally:
        best_effort_ctx.reset(token)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("C[Si]C.O", "dimethyl-λ2-silane—water (1/1)"),
    ("[SnH2].O", "λ2-stannane—water (1/1)"),
    ("[PbH2].[Pb]", "λ2-plumbane—lead (1/1)"),
    ("C[B].O", "methyl-λ1-borane—water (1/1)"),
])
def test_a_radical_component_takes_its_lambda_name_where_opsin_reads_no_radical_adduct(smiles, name):
    if _radical_adducts_read():
        pytest.skip("the OPSIN in use reads radical adduct components: the radical name ships")
    r = name_best_effort(smiles)
    assert r["name"] == name and not r["is_pin"], r
    assert name_is_rt_exact(name, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("CC(C)(C)[N-]C([N-]C(C)(C)C)c1ccc([N+](=O)[O-])cc1", "1-[bis(tert-butylazanidyl)methyl]-4-nitrobenzene"),
])
def test_the_enclosed_azanidyl_prefix(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "systematic_verified", r
    assert name_is_rt_exact(name, smiles)


def test_an_anionic_hydride_yl_makes_a_compound_prefix():
    from orthonym.assembly.naming_utils import enclose_if_compound
    assert enclose_if_compound("tert-butylazanidyl") == "(tert-butylazanidyl)"
    assert enclose_if_compound("azanidyl") == "azanidyl"


def test_a_hydrido_ligand_beside_a_halide_but_no_metal_hydride_alone():
    from orthonym.rules.adducts import _metal_halide_oxide_component_name as f
    assert f(Chem.MolFromSmiles("[Cl][RuH]")) == "chlorohydridoruthenium"
    assert f(Chem.MolFromSmiles("Cl[Pd]Cl")) == "dichloropalladium"
    assert f(Chem.MolFromSmiles("[RuH2]")) is None
    assert f(Chem.MolFromSmiles("O=[MoH2]")) is None   # an oxo hydride keeps the pipeline's 'dihydrido(oxo)molybdenum'


def test_with_opsin_main_lambda_spellings_only_beside_a_second_radical(monkeypatch):
    """One radical component is read last on OPSIN main and keeps its radical name; two need a λ name for one."""
    import orthonym.opsin_capabilities as caps
    from orthonym.rules.adducts import _lambda_spellings_allowed as allowed
    monkeypatch.setattr(caps, "radical_adduct_enabled", lambda: True)
    assert not allowed([("O", 1), ("C[Si]C", 1)])
    assert allowed([("C[Si]C", 1), ("C[Ge]C", 1)])
    monkeypatch.setattr(caps, "radical_adduct_enabled", lambda: False)
    assert allowed([("O", 1), ("C[Si]C", 1)])
