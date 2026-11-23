"""An anionic nitrogen as an 'azanidyl' prefix in the universal floor (best-effort only).

 "Anionic centers in both parent compounds and substituent groups" (the Blue Book): "one anion must be
chosen as the parent anion and the other expressed as anionic substituent group(s)"; "Systematically formed
prefixes that include anionic center(s)" (:41227): '-NH- azanidyl (preselected prefix)', '=N- azanidylidene
(preselected prefix)'. The floor names a polyanion whose centres no anion producer joins (an N- beside an O-, an
amidinate, a silylamide) with those prefixes, in a last rung of its ladder (``_azanidyl_rung``), and ships the name
only when OPSIN reads every charge back on its drawn atom. These names are below the PIN chooses the parent
of a PIN), and an anion that a producer names keeps its PIN.
"""
import pytest
from rdkit import Chem

from tests.support.rt_assert import name_best_effort, name_is_rt_exact


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("C[N-]CC[O-]", "2-(methylazanidyl)-1-oxidoethane"),
    ("[NH-]CC1CCC1C[NH-]", "1,2-bis(azanidylmethyl)cyclobutane"),
    ("CC(C)N=C([N-]C(C)C)[N-]C(C)C", "2-({bis[(propan-2-yl)azanidyl]methylidene}amino)propane"),
    ("C[Si](C)(C)[N-]c1ccccc1[N-][Si](C)(C)C", "1,2-bis[(trimethylsilyl)azanidyl]benzene"),
    ("CCCCC(=[N-])[NH-]", "1-azanidyl-1-azanidylidenepentane"),
])
def test_a_polyanion_with_an_anionic_nitrogen_is_named_by_the_floor(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "systematic_verified" and not r["is_pin"], r
    assert name_is_rt_exact(name, smiles)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,name", [
    ("CC[N-]CC", "N-ethylethanaminide"),
    ("C[N-]CC[N-]C", "N1,N2-dimethylethane-1,2-bis(aminide)"),
    ("CC(=O)[N-]C", "acetyl(methyl)azanide"),
])
def test_an_anion_a_producer_names_keeps_its_pin(smiles, name):
    r = name_best_effort(smiles)
    assert r["name"] == name and r["tier"] == "pin_verified", r


def test_the_neutral_floor_and_the_prefix_entry_never_spell_azanidyl():
    from orthonym.assembly.universal_substituent import (
        name_universal_substituent_prefix, name_universal_substitutive)
    assert name_universal_substitutive(Chem.MolFromSmiles("CNCCO")).name == "1-hydroxy-2-(methylamino)ethane"
    prefix = name_universal_substituent_prefix(Chem.MolFromSmiles("C[N-]CCO"), [0, 1], 2)
    assert prefix is None or "azanidyl" not in prefix


def test_a_reading_with_the_charge_on_another_nitrogen_is_refused(monkeypatch):
    """The two amide anions share one standard InChIKey; the rung's read-back sees where the charge sits."""
    from orthonym.assembly import universal_substituent as U
    import orthonym.namer as N
    drawn, shifted = "CC(=O)[N-]CC(=O)NC", "CC(=O)NCC(=O)[N-]C"
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(drawn)) == Chem.MolToInchiKey(Chem.MolFromSmiles(shifted))
    monkeypatch.setattr(N, "_validity_gate_name_to_smiles", lambda name: shifted)
    assert not U._reads_back_with_drawn_charges("x", Chem.MolFromSmiles(drawn))
    monkeypatch.setattr(N, "_validity_gate_name_to_smiles", lambda name: drawn)
    assert U._reads_back_with_drawn_charges("x", Chem.MolFromSmiles(drawn))
