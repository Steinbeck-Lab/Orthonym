from orthonym import Orthonym

BE = dict(general_fallback=True, allow_aromatic_general=True, general_fallback_unverified=True)

def _rt_inchikey(name):
    from orthonym.namer import _validity_gate_name_to_smiles
    from rdkit import Chem
    smi = _validity_gate_name_to_smiles(name)
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smi)) if smi else None

def test_buspirone_hcl_names_and_roundtrips_at_best_effort():
    # CHEBI:3224 — parent nameable only at T4; adduct must emit + RT.
    from rdkit import Chem
    smi = "Cl.O=C1CC2(CCCC2)CC(=O)N1CCCCN1CCN(c2ncccn2)CC1"
    name = Orthonym(style="pin", **BE).name(smi)
    assert name and "unknown" not in name
    assert " hydrochloride" in name or name.endswith("(1/1)")  # general or em-dash adduct
    assert _rt_inchikey(name) == Chem.MolToInchiKey(Chem.MolFromSmiles(smi))

def test_name_adduct_accepts_gfu_kwarg():
    from orthonym.rules import adducts
    from rdkit import Chem
    m = Chem.MolFromSmiles("Cl.O=C1CC2(CCCC2)CC(=O)N1CCCCN1CCN(c2ncccn2)CC1")
    out = adducts.name_adduct(m, Chem.MolToSmiles(m), style="pin",
                              general_fallback=True, allow_aromatic_general=True,
                              general_fallback_unverified=True)
    assert out is not None
