from orthonym import name_compound
from rdkit import Chem
from rdkit.Chem import inchi

def _rt(name):
    from orthonym.namer import _validity_gate_name_to_smiles
    smi = _validity_gate_name_to_smiles(name)
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smi)) if smi else None

def test_diazanium_dichloride_is_never_named_as_neutral_diamine():
    # CHEBI:53452 — +2 di-azanium dichloride. The old bug shipped
    # 'N-2-aminoethylnaphthalen-1-amine dichloride' (neutral diamine, 2 H+ dropped).
    smi = "[Cl-].[Cl-].[NH3+]CC[NH2+]c1cccc2ccccc12"
    name = name_compound(smi, style="pin")
    target = inchi.MolToInchiKey(Chem.MolFromSmiles(smi))
    # 0-wrong: either abstain, or a name that round-trips to the SAME molecule.
    if name and "unknown" not in name:
        assert _rt(name) == target, f"WRONG-MOLECULE: {name!r} -> {_rt(name)} != {target}"

def test_homogeneous_poly_aminium_still_names():
    # regression guard: ethane-1,2-bis(aminium) must still work (route_charged path).
    from orthonym.rules.ions import name_cation
    m = Chem.MolFromSmiles("[NH3+]CC[NH3+]")
    assert name_cation(m, "pin")  # non-empty
