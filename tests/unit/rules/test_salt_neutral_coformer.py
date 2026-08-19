"""v33 Phase 4 Lever A2: fold a recognized neutral solvate (water of
crystallization) into an ionic-salt name instead of abstaining.

`name_salt` used to fail-closed unconditionally on ANY neutral co-fragment
(`if neutrals: return ''`). A salt with water of crystallization -- e.g.
cetylpyridinium chloride monohydrate (CHEBI:3566) -- therefore abstained
even though the ionic part names fine. This relaxes the guard to fold a
recognized water solvate as a '<salt> <mult>hydrate' suffix (P-16.4 /
P-14.8.2 general nomenclature) while keeping the hard fail-closed for any
UNRECOGNIZED neutral co-former (0-wrong).
"""
from orthonym import name_compound
from rdkit import Chem
from rdkit.Chem import inchi


def _rt(name):
    from orthonym.namer import _validity_gate_name_to_smiles
    smi = _validity_gate_name_to_smiles(name)
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smi)) if smi else None


def test_ionic_salt_plus_water_names_as_hydrate():
    # CHEBI:3566 -- 1-hexadecylpyridin-1-ium chloride + water of crystallization.
    smi = "CCCCCCCCCCCCCCCC[n+]1ccccc1.O.[Cl-]"
    name = name_compound(smi, style="pin")
    assert name and "unknown" not in name
    assert _rt(name) == inchi.MolToInchiKey(Chem.MolFromSmiles(smi))


def test_pure_salt_without_neutral_unchanged():
    # regression: a plain salt with no neutral co-former still names.
    assert name_compound("[Na+].CC(=O)[O-]", style="pin") == "sodium acetate"


def test_unrecognized_neutral_coformer_still_abstains():
    # 0-wrong: an unrecognized neutral organic co-former (ethanol, not water)
    # must NOT be silently dropped or forced into a hydrate-shaped name --
    # name_salt has no mechanism to fold it, so it must fail closed.
    smi = "[Na+].CC(=O)[O-].CCO"
    assert name_salt_abstains(smi)


def name_salt_abstains(smi: str) -> bool:
    from orthonym.rules.salts import name_salt
    mol = Chem.MolFromSmiles(smi)
    return name_salt(mol, "pin") == ""
