import pytest
from rdkit import Chem
from rdkit.Chem import inchi
from orthonym import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse
from orthonym.assembly.composer import _build_bicyclo_substituent_prefix
from orthonym.rules.bicyclo import get_bicyclo_ring_atoms, get_bicyclo_substituents


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def _full_rt(smiles: str, name: str) -> bool:
    o = opsin_parse(name)
    if not o:
        return False
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles)) == \
        inchi.MolToInchiKey(Chem.MolFromSmiles(o))


def test_build_bicyclo_substituent_prefix_isopropyl_not_n_propyl():
    # Function-level fail-first oracle (ledger ruling R3): namer.name() abstains
    # at HEAD (SELF-01 hides the wrong candidate), so a name-level test alone is
    # not fail-first. Call the composer helper directly on the isopropyl
    # fragment and assert it never returns the n-propyl ('2-propyl') form.
    smi = "CC(C)C1CC2CCC1C2"
    mol = Chem.MolFromSmiles(smi)
    ring_atoms = get_bicyclo_ring_atoms(mol)
    substituents = get_bicyclo_substituents(mol, ring_atoms)

    # Build a minimal atom_to_locant map: ring atoms get sequential locants
    # (exact locant values do not matter for this assertion -- only the
    # substituent NAME token does).
    atom_to_locant = {idx: i + 1 for i, idx in enumerate(sorted(ring_atoms))}

    prefix = _build_bicyclo_substituent_prefix(mol, substituents, atom_to_locant)
    assert prefix is not None
    assert "2-propyl" not in prefix, prefix
    assert "propan-2-yl" in prefix, prefix


@pytest.mark.opsin_gate
def test_isopropyl_norbornane_not_misnamed(namer):
    smi = "CC(C)C1CC2CCC1C2"
    name = namer.name(smi)
    # must never emit the n-propyl form; either the correct propan-2-yl name or abstain.
    assert "2-propyl" not in (name or "")
    if name and "unknown" not in name:
        assert _full_rt(smi, name), name


@pytest.mark.opsin_gate
def test_methyl_norbornane_still_names(namer):
    # control: a straight substituent must still name.
    smi = "CC1CC2CCC1C2"
    name = namer.name(smi)
    assert name and "unknown" not in name, name
    assert _full_rt(smi, name), name
