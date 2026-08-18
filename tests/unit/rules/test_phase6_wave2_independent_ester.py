import pytest
from rdkit import Chem
from rdkit.Chem import inchi
from orthonym import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse


@pytest.fixture(scope="module")
def namer():
    return Orthonym()


def _full_rt(smiles: str, name: str) -> bool:
    o = opsin_parse(name)
    if not o:
        return False
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles)) == \
        inchi.MolToInchiKey(Chem.MolFromSmiles(o))


@pytest.mark.opsin_gate
def test_acyloxymethyl_ring_methyl_ester(namer):
    smi = "COC(=O)C1CCCCC1COC(C)=O"
    name = namer.name(smi)
    assert name == "methyl 2-[(acetyloxy)methyl]cyclohexane-1-carboxylate", name
    assert " methyl " not in name  # no stray-space token
    assert _full_rt(smi, name), name


@pytest.mark.opsin_gate
def test_ring_acyloxy_direct(namer):
    # acyloxy directly on the ring (spy's confirmed clean win)
    smi = "COC(=O)c1ccc(OC(C)=O)cc1"
    name = namer.name(smi)
    assert name and "unknown" not in name, name
    assert _full_rt(smi, name), name


@pytest.mark.opsin_gate
def test_topological_acyl_never_wrong(namer):
    # a macrolactone / long topological acyl must abstain or RT -- never a wrong flattening
    smi = "O=C1CCCCCCCCCCCOC1"  # a macrolactone (oxacyclotridecan-2-one class)
    name = namer.name(smi)
    if name and "unknown" not in name:
        assert _full_rt(smi, name), name


# ---------------------------------------------------------------------------
# Review fix: ring-acid detection must be SMILES-atom-order-invariant.
#
# parse_ester_fragments's BFS excludes only THIS ester's own ester_o, so for
# a ring principal acid_atoms can be contaminated with the OTHER ester's
# atoms (anything reachable around the ring). Before the fix,
# acid_is_ring_acid(mol, acid_atoms) scanned that (possibly contaminated) set
# and returned based on the FIRST carbon that looked like a carbonyl -- which
# carbon that is depends on Python set/atom-index iteration order, so the
# SAME molecule (InChIKey AVNYEDDAAFGZHZ), spelled with a different SMILES
# atom order, could take a different (wrong, fabricating) branch.
# ---------------------------------------------------------------------------

_WITNESS_ORDERINGS = [
    "COC(=O)C1CCCCC1COC(C)=O",
    "CC(=O)OCC1CCCCC1C(=O)OC",
]
_EXPECTED_WITNESS_NAME = "methyl 2-[(acetyloxy)methyl]cyclohexane-1-carboxylate"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", _WITNESS_ORDERINGS)
def test_ring_acid_detection_is_smiles_order_invariant(namer, smi):
    name = namer.name(smi)
    assert name == _EXPECTED_WITNESS_NAME, (smi, name)
    assert _full_rt(smi, name), (smi, name)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", _WITNESS_ORDERINGS)
def test_ring_acid_detection_stable_under_rdkit_canonicalization(namer, smi):
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    name = namer.name(canonical)
    assert name == _EXPECTED_WITNESS_NAME, (canonical, name)
    assert _full_rt(canonical, name), (canonical, name)


@pytest.mark.opsin_gate
def test_chain_acid_principal_with_ring_non_principal_never_fabricates(namer):
    # Mirror case: a CHAIN-acid principal ester whose non-principal ester's
    # acid is a ring (cyclohexanecarbonyl) attached at the far end of the
    # chain -- must never fabricate a wrong stem like 'hexadecanoic'; either
    # the correct name or an honest abstention.
    smi = "COC(=O)CCCCCCCCOC(=O)C1CCCCC1"
    name = namer.name(smi)
    assert not name or "unknown" in name or _full_rt(smi, name), name
    if name:
        assert "hexadecan" not in name, name
