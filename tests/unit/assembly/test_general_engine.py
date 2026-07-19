# tests/unit/assembly/test_general_engine.py
"""v25 G1: general chain engine — partition, refusals, naming, bindings."""
import pytest
from rdkit import Chem

from orthonym.assembly.general_engine import (
    GeneralEngineResult, TokenBinding, name_general_chain,
)

pytestmark = pytest.mark.unit


def _features(smiles):
    """Build perceived+classified features exactly as the GENERAL block does."""
    from orthonym.namer import Orthonym
    nm = Orthonym(_disable_opsin_validity_gate=True)
    mol = Chem.MolFromSmiles(smiles)
    canonical = Chem.MolToSmiles(mol, canonical=True)
    feats = nm._perceive(mol, smiles, canonical)
    nm._classify(feats)
    return mol, feats


class TestRefusals:
    def test_refuses_net_charge(self):
        mol, feats = _features("CC[O-]")
        assert name_general_chain(mol, feats) is None

    def test_refuses_multifragment(self):
        mol, feats = _features("CCO.CCO")
        assert name_general_chain(mol, feats) is None

    def test_refuses_ring_parent(self):
        mol, feats = _features("c1ccccc1CC")  # ring is parent (P-44.1.2.2)
        assert name_general_chain(mol, feats) is None

    def test_refuses_single_carbon_chain(self):
        mol, feats = _features("CO")  # methanol: 1-carbon chain, G1 out of scope
        assert name_general_chain(mol, feats) is None


from orthonym.validation.e1_certificate import verify_certificate


def _name(smiles):
    mol, feats = _features(smiles)
    res = name_general_chain(mol, feats)
    if res is not None:
        v = verify_certificate(mol, res)
        assert v.ok, v.reason
    return res


class TestChainNaming:
    def test_ethanol(self):
        assert _name("CCO").name == "ethan-1-ol"

    def test_propan2ol(self):
        assert _name("CC(O)C").name == "propan-2-ol"

    def test_gem_dimethyl(self):
        assert _name("CCCC(C)(C)CC").name == "3,3-dimethylhexane"

    def test_terminal_acid_with_branch(self):
        assert _name("CC(C)CC(O)=O").name == "3-methylbutanoic acid"

    def test_unsaturation_with_suffix(self):
        assert _name("C=CCO").name == "prop-2-en-1-ol"

    def test_halide_prefix(self):
        assert _name("CC(Cl)CC").name == "2-chlorobutane"

    def test_ring_substituent_on_chain_parent_recurses(self):
        # PCG on chain -> chain parent; phenyl named via recursion tiers
        assert _name("OCCc1ccccc1").name == "2-phenylethan-1-ol"

    def test_diol_multiplied_suffix(self):
        assert _name("OCCO").name == "ethane-1,2-diol"

    def test_bindings_cover_all_heavy_atoms(self):
        res = _name("CC(Cl)CC")
        bound = sorted(i for b in res.bindings for i in b.atom_ids)
        assert bound == list(range(5))


from orthonym.assembly.general_engine import name_general, name_general_ring


class TestRingNaming:
    def test_tetralin_vonbaeyer_triene(self):
        mol, feats = _features("C1CCc2ccccc2C1")
        res = name_general_ring(mol, feats)
        assert res is not None
        assert res.name.startswith("bicyclo[4.4.0]deca-")
        assert res.name.endswith("triene")
        assert verify_certificate(mol, res).ok

    def test_benzimidazole_diazabicyclo(self):
        mol, feats = _features("c1ccc2[nH]cnc2c1")
        res = name_general_ring(mol, feats)
        assert res is not None
        assert "diazabicyclo[4.3.0]nona-" in res.name
        assert "tetraene" in res.name
        assert verify_certificate(mol, res).ok

    def test_cage_with_suffix(self):
        # 2-decalone (bicyclo[4.4.0]decan-2-one shape): ring ketone
        mol, feats = _features("O=C1CCC2CCCCC2C1")
        res = name_general_ring(mol, feats)
        assert res is not None
        assert "bicyclo[4.4.0]dec" in res.name
        assert res.name.endswith("one")
        assert verify_certificate(mol, res).ok

    def test_cage_with_substituent_prefix(self):
        # 2-methyldecalin
        mol, feats = _features("CC1CCC2CCCCC2C1")
        res = name_general_ring(mol, feats)
        assert res is not None
        assert "methylbicyclo[4.4.0]decane" in res.name
        assert verify_certificate(mol, res).ok

    def test_monocycle_refused(self):
        mol, feats = _features("c1ccccc1")
        assert name_general_ring(mol, feats) is None


class TestDispatcher:
    def test_dispatch_chain(self):
        mol, feats = _features("CCO")
        assert name_general(mol, feats).name == "ethan-1-ol"

    def test_dispatch_ring(self):
        mol, feats = _features("C1CCc2ccccc2C1")
        res = name_general(mol, feats)
        assert res is not None and res.name.startswith("bicyclo")


class TestEngineStereo:
    def test_chain_stereocenter_descriptor(self):
        # C[C@H](O)CC: O>Et>Me>H, @ from methyl -> S (rdCIPLabeler-verified)
        res = _name("C[C@H](O)CC")
        assert res.name == "(2S)-butan-2-ol"

    def test_chain_stereocenter_descriptor_r(self):
        res = _name("C[C@@H](O)CC")
        assert res.name == "(2R)-butan-2-ol"

    def test_chain_ez_descriptor(self):
        res = _name("C/C=C/CO")      # (2E)-but-2-en-1-ol
        assert res.name == "(2E)-but-2-en-1-ol"

    def test_achiral_unchanged(self):
        assert _name("CCO").name == "ethan-1-ol"

    def test_ring_stereo_descriptor_present(self):
        # trans-decalin-2-ol shape: engine cage name carries (..R/..S) block
        mol, feats = _features("O[C@H]1CC[C@@H]2CCCC[C@H]2C1")
        res = name_general_ring(mol, feats)
        assert res is not None
        assert res.name.startswith("(") and "bicyclo" in res.name
        assert verify_certificate(mol, res).ok
