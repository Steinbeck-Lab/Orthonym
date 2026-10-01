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
        mol, feats = _features("c1ccccc1CC")  # ring is parent
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
    # -A: mancude/aromatic ring systems fail-close (von-Baeyer is non-PIN
    # for them; their PIN is a fused/retained parent + added/indicated H). This
    # includes caffeine, whose von-Baeyer form was the OPSIN-invalid oxo/ene
    # valence-clash name. Their real names come from the PIN path.
    @pytest.mark.parametrize("smi", [
        "C1CCc2ccccc2C1",                  # tetralin
        "c1ccc2[nH]cnc2c1",                # benzimidazole
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",    # caffeine (was invalid engine name)
    ])
    def test_mancude_ring_refused(self, smi):
        mol, feats = _features(smi)
        assert name_general_ring(mol, feats) is None

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
        mol, feats = _features("O=C1CCC2CCCCC2C1")  # decalinone: saturated cage
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


class TestChainAmideNSubstituents:
    """The chain partition keeps only the heteroatoms of a principal-group match as
    suffix atoms; the N-substituents of a chain amide are cited with the locant N.

     (heading ' *N*-Substitution', the Blue Book;
    sentence:32774): "Substituted primary amides, with general structures such as
    R-CO-NHR' and R-CO-NR'R'',... are named by citing the substituents R' and R''
    as prefixes preceded by the locant *N* when one amide group is present. In di-
    and polyamides... *N* locants with superscripted arabic numbers... are used";
    example 'N,N-dimethylpropanamide (PIN)' (:32788). The N locant comes before the
    numeric locants of an identical prefix: 'N,N,2-trimethyl-3-{...}propanamide
    (PIN)' (:21624,."""

    def test_n_methyl(self):
        assert _name("CCC(=O)NC").name == "N-methylpropanamide"

    def test_n_n_dimethyl_bb_example(self):
        assert _name("CCC(=O)N(C)C").name == "N,N-dimethylpropanamide"

    def test_n_locant_merged_before_numeric_locant(self):
        assert _name("CC(C)C(=O)NC").name == "N,2-dimethylpropanamide"

    def test_n_substituent_with_its_own_substituents(self):
        assert (_name("CCCCCC(=O)NC(CO)C(O)CCC").name
                == "N-(1,3-dihydroxyhexan-2-yl)hexanamide")

    def test_n_substituent_bindings_cover_all_heavy_atoms(self):
        mol = Chem.MolFromSmiles("CCCC(=O)N(C)c1ccccc1")
        res = _name("CCCC(=O)N(C)c1ccccc1")
        assert res.name == "N-methyl-N-phenylbutanamide"
        bound = sorted(i for b in res.bindings for i in b.atom_ids)
        assert bound == list(range(mol.GetNumHeavyAtoms()))

    def test_ketone_flanking_carbon_is_a_substituent(self):
        # (the Blue Book), '1-phenylpropan-1-one (PIN)' (:28342): the
        # ketone match's off-chain flanking carbon is the phenyl substituent, not a
        # suffix atom.
        assert _name("CCC(=O)c1ccccc1").name == "1-phenylpropan-1-one"

    @pytest.mark.parametrize("smi", [
        "O=C(CCC(=O)NC)NC",   # N1,N4-dimethylbutanediamide: superscript N locants not built
        "CC(=O)N1CCCC1",      # amide N in a ring: the ring is not an N-substituent
        "CC(=O)NCCCC",        # principal chain through the N-substituent, not the acyl carbon
    ])
    def test_out_of_scope_amide_fails_closed(self, smi):
        mol, feats = _features(smi)
        assert name_general_chain(mol, feats) is None
