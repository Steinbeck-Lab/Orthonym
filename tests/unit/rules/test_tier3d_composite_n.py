"""Wave2 — composite-N families (amidrazone / hydrazidine / amidoxime /
sulfinimidamide / thiohydrazide / hydrazide N-N' / hydrazinecarboxylic /
amidine-as-prefix), built per the adversarially-verified research specs.

Each new perception SMARTS overlaps a shipped C=N family and is resolved by a
cascade-suppression entry (the resolver order is load-bearing). The regression
class below independently confirms the shipped imine / oxime / hydrazone /
amidine / amide / carbamic / guanidine golds are NOT stolen — the whole point
of the perception-priority prototype the spec mandated.
"""

import pytest

from orthonym import name_compound


@pytest.mark.unit
class TestCompositeNSuffixes:
    @pytest.mark.parametrize("smiles,expected", [
        # sulfinimidamide item 25) — SX3, disjoint from SX4 sulfonimidamide
        ("CS(=N)N", "methanesulfinimidamide"),
        ("CCS(=N)N", "ethanesulfinimidamide"),
        # amidrazone / hydrazonamide
        ("CC(N)=NN", "ethanehydrazonamide"),
        ("NC(=NN)c1ccccc1", "benzenecarbohydrazonamide"),
        # hydrazidine / hydrazonohydrazide
        ("CC(=NN)NN", "ethanehydrazonohydrazide"),
        ("C(=NN)(NN)c1ccccc1", "benzenecarbohydrazonohydrazide"),
        # amidoxime — N'-hydroxy amidine
        ("CC(=NO)N", "N'-hydroxyethanimidamide"),
        ("NC(=NO)c1ccccc1", "N'-hydroxybenzenecarboximidamide"),
        ("CCON=C(N)C", "N'-ethoxyethanimidamide"),
        # thiohydrazide
        ("CC(=S)NN", "ethanethiohydrazide"),
        ("CCC(=S)NN", "propanethiohydrazide"),
        ("NNC(=S)c1ccccc1", "benzenecarbothiohydrazide"),
    ])
    def test_suffix_families(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestHydrazideNNPrefixes:
    @pytest.mark.parametrize("smiles,expected", [
        ("CNNC(C)=O", "N'-methylethanehydrazide"),
        ("CN(N)C(C)=O", "N-methylethanehydrazide"),
        ("CNNC(=S)C", "N'-methylethanethiohydrazide"),
        # hydrazinecarboxylic acid — carbamic_acid SMARTS tightened
        ("NNC(=O)O", "hydrazinecarboxylic acid"),
        ("CNNC(=O)O", "N'-methylhydrazinecarboxylic acid"),
    ])
    def test_hydrazide_nn(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestAmidineAsPrefix:
    @pytest.mark.parametrize("smiles,expected", [
        ("CC(=N)Nc1ccc(C(=O)O)cc1", "4-ethanimidamidobenzoic acid"),
    ])
    def test_imidamido(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # RING-attached amidine stays carbamimidoyl (PIN for ring/off-chain, BB 34332)
        ("N=C(N)c1ccc(C(=O)O)cc1", "4-carbamimidoylbenzoic acid"),
        #, BB 34338): chain-terminal amidine -> amino+imino
        ("N=C(N)CCC(=O)O", "4-amino-4-iminobutanoic acid"),
        # amido family unaffected (C=O branch)
        ("CC(=O)Nc1ccc(C(=O)O)cc1", "4-acetamidobenzoic acid"),
        # Schiff base (imino N on ring) must NOT be captured as imidamido
        ("CC=Nc1ccc(C(=O)O)cc1", "4-ethaniminylbenzoic acid"),
        # guanidine must NOT be captured (aryl guanidino demoted to the PIN
        # prefix carbamimidoylamino per / BBv2 L1700; Wave-3 heal)
        ("N=C(N)Nc1ccc(C(=O)O)cc1", "4-(carbamimidoylamino)benzoic acid"),
    ])
    def test_amidine_prefix_controls(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestNoPerceptionRegression:
    """The composite-N SMARTS overlap the shipped C=N families; cascade
    suppression must leave every shipped gold correct."""

    @pytest.mark.parametrize("smiles,expected", [
        ("CC=NC", "N-methylethanimine"),
        ("N=C1CCCCC1", "cyclohexan-1-imine"),
        ("CC=NO", "N-hydroxyethanimine"),
        ("CCC=NO", "N-hydroxypropan-1-imine"),
        ("CC(C)=NO", "N-hydroxypropan-2-imine"),
        # Wave-3: substitutive ylidene-hydrazine is PIN;
        # 'acetophenone hydrazone' is functional-class / general-only.
        ("CC(=NN)c1ccccc1", "(1-phenylethylidene)hydrazine"),
        ("CCC(N)=N", "propanimidamide"),
        ("CC(=N)N", "ethanimidamide"),
        ("NNC(C)=O", "acetohydrazide"),
        ("CCCCC(=O)NN", "pentanehydrazide"),
        ("NNC(=O)c1ccccc1", "benzohydrazide"),
        ("NC(=O)O", "carbamic acid"),
        ("CNC(=O)O", "methylcarbamic acid"),
        ("CC(=S)N", "ethanethioamide"),
        ("NC(=N)N", "guanidine"),
        ("N=C(N)NC", "N-methylguanidine"),
        ("CS(=O)(=O)N", "methanesulfonamide"),
        ("CS(=O)(=N)N", "methanesulfonimidamide"),
    ])
    def test_shipped_golds_hold(self, smiles, expected):
        assert name_compound(smiles) == expected
