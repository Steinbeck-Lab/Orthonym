"""Wave2 + — aryl-vinyl/styryl + substituted-ring + sulfonimidamide.

T3c: the styryl -CH=CH-C6H5 arm once counted carbons THROUGH the far ring
(8 C -> '(4E)-4-octylbenzoic acid', a wrong-structure leak). The Wave2 T3a
conservation guard first killed the leak (fail-closed); the FULL-Tier-3 build
(user directive "complete all") then added the POSITIVE aryl-vinyl namer
(_name_aryl_vinyl_substituent) so it names 4-[(E)-2-phenylethenyl]benzoic acid,
plus the both-ends-branched free-valence tie-break ((2-methylpentan-3-yl)benzene)
and the benzenyl->phenyl guard.

 (additive coverage): sulfonimidamide -S(=O)(=NH)-NH2 is a preselected
suffix (BB Table 6.1 item 20, ranked just below sulfonamide). SX4 on sulfur,
disjoint from the carbon C=N families. (The composite-N CARBON families now
live in test_tier3d_composite_n.py.)
"""

import pytest

from orthonym import name_compound


@pytest.mark.unit
class TestArylVinyl:
    @pytest.mark.parametrize("smiles,expected", [
        # both the strict-PIN bracket form and the emitted paren form OPSIN-RT.
        ("OC(=O)c1ccc(/C=C/c2ccccc2)cc1", "4-[(E)-2-phenylethenyl]benzoic acid"),
        ("OC(=O)c1ccc(C=Cc2ccccc2)cc1", "4-(2-phenylethenyl)benzoic acid"),
    ], ids=["E-styryl", "no-stereo"])
    def test_aryl_vinyl(self, smiles, expected):
        got = name_compound(smiles)
        # accept the RT-equivalent paren form the assembler currently emits
        alts = {expected, expected.replace("[", "(").replace("]", ")")}
        assert got in alts, f"got {got!r}"

    @pytest.mark.parametrize("smiles,expected", [
        ("C=Cc1ccccc1", "ethenylbenzene"),           # plain vinyl still names
        ("CCc1ccccc1", "ethylbenzene"),               # plain alkyl unaffected
        ("Cc1ccccc1", "toluene"),
        ("OC(=O)c1ccc(CCC(=O)O)cc1", "4-(2-carboxyethyl)benzoic acid"),
        ("c1ccc(CCc2ccccc2)cc1", "1,1'-(ethane-1,2-diyl)dibenzene"),
        ("OC(=O)c1ccc(C=C)cc1", "4-ethenylbenzoic acid"),
        ("OC(=O)/C=C/c1ccccc1", "(2E)-3-phenylprop-2-enoic acid"),
    ])
    def test_controls(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestSubstitutedCarbocyclicAndTieBreak:
    @pytest.mark.parametrize("smiles,expected", [
        ("CC1CCCC1c1ccccc1", "(2-methylcyclopentyl)benzene"),
        ("ClC1CCCC1c1ccccc1", "(2-chlorocyclopentyl)benzene"),
        ("CCC(c1ccccc1)C(C)C", "(2-methylpentan-3-yl)benzene"),
    ])
    def test_positive(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("CC(C)c1ccccc1", "(propan-2-yl)benzene"),
        ("CCC(C)c1ccccc1", "(butan-2-yl)benzene"),
        ("CCCC(C)c1ccccc1", "(pentan-2-yl)benzene"),
        ("CCC(CC)c1ccccc1", "(pentan-3-yl)benzene"),
    ])
    def test_located_alkyl_controls(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestSulfonimidamide:
    @pytest.mark.parametrize("smiles,expected", [
        ("CS(=O)(=N)N", "methanesulfonimidamide"),
        ("CCS(=O)(=N)N", "ethanesulfonimidamide"),           # ethane elision
        ("CCCS(=O)(=N)N", "propane-1-sulfonimidamide"),      # C3 keeps locant
    ])
    def test_sulfonimidamide(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # Disjoint-SMARTS controls: the S-based FG must not perturb the
        # carbon-N families or the plain sulfonamide.
        ("CS(=O)(=O)N", "methanesulfonamide"),
        ("CCS(N)(=O)=O", "ethanesulfonamide"),
        ("NNC(C)=O", "acetohydrazide"),
        # Wave-3: substitutive ylidene-hydrazine is PIN;
        # 'acetophenone hydrazone' is functional-class / general-only.
        ("CC(=NN)c1ccccc1", "(1-phenylethylidene)hydrazine"),
        ("CC=NO", "N-hydroxyethanimine"),
    ])
    def test_no_perception_regression(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_composite_n_carbon_families_now_built(self):
        # These were deferred fail-closed; the full-Tier-3 build (user directive
        # "complete all") now names them (see test_tier3d_composite_n.py). This
        # is the cross-file sanity that the deferral is gone.
        assert name_compound("CC(=NN)N") == "ethanehydrazonamide"
        assert name_compound("CC(=NO)N") == "N'-hydroxyethanimidamide"
