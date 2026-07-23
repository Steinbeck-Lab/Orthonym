"""v28 Cluster D — PIN-backlog per-class builds (OPSIN-free, root-cause).

Each target's gold is Blue-Book-verified (research workflow wf_bebb8a05):
  D-1 subordinate thioanhydride  -> acylsulfanyl/oxo split      (P-41, P-35.5.1)
  D-2 no-PCG polyfunctional      -> prefix-only substitutive name (P-41, P-63.6)
  D-3 ring-ester decoration      -> alkoxycarbonyl ring prefix    (P-65.6.3.2.3, P-16.3.3)
  D-4 aromatic nitrile oxide     -> benzonitrile-oxide + prefix   (P-66.5.4.2, P-16.3.3)

All assertions use name_compound (PIN style, OPSIN-free) so the whole file is
safe for touched-module HEAD-A/B. Regression cases lock the blast radius.
"""
import pytest
from rdkit import RDLogger

from orthonym import name_compound

RDLogger.logger().setLevel(RDLogger.ERROR)


class TestD1SubordinateThioanhydride:
    """A -CO-S-CO- linkage SUBORDINATE to a senior acid is expressed
    substitutively (in-chain acyl -> oxo, acyl-S -> acylsulfanyl), never as an
    'anhydride' word (P-41 class seniority: acids > anhydrides)."""

    def test_thioanhydride_of_acid_is_acylsulfanyl_oxo(self):
        # HOOC-(CH2)7-C(=O)-S-C(=O)-CH3 ; BB P-35.5.1 (CH3-CO-S- = acetylsulfanyl)
        assert (name_compound("CC(=O)SC(=O)CCCCCCCC(=O)O")
                == "9-(acetylsulfanyl)-9-oxononanoic acid")

    @pytest.mark.parametrize("smiles,expected", [
        # Genuine anhydrides (anhydride IS the senior class) must be UNCHANGED.
        ("CC(=O)OC(=O)C", "acetic anhydride"),
        ("O=C(OC(=O)c1ccccc1)c1ccccc1", "benzoic anhydride"),
        # Existing thioester-of-diacid precedent (never was folded) unchanged.
        ("CCSC(=O)CCC(=O)O", "4-(ethylsulfanyl)-4-oxobutanoic acid"),
    ])
    def test_genuine_anhydrides_and_thioesters_unchanged(self, smiles, expected):
        assert name_compound(smiles) == expected


class TestD2NoPcgPolyfunctional:
    """A polyfunctional parent bearing ONLY prefix-cited groups (no principal
    characteristic group) still names substitutively, with no suffix
    (P-41 worked example BB:18282; P-63.6).

    DEFERRED (v28 Cluster D): the research (wf_bebb8a05) proposed a contained
    'no-suffix arm at polyfunctional.py:2108', but tracing shows that is NOT the
    reachable path. For CSCCS(=O)C the clean polyfunctional path (which builds
    'methanesulfinyl' via get_sulfinyl_prefix) is never reached — composer.
    assemble_name is not even called — and a general/skeletal-replacement engine
    path emits the WRONG '1,2-bis(methylsulfanyl)-1-sulfinylethane' (mis-splitting
    CH3-S(=O)- into methylsulfanyl + bare sulfinyl). Production fail-closes safely
    (SELF-01 round-trips to CSCC(SC)=S=O, a different molecule -> 'unknown'), so
    NO wrong name ships. The correct fix is a real no-PCG substitutive-parent
    build (route no-PCG polyfunctional acyclics through the polyfunctional prefix
    assembler and make it win over the skeletal path), not a one-liner. Deferred
    to avoid a half-mapped change to the core engine; re-scope before building.
    """

    @pytest.mark.xfail(reason="v28 Cluster D deferred: no-PCG substitutive-parent "
                              "build (research under-scoped); fails closed today",
                       strict=True)
    def test_sulfinyl_sulfanyl_ethane(self):
        # CH3-S-CH2-CH2-S(=O)-CH3 ; BB P-41 verbatim PIN. Note: CH3-S(=O)- is
        # 'methanesulfinyl' (Table 5.1 -S(O)-R -> alkanesulfinyl), not 'methylsulfinyl'.
        assert (name_compound("CSCCS(=O)C")
                == "1-(methanesulfinyl)-2-(methylsulfanyl)ethane")


class TestD3RingEsterAlkoxycarbonyl:
    """A monocyclic ring substituent carrying an alkyl-ester decoration is
    expressed with the alkoxycarbonyl prefix (P-65.6.3), enclosed per P-16.3.3
    because it is a compound substituent prefix."""

    def test_methoxycarbonyl_cyclohexyl_butanoic_acid(self):
        # senior acid parent (butanoic acid); ring demoted; ester -> methoxycarbonyl
        assert (name_compound("COC(=O)C1CCCCC1CCCC(=O)O")
                == "4-[2-(methoxycarbonyl)cyclohexyl]butanoic acid")

    @pytest.mark.parametrize("smiles,expected", [
        # simple alkyl ring substituent unchanged (no enclosure).
        ("CC1CCCCC1CCCC(=O)O", "4-(2-methylcyclohexyl)butanoic acid"),
        # ethyl ester -> ethoxycarbonyl (class generalises beyond methyl).
        ("CCOC(=O)C1CCCCC1CCCC(=O)O", "4-[2-(ethoxycarbonyl)cyclohexyl]butanoic acid"),
    ])
    def test_ring_substituent_regressions(self, smiles, expected):
        assert name_compound(smiles) == expected


class TestD4AromaticNitrileOxide:
    """A neutral aromatic nitrile oxide bearing a co-substituent is named by
    functional-class method (1): '<benzonitrile> oxide' (P-66.5.4.2), with the
    senior nitrile oxide demoting a co-present ester to the methoxycarbonyl
    prefix (P-65.6.3), enclosed per P-16.3.3."""

    def test_methoxycarbonyl_benzonitrile_oxide(self):
        # BB:34893 verbatim PIN (rejects 'methyl 4-...benzoate').
        assert (name_compound("COC(=O)C1=CC=C(C#[N+][O-])C=C1")
                == "4-(methoxycarbonyl)benzonitrile oxide")

    @pytest.mark.parametrize("smiles,expected", [
        # unsubstituted / simple nitrile oxides must be UNCHANGED.
        ("[O-][N+]#Cc1ccccc1", "benzonitrile oxide"),
        ("CC#[N+][O-]", "acetonitrile oxide"),
    ])
    def test_simple_nitrile_oxides_unchanged(self, smiles, expected):
        assert name_compound(smiles) == expected
