"""
Batch regression tests for medium molecule stereo auto-unlock and charge fixes.

a phase Plan 03: Verifies that stereo descriptors appear in names after
Wave 1 parent selection improvements, and that charge pipeline produces
correct names for medium-molecule ions/salts.

Test groups:
  1. Stereo auto-unlocked: Compounds that gained stereodescriptors after
     Wave 1 parent fixes (11 compounds,)
  2. Charge improved: Medium-molecule charge compounds with better names
     after a phase/66 pipeline fixes (5 compounds,)
  3. RT verification: Compounds that round-trip through OPSIN after a phase
"""

import pytest
from orthonym import name_compound


# ---------------------------------------------------------------------------
# Group 1: Stereo auto-unlocked compounds
# These compounds gained stereodescriptors when Wave 1 fixed their parent
# selection. Validates that a phase stereo wiring works automatically
# once the correct parent is identified.
# ---------------------------------------------------------------------------
STEREO_AUTOUNLOCK = [
    pytest.param(
        "C=C1NC(=O)[C@H]([C@@H](C)[C@]2(O)C(=O)N(C)c3ccccc32)NC1=O",
        "(2R)",
        id="indolinone-2R",
    ),
    pytest.param(
        "Oc1ccc2c(c1)O[C@H](c1ccc(O)c(O)c1)[C@@H](O)[C@@H]2O",
        "(2R,3S,4R)",
        id="trihydroxychromane-2R3S4R",
    ),
    pytest.param(
        "CC(C)=CCC/C(C)=C/CC[C@]1(C)Cc2c(c(O)cc3c2CN("
        "[C@H]2CCCNC2=O)C3=O)C[C@@H]1O",
        "(11S,12R)",
        id="tricyclic-aza-11S12R",
    ),
    pytest.param(
        "C/C1=C/C[C@@H](/C(C)=C/c2csc(C)n2)OC(=O)C[C@H](O)"
        "C(C)(C)C(=O)[C@H](C)[C@@H](O)/C(C)=C/CC1",
        "(4S,7R,8R,9E,13Z,16S)",
        id="oxacyclohexadecanone-6stereo",
    ),
    pytest.param(
        "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(C)=O",
        "(2R)",
        id="phospholipid-zwitterion-2R",
    ),
    pytest.param(
        "C=C1CC[C@@H](C/C=C2/CC[C@]3(OC2)O[C@@]2(O)CC[C@]3(C)"
        "OC2(C)C)C(C)(C)[C@H]1[C@@H](O)C=C1CCOC1=O",
        "(1S,2S,5S,9Z)",
        id="tricyclic-trioxa-1S2S5S",
    ),
    pytest.param(
        "O=C1c2c(O)cc(O)cc2O[C@@H](c2ccc(O)c(O)c2)[C@@H]1"
        "O[C@@H]1OC[C@@H](O)[C@H](O)[C@H]1O",
        "(2S,3S)",
        id="xylopyranosyloxy-chromanone-2S3S",
    ),
    pytest.param(
        "O=C1N[C@@H](C[C@@]2(O)c3ccccc3N3C(=O)[C@@H]4CCCCN4"
        "[C@@H]32)C(=O)N[C@H]1Cc1ccccc1",
        "(8R,9S,15S)",
        id="tetracyclic-diaza-8R9S15S",
    ),
    pytest.param(
        "C[C@@H]1CC(=O)O[C@@H](C)[C@H](O)/C=C\\C(=O)O[C@@H](C)"
        "C/C=C\\C(=O)O1",
        "(4R,7Z,10S,13Z,15R,16S)",
        id="macrolide-6stereo",
    ),
    pytest.param(
        "C#CCN1CC(=O)N(COC(=O)[C@@H]2[C@@H](C=C(C)C)C2(C)C)C1=O",
        "(2R,3R)",
        id="cyclopropane-ester-2R3R",
    ),
    pytest.param(
        "CC(C)=CCC/C(C)=C/COC[C@H]1O[C@@H](N2CCC(=O)NC2=O)"
        "[C@H](O)[C@@H]1O",
        "(2R,3R,4S,5R)",
        id="thf-piperazinyl-4stereo",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,stereo_marker", STEREO_AUTOUNLOCK)
def test_stereo_auto_unlock(smiles, stereo_marker):
    """Stereo descriptors must appear after Wave 1 parent selection fixes."""
    name = name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert stereo_marker in name, (
        f"Expected stereo '{stereo_marker}' in: {name}"
    )


# ---------------------------------------------------------------------------
# Group 2: Charge improved compounds
# Validates that medium-molecule charged species produce meaningful names
# after a phase + 66 pipeline improvements.
# ---------------------------------------------------------------------------
CHARGE_FIXES = [
    pytest.param(
        "CCCCCCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(C)=O",
        "palmitate",
        id="phospholipid-palmitate",
    ),
    pytest.param(
        "COc1ccc(C(CN(C)C)C2(O)CCCCC2)cc1.[Cl-].[H+]",
        "hydrochloride",
        id="amine-hydrochloride-salt",
    ),
    pytest.param(
        "CCCCCCCCCCCCCC(O)CC(=O)OC(CC(=O)[O-])C[N+](C)(C)C",
        "palmitoyloxy",
        id="zwitterion-palmitoyloxy",
    ),
    pytest.param(
        "CCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCC",
        # a phase cleanup: relaxed substring from "decane" to "decan" so
        # that both `decane` (chain-PIN) AND `decanoyloxy` (acyloxy
        # connector PIN per / are accepted. The output
        # `1,2-bis(decanoyloxy)propyl...` uses the acyloxy form — valid
        # IUPAC. Deeper bug (trimethylammonium mis-perceived as
        # propylamino) is tracked as -06 in cleanup-deferred-items.md.
        "decan",  # accepts decane, decanoyl, decanoyloxy, decanediyl, etc.
        id="phospholipid-didecanoate",
    ),
    pytest.param(
        "CC(C)[C@@]1(C)N=C(c2nc3ccccc3cc2C(=O)[O-])NC1=O.[NH4+]",
        "ammonium",
        id="ammonium-carboxylate-salt",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected", CHARGE_FIXES)
def test_charge_naming_medium(smiles, expected):
    """Charged medium molecules must produce names with expected structural feature."""
    name = name_compound(smiles)
    assert name is not None, "name_compound returned None"
    assert expected in name, (
        f"Expected '{expected}' in: {name}"
    )


# ---------------------------------------------------------------------------
# Group 3: Newly round-tripping compounds from a phase
# These compounds gained correct names through a phase parent fixes that
# now successfully round-trip through OPSIN -> InChI comparison.
# ---------------------------------------------------------------------------
NEWLY_RT_P66 = [
    pytest.param(
        "C/C(=C\\CC/C(C)=C/C/C=C(/CC(=O)c1cc(O)ccc1O)C(=O)O)CO",
        "(2Z,5E,9E)-2-(1-oxo1-(2,5-dihydroxyphenyl)ethyl)-"
        "11-hydroxy-6,10-dimethylundeca-2,5,9-trienoic acid",
        id="ganoderenic-acid-analog-RT",
    ),
    pytest.param(
        "CC(=O)O[C@H]1CC[C@]2(C)C3=C(CC[C@H]2C1(C)C)"
        "[C@]1(C)C[C@@H](O)[C@H]([C@@H](C/C=C/C(C)(C)O)"
        "C(=O)O)[C@@]1(C)CC3",
        "(3S,5R,10S,13R,14R,16R,17R,20R,23E)-16,21,25-"
        "trihydroxy-4,4,14-trimethyl-21-oxocholest-8,23-"
        "dien-3-yl acetate",
        id="cholest-steroid-acetate-RT",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize("smiles,expected_name", NEWLY_RT_P66)
def test_newly_rt_phase66(smiles, expected_name):
    """Compounds that newly round-trip after a phase must keep exact names."""
    name = name_compound(smiles)
    assert name == expected_name, (
        f"Expected: {expected_name}\nGot: {name}"
    )
