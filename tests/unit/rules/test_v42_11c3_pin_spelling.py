""" a phase (batch 11C3) — euphony/mancude + charged parents + misc spelling.

Eight shipped fixes. Every target below emits its Blue-Book PIN and round-trips
0-wrong. The traced real sites differed from the brief's guesses for #25 (the
euphonic-'a' lived in polycyclic._build_parent_with_unsaturation, NOT
terminal_ring) and #26a (fragment_assembly._amine_to_prefix, NOT sulfonamides),
recorded in task-11C3-report.md.

#11 multiplicative bridge cited BARE before the multiplier:
     '4,4'-oxydi(...)' / '4,4'-methylenedi(...)', not '(oxy)di'/'(methylene)di'.
#25 euphonic connective 'a' before the FIRST unsaturation ending only when it is
     MULTIPLIED: 'cyclooct-3-en-7-yne' (single en+yn, elide) vs
     'cycloocta-3,7-diyne' (diyne, retain). The engine had BOTH backwards.
#34 retained mancude ring 'phosphinine' as an -yl substituent, not
     the systematic '1-phosphacyclohexa-1,3,5-trien-4-yl'.
#27 'benzenylium' — a ring ylium uses the parent-hydride name,
     the Blue Book 'benzenylium (PIN)'), not the substituent 'phenylium'.
#28 'hydroxyazanide' — the N-anion of hydroxylamine is built on azane
     , the Blue Book preselected), not the retained 'hydroxylaminide'.
#26a 'N-hydroxymethanesulfonamide' — the N-OH prefix is 'hydroxy' not 'hydroxyl'
     .
#26b '(hydroxyimino)' compound prefix takes enclosing marks, even on
     the ring/seniority composer path.
#26c 'disilylmethyl' — a one-carbon (methyl) substituent parent omits its
     redundant '1,1' locants (a)).
#26d 'tetraphenyl-λ5-phosphanyl' — a 5-bonded organyl P carries the λ descriptor
     , routed through the shared LAMBDA constant.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.lambda_convention import LAMBDA
from tests.support.jars import jar_or_none

_OPSIN_JAR = jar_or_none()
_OPSIN_OK = shutil.which("java") is not None and _OPSIN_JAR is not None


def _ikey(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(m) if m is not None else None


def _name_general_fallback(smiles: str) -> str:
    """Name via the bb_conformance general_fallback config in a FRESH interpreter
    (mixing a general_fallback namer with default name_compound in one process
    pollutes global selection state — see test_v42_11c2_selection_pins.py)."""
    code = (
        "from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')\n"
        "from orthonym import Orthonym\n"
        "n=Orthonym(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)\n"
        f"print(n.name_tiered({smiles!r}).get('name'))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""


# ---------------------------------------------------------------------------
# #11 — multiplicative bridge cited BARE
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    ("O=C(O)C1CCC(OC2CCC(C(=O)O)CC2)CC1",
     "4,4'-oxydi(cyclohexane-1-carboxylic acid)"),
    ("O=S(=O)(O)c1ccc(Oc2ccc(S(=O)(=O)O)cc2)cc1",
     "4,4'-oxydi(benzene-1-sulfonic acid)"),
    # bonus: the same fix un-parenthesises the 'methylene' bridge.
    ("O=C(O)C1CCC(CC2CCC(C(=O)O)CC2)CC1",
     "4,4'-methylenedi(cyclohexane-1-carboxylic acid)"),
    # the simple 'disulfanediyl' bridge is bare too (comment in multiplicative.py).
    ("O=C(O)C1CCC(SSC2CCC(C(=O)O)CC2)CC1",
     "4,4'-disulfanediyldi(cyclohexane-1-carboxylic acid)"),
])
def test_simple_bridge_cited_bare(smiles, expected):
    assert name_compound(smiles) == expected


# ---------------------------------------------------------------------------
# #25 — euphonic connective 'a'. systematic_verified tier.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    # single en + single yn -> first ending 'en' vowel-initial -> elide 'a'.
    ("C1#C[SiH2][SiH2]C=C[SiH2][SiH2]1", "1,2,5,6-tetrasilacyclooct-3-en-7-yne"),
    # bare diyne -> first ending 'diyn' consonant-initial -> RETAIN 'a'.
    ("C1#C[SiH2][SiH2]C#C[SiH2][SiH2]1", "1,2,5,6-tetrasilacycloocta-3,7-diyne"),
])
def test_euphonic_a_first_ending_multiplied(smiles, expected):
    assert _name_general_fallback(smiles) == expected


# ---------------------------------------------------------------------------
# #34 — retained mancude ring 'phosphinine' as a substituent
# ---------------------------------------------------------------------------
def test_phosphinine_substituent():
    assert name_compound("N#Cc1cc(C2CCOC(C#N)C2)ccp1") == \
        "4-(2-cyanophosphinin-4-yl)oxane-2-carbonitrile"


@pytest.mark.parametrize("smiles,expected", [
    # regression: the phosphinine PARENT-ring rows (all MATCH) are untouched.
    ("C1=CPC=CC1", "1,4-dihydrophosphinine"),
    ("C1=CCPC=C1", "1,2-dihydrophosphinine"),
    ("c1pcpcp1", "1,3,5-triphosphinine"),
    ("c1ccc(-c2ccccp2)pc1", "2,2'-biphosphinine"),
])
def test_phosphinine_parent_unchanged(smiles, expected):
    assert name_compound(smiles) == expected


def test_saturated_phosphinane_ring_unchanged():
    """Regression: the SATURATED ring keeps its distinct replacement name; my
    aromatic-phosphinine registration must not touch it. (Emits only under the
    general_fallback best-effort tier, like bb_conformance.)"""
    assert _name_general_fallback("O=C(OP1CCCCC1)c1ccccc1") == \
        "1-phosphacyclohexan-1-yl benzoate"


# ---------------------------------------------------------------------------
# #27 — benzenylium, the Blue Book)
# ---------------------------------------------------------------------------
def test_benzenylium():
    assert name_compound("[C+]1=CC=CC=C1") == "benzenylium"


# ---------------------------------------------------------------------------
# #28 — hydroxyazanide, the Blue Book preselected)
# ---------------------------------------------------------------------------
def test_hydroxyazanide():
    assert name_compound("[NH-]O") == "hydroxyazanide"


@pytest.mark.parametrize("smiles,expected", [
    # regression: the NEUTRAL retained hydroxylamine name must stay (bare
    # parent, no N-carbon substituent).
    ("NO", "hydroxylamine"),
    # v52 P2, BB:38308/:38314): R-NH-OH is named as an
    # N-derivative of the senior amine ('N-hydroxymethanamine (PIN)
    # N-methylhydroxylamine' for the CH3 case), not the old functional-class
    # 'N-methylhydroxylamine' direction.
    ("CNO", "N-hydroxymethanamine"),
    # regression: the bare azanide anion is unchanged.
    ("[NH2-]", "azanide"),
])
def test_neutral_hydroxylamine_and_azanide_unchanged(smiles, expected):
    assert name_compound(smiles) == expected


# ---------------------------------------------------------------------------
# #26a — N-hydroxy... not N-hydroxyl...
# ---------------------------------------------------------------------------
def test_n_hydroxy_sulfonamide():
    assert name_compound("CS(=O)(=O)NO") == "N-hydroxymethanesulfonamide"


# ---------------------------------------------------------------------------
# #26b — (hydroxyimino) enclosing marks on the ring/seniority path
# ---------------------------------------------------------------------------
def test_ring_oxime_prefix_enclosed():
    assert name_compound("CC1(C(=O)O)C=CC(=NO)C=C1") == \
        "4-(hydroxyimino)-1-methylcyclohexa-2,5-diene-1-carboxylic acid"


@pytest.mark.parametrize("smiles,expected", [
    # regression: the acyclic oxime prefixes already carried parens; no double-wrap.
    ("CC(=O)C(C)=NO", "3-(hydroxyimino)butan-2-one"),
    ("CC(CC=O)=NO", "3-(hydroxyimino)butanal"),
    ("CCC(=O)C(C)=NO", "2-(hydroxyimino)pentan-3-one"),
    ("O=C(O)CCCC(O)=NO", "5-hydroxy-5-(hydroxyimino)pentanoic acid"),
])
def test_acyclic_oxime_prefix_unchanged(smiles, expected):
    assert name_compound(smiles) == expected


# ---------------------------------------------------------------------------
# #26c — one-carbon substituent parent omits redundant locants (a))
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("smiles,expected", [
    ("CCC(CC(C)=O)C([SiH3])[SiH3]", "4-(disilylmethyl)hexan-2-one"),
    # the same rule applies to any one-carbon parent (dichloromethyl).
    ("CCC(CC(C)=O)C(Cl)Cl", "4-(dichloromethyl)hexan-2-one"),
])
def test_one_carbon_parent_omits_locants(smiles, expected):
    assert name_compound(smiles) == expected


# ---------------------------------------------------------------------------
# #26d — λ5 on a 5-bonded organyl phosphanyl
# ---------------------------------------------------------------------------
def test_lambda5_phosphanyl():
    assert name_compound(
        "O=C(O)c1ccc(P(c2ccccc2)(c2ccccc2)(c2ccccc2)c2ccccc2)cc1"
    ) == "4-(tetraphenyl-λ5-phosphanyl)benzoic acid"


def test_lambda5_uses_shared_lambda_constant():
    """The λ must be the shared LAMBDA constant, not a hardcoded 'lambda'/'λ'."""
    name = name_compound(
        "O=C(O)c1ccc(P(c2ccccc2)(c2ccccc2)(c2ccccc2)c2ccccc2)cc1")
    assert f"{LAMBDA}5" in name
    assert "lambda5" not in name


@pytest.mark.parametrize("smiles,expected", [
    # regression: standard-valence (3-bond) phosphanyl carries NO λ.
    ("O=C(O)c1ccc(P(c2ccccc2)c2ccccc2)cc1", "4-(diphenylphosphanyl)benzoic acid"),
    ("O=C(O)c1ccc(P(C)C)cc1", "4-(dimethylphosphanyl)benzoic acid"),
])
def test_standard_phosphanyl_no_lambda(smiles, expected):
    assert name_compound(smiles) == expected


# ---------------------------------------------------------------------------
# round-trip proofs (0-wrong)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _OPSIN_OK, reason="OPSIN JVM required")
@pytest.mark.parametrize("smiles", [
    "O=C(O)C1CCC(OC2CCC(C(=O)O)CC2)CC1",
    "N#Cc1cc(C2CCOC(C#N)C2)ccp1",
    "[C+]1=CC=CC=C1",
    "[NH-]O",
    "CS(=O)(=O)NO",
    "CC1(C(=O)O)C=CC(=NO)C=C1",
    "CCC(CC(C)=O)C([SiH3])[SiH3]",
    "O=C(O)c1ccc(P(c2ccccc2)(c2ccccc2)(c2ccccc2)c2ccccc2)cc1",
])
def test_targets_roundtrip(smiles):
    from orthonym.validation.opsin_roundtrip import opsin_parse
    name = name_compound(smiles)
    osmi = opsin_parse(name)
    assert osmi and _ikey(osmi) == _ikey(smiles), f"{name} did not round-trip"
