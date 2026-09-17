"""
Wave2 Tier 5a — multiplicative substituted-unit numbering + composite bridges.

Blue Book rules exercised (all verbatim-verified against the Blue Book):

  * — SUBSTITUTED identical units take bis/tris + enclosure:
    "4,4'-oxybis(2-chlorobenzoic acid)" (PIN, the example). The unit's
    prefix locants are assigned on the FREE unit by the fragment namer, so the
    joint numbering is only direction-invariant at a para attachment on a
    benzene unit — every other substituted-unit shape fails closed
    (_ring_unit_direction_safe).
  * — units with mere suffix locants keep di/tri + parentheses
    ("2,2'-oxydi(ethan-1-ol)"); locant-free units stay bare ("oxydiacetic").
  * (3) — both units' attachment locants must be identical: the
    para/meta mixed diether declines structurally (jar-independent).
  * Composite bridge "[ethane-1,2-diylbis(oxy)]" (BB verbatim:
    "2,2'-[ethane-1,2-diylbis(oxy)]diacetic acid" (PIN)); square brackets per
    the nesting order.
  * Acyclic N connectors: azanediyl (NH) and nitrilo (N<) —
    "2,2',2''-nitrilotri(ethan-1-ol)" (BB verbatim PIN, triethanolamine),
    "2,2'-azanediyldiacetic acid" (iminodiacetic; BB sibling
    "3,3'-azanediyldipropanenitrile" (PIN)). Amine arms are allowed for the
    CHALCOGEN bridges only (BB "2,2'-oxydi(ethan-1-amine)" (PIN)) — never for
    N bridges, where the substitutive polyamine parent is the PIN.
  * Central-arene alcohol arms: "(benzene-1,3,5-triyl)trimethanol"
    (mononuclear methanol units carry no locants per, cf. BB
    "[oxydi(pyridazine-4,3,5-triyl)]tetramethanol").
  * hyphenated italic prefixes under a multiplier:
    "1,2-di-tert-butylbenzene" (PIN) — never "ditert-butyl".

Retained-name demotions (pin:false, acetophenone precedent):
triethanolamine, diethanolamine, terephthalyl alcohol.
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.rules.multiplicative import name_multiplicative


# ---------------------------------------------------------------------------
# New multiplicative heals — each was 'unknown' or a non-PIN form at HEAD.
# All OPSIN round-trip verified.
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        # substituted units -> bis (BB verbatim)
        ("OC(=O)c1ccc(Oc2ccc(C(=O)O)c(Cl)c2)cc1Cl",
         "4,4'-oxybis(2-chlorobenzoic acid)"),
        # composite bridge (BB verbatim)
        ("OC(=O)COCCOCC(=O)O",
         "2,2'-[ethane-1,2-diylbis(oxy)]diacetic acid"),
        # triethylene glycol
        ("OCCOCCOCCO",
         "2,2'-[ethane-1,2-diylbis(oxy)]di(ethan-1-ol)"),
        # acyclic azanediyl (NH) bridge
        ("OC(=O)CNCC(=O)O", "2,2'-azanediyldiacetic acid"),
        ("OCCNCCO", "2,2'-azanediyldi(ethan-1-ol)"),
        # acyclic nitrilo (N<) star
        ("N(CC(=O)O)(CC(=O)O)CC(=O)O", "2,2',2''-nitrilotriacetic acid"),
        ("OCCN(CCO)CCO", "2,2',2''-nitrilotri(ethan-1-ol)"),
        ("N(CCC(=O)O)(CCC(=O)O)CCC(=O)O",
         "3,3',3''-nitrilotripropanoic acid"),
        # amine arms on chalcogen bridges (BB verbatim)
        ("NCCOCCN", "2,2'-oxydi(ethan-1-amine)"),
        ("NCCSCCN", "2,2'-sulfanediyldi(ethan-1-amine)"),
        # central-arene alcohol arms (mononuclear units, no locants)
        ("OCc1cc(CO)cc(CO)c1", "(benzene-1,3,5-triyl)trimethanol"),
        ("OCc1ccc(CO)cc1", "(1,4-phenylene)dimethanol"),
    ],
)
def test_tier5a_multiplicative_heals(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        #: multiplier joins an italicized prefix with a hyphen
        ("CC(C)(C)c1ccccc1C(C)(C)C", "1,2-di-tert-butylbenzene"),
        ("CC(C)(C)c1cccc(C(C)(C)C)c1O", "2,6-di-tert-butylphenol"),
    ],
)
def test_tier5a_di_tert_butyl_hyphenation(smiles, expected):
    assert name_compound(smiles) == expected


# ---------------------------------------------------------------------------
# Retained demotions: the multiplicative PIN replaces the general-only name.
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_tier5a_retained_demotions_filtered():
    from orthonym.data import ALL_RETAINED_NAMES, GENERAL_RETAINED_NAMES
    for smi, general_name in [
        ("OCCN(CCO)CCO", "triethanolamine"),
        ("OCCNCCO", "diethanolamine"),
        ("OCc1ccc(CO)cc1", "terephthalyl alcohol"),
    ]:
        canon = Chem.CanonSmiles(smi)
        assert ALL_RETAINED_NAMES.get(canon) != general_name
        assert GENERAL_RETAINED_NAMES.get(canon) == general_name


# ---------------------------------------------------------------------------
# Regression controls — shipped multiplicative behavior must not move.
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("OC(=O)COCC(=O)O", "2,2'-oxydiacetic acid"),
        ("OC(=O)CCOCCC(=O)O", "3,3'-oxydipropanoic acid"),
        ("OCCOCCO", "2,2'-oxydi(ethan-1-ol)"),
        ("OCCSCCO", "2,2'-sulfanediyldi(ethan-1-ol)"),
        ("OC(=O)c1ccc(Oc2ccc(C(=O)O)cc2)cc1", "4,4'-oxydibenzoic acid"),
        ("O(c1ccccc1)c1ccccc1", "1,1'-oxydibenzene"),
        ("Oc1ccc(Cc2ccc(O)cc2)cc1", "4,4'-methylenediphenol"),
        ("Nc1ccc(Cc2ccc(N)cc2)cc1", "4,4'-methylenedianiline"),
        ("OC(=O)c1ccc(CCc2ccc(C(=O)O)cc2)cc1",
         "4,4'-(ethane-1,2-diyl)dibenzoic acid"),
        ("OC(=O)c1ccc(OOc2ccc(C(=O)O)cc2)cc1", "4,4'-peroxydibenzoic acid"),
        ("Oc1ccc(N(c2ccc(O)cc2)c2ccc(O)cc2)cc1", "4,4',4''-nitrilotriphenol"),
        ("OC(=O)Cc1cc(CC(=O)O)cc(CC(=O)O)c1",
         "2,2',2''-(benzene-1,3,5-triyl)triacetic acid"),
        # C4 guard: NH between plain benzenes stays substitutive
        ("c1ccc(Nc2ccccc2)cc1", "N-phenylaniline"),
        # simple secondary amine: never a multiplicative azanediyl target
        ("CCNCC", "N-ethylethanamine"),
    ],
)
def test_tier5a_regression_controls(smiles, expected):
    assert name_compound(smiles) == expected


# ---------------------------------------------------------------------------
# Fail-closed guards (name_multiplicative-level: the handler must decline).
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    "smiles,why",
    [
        ("OC(=O)c1ccc(Oc2cccc(C(=O)O)c2)cc1",
         "para/meta mixed attachment: P-15.3.1.1(3) locants differ"),
        ("CN(CC(=O)O)CC(=O)O",
         "methylazanediyl (substituted N bridge) — deferred, fail closed"),
        ("CN(CCO)CCO", "N-methyl diethanolamine — methyl arm has no PCG"),
        ("NCCNCCN",
         "N bridge + amine arms: substitutive polyamine parent is the PIN"),
        ("CNCC(=O)O", "sarcosine: methyl arm has no PCG"),
        ("NCCOCCO", "asymmetric arms (amine vs ol)"),
        ("OCCOCC(=O)O", "asymmetric arms (ol vs acid)"),
        ("CCOCC", "plain ether: no PCG on arms"),
        ("OC(=O)COC(C)C(=O)O", "asymmetric arms (methyl-branched side)"),
        ("O(/C=C/C(=O)O)/C=C/C(=O)O",
         "unsaturated arms: the saturated-chain arm name would drop C=C"),
        # EDTA (the composite diyldinitrilo bridge) is no longer deferred: it is
        # built by _try_diamine_dinitrilo_bridge and covered with an OPSIN
        # round-trip in tests/unit/rules/test_edta.py.
        ("OCCOCCOCCOCCO",
         "tetraethylene glycol: arm carries a second ether O — deferred"),
    ],
)
def test_tier5a_fail_closed(smiles, why):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"bad test SMILES: {smiles}"
    assert name_multiplicative(mol) is None, (
        f"expected fail-closed None for {smiles} ({why})"
    )


# ---------------------------------------------------------------------------
# _select_multiplier unit behavior (leading-locant rule).
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize(
    "parent,count,expected",
    [
        ("benzoic acid", 2, "di"),
        ("ethan-1-ol", 2, "di"),
        ("cyclohexane-1-carboxylic acid", 2, "di"),
        ("benzene-1,4-diamine", 2, "di"),       # mid-name comma:
        ("2-chlorobenzoic acid", 2, "bis"),     # substituted:
        ("4-bromobenzene", 2, "bis"),
        ("1,3-thiazole", 2, "bis"),             # leading locant:
        ("N-methylmethanamine", 2, "bis"),
        ("acetic acid", 3, "tri"),
    ],
)
def test_tier5a_select_multiplier(parent, count, expected):
    from orthonym.rules.multiplicative import _select_multiplier
    assert _select_multiplier(parent, count) == expected


@pytest.mark.unit
def test_tier5a_dec_stem_units_take_parens():
    """(d): unit names beginning with 'dec' are enclosed so
    'di(decanoic acid)' cannot read as 'didecanoic acid'."""
    from orthonym.rules.multiplicative import _assemble_multiplicative_name
    name = _assemble_multiplicative_name(10, "oxy", "decanoic acid")
    assert name == "10,10'-oxydi(decanoic acid)"
