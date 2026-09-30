"""Suite fix j7-defects-misc (TRIAGE.md 'Suite fix -- j7-defects-misc'): rows
with teeth for the job's producer fixes. Names are asserted exactly (gate on,
PIN tier unless stated) and each is checked by an independent full-InChIKey
OPSIN round trip (tests/support/rt_assert.py, not the engine's own gate).
"""
import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from tests.support.rt_assert import name_is_rt_exact
from tests.support.default_tier import (  # noqa: E402
    declined_pin_row,
    default_tier_rule_applies,
)

# Default tier: the paper, Methods, "Tiers" (L73): "The default configuration emits a
# name only when the pipeline can build the preferred IUPAC name (PIN); otherwise, it
# declines." User decision 2026-09-30 ("Ship it in 1.0.2"): a name the code records
# as not the PIN is declined at the default tier with NO_VERIFIED_PIN; for the
# molecules below the test asserts that decline, the strict path's name and label,
# and the same name at the best-effort tier (tests/support/default_tier.py).
DEFAULT_TIER_DECLINES = frozenset({
    "CC(C)(C)[Li]",
    "CC(C)(C)[Mg]Cl",
    "CC(C)=CCC[C@@H](C(=O)O)[C@H]1C(=O)C[C@@]2(C)C3=C(CC[C@]12C)[C@@]1(C)CCC(=O)C(C)(C)[C@@H]1[C@@H](O)C3",
    "C[C@H](CCC(=O)O)[C@H]1C[C@H](O)[C@@]2(C)C3=CCC4C(C)(C)C(=O)CC[C@]4(C)C3=CC[C@]12C",
    "[Li]C(C)(C)C",
})
#... whose best-effort name is another one (it reads back exactly)
BEST_EFFORT_NAMES_IT_OTHERWISE = frozenset()


def _declined_pin_row(smiles):
    return declined_pin_row(
        smiles, best_effort_same=smiles not in BEST_EFFORT_NAMES_IT_OTHERWISE)


def _dt_name_compound(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return name_compound(smiles)


def _dt_name(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)["name"]
    return Orthonym(style="pin").name(smiles)


def _dt_row(smiles):
    from orthonym import Orthonym, name_compound  # noqa: F811
    if smiles in DEFAULT_TIER_DECLINES and default_tier_rule_applies():
        return _declined_pin_row(smiles)
    return Orthonym(style="pin").name_tiered(smiles)



# ---------------------------------------------------------------------------
# g2 G2-C7: a junior anionic group on a carboxylate parent is cited by its
# anionic prefix, the Blue Book), and a site the prefix step
# cannot express fails closed instead of shipping the neutral prefix.
# -O-P(=O)(O-)2 -> 'phosphonatooxy': (:41213 "–P(O)(O–)2 phosphonato
# (preselected prefix)"), the anionic twin of 'phosphonooxy',
#:36333 "(phosphonooxy)acetic acid (PIN)").
# ---------------------------------------------------------------------------

PHOSPHONATOOXY_NAMES = [
    ("O=C([O-])[C@H](O)[C@H](O)COP(=O)([O-])[O-]",
     "(2R,3R)-2,3-dihydroxy-4-(phosphonatooxy)butanoate"),
    ("O=C([O-])CCOP(=O)([O-])[O-]", "3-(phosphonatooxy)propanoate"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", PHOSPHONATOOXY_NAMES,
                         ids=[e for _, e in PHOSPHONATOOXY_NAMES])
def test_phosphonatooxy_junior_anion(smiles, expected):
    r = _dt_row(smiles)
    assert r["name"] == expected
    assert r["tier"] == "pin_verified"
    assert name_is_rt_exact(expected, smiles)


@pytest.mark.parametrize("smiles", [
    # the trianion: the only thing between the neutral-prefix name and a
    # shipped wrong species was the validity gate
    "O=C([O-])[C@H](O)[C@H](O)COP(=O)([O-])[O-]",
    # -O-P(=O)(OH)O-: one acid O left, not a 'phosphonato' group; no producer
    "O=C([O-])COP(=O)(O)[O-]",
])
def test_gate_off_never_ships_the_charge_dropping_name(smiles):
    """Gate off (the suite default): whatever ships must still be the input
    species; the old '(phosphonooxy)...ate' denoted a less anionic molecule."""
    name = _dt_name(smiles)
    assert "phosphonooxy" not in name
    if name and name != "unknown organic compound":
        assert name_is_rt_exact(name, smiles), name


def test_unexpressible_junior_site_returns_empty():
    """Producer contract: '' (fail closed), never the unconverted name."""
    from orthonym.perception.ions import get_ion_sites
    from orthonym.rules.ions import _apply_anionic_substituent_prefixes
    smi = "O=C([O-])COP(=O)(O)[O-]"
    mol = Chem.MolFromSmiles(smi)
    anions = get_ion_sites(mol)["anions"]
    assert _apply_anionic_substituent_prefixes(mol, anions, "(phosphonooxy)acetate") == ""


# ---------------------------------------------------------------------------
# g5 C13: the partial acid salt method (1), the Blue Book,
# 'ammonium 3-carboxypropanoate (PIN)':31606) carries the E/Z of a backbone
# double bond,:46740); it used to drop it at the gate-off raw
# level and abstain at the PIN tier.
# ---------------------------------------------------------------------------

PARTIAL_ACID_SALT_EZ = [
    ("O=C([O-])/C=C/C(=O)O.[Na+]", "sodium (2E)-3-carboxyprop-2-enoate"),
    ("O=C([O-])/C=C\\C(=O)O.[Na+]", "sodium (2Z)-3-carboxyprop-2-enoate"),
    ("OC(=O)C/C=C/CC(=O)[O-].[K+]", "potassium (3E)-5-carboxypent-3-enoate"),
    ("O=C([O-])/C=C/C(=O)O", "(2E)-3-carboxyprop-2-enoate"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", PARTIAL_ACID_SALT_EZ,
                         ids=[e for _, e in PARTIAL_ACID_SALT_EZ])
def test_partial_acid_salt_keeps_backbone_ez(smiles, expected):
    r = _dt_row(smiles)
    assert r["name"] == expected
    assert r["tier"] == "pin_verified"
    assert name_is_rt_exact(expected, smiles)


@pytest.mark.parametrize("smiles,expected", PARTIAL_ACID_SALT_EZ[:2])
def test_partial_acid_salt_ez_gate_off(smiles, expected):
    assert _dt_name(smiles) == expected


# ---------------------------------------------------------------------------
# g2 G2-C6 / g7 C12: steroid stereoparent names.
# * a terminal -COOH on the stereoparent is the '-oic acid' suffix,
# the Blue Book, '3-oxoandrost-4-en-18-oic acid':52554;:18158),
# never '26-hydroxy...-26-one';
# * a double bond whose end locants differ by more than one takes the compound
# locant (1),:16634): 9(11), 24(28);
# * every steroid numbering map follows the skeleton bonds (C-9-C-11, C-11-C-12,
# C-12-C-13); the estrane, cholane, ergostane, campestane and stigmastane maps had
# C-11/C-12 swapped;
# * a stereocentre the parent name implies:51047) that the input leaves
# undefined declines the stereoparent name .
# ---------------------------------------------------------------------------

STEROID_ACID_NAMES = [
    # (smiles, name, tier at the PIN tier). The last two carry a whole-graph R/S
    # descriptor block (DK-P101CIP, recorded non-PIN): the PIN path built them and
    # the gate verified them (; name_is_rt_exact below), and the code records
    # them as not the PIN, so systematic_verified -- "a correct systematic name that
    # is not the PIN" (user decision 2026-09-30; Methods, "Tiers").
    ("C[C@H](CCC(=O)O)[C@H]1CC[C@H]2[C@@H]3CC[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
     "3α-hydroxy-5β-cholan-24-oic acid", "pin_verified"),
    ("C[C@H](CCC[C@H](C)C(=O)O)[C@H]1CC[C@H]2[C@@H]3CC[C@H]4CC(=O)CC[C@]4(C)[C@H]3CC[C@]12C",
     "(25S)-3-oxo-5α-cholestan-26-oic acid", "pin_verified"),
    ("C[C@H](CCC(=O)O)[C@H]1C[C@H](O)[C@@]2(C)C3=CCC4C(C)(C)C(=O)CC[C@]4(C)C3=CC[C@]12C",
     "(10S,13R,14R,15S,17R,20R)-15-hydroxy-4,4,14-trimethyl-3-oxochola-7,9(11)-dien-24-oic acid",
     "systematic_verified"),
    ("CC(C)=CCC[C@@H](C(=O)O)[C@H]1C(=O)C[C@@]2(C)C3=C(CC[C@]12C)[C@@]1(C)CCC(=O)C(C)(C)[C@@H]1[C@@H](O)C3",
     "(5R,6S,10S,13R,14R,17R,20R)-6-hydroxy-4,4,14-trimethyl-3,16-dioxocholesta-8,24-dien-21-oic acid",
     "systematic_verified"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected,tier", STEROID_ACID_NAMES,
                         ids=[e for _, e, _ in STEROID_ACID_NAMES])
def test_steroid_terminal_acid_is_the_suffix(smiles, expected, tier):
    r = _dt_row(smiles)
    assert r["name"] == expected
    assert r["tier"] == tier
    assert name_is_rt_exact(expected, smiles)


def test_steroid_numbering_maps_follow_the_skeleton():
    from orthonym.data.natural_products import STEROID_NUMBERING_MAPS
    skeleton = {(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
                (1, 10), (5, 10), (8, 14), (9, 11), (11, 12), (12, 13), (13, 14), (14, 15),
                (15, 16), (16, 17), (13, 17), (13, 18), (10, 19), (17, 20), (20, 21),
                (20, 22), (22, 23), (23, 24), (24, 25), (25, 26), (25, 27), (24, 28),
                (28, 29)}
    for smi, numbering in STEROID_NUMBERING_MAPS.items():
        mol = Chem.MolFromSmiles(smi)
        for b in mol.GetBonds():
            a, c = numbering[b.GetBeginAtomIdx()], numbering[b.GetEndAtomIdx()]
            assert (min(a, c), max(a, c)) in skeleton, (smi, a, c)


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("C/C=C(/CC[C@@H](C)[C@H]1CC[C@H]2C3=CC[C@H]4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C)C(C)C",
     "(24Z)-5α-stigmasta-7,24(28)-dien-3β-ol"),
    ("C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1[C@@H](O)[C@@H](O)[C@@H]2O",
     "estra-1,3,5(10)-triene-3,15α,16α,17β-tetrol"),
])
def test_steroid_compound_locant_and_alpha_beta(smiles, expected):
    r = _dt_row(smiles)
    assert r["name"] == expected
    assert name_is_rt_exact(expected, smiles)


@pytest.mark.parametrize("smiles", [
    #: C-10/13/17/20 undefined
    "CC(CC(=O)CC(C)C1C[C@H](O)[C@@]2(C)C3=C(C(=O)CC12C)C1(C)CC[C@H](O)C(C)(C)C1C[C@@H]3O)C(=O)O",
    # androstan-3-one with only C-13 and C-17 defined
    "C[C@]12CCC3C(CCC4CC(=O)CCC34C)C1CC[C@@H]2O",
])
def test_undefined_implied_centre_declines_the_stereoparent(smiles):
    """Gate off: the raw producer never over-specifies an implied centre."""
    name = _dt_name(smiles)
    assert not any(stem in name for stem in ("cholest", "androst"))
    if name != "unknown organic compound":
        assert name_is_rt_exact(name, smiles), name


# ---------------------------------------------------------------------------
# g6 C07 / g7 C17: the fused-catalog tautomer guard counted skeleton-saturated
# atoms (a bridge CH2, a three-connected fusion N) as indicated-hydrogen
# positions and rejected two entries on their own reference tautomer
#, the Blue Book: indicated hydrogen only where the mancude
# parent would otherwise carry a double bond). g6 C09: numeric locants are ints.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("C1=C2COc3coc(c31)CO2", "2H-3,5-(epoxymethano)furo[3,4-b]pyran"),
    ("Cc1oc2c3c1OCC(=C3)OC2", "2H-3,5-(epoxymethano)furo[3,4-b]pyran"),
    ("C1=CN2C=CC3=CNOC3=C2O1", "2H-[1,2]oxazolo[5,4-c][1,3]oxazolo[3,2-a]pyridine"),
    # a real tautomer swap is still re-anchored : the 3H thienoimidazole
    ("c1[nH]c2sccc2n1", None),
])
def test_fused_catalog_tautomer_guard(smiles, expected):
    from orthonym.data.fused_heterocycles import match_fused_heterocycle_core
    r = match_fused_heterocycle_core(Chem.MolFromSmiles(smiles))
    if expected is None:
        assert r is None or not r[0].startswith("1H-"), r
    else:
        assert r is not None and r[0] == expected, r


def test_skeleton_saturated_set_needs_a_consistent_entry():
    from orthonym.data.fused_heterocycles import _skeleton_saturated_locants
    assert _skeleton_saturated_locants("C1=C2COc3coc(c31)CO2", {2}) == {8}
    assert _skeleton_saturated_locants("C1=CN2C=CC3=CNOC3=C2O1", {2}) == {6}
    # keyed on an NH at locant 7 while the name says 1H: contributes nothing
    assert _skeleton_saturated_locants("c1ncc2[nH]cnc2n1", {1}) == frozenset()


# ---------------------------------------------------------------------------
# g3 C12: the Group-1/2 sigma-ligand recogniser names tert-butyl,
# the Blue Book '*tert*-butyldi(methyl)phosphane (PIN)'), fail closed on
# other branched ligands.
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("CC(C)(C)[Li]", "tert-butyllithium"),
    ("[Li]C(C)(C)C", "tert-butyllithium"),
    ("CC(C)(C)[Mg]Cl", "tert-butylmagnesium chloride"),
])
def test_tert_butyl_sigma_ligand(smiles, expected):
    name = _dt_name(smiles)
    assert name == expected
    assert name_is_rt_exact(name, smiles)


def test_other_branched_ligand_still_fails_closed():
    from rdkit import Chem as _C
    from orthonym.rules.organometallics import _ligand_name_from_atoms
    m = _C.MolFromSmiles("CC(C)[Li]")          # isopropyl: no recogniser
    assert _ligand_name_from_atoms(m, (0, 1, 2)) is None
    m = _C.MolFromSmiles("C=C(C)C[Li]")        # 2-methylallyl: multiple bond
    assert _ligand_name_from_atoms(m, (0, 1, 2, 3)) is None


# ---------------------------------------------------------------------------
# g7 C20: the italicized-prefix carve-out at three sites that
# decided with a raw hyphen test: a simple 'tert-butyl' takes a hyphenated
# plain multiplier ('1,2-di-*tert*-butylbenzene (PIN)', the Blue Book).
# ---------------------------------------------------------------------------

@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", [
    ("CC(C)(C)OP(OC(C)(C)C)OC(C)(C)C", "tri-tert-butyl phosphite"),
    ("CC(C)(C)OP(=O)(OC(C)(C)C)OC(C)(C)C", "tri-tert-butyl phosphate"),
    ("[Mg+2].CC(C)(C)[O-].CC(C)(C)[O-]", "magnesium di-tert-butoxide"),
    ("CCOP(OCC)OCC", "triethyl phosphite"),
    ("[Ca+2].CC(=O)[O-].CC(=O)[O-]", "calcium diacetate"),
])
def test_italicized_prefix_multiplier(smiles, expected):
    name = _dt_name(smiles)
    assert name == expected
    assert name_is_rt_exact(name, smiles)


# ---------------------------------------------------------------------------
# g3 C05 / g5 C15: a tripeptide takes the substitutive PIN -- the nested amido
# prefix is built from the substitutive name of its dipeptide acid fragment
# method (1), the Blue Book; controller ruling: peptide
# names are not PINs,:50943) -- and a substituted 2-carbon acid keeps the
# retained 'acetic acid' when its only stereocentres lie inside the prefix
#,:29717 /:29725;:3031).
# ---------------------------------------------------------------------------

PEPTIDE_PINS = [
    ("C[C@H](NC(=O)[C@H](C)NC(=O)[C@@H]1CCCN1)C(=O)O",
     "(2S)-2-{(2S)-2-[(2S)-pyrrolidine-2-carboxamido]propanamido}propanoic acid"),
    ("CC(C)[C@H](NC(=O)[C@@H](N)[C@@H](C)O)C(=O)N[C@@H](CCCCN)C(=O)O",
     "(2S)-6-amino-2-{(2S)-2-[(2S,3R)-2-amino-3-hydroxybutanamido]-3-methylbutanamido}"
     "hexanoic acid"),
    ("NCC(=O)NCC(=O)NCC(=O)O", "[2-(2-aminoacetamido)acetamido]acetic acid"),
    ("N[C@@H](C)C(=O)NCC(=O)O", "[(2S)-2-aminopropanamido]acetic acid"),
    ("OC(=O)CNC(=O)[C@@H]1CCCN1", "[(2S)-pyrrolidine-2-carboxamido]acetic acid"),
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,expected", PEPTIDE_PINS, ids=[e[:40] for _, e in PEPTIDE_PINS])
def test_peptide_substitutive_pin(smiles, expected):
    r = _dt_row(smiles)
    assert r["name"] == expected
    assert r["tier"] == "pin_verified"
    assert name_is_rt_exact(expected, smiles)


@pytest.mark.opsin_gate
def test_chain_stereocentre_keeps_the_retained_acid():
    """A stereocentre ON the 2-carbon chain keeps the retained parent too; its
    descriptor is cited bare on the locant-free parent (TRIAGE j12 findings
    2/4/7): (the Blue Book) 'acetic acid (PIN) ethanoic acid';
     (:44643) descriptors take a locant "when such locants are present";
    '(R)-{bis[(1R)-1-hydroxyethyl]amino}{...}acetic acid (PIN)' (:45731),
    '(S)-cyclopropyl(hydroxy)acetaldehyde (PIN)' (:45259). Was '(2S)-2-hydroxy-2-
    phenylethanoic acid' at pin_verified."""
    smiles = "O[C@@H](c1ccccc1)C(=O)O"
    r = _dt_row(smiles)
    assert r["name"] == "(S)-hydroxy(phenyl)acetic acid", r
    assert r["tier"] == "pin_verified", r
    assert name_is_rt_exact(r["name"], smiles)
