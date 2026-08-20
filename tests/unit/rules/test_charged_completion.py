"""v33 charged Slice B — PIN-correct azaniumyl zwitterion namer (P-74.2.1.2).

Corrects the 4782742f over-reach (a zwitterion shipped its NON-PIN neutral name
on the DEFAULT/PIN path) and builds the P-74.2.1.2 IONIC PIN instead: the ANION
is the parent (keeps its -oate/... suffix) and each PROTONATED-amine cation is an
`azaniumyl` prefix (two or more of the same kind -> `bis(azaniumyl)` /
`tris(azaniumyl)`, P-16.3.4). Extends the namer to NET-CHARGED mixed-sign species
(protonated diamino-acids) via a new Tier-1 dispatch door
(`routing.dispatch_table._is_mixed_sign_zwitterion`, priority 302) -- those have
net charge != 0, so `detect_species_type` buckets them as 'ion' and ZWITTERION@300
never sees them.

Design (see `charged_router._name_primary_amine_azaniumyl_zwitterion`):
  * The betaine SEVER path (P-74.1.3) cannot serve a PRIMARY amino-acid zwitterion
    -- severing the -NH3+ and capping its carbon with H DESTROYS the alpha
    stereocentre, so it could never spell the (2R)/(2S) descriptor. The builder
    instead NEUTRALIZES in place (-NH3+ -> -NH2, -COO- -> -COOH), names the neutral
    amino acid (keeping stereo + numbering), converts acid -> carboxylate anion,
    and re-expresses the grouped `amino` prefix as `azaniumyl`.
  * 0-wrong ABSOLUTE: every emission is full-InChIKey RT-gated (charges + stereo)
    inside the producer; a mismatch fails CLOSED. Net charge is preserved (the
    species is never neutralized in the emitted name).
  * PIN-first: the DEFAULT/PIN tier emits azaniumyl or ABSTAINS -- never the
    non-PIN neutral name (that fallback is best-effort-only).
  * The standard-AA retained table (P-103.2.4.4) still fires FIRST (before
    route_charged), so D-leucine / glycine / L-alanine are unchanged.

Every RT-verified target below was checked against OPSIN 2.9.0 + RDKit full
InChIKey during development (see impl report).
"""
import pytest
from rdkit import Chem

from orthonym.namer import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse

pytestmark = pytest.mark.unit

G = Orthonym()                                   # default: PIN tier, gate on
RAW = Orthonym(_disable_opsin_validity_gate=True)  # gate off -> proves producer RT gate


def _full_ik_rt(smi, name):
    """OPSIN-parse `name` and compare the FULL standard InChIKey (constitution +
    charge + stereo) to the input's -- the 0-wrong contract for every shipped name."""
    assert name and "unknown" not in name, f"expected a real name, got {name!r}"
    g = opsin_parse(name)
    assert g, f"OPSIN could not parse {name!r}"
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(g)) == \
        Chem.MolToInchiKey(Chem.MolFromSmiles(smi)), \
        f"full-InChIKey RT mismatch for {name!r}"


# --- 0. preserved Slice-B1/B2 tests (retained-table stereo audit + best-effort
# net-zero AA round-trip; kept from the original file, still valid) -----------
def test_retained_aa_zwitterion_table_is_stereo_correct():
    from orthonym.rules.salts import RETAINED_AMINO_ACID_ZWITTERIONS
    bad = []
    for smi, name in RETAINED_AMINO_ACID_ZWITTERIONS.items():
        g = opsin_parse(name)
        ok = g and Chem.MolToInchiKey(Chem.MolFromSmiles(g)) == \
            Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
        if not ok:
            bad.append((smi, name, g))
    assert not bad, f"mislabeled table entries: {bad}"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", [
    "[NH3+][C@@H](CC[SeH])C(=O)[O-]",   # selenohomocysteine zwitterion
    "CC(C)C[C@@H]([NH3+])C(=O)[O-]",     # leucine zwitterion (retained path)
])
def test_netzero_aa_zwitterion_names_and_roundtrips(smi):
    # Best-effort tier: any name that full-InChIKey round-trips is acceptable
    # (selenohomocysteine now -> azaniumyl PIN; leucine -> retained D-leucine).
    n = Orthonym(general_fallback=True, general_fallback_unverified=True,
                  allow_aromatic_general=True).name(smi)
    assert n and n != "unknown organic compound"
    g = opsin_parse(n)
    assert g and Chem.MolToInchiKey(Chem.MolFromSmiles(g)) == \
        Chem.MolToInchiKey(Chem.MolFromSmiles(smi))


# --- 1. standard AA retained table unchanged (P-103.2.4.4, fires first) ----
@pytest.mark.opsin_gate
def test_leucine_zwitterion_retained_unchanged():
    # D-leucine is a Table-10.4 standard AA -> the retained name (fired before
    # route_charged); azaniumyl only fires for NON-standard / net-charged.
    assert G.name("CC(C)C[C@@H]([NH3+])C(=O)[O-]") == "D-leucine"


# --- 2. single-cation net-0 amino-acid zwitterion -> azaniumyl PIN ----------
@pytest.mark.opsin_gate
def test_s_methylcysteine_azaniumyl_default():
    # BB P-103.2.4.4 example. Default path must be the ionic PIN, NOT the neutral
    # acid (the 4782742f over-reach). Descriptor is (2R) for this SMILES.
    smi = "[NH3+][C@@H](CSC)C(=O)[O-]"
    name = G.name(smi)
    assert name == "(2R)-2-azaniumyl-3-(methylsulfanyl)propanoate"
    assert "propanoic acid" not in name  # never the neutral name on default
    _full_ik_rt(smi, name)


@pytest.mark.opsin_gate
def test_s_methylcysteine_azaniumyl_gate_off():
    # RAW (gate off) proves the PRODUCER's own full-InChIKey RT gate, not SELF-01.
    smi = "[NH3+][C@@H](CSC)C(=O)[O-]"
    name = RAW.name(smi)
    assert name == "(2R)-2-azaniumyl-3-(methylsulfanyl)propanoate"
    _full_ik_rt(smi, name)


@pytest.mark.opsin_gate
def test_selenohomocysteine_azaniumyl_default():
    # Se changes the alpha-carbon CIP priority -> (2S) here (vs (2R) for the S
    # analogue): the descriptor is derived per-molecule (neutralize-in-place),
    # never hardcoded, and RT-gated.
    smi = "[NH3+][C@@H](CC[SeH])C(=O)[O-]"
    name = G.name(smi)
    assert name == "(2S)-2-azaniumyl-4-selanylbutanoate"
    _full_ik_rt(smi, name)


# --- 3. multi-cation -> bis(azaniumyl); net-charged door --------------------
@pytest.mark.opsin_gate
def test_diammonium_pentanoate_net_charged_door():
    # NET +1 (protonated 2,5-diaminopentanoate): detect_species_type -> 'ion',
    # so ZWITTERION@300 never sees it; the new MIXED_SIGN_ZWITTERION@302 dispatch
    # door routes it to the azaniumyl builder. Net charge must be PRESERVED
    # (never neutralized to 'ornithine', the old gate-suppressed wrong name).
    smi = "[NH3+]CCCC([NH3+])C(=O)[O-]"
    name = G.name(smi)
    assert name == "2,5-bis(azaniumyl)pentanoate"
    _full_ik_rt(smi, name)
    assert Chem.GetFormalCharge(Chem.MolFromSmiles(opsin_parse(name))) == 1


@pytest.mark.opsin_gate
def test_two_cation_diacid_bis_azaniumyl():
    # NET 0, 2 primary -NH3+ + 2 carboxylate -> 2,3-bis(azaniumyl)pentanedioate
    # (replaces the 4782742f neutral 2,3-diaminopentanedioic acid).
    smi = "[NH3+]C(CC(=O)[O-])C([NH3+])C(=O)[O-]"
    name = G.name(smi)
    assert name == "2,3-bis(azaniumyl)pentanedioate"
    _full_ik_rt(smi, name)


# --- 4. PIN-first: default abstains rather than ship a non-PIN neutral name --
def test_default_never_ships_neutral_zwitterion_name():
    # A sulfonate amino-acid zwitterion (taurine) is out of the carboxylate-only
    # azaniumyl builder's scope -> route_charged declines. On the DEFAULT tier the
    # contract is azaniumyl-or-ABSTAIN, so the non-PIN neutral name must NOT ship.
    from orthonym.errors import is_failure_name
    out = G.name("[NH3+]CCS(=O)(=O)[O-]")
    assert out is not None
    assert is_failure_name(out) or "azaniumyl" in out


@pytest.mark.opsin_gate
def test_best_effort_tier_still_degrades_to_neutral():
    # The neutral systematic name remains reachable ABOVE the PIN tier (T4:
    # degrade a table/PIN miss to an uglier name, never to silence).
    BE = Orthonym(general_fallback=True, general_fallback_unverified=True,
                   allow_aromatic_general=True)
    out = BE.name("[NH3+]CCS(=O)(=O)[O-]")
    assert out and "unknown" not in out


# --- 5. fail-closed: 0-wrong on out-of-scope shapes -------------------------
@pytest.mark.opsin_gate
def test_mixed_anion_zwitterion_failclosed():
    # 1 cation + carboxylate + sulfonate: not every anion is a carboxylate, so the
    # builder declines; a wrong/fabricated name must never ship.
    from orthonym.errors import is_failure_name
    smi = "[NH3+]C(CS(=O)(=O)[O-])C(=O)[O-]"
    out = G.name(smi)
    assert out is not None
    assert is_failure_name(out), f"expected honest abstain, got {out!r}"


def test_builder_declines_quaternary_cation_directly():
    # Unit-level (no JVM): a quaternary betaine (0 H on N) is NOT a primary amine,
    # so the azaniumyl builder declines it -- the existing SEVER path owns it.
    from orthonym.perception.ions import get_ion_sites
    from orthonym.rules.charged_router import _name_primary_amine_azaniumyl_zwitterion
    mol = Chem.MolFromSmiles("C[N+](C)(C)CCC(=O)[O-]")
    sites = get_ion_sites(mol)
    assert _name_primary_amine_azaniumyl_zwitterion(
        mol, sites['cations'], sites['anions'], 'pin') == ''


def test_builder_declines_extra_nitrogen_directly():
    # Unit-level: an extra (non-cation) nitrogen would leave a stray `amino` token
    # the re-expression could mis-rewrite -> the "no other N" scope declines.
    from orthonym.perception.ions import get_ion_sites
    from orthonym.rules.charged_router import _name_primary_amine_azaniumyl_zwitterion
    # 2,5-diaminopentanamide-like: a carboxamide N in addition to the -NH3+.
    mol = Chem.MolFromSmiles("NC(=O)C[C@@H]([NH3+])C(=O)[O-]")
    sites = get_ion_sites(mol)
    assert _name_primary_amine_azaniumyl_zwitterion(
        mol, sites['cations'], sites['anions'], 'pin') == ''


# =============================================================================
# Task C1 (v33 charged completion): thiolate `sulfido` prefix + cysteinate
# multi-anion fallback.
#
# `classify_anion` (ions.py:112) already returns 'thiolate' for a terminal
# C-bonded [S-]; `_apply_anionic_substituent_prefixes` (ions.py:~3986) did NOT
# know how to cite a JUNIOR thiolate centre -- the neutral-acid path built
# 'sulfanyl...' (the neutral -SH substituent prefix), which round-trips to a
# DIFFERENT (neutral-thiol) molecule and SELF-01 correctly rejected it.
# FIX 3a mirrors the existing sulfinato/sulfonato/phosphonato swap blocks with
# a 'sulfanyl'->'sulfido' single-occurrence swap.
#
# Cysteinate ('N[C@@H](C[S-])C(=O)[O-]') additionally stacks a SECOND defect:
# the multi-anion branch's neutralize-then-name step (`_try_neutralize_and_name`)
# resolves the fully-neutral skeleton to the RETAINED amino-acid word
# ('L-cysteine'), which has no '-oic acid'/'-ic acid' suffix to convert and no
# morpheme for the extra (thiolate) anionic centre. FIX 3b adds
# `_try_neutralize_and_name_systematic` (ions.py), a sibling that requests the
# SYSTEMATIC name of the same fully-neutral skeleton
# ('...-3-sulfanylpropanoic acid'), which DOES convert and DOES carry the
# 'sulfanyl' token FIX 3a can re-ionize -- RT-gated with a strict full-InChIKey
# check before shipping (mirrors the ester-anion RT pattern elsewhere in
# ions.py's multi-anion branch).
#
# The multi-anion dispatch entry for THIS shape is actually
# `routing.dispatch_table._handle_poly_anion` (predicate `_is_poly_anion`:
# >=2 anions, no cations), a near-duplicate of ions.py's own multi-anion
# branch that only neutralizes OXYGEN anions (leaving a thiolate charged) and
# had no fallback when its own '-oic acid' conversion failed -- it shipped the
# unconverted/wrong intermediate name outright (silently dropping the
# thiolate's charge). Extended (same commit) to defer to the now-fixed
# `ions.py::name_anion` when its own conversion cannot express every anionic
# centre, rather than ship the incomplete name.
# =============================================================================

@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("[S-]CC(=O)[O-]", "sulfidoacetate"),
    ("N[C@@H](C[S-])C(=O)[O-]", "(2R)-2-amino-3-sulfidopropanoate"),
])
def test_thiolate_sulfido(smi, expected):
    n = Orthonym(general_fallback=True, general_fallback_unverified=True,
                  allow_aromatic_general=True).name(smi)
    assert n == expected
    _full_ik_rt(smi, n)


# --- regression: shapes _apply_anionic_substituent_prefixes / the poly-anion
# dispatch already handled correctly must stay byte-identical.
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("[O-]C(=O)CC(=O)[O-]", "propanedioate"),
    ("CP(=O)([O-])[O-]", "methylphosphonate"),
    ("OC(=O)CCCCC(=O)[O-]", "5-carboxypentanoate"),
])
def test_c1_no_regression_existing_partial_and_dianion_shapes(smi, expected):
    n = Orthonym(general_fallback=True, general_fallback_unverified=True,
                  allow_aromatic_general=True).name(smi)
    assert n == expected


# =============================================================================
# Task C2 (v33 charged completion): partial-acid-salt CHAIN widening
# (guard AND builder together).
#
# `_name_partial_acid_salt_anion` (ions.py) had 3 guards protecting its
# atom-drop-unsafe builder (`f"{parent_len}-carboxy{prefix}anoate"`, which
# cannot express a substituent, unsaturation, or a ring): ring-exclusion
# (kept UNCHANGED -- the ring case is a separate follow-up), an
# all-carbon-except-carboxyl-oxygens guard, and a fully-saturated-backbone
# guard. This widens the latter two -- admitting a plain terminal
# hydroxy/amino substituent on a PARENT-chain carbon, and backbone
# unsaturation strictly BETWEEN parent-chain carbons -- AND extends the
# builder (`_build_partial_acid_salt_chain_name`, new) in the SAME change to
# (a) enumerate the substituents with locants and (b) emit a `-en-`/`-yn-`
# infix with locant, so no atom the widened guards now admit can be silently
# dropped. Any shape the builder still cannot express (an alkyl branch, an
# ether/amide/secondary-amine, a substituent riding on the still-protonated
# carboxyl carbon itself, unsaturation into that carbon) fails closed to ''
# at the GUARD, never reaching the builder.
# =============================================================================

@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("O=C([O-])[C@H](O)[C@@H](O)C(=O)O", "(2R,3R)-3-carboxy-2,3-dihydroxypropanoate"),
    ("O=C([O-])C#CC(=O)O", "3-carboxyprop-2-ynoate"),
])
def test_partial_acid_salt_chain_widening(smi, expected):
    n = Orthonym(general_fallback=True, general_fallback_unverified=True,
                  allow_aromatic_general=True).name(smi)
    assert n == expected
    _full_ik_rt(smi, n)


# --- regression: the non-partial-acid-salt shapes AND the existing
# all-carbon partial-acid-salt case must stay byte-identical.
@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi,expected", [
    ("[O-]C(=O)CC(=O)[O-]", "propanedioate"),
    ("CP(=O)([O-])[O-]", "methylphosphonate"),
    ("OC(=O)CCCCC(=O)[O-]", "5-carboxypentanoate"),
])
def test_c2_no_regression_existing_shapes(smi, expected):
    n = Orthonym(general_fallback=True, general_fallback_unverified=True,
                  allow_aromatic_general=True).name(smi)
    assert n == expected


def test_partial_acid_salt_chain_alkyl_branch_fails_closed():
    # An alkyl branch (methylmalonate mono-anion) is a shape the builder
    # cannot express (it isn't a plain hydroxy/amino/unsaturation widening) --
    # the guard must fail closed ('') rather than drop the branch.
    from orthonym.perception.ions import get_ion_sites
    from orthonym.rules.ions import _name_partial_acid_salt_anion, classify_anion
    mol = Chem.MolFromSmiles("OC(=O)C(C)C(=O)[O-]")
    sites = get_ion_sites(mol)
    carboxylates = [a for a in sites['anions'] if classify_anion(mol, a) == 'carboxylate']
    assert _name_partial_acid_salt_anion(mol, carboxylates[0]) == ''


def test_partial_acid_salt_chain_ether_substituent_fails_closed():
    # An ether-bridging oxygen (not a plain terminal -OH) is not expressible
    # by the hydroxy/amino widening -- must fail closed.
    from orthonym.perception.ions import get_ion_sites
    from orthonym.rules.ions import _name_partial_acid_salt_anion, classify_anion
    mol = Chem.MolFromSmiles("OC(=O)C(OC)C(=O)[O-]")
    sites = get_ion_sites(mol)
    carboxylates = [a for a in sites['anions'] if classify_anion(mol, a) == 'carboxylate']
    assert _name_partial_acid_salt_anion(mol, carboxylates[0]) == ''
