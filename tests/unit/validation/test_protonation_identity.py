"""Protonation-site identity (fix a performance pass, wp7; whole-branch verification panel RISK 1).

The standard InChI moves an onium atom's hydrons into one mobile /p layer, so the full
InChIKey cannot tell protonation isomers apart. Five names shipped at pin_verified with
the '-ium' on the WRONG nitrogen, and every gate ('s full-key shortcut, the
charged-router RT helpers, tests/support/rt_assert) accepted them:

    C[NH2+]CCC(=O)NC 'N-methyl-3-(methylamino)propanamidium'
    C[NH2+]CCC(=O)NCC 'N-ethyl-3-(methylamino)propanamidium'
    C[NH2+]CCC#N '3-(methylamino)propanenitrilium'
    C[NH+](C)CCN '2-(dimethylamino)ethan-1-aminium'
    [NH3+]CC(=O)NCC(=O)O 'glycylglycinium'

 (the Blue Book) / Table 7.4 (:41417): 'amidium', 'nitrilium', 'aminium'
are the cationic forms OF those suffix groups, so each name above denotes a protonation
isomer of the input (independent OPSIN 2.9.0 parse, e.g. 'N-methyl-3-(methylamino)-
propanamidium' -> CNCCC(=O)[NH2+]C). With the charge elsewhere the cation is the
principal group, '6 Cations' before the neutral classes,:18169).
"""
import pytest

from orthonym.validation.protonation_identity import protonation_site_verdict
from tests.support.jars import jar_or_skip

pytestmark = pytest.mark.unit

# (input, OPSIN 2.9.0 parse of the old wrong name)
_WRONG_SITE = [
    ("C[NH2+]CCC(=O)NC", "C[NH2+]C(CCNC)=O"),
    ("C[NH2+]CCC(=O)NCC", "C(C)[NH2+]C(CCNC)=O"),
    ("C[NH2+]CCC#N", "CNCCC#[NH+]"),
    ("C[NH+](C)CCN", "CN(CC[NH3+])C"),
    ("[NH3+]CC(=O)NCC(=O)O", "NCC(=O)[NH2+]CC(=O)O"),
    ("Nc1cc[nH+]cc1", "[NH3+]c1ccncc1"),   # 'pyridin-4-aminium' for the ring-N cation
]

_OLD_WRONG_NAMES = {
    "C[NH2+]CCC(=O)NC": "N-methyl-3-(methylamino)propanamidium",
    "C[NH2+]CCC(=O)NCC": "N-ethyl-3-(methylamino)propanamidium",
    "C[NH2+]CCC#N": "3-(methylamino)propanenitrilium",
    "C[NH+](C)CCN": "2-(dimethylamino)ethan-1-aminium",
    "[NH3+]CC(=O)NCC(=O)O": "glycylglycinium",
    "Nc1cc[nH+]cc1": "pyridin-4-aminium",
}


@pytest.mark.parametrize("smiles,parsed", _WRONG_SITE)
def test_wrong_protonation_site_is_a_mismatch(smiles, parsed):
    from rdkit import Chem
    # the trap: the full standard InChIKey is equal
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)) == \
        Chem.MolToInchiKey(Chem.MolFromSmiles(parsed))
    assert protonation_site_verdict(smiles, parsed) == "mismatch"


@pytest.mark.parametrize("smiles,parsed", [
    # resonance drawings of one charge-delocalised cation are the same species
    ("CC(=[NH2+])NC", "CC(N)=[NH+]C"),
    ("Nc1cc[nH+]cc1", "[NH2+]=C1C=CNC=C1"),
    ("NC(=[NH2+])NC", "NC(N)=[NH+]C"),
    # the same drawing
    ("C[NH3+]", "C[NH3+]"),
    ("NCC[NH3+].[Cl-]", "[Cl-].[NH3+]CCN"),
])
def test_same_species_is_ok(smiles, parsed):
    assert protonation_site_verdict(smiles, parsed) == "ok"


@pytest.mark.parametrize("smiles,parsed", [
    # a salt named as its neutral acid-base form, and a zwitterion named as the
    # neutral amino acid:54518): not a protonation-SITE question
    ("C[NH3+].[Cl-]", "CN.Cl"),
    ("[NH3+]CC(=O)[O-]", "NCC(=O)O"),
    # no protonated heavy atom at all
    ("CCO", "CCO"),
    ("C[N+](C)(C)C", "C[N+](C)(C)C"),
])
def test_out_of_scope_is_na(smiles, parsed):
    assert protonation_site_verdict(smiles, parsed) == "n/a"


def test_self_consistency_verdict_no_longer_short_circuits():
    """'s full-InChIKey shortcut accepted every protonation isomer."""
    from orthonym.namer import _self_consistency_verdict
    for smiles, parsed in _WRONG_SITE:
        assert _self_consistency_verdict(smiles, parsed) == "mismatch", smiles
    assert _self_consistency_verdict("CC(=[NH2+])NC", "CC(N)=[NH+]C") == "ok"


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", list(_OLD_WRONG_NAMES))
def test_wrong_site_name_never_ships(smiles):
    """Gate on: the protonation-isomer name is never shipped; the PIN tier ships an
    exact name (at most a demoted one) or fails closed, and best-effort names the
    molecule exactly (breadth)."""
    from orthonym import Orthonym
    from tests.support.rt_assert import assert_tier_contract, name_is_rt_exact
    assert not name_is_rt_exact(_OLD_WRONG_NAMES[smiles], smiles)
    assert_tier_contract(smiles)
    r = Orthonym(style="pin").name_tiered(smiles)
    assert r["name"] != _OLD_WRONG_NAMES[smiles], r
    if r["tier"] == "pin_verified":
        assert name_is_rt_exact(r["name"], smiles), r


@pytest.mark.parametrize("smiles", [
    "C[NH2+]CCC(=O)NC", "C[NH2+]CCC#N", "C=CC(=O)NCCC[N+](C)(C)C"])
def test_suffix_swap_declines_off_the_suffix_nitrogen(smiles):
    """Producer: the amidium/nitrilium/aminium suffix swap (composer
    _try_ion_aspect_composition) runs only when every cationic atom is a suffix
    nitrogen; it used to append 'ium' by suffix text alone."""
    from orthonym import Orthonym
    from orthonym.assembly.composer import _try_ion_aspect_composition
    from rdkit import Chem
    namer = Orthonym(style="pin", _disable_opsin_validity_gate=True)
    mol = Chem.MolFromSmiles(smiles)
    can = Chem.MolToSmiles(mol)
    features = namer._perceive(mol, can, can)
    namer._classify(features)
    assert _try_ion_aspect_composition(features, "pin") is None


@pytest.mark.opsin_gate
def test_diamine_monocation_takes_the_aminium_pin():
    """The suffix swap used to turn NCC[NH3+] into 'ethan-1-aminium' (the neutral
    NH2 was the suffix; the other nitrogen was dropped), which the gate voided, so
    the PIN tier abstained. Declining the swap lets the cation path name it:
    '2-aminoethan-1-aminium chloride (PIN)' (the Blue Book) is this cation's
    salt."""
    from orthonym import Orthonym
    from tests.support.rt_assert import name_is_rt_exact
    r = Orthonym(style="pin").name_tiered("NCC[NH3+]")
    assert r["name"] == "2-aminoethan-1-aminium" and r["tier"] == "pin_verified", r
    assert name_is_rt_exact(r["name"], "NCC[NH3+]")
    assert Orthonym(style="pin").name_tiered("CC[NH3+]")["name"] == "ethanaminium"


# ---------------------------------------------------------------------------
# Rule 2: a deprotonated site, or the hydron of a cation whose charge is drawn on
# a substituted atom. The standard InChI adds the missing hydron back (or takes
# the extra one off) into its /p layer, so the full key cannot place it either.
# Each pair is (input, OPSIN 2.9.0 parse of the name that used to ship); the
# fixed-H InChI is computed here with RDKit alone, independent of the code under
# test.
# ---------------------------------------------------------------------------

def _std_key(smi):
    from rdkit import Chem
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smi))


def _fixed_h(smi):
    from rdkit import Chem
    from rdkit.Chem import inchi
    return inchi.MolToInchi(Chem.MolFromSmiles(smi), options="/FixedH /SNon")


_WRONG_CHARGE_SITE = [
    # 'sodium 2-(carboxymethyl)benzoate' for the salt of the CH2 carboxylate
    # method (1), the Blue Book: "the free acid is cited as a
    # prefix to the name of the anion"; the ring carboxylate is the anion here)
    ("O=C([O-])Cc1ccccc1C(=O)O.[Na+]", "C(=O)(O)CC1=C(C(=O)[O-])C=CC=C1.[Na+]"),
    # '2-(2-carboxyethyl)-4-methyl-5-propylfuran-3-carboxylate'
    ("CCCc1oc(CCC(=O)[O-])c(C(=O)O)c1C", "C(=O)(O)CCC=1OC(=C(C1C(=O)[O-])C)CCC"),
    # '4-hydroxynaphthalene-2,7-disulfonate' for the sulfonate / sulfonic acid /
    # naphtholate
    ("O=S(=O)([O-])c1ccc2c([O-])cc(S(=O)(=O)O)cc2c1",
     "OC1=CC(=CC2=CC(=CC=C12)S(=O)(=O)[O-])S(=O)(=O)[O-]"),
    # citric acid monoanion: the central vs a terminal carboxylate
    ("OC(=O)CC(O)(CC(=O)O)C(=O)[O-]", "[O-]C(=O)CC(O)(CC(=O)O)C(=O)O"),
    # two phosphonic acid groups joined through carbon are two acid groups, not
    # one polynuclear oxoacid unit: each hydron's site counts
    ("O=P([O-])(O)C(Cl)(Cl)P(=O)([O-])O", "ClC(P(=O)(O)O)(Cl)P([O-])([O-])=O"),
    # '1-methyl-1H-benzimidazol-1-ium': the hydron on the methylated N-1, not on N-3
    ("C[N+]1=CNc2ccccc21", "C[NH+]1C=NC2=C1C=CC=C2"),
    # '2,3-diethyl-5,7-dimethyl-1-phenyl-1H-1,4-diazepin-1-ium'
    ("CCC1=C([N+](=C(C=C(N1)C)C)c2ccccc2)CC",
     "C(C)C=1[NH+](C(=CC(=NC1CC)C)C)C1=CC=CC=C1"),
]


@pytest.mark.parametrize("smiles,parsed", _WRONG_CHARGE_SITE)
def test_wrong_charge_site_is_a_mismatch(smiles, parsed):
    assert _std_key(smiles) == _std_key(parsed)          # the trap
    assert _fixed_h(smiles) != _fixed_h(parsed)          # independent: another species
    assert protonation_site_verdict(smiles, parsed) == "mismatch"


_SAME_SPECIES_CHARGED = [
    # a neutral tautomer beside a fixed charge: the benzimidazole NH on the other N
    ("O=[N+]([O-])c1ccc2nc(C)[nH]c2c1", "O=[N+]([O-])c1ccc2[nH]c(C)nc2c1"),
    # a guanidine tautomer beside a sodium carboxylate
    ("[Na+].[O-]C(=O)CCN=C(N)N", "[Na+].[O-]C(=O)CCNC(N)=N"),
    # the lactam drawn as the lactim beside the imidazolium charge (7-methylinosine)
    ("Cn1c[n+]([C@@H]2O[C@H](CO)[C@@H](O)[C@H]2O)c2[nH]cnc(=O)c21",
     "C[N+]1=CN([C@H]2[C@H](O)[C@H](O)[C@@H](CO)O2)C=2N=CN=C(C12)O"),
    # an inner-salt pair (NH3+ / COO-) drawn in its neutral form, the other
    # carboxylate kept, the Blue Book)
    ("[NH3+][C@@H](CCC(=O)N[C@@H](CSCCl)C(=O)NCC(=O)[O-])C(=O)[O-]",
     "N[C@@H](CCC(=O)N[C@H](C(=O)NCC(=O)[O-])CSCCl)C(=O)O"),
]


@pytest.mark.parametrize("smiles,parsed", _SAME_SPECIES_CHARGED)
def test_tautomer_or_inner_salt_pair_is_not_a_mismatch(smiles, parsed):
    assert _std_key(smiles) == _std_key(parsed)
    assert _fixed_h(smiles) != _fixed_h(parsed)
    assert protonation_site_verdict(smiles, parsed) != "mismatch"


@pytest.mark.parametrize("smiles,parsed", [
    # the right partial salt, and the carboxylate drawn on its other oxygen
    ("[Na+].OC(=O)CCC(=O)[O-]", "C(CCC(=O)[O-])(=O)O.[Na+]"),
    ("O=C([O-])Cc1ccccc1C(=O)O.[Na+]", "C(=O)(O)C1=C(C=CC=C1)CC(=O)[O-].[Na+]"),
    # the other resonance drawing of the ring cation, the hydron on the same N
    ("C[N+]1=CNc2ccccc21", "CN1C=[NH+]C2=C1C=CC=C2"),
    # an acid salt drawn with a bare proton leaves the site open,
    # the Blue Book "except when the structure of the acid salt is unknown")
    ("[Na+].[H+].[O-]C(=O)Cc1ccccc1C(=O)[O-]", "C(=O)([O-])CC1=C(C(=O)O)C=CC=C1.[Na+]"),
])
def test_same_charge_site_is_not_a_mismatch(smiles, parsed):
    assert _std_key(smiles) == _std_key(parsed)
    assert protonation_site_verdict(smiles, parsed) != "mismatch"


def test_self_consistency_verdict_sees_the_charge_site():
    from orthonym.namer import _self_consistency_verdict
    for smiles, parsed in _WRONG_CHARGE_SITE:
        assert _self_consistency_verdict(smiles, parsed) == "mismatch", smiles
    for smiles, parsed in _SAME_SPECIES_CHARGED:
        assert _self_consistency_verdict(smiles, parsed) == "ok", smiles


def test_full_key_exits_see_the_charge_site():
    """The other full-key compares: the offer match and the shipped-name round
    trip (general tiers), with OPSIN's read of the old name."""
    from orthonym.namer import _full_inchikey_offer_match, _shipped_name_round_trip
    witness = "O=C([O-])Cc1ccccc1C(=O)O.[Na+]"
    assert not _full_inchikey_offer_match(witness, _WRONG_CHARGE_SITE[0][1])
    assert _full_inchikey_offer_match(witness, "C(=O)(O)C1=C(C=CC=C1)CC(=O)[O-].[Na+]")
    jar_or_skip()
    assert _shipped_name_round_trip("sodium 2-(carboxymethyl)benzoate", witness) == "failed"
    assert _shipped_name_round_trip("sodium (2-carboxyphenyl)acetate", witness) == "verified"


def test_verify_or_none_sees_the_charge_site():
    jar_or_skip()
    from orthonym.validation.reconstruct import verify_or_none
    witness = "O=C([O-])Cc1ccccc1C(=O)O.[Na+]"
    assert verify_or_none("sodium 2-(carboxymethyl)benzoate", witness) is None
    assert verify_or_none("sodium (2-carboxyphenyl)acetate", witness) is not None


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles,old_name", [
    ("O=C([O-])Cc1ccccc1C(=O)O.[Na+]", "sodium 2-(carboxymethyl)benzoate"),
    ("C[N+]1=CNc2ccccc21", "1-methyl-1H-benzimidazol-1-ium"),
])
def test_wrong_charge_site_name_never_ships(smiles, old_name):
    """Neither tier ships the old name, and best-effort names the molecule with an
    independent OPSIN read-back whose fixed-H InChI is the input's."""
    from orthonym import Orthonym
    from orthonym.cli import _emit_tier_flags
    from tests.support.rt_assert import _independent_parse
    jar_or_skip()
    assert _fixed_h(_independent_parse(old_name)) != _fixed_h(smiles)
    for kw in ({}, _emit_tier_flags("best-effort")):
        r = Orthonym(style="pin", **kw).name_tiered(smiles)
        assert r["name"] != old_name, r
    be = Orthonym(style="pin", **_emit_tier_flags("best-effort")).name_tiered(smiles)
    parsed = _independent_parse(be["name"])
    assert parsed and _std_key(parsed) == _std_key(smiles) and _fixed_h(parsed) == _fixed_h(smiles), be


# ---------------------------------------------------------------------------
# Rule 2 (b) is the case only: ONE alpha-amino-acid pair neutralised.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,parsed,verdict", [
    # monosodium glutamate: the alpha pair drawn neutral is the amino acid...
    ("[NH3+]C(CCC(=O)[O-])C(=O)[O-].[Na+]", "NC(CCC(=O)[O-])C(=O)O.[Na+]", "n/a"),
    #... the ammonium paired with the side-chain carboxylate is another protomer
    ("[NH3+]C(CCC(=O)[O-])C(=O)[O-].[Na+]", "NC(CCC(=O)O)C(=O)[O-].[Na+]", "mismatch"),
    # an arylammonium beside a sulfonate is not an amino-acid pair
    ("[NH3+]c1ccc(S(=O)(=O)[O-])cc1C(=O)[O-].[Na+]",
     "Nc1ccc(S(=O)(=O)O)cc1C(=O)[O-].[Na+]", "mismatch"),
])
def test_only_the_amino_acid_zwitterion_pair_is_neutralised(smiles, parsed, verdict):
    assert _std_key(smiles) == _std_key(parsed)
    assert protonation_site_verdict(smiles, parsed) == verdict


# ---------------------------------------------------------------------------
# Rule 2 (c) reads the name: only a method (2) 'hydrogen' name leaves the hydrons'
# site open, and only among EQUIVALENT noncarbon oxoacid centres.
# (the Blue Book): "Acid salts of di- and polynuclear noncarbon oxoacids are
# named in the same way as neutral salts, the remaining acid hydrogen atom(s) being
# indicated by the word 'hydrogen' (or 'dihydrogen', etc., as appropriate)";
# (:31619) the acid salts of organic derivatives of polybasic
# inorganic oxoacids by method (2). OPSIN reads each 'dihydrogen' name below with
# both hydrons on one phosphorus (the parse strings are OPSIN 2.9.0's).
# ---------------------------------------------------------------------------

_OPEN_SITE_NAMES = [
    ("[Na+].[Na+].OP(=O)([O-])OP(=O)([O-])O", "OP(O)(=O)OP(=O)([O-])[O-].[Na+].[Na+]",
     "disodium dihydrogen diphosphate"),                                  # gold W3-P10-13
    ("[Na+].[Na+].OP(=O)([O-])CP(=O)([O-])O", "C(P([O-])([O-])=O)P(O)(O)=O.[Na+].[Na+]",
     "disodium dihydrogen methylenebis(phosphonate)"),
    ("[Na+].[Na+].OP(=O)([O-])C(Cl)(Cl)P(=O)([O-])O",
     "ClC(P([O-])([O-])=O)(P(O)(O)=O)Cl.[Na+].[Na+]",
     "disodium dihydrogen (dichloromethylene)bis(phosphonate)"),
]


@pytest.mark.parametrize("smiles,parsed,name", _OPEN_SITE_NAMES)
def test_hydrogen_word_leaves_the_site_open_among_equivalent_centres(smiles, parsed, name):
    assert _std_key(smiles) == _std_key(parsed)
    assert _fixed_h(smiles) != _fixed_h(parsed)
    assert protonation_site_verdict(smiles, parsed, name) == "n/a"
    assert protonation_site_verdict(smiles, parsed) == "mismatch"   # no name, no licence
    from orthonym.namer import _self_consistency_verdict
    assert _self_consistency_verdict(smiles, parsed, name=name) == "ok"


@pytest.mark.parametrize("smiles,parsed,name", [
    # a substitutive name places the hydrons: '(dichloro-phosphonomethyl)phosphonate'
    ("O=P([O-])(O)C(Cl)(Cl)P(=O)([O-])O", "ClC(P(=O)(O)O)(Cl)P([O-])([O-])=O",
     "(dichloro-phosphonomethyl)phosphonate"),
    # centres of different kinds: the ester P of a methyl diphosphate, S vs P
    ("COP(=O)([O-])OP(=O)(O)O.[Na+]", "COP(=O)(O)OP(=O)([O-])O.[Na+]",
     "sodium methyl dihydrogen diphosphate"),
    ("OS(=O)(=O)OP(=O)([O-])O.[Na+]", "[O-]S(=O)(=O)OP(=O)(O)O.[Na+]",
     "sodium dihydrogen sulfatophosphate"),
])
def test_hydron_on_another_kind_of_centre_is_a_mismatch(smiles, parsed, name):
    assert _std_key(smiles) == _std_key(parsed)
    assert protonation_site_verdict(smiles, parsed, name) == "mismatch"


@pytest.mark.parametrize("smiles,parsed,name", [
    # hydrons moved between two SEPARATE fragments: a double salt vs the acid plus
    # the fully ionized salt,:36903, speaks of one di- or
    # polynuclear acid)
    ("[Na+].[Na+].[Na+].OP(=O)(O)[O-].OP(=O)([O-])[O-]",
     "OP(=O)(O)O.[O-]P(=O)([O-])[O-].[Na+].[Na+].[Na+]",
     "trisodium dihydrogen phosphate hydrogen phosphate"),
    ("[Na+].[Na+].CP(=O)(O)[O-].CP(=O)(O)[O-]",
     "CP(=O)(O)O.CP(=O)([O-])[O-].[Na+].[Na+]",
     "disodium dihydrogen bis(methylphosphonate)"),
    # the 'hydrogen' word outside the salt position licenses nothing
    ("[Na+].[Na+].OP(=O)([O-])CP(=O)([O-])O", "C(P([O-])([O-])=O)P(O)(O)=O.[Na+].[Na+]",
     "hydrogen methylenebis(phosphonate)"),
    ("[Na+].[Na+].OP(=O)([O-])CP(=O)([O-])O", "C(P([O-])([O-])=O)P(O)(O)=O.[Na+].[Na+]",
     "disodium [(hydrogen phosphonato)methyl]phosphonate"),
])
def test_hydrogen_word_pools_within_one_component_in_the_salt_position(smiles, parsed, name):
    assert _std_key(smiles) == _std_key(parsed)
    assert protonation_site_verdict(smiles, parsed, name) == "mismatch"


def test_multiplied_hydrogen_unit_pools_identical_units_as_drawn():
    """'disodium bis(methyl hydrogen phosphate)' names two identical units, each
    with its own hydron: for the input drawn as those units the reading may put
    the hydrons on either one; the other drawing (acid + fully ionized ester) is
    not what the name says."""
    name = "disodium bis(methyl hydrogen phosphate)"
    one_one = "[Na+].[Na+].COP(=O)(O)[O-].COP(=O)(O)[O-]"
    two_zero = "[Na+].[Na+].COP(=O)(O)O.COP(=O)([O-])[O-]"
    assert _std_key(one_one) == _std_key(two_zero)
    assert protonation_site_verdict(one_one, two_zero, name) == "n/a"
    assert protonation_site_verdict(two_zero, one_one, name) == "mismatch"
    assert protonation_site_verdict(one_one, two_zero) == "mismatch"


def test_verify_or_none_sees_the_radical_graph():
    jar_or_skip()
    from orthonym.validation.reconstruct import verify_or_none
    assert verify_or_none("ethene", "[CH2][CH2]") is None
    assert verify_or_none("ethanol", "CCO") == "ethanol"
