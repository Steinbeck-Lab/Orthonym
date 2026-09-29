"""Task C: the mixed-spiro-fused ring leaf wired into the best-effort
universal core (`assembly/universal_substituent._name_ring_spine`).

A spiro atom joining a FUSED ring component (indane / chromene / indoline /
cyclopenta[b]pyridine...) to a second ring is a separable
``spiro[<fused-comp>-x,y'-<comp2>]`` system. Before this wiring the
unconditional universal FLOOR VOIDED on such a ring system (neither a
von-Baeyer cage nor a plain von-Baeyer spiro), so the whole decorated
molecule silently abstained.

0-wrong is preserved by the caller's offer full-InChIKey RT gate; these tests
pin (a) the no-abstain conversion, (b) that the primed ``_Locant`` tuple is
rendered ``"5'"`` and never leaks ``(5, "'")`` into the name, and (c) that the
governing helpers order/print primed locants correctly.

Governing rule: IUPAC 2013 (spiro ring systems, separable form).
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.assembly.universal_substituent import (
    name_universal_substitutive, _locant_sort_key, _locant_display,
)


def _const(smi):
    m = Chem.MolFromSmiles(smi)
    return inchi.MolToInchiKey(m).split("-")[0] if m else None


# --- pure-unit: the locant helpers (no OPSIN) -----------------------------

def test_locant_display_renders_prime_tuple():
    # The L2 defect leaked the raw tuple; the fix routes through _prime_token.
    assert _locant_display((5, "'")) == "5'"
    assert _locant_display((1, "'")) == "1'"
    assert _locant_display((3, "''")) == "3''"
    assert _locant_display(4) == "4"
    assert _locant_display("2'") == "2'"


def test_locant_sort_key_orders_unprimed_before_primed():
    # (prime_rank, number): all unprimed precede single-primed, then by number.
    vals = [(2, "'"), 10, 3, (1, "'"), "5'"]
    ordered = sorted(vals, key=_locant_sort_key)
    # unprimed 3,10 first (numeric), then primed 1',2',5'
    assert ordered == [3, 10, (1, "'"), (2, "'"), "5'"]


# --- integration: whole decorated molecules that used to abstain ----------

# (smiles, a substring the emitted core name must contain)
CONVERTS = [
    # indole spiro cyclopentane, decorated
    ("Cc1nc(N)nn1[C@H]1CC[C@@]2(CNc3ncc(-c4ccc(N)c(C(=O)N(C)C)c4F)c(Cl)c32)C1",
     None),
    # benzopyran spiro piperidine + inden-yl amide
    ("O=C(Nc1cccc2c1CCC2)C1=CC2(CCNCC2)Oc2ccccc21",
     "spiro[1-benzopyran-2,4'-piperidine]"),
    # dihydrobenzopyran spiro piperidine, Boc + isopropoxymethyl
    ("CC(C)OCC1CC2(CCN(C(=O)OC(C)(C)C)CC2)Oc2ccccc21",
     "spiro[3,4-dihydro-2H-1-benzopyran-2,4'-piperidine]"),
    # thiazolidine spiro dihydroindole, stereo + two aryl branches
    ("Cc1ccc(C(=O)N2CC(C)(C)S[C@]23C(=O)N(Cc2ccccc2Cl)c2ccccc23)cc1",
     "spiro[1,3-thiazolidine-2,3'-2,3-dihydro-1H-indole]"),
]


@pytest.mark.parametrize("smi,core_sub", CONVERTS)
def test_floor_names_mixed_spiro_fused_without_abstaining(smi, core_sub):
    """The universal FLOOR now BUILDS a coverage-complete name (was None)."""
    m = Chem.MolFromSmiles(smi)
    res = name_universal_substitutive(m)
    assert res is not None, "floor voided on a mixed-spiro-fused core"
    assert res.name
    if core_sub is not None:
        assert core_sub in res.name, res.name


@pytest.mark.parametrize("smi,core_sub", CONVERTS)
def test_no_locant_tuple_leaks_into_name(smi, core_sub):
    """A `(n, "'")` tuple must never appear literally in an emitted name."""
    m = Chem.MolFromSmiles(smi)
    res = name_universal_substitutive(m)
    assert res is not None
    assert "'" not in res.name or ", '" not in res.name
    assert "(" not in res.name or "', " not in res.name
    # the specific L2 signature
    for bad in ('(1, "\'")', "(1, \"'\")", "(5, \"'\")"):
        assert bad not in res.name


@pytest.mark.opsin_gate
@pytest.mark.roundtrip
@pytest.mark.parametrize("smi,core_sub", CONVERTS)
def test_floor_name_round_trips_to_input_constitution(smi, core_sub):
    """0-wrong: every emitted floor name OPSIN-parses to the input
    CONSTITUTION (full-InChIKey first block)."""
    from orthonym.validation.opsin_roundtrip import opsin_parse
    m = Chem.MolFromSmiles(smi)
    res = name_universal_substitutive(m)
    assert res is not None
    parsed = opsin_parse(res.name)
    assert parsed is not None, f"OPSIN could not parse: {res.name}"
    assert _const(parsed) == _const(smi), (
        f"wrong constitution: {res.name} -> {parsed}"
    )


# --- (c)-aliphatic: spiro-of-von-Baeyer-bicyclic degrades to the
# separable form spiro[bicyclo[...]-x,y'-<comp2>] (floor-only) ------------

VONBAEYER_SPIRO = [
    # bicyclo[3.2.0]hept-2-ene spiro 1,3-dioxolane
    ("C1=CC2C(C1)CC21OCCO1", "bicyclo[3.2.0]"),
    # 3-azabicyclo[3.3.0]octane spiro piperidine
    ("CNC(=O)[C@]12CCC3(CCN(C(C)=O)CC3)[C@H]1CN(S(C)(=O)=O)C2", "bicyclo[3.3.0]"),
]


@pytest.mark.opsin_gate
@pytest.mark.roundtrip
@pytest.mark.parametrize("smi,desc", VONBAEYER_SPIRO)
def test_vonbaeyer_spiro_of_fused_floor_round_trips(smi, desc):
    """The (c)-aliphatic bucket: a spiro atom joining a von-Baeyer bicyclic to a
    second ring now degrades to a valid covering von-Baeyer separable spiro name
    (was None -> abstain), full-InChIKey RT-correct."""
    from orthonym.validation.opsin_roundtrip import opsin_parse
    m = Chem.MolFromSmiles(smi)
    res = name_universal_substitutive(m)
    assert res is not None, "floor voided on a spiro-of-vonBaeyer core"
    assert desc in res.name, res.name
    parsed = opsin_parse(res.name)
    assert parsed is not None, f"OPSIN could not parse: {res.name}"
    assert _const(parsed) == _const(smi), f"{res.name} -> {parsed}"


def test_vonbaeyer_component_is_floor_only_pin_untouched():
    """`name_mixed_spiro_fused` default (PIN path) still declines a
    spiro-of-vonBaeyer core; only the floor flag names it."""
    from orthonym.rules.spiro import name_mixed_spiro_fused
    m = Chem.MolFromSmiles("C1=CC2C(C1)CC21OCCO1")
    assert name_mixed_spiro_fused(m) is None                       # PIN default
    assert name_mixed_spiro_fused(m, allow_vonbaeyer_component=True) is not None


# --- both-sides-fused: a monospiro joining TWO fused/bridged ring systems -----

BOTH_SIDES_FUSED = [
    # 1,3-dihydro-2-benzofuran spiro xanthene
    ("C1OC2(c3ccccc31)c1ccccc1Oc1ccccc12", "9'-xanthene]"),
    # tetrahydroisoquinoline spiro a trioxatricyclododecatriene
    ("C1NC2(Cc3ccccc31)OCc1cc3c(cc12)OCO3", "trioxatricyclo"),
    # dihydrobenzofuran spiro an azabicyclodecene
    ("C1CCC2=C(CC[C@]3(C2)Cc2ccccc2O3)N1", "2-azabicyclo[4.4.0]"),
]


@pytest.mark.opsin_gate
@pytest.mark.roundtrip
@pytest.mark.parametrize("core_smi,sub", BOTH_SIDES_FUSED)
def test_both_sides_fused_monospiro_round_trips(core_smi, sub):
    """The both-sides-fused (c)-bucket: a spiro atom joining two fused/bridged
    ring systems now names each side independently and joins the
    separable form (was None -> abstain), RT-correct to the CORE constitution."""
    from orthonym.rules.spiro import _name_general_monospiro_fused
    from orthonym.validation.opsin_roundtrip import opsin_parse
    m = Chem.MolFromSmiles(core_smi)
    res = _name_general_monospiro_fused(m)
    assert res is not None, "both-sides-fused namer declined a separable core"
    assert sub in res[0], res[0]
    parsed = opsin_parse(res[0])
    assert parsed is not None, f"OPSIN rejected: {res[0]}"
    core = Chem.MolFragmentToSmiles(m, atomsToUse=sorted(res[1]))
    assert _const(parsed) == _const(core), f"{res[0]} -> {parsed}"


# --- linear polyspiro: a dispiro/trispiro chain of fused/ring components ------

# separable named form fires only for a chain with a POLYCYCLIC component
POLYSPIRO = [
    # dispiro: two piperidines spiro'd on a central dioxa-tricyclic (polycyclic)
    ("CN1CCC2(CC1)CC(=O)c1c(ccc3c1OC1(CCN(C)CC1)CC3=O)O2", "dispiro["),
]


@pytest.mark.opsin_gate
@pytest.mark.roundtrip
@pytest.mark.parametrize("core_smi,mult", POLYSPIRO)
def test_linear_polyspiro_round_trips(core_smi, mult):
    """The polyspiro (c)-bucket with a POLYCYCLIC component: a linear
    dispiro/trispiro chain assembles the separable form (was None ->
    abstain), RT-correct to the CORE."""
    from orthonym.rules.spiro import _name_linear_polyspiro_fused
    from orthonym.validation.opsin_roundtrip import opsin_parse
    m = Chem.MolFromSmiles(core_smi)
    res = _name_linear_polyspiro_fused(m)
    assert res is not None, "linear polyspiro namer declined a chain core"
    assert res[0].startswith(mult), res[0]
    parsed = opsin_parse(res[0])
    assert parsed is not None, f"OPSIN rejected: {res[0]}"
    core = Chem.MolFragmentToSmiles(m, atomsToUse=sorted(res[1]))
    assert _const(parsed) == _const(core), f"{res[0]} -> {parsed}"


def test_all_monocyclic_polyspiro_defers_to_numeric():
    """The two-path rule: an ALL-MONOCYCLIC dispiro takes the NUMERIC
    von-Baeyer ``dispiro[a.b.c.d]`` path (analyze_spiro_universal), so the
    separable namer defers (returns None) and the numeric PIN ships RT-correct."""
    from orthonym.rules.spiro import _name_linear_polyspiro_fused
    from orthonym.assembly.universal_substituent import name_universal_substitutive
    from orthonym.validation.opsin_roundtrip import opsin_parse
    smi = "O1CCC2(CC1)CCC1(CCOCC1)CC2"  # dispiro of oxanes + central cyclohexane
    m = Chem.MolFromSmiles(smi)
    assert _name_linear_polyspiro_fused(m) is None
    res = name_universal_substitutive(m)
    assert res is not None
    assert "dispiro[" in res.name and res.name[0].isdigit() or "dispiro[5" in res.name
    assert _const(opsin_parse(res.name)) == _const(smi)


def test_linear_polyspiro_declines_monospiro():
    """Fail-closed: a monospiro (n_spiro<2) is not this namer's job."""
    from orthonym.rules.spiro import _name_linear_polyspiro_fused
    m = Chem.MolFromSmiles("C1=CC2C(C1)CC21OCCO1")  # monospiro
    assert _name_linear_polyspiro_fused(m) is None


# --- von-Baeyer retry: a fused-spiro whose SYSTEMATIC fusion name is wrong/
# unparseable ships the faithful von-Baeyer polyene form instead -----------

@pytest.mark.opsin_gate
@pytest.mark.roundtrip
def test_force_vonbaeyer_spiro_recovers_wrong_fusion_descriptor():
    """furo[3,4-c]pyridine spiro pyrrolidine: the systematic fusion name is
    round-trip-WRONG, so the ``force_vonbaeyer_spiro`` retry names the fused
    component as its von-Baeyer polyene and the molecule ships 0-wrong."""
    from orthonym.assembly.universal_substituent import name_universal_substitutive
    from orthonym.validation.opsin_roundtrip import opsin_parse
    from orthonym.validation.reconstruct import verify_or_none
    smi = "O=C1OC2(CCNC2)c2ccncc21"
    m = Chem.MolFromSmiles(smi)
    # the forced form names the fused component as an oxa-aza-bicyclo polyene
    res = name_universal_substitutive(m, force_vonbaeyer_spiro=True)
    assert res is not None and "bicyclo[" in res.name, res
    assert verify_or_none(res.name, Chem.MolToSmiles(m)) is not None
    assert _const(opsin_parse(res.name)) == _const(smi)


@pytest.mark.opsin_gate
@pytest.mark.roundtrip
@pytest.mark.parametrize("smi", [
    # two chain-bridged spiro-hydantoin cores
    "O=C(CN1C(=O)NC2(CCc3ccccc32)C1=O)NN1C(=O)NC2(CCCCC2)C1=O",
    # a spiro-fused core plus a spiro-dioxolane substituent
    "O=C1CSC2(C(=O)N(CN3CCC4(CC3)OCCO4)c3ccccc32)N1c1ccc(F)cc1",
])
def test_disjoint_spiro_cores_each_named_independently(smi):
    """A molecule with SEVERAL disjoint spiro cores: each ring system is named
    on its own (the spiro leaf restricts to THIS ring system, so a sibling core
    is not miscounted as a second spiro atom), and the whole composes 0-wrong."""
    from orthonym.namer import Orthonym
    from orthonym.validation.opsin_roundtrip import opsin_parse
    nm = Orthonym(general_fallback=True, general_fallback_unverified=True,
                   allow_aromatic_general=True)
    name = nm.name(smi)
    assert name and name != "unknown organic compound", name
    assert _const(opsin_parse(name)) == _const(smi), name


def test_force_vonbaeyer_spiro_default_off_is_systematic():
    """Default (no force) keeps the systematic naming for a catalog fused core
    (retry is opt-in, never the default nomenclature)."""
    from orthonym.assembly.universal_substituent import name_universal_substitutive
    # an indole spiro that names via the systematic separable form
    smi = "O=C(Nc1cccc2c1CCC2)C1=CC2(CCNCC2)Oc2ccccc21"
    r_default = name_universal_substitutive(Chem.MolFromSmiles(smi))
    assert r_default is not None and "1-benzopyran" in r_default.name
