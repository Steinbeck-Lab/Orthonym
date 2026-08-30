"""Task C: the mixed-spiro-fused ring leaf wired into the best-effort
universal core (`assembly/universal_substituent._name_ring_spine`).

A spiro atom joining a FUSED ring component (indane / chromene / indoline /
cyclopenta[b]pyridine ...) to a second ring is a P-24.5.1 separable
``spiro[<fused-comp>-x,y'-<comp2>]`` system. Before this wiring the
unconditional universal FLOOR VOIDED on such a ring system (neither a
von-Baeyer cage nor a plain von-Baeyer spiro), so the whole decorated
molecule silently abstained.

0-wrong is preserved by the caller's offer full-InChIKey RT gate; these tests
pin (a) the no-abstain conversion, (b) that the primed ``_Locant`` tuple is
rendered ``"5'"`` and never leaks ``(5, "'")`` into the name, and (c) that the
governing helpers order/print primed locants correctly.

Governing rule: IUPAC 2013 P-24.5.1 (spiro ring systems, separable form).
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


# --- (c)-aliphatic: spiro-of-von-Baeyer-bicyclic degrades to the P-24.5.1
#     separable form spiro[bicyclo[...]-x,y'-<comp2>] (floor-only) ------------

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
