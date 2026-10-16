"""SF lane E, task 2: the read-back of a name uses OPSIN ``-r`` only for a radical input.

OPSIN with ``-r`` (``allowRadicals``) reads a name with unfilled valences, e.g. '(phosphoryl)oxy'
(``-PO<``, three free valences) for ``-O-P=O``, and writes a SMILES that does not mark the radical
(P is read as P(III)), so the structure compares equal to the closed-shell input by InChIKey.
"GENERAL METHODOLOGY" (the Blue Book): "Radicals are named by modifying a parent hydride name
to signal the subtraction or addition of one or more hydrogen atoms"; "The suffixes 'yl' (-H.),
'ylidene' (-2H.), 'ylidyne' (-3H.) denote the removal of hydrogen atoms". A name with a free valence
is the name of a radical or a substituent group, never of the closed-shell molecule.

The policy (``validation/read_back_policy.py``): ``-r`` only when the input handed to the outermost
``name`` has radical electrons; outside a ``name`` call the legacy reading (``-r``) stays.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

import orthonym.rules.phosphorus as phosphorus
from orthonym import Orthonym
from orthonym.assembly.fragment_naming import (
    enter_name_scope, exit_name_scope, innermost_name_input)
from orthonym.assembly.retained_substitution import OpsinOracle
from orthonym.validation.opsin_roundtrip import _find_opsin_jar, opsin_parse
from orthonym.rules.charged_router import _full_inchikey_rt_ok
from orthonym.validation.read_back_policy import (
    cache_key,
    input_allows_radicals,
    legacy_reading,
    reading_base_for,
    reading_for,
    smiles_has_radical_electrons,
)

pytestmark = [pytest.mark.unit, pytest.mark.opsin_gate]

RADICAL_ONLY_NAME = "1-[(phosphoryl)oxy]heptane"      # -O-P=O named with the three-valence prefix
CLOSED_SHELL = "CCCCCCCOP=O"


def _key(smiles):
    return inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))


@pytest.fixture
def oracle():
    jar = _find_opsin_jar()
    if jar is None:
        pytest.skip("no OPSIN jar")
    return OpsinOracle(opsin_jar=jar)


@pytest.fixture
def named_input():
    """Open a ``name`` frame for ``smiles`` (what the policy reads: the innermost live frame)."""
    opened = []

    def _set(smiles):
        enter_name_scope(smiles)
        opened.append(smiles)
    yield _set
    while opened:
        exit_name_scope()
        opened.pop()


# ---- the policy -------------------------------------------------------------------------

@pytest.mark.parametrize("smiles,expected", [
    ("CCO", False), ("CCCCCCCOP=O", False), ("c1ccccc1", False), ("[Na+].[Cl-]", False),
    ("C[CH2]", True), ("C[CH]C", True), ("[CH3]", True), ("[O]c1ccc([O])cc1", True),
    ("not a smiles", True),                       # unreadable: no verdict, the legacy reading
])
def test_smiles_has_radical_electrons(smiles, expected):
    assert smiles_has_radical_electrons(smiles) is expected


def test_outside_a_name_call_the_legacy_reading_stays():
    assert innermost_name_input() is None
    assert input_allows_radicals() is True
    assert cache_key("ethanol") == "ethanol"


def test_inside_a_name_call_the_input_decides(named_input):
    named_input("CCCCCCCOP=O")
    assert input_allows_radicals() is False
    assert cache_key("ethanol") == ("ethanol", False)


def test_the_innermost_name_frame_decides_and_the_outer_one_returns(named_input):
    """The engine names a radical fragment of a closed-shell input through a nested name call
    (the nitro group is named for ``O=[N+][O-]``): that frame reads for its fragment."""
    named_input("CCCCCCCOP=O")
    named_input("O=[N+][O-]")
    assert input_allows_radicals() is True
    exit_name_scope()
    assert innermost_name_input() == "CCCCCCCOP=O" and input_allows_radicals() is False
    enter_name_scope("O=[N+][O-]")             # the fixture closes it with the others


def test_after_name_returns_the_base_input_is_read():
    with reading_base_for("CCCCCCCOP=O"):
        assert input_allows_radicals() is False
        enter_name_scope("C[CH]C")
        try:
            assert input_allows_radicals() is True
        finally:
            exit_name_scope()
        assert input_allows_radicals() is False
    assert input_allows_radicals() is True


def test_the_full_round_trip_that_promotes_a_shipped_name_reads_for_the_input():
    """``name_tiered`` sets the label of the name it ships after ``name`` returned, where no name
    frame is live: ``_shipped_name_round_trip`` must read for the input there, not with ``-r``."""
    from orthonym.namer import _shipped_name_round_trip
    assert _shipped_name_round_trip(RADICAL_ONLY_NAME, CLOSED_SHELL) == "verified"      # legacy reading
    with reading_base_for(CLOSED_SHELL):
        assert _shipped_name_round_trip(RADICAL_ONLY_NAME, CLOSED_SHELL) == "failed"
        assert _shipped_name_round_trip("1-methoxyheptane", "CCCCCCCOC") == "verified"


def test_a_failed_name_call_leaves_no_frame_and_no_forced_reading():
    with pytest.raises(ValueError):
        Orthonym().name("not a smiles")
    assert innermost_name_input() is None
    assert input_allows_radicals() is True


def test_a_check_that_knows_its_molecule_overrides_the_input(named_input):
    named_input(CLOSED_SHELL)
    with reading_for(Chem.MolFromSmiles("C[As]C")):          # a radical fragment of the input
        assert input_allows_radicals() is True
    assert input_allows_radicals() is False
    named_input("C[CH]C")
    with reading_for(Chem.MolFromSmiles("CCO")):
        assert input_allows_radicals() is False
    assert input_allows_radicals() is True


def test_legacy_reading_forces_the_radical_reading(named_input):
    named_input(CLOSED_SHELL)
    with legacy_reading():
        assert input_allows_radicals() is True
        assert cache_key("x") == "x"
    assert input_allows_radicals() is False


def test_charged_router_gate_keeps_the_radical_fragments_names(named_input):
    """'dimethylarsanyl' is the name route_charged gives the C[As]C fragment of a closed-shell
    4-(dimethylarsanyl)quinoline: the check follows the fragment, not the input."""
    named_input("C[As](C)c1ccnc2ccccc12")
    assert _full_inchikey_rt_ok(Chem.MolFromSmiles("C[As]C"), "dimethylarsanyl") is True
    assert _full_inchikey_rt_ok(Chem.MolFromSmiles("CCO"), "ethanol") is True
    assert _full_inchikey_rt_ok(Chem.MolFromSmiles("CCO"), "ethyl") is False   # a closed-shell mol


def test_arsenic_pin_survives_the_policy():
    out = Orthonym().name_tiered("C[As](C)c1ccnc2ccccc12")
    assert (out.get("name"), out.get("tier")) == ("4-(dimethylarsanyl)quinoline", "pin_verified"), out


# ---- the oracle -------------------------------------------------------------------------

def test_a_name_with_free_valences_reads_only_under_the_radical_reading(oracle, named_input):
    assert opsin_parse(RADICAL_ONLY_NAME) is None            # no -r: OPSIN refuses it
    assert oracle.parse_status(RADICAL_ONLY_NAME) == "parsed"      # no input: legacy -r
    named_input(CLOSED_SHELL)
    assert oracle.parse_status(RADICAL_ONLY_NAME) == "rejected"
    assert oracle.name_to_smiles(RADICAL_ONLY_NAME) is None
    assert oracle.name_to_opsin_smiles(RADICAL_ONLY_NAME) is None


def test_the_two_readings_do_not_share_cache_entries(oracle, named_input):
    named_input("C[CH]C")
    assert oracle.name_to_smiles("propan-2-yl") is not None  # a radical input keeps its names
    named_input(CLOSED_SHELL)
    assert oracle.name_to_smiles("propan-2-yl") is None      # the same name, closed-shell reading
    named_input("C[CH]C")
    assert oracle.name_to_smiles("propan-2-yl") is not None  # and the first entry was not poisoned


def test_a_closed_shell_name_reads_the_same_under_both_readings(oracle, named_input):
    legacy = oracle.name_to_smiles("1-methoxyheptane")
    named_input("CCCCCCCOC")
    assert oracle.name_to_smiles("1-methoxyheptane") == legacy is not None


# ---- naming -----------------------------------------------------------------------------

def test_a_radical_input_is_still_named_as_a_radical():
    out = Orthonym().name_tiered("C[CH]C")
    assert out.get("name") == "propan-2-yl", out


def test_a_producer_that_spells_the_old_radical_only_prefix_cannot_ship(monkeypatch):
    """The pre-fix producer returned '(phosphoryl)oxy' for the bare -O-P=O. The read-back of the
    closed-shell input must refuse every name built from it, at both tiers, and ship nothing that
    OPSIN reads only with ``-r``. Positive control: with the legacy reading the same patched
    producer ships it (that is the hole)."""
    def old_producer(mol, o_idx, from_idx, _depth=0):
        return '(phosphoryl)oxy' if mol.GetAtomWithIdx(o_idx).GetDegree() == 2 else None
    best = dict(general_fallback=True, general_fallback_unverified=True, allow_aromatic_general=True)

    def names(allow):
        monkeypatch.setattr(phosphorus, "name_phosphoanhydride_oxy_substituent", old_producer)
        monkeypatch.setattr("orthonym.validation.read_back_policy.input_allows_radicals",
                            lambda: allow)
        return [Orthonym(**kw).name_tiered(CLOSED_SHELL) for kw in ({}, best)]

    for res in names(False):
        n = res.get("name") or ""
        assert "(phosphoryl)oxy" not in n, res
        if res.get("tier") not in ("abstain", None):
            back = opsin_parse(n)
            assert back is not None and _key(back) == _key(CLOSED_SHELL), res
    shipped = [r for r in names(True) if "(phosphoryl)oxy" in (r.get("name") or "")]
    assert shipped, "positive control: the legacy reading ships the radical-only name"
