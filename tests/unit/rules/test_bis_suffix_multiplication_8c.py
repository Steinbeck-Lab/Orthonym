""" — a phase Item 3 Case A: bis-suffix multiplication for >= 2
identical charged skeletal centres on ONE parent (P-73.5.1 / P-73.2.2.1.1).

Case A is suffix multiplication (the charged atom is the skeletal / characteristic
group of ONE parent) -> ``parent-<locants>-bis(SUFFIX)``:

  * ylium (carbenium hydride-loss, P-73.2.2.1.1, the Blue Book "bis(ylium)" NOT
    "diylium") — acyclic all-carbon parent.
  * nitrilium (protonated nitrile, P-73.5.1.2) — routed through the poly-aminium
    branch (classify_cation lumps the protonated-nitrile N as 'aminium'); the
    neutral-name ending 'dinitrile' disambiguates it from a real poly-amine.
  * diazonium (terminal -N#N+, P-73.5.1.1) — ring parent, attachment locants.

The ring ylium (``c1c[cH+][cH+]1`` -> cyclobut-3-ene-1,2-bis(ylium)) is deferred
to Task 8E (it needs the Item-4 ring-cation numbering); it must ABSTAIN here.

Invariant 9 / 0-wrong: single-charge ylium/diazonium are unchanged, every new
bis-name OPSIN-round-trips to its input, and any out-of-scope multi-cation
(mixed / hetero / substituted / off-parent) fails closed to an abstention rather
than a wrong bis-name.
"""
import pytest
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

from orthonym.namer import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse


def _name(smi: str) -> str:
    return Orthonym().name(smi)


def _abstains(name: str) -> bool:
    low = (name or "").lower()
    return (not name) or "unknown" in low or "not supported" in low


# === Case A shipped rows (verified abstain -> required PIN) ====================

SHIPPED = {
    "[CH2+]C[CH2+]": "propane-1,3-bis(ylium)",          # P-73.2.2.1.1 / the Blue Book
    "[CH2+][CH2+]": "ethane-1,2-bis(ylium)",            # P-73.2.1
    "[CH2+]CC[CH2+]": "butane-1,4-bis(ylium)",          # general acyclic
    "[NH+]#CCCC#[NH+]": "butanebis(nitrilium)",         # P-73.5.1.2
    "N#[N+]c1ccc([N+]#N)cc1": "benzene-1,4-bis(diazonium)",   # P-73.5.1.1 (para)
    "N#[N+]c1ccccc1[N+]#N": "benzene-1,2-bis(diazonium)",     # ortho
    "N#[N+]c1cccc([N+]#N)c1": "benzene-1,3-bis(diazonium)",   # meta
    "N#[N+]c1cc([N+]#N)cc([N+]#N)c1": "benzene-1,3,5-tris(diazonium)",  # tris
}


@pytest.mark.parametrize("smi,expected", SHIPPED.items())
def test_bis_suffix_pin(smi, expected):
    assert _name(smi) == expected


@pytest.mark.parametrize("smi,expected", SHIPPED.items())
def test_bis_suffix_round_trips(smi, expected):
    """0-wrong: every shipped bis-name parses back to the input structure."""
    parsed = opsin_parse(expected)
    assert parsed, f"OPSIN could not parse {expected!r}"
    pm = Chem.MolFromSmiles(parsed)
    assert pm is not None
    assert MolToInchiKey(pm) == MolToInchiKey(Chem.MolFromSmiles(smi))


# === Deferred to Task 8E (ring ylium needs Item-4 numbering) — must abstain ====

def test_ring_ylium_deferred_abstains():
    assert _abstains(_name("c1c[cH+][cH+]1"))


# === Fail-closed: out-of-scope multi-cation shapes abstain, never a wrong name =

FAIL_CLOSED = [
    "[CH2+]C(C)[CH2+]",        # substituted acyclic poly-ylium -> out of v1 scope
    "[CH2+]C([CH2+])C[CH2+]",  # centres not collinear on one chain
    "[CH2+]C[N+]#N",           # mixed ylium + diazonium (heterogeneous)
    "N#[N+]CC[N+]#N",          # acyclic (non-ring) poly-diazonium -> no ring parent
]


@pytest.mark.parametrize("smi", FAIL_CLOSED)
def test_out_of_scope_fails_closed(smi):
    assert _abstains(_name(smi))


# === Single-charge rows UNCHANGED (a project rule) ===============================

SINGLE_UNCHANGED = {
    "[CH3+]": "methylium",                 # single ylium
    "CC[CH2+]": "propylium",               # single ylium
    "c1ccccc1[N+]#N": "benzenediazonium",  # single diazonium
    "[NH+]#CC": "acetonitrilium",          # single nitrilium
}


@pytest.mark.parametrize("smi,expected", SINGLE_UNCHANGED.items())
def test_single_charge_unchanged(smi, expected):
    assert _name(smi) == expected
