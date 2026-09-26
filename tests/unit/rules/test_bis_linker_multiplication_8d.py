""" — a phase Item 3 Case B: multiplicative LINKER for >= 2 identical
parent-ion units on ONE di/polyvalent linker /.

Case B is a MULTIPLICATIVE parent (a different mechanism from Case A's suffix
multiplication): the charged unit hangs off a shared di/polyvalent LINKER ->
``(LINKER)bis(PARENT-ION)``:

  * a heteroatom-hydride -ide anion (P/As/Si/…) or an onium cation (O/S/P/Se+)
    bonded DIRECTLY to the linker -> a benzene phenylene / linear alkanediyl
    linker + the bare free-ion unit
    ([PH-]c1ccc([PH-])cc1 -> (1,4-phenylene)bis(phosphanide),
     [PH3+]c1ccc([PH3+])cc1 -> (1,4-phenylene)bis(phosphanium)).
  * an acyl-azanide anion (R-CO-NH-) sharing a di/polyacyl linker ->
    ``<polyacyl>bis(azanide)``
    ([NH-]C(=O)CCC([NH-])=O -> butanedioylbis(azanide)).

Invariant 9 / 0-wrong: emit_bis_quaternary_ammonium / single-ion / Case-A rows
are unchanged; every new name OPSIN-round-trips to its input; and any out-of-scope
shape (substituted unit, fused/hetero linker, asymmetric units) fails closed to an
abstention rather than a wrong multiplicative name.

⚠ Out of Case B scope (a substitutive azaniumyl-prefix form, NOT a linker):
``[NH3+]CCC(C[NH3+])CC[NH3+]`` (verified PIN
``3-(azaniumylmethyl)pentane-1,5-bis(aminium)``). Its blocker is an upstream
NEUTRAL polyamine parent-selection bug (``NCCC(CN)CCN`` is mis-named
``3-(aminomethyl)pentane-1,3,5-triamine``), so it currently abstains (0-wrong).
Pinned here as ``test_target4_abstains`` so a later neutral-amine fix is noticed.
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


# === Case B shipped rows (verified abstain -> required PIN) ====================

SHIPPED = {
    # required targets (task-8D brief)
    "[PH-]c1ccc([PH-])cc1": "(1,4-phenylene)bis(phosphanide)",   # the Blue Book
    "[PH3+]c1ccc([PH3+])cc1": "(1,4-phenylene)bis(phosphanium)",  # the Blue Book
    "[NH-]C(=O)CCC([NH-])=O": "butanedioylbis(azanide)",          #
    # phenylene isomers (lowest ring locants to the ion set)
    "[PH-]c1ccccc1[PH-]": "(1,2-phenylene)bis(phosphanide)",      # ortho
    "[PH3+]c1cccc([PH3+])c1": "(1,3-phenylene)bis(phosphanium)",  # meta
    # other heteroatom-hydride elements on a phenylene
    "[AsH-]c1ccc([AsH-])cc1": "(1,4-phenylene)bis(arsanide)",     # As
    # linear alkanediyl linker (the emit_bis_quaternary_ammonium bridge, anion side)
    "[PH-]CCCC[PH-]": "butane-1,4-diylbis(phosphanide)",
    # longer acyl linker
    "[NH-]C(=O)CCCC([NH-])=O": "pentanedioylbis(azanide)",
}


@pytest.mark.parametrize("smi,expected", SHIPPED.items())
def test_bis_linker_pin(smi, expected):
    assert _name(smi) == expected


@pytest.mark.parametrize("smi,expected", SHIPPED.items())
def test_bis_linker_round_trips(smi, expected):
    """0-wrong: every shipped multiplicative name parses back to the input."""
    parsed = opsin_parse(expected)
    assert parsed, f"OPSIN could not parse {expected!r}"
    pm = Chem.MolFromSmiles(parsed)
    assert pm is not None
    assert MolToInchiKey(pm) == MolToInchiKey(Chem.MolFromSmiles(smi))


# === Fail-closed: my emitters return '' on out-of-scope shapes ================
# These test the Case B EMITTERS directly, so they are independent of the
# suite-wide OPSIN validity gate (disabled by default, tests/conftest.py) — a
# '' return is the load-bearing 0-wrong guarantee that no wrong multiplicative
# name is produced, whatever the downstream best-effort path then does.

def _sites(smi):
    from orthonym.perception.ions import get_ion_sites
    return Chem.MolFromSmiles(smi), get_ion_sites(Chem.MolFromSmiles(smi))


def test_emitter_declines_substituted_anion_units():
    from orthonym.rules.charged_router import _name_bis_parent_ion_linker
    mol, sites = _sites("C[P-]c1ccc([P-]C)cc1")   # methylphosphanide units, out of scope
    idxs = [a["atom_idx"] for a in sites["anions"]]
    assert _name_bis_parent_ion_linker(mol, idxs, "heteroatom_hydride_anion", "pin") == ""


def test_emitter_declines_fused_ring_linker():
    from orthonym.rules.charged_router import _name_bis_parent_ion_linker
    mol, sites = _sites("[PH-]c1ccc2ccccc2c1[PH-]")   # naphthalene linker, not phenylene
    idxs = [a["atom_idx"] for a in sites["anions"]]
    assert _name_bis_parent_ion_linker(mol, idxs, "heteroatom_hydride_anion", "pin") == ""


def test_emitter_declines_non_acyl_azanide():
    from orthonym.rules.charged_router import _name_bis_acyl_azanide
    mol, sites = _sites("[NH-]CC[NH-]")   # aminide (not on an acyl carbon) -> ''
    idxs = [a["atom_idx"] for a in sites["anions"]]
    assert _name_bis_acyl_azanide(mol, idxs, "pin") == ""


# === Fail-closed end-to-end: production (gate ON) abstains, never a wrong ship =
# With the OPSIN validity gate ON (as in production / the CLI), the best-effort
# charge-dropped fallthrough for these out-of-scope shapes is -suppressed,
# so the molecule abstains rather than shipping a different-molecule name.

FAIL_CLOSED = [
    "C[P-]c1ccc([P-]C)cc1",      # substituted P- units (>1 heavy nbr) -> out of scope
    "[PH-]c1ccc2ccccc2c1[PH-]",  # fused-ring (naphthalene) linker -> not phenylene
]


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smi", FAIL_CLOSED)
def test_out_of_scope_abstains_in_production(smi):
    assert _abstains(_name(smi))


# === Single-ion / Case-A / bis-quaternary rows UNCHANGED (a project rule) ========

UNCHANGED = {
    "C[P-]C": "dimethylphosphanide",                 # single heteroatom-hydride anion
    "[PH-]C": "methylphosphanide",
    "C[Si-](C)C": "trimethylsilanide",
    "CC(=O)[NH-]": "acetylazanide",                  # single acyl azanide
    "C[N+](C)(C)CCCCCC[N+](C)(C)C":                 # emit_bis_quaternary_aminium:
        "N1,N1,N1,N6,N6,N6-hexamethylhexane-1,6-bis(aminium)",  # PIN (:42154)
    "[NH3+]CC[NH3+]": "ethane-1,2-bis(aminium)",     # poly-aminium suffix path
    "[NH-]CC[NH-]": "ethane-1,2-bis(aminide)",       # poly-aminide suffix path
    "[O-]CC[O-]": "ethane-1,2-bis(olate)",           # poly-olate suffix path
    "[CH2+]C[CH2+]": "propane-1,3-bis(ylium)",       # Case A ylium
    "N#[N+]c1ccc([N+]#N)cc1": "benzene-1,4-bis(diazonium)",  # Case A diazonium
}


@pytest.mark.parametrize("smi,expected", UNCHANGED.items())
def test_unchanged(smi, expected):
    assert _name(smi) == expected


# === Target 4: out of Case B scope, abstains (upstream neutral-amine bug) ======

@pytest.mark.opsin_gate
def test_target4_abstains():
    """[NH3+]CCC(C[NH3+])CC[NH3+] is a substitutive azaniumyl-prefix form (verified
    PIN 3-(azaniumylmethyl)pentane-1,5-bis(aminium)), NOT a multiplicative linker.
    It is blocked by an upstream neutral polyamine parent-selection defect
    (NCCC(CN)CCN is itself mis-named 3-(aminomethyl)pentane-1,3,5-triamine), so in
    production it stays 0-wrong (suppresses the wrong best-effort name)."""
    assert _abstains(_name("[NH3+]CCC(C[NH3+])CC[NH3+]"))
