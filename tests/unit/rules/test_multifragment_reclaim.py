""" a phase — multi-component reclaim (flag-forward: lever 1c + 1a residual).

The dispatch caller in ``namer.py`` forwards the best-effort flags
(``general_fallback`` / ``allow_aromatic_general`` / ``general_fallback_unverified``)
into the dispatch handlers, so ``_handle_multi_component_neutral`` -> ``name_adduct``
names a multi-fragment input whose only unnameable component needs the general engine
tier. PIN tier (all three flags default ``False``) forwards ``False`` -> byte-identical
output (a trace: `internal notes`). Rule for the adduct spelling:
 (components joined by an em dash, proportion ``(n/m)`` appended after a space).
"""
import pytest

from orthonym.jvm_budget import jvm_slots
from orthonym.namer import Orthonym
from orthonym.errors import is_failure_name


def _best_effort():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)


def test_1c_neutral_composer_under_best_effort():
    """A neutral 2-fragment mixture whose one hard component names only under the
    general engine now composes (lever 1c).."""
    # S2c-1 fixture update (was the silabicyclic ring c1ccc2c(c1)[SiH2]cc2, now named
    # '1H-1-benzosilole' at PIN by the benzo name, the Blue Book): the
    # lambda-convention benzodioxathiole is the hard component (PIN abstains; the general
    # engine names it). OPSIN 2.9.0 full-InChIKey exact.
    with jvm_slots(1, purpose="p1-task2-test"):
        name = _best_effort().name("CCO.c1ccc2c(c1)O[SH2]O2")
    assert name == "ethanol—7,9-dioxa-8λ4-thiabicyclo[4.3.0]nona-1,3,5-triene (1/1)"


def test_pin_tier_adduct_unchanged():
    """Regression guard: the default/PIN path forwards the flags as False, so an
    all-trivially-nameable adduct is byte-identical to before the flag-forward."""
    with jvm_slots(1, purpose="p1-task2-test"):
        name = Orthonym(style="pin").name("CCO.CC(=O)O")
    assert name == "acetic acid—ethanol (1/1)"


def test_1a_single_atom_methane_adduct():
    """Lever 1a (shipped): a bare-carbon co-component names as ``methane`` in the
    adduct.."""
    with jvm_slots(1, purpose="p1-task2-test"):
        name = _best_effort().name("C.CCO")
    assert name == "ethanol—methane (1/1)"


# 0-wrong tripwire: a best-effort fragment must NEVER bypass the metal sentinel and
# emit a name that drops or misdraws part of a metal mixture (these are from the
# frozen abstain sample). Breadth Job 2 (the user's decision, 2026-09-28) names a
# disconnected metal drawing at the best-effort tier as a adduct of its
# components, labelled below PIN -- "Mixed organic - inorganic adducts"
# (the Blue Book): "preferred IUPAC names cannot be assigned to mixed adducts
# because preferred IUPAC names have not yet been determined for inorganic
# components". So each mixture either still abstains, or ships a name that a fresh
# OPSIN call reads back to exactly the drawn structure (same canonical SMILES: every
# fragment, charge and hydrogen), never a PIN label.
@pytest.mark.parametrize("smi", [
    # bare tungsten metal + two neutral organics
    "CC(C)(C)C1=NC2=C(CCNC2)C=C1.CC(C)(C)C1=CC=CC=N1.[W]",
    # bare Pt(2+) ion + a carbanion
    "C1=CC=NC(=C1)C2=[C-]C(=CC=C2)C3=CC=CC=N3.C1=C[N-]N=C1.[Pt+2]",
    # bare Fe(2+) ion in a net-charged assembly
    "C1=CC(=CC=C1NC[C]2[CH][CH][CH][CH]2)N=CC3=CC=C(O3)[N+](=O)[O-]."
    "[CH]1[CH][CH][CH][CH]1.[Fe+2]",
])
def test_metal_mixture_abstains_or_names_exactly_the_drawing(smi):
    from rdkit import Chem
    from tests.support.rt_assert import _independent_parse
    with jvm_slots(1, purpose="p1-task2-test"):
        row = _best_effort().name_tiered(smi)
    name = row.get("name")
    if name is None or is_failure_name(name):
        return  # still abstains
    parsed = _independent_parse(name)
    canon = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))  # noqa: E731
    assert parsed and canon(parsed) == canon(smi), (
        f"metal mixture emitted {name!r}, which reads back as {parsed!r}")
    assert row["is_pin"] is False and row["tier"] in (
        "systematic_verified", "best_effort"), (name, row["tier"])


# ---- lever 1b: net-0 salts with a complex organic ion (fresh-instance route) ----

def test_1b_salt_complex_organic_cation_best_effort():
    """A net-0 salt whose organic cation the ordinary namer cannot build now names
    via the best-effort fresh-instance route; the anion word is appended as a
    separate word. OPSIN round-trips this to the input full InChIKey."""
    with jvm_slots(1, purpose="p1-task2-test"):
        name = _best_effort().name("C1=CC=C2C(=C1)[Se]N=[Se+]2.[Cl-]")
    assert name == ("7,9-diselena-8-azabicyclo[4.3.0]nona-1,3,5,7-tetraen-7-ium "
                    "chloride")


def test_1b_pin_salt_unchanged():
    """Regression guard: an ordinary salt names identically on the PIN path and
    under best-effort — the 1b widening only adds a fallback for ions the PIN
    namer already fails on, so it never changes a name the PIN path produced."""
    with jvm_slots(1, purpose="p1-task2-test"):
        pin = Orthonym(style="pin").name("[Na+].CC(=O)[O-]")
        be = _best_effort().name("[Na+].CC(=O)[O-]")
    assert pin == "sodium acetate"
    assert be == "sodium acetate"


# ---- lever 1e: identical-only X.X (best-effort flag threading, space-join form) ----

def test_1e_hard_identical_pair_reclaims_under_best_effort():
    """The 1e reclaim comes from threading the best-effort flags into the identical-
    path fresh instance: a HARD identical fragment (a lambda-convention ring the PIN
    namer abstains on) now names, and the space-join round-trips to the input full
    InChIKey. PIN abstains on the same input (test below), so this is the reclaim."""
    # S2c-1 fixture update (was the silabicyclic ring c1ccc2c(c1)[SiH2]cc2, now named
    # '1H-1-benzosilole' at PIN by the benzo name, the Blue Book): the
    # lambda-convention benzodioxathiole is the hard component (PIN abstains; the general
    # engine names it). OPSIN 2.9.0 full-InChIKey exact.
    with jvm_slots(1, purpose="p1-task2-test"):
        name = _best_effort().name("c1ccc2c(c1)O[SH2]O2.c1ccc2c(c1)O[SH2]O2")
    assert name == ("7,9-dioxa-8λ4-thiabicyclo[4.3.0]nona-1,3,5-triene "
                    "7,9-dioxa-8λ4-thiabicyclo[4.3.0]nona-1,3,5-triene")


def test_1e_flag_threading_changes_hard_identical():
    """Regression guard: the flag threading is what reclaims — the PIN path (no
    flags) does NOT produce the correct space-join for the hard identical pair, so
    best-effort differs from PIN. (The unit conftest disables the OPSIN gate, so the
    PIN producer here emits a gross-mismatch name that production would suppress;
    asserting best-effort != PIN captures the reclaim without depending on that.)"""
    # S2c-1 fixture update (was the silabicyclic ring c1ccc2c(c1)[SiH2]cc2, now named
    # '1H-1-benzosilole' at PIN by the benzo name, the Blue Book): the
    # lambda-convention benzodioxathiole is the hard component (PIN abstains; the general
    # engine names it). OPSIN 2.9.0 full-InChIKey exact.
    with jvm_slots(1, purpose="p1-task2-test"):
        pin = Orthonym(style="pin").name("c1ccc2c(c1)O[SH2]O2.c1ccc2c(c1)O[SH2]O2")
        be = _best_effort().name("c1ccc2c(c1)O[SH2]O2.c1ccc2c(c1)O[SH2]O2")
    assert be == ("7,9-dioxa-8λ4-thiabicyclo[4.3.0]nona-1,3,5-triene "
                  "7,9-dioxa-8λ4-thiabicyclo[4.3.0]nona-1,3,5-triene")
    assert be != pin


def test_1e_trivial_identical_unchanged():
    """A trivially-nameable identical pair is byte-identical on both paths (the
    space-join form, unchanged from before a phase)."""
    with jvm_slots(1, purpose="p1-task2-test"):
        pin = Orthonym(style="pin").name("CCO.CCO")
        be = _best_effort().name("CCO.CCO")
    assert pin == "ethanol ethanol"
    assert be == "ethanol ethanol"


# ---- lever 1d: net-charged multi assembly (best-effort charged em-dash adduct) ----

def test_1d_net_charged_adduct_best_effort():
    """A net-charged multi-fragment ionic assembly composes as a em-dash
    '(1/1)' adduct of its ion components under best-effort; OPSIN preserves the net
    charge and it round-trips to the input full InChIKey."""
    # The cation is a 1-benzothiophene, named by fusion: "Five-membered
    # ring requirement" (the Blue Book): "Fusion nomenclature gives preferred
    # IUPAC names only to compounds having at least two rings of at least five or
    # more members... When fusion names are not allowed, unsaturated von Baeyer
    # ring system names are preferred IUPAC names"; the benzo name,
    # (:11815, '1-benzofuran (PIN)':11827); the '-ium', (:41368). The
    # von Baeyer 'thiabicyclo[4.3.0]nona...-7-ium' asserted before is the non-fusion
    # form of the same cation; the engine has given the fusion name since 982bce233
    # (v52 phase 5, parent-hydride cations), before the paper code f67429619. Both
    # read back (OPSIN 2.9.0, fresh java run) to the input's full InChIKey
    # an InChIKey; mutation check with the old name: exit 0.
    with jvm_slots(1, purpose="p1-task2-test"):
        name = _best_effort().name("C1CC1C2=CC3=C([S+]2C(F)(F)F)C=C(C=C3)F.Cl")
    assert name == ("2-cyclopropyl-6-fluoro-1-(trifluoromethyl)-1-benzothiophen-"
                    "1-ium—hydrogen chloride (1/1)")


def test_1d_pin_net_charged_unchanged():
    """Regression guard: the PIN path keeps the charge refusal (returns the abstain
    sentinel) — the charged-adduct widening is best-effort only."""
    with jvm_slots(1, purpose="p1-task2-test"):
        name = Orthonym(style="pin").name("C1CC1C2=CC3=C([S+]2C(F)(F)F)C=C(C=C3)F.Cl")
    assert is_failure_name(name)
