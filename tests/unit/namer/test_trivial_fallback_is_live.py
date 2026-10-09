"""``--trivial`` is a LIVE flag, not an inert one — and it can only ever ADD.

 residue Task K. The residue item recorded "``--trivial`` unreachable".
**The premise is refuted twice over**: the flag is registered in the CLI
(``cli.py:159``), threaded through to ``Orthonym(trivial_fallback=...)``
(``cli.py:388``), and it demonstrably changes the emitted name. Enumerating the
ONLY set it can possibly affect — the 150 entries of
``data.GENERAL_RETAINED_NAMES``, since ``namer.py::_apply_trivial_fallback``
looks the molecule up there by canonical SMILES — found 5 molecules whose name
differs with the flag set, confirmed in fresh CLI processes.

The flag's contract (``namer.py::_apply_trivial_fallback``) is that ALL of the
following must hold before it substitutes anything:

  1. ``self._trivial_fallback`` is set (opt-in);
  2. the call is top level (recursive fragment calls are unaffected);
  3. naming produced ONLY the failure signal (``errors.is_failure_name``);
  4. a general-only, PIN-denied retained name exists for the molecule.

Condition (3) is the safety property that matters: a real name is never a
failure name, so the flag is structurally incapable of downgrading a derived
PIN to a non-PIN trivial name. ``test_trivial_never_downgrades_a_derived_pin``
below pins that with a molecule that is BOTH nameable and present in the
general-only table, so the two halves of the contract are in genuine tension.

Nothing here asserts that a particular molecule is or is not PIN-deniable —
that is the business of ``data/`` and its own gate, and those tables move. The
assertions are about the FLAG's wiring and its fail-safe direction.
"""

import pytest

from orthonym.data import GENERAL_RETAINED_NAMES
from orthonym.errors import is_failure_name
from orthonym.namer import Orthonym

# ⚠ REQUIRED. ``tests/conftest.py:371`` turns the OPSIN validity gate OFF for
# every test by default; ``pytest.mark.opsin_gate`` (``conftest.py:379``) is the
# only supported way to get production behaviour back.
#
# This is not a formality here — it is load-bearing, and it is a finding in its
# own right. Clause (3) of the fallback's contract requires the default pipeline
# to have produced a FAILURE name, and for four of the five measured molecules
# the thing that produces that failure is the OPSIN gate suppressing a name the
# generator did emit. With the gate off, ``[C-]#[N+]O`` emits
# ``N-hydroxy-λ2-methanamine`` (which rejects as a DIFFERENT molecule)
# and ``CC(=O)Nc1ccc(O)cc1`` emits a name too, so the fallback never fires and
# every assertion below inverts. The reachability of ``--trivial`` is therefore
# a property of the GATED configuration, not of the generator.
pytestmark = pytest.mark.opsin_gate


# Measured live set (fresh CLI process per molecule): every one of these emits
# a failure name by default and the general-only retained name under --trivial.
# The tables these are drawn from are actively curated, so the suite requires
# only that the flag is STILL LIVE on at least one of them, not on all five.
MEASURED_LIVE = [
    ("CC(=O)Nc1ccc(O)cc1", "paracetamol"),
    ("NC(=S)SSC(N)=S", "thiuram disulfide"),
    ("O=C(O)CC(O)(CC(=O)O)C(=O)O", "trihydrocitrate"),
    ("[C-]#[N+]O", "isofulminic acid"),
]


# Rows of MEASURED_LIVE retired because the default pipeline now derives a name (the gate no
# longer suppresses anything for them). Each name is OPSIN 2.9.0 round-trip exact to the
# input's full InChIKey (an InChIKey, an InChIKey);
# citric acid's is the Blue Book PIN, "Retained names only for general
# nomenclature" (the Blue Book '2-hydroxypropane-1,2,3-tricarboxylic acid (PIN)').
NOW_DERIVED = {
    "CC(=O)Nc1ccc(O)cc1": "N-(4-hydroxyphenyl)acetamide",
    "O=C(O)CC(O)(CC(=O)O)C(=O)O": "2-hydroxypropane-1,2,3-tricarboxylic acid",
}


@pytest.fixture(scope="module")
def plain():
    return Orthonym()


@pytest.fixture(scope="module")
def trivial():
    return Orthonym(trivial_fallback=True)


def test_the_flag_changes_at_least_one_emitted_name(plain, trivial):
    """The whole point of Task K: ``--trivial`` is not inert.

    If this ever fails, the flag has become a user-facing control that
    silently does nothing — which is a defect whichever way it is resolved
    (wire it, or remove it from the CLI).
    """
    live = []
    for smiles, _expected in MEASURED_LIVE:
        if smiles not in GENERAL_RETAINED_NAMES:
            continue  # withdrawn from the general-only table; not this test's business
        if plain.name(smiles) != trivial.name(smiles):
            live.append(smiles)
    assert live, (
        "--trivial changed no name across %d measured candidates: the flag is "
        "INERT. Either wire it or remove it from cli.py." % len(MEASURED_LIVE)
    )


@pytest.mark.parametrize(
    "smiles,expected", MEASURED_LIVE, ids=[s for s, _ in MEASURED_LIVE]
)
def test_flag_emits_the_general_only_name_when_pin_derivation_fails(
    plain, trivial, smiles, expected
):
    """Contract clauses (3) + (4): failure name in, table value out."""
    if GENERAL_RETAINED_NAMES.get(smiles) != expected:
        pytest.skip("%r no longer maps to %r in GENERAL_RETAINED_NAMES" % (smiles, expected))
    default = plain.name(smiles)
    if smiles in NOW_DERIVED:
        # Retired witness: the systematic pipeline now derives a real name for it, so the
        # fallback can no longer fire (clause 3) -- an IMPROVEMENT, not a regression. The row
        # keeps its id and now pins the safety half of the contract on the same molecule:
        # the flag must NOT replace a derived name with the trivial one.
        assert default == NOW_DERIVED[smiles]
        assert trivial.name(smiles) == default != expected
        return
    assert is_failure_name(default), (
        "precondition moved: %s now names as %r, so the fallback can no longer "
        "fire on it (clause 3). This is an IMPROVEMENT, not a regression — "
        "retire this row (add it to NOW_DERIVED)." % (smiles, default)
    )
    assert trivial.name(smiles) == expected


def test_trivial_never_downgrades_a_derived_pin(plain, trivial):
    """Contract clause (3), the safety half — the fail-safe direction.

    ``ClC(Cl)Cl`` is the sharp case: it is in ``GENERAL_RETAINED_NAMES``
    (-> ``chloroform``), so clause (4) is satisfied, yet the systematic
    pipeline derives a real PIN for it. Clause (3) must therefore veto the
    substitution. A regression that dropped the ``is_failure_name`` guard
    would turn ``trichloromethane`` into ``chloroform`` and would be caught
    here and nowhere else in this suite.
    """
    assert GENERAL_RETAINED_NAMES.get("ClC(Cl)Cl") == "chloroform", (
        "test precondition: chloroform must be a general-only (PIN-denied) name"
    )
    assert plain.name("ClC(Cl)Cl") == "trichloromethane"
    assert trivial.name("ClC(Cl)Cl") == "trichloromethane"


def test_flag_defaults_off(plain):
    """PIN fails closed by default: the fallback is opt-in only."""
    assert plain._trivial_fallback is False
    assert plain.name("CC(=O)Nc1ccc(O)cc1") != "paracetamol"
