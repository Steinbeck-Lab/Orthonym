"""A neutral trivalent amine nitrogen whose configuration standard InChI does NOT
retain (a quinuclidine-type bridgehead/cage amine, or any pyramidal-inverting
amine) must not carry a CIP label into naming: keeping it makes the PIN path emit
an OPSIN-unparseable descriptor at the bridgehead locant, and makes best-effort
over-count stereo and abstain. So quinine written with an explicit [N@] used to
FAIL to name though the tag-free form named fine (an input-order sensitivity).

The fix (perception/stereo.py::_clear_nonexpressible_amine_n_cip) clears the CIP
label ONLY when clearing that atom's chiral tag leaves MolToInchi byte-identical,
so it can never drop a descriptor InChI retains (0-wrong). Genuine, InChI-retained
N stereocentres (chiral N+, amine N-oxide) must keep their descriptor.

These assertions are deterministic and do NOT call live OPSIN: `opsin_parse` is
flaky under heavy concurrent JVM load (leaked hsperfdata pollutes a JVM's stdout),
and the round-trip correctness of these names is verified in isolation elsewhere.
The regression guard here is INPUT-INVARIANCE (the [N@] form names identically to
the tag-free form) plus descriptor-presence for the must-not-strip witnesses.

Cite: pyramidal amine inversion (a neutral trivalent amine N is not a
configurationally citable stereogenic unit); standard InChI + OPSIN both treat it
as non-stereogenic. All three quinine forms share InChIKey LOUPRKONTZGTKE-WZBLMQSHSA-N.
"""
import pytest

from orthonym import Orthonym

QUININE_NO_NSTEREO = "COC1=CC2=C(C=CN=C2C=C1)[C@H]([C@@H]3C[C@@H]4CCN3C[C@@H]4C=C)O"
QUININE_N_TAG = "C=C[C@H]1C[N@]2CC[C@H]1C[C@H]2[C@H](O)C1=CC=NC2=C1C=C(OC)C=C2"


@pytest.fixture(scope="module")
def eng():
    return Orthonym(general_fallback=True, general_fallback_unverified=True,
                     allow_aromatic_general=True)


@pytest.mark.unit
def test_quinine_names_input_invariant_over_bridgehead_n_stereo(eng):
    """The explicit-[N@] form names identically to the tag-free form (was: abstain).
    Input-invariance is the deterministic guard; it does not depend on live OPSIN."""
    nb = eng.name_tiered(QUININE_N_TAG).get("name")
    na = eng.name_tiered(QUININE_NO_NSTEREO).get("name")
    assert nb and "unknown" not in nb, "explicit-[N@] quinine still abstains"
    assert nb == na, f"not input-invariant:\n  [N@]: {nb}\n  bare: {na}"


@pytest.mark.unit
def test_cinchonidine_class_generalises(eng):
    """Same quinuclidine-bridgehead class (cinchonidine) now names (was: abstain)."""
    nm = eng.name_tiered(
        "C=C[C@H]1C[N@]2CC[C@H]1C[C@H]2[C@H](O)c1ccnc2ccccc12").get("name")
    assert nm and "unknown" not in nm


@pytest.mark.unit
@pytest.mark.parametrize(
    "smi",
    [
        "CCC[N@+](C)(CC)Cc1ccccc1",   # chiral quaternary ammonium N+ (InChI RETAINS)
        "CC[N@+](C)([O-])CCC",        # chiral amine N-oxide (InChI RETAINS)
    ],
)
def test_genuine_n_stereocentre_descriptor_not_stripped(eng, smi):
    """A configurationally citable N (charge/degree excluded by the predicate)
    must keep its stereo descriptor -- the fix must not touch it."""
    nm = eng.name_tiered(smi).get("name")
    assert nm and "unknown" not in nm
    assert "(2S)" in nm or "(S)" in nm or "(R)" in nm, f"descriptor stripped: {nm}"
