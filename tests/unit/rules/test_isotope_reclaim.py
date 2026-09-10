""" a phase — isotope P-82 best-effort reclaim (0-wrong).

The isotope decorator named the isotope-STRIPPED skeleton with a FLAG-LESS
``Orthonym(style="systematic")`` (``rules/isotopes.py``), so a skeleton the general
engine could only build at the best-effort tier returned ``'unknown organic compound'``
and the decoration failed closed (``isotope_decorator_failed``). Propagating the
caller's ``general_fallback`` / ``general_fallback_unverified`` / ``allow_aromatic_general``
flags to that fresh engine (via ``_flagged_systematic_namer``) lets the skeleton name at
the caller's tier, so the isotope label composes onto it. Best-effort only, and
``_isotope_round_trips`` gates every emission → 0-wrong; PIN tier byte-identical.

Measured (p4-trace, frozen isotope stratum): 57/73 rows skeleton-name after the fix
(decoration-only failures), 19/73 = 26% reclaim end-to-end at 0-wrong.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.namer import Orthonym
from orthonym.jvm_budget import jvm_slots
from orthonym.validation import opsin_roundtrip_check


def _best_effort():
    return Orthonym(style="pin", general_fallback=True,
                     general_fallback_unverified=True, allow_aromatic_general=True)


# (input SMILES, exact best-effort name) — each verified 0-wrong end-to-end.
RECLAIM_CASES = [
    # p4-trace witness: a deuterated oxatricyclic skeleton the general engine names
    # only at the best-effort tier (PIN abstains on the fused tricycle).
    # (item 3, P-82.6.1.1): the descriptor carries its REQUIRED locant and is
    # front-placed. The old locant-free `3-(2H1)oxatricyclo…` was Blue-Book-wrong
    # (it only round-tripped via OPSIN's default placement on this asymmetric cage);
    # positions 2 and 4 are inequivalent so the locant cannot be omitted.
    ("[2H][C@@]12[C@@H](O1)CCC3=CC=CC=C23",
     "(2R,4S)-(2-2H1)-3-oxatricyclo[5.4.0.0^2,4]undeca-1(11),7,9-triene"),
]


@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles,expected", RECLAIM_CASES)
def test_isotope_skeleton_reclaimed_at_best_effort(smiles, expected):
    """Best-effort composes the isotope label onto a general-engine skeleton AND it
    full-InChIKey round-trips to the exact isotopologue."""
    with jvm_slots(1, purpose="p4-isotope-reclaim-test"):
        name = _best_effort().name(smiles)
        assert name == expected
        rt = opsin_roundtrip_check(smiles, name)
        assert rt.get("passed"), f"{name!r} did not round-trip: {rt.get('error')}"
        opsin_mol = Chem.MolFromSmiles(rt["opsin_smiles"])
        assert opsin_mol is not None
        # 0-wrong: exact isotopologue, full InChIKey (isotope layer included).
        assert (inchi.MolToInchiKey(opsin_mol)
                == inchi.MolToInchiKey(Chem.MolFromSmiles(smiles)))


@pytest.mark.parametrize("smiles,expected", RECLAIM_CASES)
def test_pin_tier_still_abstains_on_the_best_effort_skeleton(smiles, expected):
    """Tripwire: the PIN tier (no best-effort flags) still cannot name the skeleton,
    proving the flag propagation — not some other change — is what enables the reclaim.
    PIN output is byte-identical (the flags are False)."""
    with jvm_slots(1, purpose="p4-isotope-reclaim-test"):
        name = Orthonym(style="pin").name(smiles)
    assert "unknown" in name.lower()
