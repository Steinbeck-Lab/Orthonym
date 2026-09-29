"""Class A -- charged aromatic zwitterions (best-effort tier).

An aromatic heterocycle with a DEPROTONATED ring nitrogen (a skeletal ``-ide``
azolide/azinide anion, conjugated to an exocyclic amidinium /
guanidinium / iminium CATION. The whole species is net-neutral
 / zwitterion).

Before this class the general monocycle producer expressed a charge only when the
NET molecular charge was non-zero; a zwitterion (net 0) hit the ``_has_ionic_
centres`` backstop and abstained to ``unknown organic compound``. The fix
distributes the charge -- the ring parent spells its own ``-ide`` and the
exocyclic cation rides inside a substituent's own ``-ium`` prefix -- and CLAIMS
every charged atom so the P3 charge-totality proof passes.

These are BEST-EFFORT names (the ring is the parent, not the exocyclic iminium),
so the pin is the full-standard-InChIKey ROUND-TRIP, not a byte string: the
best-effort tier is accurate-or-abstain and 0-wrong is delivered by the
whole-name OPSIN round-trip gate. References: IUPAC 2013 (skeletal
anion ``-ide``), (skeletal cation ``-ium``), /
(zwitterion cumulative suffixes).
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym import name_compound
from orthonym.namer import _validity_gate_name_to_smiles as _n2s

pytestmark = pytest.mark.roundtrip

# The witnessed class (QM9 residual). A reference name exists for all five; each
# Orthonym name below is a DIFFERENT but full-InChIKey-equivalent construction.
WITNESSES = [
    "NC=[NH+]C1=CN=N[N-]1",          # triazol-ide + amidinium
    "NC(=[NH2+])C1=COC(=N)[N-]1",    # oxazol-ide (2-imino) + C-amidinium
    "[NH2+]=CNC1=COC(=O)[N-]1",      # oxazol-ide (2-oxo) + N-iminium
    "NC1=[NH+]C(=N)N=C(O)[N-]1",     # triazine: BOTH centres skeletal (-ium-...-ide)
    "NC=[NH+]C1=COC(=N)[N-]1",       # oxazol-ide (2-imino) + amidinium
]


def _inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return inchi.MolToInchiKey(mol) if mol is not None else None


def _best_effort(smiles):
    return name_compound(
        smiles,
        general_fallback=True,
        general_fallback_unverified=True,
        allow_aromatic_general=True,
    )


@pytest.mark.opsin_gate
@pytest.mark.parametrize("smiles", WITNESSES)
def test_charged_aromatic_zwitterion_round_trips(smiles):
    """The best-effort tier must EMIT a name that OPSIN reparses to the input's
    full standard InChIKey -- breadth with 0-wrong.

    Requires ``opsin_gate``: the best-effort tier's 0-wrong guarantee IS the
    full-round-trip validity gate, which the unit-test harness disables by
    default. With the gate off, an un-vetoed charge-dropped candidate from a
    different producer can out-rank this one; the gate (production default when
    the OPSIN jar is present) demotes it and selects the round-tripping name."""
    name = _best_effort(smiles)
    assert name and "unknown" not in name, f"abstained on {smiles}: {name!r}"
    reparsed = _n2s(name)
    assert reparsed, f"OPSIN could not parse emitted name {name!r}"
    assert _inchikey(reparsed) == _inchikey(smiles), (
        f"{smiles}: emitted {name!r} -> {reparsed!r} did not round-trip"
    )


@pytest.mark.parametrize("smiles", WITNESSES)
def test_pin_default_is_unaffected(smiles):
    """The charge distribution is behind ``allow_aromatic_general`` (best-effort
    only). The PIN/default tier still declines this out-of-PIN-scope class rather
    than emitting a general-engine name -- byte-identical to before the fix."""
    assert name_compound(smiles) == "unknown organic compound"
