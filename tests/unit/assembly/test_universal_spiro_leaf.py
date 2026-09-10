"""The universal substitutive core must name spiro ring systems as a branch
parent, not void on them.

Root cause (trace 2026-08-30): ``universal_substituent._name_ring_spine`` wired
only ``analyze_cage_universal`` (von-Baeyer) for polycyclic ring parents, so a
spiro ring -- which is NOT a von-Baeyer cage -- made the cage analyzer void and
the "unconditional" core ``return None``, voiding on EVERY spiro-ring branch.
The fix falls through to ``analyze_spiro_universal`` (which already produces the
spiro descriptor / numbering for ~74% of the spiro abstention residual). This is
the best-effort FLOOR (reachable only on the general-fallback path); the PIN
path is untouched.

Governing rules: (von-Baeyer/spiro ring numbering), (a free
valence takes the lowest locant for substituent use). 0-wrong is enforced by the
offer pool's full-InChIKey RT gate; these tests pin PRODUCTION (the core no
longer voids) and the whole-molecule round-trip.
"""
import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from orthonym.metrics.provenance import best_effort_ctx
from orthonym.assembly.universal_substituent import (
    name_universal_substitutive,
    name_universal_substituent_prefix,
)
from orthonym.namer import Orthonym


def _mol(smi):
    return Chem.MolFromSmiles(smi)


class TestUniversalCoreNamesSpiro:
    def test_carbospiro_ring_names(self):
        with _be():
            r = name_universal_substitutive(_mol("C1CC12CCC2"))
        assert r is not None and r.name == "spiro[2.3]hexane"

    def test_azaspiro_ring_names(self):
        with _be():
            r = name_universal_substitutive(_mol("C1CC12CCNCC2"))
        assert r is not None and r.name == "6-azaspiro[2.5]octane"

    def test_diazaspiro_ring_names(self):
        with _be():
            r = name_universal_substitutive(_mol("C1CC2(CCNCC2)CCN1"))
        assert r is not None and r.name == "3,9-diazaspiro[5.5]undecane"

    def test_spiro_branch_prefix(self):
        # spiro[2.3]hexane as a -yl substituent on a methyl (attach = ring atom 1)
        with _be():
            r = name_universal_substituent_prefix(_mol("CC1CC12CCC2"),
                                                  {1, 2, 3, 4, 5, 6}, 1, 1)
        assert r == "spiro[2.3]hexan-1-yl"

    def test_bicyclo_still_names(self):
        # regression: the pre-existing von-Baeyer path must be unchanged
        with _be():
            r = name_universal_substitutive(_mol("C1CC2CCC1CC2"))
        assert r is not None and r.name == "bicyclo[2.2.2]octane"


@pytest.mark.opsin_gate
class TestWholeMoleculeSpiroComposition:
    """The whole-molecule ZINC-class witnesses (spiro ring + amide backbone)
    now emit a best-effort name instead of abstaining.

    Runs with the OPSIN validity gate ON (``opsin_gate``) — the real-deployment
    configuration — so what ships is exactly what the 0-wrong RT gate lets
    through (an unparseable/malformed sibling candidate is rejected, not
    shipped). Without the marker the conftest autouse fixture leaves the gate
    OFF and an ungated malformed candidate could win."""

    @pytest.mark.parametrize("smi", [
        "CNC(=O)[C@H](C)NC(=O)[C@@H]1CC12CCC2",                 # di-amide + spiro
        "COCCOCCN(C)C(=O)[C@H](C)NC(=O)[C@@H]1CC12CCC2",        # ZINC#16
        "Cc1cnc(C(=O)N2CCC3(CC2)C[C@@H]3NC(=O)CC[C@H]2CC[C@H](C)O2)cc1Cl",  # ZINC#3
    ])
    def test_emits(self, smi):
        with _be():
            n = Orthonym(general_fallback=True, general_fallback_unverified=True,
                          allow_aromatic_general=True).name(smi)
        assert n and not n.startswith("unknown") and "not supported" not in n

    def test_deterministic(self):
        smi = "CNC(=O)[C@H](C)NC(=O)[C@@H]1CC12CCC2"
        outs = []
        for _ in range(3):
            with _be():
                outs.append(Orthonym(general_fallback=True,
                                      general_fallback_unverified=True,
                                      allow_aromatic_general=True).name(smi))
        assert len(set(outs)) == 1


class TestPinPathUntouched:
    """The fix is gated to the best-effort floor; default (PIN) naming is
    byte-identical."""

    @pytest.mark.parametrize("smi,expected", [
        ("C1CC12CCC2", "spiro[2.3]hexane"),
        ("O=C(N)C1CC12CCC2", "spiro[2.3]hexane-1-carboxamide"),
        ("C1CC2(CCNCC2)CCN1", "3,9-diazaspiro[5.5]undecane"),
        ("CCO", "ethanol"),
    ])
    def test_pin_unchanged(self, smi, expected):
        from orthonym import name_compound
        assert name_compound(smi) == expected


@pytest.mark.roundtrip
@pytest.mark.opsin_gate
class TestSpiroCompositionRoundTrips:
    """0-wrong: the emitted best-effort name round-trips to the input
    constitution through OPSIN (requires a JVM). Gate ON (``opsin_gate``) =
    real deployment: the RT gate rejects any unparseable sibling candidate, so
    what ships round-trips. Uses the maintained ``opsin_to_smiles`` fixture
    (in-process JPype) — never a hand-rolled subprocess."""

    @pytest.mark.parametrize("smi", [
        "CNC(=O)[C@H](C)NC(=O)[C@@H]1CC12CCC2",
        "COCCOCCN(C)C(=O)[C@H](C)NC(=O)[C@@H]1CC12CCC2",
    ])
    def test_roundtrips_constitution(self, smi, opsin_to_smiles):
        with _be():
            name = Orthonym(general_fallback=True, general_fallback_unverified=True,
                             allow_aromatic_general=True).name(smi)
        assert name and not name.startswith("unknown")
        out = opsin_to_smiles(name)
        assert out, f"OPSIN could not parse {name!r}"
        a, b = Chem.MolFromSmiles(smi), Chem.MolFromSmiles(out)
        assert a is not None and b is not None
        # skeleton InChIKey = constitution (stereo-insensitive) — 0-wrong test
        assert inchi.MolToInchiKey(a).split("-")[0] == inchi.MolToInchiKey(b).split("-")[0]


class _be:
    """best_effort_ctx context manager."""
    def __enter__(self):
        self._tok = best_effort_ctx.set("test")
        return self

    def __exit__(self, *exc):
        best_effort_ctx.reset(self._tok)
        return False
