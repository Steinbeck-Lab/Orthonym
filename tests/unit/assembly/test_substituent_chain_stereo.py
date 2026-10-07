""" Engine 4: a CARRIER-chain stereocentre inside a ring-bearing substituent
must emit its descriptor at the locant the substituent's own name cites.

The defect: ``_add_substituent_stereo`` could only obtain a substituent-local
locant map from ``_acyclic_alkyl_located_stereo_name``, which DECLINES on any
fragment containing ring atoms. So for a ring-on-chain substituent — a carrier
chain bridging a ring system to the parent, named by
``ring_substituents._compound_ring_on_chain_substituent`` — every carrier-borne
stereocentre was silently DROPPED (the multi-centre branch's "missing beats
wrong" fallback), shipping a stereo-incomplete, i.e. WRONG-diastereomer, prefix.

The fix threads the producer's OWN carrier numbering (``pos_out``, free valence
= 1 per through ``name_ring_system_substituent`` into the stereo
emitter's ``located`` channel, so the descriptor is cited at the locant the name
really used — never a re-derived BFS-from-attachment guess (the hazard).

Rule (VERIFIED, ``the Blue Book Blue Book``, section heading
``## **** NAMING OF STEREOISOMERS``): "When they relate to substituent
groups, they are cited at the front of the corresponding prefix. They are
preceded by a numerical or letter locant to describe the position of the
stereogenic unit *when such locants are present*." Its worked ``(PIN)``
examples pin BOTH directions:

- ``[(1R)-1-chloropropyl]benzene`` (PIN) — the prefix carries a locant, so the
  descriptor is LOCANTED;
- ``(R)-bromo(chloro)fluoromethane`` (PIN) — no locant in the name, so the
  descriptor is UNLOCANTED.

Hence the mononuclear-carrier branches (benzyl, '{ring-yl}methyl', the
alpha-halogen-decorated carrier) publish NO map and keep the bare "(R)-" form.
"""

import glob
import shutil

import pytest
from rdkit import Chem

from orthonym import namer as _namer_mod
from orthonym.perception.stereo import assign_stereochemistry
from orthonym.rules.ring_substituents import (
    _compound_ring_on_chain_substituent,
    name_ring_system_substituent,
)
from tests.support.jars import jar_or_none


pytestmark = pytest.mark.unit


# The witness: CHEBI:2364. A pentacyclic triterpenoid glycoside whose C15
# substituent is a -CH(CH3)- carrier bridging an alpha,beta-unsaturated lactone
# ring to the parent cage. The carrier C1 is a defined S centre; the ring C2 is
# a defined S centre already expressed inside the nested ring-yl prefix.
CHEBI_2364 = (
    "CC1=CC[C@@H]([C@@H](C)[C@H]2CC[C@@]3(C)[C@@H]4CC[C@H]5[C@](C)"
    "(C(=O)O)[C@@H](O[C@@H]6O[C@H](CO)[C@@H](O)[C@H](O)[C@H]6O)CC"
    "[C@@]56C[C@@]46CC[C@]23C)OC1=O"
)


def _find_opsin_jar():
    """The pinned OPSIN jar via orthonym.jars (tests.support.jars), or None."""
    return jar_or_none()


@pytest.fixture
def production_gate(opsin_gate):
    """Re-enable the production OPSIN validity gate; the suite's autouse
    fixture disables it. Without the gate this witness ships an EARLIER,
    wrong-molecule candidate ('6-hexyloxy...15-octyl...') that exists to
    suppress, so the tier under test is never reached.

    It uses the conftest ``opsin_gate`` fixture (the supported form, which also
    skips -- or fails under ORTHONYM_REQUIRE_JARS=1 -- when the jar is absent)
    instead of a hand-rolled setattr, which the ratchet in
    tests/unit/test_opsin_gate_test_harness.py refuses in new files (Task 12 fix
    a performance pass, wp6-tests; TRIAGE.md ' outcome')."""
    if not shutil.which("java") or _find_opsin_jar() is None:
        pytest.skip("OPSIN/Java not available for production-gate semantics")
    assert _namer_mod._DISABLE_VALIDITY_GATE is False


def _mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"invalid SMILES {smiles!r}"
    assign_stereochemistry(mol)
    return mol


def _frag(mol, attach_idx, blocked_idx):
    """The substituent subtree rooted at ``attach_idx``, not crossing
    ``blocked_idx`` (the parent-side atom)."""
    seen, queue = {attach_idx}, [attach_idx]
    while queue:
        i = queue.pop()
        for n in mol.GetAtomWithIdx(i).GetNeighbors():
            j = n.GetIdx()
            if j == blocked_idx or j in seen:
                continue
            seen.add(j)
            queue.append(j)
    return seen


class TestProducerPublishesItsOwnCarrierNumbering:
    """``pos_out`` must equal the numbering the emitted name cites, and must
    stay EMPTY wherever the name cites no carrier locant."""

    def test_two_carbon_carrier_publishes_map(self):
        # C[C@@H] hung off an amide N: carrier = C1(stereogenic) + C2.
        mol = _mol("CC(=O)N[C@@H](C)c1ccccc1")
        n_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
        attach = next(n.GetIdx() for n in mol.GetAtomWithIdx(n_idx).GetNeighbors()
                      if n.GetSymbol() == "C" and not n.GetIsAromatic()
                      and all(b.GetBondTypeAsDouble() == 1.0 for b in n.GetBonds()))
        frag = _frag(mol, attach, n_idx)
        ring_info = mol.GetRingInfo()
        ring_atoms = {i for i in frag if ring_info.NumAtomRings(i) > 0}
        pos = {}
        name = _compound_ring_on_chain_substituent(
            mol, sorted(frag), frag, ring_atoms, attach, ring_info,
            pos_out=pos,
        )
        assert name == "1-phenylethyl", name
        # Free valence is locant 1; the map covers the whole carrier.
        assert pos.get(attach) == 1, pos
        assert sorted(pos.values()) == [1, 2], pos

    def test_mononuclear_carrier_publishes_no_map(self):
        # A single -CH2- carrier cites no locant ('benzyl'), so the
        # unlocanted form must stay reachable -> no map published.
        mol = _mol("CC(=O)NCc1ccccc1")
        n_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
        attach = next(n.GetIdx() for n in mol.GetAtomWithIdx(n_idx).GetNeighbors()
                      if n.GetSymbol() == "C" and not n.GetIsAromatic()
                      and n.GetTotalNumHs() == 2)
        frag = _frag(mol, attach, n_idx)
        ring_info = mol.GetRingInfo()
        ring_atoms = {i for i in frag if ring_info.NumAtomRings(i) > 0}
        pos = {}
        name = _compound_ring_on_chain_substituent(
            mol, sorted(frag), frag, ring_atoms, attach, ring_info,
            pos_out=pos,
        )
        assert name == "benzyl", name
        assert pos == {}, pos

    def test_chokepoint_threads_the_map(self):
        """``name_ring_system_substituent`` must forward ``pos_out``, otherwise
        the emitter never sees it (the actual break in the shipped chain)."""
        mol = _mol("CC(=O)N[C@@H](C)C1CCCCC1")
        n_idx = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N")
        attach = next(n.GetIdx() for n in mol.GetAtomWithIdx(n_idx).GetNeighbors()
                      if n.GetSymbol() == "C" and n.HasProp("_CIPCode"))
        frag = _frag(mol, attach, n_idx)
        pos = {}
        name = name_ring_system_substituent(
            mol, sorted(frag), attach,
            allow_enumerator_fallback=False, pos_out=pos,
        )
        assert name == "1-cyclohexylethyl", name
        assert pos.get(attach) == 1, pos


class TestEmittedNameCarriesTheDescriptor:
    """End-to-end: the carrier stereocentre reaches the emitted name."""

    def test_minimal_ring_on_chain_carrier_stereo(self):
        from orthonym.namer import Orthonym
        namer = Orthonym(style="pin")
        name = namer.name("CC(=O)N[C@@H](C)C1CCCCC1")
        #: locant present in the prefix -> the descriptor is locanted.
        assert name == "N-[(1S)-1-cyclohexylethyl]acetamide", name

    def test_mononuclear_carrier_stays_unlocanted(self):
        """ negative control — a prefix with NO locant must not acquire
        one (the '(R)-bromo(chloro)fluoromethane (PIN)' direction)."""
        from orthonym.assembly.substituent_naming import _add_substituent_stereo
        mol = _mol("CC(=O)NCc1ccccc1")
        sub = [a.GetIdx() for a in mol.GetAtoms()]
        # An empty map (what the mononuclear branches publish) must never
        # produce a locanted block; it degrades to the bare form.
        out = _add_substituent_stereo(
            mol, sub, "benzyl", attach_idx=0, located=("benzyl", 1, {}))
        assert out == "benzyl", out


class TestChebi2364RoundTrips:
    """The witness molecule: full-InChIKey round trip through OPSIN."""

    def test_chebi_2364_best_effort_full_inchikey_rt(self, production_gate):
        from orthonym.namer import Orthonym
        from orthonym.jvm_bridge import opsin_stdout, opsin_available

        if not opsin_available():
            pytest.skip("OPSIN jar unavailable")

        namer = Orthonym(
            style="pin", general_fallback=True,
            general_fallback_unverified=True, allow_aromatic_general=True,
        )
        name = namer.name_tiered(CHEBI_2364)["name"]
        assert name, "CHEBI:2364 must emit a best-effort name"
        # The carrier descriptor is the whole point of the fix. The ring takes its
        # Hantzsch-Widman / retained name, the Blue Book; '2H-pyran (PIN)',
        #:2164; '3,4-dihydro-2H-pyran-3-yl (preferred prefix)',:17317).
        assert "{(1S)-1-[(2S)-5-methyl-6-oxo-3,6-dihydro-2H-pyran-2-yl]ethyl}" \
            in name, name

        raw = opsin_stdout(name, False)
        smi = raw[0] if isinstance(raw, tuple) else raw
        assert smi and smi.strip(), f"OPSIN could not parse {name!r}"
        back = Chem.MolFromSmiles(smi.strip())
        assert back is not None, smi
        assert Chem.MolToInchiKey(back) == \
            Chem.MolToInchiKey(Chem.MolFromSmiles(CHEBI_2364)), name
