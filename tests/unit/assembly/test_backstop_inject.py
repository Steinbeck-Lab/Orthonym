"""Wave-0 unit tests for the WSB-01 backstop flip (Phase 177 Plan 02).

The universal stereo backstop ``namer._final_stereo_check`` is converted from
detect-only to real injection on the ONLY cohort with an authoritative parent
``atom_to_locant`` map: ``chain`` and ``non-phenol benzene``. All other handlers
(``complex_ring`` / ``polycyclic`` / ``heterocycle`` / ``unknown``) and any
D/L-configured name stay LOG-ONLY (no descriptor added by the backstop).

These tests call ``_final_stereo_check`` directly against its POST-Task-3
target signature (gains an ``atom_to_locant`` keyword) — so they are RED until
Task 3 flips the backstop. Decisions: D-04 (threaded authoritative map),
D-05 (allowlist = chain + non-phenol benzene), D-06 (detect-only where no map),
D-09 (missing beats wrong).
"""

import pytest
from rdkit import Chem

from orthonym.namer import _final_stereo_check
from orthonym.perception.stereo import assign_stereochemistry
from orthonym.rules.locants import build_atom_to_locant


pytestmark = pytest.mark.unit


def _mol_with_cip(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"invalid SMILES {smiles!r}"
    assign_stereochemistry(mol)
    return mol


class TestBackstopInjectsChain:
    """Chain handler with a threaded authoritative atom_to_locant -> inject."""

    def test_chain_with_threaded_map_injects(self):
        # butan-2-ol: SMILES "CC[C@H](C)O" -> atoms
        #   0=CH3(ethyl end), 1=CH2, 2=C(@H) stereocenter, 3=CH3, 4=O.
        # The oriented IUPAC chain places the OH-bearing carbon at locant 2:
        #   locant 1 = atom 0, 2 = atom 1, 3 = atom 2, 4 = atom 3  (butane chain)
        # but PIN numbers from the end giving the OH the lowest locant -> the
        # stereocenter (atom 2) is locant 2.  Build that authoritative map.
        mol = _mol_with_cip("CC[C@H](C)O")
        # Oriented principal chain so atom 2 (stereocenter, OH-bearing) = locant 2.
        oriented_chain = [3, 2, 1, 0]  # CH3 - C(@) - CH2 - CH3  -> locants 1,2,3,4
        atom_to_locant = build_atom_to_locant(oriented_chain)
        assert atom_to_locant[2] == 2  # stereocenter is locant 2

        result = _final_stereo_check(
            mol, "butan-2-ol", handler="chain", atom_to_locant=atom_to_locant
        )
        # A descriptor block must have been injected at the front.
        assert result != "butan-2-ol"
        assert result.startswith("(2") and ")-butan-2-ol" in result, result


class TestBackstopLogOnly:
    """Non-injectable handlers / D-L names stay log-only (name unchanged)."""

    @pytest.mark.parametrize("handler", ["complex_ring", "polycyclic",
                                         "heterocycle", "unknown"])
    def test_non_allowlisted_handler_log_only(self, handler):
        mol = _mol_with_cip("CC[C@H](C)O")
        # Even with a map threaded, a non-allowlisted handler must NOT inject
        # (D-05/D-06: no authoritative map for these classes -> log-only).
        name = "some-complex-ring-name"
        result = _final_stereo_check(
            mol, name, handler=handler, atom_to_locant={2: 2}
        )
        assert result == name

    def test_chain_without_map_log_only(self):
        # D-06: backstop stays detect-only where NO proven map exists.
        mol = _mol_with_cip("CC[C@H](C)O")
        result = _final_stereo_check(mol, "butan-2-ol", handler="chain",
                                     atom_to_locant=None)
        assert result == "butan-2-ol"

    def test_dl_named_input_suppressed(self):
        # A D/L-configured name is stereo-already-present (Plan 01 Pattern D)
        # -> the backstop must NOT add a (nR)/(nS) block even on a chain handler.
        mol = _mol_with_cip("C[C@H](N)C(=O)O")  # alanine skeleton
        atom_to_locant = {1: 2}
        result = _final_stereo_check(
            mol, "L-alanine", handler="chain", atom_to_locant=atom_to_locant
        )
        assert result == "L-alanine"
