# tests/unit/test_v26_p0_complete_flag.py
""": `--emit-tier complete` flag + `allow_aromatic_general` threading.

Pure plumbing — P0 lands NO behavior change. These tests assert:
  (a) the CLI parser accepts `complete` as a new --emit-tier choice;
  (b) IUPACNamer(allow_aromatic_general=True) constructs and stores the flag;
  (c) analyze_cage_universal / name_general / name_general_ring accept the
      new keyword without error (and without changing existing behavior);
  (d) a simple molecule under `complete` still names correctly (falls
      through to the existing PIN path unaffected).

Test strings assert EXACT output (/OPSIN is flaky under CPU
contention per project convention; do not rely on it here).
"""
import pytest
from rdkit import Chem

from orthonym.cli import main as cli_main
from orthonym.namer import Orthonym
from orthonym.assembly.general_engine import (
    name_general, name_general_ring, name_general_chain,
)
from orthonym.rules.vonbaeyer_universal import analyze_cage_universal

pytestmark = pytest.mark.unit


class TestCLIParserAcceptsComplete:
    def test_help_lists_complete_choice(self, capsys):
        with pytest.raises(SystemExit) as exc:
            cli_main(["--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "complete" in out
        # argparse renders the --emit-tier choices as a brace list with NO
        # spaces; assert the exact rendering so `complete` is present as a real
        # tier choice (not merely a substring of some other help text) and the
        # full ladder is offered. The fifth choice, full-coverage, is the
        # --emit-tier choice that c8cd7fd0d added (cli.py, ``choices=``).
        assert "{pin,valid,complete,best-effort,full-coverage}" in out

    def test_emit_tier_complete_runs_benzene(self, capsys):
        rc = cli_main(["c1ccccc1", "--emit-tier", "complete"])
        assert rc == 0
        out = capsys.readouterr().out.strip()
        assert out == "benzene"

    def test_emit_tier_rejects_unknown_choice(self):
        with pytest.raises(SystemExit):
            cli_main(["c1ccccc1", "--emit-tier", "bogus-tier"])


class TestIUPACNamerConstructsWithFlag:
    # NOTE: the brief refers to this class conceptually as "IUPACNamer";
    # the actual class in namer.py is `Orthonym` (no separate alias exists).
    def test_default_false(self):
        nm = Orthonym()
        assert nm._allow_aromatic_general is False

    def test_true_stored(self):
        nm = Orthonym(allow_aromatic_general=True)
        assert nm._allow_aromatic_general is True


class TestEngineSignaturesAcceptNewParam:
    def _features(self, smiles):
        nm = Orthonym(_disable_opsin_validity_gate=True)
        mol = Chem.MolFromSmiles(smiles)
        canonical = Chem.MolToSmiles(mol, canonical=True)
        feats = nm._perceive(mol, smiles, canonical)
        nm._classify(feats)
        return mol, feats

    def test_analyze_cage_universal_accepts_allow_mancude(self):
        mol = Chem.MolFromSmiles("C1CC2CCC1C2")  # norbornane
        cage_false = analyze_cage_universal(mol, allow_mancude=False)
        cage_true = analyze_cage_universal(mol, allow_mancude=True)
        assert cage_false is not None
        assert cage_true is not None
        # P0: inert -- identical result regardless of the flag's value.
        assert cage_false.descriptor == cage_true.descriptor == "bicyclo[2.2.1]"

    def test_analyze_cage_universal_mancude_default_refused_p2_lifts(self):
        # allow_mancude=False (the default / PIN path) STILL refuses an
        # aromatic cage -> byte-identity preserved. (commit ba2b8ae7)
        # lifted the refusal under allow_mancude=True: an aromatic cage is now
        # expressed as a kekulized von-Baeyer polyene, so naphthalene analyzes
        # as a bicyclo[4.4.0] cage. (Reconciled 2026-07-21: this superseded the
        # P0-era inertness assertion once P2 landed.)
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")  # naphthalene
        assert analyze_cage_universal(mol, allow_mancude=False) is None
        cage = analyze_cage_universal(mol, allow_mancude=True)
        assert cage is not None
        assert cage.descriptor.startswith("bicyclo[4.4.0]")

    def test_name_general_ring_accepts_param(self):
        mol, feats = self._features("C1CC2CCC1C2")  # norbornane
        res_false = name_general_ring(mol, feats, allow_aromatic_general=False)
        res_true = name_general_ring(mol, feats, allow_aromatic_general=True)
        assert res_false is not None
        assert res_true is not None
        assert res_false.name == res_true.name  # P0: inert

    def test_name_general_accepts_param(self):
        mol, feats = self._features("CCO")  # ethanol (chain path)
        res_false = name_general(mol, feats, allow_aromatic_general=False)
        res_true = name_general(mol, feats, allow_aromatic_general=True)
        assert res_false is not None
        assert res_true is not None
        assert res_false.name == res_true.name == "ethan-1-ol"

    def test_name_general_chain_unaffected(self):
        # name_general_chain does not take the new param (brief: not needed).
        mol, feats = self._features("CCO")
        res = name_general_chain(mol, feats)
        assert res is not None
        assert res.name == "ethan-1-ol"


class TestCompleteTierStillNamesSimpleMolecule:
    def test_ethanol_under_complete(self):
        nm = Orthonym(general_fallback=True, allow_aromatic_general=True)
        # PIN path (unaffected by the flag) wins; PIN omits the locant when
        # unambiguous on a 2-carbon chain -- unlike the general engine's
        # "ethan-1-ol" (see TestEngineSignaturesAcceptNewParam above).
        assert nm.name("CCO") == "ethanol"

    def test_benzene_under_complete(self):
        nm = Orthonym(general_fallback=True, allow_aromatic_general=True)
        assert nm.name("c1ccccc1") == "benzene"

    def test_acetic_acid_under_complete(self):
        nm = Orthonym(general_fallback=True, allow_aromatic_general=True)
        assert nm.name("CC(=O)O") == "acetic acid"
