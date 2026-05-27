"""Phase 167 (HYG-04) — diaryl-methyl-ether + diaryl-ether-amine class tests.

Two independent root-cause bugs from the HERITAGE head-to-head (§5.1B):

  SITE #1 (Plan 167-03 turns these green): Ar2CH-O- (diphenylmethyl ether) is
  mis-named `benzyloxy` because three byte-duplicated code sites never count the
  central carbon's aryl neighbours. PIN is `diphenylmethoxy` (D-07; NOT the
  Beilstein `benzhydryloxy`). Fixed via ONE shared aryl-count helper.

  SITE #2 (Plan 167-04 turns these green): `_name_amino_branch` only handles a
  single N-substituent, so N,N-dialkylamino substituents on a chain parent mis-split.

  PITFALL-3 GUARD (must stay green across BOTH plans): the principal-amine path
  (`CN(C)CC` -> N,N-dimethylethan-1-amine) is already correct and untouched.

RT assertions use the conftest `opsin_to_smiles` fixture (auto-skips if OPSIN/Java
absent) and compare InChI of the OPSIN-parsed emitted name to InChI of the input.
Forbidden: xfail, postprocessors. Targets OPSIN-verified (opsin-cli 2.9.0).
"""

import glob
import shutil
import subprocess

import pytest

from orthonym import name_compound


def _find_opsin_jar():
    for pat in (
        "opsin-cli-*-jar-with-dependencies.jar",
        "opsin.jar",
        "opsin/opsin-cli-*-jar-with-dependencies.jar",
        "opsin/opsin-cli/target/opsin-cli-*-jar-with-dependencies.jar",
    ):
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return None


_OPSIN_JAR = _find_opsin_jar()
_OPSIN_AVAILABLE = bool(_OPSIN_JAR) and shutil.which("java") is not None


def _opsin_smiles(name):
    """Name -> SMILES via OPSIN over STDIN.

    OPSIN 2.9.0's CLI treats the post-`-osmi` argv as an input *file*; the name
    must be fed on stdin (the conftest `opsin_to_smiles` fixture uses argv and
    therefore returns None for every name on this OPSIN version). stdout carries
    only the SMILES (empty on parse failure); the prompt banner goes to stderr.
    """
    if not _OPSIN_AVAILABLE:
        return None
    try:
        result = subprocess.run(
            ["java", "-jar", _OPSIN_JAR, "-osmi"],
            input=name + "\n",
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return None
    out = result.stdout.strip()
    return out or None


def _inchi(smiles):
    """InChI for a SMILES (None if unparseable)."""
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchi(mol) if mol else None


def _assert_roundtrips(smiles_input):
    """The name Orthonym emits for `smiles_input` must round-trip via OPSIN."""
    if not _OPSIN_AVAILABLE:
        pytest.skip("OPSIN/Java not available")
    emitted = name_compound(smiles_input)
    parsed = _opsin_smiles(emitted)
    assert parsed is not None, f"OPSIN could not parse emitted name {emitted!r}"
    assert _inchi(parsed) == _inchi(smiles_input), (
        f"round-trip mismatch for {smiles_input!r}: emitted {emitted!r} -> {parsed!r}"
    )


class TestDiarylMethoxy:
    """SITE #1 — Ar2CH-O- on a parent must read `diphenylmethoxy`, not `benzyloxy`.

    These probes route through the substituent-naming path (the 3 duplicated
    sites) — the isolated `Ph2CH-O-Me` whole-molecule case routes elsewhere
    (A2/Pitfall 2), so the load-bearing assertions use substituent context.
    Plan 167-03 makes these green.
    """

    # >=3 diphenylmethoxy analogs (D-08) spanning chain length.
    DIPHENYLMETHOXY = [
        "OCCOC(c1ccccc1)c1ccccc1",    # 2-(diphenylmethoxy)ethan-1-ol
        "OCCCOC(c1ccccc1)c1ccccc1",   # 3-(diphenylmethoxy)propan-1-ol
        "OCCCCOC(c1ccccc1)c1ccccc1",  # 4-(diphenylmethoxy)butan-1-ol
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("smiles", DIPHENYLMETHOXY)
    def test_emits_diphenylmethoxy(self, smiles):
        name = name_compound(smiles)
        assert "diphenylmethoxy" in name, f"expected diphenylmethoxy, got {name!r}"
        assert "benzyloxy" not in name, f"still benzyloxy (bug): {name!r}"
        assert "benzhydryl" not in name, f"emitted Beilstein benzhydryl (D-07 violation): {name!r}"

    @pytest.mark.unit
    @pytest.mark.roundtrip
    @pytest.mark.parametrize("smiles", DIPHENYLMETHOXY)
    def test_diphenylmethoxy_roundtrips(self, smiles):
        _assert_roundtrips(smiles)

    @pytest.mark.unit
    def test_negative_control_benzyl_unchanged(self):
        """True PhCH2-O- (1 aryl) must STILL be benzyloxy — the fix discriminates."""
        name = name_compound("OCCOCc1ccccc1")
        assert "benzyloxy" in name, f"benzyl ether regressed: {name!r}"
        assert "diphenylmethoxy" not in name, f"blanket-renamed benzyl: {name!r}"

    @pytest.mark.unit
    @pytest.mark.roundtrip
    def test_negative_control_roundtrips(self):
        _assert_roundtrips("OCCOCc1ccccc1")


class TestDialkylaminoChain:
    """SITE #2 — N,N-dialkylamino substituent on a chain parent (Plan 167-04 turns green).

    RED in Plan 167-03 (the amine branch only handles one N-substituent today).
    """

    @pytest.mark.unit
    def test_dimethylamino_ethanol(self):
        name = name_compound("CN(C)CCO")
        assert name == "2-(dimethylamino)ethan-1-ol", f"got {name!r}"

    @pytest.mark.unit
    @pytest.mark.roundtrip
    def test_dimethylamino_ethanol_roundtrips(self):
        _assert_roundtrips("CN(C)CCO")

    @pytest.mark.unit
    def test_diethylamino_ethanol(self):
        name = name_compound("CCN(CC)CCO")
        assert "diethylamino" in name, f"got {name!r}"

    @pytest.mark.unit
    @pytest.mark.roundtrip
    def test_diethylamino_ethanol_roundtrips(self):
        _assert_roundtrips("CCN(CC)CCO")

    @pytest.mark.unit
    def test_diphenhydramine_core_combined(self):
        """Combined site#1 + site#2 target (diphenhydramine core)."""
        name = name_compound("CN(C)CCOC(c1ccccc1)c1ccccc1")
        assert name == "2-(diphenylmethoxy)-N,N-dimethylethan-1-amine", f"got {name!r}"

    @pytest.mark.unit
    @pytest.mark.roundtrip
    def test_diphenhydramine_core_roundtrips(self):
        _assert_roundtrips("CN(C)CCOC(c1ccccc1)c1ccccc1")


class TestPrincipalAmineGuard:
    """PITFALL-3 — the principal-amine path is already correct; must stay byte-identical.

    Green now AND after both plans (regression guard — do NOT touch _assemble_amine_name).
    """

    @pytest.mark.unit
    def test_principal_dimethylamine_unchanged(self):
        assert name_compound("CN(C)CC") == "N,N-dimethylethan-1-amine"
