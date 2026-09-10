"""a phase Plan-03: deny-list regression corpus (internal notes + a phase /).

One `class TestXxxNotEmitted` per deny entry. The correct invariant is "no pin=false
form leaks into emission VIA THE CONTROLLER" — so the assertion is DIFFERENTIAL (ON vs OFF): the
controller must never INTRODUCE a denied form. This is necessary because Orthonym's existing
retained-name machinery already emits a few of these (resorcinol/hydroquinone/catechol/cumene) in
OFF mode — those are pre-existing (a separate concern), NOT controller-introduced. The differential
assertion isolates the controller's contribution (which is 0-fire, so ON == OFF).

Source: 168-internal notes; RESEARCH section 6.5 + 6.7; analog test_retained_names_pin_cleanup.py.
"""

import glob
import shutil
import subprocess
from typing import Optional

import pytest

from orthonym import name_compound


def _find_opsin_jar():
    for pat in ("opsin-cli-*-jar-with-dependencies.jar",
                "opsin/opsin-cli-*-jar-with-dependencies.jar"):
        m = glob.glob(pat)
        if m:
            return m[0]
    return None


_OPSIN_JAR = _find_opsin_jar()
_OPSIN_AVAILABLE = bool(_OPSIN_JAR) and shutil.which("java") is not None


def _opsin_smiles(name: str) -> Optional[str]:
    if not _OPSIN_AVAILABLE or not name:
        return None
    try:
        r = subprocess.run(["java", "-jar", _OPSIN_JAR, "-osmi"], input=name + "\n",
                           capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    return r.stdout.strip() or None


def _assert_controller_does_not_introduce(smiles: str, archaic: str):
    """The controller must never INTRODUCE a denied form: if ON emits it, OFF must too (pre-existing)."""
    off = name_compound(smiles, enable_triviality_controller=False).lower()
    on = name_compound(smiles, enable_triviality_controller=True).lower()
    if archaic.lower() in on:
        assert archaic.lower() in off, (
            f"controller INTRODUCED denied form {archaic!r}: off={off!r} on={on!r}"
        )


# ---- a phase inheritance (3) ----
class TestErythreneNotEmitted:
    SMILES, ARCHAIC = "C=CC=C", "erythrene"
    @pytest.mark.integration
    def test_not_introduced(self):
        on = name_compound(self.SMILES, enable_triviality_controller=True)
        _assert_controller_does_not_introduce(self.SMILES, self.ARCHAIC)
        assert on  # sanity


class TestTrimethyleneGlycolNotEmitted:
    SMILES, ARCHAIC = "OCCCO", "trimethylene glycol"
    @pytest.mark.integration
    def test_not_introduced(self):
        _ = name_compound(self.SMILES, enable_triviality_controller=True)
        _assert_controller_does_not_introduce(self.SMILES, self.ARCHAIC)


class TestAspirinNotEmitted:
    SMILES, ARCHAIC = "CC(=O)Oc1ccccc1C(=O)O", "aspirin"
    @pytest.mark.integration
    def test_not_introduced(self):
        _ = name_compound(self.SMILES, enable_triviality_controller=True)
        _assert_controller_does_not_introduce(self.SMILES, self.ARCHAIC)


# ---- a phase inheritance (benzhydryl never introduced; diphenylmethoxy preserved) ----
class TestBenzhydrylNotEmitted:
    SMILES = "CN(C)CCOC(c1ccccc1)c1ccccc1"
    @pytest.mark.integration
    def test_benzhydryl_not_introduced(self):
        on = name_compound(self.SMILES, enable_triviality_controller=True).lower()
        assert "benzhydryl" not in on, f"benzhydryl introduced: {on!r}"

    @pytest.mark.integration
    def test_diphenylmethoxy_preserved(self):
        on = name_compound(self.SMILES, enable_triviality_controller=True).lower()
        assert "diphenylmethoxy" in on, f"diphenylmethoxy lost: {on!r}"


# ---- italic-letter locants (3): never emit o-/m-/p-xylene ----
class TestOXyleneNotEmitted:
    SMILES, ARCHAIC = "Cc1ccccc1C", "o-xylene"
    @pytest.mark.integration
    def test_not_introduced(self):
        on = name_compound(self.SMILES, enable_triviality_controller=True).lower()
        assert self.ARCHAIC not in on, f"italic locant leaked: {on!r}"


class TestMXyleneNotEmitted:
    SMILES, ARCHAIC = "Cc1cccc(C)c1", "m-xylene"
    @pytest.mark.integration
    def test_not_introduced(self):
        on = name_compound(self.SMILES, enable_triviality_controller=True).lower()
        assert self.ARCHAIC not in on, f"italic locant leaked: {on!r}"


class TestPXyleneNotEmitted:
    SMILES, ARCHAIC = "Cc1ccc(C)cc1", "p-xylene"
    @pytest.mark.integration
    def test_not_introduced(self):
        on = name_compound(self.SMILES, enable_triviality_controller=True).lower()
        assert self.ARCHAIC not in on, f"italic locant leaked: {on!r}"


# ---- Beilstein house-style (3) ----
class TestPhenylamineNotEmitted:
    SMILES, ARCHAIC = "Nc1ccccc1", "phenylamine"
    @pytest.mark.integration
    def test_not_introduced(self):
        _ = name_compound(self.SMILES, enable_triviality_controller=True)
        _assert_controller_does_not_introduce(self.SMILES, self.ARCHAIC)


class TestIsobutaneNotEmitted:
    SMILES, ARCHAIC = "CC(C)C", "isobutane"
    @pytest.mark.integration
    def test_not_introduced(self):
        _ = name_compound(self.SMILES, enable_triviality_controller=True)
        _assert_controller_does_not_introduce(self.SMILES, self.ARCHAIC)


class TestAcetoxyNotEmitted:
    SMILES, ARCHAIC = "CC(=O)OC", "acetoxy"
    @pytest.mark.integration
    def test_not_introduced(self):
        _ = name_compound(self.SMILES, enable_triviality_controller=True)
        _assert_controller_does_not_introduce(self.SMILES, self.ARCHAIC)


# ---- general-only (3) ----
class TestCumeneNotEmitted:
    SMILES, ARCHAIC = "CC(C)c1ccccc1", "cumene"
    @pytest.mark.integration
    def test_not_introduced(self):
        _ = name_compound(self.SMILES, enable_triviality_controller=True)
        _assert_controller_does_not_introduce(self.SMILES, self.ARCHAIC)


class TestCymeneNotEmitted:
    SMILES, ARCHAIC = "Cc1ccc(C(C)C)cc1", "cymene"
    @pytest.mark.integration
    def test_not_introduced(self):
        _ = name_compound(self.SMILES, enable_triviality_controller=True)
        _assert_controller_does_not_introduce(self.SMILES, self.ARCHAIC)


class TestMesityleneNotEmitted:
    SMILES, ARCHAIC = "Cc1cc(C)cc(C)c1", "mesitylene"
    @pytest.mark.integration
    def test_not_introduced(self):
        _ = name_compound(self.SMILES, enable_triviality_controller=True)
        _assert_controller_does_not_introduce(self.SMILES, self.ARCHAIC)


# ---- general-only retained names (5) ----
class TestResorcinolNotEmitted:
    SMILES, ARCHAIC = "Oc1cccc(O)c1", "resorcinol"
    @pytest.mark.integration
    def test_not_introduced(self):
        _ = name_compound(self.SMILES, enable_triviality_controller=True)
        _assert_controller_does_not_introduce(self.SMILES, self.ARCHAIC)


class TestHydroquinoneNotEmitted:
    SMILES, ARCHAIC = "Oc1ccc(O)cc1", "hydroquinone"
    @pytest.mark.integration
    def test_not_introduced(self):
        _ = name_compound(self.SMILES, enable_triviality_controller=True)
        _assert_controller_does_not_introduce(self.SMILES, self.ARCHAIC)


class TestCatecholNotEmitted:
    SMILES, ARCHAIC = "Oc1ccccc1O", "catechol"
    @pytest.mark.integration
    def test_not_introduced(self):
        _ = name_compound(self.SMILES, enable_triviality_controller=True)
        _assert_controller_does_not_introduce(self.SMILES, self.ARCHAIC)


class TestEthyleneGlycolNotEmitted:
    SMILES, ARCHAIC = "OCCO", "ethylene glycol"
    @pytest.mark.integration
    def test_not_introduced(self):
        _ = name_compound(self.SMILES, enable_triviality_controller=True)
        _assert_controller_does_not_introduce(self.SMILES, self.ARCHAIC)


class TestSalicylicAcidNotEmitted:
    SMILES, ARCHAIC = "OC(=O)c1ccccc1O", "salicylic acid"
    @pytest.mark.integration
    def test_not_introduced(self):
        _ = name_compound(self.SMILES, enable_triviality_controller=True)
        _assert_controller_does_not_introduce(self.SMILES, self.ARCHAIC)


# ---- general-only acids (3) ----
class TestPropionicAcidNotEmitted:
    SMILES, ARCHAIC = "CCC(=O)O", "propionic acid"
    @pytest.mark.integration
    def test_not_introduced(self):
        _ = name_compound(self.SMILES, enable_triviality_controller=True)
        _assert_controller_does_not_introduce(self.SMILES, self.ARCHAIC)


class TestButyricAcidNotEmitted:
    SMILES, ARCHAIC = "CCCC(=O)O", "butyric acid"
    @pytest.mark.integration
    def test_not_introduced(self):
        _ = name_compound(self.SMILES, enable_triviality_controller=True)
        _assert_controller_does_not_introduce(self.SMILES, self.ARCHAIC)


class TestMalonicAcidNotEmitted:
    SMILES, ARCHAIC = "OC(=O)CC(=O)O", "malonic acid"
    @pytest.mark.integration
    def test_not_introduced(self):
        _ = name_compound(self.SMILES, enable_triviality_controller=True)
        _assert_controller_does_not_introduce(self.SMILES, self.ARCHAIC)
