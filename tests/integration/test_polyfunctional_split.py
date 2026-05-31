"""Phase 169 Plan-03 — Group-splitting integration + OPSIN-RT corpus (POLY-01/02).

Demonstrates the split mechanism end-to-end with the flag forced ON per-test
(production default stays OFF — Stage A byte-identical preserved):

* POLY-01: a composite loser FG is SPLIT (its sub-group prefixes appear in the
  name) instead of dropped — with the thioester as the hard OPSIN-RT anchor.
* POLY-02: the split components share the central-carbon locant and are
  alphabetized in place.
* D-03 negative contract: genuine functional-class FGs are NOT force-split.

HONEST NOTE (RESEARCH Pitfall 4 — confirmed in Plan-02): the analyst's ester
anchor ``OC(=O)CCC(=O)OCC`` (monoethyl succinate) is NOT a DROP-23 case — it is
named ``4-ethoxycarbonylbutanoic acid`` via ``get_alkoxycarbonyl_prefix`` (a
moving-base-atom / Phase-172 concern, not a group-split). So the ester whole-
molecule split does not fire there; the ester claim is asserted at the
decomposition + fragment-RT level (the split FORM round-trips), and the
thioester carries the hard whole-molecule RT anchor.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from rdkit import Chem, RDLogger

from orthonym import name_compound
from orthonym.assembly.group_splitting import split_composite_fg

RDLogger.logger().setLevel(RDLogger.ERROR)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ESTER_SMARTS = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")


def _find_jar():
    cand = _PROJECT_ROOT / "opsin-cli-2.9.0-jar-with-dependencies.jar"
    if cand.exists():
        return str(cand)
    try:
        sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
        from validate_retained_names import find_opsin_jar
        return find_opsin_jar() or None
    except Exception:
        return None


_JAR = _find_jar()
_opsin_rt = pytest.mark.skipif(_JAR is None, reason="OPSIN jar not available")


def _conn(smi):
    """Connectivity-only InChI (stereo + isotope stripped) for an RT compare."""
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if not m:
        return None
    ic = Chem.MolToInchi(m, options="-SNon")
    if not ic:
        return None
    parts = ic.split("/")
    return "/".join([parts[0]] + [x for x in parts[1:] if x and x[0] in "Cc"])


def opsin_roundtrips(name, ref_smiles):
    """True iff OPSIN parses `name` to a structure connectivity-matching `ref_smiles`."""
    if not name or _JAR is None:
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(name + "\n")
        path = f.name
    try:
        out = subprocess.run(["java", "-jar", _JAR, "-osmi", path],
                             capture_output=True, text=True, timeout=20)
        parsed = out.stdout.strip()
    except Exception:
        return False
    finally:
        os.unlink(path)
    ref, got = _conn(ref_smiles), _conn(parsed)
    return ref is not None and ref == got


# OPSIN-RT-verified targets (RESEARCH § Code Examples + Plan-02 demonstration).
THIOESTER_SMILES = "OC(=O)CCC(=O)SCC"   # -> 4-(ethylsulfanyl)-4-oxobutanoic acid (fires + RT)
ESTER_SMILES = "OC(=O)CCC(=O)OCC"       # decomposes to oxo+ethoxy; whole-mol via ethoxycarbonyl (Pitfall 4)


@pytest.mark.integration
class TestHeadline:
    """POLY-01: a composite loser FG is split (components present), not dropped."""

    def test_headline_loser_split_not_dropped(self):
        # The thioester loses to the -COOH (principal); instead of dropping the
        # thioester (today's DROP-23 -> "4-(ethylsulfanyl)butanoic acid", =O lost),
        # the split recovers the dropped chalcogen as oxo. Both sub-group prefixes
        # of the previously-vanished group now appear.
        name = name_compound(THIOESTER_SMILES, enable_group_splitting=True)
        assert name is not None
        low = name.lower()
        assert "oxo" in low and "sulfanyl" in low, f"loser not split: {name}"


@pytest.mark.integration
class TestEsterLoser:
    """POLY-01 (ester): -C(=O)-O-R decomposes to oxo + R-oxy (alkoxy).

    Asserted at the decomposition + fragment-RT level (Pitfall 4: the monoethyl-
    succinate whole-molecule path goes through ethoxycarbonyl, not DROP-23)."""

    def test_ester_loser_decomposes_to_oxo_plus_alkoxy(self):
        mol = Chem.MolFromSmiles(ESTER_SMILES)
        match = mol.GetSubstructMatches(_ESTER_SMARTS)[0]
        comps = split_composite_fg("ester", mol, match, None)
        assert comps is not None and len(comps) == 2
        forms = {c.prefix_form for c in comps}
        assert "oxo" in forms and any(f.endswith("oxy") and f != "oxo" for f in forms), forms

    @_opsin_rt
    @pytest.mark.roundtrip
    def test_ester_split_form_roundtrips(self):
        # The split FORM round-trips (the oxo+alkoxy decomposition is RT-correct).
        assert opsin_roundtrips("4-ethoxy-4-oxobutanoic acid", ESTER_SMILES)


@pytest.mark.integration
class TestThioesterLoser:
    """POLY-01 (thioester): the hard whole-molecule OPSIN-RT anchor."""

    def test_thioester_loser_name_has_oxo_and_sulfanyl(self):
        name = name_compound(THIOESTER_SMILES, enable_group_splitting=True)
        low = name.lower()
        assert "oxo" in low and "ethylsulfanyl" in low, name

    @_opsin_rt
    @pytest.mark.roundtrip
    def test_thioester_loser_roundtrips(self):
        name = name_compound(THIOESTER_SMILES, enable_group_splitting=True)
        assert opsin_roundtrips(name, THIOESTER_SMILES), name


@pytest.mark.integration
class TestLocants:
    """POLY-02: both split components share the central-carbon locant (Pitfall 3)."""

    def test_split_components_share_central_carbon_locant(self):
        name = name_compound(THIOESTER_SMILES, enable_group_splitting=True)
        # "4-(ethylsulfanyl)-4-oxobutanoic acid": oxo and ethylsulfanyl both at C4.
        import re
        oxo_loc = re.search(r"(\d+)-oxo", name)
        sulf_loc = re.search(r"(\d+)-\(ethylsulfanyl\)", name)
        assert oxo_loc and sulf_loc, name
        assert oxo_loc.group(1) == sulf_loc.group(1), f"locants differ: {name}"


@pytest.mark.integration
class TestAlphabetization:
    """POLY-02: split components are alphabetized in place (P-14.5)."""

    def test_split_components_alphabetized(self):
        name = name_compound(THIOESTER_SMILES, enable_group_splitting=True)
        # ethylsulfanyl (e) must precede oxo (o) in the assembled name.
        assert name.lower().index("ethylsulfanyl") < name.lower().index("oxo"), name


@pytest.mark.integration
class TestFunctionalClassNotSplit:
    """D-03 negative contract: functional-class FGs are NOT force-split (stay dropped)."""

    @pytest.mark.parametrize("fg", ["secondary_amide", "phosphate_diester",
                                    "anhydride", "imide"])
    def test_functional_class_split_returns_none(self, fg):
        assert split_composite_fg(fg, None, None, None) is None

    def test_secondary_amide_compound_unchanged_flag_on(self):
        # A secondary-amide-bearing acid: flag-ON must NOT fabricate split prefixes
        # for the amide carbon (the amide is named via its own path, not split).
        smi = "CCNC(=O)CCC(=O)O"
        off = name_compound(smi)
        on = name_compound(smi, enable_group_splitting=True)
        assert off == on, f"functional-class FG was force-split: {off!r} -> {on!r}"
