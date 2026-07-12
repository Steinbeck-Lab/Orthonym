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


@pytest.mark.integration
class TestNarrowDefaultOnP6563:
    """W2F-P2 Task 3 (P-65.6.3.3.5 method (1), BB 31958-31962): acid-principal
    partial esters split by DEFAULT (no flag) — every split is per-candidate
    OPSIN-RT gated, so a wrong assembly can never be emitted."""

    @_opsin_rt
    @pytest.mark.roundtrip
    @pytest.mark.parametrize("smiles,expected", [
        ("COC(=O)CCC(=O)O", "4-methoxy-4-oxobutanoic acid"),
        ("O=C(OC)CCCCCCCC(=O)O", "9-methoxy-9-oxononanoic acid"),
        # curated target, brief items 5+6 (benzyloxy IS the preferred prefix,
        # P-29.6.2.1/P-35.3.2:18097; alphanumerical benzyloxy < oxo, P-14.5.2)
        ("O=C(OCc1ccccc1)CCCCCCCC(=O)O", "9-(benzyloxy)-9-oxononanoic acid"),
        # GS-ON preview verified RT OK at HEAD 2026-07-11 (research item 1 F.3)
        ("O=C(O)CCCC(=O)OCCCO", "5-(3-hydroxypropoxy)-5-oxopentanoic acid"),
    ])
    def test_acid_ester_split_default_on(self, smiles, expected):
        assert name_compound(smiles) == expected

    @_opsin_rt
    @pytest.mark.roundtrip
    def test_substituted_benzyl_boundary_fail_closed(self, monkeypatch):
        # P-29.6.2.1: substituted benzyl is out of v1. The split declines the
        # nested-bracket linker prefix '(4-hydroxyphenyl)methoxy' (it would need
        # '[...]' escalation the single-level emit path cannot render as a PIN),
        # so the ester is dropped and the only remaining candidate is a
        # wrong-structure ester name that the PRODUCTION OPSIN validity gate
        # (SELF-01) suppresses -> 'unknown'. The autouse test fixture disables
        # that gate for speed, so re-enable it here to exercise the real
        # production fail-closed path.
        import orthonym.namer as _namer
        monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
        out = name_compound("O=C(OCc1ccc(O)cc1)CCCCCCCC(=O)O")
        assert out == "unknown organic compound"

    @_opsin_rt
    @pytest.mark.roundtrip
    def test_acyloxy_orientation_untouched(self):
        # Gold P3A-P06: acyloxy orientation (ester O on chain, carbonyl off) ->
        # anchored filter empty -> byte-identical continue; composer acyloxy
        # path keeps ownership.
        assert name_compound("OC(=O)CCOC(=O)C") == "3-(acetyloxy)propanoic acid"


@pytest.mark.integration
class TestAcylSulfanylWholeMolecule:
    """W2F-P2 Task 4: P-35.5.1 whole-molecule anchors (all expected names
    OPSIN-2.9-verified 2026-07-11, research P-35.5.1 §D)."""

    @_opsin_rt
    @pytest.mark.roundtrip
    @pytest.mark.parametrize("smiles,expected", [
        # curated target, brief item 7 (acetylsulfanyl < oxo, P-14.5.2)
        ("CC(=O)SC(=O)CCCCCCCC(=O)O", "9-(acetylsulfanyl)-9-oxononanoic acid"),
        # thioether-prefix leg ALONE (thioester carbonyl off-chain, no split)
        ("CC(=O)SCCCCCCCC(=O)O", "8-(acetylsulfanyl)octanoic acid"),
        ("CCC(=O)SCCC(=O)O", "3-(propanoylsulfanyl)propanoic acid"),
        # genuine-alkyl split sibling (BB 31946 parallel); GS-ON preview OK at HEAD
        ("CCSC(=O)CCC(=O)O", "4-(ethylsulfanyl)-4-oxobutanoic acid"),
    ])
    def test_thioester_class(self, smiles, expected):
        assert name_compound(smiles) == expected

    @_opsin_rt
    @pytest.mark.roundtrip
    def test_benzoylsulfanyl_whole_molecule(self):
        # Conditional gold W2F-P2-09: the prefix builder is proven at unit
        # level; if the assembly routes this ring-bearing molecule elsewhere
        # and refuses, record the deviation and DROP the gold (fail-closed
        # beats forcing). Do NOT weaken the assert to make it pass.
        assert name_compound("O=C(c1ccccc1)SCCC(=O)O") == "3-(benzoylsulfanyl)propanoic acid"

    @_opsin_rt
    @pytest.mark.roundtrip
    @pytest.mark.parametrize("smiles", [
        "OCC(=O)SCCC(=O)O",   # substituted acyl -> v1 refuses (research §E Q4)
        "CC(=O)SC(=O)C",      # symmetric thioanhydride: class = anhydride
                              # (P-65.7.3), no free acid -> substitutive naming
                              # would be WRONG-class; thioanhydride namer not
                              # built -> must refuse
    ])
    def test_fail_closed_boundaries(self, smiles, monkeypatch):
        # These molecules fail-close via the PRODUCTION OPSIN validity gate /
        # per-split RT gate. The autouse test fixture disables the validity gate
        # for speed; re-enable it so the assertion sees the real production
        # refusal instead of a gate-off raw leak.
        import orthonym.namer as _namer
        monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
        assert name_compound(smiles) == "unknown organic compound"


@pytest.mark.integration
class TestW2FP2Determinism:
    """W2F-P2 Task 6: the default split + acyl branch are spelling-invariant
    (P-45-adjacent determinism; seeded random spellings, reproducible)."""

    @_opsin_rt
    @pytest.mark.roundtrip
    @pytest.mark.parametrize("canonical_smiles,expected", [
        ("O=C(OCc1ccccc1)CCCCCCCC(=O)O", "9-(benzyloxy)-9-oxononanoic acid"),
        ("CC(=O)SC(=O)CCCCCCCC(=O)O", "9-(acetylsulfanyl)-9-oxononanoic acid"),
        ("CCSC(=O)CCC(=O)O", "4-(ethylsulfanyl)-4-oxobutanoic acid"),
    ])
    def test_spelling_invariance(self, canonical_smiles, expected):
        from rdkit import Chem
        from rdkit.Chem import rdmolfiles
        mol = Chem.MolFromSmiles(canonical_smiles)
        spellings = set(rdmolfiles.MolToRandomSmilesVect(mol, 10, randomSeed=42))
        names = {name_compound(s) for s in spellings}
        assert names == {expected}, names

    @_opsin_rt
    @pytest.mark.roundtrip
    def test_multi_anchored_mixed_diester_fail_closed(self, monkeypatch):
        # TWO ester carbonyls + one free acid: v1 splits only when EXACTLY ONE
        # anchored match exists; whichever chain perception picks, the unsplit
        # ester's atoms stay unaccounted -> RT gate/SELF-01 refuse. Production
        # is 'unknown' (diagnose 2026-07-11). The autouse fixture disables the
        # validity gate for speed; re-enable it to see the production refusal
        # instead of a gate-off raw leak.
        import orthonym.namer as _namer
        monkeypatch.setattr(_namer, "_DISABLE_VALIDITY_GATE", False, raising=False)
        assert name_compound("COC(=O)CC(CC(=O)OCC)CC(=O)O") == "unknown organic compound"
