"""Tests for fragment naming cycle guard, canonical SMILES, and compound regression.

Tests the infrastructure in fragment_naming.py:
- MAX_NAMING_DEPTH = 7 (legacy constant, kept for backward compatibility)
- Cycle-detection via visited-SMILES set
- Canonical SMILES normalization before calling name_compound()
- Cycle-guard compound regression tests (V8-DEPTH-01)
"""

import pytest
from rdkit import Chem
from rdkit.Chem import inchi
from orthonym.assembly.fragment_naming import (
    MAX_NAMING_DEPTH,
    _fragment_guard,
    _get_visited,
    name_fragment_recursively,
)
from orthonym import name_compound


def _inchikey(smiles: str) -> str:
    """Full (3-block) InChIKey oracle, local to this file. STRICTER than the
    shared ``canonical`` SMILES fixture: RDKit canonical SMILES is
    tautomer-sensitive (it does not normalize mobile-H tautomers), which
    mis-flags e.g. a guanidino ``N=C(N)N`` vs ``NC(N)=N`` redraw as a wrong
    molecule even though the standard-InChI mobile-H layer makes them one
    species. A full InChIKey compares skeleton + stereo + protonation, so it is
    strictly stronger than production's own skeleton-only SELF-01 check and
    stays a genuine 0-wrong guard. Used only by
    ``test_production_never_emits_a_wrong_name`` -- see RB-6 in
    internal notes."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    return inchi.MolToInchiKey(mol)


@pytest.fixture(autouse=True)
def reset_fragment_guard():
    """Reset thread-local state before and after each test."""
    _fragment_guard.visited = set()
    _fragment_guard.cache = None
    yield
    _fragment_guard.visited = set()
    _fragment_guard.cache = None


# ============================================================================
# Group 1: Constants and basic functionality
# ============================================================================


class TestConstants:
    """Verify module-level constants."""

    def test_max_naming_depth_is_7(self):
        assert MAX_NAMING_DEPTH == 7


    def test_basic_fragment_naming_ethanol(self):
        result = name_fragment_recursively("CCO")
        assert result is not None
        assert "ethanol" in result.lower()

    def test_basic_fragment_naming_acetic_acid(self):
        result = name_fragment_recursively("CC(=O)O")
        assert result is not None
        assert "acetic acid" in result.lower()


# ============================================================================
# Group 2: Depth limit behavior
# ============================================================================


class TestCycleGuard:
    """Test cycle-detection guard enforcement."""

    def test_cycle_detected_returns_none_for_uncached(self):
        """When SMILES is in visited set, uncached fragments return None."""
        visited = _get_visited()
        visited.add("CCCCCCCCCCCCCC")  # tetradecane, not in static cache
        result = name_fragment_recursively("CCCCCCCCCCCCCC")
        assert result is None

    def test_cycle_detected_returns_cached(self):
        """When SMILES is in visited set, cached fragments still resolve."""
        visited = _get_visited()
        visited.add("CCO")
        result = name_fragment_recursively("CCO")
        assert result == "ethanol"

    def test_no_cycle_returns_name(self):
        """When SMILES is NOT in visited set, should return a name."""
        result = name_fragment_recursively("CCO")
        assert result is not None

    def test_visited_set_restored_after_call(self):
        """Visited set should not grow after a completed call."""
        result = name_fragment_recursively("CCO")
        assert result is not None
        assert "CCO" not in _get_visited()

    def test_parent_entries_preserved(self):
        """Parent entries in visited set should remain after nested call."""
        visited = _get_visited()
        visited.add("FAKE_PARENT")
        name_fragment_recursively("CCO")
        assert "FAKE_PARENT" in visited


# ============================================================================
# Group 3: Canonical SMILES consistency
# ============================================================================


class TestCanonicalSmilesConsistency:
    """Test that canonical SMILES normalization ensures consistent naming."""

    def test_equivalent_smiles_same_name(self):
        """Same molecule in different SMILES notation produces same name."""
        result1 = name_fragment_recursively("CCO")
        _fragment_guard.visited = set()
        result2 = name_fragment_recursively("OCC")
        assert result1 == result2

    def test_equivalent_smiles_propanol(self):
        """Propan-1-ol in different SMILES forms gives same name."""
        result1 = name_fragment_recursively("CCCO")
        _fragment_guard.visited = set()
        result2 = name_fragment_recursively("OCCC")
        assert result1 == result2

    def test_equivalent_smiles_branched(self):
        """2-methylpropane in different forms gives same name."""
        result1 = name_fragment_recursively("CC(C)C")
        _fragment_guard.visited = set()
        result2 = name_fragment_recursively("C(C)(C)C")
        assert result1 == result2

    def test_sequential_toplevel_calls_both_succeed(self):
        """Two consecutive top-level calls should both succeed."""
        _fragment_guard.visited = set()
        result1 = name_fragment_recursively("CCO")
        assert result1 is not None

        _fragment_guard.visited = set()
        result2 = name_fragment_recursively("CCCO")
        assert result2 is not None


# ============================================================================
# Group 4: Expanded cache entry verification
# ============================================================================


class TestExpandedCacheEntries:
    """Verify all expanded cache entries are correct and present."""

    @pytest.mark.parametrize("smiles,expected_name", [
        ("CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCC(=O)O",
         "(4Z,7Z,10Z,13Z,16Z,19Z)-docosa-4,7,10,13,16,19-hexaenoic acid"),
        ("CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCC(=O)O",
         "(5Z,8Z,11Z,14Z,17Z)-icosa-5,8,11,14,17-pentaenoic acid"),
        ("CCCCCC/C=C\\CCCCCCCC(=O)O", "(9Z)-hexadec-9-enoic acid"),
        ("CCCC/C=C\\CCCCCCCC(=O)O", "(9Z)-tetradec-9-enoic acid"),
        ("CCCCC/C=C\\CCCCCCCC(=O)O", "(9Z)-pentadec-9-enoic acid"),
        ("CCCCC/C=C\\C/C=C\\C/C=C\\CCCCCCC(=O)O",
         "(8Z,11Z,14Z)-icosa-8,11,14-trienoic acid"),
        ("CCCCC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCCCC(=O)O",
         "(7Z,10Z,13Z,16Z)-docosa-7,10,13,16-tetraenoic acid"),
        ("CCCCC/C=C\\C/C=C\\CCCCCCCCCC(=O)O",
         "(11Z,14Z)-icosa-11,14-dienoic acid"),
        ("CC(C)=CC(=O)O", "3-methylbut-2-enoic acid"),
        ("C/C=C(/C)C(=O)O", "(2Z)-2-methylbut-2-enoic acid"),
        ("CCCCCCCCCCCCCCC(=O)O", "pentadecanoic acid"),
        ("CCCCCCCCCCCCCCCCCCC(=O)O", "nonadecanoic acid"),
        ("CC/C=C\\CCO", "(3Z)-hex-3-en-1-ol"),
        ("CC/C=C/CCCCCO", "(6E)-non-6-en-1-ol"),
        ("CCCCCCCCCCCCO", "dodecan-1-ol"),
        ("CCCCCCO", "hexan-1-ol"),
        ("CC(C)CCCCCCCCCCCC(=O)O", "13-methyltetradecanoic acid"),
    ])
    def test_cache_entry_correct(self, smiles, expected_name):
        """Each cache entry must match its verified name."""
        from orthonym.assembly.fragment_naming import FRAGMENT_NAME_CACHE
        from rdkit import Chem
        canonical = Chem.CanonSmiles(smiles)
        assert canonical in FRAGMENT_NAME_CACHE, (
            f"Missing from cache: {smiles} (canonical: {canonical})"
        )
        assert FRAGMENT_NAME_CACHE[canonical] == expected_name, (
            f"Cache mismatch for {canonical}: "
            f"cache={FRAGMENT_NAME_CACHE[canonical]!r}, expected={expected_name!r}"
        )


# ============================================================================
# Group 5: Depth-limit compound regression tests (V8-DEPTH-01)
# ============================================================================

# 25 compounds from the medium molecule triage that mention depth_limit_reached.
# These compounds produce names at depth 7 (some hit internal depth limits but
# still return results via fallback paths). Each must produce a valid name.

DEPTH_LIMIT_COMPOUNDS = [
    pytest.param(
        "CCCCCC(C)OC(=O)COc1ccc(Cl)c2cccnc12",
        id="001_chloroquinoline_ester",
    ),
    pytest.param(
        "CC1(C)SC(C(NC(=O)COc2ccccc2)C(=O)O)NC1C(=O)O",
        id="002_penicillin_like",
    ),
    pytest.param(
        "C=C[C@](C)(O)CCC=C(C)CCC1OC(C)(C)OC1(C)C",
        id="003_terpene_dioxolane",
    ),
    pytest.param(
        "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O[C@@H]1OC[C@@H](O)[C@H](O)[C@H]1O",
        id="004_disaccharide_xylose",
    ),
    pytest.param(
        "C[C@H]1C[C@@H](O)[C@@]23C1=C[C@@]1(C)CC[C@](C)(C[C@H](O)[C@H](O)[C@@](C)(O)CO)[C@H]1[C@@H]2CC[C@@H]3C",
        id="005_steroid_polyol",
    ),
    pytest.param(
        "COc1cc(-c2ccc(O)c(CC=C(C)C)c2)c(OC)c(O)c1-c1ccc(O)c(O)c1",
        id="006_terphenyl_prenyl",
    ),
    pytest.param(
        "OC[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)CO[C@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O",
        id="007_galactitol_glucoside",
    ),
    pytest.param(
        "C=CCN(C)CCCCCCOc1ccc(C(=O)c2ccc(Br)cc2)c(F)c1",
        id="008_allylamine_benzophenone",
    ),
    pytest.param(
        "CCCCCCCCCCCCCCCCCCCCCCCC(=O)NCCc1c[nH]c2ccccc12",
        id="009_tryptamine_amide",
    ),
    pytest.param(
        "OC[C@H]1O[C@H](O)[C@H](O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]2O)[C@@H](O)[C@H]1O",
        id="010_disaccharide_mannose",
    ),
    pytest.param(
        "C=C1C(=O)O[C@@H]2C[C@@H](C)/C=C\\C(=O)[C@@](C)(O)C[C@@H](OC(=O)CC(C)C)[C@@H]12",
        id="011_macrolide_lactone",
    ),
    pytest.param(
        "COc1cc(OC)c(C(C)=O)c(O)c1CCOCCc1c(O)cc(OC)c(C(C)=O)c1O",
        id="012_biaryl_ether",
    ),
    pytest.param(
        "NC(=O)CC[C@H](NC(=O)[C@@H]1CCCN1C(=O)[C@@H]1CCCN1)C(=O)O",
        id="013_dipeptide_proline",
    ),
    pytest.param(
        "CNCC[C@H](Oc1cccc2ccccc12)c1cccs1",
        id="014_naphthyl_thiophene",
    ),
    pytest.param(
        "COc1c(-c2ccc(O)cc2)oc2c(O)c(O)ccc2c1=O",
        id="015_flavonoid",
    ),
    pytest.param(
        "C=C(C)C(=O)Cc1c(C)cc(Oc2cc(CO)cc(OC)c2)cc1OC",
        id="016_phenol_ether_ketone",
    ),
    pytest.param(
        "CC[C@H](C)[C@H](NC(=O)[C@H](C)NC(=O)[C@@H](N)CCCN=C(N)N)C(=O)O",
        id="017_tripeptide_arginine",
    ),
    pytest.param(
        "C[C@]12CC[C@H](O)c3coc(c31)C(=O)C1=C2[C@@H](O)C[C@]2(C)C(=O)CC[C@@H]12",
        id="018_steroid_furanone",
    ),
    pytest.param(
        "CC(C)C1=C(O)C(N)=C(/C=C/c2ccccc2)C(=O)C1=O",
        id="019_aminoquinone_styryl",
    ),
    pytest.param(
        "CC(C)[C@H](NC(=O)[C@@H](N)CC(=O)O)C(=O)NCC(=O)N1CCC[C@H]1C(=O)O",
        id="020_tetrapeptide",
    ),
    pytest.param(
        "CC(=O)N[C@H]1C(OP(=O)(O)OP(=O)(O)OC[C@H]2O[C@@H](n3ccc(=O)[nH]c3=O)[C@H](O)[C@@H]2O)O[C@H](CO)[C@H](O)[C@@H]1O",
        id="021_udp_sugar",
    ),
    pytest.param(
        "COc1cc(OC)c2c(=O)c3c(O)cc(C)cc3oc2c1",
        id="022_xanthone_dimethoxy",
    ),
    pytest.param(
        "N[C@@H](COC(=O)CCC(=O)O)C(=O)O",
        id="023_serine_succinate",
    ),
    pytest.param(
        "C=C(C(=O)OC)N1C(=O)C[C@@H](C)C1=O",
        id="024_acrylate_pyrrolidinedione",
    ),
    pytest.param(
        "CCCCCCC(=O)NC1=CC(=O)[C@@H]2CCCN12",
        id="025_pyrrolizinone_amide",
    ),
]


# ---------------------------------------------------------------------------
# Round-trip expectations for the depth-limit compounds.
#
# ⚠ These assertions had NEVER RUN before 2026-08-01. The shared
# `opsin_to_smiles` fixture invoked OPSIN as `java -jar <jar> -osmi <name>`;
# OPSIN's CLI reads a trailing argument as an INPUT FILE, so every call raised
# FileNotFoundException and the fixture returned None for every name. The
# assertion sat behind `if parsed:` and so was skipped on every run.
#
# With the fixture repaired, 16 of the 25 compounds do not round-trip. MEASURED
# what production actually does with those 16 (the suite disables the OPSIN
# validity gate; production has it ON — `namer._DISABLE_VALIDITY_GATE` is False
# by default):
#
# failing round-trips................................ 16 / 25
# of those, production emits 'unknown organic compound' 16 / 16
# of those, production ships the raw name.............. 0 / 16
#
# So these are BREADTH gaps (the generator cannot yet name these structures and
# the gate correctly abstains), NOT wrong-name defects — no incorrect name
# reaches a caller. They are xfailed with `strict=True` so that closing any one
# of them turns this suite RED and forces the entry to be removed, rather than
# rotting into a permanent excuse. `test_production_never_emits_a_wrong_name`
# below pins the invariant that actually matters for these rows.
#
# Full evidence: internal notes
# ---------------------------------------------------------------------------
_RT_BREADTH_GAPS = {
    "001_chloroquinoline_ester",
    "002_penicillin_like",
    "003_terpene_dioxolane",
    "005_steroid_polyol",
    "007_galactitol_glucoside",
    # "008_allylamine_benzophenone" -- REMOVED 2026-08-21: now round-trips
    # (verify_or_none True): '(4-bromophenyl)({2-fluoro-4-[6-(methyl(prop-2-en-
    # 1-yl)amino)hexyloxy]phenyl})methanone'. Same treatment as 013/020.
    "011_macrolide_lactone",
    "012_biaryl_ether",
    # "013_dipeptide_proline" -- REMOVED 2026-08-21 (a phase Task 2.2
    # cleanup): verified via `scripts/an A/B check` that this already
    # round-trips correctly at HEAD 799d3491 (before Task 2.2's own code),
    # i.e. Task 2.0/2.1 (the peptide dispatch SMARTS fix + Lever C) closed
    # this gap and nobody removed the stale xfail entry. Now
    # 'prolylprolylglutamine', OPSIN round-trip verified exact.
    # "016_phenol_ether_ketone" -- REMOVED 2026-08-21: now round-trips
    # (verify_or_none True): '1-({4-[3-(hydroxymethyl)-5-methoxyphenoxy]-2-
    # methoxy-6-methylphenyl})-3-methylbut-3-en-2-one'.
    "017_tripeptide_arginine",
    "018_steroid_furanone",
    # "020_tetrapeptide" -- REMOVED 2026-08-21, same cause/verification as
    # 013 above. Now 'aspartylvalylglycylproline', OPSIN round-trip
    # verified exact.
    "021_udp_sugar",
    # "023_serine_succinate" -- REMOVED 2026-08-21: now round-trips
    # (verify_or_none True): '4-[(S)-2-amino-2-carboxyethoxy]-4-oxobutanoic acid'.
    "025_pyrrolizinone_amide",
}

_RT_GAP_REASON = (
    "BREADTH GAP measured 2026-08-01, not a wrong-name defect. With the OPSIN "
    "validity gate DISABLED (the suite-wide test default) the raw generator "
    "emits a name that either OPSIN cannot parse (e.g. "
    "'N-(4-oxo-3-propan-3-ylsubstituent)heptanamide') or that denotes a "
    "different structure (e.g. serine succinate -> 'butanedioic acid', a "
    "silent atom drop). With the gate ON — which is production's default — "
    "these compounds emit 'unknown organic compound' instead, so no "
    "wrong name ships (originally 16; 2 peptide entries closed 2026-08-21, "
    "see _RT_BREADTH_GAPS). strict=True: fix the generator and this goes red."
)

_RT_PARAMS = [
    pytest.param(
        p.values[0],
        id=p.id,
        marks=pytest.mark.xfail(strict=True, reason=_RT_GAP_REASON),
    )
    if p.id in _RT_BREADTH_GAPS
    else p
    for p in DEPTH_LIMIT_COMPOUNDS
]

assert len(_RT_BREADTH_GAPS) == 11, "the measured gap list changed size"
assert _RT_BREADTH_GAPS <= {p.id for p in DEPTH_LIMIT_COMPOUNDS}, (
    "a _RT_BREADTH_GAPS id does not match any DEPTH_LIMIT_COMPOUNDS param"
)


class TestDepthLimitCompounds:
    """Regression tests for compounds that previously hit depth_limit_reached.

    These compounds produce names at depth 7 via fallback paths. The test
    verifies that naming completes without producing None or 'unknown'.
    """

    @pytest.mark.parametrize("smiles", DEPTH_LIMIT_COMPOUNDS)
    def test_naming_completes(self, smiles):
        """Compound should produce a valid name (not None, not 'unknown')."""
        result = name_compound(smiles)
        assert result is not None, f"name_compound returned None for {smiles}"
        assert "unknown" not in result.lower(), f"Name contains unknown: {result}"

    @pytest.mark.roundtrip
    @pytest.mark.parametrize("smiles", _RT_PARAMS)
    def test_opsin_roundtrip(self, smiles, opsin_to_smiles, canonical):
        """Named compound should parse back via OPSIN (Tier 2 round-trip)."""
        result = name_compound(smiles)
        assert result is not None, f"name_compound returned None for {smiles}"
        parsed = opsin_to_smiles(result)
        assert parsed is not None, (
            f"OPSIN could not parse the emitted name: {smiles} -> '{result}'"
        )
        assert canonical(parsed) == canonical(smiles), (
            f"Round-trip mismatch: {smiles} -> '{result}' -> {parsed}"
        )

    @pytest.mark.roundtrip
    @pytest.mark.opsin_gate
    def test_production_never_emits_a_wrong_name(self, opsin_to_smiles):
        """The invariant that actually holds for all 25: production either
        names the compound CORRECTLY or abstains — it never ships a name that
        denotes a different structure.

        This runs with the OPSIN validity gate ENABLED (`opsin_gate` marker),
        i.e. production's real configuration, unlike every other test in this
        file. Skips rather than passes when the jar is absent, because the gate
        fails OPEN without it (see tests/conftest.py).

        RB-6: the oracle is the full-InChIKey ``_inchikey`` helper, not the
        shared (tautomer-sensitive) ``canonical`` SMILES fixture, so a mobile-H
        redraw of the same species (e.g. row 017's guanidino group) is not
        mis-flagged as a wrong molecule. See RB3-RB6-DERIVATION.md.
        """
        emitted, abstained, wrong = [], [], []
        for param in DEPTH_LIMIT_COMPOUNDS:
            smiles = param.values[0]
            name = name_compound(smiles)
            assert name is not None, f"name_compound returned None for {smiles}"
            if "unknown" in name.lower():
                abstained.append(param.id)
                continue
            parsed = opsin_to_smiles(name)
            if parsed is not None and _inchikey(parsed) == _inchikey(smiles):
                emitted.append(param.id)
            else:
                wrong.append((param.id, name, parsed))

        # Anti-vacuity: a generator that abstained on all 25 would satisfy
        # "never wrong" trivially. Pin the measured floor so a coverage
        # regression is a failure, not a silent pass.
        assert len(emitted) >= 9, (
            f"only {len(emitted)} of 25 depth-limit compounds round-trip under "
            f"the production gate; 9 did on 2026-08-01. Coverage regressed. "
            f"Correct: {emitted}"
        )
        assert not wrong, (
            "production emitted a name denoting a DIFFERENT structure — this "
            f"is a 0-wrong violation, not a breadth gap: {wrong}"
        )
        assert len(emitted) + len(abstained) == len(DEPTH_LIMIT_COMPOUNDS)
