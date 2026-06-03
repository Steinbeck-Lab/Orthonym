"""Wave-0 unit suite for the generalized ionic-suffix re-application seam.

Phase 169.6 Plan 02 (CHOKE-02). This is the file named in 169.6-VALIDATION.md
Wave 0: it MUST exist and pass before Plan 03 wires ``route_charged`` to the
generalized primitive ``apply_ion_suffix_to_name``.

Every assertion carries the verbatim Blue Book authority that governs the
transform (USER DIRECTIVE: every charged sub-class transform is cross-checked
against the Blue Book rule, with the citation in the test). Worked targets come
from 169.6-AUDIT-CHOKEPOINT.md §4/§5 and the verbatim P-72/P-73 extractions in


Covered:
  - the 4 corrected map entries (audit §4.1): anion amide/carboxamide/
    carbonitrile no longer map to -ate; cation ol no longer maps to -olium.
  - the added map entries (audit §4.2): anion ous acid->ite;
    cation amide/carboxamide/imide/carboximide/nitrile/carbonitrile/imine->...ium.
  - the 3 class-keyed cation transforms (audit §4.3): carbenium ane->ylium,
    acylium acid->oylium/carbonylium, diazonium append diazonium.
  - GUARD 1 (allowed_suffixes): the heptanolate misroute is mechanically
    impossible (a sulfonate stem cannot mis-fire to -olate).
  - the anion path is unchanged (byte-identical 169.5 seam reuse).
"""

import pytest

from orthonym.rules.ions import apply_ion_suffix_to_name, _ionize_acid_name
from orthonym.assembly.resolvers import (
    _ANION_SUFFIX_MAP,
    _CATION_SUFFIX_MAP,
    _apply_anion_modification,
    SuffixInfo,
)

pytestmark = pytest.mark.unit


# ===========================================================================
# §4.1 FIX — the 4 non-IUPAC map entries are corrected
# ===========================================================================

class TestCorrectedAnionMapEntries:
    """Audit §4.1: amide-family anions take the anionic-parent-hydride form."""

    def test_amide_no_longer_maps_to_amidate(self):
        """P-72.2.2.2.4: CH3-CO-NH(-) -> acetylazanide (anionic parent hydride),
        NOT a suffix -ate. The wrong 'amide->amidate' entry was removed; the
        generic anion seam now returns '' so route_charged falls through."""
        assert "amide" not in _ANION_SUFFIX_MAP
        assert apply_ion_suffix_to_name("acetamide", -1) == ""

    def test_carboxamide_no_longer_maps_to_carboxamidate(self):
        """P-72.2.2.2.4: 'Suffixes such as ... carboxamidide are not
        recommended' — the carboxamide->carboxamidate entry was removed."""
        assert "carboxamide" not in _ANION_SUFFIX_MAP
        assert apply_ion_suffix_to_name("cyclohexanecarboxamide", -1) == ""

    def test_carbonitrile_no_longer_maps_to_carbonitrilate(self):
        """P-72.2.2.2.4: nitrile anions are named on the anionic parent hydride,
        NOT a suffix -ate. The carbonitrile->carbonitrilate entry was removed."""
        assert "carbonitrile" not in _ANION_SUFFIX_MAP
        assert apply_ion_suffix_to_name("benzenecarbonitrile", -1) == ""

    def test_apply_anion_modification_leaves_amide_unchanged(self):
        """Audit §4.1 sanity: the generic seam, given SuffixInfo(text='amide'),
        returns the text UNCHANGED (no -ate) per P-72.2.2.2.4."""
        out = _apply_anion_modification(SuffixInfo(text="amide", is_terminal=True))
        assert out.text == "amide"

    def test_carboxylic_acid_to_carboxylate_still_correct(self):
        """P-72.2.2.2.1.1: carboxylic acid->carboxylate is CORRECT and STAYS —
        it was never in the FIX set (audit §4.1 note)."""
        assert _ANION_SUFFIX_MAP.get("carboxylic acid") == "carboxylate"


class TestCorrectedCationMapEntries:
    """Audit §4.1: protonated alcohol is oxidanium-based, not -olium."""

    def test_ol_no_longer_maps_to_olium(self):
        """P-73.1.2.1: a protonated alcohol is named on the oxidanium parent
        cation (e.g. ethylideneoxidanium), NOT a bogus -olium suffix. The wrong
        'ol->olium' entry was removed."""
        assert "ol" not in _CATION_SUFFIX_MAP


# ===========================================================================
# §4.2 ADD — the missing map entries
# ===========================================================================

class TestAddedAnionMapEntries:
    def test_ous_acid_to_ite(self):
        """P-72.2.2.2.1.1: "the 'ic acid' or 'ous acid' ending ... by 'ate' or
        'ite', respectively" — the -ous-acid anion takes -ite."""
        assert _ANION_SUFFIX_MAP.get("ous acid") == "ite"


class TestAddedCationMapEntries:
    """Table 7.4 (P-73.1.2.1): cationic characteristic-group suffixes formed by
    adding 'ium' to the neutral nitrogen-bearing suffix."""

    @pytest.mark.parametrize("neutral,cationic", [
        ("amide", "amidium"),            # Table 7.4
        ("carboxamide", "carboxamidium"),  # Table 7.4
        ("imide", "imidium"),            # Table 7.4
        ("carboximide", "carboximidium"),  # Table 7.4
        ("nitrile", "nitrilium"),        # Table 7.4
        ("carbonitrile", "carbonitrilium"),  # Table 7.4
        ("imine", "iminium"),            # Table 7.4
    ])
    def test_nitrogen_suffix_plus_ium(self, neutral, cationic):
        """Table 7.4: neutral N-bearing suffix + 'ium' -> cationic suffix."""
        assert _CATION_SUFFIX_MAP.get(neutral) == cationic

    def test_amine_to_aminium_retained(self):
        """P-73.1.2.1 WAY1 / Table 7.4: protonated amine -> -aminium (the PIN);
        this pre-existing entry stays."""
        assert _CATION_SUFFIX_MAP.get("amine") == "aminium"


# ===========================================================================
# §4.3 — the 3 class-keyed cation transforms (verbatim Blue Book targets)
# ===========================================================================

class TestClassKeyedCarbenium:
    def test_methane_to_methylium(self):
        """P-73.2.2.1.1: 'replacing the ane ending ... by the suffix ylium'.
        CH3+ -> methylium (PIN). "carbenium" is NOT a substitutive PIN."""
        assert apply_ion_suffix_to_name("methane", 1, cation_class="ylium") == "methylium"

    def test_ethane_to_ethylium(self):
        """P-73.2.2.1.1: ethane -> ethylium (ane->ylium)."""
        assert apply_ion_suffix_to_name("ethane", 1, cation_class="ylium") == "ethylium"

    def test_propane_to_propylium(self):
        """P-73.2.2.1.1: propane -> propylium (the verbatim example, P-73 detail)."""
        assert apply_ion_suffix_to_name("propane", 1, cation_class="ylium") == "propylium"

    def test_cyclobutane_to_cyclobutylium(self):
        """P-73.2.2.1.1: saturated monocyclic hydride cyclobutane -> cyclobutylium
        (the verbatim example)."""
        assert apply_ion_suffix_to_name("cyclobutane", 1, cation_class="ylium") == "cyclobutylium"


class TestClassKeyedAcylium:
    def test_butanoic_acid_to_butanoylium(self):
        """P-73.2.3.1: 'replacing the oic acid ... ending by the suffix oylium'.
        Operates on the ACID name (CONTEXT D-02)."""
        assert apply_ion_suffix_to_name("butanoic acid", 1, cation_class="acylium") == "butanoylium"

    def test_cyclohexanecarboxylic_acid_to_cyclohexanecarbonylium(self):
        """P-73.2.3.1: 'the carboxylic acid ending by carbonylium'.
        cyclohexanecarboxylic acid -> cyclohexanecarbonylium (verbatim example)."""
        assert apply_ion_suffix_to_name(
            "cyclohexanecarboxylic acid", 1, cation_class="acylium"
        ) == "cyclohexanecarbonylium"


class TestClassKeyedDiazonium:
    def test_benzene_to_benzenediazonium(self):
        """P-73.2.2.3: 'using the suffix diazonium' — append to the hydride name
        (no elision; consonant-initial). benzene -> benzenediazonium."""
        assert apply_ion_suffix_to_name("benzene", 1, cation_class="diazonium") == "benzenediazonium"

    def test_methane_to_methanediazonium(self):
        """P-73.2.2.3: CH3-N2+ -> methanediazonium (PIN; verbatim example, no
        e-elision because 'diazonium' begins with a consonant)."""
        assert apply_ion_suffix_to_name("methane", 1, cation_class="diazonium") == "methanediazonium"


# ===========================================================================
# GUARD 1 — allowed_suffixes makes the heptanolate misroute impossible
# ===========================================================================

class TestGuard1AllowedSuffixes:
    def test_sulfonate_stem_cannot_mis_fire_to_olate(self):
        """GUARD 1 (the heptanolate-dead proof): a name ending '...sulfonic acid'
        with allowed_suffixes={'ol'} returns '' — it can NEVER produce -olate.
        P-72.2.2.2.1 (acid anions->-ate) vs P-72.2.2.2.2 (hydroxy anions->-olate):
        the seam is purely textual, so the per-class subset must gate it."""
        assert apply_ion_suffix_to_name(
            "hexane-1-sulfonic acid", -1, allowed_suffixes=frozenset({"ol"})
        ) == ""

    def test_sulfonate_stem_with_correct_allowed_set_fires(self):
        """GUARD 1 positive: with allowed_suffixes={'sulfonic acid'} the same stem
        correctly produces -sulfonate (P-72.2.2.2.1.1)."""
        assert apply_ion_suffix_to_name(
            "propane-1-sulfonic acid", -1, allowed_suffixes=frozenset({"sulfonic acid"})
        ) == "propane-1-sulfonate"


# ===========================================================================
# The anion path is unchanged (byte-identical 169.5 seam reuse)
# ===========================================================================

class TestAnionPathUnchanged:
    def test_propanesulfonic_acid_to_propanesulfonate(self):
        """P-72.2.2.2.1.1: the 169.5 byte-identical anion path is reused
        unchanged through the generalized primitive."""
        assert apply_ion_suffix_to_name(
            "propane-1-sulfonic acid", -1, allowed_suffixes=frozenset({"sulfonic acid"})
        ) == "propane-1-sulfonate"

    def test_butanoic_acid_to_butanoate(self):
        """P-72.2.2.2.1.1: oic acid->oate (the canonical carboxylate transform)."""
        assert apply_ion_suffix_to_name("butanoic acid", -1) == "butanoate"

    def test_ol_to_olate(self):
        """P-72.2.2.2.2: -ol->-olate (alkoxide). Unchanged generic-path behavior."""
        assert apply_ion_suffix_to_name("butan-1-ol", -1) == "butan-1-olate"

    def test_back_compat_wrapper_matches_primitive(self):
        """169.6-02: _ionize_acid_name remains a thin wrapper delegating to
        apply_ion_suffix_to_name, so _name_oxoacid_anion keeps working
        byte-identical (no cation_class)."""
        assert _ionize_acid_name(
            "propane-1-sulfonic acid", -1, allowed_suffixes=frozenset({"sulfonic acid"})
        ) == "propane-1-sulfonate"
        assert _ionize_acid_name("butanoic acid", -1) == "butanoate"


# ===========================================================================
# No transform applies -> '' (v18 fall-through contract)
# ===========================================================================

class TestNoTransformReturnsEmpty:
    def test_unmatched_neutral_name_returns_empty(self):
        """v18 contract: when no canonical transform applies the primitive
        returns '' so the caller falls through to the existing cascade."""
        assert apply_ion_suffix_to_name("benzene", -1) == ""

    def test_empty_name_returns_empty(self):
        assert apply_ion_suffix_to_name("", 1, cation_class="ylium") == ""
