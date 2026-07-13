"""Unit tests for PREFIX_FORMS completeness against SENIORITY_ORDER.

Verifies IUPAC P-59.1 compliance:
- Every FG type in SENIORITY_ORDER has a corresponding key in PREFIX_FORMS.
- Entries with None prefix form are genuinely functional-class-only (IUPAC has
  no substitutive prefix for these groups).
- No FG in SENIORITY_ORDER with a SUFFIX_FORMS entry but no PREFIX_FORMS entry
  (unless legitimately functional-class-only).
"""

import pytest
from orthonym.rules.seniority import SENIORITY_ORDER, PREFIX_FORMS, SUFFIX_FORMS


# Functional-class-only FG types: these have no IUPAC substitutive prefix form.
# They use functional class naming, decomposition pathways, or heterocyclic naming.
# Each entry is justified by an IUPAC rule reference.
KNOWN_NONE_PREFIX_FGS = frozenset({
    'ester',             # P-65.1: functional class (alkyl alkanoate) or alkoxycarbonyl
    'ether',             # P-63.2: named by substitution (methoxy, ethoxy)
    'thioether',         # P-63.2: functional class (dialkyl sulfide)
    'thioester',         # P-65.3.1: functional class (S-alkyl alkanethioate)
    'carbamate',         # P-65.2.3: functional class naming
    'n_oxide_aromatic',  # P-62.5: functional class only
    'n_oxide_aliphatic', # P-62.5: functional class only
    'anhydride',         # P-65.1: functional class naming only
    'secondary_amide',   # Named via acylamino pathway in universal pipeline
    'tertiary_amide',    # Named via acylamino pathway in universal pipeline
    'imide',             # Named as heterocyclic ring substituent
    'phosphine_oxide',   # P-68.3: functional class naming
    'phosphate_triester',  # P-68: functional class naming
    'phosphate_diester',   # P-68: functional class naming
    'tertiary_phosphine',  # P-68.3: parent hydride naming (phosphane)
    'secondary_phosphine', # P-68.3: parent hydride naming
    'primary_phosphine',   # P-68.3: parent hydride naming
    # Phase 163 P-25.3 functional replacement nomenclature (additive seniority extension,
    # ADR-19-09). Each entry mirrors its non-chalcogen ester counterpart's functional-class
    # PIN rule. Substituent rendering is handled by acyl-derived prefixes (alkanimidoyl,
    # alkaneselenoyl, alkanetelluroyl) where needed; none is the principal-group prefix.
    # Wave2 completion B4 (fail-closed demotions, no BB-attested prefix):
    'peroxy_acid',       # P-43.1: demoted case fails closed (no PIN acid prefix)
    'imidic_acid',       # P-65.1.3.1: demoted case fails closed
    # W3-P02 (P-65.1.3.2): hydrazonic acid has no SINGLE static prefix — when
    # demoted at a chain end it splits into 'hydroxy' + 'hydrazinylidene'
    # (P-65.1.3.2.2), emitted by name_polyfunctional's chain-end block.
    'hydrazonic_acid',
    'sulfinohydrazonohydrazide',  # P-66.4.3.2: demoted case fails closed
    'iminoester',        # P-65.1.7: functional class (alkyl alkanimidate); imidate handler @ INNER_DISPATCH 2900
    'selenoester',       # P-65.3: functional class (Se-alkyl alkaneselenoate); chalcogen analog of ester
    'telluroester',      # P-65.3: functional class (Te-alkyl alkanetelluroate); chalcogen analog of ester
    # BBR-PERC (Phase 169.7): functional parents (P-67/P-68.3) + Se/Te ethers (P-63.6).
    'hydroxylamine',     # P-68.3: parent hydride "hydroxylamine"; named via the hydroxylamine handler
    'phosphoric_acid',   # P-67: free inorganic oxoacid functional parent
    'sulfuric_acid',     # P-67: free inorganic oxoacid functional parent
    'nitric_acid',       # P-67: free inorganic oxoacid functional parent
    'carbonic_acid',     # P-65.2.1: functional parent (HO-C(=O)-OH)
    'selenoether',       # P-63.6: functional class / substitutive (alkyl)selanyl
    'telluroether',      # P-63.6: functional class / substitutive (alkyl)tellanyl
    # DD2 (Phase D, P-63.3.1(1)): R-OO-R' substituent prefix is (R)peroxy, generated
    # dynamically by substituent_prefix_forms.get_peroxy_prefix (exactly like ether/ester
    # above — no static principal-group prefix form).
    'peroxide',          # P-63.3.1(1): substitutive (R)peroxy via get_peroxy_prefix
})


@pytest.mark.unit
class TestPrefixFormsCompleteness:
    """Every FG in SENIORITY_ORDER must have a PREFIX_FORMS entry."""

    def test_all_seniority_fgs_have_prefix_form_entry(self):
        """Every FG type in SENIORITY_ORDER must have a key in PREFIX_FORMS."""
        missing = [fg for fg in SENIORITY_ORDER if fg not in PREFIX_FORMS]
        assert missing == [], (
            f"FG types in SENIORITY_ORDER but missing from PREFIX_FORMS: {missing}"
        )

    def test_none_entries_are_known_safe(self):
        """Entries with None prefix form must be in the known-safe list.
        These are genuinely functional-class-only groups with no IUPAC prefix.
        """
        none_entries = {fg for fg, v in PREFIX_FORMS.items() if v is None}
        unexpected_none = none_entries - KNOWN_NONE_PREFIX_FGS
        assert unexpected_none == set(), (
            f"Unexpected None entries in PREFIX_FORMS (not in known-safe list): "
            f"{unexpected_none}. If these are genuinely functional-class-only, "
            f"add them to KNOWN_NONE_PREFIX_FGS with IUPAC rule reference."
        )

    def test_suffix_fgs_without_prefix_are_justified(self):
        """FG types with suffix forms but None prefix must be functional-class-only.
        If a group has a suffix (can be principal group), it should also have a
        prefix (for when it's non-principal) UNLESS it's genuinely functional-class-only.
        """
        problematic = []
        for fg in SENIORITY_ORDER:
            has_suffix = fg in SUFFIX_FORMS and SUFFIX_FORMS[fg] is not None
            prefix = PREFIX_FORMS.get(fg)
            if has_suffix and prefix is None:
                if fg not in KNOWN_NONE_PREFIX_FGS:
                    problematic.append(fg)
        assert problematic == [], (
            f"FG types with suffix forms but no prefix form and not in known-safe list: "
            f"{problematic}. These groups need either a prefix form or explicit "
            f"justification in KNOWN_NONE_PREFIX_FGS."
        )
