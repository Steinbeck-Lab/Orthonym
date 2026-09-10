"""Unit tests for PREFIX_FORMS completeness against SENIORITY_ORDER.

Verifies IUPAC compliance:
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
    'ester',             #: functional class (alkyl alkanoate) or alkoxycarbonyl
    'ether',             #: named by substitution (methoxy, ethoxy)
    'thioether',         #: functional class (dialkyl sulfide)
    'thioester',         #: functional class (S-alkyl alkanethioate)
    'carbamate',         #: functional class naming
    'n_oxide_aromatic',  #: functional class only
    'n_oxide_aliphatic', #: functional class only
    'anhydride',         #: functional class naming only
    'secondary_amide',   # Named via acylamino pathway in universal pipeline
    'tertiary_amide',    # Named via acylamino pathway in universal pipeline
    'imide',             # Named as heterocyclic ring substituent
    'phosphine_oxide',   #: functional class naming
    'phosphate_triester',  #: functional class naming
    'phosphate_diester',   #: functional class naming
    'tertiary_phosphine',  #: parent hydride naming (phosphane)
    'secondary_phosphine', #: parent hydride naming
    'primary_phosphine',   #: parent hydride naming
    # a phase functional replacement nomenclature (additive seniority extension,
    # -09). Each entry mirrors its non-chalcogen ester counterpart's functional-class
    # PIN rule. Substituent rendering is handled by acyl-derived prefixes (alkanimidoyl,
    # alkaneselenoyl, alkanetelluroyl) where needed; none is the principal-group prefix.
    # Wave2 completion B4 (fail-closed demotions, no BB-attested prefix):
    'peroxy_acid',       #: demoted case fails closed (no PIN acid prefix)
    'imidic_acid',       #: demoted case fails closed
    # W3-P02: hydrazonic acid has no SINGLE static prefix — when
    # demoted at a chain end it splits into 'hydroxy' + 'hydrazinylidene'
    #, emitted by name_polyfunctional's chain-end block.
    'hydrazonic_acid',
    # W3-P02: hydroximic acid — suffix is the N-hydroxy imidic acid
    # (dedicated handler, no static suffix); demoted it splits into 'hydroxy' +
    # 'hydroxyimino', emitted by name_polyfunctional's chain-end block.
    'hydroximic_acid',
    'sulfinohydrazonohydrazide',  #: demoted case fails closed
    'iminoester',        #: functional class (alkyl alkanimidate); imidate handler @ INNER_DISPATCH 2900
    'selenoester',       #: functional class (Se-alkyl alkaneselenoate); chalcogen analog of ester
    'telluroester',      #: functional class (Te-alkyl alkanetelluroate); chalcogen analog of ester
    # W3-P07: non-carbon esters — functional-class only, named by the shared
    # esters.name_noncarbon_ester handler (no principal-group substituent prefix).
    'pseudoester',       # /: functional class (Zyl acylate)
    'sulfonic_ester',    #: functional class (alkyl alkanesulfonate)
    'sulfinic_ester',    #: functional class (alkyl alkanesulfinate)
    # BBR-PERC (a phase): functional parents / + Se/Te ethers.
    'hydroxylamine',     #: parent hydride "hydroxylamine"; named via the hydroxylamine handler
    'phosphoric_acid',   #: free inorganic oxoacid functional parent
    'sulfuric_acid',     #: free inorganic oxoacid functional parent
    'nitric_acid',       #: free inorganic oxoacid functional parent
    'carbonic_acid',     #: functional parent (HO-C(=O)-OH)
    'selenoether',       #: functional class / substitutive (alkyl)selanyl
    'telluroether',      #: functional class / substitutive (alkyl)tellanyl
    # DD2 (Phase D, (1)): R-OO-R' substituent prefix is (R)peroxy, generated
    # dynamically by substituent_prefix_forms.get_peroxy_prefix (exactly like ether/ester
    # above — no static principal-group prefix form).
    'peroxide',          # (1): substitutive (R)peroxy via get_peroxy_prefix
    # ------------------------------------------------------------------
    # -CLEANUP: these nine were added to SENIORITY_ORDER by later waves
    # WITHOUT being justified here, so both tests in this class had been FAILING.
    # That is the audit working as designed — the tripwire fired and nobody
    # answered it. Each already carries an explicit, rule-cited "demoted case
    # fails closed" decision in `rules/seniority.py`; the justification is
    # transcribed here so the list is once again a complete record.
    #
    # All nine are DELIBERATE fail-closed, not oversights: `PREFIX_FORMS[fg] is
    # None` means a molecule where the group is DEMOTED (a senior group is
    # co-present) abstains instead of emitting an unattested prefix. Per
    # a project rule that is the right trade only because abstention here is not
    # masking a worse generator — the alternative is inventing a prefix string.
    #
    # The two halides are a DIFFERENT case from the other seven and are noted as
    # such: a Blue Book prefix genuinely EXISTS for them, but it is
    # HALOGEN-DEPENDENT, so no single static string can express it.
    'sulfonyl_halide',   # -SO2-X. A BB preselected prefix EXISTS but varies with
                         # the halogen: `chlorosulfonyl` (the Blue Book,
                         # `--SO2-Cl chlorosulfonyl (preselected prefix)`,
                         # /, `fluorosulfonyl`, etc.,
                         # cf. the PIN `3-[(chlorosulfonyl)oxy]propanoic acid`
                         # (:36492). A static PREFIX_FORMS string cannot carry
                         # the halogen, so this stays None and the demoted case
                         # fails closed rather than guess one.
    'sulfinyl_halide',   # -S(=O)-X, same shape: `chlorosulfinyl` (:36482,
                         # `--S(=O)-Cl chlorosulfinyl (preselected prefix)`).
    'sulfonoperoxoic_acid',   # W3-P04: demoted prefix fails closed
    'sulfonothioic_S_acid',   # W3-P04: demoted prefix fails closed
    'sulfonimidic_acid',      # W3-P04: demoted prefix fails closed
    'sulfinimidic_acid',      # W3-P04: demoted prefix fails closed
    'selenonimidamide',       #: demoted prefix fails closed
    'seleninimidamide',       #: demoted prefix fails closed
    'imidohydrazide',         # ring prefix not built; the chain-end
                              # split is owned elsewhere -> fail closed
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
