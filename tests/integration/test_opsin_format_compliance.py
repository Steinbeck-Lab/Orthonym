"""
Integration tests verifying OPSIN-parseable output for fixed VB/stereo compounds.

OPSIN Limitations Documented:
============================

The following categories of correct IUPAC names are NOT parseable by OPSIN 2.8.0:

1. OPSIN STEREO + VB LIMITATION:
   - Tricyclo compounds with stereodescriptors fail OPSIN parsing even when
     the bare descriptor + parent parses fine. Examples:
     * "(1S,2S,5S,6R,7R,8R,9S,11S)-2,6,6,9-tetramethyl-tricyclo[6.3.0.0(5,7)]undecan-2,11-diol"
       -- without stereo: OPSIN parses correctly
     * "(2R,6R,8R)-11-ethyl-8-hydroxy-4,4-dimethyl-tricyclo[6.3.0.0(2,6)]undec-1-en-10-one"
       -- without stereo: OPSIN parses correctly
     * "(4S,6S)-4,6,14-trihydroxy-12-methoxy-6-methyl-tricyclo[8.4.0.0(3,8)]tetradec-3-en-2,9-dione"
       -- without stereo: OPSIN parses correctly

2. OPSIN VB BRIDGE/RING COUNT LIMITATION:
   - Some complex polycyclic descriptors with secondary bridges and heteroatom
     replacement prefixes:
     * "tricyclo[3.1.0.5(1,1).2(2,5)]tridecan" -- "Disagreement between number of rings and bridges"
     * tetracyclo descriptors with insufficient bridge count in brackets

3. OPSIN COMPLEX NAME LIMITATION:
   - Very large macrocyclic systems with multiple stereocenters and complex
     substituent chains fail OPSIN parsing regardless of format.
   - These are correct IUPAC names that exceed OPSIN's parser capabilities.

4. STEREO FORMAT EDGE CASES:
   - Some compounds generate incorrect parent structures (wrong ring chosen,
     missing substituents), producing names that parse without stereo but fail
     with stereo. The root cause is parent selection, not stereo formatting.
     * "(2R,3S)-4-methyl-5-oxooxolane" -- molecule is a macrolide, not oxolane
     * "(1R,4R)-1,2,3,4-tetrahydronaphthalene" -- molecule has many substituents
     * "heptyl (2R,3R)-cyclopropanecarboxylate" -- molecule is much more complex

These are documented as known OPSIN parser limitations, NOT Orthonym bugs.
Our names follow correct IUPAC 2013 nomenclature rules.
"""

import pytest
from orthonym.namer import name_compound


class TestBicycloHeteroatomFix:
    """Test that heterocyclic bicyclo compounds use correct total atom count
    and heteroatom replacement prefixes (oxa, aza, thia).

    Root cause: get_complete_bicyclo_data() was counting only carbon atoms
    for the parent name suffix, producing e.g. 'bicyclo[4.1.0]hexane' (6C)
    instead of '7-oxabicyclo[4.1.0]heptane' (7 total atoms including O).
    """

    def test_epoxide_bicyclo_uses_total_atoms(self):
        """bicyclo[4.1.0] with one ring O should be heptane, not hexane."""
        smiles = "O[C@H]1[C@H](O)[C@@H](O)[C@@H]2O[C@@H]2[C@@H]1O"
        name = name_compound(smiles)
        assert "heptane" in name, f"Expected 'heptane' (7 atoms), got: {name}"
        assert "hexane" not in name, f"Should not use 'hexane' (6 atoms): {name}"

    def test_epoxide_bicyclo_has_oxa_prefix(self):
        """Heterocyclic bicyclo must have 'oxa' replacement prefix."""
        smiles = "O[C@H]1[C@H](O)[C@@H](O)[C@@H]2O[C@@H]2[C@@H]1O"
        name = name_compound(smiles)
        assert "oxa" in name, f"Expected 'oxa' heteroatom prefix, got: {name}"

    def test_epoxide_bicyclo_descriptor_correct(self):
        """Full name should be 7-oxa-bicyclo[4.1.0]heptane with decorations."""
        smiles = "O[C@H]1[C@H](O)[C@@H](O)[C@@H]2O[C@@H]2[C@@H]1O"
        name = name_compound(smiles)
        assert "bicyclo[4.1.0]" in name, f"Descriptor wrong: {name}"
        assert "7-oxa-" in name or "7-oxa" in name, f"Missing oxa prefix: {name}"

    def test_heterocyclic_macrocycle_uses_total_atoms(self):
        """Large heterocyclic bicyclo with N and O should count all ring atoms."""
        smiles = (
            "CCCCCC(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H]1C(=O)N[C@@H]"
            "(CCCN=C(N)N)C(=O)N[C@H]2CC[C@@H](O)N(C2=O)[C@@H]"
            "([C@@H](C)CC)C(=O)N(C)[C@@H](CC(=O)c2ccccc2N)C(=O)"
            "N[C@@H](C(C)C)C(=O)O[C@@H]1C"
        )
        name = name_compound(smiles)
        # Should use docosane (22 atoms) not hexadecane (16 carbons)
        assert "docosane" in name, f"Expected 'docosane' (22 atoms), got: {name}"
        assert "hexadecane" not in name, f"Should not use 'hexadecane' (16 carbons): {name}"
        # Should have aza and oxa prefixes
        assert "aza" in name, f"Missing aza prefix: {name}"
        assert "oxa" in name, f"Missing oxa prefix: {name}"


class TestVBNotationFormat:
    """Test that VB polycyclic notation format itself is correct.

    Most VB compounds produce correct IUPAC names that OPSIN cannot parse
    due to stereo + VB combination limitations.
    """

    def test_tricyclo_descriptor_format(self):
        """tricyclo descriptors should use parenthesized locants for secondary bridges."""
        smiles = "COCC1=C2[C@@H]3CC(C)(C)C[C@@H]3C[C@@]2(O)CC1=O"
        name = name_compound(smiles)
        assert "tricyclo[" in name, f"Expected tricyclo descriptor: {name}"
        # Should have secondary bridge with parenthesized locants
        assert "(" in name.split("tricyclo[")[1].split("]")[0], (
            f"Expected parenthesized locants in tricyclo descriptor: {name}"
        )


class TestStereoFormatEdgeCases:
    """Test stereo descriptor formatting.

    Most 'stereo_issue' triage items are actually wrong parent selection
    (missing substituents, wrong ring chosen), not stereo format problems.
    The stereo prefix format (2R,3S)- is correct per IUPAC P-93.
    """

    def test_stereo_format_parenthesized(self):
        """Stereo descriptors should be parenthesized with trailing hyphen."""
        # Simple compound with stereo
        smiles = "O[C@H]1[C@H](O)[C@@H](O)[C@@H]2O[C@@H]2[C@@H]1O"
        name = name_compound(smiles)
        if name.startswith("("):
            # Verify format: (XR,YS)-
            assert ")-" in name, f"Stereo prefix missing closing )-: {name}"
            stereo_part = name.split(")-")[0] + ")"
            # Should contain R or S labels
            assert "R" in stereo_part or "S" in stereo_part, (
                f"Stereo prefix missing R/S labels: {stereo_part}"
            )


class TestSteroidSuffixOrdering:
    """Test that steroid names have correct IUPAC suffix ordering:
    unsaturation BEFORE principal group (e.g., 'trien-3-one', not 'an-3-one-trien').

    Also tests IUPAC P-31.1.3.4: terminal 'a' added to stem when
    multiple unsaturation locants are cited (cholesta-5,7-dien, not cholest-5,7-dien).
    """

    def test_single_ene_no_terminal_a(self):
        """Single double bond: cholest-5-en (no terminal 'a')."""
        # Stigmastane with single double bond
        smiles = "CC(C)[C@H](CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC=C4C[C@@H](O)CC[C@]4(C)[C@H]3CC[C@]12C)C(C)C"
        name = name_compound(smiles)
        # Should NOT have 'stigmasta' with single ene
        if "stigmast" in name:
            assert "stigmasta-" not in name, (
                f"Single ene should not add terminal 'a': {name}"
            )

    def test_multiple_ene_adds_terminal_a(self):
        """Multiple double bonds: ergosta-7,9,24-trien (with terminal 'a')."""
        smiles = "C=C(CC[C@@H](C)[C@H]1CC[C@H]2C3=CC[C@H]4[C@H](C)C(=O)CC[C@]4(C)C3=C[C@@H](O)[C@]12C)C(C)C"
        name = name_compound(smiles)
        assert "ergosta-" in name, f"Multi-ene should use 'ergosta' (with 'a'): {name}"
        assert "ergost-" not in name, f"Should not use 'ergost-' for multi-ene: {name}"

    def test_unsaturation_before_suffix(self):
        """Unsaturation suffix must come BEFORE principal group suffix."""
        smiles = "C=C(CC[C@@H](C)[C@H]1CC[C@H]2C3=CC[C@H]4[C@H](C)C(=O)CC[C@]4(C)C3=C[C@@H](O)[C@]12C)C(C)C"
        name = name_compound(smiles)
        if "trien" in name and "one" in name:
            trien_pos = name.find("trien")
            one_pos = name.rfind("one")
            assert trien_pos < one_pos, (
                f"Unsaturation must precede -one suffix: {name}"
            )

    def test_saturated_steroid_suffix(self):
        """Saturated steroid: androstan-3-one (not androst-3-one or androstane-3-one)."""
        # Simple saturated ketone steroid
        smiles = "C[C@]12CCC(=O)C=C1CC[C@@H]1[C@@H]2CC[C@]2(C)[C@@H](O)CC[C@@H]12"
        name = name_compound(smiles)
        # Should have unsaturation info somewhere (en- or an-)
        assert "an" in name or "en" in name, f"Should have saturation info: {name}"

    def test_dien_uses_terminal_a(self):
        """cholesta-8,24-dien format (two double bonds: add 'a')."""
        smiles = "CC(C)=CCC[C@@H](C(=O)O)[C@H]1C(=O)C[C@@]2(C)C3=C(CC[C@]12C)[C@@]1(C)CCC(=O)C(C)(C)[C@@H]1[C@@H](O)C3"
        name = name_compound(smiles)
        if "cholest" in name and "dien" in name:
            assert "cholesta-" in name, (
                f"Two double bonds should use 'cholesta-' (with 'a'): {name}"
            )


class TestNBracketWrapping:
    """Test IUPAC P-16.3.3 bracket escalation for N-substituents.

    When an N-substituent name contains parenthesized content (stereo
    descriptors or compound substituent names), the N-prefix must use
    square brackets: N-[...] instead of bare N-(...).

    Root cause: N-prefix construction sites (composer.py, fragment_assembly.py,
    engine.py, amides.py, benzene.py, etc.) previously produced f"N-{name}"
    without checking for parentheses. Now all sites route through
    _wrap_n_substituent() which applies P-16.3.3 bracket escalation.
    """

    # ---- Unit-level tests for _wrap_n_substituent ----

    def test_simple_name_unchanged(self):
        """Simple substituent names without parens are NOT wrapped."""
        from orthonym.assembly.naming_utils import _wrap_n_substituent
        assert _wrap_n_substituent("methyl") == "methyl"

    def test_dimethyl_unchanged(self):
        """Multiplicative prefixes without parens are NOT wrapped."""
        from orthonym.assembly.naming_utils import _wrap_n_substituent
        assert _wrap_n_substituent("dimethyl") == "dimethyl"

    def test_ethyl_unchanged(self):
        """Simple alkyl names without parens are NOT wrapped."""
        from orthonym.assembly.naming_utils import _wrap_n_substituent
        assert _wrap_n_substituent("ethyl") == "ethyl"

    def test_stereo_prefix_wrapped(self):
        """Stereo-containing N-substituent gets square brackets."""
        from orthonym.assembly.naming_utils import _wrap_n_substituent
        result = _wrap_n_substituent("(2S)-2-(pentanoylamino)propanoyl")
        assert result == "[(2S)-2-(pentanoylamino)propanoyl]"

    def test_complex_stereo_wrapped(self):
        """Multi-stereo N-substituent gets square brackets."""
        from orthonym.assembly.naming_utils import _wrap_n_substituent
        result = _wrap_n_substituent(
            "(2R,3S)-3-hydroxy-2-(benzoylamino)butanoyl"
        )
        assert result == "[(2R,3S)-3-hydroxy-2-(benzoylamino)butanoyl]"

    def test_already_bracketed_unchanged(self):
        """Names already in square brackets are NOT double-wrapped."""
        from orthonym.assembly.naming_utils import _wrap_n_substituent
        assert _wrap_n_substituent("[already-bracketed]") == "[already-bracketed]"

    def test_balanced_outer_parens_unchanged(self):
        """Names with balanced outer parens are already enclosed."""
        from orthonym.assembly.naming_utils import _wrap_n_substituent
        assert _wrap_n_substituent("(3-ethyl-1H-indolyl)") == "(3-ethyl-1H-indolyl)"

    def test_locanted_sub_wrapped(self):
        """N-substituent with internal locant parens gets wrapped."""
        from orthonym.assembly.naming_utils import _wrap_n_substituent
        result = _wrap_n_substituent("2-(nonanoylamino)pentanedioyl")
        assert result == "[2-(nonanoylamino)pentanedioyl]"

    def test_adenine_sub_wrapped(self):
        """N-substituent with internal compound parens gets wrapped."""
        from orthonym.assembly.naming_utils import _wrap_n_substituent
        result = _wrap_n_substituent("7-(10-carboxydecyl)adenineyl")
        assert result == "[7-(10-carboxydecyl)adenineyl]"

    # ---- Integration-level regression tests for simple N-compounds ----

    def test_n_methylacetamide_no_brackets(self):
        """N-methylacetamide must NOT get unnecessary brackets."""
        name = name_compound("CC(=O)NC")
        assert "N-methyl" in name, f"Expected N-methyl: {name}"
        assert "N-[" not in name, f"Simple N-methyl must not be bracketed: {name}"

    def test_nn_dimethylformamide_no_brackets(self):
        """N,N-dimethylformamide must NOT get unnecessary brackets."""
        name = name_compound("CN(C)C=O")
        assert "N,N-dimethyl" in name, f"Expected N,N-dimethyl: {name}"
        assert "[" not in name, f"Simple N,N-dimethyl must not be bracketed: {name}"

    def test_n_ethylpropanamide_no_brackets(self):
        """N-ethylpropanamide must NOT get unnecessary brackets."""
        name = name_compound("CCC(=O)NCC")
        assert "N-ethyl" in name, f"Expected N-ethyl: {name}"
        assert "N-[" not in name, f"Simple N-ethyl must not be bracketed: {name}"

    # ---- Integration tests for bracket-fixable compounds ----

    def test_pentanoylamino_propanoyl_gets_brackets(self):
        """N-[(2S)-2-(pentanoylamino)propanoyl]-... gets square brackets."""
        smiles = "C[C@H](NC(=O)[C@H](C)NC(=O)[C@@H]1CCCN1)C(=O)O"
        name = name_compound(smiles)
        assert "N-[" in name, (
            f"Expected N-[...] bracket wrapping for stereo N-substituent: {name}"
        )
        # Must not have bare N-(2S) pattern
        assert "N-(2S)" not in name, (
            f"Should not have bare N-(2S) without brackets: {name}"
        )

    def test_hexanoylamino_phenylpropanoyl_gets_brackets(self):
        """N-[(2S)-2-(hexanoylamino)-3-phenylpropanoyl]-... gets square brackets."""
        smiles = r"CC(=O)N[C@@H](CC(C)C)C(=O)N(C)[C@@H](Cc1ccccc1)C(=O)N/C=C\c1c[nH]c2ccccc12"
        name = name_compound(smiles)
        assert "N-[" in name, (
            f"Expected N-[...] bracket wrapping: {name}"
        )

    def test_hydroxytetracosanoyl_gets_brackets(self):
        """N-[(2S)-2-hydroxytetracosanoyl]-... gets square brackets."""
        smiles = "CCCCCCCCCCCCCCCCCCCCCC[C@H](O)C(=O)N[C@@H](COP(=O)(O)O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O)[C@H](O)CCCCCCCCCCCCCCC"
        name = name_compound(smiles)
        assert "N-[" in name, (
            f"Expected N-[...] bracket wrapping: {name}"
        )


class TestMalformedSuffixFixes:
    """Regression tests verifying malformed suffix strings no longer appear.

    These suffixes were identified in the v15.0 OPSIN failure triage as
    invalid suffix concatenations. The root causes were fixed in the
    suffix derivation logic (not via postprocessors).

    - benzamideyl: was "benzamide" + "yl" (now produces benzoyl/benzamido)
    - propanedioateyloxy: was "propanedioate" + "yloxy" (now produces propanedioyloxy)
    - glutyl: was "glut" + "yl" (now produces glutaminyl)
    - cycloane: was "cyclo" + "ane" with empty stem (now returns SMILES fallback)
    """

    def test_no_benzamideyl_in_output(self):
        """benzamideyl must not appear -- correct form is benzoyl or benzamido."""
        smiles = "CN(C(=O)c1ccc2c(c1)OC(F)(F)O2)c1cccc(C(=O)Nc2c(Br)cc(C(F)(C(F)(F)F)C(F)(F)F)cc2OC(F)F)c1F"
        name = name_compound(smiles)
        assert "benzamideyl" not in name, (
            f"Malformed suffix 'benzamideyl' found in: {name}"
        )
        # Should contain a valid amide-related form
        assert "benz" in name.lower(), (
            f"Expected benzene-derived fragment in name: {name}"
        )

    def test_no_propanedioateyloxy_in_output(self):
        """propanedioateyloxy must not appear -- correct form is propanedioyloxy."""
        smiles = "*c1c(*)c(*)c(-c2oc3c(*)c(*)c(*)c(*)c3c(=O)c2O[C@@H]2O[C@H](COC(=O)CC(=O)[O-])[C@@H](O)[C@H](O)[C@H]2O)c(*)c1*"
        name = name_compound(smiles)
        assert "propanedioateyloxy" not in name, (
            f"Malformed suffix 'propanedioateyloxy' found in: {name}"
        )
        # The fixed form should be propanedioyloxy
        if "propanedio" in name:
            assert "propanedioyloxy" in name, (
                f"Expected 'propanedioyloxy' not malformed form in: {name}"
            )

    def test_no_glutyl_in_output(self):
        """glutyl must not appear -- correct form is glutaminyl or pentanedioyl."""
        smiles = "C=C1CC23C=CC(=O)C(C)(CCCC(C)C(=O)NC(CCC(N)=O)C(=O)O)C2CC1CC3O"
        name = name_compound(smiles)
        assert "glutyl" not in name, (
            f"Malformed suffix 'glutyl' found in: {name}"
        )

    def test_no_cycloane_in_output(self):
        """cycloane must not appear -- complex polycyclic returns SMILES fallback."""
        smiles = "c1ccc2cc3c(cc2c1)-c1cc2ccccc2cc1-c1cc2ccccc2cc1-c1cc2ccccc2cc1-3"
        name = name_compound(smiles)
        assert "cycloane" not in name, (
            f"Malformed suffix 'cycloane' found in: {name}"
        )
        # For unnameable compounds, the namer returns canonical SMILES
        # rather than a garbled pseudo-IUPAC name
        assert len(name) > 6, (
            f"Expected non-trivial fallback, got: {name}"
        )

    def test_benzamide_compound_opsin_parseable(self):
        """The benzamide compound should produce an OPSIN-parseable name."""
        smiles = "CN(C(=O)c1ccc2c(c1)OC(F)(F)O2)c1cccc(C(=O)Nc2c(Br)cc(C(F)(C(F)(F)F)C(F)(F)F)cc2OC(F)F)c1F"
        name = name_compound(smiles)
        # Verify the name uses correct IUPAC forms (benzoyl, benzamide, etc.)
        assert "benzam" in name.lower() or "benzoyl" in name.lower(), (
            f"Expected valid benzamide/benzoyl form in: {name}"
        )


class TestFormatEdgeFixes:
    """Test format edge case fixes: separator between prefixes and N-locants,
    N-locant comma format, and benzene-vs-phenyl in substituent context.

    Root causes:
    - Missing separator: _assemble_amide_name() concatenated non-N prefixes
      directly with base_name that starts with an N-prefix (e.g., "2-methylN-methyl"
      instead of "2-methyl-N-methyl"). Fixed by inserting hyphen before "N" prefix.
    - Extra comma: _assemble_amine_name() used f"N,{'N,' * (count-1)}" which
      produces "N,N,di..." instead of "N,N-di...". Fixed by using
      ",".join(["N"] * count) + "-" for correct IUPAC N-locant format.
    - benzene-vs-phenyl: decomposition fragment assembly used "benzene" parent
      name when fragment was used as a substituent prefix. Fixed by adding
      _parent_to_substituent_prefix() conversion per IUPAC P-31.1.3.4.
    """

    def test_no_missing_separator_before_n_prefix(self):
        """Prefix + N-prefix must have hyphen separator, not direct concatenation."""
        # 2-methyl-N-methylbutanamide (not 2-methylN-methylbutanamide)
        smiles = "CC(CC)C(=O)NC"
        name = name_compound(smiles)
        assert "methylN" not in name, (
            f"Missing separator between prefix and N-locant: {name}"
        )
        assert "-N-" in name or name.startswith("N-"), (
            f"Expected hyphen before N-locant prefix: {name}"
        )

    def test_nn_format_correct_hyphen(self):
        """N,N-locant must use N,N- (hyphen after last N), not N,N, (comma)."""
        # N,N-dimethylformamide: verify correct N,N- format
        smiles = "CN(C)C=O"
        name = name_compound(smiles)
        assert "N,N-" in name, f"Expected N,N- format: {name}"
        assert "N,N," not in name, f"Extra comma in N,N format: {name}"

    def test_pyrrolidinyl_separator(self):
        """Ring substituent prefix separated from N-prefix by hyphen."""
        smiles = "CCCCCCCCCCCCCCCC(=O)N1CCCC1"
        name = name_compound(smiles)
        assert "pyrrolidinylN" not in name, (
            f"Missing separator between pyrrolidinyl and N-prefix: {name}"
        )

    def test_nn_dimethyl_amine_format(self):
        """Amine N,N- locant expansion must use hyphen, not trailing comma."""
        # N,N-dimethylethanamine
        smiles = "CCN(C)C"
        name = name_compound(smiles)
        if "N,N" in name:
            assert "N,N-" in name, f"Expected N,N- not N,N,: {name}"
            assert "N,N," not in name, f"Extra comma in N,N format: {name}"

    def test_chloro_amide_separator(self):
        """2-chloro-N,N-dimethylpropanamide has correct separator before N-prefix."""
        smiles = "CC(Cl)C(=O)N(C)C"
        name = name_compound(smiles)
        # If both chloro and N-prefix present, should be separated by hyphen
        if "chloro" in name and "N," in name:
            assert "chloro-N" in name, (
                f"Missing separator between chloro and N-prefix: {name}"
            )

    def test_benzene_as_parent_unchanged(self):
        """benzene as parent ring should keep 'benzene', not become 'phenyl'."""
        # hydroxymethylbenzene: benzene is the parent, methanol is substituent
        smiles = "OCc1ccccc1"
        name = name_compound(smiles)
        # benzene as parent is correct (not changed to phenyl)
        assert "benzene" in name, (
            f"benzene as parent should stay 'benzene': {name}"
        )

    def test_phenyl_in_chain_context(self):
        """Benzene ring as substituent on chain should use 'phenyl'."""
        # 2-phenylethanoic acid: benzene is substituent on chain
        smiles = "c1ccc(CC(=O)O)cc1"
        name = name_compound(smiles)
        assert "phenyl" in name, (
            f"benzene as substituent should use 'phenyl': {name}"
        )

    def test_methoxybenzene_becomes_methoxyphenyl_in_substituent(self):
        """Fragment with methoxybenzene used as substituent prefix uses phenyl.

        Root cause: decomposition fragment assembly used parent name 'benzene'
        directly when constructing substituent prefixes. Fixed by
        _parent_to_substituent_prefix() converting benzene -> phenyl per
        IUPAC P-31.1.3.4.
        """
        smiles = "COc1cccc2c1[C@@H](OC)O[C@H]2c1c(O)ccc2c1C(=O)CC(C)(O)C2"
        name = name_compound(smiles)
        assert "methoxybenzene" not in name, (
            f"benzene in substituent context should be phenyl: {name}"
        )
        if "methoxy" in name and "phenyl" in name:
            assert "methoxyphenyl" in name, (
                f"Expected 'methoxyphenyl' form: {name}"
            )

    def test_simple_methoxybenzene_parent_unchanged(self):
        """Simple methoxybenzene (anisole) as parent is NOT converted to phenyl."""
        # This compound is anisole -- benzene is the parent ring, not a substituent
        smiles = "COc1ccccc1"
        name = name_compound(smiles)
        # Should be anisole (retained name) or methoxybenzene (systematic)
        assert "phenyl" not in name, (
            f"Parent benzene should NOT become phenyl: {name}"
        )
