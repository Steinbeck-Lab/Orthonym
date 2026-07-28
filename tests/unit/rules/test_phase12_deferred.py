"""v23 Phase 12 follow-on — the three deferred items resolved (F3 / F2 / F1).

  - F3 ureido : -NH-C(=O)-NH2 substituent was double-prefixed
                (carbamoylamino + phantom methanoylamino). Now a single
                (carbamoylamino). Fixes every ureido-acid, not just citrulline.
  - F2 cystine: both enantiomers get their P-103.1.3.1 descriptor
                (L-cystine / D-cystine); meso defers. NOTE the original line here
                read "L-cystine -> 'cystine' (was always fine)" -- it was not
                fine, it was a stereo loss, corrected in v29 P3-CLEANUP Item 1.
  - F1 inositol: the 7 meso inositols get their P-104.2.1 retained PIN
                (name-exact, OPSIN-unparseable). The chiral D/L-chiro pair is
                DETERMINISTICALLY refused (RDKit perceives it non-deterministically).
"""

import pytest

from rdkit import Chem

from orthonym.namer import name_compound


class TestUreido:
    """F3: ureido substituent must be a single (carbamoylamino), not split."""

    @pytest.mark.parametrize("smiles,expected", [
        ("NC(=O)NCCC[C@H](N)C(=O)O", "(2S)-2-amino-5-(carbamoylamino)pentanoic acid"),  # citrulline
        ("NC(=O)NCCC(=O)O", "3-(carbamoylamino)propanoic acid"),
        ("NC(=O)NCCCC(=O)O", "4-(carbamoylamino)butanoic acid"),
    ])
    def test_ureido_single_prefix(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        ("O=CNCCCC(=O)O", "4-formamidobutanoic acid"),    # real formamido — P-66.1.1.4.3(1)
        ("CC(=O)NCCC(=O)O", "3-acetamidopropanoic acid"),  # real acetamido — P-66.1.1.4.3(1)
        ("NCCC(=O)O", "3-aminopropanoic acid"),            # plain amino — unaffected
        ("CNC(N)=O", "N-methylurea"),                      # terminal urea — unaffected
    ])
    def test_acylamino_regressions(self, smiles, expected):
        assert name_compound(smiles) == expected


class TestCystine:
    """F2: cystine disulfide-dimer amino acid."""

    @pytest.mark.parametrize("smiles,expected", [
        ("N[C@@H](CSSC[C@H](N)C(=O)O)C(=O)O", "L-cystine"),   # L-cystine (2R,2'R)
        ("N[C@H](CSSC[C@@H](N)C(=O)O)C(=O)O", "D-cystine"),   # D-cystine (2S,2'S)
    ])
    def test_cystine(self, smiles, expected):
        """v29 P3-CLEANUP Item 1: the L form used to ship BARE while the D form
        kept its descriptor. `## **P-103.1.3.1** The stereodescriptors 'D' and 'L'`
        (BlueBookV2.md:54291) applies to "*the alpha-amino carboxylic acids*" and
        at :54301 names cystine explicitly; cystine is Table 10.5 / P-103.1.1.2."""
        assert name_compound(smiles) == expected

    def test_L_and_D_are_symmetric(self):
        """The two enantiomers must be spelled the same way modulo the descriptor.

        This is the property the old table violated: `D-cystine` was written into
        the table VALUE while the L form resolved to a bare OPSIN-import name, so
        no test could catch the pair drifting apart. Asserting the SHARED stem
        rather than two literals means a future one-sided edit fails here."""
        l_name = name_compound("N[C@@H](CSSC[C@H](N)C(=O)O)C(=O)O")
        d_name = name_compound("N[C@H](CSSC[C@@H](N)C(=O)O)C(=O)O")
        assert l_name[:2] == "L-" and d_name[:2] == "D-", (l_name, d_name)
        assert l_name[2:] == d_name[2:] == "cystine", (l_name, d_name)

    def test_table_values_are_bare_for_both_enantiomers(self):
        """The STRUCTURAL half of symmetry — and the one that actually bites.

        Asserting only the emitted strings does not catch a regression to the old
        design, because writing `D-cystine` back into the table VALUE emits the
        same `D-cystine`: the descriptor path simply never runs for it (the name
        `'D-cystine'` is not in `_DL_CAPABLE_NONSTANDARD`, so the `is_standard`
        gate returns it verbatim). Mutation-verified: that change leaves every
        behavioural assertion in this class green. The defect it re-introduces is
        the MECHANISM asymmetry -- one enantiomer bypassing the shared descriptor
        path -- which is exactly how the pair drifted apart the first time. So the
        table itself is pinned: both keys must map to the BARE retained name."""
        from orthonym.data.amino_acids import NON_STANDARD_AMINO_ACIDS
        for smiles in ("N[C@@H](CSSC[C@H](N)C(=O)O)C(=O)O",     # (R,R)
                       "N[C@H](CSSC[C@@H](N)C(=O)O)C(=O)O"):    # (S,S)
            key = Chem.CanonSmiles(smiles)
            assert NON_STANDARD_AMINO_ACIDS[key] == "cystine", (
                f"{key} maps to {NON_STANDARD_AMINO_ACIDS[key]!r}; a descriptor "
                f"baked into the table VALUE bypasses the shared descriptor path"
            )

    def test_meso_cystine_has_no_retained_form(self):
        """meso-cystine (2R,2'S) must DEFER, not borrow a descriptor.

        P-103.1.3.1 (:54305) reserves 'DL' for "*a mixture of equimolar amounts of
        'D' and 'L' compounds*" -- a racemate. meso-cystine is a single achiral
        compound, so no D/L/DL retained form applies and the retained lookup must
        return None (the systematic namer then owns it).

        Asserted at the DATA layer on purpose: `name_compound` here runs with the
        OPSIN validity gate DISABLED (conftest autouse fixture), so it would show
        the raw systematic name and prove nothing about the retained-name refusal.
        This is the claim `amino_acids.py` makes in its own comment, now tested."""
        from orthonym.data.amino_acids import (
            get_amino_acid_name, _dimeric_aa_forms,
        )
        meso = Chem.CanonSmiles("N[C@@H](CSSC[C@@H](N)C(=O)O)C(=O)O")  # (2R,2'S)
        m = Chem.MolFromSmiles(meso)
        assert get_amino_acid_name(meso, mol=m, with_descriptor=True) is None
        assert get_amino_acid_name(meso, mol=m, with_descriptor=False) is None

        # Defence in depth. The assertions above are carried by meso simply not
        # being a table key, so they would survive a descriptor map that DID list
        # meso -- the mutation that matters. Pin the map directly: exactly the two
        # enantiomers, and meso absent, so the refusal holds even if the retained
        # lookup later learns the meso structure for some other reason.
        forms = _dimeric_aa_forms()["cystine"]
        assert meso not in forms
        assert sorted(forms.values()) == ["D-", "L-"]


class TestDopa:
    """v29 P3-CLEANUP Item 1: the OTHER Table 10.5 member losing its descriptor.

    `dopa` is `P-103.1.1.2` Table 10.5 (BlueBookV2.md:54241, systematic
    `3-hydroxytyrosine`), so `## **P-103.1.3.1** The stereodescriptors 'D' and 'L'`
    (:54291) designates its alpha-carbon exactly as it does cystine's. The OPSIN
    import supplied only the L key, so L shipped BARE and D had no retained name
    at all."""

    @pytest.mark.parametrize("smiles,expected", [
        ("N[C@@H](Cc1ccc(O)c(O)c1)C(=O)O", "L-dopa"),   # (2S)
        ("N[C@H](Cc1ccc(O)c(O)c1)C(=O)O", "D-dopa"),    # (2R)
    ])
    def test_dopa_carries_its_descriptor(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_unspecified_alpha_carbon_does_not_get_a_descriptor(self):
        """The mirror of the loss: a configuration that is NOT specified must not
        acquire one. An invented `L-` here would be worse than the bare name."""
        n = name_compound("NC(Cc1ccc(O)c(O)c1)C(=O)O")
        assert not n.startswith(("L-", "D-")), n
        assert n == "2-amino-3-(3,4-dihydroxyphenyl)propanoic acid"


class TestNonBlueBookAminoAcidsStayBare:
    """The boundary of the class fix — this is what keeps it from being a
    fabrication engine.

    Membership of `_DL_CAPABLE_NONSTANDARD` requires a Blue Book Table 10.4/10.5
    entry. The merged table also holds ~47 other stereo-specified OPSIN-vocabulary
    names with NO Blue Book occurrence at all; prepending `L-` to those would
    invent a name that OPSIN cannot parse, and the validity gate would then
    suppress the molecule to `unknown organic compound` -- trading a spelling for
    a coverage loss (invariant 11)."""

    def test_registry_is_bluebook_attested_only(self):
        from orthonym.data.amino_acids import _DL_CAPABLE_NONSTANDARD
        assert _DL_CAPABLE_NONSTANDARD == {"cystine", "dopa"}

    @pytest.mark.parametrize("name", [
        "statine", "homocystine", "selenocystine", "tellurocystine", "carnitine",
    ])
    def test_bluebook_absent_names_are_not_in_the_registry(self, name):
        """Tripwire: adding one of these needs a Blue Book citation first."""
        from orthonym.data.amino_acids import _DL_CAPABLE_NONSTANDARD
        assert name not in _DL_CAPABLE_NONSTANDARD

    def test_a_non_bluebook_amino_acid_keeps_its_bare_name(self):
        """Behavioural half: the gate still silences the descriptor off-registry."""
        from orthonym.data.amino_acids import (
            get_amino_acid_name, NON_STANDARD_AMINO_ACIDS,
        )
        smi = next((s for s, n in NON_STANDARD_AMINO_ACIDS.items()
                    if n == "selenocystine"), None)
        assert smi is not None, "fixture drifted: 'selenocystine' left the table"
        m = Chem.MolFromSmiles(smi)
        assert get_amino_acid_name(smi, mol=m, with_descriptor=True) == "selenocystine"


class TestInositol:
    """F1: the seven meso inositols (P-104.2.1); chiro deterministically refused."""

    # InChIKey -> (representative SMILES, retained name) for the 7 meso inositols.
    MESO = {
        "myo-inositol":    "O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O",
        "scyllo-inositol": "O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
        "muco-inositol":   "O[C@H]1[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)[C@H]1O",
        "epi-inositol":    "O[C@H]1[C@@H](O)[C@@H](O)[C@@H](O)[C@@H](O)[C@@H]1O",
        "allo-inositol":   "O[C@H]1[C@H](O)[C@H](O)[C@@H](O)[C@@H](O)[C@H]1O",
        "cis-inositol":    "O[C@H]1[C@@H](O)[C@@H](O)[C@@H](O)[C@@H](O)[C@H]1O",
        "neo-inositol":    "O[C@H]1[C@H](O)[C@H](O)[C@H](O)[C@@H](O)[C@H]1O",
    }

    @pytest.mark.parametrize("name,smiles", list(MESO.items()))
    def test_meso_inositol(self, name, smiles):
        assert name_compound(smiles) == name

    @pytest.mark.parametrize("name,smiles", list(MESO.items()))
    def test_meso_inositol_deterministic(self, name, smiles):
        m = Chem.MolFromSmiles(smiles)
        out = {name_compound(Chem.MolToSmiles(m, doRandom=True)) for _ in range(8)}
        assert out == {name}, f"{name} non-deterministic: {out}"

    @pytest.mark.parametrize("smiles", [
        "O[C@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1O",   # chiro (…-LKPKBOIGSA-N)
        "O[C@H]1[C@H](O)[C@@H](O)[C@H](O)[C@H](O)[C@H]1O",   # chiro enantiomer (…-SHFUYGGZSA-N)
    ])
    def test_chiro_deterministic_unknown(self, smiles):
        # The chiral chiro pair is refused to a DETERMINISTIC unknown (RDKit
        # perceives its absolute config non-deterministically).
        m = Chem.MolFromSmiles(smiles)
        out = {name_compound(Chem.MolToSmiles(m, doRandom=True)) for _ in range(10)}
        assert len(out) == 1, f"chiro not deterministic: {out}"
        assert out.pop().startswith("unknown")

    def test_undefined_stereo_hexol_keeps_systematic(self):
        # Fail-closed: an undefined-stereo cyclohexanehexol is NOT an inositol match.
        assert name_compound("OC1C(O)C(O)C(O)C(O)C1O") == "cyclohexane-1,2,3,4,5,6-hexol"
