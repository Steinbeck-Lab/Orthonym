""" mancude-ring exocyclic imine suffix (W2F p4).
BB (the Blue Book): 'naphthalen-2(1H)-imine (PIN)'. The
added-indicated-H engine is suffix-agnostic; a ring bare =NH takes '-imine' with
the identical (nH) numbering as '-one'. Imine is the LAST seniority class, so a
ring bearing BOTH =O and =NH keeps -one + imino prefix (already works at HEAD)."""
import orthonym


class TestRingImine:
    def test_naphthalen_2_1h_imine(self):
        # BB-verbatim PIN (the Blue Book)
        assert orthonym.name_compound("N=C1Cc2ccccc2C=C1", style="pin") == "naphthalen-2(1H)-imine"

    def test_indanimine(self):
        assert orthonym.name_compound("N=C1CCc2ccccc21", style="pin") == "2,3-dihydro-1H-inden-1-imine"

    def test_inden_1_imine(self):
        assert orthonym.name_compound("N=C1C=Cc2ccccc21", style="pin") == "1H-inden-1-imine"

    def test_mixed_oxo_imine_regression(self):
        # =O senior → -one suffix, =NH → imino prefix (ALREADY works at HEAD; must stay)
        assert orthonym.name_compound("O=C1C=Cc2ccccc2C1=N", style="pin") == "1-iminonaphthalen-2(1H)-one"

    def test_simple_ring_imine_regression(self):
        assert orthonym.name_compound("N=C1CCCCC1", style="pin") == "cyclohexan-1-imine"

    def test_n_substituted_ring_imine(self):
        # v52-p04 extended the imine suffix to N-substituted ring imines (=N-R):
        # a single imine N is cited with the bare italic 'N' locant;
        # cf. BB N-methylethanimine PIN at the Blue Book), on the
        # naphthalen-2(1H)-imine parent (BB PIN:26536). The bare-=NH detector
        # (degree-1 only) still declines the degree-2 N; the N-substituted branch
        # (include_nsub=True) handles it and renders the N-locant prefix.
        from rdkit import Chem
        from orthonym.rules.partial_saturation import _ring_imine_carbons
        mol = Chem.MolFromSmiles("CN=C1Cc2ccccc2C=C1")
        ring = {a for r in mol.GetRingInfo().AtomRings() for a in r}
        assert _ring_imine_carbons(mol, ring) == []   # bare detector declines (N degree 2)
        assert _ring_imine_carbons(mol, ring, include_nsub=True) != []  # nsub branch accepts
        out = orthonym.name_compound("CN=C1Cc2ccccc2C=C1", style="pin")
        assert out == "N-methylnaphthalen-2(1H)-imine"

    def test_n_substituted_ring_diimine(self):
        # BB-verbatim PIN (the Blue Book): the N-locant carries a number
        # to distinguish the two imine nitrogens.
        assert (orthonym.name_compound("CN=C1C=CC(=NC)c2ccccc21", style="pin")
                == "N1,N4-dimethylnaphthalene-1,4-diimine")

    # --- Phase-4 a review defects D1/D2/D3 (fail-closed / spelling) ---------------

    def test_d3_hyphen_before_digit_initial_parent(self):
        # D3 (a), the Blue Book): an N-substituent prefix that
        # precedes a DIGIT-initial parent ('1H-inden-1-imine') needs the hyphen
        # separator. At HEAD the render glued them: 'N-methyl1H-inden-1-imine'.
        assert (orthonym.name_compound("CN=C1C=Cc2ccccc21", style="pin")
                == "N-methyl-1H-inden-1-imine")

    def test_d1_acyl_n_substituent_producer_fails_closed(self):
        # D1: amide class 11 is SENIOR to imine class 20;
        # the Blue Book): an ACYL N-substituent (=N-C(=O)R) is an
        # N-ylidene amide, not an N-imine prefix. The imine producer must decline
        # (None) so it never emits the false PIN 'N-acetyl1H-inden-1-imine'.
        # Producer-level (deterministic, no OPSIN-gate dependence): at HEAD it
        # returned the false PIN string; correct 'N-(1H-inden-1-ylidene)acetamide'
        # is a buildable follow-on.
        from rdkit import Chem
        from orthonym.rules.partial_saturation import name_cyclic_oxo_compound
        assert name_cyclic_oxo_compound(Chem.MolFromSmiles("CC(=O)N=C1C=Cc2ccccc21")) is None

    def test_d2_diacyl_n_substituent_producer_fails_closed(self):
        # D2: same class as D1, the diimine mirror ('N1,N4-diacetylnaphthalene-
        # 1,4-diimine' was the false PIN). The producer must decline (None).
        from rdkit import Chem
        from orthonym.rules.partial_saturation import name_cyclic_oxo_compound
        assert name_cyclic_oxo_compound(
            Chem.MolFromSmiles("CC(=O)N=C1C=CC(=NC(C)=O)c2ccccc21")) is None
