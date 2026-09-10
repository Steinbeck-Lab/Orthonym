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

    def test_n_substituted_ring_imine_fails_closed(self):
        # N-methyl imine: N is degree 2, so the v1 bare-=NH detector (degree-1
        # only) declines and the new imine branch never fires. Under pytest the
        # OPSIN validity gate is disabled (conftest autouse), so the raw namer
        # emits its pre-existing best-guess with the imine dropped; production
        # suppresses that to 'unknown organic compound' (verified with
        # the gate ON). The p4 boundary that matters: no '-imine' is emitted.
        from rdkit import Chem
        from orthonym.rules.partial_saturation import _ring_imine_carbons
        mol = Chem.MolFromSmiles("CN=C1Cc2ccccc2C=C1")
        ring = {a for r in mol.GetRingInfo().AtomRings() for a in r}
        assert _ring_imine_carbons(mol, ring) == []   # detector declines (N degree 2)
        out = orthonym.name_compound("CN=C1Cc2ccccc2C=C1", style="pin")
        assert "imine" not in (out or "")
