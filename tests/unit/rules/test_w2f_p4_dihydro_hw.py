"""P-54.4.1 partial-saturation of non-aromatizable HW rings (W2F p4).
BB P-54.4.1 (BlueBookV2.md:24169): hydro-prefixed HW names are PINs for partially
unsaturated rings. Phosphete mancude = 2 double bonds, 0 indicated-H → the 1-ene
evidence is a DIhydro name, not '1H-phosphete'."""
import orthonym


class TestDihydroHW:
    def test_dihydrophosphete(self):
        assert orthonym.name_compound("C1=CCP1", style="pin") == "1,2-dihydrophosphete"

    def test_dihydroazete(self):
        assert orthonym.name_compound("C1=CCN1", style="pin") == "1,2-dihydroazete"

    def test_dihydrophosphete_isomer(self):
        assert orthonym.name_compound("C1=PCC1", style="pin") == "2,3-dihydrophosphete"

    def test_2h_thiete_regression(self):
        # divalent-S 4-ring: indicated-H path (not the new branch) — must stay
        assert orthonym.name_compound("C1=CCS1", style="pin") == "2H-thiete"

    def test_dihydrophosphole_regression(self):
        assert orthonym.name_compound("P1CCC=C1", style="pin") == "2,3-dihydro-1H-phosphole"

    def test_phosphetane_regression(self):
        assert orthonym.name_compound("C1CPC1", style="pin") == "phosphetane"

    def test_dihydroazepine_boundary(self):
        # 7-ring mancude needs 1 indicated-H → the new dihydro-HW branch (gate b)
        # DECLINES, so this ring is left to its pre-existing namer. At HEAD
        # 86fa454a that emits the RT-verified replacement PIN below (the plan's
        # None/unknown expectation was stale, from older HEAD 7d00a785). The
        # boundary that matters: the new branch must NOT emit a 'dihydroazepine'.
        out = orthonym.name_compound("C1CC=CC=CN1", style="pin")
        assert out == "1-azacyclohepta-2,4-diene"
        assert "dihydro" not in (out or "")
