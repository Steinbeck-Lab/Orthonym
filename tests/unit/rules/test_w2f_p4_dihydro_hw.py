""" partial-saturation of non-aromatizable HW rings (W2F p4).
BB (the Blue Book): hydro-prefixed HW names are PINs for partially
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
        # A 7-membered ring is Hantzsch-Widman territory:
        # (the Blue Book) "Mancude and saturated heteromonocyclic
        # compounds with up to and including ten ring members are named by the
        # extended Hantzsch-Widman system". Partial saturation is expressed as
        # hydro prefixes on that mancude parent -- (:16906), whose
        # own PIN examples include 2,7-dihydro-1H-azepine (:16920) and
        # 4,5-dihydro-3H-azepine (:16888).
        #
        # This assertion previously demanded "1-azacyclohepta-2,4-diene",
        # justified in-comment as an "RT-verified replacement PIN". An OPSIN
        # round-trip proves a name denotes the right STRUCTURE; it never
        # decides which name is PREFERRED, and no Blue Book rule was cited for
        # sending an unsaturated 7-ring to replacement nomenclature.
        out = orthonym.name_compound("C1CC=CC=CN1", style="pin")
        assert out == "2,3-dihydro-1H-azepine"
