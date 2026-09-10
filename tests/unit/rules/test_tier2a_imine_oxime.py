"""Wave2 — imine broadening + oxime/isocyanato substitutive PINs.

Five units / (f) / /:

U1 methanimine: 'methyleneimine' (OPSIN parser alias) de-headlined via
   pin:false — the PIN is substitutive 'methanimine'.
U2 ethanimine: 'imine' joined _ETHANE_SUFFIX_ELIDE_FGS (BB VERBATIM
   'N-methylethanimine'); 3+C chains keep the locant ('propan-1-imine').
U3 N-substituted imines: broadened imine SMARTS ([CX3]=[NX2] with !R and
   oxime/hydrazone/oxime-ether/N-halo exclusions) + new imine handler
   citing the lone N-substituent as an italic-N prefix
   (composer._assemble_imine_name).
U4 oximes named substitutively as N-hydroxy imines (BB VERBATIM (f)
   + 'N-hydroxypropan-1-imine (PIN)'); bounded surgery builder, stereo /
   substituted forms keep the functional-class fallback.
U5 isocyanato/isothiocyanato substitutive parents (BB VERBATIM
   'isocyanatocyclohexane (PIN) cyclohexyl isocyanate'): handlers decline
   under PIN style; claimed-atom mask extended to the general-acyclic
   (guard 3b) and ring walkers; symmetric-carbocycle FG-prefix locant
   elision (isocyanatocyclohexane, not 1-isocyanatocyclohexane).
"""

import pytest

from orthonym import name_compound


@pytest.mark.unit
class TestImineParents:
    """U1/U2: bare imine PIN forms."""

    @pytest.mark.parametrize("smiles,expected", [
        ("C=N", "methanimine"),          # was 'methyleneimine'
        ("CC=N", "ethanimine"),          # was 'ethan-1-imine'
        ("CCC=N", "propan-1-imine"),     # 3+C keeps locant
        ("CC(C)=N", "propan-2-imine"),
        ("N=C1CCCCC1", "cyclohexan-1-imine"),
    ])
    def test_bare_imines(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestNSubstitutedImines:
    """U3: broadened SMARTS + italic-N prefix emission."""

    @pytest.mark.parametrize("smiles,expected", [
        ("CC=NC", "N-methylethanimine"),      # BB VERBATIM PIN
        ("C=NC", "N-methylmethanimine"),
        ("CC(C)=NCC", "N-ethylpropan-2-imine"),
        ("CC=Nc1ccccc1", "N-phenylethanimine"),
    ])
    def test_n_substituted(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # ring C=N is a skeletal feature, never an imine FG (!R guard)
        ("N1=CC1", "2H-azirine"),
        ("C1=CN1", "1H-azirine"),
        ("c1ccncc1", "pyridine"),
        # amines keep their own handler
        ("CNCC", "N-methylethanamine"),
        ("CN(C)CC", "N,N-dimethylethanamine"),
    ])
    def test_protected(self, smiles, expected):
        assert name_compound(smiles) == expected


@pytest.mark.unit
class TestSubstitutiveOximes:
    """U4: N-hydroxy imine PINs; bounded fallback."""

    @pytest.mark.parametrize("smiles,expected", [
        ("CCC=NO", "N-hydroxypropan-1-imine"),   # BB VERBATIM PIN
        ("CC=NO", "N-hydroxyethanimine"),
        ("CC(C)=NO", "N-hydroxypropan-2-imine"),
        ("CC(=NO)CC", "N-hydroxybutan-2-imine"),
        ("ON=C1CCCCC1", "N-hydroxycyclohexan-1-imine"),
    ])
    def test_substitutive_pins(self, smiles, expected):
        assert name_compound(smiles) == expected

    def test_stereo_oxime_keeps_fallback(self):
        # E/Z priorities differ between C=N-OH and the surgered C=N-H, so
        # the bounded builder declines; the functional-class name (RT-valid)
        # remains until the stereo-aware substitutive builder lands.
        assert name_compound("C/C=N/O") == "(E)-acetaldehyde oxime"

    def test_oxime_prefix_on_senior_parent_unchanged(self):
        assert name_compound("ON=CCCC(=O)O") == "4-hydroxyiminobutanoic acid"


@pytest.mark.unit
class TestSubstitutiveIsoCyanates:
    """U5: substitutive isocyanato parents + mask + ring locant elision."""

    @pytest.mark.parametrize("smiles,expected", [
        ("O=C=NCC", "isocyanatoethane"),
        ("O=C=NC", "isocyanatomethane"),
        ("O=C=NCCCC", "1-isocyanatobutane"),
        ("S=C=NCC", "isothiocyanatoethane"),
        ("O=C=NC1CCCCC1", "isocyanatocyclohexane"),   # BB VERBATIM PIN
        ("S=C=NC1CCCCC1", "isothiocyanatocyclohexane"),
        ("NC(=N)NC1CCCCC1", "N-cyclohexylguanidine"),  # ring-mask control
    ])
    def test_substitutive_parents(self, smiles, expected):
        assert name_compound(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # ring FG locant elision must not leak to multi-substituted rings
        ("FC1CCCCC1", "fluorocyclohexane"),
        ("FC1CCCCC1F", "1,2-difluorocyclohexane"),
        # senior-group chain path unchanged (2b territory)
        ("O=C=NCCCCCCCC(=O)O", "8-isocyanatooctanoic acid"),
    ])
    def test_protected(self, smiles, expected):
        assert name_compound(smiles) == expected
