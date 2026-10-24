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
   + 'N-hydroxypropan-1-imine (PIN)'); bounded surgery builder (the configured C=N,
   a second oxime group and the substituted forms followed: test_oxime_substitutive_pin).
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

    def test_stereo_oxime_has_the_substitutive_pin(self):
        # CHANGED from test_stereo_oxime_keeps_fallback, "(E)-acetaldehyde oxime" -> "(E)-N-hydroxyethanimine";
        # the old comment said the functional-class name "remains until the stereo-aware substitutive builder
        # lands": it has landed. Source: "Oximes" (the Blue Book): "In these recommendations
        # preferred IUPAC names for oximes are generated substitutively as N-hydroxy derivatives of imines
        # rather than by functional class nomenclature as in previous recommendations"; the descriptor of the
        # C=N stands at the front,:44643, a locant "when such locants are present": ethanimine cites
        # none). Independent check: OPSIN 2.9.0 reads this name to the isomeric SMILES of the input and the
        # name of the other configuration to the other one (test_oxime_substitutive_pin.py). Mutation: with the
        # builder as it was, the name is not built (the same file's gold row, STEREO-04, fails). Would make it
        # wrong: a Blue Book licence for the functional class name of a configured oxime (none::38462 is
        # unconditional).
        assert name_compound("C/C=N/O") == "(E)-N-hydroxyethanimine"

    def test_oxime_prefix_on_senior_parent_unchanged(self):
        assert name_compound("ON=CCCC(=O)O") == "4-(hydroxyimino)butanoic acid"


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
