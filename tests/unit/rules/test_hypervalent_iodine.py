"""Unit tests for hypervalent-iodine naming (λ3-iodane + the iodanium cation).

Blue Book authority (the Blue Book Blue Book):
  * "Nomenclature based on halogen parent hydrides" (the Blue Book):
      C6H5-I(OH)2 -> phenyl-λ3-iodanediol (PIN) (the Blue Book)
    The λ-convention is used for the nonstandard bonding number of iodine.
  * (the Blue Book): the mononuclear Group-17 cation is named on the parent
    hydride + '-ium' -> "diphenyliodanium (PIN)" (the Blue Book), the SYSTEMATIC PIN, NOT
    the Table-7.3 retained '-onium' spelling.
  *: a symmetric diol diester is the functional-class '<diol-diyl>
    di<acid>ate' form (ethane-1,2-diyl diacetate), so PhI(OAc)2 — the diacetate ester
    of phenyl-λ3-iodanediol — is 'phenyl-λ3-iodanediyl diacetate'.

Each target is pinned to its exact PIN string AND verified to OPSIN-round-trip to the
input structure (0-wrong: name -> OPSIN -> canonical InChIKey == input InChIKey).
"""
import pytest
from rdkit import Chem

from orthonym import Orthonym
from orthonym.validation.opsin_roundtrip import opsin_parse

_FLAGS = dict(general_fallback=True, general_fallback_unverified=True,
              allow_aromatic_general=True)

# One warm engine for the module (OPSIN JVM startup dominates).
_ENGINE = Orthonym(**_FLAGS)


def _name(smiles):
    return _ENGINE.name_tiered(smiles).get("name")


def _inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else None


# (input SMILES, expected PIN) — every row must ALSO OPSIN-round-trip (test below).
_TARGETS = [
    #: triaryl-λ3-iodane (pure-organyl hypervalent iodine hub).
    ("c1ccccc1I(c2ccccc2)c3ccccc3", "triphenyl-λ3-iodane"),
    # /: PhI(OAc)2 (PIDA), the diacetate ester of the iodanediol.
    ("CC(=O)OI(OC(C)=O)c1ccccc1", "phenyl-λ3-iodanediyl diacetate"),
    # (the Blue Book): the diaryliodanium CATION — 'iodanium', NOT 'iodonium'.
    ("c1ccccc1[I+]c1ccccc1", "diphenyliodanium"),
]


class TestHypervalentIodinePINs:
    @pytest.mark.parametrize("smiles,expected", _TARGETS)
    def test_pin_string(self, smiles, expected):
        assert _name(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", _TARGETS)
    def test_opsin_roundtrip(self, smiles, expected):
        """0-wrong: the emitted PIN parses back to the input structure."""
        name = _name(smiles)
        assert name == expected
        parsed = opsin_parse(name)
        assert parsed, f"OPSIN could not parse {name!r}"
        assert _inchikey(parsed) == _inchikey(smiles)

    def test_iodanium_not_iodonium(self):
        """ (the Blue Book): the PIN is the systematic 'iodanium', never the
        Table-7.3 retained 'iodonium'."""
        name = _name("c1ccccc1[I+]c1ccccc1")
        assert "iodanium" in name
        assert "iodonium" not in name

    def test_diaryliodanium_salt_recovered(self):
        """Bonus binary salt): fixing the iodanium cation recovers the
        diaryliodonium-triflate salt cluster for free."""
        smi = "FC(F)(F)S(=O)(=O)[O-].c1ccccc1[I+]c1ccccc1"
        assert _name(smi) == "diphenyliodanium trifluoromethanesulfonate"


class TestMonoIodineUnaffected:
    """The hypervalent guard must keep standard-valence (mono-coordinate) iodine
    OUT of the iodane regime — CH3-I stays an 'iodo' alkane."""
    @pytest.mark.parametrize("smiles,expected", [
        ("CI", "iodomethane"),
        ("c1ccccc1I", "iodobenzene"),
    ])
    def test_mono_iodo_stays_halide(self, smiles, expected):
        assert _name(smiles) == expected

    @pytest.mark.parametrize("smiles,expected", [
        # Pre-existing mixed/all-halogen iodanes must not regress.
        ("c1ccccc1I(Cl)Cl", "dichloro(phenyl)-λ3-iodane"),
        ("FI(F)(F)(F)F", "pentafluoro-λ5-iodane"),
        # Pure-organyl trialkyl-λ3-iodane (same regime as the triaryl target).
        ("CI(C)C", "trimethyl-λ3-iodane"),
    ])
    def test_existing_iodanes_hold(self, smiles, expected):
        assert _name(smiles) == expected
