"""v50 B2 — organic sulfate esters (esters of sulfuric acid),.

Sulfuric acid (H2SO4) is a dibasic mononuclear noncarbon oxoacid.
Its esters are named exactly like esters of organic acids: the
organyl group(s) are cited as separate words, in alphanumerical order when more
than one, followed by the anion name of the acid ('sulfate'); a partial (mono)
ester of the dibasic acid inserts the word 'hydrogen' for the remaining acidic
-OH, ``the Blue Book Blue Book``).

Governing PIN example, verbatim:

    CH3-O-SO2-OH methyl hydrogen sulfate (PIN):35968

so the class is:

    R-O-SO2-O-R' dialkyl sulfate (full/di ester) -> 'dimethyl sulfate'
    R-O-SO2-OH alkyl hydrogen sulfate (partial/mono) -> 'methyl hydrogen sulfate'

These are distinct from the ALREADY-handled sulfonate esters (R-SO2-O-R', an
S-C bond, 'ethyl benzenesulfonate') — a sulfate ester has NO S-C bond; the
sulfur is bonded only to oxygen. This class abstained before v50 B2.

Every assertion pairs the exact PIN string with a REAL OPSIN round-trip
(name -> OPSIN extended SMILES -> InChIKey) back to the input InChIKey, so a
string that matched but denoted a different structure would fail (0-wrong).
"""

import pytest
from rdkit import Chem

from orthonym import name_compound
from orthonym.jvm_bridge import opsin_extended_smiles, opsin_available
from orthonym.jvm_budget import jvm_slots


# (SMILES, expected PIN name)
SULFATE_ESTERS = [
    ("O=S(=O)(OC)OC", "dimethyl sulfate"),          # di-ester, identical owners
    ("O=S(=O)(O)OC", "methyl hydrogen sulfate"),    # mono-ester (BB:35968)
    ("O=S(=O)(O)Oc1ccccc1", "phenyl hydrogen sulfate"),  # aryl mono-ester
    ("O=S(=O)(O)OCC", "ethyl hydrogen sulfate"),    # ethyl mono-ester
]


def _inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    return Chem.MolToInchiKey(mol)


def _opsin_inchikey(name):
    """Parse ``name`` with OPSIN and return the InChIKey of the parsed structure,
    or None if OPSIN cannot parse it."""
    ext, served = opsin_extended_smiles(name)
    if not served or ext is None or ext == "REJECTED":
        return None
    smi = ext.split()[0]  # strip the " |$_AV:...$|" locant annotation
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(mol) if mol is not None else None


@pytest.mark.parametrize("smiles,expected", SULFATE_ESTERS)
def test_sulfate_ester_name(smiles, expected):
    assert name_compound(smiles) == expected


@pytest.mark.roundtrip
@pytest.mark.parametrize("smiles,expected", SULFATE_ESTERS)
def test_sulfate_ester_roundtrips_zero_wrong(smiles, expected):
    """0-wrong: the emitted name must OPSIN round-trip to the input's full
    InChIKey."""
    if not opsin_available():
        pytest.skip("OPSIN/JVM not available")
    name = name_compound(smiles)
    assert name == expected
    with jvm_slots(1, purpose="test-sulfate-ester-rt"):
        assert _opsin_inchikey(name) == _inchikey(smiles)


def test_sulfonate_ester_positive_control_unchanged():
    """Regression: the pre-existing sulfonate-ester path (an S-C bond) must stay
    byte-identical — the sulfate class shares its dispatch neighbourhood."""
    assert name_compound("CCOS(=O)(=O)c1ccccc1") == "ethyl benzenesulfonate"


# a performance pass review fix: an owner whose ester-oxygen carbon is a carbonyl carbon
# (an acyl group) is a mixed carboxylic/sulfuric ANHYDRIDE, not a
# sulfate ester — 's ester owners are alkyl/aryl groups only. The
# SMARTS is intentionally broad enough to still MATCH these (so the class is
# recognised), but the namer must decline the acyl owner and fall through to
# abstain rather than emit a non-PIN 'acetyl methyl sulfate'-style name.
ACYL_OWNER_ABSTAINS = [
    "CC(=O)OS(=O)(=O)OC",   # mixed diester: acetyl + methyl owners
    "CC(=O)OS(=O)(=O)O",    # mixed monoester: acetyl owner, free -OH
]


@pytest.mark.parametrize("smiles", ACYL_OWNER_ABSTAINS)
def test_sulfate_ester_acyl_owner_abstains(smiles):
    """An acyl (carbonyl-carbon) owner is out of the sulfate-ester class —
    the namer must decline it, not mis-name it as a sulfate ester."""
    assert name_compound(smiles) == "unknown organic compound"
